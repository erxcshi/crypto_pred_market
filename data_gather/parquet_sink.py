"""
ParquetDataSink — local hourly-partitioned parquet storage for all scrapers.

Drop-in third mode for the DataSink seam in config.py: identical
`submit_rows(table_name, rows_data, csv_filename=None, fieldnames=None)`
signature, so no scraper changes. Select with DATA_SINK_MODE=parquet.

Design (see CLAUDE.md / collection plan):
  - submit_rows() is a non-blocking enqueue onto a bounded queue.Queue; a
    single daemon writer thread owns all buffer state, so the asyncio event
    loop is never stalled by disk writes (or an antivirus scan of the file).
  - File-per-flush, never an appendable file: each flush writes a complete
    parquet file to `<table>/date=YYYY-MM-DD/hour=HH/part-HHMMSS-xxxxxxxx.parquet`
    via tmp file + os.replace (atomic on NTFS/POSIX). A hard kill (Windows
    terminate, reboot, power loss) loses at most one buffer window
    (FLUSH_MAX_SECONDS) and can never corrupt files already on disk.
  - Explicit per-table schemas (parquet_schemas.SCHEMAS) with value coercion:
    several APIs return numeric fields as strings; missing keys become nulls;
    unknown keys are dropped (warned once per table+key). Every file for a
    table therefore has an identical schema — no merge conflicts downstream.
  - Heartbeat: every STATUS_INTERVAL_SECONDS the writer thread atomically
    writes data/_status/<stream_name>.json with per-table counters, for
    data_gather.monitor to watch.

`submit_rows` returning True means *accepted*, not yet durable (durable within
FLUSH_MAX_SECONDS). Write errors are logged and surfaced via the status file's
`last_error`, not the return value — matching how scrapers already treat the
sink (fire-and-forget with console logging).
"""
from __future__ import annotations

import atexit
import json
import os
import queue
import threading
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

from data_gather.parquet_schemas import SCHEMAS

PROJECT_ROOT = Path(__file__).resolve().parents[1]

DEFAULT_FLUSH_MAX_ROWS = 2000
DEFAULT_FLUSH_MAX_SECONDS = 30.0
QUEUE_MAX_BATCHES = 10_000
STATUS_INTERVAL_SECONDS = 60.0
_SENTINEL = object()


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


class _TableBuffer:
    __slots__ = ("rows", "first_row_at")

    def __init__(self) -> None:
        self.rows: list[dict] = []
        self.first_row_at: float = 0.0

    def add(self, rows: list[dict]) -> None:
        if not self.rows:
            self.first_row_at = time.monotonic()
        self.rows.extend(rows)


class _WriterThread(threading.Thread):
    """Single owner of all parquet buffers, schemas, and status output."""

    def __init__(
        self,
        work_queue: queue.Queue,
        base_dir: Path,
        flush_max_rows: int,
        flush_max_seconds: float,
        compression: str,
        compression_level: int,
        stream_name: str,
    ) -> None:
        super().__init__(name="parquet-writer", daemon=True)
        self._queue = work_queue
        self._base_dir = base_dir
        self._flush_max_rows = flush_max_rows
        self._flush_max_seconds = flush_max_seconds
        self._compression = compression
        self._compression_level = compression_level
        self._stream_name = stream_name

        self._buffers: dict[str, _TableBuffer] = {}
        self._inferred_schemas: dict[str, pa.Schema] = {}
        self._warned_keys: set[tuple[str, str]] = set()
        self._warned_coercions: set[tuple[str, str]] = set()

        # status counters
        self._stats: dict[str, dict] = {}   # table -> {rows_total, files_written, last_row_written_at}
        self._last_error: str = ""
        self._last_status_write = 0.0

    # ------------------------------------------------------------------
    # Main loop
    # ------------------------------------------------------------------

    def run(self) -> None:
        while True:
            try:
                item = self._queue.get(timeout=1.0)
            except queue.Empty:
                item = None

            if item is _SENTINEL:
                self._flush_all()
                self._write_status(force=True)
                return

            if item is not None:
                table_name, rows = item
                self._buffers.setdefault(table_name, _TableBuffer()).add(rows)

            now = time.monotonic()
            for table_name, buf in list(self._buffers.items()):
                if not buf.rows:
                    continue
                if (len(buf.rows) >= self._flush_max_rows
                        or now - buf.first_row_at >= self._flush_max_seconds):
                    self._flush(table_name)

            if now - self._last_status_write >= STATUS_INTERVAL_SECONDS:
                self._write_status()

    def _flush_all(self) -> None:
        for table_name in list(self._buffers):
            self._flush(table_name)

    def _flush(self, table_name: str) -> None:
        buf = self._buffers.get(table_name)
        if buf is None or not buf.rows:
            return
        rows = buf.rows
        buf.rows = []
        try:
            self._write_file(table_name, rows)
            stats = self._stats.setdefault(
                table_name, {"rows_total": 0, "files_written": 0, "last_row_written_at": ""}
            )
            stats["rows_total"] += len(rows)
            stats["files_written"] += 1
            stats["last_row_written_at"] = _utcnow().isoformat()
        except Exception as e:  # noqa: BLE001 — collection must survive any write error
            self._last_error = f"{_utcnow().isoformat()} {table_name}: {type(e).__name__}: {e}"
            print(f"parquet_sink: WRITE FAILED for {table_name}: {type(e).__name__}: {e}")
            # Put rows back (front) so they retry next tick; cap to avoid OOM
            # if the disk stays broken.
            buf.rows = (rows + buf.rows)[: self._flush_max_rows * 10]
            buf.first_row_at = time.monotonic()

    # ------------------------------------------------------------------
    # Writing
    # ------------------------------------------------------------------

    def _target_dir(self, table_name: str, now: datetime) -> Path:
        return (
            self._base_dir
            / table_name
            / f"date={now.strftime('%Y-%m-%d')}"
            / f"hour={now.strftime('%H')}"
        )

    def _write_file(self, table_name: str, rows: list[dict]) -> Path:
        schema = self._resolve_schema(table_name, rows)
        coerced = [self._coerce_row(table_name, row, schema) for row in rows]
        table = pa.Table.from_pylist(coerced, schema=schema)

        now = _utcnow()
        target_dir = self._target_dir(table_name, now)
        target_dir.mkdir(parents=True, exist_ok=True)
        fname = f"part-{now.strftime('%H%M%S')}-{uuid.uuid4().hex[:8]}.parquet"
        final_path = target_dir / fname
        tmp_path = target_dir / (fname + ".tmp")

        pq.write_table(
            table,
            tmp_path,
            compression=self._compression,
            compression_level=self._compression_level,
        )
        os.replace(tmp_path, final_path)
        return final_path

    # ------------------------------------------------------------------
    # Schema resolution + coercion
    # ------------------------------------------------------------------

    def _resolve_schema(self, table_name: str, rows: list[dict]) -> pa.Schema:
        schema = SCHEMAS.get(table_name) or self._inferred_schemas.get(table_name)
        if schema is not None:
            return schema

        # Infer once from the first batch and cache for the process lifetime.
        fields: dict[str, pa.DataType] = {}
        for row in rows:
            for key, val in row.items():
                if key in fields and fields[key] != pa.string():
                    continue
                if isinstance(val, bool):
                    fields[key] = pa.string()
                elif isinstance(val, int):
                    fields.setdefault(key, pa.int64())
                elif isinstance(val, float):
                    fields[key] = pa.float64()
                elif val is not None:
                    fields[key] = pa.string()
                else:
                    fields.setdefault(key, pa.string())
        schema = pa.schema(list(fields.items()))
        self._inferred_schemas[table_name] = schema
        print(f"parquet_sink: inferred schema for unknown table {table_name!r} "
              f"({len(schema)} cols) — add it to parquet_schemas.py for long runs")
        return schema

    def _coerce_row(self, table_name: str, row: dict, schema: pa.Schema) -> dict:
        out: dict = {}
        names = set(schema.names)
        for key in row.keys() - names:
            if (table_name, key) not in self._warned_keys:
                self._warned_keys.add((table_name, key))
                print(f"parquet_sink: dropping unknown key {key!r} for table {table_name!r}")
        for field in schema:
            val = row.get(field.name)
            if val is None:
                out[field.name] = None
                continue
            if val == "" and not pa.types.is_string(field.type):
                out[field.name] = None  # empty string in a numeric field
                continue
            try:
                if pa.types.is_floating(field.type):
                    out[field.name] = float(val)
                elif pa.types.is_integer(field.type):
                    out[field.name] = int(float(val))
                elif isinstance(val, bool):
                    out[field.name] = str(val).lower()
                else:
                    out[field.name] = str(val)
            except (TypeError, ValueError):
                out[field.name] = None
                if (table_name, field.name) not in self._warned_coercions:
                    self._warned_coercions.add((table_name, field.name))
                    print(
                        f"parquet_sink: coercion failed for {table_name}.{field.name} "
                        f"value={val!r} ({type(val).__name__}) -> null"
                    )
        return out

    # ------------------------------------------------------------------
    # Status heartbeat
    # ------------------------------------------------------------------

    def _write_status(self, force: bool = False) -> None:
        self._last_status_write = time.monotonic()
        status_dir = self._base_dir / "_status"
        try:
            status_dir.mkdir(parents=True, exist_ok=True)
            payload = {
                "stream_name": self._stream_name,
                "written_at": _utcnow().isoformat(),
                "pid": os.getpid(),
                "queue_depth": self._queue.qsize(),
                "last_error": self._last_error,
                "tables": self._stats,
            }
            tmp = status_dir / f"{self._stream_name}.json.tmp"
            final = status_dir / f"{self._stream_name}.json"
            tmp.write_text(json.dumps(payload, indent=1))
            os.replace(tmp, final)
        except Exception as e:  # noqa: BLE001
            print(f"parquet_sink: status write failed: {type(e).__name__}: {e}")


class ParquetDataSink:
    """DataSink-compatible parquet writer. See module docstring."""

    def __init__(
        self,
        base_dir: Path | str | None = None,
        *,
        flush_max_rows: int = DEFAULT_FLUSH_MAX_ROWS,
        flush_max_seconds: float = DEFAULT_FLUSH_MAX_SECONDS,
        compression: str = "zstd",
        compression_level: int = 3,
        stream_name: str | None = None,
    ) -> None:
        base = Path(base_dir or os.getenv("PARQUET_DATA_DIR") or (PROJECT_ROOT / "data"))
        name = stream_name or os.getenv("SCRAPER_NAME") or f"pid{os.getpid()}"

        self.mode = "parquet"  # parity with DataSink.mode
        self._queue: queue.Queue = queue.Queue(maxsize=QUEUE_MAX_BATCHES)
        self._writer = _WriterThread(
            self._queue, base, flush_max_rows, flush_max_seconds,
            compression, compression_level, name,
        )
        self._closed = False
        self._writer.start()
        atexit.register(self.close)

    async def submit_rows(self, table_name, rows_data, csv_filename=None, fieldnames=None) -> bool:
        """Signature-compatible with DataSink.submit_rows. csv_filename and
        fieldnames are accepted and ignored (schema comes from parquet_schemas)."""
        if not rows_data:
            return True
        try:
            self._queue.put_nowait((table_name, list(rows_data)))
            return True
        except queue.Full:
            print(f"parquet_sink: queue full — dropping {len(rows_data)} rows for {table_name}")
            return False

    def close(self, timeout: float = 15.0) -> None:
        """Drain the queue and flush all buffers. Registered via atexit; safe
        to call multiple times."""
        if self._closed:
            return
        self._closed = True
        try:
            self._queue.put(_SENTINEL, timeout=2.0)
        except queue.Full:
            pass
        self._writer.join(timeout=timeout)


def create_parquet_sink(**kwargs) -> ParquetDataSink:
    return ParquetDataSink(**kwargs)

"""
Daily parquet compaction + retention for the collector's data/ tree.

The sink writes many small part files (one per flush). This CLI merges each
CLOSED date (strictly before today, UTC) into a single
`data/{table}/date=D/compacted-D.parquet`, verifies row counts, then deletes
the parts and empty hour dirs. Idempotent: stray parts written after a
previous compaction are merged into the existing compacted file on re-run.

Usage:
    python -m data_gather.compact                    # compact all closed dates
    python -m data_gather.compact --date 2026-07-08  # one date
    python -m data_gather.compact --dry-run
    python -m data_gather.compact --retention-days 60   # delete local dates older
                                                        # than N days (only use once
                                                        # an off-box copy exists)

Schedule daily (00:15 UTC): Task Scheduler on Windows, systemd timer on Linux
(see deploy/). Off-box sync (optional): after compaction,
    rclone copy data remote:bucket/market-data --include "*/date=*/compacted-*.parquet"
"""
from __future__ import annotations

import argparse
import os
import shutil
from datetime import datetime, timezone
from pathlib import Path

import pyarrow as pa
import pyarrow.dataset as ds
import pyarrow.parquet as pq

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA_DIR = Path(os.getenv("PARQUET_DATA_DIR") or (PROJECT_ROOT / "data"))
COMPRESSION = "zstd"
COMPRESSION_LEVEL = 6          # higher than the live sink — write-once files
# Above this row count, stream part files into the output instead of loading
# the whole date into memory (a full day of coinbase_l2 / polymarket events is
# 50-90M rows — to_table() OOM-killed the 07-09 run on a 16GB machine).
# Streaming skips the global curr_time sort, but parts are flush-ordered so
# the output is already ~time-sorted; sort at read time if exact order matters.
ROWS_IN_MEMORY_LIMIT = 5_000_000
STREAM_ROW_GROUP_ROWS = 2_000_000


def _today_str() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%d")


def find_compactable(data_dir: Path, only_date: str | None = None) -> list[Path]:
    """date= dirs strictly before today (or the specific date) that still
    contain part files."""
    out = []
    today = _today_str()
    for table_dir in sorted(p for p in data_dir.iterdir() if p.is_dir() and p.name != "_status"):
        for date_dir in sorted(table_dir.glob("date=*")):
            date_str = date_dir.name.split("=", 1)[1]
            if only_date is not None and date_str != only_date:
                continue
            if only_date is None and date_str >= today:
                continue
            if any(date_dir.rglob("part-*.parquet")):
                out.append(date_dir)
    return out


def _write_streamed(sources: list[Path], tmp: Path) -> int:
    """Concatenate sources into tmp one chunk at a time (bounded memory).
    Returns rows written."""
    schema = pq.read_schema(sources[0])
    written = 0
    pending: list = []
    pending_rows = 0
    with pq.ParquetWriter(tmp, schema, compression=COMPRESSION,
                          compression_level=COMPRESSION_LEVEL) as writer:
        def flush() -> None:
            nonlocal pending, pending_rows, written
            if pending:
                chunk = pa.concat_tables(pending).combine_chunks()
                writer.write_table(chunk)
                written += chunk.num_rows
                pending, pending_rows = [], 0

        for src in sources:
            # iter_batches keeps even a large pre-existing compacted file
            # from being materialized whole.
            for batch in pq.ParquetFile(src).iter_batches():
                pending.append(pa.Table.from_batches([batch]).cast(schema))
                pending_rows += batch.num_rows
                if pending_rows >= STREAM_ROW_GROUP_ROWS:
                    flush()
        flush()
    return written


def compact_date_dir(date_dir: Path, dry_run: bool = False) -> tuple[int, int]:
    """Merge all parquet under date_dir into compacted-{date}.parquet.
    Returns (rows, files_merged)."""
    date_str = date_dir.name.split("=", 1)[1]
    parts = sorted(date_dir.rglob("part-*.parquet"))
    existing = date_dir / f"compacted-{date_str}.parquet"
    # existing first: it holds the earliest rows (a prior partial compaction),
    # keeping the streamed output ~time-ordered.
    sources = ([existing] if existing.exists() else []) + parts
    if not parts:
        return (0, 0)

    expected = sum(pq.read_metadata(p).num_rows for p in sources)

    if dry_run:
        mode = "stream" if expected > ROWS_IN_MEMORY_LIMIT else "in-memory"
        print(f"[dry-run] {date_dir}: would compact {len(sources)} files, "
              f"{expected} rows ({mode})")
        return (expected, len(sources))

    tmp = date_dir / f"compacted-{date_str}.parquet.tmp"
    if expected > ROWS_IN_MEMORY_LIMIT:
        written = _write_streamed(sources, tmp)
    else:
        dataset = ds.dataset([str(p) for p in sources], format="parquet")
        table = dataset.to_table()
        # Stable ordering for downstream scans.
        if "curr_time" in table.column_names:
            table = table.sort_by("curr_time")
        pq.write_table(table, tmp, compression=COMPRESSION,
                       compression_level=COMPRESSION_LEVEL)
        written = table.num_rows

    on_disk = pq.read_metadata(tmp).num_rows
    if not (written == expected == on_disk):
        tmp.unlink()
        raise RuntimeError(
            f"{date_dir}: row count mismatch (expected {expected}, wrote "
            f"{written}, on disk {on_disk}) — aborting, nothing deleted"
        )
    os.replace(tmp, existing)

    for p in parts:
        p.unlink()
    # Remove now-empty hour dirs and stray .tmp orphans.
    for orphan in date_dir.rglob("*.tmp"):
        orphan.unlink()
    for hour_dir in sorted(date_dir.glob("hour=*")):
        if not any(hour_dir.iterdir()):
            hour_dir.rmdir()

    print(f"{date_dir}: {len(parts)} parts -> compacted-{date_str}.parquet "
          f"({written} rows)", flush=True)
    return (written, len(parts))


def apply_retention(data_dir: Path, keep_days: int, dry_run: bool = False) -> None:
    """Delete local date dirs older than keep_days. Only run this once an
    off-box copy (rclone/external drive) is in place."""
    cutoff = datetime.now(timezone.utc).timestamp() - keep_days * 86400
    for table_dir in sorted(p for p in data_dir.iterdir() if p.is_dir() and p.name != "_status"):
        for date_dir in sorted(table_dir.glob("date=*")):
            date_str = date_dir.name.split("=", 1)[1]
            try:
                dir_ts = datetime.strptime(date_str, "%Y-%m-%d").replace(
                    tzinfo=timezone.utc).timestamp()
            except ValueError:
                continue
            if dir_ts < cutoff:
                if dry_run:
                    print(f"[dry-run] would delete {date_dir}")
                else:
                    shutil.rmtree(date_dir)
                    print(f"retention: deleted {date_dir}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    parser.add_argument("--data-dir", default=str(DEFAULT_DATA_DIR))
    parser.add_argument("--date", help="compact only this YYYY-MM-DD (even if today)")
    parser.add_argument("--retention-days", type=int, default=0,
                        help="delete local date dirs older than N days (0 = never)")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    if not data_dir.exists():
        print(f"nothing to do: {data_dir} does not exist")
        return

    total_rows = total_files = 0
    for date_dir in find_compactable(data_dir, args.date):
        try:
            rows, files = compact_date_dir(date_dir, dry_run=args.dry_run)
            total_rows += rows
            total_files += files
        except Exception as e:  # noqa: BLE001 — keep compacting other dates
            print(f"COMPACTION FAILED for {date_dir}: {type(e).__name__}: {e}")

    print(f"compacted {total_files} part files ({total_rows} rows)")

    if args.retention_days > 0:
        apply_retention(data_dir, args.retention_days, dry_run=args.dry_run)


if __name__ == "__main__":
    main()

"""
Shared WebSocket scraper skeleton.

Every WS scraper repeats the same shape: connect with backoff -> subscribe ->
dispatch messages into row buffers -> flush to the data sink on size/time ->
final-flush on cancel -> reconnect on failure. Subclass WsScraper and
implement the four abstract hooks; everything else is provided.

    class MyScraper(WsScraper):
        name = "my_ws"
        async def get_ws_url(self): ...
        def make_buffers(self): ...           # fresh per connection
        async def on_connect(self, ws): ...   # send subscribe message(s)
        async def handle_message(self, msg): ...  # route into self.buffers

Raise ResubscribeSignal from handle_message to force a clean reconnect
(sequence gap, subscription drop). Long-running side loops (ticker discovery,
resubscription refresh) go in background_tasks().
"""
from __future__ import annotations

import asyncio
import json
import random
import time
from typing import Awaitable, Callable, Optional

import websockets

from config import create_data_sink

DEFAULT_BATCH_SIZE = 100
DEFAULT_FLUSH_INTERVAL_SECONDS = 5.0
RECONNECT_MAX_BACKOFF = 60.0


class ResubscribeSignal(Exception):
    """Raise inside handle_message to tear down and reconnect cleanly."""


class StreamBuffer:
    """Row buffer bound to one output table. (Extracted from the _Buffer
    pattern duplicated across coinbase_ws.py / kalshi_ws.py.)"""

    def __init__(
        self,
        table: str,
        fields: list[str],
        csv_filename: Optional[str] = None,
        batch_size: int = DEFAULT_BATCH_SIZE,
        flush_interval: float = DEFAULT_FLUSH_INTERVAL_SECONDS,
    ) -> None:
        self.table = table
        self.fields = fields
        self.csv_filename = csv_filename or f"{table}_test.csv"
        self.batch_size = batch_size
        self.flush_interval = flush_interval
        self.rows: list[dict] = []
        self.last_flush = time.monotonic()

    def add(self, row: dict) -> None:
        self.rows.append(row)

    def extend(self, rows: list[dict]) -> None:
        self.rows.extend(rows)

    def should_flush(self) -> bool:
        if not self.rows:
            return False
        return (len(self.rows) >= self.batch_size
                or time.monotonic() - self.last_flush >= self.flush_interval)

    async def flush(self, data_sink) -> None:
        if not self.rows:
            return
        ok = await data_sink.submit_rows(
            self.table, self.rows,
            csv_filename=self.csv_filename, fieldnames=self.fields,
        )
        if ok:
            self.rows.clear()
            self.last_flush = time.monotonic()


class WsScraper:
    """Reconnecting WebSocket scraper base. Subclass and implement hooks."""

    name: str = "ws_scraper"
    ping_interval: int = 20
    ping_timeout: int = 20
    max_backoff: float = RECONNECT_MAX_BACKOFF

    def __init__(self, data_sink=None) -> None:
        self.data_sink = data_sink or create_data_sink()
        self.buffers: list[StreamBuffer] = []
        self.ws = None  # current connection, set by run() before on_connect

    # ------------------------------------------------------------------
    # Hooks to implement
    # ------------------------------------------------------------------

    async def get_ws_url(self) -> str:
        raise NotImplementedError

    async def connect_kwargs(self) -> dict:
        """Extra kwargs for websockets.connect (e.g. auth headers)."""
        return {}

    def make_buffers(self) -> list[StreamBuffer]:
        """Fresh buffers per connection."""
        raise NotImplementedError

    async def on_connect(self, ws) -> None:
        """Send subscribe message(s)."""
        raise NotImplementedError

    async def handle_message(self, msg: dict) -> None:
        """Route one decoded message into self.buffers. May raise
        ResubscribeSignal to force a clean reconnect."""
        raise NotImplementedError

    def background_tasks(self, ws) -> list[Awaitable]:
        """Optional coroutines run alongside the receive loop (discovery
        loops etc.). Cancelled automatically on disconnect."""
        return []

    def decode(self, raw) -> Optional[dict]:
        """Decode a raw frame; None skips it. Override for non-JSON frames."""
        try:
            return json.loads(raw)
        except Exception:
            return None

    # ------------------------------------------------------------------
    # Engine
    # ------------------------------------------------------------------

    async def _flush_loop(self) -> None:
        while True:
            await asyncio.sleep(1)
            for buf in self.buffers:
                if buf.should_flush():
                    await buf.flush(self.data_sink)

    async def _flush_all(self) -> None:
        for buf in self.buffers:
            try:
                await buf.flush(self.data_sink)
            except Exception as e:  # noqa: BLE001 — best-effort final flush
                print(f"{self.name}: final flush failed for {buf.table}: {e}")

    async def run(self) -> None:
        backoff = 1.0
        while True:
            side_tasks: list[asyncio.Task] = []
            try:
                url = await self.get_ws_url()
                kwargs = await self.connect_kwargs()
                print(f"{self.name}: connecting to {url}")
                async with websockets.connect(
                    url,
                    ping_interval=self.ping_interval,
                    ping_timeout=self.ping_timeout,
                    max_size=None,
                    **kwargs,
                ) as ws:
                    self.ws = ws
                    self.buffers = self.make_buffers()
                    await self.on_connect(ws)
                    backoff = 1.0

                    side_tasks = [asyncio.create_task(self._flush_loop())]
                    side_tasks += [asyncio.create_task(c) for c in self.background_tasks(ws)]

                    async for raw in ws:
                        msg = self.decode(raw)
                        if msg is None:
                            continue
                        await self.handle_message(msg)
                        for buf in self.buffers:
                            if buf.should_flush():
                                await buf.flush(self.data_sink)
            except asyncio.CancelledError:
                await self._flush_all()
                raise
            except ResubscribeSignal as e:
                print(f"{self.name}: resubscribe requested: {e}")
                backoff = 1.0
            except Exception as e:  # noqa: BLE001
                print(f"{self.name}: connection failed: {type(e).__name__}: {e}")
            finally:
                for t in side_tasks:
                    t.cancel()
                if side_tasks:
                    await asyncio.gather(*side_tasks, return_exceptions=True)
                await self._flush_all()

            jitter = random.uniform(0, 0.5 * backoff)
            await asyncio.sleep(backoff + jitter)
            backoff = min(self.max_backoff, backoff * 2)


def run_scraper_main(scraper_factory: Callable[[], WsScraper]) -> None:
    """Standard __main__ entrypoint for a WsScraper module."""
    async def _main() -> None:
        await scraper_factory().run()

    try:
        asyncio.run(_main())
    except KeyboardInterrupt:
        print("stopped by user")

"""
Equities scraper: Alpaca IEX WebSocket (SPY/QQQ trades+quotes, free tier) plus
a free delayed VIX poll.

Alpaca needs ALPACA_API_KEY / ALPACA_API_SECRET in .env (free account at
alpaca.markets). Without keys the WS part idles with a notice (no crash-loop)
while the VIX poll still runs.

Rows -> alpaca_trades, alpaca_quotes, vix_quotes.

Run: DATA_SINK_MODE=parquet python -m data_gather.scrapers.alpaca_ws
"""
from __future__ import annotations

import asyncio
import json
import os
from datetime import datetime, timezone

import aiohttp

from config import create_data_sink
from data_gather.scrapers.ws_base import StreamBuffer, WsScraper

WS_URL = "wss://stream.data.alpaca.markets/v2/iex"
SYMBOLS = ["SPY", "QQQ"]
VIX_POLL_SECONDS = 60
VIX_URL = "https://query1.finance.yahoo.com/v8/finance/chart/%5EVIX"

TRADE_FIELDS = ["curr_time", "symbol", "exchange_time", "price", "size", "exchange"]
QUOTE_FIELDS = ["curr_time", "symbol", "exchange_time",
                "bid_price", "bid_size", "ask_price", "ask_size"]
VIX_FIELDS = ["curr_time", "symbol", "price", "quote_time"]


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


class AlpacaWsScraper(WsScraper):
    name = "alpaca_ws"

    def __init__(self, data_sink=None) -> None:
        super().__init__(data_sink)
        self._key = os.getenv("ALPACA_API_KEY", "")
        self._secret = os.getenv("ALPACA_API_SECRET", "")

    async def get_ws_url(self) -> str:
        return WS_URL

    def make_buffers(self) -> list[StreamBuffer]:
        self._trades = StreamBuffer("alpaca_trades", TRADE_FIELDS, flush_interval=10.0)
        self._quotes = StreamBuffer("alpaca_quotes", QUOTE_FIELDS, flush_interval=10.0)
        return [self._trades, self._quotes]

    async def on_connect(self, ws) -> None:
        await ws.send(json.dumps({"action": "auth",
                                  "key": self._key, "secret": self._secret}))
        await ws.send(json.dumps({"action": "subscribe",
                                  "trades": SYMBOLS, "quotes": SYMBOLS}))
        print(f"{self.name}: auth + subscribe sent for {SYMBOLS}")

    async def handle_message(self, msg) -> None:
        events = msg if isinstance(msg, list) else [msg]
        curr_time = _now_iso()
        for ev in events:
            if not isinstance(ev, dict):
                continue
            t = ev.get("T")
            if t == "t":
                self._trades.add({
                    "curr_time": curr_time,
                    "symbol": ev.get("S", ""),
                    "exchange_time": ev.get("t", ""),
                    "price": ev.get("p"),
                    "size": ev.get("s"),
                    "exchange": ev.get("x", ""),
                })
            elif t == "q":
                self._quotes.add({
                    "curr_time": curr_time,
                    "symbol": ev.get("S", ""),
                    "exchange_time": ev.get("t", ""),
                    "bid_price": ev.get("bp"),
                    "bid_size": ev.get("bs"),
                    "ask_price": ev.get("ap"),
                    "ask_size": ev.get("as"),
                })
            elif t == "error":
                print(f"{self.name}: error: {ev}")


async def vix_poll_loop(data_sink) -> None:
    buf = StreamBuffer("vix_quotes", VIX_FIELDS, flush_interval=120.0)
    headers = {"User-Agent": "Mozilla/5.0"}
    async with aiohttp.ClientSession(headers=headers) as session:
        while True:
            try:
                async with session.get(VIX_URL, params={"interval": "1m", "range": "1d"},
                                       timeout=aiohttp.ClientTimeout(total=10)) as resp:
                    resp.raise_for_status()
                    data = await resp.json()
                meta = ((data.get("chart") or {}).get("result") or [{}])[0].get("meta") or {}
                price = meta.get("regularMarketPrice")
                qt = meta.get("regularMarketTime")
                if price is not None:
                    buf.add({
                        "curr_time": _now_iso(),
                        "symbol": "VIX",
                        "price": price,
                        "quote_time": str(qt or ""),
                    })
                    # One row per minute — flush immediately, no batching value.
                    await buf.flush(data_sink)
            except Exception as e:  # noqa: BLE001
                print(f"vix_poll: failed: {type(e).__name__}: {e}")
            await asyncio.sleep(VIX_POLL_SECONDS)


async def main() -> None:
    data_sink = create_data_sink()
    tasks = [asyncio.create_task(vix_poll_loop(data_sink))]

    if os.getenv("ALPACA_API_KEY") and os.getenv("ALPACA_API_SECRET"):
        tasks.append(asyncio.create_task(AlpacaWsScraper(data_sink).run()))
    else:
        print("alpaca_ws: ALPACA_API_KEY/ALPACA_API_SECRET not set — "
              "running VIX poll only (add free keys from alpaca.markets to enable SPY/QQQ)")

    await asyncio.gather(*tasks)


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("stopped by user")

"""
Kraken public WebSocket (v2) trade scraper — BRTI constituent venue.

Rows -> kraken_trades. Kraken v2 sends a snapshot of recent trades on
subscribe (type='snapshot') then updates; both are stored (trade_id dedupes
downstream).

Run: DATA_SINK_MODE=parquet python -m data_gather.scrapers.kraken_ws
"""
from __future__ import annotations

import json
from datetime import datetime, timezone

from data_gather.scrapers.ws_base import StreamBuffer, WsScraper, run_scraper_main

WS_URL = "wss://ws.kraken.com/v2"
SYMBOLS = ["BTC/USD", "ETH/USD", "XRP/USD", "SOL/USD"]

FIELDS = ["curr_time", "pair", "exchange_time", "price", "size", "side", "order_type"]


class KrakenWsScraper(WsScraper):
    name = "kraken_ws"

    async def get_ws_url(self) -> str:
        return WS_URL

    def make_buffers(self) -> list[StreamBuffer]:
        self._trades = StreamBuffer("kraken_trades", FIELDS)
        return [self._trades]

    async def on_connect(self, ws) -> None:
        await ws.send(json.dumps({
            "method": "subscribe",
            "params": {"channel": "trade", "symbol": SYMBOLS},
        }))
        print(f"{self.name}: subscribed trades for {SYMBOLS}")

    async def handle_message(self, msg: dict) -> None:
        if msg.get("channel") != "trade":
            if msg.get("method") == "subscribe" and not msg.get("success", True):
                print(f"{self.name}: subscribe failed: {msg}")
            return
        curr_time = datetime.now(timezone.utc).isoformat()
        for t in msg.get("data") or []:
            self._trades.add({
                "curr_time": curr_time,
                "pair": t.get("symbol", ""),
                "exchange_time": t.get("timestamp", ""),
                "price": t.get("price"),
                "size": t.get("qty"),
                "side": t.get("side", ""),
                "order_type": t.get("ord_type", ""),
            })


if __name__ == "__main__":
    run_scraper_main(KrakenWsScraper)

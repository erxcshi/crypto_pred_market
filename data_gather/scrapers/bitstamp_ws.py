"""
Bitstamp public WebSocket trade scraper — BRTI constituent venue.

Rows -> bitstamp_trades. Bitstamp side convention: type 0 = buy, 1 = sell.

Run: DATA_SINK_MODE=parquet python -m data_gather.scrapers.bitstamp_ws
"""
from __future__ import annotations

import json
from datetime import datetime, timezone

from data_gather.scrapers.ws_base import StreamBuffer, WsScraper, run_scraper_main

WS_URL = "wss://ws.bitstamp.net"
PAIRS = ["btcusd", "ethusd", "xrpusd", "solusd"]

FIELDS = ["curr_time", "pair", "trade_id", "exchange_ts", "price", "size", "side"]


class BitstampWsScraper(WsScraper):
    name = "bitstamp_ws"

    async def get_ws_url(self) -> str:
        return WS_URL

    def make_buffers(self) -> list[StreamBuffer]:
        self._trades = StreamBuffer("bitstamp_trades", FIELDS)
        return [self._trades]

    async def on_connect(self, ws) -> None:
        for pair in PAIRS:
            await ws.send(json.dumps({
                "event": "bts:subscribe",
                "data": {"channel": f"live_trades_{pair}"},
            }))
        print(f"{self.name}: subscribed live_trades for {PAIRS}")

    async def handle_message(self, msg: dict) -> None:
        event = msg.get("event", "")
        if event == "bts:request_reconnect":
            # Bitstamp asks clients to reconnect before maintenance.
            raise ConnectionResetError("bitstamp requested reconnect")
        if event != "trade":
            return
        data = msg.get("data") or {}
        channel = msg.get("channel", "")           # live_trades_btcusd
        pair = channel.rsplit("_", 1)[-1]
        self._trades.add({
            "curr_time": datetime.now(timezone.utc).isoformat(),
            "pair": pair,
            "trade_id": data.get("id"),
            "exchange_ts": data.get("microtimestamp") or data.get("timestamp"),
            "price": data.get("price"),
            "size": data.get("amount"),
            "side": "buy" if data.get("type") == 0 else "sell",
        })


if __name__ == "__main__":
    run_scraper_main(BitstampWsScraper)

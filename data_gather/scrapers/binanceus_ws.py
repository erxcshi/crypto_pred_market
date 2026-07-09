"""
Binance.US public WebSocket scraper — trades + top-of-book quotes.

Why Binance: Polymarket's 15m up/down markets resolve off Binance prices, and
Binance leads global crypto price discovery (lead-lag alpha vs Coinbase).
Binance.US is used because binance.com blocks US IPs; the .US price tracks
global via arbitrage. If collecting from a non-US VPS, point WS_URL at
wss://stream.binance.com:9443 for the deeper global feed.

Rows -> binanceus_trades (trade stream), binanceus_quotes (bookTicker).

Run: DATA_SINK_MODE=parquet python -m data_gather.scrapers.binanceus_ws
"""
from __future__ import annotations

from datetime import datetime, timezone

from data_gather.scrapers.ws_base import StreamBuffer, WsScraper, run_scraper_main

SYMBOLS = ["btcusdt", "ethusdt", "xrpusdt", "solusdt"]
STREAMS = [f"{s}@trade" for s in SYMBOLS] + [f"{s}@bookTicker" for s in SYMBOLS]
WS_URL = "wss://stream.binance.us:9443/stream?streams=" + "/".join(STREAMS)

TRADE_FIELDS = ["curr_time", "symbol", "trade_id", "exchange_ts_ms",
                "price", "size", "is_buyer_maker"]
QUOTE_FIELDS = ["curr_time", "symbol", "update_id",
                "best_bid", "best_bid_size", "best_ask", "best_ask_size"]


class BinanceUsWsScraper(WsScraper):
    name = "binanceus_ws"

    async def get_ws_url(self) -> str:
        return WS_URL

    def make_buffers(self) -> list[StreamBuffer]:
        self._trades = StreamBuffer("binanceus_trades", TRADE_FIELDS)
        self._quotes = StreamBuffer("binanceus_quotes", QUOTE_FIELDS)
        return [self._trades, self._quotes]

    async def on_connect(self, ws) -> None:
        # Subscription is in the URL for combined streams.
        print(f"{self.name}: connected combined stream for {SYMBOLS}")

    async def handle_message(self, msg: dict) -> None:
        stream = msg.get("stream", "")
        data = msg.get("data") or {}
        curr_time = datetime.now(timezone.utc).isoformat()

        if stream.endswith("@trade"):
            self._trades.add({
                "curr_time": curr_time,
                "symbol": data.get("s", ""),
                "trade_id": data.get("t"),
                "exchange_ts_ms": data.get("T"),
                "price": data.get("p"),
                "size": data.get("q"),
                "is_buyer_maker": str(data.get("m", "")).lower(),
            })
        elif stream.endswith("@bookTicker"):
            self._quotes.add({
                "curr_time": curr_time,
                "symbol": data.get("s", ""),
                "update_id": data.get("u"),
                "best_bid": data.get("b"),
                "best_bid_size": data.get("B"),
                "best_ask": data.get("a"),
                "best_ask_size": data.get("A"),
            })


if __name__ == "__main__":
    run_scraper_main(BinanceUsWsScraper)

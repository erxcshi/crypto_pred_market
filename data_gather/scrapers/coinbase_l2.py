"""
Coinbase full order-book depth via the unauthenticated `level2_batch` channel
(50ms batched l2update messages) for BTC-USD and ETH-USD.

Rows -> coinbase_l2:
  event_type='snapshot'  one row per level (top SNAPSHOT_DEPTH per side) on
                         (re)subscribe and every RESNAPSHOT_INTERVAL via an
                         in-place unsubscribe/subscribe (server re-sends the
                         snapshot without dropping the connection)
  event_type='l2update'  one row per changed level; `size` is the new ABSOLUTE
                         size at that price (0 = level removed)

Book replay: start from the last snapshot, apply l2updates in order.

Run: DATA_SINK_MODE=parquet python -m data_gather.scrapers.coinbase_l2
"""
from __future__ import annotations

import asyncio
import json
from datetime import datetime, timezone

from data_gather.scrapers.ws_base import StreamBuffer, WsScraper, run_scraper_main

WS_URL = "wss://ws-feed.exchange.coinbase.com"
PRODUCT_IDS = ["BTC-USD", "ETH-USD"]
SNAPSHOT_DEPTH = 500                 # levels per side kept from full snapshots
RESNAPSHOT_INTERVAL_SECONDS = 30 * 60

L2_FIELDS = ["curr_time", "product_id", "event_type", "exchange_time",
             "side", "price", "size"]


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _f(x):
    try:
        return float(x)
    except (TypeError, ValueError):
        return None


class CoinbaseL2Scraper(WsScraper):
    name = "coinbase_l2"

    async def get_ws_url(self) -> str:
        return WS_URL

    def make_buffers(self) -> list[StreamBuffer]:
        self._l2 = StreamBuffer("coinbase_l2", L2_FIELDS, batch_size=500)
        return [self._l2]

    async def _subscribe(self, ws) -> None:
        await ws.send(json.dumps({
            "type": "subscribe",
            "product_ids": PRODUCT_IDS,
            "channels": ["level2_batch"],
        }))

    async def on_connect(self, ws) -> None:
        await self._subscribe(ws)
        print(f"{self.name}: subscribed level2_batch for {PRODUCT_IDS}")

    async def _resnapshot_loop(self, ws) -> None:
        """Periodic in-place resubscribe -> fresh snapshot for replay recovery."""
        while True:
            await asyncio.sleep(RESNAPSHOT_INTERVAL_SECONDS)
            try:
                await ws.send(json.dumps({
                    "type": "unsubscribe",
                    "product_ids": PRODUCT_IDS,
                    "channels": ["level2_batch"],
                }))
                await self._subscribe(ws)
                print(f"{self.name}: periodic resubscribe (fresh snapshots)")
            except Exception as e:  # noqa: BLE001
                print(f"{self.name}: resnapshot failed: {type(e).__name__}: {e}")

    def background_tasks(self, ws) -> list:
        return [self._resnapshot_loop(ws)]

    async def handle_message(self, msg: dict) -> None:
        msg_type = msg.get("type")
        curr_time = _now_iso()
        product = msg.get("product_id", "")

        if msg_type == "snapshot":
            for side_key, side in (("bids", "buy"), ("asks", "sell")):
                for level in (msg.get(side_key) or [])[:SNAPSHOT_DEPTH]:
                    self._l2.add({
                        "curr_time": curr_time,
                        "product_id": product,
                        "event_type": "snapshot",
                        "exchange_time": msg.get("time", ""),
                        "side": side,
                        "price": _f(level[0]),
                        "size": _f(level[1]),
                    })
            print(f"{self.name}: snapshot {product} "
                  f"({len(msg.get('bids') or [])} bids / {len(msg.get('asks') or [])} asks, "
                  f"kept top {SNAPSHOT_DEPTH}/side)")
        elif msg_type == "l2update":
            for change in msg.get("changes") or []:
                if len(change) < 3:
                    continue
                self._l2.add({
                    "curr_time": curr_time,
                    "product_id": product,
                    "event_type": "l2update",
                    "exchange_time": msg.get("time", ""),
                    "side": change[0],
                    "price": _f(change[1]),
                    "size": _f(change[2]),
                })
        elif msg_type == "error":
            print(f"{self.name}: error: {msg}")


if __name__ == "__main__":
    run_scraper_main(CoinbaseL2Scraper)

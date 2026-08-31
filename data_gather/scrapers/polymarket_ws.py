"""
Polymarket CLOB WebSocket scraper — replaces the 1s REST poll with event-driven
book data for the 15m up/down markets.

Two outputs:
  polymarket_book_events — every book/price_change/last_trade_price/
                           tick_size_change/best_bid_ask event, book snapshots
                           exploded into one row per level
  polymarket_markets     — one discovery snapshot per market per ~minute from
                           the Gamma API (metadata, liquidity/volume/OI,
                           token ids, strike)

Market discovery reuses the slug convention from polymarket.py:
    {coin}-updown-15m-{interval_start_unix}
The WS market channel requires resubscribing when the asset id set changes, so
the discovery loop raises ResubscribeSignal via a flag when new windows open.

Run: DATA_SINK_MODE=parquet python -m data_gather.scrapers.polymarket_ws
"""
from __future__ import annotations

import asyncio
import json
import time
from datetime import datetime, timezone

import aiohttp

from data_gather.scrapers.ws_base import StreamBuffer, WsScraper, run_scraper_main

GAMMA_BASE = 'https://gamma-api.polymarket.com'
TIME_HORIZON = 15
time_horizon_seconds = TIME_HORIZON * 60

WS_URL = "wss://ws-subscriptions-clob.polymarket.com/ws/market"
COINS = ["BTC", "ETH", "XRP", "SOL"]
DISCOVERY_INTERVAL_SECONDS = 20
SNAPSHOT_INTERVAL_SECONDS = 60
PING_INTERVAL_SECONDS = 10
BOOK_SNAPSHOT_MIN_INTERVAL_SECONDS = 30.0

BOOK_FIELDS = ["curr_time", "coin", "token_id", "market_slug", "event_type",
               "exchange_time", "side", "price", "size", "best_bid", "best_ask"]
MARKET_FIELDS = ["curr_time", "coin", "time_horizon_minutes", "interval_start_unix",
                 "market_slug", "condition_id", "strike_price", "end_date",
                 "liquidity", "volume", "open_interest", "yes_token_id", "no_token_id",
                 "yes_implied_price", "no_implied_price",
                 "yes_buy_price", "yes_sell_price", "no_buy_price", "no_sell_price"]


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


# Gamma/Binance discovery helpers inlined from the retired polymarket.py REST
# poller (removed in the 2026-07-18 cleanup).

async def get_window_starting_price(session, coin, interval_start_unix):

    # convert to ms because binance works with ms data
    start_ms = interval_start_unix * 1000
    end_ms = start_ms + (time_horizon_seconds * 1000) - 1

    async with session.get(
        'https://api.binance.us/api/v3/klines',
        params={
            'symbol': f'{coin}USDT',
            'interval': f'{TIME_HORIZON}m',
            'startTime': start_ms,
            'endTime': end_ms,
            'limit': 1,
        },
        timeout=aiohttp.ClientTimeout(total=5),
    ) as resp:
        resp.raise_for_status()

        klines = await resp.json()
        if not klines:
            raise ValueError(f'no Binance kline found for {coin} at {interval_start_unix}')

        kline = klines[0]
        return float(kline[1])


async def get_gamma_event(session, slug):
    async with session.get(f'{GAMMA_BASE}/events/slug/{slug}', timeout=aiohttp.ClientTimeout(total=5)) as resp:
        resp.raise_for_status()
        return await resp.json()


def extract_gamma_market(event):
    markets = event.get('markets', [])
    if not markets:
        raise ValueError('no Polymarket markets found in Gamma event payload')

    market = markets[0]

    outcome_prices = json.loads(market.get('outcomePrices', '[]'))
    clob_token_ids = json.loads(market.get('clobTokenIds', '[]'))

    yes_token_id = clob_token_ids[0] if len(clob_token_ids) > 0 else None
    no_token_id = clob_token_ids[1] if len(clob_token_ids) > 1 else None

    return {
        'end_date': event.get('endDate'),
        'market_slug': market.get('slug'),
        'condition_id': market.get('conditionId'),
        'liquidity': market.get('liquidity'),
        'volume': market.get('volume'),
        'open_interest': market.get('openInterest'),
        'outcome_prices': outcome_prices,
        'yes_token_id': yes_token_id,
        'no_token_id': no_token_id,
    }


def _f(x):
    try:
        return float(x)
    except (TypeError, ValueError):
        return None


class PolymarketWsScraper(WsScraper):
    name = "polymarket_ws"
    ping_interval = None          # Polymarket wants text "PING", not WS pings
    ping_timeout = None

    def __init__(self, data_sink=None) -> None:
        super().__init__(data_sink)
        self._http: aiohttp.ClientSession | None = None
        # token_id -> {coin, slug}
        self._tokens: dict[str, dict] = {}
        # token_id -> monotonic time of last full book explosion
        self._last_book_explode: dict[str, float] = {}

    # ------------------------------------------------------------------
    # Discovery (Gamma REST)
    # ------------------------------------------------------------------

    async def _discover_once(self) -> dict[str, dict]:
        """Returns {token_id: {coin, slug}} for current + next 15m windows."""
        assert self._http is not None
        now = int(time.time())
        current = now // time_horizon_seconds * time_horizon_seconds
        windows = [current, current + time_horizon_seconds]
        found: dict[str, dict] = {}

        async def _one(coin: str, start: int):
            slug = f"{coin.lower()}-updown-{TIME_HORIZON}m-{start}"
            try:
                event = await get_gamma_event(self._http, slug)
                market = extract_gamma_market(event)
                for tid in (market.get("yes_token_id"), market.get("no_token_id")):
                    if tid:
                        found[tid] = {"coin": coin, "slug": market.get("market_slug") or slug,
                                      "interval_start_unix": start, "gamma": market}
            except Exception:
                pass  # window may not exist yet

        await asyncio.gather(*[_one(c, w) for c in COINS for w in windows])
        return found

    async def _discovery_loop(self, ws) -> None:
        while True:
            await asyncio.sleep(DISCOVERY_INTERVAL_SECONDS)
            try:
                fresh = await self._discover_once()
                if set(fresh) != set(self._tokens):
                    added = len(set(fresh) - set(self._tokens))
                    removed = len(set(self._tokens) - set(fresh))
                    self._tokens = fresh
                    print(f"{self.name}: token set changed (+{added}/-{removed}) — resubscribing")
                    await ws.close()   # clean teardown; run() reconnects with new set
                    return
                else:
                    self._tokens = fresh
            except Exception as e:  # noqa: BLE001
                print(f"{self.name}: discovery failed: {type(e).__name__}: {e}")

    async def _snapshot_loop(self) -> None:
        """Periodic polymarket_markets metadata snapshots (Gamma + strike)."""
        assert self._http is not None
        buf = self._market_buffer
        while True:
            try:
                curr_time = _now_iso()
                seen_slugs: set[str] = set()
                for tid, info in list(self._tokens.items()):
                    slug_key = (info["coin"], info["interval_start_unix"])
                    if slug_key in seen_slugs:
                        continue
                    seen_slugs.add(slug_key)
                    gm = info["gamma"]
                    try:
                        strike = await get_window_starting_price(
                            self._http, info["coin"], info["interval_start_unix"])
                    except Exception:
                        strike = None
                    op = gm.get("outcome_prices") or []
                    buf.add({
                        "curr_time": curr_time,
                        "coin": info["coin"],
                        "time_horizon_minutes": TIME_HORIZON,
                        "interval_start_unix": info["interval_start_unix"],
                        "market_slug": gm.get("market_slug"),
                        "condition_id": gm.get("condition_id"),
                        "strike_price": strike,
                        "end_date": gm.get("end_date"),
                        "liquidity": _f(gm.get("liquidity")),
                        "volume": _f(gm.get("volume")),
                        "open_interest": _f(gm.get("open_interest")),
                        "yes_token_id": gm.get("yes_token_id"),
                        "no_token_id": gm.get("no_token_id"),
                        "yes_implied_price": _f(op[0]) if len(op) > 0 else None,
                        "no_implied_price": _f(op[1]) if len(op) > 1 else None,
                        "yes_buy_price": None, "yes_sell_price": None,
                        "no_buy_price": None, "no_sell_price": None,
                    })
            except Exception as e:  # noqa: BLE001
                print(f"{self.name}: snapshot loop failed: {type(e).__name__}: {e}")
            await asyncio.sleep(SNAPSHOT_INTERVAL_SECONDS)

    async def _ping_loop(self, ws) -> None:
        while True:
            await asyncio.sleep(PING_INTERVAL_SECONDS)
            await ws.send("PING")

    # ------------------------------------------------------------------
    # WsScraper hooks
    # ------------------------------------------------------------------

    async def get_ws_url(self) -> str:
        if self._http is None or self._http.closed:
            self._http = aiohttp.ClientSession()
        # Block until we have at least one token to subscribe to.
        while True:
            self._tokens = await self._discover_once()
            if self._tokens:
                return WS_URL
            print(f"{self.name}: no active updown markets found, retrying in 30s")
            await asyncio.sleep(30)

    def make_buffers(self) -> list[StreamBuffer]:
        self._book_buffer = StreamBuffer("polymarket_book_events", BOOK_FIELDS)
        self._market_buffer = StreamBuffer("polymarket_markets", MARKET_FIELDS,
                                           flush_interval=30.0)
        return [self._book_buffer, self._market_buffer]

    async def on_connect(self, ws) -> None:
        token_ids = sorted(self._tokens)
        await ws.send(json.dumps({
            "assets_ids": token_ids,
            "type": "market",
            "custom_feature_enabled": True,
        }))
        print(f"{self.name}: subscribed to {len(token_ids)} tokens "
              f"({', '.join(sorted({i['coin'] for i in self._tokens.values()}))})")

    def background_tasks(self, ws) -> list:
        return [self._discovery_loop(ws), self._snapshot_loop(), self._ping_loop(ws)]

    def decode(self, raw):
        if raw == "PONG":
            return None
        try:
            msg = json.loads(raw)
        except Exception:
            return None
        return msg

    async def handle_message(self, msg) -> None:
        # Server may batch events in a JSON array.
        events = msg if isinstance(msg, list) else [msg]
        curr_time = _now_iso()
        for ev in events:
            if not isinstance(ev, dict):
                continue
            self._handle_event(ev, curr_time)

    def _base_row(self, ev: dict, etype: str, token_id: str, curr_time: str) -> dict:
        info = self._tokens.get(token_id, {})
        return {
            "curr_time": curr_time,
            "coin": info.get("coin", ""),
            "token_id": token_id,
            "market_slug": info.get("slug", ev.get("market", "")),
            "event_type": etype,
            "exchange_time": str(ev.get("timestamp", "")),
            "side": "", "price": None, "size": None,
            "best_bid": None, "best_ask": None,
        }

    def _handle_event(self, ev: dict, curr_time: str) -> None:
        etype = ev.get("event_type") or ev.get("type") or ""
        token_id = ev.get("asset_id") or ev.get("assetId") or ""
        base = self._base_row(ev, etype, token_id, curr_time)

        if etype == "book":
            # Polymarket emits a FULL book snapshot on every trade. Exploding
            # each one is ~10x volume for no information: price_change deltas
            # reconstruct the book between snapshots. Keep a full explosion at
            # most once per token per BOOK_SNAPSHOT_MIN_INTERVAL; otherwise
            # store one top-of-book summary row.
            now_mono = time.monotonic()
            last = self._last_book_explode.get(token_id, 0.0)
            bids = ev.get("bids") or []
            asks = ev.get("asks") or []
            if now_mono - last >= BOOK_SNAPSHOT_MIN_INTERVAL_SECONDS:
                self._last_book_explode[token_id] = now_mono
                for levels, side_name in ((bids, "bid"), (asks, "ask")):
                    for level in levels:
                        self._book_buffer.add({**base, "side": side_name,
                                               "price": _f(level.get("price")),
                                               "size": _f(level.get("size"))})
            else:
                # ToB summary: books list levels ascending; best bid = last
                # bid, best ask = first ask (defensive min/max instead).
                bid_prices = [_f(b.get("price")) for b in bids if _f(b.get("price")) is not None]
                ask_prices = [_f(a.get("price")) for a in asks if _f(a.get("price")) is not None]
                self._book_buffer.add({**base, "event_type": "book_tob",
                                       "best_bid": max(bid_prices) if bid_prices else None,
                                       "best_ask": min(ask_prices) if ask_prices else None})
        elif etype == "price_change":
            # asset_id lives on each change entry (event-level is absent).
            for change in (ev.get("changes") or ev.get("price_changes") or [ev]):
                c_token = change.get("asset_id") or change.get("assetId") or token_id
                c_base = self._base_row(ev, etype, c_token, curr_time) if c_token != token_id else base
                side = str(change.get("side", "")).lower()
                self._book_buffer.add({**c_base,
                                       "side": "bid" if side == "buy" else "ask" if side == "sell" else side,
                                       "price": _f(change.get("price")),
                                       "size": _f(change.get("size")),
                                       "best_bid": _f(change.get("best_bid")),
                                       "best_ask": _f(change.get("best_ask"))})
        elif etype == "last_trade_price":
            side = str(ev.get("side", "")).lower()
            self._book_buffer.add({**base, "side": side,
                                   "price": _f(ev.get("price")),
                                   "size": _f(ev.get("size"))})
        elif etype == "tick_size_change":
            self._book_buffer.add({**base, "price": _f(ev.get("new_tick_size"))})
        elif etype == "best_bid_ask":
            self._book_buffer.add({**base,
                                   "best_bid": _f(ev.get("best_bid") or ev.get("bid")),
                                   "best_ask": _f(ev.get("best_ask") or ev.get("ask"))})


if __name__ == "__main__":
    run_scraper_main(PolymarketWsScraper)

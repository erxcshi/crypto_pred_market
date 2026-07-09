"""
Kalshi settlement poller — captures ground-truth labels.

Tracks open markets for the configured series (public /markets endpoint, no
auth), and after each market's close_time polls GET /markets/{ticker} until
`result` is posted ('yes'/'no'), then writes one kalshi_settlements row with
the settlement fields (incl. expiration_value, the 60s-TWAP CF-BRTI print
Kalshi settles on).

Also bootstraps recently settled markets on startup (status=settled listing)
so a restart doesn't lose labels for markets that closed while down.

Series set: SERIES_TEMPLATES x COINS, plus any comma-separated extras in the
KALSHI_EXTRA_SERIES env var (e.g. hourly series tickers once identified).

Run: DATA_SINK_MODE=parquet python -m data_gather.scrapers.kalshi_settlements
"""
from __future__ import annotations

import asyncio
import os
import time
from datetime import datetime, timezone

import aiohttp

from config import create_data_sink

REST_BASE = "https://api.elections.kalshi.com/trade-api/v2"
COINS = ["BTC", "ETH", "XRP", "SOL"]
# 15-minute up/down + hourly directional (above/below) series.
SERIES_TEMPLATES = ["KX{coin}15M", "KX{coin}D"]

DISCOVER_INTERVAL_SECONDS = 60
SETTLE_POLL_INTERVAL_SECONDS = 15
BOOTSTRAP_SETTLED_LIMIT = 50
REQUEST_GAP_SECONDS = 0.35        # stay far below Kalshi's public rate limit

FIELDS = ["curr_time", "ticker", "coin", "series", "open_time", "close_time",
          "status", "result", "floor_strike", "cap_strike", "expiration_value",
          "settlement_value_dollars", "last_price_dollars", "volume", "open_interest"]


def _series_list() -> list[tuple[str, str]]:
    """[(series_ticker, coin)] — templates x coins + KALSHI_EXTRA_SERIES."""
    out = [(tpl.format(coin=c), c) for tpl in SERIES_TEMPLATES for c in COINS]
    for extra in filter(None, (s.strip() for s in os.getenv("KALSHI_EXTRA_SERIES", "").split(","))):
        coin = next((c for c in COINS if c in extra), "")
        out.append((extra, coin))
    return out


def _row(market: dict, coin: str, series: str) -> dict:
    return {
        "curr_time": datetime.now(timezone.utc).isoformat(),
        "ticker": market.get("ticker"),
        "coin": coin,
        "series": series,
        "open_time": market.get("open_time"),
        "close_time": market.get("close_time"),
        "status": market.get("status"),
        "result": market.get("result"),
        "floor_strike": market.get("floor_strike"),
        "cap_strike": market.get("cap_strike"),
        "expiration_value": market.get("expiration_value"),
        "settlement_value_dollars": (market.get("settlement_value_dollars")
                                     or market.get("settlement_value")),
        "last_price_dollars": market.get("last_price_dollars"),
        "volume": market.get("volume"),
        "open_interest": market.get("open_interest"),
    }


class SettlementPoller:
    def __init__(self, data_sink=None) -> None:
        self.data_sink = data_sink or create_data_sink()
        # ticker -> {coin, series, close_ts}
        self._pending: dict[str, dict] = {}
        self._written: set[str] = set()

    async def _get(self, session: aiohttp.ClientSession, path: str, params: dict | None = None):
        async with session.get(f"{REST_BASE}{path}", params=params,
                               timeout=aiohttp.ClientTimeout(total=15)) as resp:
            if resp.status == 429:
                await asyncio.sleep(5)
                raise RuntimeError("rate limited")
            resp.raise_for_status()
            return await resp.json()

    async def _submit(self, rows: list[dict]) -> None:
        if rows:
            await self.data_sink.submit_rows("kalshi_settlements", rows, fieldnames=FIELDS)

    @staticmethod
    def _close_ts(market: dict) -> float:
        try:
            return datetime.fromisoformat(
                str(market.get("close_time", "")).replace("Z", "+00:00")).timestamp()
        except ValueError:
            return 0.0

    async def _bootstrap(self, session: aiohttp.ClientSession) -> None:
        """Grab the most recent settled markets per series so restarts don't
        lose labels."""
        rows = []
        for series, coin in _series_list():
            try:
                data = await self._get(session, "/markets", {
                    "series_ticker": series, "status": "settled",
                    "limit": BOOTSTRAP_SETTLED_LIMIT,
                    # Without min_close_ts the settled listing returns ancient
                    # markets; restrict to the last 24h.
                    "min_close_ts": int(time.time()) - 86400,
                })
                for m in data.get("markets") or []:
                    t = m.get("ticker")
                    if t and t not in self._written and m.get("result") in ("yes", "no"):
                        self._written.add(t)
                        rows.append(_row(m, coin, series))
            except Exception as e:  # noqa: BLE001
                print(f"kalshi_settlements: bootstrap failed for {series}: {type(e).__name__}: {e}")
            await asyncio.sleep(REQUEST_GAP_SECONDS)
        await self._submit(rows)
        if rows:
            print(f"kalshi_settlements: bootstrapped {len(rows)} settled markets")

    async def _discover_loop(self, session: aiohttp.ClientSession) -> None:
        while True:
            for series, coin in _series_list():
                try:
                    data = await self._get(session, "/markets", {
                        "series_ticker": series, "status": "open", "limit": 100,
                    })
                    for m in data.get("markets") or []:
                        t = m.get("ticker")
                        if t and t not in self._pending and t not in self._written:
                            self._pending[t] = {"coin": coin, "series": series,
                                                "close_ts": self._close_ts(m)}
                except Exception as e:  # noqa: BLE001
                    print(f"kalshi_settlements: discovery failed for {series}: {type(e).__name__}: {e}")
                await asyncio.sleep(REQUEST_GAP_SECONDS)
            await asyncio.sleep(DISCOVER_INTERVAL_SECONDS)

    async def _settle_loop(self, session: aiohttp.ClientSession) -> None:
        while True:
            now = time.time()
            due = [t for t, info in self._pending.items() if now > info["close_ts"] + 5]
            rows = []
            for ticker in due:
                info = self._pending[ticker]
                try:
                    data = await self._get(session, f"/markets/{ticker}")
                    market = data.get("market") or {}
                    if market.get("result") in ("yes", "no"):
                        rows.append(_row(market, info["coin"], info["series"]))
                        self._written.add(ticker)
                        self._pending.pop(ticker, None)
                        print(f"kalshi_settlements: {ticker} -> {market.get('result')} "
                              f"(exp_value={market.get('expiration_value')!r})")
                    elif now > info["close_ts"] + 3600:
                        # Give up after an hour past close (voided/rare states).
                        rows.append(_row(market, info["coin"], info["series"]))
                        self._written.add(ticker)
                        self._pending.pop(ticker, None)
                        print(f"kalshi_settlements: {ticker} unresolved 1h past close "
                              f"(status={market.get('status')}) — recorded as-is")
                except Exception as e:  # noqa: BLE001
                    print(f"kalshi_settlements: poll failed for {ticker}: {type(e).__name__}: {e}")
                await asyncio.sleep(REQUEST_GAP_SECONDS)
            await self._submit(rows)
            await asyncio.sleep(SETTLE_POLL_INTERVAL_SECONDS)

    async def run(self) -> None:
        series = _series_list()
        print(f"kalshi_settlements: tracking {[s for s, _ in series]}")
        async with aiohttp.ClientSession() as session:
            await self._bootstrap(session)
            await asyncio.gather(
                self._discover_loop(session),
                self._settle_loop(session),
            )


if __name__ == "__main__":
    try:
        asyncio.run(SettlementPoller().run())
    except KeyboardInterrupt:
        print("stopped by user")

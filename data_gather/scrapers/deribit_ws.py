"""
Deribit public WebSocket scraper — replaces the 5s REST poller with streaming:

  deribit_dvol    — DVOL implied-vol index (BTC, ETH): the sigma input for the
                    fair-value model
  deribit_index   — Deribit composite price index (btc_usd, eth_usd)
  deribit_perp    — BTC-PERPETUAL / ETH-PERPETUAL ticker (funding, basis, OI)
  deribit_ticker  — near-spot nearest-expiry option tickers (smile/skew);
                    instrument selection refreshed every 5 min via REST,
                    subscriptions updated in place (no reconnect)

Deribit WS is JSON-RPC: subscribe via public/subscribe, notifications arrive
as method=subscription with params.channel + params.data.

Run: DATA_SINK_MODE=parquet python -m data_gather.scrapers.deribit_ws
"""
from __future__ import annotations

import asyncio
import json
from datetime import datetime, timezone

import aiohttp

from data_gather.scrapers.ws_base import StreamBuffer, WsScraper, run_scraper_main

WS_URL = "wss://www.deribit.com/ws/api/v2"
REST_BASE = "https://www.deribit.com/api/v2"
CURRENCIES = ["BTC", "ETH"]
OPTION_REFRESH_SECONDS = 300
NEAR_SPOT_PER_SIDE = 8          # strikes per side (calls+puts) per currency

STATIC_CHANNELS = [
    "deribit_volatility_index.btc_usd",
    "deribit_volatility_index.eth_usd",
    "deribit_price_index.btc_usd",
    "deribit_price_index.eth_usd",
    "ticker.BTC-PERPETUAL.100ms",
    "ticker.ETH-PERPETUAL.100ms",
]

DVOL_FIELDS = ["curr_time", "currency", "exchange_ts_ms", "dvol"]
INDEX_FIELDS = ["curr_time", "index_name", "exchange_ts_ms", "price"]
PERP_FIELDS = ["curr_time", "instrument_name", "exchange_ts_ms", "last_price",
               "mark_price", "index_price", "best_bid", "best_ask",
               "current_funding", "funding_8h", "open_interest", "volume_24h"]
TICKER_FIELDS = ["curr_time", "instrument_name", "exchange_ts_ms", "underlying_price",
                 "mark_price", "mark_iv", "bid_price", "ask_price", "bid_iv", "ask_iv",
                 "delta", "gamma", "vega", "theta", "open_interest", "volume"]


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


# Instrument-name helpers inlined from the retired deribit_vol.py REST poller
# (removed in the 2026-07-18 cleanup).

def parse_instrument_name(instrument_name):
    """Parse Deribit instrument name: BTC-25DEC26-100000-C"""
    try:
        parts = instrument_name.split('-')
        if len(parts) >= 4:
            currency = parts[0]
            expiry = parts[1]
            strike = float(parts[2])
            option_type = parts[3]  # C or P
            return {'currency': currency, 'expiry': expiry, 'strike': strike, 'type': option_type}
    except (ValueError, IndexError):
        pass
    return None


def parse_expiry_date(expiry_str):
    """Parse Deribit expiry format (e.g., '27MAR26') into a datetime for proper sorting."""
    try:
        # Format: DDMMMYY (e.g., 27MAR26)
        return datetime.strptime(expiry_str, '%d%b%y')
    except ValueError:
        return None


def group_options_by_expiry(option_summaries):
    by_expiry = {}
    for opt in option_summaries:
        parsed = parse_instrument_name(opt.get('instrument_name', ''))
        if not parsed:
            continue

        expiry = parsed['expiry']
        if expiry not in by_expiry:
            by_expiry[expiry] = []
        by_expiry[expiry].append(opt)

    return by_expiry


class DeribitWsScraper(WsScraper):
    name = "deribit_ws"

    def __init__(self, data_sink=None) -> None:
        super().__init__(data_sink)
        self._http: aiohttp.ClientSession | None = None
        self._option_channels: set[str] = set()
        self._rpc_id = 0

    # ------------------------------------------------------------------
    # Option instrument selection (REST)
    # ------------------------------------------------------------------

    async def _near_spot_instruments(self, currency: str) -> list[str]:
        assert self._http is not None
        url = f"{REST_BASE}/public/get_book_summary_by_currency"
        async with self._http.get(url, params={"currency": currency, "kind": "option"},
                                  timeout=aiohttp.ClientTimeout(total=15)) as resp:
            resp.raise_for_status()
            payload = await resp.json()
        summaries = payload.get("result") or []
        by_expiry = group_options_by_expiry(summaries)
        if not by_expiry:
            return []
        nearest = sorted(by_expiry, key=lambda x: parse_expiry_date(x) or datetime.max)[0]
        options = by_expiry[nearest]
        underlying = next((o.get("underlying_price") for o in options
                           if o.get("underlying_price") is not None), None)
        if underlying is None:
            return []

        calls, puts = [], []
        for o in options:
            parsed = parse_instrument_name(o.get("instrument_name", ""))
            if not parsed:
                continue
            item = (abs(parsed["strike"] - underlying), o["instrument_name"])
            (calls if parsed["type"] == "C" else puts).append(item)
        picked = [n for _, n in sorted(calls)[:NEAR_SPOT_PER_SIDE]]
        picked += [n for _, n in sorted(puts)[:NEAR_SPOT_PER_SIDE]]
        return picked

    async def _rpc(self, ws, method: str, params: dict) -> None:
        self._rpc_id += 1
        await ws.send(json.dumps({"jsonrpc": "2.0", "id": self._rpc_id,
                                  "method": method, "params": params}))

    async def _option_refresh_loop(self, ws) -> None:
        while True:
            try:
                names: list[str] = []
                for cur in CURRENCIES:
                    names += await self._near_spot_instruments(cur)
                fresh = {f"ticker.{n}.100ms" for n in names}
                to_add = sorted(fresh - self._option_channels)
                to_drop = sorted(self._option_channels - fresh)
                if to_add:
                    await self._rpc(ws, "public/subscribe", {"channels": to_add})
                if to_drop:
                    await self._rpc(ws, "public/unsubscribe", {"channels": to_drop})
                if to_add or to_drop:
                    print(f"{self.name}: options resubscribed (+{len(to_add)}/-{len(to_drop)}), "
                          f"now {len(fresh)} option channels")
                self._option_channels = fresh
            except Exception as e:  # noqa: BLE001
                print(f"{self.name}: option refresh failed: {type(e).__name__}: {e}")
            await asyncio.sleep(OPTION_REFRESH_SECONDS)

    # ------------------------------------------------------------------
    # WsScraper hooks
    # ------------------------------------------------------------------

    async def get_ws_url(self) -> str:
        if self._http is None or self._http.closed:
            self._http = aiohttp.ClientSession()
        return WS_URL

    def make_buffers(self) -> list[StreamBuffer]:
        self._dvol = StreamBuffer("deribit_dvol", DVOL_FIELDS, flush_interval=10.0)
        self._index = StreamBuffer("deribit_index", INDEX_FIELDS, flush_interval=10.0)
        self._perp = StreamBuffer("deribit_perp", PERP_FIELDS, flush_interval=10.0)
        self._ticker = StreamBuffer("deribit_ticker", TICKER_FIELDS, flush_interval=10.0)
        return [self._dvol, self._index, self._perp, self._ticker]

    async def on_connect(self, ws) -> None:
        self._option_channels = set()
        await self._rpc(ws, "public/set_heartbeat", {"interval": 30})
        await self._rpc(ws, "public/subscribe", {"channels": STATIC_CHANNELS})
        print(f"{self.name}: subscribed to {len(STATIC_CHANNELS)} static channels")

    def background_tasks(self, ws) -> list:
        return [self._option_refresh_loop(ws)]

    async def handle_message(self, msg: dict) -> None:
        method = msg.get("method")
        if method == "heartbeat":
            # Deribit requires answering test_request or it drops the socket.
            if (msg.get("params") or {}).get("type") == "test_request":
                await self._rpc(self.ws, "public/test", {})
            return
        if method != "subscription":
            return  # RPC acks

        params = msg.get("params") or {}
        channel = params.get("channel", "")
        data = params.get("data") or {}
        curr_time = _now_iso()

        if channel.startswith("deribit_volatility_index."):
            self._dvol.add({
                "curr_time": curr_time,
                "currency": channel.rsplit(".", 1)[-1].split("_")[0].upper(),
                "exchange_ts_ms": data.get("timestamp"),
                "dvol": data.get("volatility"),
            })
        elif channel.startswith("deribit_price_index."):
            self._index.add({
                "curr_time": curr_time,
                "index_name": data.get("index_name") or channel.rsplit(".", 1)[-1],
                "exchange_ts_ms": data.get("timestamp"),
                "price": data.get("price"),
            })
        elif channel.startswith("ticker."):
            instrument = data.get("instrument_name", "")
            if instrument.endswith("-PERPETUAL"):
                stats = data.get("stats") or {}
                self._perp.add({
                    "curr_time": curr_time,
                    "instrument_name": instrument,
                    "exchange_ts_ms": data.get("timestamp"),
                    "last_price": data.get("last_price"),
                    "mark_price": data.get("mark_price"),
                    "index_price": data.get("index_price"),
                    "best_bid": data.get("best_bid_price"),
                    "best_ask": data.get("best_ask_price"),
                    "current_funding": data.get("current_funding"),
                    "funding_8h": data.get("funding_8h"),
                    "open_interest": data.get("open_interest"),
                    "volume_24h": stats.get("volume"),
                })
            else:
                greeks = data.get("greeks") or {}
                stats = data.get("stats") or {}
                self._ticker.add({
                    "curr_time": curr_time,
                    "instrument_name": instrument,
                    "exchange_ts_ms": data.get("timestamp"),
                    "underlying_price": data.get("underlying_price"),
                    "mark_price": data.get("mark_price"),
                    "mark_iv": data.get("mark_iv"),
                    "bid_price": data.get("best_bid_price"),
                    "ask_price": data.get("best_ask_price"),
                    "bid_iv": data.get("bid_iv"),
                    "ask_iv": data.get("ask_iv"),
                    "delta": greeks.get("delta"),
                    "gamma": greeks.get("gamma"),
                    "vega": greeks.get("vega"),
                    "theta": greeks.get("theta"),
                    "open_interest": data.get("open_interest"),
                    "volume": stats.get("volume"),
                })

if __name__ == "__main__":
    run_scraper_main(DeribitWsScraper)

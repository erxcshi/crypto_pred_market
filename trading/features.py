"""
Real-time feature engine — mirrors filter.py exactly.

Two independent buffers feed into a single compute_features() call:
  - KalshiMarketBuffer: per-event (open_time, close_time) rolling windows for
    each coin's Kalshi market data (BTC primary; ETH/XRP/SOL cross-asset asof).
  - SpotBuffer: rolling 1-second BTC spot bars for the btc_spot_* features.
"""
from __future__ import annotations

import math
import time
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Optional


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _safe_std(values: list[float]) -> float:
    n = len(values)
    if n < 2:
        return 0.0
    mean = sum(values) / n
    var = sum((v - mean) ** 2 for v in values) / n  # population std, matches numpy default
    return math.sqrt(var)


def _safe_mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


# ---------------------------------------------------------------------------
# Per-event Kalshi rolling state
# ---------------------------------------------------------------------------

@dataclass
class _EventState:
    """History for one (open_time, close_time) event, one coin."""
    yes_mid_history: deque = field(default_factory=lambda: deque(maxlen=65))
    yes_mid_change_history: deque = field(default_factory=lambda: deque(maxlen=61))
    yes_spread_history: deque = field(default_factory=lambda: deque(maxlen=30))

    # latest raw market snapshot
    last_price_dollars: float = 0.0
    yes_ask_dollars: float = 0.5
    yes_bid_dollars: float = 0.5
    no_ask_dollars: float = 0.5
    no_bid_dollars: float = 0.5
    floor_strike: float = 0.0
    spot_price: float = 0.0   # BTC spot price at last update


class KalshiMarketBuffer:
    """
    Tracks per-event rolling state for each coin.

    update() is called once per second when a new Kalshi poll arrives.
    get_features() returns the dict slice for that coin's features.
    """

    def __init__(self) -> None:
        # {coin: {(open_time_str, close_time_str): _EventState}}
        self._events: dict[str, dict[tuple, _EventState]] = {}

    def _get_state(self, coin: str, event_key: tuple) -> _EventState:
        coin_map = self._events.setdefault(coin, {})
        if event_key not in coin_map:
            coin_map[event_key] = _EventState()
            # Prune closed events for this coin (keep at most 4 open at a time)
            if len(coin_map) > 8:
                oldest = next(iter(coin_map))
                del coin_map[oldest]
        return coin_map[event_key]

    def update(
        self,
        coin: str,
        open_time: str,
        close_time: str,
        last_price_dollars: float,
        yes_ask_dollars: float,
        yes_bid_dollars: float,
        no_ask_dollars: float,
        no_bid_dollars: float,
        floor_strike: float,
        spot_price: float,
    ) -> None:
        key = (open_time, close_time)
        s = self._get_state(coin, key)

        yes_mid = (yes_ask_dollars + yes_bid_dollars) / 2.0
        yes_spread = yes_ask_dollars - yes_bid_dollars

        # yes_mid_change_1s: diff from previous tick
        if s.yes_mid_history:
            change_1s = yes_mid - s.yes_mid_history[-1]
        else:
            change_1s = 0.0

        s.yes_mid_history.append(yes_mid)
        s.yes_mid_change_history.append(change_1s)
        s.yes_spread_history.append(yes_spread)

        s.last_price_dollars = last_price_dollars
        s.yes_ask_dollars = yes_ask_dollars
        s.yes_bid_dollars = yes_bid_dollars
        s.no_ask_dollars = no_ask_dollars
        s.no_bid_dollars = no_bid_dollars
        s.floor_strike = floor_strike
        s.spot_price = spot_price

    def get_features(
        self,
        coin: str,
        open_time: str,
        close_time: str,
        prefix: bool = False,
    ) -> Optional[dict[str, float]]:
        key = (open_time, close_time)
        coin_map = self._events.get(coin, {})
        s = coin_map.get(key)
        if s is None:
            return None

        yes_mid = (s.yes_ask_dollars + s.yes_bid_dollars) / 2.0
        yes_spread = s.yes_ask_dollars - s.yes_bid_dollars

        changes = list(s.yes_mid_change_history)
        spreads = list(s.yes_spread_history)
        mids = list(s.yes_mid_history)

        change_1s = changes[-1] if changes else 0.0
        change_5s = (mids[-1] - mids[-6]) if len(mids) >= 6 else 0.0
        # Match training: rolling_std(30).fill_null(0), rolling_std(60).fill_null(0)
        std_30 = _safe_std(changes[-30:]) if len(changes) >= 30 else 0.0
        std_60 = _safe_std(changes[-60:]) if len(changes) >= 60 else 0.0
        # Match training: rolling_mean(30).fill_null(current_spread)
        spread_mean_30 = _safe_mean(spreads[-30:]) if len(spreads) >= 30 else yes_spread

        distance = s.spot_price - s.floor_strike

        raw = {
            "last_price_dollars": s.last_price_dollars,
            "yes_mid_dollars": yes_mid,
            "yes_spread_dollars": yes_spread,
            "distance_from_strike": distance,
            "yes_mid_change_1s": change_1s,
            "yes_mid_change_5s": change_5s,
            "yes_mid_change_std_30s": std_30,
            "yes_mid_change_std_60s": std_60,
            "yes_spread_mean_30s": spread_mean_30,
            "yes_ask_dollars": s.yes_ask_dollars,
            "yes_bid_dollars": s.yes_bid_dollars,
            "no_ask_dollars": s.no_ask_dollars,
            "no_bid_dollars": s.no_bid_dollars,
        }

        if not prefix:
            return raw

        p = f"{coin}_"
        return {f"{p}{k}": v for k, v in raw.items()}

    def latest_features_asof(self, coin: str) -> Optional[dict[str, float]]:
        """
        Returns cross-asset features for ETH/XRP/SOL using the most
        recently updated event (mirrors join_asof strategy='backward').
        """
        coin_map = self._events.get(coin, {})
        if not coin_map:
            return None
        key = next(reversed(coin_map))
        return self.get_features(coin, key[0], key[1], prefix=True)


# ---------------------------------------------------------------------------
# BTC spot 1-second bar buffer
# ---------------------------------------------------------------------------

@dataclass
class _SpotBar:
    timestamp: float          # unix seconds (right edge of 1s window)
    price: float              # last trade price
    size: float               # total size traded
    signed_size: float        # buy_size - sell_size
    ret_1s: float = 0.0
    ret_5s: float = 0.0
    ret_15s: float = 0.0
    ret_60s: float = 0.0


class SpotBuffer:
    """
    Accumulates Coinbase trade ticks into 1-second bars, then computes
    rolling statistics that match filter.py's build_btc_spot_features().
    """
    # max history needed: 300 bars for vol_5m
    _MAX_BARS = 320

    def __init__(self) -> None:
        self._bars: deque[_SpotBar] = deque(maxlen=self._MAX_BARS)
        self._pending_trades: list[tuple[float, float, str]] = []  # (price, size, side)
        self._current_bar_ts: float = 0.0    # unix second of open bar

    def add_trade(self, price: float, size: float, side: str) -> None:
        """Called for every Coinbase BTC-USD match event."""
        ts = math.floor(time.time())
        if ts > self._current_bar_ts:
            self._flush_bar(self._current_bar_ts)
            self._current_bar_ts = ts
        self._pending_trades.append((price, size, side))

    def _flush_bar(self, ts: float) -> None:
        if not self._pending_trades:
            return
        prices = [p for p, _, _ in self._pending_trades]
        price = prices[-1]
        size = sum(s for _, s, _ in self._pending_trades)
        signed_size = sum(
            s if side == "buy" else -s
            for _, s, side in self._pending_trades
        )
        bar = _SpotBar(timestamp=ts, price=price, size=size, signed_size=signed_size)
        self._pending_trades.clear()

        # returns relative to prior bars
        bars = list(self._bars)
        def _ret(n: int) -> float:
            return (price / bars[-n].price - 1.0) if len(bars) >= n else 0.0

        bar.ret_1s = _ret(1)
        bar.ret_5s = _ret(5)
        bar.ret_15s = _ret(15)
        bar.ret_60s = _ret(60)

        self._bars.append(bar)

    def get_features(self) -> dict[str, float]:
        """Returns the latest btc_spot_* feature slice."""
        # Flush any pending trades for the current second
        if self._pending_trades:
            ts = math.floor(time.time())
            if ts > self._current_bar_ts:
                self._flush_bar(self._current_bar_ts)
                self._current_bar_ts = ts

        if not self._bars:
            return {
                "btc_spot_price": 0.0,
                "btc_spot_size_1s": 0.0,
                "btc_spot_signed_size_1s": 0.0,
                "btc_spot_return_1s": 0.0,
                "btc_spot_return_5s": 0.0,
                "btc_spot_return_15s": 0.0,
                "btc_spot_return_60s": 0.0,
                "btc_spot_return_vol_30s": 0.0,
                "btc_spot_return_vol_5m": 0.0,
                "btc_spot_signed_flow_mean_30s": 0.0,
                "btc_spot_size_mean_30s": 0.0,
            }

        bars = list(self._bars)
        latest = bars[-1]

        rets_30 = [b.ret_1s for b in bars[-30:]]
        rets_300 = [b.ret_1s for b in bars[-300:]]
        signed_30 = [b.signed_size for b in bars[-30:]]
        size_30 = [b.size for b in bars[-30:]]

        return {
            "btc_spot_price": latest.price,
            "btc_spot_size_1s": latest.size,
            "btc_spot_signed_size_1s": latest.signed_size,
            "btc_spot_return_1s": latest.ret_1s,
            "btc_spot_return_5s": latest.ret_5s,
            "btc_spot_return_15s": latest.ret_15s,
            "btc_spot_return_60s": latest.ret_60s,
            "btc_spot_return_vol_30s": _safe_std(rets_30),
            "btc_spot_return_vol_5m": _safe_std(rets_300),
            "btc_spot_signed_flow_mean_30s": _safe_mean(signed_30),
            "btc_spot_size_mean_30s": _safe_mean(size_30),
        }

    @property
    def latest_price(self) -> float:
        # Prefer pending trades (current-second, not yet closed into a bar)
        if self._pending_trades:
            return self._pending_trades[-1][0]
        if self._bars:
            return self._bars[-1].price
        return 0.0


# ---------------------------------------------------------------------------
# Top-level feature engine
# ---------------------------------------------------------------------------

class FeatureEngine:
    """
    Combines KalshiMarketBuffer and SpotBuffer into the full 64-feature
    vector expected by BayesLogRegModel.
    """

    def __init__(self) -> None:
        self.kalshi = KalshiMarketBuffer()
        self.spot = SpotBuffer()
        # Per-coin latest spot prices for distance_from_strike computation.
        # BTC is served by SpotBuffer; ETH/XRP/SOL come from Coinbase feeds.
        self._coin_prices: dict[str, float] = {}

    # Track which coins we've already dumped market keys for (one-time diagnostic)
    _dumped_keys: set = set()

    def update_kalshi(
        self,
        coin: str,
        open_time: str,
        close_time: str,
        market_row: dict,
    ) -> None:
        import logging
        _log = logging.getLogger("features")

        if coin == "BTC":
            spot_price = self.spot.latest_price
        else:
            spot_price = self._coin_prices.get(coin, 0.0)

        def _dollars(val) -> float:
            """Normalise Kalshi price fields to [0, 1] dollar range."""
            if val is None:
                return 0.5
            v = float(val)
            return v / 100.0 if v > 1.0 else v

        # Try all known Kalshi field names for the binary strike price
        raw_strike = (
            market_row.get("floor_strike")
            or market_row.get("strike_level")
            or market_row.get("result_level")
            or market_row.get("strike")
        )
        floor_strike = float(raw_strike) if raw_strike is not None else 0.0

        if floor_strike == 0.0 and coin not in self._dumped_keys:
            _log.warning(
                f"floor_strike=0 for {coin} market — distance_from_strike will be ~spot price "
                f"(garbage feature). Available market keys: {list(market_row.keys())}"
            )
            self._dumped_keys.add(coin)

        yes_ask = _dollars(market_row.get("yes_ask_dollars") or market_row.get("yes_ask"))
        yes_bid = _dollars(market_row.get("yes_bid_dollars") or market_row.get("yes_bid"))
        no_ask  = _dollars(market_row.get("no_ask_dollars")  or market_row.get("no_ask"))
        no_bid  = _dollars(market_row.get("no_bid_dollars")  or market_row.get("no_bid"))

        # When Kalshi returns 0 for a side (no resting orders), fall back to complement pricing
        if no_ask <= 0:
            no_ask = max(1.0 - yes_bid, 0.01) if yes_bid > 0 else 0.5
        if no_bid <= 0:
            no_bid = max(1.0 - yes_ask, 0.01) if yes_ask > 0 else 0.5
        if yes_ask <= 0:
            yes_ask = max(1.0 - no_bid, 0.01) if no_bid > 0 else 0.5
        if yes_bid <= 0:
            yes_bid = max(1.0 - no_ask, 0.01) if no_ask > 0 else 0.5

        self.kalshi.update(
            coin=coin,
            open_time=open_time,
            close_time=close_time,
            last_price_dollars=_dollars(
                market_row.get("last_price_dollars") or market_row.get("last_price")
            ),
            yes_ask_dollars=yes_ask,
            yes_bid_dollars=yes_bid,
            no_ask_dollars=no_ask,
            no_bid_dollars=no_bid,
            floor_strike=floor_strike,
            spot_price=spot_price,
        )

    def update_spot(self, price: float, size: float, side: str) -> None:
        self.spot.add_trade(price, size, side)

    def update_coin_price(self, coin: str, price: float) -> None:
        """Called for each Coinbase trade tick for ETH/XRP/SOL."""
        self._coin_prices[coin] = price

    def compute_features(
        self,
        coin: str,
        open_time: str,
        close_time: str,
        time_to_close: float,
    ) -> tuple[Optional[dict[str, float]], str]:
        """
        Returns (feature_dict, reason) where reason is '' on success
        or a short description of what is blocking.
        """
        btc_feats = self.kalshi.get_features(coin, open_time, close_time, prefix=False)
        if btc_feats is None:
            return None, f"no BTC kalshi data for event {open_time}/{close_time}"

        cross_feats: dict[str, float] = {}
        for cross_coin in ("ETH", "XRP", "SOL"):
            cf = self.kalshi.latest_features_asof(cross_coin)
            if cf is None:
                return None, f"no {cross_coin} kalshi data yet"
            cross_feats.update(cf)

        spot_feats = self.spot.get_features()
        if spot_feats["btc_spot_price"] == 0.0:
            return None, "no BTC spot data yet (Coinbase WS not warm)"

        return {
            "time_to_close": time_to_close,
            **btc_feats,
            **cross_feats,
            **spot_feats,
        }, ""

    def buffer_status(self) -> dict:
        """Summary of what's in each buffer — for diagnostic logging."""
        status = {"spot_bars": len(self.spot._bars), "spot_price": self.spot.latest_price}
        for coin in ("BTC", "ETH", "XRP", "SOL"):
            events = self.kalshi._events.get(coin, {})
            total_samples = sum(len(s.yes_mid_history) for s in events.values())
            status[f"{coin}_events"] = len(events)
            status[f"{coin}_samples"] = total_samples
        return status

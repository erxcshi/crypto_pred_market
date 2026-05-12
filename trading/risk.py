"""
Risk management: tracks open exposure and enforces per-market and portfolio limits.
"""
from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Optional

from trading.config import (
    COOLDOWN_SECONDS_AFTER_STOP,
    MAX_OPEN_MARKETS,
    MAX_TOTAL_EXPOSURE_DOLLARS,
)


@dataclass
class _MarketPosition:
    ticker: str
    side: str           # "yes" or "no"
    contracts: int
    avg_price: float    # dollars (0-1)

    @property
    def exposure_dollars(self) -> float:
        return self.contracts * self.avg_price


class RiskManager:
    """
    Stateful tracker of open positions and portfolio exposure.

    Positions are keyed by "{ticker}_{side}" so YES and NO can coexist
    for the same market (arbitrage hedge case).

    Call check_trade() before placing any order.
    Call record_fill() after a confirmed fill.
    """

    def __init__(
        self,
        max_open_markets: int = MAX_OPEN_MARKETS,
        max_total_exposure: float = MAX_TOTAL_EXPOSURE_DOLLARS,
    ) -> None:
        self._max_markets = max_open_markets
        self._max_exposure = max_total_exposure
        self._positions: dict[str, _MarketPosition] = {}  # "{ticker}_{side}" -> position
        # Post-stop cooldown: (ticker, side) -> unix timestamp when stop-loss
        # last fired on that SPECIFIC side. Blocks same-side re-entry but leaves
        # the opposite side free to trade (model can flip its view after a stop).
        self._last_stop_time: dict[tuple[str, str], float] = {}

    # ------------------------------------------------------------------
    # Post-stop cooldown
    # ------------------------------------------------------------------

    def record_stop_exit(self, ticker: str, side: str) -> None:
        """Called after a stop-loss fires on (ticker, side). Starts the re-entry
        cooldown for that specific side; the opposite side is unaffected."""
        self._last_stop_time[(ticker, side)] = time.time()

    def in_stop_cooldown(self, ticker: str, side: str) -> tuple[bool, float]:
        """
        Returns (is_in_cooldown, seconds_remaining) for the given (ticker, side).
        seconds_remaining is 0 when not in cooldown.
        """
        last = self._last_stop_time.get((ticker, side))
        if last is None:
            return False, 0.0
        elapsed = time.time() - last
        remaining = COOLDOWN_SECONDS_AFTER_STOP - elapsed
        if remaining > 0:
            return True, remaining
        return False, 0.0

    @staticmethod
    def _key(ticker: str, side: str) -> str:
        return f"{ticker}_{side}"

    @staticmethod
    def _opposite(side: str) -> str:
        return "no" if side == "yes" else "yes"

    # ------------------------------------------------------------------
    # Queries
    # ------------------------------------------------------------------

    @property
    def total_exposure(self) -> float:
        return sum(p.exposure_dollars for p in self._positions.values())

    @property
    def open_market_count(self) -> int:
        # Count unique tickers regardless of how many sides are open
        return len({k.rsplit("_", 1)[0] for k in self._positions})

    def current_side_contracts(self, ticker: str, side: str) -> int:
        pos = self._positions.get(self._key(ticker, side))
        return pos.contracts if pos else 0

    def current_side_avg_price(self, ticker: str, side: str) -> float:
        pos = self._positions.get(self._key(ticker, side))
        return pos.avg_price if pos else 0.0

    def current_contracts(self, ticker: str) -> int:
        return self.current_side_contracts(ticker, "yes") + self.current_side_contracts(ticker, "no")

    def is_flat(self, ticker: str) -> bool:
        return (
            self._key(ticker, "yes") not in self._positions
            and self._key(ticker, "no") not in self._positions
        )

    def position_side(self, ticker: str) -> str | None:
        """Returns 'yes', 'no', 'both', or None if flat."""
        has_yes = self._key(ticker, "yes") in self._positions
        has_no  = self._key(ticker, "no")  in self._positions
        if has_yes and has_no:
            return "both"
        if has_yes:
            return "yes"
        if has_no:
            return "no"
        return None

    def net_position(self, ticker: str) -> tuple[int, str | None]:
        """
        Returns (net_contracts, net_side).
        2 YES + 1 NO → (1, 'yes')
        1 YES + 2 NO → (1, 'no')
        0 net         → (0, None)
        """
        yes = self._positions.get(self._key(ticker, "yes"))
        no  = self._positions.get(self._key(ticker, "no"))
        yes_ct = yes.contracts if yes else 0
        no_ct  = no.contracts  if no  else 0
        net = yes_ct - no_ct
        if net > 0:
            return net, "yes"
        if net < 0:
            return abs(net), "no"
        return 0, None

    def net_position_str(self, ticker: str) -> str:
        """
        Human-readable net position showing gross holdings, avg prices, and
        locked PnL from any hedged pairs.

        Examples:
          flat
          2YES@0.65
          2YES@0.65 + 1NO@0.40 → net 1 YES
          2YES@0.65 + 2NO@0.40 → net flat (locked -$0.10)
          2YES@0.70 + 2NO@0.30 → net flat (locked $0.00)
        """
        yes = self._positions.get(self._key(ticker, "yes"))
        no  = self._positions.get(self._key(ticker, "no"))
        yes_ct  = yes.contracts  if yes else 0
        no_ct   = no.contracts   if no  else 0
        yes_avg = yes.avg_price  if yes else 0.0
        no_avg  = no.avg_price   if no  else 0.0

        if yes_ct == 0 and no_ct == 0:
            return "flat"

        parts = []
        if yes_ct:
            parts.append(f"{yes_ct}YES@{yes_avg:.2f}")
        if no_ct:
            parts.append(f"{no_ct}NO@{no_avg:.2f}")
        gross = " + ".join(parts)

        if yes_ct == 0 or no_ct == 0:
            return gross  # single-sided: no hedging math needed

        # Hedged pairs lock in a guaranteed PnL regardless of outcome.
        # A profitable boxed hedge has yes_avg + no_avg < 1.0.
        hedged = min(yes_ct, no_ct)
        locked_pnl = hedged * (1.0 - yes_avg - no_avg)

        net = yes_ct - no_ct
        if net > 0:
            net_str = f"net {net} YES"
        elif net < 0:
            net_str = f"net {abs(net)} NO"
        else:
            net_str = "net flat"

        return f"{gross} → {net_str} (locked {locked_pnl:+.2f})"

    def summary(self) -> dict:
        return {
            "open_markets": self.open_market_count,
            "total_exposure_dollars": round(self.total_exposure, 2),
            "positions": {
                k: {"side": p.side, "contracts": p.contracts, "avg_price": p.avg_price}
                for k, p in self._positions.items()
            },
        }

    # ------------------------------------------------------------------
    # Pre-trade check
    # ------------------------------------------------------------------

    def check_trade(
        self,
        ticker: str,
        side: str,
        price_dollars: float,
        contracts: int,
    ) -> tuple[bool, str]:
        """Returns (allowed, reason). reason is empty string when allowed.

        Guards against two limits:
          1. Dynamic _max_exposure (may be tightened by available-cash logic
             in the signal loop).
          2. Hard MAX_TOTAL_EXPOSURE_DOLLARS — even if _max_exposure drifted
             high somehow, this catches it.
        """
        if contracts <= 0:
            return False, "contracts must be > 0"

        if self.is_flat(ticker) and self.open_market_count >= self._max_markets:
            return False, f"max open markets ({self._max_markets}) reached"

        added_exposure = contracts * price_dollars
        projected = self.total_exposure + added_exposure

        if projected > self._max_exposure:
            return False, (
                f"would exceed exposure limit "
                f"(${self.total_exposure:.2f} + ${added_exposure:.2f} > ${self._max_exposure:.2f})"
            )

        # Hard-cap safety net: even if _max_exposure was mutated higher than the
        # config MAX by some bug, this always holds.
        if projected > MAX_TOTAL_EXPOSURE_DOLLARS:
            return False, (
                f"would exceed HARD cap "
                f"(${self.total_exposure:.2f} + ${added_exposure:.2f} > ${MAX_TOTAL_EXPOSURE_DOLLARS:.2f})"
            )

        return True, ""

    def capped_contracts(
        self,
        ticker: str,
        side: str,
        price_dollars: float,
        requested: int,
    ) -> int:
        """Returns the largest contract count within the dollar exposure limit."""
        remaining_exposure = max(0.0, self._max_exposure - self.total_exposure)
        by_exposure = int(remaining_exposure / price_dollars) if price_dollars > 0 else 0
        return max(0, min(requested, by_exposure))

    # ------------------------------------------------------------------
    # State updates
    # ------------------------------------------------------------------

    def record_fill(
        self,
        ticker: str,
        side: str,
        price_dollars: float,
        contracts: int,
    ) -> None:
        key = self._key(ticker, side)
        existing = self._positions.get(key)
        if existing is None:
            self._positions[key] = _MarketPosition(
                ticker=ticker,
                side=side,
                contracts=contracts,
                avg_price=price_dollars,
            )
        else:
            total = existing.contracts + contracts
            existing.avg_price = (
                (existing.avg_price * existing.contracts + price_dollars * contracts) / total
            )
            existing.contracts = total

    def close_market(self, ticker: str) -> None:
        """Remove all positions for a market (both sides) on settlement."""
        self._positions.pop(self._key(ticker, "yes"), None)
        self._positions.pop(self._key(ticker, "no"), None)

    def reduce_position(self, ticker: str, side: str, contracts: int) -> int:
        """
        Decrements an existing side by `contracts` and returns the quantity removed.
        Used when a same-side sell order is treated as filled.
        """
        if contracts <= 0:
            return 0

        key = self._key(ticker, side)
        existing = self._positions.get(key)
        if existing is None:
            return 0

        removed = min(contracts, existing.contracts)
        existing.contracts -= removed
        if existing.contracts == 0:
            self._positions.pop(key, None)
        return removed

    def sync_from_api(self, api_positions: list[dict]) -> None:
        live_keys: set[str] = set()
        for pos in api_positions:
            ticker = pos.get("market_id") or pos.get("ticker", "")
            yes_count = int(pos.get("position", 0))
            if yes_count > 0:
                key = self._key(ticker, "yes")
                self._positions[key] = _MarketPosition(
                    ticker=ticker, side="yes", contracts=yes_count, avg_price=0.5
                )
                live_keys.add(key)
            elif yes_count < 0:
                key = self._key(ticker, "no")
                self._positions[key] = _MarketPosition(
                    ticker=ticker, side="no", contracts=abs(yes_count), avg_price=0.5
                )
                live_keys.add(key)

        for key in list(self._positions.keys()):
            if key not in live_keys:
                self._positions.pop(key)

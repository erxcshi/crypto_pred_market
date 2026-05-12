"""
Main async trading engine.

Five concurrent coroutines:
  1. _coinbase_stream       - WebSocket trade feed -> SpotBuffer
  2. _kalshi_poller         - REST poll every 1s, overrides /markets bid/ask
                              with live orderbook, feeds KalshiMarketBuffer
  3. _signal_loop           - every 1s: compute features -> infer -> trade
                              (stop-loss first, then hold gate, then TP, then entry)
  4. _pnl_settlement_loop   - polls for expired markets and settles paper positions
  5. _status_loop           - logs buffer status every 15s

Run:
    python -m trading.engine
    python -m trading.engine --dry-run      # logs signals, no orders
"""


from __future__ import annotations

import argparse
import asyncio
import csv
import json
import logging
import time
from datetime import datetime, timezone
from pathlib import Path

import websockets

from trading.config import (
    COINS,
    COOLDOWN_SECONDS_AFTER_STOP,
    EXIT_THRESHOLD,
    EXTREME_MID_EARLY_TTC_SECONDS,
    EXTREME_MID_THRESHOLD,
    KELLY_BASE_EDGE,
    KELLY_ENABLED,
    N_CONTRACTS,
    N_CONTRACTS_MAX,
    MAX_TOTAL_EXPOSURE_DOLLARS,
    MIN_ORDER_PRICE,
    MIN_TIME_TO_CLOSE_SECONDS,
    MAX_TIME_TO_CLOSE_SECONDS,
    P_HAT_BIAS_MID_HIGH,
    P_HAT_BIAS_MID_LOW,
    P_HAT_BIAS_MID_ZONE,
    P_HAT_BIAS_TAIL_ZONE,
    POLL_INTERVAL_SECONDS,
    REL_STOP_ENABLED,
    REL_STOP_LOSS_FRACTION_OF_UPSIDE,
    REL_STOP_MAX_LOSS_ABS,
    REL_STOP_MIN_LOSS_ABS,
    REL_STOP_MIN_PRICE_FLOOR,
    SERIES_TICKER_TEMPLATE,
    SIGMA,
    TAKE_PROFIT_ENABLED,
    TAKE_PROFIT_EDGE_THRESHOLD,
    TAKE_PROFIT_MIN_PROFIT,
    TP_CLOSE_FRACTION,
    TP_EXPOSURE_UTILIZATION_THRESHOLD,
    TP_MAX_CONTRACTS_PER_TICK,
    TP_USE_BOOK_WALK,
    TAU,
)


def _kelly_size_cap(top_edge: float) -> int:
    """Kelly-ish: scale N_CONTRACTS by edge, clamp to [1, N_CONTRACTS_MAX]."""
    if not KELLY_ENABLED:
        return N_CONTRACTS
    if KELLY_BASE_EDGE <= 0:
        return N_CONTRACTS
    scaled = int(N_CONTRACTS * (top_edge / KELLY_BASE_EDGE))
    return max(1, min(scaled, N_CONTRACTS_MAX))


def _apply_p_hat_bias_correction(p_hat_raw: float, yes_mid: float) -> float:
    """
    Subtract a regime-dependent offset from p_hat to counteract the model's
    measured systematic overshoot vs market mid. Applied before edge calc
    so every downstream gate (entry/hold/TP) uses the calibrated value.
    """
    if P_HAT_BIAS_MID_LOW < yes_mid < P_HAT_BIAS_MID_HIGH:
        offset = P_HAT_BIAS_MID_ZONE
    else:
        offset = P_HAT_BIAS_TAIL_ZONE
    return max(0.001, min(0.999, p_hat_raw - offset))


def _relative_stop_loss_price(avg_entry: float) -> float:
    """
    Upside-proportional stop-sell price with floor/ceiling + a minimum stop
    price floor. The floor only applies when avg_entry is ABOVE the floor —
    otherwise an entry like 0.09 with floor=0.15 would have stop > entry,
    creating an instant-stop-out trap.

    Equation (from config.py REL_STOP_* constants):
      upside         = 1 - avg_entry
      raw_loss       = FRACTION_OF_UPSIDE × upside
      loss_tolerance = clip(raw_loss, MIN_LOSS_ABS, MAX_LOSS_ABS)
      stop_price     = max(0.01, avg_entry - loss_tolerance)
      if avg_entry > MIN_PRICE_FLOOR:
          stop_price = max(MIN_PRICE_FLOOR, stop_price)
    """
    if not REL_STOP_ENABLED or avg_entry <= 0:
        return -1.0
    upside = max(0.0, 1.0 - avg_entry)
    raw = REL_STOP_LOSS_FRACTION_OF_UPSIDE * upside
    loss_tol = max(REL_STOP_MIN_LOSS_ABS, min(raw, REL_STOP_MAX_LOSS_ABS))
    stop_price = max(0.01, avg_entry - loss_tol)
    # Only apply the price floor when the entry is above it. Below-floor
    # entries (e.g., 0.09 YES bought as a cheap arb) fall through and use the
    # raw subtraction — avoids the trap where floor > entry → instant stop.
    if avg_entry > REL_STOP_MIN_PRICE_FLOOR:
        stop_price = max(REL_STOP_MIN_PRICE_FLOOR, stop_price)
    return stop_price


async def _log_kalshi_balance(client, paper, tag: str = "event") -> None:
    """
    Fetch the real Kalshi balance and log it alongside the paper summary,
    so paper-vs-real drift is visible in real time at every event end.
    One REST call; swallows errors so a bad fetch doesn't break the caller.
    """
    try:
        real_balance = await client.get_balance()
        paper_value = paper.cash + paper.open_exposure
        drift = real_balance - paper_value
        log.info(
            f"[{tag}] kalshi_balance=${real_balance:.2f}  "
            f"paper_equity=${paper_value:.2f}  drift=${drift:+.2f}  "
            f"(cash=${paper.cash:.2f} + exposure=${paper.open_exposure:.2f})"
        )
    except Exception as e:
        log.debug(f"[{tag}] kalshi balance fetch failed: {e}")
from trading.features import FeatureEngine
from trading.kalshi_client import KalshiClient
from trading.model import BayesLogRegModel, FEATURE_NAMES
from trading.risk import RiskManager

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("engine")

COINBASE_WS_URL = "wss://ws-feed.exchange.coinbase.com"
KALSHI_RECONNECT_DELAY = 5

FEATURE_LOG_PATH = Path(__file__).resolve().parent / "feature_log.csv"
_CSV_COLS = ["timestamp", "ticker", "p_hat", "p_std", "edge_yes", "edge_no"] + FEATURE_NAMES
_csv_initialized = False


def _append_feature_row(ticker: str, feats: dict, p_hat: float, p_std: float) -> None:
    global _csv_initialized
    yes_mid = feats["yes_mid_dollars"]
    row = {
        "timestamp": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S"),
        "ticker": ticker,
        "p_hat": round(p_hat, 6),
        "p_std": round(p_std, 6),
        "edge_yes": round(p_hat - yes_mid, 6),
        "edge_no": round(yes_mid - p_hat, 6),
        **{k: round(v, 8) for k, v in feats.items()},
    }
    write_header = not _csv_initialized and not FEATURE_LOG_PATH.exists()
    with FEATURE_LOG_PATH.open("a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=_CSV_COLS, extrasaction="ignore")
        if write_header:
            writer.writeheader()
        writer.writerow(row)
    _csv_initialized = True


# ---------------------------------------------------------------------------
# Coinbase WebSocket stream
# ---------------------------------------------------------------------------

COINBASE_PRODUCT_IDS = ["BTC-USD", "ETH-USD", "XRP-USD", "SOL-USD"]
COINBASE_COIN_MAP = {"BTC-USD": "BTC", "ETH-USD": "ETH", "XRP-USD": "XRP", "SOL-USD": "SOL"}

async def _coinbase_stream(features: FeatureEngine) -> None:
    subscribe_msg = json.dumps({
        "type": "subscribe",
        "product_ids": COINBASE_PRODUCT_IDS,
        "channels": ["matches"],
    })

    while True:
        try:
            log.info("Connecting to Coinbase WebSocket...")
            async with websockets.connect(
                COINBASE_WS_URL, ping_interval=20, ping_timeout=20
            ) as ws:
                await ws.send(subscribe_msg)
                log.info("Coinbase feed active (BTC + ETH + XRP + SOL)")
                async for raw in ws:
                    msg = json.loads(raw)
                    if msg.get("type") != "match":
                        continue
                    product = msg.get("product_id", "")
                    coin = COINBASE_COIN_MAP.get(product)
                    if coin is None:
                        continue
                    price = float(msg["price"])
                    size = float(msg["size"])
                    side = msg.get("side", "buy")
                    if coin == "BTC":
                        features.update_spot(price, size, side)
                    else:
                        features.update_coin_price(coin, price)
        except asyncio.CancelledError:
            raise
        except Exception as e:
            log.warning(f"Coinbase WS error: {e}. Reconnecting in {KALSHI_RECONNECT_DELAY}s...")
            await asyncio.sleep(KALSHI_RECONNECT_DELAY)


# ---------------------------------------------------------------------------
# Kalshi REST poller
# ---------------------------------------------------------------------------

async def _orderbook_top_of_book(client: KalshiClient, ticker: str) -> dict | None:
    """
    Returns {yes_bid_dollars, yes_ask_dollars, no_bid_dollars, no_ask_dollars}
    from the live /orderbook snapshot, or None if the book is one-sided or the
    request fails. Derived: yes_ask = 1 - best_no_bid, no_ask = 1 - best_yes_bid.
    """
    try:
        ob = await client.get_orderbook(ticker)
        yes_bids = sorted(ob.get("yes", []), reverse=True)
        no_bids  = sorted(ob.get("no",  []), reverse=True)
        if not yes_bids or not no_bids:
            return None
        best_yes_bid = yes_bids[0][0]
        best_no_bid  = no_bids[0][0]
        return {
            "yes_bid_dollars": best_yes_bid,
            "yes_ask_dollars": 1.0 - best_no_bid,
            "no_bid_dollars":  best_no_bid,
            "no_ask_dollars":  1.0 - best_yes_bid,
        }
    except Exception:
        return None


async def _kalshi_poller(
    features: FeatureEngine,
    client: KalshiClient,
) -> None:
    """
    Polls all four coin Kalshi 15M markets every second, then overrides each
    market's cached /markets bid/ask with live /orderbook top-of-book so the
    rolling feature buffer (yes_mid_change_*, yes_spread_mean_*, etc.) is built
    from fresh quotes. Training features were built from /markets historically,
    but live inference benefits from the real book. Orderbook fetches run in
    parallel per coin to avoid serial latency.
    """
    _empty_warned: set[str] = set()
    while True:
        start = time.monotonic()
        for coin in COINS:
            series = SERIES_TICKER_TEMPLATE.format(coin=coin)
            try:
                markets = await client.get_markets(series, status="open")
                if not markets:
                    if coin not in _empty_warned:
                        log.warning(
                            f"Kalshi poller: no open markets for {coin} "
                            f"(series_ticker={series!r}) — cross-asset features will block BTC signals"
                        )
                        _empty_warned.add(coin)
                    continue
                _empty_warned.discard(coin)

                # Parallel orderbook fetches per market.
                tickers = [m.get("ticker", "") for m in markets]
                ob_results = await asyncio.gather(
                    *[_orderbook_top_of_book(client, t) if t else _noop_none() for t in tickers],
                    return_exceptions=False,
                )

                for market, ob_quotes in zip(markets, ob_results):
                    open_time = market.get("open_time", "")
                    close_time = market.get("close_time") or market.get("expiration_time", "")
                    if not open_time or not close_time:
                        continue
                    # Override /markets cached bid/ask with orderbook top-of-book.
                    # Falls through to /markets values on failure (or empty book).
                    if ob_quotes is not None:
                        market = {**market, **ob_quotes}
                    features.update_kalshi(coin, open_time, close_time, market)
            except asyncio.CancelledError:
                raise
            except Exception as e:
                log.warning(f"Kalshi poll error ({coin}): {e}")

        elapsed = time.monotonic() - start
        await asyncio.sleep(max(0.0, POLL_INTERVAL_SECONDS - elapsed))


async def _noop_none() -> None:
    """asyncio-compatible placeholder that returns None."""
    return None


# ---------------------------------------------------------------------------
# Paper PnL tracker (dry-run only)
# ---------------------------------------------------------------------------

PNL_LOG_PATH = Path(__file__).resolve().parent / "pnl_log.csv"
_PNL_COLS = [
    "entered_at", "settled_at",
    "market_open_time", "market_close_time",
    "ticker", "side", "price", "contracts", "result", "pnl",
]


INITIAL_BANKROLL = 1000.0
BANKROLL_FLOOR = 1.0


class PaperTrader:
    """Records fills and calculates PnL when markets settle."""

    def __init__(self, initial_cash: float = INITIAL_BANKROLL) -> None:
        # ticker -> list of fills (multiple entries allowed per ticker)
        self._open: dict[str, list[dict]] = {}
        self.cash: float = initial_cash
        self.realized_pnl: float = 0.0
        self.trade_count: int = 0
        self.win_count: int = 0

    @property
    def open_exposure(self) -> float:
        return sum(
            t["contracts"] * t["price"]
            for fills in self._open.values()
            for t in fills
        )

    @property
    def bankrupt(self) -> bool:
        # Only stop if both cash AND deployed capital are nearly gone
        return self.cash + self.open_exposure < BANKROLL_FLOOR

    def record_fill(
        self,
        ticker: str,
        side: str,
        price: float,
        contracts: int,
        market_open_time: str = "",
        market_close_time: str = "",
        entry_fee: float | None = None,
    ) -> None:
        """
        entry_fee: pass the real per-contract fee reported by Kalshi
        (parse_fill["fees_per_contract"]) to keep paper == real. If None,
        falls back to the variance-formula approximation.
        """
        if entry_fee is None:
            entry_fee = KalshiClient.taker_fee(price)
        self.cash -= contracts * (price + entry_fee)
        self._open.setdefault(ticker, []).append({
            "side": side,
            "price": price,
            "contracts": contracts,
            "entry_fee": entry_fee,
            "entered_at": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S"),
            "market_open_time": market_open_time,
            "market_close_time": market_close_time,
        })

    def record_exit(
        self,
        ticker: str,
        side: str,
        exit_price: float,
        max_contracts: int | None = None,
        tag: str = "exit",
        exit_fee: float | None = None,
    ) -> float:
        """
        Sell fills for ticker/side at exit_price, FIFO (oldest first).

        max_contracts: cap on contracts to close this call. None = close all.
        tag: result-column prefix for pnl_log.csv — 'exit' for stop-loss,
             'tp' for take-profit. Lets post-hoc analysis distinguish them.
        exit_fee: real per-contract taker fee from Kalshi response
             (parse_fill["fees_per_contract"]). If None, uses the variance
             formula. Pass the real fee to keep paper == real.

        Returns realized PnL for the contracts closed on this call.
        """
        fills = self._open.get(ticker, [])
        other_side = [f for f in fills if f["side"] != side]
        same_side  = [f for f in fills if f["side"] == side]
        if not same_side:
            return 0.0

        total_same_contracts = sum(f["contracts"] for f in same_side)
        to_close = total_same_contracts if max_contracts is None else min(max_contracts, total_same_contracts)
        if to_close <= 0:
            return 0.0

        if exit_fee is None:
            exit_fee = KalshiClient.taker_fee(exit_price)
        total_pnl = 0.0
        remaining_same: list[dict] = []

        for trade in same_side:
            if to_close <= 0:
                remaining_same.append(trade)
                continue

            close_n = min(trade["contracts"], to_close)
            entry_fee = trade.get("entry_fee", KalshiClient.taker_fee(trade["price"]))
            pnl = close_n * (exit_price - trade["price"] - entry_fee - exit_fee)
            # Cash: sell proceeds net of exit fee for the closed portion.
            self.cash += close_n * (exit_price - exit_fee)
            total_pnl += pnl
            self.realized_pnl += pnl
            self.trade_count += 1
            if pnl > 0:
                self.win_count += 1

            # Write a PnL row for the closed portion (partial or whole fill).
            closed_trade = {**trade, "contracts": close_n}
            self._write_pnl_row(ticker, closed_trade, f"{tag}@{exit_price:.3f}", pnl)

            to_close -= close_n
            if close_n < trade["contracts"]:
                leftover = {**trade, "contracts": trade["contracts"] - close_n}
                remaining_same.append(leftover)

        new_fills = other_side + remaining_same
        if new_fills:
            self._open[ticker] = new_fills
        else:
            self._open.pop(ticker, None)

        return total_pnl

    def settle(self, ticker: str, result: str) -> float | None:
        """Settle all fills for ticker. Returns total PnL or None if no open position."""
        fills = self._open.pop(ticker, None)
        if not fills:
            return None

        total_pnl = 0.0
        for trade in fills:
            won = (trade["side"] == result)
            entry_fee = trade.get("entry_fee", KalshiClient.taker_fee(trade["price"]))
            # Settlement is automatic: entry fee only, no second taker fee charged.
            if won:
                pnl = trade["contracts"] * (1.0 - trade["price"] - entry_fee)
            else:
                pnl = -trade["contracts"] * (trade["price"] + entry_fee)
            self.cash += trade["contracts"] * (trade["price"] + entry_fee) + pnl
            total_pnl += pnl
            self.realized_pnl += pnl
            self.trade_count += 1
            if won:
                self.win_count += 1
            self._write_pnl_row(ticker, trade, result, pnl)

        return total_pnl

    def _write_pnl_row(self, ticker: str, trade: dict, result: str, pnl: float) -> None:
        write_header = not PNL_LOG_PATH.exists() or PNL_LOG_PATH.stat().st_size == 0
        with PNL_LOG_PATH.open("a", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=_PNL_COLS)
            if write_header:
                writer.writeheader()
            writer.writerow({
                "entered_at": trade["entered_at"],
                "settled_at": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S"),
                "market_open_time": trade["market_open_time"],
                "market_close_time": trade["market_close_time"],
                "ticker": ticker,
                "side": trade["side"],
                "price": round(trade["price"], 4),
                "contracts": trade["contracts"],
                "result": result,
                "pnl": round(pnl, 4),
            })

    @property
    def open_tickers(self) -> list[str]:
        return list(self._open.keys())

    def close_time(self, ticker: str) -> str:
        """Return market_close_time for the ticker's first fill (all fills share the same window)."""
        fills = self._open.get(ticker, [])
        return fills[0]["market_close_time"] if fills else ""

    def summary(self) -> str:
        open_count = sum(len(v) for v in self._open.values())
        win_rate = self.win_count / self.trade_count if self.trade_count else 0.0
        return (
            f"cash=${self.cash:.2f}  PnL: ${self.realized_pnl:+.2f}  "
            f"trades={self.trade_count}  wins={self.win_count}  "
            f"win_rate={win_rate:.1%}  open_fills={open_count}"
        )


async def _pnl_settlement_loop(client: KalshiClient, paper: PaperTrader, risk: RiskManager) -> None:
    """Settles positions as soon as their market close_time passes, checking every 2 seconds."""
    while not paper.bankrupt:
        await asyncio.sleep(2)
        now = datetime.now(timezone.utc)
        for ticker in list(paper.open_tickers):
            close_str = paper.close_time(ticker)
            if not close_str:
                continue
            try:
                close_dt = datetime.fromisoformat(close_str.replace("Z", "+00:00"))
            except Exception:
                continue
            if now < close_dt:
                continue  # market still open
            try:
                market = await client.get_market(ticker)
                result = market.get("result", "")
                if result in ("yes", "no"):
                    pnl = paper.settle(ticker, result)
                    risk.close_market(ticker)
                    log.info(
                        f"SETTLED {ticker}  result={result.upper()}  "
                        f"pnl=${pnl:+.2f}  |  {paper.summary()}"
                    )
                    await _log_kalshi_balance(client, paper, tag=f"settle {ticker}")
                else:
                    log.debug(f"Market {ticker} past close but result not yet posted, retrying…")
            except Exception as e:
                log.debug(f"Settlement check failed for {ticker}: {e}")


# ---------------------------------------------------------------------------
# Signal & order loop
# ---------------------------------------------------------------------------

async def _signal_loop(
    features: FeatureEngine,
    model: BayesLogRegModel,
    client: KalshiClient,
    risk: RiskManager,
    dry_run: bool,
    paper: PaperTrader | None = None,
) -> None:
    """Every second: score each open BTC 15M market and trade if edge exists."""
    if paper is None:
        raise ValueError("PaperTrader instance is required for signal execution")

    _api_dumped = False

    while True:
        await asyncio.sleep(POLL_INTERVAL_SECONDS)

        if paper.bankrupt:
            log.warning(
                f"BANKROLL FLOOR HIT: cash=${paper.cash:.4f} - stopping simulation.  "
                f"{paper.summary()}"
            )
            return

        # Hard cap: total deployed never exceeds MAX_TOTAL_EXPOSURE_DOLLARS or available cash.
        risk._max_exposure = min(risk.total_exposure + paper.cash, MAX_TOTAL_EXPOSURE_DOLLARS)

        # Visibility check: if total_exposure is ever observed to exceed the
        # hard cap, something is wrong (stale positions, bug, drift). Log loud.
        if risk.total_exposure > MAX_TOTAL_EXPOSURE_DOLLARS + 0.01:
            log.warning(
                f"EXPOSURE CAP VIOLATED: total=${risk.total_exposure:.2f} > "
                f"MAX_TOTAL_EXPOSURE_DOLLARS=${MAX_TOTAL_EXPOSURE_DOLLARS:.2f}  "
                f"positions={risk.summary()['positions']}"
            )

        series = SERIES_TICKER_TEMPLATE.format(coin="BTC")
        try:
            markets = await client.get_markets(series, status="open")
        except Exception as e:
            log.warning(f"BTC market fetch failed: {e}")
            continue

        if not markets:
            log.warning(f"No markets returned for series_ticker={series!r} - check ticker format")
            continue

        if not _api_dumped:
            log.info(f"RAW MARKET API RESPONSE:\n{json.dumps(markets[0], indent=2)}")
            _api_dumped = True

        for market in markets:
            ticker = market.get("ticker", "")
            open_time = market.get("open_time", "")
            close_time = market.get("close_time") or market.get("expiration_time", "")

            if not ticker or not open_time or not close_time:
                log.debug(f"Skipping market missing fields: {market}")
                continue

            ttc = KalshiClient.market_time_to_close(market)
            log.debug(f"Market {ticker}  ttc={ttc:.0f}s")

            if ttc < 0:
                log.debug(f"  skip {ticker}: overdue (ttc={ttc:.0f}s)")
                continue
            if ttc > MAX_TIME_TO_CLOSE_SECONDS:
                log.debug(f"  skip {ticker}: ttc={ttc:.0f}s > max {MAX_TIME_TO_CLOSE_SECONDS}s")
                continue
            if ttc < MIN_TIME_TO_CLOSE_SECONDS:
                log.debug(f"  skip {ticker}: ttc={ttc:.0f}s < min {MIN_TIME_TO_CLOSE_SECONDS}s")
                continue

            feats, reason = features.compute_features("BTC", open_time, close_time, ttc)
            if feats is None:
                log.info(f"  skip {ticker}: {reason}")
                continue

            normalize_price = KalshiClient.normalize_price

            yes_ask = normalize_price(market.get("yes_ask_dollars") or market.get("yes_ask"))
            yes_bid = normalize_price(market.get("yes_bid_dollars") or market.get("yes_bid"))
            no_ask = normalize_price(market.get("no_ask_dollars") or market.get("no_ask"))
            no_bid = normalize_price(market.get("no_bid_dollars") or market.get("no_bid"))

            yes_ask = yes_ask or feats["yes_ask_dollars"]
            yes_bid = yes_bid or feats["yes_bid_dollars"]
            no_ask = no_ask or feats["no_ask_dollars"]
            no_bid = no_bid or feats["no_bid_dollars"]

            if no_ask <= 0:
                no_ask = max(1.0 - yes_bid, 0.01) if yes_bid > 0 else feats["no_ask_dollars"]
            if no_bid <= 0:
                no_bid = max(1.0 - yes_ask, 0.01) if yes_ask > 0 else feats["no_bid_dollars"]

            # /markets yes_bid/yes_ask lag the real book by many seconds on thin 15M
            # crypto markets. Pull the live orderbook snapshot and derive top-of-book
            # from it (best YES bid, best NO bid → YES ask = 1 - best NO bid).
            # The fetched orderbook is reused below for exit/entry pricing.
            market_mid = (yes_ask + yes_bid) / 2.0
            ob: dict | None = None
            try:
                ob = await client.get_orderbook(ticker)
                yes_bids = sorted(ob.get("yes", []), reverse=True)
                no_bids  = sorted(ob.get("no",  []), reverse=True)
                if yes_bids and no_bids:
                    yes_bid = yes_bids[0][0]
                    no_bid  = no_bids[0][0]
                    yes_ask = 1.0 - no_bid
                    no_ask  = 1.0 - yes_bid
            except Exception as e:
                log.debug(f"Orderbook fetch failed for {ticker}: {e} — using /markets quotes")

            yes_mid = (yes_ask + yes_bid) / 2.0
            if ob is not None and abs(yes_mid - market_mid) > 0.015:
                log.debug(
                    f"  {ticker} mid diverges: orderbook={yes_mid:.3f}  /markets={market_mid:.3f}"
                )
            last_price = normalize_price(market.get("last_price_dollars") or market.get("last_price"))

            feats["yes_ask_dollars"] = yes_ask
            feats["yes_bid_dollars"] = yes_bid
            feats["no_ask_dollars"] = no_ask
            feats["no_bid_dollars"] = no_bid
            feats["yes_mid_dollars"] = yes_mid
            feats["yes_spread_dollars"] = yes_ask - yes_bid
            if last_price > 0:
                feats["last_price_dollars"] = last_price

            fresh_spot = features.spot.latest_price
            if fresh_spot > 0:
                feats["btc_spot_price"] = fresh_spot
                floor_strike = float(market.get("floor_strike") or 0)
                if floor_strike > 0:
                    feats["distance_from_strike"] = fresh_spot - floor_strike

            p_hat_raw, p_std = model.predict(feats)
            _append_feature_row(ticker, feats, p_hat_raw, p_std)

            # Post-hoc calibration: counter model's measured YES overshoot.
            # Applied BEFORE edge calc so all downstream gates see corrected p_hat.
            p_hat = _apply_p_hat_bias_correction(p_hat_raw, yes_mid)

            edge_yes = p_hat - yes_ask
            edge_no = (1.0 - p_hat) - no_ask

            log.info(
                f"SCORE {ticker}: ttc={ttc:.0f}s  p={p_hat:.3f} (raw {p_hat_raw:.3f})+/-{p_std:.3f}  "
                f"mid={yes_mid:.3f}  edge_yes={edge_yes:+.3f}  edge_no={edge_no:+.3f}  "
                f"distance={feats['distance_from_strike']:.1f}  spot={feats['btc_spot_price']:.0f}  "
                f"pos=[{risk.net_position_str(ticker)}]"
            )

            # Exit checks run before SIGMA filter — we always want to exit a bad
            # position even if the model is currently uncertain.
            #
            # Ordering is intentional:
            #   1. STOP-LOSS (market-based) — pure risk guard, overrides model
            #      optimism. If yes_mid has crossed the threshold (or the scaled
            #      relative stop), exit regardless of what p_hat thinks.
            #   2. TAKE-PROFIT — scale out when edge is absorbed + in profit +
            #      capital pressured. Runs BEFORE the hold gate so that sitting
            #      winners get locked even when model still sees mild +EV at the
            #      ask. (The hold gate would otherwise block every TP firing.)
            #   3. HOLD gate (model-based) — if model still sees fresh entry +EV
            #      on the ASK and neither stop nor TP fired, ride the position.
            for held_side in ("yes", "no"):
                contracts_held = risk.current_side_contracts(ticker, held_side)
                if contracts_held <= 0:
                    continue

                # --- Stop-loss: market-based, overrides hold gate ---
                # Mode selection is mutually exclusive:
                #   REL_STOP_ENABLED=True  -> only the relative (per-entry scaled) stop fires.
                #   REL_STOP_ENABLED=False -> only the absolute EXIT_THRESHOLD stop fires.
                # Toggling REL_STOP_ENABLED alone swaps the stop-loss behavior with
                # no need to separately zero out EXIT_THRESHOLD.
                should_exit = False
                rel_stop_price = -1.0
                held_bid_for_stop = 0.0
                stop_mode = "rel" if REL_STOP_ENABLED else "abs"

                if REL_STOP_ENABLED:
                    # Relative stop: held_side bid below scaled stop price.
                    avg_entry_held = risk.current_side_avg_price(ticker, held_side)
                    if avg_entry_held > 0:
                        rel_stop_price = _relative_stop_loss_price(avg_entry_held)
                        _ob_for_stop = ob
                        try:
                            if _ob_for_stop is None:
                                _ob_for_stop = await client.get_orderbook(ticker)
                                ob = _ob_for_stop
                            held_bids_list = sorted(
                                _ob_for_stop.get(held_side, []), reverse=True
                            )
                            if held_bids_list:
                                held_bid_for_stop = held_bids_list[0][0]
                                if held_bid_for_stop < rel_stop_price:
                                    should_exit = True
                        except Exception:
                            # Fetch failed — skip stop for this tick rather than
                            # silently swap in the absolute stop.
                            pass
                else:
                    # Absolute stop: yes_mid beyond EXIT_THRESHOLD on our side.
                    should_exit = (
                        (held_side == "yes" and yes_mid < EXIT_THRESHOLD)
                        or (held_side == "no" and yes_mid > 1.0 - EXIT_THRESHOLD)
                    )

                if should_exit:
                    if stop_mode == "rel":
                        log.info(
                            f"STOP-REL {ticker} {held_side.upper()} "
                            f"bid={held_bid_for_stop:.3f} < rel_stop={rel_stop_price:.3f} "
                            f"(entry={risk.current_side_avg_price(ticker, held_side):.3f})"
                        )
                    else:
                        log.info(
                            f"STOP-ABS {ticker} {held_side.upper()} "
                            f"yes_mid={yes_mid:.3f} threshold={EXIT_THRESHOLD}"
                        )
                    try:
                        ob_exit = ob if ob is not None else await client.get_orderbook(ticker)
                        bids = sorted(ob_exit.get(held_side, []), reverse=True)
                        if not bids:
                            log.info(f"  skip exit {ticker}: no {held_side.upper()} bids available")
                            continue

                        # Walk bids descending to guarantee full fill: limit at the
                        # LOWEST walked bid (floor) lets Kalshi cross the book down,
                        # filling best-first. Simulated VWAP approximates the average
                        # fill price for paper accounting; real reconciliation would
                        # read the order response (separate TODO).
                        remaining = contracts_held
                        total_proceeds = 0.0
                        filled_sim = 0
                        floor_price = bids[0][0]
                        for bid_price_level, bid_size in bids:
                            take = min(int(bid_size), remaining)
                            if take <= 0:
                                continue
                            total_proceeds += take * bid_price_level
                            filled_sim += take
                            floor_price = bid_price_level
                            remaining -= take
                            if remaining <= 0:
                                break

                        vwap = total_proceeds / filled_sim if filled_sim > 0 else floor_price
                        top_bid = bids[0][0]

                        log.info(
                            f"EXIT {ticker}  sell {held_side.upper()} x{contracts_held}  "
                            f"limit@{floor_price:.3f}  vwap~{vwap:.3f}  top_bid={top_bid:.3f}  "
                            f"depth={filled_sim}/{contracts_held}  "
                            f"yes_mid={yes_mid:.3f}  p_hat={p_hat:.3f}  threshold={EXIT_THRESHOLD}"
                        )
                        if not dry_run:
                            order = await client.place_sell_order(
                                ticker, held_side, floor_price, contracts_held
                            )
                            fill = KalshiClient.parse_fill(order)
                            actual_closed = fill["filled_count"]
                            actual_price = fill["avg_price"] if actual_closed > 0 else vwap
                            real_exit_fee = fill["fees_per_contract"] if actual_closed > 0 else None
                            order_id = fill["order_id"]
                            log.info(
                                f"  fill  closed={actual_closed}/{contracts_held}  "
                                f"avg_price={actual_price:.4f}  "
                                f"fees/c=${real_exit_fee or 0:.4f}  "
                                f"remaining={fill['remaining_count']}  "
                                f"status={fill['status']}"
                            )
                            if actual_closed > 0:
                                pnl = paper.record_exit(
                                    ticker, held_side, actual_price,
                                    max_contracts=actual_closed,
                                    exit_fee=real_exit_fee,
                                )
                                closed_contracts = risk.reduce_position(
                                    ticker, held_side, actual_closed,
                                )
                            else:
                                pnl = 0.0
                                closed_contracts = 0

                            # Cancel any resting portion — stop-loss wants OUT, not rest.
                            if fill["remaining_count"] > 0 and order_id:
                                try:
                                    await client.cancel_order(order_id)
                                    log.info(
                                        f"  cancelled resting {fill['remaining_count']} "
                                        f"on {order_id}"
                                    )
                                except Exception as e:
                                    log.warning(f"cancel failed for {order_id}: {e}")
                        else:
                            # Dry-run: record at simulated VWAP.
                            pnl = paper.record_exit(ticker, held_side, vwap)
                            closed_contracts = risk.reduce_position(
                                ticker, held_side, contracts_held
                            )

                        # Start the per-side post-stop cooldown — block re-entries
                        # on THIS side for COOLDOWN_SECONDS_AFTER_STOP. The
                        # opposite side remains eligible (model may have flipped).
                        risk.record_stop_exit(ticker, held_side)
                        log.info(
                            f"  exit PnL=${pnl:+.4f}  closed={closed_contracts}  "
                            f"|  {paper.summary()}  pos=[{risk.net_position_str(ticker)}]  "
                            f"cooldown {held_side.upper()}={COOLDOWN_SECONDS_AFTER_STOP}s"
                        )
                        await _log_kalshi_balance(
                            client, paper, tag=f"exit {ticker} {held_side}"
                        )
                    except Exception as e:
                        log.warning(f"Exit order failed for {ticker} {held_side}: {e}")
                    continue  # position closed (or exit failed); skip hold/TP on this side

                # --- Take-profit: edge absorbed + in realized profit + capital pressure ---
                # Runs BEFORE the hold gate: lock realized gains when the market
                # has caught up to the model, even if model still sees mild +EV
                # at the ask. Close size = TP_CLOSE_FRACTION of held (floor 1),
                # capped by TP_MAX_CONTRACTS_PER_TICK. Default 1.0 = full close.
                if TAKE_PROFIT_ENABLED:
                    utilization = (
                        risk.total_exposure / MAX_TOTAL_EXPOSURE_DOLLARS
                        if MAX_TOTAL_EXPOSURE_DOLLARS > 0 else 0.0
                    )
                    if utilization > TP_EXPOSURE_UTILIZATION_THRESHOLD:
                        try:
                            ob_tp = ob if ob is not None else await client.get_orderbook(ticker)
                            tp_bids = sorted(ob_tp.get(held_side, []), reverse=True)
                            if tp_bids:
                                sell_bid = tp_bids[0][0]
                                fair = p_hat if held_side == "yes" else (1.0 - p_hat)
                                remaining_edge = fair - sell_bid

                                avg_entry = risk.current_side_avg_price(ticker, held_side)
                                entry_fee = KalshiClient.taker_fee(avg_entry)
                                exit_fee = KalshiClient.taker_fee(sell_bid)
                                realized_per_contract = (
                                    sell_bid - avg_entry - entry_fee - exit_fee
                                )

                                if (
                                    remaining_edge < TAKE_PROFIT_EDGE_THRESHOLD
                                    and realized_per_contract >= TAKE_PROFIT_MIN_PROFIT
                                ):
                                    tp_n_requested = max(1, int(contracts_held * TP_CLOSE_FRACTION))
                                    tp_n_requested = min(
                                        tp_n_requested, contracts_held, TP_MAX_CONTRACTS_PER_TICK
                                    )

                                    # Book walk with profit floor: include each bid
                                    # level only if (bid - avg_entry - entry_fee -
                                    # taker_fee(bid)) >= TAKE_PROFIT_MIN_PROFIT. Stop
                                    # the moment a level would make us sell at a loss
                                    # or below the profit threshold. This means some
                                    # requested contracts may remain un-sold if the
                                    # book thins out below profitable levels.
                                    if TP_USE_BOOK_WALK:
                                        tp_remaining = tp_n_requested
                                        tp_proceeds = 0.0
                                        tp_filled_sim = 0
                                        tp_floor_price = sell_bid
                                        for bid_p, bid_sz in tp_bids:
                                            level_exit_fee = KalshiClient.taker_fee(bid_p)
                                            level_profit = (
                                                bid_p - avg_entry - entry_fee - level_exit_fee
                                            )
                                            if level_profit < TAKE_PROFIT_MIN_PROFIT:
                                                break  # next level would eat profit
                                            take = min(int(bid_sz), tp_remaining)
                                            if take <= 0:
                                                continue
                                            tp_proceeds += take * bid_p
                                            tp_filled_sim += take
                                            tp_floor_price = bid_p
                                            tp_remaining -= take
                                            if tp_remaining <= 0:
                                                break
                                        if tp_filled_sim <= 0:
                                            # Top bid passes trigger but walked depth
                                            # is zero somehow (size 0 entries). Skip.
                                            raise RuntimeError("no profitable levels")
                                        tp_n = tp_filled_sim
                                        tp_limit_price = tp_floor_price
                                        tp_vwap = tp_proceeds / tp_filled_sim
                                    else:
                                        tp_n = tp_n_requested
                                        tp_limit_price = sell_bid
                                        tp_vwap = sell_bid

                                    log.info(
                                        f"TP {ticker} sell {held_side.upper()} "
                                        f"x{tp_n}/{contracts_held}  "
                                        f"limit@{tp_limit_price:.3f}  vwap~{tp_vwap:.3f}  "
                                        f"top_bid={sell_bid:.3f}  "
                                        f"p_hat={p_hat:.3f}  rem_edge={remaining_edge:+.3f}  "
                                        f"profit/contract(avg)=${realized_per_contract:+.4f}  "
                                        f"util={utilization:.2%}  "
                                        f"walk={'on' if TP_USE_BOOK_WALK else 'off'}"
                                    )
                                    if not dry_run:
                                        order = await client.place_sell_order(
                                            ticker, held_side, tp_limit_price, tp_n
                                        )
                                        fill = KalshiClient.parse_fill(order)
                                        actual_closed = fill["filled_count"]
                                        actual_price = (
                                            fill["avg_price"] if actual_closed > 0 else tp_vwap
                                        )
                                        real_exit_fee = (
                                            fill["fees_per_contract"] if actual_closed > 0 else None
                                        )
                                        order_id = fill["order_id"]
                                        log.info(
                                            f"  fill  closed={actual_closed}/{tp_n}  "
                                            f"avg_price={actual_price:.4f}  "
                                            f"fees/c=${real_exit_fee or 0:.4f}  "
                                            f"remaining={fill['remaining_count']}  "
                                            f"status={fill['status']}"
                                        )
                                        if actual_closed > 0:
                                            pnl = paper.record_exit(
                                                ticker, held_side, actual_price,
                                                max_contracts=actual_closed,
                                                tag="tp", exit_fee=real_exit_fee,
                                            )
                                            closed = risk.reduce_position(
                                                ticker, held_side, actual_closed,
                                            )
                                        else:
                                            pnl = 0.0
                                            closed = 0

                                        # Cancel resting portion — stale TP limits
                                        # are dangerous if market moves.
                                        if fill["remaining_count"] > 0 and order_id:
                                            try:
                                                await client.cancel_order(order_id)
                                                log.info(
                                                    f"  cancelled resting {fill['remaining_count']} "
                                                    f"on {order_id}"
                                                )
                                            except Exception as e:
                                                log.warning(
                                                    f"cancel failed for {order_id}: {e}"
                                                )
                                    else:
                                        # Dry-run: record at simulated VWAP of the walked bids
                                        pnl = paper.record_exit(
                                            ticker, held_side, tp_vwap,
                                            max_contracts=tp_n, tag="tp",
                                        )
                                        closed = risk.reduce_position(ticker, held_side, tp_n)

                                    log.info(
                                        f"  TP PnL=${pnl:+.4f}  closed={closed}  "
                                        f"remaining={contracts_held - closed}  "
                                        f"|  {paper.summary()}  "
                                        f"pos=[{risk.net_position_str(ticker)}]"
                                    )
                                    await _log_kalshi_balance(
                                        client, paper, tag=f"tp {ticker} {held_side}"
                                    )
                                    continue
                        except Exception as e:
                            log.warning(f"TP order failed for {ticker} {held_side}: {e}")

                # --- Hold gate (model-based) ---
                # If the entry signal is still live on this side (model would
                # still pay the ask), HOLD — avoids thrash where we'd
                # immediately re-buy at ceiling after any spurious exit.
                hold_ref_ask = yes_ask if held_side == "yes" else no_ask
                current_entry_edge = (
                    (p_hat - yes_ask) if held_side == "yes"
                    else ((1.0 - p_hat) - no_ask)
                )
                hold_fee = KalshiClient.taker_fee(hold_ref_ask)
                if current_entry_edge > TAU + hold_fee:
                    continue  # still +EV to enter this side; hold existing position

            if p_std > SIGMA:
                log.info(f"  skip {ticker}: p_std={p_std:.3f} > SIGMA={SIGMA} (too uncertain)")
                continue

            # Reject entries on already-pinned markets with time still on the clock.
            # Payoff asymmetry at extreme mids makes small miscalibration disproportionately
            # costly; exits are unaffected (they ran above).
            if (
                ttc > EXTREME_MID_EARLY_TTC_SECONDS
                and (yes_mid > EXTREME_MID_THRESHOLD or yes_mid < 1.0 - EXTREME_MID_THRESHOLD)
            ):
                log.info(
                    f"  skip {ticker}: yes_mid={yes_mid:.3f} extreme with ttc={ttc:.0f}s "
                    f"> {EXTREME_MID_EARLY_TTC_SECONDS}s (extreme-mid early-ttc filter)"
                )
                continue

            # Entry candidates: edge clearing TAU + per-side fee AND p_hat takes
            # the direction (YES only when p_hat > 0.53, NO only when p_hat < 0.47).
            # Dead zone 0.47-0.53 blocks low-conviction trades because the model
            # still has measurable calibration bias there (crude band-aid for a
            # calibration issue Kelly+rel_stop don't solve).
            # Per-side post-stop cooldown applied here: if we just stopped out
            # of YES, we block YES entries but leave NO free to fire.
            entry_candidates: list[tuple[str, float, float]] = []
            yes_entry_fee = KalshiClient.taker_fee(yes_ask)
            no_entry_fee = KalshiClient.taker_fee(no_ask)

            yes_in_cooldown, yes_cooldown_left = risk.in_stop_cooldown(ticker, "yes")
            no_in_cooldown, no_cooldown_left = risk.in_stop_cooldown(ticker, "no")

            if edge_yes > TAU + yes_entry_fee:
                if p_hat <= 0.53:
                    log.info(
                        f"  DEAD-ZONE block {ticker} YES: edge_yes={edge_yes:+.3f} "
                        f"clears TAU+fee but p_hat={p_hat:.3f} <= 0.53"
                    )
                elif yes_in_cooldown:
                    log.info(
                        f"  skip {ticker} yes: post-stop cooldown "
                        f"{yes_cooldown_left:.1f}s remaining (edge_yes={edge_yes:+.3f})"
                    )
                else:
                    entry_candidates.append(("yes", yes_ask, edge_yes))
            if edge_no > TAU + no_entry_fee:
                if p_hat >= 0.47:
                    log.info(
                        f"  DEAD-ZONE block {ticker} NO: edge_no={edge_no:+.3f} "
                        f"clears TAU+fee but p_hat={p_hat:.3f} >= 0.47"
                    )
                elif no_in_cooldown:
                    log.info(
                        f"  skip {ticker} no: post-stop cooldown "
                        f"{no_cooldown_left:.1f}s remaining (edge_no={edge_no:+.3f})"
                    )
                else:
                    entry_candidates.append(("no", no_ask, edge_no))

            if not entry_candidates:
                # Surface near-threshold misses so "why didn't we buy?" has an answer.
                best_edge = max(edge_yes, edge_no)
                if best_edge > TAU:
                    log.info(
                        f"  no entry {ticker}: edge_yes={edge_yes:+.3f} edge_no={edge_no:+.3f} "
                        f"p_hat={p_hat:.3f} — need edge>TAU+fee "
                        f"(yes:{TAU+yes_entry_fee:.3f} no:{TAU+no_entry_fee:.3f})"
                    )
                continue

            entry_candidates.sort(key=lambda item: item[2], reverse=True)

            for side, price, edge in entry_candidates:
                if price <= 0:
                    log.info(f"  skip {ticker}: {side} ask={price:.3f} (no liquidity)")
                    continue
                if price < MIN_ORDER_PRICE:
                    log.info(
                        f"  skip {ticker}: {side} ask={price:.3f} < "
                        f"MIN_ORDER_PRICE={MIN_ORDER_PRICE} (lottery ticket)"
                    )
                    continue

                # Kelly size cap from top-of-book edge (the `edge` in entry_candidates
                # is p_hat - yes_ask, pre book walk). Bigger edge → more contracts.
                size_cap = _kelly_size_cap(edge)

                # Kalshi yes_dollars/no_dollars are bids. Buying YES crosses NO bids,
                # and buying NO crosses YES bids.
                order_price = price
                n = size_cap
                try:
                    if ob is None:
                        ob = await client.get_orderbook(ticker)
                    ask_side = KalshiClient.effective_ask_side(ob, side)
                    if not ask_side:
                        log.info(f"  skip {ticker}: {side} ask side empty")
                        continue

                    # Walk book levels cheapest-first; accumulate contracts only at
                    # levels where that level's price still clears edge > TAU + fee(level).
                    # Fee is variance-proportional per Kalshi schedule, so each level needs
                    # its own check. Stop as soon as a level would consume the edge.
                    fair = p_hat if side == "yes" else (1.0 - p_hat)
                    n_valid = 0
                    order_price = price  # fallback to REST ask
                    for ask_price, ask_size in sorted(ask_side):
                        if fair - ask_price <= TAU + KalshiClient.taker_fee(ask_price):
                            break
                        n_valid += ask_size
                        order_price = ask_price

                    n_valid = int(min(n_valid, size_cap))
                    if n_valid <= 0:
                        log.info(
                            f"  skip {ticker}: no contracts with edge > TAU+fee "
                            f"(best_ask={sorted(ask_side)[0][0]:.3f}  fair={fair:.3f}  mkt_ask={price:.3f})"
                        )
                        continue

                    n = risk.capped_contracts(ticker, side, order_price, n_valid)
                    if n <= 0:
                        log.info(
                            f"  skip {ticker} {side}: exposure cap — "
                            f"used=${risk.total_exposure:.2f}/${risk._max_exposure:.2f} "
                            f"cannot add {n_valid} @ {order_price:.3f}"
                        )
                        continue

                    edge = fair - order_price
                except Exception as e:
                    log.debug(f"Orderbook fetch failed for {ticker} {side}: {e} - using raw ask price")
                    n = risk.capped_contracts(ticker, side, order_price, size_cap)
                    if n <= 0:
                        log.info(
                            f"  skip {ticker} {side}: exposure cap (fallback path) — "
                            f"used=${risk.total_exposure:.2f}/${risk._max_exposure:.2f} "
                            f"cannot add {N_CONTRACTS} @ {order_price:.3f}"
                        )
                        continue

                ok, reason = risk.check_trade(ticker, side, order_price, n)
                if not ok:
                    log.info(f"Risk block [{ticker}] {side}: {reason}")
                    continue

                log.info(
                    f"SIGNAL  {ticker}  BUY {side.upper():3s}  "
                    f"n={n}  kelly_cap={size_cap}  ask={price:.3f}  order_price={order_price:.3f}  edge={edge:+.3f}  "
                    f"p={p_hat:.3f}+/-{p_std:.4f}  before=[{risk.net_position_str(ticker)}]"
                )

                if not dry_run:
                    try:
                        order = await client.place_limit_order(ticker, side, order_price, n)
                        fill = KalshiClient.parse_fill(order)
                        filled = fill["filled_count"]
                        actual_price = fill["avg_price"] if filled > 0 else order_price
                        real_fee = fill["fees_per_contract"] if filled > 0 else None
                        order_id = fill["order_id"] or "?"
                        log.info(
                            f"ORDER PLACED  {ticker}  {side}  order_id={order_id}  "
                            f"filled={filled}/{n}  avg_price={actual_price:.4f}  "
                            f"fees/c=${real_fee:.4f}  remaining={fill['remaining_count']}  "
                            f"status={fill['status']}"
                            if filled > 0 else
                            f"ORDER PLACED  {ticker}  {side}  order_id={order_id}  "
                            f"filled=0/{n}  status={fill['status']}"
                        )

                        if filled > 0:
                            risk.record_fill(ticker, side, actual_price, filled)
                            paper.record_fill(
                                ticker, side, actual_price, filled,
                                open_time, close_time, entry_fee=real_fee,
                            )

                        # Cancel any resting remainder to avoid dangling orders.
                        if fill["remaining_count"] > 0 and order_id and order_id != "?":
                            try:
                                await client.cancel_order(order_id)
                                log.info(
                                    f"  cancelled resting {fill['remaining_count']} "
                                    f"on {order_id}"
                                )
                            except Exception as e:
                                log.warning(f"cancel failed for {order_id}: {e}")
                    except Exception as e:
                        log.error(f"Order placement failed for {ticker} {side}: {e}")
                        continue
                else:
                    log.info("  (dry-run - paper fill recorded)")
                    risk.record_fill(ticker, side, order_price, n)
                    paper.record_fill(ticker, side, order_price, n, open_time, close_time)

                log.info(f"  post-entry position [{risk.net_position_str(ticker)}]")
                if not dry_run:
                    await _log_kalshi_balance(
                        client, paper, tag=f"entry {ticker} {side}"
                    )


# ---------------------------------------------------------------------------
# Periodic risk sync
# ---------------------------------------------------------------------------

async def _status_loop(features: FeatureEngine) -> None:
    while True:
        await asyncio.sleep(15)
        s = features.buffer_status()
        log.info(
            f"Buffer status: spot={s['spot_bars']} bars @ ${s['spot_price']:.0f} | "
            + " | ".join(
                f"{coin}: {s[f'{coin}_events']}ev/{s[f'{coin}_samples']}smp"
                for coin in ("BTC", "ETH", "XRP", "SOL")
            )
        )


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

async def main(dry_run: bool = False) -> None:
    log.info(f"Starting trading engine  (dry_run={dry_run})")
    log.info(f"TAU={TAU}  SIGMA={SIGMA}  MAX_CONTRACTS={N_CONTRACTS}")

    model = BayesLogRegModel.load()
    log.info(f"Model loaded: {model._draws.shape[0]} posterior draws")

    features = FeatureEngine()
    risk = RiskManager()
    paper = PaperTrader()

    async with KalshiClient() as client:
        balance = await client.get_balance()
        log.info(f"Account balance: ${balance:.2f}")
        if balance > 0:
            paper.cash = balance
            log.info(
                "Paper trader initialized from live balance only; "
                "startup does not sync open positions or orders"
            )

        await asyncio.gather(
            _coinbase_stream(features),
            _kalshi_poller(features, client),
            _signal_loop(features, model, client, risk, dry_run, paper),
            _pnl_settlement_loop(client, paper, risk),
            _status_loop(features),
        )
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Kalshi crypto prediction market trader")
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Log signals without placing real orders",
    )
    args = parser.parse_args()

    try:
        asyncio.run(main(dry_run=args.dry_run))
    except KeyboardInterrupt:
        log.info("Shutdown requested")

"""
Kalshi trading API client with RSA-key authentication.

Auth flow (Kalshi docs):
  message  = str(timestamp_ms) + method.upper() + path
  sig      = RSA-PKCS1v15-SHA256(private_key, message.encode())
  headers  = {
      "KALSHI-ACCESS-KEY":       key_id,
      "KALSHI-ACCESS-TIMESTAMP": str(timestamp_ms),
      "KALSHI-ACCESS-SIGNATURE": base64(sig),
  }

Set KALSHI_KEY_ID and KALSHI_PRIVATE_KEY_PATH in your .env file.
"""
from __future__ import annotations

import base64
import os
import time
import uuid
from pathlib import Path
from typing import Any, Optional

import aiohttp
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import padding
from dotenv import load_dotenv

from trading.config import KALSHI_BASE_URL, PROJECT_ROOT

load_dotenv(PROJECT_ROOT / ".env")


def _load_private_key(pem_path: str):
    path = Path(pem_path)
    with path.open("rb") as f:
        return serialization.load_pem_private_key(f.read(), password=None)


def _make_auth_headers(key_id: str, private_key, method: str, path: str) -> dict:
    ts_ms = str(int(time.time() * 1000))
    message = (ts_ms + method.upper() + "/trade-api/v2" + path).encode()
    sig = private_key.sign(
        message,
        padding.PSS(
            mgf=padding.MGF1(hashes.SHA256()),
            salt_length=padding.PSS.DIGEST_LENGTH,
        ),
        hashes.SHA256(),
    )
    return {
        "KALSHI-ACCESS-KEY": key_id,
        "KALSHI-ACCESS-TIMESTAMP": ts_ms,
        "KALSHI-ACCESS-SIGNATURE": base64.b64encode(sig).decode(),
        "Content-Type": "application/json",
    }


class KalshiClient:
    """Async Kalshi trading API client."""

    def __init__(
        self,
        key_id: Optional[str] = None,
        private_key_path: Optional[str] = None,
        base_url: str = KALSHI_BASE_URL,
    ) -> None:
        self._key_id = key_id or os.getenv("KALSHI_KEY_ID", "")
        pem_path = private_key_path or os.getenv("KALSHI_PRIVATE_KEY_PATH", "")
        self._private_key = _load_private_key(pem_path) if pem_path else None
        self._base_url = base_url.rstrip("/")
        self._session: Optional[aiohttp.ClientSession] = None

    async def __aenter__(self) -> "KalshiClient":
        self._session = aiohttp.ClientSession()
        return self

    async def __aexit__(self, *args) -> None:
        if self._session:
            await self._session.close()

    def _headers(self, method: str, path: str) -> dict:
        if not self._key_id or not self._private_key:
            raise RuntimeError(
                "Missing KALSHI_KEY_ID or KALSHI_PRIVATE_KEY_PATH in environment."
            )
        return _make_auth_headers(self._key_id, self._private_key, method, path)

    async def _get(self, path: str, params: dict | None = None) -> Any:
        url = self._base_url + path
        headers = self._headers("GET", path)
        async with self._session.get(url, headers=headers, params=params) as resp:
            if not resp.ok:
                body = await resp.text()
                raise RuntimeError(f"Kalshi {resp.status} on GET {path}: {body}")
            return await resp.json()

    async def _post(self, path: str, body: dict) -> Any:
        url = self._base_url + path
        headers = self._headers("POST", path)
        async with self._session.post(url, headers=headers, json=body) as resp:
            if not resp.ok:
                body_text = await resp.text()
                raise RuntimeError(f"Kalshi {resp.status} on POST {path}: {body_text}")
            return await resp.json()

    async def _delete(self, path: str) -> Any:
        url = self._base_url + path
        headers = self._headers("DELETE", path)
        async with self._session.delete(url, headers=headers) as resp:
            if not resp.ok:
                body = await resp.text()
                raise RuntimeError(f"Kalshi {resp.status} on DELETE {path}: {body}")
            return await resp.json()

    # ------------------------------------------------------------------
    # Market data
    # ------------------------------------------------------------------

    async def get_markets(
        self,
        series_ticker: str,
        status: str = "open",
        limit: int = 100,
    ) -> list[dict]:
        data = await self._get(
            "/markets",
            params={"series_ticker": series_ticker, "status": status, "limit": limit},
        )
        return data.get("markets", [])

    async def get_market(self, ticker: str) -> dict:
        data = await self._get(f"/markets/{ticker}")
        return data.get("market", {})

    async def get_orderbook(self, ticker: str) -> dict:
        """Returns {yes: [[price, size], ...], no: [[price, size], ...]} in dollar prices."""
        data = await self._get(f"/markets/{ticker}/orderbook")
        ob = data.get("orderbook_fp", {})
        return {
            "yes": [[float(p), float(s)] for p, s in ob.get("yes_dollars", [])],
            "no":  [[float(p), float(s)] for p, s in ob.get("no_dollars",  [])],
        }

    @staticmethod
    def effective_ask_side(orderbook: dict, side: str) -> list[list]:
        """
        Kalshi's orderbook contains BIDS for each side (yes_dollars = YES bids,
        no_dollars = NO bids).  To BUY YES you cross NO bids; to BUY NO you cross
        YES bids.  The ask price for each level = 1 - opposite_bid_price.

        Returns [[ask_price, size], ...] sorted ascending (cheapest ask first),
        ready for vwap_for_quantity / ceiling_price_for_quantity.
        """
        opposite = "no" if side == "yes" else "yes"
        opp_bids = sorted(orderbook.get(opposite, []), reverse=True)  # best bid first
        return [[1.0 - p, s] for p, s in opp_bids]  # complement → ascending asks

    @staticmethod
    def depth_at_price(orderbook_side: list[list], limit_price: float) -> float:
        """Sum of available contracts on one side at prices <= limit_price."""
        return sum(size for price, size in orderbook_side if price <= limit_price)

    @staticmethod
    def vwap_for_quantity(orderbook_side: list[list], quantity: float) -> float:
        """
        Volume-weighted average price to fill `quantity` contracts, walking the book
        ascending. Returns inf if the book cannot fill the full quantity.
        """
        remaining = quantity
        total_cost = 0.0
        for price, size in sorted(orderbook_side):  # ascending: cheapest first
            fill = min(remaining, size)
            total_cost += fill * price
            remaining -= fill
            if remaining <= 0:
                break
        if remaining > 0:
            return float("inf")
        return total_cost / quantity

    @staticmethod
    def ceiling_price_for_quantity(orderbook_side: list[list], quantity: float) -> float:
        """
        Highest ask price level touched when filling `quantity` contracts.
        Using this as the limit order price guarantees immediate full fill.
        Returns inf if the book cannot fill the full quantity.
        """
        remaining = quantity
        ceiling = float("inf")
        for price, size in sorted(orderbook_side):
            ceiling = price
            remaining -= size
            if remaining <= 0:
                break
        if remaining > 0:
            return float("inf")
        return ceiling

    # ------------------------------------------------------------------
    # Account
    # ------------------------------------------------------------------

    async def get_balance(self) -> float:
        """Returns available balance in dollars."""
        data = await self._get("/portfolio/balance")
        cents = data.get("balance", 0)
        return cents / 100.0

    async def get_positions(self) -> list[dict]:
        data = await self._get("/portfolio/positions")
        return data.get("market_positions", [])

    async def get_open_orders(self) -> list[dict]:
        data = await self._get("/portfolio/orders", params={"status": "resting"})
        return data.get("orders", [])

    # ------------------------------------------------------------------
    # Order management
    # ------------------------------------------------------------------

    async def place_limit_order(
        self,
        ticker: str,
        side: str,            # "yes" or "no"
        price_dollars: float, # e.g. 0.47  →  stored as 47 cents internally
        count: int,           # number of contracts
    ) -> dict:
        """
        Places a limit BUY order at the given price.
        price_dollars: 0–1 (will be multiplied by 100 and rounded to cents).
        Returns the order object from Kalshi.
        """
        price_cents = max(1, min(99, round(price_dollars * 100)))
        body = {
            "ticker": ticker,
            "action": "buy",
            "side": side,
            "type": "limit",
            "count": count,
            "client_order_id": str(uuid.uuid4()),
        }
        if side == "yes":
            body["yes_price"] = price_cents
        else:
            body["no_price"] = price_cents

        return await self._post("/portfolio/orders", body)

    async def place_sell_order(
        self,
        ticker: str,
        side: str,
        price_dollars: float,
        count: int,
    ) -> dict:
        """
        Places a limit SELL order. price_dollars is the minimum you'll accept.
        Use the current bid price to get immediate fill.
        """
        price_cents = max(1, min(99, round(price_dollars * 100)))
        body = {
            "ticker": ticker,
            "action": "sell",
            "side": side,
            "type": "limit",
            "count": count,
            "client_order_id": str(uuid.uuid4()),
        }
        if side == "yes":
            body["yes_price"] = price_cents
        else:
            body["no_price"] = price_cents
        return await self._post("/portfolio/orders", body)

    async def cancel_order(self, order_id: str) -> dict:
        return await self._delete(f"/portfolio/orders/{order_id}")

    async def cancel_all_open_orders(self) -> None:
        orders = await self.get_open_orders()
        for order in orders:
            order_id = order.get("order_id") or order.get("id")
            if order_id:
                try:
                    await self.cancel_order(order_id)
                    print(f"  cancelled order {order_id}")
                except Exception as e:
                    print(f"  cancel failed for {order_id}: {e}")

    # ------------------------------------------------------------------
    # Convenience
    # ------------------------------------------------------------------

    @staticmethod
    def normalize_price(raw_price) -> float:
        """Convert from Kalshi API (may be cents int OR dollar float) to [0,1]."""
        if raw_price is None:
            return 0.5
        v = float(raw_price)
        return v / 100.0 if v > 1.0 else v

    @staticmethod
    def taker_fee(price_dollars: float) -> float:
        """
        Kalshi taker fee per contract, in dollars. Variance-proportional:
        ceil(7 * P * (1-P)) cents, max 2¢ at P=0.50, min 1¢ at extremes.
        See https://kalshi.com/fee-schedule.
        """
        import math
        p = max(0.0, min(1.0, float(price_dollars)))
        return math.ceil(7.0 * p * (1.0 - p)) / 100.0

    @staticmethod
    def parse_fill(order_response: dict) -> dict:
        """
        Parse a POST /portfolio/orders response into reconciled fill values.

        Response envelope is {"order": {...}}. Relevant fields on the inner order:
          fill_count_fp, remaining_count_fp, initial_count_fp (string counts)
          taker_fill_cost_dollars, maker_fill_cost_dollars (dollars)
          taker_fees_dollars, maker_fees_dollars (dollars)
          status, order_id

        Returns:
          {
            "filled_count": int,           # total contracts that actually filled
            "remaining_count": int,        # contracts still resting
            "avg_price": float,            # (fill_cost) / filled_count; 0 if none
            "fees_per_contract": float,    # total fees / filled_count; 0 if none
            "status": str,                 # order status
            "order_id": str,               # for cancel-remaining
          }

        If fields are missing (e.g. schema change or dry-run mimic), returns
        zeros with status="unknown" and leaves reconciliation to the caller.
        """
        order = order_response.get("order", order_response) or {}

        def _to_int(x, default=0):
            try:
                return int(float(x))
            except (TypeError, ValueError):
                return default

        def _to_float(x, default=0.0):
            try:
                return float(x)
            except (TypeError, ValueError):
                return default

        filled = _to_int(order.get("fill_count_fp", 0))
        remaining = _to_int(order.get("remaining_count_fp", 0))
        taker_cost = _to_float(order.get("taker_fill_cost_dollars", 0.0))
        maker_cost = _to_float(order.get("maker_fill_cost_dollars", 0.0))
        taker_fees = _to_float(order.get("taker_fees_dollars", 0.0))
        maker_fees = _to_float(order.get("maker_fees_dollars", 0.0))

        total_cost = taker_cost + maker_cost
        total_fees = taker_fees + maker_fees
        avg_price = total_cost / filled if filled > 0 else 0.0
        fees_per_contract = total_fees / filled if filled > 0 else 0.0

        return {
            "filled_count": filled,
            "remaining_count": remaining,
            "avg_price": avg_price,
            "fees_per_contract": fees_per_contract,
            "status": order.get("status", "unknown"),
            "order_id": order.get("order_id", ""),
        }

    @staticmethod
    def market_time_to_close(market: dict) -> float:
        """Seconds until market closes, from current UTC time."""
        import datetime as _dt
        close_str = market.get("close_time") or market.get("expiration_time")
        if not close_str:
            return 0.0
        try:
            close_dt = _dt.datetime.fromisoformat(close_str.replace("Z", "+00:00"))
            now = _dt.datetime.now(_dt.timezone.utc)
            return (close_dt - now).total_seconds()
        except Exception:
            return 0.0

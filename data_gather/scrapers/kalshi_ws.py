"""
Kalshi WebSocket scraper.

Streams orderbook deltas + executed trades for KX{coin}15M crypto markets and
writes one row per event to Supabase (or CSV in test mode).

Two tables:
  kalshi_orderbook_deltas — snapshots exploded into rows + every delta
  kalshi_trades           — one row per executed trade (trade_id PK)

Run:
  python -m data_gather.scrapers.kalshi_ws

Env:
  KALSHI_KEY_ID, KALSHI_PRIVATE_KEY_PATH must be set (same as trading platform).
"""
from __future__ import annotations

import asyncio
import base64
import json
import os
import random
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import aiohttp
import websockets
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import padding
from dotenv import load_dotenv

from config import create_data_sink, PROJECT_ROOT

load_dotenv(PROJECT_ROOT / '.env')

WS_URL = 'wss://api.elections.kalshi.com/trade-api/ws/v2'
WS_SIGN_PATH = '/trade-api/ws/v2'
REST_BASE = 'https://api.elections.kalshi.com/trade-api/v2'

COINS = ['BTC', 'ETH', 'XRP', 'SOL']
DISCOVERY_INTERVAL_SECONDS = 60
BATCH_SIZE = 100
FLUSH_INTERVAL_SECONDS = 5
RECONNECT_MAX_BACKOFF = 60

DELTAS_TABLE = 'kalshi_orderbook_deltas'
TRADES_TABLE = 'kalshi_trades'

DELTAS_FIELDS = ['curr_time', 'ticker', 'coin', 'market_id', 'ts_ms',
                 'event_type', 'side', 'price', 'delta_size']
TRADES_FIELDS = ['curr_time', 'ticker', 'coin', 'trade_id', 'ts_ms',
                 'yes_price', 'no_price', 'count', 'taker_outcome_side',
                 'taker_book_side']


def _load_private_key(pem_path: str):
    with Path(pem_path).open('rb') as f:
        return serialization.load_pem_private_key(f.read(), password=None)


def _sign_headers(key_id: str, private_key, method: str, path: str) -> dict:
    ts_ms = str(int(time.time() * 1000))
    message = (ts_ms + method.upper() + path).encode()
    sig = private_key.sign(
        message,
        padding.PSS(mgf=padding.MGF1(hashes.SHA256()),
                    salt_length=padding.PSS.DIGEST_LENGTH),
        hashes.SHA256(),
    )
    return {
        'KALSHI-ACCESS-KEY': key_id,
        'KALSHI-ACCESS-TIMESTAMP': ts_ms,
        'KALSHI-ACCESS-SIGNATURE': base64.b64encode(sig).decode(),
    }


def _to_float(x, default=None):
    try:
        return float(x)
    except (TypeError, ValueError):
        return default


def _to_int(x, default=None):
    try:
        return int(float(x))
    except (TypeError, ValueError):
        return default


async def _discover_tickers(session: aiohttp.ClientSession) -> dict[str, str]:
    """Returns {ticker: coin} for all currently open KX{coin}15M markets."""
    result: dict[str, str] = {}
    for coin in COINS:
        url = f'{REST_BASE}/markets'
        params = {'series_ticker': f'KX{coin}15M', 'status': 'open', 'limit': 100}
        try:
            async with session.get(url, params=params,
                                   timeout=aiohttp.ClientTimeout(total=15)) as resp:
                resp.raise_for_status()
                data = await resp.json()
                for m in data.get('markets', []):
                    t = m.get('ticker')
                    if t:
                        result[t] = coin
        except Exception as e:
            print(f'kalshi_ws: discovery failed for {coin}: {type(e).__name__}: {e}')
    return result


def _snapshot_rows(msg: dict, ticker_to_coin: dict[str, str], curr_time: str) -> list[dict]:
    ticker = msg.get('market_ticker')
    coin = ticker_to_coin.get(ticker, '')
    market_id = msg.get('market_id')
    ts_ms = _to_int(msg.get('ts_ms'))

    rows = []
    for side, key in (('yes', 'yes_dollars_fp'), ('no', 'no_dollars_fp')):
        for level in msg.get(key) or []:
            if len(level) < 2:
                continue
            price = _to_float(level[0])
            size = _to_float(level[1])
            if price is None or size is None:
                continue
            rows.append({
                'curr_time': curr_time,
                'ticker': ticker,
                'coin': coin,
                'market_id': market_id,
                'ts_ms': ts_ms,
                'event_type': 'snapshot',
                'side': side,
                'price': price,
                'delta_size': size,
            })
    return rows


def _delta_row(msg: dict, ticker_to_coin: dict[str, str], curr_time: str) -> Optional[dict]:
    ticker = msg.get('market_ticker')
    price = _to_float(msg.get('price_dollars'))
    delta = _to_float(msg.get('delta_fp'))
    side = msg.get('side')
    if ticker is None or price is None or delta is None or side not in ('yes', 'no'):
        return None
    return {
        'curr_time': curr_time,
        'ticker': ticker,
        'coin': ticker_to_coin.get(ticker, ''),
        'market_id': msg.get('market_id'),
        'ts_ms': _to_int(msg.get('ts_ms')),
        'event_type': 'delta',
        'side': side,
        'price': price,
        'delta_size': delta,
    }


def _trade_row(msg: dict, ticker_to_coin: dict[str, str], curr_time: str) -> Optional[dict]:
    trade_id = msg.get('trade_id')
    ticker = msg.get('market_ticker')
    if not trade_id or not ticker:
        return None
    return {
        'curr_time': curr_time,
        'ticker': ticker,
        'coin': ticker_to_coin.get(ticker, ''),
        'trade_id': trade_id,
        'ts_ms': _to_int(msg.get('ts_ms')),
        'yes_price': _to_float(msg.get('yes_price_dollars')),
        'no_price': _to_float(msg.get('no_price_dollars')),
        'count': _to_float(msg.get('count_fp')),
        'taker_outcome_side': msg.get('taker_outcome_side') or msg.get('taker_side'),
        'taker_book_side': msg.get('taker_book_side'),
    }


class _Buffer:
    def __init__(self, table: str, csv_filename: str, fields: list[str]):
        self.table = table
        self.csv_filename = csv_filename
        self.fields = fields
        self.rows: list[dict] = []
        self.last_flush = time.monotonic()

    def add(self, row: dict) -> None:
        self.rows.append(row)

    def extend(self, rows: list[dict]) -> None:
        self.rows.extend(rows)

    def should_flush(self) -> bool:
        if not self.rows:
            return False
        return (len(self.rows) >= BATCH_SIZE
                or time.monotonic() - self.last_flush >= FLUSH_INTERVAL_SECONDS)

    async def flush(self, data_sink) -> None:
        if not self.rows:
            return
        ok = await data_sink.submit_rows(
            self.table, self.rows,
            csv_filename=self.csv_filename, fieldnames=self.fields,
        )
        if ok:
            print(f'kalshi_ws: inserted {len(self.rows)} rows -> {self.table}')
            self.rows.clear()
            self.last_flush = time.monotonic()


async def _discovery_loop(session: aiohttp.ClientSession, ticker_to_coin: dict[str, str],
                          ws, next_cmd_id: list[int], active_sids: list[int]) -> None:
    while True:
        await asyncio.sleep(DISCOVERY_INTERVAL_SECONDS)
        try:
            fresh = await _discover_tickers(session)
            new_tickers = sorted(set(fresh) - set(ticker_to_coin))
            stale_tickers = sorted(set(ticker_to_coin) - set(fresh))

            if new_tickers and active_sids:
                ticker_to_coin.update({t: fresh[t] for t in new_tickers})
                next_cmd_id[0] += 1
                await ws.send(json.dumps({
                    'id': next_cmd_id[0],
                    'cmd': 'update_subscription',
                    'params': {
                        'sids': list(active_sids),
                        'market_tickers': new_tickers,
                        'action': 'add_markets',
                    },
                }))
                print(f'kalshi_ws: +{len(new_tickers)} tickers: {new_tickers}')

            if stale_tickers and active_sids:
                for t in stale_tickers:
                    ticker_to_coin.pop(t, None)
                next_cmd_id[0] += 1
                await ws.send(json.dumps({
                    'id': next_cmd_id[0],
                    'cmd': 'update_subscription',
                    'params': {
                        'sids': list(active_sids),
                        'market_tickers': stale_tickers,
                        'action': 'delete_markets',
                    },
                }))
                print(f'kalshi_ws: -{len(stale_tickers)} tickers: {stale_tickers}')
        except Exception as e:
            print(f'kalshi_ws: discovery cycle failed: {type(e).__name__}: {e}')


async def _periodic_flush_loop(data_sink, buffers: list[_Buffer]) -> None:
    while True:
        await asyncio.sleep(1)
        for buf in buffers:
            if buf.should_flush():
                await buf.flush(data_sink)


async def _consume(ws, session, ticker_to_coin, data_sink) -> None:
    deltas = _Buffer(DELTAS_TABLE, 'kalshi_orderbook_deltas_test.csv', DELTAS_FIELDS)
    trades = _Buffer(TRADES_TABLE, 'kalshi_trades_test.csv', TRADES_FIELDS)
    buffers = [deltas, trades]

    tickers = sorted(ticker_to_coin.keys())
    if not tickers:
        raise RuntimeError('kalshi_ws: no open KX*15M markets found at startup')

    next_cmd_id = [1]
    await ws.send(json.dumps({
        'id': next_cmd_id[0],
        'cmd': 'subscribe',
        'params': {
            'channels': ['orderbook_delta', 'trade'],
            'market_tickers': tickers,
        },
    }))
    print(f'kalshi_ws: subscribed to {len(tickers)} tickers across {COINS}')

    active_sids: list[int] = []
    discovery_task = asyncio.create_task(
        _discovery_loop(session, ticker_to_coin, ws, next_cmd_id, active_sids)
    )
    flush_task = asyncio.create_task(_periodic_flush_loop(data_sink, buffers))

    try:
        async for raw in ws:
            try:
                msg_envelope = json.loads(raw)
            except Exception:
                continue

            msg_type = msg_envelope.get('type')
            msg = msg_envelope.get('msg') or {}
            curr_time = datetime.now(timezone.utc).isoformat()

            if msg_type == 'subscribed':
                sid = msg.get('sid')
                if isinstance(sid, int) and sid not in active_sids:
                    active_sids.append(sid)
                print(f'kalshi_ws: subscribed ack channel={msg.get("channel")} sid={sid}')
            elif msg_type == 'error':
                print(f'kalshi_ws: error msg: {msg}')
            elif msg_type == 'orderbook_snapshot':
                deltas.extend(_snapshot_rows(msg, ticker_to_coin, curr_time))
            elif msg_type == 'orderbook_delta':
                row = _delta_row(msg, ticker_to_coin, curr_time)
                if row is not None:
                    deltas.add(row)
            elif msg_type == 'trade':
                row = _trade_row(msg, ticker_to_coin, curr_time)
                if row is not None:
                    trades.add(row)

            for buf in buffers:
                if buf.should_flush():
                    await buf.flush(data_sink)
    finally:
        discovery_task.cancel()
        flush_task.cancel()
        await asyncio.gather(discovery_task, flush_task, return_exceptions=True)
        for buf in buffers:
            await buf.flush(data_sink)


async def stream_kalshi_events(data_sink=None) -> None:
    data_sink = data_sink or create_data_sink()

    key_id = os.getenv('KALSHI_KEY_ID', '')
    pem_path = os.getenv('KALSHI_PRIVATE_KEY_PATH', '')
    if not key_id or not pem_path:
        raise RuntimeError('kalshi_ws: KALSHI_KEY_ID and KALSHI_PRIVATE_KEY_PATH must be set')
    private_key = _load_private_key(pem_path)

    backoff = 1.0
    async with aiohttp.ClientSession() as session:
        while True:
            ticker_to_coin = await _discover_tickers(session)
            if not ticker_to_coin:
                print('kalshi_ws: no open markets yet, retrying in 30s')
                await asyncio.sleep(30)
                continue

            headers = _sign_headers(key_id, private_key, 'GET', WS_SIGN_PATH)
            try:
                print(f'kalshi_ws: connecting to {WS_URL}')
                async with websockets.connect(
                    WS_URL,
                    additional_headers=headers,
                    ping_interval=20,
                    ping_timeout=20,
                    max_size=None,
                ) as ws:
                    backoff = 1.0
                    await _consume(ws, session, ticker_to_coin, data_sink)
            except asyncio.CancelledError:
                raise
            except Exception as e:
                print(f'kalshi_ws: connection failed: {type(e).__name__}: {e}')
                jitter = random.uniform(0, 0.5 * backoff)
                await asyncio.sleep(backoff + jitter)
                backoff = min(RECONNECT_MAX_BACKOFF, backoff * 2)


if __name__ == '__main__':
    asyncio.run(stream_kalshi_events())

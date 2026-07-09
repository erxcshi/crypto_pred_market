import asyncio
import json
import random
import time
from datetime import datetime, timezone

import websockets

from config import create_data_sink

WS_URL = 'wss://ws-feed.exchange.coinbase.com'
PRODUCT_IDS = ['BTC-USD', 'ETH-USD', 'XRP-USD', 'SOL-USD']
CHANNELS = ['matches', 'ticker']
BATCH_SIZE = 100
FLUSH_INTERVAL_SECONDS = 5
RECONNECT_MAX_BACKOFF = 60

TRADES_TABLE = 'coinbase_trades'
QUOTES_TABLE = 'coinbase_quotes'

TRADES_FIELDS = ['curr_time', 'product_id', 'event_type', 'trade_time', 'trade_id',
                 'sequence', 'price', 'size', 'side', 'maker_order_id', 'taker_order_id']
QUOTES_FIELDS = ['curr_time', 'product_id', 'trade_time', 'sequence', 'price',
                 'best_bid', 'best_ask', 'best_bid_size', 'best_ask_size',
                 'last_size', 'side']


def build_subscribe_message():
    return {
        'type': 'subscribe',
        'product_ids': PRODUCT_IDS,
        'channels': CHANNELS,
    }


def _to_float(x, default=None):
    if x is None:
        return default
    try:
        return float(x)
    except (TypeError, ValueError):
        return default


def _to_int(x, default=None):
    if x is None:
        return default
    try:
        return int(x)
    except (TypeError, ValueError):
        return default


def build_trade_row(message):
    return {
        'curr_time': datetime.now(timezone.utc).isoformat(),
        'product_id': message.get('product_id'),
        'event_type': message.get('type'),
        'trade_time': message.get('time'),
        'trade_id': message.get('trade_id'),
        'sequence': message.get('sequence'),
        'price': _to_float(message.get('price')),
        'size': _to_float(message.get('size')),
        'side': message.get('side'),
        'maker_order_id': message.get('maker_order_id'),
        'taker_order_id': message.get('taker_order_id'),
    }


def build_quote_row(message):
    return {
        'curr_time': datetime.now(timezone.utc).isoformat(),
        'product_id': message.get('product_id'),
        'trade_time': message.get('time'),
        'sequence': message.get('sequence'),
        'price': _to_float(message.get('price')),
        'best_bid': _to_float(message.get('best_bid')),
        'best_ask': _to_float(message.get('best_ask')),
        'best_bid_size': _to_float(message.get('best_bid_size')),
        'best_ask_size': _to_float(message.get('best_ask_size')),
        'last_size': _to_float(message.get('last_size')),
        'side': message.get('side'),
    }


class _Buffer:
    def __init__(self, table, csv_filename, fields):
        self.table = table
        self.csv_filename = csv_filename
        self.fields = fields
        self.rows = []
        self.last_flush = time.monotonic()

    def add(self, row):
        self.rows.append(row)

    def should_flush(self):
        if not self.rows:
            return False
        return (len(self.rows) >= BATCH_SIZE
                or time.monotonic() - self.last_flush >= FLUSH_INTERVAL_SECONDS)

    async def flush(self, data_sink):
        if not self.rows:
            return
        if await data_sink.submit_rows(self.table, self.rows,
                                       csv_filename=self.csv_filename,
                                       fieldnames=self.fields):
            self.rows.clear()
            self.last_flush = time.monotonic()


async def _flush_loop(data_sink, buffers):
    while True:
        await asyncio.sleep(1)
        for buf in buffers:
            if buf.should_flush():
                await buf.flush(data_sink)


def _check_trade_id_gap(last_trade_id: dict, product_id, trade_id):
    """Returns gap size if trade_id skipped ahead, else 0.

    NOTE: 'sequence' is feed-wide (counts every order event), so gaps in it
    are EXPECTED when only matches/ticker are subscribed — do not use it for
    integrity checks. trade_id is dense per product on the matches channel,
    so a jump there means actually-missed trades (log it; reconnecting can't
    recover them anyway)."""
    tid = _to_int(trade_id)
    if tid is None or product_id is None:
        return 0
    prev = last_trade_id.get(product_id)
    last_trade_id[product_id] = tid
    if prev is None or tid <= prev:
        return 0
    return tid - prev - 1


async def stream_coinbase_trades(data_sink=None):
    data_sink = data_sink or create_data_sink()
    backoff = 1.0

    trades = _Buffer(TRADES_TABLE, 'coinbase_trades_test.csv', TRADES_FIELDS)
    quotes = _Buffer(QUOTES_TABLE, 'coinbase_quotes_test.csv', QUOTES_FIELDS)
    buffers = [trades, quotes]

    while True:
        last_trade_id: dict[str, int] = {}
        flush_task = None
        try:
            print(f'coinbase_ws: connecting to {WS_URL}')
            async with websockets.connect(WS_URL, ping_interval=20, ping_timeout=20) as ws:
                await ws.send(json.dumps(build_subscribe_message()))
                backoff = 1.0
                flush_task = asyncio.create_task(_flush_loop(data_sink, buffers))

                trade_count = 0
                quote_count = 0
                async for raw in ws:
                    try:
                        msg = json.loads(raw)
                    except Exception:
                        continue

                    msg_type = msg.get('type')

                    if msg_type == 'match':
                        gap = _check_trade_id_gap(last_trade_id, msg.get('product_id'),
                                                  msg.get('trade_id'))
                        if gap > 0:
                            # Log-only: the trades are already gone; a reconnect
                            # can't recover them and just loses more data.
                            print(f'coinbase_ws: missed {gap} trades on '
                                  f'{msg.get("product_id")} (trade_id gap)')

                    if msg_type == 'match':
                        trades.add(build_trade_row(msg))
                        trade_count += 1
                        if trade_count % 500 == 0:
                            print(f'coinbase_ws: {trade_count} trades, {quote_count} quotes')
                    elif msg_type == 'ticker':
                        quotes.add(build_quote_row(msg))
                        quote_count += 1
                    elif msg_type == 'subscriptions':
                        print(f'coinbase_ws: subscribed to {CHANNELS} for {PRODUCT_IDS}')
                    elif msg_type == 'error':
                        print(f'coinbase_ws: error: {msg}')

                    for buf in buffers:
                        if buf.should_flush():
                            await buf.flush(data_sink)
        except asyncio.CancelledError:
            for buf in buffers:
                await buf.flush(data_sink)
            raise
        except Exception as e:
            print(f'coinbase_ws: failed: {type(e).__name__}: {e}')
        finally:
            if flush_task is not None:
                flush_task.cancel()
                try:
                    await flush_task
                except (asyncio.CancelledError, Exception):
                    pass
            for buf in buffers:
                await buf.flush(data_sink)

        jitter = random.uniform(0, 0.5 * backoff)
        await asyncio.sleep(backoff + jitter)
        backoff = min(RECONNECT_MAX_BACKOFF, backoff * 2)


if __name__ == '__main__':
    asyncio.run(stream_coinbase_trades())

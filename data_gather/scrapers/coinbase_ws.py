import asyncio
import json
import time
from datetime import datetime, timezone

import websockets

from config import create_data_sink

WS_URL = 'wss://ws-feed.exchange.coinbase.com'
PRODUCT_IDS = ['BTC-USD', 'ETH-USD', 'XRP-USD', 'SOL-USD']
BATCH_SIZE = 100
FLUSH_INTERVAL_SECONDS = 5
INSERT_RETRIES = 3


def build_subscribe_message():
    return {
        'type': 'subscribe',
        'product_ids': PRODUCT_IDS,
        'channels': ['matches'],
    }


def build_trade_row(message):
    return {
        'curr_time': datetime.now(timezone.utc).isoformat(),
        'product_id': message.get('product_id'),
        'event_type': message.get('type'),
        'trade_time': message.get('time'),
        'trade_id': message.get('trade_id'),
        'sequence': message.get('sequence'),
        'price': float(message['price']) if message.get('price') is not None else None,
        'size': float(message['size']) if message.get('size') is not None else None,
        'side': message.get('side'),
        'maker_order_id': message.get('maker_order_id'),
        'taker_order_id': message.get('taker_order_id'),
    }


async def stream_coinbase_trades(data_sink=None):
    rows_data = []
    last_flush = time.monotonic()
    data_sink = data_sink or create_data_sink()

    while True:
        try:
            print(f'connecting to {WS_URL}')
            async with websockets.connect(WS_URL, ping_interval=20, ping_timeout=20) as websocket:
                await websocket.send(json.dumps(build_subscribe_message()))

                async for raw_message in websocket:
                    message = json.loads(raw_message)

                    if message.get('type') != 'match':
                        continue

                    row_data = build_trade_row(message)
                    rows_data.append(row_data)
                    print(row_data)

                    should_flush = len(rows_data) >= BATCH_SIZE or (
                        rows_data and time.monotonic() - last_flush >= FLUSH_INTERVAL_SECONDS
                    )
                    if should_flush:
                        if await data_sink.submit_rows('coinbase_trades', rows_data, csv_filename='coinbase_test.csv'):
                            rows_data.clear()
                            last_flush = time.monotonic()
        except asyncio.CancelledError:
            if rows_data:
                await data_sink.submit_rows('coinbase_trades', rows_data, csv_filename='coinbase_test.csv')
                rows_data.clear()
            raise
        except Exception as e:
            if rows_data:
                await data_sink.submit_rows('coinbase_trades', rows_data, csv_filename='coinbase_test.csv')
                rows_data.clear()
                last_flush = time.monotonic()
            print(f'coinbase websocket failed: {type(e).__name__}: {e}')
            await asyncio.sleep(3)


if __name__ == '__main__':
    asyncio.run(stream_coinbase_trades())

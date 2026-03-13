import asyncio
import json
import time
from datetime import datetime, timezone

import websockets

from crypto_predictions.config import create_supabase_client

WS_URL = 'wss://ws-feed.exchange.coinbase.com'
PRODUCT_IDS = ['BTC-USD', 'ETH-USD', 'XRP-USD', 'SOL-USD']
BATCH_SIZE = 100
INSERT_RETRIES = 3

supabase_client = create_supabase_client()


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


def supabase_submit(rows_data):
    global supabase_client

    for attempt in range(1, INSERT_RETRIES + 1):
        try:
            supabase_client.table('coinbase_trades').insert(rows_data).execute()
            return True
        except Exception as e:
            print(f'coinbase_trades insert failed (attempt {attempt}/{INSERT_RETRIES}): {e}')
            supabase_client = create_supabase_client()
            time.sleep(attempt)

    print(f'coinbase_trades insert gave up after {INSERT_RETRIES} attempts')
    return False


async def stream_coinbase_trades():
    rows_data = []

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

                    if len(rows_data) >= BATCH_SIZE:
                        if supabase_submit(rows_data):
                            rows_data.clear()
        except Exception as e:
            print(f'coinbase websocket failed: {type(e).__name__}: {e}')
            await asyncio.sleep(3)


if __name__ == '__main__':
    asyncio.run(stream_coinbase_trades())

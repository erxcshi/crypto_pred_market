import asyncio
import time
from datetime import datetime, timezone

import aiohttp

from crypto_pred_market.config import create_data_sink

# SUPABASE 
POLL_SECONDS = 1
INSERT_RETRIES = 3
BATCH_SIZE = 100
MARKET_FIELDS = [
    'open_time',
    'close_time',
    'last_price_dollars',
    'liquidity_dollars',
    'no_ask_dollars',
    'no_bid_dollars',
    'yes_ask_dollars',
    'yes_bid_dollars',
    'open_interest',
    'floor_strike',
    'cap_strike',
    'strike_type',
    'volume',
    'volume_24h_fp',
]


def build_market_row(coin, curr_time, market):
    row_data = {
        'curr_time': curr_time,
        'coin': coin,
    }

    for field in MARKET_FIELDS:
        row_data[field] = market.get(field)

    return row_data


async def collect_kalshi_rows(session, coin, curr_time):
    markets_url = f"https://api.elections.kalshi.com/trade-api/v2/markets?series_ticker=KX{coin}15M&status=open"
    try:
        async with session.get(markets_url, timeout=aiohttp.ClientTimeout(total=15)) as markets_response:
            markets_response.raise_for_status()
            markets_data = await markets_response.json()

            return [
                build_market_row(coin, curr_time, market)
                for market in markets_data.get('markets', [])
            ]
    except Exception as e:
        print(f'failed to collect kalshi rows for {coin}: {type(e).__name__}: {e}')
        return []


async def scrape_kalshi(coins, data_sink=None):
    rows_data = []
    data_sink = data_sink or create_data_sink()

    print(f'starting scrape for {", ".join(coins)}')

    async with aiohttp.ClientSession() as session:
        while True:
            start = time.time()
            curr_time = datetime.now(timezone.utc).isoformat()

            try:
                tasks = [collect_kalshi_rows(session, coin, curr_time) for coin in coins]
                row_batches = await asyncio.gather(*tasks)

                for coin, row_batch in zip(coins, row_batches):
                    rows_data.extend(row_batch)
                    if row_batch:
                        print(f'collected {len(row_batch)} kalshi rows for {coin}')

                while len(rows_data) >= BATCH_SIZE:
                    row_chunk = rows_data[:BATCH_SIZE]
                    if not await data_sink.submit_rows(
                        'kalshi_markets',
                        row_chunk,
                        csv_filename='kalshi_test.csv',
                        fieldnames=MARKET_FIELDS + ['curr_time', 'coin'],
                    ):
                        break

                    print(f'inserted {len(row_chunk)} kalshi rows')
                    del rows_data[:BATCH_SIZE]

                # Submit remaining rows at end of cycle
                if rows_data:
                    if await data_sink.submit_rows(
                        'kalshi_markets',
                        rows_data,
                        csv_filename='kalshi_test.csv',
                        fieldnames=MARKET_FIELDS + ['curr_time', 'coin'],
                    ):
                        print(f'inserted {len(rows_data)} remaining kalshi rows')
                    rows_data.clear()
            except Exception as e:
                print(f'kalshi cycle failed: {type(e).__name__}: {e}')

            elapsed = time.time() - start
            await asyncio.sleep(max(0, POLL_SECONDS - elapsed))

if __name__ == '__main__': 

    coins = ['ETH', 'BTC', 'XRP', 'SOL']
    asyncio.run(scrape_kalshi(coins))

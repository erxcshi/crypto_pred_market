import requests
import time
from datetime import datetime, timezone
from concurrent.futures import ThreadPoolExecutor

from crypto_predictions.config import create_supabase_client

# SUPABASE 
POLL_SECONDS = 2
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


def collect_kalshi_rows(coin, curr_time):
    session = requests.Session()
    markets_url = f"https://api.elections.kalshi.com/trade-api/v2/markets?series_ticker=KX{coin}15M&status=open"
    try:
        markets_response = session.get(markets_url, timeout=15)
        markets_response.raise_for_status()
        markets_data = markets_response.json()

        return [
            build_market_row(coin, curr_time, market)
            for market in markets_data.get('markets', [])
        ]
    except Exception as e:
        print(f'failed to collect kalshi rows for {coin}: {type(e).__name__}: {e}')
        return []


def scrape_kalshi(coins):
    rows_data = []
    supabase_client = create_supabase_client()

    print(f'starting scrape for {", ".join(coins)}')

    with ThreadPoolExecutor(max_workers=len(coins)) as exec:
        while True:
            start = time.time()
            curr_time = datetime.now(timezone.utc).isoformat()

            try:
                row_batches = list(
                    exec.map(
                        lambda coin: collect_kalshi_rows(coin, curr_time),
                        coins,
                    )
                )

                for coin, row_batch in zip(coins, row_batches):
                    rows_data.extend(row_batch)
                    if row_batch:
                        print(f'collected {len(row_batch)} kalshi rows for {coin}')

                while len(rows_data) >= BATCH_SIZE:
                    row_chunk = rows_data[:BATCH_SIZE]
                    if not supabase_submit(supabase_client, row_chunk):
                        break

                    print(f'inserted {len(row_chunk)} kalshi rows')
                    del rows_data[:BATCH_SIZE]
            except Exception as e:
                print(f'kalshi cycle failed: {type(e).__name__}: {e}')

            elapsed = time.time() - start
            time.sleep(max(0, POLL_SECONDS - elapsed))


def supabase_submit(supabase_client, rows_data):
    for attempt in range(1, INSERT_RETRIES + 1):
        try:
            supabase_client.table('kalshi_markets').insert(rows_data).execute()
            return True
        except Exception as e:
            print(f'kalshi_markets insert failed (attempt {attempt}/{INSERT_RETRIES}): {e}')
            supabase_client = create_supabase_client()
            time.sleep(attempt)

    print(f'kalshi_markets insert gave up after {INSERT_RETRIES} attempts')
    return False


if __name__ == '__main__': 

    coins = ['ETH', 'BTC', 'XRP', 'SOL']
    scrape_kalshi(coins)

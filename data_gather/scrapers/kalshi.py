import asyncio
import time
from datetime import datetime, timezone

import aiohttp

from config import create_data_sink

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


async def fetch_orderbook_quotes(session, ticker):
    """
    Returns (yes_bid, yes_ask, no_bid, no_ask) in dollars from the live orderbook_fp
    snapshot, or None if the book is empty on either side or the request fails.

    Kalshi's /markets endpoint caches yes_bid/yes_ask for several seconds; the
    /orderbook endpoint reflects the live book. Derived asks use the complement
    rule: YES ask = 1 - best NO bid.
    """
    ob_url = f"https://api.elections.kalshi.com/trade-api/v2/markets/{ticker}/orderbook"
    try:
        async with session.get(ob_url, timeout=aiohttp.ClientTimeout(total=5)) as resp:
            resp.raise_for_status()
            data = await resp.json()
            ob = data.get('orderbook_fp') or {}
            yes_bids = sorted(
                ([float(p), float(s)] for p, s in (ob.get('yes_dollars') or [])),
                reverse=True,
            )
            no_bids = sorted(
                ([float(p), float(s)] for p, s in (ob.get('no_dollars') or [])),
                reverse=True,
            )
            if not yes_bids or not no_bids:
                return None
            best_yes_bid = yes_bids[0][0]
            best_no_bid = no_bids[0][0]
            return (
                best_yes_bid,
                1.0 - best_no_bid,
                best_no_bid,
                1.0 - best_yes_bid,
            )
    except Exception:
        return None


async def collect_kalshi_rows(session, coin, curr_time):
    markets_url = f"https://api.elections.kalshi.com/trade-api/v2/markets?series_ticker=KX{coin}15M&status=open"
    try:
        async with session.get(markets_url, timeout=aiohttp.ClientTimeout(total=15)) as markets_response:
            markets_response.raise_for_status()
            markets_data = await markets_response.json()
            markets = markets_data.get('markets', [])

            # /markets yes_bid/yes_ask lag the live book by seconds in thin 15M
            # crypto markets, baking staleness into any rolling features trained
            # on this data. Pull each open market's orderbook in parallel and
            # override the four bid/ask fields with live top-of-book.
            async def _ob_or_none(t):
                return await fetch_orderbook_quotes(session, t) if t else None

            tickers = [m.get('ticker') for m in markets]
            ob_quotes = await asyncio.gather(*[_ob_or_none(t) for t in tickers])

            rows = []
            for market, quotes in zip(markets, ob_quotes):
                row = build_market_row(coin, curr_time, market)
                if quotes is not None:
                    yes_bid, yes_ask, no_bid, no_ask = quotes
                    row['yes_bid_dollars'] = yes_bid
                    row['yes_ask_dollars'] = yes_ask
                    row['no_bid_dollars']  = no_bid
                    row['no_ask_dollars']  = no_ask
                rows.append(row)
            return rows
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

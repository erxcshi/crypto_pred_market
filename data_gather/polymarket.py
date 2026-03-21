import json
import time
import asyncio
from datetime import datetime, timezone

import aiohttp

from crypto_pred_market.config import create_data_sink


GAMMA_BASE = 'https://gamma-api.polymarket.com'
CLOB_BASE = 'https://clob.polymarket.com'


# SUPABASE 
TIME_HORIZON = 15                               #5 or 15
time_horizon_seconds = TIME_HORIZON * 60
POLL_SECONDS = 1
INSERT_RETRIES = 3
BATCH_SIZE = 100


async def get_window_starting_price(session, coin, interval_start_unix):

    # convert to ms because binance works with ms data 
    start_ms = interval_start_unix * 1000                       
    end_ms = start_ms + (time_horizon_seconds * 1000) - 1

    async with session.get(
        'https://api.binance.us/api/v3/klines',
        params={
            'symbol': f'{coin}USDT',
            'interval': f'{TIME_HORIZON}m',
            'startTime': start_ms,
            'endTime': end_ms,
            'limit': 1,
        },
        timeout=aiohttp.ClientTimeout(total=5),
    ) as resp:
        resp.raise_for_status()

        klines = await resp.json()
        if not klines:
            raise ValueError(f'no Binance kline found for {coin} at {interval_start_unix}')

        kline = klines[0]
        return float(kline[1])


async def get_gamma_event(session, slug):
    async with session.get(f'{GAMMA_BASE}/events/slug/{slug}', timeout=aiohttp.ClientTimeout(total=5)) as resp:
        resp.raise_for_status()
        return await resp.json()


def extract_gamma_market(event):
    markets = event.get('markets', [])
    if not markets:
        raise ValueError('no Polymarket markets found in Gamma event payload')

    market = markets[0]

    outcome_prices = json.loads(market.get('outcomePrices', '[]'))
    clob_token_ids = json.loads(market.get('clobTokenIds', '[]'))

    yes_token_id = clob_token_ids[0] if len(clob_token_ids) > 0 else None
    no_token_id = clob_token_ids[1] if len(clob_token_ids) > 1 else None

    return {
        'end_date': event.get('endDate'),
        'market_slug': market.get('slug'),
        'condition_id': market.get('conditionId'),
        'liquidity': market.get('liquidity'),
        'volume': market.get('volume'),
        'open_interest': market.get('openInterest'),
        'outcome_prices': outcome_prices,
        'yes_token_id': yes_token_id,
        'no_token_id': no_token_id,
    }


async def get_clob_book(session, token_id):
    async with session.get(
        f'{CLOB_BASE}/book',
        params={'token_id': token_id},
        timeout=aiohttp.ClientTimeout(total=5),
    ) as resp:
        resp.raise_for_status()
        return await resp.json()


async def get_clob_price(session, token_id, side):
    async with session.get(
        f'{CLOB_BASE}/price',
        params={'token_id': token_id, 'side': side},
        timeout=aiohttp.ClientTimeout(total=5),
    ) as resp:
        resp.raise_for_status()
        return await resp.json()


async def get_clob_midpoint(session, token_id):
    async with session.get(
        f'{CLOB_BASE}/midpoint',
        params={'token_id': token_id},
        timeout=aiohttp.ClientTimeout(total=5),
    ) as resp:
        resp.raise_for_status()
        return await resp.json()


async def extract_clob_snapshot(session, token_id):
    # Make CLOB API calls concurrently
    book_task = get_clob_book(session, token_id)
    buy_price_task = get_clob_price(session, token_id, 'BUY')
    sell_price_task = get_clob_price(session, token_id, 'SELL')
    midpoint_task = get_clob_midpoint(session, token_id)
    
    book, buy_price, sell_price, midpoint = await asyncio.gather(
        book_task, buy_price_task, sell_price_task, midpoint_task
    )

    return {
        'timestamp': book.get('timestamp'),
        'best_bid': book['bids'][0] if book.get('bids') else None,
        'best_ask': book['asks'][0] if book.get('asks') else None,
        'buy_price': buy_price.get('price'),
        'sell_price': sell_price.get('price'),
        'midpoint': midpoint.get('mid') or midpoint.get('mid_price'),
    }


def build_polymarket_row(curr_time, coin, interval_start_unix, strike_price, gamma_market, up_clob, down_clob):
    return {
        'curr_time': curr_time,
        'coin': coin,
        'time_horizon_minutes': TIME_HORIZON,
        'interval_start_unix': interval_start_unix,
        'market_slug': gamma_market.get('market_slug'),
        'condition_id': gamma_market.get('condition_id'),
        'strike_price': strike_price,
        'end_date': gamma_market.get('end_date'),
        'liquidity': gamma_market.get('liquidity'),
        'volume': gamma_market.get('volume'),
        'open_interest': gamma_market.get('open_interest'),
        'yes_implied_price': gamma_market.get('outcome_prices', [None])[0] if len(gamma_market.get('outcome_prices', [])) > 0 else None,
        'no_implied_price': gamma_market.get('outcome_prices', [None, None])[1] if len(gamma_market.get('outcome_prices', [])) > 1 else None,
        'yes_buy_price': up_clob.get('buy_price'),
        'yes_sell_price': up_clob.get('sell_price'),
        'no_buy_price': down_clob.get('buy_price'),
        'no_sell_price': down_clob.get('sell_price'),
    }


async def collect_polymarket_row(session, coin, curr_time, interval_start_unix):
    try:
        # Get Binance price and Gamma event concurrently
        binance_task = get_window_starting_price(session, coin, interval_start_unix)
        slug = f'{coin.lower()}-updown-{TIME_HORIZON}m-{interval_start_unix}'
        gamma_task = get_gamma_event(session, slug)
        
        strike_price, event = await asyncio.gather(binance_task, gamma_task)
        gamma_market = extract_gamma_market(event)
        
        # Get CLOB data for YES and NO tokens concurrently
        up_task = extract_clob_snapshot(session, gamma_market['yes_token_id']) if gamma_market['yes_token_id'] else asyncio.create_task(asyncio.sleep(0, result={}))
        down_task = extract_clob_snapshot(session, gamma_market['no_token_id']) if gamma_market['no_token_id'] else asyncio.create_task(asyncio.sleep(0, result={}))
        
        up_clob, down_clob = await asyncio.gather(up_task, down_task)

        return build_polymarket_row(
            curr_time=curr_time,
            coin=coin,
            interval_start_unix=interval_start_unix,
            strike_price=strike_price,
            gamma_market=gamma_market,
            up_clob=up_clob,
            down_clob=down_clob,
        )
    except Exception as e:
        print(f'failed to collect polymarket row for {coin}: {type(e).__name__}: {e}')
        return None


async def scrape_polymarket(coins, data_sink=None):
    rows_data = []
    data_sink = data_sink or create_data_sink()

    print(f'starting scrape for {", ".join(coins)}')

    async with aiohttp.ClientSession() as session:
        while True:
            start = time.time()
            curr_time = datetime.now(timezone.utc).isoformat()
            interval_start_unix = int(time.time() // time_horizon_seconds) * time_horizon_seconds

            try:
                print("USING NEW POLYMARKET CODE")

                tasks = [collect_polymarket_row(session, coin, curr_time, interval_start_unix) for coin in coins]
                row_batch = await asyncio.gather(*tasks)
                row_batch = [row for row in row_batch if row is not None]
                rows_data.extend(row_batch)

                for row_data in row_batch:
                    print(row_data)

                while len(rows_data) >= BATCH_SIZE:
                    row_chunk = rows_data[:BATCH_SIZE]
                    if not await data_sink.submit_rows(
                        'polymarket_markets',
                        row_chunk,
                        csv_filename='polymarket_test.csv',
                    ):
                        break

                    del rows_data[:BATCH_SIZE]

                # Submit remaining rows at end of cycle
                if rows_data:
                    await data_sink.submit_rows(
                        'polymarket_markets',
                        rows_data,
                        csv_filename='polymarket_test.csv',
                    )
                    rows_data.clear()
            except Exception as e:
                print(f'polymarket cycle failed: {type(e).__name__}: {e}')

            elapsed = time.time() - start
            await asyncio.sleep(max(0, POLL_SECONDS - elapsed))



if __name__ == '__main__':
    coins = ['BTC', 'ETH', 'XRP', 'SOL']
    asyncio.run(scrape_polymarket(coins))

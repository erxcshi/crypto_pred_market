import json
import time
from datetime import datetime, timezone
from concurrent.futures import ThreadPoolExecutor

import requests

from crypto_predictions.config import create_supabase_client


GAMMA_BASE = 'https://gamma-api.polymarket.com'
CLOB_BASE = 'https://clob.polymarket.com'


# SUPABASE 
TIME_HORIZON = 15                               #5 or 15
time_horizon_seconds = TIME_HORIZON * 60
POLL_SECONDS = 2
INSERT_RETRIES = 3
BATCH_SIZE = 100


def get_window_starting_price(session, coin, interval_start_unix):

    # convert to ms because binance works with ms data 
    start_ms = interval_start_unix * 1000                       
    end_ms = start_ms + (time_horizon_seconds * 1000) - 1

    resp = session.get(
        'https://api.binance.us/api/v3/klines',
        params={
            'symbol': f'{coin}USDT',
            'interval': f'{TIME_HORIZON}m',
            'startTime': start_ms,
            'endTime': end_ms,
            'limit': 1,
        },
        timeout=10,
    )
    resp.raise_for_status()

    klines = resp.json()
    if not klines:
        raise ValueError(f'no Binance kline found for {coin} at {interval_start_unix}')

    kline = klines[0]
    return float(kline[1])


def get_gamma_event(session, slug):
    resp = session.get(f'{GAMMA_BASE}/events/slug/{slug}', timeout=10)
    resp.raise_for_status()
    return resp.json()


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


def get_clob_book(session, token_id):
    resp = session.get(
        f'{CLOB_BASE}/book',
        params={'token_id': token_id},
        timeout=10,
    )
    resp.raise_for_status()
    return resp.json()


def get_clob_price(session, token_id, side):
    resp = session.get(
        f'{CLOB_BASE}/price',
        params={'token_id': token_id, 'side': side},
        timeout=10,
    )
    resp.raise_for_status()
    return resp.json()


def get_clob_midpoint(session, token_id):
    resp = session.get(
        f'{CLOB_BASE}/midpoint',
        params={'token_id': token_id},
        timeout=10,
    )
    resp.raise_for_status()
    return resp.json()


def extract_clob_snapshot(session, token_id):
    book = get_clob_book(session, token_id)
    buy_price = get_clob_price(session, token_id, 'BUY')
    sell_price = get_clob_price(session, token_id, 'SELL')
    midpoint = get_clob_midpoint(session, token_id)

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
        'yes_token_id': gamma_market.get('yes_token_id'),
        'no_token_id': gamma_market.get('no_token_id'),
        'yes_implied_price': gamma_market.get('outcome_prices', [None])[0] if len(gamma_market.get('outcome_prices', [])) > 0 else None,
        'no_implied_price': gamma_market.get('outcome_prices', [None, None])[1] if len(gamma_market.get('outcome_prices', [])) > 1 else None,
        'yes_book_timestamp': up_clob.get('timestamp'),
        'yes_best_bid': up_clob.get('best_bid', {}).get('price') if up_clob.get('best_bid') else None,
        'yes_best_bid_size': up_clob.get('best_bid', {}).get('size') if up_clob.get('best_bid') else None,
        'yes_best_ask': up_clob.get('best_ask', {}).get('price') if up_clob.get('best_ask') else None,
        'yes_best_ask_size': up_clob.get('best_ask', {}).get('size') if up_clob.get('best_ask') else None,
        'yes_buy_price': up_clob.get('buy_price'),
        'yes_sell_price': up_clob.get('sell_price'),
        'yes_midpoint': up_clob.get('midpoint'),
        'no_book_timestamp': down_clob.get('timestamp'),
        'no_best_bid': down_clob.get('best_bid', {}).get('price') if down_clob.get('best_bid') else None,
        'no_best_bid_size': down_clob.get('best_bid', {}).get('size') if down_clob.get('best_bid') else None,
        'no_best_ask': down_clob.get('best_ask', {}).get('price') if down_clob.get('best_ask') else None,
        'no_best_ask_size': down_clob.get('best_ask', {}).get('size') if down_clob.get('best_ask') else None,
        'no_buy_price': down_clob.get('buy_price'),
        'no_sell_price': down_clob.get('sell_price'),
        'no_midpoint': down_clob.get('midpoint'),
    }


def supabase_submit(supabase_client, rows_data):
    for attempt in range(1, INSERT_RETRIES + 1):
        try:
            supabase_client.table('polymarket_markets').insert(rows_data).execute()
            return True
        except Exception as e:
            print(f'polymarket_markets insert failed (attempt {attempt}/{INSERT_RETRIES}): {e}')
            supabase_client = create_supabase_client()
            time.sleep(attempt)

    print(f'polymarket_markets insert gave up after {INSERT_RETRIES} attempts')
    return False


def collect_polymarket_row(coin, curr_time, interval_start_unix):
    session = requests.Session()

    try:
        strike_price = get_window_starting_price(session, coin, interval_start_unix)
        slug = f'{coin.lower()}-updown-{TIME_HORIZON}m-{interval_start_unix}'
        event = get_gamma_event(session, slug)
        gamma_market = extract_gamma_market(event)
        up_clob = extract_clob_snapshot(session, gamma_market['yes_token_id']) if gamma_market['yes_token_id'] else {}
        down_clob = extract_clob_snapshot(session, gamma_market['no_token_id']) if gamma_market['no_token_id'] else {}

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


def scrape_polymarket(coins):
    rows_data = []
    supabase_client = create_supabase_client()

    print(f'starting scrape for {", ".join(coins)}')

    with ThreadPoolExecutor(max_workers=len(coins)) as exec:
        while True:
            start = time.time()
            curr_time = datetime.now(timezone.utc).isoformat()
            interval_start_unix = int(time.time() // time_horizon_seconds) * time_horizon_seconds

            try:
                row_batch = list(
                    exec.map(
                        lambda coin: collect_polymarket_row(coin, curr_time, interval_start_unix),
                        coins,
                    )
                )
                row_batch = [row for row in row_batch if row is not None]
                rows_data.extend(row_batch)

                for row_data in row_batch:
                    print(row_data)

                while len(rows_data) >= BATCH_SIZE:
                    row_chunk = rows_data[:BATCH_SIZE]
                    if not supabase_submit(supabase_client, row_chunk):
                        break

                    del rows_data[:BATCH_SIZE]
            except Exception as e:
                print(f'polymarket cycle failed: {type(e).__name__}: {e}')

            elapsed = time.time() - start
            time.sleep(max(0, POLL_SECONDS - elapsed))



if __name__ == '__main__':
    coins = ['BTC', 'ETH', 'XRP', 'SOL']
    scrape_polymarket(coins)

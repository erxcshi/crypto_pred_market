import time
from datetime import datetime, timezone

import requests
import supabase


SUPABASE_URL = 'https://qiykynyfqpgxasoidcpz.supabase.co'
SUPABASE_KEY = 'sb_publishable_4UdbjMZgIQ7XEqJ4I8kIMA_z1BJ4ApR'

DERIBIT_BASE = 'https://www.deribit.com/api/v2'
CURRENCIES = ['BTC', 'ETH']
OPTION_BATCH_SIZE = 200
HV_BATCH_SIZE = 50
POLL_SECONDS = 5
INSERT_RETRIES = 3

session = requests.Session()


def make_supabase_client():
    return supabase.create_client(supabase_url=SUPABASE_URL, supabase_key=SUPABASE_KEY)


supabase_client = make_supabase_client()


def deribit_get(method, params):
    resp = session.get(
        f'{DERIBIT_BASE}/{method}',
        params=params,
        timeout=15,
    )
    resp.raise_for_status()
    payload = resp.json()

    if payload.get('error'):
        raise ValueError(payload['error'])

    return payload.get('result')


def build_option_row(currency, option_summary):
    creation_timestamp = option_summary.get('creation_timestamp')

    return {
        'curr_time': datetime.now(timezone.utc).isoformat(),
        'currency': currency,
        'instrument_name': option_summary.get('instrument_name'),
        'creation_time': datetime.fromtimestamp(creation_timestamp / 1000, tz=timezone.utc).isoformat() if creation_timestamp else None,
        'underlying_price': option_summary.get('underlying_price'),
        'mark_iv': option_summary.get('mark_iv'),
        'bid_price': option_summary.get('bid_price'),
        'ask_price': option_summary.get('ask_price'),
        'mid_price': option_summary.get('mid_price'),
        'mark_price': option_summary.get('mark_price'),
        'last_price': option_summary.get('last'),
        'open_interest': option_summary.get('open_interest'),
        'volume': option_summary.get('volume'),
        'volume_usd': option_summary.get('volume_usd'),
        'price_change': option_summary.get('price_change'),
        'underlying_index': option_summary.get('underlying_index'),
        'interest_rate': option_summary.get('interest_rate'),
    }


def build_hv_rows(currency, hist_vol_result):
    rows_data = []

    for timestamp_ms, value in hist_vol_result:
        rows_data.append({
            'curr_time': datetime.now(timezone.utc).isoformat(),
            'currency': currency,
            'vol_time': datetime.fromtimestamp(timestamp_ms / 1000, tz=timezone.utc).isoformat(),
            'vol_value': value,
        })

    return rows_data


def supabase_insert_with_retry(table_name, rows_data):
    global supabase_client

    for attempt in range(1, INSERT_RETRIES + 1):
        try:
            supabase_client.table(table_name).insert(rows_data).execute()
            return True
        except Exception as e:
            print(f'{table_name} insert failed (attempt {attempt}/{INSERT_RETRIES}): {e}')
            supabase_client = make_supabase_client()
            time.sleep(attempt)

    print(f'{table_name} insert gave up after {INSERT_RETRIES} attempts')
    return False


def supabase_submit_options(rows_data):
    return supabase_insert_with_retry('deribit_option_vols', rows_data)


def supabase_submit_hv(rows_data):
    return supabase_insert_with_retry('deribit_historical_vol', rows_data)


def scrape_deribit_vol():
    option_rows = []
    seen_hv = {currency: None for currency in CURRENCIES}

    while True:
        start = time.time()

        try:
            for currency in CURRENCIES:
                option_summaries = deribit_get(
                    'public/get_book_summary_by_currency',
                    {'currency': currency, 'kind': 'option'},
                )

                for option_summary in option_summaries:
                    row_data = build_option_row(currency, option_summary)
                    option_rows.append(row_data)

                print(f'fetched {len(option_summaries)} option summaries for {currency}')

                hv_result = deribit_get(
                    'public/get_historical_volatility',
                    {'currency': currency},
                )

                recent_hv_points = hv_result[-HV_BATCH_SIZE:]
                fresh_hv_points = [
                    point for point in recent_hv_points
                    if seen_hv[currency] is None or point[0] > seen_hv[currency]
                ]

                if fresh_hv_points:
                    fresh_hv_rows = build_hv_rows(currency, fresh_hv_points)
                    if supabase_submit_hv(fresh_hv_rows):
                        seen_hv[currency] = fresh_hv_points[-1][0]
                        print(f'inserted {len(fresh_hv_rows)} historical vol rows for {currency}')

            while len(option_rows) >= OPTION_BATCH_SIZE:
                option_chunk = option_rows[:OPTION_BATCH_SIZE]
                if not supabase_submit_options(option_chunk):
                    break

                print(f'inserted {len(option_chunk)} option rows')
                del option_rows[:OPTION_BATCH_SIZE]
        except Exception as e:
            print(f'deribit scrape failed: {type(e).__name__}: {e}')

        elapsed = time.time() - start
        time.sleep(max(0, POLL_SECONDS - elapsed))


if __name__ == '__main__':
    scrape_deribit_vol()

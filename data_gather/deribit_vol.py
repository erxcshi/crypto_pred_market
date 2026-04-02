import asyncio
import time
from datetime import datetime, timezone

import aiohttp

from crypto_pred_market.config import create_data_sink


DERIBIT_BASE = 'https://www.deribit.com/api/v2'
CURRENCIES = ['BTC', 'ETH']
OPTION_BATCH_SIZE = 200
POLL_SECONDS = 5
INSERT_RETRIES = 3
TICKER_BATCH_SIZE = 5
TICKER_RETRY_ATTEMPTS = 3
TICKER_RETRY_DELAY_SECONDS = 1
NEAR_SPOT_CANDIDATES_PER_SIDE = 12


async def deribit_get(session, method, params):
    async with session.get(
        f'{DERIBIT_BASE}/{method}',
        params=params,
        timeout=aiohttp.ClientTimeout(total=15),
    ) as resp:
        resp.raise_for_status()
        payload = await resp.json()

        if payload.get('error'):
            raise ValueError(payload['error'])

        return payload.get('result')


async def deribit_get_with_retry(session, method, params, attempts=1, retry_delay=0):
    for attempt in range(1, attempts + 1):
        try:
            return await deribit_get(session, method, params)
        except aiohttp.ClientResponseError as exc:
            is_last_attempt = attempt == attempts
            if exc.status != 429 or is_last_attempt:
                raise
            await asyncio.sleep(retry_delay)


def parse_instrument_name(instrument_name):
    """Parse Deribit instrument name: BTC-25DEC26-100000-C"""
    try:
        parts = instrument_name.split('-')
        if len(parts) >= 4:
            currency = parts[0]
            expiry = parts[1]
            strike = float(parts[2])
            option_type = parts[3]  # C or P
            return {'currency': currency, 'expiry': expiry, 'strike': strike, 'type': option_type}
    except (ValueError, IndexError):
        pass
    return None


def parse_expiry_date(expiry_str):
    """Parse Deribit expiry format (e.g., '27MAR26') into a datetime for proper sorting."""
    try:
        # Format: DDMMMYY (e.g., 27MAR26)
        from datetime import datetime
        return datetime.strptime(expiry_str, '%d%b%y')
    except ValueError:
        return None


def group_options_by_expiry(option_summaries):
    by_expiry = {}
    for opt in option_summaries:
        parsed = parse_instrument_name(opt.get('instrument_name', ''))
        if not parsed:
            continue

        expiry = parsed['expiry']
        if expiry not in by_expiry:
            by_expiry[expiry] = []
        by_expiry[expiry].append(opt)

    return by_expiry


def filter_delta_options(option_summaries):
    """Filter to keep 25-delta call, 25-delta put, and ATM for nearest expiration using actual delta values."""
    if not option_summaries:
        return []

    by_expiry = group_options_by_expiry(option_summaries)

    if not by_expiry:
        return []

    # Sort expirations chronologically and use the nearest (soonest)
    sorted_expiries = sorted(by_expiry.keys(), key=lambda x: parse_expiry_date(x) or datetime.max)
    nearest_expiry = sorted_expiries[0]
    options = by_expiry[nearest_expiry]

    selected = []
    calls = []
    puts = []

    # Separate calls and puts with their delta values from Deribit JSON
    for opt in options:
        parsed = parse_instrument_name(opt.get('instrument_name', ''))
        if not parsed:
            continue

        delta = opt.get('greeks', {}).get('delta')
        if delta is None:
            continue

        opt_type = parsed['type']

        if opt_type == 'C':
            calls.append((opt, delta))
        elif opt_type == 'P':
            puts.append((opt, delta))

    # Find 25-delta call (delta ~0.25)
    valid_calls = [(opt, delta) for opt, delta in calls if 0.10 <= delta <= 0.40]
    if valid_calls:
        best_call = min(valid_calls, key=lambda x: abs(x[1] - 0.25))
        selected.append(best_call[0])

    # Find ATM call (delta ~0.5)
    valid_atm_calls = [(opt, delta) for opt, delta in calls if 0.40 <= delta <= 0.60]
    if valid_atm_calls:
        best_atm = min(valid_atm_calls, key=lambda x: abs(x[1] - 0.5))
        if best_atm[0] not in selected:
            selected.append(best_atm[0])

    # Find 25-delta put (delta ~-0.25)
    valid_puts = [(opt, delta) for opt, delta in puts if -0.40 <= delta <= -0.10]
    if valid_puts:
        best_put = min(valid_puts, key=lambda x: abs(x[1] - (-0.25)))
        if best_put[0] not in selected:
            selected.append(best_put[0])

    return selected


def build_option_row(currency, option_summary, curr_time):
    greeks = option_summary.get('greeks', {})
    stats = option_summary.get('stats', {})
    parsed = parse_instrument_name(option_summary.get('instrument_name', '')) or {}
    expiry_datetime = parse_expiry_date(parsed.get('expiry', ''))
    return {
        'curr_time': curr_time,
        'currency': currency,
        'instrument_name': option_summary.get('instrument_name'),
        'expiry_datetime': expiry_datetime.isoformat() if expiry_datetime else None,
        'strike': parsed.get('strike'),
        'option_type': parsed.get('type'),
        'underlying_price': option_summary.get('underlying_price'),
        'delta': greeks.get('delta'),
        'mark_iv': option_summary.get('mark_iv'),
        'bid_price': option_summary.get('bid_price'),
        'ask_price': option_summary.get('ask_price'),
        'mark_price': option_summary.get('mark_price'),
        'open_interest': option_summary.get('open_interest'),
        'volume': stats.get('volume'),
    }


async def get_option_tickers_for_currency(session, currency):
    option_summaries = await deribit_get(
        session,
        'public/get_book_summary_by_currency',
        {'currency': currency, 'kind': 'option'},
    )

    if not option_summaries:
        return []

    by_expiry = group_options_by_expiry(option_summaries)
    if not by_expiry:
        return []

    sorted_expiries = sorted(by_expiry.keys(), key=lambda x: parse_expiry_date(x) or datetime.max)
    nearest_expiry = sorted_expiries[0]
    nearest_expiry_options = by_expiry[nearest_expiry]
    underlying_price = next(
        (opt.get('underlying_price') for opt in nearest_expiry_options if opt.get('underlying_price') is not None),
        None,
    )

    if underlying_price is None:
        return []

    calls = []
    puts = []
    for option_summary in nearest_expiry_options:
        parsed = parse_instrument_name(option_summary.get('instrument_name', ''))
        if not parsed:
            continue

        option_with_distance = {
            'instrument_name': option_summary.get('instrument_name'),
            'distance_to_spot': abs(parsed['strike'] - underlying_price),
        }
        if parsed['type'] == 'C':
            calls.append(option_with_distance)
        elif parsed['type'] == 'P':
            puts.append(option_with_distance)

    candidate_instruments = []
    candidate_instruments.extend(
        item['instrument_name']
        for item in sorted(calls, key=lambda item: item['distance_to_spot'])[:NEAR_SPOT_CANDIDATES_PER_SIDE]
    )
    candidate_instruments.extend(
        item['instrument_name']
        for item in sorted(puts, key=lambda item: item['distance_to_spot'])[:NEAR_SPOT_CANDIDATES_PER_SIDE]
    )

    instrument_names = list(dict.fromkeys(candidate_instruments))
    print(
        f'{currency}: nearest expiry {nearest_expiry} has {len(nearest_expiry_options)} instruments; '
        f'fetching {len(instrument_names)} near-spot tickers'
    )
    ticker_rows = []

    for idx in range(0, len(instrument_names), TICKER_BATCH_SIZE):
        batch_names = instrument_names[idx:idx + TICKER_BATCH_SIZE]
        tasks = [
            deribit_get_with_retry(
                session,
                'public/ticker',
                {'instrument_name': instrument_name},
                attempts=TICKER_RETRY_ATTEMPTS,
                retry_delay=TICKER_RETRY_DELAY_SECONDS,
            )
            for instrument_name in batch_names
        ]
        batch_results = await asyncio.gather(*tasks, return_exceptions=True)

        for instrument_name, result in zip(batch_names, batch_results):
            if isinstance(result, Exception):
                print(f'failed to fetch ticker for {instrument_name}: {type(result).__name__}: {result}')
                continue

            result['instrument_name'] = instrument_name
            ticker_rows.append(result)

        if idx + TICKER_BATCH_SIZE < len(instrument_names):
            await asyncio.sleep(TICKER_RETRY_DELAY_SECONDS)

    return ticker_rows


async def collect_currency_option_rows(session, currency, curr_time):
    option_summaries = await get_option_tickers_for_currency(session, currency)

    selected_options = filter_delta_options(option_summaries)
    print(f'{currency}: received {len(option_summaries)} option summaries')
    print(f'{currency}: selected {len(selected_options)} options after delta filter')
    if selected_options:
        print(f'{currency}: selected instruments {[opt.get("instrument_name") for opt in selected_options]}')

    row_batch = [build_option_row(currency, option_summary, curr_time) for option_summary in selected_options]
    print(
        f'fetched {len(selected_options)} options for {currency} '
        f'(25-delta call, ATM, 25-delta put from {len(option_summaries)} total)'
    )
    return row_batch


async def scrape_deribit_vol(data_sink=None):
    option_rows = []
    data_sink = data_sink or create_data_sink()
    next_run = time.monotonic()

    async with aiohttp.ClientSession() as session:
        while True:
            cycle_start = time.monotonic()
            curr_time = datetime.now(timezone.utc).isoformat()

            try:
                row_batches = await asyncio.gather(
                    *(collect_currency_option_rows(session, currency, curr_time) for currency in CURRENCIES),
                    return_exceptions=True,
                )

                for currency, row_batch in zip(CURRENCIES, row_batches):
                    if isinstance(row_batch, Exception):
                        print(f'deribit scrape failed for {currency}: {type(row_batch).__name__}: {row_batch}')
                        continue
                    option_rows.extend(row_batch)

                while len(option_rows) >= OPTION_BATCH_SIZE:
                    option_chunk = option_rows[:OPTION_BATCH_SIZE]
                    if not await data_sink.submit_rows('deribit_option_vols', option_chunk):
                        break

                    print(f'inserted {len(option_chunk)} option rows')
                    del option_rows[:OPTION_BATCH_SIZE]

                if option_rows:
                    if await data_sink.submit_rows('deribit_option_vols', option_rows):
                        print(f'inserted {len(option_rows)} remaining option rows')
                    option_rows.clear()
            except asyncio.CancelledError:
                if option_rows:
                    await data_sink.submit_rows('deribit_option_vols', option_rows)
                    option_rows.clear()
                raise
            except Exception as e:
                if option_rows:
                    await data_sink.submit_rows('deribit_option_vols', option_rows)
                    option_rows.clear()
                print(f'deribit scrape failed: {type(e).__name__}: {e}')

            elapsed = time.monotonic() - cycle_start
            print(f'deribit cycle completed in {elapsed:.2f}s')
            next_run += POLL_SECONDS
            sleep_for = next_run - time.monotonic()
            if sleep_for < 0:
                next_run = time.monotonic()
                sleep_for = 0
            await asyncio.sleep(sleep_for)


if __name__ == '__main__':
    asyncio.run(scrape_deribit_vol())

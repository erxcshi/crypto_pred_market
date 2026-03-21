import asyncio
import time
from datetime import datetime, timezone

import aiohttp

from crypto_pred_market.config import create_data_sink


DERIBIT_BASE = 'https://www.deribit.com/api/v2'
CURRENCIES = ['BTC', 'ETH']
OPTION_BATCH_SIZE = 200
HV_BATCH_SIZE = 50
POLL_SECONDS = 5
INSERT_RETRIES = 3


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


def filter_delta_options(option_summaries):
    """Filter to keep 25-delta call, 25-delta put, and ATM for nearest expiration using actual delta values."""
    if not option_summaries:
        return []

    # Group by expiration
    by_expiry = {}
    for opt in option_summaries:
        parsed = parse_instrument_name(opt.get('instrument_name', ''))
        if not parsed:
            continue

        expiry = parsed['expiry']
        if expiry not in by_expiry:
            by_expiry[expiry] = []
        by_expiry[expiry].append(opt)

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

        delta = opt.get('delta')
        if delta is None:
            continue

        opt_type = parsed['type']

        if opt_type == 'C':
            calls.append((opt, delta))
        elif opt_type == 'P':
            puts.append((opt, delta))

    # Find 25-delta call (delta ~0.25)
    valid_calls = [(opt, delta) for opt, delta in calls if 0.20 <= delta <= 0.30]
    if valid_calls:
        best_call = min(valid_calls, key=lambda x: abs(x[1] - 0.25))
        selected.append(best_call[0])

    # Find ATM call (delta ~0.5)
    valid_atm_calls = [(opt, delta) for opt, delta in calls if 0.45 <= delta <= 0.55]
    if valid_atm_calls:
        best_atm = min(valid_atm_calls, key=lambda x: abs(x[1] - 0.5))
        if best_atm[0] not in selected:
            selected.append(best_atm[0])

    # Find 25-delta put (delta ~-0.25)
    valid_puts = [(opt, delta) for opt, delta in puts if -0.30 <= delta <= -0.20]
    if valid_puts:
        best_put = min(valid_puts, key=lambda x: abs(x[1] - (-0.25)))
        if best_put[0] not in selected:
            selected.append(best_put[0])

    return selected


def build_option_row(currency, option_summary):
    return {
        'curr_time': datetime.now(timezone.utc).isoformat(),
        'currency': currency,
        'instrument_name': option_summary.get('instrument_name'),
        'underlying_price': option_summary.get('underlying_price'),
        'delta': option_summary.get('delta'),
        'mark_iv': option_summary.get('mark_iv'),
        'bid_price': option_summary.get('bid_price'),
        'ask_price': option_summary.get('ask_price'),
        'mark_price': option_summary.get('mark_price'),
        'open_interest': option_summary.get('open_interest'),
        'volume': option_summary.get('volume'),
    }


async def scrape_deribit_vol(data_sink=None):
    option_rows = []
    data_sink = data_sink or create_data_sink()

    async with aiohttp.ClientSession() as session:
        while True:
            start = time.time()

            try:
                for currency in CURRENCIES:
                    option_summaries = await deribit_get(
                        session,
                        'public/get_book_summary_by_currency',
                        {'currency': currency, 'kind': 'option'},
                    )

                    # Filter to 25-delta call, ATM, and 25-delta put
                    selected_options = filter_delta_options(option_summaries)

                    for option_summary in selected_options:
                        row_data = build_option_row(currency, option_summary)
                        option_rows.append(row_data)

                    print(f'fetched {len(selected_options)} options for {currency} (25-delta call, ATM, 25-delta put from {len(option_summaries)} total)')

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

            elapsed = time.time() - start
            await asyncio.sleep(max(0, POLL_SECONDS - elapsed))


if __name__ == '__main__':
    asyncio.run(scrape_deribit_vol())

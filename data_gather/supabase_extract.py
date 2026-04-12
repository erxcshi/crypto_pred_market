from config import create_supabase_client
from pathlib import Path
import pandas as pd
from postgrest.exceptions import APIError

client = create_supabase_client()

def fetch_data(client, table_name, columns='*', limit=None):
    batch_size = 250
    all_rows = []
    start = 0

    while True:
        end = start + batch_size - 1

        print(f'fetching {table_name} rows {start} to {end}')
        try:
            response = client.table(table_name).select(columns).range(start, end).execute()
        except APIError as e:
            print(f'{table_name} failed for rows {start} to {end}: {e}')
            raise

        rows = response.data or []

        if not rows:
            break

        all_rows.extend(rows)

        if limit is not None and len(all_rows) >= limit:
            return pd.DataFrame(all_rows[:limit])

        if len(rows) < batch_size:
            break

        start += batch_size

    return pd.DataFrame(all_rows)

# WE HAVE TO REMEMBER TO REMOVE THE LAST KALSHI/POLYMARKET 15M ROWS BEFORE TRAINING, BECAUSE THEY ARE INCOMPLETE
def main():
    client = create_supabase_client()
    raw_data_dir = Path(__file__).resolve().parent / 'raw_data'

    kalshi_df = fetch_data(client, 'kalshi_markets')
    coinbase_df = fetch_data(client, 'coinbase_trades')
    options_df = fetch_data(client, 'deribit_option_vols')
    polymarket_df = fetch_data(client, 'polymarket_markets')
    
    for df in [kalshi_df, coinbase_df, options_df, polymarket_df]:
        df.reset_index(drop=True, inplace=True )
        df['id'] = df.index

    kalshi_df.to_csv(raw_data_dir / 'kalshi_markets.csv', index=False)
    coinbase_df.to_csv(raw_data_dir / 'coinbase_trades.csv', index=False)
    options_df.to_csv(raw_data_dir / 'deribit_option_vols.csv', index=False)
    polymarket_df.to_csv(raw_data_dir / 'polymarket_markets.csv', index=False)

    print(f'csvs generated in {raw_data_dir}')

if __name__ == "__main__":
    main()

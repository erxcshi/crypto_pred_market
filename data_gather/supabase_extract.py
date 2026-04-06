from crypto_pred_market.config import create_supabase_client
from pathlib import Path
import pandas as pd

client = create_supabase_client()

def fetch_data(client, table_name, columns='*', limit=None):
    batch_size=1000 

    

    all_rows = []
    start = 0

    while True:
        end = start + batch_size - 1

        if limit is not None:
            response = client.table(table_name).select(columns).limit(limit).range(start, end).execute()
            if len(all_rows) >= limit: 
                return pd.DataFrame(all_rows[:limit])
        else:
            response = client.table(table_name).select(columns).range(start, end).execute()
        rows = response.data or []

        if not rows:
            break
        
        all_rows.extend(rows)

        if len(rows) < batch_size:
            break

        start += batch_size

    return pd.DataFrame(all_rows)

# WE HAVE TO REMEMBER TO REMOVE THE LAST KALSHI/POLYMARKET 15M ROWS BEFORE TRAINING, BECAUSE THEY ARE INCOMPLETE
def main():
    client = create_supabase_client()
    output_dir = Path(__file__).resolve().parents[0] / 'data_files'

    kalshi_df = fetch_data(client, 'kalshi_markets')
    coinbase_df = fetch_data(client, 'coinbase_trades')
    options_df = fetch_data(client, 'deribit_option_vols')
    polymarket_df = fetch_data(client, 'polymarket_markets')
    
    for df in [kalshi_df, coinbase_df, options_df, polymarket_df]:
        df.reset_index(drop=True, inplace=True )
        df['id'] = df.index

    kalshi_df.to_csv(output_dir / 'kalshi_markets.csv', index=False)
    coinbase_df.to_csv(output_dir / 'coinbase_trades.csv', index=False)
    options_df.to_csv(output_dir / 'deribit_option_vols.csv', index=False)
    polymarket_df.to_csv(output_dir / 'polymarket_markets.csv', index=False)

    print(f'csvs generated in {output_dir}')

if __name__ == "__main__":
    main()

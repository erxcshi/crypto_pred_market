from crypto_pred_market.config import create_supabase_client
from pathlib import Path
import pandas as pd

client = create_supabase_client()

def fetch_data(client, table_name, columns='*', limit = ''): 
    response = client.table(table_name).select(columns).limit(limit).execute()
    return pd.DataFrame(response.data)

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

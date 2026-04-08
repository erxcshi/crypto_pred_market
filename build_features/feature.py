import polars as pl 
from pathlib import Path

data_files_dir = Path(__file__).resolve().parent.parent / 'data_gather' / 'data_files'
print(data_files_dir)
# df = pl.scan_csv(data_files_dir / 'kalshi_markets.csv').collect()
# def build_coin_df(coin): 
#     df_copy = df[df['coin'] == coin].copy()
#     col_identifiers = ['open_time', 'close_time']

#     df_copy['next_price_dollars_lead1'] = df_copy.groupby(col_identifiers)['last_price_dollars'].shift(-1)
#     df_copy = df_copy[df_copy['time_to_close'] > -0.01]

#     df_copy = df_copy.add_prefix(f'{coin}_')
    
#     return df_copy

# df.head(5)


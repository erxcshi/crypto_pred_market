import pandas as pd
import polars as pl
from pathlib import Path

raw_data_dir = Path.cwd().parent / 'data_gather' / 'raw_data'
filtered_data_dir = Path.cwd().parent / 'data_gather' / 'filtered_data'


trades = pl.scan_csv(raw_data_dir / 'coinbase_trades.csv').collect()
options = pl.scan_csv(raw_data_dir / 'deribit_option_vols.csv').collect()
kalshi = pl.scan_csv(raw_data_dir / 'kalshi_markets.csv').collect()
polymarket = pl.scan_csv(raw_data_dir / 'polymarket_markets.csv').collect()


class filter_data(): 
    def __init__(self
                 , trades: pl.DataFrame
                 , options: pl.DataFrame
                 , kalshi: pl.DataFrame
                 , polymarket: pl.DataFrame):
        
        self.trades = trades
        self.options = options
        self.kalshi = kalshi
        self.polymarket = polymarket
    
    def clean_trades(self, coins: list) -> pl.DataFrame:
        coin_trades = self.trades.filter(pl.col('product_id').is_in([f'{coin}-USD' for coin in coins]))
        coin_trades = coin_trades.unique(subset=['trade_id'])
        coin_trades = coin_trades.sort('trade_time', descending=False)
        coin_trades = coin_trades.drop('id').with_row_index('id')
        coin_trades = coin_trades.select('id', 'curr_time', 'product_id', 'trade_time', 'price', 'size', 'side')
        return coin_trades
    
    def clean_options(self, coins: list) -> pl.DataFrame: 
        coin_options = self.options.filter(pl.col('currency').is_in([f'{coin}' for coin in coins]))
        coin_options = coin_options.sort('curr_time', descending=False)
        coin_options = coin_options.drop('id').with_row_index('id')
        coin_options = coin_options.select('id', 'curr_time', 'instrument_name', 'expiry_datetime', 'strike', 
                                           'option_type', 'underlying_price', 'delta', 'mark_iv', 'mark_price', 
                                           'open_interest', 'volume')
        return coin_options
    
    def clean_kalshi(self, coins: list) -> pl.DataFrame: 
        coin_kalshi = self.kalshi.filter(pl.col('coin').is_in([f'{coin}' for coin in coins]))
        coin_kalshi = coin_kalshi.sort('curr_time', descending=False)
        coin_kalshi = coin_kalshi.drop('id').with_row_index('id')
        coin_kalshi = coin_kalshi.select('id', 'curr_time', 'coin', 'open_time', 'close_time', 
                                         'last_price_dollars' , 'no_ask_dollars', 'no_bid_dollars', 
                                         'yes_ask_dollars', 'yes_bid_dollars', 'floor_strike', 'volume_24h_fp')
        return coin_kalshi
    
    def clean_polymarket(self, coins: list) -> pl.DataFrame:
        coin_polymarket = self.polymarket.filter(pl.col('coin').is_in([f'{coin}' for coin in coins]))
        coin_polymarket = coin_polymarket.sort('curr_time', descending=False)
        coin_polymarket = coin_polymarket.drop('id').with_row_index('id')
        coin_polymarket = coin_polymarket.select('id', 'curr_time', 'coin', 'interval_start_unix', 'end_date', 'strike_price',
                                                 'liquidity', 'volume', 'yes_implied_price', 'no_implied_price', 'yes_buy_price',
                                                 'yes_sell_price', 'no_buy_price', 'no_sell_price')
        return coin_polymarket
    

if __name__ == "__main__":
    filterer = filter_data(trades=trades, options=options, kalshi=kalshi, polymarket=polymarket)
    btc_trades = filterer.clean_trades(['BTC']).write_csv(filtered_data_dir / 'btc_trades.csv')
    btc_eth_options = filterer.clean_options(['BTC', 'ETH']).write_csv(filtered_data_dir / 'btc_eth_options.csv')
    all_coins_kalshi = filterer.clean_kalshi(['BTC', 'ETH', 'XRP', 'SOL']).write_csv(filtered_data_dir / 'all_kalshi.csv')
    all_coins_polymarket = filterer.clean_polymarket(['BTC', 'ETH', 'XRP', 'SOL']).write_csv(filtered_data_dir / 'all_polymarket.csv')
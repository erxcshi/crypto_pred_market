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
        first_event_time, last_event_time = self.kalshi.select(pl.col('open_time').min(), pl.col('close_time').max()).row(0)
        coin_kalshi = self.kalshi.filter(pl.col('coin').is_in([f'{coin}' for coin in coins]))
        coin_kalshi = coin_kalshi.filter(~((pl.col("open_time") == first_event_time) | (pl.col("close_time") == last_event_time)))
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
    
class build_main_df():
    def __init__(self, trades: pl.DataFrame, options: pl.DataFrame, kalshi: pl.DataFrame, polymarket: pl.DataFrame):
        self.trades = trades
        self.options = options
        self.kalshi = kalshi
        self.polymarket = polymarket
    
    def _to_datetime(table: pl.DataFrame, cols: list) -> pl.DataFrame:
        for col in cols: 
            table = table.with_columns(
                pl.col(col)
                .str.to_datetime(format="%Y-%m-%d %H:%M:%S%.f%#z", strict=False)
                .alias(col)
            )
        return table

    def normalize_trades(self):
        btc_trades = self._to_datetime(self.btc_trades, ['curr_time', 'trade_time'])
        btc_trades = btc_trades.with_columns(pl.col("product_id").str.replace("-USD", "").alias("coin"))
        btc_trades = btc_trades.drop('product_id')
        btc_trades = btc_trades.with_columns(((pl.col("price") * pl.col("size")) * pl.when(pl.col("side") == "buy").then(1).otherwise(-1)).alias("weighted"))

    def normalize_options(self):
        btc_eth_options = self._to_datetime(self.btc_eth_options, ['curr_time', 'expiry_datetime'])
    
    def normalize_kalshi(self):
        kalshi = self._to_datetime(self.kalshi, ['open_time', 'curr_time', 'close_time'])
        kalshi = kalshi.with_columns((pl.col('close_time') - pl.col('curr_time')).dt.total_seconds().alias('time_to_close'))
        kalshi = kalshi.with_columns(pl.col("curr_time").shift(1).over("coin").alias("prev_time")
                             ).with_columns((pl.col("curr_time") - pl.col("prev_time")).dt.total_seconds().alias("time_diff_seconds"))

        kalshi = kalshi.filter(pl.col("time_diff_seconds").is_between(0.9, 1.1))

        
    def normalize_polymarket(self):
        self.all_coins_polymarket = self._to_datetime(self.all_coins_polymarket, ['curr_time', 'end_date'])

    def build_main_df(self):
        pass
if __name__ == "__main__":
    filterer = filter_data(trades=trades, options=options, kalshi=kalshi, polymarket=polymarket)
    btc_trades = filterer.clean_trades(['BTC']).write_csv(filtered_data_dir / 'btc_trades.csv')
    btc_eth_options = filterer.clean_options(['BTC', 'ETH']).write_csv(filtered_data_dir / 'btc_eth_options.csv')
    all_coins_kalshi = filterer.clean_kalshi(['BTC', 'ETH', 'XRP', 'SOL']).write_csv(filtered_data_dir / 'all_kalshi.csv')
    all_coins_polymarket = filterer.clean_polymarket(['BTC', 'ETH', 'XRP', 'SOL']).write_csv(filtered_data_dir / 'all_polymarket.csv')
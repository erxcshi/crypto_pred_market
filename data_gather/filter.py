import polars as pl
from pathlib import Path

MODULE_DIR = Path(__file__).resolve().parent
raw_data_dir = MODULE_DIR / 'raw_data'
filtered_data_dir = MODULE_DIR / 'filtered_data'
final_data_dir = MODULE_DIR / 'final_data'


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
                                         'yes_ask_dollars', 'yes_bid_dollars', 'floor_strike')
        return coin_kalshi
    
    def clean_polymarket(self, coins: list) -> pl.DataFrame:
        coin_polymarket = self.polymarket.filter(pl.col('coin').is_in([f'{coin}' for coin in coins]))
        coin_polymarket = coin_polymarket.sort('curr_time', descending=False)
        coin_polymarket = coin_polymarket.drop('id').with_row_index('id')
        coin_polymarket = coin_polymarket.select('id', 'curr_time', 'coin', 'interval_start_unix', 'end_date', 'strike_price',
                                                 'liquidity', 'volume', 'yes_implied_price', 'no_implied_price', 'yes_buy_price',
                                                 'yes_sell_price', 'no_buy_price', 'no_sell_price')
        return coin_polymarket



def _to_datetime(table: pl.DataFrame, cols: list[str]) -> pl.DataFrame:
    for col in cols:
        table = table.with_columns(
            pl.col(col).str.to_datetime(format="%Y-%m-%d %H:%M:%S%.f%#z", strict=False).alias(col)
        )
    return table


def load_filtered_feature_inputs(base_dir: Path = filtered_data_dir) -> tuple[pl.DataFrame, pl.DataFrame, pl.DataFrame]:
    all_trades = pl.scan_csv(base_dir / "all_trades.csv").collect()
    all_kalshi = pl.scan_csv(base_dir / "all_kalshi.csv").collect()
    all_polymarket = pl.scan_csv(base_dir / "all_polymarket.csv").collect()

    all_trades = _to_datetime(all_trades, ["curr_time", "trade_time"])
    all_kalshi = _to_datetime(all_kalshi, ["curr_time", "open_time", "close_time"])
    all_polymarket = _to_datetime(all_polymarket, ["curr_time", "end_date"])

    return all_trades, all_kalshi, all_polymarket


def prepare_all_trades(all_trades: pl.DataFrame) -> pl.DataFrame:
    return (
        all_trades
        .with_columns([
            pl.col("product_id").str.replace("-USD", "").alias("coin"),
            pl.when(pl.col("side") == "buy").then(1).otherwise(-1).alias("trade_sign"),
            (pl.col("price") * pl.col("size")).alias("notional"),
            (
                (pl.col("price") * pl.col("size"))
                * pl.when(pl.col("side") == "buy").then(1).otherwise(-1)
            ).alias("weighted"),
            (
                pl.col("size")
                * pl.when(pl.col("side") == "buy").then(1).otherwise(-1)
            ).alias("signed_size"),
        ])
        .sort("trade_time")
    )


def prepare_all_kalshi(all_coins_kalshi: pl.DataFrame) -> pl.DataFrame:
    return (
        all_coins_kalshi
        .with_columns(
            (pl.col("close_time") - pl.col("curr_time")).dt.total_seconds().alias("time_to_close")
        )
        .with_columns(
            pl.col("floor_strike")
            .fill_null(strategy="forward")
            .fill_null(strategy="backward")
            .over(["coin", "open_time", "close_time"])
            .alias("floor_strike")
        )
        .unique(
            subset=["coin", "open_time", "close_time", "time_to_close"],
            keep="first",
        )
        .filter(pl.col("floor_strike").is_not_null())
        .sort(["coin", "open_time", "close_time", "curr_time"])
    )


def build_coin_df(
    all_coins_kalshi: pl.DataFrame,
    all_trades: pl.DataFrame,
    coin: str,
    prefix: bool = True,
    add_target: bool = False,
    include_event_metadata: bool = True,
    include_time_to_close: bool = True,
) -> pl.DataFrame:
    event_cols = ["coin", "open_time", "close_time"]
    kalshi = prepare_all_kalshi(all_coins_kalshi)
    trades = prepare_all_trades(all_trades)
    spot_trades = (
        trades
        .filter(pl.col("coin") == coin)
        .select("trade_time", "price")
        .sort("trade_time")
    )
    column_order = ["curr_time"]

    if include_event_metadata:
        column_order.extend(["open_time", "close_time", "prev_time"])

    if include_time_to_close:
        column_order.append("time_to_close")

    column_order.extend([
        "last_price_dollars",
        "yes_mid_dollars",
        "yes_spread_dollars",
        "distance_from_strike",
        "yes_mid_change_1s",
        "yes_mid_change_5s",
        "yes_mid_change_std_30s",
        "yes_mid_change_std_60s",
        "yes_spread_mean_30s",
    ])

    if add_target:
        column_order.append("next_price_dollars_lead1")

    build_exprs = [
        pl.col("curr_time").shift(1).over(event_cols).alias("prev_time"),
    ]
    if add_target:
        build_exprs.append(
            pl.col("last_price_dollars")
            .shift(-7)
            .over(event_cols)
            .alias("next_price_dollars_lead1")
        )

    coin_df = (
        kalshi
        .filter((pl.col("coin") == coin) & (pl.col("time_to_close") >= 0))
        .with_columns(build_exprs)
        .with_columns(
            (pl.col("curr_time") - pl.col("prev_time")).dt.total_seconds().alias("time_diff_seconds")
        )
        .filter(pl.col("time_diff_seconds").is_between(0.9, 1.1))
        .sort("curr_time")
        .join_asof(
            spot_trades,
            left_on="curr_time",
            right_on="trade_time",
            strategy="backward",
        )
        .with_columns([
            ((pl.col("yes_ask_dollars") + pl.col("yes_bid_dollars")) / 2).alias("yes_mid_dollars"),
            (pl.col("yes_ask_dollars") - pl.col("yes_bid_dollars")).alias("yes_spread_dollars"),
            (pl.col("price") - pl.col("floor_strike")).alias("distance_from_strike"),
        ])
        .with_columns([
            pl.col("yes_mid_dollars").diff().over(event_cols).fill_null(0).alias("yes_mid_change_1s"),
            pl.col("yes_mid_dollars").diff(5).over(event_cols).fill_null(0).alias("yes_mid_change_5s"),
        ])
        .with_columns([
            pl.col("yes_mid_change_1s").rolling_std(window_size=30).over(event_cols).fill_null(0).alias("yes_mid_change_std_30s"),
            pl.col("yes_mid_change_1s").rolling_std(window_size=60).over(event_cols).fill_null(0).alias("yes_mid_change_std_60s"),
            pl.col("yes_spread_dollars").rolling_mean(window_size=30).over(event_cols).fill_null(pl.col("yes_spread_dollars")).alias("yes_spread_mean_30s"),
        ])
        .select(column_order)
    )

    if prefix:
        rename_map = {
            col: f"{coin}_{col}"
            for col in coin_df.columns
            if col != "curr_time"
        }
        coin_df = coin_df.rename(rename_map)

    return coin_df


def build_btc_kalshi(all_coins_kalshi: pl.DataFrame, all_trades: pl.DataFrame) -> pl.DataFrame:
    btc_kalshi = build_coin_df(
        all_coins_kalshi=all_coins_kalshi,
        all_trades=all_trades,
        coin="BTC",
        prefix=False,
        add_target=True,
    ).with_row_index("btc_row_id")

    outcome_map = (
        btc_kalshi
        .sort("curr_time")
        .group_by(["open_time", "close_time"])
        .agg(
            pl.col("yes_mid_dollars").last().alias("last_mid_dollars")
        )
        .with_columns(
            (pl.col("last_mid_dollars") > 0.5).cast(pl.Int8).alias("outcome")
        )
        .select(["open_time", "close_time", "outcome"])
    )

    return btc_kalshi.join(
        outcome_map,
        on=["open_time", "close_time"],
        how="left",
    )


def attach_cross_coin_kalshi_features(
    btc_kalshi: pl.DataFrame,
    all_coins_kalshi: pl.DataFrame,
    all_trades: pl.DataFrame,
    coins: tuple[str, ...] = ("ETH", "XRP", "SOL"),
) -> pl.DataFrame:
    combined = btc_kalshi.sort("curr_time")

    for coin in coins:
        combined = combined.join_asof(
            build_coin_df(
                all_coins_kalshi,
                all_trades,
                coin,
                include_event_metadata=False,
                include_time_to_close=False,
            ).sort("curr_time"),
            on="curr_time",
            strategy="backward",
        )

    return combined


def build_btc_spot_features(all_trades: pl.DataFrame) -> pl.DataFrame:
    btc_trades = (
        prepare_all_trades(all_trades)
        .filter(pl.col("coin") == "BTC")
        .sort("trade_time")
    )

    return (
        btc_trades
        .group_by_dynamic(
            "trade_time",
            every="1s",
            period="1s",
            closed="right",
            label="right",
        )
        .agg([
            pl.col("price").last().alias("btc_spot_price"),
            pl.col("size").sum().alias("btc_spot_size_1s"),
            pl.col("signed_size").sum().alias("btc_spot_signed_size_1s"),
        ])
        .sort("trade_time")
        .with_columns([
            ((pl.col("btc_spot_price") / pl.col("btc_spot_price").shift(1)) - 1).fill_null(0).alias("btc_spot_return_1s"),
            ((pl.col("btc_spot_price") / pl.col("btc_spot_price").shift(5)) - 1).fill_null(0).alias("btc_spot_return_5s"),
            ((pl.col("btc_spot_price") / pl.col("btc_spot_price").shift(15)) - 1).fill_null(0).alias("btc_spot_return_15s"),
            ((pl.col("btc_spot_price") / pl.col("btc_spot_price").shift(60)) - 1).fill_null(0).alias("btc_spot_return_60s"),
        ])
        .with_columns([
            pl.col("btc_spot_return_1s").rolling_std(window_size=30).fill_null(0).alias("btc_spot_return_vol_30s"),
            pl.col("btc_spot_return_1s").rolling_std(window_size=300).fill_null(0).alias("btc_spot_return_vol_5m"),
            pl.col("btc_spot_signed_size_1s").rolling_mean(window_size=30).fill_null(0).alias("btc_spot_signed_flow_mean_30s"),
            pl.col("btc_spot_size_1s").rolling_mean(window_size=30).fill_null(0).alias("btc_spot_size_mean_30s"),
        ])
    )


def attach_btc_spot_features(combined: pl.DataFrame, all_trades: pl.DataFrame) -> pl.DataFrame:
    btc_spot_features = build_btc_spot_features(all_trades)

    return (
        combined
        .sort("curr_time")
        .join_asof(
            btc_spot_features,
            left_on="curr_time",
            right_on="trade_time",
            strategy="backward",
        )
        .with_columns([
            pl.col("btc_spot_size_1s").fill_null(0),
            pl.col("btc_spot_signed_size_1s").fill_null(0),
            pl.col("btc_spot_return_1s").fill_null(0),
            pl.col("btc_spot_return_5s").fill_null(0),
            pl.col("btc_spot_return_15s").fill_null(0),
            pl.col("btc_spot_return_60s").fill_null(0),
            pl.col("btc_spot_return_vol_30s").fill_null(0),
            pl.col("btc_spot_return_vol_5m").fill_null(0),
            pl.col("btc_spot_signed_flow_mean_30s").fill_null(0),
            pl.col("btc_spot_size_mean_30s").fill_null(0),
        ])
        .drop(["btc_row_id", "trade_time"])
    )


def build_btc_polymarket_df(all_polymarket: pl.DataFrame, all_trades: pl.DataFrame) -> pl.DataFrame:
    event_cols = ["coin", "interval_start_unix", "end_date"]
    btc_trades = (
        prepare_all_trades(all_trades)
        .filter(pl.col("coin") == "BTC")
        .select("trade_time", "price")
        .sort("trade_time")
    )

    btc_polymarket = (
        all_polymarket
        .filter(pl.col("coin") == "BTC")
        .with_columns(
            (pl.col("end_date") - pl.col("curr_time")).dt.total_seconds().alias("time_to_close")
        )
        .filter((pl.col("time_to_close") >= 0) & (pl.col("time_to_close") <= 900))
        .unique(
            subset=["coin", "interval_start_unix", "end_date", "time_to_close"],
            keep="first",
        )
        .sort(event_cols + ["curr_time"])
        .with_columns([
            pl.col("curr_time").shift(1).over(event_cols).alias("prev_time"),
            pl.col("volume").diff().over(event_cols).fill_null(0).alias("volume_delta_raw"),
        ])
        .with_columns(
            (pl.col("curr_time") - pl.col("prev_time")).dt.total_seconds().alias("time_diff_seconds")
        )
        .join_asof(
            btc_trades,
            left_on="curr_time",
            right_on="trade_time",
            strategy="backward",
        )
        .with_columns([        
            ((pl.col("yes_buy_price") + pl.col("yes_sell_price")) / 2).alias("yes_mid_price"),
            (pl.col("yes_buy_price") - pl.col("yes_sell_price")).alias("yes_spread_price"),
            (pl.col("price") - pl.col("strike_price")).alias("distance_from_strike"),
            pl.when(pl.col("volume_delta_raw") < 0).then(0).otherwise(pl.col("volume_delta_raw")).alias("volume_delta"),
        ])
        .select([
            "curr_time",
            "interval_start_unix",
            "end_date",
            "time_to_close",
            "strike_price",
            "liquidity",
            "volume",
            "volume_delta",
            "yes_implied_price",
            "yes_mid_price",
            "yes_spread_price",
            "distance_from_strike",
        ])
        .rename({
            "interval_start_unix": "poly_interval_start_unix",
            "end_date": "poly_end_date",
            "time_to_close": "poly_time_to_close",
            "strike_price": "poly_strike_price",
            "liquidity": "poly_liquidity",
            "volume": "poly_volume",
            "volume_delta": "poly_volume_delta",
            "yes_implied_price": "poly_yes_implied_price",
            "yes_mid_price": "poly_yes_mid_price",
            "yes_spread_price": "poly_yes_spread_price",
            "distance_from_strike": "poly_distance_from_strike",
        })
        .sort("curr_time")
    )

    return btc_polymarket


def attach_btc_polymarket_features(
    combined: pl.DataFrame,
    all_polymarket: pl.DataFrame,
    all_trades: pl.DataFrame,
) -> tuple[pl.DataFrame, pl.DataFrame]:
    btc_polymarket = build_btc_polymarket_df(all_polymarket, all_trades)
    combined = combined.sort("curr_time").join_asof(
        btc_polymarket,
        on="curr_time",
        strategy="backward",
    )
    return combined, btc_polymarket


def summarize_nulls(df: pl.DataFrame) -> pl.DataFrame:
    if df.height == 0:
        return pl.DataFrame(
            schema={
                "column": pl.String,
                "null_count": pl.Int64,
                "null_share": pl.Float64,
            }
        )

    return (
        pl.DataFrame({
            "column": df.columns,
            "null_count": [df.get_column(col).null_count() for col in df.columns],
        })
        .with_columns(
            (pl.col("null_count") / df.height).alias("null_share")
        )
        .filter(pl.col("null_count") > 0)
        .sort("null_count", descending=True)
    )


def prepare_training_frame(
    df: pl.DataFrame,
    target_col: str = "next_price_dollars_lead1",
    drop_null_rows: bool = True,
) -> tuple[pl.DataFrame, pl.DataFrame]:
    training_df = df.filter(pl.col(target_col).is_not_null())

    if drop_null_rows:
        training_df = training_df.drop_nulls()

    return training_df, summarize_nulls(training_df)


def build_training_ready_btc_feature_table(
    base_dir: Path = filtered_data_dir,
) -> tuple[pl.DataFrame, pl.DataFrame]:
    combined = build_btc_feature_tables(base_dir)
    return prepare_training_frame(combined)


def build_btc_feature_tables(
    base_dir: Path = filtered_data_dir,
) -> pl.DataFrame:
    all_trades, all_coins_kalshi, all_polymarket = load_filtered_feature_inputs(base_dir)

    btc_kalshi = build_btc_kalshi(all_coins_kalshi, all_trades)
    combined = attach_cross_coin_kalshi_features(btc_kalshi, all_coins_kalshi, all_trades)
    combined = attach_btc_spot_features(combined, all_trades)
    # combined, btc_polymarket = attach_btc_polymarket_features(combined, all_polymarket, all_trades)

    # return combined, btc_kalshi, btc_polymarket
    return combined
    
# class build_main_df():
#     def __init__(self, trades: pl.DataFrame, options: pl.DataFrame, kalshi: pl.DataFrame, polymarket: pl.DataFrame):
#         self.trades = trades
#         self.options = options
#         self.kalshi = kalshi
#         self.polymarket = polymarket
    
#     def _to_datetime(table: pl.DataFrame, cols: list) -> pl.DataFrame:
#         for col in cols: 
#             table = table.with_columns(
#                 pl.col(col)
#                 .str.to_datetime(format="%Y-%m-%d %H:%M:%S%.f%#z", strict=False)
#                 .alias(col)
#             )
#         return table

#     def normalize_trades(self):
#         btc_trades = self._to_datetime(self.btc_trades, ['curr_time', 'trade_time'])
#         btc_trades = btc_trades.with_columns(pl.col("product_id").str.replace("-USD", "").alias("coin"))
#         btc_trades = btc_trades.drop('product_id')
#         btc_trades = btc_trades.with_columns(((pl.col("price") * pl.col("size")) * pl.when(pl.col("side") == "buy").then(1).otherwise(-1)).alias("weighted"))

#     def normalize_options(self):
#         btc_eth_options = self._to_datetime(self.btc_eth_options, ['curr_time', 'expiry_datetime'])
    
#     def normalize_kalshi(self):
#         kalshi = self._to_datetime(self.kalshi, ['open_time', 'curr_time', 'close_time'])
#         first_event_time, last_event_time = kalshi.select(pl.col('open_time').min(), pl.col('close_time').max()).row(0)
#         kalshi = kalshi.filter(~((pl.col("open_time") == first_event_time) | (pl.col("close_time") == last_event_time)))
#         kalshi = kalshi.with_columns((pl.col('close_time') - pl.col('curr_time')).dt.total_seconds().alias('time_to_close'))
#         kalshi = kalshi.with_columns(pl.col("curr_time").shift(1).over("coin").alias("prev_time")
#                              ).with_columns((pl.col("curr_time") - pl.col("prev_time")).dt.total_seconds().alias("time_diff_seconds"))
#         kalshi = kalshi.filter(pl.col("time_diff_seconds").is_between(0.9, 1.1))

        
#     def normalize_polymarket(self):
#         self.all_coins_polymarket = self._to_datetime(self.all_coins_polymarket, ['curr_time', 'end_date'])

#     def build_main_df(self):
#         pass
if __name__ == "__main__":
    # trades = pl.scan_csv(raw_data_dir / 'coinbase_trades.csv').collect()
    # options = pl.scan_csv(raw_data_dir / 'deribit_option_vols.csv').collect()
    # kalshi = pl.scan_csv(raw_data_dir / 'kalshi_markets.csv').collect()
    # polymarket = pl.scan_csv(raw_data_dir / 'polymarket_markets.csv').collect()
    # filterer = filter_data(trades=trades, options=options, kalshi=kalshi, polymarket=polymarket)
    # btc_trades = filterer.clean_trades(['BTC', 'ETH', 'XRP', 'SOL']).write_csv(filtered_data_dir / 'all_trades.csv')

    
    # btc_eth_options = filterer.clean_options(['BTC', 'ETH']).write_csv(filtered_data_dir / 'btc_eth_options.csv')
    # all_coins_kalshi = filterer.clean_kalshi(['BTC', 'ETH', 'XRP', 'SOL']).write_csv(filtered_data_dir / 'all_kalshi.csv')
    # all_coins_polymarket = filterer.clean_polymarket(['BTC', 'ETH', 'XRP', 'SOL']).write_csv(filtered_data_dir / 'all_polymarket.csv')

    combined = build_btc_feature_tables()
    combined.write_csv(r'C:\Users\erics\courses\gradml1\crypto_pred_market\data_gather\final_data\final_data.csv')

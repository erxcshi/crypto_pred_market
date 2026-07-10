"""
Explicit pyarrow schemas for every collected table.

Why explicit: every parquet file for a table must share an identical schema so
downstream `pl.scan_parquet("data/{table}/**/*.parquet")` and daily compaction
never hit schema-merge conflicts. Rows are plain dicts from the scrapers;
missing keys become nulls, unknown keys are dropped (warned once), and values
are coerced to the declared type (several APIs return numeric fields as
strings, e.g. Kalshi *_fp fields, Polymarket CLOB prices).

Timestamps are stored as strings — the scrapers already produce ISO-8601
strings and keeping them verbatim avoids any parse/timezone risk at capture
time. Parsing happens at feature-build time.

New tables: add a pa.schema entry here. Tables without an entry get a schema
inferred from their first batch (cached per process) — fine for experiments,
but add the explicit entry before long-running collection.
"""
from __future__ import annotations

import pyarrow as pa

_S = pa.string()
_F = pa.float64()
_I = pa.int64()


SCHEMAS: dict[str, pa.Schema] = {
    # ------------------------------------------------------------------
    # Coinbase (coinbase_ws.py)
    # ------------------------------------------------------------------
    "coinbase_trades": pa.schema([
        ("curr_time", _S),
        ("product_id", _S),
        ("event_type", _S),
        ("trade_time", _S),
        ("trade_id", _I),
        ("sequence", _I),
        ("price", _F),
        ("size", _F),
        ("side", _S),
        ("maker_order_id", _S),
        ("taker_order_id", _S),
    ]),
    "coinbase_quotes": pa.schema([
        ("curr_time", _S),
        ("product_id", _S),
        ("trade_time", _S),
        ("sequence", _I),
        ("price", _F),
        ("best_bid", _F),
        ("best_ask", _F),
        ("best_bid_size", _F),
        ("best_ask_size", _F),
        ("last_size", _F),
        ("side", _S),
    ]),
    # Full-depth level2_batch stream (BTC/ETH): one row per changed level,
    # plus exploded snapshot rows on (re)subscribe.
    "coinbase_l2": pa.schema([
        ("curr_time", _S),
        ("product_id", _S),
        ("event_type", _S),      # 'snapshot' | 'l2update'
        ("exchange_time", _S),   # message 'time' field (l2update only)
        ("side", _S),            # 'buy' | 'sell'
        ("price", _F),
        ("size", _F),            # absolute size at level (0 = level removed)
    ]),

    # ------------------------------------------------------------------
    # Kalshi (kalshi_ws.py + kalshi.py + settlements poller)
    # ------------------------------------------------------------------
    "kalshi_markets": pa.schema([
        ("curr_time", _S),
        ("coin", _S),
        ("open_time", _S),
        ("close_time", _S),
        ("last_price_dollars", _F),
        ("liquidity_dollars", _F),
        ("no_ask_dollars", _F),
        ("no_bid_dollars", _F),
        ("yes_ask_dollars", _F),
        ("yes_bid_dollars", _F),
        ("open_interest", _F),
        ("floor_strike", _F),
        ("cap_strike", _F),
        ("strike_type", _S),
        ("volume", _F),
        ("volume_24h_fp", _F),
    ]),
    "kalshi_orderbook_deltas": pa.schema([
        ("curr_time", _S),
        ("ticker", _S),
        ("coin", _S),
        ("market_id", _S),
        ("ts_ms", _I),
        ("event_type", _S),      # 'snapshot' | 'delta'
        ("side", _S),
        ("price", _F),
        ("delta_size", _F),
    ]),
    "kalshi_trades": pa.schema([
        ("curr_time", _S),
        ("ticker", _S),
        ("coin", _S),
        ("trade_id", _S),
        ("ts_ms", _I),
        ("yes_price", _F),
        ("no_price", _F),
        ("count", _F),
        ("taker_outcome_side", _S),
        ("taker_book_side", _S),
    ]),
    # Fields per the AsyncAPI spec: volume/open_interest are CUMULATIVE
    # (volume_fp / open_interest_fp); there are no *_delta fields.
    "kalshi_ticker": pa.schema([
        ("curr_time", _S),
        ("ticker", _S),
        ("coin", _S),
        ("ts_ms", _I),
        ("price", _F),
        ("yes_bid", _F),
        ("yes_ask", _F),
        ("yes_bid_size", _F),
        ("yes_ask_size", _F),
        ("last_trade_size", _F),
        ("volume", _F),
        ("open_interest", _F),
        ("dollar_volume", _F),
        ("dollar_open_interest", _F),
    ]),
    # Ground-truth labels: settlement record fetched after market close.
    "kalshi_settlements": pa.schema([
        ("curr_time", _S),
        ("ticker", _S),
        ("coin", _S),
        ("series", _S),
        ("open_time", _S),
        ("close_time", _S),
        ("status", _S),
        ("result", _S),                 # 'yes' | 'no'
        ("floor_strike", _F),
        ("cap_strike", _F),
        ("expiration_value", _S),       # settlement print (string verbatim from API)
        ("settlement_value_dollars", _F),
        ("last_price_dollars", _F),
        ("volume", _F),
        ("open_interest", _F),
    ]),
    "kalshi_fills": pa.schema([
        ("curr_time", _S),
        ("ticker", _S),
        ("trade_id", _S),
        ("order_id", _S),
        ("side", _S),
        ("action", _S),
        ("is_taker", _S),
        ("yes_price", _F),
        ("no_price", _F),
        ("count", _F),
        ("ts_ms", _I),
    ]),

    # ------------------------------------------------------------------
    # Polymarket (REST discovery snapshot + WS book events)
    # ------------------------------------------------------------------
    "polymarket_markets": pa.schema([
        ("curr_time", _S),
        ("coin", _S),
        ("time_horizon_minutes", _I),
        ("interval_start_unix", _I),
        ("market_slug", _S),
        ("condition_id", _S),
        ("strike_price", _F),
        ("end_date", _S),
        ("liquidity", _F),
        ("volume", _F),
        ("open_interest", _F),
        ("yes_token_id", _S),
        ("no_token_id", _S),
        ("yes_implied_price", _F),
        ("no_implied_price", _F),
        ("yes_buy_price", _F),
        ("yes_sell_price", _F),
        ("no_buy_price", _F),
        ("no_sell_price", _F),
    ]),
    # One row per WS event side/level (book snapshots exploded) or per
    # price_change / last_trade_price / best_bid_ask event.
    "polymarket_book_events": pa.schema([
        ("curr_time", _S),
        ("coin", _S),
        ("token_id", _S),
        ("market_slug", _S),
        ("event_type", _S),      # 'book' | 'price_change' | 'last_trade_price' | 'tick_size_change' | 'best_bid_ask'
        ("exchange_time", _S),   # ws payload timestamp (ms string)
        ("side", _S),            # 'bid' | 'ask' | '' for trade events
        ("price", _F),
        ("size", _F),
        ("best_bid", _F),
        ("best_ask", _F),
    ]),

    # ------------------------------------------------------------------
    # Deribit (REST poller today; WS scraper in Phase C)
    # ------------------------------------------------------------------
    "deribit_option_vols": pa.schema([
        ("curr_time", _S),
        ("currency", _S),
        ("instrument_name", _S),
        ("expiry_datetime", _S),
        ("strike", _F),
        ("option_type", _S),
        ("underlying_price", _F),
        ("delta", _F),
        ("mark_iv", _F),
        ("bid_price", _F),
        ("ask_price", _F),
        ("mark_price", _F),
        ("open_interest", _F),
        ("volume", _F),
    ]),
    "deribit_dvol": pa.schema([
        ("curr_time", _S),
        ("currency", _S),        # 'BTC' | 'ETH'
        ("exchange_ts_ms", _I),
        ("dvol", _F),
    ]),
    "deribit_ticker": pa.schema([
        ("curr_time", _S),
        ("instrument_name", _S),
        ("exchange_ts_ms", _I),
        ("underlying_price", _F),
        ("mark_price", _F),
        ("mark_iv", _F),
        ("bid_price", _F),
        ("ask_price", _F),
        ("bid_iv", _F),
        ("ask_iv", _F),
        ("delta", _F),
        ("gamma", _F),
        ("vega", _F),
        ("theta", _F),
        ("open_interest", _F),
        ("volume", _F),
    ]),
    "deribit_perp": pa.schema([
        ("curr_time", _S),
        ("instrument_name", _S),
        ("exchange_ts_ms", _I),
        ("last_price", _F),
        ("mark_price", _F),
        ("index_price", _F),
        ("best_bid", _F),
        ("best_ask", _F),
        ("current_funding", _F),
        ("funding_8h", _F),
        ("open_interest", _F),
        ("volume_24h", _F),
    ]),
    "deribit_index": pa.schema([
        ("curr_time", _S),
        ("index_name", _S),      # 'btc_usd' | 'eth_usd'
        ("exchange_ts_ms", _I),
        ("price", _F),
    ]),

    # ------------------------------------------------------------------
    # Other spot venues (BRTI constituents + Binance.US)
    # ------------------------------------------------------------------
    "kraken_trades": pa.schema([
        ("curr_time", _S),
        ("pair", _S),
        ("exchange_time", _S),
        ("price", _F),
        ("size", _F),
        ("side", _S),
        ("order_type", _S),
    ]),
    "bitstamp_trades": pa.schema([
        ("curr_time", _S),
        ("pair", _S),
        ("trade_id", _I),
        ("exchange_ts", _I),
        ("price", _F),
        ("size", _F),
        ("side", _S),
    ]),
    "binanceus_trades": pa.schema([
        ("curr_time", _S),
        ("symbol", _S),
        ("trade_id", _I),
        ("exchange_ts_ms", _I),
        ("price", _F),
        ("size", _F),
        ("is_buyer_maker", _S),
    ]),
    "binanceus_quotes": pa.schema([
        ("curr_time", _S),
        ("symbol", _S),
        ("update_id", _I),
        ("best_bid", _F),
        ("best_bid_size", _F),
        ("best_ask", _F),
        ("best_ask_size", _F),
    ]),

    # ------------------------------------------------------------------
    # Equities / macro
    # ------------------------------------------------------------------
    "alpaca_trades": pa.schema([
        ("curr_time", _S),
        ("symbol", _S),
        ("exchange_time", _S),
        ("price", _F),
        ("size", _F),
        ("exchange", _S),
    ]),
    "alpaca_quotes": pa.schema([
        ("curr_time", _S),
        ("symbol", _S),
        ("exchange_time", _S),
        ("bid_price", _F),
        ("bid_size", _F),
        ("ask_price", _F),
        ("ask_size", _F),
    ]),
    "vix_quotes": pa.schema([
        ("curr_time", _S),
        ("symbol", _S),
        ("price", _F),
        ("quote_time", _S),
    ]),
}

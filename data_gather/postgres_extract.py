from sqlalchemy import create_engine
# from sqlalchemy.pool import NullPool
from dotenv import load_dotenv
import os
import pandas as pd
from pathlib import Path

# Load environment variables from .env
load_dotenv()

raw_data_dir = Path(__file__).resolve().parent / 'raw_data'
print(raw_data_dir)
# Fetch variables
PASSWORD = os.getenv("PASSWORD")
HOST = os.getenv("HOST")
PORT = os.getenv("PORT")
DBNAME = os.getenv("DATABASE")

# Construct the SQLAlchemy connection string
DATABASE_URL = f"postgresql+psycopg2://postgres:{PASSWORD}@{HOST}:{PORT}/{DBNAME}?sslmode=require"

# Create the SQLAlchemy engine
engine = create_engine(DATABASE_URL, connect_args={"options": "-c statement_timeout=30000000"} )  


# Test the connection
try:
    with engine.connect() as connection:
        print("Connection successful!")
except Exception as e:
    print(f"Failed to connect: {e}")

coinbase_df = pd.read_sql('SELECT * FROM coinbase_trades', engine)
coinbase_df['id'] = coinbase_df.index
coinbase_df.to_csv(raw_data_dir /'coinbase_trades.csv', index=False)

kalshi_df = pd.read_sql('SELECT * FROM kalshi_markets', engine)
kalshi_df['id'] = kalshi_df.index
kalshi_df.to_csv(raw_data_dir /'kalshi_markets.csv', index=False)

polymarket_df = pd.read_sql('SELECT * FROM polymarket_markets', engine)
polymarket_df['id'] = polymarket_df.index
polymarket_df.to_csv(raw_data_dir /'polymarket_markets.csv', index=False)

options_df = pd.read_sql('SELECT * FROM deribit_option_vols', engine)
options_df['id'] = options_df.index
options_df.to_csv(raw_data_dir /'deribit_option_vols.csv', index=False)

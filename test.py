import asyncio
import os
import sys

# Add the project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from crypto_pred_market.config import create_data_sink
from crypto_pred_market.data_gather.coinbase_ws import stream_coinbase_trades
from crypto_pred_market.data_gather.deribit_vol import scrape_deribit_vol
from crypto_pred_market.data_gather.kalshi import scrape_kalshi
from crypto_pred_market.data_gather.polymarket import scrape_polymarket

async def main():
    coins = ['BTC', 'ETH', 'XRP', 'SOL']
    test_data_sink = create_data_sink('test')

    # Clear CSV files from previous runs
    csv_files = [
        'coinbase_test.csv',
        'kalshi_test.csv', 
        'polymarket_test.csv',
        'deribit_option_vols_test.csv',
    ]
    
    for csv_file in csv_files:
        try:
            if os.path.exists(csv_file):
                os.remove(csv_file)
                print(f'Deleted {csv_file}')
        except Exception as e:
            print(f'Failed to delete {csv_file}: {e}')

    # Create tasks for each scraper
    tasks = [
        asyncio.create_task(stream_coinbase_trades(data_sink=test_data_sink)),
        asyncio.create_task(scrape_deribit_vol(data_sink=test_data_sink)),
        asyncio.create_task(scrape_kalshi(coins, data_sink=test_data_sink)),
        asyncio.create_task(scrape_polymarket(coins, data_sink=test_data_sink)),
    ]

    try:
        # Run for 600000 seconds
        await asyncio.wait_for(asyncio.gather(*tasks, return_exceptions=True), timeout=600000)
    except asyncio.TimeoutError:
        print("Test completed after 600000 seconds")
    except Exception as e:
        print(f"Test failed: {e}")
    finally:
        # Cancel all tasks
        for task in tasks:
            if not task.done():
                task.cancel()
                try:
                    await task
                except asyncio.CancelledError:
                    pass

    print("Test finished. Check the CSV files for data.")

if __name__ == '__main__':
    asyncio.run(main())

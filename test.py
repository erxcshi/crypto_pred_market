import asyncio
from pathlib import Path

from config import create_data_sink
from data_gather.scrapers.coinbase_ws import stream_coinbase_trades
from data_gather.scrapers.deribit_vol import scrape_deribit_vol
from data_gather.scrapers.kalshi import scrape_kalshi
from data_gather.scrapers.polymarket import scrape_polymarket

PROJECT_ROOT = Path(__file__).resolve().parent

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
        csv_path = PROJECT_ROOT / csv_file
        try:
            if csv_path.exists():
                csv_path.unlink()
                print(f'Deleted {csv_path}')
        except Exception as e:
            print(f'Failed to delete {csv_path}: {e}')

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

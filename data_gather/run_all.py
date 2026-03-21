import asyncio
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from crypto_pred_market.config import create_data_sink
from crypto_pred_market.data_gather.coinbase_ws import stream_coinbase_trades
from crypto_pred_market.data_gather.deribit_vol import scrape_deribit_vol
from crypto_pred_market.data_gather.kalshi import scrape_kalshi
from crypto_pred_market.data_gather.polymarket import scrape_polymarket


async def main():
    coins = ['BTC', 'ETH', 'XRP', 'SOL']
    production_data_sink = create_data_sink('actual')

    tasks = [
        asyncio.create_task(stream_coinbase_trades(data_sink=production_data_sink)),
        asyncio.create_task(scrape_deribit_vol(data_sink=production_data_sink)),
        asyncio.create_task(scrape_kalshi(coins, data_sink=production_data_sink)),
        asyncio.create_task(scrape_polymarket(coins, data_sink=production_data_sink)),
    ]

    try:
        await asyncio.gather(*tasks)
    except asyncio.CancelledError:
        raise
    except Exception:
        for task in tasks:
            task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)
        raise


if __name__ == '__main__':
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print('stopped by user')

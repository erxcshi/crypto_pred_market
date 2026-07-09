"""
Supervisor: runs every scraper module as its own child process, restarting any
child that dies (with exponential backoff) so one crashing scraper never takes
down the rest. Under a service manager (NSSM / systemd) this process itself is
also restarted, giving two layers of resilience.

Usage:
    DATA_SINK_MODE=parquet python -m data_gather.scrapers.run_all

Each child gets SCRAPER_NAME in its env, which names its heartbeat file
data/_status/{SCRAPER_NAME}.json (see data_gather/parquet_sink.py + monitor.py).
"""
import asyncio
import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
MODULES = [
    # Kalshi (traded venue): WS book/trades/ticker + REST 1s poll + settlements
    'data_gather.scrapers.kalshi_ws',
    'data_gather.scrapers.kalshi',
    'data_gather.scrapers.kalshi_settlements',
    # Polymarket: WS book events (polymarket.py REST poll retired by polymarket_ws)
    'data_gather.scrapers.polymarket_ws',
    # Spot venues (BRTI constituents + Binance.US)
    'data_gather.scrapers.coinbase_ws',
    'data_gather.scrapers.coinbase_l2',
    'data_gather.scrapers.kraken_ws',
    'data_gather.scrapers.bitstamp_ws',
    'data_gather.scrapers.binanceus_ws',
    # Vol / derivatives (deribit_vol REST poll retired by deribit_ws)
    'data_gather.scrapers.deribit_ws',
    # Equities + VIX (Alpaca keys optional)
    'data_gather.scrapers.alpaca_ws',
    # Heartbeat watcher
    'data_gather.monitor',
]

RESTART_BACKOFF_INITIAL = 5.0
RESTART_BACKOFF_MAX = 300.0
# A child that survives this long gets its backoff reset.
STABLE_RUN_SECONDS = 120.0


async def stream_output(prefix, stream):
    while True:
        line = await stream.readline()
        if not line:
            break
        print(f'[{prefix}] {line.decode(errors="replace").rstrip()}')


async def run_script_once(module_name: str) -> int:
    scraper_name = module_name.rsplit('.', 1)[-1]
    env = {**os.environ, 'SCRAPER_NAME': scraper_name,
           'PYTHONIOENCODING': 'utf-8', 'PYTHONUNBUFFERED': '1'}

    process = await asyncio.create_subprocess_exec(
        sys.executable,
        '-m',
        module_name,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
        cwd=PROJECT_ROOT,
        env=env,
    )

    stdout_task = asyncio.create_task(stream_output(scraper_name, process.stdout))
    stderr_task = asyncio.create_task(stream_output(f'{scraper_name}:stderr', process.stderr))

    try:
        returncode = await process.wait()
        await stdout_task
        await stderr_task
        return returncode
    except asyncio.CancelledError:
        process.terminate()
        await process.wait()
        await stdout_task
        await stderr_task
        raise


async def supervise(module_name: str) -> None:
    """Run a module forever, restarting on exit with exponential backoff."""
    backoff = RESTART_BACKOFF_INITIAL
    loop = asyncio.get_event_loop()
    while True:
        started = loop.time()
        try:
            returncode = await run_script_once(module_name)
        except asyncio.CancelledError:
            raise
        except Exception as e:  # spawn failure etc.
            print(f'[supervisor] {module_name} failed to start: {type(e).__name__}: {e}')
            returncode = -1

        ran_for = loop.time() - started
        if ran_for >= STABLE_RUN_SECONDS:
            backoff = RESTART_BACKOFF_INITIAL

        print(f'[supervisor] {module_name} exited code={returncode} after {ran_for:.0f}s '
              f'— restarting in {backoff:.0f}s')
        await asyncio.sleep(backoff)
        backoff = min(RESTART_BACKOFF_MAX, backoff * 2)


async def main():
    print(f'[supervisor] starting {len(MODULES)} modules '
          f'(DATA_SINK_MODE={os.getenv("DATA_SINK_MODE", "actual")!r})')
    tasks = [asyncio.create_task(supervise(m)) for m in MODULES]
    try:
        await asyncio.gather(*tasks)
    except asyncio.CancelledError:
        for task in tasks:
            task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)
        raise


if __name__ == '__main__':
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print('stopped by user')

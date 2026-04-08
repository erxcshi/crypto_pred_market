import asyncio
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
MODULES = [
    'data_gather.scrapers.coinbase_ws',
    'data_gather.scrapers.deribit_vol',
    'data_gather.scrapers.kalshi',
    'data_gather.scrapers.polymarket',
]


async def stream_output(prefix, stream):
    while True:
        line = await stream.readline()
        if not line:
            break
        print(f'[{prefix}] {line.decode().rstrip()}')


async def run_script(module_name):
    process = await asyncio.create_subprocess_exec(
        sys.executable,
        '-m',
        module_name,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
        cwd=PROJECT_ROOT,
    )

    stdout_task = asyncio.create_task(stream_output(module_name, process.stdout))
    stderr_task = asyncio.create_task(stream_output(f'{module_name}:stderr', process.stderr))

    try:
        returncode = await process.wait()
        await stdout_task
        await stderr_task

        if returncode != 0:
            raise RuntimeError(f'{module_name} exited with code {returncode}')
    except asyncio.CancelledError:
        process.terminate()
        await process.wait()
        await stdout_task
        await stderr_task
        raise


async def main():
    tasks = [asyncio.create_task(run_script(module_name)) for module_name in MODULES]

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

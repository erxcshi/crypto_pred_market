import asyncio
import os
import sys

PROJECT_PARENT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
MODULES = [
    'crypto_pred_market.data_gather.coinbase_ws',
    'crypto_pred_market.data_gather.deribit_vol',
    'crypto_pred_market.data_gather.kalshi',
    'crypto_pred_market.data_gather.polymarket',
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
        cwd=PROJECT_PARENT,
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

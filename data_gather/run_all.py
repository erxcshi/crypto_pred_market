import asyncio
import sys
from pathlib import Path


BASE_DIR = Path(__file__).resolve().parent
SCRIPTS = [
    'coinbase_ws.py',
    'deribit_vol.py',
    'kalshi.py',
    'polymarket.py',
]


async def stream_output(prefix, stream):
    while True:
        line = await stream.readline()
        if not line:
            break
        print(f'[{prefix}] {line.decode().rstrip()}')


async def run_script(script_name):
    script_path = BASE_DIR / script_name
    process = await asyncio.create_subprocess_exec(
        sys.executable,
        str(script_path),
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
        cwd=str(BASE_DIR),
    )

    stdout_task = asyncio.create_task(stream_output(script_name, process.stdout))
    stderr_task = asyncio.create_task(stream_output(f'{script_name}:stderr', process.stderr))

    try:
        returncode = await process.wait()
        await stdout_task
        await stderr_task

        if returncode != 0:
            raise RuntimeError(f'{script_name} exited with code {returncode}')
    except asyncio.CancelledError:
        process.terminate()
        await process.wait()
        await stdout_task
        await stderr_task
        raise


async def main():
    tasks = [asyncio.create_task(run_script(script_name)) for script_name in SCRIPTS]

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

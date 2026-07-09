# Running the collector 24/7 on Windows

## 0. One-time machine prep (PowerShell as admin)

```powershell
# Never sleep on AC power (sleep = data gaps)
powercfg /change standby-timeout-ac 0
powercfg /change hibernate-timeout-ac 0

# Exclude the data dir from Defender real-time scanning (write-latency)
Add-MpPreference -ExclusionPath "C:\Users\xxeri\crypto_pred_market\data"
```

Also disable "Fast Startup" if the box reboots for Windows Update often
(Control Panel → Power Options → Choose what the power buttons do).

## 1. Option A — NSSM service (recommended: runs without login, auto-restarts)

Download nssm from https://nssm.cc (single exe), then as admin:

```powershell
nssm install MarketCollector "C:\Users\xxeri\crypto_pred_market\.venv\Scripts\python.exe" "-m data_gather.scrapers.run_all"
nssm set MarketCollector AppDirectory "C:\Users\xxeri\crypto_pred_market"
nssm set MarketCollector AppEnvironmentExtra "DATA_SINK_MODE=parquet" "PYTHONIOENCODING=utf-8" "PYTHONUNBUFFERED=1"
nssm set MarketCollector AppStdout "C:\Users\xxeri\crypto_pred_market\collector.log"
nssm set MarketCollector AppStderr "C:\Users\xxeri\crypto_pred_market\collector.log"
nssm set MarketCollector AppRotateFiles 1
nssm set MarketCollector AppRotateBytes 50000000
nssm set MarketCollector AppExit Default Restart
nssm set MarketCollector AppThrottle 5000
nssm start MarketCollector
```

Check: `nssm status MarketCollector`, tail `collector.log`, and look at
`data\_status\*.json` timestamps.

## 2. Option B — Task Scheduler (no install; slightly weaker)

Task Scheduler → Create Task:
- Trigger: At startup
- Action: Start a program
  - Program: `C:\Users\xxeri\crypto_pred_market\.venv\Scripts\python.exe`
  - Arguments: `-m data_gather.scrapers.run_all`
  - Start in: `C:\Users\xxeri\crypto_pred_market`
- Settings: "If the task fails, restart every 1 minute", "Run whether user is
  logged on or not"
- Put `DATA_SINK_MODE=parquet` in the repo `.env` (config.py loads it) since
  Task Scheduler env vars are awkward.

## 3. Daily compaction (both options)

Task Scheduler → Create Task, daily 00:15 UTC (adjust for local tz):
- Program: `C:\Users\xxeri\crypto_pred_market\.venv\Scripts\python.exe`
- Arguments: `-m data_gather.compact`
- Start in: `C:\Users\xxeri\crypto_pred_market`

## 4. Off-box backup (optional but recommended once data matters)

Install rclone (https://rclone.org), configure a remote (Cloudflare R2 /
Backblaze B2 free tiers, or SFTP to another machine), then add to the daily
task after compaction:

```
rclone copy C:\Users\xxeri\crypto_pred_market\data remote:market-data --include "*/date=*/compacted-*.parquet"
```

Once backups run, add `--retention-days 60` to the compact command to bound
local disk.

## 5. Watching it

- `data\_status\*.json` — per-scraper heartbeats (written every 60s)
- The `data_gather.monitor` child prints `all streams healthy` / `MONITOR
  WARNING ...` lines into the collector log
- Optional: set `ALERT_WEBHOOK_URL` in `.env` to a Discord/Slack webhook to get
  pinged on staleness

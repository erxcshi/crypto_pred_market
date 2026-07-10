"""
Gap monitor: watches data/_status/*.json heartbeats written by ParquetDataSink
and warns when a stream goes silent or reports write errors.

Catches three failure classes with one mechanism:
  1. dead scraper process        -> its status file itself goes stale
  2. connected-but-silent WS     -> rows stop, last_row_written_at goes stale
  3. disk / write failures       -> last_error set in the status payload

Alerts: WARNING log line always (captured by NSSM/systemd/console); optional
POST to ALERT_WEBHOOK_URL (Discord/Slack-compatible {"content": ...} payload).
Re-alerts at most once per ALERT_COOLDOWN_SECONDS per stream; logs recovery.

Run standalone or as a module under run_all:
    python -m data_gather.monitor
"""
from __future__ import annotations

import asyncio
import json
import os
import time
from datetime import datetime, timezone
from pathlib import Path

import aiohttp

PROJECT_ROOT = Path(__file__).resolve().parents[1]
STATUS_DIR = Path(os.getenv("PARQUET_DATA_DIR") or (PROJECT_ROOT / "data")) / "_status"

CHECK_INTERVAL_SECONDS = 60
ALERT_COOLDOWN_SECONDS = 30 * 60

# Seconds without a written row before a TABLE is considered stale.
# Keyed by table name; falls back to DEFAULT_TABLE_THRESHOLD.
TABLE_THRESHOLDS: dict[str, int] = {
    # near-continuous streams
    "coinbase_trades": 600,    # trades are event-driven; overnight gaps of 3-5 min are normal
    "coinbase_quotes": 120,
    "coinbase_l2": 120,
    "binanceus_trades": 3600,   # thin venue — trades can be sparse
    "binanceus_quotes": 180,
    "kraken_trades": 600,
    "bitstamp_trades": 900,
    # 1s pollers
    "kalshi_markets": 120,
    "polymarket_markets": 300,
    # thin event streams — can legitimately be quiet for a while
    "kalshi_orderbook_deltas": 900,
    "kalshi_trades": 3600,
    "kalshi_ticker": 900,
    "polymarket_book_events": 900,
    # 5s poller / slow streams
    "deribit_option_vols": 300,
    "deribit_dvol": 300,
    "deribit_ticker": 300,
    "deribit_perp": 300,
    "deribit_index": 300,
    "kalshi_settlements": 2 * 3600,
}
DEFAULT_TABLE_THRESHOLD = 900

# Equity feeds only produce rows while US equities trade, so a flat threshold
# either false-alarms every night/weekend (overnight gap ~15h, weekend ~63h)
# or reacts uselessly slowly to real weekday outages. Instead, only check
# these tables during 14:30-20:00 UTC Mon-Fri — the intersection of EST and
# EDT regular sessions, so the market is open in either DST regime. Market
# holidays cost a few spurious warnings per year; accepted.
EQUITY_TABLES = {"alpaca_trades", "alpaca_quotes", "vix_quotes"}
EQUITY_SESSION_OPEN_UTC = (14, 30)
EQUITY_SESSION_CLOSE_UTC = (20, 0)
EQUITY_STALE_THRESHOLD = 1800
# Seconds without a heartbeat write before the whole STREAM (process) is stale.
STREAM_HEARTBEAT_THRESHOLD = 300


def _parse_iso(ts: str) -> datetime | None:
    try:
        return datetime.fromisoformat(ts)
    except (TypeError, ValueError):
        return None


def _equity_session_open(now: datetime) -> datetime | None:
    """Start of today's session if US equities are trading at `now`, else None."""
    if now.weekday() >= 5:
        return None
    open_dt = now.replace(hour=EQUITY_SESSION_OPEN_UTC[0], minute=EQUITY_SESSION_OPEN_UTC[1],
                          second=0, microsecond=0)
    close_dt = now.replace(hour=EQUITY_SESSION_CLOSE_UTC[0], minute=EQUITY_SESSION_CLOSE_UTC[1],
                           second=0, microsecond=0)
    return open_dt if open_dt <= now <= close_dt else None


def check_status(status_dir: Path, now: datetime) -> list[tuple[str, str]]:
    """Returns [(key, message)] for everything currently unhealthy.

    key is STABLE across sweeps (no ages/timestamps in it) — it's what dedup,
    cooldown, and recovery detection are keyed on. message carries the
    human-readable details.
    """
    alerts: list[tuple[str, str]] = []
    files = sorted(status_dir.glob("*.json")) if status_dir.exists() else []
    if not files:
        return [("no-status-files", f"no status files in {status_dir} — is any collector running?")]

    for f in files:
        try:
            payload = json.loads(f.read_text())
        except Exception as e:  # noqa: BLE001
            alerts.append((f"{f.name}:unreadable", f"{f.name}: unreadable status file ({type(e).__name__})"))
            continue

        stream = payload.get("stream_name", f.stem)
        written_at = _parse_iso(payload.get("written_at", ""))
        if written_at is None or (now - written_at).total_seconds() > STREAM_HEARTBEAT_THRESHOLD:
            age = "unknown" if written_at is None else f"{(now - written_at).total_seconds():.0f}s"
            alerts.append((f"{stream}:heartbeat",
                           f"{stream}: heartbeat stale ({age} old) — process dead or wedged?"))
            continue  # process-level alert supersedes table-level ones

        if payload.get("last_error"):
            alerts.append((f"{stream}:write-error", f"{stream}: write error: {payload['last_error']}"))

        for table, stats in (payload.get("tables") or {}).items():
            last_row = _parse_iso(stats.get("last_row_written_at", ""))
            if last_row is None:
                continue
            if table in EQUITY_TABLES:
                session_open = _equity_session_open(now)
                if session_open is None:
                    continue  # market closed — silence is expected
                # Measure from session open so the first sweeps after open
                # don't fire on the (normal) overnight gap.
                threshold = EQUITY_STALE_THRESHOLD
                silence = (now - max(last_row, session_open)).total_seconds()
            else:
                threshold = TABLE_THRESHOLDS.get(table, DEFAULT_TABLE_THRESHOLD)
                silence = (now - last_row).total_seconds()
            if silence > threshold:
                alerts.append((f"{stream}/{table}:stale",
                               f"{stream}/{table}: no rows for {silence:.0f}s (threshold {threshold}s)"))
    return alerts


async def send_alert(session: aiohttp.ClientSession | None, message: str) -> None:
    print(f"MONITOR WARNING: {message}")
    webhook = os.getenv("ALERT_WEBHOOK_URL", "")
    if webhook and session is not None:
        try:
            await session.post(
                webhook,
                json={"content": f"[collector] {message}"},
                timeout=aiohttp.ClientTimeout(total=10),
            )
        except Exception as e:  # noqa: BLE001
            print(f"monitor: webhook post failed: {type(e).__name__}: {e}")


async def watch(status_dir: Path = STATUS_DIR, interval: int = CHECK_INTERVAL_SECONDS) -> None:
    last_alert_time: dict[str, float] = {}
    active: set[str] = set()
    print(f"monitor: watching {status_dir} every {interval}s "
          f"(webhook={'set' if os.getenv('ALERT_WEBHOOK_URL') else 'unset'})")

    async with aiohttp.ClientSession() as session:
        while True:
            now = datetime.now(timezone.utc)
            alerts = check_status(status_dir, now)
            current_keys = {key for key, _ in alerts}

            # Recovery = a stable key that was alerting and no longer is.
            for recovered_key in active - current_keys:
                print(f"MONITOR RECOVERED: {recovered_key}")
            active = current_keys

            mono = time.monotonic()
            for key, message in alerts:
                if mono - last_alert_time.get(key, -1e12) >= ALERT_COOLDOWN_SECONDS:
                    last_alert_time[key] = mono
                    await send_alert(session, message)

            if not alerts:
                print(f"monitor: all streams healthy ({now.strftime('%H:%M:%S')} UTC)")
            await asyncio.sleep(interval)


if __name__ == "__main__":
    try:
        asyncio.run(watch())
    except KeyboardInterrupt:
        print("monitor stopped")

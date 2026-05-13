#!/usr/bin/env python3
"""
Derivatives data collector — snapshots OI, L/S, taker, funding to CSV.
Appends one row per run. Designed to be called from scanner.py or standalone.

Usage:
    python3 scripts/collect_derivatives.py              # single snapshot
    python3 scripts/collect_derivatives.py --loop 900    # loop every 900s (15m)
"""

import sys
import os
import time
import argparse
from datetime import datetime

import pandas as pd
import requests

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

BASE_URL = "https://fapi.binance.com"
DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "derivatives_history")
os.makedirs(DATA_DIR, exist_ok=True)

CSV_PATH = os.path.join(DATA_DIR, "derivatives_collected.csv")
LOCKFILE = os.path.join(DATA_DIR, ".collect_lock")


def fetch(endpoint, symbol="ETHUSDT", period="15m", limit=1):
    """Fetch one endpoint, return latest record."""
    params = {"symbol": symbol, "period": period, "limit": limit}
    r = requests.get(f"{BASE_URL}{endpoint}", params=params, timeout=10)
    r.raise_for_status()
    return r.json()


def snapshot(symbol="ETHUSDT"):
    """Collect one derivatives snapshot and return as dict."""
    ts = datetime.utcnow()
    row = {"timestamp": ts.isoformat(), "timestamp_ms": int(ts.timestamp() * 1000)}

    # OI
    try:
        data = fetch("/futures/data/openInterestHist", symbol=symbol, limit=1)
        if data:
            d = data[-1]
            row["oi"] = float(d["sumOpenInterest"])
            row["oi_usd"] = float(d["sumOpenInterestValue"])
    except Exception as e:
        print(f"  ⚠️  OI fetch failed: {e}")

    # Global L/S
    try:
        data = fetch("/futures/data/globalLongShortAccountRatio", symbol=symbol, limit=1)
        if data:
            d = data[-1]
            row["ls_ratio"] = float(d["longShortRatio"])
            row["long_pct"] = float(d["longAccount"])
            row["short_pct"] = float(d["shortAccount"])
    except Exception as e:
        print(f"  ⚠️  L/S fetch failed: {e}")

    # Top trader L/S
    try:
        data = fetch("/futures/data/topLongShortAccountRatio", symbol=symbol, limit=1)
        if data:
            d = data[-1]
            row["top_ls_ratio"] = float(d["longShortRatio"])
            row["top_long_pct"] = float(d["longAccount"])
            row["top_short_pct"] = float(d["shortAccount"])
    except Exception as e:
        print(f"  ⚠️  Top L/S fetch failed: {e}")

    # Taker
    try:
        data = fetch("/futures/data/takerlongshortRatio", symbol=symbol, limit=1)
        if data:
            d = data[-1]
            row["futures_taker_ratio"] = float(d["buySellRatio"])
            row["futures_buy_vol"] = float(d["buyVol"])
            row["futures_sell_vol"] = float(d["sellVol"])
    except Exception as e:
        print(f"  ⚠️  Taker fetch failed: {e}")

    # Funding
    try:
        r = requests.get(f"{BASE_URL}/fapi/v1/fundingRate",
                         params={"symbol": symbol, "limit": 1}, timeout=10)
        r.raise_for_status()
        data = r.json()
        if data:
            d = data[-1]
            row["funding_rate"] = float(d["fundingRate"])
    except Exception as e:
        print(f"  ⚠️  Funding fetch failed: {e}")

    return row


def append(row):
    """Append row to CSV, creating it if needed."""
    df_new = pd.DataFrame([row])
    if os.path.exists(CSV_PATH):
        df = pd.read_csv(CSV_PATH)
        df = pd.concat([df, df_new], ignore_index=True)
    else:
        df = df_new
    df.to_csv(CSV_PATH, index=False)
    return len(df)


def collect():
    """Main collect routine — fetch + append + print."""
    row = snapshot()
    n = append(row)

    oi_usd = row.get("oi_usd", 0)
    ls = row.get("ls_ratio", 0)
    taker = row.get("futures_taker_ratio", 0)
    fr = row.get("funding_rate", 0)

    print(f"  📊 Derivatives snapshot #{n}: "
          f"OI=${oi_usd/1e9:.2f}B  L/S={ls:.4f}  taker={taker:.4f}  fr={fr:.6f}")
    return row


def mark_ready():
    """Touch the lockfile to signal cron that scanner has run at least once."""
    if not os.path.exists(LOCKFILE):
        with open(LOCKFILE, "w") as f:
            f.write(datetime.utcnow().isoformat())
        print(f"  ✅ Cron guard activated: {LOCKFILE}")


def is_ready():
    """Check if scanner.py has run at least once."""
    return os.path.exists(LOCKFILE)


def main():
    parser = argparse.ArgumentParser(description="Collect derivatives snapshots")
    parser.add_argument("--loop", type=int, help="Loop interval in seconds (e.g. 900 for 15m)")
    parser.add_argument("--symbol", default="ETHUSDT", help="Trading pair")
    args = parser.parse_args()

    if args.loop:
        print(f"  🔄 Collecting every {args.loop}s (Ctrl+C to stop)")
        while True:
            try:
                collect()
            except Exception as e:
                print(f"  ❌ Error: {e}")
            time.sleep(args.loop)
    else:
        collect()


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
Backfill derivatives history — fetches backward in time, chunk by chunk.
Each chunk: 500 bars of derivatives (~5 days of 15m data).
Appends to existing CSV, runs analysis, pushes to git.

Usage:
    python3 scripts/backfill_derivatives.py 5  # fetch 5 chunks
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import subprocess
import pandas as pd
import numpy as np
import requests
import time

BASE_URL = "https://fapi.binance.com"
DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data', 'derivatives_history')
os.makedirs(DATA_DIR, exist_ok=True)


def fetch_chunk(end_time_ms=None, symbol="ETHUSDT", period="15m", limit=500):
    """Fetch one chunk of all 4 derivatives endpoints."""
    def _fetch(endpoint, extra_params=None):
        params = {"symbol": symbol, "period": period, "limit": limit}
        if end_time_ms:
            params["endTime"] = end_time_ms
        if extra_params:
            params.update(extra_params)
        r = requests.get(f"{BASE_URL}{endpoint}", params=params)
        r.raise_for_status()
        return r.json()

    try:
        oi_raw = _fetch("/futures/data/openInterestHist")
        ls_raw = _fetch("/futures/data/globalLongShortAccountRatio")
        top_raw = _fetch("/futures/data/topLongShortAccountRatio")
        taker_raw = _fetch("/futures/data/takerlongshortRatio")
    except requests.exceptions.HTTPError as e:
        if e.response.status_code == 400:
            print(f"  ⚠️  API 400 — reached data limit (endTime={end_time_ms})")
            return None
        raise

    if not oi_raw:
        print(f"  ⚠️  Empty response — no more data")
        return None

    # Parse and merge
    rows = []
    oi_map = {}
    for d in oi_raw:
        ts = d["timestamp"]
        oi_map[ts] = {"oi": float(d["sumOpenInterest"]), "oi_usd": float(d["sumOpenInterestValue"])}

    ls_map = {}
    for d in ls_raw:
        ts = d["timestamp"]
        ls_map[ts] = {"ls_ratio": float(d["longShortRatio"]), "long_pct": float(d["longAccount"]), "short_pct": float(d["shortAccount"])}

    top_map = {}
    for d in top_raw:
        ts = d["timestamp"]
        top_map[ts] = {"top_ls_ratio": float(d["longShortRatio"]), "top_long_pct": float(d["longAccount"]), "top_short_pct": float(d["shortAccount"])}

    taker_map = {}
    for d in taker_raw:
        ts = d["timestamp"]
        taker_map[ts] = {"futures_taker_ratio": float(d["buySellRatio"]), "futures_buy_vol": float(d["buyVol"]), "futures_sell_vol": float(d["sellVol"])}

    all_ts = sorted(set(list(oi_map.keys()) + list(ls_map.keys()) + list(top_map.keys()) + list(taker_map.keys())))

    for ts in all_ts:
        row = {"timestamp_ms": int(ts), "timestamp": pd.to_datetime(ts, unit="ms")}
        row.update(oi_map.get(ts, {}))
        row.update(ls_map.get(ts, {}))
        row.update(top_map.get(ts, {}))
        row.update(taker_map.get(ts, {}))
        rows.append(row)

    df = pd.DataFrame(rows).sort_values("timestamp_ms").reset_index(drop=True)
    df = df.ffill()
    return df


def load_existing():
    """Load existing derivatives_raw.csv."""
    path = os.path.join(DATA_DIR, "derivatives_raw.csv")
    if os.path.exists(path):
        df = pd.read_csv(path)
        df["timestamp"] = pd.to_datetime(df["timestamp"])
        return df
    return pd.DataFrame()


def save(df):
    """Save to CSV."""
    path = os.path.join(DATA_DIR, "derivatives_raw.csv")
    df.to_csv(path, index=False)
    print(f"  💾 Total: {len(df)} bars saved")


def run_analysis():
    """Run the combined analysis on all saved data."""
    # Import and run analysis from the backtest module
    from scripts.whale_structure_backtest import run_full_analysis
    return run_full_analysis()


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("chunks", type=int, help="Number of chunks to fetch")
    args = parser.parse_args()

    existing = load_existing()
    if len(existing) > 0:
        print(f"  Existing data: {len(existing)} bars ({existing['timestamp'].min()} → {existing['timestamp'].max()})")
        end_time_ms = int(existing["timestamp_ms"].min()) - 1
    else:
        end_time_ms = None  # latest

    all_stats = []

    for i in range(1, args.chunks + 1):
        print(f"\n{'═' * 60}")
        print(f"  CHUNK {i}/{args.chunks}")
        print(f"{'═' * 60}")

        if end_time_ms:
            end_dt = pd.to_datetime(end_time_ms, unit="ms")
            print(f"  Fetching before: {end_dt}")

        df = fetch_chunk(end_time_ms=end_time_ms)
        if df is None:
            print("  Stopping — no more data available.")
            break

        print(f"  Fetched: {len(df)} bars ({df['timestamp'].min()} → {df['timestamp'].max()})")

        # Merge with existing (deduplicate)
        if len(existing) > 0:
            combined = pd.concat([existing, df]).drop_duplicates(subset=["timestamp_ms"]).sort_values("timestamp_ms").reset_index(drop=True)
        else:
            combined = df.copy()

        save(combined)

        # Update pagination cursor
        end_time_ms = int(df["timestamp_ms"].min()) - 1
        existing = combined

        # Rate limit
        time.sleep(1)

    # Final combined analysis
    print(f"\n{'═' * 60}")
    print(f"  RUNNING COMBINED ANALYSIS")
    print(f"{'═' * 60}")
    print(f"  Total derivatives bars: {len(existing)}")

    # Compute signals on full dataset
    from scripts.whale_structure_backtest import (
        compute_positioning_signals, compute_structure_bias,
        compute_forward_returns, analyze_results, save_analysis
    )

    df_deriv = compute_positioning_signals(existing)

    # Fetch full price data for the derivatives range
    import ccxt
    ex = ccxt.binance({"enableRateLimit": True})
    ts_min = int(existing["timestamp_ms"].min())
    ts_max = int(existing["timestamp_ms"].max())
    fetch_since = ts_min - (2000 * 15 * 60 * 1000)
    price_end = ts_max + (96 * 15 * 60 * 1000)

    print(f"  Fetching price data for full range...")
    price_rows = []
    cursor = fetch_since
    while cursor < price_end:
        try:
            ohlcv = ex.fetch_ohlcv("ETH/USDT", "15m", since=cursor, limit=1000)
            if not ohlcv:
                break
            price_rows.extend(ohlcv)
            cursor = ohlcv[-1][0] + 1
            if len(ohlcv) < 1000:
                break
        except Exception as e:
            print(f"  ⚠️  Price fetch error: {e}")
            break

    df_price = pd.DataFrame(price_rows, columns=["Open time ms", "Open", "High", "Low", "Close", "Volume"])
    df_price["Open time"] = pd.to_datetime(df_price["Open time ms"], unit="ms")
    df_price = df_price.drop_duplicates(subset=["Open time ms"]).sort_values("Open time ms").reset_index(drop=True)
    print(f"  Price bars: {len(df_price)}")

    # Align and compute
    df_price["ts"] = pd.to_datetime(df_price["Open time"])
    df_deriv["ts"] = pd.to_datetime(df_deriv["timestamp"])
    df_price = df_price.set_index("ts")
    df_deriv = df_deriv.set_index("ts").resample("15min").last().dropna(how="all")
    df_deriv = df_deriv.reindex(df_price.index, method="ffill")

    structure_bias = compute_structure_bias(df_price.reset_index())
    fwd_returns = compute_forward_returns(df_price.reset_index())

    results = []
    min_idx = max(55, 48)
    for idx in range(min_idx, len(df_price) - 96):
        ws = df_deriv["whale_signal"].iloc[idx] if idx < len(df_deriv) and "whale_signal" in df_deriv.columns else "NEUTRAL"
        sb = structure_bias.iloc[idx] if idx < len(structure_bias) else "NEUTRAL"

        if not isinstance(ws, str) or ws == "nan" or (isinstance(ws, float) and np.isnan(ws)):
            ws = "NEUTRAL"
        if not isinstance(sb, str) or sb == "nan" or (isinstance(sb, float) and np.isnan(sb)):
            sb = "NEUTRAL"

        if ws == "NEUTRAL" or sb == "NEUTRAL":
            continue

        row = {
            "idx": idx,
            "timestamp": str(df_price.index[idx]),
            "price": float(df_price["Close"].iloc[idx]),
            "whale": ws,
            "structure": sb,
            "aligned": ws.replace("WHALE_", "") == sb,
        }
        for col in fwd_returns.columns:
            val = fwd_returns[col].iloc[idx] if idx < len(fwd_returns) else None
            row[col] = float(val) if val is not None and not pd.isna(val) else None
        results.append(row)

    df_results = pd.DataFrame(results)
    save_analysis(df_results, 999)  # combined analysis

    stats = analyze_results(df_results, "COMBINED ALL CHUNKS")

    # Push to git
    print(f"\n  Pushing to git...")
    os.chdir(os.path.dirname(os.path.dirname(__file__)))
    subprocess.run(["git", "add", "-f", "scripts/backfill_derivatives.py", "data/derivatives_history/"], check=True)
    n_bars = len(existing)
    date_range = f"{existing['timestamp'].min().strftime('%Y-%m-%d')} to {existing['timestamp'].max().strftime('%Y-%m-%d')}"
    subprocess.run(["git", "commit", "-m", f"Backfill derivatives: {n_bars} bars ({date_range})"], check=True)
    subprocess.run(["git", "push", "origin", "master"], check=True)
    print(f"  ✅ Pushed!")


if __name__ == "__main__":
    main()

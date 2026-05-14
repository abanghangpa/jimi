#!/usr/bin/env python3
"""
Whale vs Structure — Historical Backtest with Pagination

Fetches derivatives data from Binance in 500-bar chunks, paginating
backward through time. Saves each chunk to CSV for cumulative analysis.

Usage:
    python3 scripts/whale_structure_backtest.py                  # fetch latest 500 bars
    python3 scripts/whale_structure_backtest.py --paginate 10    # fetch 10 chunks (5000 bars)
    python3 scripts/whale_structure_backtest.py --analyze        # analyze all saved data
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import argparse
import json
import numpy as np
import pandas as pd
import requests
from datetime import datetime, timedelta

from src.utils.data_handler import fetch_recent, resample_ohlcv
from src.utils.indicators import calc_ema

BASE_URL = "https://fapi.binance.com"
DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data', 'derivatives_history')
os.makedirs(DATA_DIR, exist_ok=True)

CHUNK_SIZE = 500  # max per Binance API call


# ═══════════════════════════════════════════════════════════════
# DERIVATIVES FETCHERS (with endTime pagination)
# ═══════════════════════════════════════════════════════════════

def fetch_oi_chunk(symbol="ETHUSDT", period="15m", limit=500, end_time_ms=None):
    params = {"symbol": symbol, "period": period, "limit": limit}
    if end_time_ms:
        params["endTime"] = end_time_ms
    r = requests.get(f"{BASE_URL}/futures/data/openInterestHist", params=params)
    r.raise_for_status()
    rows = []
    for d in r.json():
        rows.append({
            "timestamp": pd.to_datetime(d["timestamp"], unit="ms"),
            "timestamp_ms": int(d["timestamp"]),
            "oi": float(d["sumOpenInterest"]),
            "oi_usd": float(d["sumOpenInterestValue"]),
        })
    return pd.DataFrame(rows)


def fetch_ls_chunk(symbol="ETHUSDT", period="15m", limit=500, end_time_ms=None):
    params = {"symbol": symbol, "period": period, "limit": limit}
    if end_time_ms:
        params["endTime"] = end_time_ms
    r = requests.get(f"{BASE_URL}/futures/data/globalLongShortAccountRatio", params=params)
    r.raise_for_status()
    rows = []
    for d in r.json():
        rows.append({
            "timestamp": pd.to_datetime(d["timestamp"], unit="ms"),
            "timestamp_ms": int(d["timestamp"]),
            "ls_ratio": float(d["longShortRatio"]),
            "long_pct": float(d["longAccount"]),
            "short_pct": float(d["shortAccount"]),
        })
    return pd.DataFrame(rows)


def fetch_top_trader_chunk(symbol="ETHUSDT", period="15m", limit=500, end_time_ms=None):
    params = {"symbol": symbol, "period": period, "limit": limit}
    if end_time_ms:
        params["endTime"] = end_time_ms
    r = requests.get(f"{BASE_URL}/futures/data/topLongShortAccountRatio", params=params)
    r.raise_for_status()
    rows = []
    for d in r.json():
        rows.append({
            "timestamp": pd.to_datetime(d["timestamp"], unit="ms"),
            "timestamp_ms": int(d["timestamp"]),
            "top_ls_ratio": float(d["longShortRatio"]),
            "top_long_pct": float(d["longAccount"]),
            "top_short_pct": float(d["shortAccount"]),
        })
    return pd.DataFrame(rows)


def fetch_taker_chunk(symbol="ETHUSDT", period="15m", limit=500, end_time_ms=None):
    params = {"symbol": symbol, "period": period, "limit": limit}
    if end_time_ms:
        params["endTime"] = end_time_ms
    r = requests.get(f"{BASE_URL}/futures/data/takerlongshortRatio", params=params)
    r.raise_for_status()
    rows = []
    for d in r.json():
        rows.append({
            "timestamp": pd.to_datetime(d["timestamp"], unit="ms"),
            "timestamp_ms": int(d["timestamp"]),
            "futures_taker_ratio": float(d["buySellRatio"]),
            "futures_buy_vol": float(d["buyVol"]),
            "futures_sell_vol": float(d["sellVol"]),
        })
    return pd.DataFrame(rows)


def fetch_derivatives_chunk(end_time_ms=None, symbol="ETHUSDT"):
    """Fetch one chunk of all derivatives data, merged. Returns None on API error."""
    try:
        oi = fetch_oi_chunk(symbol, end_time_ms=end_time_ms)
        ls = fetch_ls_chunk(symbol, end_time_ms=end_time_ms)
        top = fetch_top_trader_chunk(symbol, end_time_ms=end_time_ms)
        taker = fetch_taker_chunk(symbol, end_time_ms=end_time_ms)
    except requests.exceptions.HTTPError as e:
        if e.response.status_code == 400:
            print(f"  ⚠️  Binance API 400 — data likely doesn't go back this far (endTime={end_time_ms})")
            return None
        raise

    if len(oi) == 0:
        print(f"  ⚠️  Empty response from Binance — reached data limit")
        return None

    df = oi.merge(ls, on=["timestamp", "timestamp_ms"], how="outer")
    df = df.merge(top, on=["timestamp", "timestamp_ms"], how="outer")
    df = df.merge(taker, on=["timestamp", "timestamp_ms"], how="outer")
    df = df.sort_values("timestamp").reset_index(drop=True)
    df = df.ffill()
    return df


# ═══════════════════════════════════════════════════════════════
# SIGNAL COMPUTATION
# ═══════════════════════════════════════════════════════════════

def compute_positioning_signals(df):
    """Compute whale signal, positioning, taker flow from raw derivatives data."""
    df = df.copy()

    # L/S z-score (3-day window + absolute bands + Δ1h velocity)
    if 'ls_ratio' in df.columns:
        window = min(288, len(df))
        min_periods = min(24, len(df))
        ls_mean = df['ls_ratio'].rolling(window, min_periods=min_periods).mean()
        ls_std = df['ls_ratio'].rolling(window, min_periods=min_periods).std().replace(0, np.nan)
        df['ls_zscore'] = (df['ls_ratio'] - ls_mean) / ls_std
        df['ls_zscore'] = df['ls_zscore'].fillna(0)

        df['positioning'] = 'NEUTRAL'
        # Z-score triggers
        df.loc[(df['ls_zscore'] > 1.5) & (df['ls_ratio'] > 1.5), 'positioning'] = 'CROWDED_LONG'
        df.loc[(df['ls_zscore'] < -1.5) & (df['ls_ratio'] < 0.67), 'positioning'] = 'CROWDED_SHORT'
        # Absolute bands
        df.loc[df['ls_ratio'] >= 3.0, 'positioning'] = 'CROWDED_LONG'
        df.loc[df['ls_ratio'] <= 0.33, 'positioning'] = 'CROWDED_SHORT'
        # Rate-of-change (Δ1h = 4 bars at 15m)
        ls_delta_1h = df['ls_ratio'].diff(4)
        df.loc[ls_delta_1h >= 0.15, 'positioning'] = 'CROWDED_LONG'
        df.loc[ls_delta_1h <= -0.15, 'positioning'] = 'CROWDED_SHORT'

    # Whale signal
    if 'top_ls_ratio' in df.columns and 'ls_ratio' in df.columns:
        df['whale_retail_gap'] = df['top_ls_ratio'] - df['ls_ratio']
        df['whale_signal'] = 'NEUTRAL'
        df.loc[df['whale_retail_gap'] > 0.3, 'whale_signal'] = 'WHALE_BULLISH'
        df.loc[df['whale_retail_gap'] < -0.3, 'whale_signal'] = 'WHALE_BEARISH'

    # Taker flow
    if 'futures_taker_ratio' in df.columns:
        taker_ma = df['futures_taker_ratio'].rolling(8, min_periods=1).mean()
        df['futures_flow'] = 'NEUTRAL'
        df.loc[taker_ma > 1.15, 'futures_flow'] = 'BUYERS_DOMINANT'
        df.loc[taker_ma < 0.85, 'futures_flow'] = 'SELLERS_DOMINANT'

    # OI signals
    if 'oi' in df.columns:
        df['oi_roc_1h'] = df['oi'].pct_change(4)

    return df


def compute_structure_bias(df_15m, lookback=48):
    """Compute structure bias using EMA + swing structure."""
    close = df_15m['Close'].astype(float)
    ema21 = close.ewm(span=21, adjust=False).mean()
    ema55 = close.ewm(span=55, adjust=False).mean()

    highs = df_15m['High'].astype(float)
    lows = df_15m['Low'].astype(float)

    bias = pd.Series('NEUTRAL', index=df_15m.index)

    for i in range(max(lookback, 55), len(df_15m)):
        ema_bull = ema21.iloc[i] > ema55.iloc[i]

        h_slice = highs.iloc[i-lookback:i+1]
        l_slice = lows.iloc[i-lookback:i+1]
        seg = lookback // 3
        h1, h2, h3 = h_slice.iloc[:seg].max(), h_slice.iloc[seg:2*seg].max(), h_slice.iloc[2*seg:].max()
        l1, l2, l3 = l_slice.iloc[:seg].min(), l_slice.iloc[seg:2*seg].min(), l_slice.iloc[2*seg:].min()

        higher_highs = h3 > h2 > h1
        higher_lows = l3 > l2 > l1
        lower_highs = h3 < h2 < h1
        lower_lows = l3 < l2 < l1

        if ema_bull and (higher_highs or higher_lows):
            bias.iloc[i] = 'BULLISH'
        elif not ema_bull and (lower_highs or lower_lows):
            bias.iloc[i] = 'BEARISH'
        elif ema_bull:
            bias.iloc[i] = 'BULLISH'
        else:
            bias.iloc[i] = 'BEARISH'

    return bias


def compute_forward_returns(df_15m, bars_list=[16, 48, 96]):
    """Compute forward returns at 4h/12h/24h horizons."""
    close = df_15m['Close'].astype(float)
    returns = {}
    for bars in bars_list:
        label = f"fwd_{bars*15//60}h"
        returns[label] = close.shift(-bars) / close - 1
    return pd.DataFrame(returns, index=df_15m.index)


# ═══════════════════════════════════════════════════════════════
# DATA PERSISTENCE
# ═══════════════════════════════════════════════════════════════

def save_derivatives_chunk(df, chunk_num):
    """Save a derivatives chunk to CSV, deduplicating with existing data."""
    filepath = os.path.join(DATA_DIR, 'derivatives_raw.csv')

    if os.path.exists(filepath):
        existing = pd.read_csv(filepath)
        existing['timestamp'] = pd.to_datetime(existing['timestamp'])
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        combined = pd.concat([existing, df]).drop_duplicates(subset=['timestamp']).sort_values('timestamp').reset_index(drop=True)
    else:
        combined = df.copy()

    combined.to_csv(filepath, index=False)
    print(f"  💾 Saved {len(combined)} total bars to {filepath}")
    return combined


def save_price_data(df_15m):
    """Save aligned price data."""
    filepath = os.path.join(DATA_DIR, 'price_15m.csv')
    df_15m.to_csv(filepath, index=False)
    print(f"  💾 Saved {len(df_15m)} price bars to {filepath}")


def save_analysis(results_df, chunk_num):
    """Save analysis results."""
    filepath = os.path.join(DATA_DIR, f'analysis_chunk_{chunk_num:03d}.csv')
    results_df.to_csv(filepath, index=False)
    print(f"  💾 Saved analysis to {filepath}")


def load_all_derivatives():
    """Load all saved derivatives data."""
    filepath = os.path.join(DATA_DIR, 'derivatives_raw.csv')
    if not os.path.exists(filepath):
        return pd.DataFrame()
    df = pd.read_csv(filepath)
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    return df


# ═══════════════════════════════════════════════════════════════
# ANALYSIS
# ═══════════════════════════════════════════════════════════════

def analyze_results(df_results, chunk_label=""):
    """Run analysis on results DataFrame and return stats dict."""
    if len(df_results) == 0:
        print("  ⚠️  No samples to analyze.")
        return None

    fwd_cols = [c for c in df_results.columns if c.startswith('fwd_')]

    aligned = df_results[df_results['aligned'] == True]
    diverged = df_results[df_results['aligned'] == False]

    whale_bear_struct_bull = df_results[
        (df_results['whale'] == 'WHALE_BEARISH') & (df_results['structure'] == 'BULLISH')
    ]
    whale_bear_struct_bear = df_results[
        (df_results['whale'] == 'WHALE_BEARISH') & (df_results['structure'] == 'BEARISH')
    ]
    whale_bull_struct_bear = df_results[
        (df_results['whale'] == 'WHALE_BULLISH') & (df_results['structure'] == 'BEARISH')
    ]
    whale_bull_struct_bull = df_results[
        (df_results['whale'] == 'WHALE_BULLISH') & (df_results['structure'] == 'BULLISH')
    ]

    def get_returns(subset):
        out = {}
        for col in fwd_cols:
            v = subset[col].dropna()
            if len(v) > 0:
                out[col] = {'mean': v.mean(), 'wr': (v > 0).mean(), 'n': len(v)}
            else:
                out[col] = {'mean': 0, 'wr': 0, 'n': 0}
        return out

    stats = {
        'chunk': chunk_label,
        'total_samples': len(df_results),
        'time_range': f"{df_results['timestamp'].iloc[0]} → {df_results['timestamp'].iloc[-1]}",
        'aligned': {'n': len(aligned), 'returns': get_returns(aligned)},
        'diverged': {'n': len(diverged), 'returns': get_returns(diverged)},
        'whale_bear_struct_bull': {'n': len(whale_bear_struct_bull), 'returns': get_returns(whale_bear_struct_bull)},
        'whale_bear_struct_bear': {'n': len(whale_bear_struct_bear), 'returns': get_returns(whale_bear_struct_bear)},
        'whale_bull_struct_bear': {'n': len(whale_bull_struct_bear), 'returns': get_returns(whale_bull_struct_bear)},
        'whale_bull_struct_bull': {'n': len(whale_bull_struct_bull), 'returns': get_returns(whale_bull_struct_bull)},
    }

    # Print results
    print(f"\n{'═' * 60}")
    print(f"  RESULTS — {chunk_label}")
    print(f"{'═' * 60}")
    print(f"  Samples: {stats['total_samples']}")
    print(f"  Range:   {stats['time_range']}")
    print(f"  Aligned: {stats['aligned']['n']}  |  Diverged: {stats['diverged']['n']}")

    print(f"\n  {'Scenario':<35} {'4h':>8} {'12h':>8} {'24h':>8} {'n':>5}")
    print(f"  {'─' * 35} {'─' * 8} {'─' * 8} {'─' * 8} {'─' * 5}")

    scenarios = [
        ('ALL ALIGNED', stats['aligned']),
        ('ALL DIVERGED', stats['diverged']),
        ('WHALE↓ + STRUCT↑ (diverged)', stats['whale_bear_struct_bull']),
        ('WHALE↓ + STRUCT↓ (aligned)', stats['whale_bear_struct_bear']),
        ('WHALE↑ + STRUCT↓ (diverged)', stats['whale_bull_struct_bear']),
        ('WHALE↑ + STRUCT↑ (aligned)', stats['whale_bull_struct_bull']),
    ]

    for label, data in scenarios:
        if data['n'] == 0:
            print(f"  {label:<35} {'—':>8} {'—':>8} {'—':>8} {'0':>5}")
            continue
        vals = []
        for col in fwd_cols:
            r = data['returns'].get(col, {})
            if r.get('n', 0) > 0:
                vals.append(f"{r['mean']*100:+.2f}%")
            else:
                vals.append("—")
        print(f"  {label:<35} {vals[0]:>8} {vals[1]:>8} {vals[2]:>8} {data['n']:>5}")

    # Win rates
    print(f"\n  {'Scenario':<35} {'4h':>8} {'12h':>8} {'24h':>8}")
    print(f"  {'─' * 35} {'─' * 8} {'─' * 8} {'─' * 8}")

    for label, data in scenarios:
        if data['n'] == 0:
            print(f"  {label:<35} {'—':>8} {'—':>8} {'—':>8}")
            continue
        vals = []
        for col in fwd_cols:
            r = data['returns'].get(col, {})
            if r.get('n', 0) > 0:
                vals.append(f"{r['wr']*100:.1f}%")
            else:
                vals.append("—")
        print(f"  {label:<35} {vals[0]:>8} {vals[1]:>8} {vals[2]:>8}")

    # Key question
    wb = stats['whale_bear_struct_bull']
    if wb['n'] > 0:
        print(f"\n  🔑 WHALE↓ vs STRUCT↑ (n={wb['n']}):")
        for col in fwd_cols:
            r = wb['returns'].get(col, {})
            if r.get('n', 0) > 0:
                whale_won = (1 - r['wr']) * 100  # whale said down, if return is negative whale won
                struct_won = r['wr'] * 100
                print(f"    {col}: whale wins {whale_won:.1f}% | struct wins {struct_won:.1f}% | avg {r['mean']*100:+.2f}%")

    return stats


def run_fetch_cycle(end_time_ms=None, chunk_num=1):
    """Fetch one chunk of derivatives + price data, compute signals, analyze."""
    print(f"\n{'═' * 60}")
    print(f"  CHUNK {chunk_num} — Fetching derivatives...")
    print(f"{'═' * 60}")

    if end_time_ms:
        end_dt = pd.to_datetime(end_time_ms, unit='ms')
        print(f"  End time: {end_dt}")

    # Fetch derivatives
    df_deriv = fetch_derivatives_chunk(end_time_ms=end_time_ms)
    if df_deriv is None or len(df_deriv) == 0:
        print("  ⚠️  No derivatives data returned.")
        return None, None

    print(f"  Derivatives: {len(df_deriv)} bars")
    print(f"  Range: {df_deriv['timestamp'].iloc[0]} → {df_deriv['timestamp'].iloc[-1]}")

    # Save derivatives
    combined_deriv = save_derivatives_chunk(df_deriv, chunk_num)

    # Fetch price data covering the same range (need extra for EMA warmup + forward returns)
    ts_min = df_deriv['timestamp_ms'].min()
    ts_max = df_deriv['timestamp_ms'].max()
    # We need price data from ~2000 bars before ts_min (for EMA55 warmup) to ~96 bars after ts_max
    # Fetch 1500 bars ending at ts_max + 24h (for forward returns)
    price_end_ms = ts_max + (96 * 15 * 60 * 1000)  # +24h

    print(f"  Fetching price data...")
    # Use ccxt to fetch historical price data
    import ccxt
    ex = ccxt.binance({"enableRateLimit": True})
    price_rows = []
    fetch_since = ts_min - (2000 * 15 * 60 * 1000)  # ~2000 bars before

    while fetch_since < price_end_ms:
        try:
            ohlcv = ex.fetch_ohlcv('ETH/USDT', '15m', since=fetch_since, limit=1000)
            if not ohlcv:
                break
            price_rows.extend(ohlcv)
            fetch_since = ohlcv[-1][0] + 1
            if len(ohlcv) < 1000:
                break
        except Exception as e:
            print(f"  ⚠️  Price fetch error: {e}")
            break

    if not price_rows:
        print("  ⚠️  No price data fetched.")
        return df_deriv, None

    df_price = pd.DataFrame(price_rows, columns=['Open time ms', 'Open', 'High', 'Low', 'Close', 'Volume'])
    df_price['Open time'] = pd.to_datetime(df_price['Open time ms'], unit='ms')
    df_price = df_price.drop_duplicates(subset=['Open time ms']).sort_values('Open time ms').reset_index(drop=True)
    print(f"  Price: {len(df_price)} bars")

    save_price_data(df_price)

    # Compute signals
    print(f"  Computing signals...")
    df_deriv = compute_positioning_signals(df_deriv)
    structure_bias = compute_structure_bias(df_price)
    fwd_returns = compute_forward_returns(df_price)

    # Save raw derivative timestamps for pagination (before reindex)
    deriv_raw_earliest_ms = int(df_deriv['timestamp_ms'].min()) if 'timestamp_ms' in df_deriv.columns else None

    # Align derivatives to price timestamps
    df_price['ts'] = pd.to_datetime(df_price['Open time'])
    df_deriv['ts'] = pd.to_datetime(df_deriv['timestamp'])
    df_price = df_price.set_index('ts')
    df_deriv = df_deriv.set_index('ts').resample('15min').last().dropna(how='all')
    df_deriv = df_deriv.reindex(df_price.index, method='ffill')

    # Build results
    results = []
    min_idx = max(55, 48)
    for i in range(min_idx, len(df_price) - 96):
        ws = df_deriv['whale_signal'].iloc[i] if i < len(df_deriv) and 'whale_signal' in df_deriv.columns else 'NEUTRAL'
        sb = structure_bias.iloc[i] if i < len(structure_bias) else 'NEUTRAL'

        # Handle NaN from ffill alignment
        if not isinstance(ws, str) or ws == 'nan' or (isinstance(ws, float) and np.isnan(ws)):
            ws = 'NEUTRAL'
        if not isinstance(sb, str) or sb == 'nan' or (isinstance(sb, float) and np.isnan(sb)):
            sb = 'NEUTRAL'

        if ws == 'NEUTRAL' or sb == 'NEUTRAL':
            continue

        row = {
            'idx': i,
            'timestamp': str(df_price.index[i]),
            'price': float(df_price['Close'].iloc[i]),
            'whale': ws,
            'structure': sb,
            'aligned': ws.replace('WHALE_', '') == sb,
        }
        for col in fwd_returns.columns:
            val = fwd_returns[col].iloc[i] if i < len(fwd_returns) else None
            row[col] = float(val) if val is not None and not pd.isna(val) else None

        results.append(row)

    df_results = pd.DataFrame(results)
    save_analysis(df_results, chunk_num)

    # Use the raw derivatives earliest timestamp (saved before reindex)
    earliest_ts = deriv_raw_earliest_ms

    stats = analyze_results(df_results, f"Chunk {chunk_num}")

    return df_results, earliest_ts


def run_full_analysis():
    """Analyze all saved data combined."""
    print(f"\n{'═' * 60}")
    print(f"  COMBINED ANALYSIS — ALL CHUNKS")
    print(f"{'═' * 60}")

    # Load all analysis files
    all_results = []
    for f in sorted(os.listdir(DATA_DIR)):
        if f.startswith('analysis_chunk_') and f.endswith('.csv'):
            df = pd.read_csv(os.path.join(DATA_DIR, f))
            all_results.append(df)

    if not all_results:
        print("  ⚠️  No analysis files found.")
        return None

    combined = pd.concat(all_results, ignore_index=True)
    combined = combined.drop_duplicates(subset=['timestamp']).sort_values('timestamp').reset_index(drop=True)

    return analyze_results(combined, "COMBINED")


# ═══════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description='Whale vs Structure Backtest')
    parser.add_argument('--paginate', type=int, default=1, help='Number of 500-bar chunks to fetch')
    parser.add_argument('--analyze', action='store_true', help='Analyze all saved data')
    args = parser.parse_args()

    if args.analyze:
        run_full_analysis()
        return

    prev_stats = None
    end_time_ms = None  # None = latest

    for chunk in range(1, args.paginate + 1):
        df_results, earliest_ts = run_fetch_cycle(
            end_time_ms=end_time_ms,
            chunk_num=chunk,
        )

        if df_results is not None and len(df_results) > 0:
            # Compare to previous chunk if available
            if prev_stats and chunk > 1:
                print(f"\n{'─' * 60}")
                print(f"  COMPARISON: Chunk {chunk} vs Chunk {chunk-1}")
                print(f"{'─' * 60}")

                for scenario in ['aligned', 'diverged', 'whale_bear_struct_bull']:
                    prev_n = prev_stats.get(scenario, {}).get('n', 0)
                    curr_n = None
                    # Get current scenario stats
                    if scenario == 'aligned':
                        curr_n = len(df_results[df_results['aligned'] == True])
                    elif scenario == 'diverged':
                        curr_n = len(df_results[df_results['aligned'] == False])
                    elif scenario == 'whale_bear_struct_bull':
                        curr_n = len(df_results[(df_results['whale'] == 'WHALE_BEARISH') & (df_results['structure'] == 'BULLISH')])

                    print(f"  {scenario}: {prev_n} → {curr_n} samples")

        # Paginate: set end_time to earliest timestamp of this chunk
        if earliest_ts:
            end_time_ms = earliest_ts - 1
        else:
            print("  ⚠️  No more data to paginate.")
            break

        prev_stats = None  # Reset for simplicity — combined analysis handles accumulation

    # Final combined analysis
    if args.paginate > 1:
        print("\n" + "=" * 60)
        print("  FINAL COMBINED ANALYSIS")
        print("=" * 60)
        run_full_analysis()


if __name__ == '__main__':
    main()

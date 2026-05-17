#!/usr/bin/env python3
"""
Prompt A + B: Backtest CB Consumer Confidence (2018-today) using ETH/USDT 15m data.

Conference Board Consumer Confidence released at 10:00 ET (14:00 UTC)
on the last Tuesday of each month.

Session itinerary:
  US Morning (14:00 UTC) → Rate Path Evaluation → Spending Outlook

We measure:
  1. ETH returns across all session phases
  2. Wyckoff phase, vol regime, signal classification
  3. Cross-tabulation: (wyckoff × vol × signal) → avg 24h return, win rate
  4. Session transmission chain (Prompt B)
  5. Statistical significance
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from scipy import stats
import json
import os
import warnings
warnings.filterwarnings('ignore')

# ═══════════════════════════════════════════════════════════════
# CB CONSUMER CONFIDENCE RELEASE DATES (10:00 ET = 14:00 UTC)
# Format: {date: {'actual': float, 'consensus': float, 'prev': float}}
# ═══════════════════════════════════════════════════════════════

RELEASES = {
    # 2018
    '2018-01-30': {'actual': 125.4, 'consensus': 123.0, 'prev': 122.1},
    '2018-02-27': {'actual': 130.8, 'consensus': 126.5, 'prev': 125.4},
    '2018-03-27': {'actual': 127.7, 'consensus': 131.0, 'prev': 130.8},
    '2018-04-24': {'actual': 128.7, 'consensus': 126.0, 'prev': 127.0},
    '2018-05-29': {'actual': 128.0, 'consensus': 128.0, 'prev': 125.6},
    '2018-06-26': {'actual': 126.4, 'consensus': 127.5, 'prev': 128.0},
    '2018-07-31': {'actual': 127.4, 'consensus': 126.5, 'prev': 126.4},
    '2018-08-28': {'actual': 133.4, 'consensus': 126.6, 'prev': 127.4},
    '2018-09-25': {'actual': 138.4, 'consensus': 132.0, 'prev': 133.4},
    '2018-10-30': {'actual': 137.9, 'consensus': 136.0, 'prev': 138.4},
    '2018-11-27': {'actual': 135.7, 'consensus': 135.9, 'prev': 137.9},
    '2018-12-18': {'actual': 128.1, 'consensus': 133.5, 'prev': 135.7},
    # 2019
    '2019-01-29': {'actual': 120.2, 'consensus': 124.0, 'prev': 128.1},
    '2019-02-26': {'actual': 131.4, 'consensus': 124.7, 'prev': 120.2},
    '2019-03-26': {'actual': 124.1, 'consensus': 132.5, 'prev': 131.4},
    '2019-04-30': {'actual': 129.2, 'consensus': 126.0, 'prev': 124.1},
    '2019-05-28': {'actual': 134.1, 'consensus': 130.0, 'prev': 129.2},
    '2019-06-25': {'actual': 121.5, 'consensus': 131.0, 'prev': 134.1},
    '2019-07-30': {'actual': 135.7, 'consensus': 127.0, 'prev': 121.5},
    '2019-08-27': {'actual': 135.1, 'consensus': 135.8, 'prev': 135.7},
    '2019-09-24': {'actual': 125.1, 'consensus': 133.5, 'prev': 135.1},
    '2019-10-29': {'actual': 137.9, 'consensus': 128.5, 'prev': 125.1},
    '2019-11-26': {'actual': 125.5, 'consensus': 127.0, 'prev': 137.9},
    '2019-12-31': {'actual': 126.5, 'consensus': 128.2, 'prev': 125.5},
    # 2020
    '2020-01-28': {'actual': 131.6, 'consensus': 128.0, 'prev': 126.5},
    '2020-02-25': {'actual': 130.7, 'consensus': 132.6, 'prev': 131.6},
    '2020-03-31': {'actual': 120.0, 'consensus': 110.0, 'prev': 130.7},
    '2020-04-28': {'actual': 86.9, 'consensus': 87.0, 'prev': 120.0},
    '2020-05-26': {'actual': 86.6, 'consensus': 82.3, 'prev': 86.9},
    '2020-06-30': {'actual': 98.1, 'consensus': 91.8, 'prev': 86.6},
    '2020-07-28': {'actual': 92.6, 'consensus': 94.5, 'prev': 98.1},
    '2020-08-25': {'actual': 101.1, 'consensus': 93.0, 'prev': 92.6},
    '2020-09-29': {'actual': 101.8, 'consensus': 90.0, 'prev': 101.1},
    '2020-10-27': {'actual': 100.9, 'consensus': 101.0, 'prev': 101.8},
    '2020-11-24': {'actual': 96.1, 'consensus': 98.0, 'prev': 100.9},
    '2020-12-22': {'actual': 88.6, 'consensus': 97.0, 'prev': 96.1},
    # 2021
    '2021-01-26': {'actual': 89.3, 'consensus': 89.0, 'prev': 88.6},
    '2021-02-23': {'actual': 90.4, 'consensus': 90.0, 'prev': 89.3},
    '2021-03-30': {'actual': 109.7, 'consensus': 96.0, 'prev': 90.4},
    '2021-04-27': {'actual': 121.7, 'consensus': 112.0, 'prev': 109.7},
    '2021-05-25': {'actual': 117.2, 'consensus': 118.8, 'prev': 121.7},
    '2021-06-29': {'actual': 127.3, 'consensus': 119.0, 'prev': 117.2},
    '2021-07-27': {'actual': 129.1, 'consensus': 124.0, 'prev': 127.3},
    '2021-08-31': {'actual': 113.8, 'consensus': 123.0, 'prev': 129.1},
    '2021-09-28': {'actual': 109.3, 'consensus': 114.5, 'prev': 113.8},
    '2021-10-26': {'actual': 113.8, 'consensus': 108.0, 'prev': 109.3},
    '2021-11-23': {'actual': 109.5, 'consensus': 111.0, 'prev': 113.8},
    '2021-12-21': {'actual': 115.8, 'consensus': 111.0, 'prev': 109.5},
    # 2022
    '2022-01-25': {'actual': 113.8, 'consensus': 111.8, 'prev': 115.8},
    '2022-02-22': {'actual': 110.5, 'consensus': 110.0, 'prev': 113.8},
    '2022-03-29': {'actual': 107.2, 'consensus': 107.0, 'prev': 110.5},
    '2022-04-26': {'actual': 107.3, 'consensus': 108.5, 'prev': 107.2},
    '2022-05-31': {'actual': 106.4, 'consensus': 103.9, 'prev': 107.3},
    '2022-06-28': {'actual': 98.7, 'consensus': 100.4, 'prev': 106.4},
    '2022-07-26': {'actual': 95.7, 'consensus': 97.2, 'prev': 98.7},
    '2022-08-30': {'actual': 103.2, 'consensus': 97.5, 'prev': 95.7},
    '2022-09-27': {'actual': 108.0, 'consensus': 104.5, 'prev': 103.2},
    '2022-10-25': {'actual': 102.5, 'consensus': 106.0, 'prev': 108.0},
    '2022-11-29': {'actual': 100.2, 'consensus': 100.0, 'prev': 102.5},
    '2022-12-21': {'actual': 108.3, 'consensus': 101.0, 'prev': 100.2},
    # 2023
    '2023-01-31': {'actual': 107.1, 'consensus': 109.0, 'prev': 108.3},
    '2023-02-28': {'actual': 102.9, 'consensus': 108.5, 'prev': 107.1},
    '2023-03-28': {'actual': 104.2, 'consensus': 101.0, 'prev': 102.9},
    '2023-04-25': {'actual': 101.3, 'consensus': 104.0, 'prev': 104.2},
    '2023-05-30': {'actual': 102.3, 'consensus': 99.0, 'prev': 101.3},
    '2023-06-27': {'actual': 109.7, 'consensus': 103.9, 'prev': 102.3},
    '2023-07-25': {'actual': 117.0, 'consensus': 111.8, 'prev': 109.7},
    '2023-08-29': {'actual': 106.1, 'consensus': 116.0, 'prev': 117.0},
    '2023-09-26': {'actual': 103.0, 'consensus': 105.5, 'prev': 106.1},
    '2023-10-31': {'actual': 102.6, 'consensus': 100.0, 'prev': 103.0},
    '2023-11-28': {'actual': 102.0, 'consensus': 101.0, 'prev': 102.6},
    '2023-12-20': {'actual': 110.7, 'consensus': 104.0, 'prev': 102.0},
    # 2024
    '2024-01-30': {'actual': 114.8, 'consensus': 115.0, 'prev': 110.7},
    '2024-02-27': {'actual': 106.7, 'consensus': 115.0, 'prev': 114.8},
    '2024-03-26': {'actual': 104.7, 'consensus': 107.0, 'prev': 106.7},
    '2024-04-30': {'actual': 97.5, 'consensus': 104.0, 'prev': 104.7},
    '2024-05-28': {'actual': 102.0, 'consensus': 96.0, 'prev': 97.5},
    '2024-06-25': {'actual': 100.4, 'consensus': 100.0, 'prev': 102.0},
    '2024-07-30': {'actual': 100.3, 'consensus': 99.8, 'prev': 100.4},
    '2024-08-27': {'actual': 103.3, 'consensus': 100.0, 'prev': 100.3},
    '2024-09-24': {'actual': 98.7, 'consensus': 104.0, 'prev': 103.3},
    '2024-10-29': {'actual': 108.7, 'consensus': 99.5, 'prev': 98.7},
    '2024-11-26': {'actual': 111.7, 'consensus': 112.0, 'prev': 108.7},
    '2024-12-23': {'actual': 104.7, 'consensus': 113.0, 'prev': 111.7},
    # 2025
    '2025-01-28': {'actual': 104.1, 'consensus': 106.0, 'prev': 104.7},
    '2025-02-25': {'actual': 98.3, 'consensus': 102.5, 'prev': 104.1},
    '2025-03-25': {'actual': 93.9, 'consensus': 94.0, 'prev': 98.3},
    '2025-04-29': {'actual': 86.0, 'consensus': 87.0, 'prev': 93.9},
    '2025-05-27': {'actual': 93.0, 'consensus': 87.0, 'prev': 86.0},
    '2025-06-24': {'actual': 98.4, 'consensus': 95.0, 'prev': 93.0},
    '2025-07-29': {'actual': 97.2, 'consensus': 96.0, 'prev': 98.4},
    '2025-08-26': {'actual': 97.4, 'consensus': 96.5, 'prev': 97.2},
    '2025-09-30': {'actual': 94.2, 'consensus': 96.0, 'prev': 97.4},
    '2025-10-28': {'actual': 94.6, 'consensus': 93.5, 'prev': 94.2},
    '2025-11-25': {'actual': 92.0, 'consensus': 93.0, 'prev': 94.6},
    '2025-12-23': {'actual': 90.7, 'consensus': 91.5, 'prev': 92.0},
    # 2026
    '2026-01-27': {'actual': 91.3, 'consensus': 90.5, 'prev': 90.7},
    '2026-02-24': {'actual': 92.8, 'consensus': 91.5, 'prev': 91.3},
    '2026-03-31': {'actual': 89.2, 'consensus': 92.0, 'prev': 92.8},
    '2026-04-28': {'actual': 87.5, 'consensus': 89.0, 'prev': 89.2},
    '2026-05-26': {'actual': None, 'consensus': None, 'prev': 87.5},  # upcoming
}

# ═══════════════════════════════════════════════════════════════
# SESSION DEFINITIONS (UTC offsets from release time 14:00 UTC)
# ═══════════════════════════════════════════════════════════════

SESSIONS = {
    'Pre-Asia Post-NY':  {'start': -4.5, 'end': 0},      # 09:30-14:00 UTC (pre-release)
    'Release Spike':     {'start': 0, 'end': 0.25},       # 14:00-14:15 UTC
    'NY AM':             {'start': 0.25, 'end': 3.0},     # 14:15-17:00 UTC
    'NY Lunch':          {'start': 3.0, 'end': 4.5},      # 17:00-18:30 UTC
    'NY PM':             {'start': 4.5, 'end': 6.5},      # 18:30-20:30 UTC
    'NY Close':          {'start': 6.5, 'end': 8.0},      # 20:30-22:00 UTC
    'Pre-Asia':          {'start': 8.0, 'end': 10.0},     # 22:00-00:00 UTC
    'Sydney Open':       {'start': 10.0, 'end': 11.0},    # 00:00-01:00 UTC
    'Tokyo Open':        {'start': 11.0, 'end': 12.0},    # 01:00-02:00 UTC
    'Asia Mid':          {'start': 12.0, 'end': 14.0},    # 02:00-04:00 UTC
    'Asia Afternoon':    {'start': 14.0, 'end': 16.0},    # 04:00-06:00 UTC
    'Tokyo Close':       {'start': 16.0, 'end': 17.0},    # 06:00-07:00 UTC
    'Pre-London':        {'start': 17.0, 'end': 18.0},    # 07:00-08:00 UTC
    'Frankfurt Open':    {'start': 18.0, 'end': 19.0},    # 08:00-09:00 UTC
    'London Open':       {'start': 19.0, 'end': 20.0},    # 09:00-10:00 UTC
    'London Morning':    {'start': 20.0, 'end': 22.0},    # 10:00-12:00 UTC
    'London Midday':     {'start': 22.0, 'end': 24.0},    # 12:00-14:00 UTC
    'NY Pre-Open':       {'start': 24.0, 'end': 25.5},    # 14:00-15:30 UTC (next day)
    'NY Open Day2':      {'start': 25.5, 'end': 27.0},    # 15:30-17:00 UTC
    'London-NY Overlap': {'start': 27.0, 'end': 28.0},    # 17:00-18:00 UTC
    'NY AM Day2':        {'start': 28.0, 'end': 31.0},    # 18:00-21:00 UTC
    'NY Lunch Day2':     {'start': 31.0, 'end': 32.5},    # 21:00-22:30 UTC
    'NY PM Day2':        {'start': 32.5, 'end': 34.5},    # 22:30-00:30 UTC (28h total)
}


def load_eth_data(csv_path):
    """Load and prepare ETH 15m data."""
    df = pd.read_csv(csv_path)
    # Try common column names
    for ts_col in ['timestamp', 'open_time', 'date', 'datetime', 'time']:
        if ts_col in df.columns:
            break
    else:
        ts_col = df.columns[0]

    df['timestamp'] = pd.to_datetime(df[ts_col])
    df = df.sort_values('timestamp').reset_index(drop=True)

    for col in ['open', 'high', 'low', 'close']:
        for c in [col, col.capitalize(), col.upper()]:
            if c in df.columns:
                df[col] = pd.to_numeric(df[c], errors='coerce')
                break

    if 'volume' in df.columns:
        df['volume'] = pd.to_numeric(df['volume'], errors='coerce')

    return df


def get_price_at(df, ts):
    """Get close price at or just before timestamp."""
    mask = df['timestamp'] <= ts
    if mask.sum() == 0:
        return None
    return df.loc[mask].iloc[-1]['close']


def classify_signal(actual, consensus, prev):
    """Classify CB Consumer Confidence surprise."""
    if actual is None or consensus is None:
        return 'NO_DATA'
    surprise = actual - consensus
    surprise_pct = surprise / consensus * 100 if consensus != 0 else 0
    change = actual - prev if prev else 0

    if surprise_pct >= 3.0:
        return 'STRONG_BEAT'
    elif surprise_pct >= 1.0:
        return 'BEAT'
    elif surprise_pct >= -1.0:
        if change > 2:
            return 'RISING_INLINE'
        elif change < -2:
            return 'FALLING_INLINE'
        return 'INLINE'
    elif surprise_pct >= -3.0:
        return 'MISS'
    else:
        return 'BIG_MISS'


def classify_level(actual):
    """Classify absolute confidence level."""
    if actual is None:
        return 'NO_DATA'
    if actual >= 120:
        return 'STRONG'
    elif actual >= 110:
        return 'OPTIMISTIC'
    elif actual >= 100:
        return 'NEUTRAL'
    elif actual >= 90:
        return 'WEAK'
    else:
        return 'PESSIMISTIC'


def classify_wyckoff(price, ema21, ema55, atr_pct):
    """Simplified Wyckoff phase classifier."""
    if price is None or ema21 is None or ema55 is None:
        return 'UNKNOWN'
    if ema21 > ema55:
        if price > ema21:
            return 'MARKUP'
        else:
            return 'DISTRIBUTION'
    elif ema21 < ema55:
        if price < ema21:
            return 'MARKDOWN'
        else:
            return 'ACCUMULATION'
    else:
        return 'RANGE'


def classify_vol(atr, atr_ma):
    """Simplified vol regime."""
    if atr is None or atr_ma is None or atr_ma == 0:
        return 'UNKNOWN'
    ratio = atr / atr_ma
    if ratio > 1.5:
        return 'CRISIS'
    elif ratio > 1.2:
        return 'TREND'
    elif ratio > 0.9:
        return 'NEUTRAL'
    elif ratio > 0.6:
        return 'COMPRESSING'
    else:
        return 'LOW_VOL'


def compute_session_returns(df, release_ts):
    """Compute returns for each session phase."""
    results = {}
    price_release = get_price_at(df, release_ts)
    if price_release is None:
        return results

    for name, times in SESSIONS.items():
        start_ts = release_ts + timedelta(hours=times['start'])
        end_ts = release_ts + timedelta(hours=times['end'])

        p_start = get_price_at(df, start_ts)
        p_end = get_price_at(df, end_ts)

        if p_start and p_end and p_start > 0:
            ret = (p_end - p_start) / p_start * 100
            results[name] = {'return': ret, 'start_price': p_start, 'end_price': p_end}
        else:
            results[name] = {'return': 0, 'start_price': p_start, 'end_price': p_end}

    # 24h aggregate
    p_24h = get_price_at(df, release_ts + timedelta(hours=24))
    if price_release and p_24h and price_release > 0:
        results['24h_aggregate'] = {
            'return': (p_24h - price_release) / price_release * 100,
            'start_price': price_release,
            'end_price': p_24h
        }

    return results


def compute_ema(series, period):
    return series.ewm(span=period, adjust=False).mean()


def run_backtest(csv_path):
    """Main backtest loop."""
    df = load_eth_data(csv_path)

    # Pre-compute indicators
    df['ema21'] = compute_ema(df['close'], 21)
    df['ema55'] = compute_ema(df['close'], 55)
    df['atr'] = (df['high'] - df['low']).rolling(14).mean()
    df['atr_ma'] = df['atr'].rolling(96).mean()
    df['atr_pct'] = df['atr'] / df['close'] * 100

    all_results = []

    for date_str, data in RELEASES.items():
        actual = data['actual']
        if actual is None:
            continue  # skip future releases

        release_ts = pd.Timestamp(date_str + 'T14:00:00')

        # Find nearest bar
        mask = df['timestamp'] <= release_ts
        if mask.sum() == 0:
            continue
        idx = mask.sum() - 1
        row = df.iloc[idx]

        # Classify
        signal = classify_signal(actual, data['consensus'], data['prev'])
        level = classify_level(actual)
        wyckoff = classify_wyckoff(row['close'], row['ema21'], row['ema55'], row['atr_pct'])
        vol = classify_vol(row['atr'], row['atr_ma'])

        # Session returns
        returns = compute_session_returns(df, release_ts)

        result = {
            'date': date_str,
            'actual': actual,
            'consensus': data['consensus'],
            'prev': data['prev'],
            'surprise': actual - data['consensus'],
            'surprise_pct': (actual - data['consensus']) / data['consensus'] * 100,
            'change': actual - data['prev'],
            'signal': signal,
            'level': level,
            'wyckoff': wyckoff,
            'vol': vol,
            'returns': returns,
        }
        all_results.append(result)

    return all_results


def analyze_results(results):
    """Prompt A: Cross-tabulate and report edges."""
    print("\n" + "="*80)
    print("PROMPT A: CB CONSUMER CONFIDENCE BACKTEST RESULTS")
    print("="*80)

    # Overall stats
    agg_returns = [r['returns'].get('24h_aggregate', {}).get('return', 0) for r in results]
    agg_returns = [r for r in agg_returns if r != 0]

    print(f"\nTotal releases analyzed: {len(results)}")
    print(f"24h aggregate: {np.mean(agg_returns):.3f}% avg, "
          f"{sum(1 for r in agg_returns if r > 0)/len(agg_returns)*100:.1f}% win rate, "
          f"n={len(agg_returns)}")

    # t-test vs 0
    t_stat, p_val = stats.ttest_1samp(agg_returns, 0)
    print(f"One-sample t-test vs 0: t={t_stat:.3f}, p={p_val:.4f} "
          f"{'✅ SIGNIFICANT' if p_val < 0.05 else '❌ NOT significant'}")

    # Signal breakdown
    print(f"\n--- Signal Breakdown ---")
    signals = {}
    for r in results:
        sig = r['signal']
        ret_24h = r['returns'].get('24h_aggregate', {}).get('return', 0)
        if sig not in signals:
            signals[sig] = []
        signals[sig].append(ret_24h)

    for sig, rets in sorted(signals.items()):
        print(f"  {sig:20s}: {np.mean(rets):+.3f}% avg, "
              f"{sum(1 for r in rets if r > 0)/len(rets)*100:.1f}% win, "
              f"n={len(rets)}")

    # Level breakdown
    print(f"\n--- Level Breakdown ---")
    levels = {}
    for r in results:
        lvl = r['level']
        ret_24h = r['returns'].get('24h_aggregate', {}).get('return', 0)
        if lvl not in levels:
            levels[lvl] = []
        levels[lvl].append(ret_24h)

    for lvl, rets in sorted(levels.items()):
        print(f"  {lvl:20s}: {np.mean(rets):+.3f}% avg, "
              f"{sum(1 for r in rets if r > 0)/len(rets)*100:.1f}% win, "
              f"n={len(rets)}")

    # Cross-tabulation: Wyckoff × Vol × Signal
    print(f"\n--- Cross-Tabulation: Wyckoff × Vol × Signal (n≥3) ---")
    cross = {}
    for r in results:
        key = (r['wyckoff'], r['vol'], r['signal'])
        ret_24h = r['returns'].get('24h_aggregate', {}).get('return', 0)
        if key not in cross:
            cross[key] = []
        cross[key].append(ret_24h)

    edges = []
    for key, rets in sorted(cross.items()):
        if len(rets) >= 3:
            avg = np.mean(rets)
            wr = sum(1 for r in rets if r > 0) / len(rets) * 100
            wyck, vol, sig = key
            print(f"  {wyck:15s} × {vol:12s} × {sig:16s}: "
                  f"{avg:+.3f}% avg, {wr:.0f}% win, n={len(rets)}")
            if abs(avg) >= 0.5:
                edges.append((key, avg, wr, len(rets)))

    print(f"\n--- Edges (|avg| ≥ 0.5%, n ≥ 3) ---")
    for key, avg, wr, n in sorted(edges, key=lambda x: abs(x[1]), reverse=True):
        wyck, vol, sig = key
        direction = 'LONG' if avg > 0 else 'SHORT'
        print(f"  {wyck} × {vol} × {sig} → {direction}: {avg:+.3f}%, {wr:.0f}% win, n={n}")

    # MISS vs BEAT significance
    miss_rets = [r['returns'].get('24h_aggregate', {}).get('return', 0)
                 for r in results if r['signal'] in ('MISS', 'BIG_MISS')]
    beat_rets = [r['returns'].get('24h_aggregate', {}).get('return', 0)
                 for r in results if r['signal'] in ('STRONG_BEAT', 'BEAT')]

    if miss_rets and beat_rets:
        t2, p2 = stats.ttest_ind(miss_rets, beat_rets)
        print(f"\n--- MISS vs BEAT ---")
        print(f"  MISS avg: {np.mean(miss_rets):.3f}% (n={len(miss_rets)})")
        print(f"  BEAT avg: {np.mean(beat_rets):.3f}% (n={len(beat_rets)})")
        print(f"  Two-sample t-test: t={t2:.3f}, p={p2:.4f} "
              f"{'✅ SIGNIFICANT' if p2 < 0.05 else '❌ NOT significant'}")

    return edges


def analyze_transmission(results):
    """Prompt B: Session-by-session transmission chain."""
    print("\n" + "="*80)
    print("PROMPT B: SESSION TRANSMISSION CHAIN ANALYSIS")
    print("="*80)

    session_names = list(SESSIONS.keys())
    session_names.append('24h_aggregate')

    # Direction persistence between consecutive sessions
    print(f"\n--- Direction Persistence (same direction %) ---")
    print(f"  Threshold: <55% = chain broken, 55-65% = marginal, >65% = real edge\n")

    transitions = {}
    for r in results:
        rets = r['returns']
        prev_dir = None
        prev_name = None
        for name in session_names:
            ret_data = rets.get(name, {})
            ret = ret_data.get('return', 0) if ret_data else 0
            curr_dir = 1 if ret > 0 else (-1 if ret < 0 else 0)

            if prev_dir is not None and prev_dir != 0 and curr_dir != 0:
                key = f"{prev_name} → {name}"
                if key not in transitions:
                    transitions[key] = {'same': 0, 'total': 0}
                transitions[key]['total'] += 1
                if prev_dir == curr_dir:
                    transitions[key]['same'] += 1

            prev_dir = curr_dir
            prev_name = name

    for key, data in sorted(transitions.items()):
        if data['total'] >= 5:
            pct = data['same'] / data['total'] * 100
            status = '✅' if pct > 65 else ('⚠️' if pct > 55 else '❌')
            print(f"  {key:45s}: {pct:.1f}% (n={data['total']}) {status}")

    # Release spike direction vs 24h
    print(f"\n--- Release Spike → 24h Persistence ---")
    spike_persist = {'same': 0, 'total': 0}
    for r in results:
        spike_ret = r['returns'].get('Release Spike', {}).get('return', 0)
        agg_ret = r['returns'].get('24h_aggregate', {}).get('return', 0)
        if spike_ret != 0 and agg_ret != 0:
            spike_persist['total'] += 1
            if (spike_ret > 0 and agg_ret > 0) or (spike_ret < 0 and agg_ret < 0):
                spike_persist['same'] += 1

    if spike_persist['total'] > 0:
        pct = spike_persist['same'] / spike_persist['total'] * 100
        print(f"  Spike → 24h: {pct:.1f}% (n={spike_persist['total']})")

    # Session return heatmap
    print(f"\n--- Average Return by Session ---")
    for name in session_names:
        rets = [r['returns'].get(name, {}).get('return', 0) for r in results]
        avg = np.mean(rets)
        wr = sum(1 for r in rets if r > 0) / len(rets) * 100 if rets else 0
        print(f"  {name:25s}: {avg:+.3f}% avg, {wr:.1f}% win")


if __name__ == '__main__':
    csv_path = os.path.join(os.path.dirname(__file__), 'eth_15m_merged.csv')
    if not os.path.exists(csv_path):
        csv_path = 'eth_15m_merged.csv'

    print("Loading ETH 15m data...")
    results = run_backtest(csv_path)

    edges = analyze_results(results)
    analyze_transmission(results)

    # Save results
    out_path = os.path.join(os.path.dirname(__file__), 'backtest_cb_consumer_confidence_results.json')
    # Make results JSON serializable
    json_results = []
    for r in results:
        jr = {k: v for k, v in r.items() if k != 'returns'}
        jr['returns'] = {}
        for sk, sv in r['returns'].items():
            jr['returns'][sk] = {kk: vv for kk, vv in sv.items()}
        json_results.append(jr)

    with open(out_path, 'w') as f:
        json.dump(json_results, f, indent=2, default=str)
    print(f"\nResults saved to {out_path}")

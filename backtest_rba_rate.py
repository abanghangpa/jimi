#!/usr/bin/env python3
"""
Prompt A + B: Backtest RBA Rate Decision (2018-today) using ETH/USDT 15m data.

Reserve Bank of Australia rate decision released at 03:30 UTC (11:30 MYT)
on first Tuesday of month (except January).

Session itinerary (per user #36):
  Asia Afternoon (11:30 MYT) → Commodities Desk → AUD/USD Execution

Thesis:
  Australia = liquid proxy for global risk & Chinese demand.
  Hawkish hold / surprise hike → AUD/USD up → institutional bid density in ETH.
  Strong AUD → resilient global risk appetite → constructive for alt assets
  as Western sessions open. Loopback: RBA calibrated against quarterly Australia CPI.

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
# RBA RATE DECISION RELEASE DATES (03:30 UTC = 11:30 MYT)
# Format: {date: {'rate': float, 'prev_rate': float, 'signal': str}}
# signal: HIKE / HOLD / CUT / HAWKISH_HOLD / DOVISH_HOLD
# ═══════════════════════════════════════════════════════════════

RELEASES = {
    # 2018
    '2018-02-06': {'rate': 1.50, 'prev_rate': 1.50, 'signal': 'HOLD'},
    '2018-03-06': {'rate': 1.50, 'prev_rate': 1.50, 'signal': 'HOLD'},
    '2018-04-03': {'rate': 1.50, 'prev_rate': 1.50, 'signal': 'HOLD'},
    '2018-05-01': {'rate': 1.50, 'prev_rate': 1.50, 'signal': 'HOLD'},
    '2018-06-05': {'rate': 1.50, 'prev_rate': 1.50, 'signal': 'HOLD'},
    '2018-07-03': {'rate': 1.50, 'prev_rate': 1.50, 'signal': 'HOLD'},
    '2018-08-07': {'rate': 1.50, 'prev_rate': 1.50, 'signal': 'HOLD'},
    '2018-09-04': {'rate': 1.50, 'prev_rate': 1.50, 'signal': 'HOLD'},
    '2018-10-02': {'rate': 1.50, 'prev_rate': 1.50, 'signal': 'HOLD'},
    '2018-11-06': {'rate': 1.50, 'prev_rate': 1.50, 'signal': 'HOLD'},
    '2018-12-04': {'rate': 1.50, 'prev_rate': 1.50, 'signal': 'HOLD'},
    # 2019
    '2019-02-05': {'rate': 1.50, 'prev_rate': 1.50, 'signal': 'HOLD'},
    '2019-03-05': {'rate': 1.50, 'prev_rate': 1.50, 'signal': 'HOLD'},
    '2019-04-02': {'rate': 1.50, 'prev_rate': 1.50, 'signal': 'HOLD'},
    '2019-05-07': {'rate': 1.50, 'prev_rate': 1.50, 'signal': 'HOLD'},
    '2019-06-04': {'rate': 1.25, 'prev_rate': 1.50, 'signal': 'CUT'},
    '2019-07-02': {'rate': 1.00, 'prev_rate': 1.25, 'signal': 'CUT'},
    '2019-08-06': {'rate': 1.00, 'prev_rate': 1.00, 'signal': 'HOLD'},
    '2019-09-03': {'rate': 1.00, 'prev_rate': 1.00, 'signal': 'HOLD'},
    '2019-10-01': {'rate': 0.75, 'prev_rate': 1.00, 'signal': 'CUT'},
    '2019-11-05': {'rate': 0.75, 'prev_rate': 0.75, 'signal': 'HOLD'},
    '2019-12-03': {'rate': 0.75, 'prev_rate': 0.75, 'signal': 'HOLD'},
    # 2020
    '2020-02-04': {'rate': 0.75, 'prev_rate': 0.75, 'signal': 'HOLD'},
    '2020-03-03': {'rate': 0.50, 'prev_rate': 0.75, 'signal': 'CUT'},
    '2020-03-19': {'rate': 0.25, 'prev_rate': 0.50, 'signal': 'CUT'},  # emergency cut
    '2020-04-07': {'rate': 0.25, 'prev_rate': 0.25, 'signal': 'HOLD'},
    '2020-05-05': {'rate': 0.25, 'prev_rate': 0.25, 'signal': 'HOLD'},
    '2020-06-02': {'rate': 0.25, 'prev_rate': 0.25, 'signal': 'HOLD'},
    '2020-07-07': {'rate': 0.25, 'prev_rate': 0.25, 'signal': 'HOLD'},
    '2020-08-04': {'rate': 0.25, 'prev_rate': 0.25, 'signal': 'HOLD'},
    '2020-09-01': {'rate': 0.25, 'prev_rate': 0.25, 'signal': 'HOLD'},
    '2020-10-06': {'rate': 0.25, 'prev_rate': 0.25, 'signal': 'HOLD'},
    '2020-11-03': {'rate': 0.10, 'prev_rate': 0.25, 'signal': 'CUT'},
    '2020-12-01': {'rate': 0.10, 'prev_rate': 0.10, 'signal': 'HOLD'},
    # 2021
    '2021-02-02': {'rate': 0.10, 'prev_rate': 0.10, 'signal': 'HOLD'},
    '2021-03-02': {'rate': 0.10, 'prev_rate': 0.10, 'signal': 'HOLD'},
    '2021-04-06': {'rate': 0.10, 'prev_rate': 0.10, 'signal': 'HOLD'},
    '2021-05-04': {'rate': 0.10, 'prev_rate': 0.10, 'signal': 'HOLD'},
    '2021-06-01': {'rate': 0.10, 'prev_rate': 0.10, 'signal': 'HOLD'},
    '2021-07-06': {'rate': 0.10, 'prev_rate': 0.10, 'signal': 'HOLD'},
    '2021-08-03': {'rate': 0.10, 'prev_rate': 0.10, 'signal': 'HOLD'},
    '2021-09-07': {'rate': 0.10, 'prev_rate': 0.10, 'signal': 'HOLD'},
    '2021-10-05': {'rate': 0.10, 'prev_rate': 0.10, 'signal': 'HOLD'},
    '2021-11-02': {'rate': 0.10, 'prev_rate': 0.10, 'signal': 'HOLD'},
    '2021-12-07': {'rate': 0.10, 'prev_rate': 0.10, 'signal': 'HOLD'},
    # 2022
    '2022-02-01': {'rate': 0.10, 'prev_rate': 0.10, 'signal': 'HOLD'},
    '2022-03-01': {'rate': 0.10, 'prev_rate': 0.10, 'signal': 'HOLD'},
    '2022-04-05': {'rate': 0.10, 'prev_rate': 0.10, 'signal': 'HOLD'},
    '2022-05-03': {'rate': 0.35, 'prev_rate': 0.10, 'signal': 'HIKE'},
    '2022-06-07': {'rate': 0.85, 'prev_rate': 0.35, 'signal': 'HIKE'},
    '2022-07-05': {'rate': 1.35, 'prev_rate': 0.85, 'signal': 'HIKE'},
    '2022-08-02': {'rate': 1.85, 'prev_rate': 1.35, 'signal': 'HIKE'},
    '2022-09-06': {'rate': 2.35, 'prev_rate': 1.85, 'signal': 'HIKE'},
    '2022-10-04': {'rate': 2.60, 'prev_rate': 2.35, 'signal': 'HIKE'},
    '2022-11-01': {'rate': 2.85, 'prev_rate': 2.60, 'signal': 'HIKE'},
    '2022-12-06': {'rate': 3.10, 'prev_rate': 2.85, 'signal': 'HIKE'},
    # 2023
    '2023-02-07': {'rate': 3.35, 'prev_rate': 3.10, 'signal': 'HIKE'},
    '2023-03-07': {'rate': 3.60, 'prev_rate': 3.35, 'signal': 'HIKE'},
    '2023-04-04': {'rate': 3.60, 'prev_rate': 3.60, 'signal': 'HOLD'},
    '2023-05-02': {'rate': 3.85, 'prev_rate': 3.60, 'signal': 'HIKE'},
    '2023-06-06': {'rate': 4.10, 'prev_rate': 3.85, 'signal': 'HIKE'},
    '2023-07-04': {'rate': 4.10, 'prev_rate': 4.10, 'signal': 'HOLD'},
    '2023-08-01': {'rate': 4.10, 'prev_rate': 4.10, 'signal': 'HOLD'},
    '2023-09-05': {'rate': 4.10, 'prev_rate': 4.10, 'signal': 'HOLD'},
    '2023-10-03': {'rate': 4.10, 'prev_rate': 4.10, 'signal': 'HOLD'},
    '2023-11-07': {'rate': 4.35, 'prev_rate': 4.10, 'signal': 'HIKE'},
    '2023-12-05': {'rate': 4.35, 'prev_rate': 4.35, 'signal': 'HOLD'},
    # 2024
    '2024-02-06': {'rate': 4.35, 'prev_rate': 4.35, 'signal': 'HOLD'},
    '2024-03-19': {'rate': 4.35, 'prev_rate': 4.35, 'signal': 'HOLD'},
    '2024-05-07': {'rate': 4.35, 'prev_rate': 4.35, 'signal': 'HOLD'},
    '2024-06-18': {'rate': 4.35, 'prev_rate': 4.35, 'signal': 'HOLD'},
    '2024-08-06': {'rate': 4.35, 'prev_rate': 4.35, 'signal': 'HOLD'},
    '2024-09-24': {'rate': 4.35, 'prev_rate': 4.35, 'signal': 'HOLD'},
    '2024-11-05': {'rate': 4.35, 'prev_rate': 4.35, 'signal': 'HOLD'},
    '2024-12-10': {'rate': 4.35, 'prev_rate': 4.35, 'signal': 'HOLD'},
    # 2025
    '2025-02-18': {'rate': 4.35, 'prev_rate': 4.35, 'signal': 'HOLD'},
    '2025-04-01': {'rate': 4.10, 'prev_rate': 4.35, 'signal': 'CUT'},
    '2025-05-20': {'rate': 3.85, 'prev_rate': 4.10, 'signal': 'CUT'},
    '2025-07-08': {'rate': 3.85, 'prev_rate': 3.85, 'signal': 'HOLD'},
    '2025-08-12': {'rate': 3.60, 'prev_rate': 3.85, 'signal': 'CUT'},
    '2025-09-30': {'rate': 3.60, 'prev_rate': 3.60, 'signal': 'HOLD'},
    '2025-11-04': {'rate': 3.35, 'prev_rate': 3.60, 'signal': 'CUT'},
    '2025-12-09': {'rate': 3.35, 'prev_rate': 3.35, 'signal': 'HOLD'},
    # 2026
    '2026-02-17': {'rate': 3.35, 'prev_rate': 3.35, 'signal': 'HOLD'},
    '2026-03-31': {'rate': 3.10, 'prev_rate': 3.35, 'signal': 'CUT'},
    '2026-05-05': {'rate': 3.10, 'prev_rate': 3.10, 'signal': 'HOLD'},
}

# ═══════════════════════════════════════════════════════════════
# SESSION DEFINITIONS (UTC offsets from release time 03:30 UTC)
# RBA = Asia session release, different from US/Europe events
# ═══════════════════════════════════════════════════════════════

SESSIONS = {
    'Release Spike':      {'start': 0, 'end': 0.25},       # 03:30-03:45 UTC
    'Asia Mid':           {'start': 0.25, 'end': 2.5},      # 03:45-06:00 UTC
    'Asia Afternoon':     {'start': 2.5, 'end': 4.5},       # 06:00-08:00 UTC
    'Tokyo Close':        {'start': 4.5, 'end': 5.5},       # 08:00-09:00 UTC
    'Pre-London':         {'start': 5.5, 'end': 6.5},       # 09:00-10:00 UTC
    'Frankfurt Open':     {'start': 6.5, 'end': 7.5},       # 10:00-11:00 UTC
    'London Open':        {'start': 7.5, 'end': 8.5},       # 11:00-12:00 UTC
    'London Morning':     {'start': 8.5, 'end': 10.5},      # 12:00-14:00 UTC
    'London Midday':      {'start': 10.5, 'end': 12.5},     # 14:00-16:00 UTC
    'NY Pre-Open':        {'start': 12.5, 'end': 14.0},     # 16:00-17:30 UTC
    'NY Open':            {'start': 14.0, 'end': 15.5},     # 17:30-19:00 UTC
    'London-NY Overlap':  {'start': 15.5, 'end': 16.5},     # 19:00-20:00 UTC
    'NY AM':              {'start': 16.5, 'end': 19.5},     # 20:00-23:00 UTC
    'NY Lunch':           {'start': 19.5, 'end': 21.0},     # 23:00-00:30 UTC
    'NY PM':              {'start': 21.0, 'end': 23.0},     # 00:30-02:30 UTC
    'NY Close':           {'start': 23.0, 'end': 24.5},     # 02:30-04:00 UTC
    'Pre-Asia':           {'start': 24.5, 'end': 26.5},     # 04:00-06:00 UTC
    'Sydney Open D2':     {'start': 26.5, 'end': 27.5},     # 06:00-07:00 UTC
    'Tokyo Open D2':      {'start': 27.5, 'end': 28.5},     # 07:00-08:00 UTC
    'Asia Mid D2':        {'start': 28.5, 'end': 30.5},     # 08:00-10:00 UTC
    'Asia Afternoon D2':  {'start': 30.5, 'end': 32.5},     # 10:00-12:00 UTC
    '24h_aggregate':      {'start': 0, 'end': 24.0},
}


def load_eth_data(csv_path):
    df = pd.read_csv(csv_path)
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
    mask = df['timestamp'] <= ts
    if mask.sum() == 0:
        return None
    return df.loc[mask].iloc[-1]['close']


def classify_signal(signal, rate, prev_rate):
    """Classify RBA rate decision surprise."""
    if rate is None:
        return 'NO_DATA'
    change = rate - prev_rate
    if signal == 'HIKE':
        return 'HIKE'
    elif signal == 'CUT':
        return 'CUT'
    elif signal == 'HAWKISH_HOLD':
        return 'HAWKISH_HOLD'
    elif signal == 'DOVISH_HOLD':
        return 'DOVISH_HOLD'
    else:
        return 'NEUTRAL_HOLD'


def classify_rate_level(rate):
    """Classify absolute rate level."""
    if rate is None:
        return 'NO_DATA'
    if rate >= 4.0:
        return 'RESTRICTIVE'
    elif rate >= 2.5:
        return 'NEUTRAL'
    elif rate >= 1.0:
        return 'ACCOMMODATIVE'
    else:
        return 'EMERGENCY'


def classify_cycle(rate, prev_rate):
    """Classify rate cycle direction."""
    if rate is None or prev_rate is None:
        return 'UNKNOWN'
    if rate > prev_rate:
        return 'TIGHTENING'
    elif rate < prev_rate:
        return 'EASING'
    else:
        return 'PAUSE'


def classify_wyckoff(price, ema21, ema55, atr_pct):
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
            results[name] = {'return': (p_end - p_start) / p_start * 100,
                             'start_price': p_start, 'end_price': p_end}
        else:
            results[name] = {'return': 0, 'start_price': p_start, 'end_price': p_end}
    p_24h = get_price_at(df, release_ts + timedelta(hours=24))
    if price_release and p_24h and price_release > 0:
        results['24h_aggregate'] = {
            'return': (p_24h - price_release) / price_release * 100,
            'start_price': price_release, 'end_price': p_24h}
    return results


def compute_ema(series, period):
    return series.ewm(span=period, adjust=False).mean()


def run_backtest(csv_path):
    df = load_eth_data(csv_path)
    df['ema21'] = compute_ema(df['close'], 21)
    df['ema55'] = compute_ema(df['close'], 55)
    df['atr'] = (df['high'] - df['low']).rolling(14).mean()
    df['atr_ma'] = df['atr'].rolling(96).mean()
    df['atr_pct'] = df['atr'] / df['close'] * 100

    all_results = []
    for date_str, data in RELEASES.items():
        release_ts = pd.Timestamp(date_str + 'T03:30:00')
        mask = df['timestamp'] <= release_ts
        if mask.sum() == 0:
            continue
        idx = mask.sum() - 1
        row = df.iloc[idx]

        signal = classify_signal(data['signal'], data['rate'], data['prev_rate'])
        level = classify_rate_level(data['rate'])
        cycle = classify_cycle(data['rate'], data['prev_rate'])
        wyckoff = classify_wyckoff(row['close'], row['ema21'], row['ema55'], row['atr_pct'])
        vol = classify_vol(row['atr'], row['atr_ma'])
        returns = compute_session_returns(df, release_ts)

        all_results.append({
            'date': date_str, 'rate': data['rate'], 'prev_rate': data['prev_rate'],
            'signal': signal, 'level': level, 'cycle': cycle,
            'wyckoff': wyckoff, 'vol': vol, 'returns': returns,
        })
    return all_results


def analyze_results(results):
    print("\n" + "="*80)
    print("PROMPT A: RBA RATE DECISION BACKTEST RESULTS")
    print("="*80)

    agg_returns = [r['returns'].get('24h_aggregate', {}).get('return', 0) for r in results]
    agg_returns = [r for r in agg_returns if r != 0]

    print(f"\nTotal releases analyzed: {len(results)}")
    print(f"24h aggregate: {np.mean(agg_returns):.3f}% avg, "
          f"{sum(1 for r in agg_returns if r > 0)/len(agg_returns)*100:.1f}% win rate, "
          f"n={len(agg_returns)}")

    t_stat, p_val = stats.ttest_1samp(agg_returns, 0)
    print(f"One-sample t-test vs 0: t={t_stat:.3f}, p={p_val:.4f} "
          f"{'✅ SIGNIFICANT' if p_val < 0.05 else '❌ NOT significant'}")

    # Signal breakdown
    print(f"\n--- Signal Breakdown ---")
    signals = {}
    for r in results:
        sig = r['signal']
        ret_24h = r['returns'].get('24h_aggregate', {}).get('return', 0)
        signals.setdefault(sig, []).append(ret_24h)
    for sig, rets in sorted(signals.items()):
        print(f"  {sig:20s}: {np.mean(rets):+.3f}% avg, "
              f"{sum(1 for r in rets if r > 0)/len(rets)*100:.1f}% win, n={len(rets)}")

    # Cycle breakdown
    print(f"\n--- Cycle Breakdown ---")
    cycles = {}
    for r in results:
        cyc = r['cycle']
        ret_24h = r['returns'].get('24h_aggregate', {}).get('return', 0)
        cycles.setdefault(cyc, []).append(ret_24h)
    for cyc, rets in sorted(cycles.items()):
        print(f"  {cyc:20s}: {np.mean(rets):+.3f}% avg, "
              f"{sum(1 for r in rets if r > 0)/len(rets)*100:.1f}% win, n={len(rets)}")

    # Level breakdown
    print(f"\n--- Level Breakdown ---")
    levels = {}
    for r in results:
        lvl = r['level']
        ret_24h = r['returns'].get('24h_aggregate', {}).get('return', 0)
        levels.setdefault(lvl, []).append(ret_24h)
    for lvl, rets in sorted(levels.items()):
        print(f"  {lvl:20s}: {np.mean(rets):+.3f}% avg, "
              f"{sum(1 for r in rets if r > 0)/len(rets)*100:.1f}% win, n={len(rets)}")

    # Cross-tabulation: Wyckoff × Vol × Signal
    print(f"\n--- Cross-Tabulation: Wyckoff × Vol × Signal (n≥3) ---")
    cross = {}
    for r in results:
        key = (r['wyckoff'], r['vol'], r['signal'])
        ret_24h = r['returns'].get('24h_aggregate', {}).get('return', 0)
        cross.setdefault(key, []).append(ret_24h)

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

    # HIKE vs CUT
    hike_rets = [r['returns'].get('24h_aggregate', {}).get('return', 0) for r in results if r['signal'] == 'HIKE']
    cut_rets = [r['returns'].get('24h_aggregate', {}).get('return', 0) for r in results if r['signal'] == 'CUT']
    hold_rets = [r['returns'].get('24h_aggregate', {}).get('return', 0) for r in results if r['signal'] == 'NEUTRAL_HOLD']
    if hike_rets and hold_rets:
        t2, p2 = stats.ttest_ind(hike_rets, hold_rets)
        print(f"\n--- HIKE vs NEUTRAL_HOLD ---")
        print(f"  HIKE avg: {np.mean(hike_rets):.3f}% (n={len(hike_rets)})")
        print(f"  HOLD avg: {np.mean(hold_rets):.3f}% (n={len(hold_rets)})")
        print(f"  Two-sample t-test: t={t2:.3f}, p={p2:.4f} "
              f"{'✅ SIGNIFICANT' if p2 < 0.05 else '❌ NOT significant'}")
    if cut_rets and hold_rets:
        t3, p3 = stats.ttest_ind(cut_rets, hold_rets)
        print(f"\n--- CUT vs NEUTRAL_HOLD ---")
        print(f"  CUT avg: {np.mean(cut_rets):.3f}% (n={len(cut_rets)})")
        print(f"  HOLD avg: {np.mean(hold_rets):.3f}% (n={len(hold_rets)})")
        print(f"  Two-sample t-test: t={t3:.3f}, p={p3:.4f} "
              f"{'✅ SIGNIFICANT' if p3 < 0.05 else '❌ NOT significant'}")

    return edges


def analyze_transmission(results):
    print("\n" + "="*80)
    print("PROMPT B: SESSION TRANSMISSION CHAIN ANALYSIS")
    print("="*80)

    session_names = list(SESSIONS.keys())

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
                transitions.setdefault(key, {'same': 0, 'total': 0})
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

    print(f"\n--- Average Return by Session ---")
    for name in session_names:
        rets = [r['returns'].get(name, {}).get('return', 0) for r in results]
        avg = np.mean(rets)
        wr = sum(1 for r in rets if r > 0) / len(rets) * 100 if rets else 0
        if name != '24h_aggregate':
            print(f"  {name:25s}: {avg:+.3f}% avg, {wr:.1f}% win")


if __name__ == '__main__':
    csv_path = os.path.join(os.path.dirname(__file__), 'eth_15m_merged.csv')
    if not os.path.exists(csv_path):
        csv_path = 'eth_15m_merged.csv'

    print("Loading ETH 15m data...")
    results = run_backtest(csv_path)
    edges = analyze_results(results)
    analyze_transmission(results)

    out_path = os.path.join(os.path.dirname(__file__), 'backtest_rba_rate_results.json')
    json_results = []
    for r in results:
        jr = {k: v for k, v in r.items() if k != 'returns'}
        jr['returns'] = {sk: sv for sk, sv in r['returns'].items()}
        json_results.append(jr)
    with open(out_path, 'w') as f:
        json.dump(json_results, f, indent=2, default=str)
    print(f"\nResults saved to {out_path}")

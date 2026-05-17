#!/usr/bin/env python3
"""
Prompt A + B: Backtest Australia Quarterly CPI (2018-today) using ETH/USDT 15m data.

ABS Quarterly CPI released at 00:30 UTC (08:30 MYT) ~4-5 weeks after quarter end.
Key metric: Trimmed Mean CPI (RBA's preferred underlying inflation measure).

Session itinerary (per user #37):
  Asia Morning (08:30 MYT) → AUD Liquidity Loop → RBA Meeting

Thesis:
  Hot quarterly CPI → persistent underlying inflation → rate cut expectations
  priced out → AUD strengthens → positive for correlated crypto risk.
  Primary input for next RBA Rate Decision.
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
# AUSTRALIA QUARTERLY CPI RELEASE DATES (00:30 UTC = 08:30 MYT)
# Format: {date: {'headline_yoy': float, 'trimmed_mean_yoy': float,
#                 'prev_headline_yoy': float, 'prev_trimmed_yoy': float,
#                 'quarter': str}}
# ═══════════════════════════════════════════════════════════════

RELEASES = {
    # 2018
    '2018-01-31': {'headline_yoy': 1.9, 'trimmed_mean_yoy': 1.9, 'prev_headline_yoy': 1.8, 'prev_trimmed_yoy': 1.8, 'quarter': 'Q4'},
    '2018-04-24': {'headline_yoy': 1.9, 'trimmed_mean_yoy': 1.9, 'prev_headline_yoy': 1.9, 'prev_trimmed_yoy': 1.9, 'quarter': 'Q1'},
    '2018-07-25': {'headline_yoy': 2.1, 'trimmed_mean_yoy': 1.9, 'prev_headline_yoy': 1.9, 'prev_trimmed_yoy': 1.9, 'quarter': 'Q2'},
    '2018-10-31': {'headline_yoy': 1.9, 'trimmed_mean_yoy': 1.8, 'prev_headline_yoy': 2.1, 'prev_trimmed_yoy': 1.9, 'quarter': 'Q3'},
    # 2019
    '2019-01-30': {'headline_yoy': 1.8, 'trimmed_mean_yoy': 1.8, 'prev_headline_yoy': 1.9, 'prev_trimmed_yoy': 1.8, 'quarter': 'Q4'},
    '2019-04-24': {'headline_yoy': 1.3, 'trimmed_mean_yoy': 1.6, 'prev_headline_yoy': 1.8, 'prev_trimmed_yoy': 1.8, 'quarter': 'Q1'},
    '2019-07-31': {'headline_yoy': 1.6, 'trimmed_mean_yoy': 1.6, 'prev_headline_yoy': 1.3, 'prev_trimmed_yoy': 1.6, 'quarter': 'Q2'},
    '2019-10-30': {'headline_yoy': 1.7, 'trimmed_mean_yoy': 1.6, 'prev_headline_yoy': 1.6, 'prev_trimmed_yoy': 1.6, 'quarter': 'Q3'},
    # 2020
    '2020-01-29': {'headline_yoy': 1.8, 'trimmed_mean_yoy': 1.6, 'prev_headline_yoy': 1.7, 'prev_trimmed_yoy': 1.6, 'quarter': 'Q4'},
    '2020-04-29': {'headline_yoy': 2.2, 'trimmed_mean_yoy': 1.8, 'prev_headline_yoy': 1.8, 'prev_trimmed_yoy': 1.6, 'quarter': 'Q1'},
    '2020-07-29': {'headline_yoy': -0.3, 'trimmed_mean_yoy': 1.2, 'prev_headline_yoy': 2.2, 'prev_trimmed_yoy': 1.8, 'quarter': 'Q2'},
    '2020-10-28': {'headline_yoy': 0.7, 'trimmed_mean_yoy': 1.2, 'prev_headline_yoy': -0.3, 'prev_trimmed_yoy': 1.2, 'quarter': 'Q3'},
    # 2021
    '2021-01-27': {'headline_yoy': 0.9, 'trimmed_mean_yoy': 1.2, 'prev_headline_yoy': 0.7, 'prev_trimmed_yoy': 1.2, 'quarter': 'Q4'},
    '2021-04-28': {'headline_yoy': 1.1, 'trimmed_mean_yoy': 1.1, 'prev_headline_yoy': 0.9, 'prev_trimmed_yoy': 1.2, 'quarter': 'Q1'},
    '2021-07-28': {'headline_yoy': 3.8, 'trimmed_mean_yoy': 1.6, 'prev_headline_yoy': 1.1, 'prev_trimmed_yoy': 1.1, 'quarter': 'Q2'},
    '2021-10-27': {'headline_yoy': 3.0, 'trimmed_mean_yoy': 2.1, 'prev_headline_yoy': 3.8, 'prev_trimmed_yoy': 1.6, 'quarter': 'Q3'},
    # 2022
    '2022-01-25': {'headline_yoy': 3.5, 'trimmed_mean_yoy': 2.6, 'prev_headline_yoy': 3.0, 'prev_trimmed_yoy': 2.1, 'quarter': 'Q4'},
    '2022-04-27': {'headline_yoy': 5.1, 'trimmed_mean_yoy': 3.7, 'prev_headline_yoy': 3.5, 'prev_trimmed_yoy': 2.6, 'quarter': 'Q1'},
    '2022-07-27': {'headline_yoy': 6.1, 'trimmed_mean_yoy': 4.9, 'prev_headline_yoy': 5.1, 'prev_trimmed_yoy': 3.7, 'quarter': 'Q2'},
    '2022-10-26': {'headline_yoy': 7.3, 'trimmed_mean_yoy': 6.1, 'prev_headline_yoy': 6.1, 'prev_trimmed_yoy': 4.9, 'quarter': 'Q3'},
    # 2023
    '2023-01-25': {'headline_yoy': 7.8, 'trimmed_mean_yoy': 6.9, 'prev_headline_yoy': 7.3, 'prev_trimmed_yoy': 6.1, 'quarter': 'Q4'},
    '2023-04-26': {'headline_yoy': 7.0, 'trimmed_mean_yoy': 6.6, 'prev_headline_yoy': 7.8, 'prev_trimmed_yoy': 6.9, 'quarter': 'Q1'},
    '2023-07-26': {'headline_yoy': 6.0, 'trimmed_mean_yoy': 5.9, 'prev_headline_yoy': 7.0, 'prev_trimmed_yoy': 6.6, 'quarter': 'Q2'},
    '2023-10-25': {'headline_yoy': 5.4, 'trimmed_mean_yoy': 5.2, 'prev_headline_yoy': 6.0, 'prev_trimmed_yoy': 5.9, 'quarter': 'Q3'},
    # 2024
    '2024-01-31': {'headline_yoy': 4.1, 'trimmed_mean_yoy': 4.2, 'prev_headline_yoy': 5.4, 'prev_trimmed_yoy': 5.2, 'quarter': 'Q4'},
    '2024-04-24': {'headline_yoy': 3.6, 'trimmed_mean_yoy': 4.0, 'prev_headline_yoy': 4.1, 'prev_trimmed_yoy': 4.2, 'quarter': 'Q1'},
    '2024-07-31': {'headline_yoy': 3.8, 'trimmed_mean_yoy': 3.9, 'prev_headline_yoy': 3.6, 'prev_trimmed_yoy': 4.0, 'quarter': 'Q2'},
    '2024-10-30': {'headline_yoy': 2.8, 'trimmed_mean_yoy': 3.5, 'prev_headline_yoy': 3.8, 'prev_trimmed_yoy': 3.9, 'quarter': 'Q3'},
    # 2025
    '2025-01-29': {'headline_yoy': 2.4, 'trimmed_mean_yoy': 3.2, 'prev_headline_yoy': 2.8, 'prev_trimmed_yoy': 3.5, 'quarter': 'Q4'},
    '2025-04-30': {'headline_yoy': 2.4, 'trimmed_mean_yoy': 2.9, 'prev_headline_yoy': 2.4, 'prev_trimmed_yoy': 3.2, 'quarter': 'Q1'},
    '2025-07-30': {'headline_yoy': 2.1, 'trimmed_mean_yoy': 2.7, 'prev_headline_yoy': 2.4, 'prev_trimmed_yoy': 2.9, 'quarter': 'Q2'},
    '2025-10-29': {'headline_yoy': 2.3, 'trimmed_mean_yoy': 2.5, 'prev_headline_yoy': 2.1, 'prev_trimmed_yoy': 2.7, 'quarter': 'Q3'},
    # 2026
    '2026-01-28': {'headline_yoy': 2.5, 'trimmed_mean_yoy': 2.5, 'prev_headline_yoy': 2.3, 'prev_trimmed_yoy': 2.5, 'quarter': 'Q4'},
    '2026-04-29': {'headline_yoy': 2.3, 'trimmed_mean_yoy': 2.4, 'prev_headline_yoy': 2.5, 'prev_trimmed_yoy': 2.5, 'quarter': 'Q1'},
}

# ═══════════════════════════════════════════════════════════════
# SESSION DEFINITIONS (UTC offsets from release time 00:30 UTC)
# Australia CPI = early Asia session release
# ═══════════════════════════════════════════════════════════════

SESSIONS = {
    'Release Spike':      {'start': 0, 'end': 0.25},       # 00:30-00:45 UTC
    'Sydney Open':        {'start': 0.25, 'end': 1.5},      # 00:45-02:00 UTC
    'Tokyo Open':         {'start': 1.5, 'end': 2.5},       # 02:00-03:00 UTC
    'Asia Mid':           {'start': 2.5, 'end': 4.5},       # 03:00-05:00 UTC
    'Asia Afternoon':     {'start': 4.5, 'end': 6.5},       # 05:00-07:00 UTC
    'Tokyo Close':        {'start': 6.5, 'end': 7.5},       # 07:00-08:00 UTC
    'Pre-London':         {'start': 7.5, 'end': 8.5},       # 08:00-09:00 UTC
    'Frankfurt Open':     {'start': 8.5, 'end': 9.5},       # 09:00-10:00 UTC
    'London Open':        {'start': 9.5, 'end': 10.5},      # 10:00-11:00 UTC
    'London Morning':     {'start': 10.5, 'end': 12.5},     # 11:00-13:00 UTC
    'London Midday':      {'start': 12.5, 'end': 14.5},     # 13:00-15:00 UTC
    'NY Pre-Open':        {'start': 14.5, 'end': 16.0},     # 15:00-16:30 UTC
    'NY Open':            {'start': 16.0, 'end': 17.5},     # 16:30-18:00 UTC
    'London-NY Overlap':  {'start': 17.5, 'end': 18.5},     # 18:00-19:00 UTC
    'NY AM':              {'start': 18.5, 'end': 21.5},     # 19:00-22:00 UTC
    'NY Lunch':           {'start': 21.5, 'end': 23.0},     # 22:00-23:30 UTC
    'NY PM':              {'start': 23.0, 'end': 24.0},     # 23:30-00:30 UTC
    'NY Close':           {'start': 24.0, 'end': 25.5},     # 00:30-02:00 UTC
    'Pre-Asia D2':        {'start': 25.5, 'end': 27.5},     # 02:00-04:00 UTC
    'Sydney Open D2':     {'start': 27.5, 'end': 28.5},     # 04:00-05:00 UTC
    'Tokyo Open D2':      {'start': 28.5, 'end': 29.5},     # 05:00-06:00 UTC
    'Asia Mid D2':        {'start': 29.5, 'end': 31.5},     # 06:00-08:00 UTC
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


def classify_signal(trimmed_yoy, prev_trimmed_yoy, headline_yoy, prev_headline_yoy):
    """Classify CPI surprise based on trimmed mean (RBA preferred)."""
    if trimmed_yoy is None or prev_trimmed_yoy is None:
        return 'NO_DATA'
    change = trimmed_yoy - prev_trimmed_yoy
    headline_change = headline_yoy - prev_headline_yoy if headline_yoy and prev_headline_yoy else 0

    if trimmed_yoy >= 5.0:
        return 'HOT'
    elif trimmed_yoy >= 3.5:
        if change > 0.3:
            return 'HOT_RISING'
        elif change < -0.3:
            return 'HOT_FALLING'
        return 'HOT_STABLE'
    elif trimmed_yoy >= 2.5:
        if change > 0.3:
            return 'WARM_RISING'
        elif change < -0.3:
            return 'WARM_FALLING'
        return 'WARM_STABLE'
    elif trimmed_yoy >= 2.0:
        return 'TARGET'
    elif trimmed_yoy >= 1.5:
        return 'COOL'
    else:
        return 'COLD'


def classify_level(trimmed_yoy):
    """Classify absolute trimmed mean level."""
    if trimmed_yoy is None:
        return 'NO_DATA'
    if trimmed_yoy >= 5.0:
        return 'RUNAWAY'
    elif trimmed_yoy >= 3.5:
        return 'HOT'
    elif trimmed_yoy >= 2.5:
        return 'WARM'
    elif trimmed_yoy >= 2.0:
        return 'TARGET'
    elif trimmed_yoy >= 1.5:
        return 'COOL'
    else:
        return 'COLD'


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
        release_ts = pd.Timestamp(date_str + 'T00:30:00')
        mask = df['timestamp'] <= release_ts
        if mask.sum() == 0:
            continue
        idx = mask.sum() - 1
        row = df.iloc[idx]

        signal = classify_signal(data.get('trimmed_mean_yoy'), data.get('prev_trimmed_yoy'),
                                 data.get('headline_yoy'), data.get('prev_headline_yoy'))
        level = classify_level(data.get('trimmed_mean_yoy'))
        wyckoff = classify_wyckoff(row['close'], row['ema21'], row['ema55'], row['atr_pct'])
        vol = classify_vol(row['atr'], row['atr_ma'])
        returns = compute_session_returns(df, release_ts)

        all_results.append({
            'date': date_str, 'headline_yoy': data.get('headline_yoy'),
            'trimmed_mean_yoy': data.get('trimmed_mean_yoy'),
            'prev_headline_yoy': data.get('prev_headline_yoy'),
            'prev_trimmed_yoy': data.get('prev_trimmed_yoy'),
            'quarter': data.get('quarter'),
            'signal': signal, 'level': level,
            'wyckoff': wyckoff, 'vol': vol, 'returns': returns,
        })
    return all_results


def analyze_results(results):
    print("\n" + "="*80)
    print("PROMPT A: AUSTRALIA QUARTERLY CPI BACKTEST RESULTS")
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

    # Cross-tabulation
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

    out_path = os.path.join(os.path.dirname(__file__), 'backtest_au_cpi_results.json')
    json_results = []
    for r in results:
        jr = {k: v for k, v in r.items() if k != 'returns'}
        jr['returns'] = {sk: sv for sk, sv in r['returns'].items()}
        json_results.append(jr)
    with open(out_path, 'w') as f:
        json.dump(json_results, f, indent=2, default=str)
    print(f"\nResults saved to {out_path}")

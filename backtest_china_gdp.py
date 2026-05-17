#!/usr/bin/env python3
"""
Prompt A + B: Backtest China Quarterly GDP (2018-today) using ETH/USDT 15m data.

NBS China GDP released at 02:00 UTC (10:00 MYT) ~15-18 days after quarter end.
Same day also releases Retail Sales + Industrial Output.

Session itinerary (per user #38):
  Asia Morning (10:00 MYT) → Retail & Industrial Output → Global Demand Transmission

Thesis:
  China GDP = anchor for global demand. Significant miss → PBoC stimulus
  expectations → "V-bottom" liquidity play. Weak GDP → European manufacturing
  drag + commodity sell-off → risk-off headwinds for ETH in EU morning.
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
# CHINA QUARTERLY GDP RELEASE DATES (02:00 UTC = 10:00 MYT)
# Format: {date: {'gdp_yoy': float, 'consensus_yoy': float, 'prev_yoy': float,
#                 'retail_yoy': float, 'industrial_yoy': float, 'quarter': str}}
# ═══════════════════════════════════════════════════════════════

RELEASES = {
    # 2018
    '2018-01-18': {'gdp_yoy': 6.8, 'consensus_yoy': 6.7, 'prev_yoy': 6.8, 'retail_yoy': 9.4, 'industrial_yoy': 6.2, 'quarter': 'Q4'},
    '2018-04-17': {'gdp_yoy': 6.8, 'consensus_yoy': 6.7, 'prev_yoy': 6.8, 'retail_yoy': 10.1, 'industrial_yoy': 6.0, 'quarter': 'Q1'},
    '2018-07-16': {'gdp_yoy': 6.7, 'consensus_yoy': 6.7, 'prev_yoy': 6.8, 'retail_yoy': 9.0, 'industrial_yoy': 6.0, 'quarter': 'Q2'},
    '2018-10-19': {'gdp_yoy': 6.5, 'consensus_yoy': 6.6, 'prev_yoy': 6.7, 'retail_yoy': 9.2, 'industrial_yoy': 5.8, 'quarter': 'Q3'},
    # 2019
    '2019-01-21': {'gdp_yoy': 6.4, 'consensus_yoy': 6.4, 'prev_yoy': 6.5, 'retail_yoy': 8.2, 'industrial_yoy': 5.7, 'quarter': 'Q4'},
    '2019-04-17': {'gdp_yoy': 6.4, 'consensus_yoy': 6.3, 'prev_yoy': 6.4, 'retail_yoy': 8.7, 'industrial_yoy': 8.5, 'quarter': 'Q1'},
    '2019-07-15': {'gdp_yoy': 6.2, 'consensus_yoy': 6.2, 'prev_yoy': 6.4, 'retail_yoy': 9.8, 'industrial_yoy': 6.3, 'quarter': 'Q2'},
    '2019-10-18': {'gdp_yoy': 6.0, 'consensus_yoy': 6.1, 'prev_yoy': 6.2, 'retail_yoy': 7.8, 'industrial_yoy': 5.8, 'quarter': 'Q3'},
    # 2020
    '2020-01-17': {'gdp_yoy': 6.0, 'consensus_yoy': 6.0, 'prev_yoy': 6.0, 'retail_yoy': 8.0, 'industrial_yoy': 6.9, 'quarter': 'Q4'},
    '2020-04-17': {'gdp_yoy': -6.8, 'consensus_yoy': -6.5, 'prev_yoy': 6.0, 'retail_yoy': -15.8, 'industrial_yoy': -1.1, 'quarter': 'Q1'},
    '2020-07-16': {'gdp_yoy': 3.2, 'consensus_yoy': 2.5, 'prev_yoy': -6.8, 'retail_yoy': -1.1, 'industrial_yoy': 4.8, 'quarter': 'Q2'},
    '2020-10-19': {'gdp_yoy': 4.9, 'consensus_yoy': 5.5, 'prev_yoy': 3.2, 'retail_yoy': 3.3, 'industrial_yoy': 6.9, 'quarter': 'Q3'},
    # 2021
    '2021-01-18': {'gdp_yoy': 6.5, 'consensus_yoy': 6.5, 'prev_yoy': 4.9, 'retail_yoy': 4.6, 'industrial_yoy': 7.3, 'quarter': 'Q4'},
    '2021-04-16': {'gdp_yoy': 18.3, 'consensus_yoy': 18.5, 'prev_yoy': 6.5, 'retail_yoy': 34.2, 'industrial_yoy': 14.1, 'quarter': 'Q1'},
    '2021-07-15': {'gdp_yoy': 7.9, 'consensus_yoy': 8.0, 'prev_yoy': 18.3, 'retail_yoy': 12.1, 'industrial_yoy': 8.3, 'quarter': 'Q2'},
    '2021-10-18': {'gdp_yoy': 4.9, 'consensus_yoy': 5.0, 'prev_yoy': 7.9, 'retail_yoy': 4.4, 'industrial_yoy': 3.1, 'quarter': 'Q3'},
    # 2022
    '2022-01-17': {'gdp_yoy': 4.0, 'consensus_yoy': 4.0, 'prev_yoy': 4.9, 'retail_yoy': 1.7, 'industrial_yoy': 4.3, 'quarter': 'Q4'},
    '2022-04-18': {'gdp_yoy': 4.8, 'consensus_yoy': 4.4, 'prev_yoy': 4.0, 'retail_yoy': -3.5, 'industrial_yoy': 5.0, 'quarter': 'Q1'},
    '2022-07-15': {'gdp_yoy': 0.4, 'consensus_yoy': 1.0, 'prev_yoy': 4.8, 'retail_yoy': 3.1, 'industrial_yoy': 3.9, 'quarter': 'Q2'},
    '2022-10-24': {'gdp_yoy': 3.9, 'consensus_yoy': 3.4, 'prev_yoy': 0.4, 'retail_yoy': 2.5, 'industrial_yoy': 6.3, 'quarter': 'Q3'},
    # 2023
    '2023-01-17': {'gdp_yoy': 3.0, 'consensus_yoy': 2.8, 'prev_yoy': 3.9, 'retail_yoy': -1.8, 'industrial_yoy': 1.3, 'quarter': 'Q4'},
    '2023-04-18': {'gdp_yoy': 4.5, 'consensus_yoy': 4.0, 'prev_yoy': 3.0, 'retail_yoy': 10.6, 'industrial_yoy': 3.9, 'quarter': 'Q1'},
    '2023-07-17': {'gdp_yoy': 6.3, 'consensus_yoy': 7.1, 'prev_yoy': 4.5, 'retail_yoy': 3.1, 'industrial_yoy': 4.4, 'quarter': 'Q2'},
    '2023-10-18': {'gdp_yoy': 4.9, 'consensus_yoy': 4.5, 'prev_yoy': 6.3, 'retail_yoy': 5.5, 'industrial_yoy': 4.5, 'quarter': 'Q3'},
    # 2024
    '2024-01-17': {'gdp_yoy': 5.2, 'consensus_yoy': 5.3, 'prev_yoy': 4.9, 'retail_yoy': 7.4, 'industrial_yoy': 6.8, 'quarter': 'Q4'},
    '2024-04-16': {'gdp_yoy': 5.3, 'consensus_yoy': 5.0, 'prev_yoy': 5.2, 'retail_yoy': 3.1, 'industrial_yoy': 4.5, 'quarter': 'Q1'},
    '2024-07-15': {'gdp_yoy': 4.7, 'consensus_yoy': 5.1, 'prev_yoy': 5.3, 'retail_yoy': 2.0, 'industrial_yoy': 5.3, 'quarter': 'Q2'},
    '2024-10-18': {'gdp_yoy': 4.6, 'consensus_yoy': 4.5, 'prev_yoy': 4.7, 'retail_yoy': 3.2, 'industrial_yoy': 5.4, 'quarter': 'Q3'},
    # 2025
    '2025-01-17': {'gdp_yoy': 5.4, 'consensus_yoy': 5.0, 'prev_yoy': 4.6, 'retail_yoy': 3.7, 'industrial_yoy': 6.2, 'quarter': 'Q4'},
    '2025-04-16': {'gdp_yoy': 5.4, 'consensus_yoy': 5.2, 'prev_yoy': 5.4, 'retail_yoy': 5.9, 'industrial_yoy': 7.7, 'quarter': 'Q1'},
    '2025-07-15': {'gdp_yoy': 5.2, 'consensus_yoy': 5.1, 'prev_yoy': 5.4, 'retail_yoy': 4.8, 'industrial_yoy': 6.8, 'quarter': 'Q2'},
    '2025-10-20': {'gdp_yoy': 4.8, 'consensus_yoy': 4.9, 'prev_yoy': 5.2, 'retail_yoy': 3.0, 'industrial_yoy': 5.2, 'quarter': 'Q3'},
    # 2026
    '2026-01-19': {'gdp_yoy': 5.0, 'consensus_yoy': 4.9, 'prev_yoy': 4.8, 'retail_yoy': 3.5, 'industrial_yoy': 5.8, 'quarter': 'Q4'},
    '2026-04-16': {'gdp_yoy': 5.4, 'consensus_yoy': 5.1, 'prev_yoy': 5.0, 'retail_yoy': 5.5, 'industrial_yoy': 7.2, 'quarter': 'Q1'},
}

# ═══════════════════════════════════════════════════════════════
# SESSION DEFINITIONS (UTC offsets from release time 02:00 UTC)
# ═══════════════════════════════════════════════════════════════

SESSIONS = {
    'Release Spike':      {'start': 0, 'end': 0.25},
    'Tokyo Open':         {'start': 0.25, 'end': 1.0},
    'Asia Mid':           {'start': 1.0, 'end': 3.0},
    'Asia Afternoon':     {'start': 3.0, 'end': 5.0},
    'Tokyo Close':        {'start': 5.0, 'end': 6.0},
    'Pre-London':         {'start': 6.0, 'end': 7.0},
    'Frankfurt Open':     {'start': 7.0, 'end': 8.0},
    'London Open':        {'start': 8.0, 'end': 9.0},
    'London Morning':     {'start': 9.0, 'end': 11.0},
    'London Midday':      {'start': 11.0, 'end': 13.0},
    'NY Pre-Open':        {'start': 13.0, 'end': 14.5},
    'NY Open':            {'start': 14.5, 'end': 16.0},
    'London-NY Overlap':  {'start': 16.0, 'end': 17.0},
    'NY AM':              {'start': 17.0, 'end': 20.0},
    'NY Lunch':           {'start': 20.0, 'end': 21.5},
    'NY PM':              {'start': 21.5, 'end': 23.5},
    'NY Close':           {'start': 23.5, 'end': 25.0},
    'Pre-Asia D2':        {'start': 25.0, 'end': 27.0},
    'Sydney Open D2':     {'start': 27.0, 'end': 28.0},
    'Tokyo Open D2':      {'start': 28.0, 'end': 29.0},
    'Asia Mid D2':        {'start': 29.0, 'end': 31.0},
    '24h_aggregate':      {'start': 0, 'end': 24.0},
}


def load_eth_data(csv_path):
    df = pd.read_csv(csv_path)
    for ts_col in ['timestamp', 'open_time', 'date', 'datetime', 'time']:
        if ts_col in df.columns: break
    else: ts_col = df.columns[0]
    df['timestamp'] = pd.to_datetime(df[ts_col])
    df = df.sort_values('timestamp').reset_index(drop=True)
    for col in ['open', 'high', 'low', 'close']:
        for c in [col, col.capitalize(), col.upper()]:
            if c in df.columns: df[col] = pd.to_numeric(df[c], errors='coerce'); break
    if 'volume' in df.columns: df['volume'] = pd.to_numeric(df['volume'], errors='coerce')
    return df


def get_price_at(df, ts):
    mask = df['timestamp'] <= ts
    if mask.sum() == 0: return None
    return df.loc[mask].iloc[-1]['close']


def classify_signal(gdp_yoy, consensus_yoy, prev_yoy):
    if gdp_yoy is None or consensus_yoy is None: return 'NO_DATA'
    surprise = gdp_yoy - consensus_yoy
    if surprise >= 0.5: return 'STRONG_BEAT'
    elif surprise >= 0.1: return 'BEAT'
    elif surprise >= -0.1: return 'INLINE'
    elif surprise >= -0.5: return 'MISS'
    else: return 'BIG_MISS'


def classify_level(gdp_yoy):
    if gdp_yoy is None: return 'NO_DATA'
    if gdp_yoy >= 6.0: return 'STRONG'
    elif gdp_yoy >= 5.0: return 'MODERATE'
    elif gdp_yoy >= 4.0: return 'SLOWING'
    elif gdp_yoy >= 2.0: return 'WEAK'
    elif gdp_yoy >= 0.0: return 'STAGNANT'
    else: return 'CONTRACTION'


def classify_wyckoff(price, ema21, ema55):
    if price is None or ema21 is None or ema55 is None: return 'UNKNOWN'
    if ema21 > ema55:
        return 'MARKUP' if price > ema21 else 'DISTRIBUTION'
    elif ema21 < ema55:
        return 'MARKDOWN' if price < ema21 else 'ACCUMULATION'
    return 'RANGE'


def classify_vol(atr, atr_ma):
    if atr is None or atr_ma is None or atr_ma == 0: return 'UNKNOWN'
    ratio = atr / atr_ma
    if ratio > 1.5: return 'CRISIS'
    elif ratio > 1.2: return 'TREND'
    elif ratio > 0.9: return 'NEUTRAL'
    elif ratio > 0.6: return 'COMPRESSING'
    return 'LOW_VOL'


def compute_session_returns(df, release_ts):
    results = {}
    price_release = get_price_at(df, release_ts)
    if price_release is None: return results
    for name, times in SESSIONS.items():
        start_ts = release_ts + timedelta(hours=times['start'])
        end_ts = release_ts + timedelta(hours=times['end'])
        p_start = get_price_at(df, start_ts)
        p_end = get_price_at(df, end_ts)
        if p_start and p_end and p_start > 0:
            results[name] = {'return': (p_end - p_start) / p_start * 100}
        else:
            results[name] = {'return': 0}
    p_24h = get_price_at(df, release_ts + timedelta(hours=24))
    if price_release and p_24h and price_release > 0:
        results['24h_aggregate'] = {'return': (p_24h - price_release) / price_release * 100}
    return results


def compute_ema(series, period): return series.ewm(span=period, adjust=False).mean()


def run_backtest(csv_path):
    df = load_eth_data(csv_path)
    df['ema21'] = compute_ema(df['close'], 21)
    df['ema55'] = compute_ema(df['close'], 55)
    df['atr'] = (df['high'] - df['low']).rolling(14).mean()
    df['atr_ma'] = df['atr'].rolling(96).mean()

    all_results = []
    for date_str, data in RELEASES.items():
        release_ts = pd.Timestamp(date_str + 'T02:00:00')
        mask = df['timestamp'] <= release_ts
        if mask.sum() == 0: continue
        row = df.iloc[mask.sum() - 1]
        signal = classify_signal(data['gdp_yoy'], data.get('consensus_yoy'), data.get('prev_yoy'))
        level = classify_level(data['gdp_yoy'])
        wyckoff = classify_wyckoff(row['close'], row['ema21'], row['ema55'])
        vol = classify_vol(row['atr'], row['atr_ma'])
        returns = compute_session_returns(df, release_ts)
        all_results.append({
            'date': date_str, 'gdp_yoy': data['gdp_yoy'], 'consensus_yoy': data.get('consensus_yoy'),
            'prev_yoy': data.get('prev_yoy'), 'retail_yoy': data.get('retail_yoy'),
            'industrial_yoy': data.get('industrial_yoy'), 'quarter': data.get('quarter'),
            'signal': signal, 'level': level, 'wyckoff': wyckoff, 'vol': vol, 'returns': returns,
        })
    return all_results


def analyze_results(results):
    print("\n" + "="*80)
    print("PROMPT A: CHINA QUARTERLY GDP BACKTEST RESULTS")
    print("="*80)
    agg = [r['returns'].get('24h_aggregate', {}).get('return', 0) for r in results]
    agg = [r for r in agg if r != 0]
    print(f"\nTotal: {len(results)} | 24h: {np.mean(agg):.3f}% avg, {sum(1 for r in agg if r > 0)/len(agg)*100:.1f}% win, n={len(agg)}")
    t, p = stats.ttest_1samp(agg, 0)
    print(f"t-test vs 0: t={t:.3f}, p={p:.4f} {'✅' if p < 0.05 else '❌'}")

    for label, key in [('Signal', 'signal'), ('Level', 'level')]:
        print(f"\n--- {label} Breakdown ---")
        groups = {}
        for r in results: groups.setdefault(r[key], []).append(r['returns'].get('24h_aggregate', {}).get('return', 0))
        for k, rets in sorted(groups.items()):
            print(f"  {k:20s}: {np.mean(rets):+.3f}% avg, {sum(1 for r in rets if r > 0)/len(rets)*100:.1f}% win, n={len(rets)}")

    print(f"\n--- Cross-Tabulation: Wyckoff × Vol × Signal (n≥3) ---")
    cross = {}
    edges = []
    for r in results:
        key = (r['wyckoff'], r['vol'], r['signal'])
        cross.setdefault(key, []).append(r['returns'].get('24h_aggregate', {}).get('return', 0))
    for key, rets in sorted(cross.items()):
        if len(rets) >= 3:
            avg, wr = np.mean(rets), sum(1 for r in rets if r > 0)/len(rets)*100
            w, v, s = key
            print(f"  {w:15s} × {v:12s} × {s:16s}: {avg:+.3f}% avg, {wr:.0f}% win, n={len(rets)}")
            if abs(avg) >= 0.5: edges.append((key, avg, wr, len(rets)))
    print(f"\n--- Edges (|avg|≥0.5%, n≥3) ---")
    for key, avg, wr, n in sorted(edges, key=lambda x: abs(x[1]), reverse=True):
        w, v, s = key; d = 'LONG' if avg > 0 else 'SHORT'
        print(f"  {w} × {v} × {s} → {d}: {avg:+.3f}%, {wr:.0f}% win, n={n}")

    miss = [r['returns'].get('24h_aggregate', {}).get('return', 0) for r in results if r['signal'] in ('MISS', 'BIG_MISS')]
    beat = [r['returns'].get('24h_aggregate', {}).get('return', 0) for r in results if r['signal'] in ('STRONG_BEAT', 'BEAT')]
    if miss and beat:
        t2, p2 = stats.ttest_ind(miss, beat)
        print(f"\nMISS vs BEAT: miss={np.mean(miss):.3f}% (n={len(miss)}), beat={np.mean(beat):.3f}% (n={len(beat)}), t={t2:.3f}, p={p2:.4f} {'✅' if p2 < 0.05 else '❌'}")
    return edges


def analyze_transmission(results):
    print("\n" + "="*80)
    print("PROMPT B: SESSION TRANSMISSION CHAIN ANALYSIS")
    print("="*80)
    names = list(SESSIONS.keys())
    print(f"\n--- Direction Persistence (n≥5) ---")
    trans = {}
    for r in results:
        prev_dir, prev_name = None, None
        for name in names:
            ret = r['returns'].get(name, {}).get('return', 0)
            curr = 1 if ret > 0 else (-1 if ret < 0 else 0)
            if prev_dir is not None and prev_dir != 0 and curr != 0:
                key = f"{prev_name} → {name}"
                trans.setdefault(key, {'same': 0, 'total': 0})
                trans[key]['total'] += 1
                if prev_dir == curr: trans[key]['same'] += 1
            prev_dir, prev_name = curr, name
    for key, d in sorted(trans.items()):
        if d['total'] >= 5:
            pct = d['same']/d['total']*100
            s = '✅' if pct > 65 else ('⚠️' if pct > 55 else '❌')
            print(f"  {key:45s}: {pct:.1f}% (n={d['total']}) {s}")
    print(f"\n--- Average Return by Session ---")
    for name in names:
        rets = [r['returns'].get(name, {}).get('return', 0) for r in results]
        if name != '24h_aggregate': print(f"  {name:25s}: {np.mean(rets):+.3f}% avg, {sum(1 for r in rets if r > 0)/len(rets)*100:.1f}% win")


if __name__ == '__main__':
    csv_path = os.path.join(os.path.dirname(__file__), 'eth_15m_merged.csv')
    if not os.path.exists(csv_path): csv_path = 'eth_15m_merged.csv'
    print("Loading ETH 15m data...")
    results = run_backtest(csv_path)
    edges = analyze_results(results)
    analyze_transmission(results)
    out_path = os.path.join(os.path.dirname(__file__), 'backtest_china_gdp_results.json')
    with open(out_path, 'w') as f:
        json.dump([{k: v for k, v in r.items() if k != 'returns'} | {'returns': r['returns']} for r in results], f, indent=2, default=str)
    print(f"\nResults saved to {out_path}")

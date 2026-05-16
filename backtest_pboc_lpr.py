#!/usr/bin/env python3
"""
Prompt A + B: Backtest PBoC LPR Decision (2018-today) using ETH/USDT 15m data.

PBoC publishes LPR (Loan Prime Rate) on the 20th of each month at 09:15 MYT = 01:15 UTC.
1-year LPR and 5-year LPR are both published simultaneously.

Session itinerary (per user's thesis #18):
  Asia Morning (01:15 UTC) → Daily Forex Fixing → Australia Response → Global Crypto Desks

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

# ═══════════════════════════════════════════════════════════════
# PBOC LPR RELEASE DATES (09:15 MYT = 01:15 UTC)
# Published on 20th of each month (or next business day)
# Format: {date: {'lpr_1y': float, 'lpr_5y': float}}
# lpr_1y = 1-year Loan Prime Rate (%)
# lpr_5y = 5-year Loan Prime Rate (%)
# ═══════════════════════════════════════════════════════════════

RELEASES = {
    # 2019 (LPR reform started Aug 2019)
    '2019-08-20': {'lpr_1y': 4.25, 'lpr_5y': 4.85},
    '2019-09-20': {'lpr_1y': 4.20, 'lpr_5y': 4.85},
    '2019-10-21': {'lpr_1y': 4.20, 'lpr_5y': 4.85},
    '2019-11-20': {'lpr_1y': 4.15, 'lpr_5y': 4.80},
    '2019-12-20': {'lpr_1y': 4.15, 'lpr_5y': 4.80},
    # 2020
    '2020-01-20': {'lpr_1y': 4.15, 'lpr_5y': 4.80},
    '2020-02-20': {'lpr_1y': 4.05, 'lpr_5y': 4.75},
    '2020-03-20': {'lpr_1y': 4.05, 'lpr_5y': 4.75},
    '2020-04-20': {'lpr_1y': 3.85, 'lpr_5y': 4.65},
    '2020-05-20': {'lpr_1y': 3.85, 'lpr_5y': 4.65},
    '2020-06-22': {'lpr_1y': 3.85, 'lpr_5y': 4.65},
    '2020-07-20': {'lpr_1y': 3.85, 'lpr_5y': 4.65},
    '2020-08-20': {'lpr_1y': 3.85, 'lpr_5y': 4.65},
    '2020-09-21': {'lpr_1y': 3.85, 'lpr_5y': 4.65},
    '2020-10-20': {'lpr_1y': 3.85, 'lpr_5y': 4.65},
    '2020-11-20': {'lpr_1y': 3.85, 'lpr_5y': 4.65},
    '2020-12-21': {'lpr_1y': 3.85, 'lpr_5y': 4.65},
    # 2021
    '2021-01-20': {'lpr_1y': 3.85, 'lpr_5y': 4.65},
    '2021-02-22': {'lpr_1y': 3.85, 'lpr_5y': 4.65},
    '2021-03-22': {'lpr_1y': 3.85, 'lpr_5y': 4.65},
    '2021-04-20': {'lpr_1y': 3.85, 'lpr_5y': 4.65},
    '2021-05-20': {'lpr_1y': 3.85, 'lpr_5y': 4.65},
    '2021-06-21': {'lpr_1y': 3.85, 'lpr_5y': 4.65},
    '2021-07-20': {'lpr_1y': 3.85, 'lpr_5y': 4.65},
    '2021-08-20': {'lpr_1y': 3.85, 'lpr_5y': 4.65},
    '2021-09-22': {'lpr_1y': 3.85, 'lpr_5y': 4.65},
    '2021-10-20': {'lpr_1y': 3.85, 'lpr_5y': 4.65},
    '2021-11-22': {'lpr_1y': 3.85, 'lpr_5y': 4.65},
    '2021-12-20': {'lpr_1y': 3.80, 'lpr_5y': 4.65},
    # 2022
    '2022-01-20': {'lpr_1y': 3.70, 'lpr_5y': 4.60},
    '2022-02-21': {'lpr_1y': 3.70, 'lpr_5y': 4.60},
    '2022-03-21': {'lpr_1y': 3.70, 'lpr_5y': 4.60},
    '2022-04-20': {'lpr_1y': 3.70, 'lpr_5y': 4.60},
    '2022-05-20': {'lpr_1y': 3.70, 'lpr_5y': 4.45},
    '2022-06-20': {'lpr_1y': 3.70, 'lpr_5y': 4.45},
    '2022-07-20': {'lpr_1y': 3.70, 'lpr_5y': 4.45},
    '2022-08-22': {'lpr_1y': 3.65, 'lpr_5y': 4.30},
    '2022-09-20': {'lpr_1y': 3.65, 'lpr_5y': 4.30},
    '2022-10-20': {'lpr_1y': 3.65, 'lpr_5y': 4.30},
    '2022-11-21': {'lpr_1y': 3.65, 'lpr_5y': 4.30},
    '2022-12-20': {'lpr_1y': 3.65, 'lpr_5y': 4.30},
    # 2023
    '2023-01-20': {'lpr_1y': 3.65, 'lpr_5y': 4.30},
    '2023-02-20': {'lpr_1y': 3.65, 'lpr_5y': 4.30},
    '2023-03-20': {'lpr_1y': 3.65, 'lpr_5y': 4.30},
    '2023-04-20': {'lpr_1y': 3.65, 'lpr_5y': 4.30},
    '2023-05-22': {'lpr_1y': 3.65, 'lpr_5y': 4.30},
    '2023-06-20': {'lpr_1y': 3.55, 'lpr_5y': 4.20},
    '2023-07-20': {'lpr_1y': 3.55, 'lpr_5y': 4.20},
    '2023-08-21': {'lpr_1y': 3.45, 'lpr_5y': 4.20},
    '2023-09-20': {'lpr_1y': 3.45, 'lpr_5y': 4.20},
    '2023-10-20': {'lpr_1y': 3.45, 'lpr_5y': 4.20},
    '2023-11-20': {'lpr_1y': 3.45, 'lpr_5y': 4.20},
    '2023-12-20': {'lpr_1y': 3.45, 'lpr_5y': 4.20},
    # 2024
    '2024-01-22': {'lpr_1y': 3.45, 'lpr_5y': 4.20},
    '2024-02-20': {'lpr_1y': 3.45, 'lpr_5y': 4.20},
    '2024-03-20': {'lpr_1y': 3.45, 'lpr_5y': 3.95},
    '2024-04-22': {'lpr_1y': 3.45, 'lpr_5y': 3.95},
    '2024-05-20': {'lpr_1y': 3.45, 'lpr_5y': 3.95},
    '2024-06-20': {'lpr_1y': 3.45, 'lpr_5y': 3.95},
    '2024-07-22': {'lpr_1y': 3.35, 'lpr_5y': 3.85},
    '2024-08-20': {'lpr_1y': 3.35, 'lpr_5y': 3.85},
    '2024-09-20': {'lpr_1y': 3.35, 'lpr_5y': 3.85},
    '2024-10-21': {'lpr_1y': 3.10, 'lpr_5y': 3.60},
    '2024-11-20': {'lpr_1y': 3.10, 'lpr_5y': 3.60},
    '2024-12-20': {'lpr_1y': 3.10, 'lpr_5y': 3.60},
    # 2025
    '2025-01-20': {'lpr_1y': 3.10, 'lpr_5y': 3.60},
    '2025-02-20': {'lpr_1y': 3.10, 'lpr_5y': 3.60},
    '2025-03-20': {'lpr_1y': 3.10, 'lpr_5y': 3.60},
    '2025-04-21': {'lpr_1y': 3.10, 'lpr_5y': 3.60},
    '2025-05-20': {'lpr_1y': 3.00, 'lpr_5y': 3.50},
    '2025-06-20': {'lpr_1y': 3.00, 'lpr_5y': 3.50},
    '2025-07-21': {'lpr_1y': 3.00, 'lpr_5y': 3.50},
    '2025-08-20': {'lpr_1y': 3.00, 'lpr_5y': 3.50},
    '2025-09-22': {'lpr_1y': 3.00, 'lpr_5y': 3.50},
    '2025-10-20': {'lpr_1y': 3.00, 'lpr_5y': 3.50},
    '2025-11-20': {'lpr_1y': 3.00, 'lpr_5y': 3.50},
    '2025-12-22': {'lpr_1y': 3.00, 'lpr_5y': 3.50},
    # 2026
    '2026-01-20': {'lpr_1y': 3.00, 'lpr_5y': 3.50},
    '2026-02-20': {'lpr_1y': 3.00, 'lpr_5y': 3.50},
    '2026-03-20': {'lpr_1y': 2.90, 'lpr_5y': 3.40},
    '2026-04-20': {'lpr_1y': 2.90, 'lpr_5y': 3.40},
    '2026-05-20': {'lpr_1y': 2.90, 'lpr_5y': 3.40},
}


# ═══════════════════════════════════════════════════════════════
# SESSION DEFINITIONS (PBoC LPR at 01:15 UTC)
# Asia-focused: release → forex fixing → Australia → global desks
# ═══════════════════════════════════════════════════════════════

SESSION_CHAIN = [
    # Pre-release (overnight)
    ('Pre-Asia', [21, 22, 23, 0]),       # 21:00-01:00 (prev NY close → pre-release)

    # Release window
    ('Release (1h)', [1, 2]),             # 01:15-02:15 (1h post-release)
    ('Sydney Open', [22, 23, 0, 1]),     # 22:00-01:15 (Sydney pre-release)
    ('Tokyo Open', [0, 1, 2]),           # 00:00-03:00 (Tokyo reacts)
    ('Asia Mid', [3, 4]),                # 03:00-05:00
    ('Asia Afternoon', [5, 6]),          # 05:00-07:00
    ('Tokyo Close', [6, 7]),             # 06:00-08:00
    ('Pre-London', [7, 8]),              # 07:00-09:00

    # Europe
    ('Frankfurt Open', [7, 8]),
    ('London Open', [8, 9]),
    ('London Morning', [9, 10, 11]),
    ('London Midday', [12, 13]),

    # Overlap (EU-US)
    ('NY Pre-Open', [12, 13]),
    ('NY Open', [13, 14]),
    ('London-NY Overlap', [14, 15]),

    # New York
    ('NY AM', [14, 15, 16]),
    ('NY Lunch', [17]),
    ('NY PM', [18, 19, 20]),

    # Post-release windows
    ('Release (4h)', [1, 2, 3, 4, 5]),   # 01:15-05:15 (4h post-release)
]


def load_eth_data(csv_path):
    df = pd.read_csv(csv_path)
    df['Open time'] = pd.to_datetime(df['Open time'])
    df = df.sort_values('Open time').reset_index(drop=True)
    return df


def get_bar_at(df, dt):
    mask = df['Open time'] <= dt
    if mask.sum() == 0:
        return None
    return df[mask].iloc[-1]


def get_bars_between(df, start, end):
    mask = (df['Open time'] >= start) & (df['Open time'] < end)
    return df[mask]


def get_return(df, start_dt, end_dt):
    bar_start = get_bar_at(df, start_dt)
    bars_end = get_bars_between(df, start_dt, end_dt)
    if bar_start is None or len(bars_end) == 0:
        return None
    start_price = float(bar_start['Close'])
    end_price = float(bars_end.iloc[-1]['Close'])
    return (end_price - start_price) / start_price * 100


def classify_wyckoff_phase(df, release_dt, lookback=96):
    start = release_dt - timedelta(hours=24)
    bars = get_bars_between(df, start, release_dt)
    if len(bars) < 20:
        return 'RANGE'
    price_series = bars['Close'].astype(float)
    high = price_series.max()
    low = price_series.min()
    current = price_series.iloc[-1]
    range_pct = (high - low) / low * 100
    pos_in_range = (current - low) / (high - low) if high != low else 0.5
    if range_pct < 1.5:
        return 'RANGE' if 0.3 < pos_in_range < 0.7 else (
            'ACCUMULATION' if pos_in_range < 0.3 else 'DISTRIBUTION')
    elif current > price_series.mean() and pos_in_range > 0.6:
        return 'MARKUP'
    elif current < price_series.mean() and pos_in_range < 0.4:
        return 'MARKDOWN'
    elif pos_in_range > 0.7:
        return 'DISTRIBUTION'
    elif pos_in_range < 0.3:
        return 'ACCUMULATION'
    else:
        return 'RANGE'


def classify_vol_regime(df, release_dt, lookback_hours=24):
    start = release_dt - timedelta(hours=lookback_hours)
    bars = get_bars_between(df, start, release_dt)
    if len(bars) < 10:
        return 'CHOP'
    closes = bars['Close'].astype(float).values
    ranges = (bars['High'].astype(float).values - bars['Low'].astype(float).values)
    avg_range_pct = np.mean(ranges / closes) * 100
    x = np.arange(len(closes))
    slope, _, r_value, _, _ = stats.linregress(x, closes)
    r_squared = r_value ** 2
    if r_squared > 0.7 and abs(slope) > 0.5:
        return 'TREND'
    elif avg_range_pct < 0.3:
        return 'SQUEEZE'
    elif avg_range_pct < 0.6:
        return 'LOW_VOL'
    else:
        return 'CHOP'


def classify_signal(lpr_1y, lpr_5y, prev_lpr_1y, prev_lpr_5y):
    """Classify LPR signal.
    Rate cut → stimulative → CNY weakens → capital into crypto (LONG)
    Rate hold → neutral
    Unexpected hike → hawkish → SHORT
    """
    cut_1y = prev_lpr_1y - lpr_1y if prev_lpr_1y else 0
    cut_5y = prev_lpr_5y - lpr_5y if prev_lpr_5y else 0
    total_cut = cut_1y + cut_5y

    if total_cut >= 0.15:
        return 'BIG_CUT', total_cut
    elif total_cut >= 0.05:
        return 'CUT', total_cut
    elif total_cut <= -0.15:
        return 'HIKE', total_cut
    elif total_cut <= -0.05:
        return 'MILD_HIKE', total_cut
    else:
        return 'HOLD', total_cut


def run_backtest(csv_path):
    df = load_eth_data(csv_path)
    results = []
    sorted_dates = sorted(RELEASES.keys())

    for i, date_str in enumerate(sorted_dates):
        release_data = RELEASES[date_str]
        release_dt = datetime.strptime(date_str, '%Y-%m-%d').replace(hour=1, minute=15)

        bars_after = get_bars_between(df, release_dt, release_dt + timedelta(hours=24))
        if len(bars_after) < 4:
            continue

        pre_bar = get_bar_at(df, release_dt)
        if pre_bar is None:
            continue
        pre_price = float(pre_bar['Close'])

        lpr_1y = release_data['lpr_1y']
        lpr_5y = release_data['lpr_5y']
        prev_lpr_1y = RELEASES[sorted_dates[i-1]]['lpr_1y'] if i > 0 else lpr_1y
        prev_lpr_5y = RELEASES[sorted_dates[i-1]]['lpr_5y'] if i > 0 else lpr_5y

        signal, total_cut = classify_signal(lpr_1y, lpr_5y, prev_lpr_1y, prev_lpr_5y)
        wyckoff = classify_wyckoff_phase(df, release_dt)
        vol_regime = classify_vol_regime(df, release_dt)

        session_returns = {}
        for name, hours in SESSION_CHAIN:
            if len(hours) == 1:
                start = release_dt.replace(hour=hours[0], minute=0)
                end = start + timedelta(hours=1)
            else:
                start = release_dt.replace(hour=hours[0], minute=0)
                end = release_dt.replace(hour=hours[-1] + 1, minute=0)
            if end <= start:
                end += timedelta(days=1)
            ret = get_return(df, start, end)
            if ret is not None:
                session_returns[name] = round(ret, 4)

        ret_24h = get_return(df, release_dt, release_dt + timedelta(hours=24))
        ret_48h = get_return(df, release_dt, release_dt + timedelta(hours=48))
        year = int(date_str[:4])

        results.append({
            'date': date_str, 'year': year,
            'lpr_1y': lpr_1y, 'lpr_5y': lpr_5y,
            'prev_lpr_1y': prev_lpr_1y, 'prev_lpr_5y': prev_lpr_5y,
            'cut_1y': round(prev_lpr_1y - lpr_1y, 2),
            'cut_5y': round(prev_lpr_5y - lpr_5y, 2),
            'total_cut': round(total_cut, 2),
            'signal': signal,
            'wyckoff': wyckoff, 'vol_regime': vol_regime,
            'pre_price': round(pre_price, 2),
            'ret_24h': round(ret_24h, 4) if ret_24h else None,
            'ret_48h': round(ret_48h, 4) if ret_48h else None,
            'session_returns': session_returns,
        })

    return pd.DataFrame(results)


def cross_tabulate(results):
    print("\n" + "="*80)
    print("CROSS-TABULATION: Signal × Vol Regime × Wyckoff Phase")
    print("="*80)
    valid = results[results['ret_24h'].notna()]
    combos = valid.groupby(['signal', 'vol_regime', 'wyckoff'])
    rows = []
    for (sig, vol, wy), group in combos:
        n = len(group)
        avg_ret = group['ret_24h'].mean()
        win_rate = (group['ret_24h'] > 0).mean()
        rows.append({'signal': sig, 'vol': vol, 'wyckoff': wy, 'n': n,
                     'avg_24h': round(avg_ret, 3), 'win_rate': round(win_rate, 3)})
    df = pd.DataFrame(rows).sort_values('n', ascending=False)
    print(df.to_string(index=False))
    edges = df[(df['n'] >= 3) & (df['avg_24h'].abs() >= 0.5)]
    if len(edges) > 0:
        print("\n" + "-"*60)
        print("ACTIONABLE EDGES (n≥3, |avg|≥0.5%):")
        print("-"*60)
        print(edges.to_string(index=False))
    else:
        print("\nNo actionable edges found (n≥3, |avg|≥0.5%)")
    return df


def transmission_chain(results):
    print("\n" + "="*80)
    print("SESSION TRANSMISSION CHAIN (Prompt B)")
    print("="*80)
    valid = results[results['ret_24h'].notna()]
    key_sessions = ['Release (1h)', 'Tokyo Open', 'Asia Mid', 'Asia Afternoon',
                    'Tokyo Close', 'Pre-London', 'London Open', 'London Morning',
                    'London Midday', 'NY Open', 'London-NY Overlap', 'NY AM', 'NY PM']

    print(f"\n{'Session':<25} {'Avg Ret%':>10} {'Win%':>8} {'n':>5} {'Dir':>6}")
    print("-"*60)
    chain_data = []
    for session_name in key_sessions:
        rets = [sr[session_name] for _, row in valid.iterrows()
                for sr in [row.get('session_returns', {})] if session_name in sr]
        if len(rets) < 3:
            continue
        avg = np.mean(rets)
        win = sum(1 for r in rets if r > 0) / len(rets) * 100
        n = len(rets)
        curr_dir = 'UP' if avg > 0.05 else 'DOWN' if avg < -0.05 else 'FLAT'
        print(f"{session_name:<25} {avg:>+10.3f} {win:>7.0f}% {n:>5} {curr_dir:>6}")
        chain_data.append({'session': session_name, 'avg_ret': round(avg, 3),
                           'win_pct': round(win, 1), 'n': n, 'direction': curr_dir})

    print(f"\n{'Transition':<45} {'Same Dir%':>10} {'n':>5} {'Verdict':>10}")
    print("-"*75)
    transitions = []
    for i in range(len(key_sessions) - 1):
        s1, s2 = key_sessions[i], key_sessions[i+1]
        same_count = total = 0
        for _, row in valid.iterrows():
            sr = row.get('session_returns', {})
            if s1 in sr and s2 in sr:
                r1, r2 = sr[s1], sr[s2]
                if (r1 > 0.05 and r2 > 0.05) or (r1 < -0.05 and r2 < -0.05):
                    same_count += 1
                total += 1
        if total >= 3:
            pct = same_count / total * 100
            verdict = '✅ EDGE' if pct > 65 else '⚠️ MARGINAL' if pct > 55 else '❌ NO CHAIN'
            print(f"{s1 + ' → ' + s2:<45} {pct:>9.0f}% {total:>5} {verdict:>10}")
            transitions.append({'from': s1, 'to': s2, 'same_dir_pct': round(pct, 1),
                                'n': total, 'verdict': verdict})
    return chain_data, transitions


def statistical_tests(results):
    print("\n" + "="*80)
    print("STATISTICAL SIGNIFICANCE")
    print("="*80)
    valid = results[results['ret_24h'].notna()]
    rets = valid['ret_24h'].values
    t_stat, p_val = stats.ttest_1samp(rets, 0)
    print(f"\n1. One-sample t-test (24h return vs 0):")
    print(f"   Mean: {np.mean(rets):+.3f}%  t={t_stat:.3f}  p={p_val:.4f}  {'*** SIGNIFICANT' if p_val < 0.05 else 'NOT significant'}")

    cut_rets = valid[valid['signal'].isin(['BIG_CUT', 'CUT'])]['ret_24h'].values
    hold_rets = valid[valid['signal'] == 'HOLD']['ret_24h'].values
    if len(cut_rets) >= 3 and len(hold_rets) >= 3:
        t2, p2 = stats.ttest_ind(cut_rets, hold_rets)
        print(f"\n2. Two-sample t-test (CUT vs HOLD):")
        print(f"   CUT: {np.mean(cut_rets):+.3f}% (n={len(cut_rets)})  HOLD: {np.mean(hold_rets):+.3f}% (n={len(hold_rets)})")
        print(f"   t={t2:.3f}  p={p2:.4f}  {'*** SIGNIFICANT' if p2 < 0.05 else 'NOT significant'}")

    for sig in ['BIG_CUT', 'CUT', 'HOLD']:
        sig_rets = valid[valid['signal'] == sig]['ret_24h'].values
        if len(sig_rets) >= 3:
            t3, p3 = stats.ttest_1samp(sig_rets, 0)
            print(f"\n3. {sig} (n={len(sig_rets)}):")
            print(f"   Mean: {np.mean(sig_rets):+.3f}%  t={t3:.3f}  p={p3:.4f}  {'***' if p3 < 0.05 else 'ns'}")

    print(f"\n4. Year-by-year 24h returns:")
    print(f"   {'Year':<6} {'Avg%':>8} {'Win%':>7} {'n':>4}")
    print(f"   {'-'*30}")
    for year in sorted(valid['year'].unique()):
        yr = valid[valid['year'] == year]
        print(f"   {year:<6} {yr['ret_24h'].mean():>+8.3f} {(yr['ret_24h'] > 0).mean()*100:>6.0f}% {len(yr):>4}")


def main():
    csv_path = os.path.join(os.path.dirname(__file__), 'eth_15m_merged.csv')
    print("="*80)
    print("PROMPT A: PBOC LPR DECISION BACKTEST (2019-2026)")
    print("="*80)
    print("Loading ETH 15m data...")
    results = run_backtest(csv_path)
    print(f"Analyzed {len(results)} PBoC LPR releases ({results['year'].min()}-{results['year'].max()})")

    valid = results[results['ret_24h'].notna()]
    print(f"\n24h Aggregate Returns:")
    print(f"  Mean:   {valid['ret_24h'].mean():+.3f}%")
    print(f"  Median: {valid['ret_24h'].median():+.3f}%")
    print(f"  Std:    {valid['ret_24h'].std():.3f}%")
    print(f"  Win%:   {(valid['ret_24h'] > 0).mean()*100:.1f}%")
    print(f"  n:      {len(valid)}")

    print("\n  By Signal:")
    for sig in ['BIG_CUT', 'CUT', 'HOLD', 'MILD_HIKE', 'HIKE']:
        sig_data = valid[valid['signal'] == sig]
        if len(sig_data) > 0:
            print(f"  {sig:<15} avg={sig_data['ret_24h'].mean():+.3f}%  win={(sig_data['ret_24h'] > 0).mean()*100:.0f}%  n={len(sig_data)}")

    cross_tabulate(results)
    print("\n" + "="*80)
    print("PROMPT B: SESSION TRANSMISSION CHAIN VALIDATION")
    print("="*80)
    transmission_chain(results)
    statistical_tests(results)

    out_path = os.path.join(os.path.dirname(__file__), 'backtest_pboc_lpr_results.json')
    results.to_json(out_path, orient='records', indent=2)
    print(f"\nResults saved to {out_path}")


if __name__ == '__main__':
    main()

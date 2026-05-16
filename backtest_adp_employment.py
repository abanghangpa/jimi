#!/usr/bin/env python3
"""
Prompt A + B: Backtest ADP Employment Report (2018-today) using ETH/USDT 15m data.

ADP National Employment Report released at 08:15 ET (12:15 UTC) on the first Wednesday
of each month (approximately). Private payrolls tracker, 2 days before official NFP.

Session itinerary (per user's thesis #19):
  US Wednesday Morning (12:15 UTC) → Thursday Claims → Friday NFP

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
# ADP EMPLOYMENT REPORT RELEASE DATES (08:15 ET = 12:15 UTC)
# Released ~first Wednesday of each month
# Format: {date: {'adp_k': float, 'consensus_k': float, 'prev_k': float}}
# adp_k = ADP private payrolls change in thousands
# ═══════════════════════════════════════════════════════════════

RELEASES = {
    # 2018
    '2018-01-03': {'adp_k': 250, 'consensus_k': 190, 'prev_k': 185},
    '2018-02-07': {'adp_k': 234, 'consensus_k': 185, 'prev_k': 244},
    '2018-03-07': {'adp_k': 235, 'consensus_k': 195, 'prev_k': 244},
    '2018-04-04': {'adp_k': 241, 'consensus_k': 205, 'prev_k': 240},
    '2018-05-02': {'adp_k': 204, 'consensus_k': 198, 'prev_k': 228},
    '2018-06-06': {'adp_k': 178, 'consensus_k': 190, 'prev_k': 163},
    '2018-07-05': {'adp_k': 177, 'consensus_k': 190, 'prev_k': 189},
    '2018-08-01': {'adp_k': 219, 'consensus_k': 185, 'prev_k': 181},
    '2018-09-06': {'adp_k': 163, 'consensus_k': 190, 'prev_k': 217},
    '2018-10-03': {'adp_k': 230, 'consensus_k': 185, 'prev_k': 168},
    '2018-11-07': {'adp_k': 227, 'consensus_k': 189, 'prev_k': 218},
    '2018-12-05': {'adp_k': 179, 'consensus_k': 195, 'prev_k': 225},
    # 2019
    '2019-01-03': {'adp_k': 271, 'consensus_k': 178, 'prev_k': 157},
    '2019-02-06': {'adp_k': 213, 'consensus_k': 181, 'prev_k': 263},
    '2019-03-06': {'adp_k': 183, 'consensus_k': 185, 'prev_k': 209},
    '2019-04-03': {'adp_k': 129, 'consensus_k': 170, 'prev_k': 197},
    '2019-05-01': {'adp_k': 275, 'consensus_k': 180, 'prev_k': 151},
    '2019-06-05': {'adp_k': 27, 'consensus_k': 180, 'prev_k': 271},
    '2019-07-03': {'adp_k': 102, 'consensus_k': 140, 'prev_k': 41},
    '2019-08-01': {'adp_k': 156, 'consensus_k': 150, 'prev_k': 112},
    '2019-09-05': {'adp_k': 195, 'consensus_k': 140, 'prev_k': 142},
    '2019-10-02': {'adp_k': 135, 'consensus_k': 140, 'prev_k': 157},
    '2019-11-06': {'adp_k': 125, 'consensus_k': 135, 'prev_k': 93},
    '2019-12-04': {'adp_k': 67, 'consensus_k': 140, 'prev_k': 121},
    # 2020
    '2020-01-08': {'adp_k': 202, 'consensus_k': 160, 'prev_k': 199},
    '2020-02-05': {'adp_k': 291, 'consensus_k': 156, 'prev_k': 199},
    '2020-03-04': {'adp_k': 183, 'consensus_k': 170, 'prev_k': 209},
    '2020-04-01': {'adp_k': -27, 'consensus_k': -150, 'prev_k': 147},
    '2020-05-06': {'adp_k': -20236, 'consensus_k': -20050, 'prev_k': -149},
    '2020-06-03': {'adp_k': -2760, 'consensus_k': -9000, 'prev_k': -19557},
    '2020-07-01': {'adp_k': 2369, 'consensus_k': -3000, 'prev_k': -3065},
    '2020-08-05': {'adp_k': 167, 'consensus_k': 1200, 'prev_k': 4314},
    '2020-09-02': {'adp_k': 428, 'consensus_k': 950, 'prev_k': 212},
    '2020-10-02': {'adp_k': 749, 'consensus_k': 600, 'prev_k': 481},
    '2020-11-04': {'adp_k': 365, 'consensus_k': 600, 'prev_k': 749},
    '2020-12-02': {'adp_k': 307, 'consensus_k': 440, 'prev_k': 404},
    # 2021
    '2021-01-06': {'adp_k': -123, 'consensus_k': 60, 'prev_k': -75},
    '2021-02-03': {'adp_k': 174, 'consensus_k': 50, 'prev_k': -78},
    '2021-03-03': {'adp_k': 117, 'consensus_k': 205, 'prev_k': 195},
    '2021-04-01': {'adp_k': 517, 'consensus_k': 525, 'prev_k': 176},
    '2021-05-05': {'adp_k': 742, 'consensus_k': 800, 'prev_k': 565},
    '2021-06-03': {'adp_k': 978, 'consensus_k': 650, 'prev_k': 654},
    '2021-07-01': {'adp_k': 692, 'consensus_k': 600, 'prev_k': 882},
    '2021-08-04': {'adp_k': 330, 'consensus_k': 695, 'prev_k': 680},
    '2021-09-01': {'adp_k': 374, 'consensus_k': 613, 'prev_k': 326},
    '2021-10-06': {'adp_k': 568, 'consensus_k': 430, 'prev_k': 340},
    '2021-11-03': {'adp_k': 571, 'consensus_k': 400, 'prev_k': 523},
    '2021-12-01': {'adp_k': 534, 'consensus_k': 525, 'prev_k': 570},
    # 2022
    '2022-01-05': {'adp_k': 807, 'consensus_k': 400, 'prev_k': 505},
    '2022-02-02': {'adp_k': -301, 'consensus_k': 207, 'prev_k': 776},
    '2022-03-02': {'adp_k': 475, 'consensus_k': 388, 'prev_k': 509},
    '2022-04-06': {'adp_k': 455, 'consensus_k': 450, 'prev_k': 486},
    '2022-05-04': {'adp_k': 247, 'consensus_k': 395, 'prev_k': 479},
    '2022-06-01': {'adp_k': 128, 'consensus_k': 300, 'prev_k': 202},
    '2022-07-07': {'adp_k': 132, 'consensus_k': 200, 'prev_k': -306},
    '2022-08-03': {'adp_k': 272, 'consensus_k': 200, 'prev_k': 128},
    '2022-09-01': {'adp_k': 132, 'consensus_k': 288, 'prev_k': 270},
    '2022-10-05': {'adp_k': 208, 'consensus_k': 200, 'prev_k': 185},
    '2022-11-02': {'adp_k': 239, 'consensus_k': 193, 'prev_k': 312},
    '2022-12-01': {'adp_k': 127, 'consensus_k': 200, 'prev_k': 239},
    # 2023
    '2023-01-05': {'adp_k': 235, 'consensus_k': 153, 'prev_k': 182},
    '2023-02-01': {'adp_k': 106, 'consensus_k': 178, 'prev_k': 253},
    '2023-03-08': {'adp_k': 242, 'consensus_k': 210, 'prev_k': 119},
    '2023-04-05': {'adp_k': 145, 'consensus_k': 210, 'prev_k': 261},
    '2023-05-03': {'adp_k': 296, 'consensus_k': 133, 'prev_k': 142},
    '2023-06-01': {'adp_k': 278, 'consensus_k': 170, 'prev_k': 291},
    '2023-07-06': {'adp_k': 497, 'consensus_k': 225, 'prev_k': 267},
    '2023-08-02': {'adp_k': 324, 'consensus_k': 189, 'prev_k': 455},
    '2023-09-06': {'adp_k': 177, 'consensus_k': 195, 'prev_k': 371},
    '2023-10-04': {'adp_k': 89, 'consensus_k': 153, 'prev_k': 180},
    '2023-11-01': {'adp_k': 113, 'consensus_k': 130, 'prev_k': 89},
    '2023-12-06': {'adp_k': 103, 'consensus_k': 130, 'prev_k': 106},
    # 2024
    '2024-01-04': {'adp_k': 164, 'consensus_k': 115, 'prev_k': 101},
    '2024-02-01': {'adp_k': 107, 'consensus_k': 145, 'prev_k': 158},
    '2024-03-06': {'adp_k': 140, 'consensus_k': 148, 'prev_k': 111},
    '2024-04-03': {'adp_k': 184, 'consensus_k': 148, 'prev_k': 155},
    '2024-05-01': {'adp_k': 192, 'consensus_k': 175, 'prev_k': 183},
    '2024-06-05': {'adp_k': 152, 'consensus_k': 173, 'prev_k': 188},
    '2024-07-03': {'adp_k': 150, 'consensus_k': 165, 'prev_k': 157},
    '2024-08-01': {'adp_k': 122, 'consensus_k': 150, 'prev_k': 155},
    '2024-09-05': {'adp_k': 99, 'consensus_k': 140, 'prev_k': 111},
    '2024-10-02': {'adp_k': 143, 'consensus_k': 125, 'prev_k': 103},
    '2024-11-06': {'adp_k': 233, 'consensus_k': 113, 'prev_k': 159},
    '2024-12-04': {'adp_k': 146, 'consensus_k': 150, 'prev_k': 184},
    # 2025
    '2025-01-08': {'adp_k': 122, 'consensus_k': 136, 'prev_k': 146},
    '2025-02-05': {'adp_k': 183, 'consensus_k': 150, 'prev_k': 176},
    '2025-03-05': {'adp_k': 77, 'consensus_k': 148, 'prev_k': 186},
    '2025-04-02': {'adp_k': 155, 'consensus_k': 115, 'prev_k': 84},
    '2025-05-07': {'adp_k': 62, 'consensus_k': 110, 'prev_k': 147},
    '2025-06-04': {'adp_k': 140, 'consensus_k': 110, 'prev_k': 73},
    '2025-07-02': {'adp_k': 150, 'consensus_k': 120, 'prev_k': 137},
    '2025-08-06': {'adp_k': 104, 'consensus_k': 130, 'prev_k': 148},
    '2025-09-03': {'adp_k': 135, 'consensus_k': 125, 'prev_k': 106},
    '2025-10-01': {'adp_k': 120, 'consensus_k': 130, 'prev_k': 133},
    '2025-11-05': {'adp_k': 145, 'consensus_k': 120, 'prev_k': 122},
    '2025-12-03': {'adp_k': 130, 'consensus_k': 125, 'prev_k': 143},
    # 2026
    '2026-01-07': {'adp_k': 140, 'consensus_k': 130, 'prev_k': 128},
    '2026-02-04': {'adp_k': 125, 'consensus_k': 135, 'prev_k': 138},
    '2026-03-04': {'adp_k': 150, 'consensus_k': 130, 'prev_k': 127},
    '2026-04-01': {'adp_k': 132, 'consensus_k': 128, 'prev_k': 148},
    '2026-05-06': {'adp_k': 118, 'consensus_k': 130, 'prev_k': 135},
}


# ═══════════════════════════════════════════════════════════════
# SESSION DEFINITIONS (ADP at 12:15 UTC)
# US-focused: release → US session → next-day claims calibration
# ═══════════════════════════════════════════════════════════════

SESSION_CHAIN = [
    # Pre-release
    ('Pre-Asia', [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]),

    # Release window (12:15 UTC = 08:15 ET)
    ('Release (1h)', [12, 13]),           # 12:15-13:15
    ('NY Pre-Open', [12, 13]),            # 12:00-14:00
    ('NY Open', [13, 14]),                # 13:30-15:00
    ('London-NY Overlap', [14, 15]),      # 14:00-16:00

    # NY session
    ('NY AM', [14, 15, 16]),              # 14:00-17:00
    ('NY Lunch', [17]),                   # 17:00-18:00
    ('NY PM', [18, 19, 20]),              # 18:00-21:00

    # Asia next day (claims calibration)
    ('Asia Next Day', [0, 1, 2, 3, 4, 5, 6, 7]),
    ('London Next Day', [8, 9, 10, 11, 12, 13]),

    # Post-release windows
    ('Release (4h)', [12, 13, 14, 15, 16]),  # 12:15-16:15
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


def classify_signal(adp_k, consensus_k, prev_k):
    """Classify ADP signal based on surprise vs consensus."""
    surprise = adp_k - consensus_k

    if surprise > 75:
        return 'STRONG_BEAT', surprise
    elif surprise > 25:
        return 'BEAT', surprise
    elif surprise < -75:
        return 'BIG_MISS', surprise
    elif surprise < -25:
        return 'MISS', surprise
    else:
        return 'INLINE', surprise


def run_backtest(csv_path):
    df = load_eth_data(csv_path)
    results = []
    sorted_dates = sorted(RELEASES.keys())

    for i, date_str in enumerate(sorted_dates):
        release_data = RELEASES[date_str]
        release_dt = datetime.strptime(date_str, '%Y-%m-%d').replace(hour=12, minute=15)

        bars_after = get_bars_between(df, release_dt, release_dt + timedelta(hours=24))
        if len(bars_after) < 4:
            continue

        pre_bar = get_bar_at(df, release_dt)
        if pre_bar is None:
            continue
        pre_price = float(pre_bar['Close'])

        adp_k = release_data['adp_k']
        consensus_k = release_data['consensus_k']
        prev_k = release_data['prev_k']
        signal, surprise = classify_signal(adp_k, consensus_k, prev_k)

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
            'adp_k': adp_k, 'consensus_k': consensus_k, 'prev_k': prev_k,
            'surprise': surprise, 'signal': signal,
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
    key_sessions = ['Release (1h)', 'NY Open', 'London-NY Overlap',
                    'NY AM', 'NY Lunch', 'NY PM',
                    'Asia Next Day', 'London Next Day']

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

    miss_rets = valid[valid['signal'].isin(['MISS', 'BIG_MISS'])]['ret_24h'].values
    beat_rets = valid[valid['signal'].isin(['BEAT', 'STRONG_BEAT'])]['ret_24h'].values
    if len(miss_rets) >= 3 and len(beat_rets) >= 3:
        t2, p2 = stats.ttest_ind(miss_rets, beat_rets)
        print(f"\n2. Two-sample t-test (MISS vs BEAT):")
        print(f"   MISS: {np.mean(miss_rets):+.3f}% (n={len(miss_rets)})  BEAT: {np.mean(beat_rets):+.3f}% (n={len(beat_rets)})")
        print(f"   t={t2:.3f}  p={p2:.4f}  {'*** SIGNIFICANT' if p2 < 0.05 else 'NOT significant'}")

    for sig in ['STRONG_BEAT', 'BEAT', 'INLINE', 'MISS', 'BIG_MISS']:
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
    print("PROMPT A: ADP EMPLOYMENT REPORT BACKTEST (2018-2026)")
    print("="*80)
    print("Loading ETH 15m data...")
    results = run_backtest(csv_path)
    print(f"Analyzed {len(results)} ADP releases ({results['year'].min()}-{results['year'].max()})")

    valid = results[results['ret_24h'].notna()]
    print(f"\n24h Aggregate Returns:")
    print(f"  Mean:   {valid['ret_24h'].mean():+.3f}%")
    print(f"  Median: {valid['ret_24h'].median():+.3f}%")
    print(f"  Std:    {valid['ret_24h'].std():.3f}%")
    print(f"  Win%:   {(valid['ret_24h'] > 0).mean()*100:.1f}%")
    print(f"  n:      {len(valid)}")

    print("\n  By Signal:")
    for sig in ['STRONG_BEAT', 'BEAT', 'INLINE', 'MISS', 'BIG_MISS']:
        sig_data = valid[valid['signal'] == sig]
        if len(sig_data) > 0:
            print(f"  {sig:<15} avg={sig_data['ret_24h'].mean():+.3f}%  win={(sig_data['ret_24h'] > 0).mean()*100:.0f}%  n={len(sig_data)}")

    cross_tabulate(results)
    print("\n" + "="*80)
    print("PROMPT B: SESSION TRANSMISSION CHAIN VALIDATION")
    print("="*80)
    transmission_chain(results)
    statistical_tests(results)

    out_path = os.path.join(os.path.dirname(__file__), 'backtest_adp_results.json')
    results.to_json(out_path, orient='records', indent=2)
    print(f"\nResults saved to {out_path}")


if __name__ == '__main__':
    main()

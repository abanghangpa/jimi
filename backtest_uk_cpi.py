#!/usr/bin/env python3
"""
Prompt A: Backtest UK CPI (2018-today) using ETH/USDT 15m data.

UK CPI released by ONS at 07:00 GMT/BST (07:00 UTC winter, 06:00 UTC summer).
We use 07:00 UTC as baseline for simplicity.

Thesis chain (per user #13):
  Europe Morning → US Session → UK Central Bank Open

Key sub-component: UK Services CPI (BoE watches this above all else)
  - Hot Services CPI → GBP surges → DXY deflates → ETH bounces (mechanical)
  - Hot UK CPI → US desks see global wage-price spiral → halts dovish positioning
  - Data → BoE MPC Vote split → hawkish dissent controls regional liquidity

We measure:
  1. ETH returns across all session phases
  2. Wyckoff phase, vol regime, signal classification
  3. Cross-tabulation
  4. Session transmission chain
  5. Statistical significance
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from scipy import stats
import json
import os


# ═══════════════════════════════════════════════════════════════
# UK CPI RELEASE DATES + ACTUAL VALUES (07:00 UTC)
# Format: {date: {'cpi_yoy': float, 'services_cpi_yoy': float}}
# Services CPI is the BoE's key metric
# ═══════════════════════════════════════════════════════════════

RELEASES = {
    # 2018
    '2018-01-16': {'cpi_yoy': 3.0, 'services_yoy': 2.8},
    '2018-02-13': {'cpi_yoy': 3.0, 'services_yoy': 2.8},
    '2018-03-20': {'cpi_yoy': 2.7, 'services_yoy': 2.6},
    '2018-04-18': {'cpi_yoy': 2.5, 'services_yoy': 2.5},
    '2018-05-23': {'cpi_yoy': 2.4, 'services_yoy': 2.5},
    '2018-06-13': {'cpi_yoy': 2.4, 'services_yoy': 2.5},
    '2018-07-18': {'cpi_yoy': 2.4, 'services_yoy': 2.4},
    '2018-08-15': {'cpi_yoy': 2.5, 'services_yoy': 2.5},
    '2018-09-19': {'cpi_yoy': 2.4, 'services_yoy': 2.5},
    '2018-10-17': {'cpi_yoy': 2.4, 'services_yoy': 2.4},
    '2018-11-14': {'cpi_yoy': 2.4, 'services_yoy': 2.4},
    '2018-12-19': {'cpi_yoy': 2.3, 'services_yoy': 2.3},
    # 2019
    '2019-01-16': {'cpi_yoy': 2.1, 'services_yoy': 2.2},
    '2019-02-13': {'cpi_yoy': 1.8, 'services_yoy': 2.1},
    '2019-03-20': {'cpi_yoy': 1.9, 'services_yoy': 2.1},
    '2019-04-17': {'cpi_yoy': 1.9, 'services_yoy': 2.0},
    '2019-05-22': {'cpi_yoy': 2.1, 'services_yoy': 2.1},
    '2019-06-19': {'cpi_yoy': 2.0, 'services_yoy': 2.0},
    '2019-07-17': {'cpi_yoy': 2.0, 'services_yoy': 2.0},
    '2019-08-14': {'cpi_yoy': 2.1, 'services_yoy': 2.0},
    '2019-09-18': {'cpi_yoy': 1.7, 'services_yoy': 1.9},
    '2019-10-16': {'cpi_yoy': 1.5, 'services_yoy': 1.8},
    '2019-11-20': {'cpi_yoy': 1.5, 'services_yoy': 1.8},
    '2019-12-18': {'cpi_yoy': 1.5, 'services_yoy': 1.7},
    # 2020
    '2020-01-15': {'cpi_yoy': 1.4, 'services_yoy': 1.6},
    '2020-02-19': {'cpi_yoy': 1.8, 'services_yoy': 1.7},
    '2020-03-25': {'cpi_yoy': 1.5, 'services_yoy': 1.6},
    '2020-04-22': {'cpi_yoy': 0.8, 'services_yoy': 1.2},
    '2020-05-20': {'cpi_yoy': 0.5, 'services_yoy': 0.9},
    '2020-06-17': {'cpi_yoy': 0.5, 'services_yoy': 0.8},
    '2020-07-15': {'cpi_yoy': 0.6, 'services_yoy': 0.9},
    '2020-08-19': {'cpi_yoy': 0.2, 'services_yoy': 0.6},
    '2020-09-16': {'cpi_yoy': 0.5, 'services_yoy': 0.7},
    '2020-10-21': {'cpi_yoy': 0.5, 'services_yoy': 0.7},
    '2020-11-18': {'cpi_yoy': 0.3, 'services_yoy': 0.5},
    '2020-12-16': {'cpi_yoy': 0.3, 'services_yoy': 0.5},
    # 2021
    '2021-01-20': {'cpi_yoy': 0.6, 'services_yoy': 0.7},
    '2021-02-17': {'cpi_yoy': 0.4, 'services_yoy': 0.6},
    '2021-03-24': {'cpi_yoy': 0.7, 'services_yoy': 0.8},
    '2021-04-21': {'cpi_yoy': 1.5, 'services_yoy': 1.1},
    '2021-05-19': {'cpi_yoy': 2.1, 'services_yoy': 1.5},
    '2021-06-16': {'cpi_yoy': 2.5, 'services_yoy': 1.8},
    '2021-07-14': {'cpi_yoy': 2.5, 'services_yoy': 1.8},
    '2021-08-18': {'cpi_yoy': 3.2, 'services_yoy': 2.1},
    '2021-09-15': {'cpi_yoy': 3.2, 'services_yoy': 2.2},
    '2021-10-20': {'cpi_yoy': 4.2, 'services_yoy': 2.9},
    '2021-11-17': {'cpi_yoy': 5.1, 'services_yoy': 3.6},
    '2021-12-15': {'cpi_yoy': 5.1, 'services_yoy': 3.6},
    # 2022
    '2022-01-19': {'cpi_yoy': 5.5, 'services_yoy': 3.9},
    '2022-02-16': {'cpi_yoy': 5.5, 'services_yoy': 3.9},
    '2022-03-23': {'cpi_yoy': 7.0, 'services_yoy': 4.6},
    '2022-04-13': {'cpi_yoy': 7.0, 'services_yoy': 4.6},
    '2022-05-18': {'cpi_yoy': 9.0, 'services_yoy': 5.1},
    '2022-06-22': {'cpi_yoy': 9.1, 'services_yoy': 5.2},
    '2022-07-20': {'cpi_yoy': 10.1, 'services_yoy': 5.7},
    '2022-08-17': {'cpi_yoy': 10.1, 'services_yoy': 5.7},
    '2022-09-14': {'cpi_yoy': 9.9, 'services_yoy': 5.8},
    '2022-10-19': {'cpi_yoy': 11.1, 'services_yoy': 6.2},
    '2022-11-16': {'cpi_yoy': 11.1, 'services_yoy': 6.3},
    '2022-12-14': {'cpi_yoy': 10.7, 'services_yoy': 6.3},
    # 2023
    '2023-01-18': {'cpi_yoy': 10.5, 'services_yoy': 6.3},
    '2023-02-15': {'cpi_yoy': 10.1, 'services_yoy': 6.2},
    '2023-03-22': {'cpi_yoy': 10.4, 'services_yoy': 6.6},
    '2023-04-19': {'cpi_yoy': 10.1, 'services_yoy': 6.6},
    '2023-05-24': {'cpi_yoy': 8.7, 'services_yoy': 6.4},
    '2023-06-21': {'cpi_yoy': 8.7, 'services_yoy': 6.5},
    '2023-07-19': {'cpi_yoy': 7.9, 'services_yoy': 6.5},
    '2023-08-16': {'cpi_yoy': 6.8, 'services_yoy': 6.2},
    '2023-09-20': {'cpi_yoy': 6.7, 'services_yoy': 6.1},
    '2023-10-18': {'cpi_yoy': 6.7, 'services_yoy': 6.1},
    '2023-11-15': {'cpi_yoy': 4.6, 'services_yoy': 5.7},
    '2023-12-20': {'cpi_yoy': 3.9, 'services_yoy': 5.3},
    # 2024
    '2024-01-17': {'cpi_yoy': 4.0, 'services_yoy': 5.4},
    '2024-02-14': {'cpi_yoy': 4.0, 'services_yoy': 5.4},
    '2024-03-20': {'cpi_yoy': 3.4, 'services_yoy': 5.0},
    '2024-04-17': {'cpi_yoy': 3.2, 'services_yoy': 4.9},
    '2024-05-22': {'cpi_yoy': 2.3, 'services_yoy': 4.5},
    '2024-06-19': {'cpi_yoy': 2.0, 'services_yoy': 4.3},
    '2024-07-17': {'cpi_yoy': 2.0, 'services_yoy': 4.2},
    '2024-08-14': {'cpi_yoy': 2.2, 'services_yoy': 4.3},
    '2024-09-18': {'cpi_yoy': 2.2, 'services_yoy': 4.3},
    '2024-10-16': {'cpi_yoy': 2.2, 'services_yoy': 4.3},
    '2024-11-20': {'cpi_yoy': 2.3, 'services_yoy': 4.4},
    '2024-12-18': {'cpi_yoy': 2.6, 'services_yoy': 4.5},
    # 2025
    '2025-01-15': {'cpi_yoy': 2.5, 'services_yoy': 4.4},
    '2025-02-19': {'cpi_yoy': 3.0, 'services_yoy': 4.6},
    '2025-03-26': {'cpi_yoy': 2.8, 'services_yoy': 4.5},
    '2025-04-16': {'cpi_yoy': 2.6, 'services_yoy': 4.4},
    '2025-05-21': {'cpi_yoy': 3.5, 'services_yoy': 4.8},
    '2025-06-18': {'cpi_yoy': 3.4, 'services_yoy': 4.7},
    '2025-07-16': {'cpi_yoy': 3.6, 'services_yoy': 4.8},
    '2025-08-20': {'cpi_yoy': 3.8, 'services_yoy': 4.9},
    '2025-09-17': {'cpi_yoy': 3.5, 'services_yoy': 4.7},
    '2025-10-22': {'cpi_yoy': 3.2, 'services_yoy': 4.5},
    '2025-11-19': {'cpi_yoy': 2.9, 'services_yoy': 4.3},
    '2025-12-17': {'cpi_yoy': 2.7, 'services_yoy': 4.2},
    # 2026
    '2026-01-21': {'cpi_yoy': 2.8, 'services_yoy': 4.3},
    '2026-02-18': {'cpi_yoy': 2.9, 'services_yoy': 4.4},
    '2026-03-25': {'cpi_yoy': 2.6, 'services_yoy': 4.2},
    '2026-04-22': {'cpi_yoy': 2.4, 'services_yoy': 4.0},
    '2026-05-13': {'cpi_yoy': 2.3, 'services_yoy': 3.9},
}


# ═══════════════════════════════════════════════════════════════
# SESSION DEFINITIONS (UK CPI release at 07:00 UTC)
# ═══════════════════════════════════════════════════════════════

SESSIONS = {
    # Europe Morning (release window)
    'release_1h':       {'start': 7, 'end': 8},      # 07:00-08:00
    'london_open':      {'start': 8, 'end': 9},       # 08:00-09:00
    'london_morning':   {'start': 9, 'end': 12},      # 09:00-12:00
    'london_midday':    {'start': 12, 'end': 14},     # 12:00-14:00

    # Overlap (EU-US)
    'ny_pre_open':      {'start': 12, 'end': 13},     # 12:00-13:30
    'ny_open':          {'start': 13, 'end': 14},     # 13:30-14:00
    'london_ny_overlap':{'start': 14, 'end': 16},     # 14:00-16:00

    # US Session
    'ny_am':            {'start': 14, 'end': 17},     # 14:00-17:00
    'ny_lunch':         {'start': 17, 'end': 18},     # 17:00-18:00
    'ny_pm':            {'start': 18, 'end': 21},     # 18:00-21:00

    # Post-NY → Asia → UK Central Bank Open (next day)
    'post_ny':          {'start': 21, 'end': 23, 'minute_end': 59},  # 21:00-23:59
    'sydney_open':      {'start': 0, 'end': 1, 'next_day': True},       # 00:00-01:00 (next day)
    'tokyo_open':       {'start': 1, 'end': 3, 'next_day': True},       # 01:00-03:00 (next day)
    'asia_mid':         {'start': 3, 'end': 5, 'next_day': True},       # 03:00-05:00 (next day)
    'asia_afternoon':   {'start': 5, 'end': 7, 'next_day': True},       # 05:00-07:00 (next day)
    'boe_reopen':       {'start': 7, 'end': 8, 'next_day': True},       # 07:00-08:00 (next day, BoE opens)
}

SESSION_CHAIN = [
    ('Release (1h)', 'release_1h'),
    ('London Open', 'london_open'),
    ('London Morning', 'london_morning'),
    ('London Midday', 'london_midday'),
    ('NY Pre-Open', 'ny_pre_open'),
    ('NY Open', 'ny_open'),
    ('London-NY Overlap', 'london_ny_overlap'),
    ('NY AM', 'ny_am'),
    ('NY Lunch', 'ny_lunch'),
    ('NY PM', 'ny_pm'),
    ('Post-NY', 'post_ny'),
    ('Sydney Open', 'sydney_open'),
    ('Tokyo Open', 'tokyo_open'),
    ('Asia Mid', 'asia_mid'),
    ('Asia Afternoon', 'asia_afternoon'),
    ('BoE Re-open', 'boe_reopen'),
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


def classify_wyckoff_phase(price_series, lookback=96):
    if len(price_series) < lookback:
        return 'RANGE'
    recent = price_series[-lookback:]
    high = recent.max()
    low = recent.min()
    current = recent.iloc[-1]
    range_pct = (high - low) / low * 100
    pos = (current - low) / (high - low) if high != low else 0.5
    if range_pct < 1.5:
        return 'RANGE' if 0.3 < pos < 0.7 else ('ACCUMULATION' if pos < 0.3 else 'DISTRIBUTION')
    elif current > recent.mean() and pos > 0.6:
        return 'MARKUP'
    elif current < recent.mean() and pos < 0.4:
        return 'MARKDOWN'
    elif pos > 0.7:
        return 'DISTRIBUTION'
    elif pos < 0.3:
        return 'ACCUMULATION'
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
    return 'CHOP'


def classify_signal(cpi_yoy, services_yoy):
    """Classify UK CPI signal focusing on Services CPI (BoE's key metric).
    Hot Services CPI → BoE hawkish → GBP up → DXY down → ETH up (mechanical)
    """
    if services_yoy >= 5.0:
        return 'HOT_SERVICES'
    elif services_yoy >= 4.0:
        return 'WARM_SERVICES'
    elif services_yoy >= 3.0:
        return 'COOL_SERVICES'
    else:
        return 'COLD_SERVICES'


def classify_direction(cpi_yoy, services_yoy, prev_cpi=None, prev_services=None):
    """Surprise direction based on Services CPI change."""
    if prev_services is not None:
        delta = services_yoy - prev_services
        if delta > 0.2:
            return 'MISS'   # Services hotter = surprise higher
        elif delta < -0.2:
            return 'BEAT'   # Services cooler = surprise lower
    if services_yoy >= 4.5:
        return 'MISS'
    elif services_yoy < 3.5:
        return 'BEAT'
    return 'INLINE'


def run_backtest(csv_path):
    df = load_eth_data(csv_path)
    results = []
    sorted_dates = sorted(RELEASES.keys())

    for i, date_str in enumerate(sorted_dates):
        release_data = RELEASES[date_str]
        release_dt = datetime.strptime(date_str, '%Y-%m-%d').replace(hour=7, minute=0)

        bars_after = get_bars_between(df, release_dt, release_dt + timedelta(hours=30))
        if len(bars_after) < 4:
            continue

        pre_bar = get_bar_at(df, release_dt)
        if pre_bar is None:
            continue
        pre_price = float(pre_bar['Close'])

        cpi_yoy = release_data['cpi_yoy']
        services_yoy = release_data['services_yoy']
        prev_cpi = RELEASES[sorted_dates[i-1]]['cpi_yoy'] if i > 0 else None
        prev_services = RELEASES[sorted_dates[i-1]]['services_yoy'] if i > 0 else None

        signal = classify_signal(cpi_yoy, services_yoy)
        direction = classify_direction(cpi_yoy, services_yoy, prev_cpi, prev_services)
        wyckoff = classify_wyckoff_phase(df['Close'].astype(float))
        vol_regime = classify_vol_regime(df, release_dt)

        session_returns = {}
        for name, key in SESSION_CHAIN:
            s = SESSIONS[key]
            start_dt = release_dt.replace(hour=s['start'], minute=0)
            end_minute = s.get('minute_end', 0)
            end_dt = release_dt.replace(hour=s['end'], minute=end_minute)
            if s.get('next_day'):
                start_dt += timedelta(days=1)
                end_dt += timedelta(days=1)
            elif end_dt <= start_dt:
                end_dt += timedelta(days=1)

            ret = get_return(df, start_dt, end_dt)
            if ret is not None:
                session_returns[name] = round(ret, 4)

        ret_24h = get_return(df, release_dt, release_dt + timedelta(hours=24))
        ret_48h = get_return(df, release_dt, release_dt + timedelta(hours=48))
        year = int(date_str[:4])

        results.append({
            'date': date_str,
            'year': year,
            'cpi_yoy': cpi_yoy,
            'services_yoy': services_yoy,
            'prev_cpi': prev_cpi,
            'prev_services': prev_services,
            'signal': signal,
            'direction': direction,
            'wyckoff': wyckoff,
            'vol_regime': vol_regime,
            'pre_price': round(pre_price, 2),
            'ret_24h': round(ret_24h, 4) if ret_24h is not None else None,
            'ret_48h': round(ret_48h, 4) if ret_48h is not None else None,
            'session_returns': session_returns,
        })

    return pd.DataFrame(results)


def cross_tabulate(results):
    print("\n" + "="*80)
    print("CROSS-TABULATION: Signal × Vol Regime × Direction")
    print("="*80)
    valid = results[results['ret_24h'].notna()]
    combos = valid.groupby(['signal', 'vol_regime', 'direction'])
    rows = []
    for (sig, vol, d), group in combos:
        n = len(group)
        avg_ret = group['ret_24h'].mean()
        win_rate = (group['ret_24h'] > 0).mean()
        rows.append({
            'signal': sig, 'vol': vol, 'direction': d,
            'n': n, 'avg_24h': round(avg_ret, 3),
            'win_rate': round(win_rate, 3),
        })
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
    print("SESSION TRANSMISSION CHAIN")
    print("="*80)
    valid = results[results['ret_24h'].notna()]
    session_names = [s[0] for s in SESSION_CHAIN]

    print(f"\n{'Session':<25} {'Avg Ret%':>10} {'Win%':>8} {'n':>5}")
    print("-"*55)

    for session_name in session_names:
        rets = []
        for _, row in valid.iterrows():
            sr = row.get('session_returns', {})
            if session_name in sr:
                rets.append(sr[session_name])
        if len(rets) < 3:
            continue
        avg = np.mean(rets)
        win = sum(1 for r in rets if r > 0) / len(rets) * 100
        n = len(rets)
        print(f"{session_name:<25} {avg:>+10.3f} {win:>7.0f}% {n:>5}")

    # Direction persistence
    print(f"\n{'Transition':<45} {'Same Dir%':>10} {'n':>5} {'Status':>12}")
    print("-"*75)

    for i in range(len(session_names) - 1):
        s1 = session_names[i]
        s2 = session_names[i + 1]
        same_count = 0
        total = 0
        for _, row in valid.iterrows():
            sr = row.get('session_returns', {})
            if s1 in sr and s2 in sr:
                r1 = sr[s1]
                r2 = sr[s2]
                if (r1 > 0.05 and r2 > 0.05) or (r1 < -0.05 and r2 < -0.05):
                    same_count += 1
                total += 1
        if total < 5:
            continue
        same_pct = same_count / total * 100
        status = '✅ EDGE' if same_pct >= 65 else '🟡 MARGINAL' if same_pct >= 55 else '❌ BROKEN'
        print(f"  {s1} → {s2:<25} {same_pct:>9.1f}% {total:>5} {status:>12}")


def statistical_tests(results):
    print("\n" + "="*80)
    print("STATISTICAL SIGNIFICANCE")
    print("="*80)
    valid = results[results['ret_24h'].notna()]

    rets = valid['ret_24h'].values
    t, p = stats.ttest_1samp(rets, 0)
    print(f"\n1. One-sample t-test (24h return vs 0, n={len(rets)}):")
    print(f"   Mean: {np.mean(rets):+.3f}%  t={t:.3f}  p={p:.4f}  {'*** SIGNIFICANT' if p < 0.05 else 'NOT significant'}")

    miss = valid[valid['direction'] == 'MISS']['ret_24h'].values
    beat = valid[valid['direction'] == 'BEAT']['ret_24h'].values
    if len(miss) >= 3 and len(beat) >= 3:
        t2, p2 = stats.ttest_ind(miss, beat)
        print(f"\n2. MISS vs BEAT:")
        print(f"   MISS: {np.mean(miss):+.3f}% (n={len(miss)})  BEAT: {np.mean(beat):+.3f}% (n={len(beat)})")
        print(f"   t={t2:.3f}  p={p2:.4f}  {'*** SIGNIFICANT' if p2 < 0.05 else 'NOT significant'}")

    for sig in ['HOT_SERVICES', 'WARM_SERVICES', 'COOL_SERVICES', 'COLD_SERVICES']:
        sig_rets = valid[valid['signal'] == sig]['ret_24h'].values
        if len(sig_rets) >= 3:
            t3, p3 = stats.ttest_1samp(sig_rets, 0)
            print(f"\n3. {sig} (n={len(sig_rets)}):")
            print(f"   Mean: {np.mean(sig_rets):+.3f}%  win={np.mean(sig_rets > 0)*100:.0f}%  t={t3:.3f}  p={p3:.4f}  {'***' if p3 < 0.05 else 'ns'}")

    print(f"\n4. Year-by-year 24h returns:")
    print(f"   {'Year':<6} {'Avg%':>8} {'Win%':>7} {'n':>4}")
    print(f"   {'-'*30}")
    for year in sorted(valid['year'].unique()):
        yr = valid[valid['year'] == year]
        avg = yr['ret_24h'].mean()
        win = (yr['ret_24h'] > 0).mean() * 100
        n = len(yr)
        print(f"   {year:<6} {avg:>+8.3f} {win:>6.0f}% {n:>4}")


def main():
    csv_path = os.path.join(os.path.dirname(__file__), 'eth_15m_merged.csv')
    print("Loading ETH 15m data...")
    results = run_backtest(csv_path)
    print(f"Analyzed {len(results)} UK CPI releases ({results['year'].min()}-{results['year'].max()})")

    valid = results[results['ret_24h'].notna()]
    print(f"\n24h Aggregate Returns:")
    print(f"  Mean:   {valid['ret_24h'].mean():+.3f}%")
    print(f"  Median: {valid['ret_24h'].median():+.3f}%")
    print(f"  Std:    {valid['ret_24h'].std():.3f}%")
    print(f"  Win%:   {(valid['ret_24h'] > 0).mean()*100:.1f}%")
    print(f"  n:      {len(valid)}")

    cross_tabulate(results)
    transmission_chain(results)
    statistical_tests(results)

    print("\n" + "="*80)
    print("SIGNAL BREAKDOWN")
    print("="*80)
    for sig in ['HOT_SERVICES', 'WARM_SERVICES', 'COOL_SERVICES', 'COLD_SERVICES']:
        sig_data = valid[valid['signal'] == sig]
        if len(sig_data) > 0:
            avg = sig_data['ret_24h'].mean()
            win = (sig_data['ret_24h'] > 0).mean() * 100
            n = len(sig_data)
            print(f"  {sig:<20} avg={avg:+.3f}%  win={win:.0f}%  n={n}")

    print("\n  By Direction:")
    for d in ['BEAT', 'MISS', 'INLINE']:
        d_data = valid[valid['direction'] == d]
        if len(d_data) > 0:
            avg = d_data['ret_24h'].mean()
            win = (d_data['ret_24h'] > 0).mean() * 100
            n = len(d_data)
            print(f"  {d:<10} avg={avg:+.3f}%  win={win:.0f}%  n={n}")

    out_path = os.path.join(os.path.dirname(__file__), 'backtest_uk_results.json')
    results.to_json(out_path, orient='records', indent=2)
    print(f"\nResults saved to {out_path}")

if __name__ == '__main__':
    main()

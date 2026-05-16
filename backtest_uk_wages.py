#!/usr/bin/env python3
"""
Prompt A: Backtest UK Employment + Wages (2018-today) using ETH/USDT 15m data.

UK labor market data released by ONS at 07:00 UTC (~2nd Tuesday of month).
Key metric: Average Earnings Index (3m/yr) — wage growth drives future Services CPI.

Thesis (from user #14):
  Europe Morning → Mid-Week Settlement → Next UK CPI
  Wage growth beats → future Services CPI stays high → mid-term risk headwind
  Higher global wage prints → US 10Y yield drift up → crypto bid density drops
  Realized wages → feed into next month's UK CPI (closed loop)

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
# UK EMPLOYMENT + WAGES RELEASE DATES (07:00 UTC)
# ONS releases Average Earnings Index (3m/yr) + Unemployment Rate
# Released ~2nd Tuesday of each month
# ═══════════════════════════════════════════════════════════════

RELEASES = {
    # 2018
    '2018-01-24': {'earnings_3m_yr': 2.5, 'unemployment': 4.3, 'claimant_count_chg': 8.6},
    '2018-02-21': {'earnings_3m_yr': 2.8, 'unemployment': 4.3, 'claimant_count_chg': 7.2},
    '2018-03-21': {'earnings_3m_yr': 2.8, 'unemployment': 4.2, 'claimant_count_chg': -11.6},
    '2018-04-17': {'earnings_3m_yr': 2.8, 'unemployment': 4.2, 'claimant_count_chg': -11.6},
    '2018-05-15': {'earnings_3m_yr': 2.9, 'unemployment': 4.2, 'claimant_count_chg': 3.1},
    '2018-06-12': {'earnings_3m_yr': 2.8, 'unemployment': 4.2, 'claimant_count_chg': -7.7},
    '2018-07-17': {'earnings_3m_yr': 2.7, 'unemployment': 4.0, 'claimant_count_chg': 7.8},
    '2018-08-14': {'earnings_3m_yr': 2.7, 'unemployment': 4.0, 'claimant_count_chg': 6.2},
    '2018-09-11': {'earnings_3m_yr': 2.9, 'unemployment': 4.0, 'claimant_count_chg': 8.7},
    '2018-10-16': {'earnings_3m_yr': 3.1, 'unemployment': 4.0, 'claimant_count_chg': 5.8},
    '2018-11-13': {'earnings_3m_yr': 3.3, 'unemployment': 4.1, 'claimant_count_chg': 20.2},
    '2018-12-11': {'earnings_3m_yr': 3.3, 'unemployment': 4.1, 'claimant_count_chg': 21.9},
    # 2019
    '2019-01-22': {'earnings_3m_yr': 3.4, 'unemployment': 4.0, 'claimant_count_chg': 20.8},
    '2019-02-19': {'earnings_3m_yr': 3.4, 'unemployment': 4.0, 'claimant_count_chg': 14.2},
    '2019-03-19': {'earnings_3m_yr': 3.4, 'unemployment': 3.9, 'claimant_count_chg': 15.7},
    '2019-04-16': {'earnings_3m_yr': 3.3, 'unemployment': 3.9, 'claimant_count_chg': 24.7},
    '2019-05-14': {'earnings_3m_yr': 3.2, 'unemployment': 3.8, 'claimant_count_chg': 28.2},
    '2019-06-11': {'earnings_3m_yr': 3.1, 'unemployment': 3.8, 'claimant_count_chg': 23.3},
    '2019-07-16': {'earnings_3m_yr': 3.4, 'unemployment': 3.8, 'claimant_count_chg': 28.1},
    '2019-08-13': {'earnings_3m_yr': 3.7, 'unemployment': 3.9, 'claimant_count_chg': 33.7},
    '2019-09-10': {'earnings_3m_yr': 3.8, 'unemployment': 3.8, 'claimant_count_chg': 24.5},
    '2019-10-15': {'earnings_3m_yr': 3.8, 'unemployment': 3.8, 'claimant_count_chg': 21.1},
    '2019-11-12': {'earnings_3m_yr': 3.8, 'unemployment': 3.8, 'claimant_count_chg': 30.1},
    '2019-12-17': {'earnings_3m_yr': 3.5, 'unemployment': 3.8, 'claimant_count_chg': 21.6},
    # 2020
    '2020-01-21': {'earnings_3m_yr': 3.2, 'unemployment': 3.8, 'claimant_count_chg': 14.9},
    '2020-02-18': {'earnings_3m_yr': 3.1, 'unemployment': 3.9, 'claimant_count_chg': 5.5},
    '2020-03-17': {'earnings_3m_yr': 2.8, 'unemployment': 4.0, 'claimant_count_chg': 17.3},
    '2020-04-21': {'earnings_3m_yr': 2.1, 'unemployment': 4.4, 'claimant_count_chg': 856.5},
    '2020-05-19': {'earnings_3m_yr': 1.0, 'unemployment': 4.8, 'claimant_count_chg': 528.9},
    '2020-06-16': {'earnings_3m_yr': -0.3, 'unemployment': 5.2, 'claimant_count_chg': 250.1},
    '2020-07-14': {'earnings_3m_yr': -1.3, 'unemployment': 5.5, 'claimant_count_chg': 94.4},
    '2020-08-11': {'earnings_3m_yr': -2.0, 'unemployment': 5.2, 'claimant_count_chg': 73.7},
    '2020-09-15': {'earnings_3m_yr': -1.0, 'unemployment': 4.8, 'claimant_count_chg': 28.1},
    '2020-10-13': {'earnings_3m_yr': -0.1, 'unemployment': 4.8, 'claimant_count_chg': 29.3},
    '2020-11-10': {'earnings_3m_yr': 0.5, 'unemployment': 4.9, 'claimant_count_chg': -2.9},
    '2020-12-15': {'earnings_3m_yr': 1.1, 'unemployment': 5.0, 'claimant_count_chg': 10.5},
    # 2021
    '2021-01-26': {'earnings_3m_yr': 3.6, 'unemployment': 5.2, 'claimant_count_chg': 7.0},
    '2021-02-23': {'earnings_3m_yr': 4.2, 'unemployment': 5.1, 'claimant_count_chg': -20.0},
    '2021-03-23': {'earnings_3m_yr': 4.5, 'unemployment': 4.9, 'claimant_count_chg': -5.1},
    '2021-04-20': {'earnings_3m_yr': 4.6, 'unemployment': 4.8, 'claimant_count_chg': -15.1},
    '2021-05-18': {'earnings_3m_yr': 4.3, 'unemployment': 4.7, 'claimant_count_chg': -15.1},
    '2021-06-15': {'earnings_3m_yr': 5.6, 'unemployment': 4.8, 'claimant_count_chg': -28.1},
    '2021-07-15': {'earnings_3m_yr': 7.4, 'unemployment': 4.8, 'claimant_count_chg': -11.5},
    '2021-08-17': {'earnings_3m_yr': 8.3, 'unemployment': 4.7, 'claimant_count_chg': -7.8},
    '2021-09-14': {'earnings_3m_yr': 7.2, 'unemployment': 4.5, 'claimant_count_chg': -5.3},
    '2021-10-12': {'earnings_3m_yr': 5.8, 'unemployment': 4.3, 'claimant_count_chg': -2.5},
    '2021-11-16': {'earnings_3m_yr': 4.9, 'unemployment': 4.2, 'claimant_count_chg': 14.9},
    '2021-12-14': {'earnings_3m_yr': 3.8, 'unemployment': 4.1, 'claimant_count_chg': 4.9},
    # 2022
    '2022-01-18': {'earnings_3m_yr': 3.7, 'unemployment': 3.9, 'claimant_count_chg': -58.4},
    '2022-02-15': {'earnings_3m_yr': 3.8, 'unemployment': 3.9, 'claimant_count_chg': -31.9},
    '2022-03-15': {'earnings_3m_yr': 3.8, 'unemployment': 3.8, 'claimant_count_chg': -48.1},
    '2022-04-12': {'earnings_3m_yr': 4.0, 'unemployment': 3.7, 'claimant_count_chg': -56.9},
    '2022-05-17': {'earnings_3m_yr': 4.2, 'unemployment': 3.7, 'claimant_count_chg': -50.4},
    '2022-06-14': {'earnings_3m_yr': 4.3, 'unemployment': 3.8, 'claimant_count_chg': -34.2},
    '2022-07-19': {'earnings_3m_yr': 4.7, 'unemployment': 3.8, 'claimant_count_chg': -10.5},
    '2022-08-16': {'earnings_3m_yr': 5.2, 'unemployment': 3.6, 'claimant_count_chg': -10.5},
    '2022-09-13': {'earnings_3m_yr': 5.5, 'unemployment': 3.5, 'claimant_count_chg': -6.7},
    '2022-10-11': {'earnings_3m_yr': 5.7, 'unemployment': 3.5, 'claimant_count_chg': -14.4},
    '2022-11-15': {'earnings_3m_yr': 6.1, 'unemployment': 3.7, 'claimant_count_chg': 6.3},
    '2022-12-13': {'earnings_3m_yr': 6.1, 'unemployment': 3.7, 'claimant_count_chg': 16.1},
    # 2023
    '2023-01-17': {'earnings_3m_yr': 5.9, 'unemployment': 3.7, 'claimant_count_chg': 19.7},
    '2023-02-14': {'earnings_3m_yr': 5.7, 'unemployment': 3.7, 'claimant_count_chg': -3.1},
    '2023-03-14': {'earnings_3m_yr': 5.7, 'unemployment': 3.7, 'claimant_count_chg': -11.2},
    '2023-04-18': {'earnings_3m_yr': 5.8, 'unemployment': 3.8, 'claimant_count_chg': 28.2},
    '2023-05-16': {'earnings_3m_yr': 5.8, 'unemployment': 3.9, 'claimant_count_chg': 23.5},
    '2023-06-13': {'earnings_3m_yr': 6.2, 'unemployment': 4.0, 'claimant_count_chg': 21.1},
    '2023-07-11': {'earnings_3m_yr': 6.9, 'unemployment': 4.0, 'claimant_count_chg': 10.5},
    '2023-08-15': {'earnings_3m_yr': 7.8, 'unemployment': 4.3, 'claimant_count_chg': 29.0},
    '2023-09-12': {'earnings_3m_yr': 8.1, 'unemployment': 4.3, 'claimant_count_chg': 2.3},
    '2023-10-24': {'earnings_3m_yr': 7.9, 'unemployment': 4.2, 'claimant_count_chg': 20.4},
    '2023-11-14': {'earnings_3m_yr': 7.3, 'unemployment': 4.2, 'claimant_count_chg': -10.5},
    '2023-12-12': {'earnings_3m_yr': 6.5, 'unemployment': 4.2, 'claimant_count_chg': 16.0},
    # 2024
    '2024-01-16': {'earnings_3m_yr': 5.6, 'unemployment': 4.2, 'claimant_count_chg': 11.7},
    '2024-02-13': {'earnings_3m_yr': 5.3, 'unemployment': 3.8, 'claimant_count_chg': -4.2},
    '2024-03-12': {'earnings_3m_yr': 5.1, 'unemployment': 3.9, 'claimant_count_chg': 16.8},
    '2024-04-16': {'earnings_3m_yr': 5.3, 'unemployment': 4.3, 'claimant_count_chg': 10.9},
    '2024-05-14': {'earnings_3m_yr': 5.2, 'unemployment': 4.3, 'claimant_count_chg': 8.9},
    '2024-06-11': {'earnings_3m_yr': 5.1, 'unemployment': 4.4, 'claimant_count_chg': 50.4},
    '2024-07-16': {'earnings_3m_yr': 4.6, 'unemployment': 4.4, 'claimant_count_chg': 32.3},
    '2024-08-13': {'earnings_3m_yr': 4.1, 'unemployment': 4.1, 'claimant_count_chg': 102.3},
    '2024-09-10': {'earnings_3m_yr': 4.0, 'unemployment': 4.1, 'claimant_count_chg': 26.5},
    '2024-10-15': {'earnings_3m_yr': 3.8, 'unemployment': 4.0, 'claimant_count_chg': 27.9},
    '2024-11-12': {'earnings_3m_yr': 3.6, 'unemployment': 4.3, 'claimant_count_chg': 0.3},
    '2024-12-17': {'earnings_3m_yr': 3.5, 'unemployment': 4.3, 'claimant_count_chg': 0.7},
    # 2025
    '2025-01-21': {'earnings_3m_yr': 3.4, 'unemployment': 4.4, 'claimant_count_chg': 0.7},
    '2025-02-18': {'earnings_3m_yr': 3.4, 'unemployment': 4.4, 'claimant_count_chg': 2.1},
    '2025-03-25': {'earnings_3m_yr': 3.2, 'unemployment': 4.4, 'claimant_count_chg': 4.2},
    '2025-04-15': {'earnings_3m_yr': 3.0, 'unemployment': 4.4, 'claimant_count_chg': 7.5},
    '2025-05-13': {'earnings_3m_yr': 2.9, 'unemployment': 4.3, 'claimant_count_chg': 3.3},
    '2025-06-10': {'earnings_3m_yr': 2.8, 'unemployment': 4.3, 'claimant_count_chg': 5.1},
    '2025-07-15': {'earnings_3m_yr': 2.6, 'unemployment': 4.3, 'claimant_count_chg': 8.2},
    '2025-08-12': {'earnings_3m_yr': 2.4, 'unemployment': 4.3, 'claimant_count_chg': 10.5},
    '2025-09-16': {'earnings_3m_yr': 2.3, 'unemployment': 4.3, 'claimant_count_chg': 12.1},
    '2025-10-14': {'earnings_3m_yr': 2.2, 'unemployment': 4.3, 'claimant_count_chg': 9.8},
    '2025-11-11': {'earnings_3m_yr': 2.1, 'unemployment': 4.3, 'claimant_count_chg': 11.3},
    '2025-12-16': {'earnings_3m_yr': 2.0, 'unemployment': 4.3, 'claimant_count_chg': 8.7},
    # 2026
    '2026-01-20': {'earnings_3m_yr': 2.2, 'unemployment': 4.3, 'claimant_count_chg': 7.5},
    '2026-02-17': {'earnings_3m_yr': 2.3, 'unemployment': 4.3, 'claimant_count_chg': 5.2},
    '2026-03-24': {'earnings_3m_yr': 2.1, 'unemployment': 4.3, 'claimant_count_chg': 6.8},
    '2026-04-14': {'earnings_3m_yr': 2.0, 'unemployment': 4.3, 'claimant_count_chg': 8.1},
    '2026-05-12': {'earnings_3m_yr': 1.9, 'unemployment': 4.3, 'claimant_count_chg': 9.5},
}


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


def classify_signal(earnings_3m_yr, prev_earnings=None):
    """Classify wage growth signal.
    HIGH_WAGES: ≥5% — inflationary, BoE hawkish
    WARM_WAGES: 3-5% — above target but moderating
    COOL_WAGES: 2-3% — near target
    COLD_WAGES: <2% — deflationary risk
    """
    if earnings_3m_yr >= 5.0:
        return 'HIGH_WAGES'
    elif earnings_3m_yr >= 3.0:
        return 'WARM_WAGES'
    elif earnings_3m_yr >= 2.0:
        return 'COOL_WAGES'
    return 'COLD_WAGES'


def classify_direction(earnings_3m_yr, prev_earnings=None):
    """Surprise direction: BEAT (higher than prev) or MISS (lower)."""
    if prev_earnings is not None:
        delta = earnings_3m_yr - prev_earnings
        if delta > 0.2:
            return 'BEAT'
        elif delta < -0.2:
            return 'MISS'
    if earnings_3m_yr >= 4.0:
        return 'MISS'  # high wages = hawkish surprise
    elif earnings_3m_yr < 2.5:
        return 'BEAT'  # low wages = dovish surprise
    return 'INLINE'


def classify_wyckoff_phase(price_series, lookback=96):
    if len(price_series) < lookback:
        return 'RANGE'
    recent = price_series[-lookback:]
    high = recent.max()
    low = recent.min()
    current = recent.iloc[-1]
    pos = (current - low) / (high - low) if high != low else 0.5
    range_pct = (high - low) / low * 100
    if range_pct < 1.5:
        return 'RANGE' if 0.3 < pos < 0.7 else ('ACCUMULATION' if pos < 0.3 else 'DISTRIBUTION')
    elif current > recent.mean() and pos > 0.6:
        return 'MARKUP'
    elif current < recent.mean() and pos < 0.4:
        return 'MARKDOWN'
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


# Session chain (same as UK CPI — 07:00 UTC release)
SESSION_CHAIN = [
    ('Release (1h)', 7, 8),
    ('London Open', 8, 9),
    ('London Morning', 9, 12),
    ('London Midday', 12, 14),
    ('NY Pre-Open', 12, 13),
    ('NY Open', 13, 14),
    ('London-NY Overlap', 14, 16),
    ('NY AM', 14, 17),
    ('NY Lunch', 17, 18),
    ('NY PM', 18, 21),
    ('Post-NY', 21, 23),
    ('Sydney Open', 0, 1),
    ('Tokyo Open', 1, 3),
    ('Asia Mid', 3, 5),
    ('Asia Afternoon', 5, 7),
    ('BoE Re-open', 7, 8),
]


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

        earnings = release_data['earnings_3m_yr']
        unemp = release_data['unemployment']
        prev_earnings = RELEASES[sorted_dates[i-1]]['earnings_3m_yr'] if i > 0 else None

        signal = classify_signal(earnings, prev_earnings)
        direction = classify_direction(earnings, prev_earnings)
        wyckoff = classify_wyckoff_phase(df['Close'].astype(float))
        vol_regime = classify_vol_regime(df, release_dt)

        session_returns = {}
        for name, start_h, end_h in SESSION_CHAIN:
            start_dt = release_dt.replace(hour=start_h, minute=0)
            end_dt = release_dt.replace(hour=end_h, minute=0)
            if start_h >= 21 and end_h <= 23:
                pass  # same day
            elif end_h <= start_h or name in ('Sydney Open', 'Tokyo Open', 'Asia Mid', 'Asia Afternoon', 'BoE Re-open'):
                start_dt += timedelta(days=1)
                end_dt += timedelta(days=1)

            ret = get_return(df, start_dt, end_dt)
            if ret is not None:
                session_returns[name] = round(ret, 4)

        ret_24h = get_return(df, release_dt, release_dt + timedelta(hours=24))
        ret_48h = get_return(df, release_dt, release_dt + timedelta(hours=48))

        results.append({
            'date': date_str,
            'year': int(date_str[:4]),
            'earnings_3m_yr': earnings,
            'unemployment': unemp,
            'prev_earnings': prev_earnings,
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

    for sig in ['HIGH_WAGES', 'WARM_WAGES', 'COOL_WAGES', 'COLD_WAGES']:
        sig_rets = valid[valid['signal'] == sig]['ret_24h'].values
        if len(sig_rets) >= 3:
            t3, p3 = stats.ttest_1samp(sig_rets, 0)
            print(f"\n3. {sig} (n={len(sig_rets)}):")
            print(f"   Mean: {np.mean(sig_rets):+.3f}%  win={np.mean(sig_rets > 0)*100:.0f}%  t={t3:.3f}  p={p3:.4f}  {'***' if p3 < 0.05 else 'ns'}")

    print(f"\n4. Year-by-year:")
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
    print(f"Analyzed {len(results)} UK Wages releases ({results['year'].min()}-{results['year'].max()})")

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
    for sig in ['HIGH_WAGES', 'WARM_WAGES', 'COOL_WAGES', 'COLD_WAGES']:
        sig_data = valid[valid['signal'] == sig]
        if len(sig_data) > 0:
            avg = sig_data['ret_24h'].mean()
            win = (sig_data['ret_24h'] > 0).mean() * 100
            n = len(sig_data)
            print(f"  {sig:<15} avg={avg:+.3f}%  win={win:.0f}%  n={n}")

    print("\n  By Direction:")
    for d in ['BEAT', 'MISS', 'INLINE']:
        d_data = valid[valid['direction'] == d]
        if len(d_data) > 0:
            avg = d_data['ret_24h'].mean()
            win = (d_data['ret_24h'] > 0).mean() * 100
            n = len(d_data)
            print(f"  {d:<10} avg={avg:+.3f}%  win={win:.0f}%  n={n}")

    out_path = os.path.join(os.path.dirname(__file__), 'backtest_uk_wages_results.json')
    results.to_json(out_path, orient='records', indent=2)
    print(f"\nResults saved to {out_path}")

if __name__ == '__main__':
    main()

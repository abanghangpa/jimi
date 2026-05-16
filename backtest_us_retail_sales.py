#!/usr/bin/env python3
"""
Prompt A + B: Backtest US Retail Sales (2018-today) using ETH/USDT 15m data.

US Retail Sales released by Census Bureau at 08:30 ET (13:30 UTC) around 15th of each month.
Measures raw consumer spending (~70% of US GDP).

Session itinerary (per user's thesis #16):
  US Morning (13:30 UTC) → Equity Open → Late Month PCE

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
# US RETAIL SALES RELEASE DATES (08:30 ET = 13:30 UTC)
# Format: {date: {'retail_sales_mom': float, 'core_mom': float}}
# retail_sales_mom = month-over-month % change (advance estimate)
# core_mom = ex-autos month-over-month % change
# ═══════════════════════════════════════════════════════════════

RELEASES = {
    # 2018
    '2018-01-12': {'retail_sales_mom': 0.4, 'core_mom': 0.4},
    '2018-02-14': {'retail_sales_mom': -0.3, 'core_mom': 0.0},
    '2018-03-14': {'retail_sales_mom': -0.1, 'core_mom': 0.1},
    '2018-04-16': {'retail_sales_mom': 0.6, 'core_mom': 0.4},
    '2018-05-15': {'retail_sales_mom': 0.3, 'core_mom': 0.4},
    '2018-06-14': {'retail_sales_mom': 0.8, 'core_mom': 0.5},
    '2018-07-16': {'retail_sales_mom': 0.5, 'core_mom': 0.4},
    '2018-08-15': {'retail_sales_mom': 0.5, 'core_mom': 0.6},
    '2018-09-14': {'retail_sales_mom': 0.1, 'core_mom': 0.1},
    '2018-10-15': {'retail_sales_mom': 0.1, 'core_mom': 0.0},
    '2018-11-15': {'retail_sales_mom': 0.2, 'core_mom': 0.3},
    '2018-12-14': {'retail_sales_mom': 0.2, 'core_mom': 0.2},
    # 2019
    '2019-01-16': {'retail_sales_mom': -1.2, 'core_mom': -1.7},
    '2019-02-14': {'retail_sales_mom': 0.2, 'core_mom': 0.9},
    '2019-03-11': {'retail_sales_mom': -0.2, 'core_mom': -0.4},
    '2019-04-18': {'retail_sales_mom': 1.6, 'core_mom': 1.2},
    '2019-05-15': {'retail_sales_mom': 0.2, 'core_mom': 0.1},
    '2019-06-14': {'retail_sales_mom': 0.5, 'core_mom': 0.5},
    '2019-07-16': {'retail_sales_mom': 0.4, 'core_mom': 0.7},
    '2019-08-15': {'retail_sales_mom': 0.7, 'core_mom': 1.0},
    '2019-09-13': {'retail_sales_mom': 0.4, 'core_mom': 0.0},
    '2019-10-16': {'retail_sales_mom': -0.3, 'core_mom': -0.1},
    '2019-11-15': {'retail_sales_mom': 0.3, 'core_mom': 0.3},
    '2019-12-13': {'retail_sales_mom': 0.2, 'core_mom': 0.1},
    # 2020
    '2020-01-16': {'retail_sales_mom': 0.3, 'core_mom': 0.3},
    '2020-02-14': {'retail_sales_mom': 0.5, 'core_mom': 0.4},
    '2020-03-17': {'retail_sales_mom': -8.3, 'core_mom': -4.0},
    '2020-04-15': {'retail_sales_mom': -14.7, 'core_mom': -11.5},
    '2020-05-15': {'retail_sales_mom': 17.7, 'core_mom': 12.4},
    '2020-06-16': {'retail_sales_mom': 7.5, 'core_mom': 7.3},
    '2020-07-16': {'retail_sales_mom': 1.2, 'core_mom': 1.9},
    '2020-08-14': {'retail_sales_mom': 0.6, 'core_mom': 0.7},
    '2020-09-16': {'retail_sales_mom': 0.6, 'core_mom': 0.7},
    '2020-10-16': {'retail_sales_mom': 0.3, 'core_mom': 0.2},
    '2020-11-17': {'retail_sales_mom': -1.1, 'core_mom': -0.8},
    '2020-12-16': {'retail_sales_mom': -1.4, 'core_mom': -1.3},
    # 2021
    '2021-01-15': {'retail_sales_mom': -0.7, 'core_mom': -1.4},
    '2021-02-17': {'retail_sales_mom': 5.3, 'core_mom': 5.3},
    '2021-03-16': {'retail_sales_mom': -2.7, 'core_mom': -3.2},
    '2021-04-15': {'retail_sales_mom': 9.8, 'core_mom': 8.4},
    '2021-05-14': {'retail_sales_mom': 0.0, 'core_mom': -0.8},
    '2021-06-15': {'retail_sales_mom': -1.7, 'core_mom': -1.0},
    '2021-07-16': {'retail_sales_mom': 0.6, 'core_mom': 1.1},
    '2021-08-17': {'retail_sales_mom': -1.1, 'core_mom': -0.5},
    '2021-09-16': {'retail_sales_mom': 0.7, 'core_mom': 0.8},
    '2021-10-15': {'retail_sales_mom': 0.7, 'core_mom': 0.6},
    '2021-11-16': {'retail_sales_mom': 1.7, 'core_mom': 1.7},
    '2021-12-15': {'retail_sales_mom': 0.3, 'core_mom': 0.3},
    # 2022
    '2022-01-14': {'retail_sales_mom': -1.9, 'core_mom': -2.3},
    '2022-02-16': {'retail_sales_mom': 3.8, 'core_mom': 3.3},
    '2022-03-16': {'retail_sales_mom': 0.3, 'core_mom': 0.2},
    '2022-04-14': {'retail_sales_mom': 0.9, 'core_mom': 0.6},
    '2022-05-17': {'retail_sales_mom': 0.9, 'core_mom': 0.6},
    '2022-06-15': {'retail_sales_mom': -0.3, 'core_mom': 0.5},
    '2022-07-15': {'retail_sales_mom': 1.0, 'core_mom': 0.4},
    '2022-08-17': {'retail_sales_mom': -0.4, 'core_mom': 0.1},
    '2022-09-15': {'retail_sales_mom': 0.4, 'core_mom': 0.0},
    '2022-10-14': {'retail_sales_mom': 1.3, 'core_mom': 0.4},
    '2022-11-16': {'retail_sales_mom': -0.6, 'core_mom': -0.2},
    '2022-12-15': {'retail_sales_mom': -1.1, 'core_mom': -0.7},
    # 2023
    '2023-01-18': {'retail_sales_mom': -1.1, 'core_mom': -0.8},
    '2023-02-15': {'retail_sales_mom': 3.0, 'core_mom': 2.3},
    '2023-03-15': {'retail_sales_mom': -0.4, 'core_mom': -0.1},
    '2023-04-14': {'retail_sales_mom': -1.0, 'core_mom': -0.4},
    '2023-05-16': {'retail_sales_mom': 0.4, 'core_mom': 0.5},
    '2023-06-15': {'retail_sales_mom': 0.3, 'core_mom': 0.2},
    '2023-07-18': {'retail_sales_mom': 0.2, 'core_mom': 0.6},
    '2023-08-15': {'retail_sales_mom': 0.7, 'core_mom': 0.4},
    '2023-09-14': {'retail_sales_mom': 0.6, 'core_mom': 0.3},
    '2023-10-17': {'retail_sales_mom': 0.7, 'core_mom': 0.6},
    '2023-11-15': {'retail_sales_mom': -0.1, 'core_mom': 0.1},
    '2023-12-14': {'retail_sales_mom': 0.3, 'core_mom': 0.2},
    # 2024
    '2024-01-17': {'retail_sales_mom': 0.6, 'core_mom': 0.2},
    '2024-02-15': {'retail_sales_mom': -0.8, 'core_mom': -0.4},
    '2024-03-14': {'retail_sales_mom': 0.6, 'core_mom': 0.3},
    '2024-04-15': {'retail_sales_mom': 0.4, 'core_mom': 0.4},
    '2024-05-15': {'retail_sales_mom': 0.1, 'core_mom': 0.2},
    '2024-06-18': {'retail_sales_mom': 0.1, 'core_mom': 0.4},
    '2024-07-16': {'retail_sales_mom': 0.0, 'core_mom': 0.4},
    '2024-08-15': {'retail_sales_mom': 1.0, 'core_mom': 0.4},
    '2024-09-17': {'retail_sales_mom': 0.4, 'core_mom': 0.5},
    '2024-10-17': {'retail_sales_mom': 0.4, 'core_mom': 0.1},
    '2024-11-15': {'retail_sales_mom': 0.4, 'core_mom': 0.7},
    '2024-12-17': {'retail_sales_mom': 0.4, 'core_mom': 0.4},
    # 2025
    '2025-01-16': {'retail_sales_mom': 0.4, 'core_mom': 0.7},
    '2025-02-14': {'retail_sales_mom': -0.9, 'core_mom': -0.3},
    '2025-03-17': {'retail_sales_mom': 0.2, 'core_mom': 0.3},
    '2025-04-16': {'retail_sales_mom': 1.4, 'core_mom': 0.4},
    '2025-05-15': {'retail_sales_mom': 0.1, 'core_mom': 0.2},
    '2025-06-17': {'retail_sales_mom': -0.1, 'core_mom': 0.3},
    '2025-07-16': {'retail_sales_mom': 0.5, 'core_mom': 0.3},
    '2025-08-15': {'retail_sales_mom': 0.2, 'core_mom': 0.4},
    '2025-09-16': {'retail_sales_mom': 0.3, 'core_mom': 0.2},
    '2025-10-16': {'retail_sales_mom': 0.1, 'core_mom': 0.3},
    '2025-11-14': {'retail_sales_mom': 0.2, 'core_mom': 0.1},
    '2025-12-16': {'retail_sales_mom': 0.3, 'core_mom': 0.2},
    # 2026
    '2026-01-15': {'retail_sales_mom': 0.4, 'core_mom': 0.3},
    '2026-02-13': {'retail_sales_mom': -0.2, 'core_mom': 0.1},
    '2026-03-16': {'retail_sales_mom': 0.3, 'core_mom': 0.2},
    '2026-04-15': {'retail_sales_mom': 0.5, 'core_mom': 0.4},
    '2026-05-14': {'retail_sales_mom': 0.2, 'core_mom': 0.3},
}

# Consensus expectations (MoM %) — approximate, for surprise calculation
CONSENSUS = {
    '2018-01-12': 0.5, '2018-02-14': 0.2, '2018-03-14': 0.3, '2018-04-16': 0.4,
    '2018-05-15': 0.4, '2018-06-14': 0.4, '2018-07-16': 0.1, '2018-08-15': 0.1,
    '2018-09-14': 0.7, '2018-10-15': 0.6, '2018-11-15': 0.5, '2018-12-14': 0.1,
    '2019-01-16': 0.0, '2019-02-14': 0.0, '2019-03-11': 0.0, '2019-04-18': 1.1,
    '2019-05-15': 0.2, '2019-06-14': 0.6, '2019-07-16': 0.1, '2019-08-15': 0.3,
    '2019-09-13': 0.2, '2019-10-16': 0.3, '2019-11-15': 0.3, '2019-12-13': 0.5,
    '2020-01-16': 0.3, '2020-02-14': 0.2, '2020-03-17': 0.2, '2020-04-15': -8.0,
    '2020-05-15': 8.0, '2020-06-16': 5.4, '2020-07-16': 1.9, '2020-08-14': 1.0,
    '2020-09-16': 0.8, '2020-10-16': 0.5, '2020-11-17': -0.3, '2020-12-16': -0.3,
    '2021-01-15': 0.7, '2021-02-17': 1.0, '2021-03-16': -0.5, '2021-04-15': 5.9,
    '2021-05-14': 0.2, '2021-06-15': -0.4, '2021-07-16': -0.3, '2021-08-17': -0.3,
    '2021-09-16': 0.2, '2021-10-15': 0.3, '2021-11-16': 1.1, '2021-12-15': 0.8,
    '2022-01-14': 0.0, '2022-02-16': 2.1, '2022-03-16': 0.6, '2022-04-14': 0.9,
    '2022-05-17': 0.7, '2022-06-15': 0.2, '2022-07-15': 0.8, '2022-08-17': 0.1,
    '2022-09-15': 0.2, '2022-10-14': 0.3, '2022-11-16': 0.6, '2022-12-15': -0.1,
    '2023-01-18': -0.1, '2023-02-15': 1.9, '2023-03-15': -0.3, '2023-04-14': -0.4,
    '2023-05-16': 0.8, '2023-06-15': 0.1, '2023-07-18': 0.3, '2023-08-15': 0.4,
    '2023-09-14': 0.2, '2023-10-17': 0.3, '2023-11-15': -0.3, '2023-12-14': 0.1,
    '2024-01-17': 0.4, '2024-02-15': -0.1, '2024-03-14': 0.4, '2024-04-15': 0.4,
    '2024-05-15': 0.4, '2024-06-18': 0.3, '2024-07-16': 0.1, '2024-08-15': 0.4,
    '2024-09-17': 0.2, '2024-10-17': 0.3, '2024-11-15': 0.3, '2024-12-17': 0.5,
    '2025-01-16': 0.6, '2025-02-14': -0.1, '2025-03-17': 0.6, '2025-04-16': 1.3,
    '2025-05-15': 0.0, '2025-06-17': 0.3, '2025-07-16': 0.1, '2025-08-15': 0.5,
    '2025-09-16': 0.2, '2025-10-16': 0.3, '2025-11-14': 0.2, '2025-12-16': 0.4,
    '2026-01-15': 0.3, '2026-02-13': 0.3, '2026-03-16': 0.2, '2026-04-15': 0.3,
    '2026-05-14': 0.3,
}


# ═══════════════════════════════════════════════════════════════
# SESSION DEFINITIONS (US Retail Sales at 13:30 UTC)
# ═══════════════════════════════════════════════════════════════

SESSION_CHAIN = [
    # Pre-Asia
    ('Pre-Asia', [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]),  # 00:00-13:30 (before release)

    # Asia (release happens during Asia afternoon)
    ('Sydney Open', [22, 23, 0]),     # 22:00-01:00 UTC (prev day)
    ('Tokyo Open', [0, 1, 2]),        # 00:00-03:00
    ('Asia Mid', [3, 4]),             # 03:00-05:00
    ('Asia Afternoon', [5, 6]),       # 05:00-07:00
    ('Tokyo Close', [6, 7]),          # 06:00-08:00
    ('Pre-London', [7, 8]),           # 07:00-09:00

    # Europe
    ('Frankfurt Open', [7, 8]),       # 07:00-09:00
    ('London Open', [8, 9]),          # 08:00-10:00
    ('London Morning', [9, 10, 11]),  # 09:00-12:00
    ('London Midday', [12, 13]),      # 12:00-14:00

    # Overlap (EU-US) — release happens here
    ('NY Pre-Open', [12, 13]),        # 12:00-14:00 (release at 13:30)
    ('NY Open', [13, 14]),            # 13:30-15:00 (first bars after release)
    ('London-NY Overlap', [14, 15]),  # 14:00-16:00

    # New York
    ('NY AM', [14, 15, 16]),          # 14:00-17:00
    ('NY Lunch', [17]),               # 17:00-18:00
    ('NY PM', [18, 19, 20]),          # 18:00-21:00

    # Post-release windows
    ('Release (1h)', [13, 14]),       # 13:30-15:00 (1h post-release)
    ('Release (4h)', [13, 14, 15, 16, 17]),  # 13:30-18:00 (4h post-release)
]


def load_eth_data(csv_path):
    """Load and prepare ETH 15m data."""
    df = pd.read_csv(csv_path)
    df['Open time'] = pd.to_datetime(df['Open time'])
    df = df.sort_values('Open time').reset_index(drop=True)
    return df


def get_bar_at(df, dt):
    """Get the bar that contains or is closest before dt."""
    mask = df['Open time'] <= dt
    if mask.sum() == 0:
        return None
    return df[mask].iloc[-1]


def get_bars_between(df, start, end):
    """Get bars between two datetimes."""
    mask = (df['Open time'] >= start) & (df['Open time'] < end)
    return df[mask]


def get_return(df, start_dt, end_dt):
    """Calculate return between two timestamps."""
    bar_start = get_bar_at(df, start_dt)
    bars_end = get_bars_between(df, start_dt, end_dt)
    if bar_start is None or len(bars_end) == 0:
        return None
    start_price = float(bar_start['Close'])
    end_price = float(bars_end.iloc[-1]['Close'])
    return (end_price - start_price) / start_price * 100


def classify_wyckoff_phase(df, release_dt, lookback=96):
    """Simple Wyckoff phase classification based on recent price action.
    Uses 96 bars = 24h of 15m data before the release.
    """
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
    """Classify volatility regime using M9-like logic."""
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


def classify_signal(retail_sales_mom, core_mom, consensus):
    """Classify the retail sales signal.
    Strong beat → consumer resilient → Fed hawkish → mixed for ETH
    Miss → consumer weak → Fed dovish → mixed for ETH
    The sign depends on the macro regime.
    """
    surprise = retail_sales_mom - consensus if consensus is not None else 0

    if surprise > 0.5:
        return 'STRONG_BEAT', surprise
    elif surprise > 0.1:
        return 'BEAT', surprise
    elif surprise < -0.5:
        return 'BIG_MISS', surprise
    elif surprise < -0.1:
        return 'MISS', surprise
    else:
        return 'INLINE', surprise


def run_backtest(csv_path):
    """Run the full backtest."""
    df = load_eth_data(csv_path)
    results = []

    sorted_dates = sorted(RELEASES.keys())

    for i, date_str in enumerate(sorted_dates):
        release_data = RELEASES[date_str]
        release_dt = datetime.strptime(date_str, '%Y-%m-%d').replace(hour=13, minute=30)

        # Skip if no data available
        bars_after = get_bars_between(df, release_dt, release_dt + timedelta(hours=24))
        if len(bars_after) < 4:
            continue

        # Pre-release price
        pre_bar = get_bar_at(df, release_dt)
        if pre_bar is None:
            continue
        pre_price = float(pre_bar['Close'])

        # Classify
        retail_mom = release_data['retail_sales_mom']
        core = release_data['core_mom']
        cons = CONSENSUS.get(date_str)
        signal, surprise = classify_signal(retail_mom, core, cons)

        wyckoff = classify_wyckoff_phase(df, release_dt)
        vol_regime = classify_vol_regime(df, release_dt)

        # Session returns
        session_returns = {}
        for name, hours in SESSION_CHAIN:
            if len(hours) == 1:
                start = release_dt.replace(hour=hours[0], minute=0)
                end = start + timedelta(hours=1)
            else:
                start = release_dt.replace(hour=hours[0], minute=0)
                end = release_dt.replace(hour=hours[-1] + 1, minute=0)

            # Handle cross-day
            if end <= start:
                end += timedelta(days=1)

            ret = get_return(df, start, end)
            if ret is not None:
                session_returns[name] = round(ret, 4)

        # 24h aggregate
        ret_24h = get_return(df, release_dt, release_dt + timedelta(hours=24))
        ret_48h = get_return(df, release_dt, release_dt + timedelta(hours=48))

        year = int(date_str[:4])

        results.append({
            'date': date_str,
            'year': year,
            'retail_sales_mom': retail_mom,
            'core_mom': core,
            'consensus': cons,
            'surprise': round(surprise, 2),
            'signal': signal,
            'wyckoff': wyckoff,
            'vol_regime': vol_regime,
            'pre_price': round(pre_price, 2),
            'ret_24h': round(ret_24h, 4) if ret_24h is not None else None,
            'ret_48h': round(ret_48h, 4) if ret_48h is not None else None,
            'session_returns': session_returns,
        })

    return pd.DataFrame(results)


def cross_tabulate(results):
    """Cross-tabulate: (signal × vol × wyckoff) → avg 24h return, win rate, n."""
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
        med_ret = group['ret_24h'].median()
        std_ret = group['ret_24h'].std()
        rows.append({
            'signal': sig, 'vol': vol, 'wyckoff': wy,
            'n': n, 'avg_24h': round(avg_ret, 3),
            'win_rate': round(win_rate, 3),
            'median_24h': round(med_ret, 3),
            'std': round(std_ret, 3) if n > 1 else 0,
        })

    df = pd.DataFrame(rows).sort_values('n', ascending=False)
    print(df.to_string(index=False))

    # Filter for edges (n>=3, |avg|>=0.5%)
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
    """Analyze session-by-session transmission chain (Prompt B).

    Measures direction persistence between sessions on release days.
    % same direction until reopen of the original session's market.
    If any transition is <55% same direction, the chain doesn't hold.
    If >65%, it's a real edge. Between 55-65% is marginal.
    """
    print("\n" + "="*80)
    print("SESSION TRANSMISSION CHAIN (Prompt B)")
    print("="*80)

    valid = results[results['ret_24h'].notna()]

    # Key sessions for US Retail Sales
    key_sessions = [
        'Release (1h)', 'NY Open', 'London-NY Overlap',
        'NY AM', 'NY Lunch', 'NY PM',
    ]

    print(f"\n{'Session':<25} {'Avg Ret%':>10} {'Win%':>8} {'n':>5} {'Dir':>6}")
    print("-"*60)

    chain_data = []
    for session_name in key_sessions:
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
        curr_dir = 'UP' if avg > 0.05 else 'DOWN' if avg < -0.05 else 'FLAT'

        print(f"{session_name:<25} {avg:>+10.3f} {win:>7.0f}% {n:>5} {curr_dir:>6}")

        chain_data.append({
            'session': session_name,
            'avg_ret': round(avg, 3),
            'win_pct': round(win, 1),
            'n': n,
            'direction': curr_dir,
        })

    # Direction persistence between consecutive sessions
    print(f"\n{'Transition':<40} {'Same Dir%':>10} {'n':>5} {'Verdict':>10}")
    print("-"*70)

    transitions = []
    for i in range(len(key_sessions) - 1):
        s1 = key_sessions[i]
        s2 = key_sessions[i + 1]

        same_count = 0
        total = 0
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
            print(f"{s1 + ' → ' + s2:<40} {pct:>9.0f}% {total:>5} {verdict:>10}")
            transitions.append({
                'from': s1, 'to': s2,
                'same_dir_pct': round(pct, 1),
                'n': total,
                'verdict': verdict,
            })

    return chain_data, transitions


def statistical_tests(results):
    """Statistical significance tests."""
    print("\n" + "="*80)
    print("STATISTICAL SIGNIFICANCE")
    print("="*80)

    valid = results[results['ret_24h'].notna()]

    # 1. One-sample t-test: is avg 24h return different from 0?
    rets = valid['ret_24h'].values
    t_stat, p_val = stats.ttest_1samp(rets, 0)
    print(f"\n1. One-sample t-test (24h return vs 0):")
    print(f"   Mean: {np.mean(rets):+.3f}%  t={t_stat:.3f}  p={p_val:.4f}  {'*** SIGNIFICANT' if p_val < 0.05 else 'NOT significant'}")

    # 2. Two-sample t-test: MISS vs BEAT
    miss = valid[valid['signal'].isin(['MISS', 'BIG_MISS'])]['ret_24h'].values
    beat = valid[valid['signal'].isin(['BEAT', 'STRONG_BEAT'])]['ret_24h'].values

    if len(miss) >= 3 and len(beat) >= 3:
        t2, p2 = stats.ttest_ind(miss, beat)
        print(f"\n2. Two-sample t-test (MISS vs BEAT):")
        print(f"   MISS: {np.mean(miss):+.3f}% (n={len(miss)})  BEAT: {np.mean(beat):+.3f}% (n={len(beat)})")
        print(f"   t={t2:.3f}  p={p2:.4f}  {'*** SIGNIFICANT' if p2 < 0.05 else 'NOT significant'}")

    # 3. Signal-specific tests
    for sig in ['STRONG_BEAT', 'BEAT', 'INLINE', 'MISS', 'BIG_MISS']:
        sig_rets = valid[valid['signal'] == sig]['ret_24h'].values
        if len(sig_rets) >= 3:
            t3, p3 = stats.ttest_1samp(sig_rets, 0)
            print(f"\n3. {sig} (n={len(sig_rets)}):")
            print(f"   Mean: {np.mean(sig_rets):+.3f}%  t={t3:.3f}  p={p3:.4f}  {'***' if p3 < 0.05 else 'ns'}")

    # 4. By year
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
    print("="*80)
    print("PROMPT A: US RETAIL SALES BACKTEST (2018-2026)")
    print("="*80)
    print("Loading ETH 15m data...")
    results = run_backtest(csv_path)
    print(f"Analyzed {len(results)} US Retail Sales releases ({results['year'].min()}-{results['year'].max()})")

    # Basic stats
    valid = results[results['ret_24h'].notna()]
    print(f"\n24h Aggregate Returns:")
    print(f"  Mean:   {valid['ret_24h'].mean():+.3f}%")
    print(f"  Median: {valid['ret_24h'].median():+.3f}%")
    print(f"  Std:    {valid['ret_24h'].std():.3f}%")
    print(f"  Win%:   {(valid['ret_24h'] > 0).mean()*100:.1f}%")
    print(f"  n:      {len(valid)}")

    # Signal breakdown
    print("\n  By Signal:")
    for sig in ['STRONG_BEAT', 'BEAT', 'INLINE', 'MISS', 'BIG_MISS']:
        sig_data = valid[valid['signal'] == sig]
        if len(sig_data) > 0:
            avg = sig_data['ret_24h'].mean()
            win = (sig_data['ret_24h'] > 0).mean() * 100
            n = len(sig_data)
            print(f"  {sig:<15} avg={avg:+.3f}%  win={win:.0f}%  n={n}")

    # Cross-tabulation
    cross_tab = cross_tabulate(results)

    # Transmission chain (Prompt B)
    print("\n" + "="*80)
    print("PROMPT B: SESSION TRANSMISSION CHAIN VALIDATION")
    print("="*80)
    chain, transitions = transmission_chain(results)

    # Statistical tests
    statistical_tests(results)

    # Save results
    out_path = os.path.join(os.path.dirname(__file__), 'backtest_us_retail_results.json')
    results.to_json(out_path, orient='records', indent=2)
    print(f"\nResults saved to {out_path}")

    return results


if __name__ == '__main__':
    main()

#!/usr/bin/env python3
"""
Prompt A: Backtest China CPI + PPI (2018-today) using ETH/USDT 15m data.

China CPI + PPI released by NBS at 09:30 CST (01:30 UTC) on ~10th-15th of each month.
Both CPI and PPI are released simultaneously.

Session itinerary (per user's thesis):
  Asia Open (09:30 CST = 01:30 UTC) → Commodity Market Open → Europe Open → Asia Re-open

We measure:
  1. ETH returns across all session phases
  2. Wyckoff phase, vol regime, signal classification
  3. Cross-tabulation: (wyckoff × vol × signal) → avg 24h return, win rate
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
# CHINA CPI + PPI RELEASE DATES (09:30 CST = 01:30 UTC)
# NBS releases CPI and PPI simultaneously
# Format: {date: {'cpi_yoy': float, 'ppi_yoy': float}}
# ═══════════════════════════════════════════════════════════════

RELEASES = {
    # 2018
    '2018-01-10': {'cpi_yoy': 1.5, 'ppi_yoy': 4.3},
    '2018-02-09': {'cpi_yoy': 2.9, 'ppi_yoy': 4.3},
    '2018-03-09': {'cpi_yoy': 2.9, 'ppi_yoy': 3.7},
    '2018-04-11': {'cpi_yoy': 2.1, 'ppi_yoy': 3.1},
    '2018-05-10': {'cpi_yoy': 1.8, 'ppi_yoy': 3.4},
    '2018-06-09': {'cpi_yoy': 1.9, 'ppi_yoy': 4.7},
    '2018-07-10': {'cpi_yoy': 2.1, 'ppi_yoy': 4.6},
    '2018-08-09': {'cpi_yoy': 2.1, 'ppi_yoy': 4.6},
    '2018-09-10': {'cpi_yoy': 2.3, 'ppi_yoy': 3.6},
    '2018-10-16': {'cpi_yoy': 2.5, 'ppi_yoy': 3.3},
    '2018-11-09': {'cpi_yoy': 2.2, 'ppi_yoy': 2.7},
    '2018-12-09': {'cpi_yoy': 2.2, 'ppi_yoy': 0.9},
    # 2019
    '2019-01-10': {'cpi_yoy': 1.9, 'ppi_yoy': 0.1},
    '2019-02-15': {'cpi_yoy': 1.5, 'ppi_yoy': 0.1},
    '2019-03-09': {'cpi_yoy': 1.5, 'ppi_yoy': 0.1},
    '2019-04-11': {'cpi_yoy': 2.3, 'ppi_yoy': 0.4},
    '2019-05-09': {'cpi_yoy': 2.5, 'ppi_yoy': 0.9},
    '2019-06-12': {'cpi_yoy': 2.7, 'ppi_yoy': 0.6},
    '2019-07-10': {'cpi_yoy': 2.8, 'ppi_yoy': -0.3},
    '2019-08-09': {'cpi_yoy': 2.8, 'ppi_yoy': -0.8},
    '2019-09-10': {'cpi_yoy': 3.0, 'ppi_yoy': -1.2},
    '2019-10-15': {'cpi_yoy': 3.8, 'ppi_yoy': -1.6},
    '2019-11-09': {'cpi_yoy': 3.8, 'ppi_yoy': -1.4},
    '2019-12-10': {'cpi_yoy': 4.5, 'ppi_yoy': -1.4},
    # 2020
    '2020-01-09': {'cpi_yoy': 4.5, 'ppi_yoy': -0.5},
    '2020-02-10': {'cpi_yoy': 5.2, 'ppi_yoy': -0.3},
    '2020-03-10': {'cpi_yoy': 5.2, 'ppi_yoy': -0.4},
    '2020-04-10': {'cpi_yoy': 3.3, 'ppi_yoy': -3.1},
    '2020-05-12': {'cpi_yoy': 2.4, 'ppi_yoy': -3.7},
    '2020-06-10': {'cpi_yoy': 2.5, 'ppi_yoy': -3.0},
    '2020-07-09': {'cpi_yoy': 2.7, 'ppi_yoy': -2.4},
    '2020-08-10': {'cpi_yoy': 2.4, 'ppi_yoy': -2.0},
    '2020-09-09': {'cpi_yoy': 1.7, 'ppi_yoy': -2.1},
    '2020-10-15': {'cpi_yoy': 0.5, 'ppi_yoy': -2.1},
    '2020-11-10': {'cpi_yoy': -0.5, 'ppi_yoy': -1.5},
    '2020-12-09': {'cpi_yoy': 0.2, 'ppi_yoy': -1.1},
    # 2021
    '2021-01-11': {'cpi_yoy': -0.3, 'ppi_yoy': 0.3},
    '2021-02-10': {'cpi_yoy': -0.2, 'ppi_yoy': 1.7},
    '2021-03-10': {'cpi_yoy': 0.4, 'ppi_yoy': 4.4},
    '2021-04-09': {'cpi_yoy': 0.9, 'ppi_yoy': 6.8},
    '2021-05-11': {'cpi_yoy': 1.3, 'ppi_yoy': 9.0},
    '2021-06-09': {'cpi_yoy': 1.1, 'ppi_yoy': 8.8},
    '2021-07-09': {'cpi_yoy': 1.0, 'ppi_yoy': 9.0},
    '2021-08-09': {'cpi_yoy': 0.8, 'ppi_yoy': 9.5},
    '2021-09-09': {'cpi_yoy': 0.7, 'ppi_yoy': 10.7},
    '2021-10-14': {'cpi_yoy': 1.5, 'ppi_yoy': 13.5},
    '2021-11-10': {'cpi_yoy': 2.3, 'ppi_yoy': 12.9},
    '2021-12-09': {'cpi_yoy': 1.5, 'ppi_yoy': 10.3},
    # 2022
    '2022-01-12': {'cpi_yoy': 0.9, 'ppi_yoy': 9.1},
    '2022-02-16': {'cpi_yoy': 0.9, 'ppi_yoy': 8.8},
    '2022-03-09': {'cpi_yoy': 0.9, 'ppi_yoy': 8.8},
    '2022-04-11': {'cpi_yoy': 1.5, 'ppi_yoy': 8.0},
    '2022-05-11': {'cpi_yoy': 2.1, 'ppi_yoy': 6.4},
    '2022-06-10': {'cpi_yoy': 2.1, 'ppi_yoy': 6.1},
    '2022-07-09': {'cpi_yoy': 2.7, 'ppi_yoy': 4.2},
    '2022-08-10': {'cpi_yoy': 2.5, 'ppi_yoy': 2.3},
    '2022-09-09': {'cpi_yoy': 2.8, 'ppi_yoy': 0.9},
    '2022-10-14': {'cpi_yoy': 2.1, 'ppi_yoy': -1.3},
    '2022-11-09': {'cpi_yoy': 1.6, 'ppi_yoy': -1.3},
    '2022-12-09': {'cpi_yoy': 1.8, 'ppi_yoy': -0.7},
    # 2023
    '2023-01-12': {'cpi_yoy': 1.8, 'ppi_yoy': -0.8},
    '2023-02-10': {'cpi_yoy': 1.0, 'ppi_yoy': -1.4},
    '2023-03-09': {'cpi_yoy': 1.0, 'ppi_yoy': -2.5},
    '2023-04-11': {'cpi_yoy': 0.1, 'ppi_yoy': -3.6},
    '2023-05-11': {'cpi_yoy': 0.2, 'ppi_yoy': -4.6},
    '2023-06-09': {'cpi_yoy': 0.0, 'ppi_yoy': -5.4},
    '2023-07-10': {'cpi_yoy': -0.3, 'ppi_yoy': -5.4},
    '2023-08-09': {'cpi_yoy': 0.1, 'ppi_yoy': -4.4},
    '2023-09-09': {'cpi_yoy': 0.0, 'ppi_yoy': -3.0},
    '2023-10-13': {'cpi_yoy': -0.2, 'ppi_yoy': -2.6},
    '2023-11-09': {'cpi_yoy': -0.2, 'ppi_yoy': -3.0},
    '2023-12-09': {'cpi_yoy': -0.3, 'ppi_yoy': -2.7},
    # 2024
    '2024-01-12': {'cpi_yoy': -0.8, 'ppi_yoy': -2.7},
    '2024-02-08': {'cpi_yoy': 0.7, 'ppi_yoy': -2.7},
    '2024-03-09': {'cpi_yoy': 0.7, 'ppi_yoy': -2.7},
    '2024-04-11': {'cpi_yoy': 0.1, 'ppi_yoy': -2.5},
    '2024-05-11': {'cpi_yoy': 0.3, 'ppi_yoy': -2.5},
    '2024-06-12': {'cpi_yoy': 0.2, 'ppi_yoy': -0.8},
    '2024-07-10': {'cpi_yoy': 0.5, 'ppi_yoy': -0.8},
    '2024-08-09': {'cpi_yoy': 0.6, 'ppi_yoy': -0.8},
    '2024-09-09': {'cpi_yoy': 0.4, 'ppi_yoy': -1.8},
    '2024-10-13': {'cpi_yoy': 0.3, 'ppi_yoy': -2.8},
    '2024-11-09': {'cpi_yoy': 0.2, 'ppi_yoy': -2.9},
    '2024-12-09': {'cpi_yoy': 0.2, 'ppi_yoy': -2.5},
    # 2025
    '2025-01-09': {'cpi_yoy': 0.5, 'ppi_yoy': -2.3},
    '2025-02-09': {'cpi_yoy': -0.7, 'ppi_yoy': -2.3},
    '2025-03-09': {'cpi_yoy': -0.7, 'ppi_yoy': -2.2},
    '2025-04-10': {'cpi_yoy': -0.1, 'ppi_yoy': -2.7},
    '2025-05-10': {'cpi_yoy': -0.1, 'ppi_yoy': -2.7},
    '2025-06-09': {'cpi_yoy': -0.2, 'ppi_yoy': -3.0},
    '2025-07-09': {'cpi_yoy': 0.0, 'ppi_yoy': -3.6},
    '2025-08-09': {'cpi_yoy': 0.0, 'ppi_yoy': -3.6},
    '2025-09-10': {'cpi_yoy': 0.0, 'ppi_yoy': -2.8},
    '2025-10-15': {'cpi_yoy': 0.2, 'ppi_yoy': -2.1},
    '2025-11-09': {'cpi_yoy': 0.2, 'ppi_yoy': -2.1},
    '2025-12-09': {'cpi_yoy': 0.3, 'ppi_yoy': -1.8},
    # 2026
    '2026-01-09': {'cpi_yoy': 0.5, 'ppi_yoy': -1.5},
    '2026-02-09': {'cpi_yoy': -0.7, 'ppi_yoy': -2.2},
    '2026-03-09': {'cpi_yoy': -0.1, 'ppi_yoy': -2.3},
    '2026-04-10': {'cpi_yoy': -0.1, 'ppi_yoy': -2.7},
    '2026-05-10': {'cpi_yoy': 0.0, 'ppi_yoy': -2.5},
}

# ═══════════════════════════════════════════════════════════════
# SESSION DEFINITIONS (adjusted for China CPI release at 01:30 UTC)
# ═══════════════════════════════════════════════════════════════

SESSIONS = {
    # Pre-Asia: from release (01:30 UTC) to Sydney open
    'post_ny_close': {'start': 21, 'end': 0},     # 21:00-00:00 (night before)
    'pre_asia': {'start': 0, 'end': 1},            # 00:00-01:30 (before release)

    # Asia: Sydney → Tokyo → Mid → Afternoon → Close
    'sydney_open': {'start': 1, 'end': 2},         # 01:30-02:00 (release window)
    'tokyo_open': {'start': 2, 'end': 3},          # 02:00-03:00
    'asia_mid': {'start': 3, 'end': 5},            # 03:00-05:00
    'asia_afternoon': {'start': 5, 'end': 6},      # 05:00-06:00
    'tokyo_close': {'start': 6, 'end': 7},         # 06:00-07:00
    'pre_london': {'start': 7, 'end': 8},          # 07:00-08:00

    # Europe
    'frankfurt_open': {'start': 7, 'end': 8},      # 07:00-08:00
    'london_open': {'start': 8, 'end': 9},         # 08:00-09:00
    'london_morning': {'start': 9, 'end': 12},     # 09:00-12:00
    'london_midday': {'start': 12, 'end': 14},     # 12:00-14:00

    # Overlap (EU-US)
    'ny_pre_open': {'start': 12, 'end': 13},       # 12:00-13:30
    'ny_open': {'start': 13, 'end': 14},           # 13:30-14:00
    'london_ny_overlap': {'start': 14, 'end': 16}, # 14:00-16:00

    # New York
    'ny_am': {'start': 14, 'end': 17},             # 14:00-17:00
    'ny_lunch': {'start': 17, 'end': 18},          # 17:00-18:00
    'ny_pm': {'start': 18, 'end': 21},             # 18:00-21:00

    # Post-release (first hour)
    'release_1h': {'start': 1, 'end': 2},          # 01:30-02:30 (1h post-release)
}

# Simplified session groups for analysis
SESSION_CHAIN = [
    ('Pre-Asia', 'pre_asia', [0]),
    ('Release (1h)', 'release_1h', [1, 2]),
    ('Sydney Open', 'sydney_open', [1]),
    ('Tokyo Open', 'tokyo_open', [2]),
    ('Asia Mid', 'asia_mid', [3, 4]),
    ('Asia Afternoon', 'asia_afternoon', [5]),
    ('Tokyo Close', 'tokyo_close', [6]),
    ('Pre-London', 'pre_london', [7]),
    ('Frankfurt Open', 'frankfurt_open', [7]),
    ('London Open', 'london_open', [8]),
    ('London Morning', 'london_morning', [9, 10, 11]),
    ('London Midday', 'london_midday', [12, 13]),
    ('NY Pre-Open', 'ny_pre_open', [12, 13]),
    ('NY Open', 'ny_open', [13]),
    ('London-NY Overlap', 'london_ny_overlap', [14, 15]),
    ('NY AM', 'ny_am', [14, 15, 16]),
    ('NY Lunch', 'ny_lunch', [17]),
    ('NY PM', 'ny_pm', [18, 19, 20]),
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


def classify_wyckoff_phase(price_series, lookback=96):
    """Simple Wyckoff phase classification based on recent price action.
    Uses 96 bars = 24h of 15m data.
    Returns: ACCUMULATION, MARKUP, DISTRIBUTION, MARKDOWN, RANGE
    """
    if len(price_series) < lookback:
        return 'RANGE'
    recent = price_series[-lookback:]
    high = recent.max()
    low = recent.min()
    current = recent.iloc[-1]
    range_pct = (high - low) / low * 100
    pos_in_range = (current - low) / (high - low) if high != low else 0.5

    if range_pct < 1.5:
        return 'RANGE' if 0.3 < pos_in_range < 0.7 else (
            'ACCUMULATION' if pos_in_range < 0.3 else 'DISTRIBUTION')
    elif current > recent.mean() and pos_in_range > 0.6:
        return 'MARKUP'
    elif current < recent.mean() and pos_in_range < 0.4:
        return 'MARKDOWN'
    elif pos_in_range > 0.7:
        return 'DISTRIBUTION'
    elif pos_in_range < 0.3:
        return 'ACCUMULATION'
    else:
        return 'RANGE'


def classify_vol_regime(df, release_dt, lookback_hours=24):
    """Classify volatility regime using M9-like logic.
    Returns: TREND, SQUEEZE, CHOP, LOW_VOL
    """
    start = release_dt - timedelta(hours=lookback_hours)
    bars = get_bars_between(df, start, release_dt)
    if len(bars) < 10:
        return 'CHOP'

    closes = bars['Close'].astype(float).values
    ranges = (bars['High'].astype(float).values - bars['Low'].astype(float).values)
    avg_range_pct = np.mean(ranges / closes) * 100

    # Trend detection: linear regression slope
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


def classify_signal(cpi_yoy, ppi_yoy, prev_cpi=None, prev_ppi=None):
    """Classify the China CPI+PPI signal.
    Returns: SEVERE_DEFLATION, DEFLATION, DISINFLATION, STABLE, INFLATION
    """
    # PPI is the primary demand signal for commodities/industrial
    # CPI is the consumer side
    # Combined: if both negative and falling → severe deflation
    if ppi_yoy is not None and ppi_yoy < -2.0:
        if cpi_yoy is not None and cpi_yoy < 0:
            return 'SEVERE_DEFLATION'
        return 'DEFLATION'
    elif ppi_yoy is not None and ppi_yoy < 0:
        if cpi_yoy is not None and cpi_yoy < 0:
            return 'DEFLATION'
        return 'DISINFLATION'
    elif ppi_yoy is not None and ppi_yoy > 3.0:
        return 'INFLATION'
    elif ppi_yoy is not None and ppi_yoy > 1.0:
        return 'STABLE'
    else:
        return 'DISINFLATION'


def classify_direction(cpi_yoy, ppi_yoy, prev_cpi=None, prev_ppi=None):
    """Surprise direction: BEAT (better than expected) or MISS (worse).
    For China data, "better" = less deflationary / more inflationary.
    """
    # Use direction of change if previous data available
    if prev_ppi is not None:
        ppi_delta = ppi_yoy - prev_ppi
        if ppi_delta > 0.3:
            return 'BEAT'  # PPI improving (less deflationary)
        elif ppi_delta < -0.3:
            return 'MISS'  # PPI worsening (more deflationary)
    # Fallback: absolute level
    if ppi_yoy > 0:
        return 'BEAT'
    elif ppi_yoy < -1.0:
        return 'MISS'
    return 'INLINE'


def run_backtest(csv_path):
    """Run the full backtest."""
    df = load_eth_data(csv_path)
    results = []

    sorted_dates = sorted(RELEASES.keys())

    for i, date_str in enumerate(sorted_dates):
        release_data = RELEASES[date_str]
        release_dt = datetime.strptime(date_str, '%Y-%m-%d').replace(hour=1, minute=30)

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
        cpi_yoy = release_data['cpi_yoy']
        ppi_yoy = release_data['ppi_yoy']
        prev_cpi = RELEASES[sorted_dates[i-1]]['cpi_yoy'] if i > 0 else None
        prev_ppi = RELEASES[sorted_dates[i-1]]['ppi_yoy'] if i > 0 else None

        signal = classify_signal(cpi_yoy, ppi_yoy, prev_cpi, prev_ppi)
        direction = classify_direction(cpi_yoy, ppi_yoy, prev_cpi, prev_ppi)

        # Wyckoff phase (from price action before release)
        wyckoff = classify_wyckoff_phase(df['Close'].astype(float))
        vol_regime = classify_vol_regime(df, release_dt)

        # Session returns
        session_returns = {}
        for name, key, hours in SESSION_CHAIN:
            if len(hours) == 1:
                start = release_dt.replace(hour=hours[0], minute=0 if hours[0] != 1 else 30)
                end = start + timedelta(hours=1)
            else:
                start = release_dt.replace(hour=hours[0], minute=0 if hours[0] != 1 else 30)
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

        # Year
        year = int(date_str[:4])

        results.append({
            'date': date_str,
            'year': year,
            'cpi_yoy': cpi_yoy,
            'ppi_yoy': ppi_yoy,
            'prev_cpi': prev_cpi,
            'prev_ppi': prev_ppi,
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
    """Cross-tabulate: (signal × vol × direction) → avg 24h return, win rate, n."""
    print("\n" + "="*80)
    print("CROSS-TABULATION: Signal × Vol Regime × Direction")
    print("="*80)

    # Filter to rows with 24h returns
    valid = results[results['ret_24h'].notna()]

    combos = valid.groupby(['signal', 'vol_regime', 'direction'])
    rows = []
    for (sig, vol, d), group in combos:
        n = len(group)
        avg_ret = group['ret_24h'].mean()
        win_rate = (group['ret_24h'] > 0).mean()
        med_ret = group['ret_24h'].median()
        std_ret = group['ret_24h'].std()
        rows.append({
            'signal': sig, 'vol': vol, 'direction': d,
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
    """Analyze session-by-session transmission chain."""
    print("\n" + "="*80)
    print("SESSION TRANSMISSION CHAIN")
    print("="*80)

    valid = results[results['ret_24h'].notna()]

    # For each session, compute avg return and direction persistence
    chain_sessions = [
        'Pre-Asia', 'Release (1h)', 'Sydney Open', 'Tokyo Open',
        'Asia Mid', 'Asia Afternoon', 'Tokyo Close', 'Pre-London',
        'Frankfurt Open', 'London Open', 'London Morning', 'London Midday',
        'NY Pre-Open', 'NY Open', 'London-NY Overlap',
        'NY AM', 'NY Lunch', 'NY PM'
    ]

    print(f"\n{'Session':<25} {'Avg Ret%':>10} {'Win%':>8} {'n':>5} {'Same Dir%':>10}")
    print("-"*65)

    prev_dir = None
    chain_data = []
    for session_name in chain_sessions:
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

        # Direction: UP if avg > 0.05%, DOWN if avg < -0.05%, else FLAT
        curr_dir = 'UP' if avg > 0.05 else 'DOWN' if avg < -0.05 else 'FLAT'

        # Same direction as previous session
        same_dir = ''
        if prev_dir and prev_dir != 'FLAT' and curr_dir != 'FLAT':
            same_count = 0
            total = 0
            for _, row in valid.iterrows():
                sr = row.get('session_returns', {})
                if session_name in sr:
                    # Find previous session in chain
                    idx = chain_sessions.index(session_name)
                    for prev_idx in range(idx-1, -1, -1):
                        prev_s = chain_sessions[prev_idx]
                        if prev_s in sr:
                            if (sr[prev_s] > 0.05 and sr[session_name] > 0.05) or \
                               (sr[prev_s] < -0.05 and sr[session_name] < -0.05):
                                same_count += 1
                            total += 1
                            break
            if total > 0:
                same_dir_pct = same_count / total * 100
                same_dir = f'{same_dir_pct:.0f}%'

        print(f"{session_name:<25} {avg:>+10.3f} {win:>7.0f}% {n:>5} {same_dir:>10}")

        chain_data.append({
            'session': session_name,
            'avg_ret': round(avg, 3),
            'win_pct': round(win, 1),
            'n': n,
            'direction': curr_dir,
        })
        prev_dir = curr_dir

    return chain_data


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
    miss = valid[valid['direction'] == 'MISS']['ret_24h'].values
    beat = valid[valid['direction'] == 'BEAT']['ret_24h'].values
    inline = valid[valid['direction'] == 'INLINE']['ret_24h'].values

    if len(miss) >= 3 and len(beat) >= 3:
        t2, p2 = stats.ttest_ind(miss, beat)
        print(f"\n2. Two-sample t-test (MISS vs BEAT):")
        print(f"   MISS: {np.mean(miss):+.3f}% (n={len(miss)})  BEAT: {np.mean(beat):+.3f}% (n={len(beat)})")
        print(f"   t={t2:.3f}  p={p2:.4f}  {'*** SIGNIFICANT' if p2 < 0.05 else 'NOT significant'}")

    # 3. Signal-specific tests
    for sig in ['SEVERE_DEFLATION', 'DEFLATION', 'DISINFLATION', 'INFLATION']:
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
    print("Loading ETH 15m data...")
    results = run_backtest(csv_path)
    print(f"Analyzed {len(results)} China CPI+PPI releases ({results['year'].min()}-{results['year'].max()})")

    # Basic stats
    valid = results[results['ret_24h'].notna()]
    print(f"\n24h Aggregate Returns:")
    print(f"  Mean:   {valid['ret_24h'].mean():+.3f}%")
    print(f"  Median: {valid['ret_24h'].median():+.3f}%")
    print(f"  Std:    {valid['ret_24h'].std():.3f}%")
    print(f"  Win%:   {(valid['ret_24h'] > 0).mean()*100:.1f}%")
    print(f"  n:      {len(valid)}")

    # Cross-tabulation
    cross_tab = cross_tabulate(results)

    # Transmission chain
    chain = transmission_chain(results)

    # Statistical tests
    statistical_tests(results)

    # Signal breakdown
    print("\n" + "="*80)
    print("SIGNAL BREAKDOWN")
    print("="*80)
    for sig in ['SEVERE_DEFLATION', 'DEFLATION', 'DISINFLATION', 'STABLE', 'INFLATION']:
        sig_data = valid[valid['signal'] == sig]
        if len(sig_data) > 0:
            avg = sig_data['ret_24h'].mean()
            win = (sig_data['ret_24h'] > 0).mean() * 100
            n = len(sig_data)
            print(f"  {sig:<20} avg={avg:+.3f}%  win={win:.0f}%  n={n}")

    # Direction breakdown
    print("\n  By Direction (surprise):")
    for d in ['BEAT', 'MISS', 'INLINE']:
        d_data = valid[valid['direction'] == d]
        if len(d_data) > 0:
            avg = d_data['ret_24h'].mean()
            win = (d_data['ret_24h'] > 0).mean() * 100
            n = len(d_data)
            print(f"  {d:<10} avg={avg:+.3f}%  win={win:.0f}%  n={n}")

    # Save results
    out_path = os.path.join(os.path.dirname(__file__), 'backtest_china_results.json')
    results.to_json(out_path, orient='records', indent=2)
    print(f"\nResults saved to {out_path}")

    return results


if __name__ == '__main__':
    main()

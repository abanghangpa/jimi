#!/usr/bin/env python3
"""
PPI & CPI Data Release Analysis — ETH/USDT 15m (2018-2026)

Analyzes how ETH reacts to US inflation data releases:
  - Pre-release price → 1h spike → US session close → Asia session → Next US

Usage:
    python3 scripts/ppi_cpi_analysis.py
    python3 scripts/ppi_cpi_analysis.py --json
    python3 scripts/ppi_cpi_analysis.py --csv results.csv
"""

import argparse
import sys
import os
import json
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from collections import defaultdict

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

# ═══════════════════════════════════════════════════════════════
# RELEASE DATES — PPI (8:30 AM ET = 13:30 UTC)
# ═══════════════════════════════════════════════════════════════

PPI_DATES = {
    # 2018
    '2018-01-11', '2018-02-15', '2018-03-14', '2018-04-11',
    '2018-05-10', '2018-06-12', '2018-07-11', '2018-09-12',
    '2018-10-10', '2018-11-14', '2018-12-11',
    # 2019
    '2019-01-15', '2019-02-14', '2019-03-14', '2019-04-11',
    '2019-05-09', '2019-06-11', '2019-07-12', '2019-08-09',
    '2019-09-11', '2019-10-08', '2019-11-14', '2019-12-12',
    # 2020
    '2020-01-14', '2020-02-19', '2020-03-12', '2020-04-09',
    '2020-05-12', '2020-06-10', '2020-07-14', '2020-08-11',
    '2020-09-11', '2020-10-14', '2020-11-13', '2020-12-11',
    # 2021
    '2021-01-13', '2021-02-17', '2021-03-12', '2021-04-09',
    '2021-05-12', '2021-06-11', '2021-07-13', '2021-08-12',
    '2021-09-10', '2021-10-14', '2021-11-09', '2021-12-14',
    # 2022
    '2022-01-13', '2022-02-15', '2022-03-11', '2022-04-12',
    '2022-05-12', '2022-06-14', '2022-07-14', '2022-08-11',
    '2022-09-14', '2022-10-12', '2022-11-15', '2022-12-09',
    # 2023
    '2023-01-18', '2023-02-16', '2023-03-15', '2023-04-13',
    '2023-05-11', '2023-06-14', '2023-07-13', '2023-08-11',
    '2023-09-14', '2023-10-12', '2023-11-15', '2023-12-13',
    # 2024
    '2024-01-12', '2024-02-16', '2024-03-14', '2024-04-11',
    '2024-05-14', '2024-06-13', '2024-07-12', '2024-08-13',
    '2024-09-12', '2024-10-11', '2024-11-14', '2024-12-12',
    # 2025
    '2025-01-14', '2025-02-13', '2025-03-13', '2025-04-10',
    '2025-05-15', '2025-06-12', '2025-07-16', '2025-08-14',
    '2025-09-11', '2025-10-15', '2025-11-13', '2025-12-11',
    # 2026 (confirmed from BLS schedule)
    '2026-01-14', '2026-02-13', '2026-03-13', '2026-04-14',
    '2026-05-13', '2026-06-11', '2026-07-10', '2026-08-13',
    '2026-09-11', '2026-10-14', '2026-11-13', '2026-12-10',
}

# ═══════════════════════════════════════════════════════════════
# RELEASE DATES — CPI (8:30 AM ET = 13:30 UTC)
# ═══════════════════════════════════════════════════════════════

CPI_DATES = {
    # 2018
    '2018-01-11', '2018-02-14', '2018-03-13', '2018-04-11',
    '2018-05-10', '2018-06-12', '2018-07-12', '2018-08-10',
    '2018-09-13', '2018-10-11', '2018-11-14', '2018-12-12',
    # 2019
    '2019-01-11', '2019-02-13', '2019-03-12', '2019-04-10',
    '2019-05-09', '2019-06-12', '2019-07-11', '2019-08-13',
    '2019-09-12', '2019-10-10', '2019-11-13', '2019-12-11',
    # 2020
    '2020-01-14', '2020-02-13', '2020-03-11', '2020-04-10',
    '2020-05-12', '2020-06-10', '2020-07-14', '2020-08-12',
    '2020-09-11', '2020-10-13', '2020-11-12', '2020-12-10',
    # 2021
    '2021-01-13', '2021-02-10', '2021-03-10', '2021-04-13',
    '2021-05-12', '2021-06-10', '2021-07-13', '2021-08-11',
    '2021-09-14', '2021-10-13', '2021-11-10', '2021-12-10',
    # 2022
    '2022-01-12', '2022-02-10', '2022-03-10', '2022-04-12',
    '2022-05-11', '2022-06-10', '2022-07-13', '2022-08-10',
    '2022-09-13', '2022-10-13', '2022-11-10', '2022-12-13',
    # 2023
    '2023-01-12', '2023-02-14', '2023-03-14', '2023-04-12',
    '2023-05-10', '2023-06-13', '2023-07-12', '2023-08-10',
    '2023-09-13', '2023-10-12', '2023-11-14', '2023-12-12',
    # 2024
    '2024-01-11', '2024-02-13', '2024-03-12', '2024-04-10',
    '2024-05-15', '2024-06-12', '2024-07-11', '2024-08-14',
    '2024-09-11', '2024-10-10', '2024-11-13', '2024-12-11',
    # 2025
    '2025-01-15', '2025-02-12', '2025-03-12', '2025-04-10',
    '2025-05-13', '2025-06-11', '2025-07-15', '2025-08-12',
    '2025-09-10', '2025-10-14', '2025-11-12', '2025-12-10',
    # 2026 (confirmed from BLS schedule)
    '2026-01-14', '2026-02-11', '2026-03-11', '2026-04-10',
    '2026-05-12', '2026-06-10', '2026-07-14', '2026-08-12',
    '2026-09-10', '2026-10-13', '2026-11-10', '2026-12-09',
}

# Release time
RELEASE_HOUR_UTC = 13
RELEASE_MINUTE_UTC = 30

# Session windows (UTC)
US_POST_RELEASE_END_HOUR = 21    # 4 PM ET
ASIA_START_HOUR = 0              # next day
ASIA_END_HOUR = 8                # next day 8 AM UTC = 4 PM CST/SGT


# ═══════════════════════════════════════════════════════════════
# MACRO REGIME CLASSIFICATION (by year)
# ═══════════════════════════════════════════════════════════════

YEAR_REGIME = {
    2018: 'TIGHTENING',      # Fed hiking, low vol early, crash Q4
    2019: 'EASING',           # Fed cut 3x, risk-on
    2020: 'CRISIS_RECOVERY',  # COVID crash → massive stimulus
    2021: 'BULL',             # Everything bubble, ETH 4x
    2022: 'BEAR',             # Fed aggressive hikes, crypto winter
    2023: 'RECOVERY',         # Inflation cooling, rate pause
    2024: 'ACCELERATION',     # Rate cuts begin, crypto bull
    2025: 'STAGFLATION',      # Inflation re-accelerating, uncertainty
    2026: 'STAGFLATION',      # PPI 4.9%, CPI 3.8%, Fed holding
}

# Known actual PPI YoY values (for surprise calc)
PPI_ACTUAL_YOY = {
    '2018-01': 2.7, '2018-02': 2.8, '2018-03': 3.0, '2018-04': 2.6,
    '2018-05': 3.1, '2018-06': 3.4, '2018-07': 3.3, '2018-08': 2.8,
    '2018-09': 2.6, '2018-10': 2.9, '2018-11': 2.5, '2018-12': 2.5,
    '2019-01': 2.0, '2019-02': 1.9, '2019-03': 2.2, '2019-04': 2.2,
    '2019-05': 1.8, '2019-06': 1.7, '2019-07': 1.7, '2019-08': 1.8,
    '2019-09': 1.4, '2019-10': 1.1, '2019-11': 1.1, '2019-12': 1.3,
    '2020-01': 2.1, '2020-02': 1.3, '2020-03': 0.7, '2020-04': -1.2,
    '2020-05': -0.8, '2020-06': -0.7, '2020-07': -0.4, '2020-08': -0.2,
    '2020-09': 0.4, '2020-10': 0.5, '2020-11': 0.8, '2020-12': 0.8,
    '2021-01': 1.7, '2021-02': 2.8, '2021-03': 4.2, '2021-04': 6.2,
    '2021-05': 6.6, '2021-06': 7.3, '2021-07': 7.8, '2021-08': 8.3,
    '2021-09': 8.6, '2021-10': 8.8, '2021-11': 9.6, '2021-12': 9.7,
    '2022-01': 10.0, '2022-02': 10.3, '2022-03': 11.2, '2022-04': 10.8,
    '2022-05': 10.8, '2022-06': 11.3, '2022-07': 9.8, '2022-08': 8.7,
    '2022-09': 8.4, '2022-10': 8.0, '2022-11': 7.4, '2022-12': 6.5,
    '2023-01': 5.7, '2023-02': 4.6, '2023-03': 2.7, '2023-04': 2.4,
    '2023-05': 1.1, '2023-06': 0.2, '2023-07': 0.8, '2023-08': 1.6,
    '2023-09': 2.2, '2023-10': 1.3, '2023-11': 0.8, '2023-12': 1.0,
    '2024-01': 0.9, '2024-02': 1.6, '2024-03': 2.1, '2024-04': 2.3,
    '2024-05': 2.4, '2024-06': 2.7, '2024-07': 2.7, '2024-08': 1.9,
    '2024-09': 1.8, '2024-10': 2.4, '2024-11': 3.0, '2024-12': 3.3,
    '2025-01': 3.7, '2025-02': 3.2, '2025-03': 2.7, '2025-04': 2.4,
    '2025-05': 2.3, '2025-06': 2.3, '2025-07': 2.7, '2025-08': 2.6,
    '2025-09': 2.8, '2025-10': 2.4, '2025-11': 3.0, '2025-12': 3.3,
    '2026-01': 3.5, '2026-02': 3.2, '2026-03': 3.5, '2026-04': 4.9,
}

# Known actual CPI YoY values
CPI_ACTUAL_YOY = {
    '2018-01': 2.1, '2018-02': 2.2, '2018-03': 2.4, '2018-04': 2.5,
    '2018-05': 2.8, '2018-06': 2.9, '2018-07': 2.9, '2018-08': 2.7,
    '2018-09': 2.3, '2018-10': 2.5, '2018-11': 2.2, '2018-12': 1.9,
    '2019-01': 1.6, '2019-02': 1.5, '2019-03': 1.9, '2019-04': 2.0,
    '2019-05': 1.8, '2019-06': 1.6, '2019-07': 1.8, '2019-08': 1.7,
    '2019-09': 1.7, '2019-10': 1.8, '2019-11': 2.1, '2019-12': 2.3,
    '2020-01': 2.5, '2020-02': 2.3, '2020-03': 1.5, '2020-04': 0.3,
    '2020-05': 0.1, '2020-06': 0.6, '2020-07': 1.0, '2020-08': 1.3,
    '2020-09': 1.4, '2020-10': 1.2, '2020-11': 1.2, '2020-12': 1.4,
    '2021-01': 1.4, '2021-02': 1.7, '2021-03': 2.6, '2021-04': 4.2,
    '2021-05': 5.0, '2021-06': 5.4, '2021-07': 5.4, '2021-08': 5.3,
    '2021-09': 5.4, '2021-10': 6.2, '2021-11': 6.8, '2021-12': 7.0,
    '2022-01': 7.5, '2022-02': 7.9, '2022-03': 8.5, '2022-04': 8.3,
    '2022-05': 8.6, '2022-06': 9.1, '2022-07': 8.5, '2022-08': 8.3,
    '2022-09': 8.2, '2022-10': 7.7, '2022-11': 7.1, '2022-12': 6.5,
    '2023-01': 6.4, '2023-02': 6.0, '2023-03': 5.0, '2023-04': 4.9,
    '2023-05': 4.0, '2023-06': 3.0, '2023-07': 3.2, '2023-08': 3.7,
    '2023-09': 3.7, '2023-10': 3.2, '2023-11': 3.1, '2023-12': 3.4,
    '2024-01': 3.1, '2024-02': 3.2, '2024-03': 3.5, '2024-04': 3.4,
    '2024-05': 3.3, '2024-06': 3.0, '2024-07': 2.9, '2024-08': 2.5,
    '2024-09': 2.4, '2024-10': 2.6, '2024-11': 2.7, '2024-12': 2.9,
    '2025-01': 3.0, '2025-02': 2.8, '2025-03': 2.4, '2025-04': 2.3,
    '2025-05': 2.3, '2025-06': 2.7, '2025-07': 2.9, '2025-08': 2.5,
    '2025-09': 2.4, '2025-10': 2.6, '2025-11': 2.7, '2025-12': 2.9,
    '2026-01': 3.0, '2026-02': 2.8, '2026-03': 3.5, '2026-04': 3.8,
}


# ═══════════════════════════════════════════════════════════════
# DATA HELPERS
# ═══════════════════════════════════════════════════════════════

def load_csv(csv_path=None):
    """Load 15m OHLCV CSV."""
    if csv_path is None:
        csv_path = os.path.join(os.path.dirname(os.path.dirname(__file__)),
                                'eth_15m_merged.csv')
        if not os.path.exists(csv_path):
            csv_path = os.path.join(os.path.dirname(os.path.dirname(__file__)),
                                    'data', 'eth_15m_merged.csv')

    df = pd.read_csv(csv_path)
    df['Open time'] = pd.to_datetime(df['Open time'])
    for c in ['Open', 'High', 'Low', 'Close', 'Volume']:
        df[c] = pd.to_numeric(df[c], errors='coerce')
    df = df.dropna(subset=['Close'])
    df = df.sort_values('Open time').reset_index(drop=True)
    return df


def get_bar_at(df, dt):
    """Get the bar at or just before a datetime."""
    mask = df['Open time'] <= dt
    bars = df[mask]
    if len(bars) == 0:
        return None
    return bars.iloc[-1]


def get_bars_between(df, start, end):
    """Get bars between two datetimes."""
    mask = (df['Open time'] >= start) & (df['Open time'] < end)
    return df[mask]


def compute_release_analysis(df, release_date, release_type):
    """Compute full analysis for one data release.

    Returns dict with all session metrics, or None if data insufficient.
    """
    if isinstance(release_date, str):
        release_date = datetime.strptime(release_date, '%Y-%m-%d')

    release_dt = release_date.replace(hour=RELEASE_HOUR_UTC, minute=RELEASE_MINUTE_UTC)
    us_end = release_date.replace(hour=US_POST_RELEASE_END_HOUR)
    asia_start = (release_date + timedelta(days=1)).replace(hour=ASIA_START_HOUR)
    asia_end = (release_date + timedelta(days=1)).replace(hour=ASIA_END_HOUR)
    next_us_start = (release_date + timedelta(days=1)).replace(hour=RELEASE_HOUR_UTC)
    next_us_end = (release_date + timedelta(days=1)).replace(hour=US_POST_RELEASE_END_HOUR)

    # Pre-release price (last bar before 8:30 AM ET)
    pre_bar = get_bar_at(df, release_dt - timedelta(minutes=1))
    if pre_bar is None:
        return None
    pre_price = float(pre_bar['Close'])

    # First 1h post-release (4 bars of 15m)
    first_1h_bars = get_bars_between(df, release_dt, release_dt + timedelta(hours=1))
    if len(first_1h_bars) < 2:
        return None
    spike_close = float(first_1h_bars.iloc[-1]['Close'])
    spike_high = float(first_1h_bars['High'].max())
    spike_low = float(first_1h_bars['Low'].min())
    spike_pct = (spike_close - pre_price) / pre_price * 100
    spike_range = (spike_high - spike_low) / pre_price * 100

    # US session (full day from release to 4 PM ET)
    us_bars = get_bars_between(df, release_dt, us_end)
    if len(us_bars) < 2:
        return None
    us_close = float(us_bars.iloc[-1]['Close'])
    us_high = float(us_bars['High'].max())
    us_low = float(us_bars['Low'].min())
    us_move = (us_close - pre_price) / pre_price * 100
    us_range = (us_high - us_low) / pre_price * 100
    us_max_up = (us_high - pre_price) / pre_price * 100
    us_max_down = (us_low - pre_price) / pre_price * 100

    # Asia session (next day 00:00 - 08:00 UTC)
    asia_bars = get_bars_between(df, asia_start, asia_end)
    asia_data = None
    if len(asia_bars) >= 2:
        asia_open = float(asia_bars.iloc[0]['Open'])
        asia_close = float(asia_bars.iloc[-1]['Close'])
        asia_high = float(asia_bars['High'].max())
        asia_low = float(asia_bars['Low'].min())
        asia_move = (asia_close - us_close) / us_close * 100
        asia_gap = (asia_open - us_close) / us_close * 100
        asia_range = (asia_high - asia_low) / us_close * 100
        asia_data = {
            'asia_open': round(asia_open, 2),
            'asia_close': round(asia_close, 2),
            'asia_high': round(asia_high, 2),
            'asia_low': round(asia_low, 2),
            'asia_move': round(asia_move, 3),
            'asia_gap': round(asia_gap, 3),
            'asia_range': round(asia_range, 3),
        }

    # Next US session
    next_us_bars = get_bars_between(df, next_us_start, next_us_end)
    next_us_data = None
    if len(next_us_bars) >= 2:
        nu_close = float(next_us_bars.iloc[-1]['Close'])
        nu_high = float(next_us_bars['High'].max())
        nu_low = float(next_us_bars['Low'].min())
        nu_move = (nu_close - us_close) / us_close * 100
        nu_range = (nu_high - nu_low) / us_close * 100
        next_us_data = {
            'next_us_close': round(nu_close, 2),
            'next_us_move': round(nu_move, 3),
            'next_us_range': round(nu_range, 3),
        }

    # Inflation data
    date_key = release_date.strftime('%Y-%m')
    if release_type == 'PPI':
        actual_yoy = PPI_ACTUAL_YOY.get(date_key)
    else:
        actual_yoy = CPI_ACTUAL_YOY.get(date_key)

    # Get previous month for surprise calc
    prev_month_key = (release_date.replace(day=1) - timedelta(days=1)).strftime('%Y-%m')
    if release_type == 'PPI':
        prev_yoy = PPI_ACTUAL_YOY.get(prev_month_key)
    else:
        prev_yoy = CPI_ACTUAL_YOY.get(prev_month_key)

    surprise = None
    if actual_yoy is not None and prev_yoy is not None:
        surprise = actual_yoy - prev_yoy  # positive = hotter than prev

    # Regime
    year = release_date.year
    regime = YEAR_REGIME.get(year, 'UNKNOWN')

    # Direction classification
    us_dir = 'DUMP' if us_move < -0.5 else 'RALLY' if us_move > 0.5 else 'FLAT'
    spike_dir = 'DOWN' if spike_pct < -0.3 else 'UP' if spike_pct > 0.3 else 'FLAT'
    us_magnitude = 'BIG' if abs(us_move) > 3.0 else 'MEDIUM' if abs(us_move) > 1.5 else 'SMALL'

    result = {
        'date': release_date.strftime('%Y-%m-%d'),
        'type': release_type,
        'year': year,
        'regime': regime,
        'pre_price': round(pre_price, 2),
        'ppi_yoy': actual_yoy,
        'cpi_yoy': CPI_ACTUAL_YOY.get(date_key) if release_type == 'PPI' else actual_yoy,
        'ppi_yoy': PPI_ACTUAL_YOY.get(date_key) if release_type == 'CPI' else actual_yoy,
        'surprise': round(surprise, 2) if surprise is not None else None,
        # 1h spike
        'spike_pct': round(spike_pct, 3),
        'spike_dir': spike_dir,
        'spike_range': round(spike_range, 3),
        # US session
        'us_move': round(us_move, 3),
        'us_dir': us_dir,
        'us_magnitude': us_magnitude,
        'us_range': round(us_range, 3),
        'us_max_up': round(us_max_up, 3),
        'us_max_down': round(us_max_down, 3),
        'us_close': round(us_close, 2),
    }

    if asia_data:
        result.update(asia_data)
        asia_dir_val = asia_data['asia_move']
        gap_dir = 'UP' if asia_data['asia_gap'] > 0.2 else 'DOWN' if asia_data['asia_gap'] < -0.2 else 'FLAT'
        asia_dir = 'UP' if asia_dir_val > 0.3 else 'DOWN' if asia_dir_val < -0.3 else 'FLAT'
        result['asia_dir'] = asia_dir
        result['gap_dir'] = gap_dir
        result['gap_held'] = (gap_dir == asia_dir) or gap_dir == 'FLAT'
        result['asia_faded'] = (us_dir == 'RALLY' and asia_dir == 'DOWN') or \
                               (us_dir == 'DUMP' and asia_dir == 'UP')

    if next_us_data:
        result.update(next_us_data)
        nu_dir = 'UP' if next_us_data['next_us_move'] > 0.3 else 'DOWN' if next_us_data['next_us_move'] < -0.3 else 'FLAT'
        result['next_us_dir'] = nu_dir
        # Next US reversal check
        if us_dir != 'FLAT':
            result['next_us_reversed'] = (us_dir == 'RALLY' and nu_dir == 'DOWN') or \
                                          (us_dir == 'DUMP' and nu_dir == 'UP')

    return result


# ═══════════════════════════════════════════════════════════════
# AGGREGATE STATISTICS
# ═══════════════════════════════════════════════════════════════

def compute_stats(results, group_by=None):
    """Compute aggregate statistics for a group of results."""
    if not results:
        return {}

    n = len(results)

    # Spike stats
    spike_pcts = [r['spike_pct'] for r in results]
    spike_up = sum(1 for r in results if r['spike_dir'] == 'UP')
    spike_down = sum(1 for r in results if r['spike_dir'] == 'DOWN')

    # US session stats
    us_moves = [r['us_move'] for r in results]
    us_rally = sum(1 for r in results if r['us_dir'] == 'RALLY')
    us_dump = sum(1 for r in results if r['us_dir'] == 'DUMP')
    us_flat = sum(1 for r in results if r['us_dir'] == 'FLAT')
    us_ranges = [r['us_range'] for r in results]

    # Spike→US accuracy
    spike_us_agree = sum(1 for r in results
                         if r['spike_dir'] != 'FLAT' and r['us_dir'] != 'FLAT'
                         and r['spike_dir'] == ('UP' if r['us_dir'] == 'RALLY' else 'DOWN'))
    spike_us_total = sum(1 for r in results if r['spike_dir'] != 'FLAT' and r['us_dir'] != 'FLAT')
    spike_accuracy = spike_us_agree / spike_us_total if spike_us_total > 0 else 0

    # Asia stats
    asia_results = [r for r in results if 'asia_move' in r]
    asia_moves = [r['asia_move'] for r in asia_results] if asia_results else []
    gap_held_count = sum(1 for r in asia_results if r.get('gap_held', False))
    asia_faded_count = sum(1 for r in asia_results if r.get('asia_faded', False))

    # Next US stats
    next_results = [r for r in results if 'next_us_move' in r]
    next_reversed = sum(1 for r in next_results if r.get('next_us_reversed', False))

    # Big move stats (>3%)
    big_moves = [r for r in results if abs(r['us_move']) > 3.0]
    big_reversal = sum(1 for r in big_moves if r.get('next_us_reversed', False))

    stats = {
        'count': n,
        'spike': {
            'up_pct': round(spike_up / n * 100, 1) if n else 0,
            'down_pct': round(spike_down / n * 100, 1) if n else 0,
            'avg_pct': round(np.mean(spike_pcts), 3) if spike_pcts else 0,
            'median_pct': round(np.median(spike_pcts), 3) if spike_pcts else 0,
            'spike_to_us_accuracy': round(spike_accuracy * 100, 1),
        },
        'us_session': {
            'rally_pct': round(us_rally / n * 100, 1) if n else 0,
            'dump_pct': round(us_dump / n * 100, 1) if n else 0,
            'flat_pct': round(us_flat / n * 100, 1) if n else 0,
            'avg_move': round(np.mean(us_moves), 3) if us_moves else 0,
            'median_move': round(np.median(us_moves), 3) if us_moves else 0,
            'avg_range': round(np.mean(us_ranges), 3) if us_ranges else 0,
            'max_up': round(max(us_moves), 3) if us_moves else 0,
            'max_down': round(min(us_moves), 3) if us_moves else 0,
        },
    }

    if asia_results:
        stats['asia'] = {
            'count': len(asia_results),
            'avg_move': round(np.mean(asia_moves), 3),
            'gap_held_pct': round(gap_held_count / len(asia_results) * 100, 1),
            'fade_pct': round(asia_faded_count / len(asia_results) * 100, 1),
            'continuation_pct': round((len(asia_results) - asia_faded_count) / len(asia_results) * 100, 1),
        }

    if next_results:
        stats['next_us'] = {
            'count': len(next_results),
            'reversal_pct': round(next_reversed / len(next_results) * 100, 1),
        }

    if big_moves:
        stats['big_moves'] = {
            'count': len(big_moves),
            'reversal_pct': round(big_reversal / len(big_moves) * 100, 1),
        }

    return stats


# ═══════════════════════════════════════════════════════════════
# OUTPUT FORMATTERS
# ═══════════════════════════════════════════════════════════════

def print_report(all_results, ppi_results, cpi_results):
    """Print comprehensive analysis report."""

    print("\n" + "═" * 70)
    print("  PPI & CPI DATA RELEASE ANALYSIS — ETH/USDT (2018-2026)")
    print("═" * 70)

    # ── Overall Stats ──
    print(f"\n  📊 OVERALL STATISTICS")
    print(f"  {'─' * 60}")
    print(f"  Total releases analyzed: {len(all_results)}  "
          f"(PPI: {len(ppi_results)}, CPI: {len(cpi_results)})")

    for label, subset in [('ALL', all_results), ('PPI', ppi_results), ('CPI', cpi_results)]:
        if not subset:
            continue
        s = compute_stats(subset)
        sp = s['spike']
        us = s['us_session']
        print(f"\n  ── {label} ──")
        print(f"    1h Spike:  ↑{sp['up_pct']}%  ↓{sp['down_pct']}%  "
              f"avg={sp['avg_pct']:+.3f}%  median={sp['median_pct']:+.3f}%")
        print(f"    Spike→US accuracy: {sp['spike_to_us_accuracy']:.1f}%")
        print(f"    US Session: RALLY {us['rally_pct']}%  DUMP {us['dump_pct']}%  FLAT {us['flat_pct']}%")
        print(f"    US Move:   avg={us['avg_move']:+.3f}%  median={us['median_move']:+.3f}%")
        print(f"    US Range:  avg={us['avg_range']:.3f}%  max↑={us['max_up']:+.3f}%  max↓={us['max_down']:+.3f}%")
        if 'asia' in s:
            a = s['asia']
            print(f"    Asia:      avg move={a['avg_move']:+.3f}%  gap held={a['gap_held_pct']}%  "
                  f"fade={a['fade_pct']}%  continuation={a['continuation_pct']}%")
        if 'next_us' in s:
            nu = s['next_us']
            print(f"    Next US:   reversal={nu['reversal_pct']}%")
        if 'big_moves' in s:
            bm = s['big_moves']
            print(f"    Big moves (>{'3'}%): {bm['count']}  reversal={bm['reversal_pct']}%")

    # ── By Year ──
    print(f"\n  {'═' * 60}")
    print(f"  📅 BY YEAR")
    print(f"  {'─' * 60}")
    years = sorted(set(r['year'] for r in all_results))
    print(f"  {'Year':<6} {'Regime':<14} {'N':>3} {'Spike%':>8} {'US Move%':>9} "
          f"{'US Range%':>10} {'Asia Gap%':>10} {'Asia Move%':>11} {'Fade%':>7}")
    for year in years:
        yr_results = [r for r in all_results if r['year'] == year]
        regime = yr_results[0]['regime']
        n = len(yr_results)
        avg_spike = np.mean([r['spike_pct'] for r in yr_results])
        avg_us = np.mean([r['us_move'] for r in yr_results])
        avg_range = np.mean([r['us_range'] for r in yr_results])
        asia_r = [r for r in yr_results if 'asia_move' in r]
        avg_gap = np.mean([r['asia_gap'] for r in asia_r]) if asia_r else 0
        avg_asia = np.mean([r['asia_move'] for r in asia_r]) if asia_r else 0
        fade_pct = sum(1 for r in asia_r if r.get('asia_faded', False)) / len(asia_r) * 100 if asia_r else 0
        print(f"  {year:<6} {regime:<14} {n:>3} {avg_spike:>+8.3f} {avg_us:>+9.3f} "
              f"{avg_range:>10.3f} {avg_gap:>+10.3f} {avg_asia:>+11.3f} {fade_pct:>6.1f}%")

    # ── PPI vs CPI Comparison ──
    print(f"\n  {'═' * 60}")
    print(f"  🔄 PPI vs CPI COMPARISON")
    print(f"  {'─' * 60}")

    for year in years:
        yr_ppi = [r for r in ppi_results if r['year'] == year]
        yr_cpi = [r for r in cpi_results if r['year'] == year]
        if not yr_ppi or not yr_cpi:
            continue

        ppi_us = np.mean([r['us_move'] for r in yr_ppi])
        cpi_us = np.mean([r['us_move'] for r in yr_cpi])
        ppi_range = np.mean([r['us_range'] for r in yr_ppi])
        cpi_range = np.mean([r['us_range'] for r in yr_cpi])
        ppi_spike = np.mean([r['spike_pct'] for r in yr_ppi])
        cpi_spike = np.mean([r['spike_pct'] for r in yr_cpi])

        # Which moves more?
        bigger = 'CPI' if abs(cpi_us) > abs(ppi_us) else 'PPI' if abs(ppi_us) > abs(cpi_us) else 'TIE'
        vol_bigger = 'CPI' if cpi_range > ppi_range else 'PPI' if ppi_range > cpi_range else 'TIE'

        print(f"\n  {year} ({yr_ppi[0]['regime']}):")
        print(f"    {'Metric':<20} {'PPI':>10} {'CPI':>10} {'Winner':>8}")
        print(f"    {'Avg 1h Spike':<20} {ppi_spike:>+10.3f} {cpi_spike:>+10.3f} {'CPI' if abs(cpi_spike) > abs(ppi_spike) else 'PPI':>8}")
        print(f"    {'Avg US Move':<20} {ppi_us:>+10.3f} {cpi_us:>+10.3f} {bigger:>8}")
        print(f"    {'Avg US Range':<20} {ppi_range:>10.3f} {cpi_range:>10.3f} {vol_bigger:>8}")

    # ── Inflation Surprise Analysis ──
    print(f"\n  {'═' * 60}")
    print(f"  🔥 INFLATION SURPRISE ANALYSIS (Hotter vs Cooler)")
    print(f"  {'─' * 60}")

    hot_results = [r for r in all_results if r.get('surprise') is not None and r['surprise'] > 0.3]
    cold_results = [r for r in all_results if r.get('surprise') is not None and r['surprise'] < -0.3]
    inline_results = [r for r in all_results if r.get('surprise') is not None and -0.3 <= r['surprise'] <= 0.3]

    for label, subset in [('HOTTER (surprise > +0.3)', hot_results),
                           ('COOLER (surprise < -0.3)', cold_results),
                           ('INLINE (±0.3)', inline_results)]:
        if not subset:
            continue
        s = compute_stats(subset)
        us = s['us_session']
        print(f"\n  {label}: {len(subset)} releases")
        print(f"    US Move: avg={us['avg_move']:+.3f}%  RALLY {us['rally_pct']}%  DUMP {us['dump_pct']}%")
        if 'asia' in s:
            a = s['asia']
            print(f"    Asia:    avg={a['avg_move']:+.3f}%  fade={a['fade_pct']}%  gap_held={a['gap_held_pct']}%")

    # ── Extreme Moves ──
    print(f"\n  {'═' * 60}")
    print(f"  💥 EXTREME MOVES (>3% US session)")
    print(f"  {'─' * 60}")
    big = [r for r in all_results if abs(r['us_move']) > 3.0]
    if big:
        big.sort(key=lambda x: abs(x['us_move']), reverse=True)
        print(f"  {'Date':<12} {'Type':<5} {'US Move%':>9} {'Spike%':>8} {'Regime':<14} {'Next US%':>9} {'Rev?':>5}")
        for r in big[:20]:
            nu = r.get('next_us_move', None)
            rev = '✅' if r.get('next_us_reversed', False) else '❌' if nu is not None else '—'
            nu_str = f"{nu:+.2f}%" if nu is not None else "N/A"
            print(f"  {r['date']:<12} {r['type']:<5} {r['us_move']:>+9.2f} {r['spike_pct']:>+8.2f} "
                  f"{r['regime']:<14} {nu_str:>9} {rev:>5}")
    else:
        print(f"  No extreme moves found.")

    # ── Asia Gap Reliability ──
    print(f"\n  {'═' * 60}")
    print(f"  🌏 ASIA GAP RELIABILITY")
    print(f"  {'─' * 60}")
    asia_with_gap = [r for r in all_results if 'asia_gap' in r and abs(r['asia_gap']) > 0.2]
    if asia_with_gap:
        gap_up = [r for r in asia_with_gap if r['asia_gap'] > 0.2]
        gap_down = [r for r in asia_with_gap if r['asia_gap'] < -0.2]
        gap_up_held = sum(1 for r in gap_up if r.get('gap_held', False))
        gap_down_held = sum(1 for r in gap_down if r.get('gap_held', False))
        print(f"  Gap UP:   {len(gap_up)} events, held {gap_up_held}/{len(gap_up)} "
              f"({gap_up_held/len(gap_up)*100:.1f}%)" if gap_up else "  Gap UP: none")
        print(f"  Gap DOWN: {len(gap_down)} events, held {gap_down_held}/{len(gap_down)} "
              f"({gap_down_held/len(gap_down)*100:.1f}%)" if gap_down else "  Gap DOWN: none")
        total_held = sum(1 for r in asia_with_gap if r.get('gap_held', False))
        print(f"  Overall:  {total_held}/{len(asia_with_gap)} gaps held "
              f"({total_held/len(asia_with_gap)*100:.1f}%)")
    else:
        print(f"  No significant gaps found.")

    # ── Regime Breakdown ──
    print(f"\n  {'═' * 60}")
    print(f"  📈 REGIME BREAKDOWN")
    print(f"  {'─' * 60}")
    regimes = sorted(set(r['regime'] for r in all_results))
    print(f"  {'Regime':<16} {'N':>3} {'US Avg%':>9} {'US Range%':>10} {'Fade%':>7} {'Spike Acc%':>11}")
    for regime in regimes:
        reg_results = [r for r in all_results if r['regime'] == regime]
        n = len(reg_results)
        avg_us = np.mean([r['us_move'] for r in reg_results])
        avg_range = np.mean([r['us_range'] for r in reg_results])
        asia_r = [r for r in reg_results if 'asia_move' in r]
        fade_pct = sum(1 for r in asia_r if r.get('asia_faded', False)) / len(asia_r) * 100 if asia_r else 0
        s = compute_stats(reg_results)
        spike_acc = s['spike']['spike_to_us_accuracy']
        print(f"  {regime:<16} {n:>3} {avg_us:>+9.3f} {avg_range:>10.3f} {fade_pct:>6.1f}% {spike_acc:>10.1f}%")

    # ── Monthly Seasonality ──
    print(f"\n  {'═' * 60}")
    print(f"  📆 MONTHLY SEASONALITY (avg US session move)")
    print(f"  {'─' * 60}")
    months = range(1, 13)
    month_names = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
                   'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
    print(f"  {'Month':<6} {'PPI N':>5} {'PPI Avg%':>9} {'CPI N':>5} {'CPI Avg%':>9} {'Combined Avg%':>14}")
    for m in months:
        m_ppi = [r for r in ppi_results if datetime.strptime(r['date'], '%Y-%m-%d').month == m]
        m_cpi = [r for r in cpi_results if datetime.strptime(r['date'], '%Y-%m-%d').month == m]
        m_all = m_ppi + m_cpi
        ppi_avg = np.mean([r['us_move'] for r in m_ppi]) if m_ppi else 0
        cpi_avg = np.mean([r['us_move'] for r in m_cpi]) if m_cpi else 0
        all_avg = np.mean([r['us_move'] for r in m_all]) if m_all else 0
        print(f"  {month_names[m-1]:<6} {len(m_ppi):>5} {ppi_avg:>+9.3f} {len(m_cpi):>5} {cpi_avg:>+9.3f} {all_avg:>+14.3f}")

    # ── Consecutive Direction ──
    print(f"\n  {'═' * 60}")
    print(f"  🔗 DIRECTIONAL STREAKS (consecutive same-direction US moves)")
    print(f"  {'─' * 60}")
    all_sorted = sorted(all_results, key=lambda x: x['date'])
    streaks = []
    current_dir = None
    current_streak = 0
    for r in all_sorted:
        d = r['us_dir']
        if d == 'FLAT':
            continue
        if d == current_dir:
            current_streak += 1
        else:
            if current_streak >= 2:
                streaks.append((current_dir, current_streak))
            current_dir = d
            current_streak = 1
    if current_streak >= 2:
        streaks.append((current_dir, current_streak))

    if streaks:
        streak_counts = defaultdict(int)
        for d, s in streaks:
            streak_counts[(d, s)] += 1
        print(f"  {'Direction':<10} {'Streak Len':>10} {'Count':>6}")
        for (d, s), c in sorted(streak_counts.items(), key=lambda x: -x[1]):
            print(f"  {d:<10} {s:>10} {c:>6}")
    else:
        print(f"  No significant streaks found.")

    # ── Key Takeaways ──
    print(f"\n  {'═' * 60}")
    print(f"  💡 KEY TAKEAWAYS")
    print(f"  {'─' * 60}")

    all_s = compute_stats(all_results)
    ppi_s = compute_stats(ppi_results)
    cpi_s = compute_stats(cpi_results)

    # Spike accuracy
    print(f"  1. Spike→US Accuracy:")
    print(f"     PPI: {ppi_s['spike']['spike_to_us_accuracy']:.1f}%  "
          f"CPI: {cpi_s['spike']['spike_to_us_accuracy']:.1f}%  "
          f"Combined: {all_s['spike']['spike_to_us_accuracy']:.1f}%")

    # Which moves more
    ppi_avg_abs = abs(ppi_s['us_session']['avg_move'])
    cpi_avg_abs = abs(cpi_s['us_session']['avg_move'])
    print(f"\n  2. Which moves more?")
    print(f"     PPI avg |move|: {ppi_avg_abs:.3f}%  CPI avg |move|: {cpi_avg_abs:.3f}%")
    print(f"     → {'CPI' if cpi_avg_abs > ppi_avg_abs else 'PPI'} produces bigger average moves")

    # Asia behavior
    if 'asia' in all_s:
        print(f"\n  3. Asia Session:")
        print(f"     Gap held: {all_s['asia']['gap_held_pct']:.1f}%  "
              f"Fade rate: {all_s['asia']['fade_pct']:.1f}%  "
              f"Continuation: {all_s['asia']['continuation_pct']:.1f}%")

    # Regime effect
    print(f"\n  4. Regime Effect:")
    for regime in regimes:
        reg_r = [r for r in all_results if r['regime'] == regime]
        if len(reg_r) < 3:
            continue
        reg_s = compute_stats(reg_r)
        asia_r = [r for r in reg_r if 'asia_faded' in r]
        fade = sum(1 for r in asia_r if r['asia_faded']) / len(asia_r) * 100 if asia_r else 0
        print(f"     {regime:<16} avg US: {reg_s['us_session']['avg_move']:+.3f}%  "
              f"fade: {fade:.0f}%  spike acc: {reg_s['spike']['spike_to_us_accuracy']:.0f}%")

    # Big move reversal
    if 'big_moves' in all_s:
        print(f"\n  5. Big Moves (>{'3'}%):")
        print(f"     {all_s['big_moves']['count']} occurrences, "
              f"{all_s['big_moves']['reversal_pct']:.1f}% reversed next US session")

    print(f"\n{'═' * 70}\n")


# ═══════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description='PPI & CPI Release Analysis')
    parser.add_argument('--json', action='store_true', help='Output JSON')
    parser.add_argument('--csv', type=str, help='Save results to CSV')
    parser.add_argument('--data', type=str, help='Path to 15m CSV')
    args = parser.parse_args()

    print("Loading ETH/USDT 15m data...")
    df = load_csv(args.data)
    print(f"  Loaded {len(df)} bars: {df['Open time'].iloc[0]} → {df['Open time'].iloc[-1]}")

    # Run analysis for all releases
    all_results = []
    errors = []

    all_dates = set()
    for d in PPI_DATES:
        all_dates.add((d, 'PPI'))
    for d in CPI_DATES:
        all_dates.add((d, 'CPI'))

    print(f"\nAnalyzing {len(all_dates)} data releases (2018-2026)...")

    for date_str, release_type in sorted(all_dates):
        try:
            result = compute_release_analysis(df, date_str, release_type)
            if result:
                all_results.append(result)
            else:
                errors.append(f"{release_type} {date_str}: insufficient data")
        except Exception as e:
            errors.append(f"{release_type} {date_str}: {e}")

    ppi_results = [r for r in all_results if r['type'] == 'PPI']
    cpi_results = [r for r in all_results if r['type'] == 'CPI']

    print(f"  Analyzed: {len(all_results)}  (PPI: {len(ppi_results)}, CPI: {len(cpi_results)})")
    if errors:
        print(f"  Skipped:  {len(errors)} (data not available)")

    if args.json:
        output = {
            'total': len(all_results),
            'ppi_count': len(ppi_results),
            'cpi_count': len(cpi_results),
            'results': all_results,
            'stats': {
                'all': compute_stats(all_results),
                'ppi': compute_stats(ppi_results),
                'cpi': compute_stats(cpi_results),
            },
        }
        print(json.dumps(output, indent=2, default=str))
    elif args.csv:
        df_out = pd.DataFrame(all_results)
        df_out.to_csv(args.csv, index=False)
        print(f"\n  Saved to {args.csv}")
    else:
        print_report(all_results, ppi_results, cpi_results)

    # Save scan data
    scan_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data', 'scans')
    os.makedirs(scan_dir, exist_ok=True)
    scan_file = os.path.join(scan_dir, 'ppi_cpi_analysis.json')
    with open(scan_file, 'w') as f:
        json.dump({
            'generated': datetime.now().isoformat(),
            'total': len(all_results),
            'results': all_results,
            'stats': {
                'all': compute_stats(all_results),
                'ppi': compute_stats(ppi_results),
                'cpi': compute_stats(cpi_results),
            },
        }, f, indent=2, default=str)
    print(f"  💾 Full data saved: {scan_file}")


if __name__ == '__main__':
    main()

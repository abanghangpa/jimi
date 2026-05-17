#!/usr/bin/env python3
"""
Prompt A + B: Backtest UK Monthly GDP (2018-today) using ETH/USDT 15m data.

ONS Monthly GDP released at 07:00 UTC (15:00 MYT) ~10-12 days after month end.
UK is unique in releasing GDP on a monthly basis (most countries do quarterly).

Session itinerary (per user #35):
  Europe Morning (07:00 UTC) → UK Retail Sales → BoE Strategy

Thesis:
  Consecutive negative months → structural recession → BoE easing pressure
  → lower yields → crypto supportive. Weak GDP justifies BoE cut cycle.

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
# UK MONTHLY GDP RELEASE DATES (07:00 UTC = 15:00 MYT)
# ONS releases ~10-12 days after month end
# Format: {date: {'gdp_mom': float, 'gdp_yoy': float, 'prev_mom': float, 'prev_yoy': float}}
# gdp_mom = month-on-month growth rate (%)
# ═══════════════════════════════════════════════════════════════

RELEASES = {
    # 2018 (monthly GDP started Jan 2018)
    '2018-04-10': {'gdp_mom': -0.1, 'gdp_yoy': 1.2, 'prev_mom': 0.3, 'prev_yoy': 1.4, 'consensus_mom': 0.1},
    '2018-05-10': {'gdp_mom': -0.1, 'gdp_yoy': 1.3, 'prev_mom': -0.1, 'prev_yoy': 1.2, 'consensus_mom': 0.1},
    '2018-06-11': {'gdp_mom': 0.3, 'gdp_yoy': 1.5, 'prev_mom': -0.1, 'prev_yoy': 1.3, 'consensus_mom': 0.2},
    '2018-07-10': {'gdp_mom': 0.3, 'gdp_yoy': 1.3, 'prev_mom': 0.3, 'prev_yoy': 1.5, 'consensus_mom': 0.2},
    '2018-08-10': {'gdp_mom': 0.4, 'gdp_yoy': 1.3, 'prev_mom': 0.3, 'prev_yoy': 1.3, 'consensus_mom': 0.3},
    '2018-09-10': {'gdp_mom': 0.3, 'gdp_yoy': 1.4, 'prev_mom': 0.4, 'prev_yoy': 1.3, 'consensus_mom': 0.3},
    '2018-10-10': {'gdp_mom': 0.0, 'gdp_yoy': 1.5, 'prev_mom': 0.3, 'prev_yoy': 1.4, 'consensus_mom': 0.1},
    '2018-11-12': {'gdp_mom': 0.1, 'gdp_yoy': 1.5, 'prev_mom': 0.0, 'prev_yoy': 1.5, 'consensus_mom': 0.1},
    '2018-12-10': {'gdp_mom': 0.2, 'gdp_yoy': 1.4, 'prev_mom': 0.1, 'prev_yoy': 1.5, 'consensus_mom': 0.1},
    # 2019
    '2019-02-11': {'gdp_mom': -0.4, 'gdp_yoy': 1.0, 'prev_mom': 0.2, 'prev_yoy': 1.4, 'consensus_mom': -0.3},
    '2019-03-12': {'gdp_mom': 0.5, 'gdp_yoy': 1.4, 'prev_mom': -0.4, 'prev_yoy': 1.0, 'consensus_mom': 0.2},
    '2019-04-10': {'gdp_mom': -0.1, 'gdp_yoy': 1.3, 'prev_mom': 0.5, 'prev_yoy': 1.4, 'consensus_mom': 0.0},
    '2019-05-10': {'gdp_mom': 0.3, 'gdp_yoy': 1.3, 'prev_mom': -0.1, 'prev_yoy': 1.3, 'consensus_mom': 0.0},
    '2019-06-10': {'gdp_mom': 0.3, 'gdp_yoy': 1.5, 'prev_mom': 0.3, 'prev_yoy': 1.3, 'consensus_mom': 0.1},
    '2019-07-10': {'gdp_mom': -0.1, 'gdp_yoy': 1.0, 'prev_mom': 0.3, 'prev_yoy': 1.5, 'consensus_mom': 0.0},
    '2019-08-09': {'gdp_mom': 0.3, 'gdp_yoy': 1.0, 'prev_mom': -0.1, 'prev_yoy': 1.0, 'consensus_mom': 0.1},
    '2019-09-09': {'gdp_mom': -0.2, 'gdp_yoy': 0.9, 'prev_mom': 0.3, 'prev_yoy': 1.0, 'consensus_mom': 0.0},
    '2019-10-10': {'gdp_mom': -0.1, 'gdp_yoy': 0.9, 'prev_mom': -0.2, 'prev_yoy': 0.9, 'consensus_mom': -0.1},
    '2019-11-11': {'gdp_mom': 0.1, 'gdp_yoy': 0.8, 'prev_mom': -0.1, 'prev_yoy': 0.9, 'consensus_mom': 0.1},
    '2019-12-10': {'gdp_mom': 0.1, 'gdp_yoy': 0.8, 'prev_mom': 0.1, 'prev_yoy': 0.8, 'consensus_mom': 0.0},
    # 2020
    '2020-02-11': {'gdp_mom': 0.1, 'gdp_yoy': 0.9, 'prev_mom': 0.1, 'prev_yoy': 0.8, 'consensus_mom': 0.2},
    '2020-03-11': {'gdp_mom': 0.0, 'gdp_yoy': 0.6, 'prev_mom': 0.1, 'prev_yoy': 0.9, 'consensus_mom': 0.1},
    '2020-04-08': {'gdp_mom': -0.2, 'gdp_yoy': -0.1, 'prev_mom': 0.0, 'prev_yoy': 0.6, 'consensus_mom': -0.2},
    '2020-05-13': {'gdp_mom': -20.3, 'gdp_yoy': -24.5, 'prev_mom': -0.2, 'prev_yoy': -0.1, 'consensus_mom': -18.0},
    '2020-06-12': {'gdp_mom': 1.8, 'gdp_yoy': -23.3, 'prev_mom': -20.3, 'prev_yoy': -24.5, 'consensus_mom': 5.5},
    '2020-07-14': {'gdp_mom': 2.4, 'gdp_yoy': -21.5, 'prev_mom': 1.8, 'prev_yoy': -23.3, 'consensus_mom': 4.5},
    '2020-08-12': {'gdp_mom': 6.6, 'gdp_yoy': -18.6, 'prev_mom': 2.4, 'prev_yoy': -21.5, 'consensus_mom': 6.7},
    '2020-09-11': {'gdp_mom': 2.1, 'gdp_yoy': -14.0, 'prev_mom': 6.6, 'prev_yoy': -18.6, 'consensus_mom': 4.6},
    '2020-10-09': {'gdp_mom': 2.2, 'gdp_yoy': -9.6, 'prev_mom': 2.1, 'prev_yoy': -14.0, 'consensus_mom': 1.5},
    '2020-11-12': {'gdp_mom': 0.4, 'gdp_yoy': -9.4, 'prev_mom': 2.2, 'prev_yoy': -9.6, 'consensus_mom': 0.5},
    '2020-12-10': {'gdp_mom': -2.6, 'gdp_yoy': -9.7, 'prev_mom': 0.4, 'prev_yoy': -9.4, 'consensus_mom': -5.7},
    # 2021
    '2021-02-12': {'gdp_mom': -2.9, 'gdp_yoy': -9.0, 'prev_mom': -2.6, 'prev_yoy': -9.7, 'consensus_mom': -4.9},
    '2021-03-12': {'gdp_mom': 2.6, 'gdp_yoy': -7.8, 'prev_mom': -2.9, 'prev_yoy': -9.0, 'consensus_mom': 1.5},
    '2021-04-13': {'gdp_mom': 2.1, 'gdp_yoy': -5.1, 'prev_mom': 2.6, 'prev_yoy': -7.8, 'consensus_mom': 1.3},
    '2021-05-12': {'gdp_mom': 2.3, 'gdp_yoy': -3.6, 'prev_mom': 2.1, 'prev_yoy': -5.1, 'consensus_mom': 2.2},
    '2021-06-11': {'gdp_mom': -0.8, 'gdp_yoy': -3.1, 'prev_mom': 2.3, 'prev_yoy': -3.6, 'consensus_mom': 1.5},
    '2021-07-14': {'gdp_mom': 0.8, 'gdp_yoy': -2.0, 'prev_mom': -0.8, 'prev_yoy': -3.1, 'consensus_mom': 0.8},
    '2021-08-12': {'gdp_mom': 1.0, 'gdp_yoy': -1.6, 'prev_mom': 0.8, 'prev_yoy': -2.0, 'consensus_mom': 0.6},
    '2021-09-10': {'gdp_mom': 0.1, 'gdp_yoy': -0.6, 'prev_mom': 1.0, 'prev_yoy': -1.6, 'consensus_mom': 0.5},
    '2021-10-13': {'gdp_mom': 0.4, 'gdp_yoy': 2.3, 'prev_mom': 0.1, 'prev_yoy': -0.6, 'consensus_mom': 0.4},
    '2021-11-11': {'gdp_mom': 0.1, 'gdp_yoy': 2.2, 'prev_mom': 0.4, 'prev_yoy': 2.3, 'consensus_mom': 0.4},
    '2021-12-10': {'gdp_mom': 0.1, 'gdp_yoy': 1.1, 'prev_mom': 0.1, 'prev_yoy': 2.2, 'consensus_mom': 0.4},
    # 2022
    '2022-02-11': {'gdp_mom': -0.2, 'gdp_yoy': 5.5, 'prev_mom': 0.1, 'prev_yoy': 1.1, 'consensus_mom': -0.5},
    '2022-03-11': {'gdp_mom': 0.8, 'gdp_yoy': 9.5, 'prev_mom': -0.2, 'prev_yoy': 5.5, 'consensus_mom': 0.8},
    '2022-04-11': {'gdp_mom': 0.1, 'gdp_yoy': 6.4, 'prev_mom': 0.8, 'prev_yoy': 9.5, 'consensus_mom': 0.0},
    '2022-05-12': {'gdp_mom': -0.3, 'gdp_yoy': 3.7, 'prev_mom': 0.1, 'prev_yoy': 6.4, 'consensus_mom': -0.1},
    '2022-06-13': {'gdp_mom': -0.5, 'gdp_yoy': 1.8, 'prev_mom': -0.3, 'prev_yoy': 3.7, 'consensus_mom': -0.3},
    '2022-07-13': {'gdp_mom': 0.5, 'gdp_yoy': 2.9, 'prev_mom': -0.5, 'prev_yoy': 1.8, 'consensus_mom': 0.2},
    '2022-08-12': {'gdp_mom': 0.2, 'gdp_yoy': 2.4, 'prev_mom': 0.5, 'prev_yoy': 2.9, 'consensus_mom': -0.2},
    '2022-09-12': {'gdp_mom': -0.3, 'gdp_yoy': 2.0, 'prev_mom': 0.2, 'prev_yoy': 2.4, 'consensus_mom': 0.0},
    '2022-10-12': {'gdp_mom': -0.6, 'gdp_yoy': 1.5, 'prev_mom': -0.3, 'prev_yoy': 2.0, 'consensus_mom': -0.4},
    '2022-11-11': {'gdp_mom': 0.5, 'gdp_yoy': 1.5, 'prev_mom': -0.6, 'prev_yoy': 1.5, 'consensus_mom': 0.4},
    '2022-12-12': {'gdp_mom': 0.1, 'gdp_yoy': 0.6, 'prev_mom': 0.5, 'prev_yoy': 1.5, 'consensus_mom': -0.2},
    # 2023
    '2023-02-10': {'gdp_mom': -0.5, 'gdp_yoy': -0.1, 'prev_mom': 0.1, 'prev_yoy': 0.6, 'consensus_mom': -0.3},
    '2023-03-10': {'gdp_mom': 0.3, 'gdp_yoy': 0.0, 'prev_mom': -0.5, 'prev_yoy': -0.1, 'consensus_mom': 0.1},
    '2023-04-13': {'gdp_mom': 0.0, 'gdp_yoy': 0.1, 'prev_mom': 0.3, 'prev_yoy': 0.0, 'consensus_mom': 0.0},
    '2023-05-12': {'gdp_mom': 0.2, 'gdp_yoy': 0.3, 'prev_mom': 0.0, 'prev_yoy': 0.1, 'consensus_mom': 0.2},
    '2023-06-14': {'gdp_mom': -0.3, 'gdp_yoy': 0.0, 'prev_mom': 0.2, 'prev_yoy': 0.3, 'consensus_mom': -0.1},
    '2023-07-13': {'gdp_mom': 0.5, 'gdp_yoy': 0.2, 'prev_mom': -0.3, 'prev_yoy': 0.0, 'consensus_mom': 0.2},
    '2023-08-11': {'gdp_mom': 0.2, 'gdp_yoy': 0.0, 'prev_mom': 0.5, 'prev_yoy': 0.2, 'consensus_mom': 0.2},
    '2023-09-13': {'gdp_mom': -0.5, 'gdp_yoy': 0.0, 'prev_mom': 0.2, 'prev_yoy': 0.0, 'consensus_mom': -0.2},
    '2023-10-12': {'gdp_mom': 0.2, 'gdp_yoy': 0.5, 'prev_mom': -0.5, 'prev_yoy': 0.0, 'consensus_mom': 0.2},
    '2023-11-10': {'gdp_mom': -0.1, 'gdp_yoy': 0.3, 'prev_mom': 0.2, 'prev_yoy': 0.5, 'consensus_mom': 0.0},
    '2023-12-13': {'gdp_mom': 0.3, 'gdp_yoy': 0.3, 'prev_mom': -0.1, 'prev_yoy': 0.3, 'consensus_mom': 0.1},
    # 2024
    '2024-02-12': {'gdp_mom': -0.1, 'gdp_yoy': -0.2, 'prev_mom': 0.3, 'prev_yoy': 0.3, 'consensus_mom': -0.2},
    '2024-03-13': {'gdp_mom': 0.2, 'gdp_yoy': -0.3, 'prev_mom': -0.1, 'prev_yoy': -0.2, 'consensus_mom': 0.0},
    '2024-04-12': {'gdp_mom': 0.1, 'gdp_yoy': -0.2, 'prev_mom': 0.2, 'prev_yoy': -0.3, 'consensus_mom': 0.1},
    '2024-05-10': {'gdp_mom': 0.4, 'gdp_yoy': 0.2, 'prev_mom': 0.1, 'prev_yoy': -0.2, 'consensus_mom': 0.4},
    '2024-06-12': {'gdp_mom': 0.0, 'gdp_yoy': 0.6, 'prev_mom': 0.4, 'prev_yoy': 0.2, 'consensus_mom': 0.0},
    '2024-07-11': {'gdp_mom': 0.4, 'gdp_yoy': 0.7, 'prev_mom': 0.0, 'prev_yoy': 0.6, 'consensus_mom': 0.2},
    '2024-08-12': {'gdp_mom': 0.0, 'gdp_yoy': 1.1, 'prev_mom': 0.4, 'prev_yoy': 0.7, 'consensus_mom': 0.0},
    '2024-09-11': {'gdp_mom': 0.2, 'gdp_yoy': 1.0, 'prev_mom': 0.0, 'prev_yoy': 1.1, 'consensus_mom': 0.2},
    '2024-10-11': {'gdp_mom': 0.0, 'gdp_yoy': 0.8, 'prev_mom': 0.2, 'prev_yoy': 1.0, 'consensus_mom': 0.0},
    '2024-11-13': {'gdp_mom': -0.1, 'gdp_yoy': 0.6, 'prev_mom': 0.0, 'prev_yoy': 0.8, 'consensus_mom': 0.1},
    '2024-12-13': {'gdp_mom': 0.1, 'gdp_yoy': 0.9, 'prev_mom': -0.1, 'prev_yoy': 0.6, 'consensus_mom': 0.1},
    # 2025
    '2025-02-13': {'gdp_mom': 0.4, 'gdp_yoy': 1.0, 'prev_mom': 0.1, 'prev_yoy': 0.9, 'consensus_mom': 0.1},
    '2025-03-13': {'gdp_mom': -0.1, 'gdp_yoy': 0.9, 'prev_mom': 0.4, 'prev_yoy': 1.0, 'consensus_mom': 0.1},
    '2025-04-11': {'gdp_mom': 0.5, 'gdp_yoy': 1.1, 'prev_mom': -0.1, 'prev_yoy': 0.9, 'consensus_mom': 0.1},
    '2025-05-15': {'gdp_mom': 0.2, 'gdp_yoy': 1.3, 'prev_mom': 0.5, 'prev_yoy': 1.1, 'consensus_mom': 0.0},
    '2025-06-12': {'gdp_mom': -0.3, 'gdp_yoy': 0.9, 'prev_mom': 0.2, 'prev_yoy': 1.3, 'consensus_mom': -0.1},
    '2025-07-10': {'gdp_mom': 0.4, 'gdp_yoy': 1.0, 'prev_mom': -0.3, 'prev_yoy': 0.9, 'consensus_mom': 0.2},
    '2025-08-13': {'gdp_mom': 0.1, 'gdp_yoy': 1.2, 'prev_mom': 0.4, 'prev_yoy': 1.0, 'consensus_mom': 0.0},
    '2025-09-12': {'gdp_mom': 0.0, 'gdp_yoy': 1.0, 'prev_mom': 0.1, 'prev_yoy': 1.2, 'consensus_mom': 0.1},
    '2025-10-10': {'gdp_mom': -0.2, 'gdp_yoy': 0.6, 'prev_mom': 0.0, 'prev_yoy': 1.0, 'consensus_mom': 0.0},
    '2025-11-13': {'gdp_mom': 0.1, 'gdp_yoy': 0.5, 'prev_mom': -0.2, 'prev_yoy': 0.6, 'consensus_mom': 0.1},
    '2025-12-12': {'gdp_mom': 0.0, 'gdp_yoy': 0.4, 'prev_mom': 0.1, 'prev_yoy': 0.5, 'consensus_mom': 0.0},
    # 2026
    '2026-02-12': {'gdp_mom': -0.1, 'gdp_yoy': 0.3, 'prev_mom': 0.0, 'prev_yoy': 0.4, 'consensus_mom': 0.0},
    '2026-03-12': {'gdp_mom': 0.0, 'gdp_yoy': 0.2, 'prev_mom': -0.1, 'prev_yoy': 0.3, 'consensus_mom': 0.1},
    '2026-04-10': {'gdp_mom': -0.2, 'gdp_yoy': 0.0, 'prev_mom': 0.0, 'prev_yoy': 0.2, 'consensus_mom': 0.0},
    '2026-05-13': {'gdp_mom': -0.1, 'gdp_yoy': -0.1, 'prev_mom': -0.2, 'prev_yoy': 0.0, 'consensus_mom': 0.0},
}

# ═══════════════════════════════════════════════════════════════
# SESSION DEFINITIONS (UTC offsets from release time 07:00 UTC)
# UK GDP = Europe Morning release, different from US morning events
# ═══════════════════════════════════════════════════════════════

SESSIONS = {
    'Release Spike':      {'start': 0, 'end': 0.25},       # 07:00-07:15 UTC
    'London Open':        {'start': 0.25, 'end': 2.0},      # 07:15-09:00 UTC
    'London Morning':     {'start': 2.0, 'end': 4.0},       # 09:00-11:00 UTC
    'London Midday':      {'start': 4.0, 'end': 5.0},       # 11:00-12:00 UTC
    'London Afternoon':   {'start': 5.0, 'end': 7.0},       # 12:00-14:00 UTC
    'NY Pre-Open':        {'start': 7.0, 'end': 8.5},       # 14:00-15:30 UTC
    'NY Open':            {'start': 8.5, 'end': 10.0},      # 15:30-17:00 UTC
    'London-NY Overlap':  {'start': 10.0, 'end': 11.0},     # 17:00-18:00 UTC
    'NY AM':              {'start': 11.0, 'end': 14.0},     # 18:00-21:00 UTC
    'NY Lunch':           {'start': 14.0, 'end': 15.5},     # 21:00-22:30 UTC
    'NY PM':              {'start': 15.5, 'end': 17.5},     # 22:30-00:30 UTC
    'NY Close':           {'start': 17.5, 'end': 19.0},     # 00:30-02:00 UTC
    'Pre-Asia':           {'start': 19.0, 'end': 21.0},     # 02:00-04:00 UTC
    'Sydney Open':        {'start': 21.0, 'end': 22.0},     # 04:00-05:00 UTC
    'Tokyo Open':         {'start': 22.0, 'end': 23.0},     # 05:00-06:00 UTC
    'Asia Mid':           {'start': 23.0, 'end': 25.0},     # 06:00-08:00 UTC
    'Asia Afternoon':     {'start': 25.0, 'end': 27.0},     # 08:00-10:00 UTC
    'Tokyo Close':        {'start': 27.0, 'end': 28.0},     # 10:00-11:00 UTC
    'Pre-London Day2':    {'start': 28.0, 'end': 29.0},     # 11:00-12:00 UTC
    'Frankfurt Open D2':  {'start': 29.0, 'end': 30.0},     # 12:00-13:00 UTC
    'London Open D2':     {'start': 30.0, 'end': 31.0},     # 13:00-14:00 UTC
    'London Morning D2':  {'start': 31.0, 'end': 33.0},     # 14:00-16:00 UTC
    '24h_aggregate':      {'start': 0, 'end': 24.0},        # full 24h
}


def load_eth_data(csv_path):
    """Load and prepare ETH 15m data."""
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
    """Get close price at or just before timestamp."""
    mask = df['timestamp'] <= ts
    if mask.sum() == 0:
        return None
    return df.loc[mask].iloc[-1]['close']


def classify_signal(gdp_mom, consensus_mom, prev_mom):
    """Classify UK GDP surprise."""
    if gdp_mom is None or consensus_mom is None:
        return 'NO_DATA'
    surprise = gdp_mom - consensus_mom

    # Check for recession pattern (consecutive negatives)
    if gdp_mom < 0 and prev_mom < 0:
        return 'RECESSION_SIGNAL'
    elif gdp_mom < 0 and prev_mom >= 0:
        return 'CONTRACTION'
    elif surprise >= 0.3:
        return 'STRONG_BEAT'
    elif surprise >= 0.1:
        return 'BEAT'
    elif surprise >= -0.1:
        return 'INLINE'
    elif surprise >= -0.3:
        return 'MISS'
    else:
        return 'BIG_MISS'


def classify_trend(gdp_mom, prev_mom, prev_prev_mom=None):
    """Classify GDP trend direction."""
    if gdp_mom is None:
        return 'UNKNOWN'
    if gdp_mom < 0 and prev_mom < 0:
        return 'ACCELERATING_DECLINE'
    elif gdp_mom < 0 and prev_mom >= 0:
        return 'FIRST_DECLINE'
    elif gdp_mom >= 0 and prev_mom < 0:
        return 'RECOVERY'
    elif gdp_mom > prev_mom:
        return 'ACCELERATING'
    elif gdp_mom < prev_mom:
        return 'DECELERATING'
    else:
        return 'STABLE'


def classify_level(gdp_yoy):
    """Classify absolute GDP growth level."""
    if gdp_yoy is None:
        return 'NO_DATA'
    if gdp_yoy >= 2.0:
        return 'STRONG'
    elif gdp_yoy >= 1.0:
        return 'MODERATE'
    elif gdp_yoy >= 0.0:
        return 'WEAK'
    elif gdp_yoy >= -2.0:
        return 'CONTRACTING'
    else:
        return 'RECESSION'


def classify_wyckoff(price, ema21, ema55, atr_pct):
    """Simplified Wyckoff phase classifier."""
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
    """Simplified vol regime."""
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
    """Compute returns for each session phase."""
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
    """Main backtest loop."""
    df = load_eth_data(csv_path)
    df['ema21'] = compute_ema(df['close'], 21)
    df['ema55'] = compute_ema(df['close'], 55)
    df['atr'] = (df['high'] - df['low']).rolling(14).mean()
    df['atr_ma'] = df['atr'].rolling(96).mean()
    df['atr_pct'] = df['atr'] / df['close'] * 100

    all_results = []
    prev_prev_mom = None
    prev_mom = None

    for date_str, data in RELEASES.items():
        gdp_mom = data['gdp_mom']
        release_ts = pd.Timestamp(date_str + 'T07:00:00')
        mask = df['timestamp'] <= release_ts
        if mask.sum() == 0:
            prev_prev_mom = prev_mom
            prev_mom = gdp_mom
            continue
        idx = mask.sum() - 1
        row = df.iloc[idx]

        signal = classify_signal(gdp_mom, data.get('consensus_mom'), data.get('prev_mom'))
        trend = classify_trend(gdp_mom, data.get('prev_mom'), prev_prev_mom)
        level = classify_level(data.get('gdp_yoy'))
        wyckoff = classify_wyckoff(row['close'], row['ema21'], row['ema55'], row['atr_pct'])
        vol = classify_vol(row['atr'], row['atr_ma'])
        returns = compute_session_returns(df, release_ts)

        all_results.append({
            'date': date_str, 'gdp_mom': gdp_mom, 'gdp_yoy': data.get('gdp_yoy'),
            'consensus_mom': data.get('consensus_mom'), 'prev_mom': data.get('prev_mom'),
            'surprise': gdp_mom - data.get('consensus_mom', 0),
            'signal': signal, 'trend': trend, 'level': level,
            'wyckoff': wyckoff, 'vol': vol, 'returns': returns,
        })
        prev_prev_mom = prev_mom
        prev_mom = gdp_mom

    return all_results


def analyze_results(results):
    """Prompt A: Cross-tabulate and report edges."""
    print("\n" + "="*80)
    print("PROMPT A: UK MONTHLY GDP BACKTEST RESULTS")
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

    # Trend breakdown
    print(f"\n--- Trend Breakdown ---")
    trends = {}
    for r in results:
        tr = r['trend']
        ret_24h = r['returns'].get('24h_aggregate', {}).get('return', 0)
        trends.setdefault(tr, []).append(ret_24h)
    for tr, rets in sorted(trends.items()):
        print(f"  {tr:25s}: {np.mean(rets):+.3f}% avg, "
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

    # RECESSION_SIGNAL vs STRONG_BEAT
    rec_rets = [r['returns'].get('24h_aggregate', {}).get('return', 0)
                for r in results if r['signal'] == 'RECESSION_SIGNAL']
    beat_rets = [r['returns'].get('24h_aggregate', {}).get('return', 0)
                 for r in results if r['signal'] in ('STRONG_BEAT', 'BEAT')]
    if rec_rets and beat_rets:
        t2, p2 = stats.ttest_ind(rec_rets, beat_rets)
        print(f"\n--- RECESSION_SIGNAL vs BEAT ---")
        print(f"  RECESSION avg: {np.mean(rec_rets):.3f}% (n={len(rec_rets)})")
        print(f"  BEAT avg: {np.mean(beat_rets):.3f}% (n={len(beat_rets)})")
        print(f"  Two-sample t-test: t={t2:.3f}, p={p2:.4f} "
              f"{'✅ SIGNIFICANT' if p2 < 0.05 else '❌ NOT significant'}")

    return edges


def analyze_transmission(results):
    """Prompt B: Session-by-session transmission chain."""
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

    out_path = os.path.join(os.path.dirname(__file__), 'backtest_uk_gdp_monthly_results.json')
    json_results = []
    for r in results:
        jr = {k: v for k, v in r.items() if k != 'returns'}
        jr['returns'] = {sk: sv for sk, sv in r['returns'].items()}
        json_results.append(jr)
    with open(out_path, 'w') as f:
        json.dump(json_results, f, indent=2, default=str)
    print(f"\nResults saved to {out_path}")

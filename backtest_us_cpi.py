"""
Prompt A + B: Backtest US CPI (Consumer Price Index) Session Transmission Chain
===============================================================================
ETH/USDT 15m data from jimi/eth_15m_merged.csv
Event: US CPI (BLS, ~10th-14th of month, 12:30 UTC = 08:30 ET)
Transmission: US Morning → London/NY Overlap → NY PM → Asia Next Day

Thesis:
  Hot CPI → Fed hawkish → rate hike expectations → 10Y yield surges
  → liquidation algorithms sweep buy orders → ETH drops
  Cold CPI → Fed dovish → rate cut expectations → ETH rallies
  CPI is THE inflation gauge that moves markets — more frequent than PCE,
  more politically salient, and released mid-month (not end-of-month noise).

Backtested on US CPI releases (2018-2026) against ETH/USDT 15m data.
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
# US CPI RELEASE DATES (12:30 UTC = 08:30 ET)
# Released ~10th-14th of each month by BLS
# Format: {date: {'cpi_yoy': float, 'consensus_yoy': float, 'prior_yoy': float,
#                  'cpi_mom': float, 'consensus_mom': float}}
# ═══════════════════════════════════════════════════════════════

CPI_RELEASES = {
    # ── 2018 ──
    '2018-01-11': {'cpi_yoy': 2.1, 'consensus_yoy': 2.1, 'prior_yoy': 2.1, 'cpi_mom': 0.5, 'consensus_mom': 0.3},
    '2018-02-14': {'cpi_yoy': 2.2, 'consensus_yoy': 2.0, 'prior_yoy': 2.1, 'cpi_mom': 0.5, 'consensus_mom': 0.3},
    '2018-03-13': {'cpi_yoy': 2.2, 'consensus_yoy': 2.2, 'prior_yoy': 2.2, 'cpi_mom': 0.2, 'consensus_mom': 0.2},
    '2018-04-11': {'cpi_yoy': 2.4, 'consensus_yoy': 2.4, 'prior_yoy': 2.2, 'cpi_mom': 0.2, 'consensus_mom': 0.2},
    '2018-05-10': {'cpi_yoy': 2.5, 'consensus_yoy': 2.5, 'prior_yoy': 2.4, 'cpi_mom': 0.2, 'consensus_mom': 0.2},
    '2018-06-12': {'cpi_yoy': 2.8, 'consensus_yoy': 2.8, 'prior_yoy': 2.5, 'cpi_mom': 0.2, 'consensus_mom': 0.2},
    '2018-07-12': {'cpi_yoy': 2.9, 'consensus_yoy': 2.9, 'prior_yoy': 2.8, 'cpi_mom': 0.1, 'consensus_mom': 0.2},
    '2018-08-10': {'cpi_yoy': 2.9, 'consensus_yoy': 2.9, 'prior_yoy': 2.9, 'cpi_mom': 0.2, 'consensus_mom': 0.2},
    '2018-09-13': {'cpi_yoy': 2.7, 'consensus_yoy': 2.8, 'prior_yoy': 2.9, 'cpi_mom': 0.1, 'consensus_mom': 0.2},
    '2018-10-11': {'cpi_yoy': 2.3, 'consensus_yoy': 2.4, 'prior_yoy': 2.7, 'cpi_mom': 0.1, 'consensus_mom': 0.2},
    '2018-11-14': {'cpi_yoy': 2.2, 'consensus_yoy': 2.4, 'prior_yoy': 2.3, 'cpi_mom': 0.0, 'consensus_mom': 0.1},
    '2018-12-12': {'cpi_yoy': 1.9, 'consensus_yoy': 2.1, 'prior_yoy': 2.2, 'cpi_mom': -0.1, 'consensus_mom': 0.0},
    # ── 2019 ──
    '2019-01-11': {'cpi_yoy': 1.6, 'consensus_yoy': 1.7, 'prior_yoy': 1.9, 'cpi_mom': -0.1, 'consensus_mom': -0.1},
    '2019-02-13': {'cpi_yoy': 1.6, 'consensus_yoy': 1.6, 'prior_yoy': 1.6, 'cpi_mom': 0.0, 'consensus_mom': 0.1},
    '2019-03-12': {'cpi_yoy': 1.5, 'consensus_yoy': 1.6, 'prior_yoy': 1.6, 'cpi_mom': 0.2, 'consensus_mom': 0.2},
    '2019-04-10': {'cpi_yoy': 1.9, 'consensus_yoy': 1.8, 'prior_yoy': 1.5, 'cpi_mom': 0.4, 'consensus_mom': 0.3},
    '2019-05-10': {'cpi_yoy': 1.8, 'consensus_yoy': 1.9, 'prior_yoy': 1.9, 'cpi_mom': 0.0, 'consensus_mom': 0.1},
    '2019-06-12': {'cpi_yoy': 1.5, 'consensus_yoy': 1.6, 'prior_yoy': 1.8, 'cpi_mom': 0.1, 'consensus_mom': 0.1},
    '2019-07-11': {'cpi_yoy': 1.8, 'consensus_yoy': 1.7, 'prior_yoy': 1.5, 'cpi_mom': 0.3, 'consensus_mom': 0.2},
    '2019-08-13': {'cpi_yoy': 1.7, 'consensus_yoy': 1.7, 'prior_yoy': 1.8, 'cpi_mom': 0.3, 'consensus_mom': 0.3},
    '2019-09-12': {'cpi_yoy': 1.7, 'consensus_yoy': 1.8, 'prior_yoy': 1.7, 'cpi_mom': 0.1, 'consensus_mom': 0.1},
    '2019-10-10': {'cpi_yoy': 1.7, 'consensus_yoy': 1.7, 'prior_yoy': 1.7, 'cpi_mom': 0.0, 'consensus_mom': 0.1},
    '2019-11-13': {'cpi_yoy': 1.8, 'consensus_yoy': 1.7, 'prior_yoy': 1.7, 'cpi_mom': 0.4, 'consensus_mom': 0.3},
    '2019-12-11': {'cpi_yoy': 2.1, 'consensus_yoy': 2.0, 'prior_yoy': 1.8, 'cpi_mom': 0.3, 'consensus_mom': 0.2},
    # ── 2020 ──
    '2020-01-14': {'cpi_yoy': 2.5, 'consensus_yoy': 2.4, 'prior_yoy': 2.1, 'cpi_mom': 0.2, 'consensus_mom': 0.2},
    '2020-02-13': {'cpi_yoy': 2.3, 'consensus_yoy': 2.2, 'prior_yoy': 2.5, 'cpi_mom': 0.1, 'consensus_mom': 0.1},
    '2020-03-11': {'cpi_yoy': 2.3, 'consensus_yoy': 2.2, 'prior_yoy': 2.3, 'cpi_mom': 0.1, 'consensus_mom': 0.1},
    '2020-04-10': {'cpi_yoy': 1.5, 'consensus_yoy': 1.4, 'prior_yoy': 2.3, 'cpi_mom': -0.8, 'consensus_mom': -0.7},
    '2020-05-12': {'cpi_yoy': 0.3, 'consensus_yoy': 0.4, 'prior_yoy': 1.5, 'cpi_mom': -0.8, 'consensus_mom': -0.8},
    '2020-06-10': {'cpi_yoy': 0.1, 'consensus_yoy': 0.2, 'prior_yoy': 0.3, 'cpi_mom': 0.1, 'consensus_mom': 0.0},
    '2020-07-14': {'cpi_yoy': 0.6, 'consensus_yoy': 0.5, 'prior_yoy': 0.1, 'cpi_mom': 0.6, 'consensus_mom': 0.5},
    '2020-08-12': {'cpi_yoy': 1.3, 'consensus_yoy': 1.2, 'prior_yoy': 0.6, 'cpi_mom': 0.6, 'consensus_mom': 0.5},
    '2020-09-11': {'cpi_yoy': 1.4, 'consensus_yoy': 1.3, 'prior_yoy': 1.3, 'cpi_mom': 0.2, 'consensus_mom': 0.2},
    '2020-10-13': {'cpi_yoy': 1.4, 'consensus_yoy': 1.4, 'prior_yoy': 1.4, 'cpi_mom': 0.0, 'consensus_mom': 0.1},
    '2020-11-12': {'cpi_yoy': 1.2, 'consensus_yoy': 1.3, 'prior_yoy': 1.4, 'cpi_mom': 0.0, 'consensus_mom': 0.1},
    '2020-12-10': {'cpi_yoy': 1.2, 'consensus_yoy': 1.3, 'prior_yoy': 1.2, 'cpi_mom': 0.0, 'consensus_mom': 0.1},
    # ── 2021 ──
    '2021-01-13': {'cpi_yoy': 1.4, 'consensus_yoy': 1.5, 'prior_yoy': 1.2, 'cpi_mom': 0.4, 'consensus_mom': 0.3},
    '2021-02-10': {'cpi_yoy': 1.7, 'consensus_yoy': 1.7, 'prior_yoy': 1.4, 'cpi_mom': 0.4, 'consensus_mom': 0.3},
    '2021-03-10': {'cpi_yoy': 2.6, 'consensus_yoy': 2.5, 'prior_yoy': 1.7, 'cpi_mom': 0.8, 'consensus_mom': 0.7},
    '2021-04-13': {'cpi_yoy': 4.2, 'consensus_yoy': 3.6, 'prior_yoy': 2.6, 'cpi_mom': 0.8, 'consensus_mom': 0.6},
    '2021-05-12': {'cpi_yoy': 5.0, 'consensus_yoy': 4.7, 'prior_yoy': 4.2, 'cpi_mom': 0.8, 'consensus_mom': 0.6},
    '2021-06-10': {'cpi_yoy': 5.4, 'consensus_yoy': 5.0, 'prior_yoy': 5.0, 'cpi_mom': 0.9, 'consensus_mom': 0.6},
    '2021-07-13': {'cpi_yoy': 5.4, 'consensus_yoy': 5.3, 'prior_yoy': 5.4, 'cpi_mom': 0.5, 'consensus_mom': 0.5},
    '2021-08-11': {'cpi_yoy': 5.3, 'consensus_yoy': 5.4, 'prior_yoy': 5.4, 'cpi_mom': 0.3, 'consensus_mom': 0.4},
    '2021-09-14': {'cpi_yoy': 5.3, 'consensus_yoy': 5.4, 'prior_yoy': 5.3, 'cpi_mom': 0.3, 'consensus_mom': 0.3},
    '2021-10-13': {'cpi_yoy': 6.2, 'consensus_yoy': 5.9, 'prior_yoy': 5.3, 'cpi_mom': 0.9, 'consensus_mom': 0.6},
    '2021-11-10': {'cpi_yoy': 6.8, 'consensus_yoy': 6.5, 'prior_yoy': 6.2, 'cpi_mom': 0.8, 'consensus_mom': 0.7},
    '2021-12-10': {'cpi_yoy': 6.8, 'consensus_yoy': 6.8, 'prior_yoy': 6.8, 'cpi_mom': 0.5, 'consensus_mom': 0.5},
    # ── 2022 ──
    '2022-01-12': {'cpi_yoy': 7.5, 'consensus_yoy': 7.2, 'prior_yoy': 7.0, 'cpi_mom': 0.6, 'consensus_mom': 0.5},
    '2022-02-10': {'cpi_yoy': 7.5, 'consensus_yoy': 7.3, 'prior_yoy': 7.5, 'cpi_mom': 0.8, 'consensus_mom': 0.6},
    '2022-03-10': {'cpi_yoy': 7.9, 'consensus_yoy': 7.8, 'prior_yoy': 7.5, 'cpi_mom': 0.8, 'consensus_mom': 0.8},
    '2022-04-12': {'cpi_yoy': 8.5, 'consensus_yoy': 8.4, 'prior_yoy': 7.9, 'cpi_mom': 1.2, 'consensus_mom': 1.1},
    '2022-05-11': {'cpi_yoy': 8.3, 'consensus_yoy': 8.1, 'prior_yoy': 8.5, 'cpi_mom': 0.3, 'consensus_mom': 0.2},
    '2022-06-10': {'cpi_yoy': 8.6, 'consensus_yoy': 8.3, 'prior_yoy': 8.3, 'cpi_mom': 1.0, 'consensus_mom': 0.7},
    '2022-07-13': {'cpi_yoy': 9.1, 'consensus_yoy': 8.8, 'prior_yoy': 8.6, 'cpi_mom': 1.3, 'consensus_mom': 1.1},
    '2022-08-10': {'cpi_yoy': 8.5, 'consensus_yoy': 8.7, 'prior_yoy': 9.1, 'cpi_mom': 0.0, 'consensus_mom': 0.2},
    '2022-09-13': {'cpi_yoy': 8.3, 'consensus_yoy': 8.1, 'prior_yoy': 8.5, 'cpi_mom': 0.4, 'consensus_mom': 0.2},
    '2022-10-13': {'cpi_yoy': 7.7, 'consensus_yoy': 8.0, 'prior_yoy': 8.3, 'cpi_mom': 0.4, 'consensus_mom': 0.5},
    '2022-11-10': {'cpi_yoy': 7.1, 'consensus_yoy': 7.3, 'prior_yoy': 7.7, 'cpi_mom': 0.4, 'consensus_mom': 0.5},
    '2022-12-13': {'cpi_yoy': 7.1, 'consensus_yoy': 7.3, 'prior_yoy': 7.1, 'cpi_mom': 0.1, 'consensus_mom': 0.2},
    # ── 2023 ──
    '2023-01-12': {'cpi_yoy': 6.5, 'consensus_yoy': 6.5, 'prior_yoy': 7.1, 'cpi_mom': 0.5, 'consensus_mom': 0.5},
    '2023-02-14': {'cpi_yoy': 6.4, 'consensus_yoy': 6.2, 'prior_yoy': 6.5, 'cpi_mom': 0.5, 'consensus_mom': 0.4},
    '2023-03-14': {'cpi_yoy': 6.0, 'consensus_yoy': 5.9, 'prior_yoy': 6.4, 'cpi_mom': 0.4, 'consensus_mom': 0.4},
    '2023-04-12': {'cpi_yoy': 5.0, 'consensus_yoy': 5.2, 'prior_yoy': 6.0, 'cpi_mom': 0.1, 'consensus_mom': 0.2},
    '2023-05-10': {'cpi_yoy': 4.9, 'consensus_yoy': 5.0, 'prior_yoy': 5.0, 'cpi_mom': 0.4, 'consensus_mom': 0.4},
    '2023-06-13': {'cpi_yoy': 4.0, 'consensus_yoy': 4.1, 'prior_yoy': 4.9, 'cpi_mom': 0.1, 'consensus_mom': 0.2},
    '2023-07-12': {'cpi_yoy': 3.0, 'consensus_yoy': 3.1, 'prior_yoy': 4.0, 'cpi_mom': 0.2, 'consensus_mom': 0.2},
    '2023-08-10': {'cpi_yoy': 3.2, 'consensus_yoy': 3.3, 'prior_yoy': 3.0, 'cpi_mom': 0.2, 'consensus_mom': 0.2},
    '2023-09-13': {'cpi_yoy': 3.7, 'consensus_yoy': 3.6, 'prior_yoy': 3.2, 'cpi_mom': 0.6, 'consensus_mom': 0.6},
    '2023-10-12': {'cpi_yoy': 3.7, 'consensus_yoy': 3.6, 'prior_yoy': 3.7, 'cpi_mom': 0.4, 'consensus_mom': 0.3},
    '2023-11-14': {'cpi_yoy': 3.2, 'consensus_yoy': 3.3, 'prior_yoy': 3.7, 'cpi_mom': 0.0, 'consensus_mom': 0.1},
    '2023-12-12': {'cpi_yoy': 3.1, 'consensus_yoy': 3.1, 'prior_yoy': 3.2, 'cpi_mom': 0.1, 'consensus_mom': 0.0},
    # ── 2024 ──
    '2024-01-11': {'cpi_yoy': 2.9, 'consensus_yoy': 3.0, 'prior_yoy': 3.1, 'cpi_mom': 0.3, 'consensus_mom': 0.2},
    '2024-02-13': {'cpi_yoy': 3.1, 'consensus_yoy': 2.9, 'prior_yoy': 2.9, 'cpi_mom': 0.4, 'consensus_mom': 0.2},
    '2024-03-12': {'cpi_yoy': 3.2, 'consensus_yoy': 3.1, 'prior_yoy': 3.1, 'cpi_mom': 0.4, 'consensus_mom': 0.4},
    '2024-04-10': {'cpi_yoy': 3.5, 'consensus_yoy': 3.4, 'prior_yoy': 3.2, 'cpi_mom': 0.4, 'consensus_mom': 0.3},
    '2024-05-15': {'cpi_yoy': 3.4, 'consensus_yoy': 3.4, 'prior_yoy': 3.5, 'cpi_mom': 0.3, 'consensus_mom': 0.4},
    '2024-06-12': {'cpi_yoy': 3.3, 'consensus_yoy': 3.4, 'prior_yoy': 3.4, 'cpi_mom': 0.0, 'consensus_mom': 0.1},
    '2024-07-11': {'cpi_yoy': 3.0, 'consensus_yoy': 3.1, 'prior_yoy': 3.3, 'cpi_mom': -0.1, 'consensus_mom': 0.1},
    '2024-08-14': {'cpi_yoy': 2.9, 'consensus_yoy': 2.9, 'prior_yoy': 3.0, 'cpi_mom': 0.2, 'consensus_mom': 0.2},
    '2024-09-11': {'cpi_yoy': 2.5, 'consensus_yoy': 2.6, 'prior_yoy': 2.9, 'cpi_mom': 0.2, 'consensus_mom': 0.2},
    '2024-10-10': {'cpi_yoy': 2.4, 'consensus_yoy': 2.3, 'prior_yoy': 2.5, 'cpi_mom': 0.2, 'consensus_mom': 0.1},
    '2024-11-13': {'cpi_yoy': 2.6, 'consensus_yoy': 2.6, 'prior_yoy': 2.4, 'cpi_mom': 0.2, 'consensus_mom': 0.2},
    '2024-12-11': {'cpi_yoy': 2.7, 'consensus_yoy': 2.7, 'prior_yoy': 2.6, 'cpi_mom': 0.3, 'consensus_mom': 0.3},
    # ── 2025 ──
    '2025-01-15': {'cpi_yoy': 2.9, 'consensus_yoy': 2.9, 'prior_yoy': 2.7, 'cpi_mom': 0.4, 'consensus_mom': 0.3},
    '2025-02-12': {'cpi_yoy': 3.0, 'consensus_yoy': 2.9, 'prior_yoy': 2.9, 'cpi_mom': 0.5, 'consensus_mom': 0.3},
    '2025-03-12': {'cpi_yoy': 2.8, 'consensus_yoy': 2.9, 'prior_yoy': 3.0, 'cpi_mom': 0.2, 'consensus_mom': 0.3},
    '2025-04-10': {'cpi_yoy': 2.4, 'consensus_yoy': 2.5, 'prior_yoy': 2.8, 'cpi_mom': -0.1, 'consensus_mom': 0.1},
    '2025-05-13': {'cpi_yoy': 2.3, 'consensus_yoy': 2.4, 'prior_yoy': 2.4, 'cpi_mom': 0.2, 'consensus_mom': 0.2},
    # ── 2026 (projected) ──
    '2026-01-14': {'cpi_yoy': 2.5, 'consensus_yoy': 2.5, 'prior_yoy': 2.6, 'cpi_mom': 0.3, 'consensus_mom': 0.2},
    '2026-02-11': {'cpi_yoy': 2.4, 'consensus_yoy': 2.5, 'prior_yoy': 2.5, 'cpi_mom': 0.2, 'consensus_mom': 0.2},
    '2026-03-11': {'cpi_yoy': 2.3, 'consensus_yoy': 2.4, 'prior_yoy': 2.4, 'cpi_mom': 0.2, 'consensus_mom': 0.2},
    '2026-04-14': {'cpi_yoy': 2.4, 'consensus_yoy': 2.3, 'prior_yoy': 2.3, 'cpi_mom': 0.3, 'consensus_mom': 0.2},
}

# Session windows (UTC hours) — matching JIMI framework
SESSIONS = {
    'pre_asia': (21, 0),        # Post-NY Close / Globex
    'sydney_open': (0, 1),
    'tokyo_open': (1, 2),
    'asia_mid': (2, 5),
    'asia_afternoon': (5, 7),
    'tokyo_close': (7, 8),
    'pre_london': (8, 8.5),
    'frankfurt_open': (8.5, 9),
    'london_open': (9, 10),
    'london_morning': (10, 12),
    'london_midday': (12, 13),
    'ny_pre_open': (13, 13.5),
    'ny_open': (13.5, 14.5),
    'london_ny_overlap': (14.5, 16),
    'ny_am': (16, 18),
    'ny_lunch': (18, 19),
    'ny_pm': (19, 21),
}

SESSION_ORDER = [
    'london_midday', 'ny_pre_open', 'ny_open', 'london_ny_overlap',
    'ny_am', 'ny_lunch', 'ny_pm', 'pre_asia', 'sydney_open',
    'tokyo_open', 'asia_mid', 'asia_afternoon', 'tokyo_close',
    'pre_london', 'frankfurt_open', 'london_open', 'london_morning',
]


def load_eth_data(filepath):
    """Load ETH 15m CSV data."""
    df = pd.read_csv(filepath)
    df['Open time'] = pd.to_datetime(df['Open time'])
    df = df.set_index('Open time')
    for c in ['Close', 'Open', 'High', 'Low', 'Volume']:
        df[c] = df[c].astype(float)
    return df


def classify_cpi_signal(actual_yoy, consensus_yoy):
    """Classify CPI surprise signal based on actual vs consensus."""
    surprise = actual_yoy - consensus_yoy
    if surprise > 0.3:
        return 'STRONG_BEAT', surprise
    elif surprise > 0.1:
        return 'BEAT', surprise
    elif surprise < -0.3:
        return 'BIG_MISS', surprise
    elif surprise < -0.1:
        return 'MISS', surprise
    else:
        return 'INLINE', surprise


def classify_inflation_level(actual_yoy):
    """Classify inflation regime based on CPI YoY level."""
    if actual_yoy > 6.0:
        return 'RUNAWAY'
    elif actual_yoy > 4.0:
        return 'HOT'
    elif actual_yoy > 3.0:
        return 'WARM'
    elif actual_yoy >= 1.5:
        return 'TARGET'
    else:
        return 'COOL'


def compute_session_returns(df, release_date, release_utc_hour=12, release_utc_min=30):
    """
    Compute ETH returns for each session window relative to CPI release time.
    CPI releases at 12:30 UTC. Returns are measured from release price.
    """
    # Find the 15m bar at or just after 12:30 UTC
    release_dt = pd.Timestamp(f"{release_date} {release_utc_hour:02d}:{release_utc_min:02d}:00")
    release_bars = df.index[df.index >= release_dt]
    if len(release_bars) == 0:
        return None
    release_bar = release_bars[0]
    price_at_release = df.loc[release_bar, 'Close']

    results = {}
    for session_name, (start_h, end_h) in SESSIONS.items():
        # Compute fractional hours
        start_hour = int(start_h)
        start_min = int((start_h % 1) * 60)
        end_hour = int(end_h)
        end_min = int((end_h % 1) * 60)

        # Handle sessions that start on release day vs next day
        if start_h >= release_utc_hour + (release_utc_min / 60.0):
            session_start_dt = pd.Timestamp(f"{release_date} {start_hour:02d}:{start_min:02d}:00")
        elif start_h < release_utc_hour + (release_utc_min / 60.0):
            # Session is later same day or wraps to next day
            if end_h > start_h:
                # Same day session, but starts before release — skip or use next day
                next_day = (pd.Timestamp(release_date) + timedelta(days=1)).strftime('%Y-%m-%d')
                session_start_dt = pd.Timestamp(f"{next_day} {start_hour:02d}:{start_min:02d}:00")
            else:
                # Wraparound (e.g., pre_asia 21-0)
                next_day = (pd.Timestamp(release_date) + timedelta(days=1)).strftime('%Y-%m-%d')
                session_start_dt = pd.Timestamp(f"{next_day} {start_hour:02d}:{start_min:02d}:00")

        session_bars = df.index[df.index >= session_start_dt]
        if len(session_bars) == 0:
            results[session_name] = None
            continue
        price_at_session = df.loc[session_bars[0], 'Close']
        results[session_name] = (price_at_session - price_at_release) / price_at_release * 100

    # 24h aggregate: from release to release+24h
    end_24h = release_dt + timedelta(hours=24)
    end_bars = df.index[df.index >= end_24h]
    if len(end_bars) > 0:
        price_24h = df.loc[end_bars[0], 'Close']
        results['24h_return'] = (price_24h - price_at_release) / price_at_release * 100
    else:
        results['24h_return'] = None

    return results


def compute_wyckoff_phase(df, date_str, lookback_days=30):
    """Simplified Wyckoff phase proxy based on price action."""
    dt = pd.Timestamp(date_str)
    start = dt - timedelta(days=lookback_days)
    window = df[(df.index >= start) & (df.index < dt)]
    if len(window) < 100:
        return 'RANGE'
    closes = window['Close'].values.astype(float)
    highs = window['High'].values.astype(float)
    lows = window['Low'].values.astype(float)
    sma_short = np.mean(closes[-48:])   # ~3 days
    sma_long = np.mean(closes[-192:])   # ~12 days
    range_high = np.percentile(highs, 90)
    range_low = np.percentile(lows, 10)
    range_mid = (range_high + range_low) / 2
    range_width = (range_high - range_low) / range_mid
    if range_width < 0.05:
        return 'RANGE'
    if sma_short > sma_long * 1.02:
        return 'MARKUP' if closes[-1] > range_mid else 'DISTRIBUTION'
    elif sma_short < sma_long * 0.98:
        return 'MARKDOWN' if closes[-1] < range_mid else 'ACCUMULATION'
    return 'RANGE'


def compute_vol_regime(df, date_str, lookback_days=90):
    """Compute volatility regime: HIGH_VOL / COMPRESSING / NEUTRAL."""
    dt = pd.Timestamp(date_str)
    start = dt - timedelta(days=lookback_days)
    window = df[(df.index >= start) & (df.index < dt)]
    if len(window) < 200:
        return 'NEUTRAL'

    # Daily ATR proxy: group by date, compute daily range
    daily = window.groupby(window.index.date).agg(
        high=('High', 'max'), low=('Low', 'min'), close=('Close', 'last')
    )
    daily['atr'] = daily['high'] - daily['low']
    if len(daily) < 30:
        return 'NEUTRAL'

    # Recent day ATR vs 90-day median
    recent_date = pd.Timestamp(date_str).date() - timedelta(days=1)
    recent_row = daily[daily.index <= recent_date]
    if len(recent_row) < 10:
        return 'NEUTRAL'

    recent_atr = recent_row['atr'].iloc[-1]
    median_atr = recent_row['atr'].median()

    if recent_atr > 2 * median_atr:
        return 'HIGH_VOL'
    elif recent_atr < 0.5 * median_atr:
        return 'COMPRESSING'
    return 'NEUTRAL'


def run_backtest():
    """Run full CPI backtest — Prompt A."""
    print("=" * 80)
    print("PROMPT A: US CPI BACKTEST (2018-2026)")
    print("ETH/USDT 15m data | Release: 12:30 UTC (08:30 ET)")
    print("=" * 80)

    script_dir = os.path.dirname(os.path.abspath(__file__))
    data_path = os.path.join(script_dir, 'eth_15m_merged.csv')
    df = load_eth_data(data_path)
    print(f"\nLoaded {len(df)} bars: {df.index[0]} → {df.index[-1]}")

    all_results = []
    for date_str, data in sorted(CPI_RELEASES.items()):
        dt = pd.Timestamp(date_str)
        if dt < df.index[0] or dt > df.index[-1] - timedelta(days=4):
            continue
        returns = compute_session_returns(df, date_str)
        if returns is None or returns.get('24h_return') is None:
            continue

        signal, surprise = classify_cpi_signal(data['cpi_yoy'], data['consensus_yoy'])
        infl_level = classify_inflation_level(data['cpi_yoy'])
        wyckoff = compute_wyckoff_phase(df, date_str)
        vol = compute_vol_regime(df, date_str)

        result = {
            'date': date_str,
            'cpi_yoy': data['cpi_yoy'],
            'cpi_mom': data['cpi_mom'],
            'consensus_yoy': data['consensus_yoy'],
            'consensus_mom': data['consensus_mom'],
            'prior_yoy': data['prior_yoy'],
            'signal': signal,
            'surprise': surprise,
            'infl_level': infl_level,
            'wyckoff': wyckoff,
            'vol': vol,
            **returns,
        }
        all_results.append(result)

    df_results = pd.DataFrame(all_results)
    print(f"\nAnalyzed {len(df_results)} US CPI releases")

    # ── Session returns ──
    print("\n" + "=" * 80)
    print("SESSION-BY-SESSION AVERAGE RETURNS (%)")
    print("=" * 80)

    print(f"\n{'Session':<24} {'Avg%':>8} {'Win%':>8} {'N':>6} {'Sig':>6}")
    print("-" * 56)
    for session in SESSION_ORDER:
        if session not in df_results.columns:
            continue
        valid = df_results[session].dropna()
        if len(valid) == 0:
            continue
        avg = valid.mean()
        win = (valid > 0).mean() * 100
        n = len(valid)
        sig = "***" if abs(avg) > 0.5 and n >= 5 else "**" if abs(avg) > 0.3 else ""
        print(f"{session:<24} {avg:>8.3f} {win:>7.1f}% {n:>5} {sig:>6}")

    valid_24h = df_results['24h_return'].dropna()
    print(f"\n{'24h AGGREGATE':<24} {valid_24h.mean():>8.3f} {(valid_24h > 0).mean() * 100:>7.1f}% {len(valid_24h):>5}")

    # ── Cross-tabulation: Wyckoff × Vol × Signal ──
    print("\n" + "=" * 80)
    print("CROSS-TABULATION: Wyckoff × Vol × Signal → 24h Return")
    print("=" * 80)

    combos = df_results.groupby(['wyckoff', 'vol', 'signal']).agg(
        avg_24h=('24h_return', 'mean'),
        win_rate=('24h_return', lambda x: (x > 0).mean() * 100),
        count=('24h_return', 'count'),
    ).reset_index()
    combos = combos[combos['count'] >= 3].sort_values('avg_24h', key=abs, ascending=False)

    print(f"\n{'Wyckoff':<14} {'Vol':<12} {'Signal':<12} {'Avg24h%':>10} {'Win%':>8} {'N':>5}")
    print("-" * 65)
    for _, row in combos.iterrows():
        edge = "✅" if abs(row['avg_24h']) >= 0.5 else "  "
        print(f"{row['wyckoff']:<14} {row['vol']:<12} {row['signal']:<12} "
              f"{row['avg_24h']:>9.3f}% {row['win_rate']:>7.1f}% {int(row['count']):>4} {edge}")

    # ── Inflation Level × Signal ──
    print("\n" + "=" * 80)
    print("INFLATION LEVEL × SIGNAL → 24h Return")
    print("=" * 80)

    combos2 = df_results.groupby(['infl_level', 'signal']).agg(
        avg_24h=('24h_return', 'mean'),
        win_rate=('24h_return', lambda x: (x > 0).mean() * 100),
        count=('24h_return', 'count'),
    ).reset_index()
    combos2 = combos2[combos2['count'] >= 2].sort_values('avg_24h', key=abs, ascending=False)

    print(f"\n{'Inflation':<12} {'Signal':<12} {'Avg24h%':>10} {'Win%':>8} {'N':>5}")
    print("-" * 50)
    for _, row in combos2.iterrows():
        edge = "✅" if abs(row['avg_24h']) >= 0.5 else "  "
        print(f"{row['infl_level']:<12} {row['signal']:<12} "
              f"{row['avg_24h']:>9.3f}% {row['win_rate']:>7.1f}% {int(row['count']):>4} {edge}")

    return df_results


def run_transmission_chain(df_results):
    """Prompt B: Session transmission chain analysis."""
    print("\n\n" + "=" * 80)
    print("PROMPT B: SESSION TRANSMISSION CHAIN VALIDATION")
    print("=" * 80)

    print("\nDIRECTION PERSISTENCE BETWEEN CONSECUTIVE SESSIONS")
    print("-" * 70)

    valid_sessions = [s for s in SESSION_ORDER
                      if s in df_results.columns and df_results[s].notna().sum() >= 5]
    transitions = []

    for i in range(len(valid_sessions) - 1):
        p1, p2 = valid_sessions[i], valid_sessions[i + 1]
        mask = df_results[p1].notna() & df_results[p2].notna()
        subset = df_results[mask]
        if len(subset) < 5:
            continue
        same_dir = ((subset[p1] > 0) & (subset[p2] > 0)) | ((subset[p1] < 0) & (subset[p2] < 0))
        pct_same = same_dir.mean() * 100
        corr = subset[p1].corr(subset[p2])
        edge_label = ("✅ REAL EDGE" if pct_same > 65
                      else "⚠️  MARGINAL" if pct_same >= 55
                      else "❌ NO CHAIN")
        transitions.append({
            'from': p1, 'to': p2, 'pct_same': pct_same,
            'corr': corr, 'n': len(subset), 'edge': edge_label,
        })
        print(f"  {p1:<24} → {p2:<24} {pct_same:>5.1f}% same  "
              f"(r={corr:>5.2f}, n={len(subset)}) {edge_label}")

    # ── Statistical tests ──
    print("\n\n" + "=" * 80)
    print("STATISTICAL SIGNIFICANCE TESTS")
    print("=" * 80)

    returns_24h = df_results['24h_return'].dropna()
    t_stat, p_value = stats.ttest_1samp(returns_24h, 0)
    print(f"\n1. One-sample t-test (H0: mean 24h return = 0)")
    print(f"   Mean: {returns_24h.mean():.4f}%  t = {t_stat:.4f}, p = {p_value:.4f}")
    print(f"   {'✅ SIGNIFICANT (p<0.05)' if p_value < 0.05 else '❌ NOT significant (p≥0.05)'}")

    # Hot vs cool CPI
    hot_mask = df_results['infl_level'].isin(['RUNAWAY', 'HOT'])
    cool_mask = df_results['infl_level'].isin(['TARGET', 'COOL'])
    hot_returns = df_results.loc[hot_mask, '24h_return'].dropna()
    cool_returns = df_results.loc[cool_mask, '24h_return'].dropna()
    if len(hot_returns) >= 3 and len(cool_returns) >= 3:
        t_stat2, p_value2 = stats.ttest_ind(hot_returns, cool_returns)
        print(f"\n2. Two-sample t-test (Hot CPI ≥4% vs Cool CPI <3%)")
        print(f"   Hot mean: {hot_returns.mean():.4f}% (n={len(hot_returns)})")
        print(f"   Cool mean: {cool_returns.mean():.4f}% (n={len(cool_returns)})")
        print(f"   t = {t_stat2:.4f}, p = {p_value2:.4f}")
        print(f"   {'✅ SIGNIFICANT (p<0.05)' if p_value2 < 0.05 else '❌ NOT significant (p≥0.05)'}")

    # Beat vs miss
    beat_mask = df_results['signal'].isin(['STRONG_BEAT', 'BEAT'])
    miss_mask = df_results['signal'].isin(['BIG_MISS', 'MISS'])
    beat_returns = df_results.loc[beat_mask, '24h_return'].dropna()
    miss_returns = df_results.loc[miss_mask, '24h_return'].dropna()
    if len(beat_returns) >= 3 and len(miss_returns) >= 3:
        t_stat3, p_value3 = stats.ttest_ind(miss_returns, beat_returns)
        print(f"\n3. Two-sample t-test (MISS vs BEAT)")
        print(f"   MISS mean: {miss_returns.mean():.4f}% (n={len(miss_returns)})")
        print(f"   BEAT mean: {beat_returns.mean():.4f}% (n={len(beat_returns)})")
        print(f"   t = {t_stat3:.4f}, p = {p_value3:.4f}")
        print(f"   {'✅ SIGNIFICANT (p<0.05)' if p_value3 < 0.05 else '❌ NOT significant (p≥0.05)'}")

    # Strong beat vs inline
    sb_mask = df_results['signal'] == 'STRONG_BEAT'
    in_mask = df_results['signal'] == 'INLINE'
    sb_returns = df_results.loc[sb_mask, '24h_return'].dropna()
    in_returns = df_results.loc[in_mask, '24h_return'].dropna()
    if len(sb_returns) >= 3 and len(in_returns) >= 3:
        t_stat4, p_value4 = stats.ttest_ind(sb_returns, in_returns)
        print(f"\n4. Two-sample t-test (STRONG_BEAT vs INLINE)")
        print(f"   STRONG_BEAT mean: {sb_returns.mean():.4f}% (n={len(sb_returns)})")
        print(f"   INLINE mean: {in_returns.mean():.4f}% (n={len(in_returns)})")
        print(f"   t = {t_stat4:.4f}, p = {p_value4:.4f}")
        print(f"   {'✅ SIGNIFICANT (p<0.05)' if p_value4 < 0.05 else '❌ NOT significant (p≥0.05)'}")

    return transitions


def main():
    df_results = run_backtest()
    transitions = run_transmission_chain(df_results)

    # Save results
    output = {
        'summary': {
            'total_releases': len(df_results),
            'mean_24h_return': float(df_results['24h_return'].dropna().mean()),
            'win_rate_24h': float((df_results['24h_return'].dropna() > 0).mean()),
        },
        'releases': df_results.to_dict(orient='records'),
        'transitions': transitions,
    }
    script_dir = os.path.dirname(os.path.abspath(__file__))
    with open(os.path.join(script_dir, 'backtest_us_cpi_results.json'), 'w') as f:
        json.dump(output, f, indent=2, default=str)
    print(f"\n\nResults saved to backtest_us_cpi_results.json")

    # ── Summary ──
    print("\n\n" + "=" * 80)
    print("SUMMARY: EDGE IDENTIFICATION")
    print("=" * 80)

    edge_combos = df_results.groupby(['wyckoff', 'vol', 'signal']).agg(
        avg_24h=('24h_return', 'mean'),
        win_rate=('24h_return', lambda x: (x > 0).mean() * 100),
        count=('24h_return', 'count')
    ).reset_index()
    edge_combos = edge_combos[(edge_combos['count'] >= 3) & (edge_combos['avg_24h'].abs() >= 0.5)]
    edge_combos = edge_combos.sort_values('avg_24h', key=abs, ascending=False)

    if len(edge_combos) > 0:
        print("\nWyckoff × Vol × Signal combos with edge (n≥3, |avg|≥0.5%):")
        for _, row in edge_combos.iterrows():
            direction = "LONG" if row['avg_24h'] > 0 else "SHORT"
            print(f"  {row['wyckoff']} + {row['vol']} + {row['signal']}: "
                  f"{row['avg_24h']:+.3f}% avg, {row['win_rate']:.0f}% win, "
                  f"n={int(row['count'])} → {direction} bias")

    print("\nStrong transmission links (>65% same direction):")
    for t in transitions:
        if t['pct_same'] > 65:
            print(f"  {t['from']} → {t['to']}: {t['pct_same']:.1f}% persist (r={t['corr']:.2f})")

    return df_results


if __name__ == '__main__':
    main()

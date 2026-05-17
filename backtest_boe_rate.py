#!/usr/bin/env python3
"""
Prompt A + B: Backtest BoE Rate Decision + MPC Vote Split (2018-today)
using ETH/USDT 15m data.

BoE announces rate decision at 12:00 GMT = 12:00 UTC = 20:00 MYT.
Falls in the Europe Mid-Day session.

Session itinerary (per user's thesis #34):
  Europe Mid-Day (12:00 UTC) → US Session Pre-Market → Next UK CPI loopback
  - MPC vote split is the key signal
  - Dovish cut → GBP drops → DXY up → ETH selling pressure
  - DXY shift carries into US pre-market for cross-currency hedging
  - Loopback: next UK GDP release validates policy path
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from scipy import stats
import json
import os

# ═══════════════════════════════════════════════════════════════
# BOE RATE DECISION DATES + OUTCOMES
# Decision at 12:00 GMT = 12:00 UTC = 20:00 MYT
# Format: {date: {'rate': float, 'prev_rate': float, 'signal': str,
#                 'vote_hike': int, 'vote_hold': int, 'vote_cut': int}}
# vote_* = number of MPC members voting for each action
# ═══════════════════════════════════════════════════════════════

RELEASES = {
    # 2018 — post-Brexit vote, gradual normalization
    '2018-02-08': {'rate': 0.50, 'prev_rate': 0.50, 'signal': 'HOLD', 'vote_hike': 0, 'vote_hold': 9, 'vote_cut': 0, 'note': 'Unanimous hold'},
    '2018-03-22': {'rate': 0.50, 'prev_rate': 0.50, 'signal': 'HOLD', 'vote_hike': 2, 'vote_hold': 7, 'vote_cut': 0, 'note': '2 dissented for hike (Saunders, Haldane)'},
    '2018-05-10': {'rate': 0.50, 'prev_rate': 0.50, 'signal': 'HOLD', 'vote_hike': 0, 'vote_hold': 9, 'vote_cut': 0, 'note': 'Unanimous hold, cut hike expectations'},
    '2018-06-21': {'rate': 0.50, 'prev_rate': 0.50, 'signal': 'HOLD', 'vote_hike': 3, 'vote_hold': 6, 'vote_cut': 0, 'note': '3 dissented for hike'},
    '2018-08-02': {'rate': 0.75, 'prev_rate': 0.50, 'signal': 'HIKE', 'vote_hike': 9, 'vote_hold': 0, 'vote_cut': 0, 'note': 'Unanimous hike'},
    '2018-09-13': {'rate': 0.75, 'prev_rate': 0.75, 'signal': 'HOLD', 'vote_hike': 0, 'vote_hold': 9, 'vote_cut': 0, 'note': 'Unanimous hold'},
    '2018-11-01': {'rate': 0.75, 'prev_rate': 0.75, 'signal': 'HOLD', 'vote_hike': 0, 'vote_hold': 9, 'vote_cut': 0, 'note': 'Unanimous hold'},
    '2018-12-20': {'rate': 0.75, 'prev_rate': 0.75, 'signal': 'HOLD', 'vote_hike': 0, 'vote_hold': 9, 'vote_cut': 0, 'note': 'Unanimous hold, Brexit uncertainty'},

    # 2019 — Brexit limbo
    '2019-02-07': {'rate': 0.75, 'prev_rate': 0.75, 'signal': 'HOLD', 'vote_hike': 0, 'vote_hold': 9, 'vote_cut': 0, 'note': 'Unanimous hold'},
    '2019-03-21': {'rate': 0.75, 'prev_rate': 0.75, 'signal': 'HOLD', 'vote_hike': 0, 'vote_hold': 9, 'vote_cut': 0, 'note': 'Unanimous hold'},
    '2019-05-02': {'rate': 0.75, 'prev_rate': 0.75, 'signal': 'HOLD', 'vote_hike': 0, 'vote_hold': 9, 'vote_cut': 0, 'note': 'Unanimous hold'},
    '2019-06-20': {'rate': 0.75, 'prev_rate': 0.75, 'signal': 'HOLD', 'vote_hike': 0, 'vote_hold': 9, 'vote_cut': 0, 'note': 'Unanimous hold'},
    '2019-08-01': {'rate': 0.75, 'prev_rate': 0.75, 'signal': 'HOLD', 'vote_hike': 0, 'vote_hold': 9, 'vote_cut': 0, 'note': 'Unanimous hold'},
    '2019-09-19': {'rate': 0.75, 'prev_rate': 0.75, 'signal': 'HOLD', 'vote_hike': 0, 'vote_hold': 9, 'vote_cut': 0, 'note': 'Unanimous hold'},
    '2019-11-07': {'rate': 0.75, 'prev_rate': 0.75, 'signal': 'HOLD', 'vote_hike': 0, 'vote_hold': 7, 'vote_cut': 2, 'note': '2 dissented for cut (Haskel, Saunders)'},
    '2019-12-19': {'rate': 0.75, 'prev_rate': 0.75, 'signal': 'HOLD', 'vote_hike': 0, 'vote_hold': 7, 'vote_cut': 2, 'note': '2 dissented for cut'},

    # 2020 — COVID response
    '2020-01-30': {'rate': 0.75, 'prev_rate': 0.75, 'signal': 'HOLD', 'vote_hike': 0, 'vote_hold': 7, 'vote_cut': 2, 'note': '2 dissented for cut'},
    '2020-03-11': {'rate': 0.25, 'prev_rate': 0.75, 'signal': 'CUT', 'vote_hike': 0, 'vote_hold': 0, 'vote_cut': 9, 'note': 'Emergency cut -50bp, unanimous'},
    '2020-03-26': {'rate': 0.10, 'prev_rate': 0.25, 'signal': 'CUT', 'vote_hike': 0, 'vote_hold': 0, 'vote_cut': 9, 'note': 'Emergency cut -15bp, QE expanded'},
    '2020-05-07': {'rate': 0.10, 'prev_rate': 0.10, 'signal': 'HOLD', 'vote_hike': 0, 'vote_hold': 7, 'vote_cut': 2, 'note': '2 dissented for cut to 0'},
    '2020-06-18': {'rate': 0.10, 'prev_rate': 0.10, 'signal': 'HOLD', 'vote_hike': 0, 'vote_hold': 7, 'vote_cut': 2, 'note': 'QE expanded £100B'},
    '2020-08-06': {'rate': 0.10, 'prev_rate': 0.10, 'signal': 'HOLD', 'vote_hike': 0, 'vote_hold': 9, 'vote_cut': 0, 'note': 'Unanimous hold'},
    '2020-09-17': {'rate': 0.10, 'prev_rate': 0.10, 'signal': 'HOLD', 'vote_hike': 0, 'vote_hold': 9, 'vote_cut': 0, 'note': 'Reviewed negative rates'},
    '2020-11-05': {'rate': 0.10, 'prev_rate': 0.10, 'signal': 'HOLD', 'vote_hike': 0, 'vote_hold': 9, 'vote_cut': 0, 'note': 'QE expanded £150B'},
    '2020-12-17': {'rate': 0.10, 'prev_rate': 0.10, 'signal': 'HOLD', 'vote_hike': 0, 'vote_hold': 9, 'vote_cut': 0, 'note': 'Unanimous hold'},

    # 2021 — holding, inflation building
    '2021-02-04': {'rate': 0.10, 'prev_rate': 0.10, 'signal': 'HOLD', 'vote_hike': 0, 'vote_hold': 9, 'vote_cut': 0, 'note': 'Unanimous hold'},
    '2021-03-18': {'rate': 0.10, 'prev_rate': 0.10, 'signal': 'HOLD', 'vote_hike': 0, 'vote_hold': 9, 'vote_cut': 0, 'note': 'QE maintained'},
    '2021-05-06': {'rate': 0.10, 'prev_rate': 0.10, 'signal': 'HOLD', 'vote_hike': 0, 'vote_hold': 9, 'vote_cut': 0, 'note': 'QE tapered £50B'},
    '2021-06-24': {'rate': 0.10, 'prev_rate': 0.10, 'signal': 'HOLD', 'vote_hike': 0, 'vote_hold': 9, 'vote_cut': 0, 'note': 'Inflation rising'},
    '2021-08-05': {'rate': 0.10, 'prev_rate': 0.10, 'signal': 'HOLD', 'vote_hike': 0, 'vote_hold': 9, 'vote_cut': 0, 'note': 'QE tapered'},
    '2021-09-23': {'rate': 0.10, 'prev_rate': 0.10, 'signal': 'HOLD', 'vote_hike': 0, 'vote_hold': 9, 'vote_cut': 0, 'note': 'Hawkish tilt'},
    '2021-11-04': {'rate': 0.10, 'prev_rate': 0.10, 'signal': 'HOLD', 'vote_hike': 0, 'vote_hold': 7, 'vote_cut': 2, 'note': '2 dissented for hike (Ramsden, Saunders)'},
    '2021-12-16': {'rate': 0.25, 'prev_rate': 0.10, 'signal': 'HIKE', 'vote_hike': 8, 'vote_hold': 1, 'vote_cut': 0, 'note': 'Surprise hike, 1 dissented for hold (Tenreyro)'},

    # 2022 — aggressive tightening
    '2022-02-03': {'rate': 0.50, 'prev_rate': 0.25, 'signal': 'HIKE', 'vote_hike': 9, 'vote_hold': 0, 'vote_cut': 0, 'note': 'Unanimous hike'},
    '2022-03-17': {'rate': 0.75, 'prev_rate': 0.50, 'signal': 'HIKE', 'vote_hike': 8, 'vote_hold': 1, 'vote_cut': 0, 'note': '1 dissented for hold'},
    '2022-05-05': {'rate': 1.00, 'prev_rate': 0.75, 'signal': 'HIKE', 'vote_hike': 6, 'vote_hold': 3, 'vote_cut': 0, 'note': '3 dissented for hold (dovish split)'},
    '2022-06-16': {'rate': 1.25, 'prev_rate': 1.00, 'signal': 'HIKE', 'vote_hike': 6, 'vote_hold': 3, 'vote_cut': 0, 'note': '3 dissented for hold'},
    '2022-08-04': {'rate': 1.75, 'prev_rate': 1.25, 'signal': 'HIKE', 'vote_hike': 9, 'vote_hold': 0, 'vote_cut': 0, 'note': 'Unanimous +50bp'},
    '2022-09-22': {'rate': 2.25, 'prev_rate': 1.75, 'signal': 'HIKE', 'vote_hike': 5, 'vote_hold': 3, 'vote_cut': 1, 'note': '3-way split, 1 voted for cut'},
    '2022-11-03': {'rate': 3.00, 'prev_rate': 2.25, 'signal': 'HIKE', 'vote_hike': 7, 'vote_hold': 2, 'vote_cut': 0, 'note': 'Largest hike in 33 years'},
    '2022-12-15': {'rate': 3.50, 'prev_rate': 3.00, 'signal': 'HIKE', 'vote_hike': 6, 'vote_hold': 3, 'vote_cut': 0, 'note': '2-way split'},

    # 2023 — peak rates
    '2023-02-02': {'rate': 4.00, 'prev_rate': 3.50, 'signal': 'HIKE', 'vote_hike': 7, 'vote_hold': 2, 'vote_cut': 0, 'note': 'Dropped "forcefully" language'},
    '2023-03-23': {'rate': 4.25, 'prev_rate': 4.00, 'signal': 'HIKE', 'vote_hike': 7, 'vote_hold': 2, 'vote_cut': 0, 'note': 'SVB fallout, 2 dissented'},
    '2023-05-11': {'rate': 4.50, 'prev_rate': 4.25, 'signal': 'HIKE', 'vote_hike': 7, 'vote_hold': 2, 'vote_cut': 0, 'note': '+25bp'},
    '2023-06-22': {'rate': 5.00, 'prev_rate': 4.50, 'signal': 'HIKE', 'vote_hike': 7, 'vote_hold': 2, 'vote_cut': 0, 'note': 'Surprise +50bp'},
    '2023-08-03': {'rate': 5.25, 'prev_rate': 5.00, 'signal': 'HIKE', 'vote_hike': 6, 'vote_hold': 3, 'vote_cut': 0, 'note': 'Peak rate, 3-way split begins'},
    '2023-09-21': {'rate': 5.25, 'prev_rate': 5.25, 'signal': 'HOLD', 'vote_hike': 0, 'vote_hold': 9, 'vote_cut': 0, 'note': 'First hold after 14 hikes'},
    '2023-11-02': {'rate': 5.25, 'prev_rate': 5.25, 'signal': 'HOLD', 'vote_hike': 0, 'vote_hold': 6, 'vote_cut': 3, 'note': '3 dissented for cut'},
    '2023-12-14': {'rate': 5.25, 'prev_rate': 5.25, 'signal': 'HOLD', 'vote_hike': 0, 'vote_hold': 6, 'vote_cut': 3, 'note': '3 dissented for cut, dovish pivot'},

    # 2024 — cutting cycle
    '2024-02-01': {'rate': 5.25, 'prev_rate': 5.25, 'signal': 'HOLD', 'vote_hike': 0, 'vote_hold': 6, 'vote_cut': 3, 'note': '3 dissented for cut'},
    '2024-03-21': {'rate': 5.25, 'prev_rate': 5.25, 'signal': 'HOLD', 'vote_hike': 0, 'vote_hold': 8, 'vote_cut': 1, 'note': '1 dissented for cut (Dhingra)'},
    '2024-05-09': {'rate': 5.25, 'prev_rate': 5.25, 'signal': 'HOLD', 'vote_hike': 0, 'vote_hold': 7, 'vote_cut': 2, 'note': '2 dissented for cut'},
    '2024-06-20': {'rate': 5.25, 'prev_rate': 5.25, 'signal': 'HOLD', 'vote_hike': 0, 'vote_hold': 7, 'vote_cut': 2, 'note': '2 dissented, signaled Aug cut'},
    '2024-08-01': {'rate': 5.00, 'prev_rate': 5.25, 'signal': 'CUT', 'vote_hike': 0, 'vote_hold': 5, 'vote_cut': 4, 'note': 'First cut since 2020, razor-thin 5-4'},
    '2024-09-19': {'rate': 5.00, 'prev_rate': 5.00, 'signal': 'HOLD', 'vote_hike': 0, 'vote_hold': 8, 'vote_cut': 1, 'note': '1 dissented for cut'},
    '2024-11-07': {'rate': 4.75, 'prev_rate': 5.00, 'signal': 'CUT', 'vote_hike': 0, 'vote_hold': 3, 'vote_cut': 6, 'note': 'Cut -25bp'},
    '2024-12-19': {'rate': 4.75, 'prev_rate': 4.75, 'signal': 'HOLD', 'vote_hike': 0, 'vote_hold': 6, 'vote_cut': 3, 'note': '3 dissented for cut'},

    # 2025 — gradual easing
    '2025-02-06': {'rate': 4.50, 'prev_rate': 4.75, 'signal': 'CUT', 'vote_hike': 0, 'vote_hold': 3, 'vote_cut': 6, 'note': 'Cut -25bp'},
    '2025-03-20': {'rate': 4.50, 'prev_rate': 4.50, 'signal': 'HOLD', 'vote_hike': 0, 'vote_hold': 6, 'vote_cut': 3, 'note': '3 dissented for cut'},
    '2025-05-08': {'rate': 4.25, 'prev_rate': 4.50, 'signal': 'CUT', 'vote_hike': 0, 'vote_hold': 2, 'vote_cut': 7, 'note': 'Cut -25bp, tariff uncertainty'},
    '2025-06-19': {'rate': 4.25, 'prev_rate': 4.25, 'signal': 'HOLD', 'vote_hike': 0, 'vote_hold': 5, 'vote_cut': 4, 'note': '4 dissented for cut'},
    '2025-08-07': {'rate': 4.00, 'prev_rate': 4.25, 'signal': 'CUT', 'vote_hike': 0, 'vote_hold': 4, 'vote_cut': 5, 'note': 'Cut -25bp'},
    '2025-09-18': {'rate': 4.00, 'prev_rate': 4.00, 'signal': 'HOLD', 'vote_hike': 0, 'vote_hold': 6, 'vote_cut': 3, 'note': 'Hold'},
    '2025-11-06': {'rate': 3.75, 'prev_rate': 4.00, 'signal': 'CUT', 'vote_hike': 0, 'vote_hold': 3, 'vote_cut': 6, 'note': 'Cut -25bp'},
    '2025-12-18': {'rate': 3.75, 'prev_rate': 3.75, 'signal': 'HOLD', 'vote_hike': 0, 'vote_hold': 5, 'vote_cut': 4, 'note': 'Hold'},

    # 2026
    '2026-02-05': {'rate': 3.75, 'prev_rate': 3.75, 'signal': 'HOLD', 'vote_hike': 0, 'vote_hold': 6, 'vote_cut': 3, 'note': 'Hold'},
    '2026-03-19': {'rate': 3.50, 'prev_rate': 3.75, 'signal': 'CUT', 'vote_hike': 0, 'vote_hold': 2, 'vote_cut': 7, 'note': 'Cut -25bp'},
    '2026-04-30': {'rate': 3.50, 'prev_rate': 3.50, 'signal': 'HOLD', 'vote_hike': 0, 'vote_hold': 5, 'vote_cut': 4, 'note': 'Hold, 4 dissented for cut'},
}


# ═══════════════════════════════════════════════════════════════
# SESSION DEFINITIONS (BoE at 12:00 UTC = 20:00 MYT)
# Europe Mid-Day → US Pre-Market → Next day
# ═══════════════════════════════════════════════════════════════

SESSION_CHAIN = [
    # Pre-release (Europe morning)
    ('Pre-Asia', [0, 1, 2]),
    ('Sydney Open', [22, 23, 0, 1]),
    ('Tokyo Open', [0, 1, 2]),
    ('Asia Mid', [3, 4, 5]),
    ('Asia Afternoon', [5, 6, 7]),
    ('Tokyo Close', [6, 7]),
    ('Pre-London', [7, 8]),
    ('Frankfurt Open', [7, 8]),
    ('London Open', [8, 9]),
    ('London Morning', [9, 10, 11]),

    # Release window (Europe mid-day)
    ('BoE Decision (1h)', [12, 13]),       # 12:00-13:00 UTC
    ('London Midday', [12, 13]),
    ('NY Pre-Open', [12, 13]),
    ('NY Open', [13, 14]),
    ('London-NY Overlap', [14, 15]),

    # Post-release
    ('NY AM', [14, 15, 16]),
    ('NY Lunch', [17]),
    ('NY PM', [18, 19, 20]),

    # Aggregate windows
    ('BoE Window (2h)', [12, 13, 14]),     # 12:00-14:00 (decision + reaction)
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


def classify_wyckoff_phase(df, release_dt):
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


def classify_boe_signal(rate, prev_rate, signal_hint, vote_hike, vote_hold, vote_cut):
    """
    Classify BoE signal incorporating MPC vote split.
    HIKE → hawkish → GBP up → DXY down → ETH LONG
    CUT → dovish → GBP down → DXY up → ETH SHORT
    DOVISH_HOLD → growing cut votes → GBP down → DXY up → ETH SHORT
    HAWKISH_HOLD → growing hike votes → GBP up → DXY down → ETH LONG
    """
    if signal_hint == 'HIKE':
        return 'HIKE'
    elif signal_hint == 'CUT':
        return 'CUT'

    # For HOLD: use vote split to classify
    if vote_cut >= 3:
        return 'DOVISH_HOLD'
    elif vote_hike >= 3:
        return 'HAWKISH_HOLD'
    return 'NEUTRAL_HOLD'


def run_backtest(csv_path):
    df = load_eth_data(csv_path)
    results = []
    sorted_dates = sorted(RELEASES.keys())

    for i, date_str in enumerate(sorted_dates):
        release_data = RELEASES[date_str]
        release_dt = datetime.strptime(date_str, '%Y-%m-%d').replace(hour=12, minute=0)

        bars_after = get_bars_between(df, release_dt, release_dt + timedelta(hours=24))
        if len(bars_after) < 4:
            continue

        pre_bar = get_bar_at(df, release_dt)
        if pre_bar is None:
            continue

        pre_price = float(pre_bar['Close'])

        rate = release_data['rate']
        prev_rate = release_data['prev_rate']
        signal_hint = release_data.get('signal', 'HOLD')
        vote_hike = release_data.get('vote_hike', 0)
        vote_hold = release_data.get('vote_hold', 9)
        vote_cut = release_data.get('vote_cut', 0)
        signal = classify_boe_signal(rate, prev_rate, signal_hint, vote_hike, vote_hold, vote_cut)
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
            'rate': rate, 'prev_rate': prev_rate,
            'rate_change': round(rate - prev_rate, 2),
            'signal': signal, 'signal_hint': signal_hint,
            'vote_hike': vote_hike, 'vote_hold': vote_hold, 'vote_cut': vote_cut,
            'vote_split': f"{vote_hike}-{vote_hold}-{vote_cut}",
            'note': release_data.get('note', ''),
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
    """Prompt B: Session-by-session transmission chain validation."""
    print("\n" + "="*80)
    print("SESSION TRANSMISSION CHAIN (Prompt B)")
    print("="*80)
    valid = results[results['ret_24h'].notna()]
    key_sessions = ['London Open', 'London Morning', 'BoE Decision (1h)',
                    'London Midday', 'NY Pre-Open', 'NY Open',
                    'London-NY Overlap', 'NY AM', 'NY PM']

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

    print(f"\n{'Transition':<50} {'Same Dir%':>10} {'n':>5} {'Verdict':>10}")
    print("-"*80)
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
            print(f"{s1 + ' → ' + s2:<50} {pct:>9.0f}% {total:>5} {verdict:>10}")
            transitions.append({'from': s1, 'to': s2, 'same_dir_pct': round(pct, 1),
                                'n': total, 'verdict': verdict})
    return chain_data, transitions


def direction_persistence_to_reopen(results):
    print("\n" + "="*80)
    print("DIRECTION PERSISTENCE: Session → End of Cycle (NY PM)")
    print("="*80)
    valid = results[results['ret_24h'].notna()]

    sessions_ordered = [
        'London Open', 'London Morning', 'BoE Decision (1h)',
        'London Midday', 'NY Pre-Open', 'NY Open',
        'London-NY Overlap', 'NY AM'
    ]

    print(f"\n{'From Session':<25} {'To Session':<25} {'Same Dir%':>10} {'n':>5} {'Verdict':>10}")
    print("-"*80)

    for s1 in sessions_ordered:
        s2 = 'NY PM'
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
            verdict = '✅ PERSISTS' if pct > 65 else '⚠️ MARGINAL' if pct > 55 else '❌ DECAYS'
            print(f"{s1:<25} {'NY PM (end)':<25} {pct:>9.0f}% {total:>5} {verdict:>10}")


def statistical_tests(results):
    print("\n" + "="*80)
    print("STATISTICAL SIGNIFICANCE")
    print("="*80)
    valid = results[results['ret_24h'].notna()]
    rets = valid['ret_24h'].values

    if len(rets) < 3:
        print("Not enough data for statistical tests (n<3)")
        return

    t_stat, p_val = stats.ttest_1samp(rets, 0)
    print(f"\n1. One-sample t-test (24h return vs 0):")
    print(f"   Mean: {np.mean(rets):+.3f}%  t={t_stat:.3f}  p={p_val:.4f}  {'*** SIGNIFICANT' if p_val < 0.05 else 'NOT significant'}")

    signal_groups = {}
    for sig in ['HIKE', 'CUT', 'HOLD', 'DOVISH_HOLD', 'HAWKISH_HOLD', 'NEUTRAL_HOLD']:
        sig_rets = valid[valid['signal'] == sig]['ret_24h'].values
        if len(sig_rets) >= 3:
            signal_groups[sig] = sig_rets

    test_num = 2
    sig_names = list(signal_groups.keys())
    for i in range(len(sig_names)):
        for j in range(i+1, len(sig_names)):
            n1, n2 = sig_names[i], sig_names[j]
            r1, r2 = signal_groups[n1], signal_groups[n2]
            if len(r1) >= 3 and len(r2) >= 3:
                t2, p2 = stats.ttest_ind(r1, r2)
                print(f"\n{test_num}. Two-sample t-test ({n1} vs {n2}):")
                print(f"   {n1}: {np.mean(r1):+.3f}% (n={len(r1)})  {n2}: {np.mean(r2):+.3f}% (n={len(r2)})")
                print(f"   t={t2:.3f}  p={p2:.4f}  {'*** SIGNIFICANT' if p2 < 0.05 else 'NOT significant'}")
                test_num += 1

    for sig, sig_rets in signal_groups.items():
        t5, p5 = stats.ttest_1samp(sig_rets, 0)
        print(f"\n{test_num}. {sig} one-sample t-test (n={len(sig_rets)}):")
        print(f"   Mean: {np.mean(sig_rets):+.3f}%  t={t5:.3f}  p={p5:.4f}  {'***' if p5 < 0.05 else 'ns'}")
        test_num += 1

    # Vote split analysis
    print(f"\n{test_num}. MPC Vote Split Analysis:")
    print(f"   {'Vote Split':<15} {'Avg%':>8} {'Win%':>7} {'n':>4}")
    print(f"   {'-'*40}")
    for vote_str in sorted(valid['vote_split'].unique()):
        vd = valid[valid['vote_split'] == vote_str]
        if len(vd) >= 2:
            print(f"   {vote_str:<15} {vd['ret_24h'].mean():>+8.3f} {(vd['ret_24h'] > 0).mean()*100:>6.0f}% {len(vd):>4}")
    test_num += 1

    print(f"\n{test_num}. Year-by-year 24h returns:")
    print(f"   {'Year':<6} {'Avg%':>8} {'Win%':>7} {'n':>4}")
    print(f"   {'-'*30}")
    for year in sorted(valid['year'].unique()):
        yr = valid[valid['year'] == year]
        print(f"   {year:<6} {yr['ret_24h'].mean():>+8.3f} {(yr['ret_24h'] > 0).mean()*100:>6.0f}% {len(yr):>4}")


def main():
    csv_path = os.path.join(os.path.dirname(__file__), 'eth_15m_merged.csv')
    print("="*80)
    print("PROMPT A: BOE RATE DECISION + MPC VOTE BACKTEST (2018-2026)")
    print("="*80)
    print("Loading ETH 15m data...")
    results = run_backtest(csv_path)
    print(f"Analyzed {len(results)} BoE rate decisions ({results['year'].min()}-{results['year'].max()})")

    valid = results[results['ret_24h'].notna()]
    print(f"\n24h Aggregate Returns:")
    print(f"  Mean:   {valid['ret_24h'].mean():+.3f}%")
    print(f"  Median: {valid['ret_24h'].median():+.3f}%")
    print(f"  Std:    {valid['ret_24h'].std():.3f}%")
    print(f"  Win%:   {(valid['ret_24h'] > 0).mean()*100:.1f}%")
    print(f"  n:      {len(valid)}")

    print("\n  By Signal:")
    for sig in ['HIKE', 'CUT', 'DOVISH_HOLD', 'HAWKISH_HOLD', 'NEUTRAL_HOLD']:
        sig_data = valid[valid['signal'] == sig]
        if len(sig_data) > 0:
            print(f"  {sig:<18} avg={sig_data['ret_24h'].mean():+.3f}%  win={(sig_data['ret_24h'] > 0).mean()*100:.0f}%  n={len(sig_data)}")

    print("\n  By Vote Split (top combos):")
    for vs in valid['vote_split'].value_counts().head(10).index:
        vd = valid[valid['vote_split'] == vs]
        print(f"  {vs:<15} avg={vd['ret_24h'].mean():+.3f}%  win={(vd['ret_24h'] > 0).mean()*100:.0f}%  n={len(vd)}")

    cross_tabulate(results)

    print("\n" + "="*80)
    print("PROMPT B: SESSION TRANSMISSION CHAIN VALIDATION")
    print("="*80)
    transmission_chain(results)
    direction_persistence_to_reopen(results)
    statistical_tests(results)

    out_path = os.path.join(os.path.dirname(__file__), 'backtest_boe_rate_results.json')
    results.to_json(out_path, orient='records', indent=2)
    print(f"\nResults saved to {out_path}")


if __name__ == '__main__':
    main()

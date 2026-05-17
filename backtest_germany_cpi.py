"""
Prompt A + B: Backtest Germany CPI Session Transmission Chain
=============================================================
ETH/USDT 15m data from jimi/eth_15m_merged.csv
Event: Germany CPI YoY (Destatis, ~10th-20th each month, 07:00 UTC)
Transmission: Europe Morning → Eurozone Open (next day) → US Morning

Thesis:
  Hot German CPI → EUR/USD surge → DXY lower → ETH fast intraday rally
  European desks lock positions ahead of Eurozone CPI Flash (1-2 days later)
  Hawkish ECB pressure → reduces loose global liquidity expectations
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from scipy import stats
import json
import warnings
warnings.filterwarnings('ignore')

# ═══════════════════════════════════════════════════════════════
# GERMANY CPI RELEASE DATES + ACTUAL VALUES (07:00 UTC = 08:00 CET)
# Destatis releases CPI around 10th-20th each month
# Format: {date: {'cpi_yoy': float, 'consensus': float, 'prior': float}}
# ═══════════════════════════════════════════════════════════════

GERMANY_CPI_RELEASES = {
    # 2018
    '2018-01-11': {'cpi_yoy': 1.6, 'consensus': 1.5, 'prior': 1.8},
    '2018-02-14': {'cpi_yoy': 1.4, 'consensus': 1.5, 'prior': 1.6},
    '2018-03-14': {'cpi_yoy': 1.4, 'consensus': 1.4, 'prior': 1.4},
    '2018-04-12': {'cpi_yoy': 1.6, 'consensus': 1.6, 'prior': 1.4},
    '2018-05-11': {'cpi_yoy': 1.6, 'consensus': 1.5, 'prior': 1.6},
    '2018-06-13': {'cpi_yoy': 2.2, 'consensus': 2.1, 'prior': 1.6},
    '2018-07-12': {'cpi_yoy': 2.1, 'consensus': 2.1, 'prior': 2.2},
    '2018-08-09': {'cpi_yoy': 2.0, 'consensus': 2.0, 'prior': 2.1},
    '2018-09-12': {'cpi_yoy': 1.9, 'consensus': 1.9, 'prior': 2.0},
    '2018-10-11': {'cpi_yoy': 2.3, 'consensus': 2.3, 'prior': 1.9},
    '2018-11-13': {'cpi_yoy': 2.5, 'consensus': 2.4, 'prior': 2.3},
    '2018-12-13': {'cpi_yoy': 2.3, 'consensus': 2.3, 'prior': 2.5},
    # 2019
    '2019-01-11': {'cpi_yoy': 1.7, 'consensus': 1.7, 'prior': 2.3},
    '2019-02-14': {'cpi_yoy': 1.4, 'consensus': 1.5, 'prior': 1.7},
    '2019-03-13': {'cpi_yoy': 1.5, 'consensus': 1.5, 'prior': 1.4},
    '2019-04-11': {'cpi_yoy': 1.3, 'consensus': 1.3, 'prior': 1.5},
    '2019-05-10': {'cpi_yoy': 1.4, 'consensus': 1.4, 'prior': 1.3},
    '2019-06-13': {'cpi_yoy': 1.4, 'consensus': 1.4, 'prior': 1.4},
    '2019-07-11': {'cpi_yoy': 1.5, 'consensus': 1.5, 'prior': 1.4},
    '2019-08-08': {'cpi_yoy': 1.7, 'consensus': 1.5, 'prior': 1.5},
    '2019-09-11': {'cpi_yoy': 1.4, 'consensus': 1.4, 'prior': 1.7},
    '2019-10-10': {'cpi_yoy': 1.2, 'consensus': 1.3, 'prior': 1.4},
    '2019-11-13': {'cpi_yoy': 1.1, 'consensus': 1.1, 'prior': 1.2},
    '2019-12-12': {'cpi_yoy': 1.2, 'consensus': 1.1, 'prior': 1.1},
    # 2020
    '2020-01-08': {'cpi_yoy': 1.5, 'consensus': 1.4, 'prior': 1.2},
    '2020-02-13': {'cpi_yoy': 1.7, 'consensus': 1.6, 'prior': 1.5},
    '2020-03-12': {'cpi_yoy': 1.7, 'consensus': 1.7, 'prior': 1.7},
    '2020-04-16': {'cpi_yoy': 0.8, 'consensus': 0.8, 'prior': 1.7},
    '2020-05-13': {'cpi_yoy': 0.5, 'consensus': 0.6, 'prior': 0.8},
    '2020-06-11': {'cpi_yoy': 0.6, 'consensus': 0.6, 'prior': 0.5},
    '2020-07-09': {'cpi_yoy': 0.9, 'consensus': 0.8, 'prior': 0.6},
    '2020-08-11': {'cpi_yoy': 0.0, 'consensus': 0.1, 'prior': 0.9},
    '2020-09-10': {'cpi_yoy': -0.1, 'consensus': 0.0, 'prior': 0.0},
    '2020-10-13': {'cpi_yoy': -0.2, 'consensus': -0.1, 'prior': -0.1},
    '2020-11-12': {'cpi_yoy': -0.3, 'consensus': -0.2, 'prior': -0.2},
    '2020-12-11': {'cpi_yoy': -0.3, 'consensus': -0.3, 'prior': -0.3},
    # 2021
    '2021-01-07': {'cpi_yoy': -0.3, 'consensus': -0.3, 'prior': -0.3},
    '2021-02-11': {'cpi_yoy': 1.0, 'consensus': 0.8, 'prior': -0.3},
    '2021-03-11': {'cpi_yoy': 1.3, 'consensus': 1.2, 'prior': 1.0},
    '2021-04-12': {'cpi_yoy': 1.7, 'consensus': 1.7, 'prior': 1.3},
    '2021-05-12': {'cpi_yoy': 2.0, 'consensus': 1.9, 'prior': 1.7},
    '2021-06-10': {'cpi_yoy': 2.5, 'consensus': 2.4, 'prior': 2.0},
    '2021-07-08': {'cpi_yoy': 2.3, 'consensus': 2.3, 'prior': 2.5},
    '2021-08-11': {'cpi_yoy': 3.8, 'consensus': 3.3, 'prior': 2.3},
    '2021-09-09': {'cpi_yoy': 3.9, 'consensus': 3.9, 'prior': 3.8},
    '2021-10-08': {'cpi_yoy': 4.1, 'consensus': 4.2, 'prior': 3.9},
    '2021-11-10': {'cpi_yoy': 4.5, 'consensus': 4.4, 'prior': 4.1},
    '2021-12-10': {'cpi_yoy': 5.2, 'consensus': 5.0, 'prior': 4.5},
    # 2022
    '2022-01-06': {'cpi_yoy': 5.3, 'consensus': 5.1, 'prior': 5.2},
    '2022-02-11': {'cpi_yoy': 5.1, 'consensus': 4.9, 'prior': 5.3},
    '2022-03-11': {'cpi_yoy': 5.5, 'consensus': 5.3, 'prior': 5.1},
    '2022-04-12': {'cpi_yoy': 7.4, 'consensus': 6.7, 'prior': 5.5},
    '2022-05-11': {'cpi_yoy': 7.8, 'consensus': 7.6, 'prior': 7.4},
    '2022-06-10': {'cpi_yoy': 8.7, 'consensus': 8.0, 'prior': 7.8},
    '2022-07-08': {'cpi_yoy': 8.5, 'consensus': 8.1, 'prior': 8.7},
    '2022-08-10': {'cpi_yoy': 8.5, 'consensus': 8.5, 'prior': 8.5},
    '2022-09-08': {'cpi_yoy': 8.8, 'consensus': 8.8, 'prior': 8.5},
    '2022-10-13': {'cpi_yoy': 10.4, 'consensus': 10.1, 'prior': 8.8},
    '2022-11-10': {'cpi_yoy': 11.6, 'consensus': 10.9, 'prior': 10.4},
    '2022-12-13': {'cpi_yoy': 11.3, 'consensus': 11.3, 'prior': 11.6},
    # 2023
    '2023-01-05': {'cpi_yoy': 9.6, 'consensus': 9.1, 'prior': 11.3},
    '2023-02-10': {'cpi_yoy': 9.2, 'consensus': 9.0, 'prior': 9.6},
    '2023-03-10': {'cpi_yoy': 9.3, 'consensus': 9.0, 'prior': 9.2},
    '2023-04-13': {'cpi_yoy': 7.6, 'consensus': 7.5, 'prior': 9.3},
    '2023-05-10': {'cpi_yoy': 7.2, 'consensus': 7.3, 'prior': 7.6},
    '2023-06-08': {'cpi_yoy': 6.1, 'consensus': 6.3, 'prior': 7.2},
    '2023-07-06': {'cpi_yoy': 6.4, 'consensus': 6.3, 'prior': 6.1},
    '2023-08-09': {'cpi_yoy': 6.2, 'consensus': 6.2, 'prior': 6.4},
    '2023-09-07': {'cpi_yoy': 6.1, 'consensus': 6.0, 'prior': 6.2},
    '2023-10-12': {'cpi_yoy': 4.5, 'consensus': 4.5, 'prior': 6.1},
    '2023-11-10': {'cpi_yoy': 3.0, 'consensus': 3.0, 'prior': 4.5},
    '2023-12-07': {'cpi_yoy': 2.3, 'consensus': 2.5, 'prior': 3.0},
    # 2024
    '2024-01-04': {'cpi_yoy': 2.9, 'consensus': 3.0, 'prior': 2.3},
    '2024-02-08': {'cpi_yoy': 2.9, 'consensus': 2.8, 'prior': 2.9},
    '2024-03-07': {'cpi_yoy': 2.5, 'consensus': 2.6, 'prior': 2.9},
    '2024-04-04': {'cpi_yoy': 2.2, 'consensus': 2.3, 'prior': 2.5},
    '2024-05-08': {'cpi_yoy': 2.2, 'consensus': 2.3, 'prior': 2.2},
    '2024-06-12': {'cpi_yoy': 2.4, 'consensus': 2.4, 'prior': 2.2},
    '2024-07-10': {'cpi_yoy': 2.2, 'consensus': 2.3, 'prior': 2.4},
    '2024-08-08': {'cpi_yoy': 2.0, 'consensus': 2.1, 'prior': 2.2},
    '2024-09-10': {'cpi_yoy': 1.9, 'consensus': 1.9, 'prior': 2.0},
    '2024-10-10': {'cpi_yoy': 1.6, 'consensus': 1.7, 'prior': 1.9},
    '2024-11-08': {'cpi_yoy': 2.0, 'consensus': 1.9, 'prior': 1.6},
    '2024-12-11': {'cpi_yoy': 2.2, 'consensus': 2.2, 'prior': 2.0},
    # 2025
    '2025-01-08': {'cpi_yoy': 2.6, 'consensus': 2.4, 'prior': 2.2},
    '2025-02-12': {'cpi_yoy': 2.8, 'consensus': 2.8, 'prior': 2.6},
    '2025-03-12': {'cpi_yoy': 2.6, 'consensus': 2.7, 'prior': 2.8},
    '2025-04-08': {'cpi_yoy': 2.3, 'consensus': 2.3, 'prior': 2.6},
    '2025-05-08': {'cpi_yoy': 2.2, 'consensus': 2.1, 'prior': 2.3},
    '2025-06-11': {'cpi_yoy': 2.1, 'consensus': 2.1, 'prior': 2.2},
    '2025-07-09': {'cpi_yoy': 2.0, 'consensus': 2.0, 'prior': 2.1},
    '2025-08-07': {'cpi_yoy': 2.0, 'consensus': 2.0, 'prior': 2.0},
    '2025-09-10': {'cpi_yoy': 2.0, 'consensus': 2.0, 'prior': 2.0},
    '2025-10-09': {'cpi_yoy': 2.1, 'consensus': 2.0, 'prior': 2.0},
    '2025-11-12': {'cpi_yoy': 2.2, 'consensus': 2.1, 'prior': 2.1},
    '2025-12-11': {'cpi_yoy': 2.2, 'consensus': 2.2, 'prior': 2.2},
    # 2026
    '2026-01-07': {'cpi_yoy': 2.3, 'consensus': 2.3, 'prior': 2.2},
    '2026-02-11': {'cpi_yoy': 2.4, 'consensus': 2.3, 'prior': 2.3},
    '2026-03-11': {'cpi_yoy': 2.3, 'consensus': 2.3, 'prior': 2.4},
    '2026-04-09': {'cpi_yoy': 2.2, 'consensus': 2.2, 'prior': 2.3},
    '2026-05-07': {'cpi_yoy': 2.1, 'consensus': 2.1, 'prior': 2.2},
}

# ═══════════════════════════════════════════════════════════════
# SESSION DEFINITIONS (UTC)
# ═══════════════════════════════════════════════════════════════

SESSIONS = {
    'Pre-Asia':          {'start': 21, 'end': 0,  'phases': ['Post-NY Close / Globex']},
    'Asia':              {'start': 0,  'end': 8,  'phases': [
        'Sydney Open', 'Tokyo Open', 'Asia Mid', 'Asia Afternoon',
        'Tokyo Close', 'Pre-London'
    ]},
    'Europe':            {'start': 7,  'end': 12, 'phases': [
        'Frankfurt Open', 'London Open', 'London Morning', 'London Midday'
    ]},
    'Overlap (EU–US)':   {'start': 12, 'end': 16, 'phases': [
        'NY Pre-Open', 'NY Open', 'London–NY Overlap'
    ]},
    'New York':          {'start': 13, 'end': 21, 'phases': [
        'NY AM', 'NY Lunch', 'NY PM'
    ]},
}

# Detailed phase boundaries (UTC hours)
PHASES = {
    'Post-NY Close / Globex': (21, 0),
    'Sydney Open':            (0, 1),
    'Tokyo Open':             (1, 3),
    'Asia Mid':               (3, 5),
    'Asia Afternoon':         (5, 6),
    'Tokyo Close':            (6, 7),
    'Pre-London':             (7, 8),
    'Frankfurt Open':         (7, 8),
    'London Open':            (8, 9),
    'London Morning':         (9, 11),
    'London Midday':          (11, 12),
    'NY Pre-Open':            (12, 13),
    'NY Open':                (13, 14),
    'London–NY Overlap':      (14, 16),
    'NY AM':                  (14, 16),
    'NY Lunch':               (16, 17),
    'NY PM':                  (17, 21),
}


def load_eth_data(filepath):
    """Load ETH 15m CSV data."""
    df = pd.read_csv(filepath)
    df['Open time'] = pd.to_datetime(df['Open time'])
    df = df.set_index('Open time')
    df['Close'] = df['Close'].astype(float)
    df['Open'] = df['Open'].astype(float)
    df['High'] = df['High'].astype(float)
    df['Low'] = df['Low'].astype(float)
    df['Volume'] = df['Volume'].astype(float)
    return df


def classify_signal(actual, consensus, prior):
    """Classify CPI release as BEAT / INLINE / MISS relative to consensus."""
    diff = actual - consensus
    surprise = diff / abs(consensus) if consensus != 0 else diff
    
    # Also consider trend
    trend = actual - prior
    
    if surprise > 0.15:  # >15% above consensus
        return 'STRONG_BEAT', surprise, trend
    elif surprise > 0.03:
        return 'BEAT', surprise, trend
    elif surprise < -0.15:
        return 'BIG_MISS', surprise, trend
    elif surprise < -0.03:
        return 'MISS', surprise, trend
    else:
        return 'INLINE', surprise, trend


def classify_cpi_level(actual):
    """Classify CPI by absolute level."""
    if actual >= 5.0:
        return 'HYPERINFLATION'
    elif actual >= 3.0:
        return 'HOT'
    elif actual >= 2.0:
        return 'WARM'
    elif actual >= 1.0:
        return 'MILD'
    elif actual >= 0.0:
        return 'COOL'
    else:
        return 'DEFLATION'


def classify_cpi_direction(actual, prior):
    """Classify CPI direction: rising, stable, falling."""
    change = actual - prior
    if change > 0.2:
        return 'RISING'
    elif change < -0.2:
        return 'FALLING'
    else:
        return 'STABLE'


def get_wyckoff_phase_stub(date_str):
    """
    Stub: approximate Wyckoff phase from price action around date.
    In the real framework this comes from M21. Here we use a simplified version.
    """
    # We'll compute this properly using the ETH data
    return 'UNKNOWN'  # placeholder, computed later


def get_vol_regime_stub(date_str):
    """
    Stub: approximate vol regime.
    In the real framework this comes from M9.
    """
    return 'UNKNOWN'  # placeholder, computed later


def compute_session_returns(df, release_date, release_utc_hour=7):
    """
    Compute ETH returns for each session phase on release day and next day.
    
    Germany CPI releases at 07:00 UTC (08:00 CET).
    We measure from release to each session boundary.
    """
    release_dt = pd.Timestamp(f"{release_date} {release_utc_hour:02d}:00:00")
    next_day = release_dt + timedelta(days=1)
    day_after = release_dt + timedelta(days=2)
    
    # Get price at release
    # Find the 15m bar at or just after release
    release_bar = df.index[df.index >= release_dt]
    if len(release_bar) == 0:
        return None
    release_bar = release_bar[0]
    price_at_release = df.loc[release_bar, 'Close']
    
    results = {}
    
    # For each session phase, compute return from release price
    for phase_name, (start_h, end_h) in PHASES.items():
        # Phase occurs on release day or next day
        if start_h >= release_utc_hour or phase_name in ['Post-NY Close / Globex']:
            # Same day
            phase_start_dt = pd.Timestamp(f"{release_date} {start_h:02d}:00:00")
        else:
            # Next day for phases before release time
            phase_start_dt = pd.Timestamp(f"{(pd.Timestamp(release_date) + timedelta(days=1)).strftime('%Y-%m-%d')} {start_h:02d}:00:00")
        
        # Get the bar at phase start
        phase_bars = df.index[df.index >= phase_start_dt]
        if len(phase_bars) == 0:
            results[phase_name] = None
            continue
        
        phase_bar = phase_bars[0]
        price_at_phase = df.loc[phase_bar, 'Close']
        ret = (price_at_phase - price_at_release) / price_at_release * 100
        results[phase_name] = ret
    
    # 24h aggregate: release + 24h
    end_24h = release_dt + timedelta(hours=24)
    end_bars = df.index[df.index >= end_24h]
    if len(end_bars) > 0:
        price_24h = df.loc[end_bars[0], 'Close']
        results['24h_return'] = (price_24h - price_at_release) / price_at_release * 100
    else:
        results['24h_return'] = None
    
    return results


def compute_wyckoff_phase(df, date_str, lookback_days=30):
    """
    Simplified Wyckoff phase detection using price structure.
    """
    dt = pd.Timestamp(date_str)
    start = dt - timedelta(days=lookback_days)
    
    window = df[(df.index >= start) & (df.index < dt)]
    if len(window) < 100:
        return 'RANGE'
    
    closes = window['Close'].values.astype(float)
    highs = window['High'].values.astype(float)
    lows = window['Low'].values.astype(float)
    
    # Simple trend detection
    sma_short = np.mean(closes[-48:])   # ~3 days
    sma_long = np.mean(closes[-192:])   # ~12 days
    
    range_high = np.percentile(highs, 90)
    range_low = np.percentile(lows, 10)
    range_mid = (range_high + range_low) / 2
    current = closes[-1]
    
    # Range detection
    range_width = (range_high - range_low) / range_mid
    if range_width < 0.05:  # <5% range
        return 'RANGE'
    
    if sma_short > sma_long * 1.02:
        if current > range_mid:
            return 'MARKUP'
        else:
            return 'DISTRIBUTION'
    elif sma_short < sma_long * 0.98:
        if current < range_mid:
            return 'MARKDOWN'
        else:
            return 'ACCUMULATION'
    else:
        return 'RANGE'


def compute_vol_regime(df, date_str, lookback_days=7):
    """
    Simplified vol regime detection.
    """
    dt = pd.Timestamp(date_str)
    start = dt - timedelta(days=lookback_days)
    
    window = df[(df.index >= start) & (df.index < dt)]
    if len(window) < 20:
        return 'NEUTRAL'
    
    closes = window['Close'].values.astype(float)
    highs = window['High'].values.astype(float)
    lows = window['Low'].values.astype(float)
    
    # ATR proxy
    ranges = (highs - lows) / closes
    atr_pctile = np.percentile(ranges, 50)
    recent_range = ranges[-16:]  # last 4 hours
    
    # Bollinger width proxy
    sma = np.mean(closes[-48:])
    std = np.std(closes[-48:])
    bb_width = (2 * std) / sma if sma > 0 else 0
    
    # Whipsaw: count direction changes
    diffs = np.diff(closes[-48:])
    direction_changes = np.sum(np.diff(np.sign(diffs)) != 0)
    whipsaw_rate = direction_changes / len(diffs) if len(diffs) > 0 else 0
    
    if bb_width < 0.015 and np.mean(recent_range) < 0.005:
        return 'COMPRESSING'
    elif whipsaw_rate > 0.6:
        return 'CHOP'
    elif np.std(diffs) > np.mean(np.abs(diffs)) * 1.5:
        return 'TRENDING'
    elif atr_pctile > np.percentile(ranges, 80):
        return 'CRISIS'
    else:
        return 'NEUTRAL'


def run_backtest():
    """Main backtest function — Prompt A."""
    print("=" * 80)
    print("PROMPT A: GERMANY CPI BACKTEST (2018-2026)")
    print("ETH/USDT 15m data | Release: 07:00 UTC (08:00 CET)")
    print("=" * 80)
    
    df = load_eth_data('eth_15m_merged.csv')
    print(f"\nLoaded {len(df)} bars: {df.index[0]} → {df.index[-1]}")
    
    # Collect all results
    all_results = []
    
    for date_str, data in sorted(GERMANY_CPI_RELEASES.items()):
        dt = pd.Timestamp(date_str)
        
        # Check if we have data for this date
        if dt < df.index[0] or dt > df.index[-1] - timedelta(days=2):
            continue
        
        returns = compute_session_returns(df, date_str)
        if returns is None or returns.get('24h_return') is None:
            continue
        
        # Classify signal
        signal, surprise, trend = classify_signal(
            data['cpi_yoy'], data['consensus'], data['prior'])
        
        # Classify CPI level
        cpi_level = classify_cpi_level(data['cpi_yoy'])
        
        # Classify CPI direction
        cpi_dir = classify_cpi_direction(data['cpi_yoy'], data['prior'])
        
        # Get Wyckoff phase
        wyckoff = compute_wyckoff_phase(df, date_str)
        
        # Get vol regime
        vol = compute_vol_regime(df, date_str)
        
        result = {
            'date': date_str,
            'cpi_yoy': data['cpi_yoy'],
            'consensus': data['consensus'],
            'prior': data['prior'],
            'signal': signal,
            'surprise': surprise,
            'trend': trend,
            'cpi_level': cpi_level,
            'cpi_direction': cpi_dir,
            'wyckoff': wyckoff,
            'vol': vol,
            **returns
        }
        all_results.append(result)
    
    df_results = pd.DataFrame(all_results)
    print(f"\nAnalyzed {len(df_results)} Germany CPI releases")
    
    # ═══════════════════════════════════════════════════════════════
    # SESSION-BY-SESSION RETURNS TABLE
    # ═══════════════════════════════════════════════════════════════
    print("\n" + "=" * 80)
    print("SESSION-BY-SESSION AVERAGE RETURNS (%)")
    print("=" * 80)
    
    session_phases = [
        'Post-NY Close / Globex', 'Sydney Open', 'Tokyo Open', 'Asia Mid',
        'Asia Afternoon', 'Tokyo Close', 'Pre-London',
        'Frankfurt Open', 'London Open', 'London Morning', 'London Midday',
        'NY Pre-Open', 'NY Open', 'London–NY Overlap',
        'NY AM', 'NY Lunch', 'NY PM'
    ]
    
    print(f"\n{'Phase':<28} {'Avg%':>8} {'Win%':>8} {'N':>6} {'Sig':>6}")
    print("-" * 60)
    for phase in session_phases:
        valid = df_results[phase].dropna()
        if len(valid) == 0:
            continue
        avg = valid.mean()
        win = (valid > 0).mean() * 100
        n = len(valid)
        sig = "***" if abs(avg) > 0.5 and n >= 5 else "**" if abs(avg) > 0.3 else ""
        print(f"{phase:<28} {avg:>8.3f} {win:>7.1f}% {n:>5} {sig:>6}")
    
    # 24h aggregate
    valid_24h = df_results['24h_return'].dropna()
    avg_24h = valid_24h.mean()
    win_24h = (valid_24h > 0).mean() * 100
    print(f"\n{'24h AGGREGATE':<28} {avg_24h:>8.3f} {win_24h:>7.1f}% {len(valid_24h):>5}")
    
    # ═══════════════════════════════════════════════════════════════
    # CROSS-TABULATION: wyckoff × vol × signal
    # ═══════════════════════════════════════════════════════════════
    print("\n" + "=" * 80)
    print("CROSS-TABULATION: Wyckoff × Vol × Signal → 24h Return")
    print("=" * 80)
    
    combos = df_results.groupby(['wyckoff', 'vol', 'signal']).agg(
        avg_24h=('24h_return', 'mean'),
        win_rate=('24h_return', lambda x: (x > 0).mean() * 100),
        count=('24h_return', 'count'),
        avg_surprise=('surprise', 'mean')
    ).reset_index()
    
    combos = combos[combos['count'] >= 3].sort_values('avg_24h', key=abs, ascending=False)
    
    print(f"\n{'Wyckoff':<14} {'Vol':<12} {'Signal':<14} {'Avg24h%':>10} {'Win%':>8} {'N':>5}")
    print("-" * 70)
    for _, row in combos.iterrows():
        edge = "✅" if abs(row['avg_24h']) >= 0.5 else "  "
        print(f"{row['wyckoff']:<14} {row['vol']:<12} {row['signal']:<14} "
              f"{row['avg_24h']:>9.3f}% {row['win_rate']:>7.1f}% {int(row['count']):>4} {edge}")
    
    # ═══════════════════════════════════════════════════════════════
    # CPI LEVEL × DIRECTION × SIGNAL
    # ═══════════════════════════════════════════════════════════════
    print("\n" + "=" * 80)
    print("CPI LEVEL × DIRECTION × SIGNAL → 24h Return")
    print("=" * 80)
    
    combos2 = df_results.groupby(['cpi_level', 'cpi_direction', 'signal']).agg(
        avg_24h=('24h_return', 'mean'),
        win_rate=('24h_return', lambda x: (x > 0).mean() * 100),
        count=('24h_return', 'count')
    ).reset_index()
    
    combos2 = combos2[combos2['count'] >= 2].sort_values('avg_24h', key=abs, ascending=False)
    
    print(f"\n{'CPI Level':<16} {'Direction':<12} {'Signal':<14} {'Avg24h%':>10} {'Win%':>8} {'N':>5}")
    print("-" * 70)
    for _, row in combos2.iterrows():
        edge = "✅" if abs(row['avg_24h']) >= 0.5 else "  "
        print(f"{row['cpi_level']:<16} {row['cpi_direction']:<12} {row['signal']:<14} "
              f"{row['avg_24h']:>9.3f}% {row['win_rate']:>7.1f}% {int(row['count']):>4} {edge}")
    
    return df_results


def run_transmission_chain(df_results):
    """Prompt B: Validate session-by-session transmission chain."""
    print("\n\n" + "=" * 80)
    print("PROMPT B: SESSION TRANSMISSION CHAIN VALIDATION")
    print("=" * 80)
    
    session_phases = [
        'Post-NY Close / Globex', 'Sydney Open', 'Tokyo Open', 'Asia Mid',
        'Asia Afternoon', 'Tokyo Close', 'Pre-London',
        'Frankfurt Open', 'London Open', 'London Morning', 'London Midday',
        'NY Pre-Open', 'NY Open', 'London–NY Overlap',
        'NY AM', 'NY Lunch', 'NY PM'
    ]
    
    # Direction persistence: % same direction between consecutive sessions
    print("\nDIRECTION PERSISTENCE BETWEEN CONSECUTIVE SESSIONS")
    print("(Release day only — % same direction as previous phase)")
    print("-" * 70)
    
    valid_phases = [p for p in session_phases if p in df_results.columns and df_results[p].notna().sum() >= 5]
    
    transitions = []
    for i in range(len(valid_phases) - 1):
        p1, p2 = valid_phases[i], valid_phases[i+1]
        mask = df_results[p1].notna() & df_results[p2].notna()
        subset = df_results[mask]
        
        if len(subset) < 5:
            continue
        
        # Same direction = both positive or both negative
        same_dir = ((subset[p1] > 0) & (subset[p2] > 0)) | \
                   ((subset[p1] < 0) & (subset[p2] < 0))
        pct_same = same_dir.mean() * 100
        
        # Also track magnitude correlation
        corr = subset[p1].corr(subset[p2])
        
        edge_label = ""
        if pct_same > 65:
            edge_label = "✅ REAL EDGE"
        elif pct_same >= 55:
            edge_label = "⚠️  MARGINAL"
        else:
            edge_label = "❌ NO CHAIN"
        
        transitions.append({
            'from': p1,
            'to': p2,
            'pct_same': pct_same,
            'corr': corr,
            'n': len(subset),
            'edge': edge_label
        })
        
        print(f"  {p1:<28} → {p2:<24} {pct_same:>5.1f}% same  (r={corr:>5.2f}, n={len(subset)}) {edge_label}")
    
    # Chain from release to 24h
    print("\n\nCHAIN FROM RELEASE TO END-OF-DAY (same direction as first significant move)")
    print("-" * 70)
    
    # Get first phase with data after release
    first_phase = valid_phases[0] if valid_phases else None
    if first_phase:
        for phase in valid_phases[1:]:
            mask = df_results[first_phase].notna() & df_results[phase].notna()
            subset = df_results[mask]
            if len(subset) < 3:
                continue
            
            same_dir = ((subset[first_phase] > 0) & (subset[phase] > 0)) | \
                       ((subset[first_phase] < 0) & (subset[phase] < 0))
            pct_same = same_dir.mean() * 100
            
            edge_label = ""
            if pct_same > 65:
                edge_label = "✅"
            elif pct_same >= 55:
                edge_label = "⚠️"
            else:
                edge_label = "❌"
            
            print(f"  {first_phase:<28} → {phase:<24} {pct_same:>5.1f}% persist {edge_label} (n={len(subset)})")
    
    # ═══════════════════════════════════════════════════════════════
    # STATISTICAL TESTS
    # ═══════════════════════════════════════════════════════════════
    print("\n\n" + "=" * 80)
    print("STATISTICAL SIGNIFICANCE TESTS")
    print("=" * 80)
    
    returns_24h = df_results['24h_return'].dropna()
    
    # One-sample t-test: H0: mean = 0
    t_stat, p_value = stats.ttest_1samp(returns_24h, 0)
    print(f"\n1. One-sample t-test (H0: mean 24h return = 0)")
    print(f"   Mean: {returns_24h.mean():.4f}%")
    print(f"   t = {t_stat:.4f}, p = {p_value:.4f}")
    print(f"   {'✅ SIGNIFICANT (p<0.05)' if p_value < 0.05 else '❌ NOT significant (p≥0.05)'}")
    
    # Two-sample t-test: BEAT vs MISS
    beat_mask = df_results['signal'].isin(['BEAT', 'STRONG_BEAT'])
    miss_mask = df_results['signal'].isin(['MISS', 'BIG_MISS'])
    
    beat_returns = df_results.loc[beat_mask, '24h_return'].dropna()
    miss_returns = df_results.loc[miss_mask, '24h_return'].dropna()
    
    if len(beat_returns) >= 3 and len(miss_returns) >= 3:
        t_stat2, p_value2 = stats.ttest_ind(beat_returns, miss_returns)
        print(f"\n2. Two-sample t-test (BEAT vs MISS)")
        print(f"   BEAT mean: {beat_returns.mean():.4f}% (n={len(beat_returns)})")
        print(f"   MISS mean: {miss_returns.mean():.4f}% (n={len(miss_returns)})")
        print(f"   t = {t_stat2:.4f}, p = {p_value2:.4f}")
        print(f"   {'✅ SIGNIFICANT (p<0.05)' if p_value2 < 0.05 else '❌ NOT significant (p≥0.05)'}")
    
    # Hot vs Cold CPI
    hot_mask = df_results['cpi_level'].isin(['HOT', 'HYPERINFLATION'])
    cold_mask = df_results['cpi_level'].isin(['COOL', 'DEFLATION', 'MILD'])
    
    hot_returns = df_results.loc[hot_mask, '24h_return'].dropna()
    cold_returns = df_results.loc[cold_mask, '24h_return'].dropna()
    
    if len(hot_returns) >= 3 and len(cold_returns) >= 3:
        t_stat3, p_value3 = stats.ttest_ind(hot_returns, cold_returns)
        print(f"\n3. Two-sample t-test (HOT CPI ≥3% vs COOL CPI <2%)")
        print(f"   HOT mean: {hot_returns.mean():.4f}% (n={len(hot_returns)})")
        print(f"   COOL mean: {cold_returns.mean():.4f}% (n={len(cold_returns)})")
        print(f"   t = {t_stat3:.4f}, p = {p_value3:.4f}")
        print(f"   {'✅ SIGNIFICANT (p<0.05)' if p_value3 < 0.05 else '❌ NOT significant (p≥0.05)'}")
    
    # Rising vs Falling CPI
    rising_mask = df_results['cpi_direction'] == 'RISING'
    falling_mask = df_results['cpi_direction'] == 'FALLING'
    
    rising_returns = df_results.loc[rising_mask, '24h_return'].dropna()
    falling_returns = df_results.loc[falling_mask, '24h_return'].dropna()
    
    if len(rising_returns) >= 3 and len(falling_returns) >= 3:
        t_stat4, p_value4 = stats.ttest_ind(rising_returns, falling_returns)
        print(f"\n4. Two-sample t-test (RISING vs FALLING CPI)")
        print(f"   RISING mean: {rising_returns.mean():.4f}% (n={len(rising_returns)})")
        print(f"   FALLING mean: {falling_returns.mean():.4f}% (n={len(falling_returns)})")
        print(f"   t = {t_stat4:.4f}, p = {p_value4:.4f}")
        print(f"   {'✅ SIGNIFICANT (p<0.05)' if p_value4 < 0.05 else '❌ NOT significant (p≥0.05)'}")
    
    return transitions


def main():
    df_results = run_backtest()
    transitions = run_transmission_chain(df_results)
    
    # Save results
    df_results.to_json('backtest_germany_cpi_results.json', orient='records', indent=2)
    print(f"\n\nResults saved to backtest_germany_cpi_results.json")
    
    # Summary
    print("\n\n" + "=" * 80)
    print("SUMMARY: EDGE IDENTIFICATION")
    print("=" * 80)
    
    # Find combos with edge (n≥3, |avg|≥0.5%)
    edge_combos = df_results.groupby(['wyckoff', 'vol', 'signal']).agg(
        avg_24h=('24h_return', 'mean'),
        win_rate=('24h_return', lambda x: (x > 0).mean() * 100),
        count=('24h_return', 'count')
    ).reset_index()
    edge_combos = edge_combos[(edge_combos['count'] >= 3) & (edge_combos['avg_24h'].abs() >= 0.5)]
    edge_combos = edge_combos.sort_values('avg_24h', key=abs, ascending=False)
    
    if len(edge_combos) > 0:
        print("\nCombos with edge (n≥3, |avg24h|≥0.5%):")
        for _, row in edge_combos.iterrows():
            direction = "LONG" if row['avg_24h'] > 0 else "SHORT"
            print(f"  {row['wyckoff']} + {row['vol']} + {row['signal']}: "
                  f"{row['avg_24h']:+.3f}% avg, {row['win_rate']:.0f}% win, n={int(row['count'])} → {direction} bias")
    
    # Find strong transitions
    print("\nStrong transmission links (>65% same direction):")
    for t in transitions:
        if t['pct_same'] > 65:
            print(f"  {t['from']} → {t['to']}: {t['pct_same']:.1f}% persist (r={t['corr']:.2f})")
    
    return df_results


if __name__ == '__main__':
    main()

"""
Prompt A + B: Backtest Eurozone CPI Flash Session Transmission Chain
====================================================================
ETH/USDT 15m data from jimi/eth_15m_merged.csv
Event: Eurozone CPI Flash (HICP YoY, Eurostat, ~end of month, 09:00 UTC)
Transmission: Europe Morning (18:00 MYT) → US Inflation Desk → ECB Meeting

Thesis:
  Cool EZ CPI → green light for ECB rate cuts → EUR drops → global yield easing → crypto bids
  Hot EZ CPI → hawkish ECB → EUR surges → DXY lower → ETH mechanical bounce
  US macro funds use this as global inflation comparison ahead of US CPI
  Early directional proxy for Core PCE (4 weeks later)
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

# ═══════════════════════════════════════════════════════════════
# EUROZONE CPI FLASH RELEASE DATES + ACTUAL VALUES (09:00 UTC)
# Eurostat releases HICP flash estimate ~last day of month
# Format: {date: {'hicp_yoy': float, 'core_yoy': float, 'consensus': float, 'prior': float}}
# ═══════════════════════════════════════════════════════════════

EZ_CPI_RELEASES = {
    # 2018
    '2018-01-31': {'hicp_yoy': 1.3, 'core_yoy': 1.0, 'consensus': 1.4, 'prior': 1.4},
    '2018-02-28': {'hicp_yoy': 1.2, 'core_yoy': 1.0, 'consensus': 1.2, 'prior': 1.3},
    '2018-03-29': {'hicp_yoy': 1.4, 'core_yoy': 1.0, 'consensus': 1.4, 'prior': 1.2},
    '2018-04-30': {'hicp_yoy': 1.2, 'core_yoy': 0.7, 'consensus': 1.3, 'prior': 1.4},
    '2018-05-31': {'hicp_yoy': 1.9, 'core_yoy': 1.1, 'consensus': 1.6, 'prior': 1.2},
    '2018-06-29': {'hicp_yoy': 2.0, 'core_yoy': 1.0, 'consensus': 2.0, 'prior': 1.9},
    '2018-07-31': {'hicp_yoy': 2.1, 'core_yoy': 1.1, 'consensus': 2.1, 'prior': 2.0},
    '2018-08-31': {'hicp_yoy': 2.0, 'core_yoy': 1.0, 'consensus': 2.1, 'prior': 2.1},
    '2018-09-28': {'hicp_yoy': 2.1, 'core_yoy': 0.9, 'consensus': 2.1, 'prior': 2.0},
    '2018-10-31': {'hicp_yoy': 2.2, 'core_yoy': 1.1, 'consensus': 2.2, 'prior': 2.1},
    '2018-11-30': {'hicp_yoy': 2.0, 'core_yoy': 1.0, 'consensus': 2.0, 'prior': 2.2},
    '2018-12-31': {'hicp_yoy': 1.6, 'core_yoy': 1.0, 'consensus': 1.8, 'prior': 2.0},
    # 2019
    '2019-01-31': {'hicp_yoy': 1.4, 'core_yoy': 1.1, 'consensus': 1.5, 'prior': 1.6},
    '2019-02-28': {'hicp_yoy': 1.5, 'core_yoy': 1.2, 'consensus': 1.5, 'prior': 1.4},
    '2019-03-29': {'hicp_yoy': 1.4, 'core_yoy': 0.8, 'consensus': 1.5, 'prior': 1.5},
    '2019-04-30': {'hicp_yoy': 1.7, 'core_yoy': 1.2, 'consensus': 1.6, 'prior': 1.4},
    '2019-05-31': {'hicp_yoy': 1.2, 'core_yoy': 0.8, 'consensus': 1.3, 'prior': 1.7},
    '2019-06-28': {'hicp_yoy': 1.3, 'core_yoy': 1.1, 'consensus': 1.2, 'prior': 1.2},
    '2019-07-31': {'hicp_yoy': 1.1, 'core_yoy': 0.9, 'consensus': 1.1, 'prior': 1.3},
    '2019-08-30': {'hicp_yoy': 1.0, 'core_yoy': 0.9, 'consensus': 1.0, 'prior': 1.1},
    '2019-09-30': {'hicp_yoy': 0.8, 'core_yoy': 1.0, 'consensus': 0.9, 'prior': 1.0},
    '2019-10-31': {'hicp_yoy': 0.7, 'core_yoy': 1.1, 'consensus': 0.8, 'prior': 0.8},
    '2019-11-29': {'hicp_yoy': 1.0, 'core_yoy': 1.3, 'consensus': 0.8, 'prior': 0.7},
    '2019-12-31': {'hicp_yoy': 1.3, 'core_yoy': 1.3, 'consensus': 1.3, 'prior': 1.0},
    # 2020
    '2020-01-31': {'hicp_yoy': 1.4, 'core_yoy': 1.1, 'consensus': 1.4, 'prior': 1.3},
    '2020-02-28': {'hicp_yoy': 1.2, 'core_yoy': 1.2, 'consensus': 1.2, 'prior': 1.4},
    '2020-03-31': {'hicp_yoy': 0.7, 'core_yoy': 1.0, 'consensus': 0.8, 'prior': 1.2},
    '2020-04-30': {'hicp_yoy': 0.3, 'core_yoy': 0.9, 'consensus': 0.4, 'prior': 0.7},
    '2020-05-29': {'hicp_yoy': 0.1, 'core_yoy': 0.9, 'consensus': 0.1, 'prior': 0.3},
    '2020-06-30': {'hicp_yoy': 0.3, 'core_yoy': 0.8, 'consensus': 0.2, 'prior': 0.1},
    '2020-07-31': {'hicp_yoy': 0.4, 'core_yoy': 1.2, 'consensus': 0.2, 'prior': 0.3},
    '2020-08-31': {'hicp_yoy': -0.2, 'core_yoy': 0.6, 'consensus': -0.2, 'prior': 0.4},
    '2020-09-30': {'hicp_yoy': -0.3, 'core_yoy': 0.2, 'consensus': -0.2, 'prior': -0.2},
    '2020-10-30': {'hicp_yoy': -0.3, 'core_yoy': 0.2, 'consensus': -0.3, 'prior': -0.3},
    '2020-11-30': {'hicp_yoy': -0.3, 'core_yoy': 0.2, 'consensus': -0.3, 'prior': -0.3},
    '2020-12-31': {'hicp_yoy': -0.3, 'core_yoy': 0.2, 'consensus': -0.3, 'prior': -0.3},
    # 2021
    '2021-01-29': {'hicp_yoy': 0.9, 'core_yoy': 1.4, 'consensus': 0.5, 'prior': -0.3},
    '2021-02-26': {'hicp_yoy': 0.9, 'core_yoy': 1.1, 'consensus': 0.9, 'prior': 0.9},
    '2021-03-31': {'hicp_yoy': 1.3, 'core_yoy': 0.9, 'consensus': 1.3, 'prior': 0.9},
    '2021-04-30': {'hicp_yoy': 1.6, 'core_yoy': 0.8, 'consensus': 1.6, 'prior': 1.3},
    '2021-05-31': {'hicp_yoy': 2.0, 'core_yoy': 0.9, 'consensus': 1.9, 'prior': 1.6},
    '2021-06-30': {'hicp_yoy': 1.9, 'core_yoy': 0.9, 'consensus': 1.9, 'prior': 2.0},
    '2021-07-30': {'hicp_yoy': 2.2, 'core_yoy': 0.7, 'consensus': 2.0, 'prior': 1.9},
    '2021-08-31': {'hicp_yoy': 3.0, 'core_yoy': 1.6, 'consensus': 2.7, 'prior': 2.2},
    '2021-09-30': {'hicp_yoy': 3.4, 'core_yoy': 1.9, 'consensus': 3.3, 'prior': 3.0},
    '2021-10-29': {'hicp_yoy': 4.1, 'core_yoy': 2.0, 'consensus': 3.7, 'prior': 3.4},
    '2021-11-30': {'hicp_yoy': 4.9, 'core_yoy': 2.6, 'consensus': 4.5, 'prior': 4.1},
    '2021-12-31': {'hicp_yoy': 5.0, 'core_yoy': 2.6, 'consensus': 4.7, 'prior': 4.9},
    # 2022
    '2022-01-31': {'hicp_yoy': 5.1, 'core_yoy': 2.3, 'consensus': 4.4, 'prior': 5.0},
    '2022-02-28': {'hicp_yoy': 5.8, 'core_yoy': 2.7, 'consensus': 5.3, 'prior': 5.1},
    '2022-03-31': {'hicp_yoy': 7.4, 'core_yoy': 2.9, 'consensus': 6.6, 'prior': 5.8},
    '2022-04-29': {'hicp_yoy': 7.4, 'core_yoy': 3.5, 'consensus': 7.5, 'prior': 7.4},
    '2022-05-31': {'hicp_yoy': 8.1, 'core_yoy': 3.8, 'consensus': 7.7, 'prior': 7.4},
    '2022-06-30': {'hicp_yoy': 8.6, 'core_yoy': 3.7, 'consensus': 8.5, 'prior': 8.1},
    '2022-07-29': {'hicp_yoy': 8.9, 'core_yoy': 4.0, 'consensus': 8.7, 'prior': 8.6},
    '2022-08-31': {'hicp_yoy': 9.1, 'core_yoy': 4.3, 'consensus': 9.0, 'prior': 8.9},
    '2022-09-30': {'hicp_yoy': 9.9, 'core_yoy': 4.8, 'consensus': 9.7, 'prior': 9.1},
    '2022-10-31': {'hicp_yoy': 10.6, 'core_yoy': 5.0, 'consensus': 10.2, 'prior': 9.9},
    '2022-11-30': {'hicp_yoy': 10.0, 'core_yoy': 5.0, 'consensus': 10.4, 'prior': 10.6},
    '2022-12-30': {'hicp_yoy': 9.2, 'core_yoy': 5.2, 'consensus': 9.5, 'prior': 10.0},
    # 2023
    '2023-01-31': {'hicp_yoy': 8.6, 'core_yoy': 5.3, 'consensus': 9.0, 'prior': 9.2},
    '2023-02-28': {'hicp_yoy': 8.5, 'core_yoy': 5.6, 'consensus': 8.2, 'prior': 8.6},
    '2023-03-31': {'hicp_yoy': 6.9, 'core_yoy': 5.7, 'consensus': 7.1, 'prior': 8.5},
    '2023-04-28': {'hicp_yoy': 7.0, 'core_yoy': 5.6, 'consensus': 6.9, 'prior': 6.9},
    '2023-05-31': {'hicp_yoy': 6.1, 'core_yoy': 5.3, 'consensus': 6.3, 'prior': 7.0},
    '2023-06-30': {'hicp_yoy': 5.5, 'core_yoy': 5.4, 'consensus': 5.6, 'prior': 6.1},
    '2023-07-31': {'hicp_yoy': 5.3, 'core_yoy': 5.5, 'consensus': 5.3, 'prior': 5.5},
    '2023-08-31': {'hicp_yoy': 5.2, 'core_yoy': 5.3, 'consensus': 5.3, 'prior': 5.3},
    '2023-09-29': {'hicp_yoy': 4.3, 'core_yoy': 4.5, 'consensus': 4.5, 'prior': 5.2},
    '2023-10-31': {'hicp_yoy': 2.9, 'core_yoy': 4.2, 'consensus': 3.1, 'prior': 4.3},
    '2023-11-30': {'hicp_yoy': 2.4, 'core_yoy': 3.6, 'consensus': 2.7, 'prior': 2.9},
    '2023-12-29': {'hicp_yoy': 2.9, 'core_yoy': 3.4, 'consensus': 2.7, 'prior': 2.4},
    # 2024
    '2024-01-31': {'hicp_yoy': 2.8, 'core_yoy': 3.3, 'consensus': 2.8, 'prior': 2.9},
    '2024-02-29': {'hicp_yoy': 2.6, 'core_yoy': 3.1, 'consensus': 2.5, 'prior': 2.8},
    '2024-03-29': {'hicp_yoy': 2.4, 'core_yoy': 2.9, 'consensus': 2.5, 'prior': 2.6},
    '2024-04-30': {'hicp_yoy': 2.4, 'core_yoy': 2.7, 'consensus': 2.4, 'prior': 2.4},
    '2024-05-31': {'hicp_yoy': 2.6, 'core_yoy': 2.9, 'consensus': 2.5, 'prior': 2.4},
    '2024-06-28': {'hicp_yoy': 2.5, 'core_yoy': 2.9, 'consensus': 2.5, 'prior': 2.6},
    '2024-07-31': {'hicp_yoy': 2.6, 'core_yoy': 2.9, 'consensus': 2.5, 'prior': 2.5},
    '2024-08-30': {'hicp_yoy': 2.2, 'core_yoy': 2.8, 'consensus': 2.2, 'prior': 2.6},
    '2024-09-30': {'hicp_yoy': 1.8, 'core_yoy': 2.7, 'consensus': 1.8, 'prior': 2.2},
    '2024-10-31': {'hicp_yoy': 2.0, 'core_yoy': 2.7, 'consensus': 1.9, 'prior': 1.8},
    '2024-11-29': {'hicp_yoy': 2.3, 'core_yoy': 2.7, 'consensus': 2.3, 'prior': 2.0},
    '2024-12-31': {'hicp_yoy': 2.4, 'core_yoy': 2.7, 'consensus': 2.3, 'prior': 2.3},
    # 2025
    '2025-01-31': {'hicp_yoy': 2.5, 'core_yoy': 2.7, 'consensus': 2.4, 'prior': 2.4},
    '2025-02-28': {'hicp_yoy': 2.3, 'core_yoy': 2.6, 'consensus': 2.3, 'prior': 2.5},
    '2025-03-31': {'hicp_yoy': 2.2, 'core_yoy': 2.4, 'consensus': 2.3, 'prior': 2.3},
    '2025-04-30': {'hicp_yoy': 2.2, 'core_yoy': 2.4, 'consensus': 2.1, 'prior': 2.2},
    '2025-05-30': {'hicp_yoy': 2.0, 'core_yoy': 2.3, 'consensus': 2.0, 'prior': 2.2},
    # 2026 (projected)
    '2026-01-30': {'hicp_yoy': 2.2, 'core_yoy': 2.5, 'consensus': 2.2, 'prior': 2.1},
    '2026-02-27': {'hicp_yoy': 2.1, 'core_yoy': 2.4, 'consensus': 2.1, 'prior': 2.2},
    '2026-03-31': {'hicp_yoy': 2.0, 'core_yoy': 2.3, 'consensus': 2.0, 'prior': 2.1},
}

# ═══════════════════════════════════════════════════════════════
# SESSION DEFINITIONS (UTC)
# ═══════════════════════════════════════════════════════════════

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
    df = pd.read_csv(filepath)
    df['Open time'] = pd.to_datetime(df['Open time'])
    df = df.set_index('Open time')
    for c in ['Close', 'Open', 'High', 'Low', 'Volume']:
        df[c] = df[c].astype(float)
    return df


def classify_signal(actual, consensus, prior):
    diff = actual - consensus
    surprise = diff / abs(consensus) if consensus != 0 else diff
    if surprise > 0.15:
        return 'STRONG_BEAT', surprise
    elif surprise > 0.03:
        return 'BEAT', surprise
    elif surprise < -0.15:
        return 'BIG_MISS', surprise
    elif surprise < -0.03:
        return 'MISS', surprise
    else:
        return 'INLINE', surprise


def classify_level(actual):
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


def classify_direction(actual, prior):
    change = actual - prior
    if change > 0.2:
        return 'RISING'
    elif change < -0.2:
        return 'FALLING'
    else:
        return 'STABLE'


def compute_session_returns(df, release_date, release_utc_hour=9):
    release_dt = pd.Timestamp(f"{release_date} {release_utc_hour:02d}:00:00")
    release_bar = df.index[df.index >= release_dt]
    if len(release_bar) == 0:
        return None
    release_bar = release_bar[0]
    price_at_release = df.loc[release_bar, 'Close']

    results = {}
    for phase_name, (start_h, end_h) in PHASES.items():
        if start_h >= release_utc_hour:
            phase_start_dt = pd.Timestamp(f"{release_date} {start_h:02d}:00:00")
        else:
            next_day = (pd.Timestamp(release_date) + timedelta(days=1)).strftime('%Y-%m-%d')
            phase_start_dt = pd.Timestamp(f"{next_day} {start_h:02d}:00:00")

        phase_bars = df.index[df.index >= phase_start_dt]
        if len(phase_bars) == 0:
            results[phase_name] = None
            continue
        price_at_phase = df.loc[phase_bars[0], 'Close']
        results[phase_name] = (price_at_phase - price_at_release) / price_at_release * 100

    end_24h = release_dt + timedelta(hours=24)
    end_bars = df.index[df.index >= end_24h]
    if len(end_bars) > 0:
        price_24h = df.loc[end_bars[0], 'Close']
        results['24h_return'] = (price_24h - price_at_release) / price_at_release * 100
    else:
        results['24h_return'] = None

    return results


def compute_wyckoff_phase(df, date_str, lookback_days=30):
    dt = pd.Timestamp(date_str)
    start = dt - timedelta(days=lookback_days)
    window = df[(df.index >= start) & (df.index < dt)]
    if len(window) < 100:
        return 'RANGE'
    closes = window['Close'].values.astype(float)
    highs = window['High'].values.astype(float)
    lows = window['Low'].values.astype(float)
    sma_short = np.mean(closes[-48:])
    sma_long = np.mean(closes[-192:])
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


def compute_vol_regime(df, date_str, lookback_days=7):
    dt = pd.Timestamp(date_str)
    start = dt - timedelta(days=lookback_days)
    window = df[(df.index >= start) & (df.index < dt)]
    if len(window) < 20:
        return 'NEUTRAL'
    closes = window['Close'].values.astype(float)
    highs = window['High'].values.astype(float)
    lows = window['Low'].values.astype(float)
    ranges = (highs - lows) / closes
    recent_range = ranges[-16:]
    sma = np.mean(closes[-48:])
    std = np.std(closes[-48:])
    bb_width = (2 * std) / sma if sma > 0 else 0
    diffs = np.diff(closes[-48:])
    direction_changes = np.sum(np.diff(np.sign(diffs)) != 0)
    whipsaw_rate = direction_changes / len(diffs) if len(diffs) > 0 else 0
    if bb_width < 0.015 and np.mean(recent_range) < 0.005:
        return 'COMPRESSING'
    elif whipsaw_rate > 0.6:
        return 'CHOP'
    elif np.std(diffs) > np.mean(np.abs(diffs)) * 1.5:
        return 'TRENDING'
    elif np.percentile(ranges, 50) > np.percentile(ranges, 80) * 0.8:
        return 'CRISIS'
    return 'NEUTRAL'


def run_backtest():
    print("=" * 80)
    print("PROMPT A: EUROZONE CPI FLASH BACKTEST (2018-2026)")
    print("ETH/USDT 15m data | Release: 09:00 UTC (10:00 CET / 18:00 MYT)")
    print("=" * 80)

    df = load_eth_data('eth_15m_merged.csv')
    print(f"\nLoaded {len(df)} bars: {df.index[0]} → {df.index[-1]}")

    all_results = []
    for date_str, data in sorted(EZ_CPI_RELEASES.items()):
        dt = pd.Timestamp(date_str)
        if dt < df.index[0] or dt > df.index[-1] - timedelta(days=2):
            continue
        returns = compute_session_returns(df, date_str)
        if returns is None or returns.get('24h_return') is None:
            continue

        signal, surprise = classify_signal(data['hicp_yoy'], data['consensus'], data['prior'])
        cpi_level = classify_level(data['hicp_yoy'])
        cpi_dir = classify_direction(data['hicp_yoy'], data['prior'])
        core_level = classify_level(data['core_yoy'])
        wyckoff = compute_wyckoff_phase(df, date_str)
        vol = compute_vol_regime(df, date_str)

        # Core vs headline spread (core sticky = hawkish signal)
        core_headline_spread = data['core_yoy'] - data['hicp_yoy']

        result = {
            'date': date_str, 'hicp_yoy': data['hicp_yoy'],
            'core_yoy': data['core_yoy'], 'consensus': data['consensus'],
            'prior': data['prior'], 'signal': signal, 'surprise': surprise,
            'cpi_level': cpi_level, 'cpi_direction': cpi_dir,
            'core_level': core_level, 'core_headline_spread': core_headline_spread,
            'wyckoff': wyckoff, 'vol': vol, **returns
        }
        all_results.append(result)

    df_results = pd.DataFrame(all_results)
    print(f"\nAnalyzed {len(df_results)} Eurozone CPI Flash releases")

    # Session returns
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

    valid_24h = df_results['24h_return'].dropna()
    print(f"\n{'24h AGGREGATE':<28} {valid_24h.mean():>8.3f} {(valid_24h > 0).mean() * 100:>7.1f}% {len(valid_24h):>5}")

    # Cross-tabulation
    print("\n" + "=" * 80)
    print("CROSS-TABULATION: Wyckoff × Vol × Signal → 24h Return")
    print("=" * 80)

    combos = df_results.groupby(['wyckoff', 'vol', 'signal']).agg(
        avg_24h=('24h_return', 'mean'),
        win_rate=('24h_return', lambda x: (x > 0).mean() * 100),
        count=('24h_return', 'count'),
    ).reset_index()
    combos = combos[combos['count'] >= 3].sort_values('avg_24h', key=abs, ascending=False)

    print(f"\n{'Wyckoff':<14} {'Vol':<12} {'Signal':<14} {'Avg24h%':>10} {'Win%':>8} {'N':>5}")
    print("-" * 70)
    for _, row in combos.iterrows():
        edge = "✅" if abs(row['avg_24h']) >= 0.5 else "  "
        print(f"{row['wyckoff']:<14} {row['vol']:<12} {row['signal']:<14} "
              f"{row['avg_24h']:>9.3f}% {row['win_rate']:>7.1f}% {int(row['count']):>4} {edge}")

    # Core vs Headline spread analysis
    print("\n" + "=" * 80)
    print("CORE-HEADLINE SPREAD × SIGNAL → 24h Return")
    print("(core_yoy - hicp_yoy: positive = sticky core inflation)")
    print("=" * 80)

    df_results['core_spread_bucket'] = pd.cut(
        df_results['core_headline_spread'],
        bins=[-999, -0.5, 0, 0.5, 1, 999],
        labels=['CORE_BELOW', 'CORE_EQUAL', 'CORE_SLIGHT', 'CORE_STICKY', 'CORE_VERY_STICKY']
    )

    spread_combos = df_results.groupby(['core_spread_bucket', 'signal']).agg(
        avg_24h=('24h_return', 'mean'),
        win_rate=('24h_return', lambda x: (x > 0).mean() * 100),
        count=('24h_return', 'count'),
    ).reset_index()
    spread_combos = spread_combos[spread_combos['count'] >= 3].sort_values('avg_24h', key=abs, ascending=False)

    print(f"\n{'Core Spread':<18} {'Signal':<14} {'Avg24h%':>10} {'Win%':>8} {'N':>5}")
    print("-" * 60)
    for _, row in spread_combos.iterrows():
        edge = "✅" if abs(row['avg_24h']) >= 0.5 else "  "
        print(f"{row['core_spread_bucket']:<18} {row['signal']:<14} "
              f"{row['avg_24h']:>9.3f}% {row['win_rate']:>7.1f}% {int(row['count']):>4} {edge}")

    return df_results


def run_transmission_chain(df_results):
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

    print("\nDIRECTION PERSISTENCE BETWEEN CONSECUTIVE SESSIONS")
    print("-" * 70)

    valid_phases = [p for p in session_phases if p in df_results.columns and df_results[p].notna().sum() >= 5]
    transitions = []

    for i in range(len(valid_phases) - 1):
        p1, p2 = valid_phases[i], valid_phases[i+1]
        mask = df_results[p1].notna() & df_results[p2].notna()
        subset = df_results[mask]
        if len(subset) < 5:
            continue
        same_dir = ((subset[p1] > 0) & (subset[p2] > 0)) | ((subset[p1] < 0) & (subset[p2] < 0))
        pct_same = same_dir.mean() * 100
        corr = subset[p1].corr(subset[p2])

        edge_label = "✅ REAL EDGE" if pct_same > 65 else "⚠️  MARGINAL" if pct_same >= 55 else "❌ NO CHAIN"
        transitions.append({'from': p1, 'to': p2, 'pct_same': pct_same, 'corr': corr, 'n': len(subset), 'edge': edge_label})
        print(f"  {p1:<28} → {p2:<24} {pct_same:>5.1f}% same  (r={corr:>5.2f}, n={len(subset)}) {edge_label}")

    # Chain from release to end-of-day
    print("\n\nCHAIN FROM RELEASE TO END-OF-DAY")
    print("-" * 70)
    first_phase = valid_phases[0] if valid_phases else None
    if first_phase:
        for phase in valid_phases[1:]:
            mask = df_results[first_phase].notna() & df_results[phase].notna()
            subset = df_results[mask]
            if len(subset) < 3:
                continue
            same_dir = ((subset[first_phase] > 0) & (subset[phase] > 0)) | ((subset[first_phase] < 0) & (subset[phase] < 0))
            pct_same = same_dir.mean() * 100
            edge_label = "✅" if pct_same > 65 else "⚠️" if pct_same >= 55 else "❌"
            print(f"  {first_phase:<28} → {phase:<24} {pct_same:>5.1f}% persist {edge_label} (n={len(subset)})")

    # Statistical tests
    print("\n\n" + "=" * 80)
    print("STATISTICAL SIGNIFICANCE TESTS")
    print("=" * 80)

    returns_24h = df_results['24h_return'].dropna()
    t_stat, p_value = stats.ttest_1samp(returns_24h, 0)
    print(f"\n1. One-sample t-test (H0: mean 24h return = 0)")
    print(f"   Mean: {returns_24h.mean():.4f}%  t = {t_stat:.4f}, p = {p_value:.4f}")
    print(f"   {'✅ SIGNIFICANT (p<0.05)' if p_value < 0.05 else '❌ NOT significant (p≥0.05)'}")

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

    # Core sticky vs not
    sticky_mask = df_results['core_headline_spread'] > 0.5
    not_sticky_mask = df_results['core_headline_spread'] <= 0
    sticky_returns = df_results.loc[sticky_mask, '24h_return'].dropna()
    not_sticky_returns = df_results.loc[not_sticky_mask, '24h_return'].dropna()
    if len(sticky_returns) >= 3 and len(not_sticky_returns) >= 3:
        t_stat3, p_value3 = stats.ttest_ind(sticky_returns, not_sticky_returns)
        print(f"\n3. Two-sample t-test (Core sticky >0.5pp vs Core ≤ headline)")
        print(f"   Sticky mean: {sticky_returns.mean():.4f}% (n={len(sticky_returns)})")
        print(f"   Not sticky mean: {not_sticky_returns.mean():.4f}% (n={len(not_sticky_returns)})")
        print(f"   t = {t_stat3:.4f}, p = {p_value3:.4f}")
        print(f"   {'✅ SIGNIFICANT (p<0.05)' if p_value3 < 0.05 else '❌ NOT significant (p≥0.05)'}")

    return transitions


def main():
    df_results = run_backtest()
    transitions = run_transmission_chain(df_results)
    df_results.to_json('backtest_ez_cpi_results.json', orient='records', indent=2)
    print(f"\n\nResults saved to backtest_ez_cpi_results.json")

    # Summary
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
        print("\nCombos with edge (n≥3, |avg24h|≥0.5%):")
        for _, row in edge_combos.iterrows():
            direction = "LONG" if row['avg_24h'] > 0 else "SHORT"
            print(f"  {row['wyckoff']} + {row['vol']} + {row['signal']}: "
                  f"{row['avg_24h']:+.3f}% avg, {row['win_rate']:.0f}% win, n={int(row['count'])} → {direction} bias")

    print("\nStrong transmission links (>65% same direction):")
    for t in transitions:
        if t['pct_same'] > 65:
            print(f"  {t['from']} → {t['to']}: {t['pct_same']:.1f}% persist (r={t['corr']:.2f})")

    return df_results


if __name__ == '__main__':
    main()

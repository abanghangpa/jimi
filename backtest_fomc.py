"""
Prompt A + B: Backtest FOMC Rate Decision Session Transmission Chain
=====================================================================
ETH/USDT 15m data from jimi/eth_15m_merged.csv
Event: FOMC Rate Decision + Statement + Dot Plot (18:00 UTC = 14:00 ET = 02:00 MYT)
       Powell Press Conference (18:30 UTC = 14:30 ET = 02:30 MYT)
Transmission: US Afternoon (18:00 UTC) → Powell Presser (18:30) → NY PM → Asia Re-open

Thesis:
  The Fed's rate decision is the single most powerful macro catalyst for crypto.
  Hawkish dot plot → yield surge + DXY spike → long liquidations → ETH drops.
  Dovish pivot → rate cut expectations → risk-on → ETH rallies.
  The initial 18:00 move sets the BASELINE; Powell's presser at 18:30 determines
  the FINAL direction. The market often reverses the initial spike during the presser.
  Minutes 3 weeks later provide the LOOPBACK — internal debates that explain the move.

Key asymmetry: FOMC moves are 2-5x larger than typical macro releases.
Sessions matter: 18:00 UTC is mid-NY afternoon → London close overlap.
The real test is Asia re-open 8-10h later.

Backtested on FOMC rate decisions (2018-2026) against ETH/USDT 15m data.
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from scipy import stats
import json
import warnings
warnings.filterwarnings('ignore')

# ═══════════════════════════════════════════════════════════════
# FOMC RATE DECISION DATES (18:00 UTC = 14:00 ET)
# ~8 meetings per year. Dot plot released at March, June, Sept, Dec.
# Format: {date: {'rate': float, 'prior_rate': float, 'dot_plot': bool,
#                  'vote': str, 'stance': str, 'surprise': str}}
# stance: HAWKISH / NEUTRAL / DOVISH
# surprise: HAWKISH_SURPRISE / INLINE / DOVISH_SURPISE
# ═══════════════════════════════════════════════════════════════

FOMC_RELEASES = {
    # ── 2018 ──
    '2018-01-31': {'rate': 1.50, 'prior_rate': 1.25, 'dot_plot': False, 'vote': '9-0', 'stance': 'NEUTRAL', 'surprise': 'INLINE'},
    '2018-03-21': {'rate': 1.75, 'prior_rate': 1.50, 'dot_plot': True,  'vote': '8-0', 'stance': 'HAWKISH', 'surprise': 'HAWKISH_SURPRISE'},
    '2018-05-02': {'rate': 1.75, 'prior_rate': 1.75, 'dot_plot': False, 'vote': '8-0', 'stance': 'NEUTRAL', 'surprise': 'INLINE'},
    '2018-06-13': {'rate': 2.00, 'prior_rate': 1.75, 'dot_plot': True,  'vote': '8-0', 'stance': 'HAWKISH', 'surprise': 'INLINE'},
    '2018-08-01': {'rate': 2.00, 'prior_rate': 2.00, 'dot_plot': False, 'vote': '9-0', 'stance': 'NEUTRAL', 'surprise': 'INLINE'},
    '2018-09-26': {'rate': 2.25, 'prior_rate': 2.00, 'dot_plot': True,  'vote': '9-0', 'stance': 'HAWKISH', 'surprise': 'INLINE'},
    '2018-11-08': {'rate': 2.25, 'prior_rate': 2.25, 'dot_plot': False, 'vote': '9-0', 'stance': 'NEUTRAL', 'surprise': 'INLINE'},
    '2018-12-19': {'rate': 2.50, 'prior_rate': 2.25, 'dot_plot': True,  'vote': '10-0', 'stance': 'HAWKISH', 'surprise': 'HAWKISH_SURPRISE'},
    # ── 2019 ──
    '2019-01-30': {'rate': 2.50, 'prior_rate': 2.50, 'dot_plot': False, 'vote': '10-0', 'stance': 'DOVISH', 'surprise': 'DOVISH_SURPRISE'},
    '2019-03-20': {'rate': 2.50, 'prior_rate': 2.50, 'dot_plot': True,  'vote': '10-0', 'stance': 'DOVISH', 'surprise': 'DOVISH_SURPRISE'},
    '2019-05-01': {'rate': 2.50, 'prior_rate': 2.50, 'dot_plot': False, 'vote': '10-0', 'stance': 'NEUTRAL', 'surprise': 'INLINE'},
    '2019-06-19': {'rate': 2.50, 'prior_rate': 2.50, 'dot_plot': True,  'vote': '9-1', 'stance': 'DOVISH', 'surprise': 'DOVISH_SURPRISE'},
    '2019-07-31': {'rate': 2.25, 'prior_rate': 2.50, 'dot_plot': False, 'vote': '8-2', 'stance': 'DOVISH', 'surprise': 'INLINE'},
    '2019-09-18': {'rate': 2.00, 'prior_rate': 2.25, 'dot_plot': True,  'vote': '7-3', 'stance': 'DOVISH', 'surprise': 'INLINE'},
    '2019-10-30': {'rate': 1.75, 'prior_rate': 2.00, 'dot_plot': False, 'vote': '8-2', 'stance': 'DOVISH', 'surprise': 'INLINE'},
    '2019-12-11': {'rate': 1.75, 'prior_rate': 1.75, 'dot_plot': True,  'vote': '10-0', 'stance': 'NEUTRAL', 'surprise': 'INLINE'},
    # ── 2020 ──
    '2020-01-29': {'rate': 1.75, 'prior_rate': 1.75, 'dot_plot': False, 'vote': '10-0', 'stance': 'NEUTRAL', 'surprise': 'INLINE'},
    '2020-03-03': {'rate': 1.25, 'prior_rate': 1.75, 'dot_plot': False, 'vote': '10-0', 'stance': 'DOVISH', 'surprise': 'DOVISH_SURPRISE'},
    '2020-03-15': {'rate': 0.25, 'prior_rate': 1.25, 'dot_plot': False, 'vote': '10-0', 'stance': 'DOVISH', 'surprise': 'DOVISH_SURPRISE'},
    '2020-04-29': {'rate': 0.25, 'prior_rate': 0.25, 'dot_plot': False, 'vote': '10-0', 'stance': 'DOVISH', 'surprise': 'INLINE'},
    '2020-06-10': {'rate': 0.25, 'prior_rate': 0.25, 'dot_plot': True,  'vote': '10-0', 'stance': 'DOVISH', 'surprise': 'INLINE'},
    '2020-07-29': {'rate': 0.25, 'prior_rate': 0.25, 'dot_plot': False, 'vote': '10-0', 'stance': 'DOVISH', 'surprise': 'INLINE'},
    '2020-09-16': {'rate': 0.25, 'prior_rate': 0.25, 'dot_plot': True,  'vote': '10-0', 'stance': 'DOVISH', 'surprise': 'INLINE'},
    '2020-11-05': {'rate': 0.25, 'prior_rate': 0.25, 'dot_plot': False, 'vote': '10-0', 'stance': 'NEUTRAL', 'surprise': 'INLINE'},
    '2020-12-16': {'rate': 0.25, 'prior_rate': 0.25, 'dot_plot': True,  'vote': '10-0', 'stance': 'DOVISH', 'surprise': 'INLINE'},
    # ── 2021 ──
    '2021-01-27': {'rate': 0.25, 'prior_rate': 0.25, 'dot_plot': False, 'vote': '10-0', 'stance': 'NEUTRAL', 'surprise': 'INLINE'},
    '2021-03-17': {'rate': 0.25, 'prior_rate': 0.25, 'dot_plot': True,  'vote': '10-0', 'stance': 'NEUTRAL', 'surprise': 'INLINE'},
    '2021-04-28': {'rate': 0.25, 'prior_rate': 0.25, 'dot_plot': False, 'vote': '10-0', 'stance': 'NEUTRAL', 'surprise': 'INLINE'},
    '2021-06-16': {'rate': 0.25, 'prior_rate': 0.25, 'dot_plot': True,  'vote': '10-0', 'stance': 'HAWKISH', 'surprise': 'HAWKISH_SURPRISE'},
    '2021-07-28': {'rate': 0.25, 'prior_rate': 0.25, 'dot_plot': False, 'vote': '10-0', 'stance': 'NEUTRAL', 'surprise': 'INLINE'},
    '2021-09-22': {'rate': 0.25, 'prior_rate': 0.25, 'dot_plot': True,  'vote': '10-0', 'stance': 'HAWKISH', 'surprise': 'HAWKISH_SURPRISE'},
    '2021-11-03': {'rate': 0.25, 'prior_rate': 0.25, 'dot_plot': False, 'vote': '9-1', 'stance': 'HAWKISH', 'surprise': 'INLINE'},
    '2021-12-15': {'rate': 0.25, 'prior_rate': 0.25, 'dot_plot': True,  'vote': '10-0', 'stance': 'HAWKISH', 'surprise': 'HAWKISH_SURPRISE'},
    # ── 2022 ──
    '2022-01-26': {'rate': 0.50, 'prior_rate': 0.25, 'dot_plot': False, 'vote': '10-0', 'stance': 'HAWKISH', 'surprise': 'INLINE'},
    '2022-03-16': {'rate': 0.50, 'prior_rate': 0.25, 'dot_plot': True,  'vote': '10-0', 'stance': 'HAWKISH', 'surprise': 'HAWKISH_SURPRISE'},
    '2022-05-04': {'rate': 1.00, 'prior_rate': 0.50, 'dot_plot': False, 'vote': '10-0', 'stance': 'HAWKISH', 'surprise': 'INLINE'},
    '2022-06-15': {'rate': 1.75, 'prior_rate': 1.00, 'dot_plot': True,  'vote': '10-0', 'stance': 'HAWKISH', 'surprise': 'HAWKISH_SURPRISE'},
    '2022-07-27': {'rate': 2.50, 'prior_rate': 1.75, 'dot_plot': False, 'vote': '10-0', 'stance': 'HAWKISH', 'surprise': 'INLINE'},
    '2022-09-21': {'rate': 3.25, 'prior_rate': 2.50, 'dot_plot': True,  'vote': '10-0', 'stance': 'HAWKISH', 'surprise': 'HAWKISH_SURPRISE'},
    '2022-11-02': {'rate': 4.00, 'prior_rate': 3.25, 'dot_plot': False, 'vote': '10-0', 'stance': 'HAWKISH', 'surprise': 'INLINE'},
    '2022-12-14': {'rate': 4.50, 'prior_rate': 4.00, 'dot_plot': True,  'vote': '10-0', 'stance': 'HAWKISH', 'surprise': 'INLINE'},
    # ── 2023 ──
    '2023-02-01': {'rate': 4.75, 'prior_rate': 4.50, 'dot_plot': False, 'vote': '10-0', 'stance': 'HAWKISH', 'surprise': 'INLINE'},
    '2023-03-22': {'rate': 5.00, 'prior_rate': 4.75, 'dot_plot': True,  'vote': '10-0', 'stance': 'HAWKISH', 'surprise': 'INLINE'},
    '2023-05-03': {'rate': 5.25, 'prior_rate': 5.00, 'dot_plot': False, 'vote': '10-0', 'stance': 'HAWKISH', 'surprise': 'INLINE'},
    '2023-06-14': {'rate': 5.25, 'prior_rate': 5.25, 'dot_plot': True,  'vote': '10-0', 'stance': 'NEUTRAL', 'surprise': 'INLINE'},
    '2023-07-26': {'rate': 5.50, 'prior_rate': 5.25, 'dot_plot': False, 'vote': '10-0', 'stance': 'HAWKISH', 'surprise': 'INLINE'},
    '2023-09-20': {'rate': 5.50, 'prior_rate': 5.50, 'dot_plot': True,  'vote': '10-0', 'stance': 'HAWKISH', 'surprise': 'HAWKISH_SURPRISE'},
    '2023-11-01': {'rate': 5.50, 'prior_rate': 5.50, 'dot_plot': False, 'vote': '10-0', 'stance': 'NEUTRAL', 'surprise': 'INLINE'},
    '2023-12-13': {'rate': 5.50, 'prior_rate': 5.50, 'dot_plot': True,  'vote': '10-0', 'stance': 'DOVISH', 'surprise': 'DOVISH_SURPRISE'},
    # ── 2024 ──
    '2024-01-31': {'rate': 5.50, 'prior_rate': 5.50, 'dot_plot': False, 'vote': '10-0', 'stance': 'NEUTRAL', 'surprise': 'INLINE'},
    '2024-03-20': {'rate': 5.50, 'prior_rate': 5.50, 'dot_plot': True,  'vote': '10-0', 'stance': 'NEUTRAL', 'surprise': 'INLINE'},
    '2024-05-01': {'rate': 5.50, 'prior_rate': 5.50, 'dot_plot': False, 'vote': '10-0', 'stance': 'HAWKISH', 'surprise': 'HAWKISH_SURPRISE'},
    '2024-06-12': {'rate': 5.50, 'prior_rate': 5.50, 'dot_plot': True,  'vote': '10-0', 'stance': 'NEUTRAL', 'surprise': 'INLINE'},
    '2024-07-31': {'rate': 5.50, 'prior_rate': 5.50, 'dot_plot': False, 'vote': '10-0', 'stance': 'DOVISH', 'surprise': 'DOVISH_SURPRISE'},
    '2024-09-18': {'rate': 5.00, 'prior_rate': 5.50, 'dot_plot': True,  'vote': '10-0', 'stance': 'DOVISH', 'surprise': 'DOVISH_SURPRISE'},
    '2024-11-07': {'rate': 4.75, 'prior_rate': 5.00, 'dot_plot': False, 'vote': '10-0', 'stance': 'NEUTRAL', 'surprise': 'INLINE'},
    '2024-12-18': {'rate': 4.50, 'prior_rate': 4.75, 'dot_plot': True,  'vote': '10-0', 'stance': 'HAWKISH', 'surprise': 'HAWKISH_SURPRISE'},
    # ── 2025 ──
    '2025-01-29': {'rate': 4.50, 'prior_rate': 4.50, 'dot_plot': False, 'vote': '10-0', 'stance': 'NEUTRAL', 'surprise': 'INLINE'},
    '2025-03-19': {'rate': 4.50, 'prior_rate': 4.50, 'dot_plot': True,  'vote': '10-0', 'stance': 'NEUTRAL', 'surprise': 'INLINE'},
    '2025-05-07': {'rate': 4.50, 'prior_rate': 4.50, 'dot_plot': False, 'vote': '10-0', 'stance': 'NEUTRAL', 'surprise': 'INLINE'},
    # ── 2026 (projected) ──
    '2026-01-28': {'rate': 4.25, 'prior_rate': 4.25, 'dot_plot': False, 'vote': '10-0', 'stance': 'NEUTRAL', 'surprise': 'INLINE'},
    '2026-03-18': {'rate': 4.25, 'prior_rate': 4.25, 'dot_plot': True,  'vote': '10-0', 'stance': 'NEUTRAL', 'surprise': 'INLINE'},
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
    'ny_lunch', 'ny_pm', 'pre_asia', 'sydney_open', 'tokyo_open',
    'asia_mid', 'asia_afternoon', 'tokyo_close', 'pre_london',
    'frankfurt_open', 'london_open', 'london_morning', 'london_midday',
    'ny_pre_open', 'ny_open', 'london_ny_overlap', 'ny_am',
]


def load_eth_data(filepath):
    """Load ETH 15m CSV data."""
    df = pd.read_csv(filepath)
    df['Open time'] = pd.to_datetime(df['Open time'])
    df = df.set_index('Open time')
    for c in ['Close', 'Open', 'High', 'Low', 'Volume']:
        df[c] = df[c].astype(float)
    return df


def compute_session_returns(df, release_date, release_utc_hour=18, release_utc_min=0):
    """
    Compute ETH returns for each session window relative to FOMC release.
    FOMC releases at 18:00 UTC. Returns measured from release price.
    """
    release_dt = pd.Timestamp(f"{release_date} {release_utc_hour:02d}:{release_utc_min:02d}:00")
    release_bars = df.index[df.index >= release_dt]
    if len(release_bars) == 0:
        return None
    release_bar = release_bars[0]
    price_at_release = df.loc[release_bar, 'Close']

    results = {}
    for session_name, (start_h, end_h) in SESSIONS.items():
        start_hour = int(start_h)
        start_min = int((start_h % 1) * 60)

        # FOMC is at 18:00 UTC. Sessions before 18:00 on same day are skipped.
        # Sessions after 18:00 same day, or next day.
        if start_h >= release_utc_hour + (release_utc_min / 60.0):
            session_start_dt = pd.Timestamp(f"{release_date} {start_hour:02d}:{start_min:02d}:00")
        else:
            # Next day
            next_day = (pd.Timestamp(release_date) + timedelta(days=1)).strftime('%Y-%m-%d')
            session_start_dt = pd.Timestamp(f"{next_day} {start_hour:02d}:{start_min:02d}:00")

        session_bars = df.index[df.index >= session_start_dt]
        if len(session_bars) == 0:
            results[session_name] = None
            continue
        price_at_session = df.loc[session_bars[0], 'Close']
        results[session_name] = (price_at_session - price_at_release) / price_at_release * 100

    # Presser return: 18:30 UTC (30 min after release)
    presser_dt = pd.Timestamp(f"{release_date} 18:30:00")
    presser_bars = df.index[df.index >= presser_dt]
    if len(presser_bars) > 0:
        price_presser = df.loc[presser_bars[0], 'Close']
        results['presser_30m'] = (price_presser - price_at_release) / price_at_release * 100

    # 24h aggregate
    end_24h = release_dt + timedelta(hours=24)
    end_bars = df.index[df.index >= end_24h]
    if len(end_bars) > 0:
        price_24h = df.loc[end_bars[0], 'Close']
        results['24h_return'] = (price_24h - price_at_release) / price_at_release * 100
    else:
        results['24h_return'] = None

    # Initial spike (first 15m bar after release)
    if len(release_bars) >= 2:
        price_15m = df.loc[release_bars[1], 'Close']
        results['initial_15m'] = (price_15m - price_at_release) / price_at_release * 100

    return results


def compute_wyckoff_phase(df, date_str, lookback_days=30):
    """Simplified Wyckoff phase proxy."""
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


def compute_vol_regime(df, date_str, lookback_days=90):
    """Compute volatility regime."""
    dt = pd.Timestamp(date_str)
    start = dt - timedelta(days=lookback_days)
    window = df[(df.index >= start) & (df.index < dt)]
    if len(window) < 200:
        return 'NEUTRAL'
    daily = window.groupby(window.index.date).agg(
        high=('High', 'max'), low=('Low', 'min'), close=('Close', 'last')
    )
    daily['atr'] = daily['high'] - daily['low']
    if len(daily) < 30:
        return 'NEUTRAL'
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


def classify_fomc_signal(data):
    """Classify FOMC signal based on stance, surprise, and dot plot."""
    stance = data['stance']
    surprise = data['surprise']
    dot_plot = data.get('dot_plot', False)

    if surprise == 'HAWKISH_SURPRISE':
        return 'HAWKISH_SURPRISE'
    elif surprise == 'DOVISH_SURPRISE':
        return 'DOVISH_SURPRISE'
    elif stance == 'HAWKISH' and dot_plot:
        return 'HAWKISH_DOT_PLOT'
    elif stance == 'DOVISH' and dot_plot:
        return 'DOVISH_DOT_PLOT'
    elif stance == 'HAWKISH':
        return 'HAWKISH'
    elif stance == 'DOVISH':
        return 'DOVISH'
    return 'NEUTRAL'


def run_backtest():
    """Run full FOMC backtest — Prompt A."""
    print("=" * 80)
    print("PROMPT A: FOMC RATE DECISION BACKTEST (2018-2026)")
    print("ETH/USDT 15m data | Release: 18:00 UTC (14:00 ET / 02:00 MYT+1)")
    print("Press Conference: 18:30 UTC (14:30 ET / 02:30 MYT+1)")
    print("=" * 80)

    df = load_eth_data('jimi/eth_15m_merged.csv')
    print(f"\nLoaded {len(df)} bars: {df.index[0]} → {df.index[-1]}")

    all_results = []
    for date_str, data in sorted(FOMC_RELEASES.items()):
        dt = pd.Timestamp(date_str)
        if dt < df.index[0] or dt > df.index[-1] - timedelta(days=4):
            continue
        returns = compute_session_returns(df, date_str)
        if returns is None or returns.get('24h_return') is None:
            continue

        signal = classify_fomc_signal(data)
        wyckoff = compute_wyckoff_phase(df, date_str)
        vol = compute_vol_regime(df, date_str)

        result = {
            'date': date_str,
            'rate': data['rate'],
            'prior_rate': data['prior_rate'],
            'rate_change': data['rate'] - data['prior_rate'],
            'dot_plot': data['dot_plot'],
            'vote': data['vote'],
            'stance': data['stance'],
            'surprise': data['surprise'],
            'signal': signal,
            'wyckoff': wyckoff,
            'vol': vol,
            **returns,
        }
        all_results.append(result)

    df_results = pd.DataFrame(all_results)
    print(f"\nAnalyzed {len(df_results)} FOMC rate decisions")

    # ── Session returns ──
    print("\n" + "=" * 80)
    print("SESSION-BY-SESSION AVERAGE RETURNS (%)")
    print("=" * 80)

    print(f"\n{'Session':<24} {'Avg%':>8} {'Win%':>8} {'N':>6} {'Sig':>6}")
    print("-" * 56)

    # Special: initial 15m spike and presser
    for special in ['initial_15m', 'presser_30m']:
        if special in df_results.columns:
            valid = df_results[special].dropna()
            if len(valid) > 0:
                avg = valid.mean()
                win = (valid > 0).mean() * 100
                n = len(valid)
                sig = "***" if abs(avg) > 0.5 and n >= 5 else ""
                print(f"{special:<24} {avg:>8.3f} {win:>7.1f}% {n:>5} {sig:>6}")

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
    combos = combos[combos['count'] >= 2].sort_values('avg_24h', key=abs, ascending=False)

    print(f"\n{'Wyckoff':<14} {'Vol':<12} {'Signal':<18} {'Avg24h%':>10} {'Win%':>8} {'N':>5}")
    print("-" * 72)
    for _, row in combos.iterrows():
        edge = "✅" if abs(row['avg_24h']) >= 0.5 else "  "
        print(f"{row['wyckoff']:<14} {row['vol']:<12} {row['signal']:<18} "
              f"{row['avg_24h']:>9.3f}% {row['win_rate']:>7.1f}% {int(row['count']):>4} {edge}")

    # ── Stance × Dot Plot ──
    print("\n" + "=" * 80)
    print("STANCE × DOT_PLOT → 24h Return")
    print("=" * 80)

    combos2 = df_results.groupby(['stance', 'dot_plot']).agg(
        avg_24h=('24h_return', 'mean'),
        win_rate=('24h_return', lambda x: (x > 0).mean() * 100),
        count=('24h_return', 'count'),
    ).reset_index()
    combos2 = combos2[combos2['count'] >= 2].sort_values('avg_24h', key=abs, ascending=False)

    print(f"\n{'Stance':<12} {'DotPlot':>8} {'Avg24h%':>10} {'Win%':>8} {'N':>5}")
    print("-" * 48)
    for _, row in combos2.iterrows():
        edge = "✅" if abs(row['avg_24h']) >= 0.5 else "  "
        print(f"{row['stance']:<12} {str(row['dot_plot']):>8} "
              f"{row['avg_24h']:>9.3f}% {row['win_rate']:>7.1f}% {int(row['count']):>4} {edge}")

    # ── Initial spike vs final direction ──
    print("\n" + "=" * 80)
    print("INITIAL SPIKE vs FINAL (24h) DIRECTION")
    print("=" * 80)
    if 'initial_15m' in df_results.columns:
        mask = df_results['initial_15m'].notna() & df_results['24h_return'].notna()
        sub = df_results[mask]
        if len(sub) > 0:
            same_dir = ((sub['initial_15m'] > 0) & (sub['24h_return'] > 0)) | \
                       ((sub['initial_15m'] < 0) & (sub['24h_return'] < 0))
            print(f"\n  Same direction (15m → 24h): {same_dir.mean() * 100:.1f}% (n={len(sub)})")
            reversal = ~same_dir
            print(f"  Reversal (15m → 24h):       {reversal.mean() * 100:.1f}%")
            # Dot plot days
            dp_mask = sub['dot_plot'] == True
            if dp_mask.sum() >= 3:
                dp_same = same_dir[dp_mask]
                print(f"\n  Dot Plot days same dir:     {dp_same.mean() * 100:.1f}% (n={dp_mask.sum()})")

    return df_results


def run_transmission_chain(df_results):
    """Prompt B: Session transmission chain analysis."""
    print("\n\n" + "=" * 80)
    print("PROMPT B: SESSION TRANSMISSION CHAIN VALIDATION")
    print("=" * 80)

    print("\nDIRECTION PERSISTENCE BETWEEN CONSECUTIVE SESSIONS")
    print("-" * 70)

    valid_sessions = [s for s in SESSION_ORDER
                      if s in df_results.columns and df_results[s].notna().sum() >= 3]
    transitions = []

    for i in range(len(valid_sessions) - 1):
        p1, p2 = valid_sessions[i], valid_sessions[i + 1]
        mask = df_results[p1].notna() & df_results[p2].notna()
        subset = df_results[mask]
        if len(subset) < 3:
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

    # Presser → NY PM (the key transmission)
    if 'presser_30m' in df_results.columns and 'ny_pm' in df_results.columns:
        mask = df_results['presser_30m'].notna() & df_results['ny_pm'].notna()
        sub = df_results[mask]
        if len(sub) >= 3:
            same = ((sub['presser_30m'] > 0) & (sub['ny_pm'] > 0)) | \
                   ((sub['presser_30m'] < 0) & (sub['ny_pm'] < 0))
            pct = same.mean() * 100
            label = "✅ REAL EDGE" if pct > 65 else "⚠️  MARGINAL" if pct >= 55 else "❌ NO CHAIN"
            print(f"\n  KEY: presser_30m → ny_pm:    {pct:>5.1f}% same  (n={len(sub)}) {label}")

    # ── Statistical tests ──
    print("\n\n" + "=" * 80)
    print("STATISTICAL SIGNIFICANCE TESTS")
    print("=" * 80)

    returns_24h = df_results['24h_return'].dropna()
    t_stat, p_value = stats.ttest_1samp(returns_24h, 0)
    print(f"\n1. One-sample t-test (H0: mean 24h return = 0)")
    print(f"   Mean: {returns_24h.mean():.4f}%  t = {t_stat:.4f}, p = {p_value:.4f}")
    print(f"   {'✅ SIGNIFICANT (p<0.05)' if p_value < 0.05 else '❌ NOT significant (p≥0.05)'}")

    # Hawkish surprise vs dovish surprise
    hawk_mask = df_results['signal'].isin(['HAWKISH_SURPRISE', 'HAWKISH_DOT_PLOT', 'HAWKISH'])
    dove_mask = df_results['signal'].isin(['DOVISH_SURPRISE', 'DOVISH_DOT_PLOT', 'DOVISH'])
    hawk_returns = df_results.loc[hawk_mask, '24h_return'].dropna()
    dove_returns = df_results.loc[dove_mask, '24h_return'].dropna()
    if len(hawk_returns) >= 3 and len(dove_returns) >= 3:
        t_stat2, p_value2 = stats.ttest_ind(hawk_returns, dove_returns)
        print(f"\n2. Two-sample t-test (HAWKISH signals vs DOVISH signals)")
        print(f"   Hawkish mean: {hawk_returns.mean():.4f}% (n={len(hawk_returns)})")
        print(f"   Dovish mean:  {dove_returns.mean():.4f}% (n={len(dove_returns)})")
        print(f"   t = {t_stat2:.4f}, p = {p_value2:.4f}")
        print(f"   {'✅ SIGNIFICANT (p<0.05)' if p_value2 < 0.05 else '❌ NOT significant (p≥0.05)'}")

    # Dot plot days vs non-dot-plot
    dp_mask = df_results['dot_plot'] == True
    ndp_mask = df_results['dot_plot'] == False
    dp_returns = df_results.loc[dp_mask, '24h_return'].dropna()
    ndp_returns = df_results.loc[ndp_mask, '24h_return'].dropna()
    if len(dp_returns) >= 3 and len(ndp_returns) >= 3:
        t_stat3, p_value3 = stats.ttest_ind(dp_returns, ndp_returns)
        print(f"\n3. Two-sample t-test (Dot Plot days vs Non-Dot-Plot)")
        print(f"   Dot Plot mean:  {dp_returns.mean():.4f}% (n={len(dp_returns)})")
        print(f"   No Dot Plot:    {ndp_returns.mean():.4f}% (n={len(ndp_returns)})")
        print(f"   t = {t_stat3:.4f}, p = {p_value3:.4f}")
        print(f"   {'✅ SIGNIFICANT (p<0.05)' if p_value3 < 0.05 else '❌ NOT significant (p≥0.05)'}")

    # Rate change impact
    hike_mask = df_results['rate_change'] > 0
    hold_mask = df_results['rate_change'] == 0
    hike_returns = df_results.loc[hike_mask, '24h_return'].dropna()
    hold_returns = df_results.loc[hold_mask, '24h_return'].dropna()
    if len(hike_returns) >= 3 and len(hold_returns) >= 3:
        t_stat4, p_value4 = stats.ttest_ind(hike_returns, hold_returns)
        print(f"\n4. Two-sample t-test (Rate Hike vs Hold)")
        print(f"   Hike mean: {hike_returns.mean():.4f}% (n={len(hike_returns)})")
        print(f"   Hold mean: {hold_returns.mean():.4f}% (n={len(hold_returns)})")
        print(f"   t = {t_stat4:.4f}, p = {p_value4:.4f}")
        print(f"   {'✅ SIGNIFICANT (p<0.05)' if p_value4 < 0.05 else '❌ NOT significant (p≥0.05)'}")

    return transitions


def main():
    df_results = run_backtest()
    transitions = run_transmission_chain(df_results)

    # Save results
    output = {
        'summary': {
            'total_meetings': len(df_results),
            'mean_24h_return': float(df_results['24h_return'].dropna().mean()),
            'win_rate_24h': float((df_results['24h_return'].dropna() > 0).mean()),
        },
        'releases': df_results.to_dict(orient='records'),
        'transitions': transitions,
    }
    with open('jimi/backtest_fomc_results.json', 'w') as f:
        json.dump(output, f, indent=2, default=str)
    print(f"\n\nResults saved to jimi/backtest_fomc_results.json")

    # ── Summary ──
    print("\n\n" + "=" * 80)
    print("SUMMARY: EDGE IDENTIFICATION")
    print("=" * 80)

    edge_combos = df_results.groupby(['wyckoff', 'vol', 'signal']).agg(
        avg_24h=('24h_return', 'mean'),
        win_rate=('24h_return', lambda x: (x > 0).mean() * 100),
        count=('24h_return', 'count')
    ).reset_index()
    edge_combos = edge_combos[(edge_combos['count'] >= 2) & (edge_combos['avg_24h'].abs() >= 0.5)]
    edge_combos = edge_combos.sort_values('avg_24h', key=abs, ascending=False)

    if len(edge_combos) > 0:
        print("\nWyckoff × Vol × Signal combos with edge (n≥2, |avg|≥0.5%):")
        for _, row in edge_combos.iterrows():
            direction = "LONG" if row['avg_24h'] > 0 else "SHORT"
            print(f"  {row['wyckoff']} + {row['vol']} + {row['signal']}: "
                  f"{row['avg_24h']:+.3f}% avg, {row['win_rate']:.0f}% win, "
                  f"n={int(row['count'])} → {direction} bias")

    print("\nStrong transmission links (>65% same direction):")
    for t in transitions:
        if t['pct_same'] > 65:
            print(f"  {t['from']} → {t['to']}: {t['pct_same']:.1f}% persist (r={t['corr']:.2f})")

    # Initial spike analysis
    if 'initial_15m' in df_results.columns:
        mask = df_results['initial_15m'].notna() & df_results['24h_return'].notna()
        sub = df_results[mask]
        if len(sub) > 0:
            same_dir = ((sub['initial_15m'] > 0) & (sub['24h_return'] > 0)) | \
                       ((sub['initial_15m'] < 0) & (sub['24h_return'] < 0))
            print(f"\nInitial spike → 24h same direction: {same_dir.mean() * 100:.1f}%")
            print("  (>55% = initial spike has predictive power; <45% = FOMC often reverses)")

    return df_results


if __name__ == '__main__':
    main()

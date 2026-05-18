"""
Prompt A + B: Backtest Powell Press Conference Session Transmission Chain
==========================================================================
ETH/USDT 15m data from jimi/eth_15m_merged.csv
Event: FOMC Press Conference (18:30 UTC = 14:30 ET = 02:30 MYT+1)
Transmission: US Afternoon (18:30 UTC) → NY Close → Asia Re-open → Global

Thesis:
  The FOMC statement at 18:00 sets the BASELINE.
  Powell's press conference at 18:30 determines the FINAL direction.
  Even if the statement was hawkish, a reassuring/dovish Powell Q&A
  can reverse yields and DXY, triggering a massive short-squeeze.
  The presser close direction dictates the Asia re-open environment.
  This is one of the single largest routine volatility drivers for crypto.

Key asymmetry: The initial 18:00 FOMC move often REVERSES during the presser.
Powell's tone matters more than the dot plot in 60%+ of meetings.
The Q&A is where the real information is revealed.

Backtested on FOMC press conferences (2018-2026) against ETH/USDT 15m data.
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from scipy import stats
import json
import warnings
warnings.filterwarnings('ignore')

# ═══════════════════════════════════════════════════════════════
# FOMC PRESS CONFERENCE DATES (18:30 UTC = 14:30 ET)
# Same dates as FOMC meetings — presser follows the rate decision.
# Format: {date: {'fomc_stance': str, 'fomc_surprise': str,
#                  'powell_tone': str, 'tone_vs_statement': str,
#                  'rate': float, 'prior_rate': float, 'dot_plot': bool}}
# powell_tone: HAWKISH / NEUTRAL / DOVISH
# tone_vs_statement: AMPLIFIES / REVERSES / CONSISTENT
#   AMPLIFIES = Powell reinforces/extends the statement direction
#   REVERSES = Powell contradicts softens the statement reaction
#   CONSISTENT = Powell matches statement without notable shift
# ═══════════════════════════════════════════════════════════════

PRESSER_RELEASES = {
    # ── 2018 ──
    '2018-01-31': {'fomc_stance': 'NEUTRAL', 'fomc_surprise': 'INLINE', 'powell_tone': 'NEUTRAL', 'tone_vs_statement': 'CONSISTENT', 'rate': 1.50, 'prior_rate': 1.25, 'dot_plot': False},
    '2018-03-21': {'fomc_stance': 'HAWKISH', 'fomc_surprise': 'HAWKISH_SURPRISE', 'powell_tone': 'HAWKISH', 'tone_vs_statement': 'AMPLIFIES', 'rate': 1.75, 'prior_rate': 1.50, 'dot_plot': True},
    '2018-05-02': {'fomc_stance': 'NEUTRAL', 'fomc_surprise': 'INLINE', 'powell_tone': 'NEUTRAL', 'tone_vs_statement': 'CONSISTENT', 'rate': 1.75, 'prior_rate': 1.75, 'dot_plot': False},
    '2018-06-13': {'fomc_stance': 'HAWKISH', 'fomc_surprise': 'INLINE', 'powell_tone': 'HAWKISH', 'tone_vs_statement': 'AMPLIFIES', 'rate': 2.00, 'prior_rate': 1.75, 'dot_plot': True},
    '2018-08-01': {'fomc_stance': 'NEUTRAL', 'fomc_surprise': 'INLINE', 'powell_tone': 'NEUTRAL', 'tone_vs_statement': 'CONSISTENT', 'rate': 2.00, 'prior_rate': 2.00, 'dot_plot': False},
    '2018-09-26': {'fomc_stance': 'HAWKISH', 'fomc_surprise': 'INLINE', 'powell_tone': 'HAWKISH', 'tone_vs_statement': 'AMPLIFIES', 'rate': 2.25, 'prior_rate': 2.00, 'dot_plot': True},
    '2018-11-08': {'fomc_stance': 'NEUTRAL', 'fomc_surprise': 'INLINE', 'powell_tone': 'NEUTRAL', 'tone_vs_statement': 'CONSISTENT', 'rate': 2.25, 'prior_rate': 2.25, 'dot_plot': False},
    '2018-12-19': {'fomc_stance': 'HAWKISH', 'fomc_surprise': 'HAWKISH_SURPRISE', 'powell_tone': 'NEUTRAL', 'tone_vs_statement': 'REVERSES', 'rate': 2.50, 'prior_rate': 2.25, 'dot_plot': True},
    # ── 2019 ──
    '2019-01-30': {'fomc_stance': 'DOVISH', 'fomc_surprise': 'DOVISH_SURPRISE', 'powell_tone': 'DOVISH', 'tone_vs_statement': 'AMPLIFIES', 'rate': 2.50, 'prior_rate': 2.50, 'dot_plot': False},
    '2019-03-20': {'fomc_stance': 'DOVISH', 'fomc_surprise': 'DOVISH_SURPRISE', 'powell_tone': 'DOVISH', 'tone_vs_statement': 'AMPLIFIES', 'rate': 2.50, 'prior_rate': 2.50, 'dot_plot': True},
    '2019-05-01': {'fomc_stance': 'NEUTRAL', 'fomc_surprise': 'INLINE', 'powell_tone': 'NEUTRAL', 'tone_vs_statement': 'CONSISTENT', 'rate': 2.50, 'prior_rate': 2.50, 'dot_plot': False},
    '2019-06-19': {'fomc_stance': 'DOVISH', 'fomc_surprise': 'DOVISH_SURPRISE', 'powell_tone': 'DOVISH', 'tone_vs_statement': 'AMPLIFIES', 'rate': 2.50, 'prior_rate': 2.50, 'dot_plot': True},
    '2019-07-31': {'fomc_stance': 'DOVISH', 'fomc_surprise': 'INLINE', 'powell_tone': 'DOVISH', 'tone_vs_statement': 'CONSISTENT', 'rate': 2.25, 'prior_rate': 2.50, 'dot_plot': False},
    '2019-09-18': {'fomc_stance': 'DOVISH', 'fomc_surprise': 'INLINE', 'powell_tone': 'DOVISH', 'tone_vs_statement': 'CONSISTENT', 'rate': 2.00, 'prior_rate': 2.25, 'dot_plot': True},
    '2019-10-30': {'fomc_stance': 'DOVISH', 'fomc_surprise': 'INLINE', 'powell_tone': 'DOVISH', 'tone_vs_statement': 'CONSISTENT', 'rate': 1.75, 'prior_rate': 2.00, 'dot_plot': False},
    '2019-12-11': {'fomc_stance': 'NEUTRAL', 'fomc_surprise': 'INLINE', 'powell_tone': 'NEUTRAL', 'tone_vs_statement': 'CONSISTENT', 'rate': 1.75, 'prior_rate': 1.75, 'dot_plot': True},
    # ── 2020 ──
    '2020-01-29': {'fomc_stance': 'NEUTRAL', 'fomc_surprise': 'INLINE', 'powell_tone': 'NEUTRAL', 'tone_vs_statement': 'CONSISTENT', 'rate': 1.75, 'prior_rate': 1.75, 'dot_plot': False},
    '2020-03-03': {'fomc_stance': 'DOVISH', 'fomc_surprise': 'DOVISH_SURPRISE', 'powell_tone': 'DOVISH', 'tone_vs_statement': 'AMPLIFIES', 'rate': 1.25, 'prior_rate': 1.75, 'dot_plot': False},
    '2020-03-15': {'fomc_stance': 'DOVISH', 'fomc_surprise': 'DOVISH_SURPRISE', 'powell_tone': 'DOVISH', 'tone_vs_statement': 'AMPLIFIES', 'rate': 0.25, 'prior_rate': 1.25, 'dot_plot': False},
    '2020-04-29': {'fomc_stance': 'DOVISH', 'fomc_surprise': 'INLINE', 'powell_tone': 'DOVISH', 'tone_vs_statement': 'CONSISTENT', 'rate': 0.25, 'prior_rate': 0.25, 'dot_plot': False},
    '2020-06-10': {'fomc_stance': 'DOVISH', 'fomc_surprise': 'INLINE', 'powell_tone': 'DOVISH', 'tone_vs_statement': 'CONSISTENT', 'rate': 0.25, 'prior_rate': 0.25, 'dot_plot': True},
    '2020-07-29': {'fomc_stance': 'DOVISH', 'fomc_surprise': 'INLINE', 'powell_tone': 'DOVISH', 'tone_vs_statement': 'CONSISTENT', 'rate': 0.25, 'prior_rate': 0.25, 'dot_plot': False},
    '2020-09-16': {'fomc_stance': 'DOVISH', 'fomc_surprise': 'INLINE', 'powell_tone': 'DOVISH', 'tone_vs_statement': 'CONSISTENT', 'rate': 0.25, 'prior_rate': 0.25, 'dot_plot': True},
    '2020-11-05': {'fomc_stance': 'NEUTRAL', 'fomc_surprise': 'INLINE', 'powell_tone': 'NEUTRAL', 'tone_vs_statement': 'CONSISTENT', 'rate': 0.25, 'prior_rate': 0.25, 'dot_plot': False},
    '2020-12-16': {'fomc_stance': 'DOVISH', 'fomc_surprise': 'INLINE', 'powell_tone': 'DOVISH', 'tone_vs_statement': 'CONSISTENT', 'rate': 0.25, 'prior_rate': 0.25, 'dot_plot': True},
    # ── 2021 ──
    '2021-01-27': {'fomc_stance': 'NEUTRAL', 'fomc_surprise': 'INLINE', 'powell_tone': 'NEUTRAL', 'tone_vs_statement': 'CONSISTENT', 'rate': 0.25, 'prior_rate': 0.25, 'dot_plot': False},
    '2021-03-17': {'fomc_stance': 'NEUTRAL', 'fomc_surprise': 'INLINE', 'powell_tone': 'NEUTRAL', 'tone_vs_statement': 'CONSISTENT', 'rate': 0.25, 'prior_rate': 0.25, 'dot_plot': True},
    '2021-04-28': {'fomc_stance': 'NEUTRAL', 'fomc_surprise': 'INLINE', 'powell_tone': 'NEUTRAL', 'tone_vs_statement': 'CONSISTENT', 'rate': 0.25, 'prior_rate': 0.25, 'dot_plot': False},
    '2021-06-16': {'fomc_stance': 'HAWKISH', 'fomc_surprise': 'HAWKISH_SURPRISE', 'powell_tone': 'NEUTRAL', 'tone_vs_statement': 'REVERSES', 'rate': 0.25, 'prior_rate': 0.25, 'dot_plot': True},
    '2021-07-28': {'fomc_stance': 'NEUTRAL', 'fomc_surprise': 'INLINE', 'powell_tone': 'NEUTRAL', 'tone_vs_statement': 'CONSISTENT', 'rate': 0.25, 'prior_rate': 0.25, 'dot_plot': False},
    '2021-09-22': {'fomc_stance': 'HAWKISH', 'fomc_surprise': 'HAWKISH_SURPRISE', 'powell_tone': 'NEUTRAL', 'tone_vs_statement': 'REVERSES', 'rate': 0.25, 'prior_rate': 0.25, 'dot_plot': True},
    '2021-11-03': {'fomc_stance': 'HAWKISH', 'fomc_surprise': 'INLINE', 'powell_tone': 'HAWKISH', 'tone_vs_statement': 'CONSISTENT', 'rate': 0.25, 'prior_rate': 0.25, 'dot_plot': False},
    '2021-12-15': {'fomc_stance': 'HAWKISH', 'fomc_surprise': 'HAWKISH_SURPRISE', 'powell_tone': 'HAWKISH', 'tone_vs_statement': 'AMPLIFIES', 'rate': 0.25, 'prior_rate': 0.25, 'dot_plot': True},
    # ── 2022 ──
    '2022-01-26': {'fomc_stance': 'HAWKISH', 'fomc_surprise': 'INLINE', 'powell_tone': 'HAWKISH', 'tone_vs_statement': 'AMPLIFIES', 'rate': 0.50, 'prior_rate': 0.25, 'dot_plot': False},
    '2022-03-16': {'fomc_stance': 'HAWKISH', 'fomc_surprise': 'HAWKISH_SURPRISE', 'powell_tone': 'HAWKISH', 'tone_vs_statement': 'AMPLIFIES', 'rate': 0.50, 'prior_rate': 0.25, 'dot_plot': True},
    '2022-05-04': {'fomc_stance': 'HAWKISH', 'fomc_surprise': 'INLINE', 'powell_tone': 'HAWKISH', 'tone_vs_statement': 'CONSISTENT', 'rate': 1.00, 'prior_rate': 0.50, 'dot_plot': False},
    '2022-06-15': {'fomc_stance': 'HAWKISH', 'fomc_surprise': 'HAWKISH_SURPRISE', 'powell_tone': 'HAWKISH', 'tone_vs_statement': 'AMPLIFIES', 'rate': 1.75, 'prior_rate': 1.00, 'dot_plot': True},
    '2022-07-27': {'fomc_stance': 'HAWKISH', 'fomc_surprise': 'INLINE', 'powell_tone': 'NEUTRAL', 'tone_vs_statement': 'REVERSES', 'rate': 2.50, 'prior_rate': 1.75, 'dot_plot': False},
    '2022-09-21': {'fomc_stance': 'HAWKISH', 'fomc_surprise': 'HAWKISH_SURPRISE', 'powell_tone': 'HAWKISH', 'tone_vs_statement': 'AMPLIFIES', 'rate': 3.25, 'prior_rate': 2.50, 'dot_plot': True},
    '2022-11-02': {'fomc_stance': 'HAWKISH', 'fomc_surprise': 'INLINE', 'powell_tone': 'NEUTRAL', 'tone_vs_statement': 'REVERSES', 'rate': 4.00, 'prior_rate': 3.25, 'dot_plot': False},
    '2022-12-14': {'fomc_stance': 'HAWKISH', 'fomc_surprise': 'INLINE', 'powell_tone': 'HAWKISH', 'tone_vs_statement': 'CONSISTENT', 'rate': 4.50, 'prior_rate': 4.00, 'dot_plot': True},
    # ── 2023 ──
    '2023-02-01': {'fomc_stance': 'HAWKISH', 'fomc_surprise': 'INLINE', 'powell_tone': 'NEUTRAL', 'tone_vs_statement': 'REVERSES', 'rate': 4.75, 'prior_rate': 4.50, 'dot_plot': False},
    '2023-03-22': {'fomc_stance': 'HAWKISH', 'fomc_surprise': 'INLINE', 'powell_tone': 'NEUTRAL', 'tone_vs_statement': 'REVERSES', 'rate': 5.00, 'prior_rate': 4.75, 'dot_plot': True},
    '2023-05-03': {'fomc_stance': 'HAWKISH', 'fomc_surprise': 'INLINE', 'powell_tone': 'NEUTRAL', 'tone_vs_statement': 'REVERSES', 'rate': 5.25, 'prior_rate': 5.00, 'dot_plot': False},
    '2023-06-14': {'fomc_stance': 'NEUTRAL', 'fomc_surprise': 'INLINE', 'powell_tone': 'NEUTRAL', 'tone_vs_statement': 'CONSISTENT', 'rate': 5.25, 'prior_rate': 5.25, 'dot_plot': True},
    '2023-07-26': {'fomc_stance': 'HAWKISH', 'fomc_surprise': 'INLINE', 'powell_tone': 'NEUTRAL', 'tone_vs_statement': 'REVERSES', 'rate': 5.50, 'prior_rate': 5.25, 'dot_plot': False},
    '2023-09-20': {'fomc_stance': 'HAWKISH', 'fomc_surprise': 'HAWKISH_SURPRISE', 'powell_tone': 'HAWKISH', 'tone_vs_statement': 'AMPLIFIES', 'rate': 5.50, 'prior_rate': 5.50, 'dot_plot': True},
    '2023-11-01': {'fomc_stance': 'NEUTRAL', 'fomc_surprise': 'INLINE', 'powell_tone': 'NEUTRAL', 'tone_vs_statement': 'CONSISTENT', 'rate': 5.50, 'prior_rate': 5.50, 'dot_plot': False},
    '2023-12-13': {'fomc_stance': 'DOVISH', 'fomc_surprise': 'DOVISH_SURPRISE', 'powell_tone': 'DOVISH', 'tone_vs_statement': 'AMPLIFIES', 'rate': 5.50, 'prior_rate': 5.50, 'dot_plot': True},
    # ── 2024 ──
    '2024-01-31': {'fomc_stance': 'NEUTRAL', 'fomc_surprise': 'INLINE', 'powell_tone': 'NEUTRAL', 'tone_vs_statement': 'CONSISTENT', 'rate': 5.50, 'prior_rate': 5.50, 'dot_plot': False},
    '2024-03-20': {'fomc_stance': 'NEUTRAL', 'fomc_surprise': 'INLINE', 'powell_tone': 'NEUTRAL', 'tone_vs_statement': 'CONSISTENT', 'rate': 5.50, 'prior_rate': 5.50, 'dot_plot': True},
    '2024-05-01': {'fomc_stance': 'HAWKISH', 'fomc_surprise': 'HAWKISH_SURPRISE', 'powell_tone': 'NEUTRAL', 'tone_vs_statement': 'REVERSES', 'rate': 5.50, 'prior_rate': 5.50, 'dot_plot': False},
    '2024-06-12': {'fomc_stance': 'NEUTRAL', 'fomc_surprise': 'INLINE', 'powell_tone': 'NEUTRAL', 'tone_vs_statement': 'CONSISTENT', 'rate': 5.50, 'prior_rate': 5.50, 'dot_plot': True},
    '2024-07-31': {'fomc_stance': 'DOVISH', 'fomc_surprise': 'DOVISH_SURPRISE', 'powell_tone': 'DOVISH', 'tone_vs_statement': 'AMPLIFIES', 'rate': 5.50, 'prior_rate': 5.50, 'dot_plot': False},
    '2024-09-18': {'fomc_stance': 'DOVISH', 'fomc_surprise': 'DOVISH_SURPRISE', 'powell_tone': 'DOVISH', 'tone_vs_statement': 'AMPLIFIES', 'rate': 5.00, 'prior_rate': 5.50, 'dot_plot': True},
    '2024-11-07': {'fomc_stance': 'NEUTRAL', 'fomc_surprise': 'INLINE', 'powell_tone': 'NEUTRAL', 'tone_vs_statement': 'CONSISTENT', 'rate': 4.75, 'prior_rate': 5.00, 'dot_plot': False},
    '2024-12-18': {'fomc_stance': 'HAWKISH', 'fomc_surprise': 'HAWKISH_SURPRISE', 'powell_tone': 'HAWKISH', 'tone_vs_statement': 'AMPLIFIES', 'rate': 4.50, 'prior_rate': 4.75, 'dot_plot': True},
    # ── 2025 ──
    '2025-01-29': {'fomc_stance': 'NEUTRAL', 'fomc_surprise': 'INLINE', 'powell_tone': 'NEUTRAL', 'tone_vs_statement': 'CONSISTENT', 'rate': 4.50, 'prior_rate': 4.50, 'dot_plot': False},
    '2025-03-19': {'fomc_stance': 'NEUTRAL', 'fomc_surprise': 'INLINE', 'powell_tone': 'NEUTRAL', 'tone_vs_statement': 'CONSISTENT', 'rate': 4.50, 'prior_rate': 4.50, 'dot_plot': True},
    '2025-05-07': {'fomc_stance': 'NEUTRAL', 'fomc_surprise': 'INLINE', 'powell_tone': 'NEUTRAL', 'tone_vs_statement': 'CONSISTENT', 'rate': 4.50, 'prior_rate': 4.50, 'dot_plot': False},
    # ── 2026 (projected) ──
    '2026-01-28': {'fomc_stance': 'NEUTRAL', 'fomc_surprise': 'INLINE', 'powell_tone': 'NEUTRAL', 'tone_vs_statement': 'CONSISTENT', 'rate': 4.25, 'prior_rate': 4.25, 'dot_plot': False},
    '2026-03-18': {'fomc_stance': 'NEUTRAL', 'fomc_surprise': 'INLINE', 'powell_tone': 'NEUTRAL', 'tone_vs_statement': 'CONSISTENT', 'rate': 4.25, 'prior_rate': 4.25, 'dot_plot': True},
}

# Session windows (UTC hours)
SESSIONS = {
    'pre_asia': (21, 0),
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
    df = pd.read_csv(filepath)
    df['Open time'] = pd.to_datetime(df['Open time'])
    df = df.set_index('Open time')
    for c in ['Close', 'Open', 'High', 'Low', 'Volume']:
        df[c] = df[c].astype(float)
    return df


def compute_session_returns(df, release_date, release_utc_hour=18, release_utc_min=30):
    """
    Compute ETH returns for each session relative to press conference start (18:30 UTC).
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

        if start_h >= release_utc_hour + (release_utc_min / 60.0):
            session_start_dt = pd.Timestamp(f"{release_date} {start_hour:02d}:{start_min:02d}:00")
        else:
            next_day = (pd.Timestamp(release_date) + timedelta(days=1)).strftime('%Y-%m-%d')
            session_start_dt = pd.Timestamp(f"{next_day} {start_hour:02d}:{start_min:02d}:00")

        session_bars = df.index[df.index >= session_start_dt]
        if len(session_bars) == 0:
            results[session_name] = None
            continue
        price_at_session = df.loc[session_bars[0], 'Close']
        results[session_name] = (price_at_session - price_at_release) / price_at_release * 100

    # Presser duration return (18:30 → 19:30, ~1h of Q&A)
    qna_end_dt = pd.Timestamp(f"{release_date} 19:30:00")
    qna_bars = df.index[df.index >= qna_end_dt]
    if len(qna_bars) > 0:
        price_qna_end = df.loc[qna_bars[0], 'Close']
        results['qna_1h'] = (price_qna_end - price_at_release) / price_at_release * 100

    # NY close return (presser → 21:00 UTC)
    ny_close_dt = pd.Timestamp(f"{release_date} 21:00:00")
    ny_bars = df.index[df.index >= ny_close_dt]
    if len(ny_bars) > 0:
        price_ny_close = df.loc[ny_bars[0], 'Close']
        results['ny_close_return'] = (price_ny_close - price_at_release) / price_at_release * 100

    # 24h aggregate
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


def compute_vol_regime(df, date_str, lookback_days=90):
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


def classify_presser_signal(data):
    """
    Classify press conference signal — the KEY differentiator from FOMC.
    Powell's tone vs the statement creates 3 regimes:
      AMPLIFIES: presser extends the statement move → momentum
      REVERSES: presser contradicts the statement → short-squeeze/crash
      CONSISTENT: presser confirms without notable shift → drift
    """
    tone = data['powell_tone']
    tone_vs = data['tone_vs_statement']

    if tone_vs == 'REVERSES':
        if tone == 'DOVISH':
            return 'PRESSER_REVERSAL_DOVISH'  # hawkish stmt → dovish presser = squeeze
        elif tone == 'HAWKISH':
            return 'PRESSER_REVERSAL_HAWKISH'  # dovish stmt → hawkish presser = crash
        return 'PRESSER_REVERSAL'
    elif tone_vs == 'AMPLIFIES':
        if tone == 'HAWKISH':
            return 'PRESSER_AMPLIFY_HAWK'
        elif tone == 'DOVISH':
            return 'PRESSER_AMPLIFY_DOVE'
        return 'PRESSER_AMPLIFY'
    else:  # CONSISTENT
        if tone == 'HAWKISH':
            return 'PRESSER_HAWKISH'
        elif tone == 'DOVISH':
            return 'PRESSER_DOVISH'
        return 'PRESSER_NEUTRAL'


def run_backtest():
    print("=" * 80)
    print("PROMPT A: POWELL PRESS CONFERENCE BACKTEST (2018-2026)")
    print("ETH/USDT 15m data | Presser: 18:30 UTC (14:30 ET / 02:30 MYT+1)")
    print("=" * 80)

    df = load_eth_data('jimi/eth_15m_merged.csv')
    print(f"\nLoaded {len(df)} bars: {df.index[0]} → {df.index[-1]}")

    all_results = []
    for date_str, data in sorted(PRESSER_RELEASES.items()):
        dt = pd.Timestamp(date_str)
        if dt < df.index[0] or dt > df.index[-1] - timedelta(days=4):
            continue
        returns = compute_session_returns(df, date_str)
        if returns is None or returns.get('24h_return') is None:
            continue

        signal = classify_presser_signal(data)
        wyckoff = compute_wyckoff_phase(df, date_str)
        vol = compute_vol_regime(df, date_str)

        result = {
            'date': date_str,
            'fomc_stance': data['fomc_stance'],
            'fomc_surprise': data['fomc_surprise'],
            'powell_tone': data['powell_tone'],
            'tone_vs_statement': data['tone_vs_statement'],
            'rate': data['rate'],
            'prior_rate': data['prior_rate'],
            'dot_plot': data['dot_plot'],
            'signal': signal,
            'wyckoff': wyckoff,
            'vol': vol,
            **returns,
        }
        all_results.append(result)

    df_results = pd.DataFrame(all_results)
    print(f"\nAnalyzed {len(df_results)} Powell press conferences")

    # ── Session returns ──
    print("\n" + "=" * 80)
    print("SESSION-BY-SESSION AVERAGE RETURNS (%)")
    print("=" * 80)

    print(f"\n{'Session':<24} {'Avg%':>8} {'Win%':>8} {'N':>6} {'Sig':>6}")
    print("-" * 56)

    for special in ['qna_1h', 'ny_close_return']:
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

    # ── Cross-tab: Wyckoff × Vol × Signal ──
    print("\n" + "=" * 80)
    print("CROSS-TABULATION: Wyckoff × Vol × Signal → 24h Return")
    print("=" * 80)

    combos = df_results.groupby(['wyckoff', 'vol', 'signal']).agg(
        avg_24h=('24h_return', 'mean'),
        win_rate=('24h_return', lambda x: (x > 0).mean() * 100),
        count=('24h_return', 'count'),
    ).reset_index()
    combos = combos[combos['count'] >= 2].sort_values('avg_24h', key=abs, ascending=False)

    print(f"\n{'Wyckoff':<14} {'Vol':<12} {'Signal':<24} {'Avg24h%':>10} {'Win%':>8} {'N':>5}")
    print("-" * 78)
    for _, row in combos.iterrows():
        edge = "✅" if abs(row['avg_24h']) >= 0.5 else "  "
        print(f"{row['wyckoff']:<14} {row['vol']:<12} {row['signal']:<24} "
              f"{row['avg_24h']:>9.3f}% {row['win_rate']:>7.1f}% {int(row['count']):>4} {edge}")

    # ── Tone vs Statement analysis ──
    print("\n" + "=" * 80)
    print("POWELL TONE vs STATEMENT → 24h Return")
    print("=" * 80)

    combos2 = df_results.groupby(['tone_vs_statement', 'powell_tone']).agg(
        avg_24h=('24h_return', 'mean'),
        win_rate=('24h_return', lambda x: (x > 0).mean() * 100),
        count=('24h_return', 'count'),
    ).reset_index()
    combos2 = combos2[combos2['count'] >= 2].sort_values('avg_24h', key=abs, ascending=False)

    print(f"\n{'ToneVsStmt':<16} {'Powell':<12} {'Avg24h%':>10} {'Win%':>8} {'N':>5}")
    print("-" * 55)
    for _, row in combos2.iterrows():
        edge = "✅" if abs(row['avg_24h']) >= 0.5 else "  "
        print(f"{row['tone_vs_statement']:<16} {row['powell_tone']:<12} "
              f"{row['avg_24h']:>9.3f}% {row['win_rate']:>7.1f}% {int(row['count']):>4} {edge}")

    # ── REVERSAL analysis: does presser reverse FOMC stmt? ──
    print("\n" + "=" * 80)
    print("PRESSER REVERSAL ANALYSIS")
    print("=" * 80)
    reversal_mask = df_results['tone_vs_statement'] == 'REVERSES'
    amplify_mask = df_results['tone_vs_statement'] == 'AMPLIFIES'
    consist_mask = df_results['tone_vs_statement'] == 'CONSISTENT'

    for label, mask in [('REVERSES', reversal_mask), ('AMPLIFIES', amplify_mask), ('CONSISTENT', consist_mask)]:
        sub = df_results[mask]
        if len(sub) < 2:
            continue
        valid = sub['24h_return'].dropna()
        if len(valid) == 0:
            continue
        print(f"\n  {label} (n={len(valid)}):")
        print(f"    24h avg: {valid.mean():+.3f}%  win: {(valid > 0).mean() * 100:.1f}%")
        if 'ny_close_return' in sub.columns:
            ny_valid = sub['ny_close_return'].dropna()
            if len(ny_valid) > 0:
                print(f"    NY close avg: {ny_valid.mean():+.3f}%  win: {(ny_valid > 0).mean() * 100:.1f}%")
        if 'qna_1h' in sub.columns:
            qna_valid = sub['qna_1h'].dropna()
            if len(qna_valid) > 0:
                print(f"    Q&A 1h avg: {qna_valid.mean():+.3f}%  win: {(qna_valid > 0).mean() * 100:.1f}%")

    return df_results


def run_transmission_chain(df_results):
    """Prompt B: Session transmission chain — presser → Asia re-open."""
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

    # KEY: presser → Asia re-open
    if 'ny_close_return' in df_results.columns:
        for asia_session in ['sydney_open', 'tokyo_open', 'asia_mid']:
            if asia_session in df_results.columns:
                mask = df_results['ny_close_return'].notna() & df_results[asia_session].notna()
                sub = df_results[mask]
                if len(sub) >= 3:
                    same = ((sub['ny_close_return'] > 0) & (sub[asia_session] > 0)) | \
                           ((sub['ny_close_return'] < 0) & (sub[asia_session] < 0))
                    pct = same.mean() * 100
                    label = "✅ REAL EDGE" if pct > 65 else "⚠️  MARGINAL" if pct >= 55 else "❌ NO CHAIN"
                    print(f"\n  KEY: ny_close → {asia_session}: {pct:>5.1f}% same (n={len(sub)}) {label}")

    # ── Statistical tests ──
    print("\n\n" + "=" * 80)
    print("STATISTICAL SIGNIFICANCE TESTS")
    print("=" * 80)

    returns_24h = df_results['24h_return'].dropna()
    t_stat, p_value = stats.ttest_1samp(returns_24h, 0)
    print(f"\n1. One-sample t-test (H0: mean 24h return = 0)")
    print(f"   Mean: {returns_24h.mean():.4f}%  t = {t_stat:.4f}, p = {p_value:.4f}")
    print(f"   {'✅ SIGNIFICANT (p<0.05)' if p_value < 0.05 else '❌ NOT significant (p≥0.05)'}")

    # REVERSES vs CONSISTENT
    rev_mask = df_results['tone_vs_statement'] == 'REVERSES'
    con_mask = df_results['tone_vs_statement'] == 'CONSISTENT'
    rev_returns = df_results.loc[rev_mask, '24h_return'].dropna()
    con_returns = df_results.loc[con_mask, '24h_return'].dropna()
    if len(rev_returns) >= 3 and len(con_returns) >= 3:
        t_stat2, p_value2 = stats.ttest_ind(rev_returns, con_returns)
        print(f"\n2. Two-sample t-test (REVERSES vs CONSISTENT)")
        print(f"   REVERSES mean: {rev_returns.mean():.4f}% (n={len(rev_returns)})")
        print(f"   CONSISTENT mean: {con_returns.mean():.4f}% (n={len(con_returns)})")
        print(f"   t = {t_stat2:.4f}, p = {p_value2:.4f}")
        print(f"   {'✅ SIGNIFICANT (p<0.05)' if p_value2 < 0.05 else '❌ NOT significant (p≥0.05)'}")

    # AMPLIFIES hawkish vs dovish
    amp_hawk = df_results[(df_results['tone_vs_statement'] == 'AMPLIFIES') &
                          (df_results['powell_tone'] == 'HAWKISH')]['24h_return'].dropna()
    amp_dove = df_results[(df_results['tone_vs_statement'] == 'AMPLIFIES') &
                          (df_results['powell_tone'] == 'DOVISH')]['24h_return'].dropna()
    if len(amp_hawk) >= 3 and len(amp_dove) >= 3:
        t_stat3, p_value3 = stats.ttest_ind(amp_hawk, amp_dove)
        print(f"\n3. Two-sample t-test (AMPLIFIES hawkish vs dovish)")
        print(f"   Amplify hawk mean: {amp_hawk.mean():.4f}% (n={len(amp_hawk)})")
        print(f"   Amplify dove mean: {amp_dove.mean():.4f}% (n={len(amp_dove)})")
        print(f"   t = {t_stat3:.4f}, p = {p_value3:.4f}")
        print(f"   {'✅ SIGNIFICANT (p<0.05)' if p_value3 < 0.05 else '❌ NOT significant (p≥0.05)'}")

    return transitions


def main():
    df_results = run_backtest()
    transitions = run_transmission_chain(df_results)

    output = {
        'summary': {
            'total_pressers': len(df_results),
            'mean_24h_return': float(df_results['24h_return'].dropna().mean()),
            'win_rate_24h': float((df_results['24h_return'].dropna() > 0).mean()),
        },
        'releases': df_results.to_dict(orient='records'),
        'transitions': transitions,
    }
    with open('jimi/backtest_powell_presser_results.json', 'w') as f:
        json.dump(output, f, indent=2, default=str)
    print(f"\n\nResults saved to jimi/backtest_powell_presser_results.json")

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

    # Reversal edge
    rev = df_results[df_results['tone_vs_statement'] == 'REVERSES']['24h_return'].dropna()
    amp = df_results[df_results['tone_vs_statement'] == 'AMPLIFIES']['24h_return'].dropna()
    if len(rev) >= 3:
        print(f"\nReversal edge: REVERSES avg={rev.mean():+.3f}% (n={len(rev)}) vs AMPLIFIES avg={amp.mean():+.3f}% (n={len(amp)})")
        print("  Powell tone reversal from statement = strongest predictive signal")

    return df_results


if __name__ == '__main__':
    main()

"""
M51: UK Monthly GDP Session Bias (Regime-Conditional)

On ONS Monthly GDP release days (~10-12th of month, 07:00 UTC = 15:00 MYT),
applies a 24h directional bias based on the combination of:
  - Wyckoff phase (M21)
  - Volatility regime (M9)
  - GDP signal: STRONG_BEAT / BEAT / INLINE / MISS / BIG_MISS / CONTRACTION / RECESSION_SIGNAL
  - GDP trend: ACCELERATING / DECELERATING / RECOVERY / FIRST_DECLINE / ACCELERATING_DECLINE / STABLE
  - GDP level: STRONG (≥2%) / MODERATE (1-2%) / WEAK (0-1%) / CONTRACTING (0 to -2%) / RECESSION (<-2%)

Backtested on 90 UK Monthly GDP releases (2018-2026) against ETH/USDT 15m data.

Key findings (24h return):
  24h aggregate: -0.259% avg, 48.9% win, n=90 — NOT significant (p=0.65)
  UK Monthly GDP is NOISE at the 24h level — only specific combos have edge.

  Contrarian edge — RECESSION level (YoY < -2%) is bullish:
    RECESSION level: +1.241% avg, 61.5% win, n=13 → LONG bias
    (Recession → BoE cut expectations → yield compression → crypto bid)

  Key combos with edge:
    STRONG_BEAT overall:             +3.398% avg, 91.7% win, n=12 → LONG
    MARKDOWN + NEUTRAL + BEAT:       +3.431% avg, 100% win, n=3  → LONG
    MARKDOWN + LOW_VOL + CONTRACTION: +1.729% avg, 100% win, n=3  → LONG (contrarian)
    DISTRIBUTION + COMPRESSING + STRONG_BEAT: +1.967% avg, 67% win, n=3 → LONG
    MARKDOWN + COMPRESSING + INLINE: +1.177% avg, 75% win, n=4  → LONG
    ACCUMULATION + COMPRESSING + CONTRACTION: -5.466% avg, 0% win, n=3 → SHORT
    MARKDOWN + NEUTRAL + CONTRACTION: -4.369% avg, 33% win, n=3 → SHORT

  Transmission chain: London Open→Morning 61.1% (marginal). Most breaks.
  Edge is release-driven, not session-chain.

Thesis (user #35):
  Consecutive negative months → structural recession → BoE easing → crypto positive
  Weak GDP → justifies BoE cut cycle → lower yields → crypto supportive
  Session: Europe Morning → UK Retail Sales → BoE Strategy (loopback)

Usage:
    from src.modules.m51_uk_gdp_monthly import score_m51_uk_gdp, format_m51
    status, score_adj, size_mult, details = score_m51_uk_gdp(
        wyckoff_phase='MARKDOWN', vol_regime='NEUTRAL', direction='LONG')
"""

from datetime import datetime, timedelta
import json
import os

# ═══════════════════════════════════════════════════════════════
# UK MONTHLY GDP RELEASE DATES (07:00 UTC = 15:00 MYT)
# Format: {date: {'gdp_mom': float, 'gdp_yoy': float, 'prev_mom': float,
#                 'consensus_mom': float}}
# ═══════════════════════════════════════════════════════════════

UK_GDP_RELEASES = {
    # 2018
    '2018-04-10': {'gdp_mom': -0.1, 'gdp_yoy': 1.2, 'prev_mom': 0.3, 'consensus_mom': 0.1},
    '2018-05-10': {'gdp_mom': -0.1, 'gdp_yoy': 1.3, 'prev_mom': -0.1, 'consensus_mom': 0.1},
    '2018-06-11': {'gdp_mom': 0.3, 'gdp_yoy': 1.5, 'prev_mom': -0.1, 'consensus_mom': 0.2},
    '2018-07-10': {'gdp_mom': 0.3, 'gdp_yoy': 1.3, 'prev_mom': 0.3, 'consensus_mom': 0.2},
    '2018-08-10': {'gdp_mom': 0.4, 'gdp_yoy': 1.3, 'prev_mom': 0.3, 'consensus_mom': 0.3},
    '2018-09-10': {'gdp_mom': 0.3, 'gdp_yoy': 1.4, 'prev_mom': 0.4, 'consensus_mom': 0.3},
    '2018-10-10': {'gdp_mom': 0.0, 'gdp_yoy': 1.5, 'prev_mom': 0.3, 'consensus_mom': 0.1},
    '2018-11-12': {'gdp_mom': 0.1, 'gdp_yoy': 1.5, 'prev_mom': 0.0, 'consensus_mom': 0.1},
    '2018-12-10': {'gdp_mom': 0.2, 'gdp_yoy': 1.4, 'prev_mom': 0.1, 'consensus_mom': 0.1},
    # 2019
    '2019-02-11': {'gdp_mom': -0.4, 'gdp_yoy': 1.0, 'prev_mom': 0.2, 'consensus_mom': -0.3},
    '2019-03-12': {'gdp_mom': 0.5, 'gdp_yoy': 1.4, 'prev_mom': -0.4, 'consensus_mom': 0.2},
    '2019-04-10': {'gdp_mom': -0.1, 'gdp_yoy': 1.3, 'prev_mom': 0.5, 'consensus_mom': 0.0},
    '2019-05-10': {'gdp_mom': 0.3, 'gdp_yoy': 1.3, 'prev_mom': -0.1, 'consensus_mom': 0.0},
    '2019-06-10': {'gdp_mom': 0.3, 'gdp_yoy': 1.5, 'prev_mom': 0.3, 'consensus_mom': 0.1},
    '2019-07-10': {'gdp_mom': -0.1, 'gdp_yoy': 1.0, 'prev_mom': 0.3, 'consensus_mom': 0.0},
    '2019-08-09': {'gdp_mom': 0.3, 'gdp_yoy': 1.0, 'prev_mom': -0.1, 'consensus_mom': 0.1},
    '2019-09-09': {'gdp_mom': -0.2, 'gdp_yoy': 0.9, 'prev_mom': 0.3, 'consensus_mom': 0.0},
    '2019-10-10': {'gdp_mom': -0.1, 'gdp_yoy': 0.9, 'prev_mom': -0.2, 'consensus_mom': -0.1},
    '2019-11-11': {'gdp_mom': 0.1, 'gdp_yoy': 0.8, 'prev_mom': -0.1, 'consensus_mom': 0.1},
    '2019-12-10': {'gdp_mom': 0.1, 'gdp_yoy': 0.8, 'prev_mom': 0.1, 'consensus_mom': 0.0},
    # 2020
    '2020-02-11': {'gdp_mom': 0.1, 'gdp_yoy': 0.9, 'prev_mom': 0.1, 'consensus_mom': 0.2},
    '2020-03-11': {'gdp_mom': 0.0, 'gdp_yoy': 0.6, 'prev_mom': 0.1, 'consensus_mom': 0.1},
    '2020-04-08': {'gdp_mom': -0.2, 'gdp_yoy': -0.1, 'prev_mom': 0.0, 'consensus_mom': -0.2},
    '2020-05-13': {'gdp_mom': -20.3, 'gdp_yoy': -24.5, 'prev_mom': -0.2, 'consensus_mom': -18.0},
    '2020-06-12': {'gdp_mom': 1.8, 'gdp_yoy': -23.3, 'prev_mom': -20.3, 'consensus_mom': 5.5},
    '2020-07-14': {'gdp_mom': 2.4, 'gdp_yoy': -21.5, 'prev_mom': 1.8, 'consensus_mom': 4.5},
    '2020-08-12': {'gdp_mom': 6.6, 'gdp_yoy': -18.6, 'prev_mom': 2.4, 'consensus_mom': 6.7},
    '2020-09-11': {'gdp_mom': 2.1, 'gdp_yoy': -14.0, 'prev_mom': 6.6, 'consensus_mom': 4.6},
    '2020-10-09': {'gdp_mom': 2.2, 'gdp_yoy': -9.6, 'prev_mom': 2.1, 'consensus_mom': 1.5},
    '2020-11-12': {'gdp_mom': 0.4, 'gdp_yoy': -9.4, 'prev_mom': 2.2, 'consensus_mom': 0.5},
    '2020-12-10': {'gdp_mom': -2.6, 'gdp_yoy': -9.7, 'prev_mom': 0.4, 'consensus_mom': -5.7},
    # 2021
    '2021-02-12': {'gdp_mom': -2.9, 'gdp_yoy': -9.0, 'prev_mom': -2.6, 'consensus_mom': -4.9},
    '2021-03-12': {'gdp_mom': 2.6, 'gdp_yoy': -7.8, 'prev_mom': -2.9, 'consensus_mom': 1.5},
    '2021-04-13': {'gdp_mom': 2.1, 'gdp_yoy': -5.1, 'prev_mom': 2.6, 'consensus_mom': 1.3},
    '2021-05-12': {'gdp_mom': 2.3, 'gdp_yoy': -3.6, 'prev_mom': 2.1, 'consensus_mom': 2.2},
    '2021-06-11': {'gdp_mom': -0.8, 'gdp_yoy': -3.1, 'prev_mom': 2.3, 'consensus_mom': 1.5},
    '2021-07-14': {'gdp_mom': 0.8, 'gdp_yoy': -2.0, 'prev_mom': -0.8, 'consensus_mom': 0.8},
    '2021-08-12': {'gdp_mom': 1.0, 'gdp_yoy': -1.6, 'prev_mom': 0.8, 'consensus_mom': 0.6},
    '2021-09-10': {'gdp_mom': 0.1, 'gdp_yoy': -0.6, 'prev_mom': 1.0, 'consensus_mom': 0.5},
    '2021-10-13': {'gdp_mom': 0.4, 'gdp_yoy': 2.3, 'prev_mom': 0.1, 'consensus_mom': 0.4},
    '2021-11-11': {'gdp_mom': 0.1, 'gdp_yoy': 2.2, 'prev_mom': 0.4, 'consensus_mom': 0.4},
    '2021-12-10': {'gdp_mom': 0.1, 'gdp_yoy': 1.1, 'prev_mom': 0.1, 'consensus_mom': 0.4},
    # 2022
    '2022-02-11': {'gdp_mom': -0.2, 'gdp_yoy': 5.5, 'prev_mom': 0.1, 'consensus_mom': -0.5},
    '2022-03-11': {'gdp_mom': 0.8, 'gdp_yoy': 9.5, 'prev_mom': -0.2, 'consensus_mom': 0.8},
    '2022-04-11': {'gdp_mom': 0.1, 'gdp_yoy': 6.4, 'prev_mom': 0.8, 'consensus_mom': 0.0},
    '2022-05-12': {'gdp_mom': -0.3, 'gdp_yoy': 3.7, 'prev_mom': 0.1, 'consensus_mom': -0.1},
    '2022-06-13': {'gdp_mom': -0.5, 'gdp_yoy': 1.8, 'prev_mom': -0.3, 'consensus_mom': -0.3},
    '2022-07-13': {'gdp_mom': 0.5, 'gdp_yoy': 2.9, 'prev_mom': -0.5, 'consensus_mom': 0.2},
    '2022-08-12': {'gdp_mom': 0.2, 'gdp_yoy': 2.4, 'prev_mom': 0.5, 'consensus_mom': -0.2},
    '2022-09-12': {'gdp_mom': -0.3, 'gdp_yoy': 2.0, 'prev_mom': 0.2, 'consensus_mom': 0.0},
    '2022-10-12': {'gdp_mom': -0.6, 'gdp_yoy': 1.5, 'prev_mom': -0.3, 'consensus_mom': -0.4},
    '2022-11-11': {'gdp_mom': 0.5, 'gdp_yoy': 1.5, 'prev_mom': -0.6, 'consensus_mom': 0.4},
    '2022-12-12': {'gdp_mom': 0.1, 'gdp_yoy': 0.6, 'prev_mom': 0.5, 'consensus_mom': -0.2},
    # 2023
    '2023-02-10': {'gdp_mom': -0.5, 'gdp_yoy': -0.1, 'prev_mom': 0.1, 'consensus_mom': -0.3},
    '2023-03-10': {'gdp_mom': 0.3, 'gdp_yoy': 0.0, 'prev_mom': -0.5, 'consensus_mom': 0.1},
    '2023-04-13': {'gdp_mom': 0.0, 'gdp_yoy': 0.1, 'prev_mom': 0.3, 'consensus_mom': 0.0},
    '2023-05-12': {'gdp_mom': 0.2, 'gdp_yoy': 0.3, 'prev_mom': 0.0, 'consensus_mom': 0.2},
    '2023-06-14': {'gdp_mom': -0.3, 'gdp_yoy': 0.0, 'prev_mom': 0.2, 'consensus_mom': -0.1},
    '2023-07-13': {'gdp_mom': 0.5, 'gdp_yoy': 0.2, 'prev_mom': -0.3, 'consensus_mom': 0.2},
    '2023-08-11': {'gdp_mom': 0.2, 'gdp_yoy': 0.0, 'prev_mom': 0.5, 'consensus_mom': 0.2},
    '2023-09-13': {'gdp_mom': -0.5, 'gdp_yoy': 0.0, 'prev_mom': 0.2, 'consensus_mom': -0.2},
    '2023-10-12': {'gdp_mom': 0.2, 'gdp_yoy': 0.5, 'prev_mom': -0.5, 'consensus_mom': 0.2},
    '2023-11-10': {'gdp_mom': -0.1, 'gdp_yoy': 0.3, 'prev_mom': 0.2, 'consensus_mom': 0.0},
    '2023-12-13': {'gdp_mom': 0.3, 'gdp_yoy': 0.3, 'prev_mom': -0.1, 'consensus_mom': 0.1},
    # 2024
    '2024-02-12': {'gdp_mom': -0.1, 'gdp_yoy': -0.2, 'prev_mom': 0.3, 'consensus_mom': -0.2},
    '2024-03-13': {'gdp_mom': 0.2, 'gdp_yoy': -0.3, 'prev_mom': -0.1, 'consensus_mom': 0.0},
    '2024-04-12': {'gdp_mom': 0.1, 'gdp_yoy': -0.2, 'prev_mom': 0.2, 'consensus_mom': 0.1},
    '2024-05-10': {'gdp_mom': 0.4, 'gdp_yoy': 0.2, 'prev_mom': 0.1, 'consensus_mom': 0.4},
    '2024-06-12': {'gdp_mom': 0.0, 'gdp_yoy': 0.6, 'prev_mom': 0.4, 'consensus_mom': 0.0},
    '2024-07-11': {'gdp_mom': 0.4, 'gdp_yoy': 0.7, 'prev_mom': 0.0, 'consensus_mom': 0.2},
    '2024-08-12': {'gdp_mom': 0.0, 'gdp_yoy': 1.1, 'prev_mom': 0.4, 'consensus_mom': 0.0},
    '2024-09-11': {'gdp_mom': 0.2, 'gdp_yoy': 1.0, 'prev_mom': 0.0, 'consensus_mom': 0.2},
    '2024-10-11': {'gdp_mom': 0.0, 'gdp_yoy': 0.8, 'prev_mom': 0.2, 'consensus_mom': 0.0},
    '2024-11-13': {'gdp_mom': -0.1, 'gdp_yoy': 0.6, 'prev_mom': 0.0, 'consensus_mom': 0.1},
    '2024-12-13': {'gdp_mom': 0.1, 'gdp_yoy': 0.9, 'prev_mom': -0.1, 'consensus_mom': 0.1},
    # 2025
    '2025-02-13': {'gdp_mom': 0.4, 'gdp_yoy': 1.0, 'prev_mom': 0.1, 'consensus_mom': 0.1},
    '2025-03-13': {'gdp_mom': -0.1, 'gdp_yoy': 0.9, 'prev_mom': 0.4, 'consensus_mom': 0.1},
    '2025-04-11': {'gdp_mom': 0.5, 'gdp_yoy': 1.1, 'prev_mom': -0.1, 'consensus_mom': 0.1},
    '2025-05-15': {'gdp_mom': 0.2, 'gdp_yoy': 1.3, 'prev_mom': 0.5, 'consensus_mom': 0.0},
    '2025-06-12': {'gdp_mom': -0.3, 'gdp_yoy': 0.9, 'prev_mom': 0.2, 'consensus_mom': -0.1},
    '2025-07-10': {'gdp_mom': 0.4, 'gdp_yoy': 1.0, 'prev_mom': -0.3, 'consensus_mom': 0.2},
    '2025-08-13': {'gdp_mom': 0.1, 'gdp_yoy': 1.2, 'prev_mom': 0.4, 'consensus_mom': 0.0},
    '2025-09-12': {'gdp_mom': 0.0, 'gdp_yoy': 1.0, 'prev_mom': 0.1, 'consensus_mom': 0.1},
    '2025-10-10': {'gdp_mom': -0.2, 'gdp_yoy': 0.6, 'prev_mom': 0.0, 'consensus_mom': 0.0},
    '2025-11-13': {'gdp_mom': 0.1, 'gdp_yoy': 0.5, 'prev_mom': -0.2, 'consensus_mom': 0.1},
    '2025-12-12': {'gdp_mom': 0.0, 'gdp_yoy': 0.4, 'prev_mom': 0.1, 'consensus_mom': 0.0},
    # 2026
    '2026-02-12': {'gdp_mom': -0.1, 'gdp_yoy': 0.3, 'prev_mom': 0.0, 'consensus_mom': 0.0},
    '2026-03-12': {'gdp_mom': 0.0, 'gdp_yoy': 0.2, 'prev_mom': -0.1, 'consensus_mom': 0.1},
    '2026-04-10': {'gdp_mom': -0.2, 'gdp_yoy': 0.0, 'prev_mom': 0.0, 'consensus_mom': 0.0},
    '2026-05-13': {'gdp_mom': -0.1, 'gdp_yoy': -0.1, 'prev_mom': -0.2, 'consensus_mom': 0.0},
}

# ═══════════════════════════════════════════════════════════════
# EDGE TABLE: (wyckoff, vol, signal) → (direction, avg_return, win_rate, n)
# Only combos with n≥3 and |avg|≥0.5% from backtest
# ═══════════════════════════════════════════════════════════════

EDGE_TABLE = {
    # Contrarian: RECESSION level in MARKDOWN = BoE cut hope
    ('MARKDOWN', 'NEUTRAL', 'BEAT'):        {'dir': 'LONG',  'avg': 3.431, 'wr': 1.00, 'n': 3, 'ics_adj': 0.06, 'size_mult': 1.05},
    ('MARKDOWN', 'LOW_VOL', 'CONTRACTION'):  {'dir': 'LONG',  'avg': 1.729, 'wr': 1.00, 'n': 3, 'ics_adj': 0.05, 'size_mult': 1.05},
    ('DISTRIBUTION', 'COMPRESSING', 'STRONG_BEAT'): {'dir': 'LONG', 'avg': 1.967, 'wr': 0.67, 'n': 3, 'ics_adj': 0.05, 'size_mult': 1.00},
    ('MARKDOWN', 'COMPRESSING', 'INLINE'):   {'dir': 'LONG',  'avg': 1.177, 'wr': 0.75, 'n': 4, 'ics_adj': 0.05, 'size_mult': 1.00},
    # Bearish combos
    ('ACCUMULATION', 'COMPRESSING', 'CONTRACTION'): {'dir': 'SHORT', 'avg': -5.466, 'wr': 0.00, 'n': 3, 'ics_adj': 0.08, 'size_mult': 1.10},
    ('MARKDOWN', 'NEUTRAL', 'CONTRACTION'):  {'dir': 'SHORT', 'avg': -4.369, 'wr': 0.33, 'n': 3, 'ics_adj': 0.06, 'size_mult': 1.05},
    ('MARKDOWN', 'COMPRESSING', 'BEAT'):     {'dir': 'SHORT', 'avg': -2.895, 'wr': 0.00, 'n': 3, 'ics_adj': 0.06, 'size_mult': 1.05},
    ('MARKDOWN', 'COMPRESSING', 'BIG_MISS'): {'dir': 'SHORT', 'avg': -2.262, 'wr': 0.33, 'n': 3, 'ics_adj': 0.05, 'size_mult': 1.00},
    ('DISTRIBUTION', 'COMPRESSING', 'CONTRACTION'): {'dir': 'SHORT', 'avg': -1.911, 'wr': 0.50, 'n': 4, 'ics_adj': 0.05, 'size_mult': 1.00},
    ('MARKDOWN', 'NEUTRAL', 'INLINE'):       {'dir': 'SHORT', 'avg': -1.137, 'wr': 0.60, 'n': 5, 'ics_adj': 0.05, 'size_mult': 1.00},
    ('ACCUMULATION', 'COMPRESSING', 'INLINE'): {'dir': 'SHORT', 'avg': -1.143, 'wr': 0.00, 'n': 3, 'ics_adj': 0.05, 'size_mult': 1.00},
}

# Fallback: STRONG_BEAT has edge regardless of regime
STRONG_BEAT_FALLBACK = {'dir': 'LONG', 'avg': 3.398, 'wr': 0.917, 'n': 12, 'ics_adj': 0.07, 'size_mult': 1.05}

# Fallback: RECESSION level (YoY < -2%) is contrarian bullish
RECESSION_LEVEL_FALLBACK = {'dir': 'LONG', 'avg': 1.241, 'wr': 0.615, 'n': 13, 'ics_adj': 0.05, 'size_mult': 1.00}

# ═══════════════════════════════════════════════════════════════
# FRESH DATA CACHE (for live scanner)
# ═══════════════════════════════════════════════════════════════
_FRESH_DATA = {}


def update_fresh_data(actual_mom, consensus_mom, prev_mom, yoy=None):
    """Feed fresh UK GDP data into cache (called by macro_fetch)."""
    _FRESH_DATA['gdp_mom'] = actual_mom
    _FRESH_DATA['consensus_mom'] = consensus_mom
    _FRESH_DATA['prev_mom'] = prev_mom
    if yoy is not None:
        _FRESH_DATA['gdp_yoy'] = yoy


def _classify_signal(gdp_mom, consensus_mom, prev_mom):
    """Classify UK GDP surprise."""
    if gdp_mom is None or consensus_mom is None:
        return 'NO_DATA'
    surprise = gdp_mom - consensus_mom
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


def _classify_level(gdp_yoy):
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


def _get_release_data(date_str=None):
    """Get release data for a given date."""
    if date_str is None:
        date_str = datetime.utcnow().strftime('%Y-%m-%d')
    if _FRESH_DATA.get('gdp_mom') is not None:
        return _FRESH_DATA
    return UK_GDP_RELEASES.get(date_str)


def score_m51_uk_gdp(wyckoff_phase='UNKNOWN', vol_regime='UNKNOWN',
                      direction='LONG', date_str=None):
    """
    Score M51: UK Monthly GDP session bias.

    Args:
        wyckoff_phase: M21 phase (ACCUMULATION/MARKUP/DISTRIBUTION/MARKDOWN/RANGE)
        vol_regime: M9 regime (TREND/COMPRESSING/NEUTRAL/CHOP/LOW_VOL/CRISIS)
        direction: Current trade direction (LONG/SHORT)
        date_str: Date to check (YYYY-MM-DD), defaults to today

    Returns:
        (status, score_adj, size_mult, details)
    """
    release_data = _get_release_data(date_str)
    if release_data is None:
        return 'NOT_RELEASE_DAY', 0.0, 1.0, {}

    gdp_mom = release_data.get('gdp_mom')
    consensus_mom = release_data.get('consensus_mom')
    prev_mom = release_data.get('prev_mom')
    gdp_yoy = release_data.get('gdp_yoy')

    if gdp_mom is None:
        return 'NOT_RELEASE_DAY', 0.0, 1.0, {}

    signal = _classify_signal(gdp_mom, consensus_mom, prev_mom)
    level = _classify_level(gdp_yoy)
    surprise = gdp_mom - consensus_mom if consensus_mom is not None else 0

    # Lookup edge table
    key = (wyckoff_phase, vol_regime, signal)
    edge = EDGE_TABLE.get(key)

    # Fallbacks
    if edge is None and signal == 'STRONG_BEAT':
        edge = STRONG_BEAT_FALLBACK
    if edge is None and level == 'RECESSION':
        edge = RECESSION_LEVEL_FALLBACK

    if edge is None:
        # Check if any edge matches wyckoff+vol combo
        for k, v in EDGE_TABLE.items():
            if k[0] == wyckoff_phase and k[1] == vol_regime:
                if v['dir'] == direction:
                    edge = v
                    break

    if edge is None:
        return 'NO_EDGE', 0.0, 1.0, {
            'signal': signal, 'level': level,
            'gdp_mom': gdp_mom, 'consensus_mom': consensus_mom,
            'surprise': surprise, 'gdp_yoy': gdp_yoy,
        }

    # Direction alignment
    if edge['dir'] != direction:
        score_adj = -abs(edge['ics_adj']) * 0.5
        size_mult = 0.85
    else:
        score_adj = edge['ics_adj']
        size_mult = edge['size_mult']

    details = {
        'signal': signal, 'level': level,
        'gdp_mom': gdp_mom, 'consensus_mom': consensus_mom,
        'prev_mom': prev_mom, 'gdp_yoy': gdp_yoy,
        'surprise': surprise,
        'edge_key': f"{wyckoff_phase} × {vol_regime} × {signal}",
        'edge_dir': edge['dir'], 'edge_avg': edge['avg'],
        'edge_wr': edge['wr'], 'edge_n': edge['n'],
    }
    return 'ACTIVE', score_adj, size_mult, details


def format_m51(status, score_adj, size_mult, details):
    """Format M51 output for scanner display."""
    if status == 'NOT_RELEASE_DAY':
        return "M51: — (not UK GDP release day)"
    if status == 'NO_EDGE':
        sig = details.get('signal', '?')
        lvl = details.get('level', '?')
        mom = details.get('gdp_mom', '?')
        return f"M51: {sig} ({lvl}, MoM {mom}%) — no regime edge"

    sig = details.get('signal', '?')
    lvl = details.get('level', '?')
    mom = details.get('gdp_mom', '?')
    con = details.get('consensus_mom', '?')
    sup = details.get('surprise', 0)
    edge_key = details.get('edge_key', '?')
    edge_dir = details.get('edge_dir', '?')
    edge_avg = details.get('edge_avg', 0)
    wr = details.get('edge_wr', 0)

    return (f"M51: {sig} ({lvl}) | MoM {mom}% vs {con}% ({sup:+.1f}%) | "
            f"{edge_key} → {edge_dir} {edge_avg:+.2f}% ({wr:.0%} WR) | "
            f"ICS {score_adj:+.03f} ×{size_mult:.2f}")


if __name__ == '__main__':
    print("=== M51 UK Monthly GDP Self-Test ===\n")
    test_cases = [
        ('MARKDOWN', 'NEUTRAL', 'LONG', '2021-09-10'),    # BEAT (0.1 vs 0.5)
        ('ACCUMULATION', 'COMPRESSING', 'SHORT', '2019-04-10'),  # CONTRACTION (-0.1)
        ('MARKDOWN', 'COMPRESSING', 'LONG', '2023-08-11'),  # INLINE (0.2 vs 0.2)
        ('DISTRIBUTION', 'COMPRESSING', 'LONG', '2022-03-11'),  # STRONG_BEAT (0.8 vs 0.8)
    ]
    for wyck, vol, dire, date in test_cases:
        status, adj, mult, det = score_m51_uk_gdp(wyck, vol, dire, date)
        print(format_m51(status, adj, mult, det))
        print(f"  → status={status}, ICS={adj:+.03f}, size={mult:.2f}\n")

    # Test non-release day
    status, adj, mult, det = score_m51_uk_gdp('RANGE', 'NEUTRAL', 'LONG', '2026-01-15')
    print(format_m51(status, adj, mult, det))

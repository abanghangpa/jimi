"""
M60: US PPI Session Bias (Regime-Conditional)

On BLS PPI (Producer Price Index) Final Demand release days (~mid-month, 12:30 UTC = 08:30 ET),
applies a regime-conditional directional bias based on:
  - Wyckoff phase (M21)
  - Volatility regime (M9)
  - PPI signal: STRONG_BEAT / BEAT / INLINE / MISS / BIG_MISS

PPI confirms or denies the CPI signal (released 1-2 days after CPI).
Pipeline inflation matters for Fed policy — PPI hot + CPI hot = persistent inflation.

Backtested on 100 US PPI releases (2018-2026) against ETH/USDT 15m data.

Key findings:
  Overall: -0.31% avg, 48% win, p=0.54 — NOT significant (noise).
  RANGE + COMPRESSING regime is consistently bearish across all signals.
  MARKUP + COMPRESSING + MISS: contrarian LONG (+1.57%, 75% win).
  Session chain: London Midday → NY PM 75-90% direction persistence.

Integration: lightweight modifier on BLS release days only (~12x/year).
Returns a score adjustment and size multiplier — does NOT veto.

Usage:
    from src.modules.m60_us_ppi import score_m60_us_ppi, format_m60
    status, score_adj, size_mult, details = score_m60_us_ppi(
        wyckoff_phase='RANGE', vol_regime='COMPRESSING', direction='LONG')
"""

from datetime import datetime, timedelta
import json
import os


# ═══════════════════════════════════════════════════════════════
# US PPI RELEASE DATES + ACTUAL YOY VALUES (Final Demand)
# Released 12:30 UTC (08:30 ET), usually 1-2 days after CPI
# Source: BLS PPI program, FRED WPSFD49207
# ═══════════════════════════════════════════════════════════════

PPI_RELEASES = {
    # ── 2018 ──
    '2018-01-11': {'ppi_yoy': 2.6, 'consensus_yoy': 2.5, 'prior_yoy': 2.6},
    '2018-02-15': {'ppi_yoy': 2.8, 'consensus_yoy': 2.7, 'prior_yoy': 2.6},
    '2018-03-14': {'ppi_yoy': 2.8, 'consensus_yoy': 2.8, 'prior_yoy': 2.8},
    '2018-04-11': {'ppi_yoy': 2.9, 'consensus_yoy': 2.8, 'prior_yoy': 2.8},
    '2018-05-10': {'ppi_yoy': 2.7, 'consensus_yoy': 2.8, 'prior_yoy': 2.9},
    '2018-06-12': {'ppi_yoy': 3.1, 'consensus_yoy': 2.9, 'prior_yoy': 2.7},
    '2018-07-11': {'ppi_yoy': 3.3, 'consensus_yoy': 3.2, 'prior_yoy': 3.1},
    '2018-08-09': {'ppi_yoy': 3.3, 'consensus_yoy': 3.3, 'prior_yoy': 3.3},
    '2018-09-12': {'ppi_yoy': 2.8, 'consensus_yoy': 3.2, 'prior_yoy': 3.3},
    '2018-10-10': {'ppi_yoy': 2.6, 'consensus_yoy': 2.6, 'prior_yoy': 2.8},
    '2018-11-14': {'ppi_yoy': 2.5, 'consensus_yoy': 2.5, 'prior_yoy': 2.6},
    '2018-12-11': {'ppi_yoy': 2.5, 'consensus_yoy': 2.5, 'prior_yoy': 2.5},
    # ── 2019 ──
    '2019-01-15': {'ppi_yoy': 2.0, 'consensus_yoy': 2.2, 'prior_yoy': 2.5},
    '2019-02-14': {'ppi_yoy': 1.7, 'consensus_yoy': 1.8, 'prior_yoy': 2.0},
    '2019-03-14': {'ppi_yoy': 1.9, 'consensus_yoy': 1.8, 'prior_yoy': 1.7},
    '2019-04-11': {'ppi_yoy': 2.2, 'consensus_yoy': 2.0, 'prior_yoy': 1.9},
    '2019-05-09': {'ppi_yoy': 2.2, 'consensus_yoy': 2.3, 'prior_yoy': 2.2},
    '2019-06-11': {'ppi_yoy': 1.8, 'consensus_yoy': 2.0, 'prior_yoy': 2.2},
    '2019-07-12': {'ppi_yoy': 1.7, 'consensus_yoy': 1.7, 'prior_yoy': 1.8},
    '2019-08-09': {'ppi_yoy': 1.7, 'consensus_yoy': 1.7, 'prior_yoy': 1.7},
    '2019-09-11': {'ppi_yoy': 1.4, 'consensus_yoy': 1.8, 'prior_yoy': 1.7},
    '2019-10-08': {'ppi_yoy': 1.1, 'consensus_yoy': 1.5, 'prior_yoy': 1.4},
    '2019-11-14': {'ppi_yoy': 1.1, 'consensus_yoy': 0.9, 'prior_yoy': 1.1},
    '2019-12-12': {'ppi_yoy': 1.1, 'consensus_yoy': 1.3, 'prior_yoy': 1.1},
    # ── 2020 ──
    '2020-01-14': {'ppi_yoy': 1.7, 'consensus_yoy': 1.3, 'prior_yoy': 1.1},
    '2020-02-19': {'ppi_yoy': 2.0, 'consensus_yoy': 1.6, 'prior_yoy': 1.7},
    '2020-03-12': {'ppi_yoy': 1.3, 'consensus_yoy': 1.2, 'prior_yoy': 2.0},
    '2020-04-09': {'ppi_yoy': -0.6, 'consensus_yoy': 0.2, 'prior_yoy': 1.3},
    '2020-05-12': {'ppi_yoy': -1.2, 'consensus_yoy': -1.2, 'prior_yoy': -0.6},
    '2020-06-10': {'ppi_yoy': -0.8, 'consensus_yoy': -1.2, 'prior_yoy': -1.2},
    '2020-07-14': {'ppi_yoy': -0.4, 'consensus_yoy': -0.7, 'prior_yoy': -0.8},
    '2020-08-11': {'ppi_yoy': -0.2, 'consensus_yoy': -0.3, 'prior_yoy': -0.4},
    '2020-09-11': {'ppi_yoy': 0.4, 'consensus_yoy': 0.2, 'prior_yoy': -0.2},
    '2020-10-14': {'ppi_yoy': 0.5, 'consensus_yoy': 0.4, 'prior_yoy': 0.4},
    '2020-11-13': {'ppi_yoy': 0.8, 'consensus_yoy': 0.5, 'prior_yoy': 0.5},
    '2020-12-11': {'ppi_yoy': 0.8, 'consensus_yoy': 0.7, 'prior_yoy': 0.8},
    # ── 2021 ──
    '2021-01-13': {'ppi_yoy': 1.3, 'consensus_yoy': 0.9, 'prior_yoy': 0.8},
    '2021-02-17': {'ppi_yoy': 1.7, 'consensus_yoy': 1.0, 'prior_yoy': 1.3},
    '2021-03-12': {'ppi_yoy': 2.8, 'consensus_yoy': 2.7, 'prior_yoy': 1.7},
    '2021-04-09': {'ppi_yoy': 4.2, 'consensus_yoy': 3.8, 'prior_yoy': 2.8},
    '2021-05-12': {'ppi_yoy': 6.6, 'consensus_yoy': 5.9, 'prior_yoy': 4.2},
    '2021-06-11': {'ppi_yoy': 7.3, 'consensus_yoy': 6.3, 'prior_yoy': 6.6},
    '2021-07-13': {'ppi_yoy': 7.8, 'consensus_yoy': 7.3, 'prior_yoy': 7.3},
    '2021-08-12': {'ppi_yoy': 8.6, 'consensus_yoy': 8.2, 'prior_yoy': 7.8},
    '2021-09-10': {'ppi_yoy': 8.7, 'consensus_yoy': 8.3, 'prior_yoy': 8.6},
    '2021-10-14': {'ppi_yoy': 8.8, 'consensus_yoy': 8.7, 'prior_yoy': 8.7},
    '2021-11-09': {'ppi_yoy': 8.8, 'consensus_yoy': 8.7, 'prior_yoy': 8.8},
    '2021-12-14': {'ppi_yoy': 9.8, 'consensus_yoy': 9.2, 'prior_yoy': 8.8},
    # ── 2022 ──
    '2022-01-13': {'ppi_yoy': 9.7, 'consensus_yoy': 9.8, 'prior_yoy': 9.8},
    '2022-02-15': {'ppi_yoy': 10.0, 'consensus_yoy': 9.1, 'prior_yoy': 9.7},
    '2022-03-11': {'ppi_yoy': 10.0, 'consensus_yoy': 10.0, 'prior_yoy': 10.0},
    '2022-04-12': {'ppi_yoy': 11.2, 'consensus_yoy': 10.6, 'prior_yoy': 10.0},
    '2022-05-12': {'ppi_yoy': 11.0, 'consensus_yoy': 10.7, 'prior_yoy': 11.2},
    '2022-06-14': {'ppi_yoy': 10.9, 'consensus_yoy': 10.9, 'prior_yoy': 11.0},
    '2022-07-14': {'ppi_yoy': 9.8, 'consensus_yoy': 10.4, 'prior_yoy': 10.9},
    '2022-08-11': {'ppi_yoy': 8.7, 'consensus_yoy': 8.8, 'prior_yoy': 9.8},
    '2022-09-14': {'ppi_yoy': 8.5, 'consensus_yoy': 8.8, 'prior_yoy': 8.7},
    '2022-10-12': {'ppi_yoy': 8.0, 'consensus_yoy': 8.4, 'prior_yoy': 8.5},
    '2022-11-15': {'ppi_yoy': 7.4, 'consensus_yoy': 8.0, 'prior_yoy': 8.0},
    '2022-12-09': {'ppi_yoy': 7.4, 'consensus_yoy': 7.2, 'prior_yoy': 7.4},
    # ── 2023 ──
    '2023-01-18': {'ppi_yoy': 6.2, 'consensus_yoy': 6.8, 'prior_yoy': 7.4},
    '2023-02-16': {'ppi_yoy': 6.0, 'consensus_yoy': 5.4, 'prior_yoy': 6.2},
    '2023-03-15': {'ppi_yoy': 4.6, 'consensus_yoy': 5.4, 'prior_yoy': 6.0},
    '2023-04-13': {'ppi_yoy': 2.7, 'consensus_yoy': 3.0, 'prior_yoy': 4.6},
    '2023-05-11': {'ppi_yoy': 2.3, 'consensus_yoy': 2.4, 'prior_yoy': 2.7},
    '2023-06-14': {'ppi_yoy': 1.1, 'consensus_yoy': 1.5, 'prior_yoy': 2.3},
    '2023-07-13': {'ppi_yoy': 0.2, 'consensus_yoy': 0.4, 'prior_yoy': 1.1},
    '2023-08-11': {'ppi_yoy': 0.8, 'consensus_yoy': 0.7, 'prior_yoy': 0.2},
    '2023-09-14': {'ppi_yoy': 1.6, 'consensus_yoy': 1.3, 'prior_yoy': 0.8},
    '2023-10-12': {'ppi_yoy': 2.2, 'consensus_yoy': 1.6, 'prior_yoy': 1.6},
    '2023-11-15': {'ppi_yoy': 1.3, 'consensus_yoy': 1.9, 'prior_yoy': 2.2},
    '2023-12-13': {'ppi_yoy': 1.0, 'consensus_yoy': 1.0, 'prior_yoy': 1.3},
    # ── 2024 ──
    '2024-01-12': {'ppi_yoy': 1.0, 'consensus_yoy': 1.3, 'prior_yoy': 1.0},
    '2024-02-16': {'ppi_yoy': 0.9, 'consensus_yoy': 0.6, 'prior_yoy': 1.0},
    '2024-03-14': {'ppi_yoy': 1.6, 'consensus_yoy': 1.2, 'prior_yoy': 0.9},
    '2024-04-11': {'ppi_yoy': 2.1, 'consensus_yoy': 2.2, 'prior_yoy': 1.6},
    '2024-05-14': {'ppi_yoy': 2.2, 'consensus_yoy': 2.3, 'prior_yoy': 2.1},
    '2024-06-13': {'ppi_yoy': 2.2, 'consensus_yoy': 2.3, 'prior_yoy': 2.2},
    '2024-07-12': {'ppi_yoy': 2.7, 'consensus_yoy': 2.3, 'prior_yoy': 2.2},
    '2024-08-13': {'ppi_yoy': 2.2, 'consensus_yoy': 2.3, 'prior_yoy': 2.7},
    '2024-09-12': {'ppi_yoy': 1.8, 'consensus_yoy': 1.8, 'prior_yoy': 2.2},
    '2024-10-11': {'ppi_yoy': 1.8, 'consensus_yoy': 1.6, 'prior_yoy': 1.8},
    '2024-11-14': {'ppi_yoy': 2.4, 'consensus_yoy': 2.3, 'prior_yoy': 1.8},
    '2024-12-12': {'ppi_yoy': 3.0, 'consensus_yoy': 2.6, 'prior_yoy': 2.4},
    # ── 2025 ──
    '2025-01-14': {'ppi_yoy': 3.3, 'consensus_yoy': 3.5, 'prior_yoy': 3.0},
    '2025-02-13': {'ppi_yoy': 3.5, 'consensus_yoy': 3.3, 'prior_yoy': 3.3},
    '2025-03-13': {'ppi_yoy': 3.2, 'consensus_yoy': 3.3, 'prior_yoy': 3.5},
    '2025-04-10': {'ppi_yoy': 2.7, 'consensus_yoy': 3.3, 'prior_yoy': 3.2},
    '2025-05-15': {'ppi_yoy': 2.4, 'consensus_yoy': 2.5, 'prior_yoy': 2.7},
    '2025-06-12': {'ppi_yoy': 2.3, 'consensus_yoy': 2.3, 'prior_yoy': 2.4},
    '2025-07-16': {'ppi_yoy': 2.3, 'consensus_yoy': 2.3, 'prior_yoy': 2.3},
    '2025-08-14': {'ppi_yoy': 2.2, 'consensus_yoy': 2.3, 'prior_yoy': 2.3},
    '2025-09-11': {'ppi_yoy': 2.0, 'consensus_yoy': 2.2, 'prior_yoy': 2.2},
    '2025-10-15': {'ppi_yoy': 1.8, 'consensus_yoy': 2.0, 'prior_yoy': 2.0},
    '2025-11-13': {'ppi_yoy': 1.8, 'consensus_yoy': 1.9, 'prior_yoy': 1.8},
    '2025-12-11': {'ppi_yoy': 2.0, 'consensus_yoy': 1.9, 'prior_yoy': 1.8},
    # ── 2026 ──
    '2026-01-14': {'ppi_yoy': 2.5, 'consensus_yoy': 2.0, 'prior_yoy': 2.0},
    '2026-02-13': {'ppi_yoy': 3.2, 'consensus_yoy': 2.5, 'prior_yoy': 2.5},
    '2026-03-13': {'ppi_yoy': 3.5, 'consensus_yoy': 3.3, 'prior_yoy': 3.2},
    '2026-04-14': {'ppi_yoy': 6.3, 'consensus_yoy': 4.0, 'prior_yoy': 3.5},
}

# All scheduled PPI release dates (including future dates without data yet)
# Used by M22/M23 for time decay and release detection
PPI_SCHEDULE_DATES = set(PPI_RELEASES.keys()) | {
    '2026-05-13', '2026-06-11', '2026-07-10', '2026-08-13',
    '2026-09-11', '2026-10-14', '2026-11-13', '2026-12-10',
}


# ═══════════════════════════════════════════════════════════════
# REGIME-CONDITIONAL EDGE TABLE
# Backtested: 100 US PPI releases, 2018-2026, ETH/USDT 15m
# ═══════════════════════════════════════════════════════════════

# Key: (wyckoff_phase, vol_regime, ppi_signal)
# Value: {avg_ret, win, n, bias, ics_adj, size_mult}
# Only entries with n >= 3 and |avg_24h| >= 0.5% are included
EDGE_TABLE = {
    # ── SHORT EDGES (RANGE + COMPRESSING = dominant bearish regime) ──
    ('RANGE', 'COMPRESSING', 'BIG_MISS'):   {'avg_ret': -1.917, 'win': 0.333, 'n': 9,  'bias': 'SHORT',
                                              'ics_adj': 0.06, 'size_mult': 1.00},
    ('RANGE', 'COMPRESSING', 'INLINE'):     {'avg_ret': -0.886, 'win': 0.562, 'n': 16, 'bias': 'SHORT',
                                              'ics_adj': 0.05, 'size_mult': 0.90},
    ('RANGE', 'COMPRESSING', 'STRONG_BEAT'): {'avg_ret': -0.791, 'win': 0.429, 'n': 14, 'bias': 'SHORT',
                                               'ics_adj': 0.05, 'size_mult': 0.90},
    ('RANGE', 'COMPRESSING', 'MISS'):       {'avg_ret': -0.567, 'win': 0.375, 'n': 8,  'bias': 'SHORT',
                                              'ics_adj': 0.05, 'size_mult': 0.85},
    ('RANGE', 'COMPRESSING', 'BEAT'):       {'avg_ret': -0.530, 'win': 0.273, 'n': 11, 'bias': 'SHORT',
                                              'ics_adj': 0.05, 'size_mult': 0.85},

    # ── CHOP + LOW_VOL + INLINE = deep SHORT ──
    ('CHOP', 'LOW_VOL', 'INLINE'):          {'avg_ret': -2.313, 'win': 0.500, 'n': 4,  'bias': 'SHORT',
                                              'ics_adj': 0.07, 'size_mult': 0.75},

    # ── LONG EDGES (contrarian in markup phase) ──
    ('MARKUP', 'COMPRESSING', 'MISS'):      {'avg_ret': 1.568, 'win': 0.750, 'n': 4,  'bias': 'LONG',
                                              'ics_adj': 0.06, 'size_mult': 0.85},
    ('MARKUP', 'LOW_VOL', 'STRONG_BEAT'):   {'avg_ret': 1.206, 'win': 0.667, 'n': 3,  'bias': 'LONG',
                                              'ics_adj': 0.05, 'size_mult': 0.75},
}

# Broader combos (Wyckoff + inflation level) — larger sample
BROAD_EDGE_TABLE = {
    # Key: (wyckoff_phase, inflation_level)
    ('RANGE', 'RUNAWAY'):                   {'avg_ret': -2.590, 'win': 0.000, 'n': 9,  'bias': 'SHORT'},
    ('MARKDOWN', 'LOW_VOL', 'RUNAWAY'):     {'avg_ret': 3.290, 'win': 0.500, 'n': 4,  'bias': 'LONG'},
    ('MARKDOWN', 'COMPRESSING', 'TARGET'):  {'avg_ret': -2.213, 'win': 0.667, 'n': 3, 'bias': 'SHORT'},
}

# Fallback: RANGE + COMPRESSING generally → SHORT
RANGE_COMPRESSING_FALLBACK = {'avg_ret': -0.920, 'win': 0.430, 'n': 58, 'bias': 'SHORT',
                               'ics_adj': 0.05, 'size_mult': 0.85}


def _classify_signal(actual_yoy, consensus_yoy):
    """Classify PPI surprise signal."""
    surprise = actual_yoy - consensus_yoy
    if surprise > 0.3:
        return 'STRONG_BEAT'
    elif surprise > 0.1:
        return 'BEAT'
    elif surprise < -0.3:
        return 'BIG_MISS'
    elif surprise < -0.1:
        return 'MISS'
    return 'INLINE'


def _classify_inflation(actual_yoy):
    """Classify PPI inflation level."""
    if actual_yoy > 6.0:
        return 'RUNAWAY'
    elif actual_yoy > 4.0:
        return 'HOT'
    elif actual_yoy > 2.5:
        return 'WARM'
    elif actual_yoy >= 1.5:
        return 'TARGET'
    return 'COOL'


def _is_ppi_release_day(date_str, window_days=1):
    """Check if date_str is within window_days of a PPI release.

    Returns: (is_release, release_date, release_data) or (False, None, None)
    """
    today = datetime.strptime(date_str, '%Y-%m-%d')

    for release_date_str, release_data in sorted(PPI_RELEASES.items(), reverse=True):
        release_dt = datetime.strptime(release_date_str, '%Y-%m-%d')
        days_since = (today - release_dt).days
        if 0 <= days_since <= window_days:
            return True, release_date_str, release_data

    return False, None, None


def score_m60_us_ppi(wyckoff_phase='RANGE', vol_regime='COMPRESSING',
                     direction='LONG', today_str=None, config=None):
    """Score the US PPI session bias.

    Args:
        wyckoff_phase: from M21 ('ACCUMULATION', 'MARKUP', 'DISTRIBUTION', 'MARKDOWN', 'RANGE')
        vol_regime: from M9 ('TREND', 'COMPRESSING', 'LOW_VOL', 'NEUTRAL', 'CHOP', 'CRISIS', 'HIGH_VOL')
        direction: trade direction ('LONG' or 'SHORT')
        today_str: YYYY-MM-DD override (for backtesting)
        config: config dict (optional)

    Returns:
        status: 'PASS' (active), 'SKIP' (not release day), or 'WEAK' (low confidence)
        score_adj: score adjustment (-0.10 to +0.10)
        size_mult: position size multiplier (0.50 to 1.00)
        details: dict
    """
    cfg = config or {}

    if not cfg.get('M60_ENABLED', True):
        return 'SKIP', 0.0, 1.0, {'regime': 'DISABLED'}

    if today_str is None:
        today_str = datetime.utcnow().strftime('%Y-%m-%d')

    # Check if today is a PPI release day
    is_release, release_date, release_data = _is_ppi_release_day(
        today_str, window_days=cfg.get('M60_WINDOW_DAYS', 1))

    if not is_release:
        return 'SKIP', 0.0, 1.0, {'regime': 'NOT_RELEASE_DAY'}

    ppi_yoy = release_data['ppi_yoy']
    consensus_yoy = release_data['consensus_yoy']
    prior_yoy = release_data['prior_yoy']

    signal = _classify_signal(ppi_yoy, consensus_yoy)
    infl_level = _classify_inflation(ppi_yoy)
    surprise = ppi_yoy - consensus_yoy

    # ── Lookup 1: Fine-grained (Wyckoff + Vol + Signal) ──
    fine_key = (wyckoff_phase, vol_regime, signal)
    fine_match = EDGE_TABLE.get(fine_key)

    # ── Lookup 2: RANGE + COMPRESSING fallback ──
    range_compressing_match = None
    if wyckoff_phase == 'RANGE' and vol_regime == 'COMPRESSING':
        range_compressing_match = RANGE_COMPRESSING_FALLBACK

    # ── Lookup 3: Broad (Wyckoff + inflation level) ──
    broad_key = (wyckoff_phase, infl_level)
    broad_match = BROAD_EDGE_TABLE.get(broad_key)

    # ── Determine best signal ──
    best_match = None
    best_source = 'NONE'
    confidence = 0.0

    if fine_match and fine_match['n'] >= 3:
        best_match = fine_match
        best_source = 'FINE'
        confidence = min(1.0, fine_match['n'] / 10)
    elif range_compressing_match and range_compressing_match['n'] >= 5:
        best_match = range_compressing_match
        best_source = 'RANGE_COMPRESSING'
        confidence = min(1.0, range_compressing_match['n'] / 20)
    elif broad_match and broad_match['n'] >= 3:
        best_match = broad_match
        best_source = 'BROAD'
        confidence = min(1.0, broad_match['n'] / 10)

    if best_match is None:
        return 'SKIP', 0.0, 1.0, {
            'regime': 'NO_EDGE',
            'wyckoff': wyckoff_phase,
            'vol_regime': vol_regime,
            'ppi_signal': signal,
            'infl_level': infl_level,
            'ppi_yoy': ppi_yoy,
            'consensus_yoy': consensus_yoy,
            'surprise': surprise,
            'release_date': release_date,
        }

    # ── Compute score adjustment ──
    avg_ret = best_match['avg_ret']
    win_rate = best_match['win']
    n = best_match['n']
    bias = best_match['bias']
    ics_adj = best_match.get('ics_adj', 0.05)
    base_size_mult = best_match.get('size_mult', 0.85)

    # Scale by confidence (sample size)
    score_adj = ics_adj * confidence

    # Apply direction
    if bias == 'LONG' and direction == 'LONG':
        score_adj = abs(score_adj)
    elif bias == 'LONG' and direction == 'SHORT':
        score_adj = -abs(score_adj)
    elif bias == 'SHORT' and direction == 'SHORT':
        score_adj = abs(score_adj)
    elif bias == 'SHORT' and direction == 'LONG':
        score_adj = -abs(score_adj)

    # Size multiplier: scale by confidence
    if confidence >= 0.7:
        size_mult = base_size_mult
    elif confidence >= 0.4:
        size_mult = base_size_mult * 0.85
    else:
        size_mult = base_size_mult * 0.70

    if n < 5:
        size_mult *= 0.80

    size_mult = round(max(0.50, min(1.0, size_mult)), 2)
    score_adj = round(score_adj, 3)

    status = 'PASS' if confidence >= 0.3 else 'WEAK'

    details = {
        'regime': f'PPI_{bias}',
        'release_date': release_date,
        'ppi_yoy': ppi_yoy,
        'consensus_yoy': consensus_yoy,
        'prior_yoy': prior_yoy,
        'surprise': round(surprise, 2),
        'ppi_signal': signal,
        'infl_level': infl_level,
        'wyckoff': wyckoff_phase,
        'vol_regime': vol_regime,
        'bias': bias,
        'avg_ret_24h': avg_ret,
        'win_rate': win_rate,
        'sample_size': n,
        'confidence': round(confidence, 2),
        'source': best_source,
        'score_adj': score_adj,
        'size_mult': size_mult,
    }

    return status, score_adj, size_mult, details


def format_m60(details):
    """Format M60 details for terminal output."""
    if not details or details.get('regime') in ('DISABLED', 'NOT_RELEASE_DAY', 'NO_EDGE'):
        regime = details.get('regime', '?') if details else '?'
        if regime == 'NOT_RELEASE_DAY':
            return ''  # silent when not active
        return ''

    bias = details.get('bias', '?')
    ppi_yoy = details.get('ppi_yoy', 0)
    consensus = details.get('consensus_yoy', 0)
    prior = details.get('prior_yoy', 0)
    surprise = details.get('surprise', 0)
    signal = details.get('ppi_signal', '?')
    infl = details.get('infl_level', '?')
    wyckoff = details.get('wyckoff', '?')
    vol = details.get('vol_regime', '?')
    avg_ret = details.get('avg_ret_24h', 0)
    win = details.get('win_rate', 0)
    n = details.get('sample_size', 0)
    conf = details.get('confidence', 0)
    source = details.get('source', '?')
    score_adj = details.get('score_adj', 0)
    size_mult = details.get('size_mult', 1.0)
    release = details.get('release_date', '?')

    icon = '🟢' if bias == 'LONG' else '🔴' if bias == 'SHORT' else '⚪'
    conf_icon = '🟢' if conf >= 0.7 else '🟡' if conf >= 0.4 else '🟠'
    surp_icon = '📈' if surprise > 0.1 else '📉' if surprise < -0.1 else '➡️'

    lines = []
    lines.append(f"\n  {icon} M60 US PPI SESSION BIAS: {bias}")
    lines.append(f"    Release: {release}  |  PPI YoY: {ppi_yoy:.1f}% ({signal})  |  Consensus: {consensus:.1f}%  |  Prior: {prior:.1f}%")
    lines.append(f"    Surprise: {surp_icon} {surprise:+.2f}pp  |  Inflation: {infl}")
    lines.append(f"    Context: {wyckoff} + {vol}")
    lines.append(f"    Backtest: avg 24h={avg_ret:+.2f}%  win={win*100:.0f}%  n={n}  source={source}")
    lines.append(f"    {conf_icon} Confidence: {conf:.2f}  |  Score adj: {score_adj:+.3f}  |  Size: {size_mult:.2f}x")

    return '\n'.join(lines)

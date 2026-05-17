"""
M40: Germany CPI Session Bias (Regime-Conditional)

On Destatis CPI release days (~10th-20th of each month, 07:00 UTC = 08:00 CET),
applies a session-conditional directional bias based on the combination of:
  - Wyckoff phase (M21)
  - Volatility regime (M9)
  - CPI signal: STRONG_BEAT / BEAT / INLINE / MISS / BIG_MISS
  - CPI level: HYPERINFLATION / HOT / WARM / MILD / COOL / DEFLATION
  - CPI direction: RISING / STABLE / FALLING

Thesis (from user #24):
  Europe Morning (14:00-16:00 MYT) → Eurozone Open (next day) → US Morning
  Hot German CPI → EUR/USD surge → DXY lower → ETH fast intraday rally
  European desks lock positions ahead of Eurozone CPI Flash (1-2 days later)
  Hawkish ECB pressure → reduces loose global liquidity expectations

Backtested on 101 Germany CPI releases (2018-2026) against ETH/USDT 15m data.

Key findings:
  24h aggregate: -0.113% avg, 42.6% win, n=101 — NOT significant overall
  (Germany CPI is NOISE at the 24h level. Only specific combos have edge.)

  Specific combos with edge (n≥3, |avg|≥0.5%):
    MARKDOWN + NEUTRAL + INLINE: -5.608% avg, 14% win, n=7  → SHORT bias
    RANGE + NEUTRAL + BEAT:      -4.441% avg, 33% win, n=6  → SHORT bias
    RANGE + COMPRESSING + BEAT:  +1.823% avg, 33% win, n=6  → LONG bias
    MARKUP + NEUTRAL + BEAT:     +1.606% avg, 67% win, n=3  → LONG bias
    MARKUP + NEUTRAL + INLINE:   -1.520% avg, 0% win, n=4   → SHORT bias
    MARKDOWN + NEUTRAL + BEAT:   +1.154% avg, 50% win, n=6  → LONG bias
    RANGE + NEUTRAL + INLINE:    -0.881% avg, 67% win, n=9  → SHORT bias
    ACCUMULATION + TRENDING + INLINE: -0.790% avg, 33% win, n=3 → SHORT bias

  CPI Level × Direction combos with edge:
    WARM + FALLING + MISS:        +4.630% avg, 75% win, n=4  → LONG
    WARM + STABLE + BEAT:         +4.626% avg, 80% win, n=5  → LONG
    MILD + STABLE + INLINE:       -3.436% avg, 40% win, n=10 → SHORT
    WARM + FALLING + INLINE:      -4.484% avg, 0% win, n=3   → SHORT

  Transmission chain:
    Asia session: 89-93% direction persistence ✅ (strong intra-Asia chain)
    Asia→Europe: BREAKS at Tokyo Close → Pre-London (0% persist) ❌
    London onwards: 77-89% persistence ✅ (chain restarts)
    London→NY: 89% Midday→Pre-Open, 84% Pre-Open→NY Open ✅

  Statistical tests:
    One-sample t-test: p=0.85 (NOT significant)
    BEAT vs MISS: p=0.23 (NOT significant)
    HOT vs COOL: p=0.78 (NOT significant)

Integration: lightweight modifier on Destatis release days only (~12x/year).
Returns a score adjustment and size multiplier — does NOT veto.

Usage:
    from src.modules.m40_germany_cpi import score_m40_germany_cpi, format_m40
    status, score_adj, size_mult, details = score_m40_germany_cpi(
        wyckoff_phase='RANGE', vol_regime='CHOP', direction='LONG')
"""

from datetime import datetime, timedelta
import json
import os


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
# EDGE TABLE — Wyckoff × Vol × Signal → avg 24h return
# Only combos with n≥3, |avg|≥0.5%
# ═══════════════════════════════════════════════════════════════

EDGE_TABLE = {
    # (wyckoff, vol, signal): (avg_24h_return, win_rate, sample_size, direction_bias)
    ('MARKDOWN', 'NEUTRAL', 'INLINE'):       (-5.608, 0.143, 7, 'SHORT'),
    ('RANGE', 'NEUTRAL', 'BEAT'):            (-4.441, 0.333, 6, 'SHORT'),
    ('RANGE', 'COMPRESSING', 'BEAT'):        (1.823, 0.333, 6, 'LONG'),
    ('MARKUP', 'NEUTRAL', 'BEAT'):           (1.606, 0.667, 3, 'LONG'),
    ('MARKUP', 'NEUTRAL', 'INLINE'):         (-1.520, 0.000, 4, 'SHORT'),
    ('MARKDOWN', 'NEUTRAL', 'BEAT'):         (1.154, 0.500, 6, 'LONG'),
    ('RANGE', 'NEUTRAL', 'INLINE'):          (-0.881, 0.667, 9, 'SHORT'),
    ('ACCUMULATION', 'TRENDING', 'INLINE'):  (-0.790, 0.333, 3, 'SHORT'),
}

# CPI Level × Direction edge table
LEVEL_DIR_EDGE = {
    ('WARM', 'FALLING', 'MISS'):             (4.630, 0.750, 4, 'LONG'),
    ('WARM', 'STABLE', 'BEAT'):              (4.626, 0.800, 5, 'LONG'),
    ('MILD', 'STABLE', 'INLINE'):            (-3.436, 0.400, 10, 'SHORT'),
    ('WARM', 'FALLING', 'INLINE'):           (-4.484, 0.000, 3, 'SHORT'),
    ('WARM', 'STABLE', 'INLINE'):            (0.909, 0.538, 13, 'LONG'),
}


def _classify_signal(actual, consensus, prior):
    """Classify CPI release as BEAT / INLINE / MISS relative to consensus."""
    diff = actual - consensus
    surprise = diff / abs(consensus) if consensus != 0 else diff

    if surprise > 0.15:
        return 'STRONG_BEAT'
    elif surprise > 0.03:
        return 'BEAT'
    elif surprise < -0.15:
        return 'BIG_MISS'
    elif surprise < -0.03:
        return 'MISS'
    else:
        return 'INLINE'


def _classify_level(actual):
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


def _classify_direction(actual, prior):
    """Classify CPI direction: rising, stable, falling."""
    change = actual - prior
    if change > 0.2:
        return 'RISING'
    elif change < -0.2:
        return 'FALLING'
    else:
        return 'STABLE'


def _is_release_day(today_str):
    """Check if today is a Germany CPI release day (±1 day window)."""
    today = datetime.strptime(today_str, '%Y-%m-%d')
    for date_str in GERMANY_CPI_RELEASES:
        release_dt = datetime.strptime(date_str, '%Y-%m-%d')
        delta = abs((today - release_dt).days)
        if delta <= 1:
            return date_str, GERMANY_CPI_RELEASES[date_str]
    return None, None


def score_m40_germany_cpi(wyckoff_phase='RANGE', vol_regime='CHOP',
                           direction='LONG', today_str=None, config=None):
    """
    Score M40: Germany CPI session bias.

    Args:
        wyckoff_phase: M21 Wyckoff phase (ACCUMULATION/MARKUP/DISTRIBUTION/MARKDOWN/RANGE)
        vol_regime: M9 volatility regime (TRENDING/SQUEEZE/CHOP/COMPRESSING/NEUTRAL)
        direction: Current trade direction (LONG/SHORT)
        today_str: Today's date (YYYY-MM-DD)
        config: Config dict (optional)

    Returns:
        (status, score_adj, size_mult, details)
        status: 'PASS' / 'WEAK' / 'SKIP' / 'NO_EDGE' / 'DISABLED'
        score_adj: ICS score adjustment (-0.10 to +0.10)
        size_mult: Position size multiplier (0.5-1.0)
        details: Dict with analysis details
    """
    cfg = config or {}

    # Check if M40 is enabled
    if not cfg.get('M40_ENABLED', True):
        return 'DISABLED', 0.0, 1.0, {'regime': 'DISABLED'}

    # Check if today is a release day
    if today_str is None:
        today_str = datetime.utcnow().strftime('%Y-%m-%d')

    release_date, release_data = _is_release_day(today_str)
    if release_data is None:
        return 'SKIP', 0.0, 1.0, {'regime': 'NOT_RELEASE_DAY'}

    # Classify the release
    cpi_yoy = release_data['cpi_yoy']
    consensus = release_data['consensus']
    prior = release_data['prior']

    signal = _classify_signal(cpi_yoy, consensus, prior)
    cpi_level = _classify_level(cpi_yoy)
    cpi_dir = _classify_direction(cpi_yoy, prior)

    # Look up edge table: Wyckoff × Vol × Signal
    edge_key = (wyckoff_phase, vol_regime, signal)
    edge = EDGE_TABLE.get(edge_key)

    # Fallback: try just Wyckoff × Signal (any vol)
    if edge is None:
        for (w, v, s), e in EDGE_TABLE.items():
            if w == wyckoff_phase and s == signal:
                edge = e
                edge_key = (w, v, s)
                break

    # Fallback: try Level × Direction edge
    level_dir_key = (cpi_level, cpi_dir, signal)
    level_dir_edge = LEVEL_DIR_EDGE.get(level_dir_key)

    # Determine the best edge to use
    if edge and abs(edge[0]) >= 0.5 and edge[2] >= 3:
        avg_ret, win_rate, n, bias = edge
        source = f'wyckoff_vol_signal: {edge_key}'
        confidence = min(1.0, n / 10.0)  # scale by sample size
    elif level_dir_edge and abs(level_dir_edge[0]) >= 0.5 and level_dir_edge[2] >= 3:
        avg_ret, win_rate, n, bias = level_dir_edge
        source = f'level_direction: {level_dir_key}'
        confidence = min(1.0, n / 10.0)
    else:
        # No edge found
        return 'NO_EDGE', 0.0, 1.0, {
            'regime': 'NO_EDGE',
            'release_date': release_date,
            'cpi_yoy': cpi_yoy,
            'consensus': consensus,
            'signal': signal,
            'cpi_level': cpi_level,
            'cpi_direction': cpi_dir,
            'wyckoff': wyckoff_phase,
            'vol': vol_regime,
        }

    # Compute ICS adjustment
    # Strong edges get ±0.05-0.10, weak edges get ±0.02-0.05
    if abs(avg_ret) >= 3.0 and n >= 5:
        score_adj = 0.10 if avg_ret > 0 else -0.10
    elif abs(avg_ret) >= 1.5:
        score_adj = 0.07 if avg_ret > 0 else -0.07
    elif abs(avg_ret) >= 0.5:
        score_adj = 0.05 if avg_ret > 0 else -0.05
    else:
        score_adj = 0.0

    # Direction alignment: if edge bias disagrees with current direction, reduce
    if bias != direction:
        score_adj *= -0.5  # flip and reduce

    # Size multiplier: reduce for low-confidence or low-sample edges
    if n >= 5 and win_rate >= 0.6:
        size_mult = 1.0
    elif n >= 3 and win_rate >= 0.5:
        size_mult = 0.85
    else:
        size_mult = 0.70

    # Status
    if abs(score_adj) >= 0.07:
        status = 'PASS'
    elif abs(score_adj) >= 0.03:
        status = 'WEAK'
    else:
        status = 'NO_EDGE'

    details = {
        'regime': f'{wyckoff_phase}_{vol_regime}_{signal}',
        'release_date': release_date,
        'cpi_yoy': cpi_yoy,
        'consensus': consensus,
        'prior': prior,
        'signal': signal,
        'cpi_level': cpi_level,
        'cpi_direction': cpi_dir,
        'wyckoff': wyckoff_phase,
        'vol': vol_regime,
        'bias': bias,
        'avg_ret_24h': avg_ret,
        'win_rate': win_rate,
        'sample_size': n,
        'confidence': confidence,
        'source': source,
        'score_adj': score_adj,
        'size_mult': size_mult,
    }

    return status, score_adj, size_mult, details


def format_m40(details):
    """Format M40 details for scanner output."""
    if not details or details.get('regime') in ('DISABLED', 'NOT_RELEASE_DAY', 'NO_EDGE'):
        return None

    bias = details.get('bias', '?')
    cpi = details.get('cpi_yoy', 0)
    cons = details.get('consensus', 0)
    signal = details.get('signal', '?')
    cpi_level = details.get('cpi_level', '?')
    cpi_dir = details.get('cpi_direction', '?')
    avg_ret = details.get('avg_ret_24h', 0)
    win_rate = details.get('win_rate', 0)
    n = details.get('sample_size', 0)
    conf = details.get('confidence', 0)
    wyckoff = details.get('wyckoff', '?')
    vol = details.get('vol', '?')
    score_adj = details.get('score_adj', 0)
    size_mult = details.get('size_mult', 1.0)
    release_date = details.get('release_date', '?')

    bias_icon = '🟢' if bias == 'LONG' else '🔴' if bias == 'SHORT' else '⚪'
    conf_icon = '🟢' if conf >= 0.7 else '🟡' if conf >= 0.4 else '🟠'
    dir_icon = '📈' if cpi_dir == 'RISING' else '📉' if cpi_dir == 'FALLING' else '➡️'

    lines = [
        f"  M40 Germany CPI: {bias_icon} {bias:>8}  "
        f"CPI={cpi:.1f}% cons={cons:.1f}% {dir_icon}{cpi_dir}  "
        f"level={cpi_level} signal={signal}",
        f"    Backtest: {conf_icon} {wyckoff}+{vol}+{signal}  "
        f"24h={avg_ret:+.2f}% win={win_rate*100:.0f}% n={n}  "
        f"adj={score_adj:+.3f} size={size_mult:.2f}x  "
        f"chain: Asia 89-93%, London→NY 77-89%",
    ]
    return '\n'.join(lines)

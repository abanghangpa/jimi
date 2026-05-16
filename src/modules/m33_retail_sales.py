"""
M33: US Retail Sales Session Bias (Regime-Conditional)

On US Retail Sales release days (~15th of each month, 08:30 ET = 13:30 UTC),
applies a 24h directional bias based on the combination of:
  - Wyckoff phase (M21): ACCUMULATION / MARKUP / DISTRIBUTION / MARKDOWN / RANGE
  - Volatility regime (M9): TREND / SQUEEZE / CHOP / LOW_VOL
  - Retail Sales signal: STRONG_BEAT / BEAT / INLINE / MISS / BIG_MISS

Backtested on 101 US Retail Sales releases (2018-2026) against ETH/USDT 15m data.

Key findings (24h return):
  BIG_MISS + LOW_VOL + MARKDOWN: +2.32% avg, 80% win, n=5  → LONG bias
  BIG_MISS + CHOP + MARKUP:      -3.33% avg, 0% win,  n=5  → SHORT bias
  BEAT + CHOP + RANGE:           +3.62% avg, 100% win, n=3  → LONG bias
  MISS + LOW_VOL + MARKUP:       +2.91% avg, 100% win, n=3  → LONG bias
  BIG_MISS + LOW_VOL + MARKUP:   +4.50% avg, 100% win, n=3  → LONG bias

Transmission chain: Release→NY AM holds (80%+ same direction), breaks at NY Lunch.
Edge window: ~4h post-release (13:30-17:30 UTC).

Integration: lightweight modifier on Retail Sales release days only (12x/year).
Returns a score adjustment and size multiplier — does NOT veto.

Usage:
    from src.modules.m33_retail_sales import score_m33_retail_sales, format_m33
    status, score_adj, size_mult, details = score_m33_retail_sales(
        wyckoff_phase='MARKUP', vol_regime='CHOP', direction='LONG')
"""

from datetime import datetime, timedelta
import json
import os

# ═══════════════════════════════════════════════════════════════
# US RETAIL SALES RELEASE DATES
# Released at 08:30 ET (13:30 UTC) around 15th of each month
# Format: {release_date: {'retail_sales_mom': float, 'core_mom': float}}
# ═══════════════════════════════════════════════════════════════

RETAIL_SALES_RELEASES = {
    # 2018
    '2018-01-12': {'retail_sales_mom': 0.4, 'core_mom': 0.4},
    '2018-02-14': {'retail_sales_mom': -0.3, 'core_mom': 0.0},
    '2018-03-14': {'retail_sales_mom': -0.1, 'core_mom': 0.1},
    '2018-04-16': {'retail_sales_mom': 0.6, 'core_mom': 0.4},
    '2018-05-15': {'retail_sales_mom': 0.3, 'core_mom': 0.4},
    '2018-06-14': {'retail_sales_mom': 0.8, 'core_mom': 0.5},
    '2018-07-16': {'retail_sales_mom': 0.5, 'core_mom': 0.4},
    '2018-08-15': {'retail_sales_mom': 0.5, 'core_mom': 0.6},
    '2018-09-14': {'retail_sales_mom': 0.1, 'core_mom': 0.1},
    '2018-10-15': {'retail_sales_mom': 0.1, 'core_mom': 0.0},
    '2018-11-15': {'retail_sales_mom': 0.2, 'core_mom': 0.3},
    '2018-12-14': {'retail_sales_mom': 0.2, 'core_mom': 0.2},
    # 2019
    '2019-01-16': {'retail_sales_mom': -1.2, 'core_mom': -1.7},
    '2019-02-14': {'retail_sales_mom': 0.2, 'core_mom': 0.9},
    '2019-03-11': {'retail_sales_mom': -0.2, 'core_mom': -0.4},
    '2019-04-18': {'retail_sales_mom': 1.6, 'core_mom': 1.2},
    '2019-05-15': {'retail_sales_mom': 0.2, 'core_mom': 0.1},
    '2019-06-14': {'retail_sales_mom': 0.5, 'core_mom': 0.5},
    '2019-07-16': {'retail_sales_mom': 0.4, 'core_mom': 0.7},
    '2019-08-15': {'retail_sales_mom': 0.7, 'core_mom': 1.0},
    '2019-09-13': {'retail_sales_mom': 0.4, 'core_mom': 0.0},
    '2019-10-16': {'retail_sales_mom': -0.3, 'core_mom': -0.1},
    '2019-11-15': {'retail_sales_mom': 0.3, 'core_mom': 0.3},
    '2019-12-13': {'retail_sales_mom': 0.2, 'core_mom': 0.1},
    # 2020
    '2020-01-16': {'retail_sales_mom': 0.3, 'core_mom': 0.3},
    '2020-02-14': {'retail_sales_mom': 0.5, 'core_mom': 0.4},
    '2020-03-17': {'retail_sales_mom': -8.3, 'core_mom': -4.0},
    '2020-04-15': {'retail_sales_mom': -14.7, 'core_mom': -11.5},
    '2020-05-15': {'retail_sales_mom': 17.7, 'core_mom': 12.4},
    '2020-06-16': {'retail_sales_mom': 7.5, 'core_mom': 7.3},
    '2020-07-16': {'retail_sales_mom': 1.2, 'core_mom': 1.9},
    '2020-08-14': {'retail_sales_mom': 0.6, 'core_mom': 0.7},
    '2020-09-16': {'retail_sales_mom': 0.6, 'core_mom': 0.7},
    '2020-10-16': {'retail_sales_mom': 0.3, 'core_mom': 0.2},
    '2020-11-17': {'retail_sales_mom': -1.1, 'core_mom': -0.8},
    '2020-12-16': {'retail_sales_mom': -1.4, 'core_mom': -1.3},
    # 2021
    '2021-01-15': {'retail_sales_mom': -0.7, 'core_mom': -1.4},
    '2021-02-17': {'retail_sales_mom': 5.3, 'core_mom': 5.3},
    '2021-03-16': {'retail_sales_mom': -2.7, 'core_mom': -3.2},
    '2021-04-15': {'retail_sales_mom': 9.8, 'core_mom': 8.4},
    '2021-05-14': {'retail_sales_mom': 0.0, 'core_mom': -0.8},
    '2021-06-15': {'retail_sales_mom': -1.7, 'core_mom': -1.0},
    '2021-07-16': {'retail_sales_mom': 0.6, 'core_mom': 1.1},
    '2021-08-17': {'retail_sales_mom': -1.1, 'core_mom': -0.5},
    '2021-09-16': {'retail_sales_mom': 0.7, 'core_mom': 0.8},
    '2021-10-15': {'retail_sales_mom': 0.7, 'core_mom': 0.6},
    '2021-11-16': {'retail_sales_mom': 1.7, 'core_mom': 1.7},
    '2021-12-15': {'retail_sales_mom': 0.3, 'core_mom': 0.3},
    # 2022
    '2022-01-14': {'retail_sales_mom': -1.9, 'core_mom': -2.3},
    '2022-02-16': {'retail_sales_mom': 3.8, 'core_mom': 3.3},
    '2022-03-16': {'retail_sales_mom': 0.3, 'core_mom': 0.2},
    '2022-04-14': {'retail_sales_mom': 0.9, 'core_mom': 0.6},
    '2022-05-17': {'retail_sales_mom': 0.9, 'core_mom': 0.6},
    '2022-06-15': {'retail_sales_mom': -0.3, 'core_mom': 0.5},
    '2022-07-15': {'retail_sales_mom': 1.0, 'core_mom': 0.4},
    '2022-08-17': {'retail_sales_mom': -0.4, 'core_mom': 0.1},
    '2022-09-15': {'retail_sales_mom': 0.4, 'core_mom': 0.0},
    '2022-10-14': {'retail_sales_mom': 1.3, 'core_mom': 0.4},
    '2022-11-16': {'retail_sales_mom': -0.6, 'core_mom': -0.2},
    '2022-12-15': {'retail_sales_mom': -1.1, 'core_mom': -0.7},
    # 2023
    '2023-01-18': {'retail_sales_mom': -1.1, 'core_mom': -0.8},
    '2023-02-15': {'retail_sales_mom': 3.0, 'core_mom': 2.3},
    '2023-03-15': {'retail_sales_mom': -0.4, 'core_mom': -0.1},
    '2023-04-14': {'retail_sales_mom': -1.0, 'core_mom': -0.4},
    '2023-05-16': {'retail_sales_mom': 0.4, 'core_mom': 0.5},
    '2023-06-15': {'retail_sales_mom': 0.3, 'core_mom': 0.2},
    '2023-07-18': {'retail_sales_mom': 0.2, 'core_mom': 0.6},
    '2023-08-15': {'retail_sales_mom': 0.7, 'core_mom': 0.4},
    '2023-09-14': {'retail_sales_mom': 0.6, 'core_mom': 0.3},
    '2023-10-17': {'retail_sales_mom': 0.7, 'core_mom': 0.6},
    '2023-11-15': {'retail_sales_mom': -0.1, 'core_mom': 0.1},
    '2023-12-14': {'retail_sales_mom': 0.3, 'core_mom': 0.2},
    # 2024
    '2024-01-17': {'retail_sales_mom': 0.6, 'core_mom': 0.2},
    '2024-02-15': {'retail_sales_mom': -0.8, 'core_mom': -0.4},
    '2024-03-14': {'retail_sales_mom': 0.6, 'core_mom': 0.3},
    '2024-04-15': {'retail_sales_mom': 0.4, 'core_mom': 0.4},
    '2024-05-15': {'retail_sales_mom': 0.1, 'core_mom': 0.2},
    '2024-06-18': {'retail_sales_mom': 0.1, 'core_mom': 0.4},
    '2024-07-16': {'retail_sales_mom': 0.0, 'core_mom': 0.4},
    '2024-08-15': {'retail_sales_mom': 1.0, 'core_mom': 0.4},
    '2024-09-17': {'retail_sales_mom': 0.4, 'core_mom': 0.5},
    '2024-10-17': {'retail_sales_mom': 0.4, 'core_mom': 0.1},
    '2024-11-15': {'retail_sales_mom': 0.4, 'core_mom': 0.7},
    '2024-12-17': {'retail_sales_mom': 0.4, 'core_mom': 0.4},
    # 2025
    '2025-01-16': {'retail_sales_mom': 0.4, 'core_mom': 0.7},
    '2025-02-14': {'retail_sales_mom': -0.9, 'core_mom': -0.3},
    '2025-03-17': {'retail_sales_mom': 0.2, 'core_mom': 0.3},
    '2025-04-16': {'retail_sales_mom': 1.4, 'core_mom': 0.4},
    '2025-05-15': {'retail_sales_mom': 0.1, 'core_mom': 0.2},
    '2025-06-17': {'retail_sales_mom': -0.1, 'core_mom': 0.3},
    '2025-07-16': {'retail_sales_mom': 0.5, 'core_mom': 0.3},
    '2025-08-15': {'retail_sales_mom': 0.2, 'core_mom': 0.4},
    '2025-09-16': {'retail_sales_mom': 0.3, 'core_mom': 0.2},
    '2025-10-16': {'retail_sales_mom': 0.1, 'core_mom': 0.3},
    '2025-11-14': {'retail_sales_mom': 0.2, 'core_mom': 0.1},
    '2025-12-16': {'retail_sales_mom': 0.3, 'core_mom': 0.2},
    # 2026
    '2026-01-15': {'retail_sales_mom': 0.4, 'core_mom': 0.3},
    '2026-02-13': {'retail_sales_mom': -0.2, 'core_mom': 0.1},
    '2026-03-16': {'retail_sales_mom': 0.3, 'core_mom': 0.2},
    '2026-04-15': {'retail_sales_mom': 0.5, 'core_mom': 0.4},
    '2026-05-14': {'retail_sales_mom': 0.2, 'core_mom': 0.3},
}

# Consensus expectations (MoM %) — approximate
CONSENSUS = {
    '2018-01-12': 0.5, '2018-02-14': 0.2, '2018-03-14': 0.3, '2018-04-16': 0.4,
    '2018-05-15': 0.4, '2018-06-14': 0.4, '2018-07-16': 0.1, '2018-08-15': 0.1,
    '2018-09-14': 0.7, '2018-10-15': 0.6, '2018-11-15': 0.5, '2018-12-14': 0.1,
    '2019-01-16': 0.0, '2019-02-14': 0.0, '2019-03-11': 0.0, '2019-04-18': 1.1,
    '2019-05-15': 0.2, '2019-06-14': 0.6, '2019-07-16': 0.1, '2019-08-15': 0.3,
    '2019-09-13': 0.2, '2019-10-16': 0.3, '2019-11-15': 0.3, '2019-12-13': 0.5,
    '2020-01-16': 0.3, '2020-02-14': 0.2, '2020-03-17': 0.2, '2020-04-15': -8.0,
    '2020-05-15': 8.0, '2020-06-16': 5.4, '2020-07-16': 1.9, '2020-08-14': 1.0,
    '2020-09-16': 0.8, '2020-10-16': 0.5, '2020-11-17': -0.3, '2020-12-16': -0.3,
    '2021-01-15': 0.7, '2021-02-17': 1.0, '2021-03-16': -0.5, '2021-04-15': 5.9,
    '2021-05-14': 0.2, '2021-06-15': -0.4, '2021-07-16': -0.3, '2021-08-17': -0.3,
    '2021-09-16': 0.2, '2021-10-15': 0.3, '2021-11-16': 1.1, '2021-12-15': 0.8,
    '2022-01-14': 0.0, '2022-02-16': 2.1, '2022-03-16': 0.6, '2022-04-14': 0.9,
    '2022-05-17': 0.7, '2022-06-15': 0.2, '2022-07-15': 0.8, '2022-08-17': 0.1,
    '2022-09-15': 0.2, '2022-10-14': 0.3, '2022-11-16': 0.6, '2022-12-15': -0.1,
    '2023-01-18': -0.1, '2023-02-15': 1.9, '2023-03-15': -0.3, '2023-04-14': -0.4,
    '2023-05-16': 0.8, '2023-06-15': 0.1, '2023-07-18': 0.3, '2023-08-15': 0.4,
    '2023-09-14': 0.2, '2023-10-17': 0.3, '2023-11-15': -0.3, '2023-12-14': 0.1,
    '2024-01-17': 0.4, '2024-02-15': -0.1, '2024-03-14': 0.4, '2024-04-15': 0.4,
    '2024-05-15': 0.4, '2024-06-18': 0.3, '2024-07-16': 0.1, '2024-08-15': 0.4,
    '2024-09-17': 0.2, '2024-10-17': 0.3, '2024-11-15': 0.3, '2024-12-17': 0.5,
    '2025-01-16': 0.6, '2025-02-14': -0.1, '2025-03-17': 0.6, '2025-04-16': 1.3,
    '2025-05-15': 0.0, '2025-06-17': 0.3, '2025-07-16': 0.1, '2025-08-15': 0.5,
    '2025-09-16': 0.2, '2025-10-16': 0.3, '2025-11-14': 0.2, '2025-12-16': 0.4,
    '2026-01-15': 0.3, '2026-02-13': 0.3, '2026-03-16': 0.2, '2026-04-15': 0.3,
    '2026-05-14': 0.3,
}


# ═══════════════════════════════════════════════════════════════
# REGIME-CONDITIONAL EDGE TABLE
# Backtested: 101 US Retail Sales releases, 2018-2026, ETH/USDT 15m
# ═══════════════════════════════════════════════════════════════

# Key: (wyckoff_phase, vol_regime, signal)
# Value: (avg_24h_return, win_rate, sample_size, direction_bias)
# Only entries with n >= 3 and |avg_24h| >= 0.5% are included
EDGE_TABLE = {
    # ── LONG EDGES (consumer miss in weak structure → dovish) ──
    ('MARKDOWN', 'LOW_VOL', 'BIG_MISS'):    {'avg_ret': +2.32, 'win': 0.80, 'n': 5, 'bias': 'LONG'},
    ('MARKUP', 'LOW_VOL', 'MISS'):          {'avg_ret': +2.91, 'win': 1.00, 'n': 3, 'bias': 'LONG'},
    ('MARKUP', 'LOW_VOL', 'BIG_MISS'):      {'avg_ret': +4.50, 'win': 1.00, 'n': 3, 'bias': 'LONG'},
    ('RANGE', 'CHOP', 'BEAT'):              {'avg_ret': +3.62, 'win': 1.00, 'n': 3, 'bias': 'LONG'},
    ('MARKDOWN', 'CHOP', 'INLINE'):         {'avg_ret': +3.18, 'win': 1.00, 'n': 3, 'bias': 'LONG'},

    # ── SHORT EDGES (consumer beat in weak structure → hawkish) ──
    ('MARKUP', 'CHOP', 'BIG_MISS'):         {'avg_ret': -3.33, 'win': 0.00, 'n': 5, 'bias': 'SHORT'},
    ('CHOP', 'MARKDOWN', 'BEAT'):           {'avg_ret': -2.96, 'win': 0.25, 'n': 4, 'bias': 'SHORT'},
    ('MARKUP', 'CHOP', 'BEAT'):             {'avg_ret': -0.88, 'win': 0.25, 'n': 4, 'bias': 'SHORT'},
    ('CHOP', 'RANGE', 'STABLE'):            {'avg_ret': -3.95, 'win': 0.00, 'n': 3, 'bias': 'SHORT'},
    ('MARKDOWN', 'CHOP', 'MISS'):           {'avg_ret': -2.73, 'win': 0.33, 'n': 3, 'bias': 'SHORT'},
}

# Broader combos (signal × wyckoff only, ignoring vol) — larger sample
BROAD_EDGE_TABLE = {
    ('MARKDOWN', 'BIG_MISS'):               {'avg_ret': +1.52, 'win': 0.60, 'n': 8, 'bias': 'LONG'},
    ('MARKUP', 'BIG_MISS'):                 {'avg_ret': -0.12, 'win': 0.44, 'n': 9, 'bias': 'NEUTRAL'},
    ('MARKUP', 'BEAT'):                     {'avg_ret': -0.44, 'win': 0.40, 'n': 15, 'bias': 'SHORT'},
    ('MARKDOWN', 'MISS'):                   {'avg_ret': -0.18, 'win': 0.50, 'n': 8, 'bias': 'NEUTRAL'},
    ('RANGE', 'INLINE'):                    {'avg_ret': +0.26, 'win': 0.57, 'n': 7, 'bias': 'NEUTRAL'},
}


def _classify_retail_signal(retail_sales_mom, consensus):
    """Classify retail sales into signal buckets based on surprise."""
    surprise = retail_sales_mom - consensus if consensus is not None else 0

    if surprise > 0.5:
        return 'STRONG_BEAT', surprise
    elif surprise > 0.1:
        return 'BEAT', surprise
    elif surprise < -0.5:
        return 'BIG_MISS', surprise
    elif surprise < -0.1:
        return 'MISS', surprise
    else:
        return 'INLINE', surprise


def _is_retail_release_day(today_str=None, window_days=1):
    """Check if today is within N days of a Retail Sales release.

    Args:
        today_str: YYYY-MM-DD string (default: today UTC)
        window_days: number of days after release to consider active

    Returns:
        (is_release: bool, release_date: str, release_data: dict) or (False, None, None)
    """
    if today_str is None:
        today_str = datetime.utcnow().strftime('%Y-%m-%d')

    today = datetime.strptime(today_str, '%Y-%m-%d')

    for release_date_str, release_data in sorted(RETAIL_SALES_RELEASES.items(), reverse=True):
        release_dt = datetime.strptime(release_date_str, '%Y-%m-%d')
        days_since = (today - release_dt).days
        if 0 <= days_since <= window_days:
            return True, release_date_str, release_data

    return False, None, None


def score_m33_retail_sales(wyckoff_phase='RANGE', vol_regime='CHOP',
                           direction='LONG', today_str=None, config=None):
    """Score the US Retail Sales session bias.

    Args:
        wyckoff_phase: from M21 ('ACCUMULATION', 'MARKUP', 'DISTRIBUTION', 'MARKDOWN', 'RANGE')
        vol_regime: from M9 ('TREND', 'SQUEEZE', 'CHOP', 'LOW_VOL')
        direction: trade direction ('LONG' or 'SHORT')
        today_str: YYYY-MM-DD override (for backtesting)
        config: config dict (optional)

    Returns:
        status: 'PASS' (active), 'SKIP' (not release day), or 'WEAK' (low confidence)
        score_adj: score adjustment (-0.10 to +0.10)
        size_mult: position size multiplier (0.5 to 1.0)
        details: dict
    """
    cfg = config or {}

    if not cfg.get('M33_ENABLED', True):
        return 'SKIP', 0.0, 1.0, {'regime': 'DISABLED'}

    # Check if today is a Retail Sales release day
    is_release, release_date, release_data = _is_retail_release_day(
        today_str, window_days=cfg.get('M33_WINDOW_DAYS', 1))

    if not is_release:
        return 'SKIP', 0.0, 1.0, {'regime': 'NOT_RELEASE_DAY'}

    retail_mom = release_data['retail_sales_mom']
    core_mom = release_data['core_mom']
    consensus = CONSENSUS.get(release_date, 0.0)

    signal, surprise = _classify_retail_signal(retail_mom, consensus)

    # ── Lookup 1: Fine-grained (Wyckoff + Vol + Signal) ──
    fine_key = (wyckoff_phase, vol_regime, signal)
    fine_match = EDGE_TABLE.get(fine_key)

    # ── Lookup 2: Broad (Wyckoff + Signal) ──
    broad_key = (wyckoff_phase, signal)
    broad_match = BROAD_EDGE_TABLE.get(broad_key)

    # ── Determine best signal ──
    best_match = None
    best_source = 'NONE'
    confidence = 0.0

    if fine_match and fine_match['n'] >= 3:
        best_match = fine_match
        best_source = 'FINE'
        confidence = min(1.0, fine_match['n'] / 10)
    elif broad_match and broad_match['n'] >= 5 and broad_match.get('bias') != 'NEUTRAL':
        best_match = broad_match
        best_source = 'BROAD'
        confidence = min(1.0, broad_match['n'] / 15)

    if best_match is None:
        return 'SKIP', 0.0, 1.0, {
            'regime': 'NO_EDGE',
            'wyckoff': wyckoff_phase,
            'vol_regime': vol_regime,
            'signal': signal,
            'surprise': surprise,
            'retail_mom': retail_mom,
            'consensus': consensus,
            'release_date': release_date,
        }

    # ── Compute score adjustment ──
    avg_ret = best_match['avg_ret']
    win_rate = best_match['win']
    n = best_match['n']
    bias = best_match['bias']

    abs_ret = abs(avg_ret)
    if abs_ret >= 2.0:
        raw_adj = 0.10
    elif abs_ret >= 1.0:
        raw_adj = 0.07
    else:
        raw_adj = 0.05

    score_adj = raw_adj * confidence

    # Direction alignment
    if bias == 'LONG' and direction == 'LONG':
        score_adj = abs(score_adj)
    elif bias == 'LONG' and direction == 'SHORT':
        score_adj = -abs(score_adj)
    elif bias == 'SHORT' and direction == 'SHORT':
        score_adj = abs(score_adj)
    elif bias == 'SHORT' and direction == 'LONG':
        score_adj = -abs(score_adj)

    # ── Size multiplier ──
    if confidence >= 0.7:
        size_mult = 1.0
    elif confidence >= 0.4:
        size_mult = 0.75
    else:
        size_mult = 0.50

    if n < 5:
        size_mult *= 0.75

    size_mult = round(size_mult, 2)
    score_adj = round(score_adj, 3)

    status = 'PASS' if confidence >= 0.3 else 'WEAK'

    details = {
        'regime': f'RETAIL_SALES_{bias}',
        'release_date': release_date,
        'retail_mom': retail_mom,
        'core_mom': core_mom,
        'consensus': consensus,
        'surprise': round(surprise, 2),
        'signal': signal,
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


def format_m33(details):
    """Format M33 details for terminal output."""
    if not details or details.get('regime') in ('DISABLED', 'NOT_RELEASE_DAY', 'NO_EDGE'):
        regime = details.get('regime', '?') if details else '?'
        if regime == 'NOT_RELEASE_DAY':
            return ''
        return ''

    bias = details.get('bias', '?')
    retail_mom = details.get('retail_mom', 0)
    core_mom = details.get('core_mom', 0)
    consensus = details.get('consensus', 0)
    surprise = details.get('surprise', 0)
    signal = details.get('signal', '?')
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
    surp_icon = '🟢' if surprise > 0.1 else '🔴' if surprise < -0.1 else '⚪'

    lines = []
    lines.append(f"\n  {icon} M33 US RETAIL SALES SESSION BIAS: {bias}")
    lines.append(f"    Release: {release}  |  Retail MoM: {retail_mom:+.1f}%  |  Core: {core_mom:+.1f}%  |  Consensus: {consensus:+.1f}%")
    lines.append(f"    Surprise: {surp_icon} {surprise:+.2f}  Signal: {signal}")
    lines.append(f"    Context: {wyckoff} + {vol}")
    lines.append(f"    Backtest: avg 24h={avg_ret:+.2f}%  win={win*100:.0f}%  n={n}  source={source}")
    lines.append(f"    {conf_icon} Confidence: {conf:.2f}  |  Score adj: {score_adj:+.3f}  |  Size: {size_mult:.2f}x")

    return '\n'.join(lines)


def get_retail_sales_cache_path():
    """Get path to Retail Sales cache (for live updates)."""
    cache_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), 'data', 'macro')
    return os.path.join(cache_dir, 'retail_sales_cache.json')


def load_retail_sales_cache():
    """Load cached Retail Sales data (for live updates from macro_fetch)."""
    cache_path = get_retail_sales_cache_path()
    if os.path.exists(cache_path):
        try:
            with open(cache_path) as f:
                return json.load(f)
        except (json.JSONDecodeError, IOError):
            pass
    return {}


def update_retail_sales_cache(retail_mom, core_mom=None, release_date=None):
    """Update Retail Sales cache with new data (called from macro_fetch)."""
    cache = load_retail_sales_cache()
    if release_date is None:
        release_date = datetime.utcnow().strftime('%Y-%m-%d')
    cache[release_date] = {
        'retail_mom': retail_mom,
        'core_mom': core_mom,
        'updated': datetime.utcnow().isoformat(),
    }
    cache_path = get_retail_sales_cache_path()
    os.makedirs(os.path.dirname(cache_path), exist_ok=True)
    with open(cache_path, 'w') as f:
        json.dump(cache, f, indent=2)
    return cache

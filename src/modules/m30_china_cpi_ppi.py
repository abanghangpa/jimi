"""
M30: China CPI + PPI Session Bias (Regime-Conditional)

On NBS CPI+PPI release days (~10th-15th of each month, 09:30 CST = 01:30 UTC),
applies a directional bias based on the combination of:
  - Wyckoff phase (M21): ACCUMULATION / MARKUP / DISTRIBUTION / MARKDOWN / RANGE
  - Volatility regime (M9): TREND / SQUEEZE / CHOP / LOW_VOL
  - CPI+PPI signal: SEVERE_DEFLATION / DEFLATION / DISINFLATION / STABLE / INFLATION

Thesis (from user #10):
  Asia Open & Local Shock → Commodity Market Open → Europe Open → Asia Re-open
  Severe deflation (negative CPI/PPI) → global industrial demand dead → ETH drops
  Then commodity desks dump WTI/Copper → European desks interpret as disinflationary
  PBoC LPR cut probability spikes → local whales accumulate spot ETH

Backtested on 101 China CPI+PPI releases (2018-2026) against ETH/USDT 15m data.

Key findings:
  INFLATION + CHOP + BEAT:   +3.303% avg, 70% win, n=10  → LONG bias
  DISINFLATION + LOW_VOL + MISS: -2.694% avg, 18% win, n=11 → SHORT bias
  SEVERE_DEFLATION + SQUEEZE + MISS: +6.899% avg, 100% win, n=3 → LONG (PBoC stimulus)
  SEVERE_DEFLATION + CHOP + MISS: +2.615% avg, 67% win, n=3 → LONG

  Transmission chain (weak — most transitions <55%):
    Release Spike → Sydney Open: 83% (structural, not informative)
    Pre-London → Frankfurt Open: 88% (structural, same hour)
    London-NY Overlap → NY AM: 75% (real edge)

  Overall: 0.012% avg, 48.5% win — NOT significant (p=0.98)
  China CPI+PPI is NOISE at the 24h level. Only specific combos have edge.

Integration: lightweight modifier on NBS release days only (~12x/year).
Returns a score adjustment and size multiplier — does NOT veto.

Usage:
    from src.modules.m30_china_cpi_ppi import score_m30_china_cpi, format_m30
    status, score_adj, size_mult, details = score_m30_china_cpi(
        wyckoff_phase='RANGE', vol_regime='CHOP', direction='LONG')
"""

from datetime import datetime, timedelta
import json
import os


# ═══════════════════════════════════════════════════════════════
# CHINA CPI + PPI RELEASE DATES (09:30 CST = 01:30 UTC)
# NBS releases CPI and PPI simultaneously
# ═══════════════════════════════════════════════════════════════

CHINA_RELEASES = {
    # 2018
    '2018-01-10': {'cpi_yoy': 1.5, 'ppi_yoy': 4.3},
    '2018-02-09': {'cpi_yoy': 2.9, 'ppi_yoy': 4.3},
    '2018-03-09': {'cpi_yoy': 2.9, 'ppi_yoy': 3.7},
    '2018-04-11': {'cpi_yoy': 2.1, 'ppi_yoy': 3.1},
    '2018-05-10': {'cpi_yoy': 1.8, 'ppi_yoy': 3.4},
    '2018-06-09': {'cpi_yoy': 1.9, 'ppi_yoy': 4.7},
    '2018-07-10': {'cpi_yoy': 2.1, 'ppi_yoy': 4.6},
    '2018-08-09': {'cpi_yoy': 2.1, 'ppi_yoy': 4.6},
    '2018-09-10': {'cpi_yoy': 2.3, 'ppi_yoy': 3.6},
    '2018-10-16': {'cpi_yoy': 2.5, 'ppi_yoy': 3.3},
    '2018-11-09': {'cpi_yoy': 2.2, 'ppi_yoy': 2.7},
    '2018-12-09': {'cpi_yoy': 2.2, 'ppi_yoy': 0.9},
    # 2019
    '2019-01-10': {'cpi_yoy': 1.9, 'ppi_yoy': 0.1},
    '2019-02-15': {'cpi_yoy': 1.5, 'ppi_yoy': 0.1},
    '2019-03-09': {'cpi_yoy': 1.5, 'ppi_yoy': 0.1},
    '2019-04-11': {'cpi_yoy': 2.3, 'ppi_yoy': 0.4},
    '2019-05-09': {'cpi_yoy': 2.5, 'ppi_yoy': 0.9},
    '2019-06-12': {'cpi_yoy': 2.7, 'ppi_yoy': 0.6},
    '2019-07-10': {'cpi_yoy': 2.8, 'ppi_yoy': -0.3},
    '2019-08-09': {'cpi_yoy': 2.8, 'ppi_yoy': -0.8},
    '2019-09-10': {'cpi_yoy': 3.0, 'ppi_yoy': -1.2},
    '2019-10-15': {'cpi_yoy': 3.8, 'ppi_yoy': -1.6},
    '2019-11-09': {'cpi_yoy': 3.8, 'ppi_yoy': -1.4},
    '2019-12-10': {'cpi_yoy': 4.5, 'ppi_yoy': -1.4},
    # 2020
    '2020-01-09': {'cpi_yoy': 4.5, 'ppi_yoy': -0.5},
    '2020-02-10': {'cpi_yoy': 5.2, 'ppi_yoy': -0.3},
    '2020-03-10': {'cpi_yoy': 5.2, 'ppi_yoy': -0.4},
    '2020-04-10': {'cpi_yoy': 3.3, 'ppi_yoy': -3.1},
    '2020-05-12': {'cpi_yoy': 2.4, 'ppi_yoy': -3.7},
    '2020-06-10': {'cpi_yoy': 2.5, 'ppi_yoy': -3.0},
    '2020-07-09': {'cpi_yoy': 2.7, 'ppi_yoy': -2.4},
    '2020-08-10': {'cpi_yoy': 2.4, 'ppi_yoy': -2.0},
    '2020-09-09': {'cpi_yoy': 1.7, 'ppi_yoy': -2.1},
    '2020-10-15': {'cpi_yoy': 0.5, 'ppi_yoy': -2.1},
    '2020-11-10': {'cpi_yoy': -0.5, 'ppi_yoy': -1.5},
    '2020-12-09': {'cpi_yoy': 0.2, 'ppi_yoy': -1.1},
    # 2021
    '2021-01-11': {'cpi_yoy': -0.3, 'ppi_yoy': 0.3},
    '2021-02-10': {'cpi_yoy': -0.2, 'ppi_yoy': 1.7},
    '2021-03-10': {'cpi_yoy': 0.4, 'ppi_yoy': 4.4},
    '2021-04-09': {'cpi_yoy': 0.9, 'ppi_yoy': 6.8},
    '2021-05-11': {'cpi_yoy': 1.3, 'ppi_yoy': 9.0},
    '2021-06-09': {'cpi_yoy': 1.1, 'ppi_yoy': 8.8},
    '2021-07-09': {'cpi_yoy': 1.0, 'ppi_yoy': 9.0},
    '2021-08-09': {'cpi_yoy': 0.8, 'ppi_yoy': 9.5},
    '2021-09-09': {'cpi_yoy': 0.7, 'ppi_yoy': 10.7},
    '2021-10-14': {'cpi_yoy': 1.5, 'ppi_yoy': 13.5},
    '2021-11-10': {'cpi_yoy': 2.3, 'ppi_yoy': 12.9},
    '2021-12-09': {'cpi_yoy': 1.5, 'ppi_yoy': 10.3},
    # 2022
    '2022-01-12': {'cpi_yoy': 0.9, 'ppi_yoy': 9.1},
    '2022-02-16': {'cpi_yoy': 0.9, 'ppi_yoy': 8.8},
    '2022-03-09': {'cpi_yoy': 0.9, 'ppi_yoy': 8.8},
    '2022-04-11': {'cpi_yoy': 1.5, 'ppi_yoy': 8.0},
    '2022-05-11': {'cpi_yoy': 2.1, 'ppi_yoy': 6.4},
    '2022-06-10': {'cpi_yoy': 2.1, 'ppi_yoy': 6.1},
    '2022-07-09': {'cpi_yoy': 2.7, 'ppi_yoy': 4.2},
    '2022-08-10': {'cpi_yoy': 2.5, 'ppi_yoy': 2.3},
    '2022-09-09': {'cpi_yoy': 2.8, 'ppi_yoy': 0.9},
    '2022-10-14': {'cpi_yoy': 2.1, 'ppi_yoy': -1.3},
    '2022-11-09': {'cpi_yoy': 1.6, 'ppi_yoy': -1.3},
    '2022-12-09': {'cpi_yoy': 1.8, 'ppi_yoy': -0.7},
    # 2023
    '2023-01-12': {'cpi_yoy': 1.8, 'ppi_yoy': -0.8},
    '2023-02-10': {'cpi_yoy': 1.0, 'ppi_yoy': -1.4},
    '2023-03-09': {'cpi_yoy': 1.0, 'ppi_yoy': -2.5},
    '2023-04-11': {'cpi_yoy': 0.1, 'ppi_yoy': -3.6},
    '2023-05-11': {'cpi_yoy': 0.2, 'ppi_yoy': -4.6},
    '2023-06-09': {'cpi_yoy': 0.0, 'ppi_yoy': -5.4},
    '2023-07-10': {'cpi_yoy': -0.3, 'ppi_yoy': -5.4},
    '2023-08-09': {'cpi_yoy': 0.1, 'ppi_yoy': -4.4},
    '2023-09-09': {'cpi_yoy': 0.0, 'ppi_yoy': -3.0},
    '2023-10-13': {'cpi_yoy': -0.2, 'ppi_yoy': -2.6},
    '2023-11-09': {'cpi_yoy': -0.2, 'ppi_yoy': -3.0},
    '2023-12-09': {'cpi_yoy': -0.3, 'ppi_yoy': -2.7},
    # 2024
    '2024-01-12': {'cpi_yoy': -0.8, 'ppi_yoy': -2.7},
    '2024-02-08': {'cpi_yoy': 0.7, 'ppi_yoy': -2.7},
    '2024-03-09': {'cpi_yoy': 0.7, 'ppi_yoy': -2.7},
    '2024-04-11': {'cpi_yoy': 0.1, 'ppi_yoy': -2.5},
    '2024-05-11': {'cpi_yoy': 0.3, 'ppi_yoy': -2.5},
    '2024-06-12': {'cpi_yoy': 0.2, 'ppi_yoy': -0.8},
    '2024-07-10': {'cpi_yoy': 0.5, 'ppi_yoy': -0.8},
    '2024-08-09': {'cpi_yoy': 0.6, 'ppi_yoy': -0.8},
    '2024-09-09': {'cpi_yoy': 0.4, 'ppi_yoy': -1.8},
    '2024-10-13': {'cpi_yoy': 0.3, 'ppi_yoy': -2.8},
    '2024-11-09': {'cpi_yoy': 0.2, 'ppi_yoy': -2.9},
    '2024-12-09': {'cpi_yoy': 0.2, 'ppi_yoy': -2.5},
    # 2025
    '2025-01-09': {'cpi_yoy': 0.5, 'ppi_yoy': -2.3},
    '2025-02-09': {'cpi_yoy': -0.7, 'ppi_yoy': -2.3},
    '2025-03-09': {'cpi_yoy': -0.7, 'ppi_yoy': -2.2},
    '2025-04-10': {'cpi_yoy': -0.1, 'ppi_yoy': -2.7},
    '2025-05-10': {'cpi_yoy': -0.1, 'ppi_yoy': -2.7},
    '2025-06-09': {'cpi_yoy': -0.2, 'ppi_yoy': -3.0},
    '2025-07-09': {'cpi_yoy': 0.0, 'ppi_yoy': -3.6},
    '2025-08-09': {'cpi_yoy': 0.0, 'ppi_yoy': -3.6},
    '2025-09-10': {'cpi_yoy': 0.0, 'ppi_yoy': -2.8},
    '2025-10-15': {'cpi_yoy': 0.2, 'ppi_yoy': -2.1},
    '2025-11-09': {'cpi_yoy': 0.2, 'ppi_yoy': -2.1},
    '2025-12-09': {'cpi_yoy': 0.3, 'ppi_yoy': -1.8},
    # 2026
    '2026-01-09': {'cpi_yoy': 0.5, 'ppi_yoy': -1.5},
    '2026-02-09': {'cpi_yoy': -0.7, 'ppi_yoy': -2.2},
    '2026-03-09': {'cpi_yoy': -0.1, 'ppi_yoy': -2.3},
    '2026-04-10': {'cpi_yoy': -0.1, 'ppi_yoy': -2.7},
    '2026-05-10': {'cpi_yoy': 0.0, 'ppi_yoy': -2.5},
}


# ═══════════════════════════════════════════════════════════════
# SESSION-CONDITIONAL EDGE TABLE
# Backtested: 101 China CPI+PPI releases, 2018-2026, ETH/USDT 15m
# ═══════════════════════════════════════════════════════════════

EDGE_TABLE = {
    'SEVERE_DEFLATION': {
        'n': 13,
        'avg_24h': +1.431,
        'win_rate': 0.69,
        'session_returns': {
            'release_1h': -0.15, 'sydney_open': +0.35, 'tokyo_open': +0.20,
            'asia_mid': +0.45, 'asia_afternoon': -0.10, 'tokyo_close': +0.15,
            'pre_london': -0.05, 'london_open': +0.10, 'london_morning': +0.30,
            'london_midday': +0.15, 'ny_pre_open': -0.10, 'ny_open': +0.05,
            'ny_am': +0.20, 'ny_lunch': -0.05, 'ny_pm': -0.10,
        },
        'bias': 'LONG',  # PBoC stimulus expectation
        'confidence': 0.65,
        'desc': 'Severe deflation → PBoC stimulus expected → whales accumulate',
    },
    'DEFLATION': {
        'n': 27,
        'avg_24h': +0.341,
        'win_rate': 0.44,
        'session_returns': {
            'release_1h': -0.10, 'sydney_open': +0.15, 'tokyo_open': -0.05,
            'asia_mid': +0.20, 'london_open': -0.10, 'london_morning': +0.15,
            'ny_am': -0.10, 'ny_pm': +0.05,
        },
        'bias': 'NEUTRAL',
        'confidence': 0.30,
        'desc': 'Deflation — mixed signals, commodity desks react',
    },
    'DISINFLATION': {
        'n': 31,
        'avg_24h': -1.545,
        'win_rate': 0.39,
        'session_returns': {
            'release_1h': -0.20, 'sydney_open': -0.15, 'tokyo_open': -0.30,
            'asia_mid': -0.25, 'asia_afternoon': -0.10, 'london_open': -0.15,
            'london_morning': -0.20, 'ny_am': -0.15, 'ny_pm': -0.10,
        },
        'bias': 'SHORT',
        'confidence': 0.55,
        'desc': 'Disinflation + PPI falling → global industrial demand dead',
    },
    'STABLE': {
        'n': 3,
        'avg_24h': +2.634,
        'win_rate': 0.33,
        'session_returns': {},
        'bias': 'NEUTRAL',
        'confidence': 0.20,
        'desc': 'Stable — too few samples',
    },
    'INFLATION': {
        'n': 27,
        'avg_24h': +0.495,
        'win_rate': 0.56,
        'session_returns': {
            'release_1h': +0.10, 'sydney_open': +0.20, 'tokyo_open': +0.05,
            'asia_mid': +0.15, 'london_open': +0.10, 'london_morning': +0.05,
            'ny_am': -0.05, 'ny_pm': +0.10,
        },
        'bias': 'NEUTRAL',
        'confidence': 0.35,
        'desc': 'Inflation — noisy, depends on commodity context',
    },
}

# Cross-tabulation edges (signal × vol_regime × direction) — from backtest
CROSS_TAB_EDGES = {
    ('INFLATION', 'CHOP', 'BEAT'):      {'avg_ret': +3.303, 'win': 0.70, 'n': 10, 'bias': 'LONG'},
    ('DISINFLATION', 'LOW_VOL', 'MISS'): {'avg_ret': -2.694, 'win': 0.18, 'n': 11, 'bias': 'SHORT'},
    ('SEVERE_DEFLATION', 'SQUEEZE', 'MISS'): {'avg_ret': +6.899, 'win': 1.00, 'n': 3, 'bias': 'LONG'},
    ('SEVERE_DEFLATION', 'CHOP', 'MISS'): {'avg_ret': +2.615, 'win': 0.67, 'n': 3, 'bias': 'LONG'},
    ('DEFLATION', 'SQUEEZE', 'MISS'):   {'avg_ret': -1.563, 'win': 0.25, 'n': 4, 'bias': 'SHORT'},
    ('DEFLATION', 'SQUEEZE', 'BEAT'):   {'avg_ret': -0.901, 'win': 0.25, 'n': 4, 'bias': 'SHORT'},
    ('INFLATION', 'CHOP', 'MISS'):      {'avg_ret': -1.308, 'win': 0.50, 'n': 6, 'bias': 'SHORT'},
    ('INFLATION', 'LOW_VOL', 'BEAT'):   {'avg_ret': -2.719, 'win': 0.33, 'n': 3, 'bias': 'SHORT'},
}

# Session transmission chain (from backtest — mostly broken)
TRANSMISSION_CHAIN = {
    'release_to_sydney': 83.0,     # structural
    'pre_london_to_frankfurt': 88.0,  # structural (same hour)
    'london_ny_overlap_to_ny_am': 75.0,  # real edge
}

# Asia fade rate (China data released during Asia session)
ASIA_FADE_RATE = 0.42  # 42% same direction — chain breaks


# ═══════════════════════════════════════════════════════════════
# HELPERS
# ═══════════════════════════════════════════════════════════════

def _classify_signal(cpi_yoy, ppi_yoy):
    if ppi_yoy is not None and ppi_yoy < -2.0:
        if cpi_yoy is not None and cpi_yoy < 0:
            return 'SEVERE_DEFLATION'
        return 'DEFLATION'
    elif ppi_yoy is not None and ppi_yoy < 0:
        if cpi_yoy is not None and cpi_yoy < 0:
            return 'DEFLATION'
        return 'DISINFLATION'
    elif ppi_yoy is not None and ppi_yoy > 3.0:
        return 'INFLATION'
    elif ppi_yoy is not None and ppi_yoy > 1.0:
        return 'STABLE'
    return 'DISINFLATION'


def _classify_direction(cpi_yoy, ppi_yoy, prev_ppi=None):
    if prev_ppi is not None:
        delta = ppi_yoy - prev_ppi
        if delta > 0.3:
            return 'BEAT'
        elif delta < -0.3:
            return 'MISS'
    if ppi_yoy > 0:
        return 'BEAT'
    elif ppi_yoy < -1.0:
        return 'MISS'
    return 'INLINE'


def _is_china_release_day(today_str=None, window_days=1):
    if today_str is None:
        today_str = datetime.utcnow().strftime('%Y-%m-%d')
    today = datetime.strptime(today_str, '%Y-%m-%d')
    for release_date_str in sorted(CHINA_RELEASES.keys(), reverse=True):
        release_dt = datetime.strptime(release_date_str, '%Y-%m-%d')
        days_since = (today - release_dt).days
        if 0 <= days_since <= window_days:
            return True, release_date_str, CHINA_RELEASES[release_date_str]
    return False, None, None


def score_m30_china_cpi(wyckoff_phase='RANGE', vol_regime='CHOP',
                        direction='LONG', today_str=None, config=None,
                        now_utc=None):
    """Score China CPI+PPI session bias.

    Args:
        wyckoff_phase: from M21
        vol_regime: from M9
        direction: trade direction ('LONG' or 'SHORT')
        today_str: YYYY-MM-DD override
        config: config dict (optional)
        now_utc: datetime UTC override

    Returns:
        status: 'PASS', 'SKIP', or 'WEAK'
        score_adj: score adjustment (-0.10 to +0.10)
        size_mult: position size multiplier (0.5 to 1.0)
        details: dict
    """
    cfg = config or {}

    if not cfg.get('M30_ENABLED', True):
        return 'SKIP', 0.0, 1.0, {'regime': 'DISABLED'}

    is_release, release_date, release_data = _is_china_release_day(
        today_str, window_days=cfg.get('M30_WINDOW_DAYS', 1))

    if not is_release:
        return 'SKIP', 0.0, 1.0, {'regime': 'NOT_RELEASE_DAY'}

    cpi_yoy = release_data['cpi_yoy']
    ppi_yoy = release_data['ppi_yoy']

    # Previous month for direction
    sorted_dates = sorted(CHINA_RELEASES.keys())
    idx = sorted_dates.index(release_date) if release_date in sorted_dates else -1
    prev_ppi = CHINA_RELEASES[sorted_dates[idx-1]]['ppi_yoy'] if idx > 0 else None

    signal = _classify_signal(cpi_yoy, ppi_yoy)
    surprise = _classify_direction(cpi_yoy, ppi_yoy, prev_ppi)

    # ── Lookup 1: Cross-tab edge (signal × vol × direction) ──
    cross_key = (signal, vol_regime, surprise)
    cross_edge = CROSS_TAB_EDGES.get(cross_key)

    # ── Lookup 2: Signal edge ──
    edge = EDGE_TABLE.get(signal)

    best_match = None
    best_source = 'NONE'
    confidence = 0.0

    if cross_edge and cross_edge['n'] >= 3:
        best_match = cross_edge
        best_source = 'CROSS_TAB'
        confidence = min(1.0, cross_edge['n'] / 8)
    elif edge and edge.get('confidence', 0) >= 0.45 and edge.get('bias') != 'NEUTRAL':
        best_match = {
            'avg_ret': edge.get('avg_24h', 0),
            'win': edge.get('win_rate', 0),
            'n': edge['n'],
            'bias': edge['bias'],
        }
        best_source = 'SIGNAL'
        confidence = edge['confidence']

    if best_match is None or best_match.get('bias') == 'NEUTRAL':
        return 'SKIP', 0.0, 1.0, {
            'regime': 'NO_EDGE', 'signal': signal,
            'cpi_yoy': cpi_yoy, 'ppi_yoy': ppi_yoy, 'release_date': release_date,
        }

    # ── Score adjustment ──
    avg_ret = best_match.get('avg_ret', 0)
    abs_ret = abs(avg_ret)
    if abs_ret >= 2.0:
        raw_adj = 0.10
    elif abs_ret >= 1.0:
        raw_adj = 0.07
    else:
        raw_adj = 0.05

    score_adj = raw_adj * confidence
    bias = best_match.get('bias', 'NEUTRAL')
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
    if best_match.get('n', 0) < 5:
        size_mult *= 0.75

    size_mult = round(size_mult, 2)
    score_adj = round(score_adj, 3)
    status = 'PASS' if confidence >= 0.3 else 'WEAK'

    details = {
        'regime': f'CHINA_CPI_{bias}',
        'release_date': release_date,
        'cpi_yoy': cpi_yoy,
        'ppi_yoy': ppi_yoy,
        'signal': signal,
        'surprise': surprise,
        'bias': bias,
        'avg_ret': avg_ret,
        'win_rate': best_match.get('win', 0),
        'sample_size': best_match.get('n', 0),
        'confidence': round(confidence, 2),
        'source': best_source,
        'score_adj': score_adj,
        'size_mult': size_mult,
        'edge_desc': edge.get('desc', '') if edge else '',
        'chain': TRANSMISSION_CHAIN,
    }

    return status, score_adj, size_mult, details


def format_m30(details):
    if not details or details.get('regime') in ('DISABLED', 'NOT_RELEASE_DAY', 'NO_EDGE'):
        regime = details.get('regime', '?') if details else '?'
        if regime == 'NOT_RELEASE_DAY':
            return ''
        return ''

    bias = details.get('bias', '?')
    cpi = details.get('cpi_yoy', 0)
    ppi = details.get('ppi_yoy', 0)
    signal = details.get('signal', '?')
    surprise = details.get('surprise', '?')
    release = details.get('release_date', '?')
    avg_ret = details.get('avg_ret', 0)
    win = details.get('win_rate', 0)
    n = details.get('sample_size', 0)
    conf = details.get('confidence', 0)
    source = details.get('source', '?')
    score_adj = details.get('score_adj', 0)
    size_mult = details.get('size_mult', 1.0)
    desc = details.get('edge_desc', '')

    icon = '🟢' if bias == 'LONG' else '🔴' if bias == 'SHORT' else '⚪'
    conf_icon = '🟢' if conf >= 0.7 else '🟡' if conf >= 0.4 else '🟠'

    lines = []
    lines.append(f"\n  {icon} M30 CHINA CPI+PPI: {bias}")
    lines.append(f"    Release: {release}  |  CPI: {cpi:+.1f}%  |  PPI: {ppi:+.1f}%  |  Signal: {signal}")
    lines.append(f"    Surprise: {surprise}  |  Source: {source}")
    lines.append(f"    Backtest: avg={avg_ret:+.2f}%  win={win*100:.0f}%  n={n}")
    lines.append(f"    {conf_icon} Confidence: {conf:.2f}  |  Score adj: {score_adj:+.3f}  |  Size: {size_mult:.2f}x")
    if desc:
        lines.append(f"    📖 {desc}")

    chain = details.get('chain', {})
    if chain:
        lines.append(f"    📊 Chain: Ldn-NY→NY AM {chain.get('london_ny_overlap_to_ny_am', 0):.0f}% (only real edge)")

    return '\n'.join(lines)


def get_china_cpi_cache_path():
    cache_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), 'data', 'macro')
    return os.path.join(cache_dir, 'china_cpi_cache.json')


def load_china_cpi_cache():
    cache_path = get_china_cpi_cache_path()
    if os.path.exists(cache_path):
        try:
            with open(cache_path) as f:
                return json.load(f)
        except (json.JSONDecodeError, IOError):
            pass
    return {}


def update_china_cpi_cache(cpi_yoy, ppi_yoy, release_date=None):
    cache = load_china_cpi_cache()
    if release_date is None:
        release_date = datetime.utcnow().strftime('%Y-%m-%d')
    cache[release_date] = {
        'cpi_yoy': cpi_yoy,
        'ppi_yoy': ppi_yoy,
        'updated': datetime.utcnow().isoformat(),
    }
    cache_path = get_china_cpi_cache_path()
    os.makedirs(os.path.dirname(cache_path), exist_ok=True)
    with open(cache_path, 'w') as f:
        json.dump(cache, f, indent=2)
    return cache

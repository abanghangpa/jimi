"""
M31: UK CPI Session Bias (Regime-Conditional)

On ONS CPI release days (~10th-20th of each month, 07:00 UTC),
applies a directional bias based on the combination of:
  - Wyckoff phase (M21)
  - Volatility regime (M9)
  - UK Services CPI signal (BoE's key metric)

Thesis (from user #13):
  Europe Morning → US Session → UK Central Bank Open
  Hot Services CPI → GBP surges → DXY deflates → ETH mechanical bounce
  Hot UK CPI → US desks see global wage-price spiral → halts dovish positioning
  Data → BoE MPC Vote split → hawkish dissent controls regional liquidity

Backtested on 101 UK CPI releases (2018-2026) against ETH/USDT 15m data.

Key findings:
  WARM_SERVICES + LOW_VOL + MISS: +1.165% avg, 67% win, n=6 → LONG
  HOT_SERVICES + SQUEEZE + MISS:  -1.943% avg, 0% win, n=5 → SHORT
  HOT_SERVICES + CHOP + MISS:     -3.116% avg, 0% win, n=3 → SHORT
  COLD_SERVICES + CHOP + BEAT:    +0.948% avg, 53% win, n=17 → LONG (marginal)

  Transmission chain (weak):
    London-NY Overlap → NY AM: 79% (real edge)
    London Midday → NY Pre-Open: 63% (marginal)

  Overall: -0.089% avg, 45.5% win — NOT significant (p=0.85)
  UK CPI is NOISE at the 24h level. Only specific combos have edge.

Integration: lightweight modifier on ONS release days only (~12x/year).
Returns a score adjustment and size multiplier — does NOT veto.

Usage:
    from src.modules.m31_uk_cpi import score_m31_uk_cpi, format_m31
    status, score_adj, size_mult, details = score_m31_uk_cpi(
        wyckoff_phase='RANGE', vol_regime='CHOP', direction='LONG')
"""

from datetime import datetime, timedelta
import json
import os


# ═══════════════════════════════════════════════════════════════
# UK CPI RELEASE DATES + ACTUAL VALUES (07:00 UTC)
# ONS releases CPI and Services CPI together
# Services CPI is the BoE's key metric
# ═══════════════════════════════════════════════════════════════

UK_RELEASES = {
    # 2018
    '2018-01-16': {'cpi_yoy': 3.0, 'services_yoy': 2.8},
    '2018-02-13': {'cpi_yoy': 3.0, 'services_yoy': 2.8},
    '2018-03-20': {'cpi_yoy': 2.7, 'services_yoy': 2.6},
    '2018-04-18': {'cpi_yoy': 2.5, 'services_yoy': 2.5},
    '2018-05-23': {'cpi_yoy': 2.4, 'services_yoy': 2.5},
    '2018-06-13': {'cpi_yoy': 2.4, 'services_yoy': 2.5},
    '2018-07-18': {'cpi_yoy': 2.4, 'services_yoy': 2.4},
    '2018-08-15': {'cpi_yoy': 2.5, 'services_yoy': 2.5},
    '2018-09-19': {'cpi_yoy': 2.4, 'services_yoy': 2.5},
    '2018-10-17': {'cpi_yoy': 2.4, 'services_yoy': 2.4},
    '2018-11-14': {'cpi_yoy': 2.4, 'services_yoy': 2.4},
    '2018-12-19': {'cpi_yoy': 2.3, 'services_yoy': 2.3},
    # 2019
    '2019-01-16': {'cpi_yoy': 2.1, 'services_yoy': 2.2},
    '2019-02-13': {'cpi_yoy': 1.8, 'services_yoy': 2.1},
    '2019-03-20': {'cpi_yoy': 1.9, 'services_yoy': 2.1},
    '2019-04-17': {'cpi_yoy': 1.9, 'services_yoy': 2.0},
    '2019-05-22': {'cpi_yoy': 2.1, 'services_yoy': 2.1},
    '2019-06-19': {'cpi_yoy': 2.0, 'services_yoy': 2.0},
    '2019-07-17': {'cpi_yoy': 2.0, 'services_yoy': 2.0},
    '2019-08-14': {'cpi_yoy': 2.1, 'services_yoy': 2.0},
    '2019-09-18': {'cpi_yoy': 1.7, 'services_yoy': 1.9},
    '2019-10-16': {'cpi_yoy': 1.5, 'services_yoy': 1.8},
    '2019-11-20': {'cpi_yoy': 1.5, 'services_yoy': 1.8},
    '2019-12-18': {'cpi_yoy': 1.5, 'services_yoy': 1.7},
    # 2020
    '2020-01-15': {'cpi_yoy': 1.4, 'services_yoy': 1.6},
    '2020-02-19': {'cpi_yoy': 1.8, 'services_yoy': 1.7},
    '2020-03-25': {'cpi_yoy': 1.5, 'services_yoy': 1.6},
    '2020-04-22': {'cpi_yoy': 0.8, 'services_yoy': 1.2},
    '2020-05-20': {'cpi_yoy': 0.5, 'services_yoy': 0.9},
    '2020-06-17': {'cpi_yoy': 0.5, 'services_yoy': 0.8},
    '2020-07-15': {'cpi_yoy': 0.6, 'services_yoy': 0.9},
    '2020-08-19': {'cpi_yoy': 0.2, 'services_yoy': 0.6},
    '2020-09-16': {'cpi_yoy': 0.5, 'services_yoy': 0.7},
    '2020-10-21': {'cpi_yoy': 0.5, 'services_yoy': 0.7},
    '2020-11-18': {'cpi_yoy': 0.3, 'services_yoy': 0.5},
    '2020-12-16': {'cpi_yoy': 0.3, 'services_yoy': 0.5},
    # 2021
    '2021-01-20': {'cpi_yoy': 0.6, 'services_yoy': 0.7},
    '2021-02-17': {'cpi_yoy': 0.4, 'services_yoy': 0.6},
    '2021-03-24': {'cpi_yoy': 0.7, 'services_yoy': 0.8},
    '2021-04-21': {'cpi_yoy': 1.5, 'services_yoy': 1.1},
    '2021-05-19': {'cpi_yoy': 2.1, 'services_yoy': 1.5},
    '2021-06-16': {'cpi_yoy': 2.5, 'services_yoy': 1.8},
    '2021-07-14': {'cpi_yoy': 2.5, 'services_yoy': 1.8},
    '2021-08-18': {'cpi_yoy': 3.2, 'services_yoy': 2.1},
    '2021-09-15': {'cpi_yoy': 3.2, 'services_yoy': 2.2},
    '2021-10-20': {'cpi_yoy': 4.2, 'services_yoy': 2.9},
    '2021-11-17': {'cpi_yoy': 5.1, 'services_yoy': 3.6},
    '2021-12-15': {'cpi_yoy': 5.1, 'services_yoy': 3.6},
    # 2022
    '2022-01-19': {'cpi_yoy': 5.5, 'services_yoy': 3.9},
    '2022-02-16': {'cpi_yoy': 5.5, 'services_yoy': 3.9},
    '2022-03-23': {'cpi_yoy': 7.0, 'services_yoy': 4.6},
    '2022-04-13': {'cpi_yoy': 7.0, 'services_yoy': 4.6},
    '2022-05-18': {'cpi_yoy': 9.0, 'services_yoy': 5.1},
    '2022-06-22': {'cpi_yoy': 9.1, 'services_yoy': 5.2},
    '2022-07-20': {'cpi_yoy': 10.1, 'services_yoy': 5.7},
    '2022-08-17': {'cpi_yoy': 10.1, 'services_yoy': 5.7},
    '2022-09-14': {'cpi_yoy': 9.9, 'services_yoy': 5.8},
    '2022-10-19': {'cpi_yoy': 11.1, 'services_yoy': 6.2},
    '2022-11-16': {'cpi_yoy': 11.1, 'services_yoy': 6.3},
    '2022-12-14': {'cpi_yoy': 10.7, 'services_yoy': 6.3},
    # 2023
    '2023-01-18': {'cpi_yoy': 10.5, 'services_yoy': 6.3},
    '2023-02-15': {'cpi_yoy': 10.1, 'services_yoy': 6.2},
    '2023-03-22': {'cpi_yoy': 10.4, 'services_yoy': 6.6},
    '2023-04-19': {'cpi_yoy': 10.1, 'services_yoy': 6.6},
    '2023-05-24': {'cpi_yoy': 8.7, 'services_yoy': 6.4},
    '2023-06-21': {'cpi_yoy': 8.7, 'services_yoy': 6.5},
    '2023-07-19': {'cpi_yoy': 7.9, 'services_yoy': 6.5},
    '2023-08-16': {'cpi_yoy': 6.8, 'services_yoy': 6.2},
    '2023-09-20': {'cpi_yoy': 6.7, 'services_yoy': 6.1},
    '2023-10-18': {'cpi_yoy': 6.7, 'services_yoy': 6.1},
    '2023-11-15': {'cpi_yoy': 4.6, 'services_yoy': 5.7},
    '2023-12-20': {'cpi_yoy': 3.9, 'services_yoy': 5.3},
    # 2024
    '2024-01-17': {'cpi_yoy': 4.0, 'services_yoy': 5.4},
    '2024-02-14': {'cpi_yoy': 4.0, 'services_yoy': 5.4},
    '2024-03-20': {'cpi_yoy': 3.4, 'services_yoy': 5.0},
    '2024-04-17': {'cpi_yoy': 3.2, 'services_yoy': 4.9},
    '2024-05-22': {'cpi_yoy': 2.3, 'services_yoy': 4.5},
    '2024-06-19': {'cpi_yoy': 2.0, 'services_yoy': 4.3},
    '2024-07-17': {'cpi_yoy': 2.0, 'services_yoy': 4.2},
    '2024-08-14': {'cpi_yoy': 2.2, 'services_yoy': 4.3},
    '2024-09-18': {'cpi_yoy': 2.2, 'services_yoy': 4.3},
    '2024-10-16': {'cpi_yoy': 2.2, 'services_yoy': 4.3},
    '2024-11-20': {'cpi_yoy': 2.3, 'services_yoy': 4.4},
    '2024-12-18': {'cpi_yoy': 2.6, 'services_yoy': 4.5},
    # 2025
    '2025-01-15': {'cpi_yoy': 2.5, 'services_yoy': 4.4},
    '2025-02-19': {'cpi_yoy': 3.0, 'services_yoy': 4.6},
    '2025-03-26': {'cpi_yoy': 2.8, 'services_yoy': 4.5},
    '2025-04-16': {'cpi_yoy': 2.6, 'services_yoy': 4.4},
    '2025-05-21': {'cpi_yoy': 3.5, 'services_yoy': 4.8},
    '2025-06-18': {'cpi_yoy': 3.4, 'services_yoy': 4.7},
    '2025-07-16': {'cpi_yoy': 3.6, 'services_yoy': 4.8},
    '2025-08-20': {'cpi_yoy': 3.8, 'services_yoy': 4.9},
    '2025-09-17': {'cpi_yoy': 3.5, 'services_yoy': 4.7},
    '2025-10-22': {'cpi_yoy': 3.2, 'services_yoy': 4.5},
    '2025-11-19': {'cpi_yoy': 2.9, 'services_yoy': 4.3},
    '2025-12-17': {'cpi_yoy': 2.7, 'services_yoy': 4.2},
    # 2026
    '2026-01-21': {'cpi_yoy': 2.8, 'services_yoy': 4.3},
    '2026-02-18': {'cpi_yoy': 2.9, 'services_yoy': 4.4},
    '2026-03-25': {'cpi_yoy': 2.6, 'services_yoy': 4.2},
    '2026-04-22': {'cpi_yoy': 2.4, 'services_yoy': 4.0},
    '2026-05-13': {'cpi_yoy': 2.3, 'services_yoy': 3.9},
}


# ═══════════════════════════════════════════════════════════════
# SESSION-CONDITIONAL EDGE TABLE
# Backtested: 101 UK CPI releases, 2018-2026, ETH/USDT 15m
# ═══════════════════════════════════════════════════════════════

EDGE_TABLE = {
    'HOT_SERVICES': {
        'n': 23,
        'avg_24h': -0.137,
        'win_rate': 0.30,
        'session_returns': {
            'release_1h': -0.15, 'london_open': -0.10, 'london_morning': -0.20,
            'london_midday': +0.05, 'ny_pre_open': -0.15, 'ny_open': +0.10,
            'ny_am': +0.05, 'ny_lunch': -0.10, 'ny_pm': -0.05,
            'post_ny': +0.05, 'sydney_open': +0.10, 'tokyo_open': -0.05,
            'asia_mid': +0.10, 'boe_reopen': -0.05,
        },
        'bias': 'SHORT',  # Hot services → BoE hawkish → global tightening fear
        'confidence': 0.45,
        'desc': 'Hot Services CPI → BoE hawkish → halts dovish positioning',
    },
    'WARM_SERVICES': {
        'n': 27,
        'avg_24h': +0.434,
        'win_rate': 0.59,
        'session_returns': {
            'release_1h': +0.05, 'london_open': -0.05, 'london_morning': +0.10,
            'london_midday': +0.05, 'ny_am': +0.15, 'ny_pm': +0.05,
            'sydney_open': +0.10, 'asia_mid': +0.05,
        },
        'bias': 'LONG',  # Warm = not hot enough to scare, mild positive
        'confidence': 0.45,
        'desc': 'Warm Services — not hot enough to trigger hawkish fear',
    },
    'COOL_SERVICES': {
        'n': 5,
        'avg_24h': +1.230,
        'win_rate': 0.60,
        'session_returns': {},
        'bias': 'LONG',
        'confidence': 0.35,
        'desc': 'Cool Services — BoE dovish, GBP down, ETH up',
    },
    'COLD_SERVICES': {
        'n': 46,
        'avg_24h': -0.514,
        'win_rate': 0.43,
        'session_returns': {
            'release_1h': +0.05, 'london_open': -0.05, 'london_morning': -0.10,
            'ny_am': -0.10, 'ny_pm': -0.05,
        },
        'bias': 'SHORT',  # Cold services → deflation fear
        'confidence': 0.35,
        'desc': 'Cold Services — deflation risk, global demand weakness',
    },
}

# Cross-tabulation edges (signal × vol_regime × direction) — from backtest
CROSS_TAB_EDGES = {
    ('WARM_SERVICES', 'LOW_VOL', 'MISS'):   {'avg_ret': +1.165, 'win': 0.67, 'n': 6, 'bias': 'LONG'},
    ('WARM_SERVICES', 'LOW_VOL', 'INLINE'): {'avg_ret': +0.887, 'win': 0.67, 'n': 6, 'bias': 'LONG'},
    ('HOT_SERVICES', 'SQUEEZE', 'MISS'):    {'avg_ret': -1.943, 'win': 0.00, 'n': 5, 'bias': 'SHORT'},
    ('HOT_SERVICES', 'CHOP', 'MISS'):       {'avg_ret': -3.116, 'win': 0.00, 'n': 3, 'bias': 'SHORT'},
    ('COLD_SERVICES', 'CHOP', 'BEAT'):      {'avg_ret': +0.948, 'win': 0.53, 'n': 17, 'bias': 'LONG'},
    ('COLD_SERVICES', 'LOW_VOL', 'BEAT'):   {'avg_ret': -0.562, 'win': 0.41, 'n': 17, 'bias': 'SHORT'},
    ('HOT_SERVICES', 'LOW_VOL', 'MISS'):    {'avg_ret': -0.593, 'win': 0.22, 'n': 9, 'bias': 'SHORT'},
    ('COLD_SERVICES', 'TREND', 'BEAT'):     {'avg_ret': -6.329, 'win': 0.25, 'n': 4, 'bias': 'SHORT'},
}

# Session transmission chain (from backtest — mostly broken)
TRANSMISSION_CHAIN = {
    'london_ny_overlap_to_ny_am': 79.0,  # real edge
    'london_midday_to_ny_pre_open': 63.0,  # marginal
}


# ═══════════════════════════════════════════════════════════════
# HELPERS
# ═══════════════════════════════════════════════════════════════

def _classify_signal(cpi_yoy, services_yoy):
    if services_yoy >= 5.0:
        return 'HOT_SERVICES'
    elif services_yoy >= 4.0:
        return 'WARM_SERVICES'
    elif services_yoy >= 3.0:
        return 'COOL_SERVICES'
    return 'COLD_SERVICES'


def _classify_direction(cpi_yoy, services_yoy, prev_services=None):
    if prev_services is not None:
        delta = services_yoy - prev_services
        if delta > 0.2:
            return 'MISS'
        elif delta < -0.2:
            return 'BEAT'
    if services_yoy >= 4.5:
        return 'MISS'
    elif services_yoy < 3.5:
        return 'BEAT'
    return 'INLINE'


def _is_uk_release_day(today_str=None, window_days=1):
    if today_str is None:
        today_str = datetime.utcnow().strftime('%Y-%m-%d')
    today = datetime.strptime(today_str, '%Y-%m-%d')
    for release_date_str in sorted(UK_RELEASES.keys(), reverse=True):
        release_dt = datetime.strptime(release_date_str, '%Y-%m-%d')
        days_since = (today - release_dt).days
        if 0 <= days_since <= window_days:
            return True, release_date_str, UK_RELEASES[release_date_str]
    return False, None, None


def score_m31_uk_cpi(wyckoff_phase='RANGE', vol_regime='CHOP',
                     direction='LONG', today_str=None, config=None,
                     now_utc=None):
    """Score UK CPI session bias.

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

    if not cfg.get('M31_ENABLED', True):
        return 'SKIP', 0.0, 1.0, {'regime': 'DISABLED'}

    is_release, release_date, release_data = _is_uk_release_day(
        today_str, window_days=cfg.get('M31_WINDOW_DAYS', 1))

    if not is_release:
        return 'SKIP', 0.0, 1.0, {'regime': 'NOT_RELEASE_DAY'}

    cpi_yoy = release_data['cpi_yoy']
    services_yoy = release_data['services_yoy']

    # Previous month for direction
    sorted_dates = sorted(UK_RELEASES.keys())
    idx = sorted_dates.index(release_date) if release_date in sorted_dates else -1
    prev_services = UK_RELEASES[sorted_dates[idx-1]]['services_yoy'] if idx > 0 else None

    signal = _classify_signal(cpi_yoy, services_yoy)
    surprise = _classify_direction(cpi_yoy, services_yoy, prev_services)

    # ── Lookup 1: Cross-tab edge ──
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
            'cpi_yoy': cpi_yoy, 'services_yoy': services_yoy, 'release_date': release_date,
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
        'regime': f'UK_CPI_{bias}',
        'release_date': release_date,
        'cpi_yoy': cpi_yoy,
        'services_yoy': services_yoy,
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


def format_m31(details):
    if not details or details.get('regime') in ('DISABLED', 'NOT_RELEASE_DAY', 'NO_EDGE'):
        regime = details.get('regime', '?') if details else '?'
        if regime == 'NOT_RELEASE_DAY':
            return ''
        return ''

    bias = details.get('bias', '?')
    cpi = details.get('cpi_yoy', 0)
    services = details.get('services_yoy', 0)
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
    lines.append(f"\n  {icon} M31 UK CPI: {bias}")
    lines.append(f"    Release: {release}  |  CPI: {cpi:+.1f}%  |  Services: {services:+.1f}%  |  Signal: {signal}")
    lines.append(f"    Surprise: {surprise}  |  Source: {source}")
    lines.append(f"    Backtest: avg={avg_ret:+.2f}%  win={win*100:.0f}%  n={n}")
    lines.append(f"    {conf_icon} Confidence: {conf:.2f}  |  Score adj: {score_adj:+.3f}  |  Size: {size_mult:.2f}x")
    if desc:
        lines.append(f"    📖 {desc}")

    chain = details.get('chain', {})
    if chain:
        lines.append(f"    📊 Chain: Ldn-NY→NY AM {chain.get('london_ny_overlap_to_ny_am', 0):.0f}% (real edge)")

    return '\n'.join(lines)


def get_uk_cpi_cache_path():
    cache_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), 'data', 'macro')
    return os.path.join(cache_dir, 'uk_cpi_cache.json')


def load_uk_cpi_cache():
    cache_path = get_uk_cpi_cache_path()
    if os.path.exists(cache_path):
        try:
            with open(cache_path) as f:
                return json.load(f)
        except (json.JSONDecodeError, IOError):
            pass
    return {}


def update_uk_cpi_cache(cpi_yoy, services_yoy, release_date=None):
    cache = load_uk_cpi_cache()
    if release_date is None:
        release_date = datetime.utcnow().strftime('%Y-%m-%d')
    cache[release_date] = {
        'cpi_yoy': cpi_yoy,
        'services_yoy': services_yoy,
        'updated': datetime.utcnow().isoformat(),
    }
    cache_path = get_uk_cpi_cache_path()
    os.makedirs(os.path.dirname(cache_path), exist_ok=True)
    with open(cache_path, 'w') as f:
        json.dump(cache, f, indent=2)
    return cache

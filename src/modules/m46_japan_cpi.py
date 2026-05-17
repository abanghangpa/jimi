"""
M46: Japan CPI (Tokyo Flash) Session Bias (Regime-Conditional)

On Statistics Bureau Japan CPI release days (~final Friday of month,
23:30 UTC prev day = 07:30 MYT = 08:30 JST), applies a session-conditional
directional bias based on the combination of:
  - Wyckoff phase (M21)
  - Volatility regime (M9)
  - CPI signal: HOT_BEAT / WARM_BEAT / INLINE / MILD_MISS / COOL_MISS
  - Core pressure (BoJ): EXTREME / HAWKISH / MODERATE / DOVISH

Thesis (from user #30):
  Asia Morning (07:30 MYT) → USD/JPY Level Shift → BoJ Alignment → Europe Open
  Hot Tokyo CPI → hawkish BoJ expectations → JPY strengthens → USD/JPY drops
  → carry-trade de-risking → ETH/USDT selling pressure at Asia open
  London desks adjust risk metrics downward
  Loopback: inflation trend justifies BoJ policy shifts

Backtested on 91 Japan CPI releases (2018-2026) against ETH/USDT 15m data.

Key findings:
  24h aggregate: +0.204% avg, 48.4% win, n=91 — NOT significant (p=0.66)
  Session drift: negative from Asia Afternoon (-0.31%) through London (-0.44%)
  Hot CPI ≥3%: -1.216% avg vs Cool CPI <2%: +0.800% avg (p=0.11, marginal)
  Hawkish core ≥2%: -0.465% avg vs Dovish core <2%: +0.622% avg

  Specific combos with edge:
    RANGE + COMPRESSING + COOL_MISS: +2.836% avg, 57% win, n=7  → LONG
    RANGE + NEUTRAL + WARM_BEAT:     +1.859% avg, 50% win, n=8  → LONG
    RANGE + NEUTRAL + COOL_MISS:     -3.031% avg, 40% win, n=5  → SHORT
    DOVISH + WARM_BEAT:              +3.534% avg, 75% win, n=4  → LONG
    EXTREME + WARM_BEAT:             -0.596% avg, 33% win, n=6  → SHORT
    HAWKISH + INLINE:                -0.843% avg, 46% win, n=11 → SHORT

  Transmission chain (STRONG):
    Tokyo Open→Asia Afternoon: 67-78% persistence ✅
    Asia Afternoon→London: 85-100% persistence ✅
    London→NY: 89-100% persistence ✅
    Chain holds from Tokyo Open to NY PM: 65-72% overall persist

Integration: lightweight modifier on Japan CPI release days only (~12x/year).
Returns a score adjustment and size multiplier — does NOT veto.

Usage:
    from src.modules.m46_japan_cpi import score_m46_japan_cpi, format_m46
    status, score_adj, size_mult, details = score_m46_japan_cpi(
        wyckoff_phase='RANGE', vol_regime='CHOP', direction='LONG')
"""

from datetime import datetime, timedelta
import json
import os


JAPAN_CPI_RELEASES = {
    '2018-01-26': {'cpi_yoy': 0.9, 'core_yoy': 0.7, 'consensus': 0.8, 'prior': 0.6},
    '2018-02-23': {'cpi_yoy': 1.4, 'core_yoy': 0.9, 'consensus': 1.3, 'prior': 1.0},
    '2018-03-23': {'cpi_yoy': 1.3, 'core_yoy': 0.9, 'consensus': 1.4, 'prior': 1.4},
    '2018-04-27': {'cpi_yoy': 0.6, 'core_yoy': 0.5, 'consensus': 0.7, 'prior': 1.3},
    '2018-05-25': {'cpi_yoy': 0.5, 'core_yoy': 0.5, 'consensus': 0.5, 'prior': 0.6},
    '2018-06-29': {'cpi_yoy': 0.6, 'core_yoy': 0.6, 'consensus': 0.5, 'prior': 0.5},
    '2018-07-27': {'cpi_yoy': 0.7, 'core_yoy': 0.8, 'consensus': 0.7, 'prior': 0.6},
    '2018-08-31': {'cpi_yoy': 0.9, 'core_yoy': 0.9, 'consensus': 0.9, 'prior': 0.7},
    '2018-09-28': {'cpi_yoy': 1.1, 'core_yoy': 0.9, 'consensus': 0.9, 'prior': 0.9},
    '2018-10-26': {'cpi_yoy': 1.1, 'core_yoy': 1.0, 'consensus': 1.0, 'prior': 1.1},
    '2018-11-30': {'cpi_yoy': 0.8, 'core_yoy': 0.9, 'consensus': 0.9, 'prior': 1.1},
    '2018-12-28': {'cpi_yoy': 0.3, 'core_yoy': 0.8, 'consensus': 0.8, 'prior': 0.8},
    '2019-01-25': {'cpi_yoy': 0.3, 'core_yoy': 0.7, 'consensus': 0.8, 'prior': 0.3},
    '2019-02-22': {'cpi_yoy': 0.2, 'core_yoy': 0.7, 'consensus': 0.7, 'prior': 0.3},
    '2019-03-29': {'cpi_yoy': 0.5, 'core_yoy': 0.8, 'consensus': 0.7, 'prior': 0.2},
    '2019-04-26': {'cpi_yoy': 0.9, 'core_yoy': 1.0, 'consensus': 0.8, 'prior': 0.5},
    '2019-05-31': {'cpi_yoy': 1.1, 'core_yoy': 1.1, 'consensus': 0.9, 'prior': 0.9},
    '2019-06-28': {'cpi_yoy': 0.7, 'core_yoy': 0.9, 'consensus': 0.8, 'prior': 1.1},
    '2019-07-26': {'cpi_yoy': 0.8, 'core_yoy': 0.9, 'consensus': 0.8, 'prior': 0.7},
    '2019-08-30': {'cpi_yoy': 0.5, 'core_yoy': 0.6, 'consensus': 0.6, 'prior': 0.8},
    '2019-09-27': {'cpi_yoy': 0.3, 'core_yoy': 0.5, 'consensus': 0.5, 'prior': 0.5},
    '2019-10-25': {'cpi_yoy': 0.2, 'core_yoy': 0.4, 'consensus': 0.3, 'prior': 0.3},
    '2019-11-29': {'cpi_yoy': 0.5, 'core_yoy': 0.8, 'consensus': 0.5, 'prior': 0.2},
    '2019-12-27': {'cpi_yoy': 0.8, 'core_yoy': 0.8, 'consensus': 0.5, 'prior': 0.5},
    '2020-01-31': {'cpi_yoy': 0.7, 'core_yoy': 0.8, 'consensus': 0.7, 'prior': 0.8},
    '2020-02-28': {'cpi_yoy': 0.5, 'core_yoy': 0.6, 'consensus': 0.6, 'prior': 0.7},
    '2020-03-27': {'cpi_yoy': 0.3, 'core_yoy': 0.4, 'consensus': 0.4, 'prior': 0.5},
    '2020-04-24': {'cpi_yoy': 0.1, 'core_yoy': 0.2, 'consensus': 0.4, 'prior': 0.3},
    '2020-05-29': {'cpi_yoy': -0.2, 'core_yoy': 0.0, 'consensus': 0.1, 'prior': 0.1},
    '2020-06-26': {'cpi_yoy': 0.0, 'core_yoy': 0.1, 'consensus': 0.1, 'prior': -0.2},
    '2020-07-31': {'cpi_yoy': 0.3, 'core_yoy': 0.4, 'consensus': 0.1, 'prior': 0.0},
    '2020-08-28': {'cpi_yoy': 0.1, 'core_yoy': 0.0, 'consensus': 0.0, 'prior': 0.3},
    '2020-09-25': {'cpi_yoy': 0.0, 'core_yoy': -0.1, 'consensus': -0.2, 'prior': 0.1},
    '2020-10-30': {'cpi_yoy': -0.4, 'core_yoy': -0.5, 'consensus': -0.3, 'prior': 0.0},
    '2020-11-27': {'cpi_yoy': -0.9, 'core_yoy': -0.7, 'consensus': -0.7, 'prior': -0.4},
    '2020-12-25': {'cpi_yoy': -1.2, 'core_yoy': -0.9, 'consensus': -0.9, 'prior': -0.9},
    '2021-01-29': {'cpi_yoy': -1.3, 'core_yoy': -1.1, 'consensus': -1.0, 'prior': -1.2},
    '2021-02-26': {'cpi_yoy': -1.1, 'core_yoy': -0.9, 'consensus': -0.7, 'prior': -1.3},
    '2021-03-26': {'cpi_yoy': -0.7, 'core_yoy': -0.5, 'consensus': -0.4, 'prior': -1.1},
    '2021-04-23': {'cpi_yoy': -0.9, 'core_yoy': -0.7, 'consensus': -0.2, 'prior': -0.7},
    '2021-05-28': {'cpi_yoy': -1.1, 'core_yoy': -0.8, 'consensus': -0.7, 'prior': -0.9},
    '2021-06-25': {'cpi_yoy': -0.5, 'core_yoy': -0.2, 'consensus': -0.1, 'prior': -1.1},
    '2021-07-30': {'cpi_yoy': -0.3, 'core_yoy': 0.0, 'consensus': -0.2, 'prior': -0.5},
    '2021-08-27': {'cpi_yoy': -0.4, 'core_yoy': -0.2, 'consensus': -0.2, 'prior': -0.3},
    '2021-09-24': {'cpi_yoy': 0.0, 'core_yoy': 0.1, 'consensus': 0.0, 'prior': -0.4},
    '2021-10-29': {'cpi_yoy': 0.1, 'core_yoy': 0.1, 'consensus': 0.1, 'prior': 0.0},
    '2021-11-26': {'cpi_yoy': 0.6, 'core_yoy': 0.6, 'consensus': 0.4, 'prior': 0.1},
    '2021-12-24': {'cpi_yoy': 0.8, 'core_yoy': 0.5, 'consensus': 0.5, 'prior': 0.6},
    '2022-01-28': {'cpi_yoy': 0.5, 'core_yoy': 0.2, 'consensus': 0.5, 'prior': 0.8},
    '2022-02-25': {'cpi_yoy': 0.9, 'core_yoy': 0.5, 'consensus': 0.3, 'prior': 0.5},
    '2022-03-25': {'cpi_yoy': 1.3, 'core_yoy': 0.8, 'consensus': 0.8, 'prior': 0.9},
    '2022-04-22': {'cpi_yoy': 2.5, 'core_yoy': 1.9, 'consensus': 1.4, 'prior': 1.3},
    '2022-05-27': {'cpi_yoy': 2.4, 'core_yoy': 1.9, 'consensus': 2.1, 'prior': 2.5},
    '2022-06-24': {'cpi_yoy': 2.4, 'core_yoy': 2.1, 'consensus': 2.1, 'prior': 2.4},
    '2022-07-22': {'cpi_yoy': 2.6, 'core_yoy': 2.4, 'consensus': 2.2, 'prior': 2.4},
    '2022-08-26': {'cpi_yoy': 3.0, 'core_yoy': 2.7, 'consensus': 2.6, 'prior': 2.6},
    '2022-09-30': {'cpi_yoy': 3.0, 'core_yoy': 2.8, 'consensus': 2.9, 'prior': 3.0},
    '2022-10-21': {'cpi_yoy': 3.7, 'core_yoy': 3.4, 'consensus': 3.1, 'prior': 3.0},
    '2022-11-25': {'cpi_yoy': 3.8, 'core_yoy': 3.6, 'consensus': 3.5, 'prior': 3.7},
    '2022-12-23': {'cpi_yoy': 4.0, 'core_yoy': 3.7, 'consensus': 3.8, 'prior': 3.8},
    '2023-01-27': {'cpi_yoy': 4.4, 'core_yoy': 4.2, 'consensus': 4.0, 'prior': 4.0},
    '2023-02-24': {'cpi_yoy': 4.3, 'core_yoy': 4.2, 'consensus': 4.2, 'prior': 4.4},
    '2023-03-24': {'cpi_yoy': 3.3, 'core_yoy': 3.2, 'consensus': 3.1, 'prior': 4.3},
    '2023-04-28': {'cpi_yoy': 3.5, 'core_yoy': 3.4, 'consensus': 3.2, 'prior': 3.3},
    '2023-05-26': {'cpi_yoy': 3.2, 'core_yoy': 3.1, 'consensus': 3.1, 'prior': 3.5},
    '2023-06-30': {'cpi_yoy': 3.1, 'core_yoy': 3.0, 'consensus': 3.1, 'prior': 3.2},
    '2023-07-28': {'cpi_yoy': 3.0, 'core_yoy': 2.9, 'consensus': 2.9, 'prior': 3.1},
    '2023-08-25': {'cpi_yoy': 2.8, 'core_yoy': 2.7, 'consensus': 2.7, 'prior': 3.0},
    '2023-09-29': {'cpi_yoy': 2.8, 'core_yoy': 2.7, 'consensus': 2.7, 'prior': 2.8},
    '2023-10-27': {'cpi_yoy': 2.9, 'core_yoy': 2.7, 'consensus': 2.7, 'prior': 2.8},
    '2023-11-24': {'cpi_yoy': 2.3, 'core_yoy': 2.3, 'consensus': 2.4, 'prior': 2.9},
    '2023-12-22': {'cpi_yoy': 2.2, 'core_yoy': 2.2, 'consensus': 2.5, 'prior': 2.3},
    '2024-01-26': {'cpi_yoy': 1.6, 'core_yoy': 1.8, 'consensus': 1.9, 'prior': 2.2},
    '2024-02-27': {'cpi_yoy': 2.2, 'core_yoy': 2.5, 'consensus': 1.9, 'prior': 1.6},
    '2024-03-22': {'cpi_yoy': 2.8, 'core_yoy': 3.2, 'consensus': 2.9, 'prior': 2.2},
    '2024-04-26': {'cpi_yoy': 1.8, 'core_yoy': 2.2, 'consensus': 2.2, 'prior': 2.8},
    '2024-05-24': {'cpi_yoy': 2.2, 'core_yoy': 2.4, 'consensus': 2.2, 'prior': 1.8},
    '2024-06-28': {'cpi_yoy': 2.8, 'core_yoy': 2.7, 'consensus': 2.6, 'prior': 2.2},
    '2024-07-26': {'cpi_yoy': 2.7, 'core_yoy': 2.6, 'consensus': 2.7, 'prior': 2.8},
    '2024-08-30': {'cpi_yoy': 2.7, 'core_yoy': 2.4, 'consensus': 2.3, 'prior': 2.7},
    '2024-09-27': {'cpi_yoy': 2.5, 'core_yoy': 2.0, 'consensus': 2.3, 'prior': 2.7},
    '2024-10-25': {'cpi_yoy': 1.8, 'core_yoy': 1.8, 'consensus': 2.0, 'prior': 2.5},
    '2024-11-29': {'cpi_yoy': 2.3, 'core_yoy': 2.2, 'consensus': 2.2, 'prior': 1.8},
    '2024-12-27': {'cpi_yoy': 3.0, 'core_yoy': 2.4, 'consensus': 2.5, 'prior': 2.3},
    '2025-01-31': {'cpi_yoy': 3.6, 'core_yoy': 2.5, 'consensus': 3.0, 'prior': 3.0},
    '2025-02-28': {'cpi_yoy': 3.5, 'core_yoy': 2.5, 'consensus': 3.2, 'prior': 3.6},
    '2025-03-28': {'cpi_yoy': 3.3, 'core_yoy': 2.2, 'consensus': 3.2, 'prior': 3.5},
    '2025-04-25': {'cpi_yoy': 3.5, 'core_yoy': 2.2, 'consensus': 3.2, 'prior': 3.3},
    '2025-05-30': {'cpi_yoy': 3.2, 'core_yoy': 2.1, 'consensus': 3.1, 'prior': 3.5},
    '2026-01-30': {'cpi_yoy': 2.8, 'core_yoy': 2.0, 'consensus': 2.7, 'prior': 2.9},
    '2026-02-27': {'cpi_yoy': 2.6, 'core_yoy': 1.9, 'consensus': 2.5, 'prior': 2.8},
}


# Edge table (n≥3, |avg|≥0.5%)
EDGE_TABLE = {
    ('RANGE', 'NEUTRAL', 'COOL_MISS'):         (-3.031, 0.400, 5, 'SHORT'),
    ('RANGE', 'COMPRESSING', 'COOL_MISS'):      (2.836, 0.571, 7, 'LONG'),
    ('RANGE', 'NEUTRAL', 'WARM_BEAT'):          (1.859, 0.500, 8, 'LONG'),
    ('RANGE', 'NEUTRAL', 'MILD_MISS'):          (-1.228, 0.333, 3, 'SHORT'),
    ('RANGE', 'COMPRESSING', 'WARM_BEAT'):      (-1.044, 0.500, 8, 'SHORT'),
    ('RANGE', 'NEUTRAL', 'INLINE'):             (0.932, 0.571, 7, 'LONG'),
    ('MARKUP', 'NEUTRAL', 'COOL_MISS'):         (0.670, 0.750, 4, 'LONG'),
    ('RANGE', 'TRENDING', 'WARM_BEAT'):         (-0.636, 0.500, 4, 'SHORT'),
    ('MARKDOWN', 'NEUTRAL', 'WARM_BEAT'):       (-0.523, 0.333, 3, 'SHORT'),
    ('RANGE', 'COMPRESSING', 'INLINE'):         (-0.513, 0.556, 9, 'SHORT'),
}

# Core Pressure × Signal edge
CORE_SIGNAL_EDGE = {
    ('DOVISH', 'WARM_BEAT'):                     (3.534, 0.750, 4, 'LONG'),
    ('DOVISH', 'COOL_MISS'):                     (1.405, 0.571, 21, 'LONG'),
    ('HAWKISH', 'MILD_MISS'):                    (1.468, 0.500, 2, 'LONG'),
    ('HAWKISH', 'INLINE'):                       (-0.843, 0.455, 11, 'SHORT'),
    ('EXTREME', 'WARM_BEAT'):                    (-0.596, 0.333, 6, 'SHORT'),
    ('MODERATE', 'WARM_BEAT'):                   (-0.969, 0.667, 3, 'SHORT'),
}


def _classify_signal(actual, consensus, prior):
    diff = actual - consensus
    surprise = diff / abs(consensus) if consensus != 0 else diff
    if surprise > 0.2:
        return 'HOT_BEAT'
    elif surprise > 0.05:
        return 'WARM_BEAT'
    elif surprise < -0.2:
        return 'COOL_MISS'
    elif surprise < -0.05:
        return 'MILD_MISS'
    else:
        return 'INLINE'


def _classify_inflation(actual):
    if actual >= 4.0:
        return 'RUNAWAY'
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


def _classify_core_pressure(core_yoy):
    if core_yoy >= 3.0:
        return 'EXTREME'
    elif core_yoy >= 2.0:
        return 'HAWKISH'
    elif core_yoy >= 1.0:
        return 'MODERATE'
    else:
        return 'DOVISH'


def _is_release_day(today_str):
    today = datetime.strptime(today_str, '%Y-%m-%d')
    for date_str in JAPAN_CPI_RELEASES:
        release_dt = datetime.strptime(date_str, '%Y-%m-%d')
        delta = abs((today - release_dt).days)
        if delta <= 1:
            return date_str, JAPAN_CPI_RELEASES[date_str]
    return None, None


def score_m46_japan_cpi(wyckoff_phase='RANGE', vol_regime='CHOP',
                         direction='LONG', today_str=None, config=None):
    """Score M46: Japan CPI (Tokyo Flash) session bias."""
    cfg = config or {}
    if not cfg.get('M46_ENABLED', True):
        return 'DISABLED', 0.0, 1.0, {'regime': 'DISABLED'}

    if today_str is None:
        today_str = datetime.utcnow().strftime('%Y-%m-%d')

    release_date, release_data = _is_release_day(today_str)
    if release_data is None:
        return 'SKIP', 0.0, 1.0, {'regime': 'NOT_RELEASE_DAY'}

    cpi = release_data['cpi_yoy']
    core = release_data['core_yoy']
    consensus = release_data['consensus']
    prior = release_data['prior']

    signal = _classify_signal(cpi, consensus, prior)
    infl = _classify_inflation(cpi)
    core_pressure = _classify_core_pressure(core)

    # Primary: Wyckoff × Vol × Signal
    edge_key = (wyckoff_phase, vol_regime, signal)
    edge = EDGE_TABLE.get(edge_key)
    if edge is None:
        for (w, v, s), e in EDGE_TABLE.items():
            if w == wyckoff_phase and s == signal:
                edge = e
                edge_key = (w, v, s)
                break

    # Secondary: Core Pressure × Signal
    cs_key = (core_pressure, signal)
    cs_edge = CORE_SIGNAL_EDGE.get(cs_key)

    if edge and abs(edge[0]) >= 0.5 and edge[2] >= 3:
        avg_ret, win_rate, n, bias = edge
        source = f'wyckoff_vol_signal: {edge_key}'
        confidence = min(0.6, n / 10.0)
    elif cs_edge and abs(cs_edge[0]) >= 0.5 and cs_edge[2] >= 2:
        avg_ret, win_rate, n, bias = cs_edge
        source = f'core_pressure: {cs_key}'
        confidence = min(0.5, n / 10.0)
    else:
        return 'NO_EDGE', 0.0, 1.0, {
            'regime': 'NO_EDGE', 'release_date': release_date,
            'cpi_yoy': cpi, 'core_yoy': core, 'consensus': consensus,
            'signal': signal, 'inflation': infl, 'core_pressure': core_pressure,
        }

    if abs(avg_ret) >= 2.0 and n >= 3:
        score_adj = 0.07 if avg_ret > 0 else -0.07
    elif abs(avg_ret) >= 1.0:
        score_adj = 0.05 if avg_ret > 0 else -0.05
    elif abs(avg_ret) >= 0.5:
        score_adj = 0.03 if avg_ret > 0 else -0.03
    else:
        score_adj = 0.02 if avg_ret > 0 else -0.02

    if bias != direction:
        score_adj *= -0.5

    if n >= 5 and win_rate >= 0.6:
        size_mult = 0.85
    elif n >= 3 and win_rate >= 0.5:
        size_mult = 0.75
    else:
        size_mult = 0.65

    if abs(score_adj) >= 0.05:
        status = 'PASS'
    elif abs(score_adj) >= 0.03:
        status = 'WEAK'
    else:
        status = 'NO_EDGE'

    details = {
        'regime': f'{wyckoff_phase}_{vol_regime}_{signal}',
        'release_date': release_date,
        'cpi_yoy': cpi, 'core_yoy': core,
        'consensus': consensus, 'prior': prior,
        'signal': signal, 'inflation': infl,
        'core_pressure': core_pressure,
        'wyckoff': wyckoff_phase, 'vol': vol_regime,
        'bias': bias, 'avg_ret_24h': avg_ret, 'win_rate': win_rate,
        'sample_size': n, 'confidence': confidence, 'source': source,
        'score_adj': score_adj, 'size_mult': size_mult,
        'boj_metric': True,  # flag: leads BoJ rate decisions
    }
    return status, score_adj, size_mult, details


def format_m46(details):
    if not details or details.get('regime') in ('DISABLED', 'NOT_RELEASE_DAY', 'NO_EDGE'):
        return None

    bias = details.get('bias', '?')
    cpi = details.get('cpi_yoy', 0)
    core = details.get('core_yoy', 0)
    cons = details.get('consensus', 0)
    signal = details.get('signal', '?')
    infl = details.get('inflation', '?')
    core_p = details.get('core_pressure', '?')
    avg_ret = details.get('avg_ret_24h', 0)
    win_rate = details.get('win_rate', 0)
    n = details.get('sample_size', 0)
    score_adj = details.get('score_adj', 0)
    size_mult = details.get('size_mult', 1.0)

    bias_icon = '🟢' if bias == 'LONG' else '🔴' if bias == 'SHORT' else '⚪'
    core_icon = {'EXTREME': '🔴🔴', 'HAWKISH': '🔴', 'MODERATE': '🟡',
                 'DOVISH': '🟢'}.get(core_p, '⚪')

    lines = [
        f"  M46 Japan CPI: {bias_icon} {bias:>8}  "
        f"CPI={cpi:.1f}% cons={cons:.1f}% core={core:.1f}%  "
        f"{core_icon}BoJ:{core_p}  signal={signal}",
        f"    Backtest: 24h={avg_ret:+.2f}% win={win_rate*100:.0f}% n={n}  "
        f"adj={score_adj:+.3f} size={size_mult:.2f}x  "
        f"chain: Tokyo→London 67-100%, London→NY 89-100%  🏛️BoJ metric",
    ]
    return '\n'.join(lines)

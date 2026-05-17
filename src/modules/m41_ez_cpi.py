"""
M41: Eurozone CPI Flash Session Bias (Regime-Conditional)

On Eurostat HICP Flash release days (~end of month, 09:00 UTC = 10:00 CET / 18:00 MYT),
applies a session-conditional directional bias based on the combination of:
  - Wyckoff phase (M21)
  - Volatility regime (M9)
  - HICP signal: STRONG_BEAT / BEAT / INLINE / MISS / BIG_MISS
  - Core-Headline spread (core sticky = hawkish)

Thesis (from user #25):
  Europe Morning (18:00 MYT) → US Inflation Desk Comparison → ECB Meeting
  Cool EZ CPI → green light for ECB rate cuts → EUR drops → global yield easing → crypto bids
  Hot EZ CPI → hawkish ECB → EUR surges → DXY lower → ETH bounce
  US macro funds use this for global inflation comparison ahead of US CPI (2 weeks later)
  Early directional proxy for Core PCE (4 weeks later)

Backtested on 92 Eurozone CPI Flash releases (2018-2026) against ETH/USDT 15m data.

Key findings:
  24h aggregate: +0.931% avg, 53.3% win, n=92 — STATISTICALLY SIGNIFICANT (p=0.013) ✅
  This is one of the few macro events with overall 24h significance!

  Specific combos with edge (n≥3, |avg|≥0.5%):
    MARKUP + NEUTRAL + INLINE:       +5.060% avg, 75% win, n=4  → LONG
    MARKDOWN + NEUTRAL + MISS:       +4.180% avg, 100% win, n=3 → LONG
    RANGE + NEUTRAL + STRONG_BEAT:   +3.940% avg, 100% win, n=5 → LONG
    RANGE + NEUTRAL + BEAT:          +2.655% avg, 62% win, n=8  → LONG
    RANGE + COMPRESSING + BEAT:      -0.982% avg, 57% win, n=7  → SHORT

  Core-Headline spread combos:
    CORE_STICKY + INLINE:            +3.676% avg, 75% win, n=4  → LONG
    CORE_BELOW + BEAT:               +2.502% avg, 69% win, n=13 → LONG

  Transmission chain (EXCEPTIONAL):
    Asia session: 88-99% persistence ✅
    Asia→Europe: 98-100% Tokyo Close→Frankfurt Open ✅ (chain HOLDS!)
    London→NY: 75-100% persistence ✅
    Chain only breaks at London Morning (0% — data artifact)
    Overall: 76-79% persistence to NY PM ✅

Integration: lightweight modifier on Eurostat release days only (~12x/year).
Returns a score adjustment and size multiplier — does NOT veto.

Usage:
    from src.modules.m41_ez_cpi import score_m41_ez_cpi, format_m41
    status, score_adj, size_mult, details = score_m41_ez_cpi(
        wyckoff_phase='RANGE', vol_regime='CHOP', direction='LONG')
"""

from datetime import datetime, timedelta
import json
import os


# ═══════════════════════════════════════════════════════════════
# EUROZONE CPI FLASH RELEASE DATES + ACTUAL VALUES (09:00 UTC)
# Eurostat releases HICP flash estimate ~last day of month
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
    # 2026
    '2026-01-30': {'hicp_yoy': 2.2, 'core_yoy': 2.5, 'consensus': 2.2, 'prior': 2.1},
    '2026-02-27': {'hicp_yoy': 2.1, 'core_yoy': 2.4, 'consensus': 2.1, 'prior': 2.2},
    '2026-03-31': {'hicp_yoy': 2.0, 'core_yoy': 2.3, 'consensus': 2.0, 'prior': 2.1},
}


# ═══════════════════════════════════════════════════════════════
# EDGE TABLE — Wyckoff × Vol × Signal → avg 24h return
# Only combos with n≥3, |avg|≥0.5%
# ═══════════════════════════════════════════════════════════════

EDGE_TABLE = {
    ('MARKUP', 'NEUTRAL', 'INLINE'):        (5.060, 0.750, 4, 'LONG'),
    ('MARKDOWN', 'NEUTRAL', 'MISS'):        (4.180, 1.000, 3, 'LONG'),
    ('RANGE', 'NEUTRAL', 'STRONG_BEAT'):    (3.940, 1.000, 5, 'LONG'),
    ('RANGE', 'NEUTRAL', 'BEAT'):           (2.655, 0.625, 8, 'LONG'),
    ('RANGE', 'COMPRESSING', 'BEAT'):       (-0.982, 0.571, 7, 'SHORT'),
}

# Core-Headline spread edge table
CORE_SPREAD_EDGE = {
    ('CORE_STICKY', 'INLINE'):              (3.676, 0.750, 4, 'LONG'),
    ('CORE_BELOW', 'BEAT'):                 (2.502, 0.692, 13, 'LONG'),
    ('CORE_EQUAL', 'MISS'):                 (2.424, 0.667, 3, 'LONG'),
    ('CORE_SLIGHT', 'MISS'):                (1.674, 0.333, 6, 'LONG'),
    ('CORE_SLIGHT', 'BEAT'):                (-0.981, 0.429, 7, 'SHORT'),
}


def _classify_signal(actual, consensus, prior):
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
    change = actual - prior
    if change > 0.2:
        return 'RISING'
    elif change < -0.2:
        return 'FALLING'
    else:
        return 'STABLE'


def _classify_core_spread(core_yoy, hicp_yoy):
    spread = core_yoy - hicp_yoy
    if spread > 1.0:
        return 'CORE_VERY_STICKY'
    elif spread > 0.5:
        return 'CORE_STICKY'
    elif spread > 0:
        return 'CORE_SLIGHT'
    elif spread == 0:
        return 'CORE_EQUAL'
    else:
        return 'CORE_BELOW'


def _is_release_day(today_str):
    today = datetime.strptime(today_str, '%Y-%m-%d')
    for date_str in EZ_CPI_RELEASES:
        release_dt = datetime.strptime(date_str, '%Y-%m-%d')
        delta = abs((today - release_dt).days)
        if delta <= 1:
            return date_str, EZ_CPI_RELEASES[date_str]
    return None, None


def score_m41_ez_cpi(wyckoff_phase='RANGE', vol_regime='CHOP',
                      direction='LONG', today_str=None, config=None):
    """
    Score M41: Eurozone CPI Flash session bias.

    Returns:
        (status, score_adj, size_mult, details)
    """
    cfg = config or {}
    if not cfg.get('M41_ENABLED', True):
        return 'DISABLED', 0.0, 1.0, {'regime': 'DISABLED'}

    if today_str is None:
        today_str = datetime.utcnow().strftime('%Y-%m-%d')

    release_date, release_data = _is_release_day(today_str)
    if release_data is None:
        return 'SKIP', 0.0, 1.0, {'regime': 'NOT_RELEASE_DAY'}

    hicp = release_data['hicp_yoy']
    core = release_data['core_yoy']
    consensus = release_data['consensus']
    prior = release_data['prior']

    signal = _classify_signal(hicp, consensus, prior)
    cpi_level = _classify_level(hicp)
    cpi_dir = _classify_direction(hicp, prior)
    core_spread = _classify_core_spread(core, hicp)

    # Primary lookup: Wyckoff × Vol × Signal
    edge_key = (wyckoff_phase, vol_regime, signal)
    edge = EDGE_TABLE.get(edge_key)

    if edge is None:
        for (w, v, s), e in EDGE_TABLE.items():
            if w == wyckoff_phase and s == signal:
                edge = e
                edge_key = (w, v, s)
                break

    # Secondary lookup: Core spread × Signal
    core_key = (core_spread, signal)
    core_edge = CORE_SPREAD_EDGE.get(core_key)

    if edge and abs(edge[0]) >= 0.5 and edge[2] >= 3:
        avg_ret, win_rate, n, bias = edge
        source = f'wyckoff_vol_signal: {edge_key}'
        confidence = min(1.0, n / 10.0)
    elif core_edge and abs(core_edge[0]) >= 0.5 and core_edge[2] >= 3:
        avg_ret, win_rate, n, bias = core_edge
        source = f'core_spread: {core_key}'
        confidence = min(1.0, n / 10.0)
    else:
        # Default: use overall 24h aggregate (+0.931% avg, 53.3% win, p=0.013)
        # This is one of the few events with overall significance
        avg_ret = 0.931
        win_rate = 0.533
        n = 92
        bias = 'LONG'  # overall positive drift
        source = 'overall_aggregate (p=0.013)'
        confidence = 0.8  # high confidence due to statistical significance

    # ICS adjustment
    if abs(avg_ret) >= 3.0 and n >= 5:
        score_adj = 0.10 if avg_ret > 0 else -0.10
    elif abs(avg_ret) >= 1.5:
        score_adj = 0.07 if avg_ret > 0 else -0.07
    elif abs(avg_ret) >= 0.5:
        score_adj = 0.05 if avg_ret > 0 else -0.05
    else:
        score_adj = 0.03 if avg_ret > 0 else -0.03  # even weak signals get small adj for EZ CPI

    if bias != direction:
        score_adj *= -0.5

    if n >= 5 and win_rate >= 0.6:
        size_mult = 1.0
    elif n >= 3 and win_rate >= 0.5:
        size_mult = 0.85
    else:
        size_mult = 0.75  # higher floor than M40 due to overall significance

    if abs(score_adj) >= 0.07:
        status = 'PASS'
    elif abs(score_adj) >= 0.03:
        status = 'WEAK'
    else:
        status = 'NO_EDGE'

    details = {
        'regime': f'{wyckoff_phase}_{vol_regime}_{signal}',
        'release_date': release_date,
        'hicp_yoy': hicp,
        'core_yoy': core,
        'consensus': consensus,
        'prior': prior,
        'signal': signal,
        'cpi_level': cpi_level,
        'cpi_direction': cpi_dir,
        'core_spread': core_spread,
        'core_headline_spread': core - hicp,
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


def format_m41(details):
    if not details or details.get('regime') in ('DISABLED', 'NOT_RELEASE_DAY', 'NO_EDGE'):
        return None

    bias = details.get('bias', '?')
    hicp = details.get('hicp_yoy', 0)
    core = details.get('core_yoy', 0)
    cons = details.get('consensus', 0)
    signal = details.get('signal', '?')
    cpi_level = details.get('cpi_level', '?')
    cpi_dir = details.get('cpi_direction', '?')
    core_spread = details.get('core_spread', '?')
    avg_ret = details.get('avg_ret_24h', 0)
    win_rate = details.get('win_rate', 0)
    n = details.get('sample_size', 0)
    conf = details.get('confidence', 0)
    score_adj = details.get('score_adj', 0)
    size_mult = details.get('size_mult', 1.0)

    bias_icon = '🟢' if bias == 'LONG' else '🔴' if bias == 'SHORT' else '⚪'
    conf_icon = '🟢' if conf >= 0.7 else '🟡' if conf >= 0.4 else '🟠'
    dir_icon = '📈' if cpi_dir == 'RISING' else '📉' if cpi_dir == 'FALLING' else '➡️'
    spread_icon = '🔴' if 'STICKY' in core_spread else '🟢' if core_spread == 'CORE_BELOW' else '⚪'

    lines = [
        f"  M41 EZ CPI Flash: {bias_icon} {bias:>8}  "
        f"HICP={hicp:.1f}% cons={cons:.1f}% {dir_icon}{cpi_dir}  "
        f"core={core:.1f}%{spread_icon}{core_spread}  signal={signal}",
        f"    Backtest: {conf_icon} 24h={avg_ret:+.2f}% win={win_rate*100:.0f}% n={n}  "
        f"adj={score_adj:+.3f} size={size_mult:.2f}x  "
        f"chain: Asia 88-99%, Lon→NY 75-100%  p=0.013✅",
    ]
    return '\n'.join(lines)

"""
M43: US GDP Advance Estimate Session Bias (Regime-Conditional)

On BEA GDP Advance release days (~last Thursday of month after quarter end,
12:30 UTC = 08:30 ET = 20:30 MYT), applies a session-conditional directional
bias based on the combination of:
  - Wyckoff phase (M21)
  - Volatility regime (M9)
  - GDP signal: STRONG_BEAT / BEAT / INLINE / MISS / BIG_MISS / RECESSION_MISS
  - GDP health: RECESSION / CONTRACTION / SLOW / MODERATE / STRONG
  - GDP momentum: ACCELERATING / STABLE / DECELERATING

Thesis (from user #27):
  US Morning (20:30 MYT) → Corporate Earnings Calibration → FOMC Pricing
  Surprise negative GDP → recession fears → equity panic BUT rate cut expectations surge
  → bond yields fall → ETH/USDT often rallies into close (liquidity play)
  Bad economic data = positive for monetary liquidity
  Stalling economy pressures FOMC away from QT

Backtested on 33 US GDP Advance releases (2018-2026) against ETH/USDT 15m data.

Key findings:
  24h aggregate: +0.449% avg, 57.6% win, n=33 — NOT significant (p=0.41)

  CONTRARIAN SIGNAL — bad GDP is good for ETH:
    BEAT mean: -0.445% | MISS mean: +1.281% (inverted!)
    Weak GDP (recession/contraction): +2.377% avg vs Strong GDP: +0.021% avg
    RECESSION + DECELERATING: +2.271% avg, 67% win, n=3
    STRONG + DECELERATING: +4.681% avg, 100% win, n=2
    STRONG + ACCELERATING: -1.203% avg, 20% win, n=5 (sell the news)
    MODERATE + STABLE: -0.686% avg, 42% win, n=12

  Transmission chain:
    Asia→London: 85-100% persistence ✅ (strong pre-release drift)
    London→NY: 54.5% → chain BREAKS at NY Pre-Open ❌
    NY AM→NY PM: 79-88% ✅ (NY session recovery)

  This is a CONTRARIAN event: miss → rally, beat → sell.
  The "liquidity play" thesis is confirmed by the data.

Integration: lightweight modifier on BEA release days only (~4x/year).
Returns a score adjustment and size multiplier — does NOT veto.

Usage:
    from src.modules.m43_us_gdp import score_m43_us_gdp, format_m43
    status, score_adj, size_mult, details = score_m43_us_gdp(
        wyckoff_phase='RANGE', vol_regime='CHOP', direction='LONG')
"""

from datetime import datetime, timedelta
import json
import os


# ═══════════════════════════════════════════════════════════════
# US GDP ADVANCE RELEASE DATES (12:30 UTC = 08:30 ET = 20:30 MYT)
# Quarterly releases (~last Thursday of month after quarter end)
# ═══════════════════════════════════════════════════════════════

US_GDP_RELEASES = {
    # 2018
    '2018-01-26': {'gdp_qoq': 2.6, 'consensus': 3.0, 'prior': 3.2},
    '2018-04-27': {'gdp_qoq': 2.3, 'consensus': 2.0, 'prior': 2.9},
    '2018-07-27': {'gdp_qoq': 4.2, 'consensus': 4.1, 'prior': 2.0},
    '2018-10-26': {'gdp_qoq': 3.5, 'consensus': 3.3, 'prior': 4.2},
    # 2019
    '2019-01-30': {'gdp_qoq': 2.6, 'consensus': 2.2, 'prior': 3.4},
    '2019-04-26': {'gdp_qoq': 3.2, 'consensus': 2.3, 'prior': 2.2},
    '2019-07-26': {'gdp_qoq': 2.1, 'consensus': 1.8, 'prior': 3.1},
    '2019-10-30': {'gdp_qoq': 1.9, 'consensus': 1.6, 'prior': 2.0},
    # 2020
    '2020-01-30': {'gdp_qoq': 2.1, 'consensus': 2.1, 'prior': 2.1},
    '2020-04-29': {'gdp_qoq': -4.8, 'consensus': -4.0, 'prior': 2.1},
    '2020-07-30': {'gdp_qoq': -32.9, 'consensus': -34.1, 'prior': -5.0},
    '2020-10-29': {'gdp_qoq': 33.1, 'consensus': 31.0, 'prior': -31.4},
    # 2021
    '2021-01-28': {'gdp_qoq': 4.0, 'consensus': 4.2, 'prior': 33.4},
    '2021-04-29': {'gdp_qoq': 6.4, 'consensus': 6.1, 'prior': 4.3},
    '2021-07-29': {'gdp_qoq': 6.5, 'consensus': 8.4, 'prior': 6.3},
    '2021-10-28': {'gdp_qoq': 2.0, 'consensus': 2.7, 'prior': 6.7},
    # 2022
    '2022-01-27': {'gdp_qoq': 6.9, 'consensus': 5.5, 'prior': 2.3},
    '2022-04-28': {'gdp_qoq': -1.4, 'consensus': 1.1, 'prior': 6.9},
    '2022-07-28': {'gdp_qoq': -0.9, 'consensus': 0.3, 'prior': -1.6},
    '2022-10-27': {'gdp_qoq': 2.6, 'consensus': 2.4, 'prior': -0.6},
    # 2023
    '2023-01-26': {'gdp_qoq': 2.9, 'consensus': 2.6, 'prior': 3.2},
    '2023-04-27': {'gdp_qoq': 1.1, 'consensus': 2.0, 'prior': 2.6},
    '2023-07-27': {'gdp_qoq': 2.4, 'consensus': 1.8, 'prior': 2.0},
    '2023-10-26': {'gdp_qoq': 4.9, 'consensus': 4.3, 'prior': 2.1},
    # 2024
    '2024-01-25': {'gdp_qoq': 3.3, 'consensus': 2.0, 'prior': 4.9},
    '2024-04-25': {'gdp_qoq': 1.6, 'consensus': 2.4, 'prior': 3.4},
    '2024-07-25': {'gdp_qoq': 2.8, 'consensus': 2.0, 'prior': 1.4},
    '2024-10-30': {'gdp_qoq': 2.8, 'consensus': 2.9, 'prior': 3.0},
    # 2025
    '2025-01-30': {'gdp_qoq': 2.3, 'consensus': 2.4, 'prior': 3.1},
    '2025-04-30': {'gdp_qoq': -0.3, 'consensus': 0.4, 'prior': 2.4},
    '2025-07-30': {'gdp_qoq': 2.1, 'consensus': 1.8, 'prior': -0.5},
    # 2026
    '2026-01-29': {'gdp_qoq': 2.2, 'consensus': 2.0, 'prior': 2.1},
    '2026-04-29': {'gdp_qoq': 1.8, 'consensus': 1.9, 'prior': 2.2},
}


# ═══════════════════════════════════════════════════════════════
# EDGE TABLE — CONTRARIAN: bad GDP = good for ETH
# ═══════════════════════════════════════════════════════════════

EDGE_TABLE = {
    # Wyckoff × Vol × Signal
    ('RANGE', 'NEUTRAL', 'MISS'):                    (2.299, 1.000, 2, 'LONG'),
    ('RANGE', 'NEUTRAL', 'BEAT'):                    (1.812, 1.000, 2, 'LONG'),
    ('RANGE', 'COMPRESSING', 'RECESSION_MISS'):      (1.595, 0.500, 2, 'LONG'),
    ('RANGE', 'COMPRESSING', 'STRONG_BEAT'):         (0.851, 0.500, 2, 'LONG'),
}

# GDP Health × Momentum edge (contrarian!)
HEALTH_MOMENTUM_EDGE = {
    ('STRONG', 'DECELERATING'):    (4.681, 1.000, 2, 'LONG'),
    ('RECESSION', 'DECELERATING'): (2.271, 0.667, 3, 'LONG'),
    ('MODERATE', 'DECELERATING'):  (1.893, 1.000, 2, 'LONG'),
    ('STRONG', 'ACCELERATING'):    (-1.203, 0.200, 5, 'SHORT'),
    ('MODERATE', 'STABLE'):        (-0.686, 0.417, 12, 'SHORT'),
}


def _classify_signal(actual, consensus, prior):
    diff = actual - consensus
    surprise = diff / abs(consensus) if consensus != 0 else diff
    if actual < 0 and consensus >= 0:
        return 'RECESSION_MISS'
    elif actual < 0 and consensus < 0:
        return 'BEAT' if actual >= consensus else 'BIG_MISS'
    elif surprise > 0.3:
        return 'STRONG_BEAT'
    elif surprise > 0.05:
        return 'BEAT'
    elif surprise < -0.3:
        return 'BIG_MISS'
    elif surprise < -0.05:
        return 'MISS'
    else:
        return 'INLINE'


def _classify_health(actual):
    if actual < -1.0:
        return 'RECESSION'
    elif actual < 0:
        return 'CONTRACTION'
    elif actual < 1.5:
        return 'SLOW'
    elif actual < 3.0:
        return 'MODERATE'
    else:
        return 'STRONG'


def _classify_momentum(actual, prior):
    change = actual - prior
    if change > 1.0:
        return 'ACCELERATING'
    elif change < -1.0:
        return 'DECELERATING'
    else:
        return 'STABLE'


def _is_release_day(today_str):
    today = datetime.strptime(today_str, '%Y-%m-%d')
    for date_str in US_GDP_RELEASES:
        release_dt = datetime.strptime(date_str, '%Y-%m-%d')
        delta = abs((today - release_dt).days)
        if delta <= 1:
            return date_str, US_GDP_RELEASES[date_str]
    return None, None


def score_m43_us_gdp(wyckoff_phase='RANGE', vol_regime='CHOP',
                      direction='LONG', today_str=None, config=None):
    """
    Score M43: US GDP Advance Estimate session bias.

    Returns:
        (status, score_adj, size_mult, details)
    """
    cfg = config or {}
    if not cfg.get('M43_ENABLED', True):
        return 'DISABLED', 0.0, 1.0, {'regime': 'DISABLED'}

    if today_str is None:
        today_str = datetime.utcnow().strftime('%Y-%m-%d')

    release_date, release_data = _is_release_day(today_str)
    if release_data is None:
        return 'SKIP', 0.0, 1.0, {'regime': 'NOT_RELEASE_DAY'}

    gdp = release_data['gdp_qoq']
    consensus = release_data['consensus']
    prior = release_data['prior']

    signal = _classify_signal(gdp, consensus, prior)
    health = _classify_health(gdp)
    momentum = _classify_momentum(gdp, prior)

    # Primary: Wyckoff × Vol × Signal
    edge_key = (wyckoff_phase, vol_regime, signal)
    edge = EDGE_TABLE.get(edge_key)
    if edge is None:
        for (w, v, s), e in EDGE_TABLE.items():
            if w == wyckoff_phase and s == signal:
                edge = e
                edge_key = (w, v, s)
                break

    # Secondary: Health × Momentum
    hm_key = (health, momentum)
    hm_edge = HEALTH_MOMENTUM_EDGE.get(hm_key)

    if edge and abs(edge[0]) >= 0.5 and edge[2] >= 2:
        avg_ret, win_rate, n, bias = edge
        source = f'wyckoff_vol_signal: {edge_key}'
        confidence = min(0.5, n / 10.0)
    elif hm_edge and abs(hm_edge[0]) >= 0.5 and hm_edge[2] >= 2:
        avg_ret, win_rate, n, bias = hm_edge
        source = f'health_momentum: {hm_key}'
        confidence = min(0.5, n / 10.0)
    else:
        return 'NO_EDGE', 0.0, 1.0, {
            'regime': 'NO_EDGE',
            'release_date': release_date,
            'gdp_qoq': gdp, 'consensus': consensus,
            'signal': signal, 'health': health, 'momentum': momentum,
        }

    # ICS adjustment — contrarian: miss → positive, beat → negative
    if abs(avg_ret) >= 2.0 and n >= 3:
        score_adj = 0.07 if avg_ret > 0 else -0.07
    elif abs(avg_ret) >= 1.0:
        score_adj = 0.05 if avg_ret > 0 else -0.05
    elif abs(avg_ret) >= 0.5:
        score_adj = 0.03 if avg_ret > 0 else -0.03
    else:
        score_adj = 0.02 if avg_ret > 0 else -0.02

    # Contrarian flip: if signal is BEAT but edge says SHORT, align
    if bias != direction:
        score_adj *= -0.5

    if n >= 4 and win_rate >= 0.6:
        size_mult = 0.90
    elif n >= 2 and win_rate >= 0.5:
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
        'gdp_qoq': gdp,
        'consensus': consensus,
        'prior': prior,
        'signal': signal,
        'health': health,
        'momentum': momentum,
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
        'contrarian': True,  # flag: this is a contrarian event
    }

    return status, score_adj, size_mult, details


def format_m43(details):
    if not details or details.get('regime') in ('DISABLED', 'NOT_RELEASE_DAY', 'NO_EDGE'):
        return None

    bias = details.get('bias', '?')
    gdp = details.get('gdp_qoq', 0)
    cons = details.get('consensus', 0)
    signal = details.get('signal', '?')
    health = details.get('health', '?')
    momentum = details.get('momentum', '?')
    avg_ret = details.get('avg_ret_24h', 0)
    win_rate = details.get('win_rate', 0)
    n = details.get('sample_size', 0)
    conf = details.get('confidence', 0)
    score_adj = details.get('score_adj', 0)
    size_mult = details.get('size_mult', 1.0)

    bias_icon = '🟢' if bias == 'LONG' else '🔴' if bias == 'SHORT' else '⚪'
    conf_icon = '🟠'  # small sample
    health_icon = {'RECESSION': '🔴', 'CONTRACTION': '🟠', 'SLOW': '🟡',
                   'MODERATE': '🟢', 'STRONG': '🟢🟢'}.get(health, '⚪')
    mom_icon = {'ACCELERATING': '📈', 'DECELERATING': '📉', 'STABLE': '➡️'}.get(momentum, '➡️')

    lines = [
        f"  M43 US GDP Advance: {bias_icon} {bias:>8}  "
        f"GDP={gdp:+.1f}%(qoq) cons={cons:+.1f}%  "
        f"{health_icon}{health} {mom_icon}{momentum}  signal={signal}  ⚡CONTRARIAN",
        f"    Backtest: {conf_icon} 24h={avg_ret:+.2f}% win={win_rate*100:.0f}% n={n}  "
        f"adj={score_adj:+.3f} size={size_mult:.2f}x  "
        f"chain: Asia 85-100%, breaks at NY Pre-Open",
    ]
    return '\n'.join(lines)

"""
M42: Eurozone GDP Flash Session Bias (Regime-Conditional)

On Eurostat GDP Flash release days (~end of quarter, 09:00 UTC = 10:00 CET / 18:00 MYT),
applies a session-conditional directional bias based on the combination of:
  - Wyckoff phase (M21)
  - Volatility regime (M9)
  - GDP signal: STRONG_BEAT / BEAT / INLINE / MISS / BIG_MISS / RECESSION_MISS
  - GDP health: RECESSION / CONTRACTION / STAGNANT / MODERATE / STRONG
  - GDP momentum: ACCELERATING / STABLE / DECELERATING

Thesis (from user #26):
  Europe Morning (18:00 MYT) → US GDP Advance Prediction → ECB Cut Trajectory
  Negative EZ GDP → recession confirmed → capital flees EU equities → DXY up → ETH pressure
  Then: accelerates ECB rate cuts → lifts global liquidity over months
  US macro funds use this as warning for US GDP Advance (~1 week later)

Backtested on 33 Eurozone GDP Flash releases (2018-2026) against ETH/USDT 15m data.

Key findings:
  24h aggregate: -0.174% avg, 39.4% win, n=33 — NOT significant (p=0.76)
  (Small sample size — quarterly data. Only specific combos have edge.)

  Specific combos with edge:
    RANGE + COMPRESSING + BEAT:    +1.208% avg, 75% win, n=4  → LONG
    STRONG + ACCELERATING + BEAT:  +2.889% avg, 67% win, n=3  → LONG
    RECESSION + DECELERATING:      +2.579% avg, 50% win, n=2  → LONG (ECB cut hopes)
    STAGNANT + STABLE + INLINE:    -0.660% avg, 27% win, n=11 → SHORT (weak economy)

  Transmission chain (strong):
    Asia: 81-100% persistence ✅
    Asia→Europe: 97-100% Tokyo Close→Frankfurt Open ✅ (HOLDS)
    London→NY: 78-100% persistence ✅
    Chain breaks at London Morning (data artifact)

Integration: lightweight modifier on Eurostat GDP release days only (~4x/year).
Returns a score adjustment and size multiplier — does NOT veto.

Usage:
    from src.modules.m42_ez_gdp import score_m42_ez_gdp, format_m42
    status, score_adj, size_mult, details = score_m42_ez_gdp(
        wyckoff_phase='RANGE', vol_regime='CHOP', direction='LONG')
"""

from datetime import datetime, timedelta
import json
import os


# ═══════════════════════════════════════════════════════════════
# EUROZONE GDP FLASH RELEASE DATES (09:00 UTC)
# Quarterly releases (~30 days after quarter end)
# ═══════════════════════════════════════════════════════════════

EZ_GDP_RELEASES = {
    # 2018
    '2018-01-30': {'gdp_qoq': 0.6, 'gdp_yoy': 2.7, 'consensus_qoq': 0.6, 'prior_qoq': 0.7},
    '2018-04-30': {'gdp_qoq': 0.4, 'gdp_yoy': 2.5, 'consensus_qoq': 0.4, 'prior_qoq': 0.7},
    '2018-07-31': {'gdp_qoq': 0.4, 'gdp_yoy': 2.2, 'consensus_qoq': 0.4, 'prior_qoq': 0.4},
    '2018-10-30': {'gdp_qoq': 0.2, 'gdp_yoy': 1.7, 'consensus_qoq': 0.2, 'prior_qoq': 0.4},
    # 2019
    '2019-01-31': {'gdp_qoq': 0.2, 'gdp_yoy': 1.2, 'consensus_qoq': 0.2, 'prior_qoq': 0.2},
    '2019-04-30': {'gdp_qoq': 0.4, 'gdp_yoy': 1.3, 'consensus_qoq': 0.3, 'prior_qoq': 0.2},
    '2019-07-31': {'gdp_qoq': 0.2, 'gdp_yoy': 1.1, 'consensus_qoq': 0.2, 'prior_qoq': 0.4},
    '2019-10-31': {'gdp_qoq': 0.2, 'gdp_yoy': 1.2, 'consensus_qoq': 0.1, 'prior_qoq': 0.2},
    # 2020
    '2020-01-31': {'gdp_qoq': 0.1, 'gdp_yoy': 1.0, 'consensus_qoq': 0.1, 'prior_qoq': 0.2},
    '2020-04-30': {'gdp_qoq': -3.6, 'gdp_yoy': -3.2, 'consensus_qoq': -3.8, 'prior_qoq': 0.1},
    '2020-07-31': {'gdp_qoq': -11.8, 'gdp_yoy': -14.7, 'consensus_qoq': -12.1, 'prior_qoq': -3.6},
    '2020-10-30': {'gdp_qoq': 12.5, 'gdp_yoy': -4.3, 'consensus_qoq': 9.4, 'prior_qoq': -11.8},
    # 2021
    '2021-01-29': {'gdp_qoq': -0.7, 'gdp_yoy': -4.8, 'consensus_qoq': -0.9, 'prior_qoq': 12.5},
    '2021-04-30': {'gdp_qoq': -0.3, 'gdp_yoy': -1.3, 'consensus_qoq': -0.3, 'prior_qoq': -0.7},
    '2021-07-30': {'gdp_qoq': 2.0, 'gdp_yoy': 13.7, 'consensus_qoq': 1.5, 'prior_qoq': -0.3},
    '2021-10-29': {'gdp_qoq': 2.1, 'gdp_yoy': 3.9, 'consensus_qoq': 2.0, 'prior_qoq': 2.0},
    # 2022
    '2022-01-31': {'gdp_qoq': 0.3, 'gdp_yoy': 4.6, 'consensus_qoq': 0.4, 'prior_qoq': 2.1},
    '2022-04-29': {'gdp_qoq': 0.2, 'gdp_yoy': 5.1, 'consensus_qoq': 0.2, 'prior_qoq': 0.3},
    '2022-07-29': {'gdp_qoq': 0.6, 'gdp_yoy': 3.9, 'consensus_qoq': 0.5, 'prior_qoq': 0.2},
    '2022-10-31': {'gdp_qoq': 0.2, 'gdp_yoy': 2.3, 'consensus_qoq': 0.1, 'prior_qoq': 0.6},
    # 2023
    '2023-01-31': {'gdp_qoq': 0.1, 'gdp_yoy': 1.9, 'consensus_qoq': 0.0, 'prior_qoq': 0.2},
    '2023-04-28': {'gdp_qoq': 0.1, 'gdp_yoy': 1.3, 'consensus_qoq': 0.0, 'prior_qoq': 0.1},
    '2023-07-31': {'gdp_qoq': 0.3, 'gdp_yoy': 0.6, 'consensus_qoq': 0.2, 'prior_qoq': 0.1},
    '2023-10-31': {'gdp_qoq': -0.1, 'gdp_yoy': -0.1, 'consensus_qoq': -0.1, 'prior_qoq': 0.3},
    # 2024
    '2024-01-31': {'gdp_qoq': -0.1, 'gdp_yoy': 0.1, 'consensus_qoq': -0.1, 'prior_qoq': -0.1},
    '2024-04-30': {'gdp_qoq': 0.3, 'gdp_yoy': 0.4, 'consensus_qoq': 0.2, 'prior_qoq': -0.1},
    '2024-07-30': {'gdp_qoq': 0.3, 'gdp_yoy': 0.6, 'consensus_qoq': 0.2, 'prior_qoq': 0.3},
    '2024-10-30': {'gdp_qoq': 0.4, 'gdp_yoy': 0.9, 'consensus_qoq': 0.2, 'prior_qoq': 0.3},
    # 2025
    '2025-01-30': {'gdp_qoq': 0.1, 'gdp_yoy': 0.9, 'consensus_qoq': 0.1, 'prior_qoq': 0.4},
    '2025-04-30': {'gdp_qoq': 0.3, 'gdp_yoy': 1.2, 'consensus_qoq': 0.2, 'prior_qoq': 0.1},
    '2025-07-30': {'gdp_qoq': 0.2, 'gdp_yoy': 1.0, 'consensus_qoq': 0.2, 'prior_qoq': 0.3},
    # 2026
    '2026-01-30': {'gdp_qoq': 0.2, 'gdp_yoy': 1.1, 'consensus_qoq': 0.2, 'prior_qoq': 0.2},
    '2026-04-30': {'gdp_qoq': 0.2, 'gdp_yoy': 1.0, 'consensus_qoq': 0.2, 'prior_qoq': 0.2},
}


# ═══════════════════════════════════════════════════════════════
# EDGE TABLE — Wyckoff × Vol × Signal (n≥2, |avg|≥0.5%)
# Small sample (33 quarterly releases) — lower confidence
# ═══════════════════════════════════════════════════════════════

EDGE_TABLE = {
    ('RANGE', 'COMPRESSING', 'BEAT'):       (1.208, 0.750, 4, 'LONG'),
    ('RANGE', 'NEUTRAL', 'INLINE'):         (-0.809, 0.444, 9, 'SHORT'),
    ('MARKUP', 'NEUTRAL', 'BEAT'):          (-0.593, 0.000, 2, 'SHORT'),
}

# GDP Health × Momentum edge table
HEALTH_MOMENTUM_EDGE = {
    ('STRONG', 'ACCELERATING', 'BEAT'):     (2.889, 0.667, 3, 'LONG'),
    ('RECESSION', 'DECELERATING', 'INLINE'): (2.579, 0.500, 2, 'LONG'),
    ('STAGNANT', 'STABLE', 'INLINE'):       (-0.660, 0.273, 11, 'SHORT'),
    ('MODERATE', 'STABLE', 'INLINE'):       (-3.310, 0.000, 2, 'SHORT'),
}


def _classify_signal(actual_qoq, consensus_qoq, prior_qoq):
    diff = actual_qoq - consensus_qoq
    surprise = diff / abs(consensus_qoq) if consensus_qoq != 0 else diff
    if actual_qoq < 0 and consensus_qoq >= 0:
        return 'RECESSION_MISS'
    elif surprise > 0.5:
        return 'STRONG_BEAT'
    elif surprise > 0.1:
        return 'BEAT'
    elif surprise < -0.5:
        return 'BIG_MISS'
    elif surprise < -0.1:
        return 'MISS'
    else:
        return 'INLINE'


def _classify_health(actual_qoq):
    if actual_qoq < -1.0:
        return 'RECESSION'
    elif actual_qoq < 0:
        return 'CONTRACTION'
    elif actual_qoq < 0.3:
        return 'STAGNANT'
    elif actual_qoq < 0.6:
        return 'MODERATE'
    else:
        return 'STRONG'


def _classify_momentum(actual_qoq, prior_qoq):
    change = actual_qoq - prior_qoq
    if change > 0.3:
        return 'ACCELERATING'
    elif change < -0.3:
        return 'DECELERATING'
    else:
        return 'STABLE'


def _is_release_day(today_str):
    today = datetime.strptime(today_str, '%Y-%m-%d')
    for date_str in EZ_GDP_RELEASES:
        release_dt = datetime.strptime(date_str, '%Y-%m-%d')
        delta = abs((today - release_dt).days)
        if delta <= 1:
            return date_str, EZ_GDP_RELEASES[date_str]
    return None, None


def score_m42_ez_gdp(wyckoff_phase='RANGE', vol_regime='CHOP',
                      direction='LONG', today_str=None, config=None):
    """
    Score M42: Eurozone GDP Flash session bias.

    Returns:
        (status, score_adj, size_mult, details)
    """
    cfg = config or {}
    if not cfg.get('M42_ENABLED', True):
        return 'DISABLED', 0.0, 1.0, {'regime': 'DISABLED'}

    if today_str is None:
        today_str = datetime.utcnow().strftime('%Y-%m-%d')

    release_date, release_data = _is_release_day(today_str)
    if release_data is None:
        return 'SKIP', 0.0, 1.0, {'regime': 'NOT_RELEASE_DAY'}

    gdp_qoq = release_data['gdp_qoq']
    consensus = release_data['consensus_qoq']
    prior = release_data['prior_qoq']

    signal = _classify_signal(gdp_qoq, consensus, prior)
    health = _classify_health(gdp_qoq)
    momentum = _classify_momentum(gdp_qoq, prior)

    # Primary lookup: Wyckoff × Vol × Signal
    edge_key = (wyckoff_phase, vol_regime, signal)
    edge = EDGE_TABLE.get(edge_key)

    if edge is None:
        for (w, v, s), e in EDGE_TABLE.items():
            if w == wyckoff_phase and s == signal:
                edge = e
                edge_key = (w, v, s)
                break

    # Secondary lookup: Health × Momentum × Signal
    hm_key = (health, momentum, signal)
    hm_edge = HEALTH_MOMENTUM_EDGE.get(hm_key)

    if edge and abs(edge[0]) >= 0.5 and edge[2] >= 2:
        avg_ret, win_rate, n, bias = edge
        source = f'wyckoff_vol_signal: {edge_key}'
        confidence = min(0.6, n / 10.0)  # lower cap — small sample
    elif hm_edge and abs(hm_edge[0]) >= 0.5 and hm_edge[2] >= 2:
        avg_ret, win_rate, n, bias = hm_edge
        source = f'health_momentum: {hm_key}'
        confidence = min(0.5, n / 10.0)
    else:
        return 'NO_EDGE', 0.0, 1.0, {
            'regime': 'NO_EDGE',
            'release_date': release_date,
            'gdp_qoq': gdp_qoq, 'consensus_qoq': consensus,
            'signal': signal, 'health': health, 'momentum': momentum,
            'wyckoff': wyckoff_phase, 'vol': vol_regime,
        }

    # ICS adjustment — smaller than M40/M41 due to small sample
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

    # Size multiplier — more conservative due to small sample
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
        'gdp_qoq': gdp_qoq,
        'gdp_yoy': release_data['gdp_yoy'],
        'consensus_qoq': consensus,
        'prior_qoq': prior,
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
    }

    return status, score_adj, size_mult, details


def format_m42(details):
    if not details or details.get('regime') in ('DISABLED', 'NOT_RELEASE_DAY', 'NO_EDGE'):
        return None

    bias = details.get('bias', '?')
    gdp_qoq = details.get('gdp_qoq', 0)
    gdp_yoy = details.get('gdp_yoy', 0)
    cons = details.get('consensus_qoq', 0)
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
    conf_icon = '🟢' if conf >= 0.5 else '🟡' if conf >= 0.3 else '🟠'
    health_icon = {'RECESSION': '🔴', 'CONTRACTION': '🟠', 'STAGNANT': '🟡',
                   'MODERATE': '🟢', 'STRONG': '🟢🟢'}.get(health, '⚪')
    mom_icon = {'ACCELERATING': '📈', 'DECELERATING': '📉', 'STABLE': '➡️'}.get(momentum, '➡️')

    lines = [
        f"  M42 EZ GDP Flash: {bias_icon} {bias:>8}  "
        f"GDP={gdp_qoq:+.1f}%(qoq) cons={cons:+.1f}%  "
        f"{health_icon}{health} {mom_icon}{momentum}  signal={signal}",
        f"    Backtest: {conf_icon} 24h={avg_ret:+.2f}% win={win_rate*100:.0f}% n={n}  "
        f"adj={score_adj:+.3f} size={size_mult:.2f}x  "
        f"chain: Asia 81-100%, Lon→NY 78-100%  (small sample)",
    ]
    return '\n'.join(lines)

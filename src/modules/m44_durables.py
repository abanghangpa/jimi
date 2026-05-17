"""
M44: US Durable Goods Orders Session Bias (Regime-Conditional)

On Census Bureau Durable Goods Orders release days (~25th-27th of month,
12:30 UTC = 08:30 ET = 20:30 MYT), applies a session-conditional directional
bias based on the combination of:
  - Wyckoff phase (M21)
  - Volatility regime (M9)
  - Durable goods signal: STRONG_BEAT / BEAT / INLINE / MISS / BIG_MISS
  - Capex health: BOOMING / GROWING / STABLE / WEAK / COLLAPSING
  - Momentum: ACCELERATING / STABLE / DECELERATING

Thesis (from user #28):
  US Morning (20:30 MYT) → Factory Orders (1 week) → ISM Manufacturing
  Drop in durable goods → corporations pulling back on capex → macro slowdown
  → lower treasury yields → steady support for ETH
  Rise in durable goods → corporate confidence → risk-on but tighter Fed

Backtested on 91 US Durable Goods Orders releases (2018-2026) against ETH/USDT 15m data.

Key findings:
  24h aggregate: -0.597% avg, 47.3% win, n=91 — NOT significant (p=0.18)
  Negative bias overall — durables tend to coincide with ETH weakness

  Specific combos with edge (n≥3, |avg|≥0.5%):
    ACCUMULATION + NEUTRAL + BIG_MISS:   +5.481% avg, 100% win, n=3  → LONG
    DISTRIBUTION + NEUTRAL + STRONG_BEAT: -3.228% avg, 0% win, n=3  → SHORT
    RANGE + NEUTRAL + STRONG_BEAT:       -2.448% avg, 40% win, n=10 → SHORT
    RANGE + COMPRESSING + MISS:          +0.757% avg, 67% win, n=3  → LONG
    RANGE + COMPRESSING + BIG_MISS:      +0.671% avg, 75% win, n=4  → LONG
    STABLE + STABLE + BEAT:              +4.392% avg, 100% win, n=3 → LONG
    STABLE + DECELERATING + MISS:        +2.692% avg, 100% win, n=3 → LONG

  Transmission chain:
    Asia→London: 88-100% persistence ✅
    Chain BREAKS at London Midday→NY Pre-Open (48.4%) ❌
    NY Overlap→NY PM: 67-87% ✅ (partial recovery)

  This is a SELL THE STRONG BEAT event at the 24h level.

Integration: lightweight modifier on Census Bureau release days only (~12x/year.
Returns a score adjustment and size multiplier — does NOT veto.

Usage:
    from src.modules.m44_durables import score_m44_durables, format_m44
    status, score_adj, size_mult, details = score_m44_durables(
        wyckoff_phase='RANGE', vol_regime='CHOP', direction='LONG')
"""

from datetime import datetime, timedelta
import json
import os


DURABLE_GOODS_RELEASES = {
    # 2018
    '2018-01-26': {'headline_mom': 0.7, 'core_mom': 0.7, 'consensus': 1.5, 'prior': 1.7},
    '2018-02-27': {'headline_mom': -1.2, 'core_mom': -0.2, 'consensus': -0.5, 'prior': 2.8},
    '2018-03-26': {'headline_mom': 3.1, 'core_mom': 1.2, 'consensus': 1.6, 'prior': -1.8},
    '2018-04-26': {'headline_mom': 1.7, 'core_mom': 0.0, 'consensus': 1.3, 'prior': 2.7},
    '2018-05-25': {'headline_mom': 1.6, 'core_mom': 0.9, 'consensus': 0.8, 'prior': 2.7},
    '2018-06-27': {'headline_mom': -0.4, 'core_mom': 0.3, 'consensus': 0.0, 'prior': -0.6},
    '2018-07-26': {'headline_mom': -1.7, 'core_mom': 0.4, 'consensus': 0.8, 'prior': 0.7},
    '2018-08-24': {'headline_mom': -1.7, 'core_mom': 1.4, 'consensus': -0.5, 'prior': -1.2},
    '2018-09-27': {'headline_mom': 0.8, 'core_mom': -0.2, 'consensus': 1.0, 'prior': -2.6},
    '2018-10-25': {'headline_mom': -4.4, 'core_mom': -0.1, 'consensus': -1.5, 'prior': 0.6},
    '2018-11-21': {'headline_mom': -4.3, 'core_mom': -0.1, 'consensus': -2.5, 'prior': 0.1},
    '2018-12-21': {'headline_mom': 0.8, 'core_mom': -0.3, 'consensus': 1.6, 'prior': -4.3},
    # 2019
    '2019-01-25': {'headline_mom': 1.2, 'core_mom': 0.1, 'consensus': 1.7, 'prior': 0.7},
    '2019-02-27': {'headline_mom': 0.4, 'core_mom': -0.1, 'consensus': -0.5, 'prior': 1.3},
    '2019-03-26': {'headline_mom': -1.6, 'core_mom': -0.1, 'consensus': -1.8, 'prior': 0.1},
    '2019-04-25': {'headline_mom': 2.7, 'core_mom': 0.3, 'consensus': 0.8, 'prior': -1.1},
    '2019-05-24': {'headline_mom': -2.1, 'core_mom': 0.0, 'consensus': -2.0, 'prior': 1.7},
    '2019-06-26': {'headline_mom': -1.3, 'core_mom': 0.3, 'consensus': -0.2, 'prior': -2.8},
    '2019-07-25': {'headline_mom': 2.0, 'core_mom': 1.2, 'consensus': 1.2, 'prior': 1.9},
    '2019-08-26': {'headline_mom': 0.2, 'core_mom': 0.5, 'consensus': -1.0, 'prior': 2.0},
    '2019-09-25': {'headline_mom': 0.2, 'core_mom': -0.5, 'consensus': 0.0, 'prior': 0.2},
    '2019-10-25': {'headline_mom': -1.2, 'core_mom': -0.4, 'consensus': -0.7, 'prior': -1.4},
    '2019-11-27': {'headline_mom': 0.6, 'core_mom': 0.5, 'consensus': -1.1, 'prior': 0.2},
    '2019-12-23': {'headline_mom': -2.0, 'core_mom': -0.1, 'consensus': 1.5, 'prior': 3.2},
    # 2020
    '2020-01-28': {'headline_mom': -0.9, 'core_mom': -0.6, 'consensus': 1.2, 'prior': 2.4},
    '2020-02-27': {'headline_mom': 1.2, 'core_mom': -0.8, 'consensus': -0.8, 'prior': -2.4},
    '2020-03-25': {'headline_mom': 1.2, 'core_mom': -0.2, 'consensus': -0.8, 'prior': 0.1},
    '2020-04-24': {'headline_mom': -14.7, 'core_mom': -0.3, 'consensus': -12.0, 'prior': -0.7},
    '2020-05-28': {'headline_mom': -17.2, 'core_mom': -1.4, 'consensus': -18.0, 'prior': -16.6},
    '2020-06-25': {'headline_mom': 15.7, 'core_mom': 2.3, 'consensus': 10.9, 'prior': -18.3},
    '2020-07-27': {'headline_mom': 11.2, 'core_mom': 4.3, 'consensus': 7.2, 'prior': 7.6},
    '2020-08-26': {'headline_mom': 0.4, 'core_mom': 1.8, 'consensus': 1.9, 'prior': 11.7},
    '2020-09-25': {'headline_mom': 0.4, 'core_mom': 0.6, 'consensus': 0.5, 'prior': 0.5},
    '2020-10-27': {'headline_mom': 1.9, 'core_mom': 1.0, 'consensus': 0.5, 'prior': -0.4},
    '2020-11-25': {'headline_mom': 1.3, 'core_mom': 0.4, 'consensus': 0.8, 'prior': 1.8},
    '2020-12-23': {'headline_mom': 1.0, 'core_mom': 0.7, 'consensus': 0.6, 'prior': 1.3},
    # 2021
    '2021-01-27': {'headline_mom': 0.2, 'core_mom': 1.1, 'consensus': 0.9, 'prior': 1.3},
    '2021-02-25': {'headline_mom': 3.4, 'core_mom': -0.9, 'consensus': 1.1, 'prior': 1.2},
    '2021-03-24': {'headline_mom': -1.2, 'core_mom': 0.5, 'consensus': 0.8, 'prior': -1.2},
    '2021-04-26': {'headline_mom': 0.8, 'core_mom': 1.0, 'consensus': 2.5, 'prior': -1.7},
    '2021-05-27': {'headline_mom': -1.3, 'core_mom': 0.4, 'consensus': 0.8, 'prior': 1.2},
    '2021-06-24': {'headline_mom': 2.3, 'core_mom': 0.8, 'consensus': 2.8, 'prior': 0.0},
    '2021-07-27': {'headline_mom': -0.1, 'core_mom': 1.0, 'consensus': 0.8, 'prior': 0.9},
    '2021-08-25': {'headline_mom': -0.1, 'core_mom': 0.3, 'consensus': -0.3, 'prior': -0.5},
    '2021-09-27': {'headline_mom': 1.8, 'core_mom': 0.5, 'consensus': 0.6, 'prior': 0.5},
    '2021-10-27': {'headline_mom': -0.4, 'core_mom': 0.5, 'consensus': -0.2, 'prior': 0.4},
    '2021-11-24': {'headline_mom': -0.4, 'core_mom': 0.3, 'consensus': 0.6, 'prior': 0.4},
    '2021-12-23': {'headline_mom': 2.5, 'core_mom': 0.3, 'consensus': 1.6, 'prior': 1.7},
    # 2022
    '2022-01-27': {'headline_mom': 1.6, 'core_mom': 0.9, 'consensus': 0.8, 'prior': 2.6},
    '2022-02-25': {'headline_mom': 2.2, 'core_mom': 0.0, 'consensus': 1.0, 'prior': 1.6},
    '2022-03-24': {'headline_mom': -2.2, 'core_mom': 1.0, 'consensus': -0.6, 'prior': 1.6},
    '2022-04-26': {'headline_mom': 0.4, 'core_mom': 0.3, 'consensus': 0.6, 'prior': 0.5},
    '2022-05-25': {'headline_mom': 0.4, 'core_mom': 0.3, 'consensus': 0.6, 'prior': 0.5},
    '2022-06-27': {'headline_mom': 1.9, 'core_mom': 0.5, 'consensus': 0.5, 'prior': 0.7},
    '2022-07-27': {'headline_mom': -1.2, 'core_mom': 0.4, 'consensus': -0.5, 'prior': 2.0},
    '2022-08-24': {'headline_mom': -0.1, 'core_mom': -0.3, 'consensus': -0.2, 'prior': 1.9},
    '2022-09-27': {'headline_mom': -0.2, 'core_mom': -0.7, 'consensus': 0.2, 'prior': -0.1},
    '2022-10-27': {'headline_mom': 0.4, 'core_mom': -0.5, 'consensus': 0.3, 'prior': 0.2},
    '2022-11-23': {'headline_mom': 1.0, 'core_mom': 0.5, 'consensus': 0.0, 'prior': 0.3},
    '2022-12-23': {'headline_mom': -1.7, 'core_mom': -0.1, 'consensus': -0.6, 'prior': -1.1},
    # 2023
    '2023-01-26': {'headline_mom': 5.6, 'core_mom': -0.1, 'consensus': 2.4, 'prior': -5.1},
    '2023-02-27': {'headline_mom': -4.5, 'core_mom': 0.1, 'consensus': -3.6, 'prior': -5.0},
    '2023-03-24': {'headline_mom': -1.0, 'core_mom': -0.3, 'consensus': -0.6, 'prior': -5.0},
    '2023-04-26': {'headline_mom': 3.2, 'core_mom': 0.3, 'consensus': 0.8, 'prior': -1.2},
    '2023-05-24': {'headline_mom': 1.1, 'core_mom': -0.2, 'consensus': -1.0, 'prior': 3.3},
    '2023-06-27': {'headline_mom': 1.7, 'core_mom': 0.7, 'consensus': -1.0, 'prior': 0.3},
    '2023-07-27': {'headline_mom': -5.2, 'core_mom': 0.5, 'consensus': -3.6, 'prior': 4.6},
    '2023-08-24': {'headline_mom': -5.2, 'core_mom': 0.5, 'consensus': -4.0, 'prior': 4.4},
    '2023-09-26': {'headline_mom': -0.1, 'core_mom': 0.6, 'consensus': -0.5, 'prior': -5.6},
    '2023-10-25': {'headline_mom': -4.6, 'core_mom': -0.5, 'consensus': -3.1, 'prior': 4.6},
    '2023-11-22': {'headline_mom': -5.4, 'core_mom': 0.0, 'consensus': -3.1, 'prior': 4.0},
    '2023-12-22': {'headline_mom': 5.4, 'core_mom': 0.8, 'consensus': 2.1, 'prior': -5.1},
    # 2024
    '2024-01-25': {'headline_mom': -6.1, 'core_mom': 0.2, 'consensus': -4.5, 'prior': 5.5},
    '2024-02-27': {'headline_mom': -6.1, 'core_mom': 0.2, 'consensus': -4.5, 'prior': -0.3},
    '2024-03-26': {'headline_mom': 1.4, 'core_mom': 0.7, 'consensus': 1.0, 'prior': -6.9},
    '2024-04-24': {'headline_mom': 0.7, 'core_mom': 0.2, 'consensus': 0.3, 'prior': 2.6},
    '2024-05-24': {'headline_mom': 0.7, 'core_mom': 0.3, 'consensus': 0.3, 'prior': 0.8},
    '2024-06-27': {'headline_mom': 0.7, 'core_mom': 0.5, 'consensus': 0.2, 'prior': 0.6},
    '2024-07-25': {'headline_mom': -6.7, 'core_mom': 0.5, 'consensus': 0.3, 'prior': 0.7},
    '2024-08-26': {'headline_mom': 9.8, 'core_mom': -0.2, 'consensus': 5.0, 'prior': -6.7},
    '2024-09-26': {'headline_mom': -0.8, 'core_mom': 0.5, 'consensus': -1.0, 'prior': 9.8},
    '2024-10-25': {'headline_mom': -0.8, 'core_mom': 0.4, 'consensus': -1.0, 'prior': -0.8},
    '2024-11-27': {'headline_mom': 0.2, 'core_mom': 0.2, 'consensus': 0.1, 'prior': -0.4},
    '2024-12-23': {'headline_mom': -1.1, 'core_mom': 0.4, 'consensus': 0.4, 'prior': 0.8},
    # 2025
    '2025-01-28': {'headline_mom': 2.6, 'core_mom': 0.3, 'consensus': 2.0, 'prior': -2.0},
    '2025-02-26': {'headline_mom': 3.1, 'core_mom': 0.2, 'consensus': 2.0, 'prior': 2.0},
    '2025-03-26': {'headline_mom': -0.9, 'core_mom': 0.7, 'consensus': -1.0, 'prior': 3.2},
    '2025-04-25': {'headline_mom': 2.3, 'core_mom': 0.5, 'consensus': 1.5, 'prior': -0.2},
    '2025-05-27': {'headline_mom': -1.2, 'core_mom': 0.3, 'consensus': -0.5, 'prior': 2.8},
    # 2026
    '2026-01-27': {'headline_mom': 0.8, 'core_mom': 0.4, 'consensus': 0.5, 'prior': 0.6},
    '2026-02-25': {'headline_mom': 0.5, 'core_mom': 0.3, 'consensus': 0.4, 'prior': 0.8},
}


# Edge table: Wyckoff × Vol × Signal (n≥3, |avg|≥0.5%)
EDGE_TABLE = {
    ('ACCUMULATION', 'NEUTRAL', 'BIG_MISS'):          (5.481, 1.000, 3, 'LONG'),
    ('DISTRIBUTION', 'NEUTRAL', 'STRONG_BEAT'):       (-3.228, 0.000, 3, 'SHORT'),
    ('RANGE', 'NEUTRAL', 'STRONG_BEAT'):              (-2.448, 0.400, 10, 'SHORT'),
    ('RANGE', 'COMPRESSING', 'MISS'):                 (0.757, 0.667, 3, 'LONG'),
    ('RANGE', 'COMPRESSING', 'BIG_MISS'):             (0.671, 0.750, 4, 'LONG'),
    ('RANGE', 'NEUTRAL', 'BIG_MISS'):                 (-0.576, 0.636, 11, 'SHORT'),
}

# Capex Health × Momentum edge
CAPEX_MOMENTUM_EDGE = {
    ('STABLE', 'STABLE', 'BEAT'):                     (4.392, 1.000, 3, 'LONG'),
    ('STABLE', 'DECELERATING', 'MISS'):               (2.692, 1.000, 3, 'LONG'),
    ('BOOMING', 'DECELERATING', 'BIG_MISS'):          (3.761, 1.000, 2, 'LONG'),
    ('BOOMING', 'ACCELERATING', 'STRONG_BEAT'):       (-4.388, 0.000, 3, 'SHORT'),
    ('STABLE', 'ACCELERATING', 'STRONG_BEAT'):        (-3.215, 0.000, 2, 'SHORT'),
    ('GROWING', 'STABLE', 'STRONG_BEAT'):             (-1.235, 0.400, 15, 'SHORT'),
    ('GROWING', 'STABLE', 'BIG_MISS'):                (-1.308, 0.667, 6, 'SHORT'),
}


def _classify_signal(actual, consensus):
    diff = actual - consensus
    surprise = diff / abs(consensus) if consensus != 0 else diff
    if surprise > 0.5:
        return 'STRONG_BEAT'
    elif surprise > 0.1:
        return 'BEAT'
    elif surprise < -0.5:
        return 'BIG_MISS'
    elif surprise < -0.1:
        return 'MISS'
    else:
        return 'INLINE'


def _classify_capex(core_mom):
    if core_mom >= 1.0:
        return 'BOOMING'
    elif core_mom >= 0.3:
        return 'GROWING'
    elif core_mom >= -0.3:
        return 'STABLE'
    elif core_mom >= -1.0:
        return 'WEAK'
    else:
        return 'COLLAPSING'


def _classify_momentum(actual, prior):
    change = actual - prior
    if change > 2.0:
        return 'ACCELERATING'
    elif change < -2.0:
        return 'DECELERATING'
    else:
        return 'STABLE'


def _is_release_day(today_str):
    today = datetime.strptime(today_str, '%Y-%m-%d')
    for date_str in DURABLE_GOODS_RELEASES:
        release_dt = datetime.strptime(date_str, '%Y-%m-%d')
        delta = abs((today - release_dt).days)
        if delta <= 1:
            return date_str, DURABLE_GOODS_RELEASES[date_str]
    return None, None


def score_m44_durables(wyckoff_phase='RANGE', vol_regime='CHOP',
                        direction='LONG', today_str=None, config=None):
    """Score M44: US Durable Goods Orders session bias."""
    cfg = config or {}
    if not cfg.get('M44_ENABLED', True):
        return 'DISABLED', 0.0, 1.0, {'regime': 'DISABLED'}

    if today_str is None:
        today_str = datetime.utcnow().strftime('%Y-%m-%d')

    release_date, release_data = _is_release_day(today_str)
    if release_data is None:
        return 'SKIP', 0.0, 1.0, {'regime': 'NOT_RELEASE_DAY'}

    headline = release_data['headline_mom']
    core = release_data['core_mom']
    consensus = release_data['consensus']
    prior = release_data['prior']

    signal = _classify_signal(headline, consensus)
    capex = _classify_capex(core)
    momentum = _classify_momentum(headline, prior)

    # Primary: Wyckoff × Vol × Signal
    edge_key = (wyckoff_phase, vol_regime, signal)
    edge = EDGE_TABLE.get(edge_key)
    if edge is None:
        for (w, v, s), e in EDGE_TABLE.items():
            if w == wyckoff_phase and s == signal:
                edge = e
                edge_key = (w, v, s)
                break

    # Secondary: Capex × Momentum × Signal
    cm_key = (capex, momentum, signal)
    cm_edge = CAPEX_MOMENTUM_EDGE.get(cm_key)

    if edge and abs(edge[0]) >= 0.5 and edge[2] >= 3:
        avg_ret, win_rate, n, bias = edge
        source = f'wyckoff_vol_signal: {edge_key}'
        confidence = min(0.6, n / 10.0)
    elif cm_edge and abs(cm_edge[0]) >= 0.5 and cm_edge[2] >= 2:
        avg_ret, win_rate, n, bias = cm_edge
        source = f'capex_momentum: {cm_key}'
        confidence = min(0.5, n / 10.0)
    else:
        return 'NO_EDGE', 0.0, 1.0, {
            'regime': 'NO_EDGE', 'release_date': release_date,
            'headline_mom': headline, 'core_mom': core, 'consensus': consensus,
            'signal': signal, 'capex_health': capex, 'momentum': momentum,
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
        'headline_mom': headline, 'core_mom': core,
        'consensus': consensus, 'prior': prior,
        'signal': signal, 'capex_health': capex, 'momentum': momentum,
        'wyckoff': wyckoff_phase, 'vol': vol_regime,
        'bias': bias, 'avg_ret_24h': avg_ret, 'win_rate': win_rate,
        'sample_size': n, 'confidence': confidence, 'source': source,
        'score_adj': score_adj, 'size_mult': size_mult,
    }
    return status, score_adj, size_mult, details


def format_m44(details):
    if not details or details.get('regime') in ('DISABLED', 'NOT_RELEASE_DAY', 'NO_EDGE'):
        return None

    bias = details.get('bias', '?')
    headline = details.get('headline_mom', 0)
    core = details.get('core_mom', 0)
    cons = details.get('consensus', 0)
    signal = details.get('signal', '?')
    capex = details.get('capex_health', '?')
    momentum = details.get('momentum', '?')
    avg_ret = details.get('avg_ret_24h', 0)
    win_rate = details.get('win_rate', 0)
    n = details.get('sample_size', 0)
    conf = details.get('confidence', 0)
    score_adj = details.get('score_adj', 0)
    size_mult = details.get('size_mult', 1.0)

    bias_icon = '🟢' if bias == 'LONG' else '🔴' if bias == 'SHORT' else '⚪'
    capex_icon = {'BOOMING': '🟢🟢', 'GROWING': '🟢', 'STABLE': '⚪',
                  'WEAK': '🟠', 'COLLAPSING': '🔴'}.get(capex, '⚪')
    mom_icon = {'ACCELERATING': '📈', 'DECELERATING': '📉', 'STABLE': '➡️'}.get(momentum, '➡️')

    lines = [
        f"  M44 Durables: {bias_icon} {bias:>8}  "
        f"headline={headline:+.1f}% cons={cons:+.1f}% core={core:+.1f}%  "
        f"{capex_icon}{capex} {mom_icon}{momentum}  signal={signal}",
        f"    Backtest: 24h={avg_ret:+.2f}% win={win_rate*100:.0f}% n={n}  "
        f"adj={score_adj:+.3f} size={size_mult:.2f}x  "
        f"chain: Asia 88-100%, breaks at Lon→NY",
    ]
    return '\n'.join(lines)

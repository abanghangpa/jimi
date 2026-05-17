"""
M54: China Quarterly GDP Session Bias (Regime-Conditional)

On NBS China GDP release days (~4x/year, 02:00 UTC = 10:00 MYT),
applies a 24h directional bias based on the combination of:
  - Wyckoff phase (M21)
  - Volatility regime (M9)
  - GDP signal: STRONG_BEAT / BEAT / INLINE / MISS / BIG_MISS
  - GDP level: STRONG (≥6%) / MODERATE (5-6%) / SLOWING (4-5%) / WEAK (2-4%) / CONTRACTION (<0%)

Backtested on 34 China Quarterly GDP releases (2018-2026) against ETH/USDT 15m data.

Key findings (24h return):
  24h aggregate: +0.741% avg, 52.9% win, n=34 — NOT significant (p=0.15)
  Positive bias overall but regime-dependent. Small sample (quarterly).

  BEAT overall:  +1.313% avg, 66.7% win, n=9  → LONG
  MISS overall:  +1.370% avg, 75% win, n=4    → LONG (stimulus expectations)
  STRONG_BEAT:   -0.882% avg, 33% win, n=3    → SHORT (sell the news)

  Transmission: NY Close→Pre-Asia D2 67.6% ✅. Tokyo→Asia 64.7% ⚠️.
  Asia Mid session is weak (-0.665% avg, 35% win) — avoid Asia entries.

Thesis (user #38):
  China GDP = anchor for global demand. Miss → PBoC stimulus expectations
  → V-bottom liquidity play. Weak GDP → EU manufacturing drag + commodity
  sell-off → risk-off in EU morning. Same day: Retail Sales + Industrial
  Output (loopback: quality of growth).

Usage:
    from src.modules.m54_china_gdp import score_m54_china_gdp, format_m54
"""

from datetime import datetime, timedelta
import json
import os

CHINA_GDP_RELEASES = {
    '2018-01-18': {'gdp_yoy': 6.8, 'consensus_yoy': 6.7, 'prev_yoy': 6.8, 'retail_yoy': 9.4, 'industrial_yoy': 6.2, 'quarter': 'Q4'},
    '2018-04-17': {'gdp_yoy': 6.8, 'consensus_yoy': 6.7, 'prev_yoy': 6.8, 'retail_yoy': 10.1, 'industrial_yoy': 6.0, 'quarter': 'Q1'},
    '2018-07-16': {'gdp_yoy': 6.7, 'consensus_yoy': 6.7, 'prev_yoy': 6.8, 'retail_yoy': 9.0, 'industrial_yoy': 6.0, 'quarter': 'Q2'},
    '2018-10-19': {'gdp_yoy': 6.5, 'consensus_yoy': 6.6, 'prev_yoy': 6.7, 'retail_yoy': 9.2, 'industrial_yoy': 5.8, 'quarter': 'Q3'},
    '2019-01-21': {'gdp_yoy': 6.4, 'consensus_yoy': 6.4, 'prev_yoy': 6.5, 'retail_yoy': 8.2, 'industrial_yoy': 5.7, 'quarter': 'Q4'},
    '2019-04-17': {'gdp_yoy': 6.4, 'consensus_yoy': 6.3, 'prev_yoy': 6.4, 'retail_yoy': 8.7, 'industrial_yoy': 8.5, 'quarter': 'Q1'},
    '2019-07-15': {'gdp_yoy': 6.2, 'consensus_yoy': 6.2, 'prev_yoy': 6.4, 'retail_yoy': 9.8, 'industrial_yoy': 6.3, 'quarter': 'Q2'},
    '2019-10-18': {'gdp_yoy': 6.0, 'consensus_yoy': 6.1, 'prev_yoy': 6.2, 'retail_yoy': 7.8, 'industrial_yoy': 5.8, 'quarter': 'Q3'},
    '2020-01-17': {'gdp_yoy': 6.0, 'consensus_yoy': 6.0, 'prev_yoy': 6.0, 'retail_yoy': 8.0, 'industrial_yoy': 6.9, 'quarter': 'Q4'},
    '2020-04-17': {'gdp_yoy': -6.8, 'consensus_yoy': -6.5, 'prev_yoy': 6.0, 'retail_yoy': -15.8, 'industrial_yoy': -1.1, 'quarter': 'Q1'},
    '2020-07-16': {'gdp_yoy': 3.2, 'consensus_yoy': 2.5, 'prev_yoy': -6.8, 'retail_yoy': -1.1, 'industrial_yoy': 4.8, 'quarter': 'Q2'},
    '2020-10-19': {'gdp_yoy': 4.9, 'consensus_yoy': 5.5, 'prev_yoy': 3.2, 'retail_yoy': 3.3, 'industrial_yoy': 6.9, 'quarter': 'Q3'},
    '2021-01-18': {'gdp_yoy': 6.5, 'consensus_yoy': 6.5, 'prev_yoy': 4.9, 'retail_yoy': 4.6, 'industrial_yoy': 7.3, 'quarter': 'Q4'},
    '2021-04-16': {'gdp_yoy': 18.3, 'consensus_yoy': 18.5, 'prev_yoy': 6.5, 'retail_yoy': 34.2, 'industrial_yoy': 14.1, 'quarter': 'Q1'},
    '2021-07-15': {'gdp_yoy': 7.9, 'consensus_yoy': 8.0, 'prev_yoy': 18.3, 'retail_yoy': 12.1, 'industrial_yoy': 8.3, 'quarter': 'Q2'},
    '2021-10-18': {'gdp_yoy': 4.9, 'consensus_yoy': 5.0, 'prev_yoy': 7.9, 'retail_yoy': 4.4, 'industrial_yoy': 3.1, 'quarter': 'Q3'},
    '2022-01-17': {'gdp_yoy': 4.0, 'consensus_yoy': 4.0, 'prev_yoy': 4.9, 'retail_yoy': 1.7, 'industrial_yoy': 4.3, 'quarter': 'Q4'},
    '2022-04-18': {'gdp_yoy': 4.8, 'consensus_yoy': 4.4, 'prev_yoy': 4.0, 'retail_yoy': -3.5, 'industrial_yoy': 5.0, 'quarter': 'Q1'},
    '2022-07-15': {'gdp_yoy': 0.4, 'consensus_yoy': 1.0, 'prev_yoy': 4.8, 'retail_yoy': 3.1, 'industrial_yoy': 3.9, 'quarter': 'Q2'},
    '2022-10-24': {'gdp_yoy': 3.9, 'consensus_yoy': 3.4, 'prev_yoy': 0.4, 'retail_yoy': 2.5, 'industrial_yoy': 6.3, 'quarter': 'Q3'},
    '2023-01-17': {'gdp_yoy': 3.0, 'consensus_yoy': 2.8, 'prev_yoy': 3.9, 'retail_yoy': -1.8, 'industrial_yoy': 1.3, 'quarter': 'Q4'},
    '2023-04-18': {'gdp_yoy': 4.5, 'consensus_yoy': 4.0, 'prev_yoy': 3.0, 'retail_yoy': 10.6, 'industrial_yoy': 3.9, 'quarter': 'Q1'},
    '2023-07-17': {'gdp_yoy': 6.3, 'consensus_yoy': 7.1, 'prev_yoy': 4.5, 'retail_yoy': 3.1, 'industrial_yoy': 4.4, 'quarter': 'Q2'},
    '2023-10-18': {'gdp_yoy': 4.9, 'consensus_yoy': 4.5, 'prev_yoy': 6.3, 'retail_yoy': 5.5, 'industrial_yoy': 4.5, 'quarter': 'Q3'},
    '2024-01-17': {'gdp_yoy': 5.2, 'consensus_yoy': 5.3, 'prev_yoy': 4.9, 'retail_yoy': 7.4, 'industrial_yoy': 6.8, 'quarter': 'Q4'},
    '2024-04-16': {'gdp_yoy': 5.3, 'consensus_yoy': 5.0, 'prev_yoy': 5.2, 'retail_yoy': 3.1, 'industrial_yoy': 4.5, 'quarter': 'Q1'},
    '2024-07-15': {'gdp_yoy': 4.7, 'consensus_yoy': 5.1, 'prev_yoy': 5.3, 'retail_yoy': 2.0, 'industrial_yoy': 5.3, 'quarter': 'Q2'},
    '2024-10-18': {'gdp_yoy': 4.6, 'consensus_yoy': 4.5, 'prev_yoy': 4.7, 'retail_yoy': 3.2, 'industrial_yoy': 5.4, 'quarter': 'Q3'},
    '2025-01-17': {'gdp_yoy': 5.4, 'consensus_yoy': 5.0, 'prev_yoy': 4.6, 'retail_yoy': 3.7, 'industrial_yoy': 6.2, 'quarter': 'Q4'},
    '2025-04-16': {'gdp_yoy': 5.4, 'consensus_yoy': 5.2, 'prev_yoy': 5.4, 'retail_yoy': 5.9, 'industrial_yoy': 7.7, 'quarter': 'Q1'},
    '2025-07-15': {'gdp_yoy': 5.2, 'consensus_yoy': 5.1, 'prev_yoy': 5.4, 'retail_yoy': 4.8, 'industrial_yoy': 6.8, 'quarter': 'Q2'},
    '2025-10-20': {'gdp_yoy': 4.8, 'consensus_yoy': 4.9, 'prev_yoy': 5.2, 'retail_yoy': 3.0, 'industrial_yoy': 5.2, 'quarter': 'Q3'},
    '2026-01-19': {'gdp_yoy': 5.0, 'consensus_yoy': 4.9, 'prev_yoy': 4.8, 'retail_yoy': 3.5, 'industrial_yoy': 5.8, 'quarter': 'Q4'},
    '2026-04-16': {'gdp_yoy': 5.4, 'consensus_yoy': 5.1, 'prev_yoy': 5.0, 'retail_yoy': 5.5, 'industrial_yoy': 7.2, 'quarter': 'Q1'},
}

SIGNAL_EDGES = {
    'BEAT':        {'dir': 'LONG',  'avg': 1.313, 'wr': 0.667, 'n': 9, 'ics_adj': 0.05, 'size_mult': 1.00},
    'MISS':        {'dir': 'LONG',  'avg': 1.370, 'wr': 0.750, 'n': 4, 'ics_adj': 0.05, 'size_mult': 1.00},
    'STRONG_BEAT': {'dir': 'SHORT', 'avg': -0.882, 'wr': 0.333, 'n': 3, 'ics_adj': 0.05, 'size_mult': 1.00},
}

_FRESH_DATA = {}


def update_fresh_data(gdp_yoy, consensus_yoy, prev_yoy, retail_yoy=None, industrial_yoy=None):
    _FRESH_DATA['gdp_yoy'] = gdp_yoy
    _FRESH_DATA['consensus_yoy'] = consensus_yoy
    _FRESH_DATA['prev_yoy'] = prev_yoy
    if retail_yoy is not None: _FRESH_DATA['retail_yoy'] = retail_yoy
    if industrial_yoy is not None: _FRESH_DATA['industrial_yoy'] = industrial_yoy


def _classify_signal(gdp, consensus):
    if gdp is None or consensus is None: return 'NO_DATA'
    s = gdp - consensus
    if s >= 0.5: return 'STRONG_BEAT'
    elif s >= 0.1: return 'BEAT'
    elif s >= -0.1: return 'INLINE'
    elif s >= -0.5: return 'MISS'
    return 'BIG_MISS'


def _classify_level(gdp):
    if gdp is None: return 'NO_DATA'
    if gdp >= 6.0: return 'STRONG'
    elif gdp >= 5.0: return 'MODERATE'
    elif gdp >= 4.0: return 'SLOWING'
    elif gdp >= 2.0: return 'WEAK'
    elif gdp >= 0.0: return 'STAGNANT'
    return 'CONTRACTION'


def _get_release_data(date_str=None):
    if date_str is None: date_str = datetime.utcnow().strftime('%Y-%m-%d')
    if _FRESH_DATA.get('gdp_yoy') is not None: return _FRESH_DATA
    return CHINA_GDP_RELEASES.get(date_str)


def score_m54_china_gdp(wyckoff_phase='UNKNOWN', vol_regime='UNKNOWN',
                         direction='LONG', date_str=None):
    release_data = _get_release_data(date_str)
    if release_data is None: return 'NOT_RELEASE_DAY', 0.0, 1.0, {}
    gdp = release_data.get('gdp_yoy')
    consensus = release_data.get('consensus_yoy')
    prev = release_data.get('prev_yoy')
    if gdp is None: return 'NOT_RELEASE_DAY', 0.0, 1.0, {}

    signal = _classify_signal(gdp, consensus)
    level = _classify_level(gdp)
    edge = SIGNAL_EDGES.get(signal)
    if edge is None:
        return 'NO_EDGE', 0.0, 1.0, {
            'signal': signal, 'level': level, 'gdp_yoy': gdp,
            'consensus_yoy': consensus, 'prev_yoy': prev,
            'retail_yoy': release_data.get('retail_yoy'),
            'industrial_yoy': release_data.get('industrial_yoy'),
            'quarter': release_data.get('quarter'),
        }

    if edge['dir'] != direction:
        score_adj, size_mult = -abs(edge['ics_adj']) * 0.5, 0.85
    else:
        score_adj, size_mult = edge['ics_adj'], edge['size_mult']

    details = {
        'signal': signal, 'level': level, 'gdp_yoy': gdp,
        'consensus_yoy': consensus, 'prev_yoy': prev,
        'surprise': gdp - consensus if consensus else 0,
        'retail_yoy': release_data.get('retail_yoy'),
        'industrial_yoy': release_data.get('industrial_yoy'),
        'quarter': release_data.get('quarter'),
        'edge_dir': edge['dir'], 'edge_avg': edge['avg'],
        'edge_wr': edge['wr'], 'edge_n': edge['n'],
    }
    return 'ACTIVE', score_adj, size_mult, details


def format_m54(status, score_adj, size_mult, details):
    if status == 'NOT_RELEASE_DAY': return "M54: — (not CN GDP release day)"
    if status == 'NO_EDGE':
        return f"M54: {details.get('signal','?')} ({details.get('level','?')}, GDP {details.get('gdp_yoy','?')}%) — no edge"
    sig = details.get('signal', '?')
    lvl = details.get('level', '?')
    gdp = details.get('gdp_yoy', '?')
    con = details.get('consensus_yoy', '?')
    sup = details.get('surprise', 0)
    edge_dir = details.get('edge_dir', '?')
    edge_avg = details.get('edge_avg', 0)
    wr = details.get('edge_wr', 0)
    return (f"M54: {sig} ({lvl}) | GDP {gdp}% vs {con}% ({sup:+.1f}%) | "
            f"{edge_dir} {edge_avg:+.2f}% ({wr:.0%} WR) | "
            f"ICS {score_adj:+.03f} ×{size_mult:.2f}")


if __name__ == '__main__':
    print("=== M54 China GDP Self-Test ===\n")
    for wyck, vol, dire, date in [
        ('MARKUP', 'NEUTRAL', 'LONG', '2023-04-18'),   # BEAT (4.5 vs 4.0)
        ('MARKDOWN', 'NEUTRAL', 'LONG', '2022-07-15'),  # BIG_MISS (0.4 vs 1.0)
        ('ACCUMULATION', 'COMPRESSING', 'SHORT', '2021-04-16'),  # STRONG_BEAT (18.3 vs 18.5)
    ]:
        status, adj, mult, det = score_m54_china_gdp(wyck, vol, dire, date)
        print(format_m54(status, adj, mult, det))
        print(f"  → {status}, ICS={adj:+.03f}, size={mult:.2f}\n")
    print(format_m54(*score_m54_china_gdp('RANGE', 'NEUTRAL', 'LONG', '2026-01-15')))

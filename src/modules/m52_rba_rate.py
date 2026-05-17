"""
M52: RBA Rate Decision Session Bias (Regime-Conditional)

On Reserve Bank of Australia rate decision days (~11x/year, 03:30 UTC = 11:30 MYT),
applies a 24h directional bias based on the combination of:
  - Wyckoff phase (M21)
  - Volatility regime (M9)
  - RBA signal: HIKE / CUT / NEUTRAL_HOLD / HAWKISH_HOLD / DOVISH_HOLD
  - Rate level: RESTRICTIVE (≥4%) / NEUTRAL (2.5-4%) / ACCOMMODATIVE (1-2.5%) / EMERGENCY (<1%)
  - Rate cycle: TIGHTENING / EASING / PAUSE

Backtested on 86 RBA rate decisions (2018-2026) against ETH/USDT 15m data.

Key findings (24h return):
  24h aggregate: +0.899% avg, 53.5% win, n=86 — NOT significant (p=0.11)
  Positive bias overall but regime-dependent.

  CUT overall:                             +1.520% avg, 54.5% win, n=11 → LONG
  NEUTRAL_HOLD overall:                    +0.963% avg, 53.2% win, n=62 → LONG
  ACCUMULATION + COMPRESSING + NEUTRAL_HOLD: +2.085% avg, 67% win, n=6 → LONG
  MARKDOWN + CRISIS + NEUTRAL_HOLD:        +3.252% avg, 100% win, n=3 → LONG
  MARKUP + TREND + NEUTRAL_HOLD:           +2.882% avg, 67% win, n=3 → LONG
  DISTRIBUTION + NEUTRAL + NEUTRAL_HOLD:   +0.589% avg, 75% win, n=4 → LONG
  MARKDOWN + TREND + NEUTRAL_HOLD:         -3.022% avg, 0% win, n=3 → SHORT

  Transmission chain: Sydney→Tokyo 57.6% (marginal), Tokyo→Asia 57.0% (marginal).
  Most chain breaks. Edge is release-driven + regime context.

Thesis (user #36):
  Australia = liquid proxy for global risk & Chinese demand.
  Hawkish hold / surprise hike → AUD/USD up → institutional ETH bid density.
  Strong AUD → resilient global risk appetite → constructive for alt assets.
  Loopback: RBA calibrated against quarterly Australia CPI.

Usage:
    from src.modules.m52_rba_rate import score_m52_rba_rate, format_m52
    status, score_adj, size_mult, details = score_m52_rba_rate(
        wyckoff_phase='ACCUMULATION', vol_regime='COMPRESSING', direction='LONG')
"""

from datetime import datetime, timedelta
import json
import os

# ═══════════════════════════════════════════════════════════════
# RBA RATE DECISION RELEASE DATES (03:30 UTC = 11:30 MYT)
# Format: {date: {'rate': float, 'prev_rate': float, 'signal': str}}
# ═══════════════════════════════════════════════════════════════

RBA_RELEASES = {
    # 2018
    '2018-02-06': {'rate': 1.50, 'prev_rate': 1.50, 'signal': 'HOLD'},
    '2018-03-06': {'rate': 1.50, 'prev_rate': 1.50, 'signal': 'HOLD'},
    '2018-04-03': {'rate': 1.50, 'prev_rate': 1.50, 'signal': 'HOLD'},
    '2018-05-01': {'rate': 1.50, 'prev_rate': 1.50, 'signal': 'HOLD'},
    '2018-06-05': {'rate': 1.50, 'prev_rate': 1.50, 'signal': 'HOLD'},
    '2018-07-03': {'rate': 1.50, 'prev_rate': 1.50, 'signal': 'HOLD'},
    '2018-08-07': {'rate': 1.50, 'prev_rate': 1.50, 'signal': 'HOLD'},
    '2018-09-04': {'rate': 1.50, 'prev_rate': 1.50, 'signal': 'HOLD'},
    '2018-10-02': {'rate': 1.50, 'prev_rate': 1.50, 'signal': 'HOLD'},
    '2018-11-06': {'rate': 1.50, 'prev_rate': 1.50, 'signal': 'HOLD'},
    '2018-12-04': {'rate': 1.50, 'prev_rate': 1.50, 'signal': 'HOLD'},
    # 2019
    '2019-02-05': {'rate': 1.50, 'prev_rate': 1.50, 'signal': 'HOLD'},
    '2019-03-05': {'rate': 1.50, 'prev_rate': 1.50, 'signal': 'HOLD'},
    '2019-04-02': {'rate': 1.50, 'prev_rate': 1.50, 'signal': 'HOLD'},
    '2019-05-07': {'rate': 1.50, 'prev_rate': 1.50, 'signal': 'HOLD'},
    '2019-06-04': {'rate': 1.25, 'prev_rate': 1.50, 'signal': 'CUT'},
    '2019-07-02': {'rate': 1.00, 'prev_rate': 1.25, 'signal': 'CUT'},
    '2019-08-06': {'rate': 1.00, 'prev_rate': 1.00, 'signal': 'HOLD'},
    '2019-09-03': {'rate': 1.00, 'prev_rate': 1.00, 'signal': 'HOLD'},
    '2019-10-01': {'rate': 0.75, 'prev_rate': 1.00, 'signal': 'CUT'},
    '2019-11-05': {'rate': 0.75, 'prev_rate': 0.75, 'signal': 'HOLD'},
    '2019-12-03': {'rate': 0.75, 'prev_rate': 0.75, 'signal': 'HOLD'},
    # 2020
    '2020-02-04': {'rate': 0.75, 'prev_rate': 0.75, 'signal': 'HOLD'},
    '2020-03-03': {'rate': 0.50, 'prev_rate': 0.75, 'signal': 'CUT'},
    '2020-03-19': {'rate': 0.25, 'prev_rate': 0.50, 'signal': 'CUT'},
    '2020-04-07': {'rate': 0.25, 'prev_rate': 0.25, 'signal': 'HOLD'},
    '2020-05-05': {'rate': 0.25, 'prev_rate': 0.25, 'signal': 'HOLD'},
    '2020-06-02': {'rate': 0.25, 'prev_rate': 0.25, 'signal': 'HOLD'},
    '2020-07-07': {'rate': 0.25, 'prev_rate': 0.25, 'signal': 'HOLD'},
    '2020-08-04': {'rate': 0.25, 'prev_rate': 0.25, 'signal': 'HOLD'},
    '2020-09-01': {'rate': 0.25, 'prev_rate': 0.25, 'signal': 'HOLD'},
    '2020-10-06': {'rate': 0.25, 'prev_rate': 0.25, 'signal': 'HOLD'},
    '2020-11-03': {'rate': 0.10, 'prev_rate': 0.25, 'signal': 'CUT'},
    '2020-12-01': {'rate': 0.10, 'prev_rate': 0.10, 'signal': 'HOLD'},
    # 2021
    '2021-02-02': {'rate': 0.10, 'prev_rate': 0.10, 'signal': 'HOLD'},
    '2021-03-02': {'rate': 0.10, 'prev_rate': 0.10, 'signal': 'HOLD'},
    '2021-04-06': {'rate': 0.10, 'prev_rate': 0.10, 'signal': 'HOLD'},
    '2021-05-04': {'rate': 0.10, 'prev_rate': 0.10, 'signal': 'HOLD'},
    '2021-06-01': {'rate': 0.10, 'prev_rate': 0.10, 'signal': 'HOLD'},
    '2021-07-06': {'rate': 0.10, 'prev_rate': 0.10, 'signal': 'HOLD'},
    '2021-08-03': {'rate': 0.10, 'prev_rate': 0.10, 'signal': 'HOLD'},
    '2021-09-07': {'rate': 0.10, 'prev_rate': 0.10, 'signal': 'HOLD'},
    '2021-10-05': {'rate': 0.10, 'prev_rate': 0.10, 'signal': 'HOLD'},
    '2021-11-02': {'rate': 0.10, 'prev_rate': 0.10, 'signal': 'HOLD'},
    '2021-12-07': {'rate': 0.10, 'prev_rate': 0.10, 'signal': 'HOLD'},
    # 2022
    '2022-02-01': {'rate': 0.10, 'prev_rate': 0.10, 'signal': 'HOLD'},
    '2022-03-01': {'rate': 0.10, 'prev_rate': 0.10, 'signal': 'HOLD'},
    '2022-04-05': {'rate': 0.10, 'prev_rate': 0.10, 'signal': 'HOLD'},
    '2022-05-03': {'rate': 0.35, 'prev_rate': 0.10, 'signal': 'HIKE'},
    '2022-06-07': {'rate': 0.85, 'prev_rate': 0.35, 'signal': 'HIKE'},
    '2022-07-05': {'rate': 1.35, 'prev_rate': 0.85, 'signal': 'HIKE'},
    '2022-08-02': {'rate': 1.85, 'prev_rate': 1.35, 'signal': 'HIKE'},
    '2022-09-06': {'rate': 2.35, 'prev_rate': 1.85, 'signal': 'HIKE'},
    '2022-10-04': {'rate': 2.60, 'prev_rate': 2.35, 'signal': 'HIKE'},
    '2022-11-01': {'rate': 2.85, 'prev_rate': 2.60, 'signal': 'HIKE'},
    '2022-12-06': {'rate': 3.10, 'prev_rate': 2.85, 'signal': 'HIKE'},
    # 2023
    '2023-02-07': {'rate': 3.35, 'prev_rate': 3.10, 'signal': 'HIKE'},
    '2023-03-07': {'rate': 3.60, 'prev_rate': 3.35, 'signal': 'HIKE'},
    '2023-04-04': {'rate': 3.60, 'prev_rate': 3.60, 'signal': 'HOLD'},
    '2023-05-02': {'rate': 3.85, 'prev_rate': 3.60, 'signal': 'HIKE'},
    '2023-06-06': {'rate': 4.10, 'prev_rate': 3.85, 'signal': 'HIKE'},
    '2023-07-04': {'rate': 4.10, 'prev_rate': 4.10, 'signal': 'HOLD'},
    '2023-08-01': {'rate': 4.10, 'prev_rate': 4.10, 'signal': 'HOLD'},
    '2023-09-05': {'rate': 4.10, 'prev_rate': 4.10, 'signal': 'HOLD'},
    '2023-10-03': {'rate': 4.10, 'prev_rate': 4.10, 'signal': 'HOLD'},
    '2023-11-07': {'rate': 4.35, 'prev_rate': 4.10, 'signal': 'HIKE'},
    '2023-12-05': {'rate': 4.35, 'prev_rate': 4.35, 'signal': 'HOLD'},
    # 2024
    '2024-02-06': {'rate': 4.35, 'prev_rate': 4.35, 'signal': 'HOLD'},
    '2024-03-19': {'rate': 4.35, 'prev_rate': 4.35, 'signal': 'HOLD'},
    '2024-05-07': {'rate': 4.35, 'prev_rate': 4.35, 'signal': 'HOLD'},
    '2024-06-18': {'rate': 4.35, 'prev_rate': 4.35, 'signal': 'HOLD'},
    '2024-08-06': {'rate': 4.35, 'prev_rate': 4.35, 'signal': 'HOLD'},
    '2024-09-24': {'rate': 4.35, 'prev_rate': 4.35, 'signal': 'HOLD'},
    '2024-11-05': {'rate': 4.35, 'prev_rate': 4.35, 'signal': 'HOLD'},
    '2024-12-10': {'rate': 4.35, 'prev_rate': 4.35, 'signal': 'HOLD'},
    # 2025
    '2025-02-18': {'rate': 4.35, 'prev_rate': 4.35, 'signal': 'HOLD'},
    '2025-04-01': {'rate': 4.10, 'prev_rate': 4.35, 'signal': 'CUT'},
    '2025-05-20': {'rate': 3.85, 'prev_rate': 4.10, 'signal': 'CUT'},
    '2025-07-08': {'rate': 3.85, 'prev_rate': 3.85, 'signal': 'HOLD'},
    '2025-08-12': {'rate': 3.60, 'prev_rate': 3.85, 'signal': 'CUT'},
    '2025-09-30': {'rate': 3.60, 'prev_rate': 3.60, 'signal': 'HOLD'},
    '2025-11-04': {'rate': 3.35, 'prev_rate': 3.60, 'signal': 'CUT'},
    '2025-12-09': {'rate': 3.35, 'prev_rate': 3.35, 'signal': 'HOLD'},
    # 2026
    '2026-02-17': {'rate': 3.35, 'prev_rate': 3.35, 'signal': 'HOLD'},
    '2026-03-31': {'rate': 3.10, 'prev_rate': 3.35, 'signal': 'CUT'},
    '2026-05-05': {'rate': 3.10, 'prev_rate': 3.10, 'signal': 'HOLD'},
}

# ═══════════════════════════════════════════════════════════════
# EDGE TABLE: (wyckoff, vol, signal) → (direction, avg_return, win_rate, n)
# ═══════════════════════════════════════════════════════════════

EDGE_TABLE = {
    ('ACCUMULATION', 'COMPRESSING', 'NEUTRAL_HOLD'): {'dir': 'LONG', 'avg': 2.085, 'wr': 0.67, 'n': 6, 'ics_adj': 0.06, 'size_mult': 1.05},
    ('MARKDOWN', 'CRISIS', 'NEUTRAL_HOLD'):          {'dir': 'LONG', 'avg': 3.252, 'wr': 1.00, 'n': 3, 'ics_adj': 0.06, 'size_mult': 1.05},
    ('MARKUP', 'TREND', 'NEUTRAL_HOLD'):             {'dir': 'LONG', 'avg': 2.882, 'wr': 0.67, 'n': 3, 'ics_adj': 0.06, 'size_mult': 1.05},
    ('MARKUP', 'NEUTRAL', 'NEUTRAL_HOLD'):           {'dir': 'LONG', 'avg': 3.270, 'wr': 0.43, 'n': 7, 'ics_adj': 0.05, 'size_mult': 1.00},
    ('DISTRIBUTION', 'NEUTRAL', 'NEUTRAL_HOLD'):     {'dir': 'LONG', 'avg': 0.589, 'wr': 0.75, 'n': 4, 'ics_adj': 0.05, 'size_mult': 1.00},
    ('MARKDOWN', 'NEUTRAL', 'NEUTRAL_HOLD'):         {'dir': 'LONG', 'avg': 0.560, 'wr': 0.50, 'n': 10, 'ics_adj': 0.05, 'size_mult': 1.00},
    ('MARKDOWN', 'TREND', 'NEUTRAL_HOLD'):           {'dir': 'SHORT', 'avg': -3.022, 'wr': 0.00, 'n': 3, 'ics_adj': 0.06, 'size_mult': 1.05},
}

# Fallback: CUT cycle is bullish overall
CUT_FALLBACK = {'dir': 'LONG', 'avg': 1.520, 'wr': 0.545, 'n': 11, 'ics_adj': 0.05, 'size_mult': 1.00}

# Fallback: NEUTRAL_HOLD has mild positive bias
HOLD_FALLBACK = {'dir': 'LONG', 'avg': 0.963, 'wr': 0.532, 'n': 62, 'ics_adj': 0.05, 'size_mult': 1.00}

# ═══════════════════════════════════════════════════════════════
# FRESH DATA CACHE
# ═══════════════════════════════════════════════════════════════
_FRESH_DATA = {}


def update_fresh_data(rate, prev_rate, signal):
    """Feed fresh RBA data into cache (called by macro_fetch)."""
    _FRESH_DATA['rate'] = rate
    _FRESH_DATA['prev_rate'] = prev_rate
    _FRESH_DATA['signal'] = signal


def _classify_signal(signal, rate, prev_rate):
    if rate is None:
        return 'NO_DATA'
    if signal == 'HIKE':
        return 'HIKE'
    elif signal == 'CUT':
        return 'CUT'
    elif signal == 'HAWKISH_HOLD':
        return 'HAWKISH_HOLD'
    elif signal == 'DOVISH_HOLD':
        return 'DOVISH_HOLD'
    return 'NEUTRAL_HOLD'


def _classify_level(rate):
    if rate is None:
        return 'NO_DATA'
    if rate >= 4.0:
        return 'RESTRICTIVE'
    elif rate >= 2.5:
        return 'NEUTRAL'
    elif rate >= 1.0:
        return 'ACCOMMODATIVE'
    return 'EMERGENCY'


def _get_release_data(date_str=None):
    if date_str is None:
        date_str = datetime.utcnow().strftime('%Y-%m-%d')
    if _FRESH_DATA.get('rate') is not None:
        return _FRESH_DATA
    return RBA_RELEASES.get(date_str)


def score_m52_rba_rate(wyckoff_phase='UNKNOWN', vol_regime='UNKNOWN',
                        direction='LONG', date_str=None):
    """
    Score M52: RBA Rate Decision session bias.

    Returns: (status, score_adj, size_mult, details)
    """
    release_data = _get_release_data(date_str)
    if release_data is None:
        return 'NOT_RELEASE_DAY', 0.0, 1.0, {}

    rate = release_data.get('rate')
    prev_rate = release_data.get('prev_rate')
    raw_signal = release_data.get('signal', 'HOLD')

    if rate is None:
        return 'NOT_RELEASE_DAY', 0.0, 1.0, {}

    signal = _classify_signal(raw_signal, rate, prev_rate)
    level = _classify_level(rate)
    cycle = 'TIGHTENING' if rate > prev_rate else ('EASING' if rate < prev_rate else 'PAUSE')

    # Lookup edge table
    key = (wyckoff_phase, vol_regime, signal)
    edge = EDGE_TABLE.get(key)

    # Fallbacks
    if edge is None and signal == 'CUT':
        edge = CUT_FALLBACK
    if edge is None and signal == 'NEUTRAL_HOLD':
        edge = HOLD_FALLBACK

    if edge is None:
        return 'NO_EDGE', 0.0, 1.0, {
            'signal': signal, 'level': level, 'cycle': cycle,
            'rate': rate, 'prev_rate': prev_rate,
        }

    # Direction alignment
    if edge['dir'] != direction:
        score_adj = -abs(edge['ics_adj']) * 0.5
        size_mult = 0.85
    else:
        score_adj = edge['ics_adj']
        size_mult = edge['size_mult']

    details = {
        'signal': signal, 'level': level, 'cycle': cycle,
        'rate': rate, 'prev_rate': prev_rate,
        'rate_change': rate - prev_rate,
        'edge_key': f"{wyckoff_phase} × {vol_regime} × {signal}",
        'edge_dir': edge['dir'], 'edge_avg': edge['avg'],
        'edge_wr': edge['wr'], 'edge_n': edge['n'],
    }
    return 'ACTIVE', score_adj, size_mult, details


def format_m52(status, score_adj, size_mult, details):
    """Format M52 output for scanner display."""
    if status == 'NOT_RELEASE_DAY':
        return "M52: — (not RBA release day)"
    if status == 'NO_EDGE':
        sig = details.get('signal', '?')
        lvl = details.get('level', '?')
        rate = details.get('rate', '?')
        return f"M52: {sig} ({lvl}, {rate}%) — no regime edge"

    sig = details.get('signal', '?')
    lvl = details.get('level', '?')
    rate = details.get('rate', '?')
    prev = details.get('prev_rate', '?')
    cyc = details.get('cycle', '?')
    edge_key = details.get('edge_key', '?')
    edge_dir = details.get('edge_dir', '?')
    edge_avg = details.get('edge_avg', 0)
    wr = details.get('edge_wr', 0)

    return (f"M52: {sig} ({lvl}, {cyc}) | {rate}% vs {prev}% | "
            f"{edge_key} → {edge_dir} {edge_avg:+.2f}% ({wr:.0%} WR) | "
            f"ICS {score_adj:+.03f} ×{size_mult:.2f}")


if __name__ == '__main__':
    print("=== M52 RBA Rate Decision Self-Test ===\n")
    test_cases = [
        ('ACCUMULATION', 'COMPRESSING', 'LONG', '2018-02-06'),  # HOLD
        ('MARKDOWN', 'CRISIS', 'LONG', '2020-04-07'),           # HOLD in crisis
        ('MARKUP', 'TREND', 'LONG', '2022-06-07'),              # HIKE
        ('MARKDOWN', 'TREND', 'SHORT', '2022-09-06'),           # HIKE
    ]
    for wyck, vol, dire, date in test_cases:
        status, adj, mult, det = score_m52_rba_rate(wyck, vol, dire, date)
        print(format_m52(status, adj, mult, det))
        print(f"  → status={status}, ICS={adj:+.03f}, size={mult:.2f}\n")

    status, adj, mult, det = score_m52_rba_rate('RANGE', 'NEUTRAL', 'LONG', '2026-01-15')
    print(format_m52(status, adj, mult, det))

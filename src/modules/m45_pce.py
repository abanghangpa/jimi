"""
M45: Core PCE + Personal Spending Session Bias (Regime-Conditional)

On BEA Core PCE Price Index release days (~final Friday of month,
12:30 UTC = 08:30 ET = 20:30 MYT), applies a session-conditional directional
bias based on the combination of:
  - Wyckoff phase (M21)
  - Volatility regime (M9)
  - PCE signal: HOT_BEAT / WARM_BEAT / INLINE / MILD_MISS / COOL_MISS
  - Inflation level: RUNAWAY / HOT / WARM / TARGET / COOL / DEFLATION
  - Spending signal: SURGING / STRONG / MODERATE / WEAK

Thesis (from user #29):
  US Friday Morning (20:30 MYT) → Weekend Crypto Trend Lock → FOMC Timeline
  Hot Core PCE → overrides soft inflation narratives → 10Y yield surges
  → liquidation algorithms sweep buy orders → ETH drops
  Cold Core PCE → Fed dovish → rate cut expectations → ETH rallies
  Dropped on final Friday → sets definitive trend for weekend crypto session
  Primary metric for FOMC rate decisions

Backtested on 91 Core PCE releases (2018-2026) against ETH/USDT 15m data.

Key findings:
  24h aggregate: +0.148% avg, 56.0% win, n=91 — NOT significant (p=0.71)
  Weekend return (Fri release → Mon open): +0.691% avg, 57.1% win
  The PCE event is NOISE at the 24h level — only specific combos have edge.

  Specific combos with edge:
    RANGE + CHOP + INLINE:              -3.921% avg, 67% win, n=3  → SHORT
    RANGE + NEUTRAL + MILD_MISS:        +1.629% avg, 75% win, n=4  → LONG
    MARKDOWN + NEUTRAL + INLINE:        +1.624% avg, 80% win, n=5  → LONG
    RANGE + NEUTRAL + INLINE:           +1.388% avg, 50% win, n=14 → LONG
    RUNAWAY + SURGING + INLINE:         +2.744% avg, 80% win, n=5  → LONG
    COOL + STRONG + INLINE:             +2.743% avg, 100% win, n=4 → LONG
    RANGE + COMPRESSING + MILD_MISS:    -1.074% avg, 14% win, n=7  → SHORT

  Transmission chain (strong pre-release drift):
    Asia→London: 92-100% persistence ✅
    Chain BREAKS at London Midday→NY Pre-Open (51.6%) ❌
    NY Overlap→NY PM: 79-89% ✅ (session recovery)

Integration: lightweight modifier on BEA release days only (~12x/year).
Returns a score adjustment and size multiplier — does NOT veto.

Usage:
    from src.modules.m45_pce import score_m45_pce, format_m45
    status, score_adj, size_mult, details = score_m45_pce(
        wyckoff_phase='RANGE', vol_regime='CHOP', direction='LONG')
"""

from datetime import datetime, timedelta
import json
import os


PCE_RELEASES = {
    '2018-01-29': {'core_pce_yoy': 1.5, 'core_pce_mom': 0.3, 'spending_mom': 0.4, 'consensus_yoy': 1.5, 'prior_yoy': 1.5},
    '2018-02-28': {'core_pce_yoy': 1.5, 'core_pce_mom': 0.2, 'spending_mom': 0.2, 'consensus_yoy': 1.5, 'prior_yoy': 1.5},
    '2018-03-29': {'core_pce_yoy': 1.6, 'core_pce_mom': 0.2, 'spending_mom': 0.2, 'consensus_yoy': 1.6, 'prior_yoy': 1.5},
    '2018-04-30': {'core_pce_yoy': 1.8, 'core_pce_mom': 0.2, 'spending_mom': 0.6, 'consensus_yoy': 1.9, 'prior_yoy': 1.6},
    '2018-05-31': {'core_pce_yoy': 1.8, 'core_pce_mom': 0.2, 'spending_mom': 0.3, 'consensus_yoy': 1.8, 'prior_yoy': 1.8},
    '2018-06-29': {'core_pce_yoy': 1.9, 'core_pce_mom': 0.1, 'spending_mom': 0.4, 'consensus_yoy': 1.9, 'prior_yoy': 1.8},
    '2018-07-31': {'core_pce_yoy': 2.0, 'core_pce_mom': 0.1, 'spending_mom': 0.4, 'consensus_yoy': 1.9, 'prior_yoy': 1.9},
    '2018-08-30': {'core_pce_yoy': 2.0, 'core_pce_mom': 0.1, 'spending_mom': 0.3, 'consensus_yoy': 2.0, 'prior_yoy': 2.0},
    '2018-09-28': {'core_pce_yoy': 2.0, 'core_pce_mom': 0.1, 'spending_mom': 0.3, 'consensus_yoy': 2.0, 'prior_yoy': 2.0},
    '2018-10-29': {'core_pce_yoy': 1.9, 'core_pce_mom': 0.1, 'spending_mom': 0.6, 'consensus_yoy': 2.0, 'prior_yoy': 2.0},
    '2018-11-29': {'core_pce_yoy': 1.9, 'core_pce_mom': 0.1, 'spending_mom': 0.3, 'consensus_yoy': 1.9, 'prior_yoy': 1.9},
    '2018-12-21': {'core_pce_yoy': 1.9, 'core_pce_mom': 0.1, 'spending_mom': 0.4, 'consensus_yoy': 1.9, 'prior_yoy': 1.9},
    '2019-01-31': {'core_pce_yoy': 1.9, 'core_pce_mom': 0.2, 'spending_mom': -0.5, 'consensus_yoy': 1.9, 'prior_yoy': 1.9},
    '2019-02-28': {'core_pce_yoy': 1.8, 'core_pce_mom': 0.1, 'spending_mom': 0.1, 'consensus_yoy': 1.9, 'prior_yoy': 1.9},
    '2019-03-29': {'core_pce_yoy': 1.7, 'core_pce_mom': 0.0, 'spending_mom': 0.1, 'consensus_yoy': 1.7, 'prior_yoy': 1.8},
    '2019-04-29': {'core_pce_yoy': 1.6, 'core_pce_mom': 0.2, 'spending_mom': 0.9, 'consensus_yoy': 1.6, 'prior_yoy': 1.7},
    '2019-05-30': {'core_pce_yoy': 1.6, 'core_pce_mom': 0.2, 'spending_mom': 0.3, 'consensus_yoy': 1.6, 'prior_yoy': 1.6},
    '2019-06-28': {'core_pce_yoy': 1.6, 'core_pce_mom': 0.2, 'spending_mom': 0.4, 'consensus_yoy': 1.6, 'prior_yoy': 1.6},
    '2019-07-30': {'core_pce_yoy': 1.6, 'core_pce_mom': 0.2, 'spending_mom': 0.6, 'consensus_yoy': 1.6, 'prior_yoy': 1.6},
    '2019-08-30': {'core_pce_yoy': 1.8, 'core_pce_mom': 0.1, 'spending_mom': 0.1, 'consensus_yoy': 1.7, 'prior_yoy': 1.6},
    '2019-09-27': {'core_pce_yoy': 1.7, 'core_pce_mom': 0.0, 'spending_mom': 0.2, 'consensus_yoy': 1.8, 'prior_yoy': 1.8},
    '2019-10-31': {'core_pce_yoy': 1.7, 'core_pce_mom': 0.1, 'spending_mom': 0.3, 'consensus_yoy': 1.7, 'prior_yoy': 1.7},
    '2019-11-27': {'core_pce_yoy': 1.6, 'core_pce_mom': 0.1, 'spending_mom': 0.5, 'consensus_yoy': 1.6, 'prior_yoy': 1.7},
    '2019-12-20': {'core_pce_yoy': 1.6, 'core_pce_mom': 0.1, 'spending_mom': 0.4, 'consensus_yoy': 1.6, 'prior_yoy': 1.6},
    '2020-01-31': {'core_pce_yoy': 1.6, 'core_pce_mom': 0.1, 'spending_mom': 0.2, 'consensus_yoy': 1.7, 'prior_yoy': 1.6},
    '2020-02-28': {'core_pce_yoy': 1.7, 'core_pce_mom': 0.1, 'spending_mom': 0.2, 'consensus_yoy': 1.7, 'prior_yoy': 1.6},
    '2020-03-27': {'core_pce_yoy': 1.7, 'core_pce_mom': 0.0, 'spending_mom': -7.5, 'consensus_yoy': 1.6, 'prior_yoy': 1.7},
    '2020-04-30': {'core_pce_yoy': 1.0, 'core_pce_mom': -0.3, 'spending_mom': -12.6, 'consensus_yoy': 1.1, 'prior_yoy': 1.7},
    '2020-05-29': {'core_pce_yoy': 1.0, 'core_pce_mom': -0.1, 'spending_mom': 8.2, 'consensus_yoy': 0.9, 'prior_yoy': 1.0},
    '2020-06-26': {'core_pce_yoy': 0.9, 'core_pce_mom': 0.1, 'spending_mom': 5.6, 'consensus_yoy': 1.0, 'prior_yoy': 1.0},
    '2020-07-31': {'core_pce_yoy': 1.0, 'core_pce_mom': 0.2, 'spending_mom': 1.9, 'consensus_yoy': 1.0, 'prior_yoy': 0.9},
    '2020-08-28': {'core_pce_yoy': 1.3, 'core_pce_mom': 0.3, 'spending_mom': 1.0, 'consensus_yoy': 1.2, 'prior_yoy': 1.0},
    '2020-09-30': {'core_pce_yoy': 1.5, 'core_pce_mom': 0.2, 'spending_mom': 1.2, 'consensus_yoy': 1.4, 'prior_yoy': 1.3},
    '2020-10-30': {'core_pce_yoy': 1.4, 'core_pce_mom': 0.0, 'spending_mom': 0.5, 'consensus_yoy': 1.4, 'prior_yoy': 1.5},
    '2020-11-25': {'core_pce_yoy': 1.4, 'core_pce_mom': 0.0, 'spending_mom': -0.4, 'consensus_yoy': 1.4, 'prior_yoy': 1.4},
    '2020-12-23': {'core_pce_yoy': 1.5, 'core_pce_mom': 0.0, 'spending_mom': -0.4, 'consensus_yoy': 1.5, 'prior_yoy': 1.4},
    '2021-01-29': {'core_pce_yoy': 1.5, 'core_pce_mom': 0.3, 'spending_mom': 2.4, 'consensus_yoy': 1.4, 'prior_yoy': 1.5},
    '2021-02-26': {'core_pce_yoy': 1.4, 'core_pce_mom': 0.1, 'spending_mom': -1.0, 'consensus_yoy': 1.5, 'prior_yoy': 1.5},
    '2021-03-26': {'core_pce_yoy': 1.8, 'core_pce_mom': 0.4, 'spending_mom': 0.2, 'consensus_yoy': 1.4, 'prior_yoy': 1.4},
    '2021-04-30': {'core_pce_yoy': 3.1, 'core_pce_mom': 0.7, 'spending_mom': 0.5, 'consensus_yoy': 2.9, 'prior_yoy': 1.8},
    '2021-05-28': {'core_pce_yoy': 3.4, 'core_pce_mom': 0.5, 'spending_mom': 0.0, 'consensus_yoy': 3.0, 'prior_yoy': 3.1},
    '2021-06-25': {'core_pce_yoy': 3.5, 'core_pce_mom': 0.4, 'spending_mom': 0.3, 'consensus_yoy': 3.4, 'prior_yoy': 3.4},
    '2021-07-30': {'core_pce_yoy': 3.6, 'core_pce_mom': 0.3, 'spending_mom': 0.3, 'consensus_yoy': 3.5, 'prior_yoy': 3.5},
    '2021-08-27': {'core_pce_yoy': 3.6, 'core_pce_mom': 0.3, 'spending_mom': 0.8, 'consensus_yoy': 3.5, 'prior_yoy': 3.6},
    '2021-10-01': {'core_pce_yoy': 3.6, 'core_pce_mom': 0.2, 'spending_mom': 0.6, 'consensus_yoy': 3.6, 'prior_yoy': 3.6},
    '2021-10-29': {'core_pce_yoy': 4.1, 'core_pce_mom': 0.4, 'spending_mom': 1.3, 'consensus_yoy': 4.1, 'prior_yoy': 3.7},
    '2021-11-24': {'core_pce_yoy': 4.1, 'core_pce_mom': 0.5, 'spending_mom': 1.4, 'consensus_yoy': 4.1, 'prior_yoy': 4.1},
    '2021-12-23': {'core_pce_yoy': 4.7, 'core_pce_mom': 0.5, 'spending_mom': 0.6, 'consensus_yoy': 4.4, 'prior_yoy': 4.1},
    '2022-01-28': {'core_pce_yoy': 4.9, 'core_pce_mom': 0.5, 'spending_mom': 1.1, 'consensus_yoy': 4.8, 'prior_yoy': 4.7},
    '2022-02-25': {'core_pce_yoy': 5.2, 'core_pce_mom': 0.4, 'spending_mom': 2.1, 'consensus_yoy': 5.1, 'prior_yoy': 4.9},
    '2022-03-31': {'core_pce_yoy': 5.2, 'core_pce_mom': 0.3, 'spending_mom': 1.1, 'consensus_yoy': 5.5, 'prior_yoy': 5.2},
    '2022-04-29': {'core_pce_yoy': 4.9, 'core_pce_mom': 0.3, 'spending_mom': 0.9, 'consensus_yoy': 5.2, 'prior_yoy': 5.2},
    '2022-05-27': {'core_pce_yoy': 4.7, 'core_pce_mom': 0.3, 'spending_mom': 0.2, 'consensus_yoy': 4.8, 'prior_yoy': 4.9},
    '2022-06-30': {'core_pce_yoy': 4.8, 'core_pce_mom': 0.6, 'spending_mom': 0.1, 'consensus_yoy': 4.7, 'prior_yoy': 4.7},
    '2022-07-29': {'core_pce_yoy': 4.6, 'core_pce_mom': 0.1, 'spending_mom': 0.1, 'consensus_yoy': 4.7, 'prior_yoy': 4.8},
    '2022-08-26': {'core_pce_yoy': 4.7, 'core_pce_mom': 0.6, 'spending_mom': 0.4, 'consensus_yoy': 4.6, 'prior_yoy': 4.6},
    '2022-09-30': {'core_pce_yoy': 5.1, 'core_pce_mom': 0.5, 'spending_mom': 0.4, 'consensus_yoy': 4.9, 'prior_yoy': 4.7},
    '2022-10-28': {'core_pce_yoy': 5.0, 'core_pce_mom': 0.3, 'spending_mom': 0.8, 'consensus_yoy': 5.0, 'prior_yoy': 5.1},
    '2022-11-30': {'core_pce_yoy': 4.7, 'core_pce_mom': 0.2, 'spending_mom': 0.1, 'consensus_yoy': 4.7, 'prior_yoy': 5.0},
    '2022-12-23': {'core_pce_yoy': 4.4, 'core_pce_mom': 0.1, 'spending_mom': -0.2, 'consensus_yoy': 4.4, 'prior_yoy': 4.7},
    '2023-01-27': {'core_pce_yoy': 4.4, 'core_pce_mom': 0.6, 'spending_mom': 1.8, 'consensus_yoy': 4.3, 'prior_yoy': 4.4},
    '2023-02-24': {'core_pce_yoy': 4.7, 'core_pce_mom': 0.6, 'spending_mom': 1.1, 'consensus_yoy': 4.3, 'prior_yoy': 4.4},
    '2023-03-31': {'core_pce_yoy': 4.6, 'core_pce_mom': 0.3, 'spending_mom': 0.0, 'consensus_yoy': 4.5, 'prior_yoy': 4.7},
    '2023-04-28': {'core_pce_yoy': 4.6, 'core_pce_mom': 0.4, 'spending_mom': 0.8, 'consensus_yoy': 4.5, 'prior_yoy': 4.6},
    '2023-05-26': {'core_pce_yoy': 4.7, 'core_pce_mom': 0.4, 'spending_mom': 0.8, 'consensus_yoy': 4.6, 'prior_yoy': 4.6},
    '2023-06-30': {'core_pce_yoy': 4.1, 'core_pce_mom': 0.2, 'spending_mom': 0.5, 'consensus_yoy': 4.2, 'prior_yoy': 4.7},
    '2023-07-28': {'core_pce_yoy': 4.1, 'core_pce_mom': 0.2, 'spending_mom': 0.5, 'consensus_yoy': 4.1, 'prior_yoy': 4.1},
    '2023-08-31': {'core_pce_yoy': 3.7, 'core_pce_mom': 0.2, 'spending_mom': 0.8, 'consensus_yoy': 3.8, 'prior_yoy': 4.1},
    '2023-09-29': {'core_pce_yoy': 3.7, 'core_pce_mom': 0.3, 'spending_mom': 0.4, 'consensus_yoy': 3.7, 'prior_yoy': 3.7},
    '2023-10-27': {'core_pce_yoy': 3.5, 'core_pce_mom': 0.3, 'spending_mom': 0.4, 'consensus_yoy': 3.5, 'prior_yoy': 3.7},
    '2023-11-30': {'core_pce_yoy': 3.2, 'core_pce_mom': 0.1, 'spending_mom': 0.2, 'consensus_yoy': 3.3, 'prior_yoy': 3.5},
    '2023-12-22': {'core_pce_yoy': 2.9, 'core_pce_mom': 0.1, 'spending_mom': 0.3, 'consensus_yoy': 3.0, 'prior_yoy': 3.2},
    '2024-01-26': {'core_pce_yoy': 2.8, 'core_pce_mom': 0.4, 'spending_mom': 0.7, 'consensus_yoy': 2.8, 'prior_yoy': 2.9},
    '2024-02-29': {'core_pce_yoy': 2.8, 'core_pce_mom': 0.3, 'spending_mom': 0.2, 'consensus_yoy': 2.8, 'prior_yoy': 2.8},
    '2024-03-29': {'core_pce_yoy': 2.8, 'core_pce_mom': 0.3, 'spending_mom': 0.8, 'consensus_yoy': 2.8, 'prior_yoy': 2.8},
    '2024-04-26': {'core_pce_yoy': 2.8, 'core_pce_mom': 0.3, 'spending_mom': 0.2, 'consensus_yoy': 2.7, 'prior_yoy': 2.8},
    '2024-05-31': {'core_pce_yoy': 2.6, 'core_pce_mom': 0.2, 'spending_mom': 0.2, 'consensus_yoy': 2.7, 'prior_yoy': 2.8},
    '2024-06-28': {'core_pce_yoy': 2.6, 'core_pce_mom': 0.2, 'spending_mom': 0.2, 'consensus_yoy': 2.6, 'prior_yoy': 2.6},
    '2024-07-26': {'core_pce_yoy': 2.5, 'core_pce_mom': 0.2, 'spending_mom': 0.3, 'consensus_yoy': 2.5, 'prior_yoy': 2.6},
    '2024-08-30': {'core_pce_yoy': 2.6, 'core_pce_mom': 0.2, 'spending_mom': 0.5, 'consensus_yoy': 2.6, 'prior_yoy': 2.5},
    '2024-09-27': {'core_pce_yoy': 2.2, 'core_pce_mom': 0.1, 'spending_mom': 0.5, 'consensus_yoy': 2.2, 'prior_yoy': 2.6},
    '2024-10-31': {'core_pce_yoy': 2.8, 'core_pce_mom': 0.3, 'spending_mom': 0.5, 'consensus_yoy': 2.6, 'prior_yoy': 2.2},
    '2024-11-27': {'core_pce_yoy': 2.3, 'core_pce_mom': 0.1, 'spending_mom': 0.4, 'consensus_yoy': 2.3, 'prior_yoy': 2.8},
    '2024-12-20': {'core_pce_yoy': 2.8, 'core_pce_mom': 0.1, 'spending_mom': 0.4, 'consensus_yoy': 2.9, 'prior_yoy': 2.3},
    '2025-01-31': {'core_pce_yoy': 2.6, 'core_pce_mom': 0.3, 'spending_mom': 0.7, 'consensus_yoy': 2.6, 'prior_yoy': 2.8},
    '2025-02-28': {'core_pce_yoy': 2.6, 'core_pce_mom': 0.3, 'spending_mom': 0.4, 'consensus_yoy': 2.6, 'prior_yoy': 2.6},
    '2025-03-28': {'core_pce_yoy': 2.8, 'core_pce_mom': 0.4, 'spending_mom': 0.5, 'consensus_yoy': 2.7, 'prior_yoy': 2.6},
    '2025-04-30': {'core_pce_yoy': 2.6, 'core_pce_mom': 0.0, 'spending_mom': 0.2, 'consensus_yoy': 2.6, 'prior_yoy': 2.8},
    '2025-05-30': {'core_pce_yoy': 2.5, 'core_pce_mom': 0.1, 'spending_mom': 0.2, 'consensus_yoy': 2.5, 'prior_yoy': 2.6},
    '2026-01-30': {'core_pce_yoy': 2.4, 'core_pce_mom': 0.2, 'spending_mom': 0.3, 'consensus_yoy': 2.4, 'prior_yoy': 2.3},
    '2026-02-27': {'core_pce_yoy': 2.3, 'core_pce_mom': 0.2, 'spending_mom': 0.3, 'consensus_yoy': 2.3, 'prior_yoy': 2.4},
}


# Edge table (n≥3, |avg|≥0.5%)
EDGE_TABLE = {
    ('RANGE', 'CHOP', 'INLINE'):                   (-3.921, 0.667, 3, 'SHORT'),
    ('RANGE', 'NEUTRAL', 'MILD_MISS'):             (1.629, 0.750, 4, 'LONG'),
    ('MARKDOWN', 'NEUTRAL', 'INLINE'):             (1.624, 0.800, 5, 'LONG'),
    ('RANGE', 'TRENDING', 'INLINE'):               (1.505, 0.333, 3, 'LONG'),
    ('RANGE', 'NEUTRAL', 'INLINE'):                (1.388, 0.500, 14, 'LONG'),
    ('RANGE', 'COMPRESSING', 'MILD_MISS'):         (-1.074, 0.143, 7, 'SHORT'),
    ('MARKUP', 'NEUTRAL', 'INLINE'):               (-0.957, 0.500, 4, 'SHORT'),
    ('RANGE', 'NEUTRAL', 'WARM_BEAT'):             (-0.826, 0.600, 5, 'SHORT'),
}

# Inflation × Spending edge
INFL_SPENDING_EDGE = {
    ('RUNAWAY', 'SURGING', 'INLINE'):              (2.744, 0.800, 5, 'LONG'),
    ('COOL', 'STRONG', 'INLINE'):                  (2.743, 1.000, 4, 'LONG'),
    ('COOL', 'SURGING', 'WARM_BEAT'):              (1.983, 0.750, 4, 'LONG'),
    ('COOL', 'MODERATE', 'MILD_MISS'):             (1.719, 0.667, 3, 'LONG'),
    ('RUNAWAY', 'STRONG', 'INLINE'):               (1.665, 0.800, 5, 'LONG'),
    ('COOL', 'WEAK', 'INLINE'):                    (-5.978, 0.333, 3, 'SHORT'),
    ('WARM', 'STRONG', 'WARM_BEAT'):               (-2.956, 0.000, 2, 'SHORT'),
}


def _classify_signal(actual_yoy, consensus_yoy, prior_yoy):
    diff = actual_yoy - consensus_yoy
    surprise = diff / abs(consensus_yoy) if consensus_yoy != 0 else diff
    if surprise > 0.15:
        return 'HOT_BEAT'
    elif surprise > 0.03:
        return 'WARM_BEAT'
    elif surprise < -0.15:
        return 'COOL_MISS'
    elif surprise < -0.03:
        return 'MILD_MISS'
    else:
        return 'INLINE'


def _classify_inflation(actual_yoy):
    if actual_yoy >= 4.0:
        return 'RUNAWAY'
    elif actual_yoy >= 3.0:
        return 'HOT'
    elif actual_yoy >= 2.5:
        return 'WARM'
    elif actual_yoy >= 2.0:
        return 'TARGET'
    elif actual_yoy >= 1.0:
        return 'COOL'
    else:
        return 'DEFLATION'


def _classify_spending(spending_mom):
    if spending_mom >= 1.0:
        return 'SURGING'
    elif spending_mom >= 0.5:
        return 'STRONG'
    elif spending_mom >= 0.0:
        return 'MODERATE'
    else:
        return 'WEAK'


def _is_release_day(today_str):
    today = datetime.strptime(today_str, '%Y-%m-%d')
    for date_str in PCE_RELEASES:
        release_dt = datetime.strptime(date_str, '%Y-%m-%d')
        delta = abs((today - release_dt).days)
        if delta <= 1:
            return date_str, PCE_RELEASES[date_str]
    return None, None


def score_m45_pce(wyckoff_phase='RANGE', vol_regime='CHOP',
                   direction='LONG', today_str=None, config=None):
    """Score M45: Core PCE + Personal Spending session bias."""
    cfg = config or {}
    if not cfg.get('M45_ENABLED', True):
        return 'DISABLED', 0.0, 1.0, {'regime': 'DISABLED'}

    if today_str is None:
        today_str = datetime.utcnow().strftime('%Y-%m-%d')

    release_date, release_data = _is_release_day(today_str)
    if release_data is None:
        return 'SKIP', 0.0, 1.0, {'regime': 'NOT_RELEASE_DAY'}

    core_yoy = release_data['core_pce_yoy']
    core_mom = release_data['core_pce_mom']
    spending = release_data['spending_mom']
    consensus = release_data['consensus_yoy']
    prior = release_data['prior_yoy']

    signal = _classify_signal(core_yoy, consensus, prior)
    infl = _classify_inflation(core_yoy)
    spend = _classify_spending(spending)

    # Primary: Wyckoff × Vol × Signal
    edge_key = (wyckoff_phase, vol_regime, signal)
    edge = EDGE_TABLE.get(edge_key)
    if edge is None:
        for (w, v, s), e in EDGE_TABLE.items():
            if w == wyckoff_phase and s == signal:
                edge = e
                edge_key = (w, v, s)
                break

    # Secondary: Inflation × Spending
    is_key = (infl, spend, signal)
    is_edge = INFL_SPENDING_EDGE.get(is_key)

    if edge and abs(edge[0]) >= 0.5 and edge[2] >= 3:
        avg_ret, win_rate, n, bias = edge
        source = f'wyckoff_vol_signal: {edge_key}'
        confidence = min(0.6, n / 10.0)
    elif is_edge and abs(is_edge[0]) >= 0.5 and is_edge[2] >= 2:
        avg_ret, win_rate, n, bias = is_edge
        source = f'inflation_spending: {is_key}'
        confidence = min(0.5, n / 10.0)
    else:
        return 'NO_EDGE', 0.0, 1.0, {
            'regime': 'NO_EDGE', 'release_date': release_date,
            'core_pce_yoy': core_yoy, 'consensus_yoy': consensus,
            'signal': signal, 'inflation': infl, 'spending': spend,
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
        'core_pce_yoy': core_yoy, 'core_pce_mom': core_mom,
        'spending_mom': spending, 'consensus_yoy': consensus,
        'prior_yoy': prior, 'signal': signal,
        'inflation': infl, 'spending_signal': spend,
        'wyckoff': wyckoff_phase, 'vol': vol_regime,
        'bias': bias, 'avg_ret_24h': avg_ret, 'win_rate': win_rate,
        'sample_size': n, 'confidence': confidence, 'source': source,
        'score_adj': score_adj, 'size_mult': size_mult,
        'fomc_metric': True,  # flag: this is the Fed's preferred gauge
    }
    return status, score_adj, size_mult, details


def format_m45(details):
    if not details or details.get('regime') in ('DISABLED', 'NOT_RELEASE_DAY', 'NO_EDGE'):
        return None

    bias = details.get('bias', '?')
    core_yoy = details.get('core_pce_yoy', 0)
    core_mom = details.get('core_pce_mom', 0)
    spending = details.get('spending_mom', 0)
    cons = details.get('consensus_yoy', 0)
    signal = details.get('signal', '?')
    infl = details.get('inflation', '?')
    spend = details.get('spending_signal', '?')
    avg_ret = details.get('avg_ret_24h', 0)
    win_rate = details.get('win_rate', 0)
    n = details.get('sample_size', 0)
    score_adj = details.get('score_adj', 0)
    size_mult = details.get('size_mult', 1.0)

    bias_icon = '🟢' if bias == 'LONG' else '🔴' if bias == 'SHORT' else '⚪'
    infl_icon = {'RUNAWAY': '🔴🔴', 'HOT': '🔴', 'WARM': '🟠',
                 'TARGET': '🟢', 'COOL': '🟢🟢', 'DEFLATION': '⚪'}.get(infl, '⚪')
    spend_icon = {'SURGING': '📈', 'STRONG': '🟢', 'MODERATE': '⚪', 'WEAK': '🔴'}.get(spend, '⚪')

    lines = [
        f"  M45 Core PCE: {bias_icon} {bias:>8}  "
        f"PCE={core_yoy:.1f}%(yoy) {core_mom:+.1f}%(mom) cons={cons:.1f}%  "
        f"{infl_icon}{infl} spend={spending:+.1f}%{spend_icon}{spend}  signal={signal}",
        f"    Backtest: 24h={avg_ret:+.2f}% win={win_rate*100:.0f}% n={n}  "
        f"adj={score_adj:+.3f} size={size_mult:.2f}x  "
        f"chain: Asia 92-100%, breaks at Lon→NY  🏛️FOMC metric",
    ]
    return '\n'.join(lines)

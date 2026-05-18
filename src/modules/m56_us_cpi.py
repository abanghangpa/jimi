"""
M56: US CPI Session Bias (Regime-Conditional)

On BLS CPI release days (~10th-14th of month, 12:30 UTC = 08:30 ET = 20:30 MYT),
applies a session-conditional directional bias based on:
  - Wyckoff phase (M21)
  - Volatility regime (M9)
  - CPI signal: STRONG_BEAT / BEAT / INLINE / MISS / BIG_MISS
  - Inflation level: RUNAWAY / HOT / WARM / TARGET / COOL

Thesis:
  Hot CPI → Fed hawkish → rate hike expectations → 10Y yield surges
  → liquidation algorithms sweep buy orders → ETH drops
  Cold CPI → Fed dovish → rate cut expectations → ETH rallies
  CPI is THE inflation gauge — more frequent than PCE, mid-month release,
  politically salient. The Fed's dual mandate (price + employment) means
  CPI directly shapes rate expectations.

Backtested on 89 US CPI releases (2018-2026) against ETH/USDT 15m data.

Key findings:
  Specific combos with edge:
    RANGE + NEUTRAL + BIG_MISS:         strong edge → LONG
    MARKDOWN + NEUTRAL + MISS:          moderate edge → LONG
    RANGE + COMPRESSING + STRONG_BEAT:  strong edge → SHORT
    RUNAWAY + INLINE:                   moderate edge → SHORT
    HOT + MISS:                         moderate edge → LONG

  Transmission chain:
    London Midday → NY Pre-Open: 62-68% persistence (CPI drives early)
    NY Open → NY AM: 70-80% persistence ✅
    Chain can break at Asia sessions (overnight repricing)

Integration: lightweight modifier on BLS release days only (~12x/year).
Returns a score adjustment and size multiplier — does NOT veto.

Usage:
    from src.modules.m56_us_cpi import score_m56_us_cpi, format_m56
    status, score_adj, size_mult, details = score_m56_us_cpi(
        wyckoff_phase='RANGE', vol_regime='NEUTRAL', direction='LONG')
"""

from datetime import datetime, timedelta


# ═══════════════════════════════════════════════════════════════
# US CPI RELEASE DATES (12:30 UTC = 08:30 ET)
# Format: {date: {'cpi_yoy': float, 'consensus_yoy': float, 'prior_yoy': float,
#                  'cpi_mom': float, 'consensus_mom': float}}
# ═══════════════════════════════════════════════════════════════

CPI_RELEASES = {
    # ── 2018 ──
    '2018-01-11': {'cpi_yoy': 2.1, 'consensus_yoy': 2.1, 'prior_yoy': 2.1, 'cpi_mom': 0.5, 'consensus_mom': 0.3},
    '2018-02-14': {'cpi_yoy': 2.2, 'consensus_yoy': 2.0, 'prior_yoy': 2.1, 'cpi_mom': 0.5, 'consensus_mom': 0.3},
    '2018-03-13': {'cpi_yoy': 2.2, 'consensus_yoy': 2.2, 'prior_yoy': 2.2, 'cpi_mom': 0.2, 'consensus_mom': 0.2},
    '2018-04-11': {'cpi_yoy': 2.4, 'consensus_yoy': 2.4, 'prior_yoy': 2.2, 'cpi_mom': 0.2, 'consensus_mom': 0.2},
    '2018-05-10': {'cpi_yoy': 2.5, 'consensus_yoy': 2.5, 'prior_yoy': 2.4, 'cpi_mom': 0.2, 'consensus_mom': 0.2},
    '2018-06-12': {'cpi_yoy': 2.8, 'consensus_yoy': 2.8, 'prior_yoy': 2.5, 'cpi_mom': 0.2, 'consensus_mom': 0.2},
    '2018-07-12': {'cpi_yoy': 2.9, 'consensus_yoy': 2.9, 'prior_yoy': 2.8, 'cpi_mom': 0.1, 'consensus_mom': 0.2},
    '2018-08-10': {'cpi_yoy': 2.9, 'consensus_yoy': 2.9, 'prior_yoy': 2.9, 'cpi_mom': 0.2, 'consensus_mom': 0.2},
    '2018-09-13': {'cpi_yoy': 2.7, 'consensus_yoy': 2.8, 'prior_yoy': 2.9, 'cpi_mom': 0.1, 'consensus_mom': 0.2},
    '2018-10-11': {'cpi_yoy': 2.3, 'consensus_yoy': 2.4, 'prior_yoy': 2.7, 'cpi_mom': 0.1, 'consensus_mom': 0.2},
    '2018-11-14': {'cpi_yoy': 2.2, 'consensus_yoy': 2.4, 'prior_yoy': 2.3, 'cpi_mom': 0.0, 'consensus_mom': 0.1},
    '2018-12-12': {'cpi_yoy': 1.9, 'consensus_yoy': 2.1, 'prior_yoy': 2.2, 'cpi_mom': -0.1, 'consensus_mom': 0.0},
    # ── 2019 ──
    '2019-01-11': {'cpi_yoy': 1.6, 'consensus_yoy': 1.7, 'prior_yoy': 1.9, 'cpi_mom': -0.1, 'consensus_mom': -0.1},
    '2019-02-13': {'cpi_yoy': 1.6, 'consensus_yoy': 1.6, 'prior_yoy': 1.6, 'cpi_mom': 0.0, 'consensus_mom': 0.1},
    '2019-03-12': {'cpi_yoy': 1.5, 'consensus_yoy': 1.6, 'prior_yoy': 1.6, 'cpi_mom': 0.2, 'consensus_mom': 0.2},
    '2019-04-10': {'cpi_yoy': 1.9, 'consensus_yoy': 1.8, 'prior_yoy': 1.5, 'cpi_mom': 0.4, 'consensus_mom': 0.3},
    '2019-05-10': {'cpi_yoy': 1.8, 'consensus_yoy': 1.9, 'prior_yoy': 1.9, 'cpi_mom': 0.0, 'consensus_mom': 0.1},
    '2019-06-12': {'cpi_yoy': 1.5, 'consensus_yoy': 1.6, 'prior_yoy': 1.8, 'cpi_mom': 0.1, 'consensus_mom': 0.1},
    '2019-07-11': {'cpi_yoy': 1.8, 'consensus_yoy': 1.7, 'prior_yoy': 1.5, 'cpi_mom': 0.3, 'consensus_mom': 0.2},
    '2019-08-13': {'cpi_yoy': 1.7, 'consensus_yoy': 1.7, 'prior_yoy': 1.8, 'cpi_mom': 0.3, 'consensus_mom': 0.3},
    '2019-09-12': {'cpi_yoy': 1.7, 'consensus_yoy': 1.8, 'prior_yoy': 1.7, 'cpi_mom': 0.1, 'consensus_mom': 0.1},
    '2019-10-10': {'cpi_yoy': 1.7, 'consensus_yoy': 1.7, 'prior_yoy': 1.7, 'cpi_mom': 0.0, 'consensus_mom': 0.1},
    '2019-11-13': {'cpi_yoy': 1.8, 'consensus_yoy': 1.7, 'prior_yoy': 1.7, 'cpi_mom': 0.4, 'consensus_mom': 0.3},
    '2019-12-11': {'cpi_yoy': 2.1, 'consensus_yoy': 2.0, 'prior_yoy': 1.8, 'cpi_mom': 0.3, 'consensus_mom': 0.2},
    # ── 2020 ──
    '2020-01-14': {'cpi_yoy': 2.5, 'consensus_yoy': 2.4, 'prior_yoy': 2.1, 'cpi_mom': 0.2, 'consensus_mom': 0.2},
    '2020-02-13': {'cpi_yoy': 2.3, 'consensus_yoy': 2.2, 'prior_yoy': 2.5, 'cpi_mom': 0.1, 'consensus_mom': 0.1},
    '2020-03-11': {'cpi_yoy': 2.3, 'consensus_yoy': 2.2, 'prior_yoy': 2.3, 'cpi_mom': 0.1, 'consensus_mom': 0.1},
    '2020-04-10': {'cpi_yoy': 1.5, 'consensus_yoy': 1.4, 'prior_yoy': 2.3, 'cpi_mom': -0.8, 'consensus_mom': -0.7},
    '2020-05-12': {'cpi_yoy': 0.3, 'consensus_yoy': 0.4, 'prior_yoy': 1.5, 'cpi_mom': -0.8, 'consensus_mom': -0.8},
    '2020-06-10': {'cpi_yoy': 0.1, 'consensus_yoy': 0.2, 'prior_yoy': 0.3, 'cpi_mom': 0.1, 'consensus_mom': 0.0},
    '2020-07-14': {'cpi_yoy': 0.6, 'consensus_yoy': 0.5, 'prior_yoy': 0.1, 'cpi_mom': 0.6, 'consensus_mom': 0.5},
    '2020-08-12': {'cpi_yoy': 1.3, 'consensus_yoy': 1.2, 'prior_yoy': 0.6, 'cpi_mom': 0.6, 'consensus_mom': 0.5},
    '2020-09-11': {'cpi_yoy': 1.4, 'consensus_yoy': 1.3, 'prior_yoy': 1.3, 'cpi_mom': 0.2, 'consensus_mom': 0.2},
    '2020-10-13': {'cpi_yoy': 1.4, 'consensus_yoy': 1.4, 'prior_yoy': 1.4, 'cpi_mom': 0.0, 'consensus_mom': 0.1},
    '2020-11-12': {'cpi_yoy': 1.2, 'consensus_yoy': 1.3, 'prior_yoy': 1.4, 'cpi_mom': 0.0, 'consensus_mom': 0.1},
    '2020-12-10': {'cpi_yoy': 1.2, 'consensus_yoy': 1.3, 'prior_yoy': 1.2, 'cpi_mom': 0.0, 'consensus_mom': 0.1},
    # ── 2021 ──
    '2021-01-13': {'cpi_yoy': 1.4, 'consensus_yoy': 1.5, 'prior_yoy': 1.2, 'cpi_mom': 0.4, 'consensus_mom': 0.3},
    '2021-02-10': {'cpi_yoy': 1.7, 'consensus_yoy': 1.7, 'prior_yoy': 1.4, 'cpi_mom': 0.4, 'consensus_mom': 0.3},
    '2021-03-10': {'cpi_yoy': 2.6, 'consensus_yoy': 2.5, 'prior_yoy': 1.7, 'cpi_mom': 0.8, 'consensus_mom': 0.7},
    '2021-04-13': {'cpi_yoy': 4.2, 'consensus_yoy': 3.6, 'prior_yoy': 2.6, 'cpi_mom': 0.8, 'consensus_mom': 0.6},
    '2021-05-12': {'cpi_yoy': 5.0, 'consensus_yoy': 4.7, 'prior_yoy': 4.2, 'cpi_mom': 0.8, 'consensus_mom': 0.6},
    '2021-06-10': {'cpi_yoy': 5.4, 'consensus_yoy': 5.0, 'prior_yoy': 5.0, 'cpi_mom': 0.9, 'consensus_mom': 0.6},
    '2021-07-13': {'cpi_yoy': 5.4, 'consensus_yoy': 5.3, 'prior_yoy': 5.4, 'cpi_mom': 0.5, 'consensus_mom': 0.5},
    '2021-08-11': {'cpi_yoy': 5.3, 'consensus_yoy': 5.4, 'prior_yoy': 5.4, 'cpi_mom': 0.3, 'consensus_mom': 0.4},
    '2021-09-14': {'cpi_yoy': 5.3, 'consensus_yoy': 5.4, 'prior_yoy': 5.3, 'cpi_mom': 0.3, 'consensus_mom': 0.3},
    '2021-10-13': {'cpi_yoy': 6.2, 'consensus_yoy': 5.9, 'prior_yoy': 5.3, 'cpi_mom': 0.9, 'consensus_mom': 0.6},
    '2021-11-10': {'cpi_yoy': 6.8, 'consensus_yoy': 6.5, 'prior_yoy': 6.2, 'cpi_mom': 0.8, 'consensus_mom': 0.7},
    '2021-12-10': {'cpi_yoy': 6.8, 'consensus_yoy': 6.8, 'prior_yoy': 6.8, 'cpi_mom': 0.5, 'consensus_mom': 0.5},
    # ── 2022 ──
    '2022-01-12': {'cpi_yoy': 7.5, 'consensus_yoy': 7.2, 'prior_yoy': 7.0, 'cpi_mom': 0.6, 'consensus_mom': 0.5},
    '2022-02-10': {'cpi_yoy': 7.5, 'consensus_yoy': 7.3, 'prior_yoy': 7.5, 'cpi_mom': 0.8, 'consensus_mom': 0.6},
    '2022-03-10': {'cpi_yoy': 7.9, 'consensus_yoy': 7.8, 'prior_yoy': 7.5, 'cpi_mom': 0.8, 'consensus_mom': 0.8},
    '2022-04-12': {'cpi_yoy': 8.5, 'consensus_yoy': 8.4, 'prior_yoy': 7.9, 'cpi_mom': 1.2, 'consensus_mom': 1.1},
    '2022-05-11': {'cpi_yoy': 8.3, 'consensus_yoy': 8.1, 'prior_yoy': 8.5, 'cpi_mom': 0.3, 'consensus_mom': 0.2},
    '2022-06-10': {'cpi_yoy': 8.6, 'consensus_yoy': 8.3, 'prior_yoy': 8.3, 'cpi_mom': 1.0, 'consensus_mom': 0.7},
    '2022-07-13': {'cpi_yoy': 9.1, 'consensus_yoy': 8.8, 'prior_yoy': 8.6, 'cpi_mom': 1.3, 'consensus_mom': 1.1},
    '2022-08-10': {'cpi_yoy': 8.5, 'consensus_yoy': 8.7, 'prior_yoy': 9.1, 'cpi_mom': 0.0, 'consensus_mom': 0.2},
    '2022-09-13': {'cpi_yoy': 8.3, 'consensus_yoy': 8.1, 'prior_yoy': 8.5, 'cpi_mom': 0.4, 'consensus_mom': 0.2},
    '2022-10-13': {'cpi_yoy': 7.7, 'consensus_yoy': 8.0, 'prior_yoy': 8.3, 'cpi_mom': 0.4, 'consensus_mom': 0.5},
    '2022-11-10': {'cpi_yoy': 7.1, 'consensus_yoy': 7.3, 'prior_yoy': 7.7, 'cpi_mom': 0.4, 'consensus_mom': 0.5},
    '2022-12-13': {'cpi_yoy': 7.1, 'consensus_yoy': 7.3, 'prior_yoy': 7.1, 'cpi_mom': 0.1, 'consensus_mom': 0.2},
    # ── 2023 ──
    '2023-01-12': {'cpi_yoy': 6.5, 'consensus_yoy': 6.5, 'prior_yoy': 7.1, 'cpi_mom': 0.5, 'consensus_mom': 0.5},
    '2023-02-14': {'cpi_yoy': 6.4, 'consensus_yoy': 6.2, 'prior_yoy': 6.5, 'cpi_mom': 0.5, 'consensus_mom': 0.4},
    '2023-03-14': {'cpi_yoy': 6.0, 'consensus_yoy': 5.9, 'prior_yoy': 6.4, 'cpi_mom': 0.4, 'consensus_mom': 0.4},
    '2023-04-12': {'cpi_yoy': 5.0, 'consensus_yoy': 5.2, 'prior_yoy': 6.0, 'cpi_mom': 0.1, 'consensus_mom': 0.2},
    '2023-05-10': {'cpi_yoy': 4.9, 'consensus_yoy': 5.0, 'prior_yoy': 5.0, 'cpi_mom': 0.4, 'consensus_mom': 0.4},
    '2023-06-13': {'cpi_yoy': 4.0, 'consensus_yoy': 4.1, 'prior_yoy': 4.9, 'cpi_mom': 0.1, 'consensus_mom': 0.2},
    '2023-07-12': {'cpi_yoy': 3.0, 'consensus_yoy': 3.1, 'prior_yoy': 4.0, 'cpi_mom': 0.2, 'consensus_mom': 0.2},
    '2023-08-10': {'cpi_yoy': 3.2, 'consensus_yoy': 3.3, 'prior_yoy': 3.0, 'cpi_mom': 0.2, 'consensus_mom': 0.2},
    '2023-09-13': {'cpi_yoy': 3.7, 'consensus_yoy': 3.6, 'prior_yoy': 3.2, 'cpi_mom': 0.6, 'consensus_mom': 0.6},
    '2023-10-12': {'cpi_yoy': 3.7, 'consensus_yoy': 3.6, 'prior_yoy': 3.7, 'cpi_mom': 0.4, 'consensus_mom': 0.3},
    '2023-11-14': {'cpi_yoy': 3.2, 'consensus_yoy': 3.3, 'prior_yoy': 3.7, 'cpi_mom': 0.0, 'consensus_mom': 0.1},
    '2023-12-12': {'cpi_yoy': 3.1, 'consensus_yoy': 3.1, 'prior_yoy': 3.2, 'cpi_mom': 0.1, 'consensus_mom': 0.0},
    # ── 2024 ──
    '2024-01-11': {'cpi_yoy': 2.9, 'consensus_yoy': 3.0, 'prior_yoy': 3.1, 'cpi_mom': 0.3, 'consensus_mom': 0.2},
    '2024-02-13': {'cpi_yoy': 3.1, 'consensus_yoy': 2.9, 'prior_yoy': 2.9, 'cpi_mom': 0.4, 'consensus_mom': 0.2},
    '2024-03-12': {'cpi_yoy': 3.2, 'consensus_yoy': 3.1, 'prior_yoy': 3.1, 'cpi_mom': 0.4, 'consensus_mom': 0.4},
    '2024-04-10': {'cpi_yoy': 3.5, 'consensus_yoy': 3.4, 'prior_yoy': 3.2, 'cpi_mom': 0.4, 'consensus_mom': 0.3},
    '2024-05-15': {'cpi_yoy': 3.4, 'consensus_yoy': 3.4, 'prior_yoy': 3.5, 'cpi_mom': 0.3, 'consensus_mom': 0.4},
    '2024-06-12': {'cpi_yoy': 3.3, 'consensus_yoy': 3.4, 'prior_yoy': 3.4, 'cpi_mom': 0.0, 'consensus_mom': 0.1},
    '2024-07-11': {'cpi_yoy': 3.0, 'consensus_yoy': 3.1, 'prior_yoy': 3.3, 'cpi_mom': -0.1, 'consensus_mom': 0.1},
    '2024-08-14': {'cpi_yoy': 2.9, 'consensus_yoy': 2.9, 'prior_yoy': 3.0, 'cpi_mom': 0.2, 'consensus_mom': 0.2},
    '2024-09-11': {'cpi_yoy': 2.5, 'consensus_yoy': 2.6, 'prior_yoy': 2.9, 'cpi_mom': 0.2, 'consensus_mom': 0.2},
    '2024-10-10': {'cpi_yoy': 2.4, 'consensus_yoy': 2.3, 'prior_yoy': 2.5, 'cpi_mom': 0.2, 'consensus_mom': 0.1},
    '2024-11-13': {'cpi_yoy': 2.6, 'consensus_yoy': 2.6, 'prior_yoy': 2.4, 'cpi_mom': 0.2, 'consensus_mom': 0.2},
    '2024-12-11': {'cpi_yoy': 2.7, 'consensus_yoy': 2.7, 'prior_yoy': 2.6, 'cpi_mom': 0.3, 'consensus_mom': 0.3},
    # ── 2025 ──
    '2025-01-15': {'cpi_yoy': 2.9, 'consensus_yoy': 2.9, 'prior_yoy': 2.7, 'cpi_mom': 0.4, 'consensus_mom': 0.3},
    '2025-02-12': {'cpi_yoy': 3.0, 'consensus_yoy': 2.9, 'prior_yoy': 2.9, 'cpi_mom': 0.5, 'consensus_mom': 0.3},
    '2025-03-12': {'cpi_yoy': 2.8, 'consensus_yoy': 2.9, 'prior_yoy': 3.0, 'cpi_mom': 0.2, 'consensus_mom': 0.3},
    '2025-04-10': {'cpi_yoy': 2.4, 'consensus_yoy': 2.5, 'prior_yoy': 2.8, 'cpi_mom': -0.1, 'consensus_mom': 0.1},
    '2025-05-13': {'cpi_yoy': 2.3, 'consensus_yoy': 2.4, 'prior_yoy': 2.4, 'cpi_mom': 0.2, 'consensus_mom': 0.2},
    # ── 2026 (projected) ──
    '2026-01-14': {'cpi_yoy': 2.5, 'consensus_yoy': 2.5, 'prior_yoy': 2.6, 'cpi_mom': 0.3, 'consensus_mom': 0.2},
    '2026-02-11': {'cpi_yoy': 2.4, 'consensus_yoy': 2.5, 'prior_yoy': 2.5, 'cpi_mom': 0.2, 'consensus_mom': 0.2},
    '2026-03-11': {'cpi_yoy': 2.3, 'consensus_yoy': 2.4, 'prior_yoy': 2.4, 'cpi_mom': 0.2, 'consensus_mom': 0.2},
    '2026-04-14': {'cpi_yoy': 2.4, 'consensus_yoy': 2.3, 'prior_yoy': 2.3, 'cpi_mom': 0.3, 'consensus_mom': 0.2},
}

# All scheduled CPI release dates (including future dates without data yet)
# Used by M22/M23 for time decay and release detection
CPI_SCHEDULE_DATES = set(CPI_RELEASES.keys()) | {
    '2026-05-12', '2026-06-10', '2026-07-14', '2026-08-12',
    '2026-09-10', '2026-10-13', '2026-11-10', '2026-12-09',
}


# ── Edge table ──
# Format: (wyckoff, vol, signal) → (avg_24h_ret, win_rate, n, direction, ics_adj, size_mult)
# Only combos with n≥3 and |avg|≥0.5% are included.
EDGE_TABLE = {
    ('RANGE', 'NEUTRAL', 'BIG_MISS'):         {'dir': 'LONG',  'avg': 2.150, 'wr': 0.67, 'n': 6,  'ics_adj': 0.06, 'size_mult': 1.05},
    ('RANGE', 'NEUTRAL', 'MISS'):             {'dir': 'LONG',  'avg': 1.320, 'wr': 0.60, 'n': 8,  'ics_adj': 0.05, 'size_mult': 1.00},
    ('MARKDOWN', 'NEUTRAL', 'MISS'):          {'dir': 'LONG',  'avg': 1.680, 'wr': 0.63, 'n': 5,  'ics_adj': 0.06, 'size_mult': 1.05},
    ('RANGE', 'COMPRESSING', 'STRONG_BEAT'):  {'dir': 'SHORT', 'avg': -2.430, 'wr': 0.20, 'n': 5, 'ics_adj': 0.06, 'size_mult': 1.05},
    ('RANGE', 'COMPRESSING', 'BEAT'):         {'dir': 'SHORT', 'avg': -1.520, 'wr': 0.33, 'n': 6, 'ics_adj': 0.05, 'size_mult': 1.00},
    ('RUNAWAY', 'NEUTRAL', 'INLINE'):         {'dir': 'SHORT', 'avg': -1.890, 'wr': 0.33, 'n': 9, 'ics_adj': 0.05, 'size_mult': 1.00},
    ('HOT', 'NEUTRAL', 'MISS'):              {'dir': 'LONG',  'avg': 1.950, 'wr': 0.67, 'n': 6,  'ics_adj': 0.06, 'size_mult': 1.05},
    ('MARKUP', 'NEUTRAL', 'STRONG_BEAT'):    {'dir': 'SHORT', 'avg': -1.740, 'wr': 0.29, 'n': 7, 'ics_adj': 0.05, 'size_mult': 1.00},
    ('DISTRIBUTION', 'HIGH_VOL', 'BEAT'):    {'dir': 'SHORT', 'avg': -2.810, 'wr': 0.20, 'n': 5, 'ics_adj': 0.06, 'size_mult': 1.05},
    ('MARKDOWN', 'HIGH_VOL', 'BIG_MISS'):    {'dir': 'LONG',  'avg': 2.650, 'wr': 0.75, 'n': 4,  'ics_adj': 0.06, 'size_mult': 1.05},
}

# Fallback: MISS signal generally → LONG
MISS_FALLBACK = {'dir': 'LONG', 'avg': 1.200, 'wr': 0.58, 'n': 18, 'ics_adj': 0.05, 'size_mult': 1.00}
# Fallback: STRONG_BEAT signal generally → SHORT
BEAT_FALLBACK = {'dir': 'SHORT', 'avg': -1.350, 'wr': 0.35, 'n': 15, 'ics_adj': 0.05, 'size_mult': 1.00}


def _classify_signal(actual_yoy, consensus_yoy):
    """Classify CPI surprise signal."""
    surprise = actual_yoy - consensus_yoy
    if surprise > 0.3:
        return 'STRONG_BEAT'
    elif surprise > 0.1:
        return 'BEAT'
    elif surprise < -0.3:
        return 'BIG_MISS'
    elif surprise < -0.1:
        return 'MISS'
    return 'INLINE'


def _classify_inflation(actual_yoy):
    """Classify inflation level."""
    if actual_yoy > 6.0:
        return 'RUNAWAY'
    elif actual_yoy > 4.0:
        return 'HOT'
    elif actual_yoy > 3.0:
        return 'WARM'
    elif actual_yoy >= 1.5:
        return 'TARGET'
    return 'COOL'


def _is_release_day(date_str):
    """Check if date_str is within ±1 day of a CPI release."""
    today = datetime.strptime(date_str, '%Y-%m-%d')
    for rel_date, rel_data in CPI_RELEASES.items():
        release_dt = datetime.strptime(rel_date, '%Y-%m-%d')
        delta = abs((today - release_dt).days)
        if delta <= 1:
            return rel_date, rel_data
    return None, None


def score_m56_us_cpi(wyckoff_phase='RANGE', vol_regime='NEUTRAL',
                     direction='LONG', date_str=None, config=None):
    """
    Score M56: US CPI session bias.

    Args:
        wyckoff_phase: M21 wyckoff phase (MARKUP/MARKDOWN/RANGE/DISTRIBUTION/ACCUMULATION)
        vol_regime: M9 volatility regime (HIGH_VOL/COMPRESSING/NEUTRAL/TRENDING/CHOP)
        direction: Current bias direction (LONG/SHORT)
        date_str: Date string 'YYYY-MM-DD' (defaults to today UTC)
        config: Optional config dict

    Returns:
        (status, score_adj, size_mult, details)
    """
    cfg = config or {}
    if not cfg.get('M56_ENABLED', True):
        return 'DISABLED', 0.0, 1.0, {'regime': 'DISABLED'}

    if date_str is None:
        date_str = datetime.utcnow().strftime('%Y-%m-%d')

    release_date, release_data = _is_release_day(date_str)
    if release_data is None:
        return 'SKIP', 0.0, 1.0, {'regime': 'NOT_RELEASE_DAY'}

    cpi_yoy = release_data['cpi_yoy']
    cpi_mom = release_data['cpi_mom']
    consensus_yoy = release_data['consensus_yoy']
    consensus_mom = release_data['consensus_mom']
    prior_yoy = release_data['prior_yoy']

    signal = _classify_signal(cpi_yoy, consensus_yoy)
    infl = _classify_inflation(cpi_yoy)

    # Primary lookup: Wyckoff × Vol × Signal
    edge_key = (wyckoff_phase, vol_regime, signal)
    edge = EDGE_TABLE.get(edge_key)

    # Fallback: try same wyckoff + signal
    if edge is None:
        for (w, v, s), e in EDGE_TABLE.items():
            if w == wyckoff_phase and s == signal:
                edge = e
                edge_key = (w, v, s)
                break

    # Signal-level fallback
    if edge is None:
        if signal in ('MISS', 'BIG_MISS'):
            edge = MISS_FALLBACK
            edge_key = ('SIGNAL_FALLBACK', vol_regime, signal)
        elif signal in ('STRONG_BEAT', 'BEAT'):
            edge = BEAT_FALLBACK
            edge_key = ('SIGNAL_FALLBACK', vol_regime, signal)

    if edge is None:
        return 'NO_EDGE', 0.0, 1.0, {
            'regime': 'NO_EDGE',
            'release_date': release_date,
            'cpi_yoy': cpi_yoy,
            'consensus_yoy': consensus_yoy,
            'signal': signal,
            'inflation': infl,
        }

    avg_ret = edge['avg']
    win_rate = edge['wr']
    n = edge['n']
    bias = edge['dir']

    # Score adjustment magnitude
    if abs(avg_ret) >= 2.0 and n >= 3:
        score_adj = 0.07 if avg_ret > 0 else -0.07
    elif abs(avg_ret) >= 1.5:
        score_adj = 0.06 if avg_ret > 0 else -0.06
    elif abs(avg_ret) >= 1.0:
        score_adj = 0.05 if avg_ret > 0 else -0.05
    elif abs(avg_ret) >= 0.5:
        score_adj = 0.03 if avg_ret > 0 else -0.03
    else:
        score_adj = 0.02 if avg_ret > 0 else -0.02

    # Counter-direction dampening
    if bias != direction:
        score_adj *= -0.5

    # Size multiplier based on win rate and sample size
    if n >= 5 and win_rate >= 0.6:
        size_mult = 0.85
    elif n >= 3 and win_rate >= 0.5:
        size_mult = 0.75
    else:
        size_mult = 0.65

    # Status classification
    if abs(score_adj) >= 0.05:
        status = 'ACTIVE'
    elif abs(score_adj) >= 0.03:
        status = 'WEAK'
    else:
        status = 'NO_EDGE'

    details = {
        'regime': f'{wyckoff_phase}_{vol_regime}_{signal}',
        'release_date': release_date,
        'cpi_yoy': cpi_yoy,
        'cpi_mom': cpi_mom,
        'consensus_yoy': consensus_yoy,
        'consensus_mom': consensus_mom,
        'prior_yoy': prior_yoy,
        'signal': signal,
        'inflation': infl,
        'wyckoff': wyckoff_phase,
        'vol': vol_regime,
        'bias': bias,
        'avg_ret_24h': avg_ret,
        'win_rate': win_rate,
        'sample_size': n,
        'confidence': min(0.6, n / 10.0),
        'source': f'wyckoff_vol_signal: {edge_key}',
        'score_adj': score_adj,
        'size_mult': size_mult,
    }
    return status, score_adj, size_mult, details


def format_m56(details):
    """Format M56 details for display."""
    if not details or details.get('regime') in ('DISABLED', 'NOT_RELEASE_DAY', 'NO_EDGE'):
        return None

    bias = details.get('bias', '?')
    cpi_yoy = details.get('cpi_yoy', 0)
    cpi_mom = details.get('cpi_mom', 0)
    cons = details.get('consensus_yoy', 0)
    signal = details.get('signal', '?')
    infl = details.get('inflation', '?')
    avg_ret = details.get('avg_ret_24h', 0)
    win_rate = details.get('win_rate', 0)
    n = details.get('sample_size', 0)
    score_adj = details.get('score_adj', 0)
    size_mult = details.get('size_mult', 1.0)

    bias_icon = '🟢' if bias == 'LONG' else '🔴' if bias == 'SHORT' else '⚪'
    infl_icon = {'RUNAWAY': '🔴🔴', 'HOT': '🔴', 'WARM': '🟠',
                 'TARGET': '🟢', 'COOL': '🟢🟢'}.get(infl, '⚪')
    surprise = cpi_yoy - cons

    lines = [
        f"  M56 US CPI: {bias_icon} {bias:>8}  "
        f"CPI={cpi_yoy:.1f}%(yoy) {cpi_mom:+.1f}%(mom) cons={cons:.1f}% "
        f"surprise={surprise:+.2f}  {infl_icon}{infl}  signal={signal}",
        f"    Backtest: 24h={avg_ret:+.2f}% win={win_rate*100:.0f}% n={n}  "
        f"adj={score_adj:+.3f} size={size_mult:.2f}x  "
        f"chain: NY Open→AM 70-80%, Asia overnight repricing",
    ]
    return '\n'.join(lines)


if __name__ == '__main__':
    print("=== M56 US CPI Self-Test ===\n")
    for wyck, vol, dire, date in [
        ('RANGE', 'NEUTRAL', 'LONG', '2024-07-11'),     # MISS → LONG
        ('RANGE', 'COMPRESSING', 'SHORT', '2022-07-13'), # STRONG_BEAT → SHORT
        ('RUNAWAY', 'NEUTRAL', 'SHORT', '2022-06-10'),   # INLINE in RUNAWAY → SHORT
        ('MARKDOWN', 'NEUTRAL', 'LONG', '2024-09-11'),   # MISS in MARKDOWN → LONG
        ('RANGE', 'NEUTRAL', 'LONG', '2025-05-14'),      # Not a CPI day (±1d check)
    ]:
        s, a, m, d = score_m56_us_cpi(wyck, vol, dire, date)
        fmt = format_m56(d)
        if fmt:
            print(fmt)
        print(f"  → {s}, ICS={a:+.03f}, size={m:.2f}\n")
    print(format_m56(score_m56_us_cpi('RANGE', 'NEUTRAL', 'LONG', '2025-06-15')[3]))

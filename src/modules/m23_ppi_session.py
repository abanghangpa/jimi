"""
M23: Macro Cascade Model — NFP → CPI → PPI → Claims session dynamics

Release sequence (8:30 AM ET each):
    0. NFP + Unemployment Rate (first Friday) — LABOR MARKET signal, high vol
    1. CPI (Tue/Wed, 2nd-3rd week) — PRIMARY signal, biggest ETH reaction
    2. PPI (Wed/Thu, 1-2 days after CPI) — CONFIRMS or DENIES CPI signal
    3. Jobless Claims (every Thursday) — BACKGROUND context, only matters at extremes

Architecture:
    NFP on Friday sets the LABOR MARKET tone. CPI the following week sets the
    INFLATION tone. PPI confirms/denies CPI. Claims provide background context.

    Cascade sequence when NFP + CPI in same week:
      NFP Friday → Claims Thursday → CPI Tuesday/Wednesday → PPI Wednesday/Thursday
      NFP surprise sets macro bias. CPI either amplifies or reverses it.

    Standalone NFP:
      High vol (avg |move| 2.01%), no directional bias, strong Asia fade pattern.
      After US dump: 60% Asia bounce. After US rally: 64% Asia fade.

Forensic findings (178 releases, 2018-2025):
    CPI/PPI:
    1. Cool CPI = ETH rallies: +1.06% intraday, +2.50% over 2 days
    2. Hot CPI = ETH dumps: -0.45% intraday, -0.90% over 2 days
    3. Cool PPI has INVERTED signal: -2.89% avg (deflation fears)
    4. PPI inline dominates (58% of events), near-zero return
    5. CPI signal is regime-dependent: best in 2022 (cool CPI +9.92%), weak in 2018-19
    6. 2-day window captures more alpha than intraday (+3.40% cool-hot spread)
    7. Jobless claims: no actionable standalone signal

    NFP (53 releases, 2022-2026):
    1. Avg |US move|: 2.01% (comparable to PPI 2.21%, CPI 2.27%)
    2. No directional bias (avg +0.06%), high volatility
    3. 1h spike → US direction agreement: 66%
    4. After US dump: 60% Asia bounce. After US rally: 64% Asia fade.
    5. NFP near CPI (within 5d): avg |move| 2.58% = 1.7x isolated NFP
    6. September danger month: avg -4.38%. November bullish: avg +2.42%.
    7. Unemployment rate released same time — market reacts to BOTH simultaneously.

Data sources:
    - BLS Employment Situation schedule (NFP + unemployment, first Friday)
    - BLS PPI/CPI release schedules (hardcoded dates 2018-2026)
    - BLS weekly jobless claims (every Thursday 8:30 AM ET)
    - Live 15m OHLCV data
    - M22 inflation regime (for fade/continuation bias)
    - FRED unemployment rate data (for Sahm Rule)
"""

from src.config import CONFIG
import numpy as np
import pandas as pd
from datetime import datetime, timedelta


# ═══════════════════════════════════════════════════════════════
# RELEASE DATES (8:30 AM ET = 13:30 UTC)
# Sourced from standalone modules (M60: PPI, M56: CPI, M37: NFP)
# ═══════════════════════════════════════════════════════════════

from src.modules.m60_us_ppi import PPI_SCHEDULE_DATES as PPI_RELEASE_DATES
from src.modules.m56_us_cpi import CPI_SCHEDULE_DATES as CPI_RELEASE_DATES

# ═══════════════════════════════════════════════════════════════
# NFP + UNEMPLOYMENT RATE (first Friday of each month, 8:30 AM ET)
# ═══════════════════════════════════════════════════════════════
# The Employment Situation report (Non-Farm Payrolls + Unemployment Rate)
# is released on the first Friday of each month at 8:30 AM ET (13:30 UTC).
# Both numbers come from the SAME report — market reacts to both simultaneously.
#
# Forensic findings (53 NFP releases, 2022-2026):
#   Avg |US move|: 2.01% (comparable to PPI 2.21%, CPI 2.27%)
#   No directional bias (avg +0.06%), high volatility
#   1h spike → US direction agreement: 66%
#   After US dump: 60% Asia bounce. After US rally: 64% Asia fade.
#   NFP near CPI (within 5d): avg |move| 2.58% = 1.7x isolated NFP (1.55%)
#   September danger month: avg -4.38%. November bullish: avg +2.42%.
#   Feb widest swings: avg |move| 3.38%.
#
# Key thresholds for unemployment rate:
#   <3.5%: Goldilocks — economy strong, Fed can be patient
#   3.5-4.0%: Normal — no strong signal
#   4.0-4.5%: Softening — recession whispers begin
#   >4.5%: Danger zone — recession fears, Fed forced to cut
#   Sahm Rule (3m avg rise ≥0.5pp from 12m low): Recession signal
#
# Cascade: NFP Friday → Claims Thursday → CPI Tue/Wed → PPI Wed/Thu
#   NFP sets LABOR MARKET tone. CPI sets INFLATION tone. PPI confirms/denies.
#   When NFP + CPI in same week, moves amplify 1.7x.

from src.modules.m37_nfp import NFP_SCHEDULE_DATES as NFP_RELEASE_DATES

# NFP seasonality — avg US session move by month (from 53 releases, 2022-2026)
NFP_SEASONALITY = {
    1:  {'avg_move': -0.03, 'avg_abs': 0.95,  'bias': 'NEUTRAL'},
    2:  {'avg_move': +0.84, 'avg_abs': 3.38,  'bias': 'VOLATILE'},
    3:  {'avg_move': -1.56, 'avg_abs': 2.23,  'bias': 'BEARISH'},
    4:  {'avg_move': +1.60, 'avg_abs': 1.60,  'bias': 'BULLISH'},
    5:  {'avg_move': +1.20, 'avg_abs': 1.59,  'bias': 'BULLISH'},
    6:  {'avg_move': -0.28, 'avg_abs': 1.27,  'bias': 'NEUTRAL'},
    7:  {'avg_move': +0.79, 'avg_abs': 0.84,  'bias': 'NEUTRAL'},
    8:  {'avg_move': -1.98, 'avg_abs': 2.04,  'bias': 'BEARISH'},
    9:  {'avg_move': -4.38, 'avg_abs': 4.38,  'bias': 'DANGER'},
    10: {'avg_move': +1.08, 'avg_abs': 1.08,  'bias': 'BULLISH'},
    11: {'avg_move': +2.42, 'avg_abs': 2.73,  'bias': 'BULLISH'},
    12: {'avg_move': +0.60, 'avg_abs': 2.17,  'bias': 'NEUTRAL'},
}

# NFP Asia fade rates (from 53 releases, 2022-2026)
NFP_ASIA_FADE_AFTER_DUMP = 0.60   # 60% Asia bounces after US dump
NFP_ASIA_FADE_AFTER_RALLY = 0.64  # 64% Asia fades after US rally

# NFP 1h spike → US session direction agreement
NFP_1H_AGREEMENT = 0.66  # 66% of the time, 1h spike direction = full US session direction

# NFP proximity to CPI amplification
NFP_NEAR_CPI_AMPLIFIER = 1.70  # NFP within 5d of CPI has 1.7x larger avg |move|

# NFP spike accuracy by year (1h spike direction predicts US session direction)
NFP_SPIKE_ACCURACY = {
    2022: 0.75, 2023: 0.58, 2024: 0.50, 2025: 0.75, 2026: 0.80,
}

# NFP fade rate by regime
NFP_REGIME_FADE_RATES = {
    'TIGHTENING':      0.40,
    'EASING':          0.35,
    'CRISIS_RECOVERY': 0.50,
    'BULL':            0.55,
    'BEAR':            0.45,
    'RECOVERY':        0.50,
    'ACCELERATION':    0.55,
    'STAGFLATION':     0.60,
    'STAGFLATION_HOT': 0.65,
}

# Unemployment rate thresholds for scoring
UNEMP_RATE_GOLDILOCKS = 3.5     # Below = economy strong
UNEMP_RATE_NORMAL = 4.0         # Below = normal
UNEMP_RATE_SOFTENING = 4.5      # Above = recession whispers
UNEMP_RATE_DANGER = 5.0         # Above = recession fears, Fed forced to cut
UNEMP_RATE_CRISIS = 6.0         # Above = full crisis


# ═══════════════════════════════════════════════════════════════
# JOBLESS CLAIMS DATA (released every Thursday 8:30 AM ET)
# ═══════════════════════════════════════════════════════════════
# Weekly initial jobless claims — the most timely labor market indicator.
# Released every Thursday at 8:30 AM ET (13:30 UTC) by DOL.
# Data covers the week ending the previous Saturday.
#
# Key interaction with CPI×PPI:
#   Claims low + CPI falling  = Goldilocks → Fed can cut → ETH rallies
#   Claims low + CPI rising   = No catalyst → Fed holds → ETH stuck
#   Claims high + CPI falling = Recession fear → Fed will cut → ETH dips then rallies
#   Claims high + CPI rising  = STAGFLATION → Fed trapped → ETH dumps
#
# Historical monthly averages (from DOL/FRED data):
#   2021: 900K→200K (COVID recovery)
#   2022: 210K→195K (tight labor market)
#   2023: 190K→210K (normalizing)
#   2024: 210K→215K (softening)
#   2025: 205K→199K (tariff spikes to 240K, then recovery)
#   2026: 210K→200K (stable but unemployment sticky at 4.3%)

# Monthly average jobless claims (thousands) — used for trend detection
JOBLESS_CLAIMS_MONTHLY_AVG = {
    # 2021 — COVID recovery
    '2021-01': 900, '2021-02': 750, '2021-03': 650, '2021-04': 570,
    '2021-05': 450, '2021-06': 400, '2021-07': 380, '2021-08': 350,
    '2021-09': 330, '2021-10': 280, '2021-11': 250, '2021-12': 200,
    # 2022 — tight labor market
    '2022-01': 210, '2022-02': 200, '2022-03': 190, '2022-04': 185,
    '2022-05': 190, '2022-06': 195, '2022-07': 195, '2022-08': 200,
    '2022-09': 195, '2022-10': 190, '2022-11': 190, '2022-12': 195,
    # 2023 — normalizing
    '2023-01': 190, '2023-02': 195, '2023-03': 200, '2023-04': 200,
    '2023-05': 210, '2023-06': 215, '2023-07': 220, '2023-08': 220,
    '2023-09': 215, '2023-10': 215, '2023-11': 210, '2023-12': 210,
    # 2024 — softening
    '2024-01': 210, '2024-02': 215, '2024-03': 215, '2024-04': 220,
    '2024-05': 220, '2024-06': 225, '2024-07': 235, '2024-08': 230,
    '2024-09': 225, '2024-10': 220, '2024-11': 215, '2024-12': 215,
    # 2025 — tariff shock
    '2025-01': 205, '2025-02': 210, '2025-03': 210, '2025-04': 215,
    '2025-05': 240, '2025-06': 230, '2025-07': 240, '2025-08': 225,
    '2025-09': 215, '2025-10': 210, '2025-11': 220, '2025-12': 199,
    # 2026 — stable but stagflation risk
    '2026-01': 210, '2026-02': 227, '2026-03': 215, '2026-04': 200,
    '2026-05': 200,
}

# Unemployment rate monthly (for Sahm Rule tracking)
UNEMPLOYMENT_RATE_MONTHLY = {
    # 2021
    '2021-01': 6.3, '2021-02': 6.2, '2021-03': 6.0, '2021-04': 6.1,
    '2021-05': 5.8, '2021-06': 5.9, '2021-07': 5.4, '2021-08': 5.2,
    '2021-09': 4.7, '2021-10': 4.6, '2021-11': 4.2, '2021-12': 3.9,
    # 2022
    '2022-01': 4.0, '2022-02': 3.8, '2022-03': 3.6, '2022-04': 3.6,
    '2022-05': 3.6, '2022-06': 3.6, '2022-07': 3.5, '2022-08': 3.7,
    '2022-09': 3.5, '2022-10': 3.7, '2022-11': 3.7, '2022-12': 3.5,
    # 2023
    '2023-01': 3.4, '2023-02': 3.6, '2023-03': 3.5, '2023-04': 3.4,
    '2023-05': 3.7, '2023-06': 3.6, '2023-07': 3.5, '2023-08': 3.8,
    '2023-09': 3.8, '2023-10': 3.9, '2023-11': 3.7, '2023-12': 3.7,
    # 2024
    '2024-01': 3.7, '2024-02': 3.9, '2024-03': 3.8, '2024-04': 3.9,
    '2024-05': 4.0, '2024-06': 4.1, '2024-07': 4.3, '2024-08': 4.2,
    '2024-09': 4.1, '2024-10': 4.1, '2024-11': 4.2, '2024-12': 4.1,
    # 2025
    '2025-01': 4.0, '2025-02': 4.1, '2025-03': 4.2, '2025-04': 4.2,
    '2025-05': 4.1, '2025-06': 4.2, '2025-07': 4.2, '2025-08': 4.3,
    '2025-09': 4.3, '2025-10': 4.3, '2025-11': 4.4, '2025-12': 4.4,
    # 2026
    '2026-01': 4.3, '2026-02': 4.3, '2026-03': 4.3, '2026-04': 4.3,
}

# ── FRED Cache Override ──────────────────────────────────────
# If data/fred/claims_cache.json exists (from fetch_fred_claims.py),
# override the hardcoded dicts with live FRED data.
def _load_fred_cache():
    """Override hardcoded claims/unemployment dicts with FRED cache if available."""
    import json as _json
    import os as _os
    cache_path = _os.path.join(
        _os.path.dirname(_os.path.dirname(_os.path.dirname(__file__))),
        "data", "fred", "claims_cache.json")
    if not _os.path.exists(cache_path):
        return
    try:
        with open(cache_path) as f:
            cache = _json.load(f)
        icsa = cache.get("icsa", {}).get("monthly_avg", {})
        unrate = cache.get("unrate", {}).get("monthly", {})
        if icsa:
            # Merge: cache overrides hardcoded, but keep older hardcoded months
            # that cache doesn't have (cache may start at 2021-01)
            merged_claims = dict(JOBLESS_CLAIMS_MONTHLY_AVG)
            merged_claims.update(icsa)
            JOBLESS_CLAIMS_MONTHLY_AVG.clear()
            JOBLESS_CLAIMS_MONTHLY_AVG.update(merged_claims)
        if unrate:
            merged_unemp = dict(UNEMPLOYMENT_RATE_MONTHLY)
            merged_unemp.update(unrate)
            UNEMPLOYMENT_RATE_MONTHLY.clear()
            UNEMPLOYMENT_RATE_MONTHLY.update(merged_unemp)
    except Exception:
        pass  # silently fall back to hardcoded

_load_fred_cache()

# Jobless claims thresholds for signal classification
CLAIMS_LOW_THRESHOLD = 210       # Below this = tight labor market
CLAIMS_ELEVATED_THRESHOLD = 225  # Above this = labor softening
CLAIMS_SPIKE_THRESHOLD = 240     # Above this = tariff/shock spike
CLAIMS_CRISIS_THRESHOLD = 280    # Above this = recession territory

# Claims trend: 4-week moving average change direction
CLAIMS_TREND_RISING_THRESHOLD = 5    # K increase over 4 weeks = rising
CLAIMS_TREND_FALLING_THRESHOLD = -5  # K decrease over 4 weeks = falling

# ═══════════════════════════════════════════════════════════════
# CPI CASCADE MODEL — Primary Signal + PPI Confirmation
# ═══════════════════════════════════════════════════════════════
# CPI surprise is the PRIMARY signal. PPI confirms or denies.
# Claims are background context (only extremes matter).
#
# Forensic-backed values (178 releases, 2018-2025):
#   Cool CPI intraday: +1.06% avg, 2-day: +2.50% avg
#   Hot CPI intraday:  -0.45% avg, 2-day: -0.90% avg
#   Cool PPI: -2.89% avg (INVERTED — deflation fears)
#   Hot PPI:  +0.07% avg (noisy, no directional signal)
#   Claims: no actionable standalone signal

# CPI PRIMARY SIGNAL → (base_move, base_confidence, fed_bias)
# These are the BASE values before PPI confirmation and regime adjustment.
CPI_PRIMARY = {
    'COOL':   (+1.06, 'HIGH', 'CUT'),       # Cool CPI = buy ETH (strongest macro signal)
    'WARM':   (+0.27, 'MEDIUM', 'HOLD'),     # Inline/warm = noise
    'HOT':    (-0.45, 'MEDIUM', 'HOLD'),     # Hot CPI = sell ETH (conditional on regime)
}

# PPI CONFIRMATION MODIFIER — applied on top of CPI signal
# When PPI releases 1-2 days after CPI, it either confirms or denies.
# Key insight: PPI cool often = deflation fears (bearish), not relief.
# PPI hot with CPI cool = pipeline inflation building (caution).
PPI_CONFIRMATION = {
    # (cpi_surprise, ppi_surprise) → (move_modifier, confidence_modifier, description)
    ('COOL', 'COOL'):   (+0.50, +0.10, 'Both cool — strong disinflation, Fed can cut'),
    ('COOL', 'WARM'):   (+0.00, +0.00, 'CPI cool, PPI inline — signal intact'),
    ('COOL', 'HOT'):    (-0.30, -0.10, 'CPI cool but PPI hot — pipeline inflation building, signal weakened'),
    ('WARM', 'COOL'):   (-0.80, -0.10, 'CPI inline, PPI cool — deflation fears, bearish'),
    ('WARM', 'WARM'):   (+0.00, +0.00, 'Both inline — no signal'),
    ('WARM', 'HOT'):    (-0.20, +0.00, 'CPI inline, PPI hot — mild inflation concern'),
    ('HOT', 'COOL'):    (+0.30, -0.10, 'CPI hot, PPI cool — mixed signals, reduced conviction'),
    ('HOT', 'WARM'):    (+0.00, +0.00, 'CPI hot, PPI inline — hot signal intact'),
    ('HOT', 'HOT'):     (-0.80, +0.10, 'Both hot — persistent inflation, Fed can\'t cut'),
}

# CLAIMS MODIFIER — background context, only extreme prints matter
# Claims alone don't move ETH, but they amplify/dampen the CPI signal.
CLAIMS_MODIFIER = {
    'LOW':       (+0.20, 'Tight labor — economy strong, Fed has room'),
    'NORMAL':    (+0.00, 'Normal labor market — no modifier'),
    'ELEVATED':  (-0.30, 'Labor softening — recession fear amplifies hot CPI'),
    'SPIKE':     (-0.80, 'Claims spike — risk-off, amplifies any hot signal'),
    'CRISIS':    (-1.50, 'Claims crisis — macro dominant, crypto sells off'),
}

# Regime sensitivity multiplier — how much the macro signal matters by era
# Based on forensic finding that ETH's macro sensitivity evolved dramatically
REGIME_SENSITIVITY = {
    'TIGHTENING':      0.60,   # 2018: crypto-driven, macro is noise
    'EASING':          0.50,   # 2019: crypto winter, macro backdrop benign
    'CRISIS_RECOVERY': 0.80,   # 2020: COVID created first macro-driven crash
    'BULL':            0.85,   # 2021: institutional adoption, trades like risk asset
    'BEAR':            1.20,   # 2022: Fed tightening dominates all risk assets
    'RECOVERY':        0.75,   # 2023: disinflation narrative, signal weakening
    'ACCELERATION':    0.65,   # 2024: ETF narrative competes with macro
    'STAGFLATION':     1.00,   # 2025: moderate sensitivity
    'STAGFLATION_HOT': 1.10,   # 2026: stagflation = high sensitivity
}

# Expected ETH move by (regime, cpi_surprise) — from forensic data
# This is the regime-conditional expected move used for scoring
REGIME_CPI_EXPECTED = {
    # 2022 inflation scare: cool CPI was EXPLOSIVE
    ('BEAR', 'COOL'):    +9.92,
    ('BEAR', 'HOT'):     -3.33,
    # 2021 bull: hot CPI sometimes rallied (money printing narrative)
    ('BULL', 'COOL'):    +1.88,
    ('BULL', 'HOT'):     -0.20,
    # 2023 recovery: signal weakening
    ('RECOVERY', 'COOL'): -0.55,
    ('RECOVERY', 'HOT'):  +0.84,
    # 2024 acceleration: noise
    ('ACCELERATION', 'COOL'): -0.09,
    ('ACCELERATION', 'HOT'):  +0.06,
    # 2025-2026 stagflation
    ('STAGFLATION', 'COOL'): +3.00,
    ('STAGFLATION', 'HOT'):  -2.00,
    ('STAGFLATION_HOT', 'COOL'): +4.00,
    ('STAGFLATION_HOT', 'HOT'):  -3.00,
}

# Legacy combo matrix (kept for claims context compatibility)
COMBO_MATRIX = {
    ('CPI_COOL', 'PPI_COOL', 'CLAIMS_LOW'):     (+5.0, 'HIGH', 'CUT'),
    ('CPI_COOL', 'PPI_HOT', 'CLAIMS_LOW'):      (+2.0, 'MEDIUM', 'HOLD'),
    ('CPI_HOT', 'PPI_COOL', 'CLAIMS_LOW'):       (+1.0, 'MEDIUM', 'HOLD'),
    ('CPI_HOT', 'PPI_HOT', 'CLAIMS_LOW'):        (-2.0, 'MEDIUM', 'HOLD'),
    ('CPI_COOL', 'PPI_COOL', 'CLAIMS_NORMAL'):   (+3.0, 'HIGH', 'CUT'),
    ('CPI_COOL', 'PPI_HOT', 'CLAIMS_NORMAL'):    (+0.5, 'MEDIUM', 'HOLD'),
    ('CPI_HOT', 'PPI_COOL', 'CLAIMS_NORMAL'):    (-1.0, 'MEDIUM', 'HOLD'),
    ('CPI_HOT', 'PPI_HOT', 'CLAIMS_NORMAL'):     (-3.0, 'HIGH', 'HOLD'),
    ('CPI_COOL', 'PPI_COOL', 'CLAIMS_ELEVATED'): (-2.0, 'MEDIUM', 'CUT'),
    ('CPI_COOL', 'PPI_HOT', 'CLAIMS_ELEVATED'):  (-4.0, 'HIGH', 'HOLD'),
    ('CPI_HOT', 'PPI_COOL', 'CLAIMS_ELEVATED'):  (-3.0, 'HIGH', 'HOLD'),
    ('CPI_HOT', 'PPI_HOT', 'CLAIMS_ELEVATED'):   (-6.0, 'HIGH', 'TRAPPED'),
}

# Historical combo outcomes (from our analysis)
COMBO_HISTORICAL = {
    # (period, combo, eth_result)
    'Q1_2024': ('CPI_HOT', 'PPI_COOL', 'CLAIMS_LOW'),      # ETH $3,500 peak
    'Q3_2024': ('CPI_COOL', 'PPI_COOL', 'CLAIMS_ELEVATED'), # ETH $2,500 (Sahm)
    'Q2_2025': ('CPI_COOL', 'PPI_MIXED', 'CLAIMS_SPIKE'),   # ETH $1,800→$3,200
    'Sep_2025': ('CPI_COOL', 'PPI_COOL', 'CLAIMS_NORMAL'),  # ETH $2,600 (cut)
    'Apr_2026': ('CPI_HOT', 'PPI_HOT', 'CLAIMS_NORMAL'),    # ETH $2,253 (stagflation)
}

# Fed response mapping by combo
FED_RESPONSE = {
    'CUT':      'Fed can cut → risk-on → ETH rallies',
    'HOLD':     'Fed holds → no catalyst → ETH range-bound',
    'TRAPPED':  'Fed trapped (can\'t cut, won\'t hike) → ETH dumps',
    'HIKE':     'Fed must hike → risk-off → ETH crashes',
}

# Release time: 8:30 AM ET = 13:30 UTC
RELEASE_HOUR_UTC = 13
RELEASE_MINUTE_UTC = 30

# Session windows (UTC)
US_SESSION_START = (13, 30)   # 8:30 AM ET
US_SESSION_END = (21, 0)      # 4:00 PM ET
ASIA_SESSION_START = (0, 0)   # next day 00:00 UTC
ASIA_SESSION_END = (8, 0)     # next day 08:00 UTC
UK_SESSION_START = (7, 0)     # London open 07:00 UTC (8:00 BST)
UK_SESSION_END = (16, 0)      # London close 16:00 UTC (17:00 BST)


# ═══════════════════════════════════════════════════════════════
# HISTORICAL STATS (from 199-release analysis, 2018-2026)
# ═══════════════════════════════════════════════════════════════

# Spike→US accuracy by type and year
SPIKE_ACCURACY_PPI = {
    2018: 0.56, 2019: 0.82, 2020: 0.57, 2021: 0.60,
    2022: 0.76, 2023: 0.50, 2024: 0.82, 2025: 0.83, 2026: 0.80,
}

SPIKE_ACCURACY_CPI = {
    2018: 0.56, 2019: 0.82, 2020: 0.57, 2021: 0.62,
    2022: 0.77, 2023: 0.50, 2024: 0.82, 2025: 0.71, 2026: 0.67,
}

# Overall spike accuracy (combined)
SPIKE_ACCURACY = {
    2018: 0.56, 2019: 0.82, 2020: 0.57, 2021: 0.62,
    2022: 0.77, 2023: 0.50, 2024: 0.82, 2025: 0.71, 2026: 0.71,
}

# Fade rate by regime (from full 199-release analysis)
REGIME_FADE_RATES = {
    'TIGHTENING':      0.30,   # 2018: 30%
    'EASING':          0.33,   # 2019: 33%
    'CRISIS_RECOVERY': 0.48,   # 2020: 48%
    'BULL':            0.12,   # 2021: 12%
    'BEAR':            0.29,   # 2022: 29%
    'RECOVERY':        0.29,   # 2023: 29%
    'ACCELERATION':    0.29,   # 2024: 29%
    'STAGFLATION':     0.46,   # 2025-2026: 46%
    'STAGFLATION_HOT': 0.50,
}

# Asia gap reliability by year (combined PPI+CPI)
GAP_RELIABILITY = {
    2018: 0.65, 2019: 0.71, 2020: 0.63, 2021: 0.75,
    2022: 0.71, 2023: 0.67, 2024: 0.71, 2025: 0.79, 2026: 0.75,
}

# Average US session move by type (from analysis)
AVG_US_MOVE = {
    'PPI': -0.638,   # PPI has stronger directional bias
    'CPI': -0.035,   # CPI is more noise than signal
}

# Average US range by type
AVG_US_RANGE = {
    'PPI': 4.348,
    'CPI': 4.661,    # CPI produces wider ranges
}

# ═══════════════════════════════════════════════════════════════
# 8-YEAR SWEEP REVERSAL STATS (2019-2026, 80+ PPI/CPI releases)
# ═══════════════════════════════════════════════════════════════

# Reversal rate by year (PPI hot dumps only)
PPI_REVERSAL_RATE_BY_YEAR = {
    2019: 0.50,  # 2/4 — neutral, Fed cutting
    2020: 0.40,  # 2/5 — COVID deflation
    2021: 0.67,  # 4/6 — inflation rising, bull
    2022: 0.29,  # 2/7 — inflation peaking, bear (PATTERN BREAKS)
    2023: 0.40,  # 2/5 — deflation, low vol
    2024: 0.80,  # 4/5 — re-acceleration
    2025: 0.60,  # 3/5 — ATH + max long
    2026: 1.00,  # 3/3 — stagflation + crowded
}

# Reversal rate by US dump size (PPI, 2019-2026)
# Small dumps (-0.5% to -1.5%) are noise
# Medium dumps (-1.5% to -2.5%) have moderate signal
# Big dumps (-2.5% to -3.6%) are the real signal
PPI_REVERSAL_BY_DUMP_SIZE = {
    'SMALL':   0.60,  # -0.5% to -1.5%: 60% reversal (mixed)
    'MEDIUM':  0.71,  # -1.5% to -2.5%: 71% reversal (good)
    'BIG':     0.67,  # -2.5% to -3.6%: 67% reversal (but when it reverses, recovery is strong)
    'CRASH':   0.00,  # <-4%: genuine crash, no reversal (Jun 2022, Jun 2025)
}

# Average recovery % by dump size (when reversal occurs)
PPI_AVG_RECOVERY_BY_SIZE = {
    'SMALL':   120,   # small sweeps tend to overshoot
    'MEDIUM':  90,    # medium sweeps recover most
    'BIG':     85,    # big sweeps recover most but not all
    'CRASH':   0,     # no reversal
}

# Reversal rate by inflation regime (PPI hot dumps, 2019-2026)
PPI_REVERSAL_BY_REGIME = {
    'INFLATION_RISING':  0.73,  # 2021+2024+2026: market buys dips
    'INFLATION_PEAKING': 0.29,  # 2022: genuinely bearish
    'DEFLATION':         0.40,  # 2020+2023: hot prints are noise
    'NEUTRAL':           0.50,  # 2019: mixed
}

# Crash detection thresholds
CRASH_GAP_FLAT_MAX = 0.3      # gap < 0.3% = flat (no directional gap)
CRASH_ASIA_RANGE_MIN = 7.0    # Asia range > 7% = crash territory
CRASH_NO_SWEEP = True         # no sweep pattern = genuine continuation

# ═══════════════════════════════════════════════════════════════
# UK SESSION STATS (London: 07:00-16:00 UTC)
# ═══════════════════════════════════════════════════════════════
# London opens after Asia closes (or overlaps tail end) and has
# full visibility of both US reaction AND Asia overnight response.
# Key dynamics:
#   1. London often CONTINUES Asia direction when gap held (momentum)
#   2. London FADES Asia when sweep-reversal occurred (smart money fade)
#   3. London is the "decision session" — sets the tone for next US open
#   4. UK session tends to be the highest-volume crypto session

# UK continuation rate after Asia held gap (by regime)
# Backtested from 182 PPI/CPI releases (2018-2026)
# Overall: 51.1% continue, 40.4% fade (94 gap-held instances)
UK_CONTINUATION_GAP_HELD = {
    'TIGHTENING':      0.38,   # 2018: 37.5%
    'EASING':          0.50,   # 2019: 50.0%
    'CRISIS_RECOVERY': 0.71,   # 2020: 71.4%
    'BULL':            0.58,   # 2021: 58.3%
    'BEAR':            0.73,   # 2022: 72.7%
    'RECOVERY':        0.36,   # 2023: 36.4%
    'ACCELERATION':    0.46,   # 2024: 46.2%
    'STAGFLATION':     0.53,   # 2025: 53.3%
    'STAGFLATION_HOT': 0.20,   # 2026: 20.0% (5 samples)
}

# UK fade rate after Asia sweep-reversal (by regime)
# Backtested from 182 PPI/CPI releases (2018-2026)
# Overall: 38.2% fade, 30.3% continue (76 sweep-reversal instances)
# Morning sweep → UK fades 59.3% (27 instances)
UK_FADE_SWEEP_REVERSAL = {
    'TIGHTENING':      0.50,   # 2018: 50.0%
    'EASING':          0.30,   # 2019: 30.0%
    'CRISIS_RECOVERY': 0.36,   # 2020: 36.4%
    'BULL':            0.44,   # 2021: 44.4%
    'BEAR':            0.36,   # 2022: 36.4%
    'RECOVERY':        0.33,   # 2023: 33.3%
    'ACCELERATION':    0.33,   # 2024: 33.3%
    'STAGFLATION':     0.29,   # 2025: 28.6%
    'STAGFLATION_HOT': 0.67,   # 2026: 66.7% (3 samples — small n!)
}

# Average UK move as % of Asia move
# Backtested: UK actually moves MORE than Asia on average (1.4x vol ratio)
UK_MOVE_RATIO_AVG = {
    'CONTINUATION': 1.19,   # UK moves 119% of Asia's move (momentum amplified)
    'FADE':         1.01,   # UK moves 101% of Asia's move (full reversal)
    'FLAT':         0.10,   # minimal
}

# UK session direction after (US_dump + Asia_fade) combo
# This is the "double reversal" scenario — US dumps, Asia fades, UK decides
# Backtested: 58% bounce (continues Asia's fade), 42% dumps again
UK_AFTER_DOUBLE_REVERSAL = {
    'BOUNCE':    0.58,
    'CONTINUE':  0.42,
}

# UK session direction after (US_dump + Asia_continuation) combo
# Both US and Asia dumped — is London the capitulation or more pain?
# Backtested: 52% bounce, 48% continues
UK_AFTER_DOUBLE_DUMP = {
    'BOUNCE':    0.52,
    'CONTINUE':  0.48,
}

# Asia move averages by (direction, regime) — from full analysis
ASIA_MOVE_AVG = {
    ('DUMP', 'TIGHTENING'):      -0.76,   # continuation
    ('DUMP', 'EASING'):          +0.30,   # mild fade
    ('DUMP', 'CRISIS_RECOVERY'): -1.04,   # strong continuation
    ('DUMP', 'BULL'):            +0.42,   # fade (Asia buys dips)
    ('DUMP', 'BEAR'):            +0.00,   # flat
    ('DUMP', 'RECOVERY'):        +0.17,   # mild fade
    ('DUMP', 'ACCELERATION'):    +0.18,   # mild fade
    ('DUMP', 'STAGFLATION'):     +0.62,   # fade dominant
    ('RALLY', 'TIGHTENING'):     -0.76,   # continuation
    ('RALLY', 'EASING'):         +0.30,
    ('RALLY', 'CRISIS_RECOVERY'):-1.04,
    ('RALLY', 'BULL'):           +0.42,
    ('RALLY', 'BEAR'):           +0.00,
    ('RALLY', 'RECOVERY'):       +0.17,
    ('RALLY', 'ACCELERATION'):   +0.18,
    ('RALLY', 'STAGFLATION'):    -0.62,   # Asia fades rallies
}


# ═══════════════════════════════════════════════════════════════
# HELPERS
# ═══════════════════════════════════════════════════════════════

def is_ppi_release_day(date_str=None):
    """Check if today (or given date) is a PPI release day."""
    if date_str is None:
        date_str = datetime.utcnow().strftime('%Y-%m-%d')
    return date_str in PPI_RELEASE_DATES


def is_cpi_release_day(date_str=None):
    """Check if today (or given date) is a CPI release day."""
    if date_str is None:
        date_str = datetime.utcnow().strftime('%Y-%m-%d')
    return date_str in CPI_RELEASE_DATES


def is_nfp_release_day(date_str=None):
    """Check if today (or given date) is an NFP + Unemployment Rate release day.

    NFP (Non-Farm Payrolls) and Unemployment Rate are released together
    on the first Friday of each month at 8:30 AM ET (13:30 UTC).
    """
    if date_str is None:
        date_str = datetime.utcnow().strftime('%Y-%m-%d')
    return date_str in NFP_RELEASE_DATES


def is_macro_release_day(date_str=None):
    """Check if today is any macro data release day (NFP, PPI, CPI, or Claims)."""
    if date_str is None:
        date_str = datetime.utcnow().strftime('%Y-%m-%d')
    return (date_str in PPI_RELEASE_DATES or
            date_str in CPI_RELEASE_DATES or
            date_str in NFP_RELEASE_DATES or
            is_claims_release_day(date_str))


def get_release_type(date_str):
    """Determine what macro data is released on a given date.

    Returns: combination of 'NFP', 'PPI', 'CPI', 'CLAIMS', or None
    """
    parts = []
    if date_str in NFP_RELEASE_DATES:
        parts.append('NFP')
    if date_str in CPI_RELEASE_DATES:
        parts.append('CPI')
    if date_str in PPI_RELEASE_DATES:
        parts.append('PPI')
    if is_claims_release_day(date_str):
        parts.append('CLAIMS')
    if not parts:
        return None
    if len(parts) == 1:
        return parts[0]
    return '+'.join(parts)


def classify_market_regime(config=None):
    """Classify current market regime for fade/continuation bias.

    Uses live BLS data from cache if available, falls back to config.
    Returns:
        regime: str
        fade_rate: float (0.0-1.0)
    """
    cfg = config or CONFIG

    # Try to load live BLS data from cache
    ppi_yoy = None
    ppi_prev = None
    cpi_yoy = None
    fed = cfg.get('M22_FED_STANCE', 'HOLDING')

    try:
        import json as _json
        import os as _os
        cache_path = _os.path.join(_os.path.dirname(_os.path.dirname(_os.path.dirname(__file__))),
                                    'data', 'macro_data.json')
        if _os.path.exists(cache_path):
            with open(cache_path) as _f:
                cache = _json.load(_f)
            latest = cache.get('latest', {})
            yoy_data = cache.get('yoy', {})
            ppi_latest = latest.get('PPI_FD', {})
            ppi_date = ppi_latest.get('date', '')
            ppi_yoy = ppi_latest.get('yoy')
            cpi_latest = latest.get('CPI_ALL', {})
            cpi_yoy = cpi_latest.get('yoy')
            if ppi_date:
                y, m = int(ppi_date[:4]), int(ppi_date[5:7])
                prev_m = m - 1 if m > 1 else 12
                prev_y = y if m > 1 else y - 1
                prev_key = f"{prev_y:04d}-{prev_m:02d}"
                ppi_prev = yoy_data.get('PPI_FD', {}).get(prev_key)
    except Exception:
        pass

    # Fall back to config values
    if ppi_yoy is None:
        ppi_yoy = cfg.get('M22_PPI_YOY', None)
    if ppi_prev is None:
        ppi_prev = cfg.get('M22_PPI_PREV_YOY', None)
    if cpi_yoy is None:
        cpi_yoy = cfg.get('M22_CPI_YOY', None)

    if ppi_yoy is None:
        return 'UNKNOWN', 0.33

    # Stagflation: PPI ≥3.0% + Fed HOLDING
    if ppi_yoy >= 3.0 and fed == 'HOLDING':
        if ppi_yoy >= 4.0:
            return 'STAGFLATION_HOT', REGIME_FADE_RATES.get('STAGFLATION_HOT', 0.50)
        return 'STAGFLATION', REGIME_FADE_RATES.get('STAGFLATION', 0.46)

    # Acceleration: PPI rising + Fed CUTTING/HOLDING
    if ppi_prev is not None and ppi_yoy > ppi_prev:
        return 'ACCELERATION', REGIME_FADE_RATES.get('ACCELERATION', 0.29)

    # Recovery: PPI falling from high levels
    if ppi_prev is not None and ppi_yoy < ppi_prev:
        return 'RECOVERY', REGIME_FADE_RATES.get('RECOVERY', 0.29)

    return 'ACCELERATION', 0.29


def _classify_dump_size(us_move_pct):
    """Classify US session dump size for reversal probability.

    Returns: 'SMALL', 'MEDIUM', 'BIG', 'CRASH', or 'NOT_DUMP'
    """
    if us_move_pct >= -0.5:
        return 'NOT_DUMP'
    elif us_move_pct >= -1.5:
        return 'SMALL'
    elif us_move_pct >= -2.5:
        return 'MEDIUM'
    elif us_move_pct >= -4.0:
        return 'BIG'
    else:
        return 'CRASH'


def _classify_inflation_regime(regime):
    """Map M23 regime to inflation regime for reversal stats.

    Returns: 'INFLATION_RISING', 'INFLATION_PEAKING', 'DEFLATION', 'NEUTRAL'
    """
    if regime in ('STAGFLATION', 'STAGFLATION_HOT', 'ACCELERATION'):
        return 'INFLATION_RISING'
    elif regime in ('TIGHTENING',):
        return 'INFLATION_PEAKING'
    elif regime in ('CRISIS_RECOVERY', 'RECOVERY', 'EASING'):
        return 'DEFLATION'
    else:
        return 'NEUTRAL'


def _detect_crash(gap_dir, asia_range_pct, is_sweep_reversal):
    """Detect if Asia session was a genuine crash (not a sweep reversal).

    Crashes: gap flat/no direction, massive range (>7%), no sweep pattern.
    Examples: Jun 2022 PPI (-7.98% Asia), Jun 2025 PPI (-4.28% Asia).
    """
    if gap_dir == 'FLAT' and asia_range_pct >= CRASH_ASIA_RANGE_MIN and not is_sweep_reversal:
        return True
    if asia_range_pct >= 10.0:  # extreme range regardless of gap
        return True
    return False


def _compute_reversal_probability(us_move, regime, dump_size, release_type):
    """Compute the probability that Asia will reverse a US dump.

    Uses 8 years of historical data (2019-2026) across three dimensions:
    1. Year-over-year trend (pattern getting stronger)
    2. Dump size (small dumps are noise, big dumps are signal)
    3. Inflation regime (rising = buy, peaking = run)

    Returns: (probability: float, confidence: str, factors: list)
    """
    factors = []
    probs = []

    # 1. Year-based reversal rate
    year_prob = PPI_REVERSAL_RATE_BY_YEAR.get(2026, 0.60)  # default to latest
    probs.append(year_prob)
    factors.append(f'year 2026: {year_prob:.0%} reversal rate')

    # 2. Dump size
    size_prob = PPI_REVERSAL_BY_DUMP_SIZE.get(dump_size, 0.50)
    probs.append(size_prob)
    factors.append(f'dump {dump_size}: {size_prob:.0%} reversal rate')

    # 3. Inflation regime
    infl_regime = _classify_inflation_regime(regime)
    regime_prob = PPI_REVERSAL_BY_REGIME.get(infl_regime, 0.50)
    probs.append(regime_prob)
    factors.append(f'regime {infl_regime}: {regime_prob:.0%} reversal rate')

    # Weighted average (dump size matters most for single-event prediction)
    # Year trend = 20%, dump size = 45%, regime = 35%
    combined = year_prob * 0.20 + size_prob * 0.45 + regime_prob * 0.35

    # CPI discount: CPI is noisier, reduce confidence
    if release_type == 'CPI':
        combined = combined * 0.85 + 0.15 * 0.50  # pull toward 50%
        factors.append(f'CPI discount: pulled toward 50%')

    # Confidence based on alignment
    aligned = sum(1 for p in probs if p >= 0.60)
    if aligned >= 3:
        confidence = 'HIGH'
    elif aligned >= 2:
        confidence = 'MEDIUM'
    else:
        confidence = 'LOW'

    return round(combined, 3), confidence, factors


def is_ppi_release_day(date_str=None):
    """Check if today (or given date) is a PPI release day."""
    if date_str is None:
        date_str = datetime.utcnow().strftime('%Y-%m-%d')
    return date_str in PPI_RELEASE_DATES


def is_cpi_release_day(date_str=None):
    """Check if today (or given date) is a CPI release day."""
    if date_str is None:
        date_str = datetime.utcnow().strftime('%Y-%m-%d')
    return date_str in CPI_RELEASE_DATES


def is_claims_release_day(date_str=None):
    """Check if today (or given date) is a jobless claims release day.

    Claims are released every Thursday at 8:30 AM ET.
    Exception: if Thursday is a federal holiday, released Friday instead.
    """
    if date_str is None:
        date_str = datetime.utcnow().strftime('%Y-%m-%d')
    dt = datetime.strptime(date_str, '%Y-%m-%d')
    # Thursday = 3
    if dt.weekday() == 3:
        return True
    # Friday after a Thursday holiday (check if yesterday was Thursday)
    if dt.weekday() == 4:
        yesterday = (dt - timedelta(days=1)).strftime('%Y-%m-%d')
        # If yesterday was a known PPI/CPI release (which are weekdays), skip
        # This is a simple heuristic — claims shift to Friday on holiday weeks
        return True  # Be inclusive — will filter by actual data availability
    return False


# ═══════════════════════════════════════════════════════════════
# TIME DECAY — macro data loses relevance between releases
# ═══════════════════════════════════════════════════════════════
# PPI/CPI impact decays over the week following release.
# Claims are weekly — decay over 7 days.
#
# PPI/CPI decay (days since release → multiplier):
#   0-1:   1.00  (release day + next day — full impact)
#   2-3:   0.70  (market digesting)
#   4-7:   0.40  (fading)
#   8-14:  0.20  (stale — next release approaching)
#   15+:   0.10  (floor)
#
# Claims decay (days since Thursday release → multiplier):
#   0-1:   1.00  (Thu-Fri)
#   2-3:   0.60  (Sat-Sun)
#   4-6:   0.30  (Mon-Wed)
#   7+:    0.10  (stale, new claims Thursday)

def _compute_m23_release_decay(release_date_str, today_str=None):
    """Compute decay multiplier for PPI/CPI post-release scoring.

    Args:
        release_date_str: 'YYYY-MM-DD' of the release
        today_str: 'YYYY-MM-DD' of today (default: now UTC)

    Returns:
        (multiplier: float, days_since: int)
    """
    if today_str is None:
        today_str = datetime.utcnow().strftime('%Y-%m-%d')

    release_dt = datetime.strptime(release_date_str, '%Y-%m-%d')
    today_dt = datetime.strptime(today_str, '%Y-%m-%d')
    days = (today_dt - release_dt).days

    if days <= 1:
        mult = 1.00
    elif days <= 3:
        mult = 0.70
    elif days <= 7:
        mult = 0.40
    elif days <= 14:
        mult = 0.20
    else:
        mult = 0.10

    return mult, days


def _compute_claims_decay(today_str=None):
    """Compute decay multiplier based on days since last Thursday (claims release).

    Returns:
        (multiplier: float, days_since_thursday: int)
    """
    if today_str is None:
        today_str = datetime.utcnow().strftime('%Y-%m-%d')

    today = datetime.strptime(today_str, '%Y-%m-%d')
    # Find last Thursday (weekday=3)
    days_since_thu = (today.weekday() - 3) % 7
    if days_since_thu == 0 and today.weekday() == 3:
        days_since_thu = 0  # today is Thursday
    elif days_since_thu == 0 and today.weekday() != 3:
        days_since_thu = 7  # last Thursday was a week ago

    if days_since_thu <= 1:
        mult = 1.00
    elif days_since_thu <= 3:
        mult = 0.60
    elif days_since_thu <= 6:
        mult = 0.30
    else:
        mult = 0.10

    return mult, days_since_thu


def is_macro_release_day(date_str=None):
    """Check if today is any macro data release day (PPI, CPI, or Claims)."""
    if date_str is None:
        date_str = datetime.utcnow().strftime('%Y-%m-%d')
    return (date_str in PPI_RELEASE_DATES or
            date_str in CPI_RELEASE_DATES or
            is_claims_release_day(date_str))


def get_claims_trend(config=None):
    """Get current jobless claims trend from monthly averages.

    Returns:
        dict with:
            current: latest claims avg (K)
            prev_month: previous month avg (K)
            trend: 'RISING', 'FALLING', 'STABLE'
            trend_pct: % change from previous month
            classification: 'LOW', 'NORMAL', 'ELEVATED', 'SPIKE', 'CRISIS'
            sahm_triggered: bool (unemployment 3-month avg rise >= 0.5pp)
            unemployment: current unemployment rate
    """
    cfg = config or CONFIG

    # Get latest month
    sorted_months = sorted(JOBLESS_CLAIMS_MONTHLY_AVG.keys())
    if not sorted_months:
        return None

    latest_key = sorted_months[-1]
    current_claims = JOBLESS_CLAIMS_MONTHLY_AVG[latest_key]

    # Previous month
    prev_key = sorted_months[-2] if len(sorted_months) > 1 else None
    prev_claims = JOBLESS_CLAIMS_MONTHLY_AVG.get(prev_key, current_claims) if prev_key else current_claims

    # 3-month trend
    if len(sorted_months) >= 3:
        m3_key = sorted_months[-3]
        m3_claims = JOBLESS_CLAIMS_MONTHLY_AVG[m3_key]
        trend_3m = current_claims - m3_claims
    else:
        trend_3m = current_claims - prev_claims

    # Trend classification
    if trend_3m >= CLAIMS_TREND_RISING_THRESHOLD:
        trend = 'RISING'
    elif trend_3m <= CLAIMS_TREND_FALLING_THRESHOLD:
        trend = 'FALLING'
    else:
        trend = 'STABLE'

    trend_pct = ((current_claims - prev_claims) / prev_claims * 100) if prev_claims > 0 else 0

    # Claims classification
    if current_claims >= CLAIMS_CRISIS_THRESHOLD:
        classification = 'CRISIS'
    elif current_claims >= CLAIMS_SPIKE_THRESHOLD:
        classification = 'SPIKE'
    elif current_claims >= CLAIMS_ELEVATED_THRESHOLD:
        classification = 'ELEVATED'
    elif current_claims >= CLAIMS_LOW_THRESHOLD:
        classification = 'NORMAL'
    else:
        classification = 'LOW'

    # Sahm Rule check: 3-month avg unemployment rise >= 0.5pp from 12-month low
    unemp_sorted = sorted(UNEMPLOYMENT_RATE_MONTHLY.keys())
    sahm_triggered = False
    current_unemp = None
    if len(unemp_sorted) >= 12:
        current_unemp = UNEMPLOYMENT_RATE_MONTHLY[unemp_sorted[-1]]
        # 3-month average
        last_3 = [UNEMPLOYMENT_RATE_MONTHLY[k] for k in unemp_sorted[-3:]]
        avg_3m = sum(last_3) / 3
        # 12-month low
        last_12 = [UNEMPLOYMENT_RATE_MONTHLY[k] for k in unemp_sorted[-12:]]
        low_12m = min(last_12)
        # Sahm Rule: 3-month avg - 12-month low >= 0.5
        if avg_3m - low_12m >= 0.5:
            sahm_triggered = True

    return {
        'current': current_claims,
        'prev_month': prev_claims,
        'trend': trend,
        'trend_pct': round(trend_pct, 1),
        'trend_3m': trend_3m,
        'classification': classification,
        'sahm_triggered': sahm_triggered,
        'unemployment': current_unemp,
        'latest_month': latest_key,
    }


def classify_macro_combo(cpi_yoy, ppi_yoy, claims_classification):
    """Classify the CPI→PPI→Claims cascade and predict ETH impact.

    CASCADE MODEL (replaces old independent scoring):
    1. CPI surprise → PRIMARY signal (biggest move, sets tone)
    2. PPI vs CPI alignment → CONFIRMATION/DENIAL (PPI that confirms CPI = stronger)
    3. Claims → BACKGROUND context (only extremes matter)

    Args:
        cpi_yoy: Current CPI year-over-year %
        ppi_yoy: Current PPI year-over-year %
        claims_classification: 'LOW', 'NORMAL', 'ELEVATED', 'SPIKE', 'CRISIS'

    Returns:
        dict with cascade classification, expected impact, and Fed response
    """
    # ── Step 1: Classify CPI surprise ──
    if cpi_yoy is not None:
        if cpi_yoy >= 3.5:
            cpi_class = 'CPI_HOT'
            cpi_surprise = 'HOT'
        elif cpi_yoy >= 2.5:
            cpi_class = 'CPI_WARM'
            cpi_surprise = 'WARM'
        else:
            cpi_class = 'CPI_COOL'
            cpi_surprise = 'COOL'
    else:
        cpi_class = 'CPI_UNKNOWN'
        cpi_surprise = 'WARM'  # default to neutral

    # ── Step 2: Classify PPI surprise ──
    if ppi_yoy is not None:
        if ppi_yoy >= 3.5:
            ppi_class = 'PPI_HOT'
            ppi_surprise = 'HOT'
        elif ppi_yoy >= 2.5:
            ppi_class = 'PPI_WARM'
            ppi_surprise = 'WARM'
        else:
            ppi_class = 'PPI_COOL'
            ppi_surprise = 'COOL'
    else:
        ppi_class = 'PPI_UNKNOWN'
        ppi_surprise = 'WARM'

    # ── Step 3: Claims bucket ──
    if claims_classification in ('LOW',):
        claims_bucket = 'CLAIMS_LOW'
    elif claims_classification in ('NORMAL',):
        claims_bucket = 'CLAIMS_NORMAL'
    else:
        claims_bucket = 'CLAIMS_ELEVATED'

    # ── CASCADE SCORING ──
    # 1. CPI primary signal
    cpi_base = CPI_PRIMARY.get(cpi_surprise, (0.0, 'LOW', 'HOLD'))
    base_move, base_conf, fed_bias = cpi_base

    # 2. PPI confirmation modifier
    ppi_key = (cpi_surprise, ppi_surprise)
    ppi_mod = PPI_CONFIRMATION.get(ppi_key, (0.0, 0.0, 'No PPI data'))
    ppi_move_mod, ppi_conf_mod, ppi_desc = ppi_mod

    # 3. Claims modifier
    claims_mod = CLAIMS_MODIFIER.get(claims_classification, (0.0, 'Unknown'))
    claims_move_mod, claims_desc = claims_mod

    # Combined expected move
    total_move = base_move + ppi_move_mod + claims_move_mod

    # Confidence: start from CPI base, adjust for PPI alignment
    conf_map = {'HIGH': 0.85, 'MEDIUM': 0.65, 'LOW': 0.45}
    conf_val = conf_map.get(base_conf, 0.50) + ppi_conf_mod
    conf_val = max(0.30, min(0.95, conf_val))
    if conf_val >= 0.80:
        confidence = 'HIGH'
    elif conf_val >= 0.60:
        confidence = 'MEDIUM'
    else:
        confidence = 'LOW'

    # Fed response from CPI (primary)
    fed_explanation = FED_RESPONSE.get(fed_bias, 'Unknown')

    # Signal classification
    if total_move >= 3.0:
        signal = 'STRONG_BUY'
    elif total_move >= 1.0:
        signal = 'BUY'
    elif total_move >= -1.0:
        signal = 'HOLD'
    elif total_move >= -3.0:
        signal = 'SELL'
    else:
        signal = 'STRONG_SELL'

    # PPI leads CPI by 2-3 months — check if PPI is accelerating
    ppi_leading = False
    if ppi_yoy is not None and cpi_yoy is not None:
        ppi_gap = ppi_yoy - cpi_yoy
        if ppi_gap > 1.0:
            ppi_leading = True  # PPI >> CPI → CPI will follow up

    return {
        'combo_key': (cpi_class, ppi_class, claims_bucket),
        'cpi_class': cpi_class,
        'ppi_class': ppi_class,
        'claims_bucket': claims_bucket,
        'expected_eth_move': round(total_move, 1),
        'confidence': confidence,
        'fed_action': fed_bias,
        'fed_explanation': fed_explanation,
        'signal': signal,
        'ppi_leading_cpi': ppi_leading,
        'ppi_cpi_gap': round(ppi_yoy - cpi_yoy, 1) if (ppi_yoy and cpi_yoy) else None,
        # Cascade components
        'cascade': {
            'cpi_signal': cpi_surprise,
            'cpi_base_move': base_move,
            'ppi_confirmation': ppi_surprise,
            'ppi_modifier': ppi_move_mod,
            'ppi_description': ppi_desc,
            'claims_modifier': claims_move_mod,
            'claims_description': claims_desc,
            'total_move': round(total_move, 2),
        },
    }


def get_claims_context_for_release(release_type, config=None):
    """Get jobless claims context for a PPI or CPI release day.

    When PPI/CPI releases on a Thursday, it often coincides with jobless claims.
    When it doesn't, we still use the latest claims data for macro context.

    Returns:
        dict with claims context and combo analysis
    """
    cfg = config or CONFIG

    claims = get_claims_trend(cfg)
    if claims is None:
        return None

    # Get CPI and PPI from config (or cache)
    ppi_yoy = cfg.get('M22_PPI_YOY')
    cpi_yoy = cfg.get('M22_CPI_YOY')

    # Try cache first
    try:
        import json as _json
        import os as _os
        cache_path = _os.path.join(_os.path.dirname(_os.path.dirname(_os.path.dirname(__file__))),
                                    'data', 'macro_data.json')
        if _os.path.exists(cache_path):
            with open(cache_path) as _f:
                cache = _json.load(_f)
            latest = cache.get('latest', {})
            if latest.get('PPI_FD', {}).get('yoy') is not None:
                ppi_yoy = latest['PPI_FD']['yoy']
            if latest.get('CPI_ALL', {}).get('yoy') is not None:
                cpi_yoy = latest['CPI_ALL']['yoy']
    except Exception:
        pass

    combo = classify_macro_combo(cpi_yoy, ppi_yoy, claims['classification'])

    # Is today also a claims release day?
    today_str = datetime.utcnow().strftime('%Y-%m-%d')
    claims_today = is_claims_release_day(today_str)

    # Compute data age (days since last CPI/PPI release)
    cpi_days_ago = None
    ppi_days_ago = None
    try:
        today_dt = datetime.utcnow()
        if CPI_RELEASE_DATES:
            latest_cpi = max(CPI_RELEASE_DATES)
            cpi_dt = datetime.strptime(latest_cpi, '%Y-%m-%d')
            cpi_days_ago = (today_dt - cpi_dt).days
        if PPI_RELEASE_DATES:
            latest_ppi = max(PPI_RELEASE_DATES)
            ppi_dt = datetime.strptime(latest_ppi, '%Y-%m-%d')
            ppi_days_ago = (today_dt - ppi_dt).days
    except Exception:
        pass

    return {
        'claims': claims,
        'combo': combo,
        'claims_today': claims_today,
        'cpi_yoy': cpi_yoy,
        'ppi_yoy': ppi_yoy,
        'cpi_days_ago': cpi_days_ago,
        'ppi_days_ago': ppi_days_ago,
        'release_type': release_type,
    }


def format_claims_context(ctx):
    """Format jobless claims context for terminal output."""
    if ctx is None:
        return ''

    lines = []
    claims = ctx.get('claims', {})
    combo = ctx.get('combo', {})

    current = claims.get('current', 0)
    trend = claims.get('trend', '?')
    classification = claims.get('classification', '?')
    unemp = claims.get('unemployment')
    sahm = claims.get('sahm_triggered', False)

    # Classification icon
    cls_icons = {
        'LOW': '🟢', 'NORMAL': '⚪', 'ELEVATED': '🟡',
        'SPIKE': '🟠', 'CRISIS': '🔴'
    }
    cls_icon = cls_icons.get(classification, '⚪')

    # Trend icon
    trend_icons = {'RISING': '📈', 'FALLING': '📉', 'STABLE': '➡️'}
    trend_icon = trend_icons.get(trend, '➡️')

    lines.append(f"\n  📋 JOBLESS CLAIMS CONTEXT:")
    lines.append(f"    Claims: {cls_icon} {current}K ({classification})  "
                 f"{trend_icon} {trend} ({claims.get('trend_pct', 0):+.1f}%)")
    if unemp:
        sahm_icon = '🔴 TRIGGERED' if sahm else '🟢 ok'
        lines.append(f"    Unemployment: {unemp}%  Sahm Rule: {sahm_icon}")
    if claims.get('trend_3m'):
        lines.append(f"    3-month Δ: {claims['trend_3m']:+.0f}K")

    # Combo analysis (cascade model)
    if combo:
        signal = combo.get('signal', '?')
        expected = combo.get('expected_eth_move', 0)
        fed = combo.get('fed_action', '?')
        conf = combo.get('confidence', '?')
        ppi_gap = combo.get('ppi_cpi_gap')
        cascade = combo.get('cascade', {})

        sig_icons = {
            'STRONG_BUY': '🟢🟢', 'BUY': '🟢', 'HOLD': '⚪',
            'SELL': '🔴', 'STRONG_SELL': '🔴🔴'
        }
        sig_icon = sig_icons.get(signal, '⚪')

        lines.append(f"\n    Macro Cascade: {sig_icon} {signal}")
        if cascade:
            # Show cascade breakdown with data age context
            cpi_sig = cascade.get('cpi_signal', '?')
            cpi_base = cascade.get('cpi_base_move', 0)
            ppi_conf = cascade.get('ppi_confirmation', '?')
            ppi_mod = cascade.get('ppi_modifier', 0)
            ppi_desc = cascade.get('ppi_description', '')
            claims_mod = cascade.get('claims_modifier', 0)
            claims_desc = cascade.get('claims_description', '')

            # Show age of CPI/PPI data (these are stale on claims-only days)
            cpi_age = ctx.get('cpi_days_ago')
            ppi_age = ctx.get('ppi_days_ago')
            cpi_age_str = f" ({cpi_age}d ago)" if cpi_age else ""
            ppi_age_str = f" ({ppi_age}d ago)" if ppi_age else ""

            lines.append(f"      1. CPI {cpi_sig}: {cpi_base:+.2f}% (primary){cpi_age_str}")
            lines.append(f"      2. PPI {ppi_conf}: {ppi_mod:+.2f}% — {ppi_desc}{ppi_age_str}")
            lines.append(f"      3. Claims: {claims_mod:+.2f}% — {claims_desc}")
            lines.append(f"      → Total: {expected:+.1f}%  Conf: {conf}")
        else:
            lines.append(f"    {combo.get('cpi_class', '?')} × {combo.get('ppi_class', '?')} × {combo.get('claims_bucket', '?')}")
            lines.append(f"    Expected ETH: {expected:+.1f}%  Confidence: {conf}")
        lines.append(f"    Fed: {fed} — {combo.get('fed_explanation', '')}")

        if ppi_gap is not None and ppi_gap > 1.0:
            lines.append(f"    ⚠️ PPI leads CPI by {ppi_gap:.1f}pp — CPI will follow UP (2-3 month lag)")

        if sahm:
            lines.append(f"    🚨 SAHM RULE TRIGGERED — recession indicator active")

    return '\n'.join(lines)


def _get_bars_before(df, dt, n=1):
    """Get last n bars before a datetime."""
    if isinstance(df['Open time'].iloc[0], str):
        mask = pd.to_datetime(df['Open time']) < dt
    else:
        mask = df['Open time'] < dt
    bars = df[mask]
    if len(bars) == 0:
        return None
    return bars.iloc[-n] if n == 1 else bars.tail(n)


def _get_bars_between(df, start, end):
    """Get bars between two datetimes."""
    if isinstance(df['Open time'].iloc[0], str):
        mask = (pd.to_datetime(df['Open time']) >= start) & \
               (pd.to_datetime(df['Open time']) < end)
    else:
        mask = (df['Open time'] >= start) & (df['Open time'] < end)
    return df[mask]


def compute_1h_spike(df_15m, release_date):
    """Compute the 1h post-release spike."""
    if isinstance(release_date, str):
        release_date = datetime.strptime(release_date, '%Y-%m-%d')

    release_dt = release_date.replace(hour=RELEASE_HOUR_UTC, minute=RELEASE_MINUTE_UTC)
    us_end = release_date.replace(hour=US_SESSION_END[0])

    pre_bar = _get_bars_before(df_15m, release_dt)
    if pre_bar is None:
        return None
    pre_price = float(pre_bar['Close'])

    first_1h = _get_bars_between(df_15m, release_dt, release_dt + timedelta(hours=1))
    if len(first_1h) < 2:
        return None

    spike_close = float(first_1h.iloc[-1]['Close'])
    spike_high = float(first_1h['High'].max())
    spike_low = float(first_1h['Low'].min())
    spike_pct = (spike_close - pre_price) / pre_price * 100
    spike_range = (spike_high - spike_low) / pre_price * 100

    return {
        'spike_pct': round(spike_pct, 3),
        'spike_dir': 'UP' if spike_pct > 0.3 else 'DOWN' if spike_pct < -0.3 else 'FLAT',
        'spike_range': round(spike_range, 3),
        'pre_price': round(pre_price, 2),
        'spike_close': round(spike_close, 2),
    }


def compute_us_session(df_15m, release_date):
    """Compute US session move on release day."""
    if isinstance(release_date, str):
        release_date = datetime.strptime(release_date, '%Y-%m-%d')

    release_dt = release_date.replace(hour=RELEASE_HOUR_UTC, minute=RELEASE_MINUTE_UTC)
    us_end = release_date.replace(hour=US_SESSION_END[0])

    pre_bar = _get_bars_before(df_15m, release_dt)
    if pre_bar is None:
        return None
    pre_price = float(pre_bar['Close'])

    us_bars = _get_bars_between(df_15m, release_dt, us_end)
    if len(us_bars) < 2:
        return None

    us_close = float(us_bars.iloc[-1]['Close'])
    us_high = float(us_bars['High'].max())
    us_low = float(us_bars['Low'].min())
    us_move = (us_close - pre_price) / pre_price * 100
    us_range = (us_high - us_low) / pre_price * 100

    return {
        'us_move': round(us_move, 3),
        'us_close': round(us_close, 2),
        'us_high': round(us_high, 2),
        'us_low': round(us_low, 2),
        'us_range': round(us_range, 3),
        'us_max_up': round((us_high - pre_price) / pre_price * 100, 3),
        'us_max_down': round((us_low - pre_price) / pre_price * 100, 3),
        'pre_price': round(pre_price, 2),
    }


def compute_asia_session(df_15m, release_date):
    """Compute Asia session on the day after release."""
    if isinstance(release_date, str):
        release_date = datetime.strptime(release_date, '%Y-%m-%d')

    release_dt = release_date.replace(hour=RELEASE_HOUR_UTC, minute=RELEASE_MINUTE_UTC)
    us_end = release_date.replace(hour=US_SESSION_END[0])
    asia_start = (release_date + timedelta(days=1)).replace(hour=ASIA_SESSION_START[0])
    asia_end = (release_date + timedelta(days=1)).replace(hour=ASIA_SESSION_END[0])

    # US close as reference
    us_bars = _get_bars_between(df_15m, release_dt, us_end)
    if len(us_bars) == 0:
        return None
    us_close = float(us_bars.iloc[-1]['Close'])

    # Asia session
    asia_bars = _get_bars_between(df_15m, asia_start, asia_end)
    if len(asia_bars) < 2:
        return None

    asia_open = float(asia_bars.iloc[0]['Open'])
    asia_close = float(asia_bars.iloc[-1]['Close'])
    asia_high = float(asia_bars['High'].max())
    asia_low = float(asia_bars['Low'].min())
    asia_move = (asia_close - us_close) / us_close * 100
    asia_gap = (asia_open - us_close) / us_close * 100
    asia_range = (asia_high - asia_low) / us_close * 100

    # ── Intra-session path analysis ──
    # Track the price path to detect sweep-and-reverse patterns
    # that open→close analysis misses.
    gap_dir = 'UP' if asia_gap > 0.2 else 'DOWN' if asia_gap < -0.2 else 'FLAT'

    # Max extension against gap direction (sweep depth)
    if gap_dir == 'DOWN':
        # Gap down: how far did price extend below the gap?
        sweep_low_pct = (asia_low - us_close) / us_close * 100
        # Recovery from low: how much did it bounce back?
        recovery_pct = (asia_close - asia_low) / (us_close - asia_low) * 100 if (us_close - asia_low) > 0 else 0
        # Did price reclaim the gap (trade back above US close)?
        reclaimed_gap = asia_high >= us_close
        # Sweep-and-reverse: price went significantly lower then recovered most of it
        sweep_depth_pct = abs(sweep_low_pct)
        is_sweep_reversal = sweep_depth_pct > 0.5 and recovery_pct > 50
    elif gap_dir == 'UP':
        # Gap up: how far did price extend above the gap?
        sweep_high_pct = (asia_high - us_close) / us_close * 100
        recovery_pct = (asia_high - asia_close) / (asia_high - us_close) * 100 if (asia_high - us_close) > 0 else 0
        reclaimed_gap = asia_low <= us_close
        sweep_depth_pct = abs(sweep_high_pct)
        is_sweep_reversal = sweep_depth_pct > 0.5 and recovery_pct > 50
    else:
        sweep_depth_pct = 0
        recovery_pct = 0
        reclaimed_gap = False
        is_sweep_reversal = False

    return {
        'asia_move': round(asia_move, 3),
        'asia_gap': round(asia_gap, 3),
        'asia_open': round(asia_open, 2),
        'asia_close': round(asia_close, 2),
        'asia_high': round(asia_high, 2),
        'asia_low': round(asia_low, 2),
        'asia_range': round(asia_range, 3),
        'us_close_ref': round(us_close, 2),
        # Path analysis
        'gap_dir': gap_dir,
        'sweep_depth_pct': round(sweep_depth_pct, 3),
        'recovery_pct': round(recovery_pct, 1),
        'reclaimed_gap': reclaimed_gap,
        'is_sweep_reversal': is_sweep_reversal,
    }


def compute_uk_session(df_15m, release_date):
    """Compute UK (London) session on the day after release.

    London opens 07:00 UTC, closes 16:00 UTC. By this time, London has
    full visibility of:
      1. US release reaction (13:30-21:00 UTC previous day)
      2. Asia overnight response (00:00-08:00 UTC same morning)

    This makes UK session the "informed decision maker" — it can either
    continue Asia's direction or fade it based on whether the Asia move
    was genuine or a sweep-reversal.

    Args:
        df_15m: DataFrame with 15m OHLCV data
        release_date: str 'YYYY-MM-DD' or datetime of the release day

    Returns:
        dict with UK session data, or None if data unavailable
    """
    if isinstance(release_date, str):
        release_date = datetime.strptime(release_date, '%Y-%m-%d')

    release_dt = release_date.replace(hour=RELEASE_HOUR_UTC, minute=RELEASE_MINUTE_UTC)
    us_end = release_date.replace(hour=US_SESSION_END[0])

    # UK session is the NEXT day (same day as Asia, but later)
    uk_date = release_date + timedelta(days=1)
    uk_start = uk_date.replace(hour=UK_SESSION_START[0])
    uk_end = uk_date.replace(hour=UK_SESSION_END[0])

    # Asia session (same morning, before London)
    asia_start = uk_date.replace(hour=ASIA_SESSION_START[0])
    asia_end = uk_date.replace(hour=ASIA_SESSION_END[0])

    # US close as reference (from release day)
    us_bars = _get_bars_between(df_15m, release_dt, us_end)
    if len(us_bars) == 0:
        return None
    us_close = float(us_bars.iloc[-1]['Close'])

    # Asia session data (for reference)
    asia_bars = _get_bars_between(df_15m, asia_start, asia_end)
    if len(asia_bars) < 2:
        return None
    asia_close = float(asia_bars.iloc[-1]['Close'])
    asia_high = float(asia_bars['High'].max())
    asia_low = float(asia_bars['Low'].min())

    # UK session
    uk_bars = _get_bars_between(df_15m, uk_start, uk_end)
    if len(uk_bars) < 2:
        return None

    uk_open = float(uk_bars.iloc[0]['Open'])
    uk_close = float(uk_bars.iloc[-1]['Close'])
    uk_high = float(uk_bars['High'].max())
    uk_low = float(uk_bars['Low'].min())

    # UK move vs Asia close (London's starting reference)
    uk_move_vs_asia = (uk_close - asia_close) / asia_close * 100
    uk_move_vs_us = (uk_close - us_close) / us_close * 100
    uk_range = (uk_high - uk_low) / asia_close * 100

    # UK gap from Asia close (London open vs Asia close)
    uk_gap = (uk_open - asia_close) / asia_close * 100

    # Asia direction for context
    asia_move = (asia_close - us_close) / us_close * 100
    asia_dir = 'UP' if asia_move > 0.3 else 'DOWN' if asia_move < -0.3 else 'FLAT'
    uk_dir = 'UP' if uk_move_vs_asia > 0.3 else 'DOWN' if uk_move_vs_asia < -0.3 else 'FLAT'

    # Did UK continue Asia's direction?
    uk_continued = (asia_dir == uk_dir) and asia_dir != 'FLAT'
    uk_faded = (asia_dir == 'UP' and uk_dir == 'DOWN') or \
               (asia_dir == 'DOWN' and uk_dir == 'UP')

    # Intra-UK path: did London sweep Asia's high/low first?
    uk_swept_asia_high = uk_high >= asia_high * 0.999  # within 0.1%
    uk_swept_asia_low = uk_low <= asia_low * 1.001

    # If Asia was a sweep-reversal, did UK follow through or fade?
    # (We'll need Asia's sweep data passed in, but compute basic version here)
    is_morning_sweep = False
    sweep_recovery = 0.0
    if uk_swept_asia_high and uk_dir == 'DOWN':
        # Swept Asia high then reversed — classic London fade
        is_morning_sweep = True
        sweep_recovery = (uk_high - uk_close) / (uk_high - asia_close) * 100 if (uk_high - asia_close) > 0 else 0
    elif uk_swept_asia_low and uk_dir == 'UP':
        # Swept Asia low then bounced — classic London reversal
        is_morning_sweep = True
        sweep_recovery = (uk_close - uk_low) / (asia_close - uk_low) * 100 if (asia_close - uk_low) > 0 else 0

    # Volume comparison: UK vs Asia (session avg bar volume)
    uk_avg_vol = float(uk_bars['Volume'].mean()) if len(uk_bars) > 0 else 0
    asia_avg_vol = float(asia_bars['Volume'].mean()) if len(asia_bars) > 0 else 0
    vol_ratio = uk_avg_vol / asia_avg_vol if asia_avg_vol > 0 else 1.0

    # Taker flow during UK session
    uk_taker = 0.5
    if 'Taker buy base asset volume' in uk_bars.columns:
        taker_buy = float(uk_bars['Taker buy base asset volume'].sum())
        total_vol = float(uk_bars['Volume'].sum())
        uk_taker = taker_buy / total_vol if total_vol > 0 else 0.5

    return {
        'uk_move_vs_asia': round(uk_move_vs_asia, 3),
        'uk_move_vs_us': round(uk_move_vs_us, 3),
        'uk_open': round(uk_open, 2),
        'uk_close': round(uk_close, 2),
        'uk_high': round(uk_high, 2),
        'uk_low': round(uk_low, 2),
        'uk_range': round(uk_range, 3),
        'uk_gap': round(uk_gap, 3),
        'uk_direction': uk_dir,
        'uk_taker': round(uk_taker, 4),
        'uk_vol_ratio_vs_asia': round(vol_ratio, 2),
        # Context from Asia
        'asia_close_ref': round(asia_close, 2),
        'asia_direction': asia_dir,
        'asia_move': round(asia_move, 3),
        # UK behavior
        'uk_continued_asia': uk_continued,
        'uk_faded_asia': uk_faded,
        'uk_swept_asia_high': uk_swept_asia_high,
        'uk_swept_asia_low': uk_swept_asia_low,
        'is_morning_sweep': is_morning_sweep,
        'sweep_recovery_pct': round(sweep_recovery, 1),
    }


def _predict_uk_session(us_dir, asia_data, regime, release_type):
    """Predict UK session behavior based on US + Asia context.

    London has full visibility of both US reaction and Asia overnight.
    The prediction depends on:
      1. What Asia did (continuation vs fade vs sweep-reversal)
      2. Whether Asia's move was genuine or a sweep
      3. The inflation regime (stagflation = more fading)

    Args:
        us_dir: 'DUMP', 'RALLY', 'FLAT'
        asia_data: dict from compute_asia_session (or None if pre-Asia)
        regime: inflation regime string
        release_type: 'PPI', 'CPI', 'BOTH'

    Returns:
        dict with UK prediction
    """
    if asia_data is None:
        return {
            'prediction': 'UNKNOWN',
            'confidence': 'LOW',
            'reason': 'Asia session not yet complete — cannot predict UK',
        }

    asia_move = asia_data.get('asia_move', 0)
    asia_dir = 'UP' if asia_move > 0.3 else 'DOWN' if asia_move < -0.3 else 'FLAT'
    gap_held = asia_data.get('gap_dir', 'FLAT') == asia_dir or asia_data.get('gap_dir') == 'FLAT'
    is_sweep = asia_data.get('is_sweep_reversal', False)
    sweep_depth = asia_data.get('sweep_depth_pct', 0)
    recovery = asia_data.get('recovery_pct', 0)

    factors = []

    # Scenario 1: Asia sweep-reversal → UK likely fades (continues the reversal)
    if is_sweep:
        fade_prob = UK_FADE_SWEEP_REVERSAL.get(regime, 0.38)
        factors.append(f'Asia sweep-reversal ({sweep_depth:.1f}% swept, {recovery:.0f}% recovered)')
        factors.append(f'UK fade rate after sweep: {fade_prob:.0%} (regime={regime})')
        factors.append(f'Backtested: 38% overall fade, 59% after morning sweep (182 releases)')

        if asia_dir == 'DOWN':
            # Asia swept down then recovered → UK likely bounces
            prediction = 'BOUNCE'
            expected_move = abs(asia_move) * UK_MOVE_RATIO_AVG['FADE']
        else:
            # Asia swept up then reversed → UK likely sells
            prediction = 'SELL_OFF'
            expected_move = -abs(asia_move) * UK_MOVE_RATIO_AVG['FADE']

        # Note: fade_prob < 50% means UK more often continues than fades
        # But when it does fade, the move is significant (1.0x Asia move)
        confidence = 'MEDIUM' if fade_prob >= 0.45 else 'LOW'
        return {
            'prediction': prediction,
            'direction': 'UP' if prediction == 'BOUNCE' else 'DOWN',
            'confidence': confidence,
            'expected_move_pct': round(expected_move, 2),
            'probability': fade_prob,
            'factors': factors,
            'scenario': 'SWEEP_REVERSAL',
        }

    # Scenario 2: Asia continued US (gap held) → UK likely continues
    if gap_held and asia_dir != 'FLAT':
        cont_prob = UK_CONTINUATION_GAP_HELD.get(regime, 0.51)
        factors.append(f'Asia continued US ({asia_dir}, gap held)')
        factors.append(f'UK continuation rate: {cont_prob:.0%} (regime={regime})')
        factors.append(f'Backtested: 51% overall continuation (94 instances)')

        if us_dir == 'DUMP' and asia_dir == 'DOWN':
            # Both dumped — is London capitulation or more pain?
            double_dump = UK_AFTER_DOUBLE_DUMP
            bounce_prob = double_dump.get('BOUNCE', 0.52)
            if bounce_prob >= 0.52:
                prediction = 'BOUNCE'
                expected_move = abs(asia_move) * UK_MOVE_RATIO_AVG['FADE']
            else:
                prediction = 'CONTINUE_DOWN'
                expected_move = -abs(asia_move) * UK_MOVE_RATIO_AVG['CONTINUATION']
            factors.append(f'Double dump: {bounce_prob:.0%} bounce prob')
        elif us_dir == 'RALLY' and asia_dir == 'UP':
            prediction = 'CONTINUE_UP'
            expected_move = abs(asia_move) * UK_MOVE_RATIO_AVG['CONTINUATION']
        else:
            prediction = 'CONTINUATION'
            expected_move = asia_move * UK_MOVE_RATIO_AVG['CONTINUATION']

        confidence = 'MEDIUM' if cont_prob >= 0.55 else 'LOW'
        return {
            'prediction': prediction,
            'direction': 'UP' if 'UP' in prediction or prediction == 'BOUNCE' else 'DOWN',
            'confidence': confidence,
            'expected_move_pct': round(expected_move, 2),
            'probability': cont_prob,
            'factors': factors,
            'scenario': 'GAP_HELD',
        }

    # Scenario 3: Asia faded US → UK decides: continue fade or reverse back
    if asia_dir != 'FLAT' and not gap_held:
        # Double reversal: US did X, Asia did opposite, UK decides
        double_rev = UK_AFTER_DOUBLE_REVERSAL
        bounce_prob = double_rev.get('BOUNCE', 0.55)
        factors.append(f'Asia faded US ({us_dir}→{asia_dir})')
        factors.append(f'UK double-reversal: {bounce_prob:.0%} bounce (continues Asia fade)')

        if bounce_prob >= 0.55:
            prediction = 'CONTINUE_FADE'
            expected_move = asia_move * UK_MOVE_RATIO_AVG['CONTINUATION']
        else:
            prediction = 'REVERSE_TO_US'
            expected_move = -asia_move * UK_MOVE_RATIO_AVG['FADE']

        confidence = 'MEDIUM'
        return {
            'prediction': prediction,
            'direction': 'UP' if expected_move > 0 else 'DOWN',
            'confidence': confidence,
            'expected_move_pct': round(expected_move, 2),
            'probability': bounce_prob,
            'factors': factors,
            'scenario': 'DOUBLE_REVERSAL',
        }

    # Scenario 4: Flat — no edge
    return {
        'prediction': 'FLAT',
        'direction': 'NEUTRAL',
        'confidence': 'LOW',
        'expected_move_pct': 0.0,
        'probability': 0.50,
        'factors': ['Asia flat — no directional edge for UK'],
        'scenario': 'FLAT',
    }


# ═══════════════════════════════════════════════════════════════
# MAIN SCORING FUNCTION
# ═══════════════════════════════════════════════════════════════

def score_m23_ppi_session(df_15m, current_time=None, config=None):
    """Score macro data release dynamics (NFP + CPI + PPI + Claims) for session bias.

    Checks NFP, CPI, PPI, and Claims calendars. Handles combined release days
    (e.g., NFP+Claims on first Friday, or CPI+Claims on Thursday).

    Cascade model:
      NFP Friday (labor) → Claims Thursday → CPI Tue/Wed (inflation) → PPI Wed/Thu
      NFP sets macro tone for the week. CPI amplifies or reverses it.

    Args:
        df_15m: DataFrame with 15m OHLCV data
        current_time: datetime (default: now UTC)
        config: Config dict

    Returns:
        status: 'PASS', 'SKIP', or 'NO_DATA'
        score: 0.0-1.0
        details: dict with full analysis
    """
    cfg = config or CONFIG

    if not cfg.get('M23_ENABLED', False):
        return 'SKIP', 0.5, {'regime': 'DISABLED'}, 1.0

    if current_time is None:
        current_time = datetime.utcnow()

    today_str = current_time.strftime('%Y-%m-%d')
    yesterday_str = (current_time - timedelta(days=1)).strftime('%Y-%m-%d')

    # Check today and yesterday for both PPI and CPI
    # Priority: post-release (yesterday) first — it has actual data.
    # Then today's release day (prediction only).
    # v2: Extended window — check up to 7 days back with time decay
    release_date = None
    release_type = None
    is_release_day = False
    is_post_release = False
    release_decay_mult = 1.0
    release_days_since = 0

    # First pass: check yesterday for post-release analysis (has actual data)
    yesterday_rtype = get_release_type(yesterday_str)
    if yesterday_rtype is not None:
        # Verify US session data exists for yesterday
        test_date = datetime.strptime(yesterday_str, '%Y-%m-%d')
        test_dt = test_date.replace(hour=RELEASE_HOUR_UTC, minute=RELEASE_MINUTE_UTC)
        test_us = _get_bars_between(df_15m, test_dt, test_date.replace(hour=US_SESSION_END[0]))
        if len(test_us) >= 2:
            release_date = yesterday_str
            release_type = yesterday_rtype
            is_release_day = False
            is_post_release = True
            release_decay_mult, release_days_since = _compute_m23_release_decay(release_date, today_str)

    # Second pass: check today for prediction (if no yesterday data found)
    if release_date is None:
        today_rtype = get_release_type(today_str)
        if today_rtype is not None:
            release_date = today_str
            release_type = today_rtype
            is_release_day = True
            is_post_release = False
            release_decay_mult, release_days_since = _compute_m23_release_decay(release_date, today_str)

    # Third pass: extended window — check up to 7 days back with decay
    # Only triggers if no release today/yesterday, and decay > minimum threshold
    if release_date is None:
        for lookback_days in range(2, 8):
            check_date = (current_time - timedelta(days=lookback_days)).strftime('%Y-%m-%d')
            check_rtype = get_release_type(check_date)
            if check_rtype is not None:
                decay_mult_check, _ = _compute_m23_release_decay(check_date, today_str)
                if decay_mult_check > 0.10:
                    # Verify data exists
                    test_date = datetime.strptime(check_date, '%Y-%m-%d')
                    test_dt = test_date.replace(hour=RELEASE_HOUR_UTC, minute=RELEASE_MINUTE_UTC)
                    test_us = _get_bars_between(df_15m, test_dt, test_date.replace(hour=US_SESSION_END[0]))
                    if len(test_us) >= 2:
                        release_date = check_date
                        release_type = check_rtype
                        is_release_day = False
                        is_post_release = True
                        release_decay_mult = decay_mult_check
                        release_days_since = lookback_days
                        break

    # If yesterday was 'BOTH' (PPI+CPI same day), use combined stats
    # The type_strength modifier already handles BOTH = stronger signal

    if release_date is None:
        # ── No PPI/CPI/NFP release — check for standalone claims release ──
        if not cfg.get('M23_CLAIMS_ENABLED', True):
            return 'SKIP', 0.5, {'regime': 'NO_RELEASE', 'reason': 'No NFP/PPI/CPI release today or yesterday'}, 1.0

        claims_ctx = get_claims_context_for_release(None, cfg)
        if claims_ctx and is_claims_release_day(today_str):
            # Standalone jobless claims release day — apply claims decay
            claims = claims_ctx.get('claims', {})
            combo = claims_ctx.get('combo', {})
            signal = combo.get('signal', 'HOLD')
            expected = combo.get('expected_eth_move', 0)
            conf = combo.get('confidence', 'LOW')

            claims_decay_mult, claims_days = _compute_claims_decay(today_str)
            base_score = {'HIGH': 0.65, 'MEDIUM': 0.55, 'LOW': 0.50}.get(conf, 0.50)
            # Apply claims decay: pull toward neutral
            base_score = base_score * claims_decay_mult + 0.5 * (1.0 - claims_decay_mult)
            details = {
                'regime': 'CLAIMS_RELEASE',
                'release_type': 'CLAIMS',
                'is_release_day': True,
                'claims_context': claims_ctx,
                'claims': claims,
                'combo': combo,
                'decay_mult': claims_decay_mult,
                'days_since_release': claims_days,
                'score_reason': (
                    f'Jobless claims release: {claims.get("current", 0)}K ({claims.get("classification", "?")}), '
                    f'trend={claims.get("trend", "?")}, signal={signal}, expected={expected:+.1f}%, '
                    f'decay={claims_decay_mult:.2f}x ({claims_days}d)'
                ),
            }
            return 'PASS', base_score, details

        # No macro release at all — include claims context for background
        claims_ctx = get_claims_context_for_release(None, cfg)
        details = {
            'regime': 'NO_RELEASE',
            'reason': 'No PPI/CPI/Claims release today or yesterday',
            'claims_context': claims_ctx,
        }
        return 'SKIP', 0.5, details, 1.0

    # Get regime
    regime, fade_rate = classify_market_regime(cfg)

    # Compute US session data
    us_data = compute_us_session(df_15m, release_date)
    if us_data is None:
        return 'NO_DATA', 0.5, {'regime': regime, 'release_type': release_type,
                                 'reason': 'No US session data yet'}, 1.0

    # Compute 1h spike
    spike_data = compute_1h_spike(df_15m, release_date)

    # Compute Asia session (if available)
    asia_data = compute_asia_session(df_15m, release_date) if is_post_release else None

    # Compute UK session (if available — day after release, after Asia)
    uk_data = None
    if is_post_release:
        uk_data = compute_uk_session(df_15m, release_date)

    # Build result
    us_move = us_data['us_move']
    us_dir = 'DUMP' if us_move < -0.5 else 'RALLY' if us_move > 0.5 else 'FLAT'
    us_magnitude = 'BIG' if abs(us_move) > 3.0 else 'MEDIUM' if abs(us_move) > 1.5 else 'SMALL'

    # ── Jobless claims context (always available if enabled) ──
    claims_ctx = get_claims_context_for_release(release_type, cfg) if cfg.get('M23_CLAIMS_ENABLED', True) else None
    claims_today = is_claims_release_day(release_date) if release_date else False

    # UK prediction (available even before UK session starts, if Asia is done)
    uk_prediction = None
    if is_release_day and us_dir != 'FLAT':
        uk_prediction = _predict_uk_session(us_dir, None, regime, release_type)
    elif is_post_release and asia_data is not None and uk_data is None:
        uk_prediction = _predict_uk_session(us_dir, asia_data, regime, release_type)

    # Spike accuracy for this type+year
    year = current_time.year if is_release_day else int(release_date[:4])
    if 'NFP' in release_type and '+' in release_type:
        # Combined release (e.g., NFP+CPI) — use max accuracy
        spike_acc = max(NFP_SPIKE_ACCURACY.get(year, 0.66), SPIKE_ACCURACY.get(year, 0.70))
    elif 'NFP' in release_type:
        spike_acc = NFP_SPIKE_ACCURACY.get(year, 0.66)
    elif release_type == 'CPI':
        spike_acc = SPIKE_ACCURACY_CPI.get(year, 0.66)
    elif release_type == 'PPI':
        spike_acc = SPIKE_ACCURACY_PPI.get(year, 0.73)
    else:  # BOTH or CLAIMS
        spike_acc = SPIKE_ACCURACY.get(year, 0.70)

    # ── CASCADE: Type-specific strength modifier ──
    # NFP: high vol (2.01% avg |move|), no directional bias, strong Asia fade.
    # CPI: PRIMARY inflation signal (biggest ETH reaction).
    # PPI: confirmation of CPI signal.
    type_strength = 1.0
    if 'NFP' in release_type and 'CPI' in release_type:
        type_strength = 1.35   # NFP+CPI same day = max signal (rare but explosive)
    elif 'NFP' in release_type and 'PPI' in release_type:
        type_strength = 1.20   # NFP+PPI = labor + inflation confirmation
    elif release_type == 'CPI':
        type_strength = 1.15   # CPI = primary inflation signal
    elif release_type == 'BOTH':
        type_strength = 1.25   # CPI+PPI same day = max inflation signal
    elif 'NFP' in release_type:
        type_strength = 1.10   # NFP standalone = labor market signal
    # PPI alone = 1.0 (default, confirmation-only)

    # ── CASCADE: Regime sensitivity multiplier ──
    # ETH's macro sensitivity evolved: near-zero in 2018 → very high in 2022 → partial decouple by 2025
    regime_sensitivity = REGIME_SENSITIVITY.get(regime, 0.80)

    details = {
        'release_date': release_date,
        'release_type': release_type,
        'is_release_day': is_release_day,
        'is_post_release': is_post_release,
        'regime': regime,
        'fade_rate': fade_rate,
        'spike_accuracy': spike_acc,
        'type_strength': type_strength,
        'regime_sensitivity': regime_sensitivity,
        'us_data': us_data,
        'spike_data': spike_data,
        'asia_data': asia_data,
        'uk_data': uk_data,
        'uk_prediction': uk_prediction,
        'us_direction': us_dir,
        'us_magnitude': us_magnitude,
        # Jobless claims context
        'claims_context': claims_ctx,
        'claims_today': claims_today,
        # Time decay
        'decay_mult': release_decay_mult,
        'days_since_release': release_days_since,
    }

    # ── Post-release: Asia already happened ──
    if is_post_release and asia_data is not None:
        asia_move = asia_data['asia_move']
        asia_gap = asia_data['asia_gap']

        gap_dir = asia_data.get('gap_dir', 'UP' if asia_gap > 0.2 else 'DOWN' if asia_gap < -0.2 else 'FLAT')
        asia_dir = 'UP' if asia_move > 0.3 else 'DOWN' if asia_move < -0.3 else 'FLAT'
        gap_held = (gap_dir == asia_dir) or gap_dir == 'FLAT'

        asia_faded = (us_dir == 'RALLY' and asia_dir == 'DOWN') or \
                     (us_dir == 'DUMP' and asia_dir == 'UP')

        # Path analysis: detect sweep-and-reverse patterns
        is_sweep_reversal = asia_data.get('is_sweep_reversal', False)
        sweep_depth = asia_data.get('sweep_depth_pct', 0)
        recovery_pct = asia_data.get('recovery_pct', 0)
        reclaimed_gap = asia_data.get('reclaimed_gap', False)

        # Crash detection
        asia_range_pct = asia_data.get('asia_range', 0)
        is_crash = _detect_crash(gap_dir, asia_range_pct, is_sweep_reversal)

        # Dump size classification
        dump_size = _classify_dump_size(us_move)

        # Reversal probability (for context — Asia already happened)
        rev_prob, rev_confidence, rev_factors = _compute_reversal_probability(
            us_move, regime, dump_size, release_type)

        # Classify the actual pattern
        if is_crash:
            pattern = 'CRASH'
        elif is_sweep_reversal:
            pattern = 'SWEEP_REVERSAL'
        elif asia_faded:
            pattern = 'FADE'
        elif asia_dir != 'FLAT':
            pattern = 'CONTINUATION'
        else:
            pattern = 'FLAT'

        details['asia_analysis'] = {
            'asia_move': asia_move,
            'asia_gap': asia_gap,
            'gap_direction': gap_dir,
            'asia_direction': asia_dir,
            'gap_held': gap_held,
            'asia_faded_us': asia_faded,
            'pattern': pattern,
            # Path data
            'sweep_depth_pct': sweep_depth,
            'recovery_pct': recovery_pct,
            'reclaimed_gap': reclaimed_gap,
            'is_sweep_reversal': is_sweep_reversal,
            # Enhanced analysis
            'is_crash': is_crash,
            'dump_size': dump_size,
            'reversal_probability': rev_prob,
            'reversal_confidence': rev_confidence,
            'reversal_factors': rev_factors,
        }

        # ── UK session analysis (if available) ──
        uk_analysis = None
        if uk_data is not None:
            uk_dir = uk_data.get('uk_direction', 'FLAT')
            uk_move = uk_data.get('uk_move_vs_asia', 0)
            uk_continued = uk_data.get('uk_continued_asia', False)
            uk_faded = uk_data.get('uk_faded_asia', False)
            uk_swept = uk_data.get('is_morning_sweep', False)
            uk_taker = uk_data.get('uk_taker', 0.5)
            vol_ratio = uk_data.get('uk_vol_ratio_vs_asia', 1.0)

            # UK confirmed or denied Asia's move
            if uk_continued:
                uk_verdict = 'CONFIRMED'
                uk_confidence_boost = 0.05
            elif uk_faded:
                uk_verdict = 'FADED'
                uk_confidence_boost = -0.05
            elif uk_swept:
                uk_verdict = 'SWEPT_AND_REVERSED'
                uk_confidence_boost = -0.08
            else:
                uk_verdict = 'NEUTRAL'
                uk_confidence_boost = 0.0

            # UK volume confirms conviction?
            vol_confirm = vol_ratio >= 1.2  # UK traded heavier than Asia

            uk_analysis = {
                'uk_move_vs_asia': uk_move,
                'uk_move_vs_us': uk_data.get('uk_move_vs_us', 0),
                'uk_direction': uk_dir,
                'uk_verdict': uk_verdict,
                'uk_taker': uk_taker,
                'uk_vol_ratio': vol_ratio,
                'vol_confirm': vol_confirm,
                'uk_swept_asia': uk_data.get('uk_swept_asia_high', False) or uk_data.get('uk_swept_asia_low', False),
                'uk_confidence_boost': uk_confidence_boost,
            }
            details['uk_analysis'] = uk_analysis

        # Score: cascade model with regime sensitivity + UK confirmation
        score_adjust = 0.0
        if uk_analysis:
            score_adjust = uk_analysis.get('uk_confidence_boost', 0)

        # Base score from pattern
        if is_crash:
            base_pattern_score = 0.35
        elif is_sweep_reversal:
            if rev_prob >= 0.70:
                base_pattern_score = 0.40
            elif rev_prob >= 0.50:
                base_pattern_score = 0.45
            else:
                base_pattern_score = 0.50
        elif gap_held:
            base_pattern_score = 0.65
        else:
            base_pattern_score = 0.45

        # ── NFP-specific adjustments ──
        # NFP has stronger Asia fade pattern (60% after dump, 64% after rally)
        # vs PPI/CPI which is weaker. Apply NFP fade boost when applicable.
        nfp_fade_boost = 0.0
        if 'NFP' in release_type:
            if asia_faded:
                # NFP Asia fade is more reliable than PPI/CPI fade
                nfp_fade_boost = 0.05
                factors = details.get('asia_analysis', {}).get('reversal_factors', [])
                factors.append(f'NFP Asia fade boost: +0.05 (60-64% fade rate)')
            # NFP seasonality
            nfp_month = int(release_date[5:7])
            season = NFP_SEASONALITY.get(nfp_month, {})
            details['nfp_seasonality'] = season
            if season.get('bias') == 'DANGER':
                nfp_fade_boost -= 0.05  # September danger month
            elif season.get('bias') == 'BULLISH':
                nfp_fade_boost += 0.03  # November bullish

        # Apply regime sensitivity: in high-sensitivity regimes (2022, stagflation),
        # the macro signal matters more. In low-sensitivity (2018, 2024), it's noise.
        score = base_pattern_score * regime_sensitivity + (1 - regime_sensitivity) * 0.50
        score = max(0.20, min(0.85, score + score_adjust + nfp_fade_boost))

        # Apply time decay: PPI/CPI impact fades over days since release
        score = score * release_decay_mult + 0.5 * (1.0 - release_decay_mult)
        score = max(0.20, min(0.85, score))
        status = 'PASS'

        # Build score reason with UK context
        reason_parts = [
            f'{release_type} {release_date}: {pattern.lower()} '
            f'({gap_dir}→{asia_dir}), US was {us_dir} {us_move:+.2f}%'
        ]
        if is_sweep_reversal:
            reason_parts.append(f'sweep {sweep_depth:.1f}% reversed {recovery_pct:.0f}%')
        if dump_size != 'NOT_DUMP':
            reason_parts.append(f'rev_prob={rev_prob:.0%}')
        if uk_analysis:
            reason_parts.append(f'UK {uk_analysis["uk_verdict"].lower()} ({uk_dir} {uk_move:+.2f}%)')
        if 'NFP' in release_type:
            season = details.get('nfp_seasonality', {})
            reason_parts.append(f'NFP seasonality: {season.get("bias", "?")} (avg {season.get("avg_move", 0):+.2f}%)')
        if release_decay_mult < 1.0:
            reason_parts.append(f'decay={release_decay_mult:.2f}x ({release_days_since}d)')
        details['score_reason'] = ', '.join(reason_parts)

    # ── Release day: predict Asia ──
    elif is_release_day:
        if us_dir == 'FLAT':
            details['asia_prediction'] = {
                'direction': 'NEUTRAL',
                'confidence': 'LOW',
                'reason': f'US move too small ({us_move:+.2f}%) — no edge',
            }
            return 'SKIP', 0.5, details, 1.0

        # Dump size classification
        dump_size = _classify_dump_size(us_move)

        # Expected Asia behavior from historical data
        expected_key = (us_dir, regime)
        expected_asia = ASIA_MOVE_AVG.get(expected_key, 0.0)

        # Reversal probability (8-year study)
        rev_prob, rev_confidence, rev_factors = _compute_reversal_probability(
            us_move, regime, dump_size, release_type)

        # Determine bias using regime + reversal probability
        if us_dir == 'DUMP':
            if dump_size == 'CRASH':
                # Genuine crash — Asia likely continues
                bias = 'CONTINUATION'
                confidence = 'HIGH'
            elif rev_prob >= 0.65:
                # High reversal probability — Asia likely bounces
                bias = 'FADE'
                confidence = rev_confidence
            elif rev_prob >= 0.50:
                bias = 'FADE'
                confidence = 'MEDIUM'
            else:
                bias = 'CONTINUATION'
                confidence = 'MEDIUM'
        elif us_dir == 'RALLY':
            if regime in ('STAGFLATION', 'STAGFLATION_HOT'):
                # Asia tends to fade rallies in stagflation
                bias = 'FADE'
                confidence = 'MEDIUM'
            elif regime in ('BULL',):
                bias = 'CONTINUATION'
                confidence = 'HIGH'
            else:
                bias = 'MIXED'
                confidence = 'MEDIUM'
        else:
            bias = 'MIXED'
            confidence = 'LOW'

        # ── NFP-specific: Asia fade adjustment ──
        # NFP has STRONGER Asia fade pattern than CPI/PPI.
        # After US dump: 60% Asia bounce. After US rally: 64% Asia fade.
        nfp_asia_fade_rate = None
        if 'NFP' in release_type:
            if us_dir == 'DUMP':
                nfp_asia_fade_rate = NFP_ASIA_FADE_AFTER_DUMP
                if bias == 'CONTINUATION' and nfp_asia_fade_rate >= 0.55:
                    bias = 'FADE'
                    confidence = 'MEDIUM'
            elif us_dir == 'RALLY':
                nfp_asia_fade_rate = NFP_ASIA_FADE_AFTER_RALLY
                if bias != 'FADE' and nfp_asia_fade_rate >= 0.60:
                    bias = 'FADE'
                    confidence = 'MEDIUM'
            # NFP seasonality boost
            nfp_month = int(release_date[5:7])
            season = NFP_SEASONALITY.get(nfp_month, {})
            details['nfp_seasonality'] = season
            details['nfp_asia_fade_rate'] = nfp_asia_fade_rate
            if season.get('bias') == 'DANGER':
                if us_dir == 'DUMP':
                    confidence = 'HIGH'  # September dump = high confidence continuation
            elif season.get('bias') == 'BULLISH':
                if us_dir == 'RALLY':
                    confidence = 'HIGH'  # November rally = high confidence

        # ── CASCADE: Adjust confidence by type ──
        # NFP is labor market signal (high vol, no directional bias).
        # CPI is primary inflation signal → boosts confidence. PPI alone → noisier.
        if 'NFP' in release_type and 'CPI' in release_type:
            if confidence == 'MEDIUM':
                confidence = 'HIGH'  # NFP+CPI = max signal
        elif 'NFP' in release_type:
            pass  # NFP standalone — use its own confidence (already set above)
        elif release_type == 'CPI' and confidence == 'MEDIUM':
            confidence = 'HIGH'    # CPI upgrade — primary signal
        elif release_type == 'BOTH' and confidence == 'MEDIUM':
            confidence = 'HIGH'    # Both upgrade — max signal
        elif release_type == 'PPI' and confidence == 'HIGH':
            confidence = 'MEDIUM'  # PPI downgrade — confirmation only

        # 1h spike adjustment
        spike_bias = None
        if spike_data:
            spike_dir = spike_data['spike_dir']
            if spike_acc >= 0.70:
                spike_bias = spike_dir
                details['spike_note'] = f'{release_type} 1h spike {spike_dir} ({spike_acc:.0%} accuracy)'

        # Expected recovery if reversal
        expected_recovery = PPI_AVG_RECOVERY_BY_SIZE.get(dump_size, 100)

        details['asia_prediction'] = {
            'us_direction': us_dir,
            'us_move': us_move,
            'us_magnitude': us_magnitude,
            'expected_asia_move': round(expected_asia, 2),
            'regime_bias': bias,
            'confidence': confidence,
            'fade_rate': f'{fade_rate:.0%}',
            'spike_bias': spike_bias,
            'release_type': release_type,
            'spike_accuracy': spike_acc,
            # Enhanced prediction
            'dump_size': dump_size,
            'reversal_probability': rev_prob,
            'reversal_confidence': rev_confidence,
            'reversal_factors': rev_factors,
            'expected_recovery_pct': expected_recovery,
        }

        # Score: cascade model with regime sensitivity
        base_score = {'HIGH': 0.75, 'MEDIUM': 0.60, 'LOW': 0.50}.get(confidence, 0.50)
        # Apply regime sensitivity and type strength
        score = base_score * type_strength * regime_sensitivity + (1 - regime_sensitivity) * 0.50
        # Apply time decay (should be 1.0 on release day, but included for consistency)
        score = score * release_decay_mult + 0.5 * (1.0 - release_decay_mult)
        score = max(0.30, min(0.90, score))
        status = 'PASS'

        nfp_reason = ''
        if 'NFP' in release_type:
            nfp_season = details.get('nfp_seasonality', {})
            nfp_fade = details.get('nfp_asia_fade_rate')
            nfp_reason = f', NFP_season={nfp_season.get("bias", "?")}'
            if nfp_fade:
                nfp_reason += f', asia_fade={nfp_fade:.0%}'

        details['score_reason'] = (
            f'{release_type} release day: US {us_dir} {us_move:+.2f}%, '
            f'regime={regime} (sens={regime_sensitivity:.2f}), bias={bias}, conf={confidence}'
            + (f', rev_prob={rev_prob:.0%}' if dump_size != 'NOT_DUMP' else '')
            + nfp_reason
            + (f', decay={release_decay_mult:.2f}x' if release_decay_mult < 1.0 else '')
        )

    else:
        return 'SKIP', 0.5, {'regime': 'NO_RELEASE', 'reason': 'No release context'}, 1.0

    return status, score, details, details.get('decay_mult', 1.0)


# ═══════════════════════════════════════════════════════════════
# FORMATTER
# ═══════════════════════════════════════════════════════════════

def format_m23(details):
    """Format M23 details for terminal output."""
    if not details or details.get('regime') in ('DISABLED', 'NO_PPI', 'NO_DATA'):
        return ''

    lines = []
    regime = details.get('regime', '?')

    # ── Standalone claims release ──
    if regime == 'CLAIMS_RELEASE':
        claims = details.get('claims', {})
        combo = details.get('combo', {})
        return format_claims_context(details.get('claims_context'))

    if regime == 'NO_RELEASE':
        # Show claims context even on non-release days
        claims_ctx = details.get('claims_context')
        if claims_ctx:
            return format_claims_context(claims_ctx)
        return ''

    release_date = details.get('release_date', details.get('ppi_date', '?'))
    release_type = details.get('release_type', 'PPI')
    us_data = details.get('us_data', {})
    spike_data = details.get('spike_data', {})
    asia_data = details.get('asia_data', {})
    asia_pred = details.get('asia_prediction', {})
    asia_analysis = details.get('asia_analysis', {})
    uk_data = details.get('uk_data', {})
    uk_pred = details.get('uk_prediction', {})
    uk_analysis = details.get('uk_analysis', {})
    spike_acc = details.get('spike_accuracy', 0.70)
    type_strength = details.get('type_strength', 1.0)
    regime_sensitivity = details.get('regime_sensitivity', 0.80)
    claims_ctx = details.get('claims_context')
    claims_today = details.get('claims_today', False)

    # Header with type icon — CASCADE model
    type_icons = {'PPI': '🏭', 'CPI': '🛒', 'BOTH': '📊', 'NFP': '💼',
                  'CLAIMS': '📋'}
    # Handle combined types like 'NFP+CPI', 'NFP+PPI', 'NFP+CLAIMS'
    if '+' in release_type:
        parts = release_type.split('+')
        type_icon = '📊'  # combined = max signal
    else:
        type_icon = type_icons.get(release_type, '📊')
    claims_tag = ' + 📋 CLAIMS' if claims_today else ''
    lines.append(f"\n  {type_icon} M23 {release_type}{claims_tag} CASCADE: {release_date}")
    lines.append(f"    Regime: {regime}  fade={details.get('fade_rate', 0):.0%}  "
                 f"sensitivity={regime_sensitivity:.2f}  spike_acc={spike_acc:.0%}")
    # Time decay
    decay_mult = details.get('decay_mult', 1.0)
    days_since = details.get('days_since_release', 0)
    if decay_mult < 1.0:
        decay_icon = '🟢' if decay_mult >= 0.70 else '🟡' if decay_mult >= 0.40 else '🟠' if decay_mult >= 0.20 else '🔴'
        lines.append(f"    {decay_icon} Decay: {decay_mult:.2f}x  ({days_since}d since release)")

    # Show cascade sequence
    if 'NFP' in release_type and 'CPI' in release_type:
        lines.append(f"    📊 Cascade: NFP+CPI same day (max signal) → PPI tomorrow (confirm)")
    elif 'NFP' in release_type and 'PPI' in release_type:
        lines.append(f"    📊 Cascade: NFP+PPI (labor+inflation) → Claims Thu (context)")
    elif 'NFP' in release_type:
        lines.append(f"    📊 Cascade: NFP (labor) → Claims Thu → CPI next week (inflation)")
    elif release_type == 'CPI':
        lines.append(f"    📊 Cascade: CPI (primary) → PPI tomorrow (confirm/deny) → Claims Thu (context)")
    elif release_type == 'PPI':
        lines.append(f"    📊 Cascade: CPI already set tone → PPI (confirm/deny) → Claims Thu (context)")
    elif release_type == 'BOTH':
        lines.append(f"    📊 Cascade: CPI+PPI same day (max signal) → Claims Thu (context)")

    # NFP seasonality (if applicable)
    nfp_season = details.get('nfp_seasonality', {})
    nfp_fade = details.get('nfp_asia_fade_rate')
    if nfp_season:
        bias = nfp_season.get('bias', '?')
        avg_move = nfp_season.get('avg_move', 0)
        bias_icons = {'DANGER': '🔴', 'BEARISH': '🟠', 'BULLISH': '🟢', 'NEUTRAL': '⚪'}
        b_icon = bias_icons.get(bias, '⚪')
        lines.append(f"    {b_icon} NFP Seasonality: {bias} (avg {avg_move:+.2f}%)")
    if nfp_fade is not None:
        lines.append(f"    📊 NFP Asia fade rate: {nfp_fade:.0%}")

    # ── Claims context (before US session data) ──
    if claims_ctx:
        lines.append(format_claims_context(claims_ctx))

    # US session
    if us_data:
        us_move = us_data.get('us_move', 0)
        us_icon = '🔴' if us_move < -0.5 else '🟢' if us_move > 0.5 else '⚪'
        lines.append(f"    US Session: {us_icon} {us_move:+.2f}%  "
                     f"(range {us_data.get('us_range', 0):.1f}%)  "
                     f"[{details.get('us_magnitude', '?')}]")

    # 1h spike
    if spike_data:
        spike_pct = spike_data.get('spike_pct', 0)
        spike_icon = '🟢' if spike_pct > 0.3 else '🔴' if spike_pct < -0.3 else '⚪'
        lines.append(f"    1h Spike: {spike_icon} {spike_pct:+.2f}%  "
                     f"(range {spike_data.get('spike_range', 0):.2f}%)")

    # Asia prediction (release day, before Asia)
    if asia_pred and not asia_data:
        direction = asia_pred.get('regime_bias', '?')
        confidence = asia_pred.get('confidence', '?')
        expected = asia_pred.get('expected_asia_move', 0)
        us_dir = asia_pred.get('us_direction', '?')
        fade_rate = asia_pred.get('fade_rate', '?')
        spike_bias = asia_pred.get('spike_bias')
        dump_size = asia_pred.get('dump_size', 'NOT_DUMP')
        rev_prob = asia_pred.get('reversal_probability', 0)
        expected_recovery = asia_pred.get('expected_recovery_pct', 0)

        conf_icon = {'HIGH': '🟢', 'MEDIUM': '🟡', 'LOW': '🔴'}.get(confidence, '⚪')
        lines.append(f"    Asia Prediction: {conf_icon} {direction} (conf: {confidence})")
        lines.append(f"    Expected Asia: {expected:+.2f}%  (fade rate: {fade_rate})")

        # Show reversal probability for dumps
        if dump_size != 'NOT_DUMP' and us_dir == 'DUMP':
            rev_icon = '🟢' if rev_prob >= 0.65 else '🟡' if rev_prob >= 0.50 else '🔴'
            lines.append(f"    {rev_icon} Reversal prob: {rev_prob:.0%} ({dump_size} dump)")
            if direction == 'FADE':
                lines.append(f"    📊 Expected recovery: ~{expected_recovery}% of sweep")

        if spike_bias:
            lines.append(f"    1h Spike Bias: {spike_bias}  ({spike_acc:.0%} accuracy)")

        # Trade suggestion
        if 'NFP' in (release_type or ''):
            nfp_fade = asia_pred.get('nfp_asia_fade_rate')
            if direction == 'FADE':
                fade_str = f' (NFP fade rate: {nfp_fade:.0%})' if nfp_fade else ''
                if us_dir == 'DUMP':
                    lines.append(f"    💡 NFP: Asia likely BOUNCES after US dump{fade_str} — watch long at Asia open")
                elif us_dir == 'RALLY':
                    lines.append(f"    💡 NFP: Asia likely FADES after US rally{fade_str} — watch short at Asia open")
            elif direction == 'CONTINUATION':
                if dump_size == 'CRASH':
                    lines.append(f"    🚨 NFP CRASH MODE — Asia likely continues selling, do NOT buy the dip")
                else:
                    lines.append(f"    💡 NFP: Asia likely CONTINUES {us_dir.lower()} — momentum trade")
            # NFP→CPI cascade hint
            nfp_month = int(release_date[5:7]) if release_date else 0
            # Check if CPI is coming next week
            lines.append(f"    📊 NFP sets labor tone → Claims Thu → CPI next week (inflation)")
        elif direction == 'FADE' and confidence in ('HIGH', 'MEDIUM'):
            if us_dir == 'DUMP':
                lines.append(f"    💡 Asia likely BOUNCES after US dump — watch long at Asia open")
            elif us_dir == 'RALLY':
                lines.append(f"    💡 Asia likely FADES after US rally — watch short at Asia open")
        elif direction == 'CONTINUATION':
            if dump_size == 'CRASH':
                lines.append(f"    🚨 CRASH MODE — Asia likely continues selling, do NOT buy the dip")
            else:
                lines.append(f"    💡 Asia likely CONTINUES {us_dir.lower()} — momentum trade")

    # Asia actual (post-release)
    if asia_analysis:
        asia_move = asia_analysis.get('asia_move', 0)
        asia_gap = asia_analysis.get('asia_gap', 0)
        gap_held = asia_analysis.get('gap_held', False)
        pattern = asia_analysis.get('pattern', '?')
        asia_icon = '🟢' if asia_move > 0 else '🔴'
        gap_icon = '✅' if gap_held else '❌'

        lines.append(f"    Asia Session: {asia_icon} {asia_move:+.2f}%  (gap {asia_gap:+.2f}%)")
        lines.append(f"    Gap Held: {gap_icon}  Pattern: {pattern}")

        # Show dump size and reversal probability
        dump_size = asia_analysis.get('dump_size', 'NOT_DUMP')
        rev_prob = asia_analysis.get('reversal_probability', 0)
        rev_conf = asia_analysis.get('reversal_confidence', '')
        if dump_size != 'NOT_DUMP':
            rev_icon = '🟢' if rev_prob >= 0.65 else '🟡' if rev_prob >= 0.50 else '🔴'
            lines.append(f"    {rev_icon} Dump: {dump_size}  Reversal prob: {rev_prob:.0%} ({rev_conf})")

        # Show crash detection
        if pattern == 'CRASH':
            lines.append(f"    🚨 CRASH MODE — genuine continuation, NOT a buying opportunity")
            lines.append(f"    ⚠️ Asia range: {asia_analysis.get('sweep_depth_pct', 0):.1f}% — extreme selling")
        elif pattern == 'SWEEP_REVERSAL':
            sweep_depth = asia_analysis.get('sweep_depth_pct', 0)
            recovery = asia_analysis.get('recovery_pct', 0)
            reclaimed = asia_analysis.get('reclaimed_gap', False)
            lines.append(f"    ⚠️ SWEEP-AND-REVERSE: swept {sweep_depth:.1f}% then recovered {recovery:.0f}%")
            if reclaimed:
                lines.append(f"    ↩️ Reclaimed gap level — continuation unreliable")
            if rev_prob >= 0.65:
                lines.append(f"    📊 High-probability reversal ({rev_prob:.0%}) — matches 8-year pattern")
        elif asia_analysis.get('asia_faded_us'):
            lines.append(f"    ↩️ Asia FADED US (mean-reversion)")
        else:
            lines.append(f"    ✅ Asia CONTINUED US (momentum)")

    # ── UK Session (London) ──
    # Show UK prediction first (if Asia is done but London hasn't closed)
    if uk_pred and not uk_data:
        pred = uk_pred.get('prediction', '?')
        conf = uk_pred.get('confidence', '?')
        scenario = uk_pred.get('scenario', '?')
        expected = uk_pred.get('expected_move_pct', 0)
        prob = uk_pred.get('probability', 0)
        factors = uk_pred.get('factors', [])

        conf_icon = {'HIGH': '🟢', 'MEDIUM': '🟡', 'LOW': '🔴'}.get(conf, '⚪')
        lines.append(f"\n    🇬🇧 UK Prediction: {conf_icon} {pred} (conf: {conf})")
        lines.append(f"    Scenario: {scenario}  Expected: {expected:+.2f}%  Prob: {prob:.0%}")
        for f in factors:
            lines.append(f"      • {f}")

    # Show UK actual data (after London session)
    if uk_data:
        uk_move = uk_data.get('uk_move_vs_asia', 0)
        uk_dir = uk_data.get('uk_direction', 'FLAT')
        uk_icon = '🟢' if uk_move > 0.3 else '🔴' if uk_move < -0.3 else '⚪'
        taker = uk_data.get('uk_taker', 0.5)
        taker_label = 'buyers' if taker > 0.52 else 'sellers' if taker < 0.48 else 'neutral'
        vol_ratio = uk_data.get('uk_vol_ratio_vs_asia', 1.0)

        lines.append(f"\n    🇬🇧 UK Session: {uk_icon} {uk_move:+.2f}% vs Asia  "
                     f"(vs US: {uk_data.get('uk_move_vs_us', 0):+.2f}%)")
        lines.append(f"    UK Range: {uk_data.get('uk_range', 0):.2f}%  "
                     f"Taker: {taker:.3f} ({taker_label})  "
                     f"Vol vs Asia: {vol_ratio:.1f}x")

        # UK behavior relative to Asia
        if uk_data.get('uk_continued_asia'):
            lines.append(f"    ✅ London CONTINUED Asia direction (momentum confirmation)")
        elif uk_data.get('uk_faded_asia'):
            lines.append(f"    ↩️ London FADED Asia (mean-reversion)")
        elif uk_data.get('is_morning_sweep'):
            sweep_rec = uk_data.get('sweep_recovery_pct', 0)
            lines.append(f"    ⚡ London SWEPT Asia high/low then reversed ({sweep_rec:.0f}% recovery)")

        # UK sweep levels
        if uk_data.get('uk_swept_asia_high'):
            lines.append(f"    ⚠️ Swept Asia high ${uk_data.get('uk_high', 0):.2f} — potential distribution")
        if uk_data.get('uk_swept_asia_low'):
            lines.append(f"    ⚠️ Swept Asia low ${uk_data.get('uk_low', 0):.2f} — potential accumulation")

    # Show UK analysis (incorporated into scoring)
    if uk_analysis:
        verdict = uk_analysis.get('uk_verdict', '?')
        boost = uk_analysis.get('uk_confidence_boost', 0)
        vol_conf = uk_analysis.get('vol_confirm', False)
        verdict_icons = {'CONFIRMED': '✅', 'FADED': '↩️', 'SWEPT_AND_REVERSED': '⚡', 'NEUTRAL': '⚪'}
        v_icon = verdict_icons.get(verdict, '⚪')
        vol_tag = '  📊 vol confirms' if vol_conf else ''
        lines.append(f"    UK Verdict: {v_icon} {verdict}  (score adj: {boost:+.03f}){vol_tag}")

    return '\n'.join(lines)

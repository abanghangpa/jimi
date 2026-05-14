"""
M23: Macro Data Release Analysis — PPI & CPI session dynamics

Analyzes the US→Asia→NextUS chain on PPI and CPI release days based on
8 years of historical data (2018-2026, 199 releases: 98 PPI + 101 CPI).

Key findings from full analysis:
    1. PPI moves markets harder than CPI: avg |move| 0.64% vs 0.04%
    2. PPI 1h spike→US accuracy: 73.5%  |  CPI: 66.1%
    3. Asia gap holds ~74% of the time (both PPI & CPI)
    4. Stagflation fade rate: 45.5% — Asia tends to reverse US moves
    5. Overall bias is bearish: 45.7% dump vs 32.2% rally
    6. Big moves (>3%) reverse 31% next US session
    7. CPI produces more volatile ranges (4.66%) vs PPI (4.35%)

Data sources:
    - BLS PPI/CPI release schedules (hardcoded dates 2018-2026)
    - Live 15m OHLCV data
    - M22 inflation regime (for fade/continuation bias)
"""

from src.config import CONFIG
import numpy as np
import pandas as pd
from datetime import datetime, timedelta


# ═══════════════════════════════════════════════════════════════
# RELEASE DATES (8:30 AM ET = 13:30 UTC)
# ═══════════════════════════════════════════════════════════════

PPI_RELEASE_DATES = {
    # 2018
    '2018-01-11', '2018-02-15', '2018-03-14', '2018-04-11',
    '2018-05-10', '2018-06-12', '2018-07-11', '2018-09-12',
    '2018-10-10', '2018-11-14', '2018-12-11',
    # 2019
    '2019-01-15', '2019-02-14', '2019-03-14', '2019-04-11',
    '2019-05-09', '2019-06-11', '2019-07-12', '2019-08-09',
    '2019-09-11', '2019-10-08', '2019-11-14', '2019-12-12',
    # 2020
    '2020-01-14', '2020-02-19', '2020-03-12', '2020-04-09',
    '2020-05-12', '2020-06-10', '2020-07-14', '2020-08-11',
    '2020-09-11', '2020-10-14', '2020-11-13', '2020-12-11',
    # 2021
    '2021-01-13', '2021-02-17', '2021-03-12', '2021-04-09',
    '2021-05-12', '2021-06-11', '2021-07-13', '2021-08-12',
    '2021-09-10', '2021-10-14', '2021-11-09', '2021-12-14',
    # 2022
    '2022-01-13', '2022-02-15', '2022-03-11', '2022-04-12',
    '2022-05-12', '2022-06-14', '2022-07-14', '2022-08-11',
    '2022-09-14', '2022-10-12', '2022-11-15', '2022-12-09',
    # 2023
    '2023-01-18', '2023-02-16', '2023-03-15', '2023-04-13',
    '2023-05-11', '2023-06-14', '2023-07-13', '2023-08-11',
    '2023-09-14', '2023-10-12', '2023-11-15', '2023-12-13',
    # 2024
    '2024-01-12', '2024-02-16', '2024-03-14', '2024-04-11',
    '2024-05-14', '2024-06-13', '2024-07-12', '2024-08-13',
    '2024-09-12', '2024-10-11', '2024-11-14', '2024-12-12',
    # 2025
    '2025-01-14', '2025-02-13', '2025-03-13', '2025-04-10',
    '2025-05-15', '2025-06-12', '2025-07-16', '2025-08-14',
    '2025-09-11', '2025-10-15', '2025-11-13', '2025-12-11',
    # 2026 (confirmed from BLS schedule: https://www.bls.gov/schedule/news_release/ppi.htm)
    '2026-01-14', '2026-02-13', '2026-03-13', '2026-04-14',
    '2026-05-13', '2026-06-11', '2026-07-10', '2026-08-13',
    '2026-09-11', '2026-10-14', '2026-11-13', '2026-12-10',
}

CPI_RELEASE_DATES = {
    # 2018
    '2018-01-11', '2018-02-14', '2018-03-13', '2018-04-11',
    '2018-05-10', '2018-06-12', '2018-07-12', '2018-08-10',
    '2018-09-13', '2018-10-11', '2018-11-14', '2018-12-12',
    # 2019
    '2019-01-11', '2019-02-13', '2019-03-12', '2019-04-10',
    '2019-05-09', '2019-06-12', '2019-07-11', '2019-08-13',
    '2019-09-12', '2019-10-10', '2019-11-13', '2019-12-11',
    # 2020
    '2020-01-14', '2020-02-13', '2020-03-11', '2020-04-10',
    '2020-05-12', '2020-06-10', '2020-07-14', '2020-08-12',
    '2020-09-11', '2020-10-13', '2020-11-12', '2020-12-10',
    # 2021
    '2021-01-13', '2021-02-10', '2021-03-10', '2021-04-13',
    '2021-05-12', '2021-06-10', '2021-07-13', '2021-08-11',
    '2021-09-14', '2021-10-13', '2021-11-10', '2021-12-10',
    # 2022
    '2022-01-12', '2022-02-10', '2022-03-10', '2022-04-12',
    '2022-05-11', '2022-06-10', '2022-07-13', '2022-08-10',
    '2022-09-13', '2022-10-13', '2022-11-10', '2022-12-13',
    # 2023
    '2023-01-12', '2023-02-14', '2023-03-14', '2023-04-12',
    '2023-05-10', '2023-06-13', '2023-07-12', '2023-08-10',
    '2023-09-13', '2023-10-12', '2023-11-14', '2023-12-12',
    # 2024
    '2024-01-11', '2024-02-13', '2024-03-12', '2024-04-10',
    '2024-05-15', '2024-06-12', '2024-07-11', '2024-08-14',
    '2024-09-11', '2024-10-10', '2024-11-13', '2024-12-11',
    # 2025
    '2025-01-15', '2025-02-12', '2025-03-12', '2025-04-10',
    '2025-05-13', '2025-06-11', '2025-07-15', '2025-08-12',
    '2025-09-10', '2025-10-14', '2025-11-12', '2025-12-10',
    # 2026 (confirmed from BLS schedule: https://www.bls.gov/schedule/news_release/cpi.htm)
    '2026-01-14', '2026-02-11', '2026-03-11', '2026-04-10',
    '2026-05-12', '2026-06-10', '2026-07-14', '2026-08-12',
    '2026-09-10', '2026-10-13', '2026-11-10', '2026-12-09',
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


def is_macro_release_day(date_str=None):
    """Check if today is any macro data release day (PPI or CPI)."""
    if date_str is None:
        date_str = datetime.utcnow().strftime('%Y-%m-%d')
    return date_str in PPI_RELEASE_DATES or date_str in CPI_RELEASE_DATES


def get_release_type(date_str):
    """Determine what macro data is released on a given date.

    Returns: 'PPI', 'CPI', 'BOTH', or None
    """
    is_ppi = date_str in PPI_RELEASE_DATES
    is_cpi = date_str in CPI_RELEASE_DATES
    if is_ppi and is_cpi:
        return 'BOTH'
    elif is_ppi:
        return 'PPI'
    elif is_cpi:
        return 'CPI'
    return None


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
    """Score macro data release dynamics (PPI + CPI) for session bias.

    Checks both PPI and CPI calendars. If both release on the same day,
    uses PPI (stronger signal) as primary.

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
        return 'SKIP', 0.5, {'regime': 'DISABLED'}

    if current_time is None:
        current_time = datetime.utcnow()

    today_str = current_time.strftime('%Y-%m-%d')
    yesterday_str = (current_time - timedelta(days=1)).strftime('%Y-%m-%d')

    # Check today and yesterday for both PPI and CPI
    # Priority: post-release (yesterday) first — it has actual data.
    # Then today's release day (prediction only).
    release_date = None
    release_type = None
    is_release_day = False
    is_post_release = False

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

    # Second pass: check today for prediction (if no yesterday data found)
    if release_date is None:
        today_rtype = get_release_type(today_str)
        if today_rtype is not None:
            release_date = today_str
            release_type = today_rtype
            is_release_day = True
            is_post_release = False

    # If yesterday was 'BOTH' (PPI+CPI same day), use combined stats
    # The type_strength modifier already handles BOTH = stronger signal

    if release_date is None:
        return 'SKIP', 0.5, {'regime': 'NO_RELEASE', 'reason': 'No PPI/CPI release today or yesterday'}

    # Get regime
    regime, fade_rate = classify_market_regime(cfg)

    # Compute US session data
    us_data = compute_us_session(df_15m, release_date)
    if us_data is None:
        return 'NO_DATA', 0.5, {'regime': regime, 'release_type': release_type,
                                 'reason': 'No US session data yet'}

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

    # UK prediction (available even before UK session starts, if Asia is done)
    uk_prediction = None
    if is_release_day and us_dir != 'FLAT':
        uk_prediction = _predict_uk_session(us_dir, None, regime, release_type)
    elif is_post_release and asia_data is not None and uk_data is None:
        uk_prediction = _predict_uk_session(us_dir, asia_data, regime, release_type)

    # Spike accuracy for this type+year
    year = current_time.year if is_release_day else int(release_date[:4])
    if release_type == 'PPI':
        spike_acc = SPIKE_ACCURACY_PPI.get(year, 0.73)
    elif release_type == 'CPI':
        spike_acc = SPIKE_ACCURACY_CPI.get(year, 0.66)
    else:  # BOTH
        spike_acc = SPIKE_ACCURACY.get(year, 0.70)

    # Type-specific strength modifier
    # PPI moves markets more → higher confidence; CPI is noisier
    type_strength = 1.0
    if release_type == 'CPI':
        type_strength = 0.75   # CPI has weaker directional signal
    elif release_type == 'BOTH':
        type_strength = 1.10   # Both releasing = stronger signal

    details = {
        'release_date': release_date,
        'release_type': release_type,
        'is_release_day': is_release_day,
        'is_post_release': is_post_release,
        'regime': regime,
        'fade_rate': fade_rate,
        'spike_accuracy': spike_acc,
        'type_strength': type_strength,
        'us_data': us_data,
        'spike_data': spike_data,
        'asia_data': asia_data,
        'uk_data': uk_data,
        'uk_prediction': uk_prediction,
        'us_direction': us_dir,
        'us_magnitude': us_magnitude,
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

        # Score: pattern-aware with historical context + UK confirmation
        score_adjust = 0.0
        if uk_analysis:
            score_adjust = uk_analysis.get('uk_confidence_boost', 0)

        if is_crash:
            # Genuine crash — continuation likely, not a buying opportunity
            score = max(0.20, 0.35 - 0.10 * type_strength + score_adjust)
            status = 'PASS'
        elif is_sweep_reversal:
            # Sweep-and-reverse: the strength of the reversal signal depends on
            # dump size and regime. Bigger dumps in rising-inflation regimes
            # have higher reversal probability (from 8-year study).
            if rev_prob >= 0.70:
                # High-probability reversal — strong contrarian signal
                score = max(0.25, 0.40 - 0.15 * type_strength + score_adjust)
            elif rev_prob >= 0.50:
                # Moderate reversal — reduce confidence in continuation
                score = max(0.30, 0.45 - 0.10 * type_strength + score_adjust)
            else:
                # Low reversal probability — continuation more likely
                score = max(0.35, 0.50 - 0.05 * type_strength + score_adjust)
            status = 'PASS'
        elif gap_held:
            score = min(0.80, 0.65 + 0.10 * type_strength + score_adjust)
            status = 'PASS'
        else:
            score = max(0.35, 0.45 - 0.05 * type_strength + score_adjust)
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
        details['score_reason'] = ', '.join(reason_parts)

    # ── Release day: predict Asia ──
    elif is_release_day:
        if us_dir == 'FLAT':
            details['asia_prediction'] = {
                'direction': 'NEUTRAL',
                'confidence': 'LOW',
                'reason': f'US move too small ({us_move:+.2f}%) — no edge',
            }
            return 'SKIP', 0.5, details

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

        # Adjust confidence by type strength
        if release_type == 'CPI' and confidence == 'HIGH':
            confidence = 'MEDIUM'  # CPI downgrade — weaker signal
        elif release_type == 'BOTH' and confidence == 'MEDIUM':
            confidence = 'HIGH'    # Both upgrade — stronger signal

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

        # Score based on confidence + type
        base_score = {'HIGH': 0.75, 'MEDIUM': 0.60, 'LOW': 0.50}.get(confidence, 0.50)
        score = min(0.85, base_score * type_strength)
        status = 'PASS'

        details['score_reason'] = (
            f'{release_type} release day: US {us_dir} {us_move:+.2f}%, '
            f'regime={regime}, bias={bias}, conf={confidence}'
            + (f', rev_prob={rev_prob:.0%}' if dump_size != 'NOT_DUMP' else '')
        )

    else:
        return 'SKIP', 0.5, {'regime': 'NO_RELEASE', 'reason': 'No release context'}

    return status, score, details


# ═══════════════════════════════════════════════════════════════
# FORMATTER
# ═══════════════════════════════════════════════════════════════

def format_m23(details):
    """Format M23 details for terminal output."""
    if not details or details.get('regime') in ('DISABLED', 'NO_PPI', 'NO_RELEASE', 'NO_DATA'):
        return ''

    lines = []
    regime = details.get('regime', '?')
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

    # Header with type icon
    type_icons = {'PPI': '🏭', 'CPI': '🛒', 'BOTH': '📊'}
    type_icon = type_icons.get(release_type, '📊')
    lines.append(f"\n  {type_icon} M23 {release_type} RELEASE: {release_date}")
    lines.append(f"    Regime: {regime}  (fade rate: {details.get('fade_rate', 0):.0%})  "
                 f"Spike accuracy: {spike_acc:.0%}")

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
        if direction == 'FADE' and confidence in ('HIGH', 'MEDIUM'):
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

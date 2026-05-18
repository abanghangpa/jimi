"""
M62: US Unemployment Rate Session Bias (Regime-Conditional)

On NFP release days (first Friday of month, 08:30 ET = 12:30 UTC EDT / 13:30 UTC EST),
the unemployment rate is released simultaneously with NFP payrolls.
This module captures the UNEMPLOYMENT RATE signal specifically (M37 handles NFP payrolls).

Backtested on 101 NFP releases (2018-2026) against ETH/USDT 15m data.

Key findings (24h return):
  RANGE + COMPRESSING + SURPRISE_BEAT:   +1.00% avg, 67% win, n=15  → LONG bias
  MARKUP + COMPRESSING + SURPRISE_MISS:  +1.44% avg, 100% win, n=3  → LONG bias
  MARKUP + COMPRESSING + SURPRISE_BEAT:  -3.60% avg, 67% win, n=3  → SHORT bias
  RANGE + COMPRESSING + NEUTRAL:         -0.56% avg, 35% win, n=17  → SHORT bias
  LEVEL_DANGER (>4.5%):                  +3.17% avg, 50% win, n=6   → LONG (contrarian)
  SAHM_TRIGGERED:                        -0.42% avg, 57% win, n=14  → SHORT bias

Transmission chain: London Morning→Midday→NY Pre-Open→NY Open→Overlap→NY AM
  73.1% → 74.3% → 75.2% → 83.2% → 90.1% direction persistence ✅

Key mechanism:
  - Unemployment rate is the Fed's dual mandate metric (along with inflation)
  - Rising unemployment = dovish Fed pivot = potentially bullish for crypto
  - Falling unemployment = hawkish Fed = potentially bearish for crypto
  - The relationship is REGIME-DEPENDENT (tightening vs easing)
  - Sahm Rule (3m avg rise ≥ 0.5pp from 12m low) = recession signal

Usage:
    from src.modules.m62_us_unemployment import score_m62_us_unemployment, format_m62
    status, score_adj, size_mult, details = score_m62_us_unemployment(
        wyckoff_phase='RANGE', vol_regime='COMPRESSING', direction='LONG')
"""

from datetime import datetime, timedelta
import json
import os


# ═══════════════════════════════════════════════════════════════
# NFP RELEASE DATES (first Friday of each month, 08:30 ET)
# Unemployment rate is released simultaneously with NFP
# ═══════════════════════════════════════════════════════════════

NFP_RELEASE_DATES = {
    '2018-01-05', '2018-02-02', '2018-03-09', '2018-04-06',
    '2018-05-04', '2018-06-01', '2018-07-06', '2018-08-03',
    '2018-09-07', '2018-10-05', '2018-11-02', '2018-12-07',
    '2019-01-04', '2019-02-01', '2019-03-08', '2019-04-05',
    '2019-05-03', '2019-06-07', '2019-07-05', '2019-08-02',
    '2019-09-06', '2019-10-04', '2019-11-01', '2019-12-06',
    '2020-01-10', '2020-02-07', '2020-03-06', '2020-04-03',
    '2020-05-08', '2020-06-05', '2020-07-02', '2020-08-07',
    '2020-09-04', '2020-10-02', '2020-11-06', '2020-12-04',
    '2021-01-08', '2021-02-05', '2021-03-05', '2021-04-02',
    '2021-05-07', '2021-06-04', '2021-07-02', '2021-08-06',
    '2021-09-03', '2021-10-08', '2021-11-05', '2021-12-03',
    '2022-01-07', '2022-02-04', '2022-03-04', '2022-04-01',
    '2022-05-06', '2022-06-03', '2022-07-08', '2022-08-05',
    '2022-09-02', '2022-10-07', '2022-11-04', '2022-12-02',
    '2023-01-06', '2023-02-03', '2023-03-10', '2023-04-07',
    '2023-05-05', '2023-06-02', '2023-07-07', '2023-08-04',
    '2023-09-01', '2023-10-06', '2023-11-03', '2023-12-08',
    '2024-01-05', '2024-02-02', '2024-03-08', '2024-04-05',
    '2024-05-03', '2024-06-07', '2024-07-05', '2024-08-02',
    '2024-09-06', '2024-10-04', '2024-11-01', '2024-12-06',
    '2025-01-10', '2025-02-07', '2025-03-07', '2025-04-04',
    '2025-05-02', '2025-06-06', '2025-07-03', '2025-08-01',
    '2025-09-05', '2025-10-03', '2025-11-07', '2025-12-05',
    '2026-01-09', '2026-02-06', '2026-03-06', '2026-04-03',
    '2026-05-01',
}

# Map NFP release date → data month (the month the unemployment data covers)
# NFP released first Friday covers PRIOR month's data
NFP_TO_DATA_MONTH = {}
for _d in sorted(NFP_RELEASE_DATES):
    _dt = datetime.strptime(_d, '%Y-%m-%d')
    _data_month = (_dt - timedelta(days=10)).strftime('%Y-%m')
    NFP_TO_DATA_MONTH[_d] = _data_month


# ═══════════════════════════════════════════════════════════════
# UNEMPLOYMENT RATE DATA (monthly, from BLS/FRED)
# ═══════════════════════════════════════════════════════════════

UNEMPLOYMENT_RATE_MONTHLY = {
    # 2017 (for prior-year lookback)
    '2017-01': 4.7, '2017-02': 4.6, '2017-03': 4.4, '2017-04': 4.4,
    '2017-05': 4.4, '2017-06': 4.3, '2017-07': 4.3, '2017-08': 4.3,
    '2017-09': 4.1, '2017-10': 4.1, '2017-11': 4.1, '2017-12': 4.1,
    # 2018
    '2018-01': 4.1, '2018-02': 4.1, '2018-03': 4.0, '2018-04': 3.9,
    '2018-05': 3.8, '2018-06': 4.0, '2018-07': 3.9, '2018-08': 3.8,
    '2018-09': 3.7, '2018-10': 3.8, '2018-11': 3.7, '2018-12': 3.9,
    # 2019
    '2019-01': 4.0, '2019-02': 3.8, '2019-03': 3.8, '2019-04': 3.6,
    '2019-05': 3.6, '2019-06': 3.6, '2019-07': 3.7, '2019-08': 3.7,
    '2019-09': 3.5, '2019-10': 3.6, '2019-11': 3.5, '2019-12': 3.6,
    # 2020
    '2020-01': 3.6, '2020-02': 3.5, '2020-03': 4.4, '2020-04': 14.7,
    '2020-05': 13.2, '2020-06': 11.0, '2020-07': 10.2, '2020-08': 8.4,
    '2020-09': 7.8, '2020-10': 6.9, '2020-11': 6.7, '2020-12': 6.7,
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

# Consensus expectations (approximate)
UNEMPLOYMENT_CONSENSUS = {
    '2018-01': 4.1, '2018-02': 4.1, '2018-03': 4.0, '2018-04': 4.0,
    '2018-05': 3.9, '2018-06': 3.9, '2018-07': 3.9, '2018-08': 3.8,
    '2018-09': 3.8, '2018-10': 3.7, '2018-11': 3.7, '2018-12': 3.7,
    '2019-01': 3.9, '2019-02': 3.9, '2019-03': 3.8, '2019-04': 3.8,
    '2019-05': 3.6, '2019-06': 3.6, '2019-07': 3.6, '2019-08': 3.7,
    '2019-09': 3.7, '2019-10': 3.6, '2019-11': 3.6, '2019-12': 3.5,
    '2020-01': 3.5, '2020-02': 3.6, '2020-03': 3.7, '2020-04': 14.0,
    '2020-05': 19.5, '2020-06': 12.3, '2020-07': 10.5, '2020-08': 9.8,
    '2020-09': 8.2, '2020-10': 7.7, '2020-11': 6.8, '2020-12': 6.7,
    '2021-01': 6.7, '2021-02': 6.3, '2021-03': 6.0, '2021-04': 5.8,
    '2021-05': 5.9, '2021-06': 5.7, '2021-07': 5.7, '2021-08': 5.2,
    '2021-09': 5.1, '2021-10': 4.7, '2021-11': 4.5, '2021-12': 4.1,
    '2022-01': 3.9, '2022-02': 3.9, '2022-03': 3.7, '2022-04': 3.6,
    '2022-05': 3.5, '2022-06': 3.6, '2022-07': 3.6, '2022-08': 3.5,
    '2022-09': 3.7, '2022-10': 3.6, '2022-11': 3.7, '2022-12': 3.7,
    '2023-01': 3.6, '2023-02': 3.4, '2023-03': 3.6, '2023-04': 3.6,
    '2023-05': 3.5, '2023-06': 3.7, '2023-07': 3.6, '2023-08': 3.5,
    '2023-09': 3.7, '2023-10': 3.8, '2023-11': 3.9, '2023-12': 3.8,
    '2024-01': 3.8, '2024-02': 3.7, '2024-03': 3.9, '2024-04': 3.8,
    '2024-05': 3.9, '2024-06': 4.0, '2024-07': 4.1, '2024-08': 4.2,
    '2024-09': 4.2, '2024-10': 4.2, '2024-11': 4.1, '2024-12': 4.2,
    '2025-01': 4.1, '2025-02': 4.0, '2025-03': 4.1, '2025-04': 4.2,
    '2025-05': 4.2, '2025-06': 4.1, '2025-07': 4.2, '2025-08': 4.2,
    '2025-09': 4.3, '2025-10': 4.3, '2025-11': 4.3, '2025-12': 4.4,
    '2026-01': 4.4, '2026-02': 4.3, '2026-03': 4.3, '2026-04': 4.3,
    '2026-05': 4.3,
}


# ═══════════════════════════════════════════════════════════════
# REGIME-CONDITIONAL EDGE TABLE
# Backtested: 101 NFP releases, 2018-2026, ETH/USDT 15m
# ═══════════════════════════════════════════════════════════════

# Fine-grained: (wyckoff_phase, vol_regime, signal)
EDGE_TABLE = {
    # ── SHORT EDGES ──
    ('MARKUP', 'COMPRESSING', 'SURPRISE_BEAT'):  {'avg_ret': -3.60, 'win': 0.67, 'n': 3, 'bias': 'SHORT'},
    ('MARKUP', 'LOW_VOL', 'NEUTRAL'):            {'avg_ret': -0.75, 'win': 0.00, 'n': 3, 'bias': 'SHORT'},
    ('RANGE', 'COMPRESSING', 'NEUTRAL'):          {'avg_ret': -0.56, 'win': 0.35, 'n': 17, 'bias': 'SHORT'},

    # ── LONG EDGES ──
    ('MARKUP', 'COMPRESSING', 'SURPRISE_MISS'):  {'avg_ret': +1.44, 'win': 1.00, 'n': 3, 'bias': 'LONG'},
    ('MARKUP', 'LOW_VOL', 'SURPRISE_BEAT'):      {'avg_ret': +0.71, 'win': 0.67, 'n': 3, 'bias': 'LONG'},
    ('RANGE', 'COMPRESSING', 'SURPRISE_BEAT'):    {'avg_ret': +1.00, 'win': 0.67, 'n': 15, 'bias': 'LONG'},
}

# Broad: (wyckoff_phase, signal) — ignoring vol
BROAD_EDGE_TABLE = {
    # ── SHORT EDGES ──
    ('MARKUP', 'NEUTRAL'):          {'avg_ret': -0.99, 'win': 0.20, 'n': 5, 'bias': 'SHORT'},
    ('MARKUP', 'SURPRISE_BEAT'):    {'avg_ret': -1.45, 'win': 0.67, 'n': 6, 'bias': 'SHORT'},
    ('RANGE', 'NEUTRAL'):           {'avg_ret': -0.77, 'win': 0.33, 'n': 18, 'bias': 'SHORT'},

    # ── LONG EDGES ──
    ('CHOP', 'NEUTRAL'):            {'avg_ret': +2.18, 'win': 1.00, 'n': 5, 'bias': 'LONG'},
    ('MARKDOWN', 'NEUTRAL'):        {'avg_ret': +2.47, 'win': 0.67, 'n': 3, 'bias': 'LONG'},
    ('MARKDOWN', 'SAHM_TRIGGERED'): {'avg_ret': +1.20, 'win': 0.67, 'n': 3, 'bias': 'LONG'},
    ('MARKDOWN', 'SURPRISE_MISS'):  {'avg_ret': +1.40, 'win': 1.00, 'n': 3, 'bias': 'LONG'},
    ('MARKUP', 'SURPRISE_MISS'):    {'avg_ret': +2.00, 'win': 0.80, 'n': 5, 'bias': 'LONG'},
    ('RANGE', 'LEVEL_DANGER'):      {'avg_ret': +3.69, 'win': 0.33, 'n': 3, 'bias': 'LONG'},
    ('RANGE', 'SAHM_TRIGGERED'):    {'avg_ret': +0.46, 'win': 0.63, 'n': 8, 'bias': 'LONG'},
    ('RANGE', 'SURPRISE_BEAT'):     {'avg_ret': +1.13, 'win': 0.69, 'n': 16, 'bias': 'LONG'},
}

# Signal-only: signal → stats (ignoring wyckoff and vol)
SIGNAL_EDGE_TABLE = {
    'LEVEL_DANGER':     {'avg_ret': +3.17, 'win': 0.50, 'n': 6, 'bias': 'LONG'},
    'SAHM_TRIGGERED':   {'avg_ret': -0.42, 'win': 0.57, 'n': 14, 'bias': 'SHORT'},
    'SURPRISE_BEAT':    {'avg_ret': +0.51, 'win': 0.68, 'n': 25, 'bias': 'LONG'},
    'SURPRISE_MISS':    {'avg_ret': +0.69, 'win': 0.65, 'n': 23, 'bias': 'LONG'},
}


# ═══════════════════════════════════════════════════════════════
# CLASSIFICATION HELPERS
# ═══════════════════════════════════════════════════════════════

def _classify_level(rate):
    """Classify unemployment rate level."""
    if rate < 3.5:
        return 'GOLDILOCKS'
    elif rate < 4.0:
        return 'NORMAL'
    elif rate < 4.5:
        return 'SOFTENING'
    elif rate < 6.0:
        return 'DANGER'
    else:
        return 'CRISIS'


def _classify_change(current, prior):
    """Classify change from prior month."""
    delta = current - prior
    if delta >= 0.2:
        return 'RISING', delta
    elif delta <= -0.2:
        return 'FALLING', delta
    else:
        return 'STABLE', delta


def _classify_surprise(actual, consensus):
    """Classify surprise vs consensus. Lower unemployment = BEAT."""
    diff = actual - consensus
    if diff < -0.1:
        return 'BEAT', diff
    elif diff > 0.1:
        return 'MISS', diff
    else:
        return 'INLINE', diff


def _check_sahm(unemp_dict, release_month_key):
    """Check if Sahm Rule is triggered.

    Sahm Rule: 3-month avg unemployment - 12-month low >= 0.5pp
    """
    sorted_keys = sorted(k for k in unemp_dict.keys() if k <= release_month_key)
    if len(sorted_keys) < 12:
        return False, 0.0

    last_3 = [unemp_dict[k] for k in sorted_keys[-3:]]
    avg_3m = sum(last_3) / 3

    last_12 = [unemp_dict[k] for k in sorted_keys[-12:]]
    low_12m = min(last_12)

    sahm_value = avg_3m - low_12m
    return sahm_value >= 0.5, round(sahm_value, 3)


def _build_signal(level, change, surprise, sahm_triggered):
    """Build composite signal from unemployment classifications.

    Priority: Sahm > Level > Change > Surprise
    """
    if sahm_triggered:
        return 'SAHM_TRIGGERED'
    elif level in ('CRISIS', 'DANGER'):
        return f'LEVEL_{level}'
    elif change in ('RISING', 'FALLING'):
        return f'CHANGE_{change}'
    elif surprise in ('BEAT', 'MISS'):
        return f'SURPRISE_{surprise}'
    else:
        return 'NEUTRAL'


def _get_latest_unemp_data(today_str=None):
    """Get the latest unemployment data available.

    Returns: (actual_rate, consensus, prior_rate, data_month_key, release_date)
    """
    if today_str is None:
        today_str = datetime.utcnow().strftime('%Y-%m-%d')

    today = datetime.strptime(today_str, '%Y-%m-%d')

    # Find the most recent NFP release that's on or before today
    latest_release = None
    for release_date in sorted(NFP_RELEASE_DATES, reverse=True):
        release_dt = datetime.strptime(release_date, '%Y-%m-%d')
        if release_dt <= today:
            latest_release = release_date
            break

    if latest_release is None:
        return None

    data_month = NFP_TO_DATA_MONTH.get(latest_release)
    if data_month is None or data_month not in UNEMPLOYMENT_RATE_MONTHLY:
        return None

    actual = UNEMPLOYMENT_RATE_MONTHLY[data_month]
    consensus = UNEMPLOYMENT_CONSENSUS.get(data_month, actual)

    # Prior month
    dt = datetime.strptime(data_month + '-01', '%Y-%m-%d')
    prior_dt = dt - timedelta(days=15)
    prior_month = prior_dt.strftime('%Y-%m')
    prior = UNEMPLOYMENT_RATE_MONTHLY.get(prior_month)

    return {
        'actual': actual,
        'consensus': consensus,
        'prior': prior,
        'data_month': data_month,
        'release_date': latest_release,
    }


def _is_nfp_release_day(today_str=None, window_days=1):
    """Check if today is within N days of an NFP release.

    Args:
        today_str: YYYY-MM-DD string (default: today UTC)
        window_days: number of days after release to consider active

    Returns:
        (is_release: bool, release_date: str, data_month: str) or (False, None, None)
    """
    if today_str is None:
        today_str = datetime.utcnow().strftime('%Y-%m-%d')

    today = datetime.strptime(today_str, '%Y-%m-%d')

    for release_date_str in sorted(NFP_RELEASE_DATES, reverse=True):
        release_dt = datetime.strptime(release_date_str, '%Y-%m-%d')
        days_since = (today - release_dt).days
        if 0 <= days_since <= window_days:
            data_month = NFP_TO_DATA_MONTH.get(release_date_str)
            return True, release_date_str, data_month

    return False, None, None


# ═══════════════════════════════════════════════════════════════
# MAIN SCORING FUNCTION
# ═══════════════════════════════════════════════════════════════

def score_m62_us_unemployment(wyckoff_phase='RANGE', vol_regime='CHOP',
                               direction='LONG', today_str=None, config=None):
    """Score the US Unemployment Rate session bias.

    Args:
        wyckoff_phase: from M21 ('ACCUMULATION', 'MARKUP', 'DISTRIBUTION', 'MARKDOWN', 'RANGE')
        vol_regime: from M9 ('TREND', 'SQUEEZE', 'CHOP', 'LOW_VOL', 'COMPRESSING')
        direction: trade direction ('LONG' or 'SHORT')
        today_str: YYYY-MM-DD override (for backtesting)
        config: config dict (optional)

    Returns:
        status: 'PASS' (active), 'SKIP' (not release day), or 'WEAK' (low confidence)
        score_adj: score adjustment (-0.10 to +0.10)
        size_mult: position size multiplier (0.5 to 1.0)
        details: dict
    """
    cfg = config or {}

    if not cfg.get('M62_ENABLED', True):
        return 'SKIP', 0.0, 1.0, {'regime': 'DISABLED'}

    # Check if today is an NFP release day
    is_release, release_date, data_month = _is_nfp_release_day(
        today_str, window_days=cfg.get('M62_WINDOW_DAYS', 1))

    if not is_release:
        return 'SKIP', 0.0, 1.0, {'regime': 'NOT_RELEASE_DAY'}

    if data_month is None or data_month not in UNEMPLOYMENT_RATE_MONTHLY:
        return 'SKIP', 0.0, 1.0, {'regime': 'NO_DATA'}

    actual_rate = UNEMPLOYMENT_RATE_MONTHLY[data_month]
    consensus = UNEMPLOYMENT_CONSENSUS.get(data_month, actual_rate)

    # Prior month
    dt = datetime.strptime(data_month + '-01', '%Y-%m-%d')
    prior_dt = dt - timedelta(days=15)
    prior_month = prior_dt.strftime('%Y-%m')
    prior_rate = UNEMPLOYMENT_RATE_MONTHLY.get(prior_month)

    if prior_rate is None:
        return 'SKIP', 0.0, 1.0, {'regime': 'NO_PRIOR_DATA'}

    # Classify
    level = _classify_level(actual_rate)
    change, delta = _classify_change(actual_rate, prior_rate)
    surprise, surprise_diff = _classify_surprise(actual_rate, consensus)
    sahm_triggered, sahm_value = _check_sahm(UNEMPLOYMENT_RATE_MONTHLY, data_month)

    # Build composite signal
    signal = _build_signal(level, change, surprise, sahm_triggered)

    # ── Lookup 1: Fine-grained (Wyckoff × Vol × Signal) ──
    fine_key = (wyckoff_phase, vol_regime, signal)
    fine_match = EDGE_TABLE.get(fine_key)

    # ── Lookup 2: Broad (Wyckoff × Signal) ──
    broad_key = (wyckoff_phase, signal)
    broad_match = BROAD_EDGE_TABLE.get(broad_key)

    # ── Lookup 3: Signal-only ──
    signal_match = SIGNAL_EDGE_TABLE.get(signal)

    # ── Determine best signal ──
    # Priority: fine > broad > signal-only
    best_match = None
    best_source = 'NONE'
    confidence = 0.0

    if fine_match and fine_match['n'] >= 3:
        best_match = fine_match
        best_source = 'FINE'
        confidence = min(1.0, fine_match['n'] / 10)
    elif broad_match and broad_match['n'] >= 5:
        best_match = broad_match
        best_source = 'BROAD'
        confidence = min(1.0, broad_match['n'] / 15)
    elif signal_match and signal_match['n'] >= 5:
        best_match = signal_match
        best_source = 'SIGNAL_ONLY'
        confidence = min(1.0, signal_match['n'] / 15)

    if best_match is None:
        return 'SKIP', 0.0, 1.0, {
            'regime': 'NO_EDGE',
            'wyckoff': wyckoff_phase,
            'vol_regime': vol_regime,
            'signal': signal,
            'level': level,
            'change': change,
            'surprise': surprise,
            'sahm_triggered': sahm_triggered,
            'actual_rate': actual_rate,
            'consensus': consensus,
            'release_date': release_date,
        }

    # ── Compute score adjustment ──
    avg_ret = best_match['avg_ret']
    win_rate = best_match['win']
    n = best_match['n']
    bias = best_match['bias']

    abs_ret = abs(avg_ret)
    if abs_ret >= 3.0:
        raw_adj = 0.10
    elif abs_ret >= 1.5:
        raw_adj = 0.08
    elif abs_ret >= 0.8:
        raw_adj = 0.07
    else:
        raw_adj = 0.05

    # Scale by confidence (sample size)
    score_adj = raw_adj * confidence

    # Apply direction
    if bias == 'LONG' and direction == 'LONG':
        score_adj = abs(score_adj)
    elif bias == 'LONG' and direction == 'SHORT':
        score_adj = -abs(score_adj)
    elif bias == 'SHORT' and direction == 'SHORT':
        score_adj = abs(score_adj)
    elif bias == 'SHORT' and direction == 'LONG':
        score_adj = -abs(score_adj)

    # ── Size multiplier ──
    if confidence >= 0.7:
        size_mult = 1.0
    elif confidence >= 0.4:
        size_mult = 0.75
    else:
        size_mult = 0.50

    # Extra reduction if n < 5 (very small sample)
    if n < 5:
        size_mult *= 0.75

    # Sahm rule override: reduce size when Sahm is triggered (regime uncertainty)
    if sahm_triggered:
        size_mult *= 0.80

    size_mult = round(size_mult, 2)
    score_adj = round(score_adj, 3)

    status = 'PASS' if confidence >= 0.3 else 'WEAK'

    details = {
        'regime': f'UNEMP_{bias}',
        'release_date': release_date,
        'data_month': data_month,
        'actual_rate': actual_rate,
        'consensus': consensus,
        'prior_rate': prior_rate,
        'delta': round(delta, 2),
        'level': level,
        'change': change,
        'surprise': surprise,
        'surprise_diff': round(surprise_diff, 2),
        'sahm_triggered': sahm_triggered,
        'sahm_value': sahm_value,
        'signal': signal,
        'wyckoff': wyckoff_phase,
        'vol_regime': vol_regime,
        'bias': bias,
        'avg_ret_24h': avg_ret,
        'win_rate': win_rate,
        'sample_size': n,
        'confidence': round(confidence, 2),
        'source': best_source,
        'score_adj': score_adj,
        'size_mult': size_mult,
    }

    return status, score_adj, size_mult, details


def format_m62(details):
    """Format M62 details for terminal output."""
    if not details or details.get('regime') in ('DISABLED', 'NOT_RELEASE_DAY', 'NO_EDGE', 'NO_DATA'):
        regime = details.get('regime', '?') if details else '?'
        if regime == 'NOT_RELEASE_DAY':
            return ''
        return ''

    bias = details.get('bias', '?')
    actual = details.get('actual_rate', 0)
    consensus = details.get('consensus', 0)
    prior = details.get('prior_rate', 0)
    delta = details.get('delta', 0)
    level = details.get('level', '?')
    change = details.get('change', '?')
    surprise = details.get('surprise', '?')
    sahm = details.get('sahm_triggered', False)
    sahm_val = details.get('sahm_value', 0)
    signal = details.get('signal', '?')
    wyckoff = details.get('wyckoff', '?')
    vol = details.get('vol_regime', '?')
    avg_ret = details.get('avg_ret_24h', 0)
    win = details.get('win_rate', 0)
    n = details.get('sample_size', 0)
    conf = details.get('confidence', 0)
    source = details.get('source', '?')
    score_adj = details.get('score_adj', 0)
    size_mult = details.get('size_mult', 1.0)
    release = details.get('release_date', '?')

    icon = '🟢' if bias == 'LONG' else '🔴' if bias == 'SHORT' else '⚪'
    conf_icon = '🟢' if conf >= 0.7 else '🟡' if conf >= 0.4 else '🟠'
    level_icons = {
        'GOLDILOCKS': '🟢', 'NORMAL': '⚪', 'SOFTENING': '🟡',
        'DANGER': '🟠', 'CRISIS': '🔴'
    }
    lvl_icon = level_icons.get(level, '⚪')
    sahm_icon = '🔴 TRIGGERED' if sahm else '🟢 ok'
    surp_icon = '🟢' if surprise == 'BEAT' else '🔴' if surprise == 'MISS' else '⚪'
    chg_icon = '📈' if change == 'RISING' else '📉' if change == 'FALLING' else '➡️'

    lines = []
    lines.append(f"\n  {icon} M62 US UNEMPLOYMENT RATE: {bias}")
    lines.append(f"    Release: {release}  |  Rate: {actual:.1f}%  |  Consensus: {consensus:.1f}%  |  Prior: {prior:.1f}%")
    lines.append(f"    {lvl_icon} Level: {level}  |  {chg_icon} Change: {change} ({delta:+.1f}pp)  |  {surp_icon} Surprise: {surprise}")
    lines.append(f"    Sahm Rule: {sahm_icon}  (value: {sahm_val:.3f}pp)")
    lines.append(f"    Context: {wyckoff} + {vol}  |  Signal: {signal}")
    lines.append(f"    Backtest: avg 24h={avg_ret:+.2f}%  win={win*100:.0f}%  n={n}  source={source}")
    lines.append(f"    Chain: London→NY 73-90% direction persistence")
    lines.append(f"    {conf_icon} Confidence: {conf:.2f}  |  Score adj: {score_adj:+.3f}  |  Size: {size_mult:.2f}x")

    return '\n'.join(lines)


# ═══════════════════════════════════════════════════════════════
# CACHE FUNCTIONS (for live updates from FRED)
# ═══════════════════════════════════════════════════════════════

def get_unemp_cache_path():
    """Get path to unemployment rate cache (for live updates)."""
    cache_dir = os.path.join(
        os.path.dirname(os.path.dirname(os.path.dirname(__file__))),
        'data', 'fred'
    )
    return os.path.join(cache_dir, 'unemployment_cache.json')


def load_unemp_cache():
    """Load cached unemployment data (for live updates from FRED)."""
    cache_path = get_unemp_cache_path()
    if os.path.exists(cache_path):
        try:
            with open(cache_path) as f:
                return json.load(f)
        except (json.JSONDecodeError, IOError):
            pass
    return {}


def update_unemp_cache(actual_rate, consensus=None, data_month=None):
    """Update unemployment cache with new data (called from macro_fetch)."""
    cache = load_unemp_cache()
    if data_month is None:
        data_month = datetime.utcnow().strftime('%Y-%m')
    cache[data_month] = {
        'actual': actual_rate,
        'consensus': consensus,
        'updated': datetime.utcnow().isoformat(),
    }
    cache_path = get_unemp_cache_path()
    os.makedirs(os.path.dirname(cache_path), exist_ok=True)
    with open(cache_path, 'w') as f:
        json.dump(cache, f, indent=2)
    return cache

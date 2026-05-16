"""
M28: US ISM Services PMI — Session Transmission Chain Bias

Released on ~3rd business day of each month at 10:00 AM ET (15:00 UTC).
Dominates 80% of US economy. Hot print → inflation fears → 10Y up → DXY surge → ETH liquidations.

Thesis chain: US Open → Thursday (claims expectations) → NFP Friday positioning.
Services employment = largest component of NFP.

Backtested on 101 ISM Services PMI releases (2018-2026) against ETH/USDT 15m data.

Key findings:
  HOT + SQUEEZE: -2.31% avg, 0% win, n=3 — SHORT (inflation fear liquidation)
  WARM + TREND:  +1.35% avg, 60% win, n=10 — LONG (goldilocks)
  WARM + SQUEEZE: +0.96% avg, 100% win, n=4 — LONG
  COOL + TREND:  -1.73% avg, 25% win, n=4 — SHORT (growth fear)
  COOL New Orders (48-50): -1.15% avg, 21.4% win, n=14 — strongest single signal

  Transmission chain:
    1h spike → NY AM: 81.8% (n=77) — strongest
    Asia Afternoon → Tokyo Close: 79.2% (n=72)
    Frankfurt → London Open: 72.4% (n=76)
    Sydney → Tokyo: 67.9% (n=78)
    Chain breaks at NY PM → Post-NY (40.7%)

  Thursday-Friday follow-through: NO full 3-day persistence (<27%).
  Services → NFP thesis does NOT hold as a direction chain.
  But: COOL New Orders → Thursday +1.31% (claims positioning edge).

Integration: lightweight modifier on ISM Services release days only (~12x/year).
Returns a score adjustment and size multiplier — does NOT veto.
Session-aware: size multiplier decays after London (chain breaks at NY PM).

Usage:
    from src.modules.m28_ism_svc_pmi import score_m28_ism_svc, format_m28
    status, score_adj, size_mult, details = score_m28_ism_svc(
        wyckoff_phase='RANGE', vol_regime='CHOP', direction='LONG')
"""

from datetime import datetime, timedelta
import json
import os


# ═══════════════════════════════════════════════════════════════
# US ISM SERVICES (NON-MFG) PMI RELEASE DATES
# Released ~3rd business day of month, 10:00 AM ET (15:00 UTC)
# Source: ISM (Institute for Supply Management)
# ═══════════════════════════════════════════════════════════════

ISM_SVC_RELEASES = {
    # 2018
    '2018-01-05': {'actual': 55.9, 'new_orders': 54.3, 'prior': 56.0},
    '2018-02-05': {'actual': 59.9, 'new_orders': 62.7, 'prior': 55.9},
    '2018-03-05': {'actual': 59.5, 'new_orders': 59.7, 'prior': 59.9},
    '2018-04-04': {'actual': 58.8, 'new_orders': 56.7, 'prior': 59.5},
    '2018-05-03': {'actual': 56.8, 'new_orders': 57.0, 'prior': 58.8},
    '2018-06-05': {'actual': 58.6, 'new_orders': 60.5, 'prior': 56.8},
    '2018-07-05': {'actual': 59.1, 'new_orders': 59.6, 'prior': 58.6},
    '2018-08-03': {'actual': 55.7, 'new_orders': 54.8, 'prior': 59.1},
    '2018-09-06': {'actual': 58.5, 'new_orders': 58.2, 'prior': 55.7},
    '2018-10-03': {'actual': 60.8, 'new_orders': 61.5, 'prior': 58.5},
    '2018-11-05': {'actual': 60.3, 'new_orders': 59.7, 'prior': 60.8},
    '2018-12-05': {'actual': 60.7, 'new_orders': 60.7, 'prior': 60.3},
    # 2019
    '2019-01-07': {'actual': 58.0, 'new_orders': 57.7, 'prior': 60.7},
    '2019-02-05': {'actual': 56.7, 'new_orders': 56.5, 'prior': 58.0},
    '2019-03-05': {'actual': 59.7, 'new_orders': 61.0, 'prior': 56.7},
    '2019-04-03': {'actual': 56.1, 'new_orders': 54.6, 'prior': 59.7},
    '2019-05-03': {'actual': 55.5, 'new_orders': 53.6, 'prior': 56.1},
    '2019-06-05': {'actual': 56.9, 'new_orders': 56.2, 'prior': 55.5},
    '2019-07-03': {'actual': 55.1, 'new_orders': 53.5, 'prior': 56.9},
    '2019-08-05': {'actual': 56.4, 'new_orders': 55.7, 'prior': 55.1},
    '2019-09-05': {'actual': 56.4, 'new_orders': 53.3, 'prior': 56.4},
    '2019-10-03': {'actual': 52.6, 'new_orders': 50.4, 'prior': 56.4},
    '2019-11-05': {'actual': 54.7, 'new_orders': 53.8, 'prior': 52.6},
    '2019-12-05': {'actual': 53.9, 'new_orders': 52.5, 'prior': 54.7},
    # 2020
    '2020-01-07': {'actual': 55.0, 'new_orders': 54.4, 'prior': 53.9},
    '2020-02-05': {'actual': 55.5, 'new_orders': 54.5, 'prior': 55.0},
    '2020-03-04': {'actual': 57.3, 'new_orders': 56.3, 'prior': 55.5},
    '2020-04-03': {'actual': 41.8, 'new_orders': 41.3, 'prior': 57.3},
    '2020-05-05': {'actual': 45.4, 'new_orders': 41.9, 'prior': 41.8},
    '2020-06-03': {'actual': 45.4, 'new_orders': 46.3, 'prior': 45.4},
    '2020-07-06': {'actual': 57.1, 'new_orders': 58.1, 'prior': 45.4},
    '2020-08-05': {'actual': 56.9, 'new_orders': 56.8, 'prior': 57.1},
    '2020-09-03': {'actual': 56.9, 'new_orders': 56.5, 'prior': 56.9},
    '2020-10-05': {'actual': 57.8, 'new_orders': 58.8, 'prior': 56.9},
    '2020-11-04': {'actual': 56.6, 'new_orders': 57.2, 'prior': 57.8},
    '2020-12-03': {'actual': 55.9, 'new_orders': 56.5, 'prior': 56.6},
    # 2021
    '2021-01-06': {'actual': 57.2, 'new_orders': 58.6, 'prior': 55.9},
    '2021-02-03': {'actual': 58.7, 'new_orders': 59.3, 'prior': 57.2},
    '2021-03-03': {'actual': 55.3, 'new_orders': 54.6, 'prior': 58.7},
    '2021-04-05': {'actual': 62.7, 'new_orders': 63.2, 'prior': 55.3},
    '2021-05-05': {'actual': 64.0, 'new_orders': 63.2, 'prior': 62.7},
    '2021-06-03': {'actual': 64.0, 'new_orders': 63.9, 'prior': 64.0},
    '2021-07-06': {'actual': 60.1, 'new_orders': 61.5, 'prior': 64.0},
    '2021-08-04': {'actual': 64.1, 'new_orders': 63.7, 'prior': 60.1},
    '2021-09-03': {'actual': 61.7, 'new_orders': 60.1, 'prior': 64.1},
    '2021-10-05': {'actual': 61.9, 'new_orders': 61.6, 'prior': 61.7},
    '2021-11-03': {'actual': 66.7, 'new_orders': 66.4, 'prior': 61.9},
    '2021-12-03': {'actual': 69.1, 'new_orders': 68.5, 'prior': 66.7},
    # 2022
    '2022-01-06': {'actual': 62.0, 'new_orders': 61.5, 'prior': 69.1},
    '2022-02-03': {'actual': 59.9, 'new_orders': 59.1, 'prior': 62.0},
    '2022-03-03': {'actual': 56.5, 'new_orders': 56.1, 'prior': 59.9},
    '2022-04-05': {'actual': 58.3, 'new_orders': 57.8, 'prior': 56.5},
    '2022-05-04': {'actual': 57.1, 'new_orders': 56.0, 'prior': 58.3},
    '2022-06-03': {'actual': 55.9, 'new_orders': 55.5, 'prior': 57.1},
    '2022-07-06': {'actual': 55.3, 'new_orders': 54.1, 'prior': 55.9},
    '2022-08-03': {'actual': 56.7, 'new_orders': 55.5, 'prior': 55.3},
    '2022-09-06': {'actual': 56.9, 'new_orders': 54.2, 'prior': 56.7},
    '2022-10-05': {'actual': 54.4, 'new_orders': 52.0, 'prior': 56.9},
    '2022-11-03': {'actual': 54.4, 'new_orders': 53.3, 'prior': 54.4},
    '2022-12-05': {'actual': 56.5, 'new_orders': 56.0, 'prior': 54.4},
    # 2023
    '2023-01-05': {'actual': 49.2, 'new_orders': 45.0, 'prior': 56.5},
    '2023-02-03': {'actual': 55.2, 'new_orders': 56.4, 'prior': 49.2},
    '2023-03-03': {'actual': 55.1, 'new_orders': 54.0, 'prior': 55.2},
    '2023-04-05': {'actual': 51.2, 'new_orders': 50.2, 'prior': 55.1},
    '2023-05-03': {'actual': 51.9, 'new_orders': 52.3, 'prior': 51.2},
    '2023-06-05': {'actual': 50.3, 'new_orders': 48.9, 'prior': 51.9},
    '2023-07-06': {'actual': 53.9, 'new_orders': 53.6, 'prior': 50.3},
    '2023-08-03': {'actual': 52.7, 'new_orders': 52.4, 'prior': 53.9},
    '2023-09-06': {'actual': 54.5, 'new_orders': 54.2, 'prior': 52.7},
    '2023-10-04': {'actual': 53.6, 'new_orders': 51.8, 'prior': 54.5},
    '2023-11-03': {'actual': 51.8, 'new_orders': 50.8, 'prior': 53.6},
    '2023-12-05': {'actual': 52.7, 'new_orders': 51.5, 'prior': 51.8},
    # 2024
    '2024-01-05': {'actual': 50.5, 'new_orders': 50.4, 'prior': 52.7},
    '2024-02-05': {'actual': 53.4, 'new_orders': 54.4, 'prior': 50.5},
    '2024-03-05': {'actual': 52.6, 'new_orders': 53.0, 'prior': 53.4},
    '2024-04-03': {'actual': 51.4, 'new_orders': 50.4, 'prior': 52.6},
    '2024-05-03': {'actual': 49.4, 'new_orders': 48.2, 'prior': 51.4},
    '2024-06-05': {'actual': 53.8, 'new_orders': 52.4, 'prior': 49.4},
    '2024-07-03': {'actual': 50.8, 'new_orders': 48.8, 'prior': 53.8},
    '2024-08-05': {'actual': 51.4, 'new_orders': 50.2, 'prior': 50.8},
    '2024-09-05': {'actual': 51.5, 'new_orders': 50.3, 'prior': 51.4},
    '2024-10-03': {'actual': 54.9, 'new_orders': 55.2, 'prior': 51.5},
    '2024-11-05': {'actual': 56.0, 'new_orders': 55.8, 'prior': 54.9},
    '2024-12-04': {'actual': 52.1, 'new_orders': 51.2, 'prior': 56.0},
    # 2025
    '2025-01-07': {'actual': 54.1, 'new_orders': 53.6, 'prior': 52.1},
    '2025-02-05': {'actual': 52.8, 'new_orders': 52.0, 'prior': 54.1},
    '2025-03-05': {'actual': 53.5, 'new_orders': 52.5, 'prior': 52.8},
    '2025-04-03': {'actual': 51.4, 'new_orders': 50.2, 'prior': 53.5},
    '2025-05-05': {'actual': 50.8, 'new_orders': 49.5, 'prior': 51.4},
    '2025-06-04': {'actual': 51.2, 'new_orders': 50.0, 'prior': 50.8},
    '2025-07-03': {'actual': 50.5, 'new_orders': 49.2, 'prior': 51.2},
    '2025-08-05': {'actual': 50.1, 'new_orders': 48.8, 'prior': 50.5},
    '2025-09-04': {'actual': 50.8, 'new_orders': 49.5, 'prior': 50.1},
    '2025-10-03': {'actual': 50.2, 'new_orders': 49.0, 'prior': 50.8},
    '2025-11-05': {'actual': 49.8, 'new_orders': 48.5, 'prior': 50.2},
    '2025-12-04': {'actual': 50.5, 'new_orders': 49.5, 'prior': 49.8},
    # 2026
    '2026-01-07': {'actual': 50.8, 'new_orders': 49.8, 'prior': 50.5},
    '2026-02-04': {'actual': 51.2, 'new_orders': 50.5, 'prior': 50.8},
    '2026-03-04': {'actual': 50.5, 'new_orders': 49.5, 'prior': 51.2},
    '2026-04-03': {'actual': 50.0, 'new_orders': 48.8, 'prior': 50.5},
    '2026-05-06': {'actual': 49.5, 'new_orders': 48.0, 'prior': 50.0},
}


# ═══════════════════════════════════════════════════════════════
# SESSION-CONDITIONAL EDGE TABLE
# Backtested: 101 ISM Services PMI releases, 2018-2026, ETH/USDT 15m
# ═══════════════════════════════════════════════════════════════

EDGE_TABLE = {
    'HOT': {
        'n': 43,
        'signal_threshold': 56.0,
        'session_returns': {
            'us_release': +0.01, 'us_am': +0.46, 'us_lunch': -0.19,
            'us_pm': +0.08, 'post_ny': +0.03,
            'sydney_open': -0.60, 'tokyo_open': -0.33,
            'london_open': +0.49, 'london_midday': +0.57,
            'ny_reopen_am': +0.04, 'ny_reopen_pm': +0.30,
        },
        'persistence': {
            'spike_to_us_am': 81.8,
            'asia_afternoon_to_tokyo_close': 79.2,
            'frankfurt_to_london': 72.4,
            'sydney_to_tokyo': 67.9,
        },
        'bias': 'NEUTRAL',  # no clear directional edge
        'confidence': 0.35,
    },
    'WARM': {
        'n': 24,
        'signal_threshold': 53.0,
        'session_returns': {
            'us_release': -0.02, 'us_am': -0.08, 'us_lunch': +0.22,
            'us_pm': +0.14, 'post_ny': +0.13,
            'sydney_open': +0.42, 'tokyo_open': +0.00,
            'london_open': -0.10, 'london_morning': -0.35,
            'ny_reopen_am': -0.10, 'ny_reopen_pm': +0.04,
        },
        'persistence': {
            'spike_to_us_am': 81.8,
            'post_ny_to_sydney': 62.5,
            'asia_afternoon_to_tokyo_close': 79.2,
            'frankfurt_to_london': 72.4,
        },
        'bias': 'LONG',
        'entry_session': 'us_lunch',
        'exit_session': 'london_open',
        'avg_ret_entry_to_exit': +0.76,
        'win_rate': 0.63,
        'confidence': 0.60,
    },
    'COOL': {
        'n': 27,
        'signal_threshold': 50.0,
        'session_returns': {
            'us_release': +0.14, 'us_am': -0.04, 'us_lunch': +0.10,
            'us_pm': -0.26, 'post_ny': +0.40,
            'sydney_open': -0.16, 'tokyo_open': -0.23,
            'london_open': +0.08, 'london_morning': -0.14,
            'ny_reopen_am': -0.32, 'ny_reopen_pm': -0.71,
        },
        'persistence': {
            'spike_to_us_am': 81.8,
            'tokyo_open_to_asia_mid': 63.0,
            'frankfurt_to_london': 72.4,
            'london_midday_to_ny_reopen': 56.8,
        },
        'bias': 'SHORT',
        'entry_session': 'ny_reopen_am',
        'exit_session': 'ny_reopen_pm',
        'avg_ret_entry_to_exit': -1.03,
        'win_rate': 0.37,
        'confidence': 0.50,
    },
    'COLD': {
        'n': 4,
        'signal_threshold': 47.0,
        'session_returns': {
            'us_release': +0.55, 'us_am': +0.76, 'us_pm': +0.52,
            'post_ny': -0.33, 'sydney_open': -0.36,
            'london_open': -0.05, 'ny_reopen_am': -0.54,
        },
        'persistence': {},
        'bias': 'LONG',  # contrarian: extreme weakness → DXY weakens → ETH squeeze
        'entry_session': 'us_am',
        'exit_session': 'us_pm',
        'avg_ret_entry_to_exit': +1.28,
        'win_rate': 0.75,
        'confidence': 0.55,
    },
    'FREEZE': {
        'n': 3,
        'signal_threshold': 0,
        'session_returns': {
            'us_release': -0.15, 'us_am': -0.08, 'us_pm': +0.53,
            'post_ny': +0.66, 'sydney_open': -0.11,
            'london_open': +0.15, 'ny_reopen_am': +0.40,
        },
        'persistence': {},
        'bias': 'LONG',  # extreme contraction → squeeze
        'entry_session': 'us_pm',
        'exit_session': 'ny_reopen_am',
        'avg_ret_entry_to_exit': +1.59,
        'win_rate': 0.67,
        'confidence': 0.50,
    },
}

# Cross-tabulation edges (signal × vol_regime) — from backtest
CROSS_TAB_EDGES = {
    ('HOT', 'SQUEEZE'):    {'avg_ret': -2.31, 'win': 0.00, 'n': 3, 'bias': 'SHORT'},
    ('COOL', 'TREND'):     {'avg_ret': -1.73, 'win': 0.25, 'n': 4, 'bias': 'SHORT'},
    ('WARM', 'TREND'):     {'avg_ret': +1.35, 'win': 0.60, 'n': 10, 'bias': 'LONG'},
    ('FREEZE', 'CHOP'):    {'avg_ret': +1.19, 'win': 0.67, 'n': 3, 'bias': 'LONG'},
    ('WARM', 'SQUEEZE'):   {'avg_ret': +0.96, 'win': 1.00, 'n': 4, 'bias': 'LONG'},
    ('WARM', 'CHOP'):      {'avg_ret': +0.66, 'win': 0.71, 'n': 7, 'bias': 'LONG'},
}

# New Orders sub-index buckets — from backtest
NEW_ORDERS_BUCKETS = {
    'SURGE':    {'min': 58, 'avg_ret': +0.14, 'win': 0.583, 'n': 24, 'bias': 'NEUTRAL'},
    'HOT':      {'min': 55, 'avg_ret': -0.17, 'win': 0.500, 'n': 20, 'bias': 'NEUTRAL'},
    'WARM':     {'min': 52, 'avg_ret': +0.44, 'win': 0.538, 'n': 26, 'bias': 'NEUTRAL'},
    'NEUTRAL':  {'min': 50, 'avg_ret': +0.48, 'win': 0.692, 'n': 13, 'bias': 'LONG'},
    'COOL':     {'min': 48, 'avg_ret': -1.15, 'win': 0.214, 'n': 14, 'bias': 'SHORT'},
    'COLD':     {'min': 0,  'avg_ret': +0.89, 'win': 0.750, 'n': 4,  'bias': 'LONG'},
}

# Session transmission chain — strongest links (persist >65%)
TRANSMISSION_CHAIN = {
    'spike_to_us_am':               81.8,  # n=77 — strongest
    'asia_afternoon_to_tokyo_close': 79.2,  # n=72
    'frankfurt_to_london_open':      72.4,  # n=76
    'sydney_to_tokyo':               67.9,  # n=78
}

# Chain break point: NY PM → Post-NY (40.7% persistence)
CHAIN_BREAK_SESSION = 'ny_pm'

# Thursday (claims) positioning edge
CLAIMS_POSITIONING = {
    'COOL_NO': {'thu_avg': +1.31, 'n': 14},  # COOL New Orders → Thursday bounce
}


# ═══════════════════════════════════════════════════════════════
# HELPERS
# ═══════════════════════════════════════════════════════════════

def _classify_signal(actual):
    if actual >= 56.0:
        return 'HOT'
    elif actual >= 53.0:
        return 'WARM'
    elif actual >= 50.0:
        return 'COOL'
    elif actual >= 47.0:
        return 'COLD'
    else:
        return 'FREEZE'


def _classify_new_orders_bucket(new_orders):
    if new_orders >= 58:
        return 'SURGE'
    elif new_orders >= 55:
        return 'HOT'
    elif new_orders >= 52:
        return 'WARM'
    elif new_orders >= 50:
        return 'NEUTRAL'
    elif new_orders >= 48:
        return 'COOL'
    else:
        return 'COLD'


def _get_current_session(now_utc):
    h = now_utc.hour
    m = now_utc.minute
    t = h * 60 + m

    if 900 <= t < 960:
        return 'us_release'
    elif 900 <= t < 1020:
        return 'us_am'
    elif 1020 <= t < 1080:
        return 'us_lunch'
    elif 1080 <= t < 1260:
        return 'us_pm'
    elif 1260 <= t < 1440 or t < 0:
        return 'post_ny'
    elif 0 <= t < 120:
        return 'sydney_open'
    elif 60 <= t < 180:
        return 'tokyo_open'
    elif 180 <= t < 300:
        return 'asia_mid'
    elif 300 <= t < 420:
        return 'asia_afternoon'
    elif 360 <= t < 480:
        return 'tokyo_close'
    elif 420 <= t < 540:
        return 'frankfurt_open'
    elif 480 <= t < 600:
        return 'london_open'
    elif 540 <= t < 720:
        return 'london_morning'
    elif 720 <= t < 840:
        return 'london_midday'
    elif 780 <= t < 870:
        return 'ny_pre_open'
    elif 870 <= t < 960:
        return 'ny_open'
    elif 870 <= t < 1020:
        return 'ny_reopen_am'
    elif 1020 <= t < 1080:
        return 'ny_reopen_lunch'
    elif 1080 <= t < 1260:
        return 'ny_reopen_pm'
    return None


def _is_ism_svc_release_day(today_str=None, window_days=1):
    if today_str is None:
        today_str = datetime.utcnow().strftime('%Y-%m-%d')
    today = datetime.strptime(today_str, '%Y-%m-%d')

    for release_date_str, pmi_data in sorted(ISM_SVC_RELEASES.items(), reverse=True):
        release_dt = datetime.strptime(release_date_str, '%Y-%m-%d')
        days_since = (today - release_dt).days
        if 0 <= days_since <= window_days:
            return True, release_date_str, pmi_data
    return False, None, None


def score_m28_ism_svc(wyckoff_phase='RANGE', vol_regime='CHOP',
                      direction='LONG', today_str=None, config=None,
                      now_utc=None):
    """Score the US ISM Services PMI session bias.

    Args:
        wyckoff_phase: from M21 (unused — kept for API compatibility)
        vol_regime: from M9 (used for cross-tab edge lookup)
        direction: trade direction ('LONG' or 'SHORT')
        today_str: YYYY-MM-DD override
        config: config dict (optional)
        now_utc: datetime UTC override

    Returns:
        status: 'PASS', 'SKIP', or 'WEAK'
        score_adj: score adjustment (-0.10 to +0.10)
        size_mult: position size multiplier (0.5 to 1.0)
        details: dict
    """
    cfg = config or {}

    if not cfg.get('M28_ENABLED', True):
        return 'SKIP', 0.0, 1.0, {'regime': 'DISABLED'}

    is_release, release_date, pmi_data = _is_ism_svc_release_day(
        today_str, window_days=cfg.get('M28_WINDOW_DAYS', 1))

    if not is_release:
        return 'SKIP', 0.0, 1.0, {'regime': 'NOT_RELEASE_DAY'}

    actual = pmi_data['actual']
    new_orders = pmi_data['new_orders']
    prior = pmi_data['prior']
    surprise = actual - prior
    signal = _classify_signal(actual)
    no_bucket = _classify_new_orders_bucket(new_orders)

    # ── Lookup 1: Cross-tab edge (signal × vol_regime) — most specific ──
    cross_key = (signal, vol_regime)
    cross_edge = CROSS_TAB_EDGES.get(cross_key)

    # ── Lookup 2: Signal edge ──
    edge = EDGE_TABLE.get(signal)

    # ── Lookup 3: New Orders bucket ──
    no_edge = NEW_ORDERS_BUCKETS.get(no_bucket)

    # ── Determine best signal ──
    best_match = None
    best_source = 'NONE'
    confidence = 0.0

    if cross_edge and cross_edge['n'] >= 3:
        best_match = cross_edge
        best_source = 'CROSS_TAB'
        confidence = min(1.0, cross_edge['n'] / 8)
    elif no_bucket == 'COOL' and no_edge and no_edge['n'] >= 10:
        # COOL New Orders is the strongest single signal (n=14, -1.15%)
        best_match = no_edge
        best_source = 'NEW_ORDERS_BUCKET'
        confidence = min(1.0, no_edge['n'] / 15)
    elif edge and edge.get('confidence', 0) >= 0.45 and edge.get('bias') != 'NEUTRAL':
        best_match = {
            'avg_ret': edge.get('avg_ret_entry_to_exit', 0),
            'win': edge.get('win_rate', 0),
            'n': edge['n'],
            'bias': edge['bias'],
        }
        best_source = 'SIGNAL'
        confidence = edge['confidence']

    if best_match is None or best_match.get('bias') == 'NEUTRAL':
        return 'SKIP', 0.0, 1.0, {
            'regime': 'NO_EDGE',
            'signal': signal,
            'new_orders': new_orders,
            'actual': actual,
            'release_date': release_date,
        }

    # ── Session-aware sizing ──
    current_session = None
    session_phase = 'PRE_RELEASE'
    if now_utc:
        current_session = _get_current_session(now_utc)
        active_sessions = {'us_release', 'us_am', 'us_lunch', 'us_pm',
                          'post_ny', 'sydney_open', 'tokyo_open',
                          'asia_mid', 'asia_afternoon', 'tokyo_close',
                          'frankfurt_open', 'london_open', 'london_morning'}
        fading_sessions = {'london_midday', 'ny_pre_open', 'ny_open', 'ny_reopen_am'}
        if current_session in active_sessions:
            session_phase = 'IN_WINDOW'
        elif current_session in fading_sessions:
            session_phase = 'FADING'
        else:
            session_phase = 'OUT_OF_WINDOW'

    # ── Compute score adjustment ──
    avg_ret = best_match.get('avg_ret', 0)
    abs_ret = abs(avg_ret)

    if abs_ret >= 2.0:
        raw_adj = 0.10
    elif abs_ret >= 1.0:
        raw_adj = 0.07
    else:
        raw_adj = 0.05

    score_adj = raw_adj * confidence

    bias = best_match.get('bias', 'NEUTRAL')
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

    if best_match.get('n', 0) < 5:
        size_mult *= 0.75

    if session_phase == 'FADING':
        size_mult *= 0.50
    elif session_phase == 'OUT_OF_WINDOW':
        size_mult *= 0.25

    size_mult = round(size_mult, 2)
    score_adj = round(score_adj, 3)

    status = 'PASS' if confidence >= 0.3 else 'WEAK'

    details = {
        'regime': f'ISM_SVC_{bias}',
        'release_date': release_date,
        'actual': actual,
        'new_orders': new_orders,
        'prior': prior,
        'surprise': round(surprise, 1),
        'signal': signal,
        'new_orders_bucket': no_bucket,
        'bias': bias,
        'avg_ret': avg_ret,
        'win_rate': best_match.get('win', 0),
        'sample_size': best_match.get('n', 0),
        'confidence': round(confidence, 2),
        'source': best_source,
        'session_phase': session_phase,
        'current_session': current_session,
        'entry_session': edge.get('entry_session') if edge else None,
        'exit_session': edge.get('exit_session') if edge else None,
        'score_adj': score_adj,
        'size_mult': size_mult,
        'chain': TRANSMISSION_CHAIN,
        'chain_break': CHAIN_BREAK_SESSION,
        'claims_positioning': CLAIMS_POSITIONING.get('COOL_NO'),
    }

    return status, score_adj, size_mult, details


def format_m28(details):
    """Format M28 details for terminal output."""
    if not details or details.get('regime') in ('DISABLED', 'NOT_RELEASE_DAY', 'NO_EDGE'):
        regime = details.get('regime', '?') if details else '?'
        if regime == 'NOT_RELEASE_DAY':
            return ''
        return ''

    bias = details.get('bias', '?')
    actual = details.get('actual', 0)
    new_orders = details.get('new_orders', 0)
    prior = details.get('prior', 0)
    surprise = details.get('surprise', 0)
    signal = details.get('signal', '?')
    no_bucket = details.get('new_orders_bucket', '?')
    release = details.get('release_date', '?')
    avg_ret = details.get('avg_ret', 0)
    win = details.get('win_rate', 0)
    n = details.get('sample_size', 0)
    conf = details.get('confidence', 0)
    source = details.get('source', '?')
    score_adj = details.get('score_adj', 0)
    size_mult = details.get('size_mult', 1.0)
    session_phase = details.get('session_phase', '?')
    entry_s = details.get('entry_session', '?')
    exit_s = details.get('exit_session', '?')
    claims = details.get('claims_positioning')

    icon = '🟢' if bias == 'LONG' else '🔴' if bias == 'SHORT' else '⚪'
    conf_icon = '🟢' if conf >= 0.7 else '🟡' if conf >= 0.4 else '🟠'
    phase_icon = {'IN_WINDOW': '✅', 'FADING': '⚠️', 'OUT_OF_WINDOW': '❌',
                  'PRE_RELEASE': '⏳'}.get(session_phase, '❓')

    svc_icon = '🔥' if actual >= 56 else '🌡️' if actual >= 53 else '❄️' if actual >= 50 else '🧊'

    lines = []
    lines.append(f"\n  {icon} M28 US ISM SVC PMI: {bias}")
    lines.append(f"    Release: {release}  |  Actual: {actual:.1f}  |  Prior: {prior:.1f}  |  Surprise: {surprise:+.1f}")
    lines.append(f"    {svc_icon} Services: {actual:.1f} ({signal})  |  New Orders: {new_orders:.1f} ({no_bucket})")
    if entry_s and exit_s and entry_s != 'None':
        lines.append(f"    Chain: {entry_s} → {exit_s}")
    lines.append(f"    Backtest: avg={avg_ret:+.2f}%  win={win*100:.0f}%  n={n}  source={source}")
    lines.append(f"    {conf_icon} Confidence: {conf:.2f}  |  Score adj: {score_adj:+.3f}  |  Size: {size_mult:.2f}x")
    lines.append(f"    {phase_icon} Session: {session_phase} ({details.get('current_session', '?')})")

    chain = details.get('chain', {})
    if chain:
        lines.append(f"    📊 Chain: Spike→NY AM {chain.get('spike_to_us_am', 0):.0f}% → "
                     f"Sydney→Tokyo {chain.get('sydney_to_tokyo', 0):.0f}% → "
                     f"Frankfurt→LDN {chain.get('frankfurt_to_london_open', 0):.0f}%")
        lines.append(f"    ⚠️ Chain breaks at {details.get('chain_break', '?')} (40.7% persist)")

    if claims:
        lines.append(f"    📈 Claims positioning: COOL NO → Thu {claims['thu_avg']:+.2f}% (n={claims['n']})")

    return '\n'.join(lines)


def get_ism_svc_cache_path():
    cache_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), 'data', 'macro')
    return os.path.join(cache_dir, 'ism_svc_cache.json')


def load_ism_svc_cache():
    cache_path = get_ism_svc_cache_path()
    if os.path.exists(cache_path):
        try:
            with open(cache_path) as f:
                return json.load(f)
        except (json.JSONDecodeError, IOError):
            pass
    return {}


def update_ism_svc_cache(actual, new_orders, prior=None, release_date=None):
    cache = load_ism_svc_cache()
    if release_date is None:
        release_date = datetime.utcnow().strftime('%Y-%m-%d')
    cache[release_date] = {
        'actual': actual,
        'new_orders': new_orders,
        'prior': prior,
        'updated': datetime.utcnow().isoformat(),
    }
    cache_path = get_ism_svc_cache_path()
    os.makedirs(os.path.dirname(cache_path), exist_ok=True)
    with open(cache_path, 'w') as f:
        json.dump(cache, f, indent=2)
    return cache

"""
M29: JOLTS Job Openings — Session Transmission Chain + NFP Loopback

Released ~first Tuesday of each month at 10:00 AM ET (15:00 UTC).
Tracks structural labor demand + quits rate. 2-month lag.
Collapse in openings → loosening labor → ETH surges as yields ease.

Thesis chain: US Morning → Wednesday (ADP proxy) → Friday (NFP loopback).
High/low structural openings materialize as hires/misses in NFP.

Backtested on 100 JOLTS releases (2018-2026) against ETH/USDT 15m data.

Key findings:
  SURGING + TREND: +7.17% avg, 100% win, n=5 — p=0.0196 (statistically significant!)
  RISING + TREND:  -5.18% avg, 14.3% win, n=7 — p=0.0072 (statistically significant!)
  DECLINING + TREND: +3.54% avg, 80% win, n=5
  COLLAPSE + CHOP: +2.02% avg, 100% win, n=5
  SURGING + CHOP:  +1.66% avg, 83.3% win, n=6

  Transmission chain:
    1h spike → NY AM: 81.5% (n=81)
    Tokyo Close → Frankfurt: 73.8% (n=80)
    Frankfurt → London Open: 70.6% (n=85)
    Asia Afternoon → Tokyo Close: 66.7% (n=78)

  JOLTS → NFP loopback: 20-33% full 3-day persistence (does NOT hold).
  But SURGING combined 3-day: +1.93%. RISING: -3.22%.

Integration: lightweight modifier on JOLTS release days only (~12x/year).
Returns a score adjustment and size multiplier — does NOT veto.
Session-aware: size multiplier decays after London (chain breaks at NY PM).

Usage:
    from src.modules.m29_jolts import score_m29_jolts, format_m29
    status, score_adj, size_mult, details = score_m29_jolts(
        direction='LONG', vol_regime='CHOP')
"""

from datetime import datetime, timedelta
import json
import os


# ═══════════════════════════════════════════════════════════════
# JOLTS JOB OPENINGS RELEASE DATES + ACTUAL VALUES
# Released ~first Tuesday of month, 10:00 AM ET (15:00 UTC)
# Source: BLS. 2-month lag (Jan data → Mar release).
# ═══════════════════════════════════════════════════════════════

JOLTS_RELEASES = {
    # 2018
    '2018-01-09': {'actual': 5811, 'prior': 5978, 'quits_rate': 2.2},
    '2018-02-06': {'actual': 5882, 'prior': 5811, 'quits_rate': 2.1},
    '2018-03-16': {'actual': 6052, 'prior': 5882, 'quits_rate': 2.2},
    '2018-04-06': {'actual': 6550, 'prior': 6052, 'quits_rate': 2.3},
    '2018-05-08': {'actual': 6690, 'prior': 6550, 'quits_rate': 2.3},
    '2018-06-05': {'actual': 6638, 'prior': 6690, 'quits_rate': 2.4},
    '2018-07-10': {'actual': 6662, 'prior': 6638, 'quits_rate': 2.3},
    '2018-08-07': {'actual': 6939, 'prior': 6662, 'quits_rate': 2.4},
    '2018-09-11': {'actual': 7136, 'prior': 6939, 'quits_rate': 2.4},
    '2018-10-16': {'actual': 7009, 'prior': 7136, 'quits_rate': 2.3},
    '2018-11-06': {'actual': 7079, 'prior': 7009, 'quits_rate': 2.3},
    '2018-12-11': {'actual': 6888, 'prior': 7079, 'quits_rate': 2.3},
    # 2019
    '2019-01-08': {'actual': 7335, 'prior': 6888, 'quits_rate': 2.3},
    '2019-02-12': {'actual': 7581, 'prior': 7335, 'quits_rate': 2.3},
    '2019-03-12': {'actual': 7087, 'prior': 7581, 'quits_rate': 2.3},
    '2019-04-09': {'actual': 7488, 'prior': 7087, 'quits_rate': 2.4},
    '2019-05-07': {'actual': 7449, 'prior': 7488, 'quits_rate': 2.3},
    '2019-06-11': {'actual': 7323, 'prior': 7449, 'quits_rate': 2.3},
    '2019-07-09': {'actual': 7348, 'prior': 7323, 'quits_rate': 2.3},
    '2019-08-06': {'actual': 7217, 'prior': 7348, 'quits_rate': 2.4},
    '2019-09-10': {'actual': 7051, 'prior': 7217, 'quits_rate': 2.3},
    '2019-10-29': {'actual': 7013, 'prior': 7051, 'quits_rate': 2.3},
    '2019-12-10': {'actual': 6800, 'prior': 7013, 'quits_rate': 2.2},
    # 2020
    '2020-01-07': {'actual': 6423, 'prior': 6800, 'quits_rate': 2.2},
    '2020-02-11': {'actual': 6965, 'prior': 6423, 'quits_rate': 2.2},
    '2020-03-17': {'actual': 6882, 'prior': 6965, 'quits_rate': 2.2},
    '2020-04-07': {'actual': 6191, 'prior': 6882, 'quits_rate': 1.8},
    '2020-05-12': {'actual': 5046, 'prior': 6191, 'quits_rate': 1.4},
    '2020-06-09': {'actual': 5397, 'prior': 5046, 'quits_rate': 1.6},
    '2020-07-07': {'actual': 5889, 'prior': 5397, 'quits_rate': 1.9},
    '2020-08-10': {'actual': 6270, 'prior': 5889, 'quits_rate': 2.0},
    '2020-09-10': {'actual': 6493, 'prior': 6270, 'quits_rate': 2.1},
    '2020-10-06': {'actual': 6436, 'prior': 6493, 'quits_rate': 2.1},
    '2020-11-09': {'actual': 6652, 'prior': 6436, 'quits_rate': 2.2},
    '2020-12-09': {'actual': 6527, 'prior': 6652, 'quits_rate': 2.1},
    # 2021
    '2021-01-12': {'actual': 6646, 'prior': 6527, 'quits_rate': 2.1},
    '2021-02-09': {'actual': 6917, 'prior': 6646, 'quits_rate': 2.2},
    '2021-03-31': {'actual': 7367, 'prior': 6917, 'quits_rate': 2.4},
    '2021-04-06': {'actual': 7423, 'prior': 7367, 'quits_rate': 2.5},
    '2021-05-11': {'actual': 8170, 'prior': 7423, 'quits_rate': 2.7},
    '2021-06-08': {'actual': 9193, 'prior': 8170, 'quits_rate': 2.8},
    '2021-07-07': {'actual': 9483, 'prior': 9193, 'quits_rate': 2.9},
    '2021-08-10': {'actual': 10434, 'prior': 9483, 'quits_rate': 2.9},
    '2021-09-08': {'actual': 10439, 'prior': 10434, 'quits_rate': 2.9},
    '2021-10-12': {'actual': 10438, 'prior': 10439, 'quits_rate': 3.0},
    '2021-11-12': {'actual': 10562, 'prior': 10438, 'quits_rate': 3.0},
    '2021-12-08': {'actual': 10562, 'prior': 10562, 'quits_rate': 3.0},
    # 2022
    '2022-01-04': {'actual': 10925, 'prior': 10562, 'quits_rate': 3.0},
    '2022-02-01': {'actual': 11266, 'prior': 10925, 'quits_rate': 2.8},
    '2022-03-09': {'actual': 11266, 'prior': 11266, 'quits_rate': 2.9},
    '2022-04-05': {'actual': 11549, 'prior': 11266, 'quits_rate': 3.0},
    '2022-05-03': {'actual': 11400, 'prior': 11549, 'quits_rate': 2.9},
    '2022-06-01': {'actual': 11254, 'prior': 11400, 'quits_rate': 2.8},
    '2022-07-06': {'actual': 10698, 'prior': 11254, 'quits_rate': 2.7},
    '2022-08-02': {'actual': 11239, 'prior': 10698, 'quits_rate': 2.7},
    '2022-09-07': {'actual': 10053, 'prior': 11239, 'quits_rate': 2.7},
    '2022-10-04': {'actual': 10334, 'prior': 10053, 'quits_rate': 2.6},
    '2022-11-01': {'actual': 10334, 'prior': 10334, 'quits_rate': 2.6},
    '2022-12-06': {'actual': 10458, 'prior': 10334, 'quits_rate': 2.7},
    # 2023
    '2023-01-04': {'actual': 11012, 'prior': 10458, 'quits_rate': 2.6},
    '2023-02-01': {'actual': 10824, 'prior': 11012, 'quits_rate': 2.5},
    '2023-03-08': {'actual': 10563, 'prior': 10824, 'quits_rate': 2.6},
    '2023-04-04': {'actual': 9590, 'prior': 10563, 'quits_rate': 2.5},
    '2023-05-02': {'actual': 10103, 'prior': 9590, 'quits_rate': 2.4},
    '2023-06-06': {'actual': 9824, 'prior': 10103, 'quits_rate': 2.6},
    '2023-07-06': {'actual': 9582, 'prior': 9824, 'quits_rate': 2.4},
    '2023-08-01': {'actual': 8827, 'prior': 9582, 'quits_rate': 2.3},
    '2023-09-06': {'actual': 9610, 'prior': 8827, 'quits_rate': 2.3},
    '2023-10-03': {'actual': 9553, 'prior': 9610, 'quits_rate': 2.3},
    '2023-11-01': {'actual': 8733, 'prior': 9553, 'quits_rate': 2.2},
    '2023-12-05': {'actual': 8790, 'prior': 8733, 'quits_rate': 2.2},
    # 2024
    '2024-01-03': {'actual': 8733, 'prior': 8790, 'quits_rate': 2.2},
    '2024-02-06': {'actual': 8863, 'prior': 8733, 'quits_rate': 2.1},
    '2024-03-06': {'actual': 8756, 'prior': 8863, 'quits_rate': 2.2},
    '2024-04-02': {'actual': 8488, 'prior': 8756, 'quits_rate': 2.1},
    '2024-05-01': {'actual': 8059, 'prior': 8488, 'quits_rate': 2.1},
    '2024-06-04': {'actual': 8140, 'prior': 8059, 'quits_rate': 2.1},
    '2024-07-02': {'actual': 8184, 'prior': 8140, 'quits_rate': 2.1},
    '2024-08-06': {'actual': 7673, 'prior': 8184, 'quits_rate': 2.0},
    '2024-09-04': {'actual': 7861, 'prior': 7673, 'quits_rate': 1.9},
    '2024-10-01': {'actual': 7840, 'prior': 7861, 'quits_rate': 1.9},
    '2024-11-05': {'actual': 7744, 'prior': 7840, 'quits_rate': 2.0},
    '2024-12-03': {'actual': 7839, 'prior': 7744, 'quits_rate': 1.9},
    # 2025
    '2025-01-07': {'actual': 7600, 'prior': 7839, 'quits_rate': 1.9},
    '2025-02-04': {'actual': 7740, 'prior': 7600, 'quits_rate': 2.0},
    '2025-03-04': {'actual': 7568, 'prior': 7740, 'quits_rate': 2.0},
    '2025-04-01': {'actual': 7480, 'prior': 7568, 'quits_rate': 1.9},
    '2025-05-06': {'actual': 7391, 'prior': 7480, 'quits_rate': 1.9},
    '2025-06-03': {'actual': 7320, 'prior': 7391, 'quits_rate': 1.8},
    '2025-07-01': {'actual': 7181, 'prior': 7320, 'quits_rate': 1.8},
    '2025-08-05': {'actual': 7100, 'prior': 7181, 'quits_rate': 1.7},
    '2025-09-02': {'actual': 7210, 'prior': 7100, 'quits_rate': 1.8},
    '2025-10-07': {'actual': 7050, 'prior': 7210, 'quits_rate': 1.7},
    '2025-11-04': {'actual': 6980, 'prior': 7050, 'quits_rate': 1.7},
    '2025-12-02': {'actual': 7020, 'prior': 6980, 'quits_rate': 1.7},
    # 2026
    '2026-01-06': {'actual': 7100, 'prior': 7020, 'quits_rate': 1.8},
    '2026-02-03': {'actual': 7050, 'prior': 7100, 'quits_rate': 1.7},
    '2026-03-03': {'actual': 6950, 'prior': 7050, 'quits_rate': 1.7},
    '2026-04-07': {'actual': 6880, 'prior': 6950, 'quits_rate': 1.6},
    '2026-05-05': {'actual': 6800, 'prior': 6880, 'quits_rate': 1.6},
}


# ═══════════════════════════════════════════════════════════════
# SESSION-CONDITIONAL EDGE TABLE
# Backtested: 100 JOLTS releases, 2018-2026, ETH/USDT 15m
# ═══════════════════════════════════════════════════════════════

EDGE_TABLE = {
    'COLLAPSE': {
        'n': 10,
        'session_returns': {
            'us_release': -0.10, 'us_am': +0.43, 'us_pm': +0.54,
            'post_ny': -0.08, 'sydney_open': +0.25,
            'tokyo_open': +0.12, 'london_open': -0.02,
            'ny_reopen_am': -0.42, 'ny_reopen_pm': +0.28,
        },
        'persistence': {
            'spike_to_us_am': 81.5,
            'tokyo_close_to_frankfurt': 73.8,
            'frankfurt_to_london': 70.6,
        },
        'bias': 'LONG',  # collapse → yields ease → ETH surges
        'confidence': 0.65,
    },
    'DECLINING': {
        'n': 12,
        'session_returns': {
            'us_release': -0.58, 'us_am': -0.22, 'us_pm': +0.38,
            'post_ny': +0.35, 'tokyo_open': +0.33,
            'london_open': -0.09, 'ny_reopen_am': -0.24,
        },
        'persistence': {
            'spike_to_us_am': 81.5,
            'tokyo_close_to_frankfurt': 73.8,
        },
        'bias': 'LONG',
        'confidence': 0.50,
    },
    'SOFTENING': {
        'n': 31,
        'session_returns': {
            'us_release': +0.25, 'us_am': +0.13, 'us_pm': -0.15,
            'tokyo_open': +0.32, 'london_open': +0.10,
            'ny_reopen_am': +0.47, 'ny_reopen_pm': +0.27,
        },
        'persistence': {},
        'bias': 'NEUTRAL',
        'confidence': 0.30,
    },
    'STABLE': {
        'n': 18,
        'session_returns': {
            'us_release': +0.46, 'us_am': +0.68, 'us_lunch': +0.20,
            'us_pm': +0.17, 'post_ny': +0.35,
            'frankfurt_open': +0.40, 'london_open': +0.17,
            'london_midday': +0.34, 'ny_reopen_am': -0.52,
        },
        'persistence': {
            'spike_to_us_am': 81.5,
            'tokyo_close_to_frankfurt': 73.8,
        },
        'bias': 'LONG',
        'entry_session': 'us_release',
        'exit_session': 'london_midday',
        'avg_ret_entry_to_exit': +1.95,
        'win_rate': 0.63,
        'confidence': 0.60,
    },
    'RISING': {
        'n': 14,
        'session_returns': {
            'us_release': +0.04, 'us_am': +0.09, 'us_pm': -0.13,
            'post_ny': -0.36, 'sydney_open': -0.53,
            'tokyo_open': -0.22, 'asia_mid': -0.57,
            'london_open': +0.26, 'london_midday': -0.42,
            'ny_reopen_am': -0.58, 'ny_reopen_pm': -0.34,
        },
        'persistence': {
            'spike_to_us_am': 81.5,
            'tokyo_close_to_frankfurt': 73.8,
        },
        'bias': 'SHORT',  # RISING in TREND = inflation fear (p=0.0072)
        'entry_session': 'post_ny',
        'exit_session': 'ny_reopen_pm',
        'avg_ret_entry_to_exit': -2.48,
        'win_rate': 0.43,
        'confidence': 0.70,
    },
    'SURGING': {
        'n': 15,
        'session_returns': {
            'us_release': +0.39, 'us_am': +0.53, 'us_lunch': +0.45,
            'us_pm': +0.84, 'post_ny': -0.20,
            'sydney_open': +0.54, 'tokyo_open': +0.45,
            'london_open': -0.07, 'ny_reopen_am': +0.39,
        },
        'persistence': {
            'spike_to_us_am': 81.5,
            'tokyo_close_to_frankfurt': 73.8,
            'frankfurt_to_london': 70.6,
        },
        'bias': 'LONG',  # SURGING = strong labor = risk-on (p=0.0196)
        'entry_session': 'us_am',
        'exit_session': 'ny_reopen_am',
        'avg_ret_entry_to_exit': +2.84,
        'win_rate': 0.73,
        'confidence': 0.75,
    },
}

# Cross-tabulation edges (signal × vol_regime) — from backtest
CROSS_TAB_EDGES = {
    ('SURGING', 'TREND'):   {'avg_ret': +7.17, 'win': 1.00, 'n': 5, 'bias': 'LONG'},
    ('RISING', 'TREND'):    {'avg_ret': -5.18, 'win': 0.14, 'n': 7, 'bias': 'SHORT'},
    ('DECLINING', 'TREND'): {'avg_ret': +3.54, 'win': 0.80, 'n': 5, 'bias': 'LONG'},
    ('STABLE', 'TREND'):    {'avg_ret': +2.13, 'win': 0.63, 'n': 8, 'bias': 'LONG'},
    ('COLLAPSE', 'CHOP'):   {'avg_ret': +2.02, 'win': 1.00, 'n': 5, 'bias': 'LONG'},
    ('SURGING', 'CHOP'):    {'avg_ret': +1.66, 'win': 0.83, 'n': 6, 'bias': 'LONG'},
    ('RISING', 'CHOP'):     {'avg_ret': -1.11, 'win': 0.17, 'n': 6, 'bias': 'SHORT'},
}

# Session transmission chain
TRANSMISSION_CHAIN = {
    'spike_to_us_am':           81.5,
    'tokyo_close_to_frankfurt': 73.8,
    'frankfurt_to_london_open': 70.6,
    'asia_afternoon_to_tokyo_close': 66.7,
}

CHAIN_BREAK_SESSION = 'ny_pm'


# ═══════════════════════════════════════════════════════════════
# HELPERS
# ═══════════════════════════════════════════════════════════════

def _classify_signal(actual, prior):
    change_pct = (actual - prior) / prior * 100 if prior > 0 else 0
    if change_pct <= -5.0:
        return 'COLLAPSE'
    elif change_pct <= -2.0:
        return 'DECLINING'
    elif change_pct <= 0:
        return 'SOFTENING'
    elif change_pct <= 2.0:
        return 'STABLE'
    elif change_pct <= 5.0:
        return 'RISING'
    else:
        return 'SURGING'


def _get_current_session(now_utc):
    h = now_utc.hour
    m = now_utc.minute
    t = h * 60 + m
    if 900 <= t < 960: return 'us_release'
    elif 900 <= t < 1020: return 'us_am'
    elif 1020 <= t < 1080: return 'us_lunch'
    elif 1080 <= t < 1260: return 'us_pm'
    elif 1260 <= t < 1440 or t < 0: return 'post_ny'
    elif 0 <= t < 120: return 'sydney_open'
    elif 60 <= t < 180: return 'tokyo_open'
    elif 180 <= t < 300: return 'asia_mid'
    elif 300 <= t < 420: return 'asia_afternoon'
    elif 360 <= t < 480: return 'tokyo_close'
    elif 420 <= t < 540: return 'frankfurt_open'
    elif 480 <= t < 600: return 'london_open'
    elif 540 <= t < 720: return 'london_morning'
    elif 720 <= t < 840: return 'london_midday'
    elif 870 <= t < 1020: return 'ny_reopen_am'
    elif 1080 <= t < 1260: return 'ny_reopen_pm'
    return None


def _is_jolts_release_day(today_str=None, window_days=1):
    if today_str is None:
        today_str = datetime.utcnow().strftime('%Y-%m-%d')
    today = datetime.strptime(today_str, '%Y-%m-%d')
    for release_date_str, jolts_data in sorted(JOLTS_RELEASES.items(), reverse=True):
        release_dt = datetime.strptime(release_date_str, '%Y-%m-%d')
        days_since = (today - release_dt).days
        if 0 <= days_since <= window_days:
            return True, release_date_str, jolts_data
    return False, None, None


def score_m29_jolts(wyckoff_phase='RANGE', vol_regime='CHOP',
                    direction='LONG', today_str=None, config=None,
                    now_utc=None):
    """Score the JOLTS Job Openings session bias.

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

    if not cfg.get('M29_ENABLED', True):
        return 'SKIP', 0.0, 1.0, {'regime': 'DISABLED'}

    is_release, release_date, jolts_data = _is_jolts_release_day(
        today_str, window_days=cfg.get('M29_WINDOW_DAYS', 1))

    if not is_release:
        return 'SKIP', 0.0, 1.0, {'regime': 'NOT_RELEASE_DAY'}

    actual = jolts_data['actual']
    prior = jolts_data['prior']
    quits = jolts_data['quits_rate']
    change_pct = (actual - prior) / prior * 100 if prior > 0 else 0
    signal = _classify_signal(actual, prior)

    # ── Lookup 1: Cross-tab edge (signal × vol_regime) ──
    cross_key = (signal, vol_regime)
    cross_edge = CROSS_TAB_EDGES.get(cross_key)

    # ── Lookup 2: Signal edge ──
    edge = EDGE_TABLE.get(signal)

    best_match = None
    best_source = 'NONE'
    confidence = 0.0

    if cross_edge and cross_edge['n'] >= 3:
        best_match = cross_edge
        best_source = 'CROSS_TAB'
        confidence = min(1.0, cross_edge['n'] / 8)
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
            'regime': 'NO_EDGE', 'signal': signal,
            'actual': actual, 'prior': prior, 'release_date': release_date,
        }

    # ── Session-aware sizing ──
    current_session = None
    session_phase = 'PRE_RELEASE'
    if now_utc:
        current_session = _get_current_session(now_utc)
        active = {'us_release', 'us_am', 'us_lunch', 'us_pm', 'post_ny',
                  'sydney_open', 'tokyo_open', 'asia_mid', 'asia_afternoon',
                  'tokyo_close', 'frankfurt_open', 'london_open', 'london_morning'}
        fading = {'london_midday', 'ny_reopen_am'}
        if current_session in active:
            session_phase = 'IN_WINDOW'
        elif current_session in fading:
            session_phase = 'FADING'
        else:
            session_phase = 'OUT_OF_WINDOW'

    # ── Score adjustment ──
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
        'regime': f'JOLTS_{bias}',
        'release_date': release_date,
        'actual': actual, 'prior': prior,
        'change_pct': round(change_pct, 2),
        'quits_rate': quits,
        'signal': signal, 'bias': bias,
        'avg_ret': avg_ret,
        'win_rate': best_match.get('win', 0),
        'sample_size': best_match.get('n', 0),
        'confidence': round(confidence, 2),
        'source': best_source,
        'session_phase': session_phase,
        'current_session': current_session,
        'entry_session': edge.get('entry_session') if edge else None,
        'exit_session': edge.get('exit_session') if edge else None,
        'score_adj': score_adj, 'size_mult': size_mult,
        'chain': TRANSMISSION_CHAIN,
        'chain_break': CHAIN_BREAK_SESSION,
    }

    return status, score_adj, size_mult, details


def format_m29(details):
    if not details or details.get('regime') in ('DISABLED', 'NOT_RELEASE_DAY', 'NO_EDGE'):
        regime = details.get('regime', '?') if details else '?'
        if regime == 'NOT_RELEASE_DAY':
            return ''
        return ''

    bias = details.get('bias', '?')
    actual = details.get('actual', 0)
    prior = details.get('prior', 0)
    change_pct = details.get('change_pct', 0)
    quits = details.get('quits_rate', 0)
    signal = details.get('signal', '?')
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

    icon = '🟢' if bias == 'LONG' else '🔴' if bias == 'SHORT' else '⚪'
    conf_icon = '🟢' if conf >= 0.7 else '🟡' if conf >= 0.4 else '🟠'
    phase_icon = {'IN_WINDOW': '✅', 'FADING': '⚠️', 'OUT_OF_WINDOW': '❌',
                  'PRE_RELEASE': '⏳'}.get(session_phase, '❓')

    change_icon = '📈' if change_pct > 0 else '📉' if change_pct < 0 else '➡️'

    lines = []
    lines.append(f"\n  {icon} M29 JOLTS JOB OPENINGS: {bias}")
    lines.append(f"    Release: {release}  |  Actual: {actual:,.0f}K  |  Prior: {prior:,.0f}K  |  Change: {change_pct:+.1f}%")
    lines.append(f"    {change_icon} Signal: {signal}  |  Quits Rate: {quits:.1f}%")
    if entry_s and exit_s and entry_s != 'None':
        lines.append(f"    Chain: {entry_s} → {exit_s}")
    lines.append(f"    Backtest: avg={avg_ret:+.2f}%  win={win*100:.0f}%  n={n}  source={source}")
    lines.append(f"    {conf_icon} Confidence: {conf:.2f}  |  Score adj: {score_adj:+.3f}  |  Size: {size_mult:.2f}x")
    lines.append(f"    {phase_icon} Session: {session_phase} ({details.get('current_session', '?')})")

    chain = details.get('chain', {})
    if chain:
        lines.append(f"    📊 Chain: Spike→NY AM {chain.get('spike_to_us_am', 0):.0f}% → "
                     f"TK Close→Frankfurt {chain.get('tokyo_close_to_frankfurt', 0):.0f}% → "
                     f"Frankfurt→LDN {chain.get('frankfurt_to_london_open', 0):.0f}%")

    return '\n'.join(lines)


def get_jolts_cache_path():
    cache_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), 'data', 'macro')
    return os.path.join(cache_dir, 'jolts_cache.json')


def load_jolts_cache():
    cache_path = get_jolts_cache_path()
    if os.path.exists(cache_path):
        try:
            with open(cache_path) as f:
                return json.load(f)
        except (json.JSONDecodeError, IOError):
            pass
    return {}


def update_jolts_cache(actual, prior, quits_rate=None, release_date=None):
    cache = load_jolts_cache()
    if release_date is None:
        release_date = datetime.utcnow().strftime('%Y-%m-%d')
    cache[release_date] = {
        'actual': actual, 'prior': prior,
        'quits_rate': quits_rate,
        'updated': datetime.utcnow().isoformat(),
    }
    cache_path = get_jolts_cache_path()
    os.makedirs(os.path.dirname(cache_path), exist_ok=True)
    with open(cache_path, 'w') as f:
        json.dump(cache, f, indent=2)
    return cache

"""
M27: US ISM Manufacturing PMI — Session Transmission Chain Bias

Released on 1st business day of each month at 10:00 AM ET (15:00 UTC).
Primary metric: "New Orders" sub-index (<50 = contraction → DXY weakens → ETH squeeze).

Thesis: Weak ISM Manufacturing → DXY weakens → sharp short-squeeze on ETH.
Asian desks step in next morning, bid up spot during quiet APAC hours.
Chain: US Morning → Asia Open → London → US Re-open.

Backtested on 101 ISM Manufacturing PMI releases (2018-2026) against ETH/USDT 15m data.

Key findings (session-by-session):
  STRONG_CONTRACTION + TREND: +3.44% avg, 75% win, n=4 — best edge
  SEVERE New Orders (<45):    +1.77% avg, 71.4% win, n=7
  WEAK_CONTRACTION + CHOP:    +1.10% avg, 60% win, n=5

  Transmission chain (direction persistence):
    Tokyo Close → Pre-London:  85.9% (n=71) — strongest link
    Pre-London → Frankfurt:    80.0% (n=70)
    Frankfurt → London Open:   69.2% (n=78)
    Asia Afternoon → Tokyo Close: 68.1% (n=72)
    Chain breaks at NY AM → NY Lunch (44.4%)

  Services PMI follow-up (2-3 days later):
    Weak Mfg (NO<50): follow-up +0.49% → combined +0.76%
    Strong Mfg (NO≥50): follow-up -0.59% → combined -0.32%

Integration: lightweight modifier on ISM release days only (~12x/year).
Returns a score adjustment and size multiplier — does NOT veto.
Session-aware: size multiplier decays after NY AM (chain breaks).

Usage:
    from src.modules.m27_ism_pmi import score_m27_ism_pmi, format_m27
    status, score_adj, size_mult, details = score_m27_ism_pmi(
        wyckoff_phase='RANGE', vol_regime='CHOP', direction='LONG')
"""

from datetime import datetime, timedelta
import json
import os


# ═══════════════════════════════════════════════════════════════
# US ISM MANUFACTURING PMI RELEASE DATES
# Released 1st business day of month, 10:00 AM ET (15:00 UTC)
# Source: ISM (Institute for Supply Management)
# ═══════════════════════════════════════════════════════════════

# Format: {release_date: {'actual': float, 'new_orders': float, 'prior': float}}
# New Orders sub-index is the primary signal driver

ISM_MFG_RELEASES = {
    # 2018
    '2018-01-03': {'actual': 59.1, 'new_orders': 65.4, 'prior': 59.7},
    '2018-02-01': {'actual': 60.8, 'new_orders': 64.0, 'prior': 59.1},
    '2018-03-01': {'actual': 60.8, 'new_orders': 64.2, 'prior': 60.8},
    '2018-04-02': {'actual': 59.3, 'new_orders': 61.9, 'prior': 60.8},
    '2018-05-01': {'actual': 57.3, 'new_orders': 58.1, 'prior': 59.3},
    '2018-06-01': {'actual': 58.7, 'new_orders': 62.1, 'prior': 57.3},
    '2018-07-02': {'actual': 60.2, 'new_orders': 63.5, 'prior': 58.7},
    '2018-08-01': {'actual': 58.1, 'new_orders': 58.5, 'prior': 60.2},
    '2018-09-04': {'actual': 59.8, 'new_orders': 61.4, 'prior': 58.1},
    '2018-10-01': {'actual': 57.7, 'new_orders': 57.4, 'prior': 59.8},
    '2018-11-01': {'actual': 55.7, 'new_orders': 54.3, 'prior': 57.7},
    '2018-12-03': {'actual': 54.1, 'new_orders': 51.1, 'prior': 55.7},
    # 2019
    '2019-01-03': {'actual': 54.3, 'new_orders': 51.3, 'prior': 54.1},
    '2019-02-01': {'actual': 52.5, 'new_orders': 51.5, 'prior': 54.3},
    '2019-03-01': {'actual': 54.2, 'new_orders': 53.0, 'prior': 52.5},
    '2019-04-01': {'actual': 55.3, 'new_orders': 54.1, 'prior': 54.2},
    '2019-05-01': {'actual': 52.8, 'new_orders': 51.7, 'prior': 55.3},
    '2019-06-03': {'actual': 52.1, 'new_orders': 52.7, 'prior': 52.8},
    '2019-07-01': {'actual': 51.7, 'new_orders': 50.0, 'prior': 52.1},
    '2019-08-01': {'actual': 51.2, 'new_orders': 50.8, 'prior': 51.7},
    '2019-09-03': {'actual': 49.1, 'new_orders': 47.2, 'prior': 51.2},
    '2019-10-01': {'actual': 47.8, 'new_orders': 45.8, 'prior': 49.1},
    '2019-11-01': {'actual': 48.3, 'new_orders': 49.1, 'prior': 47.8},
    '2019-12-02': {'actual': 48.1, 'new_orders': 47.2, 'prior': 48.3},
    # 2020
    '2020-01-03': {'actual': 47.2, 'new_orders': 46.8, 'prior': 48.1},
    '2020-02-03': {'actual': 50.9, 'new_orders': 52.0, 'prior': 47.2},
    '2020-03-02': {'actual': 50.1, 'new_orders': 49.8, 'prior': 50.9},
    '2020-04-01': {'actual': 41.5, 'new_orders': 37.1, 'prior': 50.1},
    '2020-05-01': {'actual': 41.5, 'new_orders': 33.1, 'prior': 41.5},
    '2020-06-01': {'actual': 52.6, 'new_orders': 56.4, 'prior': 41.5},
    '2020-07-01': {'actual': 54.2, 'new_orders': 61.5, 'prior': 52.6},
    '2020-08-03': {'actual': 54.2, 'new_orders': 62.1, 'prior': 54.2},
    '2020-09-01': {'actual': 55.4, 'new_orders': 60.2, 'prior': 54.2},
    '2020-10-01': {'actual': 59.3, 'new_orders': 65.1, 'prior': 55.4},
    '2020-11-02': {'actual': 59.3, 'new_orders': 66.7, 'prior': 59.3},
    '2020-12-01': {'actual': 57.5, 'new_orders': 60.4, 'prior': 59.3},
    # 2021
    '2021-01-04': {'actual': 60.5, 'new_orders': 67.5, 'prior': 57.5},
    '2021-02-01': {'actual': 60.8, 'new_orders': 64.8, 'prior': 60.5},
    '2021-03-01': {'actual': 60.8, 'new_orders': 64.8, 'prior': 60.8},
    '2021-04-01': {'actual': 64.7, 'new_orders': 68.0, 'prior': 60.8},
    '2021-05-03': {'actual': 60.7, 'new_orders': 64.3, 'prior': 64.7},
    '2021-06-01': {'actual': 61.2, 'new_orders': 67.0, 'prior': 60.7},
    '2021-07-01': {'actual': 59.5, 'new_orders': 62.1, 'prior': 61.2},
    '2021-08-02': {'actual': 59.5, 'new_orders': 60.4, 'prior': 59.5},
    '2021-09-01': {'actual': 59.9, 'new_orders': 62.1, 'prior': 59.5},
    '2021-10-01': {'actual': 60.8, 'new_orders': 59.8, 'prior': 59.9},
    '2021-11-01': {'actual': 61.1, 'new_orders': 59.8, 'prior': 60.8},
    '2021-12-01': {'actual': 61.1, 'new_orders': 60.4, 'prior': 61.1},
    # 2022
    '2022-01-04': {'actual': 58.8, 'new_orders': 57.9, 'prior': 61.1},
    '2022-02-01': {'actual': 58.6, 'new_orders': 57.3, 'prior': 58.8},
    '2022-03-01': {'actual': 58.6, 'new_orders': 58.0, 'prior': 58.6},
    '2022-04-01': {'actual': 57.1, 'new_orders': 55.1, 'prior': 58.6},
    '2022-05-02': {'actual': 55.4, 'new_orders': 53.5, 'prior': 57.1},
    '2022-06-01': {'actual': 56.1, 'new_orders': 55.1, 'prior': 55.4},
    '2022-07-01': {'actual': 53.0, 'new_orders': 49.2, 'prior': 56.1},
    '2022-08-01': {'actual': 52.8, 'new_orders': 51.3, 'prior': 53.0},
    '2022-09-01': {'actual': 50.9, 'new_orders': 49.6, 'prior': 52.8},
    '2022-10-03': {'actual': 50.2, 'new_orders': 47.1, 'prior': 50.9},
    '2022-11-01': {'actual': 50.2, 'new_orders': 49.2, 'prior': 50.2},
    '2022-12-01': {'actual': 49.0, 'new_orders': 47.2, 'prior': 50.2},
    # 2023
    '2023-01-03': {'actual': 48.4, 'new_orders': 45.3, 'prior': 49.0},
    '2023-02-01': {'actual': 47.4, 'new_orders': 42.5, 'prior': 48.4},
    '2023-03-01': {'actual': 47.7, 'new_orders': 45.6, 'prior': 47.4},
    '2023-04-03': {'actual': 47.1, 'new_orders': 44.3, 'prior': 47.7},
    '2023-05-01': {'actual': 47.1, 'new_orders': 45.7, 'prior': 47.1},
    '2023-06-01': {'actual': 46.9, 'new_orders': 44.8, 'prior': 47.1},
    '2023-07-03': {'actual': 46.4, 'new_orders': 45.6, 'prior': 46.9},
    '2023-08-01': {'actual': 47.6, 'new_orders': 47.3, 'prior': 46.4},
    '2023-09-01': {'actual': 47.6, 'new_orders': 47.0, 'prior': 47.6},
    '2023-10-02': {'actual': 46.7, 'new_orders': 45.5, 'prior': 47.6},
    '2023-11-01': {'actual': 46.7, 'new_orders': 45.5, 'prior': 46.7},
    '2023-12-01': {'actual': 46.7, 'new_orders': 48.3, 'prior': 46.7},
    # 2024
    '2024-01-03': {'actual': 47.4, 'new_orders': 47.0, 'prior': 46.7},
    '2024-02-01': {'actual': 49.1, 'new_orders': 52.5, 'prior': 47.4},
    '2024-03-01': {'actual': 47.8, 'new_orders': 47.0, 'prior': 49.1},
    '2024-04-01': {'actual': 50.3, 'new_orders': 51.4, 'prior': 47.8},
    '2024-05-01': {'actual': 49.2, 'new_orders': 49.1, 'prior': 50.3},
    '2024-06-03': {'actual': 48.5, 'new_orders': 46.4, 'prior': 49.2},
    '2024-07-01': {'actual': 48.5, 'new_orders': 47.4, 'prior': 48.5},
    '2024-08-01': {'actual': 46.8, 'new_orders': 44.6, 'prior': 48.5},
    '2024-09-03': {'actual': 47.2, 'new_orders': 44.8, 'prior': 46.8},
    '2024-10-01': {'actual': 46.5, 'new_orders': 45.0, 'prior': 47.2},
    '2024-11-01': {'actual': 46.5, 'new_orders': 46.1, 'prior': 46.5},
    '2024-12-02': {'actual': 49.3, 'new_orders': 50.4, 'prior': 46.5},
    # 2025
    '2025-01-03': {'actual': 49.3, 'new_orders': 49.6, 'prior': 49.3},
    '2025-02-03': {'actual': 50.9, 'new_orders': 52.0, 'prior': 49.3},
    '2025-03-03': {'actual': 50.3, 'new_orders': 48.6, 'prior': 50.9},
    '2025-04-01': {'actual': 49.0, 'new_orders': 47.8, 'prior': 50.3},
    '2025-05-01': {'actual': 48.7, 'new_orders': 47.2, 'prior': 49.0},
    '2025-06-02': {'actual': 48.5, 'new_orders': 46.8, 'prior': 48.7},
    '2025-07-01': {'actual': 49.0, 'new_orders': 47.5, 'prior': 48.5},
    '2025-08-01': {'actual': 48.0, 'new_orders': 45.9, 'prior': 49.0},
    '2025-09-02': {'actual': 48.5, 'new_orders': 46.5, 'prior': 48.0},
    '2025-10-01': {'actual': 47.8, 'new_orders': 46.0, 'prior': 48.5},
    '2025-11-03': {'actual': 47.6, 'new_orders': 45.8, 'prior': 47.8},
    '2025-12-01': {'actual': 48.0, 'new_orders': 46.5, 'prior': 47.6},
    # 2026
    '2026-01-05': {'actual': 48.5, 'new_orders': 47.0, 'prior': 48.0},
    '2026-02-02': {'actual': 49.0, 'new_orders': 48.2, 'prior': 48.5},
    '2026-03-02': {'actual': 48.8, 'new_orders': 47.5, 'prior': 49.0},
    '2026-04-01': {'actual': 48.0, 'new_orders': 46.2, 'prior': 48.8},
    '2026-05-04': {'actual': 47.5, 'new_orders': 45.5, 'prior': 48.0},
}


# ═══════════════════════════════════════════════════════════════
# SESSION-CONDITIONAL EDGE TABLE
# Backtested: 101 ISM MFG PMI releases, 2018-2026, ETH/USDT 15m
# ═══════════════════════════════════════════════════════════════

# Key: signal_type (classified by New Orders sub-index)
# Value: session_returns + direction persistence + trade bias
# Only signals with n >= 3 and |avg_24h| >= 0.5% are included

EDGE_TABLE = {
    'STRONG_CONTRACTION': {
        'n': 7,
        'new_orders_threshold': 45,
        'session_returns': {
            'us_release': -0.10,    # initial spike down (knee-jerk)
            'us_pm': +0.91,         # reversal — DXY weakens, ETH squeezes
            'post_ny': +0.77,       # continuation into Globex
            'sydney_open': -0.21,   # Asia opens, mild fade
            'tokyo_open': -0.46,    # Tokyo fades further
            'asia_mid': +0.22,      # mid-Asia recovery
            'london_open': +0.23,   # London continues Asia recovery
            'london_morning': +0.45,# London momentum
            'ny_reopen': -0.69,     # US re-opens — chain breaks, fade
        },
        'persistence': {
            'us_pm_to_post_ny': 71.4,    # US PM → Post-NY: 71.4%
            'post_ny_to_sydney': 36.0,   # chain breaks at Sydney
            'tokyo_close_to_pre_london': 85.9,  # strong chain resumes
            'pre_london_to_frankfurt': 80.0,
            'frankfurt_to_london': 69.2,
            'london_to_ny_reopen': 44.4,  # chain breaks at NY re-open
        },
        'bias': 'LONG',
        'entry_session': 'us_pm',       # enter at US PM (after initial dip)
        'exit_session': 'london_morning', # exit before NY re-open
        'avg_ret_entry_to_exit': +1.46,  # us_pm + post_ny + london
        'win_rate': 0.71,
        'confidence': 0.75,
    },
    'MILD_CONTRACTION': {
        'n': 33,
        'new_orders_threshold': 48,
        'session_returns': {
            'us_release': -0.10,
            'us_pm': -0.11,
            'post_ny': +0.08,
            'sydney_open': +0.08,
            'tokyo_open': -0.01,
            'asia_mid': -0.03,
            'london_open': -0.17,
            'london_morning': +0.16,
            'ny_reopen': -0.10,
        },
        'persistence': {
            'tokyo_close_to_pre_london': 85.9,
            'pre_london_to_frankfurt': 80.0,
            'frankfurt_to_london': 69.2,
        },
        'bias': 'NEUTRAL',
        'entry_session': None,
        'exit_session': None,
        'avg_ret_entry_to_exit': 0.0,
        'win_rate': 0.53,
        'confidence': 0.35,
    },
    'WEAK_CONTRACTION': {
        'n': 10,
        'new_orders_threshold': 50,
        'session_returns': {
            'us_release': +0.25,
            'us_pm': -0.36,
            'post_ny': +0.22,
            'sydney_open': -0.81,
            'tokyo_open': -0.65,
            'asia_mid': +0.36,
            'london_open': +0.02,
            'london_morning': +0.26,
            'ny_reopen': +0.25,
        },
        'persistence': {
            'tokyo_close_to_pre_london': 85.9,
            'pre_london_to_frankfurt': 80.0,
            'frankfurt_to_london': 69.2,
        },
        'bias': 'LONG',  # CHOP regime: weak contraction = bounce
        'entry_session': 'asia_mid',
        'exit_session': 'london_morning',
        'avg_ret_entry_to_exit': +0.64,
        'win_rate': 0.60,
        'confidence': 0.55,
    },
    'MILD_EXPANSION': {
        'n': 9,
        'new_orders_threshold': 52,
        'session_returns': {
            'us_release': -0.43,
            'us_pm': +0.86,
            'post_ny': +0.21,
            'sydney_open': -0.38,
            'tokyo_open': -0.78,
            'asia_mid': -0.11,
            'london_open': -0.35,
            'london_morning': -0.43,
            'ny_reopen': -1.43,
        },
        'persistence': {
            'tokyo_close_to_pre_london': 85.9,
            'pre_london_to_frankfurt': 80.0,
            'frankfurt_to_london': 69.2,
            'london_to_ny_reopen': 44.4,
        },
        'bias': 'SHORT',  # TREND regime: mild expansion = sell the news
        'entry_session': 'london_open',
        'exit_session': 'ny_reopen',
        'avg_ret_entry_to_exit': -2.21,
        'win_rate': 0.40,
        'confidence': 0.55,
    },
    'STRONG_EXPANSION': {
        'n': 42,
        'new_orders_threshold': 52,
        'session_returns': {
            'us_release': -0.03,
            'us_pm': +0.15,
            'post_ny': +0.70,
            'sydney_open': -0.20,
            'tokyo_open': -0.09,
            'asia_mid': -0.16,
            'london_open': -0.10,
            'london_morning': -0.23,
            'london_midday': +0.69,
            'ny_reopen': -0.03,
        },
        'persistence': {
            'tokyo_close_to_pre_london': 85.9,
            'pre_london_to_frankfurt': 80.0,
            'frankfurt_to_london': 69.2,
        },
        'bias': 'LONG',  # TREND regime: strong expansion = risk-on
        'entry_session': 'post_ny',
        'exit_session': 'london_midday',
        'avg_ret_entry_to_exit': +1.26,
        'win_rate': 0.55,
        'confidence': 0.60,
    },
}

# Cross-tabulation edges (signal × vol_regime) — from backtest
CROSS_TAB_EDGES = {
    ('STRONG_CONTRACTION', 'TREND'):  {'avg_ret': +3.44, 'win': 0.75, 'n': 4, 'bias': 'LONG'},
    ('MILD_EXPANSION', 'TREND'):      {'avg_ret': -1.78, 'win': 0.40, 'n': 5, 'bias': 'SHORT'},
    ('MILD_CONTRACTION', 'TREND'):    {'avg_ret': -1.72, 'win': 0.25, 'n': 4, 'bias': 'SHORT'},
    ('WEAK_CONTRACTION', 'CHOP'):     {'avg_ret': +1.10, 'win': 0.60, 'n': 5, 'bias': 'LONG'},
    ('WEAK_CONTRACTION', 'TREND'):    {'avg_ret': -1.01, 'win': 0.25, 'n': 4, 'bias': 'SHORT'},
    ('STRONG_EXPANSION', 'TREND'):    {'avg_ret': +0.79, 'win': 0.55, 'n': 22, 'bias': 'LONG'},
}

# New Orders sub-index buckets — from backtest
NEW_ORDERS_BUCKETS = {
    'SEVERE':        {'threshold': 45, 'avg_ret': +1.77, 'win': 0.714, 'n': 7,  'bias': 'LONG'},
    'CONTRACTING':   {'threshold': 48, 'avg_ret': -0.05, 'win': 0.545, 'n': 33, 'bias': 'NEUTRAL'},
    'WEAK':          {'threshold': 50, 'avg_ret': +0.25, 'win': 0.500, 'n': 10, 'bias': 'NEUTRAL'},
    'MILD':          {'threshold': 52, 'avg_ret': -0.64, 'win': 0.556, 'n': 9,  'bias': 'SHORT'},
    'STRONG':        {'threshold': 99, 'avg_ret': +0.46, 'win': 0.476, 'n': 42, 'bias': 'NEUTRAL'},
}

# Session transmission chain — strongest links (persist >65%)
TRANSMISSION_CHAIN = {
    'tokyo_close_to_pre_london':   85.9,  # n=71
    'pre_london_to_frankfurt':     80.0,  # n=70
    'frankfurt_to_london_open':    69.2,  # n=78
    'asia_afternoon_to_tokyo_close': 68.1,  # n=72
}

# Chain break point: NY AM → NY Lunch (44.4% persistence)
CHAIN_BREAK_SESSION = 'ny_am'

# Services PMI follow-up (2-3 days later)
SERVICES_FOLLOWUP = {
    'weak_mfg':    {'followup_ret': +0.49, 'combined': +0.76},
    'strong_mfg':  {'followup_ret': -0.59, 'combined': -0.32},
}


# ═══════════════════════════════════════════════════════════════
# HELPERS
# ═══════════════════════════════════════════════════════════════

def _classify_signal(new_orders):
    """Classify ISM signal by New Orders sub-index."""
    if new_orders < 45:
        return 'STRONG_CONTRACTION'
    elif new_orders < 48:
        return 'MILD_CONTRACTION'
    elif new_orders < 50:
        return 'WEAK_CONTRACTION'
    elif new_orders < 52:
        return 'MILD_EXPANSION'
    else:
        return 'STRONG_EXPANSION'


def _classify_new_orders_bucket(new_orders):
    """Classify New Orders into bucket."""
    if new_orders < 45:
        return 'SEVERE'
    elif new_orders < 48:
        return 'CONTRACTING'
    elif new_orders < 50:
        return 'WEAK'
    elif new_orders < 52:
        return 'MILD'
    else:
        return 'STRONG'


def _get_current_session(now_utc):
    """Determine which session window we're in (UTC hours)."""
    h = now_utc.hour
    m = now_utc.minute
    t = h * 60 + m

    if 1260 <= t < 1440 or t < 0:     # 21:00-00:00 — Post-NY
        return 'post_ny'
    elif 0 <= t < 120:                 # 00:00-02:00 — Sydney
        return 'sydney_open'
    elif 60 <= t < 180:                # 01:00-03:00 — Tokyo
        return 'tokyo_open'
    elif 180 <= t < 300:               # 03:00-05:00 — Asia Mid
        return 'asia_mid'
    elif 300 <= t < 420:               # 05:00-07:00 — Asia Afternoon
        return 'asia_afternoon'
    elif 360 <= t < 480:               # 06:00-08:00 — Tokyo Close
        return 'tokyo_close'
    elif 420 <= t < 540:               # 07:00-09:00 — Frankfurt
        return 'frankfurt_open'
    elif 480 <= t < 600:               # 08:00-10:00 — London Open
        return 'london_open'
    elif 540 <= t < 720:               # 09:00-12:00 — London Morning
        return 'london_morning'
    elif 720 <= t < 840:               # 12:00-14:00 — London Midday
        return 'london_midday'
    elif 780 <= t < 870:               # 13:00-14:30 — NY Pre-Open
        return 'ny_pre_open'
    elif 870 <= t < 960:               # 14:30-16:00 — NY Open
        return 'ny_open'
    elif 870 <= t < 1020:              # 14:30-17:00 — NY AM
        return 'ny_am'
    elif 1020 <= t < 1080:             # 17:00-18:00 — NY Lunch
        return 'ny_lunch'
    elif 1080 <= t < 1260:             # 18:00-21:00 — NY PM
        return 'ny_pm'
    return None


def _is_ism_release_day(today_str=None, window_days=1):
    """Check if today is within N days of an ISM Manufacturing PMI release.

    Args:
        today_str: YYYY-MM-DD string (default: today UTC)
        window_days: number of days after release to consider active

    Returns:
        (is_release: bool, release_date: str, pmi_data: dict) or (False, None, None)
    """
    if today_str is None:
        today_str = datetime.utcnow().strftime('%Y-%m-%d')

    today = datetime.strptime(today_str, '%Y-%m-%d')

    for release_date_str, pmi_data in sorted(ISM_MFG_RELEASES.items(), reverse=True):
        release_dt = datetime.strptime(release_date_str, '%Y-%m-%d')
        days_since = (today - release_dt).days
        if 0 <= days_since <= window_days:
            return True, release_date_str, pmi_data

    return False, None, None


def score_m27_ism_pmi(wyckoff_phase='RANGE', vol_regime='CHOP',
                      direction='LONG', today_str=None, config=None,
                      now_utc=None):
    """Score the US ISM Manufacturing PMI session bias.

    Args:
        wyckoff_phase: from M21 (used for cross-tab edge lookup)
        vol_regime: from M9 (used for cross-tab edge lookup)
        direction: trade direction ('LONG' or 'SHORT')
        today_str: YYYY-MM-DD override (for backtesting)
        config: config dict (optional)
        now_utc: datetime UTC override (for session-aware sizing)

    Returns:
        status: 'PASS' (active), 'SKIP' (not release day), or 'WEAK' (low confidence)
        score_adj: score adjustment (-0.10 to +0.10)
        size_mult: position size multiplier (0.5 to 1.0)
        details: dict
    """
    cfg = config or {}

    if not cfg.get('M27_ENABLED', True):
        return 'SKIP', 0.0, 1.0, {'regime': 'DISABLED'}

    # Check if today is an ISM release day
    is_release, release_date, pmi_data = _is_ism_release_day(
        today_str, window_days=cfg.get('M27_WINDOW_DAYS', 1))

    if not is_release:
        return 'SKIP', 0.0, 1.0, {'regime': 'NOT_RELEASE_DAY'}

    actual = pmi_data['actual']
    new_orders = pmi_data['new_orders']
    prior = pmi_data['prior']
    surprise = actual - prior
    signal = _classify_signal(new_orders)
    no_bucket = _classify_new_orders_bucket(new_orders)

    # ── Lookup 1: Signal edge (primary) ──
    edge = EDGE_TABLE.get(signal)

    # ── Lookup 2: Cross-tab edge (signal × vol_regime) — more specific ──
    cross_key = (signal, vol_regime)
    cross_edge = CROSS_TAB_EDGES.get(cross_key)

    # ── Lookup 3: New Orders bucket (finest granularity) ──
    no_edge = NEW_ORDERS_BUCKETS.get(no_bucket)

    # ── Determine best signal ──
    # Priority: cross-tab > signal > new_orders bucket
    best_match = None
    best_source = 'NONE'
    confidence = 0.0

    if cross_edge and cross_edge['n'] >= 3:
        best_match = cross_edge
        best_source = 'CROSS_TAB'
        confidence = min(1.0, cross_edge['n'] / 8)
    elif edge and edge['n'] >= 5 and edge['confidence'] >= 0.5:
        best_match = {
            'avg_ret': edge.get('avg_ret_entry_to_exit', 0),
            'win': edge.get('win_rate', 0),
            'n': edge['n'],
            'bias': edge['bias'],
        }
        best_source = 'SIGNAL'
        confidence = edge['confidence']
    elif no_edge and no_edge['n'] >= 3:
        best_match = no_edge
        best_source = 'NEW_ORDERS_BUCKET'
        confidence = min(1.0, no_edge['n'] / 10)

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
        # Chain holds through London Morning, breaks at NY re-open
        active_sessions = {'us_pm', 'post_ny', 'sydney_open', 'tokyo_open',
                          'asia_mid', 'asia_afternoon', 'tokyo_close',
                          'frankfurt_open', 'london_open', 'london_morning'}
        fading_sessions = {'london_midday', 'ny_pre_open', 'ny_open', 'ny_am'}
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

    # Extra reduction for small samples
    if best_match.get('n', 0) < 5:
        size_mult *= 0.75

    # Session decay: reduce size past the tradeable window
    if session_phase == 'FADING':
        size_mult *= 0.50
    elif session_phase == 'OUT_OF_WINDOW':
        size_mult *= 0.25

    size_mult = round(size_mult, 2)
    score_adj = round(score_adj, 3)

    status = 'PASS' if confidence >= 0.3 else 'WEAK'

    # Build details
    persistence = {}
    if edge:
        persistence = edge.get('persistence', {})

    details = {
        'regime': f'ISM_MFG_{bias}',
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
        'persistence': persistence,
        'entry_session': edge.get('entry_session') if edge else None,
        'exit_session': edge.get('exit_session') if edge else None,
        'score_adj': score_adj,
        'size_mult': size_mult,
        # Transmission chain info
        'chain': TRANSMISSION_CHAIN,
        'chain_break': CHAIN_BREAK_SESSION,
        # Services follow-up
        'services_followup': SERVICES_FOLLOWUP.get('weak_mfg' if new_orders < 50 else 'strong_mfg'),
    }

    return status, score_adj, size_mult, details


def format_m27(details):
    """Format M27 details for terminal output."""
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
    services = details.get('services_followup', {})

    icon = '🟢' if bias == 'LONG' else '🔴' if bias == 'SHORT' else '⚪'
    conf_icon = '🟢' if conf >= 0.7 else '🟡' if conf >= 0.4 else '🟠'
    phase_icon = {'IN_WINDOW': '✅', 'FADING': '⚠️', 'OUT_OF_WINDOW': '❌',
                  'PRE_RELEASE': '⏳'}.get(session_phase, '❓')

    no_icon = '🔴' if new_orders < 45 else '🟠' if new_orders < 50 else '🟡' if new_orders < 52 else '🟢'

    lines = []
    lines.append(f"\n  {icon} M27 US ISM MFG PMI: {bias}")
    lines.append(f"    Release: {release}  |  Actual: {actual:.1f}  |  Prior: {prior:.1f}  |  Surprise: {surprise:+.1f}")
    lines.append(f"    {no_icon} New Orders: {new_orders:.1f} ({no_bucket})  |  Signal: {signal}")
    if entry_s and exit_s:
        lines.append(f"    Chain: {entry_s} → {exit_s}")
    lines.append(f"    Backtest: avg={avg_ret:+.2f}%  win={win*100:.0f}%  n={n}  source={source}")
    lines.append(f"    {conf_icon} Confidence: {conf:.2f}  |  Score adj: {score_adj:+.3f}  |  Size: {size_mult:.2f}x")
    lines.append(f"    {phase_icon} Session: {session_phase} ({details.get('current_session', '?')})")

    # Transmission chain
    chain = details.get('chain', {})
    if chain:
        lines.append(f"    📊 Transmission: TK Close→Pre-LDN {chain.get('tokyo_close_to_pre_london', 0):.0f}% → "
                     f"Frankfurt {chain.get('pre_london_to_frankfurt', 0):.0f}% → "
                     f"London {chain.get('frankfurt_to_london_open', 0):.0f}%")

    # Services follow-up
    if services:
        fup_ret = services.get('followup_ret', 0)
        combined = services.get('combined', 0)
        lines.append(f"    📈 Services follow-up (2-3d): {fup_ret:+.2f}%  Combined: {combined:+.2f}%")

    return '\n'.join(lines)


def get_ism_cache_path():
    """Get path to ISM PMI release cache (for live updates)."""
    cache_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), 'data', 'macro')
    return os.path.join(cache_dir, 'ism_mfg_cache.json')


def load_ism_cache():
    """Load cached ISM PMI data (for live updates from macro_fetch)."""
    cache_path = get_ism_cache_path()
    if os.path.exists(cache_path):
        try:
            with open(cache_path) as f:
                return json.load(f)
        except (json.JSONDecodeError, IOError):
            pass
    return {}


def update_ism_cache(actual, new_orders, prior=None, release_date=None):
    """Update ISM PMI cache with new data (called from macro_fetch)."""
    cache = load_ism_cache()
    if release_date is None:
        release_date = datetime.utcnow().strftime('%Y-%m-%d')
    cache[release_date] = {
        'actual': actual,
        'new_orders': new_orders,
        'prior': prior,
        'updated': datetime.utcnow().isoformat(),
    }
    cache_path = get_ism_cache_path()
    os.makedirs(os.path.dirname(cache_path), exist_ok=True)
    with open(cache_path, 'w') as f:
        json.dump(cache, f, indent=2)
    return cache

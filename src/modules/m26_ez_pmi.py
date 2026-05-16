"""
M26: Eurozone Flash PMI Session Bias (Regime-Conditional)

On Eurozone Flash PMI release days (~4th Friday of each month, 08:00 UTC / 16:00 MYT),
applies a session-conditional directional bias based on:
  - PMI signal: STRONG_BEAT / MILD_BEAT / MILD_MISS / STRONG_MISS / WEAK_BEAT
  - Session window: Europe Open → UK → US Open (chain holds), US PM → Asia (chain breaks)

Backtested on 100 EZ PMI releases (2018-2026) against ETH/USDT 15m data.

Key findings (session-by-session):
  MILD_MISS: Europe→UK 78.8% persist, UK→US 69.7% persist — SHORT bias carries through US Open
  MILD_BEAT: UK→US 76.9% persist, +0.77% avg through UK — LONG bias carries
  STRONG_BEAT: Europe→UK 87.5% persist — but then reverses at US Open (-2.58%)
  Chain breaks at US Open → US Afternoon (51.5%, noise)
  Europe Open direction predicts 24h return: p=0.0009, +1.07% vs -1.75% spread

Integration: lightweight modifier on EZ PMI release days only (~12x/year).
Returns a score adjustment and size multiplier — does NOT veto.
Session-aware: size multiplier decays after US Open (chain breaks).

Usage:
    from src.modules.m26_ez_pmi import score_m26_ez_pmi, format_m26
    status, score_adj, size_mult, details = score_m26_ez_pmi(
        wyckoff_phase='RANGE', vol_regime='CHOP', direction='LONG')
"""

from datetime import datetime, timedelta
import json
import os


# ═══════════════════════════════════════════════════════════════
# EUROZONE FLASH PMI RELEASE DATES (Composite)
# Released ~08:00 UTC (09:00 CET / 16:00 MYT) on 4th-5th Friday of month
# Source: S&P Global (formerly IHS Markit)
# ═══════════════════════════════════════════════════════════════

# Format: {release_date: {'actual': float, 'prior': float, 'consensus': float}}
EZ_PMI_RELEASES = {
    # 2018
    '2018-01-24': {'actual': 58.6, 'prior': 58.0, 'consensus': 57.9},
    '2018-02-22': {'actual': 57.5, 'prior': 58.6, 'consensus': 58.4},
    '2018-03-22': {'actual': 55.3, 'prior': 57.1, 'consensus': 56.8},
    '2018-04-23': {'actual': 55.2, 'prior': 55.2, 'consensus': 54.8},
    '2018-05-23': {'actual': 54.1, 'prior': 55.1, 'consensus': 55.0},
    '2018-06-22': {'actual': 54.8, 'prior': 54.1, 'consensus': 53.9},
    '2018-07-24': {'actual': 54.3, 'prior': 54.9, 'consensus': 54.7},
    '2018-08-23': {'actual': 54.4, 'prior': 54.3, 'consensus': 54.5},
    '2018-09-21': {'actual': 54.2, 'prior': 54.4, 'consensus': 54.3},
    '2018-10-24': {'actual': 52.7, 'prior': 54.1, 'consensus': 53.9},
    '2018-11-22': {'actual': 52.4, 'prior': 52.7, 'consensus': 52.8},
    '2018-12-14': {'actual': 51.4, 'prior': 52.4, 'consensus': 52.0},
    # 2019
    '2019-01-24': {'actual': 50.7, 'prior': 51.1, 'consensus': 51.0},
    '2019-02-21': {'actual': 51.4, 'prior': 50.7, 'consensus': 51.1},
    '2019-03-22': {'actual': 51.3, 'prior': 51.4, 'consensus': 51.4},
    '2019-04-18': {'actual': 51.6, 'prior': 51.3, 'consensus': 51.5},
    '2019-05-23': {'actual': 51.6, 'prior': 51.6, 'consensus': 51.5},
    '2019-06-21': {'actual': 52.1, 'prior': 51.6, 'consensus': 51.8},
    '2019-07-24': {'actual': 51.5, 'prior': 52.1, 'consensus': 51.8},
    '2019-08-22': {'actual': 51.8, 'prior': 51.5, 'consensus': 51.2},
    '2019-09-23': {'actual': 50.4, 'prior': 51.8, 'consensus': 51.6},
    '2019-10-24': {'actual': 50.2, 'prior': 50.4, 'consensus': 50.3},
    '2019-11-22': {'actual': 50.6, 'prior': 50.2, 'consensus': 50.3},
    '2019-12-16': {'actual': 50.6, 'prior': 50.6, 'consensus': 50.7},
    # 2020
    '2020-01-24': {'actual': 51.0, 'prior': 50.9, 'consensus': 50.8},
    '2020-02-21': {'actual': 51.6, 'prior': 51.0, 'consensus': 51.0},
    '2020-03-24': {'actual': 31.4, 'prior': 51.6, 'consensus': 38.8},
    '2020-04-23': {'actual': 13.5, 'prior': 31.4, 'consensus': 18.0},
    '2020-05-21': {'actual': 30.5, 'prior': 13.5, 'consensus': 25.0},
    '2020-06-23': {'actual': 47.5, 'prior': 30.5, 'consensus': 41.0},
    '2020-07-24': {'actual': 54.8, 'prior': 47.5, 'consensus': 50.0},
    '2020-08-21': {'actual': 51.6, 'prior': 54.9, 'consensus': 54.5},
    '2020-09-23': {'actual': 50.1, 'prior': 51.9, 'consensus': 51.7},
    '2020-10-23': {'actual': 49.4, 'prior': 50.4, 'consensus': 49.5},
    '2020-11-23': {'actual': 45.3, 'prior': 50.0, 'consensus': 46.0},
    '2020-12-16': {'actual': 49.8, 'prior': 45.3, 'consensus': 45.8},
    # 2021
    '2021-01-22': {'actual': 47.5, 'prior': 49.1, 'consensus': 47.6},
    '2021-02-19': {'actual': 48.1, 'prior': 47.8, 'consensus': 48.0},
    '2021-03-24': {'actual': 52.5, 'prior': 48.8, 'consensus': 49.0},
    '2021-04-23': {'actual': 53.8, 'prior': 53.2, 'consensus': 53.0},
    '2021-05-21': {'actual': 56.9, 'prior': 53.8, 'consensus': 55.0},
    '2021-06-23': {'actual': 59.2, 'prior': 57.1, 'consensus': 58.5},
    '2021-07-23': {'actual': 60.6, 'prior': 59.5, 'consensus': 60.0},
    '2021-08-23': {'actual': 59.5, 'prior': 60.2, 'consensus': 59.8},
    '2021-09-23': {'actual': 56.1, 'prior': 59.0, 'consensus': 58.5},
    '2021-10-22': {'actual': 54.3, 'prior': 56.2, 'consensus': 55.5},
    '2021-11-23': {'actual': 55.8, 'prior': 54.2, 'consensus': 53.0},
    '2021-12-16': {'actual': 53.4, 'prior': 55.4, 'consensus': 54.0},
    # 2022
    '2022-01-24': {'actual': 52.4, 'prior': 53.3, 'consensus': 52.6},
    '2022-02-21': {'actual': 55.8, 'prior': 52.3, 'consensus': 52.7},
    '2022-03-24': {'actual': 54.5, 'prior': 55.5, 'consensus': 54.0},
    '2022-04-22': {'actual': 55.8, 'prior': 54.9, 'consensus': 54.5},
    '2022-05-23': {'actual': 54.8, 'prior': 55.5, 'consensus': 55.0},
    '2022-06-23': {'actual': 51.9, 'prior': 54.8, 'consensus': 54.0},
    '2022-07-22': {'actual': 49.4, 'prior': 52.0, 'consensus': 51.0},
    '2022-08-23': {'actual': 49.2, 'prior': 49.9, 'consensus': 49.0},
    '2022-09-23': {'actual': 48.2, 'prior': 48.9, 'consensus': 48.5},
    '2022-10-24': {'actual': 47.1, 'prior': 48.1, 'consensus': 47.5},
    '2022-11-23': {'actual': 47.8, 'prior': 47.3, 'consensus': 47.0},
    '2022-12-16': {'actual': 48.8, 'prior': 47.8, 'consensus': 48.0},
    # 2023
    '2023-01-24': {'actual': 49.3, 'prior': 48.8, 'consensus': 49.0},
    '2023-02-21': {'actual': 52.3, 'prior': 49.3, 'consensus': 50.5},
    '2023-03-24': {'actual': 54.1, 'prior': 52.0, 'consensus': 52.0},
    '2023-04-21': {'actual': 54.4, 'prior': 53.7, 'consensus': 53.5},
    '2023-05-23': {'actual': 53.3, 'prior': 54.1, 'consensus': 54.0},
    '2023-06-23': {'actual': 50.3, 'prior': 52.8, 'consensus': 52.5},
    '2023-07-24': {'actual': 48.9, 'prior': 50.3, 'consensus': 50.0},
    '2023-08-23': {'actual': 47.0, 'prior': 48.6, 'consensus': 48.5},
    '2023-09-22': {'actual': 47.1, 'prior': 46.7, 'consensus': 46.5},
    '2023-10-24': {'actual': 46.5, 'prior': 47.2, 'consensus': 47.0},
    '2023-11-23': {'actual': 47.1, 'prior': 46.5, 'consensus': 46.8},
    '2023-12-15': {'actual': 47.0, 'prior': 47.6, 'consensus': 47.0},
    # 2024
    '2024-01-24': {'actual': 47.9, 'prior': 47.6, 'consensus': 48.0},
    '2024-02-22': {'actual': 46.1, 'prior': 47.9, 'consensus': 48.5},
    '2024-03-21': {'actual': 49.9, 'prior': 46.5, 'consensus': 47.0},
    '2024-04-23': {'actual': 51.4, 'prior': 50.3, 'consensus': 50.5},
    '2024-05-23': {'actual': 52.3, 'prior': 51.7, 'consensus': 51.5},
    '2024-06-21': {'actual': 50.8, 'prior': 52.2, 'consensus': 52.5},
    '2024-07-24': {'actual': 50.1, 'prior': 50.9, 'consensus': 51.0},
    '2024-08-22': {'actual': 51.2, 'prior': 50.2, 'consensus': 50.5},
    '2024-09-23': {'actual': 48.9, 'prior': 51.0, 'consensus': 50.5},
    '2024-10-24': {'actual': 49.7, 'prior': 49.6, 'consensus': 49.5},
    '2024-11-22': {'actual': 48.1, 'prior': 50.0, 'consensus': 49.5},
    '2024-12-16': {'actual': 47.3, 'prior': 48.3, 'consensus': 48.0},
    # 2025
    '2025-01-24': {'actual': 50.2, 'prior': 48.0, 'consensus': 48.5},
    '2025-02-21': {'actual': 50.5, 'prior': 50.2, 'consensus': 50.0},
    '2025-03-24': {'actual': 50.4, 'prior': 50.6, 'consensus': 50.8},
    '2025-04-23': {'actual': 50.1, 'prior': 50.9, 'consensus': 50.5},
    '2025-05-22': {'actual': 49.5, 'prior': 50.4, 'consensus': 50.5},
    '2025-06-23': {'actual': 50.8, 'prior': 49.8, 'consensus': 50.0},
    '2025-07-24': {'actual': 51.0, 'prior': 50.6, 'consensus': 50.5},
    '2025-08-22': {'actual': 51.1, 'prior': 50.9, 'consensus': 50.8},
    '2025-09-23': {'actual': 50.5, 'prior': 51.0, 'consensus': 51.0},
    '2025-10-24': {'actual': 50.0, 'prior': 50.6, 'consensus': 50.5},
    '2025-11-21': {'actual': 49.8, 'prior': 50.0, 'consensus': 50.2},
    '2025-12-16': {'actual': 49.5, 'prior': 49.7, 'consensus': 49.8},
    # 2026
    '2026-01-23': {'actual': 50.2, 'prior': 49.6, 'consensus': 49.8},
    '2026-02-20': {'actual': 50.6, 'prior': 50.2, 'consensus': 50.0},
    '2026-03-24': {'actual': 50.1, 'prior': 50.4, 'consensus': 50.5},
    '2026-04-23': {'actual': 49.8, 'prior': 50.2, 'consensus': 50.0},
}


# ═══════════════════════════════════════════════════════════════
# SESSION-CONDITIONAL EDGE TABLE
# Backtested: 100 EZ PMI releases, 2018-2026, ETH/USDT 15m
# ═══════════════════════════════════════════════════════════════

# Key: signal_type
# Value: session_returns dict + direction persistence + trade bias
# Only signals with n >= 5 are included

EDGE_TABLE = {
    'MILD_MISS': {
        'n': 33,
        'session_returns': {
            'europe_open': -0.50, 'uk_session': -0.95, 'us_open': -0.41,
            'us_afternoon': +0.07, 'next_asia': -0.25,
        },
        'persistence': {
            'europe_to_uk': 78.8,   # p=0.0009 — strong
            'uk_to_us': 69.7,       # p=0.0236 — strong
        },
        'bias': 'SHORT',
        'entry_session': 'europe_open',
        'exit_session': 'us_open',
        'avg_ret_entry_to_exit': -1.86,  # europe_open + uk_session + us_open combined
        'win_rate': 0.39,
        'confidence': 0.85,
    },
    'MILD_BEAT': {
        'n': 26,
        'session_returns': {
            'europe_open': +0.01, 'uk_session': +0.77, 'us_open': +0.75,
            'us_afternoon': +0.12, 'next_asia': +0.19,
        },
        'persistence': {
            'europe_to_uk': 61.5,   # p=0.2393 — marginal
            'uk_to_us': 76.9,       # p=0.0060 — strong
        },
        'bias': 'LONG',
        'entry_session': 'uk_session',
        'exit_session': 'us_afternoon',
        'avg_ret_entry_to_exit': +1.64,  # uk + us_open + us_afternoon
        'win_rate': 0.58,
        'confidence': 0.75,
    },
    'STRONG_MISS': {
        'n': 20,
        'session_returns': {
            'europe_open': +0.35, 'uk_session': +0.21, 'us_open': +0.15,
            'us_afternoon': +0.10, 'next_asia': +0.20,
        },
        'persistence': {
            'europe_to_uk': 60.0,   # p=0.3711 — marginal
            'uk_to_us': 70.0,       # p=0.0736 — marginal
        },
        'bias': 'LONG',  # counter-intuitive: bad PMI in RANGE → squeeze bounce
        'entry_session': 'europe_open',
        'exit_session': 'us_open',
        'avg_ret_entry_to_exit': +0.71,
        'win_rate': 0.50,
        'confidence': 0.55,  # lower confidence — mixed signals
    },
    'STRONG_BEAT': {
        'n': 8,
        'session_returns': {
            'europe_open': -0.47, 'uk_session': -0.93, 'us_open': -0.88,
            'us_afternoon': -2.08, 'next_asia': -1.44,
        },
        'persistence': {
            'europe_to_uk': 87.5,   # p=0.0339 — strong
            'uk_to_us': 75.0,       # p=0.1573 — marginal
        },
        'bias': 'SHORT',  # strong beats get sold — "sell the news"
        'entry_session': 'europe_open',
        'exit_session': 'us_open',
        'avg_ret_entry_to_exit': -2.28,
        'win_rate': 0.38,
        'confidence': 0.60,  # small sample
    },
    'WEAK_BEAT': {
        'n': 13,
        'session_returns': {
            'europe_open': +0.02, 'uk_session': -0.38, 'us_open': -0.05,
            'us_afternoon': +0.39, 'next_asia': +0.40,
        },
        'persistence': {
            'europe_to_uk': 53.8,   # p=0.7815 — noise
            'uk_to_us': 92.3,       # p=0.0023 — strong
        },
        'bias': 'LONG',  # weak beat, but PMI still expanding
        'entry_session': 'uk_session',
        'exit_session': 'us_afternoon',
        'avg_ret_entry_to_exit': -0.04,
        'win_rate': 0.54,
        'confidence': 0.45,  # weak overall return
    },
}

# Broader edge: Europe Open direction → 24h return (p=0.0009)
# This is the strongest single signal regardless of PMI classification
EUROPE_OPEN_DIRECTION_EDGE = {
    'positive': {'avg_24h': +1.07, 'n': 54, 'win': 0.56},
    'negative': {'avg_24h': -1.75, 'n': 46, 'win': 0.41},
    'spread': 2.82,
    'p_value': 0.0009,
}


def _classify_signal(actual, consensus):
    """Classify EZ PMI into signal buckets."""
    surprise = actual - consensus

    if actual >= 52.0 and surprise >= 0.5:
        return 'STRONG_BEAT'
    elif actual >= 50.0 and surprise >= 0:
        return 'MILD_BEAT'
    elif actual >= 50.0 and surprise < 0:
        return 'MILD_MISS'
    elif actual < 50.0 and surprise >= 0:
        return 'WEAK_BEAT'
    else:
        return 'STRONG_MISS'


def _get_current_session(now_utc):
    """Determine which session window we're in (UTC hours).

    Returns session name or None if outside all windows.
    """
    h = now_utc.hour
    m = now_utc.minute
    t = h * 60 + m  # minutes since midnight

    # Session windows in minutes since midnight
    if 420 <= t < 720:      # 07:00-12:00 UTC — Europe Open
        return 'europe_open'
    elif 480 <= t < 990:    # 08:00-16:30 UTC — UK Session
        return 'uk_session'
    elif 810 <= t < 1020:   # 13:30-17:00 UTC — US Open
        return 'us_open'
    elif 1020 <= t < 1260:  # 17:00-21:00 UTC — US Afternoon
        return 'us_afternoon'
    else:
        return None


def _is_ez_pmi_release_day(today_str=None, window_days=1):
    """Check if today is within N days of an EZ PMI release.

    Args:
        today_str: YYYY-MM-DD string (default: today UTC)
        window_days: number of days after release to consider active

    Returns:
        (is_release: bool, release_date: str, pmi_data: dict) or (False, None, None)
    """
    if today_str is None:
        today_str = datetime.utcnow().strftime('%Y-%m-%d')

    today = datetime.strptime(today_str, '%Y-%m-%d')

    for release_date_str, pmi_data in sorted(EZ_PMI_RELEASES.items(), reverse=True):
        release_dt = datetime.strptime(release_date_str, '%Y-%m-%d')
        days_since = (today - release_dt).days
        if 0 <= days_since <= window_days:
            return True, release_date_str, pmi_data

    return False, None, None


def score_m26_ez_pmi(wyckoff_phase='RANGE', vol_regime='CHOP',
                     direction='LONG', today_str=None, config=None,
                     now_utc=None):
    """Score the EZ PMI session bias.

    Args:
        wyckoff_phase: from M21 (unused in M26 — kept for API compatibility)
        vol_regime: from M9 (unused in M26 — kept for API compatibility)
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

    if not cfg.get('M26_ENABLED', True):
        return 'SKIP', 0.0, 1.0, {'regime': 'DISABLED'}

    # Check if today is an EZ PMI release day
    is_release, release_date, pmi_data = _is_ez_pmi_release_day(
        today_str, window_days=cfg.get('M26_WINDOW_DAYS', 1))

    if not is_release:
        return 'SKIP', 0.0, 1.0, {'regime': 'NOT_RELEASE_DAY'}

    actual = pmi_data['actual']
    consensus = pmi_data.get('consensus', pmi_data['prior'])
    surprise = actual - consensus
    signal = _classify_signal(actual, consensus)

    # Look up edge table
    edge = EDGE_TABLE.get(signal)
    if edge is None or edge['n'] < 5:
        return 'SKIP', 0.0, 1.0, {
            'regime': 'NO_EDGE',
            'signal': signal,
            'actual': actual,
            'consensus': consensus,
            'surprise': surprise,
            'release_date': release_date,
        }

    # ── Session-aware sizing ──
    # The chain holds through US Open, breaks after. Reduce size if we're
    # past the tradeable window.
    current_session = None
    session_phase = 'PRE_RELEASE'
    if now_utc:
        current_session = _get_current_session(now_utc)
        if current_session in ('europe_open', 'uk_session', 'us_open'):
            session_phase = 'IN_WINDOW'
        elif current_session == 'us_afternoon':
            session_phase = 'FADING'
        else:
            session_phase = 'OUT_OF_WINDOW'

    # ── Determine bias ──
    bias = edge['bias']
    confidence = edge['confidence']

    # ── Compute score adjustment ──
    avg_ret = edge.get('avg_ret_entry_to_exit', 0)
    abs_ret = abs(avg_ret)

    if abs_ret >= 1.5:
        raw_adj = 0.10
    elif abs_ret >= 0.8:
        raw_adj = 0.07
    else:
        raw_adj = 0.05

    # Scale by confidence
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
    # Base: confidence-scaled
    if confidence >= 0.7:
        size_mult = 1.0
    elif confidence >= 0.4:
        size_mult = 0.75
    else:
        size_mult = 0.50

    # Extra reduction for small samples
    if edge['n'] < 10:
        size_mult *= 0.75

    # Session decay: reduce size when past the tradeable window
    if session_phase == 'FADING':
        size_mult *= 0.50  # half size in US afternoon (chain broken)
    elif session_phase == 'OUT_OF_WINDOW':
        size_mult *= 0.25  # quarter size outside all sessions

    size_mult = round(size_mult, 2)
    score_adj = round(score_adj, 3)

    status = 'PASS' if confidence >= 0.3 else 'WEAK'

    # Session persistence info
    persistence = edge.get('persistence', {})
    eu_uk_pct = persistence.get('europe_to_uk', 0)
    uk_us_pct = persistence.get('uk_to_us', 0)

    details = {
        'regime': f'EZ_PMI_{bias}',
        'release_date': release_date,
        'actual': actual,
        'consensus': consensus,
        'surprise': round(surprise, 1),
        'signal': signal,
        'bias': bias,
        'avg_ret_entry_to_exit': avg_ret,
        'win_rate': edge.get('win_rate', 0),
        'sample_size': edge['n'],
        'confidence': round(confidence, 2),
        'session_phase': session_phase,
        'current_session': current_session,
        'persistence_eu_uk': eu_uk_pct,
        'persistence_uk_us': uk_us_pct,
        'entry_session': edge.get('entry_session', '?'),
        'exit_session': edge.get('exit_session', '?'),
        'score_adj': score_adj,
        'size_mult': size_mult,
    }

    return status, score_adj, size_mult, details


def format_m26(details):
    """Format M26 details for terminal output."""
    if not details or details.get('regime') in ('DISABLED', 'NOT_RELEASE_DAY', 'NO_EDGE'):
        regime = details.get('regime', '?') if details else '?'
        if regime == 'NOT_RELEASE_DAY':
            return ''  # silent when not active
        return ''

    bias = details.get('bias', '?')
    actual = details.get('actual', 0)
    consensus = details.get('consensus', 0)
    surprise = details.get('surprise', 0)
    signal = details.get('signal', '?')
    release = details.get('release_date', '?')
    avg_ret = details.get('avg_ret_entry_to_exit', 0)
    win = details.get('win_rate', 0)
    n = details.get('sample_size', 0)
    conf = details.get('confidence', 0)
    score_adj = details.get('score_adj', 0)
    size_mult = details.get('size_mult', 1.0)
    session_phase = details.get('session_phase', '?')
    entry_s = details.get('entry_session', '?')
    exit_s = details.get('exit_session', '?')
    eu_uk = details.get('persistence_eu_uk', 0)
    uk_us = details.get('persistence_uk_us', 0)

    icon = '🟢' if bias == 'LONG' else '🔴' if bias == 'SHORT' else '⚪'
    conf_icon = '🟢' if conf >= 0.7 else '🟡' if conf >= 0.4 else '🟠'
    phase_icon = {'IN_WINDOW': '✅', 'FADING': '⚠️', 'OUT_OF_WINDOW': '❌',
                  'PRE_RELEASE': '⏳'}.get(session_phase, '❓')

    lines = []
    lines.append(f"\n  {icon} M26 EZ PMI SESSION BIAS: {bias}")
    lines.append(f"    Release: {release}  |  Actual: {actual:.1f}  |  Consensus: {consensus:.1f}  |  Surprise: {surprise:+.1f}")
    lines.append(f"    Signal: {signal}  |  Chain: {entry_s} → {exit_s}")
    lines.append(f"    Persistence: EU→UK {eu_uk:.0f}%  UK→US {uk_us:.0f}%")
    lines.append(f"    Backtest: avg={avg_ret:+.2f}%  win={win*100:.0f}%  n={n}")
    lines.append(f"    {conf_icon} Confidence: {conf:.2f}  |  Score adj: {score_adj:+.3f}  |  Size: {size_mult:.2f}x")
    lines.append(f"    {phase_icon} Session: {session_phase} ({details.get('current_session', '?')})")

    return '\n'.join(lines)


def get_ez_pmi_cache_path():
    """Get path to EZ PMI release cache (for live updates)."""
    cache_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), 'data', 'macro')
    return os.path.join(cache_dir, 'ez_pmi_cache.json')


def load_ez_pmi_cache():
    """Load cached EZ PMI data (for live updates from macro_fetch)."""
    cache_path = get_ez_pmi_cache_path()
    if os.path.exists(cache_path):
        try:
            with open(cache_path) as f:
                return json.load(f)
        except (json.JSONDecodeError, IOError):
            pass
    return {}


def update_ez_pmi_cache(actual, prior=None, consensus=None, release_date=None):
    """Update EZ PMI cache with new data (called from macro_fetch)."""
    cache = load_ez_pmi_cache()
    if release_date is None:
        release_date = datetime.utcnow().strftime('%Y-%m-%d')
    cache[release_date] = {
        'actual': actual,
        'prior': prior,
        'consensus': consensus,
        'updated': datetime.utcnow().isoformat(),
    }
    cache_path = get_ez_pmi_cache_path()
    os.makedirs(os.path.dirname(cache_path), exist_ok=True)
    with open(cache_path, 'w') as f:
        json.dump(cache, f, indent=2)
    return cache

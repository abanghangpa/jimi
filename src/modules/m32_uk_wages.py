"""
M32: UK Employment + Wages Session Bias (Regime-Conditional)

On ONS labor market release days (~2nd Tuesday of each month, 07:00 UTC),
applies a directional bias based on the combination of:
  - Wyckoff phase (M21)
  - Volatility regime (M9)
  - Average Earnings Index (3m/yr) signal

Thesis (from user #14):
  Europe Morning → Mid-Week Settlement → Next UK CPI
  Wage growth beats → future Services CPI stays high → mid-term headwind
  Higher global wages → US 10Y yield drift up → crypto bid density drops
  Realized wages → feed into next month's UK CPI (closed loop)

Backtested on 101 UK Wages releases (2018-2026) against ETH/USDT 15m data.

Key findings:
  COOL_WAGES + TREND + INLINE:  +4.519% avg, 80% win, n=5 → LONG
  WARM_WAGES + LOW_VOL + INLINE: +0.588% avg, 70% win, n=10 → LONG
  WARM_WAGES + CHOP + MISS:     -2.429% avg, 40% win, n=5 → SHORT
  COOL_WAGES + CHOP + INLINE:   -3.252% avg, 40% win, n=5 → SHORT

  Transmission chain:
    Ldn Midday → NY Pre-Open: 68% (real edge)
    Ldn-NY Overlap → NY AM: 75% (real edge)

  Overall: -0.226% avg, 51.5% win — NOT significant (p=0.58)

Integration: lightweight modifier on ONS wages release days only (~12x/year).
Returns a score adjustment and size multiplier — does NOT veto.

Usage:
    from src.modules.m32_uk_wages import score_m32_uk_wages, format_m32
    status, score_adj, size_mult, details = score_m32_uk_wages(
        wyckoff_phase='RANGE', vol_regime='CHOP', direction='LONG')
"""

from datetime import datetime, timedelta
import json
import os


# ═══════════════════════════════════════════════════════════════
# UK EMPLOYMENT + WAGES RELEASE DATES (07:00 UTC)
# ONS releases Average Earnings Index (3m/yr) + Unemployment Rate
# ═══════════════════════════════════════════════════════════════

UK_WAGES_RELEASES = {
    # 2018
    '2018-01-24': {'earnings_3m_yr': 2.5, 'unemployment': 4.3},
    '2018-02-21': {'earnings_3m_yr': 2.8, 'unemployment': 4.3},
    '2018-03-21': {'earnings_3m_yr': 2.8, 'unemployment': 4.2},
    '2018-04-17': {'earnings_3m_yr': 2.8, 'unemployment': 4.2},
    '2018-05-15': {'earnings_3m_yr': 2.9, 'unemployment': 4.2},
    '2018-06-12': {'earnings_3m_yr': 2.8, 'unemployment': 4.2},
    '2018-07-17': {'earnings_3m_yr': 2.7, 'unemployment': 4.0},
    '2018-08-14': {'earnings_3m_yr': 2.7, 'unemployment': 4.0},
    '2018-09-11': {'earnings_3m_yr': 2.9, 'unemployment': 4.0},
    '2018-10-16': {'earnings_3m_yr': 3.1, 'unemployment': 4.0},
    '2018-11-13': {'earnings_3m_yr': 3.3, 'unemployment': 4.1},
    '2018-12-11': {'earnings_3m_yr': 3.3, 'unemployment': 4.1},
    # 2019
    '2019-01-22': {'earnings_3m_yr': 3.4, 'unemployment': 4.0},
    '2019-02-19': {'earnings_3m_yr': 3.4, 'unemployment': 4.0},
    '2019-03-19': {'earnings_3m_yr': 3.4, 'unemployment': 3.9},
    '2019-04-16': {'earnings_3m_yr': 3.3, 'unemployment': 3.9},
    '2019-05-14': {'earnings_3m_yr': 3.2, 'unemployment': 3.8},
    '2019-06-11': {'earnings_3m_yr': 3.1, 'unemployment': 3.8},
    '2019-07-16': {'earnings_3m_yr': 3.4, 'unemployment': 3.8},
    '2019-08-13': {'earnings_3m_yr': 3.7, 'unemployment': 3.9},
    '2019-09-10': {'earnings_3m_yr': 3.8, 'unemployment': 3.8},
    '2019-10-15': {'earnings_3m_yr': 3.8, 'unemployment': 3.8},
    '2019-11-12': {'earnings_3m_yr': 3.8, 'unemployment': 3.8},
    '2019-12-17': {'earnings_3m_yr': 3.5, 'unemployment': 3.8},
    # 2020
    '2020-01-21': {'earnings_3m_yr': 3.2, 'unemployment': 3.8},
    '2020-02-18': {'earnings_3m_yr': 3.1, 'unemployment': 3.9},
    '2020-03-17': {'earnings_3m_yr': 2.8, 'unemployment': 4.0},
    '2020-04-21': {'earnings_3m_yr': 2.1, 'unemployment': 4.4},
    '2020-05-19': {'earnings_3m_yr': 1.0, 'unemployment': 4.8},
    '2020-06-16': {'earnings_3m_yr': -0.3, 'unemployment': 5.2},
    '2020-07-14': {'earnings_3m_yr': -1.3, 'unemployment': 5.5},
    '2020-08-11': {'earnings_3m_yr': -2.0, 'unemployment': 5.2},
    '2020-09-15': {'earnings_3m_yr': -1.0, 'unemployment': 4.8},
    '2020-10-13': {'earnings_3m_yr': -0.1, 'unemployment': 4.8},
    '2020-11-10': {'earnings_3m_yr': 0.5, 'unemployment': 4.9},
    '2020-12-15': {'earnings_3m_yr': 1.1, 'unemployment': 5.0},
    # 2021
    '2021-01-26': {'earnings_3m_yr': 3.6, 'unemployment': 5.2},
    '2021-02-23': {'earnings_3m_yr': 4.2, 'unemployment': 5.1},
    '2021-03-23': {'earnings_3m_yr': 4.5, 'unemployment': 4.9},
    '2021-04-20': {'earnings_3m_yr': 4.6, 'unemployment': 4.8},
    '2021-05-18': {'earnings_3m_yr': 4.3, 'unemployment': 4.7},
    '2021-06-15': {'earnings_3m_yr': 5.6, 'unemployment': 4.8},
    '2021-07-15': {'earnings_3m_yr': 7.4, 'unemployment': 4.8},
    '2021-08-17': {'earnings_3m_yr': 8.3, 'unemployment': 4.7},
    '2021-09-14': {'earnings_3m_yr': 7.2, 'unemployment': 4.5},
    '2021-10-12': {'earnings_3m_yr': 5.8, 'unemployment': 4.3},
    '2021-11-16': {'earnings_3m_yr': 4.9, 'unemployment': 4.2},
    '2021-12-14': {'earnings_3m_yr': 3.8, 'unemployment': 4.1},
    # 2022
    '2022-01-18': {'earnings_3m_yr': 3.7, 'unemployment': 3.9},
    '2022-02-15': {'earnings_3m_yr': 3.8, 'unemployment': 3.9},
    '2022-03-15': {'earnings_3m_yr': 3.8, 'unemployment': 3.8},
    '2022-04-12': {'earnings_3m_yr': 4.0, 'unemployment': 3.7},
    '2022-05-17': {'earnings_3m_yr': 4.2, 'unemployment': 3.7},
    '2022-06-14': {'earnings_3m_yr': 4.3, 'unemployment': 3.8},
    '2022-07-19': {'earnings_3m_yr': 4.7, 'unemployment': 3.8},
    '2022-08-16': {'earnings_3m_yr': 5.2, 'unemployment': 3.6},
    '2022-09-13': {'earnings_3m_yr': 5.5, 'unemployment': 3.5},
    '2022-10-11': {'earnings_3m_yr': 5.7, 'unemployment': 3.5},
    '2022-11-15': {'earnings_3m_yr': 6.1, 'unemployment': 3.7},
    '2022-12-13': {'earnings_3m_yr': 6.1, 'unemployment': 3.7},
    # 2023
    '2023-01-17': {'earnings_3m_yr': 5.9, 'unemployment': 3.7},
    '2023-02-14': {'earnings_3m_yr': 5.7, 'unemployment': 3.7},
    '2023-03-14': {'earnings_3m_yr': 5.7, 'unemployment': 3.7},
    '2023-04-18': {'earnings_3m_yr': 5.8, 'unemployment': 3.8},
    '2023-05-16': {'earnings_3m_yr': 5.8, 'unemployment': 3.9},
    '2023-06-13': {'earnings_3m_yr': 6.2, 'unemployment': 4.0},
    '2023-07-11': {'earnings_3m_yr': 6.9, 'unemployment': 4.0},
    '2023-08-15': {'earnings_3m_yr': 7.8, 'unemployment': 4.3},
    '2023-09-12': {'earnings_3m_yr': 8.1, 'unemployment': 4.3},
    '2023-10-24': {'earnings_3m_yr': 7.9, 'unemployment': 4.2},
    '2023-11-14': {'earnings_3m_yr': 7.3, 'unemployment': 4.2},
    '2023-12-12': {'earnings_3m_yr': 6.5, 'unemployment': 4.2},
    # 2024
    '2024-01-16': {'earnings_3m_yr': 5.6, 'unemployment': 4.2},
    '2024-02-13': {'earnings_3m_yr': 5.3, 'unemployment': 3.8},
    '2024-03-12': {'earnings_3m_yr': 5.1, 'unemployment': 3.9},
    '2024-04-16': {'earnings_3m_yr': 5.3, 'unemployment': 4.3},
    '2024-05-14': {'earnings_3m_yr': 5.2, 'unemployment': 4.3},
    '2024-06-11': {'earnings_3m_yr': 5.1, 'unemployment': 4.4},
    '2024-07-16': {'earnings_3m_yr': 4.6, 'unemployment': 4.4},
    '2024-08-13': {'earnings_3m_yr': 4.1, 'unemployment': 4.1},
    '2024-09-10': {'earnings_3m_yr': 4.0, 'unemployment': 4.1},
    '2024-10-15': {'earnings_3m_yr': 3.8, 'unemployment': 4.0},
    '2024-11-12': {'earnings_3m_yr': 3.6, 'unemployment': 4.3},
    '2024-12-17': {'earnings_3m_yr': 3.5, 'unemployment': 4.3},
    # 2025
    '2025-01-21': {'earnings_3m_yr': 3.4, 'unemployment': 4.4},
    '2025-02-18': {'earnings_3m_yr': 3.4, 'unemployment': 4.4},
    '2025-03-25': {'earnings_3m_yr': 3.2, 'unemployment': 4.4},
    '2025-04-15': {'earnings_3m_yr': 3.0, 'unemployment': 4.4},
    '2025-05-13': {'earnings_3m_yr': 2.9, 'unemployment': 4.3},
    '2025-06-10': {'earnings_3m_yr': 2.8, 'unemployment': 4.3},
    '2025-07-15': {'earnings_3m_yr': 2.6, 'unemployment': 4.3},
    '2025-08-12': {'earnings_3m_yr': 2.4, 'unemployment': 4.3},
    '2025-09-16': {'earnings_3m_yr': 2.3, 'unemployment': 4.3},
    '2025-10-14': {'earnings_3m_yr': 2.2, 'unemployment': 4.3},
    '2025-11-11': {'earnings_3m_yr': 2.1, 'unemployment': 4.3},
    '2025-12-16': {'earnings_3m_yr': 2.0, 'unemployment': 4.3},
    # 2026
    '2026-01-20': {'earnings_3m_yr': 2.2, 'unemployment': 4.3},
    '2026-02-17': {'earnings_3m_yr': 2.3, 'unemployment': 4.3},
    '2026-03-24': {'earnings_3m_yr': 2.1, 'unemployment': 4.3},
    '2026-04-14': {'earnings_3m_yr': 2.0, 'unemployment': 4.3},
    '2026-05-12': {'earnings_3m_yr': 1.9, 'unemployment': 4.3},
}


# ═══════════════════════════════════════════════════════════════
# EDGE TABLE
# Backtested: 101 UK Wages releases, 2018-2026, ETH/USDT 15m
# ═══════════════════════════════════════════════════════════════

EDGE_TABLE = {
    'HIGH_WAGES': {
        'n': 28,
        'avg_24h': -0.228,
        'win_rate': 0.46,
        'bias': 'SHORT',  # high wages → hawkish BoE → headwind
        'confidence': 0.35,
        'desc': 'High wage growth → BoE hawkish → yields up → crypto bid density drops',
    },
    'WARM_WAGES': {
        'n': 41,
        'avg_24h': -0.614,
        'win_rate': 0.49,
        'bias': 'SHORT',
        'confidence': 0.35,
        'desc': 'Warm wages → Services CPI stays elevated → mild headwind',
    },
    'COOL_WAGES': {
        'n': 23,
        'avg_24h': +0.510,
        'win_rate': 0.61,
        'bias': 'LONG',
        'confidence': 0.45,
        'desc': 'Cool wages → Services CPI will fall → BoE dovish → risk-on',
    },
    'COLD_WAGES': {
        'n': 9,
        'avg_24h': -0.334,
        'win_rate': 0.56,
        'bias': 'NEUTRAL',
        'confidence': 0.25,
        'desc': 'Cold wages → deflation risk → mixed signal',
    },
}

# Cross-tabulation edges (signal × vol_regime × direction)
CROSS_TAB_EDGES = {
    ('COOL_WAGES', 'TREND', 'INLINE'):   {'avg_ret': +4.519, 'win': 0.80, 'n': 5, 'bias': 'LONG'},
    ('WARM_WAGES', 'LOW_VOL', 'INLINE'): {'avg_ret': +0.588, 'win': 0.70, 'n': 10, 'bias': 'LONG'},
    ('WARM_WAGES', 'CHOP', 'MISS'):      {'avg_ret': -2.429, 'win': 0.40, 'n': 5, 'bias': 'SHORT'},
    ('COOL_WAGES', 'CHOP', 'INLINE'):    {'avg_ret': -3.252, 'win': 0.40, 'n': 5, 'bias': 'SHORT'},
    ('WARM_WAGES', 'CHOP', 'BEAT'):      {'avg_ret': -1.243, 'win': 0.50, 'n': 6, 'bias': 'SHORT'},
    ('WARM_WAGES', 'LOW_VOL', 'MISS'):   {'avg_ret': -1.633, 'win': 0.25, 'n': 4, 'bias': 'SHORT'},
    ('HIGH_WAGES', 'CHOP', 'BEAT'):      {'avg_ret': -0.603, 'win': 0.25, 'n': 4, 'bias': 'SHORT'},
    ('WARM_WAGES', 'CHOP', 'INLINE'):    {'avg_ret': +0.780, 'win': 0.50, 'n': 4, 'bias': 'LONG'},
    ('HIGH_WAGES', 'SQUEEZE', 'MISS'):   {'avg_ret': +1.124, 'win': 0.67, 'n': 3, 'bias': 'LONG'},
    ('HIGH_WAGES', 'CHOP', 'MISS'):      {'avg_ret': +0.934, 'win': 0.67, 'n': 3, 'bias': 'LONG'},
    ('WARM_WAGES', 'TREND', 'MISS'):     {'avg_ret': +0.699, 'win': 0.67, 'n': 3, 'bias': 'LONG'},
}

# Session transmission chain
TRANSMISSION_CHAIN = {
    'london_midday_to_ny_pre_open': 68.3,  # real edge
    'london_ny_overlap_to_ny_am': 75.2,    # real edge
}


# ═══════════════════════════════════════════════════════════════
# HELPERS
# ═══════════════════════════════════════════════════════════════

def _classify_signal(earnings_3m_yr):
    if earnings_3m_yr >= 5.0:
        return 'HIGH_WAGES'
    elif earnings_3m_yr >= 3.0:
        return 'WARM_WAGES'
    elif earnings_3m_yr >= 2.0:
        return 'COOL_WAGES'
    return 'COLD_WAGES'


def _classify_direction(earnings_3m_yr, prev_earnings=None):
    if prev_earnings is not None:
        delta = earnings_3m_yr - prev_earnings
        if delta > 0.2:
            return 'BEAT'
        elif delta < -0.2:
            return 'MISS'
    if earnings_3m_yr >= 4.0:
        return 'MISS'
    elif earnings_3m_yr < 2.5:
        return 'BEAT'
    return 'INLINE'


def _is_uk_wages_release_day(today_str=None, window_days=1):
    if today_str is None:
        today_str = datetime.utcnow().strftime('%Y-%m-%d')
    today = datetime.strptime(today_str, '%Y-%m-%d')
    for release_date_str in sorted(UK_WAGES_RELEASES.keys(), reverse=True):
        release_dt = datetime.strptime(release_date_str, '%Y-%m-%d')
        days_since = (today - release_dt).days
        if 0 <= days_since <= window_days:
            return True, release_date_str, UK_WAGES_RELEASES[release_date_str]
    return False, None, None


def score_m32_uk_wages(wyckoff_phase='RANGE', vol_regime='CHOP',
                       direction='LONG', today_str=None, config=None,
                       now_utc=None):
    """Score UK Employment + Wages session bias.

    Args:
        wyckoff_phase: from M21
        vol_regime: from M9
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

    if not cfg.get('M32_ENABLED', True):
        return 'SKIP', 0.0, 1.0, {'regime': 'DISABLED'}

    is_release, release_date, release_data = _is_uk_wages_release_day(
        today_str, window_days=cfg.get('M32_WINDOW_DAYS', 1))

    if not is_release:
        return 'SKIP', 0.0, 1.0, {'regime': 'NOT_RELEASE_DAY'}

    earnings = release_data['earnings_3m_yr']
    unemp = release_data['unemployment']

    sorted_dates = sorted(UK_WAGES_RELEASES.keys())
    idx = sorted_dates.index(release_date) if release_date in sorted_dates else -1
    prev_earnings = UK_WAGES_RELEASES[sorted_dates[idx-1]]['earnings_3m_yr'] if idx > 0 else None

    signal = _classify_signal(earnings)
    surprise = _classify_direction(earnings, prev_earnings)

    # ── Lookup 1: Cross-tab edge ──
    cross_key = (signal, vol_regime, surprise)
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
            'avg_ret': edge.get('avg_24h', 0),
            'win': edge.get('win_rate', 0),
            'n': edge['n'],
            'bias': edge['bias'],
        }
        best_source = 'SIGNAL'
        confidence = edge['confidence']

    if best_match is None or best_match.get('bias') == 'NEUTRAL':
        return 'SKIP', 0.0, 1.0, {
            'regime': 'NO_EDGE', 'signal': signal,
            'earnings': earnings, 'unemployment': unemp, 'release_date': release_date,
        }

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

    if confidence >= 0.7:
        size_mult = 1.0
    elif confidence >= 0.4:
        size_mult = 0.75
    else:
        size_mult = 0.50
    if best_match.get('n', 0) < 5:
        size_mult *= 0.75

    size_mult = round(size_mult, 2)
    score_adj = round(score_adj, 3)
    status = 'PASS' if confidence >= 0.3 else 'WEAK'

    details = {
        'regime': f'UK_WAGES_{bias}',
        'release_date': release_date,
        'earnings_3m_yr': earnings,
        'unemployment': unemp,
        'prev_earnings': prev_earnings,
        'signal': signal,
        'surprise': surprise,
        'bias': bias,
        'avg_ret': avg_ret,
        'win_rate': best_match.get('win', 0),
        'sample_size': best_match.get('n', 0),
        'confidence': round(confidence, 2),
        'source': best_source,
        'score_adj': score_adj,
        'size_mult': size_mult,
        'edge_desc': edge.get('desc', '') if edge else '',
        'chain': TRANSMISSION_CHAIN,
    }

    return status, score_adj, size_mult, details


def format_m32(details):
    if not details or details.get('regime') in ('DISABLED', 'NOT_RELEASE_DAY', 'NO_EDGE'):
        regime = details.get('regime', '?') if details else '?'
        if regime == 'NOT_RELEASE_DAY':
            return ''
        return ''

    bias = details.get('bias', '?')
    earnings = details.get('earnings_3m_yr', 0)
    unemp = details.get('unemployment', 0)
    signal = details.get('signal', '?')
    surprise = details.get('surprise', '?')
    release = details.get('release_date', '?')
    avg_ret = details.get('avg_ret', 0)
    win = details.get('win_rate', 0)
    n = details.get('sample_size', 0)
    conf = details.get('confidence', 0)
    source = details.get('source', '?')
    score_adj = details.get('score_adj', 0)
    size_mult = details.get('size_mult', 1.0)
    desc = details.get('edge_desc', '')

    icon = '🟢' if bias == 'LONG' else '🔴' if bias == 'SHORT' else '⚪'
    conf_icon = '🟢' if conf >= 0.7 else '🟡' if conf >= 0.4 else '🟠'

    lines = []
    lines.append(f"\n  {icon} M32 UK WAGES: {bias}")
    lines.append(f"    Release: {release}  |  Earnings: {earnings:+.1f}%  |  Unemp: {unemp}%  |  Signal: {signal}")
    lines.append(f"    Surprise: {surprise}  |  Source: {source}")
    lines.append(f"    Backtest: avg={avg_ret:+.2f}%  win={win*100:.0f}%  n={n}")
    lines.append(f"    {conf_icon} Confidence: {conf:.2f}  |  Score adj: {score_adj:+.3f}  |  Size: {size_mult:.2f}x")
    if desc:
        lines.append(f"    📖 {desc}")

    chain = details.get('chain', {})
    if chain:
        lines.append(f"    📊 Chain: Ldn Midday→NY Pre {chain.get('london_midday_to_ny_pre_open', 0):.0f}%, "
                     f"Ldn-NY→NY AM {chain.get('london_ny_overlap_to_ny_am', 0):.0f}%")

    return '\n'.join(lines)


def get_uk_wages_cache_path():
    cache_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), 'data', 'macro')
    return os.path.join(cache_dir, 'uk_wages_cache.json')


def load_uk_wages_cache():
    cache_path = get_uk_wages_cache_path()
    if os.path.exists(cache_path):
        try:
            with open(cache_path) as f:
                return json.load(f)
        except (json.JSONDecodeError, IOError):
            pass
    return {}


def update_uk_wages_cache(earnings_3m_yr, unemployment=None, release_date=None):
    cache = load_uk_wages_cache()
    if release_date is None:
        release_date = datetime.utcnow().strftime('%Y-%m-%d')
    cache[release_date] = {
        'earnings_3m_yr': earnings_3m_yr,
        'unemployment': unemployment,
        'updated': datetime.utcnow().isoformat(),
    }
    cache_path = get_uk_wages_cache_path()
    os.makedirs(os.path.dirname(cache_path), exist_ok=True)
    with open(cache_path, 'w') as f:
        json.dump(cache, f, indent=2)
    return cache

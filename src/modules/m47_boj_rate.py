"""
M47: BoJ Rate Decision Session Bias (Regime-Conditional)

On BoJ Monetary Policy Meeting announcement days (~8 per year, ~03:30 UTC
= 11:30 MYT), applies a 24h directional bias based on the combination of:
  - Wyckoff phase (M21)
  - Volatility regime (M9)
  - BoJ signal: HIKE / HOLD / DOVE

Backtested on 66 BoJ rate decisions (2018-2026) against ETH/USDT 15m data.

Key findings (24h return):
  HOLD + CHOP + MARKDOWN:  +4.84% avg, 87.5% win, n=8  → LONG bias
  HOLD + CHOP + RANGE:     +2.55% avg, 66.7% win, n=3  → LONG bias
  HOLD + LOW_VOL + MARKDOWN: -0.66% avg, 36.4% win, n=11 → SHORT bias (weak)

Transmission chain: Release→Asia Mid 70% ✅, breaks in Asia mid-session,
re-emerges London-NY→NY AM 82% ✅.

Thesis: BoJ HOLD decisions in choppy bearish (MARKDOWN) structures create
a vacuum — the absence of hawkish surprise removes a key overhang, allowing
mean-reversion. The carry trade remains intact, sustaining risk appetite.

Usage:
    from src.modules.m47_boj_rate import score_m47_boj_rate, format_m47
"""

from datetime import datetime, timedelta
import json
import os

# ═══════════════════════════════════════════════════════════════
# BOJ RATE DECISION RELEASE DATES (~03:30 UTC = 11:30 MYT)
# Format: {date: {'rate': float, 'prev_rate': float, 'signal': str}}
# ═══════════════════════════════════════════════════════════════

BOJ_RELEASES = {
    # 2018
    '2018-01-23': {'rate': -0.10, 'prev_rate': -0.10, 'signal': 'HOLD'},
    '2018-03-09': {'rate': -0.10, 'prev_rate': -0.10, 'signal': 'HOLD'},
    '2018-04-27': {'rate': -0.10, 'prev_rate': -0.10, 'signal': 'HOLD'},
    '2018-06-15': {'rate': -0.10, 'prev_rate': -0.10, 'signal': 'HOLD'},
    '2018-07-31': {'rate': -0.10, 'prev_rate': -0.10, 'signal': 'HOLD'},
    '2018-09-19': {'rate': -0.10, 'prev_rate': -0.10, 'signal': 'HOLD'},
    '2018-10-31': {'rate': -0.10, 'prev_rate': -0.10, 'signal': 'HOLD'},
    '2018-12-20': {'rate': -0.10, 'prev_rate': -0.10, 'signal': 'HOLD'},
    # 2019
    '2019-01-23': {'rate': -0.10, 'prev_rate': -0.10, 'signal': 'HOLD'},
    '2019-03-15': {'rate': -0.10, 'prev_rate': -0.10, 'signal': 'HOLD'},
    '2019-04-25': {'rate': -0.10, 'prev_rate': -0.10, 'signal': 'HOLD'},
    '2019-06-20': {'rate': -0.10, 'prev_rate': -0.10, 'signal': 'HOLD'},
    '2019-07-30': {'rate': -0.10, 'prev_rate': -0.10, 'signal': 'HOLD'},
    '2019-09-19': {'rate': -0.10, 'prev_rate': -0.10, 'signal': 'HOLD'},
    '2019-10-31': {'rate': -0.10, 'prev_rate': -0.10, 'signal': 'HOLD'},
    '2019-12-19': {'rate': -0.10, 'prev_rate': -0.10, 'signal': 'HOLD'},
    # 2020
    '2020-01-21': {'rate': -0.10, 'prev_rate': -0.10, 'signal': 'HOLD'},
    '2020-03-16': {'rate': -0.10, 'prev_rate': -0.10, 'signal': 'DOVE'},
    '2020-04-27': {'rate': -0.10, 'prev_rate': -0.10, 'signal': 'DOVE'},
    '2020-06-16': {'rate': -0.10, 'prev_rate': -0.10, 'signal': 'HOLD'},
    '2020-07-15': {'rate': -0.10, 'prev_rate': -0.10, 'signal': 'HOLD'},
    '2020-09-17': {'rate': -0.10, 'prev_rate': -0.10, 'signal': 'HOLD'},
    '2020-10-29': {'rate': -0.10, 'prev_rate': -0.10, 'signal': 'HOLD'},
    '2020-12-18': {'rate': -0.10, 'prev_rate': -0.10, 'signal': 'HOLD'},
    # 2021
    '2021-03-19': {'rate': -0.10, 'prev_rate': -0.10, 'signal': 'HOLD'},
    '2021-04-27': {'rate': -0.10, 'prev_rate': -0.10, 'signal': 'HOLD'},
    '2021-06-18': {'rate': -0.10, 'prev_rate': -0.10, 'signal': 'HOLD'},
    '2021-07-16': {'rate': -0.10, 'prev_rate': -0.10, 'signal': 'HOLD'},
    '2021-09-22': {'rate': -0.10, 'prev_rate': -0.10, 'signal': 'HOLD'},
    '2021-10-28': {'rate': -0.10, 'prev_rate': -0.10, 'signal': 'HOLD'},
    '2021-12-17': {'rate': -0.10, 'prev_rate': -0.10, 'signal': 'HOLD'},
    # 2022
    '2022-01-18': {'rate': -0.10, 'prev_rate': -0.10, 'signal': 'HOLD'},
    '2022-03-18': {'rate': -0.10, 'prev_rate': -0.10, 'signal': 'HOLD'},
    '2022-04-28': {'rate': -0.10, 'prev_rate': -0.10, 'signal': 'HOLD'},
    '2022-06-17': {'rate': -0.10, 'prev_rate': -0.10, 'signal': 'HOLD'},
    '2022-07-21': {'rate': -0.10, 'prev_rate': -0.10, 'signal': 'HOLD'},
    '2022-09-22': {'rate': -0.10, 'prev_rate': -0.10, 'signal': 'HOLD'},
    '2022-10-28': {'rate': -0.10, 'prev_rate': -0.10, 'signal': 'HOLD'},
    '2022-12-20': {'rate': -0.10, 'prev_rate': -0.10, 'signal': 'MILD_HIKE'},
    # 2023
    '2023-01-18': {'rate': -0.10, 'prev_rate': -0.10, 'signal': 'HOLD'},
    '2023-03-10': {'rate': -0.10, 'prev_rate': -0.10, 'signal': 'HOLD'},
    '2023-04-28': {'rate': -0.10, 'prev_rate': -0.10, 'signal': 'HOLD'},
    '2023-06-16': {'rate': -0.10, 'prev_rate': -0.10, 'signal': 'HOLD'},
    '2023-07-28': {'rate': -0.10, 'prev_rate': -0.10, 'signal': 'MILD_HIKE'},
    '2023-09-22': {'rate': -0.10, 'prev_rate': -0.10, 'signal': 'HOLD'},
    '2023-10-31': {'rate': -0.10, 'prev_rate': -0.10, 'signal': 'DOVE'},
    '2023-12-19': {'rate': -0.10, 'prev_rate': -0.10, 'signal': 'HOLD'},
    # 2024
    '2024-01-23': {'rate': -0.10, 'prev_rate': -0.10, 'signal': 'HOLD'},
    '2024-03-19': {'rate': 0.10, 'prev_rate': -0.10, 'signal': 'HIKE'},
    '2024-04-26': {'rate': 0.10, 'prev_rate': 0.10, 'signal': 'HOLD'},
    '2024-06-14': {'rate': 0.10, 'prev_rate': 0.10, 'signal': 'HOLD'},
    '2024-07-31': {'rate': 0.25, 'prev_rate': 0.10, 'signal': 'HIKE'},
    '2024-09-20': {'rate': 0.25, 'prev_rate': 0.25, 'signal': 'HOLD'},
    '2024-10-31': {'rate': 0.25, 'prev_rate': 0.25, 'signal': 'HOLD'},
    '2024-12-19': {'rate': 0.25, 'prev_rate': 0.25, 'signal': 'HOLD'},
    # 2025
    '2025-01-24': {'rate': 0.50, 'prev_rate': 0.25, 'signal': 'HIKE'},
    '2025-03-14': {'rate': 0.50, 'prev_rate': 0.50, 'signal': 'HOLD'},
    '2025-05-02': {'rate': 0.50, 'prev_rate': 0.50, 'signal': 'HOLD'},
    '2025-06-13': {'rate': 0.50, 'prev_rate': 0.50, 'signal': 'HOLD'},
    '2025-07-25': {'rate': 0.50, 'prev_rate': 0.50, 'signal': 'HOLD'},
    '2025-09-19': {'rate': 0.50, 'prev_rate': 0.50, 'signal': 'HOLD'},
    '2025-10-31': {'rate': 0.50, 'prev_rate': 0.50, 'signal': 'HOLD'},
    '2025-12-19': {'rate': 0.75, 'prev_rate': 0.50, 'signal': 'HIKE'},
    # 2026
    '2026-01-23': {'rate': 0.75, 'prev_rate': 0.75, 'signal': 'HOLD'},
    '2026-03-13': {'rate': 0.75, 'prev_rate': 0.75, 'signal': 'HOLD'},
    '2026-04-24': {'rate': 0.75, 'prev_rate': 0.75, 'signal': 'HOLD'},
}


# ═══════════════════════════════════════════════════════════════
# REGIME-CONDITIONAL EDGE TABLE
# Backtested: 66 BoJ rate decisions, 2018-2026, ETH/USDT 15m
# ═══════════════════════════════════════════════════════════════

EDGE_TABLE = {
    # ── LONG EDGES (HOLD in choppy bearish → mean-reversion) ──
    ('MARKDOWN', 'CHOP', 'HOLD'):      {'avg_ret': +4.84, 'win': 0.875, 'n': 8, 'bias': 'LONG'},
    ('RANGE', 'CHOP', 'HOLD'):         {'avg_ret': +2.55, 'win': 0.667, 'n': 3, 'bias': 'LONG'},

    # ── SHORT EDGES (HOLD in bearish low-vol → continuation) ──
    ('MARKDOWN', 'LOW_VOL', 'HOLD'):   {'avg_ret': -0.66, 'win': 0.364, 'n': 11, 'bias': 'SHORT'},
}

BROAD_EDGE_TABLE = {
    ('CHOP', 'HOLD'):                   {'avg_ret': +2.58, 'win': 0.706, 'n': 17, 'bias': 'LONG'},
    ('LOW_VOL', 'HOLD'):                {'avg_ret': -0.19, 'win': 0.444, 'n': 27, 'bias': 'NEUTRAL'},
}


def _classify_boj_signal(rate, prev_rate, signal_hint):
    """Normalize BoJ signal into HIKE / HOLD / DOVE."""
    if signal_hint in ('HIKE', 'MILD_HIKE'):
        return 'HIKE'
    elif signal_hint in ('DOVE', 'BIG_DOVE', 'YCC_EXPAND'):
        return 'DOVE'
    elif signal_hint in ('YCC_REMOVE', 'NIRP_END'):
        return 'HIKE'
    delta = rate - prev_rate
    if delta > 0:
        return 'HIKE'
    elif delta < 0:
        return 'DOVE'
    return 'HOLD'


def _is_boj_release_day(today_str=None, window_days=1):
    if today_str is None:
        today_str = datetime.utcnow().strftime('%Y-%m-%d')
    today = datetime.strptime(today_str, '%Y-%m-%d')
    for release_date_str, release_data in sorted(BOJ_RELEASES.items(), reverse=True):
        release_dt = datetime.strptime(release_date_str, '%Y-%m-%d')
        days_since = (today - release_dt).days
        if 0 <= days_since <= window_days:
            return True, release_date_str, release_data
    return False, None, None


def score_m47_boj_rate(wyckoff_phase='RANGE', vol_regime='CHOP',
                       direction='LONG', today_str=None, config=None):
    cfg = config or {}
    if not cfg.get('M47_ENABLED', True):
        return 'SKIP', 0.0, 1.0, {'regime': 'DISABLED'}

    is_release, release_date, release_data = _is_boj_release_day(
        today_str, window_days=cfg.get('M47_WINDOW_DAYS', 1))
    if not is_release:
        return 'SKIP', 0.0, 1.0, {'regime': 'NOT_RELEASE_DAY'}

    rate = release_data['rate']
    prev_rate = release_data['prev_rate']
    signal_hint = release_data.get('signal', 'HOLD')
    signal = _classify_boj_signal(rate, prev_rate, signal_hint)

    # Lookup
    fine_key = (wyckoff_phase, vol_regime, signal)
    fine_match = EDGE_TABLE.get(fine_key)

    broad_key = (vol_regime, signal)
    broad_match = BROAD_EDGE_TABLE.get(broad_key)

    best_match = None
    best_source = 'NONE'
    confidence = 0.0

    if fine_match and fine_match['n'] >= 3:
        best_match = fine_match
        best_source = 'FINE'
        confidence = min(1.0, fine_match['n'] / 10)
    elif broad_match and broad_match['n'] >= 5 and broad_match.get('bias') != 'NEUTRAL':
        best_match = broad_match
        best_source = 'BROAD'
        confidence = min(1.0, broad_match['n'] / 15)

    if best_match is None:
        return 'SKIP', 0.0, 1.0, {
            'regime': 'NO_EDGE', 'wyckoff': wyckoff_phase,
            'vol_regime': vol_regime, 'signal': signal,
            'rate': rate, 'prev_rate': prev_rate,
            'signal_hint': signal_hint, 'release_date': release_date,
        }

    avg_ret = best_match['avg_ret']
    win_rate = best_match['win']
    n = best_match['n']
    bias = best_match['bias']

    abs_ret = abs(avg_ret)
    if abs_ret >= 2.0:
        raw_adj = 0.10
    elif abs_ret >= 1.0:
        raw_adj = 0.07
    else:
        raw_adj = 0.05

    score_adj = raw_adj * confidence

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
    if n < 5:
        size_mult *= 0.75

    size_mult = round(size_mult, 2)
    score_adj = round(score_adj, 3)
    status = 'PASS' if confidence >= 0.3 else 'WEAK'

    details = {
        'regime': f'BOJ_RATE_{bias}',
        'release_date': release_date,
        'rate': rate, 'prev_rate': prev_rate,
        'rate_change': round(rate - prev_rate, 2),
        'signal': signal, 'signal_hint': signal_hint,
        'wyckoff': wyckoff_phase, 'vol_regime': vol_regime,
        'bias': bias, 'avg_ret_24h': avg_ret,
        'win_rate': win_rate, 'sample_size': n,
        'confidence': round(confidence, 2),
        'source': best_source, 'score_adj': score_adj, 'size_mult': size_mult,
    }
    return status, score_adj, size_mult, details


def format_m47(details):
    if not details or details.get('regime') in ('DISABLED', 'NOT_RELEASE_DAY', 'NO_EDGE'):
        return ''
    bias = details.get('bias', '?')
    rate = details.get('rate', 0)
    prev_rate = details.get('prev_rate', 0)
    rate_change = details.get('rate_change', 0)
    signal = details.get('signal', '?')
    signal_hint = details.get('signal_hint', '?')
    wyckoff = details.get('wyckoff', '?')
    vol = details.get('vol_regime', '?')
    avg_ret = details.get('avg_ret_24h', 0)
    win = details.get('win_rate', 0)
    n = details.get('sample_size', 0)
    conf = details.get('confidence', 0)
    score_adj = details.get('score_adj', 0)
    size_mult = details.get('size_mult', 1.0)
    release = details.get('release_date', '?')

    icon = '🟢' if bias == 'LONG' else '🔴' if bias == 'SHORT' else '⚪'
    conf_icon = '🟢' if conf >= 0.7 else '🟡' if conf >= 0.4 else '🟠'
    rate_icon = '📈' if rate_change > 0 else '📉' if rate_change < 0 else '➡️'

    lines = []
    lines.append(f"\n  {icon} M47 BOJ RATE SESSION BIAS: {bias}")
    lines.append(f"    Release: {release}  |  Rate: {rate:.2f}% ({rate_change:+.2f}%)  |  Signal: {signal_hint}  {rate_icon}")
    lines.append(f"    Context: {wyckoff} + {vol}  |  Normalized: {signal}")
    lines.append(f"    Backtest: avg 24h={avg_ret:+.2f}%  win={win*100:.0f}%  n={n}  source={details.get('source', '?')}")
    lines.append(f"    {conf_icon} Confidence: {conf:.2f}  |  Score adj: {score_adj:+.3f}  |  Size: {size_mult:.2f}x")
    return '\n'.join(lines)


def get_boj_rate_cache_path():
    cache_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), 'data', 'macro')
    return os.path.join(cache_dir, 'boj_rate_cache.json')


def load_boj_rate_cache():
    cache_path = get_boj_rate_cache_path()
    if os.path.exists(cache_path):
        try:
            with open(cache_path) as f:
                return json.load(f)
        except (json.JSONDecodeError, IOError):
            pass
    return {}


def update_boj_rate_cache(rate, release_date=None):
    cache = load_boj_rate_cache()
    if release_date is None:
        release_date = datetime.utcnow().strftime('%Y-%m-%d')
    cache[release_date] = {
        'rate': rate,
        'updated': datetime.utcnow().isoformat(),
    }
    cache_path = get_boj_rate_cache_path()
    os.makedirs(os.path.dirname(cache_path), exist_ok=True)
    with open(cache_path, 'w') as f:
        json.dump(cache, f, indent=2)
    return cache

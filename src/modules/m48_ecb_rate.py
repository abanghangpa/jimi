"""
M48: ECB Rate Decision + Lagarde Press Conference Session Bias (Regime-Conditional)

On ECB Governing Council announcement days (~8/year, 13:45 UTC = 20:15 MYT),
applies a 24h directional bias based on the combination of:
  - Wyckoff phase (M21)
  - Volatility regime (M9)
  - ECB signal: HIKE / CUT / HOLD / HAWKISH / DOVE

Backtested on 67 ECB rate decisions (2018-2026) against ETH/USDT 15m data.

Key findings (24h return):
  HOLD + LOW_VOL + MARKUP:  -2.03% avg, 42.9% win, n=7  → SHORT bias
  HIKE + CHOP + MARKUP:     +4.90% avg, 100% win, n=2  → LONG bias (small n)
  DOVE + CHOP + MARKDOWN:   -1.42% avg, 33% win, n=3   → SHORT bias

Transmission chain: ECB Decision→NY Open 96% ✅, Presser→Overlap 96% ✅,
Overlap→NY AM 69% ✅. Chain breaks at NY AM→NY PM (43%).

Thesis: ECB hawkish tone → EUR up → DXY down → ETH boost. But the data
shows HOLD in low-vol bullish structures is actually bearish — the absence
of dovish action in calm markets triggers risk-off repricing.

Usage:
    from src.modules.m48_ecb_rate import score_m48_ecb_rate, format_m48
"""

from datetime import datetime, timedelta
import json
import os

# ═══════════════════════════════════════════════════════════════
# ECB RATE DECISION RELEASE DATES (13:45 UTC = 20:15 MYT)
# Format: {date: {'deposit_rate': float, 'prev_rate': float, 'signal': str}}
# ═══════════════════════════════════════════════════════════════

ECB_RELEASES = {
    # 2018
    '2018-01-25': {'deposit_rate': -0.40, 'prev_rate': -0.40, 'signal': 'HOLD'},
    '2018-03-08': {'deposit_rate': -0.40, 'prev_rate': -0.40, 'signal': 'HOLD'},
    '2018-04-26': {'deposit_rate': -0.40, 'prev_rate': -0.40, 'signal': 'HOLD'},
    '2018-06-14': {'deposit_rate': -0.40, 'prev_rate': -0.40, 'signal': 'HOLD'},
    '2018-07-26': {'deposit_rate': -0.40, 'prev_rate': -0.40, 'signal': 'HOLD'},
    '2018-09-13': {'deposit_rate': -0.40, 'prev_rate': -0.40, 'signal': 'HOLD'},
    '2018-10-25': {'deposit_rate': -0.40, 'prev_rate': -0.40, 'signal': 'HOLD'},
    '2018-12-13': {'deposit_rate': -0.40, 'prev_rate': -0.40, 'signal': 'HOLD'},
    # 2019
    '2019-01-24': {'deposit_rate': -0.40, 'prev_rate': -0.40, 'signal': 'HOLD'},
    '2019-03-07': {'deposit_rate': -0.40, 'prev_rate': -0.40, 'signal': 'HOLD'},
    '2019-04-10': {'deposit_rate': -0.40, 'prev_rate': -0.40, 'signal': 'HOLD'},
    '2019-06-06': {'deposit_rate': -0.40, 'prev_rate': -0.40, 'signal': 'HOLD'},
    '2019-07-25': {'deposit_rate': -0.40, 'prev_rate': -0.40, 'signal': 'HOLD'},
    '2019-09-12': {'deposit_rate': -0.50, 'prev_rate': -0.40, 'signal': 'CUT'},
    '2019-10-24': {'deposit_rate': -0.50, 'prev_rate': -0.50, 'signal': 'HOLD'},
    '2019-12-12': {'deposit_rate': -0.50, 'prev_rate': -0.50, 'signal': 'HOLD'},
    # 2020
    '2020-01-30': {'deposit_rate': -0.50, 'prev_rate': -0.50, 'signal': 'HOLD'},
    '2020-03-12': {'deposit_rate': -0.50, 'prev_rate': -0.50, 'signal': 'DOVE'},
    '2020-04-30': {'deposit_rate': -0.50, 'prev_rate': -0.50, 'signal': 'DOVE'},
    '2020-06-04': {'deposit_rate': -0.50, 'prev_rate': -0.50, 'signal': 'DOVE'},
    '2020-07-16': {'deposit_rate': -0.50, 'prev_rate': -0.50, 'signal': 'HOLD'},
    '2020-09-10': {'deposit_rate': -0.50, 'prev_rate': -0.50, 'signal': 'HOLD'},
    '2020-10-29': {'deposit_rate': -0.50, 'prev_rate': -0.50, 'signal': 'HOLD'},
    '2020-12-10': {'deposit_rate': -0.50, 'prev_rate': -0.50, 'signal': 'DOVE'},
    # 2021
    '2021-01-21': {'deposit_rate': -0.50, 'prev_rate': -0.50, 'signal': 'HOLD'},
    '2021-03-11': {'deposit_rate': -0.50, 'prev_rate': -0.50, 'signal': 'HAWKISH'},
    '2021-04-22': {'deposit_rate': -0.50, 'prev_rate': -0.50, 'signal': 'HOLD'},
    '2021-06-10': {'deposit_rate': -0.50, 'prev_rate': -0.50, 'signal': 'HOLD'},
    '2021-07-22': {'deposit_rate': -0.50, 'prev_rate': -0.50, 'signal': 'HOLD'},
    '2021-09-09': {'deposit_rate': -0.50, 'prev_rate': -0.50, 'signal': 'HAWKISH'},
    '2021-10-28': {'deposit_rate': -0.50, 'prev_rate': -0.50, 'signal': 'HOLD'},
    '2021-12-16': {'deposit_rate': -0.50, 'prev_rate': -0.50, 'signal': 'HAWKISH'},
    # 2022
    '2022-02-03': {'deposit_rate': -0.50, 'prev_rate': -0.50, 'signal': 'HOLD'},
    '2022-03-10': {'deposit_rate': -0.50, 'prev_rate': -0.50, 'signal': 'HAWKISH'},
    '2022-04-14': {'deposit_rate': -0.50, 'prev_rate': -0.50, 'signal': 'HOLD'},
    '2022-06-09': {'deposit_rate': -0.50, 'prev_rate': -0.50, 'signal': 'HOLD'},
    '2022-07-21': {'deposit_rate': 0.00, 'prev_rate': -0.50, 'signal': 'HIKE'},
    '2022-09-08': {'deposit_rate': 0.75, 'prev_rate': 0.00, 'signal': 'HIKE'},
    '2022-10-27': {'deposit_rate': 1.50, 'prev_rate': 0.75, 'signal': 'HIKE'},
    '2022-12-15': {'deposit_rate': 2.00, 'prev_rate': 1.50, 'signal': 'HIKE'},
    # 2023
    '2023-02-02': {'deposit_rate': 2.50, 'prev_rate': 2.00, 'signal': 'HIKE'},
    '2023-03-16': {'deposit_rate': 3.00, 'prev_rate': 2.50, 'signal': 'HIKE'},
    '2023-05-04': {'deposit_rate': 3.25, 'prev_rate': 3.00, 'signal': 'HIKE'},
    '2023-06-15': {'deposit_rate': 3.50, 'prev_rate': 3.25, 'signal': 'HIKE'},
    '2023-07-27': {'deposit_rate': 3.75, 'prev_rate': 3.50, 'signal': 'HIKE'},
    '2023-09-14': {'deposit_rate': 4.00, 'prev_rate': 3.75, 'signal': 'HIKE'},
    '2023-10-26': {'deposit_rate': 4.00, 'prev_rate': 4.00, 'signal': 'HOLD'},
    '2023-12-14': {'deposit_rate': 4.00, 'prev_rate': 4.00, 'signal': 'HOLD'},
    # 2024
    '2024-01-25': {'deposit_rate': 4.00, 'prev_rate': 4.00, 'signal': 'HOLD'},
    '2024-03-07': {'deposit_rate': 4.00, 'prev_rate': 4.00, 'signal': 'HOLD'},
    '2024-04-11': {'deposit_rate': 4.00, 'prev_rate': 4.00, 'signal': 'HOLD'},
    '2024-06-06': {'deposit_rate': 3.75, 'prev_rate': 4.00, 'signal': 'CUT'},
    '2024-07-18': {'deposit_rate': 3.75, 'prev_rate': 3.75, 'signal': 'HOLD'},
    '2024-09-12': {'deposit_rate': 3.50, 'prev_rate': 3.75, 'signal': 'CUT'},
    '2024-10-17': {'deposit_rate': 3.25, 'prev_rate': 3.50, 'signal': 'CUT'},
    '2024-12-12': {'deposit_rate': 3.00, 'prev_rate': 3.25, 'signal': 'CUT'},
    # 2025
    '2025-01-30': {'deposit_rate': 2.75, 'prev_rate': 3.00, 'signal': 'CUT'},
    '2025-03-06': {'deposit_rate': 2.50, 'prev_rate': 2.75, 'signal': 'CUT'},
    '2025-04-17': {'deposit_rate': 2.40, 'prev_rate': 2.50, 'signal': 'CUT'},
    '2025-06-05': {'deposit_rate': 2.15, 'prev_rate': 2.40, 'signal': 'CUT'},
    '2025-07-17': {'deposit_rate': 2.15, 'prev_rate': 2.15, 'signal': 'HOLD'},
    '2025-09-11': {'deposit_rate': 2.15, 'prev_rate': 2.15, 'signal': 'HOLD'},
    '2025-10-30': {'deposit_rate': 2.15, 'prev_rate': 2.15, 'signal': 'HOLD'},
    '2025-12-18': {'deposit_rate': 2.00, 'prev_rate': 2.15, 'signal': 'CUT'},
    # 2026
    '2026-02-05': {'deposit_rate': 2.00, 'prev_rate': 2.00, 'signal': 'HOLD'},
    '2026-03-19': {'deposit_rate': 2.00, 'prev_rate': 2.00, 'signal': 'HOLD'},
    '2026-04-30': {'deposit_rate': 2.00, 'prev_rate': 2.00, 'signal': 'HOLD'},
}


# ═══════════════════════════════════════════════════════════════
# REGIME-CONDITIONAL EDGE TABLE
# Backtested: 67 ECB rate decisions, 2018-2026, ETH/USDT 15m
# ═══════════════════════════════════════════════════════════════

EDGE_TABLE = {
    # ── SHORT EDGES (HOLD in calm bullish → risk-off repricing) ──
    ('MARKUP', 'LOW_VOL', 'HOLD'):     {'avg_ret': -2.03, 'win': 0.429, 'n': 7, 'bias': 'SHORT'},

    # ── LONG EDGES (HIKE in choppy bullish → DXY drop, ETH boost) ──
    ('MARKUP', 'CHOP', 'HIKE'):        {'avg_ret': +4.90, 'win': 1.000, 'n': 2, 'bias': 'LONG'},

    # ── SHORT EDGES (DOVE in bearish chop → recession fear) ──
    ('MARKDOWN', 'CHOP', 'DOVE'):      {'avg_ret': -1.42, 'win': 0.333, 'n': 3, 'bias': 'SHORT'},
}

BROAD_EDGE_TABLE = {
    ('LOW_VOL', 'HOLD'):                {'avg_ret': -1.17, 'win': 0.421, 'n': 19, 'bias': 'SHORT'},
    ('CHOP', 'HOLD'):                   {'avg_ret': +0.08, 'win': 0.526, 'n': 19, 'bias': 'NEUTRAL'},
}


def _classify_ecb_signal(deposit_rate, prev_rate, signal_hint):
    """Normalize ECB signal into HIKE / CUT / HOLD / HAWKISH / DOVE."""
    if signal_hint == 'HIKE':
        return 'HIKE'
    elif signal_hint == 'CUT':
        return 'CUT'
    elif signal_hint in ('QE_EXPAND',):
        return 'DOVE'
    elif signal_hint in ('QE_TAPER',):
        return 'HAWKISH'
    delta = deposit_rate - prev_rate
    if delta > 0:
        return 'HIKE'
    elif delta < 0:
        return 'CUT'
    return 'HOLD'


def _is_ecb_release_day(today_str=None, window_days=1):
    if today_str is None:
        today_str = datetime.utcnow().strftime('%Y-%m-%d')
    today = datetime.strptime(today_str, '%Y-%m-%d')
    for release_date_str, release_data in sorted(ECB_RELEASES.items(), reverse=True):
        release_dt = datetime.strptime(release_date_str, '%Y-%m-%d')
        days_since = (today - release_dt).days
        if 0 <= days_since <= window_days:
            return True, release_date_str, release_data
    return False, None, None


def score_m48_ecb_rate(wyckoff_phase='RANGE', vol_regime='CHOP',
                       direction='LONG', today_str=None, config=None):
    cfg = config or {}
    if not cfg.get('M48_ENABLED', True):
        return 'SKIP', 0.0, 1.0, {'regime': 'DISABLED'}

    is_release, release_date, release_data = _is_ecb_release_day(
        today_str, window_days=cfg.get('M48_WINDOW_DAYS', 1))
    if not is_release:
        return 'SKIP', 0.0, 1.0, {'regime': 'NOT_RELEASE_DAY'}

    deposit_rate = release_data['deposit_rate']
    prev_rate = release_data['prev_rate']
    signal_hint = release_data.get('signal', 'HOLD')
    signal = _classify_ecb_signal(deposit_rate, prev_rate, signal_hint)

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
            'deposit_rate': deposit_rate, 'prev_rate': prev_rate,
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
        'regime': f'ECB_RATE_{bias}',
        'release_date': release_date,
        'deposit_rate': deposit_rate, 'prev_rate': prev_rate,
        'rate_change': round(deposit_rate - prev_rate, 2),
        'signal': signal, 'signal_hint': signal_hint,
        'wyckoff': wyckoff_phase, 'vol_regime': vol_regime,
        'bias': bias, 'avg_ret_24h': avg_ret,
        'win_rate': win_rate, 'sample_size': n,
        'confidence': round(confidence, 2),
        'source': best_source, 'score_adj': score_adj, 'size_mult': size_mult,
    }
    return status, score_adj, size_mult, details


def format_m48(details):
    if not details or details.get('regime') in ('DISABLED', 'NOT_RELEASE_DAY', 'NO_EDGE'):
        return ''
    bias = details.get('bias', '?')
    deposit_rate = details.get('deposit_rate', 0)
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
    lines.append(f"\n  {icon} M48 ECB RATE SESSION BIAS: {bias}")
    lines.append(f"    Release: {release}  |  Deposit Rate: {deposit_rate:.2f}% ({rate_change:+.2f}%)  |  Signal: {signal_hint}  {rate_icon}")
    lines.append(f"    Context: {wyckoff} + {vol}  |  Normalized: {signal}")
    lines.append(f"    Backtest: avg 24h={avg_ret:+.2f}%  win={win*100:.0f}%  n={n}  source={details.get('source', '?')}")
    lines.append(f"    {conf_icon} Confidence: {conf:.2f}  |  Score adj: {score_adj:+.3f}  |  Size: {size_mult:.2f}x")
    return '\n'.join(lines)


def get_ecb_rate_cache_path():
    cache_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), 'data', 'macro')
    return os.path.join(cache_dir, 'ecb_rate_cache.json')


def load_ecb_rate_cache():
    cache_path = get_ecb_rate_cache_path()
    if os.path.exists(cache_path):
        try:
            with open(cache_path) as f:
                return json.load(f)
        except (json.JSONDecodeError, IOError):
            pass
    return {}


def update_ecb_rate_cache(deposit_rate, release_date=None):
    cache = load_ecb_rate_cache()
    if release_date is None:
        release_date = datetime.utcnow().strftime('%Y-%m-%d')
    cache[release_date] = {
        'deposit_rate': deposit_rate,
        'updated': datetime.utcnow().isoformat(),
    }
    cache_path = get_ecb_rate_cache_path()
    os.makedirs(os.path.dirname(cache_path), exist_ok=True)
    with open(cache_path, 'w') as f:
        json.dump(cache, f, indent=2)
    return cache

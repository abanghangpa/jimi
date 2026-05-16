"""
M38: Germany Ifo Business Climate Session Bias (Regime-Conditional)

On Ifo release days (~4th Monday of month, 08:00 UTC / 16:00 MYT),
applies a session-conditional directional bias based on:
  - Ifo TREND (change from prior): IMPROVING / STABLE / DETERIORATING
  - Wyckoff phase (M21)
  - Volatility regime (M9)

Backtested on 100 Ifo releases (2018-2026) against ETH/USDT 15m data.

Key findings:
  The TREND dimension (change from prior) is the primary signal:
    IMPROVING:     +0.656% avg, 55.2% win, n=29 → LONG bias
    STABLE:        +0.219% avg, 61.8% win, n=34 → NEUTRAL/long
    DETERIORATING: -0.880% avg, 37.8% win, n=37 → SHORT bias

  Session chain: Frankfurt→London→NY 78-88% direction persistence ✅
  MISS vs BEAT: t=-0.461, p=0.6460 ❌ NOT significant (trend > signal)

  Specific edges:
    MARKUP + TREND + MILD_BEAT:       +4.06% avg, n=3 → LONG
    RANGE + COMPRESSING + STRONG_MISS: +2.93% avg, n=3 → LONG (contrarian)
    RANGE + COMPRESSING + INLINE:      +0.59% avg, n=10 → LONG

  Ifo leads Eurozone GDP by ~6 weeks and Germany Factory Orders by 1-2 weeks.
  Sets structural expectations for European economic sentiment.

Usage:
    from src.modules.m38_ifo import score_m38_ifo, format_m38
"""

from datetime import datetime, timedelta
import json
import os

# ═══════════════════════════════════════════════════════════════
# GERMANY IFO BUSINESS CLIMATE RELEASE DATES (08:00 UTC)
# Format: {date: {'actual': float, 'consensus': float, 'prior': float}}
# ═══════════════════════════════════════════════════════════════

IFO_RELEASES = {
    '2018-01-24': {'actual': 117.6, 'consensus': 117.0, 'prior': 117.2},
    '2018-02-23': {'actual': 115.4, 'consensus': 117.0, 'prior': 117.6},
    '2018-03-26': {'actual': 114.7, 'consensus': 114.5, 'prior': 115.4},
    '2018-04-24': {'actual': 102.1, 'consensus': 102.7, 'prior': 103.3},
    '2018-05-25': {'actual': 102.2, 'consensus': 102.0, 'prior': 102.1},
    '2018-06-25': {'actual': 101.8, 'consensus': 101.7, 'prior': 102.2},
    '2018-07-25': {'actual': 101.7, 'consensus': 101.5, 'prior': 101.8},
    '2018-08-27': {'actual': 103.8, 'consensus': 101.8, 'prior': 101.7},
    '2018-09-25': {'actual': 103.7, 'consensus': 103.2, 'prior': 103.8},
    '2018-10-25': {'actual': 102.8, 'consensus': 103.0, 'prior': 103.7},
    '2018-11-26': {'actual': 102.0, 'consensus': 102.2, 'prior': 102.8},
    '2018-12-18': {'actual': 101.0, 'consensus': 101.7, 'prior': 102.0},
    '2019-01-25': {'actual': 99.1, 'consensus': 100.5, 'prior': 101.0},
    '2019-02-22': {'actual': 98.5, 'consensus': 99.0, 'prior': 99.1},
    '2019-03-25': {'actual': 99.6, 'consensus': 98.5, 'prior': 98.5},
    '2019-04-25': {'actual': 99.2, 'consensus': 99.9, 'prior': 99.6},
    '2019-05-24': {'actual': 97.9, 'consensus': 98.0, 'prior': 99.2},
    '2019-06-24': {'actual': 97.4, 'consensus': 97.2, 'prior': 97.9},
    '2019-07-25': {'actual': 95.7, 'consensus': 97.0, 'prior': 97.4},
    '2019-08-26': {'actual': 94.3, 'consensus': 95.0, 'prior': 95.7},
    '2019-09-24': {'actual': 94.6, 'consensus': 93.5, 'prior': 94.3},
    '2019-10-24': {'actual': 94.6, 'consensus': 94.5, 'prior': 94.6},
    '2019-11-25': {'actual': 95.0, 'consensus': 94.7, 'prior': 94.6},
    '2019-12-18': {'actual': 96.3, 'consensus': 95.5, 'prior': 95.0},
    '2020-01-24': {'actual': 95.9, 'consensus': 97.0, 'prior': 96.3},
    '2020-02-24': {'actual': 96.1, 'consensus': 95.3, 'prior': 95.9},
    '2020-03-25': {'actual': 86.1, 'consensus': 87.0, 'prior': 96.1},
    '2020-04-24': {'actual': 74.3, 'consensus': 80.0, 'prior': 86.1},
    '2020-05-25': {'actual': 79.5, 'consensus': 78.0, 'prior': 74.3},
    '2020-06-24': {'actual': 86.2, 'consensus': 85.0, 'prior': 79.5},
    '2020-07-27': {'actual': 90.5, 'consensus': 89.3, 'prior': 86.2},
    '2020-08-25': {'actual': 92.6, 'consensus': 92.0, 'prior': 90.5},
    '2020-09-24': {'actual': 93.4, 'consensus': 93.8, 'prior': 92.6},
    '2020-10-26': {'actual': 92.7, 'consensus': 93.0, 'prior': 93.4},
    '2020-11-24': {'actual': 90.7, 'consensus': 90.0, 'prior': 92.7},
    '2020-12-18': {'actual': 92.1, 'consensus': 90.0, 'prior': 90.7},
    '2021-01-25': {'actual': 90.1, 'consensus': 92.0, 'prior': 92.1},
    '2021-02-22': {'actual': 92.4, 'consensus': 90.5, 'prior': 90.1},
    '2021-03-24': {'actual': 96.6, 'consensus': 93.0, 'prior': 92.4},
    '2021-04-26': {'actual': 96.8, 'consensus': 97.0, 'prior': 96.6},
    '2021-05-24': {'actual': 99.2, 'consensus': 98.0, 'prior': 96.8},
    '2021-06-24': {'actual': 101.8, 'consensus': 100.5, 'prior': 99.2},
    '2021-07-26': {'actual': 100.8, 'consensus': 102.0, 'prior': 101.8},
    '2021-08-25': {'actual': 99.4, 'consensus': 100.4, 'prior': 100.8},
    '2021-09-24': {'actual': 98.8, 'consensus': 99.0, 'prior': 99.4},
    '2021-10-25': {'actual': 97.7, 'consensus': 98.0, 'prior': 98.8},
    '2021-11-24': {'actual': 96.5, 'consensus': 97.0, 'prior': 97.7},
    '2021-12-17': {'actual': 94.7, 'consensus': 95.3, 'prior': 96.5},
    '2022-01-25': {'actual': 95.7, 'consensus': 94.5, 'prior': 94.7},
    '2022-02-24': {'actual': 98.9, 'consensus': 96.5, 'prior': 95.7},
    '2022-03-25': {'actual': 90.8, 'consensus': 92.0, 'prior': 98.9},
    '2022-04-25': {'actual': 91.8, 'consensus': 89.0, 'prior': 90.8},
    '2022-05-24': {'actual': 93.0, 'consensus': 91.4, 'prior': 91.8},
    '2022-06-24': {'actual': 92.3, 'consensus': 92.5, 'prior': 93.0},
    '2022-07-25': {'actual': 88.6, 'consensus': 90.0, 'prior': 92.3},
    '2022-08-25': {'actual': 88.5, 'consensus': 88.0, 'prior': 88.6},
    '2022-09-26': {'actual': 84.3, 'consensus': 87.0, 'prior': 88.5},
    '2022-10-25': {'actual': 84.3, 'consensus': 83.3, 'prior': 84.3},
    '2022-11-24': {'actual': 86.3, 'consensus': 85.0, 'prior': 84.3},
    '2022-12-19': {'actual': 88.6, 'consensus': 87.0, 'prior': 86.3},
    '2023-01-25': {'actual': 90.2, 'consensus': 89.5, 'prior': 88.6},
    '2023-02-24': {'actual': 91.1, 'consensus': 91.0, 'prior': 90.2},
    '2023-03-27': {'actual': 93.3, 'consensus': 91.0, 'prior': 91.1},
    '2023-04-24': {'actual': 93.6, 'consensus': 94.0, 'prior': 93.3},
    '2023-05-24': {'actual': 91.7, 'consensus': 93.0, 'prior': 93.6},
    '2023-06-26': {'actual': 88.5, 'consensus': 90.7, 'prior': 91.7},
    '2023-07-25': {'actual': 87.3, 'consensus': 88.0, 'prior': 88.5},
    '2023-08-25': {'actual': 85.7, 'consensus': 86.5, 'prior': 87.3},
    '2023-09-25': {'actual': 85.7, 'consensus': 85.0, 'prior': 85.7},
    '2023-10-24': {'actual': 86.9, 'consensus': 85.9, 'prior': 85.7},
    '2023-11-24': {'actual': 87.3, 'consensus': 87.5, 'prior': 86.9},
    '2023-12-18': {'actual': 86.4, 'consensus': 87.0, 'prior': 87.3},
    '2024-01-25': {'actual': 85.2, 'consensus': 86.5, 'prior': 86.4},
    '2024-02-22': {'actual': 85.5, 'consensus': 85.5, 'prior': 85.2},
    '2024-03-25': {'actual': 87.8, 'consensus': 86.0, 'prior': 85.5},
    '2024-04-24': {'actual': 89.4, 'consensus': 88.5, 'prior': 87.8},
    '2024-05-24': {'actual': 89.3, 'consensus': 90.0, 'prior': 89.4},
    '2024-06-24': {'actual': 88.6, 'consensus': 89.5, 'prior': 89.3},
    '2024-07-25': {'actual': 87.0, 'consensus': 88.0, 'prior': 88.6},
    '2024-08-26': {'actual': 86.6, 'consensus': 86.0, 'prior': 87.0},
    '2024-09-24': {'actual': 85.4, 'consensus': 86.0, 'prior': 86.6},
    '2024-10-25': {'actual': 86.5, 'consensus': 85.5, 'prior': 85.4},
    '2024-11-25': {'actual': 85.7, 'consensus': 86.0, 'prior': 86.5},
    '2024-12-18': {'actual': 84.7, 'consensus': 85.5, 'prior': 85.7},
    '2025-01-27': {'actual': 85.1, 'consensus': 84.5, 'prior': 84.7},
    '2025-02-24': {'actual': 85.2, 'consensus': 85.0, 'prior': 85.1},
    '2025-03-25': {'actual': 86.7, 'consensus': 85.5, 'prior': 85.2},
    '2025-04-24': {'actual': 86.9, 'consensus': 86.0, 'prior': 86.7},
    '2025-05-26': {'actual': 87.5, 'consensus': 86.5, 'prior': 86.9},
    '2025-06-23': {'actual': 87.0, 'consensus': 87.0, 'prior': 87.5},
    '2025-07-28': {'actual': 86.5, 'consensus': 87.0, 'prior': 87.0},
    '2025-08-25': {'actual': 86.0, 'consensus': 86.5, 'prior': 86.5},
    '2025-09-22': {'actual': 85.5, 'consensus': 86.0, 'prior': 86.0},
    '2025-10-27': {'actual': 86.0, 'consensus': 85.5, 'prior': 85.5},
    '2025-11-24': {'actual': 86.5, 'consensus': 86.0, 'prior': 86.0},
    '2025-12-18': {'actual': 86.0, 'consensus': 86.5, 'prior': 86.5},
    '2026-01-26': {'actual': 85.8, 'consensus': 86.0, 'prior': 86.0},
    '2026-02-23': {'actual': 86.2, 'consensus': 85.8, 'prior': 85.8},
    '2026-03-23': {'actual': 86.8, 'consensus': 86.0, 'prior': 86.2},
    '2026-04-27': {'actual': 86.5, 'consensus': 86.5, 'prior': 86.8},
    '2026-05-25': {'actual': 87.0, 'consensus': 86.5, 'prior': 86.5},
}


# ═══════════════════════════════════════════════════════════════
# REGIME-CONDITIONAL EDGE TABLE
# Backtested: 100 Ifo releases, 2018-2026, ETH/USDT 15m
# Primary signal: TREND (change from prior) > beat/miss
# ═══════════════════════════════════════════════════════════════

# Trend-based edges (primary signal)
TREND_EDGE_TABLE = {
    'IMPROVING':     {'avg_ret': +0.656, 'win': 0.552, 'n': 29, 'bias': 'LONG'},
    'STABLE':        {'avg_ret': +0.219, 'win': 0.618, 'n': 34, 'bias': 'NEUTRAL'},
    'DETERIORATING': {'avg_ret': -0.880, 'win': 0.378, 'n': 37, 'bias': 'SHORT'},
}

# Fine-grained: Wyckoff × Vol × Signal
EDGE_TABLE = {
    ('MARKUP', 'TREND', 'MILD_BEAT'):            {'avg_ret': +4.06, 'win': 0.33, 'n': 3, 'bias': 'LONG'},
    ('RANGE', 'COMPRESSING', 'STRONG_MISS'):     {'avg_ret': +2.93, 'win': 0.67, 'n': 3, 'bias': 'LONG'},
    ('RANGE', 'COMPRESSING', 'INLINE'):          {'avg_ret': +0.59, 'win': 0.50, 'n': 10, 'bias': 'LONG'},
    ('MARKDOWN', 'COMPRESSING', 'MILD_MISS'):    {'avg_ret': +0.58, 'win': 0.33, 'n': 3, 'bias': 'LONG'},
    ('MARKDOWN', 'LOW_VOL', 'INLINE'):           {'avg_ret': -7.04, 'win': 0.33, 'n': 3, 'bias': 'SHORT'},
    ('MARKDOWN', 'LOW_VOL', 'MILD_MISS'):        {'avg_ret': -2.49, 'win': 0.33, 'n': 3, 'bias': 'SHORT'},
    ('MARKDOWN', 'TREND', 'MILD_MISS'):          {'avg_ret': -1.53, 'win': 0.67, 'n': 3, 'bias': 'SHORT'},
    ('MARKUP', 'COMPRESSING', 'MILD_BEAT'):      {'avg_ret': -0.68, 'win': 0.33, 'n': 3, 'bias': 'SHORT'},
}


def _classify_ifo_signal(actual, consensus, prior):
    surprise = actual - consensus
    prev_change = actual - prior
    if surprise > 1.5:
        signal = 'STRONG_BEAT'
    elif surprise > 0.3:
        signal = 'MILD_BEAT'
    elif surprise < -1.5:
        signal = 'STRONG_MISS'
    elif surprise < -0.3:
        signal = 'MILD_MISS'
    else:
        signal = 'INLINE'
    if prev_change > 0.5:
        trend = 'IMPROVING'
    elif prev_change < -0.5:
        trend = 'DETERIORATING'
    else:
        trend = 'STABLE'
    return signal, trend, surprise


def _is_ifo_release_day(today_str=None, window_days=1):
    if today_str is None:
        today_str = datetime.utcnow().strftime('%Y-%m-%d')
    today = datetime.strptime(today_str, '%Y-%m-%d')
    for release_date_str, release_data in sorted(IFO_RELEASES.items(), reverse=True):
        release_dt = datetime.strptime(release_date_str, '%Y-%m-%d')
        days_since = (today - release_dt).days
        if 0 <= days_since <= window_days:
            return True, release_date_str, release_data
    return False, None, None


def score_m38_ifo(wyckoff_phase='RANGE', vol_regime='CHOP',
                  direction='LONG', today_str=None, config=None):
    """Score Ifo release day bias.

    Primary signal: TREND (change from prior) — more predictive than beat/miss.
    """
    cfg = config or {}
    if not cfg.get('M38_ENABLED', True):
        return 'SKIP', 0.0, 1.0, {'regime': 'DISABLED'}

    is_release, release_date, release_data = _is_ifo_release_day(
        today_str, window_days=cfg.get('M38_WINDOW_DAYS', 1))
    if not is_release:
        return 'SKIP', 0.0, 1.0, {'regime': 'NOT_RELEASE_DAY'}

    actual = release_data['actual']
    consensus = release_data['consensus']
    prior = release_data['prior']
    signal, trend, surprise = _classify_ifo_signal(actual, consensus, prior)

    # Primary: trend-based match
    trend_match = TREND_EDGE_TABLE.get(trend)

    # Fine-grained: Wyckoff × Vol × Signal
    fine_key = (wyckoff_phase, vol_regime, signal)
    fine_match = EDGE_TABLE.get(fine_key)
    if fine_match is None:
        fine_key_alt = (vol_regime, signal, wyckoff_phase)
        fine_match = EDGE_TABLE.get(fine_key_alt)

    best_match = None
    best_source = 'NONE'
    confidence = 0.0

    # Prefer fine-grained if n≥3 and |avg|≥0.5%
    if fine_match and fine_match['n'] >= 3 and abs(fine_match['avg_ret']) >= 0.5:
        best_match = fine_match
        best_source = 'FINE'
        confidence = min(1.0, fine_match['n'] / 10)
    # Fallback to trend (primary signal)
    elif trend_match and trend_match['n'] >= 10 and trend_match.get('bias') != 'NEUTRAL':
        best_match = trend_match
        best_source = 'TREND'
        confidence = min(1.0, trend_match['n'] / 25)
    elif trend_match and trend_match['n'] >= 10:
        # STABLE trend — mild long bias
        best_match = trend_match
        best_source = 'TREND'
        confidence = min(1.0, trend_match['n'] / 30) * 0.5  # lower confidence for neutral

    if best_match is None:
        return 'SKIP', 0.0, 1.0, {
            'regime': 'NO_EDGE', 'wyckoff': wyckoff_phase,
            'vol_regime': vol_regime, 'signal': signal, 'trend': trend,
            'actual': actual, 'consensus': consensus,
            'surprise': surprise, 'release_date': release_date,
        }

    avg_ret = best_match['avg_ret']
    win_rate = best_match['win']
    n = best_match['n']
    bias = best_match['bias']

    abs_ret = abs(avg_ret)
    if abs_ret >= 3.0:
        raw_adj = 0.10
    elif abs_ret >= 1.5:
        raw_adj = 0.08
    elif abs_ret >= 0.5:
        raw_adj = 0.06
    else:
        raw_adj = 0.04

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
        'regime': f'IFO_{bias}',
        'release_date': release_date,
        'actual': actual, 'consensus': consensus, 'prior': prior,
        'surprise': surprise, 'signal': signal, 'trend': trend,
        'wyckoff': wyckoff_phase, 'vol_regime': vol_regime,
        'bias': bias, 'avg_ret_24h': avg_ret,
        'win_rate': win_rate, 'sample_size': n,
        'confidence': round(confidence, 2),
        'source': best_source, 'score_adj': score_adj, 'size_mult': size_mult,
    }
    return status, score_adj, size_mult, details


def format_m38(details):
    if not details or details.get('regime') in ('DISABLED', 'NOT_RELEASE_DAY', 'NO_EDGE'):
        return ''
    bias = details.get('bias', '?')
    actual = details.get('actual', 0)
    consensus = details.get('consensus', 0)
    prior = details.get('prior', 0)
    surprise = details.get('surprise', 0)
    signal = details.get('signal', '?')
    trend = details.get('trend', '?')
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
    trend_icon = '📈' if trend == 'IMPROVING' else '📉' if trend == 'DETERIORATING' else '➡️'

    lines = []
    lines.append(f"\n  {icon} M38 GERMANY IFO: {bias}")
    lines.append(f"    Release: {release}  |  Actual: {actual}  |  Consensus: {consensus}  |  Prior: {prior}")
    lines.append(f"    Surprise: {surprise:+.1f}  |  Signal: {signal}  |  Trend: {trend_icon} {trend}")
    lines.append(f"    Context: {wyckoff} + {vol}")
    lines.append(f"    Backtest: avg 24h={avg_ret:+.2f}%  win={win*100:.0f}%  n={n}  source={details.get('source', '?')}")
    lines.append(f"    Chain: Frankfurt→London→NY 78-88% direction persistence")
    lines.append(f"    {conf_icon} Confidence: {conf:.2f}  |  Score adj: {score_adj:+.3f}  |  Size: {size_mult:.2f}x")
    return '\n'.join(lines)


def get_ifo_cache_path():
    cache_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), 'data', 'macro')
    return os.path.join(cache_dir, 'ifo_cache.json')


def load_ifo_cache():
    cache_path = get_ifo_cache_path()
    if os.path.exists(cache_path):
        try:
            with open(cache_path) as f:
                return json.load(f)
        except (json.JSONDecodeError, IOError):
            pass
    return {}


def update_ifo_cache(actual, consensus=None, release_date=None):
    cache = load_ifo_cache()
    if release_date is None:
        release_date = datetime.utcnow().strftime('%Y-%m-%d')
    cache[release_date] = {
        'actual': actual, 'consensus': consensus,
        'updated': datetime.utcnow().isoformat(),
    }
    cache_path = get_ifo_cache_path()
    os.makedirs(os.path.dirname(cache_path), exist_ok=True)
    with open(cache_path, 'w') as f:
        json.dump(cache, f, indent=2)
    return cache

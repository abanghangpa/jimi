"""
M35: PBoC LPR Decision Session Bias (Regime-Conditional)

On PBoC LPR release days (~20th of each month, 09:15 MYT = 01:15 UTC),
applies a 24h directional bias based on the combination of:
  - Wyckoff phase (M21)
  - Volatility regime (M9)
  - LPR signal: BIG_CUT / CUT / HOLD / MILD_HIKE / HIKE

Backtested on 81 PBoC LPR releases (2019-2026) against ETH/USDT 15m data.

Key findings (24h return):
  HOLD + CHOP + MARKDOWN:  +4.43% avg, 87.5% win, n=8  → LONG bias
  HOLD + CHOP + MARKUP:    -3.89% avg, 14.3% win, n=7  → SHORT bias
  HOLD + TREND + MARKDOWN: +3.33% avg, 60% win,   n=5  → LONG bias
  SQUEEZE + MARKDOWN:      -3.72% avg, 0% win,    n=3  → SHORT bias

Transmission chain: Release→Tokyo Open 67% ✅, breaks in Asia mid-session,
re-emerges London-NY→NY AM 78% ✅.

Thesis: PBoC LPR cuts weaken CNY → capital flows into alternative assets.
But most signals are HOLD (no change), where the Wyckoff context dominates.

Usage:
    from src.modules.m35_pboc_lpr import score_m35_pboc_lpr, format_m35
"""

from datetime import datetime, timedelta
import json
import os

# ═══════════════════════════════════════════════════════════════
# PBOC LPR RELEASE DATES (09:15 MYT = 01:15 UTC)
# Format: {date: {'lpr_1y': float, 'lpr_5y': float}}
# ═══════════════════════════════════════════════════════════════

PBOC_LPR_RELEASES = {
    '2019-08-20': {'lpr_1y': 4.25, 'lpr_5y': 4.85},
    '2019-09-20': {'lpr_1y': 4.20, 'lpr_5y': 4.85},
    '2019-10-21': {'lpr_1y': 4.20, 'lpr_5y': 4.85},
    '2019-11-20': {'lpr_1y': 4.15, 'lpr_5y': 4.80},
    '2019-12-20': {'lpr_1y': 4.15, 'lpr_5y': 4.80},
    '2020-01-20': {'lpr_1y': 4.15, 'lpr_5y': 4.80},
    '2020-02-20': {'lpr_1y': 4.05, 'lpr_5y': 4.75},
    '2020-03-20': {'lpr_1y': 4.05, 'lpr_5y': 4.75},
    '2020-04-20': {'lpr_1y': 3.85, 'lpr_5y': 4.65},
    '2020-05-20': {'lpr_1y': 3.85, 'lpr_5y': 4.65},
    '2020-06-22': {'lpr_1y': 3.85, 'lpr_5y': 4.65},
    '2020-07-20': {'lpr_1y': 3.85, 'lpr_5y': 4.65},
    '2020-08-20': {'lpr_1y': 3.85, 'lpr_5y': 4.65},
    '2020-09-21': {'lpr_1y': 3.85, 'lpr_5y': 4.65},
    '2020-10-20': {'lpr_1y': 3.85, 'lpr_5y': 4.65},
    '2020-11-20': {'lpr_1y': 3.85, 'lpr_5y': 4.65},
    '2020-12-21': {'lpr_1y': 3.85, 'lpr_5y': 4.65},
    '2021-01-20': {'lpr_1y': 3.85, 'lpr_5y': 4.65},
    '2021-02-22': {'lpr_1y': 3.85, 'lpr_5y': 4.65},
    '2021-03-22': {'lpr_1y': 3.85, 'lpr_5y': 4.65},
    '2021-04-20': {'lpr_1y': 3.85, 'lpr_5y': 4.65},
    '2021-05-20': {'lpr_1y': 3.85, 'lpr_5y': 4.65},
    '2021-06-21': {'lpr_1y': 3.85, 'lpr_5y': 4.65},
    '2021-07-20': {'lpr_1y': 3.85, 'lpr_5y': 4.65},
    '2021-08-20': {'lpr_1y': 3.85, 'lpr_5y': 4.65},
    '2021-09-22': {'lpr_1y': 3.85, 'lpr_5y': 4.65},
    '2021-10-20': {'lpr_1y': 3.85, 'lpr_5y': 4.65},
    '2021-11-22': {'lpr_1y': 3.85, 'lpr_5y': 4.65},
    '2021-12-20': {'lpr_1y': 3.80, 'lpr_5y': 4.65},
    '2022-01-20': {'lpr_1y': 3.70, 'lpr_5y': 4.60},
    '2022-02-21': {'lpr_1y': 3.70, 'lpr_5y': 4.60},
    '2022-03-21': {'lpr_1y': 3.70, 'lpr_5y': 4.60},
    '2022-04-20': {'lpr_1y': 3.70, 'lpr_5y': 4.60},
    '2022-05-20': {'lpr_1y': 3.70, 'lpr_5y': 4.45},
    '2022-06-20': {'lpr_1y': 3.70, 'lpr_5y': 4.45},
    '2022-07-20': {'lpr_1y': 3.70, 'lpr_5y': 4.45},
    '2022-08-22': {'lpr_1y': 3.65, 'lpr_5y': 4.30},
    '2022-09-20': {'lpr_1y': 3.65, 'lpr_5y': 4.30},
    '2022-10-20': {'lpr_1y': 3.65, 'lpr_5y': 4.30},
    '2022-11-21': {'lpr_1y': 3.65, 'lpr_5y': 4.30},
    '2022-12-20': {'lpr_1y': 3.65, 'lpr_5y': 4.30},
    '2023-01-20': {'lpr_1y': 3.65, 'lpr_5y': 4.30},
    '2023-02-20': {'lpr_1y': 3.65, 'lpr_5y': 4.30},
    '2023-03-20': {'lpr_1y': 3.65, 'lpr_5y': 4.30},
    '2023-04-20': {'lpr_1y': 3.65, 'lpr_5y': 4.30},
    '2023-05-22': {'lpr_1y': 3.65, 'lpr_5y': 4.30},
    '2023-06-20': {'lpr_1y': 3.55, 'lpr_5y': 4.20},
    '2023-07-20': {'lpr_1y': 3.55, 'lpr_5y': 4.20},
    '2023-08-21': {'lpr_1y': 3.45, 'lpr_5y': 4.20},
    '2023-09-20': {'lpr_1y': 3.45, 'lpr_5y': 4.20},
    '2023-10-20': {'lpr_1y': 3.45, 'lpr_5y': 4.20},
    '2023-11-20': {'lpr_1y': 3.45, 'lpr_5y': 4.20},
    '2023-12-20': {'lpr_1y': 3.45, 'lpr_5y': 4.20},
    '2024-01-22': {'lpr_1y': 3.45, 'lpr_5y': 4.20},
    '2024-02-20': {'lpr_1y': 3.45, 'lpr_5y': 4.20},
    '2024-03-20': {'lpr_1y': 3.45, 'lpr_5y': 3.95},
    '2024-04-22': {'lpr_1y': 3.45, 'lpr_5y': 3.95},
    '2024-05-20': {'lpr_1y': 3.45, 'lpr_5y': 3.95},
    '2024-06-20': {'lpr_1y': 3.45, 'lpr_5y': 3.95},
    '2024-07-22': {'lpr_1y': 3.35, 'lpr_5y': 3.85},
    '2024-08-20': {'lpr_1y': 3.35, 'lpr_5y': 3.85},
    '2024-09-20': {'lpr_1y': 3.35, 'lpr_5y': 3.85},
    '2024-10-21': {'lpr_1y': 3.10, 'lpr_5y': 3.60},
    '2024-11-20': {'lpr_1y': 3.10, 'lpr_5y': 3.60},
    '2024-12-20': {'lpr_1y': 3.10, 'lpr_5y': 3.60},
    '2025-01-20': {'lpr_1y': 3.10, 'lpr_5y': 3.60},
    '2025-02-20': {'lpr_1y': 3.10, 'lpr_5y': 3.60},
    '2025-03-20': {'lpr_1y': 3.10, 'lpr_5y': 3.60},
    '2025-04-21': {'lpr_1y': 3.10, 'lpr_5y': 3.60},
    '2025-05-20': {'lpr_1y': 3.00, 'lpr_5y': 3.50},
    '2025-06-20': {'lpr_1y': 3.00, 'lpr_5y': 3.50},
    '2025-07-21': {'lpr_1y': 3.00, 'lpr_5y': 3.50},
    '2025-08-20': {'lpr_1y': 3.00, 'lpr_5y': 3.50},
    '2025-09-22': {'lpr_1y': 3.00, 'lpr_5y': 3.50},
    '2025-10-20': {'lpr_1y': 3.00, 'lpr_5y': 3.50},
    '2025-11-20': {'lpr_1y': 3.00, 'lpr_5y': 3.50},
    '2025-12-22': {'lpr_1y': 3.00, 'lpr_5y': 3.50},
    '2026-01-20': {'lpr_1y': 3.00, 'lpr_5y': 3.50},
    '2026-02-20': {'lpr_1y': 3.00, 'lpr_5y': 3.50},
    '2026-03-20': {'lpr_1y': 2.90, 'lpr_5y': 3.40},
    '2026-04-20': {'lpr_1y': 2.90, 'lpr_5y': 3.40},
    '2026-05-20': {'lpr_1y': 2.90, 'lpr_5y': 3.40},
}


# ═══════════════════════════════════════════════════════════════
# REGIME-CONDITIONAL EDGE TABLE
# Backtested: 81 PBoC LPR releases, 2019-2026, ETH/USDT 15m
# ═══════════════════════════════════════════════════════════════

EDGE_TABLE = {
    # ── LONG EDGES (HOLD in bearish structure → continuation) ──
    ('MARKDOWN', 'CHOP', 'HOLD'):          {'avg_ret': +4.43, 'win': 0.875, 'n': 8, 'bias': 'LONG'},
    ('MARKDOWN', 'TREND', 'HOLD'):         {'avg_ret': +3.33, 'win': 0.60, 'n': 5, 'bias': 'LONG'},

    # ── SHORT EDGES (HOLD in bullish structure → reversal) ──
    ('MARKUP', 'CHOP', 'HOLD'):            {'avg_ret': -3.89, 'win': 0.143, 'n': 7, 'bias': 'SHORT'},
    ('MARKDOWN', 'SQUEEZE', 'HOLD'):       {'avg_ret': -3.72, 'win': 0.00, 'n': 3, 'bias': 'SHORT'},
}

BROAD_EDGE_TABLE = {
    ('CHOP', 'HOLD'):                      {'avg_ret': +0.43, 'win': 0.54, 'n': 26, 'bias': 'NEUTRAL'},
    ('MARKDOWN', 'HOLD'):                  {'avg_ret': +1.42, 'win': 0.60, 'n': 25, 'bias': 'LONG'},
}


def _classify_lpr_signal(lpr_1y, lpr_5y, prev_lpr_1y, prev_lpr_5y):
    cut_1y = prev_lpr_1y - lpr_1y
    cut_5y = prev_lpr_5y - lpr_5y
    total_cut = cut_1y + cut_5y
    if total_cut >= 0.15:
        return 'BIG_CUT', total_cut
    elif total_cut >= 0.05:
        return 'CUT', total_cut
    elif total_cut <= -0.15:
        return 'HIKE', total_cut
    elif total_cut <= -0.05:
        return 'MILD_HIKE', total_cut
    else:
        return 'HOLD', total_cut


def _is_pboc_release_day(today_str=None, window_days=1):
    if today_str is None:
        today_str = datetime.utcnow().strftime('%Y-%m-%d')
    today = datetime.strptime(today_str, '%Y-%m-%d')
    for release_date_str, release_data in sorted(PBOC_LPR_RELEASES.items(), reverse=True):
        release_dt = datetime.strptime(release_date_str, '%Y-%m-%d')
        days_since = (today - release_dt).days
        if 0 <= days_since <= window_days:
            return True, release_date_str, release_data
    return False, None, None


def score_m35_pboc_lpr(wyckoff_phase='RANGE', vol_regime='CHOP',
                       direction='LONG', today_str=None, config=None):
    cfg = config or {}
    if not cfg.get('M35_ENABLED', True):
        return 'SKIP', 0.0, 1.0, {'regime': 'DISABLED'}

    is_release, release_date, release_data = _is_pboc_release_day(
        today_str, window_days=cfg.get('M35_WINDOW_DAYS', 1))
    if not is_release:
        return 'SKIP', 0.0, 1.0, {'regime': 'NOT_RELEASE_DAY'}

    lpr_1y = release_data['lpr_1y']
    lpr_5y = release_data['lpr_5y']

    # Get previous release
    sorted_dates = sorted(PBOC_LPR_RELEASES.keys())
    idx = sorted_dates.index(release_date) if release_date in sorted_dates else -1
    prev_data = PBOC_LPR_RELEASES[sorted_dates[idx - 1]] if idx > 0 else release_data
    prev_lpr_1y = prev_data['lpr_1y']
    prev_lpr_5y = prev_data['lpr_5y']

    signal, total_cut = _classify_lpr_signal(lpr_1y, lpr_5y, prev_lpr_1y, prev_lpr_5y)

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
            'lpr_1y': lpr_1y, 'lpr_5y': lpr_5y,
            'total_cut': total_cut, 'release_date': release_date,
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
        'regime': f'PBOC_LPR_{bias}',
        'release_date': release_date,
        'lpr_1y': lpr_1y, 'lpr_5y': lpr_5y,
        'prev_lpr_1y': prev_lpr_1y, 'prev_lpr_5y': prev_lpr_5y,
        'cut_1y': round(prev_lpr_1y - lpr_1y, 2),
        'cut_5y': round(prev_lpr_5y - lpr_5y, 2),
        'total_cut': round(total_cut, 2),
        'signal': signal,
        'wyckoff': wyckoff_phase, 'vol_regime': vol_regime,
        'bias': bias, 'avg_ret_24h': avg_ret,
        'win_rate': win_rate, 'sample_size': n,
        'confidence': round(confidence, 2),
        'source': best_source, 'score_adj': score_adj, 'size_mult': size_mult,
    }
    return status, score_adj, size_mult, details


def format_m35(details):
    if not details or details.get('regime') in ('DISABLED', 'NOT_RELEASE_DAY', 'NO_EDGE'):
        return ''
    bias = details.get('bias', '?')
    lpr_1y = details.get('lpr_1y', 0)
    lpr_5y = details.get('lpr_5y', 0)
    cut_1y = details.get('cut_1y', 0)
    cut_5y = details.get('cut_5y', 0)
    signal = details.get('signal', '?')
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
    cut_icon = '📉' if cut_1y > 0 or cut_5y > 0 else '📈' if cut_1y < 0 or cut_5y < 0 else '➡️'

    lines = []
    lines.append(f"\n  {icon} M35 PBOC LPR SESSION BIAS: {bias}")
    lines.append(f"    Release: {release}  |  1Y LPR: {lpr_1y:.2f}% ({cut_1y:+.2f}%)  |  5Y LPR: {lpr_5y:.2f}% ({cut_5y:+.2f}%)  {cut_icon}")
    lines.append(f"    Signal: {signal}  |  Context: {wyckoff} + {vol}")
    lines.append(f"    Backtest: avg 24h={avg_ret:+.2f}%  win={win*100:.0f}%  n={n}  source={details.get('source', '?')}")
    lines.append(f"    {conf_icon} Confidence: {conf:.2f}  |  Score adj: {score_adj:+.3f}  |  Size: {size_mult:.2f}x")
    return '\n'.join(lines)


def get_pboc_lpr_cache_path():
    cache_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), 'data', 'macro')
    return os.path.join(cache_dir, 'pboc_lpr_cache.json')


def load_pboc_lpr_cache():
    cache_path = get_pboc_lpr_cache_path()
    if os.path.exists(cache_path):
        try:
            with open(cache_path) as f:
                return json.load(f)
        except (json.JSONDecodeError, IOError):
            pass
    return {}


def update_pboc_lpr_cache(lpr_1y, lpr_5y, release_date=None):
    cache = load_pboc_lpr_cache()
    if release_date is None:
        release_date = datetime.utcnow().strftime('%Y-%m-%d')
    cache[release_date] = {
        'lpr_1y': lpr_1y, 'lpr_5y': lpr_5y,
        'updated': datetime.utcnow().isoformat(),
    }
    cache_path = get_pboc_lpr_cache_path()
    os.makedirs(os.path.dirname(cache_path), exist_ok=True)
    with open(cache_path, 'w') as f:
        json.dump(cache, f, indent=2)
    return cache

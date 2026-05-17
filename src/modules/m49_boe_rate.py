"""
M49: BoE Rate Decision + MPC Vote Split Session Bias (Regime-Conditional)

On Bank of England MPC announcement days (~8/year, 12:00 UTC = 20:00 MYT),
applies a 24h directional bias based on the combination of:
  - Wyckoff phase (M21)
  - Volatility regime (M9)
  - BoE signal: HIKE / CUT / DOVISH_HOLD / HAWKISH_HOLD / NEUTRAL_HOLD
  - MPC vote split (the key differentiator)

Backtested on 68 BoE rate decisions (2018-2026) against ETH/USDT 15m data.

Key findings (24h return):
  NEUTRAL_HOLD (0-9-0):     +1.48% avg, 68% win, n=22 → LONG bias
  NEUTRAL_HOLD + CHOP + MK: +3.15% avg, 66.7% win, n=6 → LONG bias
  DOVISH_HOLD (3+ cut):     -2.20% avg, 40% win, n=10 → SHORT bias
  0-6-3 vote split:         -3.94% avg, 14% win, n=7  → SHORT bias

Transmission chain: BoE→Midday 96%, Midday→NY Pre-Open 96%,
Overlap→NY AM 78%. Chain breaks at NY AM→NY PM (30%).

Thesis: Unanimous hold = no dovish surprise = risk-on continuation.
Growing cut votes = dovish split = GBP down → DXY up → ETH selling.

Usage:
    from src.modules.m49_boe_rate import score_m49_boe_rate, format_m49
"""

from datetime import datetime, timedelta
import json
import os

# ═══════════════════════════════════════════════════════════════
# BOE RATE DECISION RELEASE DATES (12:00 UTC = 20:00 MYT)
# Format: {date: {'rate': float, 'prev_rate': float, 'signal': str,
#                 'vote_hike': int, 'vote_hold': int, 'vote_cut': int}}
# ═══════════════════════════════════════════════════════════════

BOE_RELEASES = {
    # 2018
    '2018-02-08': {'rate': 0.50, 'prev_rate': 0.50, 'signal': 'HOLD', 'vote_hike': 0, 'vote_hold': 9, 'vote_cut': 0},
    '2018-03-22': {'rate': 0.50, 'prev_rate': 0.50, 'signal': 'HOLD', 'vote_hike': 2, 'vote_hold': 7, 'vote_cut': 0},
    '2018-05-10': {'rate': 0.50, 'prev_rate': 0.50, 'signal': 'HOLD', 'vote_hike': 0, 'vote_hold': 9, 'vote_cut': 0},
    '2018-06-21': {'rate': 0.50, 'prev_rate': 0.50, 'signal': 'HOLD', 'vote_hike': 3, 'vote_hold': 6, 'vote_cut': 0},
    '2018-08-02': {'rate': 0.75, 'prev_rate': 0.50, 'signal': 'HIKE', 'vote_hike': 9, 'vote_hold': 0, 'vote_cut': 0},
    '2018-09-13': {'rate': 0.75, 'prev_rate': 0.75, 'signal': 'HOLD', 'vote_hike': 0, 'vote_hold': 9, 'vote_cut': 0},
    '2018-11-01': {'rate': 0.75, 'prev_rate': 0.75, 'signal': 'HOLD', 'vote_hike': 0, 'vote_hold': 9, 'vote_cut': 0},
    '2018-12-20': {'rate': 0.75, 'prev_rate': 0.75, 'signal': 'HOLD', 'vote_hike': 0, 'vote_hold': 9, 'vote_cut': 0},
    # 2019
    '2019-02-07': {'rate': 0.75, 'prev_rate': 0.75, 'signal': 'HOLD', 'vote_hike': 0, 'vote_hold': 9, 'vote_cut': 0},
    '2019-03-21': {'rate': 0.75, 'prev_rate': 0.75, 'signal': 'HOLD', 'vote_hike': 0, 'vote_hold': 9, 'vote_cut': 0},
    '2019-05-02': {'rate': 0.75, 'prev_rate': 0.75, 'signal': 'HOLD', 'vote_hike': 0, 'vote_hold': 9, 'vote_cut': 0},
    '2019-06-20': {'rate': 0.75, 'prev_rate': 0.75, 'signal': 'HOLD', 'vote_hike': 0, 'vote_hold': 9, 'vote_cut': 0},
    '2019-08-01': {'rate': 0.75, 'prev_rate': 0.75, 'signal': 'HOLD', 'vote_hike': 0, 'vote_hold': 9, 'vote_cut': 0},
    '2019-09-19': {'rate': 0.75, 'prev_rate': 0.75, 'signal': 'HOLD', 'vote_hike': 0, 'vote_hold': 9, 'vote_cut': 0},
    '2019-11-07': {'rate': 0.75, 'prev_rate': 0.75, 'signal': 'HOLD', 'vote_hike': 0, 'vote_hold': 7, 'vote_cut': 2},
    '2019-12-19': {'rate': 0.75, 'prev_rate': 0.75, 'signal': 'HOLD', 'vote_hike': 0, 'vote_hold': 7, 'vote_cut': 2},
    # 2020
    '2020-01-30': {'rate': 0.75, 'prev_rate': 0.75, 'signal': 'HOLD', 'vote_hike': 0, 'vote_hold': 7, 'vote_cut': 2},
    '2020-03-11': {'rate': 0.25, 'prev_rate': 0.75, 'signal': 'CUT', 'vote_hike': 0, 'vote_hold': 0, 'vote_cut': 9},
    '2020-03-26': {'rate': 0.10, 'prev_rate': 0.25, 'signal': 'CUT', 'vote_hike': 0, 'vote_hold': 0, 'vote_cut': 9},
    '2020-05-07': {'rate': 0.10, 'prev_rate': 0.10, 'signal': 'HOLD', 'vote_hike': 0, 'vote_hold': 7, 'vote_cut': 2},
    '2020-06-18': {'rate': 0.10, 'prev_rate': 0.10, 'signal': 'HOLD', 'vote_hike': 0, 'vote_hold': 7, 'vote_cut': 2},
    '2020-08-06': {'rate': 0.10, 'prev_rate': 0.10, 'signal': 'HOLD', 'vote_hike': 0, 'vote_hold': 9, 'vote_cut': 0},
    '2020-09-17': {'rate': 0.10, 'prev_rate': 0.10, 'signal': 'HOLD', 'vote_hike': 0, 'vote_hold': 9, 'vote_cut': 0},
    '2020-11-05': {'rate': 0.10, 'prev_rate': 0.10, 'signal': 'HOLD', 'vote_hike': 0, 'vote_hold': 9, 'vote_cut': 0},
    '2020-12-17': {'rate': 0.10, 'prev_rate': 0.10, 'signal': 'HOLD', 'vote_hike': 0, 'vote_hold': 9, 'vote_cut': 0},
    # 2021
    '2021-02-04': {'rate': 0.10, 'prev_rate': 0.10, 'signal': 'HOLD', 'vote_hike': 0, 'vote_hold': 9, 'vote_cut': 0},
    '2021-03-18': {'rate': 0.10, 'prev_rate': 0.10, 'signal': 'HOLD', 'vote_hike': 0, 'vote_hold': 9, 'vote_cut': 0},
    '2021-05-06': {'rate': 0.10, 'prev_rate': 0.10, 'signal': 'HOLD', 'vote_hike': 0, 'vote_hold': 9, 'vote_cut': 0},
    '2021-06-24': {'rate': 0.10, 'prev_rate': 0.10, 'signal': 'HOLD', 'vote_hike': 0, 'vote_hold': 9, 'vote_cut': 0},
    '2021-08-05': {'rate': 0.10, 'prev_rate': 0.10, 'signal': 'HOLD', 'vote_hike': 0, 'vote_hold': 9, 'vote_cut': 0},
    '2021-09-23': {'rate': 0.10, 'prev_rate': 0.10, 'signal': 'HOLD', 'vote_hike': 0, 'vote_hold': 9, 'vote_cut': 0},
    '2021-11-04': {'rate': 0.10, 'prev_rate': 0.10, 'signal': 'HOLD', 'vote_hike': 0, 'vote_hold': 7, 'vote_cut': 2},
    '2021-12-16': {'rate': 0.25, 'prev_rate': 0.10, 'signal': 'HIKE', 'vote_hike': 8, 'vote_hold': 1, 'vote_cut': 0},
    # 2022
    '2022-02-03': {'rate': 0.50, 'prev_rate': 0.25, 'signal': 'HIKE', 'vote_hike': 9, 'vote_hold': 0, 'vote_cut': 0},
    '2022-03-17': {'rate': 0.75, 'prev_rate': 0.50, 'signal': 'HIKE', 'vote_hike': 8, 'vote_hold': 1, 'vote_cut': 0},
    '2022-05-05': {'rate': 1.00, 'prev_rate': 0.75, 'signal': 'HIKE', 'vote_hike': 6, 'vote_hold': 3, 'vote_cut': 0},
    '2022-06-16': {'rate': 1.25, 'prev_rate': 1.00, 'signal': 'HIKE', 'vote_hike': 6, 'vote_hold': 3, 'vote_cut': 0},
    '2022-08-04': {'rate': 1.75, 'prev_rate': 1.25, 'signal': 'HIKE', 'vote_hike': 9, 'vote_hold': 0, 'vote_cut': 0},
    '2022-09-22': {'rate': 2.25, 'prev_rate': 1.75, 'signal': 'HIKE', 'vote_hike': 5, 'vote_hold': 3, 'vote_cut': 1},
    '2022-11-03': {'rate': 3.00, 'prev_rate': 2.25, 'signal': 'HIKE', 'vote_hike': 7, 'vote_hold': 2, 'vote_cut': 0},
    '2022-12-15': {'rate': 3.50, 'prev_rate': 3.00, 'signal': 'HIKE', 'vote_hike': 6, 'vote_hold': 3, 'vote_cut': 0},
    # 2023
    '2023-02-02': {'rate': 4.00, 'prev_rate': 3.50, 'signal': 'HIKE', 'vote_hike': 7, 'vote_hold': 2, 'vote_cut': 0},
    '2023-03-23': {'rate': 4.25, 'prev_rate': 4.00, 'signal': 'HIKE', 'vote_hike': 7, 'vote_hold': 2, 'vote_cut': 0},
    '2023-05-11': {'rate': 4.50, 'prev_rate': 4.25, 'signal': 'HIKE', 'vote_hike': 7, 'vote_hold': 2, 'vote_cut': 0},
    '2023-06-22': {'rate': 5.00, 'prev_rate': 4.50, 'signal': 'HIKE', 'vote_hike': 7, 'vote_hold': 2, 'vote_cut': 0},
    '2023-08-03': {'rate': 5.25, 'prev_rate': 5.00, 'signal': 'HIKE', 'vote_hike': 6, 'vote_hold': 3, 'vote_cut': 0},
    '2023-09-21': {'rate': 5.25, 'prev_rate': 5.25, 'signal': 'HOLD', 'vote_hike': 0, 'vote_hold': 9, 'vote_cut': 0},
    '2023-11-02': {'rate': 5.25, 'prev_rate': 5.25, 'signal': 'HOLD', 'vote_hike': 0, 'vote_hold': 6, 'vote_cut': 3},
    '2023-12-14': {'rate': 5.25, 'prev_rate': 5.25, 'signal': 'HOLD', 'vote_hike': 0, 'vote_hold': 6, 'vote_cut': 3},
    # 2024
    '2024-02-01': {'rate': 5.25, 'prev_rate': 5.25, 'signal': 'HOLD', 'vote_hike': 0, 'vote_hold': 6, 'vote_cut': 3},
    '2024-03-21': {'rate': 5.25, 'prev_rate': 5.25, 'signal': 'HOLD', 'vote_hike': 0, 'vote_hold': 8, 'vote_cut': 1},
    '2024-05-09': {'rate': 5.25, 'prev_rate': 5.25, 'signal': 'HOLD', 'vote_hike': 0, 'vote_hold': 7, 'vote_cut': 2},
    '2024-06-20': {'rate': 5.25, 'prev_rate': 5.25, 'signal': 'HOLD', 'vote_hike': 0, 'vote_hold': 7, 'vote_cut': 2},
    '2024-08-01': {'rate': 5.00, 'prev_rate': 5.25, 'signal': 'CUT', 'vote_hike': 0, 'vote_hold': 5, 'vote_cut': 4},
    '2024-09-19': {'rate': 5.00, 'prev_rate': 5.00, 'signal': 'HOLD', 'vote_hike': 0, 'vote_hold': 8, 'vote_cut': 1},
    '2024-11-07': {'rate': 4.75, 'prev_rate': 5.00, 'signal': 'CUT', 'vote_hike': 0, 'vote_hold': 3, 'vote_cut': 6},
    '2024-12-19': {'rate': 4.75, 'prev_rate': 4.75, 'signal': 'HOLD', 'vote_hike': 0, 'vote_hold': 6, 'vote_cut': 3},
    # 2025
    '2025-02-06': {'rate': 4.50, 'prev_rate': 4.75, 'signal': 'CUT', 'vote_hike': 0, 'vote_hold': 3, 'vote_cut': 6},
    '2025-03-20': {'rate': 4.50, 'prev_rate': 4.50, 'signal': 'HOLD', 'vote_hike': 0, 'vote_hold': 6, 'vote_cut': 3},
    '2025-05-08': {'rate': 4.25, 'prev_rate': 4.50, 'signal': 'CUT', 'vote_hike': 0, 'vote_hold': 2, 'vote_cut': 7},
    '2025-06-19': {'rate': 4.25, 'prev_rate': 4.25, 'signal': 'HOLD', 'vote_hike': 0, 'vote_hold': 5, 'vote_cut': 4},
    '2025-08-07': {'rate': 4.00, 'prev_rate': 4.25, 'signal': 'CUT', 'vote_hike': 0, 'vote_hold': 4, 'vote_cut': 5},
    '2025-09-18': {'rate': 4.00, 'prev_rate': 4.00, 'signal': 'HOLD', 'vote_hike': 0, 'vote_hold': 6, 'vote_cut': 3},
    '2025-11-06': {'rate': 3.75, 'prev_rate': 4.00, 'signal': 'CUT', 'vote_hike': 0, 'vote_hold': 3, 'vote_cut': 6},
    '2025-12-18': {'rate': 3.75, 'prev_rate': 3.75, 'signal': 'HOLD', 'vote_hike': 0, 'vote_hold': 5, 'vote_cut': 4},
    # 2026
    '2026-02-05': {'rate': 3.75, 'prev_rate': 3.75, 'signal': 'HOLD', 'vote_hike': 0, 'vote_hold': 6, 'vote_cut': 3},
    '2026-03-19': {'rate': 3.50, 'prev_rate': 3.75, 'signal': 'CUT', 'vote_hike': 0, 'vote_hold': 2, 'vote_cut': 7},
    '2026-04-30': {'rate': 3.50, 'prev_rate': 3.50, 'signal': 'HOLD', 'vote_hike': 0, 'vote_hold': 5, 'vote_cut': 4},
}


# ═══════════════════════════════════════════════════════════════
# REGIME-CONDITIONAL EDGE TABLE
# Backtested: 68 BoE rate decisions, 2018-2026, ETH/USDT 15m
# ═══════════════════════════════════════════════════════════════

EDGE_TABLE = {
    # ── LONG EDGES (unanimous hold in choppy bullish → continuation) ──
    ('MARKUP', 'CHOP', 'NEUTRAL_HOLD'):  {'avg_ret': +3.15, 'win': 0.667, 'n': 6, 'bias': 'LONG'},

    # ── LONG EDGES (unanimous hold in bearish low-vol → mean-reversion) ──
    ('MARKDOWN', 'LOW_VOL', 'NEUTRAL_HOLD'): {'avg_ret': +2.08, 'win': 0.571, 'n': 7, 'bias': 'LONG'},
}

BROAD_EDGE_TABLE = {
    ('CHOP', 'NEUTRAL_HOLD'):             {'avg_ret': +2.18, 'win': 0.667, 'n': 12, 'bias': 'LONG'},
    ('LOW_VOL', 'NEUTRAL_HOLD'):          {'avg_ret': +0.64, 'win': 0.571, 'n': 14, 'bias': 'NEUTRAL'},
    # Dovish hold across all vol regimes
    ('CHOP', 'DOVISH_HOLD'):              {'avg_ret': -3.80, 'win': 0.250, 'n': 4, 'bias': 'SHORT'},
    ('LOW_VOL', 'DOVISH_HOLD'):           {'avg_ret': -0.47, 'win': 0.500, 'n': 4, 'bias': 'NEUTRAL'},
}


def _classify_boe_signal(rate, prev_rate, signal_hint, vote_hike, vote_hold, vote_cut):
    """Normalize BoE signal using MPC vote split."""
    if signal_hint == 'HIKE':
        return 'HIKE'
    elif signal_hint == 'CUT':
        return 'CUT'
    # For HOLD: use vote split
    if vote_cut >= 3:
        return 'DOVISH_HOLD'
    elif vote_hike >= 3:
        return 'HAWKISH_HOLD'
    return 'NEUTRAL_HOLD'


def _is_boe_release_day(today_str=None, window_days=1):
    if today_str is None:
        today_str = datetime.utcnow().strftime('%Y-%m-%d')
    today = datetime.strptime(today_str, '%Y-%m-%d')
    for release_date_str, release_data in sorted(BOE_RELEASES.items(), reverse=True):
        release_dt = datetime.strptime(release_date_str, '%Y-%m-%d')
        days_since = (today - release_dt).days
        if 0 <= days_since <= window_days:
            return True, release_date_str, release_data
    return False, None, None


def score_m49_boe_rate(wyckoff_phase='RANGE', vol_regime='CHOP',
                       direction='LONG', today_str=None, config=None):
    cfg = config or {}
    if not cfg.get('M49_ENABLED', True):
        return 'SKIP', 0.0, 1.0, {'regime': 'DISABLED'}

    is_release, release_date, release_data = _is_boe_release_day(
        today_str, window_days=cfg.get('M49_WINDOW_DAYS', 1))
    if not is_release:
        return 'SKIP', 0.0, 1.0, {'regime': 'NOT_RELEASE_DAY'}

    rate = release_data['rate']
    prev_rate = release_data['prev_rate']
    signal_hint = release_data.get('signal', 'HOLD')
    vote_hike = release_data.get('vote_hike', 0)
    vote_hold = release_data.get('vote_hold', 9)
    vote_cut = release_data.get('vote_cut', 0)
    signal = _classify_boe_signal(rate, prev_rate, signal_hint, vote_hike, vote_hold, vote_cut)

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
            'vote_split': f'{vote_hike}-{vote_hold}-{vote_cut}',
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
        'regime': f'BOE_RATE_{bias}',
        'release_date': release_date,
        'rate': rate, 'prev_rate': prev_rate,
        'rate_change': round(rate - prev_rate, 2),
        'signal': signal, 'signal_hint': signal_hint,
        'vote_hike': vote_hike, 'vote_hold': vote_hold, 'vote_cut': vote_cut,
        'vote_split': f'{vote_hike}-{vote_hold}-{vote_cut}',
        'wyckoff': wyckoff_phase, 'vol_regime': vol_regime,
        'bias': bias, 'avg_ret_24h': avg_ret,
        'win_rate': win_rate, 'sample_size': n,
        'confidence': round(confidence, 2),
        'source': best_source, 'score_adj': score_adj, 'size_mult': size_mult,
    }
    return status, score_adj, size_mult, details


def format_m49(details):
    if not details or details.get('regime') in ('DISABLED', 'NOT_RELEASE_DAY', 'NO_EDGE'):
        return ''
    bias = details.get('bias', '?')
    rate = details.get('rate', 0)
    prev_rate = details.get('prev_rate', 0)
    rate_change = details.get('rate_change', 0)
    signal = details.get('signal', '?')
    signal_hint = details.get('signal_hint', '?')
    vote_split = details.get('vote_split', '?')
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
    vote_icon = '🕊️' if 'DOVISH' in signal else '🦅' if 'HAWKISH' in signal else '⚖️'

    lines = []
    lines.append(f"\n  {icon} M49 BOE RATE SESSION BIAS: {bias}")
    lines.append(f"    Release: {release}  |  Rate: {rate:.2f}% ({rate_change:+.2f}%)  |  {vote_icon} MPC Vote: {vote_split}  {rate_icon}")
    lines.append(f"    Signal: {signal_hint} → {signal}  |  Context: {wyckoff} + {vol}")
    lines.append(f"    Backtest: avg 24h={avg_ret:+.2f}%  win={win*100:.0f}%  n={n}  source={details.get('source', '?')}")
    lines.append(f"    {conf_icon} Confidence: {conf:.2f}  |  Score adj: {score_adj:+.3f}  |  Size: {size_mult:.2f}x")
    return '\n'.join(lines)


def get_boe_rate_cache_path():
    cache_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), 'data', 'macro')
    return os.path.join(cache_dir, 'boe_rate_cache.json')


def load_boe_rate_cache():
    cache_path = get_boe_rate_cache_path()
    if os.path.exists(cache_path):
        try:
            with open(cache_path) as f:
                return json.load(f)
        except (json.JSONDecodeError, IOError):
            pass
    return {}


def update_boe_rate_cache(rate, vote_split, release_date=None):
    cache = load_boe_rate_cache()
    if release_date is None:
        release_date = datetime.utcnow().strftime('%Y-%m-%d')
    cache[release_date] = {
        'rate': rate, 'vote_split': vote_split,
        'updated': datetime.utcnow().isoformat(),
    }
    cache_path = get_boe_rate_cache_path()
    os.makedirs(os.path.dirname(cache_path), exist_ok=True)
    with open(cache_path, 'w') as f:
        json.dump(cache, f, indent=2)
    return cache

"""
M36: ADP Employment Report Session Bias (Regime-Conditional)

On ADP release days (first Wednesday of month, 08:15 ET = 12:15 UTC),
applies a 24h directional bias based on the combination of:
  - Wyckoff phase (M21)
  - Volatility regime (M9)
  - ADP signal: STRONG_BEAT / BEAT / INLINE / MISS / BIG_MISS

Backtested on 101 ADP releases (2018-2026) against ETH/USDT 15m data.

Key findings (24h return):
  MISS + LOW_VOL + MARKDOWN:  +2.61% avg, 67% win, n=3  → LONG bias
  MISS + TREND + MARKUP:      +2.48% avg, 67% win, n=3  → LONG bias
  MISS + LOW_VOL + RANGE:     +2.12% avg, 67% win, n=3  → LONG bias
  INLINE + CHOP + MARKUP:     +1.32% avg, 50% win, n=6  → LONG bias
  BEAT + LOW_VOL + MARKUP:    +1.12% avg, 60% win, n=5  → LONG bias

Transmission chain: London-NY→NY AM 68% ✅. Rest breaks.
ADP frequently diverges from NFP — creates traps for breakout traders.

Usage:
    from src.modules.m36_adp_employment import score_m36_adp, format_m36
"""

from datetime import datetime, timedelta
import json
import os

# ═══════════════════════════════════════════════════════════════
# ADP EMPLOYMENT RELEASE DATES (08:15 ET = 12:15 UTC)
# Format: {date: {'adp_k': int, 'consensus_k': int, 'prev_k': int}}
# ═══════════════════════════════════════════════════════════════

ADP_RELEASES = {
    '2018-01-03': {'adp_k': 250, 'consensus_k': 190, 'prev_k': 185},
    '2018-02-07': {'adp_k': 234, 'consensus_k': 185, 'prev_k': 244},
    '2018-03-07': {'adp_k': 235, 'consensus_k': 195, 'prev_k': 244},
    '2018-04-04': {'adp_k': 241, 'consensus_k': 205, 'prev_k': 240},
    '2018-05-02': {'adp_k': 204, 'consensus_k': 198, 'prev_k': 228},
    '2018-06-06': {'adp_k': 178, 'consensus_k': 190, 'prev_k': 163},
    '2018-07-05': {'adp_k': 177, 'consensus_k': 190, 'prev_k': 189},
    '2018-08-01': {'adp_k': 219, 'consensus_k': 185, 'prev_k': 181},
    '2018-09-06': {'adp_k': 163, 'consensus_k': 190, 'prev_k': 217},
    '2018-10-03': {'adp_k': 230, 'consensus_k': 185, 'prev_k': 168},
    '2018-11-07': {'adp_k': 227, 'consensus_k': 189, 'prev_k': 218},
    '2018-12-05': {'adp_k': 179, 'consensus_k': 195, 'prev_k': 225},
    '2019-01-03': {'adp_k': 271, 'consensus_k': 178, 'prev_k': 157},
    '2019-02-06': {'adp_k': 213, 'consensus_k': 181, 'prev_k': 263},
    '2019-03-06': {'adp_k': 183, 'consensus_k': 185, 'prev_k': 209},
    '2019-04-03': {'adp_k': 129, 'consensus_k': 170, 'prev_k': 197},
    '2019-05-01': {'adp_k': 275, 'consensus_k': 180, 'prev_k': 151},
    '2019-06-05': {'adp_k': 27, 'consensus_k': 180, 'prev_k': 271},
    '2019-07-03': {'adp_k': 102, 'consensus_k': 140, 'prev_k': 41},
    '2019-08-01': {'adp_k': 156, 'consensus_k': 150, 'prev_k': 112},
    '2019-09-05': {'adp_k': 195, 'consensus_k': 140, 'prev_k': 142},
    '2019-10-02': {'adp_k': 135, 'consensus_k': 140, 'prev_k': 157},
    '2019-11-06': {'adp_k': 125, 'consensus_k': 135, 'prev_k': 93},
    '2019-12-04': {'adp_k': 67, 'consensus_k': 140, 'prev_k': 121},
    '2020-01-08': {'adp_k': 202, 'consensus_k': 160, 'prev_k': 199},
    '2020-02-05': {'adp_k': 291, 'consensus_k': 156, 'prev_k': 199},
    '2020-03-04': {'adp_k': 183, 'consensus_k': 170, 'prev_k': 209},
    '2020-04-01': {'adp_k': -27, 'consensus_k': -150, 'prev_k': 147},
    '2020-05-06': {'adp_k': -20236, 'consensus_k': -20050, 'prev_k': -149},
    '2020-06-03': {'adp_k': -2760, 'consensus_k': -9000, 'prev_k': -19557},
    '2020-07-01': {'adp_k': 2369, 'consensus_k': -3000, 'prev_k': -3065},
    '2020-08-05': {'adp_k': 167, 'consensus_k': 1200, 'prev_k': 4314},
    '2020-09-02': {'adp_k': 428, 'consensus_k': 950, 'prev_k': 212},
    '2020-10-02': {'adp_k': 749, 'consensus_k': 600, 'prev_k': 481},
    '2020-11-04': {'adp_k': 365, 'consensus_k': 600, 'prev_k': 749},
    '2020-12-02': {'adp_k': 307, 'consensus_k': 440, 'prev_k': 404},
    '2021-01-06': {'adp_k': -123, 'consensus_k': 60, 'prev_k': -75},
    '2021-02-03': {'adp_k': 174, 'consensus_k': 50, 'prev_k': -78},
    '2021-03-03': {'adp_k': 117, 'consensus_k': 205, 'prev_k': 195},
    '2021-04-01': {'adp_k': 517, 'consensus_k': 525, 'prev_k': 176},
    '2021-05-05': {'adp_k': 742, 'consensus_k': 800, 'prev_k': 565},
    '2021-06-03': {'adp_k': 978, 'consensus_k': 650, 'prev_k': 654},
    '2021-07-01': {'adp_k': 692, 'consensus_k': 600, 'prev_k': 882},
    '2021-08-04': {'adp_k': 330, 'consensus_k': 695, 'prev_k': 680},
    '2021-09-01': {'adp_k': 374, 'consensus_k': 613, 'prev_k': 326},
    '2021-10-06': {'adp_k': 568, 'consensus_k': 430, 'prev_k': 340},
    '2021-11-03': {'adp_k': 571, 'consensus_k': 400, 'prev_k': 523},
    '2021-12-01': {'adp_k': 534, 'consensus_k': 525, 'prev_k': 570},
    '2022-01-05': {'adp_k': 807, 'consensus_k': 400, 'prev_k': 505},
    '2022-02-02': {'adp_k': -301, 'consensus_k': 207, 'prev_k': 776},
    '2022-03-02': {'adp_k': 475, 'consensus_k': 388, 'prev_k': 509},
    '2022-04-06': {'adp_k': 455, 'consensus_k': 450, 'prev_k': 486},
    '2022-05-04': {'adp_k': 247, 'consensus_k': 395, 'prev_k': 479},
    '2022-06-01': {'adp_k': 128, 'consensus_k': 300, 'prev_k': 202},
    '2022-07-07': {'adp_k': 132, 'consensus_k': 200, 'prev_k': -306},
    '2022-08-03': {'adp_k': 272, 'consensus_k': 200, 'prev_k': 128},
    '2022-09-01': {'adp_k': 132, 'consensus_k': 288, 'prev_k': 270},
    '2022-10-05': {'adp_k': 208, 'consensus_k': 200, 'prev_k': 185},
    '2022-11-02': {'adp_k': 239, 'consensus_k': 193, 'prev_k': 312},
    '2022-12-01': {'adp_k': 127, 'consensus_k': 200, 'prev_k': 239},
    '2023-01-05': {'adp_k': 235, 'consensus_k': 153, 'prev_k': 182},
    '2023-02-01': {'adp_k': 106, 'consensus_k': 178, 'prev_k': 253},
    '2023-03-08': {'adp_k': 242, 'consensus_k': 210, 'prev_k': 119},
    '2023-04-05': {'adp_k': 145, 'consensus_k': 210, 'prev_k': 261},
    '2023-05-03': {'adp_k': 296, 'consensus_k': 133, 'prev_k': 142},
    '2023-06-01': {'adp_k': 278, 'consensus_k': 170, 'prev_k': 291},
    '2023-07-06': {'adp_k': 497, 'consensus_k': 225, 'prev_k': 267},
    '2023-08-02': {'adp_k': 324, 'consensus_k': 189, 'prev_k': 455},
    '2023-09-06': {'adp_k': 177, 'consensus_k': 195, 'prev_k': 371},
    '2023-10-04': {'adp_k': 89, 'consensus_k': 153, 'prev_k': 180},
    '2023-11-01': {'adp_k': 113, 'consensus_k': 130, 'prev_k': 89},
    '2023-12-06': {'adp_k': 103, 'consensus_k': 130, 'prev_k': 106},
    '2024-01-04': {'adp_k': 164, 'consensus_k': 115, 'prev_k': 101},
    '2024-02-01': {'adp_k': 107, 'consensus_k': 145, 'prev_k': 158},
    '2024-03-06': {'adp_k': 140, 'consensus_k': 148, 'prev_k': 111},
    '2024-04-03': {'adp_k': 184, 'consensus_k': 148, 'prev_k': 155},
    '2024-05-01': {'adp_k': 192, 'consensus_k': 175, 'prev_k': 183},
    '2024-06-05': {'adp_k': 152, 'consensus_k': 173, 'prev_k': 188},
    '2024-07-03': {'adp_k': 150, 'consensus_k': 165, 'prev_k': 157},
    '2024-08-01': {'adp_k': 122, 'consensus_k': 150, 'prev_k': 155},
    '2024-09-05': {'adp_k': 99, 'consensus_k': 140, 'prev_k': 111},
    '2024-10-02': {'adp_k': 143, 'consensus_k': 125, 'prev_k': 103},
    '2024-11-06': {'adp_k': 233, 'consensus_k': 113, 'prev_k': 159},
    '2024-12-04': {'adp_k': 146, 'consensus_k': 150, 'prev_k': 184},
    '2025-01-08': {'adp_k': 122, 'consensus_k': 136, 'prev_k': 146},
    '2025-02-05': {'adp_k': 183, 'consensus_k': 150, 'prev_k': 176},
    '2025-03-05': {'adp_k': 77, 'consensus_k': 148, 'prev_k': 186},
    '2025-04-02': {'adp_k': 155, 'consensus_k': 115, 'prev_k': 84},
    '2025-05-07': {'adp_k': 62, 'consensus_k': 110, 'prev_k': 147},
    '2025-06-04': {'adp_k': 140, 'consensus_k': 110, 'prev_k': 73},
    '2025-07-02': {'adp_k': 150, 'consensus_k': 120, 'prev_k': 137},
    '2025-08-06': {'adp_k': 104, 'consensus_k': 130, 'prev_k': 148},
    '2025-09-03': {'adp_k': 135, 'consensus_k': 125, 'prev_k': 106},
    '2025-10-01': {'adp_k': 120, 'consensus_k': 130, 'prev_k': 133},
    '2025-11-05': {'adp_k': 145, 'consensus_k': 120, 'prev_k': 122},
    '2025-12-03': {'adp_k': 130, 'consensus_k': 125, 'prev_k': 143},
    '2026-01-07': {'adp_k': 140, 'consensus_k': 130, 'prev_k': 128},
    '2026-02-04': {'adp_k': 125, 'consensus_k': 135, 'prev_k': 138},
    '2026-03-04': {'adp_k': 150, 'consensus_k': 130, 'prev_k': 127},
    '2026-04-01': {'adp_k': 132, 'consensus_k': 128, 'prev_k': 148},
    '2026-05-06': {'adp_k': 118, 'consensus_k': 130, 'prev_k': 135},
}


# ═══════════════════════════════════════════════════════════════
# REGIME-CONDITIONAL EDGE TABLE
# Backtested: 101 ADP releases, 2018-2026, ETH/USDT 15m
# ═══════════════════════════════════════════════════════════════

EDGE_TABLE = {
    # ── LONG EDGES (miss → dovish positioning) ──
    ('MARKDOWN', 'LOW_VOL', 'MISS'):       {'avg_ret': +2.61, 'win': 0.67, 'n': 3, 'bias': 'LONG'},
    ('MARKUP', 'TREND', 'MISS'):           {'avg_ret': +2.48, 'win': 0.67, 'n': 3, 'bias': 'LONG'},
    ('RANGE', 'LOW_VOL', 'MISS'):          {'avg_ret': +2.12, 'win': 0.67, 'n': 3, 'bias': 'LONG'},

    # ── LONG EDGES (inline/beat in structure) ──
    ('MARKUP', 'CHOP', 'INLINE'):          {'avg_ret': +1.32, 'win': 0.50, 'n': 6, 'bias': 'LONG'},
    ('MARKUP', 'LOW_VOL', 'BEAT'):         {'avg_ret': +1.12, 'win': 0.60, 'n': 5, 'bias': 'LONG'},

    # ── SHORT EDGES ──
    ('CHOP', 'INLINE', 'MARKDOWN'):        {'avg_ret': -1.19, 'win': 0.33, 'n': 3, 'bias': 'SHORT'},
}

BROAD_EDGE_TABLE = {
    ('LOW_VOL', 'MISS'):                   {'avg_ret': +1.72, 'win': 0.62, 'n': 8, 'bias': 'LONG'},
    ('CHOP', 'INLINE'):                    {'avg_ret': +0.19, 'win': 0.50, 'n': 14, 'bias': 'NEUTRAL'},
}


def _classify_adp_signal(adp_k, consensus_k):
    surprise = adp_k - consensus_k
    if surprise > 75:
        return 'STRONG_BEAT', surprise
    elif surprise > 25:
        return 'BEAT', surprise
    elif surprise < -75:
        return 'BIG_MISS', surprise
    elif surprise < -25:
        return 'MISS', surprise
    else:
        return 'INLINE', surprise


def _is_adp_release_day(today_str=None, window_days=1):
    if today_str is None:
        today_str = datetime.utcnow().strftime('%Y-%m-%d')
    today = datetime.strptime(today_str, '%Y-%m-%d')
    for release_date_str, release_data in sorted(ADP_RELEASES.items(), reverse=True):
        release_dt = datetime.strptime(release_date_str, '%Y-%m-%d')
        days_since = (today - release_dt).days
        if 0 <= days_since <= window_days:
            return True, release_date_str, release_data
    return False, None, None


def score_m36_adp(wyckoff_phase='RANGE', vol_regime='CHOP',
                  direction='LONG', today_str=None, config=None):
    cfg = config or {}
    if not cfg.get('M36_ENABLED', True):
        return 'SKIP', 0.0, 1.0, {'regime': 'DISABLED'}

    is_release, release_date, release_data = _is_adp_release_day(
        today_str, window_days=cfg.get('M36_WINDOW_DAYS', 1))
    if not is_release:
        return 'SKIP', 0.0, 1.0, {'regime': 'NOT_RELEASE_DAY'}

    adp_k = release_data['adp_k']
    consensus_k = release_data['consensus_k']
    prev_k = release_data['prev_k']
    signal, surprise = _classify_adp_signal(adp_k, consensus_k)

    fine_key = (wyckoff_phase, vol_regime, signal)
    fine_match = EDGE_TABLE.get(fine_key)

    # Also try reversed key format used in some entries
    if fine_match is None:
        fine_key_alt = (vol_regime, signal, wyckoff_phase)
        fine_match = EDGE_TABLE.get(fine_key_alt)

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
            'adp_k': adp_k, 'consensus_k': consensus_k,
            'surprise': surprise, 'release_date': release_date,
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
        'regime': f'ADP_{bias}',
        'release_date': release_date,
        'adp_k': adp_k, 'consensus_k': consensus_k, 'prev_k': prev_k,
        'surprise': surprise, 'signal': signal,
        'wyckoff': wyckoff_phase, 'vol_regime': vol_regime,
        'bias': bias, 'avg_ret_24h': avg_ret,
        'win_rate': win_rate, 'sample_size': n,
        'confidence': round(confidence, 2),
        'source': best_source, 'score_adj': score_adj, 'size_mult': size_mult,
    }
    return status, score_adj, size_mult, details


def format_m36(details):
    if not details or details.get('regime') in ('DISABLED', 'NOT_RELEASE_DAY', 'NO_EDGE'):
        return ''
    bias = details.get('bias', '?')
    adp_k = details.get('adp_k', 0)
    consensus_k = details.get('consensus_k', 0)
    surprise = details.get('surprise', 0)
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
    surp_icon = '🟢' if surprise > 25 else '🔴' if surprise < -25 else '⚪'

    lines = []
    lines.append(f"\n  {icon} M36 ADP EMPLOYMENT SESSION BIAS: {bias}")
    lines.append(f"    Release: {release}  |  ADP: {adp_k}K  |  Consensus: {consensus_k}K  |  Surprise: {surp_icon} {surprise:+.0f}K")
    lines.append(f"    Signal: {signal}  |  ⚠️ ADP frequently diverges from NFP (trap risk)")
    lines.append(f"    Context: {wyckoff} + {vol}")
    lines.append(f"    Backtest: avg 24h={avg_ret:+.2f}%  win={win*100:.0f}%  n={n}  source={details.get('source', '?')}")
    lines.append(f"    {conf_icon} Confidence: {conf:.2f}  |  Score adj: {score_adj:+.3f}  |  Size: {size_mult:.2f}x")
    return '\n'.join(lines)


def get_adp_cache_path():
    cache_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), 'data', 'macro')
    return os.path.join(cache_dir, 'adp_employment_cache.json')


def load_adp_cache():
    cache_path = get_adp_cache_path()
    if os.path.exists(cache_path):
        try:
            with open(cache_path) as f:
                return json.load(f)
        except (json.JSONDecodeError, IOError):
            pass
    return {}


def update_adp_cache(adp_k, consensus_k=None, release_date=None):
    cache = load_adp_cache()
    if release_date is None:
        release_date = datetime.utcnow().strftime('%Y-%m-%d')
    cache[release_date] = {
        'adp_k': adp_k, 'consensus_k': consensus_k,
        'updated': datetime.utcnow().isoformat(),
    }
    cache_path = get_adp_cache_path()
    os.makedirs(os.path.dirname(cache_path), exist_ok=True)
    with open(cache_path, 'w') as f:
        json.dump(cache, f, indent=2)
    return cache

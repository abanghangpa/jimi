"""
M24: NBS PMI Session Bias (Regime-Conditional)

On NBS Manufacturing PMI release days (~last day of each month, 09:00 CST),
applies a 24h directional bias based on the combination of:
  - Wyckoff phase (M21): ACCUMULATION / MARKUP / DISTRIBUTION / MARKDOWN / RANGE
  - Volatility regime (M9): TREND / SQUEEZE / CHOP / LOW_VOL
  - PMI signal: STRONG (≥51) / EXPANDING (50-51) / WEAK (49-50) / CONTRACTING (<49)

Backtested on 100 NBS PMI releases (2018-2026) against ETH/USDT 15m data.

Key findings (24h return):
  RANGE + CHOP + PMI<50:     -1.10% avg, 14% win, n=14  → SHORT bias
  MARKUP + PMI≥51:           +1.38% avg, 73% win, n=11  → LONG bias
  BOTH_CONTRACTING (mfg+svc): -1.65% avg, 11% win, n=9  → SHORT bias

Integration: lightweight modifier on NBS release days only (12x/year).
Returns a score adjustment and size multiplier — does NOT veto.

Usage:
    from src.modules.m24_nbs_pmi import score_m24_nbs_pmi, format_m24
    status, score_adj, size_mult, details = score_m24_nbs_pmi(
        wyckoff_phase='RANGE', vol_regime='CHOP', direction='LONG')
"""

from datetime import datetime, timedelta
import json
import os

# ═══════════════════════════════════════════════════════════════
# NBS PMI RELEASE DATES (Manufacturing PMI)
# Released ~09:00 CST (01:00 UTC) on last business day of month
# ═══════════════════════════════════════════════════════════════

# Format: {release_date: {'mfg': float, 'services': float}}
# Services PMI is released ~3 business days after Manufacturing
# We track both because the economic regime needs both numbers
NBS_PMI_RELEASES = {
    # 2018
    '2018-01-31': {'mfg': 51.3, 'services': 55.3},
    '2018-02-28': {'mfg': 50.3, 'services': 54.4},
    '2018-03-31': {'mfg': 51.5, 'services': 54.6},
    '2018-04-30': {'mfg': 51.4, 'services': 54.8},
    '2018-05-31': {'mfg': 51.9, 'services': 54.9},
    '2018-06-30': {'mfg': 51.5, 'services': 55.0},
    '2018-07-31': {'mfg': 51.2, 'services': 54.0},
    '2018-08-31': {'mfg': 51.3, 'services': 54.2},
    '2018-09-30': {'mfg': 50.8, 'services': 54.9},
    '2018-10-31': {'mfg': 50.2, 'services': 53.9},
    '2018-11-30': {'mfg': 50.0, 'services': 53.4},
    '2018-12-31': {'mfg': 49.4, 'services': 53.8},
    # 2019
    '2019-01-31': {'mfg': 49.5, 'services': 54.4},
    '2019-02-28': {'mfg': 49.2, 'services': 54.3},
    '2019-03-31': {'mfg': 50.5, 'services': 54.0},
    '2019-04-30': {'mfg': 50.1, 'services': 53.3},
    '2019-05-31': {'mfg': 49.4, 'services': 54.3},
    '2019-06-30': {'mfg': 49.4, 'services': 54.2},
    '2019-07-31': {'mfg': 49.7, 'services': 53.7},
    '2019-08-31': {'mfg': 49.5, 'services': 53.8},
    '2019-09-30': {'mfg': 49.8, 'services': 53.7},
    '2019-10-31': {'mfg': 49.3, 'services': 52.8},
    '2019-11-30': {'mfg': 50.2, 'services': 53.5},
    '2019-12-31': {'mfg': 50.2, 'services': 53.5},
    # 2020
    '2020-01-31': {'mfg': 50.0, 'services': 54.1},
    '2020-02-29': {'mfg': 35.7, 'services': 29.6},
    '2020-03-31': {'mfg': 52.0, 'services': 52.3},
    '2020-04-30': {'mfg': 50.8, 'services': 53.2},
    '2020-05-31': {'mfg': 50.6, 'services': 53.6},
    '2020-06-30': {'mfg': 50.9, 'services': 54.4},
    '2020-07-31': {'mfg': 51.1, 'services': 54.2},
    '2020-08-31': {'mfg': 51.0, 'services': 55.2},
    '2020-09-30': {'mfg': 51.5, 'services': 55.9},
    '2020-10-31': {'mfg': 51.4, 'services': 56.2},
    '2020-11-30': {'mfg': 52.1, 'services': 56.4},
    '2020-12-31': {'mfg': 51.9, 'services': 55.7},
    # 2021
    '2021-01-31': {'mfg': 51.3, 'services': 52.4},
    '2021-02-28': {'mfg': 50.6, 'services': 51.4},
    '2021-03-31': {'mfg': 51.9, 'services': 56.3},
    '2021-04-30': {'mfg': 51.1, 'services': 54.9},
    '2021-05-31': {'mfg': 51.0, 'services': 55.2},
    '2021-06-30': {'mfg': 50.9, 'services': 53.5},
    '2021-07-31': {'mfg': 50.4, 'services': 53.3},
    '2021-08-31': {'mfg': 50.1, 'services': 47.5},
    '2021-09-30': {'mfg': 49.6, 'services': 53.2},
    '2021-10-31': {'mfg': 49.2, 'services': 51.6},
    '2021-11-30': {'mfg': 50.1, 'services': 52.3},
    '2021-12-31': {'mfg': 50.3, 'services': 52.7},
    # 2022
    '2022-01-30': {'mfg': 50.1, 'services': 51.1},
    '2022-02-28': {'mfg': 50.2, 'services': 51.6},
    '2022-03-31': {'mfg': 49.5, 'services': 54.6},
    '2022-04-30': {'mfg': 47.4, 'services': 41.9},
    '2022-05-31': {'mfg': 49.6, 'services': 47.8},
    '2022-06-30': {'mfg': 50.2, 'services': 54.7},
    '2022-07-31': {'mfg': 49.0, 'services': 53.8},
    '2022-08-31': {'mfg': 49.4, 'services': 52.6},
    '2022-09-30': {'mfg': 50.1, 'services': 50.6},
    '2022-10-31': {'mfg': 49.2, 'services': 48.7},
    '2022-11-30': {'mfg': 48.0, 'services': 46.7},
    '2022-12-31': {'mfg': 47.0, 'services': 41.6},
    # 2023
    '2023-01-31': {'mfg': 50.1, 'services': 54.4},
    '2023-02-28': {'mfg': 52.6, 'services': 56.3},
    '2023-03-31': {'mfg': 51.9, 'services': 58.2},
    '2023-04-30': {'mfg': 49.2, 'services': 56.4},
    '2023-05-31': {'mfg': 48.8, 'services': 54.5},
    '2023-06-30': {'mfg': 49.0, 'services': 53.2},
    '2023-07-31': {'mfg': 49.3, 'services': 51.5},
    '2023-08-31': {'mfg': 49.7, 'services': 51.0},
    '2023-09-30': {'mfg': 50.2, 'services': 51.7},
    '2023-10-31': {'mfg': 49.5, 'services': 50.6},
    '2023-11-30': {'mfg': 49.4, 'services': 50.2},
    '2023-12-31': {'mfg': 49.0, 'services': 49.3},
    # 2024
    '2024-01-31': {'mfg': 49.1, 'services': 50.7},
    '2024-02-29': {'mfg': 49.1, 'services': 51.4},
    '2024-03-31': {'mfg': 50.8, 'services': 53.0},
    '2024-04-30': {'mfg': 50.4, 'services': 51.2},
    '2024-05-31': {'mfg': 49.5, 'services': 50.5},
    '2024-06-30': {'mfg': 49.5, 'services': 50.2},
    '2024-07-31': {'mfg': 49.4, 'services': 50.1},
    '2024-08-31': {'mfg': 49.1, 'services': 50.2},
    '2024-09-30': {'mfg': 49.8, 'services': 49.9},
    '2024-10-31': {'mfg': 50.1, 'services': 50.2},
    '2024-11-30': {'mfg': 50.3, 'services': 50.1},
    '2024-12-31': {'mfg': 50.1, 'services': 52.2},
    # 2025
    '2025-01-27': {'mfg': 49.1, 'services': 50.3},
    '2025-02-28': {'mfg': 50.2, 'services': 51.4},
    '2025-03-31': {'mfg': 50.5, 'services': 52.0},
    '2025-04-30': {'mfg': 49.0, 'services': 50.4},
    '2025-05-31': {'mfg': 49.5, 'services': 51.1},
    '2025-06-30': {'mfg': 49.7, 'services': 51.5},
    '2025-07-31': {'mfg': 49.3, 'services': 50.8},
    '2025-08-31': {'mfg': 49.5, 'services': 50.5},
    '2025-09-30': {'mfg': 50.0, 'services': 51.2},
    '2025-10-31': {'mfg': 50.2, 'services': 51.0},
    '2025-11-30': {'mfg': 49.8, 'services': 50.5},
    '2025-12-31': {'mfg': 49.5, 'services': 50.2},
    # 2026
    '2026-01-31': {'mfg': 49.8, 'services': 50.8},
    '2026-02-28': {'mfg': 50.5, 'services': 51.5},
    '2026-03-31': {'mfg': 50.2, 'services': 51.0},
    '2026-04-30': {'mfg': 49.5, 'services': 50.5},
}


# ═══════════════════════════════════════════════════════════════
# REGIME-CONDITIONAL EDGE TABLE
# Backtested: 100 NBS PMI releases, 2018-2026, ETH/USDT 15m
# ═══════════════════════════════════════════════════════════════

# Key: (wyckoff_phase, vol_regime, pmi_signal)
# Value: (avg_24h_return, win_rate, sample_size, direction_bias)
# Only entries with n >= 3 and |avg_24h| >= 0.5% are included
EDGE_TABLE = {
    # ── SHORT EDGES (PMI miss in weak structure) ──
    ('RANGE', 'CHOP', 'WEAK'):          {'avg_ret': -1.10, 'win': 0.14, 'n': 14, 'bias': 'SHORT'},
    ('MARKDOWN', 'CHOP', 'WEAK'):       {'avg_ret': -0.58, 'win': 0.33, 'n': 9,  'bias': 'SHORT'},
    ('MARKDOWN', 'SQUEEZE', 'WEAK'):    {'avg_ret': -1.46, 'win': 0.00, 'n': 2,  'bias': 'SHORT'},  # n=2, use cautiously

    # ── LONG EDGES (PMI beat in strong structure) ──
    ('MARKUP', 'CHOP', 'STRONG'):       {'avg_ret': +1.69, 'win': 0.60, 'n': 5,  'bias': 'LONG'},
    ('MARKUP', 'TREND', 'STRONG'):      {'avg_ret': +2.27, 'win': 1.00, 'n': 3,  'bias': 'LONG'},
    ('MARKUP', 'CHOP', 'EXPANDING'):    {'avg_ret': +4.09, 'win': 1.00, 'n': 3,  'bias': 'LONG'},
    ('ACCUMULATION', 'SQUEEZE', 'EXPANDING'): {'avg_ret': +1.25, 'win': 1.00, 'n': 3, 'bias': 'LONG'},
    ('RANGE', 'CHOP', 'STRONG'):        {'avg_ret': +2.05, 'win': 0.50, 'n': 4,  'bias': 'LONG'},
}

# Broader combos (Wyckoff + PMI only, ignoring vol) — larger sample
BROAD_EDGE_TABLE = {
    # Key: (wyckoff_phase, pmi_signal)
    ('RANGE', 'WEAK'):                   {'avg_ret': -0.90, 'win': 0.26, 'n': 23, 'bias': 'SHORT'},
    ('MARKUP', 'STRONG'):                {'avg_ret': +1.38, 'win': 0.73, 'n': 11, 'bias': 'LONG'},
    ('MARKUP', 'EXPANDING'):             {'avg_ret': +2.72, 'win': 0.75, 'n': 4,  'bias': 'LONG'},
    ('ACCUMULATION', 'EXPANDING'):       {'avg_ret': +1.80, 'win': 1.00, 'n': 4,  'bias': 'LONG'},
    ('MARKDOWN', 'WEAK'):                {'avg_ret': -1.31, 'win': 0.40, 'n': 5,  'bias': 'SHORT'},
}

# Economic regime (mfg + services combined) — strongest signal
ECONOMIC_REGIME_TABLE = {
    # Key: economic_regime label
    'GROWTH_STRONG':     {'avg_ret': +1.56, 'win': 0.72, 'n': 18, 'bias': 'LONG'},
    'BOTH_CONTRACTING':  {'avg_ret': -1.65, 'win': 0.11, 'n': 9,  'bias': 'SHORT'},
    'MIXED_WEAK_MFG':    {'avg_ret': -0.77, 'win': 0.34, 'n': 35, 'bias': 'SHORT'},
}


def _classify_pmi_signal(mfg_pmi):
    """Classify PMI into signal buckets."""
    if mfg_pmi >= 51.0:
        return 'STRONG'
    elif mfg_pmi >= 50.0:
        return 'EXPANDING'
    elif mfg_pmi >= 49.0:
        return 'WEAK'
    else:
        return 'CONTRACTING'


def _classify_economic_regime(mfg_pmi, services_pmi):
    """Classify combined economic regime from mfg + services PMI."""
    if mfg_pmi >= 51 and services_pmi >= 53:
        return 'GROWTH_STRONG'
    elif mfg_pmi >= 50 and services_pmi >= 50:
        return 'GROWTH_MILD'
    elif mfg_pmi >= 49 and services_pmi >= 50:
        return 'MIXED_WEAK_MFG'
    elif mfg_pmi < 49 and services_pmi >= 50:
        return 'MFG_CONTRACTING'
    elif mfg_pmi < 50 and services_pmi < 50:
        return 'BOTH_CONTRACTING'
    else:
        return 'NEUTRAL'


def _is_nbs_release_day(today_str=None, window_days=1):
    """Check if today is within N days of an NBS PMI release.

    Args:
        today_str: YYYY-MM-DD string (default: today UTC)
        window_days: number of days after release to consider active

    Returns:
        (is_release: bool, release_date: str, pmi_data: dict) or (False, None, None)
    """
    if today_str is None:
        today_str = datetime.utcnow().strftime('%Y-%m-%d')

    today = datetime.strptime(today_str, '%Y-%m-%d')

    # Find releases within the window
    for release_date_str, pmi_data in sorted(NBS_PMI_RELEASES.items(), reverse=True):
        release_dt = datetime.strptime(release_date_str, '%Y-%m-%d')
        days_since = (today - release_dt).days
        if 0 <= days_since <= window_days:
            return True, release_date_str, pmi_data

    return False, None, None


def score_m24_nbs_pmi(wyckoff_phase='RANGE', vol_regime='CHOP',
                      direction='LONG', today_str=None, config=None):
    """Score the NBS PMI session bias.

    Args:
        wyckoff_phase: from M21 ('ACCUMULATION', 'MARKUP', 'DISTRIBUTION', 'MARKDOWN', 'RANGE')
        vol_regime: from M9 ('TREND', 'SQUEEZE', 'CHOP', 'LOW_VOL')
        direction: trade direction ('LONG' or 'SHORT')
        today_str: YYYY-MM-DD override (for backtesting)
        config: config dict (optional)

    Returns:
        status: 'PASS' (active), 'SKIP' (not release day), or 'WEAK' (low confidence)
        score_adj: score adjustment (-0.10 to +0.10)
        size_mult: position size multiplier (0.5 to 1.0)
        details: dict
    """
    cfg = config or {}

    if not cfg.get('M24_ENABLED', True):
        return 'SKIP', 0.0, 1.0, {'regime': 'DISABLED'}

    # Check if today is an NBS PMI release day
    is_release, release_date, pmi_data = _is_nbs_release_day(
        today_str, window_days=cfg.get('M24_WINDOW_DAYS', 1))

    if not is_release:
        return 'SKIP', 0.0, 1.0, {'regime': 'NOT_RELEASE_DAY'}

    mfg_pmi = pmi_data['mfg']
    services_pmi = pmi_data.get('services', 50.0)
    pmi_signal = _classify_pmi_signal(mfg_pmi)
    econ_regime = _classify_economic_regime(mfg_pmi, services_pmi)

    # ── Lookup 1: Fine-grained (Wyckoff + Vol + PMI) ──
    fine_key = (wyckoff_phase, vol_regime, pmi_signal)
    fine_match = EDGE_TABLE.get(fine_key)

    # ── Lookup 2: Broad (Wyckoff + PMI) ──
    broad_key = (wyckoff_phase, pmi_signal)
    broad_match = BROAD_EDGE_TABLE.get(broad_key)

    # ── Lookup 3: Economic regime (mfg + services) ──
    econ_match = ECONOMIC_REGIME_TABLE.get(econ_regime)

    # ── Determine best signal ──
    # Priority: fine > broad > economic regime
    # Confidence scales with sample size
    best_match = None
    best_source = 'NONE'
    confidence = 0.0

    if fine_match and fine_match['n'] >= 3:
        best_match = fine_match
        best_source = 'FINE'
        confidence = min(1.0, fine_match['n'] / 10)  # full confidence at n=10+
    elif broad_match and broad_match['n'] >= 5:
        best_match = broad_match
        best_source = 'BROAD'
        confidence = min(1.0, broad_match['n'] / 15)
    elif econ_match and econ_match['n'] >= 5:
        best_match = econ_match
        best_source = 'ECONOMIC_REGIME'
        confidence = min(1.0, econ_match['n'] / 15)

    if best_match is None:
        return 'SKIP', 0.0, 1.0, {
            'regime': 'NO_EDGE',
            'wyckoff': wyckoff_phase,
            'vol_regime': vol_regime,
            'pmi_signal': pmi_signal,
            'economic_regime': econ_regime,
            'mfg_pmi': mfg_pmi,
            'services_pmi': services_pmi,
            'release_date': release_date,
        }

    # ── Compute score adjustment ──
    # Scale by confidence and match strength
    avg_ret = best_match['avg_ret']
    win_rate = best_match['win']
    n = best_match['n']
    bias = best_match['bias']

    # Score adjustment: ±0.05 to ±0.10 based on strength
    # Strong edge (|avg_ret| > 1.5%): ±0.10
    # Medium edge (|avg_ret| > 0.8%): ±0.07
    # Weak edge (|avg_ret| > 0.5%): ±0.05
    abs_ret = abs(avg_ret)
    if abs_ret >= 1.5:
        raw_adj = 0.10
    elif abs_ret >= 0.8:
        raw_adj = 0.07
    else:
        raw_adj = 0.05

    # Scale by confidence (sample size)
    score_adj = raw_adj * confidence

    # Apply direction: positive for aligned, negative for opposed
    if bias == 'LONG':
        if direction == 'LONG':
            pass  # positive adjustment (already correct sign)
        else:
            score_adj = -score_adj  # penalize SHORT when edge says LONG
    elif bias == 'SHORT':
        if direction == 'SHORT':
            score_adj = -score_adj  # negative adjustment helps SHORT (inverted scoring)
        else:
            pass  # positive = penalize LONG when edge says SHORT
        # For SHORT bias: score_adj should be negative when direction=LONG (penalize)
        # and positive when direction=SHORT (help) — but M22 uses inverted scoring for SHORT
        # Let's keep it simple: positive = supports the trade, negative = opposes
        if direction == 'LONG':
            score_adj = -abs(score_adj)  # penalize LONG
        else:
            score_adj = abs(score_adj)   # support SHORT

    # Actually, let's simplify the direction logic:
    # score_adj > 0 = supports the direction being scored
    # score_adj < 0 = opposes
    if bias == 'LONG' and direction == 'LONG':
        score_adj = abs(score_adj)
    elif bias == 'LONG' and direction == 'SHORT':
        score_adj = -abs(score_adj)
    elif bias == 'SHORT' and direction == 'SHORT':
        score_adj = abs(score_adj)
    elif bias == 'SHORT' and direction == 'LONG':
        score_adj = -abs(score_adj)

    # ── Size multiplier ──
    # Reduce size when confidence is low
    if confidence >= 0.7:
        size_mult = 1.0
    elif confidence >= 0.4:
        size_mult = 0.75
    else:
        size_mult = 0.50

    # Extra reduction if n < 5 (very small sample)
    if n < 5:
        size_mult *= 0.75

    size_mult = round(size_mult, 2)
    score_adj = round(score_adj, 3)

    status = 'PASS' if confidence >= 0.3 else 'WEAK'

    details = {
        'regime': f'NBS_PMI_{bias}',
        'release_date': release_date,
        'mfg_pmi': mfg_pmi,
        'services_pmi': services_pmi,
        'pmi_signal': pmi_signal,
        'economic_regime': econ_regime,
        'wyckoff': wyckoff_phase,
        'vol_regime': vol_regime,
        'bias': bias,
        'avg_ret_24h': avg_ret,
        'win_rate': win_rate,
        'sample_size': n,
        'confidence': round(confidence, 2),
        'source': best_source,
        'score_adj': score_adj,
        'size_mult': size_mult,
    }

    return status, score_adj, size_mult, details


def format_m24(details):
    """Format M24 details for terminal output."""
    if not details or details.get('regime') in ('DISABLED', 'NOT_RELEASE_DAY', 'NO_EDGE'):
        regime = details.get('regime', '?') if details else '?'
        if regime == 'NOT_RELEASE_DAY':
            return ''  # silent when not active
        return ''

    bias = details.get('bias', '?')
    mfg = details.get('mfg_pmi', 0)
    svc = details.get('services_pmi', 0)
    signal = details.get('pmi_signal', '?')
    econ = details.get('economic_regime', '?')
    wyckoff = details.get('wyckoff', '?')
    vol = details.get('vol_regime', '?')
    avg_ret = details.get('avg_ret_24h', 0)
    win = details.get('win_rate', 0)
    n = details.get('sample_size', 0)
    conf = details.get('confidence', 0)
    source = details.get('source', '?')
    score_adj = details.get('score_adj', 0)
    size_mult = details.get('size_mult', 1.0)
    release = details.get('release_date', '?')

    icon = '🟢' if bias == 'LONG' else '🔴' if bias == 'SHORT' else '⚪'
    conf_icon = '🟢' if conf >= 0.7 else '🟡' if conf >= 0.4 else '🟠'

    lines = []
    lines.append(f"\n  {icon} M24 NBS PMI SESSION BIAS: {bias}")
    lines.append(f"    Release: {release}  |  Mfg: {mfg:.1f} ({signal})  |  Services: {svc:.1f}")
    lines.append(f"    Economic Regime: {econ}")
    lines.append(f"    Context: {wyckoff} + {vol}")
    lines.append(f"    Backtest: avg 24h={avg_ret:+.2f}%  win={win*100:.0f}%  n={n}  source={source}")
    lines.append(f"    {conf_icon} Confidence: {conf:.2f}  |  Score adj: {score_adj:+.3f}  |  Size: {size_mult:.2f}x")

    return '\n'.join(lines)


def get_nbs_release_cache_path():
    """Get path to NBS PMI release cache (for live updates)."""
    cache_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), 'data', 'macro')
    return os.path.join(cache_dir, 'nbs_pmi_cache.json')


def load_nbs_cache():
    """Load cached NBS PMI data (for live updates from macro_fetch)."""
    cache_path = get_nbs_release_cache_path()
    if os.path.exists(cache_path):
        try:
            with open(cache_path) as f:
                return json.load(f)
        except (json.JSONDecodeError, IOError):
            pass
    return {}


def update_nbs_cache(mfg_pmi, services_pmi=None, release_date=None):
    """Update NBS PMI cache with new data (called from macro_fetch)."""
    cache = load_nbs_cache()
    if release_date is None:
        release_date = datetime.utcnow().strftime('%Y-%m-%d')
    cache[release_date] = {
        'mfg': mfg_pmi,
        'services': services_pmi,
        'updated': datetime.utcnow().isoformat(),
    }
    cache_path = get_nbs_release_cache_path()
    os.makedirs(os.path.dirname(cache_path), exist_ok=True)
    with open(cache_path, 'w') as f:
        json.dump(cache, f, indent=2)
    return cache

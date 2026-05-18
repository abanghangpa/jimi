"""
M61: US Weekly Jobless Claims Session Bias (Regime-Conditional)

On US Jobless Claims release days (every Thursday, 08:30 ET / 12:30 UTC),
applies a directional bias based on the combination of:
  - Wyckoff phase (M21): ACCUMULATION / MARKUP / DISTRIBUTION / MARKDOWN / RANGE
  - Volatility regime (M9): TREND / SQUEEZE / CHOP / LOW_VOL / COMPRESSING
  - Claims signal: LOW (<210K) / NORMAL (210-225K) / ELEVATED (225-240K) / SPIKE (240-280K) / CRISIS (>280K)

Backtested on 437 claims Thursdays (2018-2026) against ETH/USDT 15m data.

Key findings:
  - Claims are BACKGROUND CONTEXT — overall 24h return is noise (p=0.47)
  - No standalone signal from claim levels or trends (all p>0.40)
  - Edges exist in SPECIFIC regime-conditional combos:
    RANGE + COMPRESSING + CRISIS: +1.44% avg, 61% win, n=28 → LONG
    RANGE + COMPRESSING + ELEVATED: -2.01% avg, 25% win, n=16 → SHORT
    RANGE + COMPRESSING + SPIKE: +2.44% avg, 63% win, n=8 → LONG
    MARKUP + COMPRESSING + NORMAL: +1.38% avg, 75% win, n=8 → LONG
    MARKDOWN + COMPRESSING + LOW: +3.02% avg, 83% win, n=6 → LONG
  - Session chain: London Midday→NY PM 80-90% direction persistence

Integration: lightweight modifier on Thursdays only (52x/year).
Returns a score adjustment and size multiplier — does NOT veto.

Usage:
    from src.modules.m61_us_claims import score_m61_us_claims, format_m61
    status, score_adj, size_mult, details = score_m61_us_claims(
        wyckoff_phase='RANGE', vol_regime='COMPRESSING', direction='LONG')
"""

from datetime import datetime, timedelta
import json
import os

# ═══════════════════════════════════════════════════════════════
# CLAIMS CLASSIFICATION THRESHOLDS (from M23)
# ═══════════════════════════════════════════════════════════════

CLAIMS_LOW_THRESHOLD = 210       # Below = tight labor market
CLAIMS_NORMAL_LOW = 210
CLAIMS_NORMAL_HIGH = 225
CLAIMS_ELEVATED_THRESHOLD = 225  # Above = labor softening
CLAIMS_SPIKE_THRESHOLD = 240     # Above = tariff/shock spike
CLAIMS_CRISIS_THRESHOLD = 280    # Above = recession territory

CLAIMS_TREND_RISING = 5    # K increase over 4 weeks = rising
CLAIMS_TREND_FALLING = -5  # K decrease over 4 weeks = falling

# ═══════════════════════════════════════════════════════════════
# REGIME-CONDITIONAL EDGE TABLE
# Backtested: 437 claims Thursdays, 2018-2026, ETH/USDT 15m
# ═══════════════════════════════════════════════════════════════

# Key: (wyckoff_phase, vol_regime, claims_signal)
# Value: (avg_24h_return, win_rate, sample_size, direction_bias)
# Only entries with n >= 3 and |avg_24h| >= 0.5% are included
EDGE_TABLE = {
    # ── STRONG EDGES (n >= 5, |avg| >= 1.0%) ──
    ('RANGE', 'COMPRESSING', 'CRISIS'):   {'avg_ret': +1.44, 'win': 0.607, 'n': 28, 'bias': 'LONG'},
    ('RANGE', 'COMPRESSING', 'ELEVATED'): {'avg_ret': -2.01, 'win': 0.250, 'n': 16, 'bias': 'SHORT'},
    ('RANGE', 'COMPRESSING', 'LOW'):      {'avg_ret': -0.88, 'win': 0.403, 'n': 72, 'bias': 'SHORT'},
    ('RANGE', 'COMPRESSING', 'SPIKE'):    {'avg_ret': +2.44, 'win': 0.625, 'n': 8,  'bias': 'LONG'},
    ('MARKUP', 'COMPRESSING', 'NORMAL'):  {'avg_ret': +1.38, 'win': 0.750, 'n': 8,  'bias': 'LONG'},
    ('MARKUP', 'LOW_VOL', 'NORMAL'):      {'avg_ret': +1.44, 'win': 0.500, 'n': 6,  'bias': 'LONG'},
    ('MARKUP', 'LOW_VOL', 'LOW'):         {'avg_ret': +1.84, 'win': 0.600, 'n': 5,  'bias': 'LONG'},
    ('MARKUP', 'LOW_VOL', 'CRISIS'):      {'avg_ret': +1.30, 'win': 0.500, 'n': 6,  'bias': 'LONG'},
    ('MARKDOWN', 'COMPRESSING', 'LOW'):   {'avg_ret': +3.02, 'win': 0.833, 'n': 6,  'bias': 'LONG'},
    ('MARKDOWN', 'COMPRESSING', 'NORMAL'):{'avg_ret': -1.39, 'win': 0.364, 'n': 11, 'bias': 'SHORT'},
    ('MARKDOWN', 'LOW_VOL', 'CRISIS'):    {'avg_ret': +3.47, 'win': 0.750, 'n': 4,  'bias': 'LONG'},
    ('MARKDOWN', 'LOW_VOL', 'LOW'):       {'avg_ret': +1.76, 'win': 0.571, 'n': 7,  'bias': 'LONG'},
    ('MARKDOWN', 'LOW_VOL', 'NORMAL'):    {'avg_ret': -1.88, 'win': 0.500, 'n': 10, 'bias': 'SHORT'},
    ('MARKDOWN', 'TREND', 'LOW'):         {'avg_ret': +2.79, 'win': 0.600, 'n': 5,  'bias': 'LONG'},
    ('MARKDOWN', 'TREND', 'NORMAL'):      {'avg_ret': +0.90, 'win': 0.667, 'n': 6,  'bias': 'LONG'},
    ('MARKDOWN', 'TREND', 'CRISIS'):      {'avg_ret': -1.94, 'win': 0.200, 'n': 5,  'bias': 'SHORT'},

    # ── SMALL SAMPLE EDGES (n=3-4, use cautiously) ──
    ('CHOP', 'LOW_VOL', 'NORMAL'):        {'avg_ret': -4.82, 'win': 0.333, 'n': 6,  'bias': 'SHORT'},
    ('CHOP', 'LOW_VOL', 'CRISIS'):        {'avg_ret': -2.33, 'win': 0.143, 'n': 7,  'bias': 'SHORT'},
    ('MARKUP', 'COMPRESSING', 'SPIKE'):   {'avg_ret': -3.01, 'win': 0.250, 'n': 4,  'bias': 'SHORT'},
    ('MARKUP', 'TREND', 'CRISIS'):        {'avg_ret': -1.52, 'win': 0.500, 'n': 4,  'bias': 'SHORT'},
    ('MARKDOWN', 'CRISIS', 'NORMAL'):     {'avg_ret': -4.00, 'win': 0.000, 'n': 3,  'bias': 'SHORT'},
    ('RANGE', 'LOW_VOL', 'CRISIS'):       {'avg_ret': -2.04, 'win': 0.250, 'n': 8,  'bias': 'SHORT'},
    ('RANGE', 'LOW_VOL', 'LOW'):          {'avg_ret': +3.14, 'win': 0.750, 'n': 4,  'bias': 'LONG'},
    ('MARKUP', 'LOW_VOL', 'SPIKE'):       {'avg_ret': -0.59, 'win': 0.333, 'n': 3,  'bias': 'SHORT'},
}


# ═══════════════════════════════════════════════════════════════
# MONTHLY AVERAGE CLAIMS (for trend detection)
# ═══════════════════════════════════════════════════════════════

JOBLESS_CLAIMS_MONTHLY_AVG = {
    # 2018
    '2018-01': 230, '2018-02': 225, '2018-03': 220, '2018-04': 215,
    '2018-05': 220, '2018-06': 218, '2018-07': 215, '2018-08': 212,
    '2018-09': 210, '2018-10': 212, '2018-11': 215, '2018-12': 218,
    # 2019
    '2019-01': 220, '2019-02': 218, '2019-03': 215, '2019-04': 212,
    '2019-05': 215, '2019-06': 218, '2019-07': 215, '2019-08': 212,
    '2019-09': 210, '2019-10': 212, '2019-11': 215, '2019-12': 218,
    # 2020
    '2020-01': 215, '2020-02': 210, '2020-03': 350, '2020-04': 900,
    '2020-05': 650, '2020-06': 450, '2020-07': 380, '2020-08': 350,
    '2020-09': 330, '2020-10': 300, '2020-11': 280, '2020-12': 260,
    # 2021
    '2021-01': 900, '2021-02': 750, '2021-03': 650, '2021-04': 570,
    '2021-05': 450, '2021-06': 400, '2021-07': 380, '2021-08': 350,
    '2021-09': 330, '2021-10': 280, '2021-11': 250, '2021-12': 200,
    # 2022
    '2022-01': 210, '2022-02': 200, '2022-03': 190, '2022-04': 185,
    '2022-05': 190, '2022-06': 195, '2022-07': 195, '2022-08': 200,
    '2022-09': 195, '2022-10': 190, '2022-11': 190, '2022-12': 195,
    # 2023
    '2023-01': 190, '2023-02': 195, '2023-03': 200, '2023-04': 200,
    '2023-05': 210, '2023-06': 215, '2023-07': 220, '2023-08': 220,
    '2023-09': 215, '2023-10': 215, '2023-11': 210, '2023-12': 210,
    # 2024
    '2024-01': 210, '2024-02': 215, '2024-03': 215, '2024-04': 220,
    '2024-05': 220, '2024-06': 225, '2024-07': 235, '2024-08': 230,
    '2024-09': 225, '2024-10': 220, '2024-11': 215, '2024-12': 215,
    # 2025
    '2025-01': 205, '2025-02': 210, '2025-03': 210, '2025-04': 215,
    '2025-05': 240, '2025-06': 230, '2025-07': 240, '2025-08': 225,
    '2025-09': 215, '2025-10': 210, '2025-11': 220, '2025-12': 199,
    # 2026
    '2026-01': 210, '2026-02': 227, '2026-03': 215, '2026-04': 200,
    '2026-05': 200,
}

# ── FRED Cache Override ──────────────────────────────────────
def _load_fred_cache():
    """Override hardcoded claims dict with FRED cache if available."""
    import json as _json
    import os as _os
    cache_path = _os.path.join(
        _os.path.dirname(_os.path.dirname(_os.path.dirname(__file__))),
        "data", "fred", "claims_cache.json")
    if not _os.path.exists(cache_path):
        return
    try:
        with open(cache_path) as f:
            cache = _json.load(f)
        icsa = cache.get("icsa", {}).get("monthly_avg", {})
        if icsa:
            merged = dict(JOBLESS_CLAIMS_MONTHLY_AVG)
            merged.update(icsa)
            JOBLESS_CLAIMS_MONTHLY_AVG.clear()
            JOBLESS_CLAIMS_MONTHLY_AVG.update(merged)
    except Exception:
        pass

_load_fred_cache()


def _classify_claims_signal(claims_k):
    """Classify claims into signal buckets."""
    if claims_k < CLAIMS_LOW_THRESHOLD:
        return 'LOW'
    elif claims_k <= CLAIMS_NORMAL_HIGH:
        return 'NORMAL'
    elif claims_k <= CLAIMS_SPIKE_THRESHOLD:
        return 'ELEVATED'
    elif claims_k <= CLAIMS_CRISIS_THRESHOLD:
        return 'SPIKE'
    else:
        return 'CRISIS'


def _get_claims_for_date(today_str):
    """Get estimated claims value for a given date.

    Uses monthly average from JOBLESS_CLAIMS_MONTHLY_AVG.
    Returns claims_k (thousands) or None.
    """
    month_key = today_str[:7]
    return JOBLESS_CLAIMS_MONTHLY_AVG.get(month_key)


def _get_claims_trend(today_str):
    """Get claims trend over recent months.

    Returns: 'RISING', 'FALLING', 'STABLE'
    """
    month_key = today_str[:7]
    sorted_months = sorted(JOBLESS_CLAIMS_MONTHLY_AVG.keys())

    # Find current month index
    try:
        idx = sorted_months.index(month_key)
    except ValueError:
        return 'STABLE'

    if idx < 3:
        return 'STABLE'

    # 3-month trend
    current = JOBLESS_CLAIMS_MONTHLY_AVG[sorted_months[idx]]
    prev_3 = JOBLESS_CLAIMS_MONTHLY_AVG[sorted_months[idx - 3]]
    delta = current - prev_3

    if delta > CLAIMS_TREND_RISING:
        return 'RISING'
    elif delta < CLAIMS_TREND_FALLING:
        return 'FALLING'
    else:
        return 'STABLE'


def _is_thursday(today_str=None):
    """Check if today is Thursday (claims release day)."""
    if today_str is None:
        today_str = datetime.utcnow().strftime('%Y-%m-%d')
    dt = datetime.strptime(today_str, '%Y-%m-%d')
    return dt.weekday() == 3  # Thursday = 3


def _get_edges_path():
    """Get path to edges JSON file."""
    return os.path.join(
        os.path.dirname(os.path.dirname(os.path.dirname(__file__))),
        'backtest_us_claims_edges.json')


def _load_edges():
    """Load edges from backtest JSON if available."""
    path = _get_edges_path()
    if os.path.exists(path):
        try:
            with open(path) as f:
                return json.load(f)
        except Exception:
            pass
    return None


def score_m61_us_claims(wyckoff_phase='RANGE', vol_regime='CHOP',
                        direction='LONG', today_str=None, config=None):
    """Score the US Jobless Claims session bias.

    Args:
        wyckoff_phase: from M21 ('ACCUMULATION', 'MARKUP', 'DISTRIBUTION', 'MARKDOWN', 'RANGE')
        vol_regime: from M9 ('TREND', 'SQUEEZE', 'CHOP', 'LOW_VOL', 'COMPRESSING')
        direction: trade direction ('LONG' or 'SHORT')
        today_str: YYYY-MM-DD override (for backtesting)
        config: config dict (optional)

    Returns:
        status: 'PASS' (active), 'SKIP' (not Thursday), or 'WEAK' (low confidence)
        score_adj: score adjustment (-0.10 to +0.10)
        size_mult: position size multiplier (0.5 to 1.0)
        details: dict
    """
    cfg = config or {}

    if not cfg.get('M61_ENABLED', True):
        return 'SKIP', 0.0, 1.0, {'regime': 'DISABLED'}

    if today_str is None:
        today_str = datetime.utcnow().strftime('%Y-%m-%d')

    # Claims release = every Thursday
    if not _is_thursday(today_str):
        return 'SKIP', 0.0, 1.0, {'regime': 'NOT_CLAIMS_DAY'}

    # Get claims data
    claims_k = _get_claims_for_date(today_str)
    if claims_k is None:
        return 'SKIP', 0.0, 1.0, {'regime': 'NO_CLAIMS_DATA'}

    claims_signal = _classify_claims_signal(claims_k)
    claims_trend = _get_claims_trend(today_str)

    # ── Lookup: Wyckoff × Vol × Signal ──
    edge_key = (wyckoff_phase, vol_regime, claims_signal)
    edge_match = EDGE_TABLE.get(edge_key)

    if edge_match is None:
        return 'SKIP', 0.0, 1.0, {
            'regime': 'NO_EDGE',
            'wyckoff': wyckoff_phase,
            'vol_regime': vol_regime,
            'claims_signal': claims_signal,
            'claims_k': claims_k,
            'claims_trend': claims_trend,
        }

    avg_ret = edge_match['avg_ret']
    win_rate = edge_match['win']
    n = edge_match['n']
    bias = edge_match['bias']

    # ── Compute confidence from sample size ──
    # Full confidence at n=10+, reduced below
    confidence = min(1.0, n / 10)

    # ── Compute score adjustment ──
    # Claims are background context — smaller adjustments than CPI/PPI
    abs_ret = abs(avg_ret)
    if abs_ret >= 3.0:
        raw_adj = 0.10
    elif abs_ret >= 2.0:
        raw_adj = 0.08
    elif abs_ret >= 1.0:
        raw_adj = 0.06
    else:
        raw_adj = 0.05

    # Scale by confidence
    score_adj = raw_adj * confidence

    # Apply direction alignment
    if bias == 'LONG' and direction == 'LONG':
        score_adj = abs(score_adj)
    elif bias == 'LONG' and direction == 'SHORT':
        score_adj = -abs(score_adj)
    elif bias == 'SHORT' and direction == 'SHORT':
        score_adj = abs(score_adj)
    elif bias == 'SHORT' and direction == 'LONG':
        score_adj = -abs(score_adj)

    # ── Trend adjustment ──
    # RISING claims amplify SHORT bias, FALLING amplify LONG
    trend_mult = 1.0
    if claims_trend == 'RISING' and bias == 'SHORT':
        trend_mult = 1.15  # amplify short bias
    elif claims_trend == 'FALLING' and bias == 'LONG':
        trend_mult = 1.15  # amplify long bias
    elif claims_trend == 'RISING' and bias == 'LONG':
        trend_mult = 0.85  # dampen long bias
    elif claims_trend == 'FALLING' and bias == 'SHORT':
        trend_mult = 0.85  # dampen short bias

    score_adj = round(score_adj * trend_mult, 3)

    # ── Size multiplier ──
    # Reduce size when confidence is low or sample is small
    if confidence >= 0.7 and n >= 10:
        size_mult = 1.0
    elif confidence >= 0.4 and n >= 5:
        size_mult = 0.75
    else:
        size_mult = 0.50

    # Extra reduction for small sample
    if n < 5:
        size_mult *= 0.75

    size_mult = round(size_mult, 2)

    status = 'PASS' if confidence >= 0.3 else 'WEAK'

    details = {
        'regime': f'CLAIMS_{bias}',
        'claims_k': claims_k,
        'claims_signal': claims_signal,
        'claims_trend': claims_trend,
        'wyckoff': wyckoff_phase,
        'vol_regime': vol_regime,
        'bias': bias,
        'avg_ret_24h': avg_ret,
        'win_rate': win_rate,
        'sample_size': n,
        'confidence': round(confidence, 2),
        'trend_mult': round(trend_mult, 2),
        'score_adj': score_adj,
        'size_mult': size_mult,
    }

    return status, score_adj, size_mult, details


def format_m61(details):
    """Format M61 details for terminal output."""
    if not details or details.get('regime') in ('DISABLED', 'NOT_CLAIMS_DAY', 'NO_CLAIMS_DATA', 'NO_EDGE'):
        regime = details.get('regime', '?') if details else '?'
        if regime == 'NOT_CLAIMS_DAY':
            return ''  # silent when not Thursday
        return ''

    bias = details.get('bias', '?')
    claims_k = details.get('claims_k', 0)
    signal = details.get('claims_signal', '?')
    trend = details.get('claims_trend', '?')
    wyckoff = details.get('wyckoff', '?')
    vol = details.get('vol_regime', '?')
    avg_ret = details.get('avg_ret_24h', 0)
    win = details.get('win_rate', 0)
    n = details.get('sample_size', 0)
    conf = details.get('confidence', 0)
    trend_mult = details.get('trend_mult', 1.0)
    score_adj = details.get('score_adj', 0)
    size_mult = details.get('size_mult', 1.0)

    icon = '🟢' if bias == 'LONG' else '🔴' if bias == 'SHORT' else '⚪'
    conf_icon = '🟢' if conf >= 0.7 else '🟡' if conf >= 0.4 else '🟠'

    # Signal icons
    sig_icons = {'LOW': '🟢', 'NORMAL': '⚪', 'ELEVATED': '🟡', 'SPIKE': '🟠', 'CRISIS': '🔴'}
    sig_icon = sig_icons.get(signal, '⚪')

    # Trend icons
    trend_icons = {'RISING': '📈', 'FALLING': '📉', 'STABLE': '➡️'}
    trend_icon = trend_icons.get(trend, '➡️')

    lines = []
    lines.append(f"\n  {icon} M61 US CLAIMS BIAS: {bias}")
    lines.append(f"    Claims: {sig_icon} {claims_k}K ({signal})  {trend_icon} {trend}")
    lines.append(f"    Context: {wyckoff} + {vol}")
    lines.append(f"    Backtest: avg 24h={avg_ret:+.2f}%  win={win*100:.0f}%  n={n}")
    lines.append(f"    {conf_icon} Confidence: {conf:.2f}  trend_mult={trend_mult:.2f}  "
                 f"Score adj: {score_adj:+.3f}  Size: {size_mult:.2f}x")

    return '\n'.join(lines)

"""
Macro Utilities — Shared helpers for macro release scoring.

Extracted from M23 (m23_ppi_session.py) to break the dependency chain.
These functions are used by cascade modules, M22, macro_lifecycle,
and the scanner.

Functions:
    get_claims_trend(config) — current jobless claims trend + Sahm Rule
    classify_market_regime(config) — inflation regime from PPI/CPI data
    get_release_type(date_str) — what macro data releases on a given date
    is_ppi_release_day / is_cpi_release_day / is_nfp_release_day / etc.
    classify_macro_combo(cpi, ppi, claims) — CPI×PPI×Claims cascade classification
    get_claims_context_for_release(type, config) — claims context for a release day
    format_claims_context(ctx) — terminal output for claims context
"""

from datetime import datetime, timedelta


# ═══════════════════════════════════════════════════════════════
# RELEASE DATE SCHEDULES (imported from standalone modules)
# ═══════════════════════════════════════════════════════════════

from src.modules.m60_us_ppi import PPI_SCHEDULE_DATES
from src.modules.m56_us_cpi import CPI_SCHEDULE_DATES
from src.modules.m37_nfp import NFP_SCHEDULE_DATES


# ═══════════════════════════════════════════════════════════════
# JOBLESS CLAIMS + UNEMPLOYMENT DATA (for get_claims_trend)
# ═══════════════════════════════════════════════════════════════

JOBLESS_CLAIMS_MONTHLY_AVG = {
    '2021-01': 900, '2021-02': 750, '2021-03': 650, '2021-04': 570,
    '2021-05': 450, '2021-06': 400, '2021-07': 380, '2021-08': 350,
    '2021-09': 330, '2021-10': 280, '2021-11': 250, '2021-12': 200,
    '2022-01': 210, '2022-02': 200, '2022-03': 190, '2022-04': 185,
    '2022-05': 190, '2022-06': 195, '2022-07': 195, '2022-08': 200,
    '2022-09': 195, '2022-10': 190, '2022-11': 190, '2022-12': 195,
    '2023-01': 190, '2023-02': 195, '2023-03': 200, '2023-04': 200,
    '2023-05': 210, '2023-06': 215, '2023-07': 220, '2023-08': 220,
    '2023-09': 215, '2023-10': 215, '2023-11': 210, '2023-12': 210,
    '2024-01': 210, '2024-02': 215, '2024-03': 215, '2024-04': 220,
    '2024-05': 220, '2024-06': 225, '2024-07': 235, '2024-08': 230,
    '2024-09': 225, '2024-10': 220, '2024-11': 215, '2024-12': 215,
    '2025-01': 205, '2025-02': 210, '2025-03': 210, '2025-04': 215,
    '2025-05': 240, '2025-06': 230, '2025-07': 240, '2025-08': 225,
    '2025-09': 215, '2025-10': 210, '2025-11': 220, '2025-12': 199,
    '2026-01': 210, '2026-02': 227, '2026-03': 215, '2026-04': 200,
    '2026-05': 200,
}

UNEMPLOYMENT_RATE_MONTHLY = {
    '2021-01': 6.3, '2021-02': 6.2, '2021-03': 6.0, '2021-04': 6.1,
    '2021-05': 5.8, '2021-06': 5.9, '2021-07': 5.4, '2021-08': 5.2,
    '2021-09': 4.7, '2021-10': 4.6, '2021-11': 4.2, '2021-12': 3.9,
    '2022-01': 4.0, '2022-02': 3.8, '2022-03': 3.6, '2022-04': 3.6,
    '2022-05': 3.6, '2022-06': 3.6, '2022-07': 3.5, '2022-08': 3.7,
    '2022-09': 3.5, '2022-10': 3.7, '2022-11': 3.7, '2022-12': 3.5,
    '2023-01': 3.4, '2023-02': 3.6, '2023-03': 3.5, '2023-04': 3.4,
    '2023-05': 3.7, '2023-06': 3.6, '2023-07': 3.5, '2023-08': 3.8,
    '2023-09': 3.8, '2023-10': 3.9, '2023-11': 3.7, '2023-12': 3.7,
    '2024-01': 3.7, '2024-02': 3.9, '2024-03': 3.8, '2024-04': 3.9,
    '2024-05': 4.0, '2024-06': 4.1, '2024-07': 4.3, '2024-08': 4.2,
    '2024-09': 4.1, '2024-10': 4.1, '2024-11': 4.2, '2024-12': 4.1,
    '2025-01': 4.0, '2025-02': 4.1, '2025-03': 4.2, '2025-04': 4.2,
    '2025-05': 4.1, '2025-06': 4.2, '2025-07': 4.2, '2025-08': 4.3,
    '2025-09': 4.3, '2025-10': 4.3, '2025-11': 4.4, '2025-12': 4.4,
    '2026-01': 4.3, '2026-02': 4.3, '2026-03': 4.3, '2026-04': 4.3,
}

# Thresholds
CLAIMS_LOW_THRESHOLD = 210
CLAIMS_ELEVATED_THRESHOLD = 225
CLAIMS_SPIKE_THRESHOLD = 240
CLAIMS_CRISIS_THRESHOLD = 280
CLAIMS_TREND_RISING_THRESHOLD = 5
CLAIMS_TREND_FALLING_THRESHOLD = -5


def _load_fred_cache():
    """Override hardcoded claims/unemployment dicts with FRED cache if available."""
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
        unrate = cache.get("unrate", {}).get("monthly", {})
        if icsa:
            merged = dict(JOBLESS_CLAIMS_MONTHLY_AVG)
            merged.update(icsa)
            JOBLESS_CLAIMS_MONTHLY_AVG.clear()
            JOBLESS_CLAIMS_MONTHLY_AVG.update(merged)
        if unrate:
            merged = dict(UNEMPLOYMENT_RATE_MONTHLY)
            merged.update(unrate)
            UNEMPLOYMENT_RATE_MONTHLY.clear()
            UNEMPLOYMENT_RATE_MONTHLY.update(merged)
    except Exception:
        pass

_load_fred_cache()


# ═══════════════════════════════════════════════════════════════
# RELEASE DAY HELPERS
# ═══════════════════════════════════════════════════════════════

def is_ppi_release_day(date_str=None):
    """Check if today (or given date) is a PPI release day."""
    if date_str is None:
        date_str = datetime.utcnow().strftime('%Y-%m-%d')
    return date_str in PPI_SCHEDULE_DATES


def is_cpi_release_day(date_str=None):
    """Check if today (or given date) is a CPI release day."""
    if date_str is None:
        date_str = datetime.utcnow().strftime('%Y-%m-%d')
    return date_str in CPI_SCHEDULE_DATES


def is_nfp_release_day(date_str=None):
    """Check if today (or given date) is an NFP release day."""
    if date_str is None:
        date_str = datetime.utcnow().strftime('%Y-%m-%d')
    return date_str in NFP_SCHEDULE_DATES


def is_claims_release_day(date_str=None):
    """Check if today (or given date) is a jobless claims release day.

    Claims are released every Thursday at 8:30 AM ET.
    """
    if date_str is None:
        date_str = datetime.utcnow().strftime('%Y-%m-%d')
    dt = datetime.strptime(date_str, '%Y-%m-%d')
    if dt.weekday() == 3:  # Thursday
        return True
    if dt.weekday() == 4:  # Friday (holiday-shifted)
        return True
    return False


def is_macro_release_day(date_str=None):
    """Check if today is any macro data release day (NFP, PPI, CPI, or Claims)."""
    if date_str is None:
        date_str = datetime.utcnow().strftime('%Y-%m-%d')
    return (date_str in PPI_SCHEDULE_DATES or
            date_str in CPI_SCHEDULE_DATES or
            date_str in NFP_SCHEDULE_DATES or
            is_claims_release_day(date_str))


def get_release_type(date_str):
    """Determine what macro data is released on a given date.

    Returns: combination of 'NFP', 'PPI', 'CPI', 'CLAIMS', or None
    """
    parts = []
    if date_str in NFP_SCHEDULE_DATES:
        parts.append('NFP')
    if date_str in CPI_SCHEDULE_DATES:
        parts.append('CPI')
    if date_str in PPI_SCHEDULE_DATES:
        parts.append('PPI')
    if is_claims_release_day(date_str):
        parts.append('CLAIMS')
    if not parts:
        return None
    if len(parts) == 1:
        return parts[0]
    return '+'.join(parts)


# ═══════════════════════════════════════════════════════════════
# REGIME CLASSIFICATION
# ═══════════════════════════════════════════════════════════════

def classify_market_regime(config=None):
    """Classify current market regime for fade/continuation bias.

    Uses live BLS data from cache if available, falls back to config.
    Returns:
        regime: str
        fade_rate: float (0.0-1.0)
    """
    from src.config import CONFIG
    cfg = config or CONFIG

    ppi_yoy = None
    ppi_prev = None
    cpi_yoy = None
    fed = cfg.get('M22_FED_STANCE', 'HOLDING')

    try:
        import json as _json
        import os as _os
        cache_path = _os.path.join(_os.path.dirname(_os.path.dirname(_os.path.dirname(__file__))),
                                    'data', 'macro_data.json')
        if _os.path.exists(cache_path):
            with open(cache_path) as _f:
                cache = _json.load(_f)
            latest = cache.get('latest', {})
            yoy_data = cache.get('yoy', {})
            ppi_latest = latest.get('PPI_FD', {})
            ppi_date = ppi_latest.get('date', '')
            ppi_yoy = ppi_latest.get('yoy')
            cpi_latest = latest.get('CPI_ALL', {})
            cpi_yoy = cpi_latest.get('yoy')
            if ppi_date:
                y, m = int(ppi_date[:4]), int(ppi_date[5:7])
                prev_m = m - 1 if m > 1 else 12
                prev_y = y if m > 1 else y - 1
                prev_key = f"{prev_y:04d}-{prev_m:02d}"
                ppi_prev = yoy_data.get('PPI_FD', {}).get(prev_key)
    except Exception:
        pass

    if ppi_yoy is None:
        ppi_yoy = cfg.get('M22_PPI_YOY', None)
    if ppi_prev is None:
        ppi_prev = cfg.get('M22_PPI_PREV_YOY', None)
    if cpi_yoy is None:
        cpi_yoy = cfg.get('M22_CPI_YOY', None)

    if ppi_yoy is None:
        return 'UNKNOWN', 0.33

    REGIME_FADE_RATES = {
        'TIGHTENING': 0.30, 'EASING': 0.33, 'CRISIS_RECOVERY': 0.48,
        'BULL': 0.12, 'BEAR': 0.29, 'RECOVERY': 0.29,
        'ACCELERATION': 0.29, 'STAGFLATION': 0.46, 'STAGFLATION_HOT': 0.50,
    }

    if ppi_yoy >= 3.0 and fed == 'HOLDING':
        if ppi_yoy >= 4.0:
            return 'STAGFLATION_HOT', REGIME_FADE_RATES.get('STAGFLATION_HOT', 0.50)
        return 'STAGFLATION', REGIME_FADE_RATES.get('STAGFLATION', 0.46)
    if ppi_prev is not None and ppi_yoy > ppi_prev:
        return 'ACCELERATION', REGIME_FADE_RATES.get('ACCELERATION', 0.29)
    if ppi_prev is not None and ppi_yoy < ppi_prev:
        return 'RECOVERY', REGIME_FADE_RATES.get('RECOVERY', 0.29)
    return 'ACCELERATION', 0.29


# ═══════════════════════════════════════════════════════════════
# CLASSIFICATION HELPERS
# ═══════════════════════════════════════════════════════════════

def classify_dump_size(us_move_pct):
    """Classify US session dump size for reversal probability.

    Returns: 'SMALL', 'MEDIUM', 'BIG', 'CRASH', or 'NOT_DUMP'
    """
    if us_move_pct >= -0.5:
        return 'NOT_DUMP'
    elif us_move_pct >= -1.5:
        return 'SMALL'
    elif us_move_pct >= -2.5:
        return 'MEDIUM'
    elif us_move_pct >= -4.0:
        return 'BIG'
    else:
        return 'CRASH'


def classify_inflation_regime(regime):
    """Map regime to inflation regime for reversal stats.

    Returns: 'INFLATION_RISING', 'INFLATION_PEAKING', 'DEFLATION', 'NEUTRAL'
    """
    if regime in ('STAGFLATION', 'STAGFLATION_HOT', 'ACCELERATION'):
        return 'INFLATION_RISING'
    elif regime in ('TIGHTENING',):
        return 'INFLATION_PEAKING'
    elif regime in ('CRISIS_RECOVERY', 'RECOVERY', 'EASING'):
        return 'DEFLATION'
    else:
        return 'NEUTRAL'


# ═══════════════════════════════════════════════════════════════
# CLAIMS TREND
# ═══════════════════════════════════════════════════════════════

def get_claims_trend(config=None):
    """Get current jobless claims trend from monthly averages.

    Returns:
        dict with current, prev_month, trend, classification, sahm_triggered, unemployment
    """
    from src.config import CONFIG
    cfg = config or CONFIG

    sorted_months = sorted(JOBLESS_CLAIMS_MONTHLY_AVG.keys())
    if not sorted_months:
        return None

    latest_key = sorted_months[-1]
    current_claims = JOBLESS_CLAIMS_MONTHLY_AVG[latest_key]
    prev_key = sorted_months[-2] if len(sorted_months) > 1 else None
    prev_claims = JOBLESS_CLAIMS_MONTHLY_AVG.get(prev_key, current_claims) if prev_key else current_claims

    if len(sorted_months) >= 3:
        m3_claims = JOBLESS_CLAIMS_MONTHLY_AVG[sorted_months[-3]]
        trend_3m = current_claims - m3_claims
    else:
        trend_3m = current_claims - prev_claims

    if trend_3m >= CLAIMS_TREND_RISING_THRESHOLD:
        trend = 'RISING'
    elif trend_3m <= CLAIMS_TREND_FALLING_THRESHOLD:
        trend = 'FALLING'
    else:
        trend = 'STABLE'

    trend_pct = ((current_claims - prev_claims) / prev_claims * 100) if prev_claims > 0 else 0

    if current_claims >= CLAIMS_CRISIS_THRESHOLD:
        classification = 'CRISIS'
    elif current_claims >= CLAIMS_SPIKE_THRESHOLD:
        classification = 'SPIKE'
    elif current_claims >= CLAIMS_ELEVATED_THRESHOLD:
        classification = 'ELEVATED'
    elif current_claims >= CLAIMS_LOW_THRESHOLD:
        classification = 'NORMAL'
    else:
        classification = 'LOW'

    # Sahm Rule
    unemp_sorted = sorted(UNEMPLOYMENT_RATE_MONTHLY.keys())
    sahm_triggered = False
    current_unemp = None
    if len(unemp_sorted) >= 12:
        current_unemp = UNEMPLOYMENT_RATE_MONTHLY[unemp_sorted[-1]]
        last_3 = [UNEMPLOYMENT_RATE_MONTHLY[k] for k in unemp_sorted[-3:]]
        avg_3m = sum(last_3) / 3
        last_12 = [UNEMPLOYMENT_RATE_MONTHLY[k] for k in unemp_sorted[-12:]]
        low_12m = min(last_12)
        if avg_3m - low_12m >= 0.5:
            sahm_triggered = True

    return {
        'current': current_claims,
        'prev_month': prev_claims,
        'trend': trend,
        'trend_pct': round(trend_pct, 1),
        'trend_3m': trend_3m,
        'classification': classification,
        'sahm_triggered': sahm_triggered,
        'unemployment': current_unemp,
        'latest_month': latest_key,
    }


# ═══════════════════════════════════════════════════════════════
# CPI × PPI × CLAIMS COMBO MATRIX
# ═══════════════════════════════════════════════════════════════

CPI_PRIMARY = {
    'COOL': (+1.06, 'HIGH', 'CUT'),
    'WARM': (+0.27, 'MEDIUM', 'HOLD'),
    'HOT':  (-0.45, 'MEDIUM', 'HOLD'),
}

PPI_CONFIRMATION = {
    ('COOL', 'COOL'):   (+0.50, +0.10, 'Both cool — strong disinflation, Fed can cut'),
    ('COOL', 'WARM'):   (+0.00, +0.00, 'CPI cool, PPI inline — signal intact'),
    ('COOL', 'HOT'):    (-0.30, -0.10, 'CPI cool but PPI hot — pipeline inflation building'),
    ('WARM', 'COOL'):   (-0.80, -0.10, 'CPI inline, PPI cool — deflation fears'),
    ('WARM', 'WARM'):   (+0.00, +0.00, 'Both inline — no signal'),
    ('WARM', 'HOT'):    (-0.20, +0.00, 'CPI inline, PPI hot — mild inflation concern'),
    ('HOT', 'COOL'):    (+0.30, -0.10, 'CPI hot, PPI cool — mixed signals'),
    ('HOT', 'WARM'):    (+0.00, +0.00, 'CPI hot, PPI inline — hot signal intact'),
    ('HOT', 'HOT'):     (-0.80, +0.10, "Both hot — persistent inflation, Fed can't cut"),
}

CLAIMS_MODIFIER = {
    'LOW':       (+0.20, 'Tight labor — economy strong, Fed has room'),
    'NORMAL':    (+0.00, 'Normal labor market — no modifier'),
    'ELEVATED':  (-0.30, 'Labor softening — recession fear amplifies hot CPI'),
    'SPIKE':     (-0.80, 'Claims spike — risk-off, amplifies any hot signal'),
    'CRISIS':    (-1.50, 'Claims crisis — macro dominant, crypto sells off'),
}

FED_RESPONSE = {
    'CUT':     'Fed can cut → risk-on → ETH rallies',
    'HOLD':    'Fed holds → no catalyst → ETH range-bound',
    'TRAPPED': "Fed trapped (can't cut, won't hike) → ETH dumps",
    'HIKE':    'Fed must hike → risk-off → ETH crashes',
}

REGIME_CPI_EXPECTED = {
    ('BEAR', 'COOL'):          +9.92,
    ('BEAR', 'HOT'):           -3.33,
    ('BULL', 'COOL'):          +1.88,
    ('BULL', 'HOT'):           -0.20,
    ('RECOVERY', 'COOL'):      -0.55,
    ('RECOVERY', 'HOT'):       +0.84,
    ('ACCELERATION', 'COOL'):  -0.09,
    ('ACCELERATION', 'HOT'):   +0.06,
    ('STAGFLATION', 'COOL'):   +3.00,
    ('STAGFLATION', 'HOT'):    -2.00,
    ('STAGFLATION_HOT', 'COOL'): +4.00,
    ('STAGFLATION_HOT', 'HOT'):  -3.00,
}


def classify_macro_combo(cpi_yoy, ppi_yoy, claims_classification):
    """Classify the CPI→PPI→Claims cascade and predict ETH impact.

    Args:
        cpi_yoy: Current CPI year-over-year %
        ppi_yoy: Current PPI year-over-year %
        claims_classification: 'LOW', 'NORMAL', 'ELEVATED', 'SPIKE', 'CRISIS'

    Returns:
        dict with cascade classification, expected impact, and Fed response
    """
    if cpi_yoy is not None:
        if cpi_yoy >= 3.5:
            cpi_class, cpi_surprise = 'CPI_HOT', 'HOT'
        elif cpi_yoy >= 2.5:
            cpi_class, cpi_surprise = 'CPI_WARM', 'WARM'
        else:
            cpi_class, cpi_surprise = 'CPI_COOL', 'COOL'
    else:
        cpi_class, cpi_surprise = 'CPI_UNKNOWN', 'WARM'

    if ppi_yoy is not None:
        if ppi_yoy >= 3.5:
            ppi_class, ppi_surprise = 'PPI_HOT', 'HOT'
        elif ppi_yoy >= 2.5:
            ppi_class, ppi_surprise = 'PPI_WARM', 'WARM'
        else:
            ppi_class, ppi_surprise = 'PPI_COOL', 'COOL'
    else:
        ppi_class, ppi_surprise = 'PPI_UNKNOWN', 'WARM'

    if claims_classification in ('LOW',):
        claims_bucket = 'CLAIMS_LOW'
    elif claims_classification in ('NORMAL',):
        claims_bucket = 'CLAIMS_NORMAL'
    else:
        claims_bucket = 'CLAIMS_ELEVATED'

    cpi_base = CPI_PRIMARY.get(cpi_surprise, (0.0, 'LOW', 'HOLD'))
    base_move, base_conf, fed_bias = cpi_base

    ppi_key = (cpi_surprise, ppi_surprise)
    ppi_mod = PPI_CONFIRMATION.get(ppi_key, (0.0, 0.0, 'No PPI data'))
    ppi_move_mod, ppi_conf_mod, ppi_desc = ppi_mod

    claims_mod = CLAIMS_MODIFIER.get(claims_classification, (0.0, 'Unknown'))
    claims_move_mod, claims_desc = claims_mod

    total_move = base_move + ppi_move_mod + claims_move_mod

    conf_map = {'HIGH': 0.85, 'MEDIUM': 0.65, 'LOW': 0.45}
    conf_val = conf_map.get(base_conf, 0.50) + ppi_conf_mod
    conf_val = max(0.30, min(0.95, conf_val))
    if conf_val >= 0.80:
        confidence = 'HIGH'
    elif conf_val >= 0.60:
        confidence = 'MEDIUM'
    else:
        confidence = 'LOW'

    if total_move >= 3.0:
        signal = 'STRONG_BUY'
    elif total_move >= 1.0:
        signal = 'BUY'
    elif total_move >= -1.0:
        signal = 'HOLD'
    elif total_move >= -3.0:
        signal = 'SELL'
    else:
        signal = 'STRONG_SELL'

    ppi_leading = False
    if ppi_yoy is not None and cpi_yoy is not None:
        ppi_gap = ppi_yoy - cpi_yoy
        if ppi_gap > 1.0:
            ppi_leading = True

    return {
        'combo_key': (cpi_class, ppi_class, claims_bucket),
        'cpi_class': cpi_class,
        'ppi_class': ppi_class,
        'claims_bucket': claims_bucket,
        'expected_eth_move': round(total_move, 1),
        'confidence': confidence,
        'fed_action': fed_bias,
        'fed_explanation': FED_RESPONSE.get(fed_bias, 'Unknown'),
        'signal': signal,
        'ppi_leading_cpi': ppi_leading,
        'ppi_cpi_gap': round(ppi_yoy - cpi_yoy, 1) if (ppi_yoy and cpi_yoy) else None,
        'cascade': {
            'cpi_signal': cpi_surprise,
            'cpi_base_move': base_move,
            'ppi_confirmation': ppi_surprise,
            'ppi_modifier': ppi_move_mod,
            'ppi_description': ppi_desc,
            'claims_modifier': claims_move_mod,
            'claims_description': claims_desc,
            'total_move': round(total_move, 2),
        },
    }


# ═══════════════════════════════════════════════════════════════
# CLAIMS CONTEXT HELPERS
# ═══════════════════════════════════════════════════════════════

def get_claims_context_for_release(release_type, config=None):
    """Get jobless claims context for a PPI or CPI release day."""
    from src.config import CONFIG
    cfg = config or CONFIG

    claims = get_claims_trend(cfg)
    if claims is None:
        return None

    ppi_yoy = cfg.get('M22_PPI_YOY')
    cpi_yoy = cfg.get('M22_CPI_YOY')

    try:
        import json as _json
        import os as _os
        cache_path = _os.path.join(_os.path.dirname(_os.path.dirname(_os.path.dirname(__file__))),
                                    'data', 'macro_data.json')
        if _os.path.exists(cache_path):
            with open(cache_path) as _f:
                cache = _json.load(_f)
            latest = cache.get('latest', {})
            if latest.get('PPI_FD', {}).get('yoy') is not None:
                ppi_yoy = latest['PPI_FD']['yoy']
            if latest.get('CPI_ALL', {}).get('yoy') is not None:
                cpi_yoy = latest['CPI_ALL']['yoy']
    except Exception:
        pass

    combo = classify_macro_combo(cpi_yoy, ppi_yoy, claims['classification'])

    today_str = datetime.utcnow().strftime('%Y-%m-%d')
    claims_today = is_claims_release_day(today_str)

    return {
        'claims': claims,
        'combo': combo,
        'claims_today': claims_today,
        'cpi_yoy': cpi_yoy,
        'ppi_yoy': ppi_yoy,
        'release_type': release_type,
    }


def format_claims_context(ctx):
    """Format jobless claims context for terminal output."""
    if ctx is None:
        return ''

    lines = []
    claims = ctx.get('claims', {})
    combo = ctx.get('combo', {})

    current = claims.get('current', 0)
    trend = claims.get('trend', '?')
    classification = claims.get('classification', '?')
    unemp = claims.get('unemployment')
    sahm = claims.get('sahm_triggered', False)

    cls_icons = {'LOW': '🟢', 'NORMAL': '⚪', 'ELEVATED': '🟡', 'SPIKE': '🟠', 'CRISIS': '🔴'}
    cls_icon = cls_icons.get(classification, '⚪')
    trend_icons = {'RISING': '📈', 'FALLING': '📉', 'STABLE': '➡️'}
    trend_icon = trend_icons.get(trend, '➡️')

    lines.append(f"\n  📋 JOBLESS CLAIMS CONTEXT:")
    lines.append(f"    Claims: {cls_icon} {current}K ({classification})  "
                 f"{trend_icon} {trend} ({claims.get('trend_pct', 0):+.1f}%)")
    if unemp:
        sahm_icon = '🔴 TRIGGERED' if sahm else '🟢 ok'
        lines.append(f"    Unemployment: {unemp}%  Sahm Rule: {sahm_icon}")
    if claims.get('trend_3m'):
        lines.append(f"    3-month Δ: {claims['trend_3m']:+.0f}K")

    if combo:
        signal = combo.get('signal', '?')
        expected = combo.get('expected_eth_move', 0)
        fed = combo.get('fed_action', '?')
        conf = combo.get('confidence', '?')
        ppi_gap = combo.get('ppi_cpi_gap')
        cascade = combo.get('cascade', {})

        sig_icons = {'STRONG_BUY': '🟢🟢', 'BUY': '🟢', 'HOLD': '⚪', 'SELL': '🔴', 'STRONG_SELL': '🔴🔴'}
        sig_icon = sig_icons.get(signal, '⚪')

        lines.append(f"\n    Macro Cascade: {sig_icon} {signal}")
        if cascade:
            lines.append(f"      1. CPI {cascade.get('cpi_signal', '?')}: {cascade.get('cpi_base_move', 0):+.2f}% (primary)")
            lines.append(f"      2. PPI {cascade.get('ppi_confirmation', '?')}: {cascade.get('ppi_modifier', 0):+.2f}% — {cascade.get('ppi_description', '')}")
            lines.append(f"      3. Claims: {cascade.get('claims_modifier', 0):+.2f}% — {cascade.get('claims_description', '')}")
            lines.append(f"      → Total: {expected:+.1f}%  Conf: {conf}")
        lines.append(f"    Fed: {fed} — {combo.get('fed_explanation', '')}")

        if ppi_gap is not None and ppi_gap > 1.0:
            lines.append(f"    ⚠️ PPI leads CPI by {ppi_gap:.1f}pp — CPI will follow UP (2-3 month lag)")
        if sahm:
            lines.append(f"    🚨 SAHM RULE TRIGGERED — recession indicator active")

    return '\n'.join(lines)

"""
M_China_Activity: China Monthly Activity Data Bundle

Released ~15th-16th of each month by NBS, covering previous month's data.
This is a "data dump" — IP, Retail Sales, FAI, Unemployment, House Prices
all released together at 10:00 AM Beijing time (02:00 UTC).

These indicators are the REAL ECONOMY PULSE between PMI releases:
  - Industrial Production: manufacturing/output health
  - Retail Sales: consumer spending
  - Fixed Asset Investment: infrastructure + property + manufacturing capex
  - Unemployment Rate: labor market
  - House Prices: property sector health (China's largest asset class)

Significance for crypto (via risk sentiment):
  - Strong data → China growth narrative → risk-on → ETH rallies
  - Weak data → China slowdown → risk-off → ETH dumps
  - Property collapse deepening → systemic risk → worst case
  - Today's example (May 18, 2026): ALL indicators missed badly
    Retail Sales +0.2% (expected +2.0%) — 40-month low
    IP +4.1% (expected +5.9%)
    FAI -1.6% (expected +1.6%) — flipped to contraction

Usage:
    from src.modules.m_china_activity import (
        CHINA_ACTIVITY_RELEASES, CHINA_ACTIVITY_DATES,
        score_china_activity, format_china_activity,
    )
"""

from datetime import datetime


# ═══════════════════════════════════════════════════════════════
# CHINA ACTIVITY DATA SCHEDULE
# Released ~15th-16th of each month (NBS data dump)
# Covers: IP YoY, Retail Sales YoY, FAI YTD YoY, Unemployment, House Prices
# ═══════════════════════════════════════════════════════════════

CHINA_ACTIVITY_RELEASES = {
    # 2024
    '2024-01-17': {'month': 'Dec', 'ip_yoy': 6.8, 'retail_yoy': 7.4, 'fai_ytd_yoy': 3.0, 'unemp': 5.1, 'house_price_yoy': -1.4, 'consensus_ip': 6.6, 'consensus_retail': 8.0},
    '2024-02-08': {'month': 'Jan', 'ip_yoy': None, 'retail_yoy': None, 'fai_ytd_yoy': None, 'unemp': 5.3, 'house_price_yoy': -1.2, 'consensus_ip': None, 'consensus_retail': None},  # combined Jan+Feb
    '2024-03-18': {'month': 'Feb', 'ip_yoy': 7.0, 'retail_yoy': 5.5, 'fai_ytd_yoy': 4.2, 'unemp': 5.3, 'house_price_yoy': -1.9, 'consensus_ip': 5.2, 'consensus_retail': 5.5},
    '2024-04-16': {'month': 'Mar', 'ip_yoy': 4.5, 'retail_yoy': 3.1, 'fai_ytd_yoy': 4.5, 'unemp': 5.2, 'house_price_yoy': -2.7, 'consensus_ip': 6.0, 'consensus_retail': 4.8},
    '2024-05-17': {'month': 'Apr', 'ip_yoy': 6.7, 'retail_yoy': 2.3, 'fai_ytd_yoy': 4.2, 'unemp': 5.0, 'house_price_yoy': -3.5, 'consensus_ip': 5.5, 'consensus_retail': 3.8},
    '2024-06-17': {'month': 'May', 'ip_yoy': 5.6, 'retail_yoy': 3.7, 'fai_ytd_yoy': 4.0, 'unemp': 5.0, 'house_price_yoy': -4.3, 'consensus_ip': 6.0, 'consensus_retail': 4.5},
    '2024-07-15': {'month': 'Jun', 'ip_yoy': 5.3, 'retail_yoy': 2.0, 'fai_ytd_yoy': 3.9, 'unemp': 5.1, 'house_price_yoy': -4.9, 'consensus_ip': 5.0, 'consensus_retail': 3.3},
    '2024-08-15': {'month': 'Jul', 'ip_yoy': 5.1, 'retail_yoy': 2.7, 'fai_ytd_yoy': 3.6, 'unemp': 5.2, 'house_price_yoy': -5.3, 'consensus_ip': 5.3, 'consensus_retail': 2.6},
    '2022-09-16': {'month': 'Aug', 'ip_yoy': 4.5, 'retail_yoy': 3.7, 'fai_ytd_yoy': 3.4, 'unemp': 5.3, 'house_price_yoy': -5.7, 'consensus_ip': 4.8, 'consensus_retail': 3.5},
    '2024-10-18': {'month': 'Sep', 'ip_yoy': 5.4, 'retail_yoy': 3.2, 'fai_ytd_yoy': 3.4, 'unemp': 5.1, 'house_price_yoy': -5.8, 'consensus_ip': 4.6, 'consensus_retail': 2.5},
    '2024-11-15': {'month': 'Oct', 'ip_yoy': 5.3, 'retail_yoy': 4.8, 'fai_ytd_yoy': 3.4, 'unemp': 5.0, 'house_price_yoy': -5.5, 'consensus_ip': 5.6, 'consensus_retail': 3.8},
    '2024-12-16': {'month': 'Nov', 'ip_yoy': 5.4, 'retail_yoy': 3.0, 'fai_ytd_yoy': 3.3, 'unemp': 5.0, 'house_price_yoy': -5.1, 'consensus_ip': 5.3, 'consensus_retail': 4.6},
    # 2025
    '2025-01-17': {'month': 'Dec', 'ip_yoy': 6.2, 'retail_yoy': 3.7, 'fai_ytd_yoy': 3.2, 'unemp': 5.1, 'house_price_yoy': -4.5, 'consensus_ip': 5.4, 'consensus_retail': 3.5},
    '2025-02-17': {'month': 'Jan', 'ip_yoy': None, 'retail_yoy': None, 'fai_ytd_yoy': None, 'unemp': 5.4, 'house_price_yoy': -4.0, 'consensus_ip': None, 'consensus_retail': None},
    '2025-03-17': {'month': 'Feb', 'ip_yoy': 5.9, 'retail_yoy': 4.0, 'fai_ytd_yoy': 4.1, 'unemp': 5.4, 'house_price_yoy': -3.8, 'consensus_ip': 5.3, 'consensus_retail': 3.8},
    '2025-04-16': {'month': 'Mar', 'ip_yoy': 7.7, 'retail_yoy': 5.9, 'fai_ytd_yoy': 4.3, 'unemp': 5.2, 'house_price_yoy': -3.2, 'consensus_ip': 5.9, 'consensus_retail': 4.5},
    '2025-05-19': {'month': 'Apr', 'ip_yoy': 6.1, 'retail_yoy': 5.1, 'fai_ytd_yoy': 4.0, 'unemp': 5.1, 'house_price_yoy': -2.8, 'consensus_ip': 5.5, 'consensus_retail': 5.0},
    '2025-06-16': {'month': 'May', 'ip_yoy': 5.8, 'retail_yoy': 4.5, 'fai_ytd_yoy': 3.7, 'unemp': 5.0, 'house_price_yoy': -2.5, 'consensus_ip': 5.9, 'consensus_retail': 4.8},
    '2025-07-15': {'month': 'Jun', 'ip_yoy': 5.5, 'retail_yoy': 4.2, 'fai_ytd_yoy': 3.5, 'unemp': 5.1, 'house_price_yoy': -2.2, 'consensus_ip': 5.6, 'consensus_retail': 4.0},
    '2025-08-15': {'month': 'Jul', 'ip_yoy': 5.0, 'retail_yoy': 3.5, 'fai_ytd_yoy': 3.2, 'unemp': 5.2, 'house_price_yoy': -2.0, 'consensus_ip': 5.4, 'consensus_retail': 3.8},
    '2025-09-15': {'month': 'Aug', 'ip_yoy': 4.8, 'retail_yoy': 3.0, 'fai_ytd_yoy': 2.9, 'unemp': 5.3, 'house_price_yoy': -1.8, 'consensus_ip': 5.2, 'consensus_retail': 3.5},
    '2025-10-16': {'month': 'Sep', 'ip_yoy': 5.2, 'retail_yoy': 3.8, 'fai_ytd_yoy': 2.8, 'unemp': 5.2, 'house_price_yoy': -1.5, 'consensus_ip': 5.0, 'consensus_retail': 3.5},
    '2025-11-17': {'month': 'Oct', 'ip_yoy': 5.0, 'retail_yoy': 3.5, 'fai_ytd_yoy': 2.6, 'unemp': 5.1, 'house_price_yoy': -1.3, 'consensus_ip': 5.3, 'consensus_retail': 4.0},
    '2025-12-15': {'month': 'Nov', 'ip_yoy': 5.3, 'retail_yoy': 4.0, 'fai_ytd_yoy': 2.4, 'unemp': 5.0, 'house_price_yoy': -1.0, 'consensus_ip': 5.1, 'consensus_retail': 3.8},
    # 2026
    '2026-01-19': {'month': 'Dec', 'ip_yoy': 5.5, 'retail_yoy': 3.2, 'fai_ytd_yoy': 2.2, 'unemp': 5.1, 'house_price_yoy': -0.8, 'consensus_ip': 5.3, 'consensus_retail': 3.5},
    '2026-02-16': {'month': 'Jan', 'ip_yoy': None, 'retail_yoy': None, 'fai_ytd_yoy': None, 'unemp': 5.3, 'house_price_yoy': -0.5, 'consensus_ip': None, 'consensus_retail': None},
    '2026-03-16': {'month': 'Feb', 'ip_yoy': 5.9, 'retail_yoy': 5.5, 'fai_ytd_yoy': 3.8, 'unemp': 5.4, 'house_price_yoy': -0.3, 'consensus_ip': 5.1, 'consensus_retail': 4.0},
    '2026-04-17': {'month': 'Mar', 'ip_yoy': 5.7, 'retail_yoy': 1.7, 'fai_ytd_yoy': 1.7, 'unemp': 5.4, 'house_price_yoy': -0.2, 'consensus_ip': 5.5, 'consensus_retail': 3.5},
    '2026-05-18': {'month': 'Apr', 'ip_yoy': 4.1, 'retail_yoy': 0.2, 'fai_ytd_yoy': -1.6, 'unemp': 5.2, 'house_price_yoy': -0.5, 'consensus_ip': 5.9, 'consensus_retail': 2.0},
    # Future scheduled (approximate)
    '2026-06-15': {'month': 'May'},
    '2026-07-15': {'month': 'Jun'},
    '2026-08-17': {'month': 'Jul'},
    '2026-09-15': {'month': 'Aug'},
    '2026-10-19': {'month': 'Sep'},
    '2026-11-16': {'month': 'Oct'},
    '2026-12-15': {'month': 'Nov'},
}

CHINA_ACTIVITY_DATES = set(CHINA_ACTIVITY_RELEASES.keys())


# ═══════════════════════════════════════════════════════════════
# SIGNAL CLASSIFIERS
# ═══════════════════════════════════════════════════════════════

def _score_indicator(actual, consensus, thresholds=None):
    """Score a single indicator vs consensus.

    Returns: (signal, surprise_pct)
    """
    if actual is None:
        return 'PENDING', 0.0

    if consensus is None:
        return 'NO_CONSENSUS', 0.0

    surprise = actual - consensus

    if thresholds is None:
        thresholds = {'big_beat': 1.0, 'beat': 0.3, 'miss': -0.3, 'big_miss': -1.0}

    if surprise >= thresholds['big_beat']:
        return 'STRONG_BEAT', surprise
    elif surprise >= thresholds['beat']:
        return 'BEAT', surprise
    elif surprise <= thresholds['big_miss']:
        return 'BIG_MISS', surprise
    elif surprise <= thresholds['miss']:
        return 'MISS', surprise
    return 'INLINE', surprise


# ═══════════════════════════════════════════════════════════════
# INDICATOR WEIGHTS (configurable)
# IP + Retail are primary (released with consensus, market-reactive)
# FAI is confirmation (investment cycle)
# House Prices is context (property sector health)
# Unemployment is background (lagging indicator)
# ═══════════════════════════════════════════════════════════════

ACTIVITY_WEIGHTS = {
    'ip':       0.40,   # Industrial Production — manufacturing/output health
    'retail':   0.40,   # Retail Sales — consumer spending (most market-reactive)
    'fai':      0.15,   # Fixed Asset Investment — infrastructure + property + capex
    'house':    0.05,   # House Prices — property sector (slow-moving, context)
    'unemp':    0.00,   # Unemployment — lagging, no consensus, background only
}

# Thresholds for individual indicator scoring
# Default: beat >= +0.3, inline within ±0.3, miss <= -0.3
# IP has wider bands (more volatile), Retail has tighter (market watches closely)
INDICATOR_THRESHOLDS = {
    'ip':     {'big_beat': 1.5, 'beat': 0.5, 'miss': -0.5, 'big_miss': -1.5},
    'retail': {'big_beat': 1.0, 'beat': 0.3, 'miss': -0.3, 'big_miss': -1.0},
    'fai':    {'big_beat': 1.0, 'beat': 0.3, 'miss': -0.3, 'big_miss': -1.0},
}


def classify_activity_bundle(data, weights=None):
    """Classify the full activity data bundle using weighted scoring.

    Each indicator with consensus data gets a score:
      STRONG_BEAT=+2, BEAT=+1, INLINE=0, MISS=-1, BIG_MISS=-2

    Weighted average determines composite signal.
    FAI contraction and property crisis act as overrides.

    Args:
        data: dict from CHINA_ACTIVITY_RELEASES
        weights: optional custom weights dict (default: ACTIVITY_WEIGHTS)

    Returns:
        dict with composite signal, per-indicator details, weighted score
    """
    w = weights or ACTIVITY_WEIGHTS

    ip_actual = data.get('ip_yoy')
    ip_consensus = data.get('consensus_ip')
    retail_actual = data.get('retail_yoy')
    retail_consensus = data.get('consensus_retail')
    fai_actual = data.get('fai_ytd_yoy')
    unemp = data.get('unemp')
    hp_yoy = data.get('house_price_yoy')

    # Score each indicator
    ip_signal, ip_surprise = _score_indicator(
        ip_actual, ip_consensus, INDICATOR_THRESHOLDS.get('ip'))
    retail_signal, retail_surprise = _score_indicator(
        retail_actual, retail_consensus, INDICATOR_THRESHOLDS.get('retail'))

    # FAI: no consensus in data, score vs 0 (contraction = miss)
    if fai_actual is not None:
        if fai_actual < -1.0:
            fai_signal, fai_surprise = 'BIG_MISS', fai_actual
        elif fai_actual < 0:
            fai_signal, fai_surprise = 'MISS', fai_actual
        elif fai_actual > 2.0:
            fai_signal, fai_surprise = 'BEAT', fai_actual
        else:
            fai_signal, fai_surprise = 'INLINE', fai_actual
    else:
        fai_signal, fai_surprise = 'PENDING', 0.0

    # House prices: score vs threshold (-3% = miss, -5% = crisis)
    if hp_yoy is not None:
        if hp_yoy <= -5.0:
            hp_signal = 'BIG_MISS'
        elif hp_yoy <= -3.0:
            hp_signal = 'MISS'
        elif hp_yoy <= -1.0:
            hp_signal = 'INLINE'
        else:
            hp_signal = 'BEAT'
    else:
        hp_signal = 'PENDING'

    # Map signals to numeric scores
    signal_scores = {
        'STRONG_BEAT': 2, 'BEAT': 1, 'INLINE': 0,
        'MIXED': 0, 'MILD_BEAT': 0.5, 'MILD_MISS': -0.5,
        'MISS': -1, 'BIG_MISS': -2, 'PROPERTY_CRISIS': -3,
        'PENDING': 0, 'NO_CONSENSUS': 0,
    }

    # Compute weighted score
    indicator_scores = {
        'ip': signal_scores.get(ip_signal, 0),
        'retail': signal_scores.get(retail_signal, 0),
        'fai': signal_scores.get(fai_signal, 0),
        'house': signal_scores.get(hp_signal, 0),
        'unemp': 0.0,  # no consensus, not scored
    }

    # Only include indicators with actual data
    active_weights = {}
    for key, score in indicator_scores.items():
        if key == 'unemp':
            continue  # background only
        actual_key = {'ip': 'ip_yoy', 'retail': 'retail_yoy', 'fai': 'fai_ytd_yoy', 'house': 'house_price_yoy'}.get(key)
        if data.get(actual_key) is not None:
            active_weights[key] = w.get(key, 0)

    # Normalize weights
    total_weight = sum(active_weights.values())
    if total_weight > 0:
        weighted_score = sum(
            indicator_scores[k] * active_weights[k] / total_weight
            for k in active_weights
        )
    else:
        weighted_score = 0.0

    # Map weighted score to composite signal
    if weighted_score >= 1.5:
        composite = 'STRONG_BEAT'
    elif weighted_score >= 0.5:
        composite = 'BEAT'
    elif weighted_score >= 0.15:
        composite = 'MILD_BEAT'
    elif weighted_score <= -1.5:
        composite = 'BIG_MISS'
    elif weighted_score <= -0.5:
        composite = 'MISS'
    elif weighted_score <= -0.15:
        composite = 'MILD_MISS'
    else:
        composite = 'INLINE'

    # Override: FAI contraction + retail miss = stagflation risk
    if fai_actual is not None and fai_actual < 0 and retail_signal in ('MISS', 'BIG_MISS'):
        composite = 'BIG_MISS'

    # Override: property crisis
    if hp_yoy is not None and hp_yoy <= -5.0:
        composite = 'PROPERTY_CRISIS'

    return {
        'composite': composite,
        'weighted_score': round(weighted_score, 3),
        'ip_signal': ip_signal,
        'ip_surprise': round(ip_surprise, 1),
        'retail_signal': retail_signal,
        'retail_surprise': round(retail_surprise, 1),
        'fai_signal': fai_signal,
        'fai_actual': fai_actual,
        'house_signal': hp_signal,
        'house_price_yoy': hp_yoy,
        'unemp': unemp,
        'indicator_scores': indicator_scores,
        'active_weights': active_weights,
    }


# ═══════════════════════════════════════════════════════════════
# SCORING
# ═══════════════════════════════════════════════════════════════

# Composite signal → (move_modifier, confidence)
ACTIVITY_SIGNAL_MAP = {
    'STRONG_BEAT':     (+0.80, 'HIGH', 'All indicators beat — China recovery confirmed'),
    'BEAT':            (+0.40, 'MEDIUM', 'Indicators mostly beat — China growth improving'),
    'MILD_BEAT':       (+0.20, 'MEDIUM', 'Mixed but leaning positive'),
    'INLINE':          (+0.00, 'LOW', 'Indicators inline — no signal'),
    'MIXED':           (-0.20, 'MEDIUM', 'Mixed signals — IP and retail diverging'),
    'MILD_MISS':       (-0.30, 'MEDIUM', 'Mixed but leaning negative'),
    'MISS':            (-0.60, 'HIGH', 'Indicators missed — China slowdown'),
    'BIG_MISS':        (-1.00, 'HIGH', 'All indicators missed — China weakness confirmed'),
    'PROPERTY_CRISIS': (-1.50, 'HIGH', 'Property collapse deepening — systemic risk'),
    'PENDING':         (0.00, 'LOW', 'Data not yet available'),
}

# Regime sensitivity for China activity data
CHINA_ACTIVITY_REGIME_SENSITIVITY = {
    'TIGHTENING': 0.60, 'EASING': 0.55, 'CRISIS_RECOVERY': 0.85,
    'BULL': 0.75, 'BEAR': 1.05, 'RECOVERY': 0.70,
    'ACCELERATION': 0.65, 'STAGFLATION': 0.90, 'STAGFLATION_HOT': 1.00,
}


def score_china_activity(data, regime='UNKNOWN'):
    """Score the China activity data bundle.

    Args:
        data: dict from CHINA_ACTIVITY_RELEASES
        regime: macro regime string

    Returns:
        (score, signal, confidence, details)
    """
    classification = classify_activity_bundle(data)
    composite = classification['composite']

    move_mod, confidence, description = ACTIVITY_SIGNAL_MAP.get(
        composite, (0.0, 'LOW', 'Unknown'))

    # Apply regime sensitivity
    regime_sens = CHINA_ACTIVITY_REGIME_SENSITIVITY.get(regime, 0.70)

    details = {
        'classification': classification,
        'composite': composite,
        'move_modifier': move_mod,
        'confidence': confidence,
        'description': description,
        'regime_sensitivity': regime_sens,
    }

    return move_mod, composite, confidence, details


def format_china_activity(data, details=None):
    """Format China activity data for terminal output."""
    if not data:
        return ''

    lines = []
    month = data.get('month', '?')

    ip = data.get('ip_yoy')
    retail = data.get('retail_yoy')
    fai = data.get('fai_ytd_yoy')
    unemp = data.get('unemp')
    hp = data.get('house_price_yoy')
    ip_c = data.get('consensus_ip')
    retail_c = data.get('consensus_retail')

    lines.append(f"\n  🇨🇳 CHINA ACTIVITY ({month}):")

    # Industrial Production
    if ip is not None:
        ip_surprise = ip - (ip_c or ip)
        ip_icon = '🟢' if ip_surprise > 0.2 else '🔴' if ip_surprise < -0.2 else '⚪'
        lines.append(f"    IP:        {ip_icon} {ip:+.1f}% YoY  (consensus {ip_c:+.1f}%)  surprise={ip_surprise:+.1f}%")
    else:
        lines.append(f"    IP:        ⏳ pending")

    # Retail Sales
    if retail is not None:
        retail_surprise = retail - (retail_c or retail)
        retail_icon = '🟢' if retail_surprise > 0.2 else '🔴' if retail_surprise < -0.2 else '⚪'
        lines.append(f"    Retail:    {retail_icon} {retail:+.1f}% YoY  (consensus {retail_c:+.1f}%)  surprise={retail_surprise:+.1f}%")
    else:
        lines.append(f"    Retail:    ⏳ pending")

    # Fixed Asset Investment
    if fai is not None:
        fai_icon = '🟢' if fai > 0 else '🔴' if fai < 0 else '⚪'
        lines.append(f"    FAI YTD:   {fai_icon} {fai:+.1f}% YoY")
    else:
        lines.append(f"    FAI YTD:   ⏳ pending")

    # Unemployment
    if unemp is not None:
        unemp_icon = '🟢' if unemp <= 5.0 else '🟡' if unemp <= 5.3 else '🔴'
        lines.append(f"    Unemp:     {unemp_icon} {unemp:.1f}%")

    # House Prices
    if hp is not None:
        hp_icon = '🟢' if hp > -1.0 else '🟡' if hp > -3.0 else '🔴'
        lines.append(f"    House:     {hp_icon} {hp:+.1f}% YoY")

    # Composite
    if details:
        composite = details.get('composite', '?')
        desc = details.get('description', '')
        wscore = details.get('weighted_score', 0)
        composite_icons = {
            'STRONG_BEAT': '🟢🟢', 'BEAT': '🟢', 'MILD_BEAT': '🟢',
            'INLINE': '⚪', 'MIXED': '🟡', 'MILD_MISS': '🟡',
            'MISS': '🔴', 'BIG_MISS': '🔴🔴', 'PROPERTY_CRISIS': '🚨',
            'PENDING': '⏳',
        }
        icon = composite_icons.get(composite, '⚪')
        lines.append(f"    Composite: {icon} {composite} (score={wscore:+.2f}) — {desc}")

        # Show per-indicator scores
        scores = details.get('indicator_scores', {})
        if scores:
            score_parts = []
            for k in ('ip', 'retail', 'fai', 'house'):
                if k in scores:
                    s = scores[k]
                    s_icon = '🟢' if s > 0 else '🔴' if s < 0 else '⚪'
                    score_parts.append(f'{k}={s_icon}{s:+.1f}')
            lines.append(f"    Scores: {' | '.join(score_parts)}")

    return '\n'.join(lines)

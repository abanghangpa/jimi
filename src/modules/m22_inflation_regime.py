"""
M22: Inflation Regime Scorer

Scores macro inflation environment based on PPI/CPI direction, Fed stance,
and market positioning. Derived from 8-year historical study (2018-2026)
of PPI prints and crypto reactions.

The Grand Unified PPI Matrix:
    PPI Direction × Fed Stance × Positioning = Score

Data sources (Phase 1: manual config overrides):
    - M22_PPI_YOY, M22_PPI_MOM in settings.yaml
    - M22_CPI_YOY in settings.yaml
    - M22_FED_STANCE in settings.yaml
    - L/S ratio from live derivatives data
"""

from src.config import CONFIG


# ═══════════════════════════════════════════════════════════════
# REGIME MATRIX — 8 years of PPI × crypto, distilled
# ═══════════════════════════════════════════════════════════════

REGIME_MATRIX = {
    # (ppi_direction, fed_stance, positioning) → (score, label, severity, description)
    ('FALLING', 'CUTTING', 'NEUTRAL'): {
        'score': 0.85, 'regime': 'GOLDILOCKS',
        'severity': 'LOW',
        'desc': 'Disinflation + rate cuts + neutral positioning — best setup',
        'analog': '2019, 2020 H1, 2023',
        'expected_move': '+3% to +10% (multi-week)',
    },
    ('FALLING', 'CUTTING', 'CROWDED'): {
        'score': 0.70, 'regime': 'GOLDILOCKS_RISKY',
        'severity': 'LOW',
        'desc': 'Good macro but crowded longs — dip-buying still works',
        'analog': '2020 H2',
        'expected_move': '+2% to +5%',
    },
    ('FALLING', 'HOLDING', 'NEUTRAL'): {
        'score': 0.65, 'regime': 'DISINFLATION',
        'severity': 'LOW',
        'desc': 'Inflation cooling, Fed patient — constructive for risk',
        'analog': '2023 H1',
        'expected_move': '+1% to +3%',
    },
    ('FALLING', 'HOLDING', 'CROWDED'): {
        'score': 0.55, 'regime': 'DISINFLATION_CROWDED',
        'severity': 'MEDIUM',
        'desc': 'Inflation cooling but max long — vulnerable to shocks',
        'analog': '2025',
        'expected_move': '-1% to +2% (volatile)',
    },
    ('FALLING', 'HIKING', 'NEUTRAL'): {
        'score': 0.40, 'regime': 'TIGHTENING',
        'severity': 'MEDIUM',
        'desc': 'Fed hiking but inflation falling — late cycle pain',
        'analog': '2022 H2',
        'expected_move': '-2% to -5%',
    },
    ('FALLING', 'HIKING', 'CROWDED'): {
        'score': 0.30, 'regime': 'TIGHTENING_TRAP',
        'severity': 'HIGH',
        'desc': 'Fed hiking + inflation falling + max long — leverage unwind',
        'analog': '2022 H1',
        'expected_move': '-5% to -10%',
    },
    ('RISING', 'CUTTING', 'NEUTRAL'): {
        'score': 0.75, 'regime': 'REFLATION',
        'severity': 'LOW',
        'desc': 'Rising inflation + rate cuts = stimulus narrative — bullish',
        'analog': '2020 H2, 2021 H1',
        'expected_move': '+3% to +8%',
    },
    ('RISING', 'CUTTING', 'CROWDED'): {
        'score': 0.60, 'regime': 'REFLATION_RISKY',
        'severity': 'MEDIUM',
        'desc': 'Reflation narrative but crowded — works until PPI crosses 6%',
        'analog': '2021 H1 (before flip)',
        'expected_move': '+1% to +5% (then reverses)',
    },
    ('RISING', 'HOLDING', 'NEUTRAL'): {
        'score': 0.45, 'regime': 'STAGFLATION_LITE',
        'severity': 'MEDIUM',
        'desc': 'Inflation rising, Fed on hold — uncertain, choppy',
        'analog': 'mild version of today',
        'expected_move': '-1% to -3%',
    },
    ('RISING', 'HOLDING', 'CROWDED'): {
        'score': 0.25, 'regime': 'STAGFLATION',
        'severity': 'HIGH',
        'desc': 'WORST COMBO: inflation rising + no cuts + max long — cascade risk',
        'analog': '2026 TODAY, 2021 H2',
        'expected_move': '-3% to -5%',
    },
    ('RISING', 'HIKING', 'NEUTRAL'): {
        'score': 0.30, 'regime': 'INFLATION_SHOCK',
        'severity': 'HIGH',
        'desc': 'Rising inflation + active hikes — aggressive tightening',
        'analog': '2018 H1',
        'expected_move': '-5% to -10%',
    },
    ('RISING', 'HIKING', 'CROWDED'): {
        'score': 0.15, 'regime': 'INFLATION_CRASH',
        'severity': 'CRITICAL',
        'desc': 'Catastrophic: rising inflation + hikes + max long — expect crash',
        'analog': '2018 H1, 2022 H1',
        'expected_move': '-10% to -50%',
    },
}

# Flat (PPI direction unchanged) maps to HOLDING-equivalent
FLAT_REGIME_OVERRIDES = {
    ('FLAT', 'CUTTING', 'NEUTRAL'): 0.75,   # same as RISING/CUTTING/NEUTRAL
    ('FLAT', 'CUTTING', 'CROWDED'): 0.60,
    ('FLAT', 'HOLDING', 'NEUTRAL'): 0.55,   # slightly better than RISING/HOLDING
    ('FLAT', 'HOLDING', 'CROWDED'): 0.45,
    ('FLAT', 'HIKING', 'NEUTRAL'): 0.40,
    ('FLAT', 'HIKING', 'CROWDED'): 0.30,
}


# ═══════════════════════════════════════════════════════════════
# CLASSIFICATION HELPERS
# ═══════════════════════════════════════════════════════════════

def classify_ppi_direction(ppi_yoy, ppi_prev_yoy=None, ppi_mom=None):
    """Classify PPI direction as RISING, FALLING, or FLAT.

    Uses YoY change if previous YoY is available, otherwise falls back to MoM.
    """
    if ppi_prev_yoy is not None:
        delta = ppi_yoy - ppi_prev_yoy
        if delta > 0.2:
            return 'RISING'
        elif delta < -0.2:
            return 'FALLING'
        else:
            return 'FLAT'
    elif ppi_mom is not None:
        if ppi_mom > 0.2:
            return 'RISING'
        elif ppi_mom < -0.2:
            return 'FALLING'
        else:
            return 'FLAT'
    else:
        return 'FLAT'


def classify_positioning(ls_ratio, ls_threshold=2.0):
    """Classify positioning as CROWDED or NEUTRAL based on L/S ratio."""
    if ls_ratio is None:
        return 'NEUTRAL'
    return 'CROWDED' if ls_ratio >= ls_threshold else 'NEUTRAL'


def classify_fed_stance(fed_stance_str):
    """Normalize fed stance string to CUTTING, HOLDING, or HIKING."""
    if not fed_stance_str:
        return 'HOLDING'
    s = fed_stance_str.upper().strip()
    if s in ('CUT', 'CUTTING', 'DOVISH', 'EASING', 'RATE_CUT'):
        return 'CUTTING'
    elif s in ('HIKE', 'HIKING', 'HAWKISH', 'TIGHTENING', 'RATE_HIKE'):
        return 'HIKING'
    else:
        return 'HOLDING'


# ═══════════════════════════════════════════════════════════════
# MAIN SCORING FUNCTION
# ═══════════════════════════════════════════════════════════════

def score_m22_inflation(ppi_yoy, ppi_prev_yoy=None, ppi_mom=None,
                        cpi_yoy=None, fed_stance='HOLDING',
                        ls_ratio=None, direction='LONG',
                        config=None):
    """Score the inflation regime for a given trade direction.

    Args:
        ppi_yoy: Latest PPI year-over-year percentage (e.g., 4.9)
        ppi_prev_yoy: Previous month's PPI YoY (for direction classification)
        ppi_mom: Latest PPI month-over-month percentage (fallback for direction)
        cpi_yoy: Latest CPI year-over-year percentage (supplementary)
        fed_stance: Fed stance string ('CUTTING', 'HOLDING', 'HIKING', etc.)
        ls_ratio: Long/short ratio from derivatives data
        direction: Trade direction ('LONG' or 'SHORT')
        config: Config dict (optional)

    Returns:
        status: 'PASS', 'FAIL', or 'VETO'
        score: 0.0-1.0 score
        details: dict with full regime info
    """
    cfg = config or CONFIG

    # Classify the three dimensions
    ppi_dir = classify_ppi_direction(ppi_yoy, ppi_prev_yoy, ppi_mom)
    fed = classify_fed_stance(fed_stance)
    ls_threshold = cfg.get('M22_LS_CROWDED_THRESHOLD', 2.0)
    pos = classify_positioning(ls_ratio, ls_threshold)

    # Look up regime in matrix
    key = (ppi_dir, fed, pos)
    if key in REGIME_MATRIX:
        regime_info = REGIME_MATRIX[key]
    elif key in FLAT_REGIME_OVERRIDES:
        score_raw = FLAT_REGIME_OVERRIDES[key]
        regime_info = {
            'score': score_raw,
            'regime': f'FLAT_{fed}',
            'severity': 'MEDIUM',
            'desc': f'PPI flat, Fed {fed.lower()}, positioning {pos.lower()}',
            'analog': 'N/A',
            'expected_move': 'N/A',
        }
    else:
        # Fallback — shouldn't happen
        regime_info = {
            'score': 0.50,
            'regime': 'UNKNOWN',
            'severity': 'MEDIUM',
            'desc': f'Unknown regime: {key}',
            'analog': 'N/A',
            'expected_move': 'N/A',
        }

    score_raw = regime_info['score']
    regime = regime_info['regime']
    severity = regime_info['severity']

    # Direction adjustment: for SHORT trades, invert the score logic
    # A bad inflation regime for LONGs is good for SHORTs
    if direction == 'SHORT':
        score = 1.0 - score_raw
    else:
        score = score_raw

    score = max(0.0, min(1.0, score))

    # Determine status
    veto_threshold = cfg.get('M22_VETO_THRESHOLD', 0.20)
    if score_raw < veto_threshold and direction == 'LONG':
        status = 'VETO'
    elif score_raw < veto_threshold and direction == 'SHORT':
        status = 'PASS'  # Veto only blocks the bad direction
    elif score >= cfg.get('M22_FAIL_THRESHOLD', 0.35):
        status = 'PASS'
    else:
        status = 'FAIL'

    # CPI supplement: if CPI is also hot, increase severity
    cpi_hot = False
    if cpi_yoy is not None:
        cpi_hot_threshold = cfg.get('M22_CPI_HOT_THRESHOLD', 3.5)
        if cpi_yoy >= cpi_hot_threshold:
            cpi_hot = True
            if direction == 'LONG' and score_raw > 0.20:
                # CPI hot on top of bad PPI → reduce score further
                score *= 0.90

    # Surprise factor: if PPI was much hotter than expected, extra penalty
    surprise_penalty = 0.0
    ppi_expected = cfg.get('M22_PPI_EXPECTED', None)
    if ppi_expected is not None and ppi_yoy > ppi_expected:
        surprise = ppi_yoy - ppi_expected
        if surprise > 0.5:  # more than 0.5% above expected
            surprise_penalty = min(surprise * 0.05, 0.10)  # max 10% penalty
            if direction == 'LONG':
                score *= (1.0 - surprise_penalty)

    # Size multiplier for position sizing
    size_mult = 1.0
    if severity == 'CRITICAL':
        size_mult = cfg.get('M22_SIZE_CRITICAL', 0.30)
    elif severity == 'HIGH':
        size_mult = cfg.get('M22_SIZE_HIGH', 0.50)
    elif severity == 'MEDIUM':
        size_mult = cfg.get('M22_SIZE_MEDIUM', 0.75)
    # LOW severity → size_mult = 1.0

    details = {
        'regime': regime,
        'severity': severity,
        'ppi_yoy': ppi_yoy,
        'ppi_direction': ppi_dir,
        'ppi_mom': ppi_mom,
        'cpi_yoy': cpi_yoy,
        'cpi_hot': cpi_hot,
        'fed_stance': fed,
        'positioning': pos,
        'ls_ratio': ls_ratio,
        'ls_threshold': ls_threshold,
        'direction': direction,
        'score_raw': round(score_raw, 3),
        'size_mult': round(size_mult, 2),
        'surprise_penalty': round(surprise_penalty, 3),
        'analog': regime_info.get('analog', 'N/A'),
        'expected_move': regime_info.get('expected_move', 'N/A'),
        'description': regime_info.get('desc', ''),
    }

    return status, score, details


# ═══════════════════════════════════════════════════════════════
# CONVENIENCE WRAPPER (for scanner/engine integration)
# ═══════════════════════════════════════════════════════════════

def score_m22(direction='LONG', ls_ratio=None, config=None):
    """Score inflation regime using config overrides.

    Reads PPI/CPI/Fed values from config (settings.yaml).
    This is the Phase 1 integration — manual config, no API calls.

    Args:
        direction: Trade direction ('LONG' or 'SHORT')
        ls_ratio: Live L/S ratio from derivatives
        config: Config dict

    Returns:
        status: 'PASS', 'FAIL', or 'VETO'
        score: 0.0-1.0
        details: dict
    """
    cfg = config or CONFIG

    if not cfg.get('M22_ENABLED', False):
        return 'SKIP', 0.5, {'regime': 'DISABLED'}

    ppi_yoy = cfg.get('M22_PPI_YOY', None)
    if ppi_yoy is None:
        return 'SKIP', 0.5, {'regime': 'NO_DATA', 'reason': 'M22_PPI_YOY not set'}

    ppi_prev_yoy = cfg.get('M22_PPI_PREV_YOY', None)
    ppi_mom = cfg.get('M22_PPI_MOM', None)
    cpi_yoy = cfg.get('M22_CPI_YOY', None)
    fed_stance = cfg.get('M22_FED_STANCE', 'HOLDING')

    return score_m22_inflation(
        ppi_yoy=ppi_yoy,
        ppi_prev_yoy=ppi_prev_yoy,
        ppi_mom=ppi_mom,
        cpi_yoy=cpi_yoy,
        fed_stance=fed_stance,
        ls_ratio=ls_ratio,
        direction=direction,
        config=cfg,
    )


# ═══════════════════════════════════════════════════════════════
# FORMATTER (for scanner output)
# ═══════════════════════════════════════════════════════════════

def format_m22(details):
    """Format M22 details for terminal output."""
    if not details or details.get('regime') in ('DISABLED', 'NO_DATA', 'UNKNOWN'):
        return ''

    regime = details.get('regime', '?')
    severity = details.get('severity', '?')
    ppi = details.get('ppi_yoy', 0)
    ppi_dir = details.get('ppi_direction', '?')
    cpi = details.get('cpi_yoy')
    fed = details.get('fed_stance', '?')
    pos = details.get('positioning', '?')
    score = details.get('score_raw', 0.5)
    analog = details.get('analog', '')
    expected = details.get('expected_move', '')
    desc = details.get('description', '')
    ls = details.get('ls_ratio', 0)
    size_mult = details.get('size_mult', 1.0)

    severity_icons = {
        'LOW': '🟢', 'MEDIUM': '🟡', 'HIGH': '🟠', 'CRITICAL': '🔴'
    }
    icon = severity_icons.get(severity, '⚪')

    lines = []
    lines.append(f"\n  {icon} M22 INFLATION REGIME: {regime}")
    lines.append(f"    PPI YoY: {ppi:.1f}% ({ppi_dir})  |  CPI: {cpi:.1f}%{'  ⚠️ HOT' if cpi and cpi >= 3.5 else '' if cpi else ''}")
    lines.append(f"    Fed: {fed}  |  L/S: {ls:.2f} ({pos})  |  Score: {score:.3f}")
    lines.append(f"    Severity: {severity}  |  Size mult: {size_mult:.2f}x")
    if desc:
        lines.append(f"    📖 {desc}")
    if analog and analog != 'N/A':
        lines.append(f"    📊 Analog: {analog}")
    if expected and expected != 'N/A':
        lines.append(f"    📈 Expected: {expected}")

    return '\n'.join(lines)

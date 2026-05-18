"""
M22: Inflation Regime Scorer

Scores macro inflation environment based on PPI/CPI direction, Fed stance,
market positioning, and labor market context (jobless claims).

The Grand Unified Matrix:
    PPI Direction × Fed Stance × Positioning × Claims Context = Score

Claims context modifies the regime:
    - Claims LOW + CPI hot = Fed TRAPPED (can't cut, labor strong) → worst case
    - Claims HIGH + CPI hot = Fed WILL CUT (recession > inflation) → less bad
    - Claims LOW + CPI cool = GOLDILOCKS (Fed can cut, economy ok) → best case
    - Claims HIGH + CPI cool = RECESSION FEAR (Fed cuts aggressively) → short-term pain

Data sources:
    - M22_PPI_YOY, M22_PPI_MOM in settings.yaml
    - M22_CPI_YOY in settings.yaml
    - M22_FED_STANCE in settings.yaml
    - L/S ratio from live derivatives data
    - Jobless claims from M23 (FRED cache or hardcoded)
"""

from src.config import CONFIG
from datetime import datetime


# ═══════════════════════════════════════════════════════════════
# TIME DECAY — macro data loses relevance between releases
# ═══════════════════════════════════════════════════════════════
# PPI/CPI are released mid-month. The regime assessment is most
# impactful on release day and fades as the market digests.
# Decay schedule (days since release → multiplier):
#   0-3:   1.00  (full — fresh data, market still reacting)
#   4-7:   0.85  (digesting)
#   8-14:  0.60  (halfway to next release)
#   15-21: 0.40  (stale)
#   22+:   0.30  (floor — regime still matters, but heavily discounted)

M22_PPI_RELEASE_DATES = {
    '2026-01-14', '2026-02-13', '2026-03-13', '2026-04-14',
    '2026-05-13', '2026-06-11', '2026-07-10', '2026-08-13',
    '2026-09-11', '2026-10-14', '2026-11-13', '2026-12-10',
    # 2025
    '2025-01-14', '2025-02-13', '2025-03-13', '2025-04-10',
    '2025-05-15', '2025-06-12', '2025-07-16', '2025-08-14',
    '2025-09-11', '2025-10-15', '2025-11-13', '2025-12-11',
}

M22_CPI_RELEASE_DATES = {
    '2026-01-14', '2026-02-11', '2026-03-11', '2026-04-10',
    '2026-05-12', '2026-06-10', '2026-07-14', '2026-08-12',
    '2026-09-10', '2026-10-13', '2026-11-10', '2026-12-09',
    # 2025
    '2025-01-15', '2025-02-12', '2025-03-12', '2025-04-10',
    '2025-05-13', '2025-06-11', '2025-07-15', '2025-08-12',
    '2025-09-10', '2025-10-14', '2025-11-12', '2025-12-10',
}


def _compute_release_decay(release_dates, today_str=None):
    """Compute decay multiplier based on days since last release.

    Returns:
        (multiplier: float, days_since: int, last_release: str or None)
    """
    if today_str is None:
        today_str = datetime.utcnow().strftime('%Y-%m-%d')

    today = datetime.strptime(today_str, '%Y-%m-%d')

    # Find the most recent release on or before today
    past_releases = sorted(d for d in release_dates if d <= today_str)
    if not past_releases:
        return 1.0, 0, None  # no data, assume fresh

    last_release = past_releases[-1]
    last_dt = datetime.strptime(last_release, '%Y-%m-%d')
    days_since = (today - last_dt).days

    # Decay schedule
    if days_since <= 3:
        mult = 1.00
    elif days_since <= 7:
        mult = 0.85
    elif days_since <= 14:
        mult = 0.60
    elif days_since <= 21:
        mult = 0.40
    else:
        mult = 0.30

    return mult, days_since, last_release


def compute_m22_decay(config=None, today_str=None):
    """Compute M22 time decay from PPI/CPI release schedule.

    Uses the MORE RECENT of PPI and CPI to determine freshness.
    The regime is based on the latest data point — whichever released last.

    Returns:
        dict with decay multiplier, days_since, last_release info
    """
    cfg = config or CONFIG

    ppi_mult, ppi_days, ppi_date = _compute_release_decay(M22_PPI_RELEASE_DATES, today_str)
    cpi_mult, cpi_days, cpi_date = _compute_release_decay(M22_CPI_RELEASE_DATES, today_str)

    # Use the more recent (higher multiplier = more fresh)
    if ppi_mult >= cpi_mult:
        best_mult, best_days, best_date, best_type = ppi_mult, ppi_days, ppi_date, 'PPI'
    else:
        best_mult, best_days, best_date, best_type = cpi_mult, cpi_days, cpi_date, 'CPI'

    # If both are stale (>14 days), extra penalty
    both_stale = ppi_days > 14 and cpi_days > 14
    if both_stale:
        best_mult = min(best_mult, 0.25)

    return {
        'decay_mult': round(best_mult, 2),
        'days_since': best_days,
        'last_release': best_date,
        'last_type': best_type,
        'ppi_days': ppi_days,
        'cpi_days': cpi_days,
        'both_stale': both_stale,
    }


# ═══════════════════════════════════════════════════════════════
# CLAIMS MODIFIER — labor market context changes what inflation means
# ═══════════════════════════════════════════════════════════════
# Claims data is imported from M23's module-level cache.
# The modifier adjusts the regime score based on whether the Fed
# has room to act or is trapped by strong labor data.

# Claims classification thresholds (same as M23)
CLAIMS_LOW = 210        # Tight labor market
CLAIMS_ELEVATED = 225   # Labor softening
CLAIMS_SPIKE = 240      # Shock spike

# Modifier: (score_adjustment, description)
# Applied AFTER the base regime score is computed.
# Positive = more bullish, Negative = more bearish.
CLAIMS_REGIME_MODIFIER = {
    # Claims LOW (<210K) — tight labor, Fed's hands tied
    # Hot CPI + low claims = Fed TRAPPED (can't justify cuts)
    # Cool CPI + low claims = Goldilocks (can cut if needed)
    'LOW': {
        'RISING':  (-0.08, 'Fed trapped: labor strong + inflation rising → can\'t cut'),
        'FALLING': (+0.10, 'Goldilocks: labor strong + inflation falling → Fed can cut'),
        'FLAT':    (+0.05, 'Stable labor + flat inflation → neutral-positive'),
    },
    # Claims NORMAL (210-225K) — no modifier
    'NORMAL': {
        'RISING':  (+0.00, 'Normal labor + inflation rising → status quo'),
        'FALLING': (+0.00, 'Normal labor + inflation falling → status quo'),
        'FLAT':    (+0.00, 'Normal labor + flat inflation → no change'),
    },
    # Claims ELEVATED (>225K) — labor softening, Fed has cover to cut
    # Hot CPI + elevated claims = Fed will cut despite inflation (recession fear wins)
    # Cool CPI + elevated claims = aggressive cuts coming (but recession = short-term pain)
    'ELEVATED': {
        'RISING':  (+0.05, 'Labor softening: Fed will cut despite inflation → less stagflation'),
        'FALLING': (-0.05, 'Recession signal: labor weakening + deflation → risk-off short-term'),
        'FLAT':    (+0.00, 'Labor softening + flat inflation → wait and see'),
    },
    # Claims SPIKE (>240K) — shock territory
    # Overrides: Fed forced to cut regardless of inflation (2020 playbook)
    'SPIKE': {
        'RISING':  (+0.10, 'Claims spike: Fed forced to cut → inflation secondary'),
        'FALLING': (-0.10, 'Claims spike + deflation → genuine recession, sell risk'),
        'FLAT':    (+0.05, 'Claims spike: Fed will act → supportive eventually'),
    },
    # Claims CRISIS (>280K) — 2020 territory
    'CRISIS': {
        'RISING':  (+0.15, 'Crisis: Fed will do whatever it takes → expect emergency cuts'),
        'FALLING': (-0.15, 'Crisis + deflation → liquidity crisis, sell everything'),
        'FLAT':    (+0.10, 'Crisis: emergency response coming → eventual recovery'),
    },
}


def _get_claims_classification():
    """Get claims classification from M23's cached data.

    Returns (classification, modifier_dict) or (None, None) if unavailable.
    """
    try:
        from src.modules.macro_utils import get_claims_trend
        claims = get_claims_trend()
        if claims is None:
            return None, None
        classification = claims.get('classification', 'NORMAL')
        return classification, CLAIMS_REGIME_MODIFIER.get(classification, CLAIMS_REGIME_MODIFIER['NORMAL'])
    except Exception:
        return None, None


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

    # ── Claims modifier: labor market context changes what inflation means ──
    claims_class, claims_mods = _get_claims_classification()
    claims_adjust = 0.0
    claims_desc = ''
    if claims_mods is not None and ppi_dir in claims_mods:
        claims_adjust, claims_desc = claims_mods[ppi_dir]
        score_raw = max(0.05, min(0.95, score_raw + claims_adjust))

    # ── Time decay: macro data loses relevance between releases ──
    decay = compute_m22_decay(cfg)
    decay_mult = decay['decay_mult']

    # Apply decay to score_raw: pull it toward neutral (0.5) based on freshness
    # At full freshness (1.0): score_raw unchanged
    # At decay (0.3): score_raw pulled 70% toward neutral
    score_raw = score_raw * decay_mult + 0.5 * (1.0 - decay_mult)

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
        # Claims context
        'claims_classification': claims_class,
        'claims_adjust': round(claims_adjust, 3),
        'claims_description': claims_desc,
        # Time decay
        'decay_mult': decay['decay_mult'],
        'days_since_release': decay['days_since'],
        'last_release_date': decay['last_release'],
        'last_release_type': decay['last_type'],
        'ppi_days_since': decay['ppi_days'],
        'cpi_days_since': decay['cpi_days'],
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
    claims_class = details.get('claims_classification')
    claims_adjust = details.get('claims_adjust', 0)
    claims_desc = details.get('claims_description', '')

    severity_icons = {
        'LOW': '🟢', 'MEDIUM': '🟡', 'HIGH': '🟠', 'CRITICAL': '🔴'
    }
    icon = severity_icons.get(severity, '⚪')

    lines = []
    lines.append(f"\n  {icon} M22 INFLATION REGIME: {regime}")
    lines.append(f"    PPI YoY: {ppi:.1f}% ({ppi_dir})  |  CPI: {cpi:.1f}%{'  ⚠️ HOT' if cpi and cpi >= 3.5 else '' if cpi else ''}")
    lines.append(f"    Fed: {fed}  |  L/S: {ls:.2f} ({pos})  |  Score: {score:.3f}")
    lines.append(f"    Severity: {severity}  |  Size mult: {size_mult:.2f}x")
    # Time decay
    decay_mult = details.get('decay_mult', 1.0)
    days_since = details.get('days_since_release', 0)
    last_rel = details.get('last_release_date', '')
    last_type = details.get('last_release_type', '')
    if decay_mult < 1.0:
        decay_icon = '🟢' if decay_mult >= 0.85 else '🟡' if decay_mult >= 0.60 else '🟠' if decay_mult >= 0.40 else '🔴'
        lines.append(f"    {decay_icon} Decay: {decay_mult:.2f}x  ({days_since}d since {last_type} {last_rel})")
    # Claims context
    if claims_class:
        claims_icons = {
            'LOW': '🟢', 'NORMAL': '⚪', 'ELEVATED': '🟡',
            'SPIKE': '🟠', 'CRISIS': '🔴'
        }
        c_icon = claims_icons.get(claims_class, '⚪')
        adj_str = f'{claims_adjust:+.3f}' if claims_adjust != 0 else '—'
        lines.append(f"    Claims: {c_icon} {claims_class}  (adj: {adj_str})")
        if claims_desc:
            lines.append(f"      • {claims_desc}")
    if desc:
        lines.append(f"    📖 {desc}")
    if analog and analog != 'N/A':
        lines.append(f"    📊 Analog: {analog}")
    if expected and expected != 'N/A':
        lines.append(f"    📈 Expected: {expected}")

    return '\n'.join(lines)

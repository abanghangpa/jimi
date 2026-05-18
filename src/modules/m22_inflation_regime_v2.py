"""
M22 v2: Expanded Inflation Regime Scorer

5-dimensional macro regime classification:
    1. PPI direction (RISING / FALLING / FLAT)
    2. CPI confirmation (HOT / WARM / COOL) — co-equal, not supplementary
    3. Fed stance (CUTTING / HOLDING / HIKING)
    4. Labor context (GOLDILOCKS / NORMAL / SOFTENING / CRISIS)
    5. Positioning (CROWDED / NEUTRAL)

Plus two continuous modifiers:
    - Real rate = Fed funds - PPI YoY (tight vs stimulative)
    - PPI acceleration = current delta vs previous delta (accelerating / decelerating)

Architecture:
    Base regime = PPI_dir × Fed × Labor  (3D, labor IN the matrix)
    CPI overlay = regime shift or score adjustment
    Real rate modifier = severity adjustment
    Acceleration modifier = direction confidence
    Positioning = final score multiplier
    Claims context = Fed reaction function modifier

Key change from v1: Labor is no longer a ±0.15 afterthought.
It's a co-equal regime selector. "Fed trapped" isn't a score tweak —
it's a different regime label (STAGFLATION_TRAPPED).
"""

from src.config import CONFIG
from datetime import datetime

# Release dates sourced from standalone modules (M60: PPI, M56: CPI)
from src.modules.m60_us_ppi import PPI_SCHEDULE_DATES as _PPI_RELEASE_DATES
from src.modules.m56_us_cpi import CPI_SCHEDULE_DATES as _CPI_RELEASE_DATES


# ═══════════════════════════════════════════════════════════════
# TIME DECAY — release schedule from M60/M56
# ═══════════════════════════════════════════════════════════════


def _compute_release_decay(release_dates, today_str=None):
    """Compute decay multiplier based on days since last release."""
    if today_str is None:
        today_str = datetime.utcnow().strftime('%Y-%m-%d')
    today = datetime.strptime(today_str, '%Y-%m-%d')
    past_releases = sorted(d for d in release_dates if d <= today_str)
    if not past_releases:
        return 1.0, 0, None
    last_release = past_releases[-1]
    last_dt = datetime.strptime(last_release, '%Y-%m-%d')
    days_since = (today - last_dt).days
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
    """Compute M22 time decay from PPI/CPI release schedule (sourced from M60/M56)."""
    cfg = config or CONFIG
    ppi_mult, ppi_days, ppi_date = _compute_release_decay(_PPI_RELEASE_DATES, today_str)
    cpi_mult, cpi_days, cpi_date = _compute_release_decay(_CPI_RELEASE_DATES, today_str)
    if ppi_mult >= cpi_mult:
        best_mult, best_days, best_date, best_type = ppi_mult, ppi_days, ppi_date, 'PPI'
    else:
        best_mult, best_days, best_date, best_type = cpi_mult, cpi_days, cpi_date, 'CPI'
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
# CLASSIFICATION HELPERS
# ═══════════════════════════════════════════════════════════════

def classify_ppi_direction(ppi_yoy, ppi_prev_yoy=None, ppi_mom=None):
    """Classify PPI direction as RISING, FALLING, or FLAT."""
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
    return 'FLAT'


def classify_ppi_acceleration(ppi_yoy, ppi_prev_yoy, ppi_prev_prev_yoy=None):
    """Detect PPI acceleration (rate of change of rate of change).

    RISING + ACCELERATING = getting worse faster (2021 H2)
    RISING + DECELERATING = peaking (2022 H1 — buy signal)
    FALLING + ACCELERATING = deflation accelerating (2023)
    FALLING + DECELERATING = bottoming (2024 — early signal)

    Returns: 'ACCELERATING', 'DECELERATING', 'STABLE', or 'INSUFFICIENT'
    """
    if ppi_prev_yoy is None:
        return 'INSUFFICIENT'

    current_delta = ppi_yoy - ppi_prev_yoy

    if ppi_prev_prev_yoy is not None:
        prev_delta = ppi_prev_yoy - ppi_prev_prev_yoy
        second_deriv = current_delta - prev_delta
        if second_deriv > 0.3:
            return 'ACCELERATING'
        elif second_deriv < -0.3:
            return 'DECELERATING'
        else:
            return 'STABLE'
    else:
        # Only two data points — use magnitude as proxy
        if abs(current_delta) > 1.0:
            return 'ACCELERATING' if current_delta > 0 else 'DECELERATING'
        return 'STABLE'


def classify_cpi_confirmation(cpi_yoy, cpi_prev_yoy=None):
    """Classify CPI as HOT / WARM / COOL.

    CPI is the PRIMARY inflation signal for markets (not PPI).
    PPI confirms/denies CPI — but CPI sets the tone.

    Returns: ('HOT', 'WARM', 'COOL')
    """
    if cpi_yoy is None:
        return 'WARM'  # no data → neutral
    if cpi_yoy >= 3.5:
        return 'HOT'
    elif cpi_yoy >= 2.5:
        return 'WARM'
    else:
        return 'COOL'


def classify_positioning(ls_ratio, ls_threshold=2.0):
    """Classify positioning as CROWDED or NEUTRAL."""
    if ls_ratio is None:
        return 'NEUTRAL'
    return 'CROWDED' if ls_ratio >= ls_threshold else 'NEUTRAL'


def classify_fed_stance(fed_stance_str):
    """Normalize fed stance to CUTTING / HOLDING / HIKING."""
    if not fed_stance_str:
        return 'HOLDING'
    s = fed_stance_str.upper().strip()
    if s in ('CUT', 'CUTTING', 'DOVISH', 'EASING', 'RATE_CUT'):
        return 'CUTTING'
    elif s in ('HIKE', 'HIKING', 'HAWKISH', 'TIGHTENING', 'RATE_HIKE'):
        return 'HIKING'
    return 'HOLDING'


def classify_labor_context(claims_classification, unemployment_rate=None, sahm_triggered=False):
    """Classify labor market context for regime selection.

    This is NOT a score modifier — it changes the regime label itself.
    Fed reaction function is f(inflation, employment). We need both.

    Returns: 'GOLDILOCKS', 'NORMAL', 'SOFTENING', 'CRISIS'
    """
    # Sahm Rule override — triggered = automatic SOFTENING minimum
    if sahm_triggered:
        if claims_classification in ('SPIKE', 'CRISIS'):
            return 'CRISIS'
        return 'SOFTENING'

    # Claims-based classification
    if claims_classification == 'CRISIS':
        return 'CRISIS'
    elif claims_classification == 'SPIKE':
        return 'CRISIS'  # spike → crisis territory
    elif claims_classification == 'ELEVATED':
        return 'SOFTENING'
    elif claims_classification == 'LOW':
        return 'GOLDILOCKS'
    else:
        return 'NORMAL'


def compute_real_rate(fed_funds_rate, ppi_yoy):
    """Compute real rate = Fed funds - PPI YoY.

    Positive real rates = tight (restrictive policy)
    Negative real rates = stimulative (even with high nominal rates)

    Returns: (real_rate: float, label: str)
    """
    if fed_funds_rate is None or ppi_yoy is None:
        return None, 'UNKNOWN'
    real = fed_funds_rate - ppi_yoy
    if real > 1.0:
        label = 'TIGHT'
    elif real > 0.0:
        label = 'MILDLY_TIGHT'
    elif real > -1.0:
        label = 'MILDLY_STIMULATIVE'
    else:
        label = 'STIMULATIVE'
    return round(real, 2), label


# ═══════════════════════════════════════════════════════════════
# REGIME MATRIX v2 — PPI_dir × Fed × Labor
# ═══════════════════════════════════════════════════════════════
# Labor is now IN the matrix, not a post-hoc modifier.
# This captures the Fed's dual mandate: inflation AND employment.
#
# Rows: PPI direction × Fed stance (same as v1)
# Columns: Labor context (GOLDILOCKS / NORMAL / SOFTENING / CRISIS)
#
# Key regime changes from v1:
#   - RISING + HOLDING + GOLDILOCKS = STAGFLATION (Fed trapped, can't cut)
#   - RISING + HOLDING + SOFTENING  = SLOWFLATION (Fed will cut despite inflation)
#   - RISING + HOLDING + CRISIS     = CRISIS_OVERRIDE (Fed forced to act)
#   - FALLING + HOLDING + GOLDILOCKS = GROWTH_SCARE (no rate cuts despite cooling)

REGIME_MATRIX_V2 = {
    # ── PPI FALLING + Fed CUTTING ──────────────────────────────
    ('FALLING', 'CUTTING', 'GOLDILOCKS'): {
        'score': 0.90, 'regime': 'GOLDILOCKS',
        'severity': 'LOW',
        'desc': 'Disinflation + rate cuts + strong labor — best possible setup',
        'analog': '2019 H2, 2023 Q4',
        'expected_move': '+5% to +12% (multi-week)',
    },
    ('FALLING', 'CUTTING', 'NORMAL'): {
        'score': 0.85, 'regime': 'GOLDILOCKS',
        'severity': 'LOW',
        'desc': 'Disinflation + rate cuts — classic risk-on environment',
        'analog': '2019, 2020 H1',
        'expected_move': '+3% to +10%',
    },
    ('FALLING', 'CUTTING', 'SOFTENING'): {
        'score': 0.70, 'regime': 'EASING_SCARE',
        'severity': 'MEDIUM',
        'desc': 'Fed cutting because labor is weakening — good macro but labor risk',
        'analog': '2024 Q3 (Sahm triggered)',
        'expected_move': '+1% to +5% (volatile)',
    },
    ('FALLING', 'CUTTING', 'CRISIS'): {
        'score': 0.55, 'regime': 'CRISIS_CUTS',
        'severity': 'HIGH',
        'desc': 'Fed cutting into crisis — deflation + recession fear',
        'analog': '2020 Q1 (COVID)',
        'expected_move': '-5% to +10% (extreme vol)',
    },

    # ── PPI FALLING + Fed HOLDING ──────────────────────────────
    ('FALLING', 'HOLDING', 'GOLDILOCKS'): {
        'score': 0.70, 'regime': 'GROWTH_SCARE',
        'severity': 'LOW',
        'desc': 'Inflation cooling + strong labor but Fed won\'t cut — market impatient',
        'analog': '2023 H1 (wait for cuts)',
        'expected_move': '+1% to +3%',
    },
    ('FALLING', 'HOLDING', 'NORMAL'): {
        'score': 0.65, 'regime': 'DISINFLATION',
        'severity': 'LOW',
        'desc': 'Inflation cooling, Fed patient — constructive for risk',
        'analog': '2023 H2',
        'expected_move': '+1% to +3%',
    },
    ('FALLING', 'HOLDING', 'SOFTENING'): {
        'score': 0.55, 'regime': 'WAIT_AND_SEE',
        'severity': 'MEDIUM',
        'desc': 'Inflation cooling but labor weakening — Fed watching, market uncertain',
        'analog': '2024 H1',
        'expected_move': '-1% to +2%',
    },
    ('FALLING', 'HOLDING', 'CRISIS'): {
        'score': 0.45, 'regime': 'PRE_CRISIS',
        'severity': 'HIGH',
        'desc': 'Deflation + labor crisis + Fed hasn\'t acted — bad combo',
        'analog': '2020 Feb (pre-COVID cuts)',
        'expected_move': '-3% to -8%',
    },

    # ── PPI FALLING + Fed HIKING ───────────────────────────────
    ('FALLING', 'HIKING', 'GOLDILOCKS'): {
        'score': 0.45, 'regime': 'TIGHTENING_LATE',
        'severity': 'MEDIUM',
        'desc': 'Fed hiking but inflation already falling — late cycle, labor still strong',
        'analog': '2022 Q4',
        'expected_move': '-2% to -5%',
    },
    ('FALLING', 'HIKING', 'NORMAL'): {
        'score': 0.40, 'regime': 'TIGHTENING',
        'severity': 'MEDIUM',
        'desc': 'Fed hiking but inflation falling — late cycle pain',
        'analog': '2022 H2',
        'expected_move': '-2% to -5%',
    },
    ('FALLING', 'HIKING', 'SOFTENING'): {
        'score': 0.30, 'regime': 'POLICY_ERROR',
        'severity': 'HIGH',
        'desc': 'Fed hiking into weakening labor — policy error risk',
        'analog': '2023 Q1 (SVB era)',
        'expected_move': '-5% to -10%',
    },
    ('FALLING', 'HIKING', 'CRISIS'): {
        'score': 0.20, 'regime': 'BREAKING',
        'severity': 'CRITICAL',
        'desc': 'Fed hiking into crisis — something will break',
        'analog': '2023 Q1 (SVB, Credit Suisse)',
        'expected_move': '-10% to -30%',
    },

    # ── PPI RISING + Fed CUTTING ───────────────────────────────
    ('RISING', 'CUTTING', 'GOLDILOCKS'): {
        'score': 0.80, 'regime': 'REFLATION',
        'severity': 'LOW',
        'desc': 'Rising inflation + rate cuts + strong labor — stimulus narrative',
        'analog': '2021 H1',
        'expected_move': '+3% to +8%',
    },
    ('RISING', 'CUTTING', 'NORMAL'): {
        'score': 0.75, 'regime': 'REFLATION',
        'severity': 'LOW',
        'desc': 'Rising inflation + rate cuts = reflation — works until PPI > 6%',
        'analog': '2020 H2',
        'expected_move': '+3% to +8%',
    },
    ('RISING', 'CUTTING', 'SOFTENING'): {
        'score': 0.60, 'regime': 'REFLATION_FRAGILE',
        'severity': 'MEDIUM',
        'desc': 'Rising inflation but labor weakening — Fed cutting into inflation, fragile',
        'analog': '2024 Q3 (pre-election cuts)',
        'expected_move': '0% to +3% (volatile)',
    },
    ('RISING', 'CUTTING', 'CRISIS'): {
        'score': 0.50, 'regime': 'CRISIS_REFLATION',
        'severity': 'HIGH',
        'desc': 'Rising inflation + crisis + cuts — stagflation risk, Fed confused',
        'analog': '1970s playbook',
        'expected_move': '-5% to +5% (no direction)',
    },

    # ── PPI RISING + Fed HOLDING ───────────────────────────────
    ('RISING', 'HOLDING', 'GOLDILOCKS'): {
        'score': 0.35, 'regime': 'STAGFLATION_TRAPPED',
        'severity': 'HIGH',
        'desc': 'Inflation rising + strong labor = Fed TRAPPED — can\'t cut, won\'t hike',
        'analog': '2026 TODAY',
        'expected_move': '-3% to -5%',
    },
    ('RISING', 'HOLDING', 'NORMAL'): {
        'score': 0.40, 'regime': 'STAGFLATION_LITE',
        'severity': 'MEDIUM',
        'desc': 'Inflation rising, Fed on hold — uncertain, choppy',
        'analog': 'mild 2026',
        'expected_move': '-1% to -3%',
    },
    ('RISING', 'HOLDING', 'SOFTENING'): {
        'score': 0.50, 'regime': 'SLOWFLATION',
        'severity': 'MEDIUM',
        'desc': 'Rising inflation + weakening labor — Fed will cut despite inflation',
        'analog': '2025 H2',
        'expected_move': '-1% to +2%',
    },
    ('RISING', 'HOLDING', 'CRISIS'): {
        'score': 0.45, 'regime': 'CRISIS_OVERRIDE',
        'severity': 'HIGH',
        'desc': 'Rising inflation + labor crisis — Fed forced to act, inflation secondary',
        'analog': '2020 Mar (COVID despite inflation)',
        'expected_move': '-10% then +20% (V-shape)',
    },

    # ── PPI RISING + Fed HIKING ────────────────────────────────
    ('RISING', 'HIKING', 'GOLDILOCKS'): {
        'score': 0.30, 'regime': 'INFLATION_SHOCK',
        'severity': 'HIGH',
        'desc': 'Rising inflation + hikes + strong labor — aggressive tightening',
        'analog': '2022 H1 (strong labor, Volcker vibes)',
        'expected_move': '-5% to -10%',
    },
    ('RISING', 'HIKING', 'NORMAL'): {
        'score': 0.25, 'regime': 'INFLATION_SHOCK',
        'severity': 'HIGH',
        'desc': 'Rising inflation + active hikes — aggressive tightening',
        'analog': '2018 H1',
        'expected_move': '-5% to -10%',
    },
    ('RISING', 'HIKING', 'SOFTENING'): {
        'score': 0.20, 'regime': 'STAGFLATION_CRASH',
        'severity': 'CRITICAL',
        'desc': 'Rising inflation + hikes + labor weakening — worst of all worlds',
        'analog': '2022 Q3 (before pivot)',
        'expected_move': '-10% to -20%',
    },
    ('RISING', 'HIKING', 'CRISIS'): {
        'score': 0.10, 'regime': 'INFLATION_CRASH',
        'severity': 'CRITICAL',
        'desc': 'Catastrophic: rising inflation + hikes + labor crisis — expect crash',
        'analog': '2022 H1, 2008',
        'expected_move': '-20% to -50%',
    },

    # ── PPI FLAT + Fed CUTTING ─────────────────────────────────
    ('FLAT', 'CUTTING', 'GOLDILOCKS'): {
        'score': 0.80, 'regime': 'EASING',
        'severity': 'LOW',
        'desc': 'Stable inflation + cuts + strong labor — supportive',
        'analog': '2024 Q4',
        'expected_move': '+2% to +5%',
    },
    ('FLAT', 'CUTTING', 'NORMAL'): {
        'score': 0.75, 'regime': 'EASING',
        'severity': 'LOW',
        'desc': 'Stable inflation + cuts — standard easing cycle',
        'analog': '2024 Q3',
        'expected_move': '+2% to +5%',
    },
    ('FLAT', 'CUTTING', 'SOFTENING'): {
        'score': 0.60, 'regime': 'EASING_SCARE',
        'severity': 'MEDIUM',
        'desc': 'Stable inflation + cuts because labor weakening — growth scare',
        'analog': '2024 Q3 (Sahm)',
        'expected_move': '0% to +3%',
    },
    ('FLAT', 'CUTTING', 'CRISIS'): {
        'score': 0.50, 'regime': 'CRISIS_EASING',
        'severity': 'HIGH',
        'desc': 'Stable inflation + crisis cuts — emergency easing',
        'analog': '2020 Mar',
        'expected_move': '-5% then +15%',
    },

    # ── PPI FLAT + Fed HOLDING ─────────────────────────────────
    ('FLAT', 'HOLDING', 'GOLDILOCKS'): {
        'score': 0.60, 'regime': 'NEUTRAL_STRONG',
        'severity': 'LOW',
        'desc': 'Stable inflation + holding + strong labor — no catalyst, range-bound',
        'analog': '2024 H1',
        'expected_move': '-1% to +2%',
    },
    ('FLAT', 'HOLDING', 'NORMAL'): {
        'score': 0.55, 'regime': 'NEUTRAL',
        'severity': 'LOW',
        'desc': 'Everything stable — no macro catalyst',
        'analog': 'boring months',
        'expected_move': '-1% to +1%',
    },
    ('FLAT', 'HOLDING', 'SOFTENING'): {
        'score': 0.45, 'regime': 'NEUTRAL_WEAK',
        'severity': 'MEDIUM',
        'desc': 'Stable inflation but labor weakening — pre-easing watch',
        'analog': '2024 H1 (waiting for Sahm)',
        'expected_move': '-2% to +1%',
    },
    ('FLAT', 'HOLDING', 'CRISIS'): {
        'score': 0.35, 'regime': 'PRE_CRISIS',
        'severity': 'HIGH',
        'desc': 'Stable inflation + labor crisis — Fed behind the curve',
        'analog': '2020 Jan',
        'expected_move': '-5% to -10%',
    },

    # ── PPI FLAT + Fed HIKING ──────────────────────────────────
    ('FLAT', 'HIKING', 'GOLDILOCKS'): {
        'score': 0.40, 'regime': 'TIGHTENING',
        'severity': 'MEDIUM',
        'desc': 'Stable inflation + hikes + strong labor — can absorb tightening',
        'analog': '2018 H2',
        'expected_move': '-2% to -5%',
    },
    ('FLAT', 'HIKING', 'NORMAL'): {
        'score': 0.35, 'regime': 'TIGHTENING',
        'severity': 'MEDIUM',
        'desc': 'Stable inflation + hikes — standard tightening cycle',
        'analog': '2018',
        'expected_move': '-2% to -5%',
    },
    ('FLAT', 'HIKING', 'SOFTENING'): {
        'score': 0.25, 'regime': 'POLICY_ERROR',
        'severity': 'HIGH',
        'desc': 'Stable inflation + hikes + labor weakening — policy error risk',
        'analog': '2023 Q1',
        'expected_move': '-5% to -10%',
    },
    ('FLAT', 'HIKING', 'CRISIS'): {
        'score': 0.15, 'regime': 'BREAKING',
        'severity': 'CRITICAL',
        'desc': 'Stable inflation + hikes + crisis — something breaks',
        'analog': 'SVB, Credit Suisse',
        'expected_move': '-15% to -30%',
    },
}


# ═══════════════════════════════════════════════════════════════
# CPI OVERLAY — co-equal inflation signal
# ═══════════════════════════════════════════════════════════════
# CPI is the PRIMARY inflation signal for markets. PPI is the confirmation.
# When CPI and PPI disagree, the regime should shift.
#
# The overlay works as: (base_regime, cpi_confirmation) → (score_adj, severity_adj, desc)
# Positive adj = more bullish, Negative = more bearish.

CPI_OVERLAY = {
    # ── CPI COOL (below 2.5%) ─────────────────────────────────
    # Cool CPI is unambiguously good. Even if PPI is rising, CPI cool
    # means consumer inflation is under control. Fed has room to cut.
    ('STAGFLATION_TRAPPED', 'COOL'):        (+0.20, 'DOWN', 'CPI cool breaks the trap — Fed can cut'),
    ('STAGFLATION_LITE', 'COOL'):           (+0.15, 'DOWN', 'CPI cool — less stagflation risk'),
    ('STAGFLATION_CRASH', 'COOL'):          (+0.10, 'DOWN', 'CPI cool — crash risk reduced but labor still bad'),
    ('INFLATION_SHOCK', 'COOL'):            (+0.15, 'DOWN', 'CPI cool — hikes likely to pause'),
    ('INFLATION_CRASH', 'COOL'):            (+0.10, 'DOWN', 'CPI cool — some relief in crash scenario'),
    ('SLOWFLATION', 'COOL'):                (+0.10, 'DOWN', 'CPI cool — slowflation resolving'),
    ('CRISIS_OVERRIDE', 'COOL'):            (+0.05, None, 'CPI cool — crisis focus shifts to labor'),
    ('REFLATION', 'COOL'):                  (+0.10, None, 'CPI cool + PPI rising = mixed inflation signal'),
    ('REFLATION_FRAGILE', 'COOL'):          (+0.15, 'DOWN', 'CPI cool — fragile reflation stabilizes'),
    ('NEUTRAL', 'COOL'):                    (+0.10, None, 'CPI cool — modestly positive'),
    ('NEUTRAL_STRONG', 'COOL'):             (+0.10, None, 'CPI cool — strong neutral gets a boost'),
    ('NEUTRAL_WEAK', 'COOL'):               (+0.15, 'DOWN', 'CPI cool — weak labor + cool CPI = Fed will cut'),
    ('GOLDILOCKS', 'COOL'):                 (+0.05, None, 'Already goldilocks — CPI confirms'),
    ('GROWTH_SCARE', 'COOL'):               (+0.10, None, 'CPI cool — growth scare less inflationary'),
    ('EASING', 'COOL'):                     (+0.05, None, 'CPI cool — easing cycle confirmed'),
    ('EASING_SCARE', 'COOL'):               (+0.10, None, 'CPI cool — easing + no inflation = good'),
    ('DISINFLATION', 'COOL'):               (+0.05, None, 'CPI cool — disinflation confirmed'),
    ('WAIT_AND_SEE', 'COOL'):               (+0.10, None, 'CPI cool — Fed will see room to cut'),
    ('PRE_CRISIS', 'COOL'):                 (+0.10, 'DOWN', 'CPI cool — some room for Fed to act'),
    ('TIGHTENING', 'COOL'):                 (+0.10, None, 'CPI cool — tightening end in sight'),
    ('TIGHTENING_LATE', 'COOL'):            (+0.10, None, 'CPI cool — late tightening confirmed'),
    ('POLICY_ERROR', 'COOL'):               (+0.15, 'DOWN', 'CPI cool — policy error less severe'),
    ('BREAKING', 'COOL'):                   (+0.10, 'DOWN', 'CPI cool — some relief'),
    ('CRISIS_CUTS', 'COOL'):                (+0.05, None, 'CPI cool — crisis cuts appropriate'),
    ('CRISIS_EASING', 'COOL'):              (+0.05, None, 'CPI cool — crisis easing working'),
    ('CRISIS_REFLATION', 'COOL'):           (+0.10, None, 'CPI cool — crisis reflation less inflationary'),

    # ── CPI WARM (2.5% - 3.5%) ────────────────────────────────
    # Inline CPI — no adjustment for most regimes.
    # Only matters when it disagrees with PPI direction.
    ('STAGFLATION_TRAPPED', 'WARM'):        (+0.00, None, 'CPI warm — trap persists'),
    ('STAGFLATION_LITE', 'WARM'):           (+0.00, None, 'CPI warm — status quo'),
    ('REFLATION', 'WARM'):                  (+0.00, None, 'CPI warm — reflation intact'),
    ('INFLATION_SHOCK', 'WARM'):            (-0.05, None, 'CPI warm but PPI rising — pipeline building'),

    # ── CPI HOT (above 3.5%) ──────────────────────────────────
    # Hot CPI amplifies every bad scenario. It's the primary signal.
    ('STAGFLATION_TRAPPED', 'HOT'):         (-0.10, 'UP', 'CPI HOT confirms trap — worst case, Fed can\'t move'),
    ('STAGFLATION_LITE', 'HOT'):            (-0.10, 'UP', 'CPI HOT upgrades to full stagflation'),
    ('SLOWFLATION', 'HOT'):                 (-0.05, 'UP', 'CPI HOT — slowflation tilts toward stagflation'),
    ('NEUTRAL', 'HOT'):                     (-0.10, 'UP', 'CPI HOT — neutral becomes stagflation risk'),
    ('NEUTRAL_STRONG', 'HOT'):              (-0.10, 'UP', 'CPI HOT — strong labor + hot CPI = trapped'),
    ('NEUTRAL_WEAK', 'HOT'):                (-0.10, 'UP', 'CPI HOT + weak labor — stagflation forming'),
    ('GOLDILOCKS', 'HOT'):                  (-0.15, 'UP', 'CPI HOT breaks goldilocks — labor can\'t save it'),
    ('GROWTH_SCARE', 'HOT'):                (-0.10, 'UP', 'CPI HOT — growth scare becomes stagflation scare'),
    ('DISINFLATION', 'HOT'):                (-0.15, 'UP', 'CPI HOT — disinflation narrative breaks'),
    ('WAIT_AND_SEE', 'HOT'):                (-0.10, 'UP', 'CPI HOT — Fed forced to keep holding'),
    ('REFLATION', 'HOT'):                   (-0.05, None, 'CPI HOT — reflation accelerating, works until 6%'),
    ('REFLATION_FRAGILE', 'HOT'):           (-0.10, 'UP', 'CPI HOT — fragile reflation breaks'),
    ('TIGHTENING', 'HOT'):                  (-0.05, None, 'CPI HOT — tightening continues'),
    ('TIGHTENING_LATE', 'HOT'):             (-0.05, 'UP', 'CPI HOT — late tightening extends'),
    ('POLICY_ERROR', 'HOT'):                (-0.10, 'UP', 'CPI HOT — policy error confirmed'),
    ('EASING', 'HOT'):                      (-0.10, 'UP', 'CPI HOT — easing cycle at risk'),
    ('EASING_SCARE', 'HOT'):                (-0.10, 'UP', 'CPI HOT — easing into inflation, bad'),
    ('CRISIS_CUTS', 'HOT'):                 (-0.05, None, 'CPI HOT — crisis cuts inflationary'),
    ('CRISIS_OVERRIDE', 'HOT'):             (-0.05, None, 'CPI HOT — crisis + hot CPI = confused Fed'),
    ('CRISIS_REFLATION', 'HOT'):            (-0.05, None, 'CPI HOT — crisis reflation inflates'),
    ('CRISIS_EASING', 'HOT'):               (-0.05, None, 'CPI HOT — crisis easing inflationary'),
    ('PRE_CRISIS', 'HOT'):                  (-0.05, 'UP', 'CPI HOT — pre-crisis + hot CPI = worse'),
    ('BREAKING', 'HOT'):                    (-0.05, None, 'CPI HOT — breaking + hot CPI = chaos'),
    ('INFLATION_SHOCK', 'HOT'):             (-0.05, None, 'CPI HOT — shock confirmed'),
    ('INFLATION_CRASH', 'HOT'):             (-0.05, None, 'CPI HOT — crash scenario confirmed'),
    ('STAGFLATION_CRASH', 'HOT'):           (-0.05, None, 'CPI HOT — crash confirmed'),
}


# ═══════════════════════════════════════════════════════════════
# REAL RATE MODIFIER — continuous adjustment
# ═══════════════════════════════════════════════════════════════
# Real rate = Fed funds rate - PPI YoY
# Positive = tight (restrictive), Negative = stimulative
#
# This is a severity adjustment, not a score adjustment.
# Tight real rates make bad scenarios worse (higher severity).
# Stimulative real rates make good scenarios better (lower severity).
#
# Historical context:
#   2021: Fed funds 0.25%, PPI 8% → real = -7.75% (max stimulative)
#   2022: Fed funds 4.5%, PPI 6% → real = -1.5% (still stimulative!)
#   2023: Fed funds 5.25%, PPI 1% → real = +4.25% (max tight)
#   2026: Fed funds 4.5%, PPI 4.9% → real = -0.4% (mildly stimulative)

REAL_RATE_SEVERITY = {
    # real_rate → (score_modifier, severity_shift)
    # Positive real rates above 2% = genuinely tight
    'VERY_TIGHT':   (-0.08, 'UP', 'Real rates >2% — genuinely restrictive'),
    # Positive real rates 0-2% = mildly tight
    'TIGHT':        (-0.03, None, 'Real rates positive — mildly restrictive'),
    'MILDLY_TIGHT': (+0.00, None, 'Real rates slightly positive — neutral'),
    # Negative real rates = stimulative even with high nominal rates
    'MILDLY_STIMULATIVE': (+0.02, None, 'Real rates slightly negative — stimulative'),
    'STIMULATIVE':  (+0.05, 'DOWN', 'Real rates negative — stimulative even with high PPI'),
    'VERY_STIMULATIVE': (+0.08, 'DOWN', 'Real rates deeply negative — max stimulative'),
    'UNKNOWN':      (+0.00, None, 'Real rate unknown'),
}


def _classify_real_rate_label(real_rate):
    """Classify real rate into severity buckets."""
    if real_rate is None:
        return 'UNKNOWN'
    if real_rate > 2.0:
        return 'VERY_TIGHT'
    elif real_rate > 0.5:
        return 'TIGHT'
    elif real_rate > 0.0:
        return 'MILDLY_TIGHT'
    elif real_rate > -1.0:
        return 'MILDLY_STIMULATIVE'
    elif real_rate > -3.0:
        return 'STIMULATIVE'
    else:
        return 'VERY_STIMULATIVE'


# ═══════════════════════════════════════════════════════════════
# ACCELERATION MODIFIER — PPI momentum
# ═══════════════════════════════════════════════════════════════
# PPI direction tells you WHERE. Acceleration tells you the MOMENTUM.
#
# RISING + ACCELERATING = getting worse faster (bearish amplifier)
# RISING + DECELERATING = peaking (potential buy signal — 2022 H2 playbook)
# FALLING + ACCELERATING = deflation accelerating (can be bearish — 2023)
# FALLING + DECELERATING = bottoming (early bullish signal — 2024)

ACCELERATION_MODIFIER = {
    # (ppi_direction, acceleration) → (score_adj, desc)
    ('RISING', 'ACCELERATING'):  (-0.08, 'PPI accelerating — inflation getting worse faster'),
    ('RISING', 'DECELERATING'):  (+0.06, 'PPI decelerating — peaking, potential pivot'),
    ('RISING', 'STABLE'):        (+0.00, 'PPI rising at steady pace'),
    ('FALLING', 'ACCELERATING'): (-0.03, 'PPI falling faster — deflation risk'),
    ('FALLING', 'DECELERATING'): (+0.04, 'PPI falling slower — bottoming signal'),
    ('FALLING', 'STABLE'):       (+0.00, 'PPI falling at steady pace'),
    ('FLAT', 'ACCELERATING'):    (-0.03, 'PPI was flat, now rising — inflection'),
    ('FLAT', 'DECELERATING'):    (+0.03, 'PPI was flat, now falling — inflection'),
    ('FLAT', 'STABLE'):          (+0.00, 'PPI flat and stable'),
    ('RISING', 'INSUFFICIENT'):  (+0.00, ''),
    ('FALLING', 'INSUFFICIENT'): (+0.00, ''),
    ('FLAT', 'INSUFFICIENT'):    (+0.00, ''),
}


# ═══════════════════════════════════════════════════════════════
# CLAIMS MODIFIER — labor context changes Fed reaction function
# ═══════════════════════════════════════════════════════════════
# v2: Claims now feed into the LABOR CONTEXT classification,
# which is IN the base matrix. This modifier handles the
# remaining nuance: claims trend direction × PPI direction.

CLAIMS_TREND_MODIFIER = {
    # (claims_trend, ppi_direction) → (score_adj, desc)
    ('RISING', 'RISING'):   (-0.05, 'Claims rising + PPI rising — stagflation tightening'),
    ('RISING', 'FALLING'):  (+0.05, 'Claims rising + PPI falling — labor weakening, Fed will cut'),
    ('RISING', 'FLAT'):     (+0.00, 'Claims rising + PPI flat — watch for recession'),
    ('FALLING', 'RISING'):  (-0.08, 'Claims falling + PPI rising — Fed trapped, labor strong'),
    ('FALLING', 'FALLING'): (+0.08, 'Claims falling + PPI falling — Goldilocks confirmed'),
    ('FALLING', 'FLAT'):    (+0.05, 'Claims falling + PPI flat — labor improving'),
    ('STABLE', 'RISING'):   (-0.03, 'Claims stable + PPI rising — inflation pressure'),
    ('STABLE', 'FALLING'):  (+0.03, 'Claims stable + PPI falling — disinflation'),
    ('STABLE', 'FLAT'):     (+0.00, 'Claims stable + PPI flat — neutral'),
}


# ═══════════════════════════════════════════════════════════════
# SURPRISE MODIFIER — consensus miss amplifies signal
# ═══════════════════════════════════════════════════════════════
# When PPI or CPI beats/misses consensus, the surprise amplifies
# the directional signal. Bigger surprise = bigger reaction.

def _compute_surprise_modifier(actual, expected, label='PPI'):
    """Compute score modifier from data surprise.

    Returns: (score_adj: float, desc: str)
    """
    if expected is None or actual is None:
        return 0.0, ''
    surprise = actual - expected
    if abs(surprise) < 0.2:
        return 0.0, f'{label} inline with consensus'
    elif surprise > 0.0:
        # Hotter than expected — bearish for risk
        adj = min(-surprise * 0.04, -0.12)  # max -12%
        return adj, f'{label} beat by +{surprise:.1f}% — hotter than expected'
    else:
        # Cooler than expected — bullish for risk
        adj = min(abs(surprise) * 0.04, 0.12)  # max +12%
        return adj, f'{label} miss by {surprise:.1f}% — cooler than expected'


# ═══════════════════════════════════════════════════════════════
# POSITIONING MULTIPLIER — final layer
# ═══════════════════════════════════════════════════════════════
# Crowded positioning amplifies the directional score.
# In bad regimes, crowded = worse. In good regimes, crowded = less upside.

POSITIONING_MULTIPLIER = {
    # (regime_severity, positioning) → score multiplier
    ('LOW', 'CROWDED'):     0.90,   # good regime but crowded — 90% of score
    ('LOW', 'NEUTRAL'):     1.00,   # full score
    ('MEDIUM', 'CROWDED'):  0.80,   # medium severity + crowded = 80%
    ('MEDIUM', 'NEUTRAL'):  1.00,
    ('HIGH', 'CROWDED'):    0.65,   # high severity + crowded = 65% (amplifies loss)
    ('HIGH', 'NEUTRAL'):    0.90,   # high severity but neutral = 90%
    ('CRITICAL', 'CROWDED'): 0.50,  # critical + crowded = half score (cascade risk)
    ('CRITICAL', 'NEUTRAL'): 0.80,  # critical but neutral = 80%
}


# ═══════════════════════════════════════════════════════════════
# SIZE MULTIPLIER — position sizing by severity
# ═══════════════════════════════════════════════════════════════

SIZE_BY_SEVERITY = {
    'LOW':      1.00,
    'MEDIUM':   0.75,
    'HIGH':     0.50,
    'CRITICAL': 0.30,
}


# ═══════════════════════════════════════════════════════════════
# MAIN SCORING FUNCTION
# ═══════════════════════════════════════════════════════════════

def score_m22_v2(ppi_yoy, ppi_prev_yoy=None, ppi_prev_prev_yoy=None, ppi_mom=None,
                 cpi_yoy=None, cpi_prev_yoy=None, cpi_expected=None,
                 ppi_expected=None,
                 fed_stance='HOLDING', fed_funds_rate=None,
                 ls_ratio=None, direction='LONG',
                 claims_classification=None, claims_trend=None,
                 unemployment_rate=None, sahm_triggered=False,
                 config=None):
    """Score the inflation regime using the expanded 5D model.

    Args:
        ppi_yoy: Latest PPI year-over-year % (e.g., 4.9)
        ppi_prev_yoy: Previous month's PPI YoY (for direction + acceleration)
        ppi_prev_prev_yoy: Two months ago PPI YoY (for acceleration)
        ppi_mom: PPI month-over-month % (fallback for direction)
        cpi_yoy: Latest CPI year-over-year % (co-equal signal)
        cpi_prev_yoy: Previous CPI YoY (for CPI trend)
        cpi_expected: CPI consensus expectation (for surprise)
        ppi_expected: PPI consensus expectation (for surprise)
        fed_stance: Fed stance string ('CUTTING', 'HOLDING', 'HIKING')
        fed_funds_rate: Current Fed funds rate (for real rate calc)
        ls_ratio: Long/short ratio from derivatives
        direction: Trade direction ('LONG' or 'SHORT')
        claims_classification: 'LOW', 'NORMAL', 'ELEVATED', 'SPIKE', 'CRISIS'
        claims_trend: 'RISING', 'FALLING', 'STABLE'
        unemployment_rate: Current unemployment rate %
        sahm_triggered: Whether Sahm Rule is triggered
        config: Config dict

    Returns:
        status: 'PASS', 'FAIL', 'VETO', or 'SKIP'
        score: 0.0-1.0
        details: dict with full regime breakdown
    """
    cfg = config or CONFIG

    if not cfg.get('M22_ENABLED', False):
        return 'SKIP', 0.5, {'regime': 'DISABLED'}

    # ── Step 1: Classify all dimensions ──
    ppi_dir = classify_ppi_direction(ppi_yoy, ppi_prev_yoy, ppi_mom)
    cpi_class = classify_cpi_confirmation(cpi_yoy, cpi_prev_yoy)
    fed = classify_fed_stance(fed_stance)
    acceleration = classify_ppi_acceleration(ppi_yoy, ppi_prev_yoy, ppi_prev_prev_yoy)

    # Labor context — from claims data (M23 cache) or explicit params
    if claims_classification is None:
        try:
            from src.modules.macro_utils import get_claims_trend
            ct = get_claims_trend()
            if ct:
                claims_classification = ct.get('classification', 'NORMAL')
                claims_trend = ct.get('trend', 'STABLE')
                unemployment_rate = ct.get('unemployment')
                sahm_triggered = ct.get('sahm_triggered', False)
        except Exception:
            claims_classification = 'NORMAL'
            claims_trend = 'STABLE'

    labor = classify_labor_context(claims_classification, unemployment_rate, sahm_triggered)

    ls_threshold = cfg.get('M22_LS_CROWDED_THRESHOLD', 2.0)
    pos = classify_positioning(ls_ratio, ls_threshold)

    # ── Step 2: Base regime lookup (PPI_dir × Fed × Labor) ──
    base_key = (ppi_dir, fed, labor)
    if base_key in REGIME_MATRIX_V2:
        base = REGIME_MATRIX_V2[base_key]
    else:
        base = {
            'score': 0.50, 'regime': 'UNKNOWN',
            'severity': 'MEDIUM',
            'desc': f'Unknown combination: {base_key}',
            'analog': 'N/A', 'expected_move': 'N/A',
        }

    score_raw = base['score']
    regime = base['regime']
    severity = base['severity']
    factors = [base['desc']]

    # ── Step 3: CPI overlay ──
    cpi_key = (regime, cpi_class)
    if cpi_key in CPI_OVERLAY:
        cpi_adj, sev_shift, cpi_desc = CPI_OVERLAY[cpi_key]
        score_raw = max(0.05, min(0.95, score_raw + cpi_adj))
        if cpi_desc:
            factors.append(cpi_desc)
        if sev_shift == 'UP':
            sev_order = ['LOW', 'MEDIUM', 'HIGH', 'CRITICAL']
            idx = sev_order.index(severity)
            severity = sev_order[min(idx + 1, 3)]
        elif sev_shift == 'DOWN':
            sev_order = ['LOW', 'MEDIUM', 'HIGH', 'CRITICAL']
            idx = sev_order.index(severity)
            severity = sev_order[max(idx - 1, 0)]

    # ── Step 4: Real rate modifier ──
    real_rate, real_rate_label = compute_real_rate(fed_funds_rate, ppi_yoy)
    if real_rate_label in REAL_RATE_SEVERITY:
        rr_adj, rr_sev_shift, rr_desc = REAL_RATE_SEVERITY[real_rate_label]
        score_raw = max(0.05, min(0.95, score_raw + rr_adj))
        if rr_desc:
            factors.append(f'Real rate {real_rate:+.1f}%: {rr_desc}')
        if rr_sev_shift == 'UP':
            sev_order = ['LOW', 'MEDIUM', 'HIGH', 'CRITICAL']
            idx = sev_order.index(severity)
            severity = sev_order[min(idx + 1, 3)]
        elif rr_sev_shift == 'DOWN':
            sev_order = ['LOW', 'MEDIUM', 'HIGH', 'CRITICAL']
            idx = sev_order.index(severity)
            severity = sev_order[max(idx - 1, 0)]

    # ── Step 5: Acceleration modifier ──
    accel_key = (ppi_dir, acceleration)
    if accel_key in ACCELERATION_MODIFIER:
        accel_adj, accel_desc = ACCELERATION_MODIFIER[accel_key]
        score_raw = max(0.05, min(0.95, score_raw + accel_adj))
        if accel_desc:
            factors.append(accel_desc)

    # ── Step 6: Claims trend modifier ──
    if claims_trend and ppi_dir:
        ct_key = (claims_trend, ppi_dir)
        if ct_key in CLAIMS_TREND_MODIFIER:
            ct_adj, ct_desc = CLAIMS_TREND_MODIFIER[ct_key]
            score_raw = max(0.05, min(0.95, score_raw + ct_adj))
            if ct_desc:
                factors.append(ct_desc)

    # ── Step 7: PPI surprise ──
    ppi_surp_adj, ppi_surp_desc = _compute_surprise_modifier(ppi_yoy, ppi_expected, 'PPI')
    if ppi_surp_adj != 0:
        score_raw = max(0.05, min(0.95, score_raw + ppi_surp_adj))
        factors.append(ppi_surp_desc)

    # ── Step 8: CPI surprise ──
    cpi_surp_adj, cpi_surp_desc = _compute_surprise_modifier(cpi_yoy, cpi_expected, 'CPI')
    if cpi_surp_adj != 0:
        score_raw = max(0.05, min(0.95, score_raw + cpi_surp_adj))
        factors.append(cpi_surp_desc)

    # ── Step 9: Claims shock modifier ──
    # Claims don't move markets alone, but extreme prints amplify the regime
    if claims_classification == 'SPIKE':
        score_raw = max(0.05, score_raw - 0.08)
        factors.append('Claims SPIKE — risk-off amplifier')
    elif claims_classification == 'CRISIS':
        score_raw = max(0.05, score_raw - 0.15)
        factors.append('Claims CRISIS — macro dominant, sell risk')
    elif claims_classification == 'LOW' and ppi_dir == 'FALLING':
        score_raw = min(0.95, score_raw + 0.05)
        factors.append('Claims LOW + PPI falling — Goldilocks confirmed')

    # ── Step 10: Time decay ──
    decay = compute_m22_decay(cfg)
    decay_mult = decay['decay_mult']
    score_raw = score_raw * decay_mult + 0.5 * (1.0 - decay_mult)

    # ── Step 11: Direction adjustment ──
    # For SHORT trades, bad regimes are good (inverted score)
    if direction == 'SHORT':
        score = 1.0 - score_raw
    else:
        score = score_raw

    score = max(0.0, min(1.0, score))

    # ── Step 12: Positioning multiplier ──
    pos_key = (severity, pos)
    pos_mult = POSITIONING_MULTIPLIER.get(pos_key, 1.0)
    score = score * pos_mult + 0.5 * (1.0 - pos_mult)  # pull toward neutral
    score = max(0.0, min(1.0, score))

    # ── Step 13: Status determination ──
    veto_threshold = cfg.get('M22_VETO_THRESHOLD', 0.20)
    if score_raw < veto_threshold and direction == 'LONG':
        status = 'VETO'
    elif score >= cfg.get('M22_FAIL_THRESHOLD', 0.35):
        status = 'PASS'
    else:
        status = 'FAIL'

    # ── Step 14: Size multiplier ──
    size_mult = SIZE_BY_SEVERITY.get(severity, 1.0)

    # ── Build details ──
    details = {
        # Core regime
        'regime': regime,
        'severity': severity,
        'score_raw': round(score_raw, 3),

        # Dimensions
        'ppi_direction': ppi_dir,
        'ppi_yoy': ppi_yoy,
        'ppi_prev_yoy': ppi_prev_yoy,
        'ppi_mom': ppi_mom,
        'ppi_acceleration': acceleration,

        'cpi_confirmation': cpi_class,
        'cpi_yoy': cpi_yoy,
        'cpi_prev_yoy': cpi_prev_yoy,

        'fed_stance': fed,
        'fed_funds_rate': fed_funds_rate,

        'labor_context': labor,
        'claims_classification': claims_classification,
        'claims_trend': claims_trend,
        'unemployment_rate': unemployment_rate,
        'sahm_triggered': sahm_triggered,

        'positioning': pos,
        'ls_ratio': ls_ratio,
        'ls_threshold': ls_threshold,

        # Modifiers
        'real_rate': real_rate,
        'real_rate_label': real_rate_label,
        'ppi_surprise_adj': round(ppi_surp_adj, 3),
        'cpi_surprise_adj': round(cpi_surp_adj, 3),

        # Outputs
        'direction': direction,
        'size_mult': round(size_mult, 2),
        'positioning_multiplier': round(pos_mult, 2),
        'decay_mult': decay_mult,
        'days_since_release': decay['days_since'],
        'last_release_date': decay['last_release'],
        'last_release_type': decay['last_type'],

        # Narrative
        'factors': factors,
        'analog': base.get('analog', 'N/A'),
        'expected_move': base.get('expected_move', 'N/A'),
        'description': base.get('desc', ''),
    }

    return status, score, details


# ═══════════════════════════════════════════════════════════════
# CONVENIENCE WRAPPER (config-driven, same interface as v1)
# ═══════════════════════════════════════════════════════════════

def score_m22(direction='LONG', ls_ratio=None, config=None):
    """Score inflation regime using config + live FRED data.

    Auto-fills PPI, CPI, Fed funds rate, and Fed stance from FRED cache
    when config values are missing or stale. Config values still take
    precedence when explicitly set (non-default).

    Live data source: data/fred/macro_cache.json
    Fetched by: scripts/fetch_fred_claims.py
    """
    cfg = config or CONFIG

    if not cfg.get('M22_ENABLED', False):
        return 'SKIP', 0.5, {'regime': 'DISABLED'}

    # ── Auto-fill from FRED live data ──
    fred_overrides = {}
    fred_data = None
    try:
        from src.utils.macro_loader import get_m22_overrides, load_macro_data
        fred_overrides = get_m22_overrides()
        fred_data = load_macro_data()
    except Exception:
        pass

    # Merge: FRED live data is primary (always fresh), config is fallback.
    # Consensus expectations (CPI_EXPECTED, PPI_EXPECTED) are config-only
    # since FRED doesn't have forward-looking data.
    def _get(key, default=None):
        """Get value from FRED live data, falling back to config."""
        val = fred_overrides.get(key)
        if val is not None:
            return val
        return cfg.get(key, default)

    ppi_yoy = _get('M22_PPI_YOY')
    if ppi_yoy is None:
        return 'SKIP', 0.5, {'regime': 'NO_DATA', 'reason': 'M22_PPI_YOY not set (no config, no FRED data)'}

    # Auto-fill claims/unemployment from FRED cache if not in config
    claims_classification = _get('M22_CLAIMS_CLASSIFICATION')
    claims_trend = _get('M22_CLAIMS_TREND')
    unemployment_rate = _get('M22_UNEMPLOYMENT_RATE')
    sahm_triggered = cfg.get('M22_SAHM_TRIGGERED', False)

    # If claims classification not set, try to infer from FRED claims data
    if claims_classification is None and fred_data:
        icsa_monthly = fred_data.get('icsa_monthly_avg', {})
        if not icsa_monthly:
            # Try loading from claims cache directly
            try:
                from src.utils.macro_loader import _load_cache
                cache = _load_cache()
                if cache:
                    icsa_monthly = cache.get('icsa', {}).get('monthly_avg', {})
            except Exception:
                pass
        if icsa_monthly:
            months = sorted(icsa_monthly.keys(), reverse=True)
            if months:
                latest_claims = icsa_monthly[months[0]]
                # Classify: <220K=LOW, 220-260K=NORMAL, 260-300K=ELEVATED, >300K=SPIKE
                if latest_claims < 220:
                    claims_classification = 'LOW'
                elif latest_claims < 260:
                    claims_classification = 'NORMAL'
                elif latest_claims < 300:
                    claims_classification = 'ELEVATED'
                else:
                    claims_classification = 'SPIKE'
                # Infer trend from 3-month average
                if len(months) >= 4:
                    recent_avg = sum(icsa_monthly[m] for m in months[:3]) / 3
                    older_avg = sum(icsa_monthly[m] for m in months[3:6]) / min(3, len(months) - 3)
                    if recent_avg > older_avg * 1.05:
                        claims_trend = 'RISING'
                    elif recent_avg < older_avg * 0.95:
                        claims_trend = 'FALLING'
                    else:
                        claims_trend = 'STABLE'

    # If unemployment not set, pull from FRED cache
    if unemployment_rate is None and fred_data:
        unrate = fred_data.get('unrate_monthly', {})
        if not unrate:
            try:
                from src.utils.macro_loader import _load_cache
                cache = _load_cache()
                if cache:
                    unrate = cache.get('unrate', {}).get('monthly', {})
            except Exception:
                pass
        if unrate:
            months = sorted(unrate.keys(), reverse=True)
            if months:
                unemployment_rate = unrate[months[0]]

    result = score_m22_v2(
        ppi_yoy=ppi_yoy,
        ppi_prev_yoy=_get('M22_PPI_PREV_YOY'),
        ppi_prev_prev_yoy=_get('M22_PPI_PREV_PREV_YOY'),
        ppi_mom=_get('M22_PPI_MOM'),
        cpi_yoy=_get('M22_CPI_YOY'),
        cpi_prev_yoy=_get('M22_CPI_PREV_YOY'),
        cpi_expected=cfg.get('M22_CPI_EXPECTED'),  # expectations: config only (not on FRED)
        ppi_expected=cfg.get('M22_PPI_EXPECTED'),   # expectations: config only (not on FRED)
        fed_stance=_get('M22_FED_STANCE', 'HOLDING'),
        fed_funds_rate=_get('M22_FED_FUNDS_RATE'),
        ls_ratio=ls_ratio,
        direction=direction,
        claims_classification=claims_classification,
        claims_trend=claims_trend,
        unemployment_rate=unemployment_rate,
        sahm_triggered=sahm_triggered,
        config=cfg,
    )

    # Tag result with data source for transparency
    if len(result) >= 3 and isinstance(result[2], dict):
        result[2]['_fred_live'] = bool(fred_overrides)
        result[2]['_fred_overrides'] = list(fred_overrides.keys()) if fred_overrides else []

    # 4-tuple: (status, score, details, size_mult)
    size_mult = result[2].get('size_mult', 1.0) if len(result) >= 3 and isinstance(result[2], dict) else 1.0
    return result[0], result[1], result[2], size_mult


# ═══════════════════════════════════════════════════════════════
# FORMATTER
# ═══════════════════════════════════════════════════════════════

def format_m22(details):
    """Format M22 v2 details for terminal output."""
    if not details or details.get('regime') in ('DISABLED', 'NO_DATA', 'UNKNOWN'):
        return ''

    regime = details.get('regime', '?')
    severity = details.get('severity', '?')
    score = details.get('score_raw', 0.5)
    size_mult = details.get('size_mult', 1.0)
    factors = details.get('factors', [])

    # Dimensions
    ppi_dir = details.get('ppi_direction', '?')
    ppi = details.get('ppi_yoy', 0)
    ppi_accel = details.get('ppi_acceleration', '?')
    cpi_class = details.get('cpi_confirmation', '?')
    cpi = details.get('cpi_yoy')
    fed = details.get('fed_stance', '?')
    labor = details.get('labor_context', '?')
    pos = details.get('positioning', '?')
    ls = details.get('ls_ratio', 0)
    real_rate = details.get('real_rate')
    real_label = details.get('real_rate_label', '?')
    analog = details.get('analog', '')
    expected = details.get('expected_move', '')
    decay_mult = details.get('decay_mult', 1.0)
    days_since = details.get('days_since_release', 0)

    severity_icons = {'LOW': '🟢', 'MEDIUM': '🟡', 'HIGH': '🟠', 'CRITICAL': '🔴'}
    icon = severity_icons.get(severity, '⚪')

    labor_icons = {'GOLDILOCKS': '🟢', 'NORMAL': '⚪', 'SOFTENING': '🟡', 'CRISIS': '🔴'}
    l_icon = labor_icons.get(labor, '⚪')

    cpi_icons = {'COOL': '🟢', 'WARM': '⚪', 'HOT': '🔴'}
    c_icon = cpi_icons.get(cpi_class, '⚪')

    accel_icons = {'ACCELERATING': '📈', 'DECELERATING': '📉', 'STABLE': '➡️', 'INSUFFICIENT': '—'}
    a_icon = accel_icons.get(ppi_accel, '—')

    pos_icons = {'CROWDED': '🔴', 'NEUTRAL': '⚪'}
    p_icon = pos_icons.get(pos, '⚪')

    lines = []
    lines.append(f"\n  {icon} M22 INFLATION REGIME v2: {regime}")
    lines.append(f"    PPI: {ppi:.1f}% ({ppi_dir}) {a_icon} {ppi_accel}  |  CPI: {cpi:.1f}% {c_icon} {cpi_class}")
    lines.append(f"    Fed: {fed}  |  Labor: {l_icon} {labor}  |  Pos: {p_icon} {pos} (L/S={ls:.2f})")
    if real_rate is not None:
        rr_icon = '🟢' if real_rate < 0 else '🔴' if real_rate > 1 else '⚪'
        lines.append(f"    Real Rate: {rr_icon} {real_rate:+.1f}% ({real_label})")
    lines.append(f"    Score: {score:.3f}  |  Severity: {severity}  |  Size: {size_mult:.2f}x")

    # Time decay
    if decay_mult < 1.0:
        decay_icon = '🟢' if decay_mult >= 0.85 else '🟡' if decay_mult >= 0.60 else '🟠' if decay_mult >= 0.40 else '🔴'
        last_type = details.get('last_release_type', '?')
        last_date = details.get('last_release_date', '')
        lines.append(f"    {decay_icon} Decay: {decay_mult:.2f}x  ({days_since}d since {last_type} {last_date})")

    # Claims context
    claims_class = details.get('claims_classification')
    if claims_class:
        cls_icons = {'LOW': '🟢', 'NORMAL': '⚪', 'ELEVATED': '🟡', 'SPIKE': '🟠', 'CRISIS': '🔴'}
        c_cl_icon = cls_icons.get(claims_class, '⚪')
        claims_trend = details.get('claims_trend', '?')
        unemp = details.get('unemployment_rate')
        sahm = details.get('sahm_triggered', False)
        lines.append(f"    Claims: {c_cl_icon} {claims_class} ({claims_trend})  |  "
                     f"Unemp: {unemp or '?'}%  |  Sahm: {'🔴 TRIGGERED' if sahm else '🟢 ok'}")

    # Factors
    for f in factors:
        lines.append(f"    • {f}")

    # Analog and expected
    if analog and analog != 'N/A':
        lines.append(f"    📊 Analog: {analog}")
    if expected and expected != 'N/A':
        lines.append(f"    📈 Expected: {expected}")

    return '\n'.join(lines)

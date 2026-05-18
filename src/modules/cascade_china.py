"""
China Macro Cascade — GDP → NBS PMI → Caixin PMI → CPI/PPI → PBOC LPR

Models the Chinese economic data chain and its transmission to crypto.
China data matters because:
  1. China is the 2nd largest economy — GDP sets growth narrative
  2. NBS PMI is the official signal (monthly, government-published)
  3. Caixin PMI is the independent confirmation (private survey)
  4. CPI/PPI sets inflation context for PBOC policy
  5. PBOC LPR is the policy payoff (rate decision)

Release sequence:
  1. GDP (STRUCTURAL, quarterly) — sets growth narrative
  2. NBS PMI (PRIMARY, last day of month ~09:00 CST) — official manufacturing signal
  3. Caixin PMI (CONFIRMATION, ~1st business day) — independent confirmation
  4. China CPI/PPI (CONFIRMATION, ~10th-12th) — inflation context
  5. PBOC LPR (POLICY, 20th of month) — rate decision payoff

Thesis:
  NBS PMI ≥51 + Caixin confirms = China expansion → risk-on → ETH rallies
  Both contracting (<50) = China contraction → risk-off → ETH dumps
  PBOC cut + weak data = stimulus expectations → ETH rallies
  PBOC hold + weak data = disappointment → ETH dumps

Usage:
    from src.modules.cascade_china import score_china_cascade, format_china_cascade
"""

from datetime import datetime
from src.modules.cascade_engine import (
    CascadeEngine, CascadeRelease, format_cascade,
)
from src.modules.m54_china_gdp import CHINA_GDP_RELEASES
from src.modules.m24_nbs_pmi import NBS_PMI_RELEASES
from src.modules.m25_caixin_pmi import CAIXIN_RELEASES
from src.modules.m30_china_cpi_ppi import CHINA_RELEASES
from src.modules.m35_pboc_lpr import PBOC_LPR_RELEASES
from src.modules.m_china_activity import (
    CHINA_ACTIVITY_RELEASES, CHINA_ACTIVITY_DATES,
    classify_activity_bundle, score_china_activity, format_china_activity,
)


# ═══════════════════════════════════════════════════════════════
# SCHEDULE DATES
# ═══════════════════════════════════════════════════════════════

CHINA_GDP_DATES = set(CHINA_GDP_RELEASES.keys())
NBS_PMI_DATES = set(NBS_PMI_RELEASES.keys())
# Caixin is a list of tuples — extract dates
CAIXIN_DATES = set(r[0] for r in CAIXIN_RELEASES)
CHINA_CPI_PPI_DATES = set(CHINA_RELEASES.keys())
PBOC_LPR_DATES = set(PBOC_LPR_RELEASES.keys())


# ═══════════════════════════════════════════════════════════════
# SIGNAL CLASSIFIERS
# ═══════════════════════════════════════════════════════════════

def _classify_nbs_signal(data: dict) -> str:
    """Classify NBS PMI signal."""
    mfg = data.get('mfg', 50.0)
    svc = data.get('services', 50.0)
    if mfg >= 51.0 and svc >= 51.0:
        return 'BOTH_EXPANDING'
    elif mfg >= 50.0 and svc >= 50.0:
        return 'EXPANDING'
    elif mfg < 49.0 and svc < 49.0:
        return 'BOTH_CONTRACTING'
    elif mfg < 50.0:
        return 'MFG_CONTRACTING'
    elif svc < 50.0:
        return 'SVC_CONTRACTING'
    return 'MIXED'


def _classify_caixin_signal(data: dict) -> str:
    """Classify Caixin PMI signal."""
    pmi = data.get('pmi', 50.0)
    if pmi >= 51.5:
        return 'STRONG'
    elif pmi >= 50.0:
        return 'EXPANDING'
    elif pmi >= 49.0:
        return 'WEAK'
    return 'CONTRACTING'


def _classify_china_inflation(data: dict) -> str:
    """Classify China CPI/PPI signal."""
    cpi = data.get('cpi', 2.0)
    ppi = data.get('ppi', 0.0)
    if cpi >= 3.0 and ppi >= 3.0:
        return 'BOTH_HOT'
    elif cpi <= 0.5 and ppi <= -2.0:
        return 'DEFLATION'
    elif ppi >= 3.0:
        return 'PPI_HOT'
    elif ppi <= -2.0:
        return 'PPI_DEFLATION'
    return 'NEUTRAL'


def _classify_pboc_signal(data: dict) -> str:
    """Classify PBOC LPR signal."""
    action = data.get('action', 'HOLD')
    if action == 'CUT':
        return 'CUT'
    elif action == 'HIKE':
        return 'HIKE'
    return 'HOLD'


def _classify_activity_signal(data: dict) -> str:
    """Classify China activity data bundle signal.

    Uses the composite classifier from m_china_activity.
    Maps the composite to simplified cascade signal names.
    """
    result = classify_activity_bundle(data)
    composite = result.get('composite', 'PENDING')

    # Map to simplified signals for cascade confirmation matrix
    mapping = {
        'STRONG_BEAT': 'STRONG_BEAT',
        'BEAT': 'BEAT',
        'MILD_BEAT': 'BEAT',
        'INLINE': 'INLINE',
        'MIXED': 'MIXED',
        'MILD_MISS': 'MISS',
        'MISS': 'MISS',
        'BIG_MISS': 'BIG_MISS',
        'PROPERTY_CRISIS': 'CRISIS',
        'PENDING': 'PENDING',
    }
    return mapping.get(composite, 'MIXED')


# ═══════════════════════════════════════════════════════════════
# CONFIRMATION MATRIX
# ═══════════════════════════════════════════════════════════════

CHINA_CONFIRMATION_MATRIX = {
    # NBS + Caixin agreement
    ('BOTH_EXPANDING', 'STRONG'):       (+1.50, +0.20, 'Both PMIs strong — China booming'),
    ('BOTH_EXPANDING', 'EXPANDING'):    (+0.80, +0.10, 'Both expanding — solid growth'),
    ('EXPANDING', 'EXPANDING'):         (+0.40, +0.05, 'Both above 50 — modest growth'),
    ('BOTH_CONTRACTING', 'CONTRACTING'): (-1.50, +0.20, 'Both contracting — China slowdown confirmed'),
    ('MFG_CONTRACTING', 'CONTRACTING'): (-0.80, +0.10, 'Mfg contraction confirmed'),
    ('BOTH_CONTRACTING', 'WEAK'):       (-1.00, +0.10, 'Both weak — contraction imminent'),

    # NBS + Caixin disagreement = noise
    ('BOTH_EXPANDING', 'CONTRACTING'):  (+0.00, -0.15, 'PMIs disagree — conflicting signals'),
    ('BOTH_CONTRACTING', 'STRONG'):     (+0.00, -0.15, 'PMIs disagree — conflicting signals'),
    ('EXPANDING', 'CONTRACTING'):       (-0.20, -0.10, 'Caixin contradicts NBS — caution'),
    ('MFG_CONTRACTING', 'STRONG'):      (+0.00, -0.10, 'Caixin contradicts NBS — mixed'),
}

# China inflation context modifiers
# Activity data confirmation (NBS PMI + Activity bundle)
ACTIVITY_CONFIRMATION = {
    # PMI expanding + activity data
    ('EXPANDING', 'STRONG_BEAT'):  (+0.80, +0.10, 'PMI expanding + activity strong — confirmed recovery'),
    ('EXPANDING', 'BEAT'):         (+0.40, +0.05, 'PMI expanding + activity beat — growth improving'),
    ('EXPANDING', 'INLINE'):       (+0.10, +0.00, 'PMI expanding + activity inline — stable'),
    ('EXPANDING', 'MISS'):         (-0.30, -0.05, 'PMI expanding but activity missed — PMI may be misleading'),
    ('EXPANDING', 'BIG_MISS'):     (-0.60, -0.10, 'PMI expanding but activity collapsing — divergence warning'),
    ('EXPANDING', 'CRISIS'):       (-1.00, -0.15, 'PMI expanding but property crisis — PMI misleading'),
    # PMI contracting + activity data
    ('BOTH_CONTRACTING', 'BIG_MISS'): (-1.20, +0.15, 'PMI + activity both collapsing — confirmed recession'),
    ('BOTH_CONTRACTING', 'MISS'):     (-0.80, +0.10, 'PMI contracting + activity miss — weakness confirmed'),
    ('BOTH_CONTRACTING', 'BEAT'):     (+0.20, -0.05, 'PMI contracting but activity beat — mixed signals'),
    ('MFG_CONTRACTING', 'MISS'):      (-0.60, +0.05, 'Mfg contracting + activity miss — broad weakness'),
    # Mixed
    ('MIXED', 'BIG_MISS'):        (-0.50, +0.00, 'Mixed PMI + activity big miss — downside risk'),
    ('MIXED', 'BEAT'):            (+0.30, +0.00, 'Mixed PMI + activity beat — upside surprise'),
}

CHINA_INFLATION_CONTEXT = {
    'BOTH_HOT':     (+0.30, 'Both hot — PBOC may tighten'),
    'PPI_HOT':      (+0.20, 'PPI hot — industrial demand strong'),
    'DEFLATION':    (-0.80, 'Deflation — PBOC must cut, stimulus coming'),
    'PPI_DEFLATION': (-0.50, 'PPI deflation — industrial weakness'),
    'NEUTRAL':      (+0.00, 'Inflation neutral'),
}

# PBOC policy modifiers
PBOC_POLICY_CONTEXT = {
    'CUT':   (+0.60, 'PBOC cut — stimulus, risk-on'),
    'HIKE':  (-0.80, 'PBOC hike — tightening, risk-off'),
    'HOLD':  (+0.00, 'PBOC hold — no signal'),
}

# Regime sensitivity
CHINA_REGIME_SENSITIVITY = {
    'TIGHTENING':      0.70,
    'EASING':          0.60,
    'CRISIS_RECOVERY': 0.85,
    'BULL':            0.80,
    'BEAR':            1.10,
    'RECOVERY':        0.75,
    'ACCELERATION':    0.70,
    'STAGFLATION':     0.90,
    'STAGFLATION_HOT': 1.00,
}


# ═══════════════════════════════════════════════════════════════
# BUILD
# ═══════════════════════════════════════════════════════════════

def _build_china_cascade() -> CascadeEngine:
    return CascadeEngine(
        name='CHINA_MACRO',
        description='China Macro Chain: GDP(structural) → PMI(primary) → Caixin(confirm) → Activity(real economy) → CPI/PPI(inflation) → PBOC(policy)',
        releases=[
            CascadeRelease('GDP', CHINA_GDP_DATES, 0.10, 'STRUCTURAL', 'M54_ENABLED',
                           release_hour_utc=3, release_minute_utc=0),
            CascadeRelease('NBS_PMI', NBS_PMI_DATES, 0.30, 'PRIMARY', 'M24_ENABLED',
                           release_hour_utc=1, release_minute_utc=0,
                           signal_classifier=_classify_nbs_signal),
            CascadeRelease('CAIXIN_PMI', CAIXIN_DATES, 0.25, 'CONFIRMATION', 'M25_ENABLED',
                           release_hour_utc=1, release_minute_utc=15,
                           signal_classifier=_classify_caixin_signal),
            CascadeRelease('CHINA_CPI_PPI', CHINA_CPI_PPI_DATES, 0.10, 'CONFIRMATION', 'M30_ENABLED',
                           release_hour_utc=1, release_minute_utc=30,
                           signal_classifier=_classify_china_inflation),
            CascadeRelease('CHINA_ACTIVITY', CHINA_ACTIVITY_DATES, 0.15, 'CONFIRMATION', 'M_CHINA_ACTIVITY_ENABLED',
                           release_hour_utc=2, release_minute_utc=0,
                           signal_classifier=_classify_activity_signal),
            CascadeRelease('PBOC_LPR', PBOC_LPR_DATES, 0.20, 'POLICY', 'M35_ENABLED',
                           release_hour_utc=1, release_minute_utc=30,
                           signal_classifier=_classify_pboc_signal),
        ],
        confirmation_matrix=CHINA_CONFIRMATION_MATRIX,
        regime_sensitivity=CHINA_REGIME_SENSITIVITY,
    )


_CHINA_CASCADE = None

def _get_cascade():
    global _CHINA_CASCADE
    if _CHINA_CASCADE is None:
        _CHINA_CASCADE = _build_china_cascade()
    return _CHINA_CASCADE


def score_china_cascade(df_15m, current_time=None, config=None, regime='UNKNOWN',
                        release_data_map=None):
    """Score the China Macro cascade."""
    cascade = _get_cascade()
    if current_time is None:
        current_time = datetime.utcnow()

    status, score, details, decay = cascade.score(
        df_15m, current_time, config, regime, release_data_map)

    if status == 'SKIP':
        return status, score, details, decay

    # Build score reason
    result = details.get('result', {})
    steps = details.get('steps', [])
    combined = result.get('combined_signal', '?')
    reason_parts = [f'CHINA_MACRO cascade: {combined}']
    for step in steps:
        if step.get('signal') and step.get('signal') not in ('PENDING', 'NEUTRAL'):
            reason_parts.append(f'{step["release"]}={step["signal"]}')
    if decay < 1.0:
        reason_parts.append(f'decay={decay:.2f}')
    details['score_reason'] = ', '.join(reason_parts)

    return status, score, details, decay


def format_china_cascade(details: dict) -> str:
    """Format China Macro cascade for terminal output."""
    if not details:
        return ''
    output = format_cascade(details)
    if not output:
        return ''

    lines = [output]

    # NBS PMI sub-details
    for step in details.get('steps', []):
        if step.get('release') == 'NBS_PMI':
            nbs = step.get('nbs_data', {})
            if nbs:
                mfg = nbs.get('mfg', '?')
                svc = nbs.get('services', '?')
                lines.append(f"    📊 NBS: Mfg={mfg}  Services={svc}")
        elif step.get('release') == 'CHINA_ACTIVITY':
            # Format activity data bundle
            activity_data = CHINA_ACTIVITY_RELEASES.get(step.get('date', ''), {})
            if activity_data:
                act_details = None
                classification = classify_activity_bundle(activity_data)
                act_details = {
                    'composite': classification.get('composite', '?'),
                    'description': '',
                }
                from src.modules.m_china_activity import ACTIVITY_SIGNAL_MAP
                _, _, desc = ACTIVITY_SIGNAL_MAP.get(
                    classification.get('composite', 'PENDING'), (0, '', ''))
                act_details['description'] = desc
                act_out = format_china_activity(activity_data, act_details)
                if act_out:
                    lines.append(act_out)
        elif step.get('release') == 'PBOC_LPR':
            signal = step.get('signal', '?')
            if signal == 'CUT':
                lines.append(f"    💡 PBOC cut — stimulus expected, risk-on for crypto")
            elif signal == 'HIKE':
                lines.append(f"    ⚠️ PBOC hike — tightening, risk-off")

    return '\n'.join(lines)

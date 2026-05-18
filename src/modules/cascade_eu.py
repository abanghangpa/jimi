"""
EU Macro Cascade — EZ GDP → EZ PMI → Germany CPI → EZ CPI → IFO → ECB Rate

Models the Eurozone economic data chain and its transmission to crypto.

Release sequence:
  1. EZ GDP (STRUCTURAL, quarterly) — growth narrative
  2. EZ PMI (PRIMARY, ~1st business day) — manufacturing/services health
  3. Germany CPI (LEADING, ~12th-15th) — early inflation signal for EZ
  4. EZ CPI (CONFIRMATION, ~17th-18th) — ECB's target metric
  5. IFO (CONFIRMATION, ~25th) — German business climate
  6. ECB Rate (POLICY, ~every 6 weeks) — rate decision payoff

Thesis:
  EZ PMI expanding + EZ CPI falling → ECB can cut → risk-on → ETH rallies
  EZ PMI contracting + EZ CPI rising → ECB trapped → risk-off → ETH dumps
  Germany CPI leads EZ CPI by 3-5 days — early signal for ECB decision
  IFO confirms or denies the PMI picture (German economy = EU engine)

Usage:
    from src.modules.cascade_eu import score_eu_cascade, format_eu_cascade
"""

from datetime import datetime
from src.modules.cascade_engine import CascadeEngine, CascadeRelease, format_cascade
from src.modules.m42_ez_gdp import EZ_GDP_RELEASES
from src.modules.m26_ez_pmi import EZ_PMI_RELEASES
from src.modules.m40_germany_cpi import GERMANY_CPI_RELEASES
from src.modules.m41_ez_cpi import EZ_CPI_RELEASES
from src.modules.m38_ifo import IFO_RELEASES
from src.modules.m48_ecb_rate import ECB_RELEASES

# Schedule dates
EZ_GDP_DATES = set(EZ_GDP_RELEASES.keys())
EZ_PMI_DATES = set(EZ_PMI_RELEASES.keys())
GERMANY_CPI_DATES = set(GERMANY_CPI_RELEASES.keys())
EZ_CPI_DATES = set(EZ_CPI_RELEASES.keys())
IFO_DATES = set(IFO_RELEASES.keys())
ECB_DATES = set(ECB_RELEASES.keys())


def _classify_ez_pmi(data: dict) -> str:
    mfg = data.get('mfg', 50.0)
    svc = data.get('services', 50.0)
    if mfg >= 51.0 and svc >= 51.0: return 'BOTH_STRONG'
    if mfg >= 50.0 and svc >= 50.0: return 'EXPANDING'
    if mfg < 49.0 and svc < 49.0: return 'BOTH_CONTRACTING'
    if mfg < 50.0: return 'MFG_CONTRACTING'
    return 'MIXED'

def _classify_germany_cpi(data: dict) -> str:
    yoy = data.get('yoy', 2.0)
    if yoy >= 3.5: return 'HOT'
    if yoy >= 2.5: return 'WARM'
    if yoy <= 1.0: return 'COOL'
    return 'TARGET'

def _classify_ez_cpi(data: dict) -> str:
    yoy = data.get('yoy', 2.0)
    if yoy >= 3.5: return 'HOT'
    if yoy >= 2.5: return 'WARM'
    if yoy <= 1.0: return 'COOL'
    return 'TARGET'

def _classify_ifo(data: dict) -> str:
    climate = data.get('climate', 90.0)
    if climate >= 95.0: return 'OPTIMISTIC'
    if climate >= 90.0: return 'NEUTRAL'
    if climate >= 85.0: return 'PESSIMISTIC'
    return 'VERY_PESSIMISTIC'

def _classify_ecb(data: dict) -> str:
    action = data.get('action', 'HOLD')
    return action


EU_CONFIRMATION_MATRIX = {
    # EZ PMI + EZ CPI
    ('EXPANDING', 'COOL'):    (+0.80, +0.10, 'Growth + low inflation — ECB can cut'),
    ('EXPANDING', 'TARGET'):  (+0.40, +0.05, 'Growth + target inflation — stable'),
    ('EXPANDING', 'HOT'):     (-0.20, +0.00, 'Growth but inflation rising — ECB may hold'),
    ('BOTH_CONTRACTING', 'HOT'): (-1.20, +0.15, 'Stagflation — ECB trapped'),
    ('BOTH_CONTRACTING', 'COOL'): (-0.50, +0.05, 'Contraction + low inflation — ECB will cut'),
    ('MFG_CONTRACTING', 'TARGET'): (-0.30, +0.00, 'Mfg weakness — mild headwind'),
    # EZ PMI + IFO agreement
    ('EXPANDING', 'OPTIMISTIC'): (+0.60, +0.10, 'PMI + IFO both positive — confirmed expansion'),
    ('BOTH_CONTRACTING', 'VERY_PESSIMISTIC'): (-1.00, +0.15, 'PMI + IFO both negative — confirmed contraction'),
    ('EXPANDING', 'VERY_PESSIMISTIC'): (+0.00, -0.10, 'PMI vs IFO disagree — uncertainty'),
}

EU_INFLATION_CONTEXT = {
    'HOT':    (-0.30, 'EZ CPI hot — ECB hawkish pressure'),
    'WARM':   (+0.00, 'EZ CPI warm — neutral'),
    'TARGET': (+0.20, 'EZ CPI at target — ECB can be patient'),
    'COOL':   (+0.40, 'EZ CPI cool — ECB dovish'),
}

ECB_POLICY_CONTEXT = {
    'CUT':  (+0.50, 'ECB cut — risk-on for EU assets'),
    'HIKE': (-0.70, 'ECB hike — risk-off'),
    'HOLD': (+0.00, 'ECB hold — no signal'),
}

EU_REGIME_SENSITIVITY = {
    'TIGHTENING': 0.65, 'EASING': 0.55, 'CRISIS_RECOVERY': 0.80,
    'BULL': 0.75, 'BEAR': 1.10, 'RECOVERY': 0.70,
    'ACCELERATION': 0.65, 'STAGFLATION': 0.95, 'STAGFLATION_HOT': 1.05,
}


def _build_eu_cascade() -> CascadeEngine:
    return CascadeEngine(
        name='EU_MACRO',
        description='EU Macro Chain: GDP(structural) → PMI(primary) → DE CPI(leading) → EZ CPI(confirm) → IFO(confirm) → ECB(policy)',
        releases=[
            CascadeRelease('EZ_GDP', EZ_GDP_DATES, 0.08, 'STRUCTURAL', 'M42_ENABLED',
                           release_hour_utc=9, release_minute_utc=0),
            CascadeRelease('EZ_PMI', EZ_PMI_DATES, 0.25, 'PRIMARY', 'M26_ENABLED',
                           release_hour_utc=8, release_minute_utc=0,
                           signal_classifier=_classify_ez_pmi),
            CascadeRelease('GERMANY_CPI', GERMANY_CPI_DATES, 0.12, 'LEADING', 'M40_ENABLED',
                           release_hour_utc=13, release_minute_utc=0,
                           signal_classifier=_classify_germany_cpi),
            CascadeRelease('EZ_CPI', EZ_CPI_DATES, 0.20, 'CONFIRMATION', 'M41_ENABLED',
                           release_hour_utc=10, release_minute_utc=0,
                           signal_classifier=_classify_ez_cpi),
            CascadeRelease('IFO', IFO_DATES, 0.10, 'CONFIRMATION', 'M38_ENABLED',
                           release_hour_utc=8, release_minute_utc=0,
                           signal_classifier=_classify_ifo),
            CascadeRelease('ECB', ECB_DATES, 0.25, 'POLICY', 'M48_ENABLED',
                           release_hour_utc=12, release_minute_utc=45,
                           signal_classifier=_classify_ecb),
        ],
        confirmation_matrix=EU_CONFIRMATION_MATRIX,
        regime_sensitivity=EU_REGIME_SENSITIVITY,
    )

_EU_CASCADE = None
def _get_cascade():
    global _EU_CASCADE
    if _EU_CASCADE is None: _EU_CASCADE = _build_eu_cascade()
    return _EU_CASCADE


def score_eu_cascade(df_15m, current_time=None, config=None, regime='UNKNOWN',
                     release_data_map=None):
    cascade = _get_cascade()
    if current_time is None: current_time = datetime.utcnow()
    status, score, details, decay = cascade.score(df_15m, current_time, config, regime, release_data_map)
    if status == 'SKIP': return status, score, details, decay
    result = details.get('result', {})
    steps = details.get('steps', [])
    reason_parts = [f'EU_MACRO cascade: {result.get("combined_signal", "?")}']
    for step in steps:
        if step.get('signal') and step.get('signal') not in ('PENDING', 'NEUTRAL'):
            reason_parts.append(f'{step["release"]}={step["signal"]}')
    if decay < 1.0: reason_parts.append(f'decay={decay:.2f}')
    details['score_reason'] = ', '.join(reason_parts)
    return status, score, details, decay


def format_eu_cascade(details: dict) -> str:
    if not details: return ''
    output = format_cascade(details)
    return output if output else ''

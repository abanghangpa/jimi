"""
UK Macro Cascade — UK GDP → UK CPI → UK Wages → BoE Rate

Models the UK economic data chain and its transmission to crypto.

Release sequence:
  1. UK GDP (STRUCTURAL, monthly ~13th) — growth narrative
  2. UK CPI (PRIMARY, ~17th-20th) — inflation signal for BoE
  3. UK Wages (CONFIRMATION, ~15th-18th) — wage-price spiral risk
  4. BoE Rate (POLICY, ~every 6 weeks) — rate decision payoff

Thesis:
  UK CPI hot + wages rising → BoE must hike → GBP strong → ETH pressure
  UK CPI cooling + wages moderating → BoE can cut → risk-on
  UK GDP weak + CPI hot → stagflation → worst case for risk assets

Usage:
    from src.modules.cascade_uk import score_uk_cascade, format_uk_cascade
"""

from datetime import datetime
from src.modules.cascade_engine import CascadeEngine, CascadeRelease, format_cascade
from src.modules.m51_uk_gdp_monthly import UK_GDP_RELEASES
from src.modules.m31_uk_cpi import UK_RELEASES
from src.modules.m32_uk_wages import UK_WAGES_RELEASES
from src.modules.m49_boe_rate import BOE_RELEASES

UK_GDP_DATES = set(UK_GDP_RELEASES.keys())
UK_CPI_DATES = set(UK_RELEASES.keys())
UK_WAGES_DATES = set(UK_WAGES_RELEASES.keys())
BOE_DATES = set(BOE_RELEASES.keys())


def _classify_uk_cpi(data: dict) -> str:
    yoy = data.get('yoy', 2.0)
    if yoy >= 4.0: return 'VERY_HOT'
    if yoy >= 3.0: return 'HOT'
    if yoy >= 2.0: return 'TARGET'
    if yoy <= 1.0: return 'COOL'
    return 'WARM'

def _classify_uk_wages(data: dict) -> str:
    growth = data.get('growth', 4.0)
    if growth >= 6.0: return 'SURGING'
    if growth >= 5.0: return 'STRONG'
    if growth >= 4.0: return 'MODERATE'
    return 'WEAK'

def _classify_boe(data: dict) -> str:
    return data.get('action', 'HOLD')


UK_CONFIRMATION_MATRIX = {
    ('VERY_HOT', 'SURGING'):    (-1.20, +0.15, 'Inflation + wages surging — BoE forced to hike'),
    ('VERY_HOT', 'STRONG'):     (-0.80, +0.10, 'Inflation hot + wages strong — BoE hawkish'),
    ('HOT', 'STRONG'):          (-0.60, +0.10, 'Inflation + wages above target — BoE pressure'),
    ('HOT', 'MODERATE'):        (-0.30, +0.00, 'Inflation hot but wages moderating — mixed'),
    ('TARGET', 'MODERATE'):     (+0.30, +0.05, 'Inflation at target + wages moderate — BoE can hold'),
    ('TARGET', 'WEAK'):         (+0.50, +0.05, 'Inflation target + wages weak — BoE can cut'),
    ('COOL', 'WEAK'):           (+0.80, +0.10, 'Inflation cool + wages weak — BoE dovish'),
    ('COOL', 'MODERATE'):       (+0.40, +0.05, 'Inflation cool — BoE has room'),
}

UK_INFLATION_CONTEXT = {
    'VERY_HOT': (-0.50, 'UK CPI very hot — BoE trapped'),
    'HOT':      (-0.30, 'UK CPI hot — BoE hawkish'),
    'TARGET':   (+0.20, 'UK CPI at target — BoE patient'),
    'WARM':     (+0.00, 'UK CPI warm — neutral'),
    'COOL':     (+0.40, 'UK CPI cool — BoE dovish'),
}

BOE_POLICY_CONTEXT = {
    'CUT':  (+0.50, 'BoE cut — risk-on'),
    'HIKE': (-0.60, 'BoE hike — risk-off'),
    'HOLD': (+0.00, 'BoE hold — no signal'),
}

UK_REGIME_SENSITIVITY = {
    'TIGHTENING': 0.65, 'EASING': 0.55, 'CRISIS_RECOVERY': 0.80,
    'BULL': 0.75, 'BEAR': 1.10, 'RECOVERY': 0.70,
    'ACCELERATION': 0.65, 'STAGFLATION': 0.95, 'STAGFLATION_HOT': 1.05,
}


def _build_uk_cascade() -> CascadeEngine:
    return CascadeEngine(
        name='UK_MACRO',
        description='UK Macro Chain: GDP(structural) → CPI(primary) → Wages(confirm) → BoE(policy)',
        releases=[
            CascadeRelease('UK_GDP', UK_GDP_DATES, 0.10, 'STRUCTURAL', 'M51_ENABLED',
                           release_hour_utc=7, release_minute_utc=0),
            CascadeRelease('UK_CPI', UK_CPI_DATES, 0.35, 'PRIMARY', 'M31_ENABLED',
                           release_hour_utc=7, release_minute_utc=0,
                           signal_classifier=_classify_uk_cpi),
            CascadeRelease('UK_WAGES', UK_WAGES_DATES, 0.20, 'CONFIRMATION', 'M32_ENABLED',
                           release_hour_utc=7, release_minute_utc=0,
                           signal_classifier=_classify_uk_wages),
            CascadeRelease('BOE', BOE_DATES, 0.35, 'POLICY', 'M49_ENABLED',
                           release_hour_utc=11, release_minute_utc=0,
                           signal_classifier=_classify_boe),
        ],
        confirmation_matrix=UK_CONFIRMATION_MATRIX,
        regime_sensitivity=UK_REGIME_SENSITIVITY,
    )

_UK_CASCADE = None
def _get_cascade():
    global _UK_CASCADE
    if _UK_CASCADE is None: _UK_CASCADE = _build_uk_cascade()
    return _UK_CASCADE


def score_uk_cascade(df_15m, current_time=None, config=None, regime='UNKNOWN',
                     release_data_map=None):
    cascade = _get_cascade()
    if current_time is None: current_time = datetime.utcnow()
    status, score, details, decay = cascade.score(df_15m, current_time, config, regime, release_data_map)
    if status == 'SKIP': return status, score, details, decay
    result = details.get('result', {})
    steps = details.get('steps', [])
    reason_parts = [f'UK_MACRO cascade: {result.get("combined_signal", "?")}']
    for step in steps:
        if step.get('signal') and step.get('signal') not in ('PENDING', 'NEUTRAL'):
            reason_parts.append(f'{step["release"]}={step["signal"]}')
    if decay < 1.0: reason_parts.append(f'decay={decay:.2f}')
    details['score_reason'] = ', '.join(reason_parts)
    return status, score, details, decay


def format_uk_cascade(details: dict) -> str:
    if not details: return ''
    return format_cascade(details) or ''

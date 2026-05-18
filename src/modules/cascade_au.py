"""
Australia Macro Cascade — AU CPI → RBA Rate

Models the Australian economic data chain. Australia matters for crypto
through risk sentiment and commodity currency dynamics.

Release sequence:
  1. AU CPI (PRIMARY, quarterly ~25th-30th) — inflation signal for RBA
  2. RBA Rate (POLICY, ~1st Tuesday every month) — rate decision payoff

Thesis:
  AU CPI rising + RBA hike → AUD strengthens → risk-off pressure
  AU CPI cooling + RBA hold/cut → risk-on for APAC assets
  RBA decisions affect APAC session sentiment

Usage:
    from src.modules.cascade_au import score_au_cascade, format_au_cascade
"""

from datetime import datetime
from src.modules.cascade_engine import CascadeEngine, CascadeRelease, format_cascade
from src.modules.m53_au_cpi import AU_CPI_RELEASES
from src.modules.m52_rba_rate import RBA_RELEASES

AU_CPI_DATES = set(AU_CPI_RELEASES.keys())
RBA_DATES = set(RBA_RELEASES.keys())


def _classify_au_cpi(data: dict) -> str:
    yoy = data.get('yoy', 2.5)
    if yoy >= 4.0: return 'VERY_HOT'
    if yoy >= 3.0: return 'HOT'
    if yoy >= 2.0: return 'TARGET'
    if yoy <= 1.0: return 'COOL'
    return 'WARM'

def _classify_rba(data: dict) -> str:
    return data.get('action', 'HOLD')


AU_CONFIRMATION_MATRIX = {
    ('VERY_HOT', 'HIKE'): (-1.00, +0.15, 'CPI surging + RBA hike — risk-off'),
    ('HOT', 'HIKE'):      (-0.70, +0.10, 'CPI hot + RBA hike — hawkish'),
    ('HOT', 'HOLD'):      (-0.20, +0.00, 'CPI hot but RBA holds — patience'),
    ('TARGET', 'HOLD'):   (+0.20, +0.05, 'CPI at target + RBA holds — stable'),
    ('TARGET', 'CUT'):    (+0.50, +0.10, 'CPI target + RBA cut — dovish'),
    ('COOL', 'CUT'):      (+0.70, +0.10, 'CPI cool + RBA cut — stimulus'),
    ('COOL', 'HOLD'):     (+0.30, +0.05, 'CPI cool — RBA has room'),
    ('WARM', 'HOLD'):     (+0.00, +0.00, 'CPI warm + RBA hold — neutral'),
}

AU_REGIME_SENSITIVITY = {
    'TIGHTENING': 0.60, 'EASING': 0.50, 'CRISIS_RECOVERY': 0.75,
    'BULL': 0.70, 'BEAR': 1.00, 'RECOVERY': 0.65,
    'ACCELERATION': 0.60, 'STAGFLATION': 0.85, 'STAGFLATION_HOT': 0.95,
}


def _build_au_cascade() -> CascadeEngine:
    return CascadeEngine(
        name='AU_MACRO',
        description='Australia Macro Chain: CPI(primary) → RBA(policy)',
        releases=[
            CascadeRelease('AU_CPI', AU_CPI_DATES, 0.40, 'PRIMARY', 'M53_ENABLED',
                           release_hour_utc=0, release_minute_utc=30,
                           signal_classifier=_classify_au_cpi),
            CascadeRelease('RBA', RBA_DATES, 0.60, 'POLICY', 'M52_ENABLED',
                           release_hour_utc=4, release_minute_utc=30,
                           signal_classifier=_classify_rba),
        ],
        confirmation_matrix=AU_CONFIRMATION_MATRIX,
        regime_sensitivity=AU_REGIME_SENSITIVITY,
    )

_AU_CASCADE = None
def _get_cascade():
    global _AU_CASCADE
    if _AU_CASCADE is None: _AU_CASCADE = _build_au_cascade()
    return _AU_CASCADE


def score_au_cascade(df_15m, current_time=None, config=None, regime='UNKNOWN',
                     release_data_map=None):
    cascade = _get_cascade()
    if current_time is None: current_time = datetime.utcnow()
    status, score, details, decay = cascade.score(df_15m, current_time, config, regime, release_data_map)
    if status == 'SKIP': return status, score, details, decay
    result = details.get('result', {})
    steps = details.get('steps', [])
    reason_parts = [f'AU_MACRO cascade: {result.get("combined_signal", "?")}']
    for step in steps:
        if step.get('signal') and step.get('signal') not in ('PENDING', 'NEUTRAL'):
            reason_parts.append(f'{step["release"]}={step["signal"]}')
    if decay < 1.0: reason_parts.append(f'decay={decay:.2f}')
    details['score_reason'] = ', '.join(reason_parts)
    return status, score, details, decay


def format_au_cascade(details: dict) -> str:
    if not details: return ''
    return format_cascade(details) or ''

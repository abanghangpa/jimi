"""
Japan Macro Cascade — Japan CPI → BoJ Rate

Models the Japanese economic data chain. Japan matters for crypto because:
  1. BoJ policy affects global carry trade (JPY funding)
  2. BoJ rate hikes = JPY strengthens = carry unwind = risk-off globally
  3. BoJ hold = carry intact = risk appetite sustained

Release sequence:
  1. Japan CPI (PRIMARY, ~18th-25th) — inflation signal for BoJ
  2. BoJ Rate (POLICY, ~every 6 weeks) — rate decision payoff

Thesis:
  Japan CPI rising + BoJ hawkish → carry unwind → global risk-off → ETH dumps
  Japan CPI stable + BoJ hold → carry intact → risk-on → ETH neutral/bullish
  BoJ surprise hike = -2% to -5% global risk assets (Aug 2024 example)

Usage:
    from src.modules.cascade_japan import score_japan_cascade, format_japan_cascade
"""

from datetime import datetime
from src.modules.cascade_engine import CascadeEngine, CascadeRelease, format_cascade
from src.modules.m46_japan_cpi import JAPAN_CPI_RELEASES
from src.modules.m47_boj_rate import BOJ_RELEASES

JAPAN_CPI_DATES = set(JAPAN_CPI_RELEASES.keys())
BOJ_DATES = set(BOJ_RELEASES.keys())


def _classify_japan_cpi(data: dict) -> str:
    yoy = data.get('yoy', 2.0)
    if yoy >= 3.5: return 'VERY_HOT'
    if yoy >= 2.5: return 'HOT'
    if yoy >= 1.5: return 'TARGET'
    if yoy <= 0.5: return 'DEFLATION'
    return 'COOL'

def _classify_boj(data: dict) -> str:
    action = data.get('action', 'HOLD')
    if action == 'HIKE': return 'HIKE'
    if action == 'CUT': return 'CUT'
    signal = data.get('signal', 'NEUTRAL')
    if signal in ('HAWKISH', 'DOVISH'):
        return signal
    return 'HOLD'


JAPAN_CONFIRMATION_MATRIX = {
    ('VERY_HOT', 'HIKE'):      (-1.50, +0.20, 'CPI surging + BoJ hike — carry unwind risk'),
    ('VERY_HOT', 'HAWKISH'):   (-0.80, +0.10, 'CPI hot + BoJ hawkish — hike coming'),
    ('HOT', 'HIKE'):           (-1.20, +0.15, 'CPI hot + BoJ hike — JPY strengthens'),
    ('HOT', 'HAWKISH'):        (-0.50, +0.10, 'CPI hot + BoJ hawkish — pressure building'),
    ('HOT', 'HOLD'):           (-0.20, +0.00, 'CPI hot but BoJ holds — patience'),
    ('TARGET', 'HOLD'):        (+0.20, +0.05, 'CPI at target + BoJ hold — stable'),
    ('TARGET', 'CUT'):         (+0.40, +0.05, 'CPI target + BoJ cut — dovish'),
    ('COOL', 'HOLD'):          (+0.30, +0.05, 'CPI cool + BoJ hold — carry intact'),
    ('COOL', 'CUT'):           (+0.60, +0.10, 'CPI cool + BoJ cut — stimulus'),
    ('DEFLATION', 'HOLD'):     (+0.40, +0.05, 'Deflation + BoJ hold — must stay dovish'),
    ('DEFLATION', 'CUT'):      (+0.80, +0.10, 'Deflation + BoJ cut — maximum stimulus'),
    # Surprise scenarios (the dangerous ones)
    ('COOL', 'HIKE'):          (-1.00, +0.15, 'BoJ hike despite cool CPI — surprise hawkish'),
    ('DEFLATION', 'HIKE'):     (-2.00, +0.20, 'BoJ hike in deflation — maximum surprise'),
}

BOJ_POLICY_CONTEXT = {
    'HIKE':    (-1.00, 'BoJ hike — carry trade unwind, global risk-off'),
    'HAWKISH': (-0.40, 'BoJ hawkish — hike coming, JPY strengthening'),
    'HOLD':    (+0.00, 'BoJ hold — carry trade intact'),
    'DOVISH':  (+0.30, 'BoJ dovish — further easing possible'),
    'CUT':     (+0.50, 'BoJ cut — stimulus, JPY weakening'),
}

JAPAN_REGIME_SENSITIVITY = {
    'TIGHTENING': 0.70, 'EASING': 0.60, 'CRISIS_RECOVERY': 0.85,
    'BULL': 0.75, 'BEAR': 1.05, 'RECOVERY': 0.70,
    'ACCELERATION': 0.65, 'STAGFLATION': 0.90, 'STAGFLATION_HOT': 1.00,
}


def _build_japan_cascade() -> CascadeEngine:
    return CascadeEngine(
        name='JAPAN_MACRO',
        description='Japan Macro Chain: CPI(primary) → BoJ(policy) — carry trade dynamics',
        releases=[
            CascadeRelease('JAPAN_CPI', JAPAN_CPI_DATES, 0.35, 'PRIMARY', 'M46_ENABLED',
                           release_hour_utc=23, release_minute_utc=30,
                           signal_classifier=_classify_japan_cpi),
            CascadeRelease('BOJ', BOJ_DATES, 0.65, 'POLICY', 'M47_ENABLED',
                           release_hour_utc=3, release_minute_utc=30,
                           signal_classifier=_classify_boj),
        ],
        confirmation_matrix=JAPAN_CONFIRMATION_MATRIX,
        regime_sensitivity=JAPAN_REGIME_SENSITIVITY,
    )

_JAPAN_CASCADE = None
def _get_cascade():
    global _JAPAN_CASCADE
    if _JAPAN_CASCADE is None: _JAPAN_CASCADE = _build_japan_cascade()
    return _JAPAN_CASCADE


def score_japan_cascade(df_15m, current_time=None, config=None, regime='UNKNOWN',
                        release_data_map=None):
    cascade = _get_cascade()
    if current_time is None: current_time = datetime.utcnow()
    status, score, details, decay = cascade.score(df_15m, current_time, config, regime, release_data_map)
    if status == 'SKIP': return status, score, details, decay
    result = details.get('result', {})
    steps = details.get('steps', [])
    reason_parts = [f'JAPAN_MACRO cascade: {result.get("combined_signal", "?")}']
    for step in steps:
        if step.get('signal') and step.get('signal') not in ('PENDING', 'NEUTRAL'):
            reason_parts.append(f'{step["release"]}={step["signal"]}')
    if decay < 1.0: reason_parts.append(f'decay={decay:.2f}')
    details['score_reason'] = ', '.join(reason_parts)
    return status, score, details, decay


def format_japan_cascade(details: dict) -> str:
    if not details: return ''
    output = format_cascade(details)
    if not output: return ''
    lines = [output]
    # Carry trade context
    for step in details.get('steps', []):
        if step.get('release') == 'BOJ':
            signal = step.get('signal', '?')
            if signal == 'HIKE':
                lines.append(f"    🚨 BoJ HIKE — carry trade unwind risk, global risk-off")
            elif signal == 'HAWKISH':
                lines.append(f"    ⚠️ BoJ hawkish — hike coming, watch JPY strength")
    return '\n'.join(lines)

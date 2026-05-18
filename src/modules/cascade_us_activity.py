"""
US Activity Cascade — ISM Mfg → ISM Services → Retail Sales → Durables → GDP

Models the US economic activity chain. This cascade captures the
real economy pulse — employment and spending data that drives Fed policy.

Release sequence:
  1. ISM Mfg PMI (PRIMARY, ~1st business day) — manufacturing health
  2. ISM Services PMI (CONFIRMATION, ~3rd business day) — services health
  3. Retail Sales (CONFIRMATION, ~15th) — consumer spending
  4. Durable Goods (CONFIRMATION, ~25th) — business investment
  5. GDP (STRUCTURAL, quarterly) — lagging summary

Key forensic findings:
  ISM Mfg New Orders <50 = DXY weakens → ETH squeeze (M27 thesis)
  ISM transmission chain: 85.9% Tokyo Close→Pre-London persistence
  Retail Sales BIG_MISS + LOW_VOL + MARKDOWN: +2.32% avg, 80% win
  Retail Sales edge window: ~4h post-release (13:30-17:30 UTC)

Usage:
    from src.modules.cascade_us_activity import score_us_activity_cascade
"""

from datetime import datetime
from src.modules.cascade_engine import CascadeEngine, CascadeRelease, format_cascade
from src.modules.m27_ism_pmi import ISM_MFG_RELEASES
from src.modules.m28_ism_svc_pmi import ISM_SVC_RELEASES
from src.modules.m33_retail_sales import RETAIL_SALES_RELEASES
from src.modules.m44_durables import DURABLE_GOODS_RELEASES
from src.modules.m43_us_gdp import US_GDP_RELEASES

ISM_MFG_DATES = set(ISM_MFG_RELEASES.keys())
ISM_SVC_DATES = set(ISM_SVC_RELEASES.keys())
RETAIL_DATES = set(RETAIL_SALES_RELEASES.keys())
DURABLES_DATES = set(DURABLE_GOODS_RELEASES.keys())
US_GDP_DATES = set(US_GDP_RELEASES.keys())


def _classify_ism(data: dict) -> str:
    no = data.get('new_orders', 50.0)
    pmi = data.get('pmi', 50.0)
    if no < 45.0: return 'SEVERE_CONTRACTION'
    if no < 48.0: return 'STRONG_CONTRACTION'
    if no < 50.0: return 'WEAK_CONTRACTION'
    if pmi >= 55.0: return 'STRONG_EXPANSION'
    if pmi >= 50.0: return 'EXPANDING'
    return 'MIXED'

def _classify_retail(data: dict) -> str:
    mom = data.get('mom', 0.0)
    consensus = data.get('consensus', 0.0)
    surprise = mom - consensus
    if surprise >= 0.5: return 'STRONG_BEAT'
    if surprise >= 0.2: return 'BEAT'
    if surprise <= -0.5: return 'BIG_MISS'
    if surprise <= -0.2: return 'MISS'
    return 'INLINE'

def _classify_durables(data: dict) -> str:
    mom = data.get('mom', 0.0)
    if mom >= 2.0: return 'SURGING'
    if mom >= 0.5: return 'STRONG'
    if mom <= -2.0: return 'COLLAPSING'
    if mom <= -0.5: return 'WEAK'
    return 'STABLE'

def _classify_gdp(data: dict) -> str:
    qoq = data.get('qoq', 2.0)
    if qoq >= 3.0: return 'STRONG'
    if qoq >= 2.0: return 'MODERATE'
    if qoq <= 0.0: return 'CONTRACTION'
    if qoq <= 1.0: return 'WEAK'
    return 'MODERATE'


US_ACTIVITY_CONFIRMATION = {
    # ISM Mfg + ISM Services agreement
    ('EXPANDING', 'EXPANDING'):          (+0.60, +0.10, 'Both ISMs expanding — broad growth'),
    ('STRONG_EXPANSION', 'EXPANDING'):   (+0.80, +0.10, 'Mfg strong + services expanding'),
    ('SEVERE_CONTRACTION', 'EXPANDING'): (+0.30, -0.05, 'Mfg severe but services ok — mixed'),
    ('WEAK_CONTRACTION', 'EXPANDING'):   (+0.20, +0.00, 'Mfg weak but services carrying'),
    ('STRONG_CONTRACTION', 'WEAK_CONTRACTION'): (-1.00, +0.15, 'Both contracting — recession signal'),
    ('SEVERE_CONTRACTION', 'WEAK_CONTRACTION'): (-1.50, +0.20, 'Both severely contracting — recession'),
    # ISM + Retail Sales
    ('EXPANDING', 'STRONG_BEAT'): (+0.80, +0.10, 'ISM expanding + retail beat — consumer strong'),
    ('EXPANDING', 'BIG_MISS'):    (-0.40, -0.05, 'ISM ok but retail miss — consumer weakening'),
    ('WEAK_CONTRACTION', 'BIG_MISS'): (-0.80, +0.10, 'ISM weak + retail miss — slowdown confirmed'),
}

US_ACTIVITY_REGIME = {
    'TIGHTENING': 0.65, 'EASING': 0.55, 'CRISIS_RECOVERY': 0.85,
    'BULL': 0.80, 'BEAR': 1.15, 'RECOVERY': 0.75,
    'ACCELERATION': 0.70, 'STAGFLATION': 0.95, 'STAGFLATION_HOT': 1.05,
}


def _build_us_activity_cascade() -> CascadeEngine:
    return CascadeEngine(
        name='US_ACTIVITY',
        description='US Activity Chain: ISM Mfg(primary) → ISM Svc(confirm) → Retail Sales(confirm) → Durables(confirm) → GDP(structural)',
        releases=[
            CascadeRelease('ISM_MFG', ISM_MFG_DATES, 0.30, 'PRIMARY', 'M27_ENABLED',
                           release_hour_utc=14, release_minute_utc=0,
                           signal_classifier=_classify_ism),
            CascadeRelease('ISM_SVC', ISM_SVC_DATES, 0.20, 'CONFIRMATION', 'M28_ENABLED',
                           release_hour_utc=14, release_minute_utc=0,
                           signal_classifier=_classify_ism),
            CascadeRelease('RETAIL_SALES', RETAIL_DATES, 0.20, 'CONFIRMATION', 'M33_ENABLED',
                           release_hour_utc=12, release_minute_utc=30,
                           signal_classifier=_classify_retail),
            CascadeRelease('DURABLES', DURABLES_DATES, 0.10, 'CONFIRMATION', 'M44_ENABLED',
                           release_hour_utc=12, release_minute_utc=30,
                           signal_classifier=_classify_durables),
            CascadeRelease('US_GDP', US_GDP_DATES, 0.20, 'STRUCTURAL', 'M43_ENABLED',
                           release_hour_utc=12, release_minute_utc=30,
                           signal_classifier=_classify_gdp),
        ],
        confirmation_matrix=US_ACTIVITY_CONFIRMATION,
        regime_sensitivity=US_ACTIVITY_REGIME,
    )

_US_ACTIVITY_CASCADE = None
def _get_cascade():
    global _US_ACTIVITY_CASCADE
    if _US_ACTIVITY_CASCADE is None: _US_ACTIVITY_CASCADE = _build_us_activity_cascade()
    return _US_ACTIVITY_CASCADE


def score_us_activity_cascade(df_15m, current_time=None, config=None, regime='UNKNOWN',
                              release_data_map=None):
    cascade = _get_cascade()
    if current_time is None: current_time = datetime.utcnow()
    status, score, details, decay = cascade.score(df_15m, current_time, config, regime, release_data_map)
    if status == 'SKIP': return status, score, details, decay
    result = details.get('result', {})
    steps = details.get('steps', [])
    reason_parts = [f'US_ACTIVITY cascade: {result.get("combined_signal", "?")}']
    for step in steps:
        if step.get('signal') and step.get('signal') not in ('PENDING', 'NEUTRAL'):
            reason_parts.append(f'{step["release"]}={step["signal"]}')
    if decay < 1.0: reason_parts.append(f'decay={decay:.2f}')
    details['score_reason'] = ', '.join(reason_parts)
    return status, score, details, decay


def format_us_activity_cascade(details: dict) -> str:
    if not details: return ''
    output = format_cascade(details)
    if not output: return ''
    lines = [output]
    # ISM transmission chain note
    for step in details.get('steps', []):
        if step.get('release') == 'ISM_MFG':
            signal = step.get('signal', '?')
            if 'SEVERE' in signal or 'STRONG_CONTRACTION' in signal:
                lines.append(f"    💡 ISM Mfg severe contraction → DXY weakens → ETH squeeze potential")
    return '\n'.join(lines)

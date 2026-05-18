"""
Japan Macro Cascade — Japan GDP → Japan CPI → BoJ Rate

Models the Japanese economic data chain. Japan matters for crypto because:
  1. BoJ policy affects global carry trade (JPY funding)
  2. BoJ rate hikes = JPY strengthens = carry unwind = risk-off globally
  3. BoJ hold = carry intact = risk appetite sustained
  4. Japan GDP informs BoJ policy expectations (leading indicator)

Release sequence:
  1. Japan GDP (STRUCTURAL, quarterly ~15th Feb/May/Aug/Nov) — growth for BoJ
  2. Japan CPI (PRIMARY, ~18th-25th monthly) — inflation signal for BoJ
  3. BoJ Rate (POLICY, ~every 6 weeks) — rate decision payoff

Thesis:
  Strong GDP + CPI rising → BoJ hawkish → carry unwind → ETH dumps
  Weak GDP + CPI cool → BoJ dovish → carry intact → ETH neutral/bullish
  GDP is a leading indicator for BoJ (like ADP is to NFP).
  Aug 2024: BoJ surprise hike → BTC -23%, ETH -45% from highs.

Usage:
    from src.modules.cascade_japan import score_japan_cascade, format_japan_cascade
"""

from datetime import datetime
from src.modules.cascade_engine import CascadeEngine, CascadeRelease, format_cascade
from src.modules.m46_japan_cpi import JAPAN_CPI_RELEASES
from src.modules.m47_boj_rate import BOJ_RELEASES

# ═══════════════════════════════════════════════════════════════
# JAPAN GDP SCHEDULE (quarterly, Cabinet Office preliminary)
# Released ~15th of Feb/May/Aug/Nov, 8:50 AM JST (23:50 UTC prev day)
# Source: Cabinet Office ESRI quarterly GDP releases
# ═══════════════════════════════════════════════════════════════

JAPAN_GDP_RELEASES = {
    # 2022
    '2022-02-15': {'gdp_qoq': 1.3, 'consensus_qoq': 1.4, 'prev_qoq': -0.9, 'quarter': 'Q4', 'annualized': 5.4},
    '2022-05-18': {'gdp_qoq': -0.2, 'consensus_qoq': -0.4, 'prev_qoq': 0.0, 'quarter': 'Q1', 'annualized': -1.0},
    '2022-08-15': {'gdp_qoq': 0.5, 'consensus_qoq': 0.6, 'prev_qoq': -0.1, 'quarter': 'Q2', 'annualized': 2.2},
    '2022-11-15': {'gdp_qoq': -0.3, 'consensus_qoq': -0.2, 'prev_qoq': 0.9, 'quarter': 'Q3', 'annualized': -1.2},
    # 2023
    '2023-02-14': {'gdp_qoq': 0.2, 'consensus_qoq': 0.5, 'prev_qoq': -0.3, 'quarter': 'Q4', 'annualized': 0.6},
    '2023-05-17': {'gdp_qoq': 0.4, 'consensus_qoq': 0.2, 'prev_qoq': 0.0, 'quarter': 'Q1', 'annualized': 1.6},
    '2023-08-15': {'gdp_qoq': 1.5, 'consensus_qoq': 0.8, 'prev_qoq': 0.3, 'quarter': 'Q2', 'annualized': 6.0},
    '2023-11-15': {'gdp_qoq': -0.5, 'consensus_qoq': -0.1, 'prev_qoq': 0.9, 'quarter': 'Q3', 'annualized': -2.1},
    # 2024
    '2024-02-15': {'gdp_qoq': -0.1, 'consensus_qoq': 0.2, 'prev_qoq': -0.3, 'quarter': 'Q4', 'annualized': -0.4},
    '2024-05-16': {'gdp_qoq': -0.5, 'consensus_qoq': -0.4, 'prev_qoq': 0.0, 'quarter': 'Q1', 'annualized': -2.0},
    '2024-08-15': {'gdp_qoq': 0.8, 'consensus_qoq': 0.5, 'prev_qoq': -0.5, 'quarter': 'Q2', 'annualized': 3.2},
    '2024-11-15': {'gdp_qoq': 0.2, 'consensus_qoq': 0.3, 'prev_qoq': 0.5, 'quarter': 'Q3', 'annualized': 0.9},
    # 2025
    '2025-02-17': {'gdp_qoq': 0.4, 'consensus_qoq': 0.3, 'prev_qoq': 0.2, 'quarter': 'Q4', 'annualized': 1.6},
    '2025-05-16': {'gdp_qoq': 0.6, 'consensus_qoq': 0.4, 'prev_qoq': 0.4, 'quarter': 'Q1', 'annualized': 2.4},
    '2025-08-15': {'gdp_qoq': -0.2, 'consensus_qoq': 0.1, 'prev_qoq': 0.6, 'quarter': 'Q2', 'annualized': -0.8},
    '2025-11-17': {'gdp_qoq': 0.3, 'consensus_qoq': 0.2, 'prev_qoq': -0.2, 'quarter': 'Q3', 'annualized': 1.2},
    # 2026
    '2026-02-16': {'gdp_qoq': 0.2, 'consensus_qoq': 0.3, 'prev_qoq': 0.3, 'quarter': 'Q4', 'annualized': 0.8},
    '2026-05-18': {'gdp_qoq': None, 'consensus_qoq': 0.3, 'prev_qoq': 0.2, 'quarter': 'Q1', 'annualized': None},
    # Future scheduled
    '2026-08-17': {'gdp_qoq': None, 'consensus_qoq': None, 'prev_qoq': None, 'quarter': 'Q2', 'annualized': None},
    '2026-11-16': {'gdp_qoq': None, 'consensus_qoq': None, 'prev_qoq': None, 'quarter': 'Q3', 'annualized': None},
}

JAPAN_GDP_DATES = set(JAPAN_GDP_RELEASES.keys())

JAPAN_CPI_DATES = set(JAPAN_CPI_RELEASES.keys())
BOJ_DATES = set(BOJ_RELEASES.keys())


# ═══════════════════════════════════════════════════════════════
# SIGNAL CLASSIFIERS
# ═══════════════════════════════════════════════════════════════

def _classify_japan_gdp(data: dict) -> str:
    """Classify Japan GDP signal.

    Strong GDP → BoJ more likely to hike → JPY strengthens → carry unwind
    Weak GDP → BoJ stays dovish → carry intact
    """
    qoq = data.get('gdp_qoq')
    consensus = data.get('consensus_qoq', 0.3)
    if qoq is None:
        return 'PENDING'

    surprise = qoq - consensus
    if qoq < 0 and surprise < -0.3:
        return 'CONTRACTION'  # GDP negative + big miss
    elif qoq < 0:
        return 'WEAK'         # GDP negative but inline
    elif surprise >= 0.3:
        return 'STRONG_BEAT'  # GDP positive + big beat
    elif surprise >= 0.1:
        return 'BEAT'         # GDP positive + mild beat
    elif surprise <= -0.2:
        return 'MISS'         # GDP positive but missed
    return 'INLINE'           # GDP positive + inline


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


# ═══════════════════════════════════════════════════════════════
# CONFIRMATION MATRIX
# ═══════════════════════════════════════════════════════════════

JAPAN_GDP_CPI_MATRIX = {
    # GDP + CPI alignment
    ('STRONG_BEAT', 'VERY_HOT'): (-1.00, +0.15, 'GDP strong + CPI surging — BoJ must hike'),
    ('STRONG_BEAT', 'HOT'):      (-0.60, +0.10, 'GDP strong + CPI hot — BoJ hawkish pressure'),
    ('STRONG_BEAT', 'TARGET'):   (+0.20, +0.05, 'GDP strong + CPI target — healthy growth'),
    ('STRONG_BEAT', 'COOL'):     (+0.40, +0.05, 'GDP strong + CPI cool — goldilocks'),
    ('BEAT', 'HOT'):             (-0.30, +0.05, 'GDP beat + CPI hot — BoJ may hike'),
    ('BEAT', 'TARGET'):          (+0.30, +0.05, 'GDP beat + CPI target — stable growth'),
    ('BEAT', 'COOL'):            (+0.40, +0.05, 'GDP beat + CPI cool — growth without inflation'),
    ('INLINE', 'TARGET'):        (+0.10, +0.00, 'GDP inline + CPI target — neutral'),
    ('INLINE', 'HOT'):           (-0.20, +0.00, 'GDP inline + CPI hot — mild hawkish risk'),
    ('MISS', 'HOT'):             (-0.50, +0.10, 'GDP miss + CPI hot — stagflation risk'),
    ('MISS', 'COOL'):            (-0.20, +0.00, 'GDP miss + CPI cool — slowing economy'),
    ('WEAK', 'HOT'):             (-0.80, +0.15, 'GDP weak + CPI hot — stagflation, worst case'),
    ('WEAK', 'COOL'):            (+0.20, +0.05, 'GDP weak + CPI cool — BoJ will cut'),
    ('WEAK', 'DEFLATION'):       (+0.50, +0.10, 'GDP weak + deflation — BoJ must stimulate'),
    ('CONTRACTION', 'HOT'):      (-1.20, +0.20, 'GDP contraction + CPI hot — stagflation crisis'),
    ('CONTRACTION', 'COOL'):     (+0.40, +0.10, 'GDP contraction + CPI cool — BoJ will cut'),
    ('CONTRACTION', 'DEFLATION'):  (+0.80, +0.15, 'GDP contraction + deflation — maximum stimulus'),
}

JAPAN_GDP_BOJ_MATRIX = {
    # GDP signal + BoJ action
    ('STRONG_BEAT', 'HIKE'):    (-1.20, +0.15, 'GDP strong + BoJ hike — carry unwind confirmed'),
    ('STRONG_BEAT', 'HAWKISH'): (-0.60, +0.10, 'GDP strong + BoJ hawkish — hike coming'),
    ('STRONG_BEAT', 'HOLD'):    (+0.20, +0.05, 'GDP strong but BoJ holds — patience'),
    ('BEAT', 'HIKE'):           (-1.00, +0.10, 'GDP beat + BoJ hike — JPY strengthens'),
    ('BEAT', 'HOLD'):           (+0.20, +0.05, 'GDP beat + BoJ holds — carry intact'),
    ('WEAK', 'HOLD'):           (+0.30, +0.05, 'GDP weak + BoJ holds — dovish expectations'),
    ('WEAK', 'CUT'):            (+0.60, +0.10, 'GDP weak + BoJ cut — stimulus'),
    ('CONTRACTION', 'HOLD'):    (+0.40, +0.05, 'GDP contraction + BoJ holds — must stay dovish'),
    ('CONTRACTION', 'CUT'):     (+0.80, +0.10, 'GDP contraction + BoJ cut — maximum stimulus'),
    ('CONTRACTION', 'HIKE'):    (-2.00, +0.20, 'GDP contraction + BoJ hike — surprise, carry crash'),
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


# ═══════════════════════════════════════════════════════════════
# CARRY TRADE CONTEXT
# ═══════════════════════════════════════════════════════════════

# Aug 2024 carry trade crash reference
CARRY_CRASH_REF = {
    'date': '2024-08-05',
    'trigger': 'BoJ surprise hike + weak US data',
    'btc_move': '-23% (65K→50K)',
    'eth_move': '-45% from 2024 high',
    'mechanism': 'JPY carry trade unwind → global risk-off → crypto liquidation cascade',
}

# BoJ rate hike impact on crypto (historical)
BOJ_HIKE_IMPACT = {
    '2024-03-19': {'rate_change': '+0.10%', 'eth_1d': -1.2, 'eth_1w': -5.8, 'note': 'First hike since 2007'},
    '2024-07-31': {'rate_change': '+0.25%', 'eth_1d': -3.5, 'eth_1w': -18.0, 'note': 'Triggered Aug crash'},
    '2025-01-24': {'rate_change': '+0.25%', 'eth_1d': -2.1, 'eth_1w': -4.2, 'note': 'Carry unwind concern'},
}


def _build_japan_cascade() -> CascadeEngine:
    return CascadeEngine(
        name='JAPAN_MACRO',
        description='Japan Macro Chain: GDP(structural) → CPI(primary) → BoJ(policy) — carry trade dynamics',
        releases=[
            CascadeRelease('JAPAN_GDP', JAPAN_GDP_DATES, 0.15, 'STRUCTURAL', 'M_JAPAN_GDP_ENABLED',
                           release_hour_utc=23, release_minute_utc=50,
                           signal_classifier=_classify_japan_gdp),
            CascadeRelease('JAPAN_CPI', JAPAN_CPI_DATES, 0.30, 'PRIMARY', 'M46_ENABLED',
                           release_hour_utc=23, release_minute_utc=30,
                           signal_classifier=_classify_japan_cpi),
            CascadeRelease('BOJ', BOJ_DATES, 0.55, 'POLICY', 'M47_ENABLED',
                           release_hour_utc=3, release_minute_utc=30,
                           signal_classifier=_classify_boj),
        ],
        confirmation_matrix=JAPAN_GDP_CPI_MATRIX,
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

    # Add carry trade context
    details['carry_trade'] = {
        'reference': CARRY_CRASH_REF,
        'boj_hike_impact': BOJ_HIKE_IMPACT,
    }

    return status, score, details, decay


def format_japan_cascade(details: dict) -> str:
    if not details: return ''
    output = format_cascade(details)
    if not output: return ''
    lines = [output]

    # GDP context
    for step in details.get('steps', []):
        if step.get('release') == 'JAPAN_GDP':
            gdp_data = step.get('details', {})
            if gdp_data:
                actual = gdp_data.get('gdp_qoq')
                consensus = gdp_data.get('consensus_qoq')
                ann = gdp_data.get('annualized')
                q = gdp_data.get('quarter', '?')
                if actual is not None:
                    surprise = actual - (consensus or 0)
                    icon = '🟢' if surprise > 0 else '🔴' if surprise < 0 else '⚪'
                    lines.append(f"    {icon} GDP {q}: {actual:+.1f}% QoQ (ann. {ann:+.1f}%)  consensus={consensus:+.1f}%  surprise={surprise:+.1f}%")
                else:
                    lines.append(f"    ⏳ GDP {q}: pending  consensus={consensus:+.1f}%")

        elif step.get('release') == 'BOJ':
            signal = step.get('signal', '?')
            if signal == 'HIKE':
                lines.append(f"    🚨 BoJ HIKE — carry trade unwind risk, global risk-off")
                lines.append(f"    📊 Ref: Aug 2024 crash — BTC -23%, ETH -45% from highs")
            elif signal == 'HAWKISH':
                lines.append(f"    ⚠️ BoJ hawkish — hike coming, watch JPY strength")

    return '\n'.join(lines)

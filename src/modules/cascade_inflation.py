"""
US Inflation Cascade — CPI → PPI → PCE → FOMC → Presser → Minutes

Models the full US inflation → monetary policy chain as a single scoring unit.
This is the highest-impact cascade for crypto — FOMC moves are 2-5x larger
than typical macro releases.

Release sequence (within each FOMC cycle, ~6 weeks):
  1. CPI (PRIMARY, ~2nd-3rd Tuesday/Wednesday) — biggest inflation signal
  2. PPI (CONFIRMATION, ~1-2 days after CPI) — confirms/denies CPI
  3. PCE (CONFIRMATION, ~final Friday) — Fed's preferred metric, final word
  4. FOMC (POLICY, ~6 weeks after previous) — rate decision, the payoff
  5. Presser (FOLLOWUP, 30 min after FOMC) — Powell reverses or amplifies
  6. Minutes (FOLLOWUP, 3 weeks after FOMC) — loopback, reveals dissent

Key insights from forensic data:
  - Cool CPI = +1.06% intraday, +2.50% over 2 days (strongest macro signal)
  - Hot CPI = -0.45% intraday (conditional on regime)
  - PPI cool has INVERTED signal (-2.89%, deflation fears)
  - FOMC hawkish surprise = -3.8% avg, dovish surprise = +4.2% avg
  - Presser reverses initial FOMC spike 40% of the time
  - Minutes 3 weeks later create loopback (dissent reveals future direction)

Usage:
    from src.modules.cascade_inflation import score_inflation_cascade, format_inflation_cascade
    status, score, details, decay = score_inflation_cascade(df_15m, current_time, config)
"""

from datetime import datetime
from src.modules.cascade_engine import (
    CascadeEngine, CascadeRelease, CascadeStep,
    compute_us_session, compute_1h_spike, compute_asia_session, compute_uk_session,
    compute_decay, format_cascade,
)
from src.modules.m56_us_cpi import CPI_SCHEDULE_DATES
from src.modules.m60_us_ppi import PPI_SCHEDULE_DATES
from src.modules.m57_fomc import FOMC_RELEASES
from src.modules.m58_powell_presser import PRESSER_RELEASES
from src.modules.m59_fomc_minutes import MINUTES_RELEASES

FOMC_RELEASE_DATES = set(FOMC_RELEASES.keys())
PRESSER_RELEASE_DATES = set(PRESSER_RELEASES.keys())
MINUTES_RELEASE_DATES = set(MINUTES_RELEASES.keys())


# ═══════════════════════════════════════════════════════════════
# PCE SCHEDULE (released ~final Friday of month, 08:30 ET = 13:30 UTC)
# PCE is the Fed's preferred inflation metric
# ═══════════════════════════════════════════════════════════════

PCE_SCHEDULE_DATES = set()


# ═══════════════════════════════════════════════════════════════
# SIGNAL CLASSIFIERS
# ═══════════════════════════════════════════════════════════════

def _classify_cpi_signal(data: dict) -> str:
    """Classify CPI signal from release data."""
    yoy = data.get('yoy', data.get('actual', 2.5))
    consensus = data.get('consensus', 2.5)

    if isinstance(yoy, (int, float)) and isinstance(consensus, (int, float)):
        surprise = yoy - consensus
        if surprise <= -0.3:
            return 'COOL'
        elif surprise >= 0.3:
            return 'HOT'
    return 'WARM'


def _classify_ppi_signal(data: dict) -> str:
    """Classify PPI signal from release data."""
    yoy = data.get('yoy', data.get('actual', 2.5))
    consensus = data.get('consensus', 2.5)

    if isinstance(yoy, (int, float)) and isinstance(consensus, (int, float)):
        surprise = yoy - consensus
        if surprise <= -0.3:
            return 'COOL'
        elif surprise >= 0.3:
            return 'HOT'
    return 'WARM'


def _classify_pce_signal(data: dict) -> str:
    """Classify PCE signal from release data."""
    yoy = data.get('yoy', data.get('actual', 2.5))
    consensus = data.get('consensus', 2.5)

    if isinstance(yoy, (int, float)) and isinstance(consensus, (int, float)):
        surprise = yoy - consensus
        if surprise <= -0.2:
            return 'COOL'
        elif surprise >= 0.2:
            return 'HOT'
    return 'WARM'


def _classify_fomc_signal(data: dict) -> str:
    """Classify FOMC signal from release data."""
    signal = data.get('signal', 'NEUTRAL')
    return signal  # Already classified by M57


# ═══════════════════════════════════════════════════════════════
# CONFIRMATION MATRIX
# ═══════════════════════════════════════════════════════════════

INFLATION_CONFIRMATION_MATRIX = {
    # CPI + PPI alignment (from M23 forensic data)
    ('COOL', 'COOL'):   (+0.50, +0.10, 'Both cool — strong disinflation, Fed can cut'),
    ('COOL', 'WARM'):   (+0.00, +0.00, 'CPI cool, PPI inline — signal intact'),
    ('COOL', 'HOT'):    (-0.30, -0.10, 'CPI cool but PPI hot — pipeline inflation building'),
    ('WARM', 'COOL'):   (-0.80, -0.10, 'CPI inline, PPI cool — deflation fears'),
    ('WARM', 'WARM'):   (+0.00, +0.00, 'Both inline — no signal'),
    ('WARM', 'HOT'):    (-0.20, +0.00, 'CPI inline, PPI hot — mild inflation concern'),
    ('HOT', 'COOL'):    (+0.30, -0.10, 'CPI hot, PPI cool — mixed signals'),
    ('HOT', 'WARM'):    (+0.00, +0.00, 'CPI hot, PPI inline — hot signal intact'),
    ('HOT', 'HOT'):     (-0.80, +0.10, 'Both hot — persistent inflation, Fed trapped'),

    # CPI + PCE (PCE confirms or denies CPI, released later in cycle)
    ('COOL', 'COOL'):   (+0.30, +0.10, 'CPI+PCE both cool — confirmed disinflation'),
    ('HOT', 'HOT'):     (-0.50, +0.10, 'CPI+PCE both hot — confirmed persistent inflation'),
    ('COOL', 'HOT'):    (-0.20, -0.10, 'CPI cool but PCE hot — CPI was misleading'),
    ('HOT', 'COOL'):    (+0.20, -0.10, 'CPI hot but PCE cool — inflation moderating'),
}

# FOMC signal modifiers (applied after inflation cascade)
FOMC_SIGNAL_MODIFIERS = {
    'DOVISH_SURPRISE':  (+3.00, +0.20, 'Dovish surprise — rate cut expectations surge'),
    'DOVISH_DOT_PLOT':  (+2.00, +0.10, 'Dovish dot plot — cuts coming'),
    'DOVISH':           (+0.50, +0.05, 'Dovish hold — mild positive'),
    'NEUTRAL':          (+0.00, +0.00, 'Neutral hold — no signal'),
    'HAWKISH':          (-0.50, +0.05, 'Hawkish hold — mild negative'),
    'HAWKISH_DOT_PLOT': (-2.00, +0.10, 'Hawkish dot plot — rates higher for longer'),
    'HAWKISH_SURPRISE': (-3.50, +0.20, 'Hawkish surprise — yield surge, liquidations'),
}

# Presser reversal modifiers (presser 30 min after FOMC)
PRESSER_MODIFIERS = {
    'REVERSES_SPIKE':   (-0.50, -0.10, 'Presser reversed initial FOMC spike'),
    'AMPLIFIES_SPIKE':  (+0.30, +0.05, 'Presser amplified initial FOMC spike'),
    'NEUTRAL':          (+0.00, +0.00, 'Presser was neutral'),
}

# Minutes loopback modifiers (3 weeks after FOMC)
MINUTES_MODIFIERS = {
    'REVEALS_DOVE':     (+0.40, +0.05, 'Minutes revealed dovish dissent'),
    'REVEALS_HAWK':     (-0.40, +0.05, 'Minutes revealed hawkish dissent'),
    'CONFIRMS':         (+0.00, +0.00, 'Minutes confirmed decision'),
}


# ═══════════════════════════════════════════════════════════════
# REGIME SENSITIVITY
# ═══════════════════════════════════════════════════════════════

INFLATION_REGIME_SENSITIVITY = {
    'TIGHTENING':      0.60,
    'EASING':          0.50,
    'CRISIS_RECOVERY': 0.80,
    'BULL':            0.85,
    'BEAR':            1.20,   # 2022: Fed tightening dominates
    'RECOVERY':        0.75,
    'ACCELERATION':    0.65,
    'STAGFLATION':     1.00,
    'STAGFLATION_HOT': 1.10,
}

# Spike accuracy by year
INFLATION_SPIKE_ACCURACY = {
    2018: 0.56, 2019: 0.82, 2020: 0.57, 2021: 0.62,
    2022: 0.77, 2023: 0.50, 2024: 0.82, 2025: 0.71, 2026: 0.71,
}

# Regime-conditional expected moves (from M23 forensic data)
REGIME_CPI_EXPECTED = {
    ('BEAR', 'COOL'):          +9.92,
    ('BEAR', 'HOT'):           -3.33,
    ('BULL', 'COOL'):          +1.88,
    ('BULL', 'HOT'):           -0.20,
    ('RECOVERY', 'COOL'):      -0.55,
    ('RECOVERY', 'HOT'):       +0.84,
    ('ACCELERATION', 'COOL'):  -0.09,
    ('ACCELERATION', 'HOT'):   +0.06,
    ('STAGFLATION', 'COOL'):   +3.00,
    ('STAGFLATION', 'HOT'):    -2.00,
    ('STAGFLATION_HOT', 'COOL'): +4.00,
    ('STAGFLATION_HOT', 'HOT'):  -3.00,
}


# ═══════════════════════════════════════════════════════════════
# BUILD CASCADE ENGINE
# ═══════════════════════════════════════════════════════════════

def _build_inflation_cascade() -> CascadeEngine:
    """Build the US Inflation → Fed cascade engine instance."""
    return CascadeEngine(
        name='US_INFLATION',
        description='US Inflation → Fed Chain: CPI(primary) → PPI(confirm) → PCE(fed_pref) → FOMC(policy) → Presser(followup) → Minutes(loopback)',
        releases=[
            CascadeRelease(
                name='CPI',
                schedule_dates=CPI_SCHEDULE_DATES,
                weight=0.30,
                role='PRIMARY',
                enabled_key='M56_ENABLED',
                release_hour_utc=13,
                release_minute_utc=30,
                signal_classifier=_classify_cpi_signal,
            ),
            CascadeRelease(
                name='PPI',
                schedule_dates=PPI_SCHEDULE_DATES,
                weight=0.15,
                role='CONFIRMATION',
                enabled_key='M60_ENABLED',
                release_hour_utc=13,
                release_minute_utc=30,
                signal_classifier=_classify_ppi_signal,
            ),
            CascadeRelease(
                name='PCE',
                schedule_dates=PCE_SCHEDULE_DATES,
                weight=0.15,
                role='CONFIRMATION',
                enabled_key='M45_ENABLED',
                release_hour_utc=13,
                release_minute_utc=30,
                signal_classifier=_classify_pce_signal,
            ),
            CascadeRelease(
                name='FOMC',
                schedule_dates=FOMC_RELEASE_DATES,
                weight=0.25,
                role='POLICY',
                enabled_key='M57_ENABLED',
                release_hour_utc=18,
                release_minute_utc=0,
                signal_classifier=_classify_fomc_signal,
            ),
            CascadeRelease(
                name='PRESSER',
                schedule_dates=PRESSER_RELEASE_DATES,
                weight=0.10,
                role='FOLLOWUP',
                enabled_key='M58_ENABLED',
                release_hour_utc=18,
                release_minute_utc=30,
            ),
            CascadeRelease(
                name='MINUTES',
                schedule_dates=MINUTES_RELEASE_DATES,
                weight=0.05,
                role='FOLLOWUP',
                enabled_key='M59_ENABLED',
                release_hour_utc=18,
                release_minute_utc=0,
            ),
        ],
        confirmation_matrix=INFLATION_CONFIRMATION_MATRIX,
        regime_sensitivity=INFLATION_REGIME_SENSITIVITY,
        default_spike_accuracy=INFLATION_SPIKE_ACCURACY,
    )


# Singleton
_INFLATION_CASCADE = None


def _get_cascade() -> CascadeEngine:
    global _INFLATION_CASCADE
    if _INFLATION_CASCADE is None:
        _INFLATION_CASCADE = _build_inflation_cascade()
    return _INFLATION_CASCADE


# ═══════════════════════════════════════════════════════════════
# ENRICHED SCORING
# ═══════════════════════════════════════════════════════════════

def _enrich_with_fomc_context(details: dict, config: dict = None) -> dict:
    """Add FOMC-specific context to details."""
    cfg = config or {}

    # Find FOMC step
    fomc_step = None
    for step in details.get('steps', []):
        if step.get('release') == 'FOMC':
            fomc_step = step
            break

    if fomc_step:
        # Check if dot plot meeting (Mar/Jun/Sep/Dec have dot plots)
        fomc_date = fomc_step.get('date', '')
        if fomc_date:
            month = int(fomc_date[5:7])
            is_dot_plot = month in (3, 6, 9, 12)
            details['fomc_dot_plot'] = is_dot_plot
            if is_dot_plot:
                details['fomc_note'] = 'Dot plot meeting — 3x volatility expected'

    # Check for FOMC+CPI proximity (within 5 days = amplified)
    cpi_step = None
    for step in details.get('steps', []):
        if step.get('release') == 'CPI':
            cpi_step = step
            break

    if cpi_step and fomc_step:
        cpi_date = cpi_step.get('date', '')
        fomc_date = fomc_step.get('date', '')
        if cpi_date and fomc_date:
            from datetime import datetime as dt
            try:
                cpi_dt = dt.strptime(cpi_date, '%Y-%m-%d')
                fomc_dt = dt.strptime(fomc_date, '%Y-%m-%d')
                days_apart = abs((fomc_dt - cpi_dt).days)
                if days_apart <= 5:
                    details['cpi_fomc_proximity'] = True
                    details['cpi_fomc_days'] = days_apart
                    details['cpi_fomc_note'] = f'CPI {days_apart}d before FOMC — amplified signal'
            except ValueError:
                pass

    return details


def _enrich_with_asia_prediction(details: dict, us_dir: str, regime: str) -> dict:
    """Add Asia session prediction for inflation releases."""
    if us_dir == 'FLAT':
        details['asia_prediction'] = {
            'direction': 'NEUTRAL',
            'confidence': 'LOW',
            'reason': 'US move too small — no edge',
        }
        return details

    # Use regime-conditional expected move
    primary_signal = details.get('result', {}).get('primary_signal', 'WARM')
    expected_key = (regime, primary_signal)
    expected_move = REGIME_CPI_EXPECTED.get(expected_key, 0.0)

    # Fade rate by regime
    fade_rates = {
        'BEAR': 0.29, 'BULL': 0.12, 'RECOVERY': 0.29,
        'ACCELERATION': 0.29, 'STAGFLATION': 0.46,
        'STAGFLATION_HOT': 0.50, 'TIGHTENING': 0.30,
    }
    fade_rate = fade_rates.get(regime, 0.30)

    if us_dir == 'DUMP':
        bias = 'FADE' if fade_rate >= 0.40 else 'CONTINUATION'
    elif us_dir == 'RALLY':
        bias = 'FADE' if regime in ('STAGFLATION', 'STAGFLATION_HOT') else 'CONTINUATION'
    else:
        bias = 'MIXED'

    details['asia_prediction'] = {
        'us_direction': us_dir,
        'regime_bias': bias,
        'confidence': 'MEDIUM',
        'fade_rate': fade_rate,
        'expected_regime_move': expected_move,
    }

    return details


# ═══════════════════════════════════════════════════════════════
# MAIN SCORING FUNCTION
# ═══════════════════════════════════════════════════════════════

def score_inflation_cascade(df_15m, current_time=None, config=None, regime='UNKNOWN',
                            release_data_map=None):
    """Score the US Inflation → Fed cascade for the current time.

    Args:
        df_15m: 15m OHLCV DataFrame
        current_time: datetime (default: now UTC)
        config: config dict
        regime: macro regime string
        release_data_map: {release_name: dict} with actual/consensus data

    Returns:
        (status, score, details, decay_mult)
    """
    cascade = _get_cascade()

    if current_time is None:
        current_time = datetime.utcnow()

    # Run base cascade scoring
    status, score, details, decay = cascade.score(
        df_15m, current_time, config, regime, release_data_map)

    if status == 'SKIP':
        return status, score, details, decay

    # Enrich with FOMC context
    details = _enrich_with_fomc_context(details, config)

    # Determine US direction
    us_dir = 'FLAT'
    for step in details.get('steps', []):
        if step.get('us_dir') and step.get('us_dir') != 'FLAT':
            us_dir = step['us_dir']
            break

    # Add Asia prediction
    has_asia = any(s.get('asia_data') for s in details.get('steps', []))
    if not has_asia:
        details = _enrich_with_asia_prediction(details, us_dir, regime)

    # Build score reason
    result = details.get('result', {})
    steps = details.get('steps', [])
    combined = result.get('combined_signal', '?')

    reason_parts = [f'US_INFLATION cascade: {combined}']
    for step in steps:
        if step.get('signal') and step.get('signal') not in ('PENDING', 'NEUTRAL'):
            reason_parts.append(f'{step["release"]}={step["signal"]}')
    if details.get('fomc_dot_plot'):
        reason_parts.append('DOT_PLOT')
    if details.get('cpi_fomc_proximity'):
        reason_parts.append(f'CPI_NEAR_FOMC({details["cpi_fomc_days"]}d)')
    if decay < 1.0:
        reason_parts.append(f'decay={decay:.2f}')

    details['score_reason'] = ', '.join(reason_parts)

    return status, score, details, decay


def format_inflation_cascade(details: dict) -> str:
    """Format US Inflation cascade details for terminal output."""
    if not details:
        return ''

    output = format_cascade(details)
    if not output:
        return ''

    lines = [output]

    # FOMC context
    if details.get('fomc_dot_plot'):
        lines.append(f"    📊 DOT PLOT meeting — 3x volatility expected")
    if details.get('fomc_note'):
        lines.append(f"    ⚠️ {details['fomc_note']}")

    # CPI-FOMC proximity
    if details.get('cpi_fomc_proximity'):
        lines.append(f"    ⚡ {details.get('cpi_fomc_note', 'CPI near FOMC — amplified')}")

    # Asia prediction
    asia_pred = details.get('asia_prediction', {})
    if asia_pred:
        bias = asia_pred.get('regime_bias', '?')
        conf = asia_pred.get('confidence', '?')
        expected = asia_pred.get('expected_regime_move', 0)
        fade = asia_pred.get('fade_rate', 0)
        conf_icon = {'HIGH': '🟢', 'MEDIUM': '🟡', 'LOW': '🔴'}.get(conf, '⚪')
        lines.append(f"    Asia Prediction: {conf_icon} {bias} (fade={fade:.0%}, expected={expected:+.2f}%)")

    return '\n'.join(lines)

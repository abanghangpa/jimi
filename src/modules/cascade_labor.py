"""
US Labor Cascade — ADP → JOLTS → NFP → Unemployment → Claims

First cascade implementation using the generic cascade engine.
Models the US labor market release chain as a single scoring unit.

Release sequence (chronological within each monthly cycle):
  1. JOLTS (STRUCTURAL, ~first Tuesday) — labor demand, slow-moving
  2. ADP (LEADING, ~first Wednesday) — private payrolls preview
  3. NFP + Unemployment (PRIMARY, ~first Friday) — official labor report
  4. Claims (BACKGROUND, every Thursday) — weekly labor health

Thesis:
  ADP previews NFP direction (65% agreement). JOLTS sets the structural
  labor demand context. NFP is the primary market-mover. Unemployment
  rate confirms the trend. Claims provide weekly background context.

  When ADP and NFP agree → stronger signal (confirmation)
  When ADP and NFP disagree → weaker signal (conflicting)
  JOLTS declining + NFP weakening → recession risk amplifies

Usage:
    from src.modules.cascade_labor import score_labor_cascade, format_labor_cascade
    status, score, details, decay = score_labor_cascade(df_15m, current_time, config)
"""

from datetime import datetime
from src.modules.cascade_engine import (
    CascadeEngine, CascadeRelease, CascadeStep,
    compute_us_session, compute_1h_spike, compute_asia_session, compute_uk_session,
    compute_decay, format_cascade,
)
from src.modules.m37_nfp import NFP_SCHEDULE_DATES, NFP_RELEASES
from src.modules.m36_adp_employment import ADP_RELEASES

# ═══════════════════════════════════════════════════════════════
# SCHEDULE DATES
# ═══════════════════════════════════════════════════════════════

ADP_SCHEDULE_DATES = set(ADP_RELEASES.keys())

# JOLTS — released ~first Tuesday of month, 10:00 AM ET (15:00 UTC)
# Uses a simplified schedule (first business day of month)
JOLTS_SCHEDULE_DATES = set()

# Unemployment rate is released same day as NFP (same report)
UNEMPLOYMENT_SCHEDULE_DATES = NFP_SCHEDULE_DATES

# Claims — every Thursday (we use the existing claims module logic)
# We don't hardcode dates — the engine checks day-of-week


# ═══════════════════════════════════════════════════════════════
# SIGNAL CLASSIFIERS
# ═══════════════════════════════════════════════════════════════

def _classify_nfp_signal(data: dict) -> str:
    """Classify NFP signal from release data.

    Args:
        data: {'nfp_k': int, 'consensus_k': int, 'prev_k': int}

    Returns:
        signal: 'STRONG_BEAT', 'BEAT', 'INLINE', 'MISS', 'BIG_MISS'
    """
    nfp = data.get('nfp_k', 0)
    consensus = data.get('consensus_k', 0)
    if consensus == 0:
        return 'INLINE'

    surprise_pct = (nfp - consensus) / abs(consensus) * 100

    if surprise_pct >= 50:
        return 'STRONG_BEAT'
    elif surprise_pct >= 15:
        return 'BEAT'
    elif surprise_pct <= -50:
        return 'BIG_MISS'
    elif surprise_pct <= -15:
        return 'MISS'
    return 'INLINE'


def _classify_adp_signal(data: dict) -> str:
    """Classify ADP signal from release data."""
    adp = data.get('adp_k', 0)
    consensus = data.get('consensus_k', 0)
    if consensus == 0:
        return 'INLINE'

    surprise_pct = (adp - consensus) / abs(consensus) * 100

    if surprise_pct >= 50:
        return 'STRONG_BEAT'
    elif surprise_pct >= 15:
        return 'BEAT'
    elif surprise_pct <= -50:
        return 'BIG_MISS'
    elif surprise_pct <= -15:
        return 'MISS'
    return 'INLINE'


def _classify_unemployment_signal(data: dict) -> str:
    """Classify unemployment rate signal.

    Low unemployment = strong economy (can be bullish or bearish depending on context)
    High unemployment = recession risk (dovish expectations)
    """
    rate = data.get('rate', 4.0)
    prev = data.get('prev', 4.0)
    change = rate - prev

    if rate >= 5.0:
        return 'CRISIS'
    elif rate >= 4.5:
        return 'DANGER'
    elif change >= 0.3:
        return 'RISING_FAST'
    elif change >= 0.1:
        return 'RISING'
    elif change <= -0.3:
        return 'FALLING_FAST'
    elif change <= -0.1:
        return 'FALLING'
    return 'STABLE'


# ═══════════════════════════════════════════════════════════════
# NFP SEASONALITY (from forensic data, 53 releases 2022-2026)
# ═══════════════════════════════════════════════════════════════

NFP_SEASONALITY = {
    1:  {'avg_move': -0.03, 'avg_abs': 0.95,  'bias': 'NEUTRAL'},
    2:  {'avg_move': +0.84, 'avg_abs': 3.38,  'bias': 'VOLATILE'},
    3:  {'avg_move': -1.56, 'avg_abs': 2.23,  'bias': 'BEARISH'},
    4:  {'avg_move': +1.60, 'avg_abs': 1.60,  'bias': 'BULLISH'},
    5:  {'avg_move': +1.20, 'avg_abs': 1.59,  'bias': 'BULLISH'},
    6:  {'avg_move': -0.28, 'avg_abs': 1.27,  'bias': 'NEUTRAL'},
    7:  {'avg_move': +0.79, 'avg_abs': 0.84,  'bias': 'NEUTRAL'},
    8:  {'avg_move': -1.98, 'avg_abs': 2.04,  'bias': 'BEARISH'},
    9:  {'avg_move': -4.38, 'avg_abs': 4.38,  'bias': 'DANGER'},
    10: {'avg_move': +1.08, 'avg_abs': 1.08,  'bias': 'BULLISH'},
    11: {'avg_move': +2.42, 'avg_abs': 2.73,  'bias': 'BULLISH'},
    12: {'avg_move': +0.60, 'avg_abs': 2.17,  'bias': 'NEUTRAL'},
}

# NFP Asia fade rates (from 53 releases, 2022-2026)
NFP_ASIA_FADE_AFTER_DUMP = 0.60
NFP_ASIA_FADE_AFTER_RALLY = 0.64

# NFP proximity to ADP amplification
NFP_NEAR_ADP_AMPLIFIER = 1.3  # NFP within 5d of ADP has 1.3x larger avg |move|


# ═══════════════════════════════════════════════════════════════
# CONFIRMATION MATRIX
# ═══════════════════════════════════════════════════════════════
# (primary_signal, confirmation_signal) → (move_modifier, confidence_modifier, description)

LABOR_CONFIRMATION_MATRIX = {
    # NFP + ADP agreement = strong signal
    ('STRONG_BEAT', 'STRONG_BEAT'): (+1.50, +0.20, 'ADP confirms strong NFP — double beat'),
    ('STRONG_BEAT', 'BEAT'):        (+0.80, +0.10, 'ADP confirms NFP beat'),
    ('BEAT', 'STRONG_BEAT'):        (+0.80, +0.10, 'ADP confirms NFP beat'),
    ('BEAT', 'BEAT'):               (+0.50, +0.10, 'ADP confirms NFP — solid employment'),
    ('BIG_MISS', 'BIG_MISS'):       (-1.50, +0.20, 'ADP confirms NFP miss — double miss'),
    ('BIG_MISS', 'MISS'):           (-0.80, +0.10, 'ADP confirms NFP miss'),
    ('MISS', 'BIG_MISS'):           (-0.80, +0.10, 'ADP confirms NFP miss'),
    ('MISS', 'MISS'):               (-0.50, +0.10, 'ADP confirms NFP — weakening labor'),
    ('INLINE', 'INLINE'):           (+0.00, +0.05, 'Both inline — no signal'),

    # NFP + ADP disagreement = weak signal (noise)
    ('STRONG_BEAT', 'MISS'):        (+0.00, -0.15, 'ADP contradicts NFP — conflicting signals'),
    ('BEAT', 'MISS'):               (+0.00, -0.10, 'ADP contradicts NFP — weak signal'),
    ('MISS', 'BEAT'):               (+0.00, -0.10, 'ADP contradicts NFP — weak signal'),
    ('BIG_MISS', 'BEAT'):           (+0.00, -0.15, 'ADP contradicts NFP — conflicting signals'),
    ('STRONG_BEAT', 'BIG_MISS'):    (+0.00, -0.20, 'ADP contradicts NFP — maximum conflict'),
    ('BIG_MISS', 'STRONG_BEAT'):    (+0.00, -0.20, 'ADP contradicts NFP — maximum conflict'),
}

# Unemployment rate context modifiers
UNEMPLOYMENT_CONTEXT = {
    'CRISIS':       (-0.80, 'Recession fears — Fed forced to cut'),
    'DANGER':       (-0.40, 'Labor softening — recession whispers'),
    'RISING_FAST':  (-0.30, 'Unemployment rising fast — dovish pivot'),
    'RISING':       (-0.10, 'Unemployment rising — mild headwind'),
    'STABLE':       (+0.00, 'Unemployment stable — no modifier'),
    'FALLING':      (+0.10, 'Unemployment falling — labor market strong'),
    'FALLING_FAST': (+0.20, 'Unemployment falling fast — hawkish risk'),
}

# Claims background modifiers (extremes only)
CLAIMS_BACKGROUND = {
    'CRISIS':       (-1.50, 'Claims crisis — macro dominant'),
    'SPIKE':        (-0.80, 'Claims spike — risk-off'),
    'ELEVATED':     (-0.30, 'Claims elevated — recession fear'),
    'NORMAL':       (+0.00, 'Claims normal — no modifier'),
    'LOW':          (+0.20, 'Claims low — economy strong'),
}


# ═══════════════════════════════════════════════════════════════
# REGIME SENSITIVITY
# ═══════════════════════════════════════════════════════════════

LABOR_REGIME_SENSITIVITY = {
    'TIGHTENING':      0.60,
    'EASING':          0.50,
    'CRISIS_RECOVERY': 0.80,
    'BULL':            0.85,
    'BEAR':            1.20,
    'RECOVERY':        0.75,
    'ACCELERATION':    0.65,
    'STAGFLATION':     1.00,
    'STAGFLATION_HOT': 1.10,
}

# Spike accuracy by year (NFP 1h spike → US session direction)
NFP_SPIKE_ACCURACY = {
    2022: 0.75, 2023: 0.58, 2024: 0.50, 2025: 0.75, 2026: 0.80,
}


# ═══════════════════════════════════════════════════════════════
# BUILD CASCADE ENGINE
# ═══════════════════════════════════════════════════════════════

def _build_labor_cascade() -> CascadeEngine:
    """Build the US Labor cascade engine instance."""
    return CascadeEngine(
        name='US_LABOR',
        description='US Labor Market Chain: JOLTS(demand) → ADP(preview) → NFP(primary) → Unemployment(confirm) → Claims(background)',
        releases=[
            CascadeRelease(
                name='JOLTS',
                schedule_dates=JOLTS_SCHEDULE_DATES,
                weight=0.10,
                role='STRUCTURAL',
                enabled_key='M29_ENABLED',
                release_hour_utc=15,
                release_minute_utc=0,
            ),
            CascadeRelease(
                name='ADP',
                schedule_dates=ADP_SCHEDULE_DATES,
                weight=0.15,
                role='LEADING',
                enabled_key='M36_ENABLED',
                release_hour_utc=12,
                release_minute_utc=15,
                signal_classifier=_classify_adp_signal,
            ),
            CascadeRelease(
                name='NFP',
                schedule_dates=NFP_SCHEDULE_DATES,
                weight=0.35,
                role='PRIMARY',
                enabled_key='M37_ENABLED',
                release_hour_utc=13,
                release_minute_utc=30,
                signal_classifier=_classify_nfp_signal,
            ),
            CascadeRelease(
                name='UNEMPLOYMENT',
                schedule_dates=UNEMPLOYMENT_SCHEDULE_DATES,
                weight=0.20,
                role='CONFIRMATION',
                enabled_key='M62_ENABLED',
                release_hour_utc=13,
                release_minute_utc=30,
                signal_classifier=_classify_unemployment_signal,
            ),
            # Claims are weekly (every Thursday) — handled specially
            CascadeRelease(
                name='CLAIMS',
                schedule_dates=set(),  # claims are every Thursday, handled in engine
                weight=0.10,
                role='BACKGROUND',
                enabled_key='M61_ENABLED',
                release_hour_utc=13,
                release_minute_utc=30,
            ),
        ],
        confirmation_matrix=LABOR_CONFIRMATION_MATRIX,
        regime_sensitivity=LABOR_REGIME_SENSITIVITY,
        default_spike_accuracy=NFP_SPIKE_ACCURACY,
    )


# Singleton
_LABOR_CASCADE = None


def _get_cascade() -> CascadeEngine:
    global _LABOR_CASCADE
    if _LABOR_CASCADE is None:
        _LABOR_CASCADE = _build_labor_cascade()
    return _LABOR_CASCADE


# ═══════════════════════════════════════════════════════════════
# ENRICHED SCORING (beyond base engine)
# ═══════════════════════════════════════════════════════════════

def _enrich_with_nfp_data(details: dict, nfp_date: str, config: dict = None) -> dict:
    """Add NFP-specific forensic data to details."""
    cfg = config or {}

    if nfp_date and nfp_date in NFP_RELEASES:
        nfp_data = NFP_RELEASES[nfp_date]
        details['nfp_release_data'] = nfp_data

        # NFP seasonality
        month = int(nfp_date[5:7])
        season = NFP_SEASONALITY.get(month, {})
        details['nfp_seasonality'] = season

        # NFP near ADP amplification
        details['nfp_near_adp'] = False
        for step in details.get('steps', []):
            if step.get('release') == 'ADP' and step.get('days_since', 99) <= 5:
                details['nfp_near_adp'] = True
                details['nfp_adp_amplifier'] = NFP_NEAR_ADP_AMPLIFIER
                break

    # Claims context
    from src.modules.macro_utils import get_claims_trend
    claims = get_claims_trend(cfg)
    if claims:
        details['claims_context'] = claims
        details['claims_classification'] = claims.get('classification', 'NORMAL')

    return details


def _enrich_with_asia_prediction(details: dict, us_dir: str, regime: str) -> dict:
    """Add Asia session prediction based on NFP forensic data."""
    if us_dir == 'FLAT':
        details['asia_prediction'] = {
            'direction': 'NEUTRAL',
            'confidence': 'LOW',
            'reason': 'US move too small — no edge',
        }
        return details

    # NFP-specific Asia fade rates
    if us_dir == 'DUMP':
        fade_rate = NFP_ASIA_FADE_AFTER_DUMP
        bias = 'FADE' if fade_rate >= 0.55 else 'CONTINUATION'
    else:
        fade_rate = NFP_ASIA_FADE_AFTER_RALLY
        bias = 'FADE' if fade_rate >= 0.55 else 'CONTINUATION'

    # Seasonality adjustment
    season = details.get('nfp_seasonality', {})
    if season.get('bias') == 'DANGER':
        confidence = 'HIGH'
    elif season.get('bias') == 'BULLISH':
        confidence = 'MEDIUM' if us_dir == 'RALLY' else 'LOW'
    else:
        confidence = 'MEDIUM'

    details['asia_prediction'] = {
        'us_direction': us_dir,
        'regime_bias': bias,
        'confidence': confidence,
        'fade_rate': fade_rate,
        'nfp_seasonality': season,
    }

    return details


# ═══════════════════════════════════════════════════════════════
# MAIN SCORING FUNCTION
# ═══════════════════════════════════════════════════════════════

def score_labor_cascade(df_15m, current_time=None, config=None, regime='UNKNOWN',
                        release_data_map=None):
    """Score the US Labor cascade for the current time.

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

    # Inject claims dates (every Thursday)
    today_str = current_time.strftime('%Y-%m-%d')
    dt = current_time
    # Check if today is Thursday (weekday=3)
    if dt.weekday() == 3:
        claims_release = cascade.get_release('CLAIMS')
        if claims_release:
            claims_release.schedule_dates.add(today_str)

    # Run base cascade scoring
    status, score, details, decay = cascade.score(
        df_15m, current_time, config, regime, release_data_map)

    if status == 'SKIP':
        return status, score, details, decay

    # Enrich with NFP-specific data
    nfp_date = None
    for step in details.get('steps', []):
        if step.get('release') == 'NFP':
            nfp_date = step.get('date')
            break
    details = _enrich_with_nfp_data(details, nfp_date, config)

    # Determine US direction for Asia prediction
    us_dir = 'FLAT'
    for step in details.get('steps', []):
        if step.get('us_dir') and step.get('us_dir') != 'FLAT':
            us_dir = step['us_dir']
            break

    # Add Asia prediction (release day) or Asia analysis (post-release)
    has_asia = any(s.get('asia_data') for s in details.get('steps', []))
    if not has_asia:
        details = _enrich_with_asia_prediction(details, us_dir, regime)

    # Build score reason
    result = details.get('result', {})
    steps = details.get('steps', [])
    primary = result.get('primary_signal', '?')
    confirm = result.get('confirmation_signal', '?')
    combined = result.get('combined_signal', '?')

    reason_parts = [f'US_LABOR cascade: {combined}']
    if nfp_date:
        reason_parts.append(f'NFP={nfp_date}')
    for step in steps:
        if step.get('signal') and step.get('signal') not in ('PENDING', 'NEUTRAL'):
            reason_parts.append(f'{step["release"]}={step["signal"]}')
    if details.get('nfp_seasonality'):
        reason_parts.append(f'season={details["nfp_seasonality"].get("bias", "?")}')
    if decay < 1.0:
        reason_parts.append(f'decay={decay:.2f}')

    details['score_reason'] = ', '.join(reason_parts)

    return status, score, details, decay


def format_labor_cascade(details: dict) -> str:
    """Format US Labor cascade details for terminal output."""
    if not details:
        return ''

    # Use base formatter
    output = format_cascade(details)
    if not output:
        return ''

    lines = [output]

    # Add NFP-specific context
    nfp_season = details.get('nfp_seasonality', {})
    if nfp_season:
        bias = nfp_season.get('bias', '?')
        avg_move = nfp_season.get('avg_move', 0)
        bias_icons = {'DANGER': '🔴', 'BEARISH': '🟠', 'BULLISH': '🟢', 'NEUTRAL': '⚪', 'VOLATILE': '🟡'}
        b_icon = bias_icons.get(bias, '⚪')
        lines.append(f"    {b_icon} NFP Seasonality: {bias} (avg {avg_move:+.2f}%)")

    nfp_near_adp = details.get('nfp_near_adp', False)
    if nfp_near_adp:
        lines.append(f"    ⚡ NFP near ADP (within 5d) — signal amplified {NFP_NEAR_ADP_AMPLIFIER}x")

    # Claims context
    claims = details.get('claims_context')
    if claims:
        current = claims.get('current', 0)
        cls = claims.get('classification', '?')
        trend = claims.get('trend', '?')
        cls_icons = {'LOW': '🟢', 'NORMAL': '⚪', 'ELEVATED': '🟡', 'SPIKE': '🟠', 'CRISIS': '🔴'}
        c_icon = cls_icons.get(cls, '⚪')
        lines.append(f"    📋 Claims: {c_icon} {current}K ({cls})  {trend}")

    # Asia prediction
    asia_pred = details.get('asia_prediction', {})
    if asia_pred:
        bias = asia_pred.get('regime_bias', '?')
        conf = asia_pred.get('confidence', '?')
        fade = asia_pred.get('fade_rate', 0)
        conf_icon = {'HIGH': '🟢', 'MEDIUM': '🟡', 'LOW': '🔴'}.get(conf, '⚪')
        lines.append(f"    Asia Prediction: {conf_icon} {bias} (fade={fade:.0%}, conf={conf})")

        if bias == 'FADE':
            us_dir = asia_pred.get('us_direction', '?')
            if us_dir == 'DUMP':
                lines.append(f"    💡 NFP: Asia likely BOUNCES after US dump — watch long at Asia open")
            elif us_dir == 'RALLY':
                lines.append(f"    💡 NFP: Asia likely FADES after US rally — watch short at Asia open")

    return '\n'.join(lines)

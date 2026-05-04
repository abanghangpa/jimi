"""
M18: Short/Long Squeeze Detector.

Detects squeeze setups by combining:
  1. Compression (low vol regime / ATR squeeze)
  2. Extreme positioning (L/S z-score) — determines squeeze direction
  3. Funding rate direction/flip
  4. OI dynamics (rising OI into compression = new positions loading)
  5. Whale divergence (smart money against crowd)
  6. M4b timing (intrabar CVD confirming momentum)

When squeeze fires, it overrides the M9 regime block and boosts ICS.
"""

import numpy as np


SQUEEZE_DEFAULTS = {
    # Compression detection
    'SQUEEZE_ATR_PCTL_MAX': 0.50,
    'SQUEEZE_RANGE_WIDTH_MAX': 0.04,

    # Positioning extremes (z-score sign determines squeeze direction)
    # Negative z = shorts overcrowded → SHORT_SQUEEZE (price shoots up)
    # Positive z = longs overcrowded → LONG_SQUEEZE (price dumps)
    'SQUEEZE_LS_ZSCORE_MIN': 1.8,
    'SQUEEZE_LS_ZSCORE_STRONG': 2.5,

    # Funding
    'SQUEEZE_FUNDING_FLIP_BONUS': 0.15,

    # OI
    'SQUEEZE_OI_RISING_THRESHOLD': 0.3,
    'SQUEEZE_OI_RISING_STRONG': 1.0,

    # Thresholds
    'SQUEEZE_SCORE_THRESHOLD': 0.55,
    'SQUEEZE_STRONG_THRESHOLD': 0.70,

    # Override
    'SQUEEZE_OVERRIDE_REGIME': True,
    'SQUEEZE_ICS_BOOST': 0.08,
    'SQUEEZE_SIZE_MULT': 0.80,
}


def _score_short_squeeze(cfg, regime, ls_zscore, funding_rate, oi_roc,
                         whale, futures_flow, m4b_div, m4b_bars_ago):
    """Score SHORT squeeze potential (shorts overcrowded → price shoots up).

    Requires negative z-score (shorts crowded). Other factors add confidence.
    """
    score = 0.0
    factors = []

    # Gate: must have short crowding (negative z)
    if ls_zscore > -cfg['SQUEEZE_LS_ZSCORE_MIN']:
        return 0.0, []

    # 1. Compression
    is_compressed = regime in ('NEUTRAL_CHOP', 'CHOP_MILD_BEAR', 'CHOP_MILD_BULL')
    if is_compressed:
        score += 0.15
        factors.append(f'Regime={regime} (compressed)')

    # 2. Short crowding severity (z-score)
    if ls_zscore <= -cfg['SQUEEZE_LS_ZSCORE_STRONG']:
        score += 0.25
        factors.append(f'L/S z={ls_zscore:.2f} (extreme short crowding)')
    else:
        score += 0.15
        factors.append(f'L/S z={ls_zscore:.2f} (short crowding)')

    # 3. Funding negative = shorts paying (confirms short pressure)
    if funding_rate is not None and funding_rate < 0:
        score += 0.10
        factors.append(f'Funding {funding_rate*100:+.4f}% (shorts paying)')
    elif funding_rate is not None and funding_rate > 0:
        # Positive funding but shorts still crowded = flip in progress
        if funding_rate < 0.002:
            score += cfg['SQUEEZE_FUNDING_FLIP_BONUS']
            factors.append(f'Funding {funding_rate*100:+.4f}% (flipping — shorts capitulating)')

    # 4. OI rising = new shorts loading into compression
    if oi_roc > cfg['SQUEEZE_OI_RISING_STRONG']:
        score += 0.15
        factors.append(f'OI rising {oi_roc:+.2f}%/hr (heavy short loading)')
    elif oi_roc > cfg['SQUEEZE_OI_RISING_THRESHOLD']:
        score += 0.08
        factors.append(f'OI rising {oi_roc:+.2f}%/hr (shorts opening)')

    # 5. Whales bullish = smart money betting against crowd shorts
    if whale == 'WHALE_BULLISH':
        score += 0.15
        factors.append('Whales bullish (against crowd shorts)')
    elif futures_flow == 'BUYERS_DOMINANT':
        score += 0.08
        factors.append('Futures buyers dominant')

    # 6. M4b bullish divergence = squeeze momentum building
    if m4b_div == 'BULLISH' and m4b_bars_ago <= 24:
        score += 0.10
        factors.append(f'M4b bullish div {m4b_bars_ago} bars ago (squeeze momentum)')

    return min(score, 1.0), factors


def _score_long_squeeze(cfg, regime, ls_zscore, funding_rate, oi_roc,
                        whale, futures_flow, m4b_div, m4b_bars_ago):
    """Score LONG squeeze potential (longs overcrowded → price dumps).

    Requires positive z-score (longs crowded). Other factors add confidence.
    """
    score = 0.0
    factors = []

    # Gate: must have long crowding (positive z)
    if ls_zscore < cfg['SQUEEZE_LS_ZSCORE_MIN']:
        return 0.0, []

    # 1. Compression
    is_compressed = regime in ('NEUTRAL_CHOP', 'CHOP_MILD_BEAR', 'CHOP_MILD_BULL')
    if is_compressed:
        score += 0.15
        factors.append(f'Regime={regime} (compressed)')

    # 2. Long crowding severity (z-score)
    if ls_zscore >= cfg['SQUEEZE_LS_ZSCORE_STRONG']:
        score += 0.25
        factors.append(f'L/S z={ls_zscore:.2f} (extreme long crowding)')
    else:
        score += 0.15
        factors.append(f'L/S z={ls_zscore:.2f} (long crowding)')

    # 3. Funding positive = longs paying (confirms long pressure)
    if funding_rate is not None and funding_rate > 0:
        score += 0.10
        factors.append(f'Funding {funding_rate*100:+.4f}% (longs paying)')
    elif funding_rate is not None and funding_rate < 0:
        # Negative funding but longs still crowded = flip in progress
        if funding_rate > -0.002:
            score += cfg['SQUEEZE_FUNDING_FLIP_BONUS']
            factors.append(f'Funding {funding_rate*100:+.4f}% (flipping — longs capitulating)')

    # 4. OI rising = new longs loading into compression
    if oi_roc > cfg['SQUEEZE_OI_RISING_STRONG']:
        score += 0.15
        factors.append(f'OI rising {oi_roc:+.2f}%/hr (heavy long loading)')
    elif oi_roc > cfg['SQUEEZE_OI_RISING_THRESHOLD']:
        score += 0.08
        factors.append(f'OI rising {oi_roc:+.2f}%/hr (longs opening)')

    # 5. Whales bearish = smart money betting against crowd longs
    if whale == 'WHALE_BEARISH':
        score += 0.15
        factors.append('Whales bearish (against crowd longs)')
    elif futures_flow == 'SELLERS_DOMINANT':
        score += 0.08
        factors.append('Futures sellers dominant')

    # 6. M4b bearish divergence = dump momentum building
    if m4b_div == 'BEARISH' and m4b_bars_ago <= 24:
        score += 0.10
        factors.append(f'M4b bearish div {m4b_bars_ago} bars ago (dump momentum)')

    return min(score, 1.0), factors


def detect_squeeze(result, config=None):
    """Detect squeeze setup from scan results.

    Args:
        result: scan_signal() output dict
        config: Optional config overrides

    Returns:
        dict with squeeze_type, score, direction, factors, override info
    """
    cfg = {**SQUEEZE_DEFAULTS, **(config or {})}

    deriv = result.get('derivatives', {})
    m9 = result.get('m9', {})
    m4b = result.get('m4b', {})

    regime = m9.get('regime', 'UNKNOWN')
    ls_zscore = deriv.get('ls_zscore', 0)
    funding_rate = deriv.get('funding_rate')
    oi_roc = deriv.get('oi_roc_1h', 0)
    whale = deriv.get('whale_signal', 'NEUTRAL')
    futures_flow = deriv.get('futures_flow', 'NEUTRAL')
    m4b_div = m4b.get('divergence', 'NONE')
    m4b_bars_ago = m4b.get('bars_ago', -1)

    # Score both directions — z-score sign determines which is valid
    short_score, short_factors = _score_short_squeeze(
        cfg, regime, ls_zscore, funding_rate, oi_roc,
        whale, futures_flow, m4b_div, m4b_bars_ago)

    long_score, long_factors = _score_long_squeeze(
        cfg, regime, ls_zscore, funding_rate, oi_roc,
        whale, futures_flow, m4b_div, m4b_bars_ago)

    threshold = cfg['SQUEEZE_SCORE_THRESHOLD']
    strong_threshold = cfg['SQUEEZE_STRONG_THRESHOLD']

    # Pick the higher score (only one direction can be valid due to z-score gating)
    if short_score >= threshold and short_score > long_score:
        squeeze_type = 'SHORT_SQUEEZE'
        squeeze_score = short_score
        direction = 'LONG'  # squeeze pushes price UP
        factors = short_factors
    elif long_score >= threshold and long_score > short_score:
        squeeze_type = 'LONG_SQUEEZE'
        squeeze_score = long_score
        direction = 'SHORT'  # squeeze pushes price DOWN
        factors = long_factors
    else:
        squeeze_type = 'NONE'
        squeeze_score = max(short_score, long_score)
        direction = 'NEUTRAL'
        factors = short_factors if short_score > long_score else long_factors

    is_strong = squeeze_score >= strong_threshold
    overrides_regime = (squeeze_type != 'NONE' and cfg['SQUEEZE_OVERRIDE_REGIME'])

    ics_boost = 0.0
    if squeeze_type != 'NONE':
        ics_boost = cfg['SQUEEZE_ICS_BOOST']
        if is_strong:
            ics_boost *= 1.5

    return {
        'squeeze_type': squeeze_type,
        'squeeze_score': round(squeeze_score, 3),
        'squeeze_strong': is_strong,
        'direction': direction,
        'factors': factors,
        'overrides_regime': overrides_regime,
        'ics_boost': round(ics_boost, 4),
        'size_mult': cfg['SQUEEZE_SIZE_MULT'] if squeeze_type != 'NONE' else 1.0,
        'short_score': round(short_score, 3),
        'long_score': round(long_score, 3),
    }


def format_squeeze(sq):
    """Format squeeze result for terminal output."""
    if sq['squeeze_type'] == 'NONE':
        return ''

    lines = []
    lines.append('')
    lines.append('  🔥 SQUEEZE DETECTOR')

    type_icons = {
        'SHORT_SQUEEZE': '🟩',
        'LONG_SQUEEZE': '🟥',
    }
    icon = type_icons.get(sq['squeeze_type'], '❓')
    strength = 'STRONG' if sq['squeeze_strong'] else 'MODERATE'

    lines.append(f'  Type: {icon} {sq["squeeze_type"]}  ({strength})')
    lines.append(f'  Score: {sq["squeeze_score"]:.3f}  Direction: {sq["direction"]}')
    lines.append(f'  Short squeeze score: {sq["short_score"]:.3f}  Long squeeze score: {sq["long_score"]:.3f}')

    if sq['overrides_regime']:
        lines.append(f'  ⚡ Overrides M9 regime block!')

    if sq['factors']:
        lines.append(f'\n  Factors:')
        for f in sq['factors']:
            lines.append(f'    ✅ {f}')

    if sq['ics_boost'] > 0:
        lines.append(f'\n  ICS boost: +{sq["ics_boost"]:.4f}')
        lines.append(f'  Size multiplier: {sq["size_mult"]:.2f}x')

    return '\n'.join(lines)

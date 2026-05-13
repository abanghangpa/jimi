"""
Historical Sweep Pattern Module

Provides statistical context for conflict resolution by looking up
what historically happens when price is at range bottom with similar
conditions.

Core data point:
  When price is at range bottom with this setup, the sweep-down-then-bounce
  path is the dominant pattern (95% sweep rate, 42% bounce rate).

This module doesn't generate signals — it resolves conflicts between
other modules by providing an objective historical baseline.
"""

import numpy as np


# ═══════════════════════════════════════════════════════════════
# HISTORICAL PATTERN DATABASE
# ═══════════════════════════════════════════════════════════════

# Patterns keyed by (position_in_range, regime, phase)
# Each entry: sweep_rate, bounce_rate, avg_sweep_time, avg_bounce_rr, sample_size

PATTERNS = {
    # Range bottom + accumulation/neutral → sweep-down-then-bounce
    ('range_bottom', 'NEUTRAL_CHOP', 'ACCUMULATION'): {
        'sweep_rate': 0.95,
        'bounce_rate': 0.42,
        'avg_sweep_bars': 8,        # ~2h on 15m
        'avg_bounce_rr': 1.58,
        'expected_path': 'SWEEP_DOWN_THEN_BOUNCE',
        'confidence': 'HIGH',
        'sample_size': 429,
        'description': '95% sweep the low, 42% bounce to range top (1.58x R:R)',
    },
    ('range_bottom', 'NEUTRAL_CHOP', 'RANGE'): {
        'sweep_rate': 0.92,
        'bounce_rate': 0.45,
        'avg_sweep_bars': 6,
        'avg_bounce_rr': 1.50,
        'expected_path': 'SWEEP_DOWN_THEN_BOUNCE',
        'confidence': 'HIGH',
        'sample_size': 312,
        'description': '92% sweep the low, 45% bounce (1.50x R:R)',
    },
    ('range_bottom', 'TRENDING', 'ACCUMULATION'): {
        'sweep_rate': 0.88,
        'bounce_rate': 0.55,
        'avg_sweep_bars': 4,
        'avg_bounce_rr': 1.80,
        'expected_path': 'SWEEP_DOWN_THEN_BOUNCE',
        'confidence': 'MEDIUM',
        'sample_size': 187,
        'description': '88% sweep, 55% bounce — trending regime gives stronger bounces',
    },
    # Range top + distribution → sweep-up-then-reject
    ('range_top', 'NEUTRAL_CHOP', 'DISTRIBUTION'): {
        'sweep_rate': 0.93,
        'bounce_rate': 0.48,
        'avg_sweep_bars': 6,
        'avg_bounce_rr': 1.65,
        'expected_path': 'SWEEP_UP_THEN_REJECT',
        'confidence': 'HIGH',
        'sample_size': 385,
        'description': '93% sweep the high, 48% reject to range bottom (1.65x R:R)',
    },
    ('range_top', 'NEUTRAL_CHOP', 'RANGE'): {
        'sweep_rate': 0.90,
        'bounce_rate': 0.50,
        'avg_sweep_bars': 5,
        'avg_bounce_rr': 1.55,
        'expected_path': 'SWEEP_UP_THEN_REJECT',
        'confidence': 'HIGH',
        'sample_size': 298,
        'description': '90% sweep the high, 50% reject (1.55x R:R)',
    },
    # Mid-range → no strong pattern
    ('mid_range', None, None): {
        'sweep_rate': 0.50,
        'bounce_rate': 0.35,
        'avg_sweep_bars': 12,
        'avg_bounce_rr': 1.10,
        'expected_path': 'NO_DOMINANT_PATH',
        'confidence': 'LOW',
        'sample_size': 0,
        'description': 'No dominant pattern at mid-range — wait for edge',
    },
}

# Regime adjustments (multiply bounce_rate)
REGIME_MULTIPLIER = {
    'CRISIS': 0.60,       # crisis = bounces fail more
    'CHOP_HARD': 0.75,    # hard chop = noisy sweeps
    'CHOP_MILD': 1.00,    # baseline
    'NEUTRAL_CHOP': 1.00,
    'NEUTRAL': 1.05,
    'TRENDING': 1.20,     # trending = cleaner bounces
    'COMPRESSING': 0.90,  # compression = delayed bounces
}

# Phase0 adjustments (multiply bounce_rate)
PHASE0_MULTIPLIER = {
    'death_zone': 0.70,   # phase0 < 0.15 = weak context
    'low': 0.85,          # phase0 0.15-0.30
    'normal': 1.00,       # phase0 0.30-0.60
    'strong': 1.10,       # phase0 > 0.60
}


def _classify_position(price, sr_levels, magnets, range_info=None):
    """Determine if price is at range bottom, top, or mid.

    Returns: 'range_bottom', 'range_top', or 'mid_range'
    """
    if not sr_levels and not magnets and not range_info:
        return 'mid_range'

    # Use range info if available (from M21)
    if range_info:
        range_lo = range_info.get('range_lo', 0)
        range_hi = range_info.get('range_hi', 0)
        eq = range_info.get('eq', 0)
        if range_lo and range_hi:
            pos_pct = (price - range_lo) / (range_hi - range_lo) if range_hi > range_lo else 0.5
            if pos_pct <= 0.25:
                return 'range_bottom'
            elif pos_pct >= 0.75:
                return 'range_top'
            return 'mid_range'

    # Fallback: use S/R levels
    if sr_levels:
        supports = sorted([p for p, s, t, _, _ in sr_levels if t == 'SUPPORT'])
        resistances = sorted([p for p, s, t, _, _ in sr_levels if t == 'RESISTANCE'])
        if supports and resistances:
            nearest_sup = min(supports, key=lambda x: abs(x - price))
            nearest_res = min(resistances, key=lambda x: abs(x - price))
            dist_to_sup = abs(price - nearest_sup) / price
            dist_to_res = abs(nearest_res - price) / price
            if dist_to_sup < dist_to_res * 0.5:
                return 'range_bottom'
            elif dist_to_res < dist_to_sup * 0.5:
                return 'range_top'

    # Fallback: use magnets
    if magnets:
        below = [(p, s) for p, s, *_ in magnets if p < price]
        above = [(p, s) for p, s, *_ in magnets if p > price]
        if below and above:
            nearest_below = min(below, key=lambda x: price - x[0])
            nearest_above = min(above, key=lambda x: x[0] - price)
            below_dist = price - nearest_below[0]
            above_dist = nearest_above[0] - price
            if below_dist < above_dist * 0.5:
                return 'range_bottom'
            elif above_dist < below_dist * 0.5:
                return 'range_top'

    return 'mid_range'


def _classify_phase0(phase0):
    """Classify Phase0 value into regime bucket."""
    if phase0 is None:
        return 'normal'
    if phase0 < 0.15:
        return 'death_zone'
    elif phase0 < 0.30:
        return 'low'
    elif phase0 < 0.60:
        return 'normal'
    return 'strong'


def get_historical_pattern(result, config=None):
    """Look up the historical pattern for the current market configuration.

    Args:
        result: scan_signal() output dict
        config: optional config overrides

    Returns:
        dict with pattern data, or None if no pattern applies
    """
    price = result.get('price', 0)
    swing_bias = result.get('swing_bias', 'NEUTRAL')
    phase0 = result.get('phase0')
    regime = result.get('m9', {}).get('regime', 'UNKNOWN')
    m21 = result.get('m21', {})
    m21_phase = m21.get('phase', 'RANGE') if m21 else 'RANGE'
    sr_levels = result.get('sr_levels', [])
    magnets = result.get('magnets', [])
    direction = result.get('direction', 'NEUTRAL')

    # Get range info from M21 if available
    range_info = None
    if m21 and m21.get('details'):
        range_info = m21['details'].get('range_info')

    # Classify position
    position = _classify_position(price, sr_levels, magnets, range_info)

    # Look up pattern
    pattern = PATTERNS.get((position, regime, m21_phase))
    if pattern is None:
        # Try with None phase
        pattern = PATTERNS.get((position, regime, None))
    if pattern is None:
        # Try with None regime
        pattern = PATTERNS.get((position, None, None))
    if pattern is None:
        pattern = PATTERNS[('mid_range', None, None)]

    # Apply adjustments
    regime_mult = REGIME_MULTIPLIER.get(regime, 1.0)
    phase0_bucket = _classify_phase0(phase0)
    phase0_mult = PHASE0_MULTIPLIER.get(phase0_bucket, 1.0)

    adjusted_bounce = min(1.0, pattern['bounce_rate'] * regime_mult * phase0_mult)
    adjusted_sweep = pattern['sweep_rate']  # sweep rate is stable

    # Determine conflict resolution direction
    resolution_hint = 'NEUTRAL'
    if position == 'range_bottom' and adjusted_sweep > 0.80:
        # Historical: sweep first, then bounce
        resolution_hint = 'SWEEP_FIRST_THEN_LONG'
    elif position == 'range_top' and adjusted_sweep > 0.80:
        resolution_hint = 'SWEEP_FIRST_THEN_SHORT'

    return {
        'position': position,
        'pattern_key': (position, regime, m21_phase),
        'expected_path': pattern['expected_path'],
        'sweep_rate': round(adjusted_sweep, 2),
        'bounce_rate': round(adjusted_bounce, 2),
        'avg_sweep_bars': pattern['avg_sweep_bars'],
        'avg_bounce_rr': pattern['avg_bounce_rr'],
        'confidence': pattern['confidence'],
        'sample_size': pattern['sample_size'],
        'description': pattern['description'],
        'regime_mult': regime_mult,
        'phase0_mult': phase0_mult,
        'resolution_hint': resolution_hint,
    }


def format_historical_pattern(hp):
    """Format historical pattern for terminal output."""
    if not hp:
        return ''

    lines = []
    lines.append(f"\n  📊 HISTORICAL PATTERN")
    lines.append(f"  Position: {hp['position'].replace('_', ' ').title()}")
    lines.append(f"  Expected path: {hp['expected_path']}")
    lines.append(f"  Sweep rate: {hp['sweep_rate']:.0%}  |  Bounce rate: {hp['bounce_rate']:.0%}")
    lines.append(f"  Avg sweep time: {hp['avg_sweep_bars']} bars (~{hp['avg_sweep_bars'] * 15}min)")
    lines.append(f"  Avg bounce R:R: {hp['avg_bounce_rr']:.2f}x")
    lines.append(f"  Confidence: {hp['confidence']}  (n={hp['sample_size']})")

    if hp['resolution_hint'] == 'SWEEP_FIRST_THEN_LONG':
        lines.append(f"  → Resolution: Sweep the low FIRST, then look for LONG bounce")
    elif hp['resolution_hint'] == 'SWEEP_FIRST_THEN_SHORT':
        lines.append(f"  → Resolution: Sweep the high FIRST, then look for SHORT rejection")

    adjustments = []
    if hp['regime_mult'] != 1.0:
        adjustments.append(f"regime ×{hp['regime_mult']:.2f}")
    if hp['phase0_mult'] != 1.0:
        adjustments.append(f"phase0 ×{hp['phase0_mult']:.2f}")
    if adjustments:
        lines.append(f"  Adjustments: {', '.join(adjustments)}")

    return '\n'.join(lines)

"""
Direction Resolver — Climate → Direction → Execution pipeline.

Combines:
  Phase 1: M9  → What's the market climate? (regime)
  Phase 2: M13 → What's the structural bias? (swing direction)
  Phase 2: M7  → What's the macro bias? (ETH/BTC + BTC vol)
  Phase 3: Unified direction + size multiplier for downstream modules

This runs BEFORE the ICS scoring loop, replacing the old M1→direction path.
"""

import numpy as np


# ═══════════════════════════════════════════════════════════════
# REGIME → SIZE MULTIPLIER
# ═══════════════════════════════════════════════════════════════

REGIME_SIZE_MAP = {
    'CRISIS': 0.0,        # Hard block — no trades
    'CHOP_HARD': 0.15,    # Near-block — tiny size if anything
    'CHOP_MILD': 0.50,    # Reduced — trade small
    'COMPRESSING': 0.80,  # Slightly reduced — waiting for breakout
    'TRENDING': 1.0,      # Full size — this is the edge
    'NEUTRAL': 0.65,      # Moderate — some uncertainty
    'UNKNOWN': 0.50,      # Conservative default
}


def resolve_direction(regime, regime_score, m13_bias, m13_score,
                      m13_details, m7_score=None, m7_status=None,
                      swing_bias_1d=None, trend_dir=None,
                      config=None):
    """
    Resolve unified direction and sizing from regime + structure + macro.

    Called once per bar, BEFORE the ICS scoring loop.

    Args:
        regime: M9 regime output ('CRISIS', 'CHOP_HARD', etc.)
        regime_score: M9 regime score (0.0-1.0)
        m13_bias: M13 structure bias ('BULLISH', 'BEARISH', 'NEUTRAL')
        m13_score: M13 structure confidence (0.0-1.0)
        m13_details: M13 details dict
        m7_score: M7 macro score (optional)
        m7_status: M7 macro status (optional)
        swing_bias_1d: Daily swing bias from calc_swing_bias (optional)
        trend_dir: Daily trend direction (optional)

    Returns:
        (direction, size_mult, details)
        direction: 'LONG' | 'SHORT' | 'NEUTRAL'
        size_mult: 0.0-1.0 (regime-adjusted position size multiplier)
        details: dict with full decision trace
    """
    cfg = config or {}
    details = {
        'regime': regime,
        'regime_score': round(regime_score, 3),
        'm13_bias': m13_bias,
        'm13_score': round(m13_score, 3),
    }

    # ── Phase 1: Regime Gate ──
    size_mult = REGIME_SIZE_MAP.get(regime, 0.50)

    block_regimes = cfg.get('M9_BLOCK_REGIMES', ['CRISIS'])
    if regime in block_regimes:
        details['action'] = 'BLOCKED'
        details['reason'] = f'regime={regime} is in block list'
        return 'NEUTRAL', 0.0, details

    # ── Phase 2: Direction from Structure ──
    # Primary: M13 HTF structure
    direction = 'NEUTRAL'

    if m13_bias in ('BULLISH', 'LEAN_BULL'):
        direction = 'LONG'
    elif m13_bias in ('BEARISH', 'LEAN_BEAR'):
        direction = 'SHORT'

    # ── Phase 2b: Macro Confirmation (M7) ──
    # If M7 strongly disagrees with structure, downgrade
    if m7_score is not None and m7_status not in ('SKIP', None):
        m7_direction = 'NEUTRAL'
        if m7_score >= 0.65:
            # M7 is bullish
            if direction == 'LONG':
                m7_direction = 'AGREE'
            elif direction == 'SHORT':
                m7_direction = 'CONFLICT'
        elif m7_score <= 0.35:
            # M7 is bearish
            if direction == 'SHORT':
                m7_direction = 'AGREE'
            elif direction == 'LONG':
                m7_direction = 'CONFLICT'

        details['m7_direction'] = m7_direction
        details['m7_score'] = round(m7_score, 3)

        if m7_direction == 'CONFLICT':
            # Macro contradicts structure — reduce confidence
            size_mult *= 0.70
            details['m7_conflict_penalty'] = 0.70
        elif m7_direction == 'AGREE':
            # Macro confirms — slight boost
            size_mult = min(size_mult * 1.10, 1.0)
            details['m7_agree_bonus'] = 1.10

    # ── Phase 2c: Daily Swing Bias Confirmation ──
    if swing_bias_1d is not None and direction != 'NEUTRAL':
        daily_agrees = (
            (direction == 'LONG' and swing_bias_1d in ('BULLISH', 'LEAN_BULL')) or
            (direction == 'SHORT' and swing_bias_1d in ('BEARISH', 'LEAN_BEAR'))
        )
        daily_conflicts = (
            (direction == 'LONG' and swing_bias_1d in ('BEARISH', 'LEAN_BEAR')) or
            (direction == 'SHORT' and swing_bias_1d in ('BULLISH', 'LEAN_BULL'))
        )
        details['daily_swing'] = swing_bias_1d

        if daily_conflicts:
            size_mult *= 0.75
            details['daily_conflict_penalty'] = 0.75
        elif daily_agrees:
            size_mult = min(size_mult * 1.05, 1.0)
            details['daily_agree_bonus'] = 1.05

    # ── Phase 2d: Trend Direction Confirmation ──
    if trend_dir is not None and direction != 'NEUTRAL':
        trend_agrees = (
            (direction == 'LONG' and trend_dir in ('STRONG_UP', 'UP')) or
            (direction == 'SHORT' and trend_dir in ('STRONG_DOWN', 'DOWN'))
        )
        trend_conflicts = (
            (direction == 'LONG' and trend_dir in ('STRONG_DOWN', 'DOWN')) or
            (direction == 'SHORT' and trend_dir in ('STRONG_UP', 'UP'))
        )
        details['trend_dir'] = trend_dir

        if trend_conflicts:
            size_mult *= 0.70
            details['trend_conflict_penalty'] = 0.70
        elif trend_agrees:
            size_mult = min(size_mult * 1.10, 1.0)
            details['trend_agree_bonus'] = 1.10

    # ── Phase 3: Regime-Specific Adjustments ──
    if regime == 'TRENDING':
        # In trending regime, structure alignment is critical
        if direction != 'NEUTRAL' and m13_score >= 0.70:
            size_mult = min(size_mult * 1.15, 1.0)
            details['trending_structure_bonus'] = 1.15

    elif regime == 'COMPRESSING':
        # In compression, wait for breakout — don't pick direction yet
        # Reduce size regardless of structure
        if m13_details.get('fvg_count', 0) > 0:
            # FVGs present near squeeze — breakout is loading
            details['squeeze_fvg_hint'] = True

    elif regime == 'CHOP_MILD':
        # In chop, only trade at structure extremes
        # This is handled by entry_optimizer, but flag it
        details['chop_mode'] = True
        details['chop_advice'] = 'Trade only at swing extremes (range fade)'

    elif regime == 'CHOP_HARD':
        # Near-block — direction doesn't matter much
        direction = 'NEUTRAL'
        size_mult = 0.0
        details['action'] = 'BLOCKED'
        details['reason'] = 'CHOP_HARD — no edge'

    # ── Final: Clamp ──
    size_mult = max(0.0, min(1.0, size_mult))

    if direction == 'NEUTRAL':
        details['action'] = 'NO_BIAS'
        details['reason'] = 'No structural direction — skip'
    else:
        details['action'] = 'TRADEABLE'
        details['reason'] = f'{regime} + {m13_bias} structure → {direction}'

    details['size_mult'] = round(size_mult, 3)
    return direction, size_mult, details


def format_direction_summary(direction, size_mult, details):
    """Format direction resolver output for logging."""
    lines = []
    regime = details.get('regime', '?')
    m13_bias = details.get('m13_bias', '?')
    action = details.get('action', '?')

    lines.append(f"  Regime: {regime} (score={details.get('regime_score', 0):.3f})")
    lines.append(f"  Structure: {m13_bias} (score={details.get('m13_score', 0):.3f})")

    if 'm7_score' in details:
        lines.append(f"  Macro M7: {details['m7_score']:.3f} ({details.get('m7_direction', '?')})")
    if 'daily_swing' in details:
        lines.append(f"  Daily Swing: {details['daily_swing']}")
    if 'trend_dir' in details:
        lines.append(f"  Trend: {details['trend_dir']}")

    lines.append(f"  → Direction: {direction} | Size: {size_mult:.2f} | {action}")

    # Penalties/bonuses
    for key in sorted(details.keys()):
        if key.endswith('_penalty') or key.endswith('_bonus'):
            lines.append(f"    {key}: {details[key]}")

    if 'reason' in details:
        lines.append(f"  Reason: {details['reason']}")

    return '\n'.join(lines)

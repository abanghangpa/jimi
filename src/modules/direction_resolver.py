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
# TARGET SCORING — "If I go this way, what's ahead?"
# ═══════════════════════════════════════════════════════════════

def score_targets(current_price, magnets, gaps, sr_levels, direction, atr_1h=None):
    """Score how attractive a direction's targets are.

    Looks at what's in front of price for a given direction:
    - Volume profile magnets (HVNs) → absorption zones / targets
    - Volume gaps (LVNs) → vacuum zones price moves through fast
    - S/R levels → structural targets (resistance for LONG, support for SHORT)

    Args:
        current_price: float
        magnets: list of (price, vol, strength) from find_magnets()
        gaps: list of (price, vol) from find_gaps()
        sr_levels: list of (price, strength, touches, bounces, type) from find_support_resistance()
        direction: 'LONG' or 'SHORT'
        atr_1h: float, optional — for distance normalization

    Returns:
        (score, details) where score is 0.0-1.0
    """
    score = 0.0
    details = {'direction': direction, 'targets': [], 'gaps': [], 'sr': []}

    # Distance multiplier — closer targets are more actionable
    # Use ATR * 2 as cap (tighter), with 2% floor for low-vol environments
    dist_cap = max(atr_1h * 2 if atr_1h else 0, current_price * 0.02)
    # Minimum target distance: 0.3% of price (ignore noise)
    min_dist = current_price * 0.003

    # ── Magnets (HVNs) in direction ──
    for price, vol, strength in magnets:
        if direction == 'LONG' and price <= current_price:
            continue
        if direction == 'SHORT' and price >= current_price:
            continue
        dist = abs(price - current_price)
        if dist < min_dist:
            continue  # too close — noise, not a target
        dist_norm = min(dist / dist_cap, 1.0) if dist_cap > 0 else 1.0
        # Closer + stronger = higher contribution
        proximity = max(0, 1.0 - dist_norm)
        contrib = proximity * min(strength / 3.0, 1.0) * 0.4
        score += contrib
        details['targets'].append({
            'price': round(price, 2), 'strength': round(strength, 2),
            'dist_pct': round(dist / current_price * 100, 2),
            'contrib': round(contrib, 3),
        })

    # ── Gaps (LVNs) — vacuum zones price will move through ──
    for price, vol in gaps:
        if direction == 'LONG' and price <= current_price:
            continue
        if direction == 'SHORT' and price >= current_price:
            continue
        dist = abs(price - current_price)
        if dist < min_dist:
            continue
        dist_norm = min(dist / dist_cap, 1.0) if dist_cap > 0 else 1.0
        proximity = max(0, 1.0 - dist_norm)
        contrib = proximity * 0.2  # gaps add less than magnets
        score += contrib
        details['gaps'].append({
            'price': round(price, 2), 'dist_pct': round(dist / current_price * 100, 2),
            'contrib': round(contrib, 3),
        })

    # ── S/R levels — structural targets ──
    target_type = 'RESISTANCE' if direction == 'LONG' else 'SUPPORT'
    for level in sr_levels:
        price, strength, touches, bounces, sr_type = level
        if sr_type != target_type:
            continue
        if direction == 'LONG' and price <= current_price:
            continue
        if direction == 'SHORT' and price >= current_price:
            continue
        dist = abs(price - current_price)
        if dist < min_dist:
            continue
        dist_norm = min(dist / dist_cap, 1.0) if dist_cap > 0 else 1.0
        proximity = max(0, 1.0 - dist_norm)
        # Stronger S/R = more reliable target
        sr_strength = min(strength / 10.0, 1.0)
        contrib = proximity * sr_strength * 0.3
        score += contrib
        details['sr'].append({
            'price': round(price, 2), 'strength': round(strength, 2),
            'touches': touches, 'bounces': bounces,
            'dist_pct': round(dist / current_price * 100, 2),
            'contrib': round(contrib, 3),
        })

    score = min(score, 1.0)
    details['score'] = round(score, 3)
    return score, details


def resolve_target_tiebreaker(long_score, short_score, details_long, details_short):
    """Use target scores to break ties or boost confidence.

    Returns:
        (preferred, delta, reason)
        preferred: 'LONG' | 'SHORT' | None
        delta: absolute score difference
        reason: human-readable explanation
    """
    delta = abs(long_score - short_score)

    if delta < 0.05:
        return None, delta, 'Targets too close to call'

    preferred = 'LONG' if long_score > short_score else 'SHORT'
    best = details_long if preferred == 'LONG' else details_short

    # Summarize the best targets
    top_targets = sorted(best.get('targets', []), key=lambda x: -x.get('contrib', 0))[:2]
    top_sr = sorted(best.get('sr', []), key=lambda x: -x.get('contrib', 0))[:1]

    parts = []
    if top_targets:
        t = top_targets[0]
        parts.append(f"HVN ${t['price']:.0f} ({t['dist_pct']:+.1f}%)")
    if top_sr:
        s = top_sr[0]
        parts.append(f"S/R ${s['price']:.0f} ({s['dist_pct']:+.1f}%)")

    reason = f'{preferred} targets clearer: {", ".join(parts)}' if parts else f'{preferred} has better targets'
    return preferred, delta, reason


# ═══════════════════════════════════════════════════════════════
# REGIME → SIZE MULTIPLIER
# ═══════════════════════════════════════════════════════════════

REGIME_SIZE_MAP = {
    'CRISIS': 0.0,        # Hard block — no trades
    'CHOP_HARD': 0.15,    # Near-block — tiny size if anything
    'CHOP_MILD': 0.50,    # Reduced — trade small
    'CHOP_MILD_BEAR': 0.55,  # Directional chop — slightly more confident
    'CHOP_MILD_BULL': 0.55,  # Directional chop — slightly more confident
    'COMPRESSING': 0.80,  # Slightly reduced — waiting for breakout
    'TRENDING': 1.0,      # Full size — this is the edge
    'NEUTRAL': 0.65,      # Moderate — some uncertainty
    'NEUTRAL_TRENDING': 0.70,  # Has momentum but no direction
    'NEUTRAL_TRENDING_BULL': 0.70,  # Momentum + bullish lean
    'NEUTRAL_TRENDING_BEAR': 0.70,  # Momentum + bearish lean
    'NEUTRAL_CHOP': 0.55,     # Noise — reduced
    'UNKNOWN': 0.50,      # Conservative default
}


def resolve_direction(regime, regime_score, m13_bias, m13_score,
                      m13_details, m7_score=None, m7_status=None,
                      swing_bias_1d=None, trend_dir=None,
                      config=None,
                      long_target_score=None, short_target_score=None,
                      long_target_details=None, short_target_details=None,
                      nearest_liq_direction=None,
                      m20_score=None, m20_direction=None,
                      rsi_value=None):
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

    # ── Phase 2a: Regime Direction Hint (CHOP_MILD_BEAR/BULL, NEUTRAL_TRENDING_BEAR/BULL) ──
    # When M13 is NEUTRAL and regime has a directional lean, use it
    if direction == 'NEUTRAL' and regime in ('CHOP_MILD_BEAR', 'CHOP_MILD_BULL',
                                               'NEUTRAL_TRENDING_BEAR', 'NEUTRAL_TRENDING_BULL'):
        if regime in ('CHOP_MILD_BEAR', 'NEUTRAL_TRENDING_BEAR'):
            direction = 'SHORT'
            details['regime_direction_hint'] = f'{regime} → SHORT'
        elif regime in ('CHOP_MILD_BULL', 'NEUTRAL_TRENDING_BULL'):
            direction = 'LONG'
            details['regime_direction_hint'] = f'{regime} → LONG'

    # ── Phase 2a-2: Target Tiebreaker ──
    # When direction is still NEUTRAL, use target clarity to pick a side.
    # When direction IS set, targets can boost or penalize confidence.
    if long_target_score is not None and short_target_score is not None:
        details['long_target_score'] = round(long_target_score, 3)
        details['short_target_score'] = round(short_target_score, 3)

        if direction == 'NEUTRAL':
            # No structural bias — let targets decide
            preferred, delta, reason = resolve_target_tiebreaker(
                long_target_score, short_target_score,
                long_target_details or {}, short_target_details or {})
            if preferred is not None and delta >= 0.10:
                direction = preferred
                details['target_tiebreaker'] = reason
                details['target_delta'] = round(delta, 3)
                # Reduce size since we're relying on targets alone
                size_mult *= 0.80
                details['target_tiebreaker_penalty'] = 0.80
        else:
            # Direction already set — targets as confidence modifier
            target_agrees = (
                (direction == 'LONG' and long_target_score > short_target_score) or
                (direction == 'SHORT' and short_target_score > long_target_score)
            )
            target_disagrees = (
                (direction == 'LONG' and short_target_score > long_target_score + 0.15) or
                (direction == 'SHORT' and long_target_score > short_target_score + 0.15)
            )
            if target_agrees:
                bonus = 1.0 + min(abs(long_target_score - short_target_score) * 0.5, 0.15)
                size_mult = min(size_mult * bonus, 1.0)
                details['target_agree_bonus'] = round(bonus, 3)
            elif target_disagrees:
                penalty = 1.0 - min(abs(long_target_score - short_target_score) * 0.3, 0.15)
                size_mult *= penalty
                details['target_disagree_penalty'] = round(penalty, 3)

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

    # ── Phase 2e: Nearest Liquidity Tiebreaker ──
    # When conflict is detected and nearest unswept liquidity has a clear direction,
    # use it to resolve ambiguity. The analysis shows squeeze direction matches
    # nearest liquidity 100% of the time.
    if nearest_liq_direction is not None and nearest_liq_direction in ('LONG', 'SHORT'):
        has_conflict = details.get('daily_conflict_penalty') or details.get('m7_conflict_penalty')
        if has_conflict and direction != nearest_liq_direction:
            # Liquidity disagrees with current direction — flip and penalize
            details['liq_tiebreaker'] = f'Nearest liquidity {nearest_liq_direction} overrides {direction}'
            direction = nearest_liq_direction
            size_mult *= 0.85
            details['liq_tiebreaker_penalty'] = 0.85
        elif direction == nearest_liq_direction:
            # Liquidity confirms — slight boost
            size_mult = min(size_mult * 1.05, 1.0)
            details['liq_tiebreaker_bonus'] = 1.05

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

    elif regime in ('CHOP_MILD', 'CHOP_MILD_BEAR', 'CHOP_MILD_BULL'):
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

    # ── Phase 2f: Failed Breakout Override (M20) ──
    # When M20 detects a strong failed breakout, it can override the direction.
    # A failed breakout is a high-conviction contrarian signal — the market
    # tried to break out and failed, trapping participants on the wrong side.
    if m20_score is not None and m20_direction is not None and m20_direction in ('LONG', 'SHORT'):
        m20_threshold = cfg.get('M20_DIRECTION_OVERRIDE_THRESHOLD', 0.80)
        if m20_score >= m20_threshold:
            if direction != m20_direction and direction != 'NEUTRAL':
                # Strong failed breakout disagrees with structure — flip direction
                details['m20_override'] = f'Failed breakout {m20_score:.3f} flips {direction} → {m20_direction}'
                direction = m20_direction
                size_mult *= 0.85  # reduced size since we're going against structure
                details['m20_override_penalty'] = 0.85
            elif direction == 'NEUTRAL':
                # No structural bias — M20 provides direction
                direction = m20_direction
                details['m20_direction'] = f'Failed breakout provides direction → {m20_direction}'
                size_mult *= 0.70  # lower size when M20 is sole directional input
                details['m20_solo_penalty'] = 0.70
            elif direction == m20_direction:
                # M20 confirms existing direction — boost
                size_mult = min(size_mult * 1.15, 1.0)
                details['m20_agree_bonus'] = 1.15

            details['m20_score'] = round(m20_score, 3)
            details['m20_direction'] = m20_direction

    # ── Phase 2g: RSI Extreme Override ──
    # When RSI is at extreme levels, it's a strong contrarian signal.
    # RSI < 25 = extreme oversold → bias LONG (mean reversion)
    # RSI > 75 = extreme overbought → bias SHORT (mean reversion)
    # This is a standalone signal — doesn't need structure alignment.
    if rsi_value is not None and not np.isnan(rsi_value):
        details['rsi'] = round(rsi_value, 1)
        if rsi_value <= 25:
            # Extreme oversold — strong LONG bias
            if direction == 'SHORT':
                # RSI conflicts with SHORT — flip to NEUTRAL at minimum
                details['rsi_extreme'] = f'RSI {rsi_value:.0f} extreme oversold — conflicts with SHORT'
                if rsi_value <= 20:
                    # RSI < 20 is rare and powerful — flip direction
                    direction = 'LONG'
                    size_mult *= 0.75
                    details['rsi_override'] = f'RSI {rsi_value:.0f} < 20 → flip to LONG (reduced size)'
                else:
                    direction = 'NEUTRAL'
                    details['rsi_dampen'] = f'RSI {rsi_value:.0f} < 25 → neutralize SHORT'
            elif direction == 'NEUTRAL':
                # No direction — RSI provides one
                direction = 'LONG'
                size_mult *= 0.70
                details['rsi_direction'] = f'RSI {rsi_value:.0f} extreme oversold → LONG (reduced size)'
            elif direction == 'LONG':
                # RSI confirms LONG — slight boost
                size_mult = min(size_mult * 1.10, 1.0)
                details['rsi_confirm_bonus'] = 1.10
        elif rsi_value >= 75:
            # Extreme overbought — strong SHORT bias
            if direction == 'LONG':
                details['rsi_extreme'] = f'RSI {rsi_value:.0f} extreme overbought — conflicts with LONG'
                if rsi_value >= 80:
                    direction = 'SHORT'
                    size_mult *= 0.75
                    details['rsi_override'] = f'RSI {rsi_value:.0f} > 80 → flip to SHORT (reduced size)'
                else:
                    direction = 'NEUTRAL'
                    details['rsi_dampen'] = f'RSI {rsi_value:.0f} > 75 → neutralize LONG'
            elif direction == 'NEUTRAL':
                direction = 'SHORT'
                size_mult *= 0.70
                details['rsi_direction'] = f'RSI {rsi_value:.0f} extreme overbought → SHORT (reduced size)'
            elif direction == 'SHORT':
                size_mult = min(size_mult * 1.10, 1.0)
                details['rsi_confirm_bonus'] = 1.10

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

    if 'target_tiebreaker' in details:
        lines.append(f"    🎯 Target tiebreaker: {details['target_tiebreaker']}")
    if 'liq_tiebreaker' in details:
        lines.append(f"    💧 {details['liq_tiebreaker']}")

    lines.append(f"  → Direction: {direction} | Size: {size_mult:.2f} | {action}")

    # Penalties/bonuses
    for key in sorted(details.keys()):
        if key.endswith('_penalty') or key.endswith('_bonus'):
            lines.append(f"    {key}: {details[key]}")

    if 'reason' in details:
        lines.append(f"  Reason: {details['reason']}")

    return '\n'.join(lines)

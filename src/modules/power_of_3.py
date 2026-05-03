"""
Power of 3 Phase Detector — ICT/SMC market phase analysis (v2).

Determines whether the current market structure is:
  - ACCUMULATION: Smart money loading, range-bound, preparing for markup
  - MARKUP:       Real bullish move, structure is genuine
  - MANIPULATION: Judas swing — fake move to grab liquidity before reversal
  - DISTRIBUTION: Smart money selling to late entrants, preparing for markdown
  - MARKDOWN:     Real bearish move, structure is genuine

v2 changes:
  - Sweep completion detection: checks if the key level has already been swept
  - Minimum distance filter: key levels must be >= 0.5% from price (configurable)
  - Forward-looking reversal targets: after a sweep, targets the opposite side
  - Timing integration: uses M4b intrabar CVD divergence for sweep-in-progress detection
  - Actionable output: replaces vague "wait" with concrete entry/invalidation levels
"""

import numpy as np
from dataclasses import dataclass, field, asdict
from typing import List, Optional, Dict


# Minimum distance from price to be an actionable key level (as fraction)
MIN_KEY_LEVEL_DIST_PCT = 0.005  # 0.5%
# How many bars back to check for a completed sweep
SWEEP_LOOKBACK_BARS = 96  # 24h on 15m
# Maximum age (in bars) for a sweep to still be considered actionable
SWEEP_MAX_ACTIONABLE_AGE = 24  # 6h on 15m — older sweeps are stale
# Minimum distance reversal target must have from current price to be actionable
SWEEP_TARGET_MIN_DIST_PCT = 0.003  # 0.3%


@dataclass
class PhaseResult:
    """Output of Power of 3 phase detection."""
    phase: str              # ACCUMULATION | MARKUP | MANIPULATION | DISTRIBUTION | MARKDOWN
    confidence: float       # 0-1
    direction: str          # LONG | SHORT | NEUTRAL (what the phase implies)
    narrative: str          # Human-readable explanation
    signals_for: List[str]  # Signals supporting this phase
    signals_against: List[str]  # Signals contradicting
    key_level: Optional[float] = None  # The level that confirms/invalidate
    key_level_name: str = ''  # What the level is
    trade_bias: str = ''    # ENTER_LONG | ENTER_SHORT | WAIT | AVOID
    # v2 additions
    sweep_status: str = ''  # PENDING | IN_PROGRESS | COMPLETED | NONE
    sweep_level: Optional[float] = None  # The level being/already swept
    reversal_target: Optional[float] = None  # Where price goes after sweep
    reversal_target_name: str = ''  # What the reversal target is
    timing_signal: str = ''  # M4b-based timing: SWEEP_IMMINENT | SWEEP_FADING | NONE
    entry_zone_low: Optional[float] = None  # Actionable entry zone
    entry_zone_high: Optional[float] = None
    invalidation: Optional[float] = None  # Level that kills the thesis


def _check_sweep_completed(price, key_level, df_15m_highs, df_15m_lows,
                           current_idx, lookback=SWEEP_LOOKBACK_BARS):
    """Check if key_level has been swept by recent price action.

    Returns:
        dict with keys:
            swept: bool
            swept_at: int or None (index of sweep bar)
            sweep_depth_pct: float
            bars_ago: int (how many bars since sweep, 0 = current bar)
            is_stale: bool (True if sweep is too old to be actionable)
            sweep_price: float or None (the extreme price that triggered the sweep)
    """
    none_result = {'swept': False, 'swept_at': None, 'sweep_depth_pct': 0.0,
                   'bars_ago': -1, 'is_stale': False, 'sweep_price': None}
    if key_level is None or df_15m_highs is None:
        return none_result

    start = max(0, current_idx - lookback + 1)
    highs = df_15m_highs[start:current_idx + 1].astype(float)
    lows = df_15m_lows[start:current_idx + 1].astype(float)

    swept = False
    swept_at = None
    sweep_depth_pct = 0.0
    sweep_price = None

    if key_level > price:
        # Level is above — swept if high reached it
        for i in range(len(highs) - 1, -1, -1):  # scan backwards for most recent
            if highs[i] >= key_level:
                swept = True
                swept_at = i + start
                sweep_depth_pct = (highs[i] - key_level) / key_level
                sweep_price = float(highs[i])
                break
    else:
        # Level is below — swept if low reached it
        for i in range(len(lows) - 1, -1, -1):
            if lows[i] <= key_level:
                swept = True
                swept_at = i + start
                sweep_depth_pct = (key_level - lows[i]) / key_level
                sweep_price = float(lows[i])
                break

    bars_ago = (current_idx - swept_at) if swept and swept_at is not None else -1
    is_stale = bars_ago > SWEEP_MAX_ACTIONABLE_AGE

    return {
        'swept': swept,
        'swept_at': swept_at,
        'sweep_depth_pct': sweep_depth_pct,
        'bars_ago': bars_ago,
        'is_stale': is_stale,
        'sweep_price': sweep_price,
    }


def _check_target_already_hit(reversal_target, df_15m_highs, df_15m_lows,
                               current_idx, direction_after_sweep, lookback=SWEEP_LOOKBACK_BARS):
    """Check if the reversal target has already been reached since the sweep.

    Returns:
        dict with keys:
            hit: bool
            hit_at: int or None (index of bar that hit target)
            bars_ago: int (how many bars since target was hit)
            overshoot_pct: float (how far past target price went)
    """
    none_result = {'hit': False, 'hit_at': None, 'bars_ago': -1, 'overshoot_pct': 0.0}
    if reversal_target is None or df_15m_highs is None:
        return none_result

    start = max(0, current_idx - lookback + 1)
    highs = df_15m_highs[start:current_idx + 1].astype(float)
    lows = df_15m_lows[start:current_idx + 1].astype(float)

    hit = False
    hit_at = None
    overshoot_pct = 0.0

    if direction_after_sweep == 'SHORT':
        # Target is below — hit if low reached it
        for i in range(len(lows) - 1, -1, -1):
            if lows[i] <= reversal_target:
                hit = True
                hit_at = i + start
                overshoot_pct = (reversal_target - lows[i]) / reversal_target
                break
    else:
        # Target is above — hit if high reached it
        for i in range(len(highs) - 1, -1, -1):
            if highs[i] >= reversal_target:
                hit = True
                hit_at = i + start
                overshoot_pct = (highs[i] - reversal_target) / reversal_target
                break

    bars_ago = (current_idx - hit_at) if hit and hit_at is not None else -1

    return {
        'hit': hit,
        'hit_at': hit_at,
        'bars_ago': bars_ago,
        'overshoot_pct': overshoot_pct,
    }


def _find_reversal_target(price, swept_level, magnets, sr_levels, liq, direction_after_sweep):
    """After a sweep, find the reversal target on the opposite side.

    Picks the nearest level that's at least MIN_KEY_LEVEL_DIST_PCT away from price
    to ensure actionable R:R. Falls back to nearest if nothing qualifies.
    """
    candidates = []

    # From S/R levels
    if sr_levels:
        if direction_after_sweep == 'SHORT':
            supports = [(p, s) for p, s, t, _, _ in sr_levels if t == 'SUPPORT' and p < price]
            for p, s in supports:
                candidates.append(('SUPPORT', p, s))
        else:
            resistances = [(p, s) for p, s, t, _, _ in sr_levels if t == 'RESISTANCE' and p > price]
            for p, s in resistances:
                candidates.append(('RESISTANCE', p, s))

    # From magnets (volume clusters)
    if magnets:
        for entry in magnets:
            p = entry[0]
            s = entry[1] if len(entry) > 1 else 1
            if direction_after_sweep == 'SHORT' and p < price:
                candidates.append(('MAGNET', p, s))
            elif direction_after_sweep == 'LONG' and p > price:
                candidates.append(('MAGNET', p, s))

    # From liquidity levels (opposite side stops)
    if liq:
        if direction_after_sweep == 'SHORT':
            below = [z for z in liq.get('below', []) if not z.get('swept')]
            for z in below:
                candidates.append(('LIQUIDITY', z['price'], z.get('strength', 1)))
        else:
            above = [z for z in liq.get('above', []) if not z.get('swept')]
            for z in above:
                candidates.append(('LIQUIDITY', z['price'], z.get('strength', 1)))

    if not candidates:
        return None, '', 0.0

    # Filter to correct direction
    if direction_after_sweep == 'SHORT':
        valid = [(t, p, s) for t, p, s in candidates if p < price]
    else:
        valid = [(t, p, s) for t, p, s in candidates if p > price]

    if not valid:
        return None, '', 0.0

    # Sort by distance from price
    valid.sort(key=lambda x: abs(x[1] - price))

    # Pick the nearest level that's far enough for actionable R:R
    min_dist = price * MIN_KEY_LEVEL_DIST_PCT
    for t, p, s in valid:
        if abs(p - price) >= min_dist:
            return p, t, s

    # Fallback: use nearest even if close
    best = valid[0]
    return best[1], best[0], best[2]


def _get_timing_signal(m4b_divergence, m4b_bars_ago, m4b_slope, direction):
    """Use M4b intrabar CVD to detect if a sweep is imminent or fading.

    Returns: SWEEP_IMMINENT | SWEEP_FADING | REVERSAL_CONFIRMING | NONE
    """
    if m4b_divergence == 'NONE':
        return 'NONE'

    # Bearish divergence during a LONG setup = sweep up may be failing
    if direction == 'LONG' and m4b_divergence == 'BEARISH':
        if m4b_bars_ago <= 6:
            return 'SWEEP_FADING'  # div just detected — momentum dying
        elif m4b_bars_ago <= 16:
            return 'REVERSAL_CONFIRMING'  # div maturing — reversal likely

    # Bullish divergence during a SHORT setup = sweep down may be failing
    if direction == 'SHORT' and m4b_divergence == 'BULLISH':
        if m4b_bars_ago <= 6:
            return 'SWEEP_FADING'
        elif m4b_bars_ago <= 16:
            return 'REVERSAL_CONFIRMING'

    # Aligned divergence = sweep may still be in progress
    if direction == 'LONG' and m4b_divergence == 'BULLISH':
        if m4b_slope > 0:
            return 'SWEEP_IMMINENT'  # momentum still going

    if direction == 'SHORT' and m4b_divergence == 'BEARISH':
        if m4b_slope < 0:
            return 'SWEEP_IMMINENT'

    return 'NONE'


def detect_phase(result: dict, config: dict = None, df_15m=None) -> PhaseResult:
    """Detect the current Power of 3 phase from scan results.

    Args:
        result: scan_signal() output dict
        config: Optional config overrides
        df_15m: Optional DataFrame with High/Low columns for sweep detection

    Returns:
        PhaseResult with phase, direction, and trade bias
    """
    cfg = config or {}
    price = result.get('price', 0)
    direction = result.get('direction', 'NEUTRAL')
    swing_bias = result.get('swing_bias', 'NEUTRAL')
    phase0 = result.get('phase0')
    min_dist_pct = cfg.get('P3_MIN_KEY_LEVEL_DIST_PCT', MIN_KEY_LEVEL_DIST_PCT)

    # Structure
    m13 = result.get('m13', {})
    m13_bias = m13.get('bias', 'NEUTRAL')
    m13_score = m13.get('score', 0.5)

    # Smart money signals
    deriv = result.get('derivatives', {})
    whale = deriv.get('whale_signal', 'NEUTRAL')
    positioning = deriv.get('positioning', 'NEUTRAL')
    ls_zscore = deriv.get('ls_zscore', 0)
    futures_flow = deriv.get('futures_flow', 'NEUTRAL')
    funding_rate = deriv.get('funding_rate')
    oi_roc = deriv.get('oi_roc_1h', 0)

    # Liquidity
    liq = result.get('liquidity_levels', {})
    magnets = result.get('magnets', [])
    sr_levels = result.get('sr_levels', [])
    unswept_above = [z for z in liq.get('above', []) if not z.get('swept')]
    unswept_below = [z for z in liq.get('below', []) if not z.get('swept')]

    # Conflict history
    conflict_hist = result.get('conflict', {})
    hist = conflict_hist.get('historical', {})
    rev_24h = hist.get('windows', {}).get('24h', {}).get('reversal_rate', 50)

    # Spot
    spot_sigs = result.get('exchange_activity', {}).get('spot_signals', {})
    spot_flow = spot_sigs.get('spot_flow', '?')

    # M4b timing data
    m4b = result.get('m4b', {})
    m4b_divergence = m4b.get('divergence', 'NONE')
    m4b_bars_ago = m4b.get('bars_ago', -1)
    m4b_slope = m4b.get('cvd_slope', 0)

    # ── Scoring ──
    accum_score = 0.0
    accum_signals = []
    markup_score = 0.0
    markup_signals = []
    manip_score = 0.0
    manip_signals = []
    distrib_score = 0.0
    distrib_signals = []
    markdown_score = 0.0
    markdown_signals = []

    # ── 1. Structure vs Smart Money alignment ──
    structure_bullish = m13_bias in ('BULLISH', 'LEAN_BULL')
    structure_bearish = m13_bias in ('BEARISH', 'LEAN_BEAR')
    smart_money_bearish = whale in ('WHALE_BEARISH',) or futures_flow == 'SELLERS_DOMINANT'
    smart_money_bullish = whale in ('WHALE_BULLISH',) or futures_flow == 'BUYERS_DOMINANT'

    if structure_bullish and smart_money_bearish:
        manip_score += 0.30
        manip_signals.append(f'Structure BULLISH but whales {whale} — order flow divergence')
    if structure_bearish and smart_money_bullish:
        manip_score += 0.30
        manip_signals.append(f'Structure BEARISH but whales {whale} — order flow divergence')

    if structure_bullish and smart_money_bullish:
        markup_score += 0.25
        markup_signals.append('Structure and smart money both bullish — aligned')
    if structure_bearish and smart_money_bearish:
        markdown_score += 0.25
        markdown_signals.append('Structure and smart money both bearish — aligned')

    # ── 2. Phase0 context ──
    if phase0 is not None:
        if phase0 < 0.15:
            manip_score += 0.15
            manip_signals.append(f'Phase0={phase0:.3f} death zone — weak macro context')
            distrib_score += 0.10
            distrib_signals.append(f'Phase0={phase0:.3f} — unsustainable context')
        elif phase0 < 0.30:
            accum_score += 0.10
            accum_signals.append(f'Phase0={phase0:.3f} — weak but not death zone')
        elif phase0 >= 0.60:
            if structure_bullish:
                markup_score += 0.15
                markup_signals.append(f'Phase0={phase0:.3f} — strong macro supports markup')
            elif structure_bearish:
                markdown_score += 0.15
                markdown_signals.append(f'Phase0={phase0:.3f} — strong macro supports markdown')

    # ── 3. Unswept liquidity direction ──
    if unswept_above and not unswept_below:
        if smart_money_bearish:
            manip_score += 0.15
            manip_signals.append(f'Unswept liquidity above (${unswept_above[0]["price"]:.0f}) — Judas swing target')
        else:
            markup_score += 0.10
            markup_signals.append('Unswept liquidity above — potential targets for continuation')
    elif unswept_below and not unswept_above:
        if smart_money_bullish:
            manip_score += 0.15
            manip_signals.append(f'Unswept liquidity below (${unswept_below[0]["price"]:.0f}) — Judas swing target')
        else:
            markdown_score += 0.10
            markdown_signals.append('Unswept liquidity below — potential targets for continuation')
    elif unswept_above and unswept_below:
        accum_score += 0.05
        distrib_score += 0.05
        accum_signals.append('Liquidity on both sides — range-bound')
        distrib_signals.append('Liquidity on both sides — range-bound')

    # ── 4. Crowded positioning ──
    if positioning == 'CROWDED_LONG':
        distrib_score += 0.15
        distrib_signals.append('Crowded long — late buyers, distribution target')
        if structure_bullish:
            manip_score += 0.10
            manip_signals.append('Crowded long + bullish structure — Judas setup')
    elif positioning == 'CROWDED_SHORT':
        accum_score += 0.15
        accum_signals.append('Crowded short — late sellers, accumulation target')
        if structure_bearish:
            manip_score += 0.10
            manip_signals.append('Crowded short + bearish structure — Judas setup')

    # ── 5. Funding rate extremes ──
    if funding_rate is not None:
        if funding_rate > 0.001 and structure_bullish:
            distrib_score += 0.10
            distrib_signals.append(f'High funding ({funding_rate*100:.4f}%) — longs paying, late entrants')
        elif funding_rate < -0.001 and structure_bearish:
            accum_score += 0.10
            accum_signals.append(f'Negative funding ({funding_rate*100:.4f}%) — shorts paying, late sellers')

    # ── 6. Historical reversal rate ──
    if rev_24h > 58:
        if structure_bullish:
            manip_score += 0.10
            manip_signals.append(f'{rev_24h:.0f}% reversal rate — similar bullish setups fail often')
            distrib_score += 0.10
            distrib_signals.append(f'{rev_24h:.0f}% reversal rate — distribution pattern')
        elif structure_bearish:
            manip_score += 0.10
            manip_signals.append(f'{rev_24h:.0f}% reversal rate — similar bearish setups fail often')
            accum_score += 0.10
            accum_signals.append(f'{rev_24h:.0f}% reversal rate — accumulation pattern')

    # ── 7. Spot flow vs leverage ──
    if spot_flow == 'BUYERS' and positioning == 'CROWDED_LONG':
        distrib_score += 0.10
        distrib_signals.append('Spot buyers + crowded long — smart money selling spot to leveraged longs')
    elif spot_flow == 'SELLERS' and positioning == 'CROWDED_SHORT':
        accum_score += 0.10
        accum_signals.append('Spot sellers + crowded short — smart money buying spot from leveraged shorts')

    # ── 8. OI dynamics ──
    if oi_roc > 0.5 and structure_bullish and smart_money_bearish:
        manip_score += 0.10
        manip_signals.append(f'OI rising {oi_roc:+.2f}%/hr into bearish whale flow — new longs being trapped')
    if oi_roc > 0.5 and structure_bearish and smart_money_bullish:
        manip_score += 0.10
        manip_signals.append(f'OI rising {oi_roc:+.2f}%/hr into bullish whale flow — new shorts being trapped')

    # ── 9. M4b timing signal bonus ──
    timing = _get_timing_signal(m4b_divergence, m4b_bars_ago, m4b_slope, direction)
    if timing == 'SWEEP_FADING':
        manip_score += 0.10
        manip_signals.append(f'M4b {m4b_divergence} divergence {m4b_bars_ago} bars ago — sweep momentum dying')
    elif timing == 'REVERSAL_CONFIRMING':
        manip_score += 0.15
        manip_signals.append(f'M4b {m4b_divergence} divergence maturing ({m4b_bars_ago} bars) — reversal confirming')
    elif timing == 'SWEEP_IMMINENT':
        if direction == 'LONG':
            markup_score += 0.10
            markup_signals.append(f'M4b bullish slope={m4b_slope:.1f} — sweep upward in progress')
        else:
            markdown_score += 0.10
            markdown_signals.append(f'M4b bearish slope={m4b_slope:.1f} — sweep downward in progress')

    # ── Determine phase ──
    scores = {
        'ACCUMULATION': accum_score,
        'MARKUP': markup_score,
        'MANIPULATION': manip_score,
        'DISTRIBUTION': distrib_score,
        'MARKDOWN': markdown_score,
    }

    phase = max(scores, key=scores.get)
    raw_confidence = scores[phase]
    confidence = min(raw_confidence, 1.0)

    if confidence < 0.20:
        phase = 'AMBIGUOUS'

    # ── Determine direction and trade bias ──
    if phase == 'ACCUMULATION':
        impl_direction = 'LONG'
        trade_bias = 'WAIT'
        key_level = _find_key_level(unswept_above, magnets, price, 'above', min_dist_pct)
        key_level_name = 'markup_breakout'
    elif phase == 'MARKUP':
        impl_direction = 'LONG'
        trade_bias = 'ENTER_LONG'
        key_level = _find_key_level(unswept_above, magnets, price, 'above', min_dist_pct)
        key_level_name = 'continuation_target'
    elif phase == 'MANIPULATION':
        if structure_bullish:
            impl_direction = 'SHORT'
            key_level = _find_key_level(unswept_above, magnets, price, 'above', min_dist_pct)
            key_level_name = 'judas_sweep_target'
        else:
            impl_direction = 'LONG'
            key_level = _find_key_level(unswept_below, magnets, price, 'below', min_dist_pct)
            key_level_name = 'judas_sweep_target'
        trade_bias = 'WAIT'
    elif phase == 'DISTRIBUTION':
        impl_direction = 'SHORT'
        trade_bias = 'WAIT'
        key_level = _find_key_level(unswept_below, magnets, price, 'below', min_dist_pct)
        key_level_name = 'markdown_breakout'
    elif phase == 'MARKDOWN':
        impl_direction = 'SHORT'
        trade_bias = 'ENTER_SHORT'
        key_level = _find_key_level(unswept_below, magnets, price, 'below', min_dist_pct)
        key_level_name = 'continuation_target'
    else:
        impl_direction = 'NEUTRAL'
        trade_bias = 'AVOID'
        key_level = None
        key_level_name = ''

    # ── Sweep completion detection ──
    sweep_status = 'NONE'
    sweep_level = None
    reversal_target = None
    reversal_target_name = ''
    entry_zone_low = None
    entry_zone_high = None
    invalidation = None

    # Check if the key level (or nearest unswept) has already been swept
    highs_arr = None
    lows_arr = None
    if df_15m is not None:
        highs_arr = df_15m['High'].values if 'High' in df_15m.columns else None
        lows_arr = df_15m['Low'].values if 'Low' in df_15m.columns else None

    # Determine which level to check for sweep
    if key_level is not None:
        sweep_result = _check_sweep_completed(
            price, key_level, highs_arr, lows_arr,
            len(df_15m) - 1 if df_15m is not None else 0,
            lookback=cfg.get('P3_SWEEP_LOOKBACK', SWEEP_LOOKBACK_BARS))

        swept = sweep_result['swept']
        bars_ago = sweep_result['bars_ago']
        is_stale = sweep_result['is_stale']

        if swept and is_stale:
            # Sweep happened but it's too old — the move already played out.
            # Check if reversal target was also hit (fully completed thesis).
            reversal_target, reversal_target_name, _ = _find_reversal_target(
                price, key_level, magnets, sr_levels, liq, impl_direction)

            target_result = _check_target_already_hit(
                reversal_target, highs_arr, lows_arr,
                len(df_15m) - 1 if df_15m is not None else 0,
                impl_direction,
                lookback=cfg.get('P3_SWEEP_LOOKBACK', SWEEP_LOOKBACK_BARS))

            if target_result['hit']:
                # Both sweep AND target hit — thesis fully completed, nothing to trade
                narrative = (f"Sweep at ${key_level:.0f} occurred {bars_ago} bars ago "
                             f"and reversal target ${reversal_target:.0f} was already reached "
                             f"{target_result['bars_ago']} bars ago. Move is done — no actionable setup.")
                return PhaseResult(
                    phase=phase, confidence=round(confidence, 3),
                    direction='NEUTRAL', narrative=narrative,
                    signals_for=[], signals_against=['Sweep + target both completed'],
                    key_level=key_level, key_level_name=key_level_name,
                    trade_bias='AVOID',
                    sweep_status='EXPIRED', sweep_level=key_level,
                    reversal_target=reversal_target,
                    reversal_target_name=reversal_target_name,
                    timing_signal='NONE',
                    entry_zone_low=None, entry_zone_high=None,
                    invalidation=None,
                )
            else:
                # Sweep happened, target NOT hit yet, but sweep is old.
                # The setup is degraded — still watchable but not "enter now".
                dist_to_target = abs(reversal_target - price) / price if reversal_target else 0
                if reversal_target and dist_to_target >= SWEEP_TARGET_MIN_DIST_PCT:
                    narrative = (f"Sweep at ${key_level:.0f} occurred {bars_ago} bars ago (stale). "
                                 f"Reversal target ${reversal_target:.0f} not yet reached "
                                 f"({dist_to_target*100:.1f}% away). Setup degraded — reduced confidence.")
                    trade_bias = 'WATCH'
                else:
                    narrative = (f"Sweep at ${key_level:.0f} occurred {bars_ago} bars ago (stale). "
                                 f"No actionable reversal target remains. Move is likely done.")
                    trade_bias = 'AVOID'

                return PhaseResult(
                    phase=phase, confidence=round(confidence * 0.5, 3),  # halve confidence for stale sweeps
                    direction=impl_direction if trade_bias == 'WATCH' else 'NEUTRAL',
                    narrative=narrative,
                    signals_for=[f'Sweep {bars_ago} bars ago — stale'],
                    signals_against=[f'Sweep older than {SWEEP_MAX_ACTIONABLE_AGE} bars — setup degraded'],
                    key_level=key_level, key_level_name=key_level_name,
                    trade_bias=trade_bias,
                    sweep_status='STALE', sweep_level=key_level,
                    reversal_target=reversal_target,
                    reversal_target_name=reversal_target_name,
                    timing_signal='NONE',
                    entry_zone_low=None, entry_zone_high=None,
                    invalidation=None,
                )

        elif swept and not is_stale:
            # Fresh sweep — actionable
            sweep_status = 'COMPLETED'
            sweep_level = key_level
            reversal_target, reversal_target_name, _ = _find_reversal_target(
                price, key_level, magnets, sr_levels, liq, impl_direction)

            # Check if reversal target was already hit (even for fresh sweeps)
            target_result = _check_target_already_hit(
                reversal_target, highs_arr, lows_arr,
                len(df_15m) - 1 if df_15m is not None else 0,
                impl_direction,
                lookback=cfg.get('P3_SWEEP_LOOKBACK', SWEEP_LOOKBACK_BARS))

            if target_result['hit']:
                # Target already hit even on a fresh sweep — move done
                narrative = (f"Sweep at ${key_level:.0f} completed {bars_ago} bars ago, "
                             f"but reversal target ${reversal_target:.0f} was already reached "
                             f"{target_result['bars_ago']} bars ago. Move is done.")
                return PhaseResult(
                    phase=phase, confidence=round(confidence, 3),
                    direction='NEUTRAL', narrative=narrative,
                    signals_for=[], signals_against=['Reversal target already reached'],
                    key_level=key_level, key_level_name=key_level_name,
                    trade_bias='AVOID',
                    sweep_status='TARGET_HIT', sweep_level=key_level,
                    reversal_target=reversal_target,
                    reversal_target_name=reversal_target_name,
                    timing_signal='NONE',
                    entry_zone_low=None, entry_zone_high=None,
                    invalidation=None,
                )

            # Target not hit — set up the trade
            if reversal_target is not None:
                dist_to_target = abs(reversal_target - price) / price
                if dist_to_target >= min_dist_pct:
                    trade_bias = f'ENTER_{impl_direction}'
                    # Set entry zone around current price
                    atr_pct = 0.005  # fallback
                    atr_1h = result.get('what_if', {}).get('sl_pct')
                    if atr_1h:
                        atr_pct = abs(atr_1h) / 100
                    if impl_direction == 'SHORT':
                        entry_zone_low = price * (1 - atr_pct * 0.5)
                        entry_zone_high = price * (1 + atr_pct * 0.3)
                        invalidation = key_level * 1.002  # above sweep high
                    else:
                        entry_zone_low = price * (1 - atr_pct * 0.3)
                        entry_zone_high = price * (1 + atr_pct * 0.5)
                        invalidation = key_level * 0.998  # below sweep low

                    narrative = _build_narrative_v2(
                        phase, impl_direction, structure_bullish, smart_money_bearish,
                        phase0, rev_24h, unswept_above, unswept_below, price,
                        sweep_status='COMPLETED', sweep_level=key_level,
                        reversal_target=reversal_target, reversal_target_name=reversal_target_name,
                        timing=timing, m4b_divergence=m4b_divergence, m4b_bars_ago=m4b_bars_ago,
                        bars_ago=bars_ago,
                    )

                    return PhaseResult(
                        phase=phase, confidence=round(confidence, 3),
                        direction=impl_direction, narrative=narrative,
                        signals_for=manip_signals if phase == 'MANIPULATION' else markup_signals,
                        signals_against=_get_opposing_signals(phase, scores),
                        key_level=key_level, key_level_name=key_level_name,
                        trade_bias=trade_bias,
                        sweep_status=sweep_status, sweep_level=sweep_level,
                        reversal_target=reversal_target,
                        reversal_target_name=reversal_target_name,
                        timing_signal=timing,
                        entry_zone_low=entry_zone_low, entry_zone_high=entry_zone_high,
                        invalidation=invalidation,
                    )

    # No sweep completed — check if sweep is in progress via timing
    if timing in ('SWEEP_FADING', 'REVERSAL_CONFIRMING'):
        sweep_status = 'IN_PROGRESS'
        sweep_level = key_level
        reversal_target, reversal_target_name, _ = _find_reversal_target(
            price, key_level, magnets, sr_levels, liq, impl_direction)

        if reversal_target is not None:
            dist_to_target = abs(reversal_target - price) / price
            if dist_to_target >= min_dist_pct:
                trade_bias = 'WATCH'
                # Set entry zone near the sweep level
                if impl_direction == 'SHORT' and key_level:
                    entry_zone_low = key_level * 0.998
                    entry_zone_high = key_level * 1.001
                    invalidation = key_level * 1.003
                elif impl_direction == 'LONG' and key_level:
                    entry_zone_low = key_level * 0.999
                    entry_zone_high = key_level * 1.002
                    invalidation = key_level * 0.997

    elif timing == 'SWEEP_IMMINENT':
        sweep_status = 'PENDING'
        sweep_level = key_level

    # ── Build narrative ──
    narrative = _build_narrative_v2(
        phase, impl_direction, structure_bullish, smart_money_bearish,
        phase0, rev_24h, unswept_above, unswept_below, price,
        sweep_status=sweep_status, sweep_level=sweep_level,
        reversal_target=reversal_target, reversal_target_name=reversal_target_name,
        timing=timing, m4b_divergence=m4b_divergence, m4b_bars_ago=m4b_bars_ago,
    )

    # Collect signals
    all_signals_for = {
        'ACCUMULATION': accum_signals,
        'MARKUP': markup_signals,
        'MANIPULATION': manip_signals,
        'DISTRIBUTION': distrib_signals,
        'MARKDOWN': markdown_signals,
    }
    signals_for = all_signals_for.get(phase, [])
    signals_against = _get_opposing_signals(phase, scores)

    return PhaseResult(
        phase=phase, confidence=round(confidence, 3),
        direction=impl_direction, narrative=narrative,
        signals_for=signals_for, signals_against=signals_against[:5],
        key_level=key_level, key_level_name=key_level_name,
        trade_bias=trade_bias,
        sweep_status=sweep_status, sweep_level=sweep_level,
        reversal_target=reversal_target, reversal_target_name=reversal_target_name,
        timing_signal=timing,
        entry_zone_low=entry_zone_low, entry_zone_high=entry_zone_high,
        invalidation=invalidation,
    )


def _get_opposing_signals(phase, scores):
    """Get signals from non-winning phases as opposing evidence."""
    opposing = []
    for other_phase, score in sorted(scores.items(), key=lambda x: -x[1]):
        if other_phase != phase and score > 0:
            opposing.append(f'{other_phase} score={score:.2f}')
    return opposing[:5]


def _find_key_level(unswept, magnets, price, side, min_dist_pct):
    """Find the nearest actionable key level, respecting minimum distance filter.

    Falls back to nearest unswept level if nothing passes the distance filter
    (e.g. in tight ranges where all liquidity is close to price).
    """
    candidates = []
    fallback = []

    for z in unswept:
        p = z['price']
        dist = abs(p - price) / price
        if (side == 'above' and p > price) or (side == 'below' and p < price):
            fallback.append((p, dist))
            if dist >= min_dist_pct:
                candidates.append((p, dist))

    for entry in magnets:
        p = entry[0]
        if side == 'above' and p > price:
            dist = (p - price) / price
            fallback.append((p, dist))
            if dist >= min_dist_pct:
                candidates.append((p, dist))
        elif side == 'below' and p < price:
            dist = (price - p) / price
            fallback.append((p, dist))
            if dist >= min_dist_pct:
                candidates.append((p, dist))

    # Prefer distance-filtered candidates, fall back to nearest unswept
    pool = candidates if candidates else fallback
    if not pool:
        return None

    pool.sort(key=lambda x: x[1])
    return pool[0][0]


def _build_narrative_v2(phase, direction, structure_bullish, smart_money_bearish,
                        phase0, rev_24h, unswept_above, unswept_below, price,
                        sweep_status='NONE', sweep_level=None,
                        reversal_target=None, reversal_target_name='',
                        timing='NONE', m4b_divergence='NONE', m4b_bars_ago=-1,
                        bars_ago=-1):
    """Build actionable narrative with sweep status and targets."""

    parts = []

    # Phase description
    if phase == 'MANIPULATION' and structure_bullish and smart_money_bearish:
        if sweep_level:
            dist = (sweep_level - price) / price * 100
            parts.append(f"Bullish structure + bearish whales = Judas swing setup.")
            parts.append(f"Target: sweep ${sweep_level:.0f} ({dist:+.1f}%) to grab buy-side stops.")
        elif unswept_above:
            zone = unswept_above[0]['price']
            dist = (zone - price) / price * 100
            parts.append(f"Bullish structure + bearish whales = Judas swing setup.")
            parts.append(f"Target: sweep ${zone:.0f} ({dist:+.1f}%) to grab buy-side stops.")
        else:
            parts.append("Bullish structure + bearish whales = Judas swing, but no clear sweep target nearby.")

    elif phase == 'MANIPULATION' and not structure_bullish and smart_money_bearish:
        if sweep_level:
            dist = (price - sweep_level) / price * 100
            parts.append(f"Bearish structure + bullish whales = Judas swing setup.")
            parts.append(f"Target: sweep ${sweep_level:.0f} ({-dist:+.1f}%) to grab sell-side stops.")
        elif unswept_below:
            zone = unswept_below[0]['price']
            dist = (price - zone) / price * 100
            parts.append(f"Bearish structure + bullish whales = Judas swing setup.")
            parts.append(f"Target: sweep ${zone:.0f} ({-dist:+.1f}%) to grab sell-side stops.")
        else:
            parts.append("Bearish structure + bullish whales = Judas swing, but no clear sweep target nearby.")

    elif phase == 'DISTRIBUTION':
        parts.append(f"Smart money distributing. Bullish structure attracts late longs.")

    elif phase == 'ACCUMULATION':
        parts.append(f"Smart money accumulating. Bearish structure shakes out weak hands.")

    elif phase == 'MARKUP':
        parts.append(f"Genuine bullish move. Structure and smart money aligned.")

    elif phase == 'MARKDOWN':
        parts.append(f"Genuine bearish move. Structure and smart money aligned.")

    else:
        parts.append("Phase ambiguous — insufficient signal alignment.")

    # Sweep status
    if sweep_status == 'COMPLETED' and sweep_level and reversal_target:
        sweep_dist = abs(sweep_level - price) / price * 100
        target_dist = abs(reversal_target - price) / price * 100
        age_str = f" ({bars_ago} bars ago)" if bars_ago >= 0 else ""
        parts.append(f"\n✅ Sweep COMPLETED at ${sweep_level:.0f}{age_str} ({sweep_dist:.1f}% from price).")
        parts.append(f"🎯 Reversal target: ${reversal_target:.0f} ({reversal_target_name}, {target_dist:+.1f}% from price).")
        parts.append(f"→ Entry NOW at ${price:.0f}, target ${reversal_target:.0f}.")

    elif sweep_status == 'IN_PROGRESS':
        parts.append(f"\n⏳ Sweep IN PROGRESS at ${sweep_level:.0f}.")
        if reversal_target:
            target_dist = abs(reversal_target - price) / price * 100
            parts.append(f"🎯 After sweep: reversal target ${reversal_target:.0f} ({reversal_target_name}, {target_dist:+.1f}%).")
        parts.append("Wait for sweep completion + rejection confirmation before entering.")

    elif sweep_status == 'PENDING':
        parts.append(f"\n⏳ Sweep PENDING — momentum heading toward ${sweep_level:.0f}.")
        parts.append("Watch for volume spike at the level to confirm sweep.")

    # Timing signal
    if timing == 'SWEEP_FADING':
        parts.append(f"📊 M4b: {m4b_divergence} divergence {m4b_bars_ago} bars ago — sweep momentum fading.")
    elif timing == 'REVERSAL_CONFIRMING':
        parts.append(f"📊 M4b: {m4b_divergence} divergence maturing ({m4b_bars_ago} bars) — reversal likely.")
    elif timing == 'SWEEP_IMMINENT':
        parts.append(f"📊 M4b: momentum still pushing toward sweep level.")

    return ' '.join(parts)


def format_phase(pr: PhaseResult) -> str:
    """Format phase result for terminal output."""
    lines = []
    lines.append('')
    lines.append('  🔮 POWER OF 3 PHASE DETECTION')

    phase_icons = {
        'ACCUMULATION': '📦',
        'MARKUP': '📈',
        'MANIPULATION': '🎭',
        'DISTRIBUTION': '📤',
        'MARKDOWN': '📉',
        'AMBIGUOUS': '❓',
    }
    icon = phase_icons.get(pr.phase, '❓')

    lines.append(f'  Phase: {icon} {pr.phase}  (confidence: {pr.confidence:.0%})')
    lines.append(f'  Direction: {pr.direction}  Trade bias: {pr.trade_bias}')
    lines.append(f'  {pr.narrative}')

    if pr.signals_for:
        lines.append(f'\n  Signals supporting {pr.phase}:')
        for s in pr.signals_for:
            lines.append(f'    ✅ {s}')

    if pr.signals_against:
        lines.append(f'\n  Counter-signals:')
        for s in pr.signals_against[:3]:
            lines.append(f'    ⚠️  {s}')

    # Key level with distance check
    if pr.key_level:
        lines.append(f'\n  Key level: ${pr.key_level:.0f} ({pr.key_level_name})')

    # Sweep status
    if pr.sweep_status and pr.sweep_status != 'NONE':
        sweep_icons = {'COMPLETED': '✅', 'IN_PROGRESS': '⏳', 'PENDING': '🔄',
                       'EXPIRED': '💀', 'STALE': '⚠️', 'TARGET_HIT': '🎯'}
        icon = sweep_icons.get(pr.sweep_status, '❓')
        lines.append(f'  Sweep: {icon} {pr.sweep_status}' +
                     (f' at ${pr.sweep_level:.0f}' if pr.sweep_level else ''))

    # Reversal target
    if pr.reversal_target:
        lines.append(f'  Reversal target: ${pr.reversal_target:.0f} ({pr.reversal_target_name})')

    # Entry zone
    if pr.entry_zone_low and pr.entry_zone_high:
        lines.append(f'  Entry zone: ${pr.entry_zone_low:.0f} – ${pr.entry_zone_high:.0f}')

    # Invalidation
    if pr.invalidation:
        lines.append(f'  Invalidation: ${pr.invalidation:.0f}')

    # Timing
    if pr.timing_signal and pr.timing_signal != 'NONE':
        lines.append(f'  Timing: {pr.timing_signal}')

    # Action
    action_map = {
        'ENTER_LONG': '✅ Enter LONG — sweep completed, reversal target above',
        'ENTER_SHORT': '✅ Enter SHORT — sweep completed, reversal target below',
        'WATCH': '⏳ Watch — stale sweep, target not yet reached. Reduced confidence.',
        'WAIT': f'⏳ Wait — let the sweep play out, then enter {pr.direction}',
        'AVOID': '🚫 Avoid — no actionable setup (sweep/target already completed)',
    }
    action = action_map.get(pr.trade_bias, '❓ Unknown')
    if pr.trade_bias.startswith('ENTER_') and pr.entry_zone_low:
        action += f'\n    Entry: ${pr.entry_zone_low:.0f}–${pr.entry_zone_high:.0f}  TP: ${pr.reversal_target:.0f}  SL: ${pr.invalidation:.0f}'

    lines.append(f'\n  Action: {action}')

    return '\n'.join(lines)


def phase_to_dict(pr: PhaseResult) -> dict:
    """Serialize PhaseResult to dict for JSON output."""
    d = asdict(pr)
    return d

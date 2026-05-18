"""
Power of 3 Phase Detector — ICT/SMC market phase analysis (v3).

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

v3 changes:
  - Zone sweep detection: tolerances for near-miss sweeps (price gets 99% then reverses)
  - Partial sweep assessment: distinguishes completed sweeps from liquidity revisit setups
  - Directional confirmation: uses OI, taker, M4b, whale signals to confirm sweep direction
  - Unswept liquidity gravity: strong unswept clusters above/below act as magnets
"""

import numpy as np
from dataclasses import dataclass, field, asdict
from typing import List, Optional, Dict


# Minimum distance from price to be an actionable key level (as fraction)
MIN_KEY_LEVEL_DIST_PCT = 0.005  # 0.5%
# How many bars back to check for a completed sweep
SWEEP_LOOKBACK_BARS = 96  # 24h on 15m

# ── v3: Zone sweep detection ─────────────────────────────────────────
# Tolerance for near-miss sweeps (price gets within this % of level = swept)
SWEEP_ZONE_TOLERANCE_PCT = 0.0015  # 0.15% — catches $2,323 vs $2,326 miss
# Width of the liquidity zone to check (expand key level by this % each side)
SWEEP_ZONE_WIDTH_PCT = 0.003  # 0.3% — treats nearby stops/magnets as one zone
# Minimum unswept strength above/below to count as "gravity" pulling price
SWEEP_GRAVITY_MIN_STRENGTH = 80  # combined stop strength
# Maximum bars ago for a zone sweep to still be considered fresh
SWEEP_ZONE_FRESH_BARS = 24  # 6h on 15m

# ── Judas Sweep Historical Performance Stats ─────────────────────────
# Source: Backtested on ETH/USDT 15m (2017-08 → 2026-05), 305k+ bars
# Detection: compressed range + near resistance + sweep above + reversal measure
# These stats are embedded for instant lookup when MANIPULATION phase is detected.
JUDAS_SWEEP_STATS = {
    'source': 'ETH/USDT 15m historical (2017-2026), 3222 compressed sweeps',
    'all_sweeps': {
        'count': 3222,
        'reversal_gt_03': 0.820,   # 82.0% reverse >0.3%
        'reversal_gt_05': 0.716,   # 71.6% reverse >0.5%
        'reversal_gt_10': 0.497,   # 49.7% reverse >1.0%
        'avg_drop': 1.82,          # avg reversal drop %
        'median_drop': 1.26,
    },
    'judas_context': {              # bull structure + bearish sellers
        'count': 526,
        'reversal_gt_03': 0.825,
        'reversal_gt_05': 0.684,
        'reversal_gt_10': 0.475,
        'avg_drop': 1.68,
        'median_drop': 1.18,
        'avg_extension': 2.45,     # when it fails, avg continuation %
    },
    'recent_2025_2026': {
        'count': 142,
        'reversal_gt_03': 0.796,
        'reversal_gt_05': 0.641,
        'reversal_gt_10': 0.423,
        'avg_drop': 1.96,
        'median_drop': 1.15,
    },
    'by_sweep_depth': {
        'shallow_01_02': {'count': 314, 'reversal_gt_03': 0.799, 'avg_drop': 1.68},
        'medium_02_03':  {'count': 112, 'reversal_gt_03': 0.848, 'avg_drop': 1.88},
        'deep_03_05':    {'count': 81,  'reversal_gt_03': 0.864, 'avg_drop': 1.43},
        'extreme_05_10': {'count': 19,  'reversal_gt_03': 0.947, 'avg_drop': 1.50},
    },
    'time_to_reversal': {
        'median_bars': 6,           # 90 min
        'mean_bars': 10,            # 150 min
        'p75_bars': 15,             # 225 min
        'max_bars': 48,             # 720 min
    },
}


def get_judas_stats_for_sweep(sweep_pct: float) -> dict:
    """Get historical stats matching a given sweep depth.

    Args:
        sweep_pct: sweep depth as percentage (e.g. 0.3 = 0.3%)

    Returns:
        dict with reversal rate, avg drop, and time-to-reversal for the matching bucket.
    """
    depth_stats = JUDAS_SWEEP_STATS['by_sweep_depth']
    if sweep_pct < 0.2:
        bucket = depth_stats['shallow_01_02']
        label = 'shallow (0.1-0.2%)'
    elif sweep_pct < 0.3:
        bucket = depth_stats['medium_02_03']
        label = 'medium (0.2-0.3%)'
    elif sweep_pct < 0.5:
        bucket = depth_stats['deep_03_05']
        label = 'deep (0.3-0.5%)'
    else:
        bucket = depth_stats['extreme_05_10']
        label = 'extreme (0.5-1.0%)'

    base = JUDAS_SWEEP_STATS['judas_context']
    ttr = JUDAS_SWEEP_STATS['time_to_reversal']

    return {
        'depth_label': label,
        'depth_count': bucket['count'],
        'depth_reversal_rate': bucket['reversal_gt_03'],
        'depth_avg_drop': bucket['avg_drop'],
        'overall_reversal_rate': base['reversal_gt_03'],
        'overall_avg_drop': base['avg_drop'],
        'overall_median_drop': base['median_drop'],
        'avg_extension': base['avg_extension'],
        'median_time_bars': ttr['median_bars'],
        'mean_time_bars': ttr['mean_bars'],
        'p75_time_bars': ttr['p75_bars'],
        'recent_reversal_rate': JUDAS_SWEEP_STATS['recent_2025_2026']['reversal_gt_03'],
        'recent_avg_drop': JUDAS_SWEEP_STATS['recent_2025_2026']['avg_drop'],
    }


# ── v3: Zone Sweep Detection ─────────────────────────────────────────

def _check_zone_swept(price, key_level, df_15m_highs, df_15m_lows,
                      magnets, sr_levels, liq, current_idx, direction,
                      lookback=SWEEP_LOOKBACK_BARS):
    """Check if a liquidity zone around key_level was swept.

    Unlike _check_sweep_completed which checks exact level crossing,
    this checks whether price got within tolerance of ANY level in the zone
    (magnets, stops, S/R) and then reversed.

    Returns:
        dict with:
            zone_swept: bool — whether the zone was swept
            zone_high: float — highest price that swept the zone
            swept_levels: list — which levels in the zone were hit
            bars_ago: int — how many bars since the zone sweep
            is_fresh: bool — whether the sweep is recent enough to trade
            zone_center: float — center of the swept zone
    """
    none_result = {
        'zone_swept': False, 'zone_high': 0, 'swept_levels': [],
        'bars_ago': -1, 'is_fresh': False, 'zone_center': key_level,
    }
    if key_level is None or df_15m_highs is None:
        return none_result

    tolerance = key_level * SWEEP_ZONE_TOLERANCE_PCT
    zone_low = key_level * (1 - SWEEP_ZONE_WIDTH_PCT)
    zone_high_bound = key_level * (1 + SWEEP_ZONE_WIDTH_PCT)

    # Collect all levels within the zone
    zone_levels = []

    # Magnets in zone
    if magnets:
        for entry in magnets:
            p = entry[0]
            if zone_low <= p <= zone_high_bound:
                zone_levels.append(('MAGNET', p, entry[1] if len(entry) > 1 else 1))

    # S/R in zone
    if sr_levels:
        for entry in sr_levels:
            if len(entry) >= 3:
                p, s, t = entry[0], entry[1], entry[2]
                if zone_low <= p <= zone_high_bound:
                    zone_levels.append(('SR', p, s))

    # Liquidity levels in zone
    if liq:
        for side_key in ('above', 'below'):
            for z in liq.get(side_key, []):
                p = z.get('price', 0)
                if zone_low <= p <= zone_high_bound:
                    zone_levels.append(('LIQ', p, z.get('strength', 1)))

    # Always include the key level itself
    zone_levels.append(('KEY', key_level, 0))

    # Deduplicate
    seen_prices = set()
    unique_levels = []
    for lt, p, s in zone_levels:
        rounded = round(p, 2)
        if rounded not in seen_prices:
            seen_prices.add(rounded)
            unique_levels.append((lt, p, s))

    # Check if price swept through the zone
    start = max(0, current_idx - lookback + 1)
    highs = df_15m_highs[start:current_idx + 1].astype(float)
    lows = df_15m_lows[start:current_idx + 1].astype(float)

    # For SHORT direction (sweep above): check if highs entered the zone
    # For LONG direction (sweep below): check if lows entered the zone
    swept_levels = []
    sweep_bar_idx = None
    zone_extreme = None

    if direction == 'SHORT':
        # Zone is above price — check if highs reached it
        zone_entry_price = zone_low - tolerance
        for i in range(len(highs) - 1, -1, -1):
            if highs[i] >= zone_entry_price:
                sweep_bar_idx = i + start
                zone_extreme = float(highs[i])
                # Check which specific levels were hit
                for lt, p, s in unique_levels:
                    if highs[i] >= p - tolerance:
                        swept_levels.append({'type': lt, 'price': p, 'strength': s, 'hit': True})
                    else:
                        swept_levels.append({'type': lt, 'price': p, 'strength': s, 'hit': False})
                break
    else:
        # Zone is below price — check if lows reached it
        zone_entry_price = zone_high_bound + tolerance
        for i in range(len(lows) - 1, -1, -1):
            if lows[i] <= zone_entry_price:
                sweep_bar_idx = i + start
                zone_extreme = float(lows[i])
                for lt, p, s in unique_levels:
                    if lows[i] <= p + tolerance:
                        swept_levels.append({'type': lt, 'price': p, 'strength': s, 'hit': True})
                    else:
                        swept_levels.append({'type': lt, 'price': p, 'strength': s, 'hit': False})
                break

    if sweep_bar_idx is None:
        return none_result

    bars_ago = current_idx - sweep_bar_idx
    is_fresh = bars_ago <= SWEEP_ZONE_FRESH_BARS
    any_hit = any(item['hit'] for item in swept_levels)

    return {
        'zone_swept': any_hit,
        'zone_high': zone_extreme,
        'swept_levels': swept_levels,
        'bars_ago': bars_ago,
        'is_fresh': is_fresh,
        'zone_center': key_level,
    }


def _assess_sweep_completeness(price, swept_zone_high, unswept_above, unswept_below,
                                m4b_divergence, m4b_slope, oi_roc, taker_ratio,
                                direction_after_sweep, whale_signal, futures_flow):
    """After a zone sweep, assess if the sweep is done or price will revisit.

    Uses multiple confirmation signals to determine whether:
      - SWEEP_DONE: reversal is real, trade the opposite direction
      - PARTIAL_SWEEP: more liquidity in the sweep direction, wait for revisit
      - AMBIGUOUS: need more data

    Returns:
        dict with:
            status: 'SWEEP_DONE' | 'PARTIAL_SWEEP' | 'AMBIGUOUS'
            score: int (positive = done, negative = revisit)
            factors: list of contributing signals
            recommendation: str
    """
    score = 0
    factors = []

    # ── 1. Unswept liquidity gravity ──
    # Strong unswept levels in the sweep direction = price likely to revisit
    if direction_after_sweep == 'SHORT':
        # Sweep was upward — check unswept above
        unswept_strength = sum(z.get('strength', 0) for z in unswept_above
                               if z.get('price', 0) > price)
        unswept_count = len([z for z in unswept_above if z.get('price', 0) > price])
    else:
        # Sweep was downward — check unswept below
        unswept_strength = sum(z.get('strength', 0) for z in unswept_below
                               if z.get('price', 0) < price)
        unswept_count = len([z for z in unswept_below if z.get('price', 0) < price])

    if unswept_strength >= 150:
        score -= 3
        factors.append(f'Strong unswept gravity: str={unswept_strength:.0f} ({unswept_count} levels) — price likely to revisit')
    elif unswept_strength >= SWEEP_GRAVITY_MIN_STRENGTH:
        score -= 2
        factors.append(f'Moderate unswept gravity: str={unswept_strength:.0f} — revisit possible')
    elif unswept_strength < 30:
        score += 2
        factors.append(f'Weak unswept gravity: str={unswept_strength:.0f} — no pull remaining')
    else:
        factors.append(f'Unswept gravity: str={unswept_strength:.0f} — neutral')

    # ── 2. M4b momentum exhaustion ──
    if direction_after_sweep == 'SHORT':
        # Sweep was up — bearish div = momentum exhausted
        momentum_exhausted = (m4b_divergence == 'BEARISH' or m4b_slope < 0)
    else:
        momentum_exhausted = (m4b_divergence == 'BULLISH' or m4b_slope > 0)

    if momentum_exhausted:
        score += 1
        factors.append(f'M4b {m4b_divergence} div, slope={m4b_slope:.1f} — momentum exhausted')
    else:
        score -= 1
        factors.append(f'M4b {m4b_divergence} div, slope={m4b_slope:.1f} — momentum still pushing')

    # ── 3. Taker on reversal ──
    if direction_after_sweep == 'SHORT':
        # Sweep was up — sellers on reversal = sweep done
        sellers_on_reversal = taker_ratio < 0.48
        buyers_on_reversal = taker_ratio > 0.55
    else:
        sellers_on_reversal = taker_ratio > 0.52
        buyers_on_reversal = taker_ratio < 0.45

    if sellers_on_reversal:
        score += 1
        factors.append(f'Taker {taker_ratio:.3f} — sellers on reversal (sweep done)')
    elif buyers_on_reversal:
        score -= 1
        factors.append(f'Taker {taker_ratio:.3f} — buyers absorbing dip (revisit likely)')
    else:
        factors.append(f'Taker {taker_ratio:.3f} — neutral')

    # ── 4. OI dynamics ──
    if oi_roc < -0.5:
        score += 2
        factors.append(f'OI {oi_roc:+.2f}%/hr — positions closing (capitulation)')
    elif oi_roc < -0.2:
        score += 1
        factors.append(f'OI {oi_roc:+.2f}%/hr — mild decline')
    elif oi_roc > 0.3:
        score -= 1
        factors.append(f'OI {oi_roc:+.2f}%/hr — new positions opening (revisit setup)')
    else:
        factors.append(f'OI {oi_roc:+.2f}%/hr — stable')

    # ── 5. Whale alignment ──
    if direction_after_sweep == 'SHORT':
        whale_aligned = whale_signal == 'WHALE_BEARISH'
        whale_against = whale_signal == 'WHALE_BULLISH'
    else:
        whale_aligned = whale_signal == 'WHALE_BULLISH'
        whale_against = whale_signal == 'WHALE_BEARISH'

    if whale_aligned:
        score += 1
        factors.append(f'Whales {whale_signal} — aligned with reversal')
    elif whale_against:
        score -= 1
        factors.append(f'Whales {whale_signal} — against reversal (revisit risk)')

    # ── 6. Futures flow ──
    if direction_after_sweep == 'SHORT':
        flow_aligned = futures_flow == 'SELLERS_DOMINANT'
    else:
        flow_aligned = futures_flow == 'BUYERS_DOMINANT'

    if flow_aligned:
        score += 1
        factors.append(f'Futures flow {futures_flow} — aligned with reversal')

    # ── Determine status ──
    if score >= 3:
        status = 'SWEEP_DONE'
        recommendation = 'Trade the reversal — multiple confirmations'
    elif score >= 1:
        status = 'SWEEP_DONE'
        recommendation = 'Trade the reversal — moderate confidence'
    elif score <= -3:
        status = 'PARTIAL_SWEEP'
        recommendation = 'Wait for price to revisit sweep zone — strong gravity remaining'
    elif score <= -1:
        status = 'PARTIAL_SWEEP'
        recommendation = 'Wait for revisit — unswept liquidity still pulling'
    else:
        status = 'AMBIGUOUS'
        recommendation = 'Wait for more confirmation — mixed signals'

    return {
        'status': status,
        'score': score,
        'factors': factors,
        'recommendation': recommendation,
        'unswept_strength': unswept_strength,
        'unswept_count': unswept_count,
    }


# Maximum age (in bars) for a sweep to still be considered actionable
SWEEP_MAX_ACTIONABLE_AGE = 24  # 6h on 15m — older sweeps are stale
# Minimum distance reversal target must have from current price to be actionable
SWEEP_TARGET_MIN_DIST_PCT = 0.003  # 0.3%
# Minimum bounce from reversal target to form a secondary setup (as fraction)
SECONDARY_BOUNCE_MIN_PCT = 0.003  # 0.3%


def _detect_secondary_setup(price, sweep_level, reversal_target, target_result,
                            df_15m_highs, df_15m_lows, current_idx, phase, confidence,
                            key_level_name, impl_direction):
    """After both sweep and target are completed, check if a secondary setup is forming.

    When price sweeps a level, hits the reversal target, and then bounces,
    that bounce is itself a potential new move. This function detects that
    bounce and returns a forward-looking PhaseResult if actionable.

    Args:
        price: Current price
        sweep_level: The level that was swept (e.g. $2311)
        reversal_target: The reversal target that was hit (e.g. $2287)
        target_result: Dict from _check_target_already_hit
        df_15m_highs, df_15m_lows: Price arrays
        current_idx: Current bar index
        phase: The detected phase (MANIPULATION, etc.)
        confidence: Current confidence
        key_level_name: Name of the key level
        impl_direction: Direction implied by the sweep (SHORT if sweep above, LONG if sweep below)

    Returns:
        PhaseResult if a secondary setup is detected, else None
    """
    if reversal_target is None or target_result.get('hit_at') is None:
        return None

    hit_at = target_result['hit_at']
    bars_since_target = current_idx - hit_at
    if bars_since_target < 3:
        return None  # too soon to call a bounce

    # Check price action since the target was hit
    highs = df_15m_highs[hit_at:current_idx + 1].astype(float)
    lows = df_15m_lows[hit_at:current_idx + 1].astype(float)

    if impl_direction == 'SHORT':
        # Original: swept above → dropped to target below
        # Secondary: bounce UP from the low target
        low_since_target = float(np.min(lows))
        bounce_from_low = (price - low_since_target) / low_since_target if low_since_target > 0 else 0

        if bounce_from_low < SECONDARY_BOUNCE_MIN_PCT:
            return None  # no meaningful bounce yet

        # Bounce detected — direction is LONG (up from the low)
        new_direction = 'LONG'
        mid = (sweep_level + reversal_target) / 2

        # If midpoint already reached, target the sweep level (full re-test)
        if price >= mid:
            new_target = sweep_level
            target_label = 'sweep_retest'
        else:
            new_target = mid
            target_label = 'midpoint_retest'

        # If target is too close, try the sweep level instead
        target_dist = abs(new_target - price) / price
        if target_dist < SECONDARY_BOUNCE_MIN_PCT:
            sweep_dist = abs(sweep_level - price) / price
            if sweep_dist >= SECONDARY_BOUNCE_MIN_PCT:
                new_target = sweep_level
                target_label = 'sweep_retest'
                target_dist = sweep_dist
            else:
                return None  # neither target is actionable

        new_entry = price
        new_invalidation = low_since_target * 0.997  # 0.3% below the bounce low

        bounce_pct = bounce_from_low * 100
        signals_for = [
            f'Bounce +{bounce_pct:.1f}% from reversal target ${reversal_target:.0f}',
            f'Target: {target_label} ${new_target:.0f} (sweep ${sweep_level:.0f} / target ${reversal_target:.0f})',
        ]
        signals_against = [
            'Secondary setup — reduced confidence (0.5x)',
            f'Original sweep was {bars_since_target + target_result["bars_ago"]} bars ago',
        ]

        narrative = (
            f"Secondary LONG setup: price swept ${sweep_level:.0f} SHORT, "
            f"hit target ${reversal_target:.0f} {target_result['bars_ago']} bars ago, "
            f"then bounced +{bounce_pct:.1f}% to ${price:.0f}. "
            f"Watching for continuation to {target_label} ${new_target:.0f} (+{(new_target-price)/price*100:.1f}%). "
            f"Invalidation below ${new_invalidation:.0f}."
        )

    else:
        # Original: swept below → rallied to target above
        # Secondary: drop DOWN from the high target
        high_since_target = float(np.max(highs))
        drop_from_high = (high_since_target - price) / high_since_target if high_since_target > 0 else 0

        if drop_from_high < SECONDARY_BOUNCE_MIN_PCT:
            return None

        new_direction = 'SHORT'
        mid = (sweep_level + reversal_target) / 2

        # If midpoint already reached, target the sweep level (full re-test)
        if price <= mid:
            new_target = sweep_level
            target_label = 'sweep_retest'
        else:
            new_target = mid
            target_label = 'midpoint_retest'

        # If target is too close, try the sweep level instead
        target_dist = abs(new_target - price) / price
        if target_dist < SECONDARY_BOUNCE_MIN_PCT:
            sweep_dist = abs(sweep_level - price) / price
            if sweep_dist >= SECONDARY_BOUNCE_MIN_PCT:
                new_target = sweep_level
                target_label = 'sweep_retest'
                target_dist = sweep_dist
            else:
                return None  # neither target is actionable

        new_entry = price
        new_invalidation = high_since_target * 1.003

        drop_pct = drop_from_high * 100
        signals_for = [
            f'Drop -{drop_pct:.1f}% from reversal target ${reversal_target:.0f}',
            f'Target: {target_label} ${new_target:.0f} (sweep ${sweep_level:.0f} / target ${reversal_target:.0f})',
        ]
        signals_against = [
            'Secondary setup — reduced confidence (0.5x)',
            f'Original sweep was {bars_since_target + target_result["bars_ago"]} bars ago',
        ]

        narrative = (
            f"Secondary SHORT setup: price swept ${sweep_level:.0f} LONG, "
            f"hit target ${reversal_target:.0f} {target_result['bars_ago']} bars ago, "
            f"then dropped -{drop_pct:.1f}% to ${price:.0f}. "
            f"Watching for continuation to {target_label} ${new_target:.0f} ({(new_target-price)/price*100:.1f}%). "
            f"Invalidation above ${new_invalidation:.0f}."
        )

    return PhaseResult(
        phase=phase,
        confidence=round(confidence * 0.5, 3),  # halved — secondary setup
        direction=new_direction,
        narrative=narrative,
        signals_for=signals_for,
        signals_against=signals_against,
        key_level=sweep_level,
        key_level_name=f'secondary_from_{key_level_name}',
        trade_bias='WATCH',
        sweep_status='SECONDARY',
        sweep_level=sweep_level,
        reversal_target=new_target,
        reversal_target_name='midpoint_retest',
        timing_signal='NONE',
        entry_zone_low=min(new_entry, new_target),
        entry_zone_high=max(new_entry, new_target),
        invalidation=new_invalidation,
    )


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
    # v3 additions — cascading reversal targets for Judas sweeps
    target_tiers: List[Dict] = field(default_factory=list)  # [{price, type, strength, distance_pct, reasoning}]
    # v3 additions — zone sweep detection + completeness assessment
    sweep_completeness: Optional[Dict] = None  # _assess_sweep_completeness output
    zone_sweep: Optional[Dict] = None  # _check_zone_swept output


def _build_target_tiers(price, direction, magnets, sr_levels, liq, sweep_target, stats):
    """Build cascading reversal targets for MANIPULATION/Judas sweep setups.

    Returns list of dicts sorted by distance from price, each with:
        price, type, strength, distance_pct, reasoning, is_primary
    """
    tiers = []
    ref_price = sweep_target if sweep_target else price

    # ── Stat-based targets from historical data ──
    if stats:
        median_drop = stats.get('median_drop', 1.18)
        avg_drop = stats.get('avg_drop', 1.68)
        tiers.append({
            'price': ref_price * (1 - median_drop / 100),
            'type': 'STAT_MEDIAN',
            'strength': 0,
            'distance_pct': -median_drop,
            'reasoning': f'Historical median drop ({median_drop:.2f}% from sweep high)',
            'is_primary': True,
        })
        tiers.append({
            'price': ref_price * (1 - avg_drop / 100),
            'type': 'STAT_AVERAGE',
            'strength': 0,
            'distance_pct': -avg_drop,
            'reasoning': f'Historical avg drop ({avg_drop:.2f}% from sweep high)',
            'is_primary': True,
        })

    # ── Support levels (S/R) ──
    if sr_levels:
        for entry in sr_levels:
            if len(entry) >= 3:
                p, s, t = entry[0], entry[1], entry[2]
                if t == 'SUPPORT' and p < ref_price:
                    dist = (p - ref_price) / ref_price * 100
                    reasoning = f'Support ({s:.0f} touches)' if s > 0 else 'Support level'
                    tiers.append({
                        'price': p,
                        'type': 'SUPPORT',
                        'strength': s,
                        'distance_pct': dist,
                        'reasoning': reasoning,
                        'is_primary': s >= 100,
                    })

    # ── Magnets (volume clusters) ──
    if magnets:
        for entry in magnets:
            p = entry[0]
            s = entry[1] if len(entry) > 1 else 1
            if p < ref_price:
                dist = (p - ref_price) / ref_price * 100
                swept = entry[3] if len(entry) > 3 else False
                status = 'SWEPT' if swept else 'UNSWEPT'
                tiers.append({
                    'price': p,
                    'type': 'MAGNET',
                    'strength': s,
                    'distance_pct': dist,
                    'reasoning': f'Volume cluster ({s:.1f}x) [{status}]',
                    'is_primary': s >= 2.0 and not swept,
                })

    # ── Liquidity levels (stops/liqs below) ──
    if liq:
        below = liq.get('below', [])
        for z in below:
            if z.get('swept'):
                continue
            p = z.get('price', 0)
            s = z.get('strength', 0)
            cascade = z.get('cascade', 'LOW')
            z_type = z.get('type', 'UNKNOWN')
            dist = (p - ref_price) / ref_price * 100

            label_map = {
                'LONG_STOP': 'Long stops',
                'LONG_LIQ': 'Long liquidations',
                'BID_WALL': 'Bid wall',
            }
            label = label_map.get(z_type, z_type)
            reasoning = f'{label} (str={s}, cascade={cascade})'
            tiers.append({
                'price': p,
                'type': z_type,
                'strength': s,
                'distance_pct': dist,
                'reasoning': reasoning,
                'is_primary': cascade in ('MED', 'HIGH') and z_type in ('LONG_LIQ', 'LONG_STOP'),
            })

    # Deduplicate by price (within 0.1%)
    seen = []
    unique = []
    for t in sorted(tiers, key=lambda x: x['price'], reverse=True):
        if not any(abs(t['price'] - s) / s < 0.001 for s in seen):
            seen.append(t['price'])
            unique.append(t)

    return unique


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
    zone_result = None
    completeness = None

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

        # ── v3: Zone sweep detection (tolerance + zone awareness) ──
        # If exact match failed, check if the zone around the key level was swept.
        # This catches near-misses like $2,323 vs $2,326 (0.1% miss).
        zone_result = None
        zone_swept = False
        if not swept and df_15m is not None and highs_arr is not None:
            zone_result = _check_zone_swept(
                price, key_level, highs_arr, lows_arr,
                magnets, sr_levels, liq,
                len(df_15m) - 1, impl_direction,
                lookback=cfg.get('P3_SWEEP_LOOKBACK', SWEEP_LOOKBACK_BARS))
            zone_swept = zone_result.get('zone_swept', False) and zone_result.get('is_fresh', False)

            if zone_swept:
                # Zone was swept — now assess completeness
                m4b = result.get('m4b', {})
                completeness = _assess_sweep_completeness(
                    price=price,
                    swept_zone_high=zone_result.get('zone_high', 0),
                    unswept_above=unswept_above,
                    unswept_below=unswept_below,
                    m4b_divergence=m4b.get('divergence', 'NONE'),
                    m4b_slope=m4b.get('cvd_slope', 0),
                    oi_roc=deriv.get('oi_roc_1h', 0),
                    taker_ratio=result.get('raw_taker_ratio', 0.5),
                    direction_after_sweep=impl_direction,
                    whale_signal=whale,
                    futures_flow=futures_flow,
                )

                if completeness['status'] == 'PARTIAL_SWEEP':
                    # Not done yet — price likely to revisit the zone
                    narrative = (
                        f"Partial sweep detected at ${zone_result['zone_high']:.0f} "
                        f"({zone_result['bars_ago']} bars ago). "
                        f"{completeness['recommendation']}. "
                        f"Unswept gravity: str={completeness['unswept_strength']:.0f}. "
                        f"Watch for revisit to ${key_level:.0f} zone."
                    )
                    return PhaseResult(
                        phase=phase, confidence=round(confidence * 0.6, 3),
                        direction='NEUTRAL', narrative=narrative,
                        signals_for=completeness['factors'],
                        signals_against=['Partial sweep — not ready to trade'],
                        key_level=key_level, key_level_name=key_level_name,
                        trade_bias='WAIT',
                        sweep_status='PARTIAL_SWEEP', sweep_level=key_level,
                        reversal_target=None, reversal_target_name='',
                        timing_signal='NONE',
                        entry_zone_low=None, entry_zone_high=None,
                        invalidation=None,
                        target_tiers=[],
                        sweep_completeness=completeness,
                        zone_sweep=zone_result,
                    )
                elif completeness['status'] == 'SWEEP_DONE':
                    # Zone swept + reversal confirmed — treat as completed sweep
                    swept = True
                    bars_ago = zone_result['bars_ago']
                    is_stale = bars_ago > SWEEP_MAX_ACTIONABLE_AGE
                    # Update sweep_result for downstream logic
                    sweep_result = {
                        'swept': True, 'swept_at': len(df_15m) - 1 - bars_ago,
                        'sweep_depth_pct': 0, 'bars_ago': bars_ago,
                        'is_stale': is_stale, 'sweep_price': zone_result['zone_high'],
                    }
                # AMBIGUOUS falls through to existing logic (no sweep detected)

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
                # Both sweep AND target hit — check if a secondary setup is forming
                secondary = _detect_secondary_setup(
                    price, key_level, reversal_target, target_result,
                    highs_arr, lows_arr,
                    len(df_15m) - 1 if df_15m is not None else 0,
                    phase, confidence, key_level_name, impl_direction)

                if secondary:
                    return secondary

                # No secondary setup — move is done
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
                # Target already hit — check if secondary setup is forming
                secondary = _detect_secondary_setup(
                    price, key_level, reversal_target, target_result,
                    highs_arr, lows_arr,
                    len(df_15m) - 1 if df_15m is not None else 0,
                    phase, confidence, key_level_name, impl_direction)

                if secondary:
                    return secondary

                # No secondary — move done
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

    # ── Build target tiers for MANIPULATION phase ──
    target_tiers = []
    if phase == 'MANIPULATION' and key_level:
        # Get judas stats for the estimated sweep depth
        ref_for_stats = entry_zone_low if entry_zone_low else price
        est_sweep_pct = abs(key_level - ref_for_stats) / ref_for_stats * 100 if ref_for_stats > 0 else 0.3
        stats = get_judas_stats_for_sweep(est_sweep_pct)
        target_tiers = _build_target_tiers(
            price, impl_direction, magnets, sr_levels, liq, key_level, stats)

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
        target_tiers=target_tiers,
        sweep_completeness=completeness,
        zone_sweep=zone_result,
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
        if sweep_level is not None:
            parts.append(f"\n⏳ Sweep IN PROGRESS at ${sweep_level:.0f}.")
        else:
            parts.append(f"\n⏳ Sweep IN PROGRESS — level pending confirmation.")
        if reversal_target:
            target_dist = abs(reversal_target - price) / price * 100
            parts.append(f"🎯 After sweep: reversal target ${reversal_target:.0f} ({reversal_target_name}, {target_dist:+.1f}%).")
        parts.append("Wait for sweep completion + rejection confirmation before entering.")

    elif sweep_status == 'PENDING':
        if sweep_level is not None:
            parts.append(f"\n⏳ Sweep PENDING — momentum heading toward ${sweep_level:.0f}.")
        else:
            parts.append(f"\n⏳ Sweep PENDING — waiting for level confirmation.")
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
                       'EXPIRED': '💀', 'STALE': '⚠️', 'TARGET_HIT': '🎯',
                       'SECONDARY': '🔄', 'PARTIAL_SWEEP': '🔶'}
        icon = sweep_icons.get(pr.sweep_status, '❓')
        lines.append(f'  Sweep: {icon} {pr.sweep_status}' +
                     (f' at ${pr.sweep_level:.0f}' if pr.sweep_level else ''))

    # ── v3: Zone sweep + completeness assessment ──
    if pr.zone_sweep and pr.zone_sweep.get('zone_swept'):
        zs = pr.zone_sweep
        lines.append(f'\n  🔍 Zone Sweep Detection:')
        lines.append(f'    Zone high:    ${zs["zone_high"]:.2f}  ({zs["bars_ago"]} bars ago)')
        lines.append(f'    Zone center:  ${zs["zone_center"]:.2f}')
        hit_levels = [l for l in zs.get('swept_levels', []) if l.get('hit')]
        missed_levels = [l for l in zs.get('swept_levels', []) if not l.get('hit')]
        if hit_levels:
            lines.append(f'    Levels hit:   {", ".join(f"${l["price"]:.0f}({l["type"]})" for l in hit_levels)}')
        if missed_levels:
            lines.append(f'    Levels missed: {", ".join(f"${l["price"]:.0f}({l["type"]})" for l in missed_levels)}')

    if pr.sweep_completeness:
        sc = pr.sweep_completeness
        status_icons = {'SWEEP_DONE': '✅', 'PARTIAL_SWEEP': '🔶', 'AMBIGUOUS': '❓'}
        s_icon = status_icons.get(sc['status'], '❓')
        lines.append(f'\n  📊 Sweep Completeness: {s_icon} {sc["status"]}  (score={sc["score"]:+d})')
        lines.append(f'    Recommendation: {sc["recommendation"]}')
        for f in sc.get('factors', []):
            lines.append(f'      • {f}')

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
        'WATCH': '⏳ Watch — secondary setup forming from target bounce. Reduced confidence.',
        'WAIT': f'⏳ Wait — let the sweep play out, then enter {pr.direction}',
        'AVOID': '🚫 Avoid — no actionable setup (sweep/target already completed)',
    }
    action = action_map.get(pr.trade_bias, '❓ Unknown')
    if pr.trade_bias.startswith('ENTER_') and pr.entry_zone_low:
        action += f'\n    Entry: ${pr.entry_zone_low:.0f}–${pr.entry_zone_high:.0f}  TP: ${pr.reversal_target:.0f}  SL: ${pr.invalidation:.0f}'
    if pr.sweep_status == 'SECONDARY' and pr.invalidation:
        action += f'\n    TP: ${pr.reversal_target:.0f}  SL: ${pr.invalidation:.0f}'

    lines.append(f'\n  Action: {action}')

    # ── Judas Sweep Historical Performance (only for MANIPULATION phase) ──
    if pr.phase == 'MANIPULATION' and pr.key_level:
        # Estimate sweep depth from key level and entry zone
        ref_price = pr.entry_zone_low if pr.entry_zone_low else pr.key_level
        if ref_price and ref_price > 0:
            # key_level is the sweep target, entry_zone_low is near current price
            est_sweep_pct = abs(pr.key_level - ref_price) / ref_price * 100
        else:
            est_sweep_pct = 0.3  # default estimate

        stats = get_judas_stats_for_sweep(est_sweep_pct)
        ttr = JUDAS_SWEEP_STATS['time_to_reversal']

        lines.append('')
        lines.append('  📊 JUDAS SWEEP — HISTORICAL PERFORMANCE')
        lines.append('  ─────────────────────────────────────────')
        lines.append(f"  Source: {JUDAS_SWEEP_STATS['source']}")
        lines.append(f"  Matching depth: {stats['depth_label']}  (n={stats['depth_count']})")
        lines.append(f"  Reversal rate:  {stats['depth_reversal_rate']:.1%} (>0.3% drop)")
        lines.append(f"  Avg drop:       {stats['depth_avg_drop']:.2f}%")
        lines.append(f"  Overall (bull+sellers): {stats['overall_reversal_rate']:.1%}  "
                     f"avg={stats['overall_avg_drop']:.2f}%  med={stats['overall_median_drop']:.2f}%")
        lines.append(f"  Recent (2025-26):       {stats['recent_reversal_rate']:.1%}  "
                     f"avg={stats['recent_avg_drop']:.2f}%")
        lines.append(f"  Time to reversal:  med={ttr['median_bars']}bars ({ttr['median_bars']*15}min)  "
                     f"mean={ttr['mean_bars']}bars ({ttr['mean_bars']*15}min)")
        lines.append(f"  If fails: avg extension = {stats['avg_extension']:.2f}%")

        # R:R estimate
        est_entry = pr.entry_zone_low if pr.entry_zone_low else 0
        est_sl = pr.invalidation if pr.invalidation else 0
        est_tp = pr.reversal_target if pr.reversal_target else 0
        if est_entry > 0 and est_sl > est_entry and est_tp < est_entry:
            risk = (est_sl - est_entry) / est_entry * 100
            reward = (est_entry - est_tp) / est_entry * 100
            if risk > 0:
                lines.append(f"  R:R estimate:    {reward/risk:.2f}x  (reward={reward:.2f}% risk={risk:.2f}%)")

    # ── Reversal Target Tiers (MANIPULATION only) ──
    if pr.phase == 'MANIPULATION' and pr.target_tiers:
        lines.append('')
        lines.append('  🎯 REVERSAL TARGET TIERS (from sweep high)')
        lines.append('  ─────────────────────────────────────────')

        # Group by type
        primary = [t for t in pr.target_tiers if t.get('is_primary')]
        support = [t for t in pr.target_tiers if t['type'] == 'SUPPORT']
        magnets = [t for t in pr.target_tiers if t['type'] == 'MAGNET']
        liq_levels = [t for t in pr.target_tiers if t['type'] in ('LONG_STOP', 'LONG_LIQ', 'BID_WALL')]

        if primary:
            lines.append('  📐 Statistical targets:')
            for t in primary[:3]:
                marker = '◀ PRIMARY' if t == primary[0] else ''
                lines.append(f"    ${t['price']:.2f}  {t['distance_pct']:+.2f}%  "
                             f"{t['reasoning']}  {marker}")

        if support:
            lines.append('  🏗️ Support levels:')
            for t in support[:4]:
                marker = '◀ STRONG' if t.get('strength', 0) >= 100 else ''
                lines.append(f"    ${t['price']:.2f}  {t['distance_pct']:+.2f}%  "
                             f"{t['reasoning']}  {marker}")

        if magnets:
            lines.append('  🧲 Volume magnets:')
            for t in magnets[:3]:
                marker = '◀ HIGH' if t.get('strength', 0) >= 2.0 else ''
                lines.append(f"    ${t['price']:.2f}  {t['distance_pct']:+.2f}%  "
                             f"{t['reasoning']}  {marker}")

        if liq_levels:
            lines.append('  💧 Liquidity pools:')
            for t in liq_levels[:4]:
                cascade = '🔥' if 'HIGH' in t.get('reasoning', '') else ('⚡' if 'MED' in t.get('reasoning', '') else '')
                lines.append(f"    ${t['price']:.2f}  {t['distance_pct']:+.2f}%  "
                             f"{t['reasoning']}  {cascade}")

        # Sweet spot recommendation
        if primary and (support or magnets or liq_levels):
            # Find the level closest to the stat average target
            stat_avg = [t for t in pr.target_tiers if t['type'] == 'STAT_AVERAGE']
            if stat_avg:
                avg_price = stat_avg[0]['price']
                all_levels = support + magnets + liq_levels
                if all_levels:
                    closest = min(all_levels, key=lambda t: abs(t['price'] - avg_price))
                    if abs(closest['price'] - avg_price) / avg_price < 0.01:  # within 1%
                        lines.append(f'  ⭐ Sweet spot: ${closest["price"]:.2f} '
                                     f'(stat avg ${avg_price:.0f} + {closest["type"].lower()} confluence)')

    return '\n'.join(lines)


def phase_to_dict(pr: PhaseResult) -> dict:
    """Serialize PhaseResult to dict for JSON output."""
    d = asdict(pr)
    # Embed judas sweep stats when in MANIPULATION phase
    if pr.phase == 'MANIPULATION' and pr.key_level:
        ref_price = pr.entry_zone_low if pr.entry_zone_low else pr.key_level
        if ref_price and ref_price > 0:
            est_sweep_pct = abs(pr.key_level - ref_price) / ref_price * 100
        else:
            est_sweep_pct = 0.3
        d['judas_sweep_stats'] = {
            'estimated_sweep_pct': est_sweep_pct,
            'depth_match': get_judas_stats_for_sweep(est_sweep_pct),
            'historical': JUDAS_SWEEP_STATS['judas_context'],
            'time_to_reversal': JUDAS_SWEEP_STATS['time_to_reversal'],
            'recent': JUDAS_SWEEP_STATS['recent_2025_2026'],
        }
        # target_tiers already included via asdict
    return d

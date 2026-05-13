#!/usr/bin/env python3
"""
Conflict Resolver — Post-scan conflict detection and forward test setup.

When the scanner produces NO_SIGNAL but has conflicting signals,
this module identifies the conflict, picks the key level to watch,
and sets up a forward test with concrete triggers.

Slots into scanner.py as a post-scan step.
"""

import os
import json
from dataclasses import dataclass, field, asdict
from typing import List, Optional, Dict, Any


from src.modules.m_historical_sweep import get_historical_pattern, format_historical_pattern


@dataclass
class ConflictScenario:
    """A single conflict hypothesis."""
    name: str              # e.g. "WYCKOFF_DISTRIBUTION"
    direction: str         # SHORT or LONG
    trigger: str           # what needs to happen
    conditions: List[str]  # human-readable conditions
    confidence: float      # 0-1, how many signals support this


@dataclass
class ForwardTest:
    """A forward test watching a key level for resolution."""
    key_level_low: float
    key_level_high: float
    level_name: str        # e.g. "unswept_liquidity_zone"
    scenarios: List[ConflictScenario] = field(default_factory=list)
    resolution: str = "PENDING"  # PENDING | scenario.name when resolved


@dataclass
class ConflictResult:
    """Full output of conflict analysis."""
    has_conflict: bool
    conflict_type: str           # DIRECTION_DIVERGENCE | MODULE_DISAGREEMENT | REGIME_CONFLICT
    severity: str                # LOW | MEDIUM | HIGH
    summary: str                 # human-readable one-liner
    factors_for: List[str]       # factors supporting framework A
    factors_against: List[str]   # factors supporting framework B
    forward_test: Optional[ForwardTest] = None
    precaution: str = ""         # what to do / not do
    action_plan: List[str] = field(default_factory=list)
    historical_pattern: Optional[Dict[str, Any]] = None  # statistical tiebreaker


def detect_conflict(result: dict, config: dict = None) -> ConflictResult:
    """Analyze scan result for conflicts and set up forward tests.

    Args:
        result: The scan_signal() output dict
        config: Optional config overrides

    Returns:
        ConflictResult with conflict analysis and optional forward test
    """
    cfg = config or {}
    direction = result.get('direction', 'NEUTRAL')
    status = result.get('status', 'NO_SIGNAL')
    phase0 = result.get('phase0')
    swing_bias = result.get('swing_bias', 'NEUTRAL')

    # Extract module signals
    deriv = result.get('derivatives', {})
    whale = deriv.get('whale_signal', 'NEUTRAL')
    positioning = deriv.get('positioning', 'NEUTRAL')
    ls_zscore = deriv.get('ls_zscore', 0)
    futures_flow = deriv.get('futures_flow', 'NEUTRAL')
    funding_rate = deriv.get('funding_rate')

    m1_dir = result.get('m1', {}).get('direction', 'NEUTRAL')
    m2_status = result.get('m2', {}).get('status', 'N/A')
    m3_status = result.get('m3', {}).get('status', 'N/A')
    m4_status = result.get('m4', {}).get('status', 'N/A')
    m4_div = result.get('m4', {}).get('div', {})
    m5_status = result.get('m5', {}).get('status', 'N/A')
    m9_regime = result.get('m9', {}).get('regime', 'UNKNOWN')
    m13_bias = result.get('m13', {}).get('bias', 'NEUTRAL')

    cascade = result.get('cascade_risk', {})
    cascade_verdict = cascade.get('verdict', 'UNKNOWN')

    # Conflict history
    conflict_hist = result.get('conflict', {})
    hist = conflict_hist.get('historical', {})
    is_hist_conflict = conflict_hist.get('is_conflict', False)
    rev_24h = hist.get('windows', {}).get('24h', {}).get('reversal_rate', 50)

    # Unswept liquidity
    liq = result.get('liquidity_levels', {})
    magnets = result.get('magnets', [])
    sr_levels = result.get('sr_levels', [])

    # ── Detect conflicts ──
    conflicts = []
    factors_for = []    # supporting LONG / bullish
    factors_against = []  # supporting SHORT / bearish

    # 1. Direction vs Whales
    if direction == 'LONG' and whale in ('WHALE_BEARISH',):
        conflicts.append('DIRECTION_WHALE_DIVERGENCE')
        factors_against.append(f'Whales {whale} — smart money against {direction}')

    if direction == 'SHORT' and whale in ('WHALE_BULLISH',):
        conflicts.append('DIRECTION_WHALE_DIVERGENCE')
        factors_against.append(f'Whales {whale} — smart money against {direction}')

    # 2. Direction vs Phase0
    if phase0 is not None and phase0 < 0.15:
        conflicts.append('PHASE0_DEATH_ZONE')
        factors_against.append(f'Phase0={phase0:.3f} (death zone <0.15) — weak macro context')

    # 3. Direction vs Historical reversal rate
    if is_hist_conflict or rev_24h > 55:
        conflicts.append('HIGH_REVERSAL_RATE')
        factors_against.append(f'Historical {rev_24h:.0f}% reversal rate at 24h for similar setups')

    # 4. Module disagreements
    if m2_status == 'FAIL':
        factors_against.append('M2 EMA confluence FAIL — multi-TF trend disagreement')

    if direction == 'LONG' and m1_dir == 'BEARISH':
        conflicts.append('M1_DIRECTION_DISAGREEMENT')
        factors_against.append(f'M1 MACD says {m1_dir}, resolver says {direction}')

    if direction == 'SHORT' and m1_dir == 'BULLISH':
        conflicts.append('M1_DIRECTION_DISAGREEMENT')
        factors_against.append(f'M1 MACD says {m1_dir}, resolver says {direction}')

    # 5. Crowded positioning
    if direction == 'LONG' and positioning == 'CROWDED_LONG':
        conflicts.append('CROWDED_POSITIONING')
        factors_against.append('Crowded long positioning — squeeze risk')

    if direction == 'SHORT' and positioning == 'CROWDED_SHORT':
        conflicts.append('CROWDED_POSITIONING')
        factors_against.append('Crowded short positioning — squeeze risk')

    # 6. Cascade risk aligned against direction
    if direction == 'LONG' and cascade_verdict == 'CASCADE':
        conflicts.append('CASCADE_AGAINST_DIRECTION')
        factors_against.append('Cascade risk aligned against LONG direction')

    if direction == 'SHORT' and cascade_verdict == 'CASCADE':
        conflicts.append('CASCADE_AGAINST_DIRECTION')
        factors_against.append('Cascade risk aligned against SHORT direction')

    # ── Factors supporting the framework's direction ──
    if m13_bias != 'NEUTRAL':
        factors_for.append(f'M13 structure: {m13_bias}')

    if swing_bias != 'NEUTRAL':
        factors_for.append(f'Daily swing bias: {swing_bias}')

    if direction == 'LONG' and m3_status == 'PASS':
        factors_for.append('M3 VWAP PASS — price above VWAP zone')

    if direction == 'SHORT' and m3_status == 'PASS':
        factors_for.append('M3 VWAP PASS — price below VWAP zone')

    if m9_regime in ('NEUTRAL', 'TRENDING'):
        factors_for.append(f'M9 regime: {m9_regime} (favorable)')

    # Spot flow
    spot_sigs = result.get('exchange_activity', {}).get('spot_signals', {})
    spot_flow = spot_sigs.get('spot_flow', '?')
    if spot_flow == 'BUYERS' and direction == 'LONG':
        factors_for.append('Spot flow: buyers (supports long)')
    elif spot_flow == 'SELLERS' and direction == 'SHORT':
        factors_for.append('Spot flow: sellers (supports short)')

    # ── Historical pattern (statistical tiebreaker) ──
    historical_pattern = get_historical_pattern(result, config)

    # ── Determine severity ──
    # Historical pattern can reduce severity if it confirms one side
    if historical_pattern and historical_pattern.get('resolution_hint') != 'NEUTRAL':
        # Strong historical pattern = reduce conflict severity by one level
        hint = historical_pattern['resolution_hint']
        sweep_rate = historical_pattern.get('sweep_rate', 0)
        if sweep_rate >= 0.85:
            # Historical data strongly supports one path — downgrade severity
            if len(conflicts) >= 3:
                severity = 'MEDIUM'
            elif len(conflicts) >= 2:
                severity = 'LOW'
            elif len(conflicts) >= 1:
                severity = 'LOW'
    else:
        if len(conflicts) >= 3:
            severity = 'HIGH'
        elif len(conflicts) >= 2:
            severity = 'MEDIUM'
        elif len(conflicts) >= 1:
            severity = 'LOW'
        else:
            severity = 'NONE'

    has_conflict = severity != 'NONE'

    if not has_conflict:
        return ConflictResult(
            has_conflict=False,
            conflict_type='NONE',
            severity='NONE',
            summary='No conflict — scanner verdict stands',
            factors_for=factors_for,
            factors_against=factors_against,
            historical_pattern=historical_pattern,
        )

    # ── Determine conflict type ──
    if 'DIRECTION_WHALE_DIVERGENCE' in conflicts:
        conflict_type = 'DIRECTION_DIVERGENCE'
    elif 'M1_DIRECTION_DISAGREEMENT' in conflicts:
        conflict_type = 'MODULE_DISAGREEMENT'
    elif 'PHASE0_DEATH_ZONE' in conflicts:
        conflict_type = 'REGIME_CONFLICT'
    else:
        conflict_type = 'MIXED'

    # ── Find key level for forward test ──
    forward_test = _build_forward_test(result, direction, conflicts, magnets, sr_levels, liq,
                                       historical_pattern=historical_pattern)

    # ── Build summary ──
    opposing = len(factors_against)
    supporting = len(factors_for)
    summary = (f'{direction} bias conflicted: {opposing} signals against vs '
               f'{supporting} supporting. {severity} severity.')

    # ── Augment summary with historical pattern ──
    if historical_pattern and historical_pattern.get('confidence') in ('HIGH', 'MEDIUM'):
        hp = historical_pattern
        summary += f' Historical: {hp["expected_path"]} (sweep {hp["sweep_rate"]:.0%}, bounce {hp["bounce_rate"]:.0%}).'

    # ── Precaution ──
    precaution = _build_precaution(severity, conflict_type, direction, phase0, rev_24h)

    # ── Action plan ──
    action_plan = _build_action_plan(forward_test, direction, severity, result)

    return ConflictResult(
        has_conflict=True,
        conflict_type=conflict_type,
        severity=severity,
        summary=summary,
        factors_for=factors_for,
        factors_against=factors_against,
        forward_test=forward_test,
        precaution=precaution,
        action_plan=action_plan,
        historical_pattern=historical_pattern,
    )


def _build_forward_test(result, direction, conflicts, magnets, sr_levels, liq,
                        historical_pattern=None):
    """Pick the key level and build forward test scenarios."""
    price = result.get('price', 0)

    # Find best unswept liquidity zone
    unswept_above = []
    unswept_below = []
    if liq:
        unswept_above = [z for z in liq.get('above', []) if not z.get('swept')]
        unswept_below = [z for z in liq.get('below', []) if not z.get('swept')]

    # Pick the zone that matters most for the conflict
    key_zone = None
    zone_name = ''

    # Historical pattern can override zone selection
    if historical_pattern and historical_pattern.get('resolution_hint') == 'SWEEP_FIRST_THEN_LONG':
        # Historical says sweep DOWN first — key level is the lower boundary
        if unswept_below:
            bot = unswept_below[0]
            key_zone = (bot['price'], price)
            zone_name = f"unswept_liquidity_below_{bot['type']}"
        elif magnets:
            below_mags = [(p, s) for p, s, *_ in magnets if p < price]
            if below_mags:
                key_zone = (below_mags[0][0], price)
                zone_name = 'volume_magnet_below'
        elif sr_levels:
            supports = sorted([p for p, s, t, _, _ in sr_levels if t == 'SUPPORT'])
            if supports:
                key_zone = (supports[-1], price)
                zone_name = 'support_level'
    elif historical_pattern and historical_pattern.get('resolution_hint') == 'SWEEP_FIRST_THEN_SHORT':
        # Historical says sweep UP first — key level is the upper boundary
        if unswept_above:
            top = unswept_above[0]
            key_zone = (price, top['price'])
            zone_name = f"unswept_liquidity_above_{top['type']}"
        elif magnets:
            above_mags = [(p, s) for p, s, *_ in magnets if p > price]
            if above_mags:
                key_zone = (price, above_mags[0][0])
                zone_name = 'volume_magnet_above'
        elif sr_levels:
            resistances = sorted([p for p, s, t, _, _ in sr_levels if t == 'RESISTANCE'])
            if resistances:
                key_zone = (price, resistances[0])
                zone_name = 'resistance_level'

    # Fallback: original logic — but prefer closer S/R over distant unswept
    if not key_zone:
        if direction == 'LONG':
            # For LONG conflicts: watch above for rejection or breakout
            if unswept_above:
                top = unswept_above[0]
                # Check if a resistance S/R level is closer than the unswept zone
                if sr_levels:
                    resistances = sorted([(p, s) for p, s, t, _, _ in sr_levels if t == 'RESISTANCE' and p > price])
                    if resistances:
                        nearest_sr = resistances[0][0]
                        nearest_unswept = top['price']
                        sr_dist = abs(nearest_sr - price) / price
                        unswept_dist = abs(nearest_unswept - price) / price
                        if sr_dist < unswept_dist:
                            key_zone = (price, nearest_sr)
                            zone_name = 'resistance_level'
                        else:
                            key_zone = (price, nearest_unswept)
                            zone_name = f"unswept_liquidity_above_{top['type']}"
                    else:
                        key_zone = (price, top['price'])
                        zone_name = f"unswept_liquidity_above_{top['type']}"
                else:
                    key_zone = (price, top['price'])
                    zone_name = f"unswept_liquidity_above_{top['type']}"
            elif magnets:
                # Use nearest magnet above
                above_mags = [(p, s) for p, s, *_ in magnets if p > price]
                if above_mags:
                    key_zone = (price, above_mags[0][0])
                    zone_name = 'volume_magnet_above'
        else:
            # For SHORT conflicts: watch below for sweep or breakdown
            if unswept_below:
                bot = unswept_below[0]
                # Check if a support S/R level is closer than the unswept zone
                if sr_levels:
                    supports = sorted([(p, s) for p, s, t, _, _ in sr_levels if t == 'SUPPORT' and p < price], reverse=True)
                    if supports:
                        nearest_sr = supports[0][0]
                        nearest_unswept = bot['price']
                        sr_dist = abs(price - nearest_sr) / price
                        unswept_dist = abs(price - nearest_unswept) / price
                        if sr_dist < unswept_dist:
                            key_zone = (nearest_sr, price)
                            zone_name = 'support_level'
                        else:
                            key_zone = (bot['price'], price)
                            zone_name = f"unswept_liquidity_below_{bot['type']}"
                    else:
                        key_zone = (bot['price'], price)
                        zone_name = f"unswept_liquidity_below_{bot['type']}"
                else:
                    key_zone = (bot['price'], price)
                    zone_name = f"unswept_liquidity_below_{bot['type']}"
            elif magnets:
                below_mags = [(p, s) for p, s, *_ in magnets if p < price]
                if below_mags:
                    key_zone = (below_mags[0][0], price)
                    zone_name = 'volume_magnet_below'

    # Fallback: use S/R levels
    if not key_zone and sr_levels:
        supports = sorted([p for p, s, t, _, _ in sr_levels if t == 'SUPPORT'])
        resistances = sorted([p for p, s, t, _, _ in sr_levels if t == 'RESISTANCE'])
        if supports and resistances:
            key_zone = (supports[-1], resistances[0])
            zone_name = 'support_resistance_range'

    if not key_zone:
        return None

    # Build scenarios
    scenarios = []

    # Wyckoff / distribution scenario
    if direction == 'LONG' and 'DIRECTION_WHALE_DIVERGENCE' in conflicts:
        scenarios.append(ConflictScenario(
            name='WYCKOFF_DISTRIBUTION',
            direction='SHORT',
            trigger='sweep + volume rejection',
            conditions=[
                f'Price sweeps above ${key_zone[1]:.0f}',
                'Volume spike >= 1.5x 20MA',
                'Price drops >= 0.3% from zone high within 3 bars',
                'RSI divergence (high RSI on sweep, declining)',
            ],
            confidence=0.7,
        ))

    # Breakout scenario (opposing wyckoff)
    if direction == 'LONG':
        scenarios.append(ConflictScenario(
            name='GENUINE_BREAKOUT',
            direction='LONG',
            trigger='sweep + hold above',
            conditions=[
                f'Price holds above ${key_zone[1]:.0f} for 3+ bars',
                'Volume sustains above 1x 20MA (not fading)',
                'No immediate rejection within 2 bars',
            ],
            confidence=0.5,
        ))

    # For SHORT direction conflicts
    if direction == 'SHORT' and 'DIRECTION_WHALE_DIVERGENCE' in conflicts:
        scenarios.append(ConflictScenario(
            name='WYCKOFF_ACCUMULATION',
            direction='LONG',
            trigger='sweep below + volume recovery',
            conditions=[
                f'Price sweeps below ${key_zone[0]:.0f}',
                'Volume spike on sweep (capitulation)',
                'Price recovers above zone within 3 bars',
            ],
            confidence=0.7,
        ))

    if direction == 'SHORT':
        scenarios.append(ConflictScenario(
            name='GENUINE_BREAKDOWN',
            direction='SHORT',
            trigger='sweep below + hold below',
            conditions=[
                f'Price holds below ${key_zone[0]:.0f} for 3+ bars',
                'Volume expands on breakdown',
            ],
            confidence=0.5,
        ))

    # ── Historical pattern scenario (statistical tiebreaker) ──
    if historical_pattern and historical_pattern.get('confidence') in ('HIGH', 'MEDIUM'):
        hp = historical_pattern
        hint = hp.get('resolution_hint', 'NEUTRAL')
        sweep_rate = hp.get('sweep_rate', 0)
        bounce_rate = hp.get('bounce_rate', 0)
        avg_bars = hp.get('avg_sweep_bars', 8)
        avg_rr = hp.get('avg_bounce_rr', 1.0)

        if hint == 'SWEEP_FIRST_THEN_LONG':
            # Historical says: sweep down first, then bounce up
            scenarios.append(ConflictScenario(
                name='HISTORICAL_SWEEP_BOUNCE',
                direction='LONG',
                trigger=f'sweep below ${key_zone[0]:.0f} + reclaim within {avg_bars} bars',
                conditions=[
                    f'Sweep rate: {sweep_rate:.0%} — price likely to sweep the low first',
                    f'After sweep, {bounce_rate:.0%} chance of bounce to range top',
                    f'Avg bounce R:R: {avg_rr:.2f}x',
                    f'Wait for sweep, then evaluate bounce quality',
                ],
                confidence=sweep_rate * bounce_rate,  # combined probability
            ))
        elif hint == 'SWEEP_FIRST_THEN_SHORT':
            scenarios.append(ConflictScenario(
                name='HISTORICAL_SWEEP_REJECT',
                direction='SHORT',
                trigger=f'sweep above ${key_zone[1]:.0f} + reject within {avg_bars} bars',
                conditions=[
                    f'Sweep rate: {sweep_rate:.0%} — price likely to sweep the high first',
                    f'After sweep, {bounce_rate:.0%} chance of rejection to range bottom',
                    f'Avg rejection R:R: {avg_rr:.2f}x',
                    f'Wait for sweep, then evaluate rejection quality',
                ],
                confidence=sweep_rate * bounce_rate,
            ))

    if not scenarios:
        return None

    return ForwardTest(
        key_level_low=key_zone[0],
        key_level_high=key_zone[1],
        level_name=zone_name,
        scenarios=scenarios,
    )


def _build_precaution(severity, conflict_type, direction, phase0, rev_24h):
    """Generate precaution advice based on conflict analysis."""
    parts = []

    if severity == 'HIGH':
        parts.append('HIGH CONFLICT — do not trade until resolved.')
    elif severity == 'MEDIUM':
        parts.append('MEDIUM CONFLICT — reduce position size if trading.')

    if phase0 is not None and phase0 < 0.15:
        parts.append(f'Phase0 death zone ({phase0:.3f}) — macro context is weak.')

    if rev_24h > 55:
        parts.append(f'Historical reversal rate {rev_24h:.0f}% — similar setups fail more often than not.')

    if conflict_type == 'DIRECTION_DIVERGENCE':
        parts.append(f"Direction resolver says {direction} but smart money disagrees.")

    if not parts:
        parts.append('Minor conflict — proceed with caution.')

    return ' '.join(parts)


def _build_action_plan(forward_test, direction, severity, result):
    """Generate concrete action plan steps."""
    plan = []

    if not forward_test:
        plan.append('No clear key level identified — watch manually.')
        return plan

    ft = forward_test
    plan.append(f'Key level: ${ft.key_level_low:.0f}-${ft.key_level_high:.0f} ({ft.level_name})')
    plan.append(f'Watch for resolution:')

    for sc in ft.scenarios:
        plan.append(f'  → {sc.name}: {sc.trigger}')
        for cond in sc.conditions:
            plan.append(f'    • {cond}')

    if severity == 'HIGH':
        plan.append('Do NOT enter a trade until one scenario triggers.')
    elif severity == 'MEDIUM':
        plan.append('If entering, use reduced size (50% or less).')

    return plan


def format_conflict(cr: ConflictResult) -> str:
    """Format conflict result for terminal output."""
    if not cr.has_conflict:
        return ''

    lines = []
    lines.append('')
    lines.append('  ⚠️  CONFLICT DETECTED')
    lines.append(f'  Type: {cr.conflict_type}  Severity: {cr.severity}')
    lines.append(f'  {cr.summary}')

    lines.append(f'\n  Factors FOR framework direction:')
    for f in cr.factors_for:
        lines.append(f'    ✅ {f}')

    lines.append(f'\n  Factors AGAINST framework direction:')
    for f in cr.factors_against:
        lines.append(f'    ❌ {f}')

    lines.append(f'\n  ⚡ Precaution: {cr.precaution}')

    if cr.forward_test:
        ft = cr.forward_test
        lines.append(f'\n  🎯 Forward Test:')
        lines.append(f'     Key level: ${ft.key_level_low:.0f}-${ft.key_level_high:.0f} ({ft.level_name})')
        for sc in ft.scenarios:
            icon = '🔻' if sc.direction == 'SHORT' else '🔺'
            lines.append(f'     {icon} {sc.name} ({sc.direction}) — trigger: {sc.trigger}')
            for cond in sc.conditions:
                lines.append(f'        • {cond}')

    if cr.action_plan:
        lines.append(f'\n  📋 Action Plan:')
        for step in cr.action_plan:
            lines.append(f'     {step}')

    if cr.historical_pattern:
        lines.append(format_historical_pattern(cr.historical_pattern))

    return '\n'.join(lines)


def conflict_to_dict(cr: ConflictResult) -> dict:
    """Serialize ConflictResult to dict for JSON output."""
    d = {
        'has_conflict': cr.has_conflict,
        'conflict_type': cr.conflict_type,
        'severity': cr.severity,
        'summary': cr.summary,
        'factors_for': cr.factors_for,
        'factors_against': cr.factors_against,
        'precaution': cr.precaution,
        'action_plan': cr.action_plan,
    }
    if cr.forward_test:
        ft = cr.forward_test
        d['forward_test'] = {
            'key_level_low': ft.key_level_low,
            'key_level_high': ft.key_level_high,
            'level_name': ft.level_name,
            'resolution': ft.resolution,
            'scenarios': [asdict(s) for s in ft.scenarios],
        }
    if cr.historical_pattern:
        d['historical_pattern'] = cr.historical_pattern
    return d

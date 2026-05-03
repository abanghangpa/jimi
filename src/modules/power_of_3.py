"""
Power of 3 Phase Detector — ICT/SMC market phase analysis.

Determines whether the current market structure is:
  - ACCUMULATION: Smart money loading, range-bound, preparing for markup
  - MARKUP:       Real bullish move, structure is genuine
  - MANIPULATION: Judas swing — fake move to grab liquidity before reversal
  - DISTRIBUTION: Smart money selling to late entraders, preparing for markdown
  - MARKDOWN:     Real bearish move, structure is genuine

Uses data already computed by the framework:
  - M13 swing structure (HH/HL, LH/LL)
  - Whale positioning (smart money direction)
  - Phase0 (macro context)
  - Unswept liquidity (where stops sit)
  - Historical reversal rates
  - Funding rate / L/S ratio (crowded positioning)
  - Spot flow (real demand vs. leverage)
"""

import numpy as np
from dataclasses import dataclass, field, asdict
from typing import List, Optional, Dict


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
    trade_bias: str = ''    # What to do: ENTER_LONG | ENTER_SHORT | WAIT | AVOID


def detect_phase(result: dict, config: dict = None) -> PhaseResult:
    """Detect the current Power of 3 phase from scan results.

    Args:
        result: scan_signal() output dict
        config: Optional config overrides

    Returns:
        PhaseResult with phase, direction, and trade bias
    """
    cfg = config or {}
    price = result.get('price', 0)
    direction = result.get('direction', 'NEUTRAL')
    swing_bias = result.get('swing_bias', 'NEUTRAL')
    phase0 = result.get('phase0')

    # Structure
    m13 = result.get('m13', {})
    m13_bias = m13.get('bias', 'NEUTRAL')
    m13_score = m13.get('score', 0.5)
    m13_details = m13.get('details', {})

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
    unswept_above = [z for z in liq.get('above', []) if not z.get('swept')]
    unswept_below = [z for z in liq.get('below', []) if not z.get('swept')]

    # Conflict history
    conflict_hist = result.get('conflict', {})
    hist = conflict_hist.get('historical', {})
    rev_24h = hist.get('windows', {}).get('24h', {}).get('reversal_rate', 50)

    # Spot
    spot_sigs = result.get('exchange_activity', {}).get('spot_signals', {})
    spot_flow = spot_sigs.get('spot_flow', '?')

    # ── Scoring ──
    # Accumulation signals: price in range, smart money buying, low phase0
    accum_score = 0.0
    accum_signals = []

    # Markup signals: genuine bullish structure, aligned flow
    markup_score = 0.0
    markup_signals = []

    # Manipulation signals: bullish structure but smart money diverges
    manip_score = 0.0
    manip_signals = []

    # Distribution signals: bearish smart money, late buyers, high reversal rate
    distrib_score = 0.0
    distrib_signals = []

    # Markdown signals: genuine bearish structure, aligned flow
    markdown_score = 0.0
    markdown_signals = []

    # ── 1. Structure vs Smart Money alignment ──
    structure_bullish = m13_bias in ('BULLISH', 'LEAN_BULL')
    structure_bearish = m13_bias in ('BEARISH', 'LEAN_BEAR')
    smart_money_bearish = whale in ('WHALE_BEARISH',) or futures_flow == 'SELLERS_DOMINANT'
    smart_money_bullish = whale in ('WHALE_BULLISH',) or futures_flow == 'BUYERS_DOMINANT'

    # Divergence = manipulation signal
    if structure_bullish and smart_money_bearish:
        manip_score += 0.30
        manip_signals.append(f'Structure BULLISH but whales {whale} — order flow divergence')

    if structure_bearish and smart_money_bullish:
        manip_score += 0.30
        manip_signals.append(f'Structure BEARISH but whales {whale} — order flow divergence')

    # Alignment = markup/markdown signal
    if structure_bullish and smart_money_bullish:
        markup_score += 0.25
        markup_signals.append('Structure and smart money both bullish — aligned')

    if structure_bearish and smart_money_bearish:
        markdown_score += 0.25
        markdown_signals.append('Structure and smart money both bearish — aligned')

    # ── 2. Phase0 context ──
    if phase0 is not None:
        if phase0 < 0.15:
            # Death zone — weak macro, favors manipulation/distribution
            manip_score += 0.15
            manip_signals.append(f'Phase0={phase0:.3f} death zone — weak macro context')
            distrib_score += 0.10
            distrib_signals.append(f'Phase0={phase0:.3f} — unsustainable context')
        elif phase0 < 0.30:
            # Weak — favors accumulation
            accum_score += 0.10
            accum_signals.append(f'Phase0={phase0:.3f} — weak but not death zone')
        elif phase0 >= 0.60:
            # Strong — favors markup/markdown continuation
            if structure_bullish:
                markup_score += 0.15
                markup_signals.append(f'Phase0={phase0:.3f} — strong macro supports markup')
            elif structure_bearish:
                markdown_score += 0.15
                markdown_signals.append(f'Phase0={phase0:.3f} — strong macro supports markdown')

    # ── 3. Unswept liquidity direction ──
    # Price tends to sweep liquidity before reversing
    if unswept_above and not unswept_below:
        # Stops above = price likely to sweep up first
        if smart_money_bearish:
            manip_score += 0.15
            manip_signals.append(f'Unswept liquidity above (${unswept_above[0]["price"]:.0f}) — Judas swing target')
        else:
            markup_score += 0.10
            markup_signals.append(f'Unswept liquidity above — potential targets for continuation')

    elif unswept_below and not unswept_above:
        # Stops below = price likely to sweep down first
        if smart_money_bullish:
            manip_score += 0.15
            manip_signals.append(f'Unswept liquidity below (${unswept_below[0]["price"]:.0f}) — Judas swing target')
        else:
            markdown_score += 0.10
            markdown_signals.append(f'Unswept liquidity below — potential targets for continuation')

    elif unswept_above and unswept_below:
        # Stops on both sides = range-bound (accumulation/distribution)
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
    # Spot buying + leverage longs = distribution (spot is the exit, leverage is the trap)
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

    # Normalize confidence (cap at 1.0)
    confidence = min(raw_confidence, 1.0)

    # If confidence is too low, it's ambiguous
    if confidence < 0.20:
        phase = 'AMBIGUOUS'

    # ── Determine direction and trade bias ──
    if phase == 'ACCUMULATION':
        direction = 'LONG'
        trade_bias = 'WAIT'  # Wait for markup confirmation
        key_level = _find_breakout_level(unswept_above, magnets, price, 'above')
        key_level_name = 'markup_breakout'
    elif phase == 'MARKUP':
        direction = 'LONG'
        trade_bias = 'ENTER_LONG'
        key_level = _find_breakout_level(unswept_above, magnets, price, 'above')
        key_level_name = 'continuation_target'
    elif phase == 'MANIPULATION':
        # Manipulation implies reversal — trade against the current structure
        if structure_bullish:
            direction = 'SHORT'
            key_level = _find_sweep_level(unswept_above, magnets, price, 'above')
            key_level_name = 'judas_sweep_target'
        else:
            direction = 'LONG'
            key_level = _find_sweep_level(unswept_below, magnets, price, 'below')
            key_level_name = 'judas_sweep_target'
        trade_bias = 'WAIT'  # Wait for sweep + confirmation
    elif phase == 'DISTRIBUTION':
        direction = 'SHORT'
        trade_bias = 'WAIT'  # Wait for markdown confirmation
        key_level = _find_breakout_level(unswept_below, magnets, price, 'below')
        key_level_name = 'markdown_breakout'
    elif phase == 'MARKDOWN':
        direction = 'SHORT'
        trade_bias = 'ENTER_SHORT'
        key_level = _find_breakout_level(unswept_below, magnets, price, 'below')
        key_level_name = 'continuation_target'
    else:
        direction = 'NEUTRAL'
        trade_bias = 'AVOID'
        key_level = None
        key_level_name = ''

    # ── Build narrative ──
    narrative = _build_narrative(phase, direction, structure_bullish, smart_money_bearish,
                                 phase0, rev_24h, unswept_above, unswept_below, price)

    # Collect all signals
    all_signals_for = {
        'ACCUMULATION': accum_signals,
        'MARKUP': markup_signals,
        'MANIPULATION': manip_signals,
        'DISTRIBUTION': distrib_signals,
        'MARKDOWN': markdown_signals,
    }
    signals_for = all_signals_for.get(phase, [])
    signals_against = []
    for other_phase, other_signals in all_signals_for.items():
        if other_phase != phase:
            signals_against.extend(other_signals)

    return PhaseResult(
        phase=phase,
        confidence=round(confidence, 3),
        direction=direction,
        narrative=narrative,
        signals_for=signals_for,
        signals_against=signals_against[:5],  # Top 5 opposing
        key_level=key_level,
        key_level_name=key_level_name,
        trade_bias=trade_bias,
    )


def _find_sweep_level(unswept, magnets, price, side):
    """Find the most likely sweep target."""
    candidates = []
    for z in unswept:
        candidates.append(z['price'])
    for p, s, *_ in magnets:
        if side == 'above' and p > price:
            candidates.append(p)
        elif side == 'below' and p < price:
            candidates.append(p)
    if candidates:
        if side == 'above':
            return min(candidates)  # Nearest above
        else:
            return max(candidates)  # Nearest below
    return None


def _find_breakout_level(unswept, magnets, price, side):
    """Find the level that confirms breakout/breakdown."""
    return _find_sweep_level(unswept, magnets, price, side)


def _build_narrative(phase, direction, structure_bullish, smart_money_bearish,
                     phase0, rev_24h, unswept_above, unswept_below, price):
    """Build human-readable narrative explaining the phase."""

    if phase == 'MANIPULATION' and structure_bullish and smart_money_bearish:
        zone = unswept_above[0]['price'] if unswept_above else 'N/A'
        return (
            f"Bullish structure is likely a Judas swing. Smart money is bearish "
            f"while price makes higher highs — classic manipulation to grab "
            f"buy-side liquidity at ${zone:.0f}. After the sweep, expect reversal. "
            f"Don't buy the breakout."
        )

    elif phase == 'MANIPULATION' and not structure_bullish and smart_money_bearish:
        zone = unswept_below[0]['price'] if unswept_below else 'N/A'
        return (
            f"Bearish structure is likely a Judas swing. Smart money is bullish "
            f"while price makes lower lows — manipulation to grab sell-side "
            f"liquidity at ${zone:.0f}. After the sweep, expect reversal upward."
        )

    elif phase == 'DISTRIBUTION':
        return (
            f"Smart money is distributing. Bullish structure attracts late longs, "
            f"while whales sell into strength. Phase0={phase0:.3f} ({'weak' if phase0 and phase0 < 0.3 else 'moderate'}), "
            f"historical reversal rate {rev_24h:.0f}%. "
            f"Wait for markdown confirmation before shorting."
        )

    elif phase == 'ACCUMULATION':
        return (
            f"Smart money is accumulating. Bearish structure shakes out weak hands, "
            f"while whales buy into weakness. Wait for markup confirmation before going long."
        )

    elif phase == 'MARKUP':
        return (
            f"Genuine bullish move. Structure and smart money aligned. "
            f"Look for long entries on pullbacks to support."
        )

    elif phase == 'MARKDOWN':
        return (
            f"Genuine bearish move. Structure and smart money aligned. "
            f"Look for short entries on bounces to resistance."
        )

    else:
        return "Phase ambiguous — insufficient signal alignment. Wait for clarity."


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

    if pr.key_level:
        lines.append(f'\n  Key level: ${pr.key_level:.0f} ({pr.key_level_name})')

    # Action
    action_map = {
        'ENTER_LONG': '✅ Enter LONG on pullback to support',
        'ENTER_SHORT': '✅ Enter SHORT on bounce to resistance',
        'WAIT': f'⏳ Wait — let the Judas sweep play out, then enter {pr.direction}',
        'AVOID': '🚫 Avoid — no clear edge',
    }
    lines.append(f'\n  Action: {action_map.get(pr.trade_bias, "❓ Unknown")}')

    return '\n'.join(lines)


def phase_to_dict(pr: PhaseResult) -> dict:
    """Serialize PhaseResult to dict."""
    return asdict(pr)

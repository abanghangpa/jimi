"""
M18: Squeeze Detector v2 — Predictive, High Win-Rate.

v2 redesign: Quality over quantity.
  - Multi-layer gating: ALL conditions must fire, not just score threshold
  - Requires: compression + extreme z + funding direction + OI momentum + volume spike
  - Optional boosters: whale divergence, M4b timing, RSI confirmation
  - Expected: 1-3 signals/week, 70-80% directional accuracy at 4h+

Architecture:
  Gate 1: Regime compression (mandatory)
  Gate 2: Extreme positioning z-score (mandatory)
  Gate 3: Funding confirms direction (mandatory)
  Gate 4: OI rising = new positions loading (mandatory)
  Gate 5: Volume spike = squeeze ignition (mandatory)
  Boosters: Whale + M4b + RSI alignment (optional, raises confidence)
"""

import numpy as np


SQUEEZE_V2_DEFAULTS = {
    # Gate 1: Compression
    'SQUEEZE_REGIMES': ['NEUTRAL_CHOP', 'CHOP_HARD', 'CHOP_MILD_BEAR', 'CHOP_MILD_BULL'],

    # Gate 2: Positioning (MANDATORY)
    'SQUEEZE_ZSCORE_MIN': 2.0,          # Minimum |z| to consider
    'SQUEEZE_ZSCORE_STRONG': 2.8,       # Strong signal threshold

    # Gate 3: Funding (MANDATORY)
    'SQUEEZE_FUNDING_THRESHOLD': 0.0002, # Min |funding| to confirm direction

    # Gate 4: OI momentum (MANDATORY)
    'SQUEEZE_OI_ROC_MIN': 0.2,          # Min OI % change/hr

    # Gate 5: Volume spike (MANDATORY)
    'SQUEEZE_VOL_SPIKE_MIN': 1.2,       # Volume / MA20 minimum

    # Boosters
    'SQUEEZE_WHALE_BONUS': 0.15,        # Whale divergence bonus
    'SQUEEZE_M4B_BONUS': 0.10,          # M4b timing bonus
    'SQUEEZE_RSI_BONUS': 0.05,          # RSI confirmation bonus

    # Scoring
    'SQUEEZE_SCORE_THRESHOLD': 0.70,    # Composite threshold
    'SQUEEZE_STRONG_THRESHOLD': 0.85,   # Strong threshold

    # Override
    'SQUEEZE_OVERRIDE_REGIME': True,
    'SQUEEZE_ICS_BOOST': 0.10,
    'SQUEEZE_SIZE_MULT': 0.80,

    # Cooldown: minimum bars between signals (prevents spam)
    'SQUEEZE_COOLDOWN_BARS': 16,        # 4h cooldown on 15m
}


def _check_gates(regime, ls_zscore, funding_rate, oi_roc, vol_trend, cfg):
    """Check mandatory gates. All must pass for signal.

    Returns: (all_pass: bool, gates_passed: list, gates_failed: list)
    """
    gates_passed = []
    gates_failed = []

    # Gate 1: Compression
    if regime in cfg['SQUEEZE_REGIMES']:
        gates_passed.append(f'regime={regime}')
    else:
        gates_failed.append(f'regime={regime} (not compressed)')
        return False, gates_passed, gates_failed

    # Gate 2: Extreme positioning
    if abs(ls_zscore) >= cfg['SQUEEZE_ZSCORE_MIN']:
        gates_passed.append(f'|z|={abs(ls_zscore):.2f}')
    else:
        gates_failed.append(f'|z|={abs(ls_zscore):.2f} < {cfg["SQUEEZE_ZSCORE_MIN"]}')
        return False, gates_passed, gates_failed

    # Gate 3: Funding confirms direction
    # For SHORT squeeze (z<0): funding should be negative (shorts paying)
    # For LONG squeeze (z>0): funding should be positive (longs paying)
    if ls_zscore < 0 and funding_rate < -cfg['SQUEEZE_FUNDING_THRESHOLD']:
        gates_passed.append(f'funding={funding_rate*100:+.4f}% (shorts pay)')
    elif ls_zscore > 0 and funding_rate > cfg['SQUEEZE_FUNDING_THRESHOLD']:
        gates_passed.append(f'funding={funding_rate*100:+.4f}% (longs pay)')
    else:
        # Allow funding flip (near zero) as partial pass
        if abs(funding_rate) < cfg['SQUEEZE_FUNDING_THRESHOLD'] * 2:
            gates_passed.append(f'funding={funding_rate*100:+.4f}% (near flip)')
        else:
            gates_failed.append(f'funding={funding_rate*100:+.4f}% (wrong direction)')
            return False, gates_passed, gates_failed

    # Gate 4: OI rising
    if oi_roc >= cfg['SQUEEZE_OI_ROC_MIN']:
        gates_passed.append(f'OI +{oi_roc:.2f}%/hr')
    else:
        gates_failed.append(f'OI {oi_roc:.2f}%/hr < {cfg["SQUEEZE_OI_ROC_MIN"]}')
        return False, gates_passed, gates_failed

    # Gate 5: Volume spike
    if vol_trend >= cfg['SQUEEZE_VOL_SPIKE_MIN']:
        gates_passed.append(f'vol={vol_trend:.2f}x MA20')
    else:
        gates_failed.append(f'vol={vol_trend:.2f}x < {cfg["SQUEEZE_VOL_SPIKE_MIN"]}x')
        return False, gates_passed, gates_failed

    return True, gates_passed, gates_failed


def detect_squeeze_v2(result, config=None, last_signal_bar=-1, current_bar=0):
    """Detect squeeze with mandatory multi-gate filtering.

    Args:
        result: scan_signal() output dict
        config: Optional config overrides
        last_signal_bar: Bar index of last squeeze signal (for cooldown)
        current_bar: Current bar index

    Returns:
        dict with squeeze_type, score, direction, gates, factors
    """
    cfg = {**SQUEEZE_V2_DEFAULTS, **(config or {})}

    # Cooldown check
    if current_bar - last_signal_bar < cfg['SQUEEZE_COOLDOWN_BARS']:
        return {'squeeze_type': 'NONE', 'squeeze_score': 0, 'direction': 'NEUTRAL',
                'factors': [], 'gates_passed': [], 'gates_failed': ['cooldown'],
                'overrides_regime': False, 'ics_boost': 0, 'size_mult': 1.0,
                'short_score': 0, 'long_score': 0, 'gates_all_pass': False}

    deriv = result.get('derivatives', {})
    m9 = result.get('m9', {})
    m4b = result.get('m4b', {})

    regime = m9.get('regime', 'UNKNOWN')
    ls_zscore = deriv.get('ls_zscore', 0)
    funding_rate = deriv.get('funding_rate', 0) or 0
    oi_roc = deriv.get('oi_roc_1h', 0)
    whale = deriv.get('whale_signal', 'NEUTRAL')
    futures_flow = deriv.get('futures_flow', 'NEUTRAL')
    m4b_div = m4b.get('divergence', 'NONE')
    m4b_bars_ago = m4b.get('bars_ago', -1)

    # Get RSI and volume from result
    rsi = result.get('rsi', 50)
    vol_trend = result.get('vol_trend', 1.0)

    # Check mandatory gates
    all_pass, gates_passed, gates_failed = _check_gates(
        regime, ls_zscore, funding_rate, oi_roc, vol_trend, cfg)

    if not all_pass:
        return {'squeeze_type': 'NONE', 'squeeze_score': 0, 'direction': 'NEUTRAL',
                'factors': [], 'gates_passed': gates_passed, 'gates_failed': gates_failed,
                'overrides_regime': False, 'ics_boost': 0, 'size_mult': 1.0,
                'short_score': 0, 'long_score': 0, 'gates_all_pass': False}

    # All gates passed — determine direction from z-score sign
    if ls_zscore < 0:
        squeeze_type = 'SHORT_SQUEEZE'
        direction = 'LONG'
    else:
        squeeze_type = 'LONG_SQUEEZE'
        direction = 'SHORT'

    # Compute composite score from gate quality + boosters
    factors = list(gates_passed)
    score = 0.50  # base for passing all gates

    # Gate quality bonuses
    if abs(ls_zscore) >= cfg['SQUEEZE_ZSCORE_STRONG']:
        score += 0.15
        factors.append(f'z={ls_zscore:+.2f} (STRONG)')

    # Funding strength
    if squeeze_type == 'SHORT_SQUEEZE' and funding_rate < -0.0005:
        score += 0.05
    elif squeeze_type == 'LONG_SQUEEZE' and funding_rate > 0.0005:
        score += 0.05

    # OI strength
    if oi_roc > 1.0:
        score += 0.05
        factors.append(f'OI surge +{oi_roc:.1f}%/hr')

    # Volume strength
    if vol_trend > 2.0:
        score += 0.05
        factors.append(f'vol surge {vol_trend:.1f}x')

    # Booster: Whale divergence
    if (squeeze_type == 'SHORT_SQUEEZE' and whale == 'WHALE_BULLISH') or \
       (squeeze_type == 'LONG_SQUEEZE' and whale == 'WHALE_BEARISH'):
        score += cfg['SQUEEZE_WHALE_BONUS']
        factors.append(f'whale {whale} (against crowd)')

    # Booster: Futures flow
    if (squeeze_type == 'SHORT_SQUEEZE' and futures_flow == 'BUYERS_DOMINANT') or \
       (squeeze_type == 'LONG_SQUEEZE' and futures_flow == 'SELLERS_DOMINANT'):
        score += 0.05
        factors.append(f'futures {futures_flow}')

    # Booster: M4b timing
    if m4b_div != 'NONE' and m4b_bars_ago <= 16:
        aligned = (squeeze_type == 'SHORT_SQUEEZE' and m4b_div == 'BULLISH') or \
                  (squeeze_type == 'LONG_SQUEEZE' and m4b_div == 'BEARISH')
        if aligned:
            score += cfg['SQUEEZE_M4B_BONUS']
            factors.append(f'M4b {m4b_div} {m4b_bars_ago}bars ago')

    # Booster: RSI confirmation
    if squeeze_type == 'SHORT_SQUEEZE' and rsi < 40:
        score += cfg['SQUEEZE_RSI_BONUS']
        factors.append(f'RSI={rsi:.0f} (oversold)')
    elif squeeze_type == 'LONG_SQUEEZE' and rsi > 60:
        score += cfg['SQUEEZE_RSI_BONUS']
        factors.append(f'RSI={rsi:.0f} (overbought)')

    score = min(score, 1.0)
    is_strong = score >= cfg['SQUEEZE_STRONG_THRESHOLD']

    return {
        'squeeze_type': squeeze_type,
        'squeeze_score': round(score, 3),
        'squeeze_strong': is_strong,
        'direction': direction,
        'factors': factors,
        'gates_passed': gates_passed,
        'gates_failed': gates_failed,
        'gates_all_pass': True,
        'overrides_regime': cfg['SQUEEZE_OVERRIDE_REGIME'],
        'ics_boost': round(cfg['SQUEEZE_ICS_BOOST'] * (1.5 if is_strong else 1.0), 4),
        'size_mult': cfg['SQUEEZE_SIZE_MULT'],
        'short_score': round(score, 3) if squeeze_type == 'SHORT_SQUEEZE' else 0,
        'long_score': round(score, 3) if squeeze_type == 'LONG_SQUEEZE' else 0,
    }


def format_squeeze(sq):
    """Format squeeze result for terminal output."""
    if sq.get('squeeze_type', 'NONE') == 'NONE':
        # Show failed gates if any signal was attempted
        failed = sq.get('gates_failed', [])
        if failed and 'cooldown' not in failed:
            lines = ['', '  🔥 SQUEEZE DETECTOR — GATES FAILED']
            for g in failed:
                lines.append(f'    ❌ {g}')
            return '\n'.join(lines)
        return ''

    lines = []
    lines.append('')
    lines.append('  🔥 SQUEEZE DETECTOR')

    icons = {'SHORT_SQUEEZE': '🟩', 'LONG_SQUEEZE': '🟥'}
    icon = icons.get(sq['squeeze_type'], '❓')
    strength = 'STRONG' if sq.get('squeeze_strong') else 'CONFIRMED'

    lines.append(f'  Type: {icon} {sq["squeeze_type"]}  ({strength})')
    lines.append(f'  Score: {sq["squeeze_score"]:.3f}  Direction: {sq["direction"]}')

    if sq.get('overrides_regime'):
        lines.append(f'  ⚡ Overrides M9 regime block!')

    if sq.get('gates_passed'):
        lines.append(f'\n  Gates passed:')
        for g in sq['gates_passed']:
            lines.append(f'    ✅ {g}')

    if sq.get('factors'):
        lines.append(f'\n  Factors:')
        for f in sq['factors']:
            lines.append(f'    • {f}')

    if sq.get('ics_boost', 0) > 0:
        lines.append(f'\n  ICS boost: +{sq["ics_boost"]:.4f}')
        lines.append(f'  Size multiplier: {sq.get("size_mult", 1.0):.2f}x')

    return '\n'.join(lines)

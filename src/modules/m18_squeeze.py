"""
M18: Squeeze Detector v4 — Multi-bar state machine.

Squeezes are multi-bar patterns, not single-bar events.

Phase 1 — COMPRESSION: Range narrows (range48 < 1.5%), volume dries up
Phase 2 — IGNITION: Explosive volume spike (vol > 3x MA20) with directional move
Phase 3 — CONTINUATION: Sustained follow-through

Architecture:
  - Track compression state across bars (rolling range48 + vol dry-up)
  - Detect ignition bar within compressed context
  - Direction from taker ratio + price action on ignition bar
  - No M4b gate (was incorrectly requiring BEARISH for short squeezes)
  - Quality = compression depth + ignition strength
"""

import numpy as np


SQUEEZE_V4_DEFAULTS = {
    # Compression detection
    'SQUEEZE_RANGE48_MAX': 1.6,         # range48% must be below this (recent min)
    'SQUEEZE_RANGE48_TIGHT': 0.8,       # ultra-tight = bonus quality
    'SQUEEZE_VOL_DRYUP_MIN': 6,         # bars of vol < 0.6x MA20 in last 24
    'SQUEEZE_COMPRESSION_BARS_MIN': 12,  # minimum bars in compression (3h+)

    # Ignition detection
    'SQUEEZE_IGNITION_VOL_MIN': 5.0,    # vol/MA20 on ignition bar (raised — only big spikes)
    'SQUEEZE_IGNITION_RANGE_MIN': 0.4,  # bar range% minimum
    'SQUEEZE_TAKER_LONG': 0.52,         # taker ratio for long squeeze (lowered)
    'SQUEEZE_TAKER_SHORT': 0.48,

    # Quality thresholds
    'SQUEEZE_QUALITY_MIN': 0.40,        # minimum quality (ignition bar is the confirmation)
    'SQUEEZE_SCORE_THRESHOLD': 0.45,    # composite score threshold

    # Take profit (adaptive)
    'SQUEEZE_TP_PCT': 0.5,              # 0.5% TP (wider than v3's 0.3%)
    'SQUEEZE_SL_ATR_MULT': 1.5,

    # Override
    'SQUEEZE_OVERRIDE_REGIME': True,
    'SQUEEZE_ICS_BOOST': 0.10,
    'SQUEEZE_SIZE_MULT': 0.80,

    # Cooldown
    'SQUEEZE_COOLDOWN_BARS': 32,        # 8h on 15m (avoid re-triggering on continuation)
}


def _compute_compression_score(range48, vol_dryup_count, compression_bars, cfg):
    """Score how compressed the market is. Higher = more squeeze-ready."""
    score = 0.0

    # Range compression: 1.5% = baseline, 0.5% = max score
    range_score = max(0, min(1, (cfg['SQUEEZE_RANGE48_MAX'] - range48) /
                             (cfg['SQUEEZE_RANGE48_MAX'] - 0.4)))
    score += range_score * 0.45

    # Volume dry-up: more bars of low vol = better
    dryup_score = min(1, vol_dryup_count / 12)
    score += dryup_score * 0.35

    # Duration: longer compression = more energy stored
    duration_score = min(1, compression_bars / 32)
    score += duration_score * 0.20

    return score


def _compute_ignition_score(vol_ratio, bar_range_pct, taker_ratio, direction, cfg):
    """Score the ignition bar. Higher = stronger breakout."""
    score = 0.0

    # Volume spike: 3x = baseline, 8x+ = max
    vol_score = min(1, max(0, (vol_ratio - cfg['SQUEEZE_IGNITION_VOL_MIN']) / 5))
    score += vol_score * 0.45

    # Range expansion: 0.4% = baseline, 1.0%+ = max
    range_score = min(1, max(0, (bar_range_pct - cfg['SQUEEZE_IGNITION_RANGE_MIN']) / 0.6))
    score += range_score * 0.30

    # Taker alignment
    if direction == 'LONG' and taker_ratio >= cfg['SQUEEZE_TAKER_LONG']:
        taker_score = min(1, (taker_ratio - cfg['SQUEEZE_TAKER_LONG']) / 0.15)
        score += taker_score * 0.25
    elif direction == 'SHORT' and taker_ratio <= cfg['SQUEEZE_TAKER_SHORT']:
        taker_score = min(1, (cfg['SQUEEZE_TAKER_SHORT'] - taker_ratio) / 0.15)
        score += taker_score * 0.25
    else:
        score += 0.05  # small credit for any ignition

    return score


def detect_squeeze_v4(result, config=None, last_signal_bar=-1, current_bar=0,
                       compression_history=None):
    """Detect squeeze with multi-bar state machine.

    Args:
        result: scan_signal() output dict
        config: Optional config overrides
        last_signal_bar: Bar index of last squeeze signal (cooldown)
        current_bar: Current bar index
        compression_history: list of recent (range48, vol_ratio, bar_range_pct, taker_ratio)
                            tuples from previous bars. If None, uses single-bar detection.

    Returns:
        dict with squeeze_type, score, direction, tp/sl levels
    """
    cfg = {**SQUEEZE_V4_DEFAULTS, **(config or {})}

    # Cooldown
    if current_bar - last_signal_bar < cfg['SQUEEZE_COOLDOWN_BARS']:
        return _empty_result('cooldown')

    # Extract current bar features
    price = result.get('price', 0)
    atr = result.get('atr', 0)
    range48 = result.get('range_width', 5)  # this is range48 from scanner
    vol_ratio = result.get('vol_trend', 1.0)  # current bar vol/MA20
    bar_range_pct = result.get('raw_bar_range_pct', 0.5)  # actual bar range %
    taker_ratio = result.get('raw_taker_ratio', 0.5)  # actual taker ratio

    # ── Phase 1: Check compression state ──
    compression_bars = 0
    vol_dryup_count = 0

    if compression_history and len(compression_history) >= 4:
        # Count compression bars by scanning FORWARD through history
        # Find the longest recent compression streak ending at current bar
        # A "compression streak" = consecutive bars with range48 < threshold
        # (allowing gaps of up to 2 bars above threshold)
        best_streak = 0
        current_streak = 0
        gap_count = 0
        for r48, vr, br, tr in compression_history:
            if r48 < cfg['SQUEEZE_RANGE48_MAX']:
                current_streak += 1
                gap_count = 0
            elif gap_count < 2:
                # Allow small gaps within compression
                gap_count += 1
                current_streak += 1
            else:
                # Compression broken — reset
                best_streak = max(best_streak, current_streak)
                current_streak = 0
                gap_count = 0
            if vr < 0.6:
                vol_dryup_count += 1
        best_streak = max(best_streak, current_streak)
        compression_bars = best_streak

        # Current bar
        if range48 < cfg['SQUEEZE_RANGE48_MAX']:
            compression_bars += 1
        if vol_ratio < 0.6:
            vol_dryup_count += 1
    else:
        # Fallback: estimate from current bar only
        if range48 < cfg['SQUEEZE_RANGE48_MAX']:
            compression_bars = 1  # can't tell duration
        if vol_ratio < 0.6:
            vol_dryup_count = 1

    # Gate 1: Must be (or recently was) in compression
    # The ignition bar itself will inflate range48, so check if we were recently compressed
    # Use the minimum range48 from recent compression history
    recent_range48 = range48
    if compression_history:
        recent_bars = [r[0] for r in compression_history[-12:]]  # last 3 bars
        if recent_bars:
            recent_range48 = min(recent_bars)

    if recent_range48 >= cfg['SQUEEZE_RANGE48_MAX']:
        return _empty_result(f'range48={range48:.2f}% >= {cfg["SQUEEZE_RANGE48_MAX"]}%')

    # Gate 2: Need minimum compression duration
    if compression_bars < cfg['SQUEEZE_COMPRESSION_BARS_MIN']:
        return _empty_result(f'compression_bars={compression_bars} < {cfg["SQUEEZE_COMPRESSION_BARS_MIN"]}')

    # ── Phase 2: Check ignition ──
    current_range_pct = bar_range_pct
    current_taker = taker_ratio

    # Gate 3: Volume spike on current bar
    if vol_ratio < cfg['SQUEEZE_IGNITION_VOL_MIN']:
        return _empty_result(f'vol={vol_ratio:.2f}x < {cfg["SQUEEZE_IGNITION_VOL_MIN"]}x')

    # ── Determine direction ──
    # Primary: taker ratio (slight lean is enough — the vol spike IS the confirmation)
    if taker_ratio > 0.50:
        direction = 'LONG'
        squeeze_type = 'SHORT_SQUEEZE'
    elif taker_ratio < 0.50:
        direction = 'SHORT'
        squeeze_type = 'LONG_SQUEEZE'
    else:
        # Exactly neutral — use price action (close vs open of ignition bar)
        # If price went up, it's a short squeeze
        price_up = result.get('price', 0) > result.get('price', 0)  # can't tell from result alone
        # Default to LONG (short squeezes more common after drops)
        direction = 'LONG'
        squeeze_type = 'SHORT_SQUEEZE'

    # Gate 4: Taker must at least slightly align (no strong opposition)
    # If direction=LONG but taker < 0.45, the bar is selling into the move — not a squeeze
    if direction == 'LONG' and taker_ratio < 0.42:
        return _empty_result(f'taker={taker_ratio:.3f} selling into LONG move')
    if direction == 'SHORT' and taker_ratio > 0.58:
        return _empty_result(f'taker={taker_ratio:.3f} buying into SHORT move')

    # ── Compute scores ──
    compression_score = _compute_compression_score(
        range48, vol_dryup_count, compression_bars, cfg)

    ignition_score = _compute_ignition_score(
        vol_ratio, current_range_pct, current_taker, direction, cfg)

    # Quality = compression * 0.55 + ignition * 0.45
    quality = compression_score * 0.55 + ignition_score * 0.45

    if quality < cfg['SQUEEZE_QUALITY_MIN']:
        return _empty_result(f'quality={quality:.2f} < {cfg["SQUEEZE_QUALITY_MIN"]}')

    # Composite score
    score = quality
    score = min(score, 1.0)

    # Strong if quality > 0.70 and ignition vol > 5x
    is_strong = quality >= 0.70 and vol_ratio >= 5.0

    # ── TP/SL levels ──
    tp_pct = cfg['SQUEEZE_TP_PCT']
    sl_pct = abs(atr * cfg['SQUEEZE_SL_ATR_MULT'] / price * 100) if price > 0 and atr > 0 else 0.5

    if direction == 'LONG':
        tp = price * (1 + tp_pct / 100)
        sl = price * (1 - sl_pct / 100)
    else:
        tp = price * (1 - tp_pct / 100)
        sl = price * (1 + sl_pct / 100)

    # ── Build factors ──
    factors = [
        f'range48={range48:.2f}% (compressed)',
        f'compression={compression_bars} bars',
        f'vol_dryup={vol_dryup_count}/24 bars',
        f'ignition vol={vol_ratio:.1f}x MA20',
    ]
    if is_strong:
        factors.append('STRONG: quality + high vol')

    return {
        'squeeze_type': squeeze_type,
        'squeeze_score': round(score, 3),
        'squeeze_strong': is_strong,
        'direction': direction,
        'factors': factors,
        'quality': round(quality, 3),
        'compression_score': round(compression_score, 3),
        'ignition_score': round(ignition_score, 3),
        'compression_bars': compression_bars,
        'vol_dryup_count': vol_dryup_count,
        'overrides_regime': cfg['SQUEEZE_OVERRIDE_REGIME'],
        'ics_boost': cfg['SQUEEZE_ICS_BOOST'],
        'size_mult': cfg['SQUEEZE_SIZE_MULT'],
        'tp': round(tp, 2),
        'sl': round(sl, 2),
        'tp_pct': tp_pct,
        'sl_pct': round(sl_pct, 3),
        'gates_all_pass': True,
        'gates_passed': factors,
        'gates_failed': [],
        'short_score': round(score, 3) if squeeze_type == 'SHORT_SQUEEZE' else 0,
        'long_score': round(score, 3) if squeeze_type == 'LONG_SQUEEZE' else 0,
    }


def _empty_result(reason):
    """Return empty squeeze result."""
    return {
        'squeeze_type': 'NONE', 'squeeze_score': 0, 'squeeze_strong': False,
        'direction': 'NEUTRAL', 'factors': [], 'quality': 0,
        'compression_score': 0, 'ignition_score': 0,
        'compression_bars': 0, 'vol_dryup_count': 0,
        'overrides_regime': False, 'ics_boost': 0, 'size_mult': 1.0,
        'tp': 0, 'sl': 0, 'tp_pct': 0, 'sl_pct': 0,
        'gates_all_pass': False, 'gates_passed': [], 'gates_failed': [reason],
        'short_score': 0, 'long_score': 0,
    }


def format_squeeze(sq):
    """Format squeeze result for terminal output."""
    if sq.get('squeeze_type', 'NONE') == 'NONE':
        failed = sq.get('gates_failed', [])
        if failed and failed[0] != 'cooldown':
            lines = ['', '  🔥 SQUEEZE DETECTOR — GATE FAILED']
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
    lines.append(f'  Score: {sq["squeeze_score"]:.3f}  '
                 f'Compression: {sq.get("compression_score", 0):.3f}  '
                 f'Ignition: {sq.get("ignition_score", 0):.3f}')
    lines.append(f'  Direction: {sq["direction"]}')
    lines.append(f'  Compression: {sq.get("compression_bars", 0)} bars, '
                 f'{sq.get("vol_dryup_count", 0)} bars of vol dry-up')

    if sq.get('overrides_regime'):
        lines.append(f'  ⚡ Overrides M9 regime block!')

    if sq.get('factors'):
        lines.append(f'\n  Factors:')
        for f in sq['factors']:
            lines.append(f'    ✅ {f}')

    if sq.get('tp', 0) > 0:
        lines.append(f'\n  TP: ${sq["tp"]:.2f} ({sq["tp_pct"]:.1f}%)  '
                     f'SL: ${sq["sl"]:.2f} ({sq["sl_pct"]:.2f}%)')

    if sq.get('ics_boost', 0) > 0:
        lines.append(f'  ICS boost: +{sq["ics_boost"]:.4f}')

    return '\n'.join(lines)

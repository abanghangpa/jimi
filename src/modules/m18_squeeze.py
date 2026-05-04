"""
M18: Squeeze Detector v4 — Two-phase: Compression + CVD Trigger.

Phase 1 — COMPRESSION: Range narrows (range48 < 1.2%), volume dries up, dojis.
Phase 2 — CVD TRIGGER: Taker ratio spikes (buyers/sellers step in aggressively).
           Only enter when CVD confirms the squeeze is actually firing.

This prevents entering during dead compression (Event 3 pattern)
and catches real squeezes (Events 1 & 2) with better entries.
"""

import numpy as np


SQUEEZE_V4_DEFAULTS = {
    # Phase 1: Compression detection
    'SQUEEZE_RANGE48_MAX': 1.2,         # range48% must be below this
    'SQUEEZE_COMPRESSION_BARS_MIN': 12,  # minimum bars in compression (3h+)
    'SQUEEZE_DRY_BARS_MIN': 4,          # bars of vol < 0.6x in last 12
    'SQUEEZE_DOJI_BARS_MIN': 8,         # bars of range < 0.20% in last 12

    # Phase 2: CVD trigger (taker-based)
    'SQUEEZE_TAKER_LONG': 0.58,         # taker ratio for long entry
    'SQUEEZE_TAKER_SHORT': 0.42,        # taker ratio for short entry

    # Phase 2b: RSI divergence (alternative trigger)
    'SQUEEZE_RSI_RISE_MIN': 15,         # RSI must rise this much during compression
    'SQUEEZE_RSI_WASHOUT': 35,          # OR RSI drops below this (washout)

    # Exit levels
    'SQUEEZE_TP_PCT': 0.5,              # 0.5% TP
    'SQUEEZE_SL_ATR_MULT': 0.3,         # SL buffer below compression low

    # Override
    'SQUEEZE_OVERRIDE_REGIME': True,
    'SQUEEZE_ICS_BOOST': 0.10,
    'SQUEEZE_SIZE_MULT': 0.80,

    # Cooldown
    'SQUEEZE_COOLDOWN_BARS': 32,        # 8h on 15m
}


def _check_compression(range48, compression_history, cfg):
    """Check if market is in compression state.

    Returns: (is_compressed, compression_bars, dry_count, doji_count)
    """
    # Gate 1: Current range48 must be below threshold
    if range48 >= cfg['SQUEEZE_RANGE48_MAX']:
        return False, 0, 0, 0

    # Count compression bars (forward scan, allow 2 gaps)
    compression_bars = 0
    dry_count = 0
    doji_count = 0
    streak = 0
    gap_count = 0

    for r48, vr, br, tr in compression_history:
        if r48 < cfg['SQUEEZE_RANGE48_MAX']:
            streak += 1
            gap_count = 0
        elif gap_count < 2:
            gap_count += 1
            streak += 1
        else:
            compression_bars = max(compression_bars, streak)
            streak = 0
            gap_count = 0
        if vr < 0.6:
            dry_count += 1
        if br < 0.20:
            doji_count += 1

    compression_bars = max(compression_bars, streak)

    # Include current bar
    if range48 < cfg['SQUEEZE_RANGE48_MAX']:
        compression_bars += 1

    is_compressed = (
        compression_bars >= cfg['SQUEEZE_COMPRESSION_BARS_MIN'] and
        dry_count >= cfg['SQUEEZE_DRY_BARS_MIN'] and
        doji_count >= cfg['SQUEEZE_DOJI_BARS_MIN']
    )

    return is_compressed, compression_bars, dry_count, doji_count


def _check_cvd_trigger(taker_ratio, pre_taker_avg, cfg):
    """Check if CVD trigger fires (taker spike during compression).

    Returns: (triggered, direction, trigger_type)
    """
    # Primary: taker ratio spike
    if taker_ratio >= cfg['SQUEEZE_TAKER_LONG']:
        return True, 'LONG', 'TAKER_SPIKE'
    if taker_ratio <= cfg['SQUEEZE_TAKER_SHORT']:
        return True, 'SHORT', 'TAKER_SPIKE'

    # Secondary: taker shift (was neutral, now leaning)
    if pre_taker_avg is not None:
        if taker_ratio > 0.52 and pre_taker_avg < 0.48:
            return True, 'LONG', 'TAKER_SHIFT'
        if taker_ratio < 0.48 and pre_taker_avg > 0.52:
            return True, 'SHORT', 'TAKER_SHIFT'

    return False, 'NEUTRAL', 'NONE'


def _check_rsi_trigger(rsi, pre_rsi_values, cfg):
    """Check if RSI divergence fires (alternative to CVD trigger).

    Returns: (triggered, direction, trigger_type)
    """
    if pre_rsi_values is None or len(pre_rsi_values) < 2:
        return False, 'NEUTRAL', 'NONE'

    rsi_rise = rsi - pre_rsi_values[0]
    washout = rsi < cfg['SQUEEZE_RSI_WASHOUT']

    if rsi_rise >= cfg['SQUEEZE_RSI_RISE_MIN']:
        return True, 'LONG', f'RSI_RISE({rsi_rise:.0f})'
    if washout:
        return True, 'LONG', f'RSI_WASHOUT({rsi:.0f})'

    return False, 'NEUTRAL', 'NONE'


def detect_squeeze_v4(result, config=None, last_signal_bar=-1, current_bar=0,
                       compression_history=None):
    """Two-phase squeeze detection: compression + CVD trigger.

    Args:
        result: scan_signal() output dict with:
            - price, atr, range_width, vol_trend
            - raw_bar_range_pct, raw_taker_ratio
        config: Optional config overrides
        last_signal_bar: Bar index of last squeeze signal (cooldown)
        current_bar: Current bar index
        compression_history: list of (range48, vol_ratio, bar_range_pct, taker_ratio)

    Returns:
        dict with squeeze_type, score, direction, tp/sl levels
    """
    cfg = {**SQUEEZE_V4_DEFAULTS, **(config or {})}

    # Cooldown
    if current_bar - last_signal_bar < cfg['SQUEEZE_COOLDOWN_BARS']:
        return _empty_result('cooldown')

    # Extract features
    price = result.get('price', 0)
    atr = result.get('atr', 0)
    range48 = result.get('range_width', 5)
    vol_ratio = result.get('vol_trend', 1.0)
    taker_ratio = result.get('raw_taker_ratio', 0.5)
    rsi = result.get('rsi', 50)

    # ═══════════════════════════════════════════
    # PHASE 1: Compression Detection
    # ═══════════════════════════════════════════
    is_compressed, comp_bars, dry_count, doji_count = _check_compression(
        range48, compression_history or [], cfg)

    if not is_compressed:
        reasons = []
        if range48 >= cfg['SQUEEZE_RANGE48_MAX']:
            reasons.append(f'range48={range48:.2f}% >= {cfg["SQUEEZE_RANGE48_MAX"]}%')
        if comp_bars < cfg['SQUEEZE_COMPRESSION_BARS_MIN']:
            reasons.append(f'comp={comp_bars} < {cfg["SQUEEZE_COMPRESSION_BARS_MIN"]}')
        if dry_count < cfg['SQUEEZE_DRY_BARS_MIN']:
            reasons.append(f'dry={dry_count} < {cfg["SQUEEZE_DRY_BARS_MIN"]}')
        if doji_count < cfg['SQUEEZE_DOJI_BARS_MIN']:
            reasons.append(f'doji={doji_count} < {cfg["SQUEEZE_DOJI_BARS_MIN"]}')
        return _empty_result('; '.join(reasons))

    # ═══════════════════════════════════════════
    # PHASE 2: CVD / RSI Trigger
    # ═══════════════════════════════════════════
    # Compute pre-taker average from compression history
    pre_taker_avg = None
    pre_rsi_values = None
    if compression_history and len(compression_history) >= 4:
        pre_takers = [tr for _, _, _, tr in compression_history[-12:]]
        pre_taker_avg = np.mean(pre_takers) if pre_takers else None

    # Primary trigger: CVD (taker-based)
    cvd_triggered, cvd_direction, cvd_type = _check_cvd_trigger(
        taker_ratio, pre_taker_avg, cfg)

    # Secondary trigger: RSI divergence
    rsi_triggered, rsi_direction, rsi_type = _check_rsi_trigger(
        rsi, pre_rsi_values, cfg)

    # Need at least one trigger
    if cvd_triggered:
        direction = cvd_direction
        trigger_type = cvd_type
    elif rsi_triggered:
        direction = rsi_direction
        trigger_type = rsi_type
    else:
        return _empty_result(f'taker={taker_ratio:.3f} no CVD/RSI trigger')

    # ═══════════════════════════════════════════
    # Score & Levels
    # ═══════════════════════════════════════════
    # Quality = compression depth + trigger strength
    comp_score = 0.0
    if range48 < 0.8:
        comp_score += 0.30  # ultra-tight
    elif range48 < 1.0:
        comp_score += 0.20
    else:
        comp_score += 0.10

    comp_score += min(0.30, dry_count / 12 * 0.30)
    comp_score += min(0.20, doji_count / 12 * 0.20)
    comp_score += min(0.20, comp_bars / 48 * 0.20)

    # Trigger score
    if cvd_triggered:
        if taker_ratio >= 0.65 or taker_ratio <= 0.35:
            trigger_score = 0.40  # extreme taker
        elif taker_ratio >= 0.60 or taker_ratio <= 0.40:
            trigger_score = 0.30
        else:
            trigger_score = 0.20
    else:
        trigger_score = 0.15  # RSI trigger is weaker

    quality = comp_score + trigger_score
    quality = min(quality, 1.0)

    # Squeeze type
    squeeze_type = 'SHORT_SQUEEZE' if direction == 'LONG' else 'LONG_SQUEEZE'
    is_strong = quality >= 0.70 and (taker_ratio >= 0.65 or taker_ratio <= 0.35)

    # TP/SL
    tp_pct = cfg['SQUEEZE_TP_PCT']
    if direction == 'LONG':
        # SL at compression range low + buffer
        if compression_history:
            comp_lows = [r48 for r48, _, _, _ in compression_history
                         if r48 < cfg['SQUEEZE_RANGE48_MAX']]
            # Use the range48 values to estimate compression low
            # Actually we need price lows, not range48
            # For now, use ATR-based SL with compression buffer
        sl_pct = abs(atr * cfg['SQUEEZE_SL_ATR_MULT'] / price * 100) if price > 0 and atr > 0 else 0.5
        tp = price * (1 + tp_pct / 100)
        sl = price * (1 - sl_pct / 100)
    else:
        sl_pct = abs(atr * cfg['SQUEEZE_SL_ATR_MULT'] / price * 100) if price > 0 and atr > 0 else 0.5
        tp = price * (1 - tp_pct / 100)
        sl = price * (1 + sl_pct / 100)

    # Factors
    factors = [
        f'comp={comp_bars} bars, dry={dry_count}, doji={doji_count}',
        f'range48={range48:.2f}%',
        f'trigger={trigger_type}',
        f'taker={taker_ratio:.3f}',
    ]
    if is_strong:
        factors.append('STRONG: extreme taker + deep compression')

    return {
        'squeeze_type': squeeze_type,
        'squeeze_score': round(quality, 3),
        'squeeze_strong': is_strong,
        'direction': direction,
        'factors': factors,
        'quality': round(quality, 3),
        'compression_bars': comp_bars,
        'dry_count': dry_count,
        'doji_count': doji_count,
        'trigger_type': trigger_type,
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
        'short_score': round(quality, 3) if squeeze_type == 'SHORT_SQUEEZE' else 0,
        'long_score': round(quality, 3) if squeeze_type == 'LONG_SQUEEZE' else 0,
    }


def _empty_result(reason):
    """Return empty squeeze result."""
    return {
        'squeeze_type': 'NONE', 'squeeze_score': 0, 'squeeze_strong': False,
        'direction': 'NEUTRAL', 'factors': [], 'quality': 0,
        'compression_bars': 0, 'dry_count': 0, 'doji_count': 0,
        'trigger_type': 'NONE',
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
                 f'Compression: {sq["compression_bars"]} bars  '
                 f'Trigger: {sq.get("trigger_type", "?")}')
    lines.append(f'  Direction: {sq["direction"]}')
    lines.append(f'  Dry bars: {sq.get("dry_count", 0)}  '
                 f'Doji bars: {sq.get("doji_count", 0)}')

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

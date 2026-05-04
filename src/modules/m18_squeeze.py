"""
M18: Squeeze Detector v5 — Dual-path: 15m Compression + 2h MACD Coiling.

Path A (existing): 15m range compression → CVD taker trigger
Path B (new):      2h MACD(8,17,9) DIF/DEA convergence → histogram flip trigger

Either path can fire independently. When both agree, it's a STRONG signal.

Calibrated against 3 historical samples:
  - May 2-4 2026:   20h coil → $60+ move (biggest single bar $23.99)
  - Apr 24-26 2026: 36h coil → $50+ move over 24h
  - Apr 3-5 2026:   24h coil → $50+ move (biggest single bar $16.16)

Success definition: post-squeeze move produces 15m bars with |O-C| >= $25
  (typically 2-4 bars into the move, not the first bar)
"""

import numpy as np


SQUEEZE_V5_DEFAULTS = {
    # ── Path A: 15m compression (from v4) ──
    'SQUEEZE_RANGE48_MAX': 1.2,
    'SQUEEZE_COMPRESSION_BARS_MIN': 12,
    'SQUEEZE_DRY_BARS_MIN': 4,
    'SQUEEZE_DOJI_BARS_MIN': 8,

    # Path A trigger: CVD taker
    'SQUEEZE_TAKER_LONG': 0.58,
    'SQUEEZE_TAKER_SHORT': 0.42,

    # ── Path B: 2h MACD coiling ──
    'SQUEEZE_MACD_FAST': 8,
    'SQUEEZE_MACD_SLOW': 17,
    'SQUEEZE_MACD_SIGNAL': 9,
    'SQUEEZE_COIL_DELTA_MAX': 0.05,     # DIF-DEA delta % for "coiled"
    'SQUEEZE_COIL_BARS_MIN': 6,         # minimum 2h bars in coil (12h)
    'SQUEEZE_COIL_BARS_STRONG': 9,      # strong coil (18h+)
    'SQUEEZE_HIST_FLIP_THRESHOLD': 0.0, # histogram crosses zero

    # ── Exit levels ──
    'SQUEEZE_TP_PCT': 0.5,
    'SQUEEZE_SL_ATR_MULT': 0.3,

    # ── Override ──
    'SQUEEZE_OVERRIDE_REGIME': True,
    'SQUEEZE_ICS_BOOST': 0.10,
    'SQUEEZE_SIZE_MULT': 0.80,

    # ── Cooldown ──
    'SQUEEZE_COOLDOWN_BARS': 32,        # 8h on 15m
}


def _check_compression(range48, compression_history, cfg):
    """Check if market is in 15m compression state (Path A)."""
    if range48 >= cfg['SQUEEZE_RANGE48_MAX']:
        return False, 0, 0, 0

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

    if range48 < cfg['SQUEEZE_RANGE48_MAX']:
        compression_bars += 1

    is_compressed = (
        compression_bars >= cfg['SQUEEZE_COMPRESSION_BARS_MIN'] and
        dry_count >= cfg['SQUEEZE_DRY_BARS_MIN'] and
        doji_count >= cfg['SQUEEZE_DOJI_BARS_MIN']
    )

    return is_compressed, compression_bars, dry_count, doji_count


def _compute_2h_macd(df_15m, cfg):
    """Resample 15m data to 2h and compute MACD(8,17,9).

    Returns:
        dict with dif, dea, hist arrays + timestamps, or None if insufficient data.
    """
    if len(df_15m) < 40:  # need at least ~10 2h bars
        return None

    # Resample to 2h
    df_copy = df_15m.copy()
    if 'Open time' in df_copy.columns:
        df_copy = df_copy.set_index('Open time')

    agg_dict = {
        'Open': 'first', 'High': 'max', 'Low': 'min', 'Close': 'last',
        'Volume': 'sum',
    }
    if 'Taker buy base asset volume' in df_copy.columns:
        agg_dict['Taker buy base asset volume'] = 'sum'

    df_2h = df_copy.resample('2h').agg(agg_dict).dropna(subset=['Open'])

    if len(df_2h) < cfg['SQUEEZE_MACD_SLOW'] + cfg['SQUEEZE_MACD_SIGNAL']:
        return None

    close = df_2h['Close']
    ema_fast = close.ewm(span=cfg['SQUEEZE_MACD_FAST'], adjust=False).mean()
    ema_slow = close.ewm(span=cfg['SQUEEZE_MACD_SLOW'], adjust=False).mean()
    dif = ema_fast - ema_slow
    dea = dif.ewm(span=cfg['SQUEEZE_MACD_SIGNAL'], adjust=False).mean()
    hist = dif - dea

    return {
        'close': close.values,
        'dif': dif.values,
        'dea': dea.values,
        'hist': hist.values,
        'timestamps': df_2h.index,
        'prices': close.values,
    }


def _detect_macd_coil(macd_data, cfg):
    """Detect 2h MACD DIF/DEA coiling (Path B).

    A "coil" = DIF and DEA converged within <COIL_DELTA_MAX% for N+ bars.
    The squeeze fires when histogram flips zero after a coil period.

    Returns:
        (is_coiled, coil_bars, last_coil_bar, hist_flip, direction, details)
    """
    dif = macd_data['dif']
    dea = macd_data['dea']
    hist = macd_data['hist']
    close = macd_data['close']
    ts = macd_data['timestamps']

    n = len(dif)
    if n < 3:
        return False, 0, -1, False, 'NEUTRAL', {}

    # Compute DIF-DEA delta as % of price
    delta_pct = np.abs(dif - dea) / np.where(close > 0, close, 1.0) * 100

    # Scan for coiling: consecutive bars with delta < threshold
    # Allow 2 non-consecutive gaps (same as 15m compression logic)
    coil_threshold = cfg['SQUEEZE_COIL_DELTA_MAX']
    min_bars = cfg['SQUEEZE_COIL_BARS_MIN']

    # Find the most recent coil streak (working backwards from current bar)
    coil_bars = 0
    gap_count = 0
    max_gaps = 2

    for i in range(n - 1, -1, -1):
        if delta_pct[i] < coil_threshold:
            coil_bars += 1
            gap_count = 0
        elif gap_count < max_gaps:
            gap_count += 1
            coil_bars += 1
        else:
            break

    # Also find longest coil streak anywhere in the data
    max_streak = 0
    streak = 0
    for i in range(n):
        if delta_pct[i] < coil_threshold:
            streak += 1
            max_streak = max(max_streak, streak)
        else:
            streak = 0

    is_coiled = coil_bars >= min_bars

    # Detect histogram flip: check if histogram changed sign in last 2 bars
    hist_flip = False
    flip_direction = 'NEUTRAL'

    if n >= 2:
        prev_hist = hist[-2]
        curr_hist = hist[-1]
        # Flip: crossed zero
        if prev_hist < 0 and curr_hist >= 0:
            hist_flip = True
            flip_direction = 'LONG'
        elif prev_hist > 0 and curr_hist <= 0:
            hist_flip = True
            flip_direction = 'SHORT'

    # Also check: histogram was near zero and now expanding
    hist_expanding = False
    if n >= 3:
        prev_abs = abs(hist[-2])
        curr_abs = abs(hist[-1])
        if prev_abs < 0.5 and curr_abs > 1.0:
            hist_expanding = True

    # Direction from DIF slope during coil
    direction = 'NEUTRAL'
    if coil_bars >= 2:
        dif_slope = dif[-1] - dif[-(min(coil_bars, 6))]
        if dif_slope > 0.5:
            direction = 'LONG'
        elif dif_slope < -0.5:
            direction = 'SHORT'

    # Override direction if histogram flipped
    if hist_flip:
        direction = flip_direction

    details = {
        'coil_bars': coil_bars,
        'max_streak': max_streak,
        'coil_hours': coil_bars * 2,
        'max_streak_hours': max_streak * 2,
        'delta_pct_current': round(float(delta_pct[-1]), 4),
        'delta_pct_avg': round(float(np.mean(delta_pct[-coil_bars:])) if coil_bars > 0 else 999, 4),
        'hist_value': round(float(hist[-1]), 3),
        'hist_prev': round(float(hist[-2]) if n >= 2 else 0, 3),
        'hist_flip': hist_flip,
        'hist_expanding': hist_expanding,
        'flip_direction': flip_direction,
        'dif_current': round(float(dif[-1]), 3),
        'dea_current': round(float(dea[-1]), 3),
        'timestamp': str(ts[-1]) if n > 0 else '',
    }

    return is_coiled, coil_bars, n - 1, hist_flip, direction, details


def _check_cvd_trigger(taker_ratio, pre_taker_avg, cfg):
    """Check if CVD trigger fires (taker spike during compression)."""
    if taker_ratio >= cfg['SQUEEZE_TAKER_LONG']:
        return True, 'LONG', 'TAKER_SPIKE'
    if taker_ratio <= cfg['SQUEEZE_TAKER_SHORT']:
        return True, 'SHORT', 'TAKER_SPIKE'

    if pre_taker_avg is not None:
        if taker_ratio > 0.52 and pre_taker_avg < 0.48:
            return True, 'LONG', 'TAKER_SHIFT'
        if taker_ratio < 0.48 and pre_taker_avg > 0.52:
            return True, 'SHORT', 'TAKER_SHIFT'

    return False, 'NEUTRAL', 'NONE'


def _score_quality(path_a, path_b, cfg):
    """Score squeeze quality based on which paths fired and their strength.

    Returns: (quality, is_strong, factors)
    """
    quality = 0.0
    factors = []
    is_strong = False

    a_compressed, a_comp_bars, a_dry, a_doji = path_a
    b_coiled, b_coil_bars, b_hist_flip, b_direction, b_details = path_b

    # ── Path A scoring (15m compression) ──
    if a_compressed:
        a_score = 0.0
        if a_comp_bars >= 24:
            a_score += 0.25
            factors.append(f'15m deep compression: {a_comp_bars} bars')
        elif a_comp_bars >= 12:
            a_score += 0.15
            factors.append(f'15m compression: {a_comp_bars} bars')

        a_score += min(0.15, a_dry / 12 * 0.15)
        a_score += min(0.10, a_doji / 12 * 0.10)

        quality += a_score
        factors.append(f'15m dry={a_dry} doji={a_doji}')
    else:
        # Partial credit if close to threshold
        if a_comp_bars >= 8:
            quality += 0.05
            factors.append(f'15m near-compression: {a_comp_bars} bars (need {cfg["SQUEEZE_COMPRESSION_BARS_MIN"]})')

    # ── Path B scoring (2h MACD coil) ──
    if b_coiled:
        b_score = 0.0
        coil_hours = b_details.get('coil_hours', 0)
        max_hours = b_details.get('max_streak_hours', 0)

        # Duration scoring: longer coil = higher quality
        if coil_hours >= 18:
            b_score += 0.30
            factors.append(f'2h MACD coil: {coil_hours}h (STRONG)')
        elif coil_hours >= 12:
            b_score += 0.20
            factors.append(f'2h MACD coil: {coil_hours}h')
        else:
            b_score += 0.10
            factors.append(f'2h MACD coil: {coil_hours}h (short)')

        # Max streak bonus (even if current streak is shorter)
        if max_hours >= 24:
            b_score += 0.10
            factors.append(f'Max coil streak: {max_hours}h (extended)')

        # Histogram flip bonus
        if b_hist_flip:
            b_score += 0.20
            factors.append(f'2h histogram flip → {b_direction}')

        # Histogram expanding bonus
        if b_details.get('hist_expanding'):
            b_score += 0.10
            factors.append('2h histogram expanding')

        quality += b_score
    else:
        # Partial credit if some coiling detected
        if b_coil_bars >= 3:
            quality += 0.05
            factors.append(f'2h partial coil: {b_coil_bars} bars (need {cfg["SQUEEZE_COIL_BARS_MIN"]})')

    # ── Dual-path agreement bonus ──
    if a_compressed and b_coiled:
        quality += 0.15
        factors.append('DUAL PATH: 15m + 2h both compressed')
        if b_hist_flip:
            is_strong = True
            factors.append('STRONG: dual compression + histogram flip')

    # ── Determine strength ──
    if quality >= 0.70:
        is_strong = True

    quality = min(quality, 1.0)

    return quality, is_strong, factors


def detect_squeeze_v5(result, config=None, last_signal_bar=-1, current_bar=0,
                       compression_history=None, df_15m=None):
    """Dual-path squeeze detection: 15m compression + 2h MACD coiling.

    Args:
        result: scan_signal() output dict
        config: Optional config overrides
        last_signal_bar: Bar index of last squeeze signal (cooldown)
        current_bar: Current bar index
        compression_history: list of (range48, vol_ratio, bar_range_pct, taker_ratio)
        df_15m: Full 15m DataFrame for 2h MACD computation

    Returns:
        dict with squeeze_type, score, direction, paths fired, tp/sl levels
    """
    cfg = {**SQUEEZE_V5_DEFAULTS, **(config or {})}

    # Cooldown
    if current_bar - last_signal_bar < cfg['SQUEEZE_COOLDOWN_BARS']:
        return _empty_result('cooldown')

    # Extract features
    price = result.get('price', 0)
    atr = result.get('atr', 0)
    range48 = result.get('range_width', 5)
    taker_ratio = result.get('raw_taker_ratio', 0.5)

    # ═══════════════════════════════════════════
    # PATH A: 15m Compression Detection
    # ═══════════════════════════════════════════
    a_compressed, a_comp_bars, a_dry, a_doji = _check_compression(
        range48, compression_history or [], cfg)

    # ═══════════════════════════════════════════
    # PATH B: 2h MACD Coiling Detection
    # ═══════════════════════════════════════════
    b_coiled = False
    b_coil_bars = 0
    b_hist_flip = False
    b_direction = 'NEUTRAL'
    b_details = {}

    if df_15m is not None:
        macd_data = _compute_2h_macd(df_15m, cfg)
        if macd_data is not None:
            b_coiled, b_coil_bars, _, b_hist_flip, b_direction, b_details = \
                _detect_macd_coil(macd_data, cfg)

    # ═══════════════════════════════════════════
    # Determine if either path triggered
    # ═══════════════════════════════════════════
    path_a_fired = a_compressed
    path_b_fired = b_coiled and b_hist_flip  # need both coil + flip

    if not path_a_fired and not path_b_fired:
        reasons = []
        if not a_compressed:
            reasons.append(f'15m: range48={range48:.2f}% comp={a_comp_bars}')
        if not b_coiled:
            reasons.append(f'2h: coil={b_coil_bars} bars')
        elif not b_hist_flip:
            reasons.append(f'2h: coiled {b_coil_bars} bars but no histogram flip')
        return _empty_result('; '.join(reasons))

    # ═══════════════════════════════════════════
    # CVD Trigger (for Path A)
    # ═══════════════════════════════════════════
    pre_taker_avg = None
    if compression_history and len(compression_history) >= 4:
        pre_takers = [tr for _, _, _, tr in compression_history[-12:]]
        pre_taker_avg = np.mean(pre_takers) if pre_takers else None

    cvd_triggered, cvd_direction, cvd_type = _check_cvd_trigger(
        taker_ratio, pre_taker_avg, cfg)

    # ═══════════════════════════════════════════
    # Resolve Direction
    # ═══════════════════════════════════════════
    direction = 'NEUTRAL'
    trigger_type = 'NONE'

    if path_b_fired:
        # Path B direction from histogram flip (most reliable)
        direction = b_direction
        trigger_type = 'HIST_FLIP'
    elif path_a_fired and cvd_triggered:
        # Path A direction from CVD
        direction = cvd_direction
        trigger_type = cvd_type

    if direction == 'NEUTRAL':
        return _empty_result(f'no direction: cvd={cvd_type} hist_flip={b_hist_flip}')

    # ═══════════════════════════════════════════
    # Score & Levels
    # ═══════════════════════════════════════════
    quality, is_strong, factors = _score_quality(
        (a_compressed, a_comp_bars, a_dry, a_doji),
        (b_coiled, b_coil_bars, b_hist_flip, b_direction, b_details),
        cfg
    )

    # Add trigger info to factors
    factors.append(f'Trigger: {trigger_type} → {direction}')
    if cvd_triggered and path_b_fired:
        factors.append(f'CVD confirms: {cvd_direction}')

    # Squeeze type
    squeeze_type = 'SHORT_SQUEEZE' if direction == 'LONG' else 'LONG_SQUEEZE'

    # TP/SL
    tp_pct = cfg['SQUEEZE_TP_PCT']
    sl_pct = abs(atr * cfg['SQUEEZE_SL_ATR_MULT'] / price * 100) if price > 0 and atr > 0 else 0.5
    if direction == 'LONG':
        tp = price * (1 + tp_pct / 100)
        sl = price * (1 - sl_pct / 100)
    else:
        tp = price * (1 - tp_pct / 100)
        sl = price * (1 + sl_pct / 100)

    return {
        'squeeze_type': squeeze_type,
        'squeeze_score': round(quality, 3),
        'squeeze_strong': is_strong,
        'direction': direction,
        'factors': factors,
        'quality': round(quality, 3),
        'compression_bars': a_comp_bars,
        'dry_count': a_dry,
        'doji_count': a_doji,
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
        # Path details for diagnostics
        'path_a': {
            'fired': path_a_fired,
            'compressed': a_compressed,
            'comp_bars': a_comp_bars,
            'dry': a_dry,
            'doji': a_doji,
        },
        'path_b': {
            'fired': path_b_fired,
            'coiled': b_coiled,
            'coil_bars': b_coil_bars,
            'coil_hours': b_details.get('coil_hours', 0),
            'max_streak_hours': b_details.get('max_streak_hours', 0),
            'hist_flip': b_hist_flip,
            'hist_value': b_details.get('hist_value', 0),
            'delta_pct': b_details.get('delta_pct_current', 0),
            'direction': b_direction,
            'dif': b_details.get('dif_current', 0),
            'dea': b_details.get('dea_current', 0),
        },
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
        'path_a': {'fired': False}, 'path_b': {'fired': False},
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
    lines.append('  🔥 SQUEEZE DETECTOR v5')

    icons = {'SHORT_SQUEEZE': '🟩', 'LONG_SQUEEZE': '🟥'}
    icon = icons.get(sq['squeeze_type'], '❓')
    strength = 'STRONG' if sq.get('squeeze_strong') else 'CONFIRMED'

    lines.append(f'  Type: {icon} {sq["squeeze_type"]}  ({strength})')
    lines.append(f'  Score: {sq["squeeze_score"]:.3f}  '
                 f'Trigger: {sq.get("trigger_type", "?")}')

    # Path details
    pa = sq.get('path_a', {})
    pb = sq.get('path_b', {})

    if pa.get('fired'):
        lines.append(f'  Path A (15m): ✅ compressed {pa.get("comp_bars", 0)} bars  '
                     f'dry={pa.get("dry", 0)}  doji={pa.get("doji", 0)}')
    else:
        lines.append(f'  Path A (15m): ❌ comp={pa.get("comp_bars", 0)} bars')

    if pb.get('coiled') or pb.get('fired'):
        coil_h = pb.get('coil_hours', 0)
        max_h = pb.get('max_streak_hours', 0)
        flip = '✅ FLIP' if pb.get('hist_flip') else '⏳ waiting'
        lines.append(f'  Path B (2h):  ✅ coiled {coil_h}h (max {max_h}h)  '
                     f'hist={pb.get("hist_value", 0):.3f}  {flip}')
        lines.append(f'    MACD(8,17,9): DIF={pb.get("dif", 0):.3f}  DEA={pb.get("dea", 0):.3f}  '
                     f'Δ={pb.get("delta_pct", 0):.4f}%')
    else:
        lines.append(f'  Path B (2h):  ❌ coil={pb.get("coil_bars", 0)} bars')

    lines.append(f'  Direction: {sq["direction"]}')

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

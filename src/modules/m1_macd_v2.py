"""M1 v2: Swing-based RSI divergence + ATR-normalized momentum + fast MACD crossover."""

import numpy as np
import pandas as pd


def _find_swing_highs(highs, lookback=5):
    """Find swing highs: bar where high > all neighbors within lookback."""
    swings = np.zeros(len(highs), dtype=bool)
    for i in range(lookback, len(highs) - lookback):
        window = highs[i - lookback: i + lookback + 1]
        if highs[i] == np.max(window):
            swings[i] = True
    return swings


def _find_swing_lows(lows, lookback=5):
    """Find swing lows: bar where low < all neighbors within lookback."""
    swings = np.zeros(len(lows), dtype=bool)
    for i in range(lookback, len(lows) - lookback):
        window = lows[i - lookback: i + lookback + 1]
        if lows[i] == np.min(window):
            swings[i] = True
    return swings


def _detect_rsi_divergence(close, rsi, lookback=40, min_gap=5):
    """Detect real swing-based RSI divergence over a lookback window.

    Returns:
        bull_div: True if bullish divergence found (price lower low, RSI higher low)
        bear_div: True if bearish divergence found (price higher high, RSI lower high)
    """
    if len(close) < lookback:
        return False, False

    seg_close = close[-lookback:]
    seg_rsi = rsi[-lookback:]

    # Find local minima and maxima in price
    price_lows_idx = []
    price_highs_idx = []
    for i in range(2, len(seg_close) - 2):
        if seg_close[i] < seg_close[i-1] and seg_close[i] < seg_close[i-2] and \
           seg_close[i] < seg_close[i+1] and seg_close[i] < seg_close[i+2]:
            price_lows_idx.append(i)
        if seg_close[i] > seg_close[i-1] and seg_close[i] > seg_close[i-2] and \
           seg_close[i] > seg_close[i+1] and seg_close[i] > seg_close[i+2]:
            price_highs_idx.append(i)

    # Bullish divergence: price makes lower low, RSI makes higher low
    bull_div = False
    if len(price_lows_idx) >= 2:
        for i in range(len(price_lows_idx) - 1):
            for j in range(i + 1, len(price_lows_idx)):
                if price_lows_idx[j] - price_lows_idx[i] < min_gap:
                    continue
                p1, p2 = price_lows_idx[i], price_lows_idx[j]
                if seg_close[p2] < seg_close[p1] and seg_rsi[p2] > seg_rsi[p1]:
                    if seg_rsi[p1] < 45:  # must be in lower RSI zone
                        bull_div = True
                        break
            if bull_div:
                break

    # Bearish divergence: price makes higher high, RSI makes lower high
    bear_div = False
    if len(price_highs_idx) >= 2:
        for i in range(len(price_highs_idx) - 1):
            for j in range(i + 1, len(price_highs_idx)):
                if price_highs_idx[j] - price_highs_idx[i] < min_gap:
                    continue
                p1, p2 = price_highs_idx[i], price_highs_idx[j]
                if seg_close[p2] > seg_close[p1] and seg_rsi[p2] < seg_rsi[p1]:
                    if seg_rsi[p1] > 55:  # must be in upper RSI zone
                        bear_div = True
                        break
            if bear_div:
                break

    return bull_div, bear_div


def _detect_macd_divergence(close, macd_hist, lookback=40, min_gap=5):
    """Detect MACD histogram divergence over a lookback window.

    Uses histogram peaks/troughs (not the MACD line itself) because the
    histogram already captures momentum acceleration/deceleration — the
    same thing traders eyeball when they say "MACD divergence".

    Bullish: price makes lower low, MACD histogram makes higher low
             (selling pressure is fading even though price dropped)
    Bearish: price makes higher high, MACD histogram makes lower high
             (buying momentum is fading even though price rallied)

    Returns:
        bull_div: True if bullish MACD divergence found
        bear_div: True if bearish MACD divergence found
    """
    if len(close) < lookback or len(macd_hist) < lookback:
        return False, False

    seg_close = close[-lookback:]
    seg_hist = macd_hist[-lookback:]

    # Find histogram troughs (local minima in negative territory)
    # and histogram peaks (local maxima in positive territory)
    hist_troughs_idx = []
    hist_peaks_idx = []
    for i in range(2, len(seg_hist) - 2):
        # Trough: lower than neighbors, ideally negative
        if seg_hist[i] < seg_hist[i-1] and seg_hist[i] < seg_hist[i-2] and \
           seg_hist[i] < seg_hist[i+1] and seg_hist[i] < seg_hist[i+2]:
            hist_troughs_idx.append(i)
        # Peak: higher than neighbors, ideally positive
        if seg_hist[i] > seg_hist[i-1] and seg_hist[i] > seg_hist[i-2] and \
           seg_hist[i] > seg_hist[i+1] and seg_hist[i] > seg_hist[i+2]:
            hist_peaks_idx.append(i)

    # Also find price swing points (same as RSI divergence uses)
    price_lows_idx = []
    price_highs_idx = []
    for i in range(2, len(seg_close) - 2):
        if seg_close[i] < seg_close[i-1] and seg_close[i] < seg_close[i-2] and \
           seg_close[i] < seg_close[i+1] and seg_close[i] < seg_close[i+2]:
            price_lows_idx.append(i)
        if seg_close[i] > seg_close[i-1] and seg_close[i] > seg_close[i-2] and \
           seg_close[i] > seg_close[i+1] and seg_close[i] > seg_close[i+2]:
            price_highs_idx.append(i)

    # ── Bullish divergence: price lower low, histogram higher low ──
    # We compare price lows against histogram troughs that are near them
    # (within a few bars) to ensure we're looking at the same swing.
    bull_div = False
    if len(price_lows_idx) >= 2:
        for i in range(len(price_lows_idx) - 1):
            for j in range(i + 1, len(price_lows_idx)):
                p1, p2 = price_lows_idx[i], price_lows_idx[j]
                if p2 - p1 < min_gap:
                    continue
                if seg_close[p2] >= seg_close[p1]:
                    continue  # price didn't make lower low
                # Find nearest histogram trough to each price low (±3 bars)
                h1 = _nearest_hist_extreme(seg_hist, p1, hist_troughs_idx, radius=3, prefer='min')
                h2 = _nearest_hist_extreme(seg_hist, p2, hist_troughs_idx, radius=3, prefer='min')
                if h1 is None or h2 is None:
                    continue
                # Histogram higher low while price lower low = bullish divergence
                if seg_hist[h2] > seg_hist[h1]:
                    bull_div = True
                    break
            if bull_div:
                break

    # ── Bearish divergence: price higher high, histogram lower high ──
    bear_div = False
    if len(price_highs_idx) >= 2:
        for i in range(len(price_highs_idx) - 1):
            for j in range(i + 1, len(price_highs_idx)):
                p1, p2 = price_highs_idx[i], price_highs_idx[j]
                if p2 - p1 < min_gap:
                    continue
                if seg_close[p2] <= seg_close[p1]:
                    continue  # price didn't make higher high
                # Find nearest histogram peak to each price high (±3 bars)
                h1 = _nearest_hist_extreme(seg_hist, p1, hist_peaks_idx, radius=3, prefer='max')
                h2 = _nearest_hist_extreme(seg_hist, p2, hist_peaks_idx, radius=3, prefer='max')
                if h1 is None or h2 is None:
                    continue
                # Histogram lower high while price higher high = bearish divergence
                if seg_hist[h2] < seg_hist[h1]:
                    bear_div = True
                    break
            if bear_div:
                break

    return bull_div, bear_div


def _nearest_hist_extreme(seg_hist, price_idx, extreme_indices, radius=3, prefer='min'):
    """Find the nearest histogram extreme (peak/trough) to a price swing point.

    Args:
        seg_hist: histogram segment
        price_idx: index of the price swing point
        extreme_indices: list of indices where histogram has local extremes
        radius: max bars away to search
        prefer: 'min' for troughs, 'max' for peaks

    Returns:
        Index of nearest extreme, or None if nothing within radius.
    """
    best_idx = None
    best_dist = radius + 1
    for hi in extreme_indices:
        dist = abs(hi - price_idx)
        if dist <= radius and dist < best_dist:
            best_dist = dist
            best_idx = hi
    # Fallback: if no pre-detected extreme nearby, scan raw histogram ±radius
    if best_idx is None:
        start = max(0, price_idx - radius)
        end = min(len(seg_hist), price_idx + radius + 1)
        window = seg_hist[start:end]
        if len(window) == 0:
            return None
        if prefer == 'min':
            best_idx = start + int(np.argmin(window))
        else:
            best_idx = start + int(np.argmax(window))
    return best_idx


def _macd_crossover_score(macd_line, signal_line, lookback=3):
    """Fast MACD crossover detection with confirmation bars.

    Returns:
        direction: 'BULLISH', 'BEARISH', 'NEUTRAL'
        score: 0.5-1.0
        crossed_up: True if bullish crossover in last `lookback` bars
        crossed_down: True if bearish crossover in last `lookback` bars
    """
    n = len(macd_line)
    if n < lookback + 2:
        return 'NEUTRAL', 0.5, False, False

    # Use numpy arrays for positional indexing
    diff = np.array(macd_line) - np.array(signal_line)
    crossed_up = False
    crossed_down = False

    for i in range(n - lookback, n):
        if i < 1:
            continue
        if diff[i] > 0 and diff[i - 1] <= 0:
            crossed_up = True
        if diff[i] < 0 and diff[i - 1] >= 0:
            crossed_down = True

    # Current state
    above = diff[-1] > 0
    momentum = abs(diff[-1]) / (abs(diff[-1]) + abs(np.array(macd_line)[-1]) + 1e-10)

    if crossed_up:
        return 'BULLISH', min(0.7 + momentum * 0.3, 1.0), True, False
    elif crossed_down:
        return 'BEARISH', min(0.7 + momentum * 0.3, 1.0), False, True
    elif above and diff[-1] > diff[-2]:
        return 'BULLISH', 0.6, False, False
    elif not above and diff[-1] < diff[-2]:
        return 'BEARISH', 0.6, False, False
    else:
        return 'NEUTRAL', 0.5, False, False


def _momentum_score(close, atr, lookback=8, accel_bars=3):
    """ATR-normalized momentum with acceleration.

    Returns score 0.0-1.0 where >0.5 is bullish, <0.5 is bearish.
    """
    c = np.asarray(close, dtype=float)
    a = np.asarray(atr, dtype=float)
    n = len(c)

    if n < lookback + accel_bars + 2:
        return 0.5

    roc = (c[-1] - c[-lookback - 1]) / c[-lookback - 1] if c[-lookback - 1] != 0 else 0
    roc_prev = (c[-accel_bars - 1] - c[-lookback - accel_bars - 1]) / c[-lookback - accel_bars - 1] \
        if c[-lookback - accel_bars - 1] != 0 else roc

    accel = roc - roc_prev

    # Normalize by ATR (volatility-adjusted)
    atr_pct = a[-1] / c[-1] if c[-1] > 0 else 0.01
    if atr_pct < 0.001:
        atr_pct = 0.001

    roc_norm = roc / atr_pct

    score = 0.5
    if roc_norm > 1.0 and accel > 0:
        score = 0.85
    elif roc_norm > 1.0 and accel < 0:
        score = 0.70
    elif roc_norm > 0.3:
        score = 0.60
    elif roc_norm < -1.0 and accel < 0:
        score = 0.15
    elif roc_norm < -1.0 and accel > 0:
        score = 0.30
    elif roc_norm < -0.3:
        score = 0.40

    return score


def score_m1_v2(df_1h, idx, config, df_15m=None, idx_15m=None):
    """M1 v2: RSI divergence + MACD divergence + MACD crossover + ATR-normalized momentum.

    Args:
        df_1h: 1H DataFrame with indicators precomputed
        idx: current 1H bar index
        config: settings dict
        df_15m: optional 15m DataFrame for MTF confirmation
        idx_15m: optional current 15m bar index

    Returns:
        direction: 'BULLISH', 'BEARISH', 'NEUTRAL'
        score: 0.5-1.0
        details: dict with sub-signal breakdown
    """
    if idx < 30:
        return 'NEUTRAL', 0.5, {'reason': 'warmup'}

    close = df_1h['Close'].values.astype(float)
    rsi = df_1h['rsi'].values.astype(float)
    atr = df_1h['atr'].values.astype(float)
    macd_line = df_1h['macd_line'].values.astype(float)
    signal_line = df_1h['macd_signal'].values.astype(float)
    macd_hist = df_1h['macd_hist'].values.astype(float)

    # ── Signal 1: RSI Divergence (swing-based) ──
    lookback = config.get('M1_V2_RSI_LOOKBACK', 40)
    rsi_bull_div, rsi_bear_div = _detect_rsi_divergence(
        close[:idx + 1], rsi[:idx + 1], lookback=lookback)

    rsi_div_score = 0.5
    rsi_div_dir = 'NEUTRAL'
    if rsi_bull_div and not rsi_bear_div:
        rsi_div_score = 0.75
        rsi_div_dir = 'BULLISH'
    elif rsi_bear_div and not rsi_bull_div:
        rsi_div_score = 0.25
        rsi_div_dir = 'BEARISH'
    elif rsi_bull_div and rsi_bear_div:
        # Both detected — use RSI zone as tiebreaker
        if rsi[idx] < 45:
            rsi_div_score, rsi_div_dir = 0.65, 'BULLISH'
        elif rsi[idx] > 55:
            rsi_div_score, rsi_div_dir = 0.35, 'BEARISH'

    # ── Signal 2: MACD Histogram Divergence (swing-based) ──
    macd_div_enabled = config.get('M1_MACD_DIV_ENABLED', True)
    macd_bull_div = False
    macd_bear_div = False
    macd_div_score = 0.5
    macd_div_dir = 'NEUTRAL'

    if macd_div_enabled:
        macd_div_lookback = config.get('M1_MACD_DIV_LOOKBACK', 40)
        macd_bull_div, macd_bear_div = _detect_macd_divergence(
            close[:idx + 1], macd_hist[:idx + 1], lookback=macd_div_lookback)

        if macd_bull_div and not macd_bear_div:
            macd_div_score = 0.75
            macd_div_dir = 'BULLISH'
        elif macd_bear_div and not macd_bull_div:
            macd_div_score = 0.25
            macd_div_dir = 'BEARISH'
        elif macd_bull_div and macd_bear_div:
            # Both — use histogram sign as tiebreaker
            if macd_hist[idx] < 0:
                macd_div_score, macd_div_dir = 0.65, 'BULLISH'
            elif macd_hist[idx] > 0:
                macd_div_score, macd_div_dir = 0.35, 'BEARISH'

    # ── Signal 3: Fast MACD crossover ──
    macd_series = pd.Series(macd_line)
    signal_series = pd.Series(signal_line)
    cross_dir, macd_score_raw, cross_up, cross_down = _macd_crossover_score(
        macd_series, signal_series, lookback=config.get('M1_V2_MACD_CONFIRM_BARS', 3))

    # Convert to 0-1 scale
    if cross_dir == 'BULLISH':
        macd_score = macd_score_raw
    elif cross_dir == 'BEARISH':
        macd_score = 1.0 - macd_score_raw
    else:
        macd_score = 0.5

    # ── Signal 4: ATR-normalized momentum ──
    mom_lookback = config.get('M1_V2_MOM_LOOKBACK', 8)
    mom_score = _momentum_score(
        pd.Series(close[:idx + 1]),
        pd.Series(atr[:idx + 1]),
        lookback=mom_lookback)

    # ── Blend ──
    # Default weights: RSI div 25%, MACD div 20%, crossover 35%, momentum 20%
    # Total divergence = 45% (was 40% for RSI-only)
    w_rsi_div = config.get('M1_W_RSI_DIV', 0.25)
    w_macd_div = config.get('M1_W_MACD_DIV', 0.20)
    w_cross = config.get('M1_W_CROSSOVER', 0.35)
    w_mom = config.get('M1_W_MOMENTUM', 0.20)

    if not macd_div_enabled:
        # Redistribute MACD divergence weight to crossover
        w_cross += w_macd_div
        w_macd_div = 0.0

    total_w = w_rsi_div + w_macd_div + w_cross + w_mom
    if total_w > 0:
        w_rsi_div /= total_w
        w_macd_div /= total_w
        w_cross /= total_w
        w_mom /= total_w

    combined = (rsi_div_score * w_rsi_div +
                macd_div_score * w_macd_div +
                macd_score * w_cross +
                mom_score * w_mom)

    # Direction from combined score
    if combined > 0.58:
        direction = 'BULLISH'
    elif combined < 0.42:
        direction = 'BEARISH'
    else:
        direction = 'NEUTRAL'

    # Final score: distance from neutral, scaled to 0.5-1.0
    final_score = 0.5 + abs(combined - 0.5)
    final_score = max(0.5, min(1.0, final_score))

    # ── Optional: 15m MTF confirmation boost ──
    mtf_boost = 0.0
    if df_15m is not None and idx_15m is not None and idx_15m >= 20:
        rsi_15m = df_15m['rsi'].values.astype(float)
        close_15m = df_15m['Close'].values.astype(float)
        atr_15m = df_15m['atr'].values.astype(float)

        # 15m RSI agreement
        if direction == 'BULLISH' and rsi_15m[idx_15m] > 50:
            mtf_boost += 0.03
        elif direction == 'BEARISH' and rsi_15m[idx_15m] < 50:
            mtf_boost += 0.03

        # 15m momentum agreement
        if len(close_15m) > idx_15m + 1:
            mom_15m = _momentum_score(
                pd.Series(close_15m[:idx_15m + 1]),
                pd.Series(atr_15m[:idx_15m + 1]),
                lookback=12)
            if direction == 'BULLISH' and mom_15m > 0.55:
                mtf_boost += 0.02
            elif direction == 'BEARISH' and mom_15m < 0.45:
                mtf_boost += 0.02

    final_score = min(1.0, final_score + mtf_boost)

    # ── Divergence agreement boost ──
    # When both RSI and MACD divergence fire on the same side, it's a
    # stronger signal than either alone. Small ICS boost.
    div_agree_boost = 0.0
    if macd_div_enabled:
        both_bull = rsi_bull_div and macd_bull_div
        both_bear = rsi_bear_div and macd_bear_div
        if (both_bull and direction == 'BULLISH') or (both_bear and direction == 'BEARISH'):
            div_agree_boost = config.get('M1_DIV_AGREE_BOOST', 0.03)
            final_score = min(1.0, final_score + div_agree_boost)

    details = {
        # RSI divergence
        'rsi_div_dir': rsi_div_dir,
        'rsi_div_score': round(rsi_div_score, 3),
        'rsi_bull_div': rsi_bull_div,
        'rsi_bear_div': rsi_bear_div,
        # MACD divergence
        'macd_div_enabled': macd_div_enabled,
        'macd_div_dir': macd_div_dir,
        'macd_div_score': round(macd_div_score, 3),
        'macd_bull_div': macd_bull_div,
        'macd_bear_div': macd_bear_div,
        # Backward compat (single 'div_dir' = whichever is strongest)
        'div_dir': rsi_div_dir if abs(rsi_div_score - 0.5) >= abs(macd_div_score - 0.5) else macd_div_dir,
        'div_score': round(max(rsi_div_score, macd_div_score, key=lambda s: abs(s - 0.5)), 3),
        'bull_div': rsi_bull_div or macd_bull_div,
        'bear_div': rsi_bear_div or macd_bear_div,
        # MACD crossover
        'macd_dir': cross_dir,
        'macd_score': round(macd_score, 3),
        'cross_up': cross_up,
        'cross_down': cross_down,
        # Momentum
        'mom_score': round(mom_score, 3),
        # Blend
        'combined': round(combined, 3),
        'div_agree_boost': round(div_agree_boost, 3),
        'mtf_boost': round(mtf_boost, 3),
        'weights': {
            'rsi_div': round(w_rsi_div, 3),
            'macd_div': round(w_macd_div, 3),
            'crossover': round(w_cross, 3),
            'momentum': round(w_mom, 3),
        },
    }

    return direction, final_score, details

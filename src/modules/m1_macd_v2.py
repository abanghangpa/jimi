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
    """M1 v2: Swing-based RSI divergence + ATR-normalized momentum + fast MACD crossover.

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

    # ── Signal 1: RSI Divergence (swing-based, 40% weight) ──
    lookback = config.get('M1_V2_RSI_LOOKBACK', 40)
    bull_div, bear_div = _detect_rsi_divergence(
        close[:idx + 1], rsi[:idx + 1], lookback=lookback)

    div_score = 0.5
    div_dir = 'NEUTRAL'
    if bull_div and not bear_div:
        div_score = 0.75
        div_dir = 'BULLISH'
    elif bear_div and not bull_div:
        div_score = 0.25
        div_dir = 'BEARISH'
    elif bull_div and bear_div:
        # Both detected — use RSI zone as tiebreaker
        if rsi[idx] < 45:
            div_score, div_dir = 0.65, 'BULLISH'
        elif rsi[idx] > 55:
            div_score, div_dir = 0.35, 'BEARISH'

    # ── Signal 2: Fast MACD crossover (35% weight) ──
    macd_series = pd.Series(macd_line)
    signal_series = pd.Series(signal_line)
    macd_dir, macd_score_raw, cross_up, cross_down = _macd_crossover_score(
        macd_series, signal_series, lookback=config.get('M1_V2_MACD_CONFIRM_BARS', 3))

    # Convert to 0-1 scale
    if macd_dir == 'BULLISH':
        macd_score = macd_score_raw
    elif macd_dir == 'BEARISH':
        macd_score = 1.0 - macd_score_raw
    else:
        macd_score = 0.5

    # ── Signal 3: ATR-normalized momentum (25% weight) ──
    mom_lookback = config.get('M1_V2_MOM_LOOKBACK', 8)
    mom_score = _momentum_score(
        pd.Series(close[:idx + 1]),
        pd.Series(atr[:idx + 1]),
        lookback=mom_lookback)

    # ── Blend ──
    w_div = 0.40
    w_macd = 0.35
    w_mom = 0.25

    combined = div_score * w_div + macd_score * w_macd + mom_score * w_mom

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

    details = {
        'div_dir': div_dir,
        'div_score': round(div_score, 3),
        'bull_div': bull_div,
        'bear_div': bear_div,
        'macd_dir': macd_dir,
        'macd_score': round(macd_score, 3),
        'cross_up': cross_up,
        'cross_down': cross_down,
        'mom_score': round(mom_score, 3),
        'combined': round(combined, 3),
        'mtf_boost': round(mtf_boost, 3),
    }

    return direction, final_score, details

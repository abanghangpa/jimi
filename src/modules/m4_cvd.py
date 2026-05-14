"""M4: CVD (Cumulative Volume Delta) — 15m Divergence + 2H Zero-Line.

v7 changes:
  - calc_cvd_15m: rolling window 8→16 (4hr) to filter bid-ask churn
  - detect_cvd_divergence_15m: tighter thresholds (0.05 slope, 0.5% swing)
  - score_m4: removed 0.50 floor, added sigmoid gating + ATR scaling
  - Layer A lookback: 5→3 bars (45min) to reduce stale divergence pickup

v7.1 fixes (missed divergence on 2300→2344 move):
  - Layer A lookback: 3→8 bars (2h) to catch multi-hour divergences
  - Slope threshold: 0.05→0.03 to catch slower grinds
  - Swing search window: look//2→look (full lookback for reference point)

v7.2 fixes (missed basing + reversal setups):
  - Method 4: Basing + CVD reversal — detects price compression with directional
    CVD accumulation/distribution. Catches the early reversal signal where price
    is range-bound but smart money is positioning.
  - Fixed swing comparison asymmetry: bullish case used look//2 while bearish
    used full lookback, causing missed bullish basing at swing lows.
  - Basing signals bypass sigmoid gate and ATR scaling — low ATR during
    compression is confirmatory, not noise.
"""

import numpy as np
import pandas as pd


def _sigmoid(x, center=0.65, steepness=12):
    """Sigmoid gating function. Maps score to [0, 1] gate value.

    Args:
        x: Input score (0-1)
        center: Inflection point where gate = 0.5
        steepness: How sharply the gate transitions (higher = sharper)
    Returns:
        Gate value 0-1
    """
    return 1.0 / (1.0 + np.exp(-steepness * (x - center)))


def _atr_scaling_factor(atr_current, atr_20p_avg):
    """Scale M4 contribution by ATR percentile.

    Low ATR = market maker spread noise dominates → dampen M4.
    Normal ATR = real flow → full M4 strength.
    """
    if atr_20p_avg <= 0 or np.isnan(atr_20p_avg):
        return 1.0
    return min(1.0, atr_current / atr_20p_avg)


def calc_cvd_15m(df_15m):
    """Rolling CVD — delta per bar, then smoothed.

    v7: Window 8→16 (4 hours) to filter sub-hour bid-ask churn.
    The old 8-bar window was short enough that a single market-maker
    order cluster could produce a false CVD 'momentum' signal.
    """
    taker_buy = df_15m['Taker buy base asset volume']
    total_vol = df_15m['Volume']
    taker_sell = total_vol - taker_buy
    delta = taker_buy - taker_sell
    return delta.rolling(16).sum()


def detect_cvd_divergence_15m(df_15m, lookback=24, window=12):
    """Detect CVD divergence on 15m bars.

    v7: Tightened thresholds to reduce noise-driven false positives.
    - Slope threshold: 0.03 → 0.05 (requires stronger directional disagreement)
    - Swing comparison: 0.2% → 0.5% (requires real swing, not wick noise)
    - Exhaustion: 1.5σ → 2.0σ (requires stronger CVD momentum reversal)
    - Min bars between divergences: 4 → 6 (reduces clustering on noise)
    """
    cvd = df_15m['cvd_15m'].values
    close = df_15m['Close'].values
    high = df_15m['High'].values
    low = df_15m['Low'].values
    n = len(df_15m)
    divergence = ['NONE'] * n
    last_div_bar = -999

    for i in range(lookback + window, n):
        if i - last_div_bar < 6:
            continue

        # Method 1: Slope comparison
        w = window
        price_slice = close[i-w:i+1]
        cvd_slice = cvd[i-w:i+1]

        if len(price_slice) >= 3 and not np.any(np.isnan(cvd_slice)):
            x = np.arange(len(price_slice))
            price_slope = np.polyfit(x, price_slice, 1)[0]
            cvd_slope = np.polyfit(x, cvd_slice, 1)[0]
            price_range = np.max(price_slice) - np.min(price_slice)
            cvd_range = np.max(cvd_slice) - np.min(cvd_slice)

            if price_range > 0 and cvd_range > 0:
                price_dir = price_slope / price_range
                cvd_dir = cvd_slope / cvd_range
                # v7.1: 0.05→0.03 — catches slower grinds, still filters noise
                if price_dir > 0.03 and cvd_dir < -0.03:
                    divergence[i] = 'BEARISH'
                elif price_dir < -0.03 and cvd_dir > 0.03:
                    divergence[i] = 'BULLISH'

        # Method 2: Swing high/low comparison
        if divergence[i] == 'NONE':
            look = min(lookback, i)
            # v7.1: Use full lookback window for swing reference, not half
            if i >= 4 and (high[i] >= np.max(high[i-3:i+1]) * 0.9995):
                prev_hi = i - look
                if prev_hi >= 3 and high[prev_hi] >= np.max(high[max(0,prev_hi-3):prev_hi+1]) * 0.9995:
                    if high[i] > high[prev_hi] * 1.005:
                        cvd_at_i = np.nanmean(cvd[max(0,i-1):i+1])
                        cvd_at_prev = np.nanmean(cvd[max(0,prev_hi-1):prev_hi+1])
                        if cvd_at_i < cvd_at_prev * 0.990:
                            divergence[i] = 'BEARISH'

            if divergence[i] == 'NONE':
                if i >= 4 and (low[i] <= np.min(low[i-3:i+1]) * 1.0005):
                    prev_lo = i - look  # v7.2: was look//2 — asymmetric with bearish case
                    if prev_lo >= 3 and low[prev_lo] <= np.min(low[max(0,prev_lo-3):prev_lo+1]) * 1.0005:
                        if low[i] < low[prev_lo] * 0.995:
                            cvd_at_i = np.nanmean(cvd[max(0,i-1):i+1])
                            cvd_at_prev = np.nanmean(cvd[max(0,prev_lo-1):prev_lo+1])
                            if cvd_at_i > cvd_at_prev * 1.010:
                                divergence[i] = 'BULLISH'

        # Method 3: Exhaustion
        if divergence[i] == 'NONE' and i >= 8:
            cvd_momentum = cvd[i] - cvd[i-4]
            price_momentum = close[i] - close[i-4]
            cvd_std = np.nanstd(cvd[max(0,i-24):i+1])
            if cvd_std > 0:
                # v7: 1.5σ→2.0σ (requires stronger exhaustion signal)
                if (price_momentum > 0 and cvd_momentum < 0 and
                    high[i] >= np.max(high[max(0,i-8):i+1]) * 0.999 and
                    abs(cvd_momentum) > cvd_std * 2.0):
                    divergence[i] = 'BEARISH'
                elif (price_momentum < 0 and cvd_momentum > 0 and
                      low[i] <= np.min(low[max(0,i-8):i+1]) * 1.001 and
                      abs(cvd_momentum) > cvd_std * 2.0):
                    divergence[i] = 'BULLISH'

        # Method 4: Basing + CVD Reversal Setup (v7.2)
        # Detects when price is in a compression zone (contracting range)
        # while CVD shows directional accumulation/distribution.
        # This catches early reversal signals that Methods 1-3 miss:
        #   - Price is range-bound (flat slope, shrinking range)
        #   - CVD is quietly trending (smart money positioning)
        #   - ATR is compressing (volatility drying up = real basing)
        #   - Result: "basing before reversal" pattern
        if divergence[i] == 'NONE' and i >= 36:
            base_window = 12   # 3 hours of 15m bars for current base
            prior_bars = 24    # 6 hours prior for range comparison
            bw = base_window

            # Step 1: Is price compressing? (range contracting)
            hi_base = np.max(high[i-bw:i+1])
            lo_base = np.min(low[i-bw:i+1])
            recent_range = hi_base - lo_base

            hi_prior = np.max(high[max(0, i-prior_bars):i-bw+1])
            lo_prior = np.min(low[max(0, i-prior_bars):i-bw+1])
            prior_range = hi_prior - lo_prior

            if prior_range > 0 and recent_range > 0:
                range_ratio = recent_range / prior_range

                # Basing criterion: range is contracting (< 75% of prior range)
                if range_ratio < 0.75:
                    # Step 1b: ATR compression check (v7.3)
                    # ATR compression CONFIRMS real basing vs noise.
                    # Backtest (30d): compressed basing → 50% ATR expansion after,
                    # non-compressed → 25% expansion (false positives).
                    atr_compressed = True  # default if ATR unavailable
                    if i >= 34:  # need 14 bars for ATR + 20 for average
                        tr = np.maximum(high[i-bw:i+1] - low[i-bw:i+1],
                              np.maximum(np.abs(high[i-bw:i+1] - close[i-bw-1:i]),
                                         np.abs(low[i-bw:i+1] - close[i-bw-1:i])))
                        atr_recent = np.nanmean(tr)
                        # Compare to longer ATR average
                        tr_full = np.maximum(high[max(0,i-34):i+1] - low[max(0,i-34):i+1],
                                   np.maximum(np.abs(high[max(0,i-34):i+1] - close[max(0,i-35):i]),
                                              np.abs(low[max(0,i-34):i+1] - close[max(0,i-35):i])))
                        atr_avg = np.nanmean(tr_full)
                        if atr_avg > 0:
                            atr_ratio = atr_recent / atr_avg
                            atr_compressed = atr_ratio < 0.90  # ATR below 90% of average

                    # Non-compressed basing requires tighter price criteria
                    if not atr_compressed:
                        range_threshold = 0.60  # need stronger compression
                    else:
                        range_threshold = 0.75  # normal threshold

                    if range_ratio < range_threshold:
                        # Step 2: Is price flat? (slope near zero)
                        price_slice = close[i-bw:i+1]
                        cvd_slice = cvd[i-bw:i+1]

                        if len(price_slice) >= 3 and not np.any(np.isnan(cvd_slice)):
                            x = np.arange(len(price_slice))
                            price_slope = np.polyfit(x, price_slice, 1)[0]
                            cvd_slope = np.polyfit(x, cvd_slice, 1)[0]

                            price_range_val = np.max(price_slice) - np.min(price_slice)
                            cvd_mean = np.nanmean(np.abs(cvd_slice))
                            if price_range_val > 0 and cvd_mean > 0:
                                price_dir = price_slope / price_range_val
                                cvd_dir = cvd_slope / cvd_mean

                                # Price must be flat (basing, not trending)
                                # CVD must show directional flow
                                is_flat_price = abs(price_dir) < 0.025
                                is_strong_cvd = abs(cvd_dir) > 0.015

                                if is_flat_price and is_strong_cvd:
                                    # Step 3: Confirm with position in range
                                    # Bullish basing: near range low + CVD rising
                                    # Bearish basing: near range high + CVD falling
                                    mid_range = (hi_base + lo_base) / 2.0
                                    price_pos = (close[i] - lo_base) / recent_range if recent_range > 0 else 0.5

                                    if cvd_dir > 0 and price_pos < 0.45:
                                        # CVD accumulating while price basing near lows
                                        divergence[i] = 'BULLISH_BASE'
                                    elif cvd_dir < 0 and price_pos > 0.55:
                                        # CVD distributing while price basing near highs
                                        divergence[i] = 'BEARISH_BASE'

        # Method 5: Retest of Recent Swing (v7.2)
        # Detects when price retests a recent swing low/high and CVD holds up.
        # This catches the second touch that Methods 1-4 miss — the retest
        # after the initial reversal signal, confirming the move.
        if divergence[i] == 'NONE' and i >= 24:
            look = min(lookback, i)
            retest_tolerance = 1.003  # within 0.3% of swing level

            # Bullish retest: price retests a recent swing low, CVD improved
            for j in range(max(0, i - look), i - 4):
                if low[j] <= np.min(low[max(0, j-3):j+1]) * 1.0005:
                    # Found a swing low — is current bar retesting it?
                    if low[i] <= low[j] * retest_tolerance:
                        # Has price bounced between the two touches?
                        between_hi = np.max(high[j+1:i])
                        if between_hi > low[j] * 1.005:
                            # CVD check: is CVD at retest >= CVD at first touch?
                            cvd_at_i = np.nanmean(cvd[max(0, i-1):i+1])
                            cvd_at_j = np.nanmean(cvd[max(0, j-1):j+1])
                            if not np.isnan(cvd_at_i) and not np.isnan(cvd_at_j):
                                if cvd_at_i >= cvd_at_j * 0.995:
                                    divergence[i] = 'BULLISH'
                                    break

            # Bearish retest: price retests a recent swing high, CVD weakened
            if divergence[i] == 'NONE':
                for j in range(max(0, i - look), i - 4):
                    if high[j] >= np.max(high[max(0, j-3):j+1]) * 0.9995:
                        if high[i] >= high[j] / retest_tolerance:
                            between_lo = np.min(low[j+1:i])
                            if between_lo < high[j] / 1.005:
                                cvd_at_i = np.nanmean(cvd[max(0, i-1):i+1])
                                cvd_at_j = np.nanmean(cvd[max(0, j-1):j+1])
                                if not np.isnan(cvd_at_i) and not np.isnan(cvd_at_j):
                                    if cvd_at_i <= cvd_at_j * 1.005:
                                        divergence[i] = 'BEARISH'
                                        break

        if divergence[i] != 'NONE':
            last_div_bar = i

    return pd.Series(divergence, index=df_15m.index)


def calc_cvd_2h(df_2h):
    """Rolling CVD on 2H bars."""
    taker_buy = df_2h['Taker buy base asset volume']
    total_vol = df_2h['Volume']
    taker_sell = total_vol - taker_buy
    delta = taker_buy - taker_sell
    return delta.rolling(12).sum()


def detect_cvd_zero_cross(df_2h):
    """Detect when 2H CVD crosses the zero line."""
    cvd = df_2h['cvd_2h'].values
    n = len(df_2h)
    state = ['NONE'] * n
    cross_bar = [-1] * n
    cross_dir = ['NONE'] * n

    for i in range(1, n):
        if pd.isna(cvd[i]) or pd.isna(cvd[i-1]):
            continue
        if cvd[i-1] <= 0 and cvd[i] > 0:
            state[i] = 'CROSS_UP'
            cross_bar[i] = i
            cross_dir[i] = 'UP'
        elif cvd[i-1] >= 0 and cvd[i] < 0:
            state[i] = 'CROSS_DOWN'
            cross_bar[i] = i
            cross_dir[i] = 'DOWN'
        elif cvd[i] > 0:
            state[i] = 'ABOVE'
        else:
            state[i] = 'BELOW'

        if state[i] not in ('CROSS_UP', 'CROSS_DOWN') and i > 0:
            cross_bar[i] = cross_bar[i-1]
            cross_dir[i] = cross_dir[i-1]

    return (pd.Series(state, index=df_2h.index),
            pd.Series(cross_bar, index=df_2h.index),
            pd.Series(cross_dir, index=df_2h.index))


def score_m4(df_15m, df_2h, idx_15m, idx_2h, direction, config):
    """M4: CVD Composite — 15m divergence + 2H zero-line cross.

    v7 changes:
    - Layer A lookback: 5→3 bars (45min, reduces stale noise pickup)
    - Removed max(combined, 0.50) floor — M4 can now score below 0.50
    - Added sigmoid gating: weak signals get near-zero weight
    - Added ATR scaling: low-vol sessions dampen M4 contribution
    - Combined score = raw * sigmoid_gate * atr_mult

    v7.2 changes:
    - Recognizes _BASE divergence variants (basing + CVD reversal)
    - Basing signals use gentler sigmoid gate (center=0.45, steepness=6)
    - Basing signals skip ATR scaling (low ATR during compression is confirmatory)
    """
    layer_a_score = 0.0
    layer_a_status = 'FAIL'
    layer_a_div = 'NONE'
    layer_b_score = 0.0
    layer_b_status = 'FAIL'
    layer_b_cross = 'NONE'
    layer_b_bars_since = 999
    zl_state = 'NONE'
    basing_detected = False  # v7.2: track basing + CVD reversal signals
    atr_compressed_basing = False  # v7.3: ATR confirms real basing

    # Layer A: 15m CVD Divergence
    # v7.2: Look back 24 bars (6h), detect divergence in BOTH directions
    #   Confirming div (same direction as trade) → positive score
    #   Opposing div (against trade direction) → negative score (warning)
    #   Also detects _BASE variants (basing + CVD reversal setups)
    if idx_15m >= config['CVD_LOOKBACK']:
        confirming_div = None
        opposing_div = None
        confirming_bar = -1
        opposing_bar = -1
        basing_detected = False  # v7.2: track basing signals

        for ci in range(max(0, idx_15m - 24), idx_15m + 1):
            div = df_15m['cvd_divergence_15m'].iloc[ci]
            if div == 'NONE':
                continue
            # v7.2: Normalize _BASE variants for direction matching
            div_base = div.endswith('_BASE')
            div_clean = div.replace('_BASE', '')
            is_confirming = (direction == 'LONG' and div_clean == 'BULLISH') or \
                            (direction == 'SHORT' and div_clean == 'BEARISH')
            is_opposing = (direction == 'LONG' and div_clean == 'BEARISH') or \
                          (direction == 'SHORT' and div_clean == 'BULLISH')
            if is_confirming and confirming_div is None:
                confirming_div = div
                confirming_bar = ci
                if div_base:
                    basing_detected = True
                    # v7.3: Check ATR compression at basing signal bar
                    if ci >= 34 and 'atr' in df_15m.columns:
                        atr_at = df_15m['atr'].iloc[ci]
                        atr_avg = df_15m['atr'].iloc[max(0, ci-20):ci].mean()
                        if not pd.isna(atr_at) and not pd.isna(atr_avg) and atr_avg > 0:
                            atr_compressed_basing = (atr_at / atr_avg) < 0.90
            elif is_opposing and opposing_div is None:
                opposing_div = div
                opposing_bar = ci
            if confirming_div and opposing_div:
                break

        # Score confirming divergence (trade-supporting)
        if confirming_div is not None:
            cvd_now = df_15m['cvd_15m'].iloc[idx_15m]
            cvd_prev = df_15m['cvd_15m'].iloc[max(0, idx_15m - 8)]
            if pd.isna(cvd_now) or pd.isna(cvd_prev):
                layer_a_score = 0.55
            else:
                cvd_delta = abs(cvd_now - cvd_prev)
                cvd_std = df_15m['cvd_15m'].iloc[max(0, idx_15m-48):idx_15m+1].std()
                if cvd_std > 0:
                    layer_a_score = min(cvd_delta / (cvd_std * 2), 1.0)
                else:
                    layer_a_score = 0.5
            layer_a_score = max(layer_a_score, 0.50)
            layer_a_status = 'PASS'
            layer_a_div = confirming_div

        # Penalize opposing divergence (trade-threatening)
        if opposing_div is not None:
            # Freshness: closer bars = stronger warning
            bars_ago = idx_15m - opposing_bar
            freshness = max(0.3, 1.0 - bars_ago / 24.0)  # 1.0 at bar 0, 0.3 at bar 24
            # Opposing divergence reduces layer_a_score or flips it negative
            penalty = 0.30 * freshness
            if layer_a_score > 0:
                layer_a_score = max(0.0, layer_a_score - penalty)
            else:
                layer_a_score = -penalty
            layer_a_div = opposing_div  # expose to veto system

    # Layer B: 2H CVD Zero-Line Cross
    if idx_2h >= 1 and 'cvd_zl_state' in df_2h.columns:
        zl_state = df_2h['cvd_zl_state'].iloc[idx_2h]
        cross_bar = df_2h['cvd_zl_cross_bar'].iloc[idx_2h]
        cross_dir = df_2h['cvd_zl_cross_dir'].iloc[idx_2h]
        cvd_2h_now = df_2h['cvd_2h'].iloc[idx_2h]

        if not pd.isna(cvd_2h_now):
            bars_since = idx_2h - cross_bar if cross_bar >= 0 else 999
            layer_b_bars_since = bars_since
            fresh = bars_since <= config['M4_ZL_MOMENTUM_BARS']

            if direction == 'LONG':
                if zl_state == 'CROSS_UP':
                    layer_b_score = 0.90 if fresh else 0.70
                    layer_b_status = 'PASS'
                    layer_b_cross = 'CROSS_UP'
                elif zl_state == 'ABOVE' and cross_dir == 'UP':
                    if bars_since <= config['M4_ZL_MOMENTUM_BARS']:
                        layer_b_score = 0.80
                        layer_b_status = 'PASS'
                        layer_b_cross = 'ABOVE_FRESH'
                    elif bars_since <= config['M4_ZL_LOOKBACK']:
                        layer_b_score = 0.65
                        layer_b_status = 'PASS'
                        layer_b_cross = 'ABOVE_AFTER_UP'
                    else:
                        layer_b_score = 0.50
                        layer_b_status = 'PASS'
                        layer_b_cross = 'ABOVE_STALE'
                elif zl_state == 'ABOVE':
                    layer_b_score = 0.40
                    layer_b_status = 'PASS'
                    layer_b_cross = 'ABOVE_NO_CROSS'

            elif direction == 'SHORT':
                if zl_state == 'CROSS_DOWN':
                    layer_b_score = 0.90 if fresh else 0.70
                    layer_b_status = 'PASS'
                    layer_b_cross = 'CROSS_DOWN'
                elif zl_state == 'BELOW' and cross_dir == 'DOWN':
                    if bars_since <= config['M4_ZL_MOMENTUM_BARS']:
                        layer_b_score = 0.80
                        layer_b_status = 'PASS'
                        layer_b_cross = 'BELOW_FRESH'
                    elif bars_since <= config['M4_ZL_LOOKBACK']:
                        layer_b_score = 0.65
                        layer_b_status = 'PASS'
                        layer_b_cross = 'BELOW_AFTER_DOWN'
                    else:
                        layer_b_score = 0.50
                        layer_b_status = 'PASS'
                        layer_b_cross = 'BELOW_STALE'
                elif zl_state == 'BELOW':
                    layer_b_score = 0.40
                    layer_b_status = 'PASS'
                    layer_b_cross = 'BELOW_NO_CROSS'

            if direction == 'LONG' and zl_state in ('BELOW', 'CROSS_DOWN'):
                if layer_b_status != 'PASS':
                    layer_b_score = 0.20
                    layer_b_cross = f'CONFLICT_{zl_state}'
            elif direction == 'SHORT' and zl_state in ('ABOVE', 'CROSS_UP'):
                if layer_b_status != 'PASS':
                    layer_b_score = 0.20
                    layer_b_cross = f'CONFLICT_{zl_state}'

            if layer_b_status == 'PASS' and idx_2h >= 3:
                cvd_slope_2h = (df_2h['cvd_2h'].iloc[idx_2h] - df_2h['cvd_2h'].iloc[max(0, idx_2h-3)]) / 3
                if not pd.isna(cvd_slope_2h):
                    if (direction == 'LONG' and cvd_slope_2h > 0) or \
                       (direction == 'SHORT' and cvd_slope_2h < 0):
                        layer_b_score = min(layer_b_score * 1.15, 1.0)

    w_div = config['M4_DIV_WEIGHT']
    w_zl = config['M4_ZL_WEIGHT']
    raw_combined = layer_a_score * w_div + layer_b_score * w_zl

    if layer_a_status == 'PASS' or layer_b_status == 'PASS':
        status = 'PASS'
    else:
        status = 'FAIL'

    # ── v7: Sigmoid gating + ATR scaling (replaces max(combined, 0.50) floor) ──

    # v7.2: Basing signals bypass sigmoid + ATR gates.
    # Rationale: During basing, price compression (low ATR) IS the pattern —
    # penalizing low ATR is counterproductive. And basing produces inherently
    # weak raw signals that the sigmoid gate would crush to near-zero.
    if basing_detected:
        # Basing: gentle gate (center 0.45, steepness 6) — allows weak signals through
        gate_center = 0.45
        gate_steepness = 6
        sigmoid_gate = _sigmoid(raw_combined, center=gate_center, steepness=gate_steepness)
        atr_mult = 1.0  # Don't penalize low ATR during basing

        # v7.3: ATR-compressed basing gets a score boost.
        # Backtest: ATR-compressed basing → 53.8% ATR expansion after,
        # confirming real basing vs noise. Boost by 20%.
        if atr_compressed_basing:
            basing_atr_boost = config.get('M4_BASING_ATR_BOOST', 1.20)
        else:
            basing_atr_boost = 1.0  # no boost for non-compressed basing
    else:
        # Standard: sigmoid gate — weak M4 signals get near-zero weight
        gate_center = config.get('M4_SIGMOID_CENTER', 0.65)
        gate_steepness = config.get('M4_SIGMOID_STEEPNESS', 12)
        sigmoid_gate = _sigmoid(raw_combined, center=gate_center, steepness=gate_steepness)

        # v7.2: Layer A rescue — when 15m divergence is strong but 2H conflict
        # drags combined below gate center, blend with Layer A-based gate.
        # Example: Layer A=1.0, Layer B=0.2, combined=0.52 → sigmoid=0.17
        #   But Layer A=1.0 is a REAL signal — don't kill it.
        #   Rescue gate = sigmoid(layer_a_score) → 0.98
        #   Blend: sigmoid_gate = 0.3 * rescue + 0.7 * standard = 0.74
        if layer_a_score >= 0.70 and layer_a_status == 'PASS':
            rescue_gate = _sigmoid(layer_a_score, center=0.55, steepness=10)
            blend = config.get('M4_LAYER_A_RESCUE_BLEND', 0.40)
            sigmoid_gate = sigmoid_gate * (1 - blend) + rescue_gate * blend

        # ATR scaling: dampen M4 in low-volatility sessions (spread noise dominates)
        atr_mult = 1.0  # default if ATR data unavailable
        basing_atr_boost = 1.0  # not basing — no boost
        if idx_15m >= 20 and 'atr' in df_15m.columns:
            atr_now = df_15m['atr'].iloc[idx_15m]
            atr_avg = df_15m['atr'].iloc[max(0, idx_15m-20):idx_15m+1].mean()
            if not pd.isna(atr_now) and not pd.isna(atr_avg):
                atr_mult = _atr_scaling_factor(atr_now, atr_avg)

    # v7: NO FLOOR — M4 can score below 0.50 when signal is weak
    # Old: score = max(combined, 0.50)  ← this was the structural flaw
    # New: score = raw * gate * atr_mult, allowing M4 to contribute 0.0
    if status == 'PASS':
        score = raw_combined * sigmoid_gate * atr_mult * basing_atr_boost
        score = max(score, 0.0)  # floor at 0, not 0.50
    else:
        score = 0.0

    details = {
        'layer_a_div': layer_a_div,
        'layer_a_score': round(layer_a_score, 3),
        'layer_b_cross': layer_b_cross,
        'layer_b_zl_state': zl_state,
        'layer_b_score': round(layer_b_score, 3),
        'layer_b_bars_since': layer_b_bars_since,
        'combined': round(raw_combined, 3),
        'sigmoid_gate': round(sigmoid_gate, 3),
        'atr_mult': round(atr_mult, 3),
        'basing_detected': basing_detected,  # v7.2
        'atr_compressed_basing': atr_compressed_basing if basing_detected else False,  # v7.3
        'basing_atr_boost': round(basing_atr_boost, 2) if basing_detected else 1.0,  # v7.3
        'score': round(score, 3),
    }
    return status, score, details

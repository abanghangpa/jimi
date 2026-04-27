"""M4: CVD (Cumulative Volume Delta) — 15m Divergence + 2H Zero-Line."""

import numpy as np
import pandas as pd


def calc_cvd_15m(df_15m):
    """Rolling CVD — delta per bar, then smoothed."""
    taker_buy = df_15m['Taker buy base asset volume']
    total_vol = df_15m['Volume']
    taker_sell = total_vol - taker_buy
    delta = taker_buy - taker_sell
    return delta.rolling(8).sum()


def detect_cvd_divergence_15m(df_15m, lookback=36, window=12):
    """Detect CVD divergence on 15m bars."""
    cvd = df_15m['cvd_15m'].values
    close = df_15m['Close'].values
    high = df_15m['High'].values
    low = df_15m['Low'].values
    n = len(df_15m)
    divergence = ['NONE'] * n
    last_div_bar = -999

    for i in range(lookback + window, n):
        if i - last_div_bar < 4:
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
                if price_dir > 0.03 and cvd_dir < -0.03:
                    divergence[i] = 'BEARISH'
                elif price_dir < -0.03 and cvd_dir > 0.03:
                    divergence[i] = 'BULLISH'

        # Method 2: Swing high/low comparison
        if divergence[i] == 'NONE':
            look = min(lookback, i)
            if i >= 4 and (high[i] >= np.max(high[i-3:i+1]) * 0.9995):
                prev_hi = i - look // 2
                if prev_hi >= 3 and high[prev_hi] >= np.max(high[max(0,prev_hi-3):prev_hi+1]) * 0.9995:
                    if high[i] > high[prev_hi] * 1.002:
                        cvd_at_i = np.nanmean(cvd[max(0,i-1):i+1])
                        cvd_at_prev = np.nanmean(cvd[max(0,prev_hi-1):prev_hi+1])
                        if cvd_at_i < cvd_at_prev * 0.993:
                            divergence[i] = 'BEARISH'

            if divergence[i] == 'NONE':
                if i >= 4 and (low[i] <= np.min(low[i-3:i+1]) * 1.0005):
                    prev_lo = i - look // 2
                    if prev_lo >= 3 and low[prev_lo] <= np.min(low[max(0,prev_lo-3):prev_lo+1]) * 1.0005:
                        if low[i] < low[prev_lo] * 0.998:
                            cvd_at_i = np.nanmean(cvd[max(0,i-1):i+1])
                            cvd_at_prev = np.nanmean(cvd[max(0,prev_lo-1):prev_lo+1])
                            if cvd_at_i > cvd_at_prev * 1.007:
                                divergence[i] = 'BULLISH'

        # Method 3: Exhaustion
        if divergence[i] == 'NONE' and i >= 8:
            cvd_momentum = cvd[i] - cvd[i-4]
            price_momentum = close[i] - close[i-4]
            cvd_std = np.nanstd(cvd[max(0,i-24):i+1])
            if cvd_std > 0:
                if (price_momentum > 0 and cvd_momentum < 0 and
                    high[i] >= np.max(high[max(0,i-8):i+1]) * 0.999 and
                    abs(cvd_momentum) > cvd_std * 1.5):
                    divergence[i] = 'BEARISH'
                elif (price_momentum < 0 and cvd_momentum > 0 and
                      low[i] <= np.min(low[max(0,i-8):i+1]) * 1.001 and
                      abs(cvd_momentum) > cvd_std * 1.5):
                    divergence[i] = 'BULLISH'

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
    """M4: CVD Composite — 15m divergence + 2H zero-line cross."""
    layer_a_score = 0.0
    layer_a_status = 'FAIL'
    layer_a_div = 'NONE'
    layer_b_score = 0.0
    layer_b_status = 'FAIL'
    layer_b_cross = 'NONE'
    layer_b_bars_since = 999
    zl_state = 'NONE'

    # Layer A: 15m CVD Divergence
    if idx_15m >= config['CVD_LOOKBACK']:
        for ci in range(max(0, idx_15m - 5), idx_15m + 1):
            div = df_15m['cvd_divergence_15m'].iloc[ci]
            if (direction == 'LONG' and div == 'BULLISH') or (direction == 'SHORT' and div == 'BEARISH'):
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
                layer_a_div = div
                break

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
    combined = layer_a_score * w_div + layer_b_score * w_zl

    if layer_a_status == 'PASS' or layer_b_status == 'PASS':
        status = 'PASS'
    else:
        status = 'FAIL'

    score = max(combined, 0.50) if status == 'PASS' else 0.50

    details = {
        'layer_a_div': layer_a_div,
        'layer_a_score': round(layer_a_score, 3),
        'layer_b_cross': layer_b_cross,
        'layer_b_zl_state': zl_state,
        'layer_b_score': round(layer_b_score, 3),
        'layer_b_bars_since': layer_b_bars_since,
        'combined': round(combined, 3),
    }
    return status, score, details

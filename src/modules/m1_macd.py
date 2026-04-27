"""M1: Enhanced MACD + RSI Divergence + Momentum Shift."""

import numpy as np


def score_m1(df_1h, idx, config):
    """Enhanced M1: MACD histogram + RSI divergence + momentum shift."""
    if idx < 2:
        return 'NEUTRAL', 0.5

    hist = df_1h['macd_hist'].iloc[idx]
    hist_prev = df_1h['macd_hist'].iloc[idx - 1]

    if hist > 0 and hist > hist_prev:
        macd_dir, macd_score = 'BULLISH', 1.0
    elif hist < 0 and hist < hist_prev:
        macd_dir, macd_score = 'BEARISH', 1.0
    elif hist > 0:
        macd_dir, macd_score = 'BULLISH', 0.7
    elif hist < 0:
        macd_dir, macd_score = 'BEARISH', 0.7
    else:
        macd_dir, macd_score = 'NEUTRAL', 0.5

    rsi_score = None
    if config.get('M1_RSI_ENABLED', False) and 'rsi' in df_1h.columns:
        rsi = df_1h['rsi'].iloc[idx]
        rsi_prev = df_1h['rsi'].iloc[idx - 1]
        close = df_1h['Close'].iloc[idx]
        close_prev = df_1h['Close'].iloc[idx - 1]
        ob = config.get('M1_RSI_OVERBOUGHT', 70)
        os_ = config.get('M1_RSI_OVERSOLD', 30)

        if close > close_prev and rsi < rsi_prev and rsi > 55:
            rsi_score = 0.35
        elif close < close_prev and rsi > rsi_prev and rsi < 45:
            rsi_score = 0.65
        elif rsi > ob:
            rsi_score = 0.4
        elif rsi < os_:
            rsi_score = 0.6

    mom_score = None
    if config.get('M1_MOMENTUM_ENABLED', False):
        lookback = config.get('M1_MOMENTUM_LOOKBACK', 6)
        if idx >= lookback:
            roc_now = (df_1h['Close'].iloc[idx] - df_1h['Close'].iloc[idx - lookback]) / df_1h['Close'].iloc[idx - lookback]
            roc_prev = (df_1h['Close'].iloc[idx - 1] - df_1h['Close'].iloc[idx - lookback - 1]) / df_1h['Close'].iloc[idx - lookback - 1] if idx >= lookback + 1 else roc_now
            accel = roc_now - roc_prev

            if roc_now > 0.02 and accel > 0:
                mom_score = 0.8
            elif roc_now > 0.02 and accel < 0:
                mom_score = 0.6
            elif roc_now < -0.02 and accel < 0:
                mom_score = 0.2
            elif roc_now < -0.02 and accel > 0:
                mom_score = 0.4

    rsi_enabled = rsi_score is not None
    mom_enabled = mom_score is not None

    if not rsi_enabled and not mom_enabled:
        return macd_dir, macd_score

    active_scores = [macd_score]
    active_weights = [0.5]
    if rsi_enabled:
        active_scores.append(rsi_score)
        active_weights.append(0.25)
    if mom_enabled:
        active_scores.append(mom_score)
        active_weights.append(0.25)

    total_w = sum(active_weights)
    active_weights = [w / total_w for w in active_weights]
    combined = sum(s * w for s, w in zip(active_scores, active_weights))

    if combined > 0.6:
        direction = 'BULLISH'
    elif combined < 0.4:
        direction = 'BEARISH'
    else:
        direction = macd_dir

    final_score = 0.5 + abs(combined - 0.5)
    return direction, final_score

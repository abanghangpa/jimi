"""M11: Multi-Timeframe Momentum Divergence."""

import numpy as np
import pandas as pd


def detect_rsi_divergence(close_series, rsi_series, lookback=20, min_bars=5):
    """Detect RSI divergence on a single timeframe."""
    if len(close_series) < lookback + min_bars:
        return 'NONE'

    close = close_series.iloc[-lookback:].values
    rsi = rsi_series.iloc[-lookback:].values
    if np.any(np.isnan(rsi)):
        return 'NONE'

    for i in range(len(close) - min_bars, len(close) - 1):
        if i >= 2 and close[i] <= close[i-1] and close[i] <= close[i-2]:
            for j in range(max(0, i - lookback//2), i - min_bars + 1):
                if j >= 2 and close[j] <= close[j-1] and close[j] <= close[j-2]:
                    if close[i] < close[j] * 0.998 and rsi[i] > rsi[j] + 2:
                        return 'BULLISH'

    for i in range(len(close) - min_bars, len(close) - 1):
        if i >= 2 and close[i] >= close[i-1] and close[i] >= close[i-2]:
            for j in range(max(0, i - lookback//2), i - min_bars + 1):
                if j >= 2 and close[j] >= close[j-1] and close[j] >= close[j-2]:
                    if close[i] > close[j] * 1.002 and rsi[i] < rsi[j] - 2:
                        return 'BEARISH'

    return 'NONE'


def detect_macd_divergence(close_series, macd_hist_series, lookback=20, min_bars=5):
    """Detect MACD histogram divergence."""
    if len(close_series) < lookback + min_bars:
        return 'NONE'

    close = close_series.iloc[-lookback:].values
    macd = macd_hist_series.iloc[-lookback:].values
    if np.any(np.isnan(macd)):
        return 'NONE'

    for i in range(len(close) - min_bars, len(close) - 1):
        if i >= 2 and close[i] <= close[i-1] and close[i] <= close[i-2]:
            for j in range(max(0, i - lookback//2), i - min_bars + 1):
                if j >= 2 and close[j] <= close[j-1] and close[j] <= close[j-2]:
                    if close[i] < close[j] * 0.998 and macd[i] > macd[j]:
                        return 'BULLISH'

    for i in range(len(close) - min_bars, len(close) - 1):
        if i >= 2 and close[i] >= close[i-1] and close[i] >= close[i-2]:
            for j in range(max(0, i - lookback//2), i - min_bars + 1):
                if j >= 2 and close[j] >= close[j-1] and close[j] >= close[j-2]:
                    if close[i] > close[j] * 1.002 and macd[i] < macd[j]:
                        return 'BEARISH'

    return 'NONE'


def score_m11_mtf_momentum(df_15m, df_1h, df_4h, idx_15m, idx_1h, idx_4h, direction):
    """Score multi-timeframe momentum divergence."""
    details = {}
    divergences = {}

    if idx_15m >= 30 and 'rsi' in df_15m.columns:
        rsi_15m = df_15m['rsi'].iloc[max(0, idx_15m-30):idx_15m+1]
        close_15m = df_15m['Close'].iloc[max(0, idx_15m-30):idx_15m+1]
        divergences['rsi_15m'] = detect_rsi_divergence(close_15m, rsi_15m)
    else:
        divergences['rsi_15m'] = 'NONE'

    if idx_1h >= 30 and 'rsi' in df_1h.columns:
        rsi_1h = df_1h['rsi'].iloc[max(0, idx_1h-30):idx_1h+1]
        close_1h = df_1h['Close'].iloc[max(0, idx_1h-30):idx_1h+1]
        divergences['rsi_1h'] = detect_rsi_divergence(close_1h, rsi_1h)
    else:
        divergences['rsi_1h'] = 'NONE'

    if idx_1h >= 30 and 'macd_hist' in df_1h.columns:
        macd_1h = df_1h['macd_hist'].iloc[max(0, idx_1h-30):idx_1h+1]
        close_1h = df_1h['Close'].iloc[max(0, idx_1h-30):idx_1h+1]
        divergences['macd_1h'] = detect_macd_divergence(close_1h, macd_1h)
    else:
        divergences['macd_1h'] = 'NONE'

    if idx_4h >= 20 and 'macd_hist' in df_4h.columns:
        macd_4h = df_4h['macd_hist'].iloc[max(0, idx_4h-20):idx_4h+1]
        close_4h = df_4h['Close'].iloc[max(0, idx_4h-20):idx_4h+1]
        divergences['macd_4h'] = detect_macd_divergence(close_4h, macd_4h)
    else:
        divergences['macd_4h'] = 'NONE'

    details['divergences'] = divergences.copy()

    supporting = 0
    opposing = 0
    total = 0

    for tf, div in divergences.items():
        if div == 'NONE':
            continue
        total += 1
        if (direction == 'LONG' and div == 'BULLISH') or \
           (direction == 'SHORT' and div == 'BEARISH'):
            supporting += 1
        else:
            opposing += 1

    if total == 0:
        score = 0.50
        status = 'SKIP'
    elif supporting > 0 and opposing == 0:
        if supporting >= 3:
            score = 0.90
        elif supporting == 2:
            score = 0.78
        else:
            score = 0.68
        status = 'PASS'
    elif opposing > 0 and supporting == 0:
        if opposing >= 2:
            score = 0.15
        else:
            score = 0.30
        status = 'FAIL'
    else:
        score = 0.45
        status = 'FAIL'

    details['supporting'] = supporting
    details['opposing'] = opposing
    details['mtf_mom_score'] = round(score, 3)
    return status, score, details

"""M3: VWAP + Volume + Taker — Entry timing module."""

import numpy as np
import pandas as pd


def score_m3(df_15m, idx, direction, config):
    """M3: VWAP + Volume + Taker — soft scoring (no hard gate)."""
    if idx < config['VWAP_LOOKBACK']:
        return 'FAIL', 0.0, None

    row = df_15m.iloc[idx]
    vwap, close, volume = row['vwap'], row['Close'], row['Volume']
    vol_avg20, taker_ratio = row['vol_ma20'], row['taker_ratio']

    if pd.isna(vwap) or pd.isna(vol_avg20) or vol_avg20 == 0:
        return 'FAIL', 0.0, None

    vwap_dist = abs(close - vwap) / vwap

    in_zone = vwap_dist <= config['VWAP_ZONE_PCT']
    if in_zone:
        vwap_score = 1.0 - (vwap_dist / config['VWAP_ZONE_PCT'])
    else:
        vwap_score = max(0.0, 1.0 - (vwap_dist / (config['VWAP_ZONE_PCT'] * 2)))

    vol_ratio = volume / vol_avg20 if vol_avg20 > 0 else 0
    vol_score = min(vol_ratio / 2.0, 1.0)
    if vol_ratio < config['VOL_THRESHOLD']:
        vol_score *= 0.5

    if direction == 'LONG':
        taker_score = max(0.0, min((taker_ratio - 0.40) / 0.15, 1.0))
    else:
        taker_score = max(0.0, min((0.60 - taker_ratio) / 0.15, 1.0))

    combined = vwap_score * 0.4 + vol_score * 0.3 + taker_score * 0.3
    return 'PASS', combined, close

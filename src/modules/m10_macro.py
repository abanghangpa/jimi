"""M10: Cross-Asset Macro Scorer (BTC.D / SPX / DXY proxy)."""

import pandas as pd
import numpy as np
from src.utils.data_handler import fetch_daily_ohlcv_ccxt


M10_CACHE_DIR = "/tmp/jimi_m10_cache"


def m10_prepare_data(df_15m):
    """Fetch cross-asset daily data aligned to backtest range."""
    start = df_15m['Open time'].iloc[0].normalize() - pd.Timedelta(days=120)
    end = df_15m['Open time'].iloc[-1].normalize() + pd.Timedelta(days=2)

    result = {}
    print("    Fetching BTC/USDT daily...")
    result['btc'] = fetch_daily_ohlcv_ccxt('BTC/USDT', start, end, cache_dir=M10_CACHE_DIR)

    print("    Fetching ETH/BTC daily...")
    result['ethbtc'] = fetch_daily_ohlcv_ccxt('ETH/BTC', start, end, cache_dir=M10_CACHE_DIR)

    if result['btc'] is not None and result['ethbtc'] is not None and len(result['btc']) > 0 and len(result['ethbtc']) > 0:
        btc_d = result['btc'].set_index('date')['close']
        ethbtc_d = result['ethbtc'].set_index('date')['close']
        combined = pd.DataFrame({'btc': btc_d, 'ethbtc': ethbtc_d}).dropna()
        if len(combined) > 0:
            result['btc_dom'] = combined[['ethbtc']].rename(columns={'ethbtc': 'close'})
            result['btc_dom'] = result['btc_dom'].reset_index()

    return result


def m10_get_row(data_dict, timestamp):
    """Forward-fill lookup for macro data at a given timestamp."""
    date = timestamp.normalize()
    result = {}
    for key, df in data_dict.items():
        if df is not None and len(df) > 0 and 'date' in df.columns:
            m = df[df['date'] <= date]
            if len(m) > 0:
                result[key] = m.iloc[-1].to_dict()
    return result


def m10_compute_emas(data_dict):
    """Pre-compute EMAs and ROCs on macro daily data."""
    for key, df in data_dict.items():
        if df is not None and len(df) > 0 and 'close' in df.columns:
            df['ema21'] = df['close'].ewm(span=21, adjust=False).mean()
            df['ema55'] = df['close'].ewm(span=55, adjust=False).mean()
            df['roc_7d'] = df['close'].pct_change(7)
            df['roc_14d'] = df['close'].pct_change(14)
    return data_dict


def _directional_score(raw_value, direction, thresholds=(0.02, -0.02)):
    """Map a raw directional value to a 0.0-1.0 score with continuous scaling.

    Args:
        raw_value: the raw metric (e.g., EMA diff, ROC)
        direction: 'LONG' or 'SHORT'
        thresholds: (bullish_threshold, bearish_threshold) — values beyond these get max/min scores

    Returns:
        score: 0.0-1.0 where 0.5 is neutral
    """
    pos_thresh, neg_thresh = thresholds

    # For SHORT, invert the value so same logic applies
    if direction == 'SHORT':
        raw_value = -raw_value

    # Continuous mapping: linear interpolation between thresholds
    if raw_value >= pos_thresh:
        score = 0.80
    elif raw_value <= neg_thresh:
        score = 0.20
    elif raw_value >= 0:
        # 0 to pos_thresh → 0.50 to 0.80
        score = 0.50 + (raw_value / pos_thresh) * 0.30
    else:
        # neg_thresh to 0 → 0.20 to 0.50
        score = 0.50 + (raw_value / abs(neg_thresh)) * 0.30

    return max(0.0, min(1.0, score))


def score_m10_macro(macro_row, direction, trend_dir):
    """Score cross-asset macro regime for trade direction."""
    details = {}
    scores = {}

    btc_row = macro_row.get('btc')
    ethbtc_row = macro_row.get('ethbtc')

    # BTC trend: EMA21 vs EMA55 diff
    btc_trend_s = 0.5
    if btc_row and 'close' in btc_row:
        close = btc_row.get('close', 0)
        ema21 = btc_row.get('ema21', close)
        ema55 = btc_row.get('ema55', close)

        if ema21 and ema55 and ema55 > 0:
            ema_diff = (ema21 - ema55) / ema55
            btc_trend_s = _directional_score(ema_diff, direction, (0.02, -0.02))
            details['btc_ema_diff'] = round(ema_diff, 4)
    scores['btc_trend'] = btc_trend_s

    # ETH/BTC relative strength: 7d ROC
    ethbtc_s = 0.5
    if ethbtc_row and 'close' in ethbtc_row:
        ethbtc_roc7 = ethbtc_row.get('roc_7d', np.nan)
        if not np.isnan(ethbtc_roc7):
            ethbtc_s = _directional_score(ethbtc_roc7, direction, (0.03, -0.03))
            details['ethbtc_roc7'] = round(ethbtc_roc7, 4)
    scores['ethbtc_rel'] = ethbtc_s

    # BTC momentum: 7d ROC (with 14d confirmation bonus)
    btc_mom_s = 0.5
    if btc_row:
        btc_roc7 = btc_row.get('roc_7d', np.nan)
        btc_roc14 = btc_row.get('roc_14d', np.nan)
        if not np.isnan(btc_roc7):
            btc_mom_s = _directional_score(btc_roc7, direction, (0.05, -0.05))
            # 14d confirmation bonus: if 14d ROC agrees with 7d, boost slightly
            if not np.isnan(btc_roc14):
                if direction == 'LONG' and btc_roc14 > 0.08 and btc_roc7 > 0:
                    btc_mom_s = min(btc_mom_s + 0.05, 1.0)
                elif direction == 'SHORT' and btc_roc14 < -0.08 and btc_roc7 < 0:
                    btc_mom_s = min(btc_mom_s + 0.05, 1.0)
            details['btc_roc7'] = round(btc_roc7, 4)
    scores['btc_momentum'] = btc_mom_s

    # Consensus: how many sub-signals agree
    agreeing = sum(1 for s in scores.values() if s > 0.55)
    disagreeing = sum(1 for s in scores.values() if s < 0.45)
    total = len(scores)
    if total > 0:
        consensus = (agreeing - disagreeing) / total
        consensus_s = 0.5 + consensus * 0.30
    else:
        consensus_s = 0.5
    scores['consensus'] = consensus_s
    details['macro_agreement'] = f"{agreeing}/{total} agree"

    composite = (
        scores['btc_trend'] * 0.35 + scores['ethbtc_rel'] * 0.25 +
        scores['btc_momentum'] * 0.20 + scores['consensus'] * 0.20
    )
    composite = max(0.0, min(1.0, composite))

    status = 'PASS' if composite >= 0.48 else 'FAIL'
    details['m10_score'] = round(composite, 3)
    details['m10_components'] = {k: round(v, 3) for k, v in scores.items()}
    return status, composite, details

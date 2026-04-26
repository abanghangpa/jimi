"""
╔═══════════════════════════════════════════════════════════════════════╗
║ JIMI v6.16 BEAST MODULES                                              ║
║ 5 new direction-reading modules to add to v615_enhanced.py            ║
║                                                                       ║
║ M9:  Volatility Regime Classifier                                     ║
║ M10: Cross-Asset Macro Scorer (DXY/SPX/QQQ/BTC.D)                    ║
║ M11: Multi-TF Momentum Divergence                                     ║
║ M12: Order Book Imbalance (live only)                                 ║
║ ADAPTIVE: Probabilistic Direction Bias (replaces binary trend filter) ║
╚═══════════════════════════════════════════════════════════════════════╝
"""

import pandas as pd
import numpy as np
import requests
import time
import json
import os
import ccxt


# ═══════════════════════════════════════════════════════════════════════
# M9: VOLATILITY REGIME CLASSIFIER
# ═══════════════════════════════════════════════════════════════════════
#
# Classifies each bar into one of 4 volatility regimes:
#   TRENDING  — expanding range, directional moves → trade with trend
#   COMPRESSING — tightening range → breakout imminent, wait
#   CHOP      — random oscillation → reduce size or skip
#   CRISIS    — extreme expansion → widen stops or sit out
#
# Uses: Bollinger Band width, ATR percentile, price structure, volume

def compute_vol_regime(df_15m, df_1h, idx_15m, idx_1h):
    """
    Compute volatility regime for current bar.
    Returns: (regime: str, vol_regime_score: float, details: dict)
    
    vol_regime_score: 0.0 (worst) to 1.0 (best) for trading
    """
    details = {}
    
    if idx_1h < 20:
        return 'UNKNOWN', 0.5, details
    
    close_1h = df_1h['Close'].iloc[max(0, idx_1h-60):idx_1h+1]
    high_1h = df_1h['High'].iloc[max(0, idx_1h-60):idx_1h+1]
    low_1h = df_1h['Low'].iloc[max(0, idx_1h-60):idx_1h+1]
    
    if len(close_1h) < 20:
        return 'UNKNOWN', 0.5, details
    
    # --- 1. Bollinger Band Width (20-period, 1H) ---
    bb_sma = close_1h.rolling(20).mean()
    bb_std = close_1h.rolling(20).std()
    bb_width = (2 * bb_std / bb_sma)  # normalized width
    bb_width_current = bb_width.iloc[-1] if not pd.isna(bb_width.iloc[-1]) else 0.02
    
    # BB width percentile over last 60 bars
    bb_width_series = bb_width.dropna()
    if len(bb_width_series) >= 20:
        bb_pctl = (bb_width_series.iloc[-1] - bb_width_series.min()) / (bb_width_series.max() - bb_width_series.min() + 1e-10)
    else:
        bb_pctl = 0.5
    
    details['bb_width'] = round(bb_width_current, 5)
    details['bb_pctl'] = round(bb_pctl, 3)
    
    # --- 2. ATR Percentile (14-period, 1H) ---
    if 'atr' in df_1h.columns:
        atr_1h = df_1h['atr'].iloc[idx_1h]
        atr_series = df_1h['atr'].iloc[max(0, idx_1h-180):idx_1h+1].dropna()
        if len(atr_series) >= 20 and atr_1h > 0:
            atr_pctl = (atr_1h - atr_series.min()) / (atr_series.max() - atr_series.min() + 1e-10)
        else:
            atr_pctl = 0.5
    else:
        tr1 = high_1h - low_1h
        tr2 = (high_1h - close_1h.shift(1)).abs()
        tr3 = (low_1h - close_1h.shift(1)).abs()
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        atr_1h_series = tr.ewm(span=14, adjust=False).mean()
        atr_1h = atr_1h_series.iloc[-1] if not pd.isna(atr_1h_series.iloc[-1]) else 0
        atr_pctl = 0.5
    
    details['atr_pctl'] = round(atr_pctl, 3)
    
    # --- 3. Price Structure: directional vs mean-reverting ---
    # Count directional bars (close > open for longs, close < open for shorts)
    # over last 20 bars
    recent_20 = close_1h.iloc[-20:]
    directional_bars = ((recent_20.diff() > 0).sum() + (recent_20.diff() < 0).sum())
    # In a trending market, most bars are directional; in chop, ~50/50
    directionality = abs(directional_bars - 10) / 10  # 0 = pure random, 1 = pure trend
    details['directionality'] = round(directionality, 3)
    
    # --- 4. Volume expansion/contraction ---
    if 'Volume' in df_1h.columns:
        vol_ma20 = df_1h['Volume'].iloc[max(0, idx_1h-20):idx_1h+1].mean()
        vol_current = df_1h['Volume'].iloc[idx_1h]
        vol_ratio = vol_current / vol_ma20 if vol_ma20 > 0 else 1.0
    else:
        vol_ratio = 1.0
    details['vol_ratio'] = round(vol_ratio, 3)
    
    # --- 5. Higher highs / lower lows (trend structure) ---
    if idx_1h >= 10:
        highs = high_1h.iloc[-10:].values
        lows = low_1h.iloc[-10:].values
        hh_count = sum(1 for i in range(1, len(highs)) if highs[i] > highs[i-1])
        ll_count = sum(1 for i in range(1, len(lows)) if lows[i] < lows[i-1])
        structure_score = abs(hh_count - ll_count) / max(hh_count + ll_count, 1)
    else:
        structure_score = 0.0
    details['structure_score'] = round(structure_score, 3)
    
    # --- Composite Regime Classification ---
    # Score: higher = more tradeable
    trend_score = (
        directionality * 0.30 +
        structure_score * 0.25 +
        min(vol_ratio, 2.0) / 2.0 * 0.20 +
        (1.0 - bb_pctl) * 0.15 +  # lower BB width = better (less noise)
        (1.0 - abs(atr_pctl - 0.5) * 2) * 0.10  # mid-range ATR = best
    )
    
    # Classify
    if atr_pctl > 0.90 or bb_pctl > 0.90:
        regime = 'CRISIS'
        score = 0.15  # very bad for trading
    elif bb_pctl < 0.25 and atr_pctl < 0.35:
        regime = 'COMPRESSING'
        score = 0.40  # wait for breakout
    elif directionality > 0.4 and structure_score > 0.3 and vol_ratio > 0.8:
        regime = 'TRENDING'
        score = 0.80  # ideal for trend-following
    elif directionality < 0.2 and bb_pctl > 0.4:
        regime = 'CHOP'
        score = 0.25  # avoid or reduce size
    else:
        regime = 'NEUTRAL'
        score = 0.50
    
    details['regime'] = regime
    details['vol_regime_score'] = round(score, 3)
    
    return regime, score, details


def score_vol_regime(regime, vol_regime_score, direction, trend_dir):
    """
    Score volatility regime for trade direction.
    Returns: (status, score, details)
    """
    details = {'regime': regime}
    
    # Base score from regime classification
    base = vol_regime_score
    
    # Alignment bonus: trending regime + trade-with-trend = best
    if regime == 'TRENDING':
        if (trend_dir in ('STRONG_UP', 'UP') and direction == 'LONG') or \
           (trend_dir in ('STRONG_DOWN', 'DOWN') and direction == 'SHORT'):
            base = min(base * 1.15, 1.0)
        elif (trend_dir in ('STRONG_UP', 'UP') and direction == 'SHORT') or \
             (trend_dir in ('STRONG_DOWN', 'DOWN') and direction == 'LONG'):
            base *= 0.70  # counter-trend in trending regime = risky
    
    # Chop penalty
    if regime == 'CHOP':
        base *= 0.60
    
    # Crisis penalty
    if regime == 'CRISIS':
        base *= 0.30
    
    # Compressing: slight penalty but allow (breakout could be coming)
    if regime == 'COMPRESSING':
        base *= 0.85
    
    score = max(0.0, min(1.0, base))
    status = 'PASS' if score >= 0.45 else 'FAIL'
    details['vr_score'] = round(score, 3)
    
    return status, score, details


# ═══════════════════════════════════════════════════════════════════════
# M10: CROSS-ASSET MACRO SCORER
# ═══════════════════════════════════════════════════════════════════════
#
# Fetches daily data for:
#   - BTC.D (Bitcoin Dominance) — from CoinGecko
#   - SPX proxy (SPY or similar)
#   - DXY proxy (via crypto inverse correlation)
#
# Computes a macro regime score based on:
#   1. BTC dominance trend (rising = risk-off for alts)
#   2. BTC trend (leading indicator for ETH)
#   3. ETH/BTC relative strength
#   4. Composite macro regime

M10_CACHE_DIR = "/tmp/jimi_m10_cache"


def fetch_daily_ohlcv_ccxt(symbol, start_date, end_date, exchange_id='binance'):
    """Fetch daily OHLCV using ccxt with file caching."""
    os.makedirs(M10_CACHE_DIR, exist_ok=True)
    safe = symbol.replace("/", "_").replace(":", "_")
    cache_file = os.path.join(M10_CACHE_DIR, f"{safe}_daily.json")
    
    start = pd.Timestamp(start_date)
    end = pd.Timestamp(end_date)
    
    # Check cache
    if os.path.exists(cache_file):
        try:
            cached = pd.read_json(cache_file)
            if len(cached) > 0:
                cached['date'] = pd.to_datetime(cached['date']).dt.normalize()
                if cached['date'].min() <= start and cached['date'].max() >= end:
                    return cached[(cached['date'] >= start) & (cached['date'] <= end)].reset_index(drop=True)
        except:
            pass
    
    exchange = getattr(ccxt, exchange_id)({'enableRateLimit': True})
    since_ms = int(start.timestamp() * 1000)
    until_ms = int(end.timestamp() * 1000)
    
    candles = []
    cur = since_ms
    retries = 0
    while cur < until_ms and retries < 3:
        try:
            raw = exchange.fetch_ohlcv(symbol, '1d', since=cur, limit=1000)
            if not raw:
                break
            for c in raw:
                ts = int(c[0])
                if ts >= until_ms:
                    break
                candles.append({
                    'date': pd.to_datetime(ts, unit='ms').normalize().isoformat(),
                    'open': float(c[1]), 'high': float(c[2]),
                    'low': float(c[3]), 'close': float(c[4]),
                    'volume': float(c[5]),
                })
            last = raw[-1][0]
            if last <= cur:
                break
            cur = last + 1
            retries = 0
        except Exception as e:
            retries += 1
            time.sleep(5 * retries)
    
    df = pd.DataFrame(candles)
    if len(df) > 0:
        df['date'] = pd.to_datetime(df['date']).dt.normalize()
        # Save cache
        try:
            df.to_json(cache_file, orient='records', date_format='iso')
        except:
            pass
    
    return df


def m10_prepare_data(df_15m):
    """
    Fetch cross-asset daily data aligned to backtest range.
    Returns dict of DataFrames: {btc, ethbtc, btc_dominance_proxy}
    """
    start = df_15m['Open time'].iloc[0].normalize() - pd.Timedelta(days=120)
    end = df_15m['Open time'].iloc[-1].normalize() + pd.Timedelta(days=2)
    
    result = {}
    
    # BTC/USDT (already have via M7, but fetch independently for caching)
    print("    Fetching BTC/USDT daily...")
    result['btc'] = fetch_daily_ohlcv_ccxt('BTC/USDT', start, end)
    
    # ETH/BTC
    print("    Fetching ETH/BTC daily...")
    result['ethbtc'] = fetch_daily_ohlcv_ccxt('ETH/BTC', start, end)
    
    # BTC Dominance proxy: use BTC/USDT total market cap ratio
    # Since we can't easily get total crypto market cap from ccxt,
    # we'll use BTC/ETH relative performance as a proxy
    # (when BTC outperforms ETH = rising BTC dominance)
    if result['btc'] is not None and result['ethbtc'] is not None and len(result['btc']) > 0 and len(result['ethbtc']) > 0:
        btc_d = result['btc'].set_index('date')['close']
        ethbtc_d = result['ethbtc'].set_index('date')['close']
        # BTC dominance proxy: BTC price / (ETH price * ETH/BTC ratio)
        # Simplified: just use BTC/ETH ratio as dominance proxy
        combined = pd.DataFrame({'btc': btc_d, 'ethbtc': ethbtc_d}).dropna()
        if len(combined) > 0:
            combined['btc_dom_proxy'] = combined['btc'] / (combined['btc'] / combined['ethbtc'])  # = ethbtc * btc / btc = ethbtc
            # Actually, simpler: ETH/BTC rising = ETH outperforming = alt season = BTC dominance falling
            # So ETH/BTC IS our dominance proxy (inverse)
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


def score_m10_macro(macro_row, direction, trend_dir):
    """
    Score cross-asset macro regime for trade direction.
    
    Components:
      1. BTC Trend (35%): EMA21 vs EMA55 on BTC daily
      2. ETH/BTC Relative (25%): is ETH outperforming BTC?
      3. BTC Momentum (20%): 7d/14d ROC
      4. Macro Consensus (20%): agreement across signals
    
    Returns: (status, score, details)
    """
    details = {}
    scores = {}
    
    btc_row = macro_row.get('btc')
    ethbtc_row = macro_row.get('ethbtc')
    
    # --- 1. BTC Trend (35%) ---
    btc_trend_s = 0.5
    if btc_row and 'close' in btc_row:
        # We need EMA data - compute from close if available
        close = btc_row.get('close', 0)
        # Use simple trend: price vs 21-day and 55-day
        # Since we only have one row, we'll use the EMA values if pre-computed
        # For now, use a simplified approach: check if we have ema fields
        ema21 = btc_row.get('ema21', close)
        ema55 = btc_row.get('ema55', close)
        
        if ema21 and ema55 and ema55 > 0:
            ema_diff = (ema21 - ema55) / ema55
            if direction == 'LONG':
                if ema_diff > 0.02:
                    btc_trend_s = 0.80
                elif ema_diff > 0:
                    btc_trend_s = 0.65
                elif ema_diff < -0.02:
                    btc_trend_s = 0.25
                else:
                    btc_trend_s = 0.40
            else:  # SHORT
                if ema_diff < -0.02:
                    btc_trend_s = 0.80
                elif ema_diff < 0:
                    btc_trend_s = 0.65
                elif ema_diff > 0.02:
                    btc_trend_s = 0.25
                else:
                    btc_trend_s = 0.40
            details['btc_ema_diff'] = round(ema_diff, 4)
    scores['btc_trend'] = btc_trend_s
    
    # --- 2. ETH/BTC Relative Strength (25%) ---
    ethbtc_s = 0.5
    if ethbtc_row and 'close' in ethbtc_row:
        ethbtc_close = ethbtc_row.get('close', 0)
        ethbtc_roc7 = ethbtc_row.get('roc_7d', np.nan)
        
        if not np.isnan(ethbtc_roc7):
            if direction == 'LONG':
                # ETH outperforming BTC = good for longs
                if ethbtc_roc7 > 0.03:
                    ethbtc_s = 0.80
                elif ethbtc_roc7 > 0:
                    ethbtc_s = 0.65
                elif ethbtc_roc7 < -0.03:
                    ethbtc_s = 0.25
                else:
                    ethbtc_s = 0.40
            else:
                # ETH underperforming BTC = good for shorts
                if ethbtc_roc7 < -0.03:
                    ethbtc_s = 0.80
                elif ethbtc_roc7 < 0:
                    ethbtc_s = 0.65
                elif ethbtc_roc7 > 0.03:
                    ethbtc_s = 0.25
                else:
                    ethbtc_s = 0.40
            details['ethbtc_roc7'] = round(ethbtc_roc7, 4)
    scores['ethbtc_rel'] = ethbtc_s
    
    # --- 3. BTC Momentum (20%) ---
    btc_mom_s = 0.5
    if btc_row:
        btc_roc7 = btc_row.get('roc_7d', np.nan)
        btc_roc14 = btc_row.get('roc_14d', np.nan)
        
        if not np.isnan(btc_roc7):
            if direction == 'LONG':
                if btc_roc7 > 0.05 and (np.isnan(btc_roc14) or btc_roc14 > 0.08):
                    btc_mom_s = 0.85
                elif btc_roc7 > 0.02:
                    btc_mom_s = 0.68
                elif btc_roc7 < -0.05:
                    btc_mom_s = 0.20
                elif btc_roc7 < -0.02:
                    btc_mom_s = 0.35
            else:
                if btc_roc7 < -0.05 and (np.isnan(btc_roc14) or btc_roc14 < -0.08):
                    btc_mom_s = 0.85
                elif btc_roc7 < -0.02:
                    btc_mom_s = 0.68
                elif btc_roc7 > 0.05:
                    btc_mom_s = 0.20
                elif btc_roc7 > 0.02:
                    btc_mom_s = 0.35
            details['btc_roc7'] = round(btc_roc7, 4)
    scores['btc_momentum'] = btc_mom_s
    
    # --- 4. Macro Consensus (20%) ---
    # How many components agree with direction?
    agreeing = sum(1 for s in scores.values() if s > 0.55)
    disagreeing = sum(1 for s in scores.values() if s < 0.45)
    total = len(scores)
    
    if total > 0:
        consensus = (agreeing - disagreeing) / total  # -1 to +1
        consensus_s = 0.5 + consensus * 0.30  # 0.20 to 0.80
    else:
        consensus_s = 0.5
    scores['consensus'] = consensus_s
    details['macro_agreement'] = f"{agreeing}/{total} agree"
    
    # --- Composite ---
    composite = (
        scores['btc_trend'] * 0.35 +
        scores['ethbtc_rel'] * 0.25 +
        scores['btc_momentum'] * 0.20 +
        scores['consensus'] * 0.20
    )
    composite = max(0.0, min(1.0, composite))
    
    status = 'PASS' if composite >= 0.48 else 'FAIL'
    details['m10_score'] = round(composite, 3)
    details['m10_components'] = {k: round(v, 3) for k, v in scores.items()}
    
    return status, composite, details


# ═══════════════════════════════════════════════════════════════════════
# M11: MULTI-TF MOMENTUM DIVERGENCE
# ═══════════════════════════════════════════════════════════════════════
#
# Checks RSI and MACD divergence across 15m, 1H, 4H timeframes.
# The highest-conviction direction signal: when multiple timeframes agree
# on momentum divergence simultaneously.

def detect_rsi_divergence(close_series, rsi_series, lookback=20, min_bars=5):
    """
    Detect RSI divergence on a single timeframe.
    Returns: 'BULLISH', 'BEARISH', or 'NONE'
    
    Bullish: price makes lower low, RSI makes higher low
    Bearish: price makes higher high, RSI makes lower high
    """
    if len(close_series) < lookback + min_bars:
        return 'NONE'
    
    close = close_series.iloc[-lookback:].values
    rsi = rsi_series.iloc[-lookback:].values
    
    if np.any(np.isnan(rsi)):
        return 'NONE'
    
    # Find recent swing lows (for bullish divergence)
    for i in range(len(close) - min_bars, len(close) - 1):
        # Is this a local low?
        if i >= 2 and close[i] <= close[i-1] and close[i] <= close[i-2]:
            # Find previous low
            for j in range(max(0, i - lookback//2), i - min_bars + 1):
                if j >= 2 and close[j] <= close[j-1] and close[j] <= close[j-2]:
                    # Bullish divergence: price lower low, RSI higher low
                    if close[i] < close[j] * 0.998 and rsi[i] > rsi[j] + 2:
                        return 'BULLISH'
    
    # Find recent swing highs (for bearish divergence)
    for i in range(len(close) - min_bars, len(close) - 1):
        if i >= 2 and close[i] >= close[i-1] and close[i] >= close[i-2]:
            for j in range(max(0, i - lookback//2), i - min_bars + 1):
                if j >= 2 and close[j] >= close[j-1] and close[j] >= close[j-2]:
                    # Bearish divergence: price higher high, RSI lower high
                    if close[i] > close[j] * 1.002 and rsi[i] < rsi[j] - 2:
                        return 'BEARISH'
    
    return 'NONE'


def detect_macd_divergence(close_series, macd_hist_series, lookback=20, min_bars=5):
    """
    Detect MACD histogram divergence.
    Returns: 'BULLISH', 'BEARISH', or 'NONE'
    """
    if len(close_series) < lookback + min_bars:
        return 'NONE'
    
    close = close_series.iloc[-lookback:].values
    macd = macd_hist_series.iloc[-lookback:].values
    
    if np.any(np.isnan(macd)):
        return 'NONE'
    
    # Bullish: price makes lower low, MACD histogram makes higher low (less negative)
    for i in range(len(close) - min_bars, len(close) - 1):
        if i >= 2 and close[i] <= close[i-1] and close[i] <= close[i-2]:
            for j in range(max(0, i - lookback//2), i - min_bars + 1):
                if j >= 2 and close[j] <= close[j-1] and close[j] <= close[j-2]:
                    if close[i] < close[j] * 0.998 and macd[i] > macd[j]:
                        return 'BULLISH'
    
    # Bearish: price makes higher high, MACD histogram makes lower high (less positive)
    for i in range(len(close) - min_bars, len(close) - 1):
        if i >= 2 and close[i] >= close[i-1] and close[i] >= close[i-2]:
            for j in range(max(0, i - lookback//2), i - min_bars + 1):
                if j >= 2 and close[j] >= close[j-1] and close[j] >= close[j-2]:
                    if close[i] > close[j] * 1.002 and macd[i] < macd[j]:
                        return 'BEARISH'
    
    return 'NONE'


def score_m11_mtf_momentum(df_15m, df_1h, df_4h, idx_15m, idx_1h, idx_4h, direction):
    """
    Score multi-timeframe momentum divergence.
    
    Checks RSI and MACD divergence on 15m, 1H, 4H.
    Agreement across timeframes = highest conviction.
    
    Returns: (status, score, details)
    """
    details = {}
    divergences = {}
    
    # --- 15m RSI divergence ---
    if idx_15m >= 30 and 'rsi' in df_15m.columns:
        rsi_15m = df_15m['rsi'].iloc[max(0, idx_15m-30):idx_15m+1]
        close_15m = df_15m['Close'].iloc[max(0, idx_15m-30):idx_15m+1]
        divergences['rsi_15m'] = detect_rsi_divergence(close_15m, rsi_15m)
    else:
        divergences['rsi_15m'] = 'NONE'
    
    # --- 1H RSI divergence ---
    if idx_1h >= 30 and 'rsi' in df_1h.columns:
        rsi_1h = df_1h['rsi'].iloc[max(0, idx_1h-30):idx_1h+1]
        close_1h = df_1h['Close'].iloc[max(0, idx_1h-30):idx_1h+1]
        divergences['rsi_1h'] = detect_rsi_divergence(close_1h, rsi_1h)
    else:
        divergences['rsi_1h'] = 'NONE'
    
    # --- 1H MACD divergence ---
    if idx_1h >= 30 and 'macd_hist' in df_1h.columns:
        macd_1h = df_1h['macd_hist'].iloc[max(0, idx_1h-30):idx_1h+1]
        close_1h = df_1h['Close'].iloc[max(0, idx_1h-30):idx_1h+1]
        divergences['macd_1h'] = detect_macd_divergence(close_1h, macd_1h)
    else:
        divergences['macd_1h'] = 'NONE'
    
    # --- 4H MACD divergence ---
    if idx_4h >= 20 and 'macd_hist' in df_4h.columns:
        macd_4h = df_4h['macd_hist'].iloc[max(0, idx_4h-20):idx_4h+1]
        close_4h = df_4h['Close'].iloc[max(0, idx_4h-20):idx_4h+1]
        divergences['macd_4h'] = detect_macd_divergence(close_4h, macd_4h)
    else:
        divergences['macd_4h'] = 'NONE'
    
    details['divergences'] = divergences.copy()
    
    # --- Score based on agreement ---
    # How many divergences support the trade direction?
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
        # No divergence detected — neutral
        score = 0.50
        status = 'SKIP'
    elif supporting > 0 and opposing == 0:
        # All divergences support direction — strong signal
        if supporting >= 3:
            score = 0.90  # 3+ TFs agree = very strong
        elif supporting == 2:
            score = 0.78
        else:
            score = 0.68
        status = 'PASS'
    elif opposing > 0 and supporting == 0:
        # All divergences oppose direction
        if opposing >= 2:
            score = 0.15
        else:
            score = 0.30
        status = 'FAIL'
    else:
        # Mixed signals
        score = 0.45
        status = 'FAIL'
    
    details['supporting'] = supporting
    details['opposing'] = opposing
    details['mtf_mom_score'] = round(score, 3)
    
    return status, score, details


# ═══════════════════════════════════════════════════════════════════════
# M12: ORDER BOOK IMBALANCE (LIVE ONLY)
# ═══════════════════════════════════════════════════════════════════════
#
# Fetches live order book from Binance and computes:
#   1. Bid/ask depth ratio at ±1% from mid price
#   2. Large order detection (walls)
#   3. Book imbalance delta over time
#
# NOTE: This module only works in live/scan mode, not backtest.
# For backtest, returns neutral score.

def fetch_order_book_imbalance(symbol='ETHUSDT', depth=20):
    """
    Fetch order book and compute imbalance metrics.
    Returns: (bid_ask_ratio, wall_info, imbalance_score)
    """
    try:
        r = requests.get(
            'https://api.binance.com/api/v3/depth',
            params={'symbol': symbol, 'limit': depth},
            timeout=5
        )
        r.raise_for_status()
        data = r.json()
        
        bids = [(float(p), float(q)) for p, q in data['bids']]
        asks = [(float(p), float(q)) for p, q in data['asks']]
        
        if not bids or not asks:
            return 1.0, {}, 0.5
        
        mid = (bids[0][0] + asks[0][0]) / 2
        pct_range = 0.01  # ±1%
        
        bid_vol = sum(q for p, q in bids if p >= mid * (1 - pct_range))
        ask_vol = sum(q for p, q in asks if p <= mid * (1 + pct_range))
        
        # Bid/ask ratio
        ba_ratio = bid_vol / ask_vol if ask_vol > 0 else 1.0
        
        # Large order detection (walls)
        avg_bid = bid_vol / len(bids) if bids else 0
        avg_ask = ask_vol / len(asks) if asks else 0
        
        bid_walls = [(p, q) for p, q in bids if q > avg_bid * 3 and p >= mid * (1 - pct_range)]
        ask_walls = [(p, q) for p, q in asks if q > avg_ask * 3 and p <= mid * (1 + pct_range)]
        
        wall_info = {
            'bid_walls': len(bid_walls),
            'ask_walls': len(ask_walls),
            'largest_bid': max((q for _, q in bid_walls), default=0),
            'largest_ask': max((q for _, q in ask_walls), default=0),
        }
        
        # Imbalance score: >1 = more buying pressure, <1 = more selling
        imbalance = ba_ratio
        
        return ba_ratio, wall_info, imbalance
        
    except Exception as e:
        return 1.0, {}, 0.5


def score_m12_orderbook(direction, live=True):
    """
    Score order book imbalance for trade direction.
    Returns: (status, score, details)
    """
    details = {}
    
    if not live:
        # Backtest mode: return neutral
        return 'SKIP', 0.5, {'mode': 'backtest_neutral'}
    
    ba_ratio, wall_info, imbalance = fetch_order_book_imbalance()
    
    details['bid_ask_ratio'] = round(ba_ratio, 4)
    details.update(wall_info)
    
    score = 0.5
    
    if direction == 'LONG':
        # More bids than asks = bullish
        if ba_ratio > 1.5:
            score = 0.75
        elif ba_ratio > 1.2:
            score = 0.65
        elif ba_ratio < 0.7:
            score = 0.30
        elif ba_ratio < 0.85:
            score = 0.40
        
        # Wall analysis: large bid wall below = support = bullish
        if wall_info.get('bid_walls', 0) > wall_info.get('ask_walls', 0):
            score = min(score + 0.05, 1.0)
        elif wall_info.get('ask_walls', 0) > wall_info.get('bid_walls', 0) + 1:
            score = max(score - 0.05, 0.0)
    
    else:  # SHORT
        # More asks than bids = bearish
        if ba_ratio < 0.67:
            score = 0.75
        elif ba_ratio < 0.83:
            score = 0.65
        elif ba_ratio > 1.5:
            score = 0.30
        elif ba_ratio > 1.2:
            score = 0.40
        
        # Large ask wall above = resistance = bearish
        if wall_info.get('ask_walls', 0) > wall_info.get('bid_walls', 0):
            score = min(score + 0.05, 1.0)
        elif wall_info.get('bid_walls', 0) > wall_info.get('ask_walls', 0) + 1:
            score = max(score - 0.05, 0.0)
    
    score = max(0.0, min(1.0, score))
    status = 'PASS' if score >= 0.50 else 'FAIL'
    details['ob_score'] = round(score, 3)
    
    return status, score, details


# ═══════════════════════════════════════════════════════════════════════
# ADAPTIVE DIRECTION BIAS
# ═══════════════════════════════════════════════════════════════════════
#
# Replaces the binary trend filter with a probabilistic direction score.
# Instead of "LONG or SHORT" based on a single trend signal, this computes
# a continuous direction confidence from multiple inputs:
#
#   1. Daily trend state (existing) → weighted 30%
#   2. 4H trend alignment → weighted 20%
#   3. Multi-TF EMA agreement → weighted 20%
#   4. Recent trade performance (momentum) → weighted 15%
#   5. Volatility regime context → weighted 15%
#
# Output: direction_bias: float (-1.0 = max short, +1.0 = max long)
#         direction_allowed: bool (whether to allow trade in given direction)

def compute_adaptive_direction(
    trend_dir, trend_score,      # daily trend
    ema_1h_fast, ema_1h_slow,    # 1H EMA
    ema_4h_fast, ema_4h_slow,    # 4H EMA
    ema_1d_fast, ema_1d_slow,    # 1D EMA
    vol_regime,                   # from M9
    recent_trades=None,           # last N trades for momentum
    direction='LONG'              # proposed direction
):
    """
    Compute adaptive direction bias.
    
    Returns: (bias: float, allowed: bool, details: dict)
    bias: -1.0 (max short) to +1.0 (max long)
    allowed: whether the proposed direction is viable
    """
    details = {}
    components = {}
    
    # --- 1. Daily Trend State (30%) ---
    trend_map = {
        'STRONG_UP': 0.8, 'UP': 0.4, 'NEUTRAL': 0.0,
        'DOWN': -0.4, 'STRONG_DOWN': -0.8
    }
    daily_val = trend_map.get(trend_dir, 0.0)
    # Weight by trend_score magnitude
    daily_bias = daily_val * min(abs(trend_score) / 0.5, 1.0)
    components['daily_trend'] = daily_bias
    
    # --- 2. 1H EMA Alignment (20%) ---
    if ema_1h_fast is not None and ema_1h_slow is not None and ema_1h_slow > 0:
        ema_1h_diff = (ema_1h_fast - ema_1h_slow) / ema_1h_slow
        ema_1h_val = np.clip(ema_1h_diff * 10, -1.0, 1.0)  # scale: 10% diff = max
    else:
        ema_1h_val = 0.0
    components['ema_1h'] = ema_1h_val
    
    # --- 3. 4H EMA Alignment (20%) ---
    if ema_4h_fast is not None and ema_4h_slow is not None and ema_4h_slow > 0:
        ema_4h_diff = (ema_4h_fast - ema_4h_slow) / ema_4h_slow
        ema_4h_val = np.clip(ema_4h_diff * 10, -1.0, 1.0)
    else:
        ema_4h_val = 0.0
    components['ema_4h'] = ema_4h_val
    
    # --- 4. Daily EMA Confirmation (15%) ---
    if ema_1d_fast is not None and ema_1d_slow is not None and ema_1d_slow > 0:
        ema_1d_diff = (ema_1d_fast - ema_1d_slow) / ema_1d_slow
        ema_1d_val = np.clip(ema_1d_diff * 10, -1.0, 1.0)
    else:
        ema_1d_val = 0.0
    components['ema_1d'] = ema_1d_val
    
    # --- 5. Recent Trade Momentum (15%) ---
    if recent_trades and len(recent_trades) >= 3:
        last_n = recent_trades[-min(8, len(recent_trades)):]
        long_pnl = sum(t.pnl_pct * t.size_pct for t in last_n if t.direction == 'LONG')
        short_pnl = sum(t.pnl_pct * t.size_pct for t in last_n if t.direction == 'SHORT')
        # If longs are winning, bias long; if shorts winning, bias short
        momentum = np.clip((long_pnl - short_pnl) * 5, -1.0, 1.0)
    else:
        momentum = 0.0
    components['momentum'] = momentum
    
    # --- Composite ---
    bias = (
        components['daily_trend'] * 0.30 +
        components['ema_1h'] * 0.20 +
        components['ema_4h'] * 0.20 +
        components['ema_1d'] * 0.15 +
        components['momentum'] * 0.15
    )
    bias = np.clip(bias, -1.0, 1.0)
    
    # --- Determine if direction is allowed ---
    # Minimum bias required to trade in a direction
    min_bias = CONFIG.get('ADAPTIVE_DIR_MIN_BIAS', 0.10) if 'CONFIG' in dir() else 0.10
    block_threshold = 0.60  # only block when bias is very strongly against

    if direction == 'LONG':
        allowed = bias >= -min_bias  # allow longs unless strongly bearish
        if bias < -block_threshold:
            allowed = False
    else:
        allowed = bias <= min_bias  # allow shorts unless strongly bullish
        if bias > block_threshold:
            allowed = False
    
    details['direction_bias'] = round(bias, 4)
    details['bias_components'] = {k: round(v, 4) for k, v in components.items()}
    details['bias_allowed'] = allowed
    
    return bias, allowed, details


# ═══════════════════════════════════════════════════════════════════════
# HELPER: Pre-compute EMA for macro data
# ═══════════════════════════════════════════════════════════════════════

def m10_compute_emas(data_dict):
    """Pre-compute EMAs and ROCs on macro daily data."""
    for key, df in data_dict.items():
        if df is not None and len(df) > 0 and 'close' in df.columns:
            df['ema21'] = df['close'].ewm(span=21, adjust=False).mean()
            df['ema55'] = df['close'].ewm(span=55, adjust=False).mean()
            df['roc_7d'] = df['close'].pct_change(7)
            df['roc_14d'] = df['close'].pct_change(14)
    return data_dict


# ═══════════════════════════════════════════════════════════════════════
# CONFIG ADDITIONS FOR NEW MODULES
# ═══════════════════════════════════════════════════════════════════════

BEAST_CONFIG = {
    # --- M9: Volatility Regime ---
    'M9_ENABLED': True,
    'M9_WEIGHT': 0.10,
    'M9_BLOCK_REGIMES': ['CRISIS'],        # regimes that block entries entirely
    'M9_SIZE_CHOP': 0.60,                  # size multiplier in CHOP
    'M9_SIZE_COMPRESSING': 0.80,           # size multiplier in COMPRESSING
    
    # --- M10: Cross-Asset Macro ---
    'M10_ENABLED': True,
    'M10_WEIGHT': 0.10,
    'M10_FETCH_ON_BACKTEST': True,         # fetch daily data during backtest
    
    # --- M11: Multi-TF Momentum Divergence ---
    'M11_ENABLED': True,
    'M11_WEIGHT': 0.12,
    'M11_RSI_PERIOD_15M': 14,
    'M11_RSI_PERIOD_1H': 14,
    'M11_REQUIRE_AGREEMENT': False,        # if True, need 2+ TFs to agree
    
    # --- M12: Order Book Imbalance ---
    'M12_ENABLED': True,
    'M12_WEIGHT': 0.05,                    # small weight (live only)
    'M12_LIVE_ONLY': True,                 # skip in backtest
    
    # --- Adaptive Direction Bias ---
    'ADAPTIVE_DIR_ENABLED': True,
    'ADAPTIVE_DIR_MIN_BIAS': 0.10,         # minimum bias to allow trade
    'ADAPTIVE_DIR_BLOCK_THRESHOLD': 0.40,  # block if bias opposes this much
    'ADAPTIVE_DIR_CHOP_BIAS': 0.15,        # stronger bias needed in chop
}

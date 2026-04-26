#!/usr/bin/env python3
"""
╔═══════════════════════════════════════════════════════════════════════╗
║ JIMI v6.14 — M7 Module: Market Regime + Relative Strength            ║
║                                                                       ║
║ All data sourced from Binance (no external API keys needed).          ║
║                                                                       ║
║ Signals:                                                              ║
║   1. ETH/BTC trend (21d/55d EMA) — altcoin rotation / risk appetite  ║
║   2. ETH/BTC momentum (7d/30d ROC) — accelerating rotation            ║
║   3. BTC volatility regime — BTC ATR vs historical → risk environment ║
║   4. Volume regime — ETH/USDT volume trend → market participation     ║
║   5. Cross-asset momentum — ETH vs BTC short-term relative move       ║
║                                                                       ║
║ Proxy Logic:                                                          ║
║   Rising ETH/BTC + low BTC vol = risk-on (favor LONG ETH)            ║
║   Falling ETH/BTC + high BTC vol = risk-off (favor SHORT ETH)        ║
║   This captures the same regime as USDT dominance without needing     ║
║   external API access.                                                ║
╚═══════════════════════════════════════════════════════════════════════╝
"""

import ccxt
import pandas as pd
import numpy as np
import os
import json
import time


CACHE_DIR = "/tmp/jimi_m7_cache"
exchange = ccxt.binance({"enableRateLimit": True})


def _ensure_cache_dir():
    os.makedirs(CACHE_DIR, exist_ok=True)


# ═══════════════════════════════════════════════════════════════════════
# DATA FETCHING — All from Binance
# ═══════════════════════════════════════════════════════════════════════

def _ensure_datetime_col(df, col="date"):
    """Ensure a column is datetime type."""
    if col in df.columns and not pd.api.types.is_datetime64_any_dtype(df[col]):
        df[col] = pd.to_datetime(df[col]).dt.normalize()
    return df


def fetch_daily_ohlcv(symbol, since_ms, until_ms):
    """Fetch daily OHLCV from Binance with caching."""
    _ensure_cache_dir()
    safe_name = symbol.replace("/", "_")
    cache_file = os.path.join(CACHE_DIR, f"{safe_name}_daily.json")

    if os.path.exists(cache_file):
        age = time.time() - os.path.getmtime(cache_file)
        if age < 86400:  # 24h cache
            with open(cache_file) as f:
                data = json.load(f)
                df = pd.DataFrame(data)
                df["date"] = pd.to_datetime(df["date"]).dt.normalize()
                return df

    all_candles = []
    current = since_ms
    while current < until_ms:
        try:
            raw = exchange.fetch_ohlcv(symbol, "1d", since=current, limit=1000)
        except Exception as e:
            print(f"  [M7] Fetch error for {symbol}: {e}, retrying...")
            time.sleep(5)
            raw = exchange.fetch_ohlcv(symbol, "1d", since=current, limit=1000)
        if not raw:
            break
        for c in raw:
            ts = int(c[0])
            if ts >= until_ms:
                break
            all_candles.append({
                "date": pd.to_datetime(ts, unit="ms").isoformat(),
                "open": float(c[1]), "high": float(c[2]),
                "low": float(c[3]), "close": float(c[4]),
                "volume": float(c[5]),
            })
        last_ts = raw[-1][0]
        if last_ts <= current:
            break
        current = last_ts + 1

    with open(cache_file, "w") as f:
        json.dump(all_candles, f)

    df = pd.DataFrame(all_candles)
    df["date"] = pd.to_datetime(df["date"]).dt.normalize()
    return df


# ═══════════════════════════════════════════════════════════════════════
# DATA PREPARATION
# ═══════════════════════════════════════════════════════════════════════

def prepare_m7_data(df_15m):
    """
    Fetch and prepare M7 data aligned to the 15m DataFrame's date range.
    Returns: (eth_btc_df, btc_usdt_df) — daily DataFrames with computed signals.
    """
    start_date = df_15m["Open time"].iloc[0].normalize()
    end_date = df_15m["Open time"].iloc[-1].normalize()
    buffer_start = start_date - pd.Timedelta(days=90)
    fetch_end = end_date + pd.Timedelta(days=2)

    since_ms = int(buffer_start.timestamp() * 1000)
    until_ms = int(fetch_end.timestamp() * 1000)

    print(f"\n[M7] Preparing regime data for {start_date.strftime('%Y-%m-%d')} → {end_date.strftime('%Y-%m-%d')}...")

    # Fetch ETH/BTC
    print("  [M7] Fetching ETH/BTC daily from Binance...")
    ethbtc_df = fetch_daily_ohlcv("ETH/BTC", since_ms, until_ms)
    if len(ethbtc_df) > 0:
        ethbtc_df = _ensure_datetime_col(ethbtc_df)
        ethbtc_df = _compute_ethbtc_signals(ethbtc_df)
        ethbtc_df = ethbtc_df[(ethbtc_df["date"] >= buffer_start) & (ethbtc_df["date"] <= end_date)].reset_index(drop=True)
        print(f"  [M7] ETH/BTC: {len(ethbtc_df)} days, latest={ethbtc_df['close'].iloc[-1]:.6f}")

    # Fetch BTC/USDT
    print("  [M7] Fetching BTC/USDT daily from Binance...")
    btc_df = fetch_daily_ohlcv("BTC/USDT", since_ms, until_ms)
    if len(btc_df) > 0:
        btc_df = _ensure_datetime_col(btc_df)
        btc_df = _compute_btc_volatility_signals(btc_df)
        btc_df = btc_df[(btc_df["date"] >= buffer_start) & (btc_df["date"] <= end_date)].reset_index(drop=True)
        print(f"  [M7] BTC/USDT: {len(btc_df)} days, ATR14={btc_df['atr14'].iloc[-1]:.0f}")

    return ethbtc_df, btc_df


def _compute_ethbtc_signals(df):
    """Compute ETH/BTC technical signals."""
    df = df.sort_values("date").reset_index(drop=True)
    df["ema21"] = df["close"].ewm(span=21, adjust=False).mean()
    df["ema55"] = df["close"].ewm(span=55, adjust=False).mean()
    df["roc_7d"] = df["close"].pct_change(7)
    df["roc_30d"] = df["close"].pct_change(30)

    # Trend state
    df["trend"] = "NEUTRAL"
    df.loc[df["ema21"] > df["ema55"], "trend"] = "BULL"
    df.loc[df["ema21"] < df["ema55"], "trend"] = "BEAR"

    # Distance from EMA (mean reversion signal)
    df["ema_dist"] = (df["close"] - df["ema55"]) / df["ema55"]

    # Z-score of price vs 90d mean
    df["zscore_90"] = (
        (df["close"] - df["close"].rolling(90).mean())
        / df["close"].rolling(90).std().replace(0, np.nan)
    )

    return df


def _compute_btc_volatility_signals(df):
    """Compute BTC volatility regime signals."""
    df = df.sort_values("date").reset_index(drop=True)

    # ATR
    tr1 = df["high"] - df["low"]
    tr2 = (df["high"] - df["close"].shift(1)).abs()
    tr3 = (df["low"] - df["close"].shift(1)).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    df["atr14"] = tr.ewm(span=14, adjust=False).mean()

    # ATR as % of price
    df["atr_pct"] = df["atr14"] / df["close"] * 100

    # ATR percentile (0-1, relative to 180d history)
    df["atr_percentile"] = df["atr_pct"].rolling(180).apply(
        lambda x: (x.iloc[-1] - x.min()) / (x.max() - x.min()) if x.max() > x.min() else 0.5,
        raw=False
    )

    # BTC trend
    df["ema21"] = df["close"].ewm(span=21, adjust=False).mean()
    df["ema55"] = df["close"].ewm(span=55, adjust=False).mean()
    df["trend"] = "NEUTRAL"
    df.loc[df["ema21"] > df["ema55"], "trend"] = "BULL"
    df.loc[df["ema21"] < df["ema55"], "trend"] = "BEAR"

    # BTC momentum
    df["roc_7d"] = df["close"].pct_change(7)
    df["roc_30d"] = df["close"].pct_change(30)

    return df


# ═══════════════════════════════════════════════════════════════════════
# M7 SCORING
# ═══════════════════════════════════════════════════════════════════════

def score_m7(ethbtc_row, btc_row, df_15m_row, direction):
    """
    Score M7: Market Regime + Relative Strength.

    Components:
    1. ETH/BTC Trend (30%): EMA21 vs EMA55 → altcoin rotation
    2. ETH/BTC Momentum (20%): 7d/30d ROC → rotation acceleration
    3. BTC Volatility Regime (20%): ATR percentile → risk environment
    4. Volume Regime (15%): ETH/USDT volume trend → participation
    5. Cross-Asset Momentum (15%): ETH vs BTC 7d performance → relative flow

    Returns: (status, score, details)
    """
    score = 0.5
    details = {}

    # --- Component 1: ETH/BTC Trend (30%) ---
    trend_score = 0.5
    if ethbtc_row is not None:
        ema21 = ethbtc_row.get("ema21", 0)
        ema55 = ethbtc_row.get("ema55", 0)
        trend = ethbtc_row.get("trend", "NEUTRAL")
        ema_dist = ethbtc_row.get("ema_dist", 0)

        if direction == "LONG":
            if trend == "BULL":
                trend_score = 0.72
                if not np.isnan(ema_dist) and ema_dist > 0.02:
                    trend_score = 0.82  # strong ETH outperformance
            elif trend == "BEAR":
                trend_score = 0.30
                if not np.isnan(ema_dist) and ema_dist < -0.03:
                    trend_score = 0.20  # ETH heavily underperforming
        else:  # SHORT
            if trend == "BEAR":
                trend_score = 0.72
                if not np.isnan(ema_dist) and ema_dist < -0.03:
                    trend_score = 0.82
            elif trend == "BULL":
                trend_score = 0.30
                if not np.isnan(ema_dist) and ema_dist > 0.02:
                    trend_score = 0.20

        details["eth_btc"] = round(ethbtc_row.get("close", 0), 6)
        details["eth_btc_trend"] = trend
        details["eth_btc_ema_dist"] = round(ema_dist * 100, 2) if not np.isnan(ema_dist) else None
        details["trend_score"] = round(trend_score, 3)

    # --- Component 2: ETH/BTC Momentum (20%) ---
    momentum_score = 0.5
    if ethbtc_row is not None:
        roc_7d = ethbtc_row.get("roc_7d", np.nan)
        roc_30d = ethbtc_row.get("roc_30d", np.nan)

        if not np.isnan(roc_7d) and not np.isnan(roc_30d):
            if direction == "LONG":
                # LONG: want ETH/BTC rising (ETH gaining on BTC)
                if roc_7d > 0.03 and roc_30d > 0.05:
                    momentum_score = 0.85  # strong rotation into ETH
                elif roc_7d > 0.01:
                    momentum_score = 0.68
                elif roc_7d < -0.03 and roc_30d < -0.05:
                    momentum_score = 0.20  # capital flowing to BTC
                elif roc_7d < -0.01:
                    momentum_score = 0.35
            else:  # SHORT
                if roc_7d < -0.03 and roc_30d < -0.05:
                    momentum_score = 0.85
                elif roc_7d < -0.01:
                    momentum_score = 0.68
                elif roc_7d > 0.03 and roc_30d > 0.05:
                    momentum_score = 0.20
                elif roc_7d > 0.01:
                    momentum_score = 0.35

        details["eth_btc_roc_7d"] = round(roc_7d * 100, 2) if not np.isnan(roc_7d) else None
        details["eth_btc_roc_30d"] = round(roc_30d * 100, 2) if not np.isnan(roc_30d) else None
        details["momentum_score"] = round(momentum_score, 3)

    # --- Component 3: BTC Volatility Regime (20%) ---
    vol_regime_score = 0.5
    if btc_row is not None:
        atr_pct = btc_row.get("atr_pct", np.nan)
        atr_pctl = btc_row.get("atr_percentile", np.nan)
        btc_trend = btc_row.get("trend", "NEUTRAL")

        if not np.isnan(atr_pctl):
            if direction == "LONG":
                # LONG: prefer low/medium volatility (not panic selling)
                if atr_pctl > 0.85:
                    vol_regime_score = 0.25  # extreme vol — dangerous for longs
                elif atr_pctl > 0.70:
                    vol_regime_score = 0.35
                elif atr_pctl < 0.30:
                    vol_regime_score = 0.70  # calm market, favorable
                elif atr_pctl < 0.50:
                    vol_regime_score = 0.60

                # BTC trend matters: BTC dumping = bad for all alts
                if btc_trend == "BEAR":
                    vol_regime_score *= 0.8
                elif btc_trend == "BULL":
                    vol_regime_score = min(vol_regime_score * 1.15, 1.0)
            else:  # SHORT
                # SHORT: prefer higher volatility (panic = opportunity)
                if atr_pctl > 0.85:
                    vol_regime_score = 0.75  # high vol — great for shorts
                elif atr_pctl > 0.70:
                    vol_regime_score = 0.65
                elif atr_pctl < 0.30:
                    vol_regime_score = 0.35  # calm market, shorts less useful
                elif atr_pctl < 0.50:
                    vol_regime_score = 0.40

                if btc_trend == "BULL":
                    vol_regime_score *= 0.8
                elif btc_trend == "BEAR":
                    vol_regime_score = min(vol_regime_score * 1.15, 1.0)

        details["btc_atr_pct"] = round(atr_pct, 2) if not np.isnan(atr_pct) else None
        details["btc_atr_percentile"] = round(atr_pctl, 3) if not np.isnan(atr_pctl) else None
        details["btc_trend"] = btc_trend
        details["vol_regime_score"] = round(vol_regime_score, 3)

    # --- Component 4: Volume Regime (15%) ---
    vol_score = 0.5
    if df_15m_row is not None:
        vol_ratio = df_15m_row.get("vol_ratio", np.nan)
        if not np.isnan(vol_ratio):
            if vol_ratio > 1.3:
                vol_score = 0.75  # expanding volume — strong participation
            elif vol_ratio > 1.0:
                vol_score = 0.60
            elif vol_ratio < 0.5:
                vol_score = 0.25  # dead volume — risky
            elif vol_ratio < 0.7:
                vol_score = 0.38
        details["vol_ratio"] = round(vol_ratio, 3) if not np.isnan(vol_ratio) else None
        details["vol_score"] = round(vol_score, 3)

    # --- Component 5: Cross-Asset Momentum (15%) ---
    cross_score = 0.5
    if ethbtc_row is not None and btc_row is not None:
        eth_roc = ethbtc_row.get("roc_7d", np.nan)
        btc_roc = btc_row.get("roc_7d", np.nan)

        if not np.isnan(eth_roc) and not np.isnan(btc_roc):
            # ETH/BTC ROC captures ETH vs BTC relative, but we also want
            # absolute ETH direction: ETH/BTC up * BTC/USDT up = ETH pumping
            # ETH/BTC up * BTC/USDT down = ETH just falling less
            # ETH/BTC down * BTC/USDT up = ETH getting destroyed

            # Combined: ETH/BTC relative + BTC absolute
            if direction == "LONG":
                if eth_roc > 0 and btc_roc > 0:
                    cross_score = 0.80  # both up = strong bull
                elif eth_roc > 0 and btc_roc < -0.02:
                    cross_score = 0.55  # ETH holding while BTC dumps — mixed
                elif eth_roc < -0.02 and btc_roc > 0:
                    cross_score = 0.25  # ETH losing while BTC pumps — very bearish for ETH
                elif eth_roc < -0.02 and btc_roc < -0.02:
                    cross_score = 0.30  # both dumping
            else:  # SHORT
                if eth_roc < 0 and btc_roc < 0:
                    cross_score = 0.80  # both down = strong bear
                elif eth_roc < 0 and btc_roc > 0.02:
                    cross_score = 0.55
                elif eth_roc > 0.02 and btc_roc < 0:
                    cross_score = 0.25
                elif eth_roc > 0.02 and btc_roc > 0.02:
                    cross_score = 0.30

        details["eth_roc_7d"] = round(eth_roc * 100, 2) if not np.isnan(eth_roc) else None
        details["btc_roc_7d"] = round(btc_roc * 100, 2) if not np.isnan(btc_roc) else None
        details["cross_score"] = round(cross_score, 3)

    # --- Composite M7 Score ---
    composite = (
        trend_score * 0.30 +
        momentum_score * 0.20 +
        vol_regime_score * 0.20 +
        vol_score * 0.15 +
        cross_score * 0.15
    )
    composite = max(0.0, min(1.0, composite))

    # Regime label
    if composite >= 0.65:
        regime = "RISK_ON" if direction == "LONG" else "RISK_OFF_FAVOR"
    elif composite <= 0.35:
        regime = "RISK_OFF" if direction == "LONG" else "RISK_ON_FAVOR"
    else:
        regime = "NEUTRAL"

    details["composite"] = round(composite, 3)
    details["regime"] = regime

    status = "PASS" if composite >= 0.50 else "FAIL"
    return status, composite, details


# ═══════════════════════════════════════════════════════════════════════
# LOOKUP HELPERS
# ═══════════════════════════════════════════════════════════════════════

def get_m7_row_for_date(ethbtc_df, btc_df, timestamp):
    """Get M7 data rows for a given 15m timestamp (daily resolution, forward-fill)."""
    date = timestamp.normalize()

    ethbtc_row = None
    if ethbtc_df is not None and len(ethbtc_df) > 0:
        df = _ensure_datetime_col(ethbtc_df.copy())
        matches = df[df["date"] <= date]
        if len(matches) > 0:
            ethbtc_row = matches.iloc[-1].to_dict()

    btc_row = None
    if btc_df is not None and len(btc_df) > 0:
        df = _ensure_datetime_col(btc_df.copy())
        matches = df[df["date"] <= date]
        if len(matches) > 0:
            btc_row = matches.iloc[-1].to_dict()

    return ethbtc_row, btc_row


# ═══════════════════════════════════════════════════════════════════════
# STANDALONE TEST
# ═══════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("JIMI M7 Module — Standalone Test (Binance data)")
    print("=" * 50)

    import ccxt as _ccxt
    _exchange = _ccxt.binance({"enableRateLimit": True})
    now = pd.Timestamp.now(tz="UTC")
    since = int((now - pd.Timedelta(days=2000)).timestamp() * 1000)
    until = int(now.timestamp() * 1000)

    ethbtc_df = fetch_daily_ohlcv("ETH/BTC", since, until)
    btc_df = fetch_daily_ohlcv("BTC/USDT", since, until)

    if len(ethbtc_df) > 0:
        ethbtc_df = _compute_ethbtc_signals(ethbtc_df)
        print(f"\nETH/BTC (last 5 days):")
        print(ethbtc_df[["date", "close", "ema21", "ema55", "trend", "roc_7d"]].tail())

    if len(btc_df) > 0:
        btc_df = _compute_btc_volatility_signals(btc_df)
        print(f"\nBTC/USDT (last 5 days):")
        print(btc_df[["date", "close", "atr14", "atr_pct", "atr_percentile", "trend"]].tail())

    # Test scoring
    if len(ethbtc_df) > 60:
        ethbtc_row = ethbtc_df.iloc[-1].to_dict()
        btc_row = btc_df.iloc[-1].to_dict() if len(btc_df) > 0 else None
        for d in ["LONG", "SHORT"]:
            status, score, details = score_m7(ethbtc_row, btc_row, None, d)
            print(f"\n  {d}: {status} score={score:.3f} regime={details.get('regime')}")
            for k, v in details.items():
                if k != "regime":
                    print(f"    {k}: {v}")

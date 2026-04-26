#!/usr/bin/env python3
"""
╔═══════════════════════════════════════════════════════════════════════╗
║ JIMI FRAMEWORK v6.13 — Multi-Module Scoring System + Trend Filter     ║
║ ETH/USDT 15m + Market Regime (ETH/BTC + BTC Volatility)              ║
║                                                                       ║
║ Modules: M1(1H MACD) M2(Multi-TF EMA) M3(VWAP+Vol+Taker)            ║
║          M4(15m CVD) M5(Liquidation Magnet) M6(Derivatives)          ║
║          M7(Market Regime: ETH/BTC + BTC Vol + Volume)                ║
║          Trend Filter (Multi-signal daily trend detection)             ║
║                                                                       ║
║ v6.13 Changes:                                                        ║
║   - Multi-signal trend filter (EMA, ROC, RSI, price structure)        ║
║   - Hard directional filter: never trade against the trend            ║
║   - TP-before-SL fix: TP checked before SL on same bar               ║
║   - Adaptive R:R: SL=1.3 ATR, TP1=1.5 ATR (1.15x R:R)              ║
║   - Early exit: kill stale losing trades after 16 bars (4h)          ║
║   - Embedded M7: ETH/BTC trend, BTC volatility regime, volume regime  ║
║   - Seasonal risk controls (summer/shoulder months)                   ║
║   - All data from Binance (no external API keys)                      ║
║                                                                       ║
║ Usage:                                                                ║
║   python3 jimi_v613.py backtest eth_15m_data.csv                      ║
║   python3 jimi_v613.py scan                                           ║
╚═══════════════════════════════════════════════════════════════════════╝
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import sys
import os
import json
import requests
import time
import ccxt

HAS_DERIVATIVES = True

# ═══════════════════════════════════════════════════════════════════════
# DERIVATIVES DATA MODULE (inline)
# ═══════════════════════════════════════════════════════════════════════

import requests
import pandas as pd
import numpy as np
from datetime import datetime

BASE_URL = "https://fapi.binance.com"
PERIOD = "15m"
HISTORY_BARS = 96  # 24h of 15m bars


def fetch_oi_history(symbol="ETHUSDT", period=PERIOD, limit=HISTORY_BARS):
    """Open Interest history — sum across all contracts."""
    r = requests.get(f"{BASE_URL}/futures/data/openInterestHist",
                     params={"symbol": symbol, "period": period, "limit": limit})
    r.raise_for_status()
    data = r.json()
    rows = []
    for d in data:
        rows.append({
            "timestamp": pd.to_datetime(d["timestamp"], unit="ms"),
            "oi": float(d["sumOpenInterest"]),
            "oi_usd": float(d["sumOpenInterestValue"]),
        })
    return pd.DataFrame(rows)


def fetch_ls_ratio(symbol="ETHUSDT", period=PERIOD, limit=HISTORY_BARS):
    """Global Long/Short Account Ratio — all traders."""
    r = requests.get(f"{BASE_URL}/futures/data/globalLongShortAccountRatio",
                     params={"symbol": symbol, "period": period, "limit": limit})
    r.raise_for_status()
    data = r.json()
    rows = []
    for d in data:
        rows.append({
            "timestamp": pd.to_datetime(d["timestamp"], unit="ms"),
            "ls_ratio": float(d["longShortRatio"]),
            "long_pct": float(d["longAccount"]),
            "short_pct": float(d["shortAccount"]),
        })
    return pd.DataFrame(rows)


def fetch_top_trader_ls(symbol="ETHUSDT", period=PERIOD, limit=HISTORY_BARS):
    """Top Trader Long/Short Ratio — accounts with largest positions."""
    r = requests.get(f"{BASE_URL}/futures/data/topLongShortAccountRatio",
                     params={"symbol": symbol, "period": period, "limit": limit})
    r.raise_for_status()
    data = r.json()
    rows = []
    for d in data:
        rows.append({
            "timestamp": pd.to_datetime(d["timestamp"], unit="ms"),
            "top_ls_ratio": float(d["longShortRatio"]),
            "top_long_pct": float(d["longAccount"]),
            "top_short_pct": float(d["shortAccount"]),
        })
    return pd.DataFrame(rows)


def fetch_taker_ratio(symbol="ETHUSDT", period=PERIOD, limit=HISTORY_BARS):
    """Futures Taker Buy/Sell Volume Ratio."""
    r = requests.get(f"{BASE_URL}/futures/data/takerlongshortRatio",
                     params={"symbol": symbol, "period": period, "limit": limit})
    r.raise_for_status()
    data = r.json()
    rows = []
    for d in data:
        rows.append({
            "timestamp": pd.to_datetime(d["timestamp"], unit="ms"),
            "futures_taker_ratio": float(d["buySellRatio"]),
            "futures_buy_vol": float(d["buyVol"]),
            "futures_sell_vol": float(d["sellVol"]),
        })
    return pd.DataFrame(rows)


def fetch_funding_rate(symbol="ETHUSDT", limit=10):
    """Recent funding rate history."""
    r = requests.get(f"{BASE_URL}/fapi/v1/fundingRate",
                     params={"symbol": symbol, "limit": limit})
    r.raise_for_status()
    data = r.json()
    rows = []
    for d in data:
        rows.append({
            "timestamp": pd.to_datetime(d["fundingTime"], unit="ms"),
            "funding_rate": float(d["fundingRate"]),
            "mark_price": float(d["markPrice"]),
        })
    return pd.DataFrame(rows)


def fetch_all_derivatives(symbol="ETHUSDT"):
    """Fetch all derivatives data and merge into a single DataFrame."""
    oi = fetch_oi_history(symbol)
    ls = fetch_ls_ratio(symbol)
    top = fetch_top_trader_ls(symbol)
    taker = fetch_taker_ratio(symbol)
    funding = fetch_funding_rate(symbol)

    # Merge on timestamp
    df = oi.merge(ls, on="timestamp", how="outer")
    df = df.merge(top, on="timestamp", how="outer")
    df = df.merge(taker, on="timestamp", how="outer")
    df = df.sort_values("timestamp").reset_index(drop=True)

    # Forward-fill any gaps
    df = df.ffill()

    return df, funding


# ═══════════════════════════════════════════════════════════════════════
# SIGNAL MODULES — Derivatives-based
# ═══════════════════════════════════════════════════════════════════════

def compute_oi_signals(df_deriv, df_15m=None):
    """
    OI-based signals:
    1. OI divergence — price up but OI down (or vice versa)
    2. OI spike — sudden large OI change (new positions opening)
    3. OI unwind — rapid OI drop (positions closing / liquidations)
    """
    df = df_deriv.copy()

    # OI rate of change (4-bar = 1h)
    df["oi_roc_1h"] = df["oi"].pct_change(4)
    # OI rate of change (8-bar = 2h)
    df["oi_roc_2h"] = df["oi"].pct_change(8)

    # OI spike detection (>2% in 1h)
    df["oi_spike"] = df["oi_roc_1h"].abs() > 0.02

    # OI divergence with price (if price data available)
    if df_15m is not None:
        # Align price data to derivatives timestamps
        price_col = df_15m.set_index("Open time")["Close"]
        df = df.set_index("timestamp")
        df["price"] = price_col.reindex(df.index, method="ffill")
        df = df.reset_index()

        # Price ROC
        df["price_roc_1h"] = df["price"].pct_change(4)
        df["price_roc_2h"] = df["price"].pct_change(8)

        # Divergence: price up + OI down = bearish (longs closing)
        #            price down + OI up = bullish (shorts opening into support)
        df["oi_price_div"] = "NONE"
        mask_bear = (df["price_roc_1h"] > 0.005) & (df["oi_roc_1h"] < -0.01)
        mask_bull = (df["price_roc_1h"] < -0.005) & (df["oi_roc_1h"] > 0.01)
        df.loc[mask_bear, "oi_price_div"] = "BEARISH"
        df.loc[mask_bull, "oi_price_div"] = "BULLISH"

    return df


def compute_positioning_signals(df_deriv):
    """
    Positioning-based signals:
    1. L/S ratio extreme — crowded long/short
    2. Top trader divergence — top traders vs retail
    3. Funding rate context — positive/negative funding
    """
    df = df_deriv.copy()

    # L/S ratio z-score (how extreme is current positioning)
    ls_mean = df["ls_ratio"].rolling(48).mean()
    ls_std = df["ls_ratio"].rolling(48).std()
    df["ls_zscore"] = (df["ls_ratio"] - ls_mean) / ls_std.replace(0, np.nan)

    # Positioning trap detection
    df["positioning"] = "NEUTRAL"
    # Crowded long (z-score > 1.5) — vulnerable to short squeeze down
    df.loc[df["ls_zscore"] > 1.5, "positioning"] = "CROWDED_LONG"
    # Crowded short (z-score < -1.5) — vulnerable to short squeeze up
    df.loc[df["ls_zscore"] < -1.5, "positioning"] = "CROWDED_SHORT"

    # Top trader vs retail divergence
    if "top_ls_ratio" in df.columns:
        df["whale_retail_gap"] = df["top_ls_ratio"] - df["ls_ratio"]
        # Whale more bullish than retail = potential accumulation
        # Whale more bearish than retail = potential distribution
        df["whale_signal"] = "NEUTRAL"
        df.loc[df["whale_retail_gap"] > 0.3, "whale_signal"] = "WHALE_BULLISH"
        df.loc[df["whale_retail_gap"] < -0.3, "whale_signal"] = "WHALE_BEARISH"

    # Futures taker ratio context
    if "futures_taker_ratio" in df.columns:
        taker_ma = df["futures_taker_ratio"].rolling(8).mean()
        df["futures_taker_ma"] = taker_ma
        df["futures_flow"] = "NEUTRAL"
        df.loc[taker_ma > 1.15, "futures_flow"] = "BUYERS_DOMINANT"
        df.loc[taker_ma < 0.85, "futures_flow"] = "SELLERS_DOMINANT"

    return df


def score_derivatives(df_deriv_latest, direction):
    """
    Score derivatives data for a given trade direction.
    Accepts dict, Series, or DataFrame row.

    Returns: (status, score, details)
    """
    if df_deriv_latest is None:
        return "SKIP", 0.5, {}

    # Normalize to dict-like access
    if isinstance(df_deriv_latest, dict):
        last = df_deriv_latest
    elif hasattr(df_deriv_latest, "iloc"):
        last = df_deriv_latest.iloc[-1] if hasattr(df_deriv_latest, "iloc") and len(df_deriv_latest) > 0 else df_deriv_latest
    else:
        last = df_deriv_latest

    def _get(key, default=None):
        if isinstance(last, dict):
            return last.get(key, default)
        try:
            val = last[key]
            return val if not (isinstance(val, float) and np.isnan(val)) else default
        except (KeyError, IndexError):
            return default

    details = {}
    score = 0.5  # neutral baseline

    # --- OI Divergence ---
    oi_div = _get("oi_price_div", "NONE")
    if direction == "LONG":
        if oi_div == "BULLISH":  # price down + OI up = shorts opening, potential squeeze
            score += 0.10
        elif oi_div == "BEARISH":  # price up + OI down = longs closing
            score -= 0.05
    elif direction == "SHORT":
        if oi_div == "BEARISH":  # price up + OI down = longs closing, weakness
            score += 0.10
        elif oi_div == "BULLISH":  # price down + OI up = shorts opening, risky
            score -= 0.05
    details["oi_div"] = oi_div

    # --- Positioning Trap ---
    positioning = _get("positioning", "NEUTRAL")
    if direction == "LONG" and positioning == "CROWDED_SHORT":
        score += 0.12  # crowded short = squeeze potential
    elif direction == "SHORT" and positioning == "CROWDED_LONG":
        score += 0.12  # crowded long = dump potential
    elif direction == "LONG" and positioning == "CROWDED_LONG":
        score -= 0.08  # crowded long = vulnerable
    elif direction == "SHORT" and positioning == "CROWDED_SHORT":
        score -= 0.08  # crowded short = vulnerable
    details["positioning"] = positioning
    details["ls_zscore"] = round(float(_get("ls_zscore", 0)), 3)

    # --- Whale Signal ---
    whale = _get("whale_signal", "NEUTRAL")
    if direction == "LONG" and whale == "WHALE_BULLISH":
        score += 0.08
    elif direction == "SHORT" and whale == "WHALE_BEARISH":
        score += 0.08
    elif direction == "LONG" and whale == "WHALE_BEARISH":
        score -= 0.05
    elif direction == "SHORT" and whale == "WHALE_BULLISH":
        score -= 0.05
    details["whale_signal"] = whale

    # --- Futures Taker Flow ---
    futures_flow = _get("futures_flow", "NEUTRAL")
    if direction == "LONG" and futures_flow == "BUYERS_DOMINANT":
        score += 0.08
    elif direction == "SHORT" and futures_flow == "SELLERS_DOMINANT":
        score += 0.08
    elif direction == "LONG" and futures_flow == "SELLERS_DOMINANT":
        score -= 0.05
    elif direction == "SHORT" and futures_flow == "BUYERS_DOMINANT":
        score -= 0.05
    details["futures_flow"] = futures_flow

    score = max(0.0, min(1.0, score))
    status = "PASS" if score >= 0.50 else "FAIL"
    details["deriv_score"] = round(score, 3)

    return status, score, details


# ═══════════════════════════════════════════════════════════════════════
# SCAN OUTPUT — For live scanner
# ═══════════════════════════════════════════════════════════════════════

def get_derivatives_summary(symbol="ETHUSDT"):
    """Fetch and summarize current derivatives state for scan output."""
    try:
        df_deriv, funding = fetch_all_derivatives(symbol)
        df_deriv = compute_oi_signals(df_deriv)
        df_deriv = compute_positioning_signals(df_deriv)

        last = df_deriv.iloc[-1]
        latest_funding = funding.iloc[-1] if not funding.empty else None

        summary = {
            "oi": round(float(last.get("oi", 0)), 0),
            "oi_usd": round(float(last.get("oi_usd", 0)), 0),
            "oi_roc_1h": round(float(last.get("oi_roc_1h", 0)) * 100, 3),
            "ls_ratio": round(float(last.get("ls_ratio", 0)), 4),
            "long_pct": round(float(last.get("long_pct", 0)) * 100, 1),
            "short_pct": round(float(last.get("short_pct", 0)) * 100, 1),
            "ls_zscore": round(float(last.get("ls_zscore", 0)), 3),
            "positioning": last.get("positioning", "NEUTRAL"),
            "top_ls_ratio": round(float(last.get("top_ls_ratio", 0)), 4),
            "whale_signal": last.get("whale_signal", "NEUTRAL"),
            "whale_retail_gap": round(float(last.get("whale_retail_gap", 0)), 4),
            "futures_taker_ratio": round(float(last.get("futures_taker_ratio", 0)), 4),
            "futures_flow": last.get("futures_flow", "NEUTRAL"),
            "funding_rate": round(float(latest_funding["funding_rate"]), 6) if latest_funding is not None else None,
            "oi_price_div": last.get("oi_price_div", "NONE"),
        }
        return summary
    except Exception as e:
        return {"error": str(e)}


# ═══════════════════════════════════════════════════════════════════════
# M7: MARKET REGIME MODULE — ETH/BTC + BTC Volatility + Volume
# ═══════════════════════════════════════════════════════════════════════

M7_CACHE_DIR = "/tmp/jimi_m7_cache"
_binance_exchange = None


def _get_binance():
    global _binance_exchange
    if _binance_exchange is None:
        _binance_exchange = ccxt.binance({"enableRateLimit": True})
    return _binance_exchange


def m7_fetch_daily(symbol, since_ms, until_ms):
    """Fetch daily OHLCV from Binance with file caching."""
    os.makedirs(M7_CACHE_DIR, exist_ok=True)
    safe = symbol.replace("/", "_")
    cache_file = os.path.join(M7_CACHE_DIR, f"{safe}_daily.json")

    if os.path.exists(cache_file):
        age = time.time() - os.path.getmtime(cache_file)
        if age < 86400:
            with open(cache_file) as f:
                data = json.load(f)
                df = pd.DataFrame(data)
                df["date"] = pd.to_datetime(df["date"]).dt.normalize()
                return df

    ex = _get_binance()
    candles = []
    cur = since_ms
    while cur < until_ms:
        try:
            raw = ex.fetch_ohlcv(symbol, "1d", since=cur, limit=1000)
        except Exception:
            time.sleep(5)
            raw = ex.fetch_ohlcv(symbol, "1d", since=cur, limit=1000)
        if not raw:
            break
        for c in raw:
            ts = int(c[0])
            if ts >= until_ms:
                break
            candles.append({
                "date": pd.to_datetime(ts, unit="ms").isoformat(),
                "open": float(c[1]), "high": float(c[2]),
                "low": float(c[3]), "close": float(c[4]),
                "volume": float(c[5]),
            })
        last = raw[-1][0]
        if last <= cur:
            break
        cur = last + 1

    with open(cache_file, "w") as f:
        json.dump(candles, f)
    df = pd.DataFrame(candles)
    df["date"] = pd.to_datetime(df["date"]).dt.normalize()
    return df


def m7_prepare_data(df_15m):
    """Fetch and compute M7 signals aligned to the 15m data range."""
    start = df_15m["Open time"].iloc[0].normalize() - pd.Timedelta(days=90)
    end = df_15m["Open time"].iloc[-1].normalize() + pd.Timedelta(days=2)
    since_ms = int(start.timestamp() * 1000)
    until_ms = int(end.timestamp() * 1000)

    # ETH/BTC
    ethbtc = m7_fetch_daily("ETH/BTC", since_ms, until_ms)
    if len(ethbtc) > 0:
        ethbtc["ema21"] = ethbtc["close"].ewm(span=21, adjust=False).mean()
        ethbtc["ema55"] = ethbtc["close"].ewm(span=55, adjust=False).mean()
        ethbtc["trend"] = "NEUTRAL"
        ethbtc.loc[ethbtc["ema21"] > ethbtc["ema55"], "trend"] = "BULL"
        ethbtc.loc[ethbtc["ema21"] < ethbtc["ema55"], "trend"] = "BEAR"
        ethbtc["ema_dist"] = (ethbtc["close"] - ethbtc["ema55"]) / ethbtc["ema55"]
        ethbtc["roc_7d"] = ethbtc["close"].pct_change(7)
        ethbtc["roc_30d"] = ethbtc["close"].pct_change(30)
        ethbtc = ethbtc[(ethbtc["date"] >= start) & (ethbtc["date"] <= end)].reset_index(drop=True)

    # BTC/USDT
    btc = m7_fetch_daily("BTC/USDT", since_ms, until_ms)
    if len(btc) > 0:
        tr1 = btc["high"] - btc["low"]
        tr2 = (btc["high"] - btc["close"].shift(1)).abs()
        tr3 = (btc["low"] - btc["close"].shift(1)).abs()
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        btc["atr14"] = tr.ewm(span=14, adjust=False).mean()
        btc["atr_pct"] = btc["atr14"] / btc["close"] * 100
        btc["atr_pctl"] = btc["atr_pct"].rolling(180).apply(
            lambda x: (x.iloc[-1] - x.min()) / (x.max() - x.min()) if x.max() > x.min() else 0.5,
            raw=False)
        btc["ema21"] = btc["close"].ewm(span=21, adjust=False).mean()
        btc["ema55"] = btc["close"].ewm(span=55, adjust=False).mean()
        btc["trend"] = "NEUTRAL"
        btc.loc[btc["ema21"] > btc["ema55"], "trend"] = "BULL"
        btc.loc[btc["ema21"] < btc["ema55"], "trend"] = "BEAR"
        btc["roc_7d"] = btc["close"].pct_change(7)
        btc = btc[(btc["date"] >= start) & (btc["date"] <= end)].reset_index(drop=True)

    return ethbtc, btc


def m7_get_row(ethbtc_df, btc_df, timestamp):
    """Forward-fill lookup for daily M7 data at a given timestamp."""
    date = timestamp.normalize()
    eb_row = None
    if ethbtc_df is not None and len(ethbtc_df) > 0:
        m = ethbtc_df[ethbtc_df["date"] <= date]
        if len(m) > 0:
            eb_row = m.iloc[-1].to_dict()
    bt_row = None
    if btc_df is not None and len(btc_df) > 0:
        m = btc_df[btc_df["date"] <= date]
        if len(m) > 0:
            bt_row = m.iloc[-1].to_dict()
    return eb_row, bt_row


def score_m7(ethbtc_row, btc_row, vol_ratio, direction):
    """
    Score M7: Market Regime.

    Components:
      1. ETH/BTC Trend (30%): EMA21 vs EMA55
      2. ETH/BTC Momentum (20%): 7d/30d ROC
      3. BTC Volatility Regime (20%): ATR percentile
      4. Volume Regime (15%): ETH/USDT vol ratio
      5. Cross-Asset Momentum (15%): ETH vs BTC relative

    Returns: (status, score, details)
    """
    details = {}

    # --- 1. ETH/BTC Trend (30%) ---
    trend_s = 0.5
    if ethbtc_row:
        trend = ethbtc_row.get("trend", "NEUTRAL")
        ema_dist = ethbtc_row.get("ema_dist", 0)
        if direction == "LONG":
            if trend == "BULL":
                trend_s = 0.82 if (not np.isnan(ema_dist) and ema_dist > 0.02) else 0.72
            elif trend == "BEAR":
                trend_s = 0.20 if (not np.isnan(ema_dist) and ema_dist < -0.03) else 0.30
        else:
            if trend == "BEAR":
                trend_s = 0.82 if (not np.isnan(ema_dist) and ema_dist < -0.03) else 0.72
            elif trend == "BULL":
                trend_s = 0.20 if (not np.isnan(ema_dist) and ema_dist > 0.02) else 0.30
        details["eth_btc_trend"] = trend

    # --- 2. ETH/BTC Momentum (20%) ---
    mom_s = 0.5
    if ethbtc_row:
        r7 = ethbtc_row.get("roc_7d", np.nan)
        r30 = ethbtc_row.get("roc_30d", np.nan)
        if not np.isnan(r7) and not np.isnan(r30):
            if direction == "LONG":
                if r7 > 0.03 and r30 > 0.05: mom_s = 0.85
                elif r7 > 0.01: mom_s = 0.68
                elif r7 < -0.03 and r30 < -0.05: mom_s = 0.20
                elif r7 < -0.01: mom_s = 0.35
            else:
                if r7 < -0.03 and r30 < -0.05: mom_s = 0.85
                elif r7 < -0.01: mom_s = 0.68
                elif r7 > 0.03 and r30 > 0.05: mom_s = 0.20
                elif r7 > 0.01: mom_s = 0.35

    # --- 3. BTC Volatility Regime (20%) ---
    vol_reg_s = 0.5
    if btc_row:
        pctl = btc_row.get("atr_pctl", np.nan)
        btc_trend = btc_row.get("trend", "NEUTRAL")
        if not np.isnan(pctl):
            if direction == "LONG":
                if pctl > 0.85: vol_reg_s = 0.25
                elif pctl > 0.70: vol_reg_s = 0.35
                elif pctl < 0.30: vol_reg_s = 0.70
                elif pctl < 0.50: vol_reg_s = 0.60
                if btc_trend == "BEAR": vol_reg_s *= 0.8
                elif btc_trend == "BULL": vol_reg_s = min(vol_reg_s * 1.15, 1.0)
            else:
                if pctl > 0.85: vol_reg_s = 0.75
                elif pctl > 0.70: vol_reg_s = 0.65
                elif pctl < 0.30: vol_reg_s = 0.35
                elif pctl < 0.50: vol_reg_s = 0.40
                if btc_trend == "BULL": vol_reg_s *= 0.8
                elif btc_trend == "BEAR": vol_reg_s = min(vol_reg_s * 1.15, 1.0)
        details["btc_trend"] = btc_trend
        details["btc_atr_pctl"] = round(pctl, 3) if not np.isnan(pctl) else None

    # --- 4. Volume Regime (15%) ---
    vr_s = 0.5
    if not np.isnan(vol_ratio):
        if vol_ratio > 1.3: vr_s = 0.75
        elif vol_ratio > 1.0: vr_s = 0.60
        elif vol_ratio < 0.5: vr_s = 0.25
        elif vol_ratio < 0.7: vr_s = 0.38

    # --- 5. Cross-Asset Momentum (15%) ---
    cross_s = 0.5
    if ethbtc_row and btc_row:
        er = ethbtc_row.get("roc_7d", np.nan)
        br = btc_row.get("roc_7d", np.nan)
        if not np.isnan(er) and not np.isnan(br):
            if direction == "LONG":
                if er > 0 and br > 0: cross_s = 0.80
                elif er > 0 and br < -0.02: cross_s = 0.55
                elif er < -0.02 and br > 0: cross_s = 0.25
                elif er < -0.02 and br < -0.02: cross_s = 0.30
            else:
                if er < 0 and br < 0: cross_s = 0.80
                elif er < 0 and br > 0.02: cross_s = 0.55
                elif er > 0.02 and br < 0: cross_s = 0.25
                elif er > 0.02 and br > 0.02: cross_s = 0.30

    # --- Composite ---
    composite = (trend_s * 0.30 + mom_s * 0.20 + vol_reg_s * 0.20 + vr_s * 0.15 + cross_s * 0.15)
    composite = max(0.0, min(1.0, composite))
    details["m7_score"] = round(composite, 3)
    status = "PASS" if composite >= 0.50 else "FAIL"
    return status, composite, details




# ═══════════════════════════════════════════════════════════════════════
# TREND DETECTION — Multi-signal trend state machine
# ═══════════════════════════════════════════════════════════════════════

def calc_trend_state(df_1d):
    """
    Compute daily trend state using multiple confirmations.
    
    Returns per-bar:
    - trend: STRONG_UP / UP / NEUTRAL / DOWN / STRONG_DOWN
    - trend_score: -1.0 (max bearish) to +1.0 (max bullish)
    
    Signals:
    1. EMA21 vs EMA55 (direction)
    2. Price vs EMA21 (momentum confirmation)
    3. 7d ROC (acceleration)
    4. 14d RSI (overbought/oversold context)
    5. Higher highs / lower lows (structure)
    """
    close = df_1d['Close']
    high = df_1d['High']
    low = df_1d['Low']
    
    ema21 = calc_ema(close, 21)
    ema55 = calc_ema(close, 55)
    rsi = calc_rsi(close, 14)
    roc_7d = close.pct_change(7)
    roc_14d = close.pct_change(14)
    
    # Higher highs / lower lows over last 5 bars
    hh = (high > high.shift(1)) & (high.shift(1) > high.shift(2))
    ll = (low < low.shift(1)) & (low.shift(1) < low.shift(2))
    
    trend_score = pd.Series(0.0, index=df_1d.index)
    
    # 1. EMA direction (±0.30)
    ema_diff = (ema21 - ema55) / ema55
    trend_score += ema_diff.clip(-0.10, 0.10) * 3.0  # ±0.30 max
    
    # 2. Price vs EMA21 (±0.20)
    price_vs_ema = (close - ema21) / ema21
    trend_score += price_vs_ema.clip(-0.05, 0.05) * 4.0  # ±0.20 max
    
    # 3. ROC acceleration (±0.25)
    trend_score += roc_7d.clip(-0.10, 0.10) * 2.5  # ±0.25 max
    
    # 4. RSI confirmation (±0.15)
    rsi_signal = (rsi - 50) / 50
    trend_score += rsi_signal.clip(-0.50, 0.50) * 0.30  # ±0.15 max
    
    # 5. Structure: higher highs = bullish, lower lows = bearish (±0.10)
    structure = hh.astype(float) - ll.astype(float)
    trend_score += structure.rolling(3).mean().clip(-0.10, 0.10)
    
    trend_score = trend_score.clip(-1.0, 1.0)
    
    # Classify
    trend = pd.Series('NEUTRAL', index=df_1d.index)
    trend[trend_score > 0.40] = 'STRONG_UP'
    trend[(trend_score > 0.15) & (trend_score <= 0.40)] = 'UP'
    trend[(trend_score < -0.15) & (trend_score >= -0.40)] = 'DOWN'
    trend[trend_score < -0.40] = 'STRONG_DOWN'
    
    return trend, trend_score

# ═══════════════════════════════════════════════════════════════════════
# CONFIGURATION — JIMI v6.10 Fine-Tuned + M5
# ═══════════════════════════════════════════════════════════════════════

CONFIG = {
    # --- GATE THRESHOLDS ---
    "ICS_THRESHOLD_NORMAL": 0.55,   # raised floor (was 0.50)
    "ICS_THRESHOLD_CAUTION": 0.55,  # same (was 0.54)
    "ICS_FLOOR": 0.55,              # raised (was 0.50)
    "ICS_FLOOR_M4_FALSE": 0.55,     # raised (was 0.50)
    "ICS_CEILING": 0.70,            # NEW — reject overconfident signals

    # --- v6.12 IMPROVEMENTS ---
    "BIAS_GATE_ENABLED": True,       # Directional bias gate for longs
    "BIAS_GATE_LONG_ICS": 0.65,
    "TREND_FILTER_ENABLED": True,      # Hard directional filter
    "TREND_BLOCK_COUNTER_TREND": True, # Block trades against trend
    "TREND_STRONG_ONLY": False,        # If True, only trade STRONG trends
    "TREND_MIN_SCORE": 0.15,           # Minimum trend_score to allow trade in trend dir      # Higher ICS required for LONG when bias is BEARISH
    "MONTHLY_DD_CIRCUIT": 3.0,       # v6.13: Kept at 300% — summer-specific controls handle the rest
    "LONG_SUMMER_SIZE": 1.0,         # Disabled — seasonal bias gate is better approach
    "LONG_SUMMER_MONTHS": [6, 7, 8, 9],  # Months to apply long size reduction

    # --- v6.13: ZERO-PNL MONTH FIXES ---
    "SUMMER_MONTHS": [6, 7, 8, 9],   # Months with historically weak/negative PnL
    "SUMMER_SIZE_MULT": 0.60,        # Reduce position size 40% in summer chop
    "SUMMER_ICS_BOOST": 0.03,        # Require slightly higher ICS in summer months
    "TP1_CLOSE_BASE": 0.30,          # Base TP1 close fraction
    "TP1_CLOSE_SUMMER": 0.45,        # Higher TP1 close in summer — lock in gains before reversal
    "TP1_ATR_SUMMER": 0.7,           # Tighter TP1 in summer — hit target faster
    "MAX_CONSEC_LOSS_SUMMER": 2,     # Pause after 2 consecutive losses in summer (was 3)
    "CONSEC_LOSS_PAUSE_SUMMER": 12,  # Pause for 3 hours in summer (was 2h)
    "PHASE0_SUMMER_BLOCK": 0.60,     # Block entries in summer when phase0 > 0.60 (was 0.90)

    # --- v6.13: SHOULDER MONTH FIXES (October, March) ---
    "SHOULDER_MONTHS": [3, 10],      # Transition months — choppy, trend-uncertain
    "SHOULDER_SIZE_MULT": 0.50,      # Half size in shoulder months
    "SHOULDER_SL_ATR": 1.4,          # Tighter SL (was 2.0) — lose less per stop
    "SHOULDER_SL_HARD_MAX": 0.016,   # Tighter hard cap (was 2.5%)
    "SHOULDER_TP1_ATR": 0.8,         # Slightly tighter TP1 (was 0.9)
    "SHOULDER_TP1_CLOSE": 0.45,      # Close more at TP1 (was 30%) — capture gains
    "SHOULDER_ICS_BOOST": 0.02,      # Slightly higher ICS required
    "SHOULDER_MAX_TRADES_DAY": 3,    # Fewer trades per day
    "SHOULDER_COOLDOWN": 15,         # Longer cooldown between trades

    # --- MODULE WEIGHTS (ICS) — v6.12: Rebalanced toward positioning/flow ---
    "M1_WEIGHT": 0.10,              # v6.12: was 0.18 — direction is NOT the edge
    "M2_WEIGHT": 0.05,              # v6.12: was 0.12 — trend conf, low alpha
    "M3_WEIGHT": 0.25,              # v6.12: was 0.30 — entry timing, still important
    "M4_WEIGHT": 0.30,              # v6.12: was 0.25 — CVD positioning flow, core alpha
    "M5_WEIGHT": 0.20,              # v6.12: was 0.15 — liquidation magnets, core alpha
    "CASCADE_MULTIPLIER": 1.12,     # v6.12: NEW — ICS boost when cascade WITH trade
    "CASCADE_PENALTY": 0.85,        # v6.12: NEW — ICS penalty when cascade AGAINST trade
    "DIR_VETO_ENABLED": True,       # v6.12: NEW — block when M4+M5 disagree with M1/M2

    # --- TECHNICAL INDICATORS ---
    "MACD_FAST": 12,
    "MACD_SLOW": 26,
    "MACD_SIGNAL": 9,
    "EMA_FAST": 21,
    "EMA_SLOW": 55,
    "RSI_PERIOD": 14,
    "ATR_PERIOD": 14,

    # --- M3: VWAP ENTRY ZONE ---
    "VWAP_LOOKBACK": 96,
    "VWAP_ZONE_PCT": 0.012,     # wider zone for more signals (was 0.008)
    "VOL_THRESHOLD": 0.25,      # slightly relaxed (was 0.30)
    "TAKER_LONG": 0.52,
    "TAKER_SHORT": 0.48,
    "TAKER_FILLNA": 0.50,
    "SIGNAL_EXPIRY": 3,

    # --- M4: CVD (15m divergence + 2H zero-line) ---
    "CVD_LOOKBACK": 36,
    "CVD_DIVERGENCE_WINDOW": 12,
    "M4_ZL_LOOKBACK": 18,
    "M4_ZL_MOMENTUM_BARS": 8,
    "M4_DIV_WEIGHT": 0.40,
    "M4_ZL_WEIGHT": 0.60,

    # --- M5: LIQUIDATION MAGNET ---
    "M5_VP_LOOKBACK": 672,
    "M5_VP_BINS": 50,
    "M5_MIN_SCORE": 0.25,       # slightly relaxed (was 0.30)

    # --- STOP LOSS ---
    "SL_ATR_STD": 1.3,          # wider SL to let trades develop (was 1.5)
    "SL_ATR_STD_SUMMER": 1.6,   # v6.13: tighter SL in summer — less room for adverse
    "SL_HARD_MAX_PCT": 0.018,
    "SL_HARD_MAX_SUMMER": 0.018, # v6.13: tighter hard cap in summer
    "SL_BREAKEVEN_AFTER_TP1": True,
    "EARLY_EXIT_BARS": 16,     # DISABLED — early exit kills winners
    "EARLY_EXIT_MIN_LOSS": 0.003,
    "EARLY_EXIT_BARS_SUMMER": 12, # v6.13: early exit after 3h in summer if losing
    "EARLY_EXIT_MIN_LOSS_SUMMER": 0.002,

    # --- TAKE PROFIT LADDER ---
    "TP1_ATR": 1.5,             # balanced (was 0.8)
    "TP2_ATR": 2.0,
    "TP3_ATR": 3.5,
    "TP1_CLOSE": 0.30,          # take a bit more at TP1 (was 0.25)
    "TP2_CLOSE": 0.30,

    # --- POSITION SIZING ---
    "SIZE_STD": 5.0,            # default size (both directions)
    "SIZE_LONG": 5.0,           # LONG size — equalized (was 2.5, penalized longs)
    "SIZE_M2_NEUTRAL": 2.5,
    "SIZE_CAUTION": 3.5,

    # --- ENTRY FILTERS ---
    "BAR_MOVE_ATR": 0.5,
    "ATR_FILTER_MAX": 0.035,
    "MIN_ENTRY_DIST_PCT": 0.002,  # min 0.2% price distance between entries (NEW)

    # --- RISK MANAGEMENT ---
    "MAX_TRADES_DAY": 5,
    "MAX_TRADES_DAY_SUMMER": 3,    # v6.13: fewer trades in summer chop
    "MAX_DAILY_LOSS": 0.05,     # 5% daily loss limit (was 0.06)
    "MAX_DAILY_LOSS_SUMMER": 0.03, # v6.13: tighter daily loss in summer
    "COOLDOWN_MINUTES": 10,
    "COOLDOWN_MINUTES_SUMMER": 20, # v6.13: longer cooldown in summer
    "ROLLING_WR_WINDOW": 8,
    "ROLLING_WR_MIN": 0.40,
    "MAX_CONSEC_LOSS": 3,       # pause after 3 consecutive losses (NEW)
    "CONSEC_LOSS_PAUSE_BARS": 8, # pause for 2 hours (NEW)

    # --- LONG FILTER ---
    "LONG_MIN_ICS": 0.55,       # same as general threshold (was 0.65, penalized longs)

    # --- DERIVATIVES (M6) ---
    "M6_WEIGHT": 0.10,          # derivatives signal weight in ICS
    "M3_WEIGHT_DERIV": 0.25,    # reduced from 0.30 to make room
    "M4_WEIGHT_DERIV": 0.20,    # reduced from 0.25 to make room
    "DERIV_ENABLED": True,

    # --- M7: MARKET REGIME (ETH/BTC + BTC Vol + Volume) ---
    "M7_ENABLED": True,          # enable M7 market regime module
    "M7_WEIGHT": 0.08,           # M7 weight in ICS (macro filter)
    "M7_SIZE_REDUCTION": 0.70,   # reduce size 30% when M7 very bearish (<0.35)
    "M7_SIZE_MILD": 0.85,        # reduce size 15% when M7 mildly bearish (<0.45)

    # --- DATA ---
    "PAIR": "ETHUSDT",
    "WARMUP_BARS_1H": 168,
}


# ═══════════════════════════════════════════════════════════════════════
# DATA LOADING & RESAMPLING
# ═══════════════════════════════════════════════════════════════════════

def load_data(filepath):
    df = pd.read_csv(filepath)
    df.columns = df.columns.str.strip()
    df['Open time'] = pd.to_datetime(df['Open time'].str.strip())
    df['Close time'] = pd.to_datetime(df['Close time'].str.strip())
    for col in ['Open', 'High', 'Low', 'Close', 'Volume',
                'Quote asset volume', 'Number of trades',
                'Taker buy base asset volume', 'Taker buy quote asset volume']:
        df[col] = pd.to_numeric(df[col].astype(str).str.strip(), errors='coerce')
    df = df.sort_values('Open time').reset_index(drop=True)
    df = df.dropna(subset=['Open', 'High', 'Low', 'Close', 'Volume'])
    return df


def resample_ohlcv(df_15m, timeframe):
    df = df_15m.copy().set_index('Open time')
    agg = {
        'Open': 'first', 'High': 'max', 'Low': 'min', 'Close': 'last',
        'Volume': 'sum', 'Quote asset volume': 'sum',
        'Number of trades': 'sum',
        'Taker buy base asset volume': 'sum',
        'Taker buy quote asset volume': 'sum',
    }
    rule_map = {'30m': '30min', '1H': '1h', '2H': '2h', '4H': '4h', '1D': '1D'}
    return df.resample(rule_map[timeframe]).agg(agg).dropna(subset=['Open']).reset_index()


# ═══════════════════════════════════════════════════════════════════════
# TECHNICAL INDICATORS
# ═══════════════════════════════════════════════════════════════════════

def calc_ema(series, period):
    return series.ewm(span=period, adjust=False).mean()

def calc_macd(close, fast=12, slow=26, signal=9):
    ema_fast = calc_ema(close, fast)
    ema_slow = calc_ema(close, slow)
    macd_line = ema_fast - ema_slow
    signal_line = calc_ema(macd_line, signal)
    histogram = macd_line - signal_line
    return macd_line, signal_line, histogram

def calc_rsi(close, period=14):
    delta = close.diff()
    gain = delta.where(delta > 0, 0.0)
    loss = -delta.where(delta < 0, 0.0)
    avg_gain = gain.ewm(alpha=1/period, min_periods=period).mean()
    avg_loss = loss.ewm(alpha=1/period, min_periods=period).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    return 100 - (100 / (1 + rs))

def calc_atr(high, low, close, period=14):
    tr1 = high - low
    tr2 = (high - close.shift(1)).abs()
    tr3 = (low - close.shift(1)).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    return tr.ewm(span=period, adjust=False).mean()

def calc_vwap(high, low, close, volume, lookback=96):
    typical = (high + low + close) / 3
    cum_tv = (typical * volume).rolling(lookback).sum()
    cum_vol = volume.rolling(lookback).sum()
    return cum_tv / cum_vol.replace(0, np.nan)

def calc_vol_ratio(volume_15m):
    vol_24h = volume_15m.rolling(96).sum()
    vol_7d = volume_15m.rolling(672).sum()
    return vol_24h / vol_7d.replace(0, np.nan)

def calc_swing_bias(df_1d):
    ema21 = calc_ema(df_1d['Close'], 21)
    ema55 = calc_ema(df_1d['Close'], 55)
    bias = pd.Series('NEUTRAL', index=df_1d.index)
    bias[ema21 > ema55] = 'BULLISH'
    bias[ema21 < ema55] = 'BEARISH'
    return bias

def calc_phase0(df_1d):
    rsi = calc_rsi(df_1d['Close'], 14)
    vol_ma20 = df_1d['Volume'].rolling(20).mean()
    vol_ratio = df_1d['Volume'] / vol_ma20.replace(0, np.nan)
    rsi_score = (rsi - 50).abs() / 50
    vol_score = vol_ratio.clip(0, 3) / 3
    return (rsi_score * 0.6 + vol_score * 0.4).clip(0, 1)


# ═══════════════════════════════════════════════════════════════════════
# CVD (Cumulative Volume Delta) — 15m Rolling
# ═══════════════════════════════════════════════════════════════════════

def calc_cvd_15m(df_15m):
    """Rolling CVD — delta per bar, then smoothed. Not cumsum."""
    taker_buy = df_15m['Taker buy base asset volume']
    total_vol = df_15m['Volume']
    taker_sell = total_vol - taker_buy
    delta = taker_buy - taker_sell
    # Smoothed CVD: rolling sum over ~2H window (8 bars)
    cvd_rolling = delta.rolling(8).sum()
    return cvd_rolling

def detect_cvd_divergence_15m(df_15m, lookback=24, window=12):
    """
    Detect CVD divergence on 15m bars.
    
    Three methods:
    1. Slope comparison (linear trend divergence)
    2. Swing high/low comparison (non-linear divergence)
    3. Exhaustion detection (momentum collapse at extremes)
    
    Target: ~5-8% of bars flagged as divergent.
    """
    cvd = df_15m['cvd_15m'].values
    close = df_15m['Close'].values
    high = df_15m['High'].values
    low = df_15m['Low'].values
    n = len(df_15m)
    divergence = ['NONE'] * n
    last_div_bar = -999  # cooldown tracker

    for i in range(lookback + window, n):
        # Cooldown: skip if a divergence was flagged within last 4 bars
        if i - last_div_bar < 4:
            continue

        # --- Method 1: Slope comparison over window ---
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

                # Bearish: price trending up, CVD trending down
                if price_dir > 0.03 and cvd_dir < -0.03:
                    divergence[i] = 'BEARISH'
                # Bullish: price trending down, CVD trending up
                elif price_dir < -0.03 and cvd_dir > 0.03:
                    divergence[i] = 'BULLISH'

        # --- Method 2: Swing high/low comparison ---
        if divergence[i] == 'NONE':
            look = min(lookback, i)
            # Bearish: price makes higher high, CVD makes lower high
            if i >= 4 and (high[i] >= np.max(high[i-3:i+1]) * 0.9995):
                prev_hi = i - look // 2
                if prev_hi >= 3 and high[prev_hi] >= np.max(high[max(0,prev_hi-3):prev_hi+1]) * 0.9995:
                    if high[i] > high[prev_hi] * 1.002:  # price clearly higher
                        cvd_at_i = np.nanmean(cvd[max(0,i-1):i+1])
                        cvd_at_prev = np.nanmean(cvd[max(0,prev_hi-1):prev_hi+1])
                        if cvd_at_i < cvd_at_prev * 0.993:  # CVD clearly lower
                            divergence[i] = 'BEARISH'

            # Bullish: price makes lower low, CVD makes higher low
            if divergence[i] == 'NONE':
                if i >= 4 and (low[i] <= np.min(low[i-3:i+1]) * 1.0005):
                    prev_lo = i - look // 2
                    if prev_lo >= 3 and low[prev_lo] <= np.min(low[max(0,prev_lo-3):prev_lo+1]) * 1.0005:
                        if low[i] < low[prev_lo] * 0.998:  # price clearly lower
                            cvd_at_i = np.nanmean(cvd[max(0,i-1):i+1])
                            cvd_at_prev = np.nanmean(cvd[max(0,prev_lo-1):prev_lo+1])
                            if cvd_at_i > cvd_at_prev * 1.007:  # CVD clearly higher
                                divergence[i] = 'BULLISH'

        # --- Method 3: Exhaustion — price at extreme, CVD momentum collapsing ---
        if divergence[i] == 'NONE' and i >= 8:
            cvd_momentum = cvd[i] - cvd[i-4]
            price_momentum = close[i] - close[i-4]
            cvd_std = np.nanstd(cvd[max(0,i-24):i+1])
            if cvd_std > 0:
                # Bearish exhaustion
                if (price_momentum > 0 and cvd_momentum < 0 and
                    high[i] >= np.max(high[max(0,i-8):i+1]) * 0.999 and
                    abs(cvd_momentum) > cvd_std * 1.5):
                    divergence[i] = 'BEARISH'
                # Bullish exhaustion
                elif (price_momentum < 0 and cvd_momentum > 0 and
                      low[i] <= np.min(low[max(0,i-8):i+1]) * 1.001 and
                      abs(cvd_momentum) > cvd_std * 1.5):
                    divergence[i] = 'BULLISH'

        if divergence[i] != 'NONE':
            last_div_bar = i

    return pd.Series(divergence, index=df_15m.index)


# ═══════════════════════════════════════════════════════════════════════
# CVD 2H — Zero-Line Crossover (Regime Shift)
# ═══════════════════════════════════════════════════════════════════════

def calc_cvd_2h(df_2h):
    """
    Rolling CVD on 2H bars. Uses rolling sum (not cumsum) so the zero line
    is meaningful — positive = net buyer pressure, negative = net seller.
    """
    taker_buy = df_2h['Taker buy base asset volume']
    total_vol = df_2h['Volume']
    taker_sell = total_vol - taker_buy
    delta = taker_buy - taker_sell
    # Rolling sum over ~24H window (12 × 2H bars) for smooth zero-line
    cvd_rolling = delta.rolling(12).sum()
    return cvd_rolling

def detect_cvd_zero_cross(df_2h, lookback=6):
    """
    Detect when 2H CVD crosses the zero line.
    
    Returns per-bar state:
    - 'CROSS_UP'   : CVD just crossed from negative → positive (buyers taking control)
    - 'CROSS_DOWN' : CVD just crossed from positive → negative (sellers taking control)
    - 'ABOVE'      : CVD is above zero (buyers in control, no fresh cross)
    - 'BELOW'      : CVD is below zero (sellers in control, no fresh cross)
    - 'NONE'       : insufficient data
    
    Also computes momentum_streak: how many bars since the last cross,
    and CVD slope for strength scoring.
    """
    cvd = df_2h['cvd_2h'].values
    n = len(df_2h)
    state = ['NONE'] * n
    cross_bar = [-1] * n   # bar index of last cross
    cross_dir = ['NONE'] * n  # direction of last cross

    for i in range(1, n):
        if pd.isna(cvd[i]) or pd.isna(cvd[i-1]):
            continue

        # Detect zero-line cross
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

        # Inherit last cross info
        if state[i] not in ('CROSS_UP', 'CROSS_DOWN') and i > 0:
            cross_bar[i] = cross_bar[i-1]
            cross_dir[i] = cross_dir[i-1]

    return (pd.Series(state, index=df_2h.index),
            pd.Series(cross_bar, index=df_2h.index),
            pd.Series(cross_dir, index=df_2h.index))


# ═══════════════════════════════════════════════════════════════════════
# M5: LIQUIDATION MAGNET MODULE
# ═══════════════════════════════════════════════════════════════════════

def build_volume_profile(highs, lows, closes, volumes, n_bins=50, lookback=672):
    h = highs[-lookback:]
    l = lows[-lookback:]
    v = volumes[-lookback:]
    price_min, price_max = np.min(l), np.max(h)
    if price_max == price_min:
        return None, None, None

    bin_edges = np.linspace(price_min, price_max, n_bins + 1)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    vol_profile = np.zeros(n_bins)
    bar_ranges = h - l
    bar_ranges[bar_ranges == 0] = 1

    for j in range(n_bins):
        overlap_low = np.maximum(l, bin_edges[j])
        overlap_high = np.minimum(h, bin_edges[j+1])
        overlap = np.maximum(overlap_high - overlap_low, 0)
        proportion = overlap / bar_ranges
        vol_profile[j] = np.sum(v * proportion)

    return bin_centers, vol_profile, bin_edges

def find_magnets(bin_centers, vol_profile, n_magnets=5, min_gap_pct=0.005):
    if vol_profile is None or len(vol_profile) == 0:
        return []
    mean_vol = np.mean(vol_profile)
    if mean_vol == 0:
        return []

    peaks = []
    for i in range(1, len(vol_profile) - 1):
        if (vol_profile[i] > vol_profile[i-1] and
            vol_profile[i] > vol_profile[i+1] and
            vol_profile[i] > mean_vol * 1.2):
            peaks.append((bin_centers[i], vol_profile[i], vol_profile[i] / mean_vol))

    peaks.sort(key=lambda x: x[1], reverse=True)
    filtered = []
    for peak in peaks:
        if not any(abs(peak[0] - e[0]) / e[0] < min_gap_pct for e in filtered):
            filtered.append(peak)
    return filtered[:n_magnets]

def find_gaps(bin_centers, vol_profile, n_gaps=5):
    if vol_profile is None or len(vol_profile) == 0:
        return []
    mean_vol = np.mean(vol_profile)
    if mean_vol == 0:
        return []
    gaps = [(bin_centers[i], vol_profile[i]) for i in range(len(vol_profile))
            if vol_profile[i] < mean_vol * 0.3]
    gaps.sort(key=lambda x: x[1])
    return gaps[:n_gaps]

# ═══════════════════════════════════════════════════════════════════════
# SUPPORT / RESISTANCE — Behavioral Level Detection
# ═══════════════════════════════════════════════════════════════════════

def find_support_resistance(df_15m, idx=None, lookback=672, n_levels=10,
                             bin_pct=0.002, touch_pct=0.004, bounce_pct=0.003,
                             bounce_bars=8, min_touches=3):
    """
    Find support/resistance levels based on price rejection behavior.

    Unlike volume-profile magnets (which find where volume *concentrated*),
    this finds where price *touched and bounced* repeatedly — behavioral S/R.

    Algorithm:
    1. Discretize price into bins (bin_pct width)
    2. For each bin, count touches (low within touch_pct of bin center)
    3. For each touch, check if price bounced (moved bounce_pct away within N bars)
    4. Levels with high touch+bounce count are S/R
    5. Strength = bounce_count / touch_count (consistency) * touch_count (frequency)

    Returns: list of (price, strength, touches, bounces, type) tuples
    - type: 'SUPPORT' or 'RESISTANCE' based on whether price is above or below
    """
    if idx is None:
        idx = len(df_15m) - 1
    if idx < lookback:
        return []

    start = max(0, idx - lookback + 1)
    highs = df_15m['High'].values[start:idx+1].astype(float)
    lows = df_15m['Low'].values[start:idx+1].astype(float)
    closes = df_15m['Close'].values[start:idx+1].astype(float)
    current_price = closes[-1]

    if len(closes) < 20:
        return []

    price_min, price_max = lows.min(), highs.max()
    price_range = price_max - price_min
    if price_range <= 0:
        return []

    # Create bins
    n_bins = max(int(price_range / (current_price * bin_pct)), 20)
    bin_edges = np.linspace(price_min, price_max, n_bins + 1)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    bin_width = bin_edges[1] - bin_edges[0]

    levels = []
    for bi in range(len(bin_centers)):
        bc = bin_centers[bi]
        touches = 0
        bounces = 0

        for i in range(len(closes)):
            # Touch: low comes within touch_pct of bin center
            touch_dist = abs(lows[i] - bc) / bc
            if touch_dist <= touch_pct:
                touches += 1
                # Bounce: price moves away within bounce_bars bars
                bounced = False
                for j in range(i+1, min(i+1+bounce_bars, len(closes))):
                    if abs(closes[j] - bc) / bc >= bounce_pct:
                        bounced = True
                        break
                if bounced:
                    bounces += 1

        if touches >= min_touches and bounces >= min_touches:
            consistency = bounces / touches if touches > 0 else 0
            strength = touches * consistency
            # Determine type based on recent price action
            # If current price is above, it's support; below = resistance
            sr_type = 'SUPPORT' if current_price > bc else 'RESISTANCE'
            levels.append((bc, strength, touches, bounces, sr_type))

    # Deduplicate nearby levels
    levels.sort(key=lambda x: x[1], reverse=True)
    filtered = []
    for level in levels:
        if not any(abs(level[0] - e[0]) / e[0] < bin_pct for e in filtered):
            filtered.append(level)

    # Sort by distance from current price
    filtered.sort(key=lambda x: abs(x[0] - current_price))
    return filtered[:n_levels]


def calc_magnetic_pull(current_price, magnets, direction):
    if not magnets:
        return 0.0, None, None
    relevant = []
    for price, vol, strength in magnets:
        if direction == 'LONG' and price > current_price:
            dist = (price - current_price) / current_price
            relevant.append((price, vol, strength, dist))
        elif direction == 'SHORT' and price < current_price:
            dist = (current_price - price) / current_price
            relevant.append((price, vol, strength, dist))
    if not relevant:
        return 0.0, None, None
    relevant.sort(key=lambda x: x[3])
    nearest = relevant[0]
    dist_factor = max(0, 1.0 - nearest[3] / 0.02)
    strength_factor = min(nearest[2] / 3.0, 1.0)
    return dist_factor * 0.6 + strength_factor * 0.4, nearest[0], nearest[3]

def calc_gap_acceleration(current_price, gaps, direction):
    if not gaps:
        return 0.0, False
    gap_between = False
    nearest_dist = float('inf')
    for price, vol in gaps:
        if direction == 'LONG' and price > current_price:
            dist = (price - current_price) / current_price
        elif direction == 'SHORT' and price < current_price:
            dist = (current_price - price) / current_price
        else:
            continue
        if dist < nearest_dist:
            nearest_dist = dist
        if dist < 0.005:
            gap_between = True
    if nearest_dist == float('inf'):
        return 0.0, False
    return max(0, 1.0 - nearest_dist / 0.01), gap_between

def detect_cascade_mode(df_15m, idx, magnets, direction):
    """
    Detect if price is in CASCADE mode approaching a magnet.
    
    Slow approach → magnet acts as support/resistance (reversal)
    Fast approach + volume spike → magnet triggers liquidations (cascade/push-through)
    
    Returns: (is_cascade, cascade_direction, cascade_strength, details)
    - cascade_direction: 'WITH' if trade direction matches cascade, 'AGAINST' if opposing
    - cascade_strength: 0-1, how violent the cascade is
    """
    if idx < 20 or not magnets:
        return False, 'NONE', 0.0, {}

    closes = df_15m['Close'].values.astype(float)
    highs = df_15m['High'].values.astype(float)
    lows = df_15m['Low'].values.astype(float)
    volumes = df_15m['Volume'].values.astype(float)
    current_price = closes[idx]

    # Find nearest magnet in approach direction
    approach_magnets = []
    for price, vol, strength in magnets:
        dist = abs(price - current_price) / current_price
        if dist < 0.015:  # within 1.5% of a magnet
            approach_magnets.append((price, vol, strength, dist))
    
    if not approach_magnets:
        return False, 'NONE', 0.0, {}

    approach_magnets.sort(key=lambda x: x[3])
    nearest_mag_price = approach_magnets[0][0]
    nearest_mag_dist = approach_magnets[0][3]

    # --- Cascade Detection Signals ---

    # 1. Momentum: price velocity over last 4-8 bars
    if idx >= 8:
        momentum_4 = (closes[idx] - closes[idx-4]) / closes[idx-4]
        momentum_8 = (closes[idx] - closes[idx-8]) / closes[idx-8]
        # Accelerating momentum (second derivative)
        momentum_accel = abs(momentum_4) - abs((closes[idx-4] - closes[idx-8]) / closes[idx-8]) if idx >= 8 else 0
    else:
        momentum_4 = 0
        momentum_8 = 0
        momentum_accel = 0

    # 2. Volume spike
    vol_avg = np.mean(volumes[max(0,idx-20):idx])
    vol_spike = volumes[idx] / vol_avg if vol_avg > 0 else 0

    # 3. Range expansion
    current_range = highs[idx] - lows[idx]
    avg_range = np.mean(highs[max(0,idx-20):idx] - lows[max(0,idx-20):idx])
    range_expansion = current_range / avg_range if avg_range > 0 else 0

    # 4. Taker aggression (who's driving)
    if 'Taker buy base asset volume' in df_15m.columns:
        taker_buy = df_15m['Taker buy base asset volume'].iloc[idx]
        total_vol = df_15m['Volume'].iloc[idx]
        taker_ratio = taker_buy / total_vol if total_vol > 0 else 0.5
    else:
        taker_ratio = CONFIG['TAKER_FILLNA']

    # --- Cascade Direction ---
    # Price making new lows with volume → cascade DOWN
    # Price making new highs with volume → cascade UP
    if idx >= 4:
        making_new_low = lows[idx] <= np.min(lows[max(0,idx-4):idx])
        making_new_high = highs[idx] >= np.max(highs[max(0,idx-4):idx])
    else:
        making_new_low = False
        making_new_high = False

    cascade_down = making_new_low and momentum_4 < -0.003 and vol_spike > 1.3
    cascade_up = making_new_high and momentum_4 > 0.003 and vol_spike > 1.3

    # --- Is this a cascade? ---
    is_cascade = False
    cascade_dir = 'NONE'
    cascade_strength = 0.0

    if cascade_down or cascade_up:
        # Check if approaching a magnet (not already past it)
        if nearest_mag_dist < 0.01:  # within 1% of magnet
            is_cascade = True
            cascade_dir_raw = 'DOWN' if cascade_down else 'UP'
            
            # Does the cascade align with the trade direction?
            if (direction == 'LONG' and cascade_dir_raw == 'DOWN') or \
               (direction == 'SHORT' and cascade_dir_raw == 'UP'):
                cascade_dir = 'AGAINST'  # cascade is against our trade
            else:
                cascade_dir = 'WITH'     # cascade is with our trade

            # Strength based on momentum + volume + range
            cascade_strength = min(
                (abs(momentum_4) / 0.01) * 0.4 +
                (vol_spike / 3.0) * 0.4 +
                (range_expansion / 2.0) * 0.2,
                1.0
            )

    details = {
        'momentum_4': round(momentum_4 * 100, 3),
        'momentum_8': round(momentum_8 * 100, 3),
        'vol_spike': round(vol_spike, 2),
        'range_expansion': round(range_expansion, 2),
        'taker_ratio': round(taker_ratio, 3),
        'making_new_low': making_new_low,
        'making_new_high': making_new_high,
        'nearest_mag_dist': round(nearest_mag_dist * 100, 3),
        'cascade_down': cascade_down,
        'cascade_up': cascade_up,
    }

    return is_cascade, cascade_dir, cascade_strength, details


def score_m5(df_15m, idx, direction, n_bins=50, lookback=672):
    if idx < lookback:
        return 'FAIL', 0.0, {'reason': 'insufficient data'}

    highs = df_15m['High'].values.astype(float)
    lows = df_15m['Low'].values.astype(float)
    closes = df_15m['Close'].values.astype(float)
    volumes = df_15m['Volume'].values.astype(float)
    current_price = closes[idx]

    bin_centers, vol_profile, bin_edges = build_volume_profile(
        highs[:idx+1], lows[:idx+1], closes[:idx+1], volumes[:idx+1],
        n_bins=n_bins, lookback=lookback)
    if bin_centers is None:
        return 'FAIL', 0.0, {'reason': 'profile build failed'}

    magnets = find_magnets(bin_centers, vol_profile)
    gaps = find_gaps(bin_centers, vol_profile)
    pull_score, nearest_magnet, magnet_dist = calc_magnetic_pull(current_price, magnets, direction)
    accel_score, gap_between = calc_gap_acceleration(current_price, gaps, direction)

    current_bin = np.searchsorted(bin_edges, current_price) - 1
    current_bin = max(0, min(current_bin, len(vol_profile) - 1))
    vol_above = np.sum(vol_profile[current_bin+1:])
    vol_below = np.sum(vol_profile[:current_bin])
    total_vol = vol_above + vol_below
    skew = (vol_above / total_vol if direction == 'LONG' else vol_below / total_vol) if total_vol > 0 else 0.5
    skew_score = min(skew / 0.7, 1.0)

    # --- Cascade Detection ---
    is_cascade, cascade_dir, cascade_strength, cascade_details = detect_cascade_mode(
        df_15m, idx, magnets, direction)

    # --- Score Adjustment for Cascade ---
    if is_cascade:
        if cascade_dir == 'WITH':
            # Cascade is WITH our trade direction → huge bonus
            # Price is being pushed in our favor by liquidations
            cascade_bonus = cascade_strength * 0.4
            pull_score = min(pull_score + cascade_bonus, 1.0)
        elif cascade_dir == 'AGAINST':
            # Cascade is AGAINST our trade → strong penalty
            # We'd be caught in the liquidation wave
            cascade_penalty = cascade_strength * 0.6
            pull_score = max(pull_score - cascade_penalty, 0.0)

    score = pull_score * 0.5 + accel_score * 0.3 + skew_score * 0.2
    details = {
        'magnets': [(round(p, 2), round(s, 2)) for p, _, s in magnets[:3]],
        'gaps': [round(p, 2) for p, _ in gaps[:3]],
        'nearest_magnet': round(nearest_magnet, 2) if nearest_magnet else None,
        'magnet_dist_pct': round(magnet_dist * 100, 3) if magnet_dist else None,
        'pull_score': round(pull_score, 3),
        'accel_score': round(accel_score, 3),
        'skew_score': round(skew_score, 3),
        'gap_between': gap_between,
        'cascade': is_cascade,
        'cascade_dir': cascade_dir,
        'cascade_strength': round(cascade_strength, 3),
        'cascade_details': cascade_details,
    }
    return ('PASS', score, details) if score >= CONFIG['M5_MIN_SCORE'] else ('FAIL', score, details)

def detect_cascade_setup(df_15m, idx, lookback=96):
    if idx < lookback:
        return {'cascade': False, 'reason': 'insufficient data'}
    closes = df_15m['Close'].values.astype(float)
    volumes = df_15m['Volume'].values.astype(float)
    highs = df_15m['High'].values.astype(float)
    lows = df_15m['Low'].values.astype(float)

    momentum = abs(closes[idx] - closes[idx-4]) / closes[idx-4] if idx >= 4 else 0
    vol_avg = np.mean(volumes[max(0,idx-20):idx])
    vol_spike = volumes[idx] / vol_avg if vol_avg > 0 else 0
    current_range = highs[idx] - lows[idx]
    avg_range = np.mean(highs[max(0,idx-20):idx] - lows[max(0,idx-20):idx])
    range_expansion = current_range / avg_range if avg_range > 0 else 0

    cascade = momentum > 0.005 and vol_spike > 1.5 and range_expansion > 1.3
    return {
        'cascade': cascade,
        'momentum': round(momentum * 100, 3),
        'vol_spike': round(vol_spike, 2),
        'range_expansion': round(range_expansion, 2),
    }


# ═══════════════════════════════════════════════════════════════════════
# MODULE SCORING
# ═══════════════════════════════════════════════════════════════════════

def score_m1(df_1h, idx):
    if idx < 1:
        return 'NEUTRAL', 0.5
    hist = df_1h['macd_hist'].iloc[idx]
    hist_prev = df_1h['macd_hist'].iloc[idx - 1]
    if hist > 0 and hist > hist_prev:
        return 'BULLISH', 1.0
    elif hist < 0 and hist < hist_prev:
        return 'BEARISH', 1.0
    elif hist > 0:
        return 'BULLISH', 0.7
    elif hist < 0:
        return 'BEARISH', 0.7
    return 'NEUTRAL', 0.5

def score_m2(df_1h, df_2h, df_4h, df_1d, idx_1h, idx_2h, idx_4h, idx_1d):
    macro_dir = None
    if idx_1d >= 0:
        bias = df_1d['swing_bias'].iloc[idx_1d]
        macro_dir = {'BULLISH': 'BULL', 'BEARISH': 'BEAR'}.get(bias, 'NEUTRAL')

    tf_4h = None
    if idx_4h >= 1:
        ef, es = df_4h['ema_fast'].iloc[idx_4h], df_4h['ema_slow'].iloc[idx_4h]
        tf_4h = 'BULL' if ef > es else 'BEAR' if ef < es else 'NEUTRAL'

    tf_2h = None
    if idx_2h >= 1:
        ef, es = df_2h['ema_fast'].iloc[idx_2h], df_2h['ema_slow'].iloc[idx_2h]
        tf_2h = 'BULL' if ef > es else 'BEAR' if ef < es else 'NEUTRAL'

    tf_1h = None
    if idx_1h >= 1:
        ef, es = df_1h['ema_fast'].iloc[idx_1h], df_1h['ema_slow'].iloc[idx_1h]
        tf_1h = 'BULL' if ef > es else 'BEAR' if ef < es else 'NEUTRAL'

    if macro_dir == 'NEUTRAL' and tf_4h == 'NEUTRAL':
        return 'NEUTRAL', 0.35

    direction = macro_dir if macro_dir != 'NEUTRAL' else (tf_4h if tf_4h != 'NEUTRAL' else tf_2h)
    if direction == 'NEUTRAL':
        return 'NEUTRAL', 0.35

    confirmations, layers, traps = 0, 0, 0
    if tf_4h and macro_dir != 'NEUTRAL':
        layers += 1
        if tf_4h == macro_dir or tf_4h == 'NEUTRAL': confirmations += 1
        else: traps += 1
    if tf_2h and tf_4h and tf_4h != 'NEUTRAL':
        layers += 1
        if tf_2h == tf_4h or tf_2h == 'NEUTRAL': confirmations += 1
        else: traps += 1
    if tf_1h and tf_2h and tf_2h != 'NEUTRAL':
        layers += 1
        if tf_1h == tf_2h or tf_1h == 'NEUTRAL': confirmations += 1
        else: traps += 1

    if layers == 0:
        return 'NEUTRAL', 0.40
    if traps > 0 and traps / layers >= 0.5:
        return 'FAIL', max(0.15, 0.5 - traps / layers)

    confirm_ratio = confirmations / layers
    higher_score = 0.5 + (0.25 if macro_dir == direction else 0) + (0.25 if tf_4h == direction else 0)
    final = min(higher_score * 0.5 + confirm_ratio * 0.5, 1.0)

    if final >= 0.65: return 'PASS', final
    elif final >= 0.35: return 'NEUTRAL', final
    return 'FAIL', final

def score_m3(df_15m, idx, direction):
    """
    M3: VWAP + Volume + Taker — soft scoring (no hard gate).
    Each component contributes a continuous score; no single condition
    can zero out the module.  The ICS threshold downstream still gates entries.
    """
    if idx < CONFIG['VWAP_LOOKBACK']:
        return 'FAIL', 0.0, None
    row = df_15m.iloc[idx]
    vwap, close, volume = row['vwap'], row['Close'], row['Volume']
    vol_avg20, taker_ratio = row['vol_ma20'], row['taker_ratio']
    if pd.isna(vwap) or pd.isna(vol_avg20) or vol_avg20 == 0:
        return 'FAIL', 0.0, None

    vwap_dist = abs(close - vwap) / vwap

    # --- VWAP score: continuous, decays to 0 at 2x zone width ---
    in_zone = vwap_dist <= CONFIG['VWAP_ZONE_PCT']
    if in_zone:
        vwap_score = 1.0 - (vwap_dist / CONFIG['VWAP_ZONE_PCT'])
    else:
        vwap_score = max(0.0, 1.0 - (vwap_dist / (CONFIG['VWAP_ZONE_PCT'] * 2)))

    # --- Volume score: soft ramp, half penalty below threshold ---
    vol_ratio = volume / vol_avg20 if vol_avg20 > 0 else 0
    vol_score = min(vol_ratio / 2.0, 1.0)
    if vol_ratio < CONFIG['VOL_THRESHOLD']:
        vol_score *= 0.5

    # --- Taker score: continuous, centered at 0.50 ---
    if direction == 'LONG':
        taker_score = max(0.0, min((taker_ratio - 0.40) / 0.15, 1.0))
    else:
        taker_score = max(0.0, min((0.60 - taker_ratio) / 0.15, 1.0))

    combined = vwap_score * 0.4 + vol_score * 0.3 + taker_score * 0.3
    return 'PASS', combined, close

def score_m4(df_15m, df_2h, idx_15m, idx_2h, direction):
    """
    M4: CVD Composite — two layers:
    Layer A: 15m CVD divergence (catches reversals)
    Layer B: 2H CVD zero-line cross (catches regime shifts / momentum rides)
    
    Returns: (status, score, details_dict)
    """
    layer_a_score = 0.0
    layer_a_status = 'FAIL'
    layer_a_div = 'NONE'
    
    layer_b_score = 0.0
    layer_b_status = 'FAIL'
    layer_b_cross = 'NONE'
    layer_b_bars_since = 999
    zl_state = 'NONE'

    # ── Layer A: 15m CVD Divergence ──
    if idx_15m >= CONFIG['CVD_LOOKBACK']:
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

    # ── Layer B: 2H CVD Zero-Line Cross ──
    if idx_2h >= 1 and 'cvd_zl_state' in df_2h.columns:
        zl_state = df_2h['cvd_zl_state'].iloc[idx_2h]
        cross_bar = df_2h['cvd_zl_cross_bar'].iloc[idx_2h]
        cross_dir = df_2h['cvd_zl_cross_dir'].iloc[idx_2h]
        cvd_2h_now = df_2h['cvd_2h'].iloc[idx_2h]

        if not pd.isna(cvd_2h_now):
            # Fresh cross = within momentum window
            bars_since = idx_2h - cross_bar if cross_bar >= 0 else 999
            layer_b_bars_since = bars_since
            fresh = bars_since <= CONFIG['M4_ZL_MOMENTUM_BARS']

            if direction == 'LONG':
                # Bullish: CVD crossed up, or is above zero with momentum
                if zl_state == 'CROSS_UP':
                    layer_b_score = 0.90 if fresh else 0.70
                    layer_b_status = 'PASS'
                    layer_b_cross = 'CROSS_UP'
                elif zl_state == 'ABOVE' and cross_dir == 'UP':
                    # Above zero after an up-cross
                    if bars_since <= CONFIG['M4_ZL_MOMENTUM_BARS']:
                        layer_b_score = 0.80
                        layer_b_status = 'PASS'
                        layer_b_cross = 'ABOVE_FRESH'
                    elif bars_since <= CONFIG['M4_ZL_LOOKBACK']:
                        layer_b_score = 0.65
                        layer_b_status = 'PASS'
                        layer_b_cross = 'ABOVE_AFTER_UP'
                    else:
                        layer_b_score = 0.50
                        layer_b_status = 'PASS'   # stale but CVD still positive = regime holds
                        layer_b_cross = 'ABOVE_STALE'
                elif zl_state == 'ABOVE':
                    # Above zero but no recorded cross — weak but positive regime
                    layer_b_score = 0.40
                    layer_b_status = 'PASS'
                    layer_b_cross = 'ABOVE_NO_CROSS'

            elif direction == 'SHORT':
                # Bearish: CVD crossed down, or is below zero with momentum
                if zl_state == 'CROSS_DOWN':
                    layer_b_score = 0.90 if fresh else 0.70
                    layer_b_status = 'PASS'
                    layer_b_cross = 'CROSS_DOWN'
                elif zl_state == 'BELOW' and cross_dir == 'DOWN':
                    if bars_since <= CONFIG['M4_ZL_MOMENTUM_BARS']:
                        layer_b_score = 0.80
                        layer_b_status = 'PASS'
                        layer_b_cross = 'BELOW_FRESH'
                    elif bars_since <= CONFIG['M4_ZL_LOOKBACK']:
                        layer_b_score = 0.65
                        layer_b_status = 'PASS'
                        layer_b_cross = 'BELOW_AFTER_DOWN'
                    else:
                        layer_b_score = 0.50
                        layer_b_status = 'PASS'   # stale but CVD still negative = regime holds
                        layer_b_cross = 'BELOW_STALE'
                elif zl_state == 'BELOW':
                    layer_b_score = 0.40
                    layer_b_status = 'PASS'
                    layer_b_cross = 'BELOW_NO_CROSS'

            # Conflict: CVD regime opposes trade direction
            if direction == 'LONG' and zl_state in ('BELOW', 'CROSS_DOWN'):
                if layer_b_status != 'PASS':  # not already scored above
                    layer_b_score = 0.20
                    layer_b_cross = f'CONFLICT_{zl_state}'
            elif direction == 'SHORT' and zl_state in ('ABOVE', 'CROSS_UP'):
                if layer_b_status != 'PASS':
                    layer_b_score = 0.20
                    layer_b_cross = f'CONFLICT_{zl_state}'

            # Boost: CVD slope after cross confirms direction
            if layer_b_status == 'PASS' and idx_2h >= 3:
                cvd_slope_2h = (df_2h['cvd_2h'].iloc[idx_2h] - df_2h['cvd_2h'].iloc[max(0, idx_2h-3)]) / 3
                if not pd.isna(cvd_slope_2h):
                    if (direction == 'LONG' and cvd_slope_2h > 0) or \
                       (direction == 'SHORT' and cvd_slope_2h < 0):
                        layer_b_score = min(layer_b_score * 1.15, 1.0)  # slope confirms

    # ── Combine Layers ──
    w_div = CONFIG['M4_DIV_WEIGHT']
    w_zl = CONFIG['M4_ZL_WEIGHT']
    combined = layer_a_score * w_div + layer_b_score * w_zl

    # At least one layer must PASS for M4 to PASS
    if layer_a_status == 'PASS' or layer_b_status == 'PASS':
        status = 'PASS'
    else:
        status = 'FAIL'
    
    score = max(combined, 0.50) if status == 'PASS' else 0.50
    
    details = {
        'layer_a_div': layer_a_div,
        'layer_a_score': round(layer_a_score, 3),
        'layer_b_cross': layer_b_cross,
        'layer_b_zl_state': zl_state if 'zl_state' in dir() else 'N/A',
        'layer_b_score': round(layer_b_score, 3),
        'layer_b_bars_since': layer_b_bars_since,
        'combined': round(combined, 3),
    }
    return status, score, details


# ═══════════════════════════════════════════════════════════════════════
# ICS COMPOSITE SCORE
# ═══════════════════════════════════════════════════════════════════════

def calc_ics(m1_score, m2_score, m3_score, m4_score, m4_status, m5_score=0.5, m6_score=0.5, m7_score=0.5, use_derivatives=False, use_m7=False, cascade_dir='NONE', cascade_strength=0.0):
    m4_contrib = m4_score if m4_status == 'PASS' else 0.5

    if use_m7 and CONFIG.get('M7_ENABLED', False):
        # Redistribute weights to include M7
        m6_w = CONFIG['M6_WEIGHT'] * 0.7
        m7_w = CONFIG['M7_WEIGHT']
        other_w = 1.0 - m6_w - m7_w
        base_sum = (CONFIG['M1_WEIGHT'] + CONFIG['M2_WEIGHT'] +
                    CONFIG['M3_WEIGHT'] + CONFIG['M4_WEIGHT'] + CONFIG['M5_WEIGHT'])
        ics = (
            m1_score * (CONFIG['M1_WEIGHT'] / base_sum * other_w) +
            m2_score * (CONFIG['M2_WEIGHT'] / base_sum * other_w) +
            m3_score * (CONFIG['M3_WEIGHT'] / base_sum * other_w) +
            m4_contrib * (CONFIG['M4_WEIGHT'] / base_sum * other_w) +
            m5_score * (CONFIG['M5_WEIGHT'] / base_sum * other_w) +
            m6_score * m6_w +
            m7_score * m7_w
        )
    elif use_derivatives and HAS_DERIVATIVES:
        ics = (m1_score * CONFIG['M1_WEIGHT'] +
               m2_score * CONFIG['M2_WEIGHT'] +
               m3_score * CONFIG['M3_WEIGHT_DERIV'] +
               m4_contrib * CONFIG['M4_WEIGHT_DERIV'] +
               m5_score * CONFIG['M5_WEIGHT'] +
               m6_score * CONFIG['M6_WEIGHT'])
    else:
        ics = (m1_score * CONFIG['M1_WEIGHT'] +
               m2_score * CONFIG['M2_WEIGHT'] +
               m3_score * CONFIG['M3_WEIGHT'] +
               m4_contrib * CONFIG['M4_WEIGHT'] +
               m5_score * CONFIG['M5_WEIGHT'])
    # v6.12: Cascade multiplier — boost or penalty based on cascade alignment
    if cascade_dir == 'WITH' and cascade_strength > 0:
        ics *= 1.0 + (CONFIG.get('CASCADE_MULTIPLIER', 1.12) - 1.0) * cascade_strength
    elif cascade_dir == 'AGAINST' and cascade_strength > 0:
        ics *= 1.0 - (1.0 - CONFIG.get('CASCADE_PENALTY', 0.85)) * cascade_strength
    effective_floor = CONFIG['ICS_FLOOR_M4_FALSE'] if m4_status == 'FAIL' else CONFIG['ICS_FLOOR']
    return ics, effective_floor


# ═══════════════════════════════════════════════════════════════════════
# ENTRY FILTERS
# ═══════════════════════════════════════════════════════════════════════

def check_entry_filters(df_15m, idx, direction, swing_bias, phase0_val, atr_1h):
    row = df_15m.iloc[idx]
    # Swing bias no longer blocks entries — it's too laggy and causes
    # the framework to miss recoveries (2023) and fight trends (2020).
    # Instead, swing_bias now only affects ICS floor via phase0.
    if not pd.isna(atr_1h) and atr_1h > 0:
        bar_move = abs(row['Close'] - row['Open'])
        if direction == 'LONG' and row['Close'] < row['Open']:
            if bar_move > CONFIG['BAR_MOVE_ATR'] * atr_1h:
                return False, "bar_move_against"
        if direction == 'SHORT' and row['Close'] > row['Open']:
            if bar_move > CONFIG['BAR_MOVE_ATR'] * atr_1h:
                return False, "bar_move_against"
    if not pd.isna(atr_1h) and row['Close'] > 0:
        if atr_1h / row['Close'] > CONFIG['ATR_FILTER_MAX']:
            return False, "atr_too_high"
    if phase0_val >= 0.90:
        return False, "phase0_red"
    return True, "ok"


def get_tp_multipliers(vol_ratio):
    return CONFIG['TP2_ATR'], CONFIG['TP3_ATR']


# ═══════════════════════════════════════════════════════════════════════
# TRADE CLASS
# ═══════════════════════════════════════════════════════════════════════

class Trade:
    def __init__(self, entry_time, direction, entry_price, sl, tp1, tp2, tp3,
                 size_pct, m1_dir, m2_status, m3_score, m4_status, m5_status, m5_score,
                 ics, phase0, reason):
        self.entry_time = entry_time
        self.direction = direction
        self.entry_price = entry_price
        self.sl = sl
        self.tp1 = tp1
        self.tp2 = tp2
        self.tp3 = tp3
        self.size_pct = size_pct
        self.m1_dir = m1_dir
        self.m2_status = m2_status
        self.m3_score = m3_score
        self.m4_status = m4_status
        self.m5_status = m5_status
        self.m5_score = m5_score
        self.ics = ics
        self.phase0 = phase0
        self.reason = reason
        self.remaining = 1.0
        self.tp1_hit = False
        self.tp2_hit = False
        self.exit_price = None
        self.exit_time = None
        self.exit_reason = None
        self.pnl_pct = 0.0
        self.bars_held = 0

    def update_sl_trail(self):
        if self.tp2_hit:
            self.sl = self.tp1
        elif self.tp1_hit:
            self.sl = self.entry_price

    def close(self, price, time, reason, fraction=1.0):
        close_amount = min(fraction, self.remaining)
        pnl = ((price - self.entry_price) / self.entry_price if self.direction == 'LONG'
               else (self.entry_price - price) / self.entry_price)
        self.pnl_pct += pnl * close_amount
        self.remaining -= close_amount
        self.exit_price = price
        self.exit_time = time
        self.exit_reason = reason
        if self.remaining <= 0.001:
            self.remaining = 0

    @property
    def is_open(self):
        return self.remaining > 0.001


# ═══════════════════════════════════════════════════════════════════════
# BACKTESTER
# ═══════════════════════════════════════════════════════════════════════

def run_backtest(csv_path, verbose=False, date_start=None, date_end=None):
    print("=" * 70)
    print("  JIMI FRAMEWORK v6.13 — Backtest Engine (M1-M7 + Liquidation)")
    if date_start or date_end:
        print(f"  Date Range: {date_start or 'start'} → {date_end or 'end'}")
    print("=" * 70)

    print("\n[1/6] Loading data...")
    df_15m = load_data(csv_path)
    print(f"  15m bars loaded: {len(df_15m):,}")
    print(f"  Date range: {df_15m['Open time'].iloc[0]} → {df_15m['Open time'].iloc[-1]}")

    print("[2/6] Resampling to 1H, 2H, 4H, 1D...")
    df_1h = resample_ohlcv(df_15m, '1H')
    df_2h = resample_ohlcv(df_15m, '2H')
    df_4h = resample_ohlcv(df_15m, '4H')
    df_1d = resample_ohlcv(df_15m, '1D')
    print(f"  1H: {len(df_1h):,} | 2H: {len(df_2h):,} | 4H: {len(df_4h):,} | 1D: {len(df_1d):,}")

    print("[3/6] Computing indicators...")
    df_15m['vwap'] = calc_vwap(df_15m['High'], df_15m['Low'], df_15m['Close'], df_15m['Volume'], CONFIG['VWAP_LOOKBACK'])
    df_15m['vol_ma20'] = df_15m['Volume'].rolling(20).mean()
    taker_base = df_15m['Taker buy base asset volume']
    total_vol = df_15m['Volume']
    df_15m['taker_ratio'] = (taker_base / total_vol.replace(0, np.nan)).fillna(CONFIG['TAKER_FILLNA'])
    df_15m['atr'] = calc_atr(df_15m['High'], df_15m['Low'], df_15m['Close'], CONFIG['ATR_PERIOD'])
    df_15m['vol_ratio'] = calc_vol_ratio(df_15m['Volume'])

    df_1h['macd_line'], df_1h['macd_signal'], df_1h['macd_hist'] = calc_macd(
        df_1h['Close'], CONFIG['MACD_FAST'], CONFIG['MACD_SLOW'], CONFIG['MACD_SIGNAL'])
    df_1h['ema_fast'] = calc_ema(df_1h['Close'], CONFIG['EMA_FAST'])
    df_1h['ema_slow'] = calc_ema(df_1h['Close'], CONFIG['EMA_SLOW'])
    df_1h['atr'] = calc_atr(df_1h['High'], df_1h['Low'], df_1h['Close'], CONFIG['ATR_PERIOD'])

    df_4h['ema_fast'] = calc_ema(df_4h['Close'], CONFIG['EMA_FAST'])
    df_4h['ema_slow'] = calc_ema(df_4h['Close'], CONFIG['EMA_SLOW'])
    df_2h['ema_fast'] = calc_ema(df_2h['Close'], CONFIG['EMA_FAST'])
    df_2h['ema_slow'] = calc_ema(df_2h['Close'], CONFIG['EMA_SLOW'])

    df_15m['cvd_15m'] = calc_cvd_15m(df_15m)
    df_15m['cvd_divergence_15m'] = detect_cvd_divergence_15m(df_15m, CONFIG['CVD_LOOKBACK'], CONFIG['CVD_DIVERGENCE_WINDOW'])
    print(f"  CVD divergences (15m): {(df_15m['cvd_divergence_15m']=='BULLISH').sum()} bullish, {(df_15m['cvd_divergence_15m']=='BEARISH').sum()} bearish")

    df_2h['cvd_2h'] = calc_cvd_2h(df_2h)
    df_2h['cvd_zl_state'], df_2h['cvd_zl_cross_bar'], df_2h['cvd_zl_cross_dir'] = detect_cvd_zero_cross(df_2h)
    zl_up = (df_2h['cvd_zl_state'] == 'CROSS_UP').sum()
    zl_down = (df_2h['cvd_zl_state'] == 'CROSS_DOWN').sum()
    print(f"  CVD zero-line (2H):    {zl_up} cross-up, {zl_down} cross-down")

    df_1d['swing_bias'] = calc_swing_bias(df_1d)
    df_1d['phase0'] = calc_phase0(df_1d)
    df_1d['trend'], df_1d['trend_score'] = calc_trend_state(df_1d)
    print("  Indicators computed.")

    # --- M7: Prepare market regime data ---
    m7_ethbtc_df, m7_btc_df = None, None
    if CONFIG.get('M7_ENABLED', False):
        print("[3b] Fetching M7 market regime data (ETH/BTC + BTC)...")
        m7_ethbtc_df, m7_btc_df = m7_prepare_data(df_15m)
        eb_n = len(m7_ethbtc_df) if m7_ethbtc_df is not None else 0
        bt_n = len(m7_btc_df) if m7_btc_df is not None else 0
        print(f"  M7 data: ETH/BTC={eb_n} days, BTC/USDT={bt_n} days")

    print("[4/7] Building timeframe index maps...")
    df_1h['_ts'] = df_1h['Open time'].values.astype('datetime64[ns]')
    df_2h['_ts'] = df_2h['Open time'].values.astype('datetime64[ns]')
    df_4h['_ts'] = df_4h['Open time'].values.astype('datetime64[ns]')
    df_1d['_ts'] = df_1d['Open time'].values.astype('datetime64[ns]')

    def find_tf_idx(ts, df_tf):
        idx = df_tf['_ts'].searchsorted(ts, side='right') - 1
        return max(idx, -1)

    warmup_time = df_1h['Open time'].iloc[min(CONFIG['WARMUP_BARS_1H'], len(df_1h)-1)]
    print(f"  Warmup: skip until {warmup_time}")

    print("[5/6] Running backtest...")
    trades, open_trades = [], []
    daily_trades, daily_pnl = {}, {}
    last_loss_time = None
    stats = {
        'signals_checked': 0, 'ics_blocked': 0, 'filter_blocked': 0, 'entries': 0,
        'exits_sl': 0, 'exits_tp1': 0, 'exits_tp2': 0, 'exits_tp3': 0, 'exits_signal': 0, 'exits_early': 0,
        'm4_false_anchored': 0, 'm5_pass': 0, 'm5_fail': 0, 'cascade_detected': 0,
        'm1_neutral_skip': 0, 'm3_fail': 0, 'm2_neutral_long_skip': 0, 'rolling_wr_skip': 0,
        'dedup_skip': 0, 'long_ics_skip': 0, 'consec_pause': 0,
        'ics_ceiling_skip': 0, 'm4_required_skip': 0, 'long_disabled': 0, 'long_phase0_skip': 0, 'long_m5_skip': 0,
        'bias_gate_skip': 0, 'monthly_dd_skip': 0, 'dir_veto_skip': 0, 'trend_block': 0, 'trend_weak': 0,
    }

    for idx in range(len(df_15m)):
        row = df_15m.iloc[idx]
        ts = row['Open time']
        if ts < warmup_time: continue
        if date_start and str(ts) < date_start: continue
        if date_end and str(ts) > date_end: continue
        if pd.isna(row['taker_ratio']) or pd.isna(row['atr']): continue

        idx_1h = find_tf_idx(ts, df_1h)
        idx_2h = find_tf_idx(ts, df_2h)
        idx_4h = find_tf_idx(ts, df_4h)
        idx_1d = find_tf_idx(ts, df_1d)
        if idx_1h < 1 or idx_2h < 0 or idx_4h < 0 or idx_1d < 0: continue

        atr_1h = df_1h['atr'].iloc[idx_1h]
        swing_bias = df_1d['swing_bias'].iloc[idx_1d]
        phase0_val = df_1d['phase0'].iloc[idx_1d]
        trend_dir = df_1d['trend'].iloc[idx_1d]
        trend_val = df_1d['trend_score'].iloc[idx_1d]

        # --- Check existing trades for SL/TP ---
        is_summer = ts.month in CONFIG.get('SUMMER_MONTHS', [6, 7, 8, 9])
        is_shoulder = ts.month in CONFIG.get('SHOULDER_MONTHS', [3, 10])
        for trade in open_trades[:]:
            if not trade.is_open: continue
            high, low = row['High'], row['Low']
            trade.bars_held += 1

            # Early exit: close losing trades that haven't moved after N bars
            early_exit_bars = CONFIG.get('EARLY_EXIT_BARS_SUMMER', CONFIG['EARLY_EXIT_BARS']) if is_summer else CONFIG['EARLY_EXIT_BARS']
            early_exit_loss = CONFIG.get('EARLY_EXIT_MIN_LOSS_SUMMER', CONFIG['EARLY_EXIT_MIN_LOSS']) if is_summer else CONFIG['EARLY_EXIT_MIN_LOSS']
            if trade.bars_held >= early_exit_bars and not trade.tp1_hit:
                current_pnl = ((row['Close'] - trade.entry_price) / trade.entry_price
                               if trade.direction == 'LONG'
                               else (trade.entry_price - row['Close']) / trade.entry_price)
                if current_pnl < -early_exit_loss:
                    trade.close(row['Close'], ts, 'EARLY_EXIT')
                    stats['exits_early'] = stats.get('exits_early', 0) + 1
                    continue

            if trade.direction == 'LONG' and low <= trade.sl:
                trade.close(trade.sl, ts, 'SL'); stats['exits_sl'] += 1; continue
            elif trade.direction == 'SHORT' and high >= trade.sl:
                trade.close(trade.sl, ts, 'SL'); stats['exits_sl'] += 1; continue

            if trade.tp1_hit and trade.tp2_hit:
                if (trade.direction == 'LONG' and high >= trade.tp3) or \
                   (trade.direction == 'SHORT' and low <= trade.tp3):
                    trade.close(trade.tp3, ts, 'TP3', trade.remaining); stats['exits_tp3'] += 1

            if trade.tp1_hit and not trade.tp2_hit:
                if trade.direction == 'LONG' and high >= trade.tp2:
                    frac = CONFIG['TP2_CLOSE'] / (1 - CONFIG['TP1_CLOSE'])
                    trade.close(trade.tp2, ts, 'TP2', frac); trade.tp2_hit = True; trade.update_sl_trail(); stats['exits_tp2'] += 1
                elif trade.direction == 'SHORT' and low <= trade.tp2:
                    frac = CONFIG['TP2_CLOSE'] / (1 - CONFIG['TP1_CLOSE'])
                    trade.close(trade.tp2, ts, 'TP2', frac); trade.tp2_hit = True; trade.update_sl_trail(); stats['exits_tp2'] += 1

            if not trade.tp1_hit:
                # v6.13: Higher TP1 close in summer/shoulder — lock in gains before reversal
                if is_summer:
                    tp1_close_frac = CONFIG.get('TP1_CLOSE_SUMMER', CONFIG['TP1_CLOSE'])
                elif is_shoulder:
                    tp1_close_frac = CONFIG.get('SHOULDER_TP1_CLOSE', CONFIG['TP1_CLOSE'])
                else:
                    tp1_close_frac = CONFIG['TP1_CLOSE']
                if trade.direction == 'LONG' and high >= trade.tp1:
                    trade.close(trade.tp1, ts, 'TP1', tp1_close_frac); trade.tp1_hit = True; trade.update_sl_trail(); stats['exits_tp1'] += 1
                elif trade.direction == 'SHORT' and low <= trade.tp1:
                    trade.close(trade.tp1, ts, 'TP1', tp1_close_frac); trade.tp1_hit = True; trade.update_sl_trail(); stats['exits_tp1'] += 1

        open_trades = [t for t in open_trades if t.is_open]

        # --- Risk checks ---
        today = ts.date()
        if today not in daily_trades: daily_trades[today] = 0; daily_pnl[today] = 0.0

        # --- Update daily PnL from closed trades ---
        today_closed = [t for t in trades if t.exit_time is not None and hasattr(t.exit_time, 'date') and t.exit_time.date() == today]
        daily_pnl[today] = sum(t.pnl_pct * t.size_pct for t in today_closed)
        max_trades_today = CONFIG.get('MAX_TRADES_DAY_SUMMER', CONFIG['MAX_TRADES_DAY']) if is_summer else (CONFIG.get('SHOULDER_MAX_TRADES_DAY', CONFIG['MAX_TRADES_DAY']) if is_shoulder else CONFIG['MAX_TRADES_DAY'])
        if daily_trades[today] >= max_trades_today: continue
        max_daily_loss = CONFIG.get('MAX_DAILY_LOSS_SUMMER', CONFIG['MAX_DAILY_LOSS']) if is_summer else CONFIG['MAX_DAILY_LOSS']
        if daily_pnl[today] <= -max_daily_loss: continue
        cooldown = CONFIG.get('COOLDOWN_MINUTES_SUMMER', CONFIG['COOLDOWN_MINUTES']) if is_summer else (CONFIG.get('SHOULDER_COOLDOWN', CONFIG['COOLDOWN_MINUTES']) if is_shoulder else CONFIG['COOLDOWN_MINUTES'])
        if last_loss_time and (ts - last_loss_time).total_seconds() / 60 < cooldown: continue
        phase0_block = CONFIG.get('PHASE0_SUMMER_BLOCK', 0.90) if is_summer else 0.90
        if phase0_val >= phase0_block: continue

        # === TREND FILTER — block counter-trend trades ===
        if CONFIG.get('TREND_FILTER_ENABLED', False):
            _trend_is_bull = trend_dir in ('STRONG_UP', 'UP')
            _trend_is_bear = trend_dir in ('STRONG_DOWN', 'DOWN')
            _trend_is_strong = trend_dir in ('STRONG_UP', 'STRONG_DOWN')
            
            # In strong trends, only trade with the trend
            if CONFIG.get('TREND_BLOCK_COUNTER_TREND', False):
                if _trend_is_bear and direction == 'LONG':
                    stats['trend_block'] = stats.get('trend_block', 0) + 1
                    continue
                if _trend_is_bull and direction == 'SHORT':
                    stats['trend_block'] = stats.get('trend_block', 0) + 1
                    continue
            
            # Require minimum trend score for entry
            min_score = CONFIG.get('TREND_MIN_SCORE', 0.15)
            if abs(trend_val) < min_score:
                stats['trend_weak'] = stats.get('trend_weak', 0) + 1
                continue

        # --- Consecutive Loss Pause ---
        max_consec = CONFIG.get('MAX_CONSEC_LOSS_SUMMER', 999) if is_summer else CONFIG.get('MAX_CONSEC_LOSS', 999)
        if max_consec < 999 and len(trades) >= max_consec:
            recent = trades[-max_consec:]
            if all(t.pnl_pct < 0 for t in recent):
                last_exit = max(t.exit_time for t in recent if t.exit_time is not None)
                if last_exit is not None and hasattr(last_exit, 'total_seconds'):
                    pause_bars = CONFIG.get('CONSEC_LOSS_PAUSE_SUMMER', 8) if is_summer else CONFIG.get('CONSEC_LOSS_PAUSE_BARS', 8)
                    if (ts - last_exit).total_seconds() / 900 < pause_bars:
                        stats['consec_pause'] = stats.get('consec_pause', 0) + 1
                        continue

        # --- v6.11: Monthly Drawdown Circuit Breaker ---
        monthly_dd_limit = CONFIG.get('MONTHLY_DD_CIRCUIT', 0)
        if monthly_dd_limit > 0:
            month_key = f"{ts.year}-{ts.month:02d}"
            month_trades = [t for t in trades if t.exit_time is not None and hasattr(t.exit_time, 'year') and f"{t.exit_time.year}-{t.exit_time.month:02d}" == month_key]
            month_pnl = sum(t.pnl_pct * t.size_pct for t in month_trades)
            if month_pnl <= -monthly_dd_limit:
                stats['monthly_dd_skip'] = stats.get('monthly_dd_skip', 0) + 1
                continue

        stats['signals_checked'] += 1

        # --- Module Scoring ---
        m1_dir, m1_score = score_m1(df_1h, idx_1h)
        m2_status, m2_score = score_m2(df_1h, df_2h, df_4h, df_1d, idx_1h, idx_2h, idx_4h, idx_1d)

        if m1_dir == 'BULLISH': direction = 'LONG'
        elif m1_dir == 'BEARISH': direction = 'SHORT'
        else: stats['m1_neutral_skip'] += 1; continue

        m3_status, m3_score, m3_entry = score_m3(df_15m, idx, direction)
        if m3_status == 'FAIL': stats['m3_fail'] += 1; continue

        # --- LONG CONFLUENCE FILTER ---
        # LONG trades underperform — require stronger confluence
        if direction == 'LONG' and m2_status == 'NEUTRAL':
            stats['m2_neutral_long_skip'] = stats.get('m2_neutral_long_skip', 0) + 1
            continue

        m4_status, m4_score, m4_div = score_m4(df_15m, df_2h, idx, idx_2h, direction)

        # --- Pre-M5 ICS check (v6.12: no cascade info yet, use neutral) ---
        ics_pre, effective_floor = calc_ics(m1_score, m2_score, m3_score, m4_score, m4_status, 0.5)
        if m4_status == 'FAIL': stats['m4_false_anchored'] += 1
        threshold = CONFIG['ICS_THRESHOLD_CAUTION'] if phase0_val >= 0.40 else CONFIG['ICS_THRESHOLD_NORMAL']
        # v6.13: Summer/shoulder ICS boost — require higher confidence in choppy months
        if is_summer:
            threshold += CONFIG.get('SUMMER_ICS_BOOST', 0)
        elif is_shoulder:
            threshold += CONFIG.get('SHOULDER_ICS_BOOST', 0)
        if ics_pre < effective_floor or ics_pre < threshold: stats['ics_blocked'] += 1; continue

        # --- M5 (lazy, cached every 4 bars) ---
        m5_cache_key = idx // 4
        if not hasattr(score_m5, '_cache') or score_m5._cache_key != m5_cache_key:
            m5_status, m5_score, m5_details = score_m5(df_15m, idx, direction,
                n_bins=CONFIG['M5_VP_BINS'], lookback=CONFIG['M5_VP_LOOKBACK'])
            cascade = detect_cascade_setup(df_15m, idx)
            score_m5._cache = (m5_status, m5_score, m5_details, cascade)
            score_m5._cache_key = m5_cache_key
        else:
            m5_status, m5_score, m5_details, cascade = score_m5._cache

        if m5_status == 'PASS': stats['m5_pass'] += 1
        else: stats['m5_fail'] += 1
        if cascade.get('cascade'): stats['cascade_detected'] += 1

        # v6.12: Extract cascade info from M5 details (already computed inside score_m5)
        cascade_dir = m5_details.get('cascade_dir', 'NONE') if isinstance(m5_details, dict) else 'NONE'
        cascade_strength = m5_details.get('cascade_strength', 0.0) if isinstance(m5_details, dict) else 0.0

        # ===== M7: Market Regime Scoring =====
        m7_score = 0.5
        m7_status = 'SKIP'
        if CONFIG.get('M7_ENABLED', False) and m7_ethbtc_df is not None:
            eb_row, bt_row = m7_get_row(m7_ethbtc_df, m7_btc_df, ts)
            m7_status, m7_score, m7_details = score_m7(eb_row, bt_row, row.get('vol_ratio', np.nan), direction)
            stats['m7_pass'] = stats.get('m7_pass', 0) + (1 if m7_status == 'PASS' else 0)
            stats['m7_fail'] = stats.get('m7_fail', 0) + (1 if m7_status == 'FAIL' else 0)

        use_m7 = CONFIG.get('M7_ENABLED', False) and m7_ethbtc_df is not None
        ics, effective_floor = calc_ics(m1_score, m2_score, m3_score, m4_score, m4_status, m5_score,
                                        m7_score=m7_score, use_m7=use_m7,
                                        cascade_dir=cascade_dir, cascade_strength=cascade_strength)
        if ics < effective_floor or ics < threshold: stats['ics_blocked'] += 1; continue

        # --- ICS Ceiling: reject overconfident signals ---
        ics_ceiling = CONFIG.get('ICS_CEILING', 1.0)
        if ics > ics_ceiling: stats['ics_ceiling_skip'] = stats.get('ics_ceiling_skip', 0) + 1; continue

        # --- M4 PASS Required ---
        if m4_status == 'FAIL': stats['m4_required_skip'] = stats.get('m4_required_skip', 0) + 1; continue

        # --- v6.12: Directional Veto ---
        # If M4 CVD + M5 magnet BOTH disagree with M1/M2 direction, block regardless of ICS
        if CONFIG.get('DIR_VETO_ENABLED', False):
            m4_disagree = (direction == 'LONG' and m4_div == 'BEARISH') or \
                          (direction == 'SHORT' and m4_div == 'BULLISH')
            m5_disagree = (m5_status == 'FAIL')  # M5 FAIL means magnet doesn't support direction
            if m4_disagree and m5_disagree:
                stats['dir_veto_skip'] = stats.get('dir_veto_skip', 0) + 1
                continue

        # --- v6.11: Directional Bias Gate (seasonal — bad months only) ---
        # Only restrict LONGs when bias is BEARISH during historically bad months (Mar, Jul, Sep)
        bad_gate_months = [3, 7, 9]
        if CONFIG.get('BIAS_GATE_ENABLED', False) and direction == 'LONG' and swing_bias == 'BEARISH' and ts.month in bad_gate_months:
            bias_ics = CONFIG.get('BIAS_GATE_LONG_ICS', 0.65)
            if ics < bias_ics:
                stats['bias_gate_skip'] = stats.get('bias_gate_skip', 0) + 1
                continue

        # Symmetric direction filter — both long and short treated equally.
        # (Removed: hardcoded long_min_ics=0.65, m2_neutral_long_skip, long_phase0_skip)

        passed, reason = check_entry_filters(df_15m, idx, direction, swing_bias, phase0_val, atr_1h)
        if not passed:
            stats['filter_blocked'] += 1
            if verbose and stats['filter_blocked'] <= 20: print(f"  FILTER: {ts} {direction} blocked: {reason}")
            continue

        # --- Entry Dedup: skip if too close to last entry price ---
        min_dist = CONFIG.get('MIN_ENTRY_DIST_PCT', 0)
        if min_dist > 0 and trades:
            last_trade = trades[-1]
            price_dist = abs(row['Close'] - last_trade.entry_price) / last_trade.entry_price
            if price_dist < min_dist:
                stats['dedup_skip'] = stats.get('dedup_skip', 0) + 1
                continue

        # --- Position Sizing ---
        size = CONFIG.get('SIZE_LONG', CONFIG['SIZE_STD']) if direction == 'LONG' else CONFIG['SIZE_STD']
        if m2_status == 'NEUTRAL': size *= CONFIG['SIZE_M2_NEUTRAL']
        if phase0_val >= 0.40: size *= CONFIG['SIZE_CAUTION']
        # v6.13: Reduce size in summer chop months
        if is_summer:
            size *= CONFIG.get('SUMMER_SIZE_MULT', 1.0)
        elif is_shoulder:
            size *= CONFIG.get('SHOULDER_SIZE_MULT', 1.0)
        # v6.13: M7 size adjustment — reduce when macro regime is unfavorable
        if CONFIG.get('M7_ENABLED', False) and m7_ethbtc_df is not None:
            if m7_score < 0.35:
                size *= CONFIG.get('M7_SIZE_REDUCTION', 0.70)
            elif m7_score < 0.45:
                size *= CONFIG.get('M7_SIZE_MILD', 0.85)
        if size < 0.01: continue

        entry_price = row['Close']
        atr_for_sl = atr_1h if not pd.isna(atr_1h) else row['atr']
        # v6.13: Use tighter SL in summer
        if is_summer:
            sl_std = CONFIG.get('SL_ATR_STD_SUMMER', CONFIG['SL_ATR_STD'])
            sl_hard_max = CONFIG.get('SL_HARD_MAX_SUMMER', CONFIG['SL_HARD_MAX_PCT'])
        elif is_shoulder:
            sl_std = CONFIG.get('SHOULDER_SL_ATR', CONFIG['SL_ATR_STD'])
            sl_hard_max = CONFIG.get('SHOULDER_SL_HARD_MAX', CONFIG['SL_HARD_MAX_PCT'])
        else:
            sl_std = CONFIG['SL_ATR_STD']
            sl_hard_max = CONFIG['SL_HARD_MAX_PCT']
        sl_dist = min(sl_std * atr_for_sl, sl_hard_max * entry_price)
        # v6.13: Use tighter TP1 in summer to lock in gains faster
        if is_summer:
            tp1_atr = CONFIG.get('TP1_ATR_SUMMER', CONFIG['TP1_ATR'])
        elif is_shoulder:
            tp1_atr = CONFIG.get('SHOULDER_TP1_ATR', CONFIG['TP1_ATR'])
        else:
            tp1_atr = CONFIG['TP1_ATR']
        tp1_dist = tp1_atr * atr_for_sl
        tp2_mult, tp3_mult = get_tp_multipliers(row.get('vol_ratio', np.nan))
        tp2_dist, tp3_dist = tp2_mult * atr_for_sl, tp3_mult * atr_for_sl

        if direction == 'LONG':
            sl, tp1, tp2, tp3 = entry_price - sl_dist, entry_price + tp1_dist, entry_price + tp2_dist, entry_price + tp3_dist
        else:
            sl, tp1, tp2, tp3 = entry_price + sl_dist, entry_price - tp1_dist, entry_price - tp2_dist, entry_price - tp3_dist

        trade = Trade(ts, direction, entry_price, sl, tp1, tp2, tp3, size,
                      m1_dir, m2_status, m3_score, m4_status, m5_status, m5_score,
                      ics, phase0_val,
                      f"M1={m1_dir} M2={m2_status} M3={m3_status} M4={m4_status} M5={m5_status} M7={m7_status}({m7_score:.2f}) ICS={ics:.3f}")
        open_trades.append(trade); trades.append(trade)
        daily_trades[today] += 1; stats['entries'] += 1

        if verbose and stats['entries'] <= 50:
            print(f"  ENTRY #{stats['entries']}: {ts} {direction} @ {entry_price:.2f} "
                  f"SL={sl:.2f} TP1={tp1:.2f} ICS={ics:.3f} M5={m5_status}({m5_score:.2f}) M7={m7_score:.2f} size={size:.2f}")

    if open_trades:
        last_row = df_15m.iloc[-1]
        for trade in open_trades:
            if trade.is_open:
                trade.close(last_row['Close'], last_row['Open time'], 'END'); stats['exits_signal'] += 1

    print("\n[7/7] Computing results...")
    m7p = stats.get('m7_pass', 0)
    m7f = stats.get('m7_fail', 0)
    if m7p + m7f > 0:
        print(f"  M7: {m7p} pass / {m7f} fail ({m7p/(m7p+m7f)*100:.1f}% pass rate)")
    return trades, stats, df_15m


# ═══════════════════════════════════════════════════════════════════════
# PERFORMANCE REPORT
# ═══════════════════════════════════════════════════════════════════════

def print_report(trades, stats):
    if not trades:
        print("\n  No trades generated."); return

    total = len(trades)
    winners = [t for t in trades if t.pnl_pct > 0]
    losers = [t for t in trades if t.pnl_pct < 0]
    win_rate = len(winners) / total * 100
    total_pnl = sum(t.pnl_pct * t.size_pct for t in trades)
    avg_win = np.mean([t.pnl_pct for t in winners]) if winners else 0
    avg_loss = np.mean([t.pnl_pct for t in losers]) if losers else 0
    gross_profit = sum(t.pnl_pct * t.size_pct for t in winners)
    gross_loss = abs(sum(t.pnl_pct * t.size_pct for t in losers))
    profit_factor = gross_profit / gross_loss if gross_loss > 0 else float('inf')

    equity = [0]
    for t in sorted(trades, key=lambda x: x.exit_time):
        equity.append(equity[-1] + t.pnl_pct * t.size_pct)
    equity = np.array(equity)
    peak = np.maximum.accumulate(equity)
    max_dd = abs((equity - peak).min()) if len(equity) > 0 else 0
    ret_dd = total_pnl / max_dd if max_dd > 0 else float('inf')

    longs = [t for t in trades if t.direction == 'LONG']
    shorts = [t for t in trades if t.direction == 'SHORT']
    long_wr = len([t for t in longs if t.pnl_pct > 0]) / len(longs) * 100 if longs else 0
    short_wr = len([t for t in shorts if t.pnl_pct > 0]) / len(shorts) * 100 if shorts else 0

    tp1_count = len([t for t in trades if t.tp1_hit])
    tp2_count = len([t for t in trades if t.tp2_hit])
    tp3_count = len([t for t in trades if t.exit_reason == 'TP3'])
    sl_count = len([t for t in trades if t.exit_reason == 'SL'])

    m4_pass = len([t for t in trades if t.m4_status == 'PASS'])
    m4_fail = len([t for t in trades if t.m4_status == 'FAIL'])
    m5_pass = len([t for t in trades if t.m5_status == 'PASS'])
    m5_fail = len([t for t in trades if t.m5_status == 'FAIL'])
    m5_avg = np.mean([t.m5_score for t in trades])

    print("\n" + "═" * 70)
    print("  JIMI v6.10 — BACKTEST RESULTS (5 Modules + Liquidation Magnet)")
    print("═" * 70)

    print(f"\n  {'Total Trades:':<28} {total}")
    print(f"  {'Winners:':<28} {len(winners)} ({win_rate:.1f}%)")
    print(f"  {'Losers:':<28} {len(losers)}")
    print(f"\n  {'Net PnL (weighted):':<28} {total_pnl*100:.2f}%")
    print(f"  {'Avg Win:':<28} {avg_win*100:.2f}%")
    print(f"  {'Avg Loss:':<28} {avg_loss*100:.2f}%")
    print(f"  {'Profit Factor:':<28} {profit_factor:.2f}")
    print(f"  {'Max Drawdown:':<28} {max_dd*100:.2f}%")
    print(f"  {'Return/DD Ratio:':<28} {ret_dd:.1f}×")
    print(f"\n  {'Avg Bars Held:':<28} {np.mean([t.bars_held for t in trades]):.1f}")
    print(f"\n  Direction Breakdown:")
    print(f"    LONG:  {len(longs)} trades, WR {long_wr:.1f}%")
    print(f"    SHORT: {len(shorts)} trades, WR {short_wr:.1f}%")
    print(f"\n  M4 CVD Status:")
    print(f"    PASS: {m4_pass}  |  FAIL: {m4_fail}")
    print(f"\n  M5 Liquidation Magnet:")
    print(f"    PASS: {m5_pass}  |  FAIL: {m5_fail}  |  Avg score: {m5_avg:.3f}")
    print(f"\n  Exit Breakdown:")
    print(f"    TP1: {tp1_count} ({tp1_count/total*100:.1f}%)  TP2: {tp2_count} ({tp2_count/total*100:.1f}%)  TP3: {tp3_count} ({tp3_count/total*100:.1f}%)  SL: {sl_count} ({sl_count/total*100:.1f}%)")
    early_count = len([t for t in trades if t.exit_reason == 'EARLY_EXIT'])
    if early_count:
        print(f"    EARLY_EXIT: {early_count} ({early_count/total*100:.1f}%)")
    print(f"\n  Signal Flow:")
    for k in ['signals_checked','m1_neutral_skip','m3_fail','m2_neutral_long_skip','long_disabled','long_ics_skip','long_m5_skip','long_phase0_skip','ics_blocked','ics_ceiling_skip','m4_required_skip','m4_false_anchored','m5_pass','m5_fail','cascade_detected','dedup_skip','filter_blocked','consec_pause','bias_gate_skip','monthly_dd_skip','dir_veto_skip','entries']:
        if k in stats:
            print(f"    {k+':':<26} {stats[k]}")

    monthly = {}
    for t in trades:
        mk = t.entry_time.strftime('%Y-%m')
        if mk not in monthly: monthly[mk] = {'count': 0, 'pnl': 0, 'wins': 0}
        monthly[mk]['count'] += 1; monthly[mk]['pnl'] += t.pnl_pct * t.size_pct
        if t.pnl_pct > 0: monthly[mk]['wins'] += 1

    print(f"\n  Monthly Performance:")
    print(f"    {'Month':<10} {'Trades':>7} {'WR':>7} {'PnL':>10}")
    print(f"    {'─'*10} {'─'*7} {'─'*7} {'─'*10}")
    for month in sorted(monthly.keys()):
        m = monthly[month]
        wr = m['wins'] / m['count'] * 100 if m['count'] > 0 else 0
        print(f"    {month:<10} {m['count']:>7} {wr:>6.1f}% {m['pnl']*100:>9.2f}%")
    print("═" * 70)

    return {'total_trades': total, 'win_rate': win_rate, 'total_pnl': total_pnl,
            'profit_factor': profit_factor, 'max_drawdown': max_dd, 'return_dd_ratio': ret_dd}


def export_trades(trades, filepath):
    rows = []
    for t in trades:
        rows.append({
            'entry_time': t.entry_time, 'exit_time': t.exit_time, 'direction': t.direction,
            'entry_price': t.entry_price, 'exit_price': t.exit_price,
            'sl': t.sl, 'tp1': t.tp1, 'tp2': t.tp2, 'tp3': t.tp3,
            'pnl_pct': t.pnl_pct * 100, 'size_pct': t.size_pct, 'bars_held': t.bars_held,
            'ics': t.ics, 'm1_dir': t.m1_dir, 'm2_status': t.m2_status,
            'm4_status': t.m4_status, 'm5_status': t.m5_status, 'm5_score': t.m5_score,
            'phase0': t.phase0, 'exit_reason': t.exit_reason, 'reason': t.reason,
        })
    pd.DataFrame(rows).to_csv(filepath, index=False)
    print(f"\n  Trade log exported: {filepath}")


# ═══════════════════════════════════════════════════════════════════════
# LIVE SCANNER
# ═══════════════════════════════════════════════════════════════════════

def fetch_recent(symbol='ETH/USDT', timeframe='15m', bars=1000):
    import ccxt
    exchange = ccxt.binance({'enableRateLimit': True})
    symbol_raw = symbol.replace('/', '')
    raw = exchange.publicGetKlines({
        'symbol': symbol_raw,
        'interval': timeframe,
        'limit': bars,
    })
    rows = []
    for c in raw:
        rows.append({
            'Open time': pd.to_datetime(int(c[0]), unit='ms'),
            'Open': float(c[1]), 'High': float(c[2]), 'Low': float(c[3]),
            'Close': float(c[4]), 'Volume': float(c[5]),
            'Close time': pd.to_datetime(int(c[6]), unit='ms'),
            'Quote asset volume': float(c[7]),
            'Number of trades': int(c[8]),
            'Taker buy base asset volume': float(c[9]),
            'Taker buy quote asset volume': float(c[10]),
        })
    df = pd.DataFrame(rows)
    return df

def compute_indicators(df_15m):
    df_15m['vwap'] = calc_vwap(df_15m['High'], df_15m['Low'], df_15m['Close'], df_15m['Volume'], CONFIG['VWAP_LOOKBACK'])
    df_15m['vol_ma20'] = df_15m['Volume'].rolling(20).mean()
    taker_base = df_15m['Taker buy base asset volume']
    total_vol = df_15m['Volume']
    df_15m['taker_ratio'] = (taker_base / total_vol.replace(0, np.nan)).fillna(CONFIG['TAKER_FILLNA'])
    df_15m['atr'] = calc_atr(df_15m['High'], df_15m['Low'], df_15m['Close'], CONFIG['ATR_PERIOD'])
    df_15m['vol_ratio'] = calc_vol_ratio(df_15m['Volume'])

    df_1h = resample_ohlcv(df_15m, '1H')
    df_2h = resample_ohlcv(df_15m, '2H')
    df_4h = resample_ohlcv(df_15m, '4H')
    df_1d = resample_ohlcv(df_15m, '1D')

    df_1h['macd_line'], df_1h['macd_signal'], df_1h['macd_hist'] = calc_macd(
        df_1h['Close'], CONFIG['MACD_FAST'], CONFIG['MACD_SLOW'], CONFIG['MACD_SIGNAL'])
    df_1h['ema_fast'] = calc_ema(df_1h['Close'], CONFIG['EMA_FAST'])
    df_1h['ema_slow'] = calc_ema(df_1h['Close'], CONFIG['EMA_SLOW'])
    df_1h['atr'] = calc_atr(df_1h['High'], df_1h['Low'], df_1h['Close'], CONFIG['ATR_PERIOD'])
    df_4h['ema_fast'] = calc_ema(df_4h['Close'], CONFIG['EMA_FAST'])
    df_4h['ema_slow'] = calc_ema(df_4h['Close'], CONFIG['EMA_SLOW'])
    df_2h['ema_fast'] = calc_ema(df_2h['Close'], CONFIG['EMA_FAST'])
    df_2h['ema_slow'] = calc_ema(df_2h['Close'], CONFIG['EMA_SLOW'])
    df_15m['cvd_15m'] = calc_cvd_15m(df_15m)
    df_15m['cvd_divergence_15m'] = detect_cvd_divergence_15m(df_15m, CONFIG['CVD_LOOKBACK'], CONFIG['CVD_DIVERGENCE_WINDOW'])
    df_2h['cvd_2h'] = calc_cvd_2h(df_2h)
    df_2h['cvd_zl_state'], df_2h['cvd_zl_cross_bar'], df_2h['cvd_zl_cross_dir'] = detect_cvd_zero_cross(df_2h)
    df_1d['swing_bias'] = calc_swing_bias(df_1d)
    df_1d['phase0'] = calc_phase0(df_1d)
    df_1d['trend'], df_1d['trend_score'] = calc_trend_state(df_1d)
    return df_15m, df_1h, df_2h, df_4h, df_1d

def scan_signal(df_15m, df_1h, df_2h, df_4h, df_1d):
    idx = len(df_15m) - 1
    row = df_15m.iloc[idx]
    ts = row['Open time']
    idx_1h, idx_2h, idx_4h, idx_1d = len(df_1h)-1, len(df_2h)-1, len(df_4h)-1, len(df_1d)-1
    atr_1h = df_1h['atr'].iloc[idx_1h]
    swing_bias = df_1d['swing_bias'].iloc[idx_1d]
    phase0_val = df_1d['phase0'].iloc[idx_1d]

    m1_dir, m1_score = score_m1(df_1h, idx_1h)
    m2_status, m2_score = score_m2(df_1h, df_2h, df_4h, df_1d, idx_1h, idx_2h, idx_4h, idx_1d)
    direction = 'LONG' if m1_dir == 'BULLISH' else 'SHORT' if m1_dir == 'BEARISH' else None

    result = {
        'timestamp': str(ts), 'price': float(row['Close']),
        'swing_bias': swing_bias, 'phase0': float(phase0_val) if not pd.isna(phase0_val) else None,
        'm1': {'direction': m1_dir, 'score': float(m1_score)},
        'm2': {'status': m2_status, 'score': float(m2_score)},
        'direction': direction,
    }

    # --- Market Microstructure Data ---
    vwap_val = row.get('vwap', None)
    taker_val = row.get('taker_ratio', None)
    atr_val = row.get('atr', None)
    vol_ratio_val = row.get('vol_ratio', None)
    result['vwap'] = float(vwap_val) if vwap_val is not None and not pd.isna(vwap_val) else None
    result['vwap_dist_pct'] = float((row['Close'] - vwap_val) / vwap_val * 100) if vwap_val and not pd.isna(vwap_val) else None
    result['taker_ratio'] = float(taker_val) if taker_val is not None and not pd.isna(taker_val) else None
    result['atr_1h'] = float(atr_1h) if not pd.isna(atr_1h) else None
    result['vol_ratio'] = float(vol_ratio_val) if vol_ratio_val is not None and not pd.isna(vol_ratio_val) else None

    # --- Volume Profile & Magnets ---
    highs = df_15m['High'].values.astype(float)
    lows = df_15m['Low'].values.astype(float)
    closes = df_15m['Close'].values.astype(float)
    volumes = df_15m['Volume'].values.astype(float)
    bin_centers, vol_profile, bin_edges = build_volume_profile(
        highs[:idx+1], lows[:idx+1], closes[:idx+1], volumes[:idx+1],
        n_bins=CONFIG['M5_VP_BINS'], lookback=CONFIG['M5_VP_LOOKBACK'])
    magnets = find_magnets(bin_centers, vol_profile) if bin_centers is not None else []
    gaps = find_gaps(bin_centers, vol_profile) if bin_centers is not None else []
    result['magnets'] = [(round(p, 2), round(s, 2)) for p, _, s in magnets[:5]]
    result['gaps'] = [round(p, 2) for p, _ in gaps[:5]]

    # --- Support / Resistance (behavioral levels) ---
    sr_levels = find_support_resistance(df_15m, idx)
    # Sort by strength (most significant first), keep top 8
    sr_levels.sort(key=lambda x: x[1], reverse=True)
    result['sr_levels'] = [(round(p, 2), round(s, 2), t, touches, bounces)
                           for p, s, touches, bounces, t in sr_levels[:8]]

    # --- Estimated Liquidation Levels ---
    # Based on ATR-derived SL distances for typical leverage
    price = float(row['Close'])
    atr_for_liq = float(atr_1h) if not pd.isna(atr_1h) else float(row['atr'])
    liq_levels = {}
    for lev in [5, 10, 20, 50]:
        # Approximate liquidation distance = margin / leverage
        # For isolated margin: ~100% margin → liq at ~price * (1 - 1/leverage)
        long_liq = price * (1 - 0.9/lev)   # 90% of theoretical (maintenance margin)
        short_liq = price * (1 + 0.9/lev)
        liq_levels[f'x{lev}'] = {
            'long': round(long_liq, 2),
            'short': round(short_liq, 2),
            'long_dist_pct': round((price - long_liq) / price * 100, 2),
            'short_dist_pct': round((short_liq - price) / price * 100, 2),
        }
    result['liquidation_levels'] = liq_levels

    # --- Derivatives Data (always fetch for context) ---
    deriv_summary = {}
    use_deriv = CONFIG.get('DERIV_ENABLED', False) and HAS_DERIVATIVES
    if use_deriv:
        try:
            deriv_summary = get_derivatives_summary()
            if 'error' not in deriv_summary:
                result['derivatives'] = deriv_summary
            else:
                deriv_summary = {}
        except Exception:
            pass

    if direction is None:
        result['status'] = 'NO_SIGNAL'; result['reason'] = 'M1 neutral'; return result

    m3_status, m3_score, _ = score_m3(df_15m, idx, direction)
    result['m3'] = {'status': m3_status, 'score': float(m3_score)}

    m4_status, m4_score, m4_div = score_m4(df_15m, df_2h, idx, idx_2h, direction)
    result['m4'] = {'status': m4_status, 'score': float(m4_score), 'details': m4_div}

    m5_status, m5_score, m5_details = score_m5(df_15m, idx, direction,
        n_bins=CONFIG['M5_VP_BINS'], lookback=CONFIG['M5_VP_LOOKBACK'])
    cascade = detect_cascade_setup(df_15m, idx)
    result['m5'] = {'status': m5_status, 'score': float(m5_score), 'details': m5_details}
    result['cascade'] = cascade

    # --- M6: Derivatives Scoring ---
    m6_score = 0.5
    m6_status = 'SKIP'
    if use_deriv and deriv_summary and 'error' not in deriv_summary:
        try:
            m6_status, m6_score, _ = score_derivatives(deriv_summary, direction)
        except Exception:
            pass

    # --- M7: Market Regime Scoring ---
    m7_score = 0.5
    m7_status = 'SKIP'
    use_m7 = CONFIG.get('M7_ENABLED', False)
    if use_m7:
        try:
            m7_ethbtc, m7_btc = m7_prepare_data(df_15m)
            eb_row, bt_row = m7_get_row(m7_ethbtc, m7_btc, row['Open time'])
            m7_status, m7_score, m7_details = score_m7(eb_row, bt_row, row.get('vol_ratio', np.nan), direction)
        except Exception:
            use_m7 = False

    ics, effective_floor = calc_ics(m1_score, m2_score, m3_score, m4_score, m4_status, m5_score,
                                     m6_score, m7_score=m7_score, use_derivatives=use_deriv, use_m7=use_m7)
    result['ics'] = float(ics)
    result['effective_floor'] = float(effective_floor)
    result['m6'] = {'status': m6_status, 'score': float(m6_score)}
    result['m7'] = {'status': m7_status, 'score': float(m7_score)}

    threshold = CONFIG['ICS_THRESHOLD_CAUTION'] if phase0_val and phase0_val >= 0.40 else CONFIG['ICS_THRESHOLD_NORMAL']
    result['threshold'] = float(threshold)

    # Gate checks (all 5 modules always computed above)
    if m3_status == 'FAIL':
        result['status'] = 'NO_SIGNAL'; result['reason'] = 'M3 VWAP fail'; return result
    if ics < effective_floor or ics < threshold:
        result['status'] = 'NO_SIGNAL'; result['reason'] = f'ICS {ics:.3f} < {threshold:.2f}'; return result

    passed, reason = check_entry_filters(df_15m, idx, direction, swing_bias, phase0_val, atr_1h)
    if not passed:
        result['status'] = 'FILTERED'; result['reason'] = reason; return result

    entry_price = float(row['Close'])
    atr_for_sl = float(atr_1h) if not pd.isna(atr_1h) else float(row['atr'])
    sl_dist = min(CONFIG['SL_ATR_STD'] * atr_for_sl, CONFIG['SL_HARD_MAX_PCT'] * entry_price)
    tp1_dist = CONFIG['TP1_ATR'] * atr_for_sl
    tp2_mult, tp3_mult = get_tp_multipliers(row.get('vol_ratio', np.nan))
    tp2_dist, tp3_dist = tp2_mult * atr_for_sl, tp3_mult * atr_for_sl

    if direction == 'LONG':
        sl, tp1, tp2, tp3 = entry_price - sl_dist, entry_price + tp1_dist, entry_price + tp2_dist, entry_price + tp3_dist
    else:
        sl, tp1, tp2, tp3 = entry_price + sl_dist, entry_price - tp1_dist, entry_price - tp2_dist, entry_price - tp3_dist

    result.update({'status': 'SIGNAL', 'entry': entry_price, 'sl': float(sl),
                   'tp1': float(tp1), 'tp2': float(tp2), 'tp3': float(tp3),
                   'sl_pct': float(abs(entry_price - sl) / entry_price * 100),
                   'tp1_pct': float(abs(tp1 - entry_price) / entry_price * 100)})
    return result

def print_signal(result):
    print("\n" + "═" * 60)
    print("  JIMI v6.10 — LIVE SIGNAL SCAN")
    print("═" * 60)
    print(f"\n  Time:   {result['timestamp']}")
    print(f"  Price:  ${result['price']:.2f}")
    print(f"  Bias:   {result['swing_bias']}")
    print(f"  Phase0: {result.get('phase0', 'N/A')}")

    # --- Market Data ---
    print(f"\n  Market Data:")
    vwap = result.get('vwap')
    vwap_dist = result.get('vwap_dist_pct')
    taker = result.get('taker_ratio')
    atr = result.get('atr_1h')
    vol_r = result.get('vol_ratio')
    if vwap: print(f"    VWAP:           ${vwap:.2f}  ({vwap_dist:+.2f}% from price)")
    if taker is not None:
        taker_label = "buyers" if taker > 0.52 else "sellers" if taker < 0.48 else "neutral"
        print(f"    Taker Ratio:    {taker:.4f}  ({taker_label}, {taker*100:.1f}% buy)")
    if atr: print(f"    ATR (1H):       ${atr:.2f}  ({atr/result['price']*100:.2f}% of price)")
    if vol_r: print(f"    Vol Ratio:      {vol_r:.2f}x  (24h vs 7d)")

    # --- Module Scores ---
    print(f"\n  Module Scores:")
    print(f"    M1 (1H MACD):  {result['m1']['direction']:>8}  score={result['m1']['score']:.2f}")
    print(f"    M2 (EMA conf): {result['m2']['status']:>8}  score={result['m2']['score']:.2f}")
    if 'm3' in result:
        print(f"    M3 (VWAP):     {result['m3']['status']:>8}  score={result['m3']['score']:.2f}")
    if 'm4' in result:
        m4 = result['m4']
        det = m4.get('details', {})
        if isinstance(det, dict):
            div_str = det.get('layer_a_div', 'NONE')
            zl_str = det.get('layer_b_cross', 'NONE')
            zl_raw = det.get('layer_b_zl_state', '')
            bars = det.get('layer_b_bars_since', '')
            la = det.get('layer_a_score', 0)
            lb = det.get('layer_b_score', 0)
            zl_tag = f"[2H:{zl_raw}]" if zl_raw and zl_raw != 'NONE' else ""
            print(f"    M4 (CVD):        {m4['status']:>8}  score={m4['score']:.2f}  "
                  f"div={div_str}({la:.2f})  zl={zl_str}{zl_tag}({lb:.2f}) {bars}bars")
        else:
            print(f"    M4 (CVD):        {m4['status']:>8}  score={m4['score']:.2f}")
    if 'm5' in result:
        m5 = result['m5']
        print(f"    M5 (LiqtMag):  {m5['status']:>8}  score={m5['score']:.2f}  nearest={m5['details'].get('nearest_magnet')} ({m5['details'].get('magnet_dist_pct')}%)")
    if 'm6' in result:
        m6 = result['m6']
        if m6['status'] != 'SKIP':
            print(f"    M6 (Deriv):    {m6['status']:>8}  score={m6['score']:.2f}")
    if 'cascade' in result and result['cascade'].get('cascade'):
        c = result['cascade']
        print(f"    ⚡ CASCADE:    momentum={c['momentum']}% vol_spike={c['vol_spike']}x range={c['range_expansion']}x")

    # --- Liquidation Magnets ---
    magnets = result.get('magnets', [])
    gaps = result.get('gaps', [])
    if magnets:
        print(f"\n  Liquidation Magnets (volume clusters):")
        price = result['price']
        for i, (p, s) in enumerate(magnets[:5]):
            dist = (p - price) / price * 100
            direction = "↑" if dist > 0 else "↓"
            print(f"    #{i+1}: ${p:.2f}  strength={s:.2f}x  ({direction}{abs(dist):.2f}%)")
    if gaps:
        print(f"  Volume Gaps (low liquidity):")
        for i, p in enumerate(gaps[:3]):
            dist = (p - price) / price * 100
            print(f"    #{i+1}: ${p:.2f}  ({dist:+.2f}%)")

    # --- Support / Resistance Levels ---
    sr = result.get('sr_levels', [])
    if sr:
        supports = [(p, s, t, touches, bounces) for p, s, t, touches, bounces in sr if t == 'SUPPORT']
        resistances = [(p, s, t, touches, bounces) for p, s, t, touches, bounces in sr if t == 'RESISTANCE']
        # Sort each by strength (strongest first)
        supports.sort(key=lambda x: x[1], reverse=True)
        resistances.sort(key=lambda x: x[1], reverse=True)
        if supports:
            print(f"  Support Levels (price rejection):")
            for i, (p, s, _, touches, bounces) in enumerate(supports[:4]):
                dist = (p - price) / price * 100
                print(f"    #{i+1}: ${p:.2f}  strength={s:.1f}  touches={touches} bounces={bounces}  ({dist:+.2f}%)")
        if resistances:
            print(f"  Resistance Levels (price rejection):")
            for i, (p, s, _, touches, bounces) in enumerate(resistances[:4]):
                dist = (p - price) / price * 100
                print(f"    #{i+1}: ${p:.2f}  strength={s:.1f}  touches={touches} bounces={bounces}  ({dist:+.2f}%)")

    # --- Estimated Liquidation Levels ---
    liq = result.get('liquidation_levels', {})
    if liq:
        print(f"\n  Est. Liquidation Levels (isolated margin):")
        print(f"    {'Leverage':<10} {'Long Liq':>12} {'Dist':>8}  {'Short Liq':>12} {'Dist':>8}")
        for lev in ['x5', 'x10', 'x20', 'x50']:
            if lev in liq:
                l = liq[lev]
                print(f"    {lev:<10} ${l['long']:>10.2f}  {l['long_dist_pct']:>6.2f}%  ${l['short']:>10.2f}  {l['short_dist_pct']:>6.2f}%")

    # --- Derivatives Data (M6) ---
    deriv = result.get('derivatives', {})
    if deriv and 'error' not in deriv:
        print(f"\n  Derivatives Data:")
        oi_usd = deriv.get('oi_usd', 0)
        print(f"    OI:             {deriv.get('oi', 0):,.0f} ETH  (${oi_usd/1e9:.2f}B)  1h Δ: {deriv.get('oi_roc_1h', 0):+.3f}%")
        print(f"    L/S Ratio:      {deriv.get('ls_ratio', 0):.4f}  (long {deriv.get('long_pct', 0):.1f}% / short {deriv.get('short_pct', 0):.1f}%)  z={deriv.get('ls_zscore', 0):.2f}")
        pos = deriv.get('positioning', 'NEUTRAL')
        pos_icon = {'CROWDED_LONG': '🔴', 'CROWDED_SHORT': '🟢'}.get(pos, '⚪')
        print(f"    Positioning:    {pos_icon} {pos}")
        print(f"    Top Traders:    L/S={deriv.get('top_ls_ratio', 0):.4f}  whale={deriv.get('whale_signal', 'NEUTRAL')}  gap={deriv.get('whale_retail_gap', 0):+.4f}")
        print(f"    Futures Taker:  {deriv.get('futures_taker_ratio', 0):.4f}  flow={deriv.get('futures_flow', 'NEUTRAL')}")
        fr = deriv.get('funding_rate')
        if fr is not None:
            fr_label = "longs pay" if fr > 0 else "shorts pay"
            print(f"    Funding Rate:   {fr*100:+.4f}%  ({fr_label})")
        oi_div = deriv.get('oi_price_div', 'NONE')
        if oi_div != 'NONE':
            print(f"    ⚡ OI Divergence: {oi_div}")
    if 'm6' in result:
        m6 = result['m6']
        print(f"    M6 (Deriv):    {m6['status']:>8}  score={m6['score']:.2f}")

    # --- ICS & Signal ---
    if 'ics' in result:
        print(f"\n  ICS: {result['ics']:.3f}  (floor={result['effective_floor']:.3f})")

    status = result['status']
    if status == 'SIGNAL':
        print(f"\n  ✅ SIGNAL: {result['direction']}")
        print(f"    Entry: ${result['entry']:.2f}")
        print(f"    SL:    ${result['sl']:.2f}  ({result['sl_pct']:.2f}%)")
        print(f"    TP1:   ${result['tp1']:.2f}  ({result['tp1_pct']:.2f}%)")
        print(f"    TP2:   ${result['tp2']:.2f}")
        print(f"    TP3:   ${result['tp3']:.2f}")
    else:
        print(f"\n  ⛔ {status}: {result.get('reason', 'N/A')}")
    print("═" * 60)


# ═══════════════════════════════════════════════════════════════════════
# DASHBOARD (optional — requires no external deps)
# ═══════════════════════════════════════════════════════════════════════

def run_dashboard(port=8888):
    from http.server import HTTPServer, SimpleHTTPRequestHandler
    import threading

    class DashboardState:
        last_scan = None
        scan_history = []
        running = True

    state = DashboardState()

    def scanner_loop():
        while state.running:
            try:
                df_15m = fetch_recent(bars=1000)
                df_15m, df_1h, df_2h, df_4h, df_1d = compute_indicators(df_15m)
                result = scan_signal(df_15m, df_1h, df_2h, df_4h, df_1d)
                result['scan_time'] = datetime.utcnow().isoformat()
                state.last_scan = result
                state.scan_history.append(result)
                if len(state.scan_history) > 100:
                    state.scan_history = state.scan_history[-100:]
                print(f"[{result['scan_time']}] {result['status']} {result.get('direction','')} ICS={result.get('ics','N/A')}")
            except Exception as e:
                print(f"Scan error: {e}")
            import time; time.sleep(60)

    class Handler(SimpleHTTPRequestHandler):
        def do_GET(self):
            if self.path == '/api/scan':
                self.send_response(200); self.send_header('Content-Type', 'application/json'); self.end_headers()
                self.wfile.write(json.dumps(state.last_scan or {'status': 'INITIALIZING'}, default=str).encode())
            elif self.path == '/api/history':
                self.send_response(200); self.send_header('Content-Type', 'application/json'); self.end_headers()
                self.wfile.write(json.dumps(state.scan_history, default=str).encode())
            elif self.path in ('/', '/index.html'):
                self.send_response(200); self.send_header('Content-Type', 'text/html'); self.end_headers()
                self.wfile.write(DASHBOARD_HTML.encode())
            else:
                self.send_response(404); self.end_headers()
        def log_message(self, *a): pass

    DASHBOARD_HTML = """<!DOCTYPE html>
<html><head><meta charset="utf-8"><title>JIMI v6.10</title>
<meta name="viewport" content="width=device-width,initial-scale=1">
<style>*{margin:0;padding:0;box-sizing:border-box}body{background:#0a0a0f;color:#e0e0e0;font-family:'JetBrains Mono',monospace;padding:20px}
h1{color:#00ff88;font-size:24px;margin-bottom:5px}.subtitle{color:#666;font-size:12px;margin-bottom:20px}
.grid{display:grid;grid-template-columns:1fr 1fr;gap:15px;max-width:1400px}
.card{background:#12121a;border:1px solid #1a1a2e;border-radius:8px;padding:15px}
.card h2{color:#888;font-size:13px;text-transform:uppercase;letter-spacing:1px;margin-bottom:10px}
.signal{font-size:28px;font-weight:bold}.signal.LONG{color:#00ff88}.signal.SHORT{color:#ff4444}
.signal.NO_SIGNAL,.signal.FILTERED{color:#666}.price{font-size:32px;color:#fff;font-weight:bold}
.module{display:flex;justify-content:space-between;padding:6px 0;border-bottom:1px solid #1a1a2e}
.module:last-child{border:none}.mod-name{color:#888}.mod-score{font-weight:bold}
.mod-score.PASS{color:#00ff88}.mod-score.FAIL{color:#ff4444}.mod-score.NEUTRAL{color:#ff8800}.mod-score.SKIP{color:#666}
.ics-bar{height:8px;background:#1a1a2e;border-radius:4px;margin:10px 0;overflow:hidden}
.ics-fill{height:100%;border-radius:4px;transition:width .3s}.ics-fill.good{background:#00ff88}
.ics-fill.mid{background:#ff8800}.ics-fill.low{background:#ff4444}
.levels{display:grid;grid-template-columns:1fr 1fr;gap:8px}
.level{text-align:center;padding:8px;border-radius:4px}.level-label{font-size:11px;color:#888}
.level-value{font-size:16px;font-weight:bold}.level.sl{background:#2a1015}.level.sl .level-value{color:#ff4444}
.level.tp{background:#0a2015}.level.tp .level-value{color:#00ff88}
.history{max-height:300px;overflow-y:auto}.hist-row{display:flex;justify-content:space-between;padding:4px 0;font-size:12px;border-bottom:1px solid #111}
.refresh{color:#444;font-size:11px}
.deriv-row{display:flex;justify-content:space-between;padding:5px 0;border-bottom:1px solid #1a1a2e;font-size:12px}
.deriv-row:last-child{border:none}.deriv-label{color:#888}.deriv-value{font-weight:bold}
.pos-CROWDED_LONG{color:#ff4444}.pos-CROWDED_SHORT{color:#00ff88}.pos-NEUTRAL{color:#888}
.whale-WHALE_BULLISH{color:#00ff88}.whale-WHALE_BEARISH{color:#ff4444}.whale-NEUTRAL{color:#888}
.flow-BUYERS_DOMINANT{color:#00ff88}.flow-SELLERS_DOMINANT{color:#ff4444}.flow-NEUTRAL{color:#888}
.fr-pos{color:#ff4444}.fr-neg{color:#00ff88}</style></head><body>
<h1>⚡ JIMI v6.10</h1><div class="subtitle">ETH/USDT 15m — 6 Module Signal Scanner + Liquidation Magnets + Derivatives</div>
<div class="grid">
<div class="card"><h2>Current Signal</h2><div id="signal" class="signal NO_SIGNAL">SCANNING...</div>
<div id="price" class="price">--</div><div id="meta" style="color:#666;font-size:12px;margin-top:5px"></div></div>
<div class="card"><h2>ICS Score</h2><div id="ics-val" style="font-size:36px;font-weight:bold">--</div>
<div class="ics-bar"><div id="ics-fill" class="ics-fill" style="width:0%"></div></div>
<div id="ics-floor" style="color:#666;font-size:12px"></div></div>
<div class="card"><h2>Module Breakdown</h2><div id="modules"></div></div>
<div class="card"><h2>Derivatives (M6)</h2><div id="derivatives"></div></div>
<div class="card"><h2>Entry Levels</h2><div id="levels" class="levels"></div></div>
<div class="card"><h2>Liquidation Magnets</h2><div id="magnets"></div></div>
<div class="card" style="grid-column:1/-1"><h2>Scan History</h2><div id="history" class="history"></div></div>
</div>
<div class="refresh" style="margin-top:15px">Auto-refresh: 60s | <span id="last-update"></span></div>
<script>
function dr(label,value,cls){return '<div class="deriv-row"><span class="deriv-label">'+label+'</span><span class="deriv-value'+(cls?' '+cls:'')+'">'+value+'</span></div>'}
async function scan(){try{const r=await fetch('/api/scan');const d=await r.json();
document.getElementById('signal').textContent=d.status+(d.direction?': '+d.direction:'');
document.getElementById('signal').className='signal '+(d.direction||d.status);
document.getElementById('price').textContent='$'+(d.price?.toFixed(2)||'--');
document.getElementById('meta').textContent='Bias: '+d.swing_bias+' | Phase0: '+(d.phase0?.toFixed(3)||'N/A');
if(d.ics!==undefined){document.getElementById('ics-val').textContent=d.ics.toFixed(3);
const pct=Math.min(d.ics*100,100);const fill=document.getElementById('ics-fill');
fill.style.width=pct+'%';fill.className='ics-fill '+(pct>60?'good':pct>48?'mid':'low');
document.getElementById('ics-floor').textContent='Floor: '+(d.effective_floor?.toFixed(3)||'--')+' | Threshold: '+(d.phase0>=0.4?'0.52':'0.48')}
let mods='';[['M1','m1'],['M2','m2'],['M3','m3'],['M4','m4'],['M5','m5'],['M6','m6']].forEach(([n,k])=>{const m=d[k];if(!m||m.status==='SKIP')return;
const s=m.status||m.direction||'--';const sc=m.score?.toFixed(2)||'--';
mods+='<div class="module"><span class="mod-name">'+n+'</span><span class="mod-score '+s+'">'+s+' ('+sc+')</span></div>'});
document.getElementById('modules').innerHTML=mods;
const v=d.derivatives;if(v&&!v.error){let h='';
h+=dr('OI',v.oi?.toLocaleString()+' ETH ($'+(v.oi_usd/1e9)?.toFixed(2)+'B)  1h: '+(v.oi_roc_1h>0?'+':'')+v.oi_roc_1h?.toFixed(3)+'%');
h+=dr('L/S Ratio',v.ls_ratio?.toFixed(4)+'  long '+v.long_pct?.toFixed(1)+'% / short '+v.short_pct?.toFixed(1)+'%  z='+v.ls_zscore?.toFixed(2));
h+=dr('Positioning',v.positioning,'pos-'+v.positioning);
h+=dr('Top Traders','L/S='+v.top_ls_ratio?.toFixed(4)+'  '+v.whale_signal,'whale-'+v.whale_signal);
h+=dr('Futures Taker',v.futures_taker_ratio?.toFixed(4)+'  '+v.futures_flow,'flow-'+v.futures_flow);
if(v.funding_rate!==null&&v.funding_rate!==undefined){const fr=v.funding_rate;const cls=fr>0?'fr-pos':'fr-neg';
h+=dr('Funding',(fr*100).toFixed(4)+'%  '+(fr>0?'longs pay':'shorts pay'),cls)}
if(v.oi_price_div&&v.oi_price_div!=='NONE')h+=dr('⚡ OI Diverg.',v.oi_price_div,'pos-'+v.oi_price_div);
document.getElementById('derivatives').innerHTML=h}else{document.getElementById('derivatives').innerHTML='<div style="color:#666">No derivatives data</div>'}
let mg='';const magnets=d.magnets||[];magnets.slice(0,5).forEach(([p,s],i)=>{
const dist=((p-d.price)/d.price*100);const dir=dist>0?'↑':'↓';
mg+='<div class="deriv-row"><span class="deriv-label">#'+(i+1)+' $'+p.toFixed(2)+'</span><span class="deriv-value">'+s.toFixed(2)+'x  '+dir+Math.abs(dist).toFixed(2)+'%</span></div>'});
const gaps=d.gaps||[];if(gaps.length){mg+='<div style="color:#666;font-size:11px;margin-top:6px">Gaps: '+gaps.slice(0,3).map(p=>'$'+p.toFixed(2)).join(', ')+'</div>'}
document.getElementById('magnets').innerHTML=mg||'<div style="color:#666">No magnets</div>';
if(d.entry){let l='';[['SL','sl'],['TP1','tp1'],['TP2','tp2'],['TP3','tp3']].forEach(([lb,k])=>{
const c=k==='sl'?'sl':'tp';l+='<div class="level '+c+'"><div class="level-label">'+lb+'</div><div class="level-value">$'+d[k]?.toFixed(2)+'</div></div>'});
document.getElementById('levels').innerHTML=l}else{document.getElementById('levels').innerHTML='<div style="color:#666">No entry levels</div>'}
document.getElementById('last-update').textContent=d.scan_time||new Date().toISOString()}catch(e){console.error(e)}}
async function history(){try{const r=await fetch('/api/history');const d=await r.json();let h='';
d.reverse().forEach(s=>{const c=s.status==='SIGNAL'?'signal':'nosignal';
h+='<div class="hist-row '+c+'"><span>'+s.scan_time+'</span><span>'+s.status+' '+(s.direction||'')+'</span><span>ICS:'+(s.ics?.toFixed(3)||'--')+'</span><span>$'+s.price?.toFixed(2)+'</span></div>'});
document.getElementById('history').innerHTML=h}catch(e){}}
scan();history();setInterval(scan,60000);setInterval(history,60000);
</script></body></html>"""

    t = threading.Thread(target=scanner_loop, daemon=True); t.start()
    print(f"JIMI v6.10 Dashboard running on http://localhost:{port}")
    HTTPServer(('0.0.0.0', port), Handler).serve_forever()


# ═══════════════════════════════════════════════════════════════════════
# CLI
# ═══════════════════════════════════════════════════════════════════════

def main():
    if len(sys.argv) < 2:
        print("JIMI v6.10 — Complete Trading Framework")
        print()
        print("Usage:")
        print("  python3 jimi_v610_complete.py backtest <csv> [--verbose] [--start=YYYY-MM-DD] [--end=YYYY-MM-DD]")
        print("  python3 jimi_v610_complete.py scan [--json]")
        print("  python3 jimi_v610_complete.py dashboard [port]")
        print()
        print("Examples:")
        print("  python3 jimi_v610_complete.py backtest eth_15m_data.csv --verbose")
        print("  python3 jimi_v610_complete.py scan --json")
        print("  python3 jimi_v610_complete.py dashboard 8888")
        sys.exit(0)

    cmd = sys.argv[1].lower()

    if cmd == 'backtest':
        csv_path = sys.argv[2] if len(sys.argv) > 2 and not sys.argv[2].startswith('-') else 'eth_15m_data.csv'
        verbose = '--verbose' in sys.argv or '-v' in sys.argv
        date_start = next((a.split('=',1)[1] for a in sys.argv if a.startswith('--start=')), None)
        date_end = next((a.split('=',1)[1] for a in sys.argv if a.startswith('--end=')), None)

        if not os.path.exists(csv_path):
            print(f"ERROR: File not found: {csv_path}"); sys.exit(1)

        trades, stats, df = run_backtest(csv_path, verbose=verbose, date_start=date_start, date_end=date_end)
        print_report(trades, stats)
        if trades:
            export_trades(trades, 'jimi_v610_trades.csv')
        print("\nDone.")

    elif cmd == 'scan':
        print("Fetching recent data...")
        df_15m = fetch_recent(bars=1000)
        print("Computing indicators...")
        df_15m, df_1h, df_2h, df_4h, df_1d = compute_indicators(df_15m)
        print("Scanning...")
        result = scan_signal(df_15m, df_1h, df_2h, df_4h, df_1d)
        print_signal(result)
        if '--json' in sys.argv:
            print(json.dumps(result, indent=2, default=str))

    elif cmd == 'dashboard':
        port = int(sys.argv[2]) if len(sys.argv) > 2 else 8888
        run_dashboard(port)

    else:
        print(f"Unknown command: {cmd}. Use 'backtest', 'scan', or 'dashboard'.")
        sys.exit(1)

if __name__ == '__main__':
    main()

"""M7: Market Regime — ETH/BTC + BTC Volatility + Volume."""

import numpy as np
import pandas as pd
import os
import json
import time
import ccxt

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

    ethbtc = m7_fetch_daily("ETH/BTC", since_ms, until_ms)
    if len(ethbtc) > 0:
        ethbtc["ema21"] = ethbtc["close"].ewm(span=21, adjust=False).mean()
        ethbtc["ema55"] = ethbtc["close"].ewm(span=55, adjust=False).mean()
        ethbtc["trend"] = "NEUTRAL"
        ethbtc.loc[ethbtc["ema21"] > ethbtc["ema55"], "trend"] = "BULL"
        ethbtc.loc[ethbtc["ema21"] < ethbtc["ema55"], "trend"] = "BEAR"
        ethbtc["ema_dist"] = (ethbtc["close"] - ethbtc["ema55"]) / ethbtc["ema55"]
        ethbtc["zscore_90"] = (
            (ethbtc["close"] - ethbtc["close"].rolling(90).mean())
            / ethbtc["close"].rolling(90).std().replace(0, np.nan)
        )
        ethbtc["roc_7d"] = ethbtc["close"].pct_change(7)
        ethbtc["roc_30d"] = ethbtc["close"].pct_change(30)
        ethbtc = ethbtc[(ethbtc["date"] >= start) & (ethbtc["date"] <= end)].reset_index(drop=True)

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
    """Score M7: Market Regime."""
    details = {}

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

    mom_s = 0.5
    if ethbtc_row:
        r7 = ethbtc_row.get("roc_7d", np.nan)
        r30 = ethbtc_row.get("roc_30d", np.nan)
        if not np.isnan(r7) and not np.isnan(r30):
            if direction == "LONG":
                if r7 > 0.03 and r30 > 0.05:
                    mom_s = 0.85
                elif r7 > 0.01:
                    mom_s = 0.68
                elif r7 < -0.03 and r30 < -0.05:
                    mom_s = 0.20
                elif r7 < -0.01:
                    mom_s = 0.35
            else:
                if r7 < -0.03 and r30 < -0.05:
                    mom_s = 0.85
                elif r7 < -0.01:
                    mom_s = 0.68
                elif r7 > 0.03 and r30 > 0.05:
                    mom_s = 0.20
                elif r7 > 0.01:
                    mom_s = 0.35

    vol_reg_s = 0.5
    if btc_row:
        pctl = btc_row.get("atr_pctl", np.nan)
        btc_trend = btc_row.get("trend", "NEUTRAL")
        if not np.isnan(pctl):
            if direction == "LONG":
                if pctl > 0.85:
                    vol_reg_s = 0.25
                elif pctl > 0.70:
                    vol_reg_s = 0.35
                elif pctl < 0.30:
                    vol_reg_s = 0.70
                elif pctl < 0.50:
                    vol_reg_s = 0.60
                if btc_trend == "BEAR":
                    vol_reg_s *= 0.8
                elif btc_trend == "BULL":
                    vol_reg_s = min(vol_reg_s * 1.15, 1.0)
            else:
                if pctl > 0.85:
                    vol_reg_s = 0.75
                elif pctl > 0.70:
                    vol_reg_s = 0.65
                elif pctl < 0.30:
                    vol_reg_s = 0.35
                elif pctl < 0.50:
                    vol_reg_s = 0.40
                if btc_trend == "BULL":
                    vol_reg_s *= 0.8
                elif btc_trend == "BEAR":
                    vol_reg_s = min(vol_reg_s * 1.15, 1.0)
        details["btc_trend"] = btc_trend
        details["btc_atr_pctl"] = round(pctl, 3) if not np.isnan(pctl) else None

    vr_s = 0.5
    if not np.isnan(vol_ratio):
        if vol_ratio > 1.3:
            vr_s = 0.75
        elif vol_ratio > 1.0:
            vr_s = 0.60
        elif vol_ratio < 0.5:
            vr_s = 0.25
        elif vol_ratio < 0.7:
            vr_s = 0.38

    cross_s = 0.5
    if ethbtc_row and btc_row:
        er = ethbtc_row.get("roc_7d", np.nan)
        br = btc_row.get("roc_7d", np.nan)
        if not np.isnan(er) and not np.isnan(br):
            if direction == "LONG":
                if er > 0 and br > 0:
                    cross_s = 0.80
                elif er > 0 and br < -0.02:
                    cross_s = 0.55
                elif er < -0.02 and br > 0:
                    cross_s = 0.25
                elif er < -0.02 and br < -0.02:
                    cross_s = 0.30
            else:
                if er < 0 and br < 0:
                    cross_s = 0.80
                elif er < 0 and br > 0.02:
                    cross_s = 0.55
                elif er > 0.02 and br < 0:
                    cross_s = 0.25
                elif er > 0.02 and br > 0.02:
                    cross_s = 0.30

    composite = (trend_s * 0.30 + mom_s * 0.20 + vol_reg_s * 0.20 + vr_s * 0.15 + cross_s * 0.15)
    composite = max(0.0, min(1.0, composite))
    details["m7_score"] = round(composite, 3)
    status = "PASS" if composite >= 0.50 else "FAIL"
    return status, composite, details

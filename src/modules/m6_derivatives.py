"""M6: Derivatives Data — OI, L/S Ratio, Taker Flow, Funding."""

import numpy as np
import pandas as pd
import requests

BASE_URL = "https://fapi.binance.com"
PERIOD = "15m"
HISTORY_BARS = 288  # 3 days of 15m data (sweet spot: filters noise, doesn't absorb extremes)


def fetch_oi_history(symbol="ETHUSDT", period=PERIOD, limit=HISTORY_BARS):
    r = requests.get(f"{BASE_URL}/futures/data/openInterestHist",
                     params={"symbol": symbol, "period": period, "limit": limit})
    r.raise_for_status()
    rows = []
    for d in r.json():
        rows.append({
            "timestamp": pd.to_datetime(d["timestamp"], unit="ms"),
            "oi": float(d["sumOpenInterest"]),
            "oi_usd": float(d["sumOpenInterestValue"]),
        })
    return pd.DataFrame(rows)


def fetch_ls_ratio(symbol="ETHUSDT", period=PERIOD, limit=HISTORY_BARS):
    r = requests.get(f"{BASE_URL}/futures/data/globalLongShortAccountRatio",
                     params={"symbol": symbol, "period": period, "limit": limit})
    r.raise_for_status()
    rows = []
    for d in r.json():
        rows.append({
            "timestamp": pd.to_datetime(d["timestamp"], unit="ms"),
            "ls_ratio": float(d["longShortRatio"]),
            "long_pct": float(d["longAccount"]),
            "short_pct": float(d["shortAccount"]),
        })
    return pd.DataFrame(rows)


def fetch_top_trader_ls(symbol="ETHUSDT", period=PERIOD, limit=HISTORY_BARS):
    r = requests.get(f"{BASE_URL}/futures/data/topLongShortAccountRatio",
                     params={"symbol": symbol, "period": period, "limit": limit})
    r.raise_for_status()
    rows = []
    for d in r.json():
        rows.append({
            "timestamp": pd.to_datetime(d["timestamp"], unit="ms"),
            "top_ls_ratio": float(d["longShortRatio"]),
            "top_long_pct": float(d["longAccount"]),
            "top_short_pct": float(d["shortAccount"]),
        })
    return pd.DataFrame(rows)


def fetch_taker_ratio(symbol="ETHUSDT", period=PERIOD, limit=HISTORY_BARS):
    r = requests.get(f"{BASE_URL}/futures/data/takerlongshortRatio",
                     params={"symbol": symbol, "period": period, "limit": limit})
    r.raise_for_status()
    rows = []
    for d in r.json():
        rows.append({
            "timestamp": pd.to_datetime(d["timestamp"], unit="ms"),
            "futures_taker_ratio": float(d["buySellRatio"]),
            "futures_buy_vol": float(d["buyVol"]),
            "futures_sell_vol": float(d["sellVol"]),
        })
    return pd.DataFrame(rows)


def fetch_funding_rate(symbol="ETHUSDT", limit=10):
    r = requests.get(f"{BASE_URL}/fapi/v1/fundingRate",
                     params={"symbol": symbol, "limit": limit})
    r.raise_for_status()
    rows = []
    for d in r.json():
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

    df = oi.merge(ls, on="timestamp", how="outer")
    df = df.merge(top, on="timestamp", how="outer")
    df = df.merge(taker, on="timestamp", how="outer")
    df = df.sort_values("timestamp").reset_index(drop=True)
    df = df.ffill()
    return df, funding


def compute_oi_signals(df_deriv, df_15m=None):
    """OI-based signals: divergence, spike, unwind."""
    df = df_deriv.copy()
    df["oi_roc_1h"] = df["oi"].pct_change(4)
    df["oi_roc_2h"] = df["oi"].pct_change(8)
    df["oi_spike"] = df["oi_roc_1h"].abs() > 0.02

    if df_15m is not None:
        price_col = df_15m.set_index("Open time")["Close"]
        df = df.set_index("timestamp")
        df["price"] = price_col.reindex(df.index, method="ffill")
        df = df.reset_index()
        df["price_roc_1h"] = df["price"].pct_change(4)
        df["price_roc_2h"] = df["price"].pct_change(8)
        df["oi_price_div"] = "NONE"
        mask_bear = (df["price_roc_1h"] > 0.005) & (df["oi_roc_1h"] < -0.01)
        mask_bull = (df["price_roc_1h"] < -0.005) & (df["oi_roc_1h"] > 0.01)
        df.loc[mask_bear, "oi_price_div"] = "BEARISH"
        df.loc[mask_bull, "oi_price_div"] = "BULLISH"

    return df


def compute_positioning_signals(df_deriv):
    """Positioning-based signals: L/S ratio, whale divergence, taker flow."""
    df = df_deriv.copy()
    # Use 3-day window (288 bars at 15m) for z-scores.
    # Long enough to filter 15m noise, short enough to not absorb extremes.
    # Fall back to expanding min_periods=24 when data is shorter.
    window = min(288, len(df))
    min_periods = min(24, len(df))
    ls_mean = df["ls_ratio"].rolling(window, min_periods=min_periods).mean()
    ls_std = df["ls_ratio"].rolling(window, min_periods=min_periods).std()
    df["ls_zscore"] = (df["ls_ratio"] - ls_mean) / ls_std.replace(0, np.nan)

    # Positioning label combines THREE signals:
    # 1. Z-score (relative): is L/S unusual vs 7-day baseline?
    # 2. Absolute band: is L/S objectively extreme regardless of baseline?
    # 3. Rate-of-change: is L/S moving too fast? (catches velocity, not position)
    #
    # The z-score alone has blind spots:
    #   - After sustained extremes, rolling mean adapts → signal fades
    #   - Can't detect fast moves that stay within the normal distribution
    # Absolute bands catch #1 but fire during normal basing.
    # Rate-of-change catches #2 — a 0.15+ move in 4 bars (1h) is abnormal.
    #
    # Absolute bands (high — only truly extreme, sustained crowding):
    #   L/S ≥ 3.0  (75%+ long)  = objectively extreme long-crowding
    #   L/S ≤ 0.33 (75%+ short) = objectively extreme short-crowding
    # Rate-of-change (4-bar delta = 1 hour on 15m):
    #   Δ ≥ +0.15  = rapid long accumulation (potential squeeze loading)
    #   Δ ≤ -0.15  = rapid long liquidation (potential cascade)
    df["positioning"] = "NEUTRAL"
    # Z-score triggers (relative to 7-day baseline)
    df.loc[(df["ls_zscore"] > 1.5) & (df["ls_ratio"] > 1.5), "positioning"] = "CROWDED_LONG"
    df.loc[(df["ls_zscore"] < -1.5) & (df["ls_ratio"] < 0.67), "positioning"] = "CROWDED_SHORT"
    # Absolute triggers (override z-score decay — these levels are always crowded)
    df.loc[df["ls_ratio"] >= 3.0, "positioning"] = "CROWDED_LONG"
    df.loc[df["ls_ratio"] <= 0.33, "positioning"] = "CROWDED_SHORT"
    # Rate-of-change triggers (velocity — catches fast moves z-score misses)
    ls_delta_1h = df["ls_ratio"].diff(4)  # 4 bars × 15m = 1 hour
    df.loc[ls_delta_1h >= 0.15, "positioning"] = "CROWDED_LONG"
    df.loc[ls_delta_1h <= -0.15, "positioning"] = "CROWDED_SHORT"

    if "top_ls_ratio" in df.columns:
        df["whale_retail_gap"] = df["top_ls_ratio"] - df["ls_ratio"]
        df["whale_signal"] = "NEUTRAL"
        df.loc[df["whale_retail_gap"] > 0.3, "whale_signal"] = "WHALE_BULLISH"
        df.loc[df["whale_retail_gap"] < -0.3, "whale_signal"] = "WHALE_BEARISH"

    if "futures_taker_ratio" in df.columns:
        taker_ma = df["futures_taker_ratio"].rolling(8).mean()
        df["futures_taker_ma"] = taker_ma
        df["futures_flow"] = "NEUTRAL"
        df.loc[taker_ma > 1.15, "futures_flow"] = "BUYERS_DOMINANT"
        df.loc[taker_ma < 0.85, "futures_flow"] = "SELLERS_DOMINANT"

    return df


def score_derivatives(df_deriv_latest, direction):
    """Score derivatives data for a given trade direction."""
    if df_deriv_latest is None:
        return "SKIP", 0.5, {}

    if isinstance(df_deriv_latest, dict):
        last = df_deriv_latest
    elif hasattr(df_deriv_latest, "iloc"):
        last = df_deriv_latest.iloc[-1] if len(df_deriv_latest) > 0 else df_deriv_latest
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
    score = 0.5

    oi_div = _get("oi_price_div", "NONE")
    if direction == "LONG":
        if oi_div == "BULLISH":
            score += 0.10
        elif oi_div == "BEARISH":
            score -= 0.05
    elif direction == "SHORT":
        if oi_div == "BEARISH":
            score += 0.10
        elif oi_div == "BULLISH":
            score -= 0.05
    details["oi_div"] = oi_div

    positioning = _get("positioning", "NEUTRAL")
    if direction == "LONG" and positioning == "CROWDED_SHORT":
        score += 0.12
    elif direction == "SHORT" and positioning == "CROWDED_LONG":
        score += 0.12
    elif direction == "LONG" and positioning == "CROWDED_LONG":
        score -= 0.08
    elif direction == "SHORT" and positioning == "CROWDED_SHORT":
        score -= 0.08
    details["positioning"] = positioning
    details["ls_zscore"] = round(float(_get("ls_zscore", 0)), 3)

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


def get_derivatives_summary(symbol="ETHUSDT"):
    """Fetch and summarize current derivatives state for scan output."""
    try:
        df_deriv, funding = fetch_all_derivatives(symbol)
        df_deriv = compute_oi_signals(df_deriv)
        df_deriv = compute_positioning_signals(df_deriv)

        last = df_deriv.iloc[-1]
        latest_funding = funding.iloc[-1] if not funding.empty else None

        return {
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
    except Exception as e:
        return {"error": str(e)}

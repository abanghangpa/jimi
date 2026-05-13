#!/usr/bin/env python3
"""
Liquidity Staircase - 1m Live Monitor
Fast check: price, volume, OI, taker, funding vs key levels.
"""

import json, sys, os
from datetime import datetime, timezone

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
import ccxt, numpy as np

# ── Key Levels ──
LEVELS_BELOW = [
    (2269.00, "Stop cluster 1"),
    (2259.00, "Stop cluster 2"),
    (2254.69, "Liquidation"),
    (2245.39, "🎯 MAIN TARGET"),
]
LEVELS_ABOVE = [
    (2306.04, "Squeeze trigger"),
    (2328.90, "Short stop 1"),
    (2330.45, "Short liq"),
]

STATE_FILE = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "liq_watch_1m.json")


def load_state():
    try:
        return json.load(open(STATE_FILE))
    except Exception:
        return {"hit": [], "prev_oi": None, "prev_vol": None, "prev_price": None}


def save_state(s):
    os.makedirs(os.path.dirname(STATE_FILE), exist_ok=True)
    json.dump(s, open(STATE_FILE, "w"), indent=2)


def fetch_all():
    ex = ccxt.binance({"enableRateLimit": True, "options": {"defaultType": "future"}})
    # Ticker
    tk = ex.fetch_ticker("ETH/USDT")
    price = float(tk["last"])
    high24 = float(tk.get("high", 0))
    low24 = float(tk.get("low", 0))
    vol24 = float(tk.get("quoteVolume", 0))

    # OI (open interest)
    oi = None
    try:
        oi_data = ex.fetch_open_interest("ETH/USDT")
        oi = float(oi_data["openInterestAmount"])
    except Exception:
        pass

    # Funding rate
    fr = None
    try:
        fr_data = ex.fetch_funding_rate("ETH/USDT")
        fr = float(fr_data["fundingRate"])
    except Exception:
        pass

    # Long/Short ratio (via REST since ccxt doesn't wrap it cleanly)
    ls = None
    try:
        import requests
        r = requests.get("https://fapi.binance.com/futures/data/globalLongShortAccountRatio",
                         params={"symbol": "ETHUSDT", "period": "5m", "limit": 1}, timeout=5)
        if r.ok:
            ls = float(r.json()[0]["longShortRatio"])
    except Exception:
        pass

    # Last few klines for volume/taker check (5m, 3 bars)
    vol_5m = None
    taker_ratio = None
    try:
        klines = ex.fetch_ohlcv("ETH/USDT", "5m", limit=3)
        if klines:
            last = klines[-1]
            vol_5m = float(last[5])
            # taker buy volume not in standard OHLCV, estimate from previous
            # Use 5m vol vs 20-period average as proxy
            if len(klines) >= 3:
                avg_vol = sum(float(k[5]) for k in klines) / len(klines)
                vol_spike = vol_5m / avg_vol if avg_vol > 0 else 1.0
                taker_ratio = vol_spike  # vol spike as proxy for aggression
    except Exception:
        pass

    # Taker buy ratio from futures (more accurate)
    try:
        r2 = requests.get("https://fapi.binance.com/futures/data/takerlongshortRatio",
                          params={"symbol": "ETHUSDT", "period": "5m", "limit": 1}, timeout=5)
        if r2.ok:
            d2 = r2.json()[0]
            taker_ratio = float(d2["buySellRatio"])
    except Exception:
        pass

    return {
        "price": price, "high24": high24, "low24": low24, "vol24": vol24,
        "oi": oi, "fr": fr, "ls": ls, "vol_5m": vol_5m, "taker": taker_ratio,
    }


def main():
    state = load_state()
    d = fetch_all()
    price = d["price"]
    alerts = []
    info = []

    # ── Level checks ──
    for lp, label in LEVELS_BELOW + LEVELS_ABOVE:
        key = f"{lp:.2f}"
        if key in state.get("hit", []):
            continue
        dist = (price - lp) / lp * 100
        swept = False
        if lp < price and d["low24"] <= lp:
            swept = True
        elif lp > price and d["high24"] >= lp:
            swept = True

        if swept:
            state.setdefault("hit", []).append(key)
            alerts.append(f"💥 {label} ${lp:.0f} SWEPT ({dist:+.1f}%)")
        elif abs(dist) < 0.5:
            alerts.append(f"⚠️ {label} ${lp:.0f} ({dist:+.1f}%)")

    # ── OI change ──
    oi_str = ""
    if d["oi"] and state.get("prev_oi"):
        oi_chg = (d["oi"] - state["prev_oi"]) / state["prev_oi"] * 100
        oi_str = f"OI={d['oi']/1e6:.2f}M ({oi_chg:+.3f}%)"
        if abs(oi_chg) > 0.1:
            direction = "positions opening" if oi_chg > 0 else "positions closing"
            alerts.append(f"📊 OI {oi_chg:+.3f}% — {direction}")
    elif d["oi"]:
        oi_str = f"OI={d['oi']/1e6:.2f}M"
    state["prev_oi"] = d["oi"]

    # ── Price change ──
    price_str = f"${price:.2f}"
    if state.get("prev_price"):
        p_chg = (price - state["prev_price"]) / state["prev_price"] * 100
        if abs(p_chg) > 0.15:
            alerts.append(f"🚨 Price {p_chg:+.2f}% in 1min")
    state["prev_price"] = price

    # ── Volume spike ──
    vol_str = ""
    if d["vol_5m"]:
        vol_str = f"5mVol={d['vol_5m']/1e3:.0f}K"
        if state.get("prev_vol") and d["vol_5m"] > state["prev_vol"] * 2:
            alerts.append(f"🔊 Volume spike {d['vol_5m']/state['prev_vol']:.1f}x")
        state["prev_vol"] = d["vol_5m"]

    # ── Taker ratio ──
    taker_str = ""
    if d["taker"] is not None:
        taker_str = f"Taker={d['taker']:.3f}"
        if d["taker"] > 1.3:
            alerts.append(f"🟢 Aggressive buyers ({d['taker']:.2f}x)")
        elif d["taker"] < 0.7:
            alerts.append(f"🔴 Aggressive sellers ({d['taker']:.2f}x)")

    # ── L/S ratio ──
    ls_str = ""
    if d["ls"]:
        ls_str = f"L/S={d['ls']:.2f}"

    # ── Funding ──
    fr_str = ""
    if d["fr"] is not None:
        fr_str = f"FR={d['fr']*100:+.4f}%"

    # ── Staircase progress ──
    staircase = [2269.00, 2259.00, 2254.69, 2245.39]
    taken = [p for p in staircase if f"{p:.2f}" in state.get("hit", [])]
    stair_str = f"stair:{len(taken)}/{len(staircase)}"

    # ── Compose output ──
    parts = [price_str, oi_str, vol_str, taker_str, ls_str, fr_str, stair_str]
    summary = " | ".join(p for p in parts if p)

    save_state(state)

    if alerts:
        print(f"ETH {summary}")
        for a in alerts:
            print(a)
    else:
        print(f"ETH {summary}")


if __name__ == "__main__":
    main()

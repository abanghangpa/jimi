#!/usr/bin/env python3
"""
Liquidity Staircase Watcher
Tracks price vs key liquidity levels and reports when levels are approached/hit.
"""

import json
import sys
import os
from datetime import datetime, timezone

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import ccxt

# Key levels from scan_20260513_205026
LEVELS = [
    {"price": 2283.93, "type": "BID_WALL",   "label": "Bid wall",        "status": "intact"},
    {"price": 2269.00, "type": "LONG_STOP",   "label": "Stop cluster 1",  "status": "intact"},
    {"price": 2259.00, "type": "LONG_STOP",   "label": "Stop cluster 2",  "status": "intact"},
    {"price": 2254.69, "type": "LONG_LIQ",    "label": "Liquidation",     "status": "intact"},
    {"price": 2245.39, "type": "LONG_STOP",   "label": "🎯 Main target",  "status": "intact"},
    # Above
    {"price": 2306.04, "type": "SQUEEZE_TRIG","label": "Squeeze trigger", "status": "intact"},
    {"price": 2328.90, "type": "SHORT_STOP",  "label": "Short stop 1",    "status": "intact"},
    {"price": 2330.45, "type": "SHORT_LIQ",   "label": "Short liq",       "status": "intact"},
]

STATE_FILE = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "liquidity_watch_state.json")


def load_state():
    try:
        with open(STATE_FILE) as f:
            return json.load(f)
    except Exception:
        return {"hit": [], "last_price": None, "checks": 0}


def save_state(state):
    os.makedirs(os.path.dirname(STATE_FILE), exist_ok=True)
    with open(STATE_FILE, "w") as f:
        json.dump(state, f, indent=2)


def fetch_price():
    ex = ccxt.binance({"enableRateLimit": True})
    ticker = ex.fetch_ticker("ETH/USDT")
    return {
        "last": float(ticker["last"]),
        "high": float(ticker.get("high", 0)),
        "low": float(ticker.get("low", 0)),
        "bid": float(ticker.get("bid", 0)),
        "ask": float(ticker.get("ask", 0)),
        "vol": float(ticker.get("quoteVolume", 0)),
    }


def check_levels(price_data, state):
    price = price_data["last"]
    high = price_data["high"]
    low = price_data["low"]
    hit_now = []
    alerts = []

    for lvl in LEVELS:
        key = f"{lvl['price']:.2f}"
        already_hit = key in state.get("hit", [])
        lp = lvl["price"]

        # Check if level was swept
        swept = False
        if lvl["type"] in ("LONG_STOP", "LONG_LIQ", "BID_WALL"):
            # Below levels: swept if low went below
            swept = low <= lp
        elif lvl["type"] in ("SHORT_STOP", "SHORT_LIQ"):
            # Above levels: swept if high went above
            swept = high >= lp
        elif lvl["type"] == "SQUEEZE_TRIG":
            swept = high >= lp

        # Check proximity (within 0.3%)
        dist_pct = (price - lp) / lp * 100
        near = abs(dist_pct) < 0.3

        if swept and not already_hit:
            hit_now.append(key)
            direction = "↓" if dist_pct < 0 else "↑"
            alerts.append(f"💥 {lvl['label']} HIT @ ${lp:.2f} ({direction}{abs(dist_pct):.2f}%)")
        elif near and not already_hit:
            direction = "↓" if dist_pct < 0 else "↑"
            alerts.append(f"⚠️ {lvl['label']} NEAR @ ${lp:.2f} ({direction}{abs(dist_pct):.2f}%)")

    return hit_now, alerts


def main():
    state = load_state()
    price_data = fetch_price()
    price = price_data["last"]

    hit_now, alerts = check_levels(price_data, state)

    # Update state
    state["hit"].extend(hit_now)
    state["last_price"] = price
    state["checks"] = state.get("checks", 0) + 1
    state["last_check"] = datetime.now(timezone.utc).isoformat()
    save_state(state)

    # Build report
    lines = []
    lines.append(f"ETH ${price:.2f}  (H: ${price_data['high']:.2f}  L: ${price_data['low']:.2f})")

    if alerts:
        for a in alerts:
            lines.append(a)
    else:
        # Show proximity summary
        below = [(l, (price - l["price"]) / l["price"] * 100) for l in LEVELS if l["price"] < price and f"{l['price']:.2f}" not in state.get("hit", [])]
        above = [(l, (l["price"] - price) / price * 100) for l in LEVELS if l["price"] > price and f"{l['price']:.2f}" not in state.get("hit", [])]
        if below:
            nearest = min(below, key=lambda x: x[1])
            lines.append(f"Nearest below: {nearest[0]['label']} ${nearest[0]['price']:.2f} ({nearest[1]:.2f}% away)")
        if above:
            nearest = min(above, key=lambda x: x[1])
            lines.append(f"Nearest above: {nearest[0]['label']} ${nearest[0]['price']:.2f} ({nearest[1]:.2f}% away)")

    # Staircase progress
    staircase_prices = [2269.00, 2259.00, 2254.69, 2245.39]
    taken = [p for p in staircase_prices if f"{p:.2f}" in state.get("hit", [])]
    if taken:
        lines.append(f"Staircase: {len(taken)}/{len(staircase_prices)} levels taken → {' → '.join(f'${p:.0f}' for p in staircase_prices)}")

    print("\n".join(lines))

    # Return alerts for cron delivery
    if alerts:
        return "\n".join(lines)
    return None


if __name__ == "__main__":
    result = main()
    if result:
        print("\n" + result)

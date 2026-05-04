#!/usr/bin/env python3
"""Backtest M18 v2 squeeze detector."""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import numpy as np
import pandas as pd
from src.config import CONFIG
from src.utils.data_handler import load_data
from src.utils.indicators import calc_atr, calc_rsi, calc_vol_ratio
from src.modules.m4_cvd import calc_cvd_15m, detect_cvd_divergence_15m
from src.modules.m18_squeeze import detect_squeeze_v2, SQUEEZE_V2_DEFAULTS

CSV = os.path.join(os.path.dirname(os.path.dirname(__file__)), "eth_15m_merged.csv")
df = load_data(CSV)
df = df[df["Open time"] >= "2026-01-01"].reset_index(drop=True)

cfg = dict(CONFIG)
cfg.update(SQUEEZE_V2_DEFAULTS)

df["atr"] = calc_atr(df["High"], df["Low"], df["Close"], cfg["ATR_PERIOD"])
df["rsi"] = calc_rsi(df["Close"], 14)
df["vol_ratio"] = calc_vol_ratio(df["Volume"])
df["taker_ratio"] = (df["Taker buy base asset volume"] / df["Volume"].replace(0, np.nan)).fillna(0.5)
df["vol_ma20"] = df["Volume"].rolling(20).mean()
df["vol_trend"] = df["Volume"] / df["vol_ma20"]
df["cvd_15m"] = calc_cvd_15m(df)
df["cvd_divergence_15m"] = detect_cvd_divergence_15m(df, cfg["CVD_LOOKBACK"], cfg["CVD_DIVERGENCE_WINDOW"])

taker_ma = df["taker_ratio"].rolling(50).mean()
taker_std = df["taker_ratio"].rolling(50).std()
df["ls_zscore"] = (df["taker_ratio"] - taker_ma) / taker_std.replace(0, 1)
df["funding_sim"] = (df["rsi"] - 50) / 50 * 0.001
df["oi_roc_sim"] = df["Volume"].pct_change(4) * 100

vol_pctl = df["Volume"].rolling(100).apply(
    lambda x: (x.iloc[-1] - x.min()) / (x.max() - x.min()) if x.max() > x.min() else 0.5, raw=False)
price_change = df["Close"].pct_change(4)
df["whale"] = "NEUTRAL"
df.loc[(vol_pctl > 0.8) & (price_change > 0.005), "whale"] = "WHALE_BULLISH"
df.loc[(vol_pctl > 0.8) & (price_change < -0.005), "whale"] = "WHALE_BEARISH"

atr_pctl = df["atr"].rolling(500, min_periods=100).apply(
    lambda x: (x.iloc[-1] - x.min()) / (x.max() - x.min()) if x.max() > x.min() else 0.5, raw=False)

MIN_BARS = 500
HOLD_PERIODS = {"1h": 4, "2h": 8, "4h": 32, "8h": 48, "12h": 64, "24h": 96}
signals = []
last_signal_bar = -999

for idx in range(MIN_BARS, len(df)):
    ap = float(atr_pctl.iloc[idx]) if not pd.isna(atr_pctl.iloc[idx]) else 0.5
    regime = "NEUTRAL_CHOP" if ap < 0.30 else "CHOP_HARD" if ap < 0.20 else "CRISIS" if ap > 0.80 else "CHOP_MILD_BEAR" if ap < 0.50 else "TRENDING" if ap >= 0.70 else "NEUTRAL"

    ls_z = float(df["ls_zscore"].iloc[idx]) if not pd.isna(df["ls_zscore"].iloc[idx]) else 0
    funding = float(df["funding_sim"].iloc[idx]) if not pd.isna(df["funding_sim"].iloc[idx]) else 0
    oi_roc = float(df["oi_roc_sim"].iloc[idx]) if not pd.isna(df["oi_roc_sim"].iloc[idx]) else 0
    whale = df["whale"].iloc[idx]
    rsi_val = float(df["rsi"].iloc[idx]) if not pd.isna(df["rsi"].iloc[idx]) else 50
    vt = float(df["vol_trend"].iloc[idx]) if not pd.isna(df["vol_trend"].iloc[idx]) else 1.0

    m4b_div = "NONE"
    m4b_ago = 99
    for ci in range(max(0, idx - 24), idx + 1):
        div = df["cvd_divergence_15m"].iloc[ci]
        if div != "NONE":
            m4b_div = div
            m4b_ago = idx - ci
            break

    result = {
        "price": float(df["Close"].iloc[idx]),
        "m9": {"regime": regime, "raw": ap},
        "derivatives": {
            "ls_zscore": ls_z, "funding_rate": funding,
            "oi_roc_1h": oi_roc, "whale_signal": whale,
            "futures_flow": "NEUTRAL",
        },
        "m4b": {"divergence": m4b_div, "bars_ago": m4b_ago, "cvd_slope": 0},
        "rsi": rsi_val, "vol_trend": vt,
    }

    sq = detect_squeeze_v2(result, config=cfg, last_signal_bar=last_signal_bar, current_bar=idx)

    if sq["squeeze_type"] != "NONE":
        last_signal_bar = idx
        entry = float(df["Close"].iloc[idx])
        direction = sq["direction"]

        returns = {}
        for label, hold in HOLD_PERIODS.items():
            if idx + hold < len(df):
                ex = float(df["Close"].iloc[idx + hold])
                ret = (ex - entry) / entry * 100 if direction == "LONG" else (entry - ex) / entry * 100
                returns[f"ret_{label}"] = round(ret, 3)

        signals.append({
            "time": str(df["Open time"].iloc[idx]), "price": entry,
            "type": sq["squeeze_type"], "score": sq["squeeze_score"],
            "strong": sq["squeeze_strong"], "direction": direction,
            "ls_z": round(ls_z, 2), "funding": round(funding, 6),
            "oi_roc": round(oi_roc, 2), "whale": whale,
            "m4b_div": m4b_div, "rsi": round(rsi_val, 1),
            "vol_trend": round(vt, 2), "gates": len(sq["gates_passed"]),
            **returns,
        })

df_s = pd.DataFrame(signals)

print("=" * 72)
print("  M18 SQUEEZE v2 — BACKTEST RESULTS")
print("=" * 72)
print(f"\n  Total signals: {len(df_s)}")
if len(df_s) > 0:
    print(f"  SHORT_SQUEEZE: {len(df_s[df_s['type']=='SHORT_SQUEEZE'])}")
    print(f"  LONG_SQUEEZE:  {len(df_s[df_s['type']=='LONG_SQUEEZE'])}")
    print(f"  Strong:        {len(df_s[df_s['strong']])}")

    print(f"\n  {'Hold':>6}  {'Win%':>6}  {'Avg%':>7}  {'Med%':>7}  {'Max%':>7}  {'Min%':>7}  {'n':>4}")
    print(f"  {'-'*56}")
    for label in ["1h", "2h", "4h", "8h", "12h", "24h"]:
        col = f"ret_{label}"
        if col in df_s.columns:
            valid = df_s[col].dropna()
            if len(valid) > 0:
                wr = (valid > 0).mean() * 100
                print(f"  {label:>6}  {wr:>5.1f}%  {valid.mean():>+6.2f}%  {valid.median():>+6.2f}%  {valid.max():>+6.2f}%  {valid.min():>+6.2f}%  {len(valid):>4}")

    print(f"\n  ALL SIGNALS:")
    print(f"  {'Time':>16}  {'Type':<15} {'Score':>5} {'Dir':<5} {'Price':>8} {'z':>6} {'Vol':>5} {'Whale':<16} {'4h':>7} {'24h':>7}")
    print(f"  {'-'*90}")
    for _, s in df_s.iterrows():
        h4 = f"{s.get('ret_4h', 0):+.2f}%" if not pd.isna(s.get('ret_4h')) else "-"
        h24 = f"{s.get('ret_24h', 0):+.2f}%" if not pd.isna(s.get('ret_24h')) else "-"
        st = "*" if s["strong"] else " "
        print(f"  {s['time'][:16]}  {s['type']:<15} {s['score']:.3f} {st}{s['direction']:<4} ${s['price']:>7.0f} {s['ls_z']:>+5.2f} {s['vol_trend']:>4.1f}x {s['whale']:<16} {h4:>7} {h24:>7}")
else:
    print("  No signals generated — v2 filters are very selective.")

print("=" * 72)

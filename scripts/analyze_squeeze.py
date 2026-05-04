#!/usr/bin/env python3
"""Analyze squeeze winners vs losers to find optimal filters."""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import numpy as np
import pandas as pd
from src.config import CONFIG
from src.utils.data_handler import load_data
from src.utils.indicators import calc_atr, calc_rsi, calc_vol_ratio
from src.modules.m4_cvd import calc_cvd_15m, detect_cvd_divergence_15m

CSV = os.path.join(os.path.dirname(os.path.dirname(__file__)), "eth_15m_merged.csv")
df = load_data(CSV)
df = df[df["Open time"] >= "2026-01-01"].reset_index(drop=True)

cfg = dict(CONFIG)
df["atr"] = calc_atr(df["High"], df["Low"], df["Close"], cfg["ATR_PERIOD"])
df["rsi"] = calc_rsi(df["Close"], 14)
df["vol_ratio"] = calc_vol_ratio(df["Volume"])
df["taker_ratio"] = (df["Taker buy base asset volume"] / df["Volume"].replace(0, np.nan)).fillna(0.5)
df["cvd_15m"] = calc_cvd_15m(df)
df["cvd_divergence_15m"] = detect_cvd_divergence_15m(df, cfg["CVD_LOOKBACK"], cfg["CVD_DIVERGENCE_WINDOW"])

# Synthetic signals
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

rsi_ma = df["rsi"].rolling(8).mean()
df["rsi_slope"] = df["rsi"] - rsi_ma
df["vol_trend"] = df["Volume"] / df["Volume"].rolling(20).mean()
df["mom_4h"] = df["Close"].pct_change(16) * 100

# Collect signals
MIN_BARS = 500
rows = []
for idx in range(MIN_BARS, len(df)):
    ap = float(atr_pctl.iloc[idx]) if not pd.isna(atr_pctl.iloc[idx]) else 0.5
    regime = "NEUTRAL_CHOP" if ap < 0.30 else "CHOP_HARD" if ap < 0.20 else "CRISIS" if ap > 0.80 else "CHOP_MILD" if ap < 0.50 else "TRENDING" if ap >= 0.70 else "NEUTRAL"
    compressed = regime in ("NEUTRAL_CHOP", "CHOP_HARD", "CHOP_MILD")

    ls_z = float(df["ls_zscore"].iloc[idx]) if not pd.isna(df["ls_zscore"].iloc[idx]) else 0
    funding = float(df["funding_sim"].iloc[idx]) if not pd.isna(df["funding_sim"].iloc[idx]) else 0
    oi_roc = float(df["oi_roc_sim"].iloc[idx]) if not pd.isna(df["oi_roc_sim"].iloc[idx]) else 0
    whale = df["whale"].iloc[idx]
    rsi = float(df["rsi"].iloc[idx]) if not pd.isna(df["rsi"].iloc[idx]) else 50
    rsi_s = float(df["rsi_slope"].iloc[idx]) if not pd.isna(df["rsi_slope"].iloc[idx]) else 0
    vol_t = float(df["vol_trend"].iloc[idx]) if not pd.isna(df["vol_trend"].iloc[idx]) else 1.0
    mom = float(df["mom_4h"].iloc[idx]) if not pd.isna(df["mom_4h"].iloc[idx]) else 0

    m4b_div = "NONE"
    m4b_ago = 99
    for ci in range(max(0, idx - 24), idx + 1):
        div = df["cvd_divergence_15m"].iloc[ci]
        if div != "NONE":
            m4b_div = div
            m4b_ago = idx - ci
            break

    # Score both directions
    short_sc = 0.0
    long_sc = 0.0
    if ls_z <= -1.8 and compressed:
        short_sc = 0.15 + (0.25 if ls_z <= -2.5 else 0.15)
        if funding < 0: short_sc += 0.10
        elif funding < 0.002: short_sc += 0.15
        if oi_roc > 1.0: short_sc += 0.15
        elif oi_roc > 0.3: short_sc += 0.08
        if whale == "WHALE_BULLISH": short_sc += 0.15
        if m4b_div == "BULLISH" and m4b_ago <= 24: short_sc += 0.10

    if ls_z >= 1.8 and compressed:
        long_sc = 0.15 + (0.25 if ls_z >= 2.5 else 0.15)
        if funding > 0: long_sc += 0.10
        elif funding > -0.002: long_sc += 0.15
        if oi_roc > 1.0: long_sc += 0.15
        elif oi_roc > 0.3: long_sc += 0.08
        if whale == "WHALE_BEARISH": long_sc += 0.15
        if m4b_div == "BEARISH" and m4b_ago <= 24: long_sc += 0.10

    if short_sc >= 0.55 and short_sc > long_sc:
        sq_type, direction, score = "SHORT_SQUEEZE", "LONG", min(short_sc, 1.0)
    elif long_sc >= 0.55 and long_sc > short_sc:
        sq_type, direction, score = "LONG_SQUEEZE", "SHORT", min(long_sc, 1.0)
    else:
        continue

    entry = float(df["Close"].iloc[idx])
    def calc_ret(hold):
        if idx + hold < len(df):
            ex = float(df["Close"].iloc[idx + hold])
            return (ex - entry) / entry * 100 if direction == "LONG" else (entry - ex) / entry * 100
        return None

    rows.append({
        "type": sq_type, "score": score, "direction": direction,
        "regime": regime, "ls_z": round(ls_z, 2), "funding": round(funding, 6),
        "oi_roc": round(oi_roc, 2), "whale": whale, "m4b_div": m4b_div, "m4b_ago": m4b_ago,
        "rsi": round(rsi, 1), "rsi_slope": round(rsi_s, 2), "vol_trend": round(vol_t, 2),
        "mom_4h": round(mom, 2), "atr_pctl": round(ap, 3),
        "ret_1h": calc_ret(4), "ret_2h": calc_ret(8), "ret_4h": calc_ret(32),
        "ret_8h": calc_ret(48), "ret_12h": calc_ret(64), "ret_24h": calc_ret(96),
    })

df_s = pd.DataFrame(rows).dropna(subset=["ret_4h"])
print(f"Total signals: {len(df_s)}, Win rate 4h: {(df_s['ret_4h']>0).mean()*100:.1f}%")

# Winners vs losers
w = df_s[df_s["ret_4h"] > 0]
l = df_s[df_s["ret_4h"] <= 0]
print(f"\n{'='*60}")
print("  WINNERS vs LOSERS")
print(f"{'='*60}")
for col in ["ls_z", "score", "rsi", "rsi_slope", "vol_trend", "mom_4h", "atr_pctl", "oi_roc", "m4b_ago"]:
    print(f"  {col:>12}: W={w[col].mean():+.3f}  L={l[col].mean():+.3f}  delta={w[col].mean()-l[col].mean():+.3f}")

print(f"\n  Whale:")
for v in ["NEUTRAL", "WHALE_BULLISH", "WHALE_BEARISH"]:
    wn = len(w[w["whale"]==v]); ln = len(l[l["whale"]==v]); t = wn+ln
    print(f"    {v:<16} W={wn} L={ln} WR={wn/t*100:.1f}%" if t > 0 else f"    {v}: n=0")

print(f"\n  M4b div:")
for v in ["NONE", "BULLISH", "BEARISH"]:
    wn = len(w[w["m4b_div"]==v]); ln = len(l[l["m4b_div"]==v]); t = wn+ln
    print(f"    {v:<10} W={wn} L={ln} WR={wn/t*100:.1f}%" if t > 0 else f"    {v}: n=0")

# Test filter combos
print(f"\n{'='*60}")
print("  FILTER COMBOS (4h hold)")
print(f"{'='*60}")
combos = [
    ("baseline", pd.Series(True, index=df_s.index)),
    ("|z|>=2.0", df_s["ls_z"].abs() >= 2.0),
    ("|z|>=2.5", df_s["ls_z"].abs() >= 2.5),
    ("|z|>=3.0", df_s["ls_z"].abs() >= 3.0),
    ("|z|>=2.0 + whale", (df_s["ls_z"].abs() >= 2.0) & (df_s["whale"] != "NEUTRAL")),
    ("|z|>=2.5 + whale", (df_s["ls_z"].abs() >= 2.5) & (df_s["whale"] != "NEUTRAL")),
    ("|z|>=2.0 + M4b", (df_s["ls_z"].abs() >= 2.0) & (df_s["m4b_div"] != "NONE")),
    ("|z|>=2.5 + M4b", (df_s["ls_z"].abs() >= 2.5) & (df_s["m4b_div"] != "NONE")),
    ("score>=0.65", df_s["score"] >= 0.65),
    ("score>=0.70", df_s["score"] >= 0.70),
    ("score>=0.75", df_s["score"] >= 0.75),
    ("score>=0.65 + whale", (df_s["score"] >= 0.65) & (df_s["whale"] != "NEUTRAL")),
    ("score>=0.70 + whale", (df_s["score"] >= 0.70) & (df_s["whale"] != "NEUTRAL")),
    ("score>=0.70 + |z|>=2.0", (df_s["score"] >= 0.70) & (df_s["ls_z"].abs() >= 2.0)),
    ("score>=0.70 + |z|>=2.0 + whale", (df_s["score"] >= 0.70) & (df_s["ls_z"].abs() >= 2.0) & (df_s["whale"] != "NEUTRAL")),
    ("score>=0.70 + |z|>=2.5 + whale", (df_s["score"] >= 0.70) & (df_s["ls_z"].abs() >= 2.5) & (df_s["whale"] != "NEUTRAL")),
    ("score>=0.70 + |z|>=2.0 + M4b", (df_s["score"] >= 0.70) & (df_s["ls_z"].abs() >= 2.0) & (df_s["m4b_div"] != "NONE")),
    ("score>=0.70 + whale + M4b", (df_s["score"] >= 0.70) & (df_s["whale"] != "NEUTRAL") & (df_s["m4b_div"] != "NONE")),
    ("|z|>=2.0 + whale + vol>1.3", (df_s["ls_z"].abs() >= 2.0) & (df_s["whale"] != "NEUTRAL") & (df_s["vol_trend"] > 1.3)),
    ("|z|>=2.0 + whale + M4b + vol>1.3", (df_s["ls_z"].abs() >= 2.0) & (df_s["whale"] != "NEUTRAL") & (df_s["m4b_div"] != "NONE") & (df_s["vol_trend"] > 1.3)),
    ("|z|>=2.5 + whale + M4b + vol>1.3", (df_s["ls_z"].abs() >= 2.5) & (df_s["whale"] != "NEUTRAL") & (df_s["m4b_div"] != "NONE") & (df_s["vol_trend"] > 1.3)),
    ("score>=0.70 + |z|>=2.0 + whale + vol>1.2", (df_s["score"] >= 0.70) & (df_s["ls_z"].abs() >= 2.0) & (df_s["whale"] != "NEUTRAL") & (df_s["vol_trend"] > 1.2)),
    ("score>=0.70 + |z|>=2.0 + whale + M4b + vol>1.2", (df_s["score"] >= 0.70) & (df_s["ls_z"].abs() >= 2.0) & (df_s["whale"] != "NEUTRAL") & (df_s["m4b_div"] != "NONE") & (df_s["vol_trend"] > 1.2)),
]

print(f"\n  {'Filter':<48} {'n':>3} {'WR4h':>6} {'Avg4h':>7} {'WR24h':>6} {'Avg24h':>7}")
print(f"  {'-'*80}")
for name, mask in combos:
    sub = df_s[mask]
    if len(sub) == 0:
        print(f"  {name:<48} {'0':>3} {'—':>6} {'—':>7} {'—':>6} {'—':>7}")
        continue
    n = len(sub)
    wr4 = (sub["ret_4h"] > 0).mean() * 100
    a4 = sub["ret_4h"].mean()
    wr24 = (sub["ret_24h"] > 0).mean() * 100 if "ret_24h" in sub else 0
    a24 = sub["ret_24h"].mean() if "ret_24h" in sub else 0
    m4 = " <<<" if wr4 >= 75 and n >= 5 else ""
    m24 = " <<<" if wr24 >= 75 and n >= 5 else ""
    print(f"  {name:<48} {n:>3} {wr4:>5.1f}% {a4:>+6.2f}% {wr24:>5.1f}% {a24:>+6.2f}%{m4}{m24}")

# Monthly stability for best filters
print(f"\n{'='*60}")
print("  MONTHLY STABILITY (top filters)")
print(f"{'='*60}")

best_filters = [
    ("score>=0.70 + |z|>=2.0 + whale", (df_s["score"] >= 0.70) & (df_s["ls_z"].abs() >= 2.0) & (df_s["whale"] != "NEUTRAL")),
    ("score>=0.70 + whale + M4b", (df_s["score"] >= 0.70) & (df_s["whale"] != "NEUTRAL") & (df_s["m4b_div"] != "NONE")),
    ("|z|>=2.5 + whale + M4b + vol>1.3", (df_s["ls_z"].abs() >= 2.5) & (df_s["whale"] != "NEUTRAL") & (df_s["m4b_div"] != "NONE") & (df_s["vol_trend"] > 1.3)),
]

for name, mask in best_filters:
    sub = df_s[mask].copy()
    if len(sub) == 0:
        continue
    print(f"\n  {name}:")
    # We don't have timestamps in df_s, so just show overall stats
    wr4 = (sub["ret_4h"] > 0).mean() * 100
    wr24 = (sub["ret_24h"] > 0).mean() * 100
    print(f"    4h:  {wr4:.1f}% WR, avg {sub['ret_4h'].mean():+.2f}% (n={len(sub)})")
    print(f"    24h: {wr24:.1f}% WR, avg {sub['ret_24h'].mean():+.2f}%")
    # Show all signals
    for _, s in sub.iterrows():
        print(f"      {s['type']:<15} z={s['ls_z']:+.2f} whale={s['whale']:<16} m4b={s['m4b_div']:<8} 4h={s['ret_4h']:+.2f}%  24h={s['ret_24h']:+.2f}%")

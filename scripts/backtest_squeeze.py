#!/usr/bin/env python3
"""Backtest M18 Squeeze Detector on historical data for 2026."""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import numpy as np
import pandas as pd
from datetime import datetime
from src.config import CONFIG
from src.utils.data_handler import load_data, resample_ohlcv
from src.utils.indicators import calc_ema, calc_macd, calc_rsi, calc_atr, calc_vwap, calc_vol_ratio, calc_swing_bias, calc_phase0, calc_trend_state
from src.modules.m4_cvd import calc_cvd_15m, detect_cvd_divergence_15m, calc_cvd_2h, detect_cvd_zero_cross
from src.modules.m18_squeeze import detect_squeeze, SQUEEZE_DEFAULTS

SEP = "=" * 72
THIN = "-" * 56
HOLD_BARS = [4, 8, 16, 32, 48, 96]  # 1h, 2h, 4h, 8h, 12h, 24h on 15m
YEAR_START = "2026-01-01"

print(SEP)
print("  M18 SQUEEZE DETECTOR - 2026 BACKTEST")
print(SEP)

# Load data
csv_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "eth_15m_merged.csv")
if not os.path.exists(csv_path):
    csv_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "eth_15m_merged.csv")

print(f"\n  Loading data from {csv_path}...")
df = load_data(csv_path)
df = df[df["Open time"] >= YEAR_START].reset_index(drop=True)
print(f"  Bars: {len(df)}  ({df['Open time'].iloc[0]} -> {df['Open time'].iloc[-1]})")

# Compute indicators
cfg = dict(CONFIG)
cfg["M18_ENABLED"] = True
cfg["SQUEEZE_OVERRIDE_REGIME"] = True

print("  Computing indicators...")
df["vwap"] = calc_vwap(df["High"], df["Low"], df["Close"], df["Volume"], cfg["VWAP_LOOKBACK"])
df["vol_ma20"] = df["Volume"].rolling(20).mean()
taker_base = df["Taker buy base asset volume"]
total_vol = df["Volume"]
df["taker_ratio"] = (taker_base / total_vol.replace(0, np.nan)).fillna(0.5)
df["atr"] = calc_atr(df["High"], df["Low"], df["Close"], cfg["ATR_PERIOD"])
df["vol_ratio"] = calc_vol_ratio(df["Volume"])
df["rsi"] = calc_rsi(df["Close"], 14)
df["cvd_15m"] = calc_cvd_15m(df)
df["cvd_divergence_15m"] = detect_cvd_divergence_15m(df, cfg["CVD_LOOKBACK"], cfg["CVD_DIVERGENCE_WINDOW"])

df_1h = resample_ohlcv(df, "1H")
df_2h = resample_ohlcv(df, "2H")
df_4h = resample_ohlcv(df, "4H")
df_1d = resample_ohlcv(df, "1D")

df_1h["macd_line"], df_1h["macd_signal"], df_1h["macd_hist"] = calc_macd(
    df_1h["Close"], cfg["MACD_FAST"], cfg["MACD_SLOW"], cfg["MACD_SIGNAL"])
df_1h["ema_fast"] = calc_ema(df_1h["Close"], cfg["EMA_FAST"])
df_1h["ema_slow"] = calc_ema(df_1h["Close"], cfg["EMA_SLOW"])
df_1h["atr"] = calc_atr(df_1h["High"], df_1h["Low"], df_1h["Close"], cfg["ATR_PERIOD"])
df_1h["rsi"] = calc_rsi(df_1h["Close"], 14)
df_2h["ema_fast"] = calc_ema(df_2h["Close"], cfg["EMA_FAST"])
df_2h["ema_slow"] = calc_ema(df_2h["Close"], cfg["EMA_SLOW"])
df_2h["cvd_2h"] = calc_cvd_2h(df_2h)
df_2h["cvd_zl_state"], df_2h["cvd_zl_cross_bar"], df_2h["cvd_zl_cross_dir"] = detect_cvd_zero_cross(df_2h)
df_1d["swing_bias"] = calc_swing_bias(df_1d)
df_1d["phase0"] = calc_phase0(df_1d)
df_1d["trend"], df_1d["trend_score"] = calc_trend_state(df_1d)
df_4h["ema_fast"] = calc_ema(df_4h["Close"], cfg["EMA_FAST"])
df_4h["ema_slow"] = calc_ema(df_4h["Close"], cfg["EMA_SLOW"])
df_4h["macd_line"], df_4h["macd_signal"], df_4h["macd_hist"] = calc_macd(
    df_4h["Close"], cfg["MACD_FAST"], cfg["MACD_SLOW"], cfg["MACD_SIGNAL"])

# Simulate derivatives signals from available data
print("  Building synthetic derivatives signals...")

taker_ma50 = df["taker_ratio"].rolling(50).mean()
taker_std50 = df["taker_ratio"].rolling(50).std()
df["ls_zscore"] = (df["taker_ratio"] - taker_ma50) / taker_std50.replace(0, 1)

df["funding_sim"] = (df["rsi"] - 50) / 50 * 0.001
df["oi_roc_sim"] = df["Volume"].pct_change(4) * 100

df["whale_signal"] = "NEUTRAL"
vol_pctl = df["Volume"].rolling(100).apply(
    lambda x: (x.iloc[-1] - x.min()) / (x.max() - x.min()) if x.max() > x.min() else 0.5, raw=False)
price_change = df["Close"].pct_change(4)
df.loc[(vol_pctl > 0.8) & (price_change > 0.005), "whale_signal"] = "WHALE_BULLISH"
df.loc[(vol_pctl > 0.8) & (price_change < -0.005), "whale_signal"] = "WHALE_BEARISH"

atr_pctl = df["atr"].rolling(500, min_periods=100).apply(
    lambda x: (x.iloc[-1] - x.min()) / (x.max() - x.min()) if x.max() > x.min() else 0.5, raw=False)
df["regime"] = "NEUTRAL"
df.loc[atr_pctl < 0.30, "regime"] = "NEUTRAL_CHOP"
df.loc[atr_pctl < 0.20, "regime"] = "CHOP_HARD"
df.loc[atr_pctl > 0.80, "regime"] = "CRISIS"
df.loc[(atr_pctl >= 0.30) & (atr_pctl < 0.50), "regime"] = "CHOP_MILD_BEAR"
df.loc[(atr_pctl >= 0.50) & (atr_pctl < 0.70), "regime"] = "NEUTRAL_TRENDING"
df.loc[(atr_pctl >= 0.70) & (atr_pctl <= 0.80), "regime"] = "TRENDING"

# Scan for squeeze signals
print("  Scanning for squeeze signals...")

MIN_BARS = 500
signals = []

for idx in range(MIN_BARS, len(df)):
    row = df.iloc[idx]
    regime = df["regime"].iloc[idx]
    ls_z = float(df["ls_zscore"].iloc[idx]) if not pd.isna(df["ls_zscore"].iloc[idx]) else 0
    funding = float(df["funding_sim"].iloc[idx]) if not pd.isna(df["funding_sim"].iloc[idx]) else 0
    oi_roc = float(df["oi_roc_sim"].iloc[idx]) if not pd.isna(df["oi_roc_sim"].iloc[idx]) else 0
    whale = df["whale_signal"].iloc[idx]

    m4b_div = "NONE"
    m4b_bars_ago = -1
    for ci in range(max(0, idx - 24), idx + 1):
        div = df["cvd_divergence_15m"].iloc[ci]
        if div != "NONE":
            m4b_div = div
            m4b_bars_ago = idx - ci
            break

    result = {
        "price": float(row["Close"]),
        "m9": {"regime": regime, "raw": float(atr_pctl.iloc[idx]) if not pd.isna(atr_pctl.iloc[idx]) else 0.5},
        "derivatives": {
            "ls_zscore": ls_z,
            "funding_rate": funding,
            "oi_roc_1h": oi_roc,
            "whale_signal": whale,
            "futures_flow": "NEUTRAL",
        },
        "m4b": {"divergence": m4b_div, "bars_ago": m4b_bars_ago, "cvd_slope": 0},
    }

    sq = detect_squeeze(result, config=cfg)

    if sq["squeeze_type"] != "NONE":
        entry_price = float(row["Close"])
        direction = sq["direction"]
        ts = str(row["Open time"])

        returns = {}
        for bars in HOLD_BARS:
            if idx + bars < len(df):
                exit_price = float(df["Close"].iloc[idx + bars])
                if direction == "LONG":
                    ret = (exit_price - entry_price) / entry_price * 100
                else:
                    ret = (entry_price - exit_price) / entry_price * 100
                returns[f"{bars}bars"] = round(ret, 3)

        signals.append({
            "time": ts, "price": entry_price, "type": sq["squeeze_type"],
            "score": sq["squeeze_score"], "strong": sq["squeeze_strong"],
            "direction": direction, "factors": len(sq["factors"]),
            "regime": regime, "ls_z": round(ls_z, 2),
            "funding": round(funding, 6), "oi_roc": round(oi_roc, 2),
            **returns,
        })

print(f"  Found {len(signals)} squeeze signals\n")

if not signals:
    print("  No signals found. Exiting.")
    sys.exit(0)

df_sig = pd.DataFrame(signals)

print(SEP)
print("  SIGNAL OVERVIEW")
print(SEP)
print(f"  Total signals:    {len(df_sig)}")
print(f"  SHORT_SQUEEZE:    {len(df_sig[df_sig['type'] == 'SHORT_SQUEEZE'])}  (-> LONG)")
print(f"  LONG_SQUEEZE:     {len(df_sig[df_sig['type'] == 'LONG_SQUEEZE'])}  (-> SHORT)")
print(f"  Strong signals:   {len(df_sig[df_sig['strong']])}")
print(f"  Moderate signals: {len(df_sig[~df_sig['strong']])}")

# Performance by hold period
print(f"\n{SEP}")
print("  PERFORMANCE BY HOLD PERIOD")
print(SEP)
hdr = f"  {'Hold':>6}  {'Win%':>6}  {'Avg%':>7}  {'Med%':>7}  {'Max%':>7}  {'Min%':>7}  {'n':>4}"
print(hdr)
print("  " + THIN)

for bars in HOLD_BARS:
    col = f"{bars}bars"
    if col in df_sig.columns:
        valid = df_sig[col].dropna()
        if len(valid) > 0:
            wins = (valid > 0).sum()
            win_pct = wins / len(valid) * 100
            avg = valid.mean()
            med = valid.median()
            mx = valid.max()
            mn = valid.min()
            label = f"{bars*15}min" if bars * 15 < 60 else f"{bars*15/60:.0f}h"
            print(f"  {label:>6}  {win_pct:>5.1f}%  {avg:>+6.2f}%  {med:>+6.2f}%  {mx:>+6.2f}%  {mn:>+6.2f}%  {len(valid):>4}")

# By squeeze type
print(f"\n{SEP}")
print("  PERFORMANCE BY SQUEEZE TYPE")
print(SEP)

for sq_type in ["SHORT_SQUEEZE", "LONG_SQUEEZE"]:
    subset = df_sig[df_sig["type"] == sq_type]
    if len(subset) == 0:
        continue
    dir_label = "LONG" if sq_type == "SHORT_SQUEEZE" else "SHORT"
    print(f"\n  {sq_type} (-> {dir_label}):  {len(subset)} signals")
    for bars in HOLD_BARS:
        col = f"{bars}bars"
        if col in subset.columns:
            valid = subset[col].dropna()
            if len(valid) > 0:
                wins = (valid > 0).sum()
                win_pct = wins / len(valid) * 100
                avg = valid.mean()
                label = f"{bars*15}min" if bars * 15 < 60 else f"{bars*15/60:.0f}h"
                print(f"    {label:>6}:  {win_pct:>5.1f}% win  avg {avg:>+6.2f}%  (n={len(valid)})")

# By strength
print(f"\n{SEP}")
print("  PERFORMANCE BY STRENGTH")
print(SEP)

for strong_val in [True, False]:
    subset = df_sig[df_sig["strong"] == strong_val]
    label = "STRONG" if strong_val else "MODERATE"
    if len(subset) == 0:
        continue
    print(f"\n  {label}:  {len(subset)} signals")
    for bars in [8, 32, 96]:
        col = f"{bars}bars"
        if col in subset.columns:
            valid = subset[col].dropna()
            if len(valid) > 0:
                wins = (valid > 0).sum()
                win_pct = wins / len(valid) * 100
                avg = valid.mean()
                label_t = f"{bars*15}min" if bars * 15 < 60 else f"{bars*15/60:.0f}h"
                print(f"    {label_t:>6}:  {win_pct:>5.1f}% win  avg {avg:>+6.2f}%  (n={len(valid)})")

# Monthly
print(f"\n{SEP}")
print("  MONTHLY BREAKDOWN")
print(SEP)

df_sig["month"] = pd.to_datetime(df_sig["time"]).dt.to_period("M")
for month, group in df_sig.groupby("month"):
    col = "32bars"
    if col in group.columns:
        valid = group[col].dropna()
        if len(valid) > 0:
            wins = (valid > 0).sum()
            win_pct = wins / len(valid) * 100
            avg = valid.mean()
            total = len(valid)
            short_sq = len(group[group["type"] == "SHORT_SQUEEZE"])
            long_sq = len(group[group["type"] == "LONG_SQUEEZE"])
            print(f"  {month}  n={total:>3}  (S:{short_sq} L:{long_sq})  4h: {win_pct:>5.1f}% win  avg {avg:>+6.2f}%")

# Top/bottom signals
print(f"\n{SEP}")
print("  TOP 5 BEST SIGNALS (4h return)")
print(SEP)

col = "32bars"
if col in df_sig.columns:
    top = df_sig.nlargest(5, col)
    for _, s in top.iterrows():
        print(f"  {s['time'][:16]}  {s['type']:<15}  score={s['score']:.3f}  "
              f"${s['price']:.0f}  {s['direction']:<5}  4h: {s[col]:+.2f}%")

print(f"\n  TOP 5 WORST SIGNALS (4h return)")
print("  " + THIN)
if col in df_sig.columns:
    bottom = df_sig.nsmallest(5, col)
    for _, s in bottom.iterrows():
        print(f"  {s['time'][:16]}  {s['type']:<15}  score={s['score']:.3f}  "
              f"${s['price']:.0f}  {s['direction']:<5}  4h: {s[col]:+.2f}%")

# All signals
print(f"\n{SEP}")
print(f"  ALL SIGNALS ({len(df_sig)})")
print(SEP)
print(f"  {'Time':>16}  {'Type':<15}  {'Score':>5}  {'Dir':<5}  {'Price':>8}  {'1h':>7}  {'4h':>7}  {'24h':>7}")
print("  " + THIN)

for _, s in df_sig.iterrows():
    h1 = f"{s.get('4bars', 0):+.2f}%" if "4bars" in s and not pd.isna(s.get("4bars")) else "-"
    h4 = f"{s.get('32bars', 0):+.2f}%" if "32bars" in s and not pd.isna(s.get("32bars")) else "-"
    h24 = f"{s.get('96bars', 0):+.2f}%" if "96bars" in s and not pd.isna(s.get("96bars")) else "-"
    strong_tag = "*" if s["strong"] else " "
    print(f"  {s['time'][:16]}  {s['type']:<15}  {s['score']:.3f}  {strong_tag}{s['direction']:<4}  ${s['price']:>7.0f}  {h1:>7}  {h4:>7}  {h24:>7}")

print(SEP)

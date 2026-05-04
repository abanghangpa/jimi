#!/usr/bin/env python3
"""Deep analysis of squeeze signals using actual market data features."""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import numpy as np
import pandas as pd
from itertools import combinations
from src.config import CONFIG
from src.utils.data_handler import load_data
from src.utils.indicators import calc_atr, calc_rsi, calc_vwap, calc_vol_ratio, calc_ema, calc_macd
from src.modules.m4_cvd import calc_cvd_15m, detect_cvd_divergence_15m

CSV = os.path.join(os.path.dirname(os.path.dirname(__file__)), "eth_15m_merged.csv")
df = load_data(CSV)
df = df[df["Open time"] >= "2026-01-01"].reset_index(drop=True)

cfg = dict(CONFIG)

# ── Compute ALL available features ──
print("Computing features...")

df["atr"] = calc_atr(df["High"], df["Low"], df["Close"], cfg["ATR_PERIOD"])
df["rsi"] = calc_rsi(df["Close"], 14)
df["vwap"] = calc_vwap(df["High"], df["Low"], df["Close"], df["Volume"], cfg["VWAP_LOOKBACK"])
df["vol_ratio"] = calc_vol_ratio(df["Volume"])
df["vol_ma20"] = df["Volume"].rolling(20).mean()
df["vol_trend"] = df["Volume"] / df["vol_ma20"]
df["taker_ratio"] = (df["Taker buy base asset volume"] / df["Volume"].replace(0, np.nan)).fillna(0.5)
df["cvd_15m"] = calc_cvd_15m(df)
df["cvd_divergence_15m"] = detect_cvd_divergence_15m(df, cfg["CVD_LOOKBACK"], cfg["CVD_DIVERGENCE_WINDOW"])

# EMAs
df["ema_fast"] = calc_ema(df["Close"], cfg["EMA_FAST"])
df["ema_slow"] = calc_ema(df["Close"], cfg["EMA_SLOW"])
df["ema_200"] = calc_ema(df["Close"], 200)

# MACD
df["macd_line"], df["macd_signal"], df["macd_hist"] = calc_macd(
    df["Close"], cfg["MACD_FAST"], cfg["MACD_SLOW"], cfg["MACD_SIGNAL"])

# ATR percentile (rolling)
atr_pctl = df["atr"].rolling(500, min_periods=100).apply(
    lambda x: (x.iloc[-1] - x.min()) / (x.max() - x.min()) if x.max() > x.min() else 0.5, raw=False)

# Price compression: range width over last 48 bars (12h)
roll_high = df["High"].rolling(48).max()
roll_low = df["Low"].rolling(48).min()
df["range_width"] = (roll_high - roll_low) / df["Close"] * 100

# Bollinger squeeze
bb_ma = df["Close"].rolling(20).mean()
bb_std = df["Close"].rolling(20).std()
df["bb_upper"] = bb_ma + 2 * bb_std
df["bb_lower"] = bb_ma - 2 * bb_std
df["bb_width"] = (df["bb_upper"] - df["bb_lower"]) / bb_ma * 100
df["bb_pctl"] = df["bb_width"].rolling(500, min_periods=100).apply(
    lambda x: (x.iloc[-1] - x.min()) / (x.max() - x.min()) if x.max() > x.min() else 0.5, raw=False)

# Volume divergence: price making new highs/lows but volume declining
df["price_high_20"] = df["Close"].rolling(20).max()
df["price_low_20"] = df["Close"].rolling(20).min()
df["vol_ma5"] = df["Volume"].rolling(5).mean()
df["vol_declining"] = (df["vol_ma5"] < df["vol_ma20"]).astype(int)

# Taker ratio momentum
df["taker_ma"] = df["taker_ratio"].rolling(20).mean()
df["taker_std"] = df["taker_ratio"].rolling(20).std()
df["taker_zscore"] = (df["taker_ratio"] - df["taker_ma"]) / df["taker_std"].replace(0, 1)

# Taker ratio trend (is buying/selling pressure increasing?)
df["taker_slope"] = df["taker_ratio"] - df["taker_ratio"].rolling(8).mean()

# CVD slope (momentum of cumulative delta)
df["cvd_slope"] = df["cvd_15m"] - df["cvd_15m"].rolling(8).mean()

# RSI momentum
rsi_ma = df["rsi"].rolling(8).mean()
df["rsi_slope"] = df["rsi"] - rsi_ma

# Price momentum at various lookbacks
df["mom_1h"] = df["Close"].pct_change(4) * 100
df["mom_4h"] = df["Close"].pct_change(16) * 100
df["mom_12h"] = df["Close"].pct_change(48) * 100

# OI proxy: open interest estimate from volume * (1 - taker_ratio close correlation)
# Actually, use volume accumulation as OI proxy
df["vol_cumsum"] = df["Volume"].rolling(48).sum()
df["vol_cumsum_ma"] = df["vol_cumsum"].rolling(20).mean()
df["oi_proxy"] = df["vol_cumsum"] / df["vol_cumsum_ma"]

# Funding proxy: based on premium/discount to VWAP
df["vwap_dist"] = (df["Close"] - df["vwap"]) / df["vwap"] * 100

# ── Collect squeeze signals with FULL feature set ──
MIN_BARS = 500
HOLD = 32  # 4h

taker_ma50 = df["taker_ratio"].rolling(50).mean()
taker_std50 = df["taker_ratio"].rolling(50).std()
ls_zscore = (df["taker_ratio"] - taker_ma50) / taker_std50.replace(0, 1)
funding_sim = (df["rsi"] - 50) / 50 * 0.001
oi_roc = df["Volume"].pct_change(4) * 100

vol_pctl = df["Volume"].rolling(100).apply(
    lambda x: (x.iloc[-1] - x.min()) / (x.max() - x.min()) if x.max() > x.min() else 0.5, raw=False)
price_change = df["Close"].pct_change(4)
whale = pd.Series("NEUTRAL", index=df.index)
whale[(vol_pctl > 0.8) & (price_change > 0.005)] = "WHALE_BULLISH"
whale[(vol_pctl > 0.8) & (price_change < -0.005)] = "WHALE_BEARISH"

signals = []
for idx in range(MIN_BARS, len(df)):
    ap = float(atr_pctl.iloc[idx]) if not pd.isna(atr_pctl.iloc[idx]) else 0.5
    compressed = ap < 0.35

    ls_z = float(ls_zscore.iloc[idx]) if not pd.isna(ls_zscore.iloc[idx]) else 0
    if abs(ls_z) < 1.8:
        continue

    if not compressed:
        continue

    # Direction from z-score sign
    if ls_z < 0:
        direction = "LONG"
        sq_type = "SHORT_SQUEEZE"
    else:
        direction = "SHORT"
        sq_type = "LONG_SQUEEZE"

    entry = float(df["Close"].iloc[idx])
    if idx + HOLD >= len(df):
        continue
    exit_p = float(df["Close"].iloc[idx + HOLD])
    ret = (exit_p - entry) / entry * 100 if direction == "LONG" else (entry - exit_p) / entry * 100

    # Collect all features
    def safe(col, default=0):
        v = df[col].iloc[idx] if col in df.columns else default
        return round(float(v), 4) if not pd.isna(v) else default

    signals.append({
        "idx": idx, "type": sq_type, "direction": direction, "ret_4h": round(ret, 3),
        # Core
        "ls_z": round(ls_z, 2),
        "atr_pctl": round(ap, 3),
        "range_width": safe("range_width"),
        "bb_pctl": safe("bb_pctl"),
        # Volume
        "vol_trend": safe("vol_trend"),
        "vol_ratio": safe("vol_ratio"),
        "vol_declining": safe("vol_declining"),
        # Taker
        "taker_ratio": safe("taker_ratio"),
        "taker_zscore": safe("taker_zscore"),
        "taker_slope": safe("taker_slope"),
        # CVD
        "cvd_slope": safe("cvd_slope"),
        "cvd_div": df["cvd_divergence_15m"].iloc[idx] if "cvd_divergence_15m" in df.columns else "NONE",
        # Momentum
        "rsi": safe("rsi"),
        "rsi_slope": safe("rsi_slope"),
        "mom_1h": safe("mom_1h"),
        "mom_4h": safe("mom_4h"),
        "mom_12h": safe("mom_12h"),
        # Trend
        "ema_fast": safe("ema_fast"),
        "ema_slow": safe("ema_slow"),
        "ema_200": safe("ema_200"),
        "macd_hist": safe("macd_hist"),
        # VWAP
        "vwap_dist": safe("vwap_dist"),
        # OI proxy
        "oi_proxy": safe("oi_proxy"),
        "oi_roc": safe("oi_roc"),
        # Funding proxy
        "funding": round(float(funding_sim.iloc[idx]) if not pd.isna(funding_sim.iloc[idx]) else 0, 6),
        # Whale
        "whale": whale.iloc[idx],
    })

df_s = pd.DataFrame(signals)
winners = df_s[df_s["ret_4h"] > 0]
losers = df_s[df_s["ret_4h"] <= 0]

print(f"\nTotal signals: {len(df_s)}")
print(f"Win rate: {(df_s['ret_4h'] > 0).mean()*100:.1f}%")
print(f"Avg return: {df_s['ret_4h'].mean():.3f}%")

# ── Feature analysis ──
print(f"\n{'='*70}")
print("  FEATURE ANALYSIS: WINNERS vs LOSERS")
print(f"{'='*70}")

features = [
    "ls_z", "atr_pctl", "range_width", "bb_pctl",
    "vol_trend", "vol_ratio", "vol_declining",
    "taker_ratio", "taker_zscore", "taker_slope",
    "cvd_slope", "rsi", "rsi_slope",
    "mom_1h", "mom_4h", "mom_12h",
    "vwap_dist", "oi_proxy", "oi_roc", "funding",
    "ema_fast", "ema_slow", "ema_200", "macd_hist",
]

print(f"\n  {'Feature':<18} {'Win_mean':>10} {'Lose_mean':>10} {'Delta':>8} {'Sep_power':>10}")
print(f"  {'-'*60}")

feature_power = {}
for f in features:
    w = winners[f].mean()
    l = losers[f].mean()
    d = w - l
    # Separation power: |delta| / pooled_std
    ws = winners[f].std()
    ls = losers[f].std()
    pooled = np.sqrt((ws**2 + ls**2) / 2) if (ws + ls) > 0 else 1
    sep = abs(d) / pooled
    feature_power[f] = sep
    marker = " <<<" if sep > 0.15 else ""
    print(f"  {f:<18} {w:>+10.4f} {l:>+10.4f} {d:>+8.4f} {sep:>10.4f}{marker}")

# Sort by separation power
print(f"\n  FEATURES RANKED BY SEPARATION POWER:")
print(f"  {'-'*40}")
for f, p in sorted(feature_power.items(), key=lambda x: -x[1]):
    bar = "#" * int(p * 100)
    print(f"  {f:<18} {p:.4f}  {bar}")

# ── Brute-force filter combinations ──
print(f"\n{'='*70}")
print("  BRUTE-FORCE FILTER SEARCH (targeting 75%+ WR)")
print(f"{'='*70}")

# Test individual thresholds on top features
top_features = sorted(feature_power.items(), key=lambda x: -x[1])[:8]

print(f"\n  SINGLE FEATURE THRESHOLDS:")
print(f"  {'Filter':<45} {'n':>4} {'WR%':>6} {'Avg%':>7}")
print(f"  {'-'*65}")

single_filters = []
for fname, _ in top_features:
    series = df_s[fname]
    for pct in [25, 50, 75]:
        thresh_high = series.quantile(pct / 100)
        thresh_low = series.quantile(1 - pct / 100)

        # Test both directions
        mask_high = df_s[fname] >= thresh_high
        mask_low = df_s[fname] <= thresh_low

        for label, mask in [(f"{fname}>={thresh_high:.3f} (top {100-pct}%)", mask_high),
                            (f"{fname}<={thresh_low:.3f} (bottom {100-pct}%)", mask_low)]:
            sub = df_s[mask]
            if len(sub) >= 5:
                wr = (sub["ret_4h"] > 0).mean() * 100
                avg = sub["ret_4h"].mean()
                if wr >= 60 or wr <= 40:
                    single_filters.append((label, len(sub), wr, avg))
                    marker = " <<<" if wr >= 75 else ""
                    print(f"  {label:<45} {len(sub):>4} {wr:>5.1f}% {avg:>+6.2f}%{marker}")

# Test 2-feature combinations on the most promising features
print(f"\n  2-FEATURE COMBINATIONS (testing top features):")
print(f"  {'Filter':<55} {'n':>4} {'WR%':>6} {'Avg%':>7}")
print(f"  {'-'*75}")

best_combos = []

# Get promising feature pairs
promising = [f for f, _ in top_features[:6]]

for f1, f2 in combinations(promising, 2):
    s1 = df_s[f1]
    s2 = df_s[f2]

    # Test quartile combinations
    for q1 in [0.25, 0.50]:
        for q2 in [0.25, 0.50]:
            t1 = s1.quantile(q1)
            t2 = s2.quantile(q2)

            # Both high
            mask = (s1 >= t1) & (s2 >= t2)
            sub = df_s[mask]
            if len(sub) >= 5:
                wr = (sub["ret_4h"] > 0).mean() * 100
                avg = sub["ret_4h"].mean()
                if wr >= 65:
                    label = f"{f1}>={t1:.3f} & {f2}>={t2:.3f}"
                    best_combos.append((label, len(sub), wr, avg))

            # High + Low
            mask = (s1 >= t1) & (s2 <= s2.quantile(1 - q2))
            sub = df_s[mask]
            if len(sub) >= 5:
                wr = (sub["ret_4h"] > 0).mean() * 100
                avg = sub["ret_4h"].mean()
                if wr >= 65:
                    label = f"{f1}>={t1:.3f} & {f2}<={s2.quantile(1-q2):.3f}"
                    best_combos.append((label, len(sub), wr, avg))

# Sort by WR, show top 20
best_combos.sort(key=lambda x: -x[2])
for label, n, wr, avg in best_combos[:20]:
    marker = " <<<" if wr >= 75 else ""
    print(f"  {label:<55} {n:>4} {wr:>5.1f}% {avg:>+6.2f}%{marker}")

# ── 3-feature combinations on top candidates ──
print(f"\n  3-FEATURE COMBINATIONS (targeting 75%+ WR):")
print(f"  {'Filter':<65} {'n':>4} {'WR%':>6} {'Avg%':>7}")
print(f"  {'-'*85}")

three_combos = []
top3 = [f for f, _ in top_features[:5]]

for f1, f2, f3 in combinations(top3, 3):
    s1 = df_s[f1]
    s2 = df_s[f2]
    s3 = df_s[f3]

    for q in [0.25, 0.33, 0.50]:
        t1 = s1.quantile(q)
        t2 = s2.quantile(q)
        t3 = s3.quantile(q)

        # All high
        mask = (s1 >= t1) & (s2 >= t2) & (s3 >= t3)
        sub = df_s[mask]
        if len(sub) >= 5:
            wr = (sub["ret_4h"] > 0).mean() * 100
            avg = sub["ret_4h"].mean()
            if wr >= 65:
                label = f"{f1}>={t1:.3f} & {f2}>={t2:.3f} & {f3}>={t3:.3f}"
                three_combos.append((label, len(sub), wr, avg))

        # Mixed: 2 high + 1 low
        for flip in [f1, f2, f3]:
            masks = {}
            for f in [f1, f2, f3]:
                if f == flip:
                    masks[f] = df_s[f] <= df_s[f].quantile(1 - q)
                else:
                    masks[f] = df_s[f] >= df_s[f].quantile(q)
            mask = masks[f1] & masks[f2] & masks[f3]
            sub = df_s[mask]
            if len(sub) >= 5:
                wr = (sub["ret_4h"] > 0).mean() * 100
                avg = sub["ret_4h"].mean()
                if wr >= 65:
                    label = f"{f1}{'<=' if flip==f1 else '>='} & {f2}{'<=' if flip==f2 else '>='} & {f3}{'<=' if flip==f3 else '>='}"
                    three_combos.append((label, len(sub), wr, avg))

three_combos.sort(key=lambda x: -x[2])
for label, n, wr, avg in three_combos[:20]:
    marker = " <<<" if wr >= 75 else ""
    print(f"  {label:<65} {n:>4} {wr:>5.1f}% {avg:>+6.2f}%{marker}")

# ── Best overall combos with minimum sample size ──
print(f"\n{'='*70}")
print("  BEST COMBOS (WR>=70%, n>=8)")
print(f"{'='*70}")

all_combos = single_filters + best_combos + three_combos
qualified = [(l, n, w, a) for l, n, w, a in all_combos if w >= 70 and n >= 8]
qualified.sort(key=lambda x: -x[2])

print(f"\n  {'Filter':<65} {'n':>4} {'WR%':>6} {'Avg%':>7}")
print(f"  {'-'*85}")
for label, n, wr, avg in qualified[:30]:
    marker = " <<<" if wr >= 75 else ""
    print(f"  {label:<65} {n:>4} {wr:>5.1f}% {avg:>+6.2f}%{marker}")

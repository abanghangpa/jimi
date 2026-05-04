#!/usr/bin/env python3
"""Exhaustive parameter optimization for M18 squeeze v3.

Tests every combination of gates, thresholds, and filters
to find 75%+ WR configurations with statistical significance.
"""
import sys, os, itertools
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import numpy as np
import pandas as pd
from src.config import CONFIG
from src.utils.data_handler import load_data
from src.utils.indicators import calc_atr, calc_rsi, calc_vol_ratio, calc_vwap
from src.modules.m4_cvd import calc_cvd_15m, detect_cvd_divergence_15m

CSV = os.path.join(os.path.dirname(os.path.dirname(__file__)), "eth_15m_merged.csv")
df = load_data(CSV)
df = df[df["Open time"] >= "2026-01-01"].reset_index(drop=True)

cfg = dict(CONFIG)

# ── Compute indicators ──
df["atr"] = calc_atr(df["High"], df["Low"], df["Close"], cfg["ATR_PERIOD"])
df["rsi"] = calc_rsi(df["Close"], 14)
df["vol_ratio"] = calc_vol_ratio(df["Volume"])
df["vwap"] = calc_vwap(df["High"], df["Low"], df["Close"], df["Volume"], cfg["VWAP_LOOKBACK"])
df["taker_ratio"] = (df["Taker buy base asset volume"] / df["Volume"].replace(0, np.nan)).fillna(0.5)
df["vol_ma20"] = df["Volume"].rolling(20).mean()
df["vol_trend"] = df["Volume"] / df["vol_ma20"]
df["cvd_15m"] = calc_cvd_15m(df)
df["cvd_divergence_15m"] = detect_cvd_divergence_15m(df, cfg["CVD_LOOKBACK"], cfg["CVD_DIVERGENCE_WINDOW"])

# Synthetic derivatives
taker_ma = df["taker_ratio"].rolling(50).mean()
taker_std = df["taker_ratio"].rolling(50).std()
df["ls_zscore"] = (df["taker_ratio"] - taker_ma) / taker_std.replace(0, 1)
df["oi_roc_sim"] = df["Volume"].pct_change(4) * 100

vol_pctl = df["Volume"].rolling(100).apply(
    lambda x: (x.iloc[-1] - x.min()) / (x.max() - x.min()) if x.max() > x.min() else 0.5, raw=False)
price_change = df["Close"].pct_change(4)
df["whale"] = "NEUTRAL"
df.loc[(vol_pctl > 0.8) & (price_change > 0.005), "whale"] = "WHALE_BULLISH"
df.loc[(vol_pctl > 0.8) & (price_change < -0.005), "whale"] = "WHALE_BEARISH"

atr_pctl = df["atr"].rolling(500, min_periods=100).apply(
    lambda x: (x.iloc[-1] - x.min()) / (x.max() - x.min()) if x.max() > x.min() else 0.5, raw=False)

# Bar-level features
df["range_width"] = (df["High"].rolling(48).max() - df["Low"].rolling(48).min()) / df["Close"] * 100
df["vwap_dist"] = (df["Close"] - df["vwap"]) / df["vwap"] * 100
vol_cumsum_48 = df["Volume"].rolling(48).sum()
vol_cumsum_ma = df["Volume"].rolling(68).mean() * 20
df["oi_proxy"] = vol_cumsum_48 / vol_cumsum_ma.replace(0, 1)
df["bar_vol_spike"] = df["Volume"] / df["vol_ma20"]
df["bar_range"] = (df["High"] - df["Low"]) / df["Close"] * 100
bar_range_ma = df["bar_range"].rolling(20).mean()
df["bar_range_expansion"] = df["bar_range"] / bar_range_ma.replace(0, 1)
df["bar_taker_extreme"] = (df["taker_ratio"] > 0.65) | (df["taker_ratio"] < 0.35)

# RSI slope
rsi_ma = df["rsi"].rolling(8).mean()
df["rsi_slope"] = df["rsi"] - rsi_ma

# Momentum
df["mom_4h"] = df["Close"].pct_change(16) * 100
df["mom_1h"] = df["Close"].pct_change(4) * 100

# Bollinger Band width percentile
df["bb_mid"] = df["Close"].rolling(20).mean()
df["bb_std"] = df["Close"].rolling(20).std()
df["bb_upper"] = df["bb_mid"] + 2 * df["bb_std"]
df["bb_lower"] = df["bb_mid"] - 2 * df["bb_std"]
df["bb_width"] = (df["bb_upper"] - df["bb_lower"]) / df["bb_mid"] * 100
bb_width_pctl = df["bb_width"].rolling(500, min_periods=100).apply(
    lambda x: (x.iloc[-1] - x.min()) / (x.max() - x.min()) if x.max() > x.min() else 0.5, raw=False)

# Price position in Bollinger band
df["bb_pos"] = (df["Close"] - df["bb_lower"]) / (df["bb_upper"] - df["bb_lower"]).replace(0, 1)

# ── Pre-compute M4b divergence state per bar (vectorized) ──
print("Pre-computing M4b divergence states...")
div_series = df["cvd_divergence_15m"].values
n = len(div_series)
m4b_div_arr = ["NONE"] * n
m4b_ago_arr = [99] * n
# Forward scan: for each bar, find most recent divergence in last 24 bars
last_div_idx = -999
last_div_type = "NONE"
for idx in range(n):
    if div_series[idx] != "NONE":
        last_div_idx = idx
        last_div_type = div_series[idx]
    if idx - last_div_idx <= 24:
        m4b_div_arr[idx] = last_div_type
        m4b_ago_arr[idx] = idx - last_div_idx
df["m4b_div"] = m4b_div_arr
df["m4b_ago"] = m4b_ago_arr

# ── Pre-compute squeeze quality ──
def compute_quality_vec(rw, vr, oip, vd):
    rw_s = np.clip(1 - (rw - 1.5) / 4.0, 0, 1)
    vr_s = np.clip(1 - (vr - 0.05) / 0.20, 0, 1)
    oip_s = np.clip((oip - 0.7) / 0.5, 0, 1)
    vd_s = np.clip(1 - np.abs(vd) / 1.0, 0, 1)
    return rw_s * 0.30 + vr_s * 0.25 + oip_s * 0.25 + vd_s * 0.20

df["squeeze_quality"] = compute_quality_vec(
    df["range_width"].values, df["vol_ratio"].values,
    df["oi_proxy"].values, df["vwap_dist"].values)

MIN_BARS = 500
HOLD = 32  # 4h
HOLD_24H = 96

# ── Build all candidate signals with full feature set ──
print("Building candidate signal matrix...")
candidates = []
for idx in range(MIN_BARS, len(df)):
    ap = float(atr_pctl.iloc[idx]) if not pd.isna(atr_pctl.iloc[idx]) else 0.5
    ls_z = float(df["ls_zscore"].iloc[idx]) if not pd.isna(df["ls_zscore"].iloc[idx]) else 0
    vt = float(df["vol_trend"].iloc[idx]) if not pd.isna(df["vol_trend"].iloc[idx]) else 1.0
    sq = float(df["squeeze_quality"].iloc[idx]) if not pd.isna(df["squeeze_quality"].iloc[idx]) else 0.5
    bvs = float(df["bar_vol_spike"].iloc[idx]) if not pd.isna(df["bar_vol_spike"].iloc[idx]) else 1.0
    bre = float(df["bar_range_expansion"].iloc[idx]) if not pd.isna(df["bar_range_expansion"].iloc[idx]) else 1.0
    bte = bool(df["bar_taker_extreme"].iloc[idx]) if not pd.isna(df["bar_taker_extreme"].iloc[idx]) else False
    m4b_div = df["m4b_div"].iloc[idx]
    m4b_ago = int(df["m4b_ago"].iloc[idx])
    rsi_val = float(df["rsi"].iloc[idx]) if not pd.isna(df["rsi"].iloc[idx]) else 50
    rsi_s = float(df["rsi_slope"].iloc[idx]) if not pd.isna(df["rsi_slope"].iloc[idx]) else 0
    mom4 = float(df["mom_4h"].iloc[idx]) if not pd.isna(df["mom_4h"].iloc[idx]) else 0
    mom1 = float(df["mom_1h"].iloc[idx]) if not pd.isna(df["mom_1h"].iloc[idx]) else 0
    oip = float(df["oi_proxy"].iloc[idx]) if not pd.isna(df["oi_proxy"].iloc[idx]) else 1.0
    rw = float(df["range_width"].iloc[idx]) if not pd.isna(df["range_width"].iloc[idx]) else 5
    vd = abs(float(df["vwap_dist"].iloc[idx])) if not pd.isna(df["vwap_dist"].iloc[idx]) else 0
    bbp = float(bb_width_pctl.iloc[idx]) if not pd.isna(bb_width_pctl.iloc[idx]) else 0.5
    bbpos = float(df["bb_pos"].iloc[idx]) if not pd.isna(df["bb_pos"].iloc[idx]) else 0.5
    price = float(df["Close"].iloc[idx])
    atr_val = float(df["atr"].iloc[idx]) if not pd.isna(df["atr"].iloc[idx]) else 0

    # Compute direction from z-score
    if ls_z <= -1.8:
        direction = "LONG"
        sq_type = "SHORT_SQUEEZE"
    elif ls_z >= 1.8:
        direction = "SHORT"
        sq_type = "LONG_SQUEEZE"
    else:
        continue

    # Returns
    if idx + HOLD < len(df):
        ex4 = float(df["Close"].iloc[idx + HOLD])
        ret4 = (ex4 - price) / price * 100 if direction == "LONG" else (price - ex4) / price * 100
    else:
        ret4 = None
    if idx + HOLD_24H < len(df):
        ex24 = float(df["Close"].iloc[idx + HOLD_24H])
        ret24 = (ex24 - price) / price * 100 if direction == "LONG" else (price - ex24) / price * 100
    else:
        ret24 = None

    # Ignition score
    ignition = 0
    if bvs >= 1.5: ignition += 0.40
    if bre >= 1.3: ignition += 0.30
    if bte: ignition += 0.30
    score = sq * 0.60 + ignition * 0.40

    # M4b agreement
    m4b_agrees = (direction == "LONG" and m4b_div == "BEARISH") or \
                 (direction == "SHORT" and m4b_div == "BULLISH")

    candidates.append({
        "idx": idx, "time": str(df["Open time"].iloc[idx]), "price": price,
        "type": sq_type, "direction": direction, "score": score,
        "quality": sq, "ignition": ignition,
        "ls_z": ls_z, "vol_trend": vt, "atr_pctl": ap,
        "m4b_div": m4b_div, "m4b_ago": m4b_ago, "m4b_agrees": m4b_agrees,
        "rsi": rsi_val, "rsi_slope": rsi_s,
        "mom_4h": mom4, "mom_1h": mom1,
        "oi_proxy": oip, "range_width": rw, "vwap_dist_abs": vd,
        "bb_pctl": bbp, "bb_pos": bbpos,
        "bar_vol_spike": bvs, "bar_range_expansion": bre, "bar_taker_extreme": bte,
        "ret_4h": ret4, "ret_24h": ret24,
    })

df_c = pd.DataFrame(candidates).dropna(subset=["ret_4h"])
print(f"Candidate signals: {len(df_c)}")

# ── Parameter grid (focused on promising ranges) ──
atr_pctl_maxs = [0.30, 0.35, 0.40]
z_mins = [1.8, 2.0, 2.5]
vol_mins = [1.0, 1.2, 1.5]
quality_mins = [0.65, 0.75, 0.85]
score_mins = [0.55, 0.65, 0.75, 0.85]
m4b_filters = ["any", "agree_only", "agree_or_none"]
m4b_max_ago = [6, 12, 24, 99]
rsi_filters = ["any", "not_extreme"]
bb_maxs = [0.40, 1.0]
mom_filters = ["any", "against"]
cooldowns = [0, 16]

MIN_N = 5  # minimum signals for statistical relevance

print("\nRunning parameter sweep...")
results = []

# Pre-filter by regime (always exclude CRISIS/CHOP_HARD)
valid_mask = df_c["atr_pctl"] < 0.50  # exclude high-vol regimes

for atr_max, z_min, vol_min, qual_min, score_min in itertools.product(
    atr_pctl_maxs, z_mins, vol_mins, quality_mins, score_mins
):
    mask = valid_mask.copy()
    mask &= df_c["atr_pctl"] < atr_max
    mask &= df_c["ls_z"].abs() >= z_min
    mask &= df_c["vol_trend"] >= vol_min
    mask &= df_c["quality"] >= qual_min
    mask &= df_c["score"] >= score_min

    for m4b_f in m4b_filters:
        if m4b_f == "agree_only":
            m4b_mask = df_c["m4b_agrees"]
        elif m4b_f == "agree_or_none":
            m4b_mask = df_c["m4b_agrees"] | (df_c["m4b_div"] == "NONE")
        else:
            m4b_mask = pd.Series(True, index=df_c.index)

        for m4b_ago_max in m4b_max_ago:
            ago_mask = df_c["m4b_ago"] <= m4b_ago_max

            for rsi_f in rsi_filters:
                if rsi_f == "not_extreme":
                    rsi_mask = (df_c["rsi"] > 25) & (df_c["rsi"] < 75)
                elif rsi_f == "slope_down":
                    rsi_mask = df_c["rsi_slope"] < 0
                else:
                    rsi_mask = pd.Series(True, index=df_c.index)

                for bb_max in bb_maxs:
                    bb_mask = df_c["bb_pctl"] < bb_max

                    for mom_f in mom_filters:
                        if mom_f == "against":
                            mom_mask = ((df_c["direction"] == "LONG") & (df_c["mom_4h"] < 0)) | \
                                       ((df_c["direction"] == "SHORT") & (df_c["mom_4h"] > 0))
                        elif mom_f == "with":
                            mom_mask = ((df_c["direction"] == "LONG") & (df_c["mom_4h"] > 0)) | \
                                       ((df_c["direction"] == "SHORT") & (df_c["mom_4h"] < 0))
                        else:
                            mom_mask = pd.Series(True, index=df_c.index)

                        final = mask & m4b_mask & ago_mask & rsi_mask & bb_mask & mom_mask
                        sub = df_c[final]

                        if len(sub) < MIN_N:
                            continue

                        # Apply cooldown (check consecutive signals)
                        for cd in cooldowns:
                            if cd > 0:
                                # Filter by cooldown
                                keep = []
                                last_bar = -999
                                for _, row in sub.iterrows():
                                    if row["idx"] - last_bar >= cd:
                                        keep.append(row)
                                        last_bar = row["idx"]
                                sub_cd = pd.DataFrame(keep)
                            else:
                                sub_cd = sub

                            if len(sub_cd) < MIN_N:
                                continue

                            wr4 = (sub_cd["ret_4h"] > 0).mean() * 100
                            avg4 = sub_cd["ret_4h"].mean()
                            wr24 = (sub_cd["ret_24h"] > 0).mean() * 100 if "ret_24h" in sub_cd else 0
                            avg24 = sub_cd["ret_24h"].mean() if "ret_24h" in sub_cd else 0

                            # Score: balance WR, avg return, and sample size
                            # Penalize very low n
                            n_factor = min(len(sub_cd) / 20, 1.0)
                            combined = (wr4 * 0.4 + wr24 * 0.3 + avg4 * 20 + avg24 * 10) * n_factor

                            results.append({
                                "atr_max": atr_max, "z_min": z_min, "vol_min": vol_min,
                                "qual_min": qual_min, "score_min": score_min,
                                "m4b_filter": m4b_f, "m4b_ago_max": m4b_ago_max,
                                "rsi_filter": rsi_f, "bb_max": bb_max, "mom_filter": mom_f,
                                "cooldown": cd,
                                "n": len(sub_cd), "wr4": wr4, "avg4": avg4,
                                "wr24": wr24, "avg24": avg24,
                                "combined": combined,
                            })

# ── Sort and display ──
df_r = pd.DataFrame(results)
if len(df_r) == 0:
    print("No valid configurations found!")
    sys.exit(1)

# Filter for 75%+ WR
high_wr = df_r[(df_r["wr4"] >= 75) & (df_r["n"] >= MIN_N)].sort_values("combined", ascending=False)
high_wr_24 = df_r[(df_r["wr24"] >= 75) & (df_r["n"] >= MIN_N)].sort_values("combined", ascending=False)

print(f"\nTotal configurations tested: {len(df_r)}")
print(f"Configurations with WR4h ≥ 75%: {len(high_wr)}")
print(f"Configurations with WR24h ≥ 75%: {len(high_wr_24)}")

if len(high_wr) > 0:
    print(f"\n{'='*100}")
    print(f"  TOP 20 CONFIGURATIONS — WR4h ≥ 75%")
    print(f"{'='*100}")
    print(f"  {'n':>3} {'WR4h':>6} {'Avg4h':>7} {'WR24h':>6} {'Avg24h':>7} {'Score':>6}  "
          f"{'ATR<':>4} {'|z|≥':>4} {'Vol≥':>4} {'Q≥':>4} {'Sc≥':>4} {'M4b':>10} {'M4bAgo':>6} "
          f"{'RSI':>10} {'BB<':>4} {'Mom':>7} {'CD':>3}")
    print(f"  {'-'*95}")
    for _, r in high_wr.head(20).iterrows():
        print(f"  {r['n']:>3} {r['wr4']:>5.1f}% {r['avg4']:>+6.2f}% {r['wr24']:>5.1f}% "
              f"{r['avg24']:>+6.2f}% {r['combined']:>5.1f}  "
              f"{r['atr_max']:>4.2f} {r['z_min']:>4.1f} {r['vol_min']:>4.1f} "
              f"{r['qual_min']:>4.2f} {r['score_min']:>4.2f} {r['m4b_filter']:>10} "
              f"{r['m4b_ago_max']:>6} {r['rsi_filter']:>10} {r['bb_max']:>4.2f} "
              f"{r['mom_filter']:>7} {r['cooldown']:>3}")

    # Best config detail
    best = high_wr.iloc[0]
    print(f"\n  🏆 BEST CONFIG (highest combined score with WR4h ≥ 75%):")
    print(f"     ATR pctl < {best['atr_max']}")
    print(f"     |z| ≥ {best['z_min']}")
    print(f"     Vol trend ≥ {best['vol_min']}x")
    print(f"     Quality ≥ {best['qual_min']}")
    print(f"     Score ≥ {best['score_min']}")
    print(f"     M4b filter: {best['m4b_filter']}")
    print(f"     M4b recency: ≤ {best['m4b_ago_max']} bars")
    print(f"     RSI filter: {best['rsi_filter']}")
    print(f"     BB pctl < {best['bb_max']}")
    print(f"     Momentum: {best['mom_filter']}")
    print(f"     Cooldown: {best['cooldown']} bars")
    print(f"     → {best['n']} signals, WR4h={best['wr4']:.1f}%, Avg4h={best['avg4']:+.2f}%, "
          f"WR24h={best['wr24']:.1f}%, Avg24h={best['avg24']:+.2f}%")

else:
    print("\n  No configurations hit 75% WR4h. Showing best available:")
    top = df_r.sort_values("combined", ascending=False).head(20)
    print(f"\n  {'n':>3} {'WR4h':>6} {'Avg4h':>7} {'WR24h':>6} {'Avg24h':>7} {'Score':>6}  "
          f"{'ATR<':>4} {'|z|≥':>4} {'Vol≥':>4} {'Q≥':>4} {'Sc≥':>4} {'M4b':>10} {'M4bAgo':>6} "
          f"{'RSI':>10} {'BB<':>4} {'Mom':>7} {'CD':>3}")
    print(f"  {'-'*95}")
    for _, r in top.iterrows():
        print(f"  {r['n']:>3} {r['wr4']:>5.1f}% {r['avg4']:>+6.2f}% {r['wr24']:>5.1f}% "
              f"{r['avg24']:>+6.2f}% {r['combined']:>5.1f}  "
              f"{r['atr_max']:>4.2f} {r['z_min']:>4.1f} {r['vol_min']:>4.1f} "
              f"{r['qual_min']:>4.2f} {r['score_min']:>4.2f} {r['m4b_filter']:>10} "
              f"{r['m4b_ago_max']:>6} {r['rsi_filter']:>10} {r['bb_max']:>4.2f} "
              f"{r['mom_filter']:>7} {r['cooldown']:>3}")

# ── Also test: what if we combine M4b agreement with directional conviction ──
print(f"\n{'='*80}")
print(f"  DEEP DIVE: M4b Agreement + Directional Conviction")
print(f"{'='*80}")

# Test combinations of M4b agreement + various secondary filters
conviction_tests = [
    ("baseline (all)", pd.Series(True, index=df_c.index)),
    ("M4b agrees", df_c["m4b_agrees"]),
    ("M4b agrees + score≥0.70", df_c["m4b_agrees"] & (df_c["score"] >= 0.70)),
    ("M4b agrees + score≥0.75", df_c["m4b_agrees"] & (df_c["score"] >= 0.75)),
    ("M4b agrees + score≥0.80", df_c["m4b_agrees"] & (df_c["score"] >= 0.80)),
    ("M4b agrees + quality≥0.80", df_c["m4b_agrees"] & (df_c["quality"] >= 0.80)),
    ("M4b agrees + quality≥0.85", df_c["m4b_agrees"] & (df_c["quality"] >= 0.85)),
    ("M4b agrees + |z|≥2.0", df_c["m4b_agrees"] & (df_c["ls_z"].abs() >= 2.0)),
    ("M4b agrees + |z|≥2.5", df_c["m4b_agrees"] & (df_c["ls_z"].abs() >= 2.5)),
    ("M4b agrees + vol≥1.3", df_c["m4b_agrees"] & (df_c["vol_trend"] >= 1.3)),
    ("M4b agrees + vol≥1.5", df_c["m4b_agrees"] & (df_c["vol_trend"] >= 1.5)),
    ("M4b agrees + RSI slope<0", df_c["m4b_agrees"] & (df_c["rsi_slope"] < 0)),
    ("M4b agrees + mom against", df_c["m4b_agrees"] & (
        ((df_c["direction"] == "LONG") & (df_c["mom_4h"] < 0)) |
        ((df_c["direction"] == "SHORT") & (df_c["mom_4h"] > 0))
    )),
    ("M4b agrees + BB pctl<0.3", df_c["m4b_agrees"] & (df_c["bb_pctl"] < 0.30)),
    ("M4b agrees + BB pctl<0.4", df_c["m4b_agrees"] & (df_c["bb_pctl"] < 0.40)),
    ("M4b agrees + ago≤6", df_c["m4b_agrees"] & (df_c["m4b_ago"] <= 6)),
    ("M4b agrees + ago≤12", df_c["m4b_agrees"] & (df_c["m4b_ago"] <= 12)),
    ("M4b agrees + ago≤6 + score≥0.70", df_c["m4b_agrees"] & (df_c["m4b_ago"] <= 6) & (df_c["score"] >= 0.70)),
    ("M4b agrees + ago≤6 + quality≥0.80", df_c["m4b_agrees"] & (df_c["m4b_ago"] <= 6) & (df_c["quality"] >= 0.80)),
    ("M4b agrees + ago≤12 + |z|≥2.0", df_c["m4b_agrees"] & (df_c["m4b_ago"] <= 12) & (df_c["ls_z"].abs() >= 2.0)),
    ("M4b agrees + ago≤12 + vol≥1.3", df_c["m4b_agrees"] & (df_c["m4b_ago"] <= 12) & (df_c["vol_trend"] >= 1.3)),
    ("M4b agrees + score≥0.70 + |z|≥2.0", df_c["m4b_agrees"] & (df_c["score"] >= 0.70) & (df_c["ls_z"].abs() >= 2.0)),
    ("M4b agrees + score≥0.70 + vol≥1.3", df_c["m4b_agrees"] & (df_c["score"] >= 0.70) & (df_c["vol_trend"] >= 1.3)),
    ("M4b agrees + quality≥0.80 + |z|≥2.0", df_c["m4b_agrees"] & (df_c["quality"] >= 0.80) & (df_c["ls_z"].abs() >= 2.0)),
    ("M4b agrees + quality≥0.80 + ago≤12", df_c["m4b_agrees"] & (df_c["quality"] >= 0.80) & (df_c["m4b_ago"] <= 12)),
    ("M4b agrees + ago≤6 + |z|≥2.0 + vol≥1.2", df_c["m4b_agrees"] & (df_c["m4b_ago"] <= 6) & (df_c["ls_z"].abs() >= 2.0) & (df_c["vol_trend"] >= 1.2)),
    ("M4b agrees + ago≤6 + score≥0.70 + |z|≥2.0", df_c["m4b_agrees"] & (df_c["m4b_ago"] <= 6) & (df_c["score"] >= 0.70) & (df_c["ls_z"].abs() >= 2.0)),
    ("M4b agrees + ago≤6 + quality≥0.80 + vol≥1.2", df_c["m4b_agrees"] & (df_c["m4b_ago"] <= 6) & (df_c["quality"] >= 0.80) & (df_c["vol_trend"] >= 1.2)),
    ("M4b agrees + ago≤12 + score≥0.70 + |z|≥2.0 + vol≥1.2", df_c["m4b_agrees"] & (df_c["m4b_ago"] <= 12) & (df_c["score"] >= 0.70) & (df_c["ls_z"].abs() >= 2.0) & (df_c["vol_trend"] >= 1.2)),
]

print(f"\n  {'Filter':<58} {'n':>3} {'WR4h':>6} {'Avg4h':>7} {'WR24h':>6} {'Avg24h':>7}")
print(f"  {'-'*90}")
for name, mask in conviction_tests:
    sub = df_c[mask].dropna(subset=["ret_4h"])
    if len(sub) == 0:
        print(f"  {name:<58} {'0':>3} {'—':>6} {'—':>7} {'—':>6} {'—':>7}")
        continue
    wr4 = (sub["ret_4h"] > 0).mean() * 100
    a4 = sub["ret_4h"].mean()
    wr24 = (sub["ret_24h"] > 0).mean() * 100
    a24 = sub["ret_24h"].mean()
    marker = " ✅" if wr4 >= 75 and len(sub) >= 5 else " 🟡" if wr4 >= 70 and len(sub) >= 5 else ""
    print(f"  {name:<58} {len(sub):>3} {wr4:>5.1f}% {a4:>+6.2f}% {wr24:>5.1f}% {a24:>+6.2f}%{marker}")

# ── Temporal stability check for top combos ──
print(f"\n{'='*80}")
print(f"  TEMPORAL STABILITY (top 75%+ combos)")
print(f"{'='*80}")

# Find all combos that hit 75%+
for name, mask in conviction_tests:
    sub = df_c[mask].dropna(subset=["ret_4h"])
    if len(sub) < 5:
        continue
    wr4 = (sub["ret_4h"] > 0).mean() * 100
    if wr4 < 70:
        continue

    sub_m = sub.copy()
    sub_m["month"] = pd.to_datetime(sub_m["time"]).dt.to_period("M")
    monthly = sub_m.groupby("month").agg(
        n=("ret_4h", "count"), wr4=("ret_4h", lambda x: (x > 0).mean() * 100),
        avg4=("ret_4h", "mean"), wr24=("ret_24h", lambda x: (x > 0).mean() * 100),
        avg24=("ret_24h", "mean"),
    )

    print(f"\n  {name} (overall WR4h={wr4:.1f}%, n={len(sub)}):")
    print(f"    {'Month':>8}  {'n':>3}  {'WR4h':>6}  {'Avg4h':>7}  {'WR24h':>6}  {'Avg24h':>7}")
    for month, row in monthly.iterrows():
        print(f"    {str(month):>8}  {int(row['n']):>3}  {row['wr4']:>5.1f}%  "
              f"{row['avg4']:>+6.2f}%  {row['wr24']:>5.1f}%  {row['avg24']:>+6.2f}%")

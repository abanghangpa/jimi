#!/usr/bin/env python3
"""
M18 Squeeze v5 — Parameter sweep + module filter optimization.
Relax detection → more signals → filter to 75%+ WR.
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import numpy as np
import pandas as pd
from itertools import combinations
from src.config import CONFIG
from src.utils.data_handler import load_data, resample_ohlcv
from src.utils.indicators import calc_atr, calc_rsi, calc_vol_ratio, calc_vwap, calc_ema
from src.modules.m4_cvd import calc_cvd_15m, detect_cvd_divergence_15m
from src.modules.m18_squeeze import detect_squeeze_v5 as detect_squeeze, SQUEEZE_V5_DEFAULTS

CSV = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "eth_15m_merged.csv")
if not os.path.exists(CSV):
    CSV = os.path.join(os.path.dirname(os.path.dirname(__file__)), "eth_15m_merged.csv")

print("=" * 80)
print("  M18 SQUEEZE v5 — PARAMETER SWEEP + FILTER OPTIMIZATION")
print("=" * 80)

df = load_data(CSV)
df = df[df["Open time"] >= "2026-01-01"].reset_index(drop=True)
print(f"  Bars: {len(df)}  ({df['Open time'].iloc[0]} → {df['Open time'].iloc[-1]})")

# ── Pre-compute all indicators once ──
print("  Computing indicators...")
df["atr"] = calc_atr(df["High"], df["Low"], df["Close"], 14)
df["rsi"] = calc_rsi(df["Close"], 14)
df["vol_ratio"] = calc_vol_ratio(df["Volume"])
df["vwap"] = calc_vwap(df["High"], df["Low"], df["Close"], df["Volume"], 20)
df["taker_ratio"] = (df["Taker buy base asset volume"] / df["Volume"].replace(0, np.nan)).fillna(0.5)
df["vol_ma20"] = df["Volume"].rolling(20).mean()
df["vol_trend"] = df["Volume"] / df["vol_ma20"]
df["ema21"] = calc_ema(df["Close"], 21)
df["ema55"] = calc_ema(df["Close"], 55)
df["ema_trend"] = np.where(df["ema21"] > df["ema55"], "BULL", "BEAR")
df["cvd_15m"] = calc_cvd_15m(df)
df["cvd_divergence_15m"] = detect_cvd_divergence_15m(df, 20, 8)

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

df["range_width"] = (df["High"].rolling(48).max() - df["Low"].rolling(48).min()) / df["Close"] * 100
df["bar_range"] = (df["High"] - df["Low"]) / df["Close"] * 100

df["m4b_div"] = "NONE"
df["m4b_ago"] = 99
for idx in range(24, len(df)):
    for ci in range(max(0, idx - 24), idx + 1):
        div = df["cvd_divergence_15m"].iloc[ci]
        if div != "NONE":
            df.loc[idx, "m4b_div"] = div
            df.loc[idx, "m4b_ago"] = idx - ci
            break

# ── Build compression history for each bar ──
print("  Building compression histories...")
comp_hist = {}
for idx in range(48, len(df)):
    ch = []
    for i in range(max(0, idx - 47), idx):
        r48_h = float(df["High"].iloc[max(0, i-47):i+1].max())
        r48_l = float(df["Low"].iloc[max(0, i-47):i+1].min())
        r48_pct = (r48_h - r48_l) / float(df["Close"].iloc[i]) * 100 if float(df["Close"].iloc[i]) > 0 else 5.0
        vr = float(df["Volume"].iloc[i] / df["vol_ma20"].iloc[i]) if float(df["vol_ma20"].iloc[i]) > 0 else 1.0
        br = float(df["bar_range"].iloc[i]) if not pd.isna(df["bar_range"].iloc[i]) else 0.5
        tr = float(df["taker_ratio"].iloc[i])
        ch.append((r48_pct, vr, br, tr))
    comp_hist[idx] = ch

# ── Parameter grid ──
param_sets = [
    {"name": "default",   "R48_MAX": 1.2, "COIL_DELTA": 0.05, "COIL_MIN": 6,  "COMP_MIN": 12, "DRY_MIN": 4,  "DOJI_MIN": 8,  "REQUIRE_FLIP": True,  "EMA_FILTER": True},
    {"name": "loose_A",   "R48_MAX": 1.8, "COIL_DELTA": 0.08, "COIL_MIN": 4,  "COMP_MIN": 8,  "DRY_MIN": 3,  "DOJI_MIN": 6,  "REQUIRE_FLIP": True,  "EMA_FILTER": True},
    {"name": "loose_B",   "R48_MAX": 2.0, "COIL_DELTA": 0.10, "COIL_MIN": 4,  "COMP_MIN": 8,  "DRY_MIN": 3,  "DOJI_MIN": 6,  "REQUIRE_FLIP": False, "EMA_FILTER": True},
    {"name": "loose_C",   "R48_MAX": 2.5, "COIL_DELTA": 0.12, "COIL_MIN": 3,  "COMP_MIN": 6,  "DRY_MIN": 2,  "DOJI_MIN": 4,  "REQUIRE_FLIP": False, "EMA_FILTER": False},
    {"name": "wide_A",    "R48_MAX": 3.0, "COIL_DELTA": 0.15, "COIL_MIN": 3,  "COMP_MIN": 6,  "DRY_MIN": 2,  "DOJI_MIN": 4,  "REQUIRE_FLIP": False, "EMA_FILTER": False},
    {"name": "flip_only", "R48_MAX": 2.0, "COIL_DELTA": 0.10, "COIL_MIN": 4,  "COMP_MIN": 8,  "DRY_MIN": 3,  "DOJI_MIN": 6,  "REQUIRE_FLIP": True,  "EMA_FILTER": False},
    {"name": "ema_only",  "R48_MAX": 2.0, "COIL_DELTA": 0.10, "COIL_MIN": 4,  "COMP_MIN": 8,  "DRY_MIN": 3,  "DOJI_MIN": 6,  "REQUIRE_FLIP": False, "EMA_FILTER": True},
]

HOLD_PERIODS = {"1h": 4, "2h": 8, "4h": 32, "8h": 48, "12h": 64, "24h": 96}
EXPIRY_BARS = 32
MIN_BARS = 500

filter_names = ["ema_aligned", "cvd_agrees", "whale_agrees", "ls_agrees",
                "vol_confirms", "rsi_ok", "quality_high", "dual_path"]

all_results = []

for ps in param_sets:
    cfg = dict(CONFIG)
    cfg.update(SQUEEZE_V5_DEFAULTS)
    cfg["SQUEEZE_RANGE48_MAX"] = ps["R48_MAX"]
    cfg["SQUEEZE_COIL_DELTA_MAX"] = ps["COIL_DELTA"]
    cfg["SQUEEZE_COIL_BARS_MIN"] = ps["COIL_MIN"]
    cfg["SQUEEZE_COMPRESSION_BARS_MIN"] = ps["COMP_MIN"]
    cfg["SQUEEZE_DRY_BARS_MIN"] = ps["DRY_MIN"]
    cfg["SQUEEZE_DOJI_BARS_MIN"] = ps["DOJI_MIN"]
    cfg["SQUEEZE_REQUIRE_HIST_FLIP"] = ps["REQUIRE_FLIP"]
    cfg["SQUEEZE_EMA_FILTER"] = ps["EMA_FILTER"]

    # Detect setups
    setups = []
    last_detect = -999
    for idx in range(MIN_BARS, len(df)):
        price = float(df["Close"].iloc[idx])
        raw_vol = float(df["Volume"].iloc[idx])
        ap = float(atr_pctl.iloc[idx]) if not pd.isna(atr_pctl.iloc[idx]) else 0.5
        regime = ("CHOP_HARD" if ap < 0.20 else "NEUTRAL_CHOP" if ap < 0.30
                  else "CHOP_MILD" if ap < 0.50 else "NEUTRAL" if ap < 0.70
                  else "TRENDING" if ap <= 0.80 else "CRISIS")
        ls_z = float(df["ls_zscore"].iloc[idx]) if not pd.isna(df["ls_zscore"].iloc[idx]) else 0
        rsi_val = float(df["rsi"].iloc[idx]) if not pd.isna(df["rsi"].iloc[idx]) else 50
        vt = float(df["vol_trend"].iloc[idx]) if not pd.isna(df["vol_trend"].iloc[idx]) else 1.0
        atr_val = float(df["atr"].iloc[idx]) if not pd.isna(df["atr"].iloc[idx]) else 0
        whale = df["whale"].iloc[idx]

        result = {
            "price": price, "m9": {"regime": regime, "raw": ap},
            "derivatives": {"ls_zscore": ls_z, "funding_rate": (rsi_val-50)/50*0.001,
                            "oi_roc_1h": float(df["oi_roc_sim"].iloc[idx]) if not pd.isna(df["oi_roc_sim"].iloc[idx]) else 0,
                            "whale_signal": whale},
            "m4b": {"divergence": df["m4b_div"].iloc[idx], "bars_ago": int(df["m4b_ago"].iloc[idx]), "cvd_slope": 0},
            "rsi": rsi_val, "vol_trend": vt, "atr": atr_val,
            "range_width": float(df["range_width"].iloc[idx]) if not pd.isna(df["range_width"].iloc[idx]) else 5,
            "vol_ratio": float(df["vol_ratio"].iloc[idx]) if not pd.isna(df["vol_ratio"].iloc[idx]) else 0.15,
            "raw_taker_ratio": float(df["taker_ratio"].iloc[idx]),
            "raw_bar_range_pct": float(df["bar_range"].iloc[idx]),
            "vol_ma20": raw_vol,
        }

        sq = detect_squeeze(result, config=cfg, last_signal_bar=last_detect,
                           current_bar=idx, compression_history=comp_hist.get(idx, []),
                           df_15m=df.iloc[:idx+1])

        if sq["squeeze_type"] != "NONE":
            last_detect = idx
            setups.append({
                "idx": idx, "direction": sq["direction"],
                "coil_high": sq["coil_high"], "coil_low": sq["coil_low"],
                "score": sq["squeeze_score"], "strong": sq["squeeze_strong"],
                "quality": sq["squeeze_score"],
                "pa_fired": sq["path_a"]["fired"], "pb_fired": sq["path_b"]["fired"],
                "regime": regime, "ls_z": ls_z, "whale": whale,
                "m4b_div": df["m4b_div"].iloc[idx], "rsi": rsi_val,
                "ema_trend": df["ema_trend"].iloc[idx],
            })

    # Forward-track breakouts
    signals = []
    for s in setups:
        det_idx = s["idx"]
        direction = s["direction"]
        ch = s["coil_high"]
        cl = s["coil_low"]
        buf = 0.001

        entry_idx = None
        entry_price = None
        for fwd in range(det_idx + 1, min(det_idx + EXPIRY_BARS + 1, len(df))):
            close = float(df["Close"].iloc[fwd])
            vol = float(df["Volume"].iloc[fwd])
            vol_ma = float(df["vol_ma20"].iloc[fwd]) if not pd.isna(df["vol_ma20"].iloc[fwd]) else vol
            vol_ok = vol >= vol_ma * 1.0

            if direction == "LONG" and close > ch * (1 + buf) and vol_ok:
                entry_idx = fwd; entry_price = close; break
            elif direction == "SHORT" and close < cl * (1 - buf) and vol_ok:
                entry_idx = fwd; entry_price = close; break

        if entry_idx is None:
            continue

        # Confirmations at entry
        ema_t = df["ema_trend"].iloc[entry_idx]
        ema_aligned = (direction == "LONG" and ema_t == "BULL") or (direction == "SHORT" and ema_t == "BEAR")
        m4b = df["m4b_div"].iloc[entry_idx]
        cvd_agrees = not ((direction == "LONG" and m4b == "BEARISH") or (direction == "SHORT" and m4b == "BULLISH"))
        lz = float(df["ls_zscore"].iloc[entry_idx]) if not pd.isna(df["ls_zscore"].iloc[entry_idx]) else 0
        ls_agrees = not ((direction == "LONG" and lz > 1.5) or (direction == "SHORT" and lz < -1.5))
        wh = df["whale"].iloc[entry_idx]
        whale_agrees = not ((direction == "LONG" and wh == "WHALE_BEARISH") or (direction == "SHORT" and wh == "WHALE_BULLISH"))
        vt_e = float(df["vol_trend"].iloc[entry_idx]) if not pd.isna(df["vol_trend"].iloc[entry_idx]) else 1.0
        vol_confirms = vt_e > 0.8
        rsi_e = float(df["rsi"].iloc[entry_idx]) if not pd.isna(df["rsi"].iloc[entry_idx]) else 50
        rsi_ok = (direction == "LONG" and rsi_e < 75) or (direction == "SHORT" and rsi_e > 25)
        q = s["quality"]
        quality_high = q >= 0.5
        dual_path = s["pa_fired"] and s["pb_fired"]
        is_strong = dual_path or q >= 0.7
        n_confirm = sum([ema_aligned, cvd_agrees, whale_agrees, ls_agrees, vol_confirms, rsi_ok, quality_high, dual_path])

        returns = {}
        for label, hold in HOLD_PERIODS.items():
            ei = entry_idx + hold
            if ei < len(df):
                ex = float(df["Close"].iloc[ei])
                ret = (ex - entry_price) / entry_price * 100 if direction == "LONG" else (entry_price - ex) / entry_price * 100
                returns[f"ret_{label}"] = round(ret, 3)

        signals.append({
            "direction": direction, "strong": is_strong, "quality": round(q, 3),
            "ema_aligned": ema_aligned, "cvd_agrees": cvd_agrees,
            "whale_agrees": whale_agrees, "ls_agrees": ls_agrees,
            "vol_confirms": vol_confirms, "rsi_ok": rsi_ok,
            "quality_high": quality_high, "dual_path": dual_path,
            "n_confirm": n_confirm, **returns,
        })

    df_s = pd.DataFrame(signals)

    if len(df_s) == 0:
        all_results.append({"name": ps["name"], "n_setups": len(setups), "n_entries": 0,
                            "best_wr_4h": 0, "best_n_4h": 0, "best_combo": None})
        continue

    # Find best filter combo for 4h WR
    best_wr = 0
    best_combo = None
    best_n = 0

    for r in range(0, len(filter_names) + 1):
        for combo in combinations(filter_names, r):
            sub = df_s.copy()
            for f in combo:
                sub = sub[sub[f] == True]
            if len(sub) >= 3 and "ret_4h" in sub.columns:
                wr = (sub["ret_4h"] > 0).mean() * 100
                if wr > best_wr or (wr == best_wr and len(sub) > best_n):
                    best_wr = wr
                    best_combo = combo
                    best_n = len(sub)

    # Also compute base WR
    base_wr_4h = (df_s["ret_4h"] > 0).mean() * 100 if "ret_4h" in df_s.columns and len(df_s) > 0 else 0

    all_results.append({
        "name": ps["name"],
        "n_setups": len(setups),
        "n_entries": len(df_s),
        "base_wr_4h": base_wr_4h,
        "best_wr_4h": best_wr,
        "best_n_4h": best_n,
        "best_combo": best_combo,
        "df": df_s,
    })

# ── Summary ──
print(f"\n{'=' * 80}")
print("  PARAMETER SWEEP RESULTS")
print(f"{'=' * 80}")
print(f"  {'Config':<14}  {'Setups':>6}  {'Entries':>7}  {'Base WR':>8}  {'Best WR':>8}  {'Best n':>6}  {'Filters'}")
print(f"  {'-' * 80}")

for r in all_results:
    combo_str = str(list(r["best_combo"])) if r["best_combo"] else "none"
    print(f"  {r['name']:<14}  {r['n_setups']:>6}  {r['n_entries']:>7}  "
          f"{r.get('base_wr_4h', 0):>7.1f}%  {r['best_wr_4h']:>7.1f}%  {r['best_n_4h']:>6}  {combo_str}")

# Find the best overall config (prefer 75%+ WR with most signals)
best_overall = None
for r in all_results:
    if r["best_wr_4h"] >= 75 and r["best_n_4h"] >= 3:
        if best_overall is None or r["best_n_4h"] > best_overall["best_n_4h"]:
            best_overall = r

if not best_overall:
    # Fall back to highest WR with n>=3
    for r in sorted(all_results, key=lambda x: -x["best_wr_4h"]):
        if r["best_n_4h"] >= 3:
            best_overall = r
            break

if best_overall and best_overall.get("df") is not None:
    df_best = best_overall["df"]
    combo = best_overall["best_combo"]

    print(f"\n{'=' * 80}")
    print(f"  BEST CONFIG: {best_overall['name']}")
    print(f"  Entries: {best_overall['n_entries']}  |  4h WR: {best_overall['best_wr_4h']:.1f}%  |  n={best_overall['best_n_4h']}")
    if combo:
        print(f"  Filters: {list(combo)}")
    print(f"{'=' * 80}")

    # Apply filters
    filtered = df_best.copy()
    if combo:
        for f in combo:
            filtered = filtered[filtered[f] == True]

    print(f"\n  All timeframes (filtered):")
    print(f"  {'Hold':>6}  {'Win%':>6}  {'Avg%':>7}  {'Med%':>7}  {'Max%':>7}  {'Min%':>7}  {'n':>4}")
    print(f"  {'-' * 56}")
    for label in ["1h", "2h", "4h", "8h", "12h", "24h"]:
        col = f"ret_{label}"
        if col in filtered.columns:
            valid = filtered[col].dropna()
            if len(valid) > 0:
                wr = (valid > 0).mean() * 100
                print(f"  {label:>6}  {wr:>5.1f}%  {valid.mean():>+6.2f}%  {valid.median():>+6.2f}%  "
                      f"{valid.max():>+6.2f}%  {valid.min():>+6.2f}%  {len(valid):>4}")

    # Direction split
    for d in ["LONG", "SHORT"]:
        sub = filtered[filtered["direction"] == d]
        if len(sub) > 0 and "ret_4h" in sub.columns:
            wr = (sub["ret_4h"] > 0).mean() * 100
            print(f"  {d}: n={len(sub)}  WR4h={wr:.1f}%  avg={sub['ret_4h'].mean():+.2f}%")

    # Signal log
    print(f"\n  Filtered signals:")
    print(f"  {'Dir':<5} {'Qual':>5} {'Conf':>4} {'EMA':>3} {'CVD':>3} {'Whl':>3} {'LS':>3} {'Vol':>3} {'RSI':>3} {'Qhi':>3} {'Dbl':>3} {'1h':>7} {'2h':>7} {'4h':>7} {'24h':>7}")
    print(f"  {'-' * 85}")
    for _, s in filtered.iterrows():
        h1 = f"{s.get('ret_1h', 0):+.2f}%" if not pd.isna(s.get('ret_1h')) else "-"
        h2 = f"{s.get('ret_2h', 0):+.2f}%" if not pd.isna(s.get('ret_2h')) else "-"
        h4 = f"{s.get('ret_4h', 0):+.2f}%" if not pd.isna(s.get('ret_4h')) else "-"
        h24 = f"{s.get('ret_24h', 0):+.2f}%" if not pd.isna(s.get('ret_24h')) else "-"
        flags = ["✅" if s.get(f) else "❌" for f in ["ema_aligned","cvd_agrees","whale_agrees","ls_agrees","vol_confirms","rsi_ok","quality_high","dual_path"]]
        print(f"  {s['direction']:<5} {s['quality']:>5.2f} {s['n_confirm']:>4} {' '.join(flags)} {h1:>7} {h2:>7} {h4:>7} {h24:>7}")

    # Also show ALL entries for this config (unfiltered)
    print(f"\n  ALL entries (unfiltered, {best_overall['name']}):")
    print(f"  {'Dir':<5} {'Qual':>5} {'Conf':>4} {'1h':>7} {'2h':>7} {'4h':>7} {'24h':>7}")
    print(f"  {'-' * 50}")
    for _, s in df_best.iterrows():
        h1 = f"{s.get('ret_1h', 0):+.2f}%" if not pd.isna(s.get('ret_1h')) else "-"
        h2 = f"{s.get('ret_2h', 0):+.2f}%" if not pd.isna(s.get('ret_2h')) else "-"
        h4 = f"{s.get('ret_4h', 0):+.2f}%" if not pd.isna(s.get('ret_4h')) else "-"
        h24 = f"{s.get('ret_24h', 0):+.2f}%" if not pd.isna(s.get('ret_24h')) else "-"
        st = "*" if s["strong"] else " "
        print(f"  {st}{s['direction']:<4} {s['quality']:>5.2f} {s['n_confirm']:>4} {h1:>7} {h2:>7} {h4:>7} {h24:>7}")

print(f"\n{'=' * 80}")
print("  DONE")
print(f"{'=' * 80}")

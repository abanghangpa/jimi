#!/usr/bin/env python3
"""Backtest M18 v3 squeeze detector on real 2026 data.

Uses synthetic derivatives proxies from taker ratio/volume since
backtest data doesn't have live derivatives. Proxies are calibrated
against live scanner data distributions.
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import numpy as np
import pandas as pd
from src.config import CONFIG
from src.utils.data_handler import load_data
from src.utils.indicators import calc_atr, calc_rsi, calc_vol_ratio, calc_vwap
from src.modules.m4_cvd import calc_cvd_15m, detect_cvd_divergence_15m
from src.modules.m18_squeeze import detect_squeeze_v3, SQUEEZE_V3_DEFAULTS

CSV = os.path.join(os.path.dirname(os.path.dirname(__file__)), "eth_15m_merged.csv")
df = load_data(CSV)
df = df[df["Open time"] >= "2026-01-01"].reset_index(drop=True)

cfg = dict(CONFIG)
cfg.update(SQUEEZE_V3_DEFAULTS)

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

# ── Synthetic derivatives proxies ──
# L/S z-score from taker ratio deviation
taker_ma = df["taker_ratio"].rolling(50).mean()
taker_std = df["taker_ratio"].rolling(50).std()
df["ls_zscore"] = (df["taker_ratio"] - taker_ma) / taker_std.replace(0, 1)

# OI ROC proxy from volume momentum (OI tends to accumulate with volume)
df["oi_roc_sim"] = df["Volume"].pct_change(4) * 100

# Whale proxy: high-volume bars with directional conviction
vol_pctl = df["Volume"].rolling(100).apply(
    lambda x: (x.iloc[-1] - x.min()) / (x.max() - x.min()) if x.max() > x.min() else 0.5, raw=False)
price_change = df["Close"].pct_change(4)
df["whale"] = "NEUTRAL"
df.loc[(vol_pctl > 0.8) & (price_change > 0.005), "whale"] = "WHALE_BULLISH"
df.loc[(vol_pctl > 0.8) & (price_change < -0.005), "whale"] = "WHALE_BEARISH"

# ATR percentile (regime)
atr_pctl = df["atr"].rolling(500, min_periods=100).apply(
    lambda x: (x.iloc[-1] - x.min()) / (x.max() - x.min()) if x.max() > x.min() else 0.5, raw=False)

# ── Bar-level features for squeeze quality ──
df["range_width"] = (df["High"].rolling(48).max() - df["Low"].rolling(48).min()) / df["Close"] * 100
df["vwap_dist"] = (df["Close"] - df["vwap"]) / df["vwap"] * 100
vol_cumsum_48 = df["Volume"].rolling(48).sum()
vol_cumsum_ma = df["Volume"].rolling(68).mean() * 20  # rough MA proxy
df["oi_proxy"] = vol_cumsum_48 / vol_cumsum_ma.replace(0, 1)
df["bar_vol_spike"] = df["Volume"] / df["vol_ma20"]
df["bar_range"] = (df["High"] - df["Low"]) / df["Close"] * 100
bar_range_ma = df["bar_range"].rolling(20).mean()
df["bar_range_expansion"] = df["bar_range"] / bar_range_ma.replace(0, 1)
df["bar_taker_extreme"] = (df["taker_ratio"] > 0.65) | (df["taker_ratio"] < 0.35)

# ── Scan for squeeze signals ──
MIN_BARS = 500
HOLD_PERIODS = {"1h": 4, "2h": 8, "4h": 32, "8h": 48, "12h": 64, "24h": 96}
signals = []
last_signal_bar = -999

for idx in range(MIN_BARS, len(df)):
    ap = float(atr_pctl.iloc[idx]) if not pd.isna(atr_pctl.iloc[idx]) else 0.5
    regime = ("NEUTRAL_CHOP" if ap < 0.30 else "CHOP_HARD" if ap < 0.20
              else "CRISIS" if ap > 0.80 else "CHOP_MILD" if ap < 0.50
              else "TRENDING" if ap >= 0.70 else "NEUTRAL")

    ls_z = float(df["ls_zscore"].iloc[idx]) if not pd.isna(df["ls_zscore"].iloc[idx]) else 0
    oi_roc = float(df["oi_roc_sim"].iloc[idx]) if not pd.isna(df["oi_roc_sim"].iloc[idx]) else 0
    whale = df["whale"].iloc[idx]
    rsi_val = float(df["rsi"].iloc[idx]) if not pd.isna(df["rsi"].iloc[idx]) else 50
    vt = float(df["vol_trend"].iloc[idx]) if not pd.isna(df["vol_trend"].iloc[idx]) else 1.0
    price = float(df["Close"].iloc[idx])
    atr_val = float(df["atr"].iloc[idx]) if not pd.isna(df["atr"].iloc[idx]) else 0
    rw = float(df["range_width"].iloc[idx]) if not pd.isna(df["range_width"].iloc[idx]) else 5
    vr = float(df["vol_ratio"].iloc[idx]) if not pd.isna(df["vol_ratio"].iloc[idx]) else 0.15
    oip = float(df["oi_proxy"].iloc[idx]) if not pd.isna(df["oi_proxy"].iloc[idx]) else 1.0
    vd = float(df["vwap_dist"].iloc[idx]) if not pd.isna(df["vwap_dist"].iloc[idx]) else 0
    bvs = float(df["bar_vol_spike"].iloc[idx]) if not pd.isna(df["bar_vol_spike"].iloc[idx]) else 1.0
    bre = float(df["bar_range_expansion"].iloc[idx]) if not pd.isna(df["bar_range_expansion"].iloc[idx]) else 1.0
    bte = bool(df["bar_taker_extreme"].iloc[idx]) if not pd.isna(df["bar_taker_extreme"].iloc[idx]) else False

    # M4b intrabar CVD divergence
    m4b_div = "NONE"
    m4b_ago = 99
    for ci in range(max(0, idx - 24), idx + 1):
        div = df["cvd_divergence_15m"].iloc[ci]
        if div != "NONE":
            m4b_div = div
            m4b_ago = idx - ci
            break

    # Compute squeeze quality (same formula as scanner)
    rw_score = max(0, min(1, 1 - (rw - 1.5) / 4.0))
    vr_score = max(0, min(1, 1 - (vr - 0.05) / 0.20))
    oip_score = max(0, min(1, (oip - 0.7) / 0.5))
    vd_score = max(0, min(1, 1 - abs(vd) / 1.0))
    squeeze_quality = (rw_score * cfg.get('SQUEEZE_RW_WEIGHT', 0.30) +
                       vr_score * cfg.get('SQUEEZE_VR_WEIGHT', 0.25) +
                       oip_score * cfg.get('SQUEEZE_OIP_WEIGHT', 0.25) +
                       vd_score * cfg.get('SQUEEZE_VD_WEIGHT', 0.20))

    # Build result dict matching scanner format
    result = {
        "price": price,
        "m9": {"regime": regime, "raw": ap},
        "derivatives": {
            "ls_zscore": ls_z,
            "funding_rate": (rsi_val - 50) / 50 * 0.001,  # synthetic
            "oi_roc_1h": oi_roc,
            "whale_signal": whale,
        },
        "m4b": {"divergence": m4b_div, "bars_ago": m4b_ago, "cvd_slope": 0},
        "rsi": rsi_val,
        "vol_trend": vt,
        "atr": atr_val,
        "range_width": rw,
        "vol_ratio": vr,
        "oi_proxy": oip,
        "vwap_dist": vd,
        "squeeze_quality": squeeze_quality,
        "bar_vol_spike": bvs,
        "bar_range_expansion": bre,
        "bar_taker_extreme": bte,
    }

    sq = detect_squeeze_v3(result, config=cfg, last_signal_bar=last_signal_bar, current_bar=idx)

    if sq["squeeze_type"] != "NONE":
        last_signal_bar = idx
        direction = sq["direction"]

        returns = {}
        for label, hold in HOLD_PERIODS.items():
            if idx + hold < len(df):
                ex = float(df["Close"].iloc[idx + hold])
                ret = (ex - price) / price * 100 if direction == "LONG" else (price - ex) / price * 100
                returns[f"ret_{label}"] = round(ret, 3)

        # Check if M4b divergence agrees
        m4b_agrees = (direction == "LONG" and m4b_div == "BEARISH") or \
                     (direction == "SHORT" and m4b_div == "BULLISH")

        signals.append({
            "time": str(df["Open time"].iloc[idx]),
            "price": price,
            "type": sq["squeeze_type"],
            "score": sq["squeeze_score"],
            "strong": sq["squeeze_strong"],
            "direction": direction,
            "quality": sq["quality"],
            "ignition": sq["ignition"],
            "ls_z": round(ls_z, 2),
            "whale": whale,
            "m4b_div": m4b_div,
            "m4b_agrees": m4b_agrees,
            "m4b_ago": m4b_ago,
            "rsi": round(rsi_val, 1),
            "vol_trend": round(vt, 2),
            "regime": regime,
            "atr_pctl": round(ap, 3),
            "range_width": round(rw, 2),
            "oi_proxy": round(oip, 2),
            "squeeze_quality": round(squeeze_quality, 3),
            "gates_passed": len(sq.get("gates_passed", [])),
            **returns,
        })

df_s = pd.DataFrame(signals)

print("=" * 80)
print("  M18 SQUEEZE v3 — 2026 BACKTEST (synthetic derivatives)")
print("=" * 80)
print(f"\n  Period: {df['Open time'].iloc[0]} → {df['Open time'].iloc[-1]}")
print(f"  Total bars: {len(df)}")
print(f"  Total signals: {len(df_s)}")

if len(df_s) == 0:
    print("\n  No signals generated. The v3 gates are very selective.")
    print("  This means the squeeze module rarely fires in backtest —")
    print("  which is fine for live (low frequency, high quality) but")
    print("  means we can't validate WR from historical data alone.")
    print("=" * 80)
    sys.exit(0)

print(f"  SHORT_SQUEEZE: {len(df_s[df_s['type'] == 'SHORT_SQUEEZE'])}")
print(f"  LONG_SQUEEZE:  {len(df_s[df_s['type'] == 'LONG_SQUEEZE'])}")
print(f"  Strong (≥0.70): {len(df_s[df_s['strong']])}")

# ── Win rate by hold period ──
print(f"\n  {'Hold':>6}  {'Win%':>6}  {'Avg%':>7}  {'Med%':>7}  {'Max%':>7}  {'Min%':>7}  {'n':>4}")
print(f"  {'-' * 56}")
for label in ["1h", "2h", "4h", "8h", "12h", "24h"]:
    col = f"ret_{label}"
    if col in df_s.columns:
        valid = df_s[col].dropna()
        if len(valid) > 0:
            wr = (valid > 0).mean() * 100
            print(f"  {label:>6}  {wr:>5.1f}%  {valid.mean():>+6.2f}%  "
                  f"{valid.median():>+6.2f}%  {valid.max():>+6.2f}%  "
                  f"{valid.min():>+6.2f}%  {len(valid):>4}")

# ── Strong signals only ──
strong = df_s[df_s["strong"]]
if len(strong) > 0:
    print(f"\n  STRONG SIGNALS ONLY (score ≥ 0.70):")
    print(f"  {'Hold':>6}  {'Win%':>6}  {'Avg%':>7}  {'Med%':>7}  {'n':>4}")
    print(f"  {'-' * 40}")
    for label in ["1h", "2h", "4h", "8h", "12h", "24h"]:
        col = f"ret_{label}"
        if col in strong.columns:
            valid = strong[col].dropna()
            if len(valid) > 0:
                wr = (valid > 0).mean() * 100
                print(f"  {label:>6}  {wr:>5.1f}%  {valid.mean():>+6.2f}%  "
                      f"{valid.median():>+6.2f}%  {len(valid):>4}")

# ── M4b agreement analysis ──
m4b_agree = df_s[df_s["m4b_agrees"]]
m4b_disagree = df_s[~df_s["m4b_agrees"]]
if len(m4b_agree) > 0 and len(m4b_disagree) > 0:
    print(f"\n  M4b DIVERGENCE AGREEMENT:")
    print(f"  {'Type':>12}  {'n':>4}  {'WR4h':>6}  {'Avg4h':>7}  {'WR24h':>6}  {'Avg24h':>7}")
    print(f"  {'-' * 50}")
    for label, sub in [("Agrees", m4b_agree), ("Disagrees", m4b_disagree)]:
        wr4 = (sub["ret_4h"] > 0).mean() * 100 if "ret_4h" in sub else 0
        a4 = sub["ret_4h"].mean() if "ret_4h" in sub else 0
        wr24 = (sub["ret_24h"] > 0).mean() * 100 if "ret_24h" in sub else 0
        a24 = sub["ret_24h"].mean() if "ret_24h" in sub else 0
        print(f"  {label:>12}  {len(sub):>4}  {wr4:>5.1f}%  {a4:>+6.2f}%  {wr24:>5.1f}%  {a24:>+6.2f}%")

# ── Regime breakdown ──
print(f"\n  REGIME BREAKDOWN:")
for regime in df_s["regime"].unique():
    sub = df_s[df_s["regime"] == regime]
    if len(sub) == 0:
        continue
    wr4 = (sub["ret_4h"] > 0).mean() * 100 if "ret_4h" in sub else 0
    a4 = sub["ret_4h"].mean() if "ret_4h" in sub else 0
    print(f"  {regime:<16}  n={len(sub):>3}  WR4h={wr4:.1f}%  Avg4h={a4:+.2f}%")

# ── Full signal log ──
print(f"\n  ALL SIGNALS:")
print(f"  {'Time':>16}  {'Type':<15} {'Score':>5} {'Dir':<5} {'Price':>8} "
      f"{'z':>6} {'Vol':>5} {'Quality':>7} {'M4b':>6} {'4h':>7} {'24h':>7}")
print(f"  {'-' * 95}")
for _, s in df_s.iterrows():
    h4 = f"{s.get('ret_4h', 0):+.2f}%" if not pd.isna(s.get('ret_4h')) else "-"
    h24 = f"{s.get('ret_24h', 0):+.2f}%" if not pd.isna(s.get('ret_24h')) else "-"
    st = "*" if s["strong"] else " "
    m4b_icon = {"BEARISH": "🔻", "BULLISH": "🔺", "NONE": "—"}.get(s["m4b_div"], "—")
    print(f"  {s['time'][:16]}  {s['type']:<15} {s['score']:.3f} {st}{s['direction']:<4} "
          f"${s['price']:>7.0f} {s['ls_z']:>+5.2f} {s['vol_trend']:>4.1f}x "
          f"{s['squeeze_quality']:>6.3f}  {m4b_icon:>4}  {h4:>7} {h24:>7}")

# ── Monthly breakdown ──
df_s["month"] = pd.to_datetime(df_s["time"]).dt.to_period("M")
print(f"\n  MONTHLY BREAKDOWN:")
print(f"  {'Month':>8}  {'n':>3}  {'WR4h':>6}  {'Avg4h':>7}  {'WR24h':>6}  {'Avg24h':>7}")
print(f"  {'-' * 48}")
for month, group in df_s.groupby("month"):
    wr4 = (group["ret_4h"] > 0).mean() * 100 if "ret_4h" in group else 0
    a4 = group["ret_4h"].mean() if "ret_4h" in group else 0
    wr24 = (group["ret_24h"] > 0).mean() * 100 if "ret_24h" in group else 0
    a24 = group["ret_24h"].mean() if "ret_24h" in group else 0
    print(f"  {str(month):>8}  {len(group):>3}  {wr4:>5.1f}%  {a4:>+6.2f}%  {wr24:>5.1f}%  {a24:>+6.2f}%")

print("=" * 80)

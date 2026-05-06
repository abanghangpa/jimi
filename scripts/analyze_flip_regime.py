#!/usr/bin/env python3
"""
Analyze what market regime distinguishes contrarian-working periods
from aligned-working periods for hist_flip + coil signals.

Compares Q4 2025 (aligned works) vs Q1-Q2 2026 (contrarian works).
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import numpy as np
import pandas as pd
from src.config import CONFIG
from src.utils.data_handler import load_data
from src.utils.indicators import calc_ema, calc_atr, calc_rsi
from src.modules.m18_squeeze import _compute_2h_macd, _detect_macd_coil, SQUEEZE_V5_DEFAULTS

CSV = os.path.join(os.path.dirname(os.path.dirname(__file__)), "eth_15m_merged.csv")
df_all = load_data(CSV)

cfg = dict(CONFIG)
cfg.update(SQUEEZE_V5_DEFAULTS)

# ── Collect all coil+flip events across both periods ──
print("Collecting coil+flip events across 2025-10 to 2026-05...")

df_scan = df_all[df_all["Open time"] >= "2025-10-01"].reset_index(drop=True)
df_scan = df_scan[df_scan["Open time"] < "2026-05-06"].reset_index(drop=True)

# Pre-compute indicators
df_scan["ema21"] = calc_ema(df_scan["Close"], 21)
df_scan["ema55"] = calc_ema(df_scan["Close"], 55)
df_scan["ema200"] = calc_ema(df_scan["Close"], 200)
df_scan["atr"] = calc_atr(df_scan["High"], df_scan["Low"], df_scan["Close"], 14)
df_scan["rsi"] = calc_rsi(df_scan["Close"], 14)
df_scan["vol_ma20"] = df_scan["Volume"].rolling(20).mean()
df_scan["vol_ma50"] = df_scan["Volume"].rolling(50).mean()

# Daily trend (resample)
df_scan["date"] = pd.to_datetime(df_scan["Open time"]).dt.date
daily = df_scan.groupby("date").agg({"High": "max", "Low": "min", "Close": "last"}).reset_index()
daily["daily_ema21"] = calc_ema(daily["Close"], 21)
daily["daily_trend"] = np.where(daily["Close"] > daily["daily_ema21"], "BULL", "BEAR")

SCAN_START = 500
events = []

for idx in range(SCAN_START, len(df_scan) - 96):
    df_slice = df_scan.iloc[:idx+1]
    macd_data = _compute_2h_macd(df_slice, cfg)
    if macd_data is None:
        continue

    hist = macd_data['hist']
    n = len(hist)
    if n < 3:
        continue

    # Check for flip
    prev_hist = hist[-2]
    curr_hist = hist[-1]
    hist_flip = False
    flip_dir = 'NONE'
    if prev_hist < 0 and curr_hist >= 0:
        hist_flip = True
        flip_dir = 'LONG'
    elif prev_hist > 0 and curr_hist <= 0:
        hist_flip = True
        flip_dir = 'SHORT'

    if not hist_flip:
        continue

    # Check coil
    is_coiled, coil_bars, _, coil_dir, coil_details = _detect_macd_coil(macd_data, cfg)
    if not is_coiled:
        continue

    price = float(df_scan['Close'].iloc[idx])
    ema21 = float(df_scan['ema21'].iloc[idx])
    ema55 = float(df_scan['ema55'].iloc[idx])
    ema200 = float(df_scan['ema200'].iloc[idx])
    atr = float(df_scan['atr'].iloc[idx]) if not pd.isna(df_scan['atr'].iloc[idx]) else 0
    rsi = float(df_scan['rsi'].iloc[idx]) if not pd.isna(df_scan['rsi'].iloc[idx]) else 50
    vol_ma20 = float(df_scan['vol_ma20'].iloc[idx]) if not pd.isna(df_scan['vol_ma20'].iloc[idx]) else 0
    vol_ma50 = float(df_scan['vol_ma50'].iloc[idx]) if not pd.isna(df_scan['vol_ma50'].iloc[idx]) else 0
    vol_ratio = float(df_scan['Volume'].iloc[idx]) / vol_ma20 if vol_ma20 > 0 else 1

    ema_trend = 'BULL' if ema21 > ema55 else 'BEAR'
    contrarian = (flip_dir == 'LONG' and ema_trend == 'BEAR') or \
                 (flip_dir == 'SHORT' and ema_trend == 'BULL')

    # Forward returns
    horizons = [8, 32, 96]
    fwd = {}
    for h in horizons:
        if idx + h < len(df_scan):
            fc = float(df_scan['Close'].iloc[idx + h])
            fh = float(df_scan['High'].iloc[idx:idx+h+1].max())
            fl = float(df_scan['Low'].iloc[idx:idx+h+1].min())
            if flip_dir == 'LONG':
                fwd[f'{h}bar_close'] = (fc - price) / price * 100
                fwd[f'{h}bar_best'] = (fh - price) / price * 100
                fwd[f'{h}bar_worst'] = (fl - price) / price * 100
            else:
                fwd[f'{h}bar_close'] = (price - fc) / price * 100
                fwd[f'{h}bar_best'] = (price - fl) / price * 100
                fwd[f'{h}bar_worst'] = (price - fh) / price * 100

    # Regime features
    # ATR percentile (100-bar window)
    atr_series = df_scan['atr'].iloc[max(0,idx-500):idx+1].dropna()
    atr_pctl = (atr_series.values < atr).sum() / len(atr_series) * 100 if len(atr_series) > 0 else 50

    # Price vs EMAs
    dist_ema21 = (price - ema21) / ema21 * 100
    dist_ema55 = (price - ema55) / ema55 * 100
    dist_ema200 = (price - ema200) / ema200 * 100

    # EMA spread (trend strength)
    ema_spread = (ema21 - ema55) / ema55 * 100

    # Recent momentum (16 bars = 4h)
    mom_4h = (price - float(df_scan['Close'].iloc[idx-16])) / float(df_scan['Close'].iloc[idx-16]) * 100 if idx >= 16 else 0
    mom_24h = (price - float(df_scan['Close'].iloc[idx-96])) / float(df_scan['Close'].iloc[idx-96]) * 100 if idx >= 96 else 0

    # MACD hist magnitude
    hist_mag = abs(float(curr_hist))

    # Coil duration
    coil_hours = coil_details.get('coil_hours', 0)

    # RSI zone
    rsi_zone = 'oversold' if rsi < 30 else 'low' if rsi < 45 else 'mid' if rsi < 55 else 'high' if rsi < 70 else 'overbought'

    # Date for period assignment
    time_str = str(df_scan['Open time'].iloc[idx])
    month = time_str[:7]

    events.append({
        'time': time_str,
        'month': month,
        'price': round(price, 2),
        'flip_dir': flip_dir,
        'contrarian': contrarian,
        'ema_trend': ema_trend,
        'ema_spread': round(ema_spread, 3),
        'dist_ema21': round(dist_ema21, 2),
        'dist_ema55': round(dist_ema55, 2),
        'dist_ema200': round(dist_ema200, 2),
        'atr': round(atr, 2),
        'atr_pctl': round(atr_pctl, 1),
        'rsi': round(rsi, 1),
        'rsi_zone': rsi_zone,
        'vol_ratio': round(vol_ratio, 2),
        'mom_4h': round(mom_4h, 2),
        'mom_24h': round(mom_24h, 2),
        'hist_mag': round(hist_mag, 2),
        'coil_hours': coil_hours,
        **fwd,
    })

edf = pd.DataFrame(events)
print(f"Total coil+flip events: {len(edf)}")
print(f"Contrarian: {edf['contrarian'].sum()}, Aligned: {(~edf['contrarian']).sum()}")

# ── Split by outcome ──
h = '96bar_close'  # 24h
edf['win'] = edf[h] > 0

print("\n" + "=" * 90)
print("  REGIME ANALYSIS: What distinguishes contrarian-working vs aligned-working periods")
print("=" * 90)

# ── Compare feature distributions: Contrarian wins vs Contrarian losses ──
print(f"\n{'─'*90}")
print("  CONTRARIAN signals: winners vs losers (24h)")
print(f"{'─'*90}")

contra = edf[edf['contrarian'] == True]
contra_win = contra[contra['win'] == True]
contra_lose = contra[contra['win'] == False]

features = ['ema_spread', 'dist_ema21', 'dist_ema200', 'atr_pctl', 'rsi', 'vol_ratio',
            'mom_4h', 'mom_24h', 'hist_mag', 'coil_hours']

print(f"  {'Feature':>14s}  {'Winners (n={})'.format(len(contra_win)):>20s}  {'Losers (n={})'.format(len(contra_lose)):>20s}  {'Delta':>8s}")
for f in features:
    w = contra_win[f].mean()
    l = contra_lose[f].mean()
    d = w - l
    sig = "***" if abs(d) > (contra_win[f].std() + contra_lose[f].std()) / 2 else ""
    print(f"  {f:>14s}  {w:>20.2f}  {l:>20.2f}  {d:>+8.2f} {sig}")

# ── Compare: Aligned wins vs Aligned losses ──
print(f"\n{'─'*90}")
print("  ALIGNED signals: winners vs losers (24h)")
print(f"{'─'*90}")

align = edf[edf['contrarian'] == False]
align_win = align[align['win'] == True]
align_lose = align[align['win'] == False]

print(f"  {'Feature':>14s}  {'Winners (n={})'.format(len(align_win)):>20s}  {'Losers (n={})'.format(len(align_lose)):>20s}  {'Delta':>8s}")
for f in features:
    w = align_win[f].mean()
    l = align_lose[f].mean()
    d = w - l
    sig = "***" if abs(d) > (align_win[f].std() + align_lose[f].std()) / 2 else ""
    print(f"  {f:>14s}  {w:>20.2f}  {l:>20.2f}  {d:>+8.2f} {sig}")

# ── Monthly regime profile ──
print(f"\n{'─'*90}")
print("  MONTHLY REGIME PROFILE")
print(f"{'─'*90}")
print(f"  {'Month':>8s}  {'n':>4s}  {'Contra%':>8s}  {'WR_c':>6s}  {'WR_a':>6s}  {'Avg_c':>8s}  {'Avg_a':>8s}  {'ATR_p':>6s}  {'EmaSpr':>7s}  {'Mom24h':>7s}  {'VolR':>5s}")

for month, grp in edf.groupby('month'):
    n = len(grp)
    c_pct = grp['contrarian'].mean() * 100
    c_grp = grp[grp['contrarian'] == True]
    a_grp = grp[grp['contrarian'] == False]
    wr_c = c_grp['96bar_close'].gt(0).mean() * 100 if len(c_grp) > 0 else float('nan')
    wr_a = a_grp['96bar_close'].gt(0).mean() * 100 if len(a_grp) > 0 else float('nan')
    avg_c = c_grp['96bar_close'].mean() if len(c_grp) > 0 else float('nan')
    avg_a = a_grp['96bar_close'].mean() if len(a_grp) > 0 else float('nan')
    atr_p = grp['atr_pctl'].mean()
    ema_s = grp['ema_spread'].mean()
    mom = grp['mom_24h'].mean()
    vol = grp['vol_ratio'].mean()
    print(f"  {month:>8s}  {n:>4d}  {c_pct:>7.0f}%  {wr_c:>5.0f}%  {wr_a:>5.0f}%  {avg_c:>+7.2f}%  {avg_a:>+7.2f}%  {atr_p:>5.0f}  {ema_s:>+6.2f}%  {mom:>+6.2f}%  {vol:>5.2f}")

# ── Key discriminators ──
print(f"\n{'─'*90}")
print("  KEY DISCRIMINATORS (feature importance by period outcome)")
print(f"{'─'*90}")

# Split into "contrarian works" months vs "aligned works" months
# Based on monthly data: contrarian works when market is trending down (bear)
# Let's check which features best separate the two regimes

for feature in ['ema_spread', 'mom_24h', 'atr_pctl', 'dist_ema200', 'vol_ratio', 'hist_mag']:
    # Contrarian-good periods: where contrarian avg > aligned avg
    monthly = []
    for month, grp in edf.groupby('month'):
        c = grp[grp['contrarian'] == True]
        a = grp[grp['contrarian'] == False]
        if len(c) > 0 and len(a) > 0:
            c_avg = c['96bar_close'].mean()
            a_avg = a['96bar_close'].mean()
            monthly.append({
                'month': month,
                'contra_better': c_avg > a_avg,
                feature: grp[feature].mean(),
            })
    mdf = pd.DataFrame(monthly)
    if len(mdf) > 0:
        good = mdf[mdf['contra_better'] == True]
        bad = mdf[mdf['contra_better'] == False]
        if len(good) > 0 and len(bad) > 0:
            print(f"\n  {feature}:")
            print(f"    Contra-better months: {feature}={good[feature].mean():.2f}  ({', '.join(good['month'].tolist())})")
            print(f"    Aligned-better months: {feature}={bad[feature].mean():.2f}  ({', '.join(bad['month'].tolist())})")

# ── Threshold analysis ──
print(f"\n{'─'*90}")
print("  THRESHOLD ANALYSIS: ema_spread splits the regimes")
print(f"{'─'*90}")

for spread_thresh in [-1.0, -0.5, 0, 0.5, 1.0]:
    below = edf[edf['ema_spread'] < spread_thresh]
    above = edf[edf['ema_spread'] >= spread_thresh]

    c_below = below[below['contrarian'] == True]
    a_below = below[below['contrarian'] == False]
    c_above = above[above['contrarian'] == True]
    a_above = above[above['contrarian'] == False]

    def safe_stats(g):
        if len(g) == 0:
            return 0, float('nan'), float('nan')
        return len(g), g['96bar_close'].gt(0).mean() * 100, g['96bar_close'].mean()

    nc, wrc, avgc = safe_stats(c_below)
    na, wra, avga = safe_stats(a_below)
    nc2, wrc2, avgc2 = safe_stats(c_above)
    na2, wra2, avga2 = safe_stats(a_above)

    print(f"\n  ema_spread < {spread_thresh:+.1f}%:")
    print(f"    Contrarian: {nc:>3d} events  WR={wrc:>5.1f}%  avg={avgc:>+.2f}%")
    print(f"    Aligned:    {na:>3d} events  WR={wra:>5.1f}%  avg={avga:>+.2f}%")
    print(f"  ema_spread >= {spread_thresh:+.1f}%:")
    print(f"    Contrarian: {nc2:>3d} events  WR={wrc2:>5.1f}%  avg={avgc2:>+.2f}%")
    print(f"    Aligned:    {na2:>3d} events  WR={wra2:>5.1f}%  avg={avga2:>+.2f}%")

# ── Best combined filter ──
print(f"\n{'─'*90}")
print("  COMBINED FILTER SEARCH")
print(f"{'─'*90}")

best = {'wr': 0, 'avg': -999, 'n': 0, 'label': ''}

for spread_lo in [-2, -1, 0]:
    for spread_hi in [0, 1, 2]:
        for mom_lo in [-5, -2, 0]:
            for vol_lo in [0.5, 0.8, 1.0]:
                subset = edf[
                    (edf['ema_spread'] >= spread_lo) &
                    (edf['ema_spread'] < spread_hi) &
                    (edf['mom_24h'] >= mom_lo) &
                    (edf['vol_ratio'] >= vol_lo)
                ]
                if len(subset) < 20:
                    continue

                # Try contrarian and aligned
                for use_contra in [True, False]:
                    s = subset[subset['contrarian'] == use_contra] if use_contra else subset[subset['contrarian'] == False]
                    if len(s) < 10:
                        continue
                    wr = s['96bar_close'].gt(0).mean() * 100
                    avg = s['96bar_close'].mean()
                    tag = 'contra' if use_contra else 'aligned'
                    label = f"spread=[{spread_lo},{spread_hi}) mom>={mom_lo} vol>={vol_lo} {tag}"

                    # Score: prioritize WR * avg (edge quality)
                    score = wr * avg if avg > 0 else 0
                    if score > best['score'] if 'score' in best else 0:
                        best = {'wr': wr, 'avg': avg, 'n': len(s), 'label': label, 'score': score}

if best['n'] > 0:
    print(f"\n  Best filter: {best['label']}")
    print(f"    n={best['n']}  WR={best['wr']:.1f}%  avg={best['avg']:+.2f}%")

# ── Simple regime rule ──
print(f"\n{'─'*90}")
print("  PROPOSED REGIME RULE")
print(f"{'─'*90}")

# Test: use contrarian when ema_spread < 0 (bearish trend), aligned when >= 0
for rule_name, rule_fn in [
    ("Contra when spread<0, else aligned",
     lambda r: r['contrarian'] if r['ema_spread'] < 0 else not r['contrarian']),
    ("Contra when mom_24h<0, else aligned",
     lambda r: r['contrarian'] if r['mom_24h'] < 0 else not r['contrarian']),
    ("Contra when spread<0 AND mom<0, else aligned",
     lambda r: r['contrarian'] if (r['ema_spread'] < 0 and r['mom_24h'] < 0) else not r['contrarian']),
    ("Always contrarian",
     lambda r: True),
    ("Always aligned",
     lambda r: False),
    ("Contra when atr_pctl>50, else aligned",
     lambda r: r['contrarian'] if r['atr_pctl'] > 50 else not r['contrarian']),
]:
    edf['_use'] = edf.apply(rule_fn, axis=1)
    s = edf[edf['_use'] == True]
    if len(s) > 0:
        wr = s['96bar_close'].gt(0).mean() * 100
        avg = s['96bar_close'].mean()
        print(f"  {rule_name:>55s}  n={len(s):>3d}  WR={wr:>5.1f}%  avg={avg:>+.2f}%")

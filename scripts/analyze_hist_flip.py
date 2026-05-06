#!/usr/bin/env python3
"""
Analyze MACD histogram flip patterns on 2h timeframe for 2026.

Tracks every hist_flip event (DIF/DEA zero crossing) and measures
forward returns at multiple horizons. Also checks if preceding coil
improves outcomes.
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import numpy as np
import pandas as pd
from src.config import CONFIG
from src.utils.data_handler import load_data, resample_ohlcv
from src.utils.indicators import calc_ema, calc_atr
from src.modules.m18_squeeze import (
    _compute_2h_macd, _detect_macd_coil, SQUEEZE_V5_DEFAULTS,
)

CSV = os.path.join(os.path.dirname(os.path.dirname(__file__)), "eth_15m_merged.csv")
df_all = load_data(CSV)

# Date range from args or default
import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--start', default='2026-01-01')
parser.add_argument('--end', default=None)
args = parser.parse_args()

df_15m = df_all[df_all["Open time"] >= args.start].reset_index(drop=True)
if args.end:
    df_15m = df_15m[df_15m["Open time"] < args.end].reset_index(drop=True)

print(f"Data: {len(df_15m)} bars  ({df_15m['Open time'].iloc[0]} → {df_15m['Open time'].iloc[-1]})")

# Pre-compute EMA for trend context
df_15m["ema21"] = calc_ema(df_15m["Close"], 21)
df_15m["ema55"] = calc_ema(df_15m["Close"], 55)

cfg = dict(CONFIG)
cfg.update(SQUEEZE_V5_DEFAULTS)

# Scan every bar for 2h MACD hist_flip
print("Scanning for hist_flip events on 2h MACD(8,17,9)...\n")

events = []
SCAN_START = 500  # need enough history for MACD

for idx in range(SCAN_START, len(df_15m) - 96):  # leave room for forward look
    df_slice = df_15m.iloc[:idx+1]
    macd_data = _compute_2h_macd(df_slice, cfg)
    if macd_data is None:
        continue

    hist = macd_data['hist']
    dif = macd_data['dif']
    dea = macd_data['dea']
    ts = macd_data['timestamps']
    n = len(hist)
    if n < 3:
        continue

    # Check for flip at the last bar
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

    # Check if there was a preceding coil
    is_coiled, coil_bars, _, coil_dir, coil_details = _detect_macd_coil(macd_data, cfg)

    # Check EMA alignment
    price = float(df_15m['Close'].iloc[idx])
    ema21 = float(df_15m['ema21'].iloc[idx])
    ema55 = float(df_15m['ema55'].iloc[idx])
    ema_aligned = (flip_dir == 'LONG' and ema21 > ema55) or (flip_dir == 'SHORT' and ema21 < ema55)
    trend = 'BULL' if ema21 > ema55 else 'BEAR'

    # Measure forward returns at multiple horizons
    entry_price = price
    horizons = [4, 8, 16, 32, 48, 96]  # in 15m bars = 1h, 2h, 4h, 8h, 12h, 24h
    fwd = {}
    for h in horizons:
        if idx + h < len(df_15m):
            future_close = float(df_15m['Close'].iloc[idx + h])
            future_high = float(df_15m['High'].iloc[idx:idx+h+1].max())
            future_low = float(df_15m['Low'].iloc[idx:idx+h+1].min())

            if flip_dir == 'LONG':
                ret_close = (future_close - entry_price) / entry_price * 100
                ret_best = (future_high - entry_price) / entry_price * 100
                ret_worst = (future_low - entry_price) / entry_price * 100
            else:
                ret_close = (entry_price - future_close) / entry_price * 100
                ret_best = (entry_price - future_low) / entry_price * 100
                ret_worst = (entry_price - future_high) / entry_price * 100

            fwd[f'{h}bar_close'] = round(ret_close, 3)
            fwd[f'{h}bar_best'] = round(ret_best, 3)
            fwd[f'{h}bar_worst'] = round(ret_worst, 3)

    events.append({
        'bar_idx': idx,
        'time': str(df_15m['Open time'].iloc[idx]),
        'price': round(price, 2),
        'flip_dir': flip_dir,
        'hist_prev': round(float(prev_hist), 3),
        'hist_curr': round(float(curr_hist), 3),
        'dif': round(float(dif[-1]), 3),
        'dea': round(float(dea[-1]), 3),
        'coiled': is_coiled,
        'coil_bars': coil_bars,
        'trend': trend,
        'ema_aligned': ema_aligned,
        **fwd,
    })

if not events:
    print("No hist_flip events found in 2026 data.")
    sys.exit(0)

edf = pd.DataFrame(events)
print(f"Found {len(edf)} hist_flip events in 2026\n")

# ── Summary stats ──
print("=" * 80)
print("  HIST_FLIP PATTERN ANALYSIS — 2026")
print("=" * 80)

for direction in ['LONG', 'SHORT']:
    subset = edf[edf['flip_dir'] == direction]
    if len(subset) == 0:
        continue
    print(f"\n{'─'*80}")
    print(f"  {direction} FLIPS ({len(subset)} events)")
    print(f"{'─'*80}")

    for h in horizons:
        label = f'{h}bar'
        h_label = f'{h*15}min' if h < 4 else f'{h*15//60}h'
        close_col = f'{label}_close'
        best_col = f'{label}_best'
        worst_col = f'{label}_worst'

        closes = subset[close_col]
        wins = (closes > 0).sum()
        wr = wins / len(closes) * 100
        avg = closes.mean()
        med = closes.median()
        best_avg = subset[best_col].mean()
        worst_avg = subset[worst_col].mean()

        print(f"  {h_label:>6s}  WR={wr:5.1f}%  avg={avg:+.3f}%  med={med:+.3f}%  best_avg={best_avg:+.3f}%  worst_avg={worst_avg:+.3f}%")

# ── With coil vs without coil ──
print(f"\n{'─'*80}")
print("  COIL vs NO-COIL (hist_flip preceded by MACD coil)")
print(f"{'─'*80}")

for has_coil, label in [(True, "WITH COIL"), (False, "NO COIL")]:
    subset = edf[edf['coiled'] == has_coil]
    if len(subset) == 0:
        print(f"  {label}: 0 events")
        continue
    print(f"\n  {label} ({len(subset)} events):")
    for h in [8, 32, 96]:  # 2h, 8h, 24h
        h_label = f'{h*15//60}h'
        close_col = f'{h}bar_close'
        closes = subset[close_col]
        wins = (closes > 0).sum()
        wr = wins / len(closes) * 100
        avg = closes.mean()
        print(f"    {h_label:>4s}  WR={wr:5.1f}%  avg={avg:+.3f}%")

# ── EMA aligned vs not ──
print(f"\n{'─'*80}")
print("  EMA ALIGNED vs NOT (flip direction matches trend)")
print(f"{'─'*80}")

for aligned, label in [(True, "EMA ALIGNED"), (False, "EMA CONTRARIAN")]:
    subset = edf[edf['ema_aligned'] == aligned]
    if len(subset) == 0:
        print(f"  {label}: 0 events")
        continue
    print(f"\n  {label} ({len(subset)} events):")
    for h in [8, 32, 96]:
        h_label = f'{h*15//60}h'
        close_col = f'{h}bar_close'
        closes = subset[close_col]
        wins = (closes > 0).sum()
        wr = wins / len(closes) * 100
        avg = closes.mean()
        print(f"    {h_label:>4s}  WR={wr:5.1f}%  avg={avg:+.3f}%")

# ── Combined: coil + EMA ──
print(f"\n{'─'*80}")
print("  COMBINED: COIL + EMA ALIGNMENT")
print(f"{'─'*80}")

for coil_val in [True, False]:
    for ema_val in [True, False]:
        subset = edf[(edf['coiled'] == coil_val) & (edf['ema_aligned'] == ema_val)]
        if len(subset) == 0:
            continue
        tag = f"coil={'Y' if coil_val else 'N'} ema={'Y' if ema_val else 'N'}"
        print(f"\n  {tag} ({len(subset)} events):")
        for h in [8, 32, 96]:
            h_label = f'{h*15//60}h'
            close_col = f'{h}bar_close'
            closes = subset[close_col]
            wins = (closes > 0).sum()
            wr = wins / len(closes) * 100
            avg = closes.mean()
            print(f"    {h_label:>4s}  WR={wr:5.1f}%  avg={avg:+.3f}%")

# ── Monthly breakdown ──
print(f"\n{'─'*80}")
print("  MONTHLY BREAKDOWN")
print(f"{'─'*80}")
edf['month'] = edf['time'].str[:7]
for month, grp in edf.groupby('month'):
    print(f"\n  {month} ({len(grp)} flips):")
    for h in [8, 96]:
        h_label = f'{h*15//60}h'
        close_col = f'{h}bar_close'
        closes = grp[close_col]
        wins = (closes > 0).sum()
        wr = wins / len(closes) * 100
        avg = closes.mean()
        print(f"    {h_label:>4s}  WR={wr:5.1f}%  avg={avg:+.3f}%")

# ── Individual event table ──
print(f"\n{'─'*80}")
print("  ALL EVENTS (chronological)")
print(f"{'─'*80}")
print(f"  {'Time':>19s}  {'Dir':>5s}  {'Price':>8s}  {'Hist':>8s}  {'Coil':>4s}  {'Trend':>5s}  {'EMA':>3s}  {'2h':>7s}  {'8h':>7s}  {'24h':>7s}")
for _, r in edf.iterrows():
    coil_mark = f"{int(r['coil_bars'])}" if r['coiled'] else "—"
    ema_mark = "✓" if r['ema_aligned'] else "✗"
    h8 = f"{r['8bar_close']:+.2f}%" if '8bar_close' in r and not pd.isna(r['8bar_close']) else "?"
    h32 = f"{r['32bar_close']:+.2f}%" if '32bar_close' in r and not pd.isna(r['32bar_close']) else "?"
    h96 = f"{r['96bar_close']:+.2f}%" if '96bar_close' in r and not pd.isna(r['96bar_close']) else "?"
    print(f"  {r['time']:>19s}  {r['flip_dir']:>5s}  ${r['price']:>7.2f}  {r['hist_prev']:+.2f}→{r['hist_curr']:+.2f}  {coil_mark:>4s}  {r['trend']:>5s}  {ema_mark:>3s}  {h8:>7s}  {h32:>7s}  {h96:>7s}")

#!/usr/bin/env python3
"""
JIMI — Hourly Liquidity Collector

Captures live liquidity snapshots every hour:
  - Liquidation magnets (HVNs) with swept status
  - Support/resistance levels
  - Derivatives (OI, funding, L/S ratio, positioning)
  - Price, volume, ATR
  - Module scores & ICS

Usage:
    python scripts/liquidity_collector.py              # run once (for cron)
    python scripts/liquidity_collector.py --loop       # run every hour
    python scripts/liquidity_collector.py --loop 30    # every 30 minutes
    python scripts/liquidity_collector.py --json       # JSON output

Output: data/liquidity_snapshots.csv
"""

import argparse
import csv
import json
import os
import sys
import time
from datetime import datetime, timezone

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from src.config import CONFIG
from src.utils.data_handler import fetch_recent
from src.utils.indicators import (
    calc_ema, calc_macd, calc_rsi, calc_atr, calc_vwap, calc_vol_ratio,
    calc_swing_bias, calc_phase0, calc_trend_state,
)
from src.modules.m1_macd_v2 import score_m1_v2 as score_m1
from src.modules.m2_ema import score_m2
from src.modules.m3_vwap import score_m3
from src.modules.m4_cvd import calc_cvd_15m, detect_cvd_divergence_15m, calc_cvd_2h, detect_cvd_zero_cross, score_m4
from src.modules.m5_liquidation import (
    build_volume_profile, find_magnets, find_gaps, score_m5, find_support_resistance,
)
from src.modules.m6_derivatives import score_derivatives, get_derivatives_summary
from src.engine import calc_ics, check_entry_filters
from src.utils.data_handler import resample_ohlcv

# Import scanner helpers
sys.path.insert(0, os.path.join(os.path.dirname(__file__)))
from scanner import compute_indicators, scan_signal

DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data')
CSV_PATH = os.path.join(DATA_DIR, 'liquidity_snapshots.csv')

CSV_FIELDS = [
    'timestamp', 'price', 'high_24h', 'low_24h', 'volume_24h',
    'atr_1h', 'atr_pct', 'vol_ratio', 'taker_ratio',
    # Magnets
    'mag1_price', 'mag1_strength', 'mag1_dist_pct', 'mag1_swept',
    'mag2_price', 'mag2_strength', 'mag2_dist_pct', 'mag2_swept',
    'mag3_price', 'mag3_strength', 'mag3_dist_pct', 'mag3_swept',
    'mag4_price', 'mag4_strength', 'mag4_dist_pct', 'mag4_swept',
    'mag5_price', 'mag5_strength', 'mag5_dist_pct', 'mag5_swept',
    # S/R
    'sr1_price', 'sr1_type', 'sr1_strength',
    'sr2_price', 'sr2_type', 'sr2_strength',
    'sr3_price', 'sr3_type', 'sr3_strength',
    # Derivatives
    'oi_eth', 'oi_usd', 'oi_roc_1h',
    'ls_ratio', 'long_pct', 'short_pct', 'ls_zscore',
    'top_ls_ratio', 'whale_signal', 'futures_taker_ratio', 'futures_flow',
    'funding_rate', 'positioning', 'oi_price_div',
    # Scores
    'm1_dir', 'm1_score', 'm2_score', 'm3_score', 'm4_score', 'm5_score',
    'ics', 'direction', 'swing_bias', 'phase0',
]


def collect_snapshot():
    """Run a single liquidity scan and return structured data."""
    df_15m = fetch_recent(bars=1000)
    df_15m, df_1h, df_2h, df_4h, df_1d = compute_indicators(df_15m)
    result = scan_signal(df_15m, df_1h, df_2h, df_4h, df_1d)

    idx = len(df_15m) - 1
    row = df_15m.iloc[idx]
    price = float(row['Close'])

    # 24h stats (96 candles on 15m)
    lookback_24h = min(96, idx + 1)
    h24 = df_15m['High'].values[idx - lookback_24h + 1:idx + 1].astype(float)
    l24 = df_15m['Low'].values[idx - lookback_24h + 1:idx + 1].astype(float)
    v24 = df_15m['Volume'].values[idx - lookback_24h + 1:idx + 1].astype(float)

    snap = {
        'timestamp': str(row['Open time']),
        'price': round(price, 2),
        'high_24h': round(float(np.max(h24)), 2),
        'low_24h': round(float(np.min(l24)), 2),
        'volume_24h': round(float(np.sum(v24)), 2),
    }

    # ATR & volume
    atr_1h = df_1h['atr'].iloc[len(df_1h) - 1] if 'atr' in df_1h.columns else None
    snap['atr_1h'] = round(float(atr_1h), 2) if atr_1h and not pd.isna(atr_1h) else ''
    snap['atr_pct'] = round(float(atr_1h / price * 100), 3) if atr_1h and not pd.isna(atr_1h) else ''
    snap['vol_ratio'] = round(float(row.get('vol_ratio', 0)), 3) if not pd.isna(row.get('vol_ratio')) else ''
    snap['taker_ratio'] = round(float(row.get('taker_ratio', 0)), 4) if not pd.isna(row.get('taker_ratio')) else ''

    # Magnets (with swept status)
    magnets = result.get('magnets', [])
    for i in range(5):
        prefix = f'mag{i + 1}'
        if i < len(magnets):
            mag = magnets[i]
            if len(mag) == 4:
                p, s, swept, swept_at = mag
            else:
                p, s = mag[0], mag[1]
                swept, swept_at = False, None
            snap[f'{prefix}_price'] = p
            snap[f'{prefix}_strength'] = s
            snap[f'{prefix}_dist_pct'] = round((p - price) / price * 100, 2)
            snap[f'{prefix}_swept'] = 'YES' if swept else 'NO'
        else:
            snap[f'{prefix}_price'] = ''
            snap[f'{prefix}_strength'] = ''
            snap[f'{prefix}_dist_pct'] = ''
            snap[f'{prefix}_swept'] = ''

    # S/R levels
    sr = result.get('sr_levels', [])
    for i in range(3):
        prefix = f'sr{i + 1}'
        if i < len(sr):
            p, s, t, _, _ = sr[i]
            snap[f'{prefix}_price'] = round(p, 2)
            snap[f'{prefix}_type'] = t
            snap[f'{prefix}_strength'] = round(s, 1)
        else:
            snap[f'{prefix}_price'] = ''
            snap[f'{prefix}_type'] = ''
            snap[f'{prefix}_strength'] = ''

    # Derivatives
    deriv = result.get('derivatives', {})
    if deriv and 'error' not in deriv:
        snap['oi_eth'] = round(deriv.get('oi', 0), 0)
        snap['oi_usd'] = round(deriv.get('oi_usd', 0), 0)
        snap['oi_roc_1h'] = round(deriv.get('oi_roc_1h', 0), 4)
        snap['ls_ratio'] = round(deriv.get('ls_ratio', 0), 4)
        snap['long_pct'] = round(deriv.get('long_pct', 0), 1)
        snap['short_pct'] = round(deriv.get('short_pct', 0), 1)
        snap['ls_zscore'] = round(deriv.get('ls_zscore', 0), 2)
        snap['top_ls_ratio'] = round(deriv.get('top_ls_ratio', 0), 4)
        snap['whale_signal'] = deriv.get('whale_signal', '')
        snap['futures_taker_ratio'] = round(deriv.get('futures_taker_ratio', 0), 4)
        snap['futures_flow'] = deriv.get('futures_flow', '')
        fr = deriv.get('funding_rate')
        snap['funding_rate'] = round(fr * 100, 4) if fr is not None else ''
        snap['positioning'] = deriv.get('positioning', '')
        snap['oi_price_div'] = deriv.get('oi_price_div', '')
    else:
        for k in ['oi_eth', 'oi_usd', 'oi_roc_1h', 'ls_ratio', 'long_pct', 'short_pct',
                   'ls_zscore', 'top_ls_ratio', 'whale_signal', 'futures_taker_ratio',
                   'futures_flow', 'funding_rate', 'positioning', 'oi_price_div']:
            snap[k] = ''

    # Module scores
    snap['m1_dir'] = result.get('m1', {}).get('direction', '')
    snap['m1_score'] = result.get('m1', {}).get('score', '')
    snap['m2_score'] = result.get('m2', {}).get('score', '')
    snap['m3_score'] = result.get('m3', {}).get('score', '')
    snap['m4_score'] = result.get('m4', {}).get('score', '')
    snap['m5_score'] = result.get('m5', {}).get('score', '')
    snap['ics'] = result.get('ics', '')
    snap['direction'] = result.get('direction', '')
    snap['swing_bias'] = result.get('swing_bias', '')
    snap['phase0'] = result.get('phase0', '')

    return snap, result


def save_snapshot(snap):
    """Append snapshot to CSV."""
    os.makedirs(DATA_DIR, exist_ok=True)
    file_exists = os.path.exists(CSV_PATH)

    with open(CSV_PATH, 'a', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=CSV_FIELDS, extrasaction='ignore')
        if not file_exists:
            writer.writeheader()
        writer.writerow(snap)


def print_snapshot(snap):
    """Pretty-print the snapshot."""
    print("\n" + "═" * 60)
    print("  JIMI — LIQUIDITY SNAPSHOT")
    print("═" * 60)
    print(f"  Time:      {snap['timestamp']}")
    print(f"  Price:     ${snap['price']}")
    print(f"  24h H/L:   ${snap['high_24h']} / ${snap['low_24h']}")
    print(f"  ATR (1H):  ${snap['atr_1h']}  ({snap['atr_pct']}%)")
    print(f"  Vol Ratio: {snap['vol_ratio']}x")
    print(f"  Taker:     {snap['taker_ratio']}")

    print(f"\n  Liquidation Magnets:")
    for i in range(1, 6):
        p = snap.get(f'mag{i}_price', '')
        if p == '':
            continue
        s = snap[f'mag{i}_strength']
        d = snap[f'mag{i}_dist_pct']
        swept = snap[f'mag{i}_swept']
        swept_tag = "  ✅ SWEPT" if swept == 'YES' else ""
        arrow = "↑" if d > 0 else "↓"
        print(f"    #{i}: ${p}  str={s}x  ({arrow}{abs(d)}%){swept_tag}")

    print(f"\n  Support/Resistance:")
    for i in range(1, 4):
        p = snap.get(f'sr{i}_price', '')
        if p == '':
            continue
        t = snap[f'sr{i}_type']
        s = snap[f'sr{i}_strength']
        print(f"    #{i}: ${p}  {t}  str={s}")

    if snap.get('oi_eth'):
        print(f"\n  Derivatives:")
        print(f"    OI:        {snap['oi_eth']:,.0f} ETH  (${snap['oi_usd'] / 1e9:.2f}B)  Δ1h: {snap['oi_roc_1h']}%")
        print(f"    L/S:       {snap['ls_ratio']}  (L{snap['long_pct']}% / S{snap['short_pct']}%)  z={snap['ls_zscore']}")
        print(f"    Funding:   {snap['funding_rate']}%")
        print(f"    Position:  {snap['positioning']}  whale={snap['whale_signal']}")

    print(f"\n  Scores: M1={snap['m1_score']} M2={snap['m2_score']} M3={snap['m3_score']} "
          f"M4={snap['m4_score']} M5={snap['m5_score']}  ICS={snap['ics']}")
    print(f"  Direction: {snap['direction']}  Bias: {snap['swing_bias']}")
    print("═" * 60)


def main():
    parser = argparse.ArgumentParser(description='JIMI Liquidity Collector')
    parser.add_argument('--loop', nargs='?', const=60, type=int, metavar='MINUTES',
                        help='Run in loop (default: 60 min interval)')
    parser.add_argument('--json', action='store_true', help='Output JSON')
    args = parser.parse_args()

    if args.loop:
        interval = args.loop * 60
        print(f"Running in loop mode — capturing every {args.loop} minutes")
        print(f"Output: {CSV_PATH}")
        print("Press Ctrl+C to stop\n")

        while True:
            try:
                snap, raw = collect_snapshot()
                save_snapshot(snap)
                if args.json:
                    print(json.dumps(snap, default=str))
                else:
                    print_snapshot(snap)
                next_run = datetime.now(timezone.utc).timestamp() + interval
                next_str = datetime.fromtimestamp(next_run, tz=timezone.utc).strftime('%H:%M UTC')
                print(f"\n  Next run: {next_str}")
                time.sleep(interval)
            except KeyboardInterrupt:
                print("\nStopped.")
                break
            except Exception as e:
                print(f"Error: {e} — retrying in 60s")
                time.sleep(60)
    else:
        snap, raw = collect_snapshot()
        save_snapshot(snap)
        if args.json:
            print(json.dumps(snap, default=str))
        else:
            print_snapshot(snap)
        print(f"\nSaved to {CSV_PATH}")


if __name__ == '__main__':
    main()

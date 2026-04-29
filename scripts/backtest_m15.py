#!/usr/bin/env python3
"""
JIMI — M15 Liquidation Level Backtest

Backtests the liquidation level estimator against real price action.
For each candle, estimates liquidation zones then checks if price
subsequently swept through them.

Usage:
    python scripts/backtest_m15.py
    python scripts/backtest_m15.py --verbose
    python scripts/backtest_m15.py --export m15_results.csv
"""

import argparse
import sys
import os
import json
import numpy as np
import pandas as pd
from datetime import datetime, timedelta

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from src.config import CONFIG
from src.utils.data_handler import fetch_recent, resample_ohlcv
from src.utils.indicators import calc_atr, calc_ema, calc_rsi
from src.modules.m6_derivatives import (
    fetch_oi_history, fetch_ls_ratio, fetch_all_derivatives,
    compute_oi_signals, compute_positioning_signals,
)
from src.modules.m15_liq_levels import (
    estimate_entry_distribution, estimate_liquidation_cascades,
    find_stop_clusters, estimate_liquidity_levels,
    fetch_order_book_depth,
)
from src.modules.m5_liquidation import find_support_resistance


def fetch_extended_derivatives(symbol="ETHUSDT"):
    """Fetch max available derivatives data."""
    print("  Fetching OI history...")
    oi = fetch_oi_history(symbol, period="15m", limit=1000)
    print(f"  OI: {len(oi)} bars, {oi['timestamp'].min()} → {oi['timestamp'].max()}")

    print("  Fetching L/S ratio...")
    ls = fetch_ls_ratio(symbol, period="15m", limit=1000)

    print("  Fetching top trader L/S...")
    from src.modules.m6_derivatives import fetch_top_trader_ls, fetch_taker_ratio, fetch_funding_rate
    top = fetch_top_trader_ls(symbol, period="15m", limit=1000)
    taker = fetch_taker_ratio(symbol, period="15m", limit=1000)
    funding = fetch_funding_rate(symbol, limit=1000)

    # Merge
    df = oi.merge(ls, on="timestamp", how="outer")
    df = df.merge(top, on="timestamp", how="outer")
    df = df.merge(taker, on="timestamp", how="outer")
    df = df.sort_values("timestamp").reset_index(drop=True)
    df = df.ffill()

    return df, funding


def run_backtest_m15(df_15m, df_deriv, verbose=False):
    """Backtest M15 liquidation levels.

    For each bar:
    1. Estimate liquidation zones using M15
    2. Look forward N bars to see if price hit each zone
    3. Track hit rates, timing, and accuracy
    """
    lookback_liq = 96   # 24h for entry distribution
    forward_bars = 96   # check next 24h for hits
    min_history = 200   # need enough data for S/R

    # Align derivatives with OHLCV
    df_deriv = df_deriv.set_index("timestamp")
    df_15m_idx = df_15m.set_index("Open time")

    results = []
    zone_stats = {
        'LONG_LIQ': {'total': 0, 'hit': 0, 'avg_bars': []},
        'SHORT_LIQ': {'total': 0, 'hit': 0, 'avg_bars': []},
        'LONG_STOP': {'total': 0, 'hit': 0, 'avg_bars': []},
        'SHORT_STOP': {'total': 0, 'hit': 0, 'avg_bars': []},
    }

    total_bars = len(df_15m)
    start_idx = max(min_history, lookback_liq)
    end_idx = total_bars - forward_bars

    scan_interval = 4  # every hour (4 bars on 15m)
    scans = list(range(start_idx, end_idx, scan_interval))
    print(f"\n  Backtesting {len(scans)} scan points ({start_idx} → {end_idx})")
    print(f"  Forward window: {forward_bars} bars ({forward_bars * 15 / 60:.0f}h)\n")

    for scan_num, idx in enumerate(scans):
        ts = df_15m.iloc[idx]['Open time']
        current_price = float(df_15m.iloc[idx]['Close'])

        # Get derivatives at this time
        try:
            ts_pd = pd.Timestamp(ts)
            # Find closest derivative row
            deriv_idx = df_deriv.index.get_indexer([ts_pd], method='nearest')[0]
            if deriv_idx < 0:
                continue
            deriv_row = df_deriv.iloc[deriv_idx]

            oi_usd = float(deriv_row.get('oi_usd', 0))
            ls_ratio = float(deriv_row.get('ls_ratio', 1.0))

            # Compute positioning signals
            deriv_slice = df_deriv.iloc[max(0, deriv_idx-48):deriv_idx+1].copy()
            deriv_slice = deriv_slice.reset_index()
            deriv_slice = compute_positioning_signals(deriv_slice)
            last_deriv = deriv_slice.iloc[-1]
            positioning = last_deriv.get('positioning', 'NEUTRAL')
        except Exception:
            oi_usd = 0
            ls_ratio = 1.0
            positioning = 'NEUTRAL'

        # Compute S/R levels
        sr_levels = find_support_resistance(df_15m, idx)

        # Run M15
        try:
            zones = estimate_liquidity_levels(
                df_15m, idx, sr_levels, oi_usd, ls_ratio,
                direction_bias=None, order_book=None)
        except Exception as e:
            if verbose:
                print(f"  [{ts}] M15 error: {e}")
            continue

        if not zones:
            continue

        # Check forward: did price hit each zone?
        forward_end = min(idx + forward_bars, total_bars)
        future_highs = df_15m['High'].values[idx+1:forward_end+1].astype(float)
        future_lows = df_15m['Low'].values[idx+1:forward_end+1].astype(float)
        future_times = df_15m['Open time'].values[idx+1:forward_end+1]

        for zone in zones:
            zp = zone['price']
            ztype = zone['type']
            strength = zone['strength']
            cascade = zone.get('cascade_risk', 'LOW')
            dist_pct = zone.get('dist_pct', 0)

            hit = False
            hit_bar = None
            hit_time = None

            if zp > current_price:
                # Zone above — check if high reached it
                for bi, h in enumerate(future_highs):
                    if h >= zp:
                        hit = True
                        hit_bar = bi + 1
                        hit_time = str(future_times[bi])
                        break
            elif zp < current_price:
                # Zone below — check if low reached it
                for bi, l in enumerate(future_lows):
                    if l <= zp:
                        hit = True
                        hit_bar = bi + 1
                        hit_time = str(future_times[bi])
                        break

            if ztype in zone_stats:
                zone_stats[ztype]['total'] += 1
                if hit:
                    zone_stats[ztype]['hit'] += 1
                    zone_stats[ztype]['avg_bars'].append(hit_bar)

            results.append({
                'scan_time': str(ts),
                'scan_price': current_price,
                'zone_price': round(zp, 2),
                'zone_type': ztype,
                'strength': round(strength, 1),
                'cascade_risk': cascade,
                'dist_pct': round(dist_pct, 2),
                'hit': hit,
                'hit_bar': hit_bar,
                'hit_time': hit_time,
                'oi_usd': oi_usd,
                'positioning': positioning,
            })

        if verbose and (scan_num + 1) % 24 == 0:
            print(f"  [{scan_num+1}/{len(scans)}] {ts}  price=${current_price:.2f}  zones={len(zones)}")

    return results, zone_stats


def print_report(results, zone_stats):
    """Print backtest report."""
    total_zones = len(results)
    total_hits = sum(1 for r in results if r['hit'])

    print("\n" + "═" * 60)
    print("  M15 LIQUIDATION LEVEL BACKTEST — RESULTS")
    print("═" * 60)

    print(f"\n  Total zones tested:  {total_zones}")
    print(f"  Total hits:          {total_hits}  ({total_hits/total_zones*100:.1f}%)")
    print(f"  Misses:              {total_zones - total_hits}  ({(total_zones-total_hits)/total_zones*100:.1f}%)")

    print(f"\n  By Zone Type:")
    print(f"  {'Type':<14} {'Total':>6} {'Hit':>6} {'Rate':>8} {'Avg Bars':>10}")
    print(f"  {'─'*14} {'─'*6} {'─'*6} {'─'*8} {'─'*10}")

    for ztype in ['LONG_LIQ', 'SHORT_LIQ', 'LONG_STOP', 'SHORT_STOP']:
        stats = zone_stats.get(ztype, {})
        total = stats.get('total', 0)
        hit = stats.get('hit', 0)
        rate = hit / total * 100 if total > 0 else 0
        avg_bars = np.mean(stats.get('avg_bars', [0])) if stats.get('avg_bars') else 0
        print(f"  {ztype:<14} {total:>6} {hit:>6} {rate:>7.1f}% {avg_bars:>9.1f}")

    # By cascade risk
    print(f"\n  By Cascade Risk:")
    print(f"  {'Risk':<8} {'Total':>6} {'Hit':>6} {'Rate':>8}")
    print(f"  {'─'*8} {'─'*6} {'─'*6} {'─'*8}")

    for risk in ['HIGH', 'MED', 'LOW']:
        subset = [r for r in results if r['cascade_risk'] == risk]
        hits = sum(1 for r in subset if r['hit'])
        total = len(subset)
        rate = hits / total * 100 if total > 0 else 0
        print(f"  {risk:<8} {total:>6} {hits:>6} {rate:>7.1f}%")

    # By strength bucket
    print(f"\n  By Strength Bucket:")
    print(f"  {'Bucket':<12} {'Total':>6} {'Hit':>6} {'Rate':>8}")
    print(f"  {'─'*12} {'─'*6} {'─'*6} {'─'*8}")

    for lo, hi, label in [(0, 20, '0-20'), (20, 40, '20-40'), (40, 60, '40-60'),
                           (60, 80, '60-80'), (80, 101, '80-100')]:
        subset = [r for r in results if lo <= r['strength'] < hi]
        hits = sum(1 for r in subset if r['hit'])
        total = len(subset)
        rate = hits / total * 100 if total > 0 else 0
        print(f"  {label:<12} {total:>6} {hits:>6} {rate:>7.1f}%")

    # By distance bucket
    print(f"\n  By Distance from Price:")
    print(f"  {'Distance':<12} {'Total':>6} {'Hit':>6} {'Rate':>8}")
    print(f"  {'─'*12} {'─'*6} {'─'*6} {'─'*8}")

    for lo, hi, label in [(0, 0.5, '0-0.5%'), (0.5, 1, '0.5-1%'), (1, 2, '1-2%'),
                           (2, 3, '2-3%'), (3, 100, '3%+')]:
        subset = [r for r in results if lo <= abs(r['dist_pct']) < hi]
        hits = sum(1 for r in subset if r['hit'])
        total = len(subset)
        rate = hits / total * 100 if total > 0 else 0
        print(f"  {label:<12} {total:>6} {hits:>6} {rate:>7.1f}%")

    # By positioning
    print(f"\n  By Positioning:")
    print(f"  {'Position':<16} {'Total':>6} {'Hit':>6} {'Rate':>8}")
    print(f"  {'─'*16} {'─'*6} {'─'*6} {'─'*8}")

    for pos in ['CROWDED_LONG', 'CROWDED_SHORT', 'NEUTRAL']:
        subset = [r for r in results if r.get('positioning') == pos]
        hits = sum(1 for r in subset if r['hit'])
        total = len(subset)
        rate = hits / total * 100 if total > 0 else 0
        print(f"  {pos:<16} {total:>6} {hits:>6} {rate:>7.1f}%")

    # Top 10 best predictions (highest strength, hit)
    hits_only = [r for r in results if r['hit']]
    hits_only.sort(key=lambda x: x['strength'], reverse=True)
    if hits_only:
        print(f"\n  Top 10 Strongest Hits:")
        for r in hits_only[:10]:
            print(f"    {r['scan_time']}  ${r['zone_price']:.2f}  {r['zone_type']}  "
                  f"str={r['strength']:.0f}  cascade={r['cascade_risk']}  "
                  f"hit in {r['hit_bar']} bars")

    print("\n" + "═" * 60)


def main():
    parser = argparse.ArgumentParser(description='M15 Liquidation Level Backtest')
    parser.add_argument('--verbose', action='store_true')
    parser.add_argument('--export', type=str, help='Export results to CSV')
    args = parser.parse_args()

    print("M15 Liquidation Level Backtest")
    print("=" * 40)

    # Fetch OHLCV (last 10 days = 960 bars on 15m)
    print("\n[1/3] Fetching OHLCV data...")
    df_15m = fetch_recent(bars=1000)
    print(f"  OHLCV: {len(df_15m)} bars, {df_15m['Open time'].iloc[0]} → {df_15m['Open time'].iloc[-1]}")

    # Fetch derivatives
    print("\n[2/3] Fetching derivatives data...")
    df_deriv, funding = fetch_extended_derivatives()

    # Run backtest
    print("\n[3/3] Running backtest...")
    results, zone_stats = run_backtest_m15(df_15m, df_deriv, verbose=args.verbose)

    # Report
    print_report(results, zone_stats)

    # Export
    if args.export:
        pd.DataFrame(results).to_csv(args.export, index=False)
        print(f"\n  Exported to {args.export}")


if __name__ == '__main__':
    main()

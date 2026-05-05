#!/usr/bin/env python3
"""
Liquidity Grab Analysis — did price grab smaller or bigger liquidity first?

At each snapshot, we build a volume profile and identify liquidity magnets (HVNs).
Then we look forward to see which magnets price swept first.
We classify: did price go for the smallest available, the biggest, or something else?

Usage:
    python scripts/liquidity_grab_analysis.py
    python scripts/liquidity_grab_analysis.py --year 2026
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import numpy as np
import pandas as pd
from datetime import datetime
from collections import defaultdict

from src.modules.m5_liquidation import build_volume_profile, find_magnets


def load_data(csv_path):
    df = pd.read_csv(csv_path)
    df['Open time'] = pd.to_datetime(df['Open time'])
    for c in ['Open', 'High', 'Low', 'Close', 'Volume',
              'Taker buy base asset volume', 'Taker buy quote asset volume']:
        df[c] = pd.to_numeric(df[c], errors='coerce')
    return df


def find_liquidity_levels(df, idx, lookback=672, n_bins=50):
    """Build volume profile and return all HVNs with strength."""
    highs = df['High'].values[:idx+1].astype(float)
    lows = df['Low'].values[:idx+1].astype(float)
    closes = df['Close'].values[:idx+1].astype(float)
    volumes = df['Volume'].values[:idx+1].astype(float)

    actual_lookback = min(lookback, idx + 1)
    bin_centers, vol_profile, bin_edges = build_volume_profile(
        highs[-actual_lookback:], lows[-actual_lookback:],
        closes[-actual_lookback:], volumes[-actual_lookback:],
        n_bins=n_bins, lookback=actual_lookback)

    if bin_centers is None:
        return []

    magnets = find_magnets(bin_centers, vol_profile, n_magnets=10, min_gap_pct=0.003)
    return magnets  # list of (price, volume, strength_ratio)


def check_sweep(price_series_high, price_series_low, magnet_price, direction_hint=None):
    """Check if price swept through a magnet level.
    Returns the bar index of the sweep, or None.
    """
    for i in range(len(price_series_high)):
        h = float(price_series_high[i])
        l = float(price_series_low[i])
        if h >= magnet_price and l <= magnet_price:
            return i
    return None


def analyze_sweeps(df, year=2026, snapshot_interval=96, lookforward=192, lookback=672):
    """
    snapshot_interval: how often to take a snapshot (in bars). 96 = 1 day on 15m.
    lookforward: how many bars to look ahead for sweeps. 192 = 2 days.
    lookback: bars used for volume profile. 672 = 7 days.
    """
    # Filter to year
    df_year = df[df['Open time'].dt.year == year].copy().reset_index(drop=True)
    if len(df_year) < lookback + lookforward:
        print(f"Not enough data for {year}")
        return

    print(f"\n{'='*70}")
    print(f"  LIQUIDITY GRAB ANALYSIS — {year}")
    print(f"  Snapshots every {snapshot_interval} bars ({snapshot_interval*15}min)")
    print(f"  Lookforward: {lookforward} bars ({lookforward*15}min)")
    print(f"{'='*70}\n")

    results = []
    total_snapshots = 0

    # Walk through the year
    start_idx = lookback
    end_idx = len(df_year) - lookforward

    idx = start_idx
    while idx < end_idx:
        ts = df_year['Open time'].iloc[idx]
        price = float(df_year['Close'].iloc[idx])

        # Get liquidity levels at this snapshot
        magnets = find_liquidity_levels(df_year, idx, lookback=lookback)

        if len(magnets) < 3:
            idx += snapshot_interval
            continue

        # Look forward for sweeps
        future_highs = df_year['High'].values[idx+1:idx+1+lookforward].astype(float)
        future_lows = df_year['Low'].values[idx+1:idx+1+lookforward].astype(float)
        future_times = df_year['Open time'].values[idx+1:idx+1+lookforward]

        # For each magnet, find the first sweep bar
        sweeps = []
        for mag_price, mag_vol, mag_strength in magnets:
            # Check both directions: price passes through the level
            swept_at = None
            for i in range(len(future_highs)):
                h, l = future_highs[i], future_lows[i]
                if h >= mag_price and l <= mag_price:
                    swept_at = i
                    break

            dist_pct = (mag_price - price) / price * 100
            sweeps.append({
                'price': mag_price,
                'strength': mag_strength,
                'volume': mag_vol,
                'dist_pct': dist_pct,
                'swept': swept_at is not None,
                'swept_at_bar': swept_at,
                'swept_time': str(future_times[swept_at]) if swept_at is not None else None,
            })

        # Filter to only swept magnets
        swept = [s for s in sweeps if s['swept']]
        if len(swept) < 2:
            idx += snapshot_interval
            continue

        # Sort by sweep time (which one got grabbed first)
        swept.sort(key=lambda x: x['swept_at_bar'])

        # Categorize the first grab
        first = swept[0]
        strengths = [s['strength'] for s in swept]
        all_prices = [s['price'] for s in swept]

        min_strength = min(strengths)
        max_strength = max(strengths)
        median_strength = np.median(strengths)

        # Was the first grab the smallest, biggest, or middle?
        if first['strength'] == min_strength:
            grab_type = 'SMALLEST_FIRST'
        elif first['strength'] == max_strength:
            grab_type = 'BIGGEST_FIRST'
        else:
            grab_type = 'MIDDLE_FIRST'

        # How many smaller ones were skipped?
        skipped_smaller = sum(1 for s in swept[1:] if s['strength'] < first['strength'])
        skipped_bigger = sum(1 for s in swept[1:] if s['strength'] > first['strength'])

        # Strength percentile of first grab
        strength_rank = sorted(strengths).index(first['strength']) + 1
        strength_pctile = strength_rank / len(strengths)

        # Distance analysis
        first_dist = abs(first['dist_pct'])
        all_dists = [abs(s['dist_pct']) for s in swept]

        # Did price go for nearest first?
        nearest = min(swept, key=lambda x: abs(x['dist_pct']))
        went_nearest_first = (first == nearest)

        # Count by strength buckets
        weak = [s for s in swept if s['strength'] < 1.5]
        medium = [s for s in swept if 1.5 <= s['strength'] < 2.5]
        strong = [s for s in swept if s['strength'] >= 2.5]

        result = {
            'timestamp': str(ts),
            'price': price,
            'n_magnets': len(magnets),
            'n_swept': len(swept),
            'first_grab_type': grab_type,
            'first_strength': first['strength'],
            'first_strength_pctile': strength_pctile,
            'skipped_smaller': skipped_smaller,
            'skipped_bigger': skipped_bigger,
            'went_nearest_first': went_nearest_first,
            'first_dist_pct': first['dist_pct'],
            'n_weak_swept': len(weak),
            'n_medium_swept': len(medium),
            'n_strong_swept': len(strong),
            'all_strengths': [round(s['strength'], 2) for s in swept],
            'all_dists': [round(s['dist_pct'], 2) for s in swept],
            'sweep_order': [(round(s['strength'], 2), round(s['dist_pct'], 2)) for s in swept],
        }
        results.append(result)
        total_snapshots += 1

        idx += snapshot_interval

    # ── Aggregate Stats ──
    if not results:
        print("No sweep events found.")
        return

    print(f"Total snapshots analyzed: {total_snapshots}")
    print(f"Snapshots with 2+ swept magnets: {len(results)}\n")

    # Grab type distribution
    grab_counts = defaultdict(int)
    for r in results:
        grab_counts[r['first_grab_type']] += 1

    print("─" * 55)
    print("  FIRST GRAB TYPE DISTRIBUTION")
    print("─" * 55)
    for gtype in ['SMALLEST_FIRST', 'MIDDLE_FIRST', 'BIGGEST_FIRST']:
        count = grab_counts.get(gtype, 0)
        pct = count / len(results) * 100
        bar = '█' * int(pct / 2)
        print(f"  {gtype:<18} {count:>4}  ({pct:5.1f}%)  {bar}")
    print()

    # Nearest-first rate
    nearest_count = sum(1 for r in results if r['went_nearest_first'])
    print(f"  Went for NEAREST first:   {nearest_count}/{len(results)} ({nearest_count/len(results)*100:.1f}%)")
    print()

    # Strength percentile of first grab
    pctiles = [r['first_strength_pctile'] for r in results]
    print("─" * 55)
    print("  STRENGTH PERCENTILE OF FIRST GRAB")
    print("  (0.0 = grabbed weakest, 1.0 = grabbed strongest)")
    print("─" * 55)
    print(f"  Mean:     {np.mean(pctiles):.3f}")
    print(f"  Median:   {np.median(pctiles):.3f}")
    print(f"  Std:      {np.std(pctiles):.3f}")
    print()

    # Distribution by bucket
    buckets = {'0-25%': 0, '25-50%': 0, '50-75%': 0, '75-100%': 0}
    for p in pctiles:
        if p < 0.25:
            buckets['0-25%'] += 1
        elif p < 0.50:
            buckets['25-50%'] += 1
        elif p < 0.75:
            buckets['50-75%'] += 1
        else:
            buckets['75-100%'] += 1

    for bucket, count in buckets.items():
        pct = count / len(results) * 100
        bar = '█' * int(pct / 2)
        print(f"  {bucket:<10} {count:>4}  ({pct:5.1f}%)  {bar}")
    print()

    # Skipped analysis
    total_skipped_smaller = sum(r['skipped_smaller'] for r in results)
    total_skipped_bigger = sum(r['skipped_bigger'] for r in results)
    total_skipped = total_skipped_smaller + total_skipped_bigger

    print("─" * 55)
    print("  WHAT GETS SKIPPED WHEN GOING FIRST?")
    print("─" * 55)
    if total_skipped > 0:
        print(f"  Smaller magnets skipped:  {total_skipped_smaller} ({total_skipped_smaller/total_skipped*100:.1f}%)")
        print(f"  Bigger magnets skipped:   {total_skipped_bigger} ({total_skipped_bigger/total_skipped*100:.1f}%)")
    print()

    # Average number of sweeps before hitting biggest
    avg_sweeps_before_big = []
    for r in results:
        order = r['sweep_order']
        if not order:
            continue
        max_s = max(s for s, _ in order)
        for i, (s, _) in enumerate(order):
            if s == max_s:
                avg_sweeps_before_big.append(i)
                break

    if avg_sweeps_before_big:
        print("─" * 55)
        print("  SWEEPS BEFORE BIGGEST MAGNET")
        print("─" * 55)
        print(f"  Mean sweeps before biggest: {np.mean(avg_sweeps_before_big):.2f}")
        print(f"  Median:                     {np.median(avg_sweeps_before_big):.0f}")
        print(f"  Biggest grabbed FIRST:      {sum(1 for x in avg_sweeps_before_big if x == 0)}/{len(avg_sweeps_before_big)} ({sum(1 for x in avg_sweeps_before_big if x == 0)/len(avg_sweeps_before_big)*100:.1f}%)")
        print()

    # Monthly breakdown
    print("─" * 55)
    print("  MONTHLY BREAKDOWN")
    print("─" * 55)
    print(f"  {'Month':<10} {'N':>4} {'Smallest%':>10} {'Middle%':>10} {'Biggest%':>10} {'AvgPctile':>10}")
    monthly = defaultdict(list)
    for r in results:
        month = r['timestamp'][:7]
        monthly[month].append(r)

    for month in sorted(monthly.keys()):
        rs = monthly[month]
        n = len(rs)
        smallest_pct = sum(1 for r in rs if r['first_grab_type'] == 'SMALLEST_FIRST') / n * 100
        middle_pct = sum(1 for r in rs if r['first_grab_type'] == 'MIDDLE_FIRST') / n * 100
        biggest_pct = sum(1 for r in rs if r['first_grab_type'] == 'BIGGEST_FIRST') / n * 100
        avg_pctile = np.mean([r['first_strength_pctile'] for r in rs])
        print(f"  {month:<10} {n:>4} {smallest_pct:>9.1f}% {middle_pct:>9.1f}% {biggest_pct:>9.1f}% {avg_pctile:>10.3f}")

    print()

    # Distance analysis
    print("─" * 55)
    print("  DISTANCE vs STRENGTH")
    print("─" * 55)
    # Did price tend to go for closer targets first?
    close_first = 0
    far_first = 0
    for r in results:
        order = r['sweep_order']
        if len(order) < 2:
            continue
        first_dist = abs(r['first_dist_pct'])
        other_dists = [abs(d) for d in r['all_dists'][1:]]
        if other_dists and first_dist <= np.median(other_dists):
            close_first += 1
        else:
            far_first += 1

    total = close_first + far_first
    if total > 0:
        print(f"  Closer targets grabbed first:  {close_first}/{total} ({close_first/total*100:.1f}%)")
        print(f"  Further targets grabbed first: {far_first}/{total} ({far_first/total*100:.1f}%)")
    print()

    # Conclusion
    print("═" * 55)
    print("  CONCLUSION")
    print("═" * 55)
    smallest_rate = grab_counts.get('SMALLEST_FIRST', 0) / len(results) * 100
    biggest_rate = grab_counts.get('BIGGEST_FIRST', 0) / len(results) * 100
    mean_pctile = np.mean(pctiles)

    if mean_pctile < 0.40:
        tendency = "Price STRONGLY tends to grab smaller/weaker liquidity first"
    elif mean_pctile < 0.50:
        tendency = "Price tends to grab smaller/weaker liquidity first"
    elif mean_pctile < 0.60:
        tendency = "Price has no strong preference (roughly random)"
    elif mean_pctile < 0.70:
        tendency = "Price tends to grab bigger/stronger liquidity first"
    else:
        tendency = "Price STRONGLY tends to grab bigger/stronger liquidity first"

    print(f"\n  {tendency}")
    print(f"  Smallest-first rate: {smallest_rate:.1f}%")
    print(f"  Biggest-first rate:  {biggest_rate:.1f}%")
    print(f"  Mean strength pctile of first grab: {mean_pctile:.3f}")
    print()

    if mean_pctile < 0.50:
        print("  → Smart money appears to clear small pools before")
        print("    targeting the big ones (liquidity engineering).")
    else:
        print("  → Price often reaches for the biggest pools directly,")
        print("    suggesting momentum-driven moves rather than")
        print("    systematic liquidity clearing.")
    print("═" * 55)


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--year', type=int, default=2026)
    parser.add_argument('--interval', type=int, default=96, help='Snapshot interval in bars')
    parser.add_argument('--forward', type=int, default=192, help='Lookforward bars')
    parser.add_argument('--csv', type=str, default=None)
    args = parser.parse_args()

    csv_path = args.csv or os.path.join(os.path.dirname(os.path.dirname(__file__)), 'eth_15m_merged.csv')
    df = load_data(csv_path)
    analyze_sweeps(df, year=args.year, snapshot_interval=args.interval, lookforward=args.forward)

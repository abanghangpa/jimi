#!/usr/bin/env python3
"""
Deep-dive: Distance-first liquidity grabs.

Finding: 93% of the time, price grabs the NEAREST target first.
This script analyzes:
  1. Distance distribution of nearest grabs
  2. What happens AFTER the nearest grab (continue? reverse?)
  3. Nearest-grab + biggest-grab correlation
  4. Is nearest grab a trap? (sweep then reverse)
  5. Actionable: given current price, what's the likely first target?

Usage:
    python scripts/liquidity_distance_analysis.py
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import numpy as np
import pandas as pd
from collections import defaultdict

from src.modules.m5_liquidation import build_volume_profile, find_magnets


def load_data(csv_path):
    df = pd.read_csv(csv_path)
    df['Open time'] = pd.to_datetime(df['Open time'])
    for c in ['Open', 'High', 'Low', 'Close', 'Volume',
              'Taker buy base asset volume', 'Taker buy quote asset volume']:
        df[c] = pd.to_numeric(df[c], errors='coerce')
    return df


def analyze_distance_first(df, year=2026, snapshot_interval=48, lookforward=96, lookback=672):
    df_year = df[df['Open time'].dt.year == year].copy().reset_index(drop=True)

    print(f"\n{'='*70}")
    print(f"  DISTANCE-FIRST LIQUIDITY ANALYSIS — {year}")
    print(f"  Snapshots every {snapshot_interval} bars | Lookforward: {lookforward} bars")
    print(f"{'='*70}\n")

    # ── Collect all sweep events ──
    events = []
    idx = lookback

    while idx < len(df_year) - lookforward:
        ts = df_year['Open time'].iloc[idx]
        price = float(df_year['Close'].iloc[idx])

        # Volume profile
        h = df_year['High'].values[:idx+1].astype(float)
        l = df_year['Low'].values[:idx+1].astype(float)
        c = df_year['Close'].values[:idx+1].astype(float)
        v = df_year['Volume'].values[:idx+1].astype(float)

        actual_lb = min(lookback, idx + 1)
        bin_centers, vol_profile, bin_edges = build_volume_profile(
            h[-actual_lb:], l[-actual_lb:], c[-actual_lb:], v[-actual_lb:],
            n_bins=50, lookback=actual_lb)

        if bin_centers is None:
            idx += snapshot_interval
            continue

        magnets = find_magnets(bin_centers, vol_profile, n_magnets=10, min_gap_pct=0.003)
        if len(magnets) < 3:
            idx += snapshot_interval
            continue

        # Look forward for sweeps
        f_highs = df_year['High'].values[idx+1:idx+1+lookforward].astype(float)
        f_lows = df_year['Low'].values[idx+1:idx+1+lookforward].astype(float)
        f_closes = df_year['Close'].values[idx+1:idx+1+lookforward].astype(float)
        f_times = df_year['Open time'].values[idx+1:idx+1+lookforward]

        # Find sweep order for each magnet
        swept_list = []
        for mag_price, mag_vol, mag_strength in magnets:
            dist_pct = (mag_price - price) / price * 100
            swept_at = None
            for i in range(len(f_highs)):
                if f_highs[i] >= mag_price and f_lows[i] <= mag_price:
                    swept_at = i
                    break
            if swept_at is not None:
                swept_list.append({
                    'price': mag_price,
                    'strength': mag_strength,
                    'dist_pct': dist_pct,
                    'abs_dist': abs(dist_pct),
                    'swept_at_bar': swept_at,
                    'swept_time': f_times[swept_at],
                    'direction': 'ABOVE' if dist_pct > 0 else 'BELOW',
                })

        if len(swept_list) < 2:
            idx += snapshot_interval
            continue

        swept_list.sort(key=lambda x: x['swept_at_bar'])
        first = swept_list[0]

        # ── After-sweep behavior ──
        # What does price do after grabbing the nearest?
        first_sweep_bar = first['swept_at_bar']
        post_highs = f_highs[first_sweep_bar:]
        post_lows = f_lows[first_sweep_bar:]
        post_closes = f_closes[first_sweep_bar:]

        if len(post_closes) < 4:
            idx += snapshot_interval
            continue

        # Price direction after first grab (next 4-12 bars = 1-3 hours)
        post_bars = min(12, len(post_closes) - 1)
        if post_bars >= 4:
            post_move_4 = (float(post_closes[3]) - float(post_closes[0])) / float(post_closes[0]) * 100
            post_move_12 = (float(post_closes[min(11, len(post_closes)-1)]) - float(post_closes[0])) / float(post_closes[0]) * 100
        else:
            post_move_4 = 0
            post_move_12 = 0

        # Did it sweep through or just touch?
        sweep_bar_h = float(f_highs[first_sweep_bar])
        sweep_bar_l = float(f_lows[first_sweep_bar])
        sweep_bar_range = sweep_bar_h - sweep_bar_l
        penetration = 0
        if first['direction'] == 'ABOVE':
            # Price was below, swept above
            penetration = (sweep_bar_h - first['price']) / (sweep_bar_range if sweep_bar_range > 0 else 1)
        else:
            penetration = (first['price'] - sweep_bar_l) / (sweep_bar_range if sweep_bar_range > 0 else 1)

        # Was the nearest also the weakest?
        all_strengths = [s['strength'] for s in swept_list]
        nearest_is_weakest = first['strength'] == min(all_strengths)
        nearest_is_strongest = first['strength'] == max(all_strengths)

        # How many others were swept AFTER the nearest?
        others_after = len(swept_list) - 1

        # Did the rest get swept in distance order?
        remaining = swept_list[1:]
        remaining_by_dist = sorted(remaining, key=lambda x: x['abs_dist'])
        remaining_by_time = sorted(remaining, key=lambda x: x['swept_at_bar'])
        distance_order_matches = [remaining_by_dist[i]['price'] == remaining_by_time[i]['price']
                                   for i in range(len(remaining))]
        subsequent_in_distance_order = sum(distance_order_matches) / len(distance_order_matches) if distance_order_matches else 0

        events.append({
            'timestamp': str(ts),
            'price': price,
            'n_magnets': len(magnets),
            'n_swept': len(swept_list),
            'first_price': first['price'],
            'first_dist_pct': first['dist_pct'],
            'first_abs_dist': first['abs_dist'],
            'first_strength': first['strength'],
            'first_direction': first['direction'],
            'first_sweep_bar': first_sweep_bar,
            'penetration': penetration,
            'post_move_4h': post_move_4,
            'post_move_12h': post_move_12,
            'nearest_is_weakest': nearest_is_weakest,
            'nearest_is_strongest': nearest_is_strongest,
            'others_after': others_after,
            'subsequent_distance_order': subsequent_in_distance_order,
            'all_distances': [round(s['abs_dist'], 2) for s in swept_list],
            'all_strengths': [round(s['strength'], 2) for s in swept_list],
            'sweep_order': [(round(s['abs_dist'], 2), round(s['strength'], 2)) for s in swept_list],
        })

        idx += snapshot_interval

    if not events:
        print("No events found.")
        return

    # ══════════════════════════════════════════════════
    #  REPORT
    # ══════════════════════════════════════════════════

    N = len(events)
    print(f"  Total events: {N}\n")

    # ── 1. Distance Distribution ──
    print("─" * 60)
    print("  1. NEAREST GRAB — DISTANCE DISTRIBUTION")
    print("─" * 60)
    dists = [e['first_abs_dist'] for e in events]
    print(f"  Mean:    {np.mean(dists):.3f}%")
    print(f"  Median:  {np.median(dists):.3f}%")
    print(f"  Std:     {np.std(dists):.3f}%")
    print(f"  Min:     {np.min(dists):.3f}%")
    print(f"  Max:     {np.max(dists):.3f}%")
    print()

    # Buckets
    buckets = [(0, 0.25), (0.25, 0.50), (0.50, 1.0), (1.0, 2.0), (2.0, 5.0)]
    print(f"  {'Range':<15} {'Count':>6} {'Pct':>7}  {'Bar'}")
    for lo, hi in buckets:
        count = sum(1 for d in dists if lo <= d < hi)
        pct = count / N * 100
        bar = '█' * int(pct / 2)
        print(f"  {lo:.2f}-{hi:.2f}%      {count:>6} {pct:>6.1f}%  {bar}")
    count_5plus = sum(1 for d in dists if d >= 5.0)
    if count_5plus:
        pct = count_5plus / N * 100
        bar = '█' * int(pct / 2)
        print(f"  5.0%+           {count_5plus:>6} {pct:>6.1f}%  {bar}")
    print()

    # ── 2. Direction of nearest grab ──
    above = sum(1 for e in events if e['first_direction'] == 'ABOVE')
    below = N - above
    print("─" * 60)
    print("  2. NEAREST GRAB DIRECTION")
    print("─" * 60)
    print(f"  Above price (shorts squeezed):  {above} ({above/N*100:.1f}%)")
    print(f"  Below price (longs squeezed):   {below} ({below/N*100:.1f}%)")
    print()

    # ── 3. Post-grab behavior ──
    print("─" * 60)
    print("  3. WHAT HAPPENS AFTER NEAREST GRAB?")
    print("─" * 60)
    post4 = [e['post_move_4h'] for e in events]
    post12 = [e['post_move_12h'] for e in events]

    # After grabbing nearest above: does price continue up or reverse?
    above_events = [e for e in events if e['first_direction'] == 'ABOVE']
    below_events = [e for e in events if e['first_direction'] == 'BELOW']

    if above_events:
        above_post4 = [e['post_move_4h'] for e in above_events]
        above_post12 = [e['post_move_12h'] for e in above_events]
        cont_up_4 = sum(1 for m in above_post4 if m > 0.05)
        rev_down_4 = sum(1 for m in above_post4 if m < -0.05)
        flat_4 = len(above_post4) - cont_up_4 - rev_down_4

        print(f"\n  After grabbing ABOVE (n={len(above_events)}):")
        print(f"    4h later:  Continue UP: {cont_up_4} ({cont_up_4/len(above_events)*100:.1f}%) | "
              f"Reverse DOWN: {rev_down_4} ({rev_down_4/len(above_events)*100:.1f}%) | "
              f"Flat: {flat_4} ({flat_4/len(above_events)*100:.1f}%)")
        cont_up_12 = sum(1 for m in above_post12 if m > 0.1)
        rev_down_12 = sum(1 for m in above_post12 if m < -0.1)
        flat_12 = len(above_post12) - cont_up_12 - rev_down_12
        print(f"    12h later: Continue UP: {cont_up_12} ({cont_up_12/len(above_events)*100:.1f}%) | "
              f"Reverse DOWN: {rev_down_12} ({rev_down_12/len(above_events)*100:.1f}%) | "
              f"Flat: {flat_12} ({flat_12/len(above_events)*100:.1f}%)")
        print(f"    Avg 4h move:  {np.mean(above_post4):+.3f}%")
        print(f"    Avg 12h move: {np.mean(above_post12):+.3f}%")

    if below_events:
        below_post4 = [e['post_move_4h'] for e in below_events]
        below_post12 = [e['post_move_12h'] for e in below_events]
        cont_down_4 = sum(1 for m in below_post4 if m < -0.05)
        rev_up_4 = sum(1 for m in below_post4 if m > 0.05)
        flat_4 = len(below_post4) - cont_down_4 - rev_up_4

        print(f"\n  After grabbing BELOW (n={len(below_events)}):")
        print(f"    4h later:  Continue DOWN: {cont_down_4} ({cont_down_4/len(below_events)*100:.1f}%) | "
              f"Reverse UP: {rev_up_4} ({rev_up_4/len(below_events)*100:.1f}%) | "
              f"Flat: {flat_4} ({flat_4/len(below_events)*100:.1f}%)")
        cont_down_12 = sum(1 for m in below_post12 if m < -0.1)
        rev_up_12 = sum(1 for m in below_post12 if m > 0.1)
        flat_12 = len(below_post12) - cont_down_12 - rev_up_12
        print(f"    12h later: Continue DOWN: {cont_down_12} ({cont_down_12/len(below_events)*100:.1f}%) | "
              f"Reverse UP: {rev_up_12} ({rev_up_12/len(below_events)*100:.1f}%) | "
              f"Flat: {flat_12} ({flat_12/len(below_events)*100:.1f}%)")
        print(f"    Avg 4h move:  {np.mean(below_post4):+.3f}%")
        print(f"    Avg 12h move: {np.mean(below_post12):+.3f}%")
    print()

    # ── 4. Penetration depth ──
    print("─" * 60)
    print("  4. SWEEP PENETRATION DEPTH")
    print("─" * 60)
    pens = [e['penetration'] for e in events]
    print(f"  Mean penetration:   {np.mean(pens):.3f}")
    print(f"  Median:             {np.median(pens):.3f}")
    deep = sum(1 for p in pens if p > 0.7)
    shallow = sum(1 for p in pens if p < 0.3)
    mid = N - deep - shallow
    print(f"\n  Deep sweep (>70% of bar):   {deep} ({deep/N*100:.1f}%)")
    print(f"  Mid sweep (30-70%):         {mid} ({mid/N*100:.1f}%)")
    print(f"  Shallow sweep (<30%):       {shallow} ({shallow/N*100:.1f}%)")
    print()

    # ── 5. Strength of nearest grab ──
    print("─" * 60)
    print("  5. IS THE NEAREST ALSO THE WEAKEST?")
    print("─" * 60)
    weakest = sum(1 for e in events if e['nearest_is_weakest'])
    strongest = sum(1 for e in events if e['nearest_is_strongest'])
    middle = N - weakest - strongest
    print(f"  Nearest = weakest:   {weakest} ({weakest/N*100:.1f}%)")
    print(f"  Nearest = middle:    {middle} ({middle/N*100:.1f}%)")
    print(f"  Nearest = strongest: {strongest} ({strongest/N*100:.1f}%)")
    print()

    # ── 6. Subsequent sweeps order ──
    print("─" * 60)
    print("  6. AFTER NEAREST, DO SUBSEQUENT SWEEPS FOLLOW DISTANCE ORDER?")
    print("─" * 60)
    sub_orders = [e['subsequent_distance_order'] for e in events if e['others_after'] >= 2]
    if sub_orders:
        print(f"  Mean distance-order match: {np.mean(sub_orders)*100:.1f}%")
        print(f"  Median:                    {np.median(sub_orders)*100:.1f}%")
        high_match = sum(1 for s in sub_orders if s >= 0.8)
        low_match = sum(1 for s in sub_orders if s <= 0.2)
        print(f"  Perfect match (≥80%):      {high_match}/{len(sub_orders)} ({high_match/len(sub_orders)*100:.1f}%)")
        print(f"  Random order (≤20%):       {low_match}/{len(sub_orders)} ({low_match/len(sub_orders)*100:.1f}%)")
    print()

    # ── 7. Combined: nearest + deep sweep → reversal? ──
    print("─" * 60)
    print("  7. TRAP DETECTION: NEAREST + DEEP → REVERSAL?")
    print("─" * 60)

    # Deep nearest sweep above + price drops after = bull trap
    above_deep = [e for e in events if e['first_direction'] == 'ABOVE' and e['penetration'] > 0.7]
    above_shallow = [e for e in events if e['first_direction'] == 'ABOVE' and e['penetration'] < 0.3]

    if above_deep:
        rev = sum(1 for e in above_deep if e['post_move_4h'] < -0.05)
        print(f"  Above + deep sweep (n={len(above_deep)}): reversal {rev}/{len(above_deep)} ({rev/len(above_deep)*100:.1f}%)")
    if above_shallow:
        rev = sum(1 for e in above_shallow if e['post_move_4h'] < -0.05)
        print(f"  Above + shallow sweep (n={len(above_shallow)}): reversal {rev}/{len(above_shallow)} ({rev/len(above_shallow)*100:.1f}%)")

    below_deep = [e for e in events if e['first_direction'] == 'BELOW' and e['penetration'] > 0.7]
    below_shallow = [e for e in events if e['first_direction'] == 'BELOW' and e['penetration'] < 0.3]

    if below_deep:
        rev = sum(1 for e in below_deep if e['post_move_4h'] > 0.05)
        print(f"  Below + deep sweep (n={len(below_deep)}): reversal {rev}/{len(below_deep)} ({rev/len(below_deep)*100:.1f}%)")
    if below_shallow:
        rev = sum(1 for e in below_shallow if e['post_move_4h'] > 0.05)
        print(f"  Below + shallow sweep (n={len(below_shallow)}): reversal {rev}/{len(below_shallow)} ({rev/len(below_shallow)*100:.1f}%)")
    print()

    # ── 8. Strength vs continuation ──
    print("─" * 60)
    print("  8. DOES NEAREST STRENGTH PREDICT CONTINUATION?")
    print("─" * 60)
    # Split by strength
    weak_first = [e for e in events if e['first_strength'] < 1.5]
    medium_first = [e for e in events if 1.5 <= e['first_strength'] < 2.5]
    strong_first = [e for e in events if e['first_strength'] >= 2.5]

    for label, subset in [('Weak (<1.5x)', weak_first), ('Medium (1.5-2.5x)', medium_first), ('Strong (>2.5x)', strong_first)]:
        if not subset:
            continue
        # For above grabs: continuation = price stays above
        above_sub = [e for e in subset if e['first_direction'] == 'ABOVE']
        below_sub = [e for e in subset if e['first_direction'] == 'BELOW']

        cont_above = 0
        cont_below = 0
        if above_sub:
            cont_above = sum(1 for e in above_sub if e['post_move_4h'] > -0.05)
        if below_sub:
            cont_below = sum(1 for e in below_sub if e['post_move_4h'] < 0.05)

        total_cont = cont_above + cont_below
        total = len(above_sub) + len(below_sub)
        avg_post = np.mean([e['post_move_4h'] for e in subset])

        print(f"  {label:<22} n={len(subset):>3}  continuation={total_cont}/{total} ({total_cont/total*100:.0f}%)  avg_4h={avg_post:+.3f}%")
    print()

    # ── 9. Dist-to-nearest vs outcome ──
    print("─" * 60)
    print("  9. DOES DISTANCE TO NEAREST PREDICT OUTCOME?")
    print("─" * 60)
    close_events = [e for e in events if e['first_abs_dist'] < 0.5]
    mid_events = [e for e in events if 0.5 <= e['first_abs_dist'] < 1.5]
    far_events = [e for e in events if e['first_abs_dist'] >= 1.5]

    for label, subset in [('Close (<0.5%)', close_events), ('Mid (0.5-1.5%)', mid_events), ('Far (>1.5%)', far_events)]:
        if not subset:
            continue
        avg_post4 = np.mean([e['post_move_4h'] for e in subset])
        avg_post12 = np.mean([e['post_move_12h'] for e in subset])
        avg_pen = np.mean([e['penetration'] for e in subset])
        avg_others = np.mean([e['others_after'] for e in subset])

        print(f"  {label:<18} n={len(subset):>3}  avg_4h={avg_post4:+.3f}%  avg_12h={avg_post12:+.3f}%  pen={avg_pen:.2f}  others_after={avg_others:.1f}")
    print()

    # ══════════════════════════════════════════════════
    #  ACTIONABLE SUMMARY
    # ══════════════════════════════════════════════════
    print("═" * 60)
    print("  ACTIONABLE SUMMARY")
    print("═" * 60)

    median_dist = np.median(dists)
    mean_dist = np.mean(dists)

    print(f"""
  1. FIRST TARGET IS ALWAYS NEAREST ({above/N*100:.0f}% above / {below/N*100:.0f}% below)
     Median distance: {median_dist:.2f}% | Mean: {mean_dist:.2f}%
     → Use this to SET YOUR ENTRY. The nearest liquidity pool
       is your highest-probability first target.

  2. AFTER SWEEPING NEAREST:
     Above grabs: {sum(1 for e in above_events if e['post_move_4h'] > 0.05)/len(above_events)*100:.0f}% continue up at 4h
     Below grabs: {sum(1 for e in below_events if e['post_move_4h'] < -0.05)/len(below_events)*100:.0f}% continue down at 4h
     → Don't fade the nearest grab — continuation is more likely.

  3. SUBSEQUENT SWEEPS: {np.mean(sub_orders)*100:.0f}% follow distance order
     → After nearest is cleared, next target = next nearest.
       Price walks through liquidity in order of proximity.

  4. TRAP SIGNAL: {sum(1 for e in above_deep if e['post_move_4h'] < -0.05)/len(above_deep)*100:.0f}% of deep above sweeps reverse
     → If nearest grab has HIGH penetration (>70%), expect
       a reversal. Shallow grabs → continuation.

  5. WEAK NEAREST GRABS → STRONGEST continuation
     Weak targets: continuation rate = {sum(1 for e in weak_first if (e['first_direction']=='ABOVE' and e['post_move_4h']>-0.05) or (e['first_direction']=='BELOW' and e['post_move_4h']<0.05))/len(weak_first)*100:.0f}%
     → Price blows through weak pools. Don't expect reversal.
""")
    print("═" * 60)


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--year', type=int, default=2026)
    parser.add_argument('--interval', type=int, default=48)
    parser.add_argument('--forward', type=int, default=96)
    args = parser.parse_args()

    csv_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'eth_15m_merged.csv')
    df = load_data(csv_path)
    analyze_distance_first(df, year=args.year, snapshot_interval=args.interval, lookforward=args.forward)

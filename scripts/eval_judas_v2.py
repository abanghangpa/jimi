#!/usr/bin/env python3
"""
Judas Sweep Evaluator v2 — Tighter detection matching current setup.

Current setup characteristics:
- Price in narrow range ($2301-$2305, ~0.2% range)
- Resistance at $2305 with stop clusters at $2311-$2315
- Expected sweep: 0.1-0.5% above resistance
- Context: bull structure + bearish sellers + low volume compression
- Expected reversal: within 12-48 bars (3h-12h)

This script detects similar setups historically.
"""

import sys, os
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from src.utils.indicators import calc_ema, calc_rsi, calc_atr


def find_swing_highs(highs, period=3):
    swings = []
    for i in range(period, len(highs) - period):
        if all(highs[i] >= highs[i-j] for j in range(1, period+1)) and \
           all(highs[i] >= highs[i+j] for j in range(1, period+1)):
            swings.append(i)
    return swings


def measure_compression(highs, lows, lookback=48):
    """Measure range compression over lookback bars."""
    h = np.max(highs[-lookback:])
    l = np.min(lows[-lookback:])
    if l == 0:
        return 1.0
    return (h - l) / l * 100


def detect_judas_v2(df, config):
    """Detect Judas sweeps with tighter filters matching current setup."""
    
    closes = df['Close'].values
    highs = df['High'].values
    lows = df['Low'].values
    volumes = df['Volume'].values
    
    # Indicators
    ema21 = calc_ema(df['Close'], 21).values
    ema55 = calc_ema(df['Close'], 55).values
    rsi = calc_rsi(df['Close'], 14).values
    atr = calc_atr(df['High'], df['Low'], df['Close'], 14).values
    vol_ma20 = pd.Series(volumes).rolling(20).mean().values
    
    # Taker
    taker_base = df['Taker buy base asset volume'].values
    total_vol = df['Volume'].values
    taker_ratio = np.where(total_vol > 0, taker_base / total_vol, 0.5)
    
    sweep_min_pct = config['sweep_min_pct'] / 100
    sweep_max_pct = config['sweep_max_pct'] / 100
    compression_max = config['compression_max_pct']
    min_range_bars = config['min_range_bars']
    reversal_window = config['reversal_window']
    taker_threshold = config['taker_threshold']
    
    events = []
    
    # Step 1: Find resistance clusters from rolling swing highs
    # Pre-compute swing highs per window
    swing_period = 3
    all_swing_highs = find_swing_highs(highs, period=swing_period)
    
    for i in range(200, len(df) - reversal_window):
        # Check compression: is price in a narrow range?
        compression = measure_compression(highs[max(0,i-min_range_bars):i+1], 
                                          lows[max(0,i-min_range_bars):i+1], 
                                          lookback=min_range_bars)
        
        if compression > compression_max:
            continue  # Not compressed enough
        
        # Find nearby resistance from recent swing highs
        recent_swings = [s for s in all_swing_highs 
                         if i - 200 <= s <= i - 3]  # at least 3 bars old
        if len(recent_swings) < 3:
            continue
        
        # Cluster swing highs
        sh_prices = highs[recent_swings]
        clusters = []
        current = [sh_prices[0]]
        for p in sh_prices[1:]:
            if (p - current[-1]) / current[-1] < 0.002:  # 0.2% cluster
                current.append(p)
            else:
                if len(current) >= 2:
                    clusters.append(np.mean(current))
                current = [p]
        if len(current) >= 2:
            clusters.append(np.mean(current))
        
        if not clusters:
            continue
        
        # Get nearest resistance above current price
        current_price = closes[i]
        resistances = [c for c in clusters if c > current_price]
        if not resistances:
            continue
        
        nearest_res = min(resistances)
        dist_to_res = (nearest_res - current_price) / current_price
        
        # Must be close to resistance (within 0.5%)
        if dist_to_res > 0.005:
            continue
        
        # Check if current bar sweeps above resistance
        sweep_amount = (highs[i] - nearest_res) / nearest_res
        if sweep_amount < sweep_min_pct or sweep_amount > sweep_max_pct:
            continue
        
        # Context checks
        bull_structure = ema21[i] > ema55[i] if not (np.isnan(ema21[i]) or np.isnan(ema55[i])) else False
        
        # 4h taker average (16 bars)
        taker_4h = np.mean(taker_ratio[max(0,i-16):i]) if i > 16 else 0.5
        sellers_dominant = taker_4h < taker_threshold
        
        # Volume check — low vol compression
        vol_ratio = volumes[i] / vol_ma20[i] if vol_ma20[i] > 0 else 1.0
        low_volume = vol_ratio < 1.2
        
        # RSI
        rsi_val = rsi[i] if not np.isnan(rsi[i]) else 50
        
        # Is this a Judas context?
        is_judas = bull_structure and sellers_dominant
        
        # Measure reversal
        entry_bar = i
        end_bar = min(i + reversal_window, len(df) - 1)
        future_lows = lows[entry_bar + 1: end_bar + 1]
        future_closes = closes[entry_bar + 1: end_bar + 1]
        future_highs = highs[entry_bar + 1: end_bar + 1]
        
        if len(future_lows) == 0:
            continue
        
        min_low = np.min(future_lows)
        max_high_after = np.max(future_highs) if len(future_highs) > 0 else closes[i]
        
        reversal_pct = (closes[i] - min_low) / closes[i] * 100
        extension_pct = (max_high_after - closes[i]) / closes[i] * 100
        
        # Time to reversal (first bar that drops 0.3%+)
        time_to_rev = None
        for j in range(len(future_lows)):
            drop = (closes[i] - future_lows[j]) / closes[i] * 100
            if drop >= 0.3:
                time_to_rev = j + 1
                break
        
        # Did it reverse meaningfully? (>0.3% drop from entry)
        reversed_03 = reversal_pct >= 0.3
        reversed_05 = reversal_pct >= 0.5
        reversed_10 = reversal_pct >= 1.0
        
        events.append({
            'timestamp': df.index[i],
            'price': closes[i],
            'resistance': nearest_res,
            'sweep_pct': sweep_amount * 100,
            'compression': compression,
            'bull_structure': bull_structure,
            'taker_4h': taker_4h,
            'sellers_dominant': sellers_dominant,
            'vol_ratio': vol_ratio,
            'low_volume': low_volume,
            'rsi': rsi_val,
            'is_judas': is_judas,
            'reversal_pct': reversal_pct,
            'extension_pct': extension_pct,
            'reversed_03': reversed_03,
            'reversed_05': reversed_05,
            'reversed_10': reversed_10,
            'time_to_rev': time_to_rev,
        })
    
    return events


def analyze(events, config):
    if not events:
        print("No events found with current parameters.")
        return
    
    df = pd.DataFrame(events)
    
    print("=" * 70)
    print("  JUDAS SWEEP EVALUATION v2")
    print("  (Tight filters: compression + near-resistance + sweep)")
    print("=" * 70)
    print(f"\n  Parameters:")
    print(f"    Sweep range:     {config['sweep_min_pct']}%-{config['sweep_max_pct']}%")
    print(f"    Compression:     <{config['compression_max_pct']}% range")
    print(f"    Range bars:      {config['min_range_bars']} (={config['min_range_bars']*15}min)")
    print(f"    Reversal window: {config['reversal_window']} bars (={config['reversal_window']*15}min)")
    print(f"    Taker threshold: <{config['taker_threshold']}")
    
    print(f"\n  Total sweeps: {len(df)}")
    print(f"  Date range: {df['timestamp'].min()} → {df['timestamp'].max()}")
    
    # Split by context
    judas = df[df['is_judas']]
    non_judas = df[~df['is_judas']]
    
    print(f"\n{'─' * 70}")
    print("  ALL SWEEPS (compressed range + near resistance)")
    print(f"{'─' * 70}")
    _print_stats(df, "All")
    
    if len(judas) > 0:
        print(f"\n{'─' * 70}")
        print("  JUDAS CONTEXT (bull structure + bearish sellers)")
        print(f"{'─' * 70}")
        _print_stats(judas, "Judas")
    
    if len(non_judas) > 0:
        print(f"\n{'─' * 70}")
        print("  NON-JUDAS (other context)")
        print(f"{'─' * 70}")
        _print_stats(non_judas, "Non-Judas")
    
    # Comparison
    if len(judas) > 0 and len(non_judas) > 0:
        print(f"\n{'─' * 70}")
        print("  JUDAS vs NON-JUDAS COMPARISON")
        print(f"{'─' * 70}")
        for metric in ['reversed_03', 'reversed_05', 'reversed_10']:
            j_rate = judas[metric].mean() * 100
            n_rate = non_judas[metric].mean() * 100
            label = metric.replace('reversed_', '>')
            print(f"  {label} reversal:  Judas={j_rate:.1f}%  Non-Judas={n_rate:.1f}%  "
                  f"{'✅ Judas better' if j_rate > n_rate else '❌ Non-Judas better'}")
        
        j_rev = judas[judas['reversed_03']]['reversal_pct']
        n_rev = non_judas[non_judas['reversed_03']]['reversal_pct']
        if len(j_rev) > 0 and len(n_rev) > 0:
            print(f"  Avg drop:       Judas={j_rev.mean():.2f}%  Non-Judas={n_rev.mean():.2f}%")
            print(f"  Median drop:    Judas={j_rev.median():.2f}%  Non-Judas={n_rev.median():.2f}%")
    
    # Time to reversal
    if len(judas) > 0 and judas['time_to_rev'].notna().any():
        print(f"\n{'─' * 70}")
        print("  TIME TO REVERSAL (Judas context, bars after sweep)")
        print(f"{'─' * 70}")
        ttr = judas['time_to_rev'].dropna()
        print(f"  Mean:    {ttr.mean():.1f} bars ({ttr.mean()*15:.0f} min)")
        print(f"  Median:  {ttr.median():.1f} bars ({ttr.median()*15:.0f} min)")
        print(f"  25th:    {ttr.quantile(0.25):.0f} bars")
        print(f"  75th:    {ttr.quantile(0.75):.0f} bars")
        print(f"  Max:     {ttr.max():.0f} bars ({ttr.max()*15:.0f} min)")
    
    # By sweep depth
    print(f"\n{'─' * 70}")
    print("  BY SWEEP DEPTH (Judas context)")
    print(f"{'─' * 70}")
    if len(judas) > 0:
        for label, lo, hi in [('Shallow (0.1-0.2%)', 0.1, 0.2), ('Medium (0.2-0.3%)', 0.2, 0.3),
                               ('Deep (0.3-0.5%)', 0.3, 0.5), ('Extreme (0.5-1.0%)', 0.5, 1.0)]:
            subset = judas[(judas['sweep_pct'] >= lo) & (judas['sweep_pct'] < hi)]
            if len(subset) > 0:
                r = subset['reversed_03'].mean() * 100
                a = subset[subset['reversed_03']]['reversal_pct'].mean() if subset['reversed_03'].any() else 0
                print(f"  {label:25s}  n={len(subset):4d}  >0.3% rev={r:.1f}%  avg_drop={a:.2f}%")
    
    # By compression level
    print(f"\n{'─' * 70}")
    print("  BY COMPRESSION LEVEL (Judas context)")
    print(f"{'─' * 70}")
    if len(judas) > 0:
        for label, lo, hi in [('Ultra-tight (<0.3%)', 0, 0.3), ('Tight (0.3-0.5%)', 0.3, 0.5),
                               ('Moderate (0.5-1.0%)', 0.5, 1.0), ('Wide (1.0-2.0%)', 1.0, 2.0)]:
            subset = judas[(judas['compression'] >= lo) & (judas['compression'] < hi)]
            if len(subset) > 0:
                r = subset['reversed_03'].mean() * 100
                a = subset[subset['reversed_03']]['reversal_pct'].mean() if subset['reversed_03'].any() else 0
                print(f"  {label:25s}  n={len(subset):4d}  >0.3% rev={r:.1f}%  avg_drop={a:.2f}%")
    
    # Recent events (2025-2026)
    print(f"\n{'─' * 70}")
    print("  RECENT JUDAS SWEEPS (2025-2026)")
    print(f"{'─' * 70}")
    recent = judas[judas['timestamp'] >= '2025-01-01']
    if len(recent) > 0:
        _print_stats(recent, "Recent")
        print(f"\n  Last 20 events:")
        for _, row in recent.tail(20).iterrows():
            rev_mark = '✅' if row['reversed_03'] else '❌'
            ttr_str = f"{row['time_to_rev']:.0f}bars" if pd.notna(row['time_to_rev']) else 'n/a'
            print(f"  {row['timestamp']}  ${row['price']:.2f}  "
                  f"sweep={row['sweep_pct']:.2f}%  comp={row['compression']:.2f}%  "
                  f"drop={row['reversal_pct']:.2f}%  taker={row['taker_4h']:.3f}  "
                  f"{rev_mark} ({ttr_str})")
    else:
        print("  No recent events.")
    
    # Current setup comparison
    print(f"\n{'─' * 70}")
    print("  CURRENT SETUP ANALYSIS")
    print(f"{'─' * 70}")
    print(f"  Price:           $2,303")
    print(f"  Resistance:      $2,305 (+0.09%)")
    print(f"  Sweep target:    $2,311-$2,315 (+0.3-0.5%)")
    print(f"  Compression:     ~0.2% (48 bars)")
    print(f"  Bull structure:  Yes")
    print(f"  Sellers:         Yes (taker ~0.46)")
    print(f"  Volume:          Low (0.08x)")
    
    # Find similar: compression <0.5%, sweep 0.2-0.5%, judas context
    similar = judas[(judas['compression'] < 0.5) & (judas['sweep_pct'] >= 0.2) & (judas['sweep_pct'] < 0.5)]
    if len(similar) > 0:
        print(f"\n  Similar setups (compressed <0.5%, sweep 0.2-0.5%, judas):")
        print(f"    Count:           {len(similar)}")
        print(f"    >0.3% reversal:  {similar['reversed_03'].mean()*100:.1f}%")
        print(f"    >0.5% reversal:  {similar['reversed_05'].mean()*100:.1f}%")
        print(f"    >1.0% reversal:  {similar['reversed_10'].mean()*100:.1f}%")
        winners = similar[similar['reversed_03']]
        if len(winners) > 0:
            print(f"    Avg drop:        {winners['reversal_pct'].mean():.2f}%")
            print(f"    Median drop:     {winners['reversal_pct'].median():.2f}%")
            print(f"    Max drop:        {winners['reversal_pct'].max():.2f}%")
        losers = similar[~similar['reversed_03']]
        if len(losers) > 0:
            print(f"    Avg extension:   {losers['extension_pct'].mean():.2f}%")
            print(f"    Max extension:   {losers['extension_pct'].max():.2f}%")
        
        # R:R
        if len(winners) > 0 and len(losers) > 0:
            avg_win = winners['reversal_pct'].mean()
            avg_loss = losers['extension_pct'].mean()
            if avg_loss > 0:
                print(f"    R:R (SHORT):     {avg_win/avg_loss:.2f}x")
    else:
        print(f"\n  No exact matches found. Checking broader...")
        similar_broad = judas[(judas['compression'] < 1.0) & (judas['sweep_pct'] >= 0.1)]
        if len(similar_broad) > 0:
            print(f"  Broader matches (comp<1%, sweep>0.1%): {len(similar_broad)}")
            print(f"  >0.3% reversal: {similar_broad['reversed_03'].mean()*100:.1f}%")
    
    print(f"\n{'=' * 70}")


def _print_stats(subset, label):
    n = len(subset)
    r03 = subset['reversed_03'].mean() * 100
    r05 = subset['reversed_05'].mean() * 100
    r10 = subset['reversed_10'].mean() * 100
    
    winners = subset[subset['reversed_03']]
    losers = subset[~subset['reversed_03']]
    
    avg_drop = winners['reversal_pct'].mean() if len(winners) > 0 else 0
    med_drop = winners['reversal_pct'].median() if len(winners) > 0 else 0
    avg_ext = losers['extension_pct'].mean() if len(losers) > 0 else 0
    
    print(f"  Count:           {n}")
    print(f"  >0.3% reversal:  {r03:.1f}%")
    print(f"  >0.5% reversal:  {r05:.1f}%")
    print(f"  >1.0% reversal:  {r10:.1f}%")
    print(f"  Avg drop:        {avg_drop:.2f}%")
    print(f"  Median drop:     {med_drop:.2f}%")
    print(f"  Avg extension:   {avg_ext:.2f}% (when no reversal)")


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--csv', default='eth_15m_merged.csv')
    parser.add_argument('--sweep-min', type=float, default=0.1, dest='sweep_min_pct')
    parser.add_argument('--sweep-max', type=float, default=1.0, dest='sweep_max_pct')
    parser.add_argument('--compression-max', type=float, default=2.0, dest='compression_max_pct')
    parser.add_argument('--range-bars', type=int, default=48, dest='min_range_bars')
    parser.add_argument('--reversal-window', type=int, default=48, dest='reversal_window')
    parser.add_argument('--taker', type=float, default=0.48, dest='taker_threshold')
    args = parser.parse_args()
    
    config = {
        'sweep_min_pct': args.sweep_min_pct,
        'sweep_max_pct': args.sweep_max_pct,
        'compression_max_pct': args.compression_max_pct,
        'min_range_bars': args.min_range_bars,
        'reversal_window': args.reversal_window,
        'taker_threshold': args.taker_threshold,
    }
    
    csv_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), args.csv)
    print(f"  Loading {csv_path}...")
    df = pd.read_csv(csv_path, parse_dates=['Open time'], index_col='Open time')
    print(f"  Loaded {len(df)} bars ({df.index[0]} → {df.index[-1]})")
    
    events = detect_judas_v2(df, config)
    analyze(events, config)


if __name__ == '__main__':
    main()

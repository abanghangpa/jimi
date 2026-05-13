#!/usr/bin/env python3
"""
Judas Sweep Performance Evaluator

Detects historical Judas sweep setups (bullish structure + bearish smart money
+ sweep above resistance) and measures reversal performance.

Usage:
    python3 scripts/eval_judas_sweep.py
    python3 scripts/eval_judas_sweep.py --sweep-pct 0.3 --hold-bars 3
"""

import sys, os, argparse
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from src.utils.indicators import calc_ema, calc_rsi, calc_atr


def find_swing_highs(highs, period=3):
    """Find fractal swing highs."""
    swings = []
    for i in range(period, len(highs) - period):
        if all(highs[i] >= highs[i-j] for j in range(1, period+1)) and \
           all(highs[i] >= highs[i+j] for j in range(1, period+1)):
            swings.append(i)
    return swings


def find_resistance_levels(df, lookback=960, zone_width_pct=0.005):
    """Find resistance zones from volume-weighted price clusters."""
    levels = []
    highs = df['High'].values
    volumes = df['Volume'].values
    
    # Use swing highs from recent bars
    swing_highs = find_swing_highs(highs, period=3)
    recent = [s for s in swing_highs if s >= len(df) - lookback]
    
    if not recent:
        return levels
    
    # Cluster nearby levels
    prices = [highs[s] for s in recent]
    prices.sort()
    
    clusters = []
    current_cluster = [prices[0]]
    for p in prices[1:]:
        if (p - current_cluster[-1]) / current_cluster[-1] < zone_width_pct:
            current_cluster.append(p)
        else:
            clusters.append(current_cluster)
            current_cluster = [p]
    clusters.append(current_cluster)
    
    for cluster in clusters:
        if len(cluster) >= 2:
            avg_price = np.mean(cluster)
            touches = len(cluster)
            levels.append({
                'price': avg_price,
                'touches': touches,
                'strength': touches
            })
    
    levels.sort(key=lambda x: x['strength'], reverse=True)
    return levels[:10]


def detect_judas_sweeps(df, config):
    """Detect Judas sweep setups in historical data.
    
    A Judas sweep occurs when:
    1. Price is near a resistance level
    2. Price sweeps above that resistance (grabs stops)
    3. Price then reverses and closes back below
    4. Context: bullish structure + bearish smart money signals
    """
    sweep_pct = config['sweep_pct'] / 100  # min % above level to count as sweep
    hold_bars = config['hold_bars']        # bars price must stay above
    reverse_bars = config['reverse_bars']  # bars to measure reversal
    min_resistance_touches = config['min_touches']
    
    closes = df['Close'].values
    highs = df['High'].values
    lows = df['Low'].values
    volumes = df['Volume'].values
    
    # Calculate indicators for context
    ema21 = calc_ema(df['Close'], 21).values
    ema55 = calc_ema(df['Close'], 55).values
    rsi = calc_rsi(df['Close'], 14).values
    atr = calc_atr(df['High'], df['Low'], df['Close'], 14).values
    
    # Taker ratio
    taker_base = df['Taker buy base asset volume'].values
    total_vol = df['Volume'].values
    taker_ratio = np.where(total_vol > 0, taker_base / total_vol, 0.5)
    
    events = []
    lookback_window = 960  # 10 days of 15m bars for resistance detection
    
    # Scan with rolling window
    for i in range(lookback_window + 50, len(df) - reverse_bars):
        # Build resistance levels from trailing window
        window_start = max(0, i - lookback_window)
        window_df = df.iloc[window_start:i].copy()
        
        # Simple swing high detection in window
        swing_highs = find_swing_highs(window_df['High'].values, period=3)
        if len(swing_highs) < 3:
            continue
        
        # Get resistance levels (cluster swing highs)
        sh_prices = window_df['High'].values[swing_highs]
        sh_prices_sorted = np.sort(sh_prices)
        
        # Cluster
        clusters = []
        current = [sh_prices_sorted[0]]
        for p in sh_prices_sorted[1:]:
            if (p - current[-1]) / current[-1] < 0.003:
                current.append(p)
            else:
                if len(current) >= min_resistance_touches:
                    clusters.append(np.mean(current))
                current = [p]
        if len(current) >= min_resistance_touches:
            clusters.append(np.mean(current))
        
        if not clusters:
            continue
        
        # Check if current bar sweeps above any resistance
        current_high = highs[i]
        current_close = closes[i]
        
        for res_level in clusters:
            sweep_threshold = res_level * (1 + sweep_pct)
            
            if current_high < sweep_threshold:
                continue
            
            # High swept above resistance — check if it held above for N bars
            above_count = 0
            max_above = 0
            for j in range(1, min(hold_bars + 1, len(df) - i)):
                if highs[i + j] > res_level:
                    above_count += 1
                    max_above = max(max_above, highs[i + j])
                else:
                    break
            
            if above_count < hold_bars:
                continue  # Didn't hold long enough
            
            # It swept and held — now check if it reversed
            sweep_high = max_above
            entry_bar = i + above_count
            entry_price = closes[entry_bar]
            
            # Measure reversal over next N bars
            end_bar = min(entry_bar + reverse_bars, len(df) - 1)
            future_lows = lows[entry_bar + 1: end_bar + 1]
            future_closes = closes[entry_bar + 1: end_bar + 1]
            
            if len(future_lows) == 0:
                continue
            
            min_low = np.min(future_lows)
            max_close_after = np.max(future_closes) if len(future_closes) > 0 else entry_price
            
            # Reversal metrics
            reversal_pct = (entry_price - min_low) / entry_price * 100
            extension_pct = (max_close_after - entry_price) / entry_price * 100
            
            # Did it reverse at all?
            reversed = reversal_pct > 0.1  # at least 0.1% drop
            
            # Context at sweep time
            bull_structure = ema21[i] > ema55[i]
            rsi_val = rsi[i] if not np.isnan(rsi[i]) else 50
            atr_val = atr[i] if not np.isnan(atr[i]) else 0
            taker_4h = np.mean(taker_ratio[max(0,i-16):i]) if i > 16 else 0.5
            
            # Taker bearish = sellers dominant
            sellers_dominant = taker_4h < 0.48
            
            # Judas context = bull structure + bearish flow
            is_judas_context = bull_structure and sellers_dominant
            
            events.append({
                'timestamp': df.index[i],
                'sweep_high': sweep_high,
                'resistance': res_level,
                'sweep_pct': (sweep_high - res_level) / res_level * 100,
                'entry_price': entry_price,
                'entry_bar': entry_bar,
                'hold_bars': above_count,
                'reversal_pct': reversal_pct,
                'extension_pct': extension_pct,
                'reversed': reversed,
                'min_low': min_low,
                'max_close_after': max_close_after,
                'bull_structure': bull_structure,
                'rsi': rsi_val,
                'taker_4h': taker_4h,
                'sellers_dominant': sellers_dominant,
                'is_judas_context': is_judas_context,
                'atr': atr_val,
            })
            
            break  # one sweep per bar
    
    return events


def analyze_events(events):
    """Analyze Judas sweep performance."""
    if not events:
        print("No events found.")
        return
    
    df = pd.DataFrame(events)
    
    print("=" * 70)
    print("  JUDAS SWEEP PERFORMANCE EVALUATION")
    print("=" * 70)
    print(f"\n  Total sweeps detected: {len(df)}")
    print(f"  Date range: {df['timestamp'].min()} → {df['timestamp'].max()}")
    
    # Overall stats
    print(f"\n{'─' * 70}")
    print("  OVERALL SWEEP STATS")
    print(f"{'─' * 70}")
    rev_rate = df['reversed'].mean() * 100
    avg_rev = df[df['reversed']]['reversal_pct'].mean()
    avg_ext = df[~df['reversed']]['extension_pct'].mean() if (~df['reversed']).any() else 0
    avg_sweep = df['sweep_pct'].mean()
    
    print(f"  Reversal rate:     {rev_rate:.1f}%")
    print(f"  Avg reversal:      {avg_rev:.2f}%")
    print(f"  Avg extension:     {avg_ext:.2f}% (when no reversal)")
    print(f"  Avg sweep depth:   {avg_sweep:.2f}% above resistance")
    
    # Judas context vs non-Judas
    judas = df[df['is_judas_context']]
    non_judas = df[~df['is_judas_context']]
    
    print(f"\n{'─' * 70}")
    print("  JUDAS CONTEXT (bull structure + bearish sellers)")
    print(f"{'─' * 70}")
    
    if len(judas) > 0:
        j_rev = judas['reversed'].mean() * 100
        j_avg_rev = judas[judas['reversed']]['reversal_pct'].mean() if judas['reversed'].any() else 0
        j_avg_ext = judas[~judas['reversed']]['extension_pct'].mean() if (~judas['reversed']).any() else 0
        
        print(f"  Count:             {len(judas)}")
        print(f"  Reversal rate:     {j_rev:.1f}%")
        print(f"  Avg reversal:      {j_avg_rev:.2f}%")
        print(f"  Avg extension:     {j_avg_ext:.2f}% (when no reversal)")
        
        # R:R for Judas SHORT
        if j_avg_rev > 0 and j_avg_ext > 0:
            rr = j_avg_rev / j_avg_ext
            print(f"  Avg R:R (SHORT):   {rr:.2f}x")
    else:
        print("  No Judas context events found.")
    
    if len(non_judas) > 0:
        print(f"\n{'─' * 70}")
        print("  NON-JUDAS CONTEXT (other setups)")
        print(f"{'─' * 70}")
        nj_rev = non_judas['reversed'].mean() * 100
        nj_avg_rev = non_judas[non_judas['reversed']]['reversal_pct'].mean() if non_judas['reversed'].any() else 0
        
        print(f"  Count:             {len(non_judas)}")
        print(f"  Reversal rate:     {nj_rev:.1f}%")
        print(f"  Avg reversal:      {nj_avg_rev:.2f}%")
    
    # RSI buckets
    print(f"\n{'─' * 70}")
    print("  BY RSI AT SWEEP")
    print(f"{'─' * 70}")
    for label, lo, hi in [('Oversold (<35)', 0, 35), ('Neutral (35-65)', 35, 65), 
                           ('Overbought (65-75)', 65, 75), ('Extreme (>75)', 75, 100)]:
        subset = df[(df['rsi'] >= lo) & (df['rsi'] < hi)]
        if len(subset) > 0:
            r = subset['reversed'].mean() * 100
            print(f"  {label:25s}  n={len(subset):3d}  rev={r:.1f}%")
    
    # Sweep depth buckets
    print(f"\n{'─' * 70}")
    print("  BY SWEEP DEPTH")
    print(f"{'─' * 70}")
    for label, lo, hi in [('Shallow (0.1-0.3%)', 0.1, 0.3), ('Medium (0.3-0.5%)', 0.3, 0.5),
                           ('Deep (0.5-1.0%)', 0.5, 1.0), ('Extreme (>1.0%)', 1.0, 10.0)]:
        subset = df[(df['sweep_pct'] >= lo) & (df['sweep_pct'] < hi)]
        if len(subset) > 0:
            r = subset['reversed'].mean() * 100
            a = subset[subset['reversed']]['reversal_pct'].mean() if subset['reversed'].any() else 0
            print(f"  {label:25s}  n={len(subset):3d}  rev={r:.1f}%  avg_drop={a:.2f}%")
    
    # Time of day
    print(f"\n{'─' * 70}")
    print("  BY HOUR (UTC)")
    print(f"{'─' * 70}")
    df['hour'] = pd.to_datetime(df['timestamp']).dt.hour
    hourly = df.groupby('hour').agg(
        count=('reversed', 'size'),
        rev_rate=('reversed', 'mean'),
        avg_rev=('reversal_pct', lambda x: x[df.loc[x.index, 'reversed']].mean() if df.loc[x.index, 'reversed'].any() else 0)
    )
    for h in range(24):
        if h in hourly.index and hourly.loc[h, 'count'] >= 3:
            r = hourly.loc[h, 'rev_rate'] * 100
            n = int(hourly.loc[h, 'count'])
            a = hourly.loc[h, 'avg_rev']
            bar = '█' * int(r / 5)
            print(f"  {h:02d}:00  n={n:3d}  rev={r:.1f}%  avg={a:.2f}%  {bar}")
    
    # Best trades
    print(f"\n{'─' * 70}")
    print("  TOP 10 JUDAS SHORTS (by reversal %)")
    print(f"{'─' * 70}")
    top = df[df['is_judas_context'] & df['reversed']].nlargest(10, 'reversal_pct')
    if len(top) > 0:
        for _, row in top.iterrows():
            print(f"  {row['timestamp']}  sweep={row['sweep_pct']:.2f}%  "
                  f"drop={row['reversal_pct']:.2f}%  RSI={row['rsi']:.0f}  "
                  f"taker={row['taker_4h']:.3f}")
    else:
        print("  No qualifying trades.")
    
    # Worst trades (sweeps that failed)
    print(f"\n{'─' * 70}")
    print("  WORST 10 SWEEPS (extended against)")
    print(f"{'─' * 70}")
    worst = df[df['is_judas_context'] & ~df['reversed']].nlargest(10, 'extension_pct')
    if len(topp := worst) > 0:
        for _, row in worst.iterrows():
            print(f"  {row['timestamp']}  sweep={row['sweep_pct']:.2f}%  "
                  f"ext={row['extension_pct']:.2f}%  RSI={row['rsi']:.0f}  "
                  f"taker={row['taker_4h']:.3f}")
    else:
        print("  All Judas sweeps reversed.")
    
    # Current context comparison
    print(f"\n{'─' * 70}")
    print("  CURRENT CONTEXT MATCH")
    print(f"{'─' * 70}")
    print("  Current: bull_structure=True, sellers=True, RSI~?, sweep_target=$2311-$2315")
    similar = df[df['is_judas_context']]
    if len(similar) > 0:
        print(f"  Similar historical setups: {len(similar)}")
        print(f"  Reversal rate: {similar['reversed'].mean()*100:.1f}%")
        if similar['reversed'].any():
            winners = similar[similar['reversed']]
            print(f"  Avg reversal (winners): {winners['reversal_pct'].mean():.2f}%")
            print(f"  Median reversal: {winners['reversal_pct'].median():.2f}%")
            print(f"  Max reversal: {winners['reversal_pct'].max():.2f}%")
        losers = similar[~similar['reversed']]
        if len(losers) > 0:
            print(f"  Avg extension (losers): {losers['extension_pct'].mean():.2f}%")
            print(f"  Max extension: {losers['extension_pct'].max():.2f}%")
    
    print(f"\n{'=' * 70}")


def main():
    parser = argparse.ArgumentParser(description='Evaluate Judas sweep performance')
    parser.add_argument('--csv', default='eth_15m_merged.csv', help='CSV file')
    parser.add_argument('--sweep-pct', type=float, default=0.1, help='Min sweep %% above resistance')
    parser.add_argument('--hold-bars', type=int, default=2, help='Min bars above resistance')
    parser.add_argument('--reverse-bars', type=int, default=48, help='Bars to measure reversal')
    parser.add_argument('--min-touches', type=int, default=3, help='Min resistance touches')
    parser.add_argument('--start', help='Start date (YYYY-MM-DD)')
    parser.add_argument('--end', help='End date (YYYY-MM-DD)')
    args = parser.parse_args()
    
    csv_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), args.csv)
    print(f"  Loading {csv_path}...")
    df = pd.read_csv(csv_path, parse_dates=['Open time'], index_col='Open time')
    
    if args.start:
        df = df[df.index >= args.start]
    if args.end:
        df = df[df.index <= args.end]
    
    print(f"  Loaded {len(df)} bars ({df.index[0]} → {df.index[-1]})")
    
    config = {
        'sweep_pct': args.sweep_pct,
        'hold_bars': args.hold_bars,
        'reverse_bars': args.reverse_bars,
        'min_touches': args.min_touches,
    }
    
    events = detect_judas_sweeps(df, config)
    analyze_events(events)


if __name__ == '__main__':
    main()

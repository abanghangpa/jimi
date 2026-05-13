#!/usr/bin/env python3
"""Find NEW liquidation levels by tracing where positions accumulated.
Focus: areas where price consolidated, bounced, or rejected — these create stop clusters."""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import numpy as np
import pandas as pd
from src.utils.data_handler import load_data

CSV = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'eth_15m_merged.csv')
df = load_data(CSV)

# 72h window
df_72h = df.tail(288).copy().reset_index(drop=True)
current = float(df_72h['Close'].iloc[-1])

print(f"72h: {df_72h['Open time'].iloc[0]} → {df_72h['Open time'].iloc[-1]}")
print(f"Current: ${current:.2f}")
print()

# Method: Find BOUNCE POINTS and CONSOLIDATION ZONES
# These are where new positions get opened and stops get placed

def find_bounce_zones(df, min_touches=2, zone_pct=0.003):
    """Find price zones where multiple bars bounced (touched but didn't close through).
    These zones accumulate stops and liquidation levels."""
    zones = []
    closes = df['Close'].values.astype(float)
    highs = df['High'].values.astype(float)
    lows = df['Low'].values.astype(float)
    opens = df['Open'].values.astype(float)
    
    # Find swing lows and highs
    for i in range(2, len(df)-2):
        # Swing low: low is lower than neighbors
        if (lows[i] < lows[i-1] and lows[i] < lows[i-2] and 
            lows[i] < lows[i+1] and lows[i] < lows[i+2]):
            zones.append({
                'price': lows[i],
                'type': 'SWING_LOW',
                'time': df['Open time'].iloc[i],
                'idx': i,
                'context': 'bounce support — stops below'
            })
        
        # Swing high: high is higher than neighbors
        if (highs[i] > highs[i-1] and highs[i] > highs[i-2] and 
            highs[i] > highs[i+1] and highs[i] > highs[i+2]):
            zones.append({
                'price': highs[i],
                'type': 'SWING_HIGH',
                'time': df['Open time'].iloc[i],
                'idx': i,
                'context': 'bounce resistance — stops above'
            })
    
    # Find consolidation zones (tight range bars)
    for i in range(4, len(df)):
        window = highs[i-4:i+1] - lows[i-4:i+1]
        avg_range = np.mean(window)
        current_range = highs[i] - lows[i]
        if current_range < avg_range * 0.5:  # Tight bar
            mid = (highs[i] + lows[i]) / 2
            zones.append({
                'price': mid,
                'type': 'CONSOLIDATION',
                'time': df['Open time'].iloc[i],
                'idx': i,
                'context': 'tight range — breakout stops'
            })
    
    # Find rejection wicks (long wick = stops got placed at the extreme)
    for i in range(len(df)):
        body = abs(closes[i] - opens[i])
        total = highs[i] - lows[i]
        if total > 0 and body > 0:
            upper_wick = highs[i] - max(opens[i], closes[i])
            lower_wick = min(opens[i], closes[i]) - lows[i]
            
            if upper_wick > body * 2 and upper_wick > total * 0.4:
                zones.append({
                    'price': highs[i],
                    'type': 'REJECTION_HIGH',
                    'time': df['Open time'].iloc[i],
                    'idx': i,
                    'context': f'upper rejection wick ({upper_wick:.2f}) — stops above'
                })
            
            if lower_wick > body * 2 and lower_wick > total * 0.4:
                zones.append({
                    'price': lows[i],
                    'type': 'REJECTION_LOW',
                    'time': df['Open time'].iloc[i],
                    'idx': i,
                    'context': f'lower rejection wick ({lower_wick:.2f}) — stops below'
                })
    
    return zones

zones = find_bounce_zones(df_72h)

# Cluster nearby zones into levels
def cluster_zones(zones, tolerance=0.003):
    """Group nearby price zones into clusters."""
    if not zones:
        return []
    
    sorted_zones = sorted(zones, key=lambda z: z['price'])
    clusters = []
    current_cluster = [sorted_zones[0]]
    
    for z in sorted_zones[1:]:
        if (z['price'] - current_cluster[-1]['price']) / current_cluster[-1]['price'] < tolerance:
            current_cluster.append(z)
        else:
            clusters.append(current_cluster)
            current_cluster = [z]
    clusters.append(current_cluster)
    
    # Summarize clusters
    result = []
    for cluster in clusters:
        avg_price = np.mean([z['price'] for z in cluster])
        types = [z['type'] for z in cluster]
        times = [z['time'] for z in cluster]
        count = len(cluster)
        
        # Count type occurrences
        type_counts = {}
        for t in types:
            type_counts[t] = type_counts.get(t, 0) + 1
        
        result.append({
            'price': avg_price,
            'count': count,
            'types': type_counts,
            'first_time': min(times),
            'last_time': max(times),
            'contexts': list(set(z['context'] for z in cluster))
        })
    
    return result

clusters = cluster_zones(zones, tolerance=0.003)

# Now check which clusters have been swept
print("=" * 80)
print("  🔴 ALL FORMED LIQUIDATION LEVELS (72h bounce/rejection zones)")
print("=" * 80)
print(f"  {'#':<3} {'Price':>10} {'Touches':>8} {'Type':>25} {'First':>18} {'Last':>18} {'Swept?':>8}")
print("-" * 100)

unswept = []
for i, c in enumerate(clusters):
    price = c['price']
    # Check if price revisited this level AFTER it was formed
    last_form_idx = 0
    for j in range(len(df_72h)):
        h = float(df_72h['High'].iloc[j])
        l = float(df_72h['Low'].iloc[j])
        if l <= price <= h:
            last_form_idx = j
    
    # Check if swept after last formation
    swept_after = False
    for j in range(last_form_idx + 1, len(df_72h)):
        h = float(df_72h['High'].iloc[j])
        l = float(df_72h['Low'].iloc[j])
        if l <= price <= h:
            swept_after = True
            break
    
    type_str = ", ".join(f"{v}×{k}" for k, v in sorted(c['types'].items(), key=lambda x: -x[1]))
    first_t = str(c['first_time'])[:16]
    last_t = str(c['last_time'])[:16]
    
    if swept_after:
        status = "✅ SWEPT"
    else:
        status = "❌ UNSWEPT"
        dist = (price - current) / current * 100
        unswept.append((price, c['count'], type_str, dist, c['contexts']))
    
    print(f"  {i+1:<3} ${price:>9.2f} {c['count']:>8} {type_str:>25} {first_t:>18} {last_t:>18} {status:>8}")

# Filter: only unswept levels near current price
print()
print("=" * 80)
print("  🎯 UNSWEPT LIQUIDATION LEVELS (formed but not revisited)")
print("=" * 80)

# Sort by distance to current
unswept.sort(key=lambda x: abs(x[3]))

if unswept:
    for price, count, types, dist, contexts in unswept:
        direction = "▲ ABOVE" if dist > 0 else "▼ BELOW"
        print(f"\n  {direction}  ${price:.2f}  ({dist:+.2f}%)")
        print(f"    Touches: {count}  |  Types: {types}")
        print(f"    Context: {contexts[0] if contexts else 'N/A'}")
else:
    print("  No unswept zones found.")

# Special focus: the recovery from $2256
print()
print("=" * 80)
print("  📈 RECOVERY PHASE — New Positions & Their Stop Levels")
print("=" * 80)

# Find recovery start
recovery_start = None
for j in range(len(df_72h)):
    if float(df_72h['Low'].iloc[j]) <= 2257:
        recovery_start = j
        break

if recovery_start:
    df_rec = df_72h.iloc[recovery_start:].copy()
    
    # Key bounce levels during recovery
    print(f"\n  Recovery: {df_rec['Open time'].iloc[0]} → {df_rec['Open time'].iloc[-1]}")
    print(f"  Bounce: ${float(df_rec['Low'].min()):.2f} → ${float(df_rec['High'].max()):.2f}")
    print()
    
    # Find the key levels where new longs likely entered
    # These are the swing lows during recovery (buyers stepped in)
    rec_closes = df_rec['Close'].values.astype(float)
    rec_lows = df_rec['Low'].values.astype(float)
    rec_highs = df_rec['High'].values.astype(float)
    rec_times = df_rec['Open time'].values
    
    # Find local minima (buy zones)
    buy_zones = []
    for i in range(2, len(df_rec)-2):
        if (rec_lows[i] < rec_lows[i-1] and rec_lows[i] < rec_lows[i+1]):
            buy_zones.append((rec_lows[i], rec_times[i], i))
    
    print("  Buy zones (swing lows where longs entered):")
    for price, time, idx in buy_zones:
        dist = (price - current) / current * 100
        # Their stop loss would be ~0.5-1% below
        stop_1pct = price * 0.99
        stop_half = price * 0.995
        # Check if stops have been swept
        stop_swept_1 = any(float(rec_lows[j]) <= stop_1pct for j in range(idx+1, len(df_rec)))
        stop_swept_h = any(float(rec_lows[j]) <= stop_half for j in range(idx+1, len(df_rec)))
        
        print(f"    ${price:.2f} @ {str(time)[:16]}  dist={dist:+.2f}%")
        print(f"      Stop levels: ${stop_half:.2f} (0.5%) {'✅ swept' if stop_swept_h else '❌ UNSWEPT'}  |  ${stop_1pct:.2f} (1%) {'✅ swept' if stop_swept_1 else '❌ UNSWEPT'}")
    
    # Find local maxima (sell zones / resistance)
    sell_zones = []
    for i in range(2, len(df_rec)-2):
        if (rec_highs[i] > rec_highs[i-1] and rec_highs[i] > rec_highs[i+1]):
            sell_zones.append((rec_highs[i], rec_times[i], i))
    
    print("\n  Resistance zones (swing highs where shorts may have entered):")
    for price, time, idx in sell_zones:
        dist = (price - current) / current * 100
        # Their stop would be ~0.5-1% above
        stop_1pct = price * 1.01
        stop_half = price * 1.005
        stop_swept_1 = any(float(rec_highs[j]) >= stop_1pct for j in range(idx+1, len(df_rec)))
        stop_swept_h = any(float(rec_highs[j]) >= stop_half for j in range(idx+1, len(df_rec)))
        
        print(f"    ${price:.2f} @ {str(time)[:16]}  dist={dist:+.2f}%")
        print(f"      Stop levels: ${stop_half:.2f} (0.5%) {'✅ swept' if stop_swept_h else '❌ UNSWEPT'}  |  ${stop_1pct:.2f} (1%) {'✅ swept' if stop_swept_1 else '❌ UNSWEPT'}")

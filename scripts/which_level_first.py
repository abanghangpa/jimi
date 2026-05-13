#!/usr/bin/env python3
"""
Custom forward simulation — which level gets hit first?
Built from raw data, no framework dependency.
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import numpy as np
import pandas as pd
from datetime import datetime

CSV = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'eth_15m_merged.csv')
df = pd.read_csv(CSV)
df['Open time'] = pd.to_datetime(df['Open time'])
for c in ['Open','High','Low','Close','Volume','Taker buy base asset volume','Taker buy quote asset volume']:
    df[c] = pd.to_numeric(df[c], errors='coerce')

# ═══════════════════════════════════════════════════
#  STEP 1: Compute everything from scratch
# ═══════════════════════════════════════════════════

def ema(s, n):
    return s.ewm(span=n, adjust=False).mean()

def rsi(s, n=14):
    delta = s.diff()
    gain = delta.where(delta > 0, 0).rolling(n).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(n).mean()
    rs = gain / loss.replace(0, np.nan)
    return 100 - (100 / (1 + rs))

def atr(h, l, c, n=14):
    tr = pd.concat([h-l, (h-c.shift()).abs(), (l-c.shift()).abs()], axis=1).max(axis=1)
    return tr.rolling(n).mean()

def macd(s, fast=12, slow=26, sig=9):
    line = ema(s, fast) - ema(s, slow)
    signal = ema(line, sig)
    hist = line - signal
    return line, signal, hist

def vwap(h, l, c, v, lookback=20):
    tp = (h + l + c) / 3
    cum_tp_vol = (tp * v).rolling(lookback).sum()
    cum_vol = v.rolling(lookback).sum()
    return cum_tp_vol / cum_vol.replace(0, np.nan)

# Core indicators
df['ema9'] = ema(df['Close'], 9)
df['ema21'] = ema(df['Close'], 21)
df['ema55'] = ema(df['Close'], 55)
df['rsi'] = rsi(df['Close'], 14)
df['atr'] = atr(df['High'], df['Low'], df['Close'], 14)
df['macd_line'], df['macd_signal'], df['macd_hist'] = macd(df['Close'])
df['vwap'] = vwap(df['High'], df['Low'], df['Close'], df['Volume'], 20)
df['vol_ma20'] = df['Volume'].rolling(20).mean()
df['vol_ratio'] = df['Volume'] / df['vol_ma20'].replace(0, np.nan)
df['taker_ratio'] = df['Taker buy base asset volume'] / df['Volume'].replace(0, np.nan)

# 2h resample
df_2h = df.set_index('Open time').resample('2h').agg({
    'Open': 'first', 'High': 'max', 'Low': 'min', 'Close': 'last',
    'Volume': 'sum', 'Taker buy base asset volume': 'sum'
}).dropna()
df_2h['macd_line'], df_2h['macd_signal'], df_2h['macd_hist'] = macd(df_2h['Close'], 8, 17, 9)
df_2h['rsi'] = rsi(df_2h['Close'], 14)

# Daily
df_1d = df.set_index('Open time').resample('1D').agg({
    'Open': 'first', 'High': 'max', 'Low': 'min', 'Close': 'last',
    'Volume': 'sum'
}).dropna()

# ═══════════════════════════════════════════════════
#  STEP 2: Define current state
# ═══════════════════════════════════════════════════

n = len(df)
current = float(df['Close'].iloc[-1])
current_time = df['Open time'].iloc[-1]

# Fresh levels (from scanner output)
targets_up = [2311.39, 2320.52, 2328.90]
targets_down = [2275.53, 2269.00]

# Current indicators
cur_rsi = float(df['rsi'].iloc[-1])
cur_atr = float(df['atr'].iloc[-1])
cur_vol_ratio = float(df['vol_ratio'].iloc[-1])
cur_taker = float(df['taker_ratio'].iloc[-1])
cur_macd_hist = float(df['macd_hist'].iloc[-1])
cur_ema9 = float(df['ema9'].iloc[-1])
cur_ema21 = float(df['ema21'].iloc[-1])
cur_ema55 = float(df['ema55'].iloc[-1])

# 2h MACD state
cur_2h_hist = float(df_2h['macd_hist'].iloc[-1])
prev_2h_hist = float(df_2h['macd_hist'].iloc[-2])
cur_2h_rsi = float(df_2h['rsi'].iloc[-1])

# ATR compression
atr_50 = float(df['atr'].iloc[-50:].mean())
compression = cur_atr / atr_50

# Range position (24h)
h24 = float(df['High'].iloc[-96:].max())
l24 = float(df['Low'].iloc[-96:].min())
range_pos = (current - l24) / (h24 - l24) if h24 != l24 else 0.5

print(f"Current: ${current:.2f} @ {current_time}")
print(f"RSI: {cur_rsi:.1f}  ATR: {cur_atr:.2f} (compression: {compression:.2f}x)")
print(f"Vol ratio: {cur_vol_ratio:.2f}x  Taker: {cur_taker:.3f}")
print(f"EMA9: ${cur_ema9:.2f}  EMA21: ${cur_ema21:.2f}  EMA55: ${cur_ema55:.2f}")
print(f"MACD hist: {cur_macd_hist:.3f}  2h hist: {cur_2h_hist:.3f} (Δ={cur_2h_hist - prev_2h_hist:.4f})")
print(f"2h RSI: {cur_2h_rsi:.1f}")
print(f"24h range pos: {range_pos*100:.0f}%  (0=low, 100=high)")
print(f"24h range: ${l24:.2f} – ${h24:.2f}")
print()

# ═══════════════════════════════════════════════════
#  STEP 3: Find ALL similar historical conditions
# ═══════════════════════════════════════════════════

# Match conditions (not just squeeze — the FULL picture):
# 1. RSI 35-65 (neutral zone)
# 2. ATR compressed (<0.85x of 50-bar avg)
# 3. MACD histogram small (|hist| < 3)
# 4. Price between EMA21 ± 2% (mean-reversion zone)
# 5. Volume below average (<1.2x)
# 6. Taker ratio near 50% (0.42-0.58)

similar = []
for i in range(100, n - 40):  # need 40 bars of future data
    r = float(df['rsi'].iloc[i])
    a = float(df['atr'].iloc[i])
    a50 = float(df['atr'].iloc[max(0,i-50):i].mean())
    mh = abs(float(df['macd_hist'].iloc[i]))
    p = float(df['Close'].iloc[i])
    e21 = float(df['ema21'].iloc[i])
    vr = float(df['vol_ratio'].iloc[i])
    tr = float(df['taker_ratio'].iloc[i])

    if pd.isna(r) or pd.isna(a) or pd.isna(a50) or pd.isna(mh) or pd.isna(e21) or pd.isna(vr) or pd.isna(tr):
        continue
    if a50 == 0 or e21 == 0:
        continue

    comp = a / a50
    ema_dist = abs(p - e21) / e21

    if (35 < r < 65 and comp < 0.85 and mh < 3 and
        ema_dist < 0.02 and vr < 1.2 and 0.42 < tr < 0.58):

        # Compute which targets exist at this point
        # Use dynamic levels: find nearest swing H/L as proxy
        # Swing highs (resistance stops above)
        nearest_up = None
        nearest_down = None

        for j in range(max(0, i-48), i):
            h_j = float(df['High'].iloc[j])
            l_j = float(df['Low'].iloc[j])

            # Swing high
            if j >= 2 and j < i - 2:
                if (h_j > float(df['High'].iloc[j-1]) and h_j > float(df['High'].iloc[j-2]) and
                    h_j > float(df['High'].iloc[j+1]) and h_j > float(df['High'].iloc[j+2])):
                    dist = (h_j - p) / p * 100
                    if 0.2 < dist < 2.0:
                        if nearest_up is None or dist < nearest_up:
                            nearest_up = dist

                # Swing low
                if (l_j < float(df['Low'].iloc[j-1]) and l_j < float(df['Low'].iloc[j-2]) and
                    l_j < float(df['Low'].iloc[j+1]) and l_j < float(df['Low'].iloc[j+2])):
                    dist = (p - l_j) / p * 100
                    if 0.2 < dist < 2.0:
                        if nearest_down is None or dist < nearest_down:
                            nearest_down = dist

        if nearest_up is None or nearest_down is None:
            continue

        # Simulate forward: which target gets hit first?
        # Convert % targets to absolute prices
        target_up_price = p * (1 + nearest_up / 100)
        target_down_price = p * (1 - nearest_down / 100)

        hit_up = None
        hit_down = None
        for k in range(1, 41):  # next 40 bars = 10 hours
            if i + k >= n:
                break
            h_k = float(df['High'].iloc[i+k])
            l_k = float(df['Low'].iloc[i+k])

            if hit_up is None and h_k >= target_up_price:
                hit_up = k
            if hit_down is None and l_k <= target_down_price:
                hit_down = k

        if hit_up is not None or hit_down is not None:
            if hit_up is not None and hit_down is not None:
                first = 'UP' if hit_up < hit_down else 'DOWN'
                bars = min(hit_up, hit_down)
            elif hit_up is not None:
                first = 'UP'
                bars = hit_up
            else:
                first = 'DOWN'
                bars = hit_down

            similar.append({
                'idx': i,
                'time': df['Open time'].iloc[i],
                'price': p,
                'rsi': r,
                'compression': comp,
                'taker': tr,
                'range_pos': (p - l24) / (h24 - l24) if h24 != l24 else 0.5,
                'nearest_up': nearest_up,
                'nearest_down': nearest_down,
                'first': first,
                'bars': bars,
                'hit_up': hit_up,
                'hit_down': hit_down,
            })

print(f"=" * 70)
print(f"  HISTORICAL MATCHES — Similar conditions")
print(f"=" * 70)
print(f"  Criteria: RSI 35-65, ATR compressed, MACD quiet, near EMA21, low vol")
print(f"  Found: {len(similar)} matches")
print()

if not similar:
    print("  No matches found. Relaxing criteria...")
    sys.exit(1)

up_first = [s for s in similar if s['first'] == 'UP']
down_first = [s for s in similar if s['first'] == 'DOWN']

print(f"  Direction:")
print(f"    ▲ Up first:   {len(up_first)}/{len(similar)} = {len(up_first)/len(similar)*100:.1f}%")
print(f"    ▼ Down first: {len(down_first)}/{len(similar)} = {len(down_first)/len(similar)*100:.1f}%")
print()

if up_first:
    avg_up_bars = np.mean([s['hit_up'] for s in up_first])
    avg_up_dist = np.mean([s['nearest_up'] for s in up_first])
    print(f"  ▲ Up first stats:")
    print(f"    Avg bars to hit: {avg_up_bars:.1f} ({avg_up_bars*15:.0f} min)")
    print(f"    Avg target dist: {avg_up_dist:.2f}%")

if down_first:
    avg_down_bars = np.mean([s['hit_down'] for s in down_first])
    avg_down_dist = np.mean([s['nearest_down'] for s in down_first])
    print(f"  ▼ Down first stats:")
    print(f"    Avg bars to hit: {avg_down_bars:.1f} ({avg_down_bars*15:.0f} min)")
    print(f"    Avg target dist: {avg_down_dist:.2f}%")

# ═══════════════════════════════════════════════════
#  STEP 4: Weighted scoring based on current state
# ═══════════════════════════════════════════════════

print()
print(f"=" * 70)
print(f"  WEIGHTED ANALYSIS — Current state vs historical")
print(f"=" * 70)

# Factor 1: Raw backtest probability
base_up = len(up_first) / len(similar) * 100
base_down = len(down_first) / len(similar) * 100
print(f"\n  [1] Base probability:     ▲ {base_up:.1f}%  ▼ {base_down:.1f}%")

# Factor 2: Taker ratio shift
# Current taker = 0.580 (buyers). Compare to similar setups.
similar_taker = np.mean([s['taker'] for s in similar])
current_taker_shift = cur_taker - similar_taker
print(f"  [2] Taker shift:          {cur_taker:.3f} vs avg {similar_taker:.3f} ({current_taker_shift:+.3f})")
if current_taker_shift > 0.03:
    taker_bonus = current_taker_shift * 100  # convert to percentage points
    print(f"      → Buyers stepping in → +{taker_bonus:.1f}% to UP")
elif current_taker_shift < -0.03:
    taker_bonus = abs(current_taker_shift) * 100
    print(f"      → Sellers dominating → +{taker_bonus:.1f}% to DOWN")
else:
    taker_bonus = 0
    print(f"      → Neutral")

# Factor 3: Range position
# Price at 75% of 24h range = closer to highs, mean reversion favors down
print(f"  [3] Range position:       {range_pos*100:.0f}%")
if range_pos > 0.7:
    range_adj = (range_pos - 0.5) * 20  # 75% → +5% to DOWN
    print(f"      → Near 24h highs → +{range_adj:.1f}% to DOWN (mean reversion)")
elif range_pos < 0.3:
    range_adj = (0.5 - range_pos) * 20
    print(f"      → Near 24h lows → +{range_adj:.1f}% to UP (mean reversion)")
else:
    range_adj = 0
    print(f"      → Mid-range → neutral")

# Factor 4: EMA alignment
print(f"  [4] EMA alignment:")
ema_score_up = 0
ema_score_down = 0
if current > cur_ema9:
    ema_score_up += 1
    print(f"      ✅ Price > EMA9 (${cur_ema9:.2f})")
else:
    ema_score_down += 1
    print(f"      ❌ Price < EMA9 (${cur_ema9:.2f})")

if current > cur_ema21:
    ema_score_up += 1
    print(f"      ✅ Price > EMA21 (${cur_ema21:.2f})")
else:
    ema_score_down += 1
    print(f"      ❌ Price < EMA21 (${cur_ema21:.2f})")

if cur_ema9 > cur_ema21:
    ema_score_up += 1
    print(f"      ✅ EMA9 > EMA21 (bullish cross)")
else:
    ema_score_down += 1
    print(f"      ❌ EMA9 < EMA21 (bearish cross)")

# Factor 5: MACD momentum direction
print(f"  [5] MACD momentum:")
macd_score_up = 0
macd_score_down = 0
if cur_macd_hist > 0:
    macd_score_up += 1
    print(f"      15m hist positive ({cur_macd_hist:.3f})")
else:
    macd_score_down += 1
    print(f"      15m hist negative ({cur_macd_hist:.3f})")

if cur_2h_hist > prev_2h_hist:
    macd_score_up += 1
    print(f"      2h hist rising (Δ={cur_2h_hist - prev_2h_hist:.4f})")
else:
    macd_score_down += 1
    print(f"      2h hist falling (Δ={cur_2h_hist - prev_2h_hist:.4f})")

# Factor 6: Distance to targets
dist_up = min(targets_up) - current
dist_down = current - min(targets_down)
dist_ratio = dist_down / dist_up if dist_up > 0 else 1
print(f"  [6] Distance ratio:       up={dist_up:.2f} ({dist_up/current*100:.2f}%)  down={dist_down:.2f} ({dist_down/current*100:.2f}%)")
print(f"      Ratio: {dist_ratio:.2f}x {'(closer down)' if dist_ratio < 1 else '(closer up)' if dist_ratio > 1 else '(equidistant)'}")

# Factor 7: Squeeze state
print(f"  [7] Squeeze state:")
print(f"      ATR compression: {compression:.2f}x {'✅' if compression < 0.85 else '❌'}")
print(f"      2h MACD coil: {'✅' if abs(cur_2h_hist - prev_2h_hist) < 0.5 else '❌'}")

# Factor 8: Volume context
print(f"  [8] Volume: {cur_vol_ratio:.2f}x MA20 {'(low — false moves likely)' if cur_vol_ratio < 0.8 else ''}")

# ═══════════════════════════════════════════════════
#  STEP 5: Final verdict
# ═══════════════════════════════════════════════════

print()
print(f"=" * 70)
print(f"  🎯 FINAL VERDICT")
print(f"=" * 70)

# Compute final scores
up_score = base_up
down_score = base_down

# Taker adjustment
if current_taker_shift > 0.03:
    up_score += taker_bonus
elif current_taker_shift < -0.03:
    down_score += taker_bonus

# Range position adjustment
if range_pos > 0.7:
    down_score += range_adj
elif range_pos < 0.3:
    up_score += range_adj

# EMA adjustment
up_score += ema_score_up * 3
down_score += ema_score_down * 3

# MACD adjustment
up_score += macd_score_up * 2
down_score += macd_score_down * 2

# Distance adjustment (closer = easier to hit)
if dist_ratio < 0.8:
    down_score += 3
elif dist_ratio > 1.2:
    up_score += 3

# Volume (low vol = mean reversion, high vol = breakout)
if cur_vol_ratio < 0.8:
    # Low volume → price tends to revert to mean
    if range_pos > 0.6:
        down_score += 2
    elif range_pos < 0.4:
        up_score += 2

print(f"\n  Final scores:")
print(f"    ▲ UP (short stops at ${min(targets_up):.2f}):   {up_score:.1f}")
print(f"    ▼ DOWN (long stops at ${min(targets_down):.2f}): {down_score:.1f}")
print()

total = up_score + down_score
up_pct = up_score / total * 100
down_pct = down_score / total * 100

if up_score > down_score:
    winner = "▲ UP"
    target = min(targets_up)
    pct = up_pct
    est_bars = np.mean([s['hit_up'] for s in up_first]) if up_first else 10
else:
    winner = "▼ DOWN"
    target = min(targets_down)
    pct = down_pct
    est_bars = np.mean([s['hit_down'] for s in down_first]) if down_first else 10

print(f"  Winner: {winner}  ({pct:.0f}% confidence)")
print(f"  First level: ${target:.2f} ({(target-current)/current*100:+.2f}%)")
print(f"  Est. time: ~{est_bars:.0f} bars ({est_bars*15:.0f} min)")
print()
print(f"  Scoring breakdown:")
print(f"    Base backtest:    ▲ {base_up:.1f}%  ▼ {base_down:.1f}%")
print(f"    + Taker shift:    {'▲' if current_taker_shift > 0.03 else '▼' if current_taker_shift < -0.03 else '='} {abs(taker_bonus):.1f}%")
print(f"    + Range position: {'▼' if range_pos > 0.7 else '▲' if range_pos < 0.3 else '='} {abs(range_adj):.1f}%")
print(f"    + EMA alignment:  ▲ +{ema_score_up*3}  ▼ +{ema_score_down*3}")
print(f"    + MACD momentum:  ▲ +{macd_score_up*2}  ▼ +{macd_score_down*2}")
print(f"    + Distance:       {'▼' if dist_ratio < 0.8 else '▲' if dist_ratio > 1.2 else '='}")
print(f"    + Volume context: low vol, range top → {'▼' if range_pos > 0.6 else '▲' if range_pos < 0.4 else '='}")

# ═══════════════════════════════════════════════════
#  STEP 6: Confidence check — how often was I wrong?
# ═══════════════════════════════════════════════════

print()
print(f"=" * 70)
print(f"  📊 CONFIDENCE CALIBRATION")
print(f"=" * 70)

# For historical matches where we would have predicted the same direction,
# what was the actual outcome?
predicted = 'UP' if up_score > down_score else 'DOWN'
correct = 0
total_pred = 0

for s in similar:
    # Would we have predicted UP or DOWN for this setup?
    s_up = base_up
    s_down = base_down

    # Taker adjustment
    if s['taker'] > 0.53:
        s_up += 3
    elif s['taker'] < 0.47:
        s_down += 3

    # Range position
    rp = s.get('range_pos', 0.5)
    if rp > 0.7:
        s_down += 5
    elif rp < 0.3:
        s_up += 5

    s_predicted = 'UP' if s_up > s_down else 'DOWN'
    if s_predicted == predicted:
        total_pred += 1
        if s['first'] == predicted:
            correct += 1

if total_pred > 0:
    accuracy = correct / total_pred * 100
    print(f"  When we predicted {predicted} in similar conditions:")
    print(f"    Correct: {correct}/{total_pred} = {accuracy:.1f}%")
    print(f"    {'✅ Above 55% — decent edge' if accuracy > 55 else '⚠️ Below 55% — coin flip territory'}")
else:
    print(f"  Not enough similar predictions to calibrate.")

print()
print(f"  ⚠️  Caveats:")
print(f"  • Asian session — low liquidity, moves can reverse fast")
print(f"  • No fresh volume data (last bar may be incomplete)")
print(f"  • L/S ratio 2.86 — crowded long, squeeze risk is real")
print(f"  • Squeeze hasn't triggered yet — either direction still possible")

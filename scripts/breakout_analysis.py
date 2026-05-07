#!/usr/bin/env python3
"""Analyze the Apr 26 breakout failure and Apr 27 breakdown."""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
import pandas as pd
import numpy as np
from src.utils.data_handler import load_data, resample_ohlcv
from src.utils.indicators import calc_atr, calc_vwap, calc_vol_ratio, calc_rsi, calc_ema

df = load_data('eth_15m_merged.csv')
df['Open time'] = pd.to_datetime(df['Open time'])
mask = (df['Open time'] >= '2026-04-25') & (df['Open time'] < '2026-04-28')
df_f = df[mask].copy().reset_index(drop=True)

df_f['vwap'] = calc_vwap(df_f['High'], df_f['Low'], df_f['Close'], df_f['Volume'], 20)
df_f['vol_ma20'] = df_f['Volume'].rolling(20).mean()
df_f['atr'] = calc_atr(df_f['High'], df_f['Low'], df_f['Close'], 14)
df_f['rsi'] = calc_rsi(df_f['Close'], 14)
df_f['taker_ratio'] = df_f['Taker buy base asset volume'] / df_f['Volume'].replace(0, np.nan)

# Key price levels
df_f['date'] = df_f['Open time'].dt.date
apr25_low = df_f[df_f['date'] == pd.Timestamp('2026-04-25').date()]['Low'].min()
apr26_low = df_f[df_f['date'] == pd.Timestamp('2026-04-26').date()]['Low'].min()
apr26_high = df_f[df_f['date'] == pd.Timestamp('2026-04-26').date()]['High'].max()
apr27_high = df_f[df_f['date'] == pd.Timestamp('2026-04-27').date()]['High'].max()
apr27_low = df_f[df_f['date'] == pd.Timestamp('2026-04-27').date()]['Low'].min()

print("=" * 80)
print("  APR 26 BREAKOUT FAILURE → APR 27 BREAKDOWN ANALYSIS")
print("=" * 80)

print(f"\n  Apr 25 range: ${apr25_low:.2f} low")
print(f"  Apr 26 range: ${apr26_low:.2f} – ${apr26_high:.2f}  (rally: +{(apr26_high-apr26_low)/apr26_low*100:.2f}%)")
print(f"  Apr 27 range: ${apr27_low:.2f} – ${apr27_high:.2f}  (dump: {(apr27_low-apr27_high)/apr27_high*100:.2f}%)")

# Apr 26 rally detail
print(f"\n  {'='*70}")
print(f"  APR 26 RALLY (the breakout attempt)")
print(f"  {'='*70}")
apr26 = df_f[df_f['date'] == pd.Timestamp('2026-04-26').date()]
print(f"  {'Time':<22} {'Close':>8} {'High':>8} {'Vol':>8} {'VolR':>6} {'Taker':>6} {'RSI':>5} {'Range%':>6}")
for _, r in apr26.iterrows():
    p = r['Close']
    h = r['High']
    vol = r['Volume']
    vm = r['vol_ma20'] if not pd.isna(r['vol_ma20']) else 1
    vr = vol/vm if vm > 0 else 0
    tr = r['taker_ratio'] if not pd.isna(r['taker_ratio']) else 0.5
    rsi = r['rsi'] if not pd.isna(r['rsi']) else 50
    rng = (r['High']-r['Low'])/p*100
    marker = ''
    if h > 2360: marker = ' ◀ BREAKOUT HIGH'
    if h > 2370: marker = ' ◀ PEAK'
    print(f"  {r['Open time']:<22} {p:>8.2f} {h:>8.2f} {vol:>8.0f} {vr:>6.2f} {tr:>6.3f} {rsi:>5.1f} {rng:>6.2f}%{marker}")

# Apr 27 breakdown detail
print(f"\n  {'='*70}")
print(f"  APR 27 BREAKDOWN (the trap sprung)")
print(f"  {'='*70}")
apr27 = df_f[df_f['date'] == pd.Timestamp('2026-04-27').date()]
print(f"  {'Time':<22} {'Close':>8} {'High':>8} {'Low':>8} {'Vol':>8} {'VolR':>6} {'Taker':>6} {'RSI':>5} {'Range%':>6}")
for _, r in apr27.iterrows():
    p = r['Close']
    h = r['High']
    l = r['Low']
    vol = r['Volume']
    vm = r['vol_ma20'] if not pd.isna(r['vol_ma20']) else 1
    vr = vol/vm if vm > 0 else 0
    tr = r['taker_ratio'] if not pd.isna(r['taker_ratio']) else 0.5
    rsi = r['rsi'] if not pd.isna(r['rsi']) else 50
    rng = (h-l)/p*100
    marker = ''
    if h > 2390: marker = ' ◀ FAKEOUT PEAK'
    if p < 2300 and rng > 0.5: marker = ' ◀ PANIC'
    if l < 2280: marker = ' ◀ CAPITULATION'
    body = r['Close'] - r['Open']
    if body < -15: marker += f' body={body:.0f}'
    print(f"  {r['Open time']:<22} {p:>8.2f} {h:>8.2f} {l:>8.2f} {vol:>8.0f} {vr:>6.2f} {tr:>6.3f} {rsi:>5.1f} {rng:>6.2f}%{marker}")

# Breakout failure analysis
print(f"\n  {'='*70}")
print(f"  BREAKOUT FAILURE PATTERN DETECTION")
print(f"  {'='*70}")

# The breakout candle(s) on Apr 26
breakout_bars = apr26[apr26['High'] > 2360]
print(f"\n  Breakout candles (High > $2360): {len(breakout_bars)}")
for _, r in breakout_bars.iterrows():
    body = r['Close'] - r['Open']
    wick_up = r['High'] - max(r['Open'], r['Close'])
    wick_down = min(r['Open'], r['Close']) - r['Low']
    print(f"    {r['Open time']}  O={r['Open']:.2f} H={r['High']:.2f} L={r['Low']:.2f} C={r['Close']:.2f}")
    print(f"      body={body:+.2f}  wick_up={wick_up:.2f}  wick_down={wick_down:.2f}")
    print(f"      taker={r['taker_ratio']:.3f}  vol_ratio={r['Volume']/r['vol_ma20']:.2f}x")
    # Failed breakout check
    if wick_up > abs(body) * 0.5:
        print(f"      ⚠️  LONG WICK (wick_up={wick_up:.2f} > 50% of body) — rejection signal!")
    if body < 0:
        print(f"      ⚠️  RED CLOSE on breakout candle — weak conviction!")
    if r['taker_ratio'] < 0.45:
        print(f"      ⚠️  TAKER < 0.45 — sellers active on breakout!")

# The reversal
print(f"\n  Reversal analysis:")
first_red = None
for _, r in apr27.iterrows():
    body = r['Close'] - r['Open']
    if body < -10 and first_red is None:
        first_red = r
        print(f"    First significant red candle: {r['Open time']}")
        print(f"      O={r['Open']:.2f} → C={r['Close']:.2f}  (body={body:.2f})")
        print(f"      This is the TRAP SPRING — longs from breakout are underwater")
        break

# How many longs got trapped
if len(breakout_bars) > 0:
    breakout_avg = breakout_bars['Close'].mean()
    dump_low = apr27['Low'].min()
    trap_pct = (dump_low - breakout_avg) / breakout_avg * 100
    print(f"\n  Trap depth:")
    print(f"    Breakout avg entry: ${breakout_avg:.2f}")
    print(f"    Dump low: ${dump_low:.2f}")
    print(f"    Drawdown: {trap_pct:.2f}%  ({(breakout_avg - dump_low):.2f} pts)")

# What the scanner's M14 (sweep) saw
print(f"\n  {'='*70}")
print(f"  WHAT THE SCANNER MISSED")
print(f"  {'='*70}")
print("""
  The current scanner modules don't detect the FAILED BREAKOUT → TRAP → BREAKDOWN pattern:

  1. M14 (Sweep): Looks for sweep-retest-reclaim at swing levels.
     The Apr 26 breakout swept above resistance but M14 doesn't flag
     the FAILURE to hold as a reversal signal.

  2. M3 (VWAP): Price above VWAP on Apr 26 — scored as PASS for shorts
     but didn't penalize the breakout itself.

  3. M5 (Liquidation): Volume profile shows the breakout created new
     trapped longs above $2360, but M5 only looks at existing clusters,
     not newly-formed ones from the failed move.

  4. M18 (Squeeze): The compression before Apr 26 could have been
     detected, but the squeeze module focuses on compression → breakout,
     not breakout → failure → reversal.

  5. Direction resolver: Stayed SHORT all of Apr 26 despite the +2.8%
     rally. This was correct in hindsight, but the scanner had no way
     to express "SHORT but wait for the trap to confirm."

  MISSING MODULE: Failed Breakout Detector
  ─────────────────────────────────────────
  Pattern: Price breaks above resistance → fails to hold → reverses hard
  Signals to detect:
    - Breakout candle with long upper wick (rejection)
    - Breakout with low taker ratio (sellers defending)
    - Price returns below breakout level within N bars
    - Volume spike on the reversal (not the breakout)
    - OI increase during breakout + decrease on reversal (trapped longs liquidating)
    - RSI divergence at the breakout high
""")

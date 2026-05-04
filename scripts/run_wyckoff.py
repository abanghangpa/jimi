#!/usr/bin/env python3
"""Quick Wyckoff module runner — scans current ETH/USDT for Spring/Upthrust signals."""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import numpy as np
from src.config import CONFIG
from src.utils.data_handler import fetch_recent, load_data, resample_ohlcv
from src.utils.indicators import calc_ema, calc_macd, calc_rsi, calc_atr, calc_vwap, calc_vol_ratio
from src.modules.wyckoff import compute_range_state, score_wyckoff, is_mid_range_noise, WYCKOFF_DEFAULTS
from src.modules.m4_cvd import calc_cvd_15m, detect_cvd_divergence_15m, calc_cvd_2h, detect_cvd_zero_cross

print("═" * 60)
print("  JIMI — WYCKOFF MODULE SCAN")
print("═" * 60)

# Load historical CSV for daily warmup
csv_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data', 'eth_15m_merged.csv')
if not os.path.exists(csv_path):
    csv_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'eth_15m_merged.csv')

df_1d_hist = None
if os.path.exists(csv_path):
    df_all = load_data(csv_path)
    df_1d_hist = resample_ohlcv(df_all, '1D')
    print(f"\n  📊 Daily CSV: {len(df_1d_hist)} bars")

# Fetch live data
print("  Fetching 15m data (1000 bars)...")
df_15m = fetch_recent(bars=1000, timeframe='15m')

# Compute indicators
cfg = dict(CONFIG)
df_15m['vwap'] = calc_vwap(df_15m['High'], df_15m['Low'], df_15m['Close'], df_15m['Volume'], cfg['VWAP_LOOKBACK'])
df_15m['vol_ma20'] = df_15m['Volume'].rolling(20).mean()
taker_base = df_15m['Taker buy base asset volume']
total_vol = df_15m['Volume']
df_15m['taker_ratio'] = (taker_base / total_vol.replace(0, np.nan)).fillna(0.5)
df_15m['atr'] = calc_atr(df_15m['High'], df_15m['Low'], df_15m['Close'], cfg['ATR_PERIOD'])
df_15m['vol_ratio'] = calc_vol_ratio(df_15m['Volume'])
df_15m['rsi'] = calc_rsi(df_15m['Close'], 14)
df_15m['cvd_15m'] = calc_cvd_15m(df_15m)
df_15m['cvd_divergence_15m'] = detect_cvd_divergence_15m(df_15m, cfg['CVD_LOOKBACK'], cfg['CVD_DIVERGENCE_WINDOW'])

df_1h = resample_ohlcv(df_15m, '1H')
df_2h = resample_ohlcv(df_15m, '2H')
df_2h['ema_fast'] = calc_ema(df_2h['Close'], cfg['EMA_FAST'])
df_2h['ema_slow'] = calc_ema(df_2h['Close'], cfg['EMA_SLOW'])
df_2h['cvd_2h'] = calc_cvd_2h(df_2h)
df_2h['cvd_zl_state'], df_2h['cvd_zl_cross_bar'], df_2h['cvd_zl_cross_dir'] = detect_cvd_zero_cross(df_2h)

# Compute range state on 15m
df_15m = compute_range_state(df_15m, config=cfg)

idx = len(df_15m) - 1
idx_2h = len(df_2h) - 1
price = float(df_15m['Close'].iloc[idx])
ts = df_15m['Open time'].iloc[idx]

print(f"\n  Time:   {ts}")
print(f"  Price:  ${price:.2f}")
print(f"  RSI:    {df_15m['rsi'].iloc[idx]:.1f}")
print(f"  ATR:    {df_15m['atr'].iloc[idx]:.2f}")

# Range state
in_range = bool(df_15m['in_range'].iloc[idx])
range_pct = float(df_15m['range_pct'].iloc[idx])
range_high = float(df_15m['range_high'].iloc[idx])
range_low = float(df_15m['range_low'].iloc[idx])
range_width = float(df_15m['range_width_pct'].iloc[idx]) * 100
atr_pctl = float(df_15m['atr_pctl'].iloc[idx])

print(f"\n  ── Range Detection ──")
print(f"  In Range:     {'✅ YES' if in_range else '❌ NO (trending)'}")
print(f"  Range:        ${range_low:.2f} — ${range_high:.2f}  ({range_width:.2f}% wide)")
print(f"  Position:     {range_pct:.2f}  (0=bottom, 1=top)")
print(f"  ATR Pctl:     {atr_pctl:.2f}  ({'squeeze' if atr_pctl < 0.45 else 'normal'}/{range_width:.2f}% width)")

if in_range:
    if range_pct < 0.15:
        zone = "🔻 BOTTOM (Spring zone)"
    elif range_pct > 0.85:
        zone = "🔺 TOP (Upthrust zone)"
    else:
        zone = "⬜ MID-RANGE (noise zone)"
    print(f"  Zone:         {zone}")

# CVD divergence scan
print(f"\n  ── CVD Divergence (last 10 bars) ──")
for i in range(max(0, idx - 9), idx + 1):
    div = df_15m['cvd_divergence_15m'].iloc[i]
    if div != 'NONE':
        t = df_15m['Open time'].iloc[i]
        c = float(df_15m['Close'].iloc[i])
        r = float(df_15m['range_pct'].iloc[i]) if in_range else 0.5
        icon = '🔺' if div == 'BULLISH' else '🔻'
        print(f"    {t}  {icon} {div}  ${c:.2f}  range_pct={r:.2f}")

# Score Wyckoff for both directions
print(f"\n  ── Wyckoff Scoring ──")
for direction in ('LONG', 'SHORT'):
    status, score, details = score_wyckoff(df_15m, df_2h, idx, idx_2h, direction, config=cfg)
    phase = details['wyckoff_phase']
    la = details['layer_a_score']
    lb = details['layer_b_score']
    la_div = details['layer_a_div']
    lb_cross = details['layer_b_cross']
    zl = details['zl_state']

    phase_icon = {
        'SPRING': '🌱', 'UPTHRUST': '💥', 'CONTINUATION': '➡️',
        'NOISE': '🔇', 'NONE': '—'
    }.get(phase, '?')

    print(f"\n  {direction}:")
    print(f"    Status:     {status}  score={score:.3f}")
    print(f"    Phase:      {phase_icon} {phase}")
    print(f"    Layer A:    {la:.3f}  (CVD div={la_div})")
    print(f"    Layer B:    {lb:.3f}  (2H ZL: {zl}, cross={lb_cross})")
    print(f"    Composite:  {details['composite']:.3f}")

    if is_mid_range_noise(df_15m, idx, phase, config=cfg):
        print(f"    ⚠️  MID-RANGE NOISE — signal suppressed")

# 2H CVD state
print(f"\n  ── 2H CVD State ──")
zl_state = df_2h['cvd_zl_state'].iloc[idx_2h]
zl_cross_bar = df_2h['cvd_zl_cross_bar'].iloc[idx_2h]
zl_cross_dir = df_2h['cvd_zl_cross_dir'].iloc[idx_2h]
cvd_2h_now = float(df_2h['cvd_2h'].iloc[idx_2h])
bars_since = idx_2h - zl_cross_bar if zl_cross_bar >= 0 else 999
print(f"  ZL State:     {zl_state}")
print(f"  Last Cross:   {zl_cross_dir}  ({bars_since} bars ago)")
print(f"  CVD 2H:       {cvd_2h_now:.0f}")

# Volume analysis
print(f"\n  ── Volume Analysis ──")
vol_now = float(df_15m['Volume'].iloc[idx])
vol_ma = float(df_15m['vol_ma20'].iloc[idx])
vol_r = float(df_15m['vol_ratio'].iloc[idx])
taker = float(df_15m['taker_ratio'].iloc[idx])
print(f"  Volume:       {vol_now:,.0f}")
print(f"  Vol MA20:     {vol_ma:,.0f}  (ratio: {vol_now/vol_ma:.2f}x)")
print(f"  Vol Ratio:    {vol_r:.2f}x  (24h vs 7d)")
print(f"  Taker Ratio:  {taker:.4f}  ({'buyers' if taker > 0.52 else 'sellers' if taker < 0.48 else 'neutral'}, {taker*100:.1f}% buy)")

# Summary
print(f"\n  ── Verdict ──")
for direction in ('LONG', 'SHORT'):
    status, score, details = score_wyckoff(df_15m, df_2h, idx, idx_2h, direction, config=cfg)
    phase = details['wyckoff_phase']
    if status == 'PASS' and phase in ('SPRING', 'UPTHRUST') and not is_mid_range_noise(df_15m, idx, phase, config=cfg):
        print(f"  ✅ {direction}: {phase} detected (score={score:.3f})")
    else:
        print(f"  ⛔ {direction}: {phase or 'NO SIGNAL'} ({status}, score={score:.3f})")

print("\n" + "═" * 60)

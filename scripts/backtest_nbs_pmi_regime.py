#!/usr/bin/env python3
"""
NBS PMI Backtest — Regime-Conditional Analysis

Cross-tabulates PMI session returns against:
1. M22 Inflation Regime (PPI/CPI matrix)
2. M9 Volatility Regime (TREND / CHOP / SQUEEZE)
3. M21 Wyckoff Phase (ACCUMULATION / MARKUP / DISTRIBUTION / MARKDOWN / RANGE)
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import os, sys

# ── Load base backtest results ──
base_dir = os.path.dirname(os.path.dirname(__file__))
mfg_df = pd.read_csv(os.path.join(base_dir, 'nbs_pmi_backtest.csv'))
mfg_df['date_dt'] = pd.to_datetime(mfg_df['date'])

# ── Load ETH data ──
def load_eth(csv_path):
    df = pd.read_csv(csv_path)
    df['Open time'] = pd.to_datetime(df['Open time'])
    df = df.set_index('Open time').sort_index()
    if df.index.tz is not None:
        df.index = df.index.tz_localize(None)
    return df

eth = load_eth(os.path.join(base_dir, 'eth_15m_merged.csv'))
eth_daily = eth.resample('1D').agg({'Open': 'first', 'High': 'max', 'Low': 'min', 'Close': 'last', 'Volume': 'sum'}).dropna()

# ═══════════════════════════════════════════════════════════════
# REGIME PROXIES (simplified versions of framework logic)
# ═══════════════════════════════════════════════════════════════

def classify_wyckoff_proxy(df_daily, date):
    """Simplified Wyckoff phase from daily structure.
    
    Uses 50-day and 200-day trend + position in 60-day range.
    """
    cutoff = date - timedelta(days=1)
    hist = df_daily[df_daily.index <= cutoff].tail(200)
    if len(hist) < 60:
        return 'UNKNOWN'
    
    close = hist['Close']
    sma50 = close.rolling(50).mean().iloc[-1]
    sma200 = close.rolling(200).mean().iloc[-1] if len(hist) >= 200 else close.mean()
    current = close.iloc[-1]
    
    # 60-day range position
    recent60 = hist.tail(60)
    range_hi = recent60['High'].max()
    range_lo = recent60['Low'].min()
    range_pos = (current - range_lo) / (range_hi - range_lo) if range_hi != range_lo else 0.5
    
    # Trend direction (50-day slope)
    sma50_slope = (close.rolling(50).mean().iloc[-1] - close.rolling(50).mean().iloc[-5]) / 5 if len(hist) >= 55 else 0
    
    # Volume trend
    vol = hist['Volume']
    vol_trend = vol.tail(10).mean() / vol.tail(50).mean() if vol.tail(50).mean() > 0 else 1
    
    # Phase classification
    if current > sma50 > sma200 and sma50_slope > 0:
        if range_pos > 0.75 and vol_trend < 0.8:
            return 'DISTRIBUTION'
        return 'MARKUP'
    elif current < sma50 < sma200 and sma50_slope < 0:
        if range_pos < 0.25 and vol_trend < 0.8:
            return 'ACCUMULATION'
        return 'MARKDOWN'
    else:
        return 'RANGE'


def classify_vol_regime_proxy(df_daily, date):
    """Simplified volatility regime.
    
    TREND: ATR expanding + directional move
    SQUEEZE: ATR contracting (Bollinger squeeze)
    CHOP: ATR normal + no direction
    """
    cutoff = date - timedelta(days=1)
    hist = df_daily[df_daily.index <= cutoff].tail(50)
    if len(hist) < 20:
        return 'UNKNOWN'
    
    close = hist['Close']
    
    # ATR (14-day)
    tr = pd.concat([
        hist['High'] - hist['Low'],
        (hist['High'] - close.shift()).abs(),
        (hist['Low'] - close.shift()).abs()
    ], axis=1).max(axis=1)
    atr14 = tr.rolling(14).mean()
    
    if len(atr14.dropna()) < 10:
        return 'UNKNOWN'
    
    atr_now = atr14.iloc[-1]
    atr_avg = atr14.mean()
    atr_ratio = atr_now / atr_avg if atr_avg > 0 else 1
    
    # Bollinger Band width (squeeze detection)
    sma20 = close.rolling(20).mean()
    std20 = close.rolling(20).std()
    bb_width = (2 * std20 / sma20).iloc[-1] if sma20.iloc[-1] > 0 else 0
    bb_width_avg = (2 * std20 / sma20).mean()
    
    # Direction: 10-day return magnitude
    ret10 = (close.iloc[-1] / close.iloc[-10] - 1) * 100 if len(close) >= 10 else 0
    
    if bb_width < bb_width_avg * 0.6:
        return 'SQUEEZE'
    elif atr_ratio > 1.3 and abs(ret10) > 3:
        return 'TREND'
    elif atr_ratio < 0.7:
        return 'LOW_VOL'
    else:
        return 'CHOP'


def classify_inflation_regime_proxy(row):
    """Classify inflation regime from PMI data in the backtest row.
    
    Uses the mfg_pmi and services_pmi as proxies for economic regime.
    This is a simplification — real M22 uses PPI/CPI/Fed/labor data.
    """
    mfg = row.get('mfg_pmi', 50)
    services = row.get('services_pmi', 50)
    
    if mfg >= 51 and services >= 53:
        return 'GROWTH_STRONG'      # Analog: GOLDILOCKS / REFLATION
    elif mfg >= 50 and services >= 50:
        return 'GROWTH_MILD'        # Analog: DISINFLATION
    elif mfg >= 49 and services >= 50:
        return 'MIXED_WEAK_MFG'     # Analog: STAGFLATION_LITE
    elif mfg < 49 and services >= 50:
        return 'MFG_CONTRACTING'    # Analog: TIGHTENING
    elif mfg < 50 and services < 50:
        return 'BOTH_CONTRACTING'   # Analog: STAGFLATION / CRASH
    else:
        return 'NEUTRAL'


# ═══════════════════════════════════════════════════════════════
# COMPUTE REGIMES FOR EACH PMI RELEASE DATE
# ═══════════════════════════════════════════════════════════════

print("=" * 80)
print("  NBS PMI BACKTEST — REGIME-CONDITIONAL ANALYSIS")
print("=" * 80)

mfg_df['wyckoff'] = mfg_df['date_dt'].apply(lambda d: classify_wyckoff_proxy(eth_daily, d))
mfg_df['vol_regime'] = mfg_df['date_dt'].apply(lambda d: classify_vol_regime_proxy(eth_daily, d))
mfg_df['inflation_regime'] = mfg_df.apply(classify_inflation_regime_proxy, axis=1)

# Also add PMI signal
mfg_df['pmi_signal'] = mfg_df['mfg_pmi'].apply(
    lambda x: 'STRONG' if x >= 51 else 'EXPANDING' if x >= 50 else 'WEAK' if x >= 49 else 'CONTRACTING'
)

# ═══════════════════════════════════════════════════════════════
# ANALYSIS 1: WYCKOFF PHASE × PMI RETURNS
# ═══════════════════════════════════════════════════════════════

print("\n" + "=" * 80)
print("  1. WYCKOFF PHASE × NBS PMI RETURNS")
print("  Does PMI impact depend on where we are in the cycle?")
print("=" * 80)

sessions = ['ASIA_OPEN_ret', 'AUSTRALIA_MIDDAY_ret', 'EUROPE_MORNING_ret', 'US_OPEN_ret', 'ASIA_REOPEN_ret', '24h_ret']
sess_labels = ['Asia Open', 'Australia', 'Europe', 'US Open', 'Asia Reopen', '24h']

for phase in ['MARKUP', 'MARKDOWN', 'DISTRIBUTION', 'ACCUMULATION', 'RANGE']:
    phase_df = mfg_df[mfg_df['wyckoff'] == phase]
    if len(phase_df) < 3:
        continue
    
    print(f"\n  ── Wyckoff: {phase}  (n={len(phase_df)}) ──")
    
    # PMI signal breakdown within this phase
    for pmi_sig in ['CONTRACTING', 'WEAK', 'EXPANDING', 'STRONG']:
        sub = phase_df[phase_df['pmi_signal'] == pmi_sig]
        if len(sub) < 2:
            continue
        rets_24h = sub['24h_ret'].dropna()
        rets_asia = sub['ASIA_OPEN_ret'].dropna()
        if len(rets_24h) > 0:
            print(f"    PMI {pmi_sig:12s} (n={len(sub):2d}): "
                  f"Asia={rets_asia.mean():+.2f}%  "
                  f"24h={rets_24h.mean():+.2f}%  "
                  f"win24h={((rets_24h>0).sum()/len(rets_24h)*100):.0f}%")
    
    # Overall in this phase
    for sess, label in zip(sessions, sess_labels):
        vals = phase_df[sess].dropna()
        if len(vals) > 0:
            print(f"    {label:14s}: avg={vals.mean():+.3f}%  win={((vals>0).sum()/len(vals)*100):.0f}%  n={len(vals)}")


# ═══════════════════════════════════════════════════════════════
# ANALYSIS 2: VOL REGIME × PMI RETURNS
# ═══════════════════════════════════════════════════════════════

print("\n" + "=" * 80)
print("  2. VOL REGIME × NBS PMI RETURNS")
print("  Does PMI impact depend on volatility state?")
print("=" * 80)

for vol in ['TREND', 'SQUEEZE', 'CHOP', 'LOW_VOL']:
    vol_df = mfg_df[mfg_df['vol_regime'] == vol]
    if len(vol_df) < 3:
        continue
    
    print(f"\n  ── Vol: {vol}  (n={len(vol_df)}) ──")
    
    for pmi_sig in ['CONTRACTING', 'WEAK', 'EXPANDING', 'STRONG']:
        sub = vol_df[vol_df['pmi_signal'] == pmi_sig]
        if len(sub) < 2:
            continue
        rets_24h = sub['24h_ret'].dropna()
        if len(rets_24h) > 0:
            print(f"    PMI {pmi_sig:12s} (n={len(sub):2d}): "
                  f"24h={rets_24h.mean():+.2f}%  "
                  f"win={((rets_24h>0).sum()/len(rets_24h)*100):.0f}%")
    
    # Overall
    for sess, label in zip(['ASIA_OPEN_ret', '24h_ret'], ['Asia Open', '24h']):
        vals = vol_df[sess].dropna()
        if len(vals) > 0:
            print(f"    {label:14s}: avg={vals.mean():+.3f}%  win={((vals>0).sum()/len(vals)*100):.0f}%  n={len(vals)}")


# ═══════════════════════════════════════════════════════════════
# ANALYSIS 3: INFLATION REGIME × PMI RETURNS
# ═══════════════════════════════════════════════════════════════

print("\n" + "=" * 80)
print("  3. ECONOMIC REGIME × NBS PMI RETURNS")
print("  GROWTH_STRONG=Goldilocks, MFG_CONTRACTING=Tightening, BOTH_CONTRACTING=Stagflation")
print("=" * 80)

for regime in ['GROWTH_STRONG', 'GROWTH_MILD', 'MIXED_WEAK_MFG', 'MFG_CONTRACTING', 'BOTH_CONTRACTING', 'NEUTRAL']:
    reg_df = mfg_df[mfg_df['inflation_regime'] == regime]
    if len(reg_df) < 2:
        continue
    
    print(f"\n  ── {regime}  (n={len(reg_df)}) ──")
    
    for sess, label in zip(sessions, sess_labels):
        vals = reg_df[sess].dropna()
        if len(vals) > 0:
            print(f"    {label:14s}: avg={vals.mean():+.3f}%  win={((vals>0).sum()/len(vals)*100):.0f}%  n={len(vals)}")


# ═══════════════════════════════════════════════════════════════
# ANALYSIS 4: BEST COMBOS — WHICH REGIME + PMI SIGNAL HAS EDGE?
# ═══════════════════════════════════════════════════════════════

print("\n" + "=" * 80)
print("  4. BEST REGIME + PMI COMBINATIONS (24h return)")
print("  Ranked by average 24h return, min 3 events")
print("=" * 80)

combos = []
for _, row in mfg_df.iterrows():
    combo = f"{row['wyckoff']}+{row['pmi_signal']}"
    combos.append({
        'combo': combo,
        'wyckoff': row['wyckoff'],
        'pmi_signal': row['pmi_signal'],
        'vol_regime': row['vol_regime'],
        'ret_24h': row.get('24h_ret', np.nan),
        'ret_asia': row.get('ASIA_OPEN_ret', np.nan),
        'ret_eu': row.get('EUROPE_MORNING_ret', np.nan),
        'ret_us': row.get('US_OPEN_ret', np.nan),
    })

combo_df = pd.DataFrame(combos)
combo_stats = combo_df.groupby('combo').agg(
    n=('ret_24h', 'count'),
    avg_24h=('ret_24h', 'mean'),
    med_24h=('ret_24h', 'median'),
    win_24h=('ret_24h', lambda x: (x > 0).sum() / len(x) * 100),
    avg_asia=('ret_asia', 'mean'),
    avg_eu=('ret_eu', 'mean'),
    avg_us=('ret_us', 'mean'),
).reset_index()

combo_stats = combo_stats[combo_stats['n'] >= 3].sort_values('avg_24h')

print(f"\n  {'Combo':<35s} {'n':>3s} {'24h Avg':>8s} {'24h Med':>8s} {'Win%':>5s} {'Asia':>7s} {'EU':>7s} {'US':>7s}")
print("  " + "-" * 80)
for _, r in combo_stats.iterrows():
    print(f"  {r['combo']:<35s} {r['n']:>3.0f} {r['avg_24h']:>+8.2f}% {r['med_24h']:>+8.2f}% {r['win_24h']:>4.0f}% {r['avg_asia']:>+7.2f}% {r['avg_eu']:>+7.2f}% {r['avg_us']:>+7.2f}%")


# ═══════════════════════════════════════════════════════════════
# ANALYSIS 5: WORST COMBOS — WHICH TO AVOID?
# ═══════════════════════════════════════════════════════════════

print("\n" + "=" * 80)
print("  5. TRIPLE COMBO: WYCKOFF + VOL + PMI (min 2 events)")
print("=" * 80)

triple = mfg_df.groupby(['wyckoff', 'vol_regime', 'pmi_signal']).agg(
    n=('24h_ret', 'count'),
    avg_24h=('24h_ret', 'mean'),
    win_24h=('24h_ret', lambda x: (x > 0).sum() / len(x) * 100),
    avg_asia=('ASIA_OPEN_ret', 'mean'),
).reset_index()

triple = triple[triple['n'] >= 2].sort_values('avg_24h')

print(f"\n  {'Wyckoff':<14s} {'Vol':<10s} {'PMI':<12s} {'n':>3s} {'24h Avg':>8s} {'Win%':>5s} {'Asia':>7s}")
print("  " + "-" * 65)
for _, r in triple.head(10).iterrows():
    print(f"  {r['wyckoff']:<14s} {r['vol_regime']:<10s} {r['pmi_signal']:<12s} {r['n']:>3.0f} {r['avg_24h']:>+8.2f}% {r['win_24h']:>4.0f}% {r['avg_asia']:>+7.2f}%")

print("  ...")
print(f"  {'(worst 10 above, best 10 below)':^65s}")
print("  " + "-" * 65)
for _, r in triple.tail(10).iterrows():
    print(f"  {r['wyckoff']:<14s} {r['vol_regime']:<10s} {r['pmi_signal']:<12s} {r['n']:>3.0f} {r['avg_24h']:>+8.2f}% {r['win_24h']:>4.0f}% {r['avg_asia']:>+7.2f}%")


# ═══════════════════════════════════════════════════════════════
# ANALYSIS 6: ACTIONABLE RULES
# ═══════════════════════════════════════════════════════════════

print("\n" + "=" * 80)
print("  6. ACTIONABLE RULES — IF ANY COMBO HAS EDGE")
print("=" * 80)

# Find combos with consistent negative 24h (short edge)
short_edge = combo_stats[(combo_stats['avg_24h'] < -1.0) & (combo_stats['n'] >= 3)]
long_edge = combo_stats[(combo_stats['avg_24h'] > 1.0) & (combo_stats['n'] >= 3)]

if len(short_edge) > 0:
    print("\n  🔴 SHORT EDGE (avg 24h < -1%, n ≥ 3):")
    for _, r in short_edge.iterrows():
        print(f"    {r['combo']}: avg={r['avg_24h']:+.2f}% win={r['win_24h']:.0f}% n={r['n']:.0f}")

if len(long_edge) > 0:
    print("\n  🟢 LONG EDGE (avg 24h > +1%, n ≥ 3):")
    for _, r in long_edge.iterrows():
        print(f"    {r['combo']}: avg={r['avg_24h']:+.2f}% win={r['win_24h']:.0f}% n={r['n']:.0f}")

if len(short_edge) == 0 and len(long_edge) == 0:
    print("\n  ⚪ No combo with n≥3 has avg 24h return > |1%|")
    print("  Edge is marginal or regime-dependent — not reliable enough for standalone signal")


# ── Save ──
output = os.path.join(base_dir, 'nbs_pmi_regime_backtest.csv')
mfg_df.to_csv(output, index=False)
print(f"\n💾 Saved: {output}")

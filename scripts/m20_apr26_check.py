#!/usr/bin/env python3
"""Check what M20 would have seen on Apr 26 during the rally."""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
import pandas as pd
import numpy as np
from src.utils.data_handler import load_data, resample_ohlcv
from src.utils.indicators import calc_atr, calc_vwap, calc_vol_ratio, calc_rsi, calc_ema
from src.modules.m20_failed_breakout import score_m20, _find_breakout_levels, _detect_breakout_attempt, _score_breakout_quality
from src.modules.m13_structure import score_m13
from src.modules.m9_volatility import RegimeState, compute_vol_regime
from src.modules.direction_resolver import resolve_direction
from src.config import CONFIG

df = load_data('eth_15m_merged.csv')
df['Open time'] = pd.to_datetime(df['Open time'])

# Need warmup + Apr 26
warmup_start = '2026-04-20'
target_start = '2026-04-26'
target_end = '2026-04-27'

mask_all = (df['Open time'] >= warmup_start) & (df['Open time'] < target_end)
df_all = df[mask_all].copy().reset_index(drop=True)

# Compute indicators
cfg = dict(CONFIG)
df_all['atr'] = calc_atr(df_all['High'], df_all['Low'], df_all['Close'], 14)
df_all['vwap'] = calc_vwap(df_all['High'], df_all['Low'], df_all['Close'], df_all['Volume'], 96)
df_all['vol_ma20'] = df_all['Volume'].rolling(20).mean()
df_all['rsi'] = calc_rsi(df_all['Close'], 14)
df_all['taker_ratio'] = df_all['Taker buy base asset volume'] / df_all['Volume'].replace(0, np.nan)

df_1h = resample_ohlcv(df_all, '1H')
df_1h['atr'] = calc_atr(df_1h['High'], df_1h['Low'], df_1h['Close'], 14)
df_1h['ema_fast'] = calc_ema(df_1h['Close'], cfg['EMA_FAST'])
df_1h['ema_slow'] = calc_ema(df_1h['Close'], cfg['EMA_SLOW'])

df_1d = resample_ohlcv(df_all, '1D')

# Filter Apr 26 bars
apr26_mask = (df_all['Open time'] >= target_start) & (df_all['Open time'] < target_end)
apr26_indices = df_all[apr26_mask].index.tolist()

# For each Apr 26 bar, check what M20 sees
print("=" * 100)
print("  M20 ANALYSIS ON APR 26 RALLY — What would the module have seen?")
print("=" * 100)

for idx in apr26_indices:
    ts = df_all['Open time'].iloc[idx]
    price = float(df_all['Close'].iloc[idx])
    high = float(df_all['High'].iloc[idx])
    atr = float(df_all['atr'].iloc[idx]) if not pd.isna(df_all['atr'].iloc[idx]) else 0

    # Find breakout levels
    levels = _find_breakout_levels(df_all, idx, 48)

    # Detect breakout attempts
    attempts = _detect_breakout_attempt(df_all, idx, levels, atr, cfg)

    # Score M20
    status, score, details = score_m20(df_all, idx, 'SHORT', config=cfg, atr_1h=atr)

    # Only print interesting bars (every 2 hours, or when something changes)
    hour = ts.hour
    minute = ts.minute
    if minute == 0 and hour % 2 == 0 or status == 'PASS' or (details.get('status') not in ('NO_BREAKOUT', 'NO_LEVELS', 'NO_ACTIONABLE')):
        m20_status = details.get('status', '?')
        m20_dir = details.get('breakout_direction', '')
        m20_level = details.get('level', 0)
        m20_quality = details.get('breakout_quality', 0)
        m20_fail = details.get('failure', {}).get('status', '')

        # Direction resolver
        idx_1h = min(len(df_1h) - 1, df_1h.index.searchsorted(df_all.index[idx]) if hasattr(df_all.index[idx], 'value') else len(df_1h) - 1)
        idx_1d = min(len(df_1d) - 1, len(df_1d) - 1)

        regime_state = RegimeState(config=cfg)
        vol_regime, m9_raw, _ = compute_vol_regime(df_all, df_1h, idx, idx_1h, regime_state=regime_state, config=cfg)
        m13_status, m13_score, m13_details = score_m13(df_1h, idx_1h, 'NEUTRAL', df_all, idx)
        m13_bias = m13_details.get('m13_bias', 'NEUTRAL')
        swing_bias = df_1d['swing_bias'].iloc[idx_1d] if 'swing_bias' in df_1d.columns else 'N/A'
        trend_dir = df_1d['trend'].iloc[idx_1d] if 'trend' in df_1d.columns else 'N/A'

        direction, _, dir_details = resolve_direction(
            vol_regime, m9_raw if m9_raw else 0.5,
            m13_bias, m13_score, m13_details,
            swing_bias_1d=swing_bias, trend_dir=trend_dir, config=cfg,
        )

        print(f"\n  {ts}  ${price:.2f}  H=${high:.2f}  ATR={atr:.2f}")
        print(f"    Direction: {direction}  (regime={vol_regime}, m13={m13_bias}, swing={swing_bias})")
        print(f"    M20: status={status}  score={score:.3f}  detail={m20_status}")

        if attempts:
            for a in attempts:
                q = _score_breakout_quality(df_all, a, cfg)
                print(f"      Breakout attempt: {a['direction']} @ ${a['level']:.2f}  "
                      f"quality={q['quality']:.2f}  signals={q['signals']}")
        else:
            print(f"      No breakout attempts detected ({len(levels)} levels checked)")

        if m20_status == 'FAILED':
            print(f"      ❌ FAILED BREAKOUT! {m20_dir} @ ${m20_level:.2f} → contrarian "
                  f"{details.get('contrarian_direction', '?')}  quality={m20_quality:.2f}")

# Also check what happened AFTER the rally (Apr 26 evening through Apr 27)
print(f"\n{'='*100}")
print(f"  POST-RALLY: What M20 saw as breakout failed (Apr 26 evening → Apr 27)")
print(f"{'='*100}")

# Check Apr 26 18:00 onwards (when breakout started) through Apr 27
late_mask = (df_all['Open time'] >= '2026-04-26 18:00') & (df_all['Open time'] < '2026-04-27 08:00')
late_indices = df_all[late_mask].index.tolist()

for idx in late_indices:
    ts = df_all['Open time'].iloc[idx]
    price = float(df_all['Close'].iloc[idx])
    high = float(df_all['High'].iloc[idx])
    atr = float(df_all['atr'].iloc[idx]) if not pd.isna(df_all['atr'].iloc[idx]) else 0

    status, score, details = score_m20(df_all, idx, 'SHORT', config=cfg, atr_1h=atr)
    m20_status = details.get('status', '?')

    # Print every bar where something interesting happens
    if m20_status not in ('NO_BREAKOUT', 'NO_LEVELS', 'NO_ACTIONABLE', 'HOLDING_WEAK'):
        m20_dir = details.get('breakout_direction', '')
        m20_level = details.get('level', 0)
        m20_quality = details.get('breakout_quality', 0)
        m20_fail = details.get('failure', {}).get('status', '')
        contrarian = details.get('contrarian_direction', '')

        icon = '❌' if m20_status == 'FAILED' else '⏳' if 'HOLDING' in m20_status else '🔍'
        print(f"  {icon} {ts}  ${price:.2f}  M20={m20_status}  score={score:.3f}  "
              f"{m20_dir} @ ${m20_level:.0f}  quality={m20_quality:.2f}  "
              f"fail={m20_fail}  → {contrarian}")

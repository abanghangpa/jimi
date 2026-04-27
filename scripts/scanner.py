#!/usr/bin/env python3
"""
JIMI Framework — Live Signal Scanner

Usage:
    python scripts/scanner.py
    python scripts/scanner.py --json
    python scripts/scanner.py --dashboard 8888
"""

import argparse
import sys
import os
import json
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from src.config import CONFIG
from src.utils.data_handler import fetch_recent
from src.utils.indicators import (
    calc_ema, calc_macd, calc_rsi, calc_atr, calc_vwap, calc_vol_ratio,
    calc_swing_bias, calc_phase0, calc_trend_state,
)
from src.modules.m1_macd import score_m1
from src.modules.m2_ema import score_m2
from src.modules.m3_vwap import score_m3
from src.modules.m4_cvd import calc_cvd_15m, detect_cvd_divergence_15m, calc_cvd_2h, detect_cvd_zero_cross, score_m4
from src.modules.m5_liquidation import (
    build_volume_profile, find_magnets, find_gaps, score_m5, detect_cascade_setup,
    find_support_resistance,
)
from src.modules.m6_derivatives import score_derivatives, get_derivatives_summary
from src.modules.m7_market_regime import m7_prepare_data, m7_get_row, score_m7
from src.modules.m8_funding import score_m8_funding
from src.engine import calc_ics, check_entry_filters, get_tp_multipliers


def compute_indicators(df_15m):
    """Compute all indicators on fresh data."""
    df_15m['vwap'] = calc_vwap(df_15m['High'], df_15m['Low'], df_15m['Close'], df_15m['Volume'], CONFIG['VWAP_LOOKBACK'])
    df_15m['vol_ma20'] = df_15m['Volume'].rolling(20).mean()
    taker_base = df_15m['Taker buy base asset volume']
    total_vol = df_15m['Volume']
    df_15m['taker_ratio'] = (taker_base / total_vol.replace(0, np.nan)).fillna(CONFIG['TAKER_FILLNA'])
    df_15m['atr'] = calc_atr(df_15m['High'], df_15m['Low'], df_15m['Close'], CONFIG['ATR_PERIOD'])
    df_15m['vol_ratio'] = calc_vol_ratio(df_15m['Volume'])

    df_1h = resample_ohlcv(df_15m, '1H')
    df_2h = resample_ohlcv(df_15m, '2H')
    df_4h = resample_ohlcv(df_15m, '4H')
    df_1d = resample_ohlcv(df_15m, '1D')

    df_1h['macd_line'], df_1h['macd_signal'], df_1h['macd_hist'] = calc_macd(
        df_1h['Close'], CONFIG['MACD_FAST'], CONFIG['MACD_SLOW'], CONFIG['MACD_SIGNAL'])
    df_1h['ema_fast'] = calc_ema(df_1h['Close'], CONFIG['EMA_FAST'])
    df_1h['ema_slow'] = calc_ema(df_1h['Close'], CONFIG['EMA_SLOW'])
    df_1h['atr'] = calc_atr(df_1h['High'], df_1h['Low'], df_1h['Close'], CONFIG['ATR_PERIOD'])
    df_1h['rsi'] = calc_rsi(df_1h['Close'], 14)
    df_4h['ema_fast'] = calc_ema(df_4h['Close'], CONFIG['EMA_FAST'])
    df_4h['ema_slow'] = calc_ema(df_4h['Close'], CONFIG['EMA_SLOW'])
    df_2h['ema_fast'] = calc_ema(df_2h['Close'], CONFIG['EMA_FAST'])
    df_2h['ema_slow'] = calc_ema(df_2h['Close'], CONFIG['EMA_SLOW'])
    df_15m['cvd_15m'] = calc_cvd_15m(df_15m)
    df_15m['cvd_divergence_15m'] = detect_cvd_divergence_15m(df_15m, CONFIG['CVD_LOOKBACK'], CONFIG['CVD_DIVERGENCE_WINDOW'])
    df_2h['cvd_2h'] = calc_cvd_2h(df_2h)
    df_2h['cvd_zl_state'], df_2h['cvd_zl_cross_bar'], df_2h['cvd_zl_cross_dir'] = detect_cvd_zero_cross(df_2h)
    df_1d['swing_bias'] = calc_swing_bias(df_1d)
    df_1d['phase0'] = calc_phase0(df_1d)
    df_1d['trend'], df_1d['trend_score'] = calc_trend_state(df_1d)
    df_4h['macd_line'], df_4h['macd_signal'], df_4h['macd_hist'] = calc_macd(
        df_4h['Close'], CONFIG['MACD_FAST'], CONFIG['MACD_SLOW'], CONFIG['MACD_SIGNAL'])
    df_15m['rsi'] = calc_rsi(df_15m['Close'], 14)

    return df_15m, df_1h, df_2h, df_4h, df_1d


# Need resample_ohlcv for scanner
from src.utils.data_handler import resample_ohlcv


def scan_signal(df_15m, df_1h, df_2h, df_4h, df_1d):
    """Scan current market for trading signals."""
    idx = len(df_15m) - 1
    row = df_15m.iloc[idx]
    ts = row['Open time']
    idx_1h, idx_2h, idx_4h, idx_1d = len(df_1h)-1, len(df_2h)-1, len(df_4h)-1, len(df_1d)-1
    atr_1h = df_1h['atr'].iloc[idx_1h]
    swing_bias = df_1d['swing_bias'].iloc[idx_1d]
    phase0_val = df_1d['phase0'].iloc[idx_1d]

    m1_dir, m1_score = score_m1(df_1h, idx_1h, CONFIG)
    m2_status, m2_score = score_m2(df_1h, df_2h, df_4h, df_1d, idx_1h, idx_2h, idx_4h, idx_1d)
    direction = 'LONG' if m1_dir == 'BULLISH' else 'SHORT' if m1_dir == 'BEARISH' else None

    result = {
        'timestamp': str(ts), 'price': float(row['Close']),
        'swing_bias': swing_bias, 'phase0': float(phase0_val) if not pd.isna(phase0_val) else None,
        'm1': {'direction': m1_dir, 'score': float(m1_score)},
        'm2': {'status': m2_status, 'score': float(m2_score)},
        'direction': direction,
    }

    # Volume Profile & Magnets
    highs = df_15m['High'].values.astype(float)
    lows = df_15m['Low'].values.astype(float)
    closes = df_15m['Close'].values.astype(float)
    volumes = df_15m['Volume'].values.astype(float)
    bin_centers, vol_profile, bin_edges = build_volume_profile(
        highs[:idx+1], lows[:idx+1], closes[:idx+1], volumes[:idx+1],
        n_bins=CONFIG['M5_VP_BINS'], lookback=CONFIG['M5_VP_LOOKBACK'])
    magnets = find_magnets(bin_centers, vol_profile) if bin_centers is not None else []
    gaps = find_gaps(bin_centers, vol_profile) if bin_centers is not None else []
    result['magnets'] = [(round(p, 2), round(s, 2)) for p, _, s in magnets[:5]]
    result['gaps'] = [round(p, 2) for p, _ in gaps[:5]]

    # S/R Levels
    sr_levels = find_support_resistance(df_15m, idx)
    sr_levels.sort(key=lambda x: x[1], reverse=True)
    result['sr_levels'] = [(round(p, 2), round(s, 2), t, touches, bounces)
                           for p, s, touches, bounces, t in sr_levels[:8]]

    # Derivatives
    try:
        deriv_summary = get_derivatives_summary()
        if 'error' not in deriv_summary:
            result['derivatives'] = deriv_summary
    except Exception:
        pass

    if direction is None:
        result['status'] = 'NO_SIGNAL'
        result['reason'] = 'M1 neutral'
        return result

    m3_status, m3_score, _ = score_m3(df_15m, idx, direction, CONFIG)
    result['m3'] = {'status': m3_status, 'score': float(m3_score)}

    m4_status, m4_score, m4_div = score_m4(df_15m, df_2h, idx, idx_2h, direction, CONFIG)
    result['m4'] = {'status': m4_status, 'score': float(m4_score), 'details': m4_div}

    m5_status, m5_score, m5_details = score_m5(df_15m, idx, direction, CONFIG,
        n_bins=CONFIG['M5_VP_BINS'], lookback=CONFIG['M5_VP_LOOKBACK'])
    result['m5'] = {'status': m5_status, 'score': float(m5_score), 'details': m5_details}

    ics, effective_floor = calc_ics(m1_score, m2_score, m3_score, m4_score, m4_status, m5_score, config=CONFIG)
    result['ics'] = float(ics)
    result['effective_floor'] = float(effective_floor)

    threshold = CONFIG['ICS_THRESHOLD_CAUTION'] if phase0_val and phase0_val >= 0.40 else CONFIG['ICS_THRESHOLD_NORMAL']
    result['threshold'] = float(threshold)

    if m3_status == 'FAIL':
        result['status'] = 'NO_SIGNAL'; result['reason'] = 'M3 VWAP fail'; return result
    if ics < effective_floor or ics < threshold:
        result['status'] = 'NO_SIGNAL'; result['reason'] = f'ICS {ics:.3f} < {threshold:.2f}'; return result

    passed, reason = check_entry_filters(df_15m, idx, direction, swing_bias, phase0_val, atr_1h, config=CONFIG)
    if not passed:
        result['status'] = 'FILTERED'; result['reason'] = reason; return result

    entry_price = float(row['Close'])
    atr_for_sl = float(atr_1h) if not pd.isna(atr_1h) else float(row['atr'])
    sl_dist = min(CONFIG['SL_ATR_STD'] * atr_for_sl, CONFIG['SL_HARD_MAX_PCT'] * entry_price)
    tp1_dist = CONFIG['TP1_ATR'] * atr_for_sl
    tp2_mult, tp3_mult = get_tp_multipliers(row.get('vol_ratio', np.nan), config=CONFIG)
    tp2_dist, tp3_dist = tp2_mult * atr_for_sl, tp3_mult * atr_for_sl

    if direction == 'LONG':
        sl, tp1, tp2, tp3 = entry_price - sl_dist, entry_price + tp1_dist, entry_price + tp2_dist, entry_price + tp3_dist
    else:
        sl, tp1, tp2, tp3 = entry_price + sl_dist, entry_price - tp1_dist, entry_price - tp2_dist, entry_price - tp3_dist

    result.update({
        'status': 'SIGNAL', 'entry': entry_price, 'sl': float(sl),
        'tp1': float(tp1), 'tp2': float(tp2), 'tp3': float(tp3),
        'sl_pct': float(abs(entry_price - sl) / entry_price * 100),
        'tp1_pct': float(abs(tp1 - entry_price) / entry_price * 100),
    })
    return result


def main():
    parser = argparse.ArgumentParser(description='JIMI Live Scanner')
    parser.add_argument('--json', action='store_true', help='Output JSON only')
    parser.add_argument('--dashboard', type=int, help='Run dashboard on port')
    args = parser.parse_args()

    if args.dashboard:
        print(f"Dashboard mode on port {args.dashboard} (not implemented in refactored version)")
        print("Use the legacy scanner for dashboard mode.")
        return

    print("Fetching recent data...")
    df_15m = fetch_recent(bars=1000)
    print("Computing indicators...")
    df_15m, df_1h, df_2h, df_4h, df_1d = compute_indicators(df_15m)
    print("Scanning...")
    result = scan_signal(df_15m, df_1h, df_2h, df_4h, df_1d)

    if args.json:
        print(json.dumps(result, indent=2, default=str))
    else:
        # Pretty print
        print("\n" + "═" * 60)
        print("  JIMI — LIVE SIGNAL SCAN")
        print("═" * 60)
        print(f"\n  Time:   {result['timestamp']}")
        print(f"  Price:  ${result['price']:.2f}")
        print(f"  Bias:   {result['swing_bias']}")
        print(f"  Phase0: {result.get('phase0', 'N/A')}")
        print(f"\n  Module Scores:")
        print(f"    M1 (1H MACD):  {result['m1']['direction']:>8}  score={result['m1']['score']:.2f}")
        print(f"    M2 (EMA conf): {result['m2']['status']:>8}  score={result['m2']['score']:.2f}")
        if 'm3' in result:
            print(f"    M3 (VWAP):     {result['m3']['status']:>8}  score={result['m3']['score']:.2f}")
        if 'm4' in result:
            print(f"    M4 (CVD):      {result['m4']['status']:>8}  score={result['m4']['score']:.2f}")
        if 'm5' in result:
            print(f"    M5 (LiqtMag):  {result['m5']['status']:>8}  score={result['m5']['score']:.2f}")
        if 'ics' in result:
            print(f"\n  ICS: {result['ics']:.3f}  (floor={result['effective_floor']:.3f})")

        status = result['status']
        if status == 'SIGNAL':
            print(f"\n  ✅ SIGNAL: {result['direction']}")
            print(f"    Entry: ${result['entry']:.2f}")
            print(f"    SL:    ${result['sl']:.2f}  ({result['sl_pct']:.2f}%)")
            print(f"    TP1:   ${result['tp1']:.2f}  ({result['tp1_pct']:.2f}%)")
            print(f"    TP2:   ${result['tp2']:.2f}")
            print(f"    TP3:   ${result['tp3']:.2f}")
        else:
            print(f"\n  ⛔ {status}: {result.get('reason', 'N/A')}")
        print("═" * 60)


if __name__ == '__main__':
    main()

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
from src.modules.m1_macd_v2 import score_m1_v2 as score_m1
from src.modules.m2_ema import score_m2
from src.modules.m3_vwap import score_m3
from src.modules.m4_cvd import calc_cvd_15m, detect_cvd_divergence_15m, calc_cvd_2h, detect_cvd_zero_cross, score_m4
from src.modules.m5_liquidation import (
    build_volume_profile, find_magnets, find_gaps, score_m5, detect_cascade_setup,
    find_support_resistance,
)
from src.modules.m6_derivatives import score_derivatives, get_derivatives_summary
from src.modules.m15_liq_levels import get_liquidity_summary
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


def _check_swept_magnets(df_15m, idx, magnets, lookback_bars=96):
    """Check if magnets have been swept by recent price action.

    A magnet is 'swept' if price has already traded through it in the
    last `lookback_bars` candles (default: 96 = 24h on 15m).
    Returns list of (price, strength, swept: bool, swept_at) tuples.
    """
    if not magnets:
        return []

    start = max(0, idx - lookback_bars + 1)
    recent_highs = df_15m['High'].values[start:idx+1].astype(float)
    recent_lows = df_15m['Low'].values[start:idx+1].astype(float)
    recent_times = df_15m['Open time'].values[start:idx+1]
    session_high = np.max(recent_highs)
    session_low = np.min(recent_lows)

    result = []
    for price, vol, strength in magnets:
        swept = False
        swept_at = None
        # For magnets above current price: swept if recent high already passed it
        # For magnets below current price: swept if recent low already passed it
        current = df_15m['Close'].values[idx]
        if price > current and session_high >= price:
            swept = True
            # Find the first candle that swept it
            for i in range(len(recent_highs)):
                if recent_highs[i] >= price:
                    swept_at = str(recent_times[i])
                    break
        elif price < current and session_low <= price:
            swept = True
            for i in range(len(recent_lows)):
                if recent_lows[i] <= price:
                    swept_at = str(recent_times[i])
                    break

        result.append((round(price, 2), round(strength, 2), swept, swept_at))

    return result


def scan_signal(df_15m, df_1h, df_2h, df_4h, df_1d):
    """Scan current market for trading signals."""
    idx = len(df_15m) - 1
    row = df_15m.iloc[idx]
    ts = row['Open time']
    idx_1h, idx_2h, idx_4h, idx_1d = len(df_1h)-1, len(df_2h)-1, len(df_4h)-1, len(df_1d)-1
    atr_1h = df_1h['atr'].iloc[idx_1h]
    swing_bias = df_1d['swing_bias'].iloc[idx_1d]
    phase0_val = df_1d['phase0'].iloc[idx_1d]

    m1_dir, m1_score, _m1_details = score_m1(df_1h, idx_1h, CONFIG, df_15m=df_15m, idx_15m=idx)
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
    # Check which magnets have already been swept by recent price action
    swept_magnets = _check_swept_magnets(df_15m, idx, magnets[:5])
    result['magnets'] = swept_magnets
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

    # Real Liquidity Levels (liquidation + stop clusters + order book)
    try:
        oi_usd = result.get('derivatives', {}).get('oi_usd', 0)
        ls_ratio = result.get('derivatives', {}).get('ls_ratio', 1.0)
        liq_summary = get_liquidity_summary(
            df_15m, idx, sr_levels, oi_usd, ls_ratio, direction)
        result['liquidity_levels'] = liq_summary
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


def print_signal(result):
    """Print detailed signal analysis with all module data."""
    print("\n" + "═" * 60)
    print("  JIMI — LIVE SIGNAL SCAN")
    print("═" * 60)
    print(f"\n  Time:   {result['timestamp']}")
    print(f"  Price:  ${result['price']:.2f}")
    print(f"  Bias:   {result['swing_bias']}")
    print(f"  Phase0: {result.get('phase0', 'N/A')}")

    # Market Data
    print(f"\n  Market Data:")
    vwap = result.get('vwap')
    vwap_dist = result.get('vwap_dist_pct')
    taker = result.get('taker_ratio')
    atr = result.get('atr_1h')
    vol_r = result.get('vol_ratio')
    if vwap:
        print(f"    VWAP:           ${vwap:.2f}  ({vwap_dist:+.2f}% from price)" if vwap_dist else f"    VWAP: ${vwap:.2f}")
    if taker is not None:
        taker_label = "buyers" if taker > 0.52 else "sellers" if taker < 0.48 else "neutral"
        print(f"    Taker Ratio:    {taker:.4f}  ({taker_label}, {taker*100:.1f}% buy)")
    if atr:
        print(f"    ATR (1H):       ${atr:.2f}  ({atr/result['price']*100:.2f}% of price)")
    if vol_r:
        print(f"    Vol Ratio:      {vol_r:.2f}x  (24h vs 7d)")

    # Module Scores
    print(f"\n  Module Scores:")
    print(f"    M1 (1H MACD):  {result['m1']['direction']:>8}  score={result['m1']['score']:.2f}")
    print(f"    M2 (EMA conf): {result['m2']['status']:>8}  score={result['m2']['score']:.2f}")
    if 'm3' in result:
        print(f"    M3 (VWAP):     {result['m3']['status']:>8}  score={result['m3']['score']:.2f}")
    if 'm4' in result:
        m4 = result['m4']
        det = m4.get('details', {})
        if isinstance(det, dict):
            div_str = det.get('layer_a_div', 'NONE')
            zl_str = det.get('layer_b_cross', 'NONE')
            la = det.get('layer_a_score', 0)
            lb = det.get('layer_b_score', 0)
            bars = det.get('layer_b_bars_since', '')
            print(f"    M4 (CVD):      {m4['status']:>8}  score={m4['score']:.2f}  "
                  f"div={div_str}({la:.2f})  zl={zl_str}({lb:.2f}) {bars}bars")
        else:
            print(f"    M4 (CVD):      {m4['status']:>8}  score={m4['score']:.2f}")
    if 'm5' in result:
        m5 = result['m5']
        print(f"    M5 (LiqtMag):  {m5['status']:>8}  score={m5['score']:.2f}  "
              f"nearest={m5['details'].get('nearest_magnet')} ({m5['details'].get('magnet_dist_pct')}%)")
    if 'cascade' in result and result['cascade'].get('cascade'):
        c = result['cascade']
        print(f"    ⚡ CASCADE:    momentum={c['momentum']}% vol_spike={c['vol_spike']}x range={c['range_expansion']}x")

    # Liquidation Magnets
    magnets = result.get('magnets', [])
    if magnets:
        print(f"\n  Liquidation Magnets (volume clusters):")
        price = result['price']
        for i, mag in enumerate(magnets[:5]):
            if len(mag) == 4:
                p, s, swept, swept_at = mag
            else:
                p, s = mag[0], mag[1]
                swept, swept_at = False, None
            dist = (p - price) / price * 100
            direction = "↑" if dist > 0 else "↓"
            swept_tag = f"  ✅ SWEPT @ {swept_at}" if swept else ""
            print(f"    #{i+1}: ${p:.2f}  strength={s:.2f}x  ({direction}{abs(dist):.2f}%){swept_tag}")

    # Support / Resistance
    sr = result.get('sr_levels', [])
    if sr:
        supports = [(p, s, t, tb, bb) for p, s, t, tb, bb in sr if t == 'SUPPORT']
        resistances = [(p, s, t, tb, bb) for p, s, t, tb, bb in sr if t == 'RESISTANCE']
        supports.sort(key=lambda x: x[1], reverse=True)
        resistances.sort(key=lambda x: x[1], reverse=True)
        if supports:
            print(f"  Support Levels:")
            for i, (p, s, _, touches, bounces) in enumerate(supports[:4]):
                dist = (p - price) / price * 100
                print(f"    #{i+1}: ${p:.2f}  strength={s:.1f}  touches={touches} bounces={bounces}  ({dist:+.2f}%)")
        if resistances:
            print(f"  Resistance Levels:")
            for i, (p, s, _, touches, bounces) in enumerate(resistances[:4]):
                dist = (p - price) / price * 100
                print(f"    #{i+1}: ${p:.2f}  strength={s:.1f}  touches={touches} bounces={bounces}  ({dist:+.2f}%)")

    # Derivatives
    deriv = result.get('derivatives', {})
    if deriv and 'error' not in deriv:
        print(f"\n  Derivatives Data:")
        oi_usd = deriv.get('oi_usd', 0)
        print(f"    OI:             {deriv.get('oi', 0):,.0f} ETH  (${oi_usd/1e9:.2f}B)  1h Δ: {deriv.get('oi_roc_1h', 0):+.3f}%")
        print(f"    L/S Ratio:      {deriv.get('ls_ratio', 0):.4f}  (long {deriv.get('long_pct', 0):.1f}% / short {deriv.get('short_pct', 0):.1f}%)  z={deriv.get('ls_zscore', 0):.2f}")
        pos = deriv.get('positioning', 'NEUTRAL')
        pos_icon = {'CROWDED_LONG': '🔴', 'CROWDED_SHORT': '🟢'}.get(pos, '⚪')
        print(f"    Positioning:    {pos_icon} {pos}")
        print(f"    Top Traders:    L/S={deriv.get('top_ls_ratio', 0):.4f}  whale={deriv.get('whale_signal', 'NEUTRAL')}  gap={deriv.get('whale_retail_gap', 0):+.4f}")
        print(f"    Futures Taker:  {deriv.get('futures_taker_ratio', 0):.4f}  flow={deriv.get('futures_flow', 'NEUTRAL')}")
        fr = deriv.get('funding_rate')
        if fr is not None:
            fr_label = "longs pay" if fr > 0 else "shorts pay"
            print(f"    Funding Rate:   {fr*100:+.4f}%  ({fr_label})")
        oi_div = deriv.get('oi_price_div', 'NONE')
        if oi_div != 'NONE':
            print(f"    ⚡ OI Divergence: {oi_div}")

    # Real Liquidity Levels (liquidation + stops + order book)
    liq = result.get('liquidity_levels', {})
    if liq:
        price = result['price']
        print(f"\n  Liquidity Levels (estimated):")
        print(f"    High cascade zones: {liq.get('high_cascade_zones', 0)}  |  "
              f"Bid walls: {liq.get('bid_walls', 0)}  Ask walls: {liq.get('ask_walls', 0)}")

        below = liq.get('below', [])
        if below:
            print(f"    ▼ Below ${price:.0f} (long liquidations / stops):")
            for z in below[:6]:
                icon = {'LONG_LIQ': '💥', 'LONG_STOP': '🛑', 'BID_WALL': '🟢',
                        'SHORT_LIQ': '💥', 'SHORT_STOP': '🛑', 'ASK_WALL': '🔴'}.get(z['type'], '•')
                cascade = z.get('cascade_risk', '')
                swept_tag = f"  ✅ SWEPT" if z.get('swept') else ""
                print(f"      {icon} ${z['price']:.2f}  {z['type']}  "
                      f"str={z['strength']:.0f}  cascade={cascade}  ({z['dist_pct']:+.2f}%){swept_tag}")

        above = liq.get('above', [])
        if above:
            print(f"    ▲ Above ${price:.0f} (short liquidations / stops):")
            for z in above[:6]:
                icon = {'LONG_LIQ': '💥', 'LONG_STOP': '🛑', 'BID_WALL': '🟢',
                        'SHORT_LIQ': '💥', 'SHORT_STOP': '🛑', 'ASK_WALL': '🔴'}.get(z['type'], '•')
                cascade = z.get('cascade_risk', '')
                swept_tag = f"  ✅ SWEPT" if z.get('swept') else ""
                print(f"      {icon} ${z['price']:.2f}  {z['type']}  "
                      f"str={z['strength']:.0f}  cascade={cascade}  ({z['dist_pct']:+.2f}%){swept_tag}")

    # ICS & Signal
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
        print_signal(result)


if __name__ == '__main__':
    main()

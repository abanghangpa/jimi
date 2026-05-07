#!/usr/bin/env python3
"""Run JIMI scanner on historical bars for a date range."""
import sys, os, json
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from src.config import CONFIG
from src.utils.data_handler import load_data, resample_ohlcv
from src.utils.indicators import (
    calc_ema, calc_macd, calc_rsi, calc_atr, calc_vwap, calc_vol_ratio,
    calc_swing_bias, calc_phase0, calc_trend_state,
)
from src.modules.m1_macd_v2 import score_m1_v2 as score_m1
from src.modules.m2_ema import score_m2
from src.modules.m3_vwap import score_m3
from src.modules.m4_cvd import calc_cvd_15m, detect_cvd_divergence_15m, calc_cvd_2h, detect_cvd_zero_cross, score_m4
from src.modules.m5_liquidation import build_volume_profile, find_magnets, find_gaps, score_m5, find_support_resistance
from src.modules.m9_volatility import RegimeState, compute_vol_regime, score_vol_regime
from src.modules.m13_structure import score_m13
from src.modules.direction_resolver import resolve_direction, score_targets
from src.engine import calc_ics

CSV_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'eth_15m_merged.csv')
START = '2026-04-24'
END = '2026-04-28'  # exclusive

def compute_all(df_15m, cfg):
    df_15m['vwap'] = calc_vwap(df_15m['High'], df_15m['Low'], df_15m['Close'], df_15m['Volume'], cfg['VWAP_LOOKBACK'])
    df_15m['vol_ma20'] = df_15m['Volume'].rolling(20).mean()
    taker_base = df_15m['Taker buy base asset volume']
    total_vol = df_15m['Volume']
    df_15m['taker_ratio'] = (taker_base / total_vol.replace(0, np.nan)).fillna(cfg['TAKER_FILLNA'])
    df_15m['atr'] = calc_atr(df_15m['High'], df_15m['Low'], df_15m['Close'], cfg['ATR_PERIOD'])
    df_15m['vol_ratio'] = calc_vol_ratio(df_15m['Volume'])
    df_15m['rsi'] = calc_rsi(df_15m['Close'], 14)

    df_1h = resample_ohlcv(df_15m, '1H')
    df_2h = resample_ohlcv(df_15m, '2H')
    df_4h = resample_ohlcv(df_15m, '4H')
    df_1d = resample_ohlcv(df_15m, '1D')

    df_1h['macd_line'], df_1h['macd_signal'], df_1h['macd_hist'] = calc_macd(
        df_1h['Close'], cfg['MACD_FAST'], cfg['MACD_SLOW'], cfg['MACD_SIGNAL'])
    df_1h['ema_fast'] = calc_ema(df_1h['Close'], cfg['EMA_FAST'])
    df_1h['ema_slow'] = calc_ema(df_1h['Close'], cfg['EMA_SLOW'])
    df_1h['atr'] = calc_atr(df_1h['High'], df_1h['Low'], df_1h['Close'], cfg['ATR_PERIOD'])
    df_1h['rsi'] = calc_rsi(df_1h['Close'], 14)
    df_4h['ema_fast'] = calc_ema(df_4h['Close'], cfg['EMA_FAST'])
    df_4h['ema_slow'] = calc_ema(df_4h['Close'], cfg['EMA_SLOW'])
    df_2h['ema_fast'] = calc_ema(df_2h['Close'], cfg['EMA_FAST'])
    df_2h['ema_slow'] = calc_ema(df_2h['Close'], cfg['EMA_SLOW'])
    df_15m['cvd_15m'] = calc_cvd_15m(df_15m)
    df_15m['cvd_divergence_15m'] = detect_cvd_divergence_15m(df_15m, cfg['CVD_LOOKBACK'], cfg['CVD_DIVERGENCE_WINDOW'])
    df_2h['cvd_2h'] = calc_cvd_2h(df_2h)
    df_2h['cvd_zl_state'], df_2h['cvd_zl_cross_bar'], df_2h['cvd_zl_cross_dir'] = detect_cvd_zero_cross(df_2h)
    df_1d['swing_bias'] = calc_swing_bias(df_1d)
    df_1d['phase0'] = calc_phase0(df_1d)
    df_1d['trend'], df_1d['trend_score'] = calc_trend_state(df_1d)
    df_4h['macd_line'], df_4h['macd_signal'], df_4h['macd_hist'] = calc_macd(
        df_4h['Close'], cfg['MACD_FAST'], cfg['MACD_SLOW'], cfg['MACD_SIGNAL'])

    return df_15m, df_1h, df_2h, df_4h, df_1d


def scan_bar(df_15m, df_1h, df_2h, df_4h, df_1d, idx, cfg):
    """Run full scanner pipeline on a single bar."""
    row = df_15m.iloc[idx]
    ts = str(row['Open time'])
    price = float(row['Close'])

    # Find matching indices in higher TFs
    idx_1h = df_1h.index.searchsorted(df_15m.index[idx]) if hasattr(df_15m.index[idx], 'value') else len(df_1h) - 1
    idx_2h = df_2h.index.searchsorted(df_15m.index[idx]) if hasattr(df_15m.index[idx], 'value') else len(df_2h) - 1
    idx_4h = df_4h.index.searchsorted(df_15m.index[idx]) if hasattr(df_15m.index[idx], 'value') else len(df_4h) - 1
    idx_1d = df_1d.index.searchsorted(df_15m.index[idx]) if hasattr(df_15m.index[idx], 'value') else len(df_1d) - 1

    # Clamp
    idx_1h = min(idx_1h, len(df_1h) - 1)
    idx_2h = min(idx_2h, len(df_2h) - 1)
    idx_4h = min(idx_4h, len(df_4h) - 1)
    idx_1d = min(idx_1d, len(df_1d) - 1)

    swing_bias = df_1d['swing_bias'].iloc[idx_1d] if 'swing_bias' in df_1d.columns else 'N/A'
    phase0_val = df_1d['phase0'].iloc[idx_1d] if 'phase0' in df_1d.columns else None
    trend_dir = df_1d['trend'].iloc[idx_1d] if 'trend' in df_1d.columns else 'N/A'

    result = {
        'timestamp': ts, 'price': price,
        'swing_bias': swing_bias,
        'phase0': float(phase0_val) if phase0_val is not None and not pd.isna(phase0_val) else None,
        'trend_dir': trend_dir,
    }

    # M9 Volatility Regime
    regime_state = RegimeState(config=cfg)
    vol_regime, m9_raw, m9_vol_details = compute_vol_regime(
        df_15m, df_1h, idx, idx_1h, regime_state=regime_state, config=cfg)
    result['regime'] = vol_regime

    # M13 Structural Bias
    m13_status, m13_score_raw, m13_details = score_m13(df_1h, idx_1h, 'NEUTRAL', df_15m, idx)
    m13_bias = m13_details.get('m13_bias', 'NEUTRAL')
    result['m13_bias'] = m13_bias
    result['m13_score'] = round(float(m13_score_raw), 3)

    # Direction
    direction, dir_size_mult, dir_details = resolve_direction(
        vol_regime, m9_raw if m9_raw else 0.5,
        m13_bias, m13_score_raw, m13_details,
        swing_bias_1d=swing_bias, trend_dir=trend_dir, config=cfg,
    )
    result['direction'] = direction
    result['dir_reason'] = dir_details.get('reason', '?')

    # Module scores
    m1_dir, m1_score, _ = score_m1(df_1h, idx_1h, cfg, df_15m=df_15m, idx_15m=idx)
    if m1_dir == 'BEARISH' and direction == 'LONG':
        m1_score = 1.0 - m1_score
    elif m1_dir == 'BULLISH' and direction == 'SHORT':
        m1_score = 1.0 - m1_score
    result['m1'] = {'dir': m1_dir, 'score': round(float(m1_score), 3)}

    m2_status, m2_score = score_m2(df_1h, df_2h, df_4h, df_1d, idx_1h, idx_2h, idx_4h, idx_1d)
    result['m2'] = {'status': m2_status, 'score': round(float(m2_score), 3)}

    m3_status, m3_score, _ = score_m3(df_15m, idx, direction, cfg)
    result['m3'] = {'status': m3_status, 'score': round(float(m3_score), 3)}

    m4_status, m4_score, m4_div = score_m4(df_15m, df_2h, idx, idx_2h, direction, cfg)
    m4_div_str = 'NONE'
    if isinstance(m4_div, dict):
        m4_div_str = m4_div.get('layer_a_div', 'NONE')
    result['m4'] = {'status': m4_status, 'score': round(float(m4_score), 3), 'div': m4_div_str}

    m5_status, m5_score, m5_details = score_m5(df_15m, idx, direction, cfg,
        n_bins=cfg['M5_VP_BINS'], lookback=cfg['M5_VP_LOOKBACK'])
    result['m5'] = {'status': m5_status, 'score': round(float(m5_score), 3)}

    m9_status, m9_score, _ = score_vol_regime(vol_regime, m9_raw, direction, trend_dir)
    result['m9'] = {'regime': vol_regime, 'score': round(float(m9_score), 3), 'status': m9_status}

    # ICS
    ics, floor = calc_ics(
        m1_score, m2_score, m3_score, m4_score, m4_status, m5_score,
        m9_score=m9_score, use_m9=True,
        config=cfg,
    )
    result['ics'] = round(float(ics), 4)
    result['threshold'] = cfg['ICS_THRESHOLD_NORMAL']

    # Entry filters
    from src.engine import check_entry_filters
    atr_1h = df_1h['atr'].iloc[idx_1h]
    passed, reason = check_entry_filters(df_15m, idx, direction, swing_bias, phase0_val, atr_1h, config=cfg)
    result['entry_pass'] = passed
    result['entry_reason'] = reason if not passed else 'OK'

    if ics < cfg['ICS_THRESHOLD_NORMAL']:
        result['status'] = 'NO_SIGNAL'
        result['reason'] = f'ICS {ics:.4f} < {cfg["ICS_THRESHOLD_NORMAL"]}'
    elif not passed:
        result['status'] = 'FILTERED'
        result['reason'] = reason
    else:
        result['status'] = 'SIGNAL'

    return result


def main():
    cfg = dict(CONFIG)
    print(f"Loading CSV...")
    df_all = load_data(CSV_PATH)
    print(f"  Total bars: {len(df_all)} ({df_all['Open time'].iloc[0]} → {df_all['Open time'].iloc[-1]})")

    # Filter to date range
    mask = (df_all['Open time'] >= START) & (df_all['Open time'] < END)
    df_range = df_all[mask].copy()
    print(f"  Target range: {START} → {END} ({len(df_range)} bars)")

    if len(df_range) == 0:
        print("No data in range!")
        return

    # We need warmup data for indicators — use a generous lookback
    warmup = 1000  # bars of warmup
    start_idx = max(0, df_all.index[df_range.index[0]] - warmup) if hasattr(df_range.index[0], '__int__') else 0
    # Just use all data up to the range start
    first_ts = df_range['Open time'].iloc[0]
    warmup_mask = df_all['Open time'] < first_ts
    df_warmup = df_all[warmup_mask].tail(warmup)
    df_compute = pd.concat([df_warmup, df_range]).reset_index(drop=True)

    print(f"Computing indicators on {len(df_compute)} bars (warmup + target)...")
    df_15m, df_1h, df_2h, df_4h, df_1d = compute_all(df_compute, cfg)

    # Find the target range indices
    range_start = len(df_warmup)
    range_end = len(df_compute)

    print(f"\n{'═'*80}")
    print(f"  JIMI SCANNER READINGS — {START} to {END}")
    print(f"{'═'*80}")

    # Scan every 4th bar (hourly) to keep output manageable, or every bar for 15m
    step = 1  # every 15m bar
    results = []

    for i in range(range_start, range_end, step):
        r = scan_bar(df_15m, df_1h, df_2h, df_4h, df_1d, i, cfg)
        results.append(r)

    # Print summary table
    print(f"\n  {'Timestamp':<22} {'Price':>9} {'Dir':>6} {'Regime':<10} {'Bias':>8} "
          f"{'M1':>6} {'M2':>6} {'M3':>6} {'M4':>6} {'M5':>6} {'M9':>6} {'ICS':>7} {'Status':<10}")
    print(f"  {'─'*22} {'─'*9} {'─'*6} {'─'*10} {'─'*8} "
          f"{'─'*6} {'─'*6} {'─'*6} {'─'*6} {'─'*6} {'─'*6} {'─'*7} {'─'*10}")

    for r in results:
        m1 = r.get('m1', {})
        m2 = r.get('m2', {})
        m3 = r.get('m3', {})
        m4 = r.get('m4', {})
        m5 = r.get('m5', {})
        m9 = r.get('m9', {})
        print(f"  {r['timestamp']:<22} {r['price']:>9.2f} {r.get('direction','?'):>6} "
              f"{r.get('regime','?'):<10} {r.get('m13_bias','?'):>8} "
              f"{m1.get('score',0):>6.2f} {m2.get('score',0):>6.2f} {m3.get('score',0):>6.2f} "
              f"{m4.get('score',0):>6.2f} {m5.get('score',0):>6.2f} {m9.get('score',0):>6.2f} "
              f"{r.get('ics',0):>7.4f} {r.get('status','?'):<10}")

    # Print signals and filtered
    signals = [r for r in results if r.get('status') == 'SIGNAL']
    filtered = [r for r in results if r.get('status') == 'FILTERED']

    print(f"\n  Total bars scanned: {len(results)}")
    print(f"  SIGNALS: {len(signals)}")
    print(f"  FILTERED: {len(filtered)}")
    print(f"  NO_SIGNAL: {len([r for r in results if r.get('status') == 'NO_SIGNAL'])}")

    if signals:
        print(f"\n  {'='*60}")
        print(f"  SIGNAL BARS:")
        print(f"  {'='*60}")
        for r in signals:
            print(f"  {r['timestamp']}  ${r['price']:.2f}  {r['direction']}  ICS={r['ics']:.4f}")

    if filtered:
        print(f"\n  Filtered bars (passed ICS but failed entry):")
        for r in filtered[:20]:
            print(f"  {r['timestamp']}  ${r['price']:.2f}  {r['direction']}  ICS={r['ics']:.4f}  reason={r.get('entry_reason','')}")

    # Daily summary
    print(f"\n  {'='*60}")
    print(f"  DAILY SUMMARY:")
    print(f"  {'='*60}")
    dates = sorted(set(r['timestamp'][:10] for r in results))
    for d in dates:
        day_results = [r for r in results if r['timestamp'][:10] == d]
        day_signals = [r for r in day_results if r.get('status') == 'SIGNAL']
        day_filtered = [r for r in day_results if r.get('status') == 'FILTERED']
        directions = [r.get('direction', '?') for r in day_results]
        longs = directions.count('LONG')
        shorts = directions.count('SHORT')
        neutrals = directions.count('NEUTRAL')
        regimes = [r.get('regime', '?') for r in day_results]
        regime_mode = max(set(regimes), key=regimes.count) if regimes else '?'
        avg_ics = np.mean([r.get('ics', 0) for r in day_results])
        biases = [r.get('m13_bias', '?') for r in day_results]
        bias_mode = max(set(biases), key=biases.count) if biases else '?'
        prices = [r['price'] for r in day_results]
        print(f"\n  {d}:")
        print(f"    Price: ${min(prices):.2f} – ${max(prices):.2f}")
        print(f"    Direction: LONG={longs} SHORT={shorts} NEUTRAL={neutrals}")
        print(f"    Regime: {regime_mode}  |  M13 Bias: {bias_mode}")
        print(f"    Avg ICS: {avg_ics:.4f}")
        print(f"    Signals: {len(day_signals)}  |  Filtered: {len(day_filtered)}")
        if day_signals:
            for s in day_signals:
                print(f"      ✅ {s['timestamp'][-8:]}  ${s['price']:.2f}  {s['direction']}  ICS={s['ics']:.4f}")

if __name__ == '__main__':
    main()

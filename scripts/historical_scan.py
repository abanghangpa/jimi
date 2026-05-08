#!/usr/bin/env python3
"""Run JIMI scan on historical bars for a date range (4h intervals)."""

import sys, os, json, warnings
warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import numpy as np
import pandas as pd
from src.config import CONFIG
from src.utils.data_handler import resample_ohlcv, load_data
from src.utils.indicators import (
    calc_ema, calc_macd, calc_rsi, calc_atr, calc_vwap, calc_vol_ratio,
    calc_swing_bias, calc_phase0, calc_trend_state,
)
from src.modules.m1_macd_v2 import score_m1_v2 as score_m1
from src.modules.m2_ema import score_m2
from src.modules.m3_vwap import score_m3
from src.modules.m4_cvd import calc_cvd_15m, detect_cvd_divergence_15m, calc_cvd_2h, detect_cvd_zero_cross, score_m4
from src.modules.m5_liquidation import build_volume_profile, find_magnets, find_gaps, score_m5, find_support_resistance
from src.modules.m6_derivatives import get_derivatives_summary
from src.modules.m7_market_regime import m7_prepare_data, m7_get_row, score_m7
from src.modules.m8_funding import score_m8_funding
from src.modules.m10_macro import m10_prepare_data, m10_get_row, m10_compute_emas, score_m10_macro
from src.engine import calc_ics, check_entry_filters, run_gatekeepers
from src.modules.m9_volatility import RegimeState, compute_vol_regime, score_vol_regime
from src.modules.m13_structure import score_m13
from src.modules.direction_resolver import resolve_direction, score_targets
from src.modules.veto_system import evaluate_vetoes
from src.modules.coherence_liquidity import check_coherence
from src.modules.m14_sweep import score_m14
from src.modules.m18_squeeze import detect_squeeze_v6 as detect_squeeze
from src.modules.m20_failed_breakout import score_m20
from src.modules.m19_breakout_confirm import check_breakout_filters
from src.sl_tp import calc_trade_levels, check_sweep_gate

# ---- helpers from scanner ----
def compute_indicators(df_15m, config=None, df_1d_hist=None):
    cfg = config or CONFIG
    df_15m['vwap'] = calc_vwap(df_15m['High'], df_15m['Low'], df_15m['Close'], df_15m['Volume'], cfg['VWAP_LOOKBACK'])
    df_15m['vol_ma20'] = df_15m['Volume'].rolling(20).mean()
    taker_base = df_15m['Taker buy base asset volume']
    total_vol = df_15m['Volume']
    df_15m['taker_ratio'] = (taker_base / total_vol.replace(0, np.nan)).fillna(cfg['TAKER_FILLNA'])
    df_15m['atr'] = calc_atr(df_15m['High'], df_15m['Low'], df_15m['Close'], cfg['ATR_PERIOD'])
    df_15m['vol_ratio'] = calc_vol_ratio(df_15m['Volume'])

    df_1h = resample_ohlcv(df_15m, '1H')
    df_2h = resample_ohlcv(df_15m, '2H')
    df_4h = resample_ohlcv(df_15m, '4H')
    df_1d = df_1d_hist.copy() if df_1d_hist is not None and len(df_1d_hist) > 0 else resample_ohlcv(df_15m, '1D')

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
    df_15m['rsi'] = calc_rsi(df_15m['Close'], 14)
    return df_15m, df_1h, df_2h, df_4h, df_1d


def run_scan_at(df_15m_full, df_1d_hist, bar_idx, config=None):
    """Run full scan pipeline at a specific bar index. Returns result dict."""
    cfg = config or CONFIG
    # Slice up to bar_idx (inclusive) — simulate "live" at that moment
    df_15m = df_15m_full.iloc[:bar_idx+1].copy().reset_index(drop=True)
    if len(df_15m) < 100:
        return None

    df_15m, df_1h, df_2h, df_4h, df_1d = compute_indicators(df_15m, config=cfg, df_1d_hist=df_1d_hist)
    idx = len(df_15m) - 1
    row = df_15m.iloc[idx]
    ts = str(row['Open time'])
    idx_1h = len(df_1h) - 1
    idx_2h = len(df_2h) - 1
    idx_4h = len(df_4h) - 1
    idx_1d = len(df_1d) - 1
    atr_1h = df_1h['atr'].iloc[idx_1h]
    swing_bias = df_1d['swing_bias'].iloc[idx_1d]
    phase0_val = df_1d['phase0'].iloc[idx_1d]
    trend_dir = df_1d['trend'].iloc[idx_1d]
    trend_val = df_1d['trend_score'].iloc[idx_1d]

    result = {
        'timestamp': ts, 'price': float(row['Close']),
        'swing_bias': swing_bias, 'phase0': float(phase0_val) if not pd.isna(phase0_val) else None,
        'trend_dir': trend_dir, 'trend_val': float(trend_val),
    }

    # M9
    regime_state = RegimeState(config=cfg)
    vol_regime, m9_raw, _ = compute_vol_regime(df_15m, df_1h, idx, idx_1h, regime_state=regime_state, config=cfg)
    result['m9_regime'] = vol_regime

    # M13
    m13_status, m13_score_raw, m13_details = score_m13(df_1h, idx_1h, 'NEUTRAL', df_15m, idx)
    m13_bias = m13_details.get('m13_bias', 'NEUTRAL')

    # M7
    m7_score = 0.5; m7_status = 'SKIP'; m7_details = {}; m7_ethbtc_df = None
    if cfg.get('M7_ENABLED', False):
        try:
            m7_ethbtc_df, m7_btc_df = m7_prepare_data(df_15m)
            eb_row, bt_row = m7_get_row(m7_ethbtc_df, m7_btc_df, ts)
            m7_status, m7_score, m7_details = score_m7(eb_row, bt_row, row.get('vol_ratio', np.nan), 'NEUTRAL')
        except:
            pass

    # Targets
    highs = df_15m['High'].values.astype(float)
    lows = df_15m['Low'].values.astype(float)
    closes = df_15m['Close'].values.astype(float)
    volumes = df_15m['Volume'].values.astype(float)
    bin_centers, vol_profile, bin_edges = build_volume_profile(
        highs[:idx+1], lows[:idx+1], closes[:idx+1], volumes[:idx+1],
        n_bins=cfg['M5_VP_BINS'], lookback=cfg['M5_VP_LOOKBACK'])
    magnets = find_magnets(bin_centers, vol_profile) if bin_centers is not None else []
    gaps = find_gaps(bin_centers, vol_profile) if bin_centers is not None else []
    sr_levels = find_support_resistance(df_15m, idx)
    current_price = float(row['Close'])
    long_tgt_score, long_tgt_details = score_targets(current_price, magnets, gaps, sr_levels, 'LONG', atr_1h=atr_1h)
    short_tgt_score, short_tgt_details = score_targets(current_price, magnets, gaps, sr_levels, 'SHORT', atr_1h=atr_1h)

    # M20 pre-compute
    _m20_pre_score = None; _m20_pre_dir = None
    if cfg.get('M20_ENABLED', True):
        try:
            _s, _sc, _r = score_m20(df_15m, idx, 'NEUTRAL', sr_levels=sr_levels, magnets=magnets, config=cfg, atr_1h=atr_1h)
            if _r and _r.get('status') == 'FAILED':
                _m20_pre_dir = _r.get('contrarian_direction')
                _m20_pre_score = _sc
        except:
            pass

    # Direction
    direction, dir_size_mult, dir_details = resolve_direction(
        vol_regime, m9_raw if m9_raw else 0.5,
        m13_bias, m13_score_raw, m13_details,
        m7_score=m7_score, m7_status=m7_status,
        swing_bias_1d=swing_bias, trend_dir=trend_dir, config=cfg,
        long_target_score=long_tgt_score, short_target_score=short_tgt_score,
        long_target_details=long_tgt_details, short_target_details=short_tgt_details,
        m20_score=_m20_pre_score, m20_direction=_m20_pre_dir,
    )
    result['direction'] = direction
    result['m13_bias'] = m13_bias

    # Score modules
    m1_dir, m1_score, _ = score_m1(df_1h, idx_1h, cfg, df_15m=df_15m, idx_15m=idx)
    if m1_dir == 'BEARISH' and direction == 'LONG': m1_score = 1.0 - m1_score
    elif m1_dir == 'BULLISH' and direction == 'SHORT': m1_score = 1.0 - m1_score
    m2_status, m2_score = score_m2(df_1h, df_2h, df_4h, df_1d, idx_1h, idx_2h, idx_4h, idx_1d)
    m3_status, m3_score, _ = score_m3(df_15m, idx, direction, cfg)
    m4_status, m4_score, m4_div = score_m4(df_15m, df_2h, idx, idx_2h, direction, cfg)
    m5_status, m5_score, m5_details = score_m5(df_15m, idx, direction, cfg,
        n_bins=cfg['M5_VP_BINS'], lookback=cfg['M5_VP_LOOKBACK'])
    m9_status, m9_score, _ = score_vol_regime(vol_regime, m9_raw, direction, trend_dir)
    m10_score = 0.5; m10_status = 'SKIP'
    if cfg.get('M10_ENABLED', False):
        try:
            m10_data = m10_prepare_data(df_15m)
            if m10_data:
                m10_data = m10_compute_emas(m10_data)
                macro_row = m10_get_row(m10_data, ts)
                if macro_row:
                    m10_status, m10_score, _ = score_m10_macro(macro_row, direction, trend_dir)
        except: pass
    m11_score = 0.5; m11_status = 'SKIP'
    if cfg.get('M11_ENABLED', False):
        try:
            from src.modules.m11_momentum import score_m11_mtf_momentum
            m11_status, m11_score, _ = score_m11_mtf_momentum(df_15m, df_1h, df_4h, idx, idx_1h, idx_4h, direction)
        except: pass

    m13_status2, m13_score, _ = score_m13(df_1h, idx_1h, direction, df_15m, idx)
    m14_score = 0.5; m14_status = 'SKIP'
    _swing_levels = m13_details.get('swing_lows', []) if direction == 'LONG' else m13_details.get('swing_highs', [])
    if _swing_levels:
        m14_status, m14_score, _ = score_m14(df_15m, idx, direction, _swing_levels, config=cfg, magnets=magnets)

    # Squeeze
    rsi_val = float(df_15m['rsi'].iloc[idx]) if 'rsi' in df_15m.columns else 50
    vol_trend = float(df_15m['Volume'].iloc[idx] / df_15m['vol_ma20'].iloc[idx]) if 'vol_ma20' in df_15m.columns else 1.0
    atr_val = float(df_15m['atr'].iloc[idx]) if 'atr' in df_15m.columns else 0
    result['rsi'] = rsi_val
    result['vol_trend'] = vol_trend

    roll_high_48 = float(df_15m['High'].iloc[max(0,idx-47):idx+1].max())
    roll_low_48 = float(df_15m['Low'].iloc[max(0,idx-47):idx+1].min())
    range_width = (roll_high_48 - roll_low_48) / current_price * 100
    taker_base = float(df_15m['Taker buy base asset volume'].iloc[idx]) if 'Taker buy base asset volume' in df_15m.columns else 0
    total_vol = float(df_15m['Volume'].iloc[idx])
    taker_ratio = taker_base / total_vol if total_vol > 0 else 0.5

    sq_result = {'squeeze_type': 'NONE', 'squeeze_status': 'NONE', 'direction': 'NEUTRAL', 'squeeze_score': 0, 'ics_boost': 0}
    compression_history = []
    if idx >= 48:
        for i in range(max(0, idx - 47), idx):
            r48_h = float(df_15m['High'].iloc[max(0, i-47):i+1].max())
            r48_l = float(df_15m['Low'].iloc[max(0, i-47):i+1].min())
            r48_pct = (r48_h - r48_l) / float(df_15m['Close'].iloc[i]) * 100 if float(df_15m['Close'].iloc[i]) > 0 else 5.0
            vr = float(df_15m['Volume'].iloc[i] / df_15m['vol_ma20'].iloc[i]) if 'vol_ma20' in df_15m.columns and float(df_15m['vol_ma20'].iloc[i]) > 0 else 1.0
            br = (float(df_15m['High'].iloc[i]) - float(df_15m['Low'].iloc[i])) / float(df_15m['Close'].iloc[i]) * 100 if float(df_15m['Close'].iloc[i]) > 0 else 0.5
            tr_val = float(df_15m['Taker buy base asset volume'].iloc[i]) / float(df_15m['Volume'].iloc[i]) if float(df_15m['Volume'].iloc[i]) > 0 else 0.5
            compression_history.append((r48_pct, vr, br, tr_val))

    mock_result = {
        'range_width': range_width, 'vol_ratio': float(df_15m['vol_ratio'].iloc[idx]) if 'vol_ratio' in df_15m.columns else 0.15,
        'vol_ma20': float(df_15m['vol_ma20'].iloc[idx]) if 'vol_ma20' in df_15m.columns else 0,
        'oi_proxy': 1.0, 'vwap_dist': 0, 'bar_vol_spike': vol_trend,
        'bar_range_expansion': 1.0, 'bar_taker_extreme': taker_ratio > 0.65 or taker_ratio < 0.35,
        'raw_taker_ratio': taker_ratio, 'raw_bar_range_pct': (float(df_15m['High'].iloc[idx]) - float(df_15m['Low'].iloc[idx])) / current_price * 100,
        'squeeze_quality': 0.5, 'rsi': rsi_val, 'vol_trend': vol_trend, 'atr': atr_val,
    }
    sq_result = detect_squeeze(mock_result, config=cfg, compression_history=compression_history, df_15m=df_15m, magnets=magnets, sr_levels=sr_levels, liq_levels=None)
    result['squeeze_type'] = sq_result.get('squeeze_type', 'NONE')
    result['squeeze_dir'] = sq_result.get('direction', 'NEUTRAL')
    result['squeeze_status'] = sq_result.get('squeeze_status', 'NONE')
    result['squeeze_score'] = sq_result.get('squeeze_score', 0)

    # M20
    m20_score = 0.5; m20_status = 'SKIP'; m20_result = None
    if cfg.get('M20_ENABLED', True):
        try:
            m20_status, m20_score, m20_result = score_m20(df_15m, idx, direction, sr_levels=sr_levels, magnets=magnets, config=cfg, atr_1h=atr_1h)
            if m20_result:
                result['m20_status'] = m20_status
                result['m20_score'] = round(m20_score, 3)
                result['m20_breakout_dir'] = m20_result.get('breakout_direction', '')
                result['m20_contrarian'] = m20_result.get('contrarian_direction', '')
                result['m20_level'] = m20_result.get('level', 0)
        except: pass

    # ICS
    ics, effective_floor = calc_ics(
        m1_score, m2_score, m3_score, m4_score, m4_status, m5_score,
        m7_score=m7_score, m8_score=0.5, use_m7=False, use_m8=False,
        m9_score=m9_score, use_m9=True,
        m10_score=m10_score, use_m10=m10_status != 'SKIP',
        m11_score=m11_score, use_m11=m11_status != 'SKIP',
        m12_score=0.5, use_m12=False,
        m13_score=m13_score, use_m13=cfg.get('M13_ENABLED', False),
        m14_score=m14_score, use_m14=m14_status == 'PASS',
        config=cfg,
    )
    result['ics'] = round(float(ics), 4)
    result['m1_dir'] = m1_dir; result['m1_score'] = round(m1_score, 3)
    result['m2_status'] = m2_status; result['m2_score'] = round(m2_score, 3)
    result['m3_status'] = m3_status; result['m3_score'] = round(m3_score, 3)
    result['m4_status'] = m4_status; result['m4_score'] = round(m4_score, 3)
    result['m5_status'] = m5_status; result['m5_score'] = round(m5_score, 3)
    result['m9_score'] = round(m9_score, 3)
    result['m10_status'] = m10_status; result['m10_score'] = round(m10_score, 3)
    result['m11_status'] = m11_status; result['m11_score'] = round(m11_score, 3)

    # Threshold + signal check
    threshold = cfg['ICS_THRESHOLD_CAUTION'] if phase0_val and phase0_val >= 0.40 else cfg['ICS_THRESHOLD_NORMAL']
    result['threshold'] = round(float(threshold), 4)

    # Entry filters (with M20 Phase0 bypass)
    passed, reason = check_entry_filters(df_15m, idx, direction, swing_bias, phase0_val, atr_1h, config=cfg)
    _m20_direct_bypass = False
    if not passed:
        # M20 Phase0 death zone bypass
        if reason == 'phase0_death_zone' and m20_status == 'PASS' and m20_score >= cfg.get('M20_DIRECT_SIGNAL_THRESHOLD', 0.85):
            if m20_result and m20_result.get('status') == 'FAILED':
                _m20_direct_bypass = True
                result['m20_phase0_bypass'] = True
    result['entry_filter'] = passed or _m20_direct_bypass
    result['entry_reason'] = reason if not passed and not _m20_direct_bypass else ''

    # Gatekeepers
    gatekeeper = run_gatekeepers(direction, vol_regime, m7_score, m7_status, m7_details, m9_score, m9_status, m10_score, m10_status, trend_dir, config=cfg)
    result['gatekeeper'] = gatekeeper.passed

    # M20 direct signal path (bypass ICS gate)
    _m20_direct = False
    if ics < threshold or ics < effective_floor:
        m20_direct_threshold = cfg.get('M20_DIRECT_SIGNAL_THRESHOLD', 0.85)
        m20_override_active = dir_details.get('m20_override') is not None
        if (m20_status == 'PASS' and m20_score >= m20_direct_threshold and
                m20_result and m20_result.get('status') == 'FAILED' and m20_override_active):
            _m20_direct = True
            result['m20_direct_signal'] = True

    # Signal determination
    if _m20_direct or _m20_direct_bypass:
        # M20 direct signal — bypass ICS and Phase0
        entry_price = current_price
        levels = calc_trade_levels(entry_price, direction, float(atr_1h), row.get('vol_ratio', np.nan),
            magnets=magnets, sr_levels=sr_levels, liq_levels=None, cfg=cfg)
        result['status'] = 'SIGNAL'
        result['entry'] = entry_price
        result['sl'] = levels['sl']
        result['tp1'] = levels['tp1']
        result['sl_pct'] = levels['sl_pct']
        result['tp1_pct'] = levels['tp1_pct']
        result['m20_direct'] = True
    elif ics >= threshold and (passed or _m20_direct_bypass) and (gatekeeper.passed or _m20_direct_bypass) and m3_status != 'FAIL':
        entry_price = current_price
        levels = calc_trade_levels(entry_price, direction, float(atr_1h), row.get('vol_ratio', np.nan),
            magnets=magnets, sr_levels=sr_levels, liq_levels=None, cfg=cfg)
        result['status'] = 'SIGNAL'
        result['entry'] = entry_price
        result['sl'] = levels['sl']
        result['tp1'] = levels['tp1']
        result['sl_pct'] = levels['sl_pct']
        result['tp1_pct'] = levels['tp1_pct']
    else:
        result['status'] = 'NO_SIGNAL'
        if ics < threshold: result['reason'] = f'ICS {ics:.3f} < {threshold:.2f}'
        elif not passed and not _m20_direct_bypass: result['reason'] = reason
        elif not gatekeeper.passed: result['reason'] = 'Gatekeeper'
        elif m3_status == 'FAIL': result['reason'] = 'M3 FAIL'
        else: result['reason'] = 'filtered'

    return result


def main():
    csv_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'eth_15m_merged.csv')
    print(f"Loading CSV (optimized)...")

    # Read only what we need: ~1000 bars before Apr 24 for warmup + Apr 24-27 target
    # 1000 bars of 15m = ~10.4 days of warmup
    df = pd.read_csv(csv_path, parse_dates=['Open time'])
    df.columns = df.columns.str.strip()
    df['Open time'] = pd.to_datetime(df['Open time'].astype(str).str.strip(), format='mixed')
    for col in ['Open', 'High', 'Low', 'Close', 'Volume',
                'Quote asset volume', 'Number of trades',
                'Taker buy base asset volume', 'Taker buy quote asset volume']:
        df[col] = pd.to_numeric(df[col].astype(str).str.strip(), errors='coerce')
    df = df.sort_values('Open time').reset_index(drop=True)
    df = df.dropna(subset=['Open', 'High', 'Low', 'Close', 'Volume'])

    # Keep warmup (1200 bars before Apr 24) + target (Apr 24-27)
    cutoff = pd.Timestamp('2026-04-24') - pd.Timedelta(hours=1200*0.25)
    end = pd.Timestamp('2026-04-27 23:59:59')
    df_full = df[df['Open time'] >= cutoff].copy().reset_index(drop=True)
    print(f"Loaded {len(df_full)} bars (from {df_full['Open time'].iloc[0]} to {df_full['Open time'].iloc[-1]})")
    del df

    # Historical daily for warmup
    df_1d_hist = resample_ohlcv(df_full, '1D')
    print(f"Daily bars: {len(df_1d_hist)}")

    # Filter to Apr 24-27 for scanning
    mask = (df_full['Open time'] >= '2026-04-24') & (df_full['Open time'] <= end)
    target_indices = df_full[mask].index.tolist()

    # Sample every 16 bars = 4h intervals
    sampled = target_indices[::16]
    print(f"Scanning {len(sampled)} bars (4h intervals) from Apr 24-27...\n")

    results = []
    for i, bar_idx in enumerate(sampled):
        r = run_scan_at(df_full, df_1d_hist, bar_idx)
        if r:
            results.append(r)
            ts = r['timestamp']
            price = r['price']
            d = r['direction']
            st = r['status']
            ics = r['ics']
            regime = r.get('m9_regime', '?')
            sq = r.get('squeeze_type', 'NONE')
            sq_d = r.get('squeeze_dir', 'NEUTRAL')
            m20 = r.get('m20_status', 'SKIP')
            bias = r.get('swing_bias', '?')
            m13 = r.get('m13_bias', '?')
            rsi = r.get('rsi', 0)
            phase0 = r.get('phase0', 0) or 0

            signal_mark = '✅' if st == 'SIGNAL' else '⛔'
            sq_str = f"{sq}({sq_d})" if sq != 'NONE' else '—'
            m20_str = f"{m20}({r.get('m20_contrarian','')})" if m20 not in ('SKIP','DISABLED') else '—'

            print(f"{signal_mark} {ts}  ${price:>8.2f}  {d:>5}  ICS={ics:.3f}  "
                  f"regime={regime:<8} bias={bias:<7} M13={m13:<8} "
                  f"RSI={rsi:5.1f} P0={phase0:.3f}  "
                  f"SQ={sq_str:<20} M20={m20_str:<20}  "
                  f"{'→ '+st+': '+r.get('reason','') if st=='NO_SIGNAL' else '→ SIGNAL'}")

    # Summary
    signals = [r for r in results if r['status'] == 'SIGNAL']
    no_signals = [r for r in results if r['status'] != 'SIGNAL']
    print(f"\n{'='*80}")
    print(f"  Apr 24-27 Summary: {len(results)} bars scanned")
    print(f"  Signals: {len(signals)}  |  No Signal: {len(no_signals)}")
    if signals:
        print(f"\n  Active Signals:")
        for s in signals:
            print(f"    {s['timestamp']}  {s['direction']} @ ${s['price']:.2f}  "
                  f"SL ${s['sl']:.2f}({s['sl_pct']:.2f}%)  TP1 ${s['tp1']:.2f}({s['tp1_pct']:.2f}%)  ICS={s['ics']:.3f}")

    # Regime distribution
    from collections import Counter
    regimes = Counter(r.get('m9_regime', '?') for r in results)
    print(f"\n  Regime distribution: {dict(regimes)}")

    # Squeeze activity
    sq_active = [r for r in results if r.get('squeeze_type', 'NONE') != 'NONE']
    if sq_active:
        print(f"\n  Squeeze activity ({len(sq_active)} bars):")
        for s in sq_active:
            print(f"    {s['timestamp']}  {s['squeeze_type']}({s['squeeze_dir']}) "
                  f"status={s['squeeze_status']} score={s['squeeze_score']:.3f}")

    # M20 activity
    m20_active = [r for r in results if r.get('m20_status', 'SKIP') not in ('SKIP', 'DISABLED')]
    if m20_active:
        print(f"\n  M20 Failed Breakout activity ({len(m20_active)} bars):")
        for s in m20_active:
            print(f"    {s['timestamp']}  {s['m20_status']} "
                  f"breakout={s.get('m20_breakout_dir','')} → {s.get('m20_contrarian','')} "
                  f"@ ${s.get('m20_level',0):.2f}  score={s['m20_score']:.3f}")

if __name__ == '__main__':
    main()

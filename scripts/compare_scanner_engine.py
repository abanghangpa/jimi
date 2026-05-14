#!/usr/bin/env python3
"""Compare scanner output vs engine pipeline on the same data."""

import sys, os, json
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import numpy as np
import pandas as pd
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
from src.modules.m5_liquidation import score_m5
from src.modules.m7_market_regime import m7_prepare_data, m7_get_row, score_m7
from src.modules.m8_funding import score_m8_funding
from src.modules.m9_volatility import RegimeState, compute_vol_regime, score_vol_regime
from src.modules.m11_momentum import score_m11_mtf_momentum
from src.modules.m13_structure import score_m13
from src.modules.m14_sweep import score_m14
from src.modules.direction_resolver import resolve_direction
from src.engine import calc_ics, run_gatekeepers, check_entry_filters, get_tp_multipliers
from src.modules.veto_system import evaluate_vetoes
from src.modules.coherence_liquidity import check_coherence
from src.utils.data_handler import resample_ohlcv


def compute_indicators(df_15m, config=None):
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
    df_15m['rsi'] = calc_rsi(df_15m['Close'], 14)

    return df_15m, df_1h, df_2h, df_4h, df_1d


def engine_score(tf, df_base, df_1h, df_2h, df_4h, df_1d, config=None):
    """Replicate engine's scoring pipeline on the latest bar."""
    cfg = config or CONFIG
    idx = len(df_base) - 1
    row = df_base.iloc[idx]
    ts = row['Open time']
    idx_1h = len(df_1h) - 1
    idx_2h = len(df_2h) - 1
    idx_4h = len(df_4h) - 1
    idx_1d = len(df_1d) - 1

    atr_1h = df_1h['atr'].iloc[idx_1h]
    swing_bias = df_1d['swing_bias'].iloc[idx_1d]
    phase0_val = df_1d['phase0'].iloc[idx_1d]
    trend_dir = df_1d['trend'].iloc[idx_1d]
    trend_val = df_1d['trend_score'].iloc[idx_1d]

    result = {'tf': tf, 'timestamp': str(ts), 'price': float(row['Close'])}

    # M9
    regime_state = RegimeState(config=cfg)
    vol_regime, m9_raw, _ = compute_vol_regime(df_base, df_1h, idx, idx_1h, regime_state=regime_state, config=cfg)
    result['vol_regime'] = vol_regime

    block_regimes = cfg.get('M9_BLOCK_REGIMES', ['CRISIS'])
    if vol_regime in block_regimes:
        result['status'] = 'BLOCKED_M9'
        return result

    # M13
    m13_status, m13_score_raw, m13_details = score_m13(df_1h, idx_1h, 'NEUTRAL', df_base, idx)
    m13_bias = m13_details.get('m13_bias', 'NEUTRAL')

    # M7
    m7_score = 0.5; m7_status = 'SKIP'
    m7_ethbtc_df, m7_btc_df = None, None
    if cfg.get('M7_ENABLED', False):
        try:
            m7_ethbtc_df, m7_btc_df = m7_prepare_data(df_base)
            eb_row, bt_row = m7_get_row(m7_ethbtc_df, m7_btc_df, ts)
            m7_status, m7_score, _ = score_m7(eb_row, bt_row, row.get('vol_ratio', np.nan), 'NEUTRAL')
        except:
            pass

    # Direction
    direction, dir_size_mult, dir_details = resolve_direction(
        vol_regime, m9_raw if m9_raw else 0.5,
        m13_bias, m13_score_raw, m13_details,
        m7_score=m7_score, m7_status=m7_status,
        swing_bias_1d=swing_bias, trend_dir=trend_dir, config=cfg,
    )
    result['direction'] = direction
    result['dir_size_mult'] = float(dir_size_mult)

    if direction == 'NEUTRAL':
        result['status'] = 'NEUTRAL'
        result['reason'] = dir_details.get('reason', '')
        return result

    # Re-score with direction
    m9_status, m9_score, _ = score_vol_regime(vol_regime, m9_raw, direction, trend_dir)
    if cfg.get('M7_ENABLED', False) and m7_ethbtc_df is not None:
        eb_row, bt_row = m7_get_row(m7_ethbtc_df, m7_btc_df, ts)
        m7_status, m7_score, _ = score_m7(eb_row, bt_row, row.get('vol_ratio', np.nan), direction)
    m13_status, m13_score, m13_details = score_m13(df_1h, idx_1h, direction, df_base, idx)

    # All modules (same as scanner)
    m1_dir, m1_score, _ = score_m1(df_1h, idx_1h, cfg, df_15m=df_base, idx_15m=idx)
    m2_status, m2_score = score_m2(df_1h, df_2h, df_4h, df_1d, idx_1h, idx_2h, idx_4h, idx_1d)
    m3_status, m3_score, _ = score_m3(df_base, idx, direction, cfg)
    m4_status, m4_score, m4_div = score_m4(df_base, df_2h, idx, idx_2h, direction, cfg)
    # v7.2: Extract and normalize divergence string
    m4_div_str = 'NONE'
    if isinstance(m4_div, dict):
        m4_div_str = m4_div.get('layer_a_div', 'NONE')
    if m4_div_str.endswith('_BASE'):
        m4_div_str = m4_div_str.replace('_BASE', '')
    m5_status, m5_score, m5_details = score_m5(df_base, idx, direction, cfg)

    m8_score = 0.5; m8_status = 'SKIP'
    if cfg.get('M8_ENABLED', False):
        try:
            from src.modules.m6_derivatives import fetch_funding_rate
            fr_df = fetch_funding_rate("ETHUSDT", limit=1)
            if fr_df is not None and len(fr_df) > 0:
                fr = float(fr_df.iloc[-1].get('funding_rate', fr_df.iloc[-1].get('lastFundingRate', np.nan)))
                if not np.isnan(fr):
                    m8_status, m8_score, _ = score_m8_funding(fr, direction, cfg)
        except:
            pass

    m11_score = 0.5; m11_status = 'SKIP'
    if cfg.get('M11_ENABLED', False):
        try:
            m11_status, m11_score, _ = score_m11_mtf_momentum(df_base, df_1h, df_4h, idx, idx_1h, idx_4h, direction)
        except:
            pass

    m14_score = 0.5; m14_status = 'SKIP'
    if cfg.get('M14_ENABLED', True):
        _swing_levels = m13_details.get('swing_lows', []) if direction == 'LONG' else m13_details.get('swing_highs', [])
        if _swing_levels:
            m14_status, m14_score, _ = score_m14(df_base, idx, direction, _swing_levels, config=cfg)

    # ICS (identical to scanner)
    ics, effective_floor = calc_ics(
        m1_score, m2_score, m3_score, m4_score, m4_status, m5_score,
        m7_score=m7_score, m8_score=m8_score,
        use_m7=cfg.get('M7_ENABLED', False) and m7_ethbtc_df is not None,
        use_m8=m8_status != 'SKIP',
        m9_score=m9_score, use_m9=True,
        m10_score=0.5, use_m10=False,
        m11_score=m11_score, use_m11=m11_status != 'SKIP',
        m13_score=m13_score, use_m13=cfg.get('M13_ENABLED', False),
        m14_score=m14_score, use_m14=m14_status == 'PASS',
        config=cfg,
    )

    # Veto
    veto_penalty = 0.0
    if cfg.get('VETO_ENABLED', False):
        m4_disagree = (direction == 'LONG' and m4_div_str == 'BEARISH') or (direction == 'SHORT' and m4_div_str == 'BULLISH')
        m5_disagree = (m5_status == 'FAIL')
        dir_veto = m4_disagree and m5_disagree
        veto = evaluate_vetoes(cfg, vol_regime=vol_regime, dir_veto=dir_veto,
                               m9_status=m9_status, m10_status='SKIP', m11_status=m11_status)
        if veto.hard_blocked:
            result['status'] = 'VETOED'
            result['veto'] = veto.summary()
            return result
        veto_penalty = veto.soft_penalty

    # Coherence
    coherence_penalty = 0.0
    if cfg.get('COHERENCE_CHECK_ENABLED', True):
        is_coherent, conflicts, coherence_penalty = check_coherence(
            direction, m4_div, m5_details if isinstance(m5_details, dict) else {},
            m13_bias, vol_regime, m7_score=m7_score, m2_status=m2_status, config=cfg)
        if not is_coherent:
            result['status'] = 'COHERENCE_BLOCK'
            return result

    ics -= coherence_penalty
    ics += veto_penalty  # veto_penalty is negative or 0

    threshold = cfg['ICS_THRESHOLD_CAUTION'] if phase0_val and phase0_val >= 0.40 else cfg['ICS_THRESHOLD_NORMAL']

    # Gatekeeper
    gatekeeper = run_gatekeepers(direction, vol_regime, m7_score, m7_status, {},
                                  m9_score, m9_status, 0.5, 'SKIP', trend_dir, config=cfg)
    if not gatekeeper.passed:
        result['status'] = 'GATE_BLOCKED'
        return result

    # Entry filters
    passed, reason = check_entry_filters(df_base, idx, direction, swing_bias, phase0_val, atr_1h, config=cfg)
    if not passed:
        result['status'] = f'FILTERED: {reason}'
        return result

    result.update({
        'status': 'SIGNAL' if ics >= effective_floor and ics >= threshold else 'NO_SIGNAL',
        'ics': round(float(ics), 4),
        'effective_floor': round(float(effective_floor), 4),
        'threshold': round(float(threshold), 4),
        'm1': {'dir': m1_dir, 'score': round(float(m1_score), 3)},
        'm2': {'status': m2_status, 'score': round(float(m2_score), 3)},
        'm3': {'status': m3_status, 'score': round(float(m3_score), 3)},
        'm4': {'status': m4_status, 'score': round(float(m4_score), 3)},
        'm5': {'status': m5_status, 'score': round(float(m5_score), 3)},
        'm8': {'status': m8_status, 'score': round(float(m8_score), 3)},
        'm9': {'status': m9_status, 'score': round(float(m9_score), 3)},
        'm11': {'status': m11_status, 'score': round(float(m11_score), 3)},
        'm13': {'status': m13_status, 'score': round(float(m13_score), 3)},
        'm14': {'status': m14_status, 'score': round(float(m14_score), 3)},
    })
    return result


def main():
    tf_multipliers = {'1m': 15, '5m': 3, '15m': 1, '1h': 0.25}
    bars_map = {'1m': 3000, '5m': 2000, '15m': 1000, '1h': 500}

    for tf in ['1m', '5m', '15m', '1h']:
        tf_mult = tf_multipliers[tf]
        bars = bars_map[tf]

        scaled_config = dict(CONFIG)
        for k in ['VWAP_LOOKBACK', 'CVD_LOOKBACK', 'M4_ZL_LOOKBACK', 'M14_SWEEP_LOOKBACK', 'CROSS_ASSET_LOOKBACK']:
            if k in scaled_config:
                scaled_config[k] = max(int(CONFIG[k] * tf_mult), 10)
        scaled_config['_base_timeframe'] = tf

        print(f"\n{'='*60}")
        print(f"  ENGINE PIPELINE — {tf}")
        print(f"{'='*60}")

        df_base = fetch_recent(bars=bars, timeframe=tf)
        df_base, df_1h, df_2h, df_4h, df_1d = compute_indicators(df_base, config=scaled_config)
        result = engine_score(tf, df_base, df_1h, df_2h, df_4h, df_1d, config=scaled_config)

        for k, v in result.items():
            if k == 'tf':
                continue
            print(f"  {k}: {v}")


if __name__ == '__main__':
    main()

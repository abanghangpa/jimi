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
from src.engine import calc_ics, check_entry_filters, get_tp_multipliers, run_gatekeepers
from src.modules.m_conflict import get_conflict_stats
from src.modules.m9_volatility import RegimeState, compute_vol_regime, score_vol_regime
from src.modules.m13_structure import score_m13
from src.modules.direction_resolver import resolve_direction
from src.modules.veto_system import evaluate_vetoes
from src.modules.coherence_liquidity import check_coherence
from src.modules.m14_sweep import score_m14


def compute_indicators(df_15m, config=None):
    """Compute all indicators on fresh data."""
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


def _detect_cascade_risk(df, idx, result):
    """Detect whether the next liquidation cluster will be a quick flush or
    a full deleveraging cascade.

    Returns a dict with:
      - verdict: 'FLUSH' | 'CASCADE' | 'UNKNOWN'
      - score: 0.0 (likely flush) → 1.0 (likely cascade)
      - factors: list of contributing signals
    """
    deriv = result.get('derivatives', {})
    liq = result.get('liquidity_levels', {})
    direction = result.get('direction')
    score = 0.0
    factors = []

    # 1. OI velocity — fast OI drop = positions being force-closed
    oi_roc = deriv.get('oi_roc_1h', 0)  # % change per hour
    if oi_roc < -2.0:
        score += 0.25
        factors.append(f'OI dumping {oi_roc:+.2f}%/hr (cascade)')
    elif oi_roc < -1.0:
        score += 0.10
        factors.append(f'OI declining {oi_roc:+.2f}%/hr (deleveraging)')
    elif oi_roc > 1.0:
        score -= 0.10
        factors.append(f'OI rising {oi_roc:+.2f}%/hr (new positions)')

    # 2. Funding rate — extreme funding = crowded trade at risk
    fr = deriv.get('funding_rate')
    if fr is not None:
        if direction == 'LONG' and fr > 0.0005:
            score += 0.15
            factors.append(f'Funding {fr*100:+.4f}% (longs paying, cascade risk)')
        elif direction == 'SHORT' and fr < -0.0005:
            score += 0.15
            factors.append(f'Funding {fr*100:+.4f}% (shorts paying, cascade risk)')

    # 3. L/S z-score — extreme positioning = more cascade potential
    ls_z = deriv.get('ls_zscore', 0)
    if abs(ls_z) > 2.0:
        score += 0.20
        factors.append(f'L/S z={ls_z:.2f} (extreme positioning)')
    elif abs(ls_z) > 1.5:
        score += 0.10
        factors.append(f'L/S z={ls_z:.2f} (crowded)')

    # 4. Whale signal — whales leaning against the crowd = cascade amplified
    whale = deriv.get('whale_signal', 'NEUTRAL')
    if direction == 'LONG' and whale == 'WHALE_BEARISH':
        score += 0.15
        factors.append('Whales bearish (leaning against longs)')
    elif direction == 'SHORT' and whale == 'WHALE_BULLISH':
        score += 0.15
        factors.append('Whales bullish (leaning against shorts)')

    # 5. Order book depth — thin bids below = cascade through, thick = flush absorbed
    if liq:
        below = liq.get('below', [])
        above = liq.get('above', [])
        target_levels = below if direction == 'LONG' else above
        bid_walls = [z for z in target_levels if z['type'] == 'BID_WALL']
        ask_walls = [z for z in target_levels if z['type'] == 'ASK_WALL']
        walls = bid_walls if direction == 'LONG' else ask_walls

        if not walls:
            score += 0.15
            factors.append('No order book walls near cluster (thin support)')
        else:
            strongest = max(w['strength'] for w in walls)
            if strongest > 50:
                score -= 0.10
                factors.append(f'Order book wall str={strongest:.0f} (will absorb)')

    # 6. Recent momentum — accelerating sell-off = cascade momentum
    lookback = min(16, idx)  # 4h of candles
    if lookback >= 4:
        recent_closes = df['Close'].values[idx-lookback+1:idx+1].astype(float)
        momentum = (recent_closes[-1] - recent_closes[0]) / recent_closes[0] * 100
        if direction == 'LONG' and momentum < -1.5:
            score += 0.15
            factors.append(f'Price momentum {momentum:+.2f}% in 4h (accelerating down)')
        elif direction == 'SHORT' and momentum > 1.5:
            score += 0.15
            factors.append(f'Price momentum {momentum:+.2f}% in 4h (accelerating up)')

    # Clamp to 0-1
    score = max(0.0, min(1.0, score))

    if score >= 0.50:
        verdict = 'CASCADE'
    elif score >= 0.30:
        verdict = 'RISKY'
    else:
        verdict = 'FLUSH'

    return {
        'verdict': verdict,
        'score': round(score, 2),
        'factors': factors,
    }


def scan_signal(df_15m, df_1h, df_2h, df_4h, df_1d, config=None):
    """Scan current market for trading signals.

    Uses the same pipeline as the backtest engine:
      Phase 1: M9 (regime) → market climate
      Phase 2: M13 (structure) → swing direction
      Phase 2: M7 (macro) → ETH/BTC + BTC
      Phase 3: resolve_direction() → unified direction
      Phase 4: Score all modules (M1–M14) for ICS
      Phase 5: Veto + Coherence + Entry filters
    """
    cfg = config or CONFIG
    idx = len(df_15m) - 1
    row = df_15m.iloc[idx]
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

    result = {
        'timestamp': str(ts), 'price': float(row['Close']),
        'swing_bias': swing_bias, 'phase0': float(phase0_val) if not pd.isna(phase0_val) else None,
        'trend_dir': trend_dir, 'trend_val': float(trend_val),
    }

    # ── Phase 1: M9 Volatility Regime ──
    regime_state = RegimeState(config=cfg)
    vol_regime, m9_raw, m9_vol_details = compute_vol_regime(
        df_15m, df_1h, idx, idx_1h, regime_state=regime_state, config=cfg)
    result['m9'] = {'regime': vol_regime, 'raw': round(float(m9_raw), 3) if m9_raw else None}

    # Hard block on CRISIS
    block_regimes = cfg.get('M9_BLOCK_REGIMES', ['CRISIS'])
    if vol_regime in block_regimes:
        result['status'] = 'NO_SIGNAL'
        result['reason'] = f'M9 regime={vol_regime} (blocked)'
        return result

    # ── Phase 2: M13 Structural Bias ──
    m13_status, m13_score_raw, m13_details = score_m13(df_1h, idx_1h, 'NEUTRAL', df_15m, idx)
    m13_bias = m13_details.get('m13_bias', 'NEUTRAL')
    result['m13'] = {'bias': m13_bias, 'score': round(float(m13_score_raw), 3), 'status': m13_status}

    # ── Phase 2: M7 Macro (ETH/BTC + BTC) ──
    m7_ethbtc_df, m7_btc_df = None, None
    m7_score = 0.5
    m7_status = 'SKIP'
    m7_details = {}
    if cfg.get('M7_ENABLED', False):
        try:
            m7_ethbtc_df, m7_btc_df = m7_prepare_data(df_15m)
            eb_row, bt_row = m7_get_row(m7_ethbtc_df, m7_btc_df, ts)
            m7_status, m7_score, m7_details = score_m7(eb_row, bt_row, row.get('vol_ratio', np.nan), 'NEUTRAL')
            result['m7'] = {'score': round(float(m7_score), 3), 'status': m7_status}
        except Exception as e:
            result['m7'] = {'score': 0.5, 'status': 'SKIP', 'error': str(e)}

    # ── Phase 3: Resolve Direction ──
    direction, dir_size_mult, dir_details = resolve_direction(
        vol_regime, m9_raw if m9_raw else 0.5,
        m13_bias, m13_score_raw, m13_details,
        m7_score=m7_score, m7_status=m7_status,
        swing_bias_1d=swing_bias, trend_dir=trend_dir, config=cfg,
    )
    result['direction'] = direction
    result['direction_resolver'] = {
        'direction': direction, 'size_mult': round(float(dir_size_mult), 3),
        'action': dir_details.get('action', '?'),
        'reason': dir_details.get('reason', '?'),
    }

    if direction == 'NEUTRAL':
        result['status'] = 'NO_SIGNAL'
        result['reason'] = dir_details.get('reason', 'No direction resolved')
        return result

    # Re-score M9, M7, M13 with actual direction
    m9_status, m9_score, m9_details = score_vol_regime(vol_regime, m9_raw, direction, trend_dir)
    if cfg.get('M7_ENABLED', False) and m7_ethbtc_df is not None:
        eb_row, bt_row = m7_get_row(m7_ethbtc_df, m7_btc_df, ts)
        m7_status, m7_score, m7_details = score_m7(eb_row, bt_row, row.get('vol_ratio', np.nan), direction)
    m13_status, m13_score, m13_details = score_m13(df_1h, idx_1h, direction, df_15m, idx)
    result['m9']['score'] = round(float(m9_score), 3)
    result['m9']['status'] = m9_status
    result['m7']['score'] = round(float(m7_score), 3)
    result['m13']['score'] = round(float(m13_score), 3)

    # ── Phase 4: Score all modules ──
    # M1 (now an ICS contributor, not the gate)
    m1_dir, m1_score, _m1_details = score_m1(df_1h, idx_1h, cfg, df_15m=df_15m, idx_15m=idx)
    result['m1'] = {'direction': m1_dir, 'score': round(float(m1_score), 3)}

    # M2
    m2_status, m2_score = score_m2(df_1h, df_2h, df_4h, df_1d, idx_1h, idx_2h, idx_4h, idx_1d)
    result['m2'] = {'status': m2_status, 'score': round(float(m2_score), 3)}

    # M2 Veto
    if cfg.get('M2_VETO_ENABLED', False):
        m2_veto_thresh = cfg.get('M2_VETO_THRESHOLD', 0.40)
        if direction == 'LONG' and m2_status == 'BEARISH' and m2_score < m2_veto_thresh:
            result['status'] = 'NO_SIGNAL'
            result['reason'] = f'M2 veto: {m2_status} score={m2_score:.3f}'
            return result
        if direction == 'SHORT' and m2_status == 'BULLISH' and m2_score < m2_veto_thresh:
            result['status'] = 'NO_SIGNAL'
            result['reason'] = f'M2 veto: {m2_status} score={m2_score:.3f}'
            return result

    # M3
    m3_status, m3_score, _ = score_m3(df_15m, idx, direction, cfg)
    result['m3'] = {'status': m3_status, 'score': round(float(m3_score), 3)}

    # M4
    m4_status, m4_score, m4_div = score_m4(df_15m, df_2h, idx, idx_2h, direction, cfg)
    result['m4'] = {'status': m4_status, 'score': round(float(m4_score), 3), 'div': m4_div}

    # M5
    m5_status, m5_score, m5_details = score_m5(df_15m, idx, direction, cfg,
        n_bins=cfg['M5_VP_BINS'], lookback=cfg['M5_VP_LOOKBACK'])
    result['m5'] = {'status': m5_status, 'score': round(float(m5_score), 3)}

    # M8 (funding)
    m8_score = 0.5
    m8_status = 'SKIP'
    if cfg.get('M8_ENABLED', False):
        try:
            from src.modules.m6_derivatives import fetch_funding_rate
            fr_df = fetch_funding_rate("ETHUSDT", limit=1)
            if fr_df is not None and len(fr_df) > 0:
                fr = float(fr_df.iloc[-1].get('funding_rate', fr_df.iloc[-1].get('lastFundingRate', np.nan)))
                if not np.isnan(fr):
                    m8_status, m8_score, _ = score_m8_funding(fr, direction, cfg)
                    result['m8'] = {'status': m8_status, 'score': round(float(m8_score), 3), 'rate': round(fr, 6)}
        except Exception:
            pass

    # M10 (cross-asset macro) — skip in live scanner (requires historical data)
    m10_score = 0.5
    m10_status = 'SKIP'

    # M11 (MTF momentum)
    m11_score = 0.5
    m11_status = 'SKIP'
    if cfg.get('M11_ENABLED', False):
        try:
            from src.modules.m11_momentum import score_m11_mtf_momentum
            m11_status, m11_score, _ = score_m11_mtf_momentum(
                df_15m, df_1h, df_4h, idx, idx_1h, idx_4h, direction)
            result['m11'] = {'status': m11_status, 'score': round(float(m11_score), 3)}
        except Exception:
            pass

    # M14 (sweep-retest-reclaim)
    m14_score = 0.5
    m14_status = 'SKIP'
    if cfg.get('M14_ENABLED', True):
        _swing_levels = m13_details.get('swing_lows', []) if direction == 'LONG' else m13_details.get('swing_highs', [])
        if _swing_levels:
            m14_status, m14_score, _ = score_m14(df_15m, idx, direction, _swing_levels, config=cfg)
            result['m14'] = {'status': m14_status, 'score': round(float(m14_score), 3)}

    # ── ICS ──
    ics, effective_floor = calc_ics(
        m1_score, m2_score, m3_score, m4_score, m4_status, m5_score,
        m7_score=m7_score, m8_score=m8_score,
        use_m7=cfg.get('M7_ENABLED', False) and m7_ethbtc_df is not None,
        use_m8=m8_status != 'SKIP',
        m9_score=m9_score, use_m9=True,
        m10_score=m10_score, use_m10=m10_status != 'SKIP',
        m11_score=m11_score, use_m11=m11_status != 'SKIP',
        m13_score=m13_score, use_m13=cfg.get('M13_ENABLED', False),
        m14_score=m14_score, use_m14=m14_status == 'PASS',
        config=cfg,
    )
    result['ics'] = round(float(ics), 4)
    result['effective_floor'] = round(float(effective_floor), 4)

    # ── Phase 5: Veto + Coherence + Filters ──
    # Veto
    if cfg.get('VETO_ENABLED', False):
        m4_disagree = (direction == 'LONG' and m4_div == 'BEARISH') or (direction == 'SHORT' and m4_div == 'BULLISH')
        m5_disagree = (m5_status == 'FAIL')
        dir_veto = m4_disagree and m5_disagree

        veto = evaluate_vetoes(
            cfg, vol_regime=vol_regime,
            dir_veto=dir_veto,
            m9_status=m9_status, m10_status=m10_status, m11_status=m11_status,
        )
        if veto.hard_blocked:
            result['status'] = 'NO_SIGNAL'
            result['reason'] = f'Veto: {veto.summary()}'
            result['veto'] = veto.summary()
            return result
        result['veto'] = veto.summary() if veto.soft_vetoes else 'CLEAR'

    # Coherence check
    if cfg.get('COHERENCE_CHECK_ENABLED', True):
        is_coherent, conflicts, coherence_penalty = check_coherence(
            direction, m4_div, m5_details if isinstance(m5_details, dict) else {},
            m13_bias, vol_regime, m7_score=m7_score, m2_status=m2_status, config=cfg,
        )
        if not is_coherent:
            result['status'] = 'NO_SIGNAL'
            result['reason'] = f'Coherence block: {", ".join(conflicts)}'
            return result
        ics -= coherence_penalty
        result['ics'] = round(float(ics), 4)

    # Threshold
    threshold = cfg['ICS_THRESHOLD_CAUTION'] if phase0_val and phase0_val >= 0.40 else cfg['ICS_THRESHOLD_NORMAL']
    result['threshold'] = round(float(threshold), 4)

    # M3 hard fail
    if m3_status == 'FAIL':
        result['status'] = 'NO_SIGNAL'
        result['reason'] = 'M3 VWAP fail'
        return result

    # ICS check
    if ics < effective_floor or ics < threshold:
        result['status'] = 'NO_SIGNAL'
        result['reason'] = f'ICS {ics:.4f} < threshold {threshold:.2f}'
        return result

    # Gatekeepers
    gatekeeper = run_gatekeepers(
        direction, vol_regime, m7_score, m7_status, m7_details,
        m9_score, m9_status, m10_score, m10_status, trend_dir, config=cfg,
    )
    if not gatekeeper.passed:
        result['status'] = 'NO_SIGNAL'
        result['reason'] = f'Gatekeeper: {gatekeeper.summary()}'
        return result

    # Entry filters
    passed, reason = check_entry_filters(df_15m, idx, direction, swing_bias, phase0_val, atr_1h, config=cfg)
    if not passed:
        result['status'] = 'FILTERED'
        result['reason'] = reason
        return result

    # ── SIGNAL ──
    entry_price = float(row['Close'])
    atr_for_sl = float(atr_1h) if not pd.isna(atr_1h) else float(row['atr'])
    sl_dist = min(cfg['SL_ATR_STD'] * atr_for_sl, cfg['SL_HARD_MAX_PCT'] * entry_price)
    tp1_dist = cfg['TP1_ATR'] * atr_for_sl
    tp2_mult, tp3_mult = get_tp_multipliers(row.get('vol_ratio', np.nan), config=cfg)
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

    # ── Add market data for display ──
    # Volume Profile & Magnets
    highs = df_15m['High'].values.astype(float)
    lows = df_15m['Low'].values.astype(float)
    closes = df_15m['Close'].values.astype(float)
    volumes = df_15m['Volume'].values.astype(float)
    bin_centers, vol_profile, bin_edges = build_volume_profile(
        highs[:idx+1], lows[:idx+1], closes[:idx+1], volumes[:idx+1],
        n_bins=cfg['M5_VP_BINS'], lookback=cfg['M5_VP_LOOKBACK'])
    magnets = find_magnets(bin_centers, vol_profile) if bin_centers is not None else []
    gaps = find_gaps(bin_centers, vol_profile) if bin_centers is not None else []
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

    # Liquidity Levels
    try:
        oi_usd = result.get('derivatives', {}).get('oi_usd', 0)
        ls_ratio = result.get('derivatives', {}).get('ls_ratio', 1.0)
        liq_summary = get_liquidity_summary(
            df_15m, idx, sr_levels, oi_usd, ls_ratio, direction)
        result['liquidity_levels'] = liq_summary
    except Exception:
        pass

    # Cascade Risk
    result['cascade_risk'] = _detect_cascade_risk(df_15m, idx, result)

    # Conflict History
    if direction and swing_bias:
        try:
            conflict = get_conflict_stats(
                'BULLISH' if direction == 'LONG' else 'BEARISH',
                swing_bias)
            if conflict:
                result['conflict'] = conflict
        except Exception:
            pass

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

    # Direction Resolver
    dr = result.get('direction_resolver', {})
    print(f"\n  Direction Resolver:")
    print(f"    Regime:        {result.get('m9', {}).get('regime', '?')}  (score={result.get('m9', {}).get('score', 0):.3f})")
    print(f"    Structure:     {result.get('m13', {}).get('bias', '?')}  (score={result.get('m13', {}).get('score', 0):.3f})")
    if 'm7' in result and result['m7'].get('status') != 'SKIP':
        print(f"    Macro M7:      {result['m7']['score']:.3f}  ({result['m7']['status']})")
    print(f"    → Direction:   {dr.get('direction', '?')}  size_mult={dr.get('size_mult', 0):.2f}")
    print(f"    Reason:        {dr.get('reason', '?')}")

    # Module Scores
    print(f"\n  Module Scores:")
    if 'm1' in result:
        print(f"    M1 (1H MACD):  {result['m1']['direction']:>8}  score={result['m1']['score']:.2f}")
    if 'm2' in result:
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
        print(f"    M5 (LiqtMag):  {m5['status']:>8}  score={m5['score']:.2f}")
    if 'm8' in result:
        print(f"    M8 (Funding):  {result['m8']['status']:>8}  score={result['m8']['score']:.2f}  rate={result['m8'].get('rate', 'N/A')}")
    if 'm9' in result:
        m9 = result['m9']
        m9_st = m9.get('status', '—')
        m9_sc = m9.get('score', 0)
        print(f"    M9 (VolRegime):{m9_st:>8}  score={m9_sc:.2f}  regime={m9['regime']}")
    if 'm11' in result:
        print(f"    M11 (MTF Mom): {result['m11']['status']:>8}  score={result['m11']['score']:.2f}")
    if 'm13' in result:
        m13 = result['m13']
        print(f"    M13 (Struct):  {m13['status']:>8}  score={m13['score']:.2f}  bias={m13['bias']}")
    if 'm14' in result:
        print(f"    M14 (Sweep):   {result['m14']['status']:>8}  score={result['m14']['score']:.2f}")
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

    # Conflict History
    conflict = result.get('conflict')
    if conflict and conflict.get('historical'):
        hist = conflict['historical']
        is_conflict = conflict['is_conflict']
        label = "CONFLICT" if is_conflict else "ALIGNED"
        print(f"\n  Conflict History ({conflict['m1_direction']} M1 vs {conflict['daily_bias']} daily — {label}):")
        print(f"    Historical signals: {hist['total_signals']}  ({hist['first_seen'][:10]} → {hist['last_seen'][:10]})")
        windows = hist.get('windows', {})
        if windows:
            print(f"    {'Window':>6}  {'Rev%':>6}  {'Win%':>6}  {'AvgNet':>8}  {'Avg↓':>8}  {'Avg↑':>8}  {'n':>4}")
            for wname in ['4h', '12h', '24h', '48h', '72h']:
                w = windows.get(wname)
                if w:
                    rev_icon = "⬇️" if w['reversal_rate'] > 55 else "⬆️" if w['reversal_rate'] < 45 else "↔️"
                    print(f"    {wname:>6}  {w['reversal_rate']:>5.1f}%  {w['win_rate']:>5.1f}%  "
                          f"{w['avg_net']:>+7.2f}%  {-w['avg_down']:>+7.2f}%  {w['avg_up']:>+7.2f}%  {w['n']:>4}  {rev_icon}")

    # Cascade Risk
    cr = result.get('cascade_risk', {})
    if cr:
        verdict = cr.get('verdict', 'UNKNOWN')
        icon = {'CASCADE': '🌊', 'RISKY': '⚠️', 'FLUSH': '💧'}.get(verdict, '❓')
        print(f"\n  Cascade Risk: {icon} {verdict}  (score={cr.get('score', 0):.2f})")
        for f in cr.get('factors', []):
            print(f"    • {f}")

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


def print_summary(result):
    """Print clean one-page summary with table + verdict."""
    price = result['price']
    deriv = result.get('derivatives', {})
    liq = result.get('liquidity_levels', {})

    # Module score table
    m1_dir = result.get('m1', {}).get('direction', 'N/A')
    m1_sc = result.get('m1', {}).get('score', 0)
    m2_st = result.get('m2', {}).get('status', 'N/A')
    m2_sc = result.get('m2', {}).get('score', 0)
    m3_st = result.get('m3', {}).get('status', 'N/A') if 'm3' in result else '—'
    m3_sc = result.get('m3', {}).get('score', 0) if 'm3' in result else 0
    m4_st = result.get('m4', {}).get('status', 'N/A') if 'm4' in result else '—'
    m4_sc = result.get('m4', {}).get('score', 0) if 'm4' in result else 0
    m5_st = result.get('m5', {}).get('status', 'N/A') if 'm5' in result else '—'
    m5_sc = result.get('m5', {}).get('score', 0) if 'm5' in result else 0
    m8_st = result.get('m8', {}).get('status', '—') if 'm8' in result else '—'
    m8_sc = result.get('m8', {}).get('score', 0) if 'm8' in result else 0
    m9_st = result.get('m9', {}).get('status', '—') if 'm9' in result else '—'
    m9_sc = result.get('m9', {}).get('score', 0) if 'm9' in result else 0
    m11_st = result.get('m11', {}).get('status', '—') if 'm11' in result else '—'
    m11_sc = result.get('m11', {}).get('score', 0) if 'm11' in result else 0
    m13_st = result.get('m13', {}).get('status', '—') if 'm13' in result else '—'
    m13_sc = result.get('m13', {}).get('score', 0) if 'm13' in result else 0
    m14_st = result.get('m14', {}).get('status', '—') if 'm14' in result else '—'
    m14_sc = result.get('m14', {}).get('score', 0) if 'm14' in result else 0

    print("\n" + "─" * 55)
    print("  SUMMARY")
    print("─" * 55)
    print(f"  {'Module':<22} {'Status':>10}  {'Score':>6}")
    print(f"  {'M1 MACD (1H)':<22} {m1_dir:>10}  {m1_sc:>6.2f}")
    print(f"  {'M2 EMA confluence':<22} {m2_st:>10}  {m2_sc:>6.2f}")
    print(f"  {'M3 VWAP':<22} {m3_st:>10}  {m3_sc:>6.2f}")
    print(f"  {'M4 CVD':<22} {m4_st:>10}  {m4_sc:>6.2f}")
    print(f"  {'M5 Liquidation':<22} {m5_st:>10}  {m5_sc:>6.2f}")
    print(f"  {'M8 Funding':<22} {m8_st:>10}  {m8_sc:>6.2f}")
    print(f"  {'M9 Vol Regime':<22} {m9_st:>10}  {m9_sc:>6.2f}")
    print(f"  {'M11 MTF Momentum':<22} {m11_st:>10}  {m11_sc:>6.2f}")
    print(f"  {'M13 Structure':<22} {m13_st:>10}  {m13_sc:>6.2f}")
    print(f"  {'M14 Sweep':<22} {m14_st:>10}  {m14_sc:>6.2f}")

    ics = result.get('ics', 0)
    status = result.get('status', 'N/A')
    direction = result.get('direction', 'N/A')
    print(f"\n  ICS: {ics:.3f}", end="")
    if 'threshold' in result:
        print(f" (threshold {result['threshold']:.2f})", end="")
    print(f" → {'SIGNAL: ' + direction if status == 'SIGNAL' else 'NO SIGNAL'}")

    # Key observations
    print(f"\n  Key observations:")

    # Direction resolver
    dr = result.get('direction_resolver', {})
    m9 = result.get('m9', {})
    m13 = result.get('m13', {})
    if dr:
        print(f"  • Direction: {dr.get('direction', '?')} via resolver (regime={m9.get('regime', '?')}, structure={m13.get('bias', '?')})")

    # Price & bias
    swing = result.get('swing_bias', '')
    phase0 = result.get('phase0')
    phase_str = f", phase0={phase0}" if phase0 else ""
    print(f"  • Price ${price:.2f}, daily bias {swing}{phase_str}")

    # Derivatives
    if deriv and 'error' not in deriv:
        ls = deriv.get('ls_ratio', 0)
        lp = deriv.get('long_pct', 0)
        pos = deriv.get('positioning', '')
        whale = deriv.get('whale_signal', '')
        taker = deriv.get('futures_flow', '')
        fr = deriv.get('funding_rate')
        oi_usd = deriv.get('oi_usd', 0)

        crowd_label = f"({'crowded' if abs(deriv.get('ls_zscore', 0)) > 1.5 else 'neutral'})"
        print(f"  • L/S ratio {ls:.2f} ({lp:.1f}% long {crowd_label}), whales={whale}")
        print(f"  • Futures taker: {taker}")
        if fr is not None:
            fr_dir = "longs pay" if fr > 0 else "shorts pay"
            print(f"  • Funding rate: {fr*100:+.4f}% ({fr_dir})")
        print(f"  • OI: ${oi_usd/1e9:.2f}B  Δ1h: {deriv.get('oi_roc_1h', 0):+.3f}%")

    # Liquidity — unswept only
    if liq:
        above = [z for z in liq.get('above', []) if not z.get('swept')]
        below = [z for z in liq.get('below', []) if not z.get('swept')]

        if above:
            targets = []
            for z in above[:3]:
                icon = '💥' if 'LIQ' in z['type'] else '🛑'
                targets.append(f"{icon}${z['price']:.0f}({z['dist_pct']:+.1f}%)")
            print(f"  • Unswept above: {', '.join(targets)}")
        else:
            print(f"  • Unswept above: none — all swept")

        if below:
            targets = []
            for z in below[:3]:
                icon = '💥' if 'LIQ' in z['type'] else '🛑'
                targets.append(f"{icon}${z['price']:.0f}({z['dist_pct']:+.1f}%)")
            print(f"  • Unswept below: {', '.join(targets)}")
        else:
            print(f"  • Unswept below: none — all swept")

    # Support/Resistance
    sr = result.get('sr_levels', [])
    if sr:
        sup = [p for p, s, t, _, _ in sr if t == 'SUPPORT'][:2]
        res = [p for p, s, t, _, _ in sr if t == 'RESISTANCE'][:2]
        if sup or res:
            parts = []
            if sup:
                parts.append(f"support: {', '.join(f'${p:.0f}' for p in sup)}")
            if res:
                parts.append(f"resistance: {', '.join(f'${p:.0f}' for p in res)}")
            print(f"  • {' | '.join(parts)}")

    # Conflict history
    conflict = result.get('conflict')
    if conflict and conflict.get('historical'):
        hist = conflict['historical']
        windows = hist.get('windows', {})
        is_conflict = conflict['is_conflict']
        if is_conflict:
            label = "CONFLICT"
        else:
            label = "ALIGNED"
        w24 = windows.get('24h')
        w48 = windows.get('48h')
        parts_c = [f"{hist['total_signals']} historical signals"]
        if w24:
            parts_c.append(f"24h rev={w24['reversal_rate']:.0f}% net={w24['avg_net']:+.2f}%")
        if w48:
            parts_c.append(f"48h rev={w48['reversal_rate']:.0f}% net={w48['avg_net']:+.2f}%")
        print(f"  • {label} {conflict['m1_direction']} M1 vs {conflict['daily_bias']} daily: {'; '.join(parts_c)}")

    # Verdict
    print(f"\n  Verdict: ", end="")
    if status == 'SIGNAL':
        print(f"✅ {direction} signal — entry ${result['entry']:.2f}, "
              f"SL ${result['sl']:.2f} ({result['sl_pct']:.2f}%)")
    else:
        reasons = []
        if ics < result.get('threshold', 0.5):
            reasons.append(f"ICS {ics:.3f} below threshold")
        if result.get('swing_bias') == 'BEARISH' and direction == 'LONG':
            reasons.append("conflict: M1 bullish but daily bearish")
        if result.get('swing_bias') == 'BULLISH' and direction == 'SHORT':
            reasons.append("conflict: M1 bearish but daily bullish")
        if deriv.get('positioning') == 'CROWDED_LONG' and direction == 'LONG':
            reasons.append("crowded long")
        if deriv.get('positioning') == 'CROWDED_SHORT' and direction == 'SHORT':
            reasons.append("crowded short")
        if deriv.get('whale_signal') == 'WHALE_BEARISH' and direction == 'LONG':
            reasons.append("whales bearish")
        if deriv.get('whale_signal') == 'WHALE_BULLISH' and direction == 'SHORT':
            reasons.append("whales bullish")
        if deriv.get('futures_flow') == 'SELLERS_DOMINANT' and direction == 'LONG':
            reasons.append("seller-dominated flow")
        if deriv.get('futures_flow') == 'BUYERS_DOMINANT' and direction == 'SHORT':
            reasons.append("buyer-dominated flow")
        if not reasons:
            reasons.append(result.get('reason', 'unknown'))
        print(f"No trade — {'; '.join(reasons)}")
    print("─" * 55)


def main():
    parser = argparse.ArgumentParser(description='JIMI Live Scanner')
    parser.add_argument('--json', action='store_true', help='Output JSON only')
    parser.add_argument('--dashboard', type=int, help='Run dashboard on port')
    parser.add_argument('--tf', default='15m', choices=['1m', '5m', '15m', '1h'],
                        help='Base timeframe (default: 15m)')
    args = parser.parse_args()

    if args.dashboard:
        print(f"Dashboard mode on port {args.dashboard} (not implemented in refactored version)")
        print("Use the legacy scanner for dashboard mode.")
        return

    from datetime import datetime

    # Timeframe scaling: lookback bars are tuned for 15m, scale for other TFs
    tf_multipliers = {'1m': 15, '5m': 3, '15m': 1, '1h': 0.25}
    tf_mult = tf_multipliers[args.tf]
    bars_map = {'1m': 5000, '5m': 2000, '15m': 1000, '1h': 500}
    bars = bars_map[args.tf]

    # Scale config lookbacks for the selected timeframe
    scaled_config = dict(CONFIG)
    lookback_keys = [
        'VWAP_LOOKBACK', 'CVD_LOOKBACK', 'M4_ZL_LOOKBACK',
        'M14_SWEEP_LOOKBACK', 'CROSS_ASSET_LOOKBACK',
    ]
    for k in lookback_keys:
        if k in scaled_config:
            scaled_config[k] = max(int(CONFIG[k] * tf_mult), 10)

    print(f"Fetching {args.tf} data ({bars} bars)...")
    df_base = fetch_recent(bars=bars, timeframe=args.tf)
    print("Computing indicators...")
    df_base, df_1h, df_2h, df_4h, df_1d = compute_indicators(df_base, config=scaled_config)
    print(f"Scanning [{args.tf}]...")
    result = scan_signal(df_base, df_1h, df_2h, df_4h, df_1d, config=scaled_config)

    # Tag the result with timeframe
    result['timeframe'] = args.tf

    # Always save scan result to data/scans/
    scan_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data', 'scans')
    os.makedirs(scan_dir, exist_ok=True)
    ts = datetime.now().strftime('%Y%m%d_%H%M%S')
    scan_file = os.path.join(scan_dir, f'scan_{ts}.json')
    with open(scan_file, 'w') as f:
        json.dump(result, f, indent=2, default=str)
    print(f"\n  💾 Saved: {scan_file}")

    if args.json:
        print(json.dumps(result, indent=2, default=str))
    else:
        print_signal(result)
        print_summary(result)


if __name__ == '__main__':
    main()

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
from src.modules.intrabar_cvd import get_intrabar_cvd_summary, score_intrabar_divergence
from src.modules.m5_liquidation import (
    build_volume_profile, find_magnets, find_gaps, score_m5, detect_cascade_setup,
    find_support_resistance,
)
from src.modules.m6_derivatives import score_derivatives, get_derivatives_summary
from src.modules.m15_liq_levels import get_liquidity_summary
from src.modules.m7_market_regime import m7_prepare_data, m7_get_row, score_m7
from src.modules.m8_funding import score_m8_funding
from src.modules.m10_macro import m10_prepare_data, m10_get_row, m10_compute_emas, score_m10_macro
from src.engine import calc_ics, check_entry_filters, get_tp_multipliers, run_gatekeepers
from src.modules.m_conflict import get_conflict_stats
from src.modules.m9_volatility import RegimeState, compute_vol_regime, score_vol_regime
from src.modules.m13_structure import score_m13
from src.modules.direction_resolver import resolve_direction, score_targets
from src.modules.veto_system import evaluate_vetoes
from src.modules.coherence_liquidity import check_coherence
from src.modules.m12_orderbook import score_m12_orderbook
from src.modules.m14_sweep import score_m14
from src.modules.m17_resistance_quality import score_resistance_quality, format_resistance_quality
from src.modules.m16_exchange_activity import get_exchange_summary, fetch_all_exchange_data, compute_exchange_signals, score_exchange_activity, score_spot_signals
from src.sl_tp import calc_trade_levels, check_sweep_gate
from src.modules.conflict_resolver import detect_conflict, format_conflict, conflict_to_dict
from src.modules.power_of_3 import detect_phase, format_phase, phase_to_dict
from src.modules.m18_squeeze import detect_squeeze_v5 as detect_squeeze, format_squeeze


def compute_indicators(df_15m, config=None, df_1d_hist=None):
    """Compute all indicators on fresh data.

    Args:
        df_1d_hist: Optional pre-loaded daily DataFrame from historical CSV.
                    If provided, it replaces the daily resample from df_15m,
                    giving EMA55+ proper warmup for accurate daily bias.
    """
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

    # Use historical CSV daily data if available (fixes EMA55 warmup bug),
    # otherwise fall back to resampling from the limited live fetch.
    if df_1d_hist is not None and len(df_1d_hist) > 0:
        df_1d = df_1d_hist.copy()
    else:
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
from src.utils.data_handler import resample_ohlcv, load_data

# Minimum daily bars needed for reliable EMA55 convergence
_MIN_DAILY_BARS = 100


def ensure_csv_fresh(csv_path=None):
    """Ensure the historical CSV is patched to the latest 15m bar.

    Reads the last timestamp from the CSV. If any bars are missing,
    fetches them from Binance and appends — always up to date, no threshold.

    Returns:
        CSV path, or None if not found.
    """
    if csv_path is None:
        csv_path = os.path.join(os.path.dirname(os.path.dirname(__file__)),
                                'data', 'eth_15m_merged.csv')
        if not os.path.exists(csv_path):
            csv_path = os.path.join(os.path.dirname(os.path.dirname(__file__)),
                                    'eth_15m_merged.csv')

    if not os.path.exists(csv_path):
        print(f"  ⚠️  CSV not found at {csv_path}, using live fetch only")
        return None

    # Read last timestamp from CSV (fast: seek to end, read last line)
    last_line = None
    with open(csv_path, 'rb') as f:
        f.seek(0, 2)
        fsize = f.tell()
        f.seek(max(0, fsize - 500))
        lines = f.read().decode('utf-8', errors='replace').strip().split('\n')
        last_line = lines[-1] if lines else None

    if not last_line:
        return csv_path

    last_ts_str = last_line.split(',')[0].strip('"')
    try:
        last_dt = pd.Timestamp(last_ts_str)
    except Exception:
        return csv_path

    from datetime import datetime, timezone
    now = datetime.now(timezone.utc).replace(tzinfo=None)
    gap_bars = int((now - last_dt).total_seconds() / 900)  # 15m bars

    print(f"  📄 CSV last: {last_ts_str}  ({gap_bars} bars behind)")

    if gap_bars <= 0:
        return csv_path  # already current

    # Always fetch missing bars — no threshold
    print(f"  📥 Fetching {gap_bars} missing bars from Binance...")

    try:
        import ccxt
        ex = ccxt.binance({"enableRateLimit": True})
        since_ms = int(last_dt.timestamp() * 1000) + 1

        all_rows = []
        end_time_ms = None
        remaining = gap_bars

        while remaining > 0:
            limit = min(remaining, 1000)
            params = {'symbol': 'ETHUSDT', 'interval': '15m', 'limit': limit}
            if end_time_ms:
                params['endTime'] = end_time_ms
            raw = ex.publicGetKlines(params)
            if not raw:
                break
            all_rows = raw + all_rows
            end_time_ms = int(raw[0][0]) - 1
            remaining -= len(raw)
            if len(raw) < limit:
                break

        if all_rows:
            cols = ['Open time', 'Open', 'High', 'Low', 'Close', 'Volume',
                    'Close time', 'Quote asset volume', 'Number of trades',
                    'Taker buy base asset volume', 'Taker buy quote asset volume',
                    'Ignore']
            df_new = pd.DataFrame(all_rows, columns=cols)
            df_new['Open time'] = (pd.to_datetime(df_new['Open time'].astype(int), unit='ms')
                                  .dt.strftime('%Y-%m-%d %H:%M:%S'))
            df_new['Close time'] = (pd.to_datetime(df_new['Close time'].astype(int), unit='ms')
                                   .dt.strftime('%Y-%m-%d %H:%M:%S'))
            for c in ['Open', 'High', 'Low', 'Close', 'Volume', 'Quote asset volume',
                      'Number of trades', 'Taker buy base asset volume',
                      'Taker buy quote asset volume']:
                df_new[c] = pd.to_numeric(df_new[c])
            df_new['Ignore'] = 0

            df_new = df_new[df_new['Open time'] > last_ts_str]
            if len(df_new) > 0:
                df_new.to_csv(csv_path, mode='a', header=False, index=False)
                print(f"  ✅ Appended {len(df_new)} bars → {df_new['Open time'].iloc[-1]}")
    except Exception as e:
        print(f"  ⚠️  Fetch failed: {e}")

    return csv_path


def load_daily_from_csv(csv_path):
    """Load the full historical CSV and resample to daily.

    Returns a 1D DataFrame with swing_bias/phase0/trend pre-computed,
    or None if CSV is unavailable or too short.
    """
    if csv_path is None or not os.path.exists(csv_path):
        return None

    df_15m = load_data(csv_path)
    df_1d = resample_ohlcv(df_15m, '1D')

    n_bars = len(df_1d)
    if n_bars < _MIN_DAILY_BARS:
        print(f"  ⚠️  Only {n_bars} daily bars (need {_MIN_DAILY_BARS}), "
              f"bias may be unreliable")
        return None

    print(f"  📊 Daily: {n_bars} bars "
          f"({df_1d['Open time'].iloc[0]} → {df_1d['Open time'].iloc[-1]})")
    return df_1d





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

    # ── Prepare 15m data for M9/M13 (they expect 15m bars, not base TF) ──
    base_tf = cfg.get('_base_timeframe', '15m')
    if base_tf == '15m':
        df_15m_for_m9 = df_15m
        idx_15m_for_m9 = idx
    elif base_tf == '1h':
        # 1h data — M13 uses it directly as df_1h, pass same as df_15m (FVGs/OBs on 1h bars)
        df_15m_for_m9 = df_15m
        idx_15m_for_m9 = idx
    else:
        # 5m/1m — resample to 15m for M9/M13
        df_tmp = df_15m.copy().set_index('Open time')
        agg = {
            'Open': 'first', 'High': 'max', 'Low': 'min', 'Close': 'last',
            'Volume': 'sum', 'Quote asset volume': 'sum',
            'Number of trades': 'sum',
            'Taker buy base asset volume': 'sum',
            'Taker buy quote asset volume': 'sum',
        }
        df_15m_for_m9 = df_tmp.resample('15min').agg(agg).dropna(subset=['Open']).reset_index()
        df_15m_for_m9['atr'] = calc_atr(df_15m_for_m9['High'], df_15m_for_m9['Low'], df_15m_for_m9['Close'], cfg['ATR_PERIOD'])
        idx_15m_for_m9 = len(df_15m_for_m9) - 1

    # ── Phase 1: M9 Volatility Regime ──
    regime_state = RegimeState(config=cfg)
    vol_regime, m9_raw, m9_vol_details = compute_vol_regime(
        df_15m_for_m9, df_1h, idx_15m_for_m9, idx_1h, regime_state=regime_state, config=cfg)
    result['m9'] = {'regime': vol_regime, 'raw': round(float(m9_raw), 3) if m9_raw else None}

    # Regime block — record but don't early-return (all modules still score)
    block_regimes = cfg.get('M9_BLOCK_REGIMES', ['CRISIS'])
    regime_blocked = vol_regime in block_regimes
    if regime_blocked:
        result['regime_blocked'] = True

    # ── Phase 2: M13 Structural Bias ──
    m13_status, m13_score_raw, m13_details = score_m13(df_1h, idx_1h, 'NEUTRAL', df_15m_for_m9, idx_15m_for_m9)
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

    # ── Phase 2d: Pre-compute targets for direction resolver ──
    # Volume profile + S/R computed early so targets can inform direction
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

    atr_1h_val = df_1h['atr'].iloc[idx_1h] if idx_1h >= 0 else None
    current_price = float(row['Close'])
    long_tgt_score, long_tgt_details = score_targets(
        current_price, magnets, gaps, sr_levels, 'LONG', atr_1h=atr_1h_val)
    short_tgt_score, short_tgt_details = score_targets(
        current_price, magnets, gaps, sr_levels, 'SHORT', atr_1h=atr_1h_val)

    # ── Phase 3: Resolve Direction (now with target awareness) ──
    # Compute nearest_liq_direction from unswept magnets
    nearest_liq_dir = None
    if magnets:
        above = [(p, s) for p, v, s in magnets if p > float(row['Close'])]
        below = [(p, s) for p, v, s in magnets if p < float(row['Close'])]
        if above and below:
            nearest_above_dist = min(above, key=lambda x: x[0] - float(row['Close']))
            nearest_below_dist = min(below, key=lambda x: float(row['Close']) - x[0])
            above_dist = nearest_above_dist[0] - float(row['Close'])
            below_dist = float(row['Close']) - nearest_below_dist[0]
            if above_dist < below_dist * 0.7:
                nearest_liq_dir = 'LONG'  # closer liquidity above → go long to grab it
            elif below_dist < above_dist * 0.7:
                nearest_liq_dir = 'SHORT'  # closer liquidity below → go short to grab it

    direction, dir_size_mult, dir_details = resolve_direction(
        vol_regime, m9_raw if m9_raw else 0.5,
        m13_bias, m13_score_raw, m13_details,
        m7_score=m7_score, m7_status=m7_status,
        swing_bias_1d=swing_bias, trend_dir=trend_dir, config=cfg,
        long_target_score=long_tgt_score, short_target_score=short_tgt_score,
        long_target_details=long_tgt_details, short_target_details=short_tgt_details,
        nearest_liq_direction=nearest_liq_dir,
    )
    # ── Gather market data (always, regardless of signal status) ──
    swept_magnets = _check_swept_magnets(df_15m, idx, magnets[:5])
    result['magnets'] = swept_magnets
    result['gaps'] = [round(p, 2) for p, _ in gaps[:5]]

    # Target scores from direction resolver
    result['target_scores'] = {
        'LONG': round(long_tgt_score, 3), 'SHORT': round(short_tgt_score, 3),
        'long_details': long_tgt_details, 'short_details': short_tgt_details,
    }

    sr_levels.sort(key=lambda x: x[1], reverse=True)
    result['sr_levels'] = [(round(p, 2), round(s, 2), t, touches, bounces)
                           for p, s, touches, bounces, t in sr_levels[:8]]

    try:
        deriv_summary = get_derivatives_summary()
        if 'error' not in deriv_summary:
            result['derivatives'] = deriv_summary
    except Exception:
        pass

    result['cascade_risk'] = _detect_cascade_risk(df_15m, idx, result)

    # ── M18 Squeeze Detection (after derivatives data is available) ──
    result['rsi'] = float(df_15m['rsi'].iloc[idx]) if 'rsi' in df_15m.columns else 50
    result['vol_trend'] = float(df_15m['Volume'].iloc[idx] / df_15m['vol_ma20'].iloc[idx]) if 'vol_ma20' in df_15m.columns else 1.0
    result['atr'] = float(df_15m['atr'].iloc[idx]) if 'atr' in df_15m.columns else 0

    # Squeeze quality: compute from raw market features
    roll_high_48 = float(df_15m['High'].iloc[max(0,idx-47):idx+1].max())
    roll_low_48 = float(df_15m['Low'].iloc[max(0,idx-47):idx+1].min())
    range_width = (roll_high_48 - roll_low_48) / float(df_15m['Close'].iloc[idx]) * 100

    vol_ratio_val = float(df_15m['vol_ratio'].iloc[idx]) if 'vol_ratio' in df_15m.columns and not pd.isna(df_15m['vol_ratio'].iloc[idx]) else 0.15
    taker_base = float(df_15m['Taker buy base asset volume'].iloc[idx]) if 'Taker buy base asset volume' in df_15m.columns else 0
    total_vol = float(df_15m['Volume'].iloc[idx])
    taker_ratio = taker_base / total_vol if total_vol > 0 else 0.5

    # VWAP distance
    vwap_val = float(df_15m['vwap'].iloc[idx]) if 'vwap' in df_15m.columns else float(df_15m['Close'].iloc[idx])
    vwap_dist = (float(df_15m['Close'].iloc[idx]) - vwap_val) / vwap_val * 100 if vwap_val > 0 else 0

    # OI proxy: volume accumulation relative to average
    vol_cumsum_48 = float(df_15m['Volume'].iloc[max(0,idx-47):idx+1].sum())
    vol_cumsum_ma = float(df_15m['Volume'].iloc[max(0,idx-67):idx+1].rolling(20).mean().iloc[-1]) if idx > 67 else vol_cumsum_48
    oi_proxy = vol_cumsum_48 / vol_cumsum_ma if vol_cumsum_ma > 0 else 1.0

    # Bar-level ignition
    bar_vol_spike = float(df_15m['Volume'].iloc[idx] / df_15m['vol_ma20'].iloc[idx]) if 'vol_ma20' in df_15m.columns else 1.0
    bar_range = (float(df_15m['High'].iloc[idx]) - float(df_15m['Low'].iloc[idx])) / float(df_15m['Close'].iloc[idx]) * 100
    bar_range_ma = float(df_15m['High'].iloc[max(0,idx-19):idx+1].sub(df_15m['Low'].iloc[max(0,idx-19):idx+1]).div(df_15m['Close'].iloc[max(0,idx-19):idx+1]).mean() * 100) if idx > 19 else bar_range
    bar_range_expansion = bar_range / bar_range_ma if bar_range_ma > 0 else 1.0
    bar_taker_extreme = taker_ratio > 0.65 or taker_ratio < 0.35

    # Quality percentile (use rolling rank)
    # For now, use a simplified quality score
    # Lower range_width, lower vol_ratio, higher oi_proxy, closer to VWAP = better
    result['range_width'] = range_width
    result['vol_ratio'] = vol_ratio_val
    result['oi_proxy'] = oi_proxy
    result['vwap_dist'] = vwap_dist
    result['bar_vol_spike'] = bar_vol_spike
    result['bar_range_expansion'] = bar_range_expansion
    result['bar_taker_extreme'] = bar_taker_extreme

    # Simplified quality: normalize each factor to 0-1 and combine
    # Using rough thresholds from backtest analysis
    rw_score = max(0, min(1, 1 - (range_width - 1.5) / 4.0))  # 1.5% = best, 5.5% = worst
    vr_score = max(0, min(1, 1 - (vol_ratio_val - 0.05) / 0.20))  # 0.05 = best, 0.25 = worst
    oip_score = max(0, min(1, (oi_proxy - 0.7) / 0.5))  # 0.7 = worst, 1.2 = best
    vd_score = max(0, min(1, 1 - abs(vwap_dist) / 1.0))  # 0 = best, 1.0% = worst

    result['squeeze_quality'] = (rw_score * 0.30 + vr_score * 0.25 +
                                  oip_score * 0.25 + vd_score * 0.20)

    # Build compression history for squeeze detector (last 48 bars)
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

    # Pass raw taker_ratio and bar_range to result for squeeze detector
    result['raw_taker_ratio'] = taker_ratio
    result['raw_bar_range_pct'] = bar_range

    squeeze_result = detect_squeeze(result, config=cfg,
                                     last_signal_bar=result.get('_last_squeeze_bar', -1),
                                     current_bar=idx,
                                     compression_history=compression_history,
                                     df_15m=df_15m,
                                     magnets=magnets,
                                     sr_levels=sr_levels)
    result['squeeze'] = squeeze_result
    if squeeze_result['squeeze_status'] == 'TRIGGERED':
        result['_last_squeeze_bar'] = idx

    # ── Squeeze 4-filter confirmation gate (backtested: 84.6% WR on 4h) ──
    squeeze_confirmed = False
    squeeze_filters = {}
    if squeeze_result['squeeze_type'] != 'NONE' and squeeze_result['direction'] != 'NEUTRAL':
        sq_dir = squeeze_result['direction']

        # Filter 1: EMA trend alignment
        if cfg.get('SQUEEZE_CONFIRM_EMA', True) and len(df_15m) >= 55:
            _close = df_15m['Close']
            _ema21 = float(_close.ewm(span=21, adjust=False).mean().iloc[-1])
            _ema55 = float(_close.ewm(span=55, adjust=False).mean().iloc[-1])
            _ema_trend = 'BULL' if _ema21 > _ema55 else 'BEAR'
            squeeze_filters['ema_aligned'] = (sq_dir == 'LONG' and _ema_trend == 'BULL') or \
                                              (sq_dir == 'SHORT' and _ema_trend == 'BEAR')
        else:
            squeeze_filters['ema_aligned'] = True

        # Filter 2: CVD divergence agrees
        if cfg.get('SQUEEZE_CONFIRM_CVD', True):
            _m4b_div = result.get('m4b', {}).get('divergence', 'NONE')
            squeeze_filters['cvd_agrees'] = not ((sq_dir == 'LONG' and _m4b_div == 'BEARISH') or
                                                 (sq_dir == 'SHORT' and _m4b_div == 'BULLISH'))
        else:
            squeeze_filters['cvd_agrees'] = True

        # Filter 3: RSI not extreme against direction
        if cfg.get('SQUEEZE_CONFIRM_RSI', True) and 'rsi' in df_15m.columns:
            _rsi = float(df_15m['rsi'].iloc[-1]) if not pd.isna(df_15m['rsi'].iloc[-1]) else 50
            squeeze_filters['rsi_ok'] = (sq_dir == 'LONG' and _rsi < 75) or \
                                         (sq_dir == 'SHORT' and _rsi > 25)
        else:
            squeeze_filters['rsi_ok'] = True

        # Filter 4: Quality score >= 0.5
        if cfg.get('SQUEEZE_CONFIRM_QUALITY', True):
            squeeze_filters['quality_high'] = squeeze_result.get('squeeze_score', 0) >= 0.5
        else:
            squeeze_filters['quality_high'] = True

        # Filter 5: ATR floor — skip signals when vol is too low for a real move
        if cfg.get('SQUEEZE_CONFIRM_ATR_FLOOR', True):
            _atr_now = float(df_15m['atr'].iloc[-1]) if 'atr' in df_15m.columns and not pd.isna(df_15m['atr'].iloc[-1]) else 0
            _atr_hard_floor = cfg.get('SQUEEZE_MIN_ATR', 5.0)
            # Rolling percentile floor
            _atr_lookback = cfg.get('SQUEEZE_ATR_LOOKBACK', 8640)
            _atr_pctile = cfg.get('SQUEEZE_ATR_FLOOR_PCTILE', 15)
            _atr_series = df_15m['atr'].dropna()
            if len(_atr_series) > 100:
                _atr_window = _atr_series.iloc[-min(len(_atr_series), _atr_lookback):]
                _atr_threshold = float(np.percentile(_atr_window, _atr_pctile))
            else:
                _atr_threshold = _atr_hard_floor
            # Use the higher of hard floor and percentile floor
            _atr_effective = max(_atr_hard_floor, _atr_threshold)
            squeeze_filters['atr_floor'] = _atr_now >= _atr_effective
            squeeze_filters['_atr_value'] = round(_atr_now, 2)
            squeeze_filters['_atr_threshold'] = round(_atr_effective, 2)
        else:
            squeeze_filters['atr_floor'] = True

        squeeze_confirmed = all(squeeze_filters.values())

    result['squeeze_filters'] = squeeze_filters
    result['squeeze_confirmed'] = squeeze_confirmed

    # Only apply regime override + ICS boost if squeeze is CONFIRMED
    if squeeze_result['squeeze_status'] == 'TRIGGERED' and squeeze_confirmed and squeeze_result.get('overrides_regime'):
        regime_blocked = False
        result['regime_blocked'] = False
        result['squeeze_override'] = True

    # Exchange Activity (cross-exchange funding, OI, L/S)
    try:
        exchange_summary = get_exchange_summary()
        if 'error' not in exchange_summary:
            result['exchange_activity'] = exchange_summary
            # Score derivatives exchange data with resolved direction
            _ex_signals = exchange_summary.get('signals', {})
            if _ex_signals:
                _ex_status, _ex_score, _ex_details = score_exchange_activity(
                    _ex_signals, direction if direction != 'NEUTRAL' else 'LONG')
                result['exchange_activity']['score'] = round(_ex_score, 3)
                result['exchange_activity']['status'] = _ex_status
                result['exchange_activity']['direction_details'] = _ex_details
            # Score spot data
            _spot_signals = exchange_summary.get('spot_signals', {})
            if _spot_signals:
                _sp_status, _sp_score, _sp_details = score_spot_signals(
                    _spot_signals, direction if direction != 'NEUTRAL' else 'LONG')
                result['exchange_activity']['spot_score'] = round(_sp_score, 3)
                result['exchange_activity']['spot_status'] = _sp_status
                result['exchange_activity']['spot_details'] = _sp_details
    except Exception:
        pass

    # When regime blocks direction, use squeeze direction if available, else force LONG for display
    if direction == 'NEUTRAL' and result.get('regime_blocked'):
        sq_dir = squeeze_result.get('direction', 'NEUTRAL')
        if sq_dir != 'NEUTRAL':
            direction = sq_dir
            dir_details['reason'] = f"Squeeze {squeeze_result['squeeze_type']} override → {direction}"
        else:
            direction = 'LONG'
            dir_details['reason'] = f"regime={vol_regime} blocked — scoring LONG for display"

    result['direction'] = direction
    result['direction_resolver'] = {
        'direction': direction, 'size_mult': round(float(dir_size_mult), 3),
        'action': dir_details.get('action', '?'),
        'reason': dir_details.get('reason', '?'),
    }

    # Liquidity levels & conflict — need direction
    _liq_direction = direction if direction != 'NEUTRAL' else None
    if _liq_direction:
        try:
            oi_usd = result.get('derivatives', {}).get('oi_usd', 0)
            ls_ratio = result.get('derivatives', {}).get('ls_ratio', 1.0)
            liq_summary = get_liquidity_summary(
                df_15m, idx, sr_levels, oi_usd, ls_ratio, _liq_direction)
            result['liquidity_levels'] = liq_summary
        except Exception:
            pass

        if swing_bias:
            try:
                conflict = get_conflict_stats(
                    'BULLISH' if _liq_direction == 'LONG' else 'BEARISH',
                    swing_bias)
                if conflict:
                    result['conflict'] = conflict
            except Exception:
                pass

    # Re-score M9, M7, M13 with actual direction
    m9_status, m9_score, m9_details = score_vol_regime(vol_regime, m9_raw, direction, trend_dir)
    if cfg.get('M7_ENABLED', False) and m7_ethbtc_df is not None:
        eb_row, bt_row = m7_get_row(m7_ethbtc_df, m7_btc_df, ts)
        m7_status, m7_score, m7_details = score_m7(eb_row, bt_row, row.get('vol_ratio', np.nan), direction)
    m13_status, m13_score, m13_details = score_m13(df_1h, idx_1h, direction, df_15m_for_m9, idx_15m_for_m9)
    result['m9']['score'] = round(float(m9_score), 3)
    result['m9']['status'] = m9_status
    result['m7']['score'] = round(float(m7_score), 3)
    result['m13']['score'] = round(float(m13_score), 3)

    # ── Phase 4: Score all modules ──
    # M1 (now an ICS contributor, not the gate)
    m1_dir, m1_score, _m1_details = score_m1(df_1h, idx_1h, cfg, df_15m=df_15m, idx_15m=idx)
    # Direction-aware scoring: flip M1 score when direction disagrees with trade
    # M1 returns score 0.5-1.0 (distance from neutral), but doesn't encode trade direction.
    # If M1 says BEARISH and trade is LONG, high score should penalize, not boost.
    if m1_dir == 'BEARISH' and direction == 'LONG':
        m1_score = 1.0 - m1_score
    elif m1_dir == 'BULLISH' and direction == 'SHORT':
        m1_score = 1.0 - m1_score
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

    # M4b: Intrabar CVD (LucF-style delta volume analysis)
    m4b_status = 'SKIP'
    m4b_score = 0.5
    m4b_details = {}
    m4b_divergence = 'NONE'
    if cfg.get('M4B_INTRABAR_ENABLED', True):
        try:
            _tf_map = {'15m': '15min', '1h': '1h', '5m': '5min', '1m': '1min'}
            _target_tf = _tf_map.get(cfg.get('_base_timeframe', '15m'), '15min')
            _hours = cfg.get('M4B_INTRABAR_HOURS', 48)
            _intrabar_df, _intrabar_result = get_intrabar_cvd_summary(
                symbol='ETHUSDT', target_tf=_target_tf, hours=_hours)
            if _intrabar_result and 'error' not in _intrabar_result:
                m4b_status, m4b_score, m4b_details = score_intrabar_divergence(
                    _intrabar_result, direction)
                m4b_divergence = _intrabar_result.get('divergence', 'NONE')
                result['m4b'] = {
                    'status': m4b_status,
                    'score': round(float(m4b_score), 3),
                    'divergence': m4b_divergence,
                    'bars_ago': _intrabar_result.get('bars_ago', -1),
                    'cvd_slope': round(_intrabar_result.get('cvd_slope_12', 0), 2),
                    'details': m4b_details,
                }
                # Blend m4b into m4: if intrabar catches div that taker CVD missed
                if m4b_divergence != 'NONE':
                    m4_div_detected = m4_div.get('layer_a_div', 'NONE') if isinstance(m4_div, dict) else 'NONE'
                    if m4_div_detected == 'NONE':
                        # M4 missed it, M4b caught it — apply M4b's signal
                        m4_score = m4b_score
                        if isinstance(m4_div, dict):
                            m4_div['intrabar_div'] = m4b_divergence
                            m4_div['intrabar_source'] = 'lucf_style'
                        print(f"  📊 Intrabar CVD override: {m4b_divergence} div ({m4b_details.get('bars_ago', '?')} bars ago)")
        except Exception as e:
            result['m4b'] = {'status': 'ERROR', 'score': 0.5, 'error': str(e)}

    # M5
    m5_status, m5_score, m5_details = score_m5(df_15m, idx, direction, cfg,
        n_bins=cfg['M5_VP_BINS'], lookback=cfg['M5_VP_LOOKBACK'])
    result['m5'] = {'status': m5_status, 'score': round(float(m5_score), 3)}

    # M5 Regime Gate — neutralize M5 in unfavorable regimes (forensic P1)
    if cfg.get('M5_REGIME_GATE_ENABLED', False):
        _m5_favorable = ('NEUTRAL', 'TRENDING', 'CHOP_MILD_BEAR')
        if vol_regime not in _m5_favorable:
            m5_score = 0.5

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

    # M10 (cross-asset macro) — BTC trend + ETH/BTC relative strength
    m10_score = 0.5
    m10_status = 'SKIP'
    m10_details = {}
    m10_data = None
    if cfg.get('M10_ENABLED', False):
        try:
            m10_data = m10_prepare_data(df_15m)
            if m10_data:
                m10_data = m10_compute_emas(m10_data)
                macro_row = m10_get_row(m10_data, ts)
                if macro_row:
                    m10_status, m10_score, m10_details = score_m10_macro(macro_row, direction, trend_dir)
                    result['m10'] = {'status': m10_status, 'score': round(float(m10_score), 3), 'details': m10_details}
        except Exception as e:
            result['m10'] = {'status': 'ERROR', 'score': 0.5, 'error': str(e)}

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

    # M12 (order book imbalance) — live only
    m12_score = 0.5
    m12_status = 'SKIP'
    if cfg.get('M12_ENABLED', False) and cfg.get('M12_LIVE_ONLY', True):
        try:
            m12_status, m12_score, m12_details = score_m12_orderbook(direction, live=True)
            result['m12'] = {'status': m12_status, 'score': round(float(m12_score), 3), 'details': m12_details}
        except Exception as e:
            result['m12'] = {'status': 'ERROR', 'score': 0.5, 'error': str(e)}

    # M14 (sweep-retest-reclaim)
    m14_score = 0.5
    m14_status = 'SKIP'
    if cfg.get('M14_ENABLED', True):
        _swing_levels = m13_details.get('swing_lows', []) if direction == 'LONG' else m13_details.get('swing_highs', [])
        if _swing_levels:
            m14_status, m14_score, _ = score_m14(df_15m, idx, direction, _swing_levels, config=cfg, magnets=magnets)
            result['m14'] = {'status': m14_status, 'score': round(float(m14_score), 3)}

    # M17 (resistance quality) — validate nearest S/R level
    m17_score = 0.5
    m17_status = 'SKIP'
    m17_result = None
    if cfg.get('M17_ENABLED', True) and sr_levels:
        if direction == 'LONG':
            resistances = [sr for sr in sr_levels if sr[4] == 'RESISTANCE']
            if resistances:
                nearest_res = min(resistances, key=lambda x: abs(x[0] - current_price))
                m17_result = score_resistance_quality(
                    nearest_res[0], df_15m, idx, bin_centers, vol_profile,
                    result.get('derivatives', {}), 'LONG', config=cfg)
        elif direction == 'SHORT':
            supports = [sr for sr in sr_levels if sr[4] == 'SUPPORT']
            if supports:
                nearest_sup = min(supports, key=lambda x: abs(x[0] - current_price))
                m17_result = score_resistance_quality(
                    nearest_sup[0], df_15m, idx, bin_centers, vol_profile,
                    result.get('derivatives', {}), 'SHORT', config=cfg)
        if m17_result:
            m17_score = m17_result['composite']
            m17_status = 'PASS'
            result['m17'] = {**m17_result, 'status': m17_status, 'score': round(float(m17_score), 4)}

    # ── ICS ──
    ics, effective_floor = calc_ics(
        m1_score, m2_score, m3_score, m4_score, m4_status, m5_score,
        m7_score=m7_score, m8_score=m8_score,
        use_m7=cfg.get('M7_ENABLED', False) and m7_ethbtc_df is not None,
        use_m8=m8_status != 'SKIP',
        m9_score=m9_score, use_m9=True,
        m10_score=m10_score, use_m10=m10_status != 'SKIP',
        m11_score=m11_score, use_m11=m11_status != 'SKIP',
        m12_score=m12_score, use_m12=m12_status != 'SKIP',
        m13_score=m13_score, use_m13=cfg.get('M13_ENABLED', False),
        m14_score=m14_score, use_m14=m14_status == 'PASS',
        m17_score=m17_score, use_m17=m17_status == 'PASS',
        config=cfg,
    )
    result['ics'] = round(float(ics), 4)
    result['effective_floor'] = round(float(effective_floor), 4)

    # ── Squeeze ICS boost (only when TRIGGERED + CONFIRMED) ──
    if squeeze_result['squeeze_status'] == 'TRIGGERED' and squeeze_confirmed and squeeze_result['ics_boost'] > 0:
        ics += squeeze_result['ics_boost']
        result['ics'] = round(float(ics), 4)
        result['squeeze_ics_boost'] = squeeze_result['ics_boost']

    # ── Phase 5: Veto + Coherence + Filters ──
    # Veto
    if cfg.get('VETO_ENABLED', False):
        # Extract actual divergence string from M4 details dict
        m4_div_str = 'NONE'
        if isinstance(m4_div, dict):
            m4_div_str = m4_div.get('layer_a_div', 'NONE')
        elif isinstance(m4_div, str):
            m4_div_str = m4_div
        # Also consider intrabar divergence
        if m4_div_str == 'NONE' and m4b_divergence != 'NONE':
            m4_div_str = m4b_divergence

        m4_disagree = (direction == 'LONG' and m4_div_str == 'BEARISH') or \
                      (direction == 'SHORT' and m4_div_str == 'BULLISH')
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
            direction, m4_div_str, m5_details if isinstance(m5_details, dict) else {},
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

    # ── Compute trade levels (always, even on NO_SIGNAL) ──
    entry_price = float(row['Close'])
    atr_for_sl = float(atr_1h) if not pd.isna(atr_1h) else float(row['atr'])
    _liq_for_levels = result.get('liquidity_levels')

    levels = calc_trade_levels(
        entry_price, direction, atr_for_sl,
        row.get('vol_ratio', np.nan),
        magnets=magnets,
        sr_levels=sr_levels,
        liq_levels=_liq_for_levels,
        cfg=cfg,
    )

    # ── Build invalidation conditions ──
    invalidation = []

    # Bearish divergence active
    m4_div_str = 'NONE'
    if isinstance(m4_div, dict):
        m4_div_str = m4_div.get('layer_a_div', 'NONE')
    if m4_div_str == 'NONE' and m4b_divergence != 'NONE':
        m4_div_str = m4b_divergence
    if (direction == 'LONG' and m4_div_str == 'BEARISH') or \
       (direction == 'SHORT' and m4_div_str == 'BULLISH'):
        invalidation.append(f'{m4_div_str} CVD divergence active — momentum against you')

    # Whales leaning against
    whale = result.get('derivatives', {}).get('whale_signal', 'NEUTRAL')
    if (direction == 'LONG' and whale == 'WHALE_BEARISH') or \
       (direction == 'SHORT' and whale == 'WHALE_BULLISH'):
        invalidation.append(f'Whales {whale.replace("WHALE_", "").lower()} — smart money against direction')

    # Crowded positioning
    pos = result.get('derivatives', {}).get('positioning', 'NEUTRAL')
    if (direction == 'LONG' and pos == 'CROWDED_LONG') or \
       (direction == 'SHORT' and pos == 'CROWDED_SHORT'):
        invalidation.append(f'Crowded {direction.lower()} positioning — squeeze risk')

    # Conflict history
    conflict = result.get('conflict')
    if conflict and conflict.get('is_conflict'):
        invalidation.append(f'Historical conflict: similar setups reverse {conflict["historical"]["windows"].get("24h", {}).get("reversal_rate", 0):.0f}% at 24h')

    # Phase0 death zone
    phase0_min = cfg.get('PHASE0_MIN_BLOCK', 0.10)
    if phase0_val is not None and phase0_val < phase0_min:
        invalidation.append(f'Phase0={phase0_val:.3f} (death zone <{phase0_min}) — weak macro context')

    # M2 EMA failure
    if m2_status == 'FAIL':
        invalidation.append('M2 EMA confluence FAIL — multi-TF trend disagreement')

    # Price below key S/R
    supports = [p for p, s, t, _, _ in sr_levels if t == 'SUPPORT']
    resistances = [p for p, s, t, _, _ in sr_levels if t == 'RESISTANCE']
    nearest_support = min(supports, key=lambda x: abs(x - entry_price)) if supports else 0
    nearest_resist = min(resistances, key=lambda x: abs(x - entry_price)) if resistances else 0

    result['what_if'] = {
        'direction': direction,
        'entry': entry_price,
        'sl': levels['sl'],
        'tp1': levels['tp1'],
        'tp2': levels['tp2'],
        'tp3': levels['tp3'],
        'sl_pct': levels['sl_pct'],
        'tp1_pct': levels['tp1_pct'],
        'sl_source': levels['sl_source'],
        'tp1_source': levels['tp1_source'],
        'tp2_source': levels['tp2_source'],
        'tp3_source': levels['tp3_source'],
        'rr1': abs(levels['tp1_pct'] / levels['sl_pct']) if levels['sl_pct'] != 0 else 0,
        'nearest_support': round(nearest_support, 2),
        'nearest_resist': round(nearest_resist, 2),
        'invalidation': invalidation,
    }

    # ── Entry filters ──
    passed, reason = check_entry_filters(df_15m, idx, direction, swing_bias, phase0_val, atr_1h, config=cfg)
    if not passed:
        result['status'] = 'FILTERED'
        result['reason'] = reason
        return result

    # ── M14 Sweep Gate ──
    sweep_passed, sweep_reason = check_sweep_gate(m14_status, m14_score, cfg)
    if not sweep_passed:
        result['status'] = 'NO_SIGNAL'
        result['reason'] = sweep_reason
        return result

    # ── Regime block override (all modules scored, but regime kills the signal) ──
    if result.get('regime_blocked'):
        result['status'] = 'NO_SIGNAL'
        result['reason'] = f'M9 regime={vol_regime} (blocked)'
        return result

    # ── SIGNAL: set levels ──
    result.update({
        'status': 'SIGNAL', 'entry': entry_price,
        'sl': levels['sl'], 'tp1': levels['tp1'],
        'tp2': levels['tp2'], 'tp3': levels['tp3'],
        'sl_pct': levels['sl_pct'], 'tp1_pct': levels['tp1_pct'],
        'sl_source': levels['sl_source'],
        'tp1_source': levels['tp1_source'],
        'tp2_source': levels['tp2_source'],
        'tp3_source': levels['tp3_source'],
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

    # Direction Resolver
    dr = result.get('direction_resolver', {})
    print(f"\n  Direction Resolver:")
    print(f"    Regime:        {result.get('m9', {}).get('regime', '?')}  (score={result.get('m9', {}).get('score', 0):.3f})")
    print(f"    Structure:     {result.get('m13', {}).get('bias', '?')}  (score={result.get('m13', {}).get('score', 0):.3f})")
    if 'm7' in result and result['m7'].get('status') != 'SKIP':
        print(f"    Macro M7:      {result['m7']['score']:.3f}  ({result['m7']['status']})")
    # Target scores
    tgt = result.get('target_scores', {})
    if tgt:
        print(f"    Targets:       LONG={tgt.get('LONG', 0):.3f}  SHORT={tgt.get('SHORT', 0):.3f}")
        # Show top target for each direction
        for d in ('LONG', 'SHORT'):
            det = tgt.get(f'{d.lower()}_details', {})
            top = sorted(det.get('targets', []), key=lambda x: -x.get('contrib', 0))[:1]
            top_sr = sorted(det.get('sr', []), key=lambda x: -x.get('contrib', 0))[:1]
            parts = []
            if top:
                parts.append(f"HVN ${top[0]['price']:.0f} ({top[0]['dist_pct']:+.1f}%)")
            if top_sr:
                parts.append(f"S/R ${top_sr[0]['price']:.0f} ({top_sr[0]['dist_pct']:+.1f}%)")
            if parts:
                print(f"      {d:>6} best: {', '.join(parts)}")
    print(f"    → Direction:   {dr.get('direction', '?')}  size_mult={dr.get('size_mult', 0):.2f}")
    if dr.get('target_tiebreaker'):
        print(f"    🎯 Tiebreaker: {dr.get('target_tiebreaker')}")
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
        det = m4.get('div', {}) or m4.get('details', {})
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
    if 'm4b' in result:
        m4b = result['m4b']
        m4b_div = m4b.get('divergence', 'NONE')
        m4b_ago = m4b.get('bars_ago', -1)
        m4b_slope = m4b.get('cvd_slope', 0)
        m4b_icon = {'BEARISH': '🔻', 'BULLISH': '🔺', 'NONE': '—'}.get(m4b_div, '—')
        ago_str = f"{m4b_ago}bars ago" if m4b_ago >= 0 else ""
        print(f"    M4b(IntraCVD): {m4b['status']:>8}  score={m4b['score']:.2f}  "
              f"div={m4b_div} {m4b_icon}  slope={m4b_slope:.1f}  {ago_str}")
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
    if 'm10' in result:
        m10 = result['m10']
        m10_comp = m10.get('details', {}).get('m10_components', {})
        m10_agree = m10.get('details', {}).get('macro_agreement', '')
        print(f"    M10 (Macro):   {m10['status']:>8}  score={m10['score']:.2f}  {m10_agree}")
    if 'm11' in result:
        print(f"    M11 (MTF Mom): {result['m11']['status']:>8}  score={result['m11']['score']:.2f}")
    if 'm13' in result:
        m13 = result['m13']
        print(f"    M13 (Struct):  {m13['status']:>8}  score={m13['score']:.2f}  bias={m13['bias']}")
    if 'm12' in result:
        m12 = result['m12']
        m12_ob = m12.get('details', {}).get('bid_ask_ratio', '')
        ob_str = f"  OB={m12_ob:.2f}" if m12_ob else ""
        print(f"    M12 (OrderBook): {m12['status']:>8}  score={m12['score']:.2f}{ob_str}")
    if 'm14' in result:
        print(f"    M14 (Sweep):   {result['m14']['status']:>8}  score={result['m14']['score']:.2f}")
    if 'm17' in result:
        m17 = result['m17']
        m17_zv = m17.get('zone_volume', {}).get('zone_vol_ratio', '?')
        m17_rej = m17.get('rejection', {}).get('status', '?')
        m17_dfn = m17.get('defender', {}).get('status', '?')
        print(f"    M17 (ResQual): {m17['status']:>8}  score={m17['score']:.3f}  "
              f"zone={m17_zv}x  reject={m17_rej}  defender={m17_dfn}  → {m17['verdict']}")
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

    # M17 Resistance Quality
    if 'm17' in result and result['m17'].get('status') != 'SKIP':
        print(format_resistance_quality(result['m17']))

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

    # Exchange Activity (cross-exchange)
    exch = result.get('exchange_activity', {})
    if exch and 'error' not in exch:
        print(f"\n  Exchange Activity (cross-exchange):")
        snaps = exch.get('snapshots', {})
        sigs = exch.get('signals', {})

        # Per-exchange snapshot table
        print(f"    {'Exchange':<10} {'Funding':>12} {'OI (ETH)':>14} {'L/S Ratio':>10}")
        for ex_name in ['binance', 'okx', 'bybit', 'htx', 'phemex', 'kraken']:
            s = snaps.get(ex_name, {})
            if not s or s.get('error'):
                continue
            fr = s.get('funding_rate')
            oi = s.get('oi')
            ls = s.get('ls_ratio')
            fr_str = f"{fr*100:+.4f}%" if fr is not None else "N/A"
            oi_str = f"{oi:,.0f}" if oi else "N/A"
            ls_str = f"{ls:.4f}" if ls else "N/A"
            print(f"    {ex_name:<10} {fr_str:>12} {oi_str:>14} {ls_str:>10}")

        # Funding spread
        spread = sigs.get('funding_spread', 0)
        spread_trend = sigs.get('funding_spread_trend', '?')
        print(f"    Funding spread: {spread*100:.4f}%  trend={spread_trend}")

        # OI shares
        oi_shares = sigs.get('oi_shares', {})
        if oi_shares:
            shares_str = '  '.join(f"{k}={v*100:.1f}%" for k, v in sorted(oi_shares.items()))
            print(f"    OI share:  {shares_str}")
            print(f"    OI dominant: {sigs.get('oi_dominant_exchange', '?')}  "
                  f"concentration={sigs.get('oi_concentration', 0):.3f}")

        # OI migration
        migration = sigs.get('oi_migration')
        if migration and migration != 'BALANCED':
            print(f"    ⚡ OI Migration: {migration}")

        # L/S divergence
        ls_spread = sigs.get('ls_spread', 0)
        if ls_spread > 0.3:
            ls_by = sigs.get('ls_by_exchange', {})
            ls_str = '  '.join(f"{k}={v:.2f}" for k, v in sorted(ls_by.items()))
            print(f"    L/S divergence: {ls_spread:.3f}  ({ls_str})")

        # Scoring
        ex_score = exch.get('score', 0)
        ex_status = exch.get('status', '?')
        ex_details = exch.get('direction_details', {})
        ex_factors = ex_details.get('factors', [])
        print(f"    Score: {ex_score:.3f} ({ex_status})")
        for f in ex_factors:
            print(f"      • {f}")

        # Spot data
        spot = exch.get('spot', {})
        spot_sigs = exch.get('spot_signals', {})
        spot_details = exch.get('spot_details', {})
        if spot:
            print(f"\n    Spot Markets:")
            print(f"      {'Exchange':<10} {'Price':>10} {'24h Vol':>12} {'OB Ratio':>10} {'Flow':>8}")
            for ex_name in ['binance', 'okx', 'bybit', 'kraken', 'coinbase', 'htx']:
                s = spot.get(ex_name, {})
                if not s or not s.get('price'):
                    continue
                p = s.get('price', 0)
                v = s.get('vol_24h') or 0
                ob = s.get('ob_ratio') or 0
                buy_pct = s.get('buy_pct') or 50
                flow = f"{buy_pct:.0f}% buy"
                print(f"      {ex_name:<10} ${p:>9.2f} {v:>11,.0f} {ob:>10.3f} {flow:>8}")

            # Basis
            basis = spot_sigs.get('basis', {})
            if basis:
                basis_str = '  '.join(f"{k}={v:+.3f}%" for k, v in sorted(basis.items()))
                print(f"      Basis: {basis_str}")
                print(f"      Avg basis: {spot_sigs.get('basis_avg', 0):+.4f}% ({spot_sigs.get('basis_state', '?')})")

            # Spot walls
            sell_walls = spot_sigs.get('spot_sell_walls', [])
            bid_support = spot_sigs.get('spot_bid_support', [])
            if sell_walls or bid_support:
                parts = []
                if sell_walls:
                    parts.append(f"sell walls: {', '.join(sell_walls)}")
                if bid_support:
                    parts.append(f"bid support: {', '.join(bid_support)}")
                print(f"      Book: {' | '.join(parts)}")

            # Spot scoring
            sp_score = exch.get('spot_score', 0)
            sp_status = exch.get('spot_status', '?')
            sp_factors = spot_details.get('factors', [])
            print(f"      Spot score: {sp_score:.3f} ({sp_status})")
            for f in sp_factors:
                print(f"        • {f}")

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

    # Squeeze Detector
    sq = result.get('squeeze', {})
    if sq and sq.get('squeeze_type', 'NONE') != 'NONE':
        sq_output = format_squeeze(sq)
        if sq_output:
            print(sq_output)
        # Show 4-filter confirmation status
        sq_filters = result.get('squeeze_filters', {})
        sq_confirmed = result.get('squeeze_confirmed', False)
        if sq_filters:
            icons = {True: '✅', False: '❌'}
            print(f"\n  Squeeze Confirmation Gate:")
            print(f"    EMA aligned:  {icons.get(sq_filters.get('ema_aligned'), '?')}")
            print(f"    CVD agrees:   {icons.get(sq_filters.get('cvd_agrees'), '?')}")
            print(f"    RSI ok:       {icons.get(sq_filters.get('rsi_ok'), '?')}")
            print(f"    Quality ≥0.5: {icons.get(sq_filters.get('quality_high'), '?')}")
            _atr_val = sq_filters.get('_atr_value', '?')
            _atr_thr = sq_filters.get('_atr_threshold', '?')
            print(f"    ATR floor:    {icons.get(sq_filters.get('atr_floor'), '?')}  (ATR=${_atr_val} vs floor=${_atr_thr})")
            if sq_confirmed:
                print(f"    → ✅ CONFIRMED (backtested 84.6% WR on 4h)")
            else:
                print(f"    → ❌ NOT CONFIRMED — regime override & ICS boost skipped")
        if result.get('squeeze_override'):
            print(f"\n  ⚡ SQUEEZE OVERRIDE: regime block lifted!")

    # ICS & Signal
    if 'ics' in result:
        boost_str = f"  (squeeze +{result.get('squeeze_ics_boost', 0):.4f})" if result.get('squeeze_ics_boost') else ''
        print(f"\n  ICS: {result['ics']:.3f}  (floor={result['effective_floor']:.3f}){boost_str}")

    status = result['status']
    if status == 'SIGNAL':
        print(f"\n  ✅ SIGNAL: {result['direction']}")
        sl_src = result.get('sl_source', 'ATR')
        tp1_src = result.get('tp1_source', 'ATR')
        tp2_src = result.get('tp2_source', 'ATR')
        tp3_src = result.get('tp3_source', 'ATR')
        print(f"    Entry: ${result['entry']:.2f}")
        print(f"    SL:    ${result['sl']:.2f}  ({result['sl_pct']:.2f}%)  [{sl_src}]")
        print(f"    TP1:   ${result['tp1']:.2f}  ({result['tp1_pct']:.2f}%)  [{tp1_src}]")
        print(f"    TP2:   ${result['tp2']:.2f}  [{tp2_src}]")
        print(f"    TP3:   ${result['tp3']:.2f}  [{tp3_src}]")
    else:
        print(f"\n  ⛔ {status}: {result.get('reason', 'N/A')}")

    # ── What-if trade levels (always show) ──
    w = result.get('what_if')
    if w:
        print(f"\n  {'─' * 56}")
        print(f"  IF YOU WERE TO TRADE ({w['direction']}):")
        print(f"    Entry: ${w['entry']:.2f}")
        print(f"    SL:    ${w['sl']:.2f}  ({w['sl_pct']:.2f}%)  [{w['sl_source']}]")
        print(f"    TP1:   ${w['tp1']:.2f}  ({w['tp1_pct']:.2f}%)  [{w['tp1_source']}]")
        print(f"    TP2:   ${w['tp2']:.2f}  [{w['tp2_source']}]")
        print(f"    TP3:   ${w['tp3']:.2f}  [{w['tp3_source']}]")
        print(f"    R:R:   {w['rr1']:.2f}x (entry→TP1 vs SL)")
        print(f"    Support:    ${w['nearest_support']:.2f}")
        print(f"    Resistance: ${w['nearest_resist']:.2f}")
        if w['invalidation']:
            print(f"    ⚠️  INVALIDATION:")
            for inv in w['invalidation']:
                print(f"      • {inv}")
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
    m4b_st = result.get('m4b', {}).get('status', '—') if 'm4b' in result else '—'
    m4b_sc = result.get('m4b', {}).get('score', 0) if 'm4b' in result else 0
    print(f"  {'M4b Intrabar CVD':<22} {m4b_st:>10}  {m4b_sc:>6.2f}")
    print(f"  {'M5 Liquidation':<22} {m5_st:>10}  {m5_sc:>6.2f}")
    print(f"  {'M8 Funding':<22} {m8_st:>10}  {m8_sc:>6.2f}")
    print(f"  {'M9 Vol Regime':<22} {m9_st:>10}  {m9_sc:>6.2f}")
    m10_st = result.get('m10', {}).get('status', '—') if 'm10' in result else '—'
    m10_sc = result.get('m10', {}).get('score', 0) if 'm10' in result else 0
    print(f"  {'M10 Cross-Asset':<22} {m10_st:>10}  {m10_sc:>6.2f}")
    print(f"  {'M11 MTF Momentum':<22} {m11_st:>10}  {m11_sc:>6.2f}")
    print(f"  {'M13 Structure':<22} {m13_st:>10}  {m13_sc:>6.2f}")
    print(f"  {'M14 Sweep':<22} {m14_st:>10}  {m14_sc:>6.2f}")
    m17_st = result.get('m17', {}).get('status', '—') if 'm17' in result else '—'
    m17_sc = result.get('m17', {}).get('score', 0) if 'm17' in result else 0
    m17_vd = result.get('m17', {}).get('verdict', '') if 'm17' in result else ''
    print(f"  {'M17 Resist Qual':<22} {m17_st:>10}  {m17_sc:>6.3f}  {m17_vd}")
    m12_st = result.get('m12', {}).get('status', '—') if 'm12' in result else '—'
    m12_sc = result.get('m12', {}).get('score', 0) if 'm12' in result else 0
    print(f"  {'M12 Order Book':<22} {m12_st:>10}  {m12_sc:>6.2f}")
    m16_exch = result.get('exchange_activity', {})
    m16_sc = m16_exch.get('score', 0) if m16_exch else 0
    m16_st = m16_exch.get('status', '—') if m16_exch else '—'
    print(f"  {'M16 Exch Derivs':<22} {m16_st:>10}  {m16_sc:>6.2f}")
    m16_sp_sc = m16_exch.get('spot_score', 0) if m16_exch else 0
    m16_sp_st = m16_exch.get('spot_status', '—') if m16_exch else '—'
    print(f"  {'M16 Exch Spot':<22} {m16_sp_st:>10}  {m16_sp_sc:>6.2f}")

    # Squeeze in summary
    sq = result.get('squeeze', {})
    if sq and sq.get('squeeze_type', 'NONE') != 'NONE':
        sq_status = sq.get('squeeze_status', 'NONE')
        sq_st = f"{'STRONG' if sq.get('squeeze_strong') else 'ACTIVE'} {sq_status}"
        sq_sc = sq.get('squeeze_score', 0)
        sq_dir = sq.get('direction', '?')
        print(f"  {'M18 Squeeze':<22} {sq_st:>10}  {sq_sc:>6.3f}  → {sq_dir}")
        if sq.get('entry_condition'):
            print(f"  {'  Entry':<22} {sq['entry_condition']}")
        sq_confirmed = result.get('squeeze_confirmed', False)
        sq_filters = result.get('squeeze_filters', {})
        if sq_filters:
            n_pass = sum(1 for k, v in sq_filters.items() if v is True)
            n_total = sum(1 for k in sq_filters if not k.startswith('_'))
            conf_label = f"✅ CONFIRMED ({n_pass}/{n_total})" if sq_confirmed else f"❌ ({n_pass}/{n_total})"
            print(f"  {'  Confirmation':<22} {conf_label}")
        if result.get('squeeze_override'):
            print(f"  {'⚡ Regime Override':<22} {'ACTIVE':>10}")

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

    # M10 macro summary
    m10 = result.get('m10', {})
    if m10 and m10.get('status') not in ('SKIP', 'ERROR', None):
        m10_comp = m10.get('details', {}).get('m10_components', {})
        m10_agree = m10.get('details', {}).get('macro_agreement', '')
        btc_roc = m10.get('details', {}).get('btc_roc7', '')
        ethbtc_roc = m10.get('details', {}).get('ethbtc_roc7', '')
        parts = []
        if btc_roc:
            parts.append(f"BTC 7d ROC={btc_roc:+.1%}")
        if ethbtc_roc:
            parts.append(f"ETH/BTC 7d ROC={ethbtc_roc:+.1%}")
        if m10_agree:
            parts.append(m10_agree)
        if parts:
            print(f"  • M10 macro: {m10['status']} ({', '.join(parts)})")

    # Exchange Activity summary
    exch = result.get('exchange_activity', {})
    if exch and 'error' not in exch:
        sigs = exch.get('signals', {})
        snaps = exch.get('snapshots', {})
        spread = sigs.get('funding_spread', 0)
        spread_trend = sigs.get('funding_spread_trend', '?')
        oi_shares = sigs.get('oi_shares', {})
        migration = sigs.get('oi_migration', 'BALANCED')
        ls_spread = sigs.get('ls_spread', 0)

        parts = []
        if spread > 0:
            parts.append(f"funding spread {spread*100:.4f}% ({spread_trend})")
        if oi_shares:
            dominant = sigs.get('oi_dominant_exchange', '?')
            parts.append(f"OI dominant: {dominant} ({oi_shares.get(dominant, 0)*100:.1f}%)")
        if migration and migration != 'BALANCED':
            parts.append(f"migration: {migration}")
        if ls_spread > 0.3:
            ls_by = sigs.get('ls_by_exchange', {})
            parts.append(f"L/S spread {ls_spread:.2f}")
        if parts:
            print(f"  • Exchange: {'; '.join(parts)}")

    # Spot market summary
    spot = exch.get('spot', {}) if exch else {}
    spot_sigs = exch.get('spot_signals', {}) if exch else {}
    if spot and spot_sigs:
        spot_parts = []
        basis_avg = spot_sigs.get('basis_avg', 0)
        basis_state = spot_sigs.get('basis_state', '?')
        if basis_avg:
            spot_parts.append(f"basis {basis_avg:+.3f}% ({basis_state})")
        ob_avg = spot_sigs.get('spot_ob_avg', 0)
        if ob_avg:
            spot_parts.append(f"OB ratio {ob_avg:.3f}")
        sell_walls = spot_sigs.get('spot_sell_walls', [])
        if sell_walls:
            spot_parts.append(f"sell walls: {', '.join(sell_walls)}")
        flow = spot_sigs.get('spot_flow', '?')
        if flow != '?':
            spot_parts.append(f"flow: {flow}")
        if spot_parts:
            print(f"  • Spot: {'; '.join(spot_parts)}")

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

        # What-if levels in summary
        w = result.get('what_if')
        if w:
            print(f"\n  If you trade anyway ({w['direction']}):")
            print(f"    Entry ${w['entry']:.2f} | SL ${w['sl']:.2f} ({w['sl_pct']:.2f}%) | "
                  f"TP1 ${w['tp1']:.2f} ({w['tp1_pct']:.2f}%) | R:R {w['rr1']:.2f}x")
            if w['invalidation']:
                for inv in w['invalidation']:
                    print(f"    ⚠️  {inv}")
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
    bars_map = {'1m': 3000, '5m': 2000, '15m': 1000, '1h': 500}
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
    scaled_config['_base_timeframe'] = args.tf

    # Step 1: Ensure historical CSV is fresh (fetch gap if stale)
    csv_path = ensure_csv_fresh()

    # Step 2: Load daily data from CSV for reliable EMA55 bias
    df_1d_hist = load_daily_from_csv(csv_path)
    if df_1d_hist is not None:
        print(f"  📊 Daily bias from CSV ({len(df_1d_hist)} bars)")

    # Step 3: Fetch live base-timeframe data
    print(f"Fetching {args.tf} data ({bars} bars)...")
    df_base = fetch_recent(bars=bars, timeframe=args.tf)
    print("Computing indicators...")
    df_base, df_1h, df_2h, df_4h, df_1d = compute_indicators(
        df_base, config=scaled_config, df_1d_hist=df_1d_hist)
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
        # Add conflict resolution and phase detection to JSON output
        cr = detect_conflict(result, config=scaled_config)
        result['conflict_resolution'] = conflict_to_dict(cr)
        p3 = detect_phase(result, config=scaled_config, df_15m=df_base)
        result['power_of_3'] = phase_to_dict(p3)
        print(json.dumps(result, indent=2, default=str))
    else:
        print_signal(result)
        print_summary(result)

        # ── Power of 3 Phase Detection ──
        p3 = detect_phase(result, config=scaled_config, df_15m=df_base)
        print(format_phase(p3))
        result['power_of_3'] = phase_to_dict(p3)

        # ── Conflict Resolution ──
        cr = detect_conflict(result, config=scaled_config)
        if cr.has_conflict:
            print(format_conflict(cr))
            result['conflict_resolution'] = conflict_to_dict(cr)

            # Re-save with all analysis data
            with open(scan_file, 'w') as f:
                json.dump(result, f, indent=2, default=str)

            # Offer to spawn forward test
            if cr.forward_test:
                ft = cr.forward_test
                print(f"\n  💡 To start forward test, run:")
                print(f"     python3 scripts/forward_test.py --level {ft.key_level_low:.0f} {ft.key_level_high:.0f} "
                      f"--scenarios {','.join(s.name.lower() for s in ft.scenarios)}")
        else:
            print(f"\n  ✅ No conflict — scanner verdict stands.")

        # Re-save with phase data
        with open(scan_file, 'w') as f:
            json.dump(result, f, indent=2, default=str)


if __name__ == '__main__':
    main()

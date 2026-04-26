#!/usr/bin/env python3
"""
╔═══════════════════════════════════════════════════════════════════════╗
║ JIMI v6.14 — M7 Backtest Runner                                      ║
║                                                                       ║
║ Runs JIMI v6.12/v6.13 + M7 (USDT Dominance + Market Regime)          ║
║ on Binance ETH/USDT 15m data from 2021-2025.                         ║
║                                                                       ║
║ M7 adds macro regime awareness:                                       ║
║   - USDT dominance trend & momentum                                   ║
║   - ETH/BTC relative strength                                         ║
║   - Volume regime                                                     ║
║                                                                       ║
║ Usage:                                                                ║
║   python3 backtest_m7.py                  # Full 2021-2025 all months ║
║   python3 backtest_m7.py 2021 2024        # Custom year range         ║
║   python3 backtest_m7.py 2021 2025 4      # April only                ║
╚═══════════════════════════════════════════════════════════════════════╝
"""

import ccxt
import pandas as pd
import numpy as np
import sys
import os
import json
import calendar
import time

sys.path.insert(0, os.path.dirname(__file__))

# Import the base framework
from jimi_v612_finetuned import (
    CONFIG, load_data, resample_ohlcv, calc_ema, calc_macd, calc_rsi, calc_atr,
    calc_vwap, calc_vol_ratio, calc_swing_bias, calc_phase0,
    calc_cvd_15m, detect_cvd_divergence_15m, calc_cvd_2h, detect_cvd_zero_cross,
    score_m1, score_m2, score_m3, score_m4, score_m5, calc_ics,
    detect_cascade_setup, check_entry_filters, get_tp_multipliers, Trade
)

# Import M7 module
from jimi_m7_module import prepare_m7_data, get_m7_row_for_date, score_m7

exchange = ccxt.binance({"enableRateLimit": True})

# ═══════════════════════════════════════════════════════════════════════
# M7 CONFIG — Add to existing CONFIG
# ═══════════════════════════════════════════════════════════════════════

CONFIG["M7_WEIGHT"] = 0.08  # M7 weight in ICS (small, macro filter)
CONFIG["M7_ENABLED"] = True


# ═══════════════════════════════════════════════════════════════════════
# DATA FETCHING
# ═══════════════════════════════════════════════════════════════════════

def fetch_ohlcv(symbol, timeframe, since_ms, until_ms):
    """Fetch OHLCV data from Binance via ccxt."""
    all_candles = []
    current = since_ms
    while current < until_ms:
        try:
            raw = exchange.fetch_ohlcv(symbol, timeframe, since=current, limit=1000)
        except Exception as e:
            print(f"  Fetch error: {e}, retrying...")
            time.sleep(5)
            raw = exchange.fetch_ohlcv(symbol, timeframe, since=current, limit=1000)
        if not raw:
            break
        for c in raw:
            ts = int(c[0])
            if ts >= until_ms:
                break
            all_candles.append({
                "Open time": pd.to_datetime(ts, unit="ms"),
                "Open": float(c[1]), "High": float(c[2]),
                "Low": float(c[3]), "Close": float(c[4]),
                "Volume": float(c[5]),
                "Close time": pd.to_datetime(int(c[6]), unit="ms") if len(c) > 6 else pd.to_datetime(ts + 900000, unit="ms"),
                "Quote asset volume": float(c[7]) if len(c) > 7 else 0,
                "Number of trades": int(c[8]) if len(c) > 8 else 0,
                "Taker buy base asset volume": float(c[9]) if len(c) > 9 else 0,
                "Taker buy quote asset volume": float(c[10]) if len(c) > 10 else 0,
            })
        last_ts = raw[-1][0]
        if last_ts <= current:
            break
        current = last_ts + 1
    return pd.DataFrame(all_candles)


# ═══════════════════════════════════════════════════════════════════════
# MODIFIED ICS WITH M7
# ═══════════════════════════════════════════════════════════════════════

def calc_ics_m7(m1_score, m2_score, m3_score, m4_score, m4_status,
                m5_score=0.5, m6_score=0.5, m7_score=0.5,
                use_derivatives=False, use_m7=False,
                cascade_dir='NONE', cascade_strength=0.0):
    """Extended ICS with M7 support."""
    m4_contrib = m4_score if m4_status == 'PASS' else 0.5

    if use_m7 and CONFIG.get("M7_ENABLED", False):
        # Redistribute weights to include M7
        # Keep M1-M5 weights, reduce M6 slightly, add M7
        m6_w = CONFIG['M6_WEIGHT'] * 0.7  # reduce M6 to make room
        m7_w = CONFIG['M7_WEIGHT']
        other_w = 1.0 - m6_w - m7_w  # remaining weight for M1-M5

        # Scale M1-M5 weights proportionally
        base_sum = (CONFIG['M1_WEIGHT'] + CONFIG['M2_WEIGHT'] +
                    CONFIG['M3_WEIGHT'] + CONFIG['M4_WEIGHT'] + CONFIG['M5_WEIGHT'])

        ics = (
            m1_score * (CONFIG['M1_WEIGHT'] / base_sum * other_w) +
            m2_score * (CONFIG['M2_WEIGHT'] / base_sum * other_w) +
            m3_score * (CONFIG['M3_WEIGHT'] / base_sum * other_w) +
            m4_contrib * (CONFIG['M4_WEIGHT'] / base_sum * other_w) +
            m5_score * (CONFIG['M5_WEIGHT'] / base_sum * other_w) +
            m6_score * m6_w +
            m7_score * m7_w
        )
    elif use_derivatives:
        ics = (m1_score * CONFIG['M1_WEIGHT'] +
               m2_score * CONFIG['M2_WEIGHT'] +
               m3_score * CONFIG['M3_WEIGHT_DERIV'] +
               m4_contrib * CONFIG['M4_WEIGHT_DERIV'] +
               m5_score * CONFIG['M5_WEIGHT'] +
               m6_score * CONFIG['M6_WEIGHT'])
    else:
        ics = (m1_score * CONFIG['M1_WEIGHT'] +
               m2_score * CONFIG['M2_WEIGHT'] +
               m3_score * CONFIG['M3_WEIGHT'] +
               m4_contrib * CONFIG['M4_WEIGHT'] +
               m5_score * CONFIG['M5_WEIGHT'])

    # Cascade multiplier
    if cascade_dir == 'WITH' and cascade_strength > 0:
        ics *= 1.0 + (CONFIG.get('CASCADE_MULTIPLIER', 1.12) - 1.0) * cascade_strength
    elif cascade_dir == 'AGAINST' and cascade_strength > 0:
        ics *= 1.0 - (1.0 - CONFIG.get('CASCADE_PENALTY', 0.85)) * cascade_strength

    effective_floor = CONFIG['ICS_FLOOR_M4_FALSE'] if m4_status == 'FAIL' else CONFIG['ICS_FLOOR']
    return ics, effective_floor


# ═══════════════════════════════════════════════════════════════════════
# BACKTEST ENGINE (adapted from v6.12 + M7)
# ═══════════════════════════════════════════════════════════════════════

def run_backtest_m7(csv_path, ethbtc_df, btc_df, verbose=False, date_start=None, date_end=None):
    """
    Run JIMI backtest with M7 integrated.
    """
    print("=" * 70)
    print("  JIMI v6.14 — Backtest Engine (M7: USDT Dominance + Market Regime)")
    if date_start or date_end:
        print(f"  Date Range: {date_start or 'start'} → {date_end or 'end'}")
    print("=" * 70)

    print("\n[1/7] Loading data...")
    df_15m = load_data(csv_path)
    print(f"  15m bars loaded: {len(df_15m):,}")
    print(f"  Date range: {df_15m['Open time'].iloc[0]} → {df_15m['Open time'].iloc[-1]}")

    # Filter by date range
    if date_start:
        start_ts = pd.Timestamp(date_start)
        df_15m = df_15m[df_15m['Open time'] >= start_ts].reset_index(drop=True)
    if date_end:
        end_ts = pd.Timestamp(date_end) + pd.Timedelta(days=1)
        df_15m = df_15m[df_15m['Open time'] < end_ts].reset_index(drop=True)

    if len(df_15m) == 0:
        print("  No data in date range!")
        return [], {}, df_15m

    print(f"  Filtered: {len(df_15m):,} bars ({df_15m['Open time'].iloc[0]} → {df_15m['Open time'].iloc[-1]})")

    print("[2/7] Resampling to 1H, 2H, 4H, 1D...")
    df_1h = resample_ohlcv(df_15m, '1H')
    df_2h = resample_ohlcv(df_15m, '2H')
    df_4h = resample_ohlcv(df_15m, '4H')
    df_1d = resample_ohlcv(df_15m, '1D')
    print(f"  1H: {len(df_1h):,} | 2H: {len(df_2h):,} | 4H: {len(df_4h):,} | 1D: {len(df_1d):,}")

    print("[3/7] Computing indicators...")
    df_15m['vwap'] = calc_vwap(df_15m['High'], df_15m['Low'], df_15m['Close'], df_15m['Volume'], CONFIG['VWAP_LOOKBACK'])
    df_15m['vol_ma20'] = df_15m['Volume'].rolling(20).mean()
    taker_base = df_15m['Taker buy base asset volume']
    total_vol = df_15m['Volume']
    df_15m['taker_ratio'] = (taker_base / total_vol.replace(0, np.nan)).fillna(CONFIG['TAKER_FILLNA'])
    df_15m['atr'] = calc_atr(df_15m['High'], df_15m['Low'], df_15m['Close'], CONFIG['ATR_PERIOD'])
    df_15m['vol_ratio'] = calc_vol_ratio(df_15m['Volume'])

    df_1h['macd_line'], df_1h['macd_signal'], df_1h['macd_hist'] = calc_macd(
        df_1h['Close'], CONFIG['MACD_FAST'], CONFIG['MACD_SLOW'], CONFIG['MACD_SIGNAL'])
    df_1h['ema_fast'] = calc_ema(df_1h['Close'], CONFIG['EMA_FAST'])
    df_1h['ema_slow'] = calc_ema(df_1h['Close'], CONFIG['EMA_SLOW'])
    df_1h['atr'] = calc_atr(df_1h['High'], df_1h['Low'], df_1h['Close'], CONFIG['ATR_PERIOD'])

    df_4h['ema_fast'] = calc_ema(df_4h['Close'], CONFIG['EMA_FAST'])
    df_4h['ema_slow'] = calc_ema(df_4h['Close'], CONFIG['EMA_SLOW'])

    df_2h['ema_fast'] = calc_ema(df_2h['Close'], CONFIG['EMA_FAST'])
    df_2h['ema_slow'] = calc_ema(df_2h['Close'], CONFIG['EMA_SLOW'])

    df_1d['swing_bias'] = calc_swing_bias(df_1d)
    df_1d['phase0'] = calc_phase0(df_1d)

    df_15m['cvd_15m'] = calc_cvd_15m(df_15m)
    df_15m['cvd_divergence_15m'] = detect_cvd_divergence_15m(df_15m)

    df_2h['cvd_2h'] = calc_cvd_2h(df_2h)
    cvd_state, cvd_cross_bar, cvd_cross_dir = detect_cvd_zero_cross(df_2h)
    df_2h['cvd_state'] = cvd_state
    df_2h['cvd_cross_bar'] = cvd_cross_bar
    df_2h['cvd_cross_dir'] = cvd_cross_dir

    print("[4/7] Building time indices...")
    idx_1h = df_15m['Open time'].dt.floor('1h')
    idx_2h = df_15m['Open time'].dt.floor('2h')
    idx_4h = df_15m['Open time'].dt.floor('4h')
    idx_1d = df_15m['Open time'].dt.floor('1D')

    map_1h = {t: i for i, t in enumerate(df_1h['Open time'])}
    map_2h = {t: i for i, t in enumerate(df_2h['Open time'])}
    map_4h = {t: i for i, t in enumerate(df_4h['Open time'])}
    map_1d = {t: i for i, t in enumerate(df_1d['Open time'])}

    print("[5/7] Running backtest with M7...")
    trades = []
    open_trades = []
    stats = {
        'entries': 0, 'exits_sl': 0, 'exits_tp1': 0, 'exits_tp2': 0, 'exits_tp3': 0,
        'exits_signal': 0, 'ics_blocked': 0, 'filter_blocked': 0, 'm3_fail': 0,
        'm1_neutral_skip': 0, 'signals_checked': 0, 'm5_pass': 0, 'm5_fail': 0,
        'cascade_detected': 0, 'm4_false_anchored': 0, 'm4_required_skip': 0,
        'm7_blocked': 0,
    }
    daily_trades = {}
    m7_stats = {'pass': 0, 'fail': 0, 'boost': 0, 'penalty': 0}

    for idx in range(len(df_15m)):
        row = df_15m.iloc[idx]
        ts = row['Open time']

        # --- Close open trades ---
        for trade in open_trades[:]:
            if not trade.is_open:
                continue
            trade.bars_held += 1
            high, low = row['High'], row['Low']

            if trade.direction == 'LONG':
                if low <= trade.sl:
                    trade.close(trade.sl, ts, 'SL')
                    stats['exits_sl'] += 1
                elif not trade.tp1_hit and high >= trade.tp1:
                    trade.tp1_hit = True
                    trade.close(trade.tp1, ts, 'TP1', fraction=CONFIG['TP1_CLOSE'])
                    stats['exits_tp1'] += 1
                    trade.update_sl_trail()
                elif trade.tp1_hit and not trade.tp2_hit and high >= trade.tp2:
                    trade.tp2_hit = True
                    trade.close(trade.tp2, ts, 'TP2', fraction=CONFIG['TP2_CLOSE'])
                    stats['exits_tp2'] += 1
                    trade.update_sl_trail()
                elif trade.tp2_hit and high >= trade.tp3:
                    trade.close(trade.tp3, ts, 'TP3')
                    stats['exits_tp3'] += 1
            else:  # SHORT
                if high >= trade.sl:
                    trade.close(trade.sl, ts, 'SL')
                    stats['exits_sl'] += 1
                elif not trade.tp1_hit and low <= trade.tp1:
                    trade.tp1_hit = True
                    trade.close(trade.tp1, ts, 'TP1', fraction=CONFIG['TP1_CLOSE'])
                    stats['exits_tp1'] += 1
                    trade.update_sl_trail()
                elif trade.tp1_hit and not trade.tp2_hit and low <= trade.tp2:
                    trade.tp2_hit = True
                    trade.close(trade.tp2, ts, 'TP2', fraction=CONFIG['TP2_CLOSE'])
                    stats['exits_tp2'] += 1
                    trade.update_sl_trail()
                elif trade.tp2_hit and low <= trade.tp3:
                    trade.close(trade.tp3, ts, 'TP3')
                    stats['exits_tp3'] += 1

            if not trade.is_open:
                open_trades.remove(trade)

        # --- Early exit (summer) ---
        is_summer = ts.month in CONFIG.get('SUMMER_MONTHS', [6, 7, 8, 9])
        is_shoulder = ts.month in CONFIG.get('SHOULDER_MONTHS', [3, 10])
        early_exit_bars = CONFIG.get('EARLY_EXIT_BARS_SUMMER', 999) if is_summer else CONFIG.get('EARLY_EXIT_BARS', 999)

        for trade in open_trades[:]:
            if trade.bars_held >= early_exit_bars and trade.is_open:
                pnl = ((row['Close'] - trade.entry_price) / trade.entry_price
                       if trade.direction == 'LONG'
                       else (trade.entry_price - row['Close']) / trade.entry_price)
                min_loss = CONFIG.get('EARLY_EXIT_MIN_LOSS_SUMMER', 0.002) if is_summer else CONFIG.get('EARLY_EXIT_MIN_LOSS', 0.003)
                if pnl < -min_loss:
                    trade.close(row['Close'], ts, 'EARLY_EXIT')
                    open_trades.remove(trade)

        # --- Time gates ---
        today = ts.date()
        if today not in daily_trades:
            daily_trades[today] = 0

        max_daily = CONFIG.get('MAX_TRADES_DAY_SUMMER', 5) if is_summer else CONFIG.get('MAX_TRADES_DAY', 5)
        if daily_trades[today] >= max_daily:
            continue

        # --- Monthly DD circuit breaker ---
        monthly_dd_limit = CONFIG.get('MONTHLY_DD_CIRCUIT', 0)
        if monthly_dd_limit > 0:
            month_key = f"{ts.year}-{ts.month:02d}"
            month_trades = [t for t in trades if t.exit_time is not None and hasattr(t.exit_time, 'year') and f"{t.exit_time.year}-{t.exit_time.month:02d}" == month_key]
            month_pnl = sum(t.pnl_pct * t.size_pct for t in month_trades)
            if month_pnl <= -monthly_dd_limit:
                continue

        # --- Cooldown ---
        if trades:
            last_trade = trades[-1]
            if last_trade.exit_time is not None:
                cooldown = CONFIG.get('COOLDOWN_MINUTES_SUMMER', 10) if is_summer else CONFIG.get('COOLDOWN_MINUTES', 10)
                if (ts - last_trade.exit_time).total_seconds() / 60 < cooldown:
                    continue

        # --- Consecutive loss pause ---
        max_consec = CONFIG.get('MAX_CONSEC_LOSS_SUMMER', 3) if is_summer else CONFIG.get('MAX_CONSEC_LOSS', 3)
        if max_consec > 0 and len(trades) >= max_consec:
            recent = trades[-max_consec:]
            if all(t.pnl_pct < 0 for t in recent):
                pause_bars = CONFIG.get('CONSEC_LOSS_PAUSE_SUMMER', 8) if is_summer else CONFIG.get('CONSEC_LOSS_PAUSE_BARS', 8)
                if len(df_15m) > idx + pause_bars:
                    continue

        # --- Rolling win rate filter ---
        wr_window = CONFIG.get('ROLLING_WR_WINDOW', 8)
        wr_min = CONFIG.get('ROLLING_WR_MIN', 0.40)
        if len(trades) >= wr_window:
            recent_trades = trades[-wr_window:]
            recent_wr = sum(1 for t in recent_trades if t.pnl_pct > 0) / len(recent_trades)
            if recent_wr < wr_min:
                continue

        stats['signals_checked'] += 1

        # --- Time index lookups ---
        i_1h = map_1h.get(idx_1h.iloc[idx], -1)
        i_2h = map_2h.get(idx_2h.iloc[idx], -1)
        i_4h = map_4h.get(idx_4h.iloc[idx], -1)
        i_1d = map_1d.get(idx_1d.iloc[idx], -1)

        # --- Phase0 + swing bias ---
        phase0_val = df_1d['phase0'].iloc[i_1d] if i_1d >= 0 else 0
        swing_bias = df_1d['swing_bias'].iloc[i_1d] if i_1d >= 0 else 'NEUTRAL'
        atr_1h = df_1h['atr'].iloc[i_1h] if i_1h >= 0 else np.nan

        # --- Phase0 block ---
        phase0_block = CONFIG.get('PHASE0_SUMMER_BLOCK', 0.90) if is_summer else 0.90
        if phase0_val >= phase0_block:
            continue

        # --- Module Scoring ---
        m1_dir, m1_score = score_m1(df_1h, i_1h)
        m2_status, m2_score = score_m2(df_1h, df_2h, df_4h, df_1d, i_1h, i_2h, i_4h, i_1d)

        if m1_dir == 'BULLISH':
            direction = 'LONG'
        elif m1_dir == 'BEARISH':
            direction = 'SHORT'
        else:
            stats['m1_neutral_skip'] += 1
            continue

        m3_status, m3_score, m3_entry = score_m3(df_15m, idx, direction)
        if m3_status == 'FAIL':
            stats['m3_fail'] += 1
            continue

        if direction == 'LONG' and m2_status == 'NEUTRAL':
            continue

        m4_status, m4_score, m4_div = score_m4(df_15m, df_2h, idx, i_2h, direction)

        # --- Pre-M5 ICS check ---
        ics_pre, effective_floor = calc_ics(m1_score, m2_score, m3_score, m4_score, m4_status, 0.5)
        threshold = CONFIG['ICS_THRESHOLD_CAUTION'] if phase0_val >= 0.40 else CONFIG['ICS_THRESHOLD_NORMAL']
        if is_summer:
            threshold += CONFIG.get('SUMMER_ICS_BOOST', 0)
        elif is_shoulder:
            threshold += CONFIG.get('SHOULDER_ICS_BOOST', 0)
        if ics_pre < effective_floor or ics_pre < threshold:
            stats['ics_blocked'] += 1
            continue

        # --- M5 (cached every 4 bars) ---
        m5_cache_key = idx // 4
        if not hasattr(score_m5, '_cache') or score_m5._cache_key != m5_cache_key:
            m5_status, m5_score, m5_details = score_m5(df_15m, idx, direction,
                n_bins=CONFIG['M5_VP_BINS'], lookback=CONFIG['M5_VP_LOOKBACK'])
            cascade = detect_cascade_setup(df_15m, idx)
            score_m5._cache = (m5_status, m5_score, m5_details, cascade)
            score_m5._cache_key = m5_cache_key
        else:
            m5_status, m5_score, m5_details, cascade = score_m5._cache

        cascade_dir = m5_details.get('cascade_dir', 'NONE') if isinstance(m5_details, dict) else 'NONE'
        cascade_strength = m5_details.get('cascade_strength', 0.0) if isinstance(m5_details, dict) else 0.0

        # ===== M7 SCORING =====
        ethbtc_row, btc_row = get_m7_row_for_date(ethbtc_df, btc_df, ts)
        m7_15m_row = {
            'vol_ratio': row.get('vol_ratio', np.nan),
        }
        m7_status, m7_score, m7_details = score_m7(ethbtc_row, btc_row, m7_15m_row, direction)

        if m7_status == 'PASS':
            m7_stats['pass'] += 1
        else:
            m7_stats['fail'] += 1
            stats['m7_blocked'] += 1

        # Track M7 impact on ICS
        ics_base, _ = calc_ics(m1_score, m2_score, m3_score, m4_score, m4_status, m5_score,
                               cascade_dir=cascade_dir, cascade_strength=cascade_strength)
        ics_m7, effective_floor = calc_ics_m7(
            m1_score, m2_score, m3_score, m4_score, m4_status,
            m5_score=m5_score, m7_score=m7_score, use_m7=True,
            cascade_dir=cascade_dir, cascade_strength=cascade_strength)

        if ics_m7 > ics_base:
            m7_stats['boost'] += 1
        elif ics_m7 < ics_base:
            m7_stats['penalty'] += 1

        ics = ics_m7

        if ics < effective_floor or ics < threshold:
            stats['ics_blocked'] += 1
            continue

        # --- ICS Ceiling ---
        ics_ceiling = CONFIG.get('ICS_CEILING', 1.0)
        if ics > ics_ceiling:
            continue

        # --- M4 PASS Required ---
        if m4_status == 'FAIL':
            stats['m4_required_skip'] += 1
            continue

        # --- Directional Veto ---
        if CONFIG.get('DIR_VETO_ENABLED', False):
            m4_disagree = (direction == 'LONG' and m4_div == 'BEARISH') or \
                          (direction == 'SHORT' and m4_div == 'BULLISH')
            m5_disagree = (m5_status == 'FAIL')
            if m4_disagree and m5_disagree:
                continue

        # --- Bias gate ---
        bad_gate_months = [3, 7, 9]
        if CONFIG.get('BIAS_GATE_ENABLED', False) and direction == 'LONG' and swing_bias == 'BEARISH' and ts.month in bad_gate_months:
            if ics < CONFIG.get('BIAS_GATE_LONG_ICS', 0.65):
                continue

        # --- Entry filters ---
        passed, reason = check_entry_filters(df_15m, idx, direction, swing_bias, phase0_val, atr_1h)
        if not passed:
            stats['filter_blocked'] += 1
            continue

        # --- Entry dedup ---
        min_dist = CONFIG.get('MIN_ENTRY_DIST_PCT', 0)
        if min_dist > 0 and trades:
            last_trade = trades[-1]
            price_dist = abs(row['Close'] - last_trade.entry_price) / last_trade.entry_price
            if price_dist < min_dist:
                continue

        # --- Position Sizing ---
        size = CONFIG.get('SIZE_LONG', CONFIG['SIZE_STD']) if direction == 'LONG' else CONFIG['SIZE_STD']
        if m2_status == 'NEUTRAL':
            size *= CONFIG['SIZE_M2_NEUTRAL']
        if phase0_val >= 0.40:
            size *= CONFIG['SIZE_CAUTION']
        if is_summer:
            size *= CONFIG.get('SUMMER_SIZE_MULT', 1.0)
        elif is_shoulder:
            size *= CONFIG.get('SHOULDER_SIZE_MULT', 1.0)

        # M7 size adjustment: reduce size when M7 is bearish
        if m7_score < 0.35:
            size *= 0.7  # 30% reduction when macro is very unfavorable
        elif m7_score < 0.45:
            size *= 0.85  # 15% reduction when macro is mildly unfavorable

        if size < 0.01:
            continue

        entry_price = row['Close']
        atr_for_sl = atr_1h if not pd.isna(atr_1h) else row['atr']
        if is_summer:
            sl_std = CONFIG.get('SL_ATR_STD_SUMMER', CONFIG['SL_ATR_STD'])
            sl_hard_max = CONFIG.get('SL_HARD_MAX_SUMMER', CONFIG['SL_HARD_MAX_PCT'])
        elif is_shoulder:
            sl_std = CONFIG.get('SHOULDER_SL_ATR', CONFIG['SL_ATR_STD'])
            sl_hard_max = CONFIG.get('SHOULDER_SL_HARD_MAX', CONFIG['SL_HARD_MAX_PCT'])
        else:
            sl_std = CONFIG['SL_ATR_STD']
            sl_hard_max = CONFIG['SL_HARD_MAX_PCT']
        sl_dist = min(sl_std * atr_for_sl, sl_hard_max * entry_price)
        if is_summer:
            tp1_atr = CONFIG.get('TP1_ATR_SUMMER', CONFIG['TP1_ATR'])
        elif is_shoulder:
            tp1_atr = CONFIG.get('SHOULDER_TP1_ATR', CONFIG['TP1_ATR'])
        else:
            tp1_atr = CONFIG['TP1_ATR']
        tp1_dist = tp1_atr * atr_for_sl
        tp2_mult, tp3_mult = get_tp_multipliers(row.get('vol_ratio', np.nan))
        tp2_dist, tp3_dist = tp2_mult * atr_for_sl, tp3_mult * atr_for_sl

        if direction == 'LONG':
            sl, tp1, tp2, tp3 = entry_price - sl_dist, entry_price + tp1_dist, entry_price + tp2_dist, entry_price + tp3_dist
        else:
            sl, tp1, tp2, tp3 = entry_price + sl_dist, entry_price - tp1_dist, entry_price - tp2_dist, entry_price - tp3_dist

        trade = Trade(ts, direction, entry_price, sl, tp1, tp2, tp3, size,
                      m1_dir, m2_status, m3_score, m4_status, m5_status, m5_score,
                      ics, phase0_val,
                      f"M1={m1_dir} M2={m2_status} M3={m3_status} M4={m4_status} M5={m5_status} M7={m7_status}({m7_score:.2f}) ICS={ics:.3f}")
        open_trades.append(trade)
        trades.append(trade)
        daily_trades[today] += 1
        stats['entries'] += 1

        if verbose and stats['entries'] <= 50:
            print(f"  ENTRY #{stats['entries']}: {ts} {direction} @ {entry_price:.2f} "
                  f"SL={sl:.2f} TP1={tp1:.2f} ICS={ics:.3f} M7={m7_score:.2f}({m7_details.get('regime','?')}) size={size:.2f}")

    # Close remaining
    if open_trades:
        last_row = df_15m.iloc[-1]
        for trade in open_trades:
            if trade.is_open:
                trade.close(last_row['Close'], last_row['Open time'], 'END')
                stats['exits_signal'] += 1

    print(f"\n[6/7] M7 Impact Summary:")
    print(f"  M7 Pass: {m7_stats['pass']} | Fail: {m7_stats['fail']}")
    print(f"  M7 Boosted ICS: {m7_stats['boost']} | Penalized ICS: {m7_stats['penalty']}")

    print("\n[7/7] Computing results...")
    return trades, stats, df_15m


# ═══════════════════════════════════════════════════════════════════════
# REPORTING
# ═══════════════════════════════════════════════════════════════════════

def print_report(trades, stats, label=""):
    if not trades:
        print("\n  No trades generated.")
        return {}

    total = len(trades)
    winners = [t for t in trades if t.pnl_pct > 0]
    losers = [t for t in trades if t.pnl_pct < 0]
    win_rate = len(winners) / total * 100
    total_pnl = sum(t.pnl_pct * t.size_pct for t in trades)
    avg_win = np.mean([t.pnl_pct for t in winners]) if winners else 0
    avg_loss = np.mean([t.pnl_pct for t in losers]) if losers else 0
    gross_profit = sum(t.pnl_pct * t.size_pct for t in winners)
    gross_loss = abs(sum(t.pnl_pct * t.size_pct for t in losers))
    profit_factor = gross_profit / gross_loss if gross_loss > 0 else float('inf')

    equity = [0]
    for t in sorted(trades, key=lambda x: x.exit_time):
        equity.append(equity[-1] + t.pnl_pct * t.size_pct)
    equity = np.array(equity)
    peak = np.maximum.accumulate(equity)
    max_dd = abs((equity - peak).min()) if len(equity) > 0 else 0

    longs = [t for t in trades if t.direction == 'LONG']
    shorts = [t for t in trades if t.direction == 'SHORT']

    print(f"\n{'='*70}")
    print(f"  JIMI v6.14 + M7 — {label} RESULTS")
    print(f"{'='*70}")
    print(f"  Total Trades: {total}")
    print(f"  Win Rate: {win_rate:.1f}%")
    print(f"  Total PnL: {total_pnl*100:+.2f}%")
    print(f"  Profit Factor: {profit_factor:.2f}")
    print(f"  Max Drawdown: {max_dd*100:.2f}%")
    print(f"  Avg Win: {avg_win*100:+.2f}% | Avg Loss: {avg_loss*100:+.2f}%")
    print(f"  Longs: {len(longs)} | Shorts: {len(shorts)}")
    print(f"  Exits — SL: {stats.get('exits_sl',0)} TP1: {stats.get('exits_tp1',0)} TP2: {stats.get('exits_tp2',0)} TP3: {stats.get('exits_tp3',0)}")
    print(f"  M7 Blocked: {stats.get('m7_blocked',0)} | ICS Blocked: {stats.get('ics_blocked',0)} | Filter Blocked: {stats.get('filter_blocked',0)}")

    return {
        'trades': total, 'wins': len(winners), 'losses': len(losers),
        'win_rate': win_rate, 'total_pnl': total_pnl * 100,
        'pf': profit_factor, 'max_dd': max_dd * 100,
        'longs': len(longs), 'shorts': len(shorts),
    }


# ═══════════════════════════════════════════════════════════════════════
# MAIN — Run full backtest 2021-2025
# ═══════════════════════════════════════════════════════════════════════

def main():
    start_year = int(sys.argv[1]) if len(sys.argv) > 1 else 2021
    end_year = int(sys.argv[2]) if len(sys.argv) > 2 else 2025
    target_month = int(sys.argv[3]) if len(sys.argv) > 3 else None

    print(f"\n{'#'*70}")
    print(f"  JIMI v6.14 + M7 — Full Backtest {start_year}-{end_year}")
    if target_month:
        print(f"  Month filter: {target_month}")
    print(f"{'#'*70}")

    # Fetch data
    print("\n[DATA] Fetching ETH/USDT 15m data from Binance...")
    since = pd.Timestamp(f"{start_year}-01-01", tz='UTC')
    until = pd.Timestamp(f"{end_year+1}-01-01", tz='UTC')
    # Add warmup buffer (90 days before start)
    fetch_start = since - pd.Timedelta(days=90)

    csv_path = f"/tmp/eth_15m_m7_{start_year}_{end_year}.csv"
    if not os.path.exists(csv_path):
        df = fetch_ohlcv('ETH/USDT', '15m',
                         int(fetch_start.timestamp() * 1000),
                         int(until.timestamp() * 1000))
        df.to_csv(csv_path, index=False)
        print(f"  Fetched {len(df):,} bars, saved to {csv_path}")
    else:
        print(f"  Using cached data: {csv_path}")

    df_full = pd.read_csv(csv_path)
    df_full['Open time'] = pd.to_datetime(df_full['Open time'])
    print(f"  Date range: {df_full['Open time'].iloc[0]} → {df_full['Open time'].iloc[-1]}")

    # Prepare M7 data
    print("\n[M7] Fetching USDT dominance & ETH/BTC data...")
    ethbtc_df, btc_df = prepare_m7_data(df_full)

    # Run backtest
    all_results = []

    if target_month:
        months_to_run = [target_month]
    else:
        months_to_run = list(range(1, 13))

    for month in months_to_run:
        month_name = calendar.month_name[month]
        print(f"\n{'='*60}")
        print(f"  {month_name} ({start_year}-{end_year})")
        print(f"{'='*60}")

        month_results = []
        for year in range(start_year, end_year + 1):
            last_day = calendar.monthrange(year, month)[1]
            date_s = f"{year}-{month:02d}-01"
            date_e = f"{year}-{month:02d}-{last_day}"

            try:
                trades, stats, df_bt = run_backtest_m7(
                    csv_path, ethbtc_df, btc_df,
                    verbose=False, date_start=date_s, date_end=date_e)

                total = len(trades)
                if total == 0:
                    month_results.append({'year': year, 'trades': 0})
                    print(f"  {year}: No trades")
                    continue

                winners = [t for t in trades if t.pnl_pct > 0]
                losers = [t for t in trades if t.pnl_pct < 0]
                wins, losses = len(winners), len(losers)
                wr = wins / total * 100
                total_pnl = sum(t.pnl_pct * t.size_pct for t in trades)
                gross_profit = sum(t.pnl_pct * t.size_pct for t in winners)
                gross_loss = abs(sum(t.pnl_pct * t.size_pct for t in losers))
                pf = gross_profit / gross_loss if gross_loss > 0 else float('inf')

                equity = [0]
                for t in sorted(trades, key=lambda x: x.exit_time):
                    equity.append(equity[-1] + t.pnl_pct * t.size_pct)
                equity = np.array(equity)
                peak = np.maximum.accumulate(equity)
                max_dd = abs((equity - peak).min()) if len(equity) > 0 else 0

                month_results.append({
                    'year': year, 'trades': total, 'wins': wins, 'losses': losses,
                    'win_rate': wr, 'total_pnl': total_pnl * 100,
                    'pf': pf, 'max_dd': max_dd * 100,
                    'm7_blocked': stats.get('m7_blocked', 0),
                })
                print(f"  {year}: {total} trades, WR {wr:.1f}%, PnL {total_pnl*100:+.2f}%, PF {pf:.2f}, DD {max_dd*100:.1f}%, M7-blocked {stats.get('m7_blocked',0)}")

            except Exception as e:
                import traceback
                traceback.print_exc()
                month_results.append({'year': year, 'error': str(e)})
                print(f"  {year}: ERROR - {e}")

        # Month summary
        valid = [r for r in month_results if 'error' not in r and r['trades'] > 0]
        if valid:
            profitable = sum(1 for r in valid if r['total_pnl'] > 0)
            avg_pnl = sum(r['total_pnl'] for r in valid) / len(valid)
            avg_wr = sum(r['win_rate'] for r in valid) / len(valid)
            avg_pf = sum(r['pf'] for r in valid) / len(valid)
            avg_dd = sum(r['max_dd'] for r in valid) / len(valid)
            total_m7_blocked = sum(r.get('m7_blocked', 0) for r in valid)

            all_results.append({
                'month': month_name, 'month_num': month,
                'profitable': f"{profitable}/{len(valid)}",
                'profitable_pct': profitable / len(valid) * 100,
                'avg_pnl': avg_pnl, 'avg_wr': avg_wr,
                'avg_pf': avg_pf, 'avg_dd': avg_dd,
                'total_m7_blocked': total_m7_blocked,
                'best': max(valid, key=lambda x: x['total_pnl']),
                'worst': min(valid, key=lambda x: x['total_pnl']),
            })

    # ═══ FINAL SUMMARY ═══
    print(f"\n{'#'*80}")
    print(f"  JIMI v6.14 + M7 — MONTHLY BACKTEST SUMMARY ({start_year}-{end_year})")
    print(f"{'#'*80}")
    print(f"\n{'Month':>12} {'Profitable':>12} {'Avg PnL%':>10} {'Avg WR%':>9} {'Avg PF':>8} {'Avg DD%':>9} {'M7-Blk':>8}")
    print("-" * 75)

    for r in sorted(all_results, key=lambda x: -x['avg_pnl']):
        pf_s = f"{r['avg_pf']:.2f}" if r['avg_pf'] != float('inf') else "inf"
        print(f"{r['month']:>12} {r['profitable']:>12} {r['avg_pnl']:>+9.1f}% {r['avg_wr']:>8.1f}% {pf_s:>8} {r['avg_dd']:>8.1f}% {r['total_m7_blocked']:>8}")

    # Tier list
    print(f"\n{'='*60}")
    print("  SEASONAL TIER LIST (v6.14 + M7)")
    print(f"{'='*60}")

    green = [r for r in all_results if r['profitable_pct'] >= 67 and r['avg_pnl'] > 20]
    yellow = [r for r in all_results if r['avg_pnl'] > 0 and r not in green]
    orange = [r for r in all_results if r['avg_pnl'] <= 0 and r['avg_pnl'] > -50]
    red = [r for r in all_results if r['avg_pnl'] <= -50]

    if green:
        print(f"  🟢 RUN:       {', '.join(r['month'] for r in sorted(green, key=lambda x: -x['avg_pnl']))}")
    if yellow:
        print(f"  🟡 CONDITIONAL: {', '.join(r['month'] for r in sorted(yellow, key=lambda x: -x['avg_pnl']))}")
    if orange:
        print(f"  🟠 CAUTION:   {', '.join(r['month'] for r in sorted(orange, key=lambda x: -x['avg_pnl']))}")
    if red:
        print(f"  🔴 SKIP:      {', '.join(r['month'] for r in sorted(red, key=lambda x: -x['avg_pnl']))}")

    # Save results
    results_file = f"/tmp/jimi_m7_results_{start_year}_{end_year}.json"
    with open(results_file, "w") as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"\n  Results saved to {results_file}")


if __name__ == "__main__":
    main()

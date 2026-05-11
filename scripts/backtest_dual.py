#!/usr/bin/env python3
"""
JIMI Dual Strategy Backtest — Range Scalper + Momentum Rider

Usage:
    python scripts/backtest_dual.py eth_15m_merged.csv --start 2026-04-19 --end 2026-05-11
    python scripts/backtest_dual.py eth_15m_merged.csv --start 2026-04-19 --end 2026-05-11 --verbose
"""

import argparse
import sys
import os
import json
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from src.config import CONFIG, load_config
from src.engine import run_backtest, Trade
from src.dual_strategy import DualStrategy
from src.utils.data_handler import load_data, resample_ohlcv
from src.utils.indicators import (
    calc_ema, calc_macd, calc_rsi, calc_atr, calc_vwap, calc_vol_ratio,
    calc_swing_bias, calc_phase0, calc_trend_state,
)
from src.modules.m9_volatility import RegimeState, compute_vol_regime, score_vol_regime
from src.modules.m13_structure import score_m13
from src.modules.direction_resolver import resolve_direction, score_targets
from src.modules.m5_liquidation import build_volume_profile, find_magnets, find_gaps, find_support_resistance


def run_dual_backtest(csv_path, config=None, verbose=False, date_start=None, date_end=None):
    """Run both strategies on historical data."""
    cfg = config or CONFIG

    print("=" * 70)
    print("  JIMI — DUAL STRATEGY BACKTEST")
    print(f"  Date Range: {date_start or 'start'} → {date_end or 'end'}")
    print("=" * 70)

    # Load data
    df_15m = load_data(csv_path)
    df_1h = resample_ohlcv(df_15m, '1H')
    df_2h = resample_ohlcv(df_15m, '2H')
    df_4h = resample_ohlcv(df_15m, '4H')
    df_1d = resample_ohlcv(df_15m, '1D')

    # Indicators
    df_15m['vwap'] = calc_vwap(df_15m['High'], df_15m['Low'], df_15m['Close'], df_15m['Volume'], cfg['VWAP_LOOKBACK'])
    df_15m['vol_ma20'] = df_15m['Volume'].rolling(20).mean()
    taker_base = df_15m['Taker buy base asset volume']
    total_vol = df_15m['Volume']
    df_15m['taker_ratio'] = (taker_base / total_vol.replace(0, np.nan)).fillna(cfg['TAKER_FILLNA'])
    df_15m['atr'] = calc_atr(df_15m['High'], df_15m['Low'], df_15m['Close'], cfg['ATR_PERIOD'])
    df_15m['vol_ratio'] = calc_vol_ratio(df_15m['Volume'])
    df_1h['atr'] = calc_atr(df_1h['High'], df_1h['Low'], df_1h['Close'], cfg['ATR_PERIOD'])
    df_1d['swing_bias'] = calc_swing_bias(df_1d)
    df_1d['phase0'] = calc_phase0(df_1d)
    df_1d['trend'], df_1d['trend_score'] = calc_trend_state(df_1d)
    df_15m['rsi'] = calc_rsi(df_15m['Close'], 14)
    df_1h['rsi'] = calc_rsi(df_1h['Close'], 14)
    df_1h['ema_fast'] = calc_ema(df_1h['Close'], cfg['EMA_FAST'])
    df_1h['ema_slow'] = calc_ema(df_1h['Close'], cfg['EMA_SLOW'])
    df_1h['macd_line'], df_1h['macd_signal'], df_1h['macd_hist'] = calc_macd(
        df_1h['Close'], cfg['MACD_FAST'], cfg['MACD_SLOW'], cfg['MACD_SIGNAL'])
    df_2h['ema_fast'] = calc_ema(df_2h['Close'], cfg['EMA_FAST'])
    df_2h['ema_slow'] = calc_ema(df_2h['Close'], cfg['EMA_SLOW'])
    df_4h['ema_fast'] = calc_ema(df_4h['Close'], cfg['EMA_FAST'])
    df_4h['ema_slow'] = calc_ema(df_4h['Close'], cfg['EMA_SLOW'])
    df_4h['macd_line'], df_4h['macd_signal'], df_4h['macd_hist'] = calc_macd(
        df_4h['Close'], cfg['MACD_FAST'], cfg['MACD_SLOW'], cfg['MACD_SIGNAL'])

    # CVD for M4
    from src.modules.m4_cvd import calc_cvd_15m, detect_cvd_divergence_15m, calc_cvd_2h, detect_cvd_zero_cross
    df_15m['cvd_15m'] = calc_cvd_15m(df_15m)
    df_15m['cvd_divergence_15m'] = detect_cvd_divergence_15m(df_15m, cfg['CVD_LOOKBACK'], cfg['CVD_DIVERGENCE_WINDOW'])
    df_2h['cvd_2h'] = calc_cvd_2h(df_2h)
    df_2h['cvd_zl_state'], df_2h['cvd_zl_cross_bar'], df_2h['cvd_zl_cross_dir'] = detect_cvd_zero_cross(df_2h)

    # Index maps
    df_1h['_ts'] = df_1h['Open time'].values.astype('datetime64[ns]')
    df_2h['_ts'] = df_2h['Open time'].values.astype('datetime64[ns]')
    df_4h['_ts'] = df_4h['Open time'].values.astype('datetime64[ns]')
    df_1d['_ts'] = df_1d['Open time'].values.astype('datetime64[ns]')

    def find_tf_idx(ts, df_tf):
        idx = df_tf['_ts'].searchsorted(ts, side='right') - 1
        return max(idx, -1)

    ds = DualStrategy(config=cfg)
    regime_state = RegimeState(config=cfg)

    # Trade tracking
    trades_a, trades_b = [], []
    open_a, open_b = [], []
    last_entry_a, last_entry_b = -999, -999
    stats_a = {'entries': 0, 'exits_sl': 0, 'exits_tp1': 0, 'exits_tp2': 0, 'exits_tp3': 0, 'exits_time': 0}
    stats_b = {'entries': 0, 'exits_sl': 0, 'exits_tp1': 0, 'exits_tp2': 0, 'exits_tp3': 0, 'exits_time': 0}

    warmup = df_1h['Open time'].iloc[min(cfg['WARMUP_BARS_1H'], len(df_1h)-1)]

    print(f"\nRunning dual backtest...")
    print(f"  15m bars: {len(df_15m):,}")
    print(f"  Warmup until: {warmup}")

    for idx in range(len(df_15m)):
        row = df_15m.iloc[idx]
        ts = row['Open time']

        if ts < warmup:
            continue
        if date_start and str(ts) < date_start:
            continue
        if date_end and str(ts) > date_end:
            continue

        idx_1h = find_tf_idx(ts, df_1h)
        idx_1d = find_tf_idx(ts, df_1d)
        if idx_1h < 1 or idx_1d < 0:
            continue

        high, low, close = float(row['High']), float(row['Low']), float(row['Close'])

        # ── Check exits for open trades ──
        for trade in open_a[:]:
            if not trade.is_open:
                continue
            trade.bars_held += 1
            if trade.direction == 'LONG' and low <= trade.sl:
                trade.close(trade.sl, ts, 'SL'); stats_a['exits_sl'] += 1
            elif trade.direction == 'SHORT' and high >= trade.sl:
                trade.close(trade.sl, ts, 'SL'); stats_a['exits_sl'] += 1
            elif not trade.tp1_hit:
                if trade.direction == 'LONG' and high >= trade.tp1:
                    trade.close(trade.tp1, ts, 'TP1', 0.90); trade.tp1_hit = True; stats_a['exits_tp1'] += 1
                elif trade.direction == 'SHORT' and low <= trade.tp1:
                    trade.close(trade.tp1, ts, 'TP1', 0.90); trade.tp1_hit = True; stats_a['exits_tp1'] += 1
            # Time stop
            if trade.is_open and trade.bars_held >= cfg.get('EARLY_EXIT_BARS', 24) and not trade.tp1_hit:
                current_pnl = ((close - trade.entry_price) / trade.entry_price if trade.direction == 'LONG'
                               else (trade.entry_price - close) / trade.entry_price)
                if current_pnl < -cfg.get('EARLY_EXIT_MIN_LOSS', 0.003):
                    trade.close(close, ts, 'EARLY_EXIT'); stats_a['exits_time'] += 1

        for trade in open_b[:]:
            if not trade.is_open:
                continue
            trade.bars_held += 1
            mom_sl = trade.sl  # SL already set wider for momentum
            if trade.direction == 'LONG' and low <= mom_sl:
                trade.close(mom_sl, ts, 'SL'); stats_b['exits_sl'] += 1
            elif trade.direction == 'SHORT' and high >= mom_sl:
                trade.close(mom_sl, ts, 'SL'); stats_b['exits_sl'] += 1
            elif not trade.tp1_hit:
                if trade.direction == 'LONG' and high >= trade.tp1:
                    trade.close(trade.tp1, ts, 'TP1', 0.15); trade.tp1_hit = True; trade.update_sl_trail(); stats_b['exits_tp1'] += 1
                elif trade.direction == 'SHORT' and low <= trade.tp1:
                    trade.close(trade.tp1, ts, 'TP1', 0.15); trade.tp1_hit = True; trade.update_sl_trail(); stats_b['exits_tp1'] += 1
            elif trade.tp1_hit and not trade.tp2_hit:
                if trade.direction == 'LONG' and high >= trade.tp2:
                    frac = 0.25 / (1 - 0.15)
                    trade.close(trade.tp2, ts, 'TP2', frac); trade.tp2_hit = True; trade.update_sl_trail(); stats_b['exits_tp2'] += 1
                elif trade.direction == 'SHORT' and low <= trade.tp2:
                    frac = 0.25 / (1 - 0.15)
                    trade.close(trade.tp2, ts, 'TP2', frac); trade.tp2_hit = True; trade.update_sl_trail(); stats_b['exits_tp2'] += 1
            elif trade.tp1_hit and trade.tp2_hit:
                if trade.direction == 'LONG' and high >= trade.tp3:
                    trade.close(trade.tp3, ts, 'TP3', trade.remaining); stats_b['exits_tp3'] += 1
                elif trade.direction == 'SHORT' and low <= trade.tp3:
                    trade.close(trade.tp3, ts, 'TP3', trade.remaining); stats_b['exits_tp3'] += 1
            # Time stop for momentum
            if trade.is_open and trade.bars_held >= ds.mom_cfg.get('MOM_TIME_STOP_BARS', 80) and not trade.tp1_hit:
                trade.close(close, ts, 'TIME_STOP'); stats_b['exits_time'] += 1

        open_a = [t for t in open_a if t.is_open]
        open_b = [t for t in open_b if t.is_open]

        # ── Cooldown ──
        if idx - last_entry_a < cfg.get('COOLDOWN_BARS', 2):
            continue_a = False
        else:
            continue_a = True
        if idx - last_entry_b < ds.mom_cfg.get('MOM_COOLDOWN_BARS', 8):
            continue_b = False
        else:
            continue_b = True

        if not continue_a and not continue_b:
            continue

        # ── Run dual strategy ──
        idx_2h = find_tf_idx(ts, df_2h)
        idx_4h = find_tf_idx(ts, df_4h)
        if idx_2h < 0 or idx_4h < 0:
            continue

        result = ds.scan(df_15m, df_1h, df_2h, df_4h, df_1d,
                         config=cfg, current_idx=idx)

        sa = result.get('strategy_a') or {}
        sb = result.get('strategy_b') or {}

        # ── Strategy A entry ──
        if continue_a and sa.get('status') == 'SIGNAL' and not open_a:
            entry = sa['entry']
            sl = sa['sl']
            tp1 = sa['tp1']
            tp2 = sa['tp2']
            tp3 = sa['tp3']
            direction = sa['direction']
            ics = sa['ics']

            atr_1h = float(df_1h['atr'].iloc[idx_1h])
            trade = Trade(ts, direction, entry, sl, tp1, tp2, tp3,
                         cfg['SIZE_STD'], 'NEUTRAL', 'PASS', 0.5, 'PASS', 'PASS', 0.5,
                         ics, 0.0, f"scalp ICS={ics:.3f}")
            open_a.append(trade)
            trades_a.append(trade)
            stats_a['entries'] += 1
            last_entry_a = idx

            if verbose and stats_a['entries'] <= 20:
                print(f"  SCALP #{stats_a['entries']}: {ts} {direction} @ {entry:.2f} "
                      f"SL={sl:.2f} TP1={tp1:.2f} ICS={ics:.3f}")

        # ── Strategy B entry ──
        if continue_b and sb.get('status') == 'SIGNAL' and not open_b:
            entry = sb['entry']
            sl = sb['sl']
            tp1 = sb['tp1']
            tp2 = sb['tp2']
            tp3 = sb['tp3']
            direction = sb['direction']
            ics = sb['ics']

            trade = Trade(ts, direction, entry, sl, tp1, tp2, tp3,
                         ds.mom_cfg.get('MOM_SIZE_STD', 3.0), 'NEUTRAL', 'PASS', 0.5, 'PASS', 'PASS', 0.5,
                         ics, 0.0, f"momentum ICS={ics:.3f}")
            open_b.append(trade)
            trades_b.append(trade)
            stats_b['entries'] += 1
            last_entry_b = idx

            if verbose and stats_b['entries'] <= 20:
                strength = sb.get('momentum_strength', 0)
                reason = sb.get('momentum_reason', '')
                print(f"  MOM #{stats_b['entries']}: {ts} {direction} @ {entry:.2f} "
                      f"SL={sl:.2f} TP1={tp1:.2f} ICS={ics:.3f} str={strength:.2f} ({reason})")

    # Close remaining
    last_row = df_15m.iloc[-1]
    for t in open_a + open_b:
        if t.is_open:
            t.close(float(last_row['Close']), last_row['Open time'], 'END')

    # ── Report ──
    print("\n" + "═" * 70)
    print("  DUAL STRATEGY RESULTS")
    print("═" * 70)

    for name, trades, stats in [
        ('STRATEGY A: RANGE SCALPER', trades_a, stats_a),
        ('STRATEGY B: MOMENTUM RIDER', trades_b, stats_b),
    ]:
        print(f"\n  {'─' * 56}")
        print(f"  {name}")
        print(f"  {'─' * 56}")

        if not trades:
            print(f"  No trades.")
            continue

        total = len(trades)
        winners = [t for t in trades if t.pnl_pct > 0]
        losers = [t for t in trades if t.pnl_pct < 0]
        wr = len(winners) / total * 100
        total_pnl = sum(t.pnl_pct * t.size_pct for t in trades)
        avg_win = np.mean([t.pnl_pct for t in winners]) * 100 if winners else 0
        avg_loss = np.mean([t.pnl_pct for t in losers]) * 100 if losers else 0
        avg_bars = np.mean([t.bars_held for t in trades])

        gross_profit = sum(t.pnl_pct * t.size_pct for t in winners)
        gross_loss = abs(sum(t.pnl_pct * t.size_pct for t in losers))
        pf = gross_profit / gross_loss if gross_loss > 0 else float('inf')

        print(f"  Trades:      {total}")
        print(f"  Winners:     {len(winners)} ({wr:.1f}%)")
        print(f"  Losers:      {len(losers)}")
        print(f"  Net PnL:     {total_pnl*100:.2f}%")
        print(f"  Avg Win:     {avg_win:.2f}%")
        print(f"  Avg Loss:    {avg_loss:.2f}%")
        print(f"  Profit Factor: {pf:.2f}")
        print(f"  Avg Bars:    {avg_bars:.1f}")

        # Exit breakdown
        exits = {}
        for t in trades:
            exits[t.exit_reason] = exits.get(t.exit_reason, 0) + 1
        print(f"  Exits:       {exits}")

        # Individual trades
        print(f"\n  {'Time':<20} {'Dir':>6} {'Entry':>9} {'Exit':>9} {'PnL':>8} {'Bars':>5} {'Exit':>10}")
        for t in trades:
            pnl_str = f"{t.pnl_pct*100:+.2f}%"
            print(f"  {str(t.entry_time):<20} {t.direction:>6} {t.entry_price:>9.2f} "
                  f"{t.exit_price:>9.2f} {pnl_str:>8} {t.bars_held:>5} {t.exit_reason:>10}")

    # Combined
    all_trades = trades_a + trades_b
    if all_trades:
        total_combined = sum(t.pnl_pct * t.size_pct for t in all_trades)
        print(f"\n  {'═' * 56}")
        print(f"  COMBINED NET PnL: {total_combined*100:.2f}%")
        print(f"  Total trades:     {len(all_trades)}")
        print(f"  {'═' * 56}")

    return trades_a, trades_b, stats_a, stats_b


def main():
    parser = argparse.ArgumentParser(description='JIMI Dual Strategy Backtest')
    parser.add_argument('csv', nargs='?', help='Path to 15m OHLCV CSV')
    parser.add_argument('--start', help='Start date')
    parser.add_argument('--end', help='End date')
    parser.add_argument('--config', help='Config YAML path')
    parser.add_argument('--verbose', '-v', action='store_true')
    args = parser.parse_args()

    cfg = load_config(args.config) if args.config else None

    if not args.csv or not os.path.exists(args.csv):
        print("ERROR: Provide a valid CSV path")
        sys.exit(1)

    trades_a, trades_b, stats_a, stats_b = run_dual_backtest(
        args.csv, config=cfg, verbose=args.verbose,
        date_start=args.start, date_end=args.end)


if __name__ == '__main__':
    main()

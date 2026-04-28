#!/usr/bin/env python3
"""
M1 Isolated Module Backtest — Compare v1 vs v2.

Tests M1 signal quality in isolation: entry when M1 signals,
exit via simple TP/SL. No other modules involved.

Usage:
    python scripts/backtest_m1_isolated.py eth_15m_merged.csv
    python scripts/backtest_m1_isolated.py eth_15m_merged.csv --start 2023-01-01 --end 2024-12-31
"""

import argparse
import sys
import os
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from src.config import load_config
from src.utils.data_handler import load_data, resample_ohlcv
from src.utils.indicators import calc_ema, calc_macd, calc_rsi, calc_atr
from src.modules.m1_macd import score_m1
from src.modules.m1_macd_v2 import score_m1_v2


def run_m1_backtest(df_15m, config, version='v2', date_start=None, date_end=None,
                    sl_atr=1.3, tp1_atr=1.5, max_bars=32):
    """Run isolated M1 backtest.

    Entry: M1 signals BULLISH/BEARISH with score >= 0.60
    Exit: TP1 (1.5 ATR), SL (1.3 ATR), or max bars held
    """
    cfg = config

    # Resample to 1H
    df_1h = resample_ohlcv(df_15m, '1H')

    # Compute 1H indicators
    df_1h['macd_line'], df_1h['macd_signal'], df_1h['macd_hist'] = calc_macd(
        df_1h['Close'], cfg['MACD_FAST'], cfg['MACD_SLOW'], cfg['MACD_SIGNAL'])
    df_1h['ema_fast'] = calc_ema(df_1h['Close'], cfg['EMA_FAST'])
    df_1h['ema_slow'] = calc_ema(df_1h['Close'], cfg['EMA_SLOW'])
    df_1h['rsi'] = calc_rsi(df_1h['Close'], 14)
    df_1h['atr'] = calc_atr(df_1h['High'], df_1h['Low'], df_1h['Close'], 14)

    # Compute 15m indicators
    df_15m['rsi'] = calc_rsi(df_15m['Close'], 14)
    df_15m['atr'] = calc_atr(df_15m['High'], df_15m['Low'], df_15m['Close'], 14)

    # Build index map
    df_1h['_ts'] = df_1h['Open time'].values.astype('datetime64[ns]')
    df_15m['_ts'] = df_15m['Open time'].values.astype('datetime64[ns]')

    def find_1h_idx(ts):
        idx = df_1h['_ts'].searchsorted(ts, side='right') - 1
        return max(idx, -1)

    warmup = pd.Timestamp(df_1h['Open time'].iloc[min(168, len(df_1h) - 1)])

    trades = []
    open_trade = None

    for idx_15m in range(len(df_15m)):
        row = df_15m.iloc[idx_15m]
        ts = row['Open time']

        if ts < warmup:
            continue
        if date_start and str(ts) < date_start:
            continue
        if date_end and str(ts) > date_end:
            continue

        idx_1h = find_1h_idx(ts)
        if idx_1h < 30:
            continue

        # Check open trade for exit
        if open_trade is not None:
            high, low = row['High'], row['Low']
            open_trade['bars'] += 1

            if open_trade['dir'] == 'LONG':
                if low <= open_trade['sl']:
                    pnl = (open_trade['sl'] - open_trade['entry']) / open_trade['entry']
                    open_trade['exit'] = open_trade['sl']
                    open_trade['pnl'] = pnl
                    open_trade['reason'] = 'SL'
                    trades.append(open_trade)
                    open_trade = None
                    continue
                elif high >= open_trade['tp1']:
                    pnl = (open_trade['tp1'] - open_trade['entry']) / open_trade['entry']
                    open_trade['exit'] = open_trade['tp1']
                    open_trade['pnl'] = pnl
                    open_trade['reason'] = 'TP1'
                    trades.append(open_trade)
                    open_trade = None
                    continue
            else:  # SHORT
                if high >= open_trade['sl']:
                    pnl = (open_trade['entry'] - open_trade['sl']) / open_trade['entry']
                    open_trade['exit'] = open_trade['sl']
                    open_trade['pnl'] = pnl
                    open_trade['reason'] = 'SL'
                    trades.append(open_trade)
                    open_trade = None
                    continue
                elif low <= open_trade['tp1']:
                    pnl = (open_trade['entry'] - open_trade['tp1']) / open_trade['entry']
                    open_trade['exit'] = open_trade['tp1']
                    open_trade['pnl'] = pnl
                    open_trade['reason'] = 'TP1'
                    trades.append(open_trade)
                    open_trade = None
                    continue

            # Time exit
            if open_trade['bars'] >= max_bars:
                pnl = ((row['Close'] - open_trade['entry']) / open_trade['entry']
                       if open_trade['dir'] == 'LONG'
                       else (open_trade['entry'] - row['Close']) / open_trade['entry'])
                open_trade['exit'] = row['Close']
                open_trade['pnl'] = pnl
                open_trade['reason'] = 'TIME'
                trades.append(open_trade)
                open_trade = None
                continue

        # No open trade — check for signal
        if open_trade is not None:
            continue

        # Score M1
        if version == 'v1':
            m1_dir, m1_score = score_m1(df_1h, idx_1h, cfg)
            details = {}
        else:
            m1_dir, m1_score, details = score_m1_v2(
                df_1h, idx_1h, cfg, df_15m=df_15m, idx_15m=idx_15m)

        # Entry filter: direction + minimum score
        min_score = cfg.get('M1_V2_MIN_SCORE', 0.60)
        if m1_dir == 'NEUTRAL' or m1_score < min_score:
            continue

        atr_1h = df_1h['atr'].iloc[idx_1h]
        if pd.isna(atr_1h) or atr_1h <= 0:
            continue

        entry = row['Close']
        sl_dist = sl_atr * atr_1h
        tp_dist = tp1_atr * atr_1h

        if m1_dir == 'BULLISH':
            direction = 'LONG'
            sl = entry - sl_dist
            tp1 = entry + tp_dist
        else:
            direction = 'SHORT'
            sl = entry + sl_dist
            tp1 = entry - tp_dist

        open_trade = {
            'entry_time': ts,
            'dir': direction,
            'entry': entry,
            'sl': sl,
            'tp1': tp1,
            'm1_dir': m1_dir,
            'm1_score': m1_score,
            'details': details,
            'bars': 0,
            'exit': None,
            'pnl': 0.0,
            'reason': None,
        }

    # Close remaining
    if open_trade is not None:
        last = df_15m.iloc[-1]
        pnl = ((last['Close'] - open_trade['entry']) / open_trade['entry']
               if open_trade['dir'] == 'LONG'
               else (open_trade['entry'] - last['Close']) / open_trade['entry'])
        open_trade['exit'] = last['Close']
        open_trade['pnl'] = pnl
        open_trade['reason'] = 'END'
        trades.append(open_trade)

    return trades


def print_comparison(v1_trades, v2_trades):
    """Print side-by-side comparison of v1 vs v2."""
    def stats(trades, label):
        if not trades:
            return {'label': label, 'total': 0, 'wr': 0, 'avg_pnl': 0,
                    'total_pnl': 0, 'pf': 0, 'avg_bars': 0, 'sl': 0,
                    'tp': 0, 'time': 0}

        total = len(trades)
        winners = [t for t in trades if t['pnl'] > 0]
        losers = [t for t in trades if t['pnl'] < 0]
        wr = len(winners) / total * 100 if total else 0
        avg_pnl = np.mean([t['pnl'] for t in trades])
        total_pnl = sum(t['pnl'] for t in trades)
        gross_profit = sum(t['pnl'] for t in winners)
        gross_loss = abs(sum(t['pnl'] for t in losers))
        pf = gross_profit / gross_loss if gross_loss > 0 else float('inf')
        avg_bars = np.mean([t['bars'] for t in trades])

        sl_count = sum(1 for t in trades if t['reason'] == 'SL')
        tp_count = sum(1 for t in trades if t['reason'] == 'TP1')
        time_count = sum(1 for t in trades if t['reason'] == 'TIME')

        return {
            'label': label, 'total': total, 'wr': wr, 'avg_pnl': avg_pnl,
            'total_pnl': total_pnl, 'pf': pf, 'avg_bars': avg_bars,
            'sl': sl_count, 'tp': tp_count, 'time': time_count,
        }

    s1 = stats(v1_trades, 'M1 v1 (original)')
    s2 = stats(v2_trades, 'M1 v2 (rewritten)')

    print("\n" + "=" * 72)
    print("  M1 MODULE COMPARISON — v1 (original) vs v2 (rewritten)")
    print("=" * 72)

    header = f"  {'Metric':<28} {'M1 v1':>18} {'M1 v2':>18}"
    print(header)
    print("  " + "-" * 64)

    rows = [
        ('Total Trades', f"{s1['total']}", f"{s2['total']}"),
        ('Win Rate', f"{s1['wr']:.1f}%", f"{s2['wr']:.1f}%"),
        ('Avg PnL per Trade', f"{s1['avg_pnl']*100:.3f}%", f"{s2['avg_pnl']*100:.3f}%"),
        ('Net PnL (sum)', f"{s1['total_pnl']*100:.2f}%", f"{s2['total_pnl']*100:.2f}%"),
        ('Profit Factor', f"{s1['pf']:.2f}" if s1['pf'] != float('inf') else "inf",
         f"{s2['pf']:.2f}" if s2['pf'] != float('inf') else "inf"),
        ('Avg Bars Held', f"{s1['avg_bars']:.1f}", f"{s2['avg_bars']:.1f}"),
        ('Exits: SL', f"{s1['sl']}", f"{s2['sl']}"),
        ('Exits: TP1', f"{s1['tp']}", f"{s2['tp']}"),
        ('Exits: Time', f"{s1['time']}", f"{s2['time']}"),
    ]

    for name, v1, v2 in rows:
        print(f"  {name:<28} {v1:>18} {v2:>18}")

    # Equity drawdown
    for trades, label in [(v1_trades, 'v1'), (v2_trades, 'v2')]:
        if trades:
            sorted_t = sorted(trades, key=lambda x: x['entry_time'])
            equity = np.cumsum([t['pnl'] for t in sorted_t])
            peak = np.maximum.accumulate(equity)
            dd = np.abs(equity - peak)
            max_dd = dd.max() * 100
            print(f"\n  {label} Max Drawdown: {max_dd:.2f}%")

    # Signal quality breakdown for v2
    if v2_trades:
        print(f"\n  {'─' * 64}")
        print(f"  M1 v2 — Signal Breakdown")
        print(f"  {'─' * 64}")

        div_trades = [t for t in v2_trades if t['details'].get('bull_div') or t['details'].get('bear_div')]
        no_div = [t for t in v2_trades if not t['details'].get('bull_div') and not t['details'].get('bear_div')]

        if div_trades:
            div_wr = sum(1 for t in div_trades if t['pnl'] > 0) / len(div_trades) * 100
            print(f"  With RSI divergence:  {len(div_trades)} trades, WR={div_wr:.1f}%")
        if no_div:
            no_div_wr = sum(1 for t in no_div if t['pnl'] > 0) / len(no_div) * 100
            print(f"  Without divergence:   {len(no_div)} trades, WR={no_div_wr:.1f}%")

        cross_trades = [t for t in v2_trades if t['details'].get('cross_up') or t['details'].get('cross_down')]
        if cross_trades:
            cross_wr = sum(1 for t in cross_trades if t['pnl'] > 0) / len(cross_trades) * 100
            print(f"  MACD crossover:       {len(cross_trades)} trades, WR={cross_wr:.1f}%")

        # Per-year breakdown
        years = sorted(set(t['entry_time'].year for t in v2_trades))
        if len(years) > 1:
            print(f"\n  M1 v2 — Per-Year Performance")
            print(f"  {'Year':<8} {'Trades':>8} {'WR%':>8} {'PnL%':>10} {'PF':>8}")
            print(f"  {'-' * 42}")
            for y in years:
                yt = [t for t in v2_trades if t['entry_time'].year == y]
                y_winners = sum(1 for t in yt if t['pnl'] > 0)
                y_wr = y_winners / len(yt) * 100 if yt else 0
                y_pnl = sum(t['pnl'] for t in yt) * 100
                y_gp = sum(t['pnl'] for t in yt if t['pnl'] > 0)
                y_gl = abs(sum(t['pnl'] for t in yt if t['pnl'] < 0))
                y_pf = y_gp / y_gl if y_gl > 0 else float('inf')
                pf_str = f"{y_pf:.2f}" if y_pf != float('inf') else "inf"
                print(f"  {y:<8} {len(yt):>8} {y_wr:>7.1f}% {y_pnl:>+9.2f}% {pf_str:>8}")

    print()


def main():
    parser = argparse.ArgumentParser(description='M1 Isolated Module Backtest')
    parser.add_argument('csv', help='Path to 15m CSV data')
    parser.add_argument('--start', help='Start date (YYYY-MM-DD)')
    parser.add_argument('--end', help='End date (YYYY-MM-DD)')
    parser.add_argument('--config', help='Config YAML path')
    parser.add_argument('--sl', type=float, default=1.3, help='SL ATR multiplier (default 1.3)')
    parser.add_argument('--tp', type=float, default=1.5, help='TP1 ATR multiplier (default 1.5)')
    parser.add_argument('--max-bars', type=int, default=32, help='Max bars to hold (default 32)')
    args = parser.parse_args()

    cfg = load_config(args.config) if args.config else load_config(None)

    print("Loading data...")
    df_15m = load_data(args.csv)
    print(f"  {len(df_15m):,} bars | {df_15m['Open time'].iloc[0]} → {df_15m['Open time'].iloc[-1]}")

    print("\nRunning M1 v1 (original)...")
    v1_trades = run_m1_backtest(df_15m, cfg, version='v1',
                                date_start=args.start, date_end=args.end,
                                sl_atr=args.sl, tp1_atr=args.tp, max_bars=args.max_bars)
    print(f"  {len(v1_trades)} trades generated")

    print("Running M1 v2 (rewritten)...")
    v2_trades = run_m1_backtest(df_15m, cfg, version='v2',
                                date_start=args.start, date_end=args.end,
                                sl_atr=args.sl, tp1_atr=args.tp, max_bars=args.max_bars)
    print(f"  {len(v2_trades)} trades generated")

    print_comparison(v1_trades, v2_trades)


if __name__ == '__main__':
    main()

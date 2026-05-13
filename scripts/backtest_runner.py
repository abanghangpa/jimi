#!/usr/bin/env python3
"""
JIMI Framework — Unified Backtest Runner
Replaces all backtest_*.py files with a single configurable entry point.

Usage:
    python scripts/backtest_runner.py data/processed/eth_15m_merged.csv
    python scripts/backtest_runner.py data/processed/eth_15m_merged.csv --start 2026-03-01 --end 2026-03-31
    python scripts/backtest_runner.py data/processed/eth_15m_merged.csv --config config/v615.yaml
    python scripts/backtest_runner.py data/processed/eth_15m_merged.csv --verbose
    python scripts/backtest_runner.py --fetch --start 2026-03-01 --end 2026-03-31
"""

import argparse
import sys
import os
import json
import numpy as np
import pandas as pd

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from src.config import load_config
from src.engine import run_backtest


def print_report(trades, stats):
    """Print comprehensive backtest performance report."""
    if not trades:
        print("\n  No trades generated.")
        return

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
    ret_dd = total_pnl / max_dd if max_dd > 0 else float('inf')

    longs = [t for t in trades if t.direction == 'LONG']
    shorts = [t for t in trades if t.direction == 'SHORT']
    long_wr = len([t for t in longs if t.pnl_pct > 0]) / len(longs) * 100 if longs else 0
    short_wr = len([t for t in shorts if t.pnl_pct > 0]) / len(shorts) * 100 if shorts else 0

    tp1_count = len([t for t in trades if t.tp1_hit])
    tp2_count = len([t for t in trades if t.tp2_hit])
    tp3_count = len([t for t in trades if t.exit_reason == 'TP3'])
    sl_count = len([t for t in trades if t.exit_reason == 'SL'])

    m4_pass = len([t for t in trades if t.m4_status == 'PASS'])
    m4_fail = len([t for t in trades if t.m4_status == 'FAIL'])
    m5_pass = len([t for t in trades if t.m5_status == 'PASS'])
    m5_fail = len([t for t in trades if t.m5_status == 'FAIL'])
    m5_avg = np.mean([t.m5_score for t in trades])

    print("\n" + "═" * 70)
    print("  JIMI — BACKTEST RESULTS (M1-M12 + Adaptive Direction)")
    print("═" * 70)

    print(f"\n  {'Total Trades:':<28} {total}")
    print(f"  {'Winners:':<28} {len(winners)} ({win_rate:.1f}%)")
    print(f"  {'Losers:':<28} {len(losers)}")
    print(f"\n  {'Net PnL (weighted):':<28} {total_pnl*100:.2f}%")
    print(f"  {'Avg Win:':<28} {avg_win*100:.2f}%")
    print(f"  {'Avg Loss:':<28} {avg_loss*100:.2f}%")
    print(f"  {'Profit Factor:':<28} {profit_factor:.2f}")
    print(f"  {'Max Drawdown:':<28} {max_dd*100:.2f}%")
    print(f"  {'Return/DD Ratio:':<28} {ret_dd:.1f}×")
    print(f"\n  {'Avg Bars Held:':<28} {np.mean([t.bars_held for t in trades]):.1f}")
    print(f"\n  Direction Breakdown:")
    print(f"    LONG:  {len(longs)} trades, WR {long_wr:.1f}%")
    print(f"    SHORT: {len(shorts)} trades, WR {short_wr:.1f}%")
    print(f"\n  M4 CVD Status:")
    print(f"    PASS: {m4_pass}  |  FAIL: {m4_fail}")
    print(f"\n  M5 Liquidation Magnet:")
    print(f"    PASS: {m5_pass}  |  FAIL: {m5_fail}  |  Avg score: {m5_avg:.3f}")
    print(f"\n  Exit Breakdown:")
    print(f"    TP1: {tp1_count} ({tp1_count/total*100:.1f}%)  TP2: {tp2_count} ({tp2_count/total*100:.1f}%)  TP3: {tp3_count} ({tp3_count/total*100:.1f}%)  SL: {sl_count} ({sl_count/total*100:.1f}%)")
    early_count = len([t for t in trades if t.exit_reason == 'EARLY_EXIT'])
    if early_count:
        print(f"    EARLY_EXIT: {early_count} ({early_count/total*100:.1f}%)")
    time_stop_count = len([t for t in trades if t.exit_reason == 'TIME_STOP'])
    if time_stop_count:
        print(f"    TIME_STOP: {time_stop_count} ({time_stop_count/total*100:.1f}%)")
    adaptive_count = len([t for t in trades if 'RR_' in str(t.exit_reason) or 'momentum_decay' in str(t.exit_reason) or 'opposing_signal' in str(t.exit_reason) or 'vol_expand' in str(t.exit_reason) or 'vol_compress' in str(t.exit_reason)])
    if adaptive_count:
        rr_exits = len([t for t in trades if 'RR_' in str(t.exit_reason)])
        mom_exits = len([t for t in trades if 'momentum' in str(t.exit_reason)])
        print(f"    ADAPTIVE: {adaptive_count} ({adaptive_count/total*100:.1f}%)  [R:R milestones={rr_exits}, momentum={mom_exits}]")

    # Squeeze stats
    squeeze_trades = [t for t in trades if t.squeeze_type != 'NONE']
    non_squeeze_trades = [t for t in trades if t.squeeze_type == 'NONE']
    if squeeze_trades:
        sq_winners = [t for t in squeeze_trades if t.pnl_pct > 0]
        sq_wr = len(sq_winners) / len(squeeze_trades) * 100
        sq_avg_pnl = np.mean([t.pnl_pct for t in squeeze_trades])
        sq_long = [t for t in squeeze_trades if t.direction == 'LONG']
        sq_short = [t for t in squeeze_trades if t.direction == 'SHORT']
        sq_fb = [t for t in squeeze_trades if t.squeeze_failed_breakout]
        print(f"\n  Squeeze Trade Performance:")
        print(f"    Total squeeze trades: {len(squeeze_trades)}")
        print(f"    Win rate: {sq_wr:.1f}%  |  Avg PnL: {sq_avg_pnl*100:.3f}%")
        print(f"    LONG: {len(sq_long)}  |  SHORT: {len(sq_short)}")
        print(f"    Failed breakout signals: {len(sq_fb)}")
        # Box type breakdown (v6)
        box_types = {}
        for t in squeeze_trades:
            bt = t.squeeze_box_type
            if bt not in box_types:
                box_types[bt] = {'count': 0, 'wins': 0, 'pnl': []}
            box_types[bt]['count'] += 1
            box_types[bt]['pnl'].append(t.pnl_pct)
            if t.pnl_pct > 0:
                box_types[bt]['wins'] += 1
        if len(box_types) > 1 or 'UNKNOWN' not in box_types:
            print(f"\n    Box Type Breakdown:")
            for bt, data in sorted(box_types.items(), key=lambda x: -x[1]['count']):
                bt_wr = data['wins'] / data['count'] * 100 if data['count'] > 0 else 0
                bt_pnl = np.mean(data['pnl']) * 100 if data['pnl'] else 0
                print(f"      {bt:<24} {data['count']:>4} trades  WR={bt_wr:.1f}%  AvgPnL={bt_pnl:+.3f}%")
        # Lifecycle breakdown (v6)
        lifecycles = {}
        for t in squeeze_trades:
            lc = t.squeeze_lifecycle
            if lc not in lifecycles:
                lifecycles[lc] = {'count': 0, 'wins': 0, 'pnl': []}
            lifecycles[lc]['count'] += 1
            lifecycles[lc]['pnl'].append(t.pnl_pct)
            if t.pnl_pct > 0:
                lifecycles[lc]['wins'] += 1
        if len(lifecycles) > 1 or 'NONE' not in lifecycles:
            print(f"\n    Lifecycle Breakdown:")
            for lc, data in sorted(lifecycles.items(), key=lambda x: -x[1]['count']):
                lc_wr = data['wins'] / data['count'] * 100 if data['count'] > 0 else 0
                lc_pnl = np.mean(data['pnl']) * 100 if data['pnl'] else 0
                print(f"      {lc:<12} {data['count']:>4} trades  WR={lc_wr:.1f}%  AvgPnL={lc_pnl:+.3f}%")
        if non_squeeze_trades:
            ns_wr = len([t for t in non_squeeze_trades if t.pnl_pct > 0]) / len(non_squeeze_trades) * 100
            ns_avg_pnl = np.mean([t.pnl_pct for t in non_squeeze_trades])
            print(f"\n  Non-squeeze trades: {len(non_squeeze_trades)}, WR={ns_wr:.1f}%, Avg PnL={ns_avg_pnl*100:.3f}%")
            print(f"    Squeeze vs Non-squeeze WR delta: {sq_wr - ns_wr:+.1f}%")

    print(f"\n  Signal Flow:")
    for k in ['signals_checked', 'm1_neutral_skip', 'm3_fail', 'm2_neutral_long_skip',
              'ics_blocked', 'ics_ceiling_skip', 'm4_required_skip', 'm4_false_anchored',
              'm5_pass', 'm5_fail', 'cascade_detected', 'dedup_skip', 'filter_blocked',
              'consec_pause', 'rolling_wr_skip', 'bias_gate_skip', 'adaptive_dir_block',
              'gate_block', 'gate_m7_block', 'gate_m10_block', 'gate_trend_block',
              'm2_veto_block', 'm5_hard_block', 'session_asian_block', 'post_crash_block',
              'veto_hard_block', 'veto_soft_applied', 'data_stale_block',
              'm9_block', 'm10_pass', 'm10_fail',
              'm11_pass', 'm11_fail', 'm11_skip',
              'm12_pass', 'm12_fail', 'm12_skip',
              'squeeze_detected', 'squeeze_triggered', 'squeeze_pending',
              'squeeze_failed_breakout', 'squeeze_entries',
              'entries']:
        if k in stats and stats[k] > 0:
            print(f"    {k+':':<26} {stats[k]}")

    # Monthly
    monthly = {}
    for t in trades:
        mk = t.entry_time.strftime('%Y-%m')
        if mk not in monthly:
            monthly[mk] = {'count': 0, 'pnl': 0, 'wins': 0}
        monthly[mk]['count'] += 1
        monthly[mk]['pnl'] += t.pnl_pct * t.size_pct
        if t.pnl_pct > 0:
            monthly[mk]['wins'] += 1

    print(f"\n  Monthly Performance:")
    print(f"    {'Month':<10} {'Trades':>7} {'WR':>7} {'PnL':>10}")
    print(f"    {'─'*10} {'─'*7} {'─'*7} {'─'*10}")
    for month in sorted(monthly.keys()):
        m = monthly[month]
        wr = m['wins'] / m['count'] * 100 if m['count'] > 0 else 0
        print(f"    {month:<10} {m['count']:>7} {wr:>6.1f}% {m['pnl']*100:>9.2f}%")
    print("═" * 70)

    return {'total_trades': total, 'win_rate': win_rate, 'total_pnl': total_pnl,
            'profit_factor': profit_factor, 'max_drawdown': max_dd, 'return_dd_ratio': ret_dd,
            'squeeze_trades': len(squeeze_trades), 'squeeze_wr': sq_wr if squeeze_trades else 0,
            'squeeze_avg_pnl': sq_avg_pnl if squeeze_trades else 0}


def export_trades(trades, filepath):
    """Export trade log to CSV."""
    rows = []
    for t in trades:
        rows.append({
            'entry_time': t.entry_time, 'exit_time': t.exit_time, 'direction': t.direction,
            'entry_price': t.entry_price, 'exit_price': t.exit_price,
            'sl': t.sl, 'tp1': t.tp1, 'tp2': t.tp2, 'tp3': t.tp3,
            'pnl_pct': t.pnl_pct * 100, 'size_pct': t.size_pct, 'bars_held': t.bars_held,
            'ics': t.ics, 'm1_dir': t.m1_dir, 'm2_status': t.m2_status,
            'm4_status': t.m4_status, 'm5_status': t.m5_status, 'm5_score': t.m5_score,
            'phase0': t.phase0, 'exit_reason': t.exit_reason, 'reason': t.reason,
        })
    pd.DataFrame(rows).to_csv(filepath, index=False)
    print(f"\n  Trade log exported: {filepath}")


def export_forensic(trades, filepath):
    """Export full module state snapshot for every trade — forensic audit."""
    rows = []
    for t in trades:
        eth_btc_trend = t.m7_details.get('eth_btc_trend', 'N/A') if isinstance(t.m7_details, dict) else 'N/A'
        btc_trend = t.m7_details.get('btc_trend', 'N/A') if isinstance(t.m7_details, dict) else 'N/A'
        btc_atr_pctl = t.m7_details.get('btc_atr_pctl', None) if isinstance(t.m7_details, dict) else None
        m7_composite = t.m7_details.get('m7_score', None) if isinstance(t.m7_details, dict) else None

        rows.append({
            'entry_time': t.entry_time, 'exit_time': t.exit_time,
            'direction': t.direction, 'exit_reason': t.exit_reason,
            'pnl_pct': round(t.pnl_pct * 100, 4), 'size_pct': t.size_pct,
            'bars_held': t.bars_held, 'result': 'WIN' if t.pnl_pct > 0 else ('LOSS' if t.pnl_pct < 0 else 'FLAT'),
            'ics': round(t.ics, 4), 'gatekeeper_passed': t.gatekeeper_passed,
            'veto_soft_penalty': t.veto_soft_penalty,
            'm1_dir': t.m1_dir, 'm1_score': round(t.m1_score, 4),
            'm2_status': t.m2_status, 'm2_score': round(t.m2_score, 4),
            'm3_score': round(t.m3_score, 4),
            'm4_status': t.m4_status, 'm4_score': round(t.m4_score, 4),
            'm5_status': t.m5_status, 'm5_score': round(t.m5_score, 4),
            'm7_score': round(t.m7_score, 4), 'm7_composite': m7_composite,
            'eth_btc_trend': eth_btc_trend, 'btc_trend': btc_trend,
            'btc_atr_pctl': btc_atr_pctl,
            'm8_status': t.m8_status, 'm8_score': round(t.m8_score, 4),
            'm9_status': t.m9_status, 'm9_score': round(t.m9_score, 4),
            'vol_regime': t.vol_regime,
            'm10_status': t.m10_status, 'm10_score': round(t.m10_score, 4),
            'm11_status': t.m11_status, 'm11_score': round(t.m11_score, 4),
            'm12_status': t.m12_status, 'm12_score': round(t.m12_score, 4),
            'trend_dir': t.trend_dir, 'trend_val': round(t.trend_val, 4),
            'cross_asset_score': round(t.cross_asset_score, 4),
            'session': t.session_name, 'phase0': t.phase0,
            'squeeze_type': t.squeeze_type, 'squeeze_score': round(t.squeeze_score, 4),
            'squeeze_strong': t.squeeze_strong, 'squeeze_trigger': t.squeeze_trigger_type,
            'squeeze_failed_breakout': t.squeeze_failed_breakout,
            'squeeze_box_type': t.squeeze_box_type, 'squeeze_lifecycle': t.squeeze_lifecycle,
        })
    df = pd.DataFrame(rows)
    df.to_csv(filepath, index=False)
    print(f"  Forensic log exported: {filepath} ({len(df)} trades)")
    return df


def fetch_and_save(symbol, timeframe, start, end, output_path):
    """Fetch data from exchange and save to CSV."""
    import ccxt
    exchange = ccxt.binance({'enableRateLimit': True})
    since_ms = int(pd.Timestamp(start).timestamp() * 1000)
    until_ms = int(pd.Timestamp(end).timestamp() * 1000)

    all_candles = []
    current = since_ms
    while current < until_ms:
        raw = exchange.fetch_ohlcv(symbol, timeframe, since=current, limit=1000)
        if not raw:
            break
        for c in raw:
            ts = int(c[0])
            if ts >= until_ms:
                break
            all_candles.append({
                'Open time': pd.to_datetime(ts, unit='ms'),
                'Open': float(c[1]), 'High': float(c[2]),
                'Low': float(c[3]), 'Close': float(c[4]),
                'Volume': float(c[5]),
                'Close time': pd.to_datetime(int(c[6]), unit='ms') if len(c) > 6 else pd.to_datetime(ts + 900000, unit='ms'),
                'Quote asset volume': float(c[7]) if len(c) > 7 else 0,
                'Number of trades': int(c[8]) if len(c) > 8 else 0,
                'Taker buy base asset volume': float(c[9]) if len(c) > 9 else 0,
                'Taker buy quote asset volume': float(c[10]) if len(c) > 10 else 0,
            })
        last_ts = raw[-1][0]
        if last_ts <= current:
            break
        current = last_ts + 1

    df = pd.DataFrame(all_candles)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_csv(output_path, index=False)
    print(f"  Fetched {len(df)} bars → {output_path}")
    return output_path


def main():
    parser = argparse.ArgumentParser(description='JIMI Backtest Runner')
    parser.add_argument('csv', nargs='?', help='Path to 15m OHLCV CSV')
    parser.add_argument('--start', help='Start date (YYYY-MM-DD)')
    parser.add_argument('--end', help='End date (YYYY-MM-DD)')
    parser.add_argument('--config', help='Config YAML path')
    parser.add_argument('--verbose', '-v', action='store_true')
    parser.add_argument('--export', default='jimi_trades.csv', help='Trade log output path')
    parser.add_argument('--forensic', help='Forensic audit output path')
    parser.add_argument('--fetch', action='store_true', help='Fetch data from exchange first')
    parser.add_argument('--symbol', default='ETH/USDT', help='Trading pair')
    parser.add_argument('--timeframe', default='15m', help='Timeframe')
    args = parser.parse_args()

    cfg = load_config(args.config) if args.config else None

    csv_path = args.csv
    if args.fetch:
        if not args.start or not args.end:
            print("ERROR: --fetch requires --start and --end")
            sys.exit(1)
        csv_path = csv_path or f"data/raw/eth_15m_{args.start}_{args.end}.csv"
        fetch_and_save(args.symbol, args.timeframe, args.start, args.end, csv_path)

    if not csv_path:
        print("ERROR: Provide a CSV path or use --fetch")
        parser.print_help()
        sys.exit(1)

    if not os.path.exists(csv_path):
        print(f"ERROR: File not found: {csv_path}")
        sys.exit(1)

    trades, stats, df = run_backtest(csv_path, config=cfg, verbose=args.verbose,
                                      date_start=args.start, date_end=args.end)

    # Always print signal flow stats (even when no trades)
    print("\n  Signal Flow (debug):")
    for k in ['signals_checked', 'bias_gate_skip', 'adaptive_dir_block',
              'm3_fail', 'm2_neutral_long_skip', 'm2_veto_block',
              'ics_blocked', 'ics_ceiling_skip', 'gate_block',
              'gate_m7_block', 'gate_m10_block', 'gate_trend_block',
              'm9_block', 'm10_fail', 'm11_fail', 'post_crash_block',
              'veto_hard_block', 'trend_flip', 'trend_weak',
              'm4_false_anchored', 'm5_hard_block',
              'squeeze_detected', 'squeeze_triggered', 'squeeze_pending',
              'squeeze_failed_breakout', 'squeeze_entries',
              'entries']:
        if k in stats and stats[k] > 0:
            print(f"    {k+':':<26} {stats[k]}")

    result = print_report(trades, stats)

    if trades:
        export_trades(trades, args.export)
    if args.forensic and trades:
        export_forensic(trades, args.forensic)

    print("\nDone.")


if __name__ == '__main__':
    main()

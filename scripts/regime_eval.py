"""
Regime Detection Performance Evaluator

Runs controlled backtests to measure whether M9 regime detection
actually improves trading outcomes.

Usage:
    python scripts/regime_eval.py eth_15m_merged.csv
    python scripts/regime_eval.py eth_15m_merged.csv --days 10
    python scripts/regime_eval.py eth_15m_merged.csv --start 2026-01-01 --end 2026-01-11
    python scripts/regime_eval.py eth_15m_merged.csv --perms 1000
"""

import sys
import os
import argparse

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from src.config import load_config
from src.engine import run_backtest
from src.modules.regime_eval import print_full_report


def run_m9_backtest(csv_path, config, start=None, end=None, verbose=False):
    """Run backtest with M9 enabled (default config)."""
    print("\n" + "=" * 70)
    print("  BACKTEST A: M9 ON (regime detection enabled)")
    print("=" * 70)
    trades, stats, df = run_backtest(
        csv_path, config=config, verbose=verbose,
        date_start=start, date_end=end)
    return trades, stats


def run_no_m9_backtest(csv_path, config, start=None, end=None, verbose=False):
    """Run backtest with M9 disabled (no regime filtering)."""
    print("\n" + "=" * 70)
    print("  BACKTEST B: M9 OFF (no regime detection)")
    print("=" * 70)
    cfg_no_m9 = dict(config)
    cfg_no_m9['M9_ENABLED'] = False
    # Also disable M9-related blocks
    cfg_no_m9['VETO_CRISIS_HARD'] = False
    cfg_no_m9['VETO_CHOP_HARD'] = False
    # Keep everything else the same
    trades, stats, df = run_backtest(
        csv_path, config=cfg_no_m9, verbose=verbose,
        date_start=start, date_end=end)
    return trades, stats


def main():
    parser = argparse.ArgumentParser(
        description='JIMI — Regime Detection Performance Evaluator')
    parser.add_argument('csv', help='Path to OHLCV CSV')
    parser.add_argument('--days', type=int, default=10,
                        help='Number of days to backtest (default: 10)')
    parser.add_argument('--start', help='Start date (overrides --days)')
    parser.add_argument('--end', help='End date (overrides --days)')
    parser.add_argument('--perms', type=int, default=500,
                        help='Permutation test iterations (default: 500)')
    parser.add_argument('--config', help='Config YAML path')
    parser.add_argument('--verbose', action='store_true',
                        help='Verbose trade output')
    parser.add_argument('--skip-ab', action='store_true',
                        help='Skip M9 OFF backtest (faster)')
    args = parser.parse_args()

    # Load config
    cfg_path = args.config or os.path.join(
        os.path.dirname(os.path.dirname(__file__)), 'config', 'settings.yaml')
    config = load_config(cfg_path, validate=False)

    # Determine date range
    start = args.start
    end = args.end
    if not start and not end:
        # Use last N days from the CSV
        import pandas as pd
        df = pd.read_csv(args.csv, nrows=5)
        # Find the Open time column
        time_col = 'Open time' if 'Open time' in df.columns else df.columns[0]
        df_full = pd.read_csv(args.csv, usecols=[time_col])
        last_date = pd.to_datetime(df_full[time_col].iloc[-1])
        from datetime import timedelta
        start_date = last_date - timedelta(days=args.days)
        start = str(start_date)
        end = str(last_date)
        print(f"  Date range: {start} → {end} ({args.days} days)")

    # Run backtest A: M9 ON
    trades_m9, stats_m9 = run_m9_backtest(
        args.csv, config, start=start, end=end, verbose=args.verbose)

    # Run backtest B: M9 OFF (unless skipped)
    trades_no_m9 = None
    if not args.skip_ab:
        trades_no_m9, stats_no_m9 = run_no_m9_backtest(
            args.csv, config, start=start, end=end, verbose=args.verbose)

    # Print full evaluation report
    print_full_report(
        trades_m9,
        trades_no_m9=trades_no_m9,
        n_permutations=args.perms)

    # Summary stats
    print(f"  M9 ON:  {len(trades_m9)} trades, "
          f"{sum(1 for t in trades_m9 if t.pnl_pct > 0)}/{len(trades_m9)} wins, "
          f"total PnL: {sum(t.pnl_pct * t.size_pct for t in trades_m9)*100:+.2f}%")
    if trades_no_m9:
        print(f"  M9 OFF: {len(trades_no_m9)} trades, "
              f"{sum(1 for t in trades_no_m9 if t.pnl_pct > 0)}/{len(trades_no_m9)} wins, "
              f"total PnL: {sum(t.pnl_pct * t.size_pct for t in trades_no_m9)*100:+.2f}%")
    print()


if __name__ == '__main__':
    main()

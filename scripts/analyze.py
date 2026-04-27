#!/usr/bin/env python3
"""
JIMI Framework — Result Analyzer
Post-process backtest results: forensic analysis, monthly breakdown, module correlation.

Usage:
    python scripts/analyze.py jimi_trades.csv
    python scripts/analyze.py jimi_trades.csv --forensic
"""

import argparse
import sys
import os
import numpy as np
import pandas as pd


def forensic_analysis(trades_df):
    """Analyze module states correlated with wins vs losses."""
    winners = trades_df[trades_df['pnl_pct'] > 0]
    losers = trades_df[trades_df['pnl_pct'] < 0]

    if winners.empty or losers.empty:
        print("  Need both winners and losers for forensic analysis.")
        return

    print(f"\n{'='*80}")
    print(f"  FORENSIC MODULE ANALYSIS — {len(winners)} Winners vs {len(losers)} Losers")
    print(f"{'='*80}")

    modules = [
        ('m1_score', 'M1_score'),
        ('m2_score', 'M2_score'),
        ('m3_score', 'M3_score'),
        ('m4_score', 'M4_score'),
        ('m5_score', 'M5_score'),
        ('m7_score', 'M7_score'),
        ('m8_score', 'M8_score'),
        ('m9_score', 'M9_score'),
        ('m10_score', 'M10_score'),
        ('m11_score', 'M11_score'),
        ('m12_score', 'M12_score'),
        ('cross_asset_score', 'cross_asset'),
        ('ics', 'ICS'),
    ]

    print(f"\n  {'Module':>14} {'Win Avg':>10} {'Loss Avg':>10} {'Delta':>10} {'Verdict':>12}")
    print(f"  {'-'*56}")

    hallucinating = []
    predictive = []

    for col, name in modules:
        if col not in trades_df.columns:
            continue
        w_avg = winners[col].mean()
        l_avg = losers[col].mean()
        delta = w_avg - l_avg

        if delta < -0.05:
            verdict = '⚠ HALLUCINATING'
            hallucinating.append((name, delta, w_avg, l_avg))
        elif delta > 0.05:
            verdict = '✓ PREDICTIVE'
            predictive.append((name, delta, w_avg, l_avg))
        else:
            verdict = '~ NEUTRAL'

        print(f"  {name:>14} {w_avg:>10.4f} {l_avg:>10.4f} {delta:>+10.4f} {verdict:>12}")

    if 'vol_regime' in trades_df.columns:
        print(f"\n  Regime Distribution:")
        for regime in ['TRENDING', 'CHOP', 'COMPRESSING', 'NEUTRAL', 'CRISIS']:
            w_count = len(winners[winners['vol_regime'] == regime])
            l_count = len(losers[losers['vol_regime'] == regime])
            total = w_count + l_count
            if total > 0:
                wr = w_count / total * 100
                print(f"    {regime:>12}: {total} trades, WR {wr:.0f}% ({w_count}W/{l_count}L)")

    if 'session' in trades_df.columns:
        print(f"\n  Session Distribution:")
        for session in ['ASIAN', 'EU', 'US', 'US_OPEN', 'LATE_US']:
            w_count = len(winners[winners['session'] == session])
            l_count = len(losers[losers['session'] == session])
            total = w_count + l_count
            if total > 0:
                wr = w_count / total * 100
                print(f"    {session:>12}: {total} trades, WR {wr:.0f}% ({w_count}W/{l_count}L)")

    print(f"\n  Direction Analysis:")
    for direction in ['LONG', 'SHORT']:
        w_count = len(winners[winners['direction'] == direction])
        l_count = len(losers[losers['direction'] == direction])
        total = w_count + l_count
        if total > 0:
            wr = w_count / total * 100
            dir_trades = trades_df[trades_df['direction'] == direction]
            avg_pnl = (dir_trades['pnl_pct'] * dir_trades['size_pct']).mean()
            print(f"    {direction:>12}: {total} trades, WR {wr:.0f}%, Avg PnL {avg_pnl:+.2f}%")

    if hallucinating:
        print(f"\n  ⚠ HALLUCINATING MODULES (higher scores in losers):")
        for name, delta, w_avg, l_avg in hallucinating:
            print(f"    {name}: winners avg {w_avg:.4f}, losers avg {l_avg:.4f} (delta {delta:+.4f})")
        print(f"    → Consider reducing weight or making these veto-only.")

    if predictive:
        print(f"\n  ✓ PREDICTIVE MODULES (higher scores in winners):")
        for name, delta, w_avg, l_avg in predictive:
            print(f"    {name}: winners avg {w_avg:.4f}, losers avg {l_avg:.4f} (delta {delta:+.4f})")
        print(f"    → These modules are working. Consider increasing weight.")

    print(f"\n{'='*80}")


def main():
    parser = argparse.ArgumentParser(description='JIMI Result Analyzer')
    parser.add_argument('trades_csv', help='Path to trades CSV')
    parser.add_argument('--forensic', action='store_true', help='Run forensic analysis')
    args = parser.parse_args()

    if not os.path.exists(args.trades_csv):
        print(f"ERROR: File not found: {args.trades_csv}")
        sys.exit(1)

    df = pd.read_csv(args.trades_csv)
    print(f"\n  Loaded {len(df)} trades from {args.trades_csv}")

    total = len(df)
    winners = df[df['pnl_pct'] > 0]
    losers = df[df['pnl_pct'] < 0]
    wr = len(winners) / total * 100 if total > 0 else 0
    total_pnl = (df['pnl_pct'] * df['size_pct']).sum()

    print(f"\n  Summary:")
    print(f"    Total trades: {total}")
    print(f"    Win rate: {wr:.1f}% ({len(winners)}W / {len(losers)}L)")
    print(f"    Total PnL: {total_pnl:.2f}%")

    if args.forensic:
        forensic_analysis(df)
    else:
        # Basic monthly breakdown
        if 'entry_time' in df.columns:
            df['entry_time'] = pd.to_datetime(df['entry_time'])
            df['month'] = df['entry_time'].dt.strftime('%Y-%m')
            print(f"\n  Monthly:")
            for month in sorted(df['month'].unique()):
                m = df[df['month'] == month]
                m_wr = len(m[m['pnl_pct'] > 0]) / len(m) * 100 if len(m) > 0 else 0
                m_pnl = (m['pnl_pct'] * m['size_pct']).sum()
                print(f"    {month}: {len(m)} trades, WR {m_wr:.1f}%, PnL {m_pnl:.2f}%")

    print("\nDone.")


if __name__ == '__main__':
    main()

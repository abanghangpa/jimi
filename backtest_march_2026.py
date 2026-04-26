#!/usr/bin/env python3
"""Backtest JIMI v6.16 BEAST on March 2026 — daily breakdown."""

import sys, os
import pandas as pd
import numpy as np
from datetime import datetime

sys.path.insert(0, os.path.dirname(__file__))

from jimi_v616_beast import run_backtest, print_report, export_forensic, forensic_analysis

CSV_PATH = os.path.join(os.path.dirname(__file__), 'eth_15m_6m.csv')
DATE_START = '2026-03-01'
DATE_END = '2026-03-31'

print("=" * 80)
print("  JIMI v6.16 BEAST — March 2026 Daily Backtest")
print("=" * 80)

# Run the full backtest for March 2026
trades, stats, df_bt = run_backtest(CSV_PATH, verbose=False, date_start=DATE_START, date_end=DATE_END)

# Print standard report
print_report(trades, stats)

# ── Forensic Analysis ──
if trades:
    forensic_analysis(trades)
    export_forensic(trades, '/tmp/jimi_forensic_march_2026.csv')

# ── Daily Breakdown ──
if trades:
    print(f"\n\n{'='*100}")
    print(f"  DAILY BREAKDOWN — March 2026")
    print(f"{'='*100}")

    # Build daily trade table
    trade_dates = []
    for t in trades:
        entry_date = pd.Timestamp(t.entry_time).date() if hasattr(t, 'entry_time') else None
        exit_date = pd.Timestamp(t.exit_time).date() if hasattr(t, 'exit_time') else None
        pnl = t.pnl_pct * t.size_pct
        trade_dates.append({
            'entry_date': entry_date,
            'exit_date': exit_date,
            'direction': t.direction,
            'entry_price': t.entry_price,
            'exit_price': t.exit_price,
            'pnl_pct': pnl,
            'exit_reason': t.exit_reason,
            'ics': t.ics,
        })

    df_trades = pd.DataFrame(trade_dates)

    # Group by exit date (day the PnL is realized)
    daily_data = []
    for date in sorted(df_trades['exit_date'].unique()):
        day_trades = df_trades[df_trades['exit_date'] == date]
        wins = len(day_trades[day_trades['pnl_pct'] > 0])
        losses = len(day_trades[day_trades['pnl_pct'] < 0])
        total = len(day_trades)
        day_pnl = day_trades['pnl_pct'].sum()
        longs = len(day_trades[day_trades['direction'] == 'LONG'])
        shorts = len(day_trades[day_trades['direction'] == 'SHORT'])
        sl_count = len(day_trades[day_trades['exit_reason'] == 'SL'])
        tp_count = len(day_trades[day_trades['exit_reason'].str.startswith('TP')])
        avg_ics = day_trades['ics'].mean()

        daily_data.append({
            'date': date,
            'trades': total,
            'wins': wins,
            'losses': losses,
            'wr': wins / total * 100 if total > 0 else 0,
            'pnl': day_pnl * 100,
            'longs': longs,
            'shorts': shorts,
            'sl': sl_count,
            'tp': tp_count,
            'avg_ics': avg_ics,
        })

    df_daily = pd.DataFrame(daily_data)

    # Print daily table
    hdr = f"{'Date':>12} {'Trades':>7} {'W':>4} {'L':>4} {'WR%':>7} {'PnL%':>10} {'Long':>5} {'Short':>6} {'SL':>4} {'TP':>4} {'AvgICS':>8}"
    print(hdr)
    print("-" * 100)

    cumulative_pnl = 0
    for _, row in df_daily.iterrows():
        cumulative_pnl += row['pnl']
        wr_str = f"{row['wr']:.0f}%"
        print(f"{str(row['date']):>12} {row['trades']:>7} {row['wins']:>4} {row['losses']:>4} {wr_str:>7} "
              f"{row['pnl']:>+9.2f}% {row['longs']:>5} {row['shorts']:>6} {row['sl']:>4} {row['tp']:>4} {row['avg_ics']:>8.3f}")

    print("-" * 100)

    # Summary
    total_trades = df_daily['trades'].sum()
    total_wins = df_daily['wins'].sum()
    total_losses = df_daily['losses'].sum()
    agg_wr = total_wins / total_trades * 100 if total_trades > 0 else 0
    total_pnl = df_daily['pnl'].sum()
    best_day = df_daily.loc[df_daily['pnl'].idxmax()]
    worst_day = df_daily.loc[df_daily['pnl'].idxmin()]
    profitable_days = len(df_daily[df_daily['pnl'] > 0])
    losing_days = len(df_daily[df_daily['pnl'] < 0])
    flat_days = len(df_daily[df_daily['pnl'] == 0])

    # Equity curve for max DD
    equity = [0]
    for t in sorted(trades, key=lambda x: x.exit_time):
        equity.append(equity[-1] + t.pnl_pct * t.size_pct)
    equity = np.array(equity)
    peak = np.maximum.accumulate(equity)
    max_dd = abs((equity - peak).min()) if len(equity) > 0 else 0

    gross_profit = sum(t.pnl_pct * t.size_pct for t in trades if t.pnl_pct > 0)
    gross_loss = abs(sum(t.pnl_pct * t.size_pct for t in trades if t.pnl_pct < 0))
    pf = gross_profit / gross_loss if gross_loss > 0 else float('inf')
    pf_str = f"{pf:.2f}" if pf != float('inf') else "∞"

    print(f"\n  Total trades:    {total_trades}")
    print(f"  Win/Loss:        {total_wins}W / {total_losses}L ({agg_wr:.1f}% WR)")
    print(f"  Total PnL:       {total_pnl:+.2f}%")
    print(f"  Profit Factor:   {pf_str}")
    print(f"  Max Drawdown:    {max_dd*100:.2f}%")
    print(f"  Trading days:    {len(df_daily)}")
    print(f"  Profitable days: {profitable_days} | Losing days: {losing_days} | Flat: {flat_days}")
    print(f"  Best day:        {best_day['date']} ({best_day['pnl']:+.2f}%, {best_day['trades']} trades)")
    print(f"  Worst day:       {worst_day['date']} ({worst_day['pnl']:+.2f}%, {worst_day['trades']} trades)")
    print(f"  Avg PnL/trade:   {total_pnl/total_trades:+.3f}%" if total_trades > 0 else "")

    # Exit reason breakdown
    print(f"\n  Exit Reason Breakdown:")
    for reason in ['SL', 'TP1', 'TP2', 'TP3', 'EARLY_EXIT', 'END']:
        count = len([t for t in trades if t.exit_reason == reason])
        if count > 0:
            print(f"    {reason:>10}: {count} ({count/total_trades*100:.1f}%)")

    # Direction breakdown
    longs = [t for t in trades if t.direction == 'LONG']
    shorts = [t for t in trades if t.direction == 'SHORT']
    long_pnl = sum(t.pnl_pct * t.size_pct for t in longs) * 100
    short_pnl = sum(t.pnl_pct * t.size_pct for t in shorts) * 100
    long_wr = len([t for t in longs if t.pnl_pct > 0]) / len(longs) * 100 if longs else 0
    short_wr = len([t for t in shorts if t.pnl_pct > 0]) / len(shorts) * 100 if shorts else 0
    print(f"\n  Direction Breakdown:")
    print(f"    LONG:  {len(longs)} trades, WR {long_wr:.1f}%, PnL {long_pnl:+.2f}%")
    print(f"    SHORT: {len(shorts)} trades, WR {short_wr:.1f}%, PnL {short_pnl:+.2f}%")

    # Weekly summary
    df_daily['week'] = pd.to_datetime(df_daily['date']).dt.isocalendar().week
    print(f"\n  Weekly Summary:")
    for week, grp in df_daily.groupby('week'):
        w_pnl = grp['pnl'].sum()
        w_trades = grp['trades'].sum()
        w_wins = grp['wins'].sum()
        w_wr = w_wins / w_trades * 100 if w_trades > 0 else 0
        print(f"    Week {week}: {w_trades} trades, WR {w_wr:.0f}%, PnL {w_pnl:+.2f}%")

else:
    print("\n  ⚠ No trades generated for March 2026.")

print(f"\n{'='*80}")
print("  Done.")
print(f"{'='*80}")

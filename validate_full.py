#!/usr/bin/env python3
"""Backtest JIMI v6.16 BEAST on full 6-month dataset — monthly breakdown."""

import sys, os
import pandas as pd
import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
from jimi_v616_beast import run_backtest, print_report, export_forensic, forensic_analysis

CSV_PATH = os.path.join(os.path.dirname(__file__), 'eth_15m_6m.csv')

print("=" * 80)
print("  JIMI v6.16 BEAST — Full 6-Month Validation (Oct 2025 – Mar 2026)")
print("=" * 80)

trades, stats, df_bt = run_backtest(CSV_PATH, verbose=False)

print_report(trades, stats)

if trades:
    forensic_analysis(trades)
    export_forensic(trades, '/tmp/jimi_forensic_full.csv')

    # ── Monthly Breakdown ──
    print(f"\n\n{'='*100}")
    print(f"  MONTHLY BREAKDOWN")
    print(f"{'='*100}")

    trade_data = []
    for t in trades:
        entry_date = pd.Timestamp(t.entry_time)
        pnl = t.pnl_pct * t.size_pct
        trade_data.append({
            'month': entry_date.strftime('%Y-%m'),
            'direction': t.direction,
            'pnl': pnl,
            'exit_reason': t.exit_reason,
            'ics': t.ics,
            'vol_regime': t.vol_regime,
            'session': t.session_name,
        })

    df_trades = pd.DataFrame(trade_data)

    hdr = f"{'Month':>10} {'Trades':>7} {'W':>4} {'L':>4} {'WR%':>7} {'PnL%':>10} {'PF':>7} {'Long':>5} {'Short':>6} {'SL':>4} {'TP':>4} {'MaxDD':>8}"
    print(hdr)
    print("-" * 100)

    all_monthly_dd = []
    for month, grp in df_trades.groupby('month'):
        wins = len(grp[grp['pnl'] > 0])
        losses = len(grp[grp['pnl'] < 0])
        total = len(grp)
        wr = wins / total * 100 if total > 0 else 0
        month_pnl = grp['pnl'].sum() * 100
        longs = len(grp[grp['direction'] == 'LONG'])
        shorts = len(grp[grp['direction'] == 'SHORT'])
        sl = len(grp[grp['exit_reason'] == 'SL'])
        tp = len(grp[grp['exit_reason'].str.startswith('TP')])
        gross_profit = grp[grp['pnl'] > 0]['pnl'].sum()
        gross_loss = abs(grp[grp['pnl'] < 0]['pnl'].sum())
        pf = gross_profit / gross_loss if gross_loss > 0 else float('inf')
        pf_str = f"{pf:.2f}" if pf != float('inf') else "∞"

        # Monthly max DD
        equity = [0]
        for _, row in grp.iterrows():
            equity.append(equity[-1] + row['pnl'])
        equity = np.array(equity)
        peak = np.maximum.accumulate(equity)
        max_dd = abs((equity - peak).min()) * 100 if len(equity) > 0 else 0
        all_monthly_dd.append(max_dd)

        print(f"{month:>10} {total:>7} {wins:>4} {losses:>4} {wr:>6.0f}% {month_pnl:>+9.2f}% {pf_str:>7} {longs:>5} {shorts:>6} {sl:>4} {tp:>4} {max_dd:>7.2f}%")

    print("-" * 100)

    # ── Grand Summary ──
    total_trades = len(trades)
    total_wins = len([t for t in trades if t.pnl_pct > 0])
    total_losses = len([t for t in trades if t.pnl_pct < 0])
    agg_wr = total_wins / total_trades * 100
    total_pnl = sum(t.pnl_pct * t.size_pct for t in trades) * 100
    gross_profit = sum(t.pnl_pct * t.size_pct for t in trades if t.pnl_pct > 0)
    gross_loss = abs(sum(t.pnl_pct * t.size_pct for t in trades if t.pnl_pct < 0))
    pf = gross_profit / gross_loss if gross_loss > 0 else float('inf')
    pf_str = f"{pf:.2f}" if pf != float('inf') else "∞"

    equity = [0]
    for t in sorted(trades, key=lambda x: x.exit_time):
        equity.append(equity[-1] + t.pnl_pct * t.size_pct)
    equity = np.array(equity)
    peak = np.maximum.accumulate(equity)
    max_dd = abs((equity - peak).min()) * 100

    monthly_pnls = [df_trades[df_trades['month'] == m]['pnl'].sum() * 100 for m in df_trades['month'].unique()]
    profitable_months = len([p for p in monthly_pnls if p > 0])
    total_months = len(monthly_pnls)

    print(f"\n  GRAND SUMMARY")
    print(f"  Total trades:      {total_trades}")
    print(f"  Win/Loss:          {total_wins}W / {total_losses}L ({agg_wr:.1f}% WR)")
    print(f"  Total PnL:         {total_pnl:+.2f}%")
    print(f"  Profit Factor:     {pf_str}")
    print(f"  Max Drawdown:      {max_dd:.2f}%")
    print(f"  Return/DD:         {total_pnl/max_dd:.1f}×" if max_dd > 0 else "  Return/DD: ∞")
    print(f"  Profitable months: {profitable_months}/{total_months}")
    print(f"  Avg monthly PnL:   {np.mean(monthly_pnls):+.2f}%")
    print(f"  Best month:        {max(monthly_pnls):+.2f}%")
    print(f"  Worst month:       {min(monthly_pnls):+.2f}%")
    print(f"  Monthly Sharpe:    {np.mean(monthly_pnls)/np.std(monthly_pnls):.2f}" if np.std(monthly_pnls) > 0 else "")

    # ── Direction breakdown ──
    longs = [t for t in trades if t.direction == 'LONG']
    shorts = [t for t in trades if t.direction == 'SHORT']
    print(f"\n  LONG:  {len(longs)} trades, WR {len([t for t in longs if t.pnl_pct>0])/max(len(longs),1)*100:.0f}%")
    print(f"  SHORT: {len(shorts)} trades, WR {len([t for t in shorts if t.pnl_pct>0])/max(len(shorts),1)*100:.0f}%")

    # ── Regime breakdown ──
    print(f"\n  Regime Performance:")
    for regime in ['TRENDING', 'CHOP', 'COMPRESSING', 'NEUTRAL', 'CRISIS']:
        r_trades = [t for t in trades if t.vol_regime == regime]
        if r_trades:
            r_wins = len([t for t in r_trades if t.pnl_pct > 0])
            r_wr = r_wins / len(r_trades) * 100
            r_pnl = sum(t.pnl_pct * t.size_pct for t in r_trades) * 100
            print(f"    {regime:>12}: {len(r_trades)} trades, WR {r_wr:.0f}%, PnL {r_pnl:+.2f}%")

    # ── Session breakdown ──
    print(f"\n  Session Performance:")
    for session in ['ASIAN', 'EU', 'US_OPEN', 'US', 'LATE_US']:
        s_trades = [t for t in trades if t.session_name == session]
        if s_trades:
            s_wins = len([t for t in s_trades if t.pnl_pct > 0])
            s_wr = s_wins / len(s_trades) * 100
            s_pnl = sum(t.pnl_pct * t.size_pct for t in s_trades) * 100
            print(f"    {session:>12}: {len(s_trades)} trades, WR {s_wr:.0f}%, PnL {s_pnl:+.2f}%")

print(f"\n{'='*80}")
print("  Done.")
print(f"{'='*80}")

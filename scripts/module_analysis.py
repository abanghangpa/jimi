#!/usr/bin/env python3
"""Per-module performance analysis for JIMI scanner."""

import csv
import sys
import numpy as np
from collections import defaultdict

def load_trades(path):
    trades = []
    with open(path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            row['pnl_pct'] = float(row['pnl_pct'])
            row['size_pct'] = float(row['size_pct'])
            row['ics'] = float(row['ics'])
            trades.append(row)
    return trades

def analyze_module(trades, module_name, score_key, status_key=None, direction_filter=None):
    """Analyze a single module's predictive power."""
    filtered = trades
    if direction_filter:
        filtered = [t for t in trades if t['direction'] == direction_filter]
    
    if not filtered:
        return None
    
    winners = [t for t in filtered if t['pnl_pct'] > 0]
    losers = [t for t in filtered if t['pnl_pct'] <= 0]
    
    scores = []
    for t in filtered:
        try:
            scores.append(float(t.get(score_key, 0.5)))
        except (ValueError, TypeError):
            scores.append(0.5)
    
    win_scores = []
    loss_scores = []
    for t in filtered:
        try:
            s = float(t.get(score_key, 0.5))
        except (ValueError, TypeError):
            s = 0.5
        if t['pnl_pct'] > 0:
            win_scores.append(s)
        else:
            loss_scores.append(s)
    
    # Status breakdown if available
    status_breakdown = {}
    if status_key:
        for t in filtered:
            status = t.get(status_key, 'UNKNOWN')
            if status not in status_breakdown:
                status_breakdown[status] = {'count': 0, 'wins': 0, 'losses': 0, 'pnl': 0}
            status_breakdown[status]['count'] += 1
            status_breakdown[status]['pnl'] += t['pnl_pct'] * t['size_pct']
            if t['pnl_pct'] > 0:
                status_breakdown[status]['wins'] += 1
            else:
                status_breakdown[status]['losses'] += 1
    
    # Correlation: does higher score predict better outcomes?
    if len(scores) >= 3:
        pnl_values = [t['pnl_pct'] * t['size_pct'] for t in filtered]
        try:
            correlation = np.corrcoef(scores, pnl_values)[0, 1]
        except:
            correlation = 0.0
    else:
        correlation = 0.0
    
    result = {
        'module': module_name,
        'total': len(filtered),
        'winners': len(winners),
        'losers': len(losers),
        'win_rate': len(winners) / len(filtered) * 100 if filtered else 0,
        'win_avg': np.mean(win_scores) if win_scores else 0.5,
        'loss_avg': np.mean(loss_scores) if loss_scores else 0.5,
        'delta': (np.mean(win_scores) if win_scores else 0.5) - (np.mean(loss_scores) if loss_scores else 0.5),
        'correlation': correlation,
        'status_breakdown': status_breakdown,
        'direction': direction_filter or 'ALL',
    }
    return result

def print_module_report(results):
    """Print formatted module performance report."""
    print("\n" + "=" * 90)
    print("  JIMI — PER-MODULE PERFORMANCE ANALYSIS (30 DAYS)")
    print("=" * 90)
    
    # Overall summary
    print(f"\n  {'Module':<18} {'Trades':>6} {'WR%':>6} {'WinAvg':>7} {'LossAvg':>8} {'Delta':>7} {'Corr':>6} {'Verdict'}")
    print("  " + "-" * 84)
    
    for r in results:
        if r['direction'] != 'ALL':
            continue
        
        delta = r['delta']
        corr = r['correlation']
        
        if abs(delta) < 0.02 and abs(corr) < 0.15:
            verdict = "~ NEUTRAL"
        elif delta > 0.05 and corr > 0.1:
            verdict = "✓ PREDICTIVE"
        elif delta < -0.05 and corr < -0.1:
            verdict = "⚠ INVERSE (flip signal)"
        elif delta > 0.02:
            verdict = "~ slightly positive"
        elif delta < -0.02:
            verdict = "~ slightly negative"
        else:
            verdict = "~ NEUTRAL"
        
        print(f"  {r['module']:<18} {r['total']:>6} {r['win_rate']:>5.1f}% {r['win_avg']:>7.4f} {r['loss_avg']:>8.4f} {delta:>+7.4f} {corr:>+5.2f}  {verdict}")
    
    # Status breakdown for key modules
    for r in results:
        if r['direction'] != 'ALL' or not r['status_breakdown']:
            continue
        if r['module'] in ('M1 MACD', 'M2 EMA', 'M4 CVD', 'M5 LiqMag', 'M7 Regime', 'M13 Structure'):
            print(f"\n  {r['module']} Status Breakdown:")
            print(f"    {'Status':<12} {'Count':>5} {'Wins':>5} {'Losses':>6} {'WR%':>6} {'NetPnL':>8}")
            print("    " + "-" * 50)
            for status, data in sorted(r['status_breakdown'].items()):
                wr = data['wins'] / data['count'] * 100 if data['count'] > 0 else 0
                print(f"    {status:<12} {data['count']:>5} {data['wins']:>5} {data['losses']:>6} {wr:>5.1f}% {data['pnl']:>+7.2f}%")
    
    # Direction split
    print("\n" + "=" * 90)
    print("  DIRECTION SPLIT — LONG vs SHORT")
    print("=" * 90)
    
    for module_name in ['M1 MACD', 'M4 CVD', 'M5 LiqMag', 'M13 Structure']:
        long_r = next((r for r in results if r['module'] == module_name and r['direction'] == 'LONG'), None)
        short_r = next((r for r in results if r['module'] == module_name and r['direction'] == 'SHORT'), None)
        if long_r and short_r:
            print(f"\n  {module_name}:")
            print(f"    LONG:  {long_r['total']} trades, WR {long_r['win_rate']:.1f}%, win_avg {long_r['win_avg']:.4f}, loss_avg {long_r['loss_avg']:.4f}, delta {long_r['delta']:+.4f}")
            print(f"    SHORT: {short_r['total']} trades, WR {short_r['win_rate']:.1f}%, win_avg {short_r['win_avg']:.4f}, loss_avg {short_r['loss_avg']:.4f}, delta {short_r['delta']:+.4f}")

def main():
    path = sys.argv[1] if len(sys.argv) > 1 else 'jimi_trades.csv'
    trades = load_trades(path)
    
    print(f"\n  Loaded {len(trades)} trades from {path}")
    
    if not trades:
        print("  No trades to analyze.")
        return
    
    total_pnl = sum(t['pnl_pct'] * t['size_pct'] for t in trades)
    winners = [t for t in trades if t['pnl_pct'] > 0]
    losers = [t for t in trades if t['pnl_pct'] <= 0]
    print(f"  Overall: {len(winners)}W / {len(losers)}L ({len(winners)/len(trades)*100:.1f}% WR), Net PnL: {total_pnl:+.2f}%")
    
    modules = [
        ('M1 MACD', 'm1_score', 'm1_dir'),
        ('M2 EMA', 'm2_score', 'm2_status'),
        ('M3 VWAP', 'm3_score', None),
        ('M4 CVD', 'm4_score', 'm4_status'),
        ('M5 LiqMag', 'm5_score', 'm5_status'),
        ('M7 Regime', 'm7_score', None),
        ('M8 Funding', 'm8_score', 'm8_status'),
        ('M9 VolRegime', 'm9_score', 'm9_status'),
        ('M10 Macro', 'm10_score', 'm10_status'),
        ('M11 Momentum', 'm11_score', 'm11_status'),
        ('M12 OrderBook', 'm12_score', 'm12_status'),
        ('M13 Structure', 'm13_score', 'm13_status'),
        ('M14 Sweep', 'm14_score', 'm14_status'),
        ('ICS', 'ics', None),
    ]
    
    results = []
    for name, score_key, status_key in modules:
        # Overall
        r = analyze_module(trades, name, score_key, status_key)
        if r:
            results.append(r)
        # LONG split
        r_long = analyze_module(trades, name, score_key, status_key, 'LONG')
        if r_long:
            results.append(r_long)
        # SHORT split
        r_short = analyze_module(trades, name, score_key, status_key, 'SHORT')
        if r_short:
            results.append(r_short)
    
    print_module_report(results)
    
    # ICS quartile analysis
    print("\n" + "=" * 90)
    print("  ICS QUARTILE ANALYSIS")
    print("=" * 90)
    
    ics_values = [(t, float(t['ics'])) for t in trades]
    ics_values.sort(key=lambda x: x[1])
    n = len(ics_values)
    q1 = ics_values[:n//4]
    q2 = ics_values[n//4:n//2]
    q3 = ics_values[n//2:3*n//4]
    q4 = ics_values[3*n//4:]
    
    for label, quartile in [('Q1 (lowest)', q1), ('Q2', q2), ('Q3', q3), ('Q4 (highest)', q4)]:
        if not quartile:
            continue
        qs = [t for t, _ in quartile]
        wins = len([t for t in qs if t['pnl_pct'] > 0])
        pnl = sum(t['pnl_pct'] * t['size_pct'] for t in qs)
        avg_ics = np.mean([v for _, v in quartile])
        wr = wins / len(qs) * 100 if qs else 0
        print(f"  {label:<14} ICS={avg_ics:.3f}  trades={len(qs):>2}  WR={wr:>5.1f}%  NetPnL={pnl:>+.2f}%")
    
    # Trade timing analysis
    print("\n" + "=" * 90)
    print("  TRADE TIMING PATTERNS")
    print("=" * 90)
    
    from collections import Counter
    hours = []
    for t in trades:
        try:
            h = int(t['entry_time'].split(' ')[1].split(':')[0])
            hours.append((h, t['pnl_pct'] > 0, t['pnl_pct'] * t['size_pct']))
        except:
            pass
    
    hour_stats = defaultdict(lambda: {'count': 0, 'wins': 0, 'pnl': 0})
    for h, won, pnl in hours:
        hour_stats[h]['count'] += 1
        hour_stats[h]['pnl'] += pnl
        if won:
            hour_stats[h]['wins'] += 1
    
    print(f"\n  {'Hour(UTC)':<12} {'Trades':>6} {'WR%':>6} {'NetPnL':>8}")
    print("  " + "-" * 38)
    for h in sorted(hour_stats.keys()):
        s = hour_stats[h]
        wr = s['wins'] / s['count'] * 100 if s['count'] > 0 else 0
        print(f"  {h:02d}:00        {s['count']:>6} {wr:>5.1f}% {s['pnl']:>+7.2f}%")

if __name__ == '__main__':
    main()

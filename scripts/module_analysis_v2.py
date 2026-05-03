#!/usr/bin/env python3
"""Deep per-module performance analysis for JIMI scanner using forensic data."""

import csv
import sys
import numpy as np
from collections import defaultdict

def load_forensic(path):
    trades = []
    with open(path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            row['pnl_pct'] = float(row['pnl_pct'])
            row['size_pct'] = float(row['size_pct'])
            row['ics'] = float(row['ics'])
            for k in ['m1_score','m2_score','m3_score','m4_score','m5_score',
                       'm7_score','m8_score','m9_score','m10_score','m11_score',
                       'm12_score','cross_asset_score','trend_val','veto_soft_penalty']:
                if k in row:
                    try:
                        row[k] = float(row[k])
                    except:
                        row[k] = 0.5
            trades.append(row)
    return trades

def analyze(trades, name, score_key, status_key=None):
    winners = [t for t in trades if t['pnl_pct'] > 0]
    losers = [t for t in trades if t['pnl_pct'] <= 0]
    
    def avg(lst, key):
        vals = [t.get(key, 0.5) for t in lst if isinstance(t.get(key, 0.5), (int, float))]
        return np.mean(vals) if vals else 0.5
    
    win_avg = avg(winners, score_key)
    loss_avg = avg(losers, score_key)
    delta = win_avg - loss_avg
    
    scores = [t.get(score_key, 0.5) for t in trades if isinstance(t.get(score_key, 0.5), (int, float))]
    pnls = [t['pnl_pct'] * t['size_pct'] for t in trades]
    try:
        corr = np.corrcoef(scores[:len(pnls)], pnls[:len(scores)])[0, 1]
    except:
        corr = 0.0
    
    status_breakdown = {}
    if status_key:
        for t in trades:
            s = t.get(status_key, 'N/A')
            if s not in status_breakdown:
                status_breakdown[s] = {'count':0,'wins':0,'losses':0,'pnl':0}
            status_breakdown[s]['count'] += 1
            status_breakdown[s]['pnl'] += t['pnl_pct'] * t['size_pct']
            if t['pnl_pct'] > 0:
                status_breakdown[s]['wins'] += 1
            else:
                status_breakdown[s]['losses'] += 1
    
    return {
        'name': name, 'total': len(trades), 'winners': len(winners), 'losers': len(losers),
        'wr': len(winners)/len(trades)*100 if trades else 0,
        'win_avg': win_avg, 'loss_avg': loss_avg, 'delta': delta, 'corr': corr,
        'status': status_breakdown,
    }

def verdict(r):
    d, c = r['delta'], r['corr']
    if abs(d) < 0.02: return "~ NEUTRAL"
    if d > 0.05 and c > 0.1: return "✓ PREDICTIVE"
    if d < -0.05 and c < -0.1: return "⚠ INVERSE"
    if d > 0.02: return "~ slight +"
    if d < -0.02: return "~ slight -"
    return "~ NEUTRAL"

def main():
    path = sys.argv[1] if len(sys.argv) > 1 else 'jimi_forensic.csv'
    trades = load_forensic(path)
    print(f"\n  Loaded {len(trades)} trades from {path}")
    
    total_pnl = sum(t['pnl_pct'] * t['size_pct'] for t in trades)
    w = len([t for t in trades if t['pnl_pct'] > 0])
    print(f"  Overall: {w}W / {len(trades)-w}L ({w/len(trades)*100:.1f}% WR), Net PnL: {total_pnl:+.2f}%\n")
    
    modules = [
        ('M1 MACD',      'm1_score',          'm1_dir'),
        ('M2 EMA',       'm2_score',          'm2_status'),
        ('M3 VWAP',      'm3_score',          None),
        ('M4 CVD',       'm4_score',          'm4_status'),
        ('M5 LiqMag',    'm5_score',          'm5_status'),
        ('M7 Regime',    'm7_score',          None),
        ('M7 Composite', 'm7_composite',      None),
        ('M8 Funding',   'm8_score',          'm8_status'),
        ('M9 VolRegime', 'm9_score',          'm9_status'),
        ('M10 Macro',    'm10_score',         'm10_status'),
        ('M11 Momentum', 'm11_score',         'm11_status'),
        ('M12 OrderBook','m12_score',         'm12_status'),
        ('CrossAsset',   'cross_asset_score', None),
        ('Trend Val',    'trend_val',         None),
        ('ICS',          'ics',               None),
    ]
    
    # ── ALL TRADES ──
    print("=" * 95)
    print("  MODULE PERFORMANCE — ALL TRADES")
    print("=" * 95)
    print(f"  {'Module':<18} {'N':>3} {'WR%':>6} {'WinAvg':>7} {'LossAvg':>8} {'Delta':>7} {'Corr':>6} {'Verdict'}")
    print("  " + "-" * 88)
    for name, sk, stk in modules:
        r = analyze(trades, name, sk, stk)
        print(f"  {name:<18} {r['total']:>3} {r['wr']:>5.1f}% {r['win_avg']:>7.4f} {r['loss_avg']:>8.4f} {r['delta']:>+7.4f} {r['corr']:>+5.2f}  {verdict(r)}")
    
    # ── STATUS BREAKDOWNS ──
    print("\n" + "=" * 95)
    print("  STATUS BREAKDOWNS (PASS/FAIL/NEUTRAL)")
    print("=" * 95)
    for name, sk, stk in modules:
        if not stk: continue
        r = analyze(trades, name, sk, stk)
        if not r['status']: continue
        print(f"\n  {name}:")
        print(f"    {'Status':<14} {'N':>3} {'W':>3} {'L':>3} {'WR%':>6} {'NetPnL':>8}")
        print("    " + "-" * 44)
        for s, d in sorted(r['status'].items()):
            wr = d['wins']/d['count']*100 if d['count'] else 0
            print(f"    {s:<14} {d['count']:>3} {d['wins']:>3} {d['losses']:>3} {wr:>5.1f}% {d['pnl']:>+7.2f}%")
    
    # ── LONG vs SHORT ──
    print("\n" + "=" * 95)
    print("  LONG vs SHORT — KEY MODULES")
    print("=" * 95)
    for name, sk, stk in modules:
        if name in ('ICS','Trend Val','CrossAsset','M7 Composite'): continue
        longs = [t for t in trades if t['direction']=='LONG']
        shorts = [t for t in trades if t['direction']=='SHORT']
        rl = analyze(longs, name, sk, stk) if longs else None
        rs = analyze(shorts, name, sk, stk) if shorts else None
        if rl and rs:
            print(f"\n  {name}:")
            print(f"    LONG:  {rl['total']:>2} trades  WR={rl['wr']:>5.1f}%  win={rl['win_avg']:.4f}  loss={rl['loss_avg']:.4f}  Δ={rl['delta']:+.4f}  r={rl['corr']:+.2f}  {verdict(rl)}")
            print(f"    SHORT: {rs['total']:>2} trades  WR={rs['wr']:>5.1f}%  win={rs['win_avg']:.4f}  loss={rs['loss_avg']:.4f}  Δ={rs['delta']:+.4f}  r={rs['corr']:+.2f}  {verdict(rs)}")
    
    # ── ICS QUARTILE ──
    print("\n" + "=" * 95)
    print("  ICS QUARTILE ANALYSIS")
    print("=" * 95)
    sorted_t = sorted(trades, key=lambda t: t['ics'])
    n = len(sorted_t)
    for label, sl in [('Q1 (lowest 25%)', slice(0, n//4)), ('Q2', slice(n//4, n//2)),
                       ('Q3', slice(n//2, 3*n//4)), ('Q4 (highest 25%)', slice(3*n//4, n))]:
        chunk = sorted_t[sl]
        if not chunk: continue
        cw = len([t for t in chunk if t['pnl_pct'] > 0])
        cp = sum(t['pnl_pct'] * t['size_pct'] for t in chunk)
        avg_ics = np.mean([t['ics'] for t in chunk])
        print(f"  {label:<18} ICS={avg_ics:.3f}  n={len(chunk):>2}  WR={cw/len(chunk)*100:>5.1f}%  NetPnL={cp:>+7.2f}%")
    
    # ── ICS THRESHOLD SWEEP ──
    print("\n" + "=" * 95)
    print("  ICS THRESHOLD SWEEP — What if minimum ICS was higher?")
    print("=" * 95)
    print(f"  {'Min ICS':>8} {'Trades':>7} {'WR%':>6} {'NetPnL':>8} {'PF':>6}")
    print("  " + "-" * 40)
    for thresh in [0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60, 0.65]:
        filtered = [t for t in trades if t['ics'] >= thresh]
        if not filtered: continue
        fw = len([t for t in filtered if t['pnl_pct'] > 0])
        fp = sum(t['pnl_pct'] * t['size_pct'] for t in filtered)
        fg = sum(t['pnl_pct'] * t['size_pct'] for t in filtered if t['pnl_pct'] > 0)
        fl = abs(sum(t['pnl_pct'] * t['size_pct'] for t in filtered if t['pnl_pct'] <= 0))
        pf = fg / fl if fl > 0 else float('inf')
        print(f"  {thresh:>8.2f} {len(filtered):>7} {fw/len(filtered)*100:>5.1f}% {fp:>+7.2f}% {pf:>5.2f}")
    
    # ── VETO SOFT PENALTY IMPACT ──
    print("\n" + "=" * 95)
    print("  VETO SOFT PENALTY ANALYSIS")
    print("=" * 95)
    no_penalty = [t for t in trades if t.get('veto_soft_penalty', 0) == 0]
    with_penalty = [t for t in trades if t.get('veto_soft_penalty', 0) > 0]
    for label, group in [('No penalty', no_penalty), ('With penalty', with_penalty)]:
        if not group: continue
        gw = len([t for t in group if t['pnl_pct'] > 0])
        gp = sum(t['pnl_pct'] * t['size_pct'] for t in group)
        print(f"  {label:<14} n={len(group):>2}  WR={gw/len(group)*100:.1f}%  NetPnL={gp:+.2f}%")
    
    # ── REGIME ANALYSIS ──
    print("\n" + "=" * 95)
    print("  VOLATILITY REGIME PERFORMANCE")
    print("=" * 95)
    regimes = defaultdict(lambda: {'count':0,'wins':0,'pnl':0})
    for t in trades:
        vr = t.get('vol_regime', 'N/A')
        regimes[vr]['count'] += 1
        regimes[vr]['pnl'] += t['pnl_pct'] * t['size_pct']
        if t['pnl_pct'] > 0: regimes[vr]['wins'] += 1
    for vr, d in sorted(regimes.items()):
        wr = d['wins']/d['count']*100 if d['count'] else 0
        print(f"  {vr:<18} n={d['count']:>2}  WR={wr:.1f}%  NetPnL={d['pnl']:+.2f}%")
    
    # ── TREND DIR PERFORMANCE ──
    print("\n" + "=" * 95)
    print("  TREND DIRECTION PERFORMANCE")
    print("=" * 95)
    trends = defaultdict(lambda: {'count':0,'wins':0,'pnl':0})
    for t in trades:
        td = t.get('trend_dir', 'N/A')
        trends[td]['count'] += 1
        trends[td]['pnl'] += t['pnl_pct'] * t['size_pct']
        if t['pnl_pct'] > 0: trends[td]['wins'] += 1
    for td, d in sorted(trends.items()):
        wr = d['wins']/d['count']*100 if d['count'] else 0
        print(f"  {td:<18} n={d['count']:>2}  WR={wr:.1f}%  NetPnL={d['pnl']:+.2f}%")
    
    # ── DIRECTION RESOLVER vs ACTUAL ──
    print("\n" + "=" * 95)
    print("  DIRECTION RESOLVER ACCURACY")
    print("=" * 95)
    aligned = [t for t in trades if (t['trend_dir'] in ('UP','STRONG_UP') and t['direction']=='LONG') or
               (t['trend_dir'] in ('DOWN','STRONG_DOWN') and t['direction']=='SHORT') or
               (t['trend_dir'] == 'NEUTRAL')]
    contra = [t for t in trades if t not in aligned]
    for label, group in [('Aligned', aligned), ('Counter-trend', contra)]:
        if not group: continue
        gw = len([t for t in group if t['pnl_pct'] > 0])
        gp = sum(t['pnl_pct'] * t['size_pct'] for t in group)
        print(f"  {label:<14} n={len(group):>2}  WR={gw/len(group)*100:.1f}%  NetPnL={gp:+.2f}%")
    
    # ── M1 DIRECTION vs TRADE ──
    print("\n" + "=" * 95)
    print("  M1 MACD DIRECTION vs TRADE DIRECTION")
    print("=" * 95)
    for m1_dir in ['BULLISH', 'BEARISH', 'NEUTRAL']:
        group = [t for t in trades if t.get('m1_dir') == m1_dir]
        if not group: continue
        gw = len([t for t in group if t['pnl_pct'] > 0])
        gp = sum(t['pnl_pct'] * t['size_pct'] for t in group)
        longs = len([t for t in group if t['direction'] == 'LONG'])
        shorts = len([t for t in group if t['direction'] == 'SHORT'])
        print(f"  M1={m1_dir:<10} n={len(group):>2}  WR={gw/len(group)*100:.1f}%  NetPnL={gp:+.2f}%  (L:{longs} S:{shorts})")
    
    # ── ETH/BTC TREND ──
    print("\n" + "=" * 95)
    print("  ETH/BTC TREND vs PERFORMANCE")
    print("=" * 95)
    for trend in ['BULL', 'BEAR', 'N/A']:
        group = [t for t in trades if t.get('eth_btc_trend') == trend]
        if not group: continue
        gw = len([t for t in group if t['pnl_pct'] > 0])
        gp = sum(t['pnl_pct'] * t['size_pct'] for t in group)
        print(f"  ETH/BTC={trend:<6} n={len(group):>2}  WR={gw/len(group)*100:.1f}%  NetPnL={gp:+.2f}%")

    # ── SESSION ANALYSIS ──
    print("\n" + "=" * 95)
    print("  SESSION PERFORMANCE (Asia/London/NY)")
    print("=" * 95)
    sessions = defaultdict(lambda: {'count':0,'wins':0,'pnl':0})
    for t in trades:
        s = t.get('session', 'N/A') or 'N/A'
        sessions[s]['count'] += 1
        sessions[s]['pnl'] += t['pnl_pct'] * t['size_pct']
        if t['pnl_pct'] > 0: sessions[s]['wins'] += 1
    for s, d in sorted(sessions.items()):
        wr = d['wins']/d['count']*100 if d['count'] else 0
        print(f"  {s:<14} n={d['count']:>2}  WR={wr:.1f}%  NetPnL={d['pnl']:+.2f}%")

if __name__ == '__main__':
    main()

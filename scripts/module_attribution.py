#!/usr/bin/env python3
"""
Per-Module P&L Attribution — "When M4 scored above 0.7, what was the actual win rate?"

Runs a backtest then buckets trades by module score ranges to reveal
which modules are actually predictive vs. just historically correlated.

Usage:
    python scripts/module_attribution.py [csv_path] [--start=YYYY-MM-DD] [--end=YYYY-MM-DD]
"""

import sys
import os
import io
import numpy as np
import pandas as pd
from collections import defaultdict

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.engine import run_backtest, Trade
from src.config import CONFIG


def bucket_label(low, high):
    if low <= 0.0:
        return f"≤{high:.1f}"
    elif high >= 1.0:
        return f">{low:.1f}"
    else:
        return f"{low:.1f}-{high:.1f}"


def module_bucket_analysis(trades, module_name, score_fn, buckets):
    """
    For a given module, bucket trades by score ranges and compute P&L stats.
    Returns list of dicts with bucket stats.
    """
    results = []
    for low, high in buckets:
        bucket_trades = [t for t in trades if low < score_fn(t) <= high]
        if not bucket_trades:
            continue

        winners = [t for t in bucket_trades if t.pnl_pct > 0]
        losers = [t for t in bucket_trades if t.pnl_pct < 0]
        n = len(bucket_trades)
        wr = len(winners) / n if n > 0 else 0

        pnls = [t.pnl_pct * 100 for t in bucket_trades]
        avg_pnl = np.mean(pnls)
        total_pnl = np.sum(pnls)

        gross_win = sum(t.pnl_pct * 100 for t in winners)
        gross_loss = abs(sum(t.pnl_pct * 100 for t in losers))
        pf = gross_win / gross_loss if gross_loss > 0 else float('inf') if gross_win > 0 else 0

        avg_bars = np.mean([t.bars_held for t in bucket_trades])

        results.append({
            'module': module_name,
            'bucket': bucket_label(low, high),
            'trades': n,
            'win_rate': wr,
            'avg_pnl': avg_pnl,
            'total_pnl': total_pnl,
            'profit_factor': pf,
            'avg_bars': avg_bars,
            'wins': len(winners),
            'losses': len(losers),
        })

    return results


def regime_module_cross_analysis(trades):
    """
    Cross-tab: Regime × Module Confluence combinations.
    Answers: "In NEUTRAL regime, when M4>0.6 and M5>0.6, what's the WR?"
    """
    # Define the regime groups (keep bull/bear chop separate, collapse mild variants)
    def regime_group(regime):
        if regime in ('CHOP_MILD_BEAR', 'CHOP_HARD'):
            return 'CHOP_BEAR'
        if regime in ('CHOP_MILD_BULL',):
            return 'CHOP_BULL'
        if regime == 'CHOP_MILD':
            return 'CHOP_MILD'
        return regime

    # Define confluence conditions to test per regime
    confluences = [
        ('M4>0.6 & M5>0.6',
         lambda t: t.m4_score > 0.6 and t.m5_score > 0.6),
        ('M4>0.6 only',
         lambda t: t.m4_score > 0.6 and t.m5_score <= 0.6),
        ('M5>0.6 only',
         lambda t: t.m5_score > 0.6 and t.m4_score <= 0.6),
        ('M3+M4+M5 all >0.5',
         lambda t: t.m3_score > 0.5 and t.m4_score > 0.5 and t.m5_score > 0.5),
        ('ICS>0.65',
         lambda t: t.ics > 0.65),
        ('ICS>0.65 + M4>0.6',
         lambda t: t.ics > 0.65 and t.m4_score > 0.6),
        ('M11 PASS + M4>0.5',
         lambda t: t.m11_status == 'PASS' and t.m4_score > 0.5),
    ]

    regimes = sorted(set(regime_group(t.vol_regime) for t in trades))
    results = []

    for regime in regimes:
        regime_trades = [t for t in trades if regime_group(t.vol_regime) == regime]
        if not regime_trades:
            continue

        # Baseline for this regime (all trades)
        base_wr = len([t for t in regime_trades if t.pnl_pct > 0]) / len(regime_trades)
        base_avg = np.mean([t.pnl_pct * 100 for t in regime_trades])
        results.append({
            'regime': regime,
            'confluence': '— ALL (baseline)',
            'trades': len(regime_trades),
            'win_rate': base_wr,
            'avg_pnl': base_avg,
            'total_pnl': sum(t.pnl_pct * 100 for t in regime_trades),
            'lift_vs_base': 0.0,
        })

        for conf_name, conf_fn in confluences:
            matching = [t for t in regime_trades if conf_fn(t)]
            if len(matching) < 2:
                continue
            wr = len([t for t in matching if t.pnl_pct > 0]) / len(matching)
            avg_pnl = np.mean([t.pnl_pct * 100 for t in matching])
            total_pnl = sum(t.pnl_pct * 100 for t in matching)
            lift = (wr - base_wr) * 100  # percentage point lift

            results.append({
                'regime': regime,
                'confluence': conf_name,
                'trades': len(matching),
                'win_rate': wr,
                'avg_pnl': avg_pnl,
                'total_pnl': total_pnl,
                'lift_vs_base': lift,
            })

    return results


def print_regime_cross_table(results):
    """Print the regime × module confluence cross-tab."""
    if not results:
        print(f"\n  Regime × Module Cross-Tab: No data")
        return

    print(f"\n{'='*90}")
    print(f"  REGIME × MODULE CONFLUENCE CROSS-TAB")
    print(f"{'='*90}")
    print(f"  {'Regime':>16} {'Confluence':>24} {'Trades':>7} {'WR%':>8} {'Avg PnL':>10} {'Tot PnL':>10} {'Lift':>8}")
    print(f"  {'-'*84}")

    current_regime = None
    for r in results:
        if r['regime'] != current_regime:
            if current_regime is not None:
                print(f"  {'-'*84}")
            current_regime = r['regime']

        is_baseline = 'baseline' in r['confluence']
        wr_emoji = '🟢' if r['win_rate'] >= 0.6 else ('🟡' if r['win_rate'] >= 0.5 else '🔴')
        lift_str = f"{r['lift_vs_base']:+.1f}pp" if not is_baseline else '  —'
        prefix = '►' if is_baseline else ' '

        print(f"  {prefix}{r['regime']:>15} {r['confluence']:>24} {r['trades']:>7d} "
              f"{wr_emoji}{r['win_rate']*100:>6.1f}% {r['avg_pnl']:>+10.2f} "
              f"{r['total_pnl']:>+10.2f} {lift_str:>8}")


def ics_bucket_analysis(trades, buckets):
    """Special bucketing for ICS (composite score)."""
    return module_bucket_analysis(trades, 'ICS', lambda t: t.ics, buckets)


def confluence_analysis(trades):
    """
    Analyze module confluence — when specific modules agree, does WR improve?
    """
    results = []

    pairs = [
        ('M4+M5 both PASS',
         lambda t: t.m4_status == 'PASS' and t.m5_status == 'PASS'),
        ('M4+M5 both HIGH (>0.6)',
         lambda t: t.m4_score > 0.6 and t.m5_score > 0.6),
        ('M1+M4 direction agree',
         lambda t: ((t.m1_dir == 'BEARISH' and t.m4_score < 0.5) or
                     (t.m1_dir == 'BULLISH' and t.m4_score > 0.5))),
        ('M3+M4+M5 triple agree',
         lambda t: (t.m3_score > 0.5 and t.m4_score > 0.5 and t.m5_score > 0.5)),
        ('M7+M10 both favorable',
         lambda t: t.m7_score > 0.5 and t.m10_score > 0.5),
        ('High ICS (>0.65) + M4 PASS',
         lambda t: t.ics > 0.65 and t.m4_status == 'PASS'),
        ('M5 FAIL (should be avoided)',
         lambda t: t.m5_status == 'FAIL'),
        ('M11 PASS',
         lambda t: t.m11_status == 'PASS'),
        ('M11+M4 agree',
         lambda t: t.m11_status == 'PASS' and t.m4_score > 0.5),
    ]

    for name, fn in pairs:
        matching = [t for t in trades if fn(t)]
        if not matching:
            continue
        winners = [t for t in matching if t.pnl_pct > 0]
        wr = len(winners) / len(matching)
        avg_pnl = np.mean([t.pnl_pct * 100 for t in matching])
        total_pnl = sum(t.pnl_pct * 100 for t in matching)
        results.append({
            'confluence': name,
            'trades': len(matching),
            'win_rate': wr,
            'avg_pnl': avg_pnl,
            'total_pnl': total_pnl,
        })

    return results


def regime_attribution(trades):
    """P&L attribution by market regime."""
    results = []
    for regime in ['TRENDING', 'CHOP', 'COMPRESSING', 'NEUTRAL', 'CRISIS']:
        regime_trades = [t for t in trades if t.vol_regime == regime]
        if not regime_trades:
            continue
        winners = [t for t in regime_trades if t.pnl_pct > 0]
        wr = len(winners) / len(regime_trades)
        avg_pnl = np.mean([t.pnl_pct * 100 for t in regime_trades])
        total_pnl = sum(t.pnl_pct * 100 for t in regime_trades)
        results.append({
            'regime': regime,
            'trades': len(regime_trades),
            'win_rate': wr,
            'avg_pnl': avg_pnl,
            'total_pnl': total_pnl,
        })
    return results


def weight_calibration_check(trades):
    """
    Compare current ICS weights against empirical predictive power.
    If a module has high weight but low predictive power, it's over-weighted.
    """
    winners = [t for t in trades if t.pnl_pct > 0]
    losers = [t for t in trades if t.pnl_pct < 0]

    if not winners or not losers:
        return

    modules = [
        ('M1', lambda t: t.m1_score, CONFIG['M1_WEIGHT']),
        ('M3', lambda t: t.m3_score, CONFIG['M3_WEIGHT']),
        ('M4', lambda t: t.m4_score, CONFIG['M4_WEIGHT']),
        ('M5', lambda t: t.m5_score, CONFIG['M5_WEIGHT']),
        ('M10', lambda t: t.m10_score, CONFIG['M10_WEIGHT']),
        ('M11', lambda t: t.m11_score, CONFIG['M11_WEIGHT']),
        ('M12', lambda t: t.m12_score, CONFIG['M12_WEIGHT']),
    ]

    print(f"\n{'='*80}")
    print(f"  WEIGHT CALIBRATION CHECK — Current vs Empirical")
    print(f"{'='*80}")
    print(f"\n  {'Module':>6} {'Weight':>8} {'Win Avg':>10} {'Loss Avg':>10} {'Delta':>10} {'Empirical':>12} {'Status':>10}")
    print(f"  {'-'*70}")

    deltas = []
    for name, fn, weight in modules:
        w_avg = np.mean([fn(t) for t in winners])
        l_avg = np.mean([fn(t) for t in losers])
        delta = w_avg - l_avg
        deltas.append((name, weight, delta))

        empirical = abs(delta)
        status = ''
        if delta < -0.03:
            status = '⚠ INVERTED'
        elif delta < 0.02:
            status = '≈ WEAK'
        elif delta > 0.10:
            status = '✓ STRONG'
        else:
            status = '✓ OK'

        print(f"  {name:>6} {weight:>8.2f} {w_avg:>10.4f} {l_avg:>10.4f} {delta:>+10.4f} {empirical:>12.4f} {status:>10}")

    total_delta = sum(abs(d) for _, _, d in deltas)
    if total_delta > 0:
        print(f"\n  Suggested weight redistribution (based on |delta|):")
        print(f"  {'Module':>6} {'Current':>10} {'Suggested':>10} {'Change':>10}")
        print(f"  {'-'*40}")
        for name, weight, delta in deltas:
            suggested = abs(delta) / total_delta
            change = suggested - weight
            arrow = '↑' if change > 0.01 else ('↓' if change < -0.01 else '→')
            print(f"  {name:>6} {weight:>10.2f} {suggested:>10.2f} {arrow} {change:>+9.2f}")


def print_attribution_table(results, title):
    """Print a formatted attribution table."""
    if not results:
        print(f"\n  {title}: No data")
        return

    print(f"\n{'='*80}")
    print(f"  {title}")
    print(f"{'='*80}")

    first = results[0]
    cols = list(first.keys())

    header = f"  "
    for col in cols:
        if col in ('module', 'bucket'):
            header += f"{col:>14}"
        elif col in ('trades', 'wins', 'losses'):
            header += f"{col:>8}"
        elif col == 'win_rate':
            header += f"{'WR%':>8}"
        elif col == 'profit_factor':
            header += f"{'PF':>8}"
        elif col == 'avg_bars':
            header += f"{'Bars':>6}"
        else:
            header += f"{col:>10}"
    print(header)

    for r in results:
        row = f"  "
        for col in cols:
            val = r[col]
            if col in ('module', 'bucket', 'confluence', 'regime'):
                row += f"{val:>14}"
            elif col in ('trades', 'wins', 'losses'):
                row += f"{val:>8d}"
            elif col == 'win_rate':
                emoji = '🟢' if val >= 0.6 else ('🟡' if val >= 0.5 else '🔴')
                row += f"{emoji}{val*100:>6.1f}%"
            elif col == 'profit_factor':
                row += f"{val:>8.2f}"
            elif col == 'avg_bars':
                row += f"{val:>6.1f}"
            elif col in ('avg_pnl', 'total_pnl'):
                row += f"{val:>+10.2f}"
            else:
                row += f"{val:>10}"
        print(row)


def run_attribution(csv_path, date_start=None, date_end=None):
    """Run the full attribution analysis."""
    print(f"╔══════════════════════════════════════════════════════════════════╗")
    print(f"║  JIMI — Per-Module P&L Attribution Analysis                      ║")
    print(f"╚══════════════════════════════════════════════════════════════════╝")
    print(f"  Data: {csv_path}")
    if date_start or date_end:
        print(f"  Range: {date_start or 'start'} → {date_end or 'end'}")
    print()

    # Run backtest (suppress output)
    old_stdout = sys.stdout
    sys.stdout = io.StringIO()
    try:
        trades, stats, df = run_backtest(
            csv_path, verbose=False,
            date_start=date_start, date_end=date_end
        )
    finally:
        sys.stdout = old_stdout

    if not trades:
        print("  No trades in this period.")
        return

    print(f"  Analyzing {len(trades)} trades...")
    winners = [t for t in trades if t.pnl_pct > 0]
    losers = [t for t in trades if t.pnl_pct < 0]
    print(f"  Winners: {len(winners)} | Losers: {len(losers)} | WR: {len(winners)/len(trades)*100:.1f}%")
    print(f"  Net PnL: {sum(t.pnl_pct*100 for t in trades):+.2f}%")

    # 1. ICS Score Buckets
    ics_buckets = [(0.0, 0.50), (0.50, 0.55), (0.55, 0.60), (0.60, 0.65),
                   (0.65, 0.70), (0.70, 0.80), (0.80, 1.0)]
    ics_results = ics_bucket_analysis(trades, ics_buckets)
    print_attribution_table(ics_results,
        "ICS Composite Score — P&L by Score Bucket")

    # 2. Per-Module Score Buckets
    module_defs = [
        ('M1 (MACD)',     lambda t: t.m1_score,  [(0.0, 0.3), (0.3, 0.45), (0.45, 0.55), (0.55, 0.7), (0.7, 1.0)]),
        ('M3 (VWAP+Vol)', lambda t: t.m3_score,  [(0.0, 0.3), (0.3, 0.5), (0.5, 0.65), (0.65, 0.8), (0.8, 1.0)]),
        ('M4 (CVD)',      lambda t: t.m4_score,  [(0.0, 0.3), (0.3, 0.5), (0.5, 0.65), (0.65, 0.8), (0.8, 1.0)]),
        ('M5 (LiqMag)',   lambda t: t.m5_score,  [(0.0, 0.3), (0.3, 0.5), (0.5, 0.65), (0.65, 0.8), (0.8, 1.0)]),
        ('M7 (Regime)',   lambda t: t.m7_score,  [(0.0, 0.35), (0.35, 0.45), (0.45, 0.55), (0.55, 0.65), (0.65, 1.0)]),
        ('M10 (Macro)',   lambda t: t.m10_score, [(0.0, 0.3), (0.3, 0.5), (0.5, 0.65), (0.65, 0.8), (0.8, 1.0)]),
        ('M11 (MTF Mom)', lambda t: t.m11_score, [(0.0, 0.3), (0.3, 0.5), (0.5, 0.65), (0.65, 0.8), (0.8, 1.0)]),
    ]

    for mod_name, score_fn, buckets in module_defs:
        results = module_bucket_analysis(trades, mod_name, score_fn, buckets)
        print_attribution_table(results,
            f"{mod_name} — P&L by Score Bucket")

    # 3. Confluence Analysis
    conf_results = confluence_analysis(trades)
    print_attribution_table(conf_results,
        "Module Confluence — When Modules Agree")

    # 4. Regime Attribution
    regime_results = regime_attribution(trades)
    print_attribution_table(regime_results,
        "Market Regime — P&L by Regime State")

    # 5. Regime × Module Confluence Cross-Tab
    regime_cross_results = regime_module_cross_analysis(trades)
    print_regime_cross_table(regime_cross_results)

    # 6. Weight Calibration
    weight_calibration_check(trades)

    # 7. Summary
    print(f"\n{'='*80}")
    print(f"  KEY FINDINGS")
    print(f"{'='*80}")

    all_buckets = []
    for mod_name, score_fn, buckets in module_defs:
        results = module_bucket_analysis(trades, mod_name, score_fn, buckets)
        for r in results:
            if r['trades'] >= 2:
                all_buckets.append(r)

    if all_buckets:
        best = max(all_buckets, key=lambda x: x['avg_pnl'])
        worst = min(all_buckets, key=lambda x: x['avg_pnl'])
        print(f"\n  Best bucket:  {best['module']} {best['bucket']} → "
              f"{best['win_rate']*100:.0f}% WR, {best['avg_pnl']:+.2f}% avg PnL ({best['trades']} trades)")
        print(f"  Worst bucket: {worst['module']} {worst['bucket']} → "
              f"{worst['win_rate']*100:.0f}% WR, {worst['avg_pnl']:+.2f}% avg PnL ({worst['trades']} trades)")

    if len(trades) >= 5:
        ics_vals = [t.ics for t in trades]
        pnl_vals = [t.pnl_pct * 100 for t in trades]
        corr = np.corrcoef(ics_vals, pnl_vals)[0, 1]
        print(f"\n  ICS ↔ PnL correlation: {corr:+.3f}")
        if corr > 0.3:
            print(f"  → ICS has meaningful predictive value in this period")
        elif corr > 0:
            print(f"  → ICS has weak positive correlation — weights may need rebalancing")
        else:
            print(f"  → ICS is NOT predictive in this period — investigate module weights")

    print(f"\n{'='*80}")
    print(f"  DONE")
    print(f"{'='*80}\n")


def main():
    csv_path = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        'eth_15m_6m.csv'
    )
    date_start = None
    date_end = None

    for arg in sys.argv[1:]:
        if arg.startswith('--start='):
            date_start = arg.split('=', 1)[1]
        elif arg.startswith('--end='):
            date_end = arg.split('=', 1)[1]
        elif not arg.startswith('-'):
            csv_path = arg

    run_attribution(csv_path, date_start, date_end)


if __name__ == '__main__':
    main()

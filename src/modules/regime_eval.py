"""
Regime Detection Performance Evaluator

Measures whether M9 regime detection actually improves trading outcomes.
Three evaluation methods:
  1. Per-regime outcome analysis (what happened in each regime)
  2. Controlled A/B comparison (M9 on vs off)
  3. Regime permutation importance (shuffle regime labels, measure degradation)
"""

import numpy as np
from collections import defaultdict


# ═══════════════════════════════════════════════════════════════
# 1. PER-REGIME OUTCOME ANALYSIS
# ═══════════════════════════════════════════════════════════════

def analyze_regime_outcomes(trades):
    """Group trades by entry regime and compute per-regime statistics.

    Returns list of dicts with regime-level metrics.
    """
    regime_groups = defaultdict(list)
    for t in trades:
        regime_groups[t.entry_regime].append(t)

    results = []
    all_pnl = [t.pnl_pct * t.size_pct for t in trades]
    all_wr = sum(1 for p in all_pnl if p > 0) / len(all_pnl) if all_pnl else 0

    for regime in sorted(regime_groups.keys()):
        group = regime_groups[regime]
        pnls = [t.pnl_pct * t.size_pct for t in group]
        wins = sum(1 for p in pnls if p > 0)
        losses = sum(1 for p in pnls if p < 0)

        # Regime stability: did regime stay the same during the trade?
        stable_trades = sum(1 for t in group if t.regime_transitions == 0)
        avg_transitions = np.mean([t.regime_transitions for t in group])

        # Regime correctness: did the regime match the outcome?
        # TRENDING + win = correct, TRENDING + loss = incorrect regime or bad timing
        regime_correct = 0
        for t in group:
            if t.entry_regime == 'TRENDING' and t.pnl_pct > 0:
                regime_correct += 1
            elif t.entry_regime in ('CHOP_HARD', 'CRISIS') and t.pnl_pct < 0:
                regime_correct += 1  # correctly avoided (if blocked, won't appear)
            elif t.entry_regime == 'NEUTRAL':
                regime_correct += 0.5  # neutral is ambiguous
            elif t.entry_regime in ('CHOP_MILD', 'CHOP_MILD_BEAR', 'CHOP_MILD_BULL'):
                # Directional chop: correct if trade direction matched chop direction
                if ('BEAR' in t.entry_regime and t.direction == 'SHORT' and t.pnl_pct > 0) or \
                   ('BULL' in t.entry_regime and t.direction == 'LONG' and t.pnl_pct > 0):
                    regime_correct += 1
                elif t.pnl_pct > 0:
                    regime_correct += 0.5  # won but against chop direction

        results.append({
            'regime': regime,
            'trades': len(group),
            'wins': wins,
            'losses': losses,
            'win_rate': wins / len(group) if group else 0,
            'avg_pnl': np.mean(pnls) if pnls else 0,
            'total_pnl': sum(pnls),
            'avg_size': np.mean([t.size_pct for t in group]),
            'avg_bars_held': np.mean([t.bars_held for t in group]),
            'regime_stable_pct': stable_trades / len(group) if group else 0,
            'avg_regime_transitions': avg_transitions,
            'regime_correct_pct': regime_correct / len(group) if group else 0,
            'pnl_vs_baseline': np.mean(pnls) - np.mean(all_pnl) if all_pnl else 0,
        })

    return results


def analyze_regime_transitions(trades):
    """Analyze how regime transitions during trades affect outcomes."""
    stable = [t for t in trades if t.regime_transitions == 0]
    unstable = [t for t in trades if t.regime_transitions > 0]

    def stats(group, label):
        if not group:
            return {'label': label, 'trades': 0}
        pnls = [t.pnl_pct * t.size_pct for t in group]
        return {
            'label': label,
            'trades': len(group),
            'win_rate': sum(1 for p in pnls if p > 0) / len(pnls),
            'avg_pnl': np.mean(pnls),
            'total_pnl': sum(pnls),
            'avg_transitions': np.mean([t.regime_transitions for t in group]),
        }

    return stats(stable, 'STABLE (0 transitions)'), stats(unstable, 'UNSTABLE (1+ transitions)')


# ═══════════════════════════════════════════════════════════════
# 2. CONTROLLED A/B COMPARISON
# ═══════════════════════════════════════════════════════════════

def compare_backtest_results(trades_with, trades_without, label_with='M9 ON', label_without='M9 OFF'):
    """Compare two backtest runs: with regime detection vs without.

    Returns comparison dict.
    """
    def compute_stats(trades, label):
        if not trades:
            return {'label': label, 'trades': 0}
        pnls = [t.pnl_pct * t.size_pct for t in trades]
        wins = sum(1 for p in pnls if p > 0)
        return {
            'label': label,
            'trades': len(trades),
            'wins': wins,
            'losses': len(trades) - wins,
            'win_rate': wins / len(trades),
            'avg_pnl': np.mean(pnls),
            'total_pnl': sum(pnls),
            'max_win': max(pnls) if pnls else 0,
            'max_loss': min(pnls) if pnls else 0,
            'profit_factor': sum(p for p in pnls if p > 0) / abs(sum(p for p in pnls if p < 0)) if sum(p for p in pnls if p < 0) != 0 else float('inf'),
            'avg_bars': np.mean([t.bars_held for t in trades]),
        }

    a = compute_stats(trades_with, label_with)
    b = compute_stats(trades_without, label_without)

    # Compute lift
    lift = {}
    if b['trades'] > 0:
        lift['wr_lift'] = a.get('win_rate', 0) - b.get('win_rate', 0)
        lift['pnl_lift'] = a.get('avg_pnl', 0) - b.get('avg_pnl', 0)
        lift['total_pnl_lift'] = a.get('total_pnl', 0) - b.get('total_pnl', 0)
        lift['pf_lift'] = a.get('profit_factor', 0) - b.get('profit_factor', 0)

    return {'with': a, 'without': b, 'lift': lift}


# ═══════════════════════════════════════════════════════════════
# 3. REGIME PERMUTATION IMPORTANCE
# ═══════════════════════════════════════════════════════════════

def regime_permutation_test(trades, n_permutations=500, seed=42):
    """Shuffle regime labels across trades, recompute per-regime PnL,
    measure how much the real regime assignment outperforms random.

    If real regime→outcome correlation >> shuffled, regime detection has value.
    """
    rng = np.random.RandomState(seed)
    real_pnls = np.array([t.pnl_pct * t.size_pct for t in trades])
    real_regimes = np.array([t.entry_regime for t in trades])

    # Real per-regime avg PnL
    real_regime_pnl = {}
    for regime in set(real_regimes):
        mask = real_regimes == regime
        real_regime_pnl[regime] = np.mean(real_pnls[mask]) if mask.sum() > 0 else 0

    # Real spread: difference between best and worst regime avg PnL
    real_spread = max(real_regime_pnl.values()) - min(real_regime_pnl.values()) if real_regime_pnl else 0

    # Real correlation: regime "quality" score vs trade PnL
    # Assign numeric quality: TRENDING=1.0, COMPRESSING=0.7, NEUTRAL=0.5, CHOP_MILD*=0.3, CHOP_HARD=0.1, CRISIS=0.0
    regime_quality = {
        'TRENDING': 1.0, 'COMPRESSING': 0.7, 'NEUTRAL': 0.5,
        'CHOP_MILD': 0.3, 'CHOP_MILD_BEAR': 0.3, 'CHOP_MILD_BULL': 0.3,
        'CHOP_HARD': 0.1, 'CRISIS': 0.0, 'UNKNOWN': 0.5,
    }
    real_quality = np.array([regime_quality.get(r, 0.5) for r in real_regimes])
    real_corr = np.corrcoef(real_quality, real_pnls)[0, 1] if len(trades) >= 5 else 0

    # Permutation test
    perm_spreads = []
    perm_corrs = []
    for _ in range(n_permutations):
        shuffled = rng.permutation(real_regimes)
        perm_regime_pnl = {}
        for regime in set(shuffled):
            mask = shuffled == regime
            perm_regime_pnl[regime] = np.mean(real_pnls[mask]) if mask.sum() > 0 else 0
        spread = max(perm_regime_pnl.values()) - min(perm_regime_pnl.values()) if perm_regime_pnl else 0
        perm_spreads.append(spread)

        perm_quality = np.array([regime_quality.get(r, 0.5) for r in shuffled])
        corr = np.corrcoef(perm_quality, real_pnls)[0, 1] if len(trades) >= 5 else 0
        perm_corrs.append(corr)

    # Significance
    spread_p = np.mean([s >= real_spread for s in perm_spreads])
    corr_p = np.mean([c >= real_corr for c in perm_corrs])

    return {
        'real_spread': real_spread,
        'perm_spread_mean': np.mean(perm_spreads),
        'perm_spread_std': np.std(perm_spreads),
        'spread_p_value': spread_p,
        'spread_significant': spread_p < 0.05,
        'real_corr': real_corr,
        'perm_corr_mean': np.mean(perm_corrs),
        'perm_corr_std': np.std(perm_corrs),
        'corr_p_value': corr_p,
        'corr_significant': corr_p < 0.05,
        'n_permutations': n_permutations,
        'real_regime_pnl': real_regime_pnl,
    }


# ═══════════════════════════════════════════════════════════════
# 4. REGIME TIMING ANALYSIS
# ═══════════════════════════════════════════════════════════════

def analyze_regime_timing(trades):
    """Analyze if regime at entry predicts trade outcome better than regime at exit."""
    entry_correct = 0
    exit_correct = 0
    total = 0

    for t in trades:
        if t.pnl_pct == 0:
            continue
        total += 1
        won = t.pnl_pct > 0

        # Entry regime correctness
        if t.entry_regime == 'TRENDING' and won:
            entry_correct += 1
        elif t.entry_regime in ('CHOP_HARD', 'CRISIS') and not won:
            entry_correct += 1

        # Exit regime correctness (does exit regime tell us more?)
        if t.exit_regime == 'TRENDING' and won:
            exit_correct += 1
        elif t.exit_regime in ('CHOP_HARD', 'CRISIS') and not won:
            exit_correct += 1

    return {
        'total': total,
        'entry_regime_accuracy': entry_correct / total if total else 0,
        'exit_regime_accuracy': exit_correct / total if total else 0,
        'entry_better': entry_correct > exit_correct,
    }


# ═══════════════════════════════════════════════════════════════
# 5. PRINT REPORTS
# ═══════════════════════════════════════════════════════════════

def print_regime_outcomes(results):
    """Print per-regime outcome table."""
    print(f"\n{'='*90}")
    print(f"  REGIME DETECTION — Per-Regime Outcome Analysis")
    print(f"{'='*90}")
    print(f"  {'Regime':<18} {'Trades':>6} {'WR%':>6} {'AvgPnL':>8} {'TotPnL':>9} "
          f"{'Stable%':>8} {'Trans':>6} {'Correct%':>9} {'vsBase':>8}")
    print(f"  {'-'*84}")

    for r in results:
        regime = r['regime']
        icon = {'TRENDING': '🟢', 'NEUTRAL': '⚪', 'COMPRESSING': '🔵',
                'CHOP_MILD': '🟡', 'CHOP_MILD_BEAR': '🟠', 'CHOP_MILD_BULL': '🟠',
                'CHOP_HARD': '🔴', 'CRISIS': '💀'}.get(regime, '❓')
        wr_icon = '🟢' if r['win_rate'] >= 0.6 else ('🟡' if r['win_rate'] >= 0.5 else '🔴')

        print(f"  {icon} {regime:<16} {r['trades']:>6} {wr_icon}{r['win_rate']*100:>5.1f}% "
              f"{r['avg_pnl']*100:>+7.2f}% {r['total_pnl']*100:>+8.2f}% "
              f"{r['regime_stable_pct']*100:>7.1f}% {r['avg_regime_transitions']:>5.1f} "
              f"{r['regime_correct_pct']*100:>8.1f}% {r['pnl_vs_baseline']*100:>+7.2f}%")


def print_regime_transitions(stable, unstable):
    """Print regime stability comparison."""
    print(f"\n{'='*70}")
    print(f"  REGIME STABILITY — Do regime changes during trades hurt?")
    print(f"{'='*70}")

    for group in [stable, unstable]:
        if group['trades'] == 0:
            print(f"  {group['label']}: no trades")
            continue
        print(f"  {group['label']}:")
        print(f"    Trades: {group['trades']}  |  WR: {group['win_rate']*100:.1f}%  |  "
              f"Avg PnL: {group['avg_pnl']*100:+.2f}%  |  Total: {group['total_pnl']*100:+.2f}%")

    if stable['trades'] > 0 and unstable['trades'] > 0:
        wr_diff = stable['win_rate'] - unstable['win_rate']
        pnl_diff = stable['avg_pnl'] - unstable['avg_pnl']
        print(f"\n  Stability advantage: WR {wr_diff*100:+.1f}pp, AvgPnL {pnl_diff*100:+.2f}%")
        if wr_diff > 0.05:
            print(f"  ✅ Stable regimes significantly outperform — regime detection adds value")
        elif wr_diff < -0.05:
            print(f"  ⚠️  Unstable regimes outperform — regime changes may signal opportunity")
        else:
            print(f"  ≈ Minimal difference — regime stability doesn't matter much")


def print_ab_comparison(comp):
    """Print controlled A/B comparison."""
    print(f"\n{'='*70}")
    print(f"  CONTROLLED COMPARISON — M9 ON vs M9 OFF")
    print(f"{'='*70}")

    for key in ['with', 'without']:
        d = comp[key]
        if d['trades'] == 0:
            print(f"  {d['label']}: no trades")
            continue
        pf = f"{d['profit_factor']:.2f}" if d['profit_factor'] != float('inf') else '∞'
        print(f"\n  {d['label']}:")
        print(f"    Trades: {d['trades']}  |  WR: {d['win_rate']*100:.1f}%  |  "
              f"Avg PnL: {d['avg_pnl']*100:+.2f}%  |  Total: {d['total_pnl']*100:+.2f}%")
        print(f"    PF: {pf}  |  Max Win: {d['max_win']*100:+.2f}%  |  "
              f"Max Loss: {d['max_loss']*100:+.2f}%  |  Avg Bars: {d['avg_bars']:.1f}")

    lift = comp.get('lift', {})
    if lift:
        print(f"\n  Lift (M9 ON - M9 OFF):")
        print(f"    WR: {lift.get('wr_lift', 0)*100:+.1f}pp  |  "
              f"Avg PnL: {lift.get('pnl_lift', 0)*100:+.2f}%  |  "
              f"Total PnL: {lift.get('total_pnl_lift', 0)*100:+.2f}%")

        if lift.get('total_pnl_lift', 0) > 0:
            print(f"  ✅ M9 regime detection IMPROVES performance")
        elif lift.get('total_pnl_lift', 0) < 0:
            print(f"  ❌ M9 regime detection HURTS performance")
        else:
            print(f"  ≈ M9 has no measurable impact")


def print_permutation_test(result):
    """Print permutation importance results."""
    print(f"\n{'='*70}")
    print(f"  REGIME PERMUTATION IMPORTANCE — Is regime-outcome link real?")
    print(f"{'='*70}")
    print(f"  Method: shuffle regime labels {result['n_permutations']}x, measure degradation")
    print(f"")
    print(f"  Regime→PnL Correlation:")
    print(f"    Real:     {result['real_corr']:+.4f}")
    print(f"    Permuted: {result['perm_corr_mean']:+.4f} ± {result['perm_corr_std']:.4f}")
    print(f"    p-value:  {result['corr_p_value']:.4f}  {'✅ SIGNIFICANT' if result['corr_significant'] else '≈ NOT SIGNIFICANT'}")
    print(f"")
    print(f"  Regime PnL Spread (best - worst):")
    print(f"    Real:     {result['real_spread']*100:+.2f}%")
    print(f"    Permuted: {result['perm_spread_mean']*100:+.2f}% ± {result['perm_spread_std']*100:.2f}%")
    print(f"    p-value:  {result['spread_p_value']:.4f}  {'✅ SIGNIFICANT' if result['spread_significant'] else '≈ NOT SIGNIFICANT'}")
    print(f"")
    print(f"  Per-regime avg PnL:")
    for regime, pnl in sorted(result['real_regime_pnl'].items(), key=lambda x: -x[1]):
        icon = {'TRENDING': '🟢', 'NEUTRAL': '⚪', 'COMPRESSING': '🔵',
                'CHOP_MILD': '🟡', 'CHOP_MILD_BEAR': '🟠', 'CHOP_MILD_BULL': '🟠',
                'CHOP_HARD': '🔴', 'CRISIS': '💀'}.get(regime, '❓')
        print(f"    {icon} {regime:<18} {pnl*100:>+7.2f}%")

    if result['corr_significant']:
        print(f"\n  ✅ Regime detection has STATISTICALLY SIGNIFICANT relationship with outcomes")
    else:
        print(f"\n  ⚠️  Regime labels don't significantly predict outcomes (could be random)")


def print_timing_analysis(timing):
    """Print regime timing analysis."""
    print(f"\n{'='*70}")
    print(f"  REGIME TIMING — Entry vs Exit regime accuracy")
    print(f"{'='*70}")
    print(f"  Trades analyzed: {timing['total']}")
    print(f"  Entry regime accuracy: {timing['entry_regime_accuracy']*100:.1f}%")
    print(f"  Exit regime accuracy:  {timing['exit_regime_accuracy']*100:.1f}%")
    if timing['entry_better']:
        print(f"  → Entry regime is more predictive (regime BEFORE trade matters more)")
    else:
        print(f"  → Exit regime is more predictive (regime DURING trade matters more)")


def print_full_report(trades, trades_no_m9=None, n_permutations=500):
    """Print the complete regime evaluation report."""
    print(f"\n{'#'*90}")
    print(f"  JIMI — REGIME DETECTION PERFORMANCE EVALUATION")
    print(f"{'#'*90}")
    print(f"  Total trades: {len(trades)}")

    # 1. Per-regime outcomes
    outcomes = analyze_regime_outcomes(trades)
    print_regime_outcomes(outcomes)

    # 2. Regime stability
    stable, unstable = analyze_regime_transitions(trades)
    print_regime_transitions(stable, unstable)

    # 3. A/B comparison (if trades without M9 provided)
    if trades_no_m9 is not None:
        comp = compare_backtest_results(trades, trades_no_m9)
        print_ab_comparison(comp)

    # 4. Permutation importance
    if len(trades) >= 10:
        perm = regime_permutation_test(trades, n_permutations=n_permutations)
        print_permutation_test(perm)

    # 5. Timing analysis
    timing = analyze_regime_timing(trades)
    print_timing_analysis(timing)

    print(f"\n{'#'*90}\n")

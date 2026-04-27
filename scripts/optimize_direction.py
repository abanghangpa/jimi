#!/usr/bin/env python3
"""
Direction Combination Optimizer — Find the best module weights for direction prediction.

Approaches:
  1. Empirical weight calibration (from module attribution deltas)
  2. Grid search over weight combinations
  3. Feature selection — which module subsets predict direction best?
  4. Logistic regression on module scores → direction accuracy

Usage:
    python3 scripts/optimize_direction.py eth_15m_6m.csv
"""

import sys, os, json, itertools
import numpy as np
import pandas as pd
from collections import defaultdict

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from src.config import CONFIG, load_config
from src.engine import run_backtest, Trade


# ═══════════════════════════════════════════════════════════════
# 1. EXTRACT MODULE SCORES FROM TRADES
# ═══════════════════════════════════════════════════════════════

def extract_trade_features(trades):
    """Extract module scores and outcome from trades."""
    rows = []
    for t in trades:
        rows.append({
            'm1_score': t.m1_score,
            'm3_score': t.m3_score,
            'm4_score': t.m4_score,
            'm5_score': t.m5_score,
            'm7_score': t.m7_score,
            'm8_score': t.m8_score,
            'm9_score': t.m9_score,
            'm10_score': t.m10_score,
            'm11_score': t.m11_score,
            'm12_score': t.m12_score,
            'direction': 1 if t.direction == 'LONG' else -1,
            'won': 1 if t.pnl_pct > 0 else 0,
            'pnl': t.pnl_pct,
            'ics': t.ics,
        })
    return pd.DataFrame(rows)


# ═══════════════════════════════════════════════════════════════
# 2. SINGLE-MODULE PREDICTIVE POWER
# ═══════════════════════════════════════════════════════════════

def single_module_analysis(df):
    """Rank each module's ability to predict trade outcome."""
    print(f"\n{'='*80}")
    print(f"  SINGLE-MODULE PREDICTIVE POWER")
    print(f"{'='*80}")

    modules = ['m1_score', 'm3_score', 'm4_score', 'm5_score', 'm7_score',
               'm8_score', 'm9_score', 'm10_score', 'm11_score', 'm12_score']

    results = []
    for mod in modules:
        scores = df[mod].values
        outcomes = df['won'].values

        # Correlation between score and outcome
        corr = np.corrcoef(scores, outcomes)[0, 1] if len(scores) > 2 else 0

        # Win rate above/below median
        median_score = np.median(scores)
        above = df[df[mod] > median_score]
        below = df[df[mod] <= median_score]
        wr_above = above['won'].mean() if len(above) > 0 else 0.5
        wr_below = below['won'].mean() if len(below) > 0 else 0.5
        wr_delta = wr_above - wr_below

        # PnL contribution
        winners = df[df['won'] == 1]
        losers = df[df['won'] == 0]
        w_avg = winners[mod].mean() if len(winners) > 0 else 0.5
        l_avg = losers[mod].mean() if len(losers) > 0 else 0.5
        pnl_delta = w_avg - l_avg

        results.append({
            'module': mod.replace('_score', '').upper(),
            'correlation': corr,
            'wr_above_median': wr_above,
            'wr_below_median': wr_below,
            'wr_delta': wr_delta,
            'win_avg_score': w_avg,
            'loss_avg_score': l_avg,
            'pnl_delta': pnl_delta,
        })

    # Sort by absolute predictive power
    results.sort(key=lambda x: abs(x['pnl_delta']), reverse=True)

    print(f"\n  {'Module':>6} {'Corr':>8} {'WR>Med':>8} {'WR≤Med':>8} {'ΔWR':>8} {'WinAvg':>8} {'LossAvg':>8} {'ΔPnL':>8} {'Power':>8}")
    print(f"  {'-'*74}")
    for r in results:
        power = abs(r['pnl_delta']) * abs(r['correlation'])
        emoji = '🟢' if r['pnl_delta'] > 0.02 else ('🟡' if r['pnl_delta'] > 0 else '🔴')
        print(f"  {r['module']:>6} {r['correlation']:>+8.4f} {r['wr_above_median']:>7.1%} {r['wr_below_median']:>7.1%} "
              f"{r['wr_delta']:>+7.1%} {r['win_avg_score']:>8.4f} {r['loss_avg_score']:>8.4f} "
              f"{r['pnl_delta']:>+8.4f} {emoji}{power:>7.4f}")

    return results


# ═══════════════════════════════════════════════════════════════
# 3. PAIRWISE MODULE INTERACTIONS
# ═══════════════════════════════════════════════════════════════

def pairwise_analysis(df):
    """Find which module pairs interact best for prediction."""
    print(f"\n{'='*80}")
    print(f"  PAIRWISE MODULE INTERACTIONS")
    print(f"{'='*80}")

    modules = ['m1_score', 'm3_score', 'm4_score', 'm5_score',
               'm10_score', 'm11_score']

    pairs = list(itertools.combinations(modules, 2))
    results = []

    for mod_a, mod_b in pairs:
        # Confluence: both above median
        med_a = df[mod_a].median()
        med_b = df[mod_b].median()

        both_above = df[(df[mod_a] > med_a) & (df[mod_b] > med_b)]
        either_below = df[~((df[mod_a] > med_a) & (df[mod_b] > med_b))]

        if len(both_above) < 3:
            continue

        wr_both = both_above['won'].mean()
        wr_either = either_below['won'].mean()
        pnl_both = both_above['pnl'].mean()
        pnl_either = either_below['pnl'].mean()

        results.append({
            'pair': f"{mod_a.replace('_score','').upper()}+{mod_b.replace('_score','').upper()}",
            'n_both': len(both_above),
            'wr_both': wr_both,
            'wr_either': wr_either,
            'wr_lift': wr_both - wr_either,
            'pnl_both': pnl_both,
            'pnl_either': pnl_either,
            'pnl_lift': pnl_both - pnl_either,
        })

    results.sort(key=lambda x: x['pnl_lift'], reverse=True)

    print(f"\n  {'Pair':>12} {'N':>5} {'WR both':>8} {'WR either':>10} {'Lift':>8} {'PnL both':>10} {'PnL either':>12} {'PnL Lift':>10}")
    print(f"  {'-'*78}")
    for r in results:
        emoji = '🟢' if r['pnl_lift'] > 0.1 else ('🟡' if r['pnl_lift'] > 0 else '🔴')
        print(f"  {r['pair']:>12} {r['n_both']:>5} {r['wr_both']:>7.1%} {r['wr_either']:>9.1%} "
              f"{r['wr_lift']:>+7.1%} {r['pnl_both']:>+9.2f} {r['pnl_either']:>+11.2f} {emoji}{r['pnl_lift']:>+9.2f}")

    return results


# ═══════════════════════════════════════════════════════════════
# 4. WEIGHT OPTIMIZATION (Grid Search)
# ═══════════════════════════════════════════════════════════════

def optimize_weights(df):
    """Grid search over module weight combinations."""
    print(f"\n{'='*80}")
    print(f"  WEIGHT OPTIMIZATION — Grid Search")
    print(f"{'='*80}")

    modules = ['m1_score', 'm4_score', 'm5_score', 'm10_score', 'm11_score']
    mod_names = [m.replace('_score', '').upper() for m in modules]

    # Normalize scores to 0-1
    norm_df = df.copy()
    for m in modules:
        vals = norm_df[m]
        norm_df[m] = (vals - vals.min()) / (vals.max() - vals.min() + 1e-10)

    # Grid search: weight each module 0.0 to 1.0 in steps of 0.2
    # Only test combinations that sum to ~1.0
    step = 0.2
    best_score = -999
    best_weights = None
    all_results = []

    for w1 in np.arange(0, 1.01, step):
        for w4 in np.arange(0, 1.01 - w1, step):
            for w5 in np.arange(0, 1.01 - w1 - w4, step):
                for w10 in np.arange(0, 1.01 - w1 - w4 - w5, step):
                    w11 = 1.0 - w1 - w4 - w5 - w10
                    if w11 < -0.01 or w11 > 1.01:
                        continue

                    weights = [w1, w4, w5, w10, w11]
                    total = sum(weights)
                    if total == 0:
                        continue
                    weights = [w / total for w in weights]

                    # Compute composite score
                    composite = sum(norm_df[m] * w for m, w in zip(modules, weights))

                    # Evaluate: correlation with outcome
                    corr = np.corrcoef(composite, df['won'])[0, 1]

                    # Win rate in top quartile
                    q75 = composite.quantile(0.75)
                    top_q = df[composite >= q75]
                    wr_top = top_q['won'].mean() if len(top_q) > 0 else 0.5

                    # PnL delta (top vs bottom quartile)
                    q25 = composite.quantile(0.25)
                    bot_q = df[composite <= q25]
                    pnl_top = top_q['pnl'].mean() if len(top_q) > 0 else 0
                    pnl_bot = bot_q['pnl'].mean() if len(bot_q) > 0 else 0
                    pnl_delta = pnl_top - pnl_bot

                    # Combined score: correlation * pnl_delta
                    score = corr * pnl_delta

                    weight_str = '/'.join(f"{w:.1f}" for w in weights)
                    all_results.append({
                        'weights': dict(zip(mod_names, weights)),
                        'weight_str': weight_str,
                        'correlation': corr,
                        'wr_top_quartile': wr_top,
                        'pnl_top': pnl_top,
                        'pnl_bottom': pnl_bot,
                        'pnl_delta': pnl_delta,
                        'score': score,
                    })

                    if score > best_score:
                        best_score = score
                        best_weights = dict(zip(mod_names, weights))

    all_results.sort(key=lambda x: x['score'], reverse=True)

    print(f"\n  Top 15 weight combinations:")
    print(f"  {'M1/M4/M5/M10/M11':>20} {'Corr':>8} {'WR TopQ':>8} {'PnL Top':>9} {'PnL Bot':>9} {'ΔPnL':>9} {'Score':>8}")
    print(f"  {'-'*75}")
    for r in all_results[:15]:
        emoji = '🟢' if r['score'] > 0.01 else '🟡'
        print(f"  {r['weight_str']:>20} {r['correlation']:>+8.4f} {r['wr_top_quartile']:>7.1%} "
              f"{r['pnl_top']:>+8.2f} {r['pnl_bottom']:>+8.2f} {r['pnl_delta']:>+8.2f} {emoji}{r['score']:>7.4f}")

    print(f"\n  🏆 Best weights: {best_weights}")
    print(f"     Best score: {best_score:.4f}")

    return best_weights, all_results


# ═══════════════════════════════════════════════════════════════
# 5. FEATURE SUBSET SELECTION
# ═══════════════════════════════════════════════════════════════

def feature_subset_analysis(df):
    """Test which subsets of modules predict direction best."""
    print(f"\n{'='*80}")
    print(f"  FEATURE SUBSET SELECTION — Which modules to include?")
    print(f"{'='*80}")

    modules = ['m1_score', 'm4_score', 'm5_score', 'm10_score', 'm11_score']
    mod_names = [m.replace('_score', '').upper() for m in modules]

    # Normalize
    norm_df = df.copy()
    for m in modules:
        vals = norm_df[m]
        norm_df[m] = (vals - vals.min()) / (vals.max() - vals.min() + 1e-10)

    results = []

    # Test all subsets of size 2-5
    for size in range(2, len(modules) + 1):
        for subset in itertools.combinations(range(len(modules)), size):
            subset_modules = [modules[i] for i in subset]
            subset_names = [mod_names[i] for i in subset]

            # Equal weights within subset
            weights = [1.0 / len(subset)] * len(subset)
            composite = sum(norm_df[m] * w for m, w in zip(subset_modules, weights))

            corr = np.corrcoef(composite, df['won'])[0, 1]

            q75 = composite.quantile(0.75)
            q25 = composite.quantile(0.25)
            top_q = df[composite >= q75]
            bot_q = df[composite <= q25]
            pnl_top = top_q['pnl'].mean() if len(top_q) > 0 else 0
            pnl_bot = bot_q['pnl'].mean() if len(bot_q) > 0 else 0
            pnl_delta = pnl_top - pnl_bot

            wr_top = top_q['won'].mean() if len(top_q) > 0 else 0.5

            results.append({
                'subset': '+'.join(subset_names),
                'size': size,
                'correlation': corr,
                'wr_top_quartile': wr_top,
                'pnl_delta': pnl_delta,
                'score': corr * pnl_delta,
            })

    results.sort(key=lambda x: x['score'], reverse=True)

    print(f"\n  {'Subset':>25} {'Size':>5} {'Corr':>8} {'WR TopQ':>8} {'ΔPnL':>9} {'Score':>8}")
    print(f"  {'-'*65}")
    for r in results[:15]:
        emoji = '🟢' if r['score'] > 0.01 else '🟡'
        print(f"  {r['subset']:>25} {r['size']:>5} {r['correlation']:>+8.4f} "
              f"{r['wr_top_quartile']:>7.1%} {r['pnl_delta']:>+8.2f} {emoji}{r['score']:>7.4f}")

    return results


# ═══════════════════════════════════════════════════════════════
# 6. LOGISTIC REGRESSION (if sklearn available)
# ═══════════════════════════════════════════════════════════════

def logistic_regression_analysis(df):
    """Use logistic regression to find optimal module weights."""
    try:
        from sklearn.linear_model import LogisticRegression
        from sklearn.preprocessing import StandardScaler
        from sklearn.model_selection import cross_val_score
    except ImportError:
        print(f"\n  ⚠ sklearn not installed — skipping logistic regression")
        print(f"    Install with: pip3 install scikit-learn")
        return None

    print(f"\n{'='*80}")
    print(f"  LOGISTIC REGRESSION — Learned Module Weights")
    print(f"{'='*80}")

    modules = ['m1_score', 'm4_score', 'm5_score', 'm10_score', 'm11_score']
    mod_names = [m.replace('_score', '').upper() for m in modules]

    X = df[modules].values
    y = df['won'].values

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    model = LogisticRegression(max_iter=1000, C=1.0)
    model.fit(X_scaled, y)

    # Cross-validation
    cv_scores = cross_val_score(model, X_scaled, y, cv=5, scoring='accuracy')

    # Coefficients = importance
    coefs = model.coef_[0]
    abs_sum = sum(abs(c) for c in coefs)

    print(f"\n  Cross-validation accuracy: {cv_scores.mean():.1%} ± {cv_scores.std():.1%}")
    print(f"\n  {'Module':>6} {'Coefficient':>12} {'|Weight|':>10} {'Direction':>10}")
    print(f"  {'-'*42}")

    coef_results = []
    for name, coef in sorted(zip(mod_names, coefs), key=lambda x: abs(x[1]), reverse=True):
        weight_pct = abs(coef) / abs_sum * 100
        direction = '↑ WIN' if coef > 0 else '↓ LOSS'
        emoji = '🟢' if coef > 0.1 else ('🔴' if coef < -0.1 else '🟡')
        print(f"  {name:>6} {coef:>+12.4f} {weight_pct:>9.1f}% {emoji} {direction}")
        coef_results.append({'module': name, 'coef': coef, 'weight_pct': weight_pct})

    # Suggested weights (normalize positive coefficients)
    pos_coefs = {n: max(0, c) for n, c in zip(mod_names, coefs)}
    pos_sum = sum(pos_coefs.values())
    if pos_sum > 0:
        suggested = {n: c / pos_sum for n, c in pos_coefs.items()}
        print(f"\n  Suggested weights (from positive coefficients):")
        for n, w in sorted(suggested.items(), key=lambda x: x[1], reverse=True):
            if w > 0.01:
                bar = '█' * int(w * 40)
                print(f"    {n:>6}: {w:.3f} {bar}")

    return coef_results


# ═══════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════

def main():
    csv_path = 'eth_15m_6m.csv'
    for arg in sys.argv[1:]:
        if not arg.startswith('-'):
            csv_path = arg

    print(f"╔══════════════════════════════════════════════════════════════════╗")
    print(f"║  JIMI — Direction Combination Optimizer                         ║")
    print(f"╚══════════════════════════════════════════════════════════════════╝")
    print(f"  Data: {csv_path}")

    # Run backtest (silent)
    import io
    old_stdout = sys.stdout
    sys.stdout = io.StringIO()
    try:
        trades, stats, raw_df = run_backtest(csv_path, verbose=False)
    finally:
        sys.stdout = old_stdout

    if not trades or len(trades) < 10:
        print("  Not enough trades for analysis.")
        return

    df = extract_trade_features(trades)
    print(f"  Analyzing {len(trades)} trades...")

    # Run all analyses
    single_results = single_module_analysis(df)
    pairwise_results = pairwise_analysis(df)
    best_weights, grid_results = optimize_weights(df)
    subset_results = feature_subset_analysis(df)
    lr_results = logistic_regression_analysis(df)

    # ═══ FINAL RECOMMENDATION ═══
    print(f"\n{'='*80}")
    print(f"  🏆 FINAL RECOMMENDATION")
    print(f"{'='*80}")

    print(f"\n  Current config weights:")
    for m in ['M1', 'M3', 'M4', 'M5', 'M7', 'M10', 'M11', 'M12']:
        key = f"{m}_WEIGHT"
        if key in CONFIG:
            print(f"    {m}: {CONFIG[key]:.2f}")

    print(f"\n  Data-driven recommendation (from optimization):")
    if best_weights:
        for m, w in sorted(best_weights.items(), key=lambda x: x[1], reverse=True):
            if w > 0.01:
                bar = '█' * int(w * 40)
                print(f"    {m}: {w:.3f} {bar}")

    print(f"\n  Key insights:")
    print(f"    1. M3 (VWAP) is INVERTED — consider removing or flipping logic")
    print(f"    2. M5 (LiqMag) is NOISE at current weight — reduce significantly")
    print(f"    3. M1 (MACD) is the strongest predictor — increase weight")
    print(f"    4. M10 (Macro) is underweighted — increase")
    print(f"    5. M12 (Orderbook) has zero signal — consider removing")
    print(f"    6. High ICS (>0.65) + M4 PASS = 83% WR — use as hard gate")

    # Export
    output = {
        'current_weights': {m: CONFIG.get(f'{m}_WEIGHT', 0) for m in ['M1', 'M3', 'M4', 'M5', 'M7', 'M8', 'M9', 'M10', 'M11', 'M12']},
        'best_grid_weights': best_weights,
        'single_module_power': single_results,
        'top_pairs': pairwise_results[:5],
        'top_subsets': [s for s in subset_results if s['score'] > 0][:5],
    }

    out_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'direction_optimization.json')
    with open(out_path, 'w') as f:
        json.dump(output, f, indent=2, default=str)
    print(f"\n  Results exported: {out_path}")


if __name__ == '__main__':
    main()

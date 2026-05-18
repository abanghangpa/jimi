#!/usr/bin/env python3
"""
Dimension-level analysis: which M22 inputs actually predict 24h ETH returns?

Tests each dimension independently, then finds the minimal combination.
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

# ── Load release data (same as backtest_m22_proposal.py) ──
from backtest_m22_proposal import (
    PPI_RELEASES, CPI_RELEASES, NFP_RELEASES,
    FED_FUNDS_MONTHLY, CLAIMS_MONTHLY,
    load_eth_data, compute_return, get_claims_for_date, get_fed_rate,
)


def classify_ppi_dir(ppi_yoy, prior):
    d = ppi_yoy - prior
    if d > 0.2: return 'RISING'
    elif d < -0.2: return 'FALLING'
    return 'FLAT'

def classify_cpi(cpi_yoy):
    if cpi_yoy >= 3.5: return 'HOT'
    elif cpi_yoy >= 2.5: return 'WARM'
    return 'COOL'

def classify_fed(fed_rate, ppi_yoy):
    real = fed_rate - ppi_yoy
    if real > 1.0: return 'HIKING'
    elif real < -0.5: return 'CUTTING'
    return 'HOLDING'

def classify_labor(claims):
    if claims < 210: return 'GOLDILOCKS'
    elif claims < 240: return 'NORMAL'
    elif claims < 280: return 'SOFTENING'
    return 'CRISIS'


def build_dataset(df):
    """Build unified dataset with all dimension values and 24h returns."""
    releases = []

    for date, data in PPI_RELEASES.items():
        releases.append({
            'date': date, 'type': 'PPI',
            'actual': data['ppi_yoy'], 'consensus': data['consensus_yoy'],
            'prior': data['prior_yoy'],
        })
    for date, data in CPI_RELEASES.items():
        releases.append({
            'date': date, 'type': 'CPI',
            'actual': data['cpi_yoy'], 'consensus': data['consensus_yoy'],
            'prior': data['prior_yoy'],
        })
    for date, data in NFP_RELEASES.items():
        releases.append({
            'date': date, 'type': 'NFP',
            'actual': data['nfp_k'], 'consensus': data['consensus_k'],
            'prior': data['prev_k'],
        })

    releases.sort(key=lambda x: x['date'])
    results = []

    # Build PPI/CPI lookup for NFP context
    ppi_by_date = {r['date']: r for r in releases if r['type'] == 'PPI'}
    cpi_by_date = {r['date']: r for r in releases if r['type'] == 'CPI'}

    def latest_before(date, type_filter):
        for r in reversed(releases):
            if r['date'] < date and r['type'] == type_filter:
                return r
        return None

    for rel in releases:
        ret_24h = compute_return(df, rel['date'], hours=24)
        if ret_24h is None:
            continue

        claims = get_claims_for_date(rel['date'])
        fed_rate = get_fed_rate(rel['date'])

        # Get PPI/CPI context
        if rel['type'] == 'PPI':
            ppi_yoy, ppi_prior = rel['actual'], rel['prior']
            cpi_ref = latest_before(rel['date'], 'CPI')
            cpi_yoy = cpi_ref['actual'] if cpi_ref else ppi_yoy
        elif rel['type'] == 'CPI':
            cpi_yoy = rel['actual']
            ppi_ref = latest_before(rel['date'], 'PPI')
            ppi_yoy = ppi_ref['actual'] if ppi_ref else cpi_yoy
            ppi_prior = ppi_ref['prior'] if ppi_ref else cpi_yoy
        else:  # NFP
            ppi_ref = latest_before(rel['date'], 'PPI')
            cpi_ref = latest_before(rel['date'], 'CPI')
            ppi_yoy = ppi_ref['actual'] if ppi_ref else 2.0
            ppi_prior = ppi_ref['prior'] if ppi_ref else 2.0
            cpi_yoy = cpi_ref['actual'] if cpi_ref else 2.0

        # Raw continuous values (not classified)
        real_rate = fed_rate - ppi_yoy
        ppi_delta = ppi_yoy - ppi_prior
        inflation_avg = (ppi_yoy + cpi_yoy) / 2
        surprise = 0.0
        if rel.get('consensus') is not None:
            surprise = rel['actual'] - rel['consensus']

        # Classified values
        ppi_dir = classify_ppi_dir(ppi_yoy, ppi_prior)
        cpi_class = classify_cpi(cpi_yoy)
        fed_class = classify_fed(fed_rate, ppi_yoy)
        labor_class = classify_labor(claims)

        results.append({
            'date': rel['date'],
            'type': rel['type'],
            'return_24h': ret_24h,

            # Raw continuous values
            'ppi_yoy': ppi_yoy,
            'ppi_delta': ppi_delta,
            'cpi_yoy': cpi_yoy,
            'inflation_avg': inflation_avg,
            'fed_rate': fed_rate,
            'real_rate': real_rate,
            'claims': claims,
            'surprise': surprise,

            # Classified
            'ppi_dir': ppi_dir,
            'cpi_class': cpi_class,
            'fed_class': fed_class,
            'labor_class': labor_class,
        })

    return pd.DataFrame(results)


def test_categorical(df, col, label):
    """Test if a categorical dimension predicts returns."""
    groups = df.groupby(col)['return_24h']
    group_stats = groups.agg(['mean', 'count', 'std'])

    # ANOVA (are group means different?)
    group_list = [g['return_24h'].values for _, g in df.groupby(col) if len(g) >= 3]
    if len(group_list) >= 2:
        f_stat, p_val = stats.f_oneway(*group_list)
    else:
        f_stat, p_val = 0, 1

    # Best group spread
    means = group_stats['mean']
    spread = means.max() - means.min()

    print(f"\n  {label} (categorical):")
    print(f"    ANOVA: F={f_stat:.2f}  p={p_val:.4f}  {'✅ significant' if p_val < 0.10 else '❌ not significant'}")
    for cat, row in group_stats.iterrows():
        arrow = '🟢' if row['mean'] > 0.3 else '🔴' if row['mean'] < -0.3 else '⚪'
        print(f"    {arrow} {cat:12s}  avg={row['mean']:+.2f}%  n={int(row['count'])}")
    print(f"    Spread: {spread:+.2f}%")
    return p_val, spread


def test_continuous(df, col, label):
    """Test if a continuous dimension predicts returns."""
    subset = df[[col, 'return_24h']].dropna()
    if len(subset) < 10:
        print(f"\n  {label} (continuous): insufficient data")
        return 1.0, 0

    corr, p_val = stats.pearsonr(subset[col], subset['return_24h'])

    # Quartile analysis
    subset['quartile'] = pd.qcut(subset[col], 4, labels=['Q1(low)', 'Q2', 'Q3', 'Q4(high)'], duplicates='drop')
    quartile_means = subset.groupby('quartile')['return_24h'].mean()
    spread = quartile_means.max() - quartile_means.min()

    print(f"\n  {label} (continuous):")
    print(f"    Correlation: r={corr:.3f}  p={p_val:.4f}  {'✅ significant' if p_val < 0.10 else '❌ not significant'}")
    for q, m in quartile_means.items():
        arrow = '🟢' if m > 0.3 else '🔴' if m < -0.3 else '⚪'
        print(f"    {arrow} {str(q):10s}  avg={m:+.2f}%")
    print(f"    Q4-Q1 spread: {spread:+.2f}%")
    return p_val, spread


def test_by_type(df, col, label, is_continuous=True):
    """Test dimension by release type."""
    print(f"\n    {label} by release type:")
    for rtype in ['PPI', 'CPI', 'NFP']:
        subset = df[df['type'] == rtype]
        if len(subset) < 10:
            continue
        if is_continuous:
            corr, p_val = stats.pearsonr(subset[col], subset['return_24h'])
            sig = '✅' if p_val < 0.10 else '❌'
            print(f"      {rtype:3s}: r={corr:+.3f}  p={p_val:.4f} {sig}  (n={len(subset)})")
        else:
            groups = [g['return_24h'].values for _, g in subset.groupby(col) if len(g) >= 3]
            if len(groups) >= 2:
                f_stat, p_val = stats.f_oneway(*groups)
                sig = '✅' if p_val < 0.10 else '❌'
                print(f"      {rtype:3s}: F={f_stat:.2f}  p={p_val:.4f} {sig}  (n={len(subset)})")


def minimal_model_test(df):
    """Find the minimal set of dimensions that maximizes predictive power."""
    print("\n" + "─" * 70)
    print("  MINIMAL MODEL — Stepwise dimension selection")
    print("─" * 70)

    from scipy.stats import spearmanr

    # All continuous candidates
    candidates = {
        'ppi_yoy': 'PPI level',
        'ppi_delta': 'PPI momentum',
        'cpi_yoy': 'CPI level',
        'inflation_avg': 'Avg inflation (PPI+CPI)',
        'real_rate': 'Real rate (Fed-PPI)',
        'claims': 'Jobless claims',
        'surprise': 'Consensus surprise',
        'fed_rate': 'Fed funds rate',
    }

    # Individual correlations
    print(f"\n  Individual dimension correlations with 24h return:")
    print(f"  {'Dimension':<25} {'r':>8} {'p-value':>8} {'Spearman':>8} {'p':>8}")
    print(f"  {'─'*25} {'─'*8} {'─'*8} {'─'*8} {'─'*8}")

    dims = []
    for col, label in candidates.items():
        subset = df[[col, 'return_24h']].dropna()
        if len(subset) < 10:
            continue
        r, p = stats.pearsonr(subset[col], subset['return_24h'])
        rs, ps = spearmanr(subset[col], subset['return_24h'])
        sig = '✅' if p < 0.10 else ''
        print(f"  {label:<25} {r:>+7.3f} {p:>7.4f} {rs:>+7.3f} {ps:>7.4f} {sig}")
        dims.append((col, label, abs(r), p, r))

    # Sort by |r|
    dims.sort(key=lambda x: x[2], reverse=True)

    # Stepwise: add dimensions and track combined R²
    print(f"\n  Stepwise combination (adding one dimension at a time):")
    print(f"  {'Step':<6} {'Add dimension':<25} {'Cumul r':>8} {'Cumul p':>8} {'ΔR²':>8}")

    from sklearn.linear_model import LinearRegression
    from sklearn.metrics import r2_score

    used = []
    prev_r2 = 0

    for i, (col, label, abs_r, p, r) in enumerate(dims):
        used.append(col)
        X = df[used].dropna()
        y = df.loc[X.index, 'return_24h']
        if len(X) < 10:
            continue

        model = LinearRegression()
        model.fit(X, y)
        y_pred = model.predict(X)
        r2 = r2_score(y, y_pred)
        corr_combined = np.corrcoef(y_pred, y)[0, 1]
        delta_r2 = r2 - prev_r2

        sig = '✅' if delta_r2 > 0.01 else ''
        print(f"  {i+1:<6} {label:<25} {corr_combined:>+7.3f} {'':>8} {delta_r2:>+7.4f} {sig}")
        prev_r2 = r2

    # Test: what if we only use the top 3?
    top3 = [d[0] for d in dims[:3]]
    X = df[top3].dropna()
    y = df.loc[X.index, 'return_24h']
    model = LinearRegression()
    model.fit(X, y)
    y_pred = model.predict(X)
    r2 = r2_score(y, y_pred)
    corr3 = np.corrcoef(y_pred, y)[0, 1]

    # OOS test
    train = df[df['date'] < '2024-01-01']
    test = df[df['date'] >= '2024-01-01']

    X_train = train[top3].dropna()
    y_train = train.loc[X_train.index, 'return_24h']
    X_test = test[top3].dropna()
    y_test = test.loc[X_test.index, 'return_24h']

    model = LinearRegression()
    model.fit(X_train, y_train)
    y_pred_test = model.predict(X_test)
    oos_r2 = r2_score(y_test, y_pred_test)
    oos_corr = np.corrcoef(y_pred_test, y_test)[0, 1]

    print(f"\n  Top-3 model ({', '.join(top3)}):")
    print(f"    Full-sample r={corr3:.3f}  R²={r2:.4f}")
    print(f"    OOS (2024-2026) r={oos_corr:.3f}  R²={oos_r2:.4f}")


def main():
    print("Loading ETH data...")
    df = load_eth_data('eth_15m_merged.csv')
    data = build_dataset(df)

    print(f"\nDataset: {len(data)} releases, {data['date'].min()} → {data['date'].max()}")

    # ═══════════════════════════════════════════════════════════════
    # DIMENSION-BY-DIMENSION ANALYSIS
    # ═══════════════════════════════════════════════════════════════

    print("\n" + "=" * 70)
    print("  DIMENSION-BY-DIMENSION ANALYSIS")
    print("  Each dimension tested independently against 24h ETH return")
    print("=" * 70)

    results = {}

    # ── D1: PPI Direction (classified) ──
    p, s = test_categorical(data, 'ppi_dir', 'D1: PPI Direction')
    results['ppi_dir'] = ('categorical', p, s)
    test_by_type(data, 'ppi_dir', 'D1', is_continuous=False)

    # ── D2: CPI Class (classified) ──
    p, s = test_categorical(data, 'cpi_class', 'D2: CPI Confirmation')
    results['cpi_class'] = ('categorical', p, s)
    test_by_type(data, 'cpi_class', 'D2', is_continuous=False)

    # ── D3: Fed Stance (classified) ──
    p, s = test_categorical(data, 'fed_class', 'D3: Fed Stance')
    results['fed_class'] = ('categorical', p, s)
    test_by_type(data, 'fed_class', 'D3', is_continuous=False)

    # ── D4: Labor Context (classified) ──
    p, s = test_categorical(data, 'labor_class', 'D4: Labor Context')
    results['labor_class'] = ('categorical', p, s)
    test_by_type(data, 'labor_class', 'D4', is_continuous=False)

    # ── D5: PPI Level (continuous) ──
    p, s = test_continuous(data, 'ppi_yoy', 'D5: PPI Level (YoY%)')
    results['ppi_yoy'] = ('continuous', p, s)
    test_by_type(data, 'ppi_yoy', 'D5')

    # ── D6: PPI Momentum (delta) ──
    p, s = test_continuous(data, 'ppi_delta', 'D6: PPI Momentum (Δ from prior)')
    results['ppi_delta'] = ('continuous', p, s)
    test_by_type(data, 'ppi_delta', 'D6')

    # ── D7: CPI Level (continuous) ──
    p, s = test_continuous(data, 'cpi_yoy', 'D7: CPI Level (YoY%)')
    results['cpi_yoy'] = ('continuous', p, s)
    test_by_type(data, 'cpi_yoy', 'D7')

    # ── D8: Inflation Average (PPI+CPI) ──
    p, s = test_continuous(data, 'inflation_avg', 'D8: Avg Inflation (PPI+CPI)/2')
    results['inflation_avg'] = ('continuous', p, s)
    test_by_type(data, 'inflation_avg', 'D8')

    # ── D9: Real Rate ──
    p, s = test_continuous(data, 'real_rate', 'D9: Real Rate (Fed - PPI)')
    results['real_rate'] = ('continuous', p, s)
    test_by_type(data, 'real_rate', 'D9')

    # ── D10: Claims ──
    p, s = test_continuous(data, 'claims', 'D10: Jobless Claims (K)')
    results['claims'] = ('continuous', p, s)
    test_by_type(data, 'claims', 'D10')

    # ── D11: Consensus Surprise ──
    p, s = test_continuous(data, 'surprise', 'D11: Consensus Surprise (actual - consensus)')
    results['surprise'] = ('continuous', p, s)
    test_by_type(data, 'surprise', 'D11')

    # ── D12: Fed Funds Rate ──
    p, s = test_continuous(data, 'fed_rate', 'D12: Fed Funds Rate')
    results['fed_rate'] = ('continuous', p, s)
    test_by_type(data, 'fed_rate', 'D12')

    # ═══════════════════════════════════════════════════════════════
    # RANKING
    # ═══════════════════════════════════════════════════════════════

    print("\n" + "=" * 70)
    print("  DIMENSION RANKING — by predictive power")
    print("=" * 70)

    ranked = sorted(results.items(), key=lambda x: x[1][1])  # sort by p-value
    print(f"\n  {'Rank':<5} {'Dimension':<25} {'Type':>12} {'p-value':>8} {'Spread':>8} {'Verdict'}")
    print(f"  {'─'*5} {'─'*25} {'─'*12} {'─'*8} {'─'*8} {'─'*12}")
    for i, (name, (dtype, p, spread)) in enumerate(ranked):
        verdict = '✅ KEEP' if p < 0.10 else '⚠️ WEAK' if p < 0.25 else '❌ DROP'
        print(f"  {i+1:<5} {name:<25} {dtype:>12} {p:>7.4f} {spread:>+7.2f}%  {verdict}")

    # ═══════════════════════════════════════════════════════════════
    # MINIMAL MODEL
    # ═══════════════════════════════════════════════════════════════

    minimal_model_test(data)

    # ═══════════════════════════════════════════════════════════════
    # INTERACTION EFFECTS
    # ═══════════════════════════════════════════════════════════════

    print("\n" + "=" * 70)
    print("  INTERACTION EFFECTS — do dimensions combine non-linearly?")
    print("=" * 70)

    # PPI direction × CPI class
    print("\n  PPI direction × CPI class:")
    for ppi_dir in ['RISING', 'FALLING', 'FLAT']:
        for cpi_class in ['HOT', 'WARM', 'COOL']:
            subset = data[(data['ppi_dir'] == ppi_dir) & (data['cpi_class'] == cpi_class)]
            if len(subset) >= 3:
                avg = subset['return_24h'].mean()
                arrow = '🟢' if avg > 0.5 else '🔴' if avg < -0.5 else '⚪'
                print(f"    {arrow} {ppi_dir:8s} × {cpi_class:4s}  avg={avg:+.2f}%  n={len(subset)}")

    # Real rate × Fed stance
    print("\n  Real rate × Fed stance:")
    data['real_bucket'] = pd.cut(data['real_rate'], bins=[-10, -1, 0, 1, 10],
                                  labels=['STIM', 'MILD_STIM', 'MILD_TIGHT', 'TIGHT'])
    for fed in ['CUTTING', 'HOLDING', 'HIKING']:
        for real in ['STIM', 'MILD_STIM', 'MILD_TIGHT', 'TIGHT']:
            subset = data[(data['fed_class'] == fed) & (data['real_bucket'] == real)]
            if len(subset) >= 3:
                avg = subset['return_24h'].mean()
                arrow = '🟢' if avg > 0.5 else '🔴' if avg < -0.5 else '⚪'
                print(f"    {arrow} {fed:8s} × {str(real):10s}  avg={avg:+.2f}%  n={len(subset)}")

    # ═══════════════════════════════════════════════════════════════
    # FINAL VERDICT
    # ═══════════════════════════════════════════════════════════════

    print("\n" + "=" * 70)
    print("  VERDICT — What to keep, what to cut")
    print("=" * 70)

    sig_dims = [name for name, (_, p, _) in results.items() if p < 0.10]
    weak_dims = [name for name, (_, p, _) in results.items() if 0.10 <= p < 0.25]
    noise_dims = [name for name, (_, p, _) in results.items() if p >= 0.25]

    print(f"\n  ✅ Significant (p<0.10): {', '.join(sig_dims) if sig_dims else 'NONE'}")
    print(f"  ⚠️ Weak (0.10≤p<0.25):   {', '.join(weak_dims) if weak_dims else 'NONE'}")
    print(f"  ❌ Noise (p≥0.25):       {', '.join(noise_dims) if noise_dims else 'NONE'}")

    print(f"\n  Recommendation: {'Keep only: ' + ', '.join(sig_dims) if sig_dims else 'No dimension is individually significant — consider the minimal model above.'}")


if __name__ == '__main__':
    main()

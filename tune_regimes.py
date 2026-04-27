"""
Regime Threshold Tuner — Grid search over key parameters
to maximize forward-return discrimination between regimes.
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

import pandas as pd
import numpy as np
from itertools import product

# ═══════════════════════════════════════════════════════════════
# LOAD DATA ONCE
# ═══════════════════════════════════════════════════════════════

print("Loading data...")
df_15m = pd.read_csv('eth_15m_6m.csv')
df_15m['Open time'] = pd.to_datetime(df_15m['Open time'])
for col in ['Open', 'High', 'Low', 'Close', 'Volume']:
    df_15m[col] = pd.to_numeric(df_15m[col], errors='coerce')
df_15m = df_15m.dropna(subset=['Open', 'High', 'Low', 'Close'])
df_15m = df_15m.sort_values('Open time').reset_index(drop=True)

df_1h = df_15m.set_index('Open time').resample('1h').agg({
    'Open': 'first', 'High': 'max', 'Low': 'min', 'Close': 'last', 'Volume': 'sum',
}).dropna().reset_index()

closes_15m = df_15m['Close'].values
closes_1h = df_1h['Close'].values

# Pre-compute all raw signals for every bar
from modules.m9_volatility import (
    _compute_atr_percentile, _compute_bb_percentile, _compute_directionality,
    _compute_whipsaw_rate, _compute_retracement_ratio, _compute_volume_confirmation,
    _compute_range_tightness, _compute_1h_15m_coherence
)

print("Pre-computing signals for all bars...")

# Build 1H index mapping
df_1h_times = df_1h['Open time'].values
idx_1h_map = np.full(len(df_15m), -1, dtype=int)
j = 0
for i in range(len(df_15m)):
    bar_time = df_15m['Open time'].iloc[i]
    while j + 1 < len(df_1h) and df_1h_times[j + 1] <= bar_time:
        j += 1
    if df_1h_times[j] <= bar_time:
        idx_1h_map[i] = j

signals_all = []
for i in range(len(df_15m)):
    idx_1h = idx_1h_map[i]
    if idx_1h < 20 or i < 40:
        signals_all.append(None)
        continue

    atr_pctl = _compute_atr_percentile(df_1h, idx_1h)
    bb_pctl = _compute_bb_percentile(df_1h, idx_1h)
    dir_1h = _compute_directionality(df_1h['Close'].iloc[:idx_1h + 1])
    dir_15m = _compute_directionality(df_15m['Close'].iloc[:i + 1])
    whipsaw = _compute_whipsaw_rate(df_15m['Close'].iloc[:i + 1])
    retrace = _compute_retracement_ratio(df_15m['Close'].iloc[:i + 1])
    vol_conf = _compute_volume_confirmation(df_1h, idx_1h)
    range_tight = _compute_range_tightness(df_15m['Close'].iloc[:i + 1])
    tf_coh = _compute_1h_15m_coherence(df_15m, df_1h, i, idx_1h)

    # Structure score
    if idx_1h >= 10:
        highs = df_1h['High'].iloc[idx_1h - 10:idx_1h + 1].values
        lows_arr = df_1h['Low'].iloc[idx_1h - 10:idx_1h + 1].values
        hh = sum(1 for k in range(1, len(highs)) if highs[k] > highs[k - 1])
        ll = sum(1 for k in range(1, len(lows_arr)) if lows_arr[k] < lows_arr[k - 1])
        structure = abs(hh - ll) / max(hh + ll, 1)
    else:
        structure = 0.0

    directionality = dir_1h * 0.4 + dir_15m * 0.6

    signals_all.append({
        'atr_pctl': atr_pctl, 'bb_pctl': bb_pctl,
        'directionality': directionality, 'directionality_1h': dir_1h, 'directionality_15m': dir_15m,
        'whipsaw_rate': whipsaw, 'retrace_ratio': retrace,
        'volume_confirm': vol_conf, 'range_tight': range_tight,
        'tf_coherence': tf_coh, 'structure_score': structure,
    })

print(f"Pre-computed signals for {sum(1 for s in signals_all if s is not None)} bars")

# Forward returns (4H = 16 bars)
fwd_4h = np.full(len(df_15m), np.nan)
for i in range(len(df_15m) - 16):
    fwd_4h[i] = (closes_15m[i + 16] - closes_15m[i]) / closes_15m[i] * 100

# ═══════════════════════════════════════════════════════════════
# REGIME CLASSIFIER WITH TUNABLE PARAMS
# ═══════════════════════════════════════════════════════════════

def classify_regimes(params):
    """Classify all bars with given parameters. Returns array of regime labels."""
    cw = params['chop_weights']  # (whipsaw, retrace, 1-dir, 1-vol_conf, 1-tf_coh)
    tw = params['trend_weights']  # (dir, structure, 1-whipsaw, 1-retrace, vol_conf, tf_coh)
    
    regimes = np.empty(len(df_15m), dtype='U14')
    regimes[:] = 'UNKNOWN'
    
    for i, sig in enumerate(signals_all):
        if sig is None:
            continue
        
        # Composite scores
        chop_score = (
            sig['whipsaw_rate'] * cw[0] +
            sig['retrace_ratio'] * cw[1] +
            (1.0 - sig['directionality']) * cw[2] +
            (1.0 - sig['volume_confirm']) * cw[3] +
            (1.0 - sig['tf_coherence']) * cw[4]
        )
        
        trend_score = (
            sig['directionality'] * tw[0] +
            sig['structure_score'] * tw[1] +
            (1.0 - sig['whipsaw_rate']) * tw[2] +
            (1.0 - sig['retrace_ratio']) * tw[3] +
            sig['volume_confirm'] * tw[4] +
            sig['tf_coherence'] * tw[5]
        )
        
        # Classification with tunable thresholds
        if sig['atr_pctl'] > params['crisis_atr'] or sig['bb_pctl'] > params['crisis_bb']:
            regimes[i] = 'CRISIS'
        elif (chop_score > params['chop_hard_cs'] and 
              sig['whipsaw_rate'] > params['chop_hard_wr'] and 
              sig['retrace_ratio'] > params['chop_hard_rr']):
            regimes[i] = 'CHOP_HARD'
        elif chop_score > params['chop_mild_cs'] and trend_score < params['chop_mild_ts']:
            regimes[i] = 'CHOP_MILD'
        elif (sig['bb_pctl'] < params['comp_bb'] and 
              sig['atr_pctl'] < params['comp_atr'] and 
              sig['range_tight'] > params['comp_rt']):
            regimes[i] = 'COMPRESSING'
        elif (trend_score > params['trend_ts'] and 
              sig['directionality'] > params['trend_dir'] and
              sig['whipsaw_rate'] < params['trend_wr'] and 
              sig['retrace_ratio'] < params['trend_rr']):
            regimes[i] = 'TRENDING'
        else:
            regimes[i] = 'NEUTRAL'
    
    return regimes


def score_params(params):
    """Score a parameter set by regime discrimination quality."""
    regimes = classify_regimes(params)
    
    # Only evaluate bars with valid forward returns
    valid = ~np.isnan(fwd_4h) & (regimes != 'UNKNOWN')
    
    results = {}
    for regime in ['CRISIS', 'CHOP_HARD', 'CHOP_MILD', 'COMPRESSING', 'TRENDING', 'NEUTRAL']:
        mask = valid & (regimes == regime)
        n = mask.sum()
        if n < 10:
            results[regime] = {'n': n, 'mean': 0, 'std': 1, 'sharpe': 0, 'win': 0}
            continue
        
        r = fwd_4h[mask]
        results[regime] = {
            'n': n,
            'mean': np.mean(r),
            'std': np.std(r),
            'sharpe': np.mean(r) / np.std(r) if np.std(r) > 0 else 0,
            'win': (r > 0).mean(),
        }
    
    # Scoring metrics
    regime_counts = {r: d['n'] for r, d in results.items()}
    n_active = sum(1 for n in regime_counts.values() if n >= 50)
    
    # Sharpe spread: best regime sharpe - worst regime sharpe (only regimes with n>=50)
    sharpes = [d['sharpe'] for r, d in results.items() if d['n'] >= 50]
    sharpe_spread = max(sharpes) - min(sharpes) if sharpes else 0
    
    # Mean absolute return spread
    means = [d['mean'] for r, d in results.items() if d['n'] >= 50]
    return_spread = max(means) - min(means) if means else 0
    
    # TRENDING regime quality (if detected)
    trending = results.get('TRENDING', {})
    trending_quality = 0
    if trending.get('n', 0) >= 50:
        trending_quality = trending['sharpe'] * 2 + trending['mean'] * 10 + trending['win'] * 5
    
    # COMPRESSING regime quality
    compressing = results.get('COMPRESSING', {})
    comp_quality = 0
    if compressing.get('n', 0) >= 20:
        comp_quality = compressing['sharpe'] * 2 + compressing['mean'] * 10 + compressing['win'] * 5
    
    # Penalize if only 1-2 active regimes (no discrimination)
    diversity_bonus = n_active * 0.5
    
    # Penalize if TRENDING is never detected
    trending_penalty = 0
    if trending.get('n', 0) < 50:
        trending_penalty = -2.0
    
    # Overall score
    score = (
        sharpe_spread * 3.0 +
        return_spread * 2.0 +
        trending_quality * 1.5 +
        comp_quality * 1.0 +
        diversity_bonus +
        trending_penalty
    )
    
    return score, results, regime_counts


# ═══════════════════════════════════════════════════════════════
# GRID SEARCH
# ═══════════════════════════════════════════════════════════════

# Default weights from original code
default_chop = (0.30, 0.25, 0.20, 0.15, 0.10)
default_trend = (0.30, 0.20, 0.20, 0.15, 0.10, 0.05)

# Parameter grid — focus on the most impactful thresholds
param_grid = {
    # CRISIS thresholds
    'crisis_atr': [0.80, 0.85, 0.90],
    'crisis_bb': [0.80, 0.85, 0.90],
    
    # TRENDING thresholds (relax from defaults)
    'trend_ts': [0.40, 0.45, 0.50],
    'trend_dir': [0.25, 0.30, 0.35],
    'trend_wr': [0.55, 0.60, 0.65],
    'trend_rr': [0.70, 0.80, 0.90],
    
    # CHOP_MILD thresholds
    'chop_mild_cs': [0.50, 0.55, 0.60],
    'chop_mild_ts': [0.30, 0.35, 0.40],
}

# Full default param dict
default_params = {
    'crisis_atr': 0.85, 'crisis_bb': 0.85,
    'chop_hard_cs': 0.72, 'chop_hard_wr': 0.70, 'chop_hard_rr': 0.80,
    'chop_mild_cs': 0.55, 'chop_mild_ts': 0.35,
    'comp_bb': 0.30, 'comp_atr': 0.40, 'comp_rt': 0.60,
    'trend_ts': 0.50, 'trend_dir': 0.35, 'trend_wr': 0.55, 'trend_rr': 0.70,
    'chop_weights': default_chop,
    'trend_weights': default_trend,
}

# First, score the defaults
print("\n" + "="*70)
print("BASELINE (original parameters)")
print("="*70)
default_score, default_results, default_counts = score_params(default_params)
print(f"  Overall score: {default_score:.3f}")
for regime in ['CRISIS', 'CHOP_HARD', 'CHOP_MILD', 'COMPRESSING', 'TRENDING', 'NEUTRAL']:
    d = default_results[regime]
    if d['n'] > 0:
        print(f"  {regime:<14} n={d['n']:>5}  mean={d['mean']:+.3f}%  win={d['win']*100:.1f}%  sharpe={d['sharpe']:+.3f}")

# Now grid search over TRENDING thresholds (the most impactful)
print("\n" + "="*70)
print("GRID SEARCH: TRENDING thresholds (relaxing entry conditions)")
print("="*70)

best_score = default_score
best_params = default_params.copy()
best_results = default_results

total = 0
for trend_ts in param_grid['trend_ts']:
    for trend_dir in param_grid['trend_dir']:
        for trend_wr in param_grid['trend_wr']:
            for trend_rr in param_grid['trend_rr']:
                p = default_params.copy()
                p['trend_ts'] = trend_ts
                p['trend_dir'] = trend_dir
                p['trend_wr'] = trend_wr
                p['trend_rr'] = trend_rr
                
                s, r, c = score_params(p)
                total += 1
                
                if s > best_score:
                    best_score = s
                    best_params = p.copy()
                    best_results = r
                    trending_n = r.get('TRENDING', {}).get('n', 0)
                    trending_sharpe = r.get('TRENDING', {}).get('sharpe', 0)
                    print(f"  NEW BEST: score={s:.3f} | trend_ts={trend_ts} dir={trend_dir} wr={trend_wr} rr={trend_rr} | TRENDING n={trending_n} sharpe={trending_sharpe:+.3f}")

print(f"\n  Tested {total} combinations")

# Grid search over CHOP_MILD thresholds
print("\n" + "="*70)
print("GRID SEARCH: CHOP_MILD thresholds (from best TRENDING params)")
print("="*70)

for chop_cs in param_grid['chop_mild_cs']:
    for chop_ts in param_grid['chop_mild_ts']:
        p = best_params.copy()
        p['chop_mild_cs'] = chop_cs
        p['chop_mild_ts'] = chop_ts
        
        s, r, c = score_params(p)
        total += 1
        
        if s > best_score:
            best_score = s
            best_params = p.copy()
            best_results = r
            print(f"  NEW BEST: score={s:.3f} | chop_cs={chop_cs} chop_ts={chop_ts}")

# Grid search over CRISIS thresholds
print("\n" + "="*70)
print("GRID SEARCH: CRISIS thresholds")
print("="*70)

for crisis_atr in param_grid['crisis_atr']:
    for crisis_bb in param_grid['crisis_bb']:
        p = best_params.copy()
        p['crisis_atr'] = crisis_atr
        p['crisis_bb'] = crisis_bb
        
        s, r, c = score_params(p)
        total += 1
        
        if s > best_score:
            best_score = s
            best_params = p.copy()
            best_results = r
            print(f"  NEW BEST: score={s:.3f} | crisis_atr={crisis_atr} crisis_bb={crisis_bb}")

# Also try weight variants
print("\n" + "="*70)
print("GRID SEARCH: chop/trend weight variants")
print("="*70)

weight_variants = [
    # (chop_weights, trend_weights, label)
    ((0.30, 0.25, 0.20, 0.15, 0.10), (0.30, 0.20, 0.20, 0.15, 0.10, 0.05), "original"),
    ((0.25, 0.15, 0.25, 0.20, 0.15), (0.35, 0.25, 0.15, 0.10, 0.10, 0.05), "dir-heavy"),
    ((0.20, 0.10, 0.30, 0.25, 0.15), (0.40, 0.20, 0.15, 0.10, 0.10, 0.05), "dir+vol"),
    ((0.35, 0.20, 0.20, 0.15, 0.10), (0.25, 0.15, 0.25, 0.15, 0.15, 0.05), "whipsaw-heavy"),
    ((0.25, 0.20, 0.20, 0.20, 0.15), (0.30, 0.25, 0.20, 0.10, 0.10, 0.05), "balanced"),
    # Drop retrace_ratio entirely (it's broken at ~1.0)
    ((0.40, 0.00, 0.25, 0.20, 0.15), (0.35, 0.25, 0.25, 0.00, 0.10, 0.05), "no-retrace"),
    ((0.35, 0.00, 0.30, 0.20, 0.15), (0.40, 0.25, 0.20, 0.00, 0.10, 0.05), "no-retrace-dir"),
]

for cw, tw, label in weight_variants:
    p = best_params.copy()
    p['chop_weights'] = cw
    p['trend_weights'] = tw
    
    s, r, c = score_params(p)
    total += 1
    
    if s > best_score:
        best_score = s
        best_params = p.copy()
        best_results = r
        print(f"  NEW BEST: score={s:.3f} | weights={label}")
    else:
        trending_n = r.get('TRENDING', {}).get('n', 0)
        trending_sharpe = r.get('TRENDING', {}).get('sharpe', 0)
        print(f"  {label:<20} score={s:.3f} (delta={s-best_score:+.3f}) | TRENDING n={trending_n} sharpe={trending_sharpe:+.3f}")


# ═══════════════════════════════════════════════════════════════
# FINAL RESULTS
# ═══════════════════════════════════════════════════════════════

print("\n" + "="*70)
print(f"FINAL RESULT (tested {total} combinations)")
print("="*70)

print(f"\n  Baseline score:  {default_score:.3f}")
print(f"  Best score:      {best_score:.3f}")
print(f"  Improvement:     {best_score - default_score:+.3f}")

print(f"\n  Best parameters:")
for k, v in best_params.items():
    if k in ('chop_weights', 'trend_weights'):
        print(f"    {k}: {tuple(round(x, 2) for x in v)}")
    else:
        default_v = default_params.get(k)
        changed = " ← CHANGED" if v != default_v else ""
        print(f"    {k}: {v}{changed}")

print(f"\n  Regime distribution (best params):")
for regime in ['CRISIS', 'CHOP_HARD', 'CHOP_MILD', 'COMPRESSING', 'TRENDING', 'NEUTRAL']:
    d = best_results[regime]
    n = d['n']
    pct = n / len(df_15m) * 100
    if n > 0:
        print(f"    {regime:<14} n={n:>5} ({pct:>5.1f}%)  mean={d['mean']:+.3f}%  win={d['win']*100:.1f}%  sharpe={d['sharpe']:+.3f}")
    else:
        print(f"    {regime:<14} n=0")

print(f"\n  Default regime distribution (for comparison):")
for regime in ['CRISIS', 'CHOP_HARD', 'CHOP_MILD', 'COMPRESSING', 'TRENDING', 'NEUTRAL']:
    d = default_results[regime]
    n = d['n']
    pct = n / len(df_15m) * 100
    if n > 0:
        print(f"    {regime:<14} n={n:>5} ({pct:>5.1f}%)  mean={d['mean']:+.3f}%  win={d['win']*100:.1f}%  sharpe={d['sharpe']:+.3f}")
    else:
        print(f"    {regime:<14} n=0")

# Sharpe spread analysis
print("\n  Sharpe spread by regime (best params):")
active = [(r, d) for r, d in best_results.items() if d['n'] >= 50]
active.sort(key=lambda x: x[1]['sharpe'], reverse=True)
for regime, d in active:
    bar = "█" * int(max(0, (d['sharpe'] + 1) * 20))
    print(f"    {regime:<14} sharpe={d['sharpe']:+.3f}  {bar}")

if len(active) >= 2:
    best_r = active[0]
    worst_r = active[-1]
    print(f"\n  Best regime:  {best_r[0]} (sharpe={best_r[1]['sharpe']:+.3f})")
    print(f"  Worst regime: {worst_r[0]} (sharpe={worst_r[1]['sharpe']:+.3f})")
    print(f"  Spread:       {best_r[1]['sharpe'] - worst_r[1]['sharpe']:+.3f}")

"""Second-pass tuning: explore no-retrace weights + wider threshold ranges."""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))
import pandas as pd
import numpy as np

# Reuse cached signals from tune_regimes.py
from tune_regimes import (
    df_15m, df_1h, signals_all, fwd_4h, default_params, best_params, score_params, classify_regimes
)

print("="*70)
print("SECOND PASS: No-retrace weights + wider TRENDING gates")
print("="*70)

# The retrace_ratio is ~1.0 for all non-TRENDING regimes.
# Dropping it from chop_score means chop doesn't get a free +0.25 from retrace.
# This should make CHOP harder to trigger → more bars left for TRENDING/NEUTRAL.

second_pass_configs = [
    # label, chop_weights, trend_weights, extra param overrides
    ("no-retrace-A", (0.35, 0.00, 0.25, 0.20, 0.20), (0.35, 0.25, 0.20, 0.00, 0.15, 0.05), {}),
    ("no-retrace-B", (0.30, 0.00, 0.30, 0.20, 0.20), (0.40, 0.25, 0.15, 0.00, 0.15, 0.05), {}),
    ("no-retrace-C", (0.25, 0.00, 0.30, 0.25, 0.20), (0.35, 0.30, 0.15, 0.00, 0.15, 0.05), {}),
    # Even looser TRENDING
    ("loose-trend", (0.25, 0.20, 0.20, 0.20, 0.15), (0.30, 0.25, 0.20, 0.10, 0.10, 0.05),
     {'trend_ts': 0.35, 'trend_dir': 0.20, 'trend_wr': 0.70, 'trend_rr': 0.90}),
    ("loose+no-retrace", (0.30, 0.00, 0.25, 0.25, 0.20), (0.35, 0.30, 0.15, 0.00, 0.15, 0.05),
     {'trend_ts': 0.35, 'trend_dir': 0.20, 'trend_wr': 0.70, 'trend_rr': 0.90}),
    # Ultra-loose TRENDING: just need trend_score > 0.30 and directionality > 0.15
    ("ultra-loose", (0.25, 0.20, 0.20, 0.20, 0.15), (0.30, 0.25, 0.20, 0.10, 0.10, 0.05),
     {'trend_ts': 0.30, 'trend_dir': 0.15, 'trend_wr': 0.75, 'trend_rr': 0.95}),
    ("ultra+no-retrace", (0.30, 0.00, 0.25, 0.25, 0.20), (0.35, 0.30, 0.15, 0.00, 0.15, 0.05),
     {'trend_ts': 0.30, 'trend_dir': 0.15, 'trend_wr': 0.75, 'trend_rr': 0.95}),
]

overall_best = best_params.copy()
overall_best_score = 0
_, overall_best_results, _ = score_params(best_params)

for label, cw, tw, overrides in second_pass_configs:
    p = best_params.copy()
    p['chop_weights'] = cw
    p['trend_weights'] = tw
    p.update(overrides)
    
    s, r, c = score_params(p)
    
    trending = r.get('TRENDING', {})
    compressing = r.get('COMPRESSING', {})
    neutral = r.get('NEUTRAL', {})
    
    print(f"\n  {label:<22} score={s:.3f}")
    print(f"    TRENDING:  n={trending.get('n',0):>4}  mean={trending.get('mean',0):+.3f}%  win={trending.get('win',0)*100:.1f}%  sharpe={trending.get('sharpe',0):+.3f}")
    print(f"    COMPRESS:  n={compressing.get('n',0):>4}  mean={compressing.get('mean',0):+.3f}%  win={compressing.get('win',0)*100:.1f}%  sharpe={compressing.get('sharpe',0):+.3f}")
    print(f"    NEUTRAL:   n={neutral.get('n',0):>4}  mean={neutral.get('mean',0):+.3f}%  win={neutral.get('win',0)*100:.1f}%  sharpe={neutral.get('sharpe',0):+.3f}")
    
    if s > overall_best_score:
        overall_best_score = s
        overall_best = p.copy()
        overall_best_results = r
        print(f"    *** NEW OVERALL BEST ***")

# Also try: what if we use ONLY directionality + volume_confirm as signals?
# (drop the broken retrace_ratio from everything)
print("\n" + "="*70)
print("EXTREME TEST: What if TRENDING only needs directionality + volume?")
print("="*70)

p = best_params.copy()
p['trend_ts'] = 0.30
p['trend_dir'] = 0.15
p['trend_wr'] = 0.80  # almost no whipsaw filter
p['trend_rr'] = 0.99  # essentially disabled
p['chop_mild_cs'] = 0.50

s, r, c = score_params(p)
trending = r.get('TRENDING', {})
print(f"  score={s:.3f} | TRENDING n={trending.get('n',0)} mean={trending.get('mean',0):+.3f}% win={trending.get('win',0)*100:.1f}% sharpe={trending.get('sharpe',0):+.3f}")

if s > overall_best_score:
    overall_best_score = s
    overall_best = p.copy()
    overall_best_results = r
    print(f"  *** NEW OVERALL BEST ***")

# Final summary
print("\n" + "="*70)
print("OVERALL BEST CONFIGURATION")
print("="*70)

print(f"\n  Score: {overall_best_score:.3f} (baseline was 9.421)")
print(f"\n  Parameters:")
for k, v in overall_best.items():
    orig = default_params.get(k)
    changed = " ← CHANGED" if v != orig else ""
    if k in ('chop_weights', 'trend_weights'):
        print(f"    {k}: {tuple(round(x, 2) for x in v)}{changed}")
    else:
        print(f"    {k}: {v}{changed}")

print(f"\n  Regime distribution:")
for regime in ['CRISIS', 'CHOP_HARD', 'CHOP_MILD', 'COMPRESSING', 'TRENDING', 'NEUTRAL']:
    d = overall_best_results[regime]
    n = d['n']
    pct = n / len(df_15m) * 100
    if n > 0:
        print(f"    {regime:<14} n={n:>5} ({pct:>5.1f}%)  mean={d['mean']:+.3f}%  win={d['win']*100:.1f}%  sharpe={d['sharpe']:+.3f}")
    else:
        print(f"    {regime:<14} n=0")

# Sharpe spread
active = [(r, d) for r, d in overall_best_results.items() if d['n'] >= 20]
active.sort(key=lambda x: x[1]['sharpe'], reverse=True)
print(f"\n  Regime ranking by Sharpe:")
for regime, d in active:
    print(f"    {regime:<14} sharpe={d['sharpe']:+.3f}  (n={d['n']})")

if len(active) >= 2:
    print(f"\n  Sharpe spread: {active[0][1]['sharpe'] - active[-1][1]['sharpe']:+.3f}")
    print(f"  Best:  {active[0][0]} ({active[0][1]['sharpe']:+.3f})")
    print(f"  Worst: {active[-1][0]} ({active[-1][1]['sharpe']:+.3f})")

# Generate the diff: what changed from original
print("\n" + "="*70)
print("DIFF: Changes to apply to m9_volatility.py")
print("="*70)
changes = []
for k, v in overall_best.items():
    orig = default_params.get(k)
    if v != orig:
        if k in ('chop_weights', 'trend_weights'):
            changes.append(f"  {k}: {orig} → {tuple(round(x,2) for x in v)}")
        else:
            changes.append(f"  {k}: {orig} → {v}")

for c in changes:
    print(c)

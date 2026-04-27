"""
Regime Detection Validation Script
Runs m9_volatility classifier over historical ETH 15m data and evaluates:
1. Regime distribution
2. Forward return distributions per regime
3. Confusion matrix (detected vs realized)
4. Signal importance via permutation
5. Regime transition statistics
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

import pandas as pd
import numpy as np
from modules.m9_volatility import compute_vol_regime, RegimeState

# ═══════════════════════════════════════════════════════════════
# DATA LOADING
# ═══════════════════════════════════════════════════════════════

def load_data():
    """Load 15m data and resample to 1H."""
    print("Loading 15m data...")
    df_15m = pd.read_csv('eth_15m_6m.csv')
    df_15m['Open time'] = pd.to_datetime(df_15m['Open time'])
    
    for col in ['Open', 'High', 'Low', 'Close', 'Volume']:
        df_15m[col] = pd.to_numeric(df_15m[col], errors='coerce')
    
    df_15m = df_15m.dropna(subset=['Open', 'High', 'Low', 'Close'])
    df_15m = df_15m.sort_values('Open time').reset_index(drop=True)
    
    # Resample to 1H
    df_1h = df_15m.set_index('Open time').resample('1h').agg({
        'Open': 'first', 'High': 'max', 'Low': 'min',
        'Close': 'last', 'Volume': 'sum',
    }).dropna().reset_index()
    
    print(f"  15m: {len(df_15m)} bars ({df_15m['Open time'].iloc[0]} → {df_15m['Open time'].iloc[-1]})")
    print(f"  1H:  {len(df_1h)} bars")
    return df_15m, df_1h


# ═══════════════════════════════════════════════════════════════
# RUN CLASSIFIER OVER ALL BARS
# ═══════════════════════════════════════════════════════════════

def run_classifier(df_15m, df_1h):
    """Classify every bar and return results DataFrame."""
    print("\nRunning regime classifier over all bars...")
    
    regime_state = RegimeState()
    results = []
    
    # Build index mapping: for each 15m bar, find the corresponding 1H index
    df_1h_times = df_1h['Open time'].values
    
    step = max(1, len(df_15m) // 100)  # progress every 1%
    
    for i in range(len(df_15m)):
        if i % step == 0:
            pct = i / len(df_15m) * 100
            print(f"  {pct:.0f}% ({i}/{len(df_15m)})...", end='\r')
        
        bar_time = df_15m['Open time'].iloc[i]
        
        # Find corresponding 1H index
        idx_1h_arr = np.where(df_1h_times <= bar_time)[0]
        if len(idx_1h_arr) == 0:
            continue
        idx_1h = idx_1h_arr[-1]
        
        if idx_1h < 20 or i < 40:
            continue
        
        regime, score, details = compute_vol_regime(
            df_15m, df_1h, i, idx_1h, regime_state=regime_state
        )
        
        results.append({
            'idx': i,
            'time': bar_time,
            'regime': regime,
            'score': score,
            'atr_pctl': details.get('atr_pctl', 0.5),
            'bb_pctl': details.get('bb_pctl', 0.5),
            'directionality': details.get('directionality', 0.5),
            'directionality_1h': details.get('directionality_1h', 0.5),
            'directionality_15m': details.get('directionality_15m', 0.5),
            'whipsaw_rate': details.get('whipsaw_rate', 0.5),
            'retrace_ratio': details.get('retrace_ratio', 0.5),
            'volume_confirm': details.get('volume_confirm', 0.5),
            'range_tight': details.get('range_tight', 0.5),
            'tf_coherence': details.get('tf_coherence', 0.5),
            'structure_score': details.get('structure_score', 0.0),
            'vol_ratio': details.get('vol_ratio', 1.0),
            'raw_regime': details.get('raw_regime', regime),
            'is_transition': details.get('is_transition', False),
            'close': df_15m['Close'].iloc[i],
        })
    
    print(f"  100% — classified {len(results)} bars")
    return pd.DataFrame(results)


# ═══════════════════════════════════════════════════════════════
# FORWARD RETURN ANALYSIS
# ═══════════════════════════════════════════════════════════════

def forward_return_analysis(results_df, df_15m):
    """Compute forward returns for each detected regime."""
    print("\n" + "="*70)
    print("FORWARD RETURN ANALYSIS BY REGIME")
    print("="*70)
    
    closes = df_15m['Close'].values
    
    for horizon_name, horizon_bars in [('1H (4 bars)', 4), ('4H (16 bars)', 16), ('24H (96 bars)', 96)]:
        print(f"\n--- Forward {horizon_name} ---")
        print(f"{'Regime':<14} {'Count':>6} {'Mean%':>8} {'Median%':>8} {'Std%':>8} {'Win%':>7} {'Sharpe':>8} {'Skew':>7}")
        print("-" * 74)
        
        for regime in ['CRISIS', 'CHOP_HARD', 'CHOP_MILD', 'COMPRESSING', 'TRENDING', 'NEUTRAL']:
            mask = results_df['regime'] == regime
            if mask.sum() < 10:
                continue
            
            indices = results_df.loc[mask, 'idx'].values
            valid = indices[indices + horizon_bars < len(closes)]
            
            if len(valid) < 10:
                continue
            
            fwd = (closes[valid + horizon_bars] - closes[valid]) / closes[valid] * 100
            
            win_rate = (fwd > 0).mean() * 100
            mean_r = fwd.mean()
            median_r = np.median(fwd)
            std_r = fwd.std()
            sharpe = mean_r / std_r if std_r > 0 else 0
            skew = float(pd.Series(fwd).skew())
            
            print(f"{regime:<14} {len(valid):>6} {mean_r:>+8.3f} {median_r:>+8.3f} {std_r:>8.3f} {win_rate:>6.1f}% {sharpe:>+8.3f} {skew:>7.2f}")
    
    # Same but by RAW regime (before hysteresis)
    print("\n" + "="*70)
    print("FORWARD RETURN ANALYSIS BY RAW REGIME (before hysteresis)")
    print("="*70)
    for horizon_name, horizon_bars in [('4H (16 bars)', 16)]:
        print(f"\n--- Forward {horizon_name} ---")
        print(f"{'Raw Regime':<14} {'Count':>6} {'Mean%':>8} {'Median%':>8} {'Std%':>8} {'Win%':>7} {'Sharpe':>8}")
        print("-" * 66)
        
        for regime in ['CRISIS', 'CHOP_HARD', 'CHOP_MILD', 'COMPRESSING', 'TRENDING', 'NEUTRAL']:
            mask = results_df['raw_regime'] == regime
            if mask.sum() < 10:
                continue
            
            indices = results_df.loc[mask, 'idx'].values
            valid = indices[indices + horizon_bars < len(closes)]
            
            if len(valid) < 10:
                continue
            
            fwd = (closes[valid + horizon_bars] - closes[valid]) / closes[valid] * 100
            
            win_rate = (fwd > 0).mean() * 100
            mean_r = fwd.mean()
            median_r = np.median(fwd)
            std_r = fwd.std()
            sharpe = mean_r / std_r if std_r > 0 else 0
            
            print(f"{regime:<14} {len(valid):>6} {mean_r:>+8.3f} {median_r:>+8.3f} {std_r:>8.3f} {win_rate:>6.1f}% {sharpe:>+8.3f}")


# ═══════════════════════════════════════════════════════════════
# REGIME DISCRIMINATION TEST
# ═══════════════════════════════════════════════════════════════

def discrimination_test(results_df, df_15m):
    """KS-test: are forward return distributions actually different between regimes?"""
    from scipy import stats
    
    print("\n" + "="*70)
    print("REGIME DISCRIMINATION (Kolmogorov-Smirnov test)")
    print("H0: two regimes have the same forward return distribution")
    print("p < 0.05 → distributions are significantly different")
    print("="*70)
    
    closes = df_15m['Close'].values
    horizon = 16  # 4H
    
    regime_returns = {}
    for regime in ['CRISIS', 'CHOP_HARD', 'CHOP_MILD', 'COMPRESSING', 'TRENDING', 'NEUTRAL']:
        mask = results_df['regime'] == regime
        indices = results_df.loc[mask, 'idx'].values
        valid = indices[indices + horizon < len(closes)]
        if len(valid) >= 20:
            fwd = (closes[valid + horizon] - closes[valid]) / closes[valid] * 100
            regime_returns[regime] = fwd
    
    regimes = list(regime_returns.keys())
    print(f"\n{'':14}", end='')
    for r in regimes:
        print(f"{r:>12}", end='')
    print()
    
    for r1 in regimes:
        print(f"{r1:<14}", end='')
        for r2 in regimes:
            if r1 == r2:
                print(f"{'---':>12}", end='')
            else:
                stat, p = stats.ks_2samp(regime_returns[r1], regime_returns[r2])
                sig = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else "ns"
                print(f"{p:>11.4f}{sig}", end='')
        print()


# ═══════════════════════════════════════════════════════════════
# CONFUSION MATRIX (detected vs realized)
# ═══════════════════════════════════════════════════════════════

def confusion_matrix(results_df, df_15m):
    """Compare detected regime to what actually happened."""
    print("\n" + "="*70)
    print("CONFUSION MATRIX: Detected Regime → Realized Outcome")
    print("="*70)
    
    closes = df_15m['Close'].values
    highs = df_15m['High'].values
    lows = df_15m['Low'].values
    
    # Define realized regime based on what actually happened in next 16 bars (4H)
    horizon = 16
    realized = []
    
    for i in results_df['idx'].values:
        if i + horizon >= len(closes):
            realized.append('UNKNOWN')
            continue
        
        fwd_return = (closes[i + horizon] - closes[i]) / closes[i] * 100
        fwd_range = (highs[i+1:i+horizon+1].max() - lows[i+1:i+horizon+1].min()) / closes[i] * 100
        
        # Compute realized volatility (15m bar returns std over next 16 bars)
        bar_returns = np.diff(np.log(closes[i:i+horizon+1])) * 100
        realized_vol = np.std(bar_returns) if len(bar_returns) > 1 else 0
        
        # Classify realized
        if realized_vol > 1.5:
            realized.append('CRISIS_REAL')
        elif fwd_range < 0.8:
            realized.append('COMPRESS_REAL')
        elif abs(fwd_return) > 1.5 and fwd_range > 0:
            # Strong directional move
            retrace = 1 - abs(fwd_return) / fwd_range
            if retrace < 0.4:
                realized.append('TREND_REAL')
            else:
                realized.append('CHOP_REAL')
        elif abs(fwd_return) < 0.5:
            realized.append('CHOP_REAL')
        else:
            realized.append('NEUTRAL_REAL')
    
    results_df = results_df.copy()
    results_df['realized'] = realized[:len(results_df)]
    
    # Build confusion matrix
    detected_labels = ['CRISIS', 'CHOP_HARD', 'CHOP_MILD', 'COMPRESSING', 'TRENDING', 'NEUTRAL']
    realized_labels = ['CRISIS_REAL', 'CHOP_REAL', 'COMPRESS_REAL', 'TREND_REAL', 'NEUTRAL_REAL']
    
    print(f"\n{'Detected \\ Realized':<18}", end='')
    for rl in realized_labels:
        short = rl.replace('_REAL', '')
        print(f"{short:>10}", end='')
    print(f"{'Total':>8}")
    print("-" * (18 + 10 * len(realized_labels) + 8))
    
    for dl in detected_labels:
        mask = results_df['regime'] == dl
        subset = results_df[mask]
        print(f"{dl:<18}", end='')
        total = len(subset)
        for rl in realized_labels:
            count = (subset['realized'] == rl).sum()
            pct = count / total * 100 if total > 0 else 0
            print(f"{pct:>9.1f}%", end='')
        print(f"{total:>8}")
    
    # Accuracy: was the detected regime "correct"?
    print("\n--- Regime Accuracy (is detected regime consistent with realized?) ---")
    
    mapping = {
        'CRISIS': 'CRISIS_REAL',
        'CHOP_HARD': 'CHOP_REAL',
        'CHOP_MILD': 'CHOP_REAL',
        'COMPRESSING': 'COMPRESS_REAL',
        'TRENDING': 'TREND_REAL',
        'NEUTRAL': 'NEUTRAL_REAL',
    }
    
    for dl in detected_labels:
        mask = results_df['regime'] == dl
        subset = results_df[mask]
        if len(subset) == 0:
            continue
        expected_realized = mapping.get(dl, '')
        correct = (subset['realized'] == expected_realized).sum()
        # Also count "close" matches
        close_matches = 0
        if dl in ('CHOP_HARD', 'CHOP_MILD'):
            close_matches = (subset['realized'].isin(['CHOP_REAL', 'NEUTRAL_REAL'])).sum()
        elif dl == 'NEUTRAL':
            close_matches = (subset['realized'].isin(['NEUTRAL_REAL', 'CHOP_REAL'])).sum()
        
        exact_pct = correct / len(subset) * 100
        close_pct = close_matches / len(subset) * 100
        print(f"  {dl:<14} exact={exact_pct:5.1f}%  close={close_pct:5.1f}%  (n={len(subset)})")


# ═══════════════════════════════════════════════════════════════
# SIGNAL IMPORTANCE (permutation)
# ═══════════════════════════════════════════════════════════════

def signal_importance(results_df, df_15m):
    """Which signals matter most for regime discrimination?"""
    print("\n" + "="*70)
    print("SIGNAL IMPORTANCE (correlation with forward 4H return magnitude)")
    print("="*70)
    
    closes = df_15m['Close'].values
    horizon = 16
    
    indices = results_df['idx'].values
    valid_mask = indices + horizon < len(closes)
    valid_df = results_df[valid_mask].copy()
    valid_indices = valid_df['idx'].values
    
    fwd_abs = np.abs((closes[valid_indices + horizon] - closes[valid_indices]) / closes[valid_indices] * 100)
    fwd_signed = (closes[valid_indices + horizon] - closes[valid_indices]) / closes[valid_indices] * 100
    
    signals = ['atr_pctl', 'bb_pctl', 'directionality', 'directionality_1h', 
               'directionality_15m', 'whipsaw_rate', 'retrace_ratio', 
               'volume_confirm', 'range_tight', 'tf_coherence', 'structure_score']
    
    print(f"\n{'Signal':<22} {'|corr| with':>12} {'corr with':>12} {'regime':>12}")
    print(f"{'':22} {'|fwd return|':>12} {'fwd return':>12} {'discrim':>12}")
    print("-" * 62)
    
    for sig in signals:
        if sig not in valid_df.columns:
            continue
        vals = valid_df[sig].values
        
        # Correlation with absolute forward return (does signal predict volatility?)
        corr_abs = np.corrcoef(vals, fwd_abs)[0, 1] if np.std(vals) > 0 else 0
        
        # Correlation with signed forward return (does signal predict direction?)
        corr_signed = np.corrcoef(vals, fwd_signed)[0, 1] if np.std(vals) > 0 else 0
        
        # ANOVA F-stat: does the signal separate regimes?
        regime_groups = []
        for regime in ['CRISIS', 'CHOP_HARD', 'CHOP_MILD', 'COMPRESSING', 'TRENDING']:
            mask = valid_df['regime'] == regime
            if mask.sum() > 5:
                regime_groups.append(vals[mask])
        
        if len(regime_groups) >= 2:
            from scipy import stats
            f_stat, p_val = stats.f_oneway(*regime_groups)
            discrim = f"F={f_stat:.1f} p={p_val:.3f}"
        else:
            discrim = "N/A"
        
        print(f"{sig:<22} {abs(corr_abs):>12.3f} {corr_signed:>+12.3f} {discrim:>12}")


# ═══════════════════════════════════════════════════════════════
# REGIME TRANSITION ANALYSIS
# ═══════════════════════════════════════════════════════════════

def transition_analysis(results_df):
    """How often do regimes transition? Are transitions stable?"""
    print("\n" + "="*70)
    print("REGIME TRANSITION ANALYSIS")
    print("="*70)
    
    regimes = results_df['regime'].values
    
    # Count transitions
    transitions = {}
    for i in range(1, len(regimes)):
        pair = (regimes[i-1], regimes[i])
        transitions[pair] = transitions.get(pair, 0) + 1
    
    # Regime duration
    print("\n--- Regime Duration (bars) ---")
    print(f"{'Regime':<14} {'Mean':>8} {'Median':>8} {'Max':>8} {'Count':>8}")
    print("-" * 50)
    
    current_regime = regimes[0]
    current_len = 1
    durations = {}
    
    for i in range(1, len(regimes)):
        if regimes[i] == current_regime:
            current_len += 1
        else:
            if current_regime not in durations:
                durations[current_regime] = []
            durations[current_regime].append(current_len)
            current_regime = regimes[i]
            current_len = 1
    
    for regime in ['CRISIS', 'CHOP_HARD', 'CHOP_MILD', 'COMPRESSING', 'TRENDING', 'NEUTRAL']:
        if regime in durations and durations[regime]:
            d = durations[regime]
            print(f"{regime:<14} {np.mean(d):>8.1f} {np.median(d):>8.0f} {max(d):>8} {len(d):>8}")
    
    # Transition matrix
    print("\n--- Transition Matrix (probability of going FROM row TO column) ---")
    all_regimes = ['CRISIS', 'CHOP_HARD', 'CHOP_MILD', 'COMPRESSING', 'TRENDING', 'NEUTRAL']
    
    print(f"{'From \\ To':<14}", end='')
    for r in all_regimes:
        short = r[:8]
        print(f"{short:>10}", end='')
    print()
    
    for r1 in all_regimes:
        total_out = sum(v for (k, v) in transitions.items() if k[0] == r1)
        if total_out == 0:
            continue
        print(f"{r1:<14}", end='')
        for r2 in all_regimes:
            count = transitions.get((r1, r2), 0)
            pct = count / total_out * 100
            print(f"{pct:>9.1f}%", end='')
        print()


# ═══════════════════════════════════════════════════════════════
# TRADEABILITY SCORE VALIDATION
# ═══════════════════════════════════════════════════════════════

def tradeability_validation(results_df, df_15m):
    """Does the regime score actually predict trade quality?"""
    print("\n" + "="*70)
    print("TRADEABILITY SCORE VALIDATION")
    print("Does higher regime score → better forward returns?")
    print("="*70)
    
    closes = df_15m['Close'].values
    horizon = 16
    
    valid_mask = results_df['idx'].values + horizon < len(closes)
    valid_df = results_df[valid_mask].copy()
    valid_indices = valid_df['idx'].values
    
    fwd = (closes[valid_indices + horizon] - closes[valid_indices]) / closes[valid_indices] * 100
    
    # Split by score quintiles
    valid_df['fwd_return'] = fwd
    n_unique = valid_df['score'].nunique()
    n_bins = min(5, n_unique)
    if n_bins < 2:
        print(f"\n  Only {n_unique} unique score value(s) — can't bin into quintiles.")
        print(f"  Score distribution: mean={valid_df['score'].mean():.3f} std={valid_df['score'].std():.3f}")
        print(f"  Score value counts:")
        for val, cnt in valid_df['score'].value_counts().items():
            print(f"    {val:.3f}: {cnt} bars ({cnt/len(valid_df)*100:.1f}%)")
        fwd = valid_df['fwd_return']
        print(f"  Overall 4H forward: mean={fwd.mean():+.3f}% win={(fwd>0).mean()*100:.1f}%")
        corr = np.corrcoef(valid_df['score'].values, valid_df['fwd_return'].values)[0, 1]
        print(f"  Score ↔ Forward Return correlation: {corr:+.4f}")
        return
    try:
        valid_df['score_quintile'] = pd.qcut(valid_df['score'], n_bins, labels=[f'Q{i+1}' for i in range(n_bins)], duplicates='drop')
        
        print(f"\n{'Quintile':>10} {'Score Range':>18} {'Mean Ret%':>10} {'Median%':>10} {'Sharpe':>8} {'Win%':>7}")
        print("-" * 68)
        
        for q in [f'Q{i+1}' for i in range(n_bins)]:
            subset = valid_df[valid_df['score_quintile'] == q]
            if len(subset) < 10:
                continue
            
            mean_r = subset['fwd_return'].mean()
            median_r = subset['fwd_return'].median()
            std_r = subset['fwd_return'].std()
            sharpe = mean_r / std_r if std_r > 0 else 0
            win = (subset['fwd_return'] > 0).mean() * 100
            score_range = f"{subset['score'].min():.2f}-{subset['score'].max():.2f}"
            
            print(f"{q:>10} {score_range:>18} {mean_r:>+10.3f} {median_r:>+10.3f} {sharpe:>+8.3f} {win:>6.1f}%")
    except Exception as e:
        print(f"\n  qcut failed ({e}) — showing per-score breakdown instead:")
        for val in sorted(valid_df['score'].unique()):
            subset = valid_df[valid_df['score'] == val]
            mean_r = subset['fwd_return'].mean()
            win = (subset['fwd_return'] > 0).mean() * 100
            print(f"    score={val:.3f}: n={len(subset):>5}  mean={mean_r:+.3f}%  win={win:.1f}%")
    
    # Correlation
    corr = np.corrcoef(valid_df['score'].values, valid_df['fwd_return'].values)[0, 1]
    print(f"\n  Score ↔ Forward Return correlation: {corr:+.4f}")
    
    # Directional accuracy: does TRENDING regime + positive score → positive return?
    trending = valid_df[valid_df['regime'] == 'TRENDING']
    if len(trending) > 10:
        pos_trending = (trending['fwd_return'] > 0).mean() * 100
        print(f"  TRENDING regime → positive 4H return: {pos_trending:.1f}% (n={len(trending)})")
    
    compressing = valid_df[valid_df['regime'] == 'COMPRESSING']
    if len(compressing) > 10:
        # After compression, should see breakout (larger moves)
        abs_return = compressing['fwd_return'].abs().mean()
        neutral_abs = valid_df[valid_df['regime'] == 'NEUTRAL']['fwd_return'].abs().mean()
        print(f"  COMPRESSING avg |return|: {abs_return:.3f}% vs NEUTRAL: {neutral_abs:.3f}%")


# ═══════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════

if __name__ == '__main__':
    df_15m, df_1h = load_data()
    results_df = run_classifier(df_15m, df_1h)
    
    # Save raw results
    results_df.to_csv('regime_results.csv', index=False)
    print(f"\nSaved {len(results_df)} classified bars to regime_results.csv")
    
    # Regime distribution
    print("\n" + "="*70)
    print("REGIME DISTRIBUTION (after hysteresis)")
    print("="*70)
    dist = results_df['regime'].value_counts()
    for regime, count in dist.items():
        pct = count / len(results_df) * 100
        print(f"  {regime:<14} {count:>6} ({pct:>5.1f}%)")
    
    print("\n" + "="*70)
    print("RAW REGIME DISTRIBUTION (before hysteresis)")
    print("="*70)
    dist_raw = results_df['raw_regime'].value_counts()
    for regime, count in dist_raw.items():
        pct = count / len(results_df) * 100
        print(f"  {regime:<14} {count:>6} ({pct:>5.1f}%)")
    
    # Signal statistics by raw_regime
    print("\n" + "="*70)
    print("SIGNAL MEANS BY RAW REGIME")
    print("="*70)
    sigs = ['atr_pctl', 'bb_pctl', 'directionality', 'whipsaw_rate', 'retrace_ratio', 'volume_confirm', 'range_tight', 'tf_coherence', 'structure_score']
    header = f"{'Regime':<14}" + "".join(f"{s:>14}" for s in sigs)
    print(header)
    print("-" * (14 + 14 * len(sigs)))
    for regime in ['CRISIS', 'CHOP_HARD', 'CHOP_MILD', 'COMPRESSING', 'TRENDING', 'NEUTRAL']:
        mask = results_df['raw_regime'] == regime
        if mask.sum() == 0:
            continue
        row = f"{regime:<14}"
        for s in sigs:
            row += f"{results_df.loc[mask, s].mean():>14.3f}"
        row += f"  (n={mask.sum()})"
        print(row)
    
    # Run all analyses
    forward_return_analysis(results_df, df_15m)
    
    try:
        discrimination_test(results_df, df_15m)
    except ImportError:
        print("\n[skipping KS test — scipy not available]")
    
    confusion_matrix(results_df, df_15m)
    
    try:
        signal_importance(results_df, df_15m)
    except ImportError:
        print("\n[skipping signal importance — scipy not available]")
    
    transition_analysis(results_df)
    tradeability_validation(results_df, df_15m)
    
    print("\n" + "="*70)
    print("DONE")
    print("="*70)

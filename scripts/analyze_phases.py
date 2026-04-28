#!/usr/bin/env python3
"""
JIMI Framework — Phase 1 & 2 Diagnostic Analyzer

Post-processes the phase_diagnostics.csv exported from a backtest.
Answers the questions the standard report can't:

  - How much time does the market spend in each regime?
  - How often does each regime transition happen?
  - Does M13 structure bias actually predict winners?
  - How often does M7 conflict vs agree, and does conflict hurt?
  - What's the size multiplier distribution?
  - Which confirmation layer blocks the most?
  - Which regime produces the best trades?

Usage:
    python scripts/analyze_phases.py phase_diagnostics.csv
    python scripts/analyze_phases.py phase_diagnostics.csv --detailed
    python scripts/analyze_phases.py phase_diagnostics.csv --export report.json
"""

import argparse
import sys
import os
import json
import numpy as np
import pandas as pd

pd.set_option('display.max_columns', 20)
pd.set_option('display.width', 120)


def load_data(path):
    df = pd.read_csv(path, parse_dates=['timestamp'])
    print(f"\n  Loaded {len(df):,} bar diagnostics from {path}")
    if 'timestamp' in df.columns:
        print(f"  Range: {df['timestamp'].iloc[0]} → {df['timestamp'].iloc[-1]}")
    return df


# ═══════════════════════════════════════════════════════════════
# SECTION 1: REGIME ANALYSIS
# ═══════════════════════════════════════════════════════════════

def regime_analysis(df):
    print(f"\n{'═'*70}")
    print(f"  PHASE 1: M9 VOLATILITY REGIME ANALYSIS")
    print(f"{'═'*70}")

    if 'm9_regime' not in df.columns:
        print("  No M9 regime data found.")
        return

    total = len(df)

    # Distribution
    print(f"\n  Regime Distribution ({total:,} bars):")
    print(f"  {'Regime':<15} {'Bars':>8} {'Pct':>8} {'Avg Score':>10}")
    print(f"  {'─'*15} {'─'*8} {'─'*8} {'─'*10}")
    for regime in ['TRENDING', 'NEUTRAL', 'COMPRESSING', 'CHOP_MILD', 'CHOP_MILD_BEAR', 'CHOP_MILD_BULL', 'CHOP_HARD', 'CRISIS']:
        subset = df[df['m9_regime'] == regime]
        if len(subset) > 0:
            avg_score = subset['m9_score'].mean() if 'm9_score' in subset.columns else 0
            print(f"  {regime:<15} {len(subset):>8,} {len(subset)/total*100:>7.1f}% {avg_score:>10.3f}")

    # Regime strength
    if 'm9_regime_strength' in df.columns:
        print(f"\n  Regime Strength (how established the current regime is):")
        for regime in ['TRENDING', 'NEUTRAL', 'COMPRESSING', 'CHOP_MILD', 'CHOP_MILD_BEAR', 'CHOP_MILD_BULL']:
            subset = df[df['m9_regime'] == regime]
            if len(subset) > 0:
                strength = subset['m9_regime_strength'].dropna()
                if len(strength) > 0:
                    print(f"    {regime:<15} mean={strength.mean():.3f}  median={strength.median():.3f}  "
                          f"p25={strength.quantile(0.25):.3f}  p75={strength.quantile(0.75):.3f}")

    # Signal profiles per regime
    signal_cols = ['m9_atr_pctl', 'm9_bb_pctl', 'm9_directionality', 'm9_whipsaw_rate',
                   'm9_retrace_ratio', 'm9_volume_confirm', 'm9_range_tight',
                   'm9_tf_coherence', 'm9_chop_score', 'm9_trend_score']
    available_signals = [c for c in signal_cols if c in df.columns]

    if available_signals:
        print(f"\n  Signal Profiles by Regime (mean values):")
        regimes = ['TRENDING', 'NEUTRAL', 'COMPRESSING', 'CHOP_MILD', 'CHOP_MILD_BEAR', 'CHOP_MILD_BULL', 'CHOP_HARD', 'CRISIS']
        header = f"  {'Signal':<22}" + "".join(f"{r:>13}" for r in regimes)
        print(header)
        print(f"  {'─'*22}" + "".join(f"{'─'*13}" for _ in regimes))

        for col in available_signals:
            short_name = col.replace('m9_', '')
            vals = []
            for regime in regimes:
                subset = df[df['m9_regime'] == regime][col].dropna()
                vals.append(f"{subset.mean():>13.3f}" if len(subset) > 0 else f"{'N/A':>13}")
            print(f"  {short_name:<22}" + "".join(vals))

    # Transitions
    if 'm9_is_transition' in df.columns:
        transitions = df[df['m9_is_transition'] == True]
        print(f"\n  Regime Transitions: {len(transitions)} ({len(transitions)/total*100:.2f}% of bars)")

    # Regime autocorrelation — how sticky is each regime?
    if len(df) > 10:
        print(f"\n  Regime Stickiness (avg consecutive bars in same regime):")
        regimes = df['m9_regime'].values
        for target_regime in ['TRENDING', 'NEUTRAL', 'COMPRESSING', 'CHOP_MILD', 'CHOP_MILD_BEAR', 'CHOP_MILD_BULL', 'CHOP_HARD', 'CRISIS']:
            streaks = []
            current_streak = 0
            for r in regimes:
                if r == target_regime:
                    current_streak += 1
                else:
                    if current_streak > 0:
                        streaks.append(current_streak)
                    current_streak = 0
            if current_streak > 0:
                streaks.append(current_streak)
            if streaks:
                print(f"    {target_regime:<15} avg={np.mean(streaks):.1f} bars  "
                      f"max={max(streaks)}  median={np.median(streaks):.0f}")


# ═══════════════════════════════════════════════════════════════
# SECTION 2: M13 STRUCTURE ANALYSIS
# ═══════════════════════════════════════════════════════════════

def m13_analysis(df):
    print(f"\n{'═'*70}")
    print(f"  PHASE 2: M13 HTF STRUCTURE ANALYSIS")
    print(f"{'═'*70}")

    if 'm13_bias' not in df.columns:
        print("  No M13 structure data found.")
        return

    total = len(df)

    # Bias distribution
    print(f"\n  M13 Bias Distribution:")
    for bias in ['BULLISH', 'LEAN_BULL', 'NEUTRAL', 'LEAN_BEAR', 'BEARISH']:
        count = (df['m13_bias'] == bias).sum()
        print(f"    {bias:<12} {count:>8,} ({count/total*100:>5.1f}%)")

    # 1H vs 15m alignment
    if 'm13_swing_bias_1h' in df.columns and 'm13_swing_bias_15m' in df.columns:
        print(f"\n  1H vs 15m Swing Alignment:")
        aligned = df[
            ((df['m13_swing_bias_1h'].str.contains('BULL', na=False)) & 
             (df['m13_swing_bias_15m'].str.contains('BULL', na=False))) |
            ((df['m13_swing_bias_1h'].str.contains('BEAR', na=False)) & 
             (df['m13_swing_bias_15m'].str.contains('BEAR', na=False)))
        ]
        conflicting = df[
            ((df['m13_swing_bias_1h'].str.contains('BULL', na=False)) & 
             (df['m13_swing_bias_15m'].str.contains('BEAR', na=False))) |
            ((df['m13_swing_bias_1h'].str.contains('BEAR', na=False)) & 
             (df['m13_swing_bias_15m'].str.contains('BULL', na=False)))
        ]
        print(f"    Aligned:    {len(aligned):>8,} ({len(aligned)/total*100:>5.1f}%)")
        print(f"    Conflicting: {len(conflicting):>8,} ({len(conflicting)/total*100:>5.1f}%)")
        print(f"    Mixed/Neutral: {total - len(aligned) - len(conflicting):>8,}")

    # Bull/Bear points distribution
    if 'm13_bull_points' in df.columns and 'm13_bear_points' in df.columns:
        bp = df['m13_bull_points'].dropna()
        bp_bear = df['m13_bear_points'].dropna()
        print(f"\n  Bull Points: mean={bp.mean():.3f}  median={bp.median():.3f}")
        print(f"  Bear Points: mean={bp_bear.mean():.3f}  median={bp_bear.median():.3f}")

    # FVG and OB counts
    if 'm13_fvg_count' in df.columns:
        fvg = df['m13_fvg_count'].dropna()
        print(f"\n  FVG Count: mean={fvg.mean():.1f}  max={fvg.max():.0f}  "
              f"zero_pct={(fvg == 0).mean()*100:.1f}%")
    if 'm13_ob_count' in df.columns:
        ob = df['m13_ob_count'].dropna()
        print(f"  OB Count:  mean={ob.mean():.1f}  max={ob.max():.0f}  "
              f"zero_pct={(ob == 0).mean()*100:.1f}%")

    # M13 score distribution
    if 'm13_score' in df.columns:
        score = df['m13_score'].dropna()
        print(f"\n  M13 Score Distribution:")
        print(f"    mean={score.mean():.3f}  median={score.median():.3f}")
        print(f"    p10={score.quantile(0.10):.3f}  p25={score.quantile(0.25):.3f}  "
              f"p75={score.quantile(0.75):.3f}  p90={score.quantile(0.90):.3f}")


# ═══════════════════════════════════════════════════════════════
# SECTION 3: M7 MACRO ANALYSIS
# ═══════════════════════════════════════════════════════════════

def m7_analysis(df):
    print(f"\n{'═'*70}")
    print(f"  PHASE 2: M7 MARKET REGIME (MACRO) ANALYSIS")
    print(f"{'═'*70}")

    if 'm7_score' not in df.columns:
        print("  No M7 data found.")
        return

    m7 = df[df['m7_score'].notna()]
    if len(m7) == 0:
        print("  M7 data is all NaN.")
        return

    print(f"\n  M7 Score: mean={m7['m7_score'].mean():.3f}  "
          f"median={m7['m7_score'].median():.3f}")
    print(f"    p10={m7['m7_score'].quantile(0.10):.3f}  "
          f"p25={m7['m7_score'].quantile(0.25):.3f}  "
          f"p75={m7['m7_score'].quantile(0.75):.3f}  "
          f"p90={m7['m7_score'].quantile(0.90):.3f}")

    if 'm7_status' in m7.columns:
        print(f"\n  M7 Status Distribution:")
        for status in ['PASS', 'FAIL', 'SKIP']:
            count = (m7['m7_status'] == status).sum()
            if count > 0:
                print(f"    {status:<8} {count:>8,} ({count/len(m7)*100:>5.1f}%)")

    if 'm7_eth_btc_trend' in m7.columns:
        print(f"\n  ETH/BTC Trend Distribution:")
        for trend in ['BULL', 'BEAR', 'NEUTRAL']:
            count = (m7['m7_eth_btc_trend'] == trend).sum()
            if count > 0:
                print(f"    {trend:<10} {count:>8,} ({count/len(m7)*100:>5.1f}%)")

    if 'm7_btc_trend' in m7.columns:
        print(f"\n  BTC Trend Distribution:")
        for trend in ['BULL', 'BEAR', 'NEUTRAL']:
            count = (m7['m7_btc_trend'] == trend).sum()
            if count > 0:
                print(f"    {trend:<10} {count:>8,} ({count/len(m7)*100:>5.1f}%)")

    if 'm7_btc_atr_pctl' in m7.columns:
        atr = m7['m7_btc_atr_pctl'].dropna()
        if len(atr) > 0:
            print(f"\n  BTC ATR Percentile: mean={atr.mean():.3f}  "
                  f"p10={atr.quantile(0.10):.3f}  p90={atr.quantile(0.90):.3f}")


# ═══════════════════════════════════════════════════════════════
# SECTION 4: DIRECTION RESOLVER ANALYSIS
# ═══════════════════════════════════════════════════════════════

def direction_analysis(df):
    print(f"\n{'═'*70}")
    print(f"  PHASE 2: DIRECTION RESOLVER ANALYSIS")
    print(f"{'═'*70}")

    if 'direction' not in df.columns:
        print("  No direction data found.")
        return

    total = len(df)
    traded = df[df['direction'].isin(['LONG', 'SHORT'])]
    neutral = df[df['direction'] == 'NEUTRAL']

    print(f"\n  Direction Distribution:")
    print(f"    LONG:    {len(traded[traded['direction']=='LONG']):>8,} "
          f"({len(traded[traded['direction']=='LONG'])/total*100:>5.1f}%)")
    print(f"    SHORT:   {len(traded[traded['direction']=='SHORT']):>8,} "
          f"({len(traded[traded['direction']=='SHORT'])/total*100:>5.1f}%)")
    print(f"    NEUTRAL: {len(neutral):>8,} ({len(neutral)/total*100:>5.1f}%)")

    # Size multiplier distribution
    if 'dir_size_mult' in df.columns:
        sm = df['dir_size_mult'].dropna()
        if len(sm) > 0:
            print(f"\n  Size Multiplier Distribution (all bars):")
            print(f"    mean={sm.mean():.3f}  median={sm.median():.3f}")
            print(f"    p10={sm.quantile(0.10):.3f}  p25={sm.quantile(0.25):.3f}  "
                  f"p75={sm.quantile(0.75):.3f}  p90={sm.quantile(0.90):.3f}")
            print(f"    zeros: {(sm == 0).sum():,} ({(sm == 0).mean()*100:.1f}%)")

            # Only for tradeable bars
            tradeable_sm = traded['dir_size_mult'].dropna()
            if len(tradeable_sm) > 0:
                print(f"\n  Size Multiplier (tradeable bars only):")
                print(f"    mean={tradeable_sm.mean():.3f}  median={tradeable_sm.median():.3f}")
                print(f"    p10={tradeable_sm.quantile(0.10):.3f}  p90={tradeable_sm.quantile(0.90):.3f}")

    # Confirmation layer analysis
    print(f"\n  Confirmation Layer Impact (on tradeable bars):")
    layers = [
        ('dir_m7_penalty', 'M7 Conflict', '×0.70'),
        ('dir_m7_bonus', 'M7 Agree', '×1.10'),
        ('dir_daily_penalty', 'Daily Swing Conflict', '×0.75'),
        ('dir_daily_bonus', 'Daily Swing Agree', '×1.05'),
        ('dir_trend_penalty', 'Trend Conflict', '×0.70'),
        ('dir_trend_bonus', 'Trend Agree', '×1.10'),
        ('dir_trending_structure_bonus', 'Trending+Structure', '×1.15'),
    ]
    for col, name, mult in layers:
        if col in traded.columns:
            hits = traded[col].notna().sum()
            if hits > 0:
                pct = hits / len(traded) * 100
                print(f"    {name:<25} {hits:>6,} bars ({pct:>5.1f}%) [{mult}]")

    # Block reason analysis
    if 'block_reason' in df.columns:
        blocked = df[df['block_reason'].notna()]
        if len(blocked) > 0:
            print(f"\n  Block Reasons (bars that reached Phase 2 but were blocked):")
            print(f"  {'Module':<20} {'Reason':<35} {'Count':>8}")
            print(f"  {'─'*20} {'─'*35} {'─'*8}")
            if 'block_module' in blocked.columns:
                for (module, reason), group in blocked.groupby(['block_module', 'block_reason']):
                    print(f"  {str(module):<20} {str(reason):<35} {len(group):>8,}")


# ═══════════════════════════════════════════════════════════════
# SECTION 5: TRADE OUTCOME CORRELATION
# ═══════════════════════════════════════════════════════════════

def outcome_analysis(df):
    print(f"\n{'═'*70}")
    print(f"  TRADE OUTCOME CORRELATION (Phase 1+2 signals vs results)")
    print(f"{'═'*70}")

    if 'trade_outcome' not in df.columns:
        print("  No trade outcomes recorded. Run backtest with diagnostic logging enabled.")
        print("  (Only bars that became trades have outcomes.)")
        return

    traded = df[df['trade_outcome'].notna()].copy()
    if len(traded) == 0:
        print("  No trades with outcomes found.")
        return

    winners = traded[traded['trade_outcome'] == 'WIN']
    losers = traded[traded['trade_outcome'] == 'LOSS']

    print(f"\n  {len(traded)} trades: {len(winners)}W / {len(losers)}L "
          f"(WR {len(winners)/len(traded)*100:.1f}%)")

    # Win rate by regime
    if 'm9_regime' in traded.columns:
        print(f"\n  Win Rate by Regime:")
        print(f"  {'Regime':<15} {'Trades':>7} {'WR':>7} {'Avg PnL':>10} {'Avg Size':>10}")
        print(f"  {'─'*15} {'─'*7} {'─'*7} {'─'*10} {'─'*10}")
        for regime in ['TRENDING', 'NEUTRAL', 'COMPRESSING', 'CHOP_MILD', 'CHOP_MILD_BEAR', 'CHOP_MILD_BULL', 'CHOP_HARD', 'CRISIS']:
            r = traded[traded['m9_regime'] == regime]
            if len(r) > 0:
                wr = (r['trade_outcome'] == 'WIN').mean() * 100
                avg_pnl = r['trade_pnl'].mean() if 'trade_pnl' in r.columns else 0
                avg_size = r['dir_size_mult'].mean() if 'dir_size_mult' in r.columns else 0
                print(f"  {regime:<15} {len(r):>7} {wr:>6.1f}% {avg_pnl:>+9.2f}% {avg_size:>10.3f}")

    # Win rate by M13 bias
    if 'm13_bias' in traded.columns:
        print(f"\n  Win Rate by M13 Structure Bias:")
        print(f"  {'Bias':<15} {'Trades':>7} {'WR':>7} {'Avg PnL':>10}")
        print(f"  {'─'*15} {'─'*7} {'─'*7} {'─'*10}")
        for bias in ['BULLISH', 'LEAN_BULL', 'NEUTRAL', 'LEAN_BEAR', 'BEARISH']:
            b = traded[traded['m13_bias'] == bias]
            if len(b) > 0:
                wr = (b['trade_outcome'] == 'WIN').mean() * 100
                avg_pnl = b['trade_pnl'].mean() if 'trade_pnl' in b.columns else 0
                print(f"  {bias:<15} {len(b):>7} {wr:>6.1f}% {avg_pnl:>+9.2f}%")

    # Win rate by M7 status
    if 'm7_status' in traded.columns:
        print(f"\n  Win Rate by M7 Status:")
        for status in ['PASS', 'FAIL', 'SKIP']:
            s = traded[traded['m7_status'] == status]
            if len(s) > 0:
                wr = (s['trade_outcome'] == 'WIN').mean() * 100
                avg_pnl = s['trade_pnl'].mean() if 'trade_pnl' in s.columns else 0
                print(f"    {status:<8} {len(s):>6} trades, WR {wr:.1f}%, Avg PnL {avg_pnl:+.2f}%")

    # Win rate by M7 direction (agree/conflict)
    if 'dir_m7_direction' in traded.columns:
        print(f"\n  Win Rate by M7 Direction Alignment:")
        for d in ['AGREE', 'CONFLICT', 'NEUTRAL']:
            s = traded[traded['dir_m7_direction'] == d]
            if len(s) > 0:
                wr = (s['trade_outcome'] == 'WIN').mean() * 100
                print(f"    {d:<10} {len(s):>6} trades, WR {wr:.1f}%")

    # Win rate by direction
    print(f"\n  Win Rate by Direction:")
    for d in ['LONG', 'SHORT']:
        s = traded[traded['direction'] == d]
        if len(s) > 0:
            wr = (s['trade_outcome'] == 'WIN').mean() * 100
            avg_pnl = s['trade_pnl'].mean() if 'trade_pnl' in s.columns else 0
            print(f"    {d:<8} {len(s):>6} trades, WR {wr:.1f}%, Avg PnL {avg_pnl:+.2f}%")

    # Regime × Direction cross-tab
    if 'm9_regime' in traded.columns:
        print(f"\n  Regime × Direction Cross-Tab (Win Rate):")
        print(f"  {'':>15} {'LONG':>10} {'SHORT':>10}")
        print(f"  {'─'*15} {'─'*10} {'─'*10}")
        for regime in ['TRENDING', 'NEUTRAL', 'COMPRESSING', 'CHOP_MILD', 'CHOP_MILD_BEAR', 'CHOP_MILD_BULL']:
            row_vals = []
            for d in ['LONG', 'SHORT']:
                subset = traded[(traded['m9_regime'] == regime) & (traded['direction'] == d)]
                if len(subset) >= 3:
                    wr = (subset['trade_outcome'] == 'WIN').mean() * 100
                    row_vals.append(f"{wr:.0f}% ({len(subset)})")
                else:
                    row_vals.append(f"  -  ({len(subset)})")
            print(f"  {regime:<15} {row_vals[0]:>10} {row_vals[1]:>10}")

    # M13 score correlation with outcome
    if 'm13_score' in traded.columns:
        w_m13 = winners['m13_score'].dropna()
        l_m13 = losers['m13_score'].dropna()
        if len(w_m13) > 0 and len(l_m13) > 0:
            print(f"\n  M13 Score: Winners avg={w_m13.mean():.3f}  Losers avg={l_m13.mean():.3f}  "
                  f"Delta={w_m13.mean()-l_m13.mean():+.3f}")

    # M7 score correlation with outcome
    if 'm7_score' in traded.columns:
        w_m7 = winners['m7_score'].dropna()
        l_m7 = losers['m7_score'].dropna()
        if len(w_m7) > 0 and len(l_m7) > 0:
            print(f"  M7 Score:  Winners avg={w_m7.mean():.3f}  Losers avg={l_m7.mean():.3f}  "
                  f"Delta={w_m7.mean()-l_m7.mean():+.3f}")

    # Size multiplier vs outcome
    if 'dir_size_mult' in traded.columns:
        w_sm = winners['dir_size_mult'].dropna()
        l_sm = losers['dir_size_mult'].dropna()
        if len(w_sm) > 0 and len(l_sm) > 0:
            print(f"  Size Mult: Winners avg={w_sm.mean():.3f}  Losers avg={l_sm.mean():.3f}  "
                  f"Delta={w_sm.mean()-l_sm.mean():+.3f}")


# ═══════════════════════════════════════════════════════════════
# SECTION 6: DETAILED TIME-SERIES VIEW
# ═══════════════════════════════════════════════════════════════

def detailed_view(df):
    """Show regime/direction changes over time (sampled)."""
    print(f"\n{'═'*70}")
    print(f"  TIME-SERIES VIEW (regime + direction transitions)")
    print(f"{'═'*70}")

    if 'timestamp' not in df.columns:
        print("  No timestamp data.")
        return

    # Find regime transitions
    if 'm9_regime' in df.columns:
        regime_changes = df[df['m9_regime'] != df['m9_regime'].shift(1)].copy()
        if len(regime_changes) > 0:
            print(f"\n  Last 20 Regime Transitions:")
            print(f"  {'Timestamp':<22} {'From':<15} {'To':<15} {'Score':>8}")
            print(f"  {'─'*22} {'─'*15} {'─'*15} {'─'*8}")
            for _, row in regime_changes.tail(20).iterrows():
                ts = str(row['timestamp'])[:19] if pd.notna(row['timestamp']) else '?'
                prev_idx = df.index.get_loc(row.name) - 1
                prev_regime = df.iloc[prev_idx]['m9_regime'] if prev_idx >= 0 else '?'
                score = row.get('m9_score', '?')
                score_str = f"{score:.3f}" if isinstance(score, (int, float)) else str(score)
                print(f"  {ts:<22} {str(prev_regime):<15} {str(row['m9_regime']):<15} {score_str:>8}")

    # Find direction changes (only for tradeable bars)
    if 'direction' in df.columns:
        tradeable = df[df['direction'].isin(['LONG', 'SHORT'])]
        if len(tradeable) > 0:
            dir_changes = tradeable[tradeable['direction'] != tradeable['direction'].shift(1)]
            print(f"\n  Last 20 Direction Changes (tradeable bars):")
            print(f"  {'Timestamp':<22} {'Direction':<10} {'Regime':<15} {'M13 Bias':<12} {'Size':>8}")
            print(f"  {'─'*22} {'─'*10} {'─'*15} {'─'*12} {'─'*8}")
            for _, row in dir_changes.tail(20).iterrows():
                ts = str(row['timestamp'])[:19] if pd.notna(row['timestamp']) else '?'
                size = row.get('dir_size_mult', '?')
                size_str = f"{size:.3f}" if isinstance(size, (int, float)) else str(size)
                print(f"  {ts:<22} {str(row['direction']):<10} "
                      f"{str(row.get('m9_regime','?')):<15} "
                      f"{str(row.get('m13_bias','?')):<12} {size_str:>8}")


# ═══════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description='JIMI Phase 1+2 Diagnostic Analyzer')
    parser.add_argument('csv', help='Path to phase_diagnostics.csv')
    parser.add_argument('--detailed', '-d', action='store_true',
                        help='Show time-series transitions')
    parser.add_argument('--export', '-e', help='Export summary JSON to file')
    args = parser.parse_args()

    if not os.path.exists(args.csv):
        print(f"ERROR: File not found: {args.csv}")
        sys.exit(1)

    df = load_data(args.csv)

    regime_analysis(df)
    m13_analysis(df)
    m7_analysis(df)
    direction_analysis(df)
    outcome_analysis(df)

    if args.detailed:
        detailed_view(df)

    if args.export:
        summary = {}
        if 'm9_regime' in df.columns:
            summary['regime_distribution'] = df['m9_regime'].value_counts().to_dict()
        if 'm13_bias' in df.columns:
            summary['m13_bias_distribution'] = df['m13_bias'].value_counts().to_dict()
        if 'direction' in df.columns:
            summary['direction_distribution'] = df['direction'].value_counts().to_dict()
        with open(args.export, 'w') as f:
            json.dump(summary, f, indent=2, default=str)
        print(f"\n  Summary exported → {args.export}")

    print(f"\n{'═'*70}")
    print(f"  Done.")
    print(f"{'═'*70}")


if __name__ == '__main__':
    main()

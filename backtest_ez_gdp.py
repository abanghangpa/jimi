"""
Prompt A + B: Backtest Eurozone GDP Flash Session Transmission Chain
====================================================================
ETH/USDT 15m data from jimi/eth_15m_merged.csv
Event: Eurozone GDP Flash (Eurostat, ~end of month/early next month, 09:00 UTC)
Transmission: Europe Morning (18:00 MYT) → US GDP Advance → ECB Cut Trajectory

Thesis:
  Negative EZ GDP → recession confirmed → capital flees EU equities → DXY up → ETH pressure
  Then: negative EZ GDP → accelerates ECB rate cuts → lifts global liquidity over months
  US macro funds use this as warning for US GDP Advance (~1 week later)
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

# ═══════════════════════════════════════════════════════════════
# EUROZONE GDP FLASH RELEASE DATES + ACTUAL VALUES (09:00 UTC)
# Eurostat releases GDP flash estimate (quarter-over-quarter, annualized)
# Released ~30 days after quarter end
# ═══════════════════════════════════════════════════════════════

EZ_GDP_RELEASES = {
    # 2018
    '2018-01-30': {'gdp_qoq': 0.6, 'gdp_yoy': 2.7, 'consensus_qoq': 0.6, 'prior_qoq': 0.7},
    '2018-04-30': {'gdp_qoq': 0.4, 'gdp_yoy': 2.5, 'consensus_qoq': 0.4, 'prior_qoq': 0.7},
    '2018-07-31': {'gdp_qoq': 0.4, 'gdp_yoy': 2.2, 'consensus_qoq': 0.4, 'prior_qoq': 0.4},
    '2018-10-30': {'gdp_qoq': 0.2, 'gdp_yoy': 1.7, 'consensus_qoq': 0.2, 'prior_qoq': 0.4},
    # 2019
    '2019-01-31': {'gdp_qoq': 0.2, 'gdp_yoy': 1.2, 'consensus_qoq': 0.2, 'prior_qoq': 0.2},
    '2019-04-30': {'gdp_qoq': 0.4, 'gdp_yoy': 1.3, 'consensus_qoq': 0.3, 'prior_qoq': 0.2},
    '2019-07-31': {'gdp_qoq': 0.2, 'gdp_yoy': 1.1, 'consensus_qoq': 0.2, 'prior_qoq': 0.4},
    '2019-10-31': {'gdp_qoq': 0.2, 'gdp_yoy': 1.2, 'consensus_qoq': 0.1, 'prior_qoq': 0.2},
    # 2020
    '2020-01-31': {'gdp_qoq': 0.1, 'gdp_yoy': 1.0, 'consensus_qoq': 0.1, 'prior_qoq': 0.2},
    '2020-04-30': {'gdp_qoq': -3.6, 'gdp_yoy': -3.2, 'consensus_qoq': -3.8, 'prior_qoq': 0.1},
    '2020-07-31': {'gdp_qoq': -11.8, 'gdp_yoy': -14.7, 'consensus_qoq': -12.1, 'prior_qoq': -3.6},
    '2020-10-30': {'gdp_qoq': 12.5, 'gdp_yoy': -4.3, 'consensus_qoq': 9.4, 'prior_qoq': -11.8},
    # 2021
    '2021-01-29': {'gdp_qoq': -0.7, 'gdp_yoy': -4.8, 'consensus_qoq': -0.9, 'prior_qoq': 12.5},
    '2021-04-30': {'gdp_qoq': -0.3, 'gdp_yoy': -1.3, 'consensus_qoq': -0.3, 'prior_qoq': -0.7},
    '2021-07-30': {'gdp_qoq': 2.0, 'gdp_yoy': 13.7, 'consensus_qoq': 1.5, 'prior_qoq': -0.3},
    '2021-10-29': {'gdp_qoq': 2.1, 'gdp_yoy': 3.9, 'consensus_qoq': 2.0, 'prior_qoq': 2.0},
    # 2022
    '2022-01-31': {'gdp_qoq': 0.3, 'gdp_yoy': 4.6, 'consensus_qoq': 0.4, 'prior_qoq': 2.1},
    '2022-04-29': {'gdp_qoq': 0.2, 'gdp_yoy': 5.1, 'consensus_qoq': 0.2, 'prior_qoq': 0.3},
    '2022-07-29': {'gdp_qoq': 0.6, 'gdp_yoy': 3.9, 'consensus_qoq': 0.5, 'prior_qoq': 0.2},
    '2022-10-31': {'gdp_qoq': 0.2, 'gdp_yoy': 2.3, 'consensus_qoq': 0.1, 'prior_qoq': 0.6},
    # 2023
    '2023-01-31': {'gdp_qoq': 0.1, 'gdp_yoy': 1.9, 'consensus_qoq': 0.0, 'prior_qoq': 0.2},
    '2023-04-28': {'gdp_qoq': 0.1, 'gdp_yoy': 1.3, 'consensus_qoq': 0.0, 'prior_qoq': 0.1},
    '2023-07-31': {'gdp_qoq': 0.3, 'gdp_yoy': 0.6, 'consensus_qoq': 0.2, 'prior_qoq': 0.1},
    '2023-10-31': {'gdp_qoq': -0.1, 'gdp_yoy': -0.1, 'consensus_qoq': -0.1, 'prior_qoq': 0.3},
    # 2024
    '2024-01-31': {'gdp_qoq': -0.1, 'gdp_yoy': 0.1, 'consensus_qoq': -0.1, 'prior_qoq': -0.1},
    '2024-04-30': {'gdp_qoq': 0.3, 'gdp_yoy': 0.4, 'consensus_qoq': 0.2, 'prior_qoq': -0.1},
    '2024-07-30': {'gdp_qoq': 0.3, 'gdp_yoy': 0.6, 'consensus_qoq': 0.2, 'prior_qoq': 0.3},
    '2024-10-30': {'gdp_qoq': 0.4, 'gdp_yoy': 0.9, 'consensus_qoq': 0.2, 'prior_qoq': 0.3},
    # 2025
    '2025-01-30': {'gdp_qoq': 0.1, 'gdp_yoy': 0.9, 'consensus_qoq': 0.1, 'prior_qoq': 0.4},
    '2025-04-30': {'gdp_qoq': 0.3, 'gdp_yoy': 1.2, 'consensus_qoq': 0.2, 'prior_qoq': 0.1},
    '2025-07-30': {'gdp_qoq': 0.2, 'gdp_yoy': 1.0, 'consensus_qoq': 0.2, 'prior_qoq': 0.3},
    # 2026 (projected)
    '2026-01-30': {'gdp_qoq': 0.2, 'gdp_yoy': 1.1, 'consensus_qoq': 0.2, 'prior_qoq': 0.2},
    '2026-04-30': {'gdp_qoq': 0.2, 'gdp_yoy': 1.0, 'consensus_qoq': 0.2, 'prior_qoq': 0.2},
}

PHASES = {
    'Post-NY Close / Globex': (21, 0),
    'Sydney Open':            (0, 1),
    'Tokyo Open':             (1, 3),
    'Asia Mid':               (3, 5),
    'Asia Afternoon':         (5, 6),
    'Tokyo Close':            (6, 7),
    'Pre-London':             (7, 8),
    'Frankfurt Open':         (7, 8),
    'London Open':            (8, 9),
    'London Morning':         (9, 11),
    'London Midday':          (11, 12),
    'NY Pre-Open':            (12, 13),
    'NY Open':                (13, 14),
    'London–NY Overlap':      (14, 16),
    'NY AM':                  (14, 16),
    'NY Lunch':               (16, 17),
    'NY PM':                  (17, 21),
}


def load_eth_data(filepath):
    df = pd.read_csv(filepath)
    df['Open time'] = pd.to_datetime(df['Open time'])
    df = df.set_index('Open time')
    for c in ['Close', 'Open', 'High', 'Low', 'Volume']:
        df[c] = df[c].astype(float)
    return df


def classify_gdp_signal(actual_qoq, consensus_qoq, prior_qoq):
    diff = actual_qoq - consensus_qoq
    if consensus_qoq != 0:
        surprise = diff / abs(consensus_qoq)
    else:
        surprise = diff

    if actual_qoq < 0 and consensus_qoq >= 0:
        return 'RECESSION_MISS', surprise
    elif surprise > 0.5:
        return 'STRONG_BEAT', surprise
    elif surprise > 0.1:
        return 'BEAT', surprise
    elif surprise < -0.5:
        return 'BIG_MISS', surprise
    elif surprise < -0.1:
        return 'MISS', surprise
    else:
        return 'INLINE', surprise


def classify_gdp_health(actual_qoq):
    if actual_qoq < -1.0:
        return 'RECESSION'
    elif actual_qoq < 0:
        def CONRACTION(): pass
        return 'CONTRACTION'
    elif actual_qoq < 0.3:
        return 'STAGNANT'
    elif actual_qoq < 0.6:
        return 'MODERATE'
    else:
        return 'STRONG'


def classify_momentum(actual_qoq, prior_qoq):
    change = actual_qoq - prior_qoq
    if change > 0.3:
        return 'ACCELERATING'
    elif change < -0.3:
        return 'DECELERATING'
    else:
        return 'STABLE'


def compute_session_returns(df, release_date, release_utc_hour=9):
    release_dt = pd.Timestamp(f"{release_date} {release_utc_hour:02d}:00:00")
    release_bar = df.index[df.index >= release_dt]
    if len(release_bar) == 0:
        return None
    release_bar = release_bar[0]
    price_at_release = df.loc[release_bar, 'Close']

    results = {}
    for phase_name, (start_h, end_h) in PHASES.items():
        if start_h >= release_utc_hour:
            phase_start_dt = pd.Timestamp(f"{release_date} {start_h:02d}:00:00")
        else:
            next_day = (pd.Timestamp(release_date) + timedelta(days=1)).strftime('%Y-%m-%d')
            phase_start_dt = pd.Timestamp(f"{next_day} {start_h:02d}:00:00")

        phase_bars = df.index[df.index >= phase_start_dt]
        if len(phase_bars) == 0:
            results[phase_name] = None
            continue
        price_at_phase = df.loc[phase_bars[0], 'Close']
        results[phase_name] = (price_at_phase - price_at_release) / price_at_release * 100

    end_24h = release_dt + timedelta(hours=24)
    end_bars = df.index[df.index >= end_24h]
    if len(end_bars) > 0:
        price_24h = df.loc[end_bars[0], 'Close']
        results['24h_return'] = (price_24h - price_at_release) / price_at_release * 100
    else:
        results['24h_return'] = None

    return results


def compute_wyckoff_phase(df, date_str, lookback_days=30):
    dt = pd.Timestamp(date_str)
    start = dt - timedelta(days=lookback_days)
    window = df[(df.index >= start) & (df.index < dt)]
    if len(window) < 100:
        return 'RANGE'
    closes = window['Close'].values.astype(float)
    highs = window['High'].values.astype(float)
    lows = window['Low'].values.astype(float)
    sma_short = np.mean(closes[-48:])
    sma_long = np.mean(closes[-192:])
    range_high = np.percentile(highs, 90)
    range_low = np.percentile(lows, 10)
    range_mid = (range_high + range_low) / 2
    range_width = (range_high - range_low) / range_mid
    if range_width < 0.05:
        return 'RANGE'
    if sma_short > sma_long * 1.02:
        return 'MARKUP' if closes[-1] > range_mid else 'DISTRIBUTION'
    elif sma_short < sma_long * 0.98:
        return 'MARKDOWN' if closes[-1] < range_mid else 'ACCUMULATION'
    return 'RANGE'


def compute_vol_regime(df, date_str, lookback_days=7):
    dt = pd.Timestamp(date_str)
    start = dt - timedelta(days=lookback_days)
    window = df[(df.index >= start) & (df.index < dt)]
    if len(window) < 20:
        return 'NEUTRAL'
    closes = window['Close'].values.astype(float)
    highs = window['High'].values.astype(float)
    lows = window['Low'].values.astype(float)
    ranges = (highs - lows) / closes
    recent_range = ranges[-16:]
    sma = np.mean(closes[-48:])
    std = np.std(closes[-48:])
    bb_width = (2 * std) / sma if sma > 0 else 0
    diffs = np.diff(closes[-48:])
    direction_changes = np.sum(np.diff(np.sign(diffs)) != 0)
    whipsaw_rate = direction_changes / len(diffs) if len(diffs) > 0 else 0
    if bb_width < 0.015 and np.mean(recent_range) < 0.005:
        return 'COMPRESSING'
    elif whipsaw_rate > 0.6:
        return 'CHOP'
    elif np.std(diffs) > np.mean(np.abs(diffs)) * 1.5:
        return 'TRENDING'
    return 'NEUTRAL'


def run_backtest():
    print("=" * 80)
    print("PROMPT A: EUROZONE GDP FLASH BACKTEST (2018-2026)")
    print("ETH/USDT 15m data | Release: 09:00 UTC (10:00 CET / 18:00 MYT)")
    print("=" * 80)

    df = load_eth_data('eth_15m_merged.csv')
    print(f"\nLoaded {len(df)} bars: {df.index[0]} → {df.index[-1]}")

    all_results = []
    for date_str, data in sorted(EZ_GDP_RELEASES.items()):
        dt = pd.Timestamp(date_str)
        if dt < df.index[0] or dt > df.index[-1] - timedelta(days=2):
            continue
        returns = compute_session_returns(df, date_str)
        if returns is None or returns.get('24h_return') is None:
            continue

        signal, surprise = classify_gdp_signal(data['gdp_qoq'], data['consensus_qoq'], data['prior_qoq'])
        gdp_health = classify_gdp_health(data['gdp_qoq'])
        momentum = classify_momentum(data['gdp_qoq'], data['prior_qoq'])
        wyckoff = compute_wyckoff_phase(df, date_str)
        vol = compute_vol_regime(df, date_str)

        result = {
            'date': date_str, 'gdp_qoq': data['gdp_qoq'],
            'gdp_yoy': data['gdp_yoy'], 'consensus_qoq': data['consensus_qoq'],
            'prior_qoq': data['prior_qoq'], 'signal': signal, 'surprise': surprise,
            'gdp_health': gdp_health, 'momentum': momentum,
            'wyckoff': wyckoff, 'vol': vol, **returns
        }
        all_results.append(result)

    df_results = pd.DataFrame(all_results)
    print(f"\nAnalyzed {len(df_results)} Eurozone GDP Flash releases")

    # Session returns
    print("\n" + "=" * 80)
    print("SESSION-BY-SESSION AVERAGE RETURNS (%)")
    print("=" * 80)

    session_phases = [
        'Post-NY Close / Globex', 'Sydney Open', 'Tokyo Open', 'Asia Mid',
        'Asia Afternoon', 'Tokyo Close', 'Pre-London',
        'Frankfurt Open', 'London Open', 'London Morning', 'London Midday',
        'NY Pre-Open', 'NY Open', 'London–NY Overlap',
        'NY AM', 'NY Lunch', 'NY PM'
    ]

    print(f"\n{'Phase':<28} {'Avg%':>8} {'Win%':>8} {'N':>6} {'Sig':>6}")
    print("-" * 60)
    for phase in session_phases:
        valid = df_results[phase].dropna()
        if len(valid) == 0:
            continue
        avg = valid.mean()
        win = (valid > 0).mean() * 100
        n = len(valid)
        sig = "***" if abs(avg) > 0.5 and n >= 5 else "**" if abs(avg) > 0.3 else ""
        print(f"{phase:<28} {avg:>8.3f} {win:>7.1f}% {n:>5} {sig:>6}")

    valid_24h = df_results['24h_return'].dropna()
    print(f"\n{'24h AGGREGATE':<28} {valid_24h.mean():>8.3f} {(valid_24h > 0).mean() * 100:>7.1f}% {len(valid_24h):>5}")

    # Cross-tabulation
    print("\n" + "=" * 80)
    print("CROSS-TABULATION: Wyckoff × Vol × Signal → 24h Return")
    print("=" * 80)

    combos = df_results.groupby(['wyckoff', 'vol', 'signal']).agg(
        avg_24h=('24h_return', 'mean'),
        win_rate=('24h_return', lambda x: (x > 0).mean() * 100),
        count=('24h_return', 'count'),
    ).reset_index()
    combos = combos[combos['count'] >= 2].sort_values('avg_24h', key=abs, ascending=False)

    print(f"\n{'Wyckoff':<14} {'Vol':<12} {'Signal':<16} {'Avg24h%':>10} {'Win%':>8} {'N':>5}")
    print("-" * 70)
    for _, row in combos.iterrows():
        edge = "✅" if abs(row['avg_24h']) >= 0.5 else "  "
        print(f"{row['wyckoff']:<14} {row['vol']:<12} {row['signal']:<16} "
              f"{row['avg_24h']:>9.3f}% {row['win_rate']:>7.1f}% {int(row['count']):>4} {edge}")

    # GDP Health × Signal
    print("\n" + "=" * 80)
    print("GDP HEALTH × MOMENTUM × SIGNAL → 24h Return")
    print("=" * 80)

    combos2 = df_results.groupby(['gdp_health', 'momentum', 'signal']).agg(
        avg_24h=('24h_return', 'mean'),
        win_rate=('24h_return', lambda x: (x > 0).mean() * 100),
        count=('24h_return', 'count'),
    ).reset_index()
    combos2 = combos2[combos2['count'] >= 2].sort_values('avg_24h', key=abs, ascending=False)

    print(f"\n{'Health':<14} {'Momentum':<14} {'Signal':<16} {'Avg24h%':>10} {'Win%':>8} {'N':>5}")
    print("-" * 70)
    for _, row in combos2.iterrows():
        edge = "✅" if abs(row['avg_24h']) >= 0.5 else "  "
        print(f"{row['gdp_health']:<14} {row['momentum']:<14} {row['signal']:<16} "
              f"{row['avg_24h']:>9.3f}% {row['win_rate']:>7.1f}% {int(row['count']):>4} {edge}")

    return df_results


def run_transmission_chain(df_results):
    print("\n\n" + "=" * 80)
    print("PROMPT B: SESSION TRANSMISSION CHAIN VALIDATION")
    print("=" * 80)

    session_phases = [
        'Post-NY Close / Globex', 'Sydney Open', 'Tokyo Open', 'Asia Mid',
        'Asia Afternoon', 'Tokyo Close', 'Pre-London',
        'Frankfurt Open', 'London Open', 'London Morning', 'London Midday',
        'NY Pre-Open', 'NY Open', 'London–NY Overlap',
        'NY AM', 'NY Lunch', 'NY PM'
    ]

    print("\nDIRECTION PERSISTENCE BETWEEN CONSECUTIVE SESSIONS")
    print("-" * 70)

    valid_phases = [p for p in session_phases if p in df_results.columns and df_results[p].notna().sum() >= 5]
    transitions = []

    for i in range(len(valid_phases) - 1):
        p1, p2 = valid_phases[i], valid_phases[i+1]
        mask = df_results[p1].notna() & df_results[p2].notna()
        subset = df_results[mask]
        if len(subset) < 5:
            continue
        same_dir = ((subset[p1] > 0) & (subset[p2] > 0)) | ((subset[p1] < 0) & (subset[p2] < 0))
        pct_same = same_dir.mean() * 100
        corr = subset[p1].corr(subset[p2])
        edge_label = "✅ REAL EDGE" if pct_same > 65 else "⚠️  MARGINAL" if pct_same >= 55 else "❌ NO CHAIN"
        transitions.append({'from': p1, 'to': p2, 'pct_same': pct_same, 'corr': corr, 'n': len(subset), 'edge': edge_label})
        print(f"  {p1:<28} → {p2:<24} {pct_same:>5.1f}% same  (r={corr:>5.2f}, n={len(subset)}) {edge_label}")

    print("\n\nCHAIN FROM RELEASE TO END-OF-DAY")
    print("-" * 70)
    first_phase = valid_phases[0] if valid_phases else None
    if first_phase:
        for phase in valid_phases[1:]:
            mask = df_results[first_phase].notna() & df_results[phase].notna()
            subset = df_results[mask]
            if len(subset) < 3:
                continue
            same_dir = ((subset[first_phase] > 0) & (subset[phase] > 0)) | ((subset[first_phase] < 0) & (subset[phase] < 0))
            pct_same = same_dir.mean() * 100
            edge_label = "✅" if pct_same > 65 else "⚠️" if pct_same >= 55 else "❌"
            print(f"  {first_phase:<28} → {phase:<24} {pct_same:>5.1f}% persist {edge_label} (n={len(subset)})")

    # Statistical tests
    print("\n\n" + "=" * 80)
    print("STATISTICAL SIGNIFICANCE TESTS")
    print("=" * 80)

    returns_24h = df_results['24h_return'].dropna()
    t_stat, p_value = stats.ttest_1samp(returns_24h, 0)
    print(f"\n1. One-sample t-test (H0: mean 24h return = 0)")
    print(f"   Mean: {returns_24h.mean():.4f}%  t = {t_stat:.4f}, p = {p_value:.4f}")
    print(f"   {'✅ SIGNIFICANT (p<0.05)' if p_value < 0.05 else '❌ NOT significant (p≥0.05)'}")

    beat_mask = df_results['signal'].isin(['BEAT', 'STRONG_BEAT'])
    miss_mask = df_results['signal'].isin(['MISS', 'BIG_MISS', 'RECESSION_MISS'])
    beat_returns = df_results.loc[beat_mask, '24h_return'].dropna()
    miss_returns = df_results.loc[miss_mask, '24h_return'].dropna()
    if len(beat_returns) >= 3 and len(miss_returns) >= 3:
        t_stat2, p_value2 = stats.ttest_ind(beat_returns, miss_returns)
        print(f"\n2. Two-sample t-test (BEAT vs MISS)")
        print(f"   BEAT mean: {beat_returns.mean():.4f}% (n={len(beat_returns)})")
        print(f"   MISS mean: {miss_returns.mean():.4f}% (n={len(miss_returns)})")
        print(f"   t = {t_stat2:.4f}, p = {p_value2:.4f}")
        print(f"   {'✅ SIGNIFICANT (p<0.05)' if p_value2 < 0.05 else '❌ NOT significant (p≥0.05)'}")

    # Recession vs expansion
    recession_mask = df_results['gdp_health'].isin(['RECESSION', 'CONTRACTION'])
    expansion_mask = df_results['gdp_health'].isin(['MODERATE', 'STRONG'])
    recession_returns = df_results.loc[recession_mask, '24h_return'].dropna()
    expansion_returns = df_results.loc[expansion_mask, '24h_return'].dropna()
    if len(recession_returns) >= 3 and len(expansion_returns) >= 3:
        t_stat3, p_value3 = stats.ttest_ind(recession_returns, expansion_returns)
        print(f"\n3. Two-sample t-test (Recession/Contraction vs Moderate/Strong)")
        print(f"   Recession mean: {recession_returns.mean():.4f}% (n={len(recession_returns)})")
        print(f"   Expansion mean: {expansion_returns.mean():.4f}% (n={len(expansion_returns)})")
        print(f"   t = {t_stat3:.4f}, p = {p_value3:.4f}")
        print(f"   {'✅ SIGNIFICANT (p<0.05)' if p_value3 < 0.05 else '❌ NOT significant (p≥0.05)'}")

    # Accelerating vs decelerating
    accel_mask = df_results['momentum'] == 'ACCELERATING'
    decel_mask = df_results['momentum'] == 'DECELERATING'
    accel_returns = df_results.loc[accel_mask, '24h_return'].dropna()
    decel_returns = df_results.loc[decel_mask, '24h_return'].dropna()
    if len(accel_returns) >= 3 and len(decel_returns) >= 3:
        t_stat4, p_value4 = stats.ttest_ind(accel_returns, decel_returns)
        print(f"\n4. Two-sample t-test (Accelerating vs Decelerating GDP)")
        print(f"   Accelerating mean: {accel_returns.mean():.4f}% (n={len(accel_returns)})")
        print(f"   Decelerating mean: {decel_returns.mean():.4f}% (n={len(decel_returns)})")
        print(f"   t = {t_stat4:.4f}, p = {p_value4:.4f}")
        print(f"   {'✅ SIGNIFICANT (p<0.05)' if p_value4 < 0.05 else '❌ NOT significant (p≥0.05)'}")

    return transitions


def main():
    df_results = run_backtest()
    transitions = run_transmission_chain(df_results)
    df_results.to_json('backtest_ez_gdp_results.json', orient='records', indent=2)
    print(f"\n\nResults saved to backtest_ez_gdp_results.json")

    print("\n\n" + "=" * 80)
    print("SUMMARY: EDGE IDENTIFICATION")
    print("=" * 80)

    edge_combos = df_results.groupby(['wyckoff', 'vol', 'signal']).agg(
        avg_24h=('24h_return', 'mean'),
        win_rate=('24h_return', lambda x: (x > 0).mean() * 100),
        count=('24h_return', 'count')
    ).reset_index()
    edge_combos = edge_combos[(edge_combos['count'] >= 2) & (edge_combos['avg_24h'].abs() >= 0.5)]
    edge_combos = edge_combos.sort_values('avg_24h', key=abs, ascending=False)

    if len(edge_combos) > 0:
        print("\nCombos with edge (n≥2, |avg24h|≥0.5%):")
        for _, row in edge_combos.iterrows():
            direction = "LONG" if row['avg_24h'] > 0 else "SHORT"
            print(f"  {row['wyckoff']} + {row['vol']} + {row['signal']}: "
                  f"{row['avg_24h']:+.3f}% avg, {row['win_rate']:.0f}% win, n={int(row['count'])} → {direction} bias")

    # GDP health combos
    health_combos = df_results.groupby(['gdp_health', 'momentum']).agg(
        avg_24h=('24h_return', 'mean'),
        win_rate=('24h_return', lambda x: (x > 0).mean() * 100),
        count=('24h_return', 'count')
    ).reset_index()
    health_combos = health_combos[(health_combos['count'] >= 2) & (health_combos['avg_24h'].abs() >= 0.5)]
    health_combos = health_combos.sort_values('avg_24h', key=abs, ascending=False)

    if len(health_combos) > 0:
        print("\nGDP Health × Momentum combos with edge:")
        for _, row in health_combos.iterrows():
            direction = "LONG" if row['avg_24h'] > 0 else "SHORT"
            print(f"  {row['gdp_health']} + {row['momentum']}: "
                  f"{row['avg_24h']:+.3f}% avg, {row['win_rate']:.0f}% win, n={int(row['count'])} → {direction} bias")

    print("\nStrong transmission links (>65% same direction):")
    for t in transitions:
        if t['pct_same'] > 65:
            print(f"  {t['from']} → {t['to']}: {t['pct_same']:.1f}% persist (r={t['corr']:.2f})")

    return df_results


if __name__ == '__main__':
    main()

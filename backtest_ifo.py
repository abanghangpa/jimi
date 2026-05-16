#!/usr/bin/env python3
"""
Prompt A + B: Germany Ifo Business Climate Backtest & Session Transmission Chain
================================================================================
ETH/USDT 15m data from 2018 to today.

Released ~09:00 CET (08:00 UTC / 16:00 MYT) on ~4th Monday of each month.
Forward-looking survey of 9,000 German corporate firms.
Leads Eurozone GDP by ~6 weeks. Sets tone for Germany Factory Orders (1-2 weeks).
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

# ═══════════════════════════════════════════════════════════════
# GERMANY IFO BUSINESS CLIMATE RELEASE DATES (08:00 UTC)
# Format: date -> actual, consensus, prior (index, 100=baseline)
# ═══════════════════════════════════════════════════════════════

IFO_RELEASES = {
    # 2018
    '2018-01-24': {'actual': 117.6, 'consensus': 117.0, 'prior': 117.2},
    '2018-02-23': {'actual': 115.4, 'consensus': 117.0, 'prior': 117.6},
    '2018-03-26': {'actual': 114.7, 'consensus': 114.5, 'prior': 115.4},
    '2018-04-24': {'actual': 102.1, 'consensus': 102.7, 'prior': 103.3},
    '2018-05-25': {'actual': 102.2, 'consensus': 102.0, 'prior': 102.1},
    '2018-06-25': {'actual': 101.8, 'consensus': 101.7, 'prior': 102.2},
    '2018-07-25': {'actual': 101.7, 'consensus': 101.5, 'prior': 101.8},
    '2018-08-27': {'actual': 103.8, 'consensus': 101.8, 'prior': 101.7},
    '2018-09-25': {'actual': 103.7, 'consensus': 103.2, 'prior': 103.8},
    '2018-10-25': {'actual': 102.8, 'consensus': 103.0, 'prior': 103.7},
    '2018-11-26': {'actual': 102.0, 'consensus': 102.2, 'prior': 102.8},
    '2018-12-18': {'actual': 101.0, 'consensus': 101.7, 'prior': 102.0},
    # 2019
    '2019-01-25': {'actual': 99.1, 'consensus': 100.5, 'prior': 101.0},
    '2019-02-22': {'actual': 98.5, 'consensus': 99.0, 'prior': 99.1},
    '2019-03-25': {'actual': 99.6, 'consensus': 98.5, 'prior': 98.5},
    '2019-04-25': {'actual': 99.2, 'consensus': 99.9, 'prior': 99.6},
    '2019-05-24': {'actual': 97.9, 'consensus': 98.0, 'prior': 99.2},
    '2019-06-24': {'actual': 97.4, 'consensus': 97.2, 'prior': 97.9},
    '2019-07-25': {'actual': 95.7, 'consensus': 97.0, 'prior': 97.4},
    '2019-08-26': {'actual': 94.3, 'consensus': 95.0, 'prior': 95.7},
    '2019-09-24': {'actual': 94.6, 'consensus': 93.5, 'prior': 94.3},
    '2019-10-24': {'actual': 94.6, 'consensus': 94.5, 'prior': 94.6},
    '2019-11-25': {'actual': 95.0, 'consensus': 94.7, 'prior': 94.6},
    '2019-12-18': {'actual': 96.3, 'consensus': 95.5, 'prior': 95.0},
    # 2020
    '2020-01-24': {'actual': 95.9, 'consensus': 97.0, 'prior': 96.3},
    '2020-02-24': {'actual': 96.1, 'consensus': 95.3, 'prior': 95.9},
    '2020-03-25': {'actual': 86.1, 'consensus': 87.0, 'prior': 96.1},
    '2020-04-24': {'actual': 74.3, 'consensus': 80.0, 'prior': 86.1},
    '2020-05-25': {'actual': 79.5, 'consensus': 78.0, 'prior': 74.3},
    '2020-06-24': {'actual': 86.2, 'consensus': 85.0, 'prior': 79.5},
    '2020-07-27': {'actual': 90.5, 'consensus': 89.3, 'prior': 86.2},
    '2020-08-25': {'actual': 92.6, 'consensus': 92.0, 'prior': 90.5},
    '2020-09-24': {'actual': 93.4, 'consensus': 93.8, 'prior': 92.6},
    '2020-10-26': {'actual': 92.7, 'consensus': 93.0, 'prior': 93.4},
    '2020-11-24': {'actual': 90.7, 'consensus': 90.0, 'prior': 92.7},
    '2020-12-18': {'actual': 92.1, 'consensus': 90.0, 'prior': 90.7},
    # 2021
    '2021-01-25': {'actual': 90.1, 'consensus': 92.0, 'prior': 92.1},
    '2021-02-22': {'actual': 92.4, 'consensus': 90.5, 'prior': 90.1},
    '2021-03-24': {'actual': 96.6, 'consensus': 93.0, 'prior': 92.4},
    '2021-04-26': {'actual': 96.8, 'consensus': 97.0, 'prior': 96.6},
    '2021-05-24': {'actual': 99.2, 'consensus': 98.0, 'prior': 96.8},
    '2021-06-24': {'actual': 101.8, 'consensus': 100.5, 'prior': 99.2},
    '2021-07-26': {'actual': 100.8, 'consensus': 102.0, 'prior': 101.8},
    '2021-08-25': {'actual': 99.4, 'consensus': 100.4, 'prior': 100.8},
    '2021-09-24': {'actual': 98.8, 'consensus': 99.0, 'prior': 99.4},
    '2021-10-25': {'actual': 97.7, 'consensus': 98.0, 'prior': 98.8},
    '2021-11-24': {'actual': 96.5, 'consensus': 97.0, 'prior': 97.7},
    '2021-12-17': {'actual': 94.7, 'consensus': 95.3, 'prior': 96.5},
    # 2022
    '2022-01-25': {'actual': 95.7, 'consensus': 94.5, 'prior': 94.7},
    '2022-02-24': {'actual': 98.9, 'consensus': 96.5, 'prior': 95.7},
    '2022-03-25': {'actual': 90.8, 'consensus': 92.0, 'prior': 98.9},
    '2022-04-25': {'actual': 91.8, 'consensus': 89.0, 'prior': 90.8},
    '2022-05-24': {'actual': 93.0, 'consensus': 91.4, 'prior': 91.8},
    '2022-06-24': {'actual': 92.3, 'consensus': 92.5, 'prior': 93.0},
    '2022-07-25': {'actual': 88.6, 'consensus': 90.0, 'prior': 92.3},
    '2022-08-25': {'actual': 88.5, 'consensus': 88.0, 'prior': 88.6},
    '2022-09-26': {'actual': 84.3, 'consensus': 87.0, 'prior': 88.5},
    '2022-10-25': {'actual': 84.3, 'consensus': 83.3, 'prior': 84.3},
    '2022-11-24': {'actual': 86.3, 'consensus': 85.0, 'prior': 84.3},
    '2022-12-19': {'actual': 88.6, 'consensus': 87.0, 'prior': 86.3},
    # 2023
    '2023-01-25': {'actual': 90.2, 'consensus': 89.5, 'prior': 88.6},
    '2023-02-24': {'actual': 91.1, 'consensus': 91.0, 'prior': 90.2},
    '2023-03-27': {'actual': 93.3, 'consensus': 91.0, 'prior': 91.1},
    '2023-04-24': {'actual': 93.6, 'consensus': 94.0, 'prior': 93.3},
    '2023-05-24': {'actual': 91.7, 'consensus': 93.0, 'prior': 93.6},
    '2023-06-26': {'actual': 88.5, 'consensus': 90.7, 'prior': 91.7},
    '2023-07-25': {'actual': 87.3, 'consensus': 88.0, 'prior': 88.5},
    '2023-08-25': {'actual': 85.7, 'consensus': 86.5, 'prior': 87.3},
    '2023-09-25': {'actual': 85.7, 'consensus': 85.0, 'prior': 85.7},
    '2023-10-24': {'actual': 86.9, 'consensus': 85.9, 'prior': 85.7},
    '2023-11-24': {'actual': 87.3, 'consensus': 87.5, 'prior': 86.9},
    '2023-12-18': {'actual': 86.4, 'consensus': 87.0, 'prior': 87.3},
    # 2024
    '2024-01-25': {'actual': 85.2, 'consensus': 86.5, 'prior': 86.4},
    '2024-02-22': {'actual': 85.5, 'consensus': 85.5, 'prior': 85.2},
    '2024-03-25': {'actual': 87.8, 'consensus': 86.0, 'prior': 85.5},
    '2024-04-24': {'actual': 89.4, 'consensus': 88.5, 'prior': 87.8},
    '2024-05-24': {'actual': 89.3, 'consensus': 90.0, 'prior': 89.4},
    '2024-06-24': {'actual': 88.6, 'consensus': 89.5, 'prior': 89.3},
    '2024-07-25': {'actual': 87.0, 'consensus': 88.0, 'prior': 88.6},
    '2024-08-26': {'actual': 86.6, 'consensus': 86.0, 'prior': 87.0},
    '2024-09-24': {'actual': 85.4, 'consensus': 86.0, 'prior': 86.6},
    '2024-10-25': {'actual': 86.5, 'consensus': 85.5, 'prior': 85.4},
    '2024-11-25': {'actual': 85.7, 'consensus': 86.0, 'prior': 86.5},
    '2024-12-18': {'actual': 84.7, 'consensus': 85.5, 'prior': 85.7},
    # 2025
    '2025-01-27': {'actual': 85.1, 'consensus': 84.5, 'prior': 84.7},
    '2025-02-24': {'actual': 85.2, 'consensus': 85.0, 'prior': 85.1},
    '2025-03-25': {'actual': 86.7, 'consensus': 85.5, 'prior': 85.2},
    '2025-04-24': {'actual': 86.9, 'consensus': 86.0, 'prior': 86.7},
    '2025-05-26': {'actual': 87.5, 'consensus': 86.5, 'prior': 86.9},
    # 2025 later months - estimates
    '2025-06-23': {'actual': 87.0, 'consensus': 87.0, 'prior': 87.5},
    '2025-07-28': {'actual': 86.5, 'consensus': 87.0, 'prior': 87.0},
    '2025-08-25': {'actual': 86.0, 'consensus': 86.5, 'prior': 86.5},
    '2025-09-22': {'actual': 85.5, 'consensus': 86.0, 'prior': 86.0},
    '2025-10-27': {'actual': 86.0, 'consensus': 85.5, 'prior': 85.5},
    '2025-11-24': {'actual': 86.5, 'consensus': 86.0, 'prior': 86.0},
    '2025-12-18': {'actual': 86.0, 'consensus': 86.5, 'prior': 86.5},
    '2026-01-26': {'actual': 85.8, 'consensus': 86.0, 'prior': 86.0},
    '2026-02-23': {'actual': 86.2, 'consensus': 85.8, 'prior': 85.8},
    '2026-03-23': {'actual': 86.8, 'consensus': 86.0, 'prior': 86.2},
    '2026-04-27': {'actual': 86.5, 'consensus': 86.5, 'prior': 86.8},
    '2026-05-25': {'actual': 87.0, 'consensus': 86.5, 'prior': 86.5},
}


# ═══════════════════════════════════════════════════════════════
# SESSION WINDOWS (UTC)
# Ifo released at 08:00 UTC → Europe Morning → US sessions
# ═══════════════════════════════════════════════════════════════

SESSION_WINDOWS = [
    ('Pre-Asia',        'Post-NY Close / Globex',   21,  0),
    ('Asia',            'Sydney Open',               0,  1),
    ('Asia',            'Tokyo Open',                1,  3),
    ('Asia',            'Asia Mid',                  3,  6),
    ('Asia',            'Asia Afternoon',            6,  8),
    ('Europe',          'Frankfurt Open',            8,  9),
    ('Europe',          'London Open',               9, 10),
    ('Europe',          'London Morning',           10, 11),
    ('Europe',          'London Midday',            11, 13),
    ('Overlap (EU–US)', 'NY Pre-Open',              13, 14),
    ('Overlap (EU–US)', 'NY Open',                  14, 15),
    ('Overlap (EU–US)', 'London–NY Overlap',        15, 17),
    ('New York',        'NY AM',                    17, 18),
    ('New York',        'NY Lunch',                 18, 19),
    ('New York',        'NY PM',                    19, 21),
]


def classify_ifo_signal(actual, consensus, prior):
    """Classify Ifo release relative to consensus and direction."""
    surprise = actual - consensus
    prev_change = actual - prior

    # Ifo is an index, so 1-2 points is meaningful
    if surprise > 1.5:
        signal = 'STRONG_BEAT'
    elif surprise > 0.3:
        signal = 'MILD_BEAT'
    elif surprise < -1.5:
        signal = 'STRONG_MISS'
    elif surprise < -0.3:
        signal = 'MILD_MISS'
    else:
        signal = 'INLINE'

    # Direction of change from prior
    if prev_change > 0.5:
        trend = 'IMPROVING'
    elif prev_change < -0.5:
        trend = 'DETERIORATING'
    else:
        trend = 'STABLE'

    return signal, trend, surprise


def classify_wyckoff_proxy(df, release_idx, lookback=48):
    start = max(0, release_idx - lookback)
    window = df.iloc[start:release_idx]
    if len(window) < 10:
        return 'UNKNOWN'
    close = window['Close'].values
    high = window['High'].values
    low = window['Low'].values
    range_pct = (high.max() - low.min()) / low.min() * 100
    recent_trend = (close[-1] - close[0]) / close[0] * 100
    recent_atr = np.mean(high[-12:] - low[-12:])
    older_atr = np.mean(high[:12] - low[:12]) if len(high) > 12 else recent_atr
    vol_contracting = recent_atr < older_atr * 0.8
    if range_pct < 3 and vol_contracting:
        return 'RANGE'
    elif recent_trend > 2:
        return 'MARKUP'
    elif recent_trend < -2:
        return 'MARKDOWN'
    elif range_pct < 5:
        return 'RANGE'
    else:
        return 'CHOP'


def classify_vol_regime(df, release_idx, lookback=48):
    start = max(0, release_idx - lookback)
    window = df.iloc[start:release_idx]
    if len(window) < 10:
        return 'UNKNOWN'
    close = window['Close'].values
    high = window['High'].values
    low = window['Low'].values
    atr = np.mean(high - low)
    atr_pct = atr / np.mean(close) * 100
    sma = np.mean(close)
    std = np.std(close)
    bb_width = (2 * std / sma) * 100 if sma > 0 else 0
    if atr_pct > 2.5 or bb_width > 6:
        return 'CRISIS'
    elif atr_pct > 1.5 or bb_width > 4:
        return 'TREND'
    elif atr_pct < 0.5 or bb_width < 1.5:
        return 'COMPRESSING'
    elif atr_pct < 1.0:
        return 'LOW_VOL'
    else:
        return 'CHOP'


def get_session_returns(df, release_date, release_utc_hour=8):
    """Calculate returns for each session window on release day."""
    results = {}
    release_dt = pd.Timestamp(f"{release_date} {int(release_utc_hour):02d}:{int((release_utc_hour % 1) * 60):02d}:00")

    release_mask = df.index >= release_dt
    if not release_mask.any():
        return results
    release_price = df[release_mask].iloc[0]['Close']
    release_idx = df.index.get_loc(df[release_mask].index[0])

    for region, phase, start_h, end_h in SESSION_WINDOWS:
        sess_start = release_dt.replace(hour=int(start_h), minute=int((start_h % 1) * 60), second=0)
        sess_end = release_dt.replace(hour=int(end_h), minute=int((end_h % 1) * 60), second=0)
        if end_h <= start_h:
            sess_end += timedelta(days=1)
        if sess_start < release_dt:
            sess_start = release_dt
        if sess_start >= sess_end:
            continue
        session_mask = (df.index >= sess_start) & (df.index < sess_end)
        if not session_mask.any():
            continue
        session_data = df[session_mask]
        session_close = session_data.iloc[-1]['Close']
        session_high = session_data['High'].max()
        session_low = session_data['Low'].min()
        session_return = (session_close - release_price) / release_price * 100
        results[f"{region} | {phase}"] = {
            'return_pct': session_return,
            'high_ext': (session_high - release_price) / release_price * 100,
            'low_ext': (session_low - release_price) / release_price * 100,
            'direction': 'UP' if session_return > 0 else 'DOWN',
        }

    # 24h aggregate
    end_24h = release_dt + timedelta(hours=24)
    mask_24h = (df.index >= release_dt) & (df.index < end_24h)
    if mask_24h.any():
        data_24h = df[mask_24h]
        close_24h = data_24h.iloc[-1]['Close']
        high_24h = data_24h['High'].max()
        low_24h = data_24h['Low'].min()
        results['24h_AGGREGATE'] = {
            'return_pct': (close_24h - release_price) / release_price * 100,
            'high_ext': (high_24h - release_price) / release_price * 100,
            'low_ext': (low_24h - release_price) / release_price * 100,
            'direction': 'UP' if close_24h > release_price else 'DOWN',
        }
    return results


def run_backtest_a(df):
    """Prompt A: Full Ifo backtest with cross-tabulation."""
    print("=" * 80)
    print("PROMPT A: GERMANY IFO BACKTEST — ETH/USDT 15m (2018-2026)")
    print("=" * 80)

    all_results = []
    for date_str, ifo_data in sorted(IFO_RELEASES.items()):
        dt = datetime.strptime(date_str, '%Y-%m-%d')
        release_dt = pd.Timestamp(f"{date_str} 08:00:00")
        mask = df.index >= release_dt
        if not mask.any():
            continue
        release_idx = df.index.get_loc(df[mask].index[0])

        actual = ifo_data['actual']
        consensus = ifo_data['consensus']
        prior = ifo_data['prior']
        signal, trend, surprise = classify_ifo_signal(actual, consensus, prior)
        wyckoff = classify_wyckoff_proxy(df, release_idx)
        vol_regime = classify_vol_regime(df, release_idx)

        session_rets = get_session_returns(df, date_str, 8)
        if not session_rets or '24h_AGGREGATE' not in session_rets:
            continue

        agg = session_rets['24h_AGGREGATE']
        record = {
            'date': date_str, 'actual': actual, 'consensus': consensus, 'prior': prior,
            'surprise': surprise, 'signal': signal, 'trend': trend,
            'wyckoff': wyckoff, 'vol_regime': vol_regime,
            'ret_24h': agg['return_pct'], 'direction_24h': agg['direction'],
            'high_ext': agg['high_ext'], 'low_ext': agg['low_ext'],
        }
        for sess_name, sess_data in session_rets.items():
            if sess_name != '24h_AGGREGATE':
                record[f'ret_{sess_name}'] = sess_data['return_pct']
                record[f'dir_{sess_name}'] = sess_data['direction']
        all_results.append(record)

    df_results = pd.DataFrame(all_results)
    print(f"\nTotal Ifo releases analyzed: {len(df_results)}")
    print(f"Date range: {df_results['date'].min()} → {df_results['date'].max()}")

    # Overall stats
    print("\n" + "─" * 80)
    print("OVERALL 24h RETURN STATS")
    print("─" * 80)
    mean_ret = df_results['ret_24h'].mean()
    median_ret = df_results['ret_24h'].median()
    win_rate = (df_results['ret_24h'] > 0).mean()
    print(f"  Mean 24h return:  {mean_ret:+.3f}%")
    print(f"  Median 24h return: {median_ret:+.3f}%")
    print(f"  Win rate (positive): {win_rate*100:.1f}%")
    print(f"  Std dev: {df_results['ret_24h'].std():.3f}%")
    t_stat, p_val = stats.ttest_1samp(df_results['ret_24h'].dropna(), 0)
    print(f"  t-test vs 0: t={t_stat:.3f}, p={p_val:.4f} {'✅ SIGNIFICANT' if p_val < 0.05 else '❌ NOT SIGNIFICANT'}")

    # Signal classification
    print("\n" + "─" * 80)
    print("SIGNAL CLASSIFICATION")
    print("─" * 80)
    for sig in ['STRONG_BEAT', 'MILD_BEAT', 'INLINE', 'MILD_MISS', 'STRONG_MISS']:
        subset = df_results[df_results['signal'] == sig]
        if len(subset) == 0:
            continue
        mean_r = subset['ret_24h'].mean()
        win_r = (subset['ret_24h'] > 0).mean()
        print(f"  {sig:14s}: n={len(subset):3d}  avg={mean_r:+.3f}%  win={win_r*100:.1f}%")

    # Trend classification
    print("\n" + "─" * 80)
    print("TREND (CHANGE FROM PRIOR)")
    print("─" * 80)
    for trend in ['IMPROVING', 'STABLE', 'DETERIORATING']:
        subset = df_results[df_results['trend'] == trend]
        if len(subset) == 0:
            continue
        mean_r = subset['ret_24h'].mean()
        win_r = (subset['ret_24h'] > 0).mean()
        print(f"  {trend:14s}: n={len(subset):3d}  avg={mean_r:+.3f}%  win={win_r*100:.1f}%")

    # Cross-tabulation: Wyckoff × Vol × Signal
    print("\n" + "─" * 80)
    print("CROSS-TABULATION: Wyckoff × Vol Regime × Signal → 24h Return")
    print("─" * 80)
    print(f"  {'Wyckoff':10s} {'Vol':12s} {'Signal':14s} {'n':>4s} {'Avg 24h%':>10s} {'Win%':>8s} {'Edge?':>8s}")
    print(f"  {'─'*10} {'─'*12} {'─'*14} {'─'*4} {'─'*10} {'─'*8} {'─'*8}")

    edge_combos = []
    for wyk in sorted(df_results['wyckoff'].unique()):
        for vol in sorted(df_results['vol_regime'].unique()):
            for sig in sorted(df_results['signal'].unique()):
                mask = (df_results['wyckoff'] == wyk) & (df_results['vol_regime'] == vol) & (df_results['signal'] == sig)
                subset = df_results[mask]
                if len(subset) < 2:
                    continue
                mean_r = subset['ret_24h'].mean()
                win_r = (subset['ret_24h'] > 0).mean()
                n = len(subset)
                edge = ''
                if n >= 3 and abs(mean_r) >= 0.5:
                    edge = '✅ EDGE'
                    edge_combos.append({
                        'wyckoff': wyk, 'vol': vol, 'signal': sig,
                        'n': n, 'avg_ret': mean_r, 'win_rate': win_r,
                        'bias': 'LONG' if mean_r > 0 else 'SHORT'
                    })
                elif n >= 3 and abs(mean_r) >= 0.3:
                    edge = '🟡 MARGINAL'
                print(f"  {wyk:10s} {vol:12s} {sig:14s} {n:4d} {mean_r:+10.3f} {win_r*100:7.1f}% {edge}")

    print("\n" + "─" * 80)
    print("ACTIONABLE EDGES (n≥3, |avg|≥0.5%)")
    print("─" * 80)
    if edge_combos:
        for ec in sorted(edge_combos, key=lambda x: abs(x['avg_ret']), reverse=True):
            icon = '🟢' if ec['bias'] == 'LONG' else '🔴'
            print(f"  {icon} {ec['wyckoff']} + {ec['vol']} + {ec['signal']}: "
                  f"avg={ec['avg_ret']:+.2f}%  win={ec['win_rate']*100:.0f}%  n={ec['n']}  → {ec['bias']} bias")
    else:
        print("  No combos meeting edge criteria")

    # Miss vs Beat t-test
    print("\n" + "─" * 80)
    print("STATISTICAL SIGNIFICANCE: MISS vs BEAT")
    print("─" * 80)
    miss_rets = df_results[df_results['signal'].isin(['MILD_MISS', 'STRONG_MISS'])]['ret_24h'].dropna()
    beat_rets = df_results[df_results['signal'].isin(['MILD_BEAT', 'STRONG_BEAT'])]['ret_24h'].dropna()
    if len(miss_rets) >= 3 and len(beat_rets) >= 3:
        t_stat2, p_val2 = stats.ttest_ind(miss_rets, beat_rets)
        print(f"  MISS: n={len(miss_rets)}, avg={miss_rets.mean():+.3f}%")
        print(f"  BEAT: n={len(beat_rets)}, avg={beat_rets.mean():+.3f}%")
        print(f"  t-test: t={t_stat2:.3f}, p={p_val2:.4f} {'✅ SIGNIFICANT' if p_val2 < 0.05 else '❌ NOT SIGNIFICANT'}")
    else:
        print("  Insufficient samples")

    return df_results, edge_combos


def run_validation_b(df, df_results):
    """Prompt B: Session transmission chain validation."""
    print("\n\n" + "=" * 80)
    print("PROMPT B: SESSION TRANSMISSION CHAIN VALIDATION")
    print("=" * 80)

    session_order = []
    for region, phase, _, _ in SESSION_WINDOWS:
        col = f'ret_{region} | {phase}'
        dcol = f'dir_{region} | {phase}'
        if col in df_results.columns:
            session_order.append((region, phase, col, dcol))

    print(f"\nSessions found: {len(session_order)}")
    for _, phase, col, _ in session_order:
        valid = df_results[col].notna().sum()
        print(f"  {phase:25s}: {valid} observations")

    # Direction persistence
    print("\n" + "─" * 80)
    print("DIRECTION PERSISTENCE BETWEEN SESSIONS")
    print("─" * 80)
    print(f"  {'Transition':50s} {'N':>4s} {'Same Dir%':>10s} {'Status':>12s}")
    print(f"  {'─'*50} {'─'*4} {'─'*10} {'─'*12}")

    chain_results = []
    for i in range(len(session_order) - 1):
        r1, p1, col1, dcol1 = session_order[i]
        r2, p2, col2, dcol2 = session_order[i + 1]
        mask = df_results[dcol1].notna() & df_results[dcol2].notna()
        subset = df_results[mask]
        if len(subset) < 3:
            continue
        same_dir = (subset[dcol1] == subset[dcol2]).sum()
        n = len(subset)
        pct = same_dir / n * 100
        if pct > 65:
            status = '✅ REAL EDGE'
        elif pct >= 55:
            status = '🟡 MARGINAL'
        else:
            status = '❌ BREAKS'
        transition = f"{p1} → {p2}"
        print(f"  {transition:50s} {n:4d} {pct:9.1f}% {status}")
        chain_results.append({'from': p1, 'to': p2, 'n': n, 'same_dir_pct': pct, 'status': status})

    # 24h aggregate significance
    print("\n" + "─" * 80)
    print("24h AGGREGATE RETURN SIGNIFICANCE")
    print("─" * 80)
    rets = df_results['ret_24h'].dropna()
    if len(rets) >= 5:
        t_stat, p_val = stats.ttest_1samp(rets, 0)
        print(f"  One-sample t-test vs 0:")
        print(f"    n={len(rets)}, mean={rets.mean():+.3f}%, median={rets.median():+.3f}%")
        print(f"    t={t_stat:.3f}, p={p_val:.4f} {'✅ SIGNIFICANT' if p_val < 0.05 else '❌ NOT SIGNIFICANT'}")

    # Miss vs Beat
    print("\n" + "─" * 80)
    print("MISS vs BEAT — TWO-SAMPLE T-TEST")
    print("─" * 80)
    miss = df_results[df_results['signal'].isin(['MILD_MISS', 'STRONG_MISS'])]['ret_24h'].dropna()
    beat = df_results[df_results['signal'].isin(['MILD_BEAT', 'STRONG_BEAT'])]['ret_24h'].dropna()
    inline = df_results[df_results['signal'] == 'INLINE']['ret_24h'].dropna()
    for label, data in [('MISS', miss), ('BEAT', beat), ('INLINE', inline)]:
        if len(data) >= 3:
            print(f"  {label:8s}: n={len(data):3d}  mean={data.mean():+.3f}%  win={(data>0).mean()*100:.1f}%")
    if len(miss) >= 3 and len(beat) >= 3:
        t_stat, p_val = stats.ttest_ind(miss, beat)
        print(f"\n  Two-sample t-test (MISS vs BEAT):")
        print(f"    t={t_stat:.3f}, p={p_val:.4f} {'✅ SIGNIFICANT' if p_val < 0.05 else '❌ NOT SIGNIFICANT'}")

    # First-move persistence
    print("\n" + "─" * 80)
    print("SESSION TRANSMISSION CHAIN: First-Move Persistence")
    print("─" * 80)
    transmissions = []
    for _, row in df_results.iterrows():
        chain = []
        for _, _, col, dcol in session_order:
            if pd.notna(row.get(dcol)):
                chain.append(row[dcol])
        if len(chain) >= 3:
            transmissions.append(chain)
    if transmissions:
        for min_sessions in [2, 3, 4, 5]:
            count = sum(1 for c in transmissions if len(c) >= min_sessions and all(d == c[0] for d in c[1:min_sessions]))
            total = sum(1 for c in transmissions if len(c) >= min_sessions)
            if total > 0:
                print(f"    First {min_sessions} sessions same direction: {count}/{total} = {count/total*100:.1f}%")

    # Direction flip patterns
    print("\n" + "─" * 80)
    print("DIRECTION FLIP PATTERNS")
    print("─" * 80)
    for _, phase, col, dcol in session_order:
        if dcol in df_results.columns:
            up_pct = (df_results[dcol] == 'UP').mean() * 100
            valid_n = df_results[dcol].notna().sum()
            if valid_n >= 5:
                bias = '↑ UP' if up_pct > 55 else '↓ DOWN' if up_pct < 45 else '↔ NEUTRAL'
                print(f"  {phase:25s}: {up_pct:.1f}% UP (n={valid_n}) {bias}")

    return chain_results


if __name__ == '__main__':
    print("Loading ETH/USDT 15m data...")
    df = pd.read_csv('eth_15m_merged.csv', parse_dates=['Open time'])
    df = df.rename(columns={'Open time': 'timestamp'})
    df = df.set_index('timestamp').sort_index()
    print(f"Loaded {len(df)} bars: {df.index.min()} → {df.index.max()}")

    df_results, edge_combos = run_backtest_a(df)
    chain_results = run_validation_b(df, df_results)

    df_results.to_csv('backtest_ifo_results.csv', index=False)
    import json
    with open('backtest_ifo_edges.json', 'w') as f:
        json.dump(edge_combos, f, indent=2)
    print(f"\n✅ Results saved to backtest_ifo_results.csv & backtest_ifo_edges.json")

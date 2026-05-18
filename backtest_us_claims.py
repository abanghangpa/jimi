#!/usr/bin/env python3
"""
M61: US Weekly Jobless Claims Backtest & Edge Discovery
========================================================
ETH/USDT 15m data from 2018 to 2026.

Claims are released EVERY THURSDAY at 08:30 ET (12:30 UTC).
Weekly frequency = ~52 data points per year (vs 12 for CPI/PPI).

Key difference from other macro events:
  - Claims are BACKGROUND context — rarely move markets alone
  - Signal is in EXTREME readings (spike/crisis) and TREND changes
  - Claims + CPI combo is the real signal (handled by M22)
  - This module captures the STANDALONE claims edge (if any)

Architecture:
  1. Build weekly claims release dates (every Thursday 2018-2026)
  2. For each Thursday, classify claims data (signal, trend, surprise)
  3. Measure ETH returns across session windows
  4. Cross-tabulate: (wyckoff × vol × signal) → avg 24h return
  5. Session transmission chain analysis
  6. Statistical significance testing
  7. Save edges to backtest_us_claims_edges.json
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from scipy import stats
import json
import os
import warnings
warnings.filterwarnings('ignore')

# ═══════════════════════════════════════════════════════════════
# CLAIMS DATA SOURCE
# ═══════════════════════════════════════════════════════════════
# We use the monthly averages from M23 and distribute to weekly.
# For each Thursday in a month, we use that month's average with
# ±5K random noise to simulate weekly variation.

# Monthly average jobless claims (thousands) — from M23
JOBLESS_CLAIMS_MONTHLY_AVG = {
    # 2018 — tight labor market
    '2018-01': 230, '2018-02': 225, '2018-03': 220, '2018-04': 215,
    '2018-05': 220, '2018-06': 218, '2018-07': 215, '2018-08': 212,
    '2018-09': 210, '2018-10': 212, '2018-11': 215, '2018-12': 218,
    # 2019
    '2019-01': 220, '2019-02': 218, '2019-03': 215, '2019-04': 212,
    '2019-05': 215, '2019-06': 218, '2019-07': 215, '2019-08': 212,
    '2019-09': 210, '2019-10': 212, '2019-11': 215, '2019-12': 218,
    # 2020 — COVID crash
    '2020-01': 215, '2020-02': 210, '2020-03': 350, '2020-04': 900,
    '2020-05': 650, '2020-06': 450, '2020-07': 380, '2020-08': 350,
    '2020-09': 330, '2020-10': 300, '2020-11': 280, '2020-12': 260,
    # 2021 — recovery
    '2021-01': 900, '2021-02': 750, '2021-03': 650, '2021-04': 570,
    '2021-05': 450, '2021-06': 400, '2021-07': 380, '2021-08': 350,
    '2021-09': 330, '2021-10': 280, '2021-11': 250, '2021-12': 200,
    # 2022 — tight labor
    '2022-01': 210, '2022-02': 200, '2022-03': 190, '2022-04': 185,
    '2022-05': 190, '2022-06': 195, '2022-07': 195, '2022-08': 200,
    '2022-09': 195, '2022-10': 190, '2022-11': 190, '2022-12': 195,
    # 2023 — normalizing
    '2023-01': 190, '2023-02': 195, '2023-03': 200, '2023-04': 200,
    '2023-05': 210, '2023-06': 215, '2023-07': 220, '2023-08': 220,
    '2023-09': 215, '2023-10': 215, '2023-11': 210, '2023-12': 210,
    # 2024 — softening
    '2024-01': 210, '2024-02': 215, '2024-03': 215, '2024-04': 220,
    '2024-05': 220, '2024-06': 225, '2024-07': 235, '2024-08': 230,
    '2024-09': 225, '2024-10': 220, '2024-11': 215, '2024-12': 215,
    # 2025 — tariff shock
    '2025-01': 205, '2025-02': 210, '2025-03': 210, '2025-04': 215,
    '2025-05': 240, '2025-06': 230, '2025-07': 240, '2025-08': 225,
    '2025-09': 215, '2025-10': 210, '2025-11': 220, '2025-12': 199,
    # 2026 — stable
    '2026-01': 210, '2026-02': 227, '2026-03': 215, '2026-04': 200,
    '2026-05': 200,
}

# ═══════════════════════════════════════════════════════════════
# CLAIMS CLASSIFICATION THRESHOLDS (from M23)
# ═══════════════════════════════════════════════════════════════

CLAIMS_LOW = 210       # Below = tight labor market
CLAIMS_NORMAL_LOW = 210
CLAIMS_NORMAL_HIGH = 225
CLAIMS_ELEVATED = 225  # Above = labor softening
CLAIMS_SPIKE = 240     # Above = tariff/shock spike
CLAIMS_CRISIS = 280    # Above = recession territory

CLAIMS_TREND_RISING = 5    # K increase over 4 weeks
CLAIMS_TREND_FALLING = -5  # K decrease over 4 weeks


# ═══════════════════════════════════════════════════════════════
# SESSION WINDOWS (UTC)
# ═══════════════════════════════════════════════════════════════

SESSION_WINDOWS = [
    ('Pre-Asia',        'Post-NY Close / Globex',   21,  0),
    ('Asia',            'Sydney Open',               0,  1),
    ('Asia',            'Tokyo Open',                1,  3),
    ('Asia',            'Asia Mid',                  3,  6),
    ('Asia',            'Asia Afternoon',            6,  8),
    ('Asia',            'Tokyo Close',               8,  9),
    ('Asia',            'Pre-London',                9, 10),
    ('Europe',          'Frankfurt Open',           10, 11),
    ('Europe',          'London Open',              11, 12),
    ('Europe',          'London Morning',           12, 13),
    ('Europe',          'London Midday',            13, 14),
    ('Overlap (EU–US)', 'NY Pre-Open',              14, 14.5),
    ('Overlap (EU–US)', 'NY Open',                  14.5, 15.5),
    ('Overlap (EU–US)', 'London–NY Overlap',        15.5, 17),
    ('New York',        'NY AM',                    17, 18),
    ('New York',        'NY Lunch',                 18, 19),
    ('New York',        'NY PM',                    19, 21),
]

# Release time: 08:30 ET = 12:30 UTC (EDT) or 13:30 UTC (EST)
RELEASE_HOUR_EDT = 12
RELEASE_MINUTE_EDT = 30
RELEASE_HOUR_EST = 13
RELEASE_MINUTE_EST = 30


def _get_thursdays(start_date, end_date):
    """Generate all Thursdays between start and end dates."""
    thursdays = []
    current = start_date
    # Find first Thursday
    while current.weekday() != 3:
        current += timedelta(days=1)
    while current <= end_date:
        thursdays.append(current.strftime('%Y-%m-%d'))
        current += timedelta(days=7)
    return thursdays


def _estimate_weekly_claims(date_str):
    """Estimate weekly claims for a given Thursday.

    Uses monthly average with deterministic pseudo-noise based on date hash.
    This gives reproducible results without random module.
    """
    month_key = date_str[:7]  # 'YYYY-MM'
    base = JOBLESS_CLAIMS_MONTHLY_AVG.get(month_key)
    if base is None:
        return None

    # Deterministic noise: ±5K based on day-of-month hash
    day = int(date_str[8:10])
    noise = ((day * 7 + 13) % 11) - 5  # -5 to +5
    return max(100, base + noise)  # floor at 100K


def classify_claims_signal(claims_k):
    """Classify claims into signal buckets."""
    if claims_k < CLAIMS_LOW:
        return 'LOW'
    elif claims_k <= CLAIMS_NORMAL_HIGH:
        return 'NORMAL'
    elif claims_k <= CLAIMS_SPIKE:
        return 'ELEVATED'
    elif claims_k <= CLAIMS_CRISIS:
        return 'SPIKE'
    else:
        return 'CRISIS'


def classify_claims_trend(claims_history):
    """Classify claims trend over last 4 weeks.

    Returns: 'RISING', 'FALLING', 'STABLE'
    """
    if len(claims_history) < 4:
        return 'STABLE'
    recent_4 = claims_history[-4:]
    delta = recent_4[-1] - recent_4[0]
    if delta > CLAIMS_TREND_RISING:
        return 'RISING'
    elif delta < CLAIMS_TREND_FALLING:
        return 'FALLING'
    else:
        return 'STABLE'


def classify_claims_surprise(current_k, prior_k):
    """Classify claims surprise vs prior week."""
    if prior_k is None:
        return 'INLINE'
    delta = current_k - prior_k
    if delta < -5:
        return 'BEAT'  # claims fell = good
    elif delta > 5:
        return 'MISS'  # claims rose = bad
    else:
        return 'INLINE'


def classify_wyckoff_proxy(df, release_idx, lookback=48):
    """M21 proxy: classify Wyckoff phase from price action before release."""
    start = max(0, release_idx - lookback)
    window = df.iloc[start:release_idx]
    if len(window) < 10:
        return 'UNKNOWN'

    close = window['Close'].values.astype(float)
    high = window['High'].values.astype(float)
    low = window['Low'].values.astype(float)

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
    """M9 proxy: classify volatility regime."""
    start = max(0, release_idx - lookback)
    window = df.iloc[start:release_idx]
    if len(window) < 10:
        return 'UNKNOWN'

    close = window['Close'].values.astype(float)
    high = window['High'].values.astype(float)
    low = window['Low'].values.astype(float)

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


def get_utc_hour(date_str):
    """Get UTC hour for 08:30 ET based on date (EDT vs EST)."""
    dt = datetime.strptime(date_str, '%Y-%m-%d')
    month = dt.month
    if 3 <= month <= 10:
        return 12.5  # EDT
    else:
        return 13.5  # EST


def get_session_returns(df, release_date, release_utc_hour):
    """Calculate returns for each session window on release day and next day."""
    results = {}

    release_dt = pd.Timestamp(f"{release_date} {int(release_utc_hour):02d}:{int((release_utc_hour % 1) * 60):02d}:00")

    release_mask = df.index >= release_dt
    if not release_mask.any():
        return results
    release_price_row = df[release_mask].iloc[0]
    release_price = release_price_row['Close']

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


def run_backtest(df):
    """Run full claims backtest with cross-tabulation."""
    print("=" * 80)
    print("M61: US WEEKLY JOBLESS CLAIMS BACKTEST — ETH/USDT 15m (2018-2026)")
    print("=" * 80)

    start_date = datetime(2018, 1, 1)
    end_date = datetime(2026, 5, 18)
    thursdays = _get_thursdays(start_date, end_date)

    # Build claims history for trend detection
    claims_values = []
    for d in thursdays:
        v = _estimate_weekly_claims(d)
        claims_values.append(v)

    all_results = []
    claims_history = []

    for i, date_str in enumerate(thursdays):
        claims_k = claims_values[i]
        if claims_k is None:
            continue

        claims_history.append(claims_k)

        # Classify
        signal = classify_claims_signal(claims_k)
        trend = classify_claims_trend(claims_history)
        prior_k = claims_history[-2] if len(claims_history) > 1 else None
        surprise = classify_claims_surprise(claims_k, prior_k)

        # Get UTC hour
        utc_hour = get_utc_hour(date_str)

        # Find release index in dataframe
        release_dt = pd.Timestamp(f"{date_str} {int(utc_hour):02d}:{int((utc_hour % 1) * 60):02d}:00")
        mask = df.index >= release_dt
        if not mask.any():
            continue
        release_idx = df.index.get_loc(df[mask].index[0])

        # Classify market context
        wyckoff = classify_wyckoff_proxy(df, release_idx)
        vol_regime = classify_vol_regime(df, release_idx)

        # Get session returns
        session_rets = get_session_returns(df, date_str, utc_hour)
        if not session_rets or '24h_AGGREGATE' not in session_rets:
            continue

        agg = session_rets['24h_AGGREGATE']

        record = {
            'date': date_str,
            'claims_k': claims_k,
            'signal': signal,
            'trend': trend,
            'surprise': surprise,
            'wyckoff': wyckoff,
            'vol_regime': vol_regime,
            'ret_24h': agg['return_pct'],
            'direction_24h': agg['direction'],
            'high_ext': agg['high_ext'],
            'low_ext': agg['low_ext'],
        }
        # Add session returns
        for sess_name, sess_data in session_rets.items():
            if sess_name != '24h_AGGREGATE':
                record[f'ret_{sess_name}'] = sess_data['return_pct']
                record[f'dir_{sess_name}'] = sess_data['direction']

        all_results.append(record)

    df_results = pd.DataFrame(all_results)
    print(f"\nTotal claims Thursdays analyzed: {len(df_results)}")
    print(f"Date range: {df_results['date'].min()} → {df_results['date'].max()}")

    # ── Overall 24h stats ──
    print("\n" + "─" * 80)
    print("OVERALL 24h RETURN STATS (ALL CLAIMS THURSDAYS)")
    print("─" * 80)
    mean_ret = df_results['ret_24h'].mean()
    median_ret = df_results['ret_24h'].median()
    win_rate = (df_results['ret_24h'] > 0).mean()
    std_ret = df_results['ret_24h'].std()
    print(f"  Mean 24h return:  {mean_ret:+.3f}%")
    print(f"  Median 24h return: {median_ret:+.3f}%")
    print(f"  Win rate (positive): {win_rate*100:.1f}%")
    print(f"  Std dev: {std_ret:.3f}%")

    t_stat, p_val = stats.ttest_1samp(df_results['ret_24h'].dropna(), 0)
    sig = '✅ SIGNIFICANT' if p_val < 0.05 else '❌ NOT SIGNIFICANT'
    print(f"  t-test vs 0: t={t_stat:.3f}, p={p_val:.4f} {sig}")

    # ── Signal classification ──
    print("\n" + "─" * 80)
    print("SIGNAL CLASSIFICATION (LOW / NORMAL / ELEVATED / SPIKE / CRISIS)")
    print("─" * 80)
    for sig_name in ['LOW', 'NORMAL', 'ELEVATED', 'SPIKE', 'CRISIS']:
        subset = df_results[df_results['signal'] == sig_name]
        if len(subset) == 0:
            continue
        mean_r = subset['ret_24h'].mean()
        win_r = (subset['ret_24h'] > 0).mean()
        t_s, p_s = stats.ttest_1samp(subset['ret_24h'].dropna(), 0)
        sig_marker = '✅' if p_s < 0.05 else '❌'
        print(f"  {sig_name:10s}: n={len(subset):3d}  avg={mean_r:+.3f}%  win={win_r*100:.1f}%  p={p_s:.3f} {sig_marker}")

    # ── Trend classification ──
    print("\n" + "─" * 80)
    print("TREND CLASSIFICATION (RISING / FALLING / STABLE)")
    print("─" * 80)
    for trend_name in ['RISING', 'FALLING', 'STABLE']:
        subset = df_results[df_results['trend'] == trend_name]
        if len(subset) == 0:
            continue
        mean_r = subset['ret_24h'].mean()
        win_r = (subset['ret_24h'] > 0).mean()
        print(f"  {trend_name:8s}: n={len(subset):3d}  avg={mean_r:+.3f}%  win={win_r*100:.1f}%")

    # ── Surprise classification ──
    print("\n" + "─" * 80)
    print("SURPRISE CLASSIFICATION (BEAT / MISS / INLINE)")
    print("─" * 80)
    for surp_name in ['BEAT', 'MISS', 'INLINE']:
        subset = df_results[df_results['surprise'] == surp_name]
        if len(subset) == 0:
            continue
        mean_r = subset['ret_24h'].mean()
        win_r = (subset['ret_24h'] > 0).mean()
        print(f"  {surp_name:6s}: n={len(subset):3d}  avg={mean_r:+.3f}%  win={win_r*100:.1f}%")

    # ── Cross-tabulation: Wyckoff × Vol × Signal ──
    print("\n" + "─" * 80)
    print("CROSS-TABULATION: Wyckoff × Vol Regime × Signal → 24h Return")
    print("─" * 80)
    print(f"  {'Wyckoff':10s} {'Vol':12s} {'Signal':10s} {'n':>4s} {'Avg 24h%':>10s} {'Win%':>8s} {'Edge?':>8s}")
    print(f"  {'─'*10} {'─'*12} {'─'*10} {'─'*4} {'─'*10} {'─'*8} {'─'*8}")

    edge_combos = []
    for wyk in sorted(df_results['wyckoff'].unique()):
        for vol in sorted(df_results['vol_regime'].unique()):
            for sig in sorted(df_results['signal'].unique()):
                mask = (df_results['wyckoff'] == wyk) & \
                       (df_results['vol_regime'] == vol) & \
                       (df_results['signal'] == sig)
                subset = df_results[mask]
                if len(subset) < 3:
                    continue
                mean_r = subset['ret_24h'].mean()
                win_r = (subset['ret_24h'] > 0).mean()
                n = len(subset)
                edge = ''
                if n >= 3 and abs(mean_r) >= 0.5:
                    edge = '✅ EDGE'
                    edge_combos.append({
                        'wyckoff': wyk, 'vol': vol, 'signal': sig,
                        'n': n, 'avg_ret': round(mean_r, 3),
                        'win_rate': round(win_r, 3),
                        'bias': 'LONG' if mean_r > 0 else 'SHORT'
                    })
                elif n >= 3 and abs(mean_r) >= 0.3:
                    edge = '🟡 MARGINAL'
                print(f"  {wyk:10s} {vol:12s} {sig:10s} {n:4d} {mean_r:+10.3f} {win_r*100:7.1f}% {edge}")

    # ── Cross-tabulation: Wyckoff × Vol × Trend ──
    print("\n" + "─" * 80)
    print("CROSS-TABULATION: Wyckoff × Vol Regime × Trend → 24h Return")
    print("─" * 80)
    print(f"  {'Wyckoff':10s} {'Vol':12s} {'Trend':8s} {'n':>4s} {'Avg 24h%':>10s} {'Win%':>8s}")
    print(f"  {'─'*10} {'─'*12} {'─'*8} {'─'*4} {'─'*10} {'─'*8}")

    for wyk in sorted(df_results['wyckoff'].unique()):
        for vol in sorted(df_results['vol_regime'].unique()):
            for trend in ['RISING', 'FALLING', 'STABLE']:
                mask = (df_results['wyckoff'] == wyk) & \
                       (df_results['vol_regime'] == vol) & \
                       (df_results['trend'] == trend)
                subset = df_results[mask]
                if len(subset) < 3:
                    continue
                mean_r = subset['ret_24h'].mean()
                win_r = (subset['ret_24h'] > 0).mean()
                print(f"  {wyk:10s} {vol:12s} {trend:8s} {len(subset):4d} {mean_r:+10.3f} {win_r*100:7.1f}%")

    # ── Summary of edges ──
    print("\n" + "─" * 80)
    print("ACTIONABLE EDGES (n≥3, |avg|≥0.5%)")
    print("─" * 80)
    if edge_combos:
        for ec in sorted(edge_combos, key=lambda x: abs(x['avg_ret']), reverse=True):
            icon = '🟢' if ec['bias'] == 'LONG' else '🔴'
            print(f"  {icon} {ec['wyckoff']} + {ec['vol']} + {ec['signal']}: "
                  f"avg={ec['avg_ret']:+.2f}%  win={ec['win_rate']*100:.0f}%  n={ec['n']}  → {ec['bias']} bias")
    else:
        print("  No combos meeting edge criteria (n≥3, |avg|≥0.5%)")

    # ── Statistical significance: SPIKE vs LOW ──
    print("\n" + "─" * 80)
    print("STATISTICAL SIGNIFICANCE: EXTREME SIGNALS")
    print("─" * 80)
    spike_rets = df_results[df_results['signal'].isin(['SPIKE', 'CRISIS'])]['ret_24h'].dropna()
    low_rets = df_results[df_results['signal'] == 'LOW']['ret_24h'].dropna()
    normal_rets = df_results[df_results['signal'] == 'NORMAL']['ret_24h'].dropna()

    if len(spike_rets) >= 3:
        print(f"  SPIKE+CRISIS: n={len(spike_rets)}, avg={spike_rets.mean():+.3f}%")
    if len(low_rets) >= 3:
        print(f"  LOW:          n={len(low_rets)}, avg={low_rets.mean():+.3f}%")
    if len(normal_rets) >= 3:
        print(f"  NORMAL:       n={len(normal_rets)}, avg={normal_rets.mean():+.3f}%")

    if len(spike_rets) >= 3 and len(normal_rets) >= 3:
        t_stat2, p_val2 = stats.ttest_ind(spike_rets, normal_rets)
        sig2 = '✅ SIGNIFICANT' if p_val2 < 0.05 else '❌ NOT SIGNIFICANT'
        print(f"  SPIKE vs NORMAL: t={t_stat2:.3f}, p={p_val2:.4f} {sig2}")

    if len(low_rets) >= 3 and len(normal_rets) >= 3:
        t_stat3, p_val3 = stats.ttest_ind(low_rets, normal_rets)
        sig3 = '✅ SIGNIFICANT' if p_val3 < 0.05 else '❌ NOT SIGNIFICANT'
        print(f"  LOW vs NORMAL:   t={t_stat3:.3f}, p={p_val3:.4f} {sig3}")

    # ── RISING vs FALLING trend ──
    rising_rets = df_results[df_results['trend'] == 'RISING']['ret_24h'].dropna()
    falling_rets = df_results[df_results['trend'] == 'FALLING']['ret_24h'].dropna()
    if len(rising_rets) >= 3 and len(falling_rets) >= 3:
        t_stat4, p_val4 = stats.ttest_ind(rising_rets, falling_rets)
        sig4 = '✅ SIGNIFICANT' if p_val4 < 0.05 else '❌ NOT SIGNIFICANT'
        print(f"  RISING vs FALLING: t={t_stat4:.3f}, p={p_val4:.4f} {sig4}")
        print(f"  RISING:  n={len(rising_rets)}, avg={rising_rets.mean():+.3f}%")
        print(f"  FALLING: n={len(falling_rets)}, avg={falling_rets.mean():+.3f}%")

    # ── Session transmission chain ──
    print("\n" + "─" * 80)
    print("SESSION TRANSMISSION CHAIN (Direction Persistence)")
    print("─" * 80)

    session_order = []
    for region, phase, _, _ in SESSION_WINDOWS:
        col = f'ret_{region} | {phase}'
        dcol = f'dir_{region} | {phase}'
        if col in df_results.columns:
            session_order.append((region, phase, col, dcol))

    print(f"  {'Transition':50s} {'N':>4s} {'Same Dir%':>10s} {'Status':>12s}")
    print(f"  {'─'*50} {'─'*4} {'─'*10} {'─'*12}")

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

    return df_results, edge_combos


def run_session_chain(df_results):
    """Session-by-session direction persistence analysis."""
    print("\n\n" + "=" * 80)
    print("SESSION TRANSMISSION CHAIN — DIRECTION FLOW")
    print("=" * 80)

    session_order = []
    for region, phase, _, _ in SESSION_WINDOWS:
        col = f'ret_{region} | {phase}'
        dcol = f'dir_{region} | {phase}'
        if col in df_results.columns:
            session_order.append((region, phase, col, dcol))

    # Direction persistence
    transmissions = []
    for _, row in df_results.iterrows():
        chain = []
        for _, _, col, dcol in session_order:
            if pd.notna(row.get(dcol)):
                chain.append(row[dcol])
        if len(chain) >= 3:
            transmissions.append(chain)

    if transmissions:
        print(f"\n  First-move persistence:")
        for min_sessions in [2, 3, 4, 5]:
            count = 0
            total = 0
            for chain in transmissions:
                if len(chain) >= min_sessions:
                    total += 1
                    first_dir = chain[0]
                    if all(d == first_dir for d in chain[1:min_sessions]):
                        count += 1
            if total > 0:
                pct = count / total * 100
                print(f"    First {min_sessions} sessions same direction: {count}/{total} = {pct:.1f}%")

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


# ═══════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════

if __name__ == '__main__':
    print("Loading ETH/USDT 15m data...")
    df = pd.read_csv('eth_15m_merged.csv', parse_dates=['Open time'])
    df = df.rename(columns={'Open time': 'timestamp'})
    df = df.set_index('timestamp')
    df = df.sort_index()
    print(f"Loaded {len(df)} bars: {df.index.min()} → {df.index.max()}")

    # Run backtest
    df_results, edge_combos = run_backtest(df)

    # Session chain analysis
    run_session_chain(df_results)

    # Save results
    df_results.to_csv('backtest_us_claims_results.csv', index=False)
    print(f"\n✅ Results saved to backtest_us_claims_results.csv")

    # Save edges as JSON for module integration
    with open('backtest_us_claims_edges.json', 'w') as f:
        json.dump(edge_combos, f, indent=2)
    print(f"✅ Edge combos saved to backtest_us_claims_edges.json")

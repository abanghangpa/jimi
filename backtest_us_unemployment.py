#!/usr/bin/env python3
"""
M62: US Unemployment Rate Backtest & Edge Discovery
====================================================
ETH/USDT 15m data from 2018 to 2026.

Backtests US Unemployment Rate releases (same report as NFP — first Friday)
against ETH price action. Classifies each release by:
  - Level: GOLDILOCKS (<3.5%), NORMAL (3.5-4.0%), SOFTENING (4.0-4.5%), DANGER (>4.5%), CRISIS (>6.0%)
  - Change: RISING (≥0.2pp up), FALLING (≥0.2pp down), STABLE
  - Surprise: BEAT (lower than consensus), MISS (higher than consensus)
  - Sahm Rule: triggered or not (3m avg rise ≥ 0.5pp from 12m low)

Cross-tabulates: (wyckoff × vol × signal) → avg 24h return, win rate, sample size
Session transmission chain analysis
Statistical significance testing
Saves edges to backtest_us_unemployment_edges.json
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from scipy import stats
import json
import warnings
warnings.filterwarnings('ignore')


# ═══════════════════════════════════════════════════════════════
# NFP RELEASE DATES (first Friday of month, 08:30 ET)
# Unemployment rate is released simultaneously with NFP
# ═══════════════════════════════════════════════════════════════

NFP_RELEASE_DATES = {
    '2018-01-05', '2018-02-02', '2018-03-09', '2018-04-06',
    '2018-05-04', '2018-06-01', '2018-07-06', '2018-08-03',
    '2018-09-07', '2018-10-05', '2018-11-02', '2018-12-07',
    '2019-01-04', '2019-02-01', '2019-03-08', '2019-04-05',
    '2019-05-03', '2019-06-07', '2019-07-05', '2019-08-02',
    '2019-09-06', '2019-10-04', '2019-11-01', '2019-12-06',
    '2020-01-10', '2020-02-07', '2020-03-06', '2020-04-03',
    '2020-05-08', '2020-06-05', '2020-07-02', '2020-08-07',
    '2020-09-04', '2020-10-02', '2020-11-06', '2020-12-04',
    '2021-01-08', '2021-02-05', '2021-03-05', '2021-04-02',
    '2021-05-07', '2021-06-04', '2021-07-02', '2021-08-06',
    '2021-09-03', '2021-10-08', '2021-11-05', '2021-12-03',
    '2022-01-07', '2022-02-04', '2022-03-04', '2022-04-01',
    '2022-05-06', '2022-06-03', '2022-07-08', '2022-08-05',
    '2022-09-02', '2022-10-07', '2022-11-04', '2022-12-02',
    '2023-01-06', '2023-02-03', '2023-03-10', '2023-04-07',
    '2023-05-05', '2023-06-02', '2023-07-07', '2023-08-04',
    '2023-09-01', '2023-10-06', '2023-11-03', '2023-12-08',
    '2024-01-05', '2024-02-02', '2024-03-08', '2024-04-05',
    '2024-05-03', '2024-06-07', '2024-07-05', '2024-08-02',
    '2024-09-06', '2024-10-04', '2024-11-01', '2024-12-06',
    '2025-01-10', '2025-02-07', '2025-03-07', '2025-04-04',
    '2025-05-02', '2025-06-06', '2025-07-03', '2025-08-01',
    '2025-09-05', '2025-10-03', '2025-11-07', '2025-12-05',
    '2026-01-09', '2026-02-06', '2026-03-06', '2026-04-03',
    '2026-05-01',
}


# ═══════════════════════════════════════════════════════════════
# UNEMPLOYMENT RATE DATA (monthly, from BLS/FRED)
# Format: YYYY-MM → unemployment rate %
# ═══════════════════════════════════════════════════════════════

UNEMPLOYMENT_RATE_MONTHLY = {
    # 2017 (for prior-year lookback)
    '2017-01': 4.7, '2017-02': 4.6, '2017-03': 4.4, '2017-04': 4.4,
    '2017-05': 4.4, '2017-06': 4.3, '2017-07': 4.3, '2017-08': 4.3,
    '2017-09': 4.1, '2017-10': 4.1, '2017-11': 4.1, '2017-12': 4.1,
    # 2018
    '2018-01': 4.1, '2018-02': 4.1, '2018-03': 4.0, '2018-04': 3.9,
    '2018-05': 3.8, '2018-06': 4.0, '2018-07': 3.9, '2018-08': 3.8,
    '2018-09': 3.7, '2018-10': 3.8, '2018-11': 3.7, '2018-12': 3.9,
    # 2019
    '2019-01': 4.0, '2019-02': 3.8, '2019-03': 3.8, '2019-04': 3.6,
    '2019-05': 3.6, '2019-06': 3.6, '2019-07': 3.7, '2019-08': 3.7,
    '2019-09': 3.5, '2019-10': 3.6, '2019-11': 3.5, '2019-12': 3.6,
    # 2020
    '2020-01': 3.6, '2020-02': 3.5, '2020-03': 4.4, '2020-04': 14.7,
    '2020-05': 13.2, '2020-06': 11.0, '2020-07': 10.2, '2020-08': 8.4,
    '2020-09': 7.8, '2020-10': 6.9, '2020-11': 6.7, '2020-12': 6.7,
    # 2021
    '2021-01': 6.3, '2021-02': 6.2, '2021-03': 6.0, '2021-04': 6.1,
    '2021-05': 5.8, '2021-06': 5.9, '2021-07': 5.4, '2021-08': 5.2,
    '2021-09': 4.7, '2021-10': 4.6, '2021-11': 4.2, '2021-12': 3.9,
    # 2022
    '2022-01': 4.0, '2022-02': 3.8, '2022-03': 3.6, '2022-04': 3.6,
    '2022-05': 3.6, '2022-06': 3.6, '2022-07': 3.5, '2022-08': 3.7,
    '2022-09': 3.5, '2022-10': 3.7, '2022-11': 3.7, '2022-12': 3.5,
    # 2023
    '2023-01': 3.4, '2023-02': 3.6, '2023-03': 3.5, '2023-04': 3.4,
    '2023-05': 3.7, '2023-06': 3.6, '2023-07': 3.5, '2023-08': 3.8,
    '2023-09': 3.8, '2023-10': 3.9, '2023-11': 3.7, '2023-12': 3.7,
    # 2024
    '2024-01': 3.7, '2024-02': 3.9, '2024-03': 3.8, '2024-04': 3.9,
    '2024-05': 4.0, '2024-06': 4.1, '2024-07': 4.3, '2024-08': 4.2,
    '2024-09': 4.1, '2024-10': 4.1, '2024-11': 4.2, '2024-12': 4.1,
    # 2025
    '2025-01': 4.0, '2025-02': 4.1, '2025-03': 4.2, '2025-04': 4.2,
    '2025-05': 4.1, '2025-06': 4.2, '2025-07': 4.2, '2025-08': 4.3,
    '2025-09': 4.3, '2025-10': 4.3, '2025-11': 4.4, '2025-12': 4.4,
    # 2026
    '2026-01': 4.3, '2026-02': 4.3, '2026-03': 4.3, '2026-04': 4.3,
}

# Consensus expectations (approximate — from Bloomberg/Reuters at time of release)
# Format: YYYY-MM → expected unemployment rate %
UNEMPLOYMENT_CONSENSUS = {
    '2018-01': 4.1, '2018-02': 4.1, '2018-03': 4.0, '2018-04': 4.0,
    '2018-05': 3.9, '2018-06': 3.9, '2018-07': 3.9, '2018-08': 3.8,
    '2018-09': 3.8, '2018-10': 3.7, '2018-11': 3.7, '2018-12': 3.7,
    '2019-01': 3.9, '2019-02': 3.9, '2019-03': 3.8, '2019-04': 3.8,
    '2019-05': 3.6, '2019-06': 3.6, '2019-07': 3.6, '2019-08': 3.7,
    '2019-09': 3.7, '2019-10': 3.6, '2019-11': 3.6, '2019-12': 3.5,
    '2020-01': 3.5, '2020-02': 3.6, '2020-03': 3.7, '2020-04': 14.0,
    '2020-05': 19.5, '2020-06': 12.3, '2020-07': 10.5, '2020-08': 9.8,
    '2020-09': 8.2, '2020-10': 7.7, '2020-11': 6.8, '2020-12': 6.7,
    '2021-01': 6.7, '2021-02': 6.3, '2021-03': 6.0, '2021-04': 5.8,
    '2021-05': 5.9, '2021-06': 5.7, '2021-07': 5.7, '2021-08': 5.2,
    '2021-09': 5.1, '2021-10': 4.7, '2021-11': 4.5, '2021-12': 4.1,
    '2022-01': 3.9, '2022-02': 3.9, '2022-03': 3.7, '2022-04': 3.6,
    '2022-05': 3.5, '2022-06': 3.6, '2022-07': 3.6, '2022-08': 3.5,
    '2022-09': 3.7, '2022-10': 3.6, '2022-11': 3.7, '2022-12': 3.7,
    '2023-01': 3.6, '2023-02': 3.4, '2023-03': 3.6, '2023-04': 3.6,
    '2023-05': 3.5, '2023-06': 3.7, '2023-07': 3.6, '2023-08': 3.5,
    '2023-09': 3.7, '2023-10': 3.8, '2023-11': 3.9, '2023-12': 3.8,
    '2024-01': 3.8, '2024-02': 3.7, '2024-03': 3.9, '2024-04': 3.8,
    '2024-05': 3.9, '2024-06': 4.0, '2024-07': 4.1, '2024-08': 4.2,
    '2024-09': 4.2, '2024-10': 4.2, '2024-11': 4.1, '2024-12': 4.2,
    '2025-01': 4.1, '2025-02': 4.0, '2025-03': 4.1, '2025-04': 4.2,
    '2025-05': 4.2, '2025-06': 4.1, '2025-07': 4.2, '2025-08': 4.2,
    '2025-09': 4.3, '2025-10': 4.3, '2025-11': 4.3, '2025-12': 4.4,
    '2026-01': 4.4, '2026-02': 4.3, '2026-03': 4.3, '2026-04': 4.3,
    '2026-05': 4.3,
}


# ═══════════════════════════════════════════════════════════════
# SESSION WINDOWS (UTC) — same as backtest_nfp.py
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


# ═══════════════════════════════════════════════════════════════
# CLASSIFICATION FUNCTIONS
# ═══════════════════════════════════════════════════════════════

def classify_unemp_level(rate):
    """Classify unemployment rate level."""
    if rate < 3.5:
        return 'GOLDILOCKS'
    elif rate < 4.0:
        return 'NORMAL'
    elif rate < 4.5:
        return 'SOFTENING'
    elif rate < 6.0:
        return 'DANGER'
    else:
        return 'CRISIS'


def classify_unemp_change(current, prior):
    """Classify change from prior month."""
    delta = current - prior
    if delta >= 0.2:
        return 'RISING', delta
    elif delta <= -0.2:
        return 'FALLING', delta
    else:
        return 'STABLE', delta


def classify_unemp_surprise(actual, consensus):
    """Classify surprise vs consensus. Lower unemployment = BEAT."""
    diff = actual - consensus
    if diff < -0.1:
        return 'BEAT', diff
    elif diff > 0.1:
        return 'MISS', diff
    else:
        return 'INLINE', diff


def check_sahm_rule(unemp_dict, release_month_key):
    """Check if Sahm Rule is triggered at this release.

    Sahm Rule: 3-month avg unemployment - 12-month low >= 0.5pp
    """
    sorted_keys = sorted(k for k in unemp_dict.keys() if k <= release_month_key)
    if len(sorted_keys) < 12:
        return False, 0.0

    # 3-month average (current + 2 prior)
    last_3 = [unemp_dict[k] for k in sorted_keys[-3:]]
    avg_3m = sum(last_3) / 3

    # 12-month low
    last_12 = [unemp_dict[k] for k in sorted_keys[-12:]]
    low_12m = min(last_12)

    sahm_value = avg_3m - low_12m
    return sahm_value >= 0.5, round(sahm_value, 3)


def classify_wyckoff_proxy(df, release_idx, lookback=48):
    """M21 proxy: classify Wyckoff phase from price action before release."""
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
    """M9 proxy: classify volatility regime."""
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


def get_session_returns(df, release_date, release_utc_hour):
    """Calculate returns for each session window on release day and next day."""
    results = {}

    release_dt = pd.Timestamp(
        f"{release_date} {int(release_utc_hour):02d}:"
        f"{int((release_utc_hour % 1) * 60):02d}:00"
    )

    release_mask = df.index >= release_dt
    if not release_mask.any():
        return results
    release_price_row = df[release_mask].iloc[0]
    release_price = release_price_row['Close']
    release_idx = df.index.get_loc(df[release_mask].index[0])

    for region, phase, start_h, end_h in SESSION_WINDOWS:
        sess_start = release_dt.replace(
            hour=int(start_h), minute=int((start_h % 1) * 60), second=0
        )
        sess_end = release_dt.replace(
            hour=int(end_h), minute=int((end_h % 1) * 60), second=0
        )

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


# ═══════════════════════════════════════════════════════════════
# MAIN BACKTEST
# ═══════════════════════════════════════════════════════════════

def run_backtest(df):
    """Full unemployment rate backtest with cross-tabulation."""
    print("=" * 80)
    print("M62: US UNEMPLOYMENT RATE BACKTEST — ETH/USDT 15m (2018-2026)")
    print("=" * 80)

    all_results = []

    for date_str in sorted(NFP_RELEASE_DATES):
        # Get the month key for unemployment data
        dt = datetime.strptime(date_str, '%Y-%m-%d')
        month_key = dt.strftime('%Y-%m')

        # NFP report covers PRIOR month's unemployment
        # e.g., Jan 5 release = Dec unemployment data
        # But the month_key in our dict IS the data month
        # We need to find the correct month: release on first Friday = prior month's data
        data_month_key = (dt - timedelta(days=10)).strftime('%Y-%m')

        if data_month_key not in UNEMPLOYMENT_RATE_MONTHLY:
            # Try using the release month itself (some data is reported same month)
            if month_key in UNEMPLOYMENT_RATE_MONTHLY:
                data_month_key = month_key
            else:
                continue

        actual_rate = UNEMPLOYMENT_RATE_MONTHLY[data_month_key]
        consensus = UNEMPLOYMENT_CONSENSUS.get(data_month_key, actual_rate)

        # Prior month
        prior_dt = dt - timedelta(days=28)
        prior_month_key = prior_dt.strftime('%Y-%m')
        prior_rate = UNEMPLOYMENT_RATE_MONTHLY.get(prior_month_key)

        if prior_rate is None:
            continue

        # Classify
        level = classify_unemp_level(actual_rate)
        change, delta = classify_unemp_change(actual_rate, prior_rate)
        surprise, surprise_diff = classify_unemp_surprise(actual_rate, consensus)
        sahm_triggered, sahm_value = check_sahm_rule(UNEMPLOYMENT_RATE_MONTHLY, data_month_key)

        # Determine UTC hour (EDT vs EST)
        month = dt.month
        if 3 <= month <= 10:
            utc_hour = 12.5  # 08:30 EDT = 12:30 UTC
        else:
            utc_hour = 13.5  # 08:30 EST = 13:30 UTC

        # Find release index in dataframe
        release_dt = pd.Timestamp(
            f"{date_str} {int(utc_hour):02d}:"
            f"{int((utc_hour % 1) * 60):02d}:00"
        )
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

        # Build composite signal
        # Priority: Sahm > Level > Change > Surprise
        if sahm_triggered:
            signal = 'SAHM_TRIGGERED'
        elif level in ('CRISIS', 'DANGER'):
            signal = f'LEVEL_{level}'
        elif change in ('RISING', 'FALLING'):
            signal = f'CHANGE_{change}'
        elif surprise in ('BEAT', 'MISS'):
            signal = f'SURPRISE_{surprise}'
        else:
            signal = 'NEUTRAL'

        record = {
            'date': date_str,
            'data_month': data_month_key,
            'actual_rate': actual_rate,
            'consensus': consensus,
            'prior_rate': prior_rate,
            'delta': delta,
            'level': level,
            'change': change,
            'surprise': surprise,
            'surprise_diff': surprise_diff,
            'sahm_triggered': sahm_triggered,
            'sahm_value': sahm_value,
            'signal': signal,
            'wyckoff': wyckoff,
            'vol_regime': vol_regime,
            'ret_24h': agg['return_pct'],
            'direction_24h': agg['direction'],
            'high_ext': agg['high_ext'],
            'low_ext': agg['low_ext'],
        }

        for sess_name, sess_data in session_rets.items():
            if sess_name != '24h_AGGREGATE':
                record[f'ret_{sess_name}'] = sess_data['return_pct']
                record[f'dir_{sess_name}'] = sess_data['direction']

        all_results.append(record)

    df_results = pd.DataFrame(all_results)
    print(f"\nTotal NFP/Unemployment releases analyzed: {len(df_results)}")
    print(f"Date range: {df_results['date'].min()} → {df_results['date'].max()}")

    # ── Overall 24h stats ──
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
    sig = '✅ SIGNIFICANT' if p_val < 0.05 else '❌ NOT SIGNIFICANT'
    print(f"  t-test vs 0: t={t_stat:.3f}, p={p_val:.4f} {sig}")

    # ── Level classification ──
    print("\n" + "─" * 80)
    print("UNEMPLOYMENT LEVEL CLASSIFICATION")
    print("─" * 80)
    for lvl in ['GOLDILOCKS', 'NORMAL', 'SOFTENING', 'DANGER', 'CRISIS']:
        subset = df_results[df_results['level'] == lvl]
        if len(subset) == 0:
            continue
        mean_r = subset['ret_24h'].mean()
        win_r = (subset['ret_24h'] > 0).mean()
        print(f"  {lvl:12s}: n={len(subset):3d}  avg={mean_r:+.3f}%  win={win_r*100:.1f}%")

    # ── Change classification ──
    print("\n" + "─" * 80)
    print("UNEMPLOYMENT CHANGE CLASSIFICATION")
    print("─" * 80)
    for chg in ['RISING', 'STABLE', 'FALLING']:
        subset = df_results[df_results['change'] == chg]
        if len(subset) == 0:
            continue
        mean_r = subset['ret_24h'].mean()
        win_r = (subset['ret_24h'] > 0).mean()
        print(f"  {chg:8s}: n={len(subset):3d}  avg={mean_r:+.3f}%  win={win_r*100:.1f}%")

    # ── Surprise classification ──
    print("\n" + "─" * 80)
    print("UNEMPLOYMENT SURPRISE CLASSIFICATION")
    print("─" * 80)
    for surp in ['BEAT', 'INLINE', 'MISS']:
        subset = df_results[df_results['surprise'] == surp]
        if len(subset) == 0:
            continue
        mean_r = subset['ret_24h'].mean()
        win_r = (subset['ret_24h'] > 0).mean()
        print(f"  {surp:6s}: n={len(subset):3d}  avg={mean_r:+.3f}%  win={win_r*100:.1f}%")

    # ── Sahm Rule ──
    print("\n" + "─" * 80)
    print("SAHM RULE IMPACT")
    print("─" * 80)
    sahm_true = df_results[df_results['sahm_triggered'] == True]
    sahm_false = df_results[df_results['sahm_triggered'] == False]
    if len(sahm_true) > 0:
        print(f"  Sahm TRIGGERED: n={len(sahm_true):3d}  avg={sahm_true['ret_24h'].mean():+.3f}%  "
              f"win={(sahm_true['ret_24h'] > 0).mean()*100:.1f}%")
    if len(sahm_false) > 0:
        print(f"  Sahm NOT triggered: n={len(sahm_false):3d}  avg={sahm_false['ret_24h'].mean():+.3f}%  "
              f"win={(sahm_false['ret_24h'] > 0).mean()*100:.1f}%")

    # ── Cross-tabulation: Wyckoff × Vol × Signal ──
    print("\n" + "─" * 80)
    print("CROSS-TABULATION: Wyckoff × Vol Regime × Signal → 24h Return")
    print("─" * 80)
    print(f"  {'Wyckoff':10s} {'Vol':12s} {'Signal':16s} {'n':>4s} {'Avg 24h%':>10s} {'Win%':>8s} {'Edge?':>8s}")
    print(f"  {'─'*10} {'─'*12} {'─'*16} {'─'*4} {'─'*10} {'─'*8} {'─'*8}")

    edge_combos = []
    for wyk in sorted(df_results['wyckoff'].unique()):
        for vol in sorted(df_results['vol_regime'].unique()):
            for sig in sorted(df_results['signal'].unique()):
                mask = (
                    (df_results['wyckoff'] == wyk) &
                    (df_results['vol_regime'] == vol) &
                    (df_results['signal'] == sig)
                )
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
                        'n': n, 'avg_ret': round(mean_r, 3),
                        'win_rate': round(win_r, 3),
                        'bias': 'LONG' if mean_r > 0 else 'SHORT'
                    })
                elif n >= 3 and abs(mean_r) >= 0.3:
                    edge = '🟡 MARGINAL'
                print(f"  {wyk:10s} {vol:12s} {sig:16s} {n:4d} {mean_r:+10.3f} {win_r*100:7.1f}% {edge}")

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

    # ── Broad edges (Wyckoff + Signal, ignoring vol) ──
    print("\n" + "─" * 80)
    print("BROAD EDGES: Wyckoff × Signal (ignoring vol)")
    print("─" * 80)
    broad_edges = []
    for wyk in sorted(df_results['wyckoff'].unique()):
        for sig in sorted(df_results['signal'].unique()):
            mask = (df_results['wyckoff'] == wyk) & (df_results['signal'] == sig)
            subset = df_results[mask]
            if len(subset) < 3:
                continue
            mean_r = subset['ret_24h'].mean()
            win_r = (subset['ret_24h'] > 0).mean()
            n = len(subset)
            if abs(mean_r) >= 0.3:
                icon = '🟢' if mean_r > 0 else '🔴'
                print(f"  {icon} {wyk:10s} + {sig:16s}: n={n:3d}  avg={mean_r:+.3f}%  win={win_r*100:.1f}%")
                broad_edges.append({
                    'wyckoff': wyk, 'signal': sig,
                    'n': n, 'avg_ret': round(mean_r, 3),
                    'win_rate': round(win_r, 3),
                    'bias': 'LONG' if mean_r > 0 else 'SHORT'
                })

    # ── Signal-only edges (ignoring wyckoff and vol) ──
    print("\n" + "─" * 80)
    print("SIGNAL-ONLY EDGES")
    print("─" * 80)
    signal_edges = []
    for sig in sorted(df_results['signal'].unique()):
        subset = df_results[df_results['signal'] == sig]
        if len(subset) < 3:
            continue
        mean_r = subset['ret_24h'].mean()
        win_r = (subset['ret_24h'] > 0).mean()
        n = len(subset)
        icon = '🟢' if mean_r > 0 else '🔴' if mean_r < 0 else '⚪'
        sig_str = '✅' if abs(mean_r) >= 0.3 and n >= 5 else ''
        print(f"  {icon} {sig:16s}: n={n:3d}  avg={mean_r:+.3f}%  win={win_r*100:.1f}%  {sig_str}")
        if abs(mean_r) >= 0.3 and n >= 5:
            signal_edges.append({
                'signal': sig, 'n': n,
                'avg_ret': round(mean_r, 3),
                'win_rate': round(win_r, 3),
                'bias': 'LONG' if mean_r > 0 else 'SHORT'
            })

    # ── Statistical significance: MISS vs BEAT ──
    print("\n" + "─" * 80)
    print("STATISTICAL SIGNIFICANCE: MISS vs BEAT")
    print("─" * 80)
    miss_rets = df_results[df_results['surprise'] == 'MISS']['ret_24h'].dropna()
    beat_rets = df_results[df_results['surprise'] == 'BEAT']['ret_24h'].dropna()
    if len(miss_rets) >= 3 and len(beat_rets) >= 3:
        t_stat2, p_val2 = stats.ttest_ind(miss_rets, beat_rets)
        print(f"  MISS: n={len(miss_rets)}, avg={miss_rets.mean():+.3f}%")
        print(f"  BEAT: n={len(beat_rets)}, avg={beat_rets.mean():+.3f}%")
        sig2 = '✅ SIGNIFICANT' if p_val2 < 0.05 else '❌ NOT SIGNIFICANT'
        print(f"  t-test: t={t_stat2:.3f}, p={p_val2:.4f} {sig2}")

    # ── RISING vs FALLING ──
    print("\n" + "─" * 80)
    print("STATISTICAL SIGNIFICANCE: RISING vs FALLING")
    print("─" * 80)
    rising_rets = df_results[df_results['change'] == 'RISING']['ret_24h'].dropna()
    falling_rets = df_results[df_results['change'] == 'FALLING']['ret_24h'].dropna()
    if len(rising_rets) >= 3 and len(falling_rets) >= 3:
        t_stat3, p_val3 = stats.ttest_ind(rising_rets, falling_rets)
        print(f"  RISING:  n={len(rising_rets)}, avg={rising_rets.mean():+.3f}%")
        print(f"  FALLING: n={len(falling_rets)}, avg={falling_rets.mean():+.3f}%")
        sig3 = '✅ SIGNIFICANT' if p_val3 < 0.05 else '❌ NOT SIGNIFICANT'
        print(f"  t-test: t={t_stat3:.3f}, p={p_val3:.4f} {sig3}")

    return df_results, edge_combos, broad_edges, signal_edges


# ═══════════════════════════════════════════════════════════════
# SESSION TRANSMISSION CHAIN
# ═══════════════════════════════════════════════════════════════

def run_chain_analysis(df_results):
    """Session transmission chain validation."""
    print("\n\n" + "=" * 80)
    print("SESSION TRANSMISSION CHAIN — US UNEMPLOYMENT RATE")
    print("=" * 80)

    session_order = []
    for region, phase, _, _ in SESSION_WINDOWS:
        col = f'ret_{region} | {phase}'
        dcol = f'dir_{region} | {phase}'
        if col in df_results.columns:
            session_order.append((region, phase, col, dcol))

    print(f"\nSessions found: {len(session_order)}")

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
        chain_results.append({
            'from': p1, 'to': p2, 'n': n,
            'same_dir_pct': pct, 'status': status
        })

    return chain_results


# ═══════════════════════════════════════════════════════════════
# BUILD FINAL EDGES JSON
# ═══════════════════════════════════════════════════════════════

def build_edges_json(df_results, edge_combos, broad_edges, signal_edges):
    """Build the final edges JSON for module integration."""
    edges = {
        'fine': [],       # Wyckoff × Vol × Signal
        'broad': [],      # Wyckoff × Signal
        'signal_only': [],  # Signal only
    }

    for ec in edge_combos:
        edges['fine'].append({
            'wyckoff': ec['wyckoff'],
            'vol': ec['vol'],
            'signal': ec['signal'],
            'n': ec['n'],
            'avg_ret': ec['avg_ret'],
            'win_rate': ec['win_rate'],
            'bias': ec['bias'],
        })

    for be in broad_edges:
        edges['broad'].append({
            'wyckoff': be['wyckoff'],
            'signal': be['signal'],
            'n': be['n'],
            'avg_ret': be['avg_ret'],
            'win_rate': be['win_rate'],
            'bias': be['bias'],
        })

    for se in signal_edges:
        edges['signal_only'].append({
            'signal': se['signal'],
            'n': se['n'],
            'avg_ret': se['avg_ret'],
            'win_rate': se['win_rate'],
            'bias': se['bias'],
        })

    # Add metadata
    edges['metadata'] = {
        'backtest_date': datetime.utcnow().strftime('%Y-%m-%d %H:%M'),
        'total_releases': len(df_results),
        'date_range': f"{df_results['date'].min()} → {df_results['date'].max()}",
        'data_source': 'eth_15m_merged.csv',
        'criteria': 'n>=3, |avg_ret|>=0.5% (fine), >=0.3% (broad/signal)',
    }

    return edges


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
    df_results, edge_combos, broad_edges, signal_edges = run_backtest(df)

    # Session chain analysis
    chain_results = run_chain_analysis(df_results)

    # Save results
    df_results.to_csv('backtest_us_unemployment_results.csv', index=False)
    print(f"\n✅ Results saved to backtest_us_unemployment_results.csv")

    # Save edges JSON
    edges = build_edges_json(df_results, edge_combos, broad_edges, signal_edges)
    with open('backtest_us_unemployment_edges.json', 'w') as f:
        json.dump(edges, f, indent=2)
    print(f"✅ Edge combos saved to backtest_us_unemployment_edges.json")

    # Summary
    print("\n" + "=" * 80)
    print("BACKTEST COMPLETE — SUMMARY")
    print("=" * 80)
    print(f"  Total releases: {len(df_results)}")
    print(f"  Fine edges found: {len(edge_combos)}")
    print(f"  Broad edges found: {len(broad_edges)}")
    print(f"  Signal-only edges: {len(signal_edges)}")
    print(f"  Chain transitions analyzed: {len(chain_results)}")

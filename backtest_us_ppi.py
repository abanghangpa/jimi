#!/usr/bin/env python3
"""
US PPI (Producer Price Index) Backtest — JIMI Framework
========================================================
Backtests US PPI Final Demand YoY releases against ETH/USDT 15m data (2018-2026).

PPI is released ~monthly by BLS at 12:30 UTC (08:30 ET), usually 1-2 days after CPI.
PPI confirms or denies the CPI signal — pipeline inflation matters for Fed policy.

Session windows match the NFP/CPI backtest pattern.
Cross-tabulates: Wyckoff phase × Vol regime × PPI signal → 24h return.
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from scipy import stats
import json
import warnings
warnings.filterwarnings('ignore')

# ═══════════════════════════════════════════════════════════════
# PPI RELEASE DATES + ACTUAL YOY VALUES (Final Demand)
# Released 12:30 UTC (08:30 ET), usually 1-2 days after CPI
# Source: BLS PPI program, FRED WPSFD49207
# ═══════════════════════════════════════════════════════════════

PPI_RELEASES = {
    # ── 2018 ──
    '2018-01-11': {'ppi_yoy': 2.6, 'consensus_yoy': 2.5, 'prior_yoy': 2.6},
    '2018-02-15': {'ppi_yoy': 2.8, 'consensus_yoy': 2.7, 'prior_yoy': 2.6},
    '2018-03-14': {'ppi_yoy': 2.8, 'consensus_yoy': 2.8, 'prior_yoy': 2.8},
    '2018-04-11': {'ppi_yoy': 2.9, 'consensus_yoy': 2.8, 'prior_yoy': 2.8},
    '2018-05-10': {'ppi_yoy': 2.7, 'consensus_yoy': 2.8, 'prior_yoy': 2.9},
    '2018-06-12': {'ppi_yoy': 3.1, 'consensus_yoy': 2.9, 'prior_yoy': 2.7},
    '2018-07-11': {'ppi_yoy': 3.3, 'consensus_yoy': 3.2, 'prior_yoy': 3.1},
    '2018-08-09': {'ppi_yoy': 3.3, 'consensus_yoy': 3.3, 'prior_yoy': 3.3},
    '2018-09-12': {'ppi_yoy': 2.8, 'consensus_yoy': 3.2, 'prior_yoy': 3.3},
    '2018-10-10': {'ppi_yoy': 2.6, 'consensus_yoy': 2.6, 'prior_yoy': 2.8},
    '2018-11-14': {'ppi_yoy': 2.5, 'consensus_yoy': 2.5, 'prior_yoy': 2.6},
    '2018-12-11': {'ppi_yoy': 2.5, 'consensus_yoy': 2.5, 'prior_yoy': 2.5},
    # ── 2019 ──
    '2019-01-15': {'ppi_yoy': 2.0, 'consensus_yoy': 2.2, 'prior_yoy': 2.5},
    '2019-02-14': {'ppi_yoy': 1.7, 'consensus_yoy': 1.8, 'prior_yoy': 2.0},
    '2019-03-14': {'ppi_yoy': 1.9, 'consensus_yoy': 1.8, 'prior_yoy': 1.7},
    '2019-04-11': {'ppi_yoy': 2.2, 'consensus_yoy': 2.0, 'prior_yoy': 1.9},
    '2019-05-09': {'ppi_yoy': 2.2, 'consensus_yoy': 2.3, 'prior_yoy': 2.2},
    '2019-06-11': {'ppi_yoy': 1.8, 'consensus_yoy': 2.0, 'prior_yoy': 2.2},
    '2019-07-12': {'ppi_yoy': 1.7, 'consensus_yoy': 1.7, 'prior_yoy': 1.8},
    '2019-08-09': {'ppi_yoy': 1.7, 'consensus_yoy': 1.7, 'prior_yoy': 1.7},
    '2019-09-11': {'ppi_yoy': 1.4, 'consensus_yoy': 1.8, 'prior_yoy': 1.7},
    '2019-10-08': {'ppi_yoy': 1.1, 'consensus_yoy': 1.5, 'prior_yoy': 1.4},
    '2019-11-14': {'ppi_yoy': 1.1, 'consensus_yoy': 0.9, 'prior_yoy': 1.1},
    '2019-12-12': {'ppi_yoy': 1.1, 'consensus_yoy': 1.3, 'prior_yoy': 1.1},
    # ── 2020 ──
    '2020-01-14': {'ppi_yoy': 1.7, 'consensus_yoy': 1.3, 'prior_yoy': 1.1},
    '2020-02-19': {'ppi_yoy': 2.0, 'consensus_yoy': 1.6, 'prior_yoy': 1.7},
    '2020-03-12': {'ppi_yoy': 1.3, 'consensus_yoy': 1.2, 'prior_yoy': 2.0},
    '2020-04-09': {'ppi_yoy': -0.6, 'consensus_yoy': 0.2, 'prior_yoy': 1.3},
    '2020-05-12': {'ppi_yoy': -1.2, 'consensus_yoy': -1.2, 'prior_yoy': -0.6},
    '2020-06-10': {'ppi_yoy': -0.8, 'consensus_yoy': -1.2, 'prior_yoy': -1.2},
    '2020-07-14': {'ppi_yoy': -0.4, 'consensus_yoy': -0.7, 'prior_yoy': -0.8},
    '2020-08-11': {'ppi_yoy': -0.2, 'consensus_yoy': -0.3, 'prior_yoy': -0.4},
    '2020-09-11': {'ppi_yoy': 0.4, 'consensus_yoy': 0.2, 'prior_yoy': -0.2},
    '2020-10-14': {'ppi_yoy': 0.5, 'consensus_yoy': 0.4, 'prior_yoy': 0.4},
    '2020-11-13': {'ppi_yoy': 0.8, 'consensus_yoy': 0.5, 'prior_yoy': 0.5},
    '2020-12-11': {'ppi_yoy': 0.8, 'consensus_yoy': 0.7, 'prior_yoy': 0.8},
    # ── 2021 ──
    '2021-01-13': {'ppi_yoy': 1.3, 'consensus_yoy': 0.9, 'prior_yoy': 0.8},
    '2021-02-17': {'ppi_yoy': 1.7, 'consensus_yoy': 1.0, 'prior_yoy': 1.3},
    '2021-03-12': {'ppi_yoy': 2.8, 'consensus_yoy': 2.7, 'prior_yoy': 1.7},
    '2021-04-09': {'ppi_yoy': 4.2, 'consensus_yoy': 3.8, 'prior_yoy': 2.8},
    '2021-05-12': {'ppi_yoy': 6.6, 'consensus_yoy': 5.9, 'prior_yoy': 4.2},
    '2021-06-11': {'ppi_yoy': 7.3, 'consensus_yoy': 6.3, 'prior_yoy': 6.6},
    '2021-07-13': {'ppi_yoy': 7.8, 'consensus_yoy': 7.3, 'prior_yoy': 7.3},
    '2021-08-12': {'ppi_yoy': 8.6, 'consensus_yoy': 8.2, 'prior_yoy': 7.8},
    '2021-09-10': {'ppi_yoy': 8.7, 'consensus_yoy': 8.3, 'prior_yoy': 8.6},
    '2021-10-14': {'ppi_yoy': 8.8, 'consensus_yoy': 8.7, 'prior_yoy': 8.7},
    '2021-11-09': {'ppi_yoy': 8.8, 'consensus_yoy': 8.7, 'prior_yoy': 8.8},
    '2021-12-14': {'ppi_yoy': 9.8, 'consensus_yoy': 9.2, 'prior_yoy': 8.8},
    # ── 2022 ──
    '2022-01-13': {'ppi_yoy': 9.7, 'consensus_yoy': 9.8, 'prior_yoy': 9.8},
    '2022-02-15': {'ppi_yoy': 10.0, 'consensus_yoy': 9.1, 'prior_yoy': 9.7},
    '2022-03-11': {'ppi_yoy': 10.0, 'consensus_yoy': 10.0, 'prior_yoy': 10.0},
    '2022-04-12': {'ppi_yoy': 11.2, 'consensus_yoy': 10.6, 'prior_yoy': 10.0},
    '2022-05-12': {'ppi_yoy': 11.0, 'consensus_yoy': 10.7, 'prior_yoy': 11.2},
    '2022-06-14': {'ppi_yoy': 10.9, 'consensus_yoy': 10.9, 'prior_yoy': 11.0},
    '2022-07-14': {'ppi_yoy': 9.8, 'consensus_yoy': 10.4, 'prior_yoy': 10.9},
    '2022-08-11': {'ppi_yoy': 8.7, 'consensus_yoy': 8.8, 'prior_yoy': 9.8},
    '2022-09-14': {'ppi_yoy': 8.5, 'consensus_yoy': 8.8, 'prior_yoy': 8.7},
    '2022-10-12': {'ppi_yoy': 8.0, 'consensus_yoy': 8.4, 'prior_yoy': 8.5},
    '2022-11-15': {'ppi_yoy': 7.4, 'consensus_yoy': 8.0, 'prior_yoy': 8.0},
    '2022-12-09': {'ppi_yoy': 7.4, 'consensus_yoy': 7.2, 'prior_yoy': 7.4},
    # ── 2023 ──
    '2023-01-18': {'ppi_yoy': 6.2, 'consensus_yoy': 6.8, 'prior_yoy': 7.4},
    '2023-02-16': {'ppi_yoy': 6.0, 'consensus_yoy': 5.4, 'prior_yoy': 6.2},
    '2023-03-15': {'ppi_yoy': 4.6, 'consensus_yoy': 5.4, 'prior_yoy': 6.0},
    '2023-04-13': {'ppi_yoy': 2.7, 'consensus_yoy': 3.0, 'prior_yoy': 4.6},
    '2023-05-11': {'ppi_yoy': 2.3, 'consensus_yoy': 2.4, 'prior_yoy': 2.7},
    '2023-06-14': {'ppi_yoy': 1.1, 'consensus_yoy': 1.5, 'prior_yoy': 2.3},
    '2023-07-13': {'ppi_yoy': 0.2, 'consensus_yoy': 0.4, 'prior_yoy': 1.1},
    '2023-08-11': {'ppi_yoy': 0.8, 'consensus_yoy': 0.7, 'prior_yoy': 0.2},
    '2023-09-14': {'ppi_yoy': 1.6, 'consensus_yoy': 1.3, 'prior_yoy': 0.8},
    '2023-10-12': {'ppi_yoy': 2.2, 'consensus_yoy': 1.6, 'prior_yoy': 1.6},
    '2023-11-15': {'ppi_yoy': 1.3, 'consensus_yoy': 1.9, 'prior_yoy': 2.2},
    '2023-12-13': {'ppi_yoy': 1.0, 'consensus_yoy': 1.0, 'prior_yoy': 1.3},
    # ── 2024 ──
    '2024-01-12': {'ppi_yoy': 1.0, 'consensus_yoy': 1.3, 'prior_yoy': 1.0},
    '2024-02-16': {'ppi_yoy': 0.9, 'consensus_yoy': 0.6, 'prior_yoy': 1.0},
    '2024-03-14': {'ppi_yoy': 1.6, 'consensus_yoy': 1.2, 'prior_yoy': 0.9},
    '2024-04-11': {'ppi_yoy': 2.1, 'consensus_yoy': 2.2, 'prior_yoy': 1.6},
    '2024-05-14': {'ppi_yoy': 2.2, 'consensus_yoy': 2.3, 'prior_yoy': 2.1},
    '2024-06-13': {'ppi_yoy': 2.2, 'consensus_yoy': 2.3, 'prior_yoy': 2.2},
    '2024-07-12': {'ppi_yoy': 2.7, 'consensus_yoy': 2.3, 'prior_yoy': 2.2},
    '2024-08-13': {'ppi_yoy': 2.2, 'consensus_yoy': 2.3, 'prior_yoy': 2.7},
    '2024-09-12': {'ppi_yoy': 1.8, 'consensus_yoy': 1.8, 'prior_yoy': 2.2},
    '2024-10-11': {'ppi_yoy': 1.8, 'consensus_yoy': 1.6, 'prior_yoy': 1.8},
    '2024-11-14': {'ppi_yoy': 2.4, 'consensus_yoy': 2.3, 'prior_yoy': 1.8},
    '2024-12-12': {'ppi_yoy': 3.0, 'consensus_yoy': 2.6, 'prior_yoy': 2.4},
    # ── 2025 ──
    '2025-01-14': {'ppi_yoy': 3.3, 'consensus_yoy': 3.5, 'prior_yoy': 3.0},
    '2025-02-13': {'ppi_yoy': 3.5, 'consensus_yoy': 3.3, 'prior_yoy': 3.3},
    '2025-03-13': {'ppi_yoy': 3.2, 'consensus_yoy': 3.3, 'prior_yoy': 3.5},
    '2025-04-10': {'ppi_yoy': 2.7, 'consensus_yoy': 3.3, 'prior_yoy': 3.2},
    '2025-05-15': {'ppi_yoy': 2.4, 'consensus_yoy': 2.5, 'prior_yoy': 2.7},
    '2025-06-12': {'ppi_yoy': 2.3, 'consensus_yoy': 2.3, 'prior_yoy': 2.4},
    '2025-07-16': {'ppi_yoy': 2.3, 'consensus_yoy': 2.3, 'prior_yoy': 2.3},
    '2025-08-14': {'ppi_yoy': 2.2, 'consensus_yoy': 2.3, 'prior_yoy': 2.3},
    '2025-09-11': {'ppi_yoy': 2.0, 'consensus_yoy': 2.2, 'prior_yoy': 2.2},
    '2025-10-15': {'ppi_yoy': 1.8, 'consensus_yoy': 2.0, 'prior_yoy': 2.0},
    '2025-11-13': {'ppi_yoy': 1.8, 'consensus_yoy': 1.9, 'prior_yoy': 1.8},
    '2025-12-11': {'ppi_yoy': 2.0, 'consensus_yoy': 1.9, 'prior_yoy': 1.8},
    # ── 2026 ──
    '2026-01-14': {'ppi_yoy': 2.5, 'consensus_yoy': 2.0, 'prior_yoy': 2.0},
    '2026-02-13': {'ppi_yoy': 3.2, 'consensus_yoy': 2.5, 'prior_yoy': 2.5},
    '2026-03-13': {'ppi_yoy': 3.5, 'consensus_yoy': 3.3, 'prior_yoy': 3.2},
    '2026-04-14': {'ppi_yoy': 6.3, 'consensus_yoy': 4.0, 'prior_yoy': 3.5},
}


# ═══════════════════════════════════════════════════════════════
# SESSION WINDOWS (UTC) — same as NFP/CPI backtest
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


def classify_ppi_signal(ppi_yoy, consensus_yoy):
    """Classify PPI release as BEAT/MISS/INLINE based on surprise vs consensus."""
    surprise = ppi_yoy - consensus_yoy
    if surprise > 0.3:
        return 'STRONG_BEAT', surprise
    elif surprise > 0.1:
        return 'BEAT', surprise
    elif surprise < -0.3:
        return 'BIG_MISS', surprise
    elif surprise < -0.1:
        return 'MISS', surprise
    return 'INLINE', surprise


def classify_inflation_level(ppi_yoy):
    """Classify PPI inflation level."""
    if ppi_yoy > 6.0:
        return 'RUNAWAY'
    elif ppi_yoy > 4.0:
        return 'HOT'
    elif ppi_yoy > 2.5:
        return 'WARM'
    elif ppi_yoy >= 1.5:
        return 'TARGET'
    return 'COOL'


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

    release_dt = pd.Timestamp(f"{release_date} {int(release_utc_hour):02d}:{int((release_utc_hour % 1) * 60):02d}:00")

    release_mask = df.index >= release_dt
    if not release_mask.any():
        return results
    release_price_row = df[release_mask].iloc[0]
    release_price = release_price_row['Close']
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


def run_backtest(df):
    """Full PPI backtest with cross-tabulation."""
    print("=" * 80)
    print("US PPI BACKTEST — ETH/USDT 15m (2018-2026)")
    print("=" * 80)

    all_results = []

    for date_str, ppi_data in sorted(PPI_RELEASES.items()):
        ppi_yoy = ppi_data['ppi_yoy']
        consensus_yoy = ppi_data['consensus_yoy']
        prior_yoy = ppi_data['prior_yoy']

        dt = datetime.strptime(date_str, '%Y-%m-%d')
        month = dt.month
        if 3 <= month <= 10:
            utc_hour = 12.5  # 08:30 EDT = 12:30 UTC
        else:
            utc_hour = 13.5  # 08:30 EST = 13:30 UTC

        release_dt = pd.Timestamp(f"{date_str} {int(utc_hour):02d}:{int((utc_hour % 1) * 60):02d}:00")
        mask = df.index >= release_dt
        if not mask.any():
            continue
        release_idx = df.index.get_loc(df[mask].index[0])

        signal, surprise = classify_ppi_signal(ppi_yoy, consensus_yoy)
        infl_level = classify_inflation_level(ppi_yoy)
        wyckoff = classify_wyckoff_proxy(df, release_idx)
        vol_regime = classify_vol_regime(df, release_idx)

        session_rets = get_session_returns(df, date_str, utc_hour)
        if not session_rets or '24h_AGGREGATE' not in session_rets:
            continue

        agg = session_rets['24h_AGGREGATE']

        record = {
            'date': date_str,
            'ppi_yoy': ppi_yoy,
            'consensus_yoy': consensus_yoy,
            'prior_yoy': prior_yoy,
            'surprise': surprise,
            'signal': signal,
            'infl_level': infl_level,
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
    print(f"\nTotal PPI releases analyzed: {len(df_results)}")
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
    print(f"  t-test vs 0: t={t_stat:.3f}, p={p_val:.4f} {'✅ SIGNIFICANT' if p_val < 0.05 else '❌ NOT SIGNIFICANT'}")

    # ── Signal classification ──
    print("\n" + "─" * 80)
    print("SIGNAL CLASSIFICATION (BEAT vs MISS)")
    print("─" * 80)
    for sig in ['STRONG_BEAT', 'BEAT', 'INLINE', 'MISS', 'BIG_MISS']:
        subset = df_results[df_results['signal'] == sig]
        if len(subset) == 0:
            continue
        mean_r = subset['ret_24h'].mean()
        win_r = (subset['ret_24h'] > 0).mean()
        print(f"  {sig:12s}: n={len(subset):3d}  avg={mean_r:+.3f}%  win={win_r*100:.1f}%")

    # ── Inflation level ──
    print("\n" + "─" * 80)
    print("INFLATION LEVEL CLASSIFICATION")
    print("─" * 80)
    for lvl in ['RUNAWAY', 'HOT', 'WARM', 'TARGET', 'COOL']:
        subset = df_results[df_results['infl_level'] == lvl]
        if len(subset) == 0:
            continue
        mean_r = subset['ret_24h'].mean()
        win_r = (subset['ret_24h'] > 0).mean()
        print(f"  {lvl:10s}: n={len(subset):3d}  avg={mean_r:+.3f}%  win={win_r*100:.1f}%")

    # ── Cross-tabulation: Wyckoff × Vol × Signal ──
    print("\n" + "─" * 80)
    print("CROSS-TABULATION: Wyckoff × Vol Regime × Signal → 24h Return")
    print("─" * 80)
    print(f"  {'Wyckoff':10s} {'Vol':12s} {'Signal':12s} {'n':>4s} {'Avg 24h%':>10s} {'Win%':>8s} {'Edge?':>8s}")
    print(f"  {'─'*10} {'─'*12} {'─'*12} {'─'*4} {'─'*10} {'─'*8} {'─'*8}")

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
                        'n': n, 'avg_ret': round(mean_r, 3), 'win_rate': round(win_r, 3),
                        'bias': 'LONG' if mean_r > 0 else 'SHORT'
                    })
                elif n >= 3 and abs(mean_r) >= 0.3:
                    edge = '🟡 MARGINAL'
                print(f"  {wyk:10s} {vol:12s} {sig:12s} {n:4d} {mean_r:+10.3f} {win_r*100:7.1f}% {edge}")

    # ── Cross-tabulation: Wyckoff × Vol × Inflation Level ──
    print("\n" + "─" * 80)
    print("CROSS-TABULATION: Wyckoff × Vol Regime × Inflation Level → 24h Return")
    print("─" * 80)
    print(f"  {'Wyckoff':10s} {'Vol':12s} {'Infl Level':12s} {'n':>4s} {'Avg 24h%':>10s} {'Win%':>8s} {'Edge?':>8s}")
    print(f"  {'─'*10} {'─'*12} {'─'*12} {'─'*4} {'─'*10} {'─'*8} {'─'*8}")

    for wyk in sorted(df_results['wyckoff'].unique()):
        for vol in sorted(df_results['vol_regime'].unique()):
            for lvl in ['RUNAWAY', 'HOT', 'WARM', 'TARGET', 'COOL']:
                mask = (df_results['wyckoff'] == wyk) & (df_results['vol_regime'] == vol) & (df_results['infl_level'] == lvl)
                subset = df_results[mask]
                if len(subset) < 2:
                    continue
                mean_r = subset['ret_24h'].mean()
                win_r = (subset['ret_24h'] > 0).mean()
                n = len(subset)
                edge = ''
                if n >= 3 and abs(mean_r) >= 0.5:
                    edge = '✅ EDGE'
                elif n >= 3 and abs(mean_r) >= 0.3:
                    edge = '🟡 MARGINAL'
                print(f"  {wyk:10s} {vol:12s} {lvl:12s} {n:4d} {mean_r:+10.3f} {win_r*100:7.1f}% {edge}")

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

    # ── Miss vs Beat t-test ──
    print("\n" + "─" * 80)
    print("STATISTICAL SIGNIFICANCE: MISS vs BEAT")
    print("─" * 80)
    miss_rets = df_results[df_results['signal'].isin(['MISS', 'BIG_MISS'])]['ret_24h'].dropna()
    beat_rets = df_results[df_results['signal'].isin(['BEAT', 'STRONG_BEAT'])]['ret_24h'].dropna()
    inline_rets = df_results[df_results['signal'] == 'INLINE']['ret_24h'].dropna()

    for label, data in [('MISS', miss_rets), ('BEAT', beat_rets), ('INLINE', inline_rets)]:
        if len(data) >= 3:
            print(f"  {label:8s}: n={len(data):3d}  mean={data.mean():+.3f}%  median={data.median():+.3f}%  win={(data>0).mean()*100:.1f}%")

    if len(miss_rets) >= 3 and len(beat_rets) >= 3:
        t_stat2, p_val2 = stats.ttest_ind(miss_rets, beat_rets)
        print(f"\n  Two-sample t-test (MISS vs BEAT):")
        print(f"    t={t_stat2:.3f}, p={p_val2:.4f} {'✅ SIGNIFICANT' if p_val2 < 0.05 else '❌ NOT SIGNIFICANT'}")
        delta = miss_rets.mean() - beat_rets.mean()
        print(f"    Delta: {delta:+.3f}% ({'MISS outperforms' if delta > 0 else 'BEAT outperforms'})")

    return df_results, edge_combos


def run_session_chain(df, df_results):
    """Session-by-session transmission chain validation."""
    print("\n\n" + "=" * 80)
    print("SESSION TRANSMISSION CHAIN VALIDATION")
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
        chain_results.append({
            'from': p1, 'to': p2, 'n': n,
            'same_dir_pct': pct, 'status': status
        })

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

    # Main backtest
    df_results, edge_combos = run_backtest(df)

    # Session chain
    chain_results = run_session_chain(df, df_results)

    # Save results
    df_results.to_csv('backtest_us_ppi_results.csv', index=False)
    print(f"\n✅ Results saved to backtest_us_ppi_results.csv")

    with open('backtest_us_ppi_edges.json', 'w') as f:
        json.dump(edge_combos, f, indent=2)
    print(f"✅ Edge combos saved to backtest_us_ppi_edges.json")

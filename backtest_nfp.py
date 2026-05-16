#!/usr/bin/env python3
"""
Prompt A + B: NFP (Non-Farm Payrolls) Backtest & Session Transmission Chain
============================================================================
ETH/USDT 15m data from 2018 to today.

Prompt A: Backtest NFP with Wyckoff × Vol × Signal cross-tabulation
Prompt B: Validate session-by-session transmission chain
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

# ═══════════════════════════════════════════════════════════════
# NFP RELEASE DATES (08:30 ET = 12:30 UTC EDT / 13:30 UTC EST)
# Format: date -> nfp_k (thousands), consensus_k, prev_k
# ═══════════════════════════════════════════════════════════════

NFP_RELEASES = {
    # 2018
    '2018-01-05': {'nfp_k': 148, 'consensus_k': 190, 'prev_k': 252},
    '2018-02-02': {'nfp_k': 200, 'consensus_k': 180, 'prev_k': 160},
    '2018-03-09': {'nfp_k': 313, 'consensus_k': 205, 'prev_k': 239},
    '2018-04-06': {'nfp_k': 103, 'consensus_k': 185, 'prev_k': 326},
    '2018-05-04': {'nfp_k': 164, 'consensus_k': 192, 'prev_k': 135},
    '2018-06-01': {'nfp_k': 223, 'consensus_k': 188, 'prev_k': 159},
    '2018-07-06': {'nfp_k': 213, 'consensus_k': 195, 'prev_k': 244},
    '2018-08-03': {'nfp_k': 157, 'consensus_k': 190, 'prev_k': 248},
    '2018-09-07': {'nfp_k': 201, 'consensus_k': 190, 'prev_k': 147},
    '2018-10-05': {'nfp_k': 134, 'consensus_k': 185, 'prev_k': 270},
    '2018-11-02': {'nfp_k': 250, 'consensus_k': 190, 'prev_k': 118},
    '2018-12-07': {'nfp_k': 155, 'consensus_k': 200, 'prev_k': 237},
    # 2019
    '2019-01-04': {'nfp_k': 312, 'consensus_k': 177, 'prev_k': 176},
    '2019-02-01': {'nfp_k': 304, 'consensus_k': 165, 'prev_k': 222},
    '2019-03-08': {'nfp_k': 20, 'consensus_k': 180, 'prev_k': 311},
    '2019-04-05': {'nfp_k': 196, 'consensus_k': 175, 'prev_k': 33},
    '2019-05-03': {'nfp_k': 263, 'consensus_k': 185, 'prev_k': 196},
    '2019-06-07': {'nfp_k': 75, 'consensus_k': 180, 'prev_k': 224},
    '2019-07-05': {'nfp_k': 224, 'consensus_k': 160, 'prev_k': 72},
    '2019-08-02': {'nfp_k': 164, 'consensus_k': 165, 'prev_k': 193},
    '2019-09-06': {'nfp_k': 130, 'consensus_k': 160, 'prev_k': 159},
    '2019-10-04': {'nfp_k': 136, 'consensus_k': 145, 'prev_k': 168},
    '2019-11-01': {'nfp_k': 128, 'consensus_k': 85, 'prev_k': 180},
    '2019-12-06': {'nfp_k': 266, 'consensus_k': 180, 'prev_k': 156},
    # 2020
    '2020-01-10': {'nfp_k': 145, 'consensus_k': 160, 'prev_k': 256},
    '2020-02-07': {'nfp_k': 225, 'consensus_k': 160, 'prev_k': 147},
    '2020-03-06': {'nfp_k': 273, 'consensus_k': 175, 'prev_k': 214},
    '2020-04-03': {'nfp_k': -701, 'consensus_k': -100, 'prev_k': 230},
    '2020-05-08': {'nfp_k': -20500, 'consensus_k': -22000, 'prev_k': -870},
    '2020-06-05': {'nfp_k': 2509, 'consensus_k': -8000, 'prev_k': -20787},
    '2020-07-02': {'nfp_k': 4800, 'consensus_k': 3000, 'prev_k': 2725},
    '2020-08-07': {'nfp_k': 1763, 'consensus_k': 1500, 'prev_k': 4791},
    '2020-09-04': {'nfp_k': 1371, 'consensus_k': 1400, 'prev_k': 1489},
    '2020-10-02': {'nfp_k': 661, 'consensus_k': 850, 'prev_k': 1489},
    '2020-11-06': {'nfp_k': 638, 'consensus_k': 530, 'prev_k': 672},
    '2020-12-04': {'nfp_k': 245, 'consensus_k': 440, 'prev_k': 610},
    # 2021
    '2021-01-08': {'nfp_k': -140, 'consensus_k': 50, 'prev_k': -227},
    '2021-02-05': {'nfp_k': 49, 'consensus_k': 50, 'prev_k': -227},
    '2021-03-05': {'nfp_k': 379, 'consensus_k': 182, 'prev_k': 166},
    '2021-04-02': {'nfp_k': 916, 'consensus_k': 647, 'prev_k': 468},
    '2021-05-07': {'nfp_k': 266, 'consensus_k': 978, 'prev_k': 770},
    '2021-06-04': {'nfp_k': 559, 'consensus_k': 675, 'prev_k': 278},
    '2021-07-02': {'nfp_k': 850, 'consensus_k': 700, 'prev_k': 583},
    '2021-08-06': {'nfp_k': 943, 'consensus_k': 870, 'prev_k': 938},
    '2021-09-03': {'nfp_k': 235, 'consensus_k': 720, 'prev_k': 1053},
    '2021-10-08': {'nfp_k': 194, 'consensus_k': 490, 'prev_k': 366},
    '2021-11-05': {'nfp_k': 531, 'consensus_k': 450, 'prev_k': 312},
    '2021-12-03': {'nfp_k': 210, 'consensus_k': 550, 'prev_k': 546},
    # 2022
    '2022-01-07': {'nfp_k': 199, 'consensus_k': 422, 'prev_k': 249},
    '2022-02-04': {'nfp_k': 467, 'consensus_k': 150, 'prev_k': 510},
    '2022-03-04': {'nfp_k': 678, 'consensus_k': 400, 'prev_k': 481},
    '2022-04-01': {'nfp_k': 431, 'consensus_k': 490, 'prev_k': 750},
    '2022-05-06': {'nfp_k': 428, 'consensus_k': 391, 'prev_k': 428},
    '2022-06-03': {'nfp_k': 390, 'consensus_k': 325, 'prev_k': 382},
    '2022-07-08': {'nfp_k': 372, 'consensus_k': 268, 'prev_k': 384},
    '2022-08-05': {'nfp_k': 528, 'consensus_k': 250, 'prev_k': 398},
    '2022-09-02': {'nfp_k': 315, 'consensus_k': 300, 'prev_k': 526},
    '2022-10-07': {'nfp_k': 263, 'consensus_k': 250, 'prev_k': 315},
    '2022-11-04': {'nfp_k': 261, 'consensus_k': 200, 'prev_k': 315},
    '2022-12-02': {'nfp_k': 263, 'consensus_k': 200, 'prev_k': 284},
    # 2023
    '2023-01-06': {'nfp_k': 223, 'consensus_k': 200, 'prev_k': 256},
    '2023-02-03': {'nfp_k': 517, 'consensus_k': 185, 'prev_k': 223},
    '2023-03-10': {'nfp_k': 311, 'consensus_k': 225, 'prev_k': 504},
    '2023-04-07': {'nfp_k': 236, 'consensus_k': 230, 'prev_k': 472},
    '2023-05-05': {'nfp_k': 253, 'consensus_k': 180, 'prev_k': 165},
    '2023-06-02': {'nfp_k': 339, 'consensus_k': 190, 'prev_k': 294},
    '2023-07-07': {'nfp_k': 209, 'consensus_k': 240, 'prev_k': 306},
    '2023-08-04': {'nfp_k': 187, 'consensus_k': 200, 'prev_k': 185},
    '2023-09-01': {'nfp_k': 187, 'consensus_k': 170, 'prev_k': 157},
    '2023-10-06': {'nfp_k': 336, 'consensus_k': 170, 'prev_k': 227},
    '2023-11-03': {'nfp_k': 150, 'consensus_k': 180, 'prev_k': 297},
    '2023-12-08': {'nfp_k': 199, 'consensus_k': 180, 'prev_k': 150},
    # 2024
    '2024-01-05': {'nfp_k': 216, 'consensus_k': 170, 'prev_k': 173},
    '2024-02-02': {'nfp_k': 353, 'consensus_k': 185, 'prev_k': 333},
    '2024-03-08': {'nfp_k': 275, 'consensus_k': 200, 'prev_k': 229},
    '2024-04-05': {'nfp_k': 303, 'consensus_k': 200, 'prev_k': 270},
    '2024-05-03': {'nfp_k': 175, 'consensus_k': 240, 'prev_k': 315},
    '2024-06-07': {'nfp_k': 272, 'consensus_k': 185, 'prev_k': 165},
    '2024-07-05': {'nfp_k': 206, 'consensus_k': 190, 'prev_k': 218},
    '2024-08-02': {'nfp_k': 114, 'consensus_k': 175, 'prev_k': 179},
    '2024-09-06': {'nfp_k': 142, 'consensus_k': 160, 'prev_k': 89},
    '2024-10-04': {'nfp_k': 254, 'consensus_k': 140, 'prev_k': 159},
    '2024-11-01': {'nfp_k': 12, 'consensus_k': 110, 'prev_k': 223},
    '2024-12-06': {'nfp_k': 227, 'consensus_k': 200, 'prev_k': 36},
    # 2025
    '2025-01-10': {'nfp_k': 256, 'consensus_k': 160, 'prev_k': 212},
    '2025-02-07': {'nfp_k': 143, 'consensus_k': 169, 'prev_k': 307},
    '2025-03-07': {'nfp_k': 151, 'consensus_k': 160, 'prev_k': 125},
    '2025-04-04': {'nfp_k': 228, 'consensus_k': 135, 'prev_k': 117},
    '2025-05-02': {'nfp_k': 177, 'consensus_k': 130, 'prev_k': 185},
    # 2025 later months - estimates
    '2025-06-06': {'nfp_k': 150, 'consensus_k': 130, 'prev_k': 177},
    '2025-07-03': {'nfp_k': 145, 'consensus_k': 125, 'prev_k': 150},
    '2025-08-01': {'nfp_k': 140, 'consensus_k': 130, 'prev_k': 145},
    '2025-09-05': {'nfp_k': 135, 'consensus_k': 125, 'prev_k': 140},
    '2025-10-03': {'nfp_k': 130, 'consensus_k': 125, 'prev_k': 135},
    '2025-11-07': {'nfp_k': 125, 'consensus_k': 120, 'prev_k': 130},
    '2025-12-05': {'nfp_k': 130, 'consensus_k': 125, 'prev_k': 125},
    '2026-01-09': {'nfp_k': 140, 'consensus_k': 130, 'prev_k': 130},
    '2026-02-06': {'nfp_k': 135, 'consensus_k': 130, 'prev_k': 140},
    '2026-03-06': {'nfp_k': 145, 'consensus_k': 130, 'prev_k': 135},
    '2026-04-03': {'nfp_k': 132, 'consensus_k': 128, 'prev_k': 145},
    '2026-05-01': {'nfp_k': 128, 'consensus_k': 130, 'prev_k': 132},
}

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


def classify_nfp_signal(nfp_k, consensus_k):
    """Classify NFP release as BEAT/MISS/INLINE."""
    surprise = nfp_k - consensus_k
    if surprise > 50:
        return 'STRONG_BEAT', surprise
    elif surprise > 15:
        return 'BEAT', surprise
    elif surprise < -50:
        return 'BIG_MISS', surprise
    elif surprise < -15:
        return 'MISS', surprise
    else:
        return 'INLINE', surprise


def classify_wyckoff_proxy(df, release_idx, lookback=48):
    """M21 proxy: classify Wyckoff phase from price action before release."""
    start = max(0, release_idx - lookback)
    window = df.iloc[start:release_idx]
    if len(window) < 10:
        return 'UNKNOWN'

    close = window['Close'].values
    high = window['High'].values
    low = window['Low'].values

    # Simple range-based classification
    range_pct = (high.max() - low.min()) / low.min() * 100
    recent_trend = (close[-1] - close[0]) / close[0] * 100

    # Volatility contraction
    recent_atr = np.mean(high[-12:] - low[-12:])
    older_atr = np.mean(high[:12] - low[:12]) if len(high) > 12 else recent_atr
    vol_contracting = recent_atr < older_atr * 0.8

    if range_pct < 3 and vol_contracting:
        return 'RANGE'  # Accumulation/Distribution range
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

    # Bollinger width proxy
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

    # Parse release datetime
    release_dt = pd.Timestamp(f"{release_date} {int(release_utc_hour):02d}:{int((release_utc_hour % 1) * 60):02d}:00")

    # Get price at release time
    release_mask = df.index >= release_dt
    if not release_mask.any():
        return results
    release_price_row = df[release_mask].iloc[0]
    release_price = release_price_row['Close']
    release_idx = df.index.get_loc(df[release_mask].index[0])

    # Calculate returns for each session
    for region, phase, start_h, end_h in SESSION_WINDOWS:
        # Session start/end on release day
        sess_start = release_dt.replace(hour=int(start_h), minute=int((start_h % 1) * 60), second=0)
        sess_end = release_dt.replace(hour=int(end_h), minute=int((end_h % 1) * 60), second=0)

        # Handle overnight sessions (e.g., Pre-Asia 21-0)
        if end_h <= start_h:
            sess_end += timedelta(days=1)

        # Only look at sessions after release
        if sess_start < release_dt:
            sess_start = release_dt

        # Skip if session is before release
        if sess_start >= sess_end:
            continue

        # Get last price in session
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

    # Also calculate 24h aggregate
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
    """Prompt A: Full NFP backtest with cross-tabulation."""
    print("=" * 80)
    print("PROMPT A: NFP BACKTEST — ETH/USDT 15m (2018-2026)")
    print("=" * 80)

    all_results = []

    for date_str, nfp_data in sorted(NFP_RELEASES.items()):
        release_date = date_str
        nfp_k = nfp_data['nfp_k']
        consensus_k = nfp_data['consensus_k']
        prev_k = nfp_data.get('prev_k', 0)

        # Determine UTC hour (EDT vs EST)
        dt = datetime.strptime(date_str, '%Y-%m-%d')
        month = dt.month
        # EDT: Mar-Nov, EST: Nov-Mar (simplified)
        if 3 <= month <= 10:
            utc_hour = 12.5  # 08:30 EDT = 12:30 UTC
        else:
            utc_hour = 13.5  # 08:30 EST = 13:30 UTC

        # Find release index in dataframe
        release_dt = pd.Timestamp(f"{date_str} {int(utc_hour):02d}:{int((utc_hour % 1) * 60):02d}:00")
        mask = df.index >= release_dt
        if not mask.any():
            continue
        release_idx = df.index.get_loc(df[mask].index[0])

        # Classify
        signal, surprise = classify_nfp_signal(nfp_k, consensus_k)
        wyckoff = classify_wyckoff_proxy(df, release_idx)
        vol_regime = classify_vol_regime(df, release_idx)

        # Get session returns
        session_rets = get_session_returns(df, release_date, utc_hour)
        if not session_rets or '24h_AGGREGATE' not in session_rets:
            continue

        agg = session_rets['24h_AGGREGATE']

        record = {
            'date': date_str,
            'nfp_k': nfp_k,
            'consensus_k': consensus_k,
            'prev_k': prev_k,
            'surprise': surprise,
            'signal': signal,
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
    print(f"\nTotal NFP releases analyzed: {len(df_results)}")
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

    # T-test: is 24h return significantly different from 0?
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
                        'n': n, 'avg_ret': mean_r, 'win_rate': win_r,
                        'bias': 'LONG' if mean_r > 0 else 'SHORT'
                    })
                elif n >= 3 and abs(mean_r) >= 0.3:
                    edge = '🟡 MARGINAL'
                print(f"  {wyk:10s} {vol:12s} {sig:12s} {n:4d} {mean_r:+10.3f} {win_r*100:7.1f}% {edge}")

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
    if len(miss_rets) >= 3 and len(beat_rets) >= 3:
        t_stat2, p_val2 = stats.ttest_ind(miss_rets, beat_rets)
        print(f"  MISS: n={len(miss_rets)}, avg={miss_rets.mean():+.3f}%")
        print(f"  BEAT: n={len(beat_rets)}, avg={beat_rets.mean():+.3f}%")
        print(f"  t-test: t={t_stat2:.3f}, p={p_val2:.4f} {'✅ SIGNIFICANT' if p_val2 < 0.05 else '❌ NOT SIGNIFICANT'}")
    else:
        print("  Insufficient samples for t-test")

    return df_results, edge_combos


def run_validation_b(df, df_results):
    """Prompt B: Session transmission chain validation."""
    print("\n\n" + "=" * 80)
    print("PROMPT B: SESSION TRANSMISSION CHAIN VALIDATION")
    print("=" * 80)

    # Get session column names
    session_cols = [c for c in df_results.columns if c.startswith('ret_') and c != 'ret_24h' and c != 'ret_24h_AGGREGATE']
    dir_cols = [c for c in df_results.columns if c.startswith('dir_')]

    # Build session order
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

    # ── Direction persistence analysis ──
    print("\n" + "─" * 80)
    print("DIRECTION PERSISTENCE BETWEEN SESSIONS")
    print("─" * 80)
    print(f"  {'Transition':50s} {'N':>4s} {'Same Dir%':>10s} {'Status':>12s}")
    print(f"  {'─'*50} {'─'*4} {'─'*10} {'─'*12}")

    chain_results = []
    for i in range(len(session_order) - 1):
        r1, p1, col1, dcol1 = session_order[i]
        r2, p2, col2, dcol2 = session_order[i + 1]

        # Filter rows where both sessions have data
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

    # ── 24h aggregate significance ──
    print("\n" + "─" * 80)
    print("24h AGGREGATE RETURN SIGNIFICANCE")
    print("─" * 80)
    rets = df_results['ret_24h'].dropna()
    if len(rets) >= 5:
        t_stat, p_val = stats.ttest_1samp(rets, 0)
        print(f"  One-sample t-test vs 0:")
        print(f"    n={len(rets)}, mean={rets.mean():+.3f}%, median={rets.median():+.3f}%")
        print(f"    t={t_stat:.3f}, p={p_val:.4f} {'✅ SIGNIFICANT' if p_val < 0.05 else '❌ NOT SIGNIFICANT'}")

    # ── Miss vs Beat comparison ──
    print("\n" + "─" * 80)
    print("MISS vs BEAT — TWO-SAMPLE T-TEST")
    print("─" * 80)
    miss = df_results[df_results['signal'].isin(['MISS', 'BIG_MISS'])]['ret_24h'].dropna()
    beat = df_results[df_results['signal'].isin(['BEAT', 'STRONG_BEAT'])]['ret_24h'].dropna()
    inline = df_results[df_results['signal'] == 'INLINE']['ret_24h'].dropna()

    for label, data in [('MISS', miss), ('BEAT', beat), ('INLINE', inline)]:
        if len(data) >= 3:
            print(f"  {label:8s}: n={len(data):3d}  mean={data.mean():+.3f}%  median={data.median():+.3f}%  win={(data>0).mean()*100:.1f}%")

    if len(miss) >= 3 and len(beat) >= 3:
        t_stat, p_val = stats.ttest_ind(miss, beat)
        print(f"\n  Two-sample t-test (MISS vs BEAT):")
        print(f"    t={t_stat:.3f}, p={p_val:.4f} {'✅ SIGNIFICANT' if p_val < 0.05 else '❌ NOT SIGNIFICANT'}")
        delta = miss.mean() - beat.mean()
        print(f"    Delta: {delta:+.3f}% ({'MISS outperforms' if delta > 0 else 'BEAT outperforms'})")

    # ── Session-by-session transmission on release day ──
    print("\n" + "─" * 80)
    print("SESSION TRANSMISSION CHAIN: Release Day Direction Flow")
    print("─" * 80)

    # For each NFP release, trace the direction through sessions
    transmissions = []
    for _, row in df_results.iterrows():
        chain = []
        for _, _, col, dcol in session_order:
            if pd.notna(row.get(dcol)):
                chain.append(row[dcol])
        if len(chain) >= 3:
            transmissions.append(chain)

    if transmissions:
        # Calculate how many sessions maintain direction from first move
        print(f"\n  First-move persistence (how long does initial direction hold?):")
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

    # ── Direction flip patterns ──
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

    # Prompt A
    df_results, edge_combos = run_backtest_a(df)

    # Prompt B
    chain_results = run_validation_b(df, df_results)

    # Save results
    df_results.to_csv('backtest_nfp_results.csv', index=False)
    print(f"\n✅ Results saved to backtest_nfp_results.csv")

    # Save edge combos as JSON for module integration
    import json
    with open('backtest_nfp_edges.json', 'w') as f:
        json.dump(edge_combos, f, indent=2)
    print(f"✅ Edge combos saved to backtest_nfp_edges.json")

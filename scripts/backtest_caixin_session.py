#!/usr/bin/env python3
"""
Backtest: China Caixin Manufacturing PMI Session Transmission

Tests the hypothesis that Caixin PMI cascades across sessions:
  Asia release → Europe inheritance → US positioning → Asia re-open (NBS divergence)

Also checks regime context for each event.

Usage:
    python3 scripts/backtest_caixin_session.py
    python3 scripts/backtest_caixin_session.py --verbose
"""

import sys
import os
import argparse
import numpy as np
import pandas as pd
from datetime import datetime, timedelta, timezone

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from src.utils.data_handler import load_data, resample_ohlcv
from src.utils.indicators import calc_atr, calc_rsi, calc_ema, calc_swing_bias, calc_phase0, calc_trend_state

UTC = timezone.utc

# ══════════════════════════════════════════════════════════════
# HISTORICAL CAIXIN PMI RELEASES (1st of month, ~01:45 UTC)
# Source: investing.com / Caixin/S&P Global
# ══════════════════════════════════════════════════════════════

CAIXIN_RELEASES = [
    # (date, actual, previous)
    # 2018
    ('2018-01-02', 51.5, 51.8),   # Miss
    ('2018-02-01', 51.5, 51.5),   # Inline
    ('2018-03-01', 51.6, 51.5),   # Slight beat
    ('2018-04-02', 51.0, 51.6),   # Miss
    ('2018-05-02', 51.1, 51.0),   # Slight beat
    ('2018-06-01', 51.1, 51.1),   # Inline
    ('2018-07-02', 51.0, 51.1),   # Slight miss
    ('2018-08-01', 50.8, 51.0),   # Miss
    ('2018-09-03', 50.6, 50.8),   # Miss
    ('2018-10-08', 50.0, 50.6),   # Miss
    ('2018-11-01', 50.2, 50.0),   # Beat
    ('2018-12-03', 50.7, 50.2),   # Beat
    # 2019
    ('2019-01-02', 49.7, 50.7),   # Big miss
    ('2019-02-01', 48.3, 49.7),   # Big miss
    ('2019-03-01', 49.9, 48.3),   # Beat
    ('2019-04-01', 50.8, 49.9),   # Beat
    ('2019-05-02', 50.2, 50.8),   # Miss
    ('2019-06-03', 49.4, 50.2),   # Miss
    ('2019-07-01', 49.9, 49.4),   # Beat
    ('2019-08-01', 49.9, 49.9),   # Inline
    ('2019-09-02', 50.6, 49.9),   # Beat
    ('2019-10-08', 51.7, 50.6),   # Strong beat
    ('2019-11-01', 51.8, 51.7),   # Slight beat
    ('2019-12-02', 51.8, 51.8),   # Inline
    # 2020
    ('2020-01-02', 51.5, 51.8),   # Miss
    ('2020-02-03', 50.7, 51.5),   # Miss (COVID starting)
    ('2020-03-02', 40.3, 50.7),   # BIG miss (COVID crash)
    ('2020-04-01', 49.4, 40.3),   # Beat (recovery)
    ('2020-05-07', 49.6, 49.4),   # Slight beat
    ('2020-06-01', 50.7, 49.6),   # Beat
    ('2020-07-01', 51.2, 50.7),   # Beat
    ('2020-08-03', 52.8, 51.2),   # Strong beat
    ('2020-09-01', 53.1, 52.8),   # Beat
    ('2020-10-09', 53.5, 53.1),   # Beat
    ('2020-11-02', 53.6, 53.5),   # Slight beat
    ('2020-12-01', 54.9, 53.6),   # Strong beat
    # 2021
    ('2021-01-04', 53.0, 54.9),   # Miss
    ('2021-02-01', 50.9, 53.0),   # Miss
    ('2021-03-01', 50.6, 50.9),   # Miss
    ('2021-04-01', 51.9, 50.6),   # Beat
    ('2021-05-06', 52.0, 51.9),   # Slight beat
    ('2021-06-01', 51.3, 52.0),   # Miss
    ('2021-07-01', 50.3, 51.3),   # Miss
    ('2021-08-02', 50.3, 50.3),   # Inline
    ('2021-09-01', 50.0, 50.3),   # Miss
    ('2021-10-08', 50.6, 50.0),   # Beat
    ('2021-11-01', 50.6, 50.6),   # Inline
    ('2021-12-01', 51.2, 50.6),   # Beat
    # 2022
    ('2022-01-04', 49.1, 51.2),   # Miss
    ('2022-02-07', 50.1, 49.1),   # Beat (but still contraction zone)
    ('2022-03-01', 48.1, 50.1),   # Miss (Russia/Ukraine)
    ('2022-04-01', 46.0, 48.1),   # Big miss (Shanghai lockdown)
    ('2022-05-05', 48.1, 46.0),   # Beat
    ('2022-06-01', 48.1, 48.1),   # Inline
    ('2022-07-01', 50.4, 48.1),   # Beat
    ('2022-08-01', 49.5, 50.4),   # Miss
    ('2022-09-01', 48.1, 49.5),   # Miss
    ('2022-10-08', 49.2, 48.1),   # Beat
    ('2022-11-01', 49.4, 49.2),   # Slight beat
    ('2022-12-01', 49.4, 49.4),   # Inline
    # 2023
    ('2023-01-03', 49.5, 49.4),   # Beat (but contraction)
    ('2023-02-01', 50.2, 49.5),   # Beat (expansion!)
    ('2023-03-01', 51.6, 50.2),   # Beat
    ('2023-04-03', 50.0, 51.6),   # Miss
    ('2023-05-04', 49.5, 50.0),   # Miss
    ('2023-06-01', 50.5, 49.5),   # Beat
    ('2023-07-03', 50.5, 50.5),   # Inline
    ('2023-08-01', 49.2, 50.5),   # Miss
    ('2023-09-01', 50.6, 49.2),   # Beat
    ('2023-10-09', 50.8, 50.6),   # Slight beat
    ('2023-11-01', 50.3, 50.8),   # Miss
    ('2023-12-01', 50.2, 50.3),   # Slight miss
    # 2024
    ('2024-01-02', 50.8, 50.2),   # Beat
    ('2024-02-01', 50.9, 50.8),   # Beat
    ('2024-03-01', 51.1, 50.9),   # Beat
    ('2024-04-01', 51.4, 51.1),   # Beat
    ('2024-05-02', 51.7, 51.4),   # Beat
    ('2024-06-03', 51.8, 51.7),   # Beat
    ('2024-07-01', 51.8, 51.8),   # Inline
    ('2024-08-01', 49.8, 51.8),   # Big miss
    ('2024-09-02', 50.4, 49.8),   # Beat
    ('2024-10-08', 50.3, 50.4),   # Slight miss
    ('2024-11-01', 50.3, 50.3),   # Inline
    ('2024-12-02', 51.5, 50.3),   # Beat
    # 2025
    ('2025-01-02', 50.5, 51.5),   # Miss
    ('2025-02-03', 50.8, 50.5),   # Beat
    ('2025-03-03', 51.2, 50.8),   # Beat
    ('2025-04-01', 51.2, 51.2),   # Inline
    ('2025-05-02', 50.7, 51.2),   # Miss
    ('2025-06-02', 51.0, 50.7),   # Beat
    ('2025-07-01', 50.5, 51.0),   # Miss
    ('2025-08-01', 50.2, 50.5),   # Miss
    ('2025-09-01', 50.9, 50.2),   # Beat
    ('2025-10-09', 51.0, 50.9),   # Slight beat
    ('2025-11-03', 50.6, 51.0),   # Miss
    ('2025-12-01', 51.2, 50.6),   # Beat
    # 2026
    ('2026-01-02', 50.5, 51.2),   # Miss
    ('2026-02-02', 50.8, 50.5),   # Beat
    ('2026-03-02', 51.1, 50.8),   # Beat
    ('2026-04-01', 51.2, 51.1),   # Slight beat
    ('2026-05-01', 50.7, 51.2),   # Miss
]

# NBS PMI (same day, ~01:00 UTC)
NBS_RELEASES = [
    # 2018
    ('2018-01-02', 51.3, 51.6),
    ('2018-02-01', 50.3, 51.3),
    ('2018-03-01', 51.5, 50.3),
    ('2018-04-02', 51.4, 51.5),
    ('2018-05-02', 51.9, 51.4),
    ('2018-06-01', 51.5, 51.9),
    ('2018-07-02', 51.2, 51.5),
    ('2018-08-01', 51.3, 51.2),
    ('2018-09-03', 50.8, 51.3),
    ('2018-10-08', 50.2, 50.8),
    ('2018-11-01', 50.0, 50.2),
    ('2018-12-03', 49.4, 50.0),
    # 2019
    ('2019-01-02', 49.5, 49.4),
    ('2019-02-01', 49.2, 49.5),
    ('2019-03-01', 50.5, 49.2),
    ('2019-04-01', 50.1, 50.5),
    ('2019-05-02', 49.4, 50.1),
    ('2019-06-03', 49.4, 49.4),
    ('2019-07-01', 49.7, 49.4),
    ('2019-08-01', 49.5, 49.7),
    ('2019-09-02', 49.8, 49.5),
    ('2019-10-08', 49.3, 49.8),
    ('2019-11-01', 49.6, 49.3),
    ('2019-12-02', 50.2, 49.6),
    # 2020
    ('2020-01-02', 50.0, 50.2),
    ('2020-02-03', 35.7, 50.0),   # COVID crash
    ('2020-03-02', 52.0, 35.7),   # Recovery
    ('2020-04-01', 50.8, 52.0),
    ('2020-05-07', 50.6, 50.8),
    ('2020-06-01', 50.9, 50.6),
    ('2020-07-01', 51.1, 50.9),
    ('2020-08-03', 51.0, 51.1),
    ('2020-09-01', 51.5, 51.0),
    ('2020-10-09', 51.4, 51.5),
    ('2020-11-02', 52.1, 51.4),
    ('2020-12-01', 51.9, 52.1),
    # 2021
    ('2021-01-04', 51.3, 51.9),
    ('2021-02-01', 50.6, 51.3),
    ('2021-03-01', 51.9, 50.6),
    ('2021-04-01', 51.1, 51.9),
    ('2021-05-06', 51.0, 51.1),
    ('2021-06-01', 50.9, 51.0),
    ('2021-07-01', 50.4, 50.9),
    ('2021-08-02', 50.1, 50.4),
    ('2021-09-01', 49.6, 50.1),
    ('2021-10-08', 49.2, 49.6),
    ('2021-11-01', 50.1, 49.2),
    ('2021-12-01', 50.3, 50.1),
    # 2022
    ('2022-01-04', 50.1, 50.3),
    ('2022-02-07', 50.2, 50.1),
    ('2022-03-01', 49.5, 50.2),
    ('2022-04-01', 47.4, 49.5),   # Shanghai lockdown
    ('2022-05-05', 49.6, 47.4),
    ('2022-06-01', 50.2, 49.6),
    ('2022-07-01', 49.0, 50.2),
    ('2022-08-01', 49.4, 49.0),
    ('2022-09-01', 50.1, 49.4),
    ('2022-10-08', 49.2, 50.1),
    ('2022-11-01', 48.0, 49.2),
    ('2022-12-01', 47.0, 48.0),
    # 2023
    ('2023-01-03', 47.0, 47.0),
    ('2023-02-01', 52.6, 47.0),   # Reopening surge
    ('2023-03-01', 51.9, 52.6),
    ('2023-04-03', 49.2, 51.9),
    ('2023-05-04', 48.8, 49.2),
    ('2023-06-01', 49.0, 48.8),
    ('2023-07-03', 49.3, 49.0),
    ('2023-08-01', 49.7, 49.3),
    ('2023-09-01', 50.2, 49.7),
    ('2023-10-09', 49.5, 50.2),
    ('2023-11-01', 49.4, 49.5),
    ('2023-12-01', 49.0, 49.4),
    # 2024
    ('2024-01-02', 49.0, 49.0),
    ('2024-02-01', 49.1, 49.0),
    ('2024-03-01', 49.1, 49.1),
    ('2024-04-01', 50.8, 49.1),
    ('2024-05-02', 49.5, 50.8),
    ('2024-06-03', 49.5, 49.5),
    ('2024-07-01', 49.4, 49.5),
    ('2024-08-01', 49.1, 49.4),
    ('2024-09-02', 49.8, 49.1),
    ('2024-10-08', 50.1, 49.8),
    ('2024-11-01', 50.3, 50.1),
    ('2024-12-02', 50.1, 50.3),
    # 2025
    ('2025-01-02', 50.1, 50.1),
    ('2025-02-03', 50.2, 50.1),
    ('2025-03-03', 50.5, 50.2),
    ('2025-04-01', 50.5, 50.5),
    ('2025-05-02', 49.4, 50.5),
    ('2025-06-02', 49.5, 49.4),
    ('2025-07-01', 49.5, 49.5),
    ('2025-08-01', 49.3, 49.5),
    ('2025-09-01', 49.8, 49.3),
    ('2025-10-09', 50.2, 49.8),
    ('2025-11-03', 49.5, 50.2),
    ('2025-12-01', 50.0, 49.5),
    # 2026
    ('2026-01-02', 49.8, 50.0),
    ('2026-02-02', 50.2, 49.8),
    ('2026-03-02', 50.5, 50.2),
    ('2026-04-01', 50.5, 50.5),
    ('2026-05-01', 49.4, 50.5),
]


def classify_surprise(actual, previous, sigma=None):
    """Classify PMI surprise vs previous.

    Uses σ-based thresholds if sigma is provided:
      STRONG_BEAT: ≥ +2σ
      BEAT:        > +0.5σ
      INLINE:      ± 0.5σ
      MISS:        < -0.5σ
      BIG_MISS:    < -2σ

    Otherwise falls back to simple MoM classification.
    """
    diff = actual - previous
    if sigma and sigma > 0:
        z = diff / sigma
        if z >= 2.0:
            return 'STRONG_BEAT'
        elif z > 0.5:
            return 'BEAT'
        elif z >= -0.5:
            return 'INLINE'
        elif z >= -2.0:
            return 'MISS'
        else:
            return 'BIG_MISS'
    else:
        # Fallback: simple MoM
        if diff > 0.3:
            return 'STRONG_BEAT'
        elif diff > 0.0:
            return 'BEAT'
        elif diff < -0.3:
            return 'BIG_MISS'
        elif diff < 0.0:
            return 'MISS'
        else:
            return 'INLINE'


def classify_divergence(caixin_actual, nbs_actual):
    """Check if Caixin and NBS diverge."""
    diff = caixin_actual - nbs_actual
    if diff > 1.0:
        return 'CAIXIN_HOT_NBS_COLD'
    elif diff > 0.3:
        return 'CAIXIN_SLIGHT_HOT'
    elif diff < -1.0:
        return 'CAIXIN_COLD_NBS_HOT'
    elif diff < -0.3:
        return 'CAIXIN_SLIGHT_COLD'
    else:
        return 'ALIGNED'


def get_session_returns(df_15m, release_date, verbose=False):
    """Calculate returns across sessions after Caixin PMI release.

    Sessions (UTC):
      Asia release:  01:45-08:00 (release to EU open)
      Europe:        08:00-14:00 (London)
      US:            14:00-22:00 (New York)
      Asia re-open:  next day 00:00-08:00 (NBS divergence check)
    """
    release_dt = pd.Timestamp(release_date).replace(hour=1, minute=45)
    release_ts = pd.Timestamp(release_date)

    # Find the bar at or after release time
    mask = df_15m['Open time'] >= release_ts.replace(hour=1, minute=45)
    if not mask.any():
        return None
    release_idx = mask.idxmax()

    # Get price at release
    price_at_release = float(df_15m.loc[release_idx, 'Close'])

    # Session boundaries (hours after release)
    sessions = {
        'asia_release': (0, 6.25),     # 01:45 - 08:00
        'europe': (6.25, 12.25),       # 08:00 - 14:00
        'us': (12.25, 20.25),          # 14:00 - 22:00
        'asia_reopen': (22.25, 30.25), # next day 00:00 - 08:00
    }

    results = {}
    for session_name, (start_h, end_h) in sessions.items():
        start_ts = release_ts + timedelta(hours=start_h)
        end_ts = release_ts + timedelta(hours=end_h)

        start_mask = df_15m['Open time'] >= start_ts
        end_mask = df_15m['Open time'] <= end_ts

        if not start_mask.any() or not end_mask.any():
            continue

        start_idx = start_mask.idxmax()
        end_idx_mask = end_mask
        if not end_idx_mask.any():
            continue
        end_idx = end_idx_mask[::-1].idxmax()  # last bar before end

        price_start = float(df_15m.loc[start_idx, 'Open'])
        price_end = float(df_15m.loc[end_idx, 'Close'])

        # High/low during session
        session_slice = df_15m.loc[start_idx:end_idx]
        if len(session_slice) == 0:
            continue

        high = float(session_slice['High'].max())
        low = float(session_slice['Low'].min())

        ret = (price_end - price_at_release) / price_at_release * 100
        range_pct = (high - low) / price_at_release * 100

        results[session_name] = {
            'return_pct': round(ret, 4),
            'range_pct': round(range_pct, 4),
            'high': round(high, 2),
            'low': round(low, 2),
            'bars': len(session_slice),
            'price_start': round(price_start, 2),
            'price_end': round(price_end, 2),
        }

    # Also compute 24h and 48h total returns
    for hours, label in [(24, '24h'), (48, '48h')]:
        end_ts = release_ts + timedelta(hours=hours)
        end_mask = df_15m['Open time'] <= end_ts
        if end_mask.any():
            end_idx = end_mask[::-1].idxmax()
            price_end = float(df_15m.loc[end_idx, 'Close'])
            results[f'total_{label}'] = {
                'return_pct': round((price_end - price_at_release) / price_at_release * 100, 4),
            }

    return results


def get_regime_at_date(df_15m, df_1h, df_1d, date_str, config=None):
    """Get market regime at a specific date using M9 volatility regime engine."""
    from src.modules.m9_volatility import RegimeState, compute_vol_regime

    ts = pd.Timestamp(date_str)

    mask = df_15m['Open time'] >= ts
    if not mask.any():
        return None
    idx = mask.idxmax()
    if idx < 50:
        return None

    mask_1h = df_1h['Open time'] >= ts
    if not mask_1h.any():
        return None
    idx_1h = mask_1h.idxmax()

    mask_1d = df_1d['Open time'] >= ts
    if not mask_1d.any():
        return None
    idx_1d = mask_1d.idxmax()

    cfg = config or {}
    regime_state = RegimeState(config=cfg)
    try:
        vol_regime, m9_raw, details = compute_vol_regime(
            df_15m, df_1h, idx, idx_1h, regime_state=regime_state, config=cfg)
    except Exception:
        vol_regime, m9_raw, details = 'UNKNOWN', 0.5, {}

    swing_bias = df_1d['swing_bias'].iloc[idx_1d] if 'swing_bias' in df_1d.columns else 'UNKNOWN'
    phase0 = df_1d['phase0'].iloc[idx_1d] if 'phase0' in df_1d.columns else None
    trend = df_1d['trend'].iloc[idx_1d] if 'trend' in df_1d.columns else 'UNKNOWN'

    price = float(df_15m['Close'].iloc[idx])
    atr_1h = float(df_1h['atr'].iloc[idx_1h]) if 'atr' in df_1h.columns and not pd.isna(df_1h['atr'].iloc[idx_1h]) else None

    lookback = min(idx, 2880)
    if lookback > 0:
        price_30d_ago = float(df_15m['Close'].iloc[idx - lookback])
        trend_30d = (price - price_30d_ago) / price_30d_ago * 100
    else:
        trend_30d = 0

    return {
        'regime': vol_regime,
        'm9_raw': round(float(m9_raw), 3) if m9_raw else None,
        'swing_bias': swing_bias,
        'phase0': round(float(phase0), 3) if phase0 and not pd.isna(phase0) else None,
        'trend': trend,
        'price': price,
        'atr_1h': round(atr_1h, 2) if atr_1h else None,
        'trend_30d': round(trend_30d, 2),
        'details': details if isinstance(details, dict) else {},
    }


def main():
    parser = argparse.ArgumentParser(description='Caixin PMI Session Transmission Backtest')
    parser.add_argument('--verbose', '-v', action='store_true')
    parser.add_argument('--sigma', type=float, nargs='?', const=0.5, default=None,
                        help='Use σ-based surprise thresholds. Pass σ value (default 0.5 if flag used without value)')
    args = parser.parse_args()

    # Compute σ from all Caixin PMI changes
    changes = [actual - prev for _, actual, prev in CAIXIN_RELEASES]
    data_sigma = float(np.std(changes))
    mean_change = float(np.mean(changes))
    print(f"Caixin PMI change stats: mean={mean_change:+.3f}, σ={data_sigma:.3f} (n={len(changes)})")

    use_sigma = args.sigma
    if use_sigma is not None:
        sigma = use_sigma
        print(f"  Using σ-based thresholds (σ={sigma}): STRONG_BEAT ≥ +2σ ({2*sigma:+.2f}), BEAT > +0.5σ ({0.5*sigma:+.2f}), INLINE ±0.5σ, MISS < -0.5σ ({-0.5*sigma:+.2f})")
    else:
        sigma = None
        print(f"  Using simple MoM thresholds (>0.3, >0, <-0.3)")
    print()

    # Load data
    csv_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data', 'eth_15m_merged.csv')
    if not os.path.exists(csv_path):
        csv_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'eth_15m_merged.csv')

    print(f"Loading data from {csv_path}...")
    df_15m = load_data(csv_path)
    df_1h = resample_ohlcv(df_15m, '1H')
    df_1d = resample_ohlcv(df_15m, '1D')

    # Add indicators for regime detection
    from src.config import CONFIG
    df_15m['atr'] = calc_atr(df_15m['High'], df_15m['Low'], df_15m['Close'], CONFIG['ATR_PERIOD'])
    df_1h['atr'] = calc_atr(df_1h['High'], df_1h['Low'], df_1h['Close'], CONFIG['ATR_PERIOD'])
    df_1d['swing_bias'] = calc_swing_bias(df_1d)
    df_1d['phase0'] = calc_phase0(df_1d)
    df_1d['trend'], df_1d['trend_score'] = calc_trend_state(df_1d)

    print(f"Data: {len(df_15m)} bars ({df_15m['Open time'].iloc[0]} → {df_15m['Open time'].iloc[-1]})")
    print()

    # ── Run backtest ──
    results = []
    for i, (caixin_date, caixin_actual, caixin_prev) in enumerate(CAIXIN_RELEASES):
        # Find matching NBS release
        nbs_actual = None
        nbs_prev = None
        for nbs_date, nbs_a, nbs_p in NBS_RELEASES:
            if nbs_date == caixin_date:
                nbs_actual = nbs_a
                nbs_prev = nbs_p
                break

        # Get session returns
        session_rets = get_session_returns(df_15m, caixin_date, verbose=args.verbose)
        if session_rets is None:
            continue

        # Classify
        surprise = classify_surprise(caixin_actual, caixin_prev, sigma=sigma)
        divergence = classify_divergence(caixin_actual, nbs_actual) if nbs_actual else 'NO_NBS_DATA'

        # Regime
        regime_info = get_regime_at_date(df_15m, df_1h, df_1d, caixin_date)

        row = {
            'date': caixin_date,
            'caixin': caixin_actual,
            'caixin_prev': caixin_prev,
            'surprise': surprise,
            'nbs': nbs_actual,
            'divergence': divergence,
            'regime': regime_info['regime'] if regime_info else 'UNKNOWN',
            'swing_bias': regime_info['swing_bias'] if regime_info else 'UNKNOWN',
            'trend': regime_info['trend'] if regime_info else 'UNKNOWN',
            'phase0': regime_info['phase0'] if regime_info else None,
            'price': regime_info['price'] if regime_info else None,
            'atr_1h': regime_info['atr_1h'] if regime_info else None,
            'trend_30d': regime_info['trend_30d'] if regime_info else None,
        }
        row.update(session_rets)
        results.append(row)

    if not results:
        print("No results — check data coverage.")
        return

    df = pd.DataFrame(results)

    # ══════════════════════════════════════════════════════════════
    # ANALYSIS
    # ══════════════════════════════════════════════════════════════

    print("═" * 70)
    print("  CAIXIN PMI SESSION TRANSMISSION BACKTEST")
    method = f"σ-BASED (σ={sigma})" if sigma else "SIMPLE MoM"
    print(f"  Classification: {method}")
    print("═" * 70)
    print(f"\n  Events analyzed: {len(df)}")
    print(f"  Date range: {df['date'].iloc[0]} → {df['date'].iloc[-1]}")

    # ── 1. Session Returns by Surprise Type ──
    print(f"\n  {'─' * 66}")
    print(f"  SESSION RETURNS BY SURPRISE TYPE")
    print(f"  {'─' * 66}")

    for surprise_type in ['STRONG_BEAT', 'BEAT', 'INLINE', 'MISS', 'BIG_MISS']:
        subset = df[df['surprise'] == surprise_type]
        if len(subset) == 0:
            continue

        print(f"\n  {surprise_type} (n={len(subset)}):")
        for session in ['asia_release', 'europe', 'us', 'asia_reopen', 'total_24h', 'total_48h']:
            col = (session, 'return_pct') if isinstance(df.columns[0], tuple) else f'{session}'
            # Check if session columns exist
            rets = []
            for _, r in subset.iterrows():
                if session in r and isinstance(r[session], dict):
                    rets.append(r[session].get('return_pct', 0))
                elif f'{session}' in r and isinstance(r[f'{session}'], dict):
                    rets.append(r[f'{session}'].get('return_pct', 0))

            if not rets:
                continue

            avg = np.mean(rets)
            med = np.median(rets)
            win = sum(1 for r in rets if r > 0) / len(rets) * 100
            print(f"    {session:16s}  avg={avg:+.3f}%  med={med:+.3f}%  win={win:.0f}%")

    # ── 2. Session Returns by Divergence ──
    print(f"\n  {'─' * 66}")
    print(f"  SESSION RETURNS BY CAIXIN/NBS DIVERGENCE")
    print(f"  {'─' * 66}")

    for div_type in ['ALIGNED', 'CAIXIN_HOT_NBS_COLD', 'CAIXIN_COLD_NBS_HOT',
                     'CAIXIN_SLIGHT_HOT', 'CAIXIN_SLIGHT_COLD']:
        subset = df[df['divergence'] == div_type]
        if len(subset) == 0:
            continue

        print(f"\n  {div_type} (n={len(subset)}):")
        # Show the key question: does divergence at Asia re-open cause reversal?
        for session in ['asia_release', 'europe', 'us', 'asia_reopen']:
            rets = []
            for _, r in subset.iterrows():
                if session in r and isinstance(r[session], dict):
                    rets.append(r[session].get('return_pct', 0))

            if not rets:
                continue

            avg = np.mean(rets)
            med = np.median(rets)
            win = sum(1 for r in rets if r > 0) / len(rets) * 100
            print(f"    {session:16s}  avg={avg:+.3f}%  med={med:+.3f}%  win={win:.0f}%")

        # Check if Asia re-open reverses the prior direction
        prior_rets = []
        reopen_rets = []
        for _, r in subset.iterrows():
            us_ret = r.get('us', {})
            asia_ret = r.get('asia_reopen', {})
            if isinstance(us_ret, dict) and isinstance(asia_ret, dict):
                us_r = us_ret.get('return_pct', 0)
                asia_r = asia_ret.get('return_pct', 0)
                if us_r != 0:
                    prior_rets.append(us_r)
                    reopen_rets.append(asia_r)

        if prior_rets and reopen_rets:
            reversals = sum(1 for p, r in zip(prior_rets, reopen_rets) if (p > 0 and r < 0) or (p < 0 and r > 0))
            print(f"    Reversal rate (US→Asia re-open): {reversals}/{len(prior_rets)} = {reversals/len(prior_rets)*100:.0f}%")

    # ── 3. Regime Breakdown (M9 + Trend + Phase0) ──
    print(f"\n  {'─' * 66}")
    print(f"  SESSION RETURNS BY M9 VOLATILITY REGIME")
    print(f"  {'─' * 66}")

    regime_order = ['TRENDING', 'COMPRESSING', 'NEUTRAL', 'CHOP_MILD', 'CHOP_HARD', 'CRISIS', 'UNKNOWN']
    for regime in regime_order:
        subset = df[df['regime'] == regime]
        if len(subset) == 0:
            continue

        print(f"\n  Regime: {regime} (n={len(subset)}):")
        for session in ['asia_release', 'europe', 'us', 'asia_reopen', 'total_24h']:
            rets = []
            for _, r in subset.iterrows():
                if session in r and isinstance(r[session], dict):
                    rets.append(r[session].get('return_pct', 0))

            if not rets:
                continue

            avg = np.mean(rets)
            med = np.median(rets)
            win = sum(1 for r in rets if r > 0) / len(rets) * 100
            print(f"    {session:16s}  avg={avg:+.3f}%  med={med:+.3f}%  win={win:.0f}%")

        # Surprise breakdown within regime
        for surprise_type in ['STRONG_BEAT', 'BEAT', 'INLINE', 'MISS', 'BIG_MISS']:
            s = subset[subset['surprise'] == surprise_type]
            if len(s) == 0:
                continue
            t24 = [r.get('total_24h', {}).get('return_pct', 0) for _, r in s.iterrows() if isinstance(r.get('total_24h'), dict)]
            if t24:
                print(f"    {surprise_type:16s}  24h={np.mean(t24):+.3f}%  win={sum(1 for x in t24 if x > 0)/len(t24)*100:.0f}%  (n={len(t24)})")

    # ── 3b. Regime × Trend 30d ──
    print(f"\n  {'─' * 66}")
    print(f"  SESSION RETURNS BY 30-DAY TREND (entering event)")
    print(f"  {'─' * 66}")

    # Split into trend buckets
    df['trend_bucket'] = pd.cut(df['trend_30d'].fillna(0),
                                 bins=[-100, -10, -5, -2, 2, 5, 10, 100],
                                 labels=['STRONG_DOWN', 'DOWN', 'SLIGHT_DOWN', 'FLAT', 'SLIGHT_UP', 'UP', 'STRONG_UP'])
    for bucket in ['STRONG_DOWN', 'DOWN', 'SLIGHT_DOWN', 'FLAT', 'SLIGHT_UP', 'UP', 'STRONG_UP']:
        subset = df[df['trend_bucket'] == bucket]
        if len(subset) == 0:
            continue
        t24 = [r.get('total_24h', {}).get('return_pct', 0) for _, r in subset.iterrows() if isinstance(r.get('total_24h'), dict)]
        if t24:
            print(f"  {bucket:16s}  avg={np.mean(t24):+.3f}%  win={sum(1 for x in t24 if x > 0)/len(t24)*100:.0f}%  (n={len(t24)})")

    # ── 3c. Regime × Phase0 ──
    print(f"\n  {'─' * 66}")
    print(f"  SESSION RETURNS BY PHASE0 (macro context)")
    print(f"  {'─' * 66}")

    df['phase0_bucket'] = pd.cut(df['phase0'].fillna(0.5),
                                  bins=[0, 0.15, 0.30, 0.50, 0.70, 1.0],
                                  labels=['DEATH_ZONE', 'LOW', 'NEUTRAL', 'STRONG', 'EXTREME'])
    for bucket in ['DEATH_ZONE', 'LOW', 'NEUTRAL', 'STRONG', 'EXTREME']:
        subset = df[df['phase0_bucket'] == bucket]
        if len(subset) == 0:
            continue
        t24 = [r.get('total_24h', {}).get('return_pct', 0) for _, r in subset.iterrows() if isinstance(r.get('total_24h'), dict)]
        if t24:
            print(f"  {bucket:16s}  avg={np.mean(t24):+.3f}%  win={sum(1 for x in t24 if x > 0)/len(t24)*100:.0f}%  (n={len(t24)})")

    # ── 3d. Regime × Surprise Cross-Tab ──
    print(f"\n  {'─' * 66}")
    print(f"  CROSS-TAB: REGIME × SURPRISE (24h return)")
    print(f"  {'─' * 66}")

    for regime in regime_order:
        subset = df[df['regime'] == regime]
        if len(subset) == 0:
            continue
        parts = []
        for surprise_type in ['STRONG_BEAT', 'BEAT', 'INLINE', 'MISS', 'BIG_MISS']:
            s = subset[subset['surprise'] == surprise_type]
            if len(s) == 0:
                continue
            t24 = [r.get('total_24h', {}).get('return_pct', 0) for _, r in s.iterrows() if isinstance(r.get('total_24h'), dict)]
            if t24:
                parts.append(f"{surprise_type[:6]}={np.mean(t24):+.1f}%({len(t24)})")
        if parts:
            print(f"  {regime:14s}  {'  '.join(parts)}")

    # ── 4. Full Event Table ──
    if args.verbose:
        print(f"\n  {'─' * 66}")
        print(f"  FULL EVENT TABLE")
        print(f"  {'─' * 66}")
        print(f"  {'Date':12s} {'Caixin':>7s} {'Surprise':>12s} {'NBS':>5s} {'Divergence':>22s} {'Regime':>14s} {'Asia':>7s} {'EU':>7s} {'US':>7s} {'AsiaR':>7s} {'24h':>7s}")
        print(f"  {'─'*12} {'─'*7} {'─'*12} {'─'*5} {'─'*22} {'─'*14} {'─'*7} {'─'*7} {'─'*7} {'─'*7} {'─'*7}")

        for _, r in df.iterrows():
            def get_ret(session):
                d = r.get(session, {})
                if isinstance(d, dict):
                    return f"{d.get('return_pct', 0):+.2f}%"
                return '—'

            nbs_str = f"{r.get('nbs', 0):.1f}" if r.get('nbs') and not pd.isna(r.get('nbs', None)) else '—'
            print(f"  {r['date']:12s} {r['caixin']:>7.1f} {r['surprise']:>12s} {nbs_str:>5s} "
                  f"{r['divergence']:>22s} {r['regime']:>14s} "
                  f"{get_ret('asia_release'):>7s} {get_ret('europe'):>7s} {get_ret('us'):>7s} "
                  f"{get_ret('asia_reopen'):>7s} {get_ret('total_24h'):>7s}")

    # ── 5. Key Findings ──
    print(f"\n  {'═' * 66}")
    print(f"  KEY FINDINGS")
    print(f"  {'═' * 66}")

    # Does Caixin beat → positive Asia session?
    beats = df[df['surprise'].isin(['STRONG_BEAT', 'BEAT'])]
    misses = df[df['surprise'].isin(['BIG_MISS', 'MISS'])]

    if len(beats) > 0 and len(misses) > 0:
        beat_asia = [r.get('asia_release', {}).get('return_pct', 0) for _, r in beats.iterrows() if isinstance(r.get('asia_release'), dict)]
        miss_asia = [r.get('asia_release', {}).get('return_pct', 0) for _, r in misses.iterrows() if isinstance(r.get('asia_release'), dict)]

        if beat_asia and miss_asia:
            print(f"\n  1. CAIXIN SURPRISE → ASIA SESSION:")
            print(f"     Beat avg Asia return:    {np.mean(beat_asia):+.3f}% (n={len(beat_asia)})")
            print(f"     Miss avg Asia return:    {np.mean(miss_asia):+.3f}% (n={len(miss_asia)})")
            print(f"     Delta:                   {np.mean(beat_asia) - np.mean(miss_asia):+.3f}%")

    # Does Europe inherit Asia direction?
    print(f"\n  2. SESSION INHERITANCE (does Europe continue Asia?):")
    asia_eu_agree = 0
    asia_eu_total = 0
    for _, r in df.iterrows():
        asia = r.get('asia_release', {})
        eu = r.get('europe', {})
        if isinstance(asia, dict) and isinstance(eu, dict):
            a_ret = asia.get('return_pct', 0)
            e_ret = eu.get('return_pct', 0)
            if a_ret != 0 and e_ret != 0:
                asia_eu_total += 1
                if (a_ret > 0 and e_ret > 0) or (a_ret < 0 and e_ret < 0):
                    asia_eu_agree += 1

    if asia_eu_total > 0:
        print(f"     Asia→EU agreement:       {asia_eu_agree}/{asia_eu_total} = {asia_eu_agree/asia_eu_total*100:.0f}%")

    # Does NBS divergence cause reversal at Asia re-open?
    print(f"\n  3. NBS DIVERGENCE → ASIA RE-OPEN REVERSAL:")
    aligned = df[df['divergence'] == 'ALIGNED']
    divergent = df[df['divergence'].isin(['CAIXIN_HOT_NBS_COLD', 'CAIXIN_COLD_NBS_HOT'])]

    for label, subset in [('Aligned', aligned), ('Divergent', divergent)]:
        if len(subset) == 0:
            continue
        us_rets = []
        reopen_rets = []
        for _, r in subset.iterrows():
            us = r.get('us', {})
            reopen = r.get('asia_reopen', {})
            if isinstance(us, dict) and isinstance(reopen, dict):
                u = us.get('return_pct', 0)
                a = reopen.get('return_pct', 0)
                if u != 0:
                    us_rets.append(u)
                    reopen_rets.append(a)

        if us_rets:
            reversals = sum(1 for p, r in zip(us_rets, reopen_rets) if (p > 0 and r < 0) or (p < 0 and r > 0))
            avg_us = np.mean(us_rets)
            avg_reopen = np.mean(reopen_rets)
            print(f"     {label:10s} — US avg: {avg_us:+.3f}%, Asia re-open avg: {avg_reopen:+.3f}%, "
                  f"reversal: {reversals}/{len(us_rets)} = {reversals/len(us_rets)*100:.0f}%")

    # ── 4. Regime-specific best strategy ──
    print(f"\n  4. BEST STRATEGY BY REGIME (24h return):")
    for regime in regime_order:
        subset = df[df['regime'] == regime]
        if len(subset) < 3:
            continue
        best_surprise = None
        best_avg = -999
        worst_surprise = None
        worst_avg = 999
        for surprise_type in ['STRONG_BEAT', 'BEAT', 'INLINE', 'MISS', 'BIG_MISS']:
            s = subset[subset['surprise'] == surprise_type]
            if len(s) < 2:
                continue
            t24 = [r.get('total_24h', {}).get('return_pct', 0) for _, r in s.iterrows() if isinstance(r.get('total_24h'), dict)]
            if t24:
                avg = np.mean(t24)
                if avg > best_avg:
                    best_avg = avg
                    best_surprise = surprise_type
                if avg < worst_avg:
                    worst_avg = avg
                    worst_surprise = surprise_type
        if best_surprise and worst_surprise:
            print(f"     {regime:14s}  BUY on {best_surprise:12s} ({best_avg:+.1f}%)  SELL on {worst_surprise:12s} ({worst_avg:+.1f}%)")

    # ── 5. Trend context ──
    print(f"\n  5. TREND CONTEXT (30d price trend entering event):")
    for bucket in ['STRONG_DOWN', 'DOWN', 'SLIGHT_DOWN', 'FLAT', 'SLIGHT_UP', 'UP', 'STRONG_UP']:
        subset = df[df['trend_bucket'] == bucket]
        if len(subset) == 0:
            continue
        t24 = [r.get('total_24h', {}).get('return_pct', 0) for _, r in subset.iterrows() if isinstance(r.get('total_24h'), dict)]
        if t24:
            print(f"     {bucket:16s}  avg={np.mean(t24):+.3f}%  win={sum(1 for x in t24 if x > 0)/len(t24)*100:.0f}%  (n={len(t24)})")

    # ── 6. Phase0 context ──
    print(f"\n  6. PHASE0 CONTEXT (macro momentum):")
    for bucket in ['DEATH_ZONE', 'LOW', 'NEUTRAL', 'STRONG', 'EXTREME']:
        subset = df[df['phase0_bucket'] == bucket]
        if len(subset) == 0:
            continue
        t24 = [r.get('total_24h', {}).get('return_pct', 0) for _, r in subset.iterrows() if isinstance(r.get('total_24h'), dict)]
        if t24:
            print(f"     {bucket:16s}  avg={np.mean(t24):+.3f}%  win={sum(1 for x in t24 if x > 0)/len(t24)*100:.0f}%  (n={len(t24)})")

    for label, subset in [('Aligned', aligned), ('Divergent', divergent)]:
        if len(subset) == 0:
            continue
        us_rets = []
        reopen_rets = []
        for _, r in subset.iterrows():
            us = r.get('us', {})
            reopen = r.get('asia_reopen', {})
            if isinstance(us, dict) and isinstance(reopen, dict):
                u = us.get('return_pct', 0)
                a = reopen.get('return_pct', 0)
                if u != 0:
                    us_rets.append(u)
                    reopen_rets.append(a)

        if us_rets:
            reversals = sum(1 for p, r in zip(us_rets, reopen_rets) if (p > 0 and r < 0) or (p < 0 and r > 0))
            avg_us = np.mean(us_rets)
            avg_reopen = np.mean(reopen_rets)
            print(f"     {label:10s} — US avg: {avg_us:+.3f}%, Asia re-open avg: {avg_reopen:+.3f}%, "
                  f"reversal: {reversals}/{len(us_rets)} = {reversals/len(us_rets)*100:.0f}%")

    print()


if __name__ == '__main__':
    main()

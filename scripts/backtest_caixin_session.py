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
    # (date, actual, previous, consensus_if_available)
    ('2024-01-02', 50.8, 50.7),   # Beat
    ('2024-02-01', 50.8, 50.8),   # Inline
    ('2024-03-01', 50.9, 50.8),   # Slight beat
    ('2024-04-01', 51.1, 50.9),   # Beat
    ('2024-05-02', 51.4, 51.1),   # Beat
    ('2024-06-03', 51.7, 51.4),   # Beat
    ('2024-07-01', 51.8, 51.7),   # Slight beat
    ('2024-08-01', 49.8, 51.8),   # MISS (big miss, contraction)
    ('2024-09-02', 50.4, 49.8),   # Beat (recovery)
    ('2024-10-08', 50.3, 50.4),   # Slight miss
    ('2024-11-01', 50.3, 50.3),   # Inline
    ('2024-12-02', 51.5, 50.3),   # Beat
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
    ('2026-01-02', 50.5, 51.2),   # Miss
    ('2026-02-02', 50.8, 50.5),   # Beat
    ('2026-03-02', 51.1, 50.8),   # Beat
    ('2026-04-01', 51.2, 51.1),   # Slight beat
    ('2026-05-01', 50.7, 51.2),   # Miss
]

# NBS PMI (same day as Caixin, ~01:00 UTC, released slightly before)
# These are the official state survey numbers
NBS_RELEASES = [
    ('2024-01-02', 49.0, 49.0),
    ('2024-02-01', 49.1, 49.0),
    ('2024-03-01', 49.1, 49.1),
    ('2024-04-01', 50.8, 49.1),   # Big beat (returned to expansion)
    ('2024-05-02', 49.5, 50.8),   # Miss (back to contraction)
    ('2024-06-03', 49.5, 49.5),
    ('2024-07-01', 49.5, 49.5),
    ('2024-08-01', 49.4, 49.5),
    ('2024-09-02', 49.8, 49.4),   # Slight beat
    ('2024-10-08', 50.1, 49.8),   # Beat (expansion)
    ('2024-11-01', 50.3, 50.1),   # Beat
    ('2024-12-02', 50.1, 50.3),   # Slight miss
    ('2025-01-02', 50.1, 50.1),
    ('2025-02-03', 50.2, 50.1),   # Beat
    ('2025-03-03', 50.5, 50.2),   # Beat
    ('2025-04-01', 50.5, 50.5),
    ('2025-05-02', 49.4, 50.5),   # Miss
    ('2025-06-02', 49.5, 49.4),   # Slight beat
    ('2025-07-01', 49.5, 49.5),
    ('2025-08-01', 49.3, 49.5),   # Miss
    ('2025-09-01', 49.8, 49.3),   # Beat
    ('2025-10-09', 50.2, 49.8),   # Beat
    ('2025-11-03', 49.5, 50.2),   # Miss
    ('2025-12-01', 50.0, 49.5),   # Beat
    ('2026-01-02', 49.8, 50.0),   # Miss
    ('2026-02-02', 50.2, 49.8),   # Beat
    ('2026-03-02', 50.5, 50.2),   # Beat
    ('2026-04-01', 50.5, 50.5),
    ('2026-05-01', 49.4, 50.5),   # Miss
]


def classify_surprise(actual, previous):
    """Classify PMI surprise vs previous."""
    diff = actual - previous
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
    """Get market regime at a specific date."""
    from src.modules.m9_volatility import RegimeState, compute_vol_regime

    ts = pd.Timestamp(date_str)

    # Find closest 15m bar
    mask = df_15m['Open time'] >= ts
    if not mask.any():
        return None
    idx = mask.idxmax()
    if idx < 50:
        return None

    # Find closest 1h bar
    mask_1h = df_1h['Open time'] >= ts
    if not mask_1h.any():
        return None
    idx_1h = mask_1h.idxmax()

    # Find closest 1d bar
    mask_1d = df_1d['Open time'] >= ts
    if not mask_1d.any():
        return None
    idx_1d = mask_1d.idxmax()

    cfg = config or {}
    regime_state = RegimeState(config=cfg)
    try:
        vol_regime, m9_raw, _ = compute_vol_regime(
            df_15m, df_1h, idx, idx_1h, regime_state=regime_state, config=cfg)
    except Exception:
        vol_regime, m9_raw = 'UNKNOWN', 0.5

    swing_bias = df_1d['swing_bias'].iloc[idx_1d] if 'swing_bias' in df_1d.columns else 'UNKNOWN'
    phase0 = df_1d['phase0'].iloc[idx_1d] if 'phase0' in df_1d.columns else None
    trend = df_1d['trend'].iloc[idx_1d] if 'trend' in df_1d.columns else 'UNKNOWN'

    return {
        'regime': vol_regime,
        'm9_raw': round(float(m9_raw), 3) if m9_raw else None,
        'swing_bias': swing_bias,
        'phase0': round(float(phase0), 3) if phase0 and not pd.isna(phase0) else None,
        'trend': trend,
    }


def main():
    parser = argparse.ArgumentParser(description='Caixin PMI Session Transmission Backtest')
    parser.add_argument('--verbose', '-v', action='store_true')
    args = parser.parse_args()

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
        surprise = classify_surprise(caixin_actual, caixin_prev)
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

    # ── 3. Regime Breakdown ──
    print(f"\n  {'─' * 66}")
    print(f"  SESSION RETURNS BY REGIME")
    print(f"  {'─' * 66}")

    for regime in df['regime'].unique():
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
            win = sum(1 for r in rets if r > 0) / len(rets) * 100
            print(f"    {session:16s}  avg={avg:+.3f}%  win={win:.0f}%")

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

    print()


if __name__ == '__main__':
    main()

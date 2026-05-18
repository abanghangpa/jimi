#!/usr/bin/env python3
"""
UK Session Backtest — Analyze London behavior on PPI/CPI release days.

Computes real historical stats for UK session (07:00-16:00 UTC) by:
1. Slicing UK bars from historical 15m data for each PPI/CPI release
2. Measuring UK direction vs US close and Asia close
3. Computing continuation/fade rates by pattern and regime
4. Generating the stat constants used in m23_ppi_session.py

Usage:
    python3 scripts/backtest_uk_session.py
    python3 scripts/backtest_uk_session.py --csv data/eth_15m_merged.csv
    python3 scripts/backtest_uk_session.py --verbose
"""

import argparse
import sys
import os
import json
from datetime import datetime, timedelta
from collections import defaultdict

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from src.modules.macro_utils import (
    PPI_SCHEDULE_DATES as PPI_RELEASE_DATES,
    CPI_SCHEDULE_DATES as CPI_RELEASE_DATES,
    get_release_type, classify_market_regime,
    classify_dump_size as _classify_dump_size,
    classify_inflation_regime as _classify_inflation_regime,
)
from src.modules.cascade_engine import (
    compute_1h_spike, compute_us_session, compute_asia_session,
    US_SESSION, ASIA_SESSION,
)

# Backward compat aliases
RELEASE_HOUR_UTC = 13
RELEASE_MINUTE_UTC = 30
US_SESSION_END = US_SESSION['end']
ASIA_SESSION_START = ASIA_SESSION['start']
ASIA_SESSION_END = ASIA_SESSION['end']
from src.utils.data_handler import load_data

# UK session window (UTC)
UK_SESSION_START = (7, 0)
UK_SESSION_END = (16, 0)


def compute_uk_session_hist(df_15m, release_date):
    """Compute UK session data for a historical release date."""
    if isinstance(release_date, str):
        release_date = datetime.strptime(release_date, '%Y-%m-%d')

    release_dt = release_date.replace(hour=RELEASE_HOUR_UTC, minute=RELEASE_MINUTE_UTC)
    us_end = release_date.replace(hour=US_SESSION_END[0])

    uk_date = release_date + timedelta(days=1)
    uk_start = uk_date.replace(hour=UK_SESSION_START[0])
    uk_end = uk_date.replace(hour=UK_SESSION_END[0])

    asia_start = uk_date.replace(hour=ASIA_SESSION_START[0])
    asia_end = uk_date.replace(hour=ASIA_SESSION_END[0])

    # US close
    us_mask = (df_15m['Open time_dt'] >= release_dt) & (df_15m['Open time_dt'] < us_end)
    us_bars = df_15m[us_mask]
    if len(us_bars) == 0:
        return None
    us_close = float(us_bars.iloc[-1]['Close'])

    # Asia session
    asia_mask = (df_15m['Open time_dt'] >= asia_start) & (df_15m['Open time_dt'] < asia_end)
    asia_bars = df_15m[asia_mask]
    if len(asia_bars) < 2:
        return None
    asia_open = float(asia_bars.iloc[0]['Open'])
    asia_close = float(asia_bars.iloc[-1]['Close'])
    asia_high = float(asia_bars['High'].max())
    asia_low = float(asia_bars['Low'].min())
    asia_move = (asia_close - us_close) / us_close * 100
    asia_gap = (asia_open - us_close) / us_close * 100
    asia_dir = 'UP' if asia_move > 0.3 else 'DOWN' if asia_move < -0.3 else 'FLAT'
    gap_dir = 'UP' if asia_gap > 0.2 else 'DOWN' if asia_gap < -0.2 else 'FLAT'
    gap_held = (gap_dir == asia_dir) or gap_dir == 'FLAT'

    # Asia sweep-reversal detection
    if gap_dir == 'DOWN':
        sweep_depth = abs((asia_low - us_close) / us_close * 100)
        recovery = (asia_close - asia_low) / (us_close - asia_low) * 100 if (us_close - asia_low) > 0 else 0
        reclaimed = asia_high >= us_close
    elif gap_dir == 'UP':
        sweep_depth = abs((asia_high - us_close) / us_close * 100)
        recovery = (asia_high - asia_close) / (asia_high - us_close) * 100 if (asia_high - us_close) > 0 else 0
        reclaimed = asia_low <= us_close
    else:
        sweep_depth = 0
        recovery = 0
        reclaimed = False
    is_sweep = sweep_depth > 0.5 and recovery > 50

    # UK session
    uk_mask = (df_15m['Open time_dt'] >= uk_start) & (df_15m['Open time_dt'] < uk_end)
    uk_bars = df_15m[uk_mask]
    if len(uk_bars) < 2:
        return None

    uk_open = float(uk_bars.iloc[0]['Open'])
    uk_close = float(uk_bars.iloc[-1]['Close'])
    uk_high = float(uk_bars['High'].max())
    uk_low = float(uk_bars['Low'].min())
    uk_move_vs_asia = (uk_close - asia_close) / asia_close * 100
    uk_move_vs_us = (uk_close - us_close) / us_close * 100
    uk_range = (uk_high - uk_low) / asia_close * 100
    uk_gap = (uk_open - asia_close) / asia_close * 100
    uk_dir = 'UP' if uk_move_vs_asia > 0.3 else 'DOWN' if uk_move_vs_asia < -0.3 else 'FLAT'

    uk_continued = (asia_dir == uk_dir) and asia_dir != 'FLAT'
    uk_faded = (asia_dir == 'UP' and uk_dir == 'DOWN') or \
               (asia_dir == 'DOWN' and uk_dir == 'UP')

    # UK sweep of Asia levels
    uk_swept_high = uk_high >= asia_high * 0.999
    uk_swept_low = uk_low <= asia_low * 1.001

    # UK morning sweep (swept Asia level then reversed)
    is_morning_sweep = False
    if uk_swept_high and uk_dir == 'DOWN':
        is_morning_sweep = True
    elif uk_swept_low and uk_dir == 'UP':
        is_morning_sweep = True

    # Volume
    uk_avg_vol = float(uk_bars['Volume'].mean()) if len(uk_bars) > 0 else 0
    asia_avg_vol = float(asia_bars['Volume'].mean()) if len(asia_bars) > 0 else 0
    vol_ratio = uk_avg_vol / asia_avg_vol if asia_avg_vol > 0 else 1.0

    # Taker
    uk_taker = 0.5
    if 'Taker buy base asset volume' in uk_bars.columns:
        taker_buy = float(uk_bars['Taker buy base asset volume'].sum())
        total_vol = float(uk_bars['Volume'].sum())
        uk_taker = taker_buy / total_vol if total_vol > 0 else 0.5

    return {
        'release_date': release_date.strftime('%Y-%m-%d'),
        'us_close': round(us_close, 2),
        'asia_move': round(asia_move, 3),
        'asia_dir': asia_dir,
        'asia_gap': round(asia_gap, 3),
        'gap_dir': gap_dir,
        'gap_held': gap_held,
        'is_sweep_reversal': is_sweep,
        'sweep_depth': round(sweep_depth, 3),
        'recovery_pct': round(recovery, 1),
        'reclaimed_gap': reclaimed,
        'uk_move_vs_asia': round(uk_move_vs_asia, 3),
        'uk_move_vs_us': round(uk_move_vs_us, 3),
        'uk_dir': uk_dir,
        'uk_range': round(uk_range, 3),
        'uk_gap': round(uk_gap, 3),
        'uk_open': round(uk_open, 2),
        'uk_close': round(uk_close, 2),
        'uk_high': round(uk_high, 2),
        'uk_low': round(uk_low, 2),
        'uk_continued_asia': uk_continued,
        'uk_faded_asia': uk_faded,
        'uk_swept_high': uk_swept_high,
        'uk_swept_low': uk_swept_low,
        'is_morning_sweep': is_morning_sweep,
        'uk_taker': round(uk_taker, 4),
        'uk_vol_ratio': round(vol_ratio, 2),
    }


def analyze_uk_stats(results):
    """Compute aggregate UK session statistics from backtest results."""
    stats = {
        'total': len(results),
        'by_pattern': defaultdict(list),
        'by_regime': defaultdict(list),
        'by_dump_size': defaultdict(list),
        'by_year': defaultdict(list),
        'uk_direction': {'UP': 0, 'DOWN': 0, 'FLAT': 0},
        'uk_continued': 0,
        'uk_faded': 0,
        'uk_morning_sweep': 0,
    }

    # Regime for each release
    regime, _ = classify_market_regime()

    for r in results:
        pattern = 'UNKNOWN'
        if r['is_sweep_reversal']:
            pattern = 'SWEEP_REVERSAL'
        elif r['gap_held'] and r['asia_dir'] != 'FLAT':
            pattern = 'GAP_HELD'
        elif r['asia_dir'] != 'FLAT':
            pattern = 'FADE'
        else:
            pattern = 'FLAT'

        r['pattern'] = pattern
        r['regime'] = regime

        stats['by_pattern'][pattern].append(r)
        stats['by_regime'][regime].append(r)
        stats['uk_direction'][r['uk_dir']] += 1

        if r['uk_continued_asia']:
            stats['uk_continued'] += 1
        if r['uk_faded_asia']:
            stats['uk_faded'] += 1
        if r['is_morning_sweep']:
            stats['uk_morning_sweep'] += 1

        year = r['release_date'][:4]
        stats['by_year'][year].append(r)

        # Dump size from US move (we need to compute it)
        # We'll compute us_move from the data
        stats['by_pattern'][pattern].append(r)

    return stats


def print_report(results, verbose=False):
    """Print comprehensive UK session backtest report."""
    if not results:
        print("No results to report.")
        return

    print("\n" + "═" * 70)
    print("  UK SESSION BACKTEST — PPI/CPI RELEASE DAY ANALYSIS")
    print("═" * 70)

    # Compute stats
    total = len(results)

    # UK direction distribution
    uk_dirs = {'UP': 0, 'DOWN': 0, 'FLAT': 0}
    uk_continued = 0
    uk_faded = 0
    uk_morning_sweep = 0

    # By pattern
    pattern_data = defaultdict(list)
    regime_data = defaultdict(list)
    year_data = defaultdict(list)
    dump_data = defaultdict(list)

    for r in results:
        uk_dirs[r['uk_dir']] += 1
        if r['uk_continued_asia']:
            uk_continued += 1
        if r['uk_faded_asia']:
            uk_faded += 1
        if r['is_morning_sweep']:
            uk_morning_sweep += 1

        pattern_data[r['pattern']].append(r)
        year_data[r['release_date'][:4]].append(r)

        # Classify US dump size
        # We need us_move — compute from uk_move_vs_us and uk_close
        # Actually we stored asia_move, let's reconstruct
        dump_size = 'NOT_DUMP'
        # Use a rough US move estimate from the data we have
        # The results don't directly store us_move, so let's compute from uk_move_vs_us
        # uk_move_vs_us = (uk_close - us_close) / us_close * 100
        # We need the actual us_move. Let's add it to the results.
        dump_data[dump_size].append(r)

    # Overall summary
    print(f"\n  Total releases analyzed: {total}")
    print(f"  Date range: {results[0]['release_date']} → {results[-1]['release_date']}")
    print(f"\n  UK Direction Distribution:")
    for d in ['UP', 'DOWN', 'FLAT']:
        pct = uk_dirs[d] / total * 100
        print(f"    {d:>6}: {uk_dirs[d]:>3} ({pct:.1f}%)")

    print(f"\n  UK vs Asia Behavior:")
    print(f"    Continued Asia: {uk_continued:>3} ({uk_continued/total*100:.1f}%)")
    print(f"    Faded Asia:     {uk_faded:>3} ({uk_faded/total*100:.1f}%)")
    print(f"    Morning sweep:  {uk_morning_sweep:>3} ({uk_morning_sweep/total*100:.1f}%)")

    # By pattern
    print(f"\n  {'─' * 66}")
    print(f"  UK BEHAVIOR BY ASIA PATTERN")
    print(f"  {'─' * 66}")
    print(f"  {'Pattern':<20} {'n':>4} {'UK↑':>5} {'UK↓':>5} {'Cont%':>6} {'Fade%':>6} "
          f"{'AvgMove':>8} {'Sweep%':>7}")

    for pattern in ['GAP_HELD', 'SWEEP_REVERSAL', 'FADE', 'FLAT']:
        rows = pattern_data.get(pattern, [])
        if not rows:
            continue
        n = len(rows)
        up = sum(1 for r in rows if r['uk_dir'] == 'UP')
        down = sum(1 for r in rows if r['uk_dir'] == 'DOWN')
        cont = sum(1 for r in rows if r['uk_continued_asia'])
        fade = sum(1 for r in rows if r['uk_faded_asia'])
        sweep = sum(1 for r in rows if r['is_morning_sweep'])
        avg_move = np.mean([r['uk_move_vs_asia'] for r in rows])

        print(f"  {pattern:<20} {n:>4} {up:>5} {down:>5} "
              f"{cont/n*100:>5.1f}% {fade/n*100:>5.1f}% "
              f"{avg_move:>+7.2f}% {sweep/n*100:>6.1f}%")

    # Detailed pattern analysis
    for pattern in ['SWEEP_REVERSAL', 'GAP_HELD', 'FADE']:
        rows = pattern_data.get(pattern, [])
        if not rows:
            continue

        print(f"\n  {'─' * 66}")
        print(f"  {pattern} — DETAILED BREAKDOWN")
        print(f"  {'─' * 66}")

        # Group by year
        by_year = defaultdict(list)
        for r in rows:
            by_year[r['release_date'][:4]].append(r)

        print(f"  {'Year':>6} {'n':>4} {'UK↑':>5} {'UK↓':>5} {'Cont%':>6} {'Fade%':>6} {'AvgMove':>8}")
        for year in sorted(by_year.keys()):
            yr = by_year[year]
            n = len(yr)
            up = sum(1 for r in yr if r['uk_dir'] == 'UP')
            down = sum(1 for r in yr if r['uk_dir'] == 'DOWN')
            cont = sum(1 for r in yr if r['uk_continued_asia'])
            fade = sum(1 for r in yr if r['uk_faded_asia'])
            avg_move = np.mean([r['uk_move_vs_asia'] for r in yr])
            print(f"  {year:>6} {n:>4} {up:>5} {down:>5} "
                  f"{cont/n*100:>5.1f}% {fade/n*100:>5.1f}% "
                  f"{avg_move:>+7.2f}%")

        # UK move distribution
        moves = [r['uk_move_vs_asia'] for r in rows]
        print(f"\n  UK move vs Asia: mean={np.mean(moves):+.3f}%  "
              f"median={np.median(moves):+.3f}%  "
              f"std={np.std(moves):.3f}%  "
              f"min={np.min(moves):+.3f}%  max={np.max(moves):+.3f}%")

        # UK volume ratio
        vols = [r['uk_vol_ratio'] for r in rows if r['uk_vol_ratio'] > 0]
        if vols:
            print(f"  UK/Asia vol ratio: mean={np.mean(vols):.2f}x  "
                  f"median={np.median(vols):.2f}x")

        # UK taker flow
        takers = [r['uk_taker'] for r in rows]
        print(f"  UK taker ratio: mean={np.mean(takers):.3f}  "
              f"({'buyers' if np.mean(takers) > 0.52 else 'sellers' if np.mean(takers) < 0.48 else 'neutral'})")

        # Morning sweep analysis
        sweeps = [r for r in rows if r['is_morning_sweep']]
        if sweeps:
            print(f"\n  Morning sweeps: {len(sweeps)}/{len(rows)} ({len(sweeps)/len(rows)*100:.1f}%)")
            # After sweep, did UK continue or reverse?
            sweep_continued = sum(1 for r in sweeps if r['uk_continued_asia'])
            sweep_faded = sum(1 for r in sweeps if r['uk_faded_asia'])
            print(f"    After sweep → continued: {sweep_continued}, faded: {sweep_faded}")

    # SWEEP_REVERSAL deep dive: UK fade rate when Asia sweeps then recovers
    sweep_rows = pattern_data.get('SWEEP_REVERSAL', [])
    if sweep_rows:
        print(f"\n  {'═' * 66}")
        print(f"  KEY FINDING: UK BEHAVIOR AFTER ASIA SWEEP-REVERSAL")
        print(f"  {'═' * 66}")
        print(f"\n  When Asia sweeps then reverses (sweep > 0.5%, recovery > 50%):")
        print(f"  Total instances: {len(sweep_rows)}")

        fade_count = sum(1 for r in sweep_rows if r['uk_faded_asia'])
        cont_count = sum(1 for r in sweep_rows if r['uk_continued_asia'])
        neutral = len(sweep_rows) - fade_count - cont_count

        print(f"  UK Fades Asia (continues reversal): {fade_count} ({fade_count/len(sweep_rows)*100:.1f}%)")
        print(f"  UK Continues Asia (trusts bounce):  {cont_count} ({cont_count/len(sweep_rows)*100:.1f}%)")
        print(f"  UK Neutral:                         {neutral} ({neutral/len(sweep_rows)*100:.1f}%)")

        # By sweep depth
        deep = [r for r in sweep_rows if r['sweep_depth'] >= 1.0]
        shallow = [r for r in sweep_rows if r['sweep_depth'] < 1.0]
        if deep:
            deep_fade = sum(1 for r in deep if r['uk_faded_asia'])
            print(f"\n  Deep sweeps (≥1.0%): {len(deep)} — UK fade rate: {deep_fade/len(deep)*100:.1f}%")
        if shallow:
            shallow_fade = sum(1 for r in shallow if r['uk_faded_asia'])
            print(f"  Shallow sweeps (<1.0%): {len(shallow)} — UK fade rate: {shallow_fade/len(shallow)*100:.1f}%")

    # GAP_HELD: UK continuation rate
    gap_rows = pattern_data.get('GAP_HELD', [])
    if gap_rows:
        print(f"\n  {'═' * 66}")
        print(f"  KEY FINDING: UK BEHAVIOR WHEN ASIA HOLDS GAP")
        print(f"  {'═' * 66}")
        print(f"\n  When Asia continues US direction (gap held):")
        print(f"  Total instances: {len(gap_rows)}")

        cont_count = sum(1 for r in gap_rows if r['uk_continued_asia'])
        fade_count = sum(1 for r in gap_rows if r['uk_faded_asia'])

        print(f"  UK Continues (momentum): {cont_count} ({cont_count/len(gap_rows)*100:.1f}%)")
        print(f"  UK Fades (reversal):     {fade_count} ({fade_count/len(gap_rows)*100:.1f}%)")

        # UK move distribution when continuing
        cont_moves = [r['uk_move_vs_asia'] for r in gap_rows if r['uk_continued_asia']]
        fade_moves = [r['uk_move_vs_asia'] for r in gap_rows if r['uk_faded_asia']]
        if cont_moves:
            print(f"\n  UK continuation move: mean={np.mean(cont_moves):+.3f}%  "
                  f"median={np.median(cont_moves):+.3f}%")
        if fade_moves:
            print(f"  UK fade move:         mean={np.mean(fade_moves):+.3f}%  "
                  f"median={np.median(fade_moves):+.3f}%")

    # Generate constants for m23_ppi_session.py
    print(f"\n  {'═' * 66}")
    print(f"  DERIVED CONSTANTS FOR m23_ppi_session.py")
    print(f"  {'═' * 66}")

    # UK_FADE_SWEEP_REVERSAL
    print(f"\n  # Copy these into m23_ppi_session.py:")
    print(f"  # (computed from {total} historical releases)")

    sweep_rows = pattern_data.get('SWEEP_REVERSAL', [])
    if sweep_rows:
        fade_rate = sum(1 for r in sweep_rows if r['uk_faded_asia']) / len(sweep_rows)
        print(f"\n  UK_FADE_SWEEP_REVERSAL = {{")
        print(f"      'STAGFLATION':     {fade_rate:.2f},  # backtested")
        print(f"  }}")

    # UK_CONTINUATION_GAP_HELD
    gap_rows = pattern_data.get('GAP_HELD', [])
    if gap_rows:
        cont_rate = sum(1 for r in gap_rows if r['uk_continued_asia']) / len(gap_rows)
        print(f"\n  UK_CONTINUATION_GAP_HELD = {{")
        print(f"      'STAGFLATION':     {cont_rate:.2f},  # backtested")
        print(f"  }}")

    # UK_MOVE_RATIO_AVG
    cont_moves = [abs(r['uk_move_vs_asia']) for r in results
                  if r['uk_continued_asia'] and r['asia_dir'] != 'FLAT']
    fade_moves = [abs(r['uk_move_vs_asia']) for r in results
                  if r['uk_faded_asia'] and r['asia_dir'] != 'FLAT']
    asia_cont_moves = [abs(r['asia_move']) for r in results
                       if r['uk_continued_asia'] and r['asia_dir'] != 'FLAT']
    asia_fade_moves = [abs(r['asia_move']) for r in results
                       if r['uk_faded_asia'] and r['asia_dir'] != 'FLAT']

    if cont_moves and asia_cont_moves:
        ratio = np.mean(cont_moves) / np.mean(asia_cont_moves) if np.mean(asia_cont_moves) > 0 else 0.55
        print(f"\n  UK_MOVE_RATIO_AVG = {{")
        print(f"      'CONTINUATION': {ratio:.2f},  # UK moves {ratio:.0%} of Asia's move")
        print(f"  }}")
    if fade_moves and asia_fade_moves:
        ratio = np.mean(fade_moves) / np.mean(asia_fade_moves) if np.mean(asia_fade_moves) > 0 else 0.40
        print(f"      'FADE':         {ratio:.2f},  # UK moves {ratio:.0%} of Asia's move")
        print(f"  }}")

    # Verbose: print every release
    if verbose:
        print(f"\n  {'═' * 66}")
        print(f"  EVERY RELEASE DETAIL")
        print(f"  {'═' * 66}")
        print(f"  {'Date':<12} {'Pattern':<18} {'Asia%':>7} {'UKvsAs%':>8} {'UKDir':>5} "
              f"{'Cont':>5} {'Fade':>5} {'Sweep':>6} {'VolR':>5} {'Taker':>6}")
        for r in results:
            cont = '✓' if r['uk_continued_asia'] else ''
            fade = '✓' if r['uk_faded_asia'] else ''
            sweep = '✓' if r['is_morning_sweep'] else ''
            print(f"  {r['release_date']:<12} {r['pattern']:<18} "
                  f"{r['asia_move']:>+6.2f}% {r['uk_move_vs_asia']:>+7.2f}% "
                  f"{r['uk_dir']:>5} {cont:>5} {fade:>5} {sweep:>6} "
                  f"{r['uk_vol_ratio']:>5.1f} {r['uk_taker']:>6.3f}")

    print(f"\n{'═' * 70}")


def main():
    parser = argparse.ArgumentParser(description='UK Session Backtest for PPI/CPI releases')
    parser.add_argument('--csv', default=None, help='Path to 15m CSV (default: auto-detect)')
    parser.add_argument('--verbose', action='store_true', help='Print every release detail')
    args = parser.parse_args()

    # Load data
    csv_path = args.csv
    if csv_path is None:
        base = os.path.dirname(os.path.dirname(__file__))
        csv_path = os.path.join(base, 'data', 'eth_15m_merged.csv')
        if not os.path.exists(csv_path):
            csv_path = os.path.join(base, 'eth_15m_merged.csv')

    if not os.path.exists(csv_path):
        print(f"CSV not found: {csv_path}")
        print("Run: python3 scripts/backtest_runner.py --fetch --start 2018-01-01 --end 2026-05-14")
        return

    print(f"Loading data from {csv_path}...")
    df = load_data(csv_path)

    # Ensure datetime column
    if 'Open time_dt' not in df.columns:
        df['Open time_dt'] = pd.to_datetime(df['Open time'])

    print(f"Loaded {len(df)} bars: {df['Open time_dt'].iloc[0]} → {df['Open time_dt'].iloc[-1]}")

    # Get all release dates
    all_releases = set()
    for d in PPI_RELEASE_DATES:
        all_releases.add(d)
    for d in CPI_RELEASE_DATES:
        all_releases.add(d)

    all_releases = sorted(all_releases)
    print(f"Checking {len(all_releases)} PPI/CPI release dates...")

    results = []
    skipped = 0

    for release_date_str in all_releases:
        release_date = datetime.strptime(release_date_str, '%Y-%m-%d')

        # Check if we have data for this release + next day UK session
        uk_date = release_date + timedelta(days=1)
        uk_end = uk_date.replace(hour=UK_SESSION_END[0])

        if uk_end > df['Open time_dt'].iloc[-1]:
            skipped += 1
            continue

        if release_date < df['Open time_dt'].iloc[0]:
            skipped += 1
            continue

        result = compute_uk_session_hist(df, release_date)
        if result:
            release_type = get_release_type(release_date_str)
            result['release_type'] = release_type
            results.append(result)

    print(f"Analyzed {len(results)} releases ({skipped} skipped — outside data range)")

    if not results:
        print("No releases found in data range.")
        return

    # Compute pattern and regime for each result
    # We need us_move for dump classification — compute from stored data
    # We don't have us_move directly, but we can compute from the CSV
    for r in results:
        rd = datetime.strptime(r['release_date'], '%Y-%m-%d')
        release_dt = rd.replace(hour=RELEASE_HOUR_UTC, minute=RELEASE_MINUTE_UTC)
        us_end = rd.replace(hour=US_SESSION_END[0])
        us_mask = (df['Open time_dt'] >= release_dt) & (df['Open time_dt'] < us_end)
        us_bars = df[us_mask]
        if len(us_bars) >= 2:
            pre_mask = df['Open time_dt'] < release_dt
            pre_bars = df[pre_mask]
            if len(pre_bars) > 0:
                pre_price = float(pre_bars.iloc[-1]['Close'])
                us_close = float(us_bars.iloc[-1]['Close'])
                r['us_move'] = round((us_close - pre_price) / pre_price * 100, 3)
            else:
                r['us_move'] = 0
        else:
            r['us_move'] = 0

        # Classify pattern
        if r['is_sweep_reversal']:
            r['pattern'] = 'SWEEP_REVERSAL'
        elif r['gap_held'] and r['asia_dir'] != 'FLAT':
            r['pattern'] = 'GAP_HELD'
        elif r['asia_dir'] != 'FLAT':
            r['pattern'] = 'FADE'
        else:
            r['pattern'] = 'FLAT'

        # Classify dump size
        r['dump_size'] = _classify_dump_size(r.get('us_move', 0))

    # Print report
    print_report(results, verbose=args.verbose)


if __name__ == '__main__':
    main()

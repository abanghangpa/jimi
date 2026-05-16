#!/usr/bin/env python3
"""
Prompt B: Validate China CPI+PPI session transmission chain.

Tests whether China CPI+PPI causes a session-by-session transmission chain in ETH.
Measures direction persistence between sessions on release days.

Thresholds:
  <55% same direction: chain doesn't hold
  >55%: marginal
  >65%: real edge

Also tests: 24h aggregate significance, miss vs beat.
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from scipy import stats
import json
import os

# Import releases from backtest script
from backtest_china_cpi_ppi import RELEASES, load_eth_data, get_bar_at, get_bars_between, get_return

# Session definitions for chain testing
# Adjusted for China CPI release at 01:30 UTC
SESSION_PHASES = [
    ('Pre-Asia', 0, 1),           # 00:00-01:30
    ('Release Spike', 1, 2),      # 01:30-02:00 (first 30min)
    ('Sydney Open', 1, 2),        # 01:30-02:00
    ('Tokyo Open', 2, 3),         # 02:00-03:00
    ('Asia Mid', 3, 5),           # 03:00-05:00
    ('Asia Afternoon', 5, 6),     # 05:00-06:00
    ('Tokyo Close', 6, 7),        # 06:00-07:00
    ('Pre-London', 7, 8),         # 07:00-08:00
    ('Frankfurt Open', 7, 8),     # 07:00-08:00
    ('London Open', 8, 9),        # 08:00-09:00
    ('London Morning', 9, 12),    # 09:00-12:00
    ('London Midday', 12, 14),    # 12:00-14:00
    ('NY Pre-Open', 12, 13),      # 12:00-13:30
    ('NY Open', 13, 14),          # 13:30-14:00
    ('London-NY Overlap', 14, 16),# 14:00-16:00
    ('NY AM', 14, 17),            # 14:00-17:00
    ('NY Lunch', 17, 18),         # 17:00-18:00
    ('NY PM', 18, 21),            # 18:00-21:00
]


def compute_session_returns(df, release_date):
    """Compute returns for each session phase on release day."""
    release_dt = datetime.strptime(release_date, '%Y-%m-%d').replace(hour=1, minute=30)

    # Pre-release price
    pre_bar = get_bar_at(df, release_dt)
    if pre_bar is None:
        return None
    pre_price = float(pre_bar['Close'])

    session_rets = {}
    for name, start_h, end_h in SESSION_PHASES:
        start_dt = release_dt.replace(hour=start_h, minute=30 if start_h == 1 else 0)
        end_dt = release_dt.replace(hour=end_h, minute=0)
        if end_dt <= start_dt:
            end_dt += timedelta(days=1)

        bars = get_bars_between(df, start_dt, end_dt)
        if len(bars) < 1:
            continue

        start_bar = get_bar_at(df, start_dt)
        if start_bar is None:
            continue

        start_price = float(start_bar['Close'])
        end_price = float(bars.iloc[-1]['Close'])
        ret = (end_price - start_price) / start_price * 100
        session_rets[name] = round(ret, 4)

    # Also compute cumulative from release
    for name, start_h, end_h in SESSION_PHASES:
        start_dt = release_dt.replace(hour=start_h, minute=30 if start_h == 1 else 0)
        end_dt = release_dt.replace(hour=end_h, minute=0)
        if end_dt <= start_dt:
            end_dt += timedelta(days=1)

        bars = get_bars_between(df, start_dt, end_dt)
        if len(bars) < 1:
            continue
        end_price = float(bars.iloc[-1]['Close'])
        cum_ret = (end_price - pre_price) / pre_price * 100
        session_rets[f'{name}_cum'] = round(cum_ret, 4)

    return session_rets


def test_chain_persistence(df):
    """Test direction persistence between consecutive sessions."""
    print("="*80)
    print("SESSION TRANSMISSION CHAIN VALIDATION")
    print("="*80)

    all_session_data = []
    for date_str in sorted(RELEASES.keys()):
        session_rets = compute_session_returns(df, date_str)
        if session_rets:
            all_session_data.append({
                'date': date_str,
                'sessions': session_rets,
            })

    if not all_session_data:
        print("No data available")
        return

    n = len(all_session_data)
    print(f"\nAnalyzed {n} releases")

    # For each consecutive pair of sessions, test direction persistence
    session_names = [s[0] for s in SESSION_PHASES]

    print(f"\n{'Transition':<45} {'Same Dir%':>10} {'n':>5} {'Status':>12}")
    print("-"*75)

    chain_edges = []
    for i in range(len(session_names) - 1):
        s1 = session_names[i]
        s2 = session_names[i + 1]

        same_count = 0
        total = 0
        for data in all_session_data:
            sr = data['sessions']
            if s1 in sr and s2 in sr:
                r1 = sr[s1]
                r2 = sr[s2]
                # Same direction if both >0.05% or both <-0.05%
                if (r1 > 0.05 and r2 > 0.05) or (r1 < -0.05 and r2 < -0.05):
                    same_count += 1
                total += 1

        if total < 5:
            continue

        same_pct = same_count / total * 100

        if same_pct >= 65:
            status = '✅ REAL EDGE'
        elif same_pct >= 55:
            status = '🟡 MARGINAL'
        else:
            status = '❌ NO CHAIN'

        print(f"  {s1} → {s2:<25} {same_pct:>9.1f}% {total:>5} {status:>12}")

        chain_edges.append({
            'from': s1,
            'to': s2,
            'same_dir_pct': round(same_pct, 1),
            'n': total,
            'status': 'EDGE' if same_pct >= 65 else 'MARGINAL' if same_pct >= 55 else 'NO_CHAIN',
        })

    return chain_edges


def test_full_session_returns(df):
    """Test 24h aggregate and full session returns with significance."""
    print("\n" + "="*80)
    print("FULL SESSION RETURNS + STATISTICAL SIGNIFICANCE")
    print("="*80)

    results = []
    for date_str in sorted(RELEASES.keys()):
        release_dt = datetime.strptime(date_str, '%Y-%m-%d').replace(hour=1, minute=30)
        pre_bar = get_bar_at(df, release_dt)
        if pre_bar is None:
            continue
        pre_price = float(pre_bar['Close'])

        release_data = RELEASES[date_str]

        # 24h return
        bars_24h = get_bars_between(df, release_dt, release_dt + timedelta(hours=24))
        if len(bars_24h) < 4:
            continue
        ret_24h = (float(bars_24h.iloc[-1]['Close']) - pre_price) / pre_price * 100

        # Classify
        cpi = release_data['cpi_yoy']
        ppi = release_data['ppi_yoy']

        if ppi < -2.0 and cpi < 0:
            signal = 'SEVERE_DEFLATION'
        elif ppi < 0 and cpi < 0:
            signal = 'DEFLATION'
        elif ppi < 0:
            signal = 'DISINFLATION'
        elif ppi > 3.0:
            signal = 'INFLATION'
        else:
            signal = 'STABLE'

        results.append({
            'date': date_str,
            'ret_24h': ret_24h,
            'signal': signal,
            'cpi': cpi,
            'ppi': ppi,
        })

    df_results = pd.DataFrame(results)
    valid = df_results[df_results['ret_24h'].notna()]

    # 1. Overall 24h
    rets = valid['ret_24h'].values
    t, p = stats.ttest_1samp(rets, 0)
    print(f"\n1. Overall 24h return (n={len(rets)}):")
    print(f"   Mean: {np.mean(rets):+.3f}%  Median: {np.median(rets):+.3f}%")
    print(f"   t={t:.3f}  p={p:.4f}  {'*** SIGNIFICANT' if p < 0.05 else 'NOT significant'}")

    # 2. By signal
    print(f"\n2. By signal type:")
    for sig in ['SEVERE_DEFLATION', 'DEFLATION', 'DISINFLATION', 'INFLATION']:
        sig_rets = valid[valid['signal'] == sig]['ret_24h'].values
        if len(sig_rets) >= 3:
            t, p = stats.ttest_1samp(sig_rets, 0)
            win = (sig_rets > 0).mean() * 100
            print(f"   {sig:<20} mean={np.mean(sig_rets):+.3f}%  win={win:.0f}%  n={len(sig_rets)}  p={p:.4f}  {'***' if p < 0.05 else 'ns'}")

    # 3. MISS vs BEAT (using PPI direction)
    prev_ppi = None
    miss_rets = []
    beat_rets = []
    for _, row in valid.iterrows():
        if prev_ppi is not None:
            delta = row['ppi'] - prev_ppi
            if delta < -0.3:
                miss_rets.append(row['ret_24h'])
            elif delta > 0.3:
                beat_rets.append(row['ret_24h'])
        prev_ppi = row['ppi']

    if len(miss_rets) >= 3 and len(beat_rets) >= 3:
        t, p = stats.ttest_ind(miss_rets, beat_rets)
        print(f"\n3. MISS vs BEAT (PPI direction):")
        print(f"   MISS: {np.mean(miss_rets):+.3f}% (n={len(miss_rets)})")
        print(f"   BEAT: {np.mean(beat_rets):+.3f}% (n={len(beat_rets)})")
        print(f"   t={t:.3f}  p={p:.4f}  {'*** SIGNIFICANT' if p < 0.05 else 'NOT significant'}")

    return df_results


def main():
    csv_path = os.path.join(os.path.dirname(__file__), 'eth_15m_merged.csv')
    print("Loading ETH 15m data...")
    df = load_eth_data(csv_path)

    chain_edges = test_chain_persistence(df)
    full_results = test_full_session_returns(df)

    # Summary
    print("\n" + "="*80)
    print("SUMMARY: CHAIN VALIDATION")
    print("="*80)

    edges = [e for e in chain_edges if e['status'] == 'EDGE']
    marginal = [e for e in chain_edges if e['status'] == 'MARGINAL']
    broken = [e for e in chain_edges if e['status'] == 'NO_CHAIN']

    print(f"\n  Real edges (>65%):   {len(edges)}")
    for e in edges:
        print(f"    ✅ {e['from']} → {e['to']}: {e['same_dir_pct']:.0f}% (n={e['n']})")

    print(f"\n  Marginal (55-65%):   {len(marginal)}")
    for e in marginal:
        print(f"    🟡 {e['from']} → {e['to']}: {e['same_dir_pct']:.0f}% (n={e['n']})")

    print(f"\n  Broken (<55%):       {len(broken)}")
    for e in broken:
        print(f"    ❌ {e['from']} → {e['to']}: {e['same_dir_pct']:.0f}% (n={e['n']})")

    if len(edges) == 0:
        print(f"\n  ⚠️ CONCLUSION: No real transmission chain found.")
        print(f"  The China CPI+PPI → session-by-session chain does NOT hold.")
        print(f"  Most transitions are <55% same direction = noise, not signal.")
    else:
        print(f"\n  ✅ CONCLUSION: {len(edges)} real chain links found.")


if __name__ == '__main__':
    main()

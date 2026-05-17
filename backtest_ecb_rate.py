#!/usr/bin/env python3
"""
Prompt A + B: Backtest ECB Rate Decision + Lagarde Press Conference (2018-today)
using ETH/USDT 15m data.

ECB announces rate decision at 14:45 CET = 13:45 UTC = 20:15 MYT.
Lagarde press conference at 15:30 CET = 14:30 UTC = 20:45 MYT.
Both fall in the EU–US overlap session.

Session itinerary (per user's thesis #33):
  Europe Afternoon (13:45 UTC Decision, 14:30 UTC Presser) → US Session → Global PMI loopback
  - Hawkish tone → EUR surges → DXY drops → ETH upside boost
  - ECB reprices global DXY baseline → sets liquidity tone for NY capital
  - Loopback: Eurozone Flash PMI validates economic activity later in the week

We measure:
  1. ETH returns across all session phases
  2. Wyckoff phase, vol regime, signal classification
  3. Cross-tabulation: (wyckoff × vol × signal) → avg 24h return, win rate
  4. Session transmission chain (Prompt B)
  5. Statistical significance
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from scipy import stats
import json
import os

# ═══════════════════════════════════════════════════════════════
# ECB RATE DECISION DATES + OUTCOMES
# Decision at 14:45 CET = 13:45 UTC = 20:15 MYT
# Press conference at 15:30 CET = 14:30 UTC = 20:45 MYT
# Format: {date: {'deposit_rate': float, 'prev_rate': float, 'signal': str}}
# signal: HIKE / CUT / HOLD / QE_EXPAND / QE_TAPER / TLTRO / FORWARD_GUIDE
# ═══════════════════════════════════════════════════════════════

RELEASES = {
    # 2018 — steady state, ending QE
    '2018-01-25': {'deposit_rate': -0.40, 'prev_rate': -0.40, 'signal': 'HOLD', 'note': 'QE tapered to €30B/month'},
    '2018-03-08': {'deposit_rate': -0.40, 'prev_rate': -0.40, 'signal': 'HOLD', 'note': 'Removed easing bias'},
    '2018-04-26': {'deposit_rate': -0.40, 'prev_rate': -0.40, 'signal': 'HOLD', 'note': 'QE tapered to €15B/month'},
    '2018-06-14': {'deposit_rate': -0.40, 'prev_rate': -0.40, 'signal': 'HOLD', 'note': 'QE to end Dec 2018'},
    '2018-07-26': {'deposit_rate': -0.40, 'prev_rate': -0.40, 'signal': 'HOLD', 'note': 'Confirmed QE end timeline'},
    '2018-09-13': {'deposit_rate': -0.40, 'prev_rate': -0.40, 'signal': 'HOLD', 'note': 'QE tapered to €15B through Dec'},
    '2018-10-25': {'deposit_rate': -0.40, 'prev_rate': -0.40, 'signal': 'HOLD', 'note': 'QE end confirmed Dec'},
    '2018-12-13': {'deposit_rate': -0.40, 'prev_rate': -0.40, 'signal': 'HOLD', 'note': 'QE officially ended'},

    # 2019 — easing restart
    '2019-01-24': {'deposit_rate': -0.40, 'prev_rate': -0.40, 'signal': 'HOLD', 'note': 'Risks moving to downside'},
    '2019-03-07': {'deposit_rate': -0.40, 'prev_rate': -0.40, 'signal': 'HOLD', 'note': 'TLTRO III announced'},
    '2019-04-10': {'deposit_rate': -0.40, 'prev_rate': -0.40, 'signal': 'HOLD', 'note': 'TLTRO III details'},
    '2019-06-06': {'deposit_rate': -0.40, 'prev_rate': -0.40, 'signal': 'HOLD', 'note': 'Forward guidance: rates through H1 2020'},
    '2019-07-25': {'deposit_rate': -0.40, 'prev_rate': -0.40, 'signal': 'HOLD', 'note': 'Hinted at Sep cut'},
    '2019-09-12': {'deposit_rate': -0.50, 'prev_rate': -0.40, 'signal': 'CUT', 'note': 'Cut + tiering + TLTRO + QE restart €20B/mo'},
    '2019-10-24': {'deposit_rate': -0.50, 'prev_rate': -0.50, 'signal': 'HOLD', 'note': 'Draghi\'s last meeting'},
    '2019-12-12': {'deposit_rate': -0.50, 'prev_rate': -0.50, 'signal': 'HOLD', 'note': 'Lagarde\'s first meeting, strategic review'},

    # 2020 — pandemic response
    '2020-01-30': {'deposit_rate': -0.50, 'prev_rate': -0.50, 'signal': 'HOLD', 'note': 'Strategic review launched'},
    '2020-03-12': {'deposit_rate': -0.50, 'prev_rate': -0.50, 'signal': 'QE_EXPAND', 'note': 'PEPP €750B announced (COVID)'},
    '2020-04-30': {'deposit_rate': -0.50, 'prev_rate': -0.50, 'signal': 'QE_EXPAND', 'note': 'PEPP expanded, favorable conditions'},
    '2020-06-04': {'deposit_rate': -0.50, 'prev_rate': -0.50, 'signal': 'QE_EXPAND', 'note': 'PEPP +€600B to €1.35T'},
    '2020-07-16': {'deposit_rate': -0.50, 'prev_rate': -0.50, 'signal': 'HOLD', 'note': 'PEPP maintained'},
    '2020-09-10': {'deposit_rate': -0.50, 'prev_rate': -0.50, 'signal': 'HOLD', 'note': 'Monitoring EUR appreciation'},
    '2020-10-29': {'deposit_rate': -0.50, 'prev_rate': -0.50, 'signal': 'HOLD', 'note': 'Recalibration in Dec'},
    '2020-12-10': {'deposit_rate': -0.50, 'prev_rate': -0.50, 'signal': 'QE_EXPAND', 'note': 'PEPP +€500B to €1.85T, extended'},

    # 2021 — maintaining accommodation
    '2021-01-21': {'deposit_rate': -0.50, 'prev_rate': -0.50, 'signal': 'HOLD', 'note': 'PEPP pace significantly increased'},
    '2021-03-11': {'deposit_rate': -0.50, 'prev_rate': -0.50, 'signal': 'QE_TAPER', 'note': 'PEPP pace "significantly higher" (but markets saw hawkish tilt)'},
    '2021-04-22': {'deposit_rate': -0.50, 'prev_rate': -0.50, 'signal': 'HOLD', 'note': 'Faster PEPP purchases'},
    '2021-06-10': {'deposit_rate': -0.50, 'prev_rate': -0.50, 'signal': 'HOLD', 'note': 'PEPP pace maintained'},
    '2021-07-22': {'deposit_rate': -0.50, 'prev_rate': -0.50, 'signal': 'HOLD', 'note': 'New 2% symmetric target'},
    '2021-09-09': {'deposit_rate': -0.50, 'prev_rate': -0.50, 'signal': 'QE_TAPER', 'note': 'PEPP pace "moderately lower"'},
    '2021-10-28': {'deposit_rate': -0.50, 'prev_rate': -0.50, 'signal': 'HOLD', 'note': 'Inflation mostly transitory'},
    '2021-12-16': {'deposit_rate': -0.50, 'prev_rate': -0.50, 'signal': 'QE_TAPER', 'note': 'PEPP to end Mar 2022, APP +€40B in Q2'},

    # 2022 — pivot to tightening
    '2022-02-03': {'deposit_rate': -0.50, 'prev_rate': -0.50, 'signal': 'HOLD', 'note': 'Inflation elevated, Lagarde more hawkish'},
    '2022-03-10': {'deposit_rate': -0.50, 'prev_rate': -0.50, 'signal': 'QE_TAPER', 'note': 'APP to end Q3, faster taper'},
    '2022-04-14': {'deposit_rate': -0.50, 'prev_rate': -0.50, 'signal': 'HOLD', 'note': 'APP to end Q3'},
    '2022-06-09': {'deposit_rate': -0.50, 'prev_rate': -0.50, 'signal': 'HOLD', 'note': 'Announced July hike, ended APP'},
    '2022-07-21': {'deposit_rate': 0.00, 'prev_rate': -0.50, 'signal': 'HIKE', 'note': 'First hike since 2011 (+50bp), TPI announced'},
    '2022-09-08': {'deposit_rate': 0.75, 'prev_rate': 0.00, 'signal': 'HIKE', 'note': '+75bp, largest ever'},
    '2022-10-27': {'deposit_rate': 1.50, 'prev_rate': 0.75, 'signal': 'HIKE', 'note': '+75bp again'},
    '2022-12-15': {'deposit_rate': 2.00, 'prev_rate': 1.50, 'signal': 'HIKE', 'note': '+50bp, QT announced'},

    # 2023 — peak rates
    '2023-02-02': {'deposit_rate': 2.50, 'prev_rate': 2.00, 'signal': 'HIKE', 'note': '+50bp, more to come'},
    '2023-03-16': {'deposit_rate': 3.00, 'prev_rate': 2.50, 'signal': 'HIKE', 'note': '+50bp, banking stress ignored'},
    '2023-05-04': {'deposit_rate': 3.25, 'prev_rate': 3.00, 'signal': 'HIKE', 'note': '+25bp, slowing pace'},
    '2023-06-15': {'deposit_rate': 3.50, 'prev_rate': 3.25, 'signal': 'HIKE', 'note': '+25bp'},
    '2023-07-27': {'deposit_rate': 3.75, 'prev_rate': 3.50, 'signal': 'HIKE', 'note': '+25bp'},
    '2023-09-14': {'deposit_rate': 4.00, 'prev_rate': 3.75, 'signal': 'HIKE', 'note': 'Peak rate reached (+25bp)'},
    '2023-10-26': {'deposit_rate': 4.00, 'prev_rate': 4.00, 'signal': 'HOLD', 'note': 'First hold after 10 consecutive hikes'},
    '2023-12-14': {'deposit_rate': 4.00, 'prev_rate': 4.00, 'signal': 'HOLD', 'note': 'Hold, dovish pivot signaled'},

    # 2024 — cutting cycle begins
    '2024-01-25': {'deposit_rate': 4.00, 'prev_rate': 4.00, 'signal': 'HOLD', 'note': 'Hold, data-dependent'},
    '2024-03-07': {'deposit_rate': 4.00, 'prev_rate': 4.00, 'signal': 'HOLD', 'note': 'Hold, new projections'},
    '2024-04-11': {'deposit_rate': 4.00, 'prev_rate': 4.00, 'signal': 'HOLD', 'note': 'Hold, signaled June cut'},
    '2024-06-06': {'deposit_rate': 3.75, 'prev_rate': 4.00, 'signal': 'CUT', 'note': 'First cut since 2019 (-25bp)'},
    '2024-07-18': {'deposit_rate': 3.75, 'prev_rate': 3.75, 'signal': 'HOLD', 'note': 'Hold, data-dependent'},
    '2024-09-12': {'deposit_rate': 3.50, 'prev_rate': 3.75, 'signal': 'CUT', 'note': 'Second cut (-25bp)'},
    '2024-10-17': {'deposit_rate': 3.25, 'prev_rate': 3.50, 'signal': 'CUT', 'note': 'Third cut (-25bp)'},
    '2024-12-12': {'deposit_rate': 3.00, 'prev_rate': 3.25, 'signal': 'CUT', 'note': 'Fourth cut (-25bp)'},

    # 2025 — continued easing
    '2025-01-30': {'deposit_rate': 2.75, 'prev_rate': 3.00, 'signal': 'CUT', 'note': 'Cut -25bp'},
    '2025-03-06': {'deposit_rate': 2.50, 'prev_rate': 2.75, 'signal': 'CUT', 'note': 'Cut -25bp'},
    '2025-04-17': {'deposit_rate': 2.40, 'prev_rate': 2.50, 'signal': 'CUT', 'note': 'Cut -10bp (deposit rate vs refi convergence)'},
    '2025-06-05': {'deposit_rate': 2.15, 'prev_rate': 2.40, 'signal': 'CUT', 'note': 'Cut -25bp (refi -25bp, dep -25bp)'},
    '2025-07-17': {'deposit_rate': 2.15, 'prev_rate': 2.15, 'signal': 'HOLD', 'note': 'Hold, assessing'},
    '2025-09-11': {'deposit_rate': 2.15, 'prev_rate': 2.15, 'signal': 'HOLD', 'note': 'Hold'},
    '2025-10-30': {'deposit_rate': 2.15, 'prev_rate': 2.15, 'signal': 'HOLD', 'note': 'Hold'},
    '2025-12-18': {'deposit_rate': 2.00, 'prev_rate': 2.15, 'signal': 'CUT', 'note': 'Cut -15bp'},

    # 2026
    '2026-02-05': {'deposit_rate': 2.00, 'prev_rate': 2.00, 'signal': 'HOLD', 'note': 'Hold'},
    '2026-03-19': {'deposit_rate': 2.00, 'prev_rate': 2.00, 'signal': 'HOLD', 'note': 'Hold'},
    '2026-04-30': {'deposit_rate': 2.00, 'prev_rate': 2.00, 'signal': 'HOLD', 'note': 'Hold'},
}


# ═══════════════════════════════════════════════════════════════
# SESSION DEFINITIONS (ECB at 13:45 UTC, Presser at 14:30 UTC)
# Europe Afternoon → US Session → Next day
# ═══════════════════════════════════════════════════════════════

SESSION_CHAIN = [
    # Pre-release (Europe morning)
    ('Pre-Asia', [0, 1, 2]),
    ('Sydney Open', [22, 23, 0, 1]),
    ('Tokyo Open', [0, 1, 2]),
    ('Asia Mid', [3, 4, 5]),
    ('Asia Afternoon', [5, 6, 7]),
    ('Tokyo Close', [6, 7]),
    ('Pre-London', [7, 8]),
    ('Frankfurt Open', [7, 8]),
    ('London Open', [8, 9]),
    ('London Morning', [9, 10, 11]),
    ('London Midday', [12, 13]),

    # Release window (Europe afternoon)
    ('NY Pre-Open', [12, 13]),
    ('ECB Decision (1h)', [13, 14]),       # 13:45-14:45 UTC (decision + presser start)
    ('NY Open', [13, 14]),
    ('ECB Presser (1h)', [14, 15]),        # 14:30-15:30 UTC (press conference)
    ('London-NY Overlap', [14, 15]),

    # Post-release
    ('NY AM', [14, 15, 16]),
    ('NY Lunch', [17]),
    ('NY PM', [18, 19, 20]),

    # Aggregate windows
    ('ECB Window (2h)', [13, 14, 15]),     # 13:45-15:45 (decision + presser)
    ('Post-ECB US (4h)', [14, 15, 16, 17, 18]),  # 14:00-18:00 (US reaction)
]


def load_eth_data(csv_path):
    df = pd.read_csv(csv_path)
    df['Open time'] = pd.to_datetime(df['Open time'])
    df = df.sort_values('Open time').reset_index(drop=True)
    return df


def get_bar_at(df, dt):
    mask = df['Open time'] <= dt
    if mask.sum() == 0:
        return None
    return df[mask].iloc[-1]


def get_bars_between(df, start, end):
    mask = (df['Open time'] >= start) & (df['Open time'] < end)
    return df[mask]


def get_return(df, start_dt, end_dt):
    bar_start = get_bar_at(df, start_dt)
    bars_end = get_bars_between(df, start_dt, end_dt)
    if bar_start is None or len(bars_end) == 0:
        return None
    start_price = float(bar_start['Close'])
    end_price = float(bars_end.iloc[-1]['Close'])
    return (end_price - start_price) / start_price * 100


def classify_wyckoff_phase(df, release_dt, lookback=96):
    start = release_dt - timedelta(hours=24)
    bars = get_bars_between(df, start, release_dt)
    if len(bars) < 20:
        return 'RANGE'
    price_series = bars['Close'].astype(float)
    high = price_series.max()
    low = price_series.min()
    current = price_series.iloc[-1]
    range_pct = (high - low) / low * 100
    pos_in_range = (current - low) / (high - low) if high != low else 0.5
    if range_pct < 1.5:
        return 'RANGE' if 0.3 < pos_in_range < 0.7 else (
            'ACCUMULATION' if pos_in_range < 0.3 else 'DISTRIBUTION')
    elif current > price_series.mean() and pos_in_range > 0.6:
        return 'MARKUP'
    elif current < price_series.mean() and pos_in_range < 0.4:
        return 'MARKDOWN'
    elif pos_in_range > 0.7:
        return 'DISTRIBUTION'
    elif pos_in_range < 0.3:
        return 'ACCUMULATION'
    else:
        return 'RANGE'


def classify_vol_regime(df, release_dt, lookback_hours=24):
    start = release_dt - timedelta(hours=lookback_hours)
    bars = get_bars_between(df, start, release_dt)
    if len(bars) < 10:
        return 'CHOP'
    closes = bars['Close'].astype(float).values
    ranges = (bars['High'].astype(float).values - bars['Low'].astype(float).values)
    avg_range_pct = np.mean(ranges / closes) * 100
    x = np.arange(len(closes))
    slope, _, r_value, _, _ = stats.linregress(x, closes)
    r_squared = r_value ** 2
    if r_squared > 0.7 and abs(slope) > 0.5:
        return 'TREND'
    elif avg_range_pct < 0.3:
        return 'SQUEEZE'
    elif avg_range_pct < 0.6:
        return 'LOW_VOL'
    else:
        return 'CHOP'


def classify_ecb_signal(deposit_rate, prev_rate, signal_hint):
    """
    Classify ECB signal for backtesting.
    HIKE → hawkish → EUR up → DXY down → ETH LONG
    CUT → dovish → context-dependent (risk-on or risk-off depending on recession fears)
    HOLD → context-dependent
    """
    if signal_hint in ('HIKE',):
        return 'HIKE'
    elif signal_hint in ('CUT',):
        return 'CUT'
    elif signal_hint in ('QE_EXPAND',):
        return 'DOVE'
    elif signal_hint in ('QE_TAPER',):
        return 'HAWKISH'
    elif signal_hint in ('TLTRO', 'FORWARD_GUIDE'):
        return 'HOLD'
    delta = deposit_rate - prev_rate
    if delta > 0:
        return 'HIKE'
    elif delta < 0:
        return 'CUT'
    return 'HOLD'


def run_backtest(csv_path):
    df = load_eth_data(csv_path)
    results = []
    sorted_dates = sorted(RELEASES.keys())

    for i, date_str in enumerate(sorted_dates):
        release_data = RELEASES[date_str]
        # ECB decision at 13:45 UTC
        release_dt = datetime.strptime(date_str, '%Y-%m-%d').replace(hour=13, minute=45)

        bars_after = get_bars_between(df, release_dt, release_dt + timedelta(hours=24))
        if len(bars_after) < 4:
            continue

        pre_bar = get_bar_at(df, release_dt)
        if pre_bar is None:
            continue

        pre_price = float(pre_bar['Close'])

        deposit_rate = release_data['deposit_rate']
        prev_rate = release_data['prev_rate']
        signal_hint = release_data.get('signal', 'HOLD')
        signal = classify_ecb_signal(deposit_rate, prev_rate, signal_hint)
        wyckoff = classify_wyckoff_phase(df, release_dt)
        vol_regime = classify_vol_regime(df, release_dt)

        session_returns = {}
        for name, hours in SESSION_CHAIN:
            if len(hours) == 1:
                start = release_dt.replace(hour=hours[0], minute=0)
                end = start + timedelta(hours=1)
            else:
                start = release_dt.replace(hour=hours[0], minute=0)
                end = release_dt.replace(hour=hours[-1] + 1, minute=0)
            if end <= start:
                end += timedelta(days=1)
            ret = get_return(df, start, end)
            if ret is not None:
                session_returns[name] = round(ret, 4)

        ret_24h = get_return(df, release_dt, release_dt + timedelta(hours=24))
        ret_48h = get_return(df, release_dt, release_dt + timedelta(hours=48))
        year = int(date_str[:4])

        results.append({
            'date': date_str, 'year': year,
            'deposit_rate': deposit_rate, 'prev_rate': prev_rate,
            'rate_change': round(deposit_rate - prev_rate, 2),
            'signal': signal, 'signal_hint': signal_hint,
            'note': release_data.get('note', ''),
            'wyckoff': wyckoff, 'vol_regime': vol_regime,
            'pre_price': round(pre_price, 2),
            'ret_24h': round(ret_24h, 4) if ret_24h else None,
            'ret_48h': round(ret_48h, 4) if ret_48h else None,
            'session_returns': session_returns,
        })

    return pd.DataFrame(results)


def cross_tabulate(results):
    print("\n" + "="*80)
    print("CROSS-TABULATION: Signal × Vol Regime × Wyckoff Phase")
    print("="*80)
    valid = results[results['ret_24h'].notna()]
    combos = valid.groupby(['signal', 'vol_regime', 'wyckoff'])
    rows = []
    for (sig, vol, wy), group in combos:
        n = len(group)
        avg_ret = group['ret_24h'].mean()
        win_rate = (group['ret_24h'] > 0).mean()
        rows.append({'signal': sig, 'vol': vol, 'wyckoff': wy, 'n': n,
                     'avg_24h': round(avg_ret, 3), 'win_rate': round(win_rate, 3)})
    df = pd.DataFrame(rows).sort_values('n', ascending=False)
    print(df.to_string(index=False))
    edges = df[(df['n'] >= 3) & (df['avg_24h'].abs() >= 0.5)]
    if len(edges) > 0:
        print("\n" + "-"*60)
        print("ACTIONABLE EDGES (n≥3, |avg|≥0.5%):")
        print("-"*60)
        print(edges.to_string(index=False))
    else:
        print("\nNo actionable edges found (n≥3, |avg|≥0.5%)")
    return df


def transmission_chain(results):
    """Prompt B: Session-by-session transmission chain validation."""
    print("\n" + "="*80)
    print("SESSION TRANSMISSION CHAIN (Prompt B)")
    print("="*80)
    valid = results[results['ret_24h'].notna()]
    key_sessions = ['Asia Mid', 'Asia Afternoon', 'Pre-London',
                    'London Open', 'London Morning', 'London Midday',
                    'ECB Decision (1h)', 'NY Open', 'ECB Presser (1h)',
                    'London-NY Overlap', 'NY AM', 'NY PM']

    print(f"\n{'Session':<25} {'Avg Ret%':>10} {'Win%':>8} {'n':>5} {'Dir':>6}")
    print("-"*60)
    chain_data = []
    for session_name in key_sessions:
        rets = [sr[session_name] for _, row in valid.iterrows()
                for sr in [row.get('session_returns', {})] if session_name in sr]
        if len(rets) < 3:
            continue
        avg = np.mean(rets)
        win = sum(1 for r in rets if r > 0) / len(rets) * 100
        n = len(rets)
        curr_dir = 'UP' if avg > 0.05 else 'DOWN' if avg < -0.05 else 'FLAT'
        print(f"{session_name:<25} {avg:>+10.3f} {win:>7.0f}% {n:>5} {curr_dir:>6}")
        chain_data.append({'session': session_name, 'avg_ret': round(avg, 3),
                           'win_pct': round(win, 1), 'n': n, 'direction': curr_dir})

    print(f"\n{'Transition':<50} {'Same Dir%':>10} {'n':>5} {'Verdict':>10}")
    print("-"*80)
    transitions = []
    for i in range(len(key_sessions) - 1):
        s1, s2 = key_sessions[i], key_sessions[i+1]
        same_count = total = 0
        for _, row in valid.iterrows():
            sr = row.get('session_returns', {})
            if s1 in sr and s2 in sr:
                r1, r2 = sr[s1], sr[s2]
                if (r1 > 0.05 and r2 > 0.05) or (r1 < -0.05 and r2 < -0.05):
                    same_count += 1
                total += 1
        if total >= 3:
            pct = same_count / total * 100
            verdict = '✅ EDGE' if pct > 65 else '⚠️ MARGINAL' if pct > 55 else '❌ NO CHAIN'
            print(f"{s1 + ' → ' + s2:<50} {pct:>9.0f}% {total:>5} {verdict:>10}")
            transitions.append({'from': s1, 'to': s2, 'same_dir_pct': round(pct, 1),
                                'n': total, 'verdict': verdict})
    return chain_data, transitions


def direction_persistence_to_reopen(results):
    """Measure % same-direction from each session until NY PM (end of cycle)."""
    print("\n" + "="*80)
    print("DIRECTION PERSISTENCE: Session → End of Cycle (NY PM)")
    print("="*80)
    valid = results[results['ret_24h'].notna()]

    sessions_ordered = [
        'Asia Mid', 'Asia Afternoon', 'Pre-London',
        'London Open', 'London Morning', 'London Midday',
        'ECB Decision (1h)', 'NY Open', 'ECB Presser (1h)',
        'London-NY Overlap', 'NY AM'
    ]

    print(f"\n{'From Session':<25} {'To Session':<25} {'Same Dir%':>10} {'n':>5} {'Verdict':>10}")
    print("-"*80)

    for s1 in sessions_ordered:
        s2 = 'NY PM'
        same_count = total = 0
        for _, row in valid.iterrows():
            sr = row.get('session_returns', {})
            if s1 in sr and s2 in sr:
                r1, r2 = sr[s1], sr[s2]
                if (r1 > 0.05 and r2 > 0.05) or (r1 < -0.05 and r2 < -0.05):
                    same_count += 1
                total += 1
        if total >= 3:
            pct = same_count / total * 100
            verdict = '✅ PERSISTS' if pct > 65 else '⚠️ MARGINAL' if pct > 55 else '❌ DECAYS'
            print(f"{s1:<25} {'NY PM (end)':<25} {pct:>9.0f}% {total:>5} {verdict:>10}")


def statistical_tests(results):
    print("\n" + "="*80)
    print("STATISTICAL SIGNIFICANCE")
    print("="*80)
    valid = results[results['ret_24h'].notna()]
    rets = valid['ret_24h'].values

    if len(rets) < 3:
        print("Not enough data for statistical tests (n<3)")
        return

    t_stat, p_val = stats.ttest_1samp(rets, 0)
    print(f"\n1. One-sample t-test (24h return vs 0):")
    print(f"   Mean: {np.mean(rets):+.3f}%  t={t_stat:.3f}  p={p_val:.4f}  {'*** SIGNIFICANT' if p_val < 0.05 else 'NOT significant'}")

    hike_rets = valid[valid['signal'] == 'HIKE']['ret_24h'].values
    hold_rets = valid[valid['signal'] == 'HOLD']['ret_24h'].values
    cut_rets = valid[valid['signal'] == 'CUT']['ret_24h'].values
    hawk_rets = valid[valid['signal'] == 'HAWKISH']['ret_24h'].values
    dove_rets = valid[valid['signal'] == 'DOVE']['ret_24h'].values

    pairs = [
        ('HIKE', hike_rets, 'HOLD', hold_rets),
        ('CUT', cut_rets, 'HOLD', hold_rets),
        ('HIKE', hike_rets, 'CUT', cut_rets),
        ('HAWKISH', hawk_rets, 'HOLD', hold_rets),
        ('DOVE', dove_rets, 'HOLD', hold_rets),
    ]

    test_num = 2
    for name1, r1, name2, r2 in pairs:
        if len(r1) >= 3 and len(r2) >= 3:
            t2, p2 = stats.ttest_ind(r1, r2)
            print(f"\n{test_num}. Two-sample t-test ({name1} vs {name2}):")
            print(f"   {name1}: {np.mean(r1):+.3f}% (n={len(r1)})  {name2}: {np.mean(r2):+.3f}% (n={len(r2)})")
            print(f"   t={t2:.3f}  p={p2:.4f}  {'*** SIGNIFICANT' if p2 < 0.05 else 'NOT significant'}")
            test_num += 1

    for sig in ['HIKE', 'CUT', 'HOLD', 'HAWKISH', 'DOVE']:
        sig_rets = valid[valid['signal'] == sig]['ret_24h'].values
        if len(sig_rets) >= 3:
            t5, p5 = stats.ttest_1samp(sig_rets, 0)
            print(f"\n{test_num}. {sig} one-sample t-test (n={len(sig_rets)}):")
            print(f"   Mean: {np.mean(sig_rets):+.3f}%  t={t5:.3f}  p={p5:.4f}  {'***' if p5 < 0.05 else 'ns'}")
            test_num += 1

    print(f"\n{test_num}. Year-by-year 24h returns:")
    print(f"   {'Year':<6} {'Avg%':>8} {'Win%':>7} {'n':>4}")
    print(f"   {'-'*30}")
    for year in sorted(valid['year'].unique()):
        yr = valid[valid['year'] == year]
        print(f"   {year:<6} {yr['ret_24h'].mean():>+8.3f} {(yr['ret_24h'] > 0).mean()*100:>6.0f}% {len(yr):>4}")


def main():
    csv_path = os.path.join(os.path.dirname(__file__), 'eth_15m_merged.csv')
    print("="*80)
    print("PROMPT A: ECB RATE DECISION + LAGARDE PRESS CONFERENCE BACKTEST (2018-2026)")
    print("="*80)
    print("Loading ETH 15m data...")
    results = run_backtest(csv_path)
    print(f"Analyzed {len(results)} ECB rate decisions ({results['year'].min()}-{results['year'].max()})")

    valid = results[results['ret_24h'].notna()]
    print(f"\n24h Aggregate Returns:")
    print(f"  Mean:   {valid['ret_24h'].mean():+.3f}%")
    print(f"  Median: {valid['ret_24h'].median():+.3f}%")
    print(f"  Std:    {valid['ret_24h'].std():.3f}%")
    print(f"  Win%:   {(valid['ret_24h'] > 0).mean()*100:.1f}%")
    print(f"  n:      {len(valid)}")

    print("\n  By Signal:")
    for sig in ['HIKE', 'CUT', 'HOLD', 'HAWKISH', 'DOVE', 'QE_EXPAND', 'QE_TAPER']:
        sig_data = valid[valid['signal'] == sig]
        if len(sig_data) > 0:
            print(f"  {sig:<15} avg={sig_data['ret_24h'].mean():+.3f}%  win={(sig_data['ret_24h'] > 0).mean()*100:.0f}%  n={len(sig_data)}")

    print("\n  By Signal Hint (granular):")
    for sig in sorted(valid['signal_hint'].unique()):
        sig_data = valid[valid['signal_hint'] == sig]
        if len(sig_data) > 0:
            print(f"  {sig:<15} avg={sig_data['ret_24h'].mean():+.3f}%  win={(sig_data['ret_24h'] > 0).mean()*100:.0f}%  n={len(sig_data)}")

    cross_tabulate(results)

    print("\n" + "="*80)
    print("PROMPT B: SESSION TRANSMISSION CHAIN VALIDATION")
    print("="*80)
    transmission_chain(results)
    direction_persistence_to_reopen(results)
    statistical_tests(results)

    out_path = os.path.join(os.path.dirname(__file__), 'backtest_ecb_rate_results.json')
    results.to_json(out_path, orient='records', indent=2)
    print(f"\nResults saved to {out_path}")


if __name__ == '__main__':
    main()

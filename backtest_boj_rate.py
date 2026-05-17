#!/usr/bin/env python3
"""
Prompt A + B: Backtest BoJ Rate Decision (2018-today) using ETH/USDT 15m data.

BoJ announces rate decisions at the conclusion of Monetary Policy Meetings.
Typical announcement: ~11:00-12:30 MYT = 03:00-04:30 UTC (Asia Mid session).
Unscheduled meetings occur occasionally.

Session itinerary (per user's thesis #31):
  Asia Mid-Day (03:00-04:30 UTC) → Europe Open → US Session
  - BoJ controls global yield curve & interest rate guidance
  - Unexpected hawkish shift → JPY surge → USD/JPY plunge → carry trade unwind → ETH liquidation
  - "Shadow Wicks" in orderbooks from forced liquidation
  - Europe inherits high-vol risk-off → market makers lower bid tiers
  - Loopback: next Tokyo CPI validates/challenges the BoJ move

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
# BOJ RATE DECISION DATES + OUTCOMES
# Announcement ~11:00-12:30 MYT = 03:00-04:30 UTC
# We use 03:30 UTC as proxy announcement time
# Format: {date: {'rate': float, 'prev_rate': float, 'signal': str}}
# signal: HIKE / MILD_HIKE / HOLD / DOVE / BIG_DOVE / YCC_EXPAND / YCC_REMOVE / NIRP_END
# ═══════════════════════════════════════════════════════════════

RELEASES = {
    # 2018 — NIRP at -0.1%, 10Y YCC at ~0%
    '2018-01-23': {'rate': -0.10, 'prev_rate': -0.10, 'signal': 'HOLD', 'note': 'YCC unchanged'},
    '2018-03-09': {'rate': -0.10, 'prev_rate': -0.10, 'signal': 'HOLD', 'note': 'YCC unchanged'},
    '2018-04-27': {'rate': -0.10, 'prev_rate': -0.10, 'signal': 'HOLD', 'note': 'Removed 2% inflation target timeframe'},
    '2018-06-15': {'rate': -0.10, 'prev_rate': -0.10, 'signal': 'HOLD', 'note': 'YCC unchanged'},
    '2018-07-31': {'rate': -0.10, 'prev_rate': -0.10, 'signal': 'HOLD', 'note': 'Yield band widened to ±0.20%, introduced forward guidance'},
    '2018-09-19': {'rate': -0.10, 'prev_rate': -0.10, 'signal': 'HOLD', 'note': 'YCC band ±0.20% maintained'},
    '2018-10-31': {'rate': -0.10, 'prev_rate': -0.10, 'signal': 'HOLD', 'note': 'YCC unchanged'},
    '2018-12-20': {'rate': -0.10, 'prev_rate': -0.10, 'signal': 'HOLD', 'note': 'YCC unchanged, tweaked ETF buying'},

    # 2019 — cautious easing
    '2019-01-23': {'rate': -0.10, 'prev_rate': -0.10, 'signal': 'HOLD', 'note': 'Lowered inflation outlook'},
    '2019-03-15': {'rate': -0.10, 'prev_rate': -0.10, 'signal': 'HOLD', 'note': 'YCC unchanged'},
    '2019-04-25': {'rate': -0.10, 'prev_rate': -0.10, 'signal': 'HOLD', 'note': 'Extended forward guidance'},
    '2019-06-20': {'rate': -0.10, 'prev_rate': -0.10, 'signal': 'HOLD', 'note': 'YCC unchanged'},
    '2019-07-30': {'rate': -0.10, 'prev_rate': -0.10, 'signal': 'HOLD', 'note': 'Strengthened forward guidance'},
    '2019-09-19': {'rate': -0.10, 'prev_rate': -0.10, 'signal': 'HOLD', 'note': 'YCC unchanged'},
    '2019-10-31': {'rate': -0.10, 'prev_rate': -0.10, 'signal': 'HOLD', 'note': 'Removed timeframe for 2% target'},
    '2019-12-19': {'rate': -0.10, 'prev_rate': -0.10, 'signal': 'HOLD', 'note': 'YCC unchanged'},

    # 2020 — COVID response
    '2020-01-21': {'rate': -0.10, 'prev_rate': -0.10, 'signal': 'HOLD', 'note': 'YCC unchanged'},
    '2020-03-16': {'rate': -0.10, 'prev_rate': -0.10, 'signal': 'DOVE', 'note': 'Emergency: ETF buying doubled, new lending scheme'},
    '2020-04-27': {'rate': -0.10, 'prev_rate': -0.10, 'signal': 'BIG_DOVE', 'note': 'Removed JGB purchase ceiling, unlimited bond buying'},
    '2020-06-16': {'rate': -0.10, 'prev_rate': -0.10, 'signal': 'HOLD', 'note': 'YCC unchanged, COVID lending expanded'},
    '2020-07-15': {'rate': -0.10, 'prev_rate': -0.10, 'signal': 'HOLD', 'note': 'YCC unchanged'},
    '2020-09-17': {'rate': -0.10, 'prev_rate': -0.10, 'signal': 'HOLD', 'note': 'YCC unchanged'},
    '2020-10-29': {'rate': -0.10, 'prev_rate': -0.10, 'signal': 'HOLD', 'note': 'YCC unchanged, assessed COVID impact'},
    '2020-12-18': {'rate': -0.10, 'prev_rate': -0.10, 'signal': 'HOLD', 'note': 'Extended COVID lending by 6 months'},

    # 2021 — yield band tweak
    '2021-03-19': {'rate': -0.10, 'prev_rate': -0.10, 'signal': 'HOLD', 'note': 'Yield band widened to ±0.25%, BOJ ETF buying paused'},
    '2021-04-27': {'rate': -0.10, 'prev_rate': -0.10, 'signal': 'HOLD', 'note': 'YCC unchanged'},
    '2021-06-18': {'rate': -0.10, 'prev_rate': -0.10, 'signal': 'HOLD', 'note': 'Extended COVID lending, climate lending facility'},
    '2021-07-16': {'rate': -0.10, 'prev_rate': -0.10, 'signal': 'HOLD', 'note': 'YCC unchanged'},
    '2021-09-22': {'rate': -0.10, 'prev_rate': -0.10, 'signal': 'HOLD', 'note': 'YCC unchanged'},
    '2021-10-28': {'rate': -0.10, 'prev_rate': -0.10, 'signal': 'HOLD', 'note': 'YCC unchanged'},
    '2021-12-17': {'rate': -0.10, 'prev_rate': -0.10, 'signal': 'HOLD', 'note': 'Tapered COVID lending, but extended deadline'},

    # 2022 — global tightening era, BOJ holds
    '2022-01-18': {'rate': -0.10, 'prev_rate': -0.10, 'signal': 'HOLD', 'note': 'YCC unchanged, raised inflation forecasts'},
    '2022-03-18': {'rate': -0.10, 'prev_rate': -0.10, 'signal': 'HOLD', 'note': 'YCC unchanged'},
    '2022-04-28': {'rate': -0.10, 'prev_rate': -0.10, 'signal': 'HOLD', 'note': 'Unlimited JGB buying to defend YCC'},
    '2022-06-17': {'rate': -0.10, 'prev_rate': -0.10, 'signal': 'HOLD', 'note': 'YCC unchanged, JPY plunging'},
    '2022-07-21': {'rate': -0.10, 'prev_rate': -0.10, 'signal': 'HOLD', 'note': 'YCC ±0.25% maintained, massive JPY weakness'},
    '2022-09-22': {'rate': -0.10, 'prev_rate': -0.10, 'signal': 'HOLD', 'note': 'YCC unchanged, intervened in FX'},
    '2022-10-28': {'rate': -0.10, 'prev_rate': -0.10, 'signal': 'HOLD', 'note': 'YCC ±0.50% band widened (hawkish tweak)'},
    '2022-12-20': {'rate': -0.10, 'prev_rate': -0.10, 'signal': 'MILD_HIKE', 'note': 'YCC band widened to ±0.50% (surprise hawkish)'},

    # 2023 — gradual normalization
    '2023-01-18': {'rate': -0.10, 'prev_rate': -0.10, 'signal': 'HOLD', 'note': 'YCC ±0.50% maintained'},
    '2023-03-10': {'rate': -0.10, 'prev_rate': -0.10, 'signal': 'HOLD', 'note': 'Last meeting under Kuroda'},
    '2023-04-28': {'rate': -0.10, 'prev_rate': -0.10, 'signal': 'HOLD', 'note': 'First Ueda meeting, policy review announced'},
    '2023-06-16': {'rate': -0.10, 'prev_rate': -0.10, 'signal': 'HOLD', 'note': 'YCC unchanged'},
    '2023-07-28': {'rate': -0.10, 'prev_rate': -0.10, 'signal': 'MILD_HIKE', 'note': 'YCC band widened to ±1.0% (hawkish surprise)'},
    '2023-09-22': {'rate': -0.10, 'prev_rate': -0.10, 'signal': 'HOLD', 'note': 'YCC ±1.0% maintained'},
    '2023-10-31': {'rate': -0.10, 'prev_rate': -0.10, 'signal': 'DOVE', 'note': 'YCC band effectively removed, 1% as "reference"'},
    '2023-12-19': {'rate': -0.10, 'prev_rate': -0.10, 'signal': 'HOLD', 'note': 'NIRP maintained, markets expecting exit'},

    # 2024 — historic exit from NIRP + YCC
    '2024-01-23': {'rate': -0.10, 'prev_rate': -0.10, 'signal': 'HOLD', 'note': 'NIRP maintained, hawkish hints'},
    '2024-03-19': {'rate': 0.10, 'prev_rate': -0.10, 'signal': 'HIKE', 'note': 'HISTORIC: Ended NIRP, ended YCC, first hike since 2007'},
    '2024-04-26': {'rate': 0.10, 'prev_rate': 0.10, 'signal': 'HOLD', 'note': 'First post-NIRP hold, JPY weakness'},
    '2024-06-14': {'rate': 0.10, 'prev_rate': 0.10, 'signal': 'HOLD', 'note': 'Hold, bond taper plan announced'},
    '2024-07-31': {'rate': 0.25, 'prev_rate': 0.10, 'signal': 'HIKE', 'note': 'Second hike to 0.25%, hawkish surprise'},
    '2024-09-20': {'rate': 0.25, 'prev_rate': 0.25, 'signal': 'HOLD', 'note': 'Hold, Ueda cautious tone'},
    '2024-10-31': {'rate': 0.25, 'prev_rate': 0.25, 'signal': 'HOLD', 'note': 'Hold, markets split on Dec hike'},
    '2024-12-19': {'rate': 0.25, 'prev_rate': 0.25, 'signal': 'HOLD', 'note': 'Hold, hawkish statement for Jan 2025'},

    # 2025 — continued normalization
    '2025-01-24': {'rate': 0.50, 'prev_rate': 0.25, 'signal': 'HIKE', 'note': 'Hike to 0.50%, highest since 2008'},
    '2025-03-14': {'rate': 0.50, 'prev_rate': 0.50, 'signal': 'HOLD', 'note': 'Hold, assessing global trade risks'},
    '2025-05-02': {'rate': 0.50, 'prev_rate': 0.50, 'signal': 'HOLD', 'note': 'Hold, tariff uncertainty'},
    '2025-06-13': {'rate': 0.50, 'prev_rate': 0.50, 'signal': 'HOLD', 'note': 'Hold, trade tensions'},
    '2025-07-25': {'rate': 0.50, 'prev_rate': 0.50, 'signal': 'HOLD', 'note': 'Hold'},
    '2025-09-19': {'rate': 0.50, 'prev_rate': 0.50, 'signal': 'HOLD', 'note': 'Hold'},
    '2025-10-31': {'rate': 0.50, 'prev_rate': 0.50, 'signal': 'HOLD', 'note': 'Hold'},
    '2025-12-19': {'rate': 0.75, 'prev_rate': 0.50, 'signal': 'HIKE', 'note': 'Hike to 0.75%'},

    # 2026
    '2026-01-23': {'rate': 0.75, 'prev_rate': 0.75, 'signal': 'HOLD', 'note': 'Hold'},
    '2026-03-13': {'rate': 0.75, 'prev_rate': 0.75, 'signal': 'HOLD', 'note': 'Hold'},
    '2026-04-24': {'rate': 0.75, 'prev_rate': 0.75, 'signal': 'HOLD', 'note': 'Hold'},
}


# ═══════════════════════════════════════════════════════════════
# SESSION DEFINITIONS (BoJ at ~03:30 UTC)
# Asia Mid-Day release → Europe → US
# ═══════════════════════════════════════════════════════════════

SESSION_CHAIN = [
    # Pre-release
    ('Pre-Asia', [0, 1, 2]),              # 00:00-03:00 UTC (before release)

    # Asia reaction
    ('Release (1h)', [3, 4]),              # 03:00-05:00 (1h post-release window)
    ('Sydney Open', [22, 23, 0, 1]),      # Sydney overlap with early Asia
    ('Tokyo Open', [0, 1, 2]),            # 00:00-03:00 (pre-release positioning)
    ('Asia Mid', [3, 4, 5]),              # 03:00-06:00 (RELEASE WINDOW)
    ('Asia Afternoon', [5, 6, 7]),        # 05:00-08:00 (Asia digest)
    ('Tokyo Close', [6, 7]),              # 06:00-08:00
    ('Pre-London', [7, 8]),               # 07:00-09:00

    # Europe
    ('Frankfurt Open', [7, 8]),
    ('London Open', [8, 9]),
    ('London Morning', [9, 10, 11]),
    ('London Midday', [12, 13]),

    # Overlap (EU-US)
    ('NY Pre-Open', [12, 13]),
    ('NY Open', [13, 14]),
    ('London-NY Overlap', [14, 15]),

    # New York
    ('NY AM', [14, 15, 16]),
    ('NY Lunch', [17]),
    ('NY PM', [18, 19, 20]),

    # Post-release aggregate windows
    ('Release (4h)', [3, 4, 5, 6, 7]),    # 03:00-07:00 (4h post-release)
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


def classify_boj_signal(rate, prev_rate, signal_hint):
    """
    Classify BoJ signal for backtesting.
    HIKE → hawkish → JPY strengthens → carry trade unwind → ETH SHORT
    DOVE → dovish → JPY weakens → carry trade continues → ETH LONG
    HOLD → context-dependent
    """
    if signal_hint in ('HIKE', 'MILD_HIKE'):
        return 'HIKE'
    elif signal_hint in ('BIG_DOVE', 'DOVE'):
        return 'DOVE'
    elif signal_hint in ('YCC_EXPAND',):
        return 'DOVE'
    elif signal_hint in ('YCC_REMOVE', 'NIRP_END'):
        return 'HIKE'
    else:
        # Check rate change
        delta = rate - prev_rate
        if delta > 0:
            return 'HIKE'
        elif delta < 0:
            return 'DOVE'
        return 'HOLD'


def run_backtest(csv_path):
    df = load_eth_data(csv_path)
    results = []
    sorted_dates = sorted(RELEASES.keys())

    for i, date_str in enumerate(sorted_dates):
        release_data = RELEASES[date_str]
        # BoJ announces ~03:30 UTC
        release_dt = datetime.strptime(date_str, '%Y-%m-%d').replace(hour=3, minute=30)

        bars_after = get_bars_between(df, release_dt, release_dt + timedelta(hours=24))
        if len(bars_after) < 4:
            continue

        pre_bar = get_bar_at(df, release_dt)
        if pre_bar is None:
            continue

        pre_price = float(pre_bar['Close'])

        rate = release_data['rate']
        prev_rate = release_data['prev_rate']
        signal_hint = release_data.get('signal', 'HOLD')
        signal = classify_boj_signal(rate, prev_rate, signal_hint)
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
            'rate': rate, 'prev_rate': prev_rate,
            'rate_change': round(rate - prev_rate, 2),
            'signal': signal,
            'signal_hint': signal_hint,
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
    key_sessions = ['Release (1h)', 'Asia Mid', 'Asia Afternoon',
                    'Tokyo Close', 'Pre-London', 'London Open', 'London Morning',
                    'London Midday', 'NY Open', 'London-NY Overlap', 'NY AM', 'NY PM']

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

    print(f"\n{'Transition':<45} {'Same Dir%':>10} {'n':>5} {'Verdict':>10}")
    print("-"*75)
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
            print(f"{s1 + ' → ' + s2:<45} {pct:>9.0f}% {total:>5} {verdict:>10}")
            transitions.append({'from': s1, 'to': s2, 'same_dir_pct': round(pct, 1),
                                'n': total, 'verdict': verdict})
    return chain_data, transitions


def direction_persistence_to_reopen(results):
    """
    Prompt B special: measure % same-direction from each session until
    the reopen of the original session (24h cycle).
    """
    print("\n" + "="*80)
    print("DIRECTION PERSISTENCE: Session → Reopen of Same Session")
    print("="*80)
    valid = results[results['ret_24h'].notna()]

    # For each release day, check if direction at Asia Mid persists through
    # Europe → US → next Asia Mid
    sessions_ordered = [
        'Release (1h)', 'Asia Mid', 'Asia Afternoon',
        'Pre-London', 'London Open', 'London Morning', 'London Midday',
        'NY Open', 'London-NY Overlap', 'NY AM', 'NY PM'
    ]

    print(f"\n{'From Session':<25} {'To Session':<25} {'Same Dir%':>10} {'n':>5} {'Verdict':>10}")
    print("-"*80)

    for i, s1 in enumerate(sessions_ordered):
        # Find the "reopen" — same session next day = skip ahead ~24h
        # For simplicity, compare each session to the final NY PM
        s2 = 'NY PM'  # end of cycle
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
    dove_rets = valid[valid['signal'] == 'DOVE']['ret_24h'].values

    if len(hike_rets) >= 3 and len(hold_rets) >= 3:
        t2, p2 = stats.ttest_ind(hike_rets, hold_rets)
        print(f"\n2. Two-sample t-test (HIKE vs HOLD):")
        print(f"   HIKE: {np.mean(hike_rets):+.3f}% (n={len(hike_rets)})  HOLD: {np.mean(hold_rets):+.3f}% (n={len(hold_rets)})")
        print(f"   t={t2:.3f}  p={p2:.4f}  {'*** SIGNIFICANT' if p2 < 0.05 else 'NOT significant'}")

    if len(dove_rets) >= 3 and len(hold_rets) >= 3:
        t3, p3 = stats.ttest_ind(dove_rets, hold_rets)
        print(f"\n3. Two-sample t-test (DOVE vs HOLD):")
        print(f"   DOVE: {np.mean(dove_rets):+.3f}% (n={len(dove_rets)})  HOLD: {np.mean(hold_rets):+.3f}% (n={len(hold_rets)})")
        print(f"   t={t3:.3f}  p={p3:.4f}  {'*** SIGNIFICANT' if p3 < 0.05 else 'NOT significant'}")

    if len(hike_rets) >= 3 and len(dove_rets) >= 3:
        t4, p4 = stats.ttest_ind(hike_rets, dove_rets)
        print(f"\n4. Two-sample t-test (HIKE vs DOVE):")
        print(f"   HIKE: {np.mean(hike_rets):+.3f}% (n={len(hike_rets)})  DOVE: {np.mean(dove_rets):+.3f}% (n={len(dove_rets)})")
        print(f"   t={t4:.3f}  p={p4:.4f}  {'*** SIGNIFICANT' if p4 < 0.05 else 'NOT significant'}")

    for sig in ['HIKE', 'HOLD', 'DOVE']:
        sig_rets = valid[valid['signal'] == sig]['ret_24h'].values
        if len(sig_rets) >= 3:
            t5, p5 = stats.ttest_1samp(sig_rets, 0)
            print(f"\n5. {sig} one-sample t-test (n={len(sig_rets)}):")
            print(f"   Mean: {np.mean(sig_rets):+.3f}%  t={t5:.3f}  p={p5:.4f}  {'***' if p5 < 0.05 else 'ns'}")

    print(f"\n6. Year-by-year 24h returns:")
    print(f"   {'Year':<6} {'Avg%':>8} {'Win%':>7} {'n':>4}")
    print(f"   {'-'*30}")
    for year in sorted(valid['year'].unique()):
        yr = valid[valid['year'] == year]
        print(f"   {year:<6} {yr['ret_24h'].mean():>+8.3f} {(yr['ret_24h'] > 0).mean()*100:>6.0f}% {len(yr):>4}")


def main():
    csv_path = os.path.join(os.path.dirname(__file__), 'eth_15m_merged.csv')
    print("="*80)
    print("PROMPT A: BOJ RATE DECISION BACKTEST (2018-2026)")
    print("="*80)
    print("Loading ETH 15m data...")
    results = run_backtest(csv_path)
    print(f"Analyzed {len(results)} BoJ rate decisions ({results['year'].min()}-{results['year'].max()})")

    valid = results[results['ret_24h'].notna()]
    print(f"\n24h Aggregate Returns:")
    print(f"  Mean:   {valid['ret_24h'].mean():+.3f}%")
    print(f"  Median: {valid['ret_24h'].median():+.3f}%")
    print(f"  Std:    {valid['ret_24h'].std():.3f}%")
    print(f"  Win%:   {(valid['ret_24h'] > 0).mean()*100:.1f}%")
    print(f"  n:      {len(valid)}")

    print("\n  By Signal:")
    for sig in ['HIKE', 'HOLD', 'DOVE']:
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

    out_path = os.path.join(os.path.dirname(__file__), 'backtest_boj_rate_results.json')
    results.to_json(out_path, orient='records', indent=2)
    print(f"\nResults saved to {out_path}")


if __name__ == '__main__':
    main()

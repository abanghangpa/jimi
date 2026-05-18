"""
Prompt A + B: Backtest FOMC Meeting Minutes Session Transmission Chain
======================================================================
ETH/USDT 15m data from jimi/eth_15m_merged.csv
Event: FOMC Meeting Minutes (18:00 UTC = 14:00 ET = 02:00 MYT+1)
       Released ~3 weeks after each FOMC meeting
Transmission: US Afternoon (18:00 UTC) → Overnight Drift → Asia Re-open

Thesis:
  Released with a 3-week lag, the Minutes reveal the internal debates,
  voting nuances, and policy priorities behind the FOMC decision.
  While rarely triggering violent spikes, surprise revelations of
  internal anxiety over inflation or growth prompt subtle repricing
  via CME FedWatch, causing ETH to drift in line with revised
  liquidity expectations.
  The gradual adjustments play out through quiet overnight hours,
  offering a clear baseline for APAC desks to realign leverage.
  Minutes outline which economic indicators the committee prioritizes,
  helping traders anticipate shifts ahead of the next meeting.

Key asymmetry: Minutes are a GRADUAL repricing event, not a spike.
The edge is in the overnight drift and Asia re-open, not the initial bar.
Hawkish surprise in minutes → gradual yield repricing → ETH drifts lower.
Dovish surprise → rate cut expectations build → ETH drifts higher.

Backtested on FOMC Minutes releases (2018-2026) against ETH/USDT 15m data.
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from scipy import stats
import json
import warnings
warnings.filterwarnings('ignore')

# ═══════════════════════════════════════════════════════════════
# FOMC MEETING MINUTES RELEASE DATES (18:00 UTC = 14:00 ET)
# Released ~3 weeks after each FOMC meeting (typically Tuesday/Wednesday)
# Format: {date: {'meeting_date': str, 'meeting_stance': str,
#                  'minutes_surprise': str, 'key_revelation': str}}
# minutes_surprise: HAWKISH_SURPRISE / DOVISH_SURPRISE / INLINE
# key_revelation: INFLATION_ANXIETY / GROWTH_CONCERN / SPLIT_HAWK /
#                 SPLIT_DOVE / QE_DEBATE / PAUSE_SIGNAL / CUT_SIGNAL /
#                 DATA_DEPENDENT / NONE
# ═══════════════════════════════════════════════════════════════

MINUTES_RELEASES = {
    # ── 2018 ──
    '2018-02-21': {'meeting_date': '2018-01-31', 'meeting_stance': 'NEUTRAL', 'minutes_surprise': 'INLINE', 'key_revelation': 'DATA_DEPENDENT'},
    '2018-04-11': {'meeting_date': '2018-03-21', 'meeting_stance': 'HAWKISH', 'minutes_surprise': 'HAWKISH_SURPRISE', 'key_revelation': 'INFLATION_ANXIETY'},
    '2018-05-23': {'meeting_date': '2018-05-02', 'meeting_stance': 'NEUTRAL', 'minutes_surprise': 'INLINE', 'key_revelation': 'DATA_DEPENDENT'},
    '2018-07-05': {'meeting_date': '2018-06-13', 'meeting_stance': 'HAWKISH', 'minutes_surprise': 'INLINE', 'key_revelation': 'INFLATION_ANXIETY'},
    '2018-08-22': {'meeting_date': '2018-08-01', 'meeting_stance': 'NEUTRAL', 'minutes_surprise': 'INLINE', 'key_revelation': 'DATA_DEPENDENT'},
    '2018-10-17': {'meeting_date': '2018-09-26', 'meeting_stance': 'HAWKISH', 'minutes_surprise': 'HAWKISH_SURPRISE', 'key_revelation': 'INFLATION_ANXIETY'},
    '2018-11-29': {'meeting_date': '2018-11-08', 'meeting_stance': 'NEUTRAL', 'minutes_surprise': 'INLINE', 'key_revelation': 'DATA_DEPENDENT'},
    '2019-01-09': {'meeting_date': '2018-12-19', 'meeting_stance': 'HAWKISH', 'minutes_surprise': 'DOVISH_SURPRISE', 'key_revelation': 'SPLIT_HAWK'},
    # ── 2019 ──
    '2019-02-20': {'meeting_date': '2019-01-30', 'meeting_stance': 'DOVISH', 'minutes_surprise': 'INLINE', 'key_revelation': 'PAUSE_SIGNAL'},
    '2019-04-10': {'meeting_date': '2019-03-20', 'meeting_stance': 'DOVISH', 'minutes_surprise': 'DOVISH_SURPRISE', 'key_revelation': 'GROWTH_CONCERN'},
    '2019-05-22': {'meeting_date': '2019-05-01', 'meeting_stance': 'NEUTRAL', 'minutes_surprise': 'INLINE', 'key_revelation': 'DATA_DEPENDENT'},
    '2019-07-03': {'meeting_date': '2019-06-19', 'meeting_stance': 'DOVISH', 'minutes_surprise': 'DOVISH_SURPRISE', 'key_revelation': 'SPLIT_DOVE'},
    '2019-08-21': {'meeting_date': '2019-07-31', 'meeting_stance': 'DOVISH', 'minutes_surprise': 'INLINE', 'key_revelation': 'CUT_SIGNAL'},
    '2019-10-09': {'meeting_date': '2019-09-18', 'meeting_stance': 'DOVISH', 'minutes_surprise': 'INLINE', 'key_revelation': 'SPLIT_HAWK'},
    '2019-11-20': {'meeting_date': '2019-10-30', 'meeting_stance': 'DOVISH', 'minutes_surprise': 'INLINE', 'key_revelation': 'DATA_DEPENDENT'},
    '2020-01-03': {'meeting_date': '2019-12-11', 'meeting_stance': 'NEUTRAL', 'minutes_surprise': 'INLINE', 'key_revelation': 'DATA_DEPENDENT'},
    # ── 2020 ──
    '2020-02-19': {'meeting_date': '2020-01-29', 'meeting_stance': 'NEUTRAL', 'minutes_surprise': 'INLINE', 'key_revelation': 'DATA_DEPENDENT'},
    '2020-03-22': {'meeting_date': '2020-03-03', 'meeting_stance': 'DOVISH', 'minutes_surprise': 'DOVISH_SURPRISE', 'key_revelation': 'GROWTH_CONCERN'},
    '2020-04-08': {'meeting_date': '2020-03-15', 'meeting_stance': 'DOVISH', 'minutes_surprise': 'INLINE', 'key_revelation': 'QE_DEBATE'},
    '2020-05-20': {'meeting_date': '2020-04-29', 'meeting_stance': 'DOVISH', 'minutes_surprise': 'INLINE', 'key_revelation': 'QE_DEBATE'},
    '2020-07-01': {'meeting_date': '2020-06-10', 'meeting_stance': 'DOVISH', 'minutes_surprise': 'INLINE', 'key_revelation': 'QE_DEBATE'},
    '2020-08-19': {'meeting_date': '2020-07-29', 'meeting_stance': 'DOVISH', 'minutes_surprise': 'INLINE', 'key_revelation': 'QE_DEBATE'},
    '2020-10-07': {'meeting_date': '2020-09-16', 'meeting_stance': 'DOVISH', 'minutes_surprise': 'DOVISH_SURPRISE', 'key_revelation': 'QE_DEBATE'},
    '2020-11-25': {'meeting_date': '2020-11-05', 'meeting_stance': 'NEUTRAL', 'minutes_surprise': 'INLINE', 'key_revelation': 'DATA_DEPENDENT'},
    '2021-01-06': {'meeting_date': '2020-12-16', 'meeting_stance': 'DOVISH', 'minutes_surprise': 'INLINE', 'key_revelation': 'QE_DEBATE'},
    # ── 2021 ──
    '2021-02-17': {'meeting_date': '2021-01-27', 'meeting_stance': 'NEUTRAL', 'minutes_surprise': 'INLINE', 'key_revelation': 'DATA_DEPENDENT'},
    '2021-04-07': {'meeting_date': '2021-03-17', 'meeting_stance': 'NEUTRAL', 'minutes_surprise': 'INLINE', 'key_revelation': 'DATA_DEPENDENT'},
    '2021-05-19': {'meeting_date': '2021-04-28', 'meeting_stance': 'NEUTRAL', 'minutes_surprise': 'INLINE', 'key_revelation': 'DATA_DEPENDENT'},
    '2021-07-07': {'meeting_date': '2021-06-16', 'meeting_stance': 'HAWKISH', 'minutes_surprise': 'HAWKISH_SURPRISE', 'key_revelation': 'INFLATION_ANXIETY'},
    '2021-08-18': {'meeting_date': '2021-07-28', 'meeting_stance': 'NEUTRAL', 'minutes_surprise': 'INLINE', 'key_revelation': 'TAPER_DEBATE'},
    '2021-10-13': {'meeting_date': '2021-09-22', 'meeting_stance': 'HAWKISH', 'minutes_surprise': 'HAWKISH_SURPRISE', 'key_revelation': 'TAPER_DEBATE'},
    '2021-11-24': {'meeting_date': '2021-11-03', 'meeting_stance': 'HAWKISH', 'minutes_surprise': 'INLINE', 'key_revelation': 'TAPER_DEBATE'},
    '2022-01-05': {'meeting_date': '2021-12-15', 'meeting_stance': 'HAWKISH', 'minutes_surprise': 'HAWKISH_SURPRISE', 'key_revelation': 'INFLATION_ANXIETY'},
    # ── 2022 ──
    '2022-02-16': {'meeting_date': '2022-01-26', 'meeting_stance': 'HAWKISH', 'minutes_surprise': 'HAWKISH_SURPRISE', 'key_revelation': 'INFLATION_ANXIETY'},
    '2022-04-06': {'meeting_date': '2022-03-16', 'meeting_stance': 'HAWKISH', 'minutes_surprise': 'HAWKISH_SURPRISE', 'key_revelation': 'INFLATION_ANXIETY'},
    '2022-05-25': {'meeting_date': '2022-05-04', 'meeting_stance': 'HAWKISH', 'minutes_surprise': 'INLINE', 'key_revelation': 'INFLATION_ANXIETY'},
    '2022-07-06': {'meeting_date': '2022-06-15', 'meeting_stance': 'HAWKISH', 'minutes_surprise': 'HAWKISH_SURPRISE', 'key_revelation': 'INFLATION_ANXIETY'},
    '2022-08-17': {'meeting_date': '2022-07-27', 'meeting_stance': 'HAWKISH', 'minutes_surprise': 'DOVISH_SURPRISE', 'key_revelation': 'SPLIT_HAWK'},
    '2022-10-12': {'meeting_date': '2022-09-21', 'meeting_stance': 'HAWKISH', 'minutes_surprise': 'HAWKISH_SURPRISE', 'key_revelation': 'INFLATION_ANXIETY'},
    '2022-11-23': {'meeting_date': '2022-11-02', 'meeting_stance': 'HAWKISH', 'minutes_surprise': 'DOVISH_SURPRISE', 'key_revelation': 'SPLIT_HAWK'},
    '2023-01-04': {'meeting_date': '2022-12-14', 'meeting_stance': 'HAWKISH', 'minutes_surprise': 'INLINE', 'key_revelation': 'DATA_DEPENDENT'},
    # ── 2023 ──
    '2023-02-22': {'meeting_date': '2023-02-01', 'meeting_stance': 'HAWKISH', 'minutes_surprise': 'INLINE', 'key_revelation': 'DATA_DEPENDENT'},
    '2023-04-12': {'meeting_date': '2023-03-22', 'meeting_stance': 'HAWKISH', 'minutes_surprise': 'DOVISH_SURPRISE', 'key_revelation': 'SPLIT_HAWK'},
    '2023-05-24': {'meeting_date': '2023-05-03', 'meeting_stance': 'HAWKISH', 'minutes_surprise': 'INLINE', 'key_revelation': 'DATA_DEPENDENT'},
    '2023-07-05': {'meeting_date': '2023-06-14', 'meeting_stance': 'NEUTRAL', 'minutes_surprise': 'INLINE', 'key_revelation': 'DATA_DEPENDENT'},
    '2023-08-16': {'meeting_date': '2023-07-26', 'meeting_stance': 'HAWKISH', 'minutes_surprise': 'INLINE', 'key_revelation': 'DATA_DEPENDENT'},
    '2023-10-11': {'meeting_date': '2023-09-20', 'meeting_stance': 'HAWKISH', 'minutes_surprise': 'HAWKISH_SURPRISE', 'key_revelation': 'INFLATION_ANXIETY'},
    '2023-11-21': {'meeting_date': '2023-11-01', 'meeting_stance': 'NEUTRAL', 'minutes_surprise': 'INLINE', 'key_revelation': 'DATA_DEPENDENT'},
    '2024-01-03': {'meeting_date': '2023-12-13', 'meeting_stance': 'DOVISH', 'minutes_surprise': 'DOVISH_SURPRISE', 'key_revelation': 'CUT_SIGNAL'},
    # ── 2024 ──
    '2024-02-21': {'meeting_date': '2024-01-31', 'meeting_stance': 'NEUTRAL', 'minutes_surprise': 'INLINE', 'key_revelation': 'DATA_DEPENDENT'},
    '2024-04-10': {'meeting_date': '2024-03-20', 'meeting_stance': 'NEUTRAL', 'minutes_surprise': 'INLINE', 'key_revelation': 'DATA_DEPENDENT'},
    '2024-05-22': {'meeting_date': '2024-05-01', 'meeting_stance': 'HAWKISH', 'minutes_surprise': 'HAWKISH_SURPRISE', 'key_revelation': 'INFLATION_ANXIETY'},
    '2024-07-03': {'meeting_date': '2024-06-12', 'meeting_stance': 'NEUTRAL', 'minutes_surprise': 'INLINE', 'key_revelation': 'DATA_DEPENDENT'},
    '2024-08-21': {'meeting_date': '2024-07-31', 'meeting_stance': 'DOVISH', 'minutes_surprise': 'DOVISH_SURPRISE', 'key_revelation': 'CUT_SIGNAL'},
    '2024-10-09': {'meeting_date': '2024-09-18', 'meeting_stance': 'DOVISH', 'minutes_surprise': 'INLINE', 'key_revelation': 'CUT_SIGNAL'},
    '2024-11-26': {'meeting_date': '2024-11-07', 'meeting_stance': 'NEUTRAL', 'minutes_surprise': 'INLINE', 'key_revelation': 'DATA_DEPENDENT'},
    '2025-01-08': {'meeting_date': '2024-12-18', 'meeting_stance': 'HAWKISH', 'minutes_surprise': 'HAWKISH_SURPRISE', 'key_revelation': 'INFLATION_ANXIETY'},
    # ── 2025 ──
    '2025-02-19': {'meeting_date': '2025-01-29', 'meeting_stance': 'NEUTRAL', 'minutes_surprise': 'INLINE', 'key_revelation': 'DATA_DEPENDENT'},
    '2025-04-09': {'meeting_date': '2025-03-19', 'meeting_stance': 'NEUTRAL', 'minutes_surprise': 'INLINE', 'key_revelation': 'DATA_DEPENDENT'},
    '2025-05-28': {'meeting_date': '2025-05-07', 'meeting_stance': 'NEUTRAL', 'minutes_surprise': 'INLINE', 'key_revelation': 'DATA_DEPENDENT'},
    # ── 2026 (projected) ──
    '2026-02-18': {'meeting_date': '2026-01-28', 'meeting_stance': 'NEUTRAL', 'minutes_surprise': 'INLINE', 'key_revelation': 'DATA_DEPENDENT'},
    '2026-04-08': {'meeting_date': '2026-03-18', 'meeting_stance': 'NEUTRAL', 'minutes_surprise': 'INLINE', 'key_revelation': 'DATA_DEPENDENT'},
}

# Session windows (UTC hours)
SESSIONS = {
    'pre_asia': (21, 0),
    'sydney_open': (0, 1),
    'tokyo_open': (1, 2),
    'asia_mid': (2, 5),
    'asia_afternoon': (5, 7),
    'tokyo_close': (7, 8),
    'pre_london': (8, 8.5),
    'frankfurt_open': (8.5, 9),
    'london_open': (9, 10),
    'london_morning': (10, 12),
    'london_midday': (12, 13),
    'ny_pre_open': (13, 13.5),
    'ny_open': (13.5, 14.5),
    'london_ny_overlap': (14.5, 16),
    'ny_am': (16, 18),
    'ny_lunch': (18, 19),
    'ny_pm': (19, 21),
}

SESSION_ORDER = [
    'ny_lunch', 'ny_pm', 'pre_asia', 'sydney_open', 'tokyo_open',
    'asia_mid', 'asia_afternoon', 'tokyo_close', 'pre_london',
    'frankfurt_open', 'london_open', 'london_morning', 'london_midday',
    'ny_pre_open', 'ny_open', 'london_ny_overlap', 'ny_am',
]


def load_eth_data(filepath):
    df = pd.read_csv(filepath)
    df['Open time'] = pd.to_datetime(df['Open time'])
    df = df.set_index('Open time')
    for c in ['Close', 'Open', 'High', 'Low', 'Volume']:
        df[c] = df[c].astype(float)
    return df


def compute_session_returns(df, release_date, release_utc_hour=18, release_utc_min=0):
    """
    Compute ETH returns for each session relative to Minutes release (18:00 UTC).
    Minutes are a GRADUAL repricing — measure drift, not spike.
    """
    release_dt = pd.Timestamp(f"{release_date} {release_utc_hour:02d}:{release_utc_min:02d}:00")
    release_bars = df.index[df.index >= release_dt]
    if len(release_bars) == 0:
        return None
    release_bar = release_bars[0]
    price_at_release = df.loc[release_bar, 'Close']

    results = {}
    for session_name, (start_h, end_h) in SESSIONS.items():
        start_hour = int(start_h)
        start_min = int((start_h % 1) * 60)

        if start_h >= release_utc_hour + (release_utc_min / 60.0):
            session_start_dt = pd.Timestamp(f"{release_date} {start_hour:02d}:{start_min:02d}:00")
        else:
            next_day = (pd.Timestamp(release_date) + timedelta(days=1)).strftime('%Y-%m-%d')
            session_start_dt = pd.Timestamp(f"{next_day} {start_hour:02d}:{start_min:02d}:00")

        session_bars = df.index[df.index >= session_start_dt]
        if len(session_bars) == 0:
            results[session_name] = None
            continue
        price_at_session = df.loc[session_bars[0], 'Close']
        results[session_name] = (price_at_session - price_at_release) / price_at_release * 100

    # Overnight drift: release → next day 08:00 UTC (Asia mid-session)
    overnight_dt = pd.Timestamp(
        (pd.Timestamp(release_date) + timedelta(days=1)).strftime('%Y-%m-%d') + ' 08:00:00'
    )
    overnight_bars = df.index[df.index >= overnight_dt]
    if len(overnight_bars) > 0:
        price_overnight = df.loc[overnight_bars[0], 'Close']
        results['overnight_drift'] = (price_overnight - price_at_release) / price_at_release * 100

    # 24h aggregate
    end_24h = release_dt + timedelta(hours=24)
    end_bars = df.index[df.index >= end_24h]
    if len(end_bars) > 0:
        price_24h = df.loc[end_bars[0], 'Close']
        results['24h_return'] = (price_24h - price_at_release) / price_at_release * 100
    else:
        results['24h_return'] = None

    return results


def compute_wyckoff_phase(df, date_str, lookback_days=30):
    dt = pd.Timestamp(date_str)
    start = dt - timedelta(days=lookback_days)
    window = df[(df.index >= start) & (df.index < dt)]
    if len(window) < 100:
        return 'RANGE'
    closes = window['Close'].values.astype(float)
    highs = window['High'].values.astype(float)
    lows = window['Low'].values.astype(float)
    sma_short = np.mean(closes[-48:])
    sma_long = np.mean(closes[-192:])
    range_high = np.percentile(highs, 90)
    range_low = np.percentile(lows, 10)
    range_mid = (range_high + range_low) / 2
    range_width = (range_high - range_low) / range_mid
    if range_width < 0.05:
        return 'RANGE'
    if sma_short > sma_long * 1.02:
        return 'MARKUP' if closes[-1] > range_mid else 'DISTRIBUTION'
    elif sma_short < sma_long * 0.98:
        return 'MARKDOWN' if closes[-1] < range_mid else 'ACCUMULATION'
    return 'RANGE'


def compute_vol_regime(df, date_str, lookback_days=90):
    dt = pd.Timestamp(date_str)
    start = dt - timedelta(days=lookback_days)
    window = df[(df.index >= start) & (df.index < dt)]
    if len(window) < 200:
        return 'NEUTRAL'
    daily = window.groupby(window.index.date).agg(
        high=('High', 'max'), low=('Low', 'min'), close=('Close', 'last')
    )
    daily['atr'] = daily['high'] - daily['low']
    if len(daily) < 30:
        return 'NEUTRAL'
    recent_date = pd.Timestamp(date_str).date() - timedelta(days=1)
    recent_row = daily[daily.index <= recent_date]
    if len(recent_row) < 10:
        return 'NEUTRAL'
    recent_atr = recent_row['atr'].iloc[-1]
    median_atr = recent_row['atr'].median()
    if recent_atr > 2 * median_atr:
        return 'HIGH_VOL'
    elif recent_atr < 0.5 * median_atr:
        return 'COMPRESSING'
    return 'NEUTRAL'


def classify_minutes_signal(data):
    """
    Classify Minutes signal.
    Minutes are about REVELATION — what the market didn't know from the statement.
    Key: surprise × revelation_type determines the drift direction.
    """
    surprise = data['minutes_surprise']
    revelation = data['key_revelation']

    if surprise == 'HAWKISH_SURPRISE':
        if revelation == 'INFLATION_ANXIETY':
            return 'MINUTES_HAWK_INFLATION'
        elif revelation == 'TAPER_DEBATE':
            return 'MINUTES_HAWK_TAPER'
        return 'MINUTES_HAWKISH'
    elif surprise == 'DOVISH_SURPRISE':
        if revelation == 'CUT_SIGNAL':
            return 'MINUTES_DOVE_CUT'
        elif revelation == 'GROWTH_CONCERN':
            return 'MINUTES_DOVE_GROWTH'
        elif revelation == 'SPLIT_HAWK':
            return 'MINUTES_DOVE_SPLIT'
        return 'MINUTES_DOVISH'
    else:  # INLINE
        if revelation == 'QE_DEBATE':
            return 'MINUTES_INLINE_QE'
        elif revelation == 'TAPER_DEBATE':
            return 'MINUTES_INLINE_TAPER'
        elif revelation == 'SPLIT_HAWK':
            return 'MINUTES_INLINE_SPLIT'
        return 'MINUTES_NEUTRAL'


def run_backtest():
    print("=" * 80)
    print("PROMPT A: FOMC MEETING MINUTES BACKTEST (2018-2026)")
    print("ETH/USDT 15m data | Release: 18:00 UTC (14:00 ET / 02:00 MYT+1)")
    print("~3 weeks after each FOMC meeting")
    print("=" * 80)

    df = load_eth_data('jimi/eth_15m_merged.csv')
    print(f"\nLoaded {len(df)} bars: {df.index[0]} → {df.index[-1]}")

    all_results = []
    for date_str, data in sorted(MINUTES_RELEASES.items()):
        dt = pd.Timestamp(date_str)
        if dt < df.index[0] or dt > df.index[-1] - timedelta(days=4):
            continue
        returns = compute_session_returns(df, date_str)
        if returns is None or returns.get('24h_return') is None:
            continue

        signal = classify_minutes_signal(data)
        wyckoff = compute_wyckoff_phase(df, date_str)
        vol = compute_vol_regime(df, date_str)

        result = {
            'date': date_str,
            'meeting_date': data['meeting_date'],
            'meeting_stance': data['meeting_stance'],
            'minutes_surprise': data['minutes_surprise'],
            'key_revelation': data['key_revelation'],
            'signal': signal,
            'wyckoff': wyckoff,
            'vol': vol,
            **returns,
        }
        all_results.append(result)

    df_results = pd.DataFrame(all_results)
    print(f"\nAnalyzed {len(df_results)} FOMC Minutes releases")

    # ── Session returns ──
    print("\n" + "=" * 80)
    print("SESSION-BY-SESSION AVERAGE RETURNS (%)")
    print("=" * 80)

    print(f"\n{'Session':<24} {'Avg%':>8} {'Win%':>8} {'N':>6} {'Sig':>6}")
    print("-" * 56)

    for special in ['overnight_drift']:
        if special in df_results.columns:
            valid = df_results[special].dropna()
            if len(valid) > 0:
                avg = valid.mean()
                win = (valid > 0).mean() * 100
                n = len(valid)
                sig = "***" if abs(avg) > 0.5 and n >= 5 else ""
                print(f"{special:<24} {avg:>8.3f} {win:>7.1f}% {n:>5} {sig:>6}")

    for session in SESSION_ORDER:
        if session not in df_results.columns:
            continue
        valid = df_results[session].dropna()
        if len(valid) == 0:
            continue
        avg = valid.mean()
        win = (valid > 0).mean() * 100
        n = len(valid)
        sig = "***" if abs(avg) > 0.5 and n >= 5 else "**" if abs(avg) > 0.3 else ""
        print(f"{session:<24} {avg:>8.3f} {win:>7.1f}% {n:>5} {sig:>6}")

    valid_24h = df_results['24h_return'].dropna()
    print(f"\n{'24h AGGREGATE':<24} {valid_24h.mean():>8.3f} {(valid_24h > 0).mean() * 100:>7.1f}% {len(valid_24h):>5}")

    # ── Cross-tab: Wyckoff × Vol × Signal ──
    print("\n" + "=" * 80)
    print("CROSS-TABULATION: Wyckoff × Vol × Signal → 24h Return")
    print("=" * 80)

    combos = df_results.groupby(['wyckoff', 'vol', 'signal']).agg(
        avg_24h=('24h_return', 'mean'),
        win_rate=('24h_return', lambda x: (x > 0).mean() * 100),
        count=('24h_return', 'count'),
    ).reset_index()
    combos = combos[combos['count'] >= 2].sort_values('avg_24h', key=abs, ascending=False)

    print(f"\n{'Wyckoff':<14} {'Vol':<12} {'Signal':<24} {'Avg24h%':>10} {'Win%':>8} {'N':>5}")
    print("-" * 78)
    for _, row in combos.iterrows():
        edge = "✅" if abs(row['avg_24h']) >= 0.5 else "  "
        print(f"{row['wyckoff']:<14} {row['vol']:<12} {row['signal']:<24} "
              f"{row['avg_24h']:>9.3f}% {row['win_rate']:>7.1f}% {int(row['count']):>4} {edge}")

    # ── Minutes Surprise × Key Revelation ──
    print("\n" + "=" * 80)
    print("MINUTES SURPRISE × KEY REVELATION → 24h Return")
    print("=" * 80)

    combos2 = df_results.groupby(['minutes_surprise', 'key_revelation']).agg(
        avg_24h=('24h_return', 'mean'),
        win_rate=('24h_return', lambda x: (x > 0).mean() * 100),
        count=('24h_return', 'count'),
    ).reset_index()
    combos2 = combos2[combos2['count'] >= 2].sort_values('avg_24h', key=abs, ascending=False)

    print(f"\n{'Surprise':<18} {'Revelation':<18} {'Avg24h%':>10} {'Win%':>8} {'N':>5}")
    print("-" * 58)
    for _, row in combos2.iterrows():
        edge = "✅" if abs(row['avg_24h']) >= 0.5 else "  "
        print(f"{row['minutes_surprise']:<18} {row['key_revelation']:<18} "
              f"{row['avg_24h']:>9.3f}% {row['win_rate']:>7.1f}% {int(row['count']):>4} {edge}")

    # ── Gradual vs spike analysis ──
    print("\n" + "=" * 80)
    print("GRADUAL DRIFT ANALYSIS (Minutes = slow repricing)")
    print("=" * 80)
    if 'overnight_drift' in df_results.columns:
        od = df_results['overnight_drift'].dropna()
        if len(od) > 0:
            print(f"\n  Overnight drift (release → next Asia 08:00):")
            print(f"    Mean: {od.mean():+.3f}%  Win: {(od > 0).mean() * 100:.1f}%  N: {len(od)}")
            # Compare to 24h
            mask = df_results['overnight_drift'].notna() & df_results['24h_return'].notna()
            sub = df_results[mask]
            if len(sub) > 0:
                same = ((sub['overnight_drift'] > 0) & (sub['24h_return'] > 0)) | \
                       ((sub['overnight_drift'] < 0) & (sub['24h_return'] < 0))
                print(f"    Overnight → 24h same direction: {same.mean() * 100:.1f}%")

    return df_results


def run_transmission_chain(df_results):
    """Prompt B: Session transmission — gradual drift chain."""
    print("\n\n" + "=" * 80)
    print("PROMPT B: SESSION TRANSMISSION CHAIN VALIDATION")
    print("=" * 80)

    print("\nDIRECTION PERSISTENCE BETWEEN CONSECUTIVE SESSIONS")
    print("-" * 70)

    valid_sessions = [s for s in SESSION_ORDER
                      if s in df_results.columns and df_results[s].notna().sum() >= 3]
    transitions = []

    for i in range(len(valid_sessions) - 1):
        p1, p2 = valid_sessions[i], valid_sessions[i + 1]
        mask = df_results[p1].notna() & df_results[p2].notna()
        subset = df_results[mask]
        if len(subset) < 3:
            continue
        same_dir = ((subset[p1] > 0) & (subset[p2] > 0)) | ((subset[p1] < 0) & (subset[p2] < 0))
        pct_same = same_dir.mean() * 100
        corr = subset[p1].corr(subset[p2])
        edge_label = ("✅ REAL EDGE" if pct_same > 65
                      else "⚠️  MARGINAL" if pct_same >= 55
                      else "❌ NO CHAIN")
        transitions.append({
            'from': p1, 'to': p2, 'pct_same': pct_same,
            'corr': corr, 'n': len(subset), 'edge': edge_label,
        })
        print(f"  {p1:<24} → {p2:<24} {pct_same:>5.1f}% same  "
              f"(r={corr:>5.2f}, n={len(subset)}) {edge_label}")

    # KEY: overnight drift → Asia sessions
    if 'overnight_drift' in df_results.columns:
        for asia in ['tokyo_open', 'asia_mid', 'asia_afternoon']:
            if asia in df_results.columns:
                mask = df_results['overnight_drift'].notna() & df_results[asia].notna()
                sub = df_results[mask]
                if len(sub) >= 3:
                    same = ((sub['overnight_drift'] > 0) & (sub[asia] > 0)) | \
                           ((sub['overnight_drift'] < 0) & (sub[asia] < 0))
                    pct = same.mean() * 100
                    label = "✅ REAL EDGE" if pct > 65 else "⚠️  MARGINAL" if pct >= 55 else "❌ NO CHAIN"
                    print(f"\n  KEY: overnight_drift → {asia}: {pct:>5.1f}% same (n={len(sub)}) {label}")

    # ── Statistical tests ──
    print("\n\n" + "=" * 80)
    print("STATISTICAL SIGNIFICANCE TESTS")
    print("=" * 80)

    returns_24h = df_results['24h_return'].dropna()
    t_stat, p_value = stats.ttest_1samp(returns_24h, 0)
    print(f"\n1. One-sample t-test (H0: mean 24h return = 0)")
    print(f"   Mean: {returns_24h.mean():.4f}%  t = {t_stat:.4f}, p = {p_value:.4f}")
    print(f"   {'✅ SIGNIFICANT (p<0.05)' if p_value < 0.05 else '❌ NOT significant (p≥0.05)'}")

    # Hawkish vs dovish surprise
    hawk_mask = df_results['minutes_surprise'] == 'HAWKISH_SURPRISE'
    dove_mask = df_results['minutes_surprise'] == 'DOVISH_SURPRISE'
    hawk_returns = df_results.loc[hawk_mask, '24h_return'].dropna()
    dove_returns = df_results.loc[dove_mask, '24h_return'].dropna()
    if len(hawk_returns) >= 3 and len(dove_returns) >= 3:
        t_stat2, p_value2 = stats.ttest_ind(hawk_returns, dove_returns)
        print(f"\n2. Two-sample t-test (HAWKISH vs DOVISH surprise)")
        print(f"   Hawkish mean: {hawk_returns.mean():.4f}% (n={len(hawk_returns)})")
        print(f"   Dovish mean:  {dove_returns.mean():.4f}% (n={len(dove_returns)})")
        print(f"   t = {t_stat2:.4f}, p = {p_value2:.4f}")
        print(f"   {'✅ SIGNIFICANT (p<0.05)' if p_value2 < 0.05 else '❌ NOT significant (p≥0.05)'}")

    # Inflation anxiety vs other revelations
    infl_mask = df_results['key_revelation'] == 'INFLATION_ANXIETY'
    other_mask = df_results['key_revelation'] != 'INFLATION_ANXIETY'
    infl_returns = df_results.loc[infl_mask, '24h_return'].dropna()
    other_returns = df_results.loc[other_mask, '24h_return'].dropna()
    if len(infl_returns) >= 3 and len(other_returns) >= 3:
        t_stat3, p_value3 = stats.ttest_ind(infl_returns, other_returns)
        print(f"\n3. Two-sample t-test (INFLATION_ANXIETY vs other revelations)")
        print(f"   Inflation anxiety mean: {infl_returns.mean():.4f}% (n={len(infl_returns)})")
        print(f"   Other mean:             {other_returns.mean():.4f}% (n={len(other_returns)})")
        print(f"   t = {t_stat3:.4f}, p = {p_value3:.4f}")
        print(f"   {'✅ SIGNIFICANT (p<0.05)' if p_value3 < 0.05 else '❌ NOT significant (p≥0.05)'}")

    return transitions


def main():
    df_results = run_backtest()
    transitions = run_transmission_chain(df_results)

    output = {
        'summary': {
            'total_minutes': len(df_results),
            'mean_24h_return': float(df_results['24h_return'].dropna().mean()),
            'win_rate_24h': float((df_results['24h_return'].dropna() > 0).mean()),
        },
        'releases': df_results.to_dict(orient='records'),
        'transitions': transitions,
    }
    with open('jimi/backtest_fomc_minutes_results.json', 'w') as f:
        json.dump(output, f, indent=2, default=str)
    print(f"\n\nResults saved to jimi/backtest_fomc_minutes_results.json")

    print("\n\n" + "=" * 80)
    print("SUMMARY: EDGE IDENTIFICATION")
    print("=" * 80)

    edge_combos = df_results.groupby(['wyckoff', 'vol', 'signal']).agg(
        avg_24h=('24h_return', 'mean'),
        win_rate=('24h_return', lambda x: (x > 0).mean() * 100),
        count=('24h_return', 'count')
    ).reset_index()
    edge_combos = edge_combos[(edge_combos['count'] >= 2) & (edge_combos['avg_24h'].abs() >= 0.5)]
    edge_combos = edge_combos.sort_values('avg_24h', key=abs, ascending=False)

    if len(edge_combos) > 0:
        print("\nWyckoff × Vol × Signal combos with edge (n≥2, |avg|≥0.5%):")
        for _, row in edge_combos.iterrows():
            direction = "LONG" if row['avg_24h'] > 0 else "SHORT"
            print(f"  {row['wyckoff']} + {row['vol']} + {row['signal']}: "
                  f"{row['avg_24h']:+.3f}% avg, {row['win_rate']:.0f}% win, "
                  f"n={int(row['count'])} → {direction} bias")

    print("\nStrong transmission links (>65% same direction):")
    for t in transitions:
        if t['pct_same'] > 65:
            print(f"  {t['from']} → {t['to']}: {t['pct_same']:.1f}% persist (r={t['corr']:.2f})")

    # Gradual drift summary
    if 'overnight_drift' in df_results.columns:
        od = df_results['overnight_drift'].dropna()
        if len(od) > 0:
            print(f"\nMinutes are GRADUAL repricing (not spike):")
            print(f"  Overnight drift avg: {od.mean():+.3f}%  win: {(od > 0).mean() * 100:.1f}%")
            print(f"  → Clear baseline for APAC desks to realign leverage")

    return df_results


if __name__ == '__main__':
    main()

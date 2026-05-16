#!/usr/bin/env python3
"""
Prompt A + B: Backtest US Housing Starts + Building Permits (2018-today) using ETH/USDT 15m data.

Released by Census Bureau at 08:30 ET (13:30 UTC) around 17th of each month.
Housing Starts: new residential construction begins
Building Permits: forward-looking permits issued

Session itinerary (per user's thesis #17):
  US Morning (13:30 UTC) → Rate Path Evaluation

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
# US HOUSING STARTS + BUILDING PERMITS RELEASE DATES (08:30 ET = 13:30 UTC)
# Format: {date: {'starts_k': float, 'permits_k', 'starts_mom': float, 'permits_mom': float}}
# starts_k = housing starts in thousands (SAAR)
# permits_k = building permits in thousands (SAAR)
# starts_mom/permits_mom = month-over-month % change
# ═══════════════════════════════════════════════════════════════

RELEASES = {
    # 2018
    '2018-01-18': {'starts_k': 1326, 'permits_k': 1396, 'starts_mom': -0.7, 'permits_mom': -0.1},
    '2018-02-16': {'starts_k': 1230, 'permits_k': 1377, 'starts_mom': -7.2, 'permits_mom': -1.4},
    '2018-03-16': {'starts_k': 1310, 'permits_k': 1321, 'starts_mom': 6.5, 'permits_mom': -4.1},
    '2018-04-17': {'starts_k': 1319, 'permits_k': 1352, 'starts_mom': 0.7, 'permits_mom': 2.3},
    '2018-05-17': {'starts_k': 1287, 'permits_k': 1352, 'starts_mom': -2.4, 'permits_mom': 0.0},
    '2018-06-19': {'starts_k': 1350, 'permits_k': 1301, 'starts_mom': 4.9, 'permits_mom': -3.8},
    '2018-07-18': {'starts_k': 1173, 'permits_k': 1275, 'starts_mom': -13.1, 'permits_mom': -2.0},
    '2018-08-16': {'starts_k': 1282, 'permits_k': 1311, 'starts_mom': 9.3, 'permits_mom': 2.8},
    '2018-09-19': {'starts_k': 1201, 'permits_k': 1241, 'starts_mom': -6.3, 'permits_mom': -5.3},
    '2018-10-18': {'starts_k': 1228, 'permits_k': 1263, 'starts_mom': 2.2, 'permits_mom': 1.8},
    '2018-11-20': {'starts_k': 1228, 'permits_k': 1289, 'starts_mom': 0.0, 'permits_mom': 2.1},
    '2018-12-19': {'starts_k': 1256, 'permits_k': 1328, 'starts_mom': 2.3, 'permits_mom': 3.0},
    # 2019
    '2019-01-17': {'starts_k': 1078, 'permits_k': 1326, 'starts_mom': -14.2, 'permits_mom': -0.2},
    '2019-02-20': {'starts_k': 1162, 'permits_k': 1297, 'starts_mom': 7.8, 'permits_mom': -2.2},
    '2019-03-19': {'starts_k': 1139, 'permits_k': 1278, 'starts_mom': -2.0, 'permits_mom': -1.5},
    '2019-04-17': {'starts_k': 1235, 'permits_k': 1296, 'starts_mom': 8.4, 'permits_mom': 1.4},
    '2019-05-16': {'starts_k': 1235, 'permits_k': 1294, 'starts_mom': 0.0, 'permits_mom': -0.2},
    '2019-06-18': {'starts_k': 1269, 'permits_k': 1299, 'starts_mom': 2.8, 'permits_mom': 0.4},
    '2019-07-17': {'starts_k': 1253, 'permits_k': 1326, 'starts_mom': -1.3, 'permits_mom': 2.1},
    '2019-08-16': {'starts_k': 1364, 'permits_k': 1381, 'starts_mom': 8.9, 'permits_mom': 4.1},
    '2019-09-18': {'starts_k': 1362, 'permits_k': 1419, 'starts_mom': -0.1, 'permits_mom': 2.8},
    '2019-10-17': {'starts_k': 1314, 'permits_k': 1387, 'starts_mom': -3.5, 'permits_mom': -2.3},
    '2019-11-19': {'starts_k': 1343, 'permits_k': 1482, 'starts_mom': 2.2, 'permits_mom': 6.8},
    '2019-12-18': {'starts_k': 1608, 'permits_k': 1474, 'starts_mom': 19.7, 'permits_mom': -0.5},
    # 2020
    '2020-01-17': {'starts_k': 1567, 'permits_k': 1615, 'starts_mom': -2.6, 'permits_mom': 9.6},
    '2020-02-19': {'starts_k': 1599, 'permits_k': 1553, 'starts_mom': 2.0, 'permits_mom': -3.8},
    '2020-03-18': {'starts_k': 1499, 'permits_k': 1353, 'starts_mom': -6.3, 'permits_mom': -12.9},
    '2020-04-16': {'starts_k': 1056, 'permits_k': 1074, 'starts_mom': -29.6, 'permits_mom': -20.6},
    '2020-05-19': {'starts_k': 1010, 'permits_k': 1123, 'starts_mom': -4.4, 'permits_mom': 4.6},
    '2020-06-17': {'starts_k': 1186, 'permits_k': 1220, 'starts_mom': 17.4, 'permits_mom': 8.6},
    '2020-07-17': {'starts_k': 1496, 'permits_k': 1495, 'starts_mom': 26.1, 'permits_mom': 22.5},
    '2020-08-18': {'starts_k': 1483, 'permits_k': 1521, 'starts_mom': -0.9, 'permits_mom': 1.7},
    '2020-09-17': {'starts_k': 1415, 'permits_k': 1553, 'starts_mom': -4.6, 'permits_mom': 2.1},
    '2020-10-20': {'starts_k': 1530, 'permits_k': 1620, 'starts_mom': 8.1, 'permits_mom': 4.3},
    '2020-11-18': {'starts_k': 1547, 'permits_k': 1639, 'starts_mom': 1.1, 'permits_mom': 1.2},
    '2020-12-17': {'starts_k': 1580, 'permits_k': 1685, 'starts_mom': 2.1, 'permits_mom': 2.8},
    # 2021
    '2021-01-21': {'starts_k': 1580, 'permits_k': 1766, 'starts_mom': 0.0, 'permits_mom': 4.8},
    '2021-02-18': {'starts_k': 1537, 'permits_k': 1738, 'starts_mom': -2.7, 'permits_mom': -1.6},
    '2021-03-17': {'starts_k': 1537, 'permits_k': 1774, 'starts_mom': 0.0, 'permits_mom': 2.1},
    '2021-04-16': {'starts_k': 1569, 'permits_k': 1760, 'starts_mom': 2.1, 'permits_mom': -0.8},
    '2021-05-18': {'starts_k': 1528, 'permits_k': 1681, 'starts_mom': -2.6, 'permits_mom': -4.5},
    '2021-06-17': {'starts_k': 1643, 'permits_k': 1683, 'starts_mom': 7.5, 'permits_mom': 0.1},
    '2021-07-20': {'starts_k': 1534, 'permits_k': 1635, 'starts_mom': -6.6, 'permits_mom': -2.8},
    '2021-08-18': {'starts_k': 1617, 'permits_k': 1728, 'starts_mom': 5.4, 'permits_mom': 5.7},
    '2021-09-21': {'starts_k': 1555, 'permits_k': 1589, 'starts_mom': -3.8, 'permits_mom': -8.0},
    '2021-10-19': {'starts_k': 1520, 'permits_k': 1645, 'starts_mom': -2.3, 'permits_mom': 3.5},
    '2021-11-17': {'starts_k': 1679, 'permits_k': 1712, 'starts_mom': 10.5, 'permits_mom': 4.1},
    '2021-12-16': {'starts_k': 1608, 'permits_k': 1713, 'starts_mom': -4.2, 'permits_mom': 0.1},
    # 2022
    '2022-01-19': {'starts_k': 1673, 'permits_k': 1813, 'starts_mom': 4.0, 'permits_mom': 5.8},
    '2022-02-17': {'starts_k': 1769, 'permits_k': 1859, 'starts_mom': 5.7, 'permits_mom': 2.5},
    '2022-03-17': {'starts_k': 1793, 'permits_k': 1865, 'starts_mom': 1.4, 'permits_mom': 0.3},
    '2022-04-19': {'starts_k': 1724, 'permits_k': 1819, 'starts_mom': -3.8, 'permits_mom': -2.5},
    '2022-05-18': {'starts_k': 1549, 'permits_k': 1744, 'starts_mom': -10.2, 'permits_mom': -4.1},
    '2022-06-17': {'starts_k': 1559, 'permits_k': 1685, 'starts_mom': 0.6, 'permits_mom': -3.4},
    '2022-07-19': {'starts_k': 1534, 'permits_k': 1685, 'starts_mom': -1.6, 'permits_mom': 0.0},
    '2022-08-16': {'starts_k': 1517, 'permits_k': 1600, 'starts_mom': -1.1, 'permits_mom': -5.0},
    '2022-09-20': {'starts_k': 1439, 'permits_k': 1542, 'starts_mom': -5.1, 'permits_mom': -3.6},
    '2022-10-19': {'starts_k': 1425, 'permits_k': 1526, 'starts_mom': -1.0, 'permits_mom': -1.0},
    '2022-11-17': {'starts_k': 1427, 'permits_k': 1492, 'starts_mom': 0.1, 'permits_mom': -2.2},
    '2022-12-20': {'starts_k': 1382, 'permits_k': 1330, 'starts_mom': -3.2, 'permits_mom': -10.9},
    # 2023
    '2023-01-19': {'starts_k': 1309, 'permits_k': 1370, 'starts_mom': -5.3, 'permits_mom': 3.0},
    '2023-02-16': {'starts_k': 1450, 'permits_k': 1462, 'starts_mom': 10.8, 'permits_mom': 6.7},
    '2023-03-16': {'starts_k': 1420, 'permits_k': 1413, 'starts_mom': -2.1, 'permits_mom': -3.4},
    '2023-04-18': {'starts_k': 1401, 'permits_k': 1416, 'starts_mom': -1.3, 'permits_mom': 0.2},
    '2023-05-17': {'starts_k': 1631, 'permits_k': 1416, 'starts_mom': 16.4, 'permits_mom': 0.0},
    '2023-06-20': {'starts_k': 1631, 'permits_k': 1491, 'starts_mom': 0.0, 'permits_mom': 5.3},
    '2023-07-19': {'starts_k': 1434, 'permits_k': 1442, 'starts_mom': -12.1, 'permits_mom': -3.3},
    '2023-08-16': {'starts_k': 1283, 'permits_k': 1443, 'starts_mom': -10.5, 'permits_mom': 0.1},
    '2023-09-19': {'starts_k': 1358, 'permits_k': 1541, 'starts_mom': 5.8, 'permits_mom': 6.8},
    '2023-10-18': {'starts_k': 1372, 'permits_k': 1487, 'starts_mom': 1.0, 'permits_mom': -3.5},
    '2023-11-17': {'starts_k': 1560, 'permits_k': 1460, 'starts_mom': 13.7, 'permits_mom': -1.8},
    '2023-12-19': {'starts_k': 1521, 'permits_k': 1495, 'starts_mom': -2.5, 'permits_mom': 2.4},
    # 2024
    '2024-01-18': {'starts_k': 1460, 'permits_k': 1515, 'starts_mom': -4.0, 'permits_mom': 1.3},
    '2024-02-16': {'starts_k': 1521, 'permits_k': 1489, 'starts_mom': 4.2, 'permits_mom': -1.7},
    '2024-03-19': {'starts_k': 1518, 'permits_k': 1524, 'starts_mom': -0.2, 'permits_mom': 2.4},
    '2024-04-16': {'starts_k': 1446, 'permits_k': 1440, 'starts_mom': -4.7, 'permits_mom': -5.5},
    '2024-05-16': {'starts_k': 1277, 'permits_k': 1440, 'starts_mom': -11.7, 'permits_mom': 0.0},
    '2024-06-20': {'starts_k': 1277, 'permits_k': 1399, 'starts_mom': 0.0, 'permits_mom': -2.8},
    '2024-07-17': {'starts_k': 1354, 'permits_k': 1401, 'starts_mom': 6.0, 'permits_mom': 0.1},
    '2024-08-16': {'starts_k': 1356, 'permits_k': 1475, 'starts_mom': 0.1, 'permits_mom': 5.3},
    '2024-09-18': {'starts_k': 1307, 'permits_k': 1428, 'starts_mom': -3.6, 'permits_mom': -3.2},
    '2024-10-18': {'starts_k': 1311, 'permits_k': 1439, 'starts_mom': 0.3, 'permits_mom': 0.8},
    '2024-11-19': {'starts_k': 1272, 'permits_k': 1430, 'starts_mom': -3.0, 'permits_mom': -0.6},
    '2024-12-18': {'starts_k': 1297, 'permits_k': 1483, 'starts_mom': 2.0, 'permits_mom': 3.7},
    # 2025
    '2025-01-17': {'starts_k': 1483, 'permits_k': 1483, 'starts_mom': 14.3, 'permits_mom': 0.0},
    '2025-02-19': {'starts_k': 1501, 'permits_k': 1456, 'starts_mom': 1.2, 'permits_mom': -1.8},
    '2025-03-18': {'starts_k': 1432, 'permits_k': 1457, 'starts_mom': -4.6, 'permits_mom': 0.1},
    '2025-04-17': {'starts_k': 1361, 'permits_k': 1412, 'starts_mom': -4.9, 'permits_mom': -3.1},
    '2025-05-16': {'starts_k': 1337, 'permits_k': 1393, 'starts_mom': -1.8, 'permits_mom': -1.3},
    '2025-06-18': {'starts_k': 1321, 'permits_k': 1393, 'starts_mom': -1.2, 'permits_mom': 0.0},
    '2025-07-17': {'starts_k': 1354, 'permits_k': 1354, 'starts_mom': 2.5, 'permits_mom': -2.8},
    '2025-08-19': {'starts_k': 1331, 'permits_k': 1342, 'starts_mom': -1.7, 'permits_mom': -0.9},
    '2025-09-17': {'starts_k': 1314, 'permits_k': 1329, 'starts_mom': -1.3, 'permits_mom': -1.0},
    '2025-10-17': {'starts_k': 1342, 'permits_k': 1351, 'starts_mom': 2.1, 'permits_mom': 1.7},
    '2025-11-19': {'starts_k': 1327, 'permits_k': 1338, 'starts_mom': -1.1, 'permits_mom': -1.0},
    '2025-12-17': {'starts_k': 1353, 'permits_k': 1364, 'starts_mom': 2.0, 'permits_mom': 1.9},
    # 2026
    '2026-01-20': {'starts_k': 1360, 'permits_k': 1370, 'starts_mom': 0.5, 'permits_mom': 0.4},
    '2026-02-18': {'starts_k': 1345, 'permits_k': 1355, 'starts_mom': -1.1, 'permits_mom': -1.1},
    '2026-03-17': {'starts_k': 1368, 'permits_k': 1380, 'starts_mom': 1.7, 'permits_mom': 1.8},
    '2026-04-16': {'starts_k': 1335, 'permits_k': 1345, 'starts_mom': -2.4, 'permits_mom': -2.5},
    '2026-05-19': {'starts_k': 1350, 'permits_k': 1360, 'starts_mom': 1.1, 'permits_mom': 1.1},
}


# ═══════════════════════════════════════════════════════════════
# SESSION DEFINITIONS (US Housing at 13:30 UTC)
# Same as Retail Sales — 08:30 ET release
# ═══════════════════════════════════════════════════════════════

SESSION_CHAIN = [
    # Pre-Asia
    ('Pre-Asia', [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]),

    # Asia
    ('Sydney Open', [22, 23, 0]),
    ('Tokyo Open', [0, 1, 2]),
    ('Asia Mid', [3, 4]),
    ('Asia Afternoon', [5, 6]),
    ('Tokyo Close', [6, 7]),
    ('Pre-London', [7, 8]),

    # Europe
    ('Frankfurt Open', [7, 8]),
    ('London Open', [8, 9]),
    ('London Morning', [9, 10, 11]),
    ('London Midday', [12, 13]),

    # Overlap (EU-US) — release happens here
    ('NY Pre-Open', [12, 13]),
    ('NY Open', [13, 14]),
    ('London-NY Overlap', [14, 15]),

    # New York
    ('NY AM', [14, 15, 16]),
    ('NY Lunch', [17]),
    ('NY PM', [18, 19, 20]),

    # Post-release windows
    ('Release (1h)', [13, 14]),
    ('Release (4h)', [13, 14, 15, 16, 17]),
]


def load_eth_data(csv_path):
    """Load and prepare ETH 15m data."""
    df = pd.read_csv(csv_path)
    df['Open time'] = pd.to_datetime(df['Open time'])
    df = df.sort_values('Open time').reset_index(drop=True)
    return df


def get_bar_at(df, dt):
    """Get the bar that contains or is closest before dt."""
    mask = df['Open time'] <= dt
    if mask.sum() == 0:
        return None
    return df[mask].iloc[-1]


def get_bars_between(df, start, end):
    """Get bars between two datetimes."""
    mask = (df['Open time'] >= start) & (df['Open time'] < end)
    return df[mask]


def get_return(df, start_dt, end_dt):
    """Calculate return between two timestamps."""
    bar_start = get_bar_at(df, start_dt)
    bars_end = get_bars_between(df, start_dt, end_dt)
    if bar_start is None or len(bars_end) == 0:
        return None
    start_price = float(bar_start['Close'])
    end_price = float(bars_end.iloc[-1]['Close'])
    return (end_price - start_price) / start_price * 100


def classify_wyckoff_phase(df, release_dt, lookback=96):
    """Simple Wyckoff phase classification based on recent price action."""
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
    """Classify volatility regime."""
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


def classify_signal(starts_mom, permits_mom):
    """Classify the housing signal.
    Housing is a rate-sensitive sector:
      - Collapse → economy breaking → DXY drops → ETH up (dovish bets)
      - Surge → economy hot → DXY up → ETH down (hawkish bets)
    """
    # Average the two signals
    avg_mom = (starts_mom + permits_mom) / 2

    if avg_mom < -5.0:
        return 'COLLAPSE', avg_mom
    elif avg_mom < -2.0:
        return 'WEAK', avg_mom
    elif avg_mom < 2.0:
        return 'STABLE', avg_mom
    elif avg_mom < 5.0:
        return 'STRONG', avg_mom
    else:
        return 'SURGE', avg_mom


def run_backtest(csv_path):
    """Run the full backtest."""
    df = load_eth_data(csv_path)
    results = []

    sorted_dates = sorted(RELEASES.keys())

    for i, date_str in enumerate(sorted_dates):
        release_data = RELEASES[date_str]
        release_dt = datetime.strptime(date_str, '%Y-%m-%d').replace(hour=13, minute=30)

        # Skip if no data available
        bars_after = get_bars_between(df, release_dt, release_dt + timedelta(hours=24))
        if len(bars_after) < 4:
            continue

        # Pre-release price
        pre_bar = get_bar_at(df, release_dt)
        if pre_bar is None:
            continue
        pre_price = float(pre_bar['Close'])

        # Classify
        starts_mom = release_data['starts_mom']
        permits_mom = release_data['permits_mom']
        signal, avg_mom = classify_signal(starts_mom, permits_mom)

        wyckoff = classify_wyckoff_phase(df, release_dt)
        vol_regime = classify_vol_regime(df, release_dt)

        # Session returns
        session_returns = {}
        for name, hours in SESSION_CHAIN:
            if len(hours) == 1:
                start = release_dt.replace(hour=hours[0], minute=0)
                end = start + timedelta(hours=1)
            else:
                start = release_dt.replace(hour=hours[0], minute=0)
                end = release_dt.replace(hour=hours[-1] + 1, minute=0)

            # Handle cross-day
            if end <= start:
                end += timedelta(days=1)

            ret = get_return(df, start, end)
            if ret is not None:
                session_returns[name] = round(ret, 4)

        # 24h aggregate
        ret_24h = get_return(df, release_dt, release_dt + timedelta(hours=24))
        ret_48h = get_return(df, release_dt, release_dt + timedelta(hours=48))

        year = int(date_str[:4])

        results.append({
            'date': date_str,
            'year': year,
            'starts_k': release_data['starts_k'],
            'permits_k': release_data['permits_k'],
            'starts_mom': starts_mom,
            'permits_mom': permits_mom,
            'avg_mom': round(avg_mom, 2),
            'signal': signal,
            'wyckoff': wyckoff,
            'vol_regime': vol_regime,
            'pre_price': round(pre_price, 2),
            'ret_24h': round(ret_24h, 4) if ret_24h is not None else None,
            'ret_48h': round(ret_48h, 4) if ret_48h is not None else None,
            'session_returns': session_returns,
        })

    return pd.DataFrame(results)


def cross_tabulate(results):
    """Cross-tabulate: (signal × vol × wyckoff) → avg 24h return, win rate, n."""
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
        med_ret = group['ret_24h'].median()
        std_ret = group['ret_24h'].std()
        rows.append({
            'signal': sig, 'vol': vol, 'wyckoff': wy,
            'n': n, 'avg_24h': round(avg_ret, 3),
            'win_rate': round(win_rate, 3),
            'median_24h': round(med_ret, 3),
            'std': round(std_ret, 3) if n > 1 else 0,
        })

    df = pd.DataFrame(rows).sort_values('n', ascending=False)
    print(df.to_string(index=False))

    # Filter for edges (n>=3, |avg|>=0.5%)
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
    """Analyze session-by-session transmission chain (Prompt B)."""
    print("\n" + "="*80)
    print("SESSION TRANSMISSION CHAIN (Prompt B)")
    print("="*80)

    valid = results[results['ret_24h'].notna()]

    key_sessions = [
        'Release (1h)', 'NY Open', 'London-NY Overlap',
        'NY AM', 'NY Lunch', 'NY PM',
    ]

    print(f"\n{'Session':<25} {'Avg Ret%':>10} {'Win%':>8} {'n':>5} {'Dir':>6}")
    print("-"*60)

    chain_data = []
    for session_name in key_sessions:
        rets = []
        for _, row in valid.iterrows():
            sr = row.get('session_returns', {})
            if session_name in sr:
                rets.append(sr[session_name])

        if len(rets) < 3:
            continue

        avg = np.mean(rets)
        win = sum(1 for r in rets if r > 0) / len(rets) * 100
        n = len(rets)
        curr_dir = 'UP' if avg > 0.05 else 'DOWN' if avg < -0.05 else 'FLAT'

        print(f"{session_name:<25} {avg:>+10.3f} {win:>7.0f}% {n:>5} {curr_dir:>6}")

        chain_data.append({
            'session': session_name,
            'avg_ret': round(avg, 3),
            'win_pct': round(win, 1),
            'n': n,
            'direction': curr_dir,
        })

    # Direction persistence between consecutive sessions
    print(f"\n{'Transition':<40} {'Same Dir%':>10} {'n':>5} {'Verdict':>10}")
    print("-"*70)

    transitions = []
    for i in range(len(key_sessions) - 1):
        s1 = key_sessions[i]
        s2 = key_sessions[i + 1]

        same_count = 0
        total = 0
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
            print(f"{s1 + ' → ' + s2:<40} {pct:>9.0f}% {total:>5} {verdict:>10}")
            transitions.append({
                'from': s1, 'to': s2,
                'same_dir_pct': round(pct, 1),
                'n': total,
                'verdict': verdict,
            })

    return chain_data, transitions


def statistical_tests(results):
    """Statistical significance tests."""
    print("\n" + "="*80)
    print("STATISTICAL SIGNIFICANCE")
    print("="*80)

    valid = results[results['ret_24h'].notna()]

    # 1. One-sample t-test
    rets = valid['ret_24h'].values
    t_stat, p_val = stats.ttest_1samp(rets, 0)
    print(f"\n1. One-sample t-test (24h return vs 0):")
    print(f"   Mean: {np.mean(rets):+.3f}%  t={t_stat:.3f}  p={p_val:.4f}  {'*** SIGNIFICANT' if p_val < 0.05 else 'NOT significant'}")

    # 2. Two-sample t-test: WEAK/COLLAPSE vs STRONG/SURGE
    weak = valid[valid['signal'].isin(['WEAK', 'COLLAPSE'])]['ret_24h'].values
    strong = valid[valid['signal'].isin(['STRONG', 'SURGE'])]['ret_24h'].values

    if len(weak) >= 3 and len(strong) >= 3:
        t2, p2 = stats.ttest_ind(weak, strong)
        print(f"\n2. Two-sample t-test (WEAK/COLLAPSE vs STRONG/SURGE):")
        print(f"   WEAK: {np.mean(weak):+.3f}% (n={len(weak)})  STRONG: {np.mean(strong):+.3f}% (n={len(strong)})")
        print(f"   t={t2:.3f}  p={p2:.4f}  {'*** SIGNIFICANT' if p2 < 0.05 else 'NOT significant'}")

    # 3. By year
    print(f"\n3. Year-by-year 24h returns:")
    print(f"   {'Year':<6} {'Avg%':>8} {'Win%':>7} {'n':>4}")
    print(f"   {'-'*30}")
    for year in sorted(valid['year'].unique()):
        yr = valid[valid['year'] == year]
        avg = yr['ret_24h'].mean()
        win = (yr['ret_24h'] > 0).mean() * 100
        n = len(yr)
        print(f"   {year:<6} {avg:>+8.3f} {win:>6.0f}% {n:>4}")


def main():
    csv_path = os.path.join(os.path.dirname(__file__), 'eth_15m_merged.csv')
    print("="*80)
    print("PROMPT A: US HOUSING STARTS + BUILDING PERMITS BACKTEST (2018-2026)")
    print("="*80)
    print("Loading ETH 15m data...")
    results = run_backtest(csv_path)
    print(f"Analyzed {len(results)} Housing releases ({results['year'].min()}-{results['year'].max()})")

    # Basic stats
    valid = results[results['ret_24h'].notna()]
    print(f"\n24h Aggregate Returns:")
    print(f"  Mean:   {valid['ret_24h'].mean():+.3f}%")
    print(f"  Median: {valid['ret_24h'].median():+.3f}%")
    print(f"  Std:    {valid['ret_24h'].std():.3f}%")
    print(f"  Win%:   {(valid['ret_24h'] > 0).mean()*100:.1f}%")
    print(f"  n:      {len(valid)}")

    # Signal breakdown
    print("\n  By Signal:")
    for sig in ['COLLAPSE', 'WEAK', 'STABLE', 'STRONG', 'SURGE']:
        sig_data = valid[valid['signal'] == sig]
        if len(sig_data) > 0:
            avg = sig_data['ret_24h'].mean()
            win = (sig_data['ret_24h'] > 0).mean() * 100
            n = len(sig_data)
            print(f"  {sig:<15} avg={avg:+.3f}%  win={win:.0f}%  n={n}")

    # Cross-tabulation
    cross_tab = cross_tabulate(results)

    # Transmission chain (Prompt B)
    print("\n" + "="*80)
    print("PROMPT B: SESSION TRANSMISSION CHAIN VALIDATION")
    print("="*80)
    chain, transitions = transmission_chain(results)

    # Statistical tests
    statistical_tests(results)

    # Save results
    out_path = os.path.join(os.path.dirname(__file__), 'backtest_us_housing_results.json')
    results.to_json(out_path, orient='records', indent=2)
    print(f"\nResults saved to {out_path}")

    return results


if __name__ == '__main__':
    main()

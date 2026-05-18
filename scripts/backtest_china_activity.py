#!/usr/bin/env python3
"""
Prompt A + B + C: Backtest China NBS Monthly Activity Data (IP, Retail, FAI, Unemp, House)
on ETH/USDT 15m from 2018 to today.

Prompt A: Full backtest with session returns, Wyckoff/vol classification, cross-tabs
Prompt B: Session transmission chain validation
Prompt C: Integration scaffold
"""

import sys, os, json, time
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import numpy as np
import pandas as pd
from datetime import datetime, timedelta, timezone
from scipy import stats

# ═══════════════════════════════════════════════════════════════
# CHINA NBS ACTIVITY RELEASE DATES (2018-2026)
# Released ~15th-16th of each month at 10:00 Beijing (02:00 UTC)
# Covers: IP YoY, Retail Sales YoY, FAI YTD YoY, Unemployment, House Prices
# ═══════════════════════════════════════════════════════════════

RELEASES = {
    # 2018
    '2018-01-18': {'month': 'Dec-17', 'ip_yoy': 6.2, 'retail_yoy': 9.4, 'fai_ytd_yoy': 7.2, 'unemp': 4.98, 'house_price_yoy': 5.7},
    '2018-02-28': {'month': 'Jan', 'ip_yoy': None, 'retail_yoy': None, 'fai_ytd_yoy': None, 'unemp': 4.99},
    '2018-03-14': {'month': 'Feb', 'ip_yoy': None, 'retail_yoy': None, 'fai_ytd_yoy': None, 'unemp': 5.0},
    '2018-04-17': {'month': 'Mar', 'ip_yoy': 6.0, 'retail_yoy': 10.1, 'fai_ytd_yoy': 7.5, 'unemp': 5.0},
    '2018-05-15': {'month': 'Apr', 'ip_yoy': 7.0, 'retail_yoy': 9.4, 'fai_ytd_yoy': 7.0, 'unemp': 4.9},
    '2018-06-14': {'month': 'May', 'ip_yoy': 6.8, 'retail_yoy': 8.5, 'fai_ytd_yoy': 6.1, 'unemp': 4.8},
    '2018-07-16': {'month': 'Jun', 'ip_yoy': 6.0, 'retail_yoy': 9.0, 'fai_ytd_yoy': 6.0, 'unemp': 4.8},
    '2018-08-14': {'month': 'Jul', 'ip_yoy': 6.0, 'retail_yoy': 8.8, 'fai_ytd_yoy': 5.5, 'unemp': 5.1},
    '2018-09-14': {'month': 'Aug', 'ip_yoy': 6.1, 'retail_yoy': 9.0, 'fai_ytd_yoy': 5.3, 'unemp': 5.0},
    '2018-10-19': {'month': 'Sep', 'ip_yoy': 5.8, 'retail_yoy': 9.2, 'fai_ytd_yoy': 5.4, 'unemp': 4.9},
    '2018-11-14': {'month': 'Oct', 'ip_yoy': 5.9, 'retail_yoy': 8.6, 'fai_ytd_yoy': 5.7, 'unemp': 4.9},
    '2018-12-14': {'month': 'Nov', 'ip_yoy': 5.4, 'retail_yoy': 8.1, 'fai_ytd_yoy': 5.9, 'unemp': 4.8},
    # 2019
    '2019-01-21': {'month': 'Dec-18', 'ip_yoy': 5.7, 'retail_yoy': 8.2, 'fai_ytd_yoy': 5.9, 'unemp': 4.9},
    '2019-03-14': {'month': 'Feb', 'ip_yoy': None, 'retail_yoy': None, 'fai_ytd_yoy': None, 'unemp': 5.3},
    '2019-04-17': {'month': 'Mar', 'ip_yoy': 8.5, 'retail_yoy': 8.7, 'fai_ytd_yoy': 6.3, 'unemp': 5.2},
    '2019-05-15': {'month': 'Apr', 'ip_yoy': 5.4, 'retail_yoy': 7.2, 'fai_ytd_yoy': 6.1, 'unemp': 5.0},
    '2019-06-14': {'month': 'May', 'ip_yoy': 5.0, 'retail_yoy': 8.6, 'fai_ytd_yoy': 5.6, 'unemp': 5.0},
    '2019-07-15': {'month': 'Jun', 'ip_yoy': 6.3, 'retail_yoy': 9.8, 'fai_ytd_yoy': 5.8, 'unemp': 5.1},
    '2019-08-14': {'month': 'Jul', 'ip_yoy': 4.8, 'retail_yoy': 7.6, 'fai_ytd_yoy': 5.7, 'unemp': 5.3},
    '2019-09-16': {'month': 'Aug', 'ip_yoy': 4.4, 'retail_yoy': 7.5, 'fai_ytd_yoy': 5.5, 'unemp': 5.2},
    '2019-10-18': {'month': 'Sep', 'ip_yoy': 5.8, 'retail_yoy': 7.8, 'fai_ytd_yoy': 5.4, 'unemp': 5.2},
    '2019-11-14': {'month': 'Oct', 'ip_yoy': 4.7, 'retail_yoy': 7.2, 'fai_ytd_yoy': 5.2, 'unemp': 5.1},
    '2019-12-16': {'month': 'Nov', 'ip_yoy': 6.2, 'retail_yoy': 8.0, 'fai_ytd_yoy': 5.2, 'unemp': 5.1},
    # 2020 (COVID disruption — Jan/Feb combined released in March)
    '2020-01-17': {'month': 'Dec-19', 'ip_yoy': 6.9, 'retail_yoy': 8.0, 'fai_ytd_yoy': 5.4, 'unemp': 5.2},
    '2020-03-16': {'month': 'Jan-Feb', 'ip_yoy': -13.5, 'retail_yoy': -20.5, 'fai_ytd_yoy': -24.5, 'unemp': 6.2},
    '2020-04-17': {'month': 'Mar', 'ip_yoy': -1.1, 'retail_yoy': -15.8, 'fai_ytd_yoy': -16.1, 'unemp': 5.9},
    '2020-05-15': {'month': 'Apr', 'ip_yoy': 3.9, 'retail_yoy': -7.5, 'fai_ytd_yoy': -10.3, 'unemp': 6.0},
    '2020-06-15': {'month': 'May', 'ip_yoy': 4.4, 'retail_yoy': -2.8, 'fai_ytd_yoy': -6.3, 'unemp': 5.9},
    '2020-07-16': {'month': 'Jun', 'ip_yoy': 4.8, 'retail_yoy': -1.8, 'fai_ytd_yoy': -3.1, 'unemp': 5.7},
    '2020-08-14': {'month': 'Jul', 'ip_yoy': 4.8, 'retail_yoy': -1.1, 'fai_ytd_yoy': -1.6, 'unemp': 5.7},
    '2020-09-15': {'month': 'Aug', 'ip_yoy': 5.6, 'retail_yoy': 0.5, 'fai_ytd_yoy': -0.3, 'unemp': 5.6},
    '2020-10-19': {'month': 'Sep', 'ip_yoy': 6.9, 'retail_yoy': 3.3, 'fai_ytd_yoy': 0.8, 'unemp': 5.4},
    '2020-11-16': {'month': 'Oct', 'ip_yoy': 6.9, 'retail_yoy': 4.3, 'fai_ytd_yoy': 1.8, 'unemp': 5.3},
    '2020-12-15': {'month': 'Nov', 'ip_yoy': 7.0, 'retail_yoy': 5.0, 'fai_ytd_yoy': 2.6, 'unemp': 5.2},
    # 2021
    '2021-01-18': {'month': 'Dec-20', 'ip_yoy': 7.3, 'retail_yoy': 4.6, 'fai_ytd_yoy': 2.9, 'unemp': 5.2},
    '2021-03-15': {'month': 'Feb', 'ip_yoy': None, 'retail_yoy': None, 'fai_ytd_yoy': None, 'unemp': 5.5},
    '2021-04-16': {'month': 'Mar', 'ip_yoy': 14.1, 'retail_yoy': 34.2, 'fai_ytd_yoy': 25.6, 'unemp': 5.3},
    '2021-05-17': {'month': 'Apr', 'ip_yoy': 9.8, 'retail_yoy': 17.7, 'fai_ytd_yoy': 19.9, 'unemp': 5.1},
    '2021-06-16': {'month': 'May', 'ip_yoy': 8.8, 'retail_yoy': 12.4, 'fai_ytd_yoy': 15.4, 'unemp': 5.0},
    '2021-07-15': {'month': 'Jun', 'ip_yoy': 8.3, 'retail_yoy': 12.1, 'fai_ytd_yoy': 12.6, 'unemp': 5.0},
    '2021-08-16': {'month': 'Jul', 'ip_yoy': 6.4, 'retail_yoy': 8.5, 'fai_ytd_yoy': 10.3, 'unemp': 5.1},
    '2021-09-15': {'month': 'Aug', 'ip_yoy': 5.3, 'retail_yoy': 2.5, 'fai_ytd_yoy': 8.9, 'unemp': 5.1},
    '2021-10-18': {'month': 'Sep', 'ip_yoy': 3.1, 'retail_yoy': 4.4, 'fai_ytd_yoy': 7.3, 'unemp': 4.9},
    '2021-11-15': {'month': 'Oct', 'ip_yoy': 3.5, 'retail_yoy': 4.9, 'fai_ytd_yoy': 6.1, 'unemp': 4.9},
    '2021-12-15': {'month': 'Nov', 'ip_yoy': 3.8, 'retail_yoy': 3.9, 'fai_ytd_yoy': 5.2, 'unemp': 5.0},
    # 2022
    '2022-01-17': {'month': 'Dec-21', 'ip_yoy': 4.3, 'retail_yoy': 1.7, 'fai_ytd_yoy': 4.9, 'unemp': 5.1},
    '2022-03-15': {'month': 'Feb', 'ip_yoy': None, 'retail_yoy': None, 'fai_ytd_yoy': None, 'unemp': 5.5},
    '2022-04-18': {'month': 'Mar', 'ip_yoy': 5.0, 'retail_yoy': -3.5, 'fai_ytd_yoy': 9.3, 'unemp': 5.8},
    '2022-05-16': {'month': 'Apr', 'ip_yoy': -2.9, 'retail_yoy': -11.1, 'fai_ytd_yoy': 6.8, 'unemp': 6.1},
    '2022-06-15': {'month': 'May', 'ip_yoy': 0.7, 'retail_yoy': -6.7, 'fai_ytd_yoy': 6.2, 'unemp': 5.9},
    '2022-07-15': {'month': 'Jun', 'ip_yoy': 3.9, 'retail_yoy': 3.1, 'fai_ytd_yoy': 6.1, 'unemp': 5.5},
    '2022-08-15': {'month': 'Jul', 'ip_yoy': 3.8, 'retail_yoy': 2.7, 'fai_ytd_yoy': 5.7, 'unemp': 5.4},
    '2022-09-16': {'month': 'Aug', 'ip_yoy': 4.2, 'retail_yoy': 5.4, 'fai_ytd_yoy': 5.8, 'unemp': 5.3},
    '2022-10-24': {'month': 'Sep', 'ip_yoy': 6.3, 'retail_yoy': 2.5, 'fai_ytd_yoy': 5.9, 'unemp': 5.5},
    '2022-11-15': {'month': 'Oct', 'ip_yoy': 5.0, 'retail_yoy': -0.5, 'fai_ytd_yoy': 5.8, 'unemp': 5.5},
    '2022-12-15': {'month': 'Nov', 'ip_yoy': 2.2, 'retail_yoy': -5.9, 'fai_ytd_yoy': 5.3, 'unemp': 5.7},
    # 2023
    '2023-01-17': {'month': 'Dec-22', 'ip_yoy': 1.3, 'retail_yoy': -1.8, 'fai_ytd_yoy': 5.1, 'unemp': 5.5},
    '2023-03-15': {'month': 'Feb', 'ip_yoy': None, 'retail_yoy': None, 'fai_ytd_yoy': None, 'unemp': 5.6},
    '2023-04-18': {'month': 'Mar', 'ip_yoy': 3.9, 'retail_yoy': 10.6, 'fai_ytd_yoy': 5.1, 'unemp': 5.3},
    '2023-05-16': {'month': 'Apr', 'ip_yoy': 5.6, 'retail_yoy': 18.4, 'fai_ytd_yoy': 4.7, 'unemp': 5.2},
    '2023-06-15': {'month': 'May', 'ip_yoy': 3.5, 'retail_yoy': 12.7, 'fai_ytd_yoy': 4.0, 'unemp': 5.2},
    '2023-07-17': {'month': 'Jun', 'ip_yoy': 4.4, 'retail_yoy': 3.1, 'fai_ytd_yoy': 3.8, 'unemp': 5.2},
    '2023-08-15': {'month': 'Jul', 'ip_yoy': 3.7, 'retail_yoy': 2.5, 'fai_ytd_yoy': 3.4, 'unemp': 5.3},
    '2023-09-15': {'month': 'Aug', 'ip_yoy': 4.5, 'retail_yoy': 4.6, 'fai_ytd_yoy': 3.2, 'unemp': 5.2},
    '2023-10-18': {'month': 'Sep', 'ip_yoy': 4.5, 'retail_yoy': 5.5, 'fai_ytd_yoy': 3.1, 'unemp': 5.0},
    '2023-11-15': {'month': 'Oct', 'ip_yoy': 4.6, 'retail_yoy': 7.6, 'fai_ytd_yoy': 2.9, 'unemp': 5.0},
    '2023-12-15': {'month': 'Nov', 'ip_yoy': 6.6, 'retail_yoy': 10.1, 'fai_ytd_yoy': 2.9, 'unemp': 5.0},
    # 2024
    '2024-01-17': {'month': 'Dec-23', 'ip_yoy': 6.8, 'retail_yoy': 7.4, 'fai_ytd_yoy': 3.0, 'unemp': 5.1, 'consensus_ip': 6.6, 'consensus_retail': 8.0},
    '2024-03-18': {'month': 'Feb', 'ip_yoy': 7.0, 'retail_yoy': 5.5, 'fai_ytd_yoy': 4.2, 'unemp': 5.3, 'consensus_ip': 5.2, 'consensus_retail': 5.5},
    '2024-04-16': {'month': 'Mar', 'ip_yoy': 4.5, 'retail_yoy': 3.1, 'fai_ytd_yoy': 4.5, 'unemp': 5.2, 'consensus_ip': 6.0, 'consensus_retail': 4.8},
    '2024-05-17': {'month': 'Apr', 'ip_yoy': 6.7, 'retail_yoy': 2.3, 'fai_ytd_yoy': 4.2, 'unemp': 5.0, 'consensus_ip': 5.5, 'consensus_retail': 3.8},
    '2024-06-17': {'month': 'May', 'ip_yoy': 5.6, 'retail_yoy': 3.7, 'fai_ytd_yoy': 4.0, 'unemp': 5.0, 'consensus_ip': 6.0, 'consensus_retail': 4.5},
    '2024-07-15': {'month': 'Jun', 'ip_yoy': 5.3, 'retail_yoy': 2.0, 'fai_ytd_yoy': 3.9, 'unemp': 5.1, 'consensus_ip': 5.0, 'consensus_retail': 3.3},
    '2024-08-15': {'month': 'Jul', 'ip_yoy': 5.1, 'retail_yoy': 2.7, 'fai_ytd_yoy': 3.6, 'unemp': 5.2, 'consensus_ip': 5.3, 'consensus_retail': 2.6},
    '2024-09-16': {'month': 'Aug', 'ip_yoy': 4.5, 'retail_yoy': 3.7, 'fai_ytd_yoy': 3.4, 'unemp': 5.3, 'consensus_ip': 4.8, 'consensus_retail': 3.5},
    '2024-10-18': {'month': 'Sep', 'ip_yoy': 5.4, 'retail_yoy': 3.2, 'fai_ytd_yoy': 3.4, 'unemp': 5.1, 'consensus_ip': 4.6, 'consensus_retail': 2.5},
    '2024-11-15': {'month': 'Oct', 'ip_yoy': 5.3, 'retail_yoy': 4.8, 'fai_ytd_yoy': 3.4, 'unemp': 5.0, 'consensus_ip': 5.6, 'consensus_retail': 3.8},
    '2024-12-16': {'month': 'Nov', 'ip_yoy': 5.4, 'retail_yoy': 3.0, 'fai_ytd_yoy': 3.3, 'unemp': 5.0, 'consensus_ip': 5.3, 'consensus_retail': 4.6},
    # 2025
    '2025-01-17': {'month': 'Dec-24', 'ip_yoy': 6.2, 'retail_yoy': 3.7, 'fai_ytd_yoy': 3.2, 'unemp': 5.1, 'consensus_ip': 5.4, 'consensus_retail': 3.5},
    '2025-03-17': {'month': 'Feb', 'ip_yoy': 5.9, 'retail_yoy': 4.0, 'fai_ytd_yoy': 4.1, 'unemp': 5.4, 'consensus_ip': 5.3, 'consensus_retail': 3.8},
    '2025-04-16': {'month': 'Mar', 'ip_yoy': 7.7, 'retail_yoy': 5.9, 'fai_ytd_yoy': 4.3, 'unemp': 5.2, 'consensus_ip': 5.9, 'consensus_retail': 4.5},
    '2025-05-19': {'month': 'Apr', 'ip_yoy': 6.1, 'retail_yoy': 5.1, 'fai_ytd_yoy': 4.0, 'unemp': 5.1, 'consensus_ip': 5.5, 'consensus_retail': 5.0},
    '2025-06-16': {'month': 'May', 'ip_yoy': 5.8, 'retail_yoy': 4.5, 'fai_ytd_yoy': 3.7, 'unemp': 5.0, 'consensus_ip': 5.9, 'consensus_retail': 4.8},
    '2025-07-15': {'month': 'Jun', 'ip_yoy': 5.5, 'retail_yoy': 4.2, 'fai_ytd_yoy': 3.5, 'unemp': 5.1, 'consensus_ip': 5.6, 'consensus_retail': 4.0},
    '2025-08-15': {'month': 'Jul', 'ip_yoy': 5.0, 'retail_yoy': 3.5, 'fai_ytd_yoy': 3.2, 'unemp': 5.2, 'consensus_ip': 5.4, 'consensus_retail': 3.8},
    '2025-09-15': {'month': 'Aug', 'ip_yoy': 4.8, 'retail_yoy': 3.0, 'fai_ytd_yoy': 2.9, 'unemp': 5.3, 'consensus_ip': 5.2, 'consensus_retail': 3.5},
    '2025-10-16': {'month': 'Sep', 'ip_yoy': 5.2, 'retail_yoy': 3.8, 'fai_ytd_yoy': 2.8, 'unemp': 5.2, 'consensus_ip': 5.0, 'consensus_retail': 3.5},
    '2025-11-17': {'month': 'Oct', 'ip_yoy': 5.0, 'retail_yoy': 3.5, 'fai_ytd_yoy': 2.6, 'unemp': 5.1, 'consensus_ip': 5.3, 'consensus_retail': 4.0},
    '2025-12-15': {'month': 'Nov', 'ip_yoy': 5.3, 'retail_yoy': 4.0, 'fai_ytd_yoy': 2.4, 'unemp': 5.0, 'consensus_ip': 5.1, 'consensus_retail': 3.8},
    # 2026
    '2026-01-19': {'month': 'Dec-25', 'ip_yoy': 5.5, 'retail_yoy': 3.2, 'fai_ytd_yoy': 2.2, 'unemp': 5.1, 'consensus_ip': 5.3, 'consensus_retail': 3.5},
    '2026-03-16': {'month': 'Feb', 'ip_yoy': 5.9, 'retail_yoy': 5.5, 'fai_ytd_yoy': 3.8, 'unemp': 5.4, 'consensus_ip': 5.1, 'consensus_retail': 4.0},
    '2026-04-17': {'month': 'Mar', 'ip_yoy': 5.7, 'retail_yoy': 1.7, 'fai_ytd_yoy': 1.7, 'unemp': 5.4, 'consensus_ip': 5.5, 'consensus_retail': 3.5},
    '2026-05-18': {'month': 'Apr', 'ip_yoy': 4.1, 'retail_yoy': 0.2, 'fai_ytd_yoy': -1.6, 'unemp': 5.2, 'consensus_ip': 5.9, 'consensus_retail': 2.0},
}


# ═══════════════════════════════════════════════════════════════
# SESSION DEFINITIONS (UTC hours)
# ═══════════════════════════════════════════════════════════════

SESSIONS = [
    ('Pre-Asia',         20, 22),   # Post-NY Close / Globex
    ('Sydney Open',      22, 23),
    ('Tokyo Open',        0,  1),
    ('Asia Mid',          2,  4),   # Release is at 02:00 UTC
    ('Asia Afternoon',    4,  7),
    ('Tokyo Close',       7,  8),
    ('Pre-London',        8,  9),
    ('Frankfurt Open',    9, 10),
    ('London Open',      10, 11),
    ('London Morning',   11, 13),
    ('London Midday',    13, 14),
    ('NY Pre-Open',      14, 15),
    ('NY Open',          15, 16),
    ('London-NY Overlap', 16, 17),
    ('NY AM',            17, 19),
    ('NY Lunch',         19, 20),
    ('NY PM',            20, 22),
]


def fetch_eth_15m(binance_client, start_date, end_date):
    """Fetch ETH/USDT 15m OHLCV from Binance."""
    import ccxt
    ex = ccxt.binance({"enableRateLimit": True})

    start_ms = int(pd.Timestamp(start_date).timestamp() * 1000)
    end_ms = int(pd.Timestamp(end_date).timestamp() * 1000)

    all_rows = []
    current_ms = start_ms

    print(f"  Fetching ETH/USDT 15m from {start_date} to {end_date}...")
    while current_ms < end_ms:
        try:
            raw = ex.publicGetKlines({
                'symbol': 'ETHUSDT', 'interval': '15m',
                'startTime': current_ms, 'limit': 1000
            })
            if not raw:
                break
            all_rows.extend(raw)
            current_ms = int(raw[-1][0]) + 1
            if len(raw) < 1000:
                break
            time.sleep(0.1)
        except Exception as e:
            print(f"  ⚠️ Fetch error: {e}, retrying...")
            time.sleep(2)

    cols = ['Open time', 'Open', 'High', 'Low', 'Close', 'Volume',
            'Close time', 'Quote asset volume', 'Number of trades',
            'Taker buy base asset volume', 'Taker buy quote asset volume', 'Ignore']
    df = pd.DataFrame(all_rows, columns=cols)
    df['Open time'] = pd.to_datetime(df['Open time'].astype(int), unit='ms')
    for c in ['Open', 'High', 'Low', 'Close', 'Volume']:
        df[c] = pd.to_numeric(df[c])
    df = df.drop_duplicates(subset='Open time').sort_values('Open time').reset_index(drop=True)
    print(f"  ✅ Fetched {len(df)} bars ({df['Open time'].iloc[0]} → {df['Open time'].iloc[-1]})")
    return df


def get_session_return(df, release_dt, session_start_h, session_end_h):
    """Get return during a specific session window on release day.

    release_dt: datetime of release (UTC)
    session_start_h, session_end_h: UTC hours for session window
    """
    release_date = release_dt.date()

    # For sessions that span midnight (e.g. Pre-Asia 20-22 on release day)
    # The release is at 02:00 UTC, so we measure from release time forward
    if session_start_h >= 20:
        # Evening session on release day
        start = pd.Timestamp(f"{release_date} {session_start_h:02d}:00:00")
        end = pd.Timestamp(f"{release_date} {session_end_h:02d}:00:00")
    elif session_end_h <= 2:
        # Cross-midnight
        start = pd.Timestamp(f"{release_date} {session_start_h:02d}:00:00")
        end = pd.Timestamp(f"{release_date}") + timedelta(days=1, hours=session_end_h)
    else:
        start = pd.Timestamp(f"{release_date} {session_start_h:02d}:00:00")
        end = pd.Timestamp(f"{release_date} {session_end_h:02d}:00:00")

    # Release is at 02:00 UTC — only measure from release time forward
    release_ts = pd.Timestamp(release_dt)
    if start < release_ts:
        start = release_ts

    mask = (df['Open time'] >= start) & (df['Open time'] < end)
    session_df = df[mask]

    if len(session_df) < 1:
        return None, None, None

    open_price = float(session_df['Open'].iloc[0])
    close_price = float(session_df['Close'].iloc[-1])
    high = float(session_df['High'].max())
    low = float(session_df['Low'].min())

    ret = (close_price - open_price) / open_price * 100
    return ret, open_price, close_price


def get_24h_return(df, release_dt):
    """Get 24h aggregate return from release time."""
    start = pd.Timestamp(release_dt)
    end = start + timedelta(hours=24)

    mask = (df['Open time'] >= start) & (df['Open time'] < end)
    window = df[mask]

    if len(window) < 1:
        return None

    open_price = float(window['Open'].iloc[0])
    close_price = float(window['Close'].iloc[-1])
    return (close_price - open_price) / open_price * 100


def classify_signal(data):
    """Classify release as BEAT/MISS/INLINE based on consensus."""
    ip = data.get('ip_yoy')
    retail = data.get('retail_yoy')
    ip_c = data.get('consensus_ip')
    retail_c = data.get('consensus_retail')

    surprises = []
    if ip is not None and ip_c is not None:
        surprises.append(ip - ip_c)
    if retail is not None and retail_c is not None:
        surprises.append(retail - retail_c)

    if not surprises:
        return 'NO_CONSENSUS'

    avg_surprise = np.mean(surprises)
    if avg_surprise >= 0.5:
        return 'STRONG_BEAT'
    elif avg_surprise >= 0.15:
        return 'BEAT'
    elif avg_surprise <= -0.5:
        return 'BIG_MISS'
    elif avg_surprise <= -0.15:
        return 'MISS'
    return 'INLINE'


def classify_composite(data):
    """Classify composite signal using IP + Retail + FAI."""
    ip = data.get('ip_yoy')
    retail = data.get('retail_yoy')
    fai = data.get('fai_ytd_yoy')
    ip_c = data.get('consensus_ip')
    retail_c = data.get('consensus_retail')

    signals = []
    if ip is not None and ip_c is not None:
        s = ip - ip_c
        if s >= 1.0: signals.append(('IP', 'STRONG_BEAT', s))
        elif s >= 0.3: signals.append(('IP', 'BEAT', s))
        elif s <= -1.0: signals.append(('IP', 'BIG_MISS', s))
        elif s <= -0.3: signals.append(('IP', 'MISS', s))
        else: signals.append(('IP', 'INLINE', s))

    if retail is not None and retail_c is not None:
        s = retail - retail_c
        if s >= 1.0: signals.append(('Retail', 'STRONG_BEAT', s))
        elif s >= 0.3: signals.append(('Retail', 'BEAT', s))
        elif s <= -1.0: signals.append(('Retail', 'BIG_MISS', s))
        elif s <= -0.3: signals.append(('Retail', 'MISS', s))
        else: signals.append(('Retail', 'INLINE', s))

    if fai is not None:
        if fai < 0: signals.append(('FAI', 'CONTRACTION', fai))
        elif fai < 2.0: signals.append(('FAI', 'WEAK', fai))
        else: signals.append(('FAI', 'OK', fai))

    if not signals:
        return 'NO_DATA', 0

    score_map = {'STRONG_BEAT': 2, 'BEAT': 1, 'INLINE': 0, 'OK': 0,
                 'WEAK': -0.5, 'MISS': -1, 'BIG_MISS': -2, 'CONTRACTION': -1.5}
    total = sum(score_map.get(s[1], 0) for s in signals)
    avg = total / len(signals)

    if avg >= 1.0: return 'STRONG_BEAT', avg
    elif avg >= 0.3: return 'BEAT', avg
    elif avg <= -1.0: return 'BIG_MISS', avg
    elif avg <= -0.3: return 'MISS', avg
    return 'INLINE', avg


def classify_fai(data):
    """FAI-specific classification."""
    fai = data.get('fai_ytd_yoy')
    if fai is None: return 'NO_DATA'
    if fai < -5: return 'COLLAPSE'
    elif fai < 0: return 'CONTRACTION'
    elif fai < 2: return 'WEAK'
    elif fai < 5: return 'MODERATE'
    return 'STRONG'


def classify_house(data):
    """House price classification."""
    hp = data.get('house_price_yoy')
    if hp is None: return 'NO_DATA'
    if hp <= -5: return 'CRISIS'
    elif hp <= -3: return 'DECLINING'
    elif hp <= -1: return 'MILD_DECLINE'
    elif hp <= 0: return 'STABLE'
    return 'RISING'


def get_wyckoff_phase(price_at_release, df_1d, release_dt):
    """Approximate Wyckoff phase from daily data."""
    # Simple: check 20d and 50d price position
    mask_20d = df_1d['Open time'] < pd.Timestamp(release_dt)
    recent = df_1d[mask_20d].tail(20)
    if len(recent) < 10:
        return 'UNKNOWN'

    high_20 = float(recent['High'].max())
    low_20 = float(recent['Low'].min())
    range_pct = (high_20 - low_20) / low_20 * 100
    position = (price_at_release - low_20) / (high_20 - low_20) if high_20 != low_20 else 0.5

    # Check 50d trend
    recent_50 = df_1d[mask_20d].tail(50)
    if len(recent_50) >= 20:
        ma20 = float(recent_50['Close'].tail(20).mean())
        ma50 = float(recent_50['Close'].mean())
        trend_up = ma20 > ma50
    else:
        trend_up = position > 0.5

    if range_pct < 8:
        return 'RANGE'  # tight range
    elif position < 0.25 and not trend_up:
        return 'ACCUMULATION'
    elif position > 0.75 and trend_up:
        return 'DISTRIBUTION'
    elif trend_up:
        return 'MARKUP'
    else:
        return 'MARKDOWN'


def get_vol_regime(price_at_release, df_15m, release_dt):
    """Approximate vol regime from 15m data."""
    mask = df_15m['Open time'] < pd.Timestamp(release_dt)
    recent = df_15m[mask].tail(192)  # 48h of 15m bars
    if len(recent) < 48:
        return 'UNKNOWN'

    # ATR-based
    highs = recent['High'].values.astype(float)
    lows = recent['Low'].values.astype(float)
    closes = recent['Close'].values.astype(float)

    tr = np.maximum(highs[1:] - lows[1:],
                    np.maximum(np.abs(highs[1:] - closes[:-1]),
                               np.abs(lows[1:] - closes[:-1])))
    atr = np.mean(tr[-14:])
    atr_pct = atr / price_at_release * 100

    # Range
    range_48h = (max(highs[-48:]) - min(lows[-48:])) / price_at_release * 100

    if atr_pct > 1.5 or range_48h > 6:
        return 'CRISIS'
    elif atr_pct > 0.8 or range_48h > 3.5:
        return 'TRENDING'
    elif atr_pct < 0.3 or range_48h < 1.5:
        return 'COMPRESSING'
    return 'NEUTRAL'


def main():
    print("=" * 70)
    print("  CHINA NBS ACTIVITY DATA — FULL BACKTEST (2018-2026)")
    print("=" * 70)

    # Step 1: Fetch ETH 15m data
    cache_path = os.path.join(os.path.dirname(os.path.dirname(__file__)),
                              'data', 'eth_15m_merged.csv')

    if os.path.exists(cache_path):
        print(f"\n  📄 Loading cached CSV ({cache_path})...")
        df = pd.read_csv(cache_path)
        df['Open time'] = pd.to_datetime(df['Open time'])
        for c in ['Open', 'High', 'Low', 'Close', 'Volume']:
            df[c] = pd.to_numeric(df[c])
        print(f"  ✅ Loaded {len(df)} bars")
    else:
        print(f"\n  📥 No CSV found, fetching from Binance (2018-01-01 → today)...")
        df = fetch_eth_15m(None, '2018-01-01', datetime.now().strftime('%Y-%m-%d'))
        df.to_csv(cache_path, index=False)
        print(f"  💾 Saved to {cache_path}")

    # Create daily resample for Wyckoff
    df_1d = df.set_index('Open time').resample('1D').agg({
        'Open': 'first', 'High': 'max', 'Low': 'min', 'Close': 'last', 'Volume': 'sum'
    }).dropna().reset_index()

    # Step 2: Process each release
    print(f"\n  Processing {len(RELEASES)} releases...")
    results = []

    for date_str, data in sorted(RELEASES.items()):
        # Skip combined Jan+Feb releases with no data
        if data.get('ip_yoy') is None and data.get('retail_yoy') is None:
            continue

        release_dt = datetime.strptime(date_str, '%Y-%m-%d').replace(hour=2, minute=0)

        # Find price at release
        mask = df['Open time'] >= pd.Timestamp(release_dt)
        if not mask.any():
            continue
        release_bar = df[mask].iloc[0]
        price = float(release_bar['Close'])

        # 24h return
        ret_24h = get_24h_return(df, release_dt)
        if ret_24h is None:
            continue

        # Session returns
        session_rets = {}
        for sname, sh, eh in SESSIONS:
            ret, _, _ = get_session_return(df, release_dt, sh, eh)
            if ret is not None:
                session_rets[sname] = ret

        # Classifications
        signal = classify_signal(data)
        composite, comp_score = classify_composite(data)
        fai_class = classify_fai(data)
        house_class = classify_house(data)
        wyckoff = get_wyckoff_phase(price, df_1d, release_dt)
        vol = get_vol_regime(price, df, release_dt)

        results.append({
            'date': date_str,
            'month': data.get('month'),
            'price': price,
            'ret_24h': ret_24h,
            'signal': signal,
            'composite': composite,
            'comp_score': comp_score,
            'fai_class': fai_class,
            'house_class': house_class,
            'wyckoff': wyckoff,
            'vol': vol,
            'ip': data.get('ip_yoy'),
            'retail': data.get('retail_yoy'),
            'fai': data.get('fai_ytd_yoy'),
            'house': data.get('house_price_yoy'),
            'unemp': data.get('unemp'),
            'ip_c': data.get('consensus_ip'),
            'retail_c': data.get('consensus_retail'),
            'sessions': session_rets,
        })

    rdf = pd.DataFrame(results)
    print(f"  ✅ Processed {len(rdf)} releases with data")

    # ═══════════════════════════════════════════════════════════════
    # PROMPT A: FULL BACKTEST
    # ═══════════════════════════════════════════════════════════════
    print("\n" + "=" * 70)
    print("  PROMPT A: FULL BACKTEST RESULTS")
    print("=" * 70)

    # 24h aggregate stats
    print(f"\n  ── 24h AGGREGATE ──")
    print(f"  N={len(rdf)}  Mean={rdf['ret_24h'].mean():+.3f}%  "
          f"Median={rdf['ret_24h'].median():+.3f}%  "
          f"Std={rdf['ret_24h'].std():.3f}%")
    win_rate = (rdf['ret_24h'] > 0).mean()
    print(f"  Win rate: {win_rate*100:.1f}%")

    # t-test vs 0
    t_stat, p_val = stats.ttest_1samp(rdf['ret_24h'].dropna(), 0)
    sig = "✅ SIGNIFICANT" if p_val < 0.05 else "❌ NOT significant"
    print(f"  t-test vs 0: t={t_stat:.3f}  p={p_val:.4f}  {sig}")

    # Cross-tab: composite signal × 24h return
    print(f"\n  ── COMPOSITE SIGNAL → 24h RETURN ──")
    print(f"  {'Signal':<16} {'N':>4} {'Avg%':>8} {'Win%':>6} {'Med%':>8}")
    print(f"  {'─'*48}")
    for sig_name in ['STRONG_BEAT', 'BEAT', 'INLINE', 'MISS', 'BIG_MISS']:
        sub = rdf[rdf['composite'] == sig_name]
        if len(sub) >= 1:
            avg = sub['ret_24h'].mean()
            med = sub['ret_24h'].median()
            wr = (sub['ret_24h'] > 0).mean() * 100
            icon = '🟢' if avg > 0.5 else '🔴' if avg < -0.5 else '⚪'
            print(f"  {icon} {sig_name:<14} {len(sub):>4} {avg:>+7.2f}% {wr:>5.1f}% {med:>+7.2f}%")

    # FAI classification
    print(f"\n  ── FAI CLASSIFICATION → 24h RETURN ──")
    print(f"  {'FAI Status':<16} {'N':>4} {'Avg%':>8} {'Win%':>6}")
    print(f"  {'─'*40}")
    for fai_name in ['COLLAPSE', 'CONTRACTION', 'WEAK', 'MODERATE', 'STRONG']:
        sub = rdf[rdf['fai_class'] == fai_name]
        if len(sub) >= 1:
            avg = sub['ret_24h'].mean()
            wr = (sub['ret_24h'] > 0).mean() * 100
            print(f"  {fai_name:<16} {len(sub):>4} {avg:>+7.2f}% {wr:>5.1f}%")

    # House price classification
    print(f"\n  ── HOUSE PRICE CLASSIFICATION → 24h RETURN ──")
    print(f"  {'House Status':<16} {'N':>4} {'Avg%':>8} {'Win%':>6}")
    print(f"  {'─'*40}")
    for hp_name in ['CRISIS', 'DECLINING', 'MILD_DECLINE', 'STABLE', 'RISING']:
        sub = rdf[rdf['house_class'] == hp_name]
        if len(sub) >= 1:
            avg = sub['ret_24h'].mean()
            wr = (sub['ret_24h'] > 0).mean() * 100
            print(f"  {hp_name:<16} {len(sub):>4} {avg:>+7.2f}% {wr:>5.1f}%")

    # Wyckoff × Vol × Signal cross-tab
    print(f"\n  ── WYCKOFF × VOL REGIME × SIGNAL → 24h RETURN ──")
    print(f"  {'Wyckoff':<16} {'Vol':<12} {'Signal':<14} {'N':>4} {'Avg%':>8} {'Win%':>6}")
    print(f"  {'─'*66}")
    for wy in ['ACCUMULATION', 'MARKUP', 'DISTRIBUTION', 'MARKDOWN', 'RANGE']:
        for v in ['COMPRESSING', 'NEUTRAL', 'TRENDING', 'CRISIS']:
            for s in ['BIG_MISS', 'MISS', 'INLINE', 'BEAT', 'STRONG_BEAT']:
                sub = rdf[(rdf['wyckoff'] == wy) & (rdf['vol'] == v) & (rdf['composite'] == s)]
                if len(sub) >= 2:
                    avg = sub['ret_24h'].mean()
                    wr = (sub['ret_24h'] > 0).mean() * 100
                    print(f"  {wy:<16} {v:<12} {s:<14} {len(sub):>4} {avg:>+7.2f}% {wr:>5.1f}%")

    # Edge combos (n≥3, |avg|≥0.5%)
    print(f"\n  ── EDGE COMBOS (n≥3, |avg|≥0.5%) ──")
    print(f"  {'Wyckoff':<16} {'Vol':<12} {'Signal':<14} {'N':>4} {'Avg%':>8} {'Win%':>6} {'Sig?':>6}")
    print(f"  {'─'*70}")
    edges = []
    for wy in rdf['wyckoff'].unique():
        for v in rdf['vol'].unique():
            for s in rdf['composite'].unique():
                sub = rdf[(rdf['wyckoff'] == wy) & (rdf['vol'] == v) & (rdf['composite'] == s)]
                if len(sub) >= 3:
                    avg = sub['ret_24h'].mean()
                    if abs(avg) >= 0.5:
                        wr = (sub['ret_24h'] > 0).mean() * 100
                        # Two-sample t-test: this combo vs rest
                        rest = rdf[~rdf.index.isin(sub.index)]['ret_24h'].dropna()
                        if len(rest) >= 5:
                            _, p = stats.ttest_ind(sub['ret_24h'].dropna(), rest)
                            sig_mark = '✅' if p < 0.05 else '⚠️' if p < 0.10 else '❌'
                        else:
                            sig_mark = '?'
                        print(f"  {wy:<16} {v:<12} {s:<14} {len(sub):>4} {avg:>+7.2f}% {wr:>5.1f}% {sig_mark:>6}")
                        edges.append((wy, v, s, len(sub), avg, wr))

    # ═══════════════════════════════════════════════════════════════
    # PROMPT B: SESSION TRANSMISSION CHAIN
    # ═══════════════════════════════════════════════════════════════
    print("\n" + "=" * 70)
    print("  PROMPT B: SESSION TRANSMISSION CHAIN VALIDATION")
    print("=" * 70)

    # Build session return matrix
    session_names = [s[0] for s in SESSIONS]
    session_matrix = []
    for _, row in rdf.iterrows():
        srets = row['sessions']
        session_matrix.append([srets.get(s, np.nan) for s in session_names])

    smat = pd.DataFrame(session_matrix, columns=session_names)

    # Session-by-session stats
    print(f"\n  ── SESSION RETURN STATS ──")
    print(f"  {'Session':<22} {'N':>4} {'Avg%':>8} {'Win%':>6} {'Std%':>8}")
    print(f"  {'─'*54}")
    for s in session_names:
        col = smat[s].dropna()
        if len(col) >= 3:
            avg = col.mean()
            wr = (col > 0).mean() * 100
            std = col.std()
            icon = '🟢' if avg > 0.1 else '🔴' if avg < -0.1 else '⚪'
            print(f"  {icon} {s:<20} {len(col):>4} {avg:>+7.3f}% {wr:>5.1f}% {std:>7.3f}%")

    # Direction persistence chain
    print(f"\n  ── DIRECTION PERSISTENCE CHAIN ──")
    print(f"  (Does the initial direction carry through sessions?)")

    # For each consecutive session pair, measure % same direction
    print(f"\n  {'Transition':<36} {'N':>4} {'Same%':>6} {'Edge?':>8}")
    print(f"  {'─'*58}")

    # Find the first session with data (usually Asia Mid = release window)
    first_session_idx = None
    for i, s in enumerate(session_names):
        if smat[s].notna().sum() >= 5:
            first_session_idx = i
            break

    if first_session_idx is not None:
        for i in range(first_session_idx, len(session_names) - 1):
            s1 = session_names[i]
            s2 = session_names[i + 1]
            valid = smat[[s1, s2]].dropna()
            if len(valid) >= 3:
                same_dir = ((valid[s1] > 0) & (valid[s2] > 0)) | \
                           ((valid[s1] < 0) & (valid[s2] < 0))
                same_pct = same_dir.mean() * 100
                edge = '✅ EDGE' if same_pct >= 65 else '⚠️ MARGINAL' if same_pct >= 55 else '❌ BROKEN'
                print(f"  {s1:<18} → {s2:<14} {len(valid):>4} {same_pct:>5.1f}% {edge}")

    # Full chain persistence (from release to end of day)
    print(f"\n  ── FULL CHAIN: Release → each subsequent session ──")
    if first_session_idx is not None:
        base_session = session_names[first_session_idx]
        print(f"  Base: {base_session}")
        print(f"  {'Target Session':<22} {'N':>4} {'Same%':>6} {'Edge?':>8} {'Avg Ret%':>10}")
        print(f"  {'─'*56}")

        for i in range(first_session_idx + 1, len(session_names)):
            target = session_names[i]
            valid = smat[[base_session, target]].dropna()
            if len(valid) >= 3:
                same_dir = ((valid[base_session] > 0) & (valid[target] > 0)) | \
                           ((valid[base_session] < 0) & (valid[target] < 0))
                same_pct = same_dir.mean() * 100
                avg_ret = valid[target].mean()
                edge = '✅' if same_pct >= 65 else '⚠️' if same_pct >= 55 else '❌'
                print(f"  {target:<22} {len(valid):>4} {same_pct:>5.1f}% {edge:>8} {avg_ret:>+9.3f}%")

    # Statistical tests
    print(f"\n  ── STATISTICAL TESTS ──")

    # Miss vs Beat t-test
    miss_rets = rdf[rdf['composite'].isin(['MISS', 'BIG_MISS'])]['ret_24h'].dropna()
    beat_rets = rdf[rdf['composite'].isin(['BEAT', 'STRONG_BEAT'])]['ret_24h'].dropna()

    if len(miss_rets) >= 3 and len(beat_rets) >= 3:
        t_stat, p_val = stats.ttest_ind(miss_rets, beat_rets)
        sig = "✅ SIGNIFICANT" if p_val < 0.05 else "⚠️ MARGINAL" if p_val < 0.10 else "❌ NOT significant"
        print(f"  Miss vs Beat: miss_avg={miss_rets.mean():+.2f}% (n={len(miss_rets)})  "
              f"beat_avg={beat_rets.mean():+.2f}% (n={len(beat_rets)})")
        print(f"  t={t_stat:.3f}  p={p_val:.4f}  {sig}")

    # FAI contraction vs expansion
    fai_neg = rdf[rdf['fai_class'].isin(['COLLAPSE', 'CONTRACTION'])]['ret_24h'].dropna()
    fai_pos = rdf[rdf['fai_class'].isin(['MODERATE', 'STRONG'])]['ret_24h'].dropna()
    if len(fai_neg) >= 3 and len(fai_pos) >= 3:
        t_stat, p_val = stats.ttest_ind(fai_neg, fai_pos)
        sig = "✅" if p_val < 0.05 else "⚠️" if p_val < 0.10 else "❌"
        print(f"\n  FAI Contraction vs Expansion:")
        print(f"  Contraction avg={fai_neg.mean():+.2f}% (n={len(fai_neg)})  "
              f"Expansion avg={fai_pos.mean():+.2f}% (n={len(fai_pos)})")
        print(f"  t={t_stat:.3f}  p={p_val:.4f}  {sig}")

    # House crisis vs non-crisis
    hp_crisis = rdf[rdf['house_class'] == 'CRISIS']['ret_24h'].dropna()
    hp_ok = rdf[rdf['house_class'].isin(['STABLE', 'RISING', 'MILD_DECLINE'])]['ret_24h'].dropna()
    if len(hp_crisis) >= 3 and len(hp_ok) >= 3:
        t_stat, p_val = stats.ttest_ind(hp_crisis, hp_ok)
        sig = "✅" if p_val < 0.05 else "⚠️" if p_val < 0.10 else "❌"
        print(f"\n  House Crisis vs Non-Crisis:")
        print(f"  Crisis avg={hp_crisis.mean():+.2f}% (n={len(hp_crisis)})  "
              f"Non-crisis avg={hp_ok.mean():+.2f}% (n={len(hp_ok)})")
        print(f"  t={t_stat:.3f}  p={p_val:.4f}  {sig}")

    # ═══════════════════════════════════════════════════════════════
    # PROMPT C: INTEGRATION RECOMMENDATIONS
    # ═══════════════════════════════════════════════════════════════
    print("\n" + "=" * 70)
    print("  PROMPT C: INTEGRATION RECOMMENDATIONS")
    print("=" * 70)

    print(f"""
  Existing module: src/modules/m_china_activity.py
  Current state: Used ONLY in cascade_china.py (as CHINA_ACTIVITY step)
  Missing: Standalone scoring module (like M24, M25, M26 pattern)

  Module number: M65 (next available — M62 is last US module)

  Recommended integration:
  1. Create src/modules/m65_china_activity.py (extract from m_china_activity.py)
  2. Wire into scanner.py: score after M24 (NBS PMI), before M22 aggregator
  3. ICS adjustment: ±0.03-0.05 (moderate — China activity is monthly, not high-freq)
  4. Size multiplier: 0.80 on BIG_MISS, 0.90 on MISS, 1.0 otherwise
  5. Config: M65_ENABLED=true, M65_WINDOW_DAYS=1

  Signal summary from backtest:""")

    # Summarize best edges
    if edges:
        edges.sort(key=lambda x: abs(x[4]), reverse=True)
        for wy, v, s, n, avg, wr in edges[:5]:
            direction = 'LONG' if avg > 0 else 'SHORT'
            print(f"    {wy} + {v} + {s}: {avg:+.2f}% avg, {wr:.0f}% win, n={n} → {direction}")

    # Today's signal
    today = rdf[rdf['date'] == '2026-05-18']
    if len(today) > 0:
        t = today.iloc[0]
        print(f"\n  TODAY (2026-05-18):")
        print(f"    Composite: {t['composite']} (score={t['comp_score']:+.2f})")
        print(f"    IP: {t['ip']}% (cons {t['ip_c']}%) → surprise {t['ip']-t['ip_c']:+.1f}%")
        print(f"    Retail: {t['retail']}% (cons {t['retail_c']}%) → surprise {t['retail']-t['retail_c']:+.1f}%")
        print(f"    FAI: {t['fai']}% → {t['fai_class']}")
        print(f"    House: {t['house']}% → {t['house_class']}")
        print(f"    Wyckoff: {t['wyckoff']}  Vol: {t['vol']}")
        print(f"    24h return: {t['ret_24h']:+.2f}%")

    # Save results
    out_path = os.path.join(os.path.dirname(os.path.dirname(__file__)),
                            'data', 'scans', 'china_activity_backtest.json')
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\n  💾 Saved: {out_path}")


if __name__ == '__main__':
    main()

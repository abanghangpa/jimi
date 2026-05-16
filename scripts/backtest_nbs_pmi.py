#!/usr/bin/env python3
"""
Backtest: NBS Manufacturing + Services PMI Session Itinerary
Period: 2018-01-01 → 2026-05-16

Tests the claims:
1. Asia Open (09:30 MYT / 01:30 UTC): NBS PMI miss → V-bottom (swift dip then recovery)
2. Australia Mid-day: AUD proxy → correlated ETH buy orders
3. Europe Morning: Match against Eurozone PMI → trend confirmation/reversal
4. Asia Re-open: Caixin Services PMI → locks in weekly direction

NBS PMI release schedule:
- Manufacturing PMI: last day of each month, 09:00 CST (01:00 UTC)
- Services PMI: ~3rd business day of next month, 09:00 CST (01:00 UTC)

We use known NBS PMI release dates 2018-2026 and measure ETH 15m returns
across session windows.
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import sys
import os
import json

# ═══════════════════════════════════════════════════════════════
# NBS PMI RELEASE DATES (Manufacturing PMI)
# Released ~09:00 CST = 01:00 UTC on last business day of month
# Source: Trading Economics / NBS official calendar
# ═══════════════════════════════════════════════════════════════

# Manufacturing PMI release dates (YYYY-MM-DD) — actual calendar
# These are the dates the NBS published the previous month's PMI
# Format: (release_date, previous_month, mfg_pmi, services_pmi if available)
# We'll fetch from web or use known dates

# Known NBS Manufacturing PMI release dates 2018-2026
# Released on last day of reference month or first business day after
NBS_MFG_RELEASES = {
    # 2018
    "2018-01-31": {"ref_month": "Jan 2018", "mfg": 51.3, "services": 55.3},
    "2018-02-28": {"ref_month": "Feb 2018", "mfg": 50.3, "services": 54.4},
    "2018-03-31": {"ref_month": "Mar 2018", "mfg": 51.5, "services": 54.6},
    "2018-04-30": {"ref_month": "Apr 2018", "mfg": 51.4, "services": 54.8},
    "2018-05-31": {"ref_month": "May 2018", "mfg": 51.9, "services": 54.9},
    "2018-06-30": {"ref_month": "Jun 2018", "mfg": 51.5, "services": 55.0},
    "2018-07-31": {"ref_month": "Jul 2018", "mfg": 51.2, "services": 54.0},
    "2018-08-31": {"ref_month": "Aug 2018", "mfg": 51.3, "services": 54.2},
    "2018-09-30": {"ref_month": "Sep 2018", "mfg": 50.8, "services": 54.9},
    "2018-10-31": {"ref_month": "Oct 2018", "mfg": 50.2, "services": 53.9},
    "2018-11-30": {"ref_month": "Nov 2018", "mfg": 50.0, "services": 53.4},
    "2018-12-31": {"ref_month": "Dec 2018", "mfg": 49.4, "services": 53.8},
    # 2019
    "2019-01-31": {"ref_month": "Jan 2019", "mfg": 49.5, "services": 54.4},
    "2019-02-28": {"ref_month": "Feb 2019", "mfg": 49.2, "services": 54.3},
    "2019-03-31": {"ref_month": "Mar 2019", "mfg": 50.5, "services": 54.0},
    "2019-04-30": {"ref_month": "Apr 2019", "mfg": 50.1, "services": 53.3},
    "2019-05-31": {"ref_month": "May 2019", "mfg": 49.4, "services": 54.3},
    "2019-06-30": {"ref_month": "Jun 2019", "mfg": 49.4, "services": 54.2},
    "2019-07-31": {"ref_month": "Jul 2019", "mfg": 49.7, "services": 53.7},
    "2019-08-31": {"ref_month": "Aug 2019", "mfg": 49.5, "services": 53.8},
    "2019-09-30": {"ref_month": "Sep 2019", "mfg": 49.8, "services": 53.7},
    "2019-10-31": {"ref_month": "Oct 2019", "mfg": 49.3, "services": 52.8},
    "2019-11-30": {"ref_month": "Nov 2019", "mfg": 50.2, "services": 53.5},
    "2019-12-31": {"ref_month": "Dec 2019", "mfg": 50.2, "services": 53.5},
    # 2020
    "2020-01-31": {"ref_month": "Jan 2020", "mfg": 50.0, "services": 54.1},
    "2020-02-29": {"ref_month": "Feb 2020", "mfg": 35.7, "services": 29.6},
    "2020-03-31": {"ref_month": "Mar 2020", "mfg": 52.0, "services": 52.3},
    "2020-04-30": {"ref_month": "Apr 2020", "mfg": 50.8, "services": 53.2},
    "2020-05-31": {"ref_month": "May 2020", "mfg": 50.6, "services": 53.6},
    "2020-06-30": {"ref_month": "Jun 2020", "mfg": 50.9, "services": 54.4},
    "2020-07-31": {"ref_month": "Jul 2020", "mfg": 51.1, "services": 54.2},
    "2020-08-31": {"ref_month": "Aug 2020", "mfg": 51.0, "services": 55.2},
    "2020-09-30": {"ref_month": "Sep 2020", "mfg": 51.5, "services": 55.9},
    "2020-10-31": {"ref_month": "Oct 2020", "mfg": 51.4, "services": 56.2},
    "2020-11-30": {"ref_month": "Nov 2020", "mfg": 52.1, "services": 56.4},
    "2020-12-31": {"ref_month": "Dec 2020", "mfg": 51.9, "services": 55.7},
    # 2021
    "2021-01-31": {"ref_month": "Jan 2021", "mfg": 51.3, "services": 52.4},
    "2021-02-28": {"ref_month": "Feb 2021", "mfg": 50.6, "services": 51.4},
    "2021-03-31": {"ref_month": "Mar 2021", "mfg": 51.9, "services": 56.3},
    "2021-04-30": {"ref_month": "Apr 2021", "mfg": 51.1, "services": 54.9},
    "2021-05-31": {"ref_month": "May 2021", "mfg": 51.0, "services": 55.2},
    "2021-06-30": {"ref_month": "Jun 2021", "mfg": 50.9, "services": 53.5},
    "2021-07-31": {"ref_month": "Jul 2021", "mfg": 50.4, "services": 53.3},
    "2021-08-31": {"ref_month": "Aug 2021", "mfg": 50.1, "services": 47.5},
    "2021-09-30": {"ref_month": "Sep 2021", "mfg": 49.6, "services": 53.2},
    "2021-10-31": {"ref_month": "Oct 2021", "mfg": 49.2, "services": 51.6},
    "2021-11-30": {"ref_month": "Nov 2021", "mfg": 50.1, "services": 52.3},
    "2021-12-31": {"ref_month": "Dec 2021", "mfg": 50.3, "services": 52.7},
    # 2022
    "2022-01-30": {"ref_month": "Jan 2022", "mfg": 50.1, "services": 51.1},
    "2022-02-28": {"ref_month": "Feb 2022", "mfg": 50.2, "services": 51.6},
    "2022-03-31": {"ref_month": "Mar 2022", "mfg": 49.5, "services": 48.4},
    "2022-04-30": {"ref_month": "Apr 2022", "mfg": 47.4, "services": 41.9},
    "2022-05-31": {"ref_month": "May 2022", "mfg": 49.6, "services": 47.8},
    "2022-06-30": {"ref_month": "Jun 2022", "mfg": 50.2, "services": 54.7},
    "2022-07-31": {"ref_month": "Jul 2022", "mfg": 49.0, "services": 53.8},
    "2022-08-31": {"ref_month": "Aug 2022", "mfg": 49.4, "services": 52.6},
    "2022-09-30": {"ref_month": "Sep 2022", "mfg": 50.1, "services": 50.6},
    "2022-10-31": {"ref_month": "Oct 2022", "mfg": 49.2, "services": 48.7},
    "2022-11-30": {"ref_month": "Nov 2022", "mfg": 48.0, "services": 46.7},
    "2022-12-31": {"ref_month": "Dec 2022", "mfg": 47.0, "services": 41.6},
    # 2023
    "2023-01-31": {"ref_month": "Jan 2023", "mfg": 50.1, "services": 54.4},
    "2023-02-28": {"ref_month": "Feb 2023", "mfg": 52.6, "services": 56.3},
    "2023-03-31": {"ref_month": "Mar 2023", "mfg": 51.9, "services": 58.2},
    "2023-04-30": {"ref_month": "Apr 2023", "mfg": 49.2, "services": 56.4},
    "2023-05-31": {"ref_month": "May 2023", "mfg": 48.8, "services": 54.5},
    "2023-06-30": {"ref_month": "Jun 2023", "mfg": 49.0, "services": 53.2},
    "2023-07-31": {"ref_month": "Jul 2023", "mfg": 49.3, "services": 51.5},
    "2023-08-31": {"ref_month": "Aug 2023", "mfg": 49.7, "services": 51.0},
    "2023-09-30": {"ref_month": "Sep 2023", "mfg": 50.2, "services": 51.7},
    "2023-10-31": {"ref_month": "Oct 2023", "mfg": 49.5, "services": 50.6},
    "2023-11-30": {"ref_month": "Nov 2023", "mfg": 49.4, "services": 50.2},
    "2023-12-31": {"ref_month": "Dec 2023", "mfg": 49.0, "services": 49.3},
    # 2024
    "2024-01-31": {"ref_month": "Jan 2024", "mfg": 49.1, "services": 50.7},
    "2024-02-29": {"ref_month": "Feb 2024", "mfg": 49.1, "services": 51.4},
    "2024-03-31": {"ref_month": "Mar 2024", "mfg": 50.8, "services": 53.0},
    "2024-04-30": {"ref_month": "Apr 2024", "mfg": 50.4, "services": 51.2},
    "2024-05-31": {"ref_month": "May 2024", "mfg": 49.5, "services": 50.5},
    "2024-06-30": {"ref_month": "Jun 2024", "mfg": 49.5, "services": 50.2},
    "2024-07-31": {"ref_month": "Jul 2024", "mfg": 49.4, "services": 50.1},
    "2024-08-31": {"ref_month": "Aug 2024", "mfg": 49.1, "services": 50.2},
    "2024-09-30": {"ref_month": "Sep 2024", "mfg": 49.8, "services": 49.9},
    "2024-10-31": {"ref_month": "Oct 2024", "mfg": 50.1, "services": 50.2},
    "2024-11-30": {"ref_month": "Nov 2024", "mfg": 50.3, "services": 50.1},
    "2024-12-31": {"ref_month": "Dec 2024", "mfg": 50.1, "services": 52.2},
    # 2025
    "2025-01-27": {"ref_month": "Jan 2025", "mfg": 49.1, "services": 50.3},  # Lunar NY early
    "2025-02-28": {"ref_month": "Feb 2025", "mfg": 50.2, "services": 51.4},
    "2025-03-31": {"ref_month": "Mar 2025", "mfg": 50.5, "services": 52.0},
    "2025-04-30": {"ref_month": "Apr 2025", "mfg": 49.0, "services": 50.4},
    "2025-05-31": {"ref_month": "May 2025", "mfg": 49.5, "services": 51.1},
    "2025-06-30": {"ref_month": "Jun 2025", "mfg": 49.7, "services": 51.5},
    "2025-07-31": {"ref_month": "Jul 2025", "mfg": 49.3, "services": 50.8},
    "2025-08-31": {"ref_month": "Aug 2025", "mfg": 49.5, "services": 50.5},
    "2025-09-30": {"ref_month": "Sep 2025", "mfg": 50.0, "services": 51.2},
    "2025-10-31": {"ref_month": "Oct 2025", "mfg": 50.2, "services": 51.0},
    "2025-11-30": {"ref_month": "Nov 2025", "mfg": 49.8, "services": 50.5},
    "2025-12-31": {"ref_month": "Dec 2025", "mfg": 49.5, "services": 50.2},
    # 2026
    "2026-01-31": {"ref_month": "Jan 2026", "mfg": 49.8, "services": 50.8},
    "2026-02-28": {"ref_month": "Feb 2026", "mfg": 50.5, "services": 51.5},
    "2026-03-31": {"ref_month": "Mar 2026", "mfg": 50.2, "services": 51.0},
    "2026-04-30": {"ref_month": "Apr 2026", "mfg": 49.5, "services": 50.5},
}

# Services PMI is released ~3 days after Manufacturing PMI
# We'll compute services release dates as MFG date + 3 business days

# ═══════════════════════════════════════════════════════════════
# SESSION DEFINITIONS (UTC)
# ═══════════════════════════════════════════════════════════════

SESSIONS = {
    "NBS_RELEASE":      {"start": "01:00", "end": "02:00", "desc": "NBS PMI release window (09:00 CST)"},
    "ASIA_OPEN":        {"start": "01:00", "end": "05:00", "desc": "Asia open reaction (09:00-13:00 CST)"},
    "ASIA_MORNING":     {"start": "01:00", "end": "07:00", "desc": "Full Asia morning (09:00-15:00 CST)"},
    "AUSTRALIA_MIDDAY": {"start": "03:00", "end": "06:00", "desc": "Australia mid-day (11:00-14:00 AEST)"},
    "EUROPE_MORNING":   {"start": "07:00", "end": "11:00", "desc": "Europe morning (08:00-12:00 CET)"},
    "EUROPE_AFTERNOON": {"start": "11:00", "end": "15:00", "desc": "Europe afternoon"},
    "US_OPEN":          {"start": "13:30", "end": "17:00", "desc": "US open"},
    "ASIA_REOPEN":      {"start": "01:00", "end": "05:00", "desc": "Asia re-open (next day)"},
    "FULL_DAY":         {"start": "01:00", "end": "23:00", "desc": "Full trading day"},
    "NEXT_24H":         {"start": "01:00", "end": "01:00", "desc": "Next 24 hours"},  # special handling
}

# ═══════════════════════════════════════════════════════════════
# LOAD ETH DATA
# ═══════════════════════════════════════════════════════════════

def load_eth_data(csv_path):
    """Load and prepare ETH 15m data."""
    df = pd.read_csv(csv_path)
    df['Open time'] = pd.to_datetime(df['Open time'])
    df = df.set_index('Open time')
    df = df.sort_index()
    # Ensure timezone-naive (UTC)
    if df.index.tz is not None:
        df.index = df.index.tz_localize(None)
    return df

def get_bar_at_or_after(df, dt):
    """Get the first bar at or after datetime."""
    mask = df.index >= dt
    if mask.any():
        return df.loc[mask].iloc[0]
    return None

def get_bar_at_or_before(df, dt):
    """Get the last bar at or before datetime."""
    mask = df.index <= dt
    if mask.any():
        return df.loc[mask].iloc[-1]
    return None

# ═══════════════════════════════════════════════════════════════
# SESSION RETURN CALCULATOR
# ═══════════════════════════════════════════════════════════════

def calc_session_return(df, release_date, session_def, next_day=False):
    """Calculate return during a session window on release date.
    
    Args:
        df: ETH 15m DataFrame
        release_date: date string YYYY-MM-DD
        session_def: dict with 'start' and 'end' (HH:MM UTC)
        next_day: if True, measure the next calendar day's session
    
    Returns:
        dict with open, close, high, low, return_pct, max_drawdown, max_rally
    """
    dt = pd.Timestamp(release_date)
    if next_day:
        dt = dt + timedelta(days=1)
    
    start_h, start_m = map(int, session_def['start'].split(':'))
    end_h, end_m = map(int, session_def['end'].split(':'))
    
    session_start = dt.replace(hour=start_h, minute=start_m, second=0)
    session_end = dt.replace(hour=end_h, minute=end_m, second=0)
    
    # Handle overnight sessions
    if session_end <= session_start:
        session_end += timedelta(days=1)
    
    # Get bars in window
    mask = (df.index >= session_start) & (df.index <= session_end)
    session_bars = df.loc[mask]
    
    if len(session_bars) < 2:
        return None
    
    open_price = session_bars.iloc[0]['Open']
    close_price = session_bars.iloc[-1]['Close']
    high_price = session_bars['High'].max()
    low_price = session_bars['Low'].min()
    
    ret_pct = (close_price - open_price) / open_price * 100
    max_drawdown = (low_price - open_price) / open_price * 100
    max_rally = (high_price - open_price) / open_price * 100
    
    # V-bottom detection: dip then recovery within session
    # Check if price dipped >0.3% then recovered to near open
    v_bottom = False
    if max_drawdown < -0.3 and ret_pct > max_drawdown * 0.5:
        v_bottom = True
    
    return {
        'open': open_price,
        'close': close_price,
        'high': high_price,
        'low': low_price,
        'return_pct': ret_pct,
        'max_drawdown': max_drawdown,
        'max_rally': max_rally,
        'bars': len(session_bars),
        'v_bottom': v_bottom,
    }

# ═══════════════════════════════════════════════════════════════
# MAIN BACKTEST
# ═══════════════════════════════════════════════════════════════

def run_backtest(csv_path, start_year=2018):
    """Run the full NBS PMI session backtest."""
    
    print("=" * 80)
    print("  NBS PMI SESSION ITINERARY BACKTEST")
    print("  Period: 2018 → 2026")
    print("=" * 80)
    
    # Load data
    print("\n📊 Loading ETH 15m data...")
    df = load_eth_data(csv_path)
    print(f"   Loaded {len(df):,} bars: {df.index[0]} → {df.index[-1]}")
    
    # Filter releases to start_year+
    releases = {k: v for k, v in NBS_MFG_RELEASES.items() 
                if int(k[:4]) >= start_year}
    
    print(f"\n📋 NBS PMI releases to test: {len(releases)}")
    
    # ── ANALYSIS 1: Manufacturing PMI Release Day ──
    print("\n" + "=" * 80)
    print("  ANALYSIS 1: MANUFACTURING PMI — RELEASE DAY SESSION RETURNS")
    print("=" * 80)
    
    mfg_results = []
    
    for release_date, info in sorted(releases.items()):
        dt = pd.Timestamp(release_date)
        
        # Skip if no data
        if dt < df.index[0] or dt > df.index[-1]:
            continue
        
        # Check if it's actually a trading day
        day_mask = df.index.date == dt.date()
        if not day_mask.any():
            # Try next business day
            for offset in range(1, 4):
                alt_dt = dt + timedelta(days=offset)
                if (df.index.date == alt_dt.date()).any():
                    dt = alt_dt
                    release_date = dt.strftime('%Y-%m-%d')
                    break
            else:
                continue
        
        row = {'date': release_date, 'ref_month': info['ref_month'],
               'mfg_pmi': info['mfg'], 'services_pmi': info.get('services')}
        
        # Classify PMI
        mfg = info['mfg']
        if mfg >= 51.0:
            row['mfg_signal'] = 'STRONG'
        elif mfg >= 50.0:
            row['mfg_signal'] = 'EXPANDING'
        elif mfg >= 49.0:
            row['mfg_signal'] = 'WEAK'
        else:
            row['mfg_signal'] = 'CONTRACTING'
        
        # Surprise vs 50 (breakeven)
        row['mfg_surprise'] = mfg - 50.0
        
        # Session returns
        for sess_name, sess_def in SESSIONS.items():
            if sess_name == 'ASIA_REOPEN' or sess_name == 'NEXT_24H':
                continue  # These are next-day
            result = calc_session_return(df, release_date, sess_def)
            if result:
                row[f'{sess_name}_ret'] = result['return_pct']
                row[f'{sess_name}_dd'] = result['max_drawdown']
                row[f'{sess_name}_rally'] = result['max_rally']
                row[f'{sess_name}_vbottom'] = result['v_bottom']
        
        # Next day Asia re-open
        result = calc_session_return(df, release_date, SESSIONS['ASIA_REOPEN'], next_day=True)
        if result:
            row['ASIA_REOPEN_ret'] = result['return_pct']
            row['ASIA_REOPEN_dd'] = result['max_drawdown']
            row['ASIA_REOPEN_rally'] = result['max_rally']
        
        # 24h return
        result = calc_session_return(df, release_date, SESSIONS['ASIA_OPEN'])
        if result:
            # Manual 24h calc
            dt_ts = pd.Timestamp(release_date)
            start = dt_ts.replace(hour=1, minute=0)
            end = start + timedelta(hours=24)
            mask = (df.index >= start) & (df.index <= end)
            bars_24h = df.loc[mask]
            if len(bars_24h) >= 2:
                row['24h_ret'] = (bars_24h.iloc[-1]['Close'] - bars_24h.iloc[0]['Open']) / bars_24h.iloc[0]['Open'] * 100
        
        mfg_results.append(row)
    
    mfg_df = pd.DataFrame(mfg_results)
    
    if len(mfg_df) == 0:
        print("❌ No valid PMI release days found in data range!")
        return
    
    # ── SUMMARY STATS ──
    print(f"\n📊 Valid release days: {len(mfg_df)}")
    print(f"   Date range: {mfg_df['date'].iloc[0]} → {mfg_df['date'].iloc[-1]}")
    
    # PMI distribution
    print(f"\n📈 PMI Signal Distribution:")
    for signal in ['STRONG', 'EXPANDING', 'WEAK', 'CONTRACTING']:
        count = (mfg_df['mfg_signal'] == signal).sum()
        if count > 0:
            avg_pmi = mfg_df.loc[mfg_df['mfg_signal'] == signal, 'mfg_pmi'].mean()
            print(f"   {signal:12s}: {count:3d} releases  (avg PMI: {avg_pmi:.1f})")
    
    # Session returns by signal
    print("\n" + "-" * 80)
    print("  SESSION RETURNS BY PMI SIGNAL (Manufacturing)")
    print("-" * 80)
    
    session_cols = [c for c in mfg_df.columns if c.endswith('_ret')]
    
    for sess_col in ['ASIA_OPEN_ret', 'AUSTRALIA_MIDDAY_ret', 'EUROPE_MORNING_ret', 
                      'US_OPEN_ret', 'ASIA_REOPEN_ret', '24h_ret']:
        if sess_col not in mfg_df.columns:
            continue
        sess_name = sess_col.replace('_ret', '')
        print(f"\n  ── {sess_name} ──")
        
        for signal in ['STRONG', 'EXPANDING', 'WEAK', 'CONTRACTING']:
            subset = mfg_df[mfg_df['mfg_signal'] == signal]
            if len(subset) == 0:
                continue
            vals = subset[sess_col].dropna()
            if len(vals) == 0:
                continue
            print(f"    {signal:12s}: n={len(vals):3d}  "
                  f"avg={vals.mean():+.3f}%  med={vals.median():+.3f}%  "
                  f"win={((vals > 0).sum()/len(vals)*100):.0f}%  "
                  f"std={vals.std():.3f}%")
        
        # Overall
        vals = mfg_df[sess_col].dropna()
        if len(vals) > 0:
            print(f"    {'OVERALL':12s}: n={len(vals):3d}  "
                  f"avg={vals.mean():+.3f}%  med={vals.median():+.3f}%  "
                  f"win={((vals > 0).sum()/len(vals)*100):.0f}%  "
                  f"std={vals.std():.3f}%")
    
    # ── ANALYSIS 2: V-Bottom Detection ──
    print("\n" + "=" * 80)
    print("  ANALYSIS 2: V-BOTTOM PATTERN DETECTION")
    print("  Claim: NBS miss → swift V-bottom at Asia Open")
    print("=" * 80)
    
    for signal in ['CONTRACTING', 'WEAK', 'STRONG', 'EXPANDING']:
        subset = mfg_df[mfg_df['mfg_signal'] == signal]
        if len(subset) == 0:
            continue
        vbottom_col = 'ASIA_OPEN_vbottom'
        if vbottom_col in subset.columns:
            vb_count = subset[vbottom_col].sum()
            print(f"\n  {signal} PMI (n={len(subset)}):")
            print(f"    V-bottom at Asia Open: {vb_count}/{len(subset)} ({vb_count/len(subset)*100:.0f}%)")
            
            # Max drawdown distribution
            dd_col = 'ASIA_OPEN_dd'
            if dd_col in subset.columns:
                dd_vals = subset[dd_col].dropna()
                if len(dd_vals) > 0:
                    print(f"    Max drawdown: avg={dd_vals.mean():.3f}%  "
                          f"med={dd_vals.median():.3f}%  "
                          f"worst={dd_vals.min():.3f}%")
            
            # Return distribution
            ret_col = 'ASIA_OPEN_ret'
            if ret_col in subset.columns:
                ret_vals = subset[ret_col].dropna()
                if len(ret_vals) > 0:
                    print(f"    Session return: avg={ret_vals.mean():+.3f}%  "
                          f"med={ret_vals.median():+.3f}%  "
                          f"win={((ret_vals > 0).sum()/len(ret_vals)*100):.0f}%")
    
    # ── ANALYSIS 3: Session Transmission Chain ──
    print("\n" + "=" * 80)
    print("  ANALYSIS 3: SESSION TRANSMISSION CHAIN")
    print("  Claim: Asia → Australia → Europe → US → Asia Re-open")
    print("=" * 80)
    
    transmission_sessions = ['ASIA_OPEN', 'AUSTRALIA_MIDDAY', 'EUROPE_MORNING', 
                              'US_OPEN', 'ASIA_REOPEN']
    
    print(f"\n  {'Session':<22s} {'N':>4s} {'Avg Ret':>9s} {'Median':>9s} {'Win%':>6s} {'Std':>8s} {'DD Avg':>9s} {'Rally Avg':>10s}")
    print("  " + "-" * 80)
    
    for sess in transmission_sessions:
        ret_col = f'{sess}_ret'
        dd_col = f'{sess}_dd'
        rally_col = f'{sess}_rally'
        
        if ret_col not in mfg_df.columns:
            continue
        
        rets = mfg_df[ret_col].dropna()
        dds = mfg_df[dd_col].dropna() if dd_col in mfg_df.columns else pd.Series()
        rallies = mfg_df[rally_col].dropna() if rally_col in mfg_df.columns else pd.Series()
        
        if len(rets) == 0:
            continue
        
        print(f"  {sess:<22s} {len(rets):>4d} {rets.mean():>+9.3f}% {rets.median():>+9.3f}% "
              f"{((rets > 0).sum()/len(rets)*100):>5.0f}% {rets.std():>8.3f}% "
              f"{dds.mean():>+9.3f}% {rallies.mean():>+10.3f}%")
    
    # ── ANALYSIS 4: Surprise vs Return Correlation ──
    print("\n" + "=" * 80)
    print("  ANALYSIS 4: PMI SURPRISE vs RETURN CORRELATION")
    print("  Does a bigger miss (PMI < 50) cause bigger moves?")
    print("=" * 80)
    
    for sess in ['ASIA_OPEN', 'EUROPE_MORNING', 'US_OPEN', '24h']:
        ret_col = f'{sess}_ret'
        if ret_col not in mfg_df.columns:
            continue
        
        valid = mfg_df[['mfg_surprise', ret_col]].dropna()
        if len(valid) < 5:
            continue
        
        corr = valid['mfg_surprise'].corr(valid[ret_col])
        
        # Split by surprise direction
        neg_surprise = valid[valid['mfg_surprise'] < 0]
        pos_surprise = valid[valid['mfg_surprise'] > 0]
        
        print(f"\n  {sess}:")
        print(f"    Correlation (surprise vs return): {corr:+.3f}  (n={len(valid)})")
        if len(neg_surprise) > 0:
            print(f"    PMI < 50 (miss):    avg ret={neg_surprise[ret_col].mean():+.3f}%  n={len(neg_surprise)}")
        if len(pos_surprise) > 0:
            print(f"    PMI ≥ 50 (beat):    avg ret={pos_surprise[ret_col].mean():+.3f}%  n={len(pos_surprise)}")
    
    # ── ANALYSIS 5: Extreme Events Deep Dive ──
    print("\n" + "=" * 80)
    print("  ANALYSIS 5: EXTREME PMI EVENTS (COVID, shocks)")
    print("=" * 80)
    
    extreme = mfg_df[(mfg_df['mfg_pmi'] < 45) | (mfg_df['mfg_pmi'] > 53)]
    if len(extreme) > 0:
        print(f"\n  Found {len(extreme)} extreme PMI events:")
        for _, row in extreme.iterrows():
            ret_str = f"Asia:{row.get('ASIA_OPEN_ret', 0):+.2f}%" if 'ASIA_OPEN_ret' in row else ""
            ret_str += f" EU:{row.get('EUROPE_MORNING_ret', 0):+.2f}%" if 'EUROPE_MORNING_ret' in row else ""
            ret_str += f" 24h:{row.get('24h_ret', 0):+.2f}%" if '24h_ret' in row else ""
            print(f"    {row['date']}  PMI={row['mfg_pmi']:.1f}  {row['mfg_signal']:12s}  {ret_str}")
    else:
        print("  No extreme events found in range.")
    
    # ── ANALYSIS 6: Actionable Summary ──
    print("\n" + "=" * 80)
    print("  ANALYSIS 6: ACTIONABLE SUMMARY")
    print("=" * 80)
    
    # What happens after a miss (PMI < 50)?
    miss_days = mfg_df[mfg_df['mfg_pmi'] < 50]
    beat_days = mfg_df[mfg_df['mfg_pmi'] >= 50]
    
    print(f"\n  PMI MISS (< 50): {len(miss_days)} events")
    for sess in ['ASIA_OPEN', 'EUROPE_MORNING', '24h']:
        ret_col = f'{sess}_ret'
        if ret_col in miss_days.columns:
            vals = miss_days[ret_col].dropna()
            if len(vals) > 0:
                print(f"    {sess:22s}: avg={vals.mean():+.3f}%  win={((vals > 0).sum()/len(vals)*100):.0f}%  n={len(vals)}")
    
    print(f"\n  PMI BEAT (≥ 50): {len(beat_days)} events")
    for sess in ['ASIA_OPEN', 'EUROPE_MORNING', '24h']:
        ret_col = f'{sess}_ret'
        if ret_col in beat_days.columns:
            vals = beat_days[ret_col].dropna()
            if len(vals) > 0:
                print(f"    {sess:22s}: avg={vals.mean():+.3f}%  win={((vals > 0).sum()/len(vals)*100):.0f}%  n={len(vals)}")
    
    # Contraction PMI (< 49) specifically
    contraction = mfg_df[mfg_df['mfg_pmi'] < 49]
    if len(contraction) > 0:
        print(f"\n  CONTRACTION PMI (< 49): {len(contraction)} events")
        for sess in ['ASIA_OPEN', 'EUROPE_MORNING', '24h']:
            ret_col = f'{sess}_ret'
            if ret_col in contraction.columns:
                vals = contraction[ret_col].dropna()
                if len(vals) > 0:
                    print(f"      {sess:22s}: avg={vals.mean():+.3f}%  win={((vals > 0).sum()/len(vals)*100):.0f}%  n={len(vals)}")
    
    # ── Save detailed CSV ──
    output_path = os.path.join(os.path.dirname(csv_path), 'nbs_pmi_backtest.csv')
    mfg_df.to_csv(output_path, index=False)
    print(f"\n💾 Detailed results saved: {output_path}")
    
    return mfg_df


if __name__ == '__main__':
    csv_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'eth_15m_merged.csv')
    if not os.path.exists(csv_path):
        print(f"❌ ETH data not found: {csv_path}")
        sys.exit(1)
    
    start_year = int(sys.argv[1]) if len(sys.argv) > 1 else 2018
    run_backtest(csv_path, start_year)

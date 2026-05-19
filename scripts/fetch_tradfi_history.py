#!/usr/bin/env python3
"""
Fetch historical tradfi data for M66-M71 backtesting.

Stores OHLCV data for 6 symbols, aligned to the ETH 15m grid.
Two data tiers per symbol:
  - Daily: full history (2017+ for most symbols)
  - Intraday: last 60 days at finest available yfinance interval

Usage:
    python scripts/fetch_tradfi_history.py              # fetch all
    python scripts/fetch_tradfi_history.py --symbol DXY  # fetch one
    python scripts/fetch_tradfi_history.py --align        # re-align to ETH grid only

Output:
    data/tradfi/<symbol>_daily.csv
    data/tradfi/<symbol>_15m.csv     (where available)
    data/tradfi/aligned.csv          (merged, forward-filled to ETH 15m grid)
"""

import argparse
import os
import sys
import time
from datetime import datetime, timedelta

import pandas as pd

# ── Symbol definitions ──
SYMBOLS = {
    'USDJPY': {
        'yf_ticker': 'JPY=X',
        'description': 'USD/JPY exchange rate',
        'intraday_interval': '15m',
        'intraday_period': '60d',
        'daily_period': 'max',
        'module': 'M66',
    },
    'DXY': {
        'yf_ticker': 'DX-Y.NYB',
        'description': 'US Dollar Index',
        'intraday_interval': '15m',
        'intraday_period': '60d',
        'daily_period': 'max',
        'module': 'M67',
    },
    'TNX': {
        'yf_ticker': '^TNX',
        'description': '10Y Treasury Yield (×10)',
        'intraday_interval': '1h',
        'intraday_period': '730d',
        'daily_period': 'max',
        'module': 'M68',
        'transform': None,  # yfinance returns ^TNX as yield/10; m68 module handles conversion
    },
    'VIX': {
        'yf_ticker': '^VIX',
        'description': 'CBOE Volatility Index',
        'intraday_interval': None,  # VIX intraday not useful for backtest
        'daily_period': 'max',
        'module': 'M69',
    },
    'WTI': {
        'yf_ticker': 'CL=F',
        'description': 'WTI Crude Oil Futures',
        'intraday_interval': '4h',
        'intraday_period': '60d',
        'daily_period': 'max',
        'module': 'M70',
    },
    'GOLD': {
        'yf_ticker': 'GC=F',
        'description': 'Gold Futures',
        'intraday_interval': '4h',
        'intraday_period': '60d',
        'daily_period': 'max',
        'module': 'M71',
    },
}

DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data', 'tradfi')
ETH_CSV = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data', 'eth_15m_merged.csv')
if not os.path.exists(ETH_CSV):
    ETH_CSV = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'eth_15m_merged.csv')


def fetch_yfinance(ticker, period='max', interval='1d', max_retries=3):
    """Fetch OHLCV from yfinance with retry logic.

    Returns:
        DataFrame with columns: datetime, Open, High, Low, Close, Volume
        or None on failure
    """
    try:
        import yfinance as yf
    except ImportError:
        print(f"  ❌ yfinance not installed")
        return None

    for attempt in range(max_retries):
        try:
            df = yf.download(ticker, period=period, interval=interval,
                             progress=False, auto_adjust=True)
            if df is None or len(df) == 0:
                return None

            # Flatten MultiIndex columns (yfinance >= 1.0)
            if hasattr(df.columns, 'levels') and len(df.columns.levels) > 1:
                df.columns = df.columns.droplevel(1)

            # Normalize index to column
            df = df.reset_index()
            # Rename the first column (Datetime or Date) to datetime
            first_col = df.columns[0]
            df = df.rename(columns={first_col: 'datetime'})

            # Ensure datetime is string for CSV storage
            df['datetime'] = pd.to_datetime(df['datetime']).dt.strftime('%Y-%m-%d %H:%M:%S')

            # Keep only OHLCV
            for col in ['Open', 'High', 'Low', 'Close', 'Volume']:
                if col not in df.columns:
                    df[col] = 0.0

            df = df[['datetime', 'Open', 'High', 'Low', 'Close', 'Volume']]
            return df

        except Exception as e:
            if attempt < max_retries - 1:
                time.sleep(2 * (attempt + 1))
            else:
                print(f"  ❌ Fetch failed after {max_retries} attempts: {e}")
                return None

    return None


def fetch_symbol(symbol_name, config, force=False):
    """Fetch daily + intraday data for a single symbol.

    Returns:
        (daily_path, intraday_path) — paths to saved CSVs, or None
    """
    os.makedirs(DATA_DIR, exist_ok=True)

    ticker = config['yf_ticker']
    desc = config['description']
    module = config['module']
    transform = config.get('transform')

    daily_path = os.path.join(DATA_DIR, f'{symbol_name}_daily.csv')
    intraday_path = os.path.join(DATA_DIR, f'{symbol_name}_{config.get("intraday_interval", "15m")}.csv')
    if config.get('intraday_interval') is None:
        intraday_path = None

    print(f"\n  📡 {module} {symbol_name} ({desc})")
    print(f"     Ticker: {ticker}")

    # ── Daily data ──
    if os.path.exists(daily_path) and not force:
        existing = pd.read_csv(daily_path)
        last = existing['datetime'].iloc[-1] if len(existing) > 0 else 'N/A'
        print(f"     Daily: {len(existing):,} bars (last: {last}) — skipping (use --force to refresh)")
    else:
        print(f"     Daily: fetching {config['daily_period']}...")
        df_daily = fetch_yfinance(ticker, period=config['daily_period'], interval='1d')
        if df_daily is not None:
            if transform:
                df_daily = transform(df_daily)
            df_daily.to_csv(daily_path, index=False)
            print(f"     Daily: ✅ {len(df_daily):,} bars → {daily_path}")
            print(f"            Range: {df_daily['datetime'].iloc[0]} → {df_daily['datetime'].iloc[-1]}")
        else:
            print(f"     Daily: ❌ No data returned")
            daily_path = None

    # ── Intraday data ──
    if intraday_path and config.get('intraday_interval'):
        if os.path.exists(intraday_path) and not force:
            existing = pd.read_csv(intraday_path)
            last = existing['datetime'].iloc[-1] if len(existing) > 0 else 'N/A'
            print(f"     {config['intraday_interval']}: {len(existing):,} bars (last: {last}) — skipping")
        else:
            print(f"     {config['intraday_interval']}: fetching {config['intraday_period']}...")
            df_intra = fetch_yfinance(
                ticker, period=config['intraday_period'],
                interval=config['intraday_interval'])
            if df_intra is not None:
                if transform:
                    df_intra = transform(df_intra)
                df_intra.to_csv(intraday_path, index=False)
                print(f"     {config['intraday_interval']}: ✅ {len(df_intra):,} bars → {intraday_path}")
            else:
                print(f"     {config['intraday_interval']}: ⚠️  No data (may not be available)")
                intraday_path = None

    return daily_path, intraday_path


def build_aligned_grid(eth_csv=None, output_path=None):
    """Build the aligned tradfi grid on the ETH 15m timestamp base.

    For each ETH 15m bar timestamp, picks the finest available tradfi data
    (intraday > daily) and forward-fills missing values.

    Output: data/tradfi/aligned.csv with columns:
        datetime, USDJPY, DXY, TNX, VIX, WTI, GOLD
    """
    eth_path = eth_csv or ETH_CSV
    out_path = output_path or os.path.join(DATA_DIR, 'aligned.csv')

    if not os.path.exists(eth_path):
        print(f"  ❌ ETH CSV not found: {eth_path}")
        return None

    print(f"\n  🔗 Building aligned tradfi grid...")

    # Load ETH timestamps
    eth = pd.read_csv(eth_path, usecols=['Open time'])
    eth.columns = ['datetime']
    eth['datetime'] = pd.to_datetime(eth['datetime'])
    eth = eth.sort_values('datetime').reset_index(drop=True)
    print(f"     ETH grid: {len(eth):,} bars ({eth['datetime'].iloc[0]} → {eth['datetime'].iloc[-1]})")

    # Load each symbol and merge
    merged = eth.copy()

    for symbol_name, config in SYMBOLS.items():
        col_name = symbol_name.lower()
        intra_interval = config.get('intraday_interval')
        intra_path = os.path.join(DATA_DIR, f'{symbol_name}_{intra_interval}.csv') if intra_interval else None
        daily_path = os.path.join(DATA_DIR, f'{symbol_name}_daily.csv')

        # Prefer intraday, fallback to daily
        df = None
        source = None
        if intra_path and os.path.exists(intra_path):
            df_intra = pd.read_csv(intra_path)
            df_intra['datetime'] = pd.to_datetime(df_intra['datetime'])
            if os.path.exists(daily_path):
                df_daily = pd.read_csv(daily_path)
                df_daily['datetime'] = pd.to_datetime(df_daily['datetime'])
                # Combine: daily for old data, intraday for recent
                # Intraday overwrites daily where timestamps overlap
                df_combined = pd.concat([df_daily, df_intra]).drop_duplicates(
                    subset='datetime', keep='last').sort_values('datetime').reset_index(drop=True)
                df = df_combined
                source = f'{intra_interval}+daily'
            else:
                df = df_intra
                source = intra_interval
        elif os.path.exists(daily_path):
            df = pd.read_csv(daily_path)
            source = 'daily'

        if df is None or len(df) == 0:
            print(f"     {symbol_name}: ⚠️  No data files found — filling with NaN")
            merged[col_name] = float('nan')
            continue

        df['datetime'] = pd.to_datetime(df['datetime'])
        df = df.sort_values('datetime').drop_duplicates(subset='datetime', keep='last')
        df = df.rename(columns={'Close': col_name})
        df = df[['datetime', col_name]]

        # Merge_asof: for each ETH timestamp, find the most recent tradfi bar
        merged = pd.merge_asof(
            merged, df,
            on='datetime',
            direction='backward',
            tolerance=pd.Timedelta(days=7)  # allow up to 7d gap (weekends, holidays)
        )

        coverage = merged[col_name].notna().mean() * 100
        n_bars = merged[col_name].notna().sum()
        print(f"     {symbol_name}: {n_bars:,} matched ({coverage:.1f}%) from {source}")

    # Save
    merged['datetime'] = merged['datetime'].dt.strftime('%Y-%m-%d %H:%M:%S')
    merged.to_csv(out_path, index=False)
    print(f"\n  ✅ Aligned grid: {len(merged):,} rows → {out_path}")

    # Summary
    print(f"\n  Coverage summary:")
    for symbol_name in SYMBOLS:
        col = symbol_name.lower()
        if col in merged.columns:
            pct = merged[col].notna().mean() * 100
            n = merged[col].notna().sum()
            print(f"    {symbol_name:>8}: {n:>8,} bars ({pct:5.1f}%)")

    return out_path


def main():
    parser = argparse.ArgumentParser(description='Fetch tradfi history for M66-M71 backtesting')
    parser.add_argument('--symbol', type=str, help='Fetch only this symbol (e.g. DXY, USDJPY)')
    parser.add_argument('--force', action='store_true', help='Re-fetch even if data exists')
    parser.add_argument('--align', action='store_true', help='Only rebuild aligned grid (skip fetch)')
    parser.add_argument('--eth-csv', type=str, help='Path to ETH 15m CSV (default: auto-detect)')
    args = parser.parse_args()

    print("=" * 60)
    print("  JIMI — Tradfi History Fetcher (M66-M71)")
    print("=" * 60)

    if not args.align:
        if args.symbol:
            sym = args.symbol.upper()
            if sym not in SYMBOLS:
                print(f"\n  ❌ Unknown symbol: {sym}")
                print(f"     Available: {', '.join(SYMBOLS.keys())}")
                sys.exit(1)
            fetch_symbol(sym, SYMBOLS[sym], force=args.force)
        else:
            for sym, cfg in SYMBOLS.items():
                fetch_symbol(sym, cfg, force=args.force)
                time.sleep(1)  # rate limit

    # Build aligned grid
    build_aligned_grid(eth_csv=args.eth_csv)

    print(f"\n{'=' * 60}")
    print(f"  Done. Files in: {DATA_DIR}")
    print(f"{'=' * 60}")


if __name__ == '__main__':
    main()

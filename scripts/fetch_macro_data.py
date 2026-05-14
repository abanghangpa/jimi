#!/usr/bin/env python3
"""
Fetch PPI & CPI YoY data from BLS (Bureau of Labor Statistics) public API.

No API key required (v1 endpoint, 25 queries/day limit).
Updates local cache at data/macro_data.json which M23 reads automatically.

Usage:
    python3 scripts/fetch_macro_data.py
    python3 scripts/fetch_macro_data.py --refresh   # force re-fetch
"""

import json
import os
import sys
import argparse
from datetime import datetime, timedelta

import requests

# ═══════════════════════════════════════════════════════════════
# BLS SERIES IDs
# ═══════════════════════════════════════════════════════════════

# PPI: Producer Price Index - Final Demand (Seasonally Adjusted)
# This is the headline PPI number reported in financial media
BLS_SERIES = {
    # PPI Final Demand (SA) - Monthly, YoY calc from this
    'PPI_FD': 'WPSFD49107',
    # PPI Final Demand Less Food and Energy (Core PPI, SA)
    'PPI_CORE': 'WPSFD49104',
    # CPI All Urban Consumers (SA) - Seasonally Adjusted (the headline CPI)
    'CPI_ALL': 'CUSR0000SA0',
    # CPI All Urban Consumers Less Food and Energy (Core CPI, SA)
    'CPI_CORE': 'CUSR0000SA0L1E',
}

BLS_API_URL = 'https://api.bls.gov/publicAPI/v1/timeseries/data/'

# Cache file
CACHE_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data')
CACHE_FILE = os.path.join(CACHE_DIR, 'macro_data.json')

# BLS v1 allows up to 10 years per request, max 25 queries/day
BLS_START_YEAR = 2017
BLS_END_YEAR = datetime.now().year


def fetch_bls_series(series_id, start_year=None, end_year=None):
    """Fetch a single BLS time series.

    Args:
        series_id: BLS series ID (e.g., 'WPSFD49107')
        start_year: first year of data
        end_year: last year of data

    Returns:
        list of (year, period, value) tuples, or None on error
    """
    start = start_year or BLS_START_YEAR
    end = end_year or BLS_END_YEAR

    payload = {
        'seriesid': [series_id],
        'startyear': str(start),
        'endyear': str(end),
    }

    try:
        resp = requests.post(BLS_API_URL, json=payload, timeout=30)
        resp.raise_for_status()
        data = resp.json()

        if data.get('status') != 'REQUEST_SUCCEEDED':
            print(f"  ⚠️  BLS API error for {series_id}: {data.get('message', 'unknown')}")
            return None

        series = data['Results']['series'][0]
        if series.get('seriesID') != series_id:
            print(f"  ⚠️  Series ID mismatch: expected {series_id}, got {series.get('seriesID')}")
            return None

        results = []
        for item in series.get('data', []):
            year = int(item['year'])
            period = item['period']  # e.g., 'M01' for January
            value = item.get('value')
            if value and period.startswith('M') and value != '-':
                month = int(period[1:])
                if 1 <= month <= 12:
                    try:
                        results.append({
                            'year': year,
                            'month': month,
                            'value': float(value),
                        })
                    except (ValueError, TypeError):
                        continue

        # Sort by date
        results.sort(key=lambda x: (x['year'], x['month']))
        return results

    except requests.exceptions.RequestException as e:
        print(f"  ⚠️  Network error fetching {series_id}: {e}")
        return None
    except (json.JSONDecodeError, KeyError, ValueError) as e:
        print(f"  ⚠️  Parse error for {series_id}: {e}")
        return None


def compute_yoy(monthly_data):
    """Compute YoY (Year-over-Year) percentage change from monthly levels.

    Args:
        monthly_data: list of {'year', 'month', 'value'} dicts

    Returns:
        dict of 'YYYY-MM' → YoY% (e.g., '2024-01' → 2.7)
    """
    if not monthly_data:
        return {}

    # Build lookup: (year, month) → value
    lookup = {}
    for item in monthly_data:
        lookup[(item['year'], item['month'])] = item['value']

    yoy = {}
    for item in monthly_data:
        y, m, val = item['year'], item['month'], item['value']
        prev_key = (y - 1, m)
        if prev_key in lookup and lookup[prev_key] != 0:
            pct = (val - lookup[prev_key]) / lookup[prev_key] * 100
            key = f"{y:04d}-{m:02d}"
            yoy[key] = round(pct, 1)

    return yoy


def fetch_all():
    """Fetch all macro series from BLS and compute YoY.

    Returns:
        dict with ppi_yoy, ppi_core_yoy, cpi_yoy, cpi_core_yoy,
        plus raw monthly data.
    """
    print("  Fetching PPI & CPI data from BLS...")

    result = {
        'fetched_at': datetime.now().isoformat(),
        'source': 'BLS (Bureau of Labor Statistics)',
        'series': {},
        'yoy': {},
        'latest': {},
    }

    for name, series_id in BLS_SERIES.items():
        print(f"    {name} ({series_id})...", end=' ')
        monthly = fetch_bls_series(series_id)
        if monthly is None:
            print("FAILED")
            continue

        print(f"OK ({len(monthly)} months)")

        result['series'][name] = monthly
        yoy = compute_yoy(monthly)
        result['yoy'][name] = yoy

        # Latest value
        if monthly:
            latest = monthly[-1]
            latest_key = f"{latest['year']:04d}-{latest['month']:02d}"
            latest_yoy = yoy.get(latest_key)
            result['latest'][name] = {
                'date': latest_key,
                'level': latest['value'],
                'yoy': latest_yoy,
            }

    return result


def get_prev_month_yoy(yoy_dict, date_key):
    """Get previous month's YoY value.

    Args:
        yoy_dict: dict of 'YYYY-MM' → YoY%
        date_key: 'YYYY-MM' string

    Returns:
        float or None
    """
    y, m = int(date_key[:4]), int(date_key[5:7])
    prev_m = m - 1 if m > 1 else 12
    prev_y = y if m > 1 else y - 1
    prev_key = f"{prev_y:04d}-{prev_m:02d}"
    return yoy_dict.get(prev_key)


def save_cache(data):
    """Save fetched data to local cache."""
    os.makedirs(CACHE_DIR, exist_ok=True)
    with open(CACHE_FILE, 'w') as f:
        json.dump(data, f, indent=2)
    print(f"  💾 Saved: {CACHE_FILE}")


def load_cache():
    """Load cached data, or None if not found/too old."""
    if not os.path.exists(CACHE_FILE):
        return None
    try:
        with open(CACHE_FILE) as f:
            data = json.load(f)
        return data
    except (json.JSONDecodeError, IOError):
        return None


def is_cache_fresh(data, max_age_hours=24):
    """Check if cached data is fresh enough."""
    if data is None:
        return False
    fetched = data.get('fetched_at')
    if not fetched:
        return False
    try:
        fetched_dt = datetime.fromisoformat(fetched)
        age = datetime.now() - fetched_dt
        return age < timedelta(hours=max_age_hours)
    except (ValueError, TypeError):
        return False


def get_current_macro_values(refresh=False):
    """Get current PPI/CPI values for M23 scoring.

    This is the main interface for the scanner. Returns a dict with
    values ready to plug into CONFIG.

    Args:
        refresh: force re-fetch from BLS

    Returns:
        dict with keys matching CONFIG names, or None on error
    """
    # Load or fetch
    cache = load_cache()
    if refresh or not is_cache_fresh(cache):
        print("  📥 Fetching fresh macro data from BLS...")
        cache = fetch_all()
        if cache and cache.get('latest'):
            save_cache(cache)
        else:
            print("  ⚠️  BLS fetch failed, using cached data")
            cache = load_cache()
            if cache is None:
                return None

    if not cache or not cache.get('latest'):
        return None

    ppi_latest = cache['latest'].get('PPI_FD', {})
    cpi_latest = cache['latest'].get('CPI_ALL', {})
    ppi_core_latest = cache['latest'].get('PPI_CORE', {})
    cpi_core_latest = cache['latest'].get('CPI_CORE', {})

    # Previous month for direction calc
    ppi_date = ppi_latest.get('date', '')
    cpi_date = cpi_latest.get('date', '')
    ppi_prev = get_prev_month_yoy(cache['yoy'].get('PPI_FD', {}), ppi_date) if ppi_date else None
    cpi_prev = get_prev_month_yoy(cache['yoy'].get('CPI_ALL', {}), cpi_date) if cpi_date else None

    result = {
        'M22_PPI_YOY': ppi_latest.get('yoy'),
        'M22_PPI_PREV_YOY': ppi_prev,
        'M22_PPI_DATE': ppi_date,
        'M22_PPI_LEVEL': ppi_latest.get('level'),
        'M22_CPI_YOY': cpi_latest.get('yoy'),
        'M22_CPI_PREV_YOY': cpi_prev,
        'M22_CPI_DATE': cpi_date,
        'M22_CPI_LEVEL': cpi_latest.get('level'),
        'M22_PPI_CORE_YOY': ppi_core_latest.get('yoy'),
        'M22_CPI_CORE_YOY': cpi_core_latest.get('yoy'),
    }

    # Infer PPI MoM from levels
    if ppi_latest.get('level') and ppi_prev is not None:
        # MoM approximation from YoY delta (rough but useful)
        pass

    return result


def print_status():
    """Print current macro data status."""
    cache = load_cache()
    if cache is None:
        print("  No cached data. Run: python3 scripts/fetch_macro_data.py")
        return

    fetched = cache.get('fetched_at', 'unknown')
    print(f"\n  📊 MACRO DATA CACHE")
    print(f"  Fetched: {fetched}")
    print(f"  Source:  {cache.get('source', 'unknown')}")
    print()

    latest = cache.get('latest', {})
    for name in ['PPI_FD', 'PPI_CORE', 'CPI_ALL', 'CPI_CORE']:
        info = latest.get(name, {})
        if info:
            yoy = info.get('yoy')
            yoy_str = f"{yoy:+.1f}%" if yoy is not None else "N/A"
            print(f"    {name:<12} {info.get('date', '?'):>8}  "
                  f"level={info.get('level', '?'):>8}  YoY={yoy_str}")


def main():
    parser = argparse.ArgumentParser(description='Fetch PPI/CPI data from BLS')
    parser.add_argument('--refresh', action='store_true', help='Force re-fetch')
    parser.add_argument('--status', action='store_true', help='Show cached data status')
    args = parser.parse_args()

    if args.status:
        print_status()
        return

    data = get_current_macro_values(refresh=args.refresh)
    if data:
        print(f"\n  Current values for M23:")
        for k, v in sorted(data.items()):
            if v is not None:
                print(f"    {k}: {v}")
    else:
        print("  ❌ Failed to fetch macro data")
        sys.exit(1)


if __name__ == '__main__':
    main()

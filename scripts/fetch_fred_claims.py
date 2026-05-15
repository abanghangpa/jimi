#!/usr/bin/env python3
"""
Fetch live macro data from FRED:
  - ICSA: Weekly jobless claims
  - UNRATE: Unemployment rate
  - WPSFD49207: PPI (Producer Price Index)
  - CPIAUCSL: CPI (Consumer Price Index)
  - FEDFUNDS: Federal funds rate

Writes data/fred/macro_cache.json consumed by:
  - m22_inflation_regime_v2.py (PPI, CPI, Fed funds, Fed stance)
  - m23_ppi_session.py (claims, unemployment)

Usage:
    python scripts/fetch_fred_claims.py
    python scripts/fetch_fred_claims.py --api-key YOUR_KEY

FRED API is free — get a key at https://fred.stlouisfed.org/docs/api/api_key.html
Or set FRED_API_KEY env var. Without a key, falls back to web scrape.
"""

import argparse
import json
import os
import sys
from datetime import datetime, timedelta

import requests

DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "fred")
CACHE_FILE = os.path.join(DATA_DIR, "claims_cache.json")
MACRO_CACHE_FILE = os.path.join(DATA_DIR, "macro_cache.json")
FRED_BASE = "https://api.stlouisfed.org/fred"


def fetch_fred_observations(series_id, api_key, start_date=None):
    """Fetch observations from FRED API (JSON)."""
    if start_date is None:
        start_date = (datetime.utcnow() - timedelta(days=365 * 6)).strftime("%Y-%m-%d")

    url = f"{FRED_BASE}/series/observations"
    params = {
        "series_id": series_id,
        "api_key": api_key,
        "file_type": "json",
        "observation_start": start_date,
        "sort_order": "asc",
    }
    resp = requests.get(url, params=params, timeout=30)
    resp.raise_for_status()
    data = resp.json()

    observations = data.get("observations", [])
    result = []
    for obs in observations:
        val = obs.get("value", ".")
        if val == "." or val == "":
            continue
        result.append({
            "date": obs["date"],
            "value": float(val),
        })
    return result


def weekly_to_monthly_avg(observations):
    """Convert weekly observations to monthly averages.

    FRED ICSA is weekly (ending Saturday). We group by year-month
    and average all observations in that month.
    """
    monthly = {}
    for obs in observations:
        d = datetime.strptime(obs["date"], "%Y-%m-%d")
        key = d.strftime("%Y-%m")
        monthly.setdefault(key, []).append(obs["value"])

    result = {}
    for key, values in sorted(monthly.items()):
        # FRED ICSA reports raw counts (e.g. 205000 = 205K claims).
        # The hardcoded dict in m23 uses thousands, so divide by 1000.
        avg = sum(values) / len(values)
        result[key] = round(avg / 1000) if avg > 1000 else round(avg)
    return result


def monthly_observations(observations):
    """Convert FRED monthly observations to {YYYY-MM: value} dict."""
    result = {}
    for obs in observations:
        d = datetime.strptime(obs["date"], "%Y-%m-%d")
        key = d.strftime("%Y-%m")
        result[key] = round(obs["value"], 1)
    return result


def compute_yoy(index_series):
    """Compute year-over-year % change from a monthly index series.

    Args:
        index_series: {YYYY-MM: index_value} dict

    Returns:
        {YYYY-MM: yoy_pct} dict
    """
    yoy = {}
    for month, val in sorted(index_series.items()):
        year = int(month[:4])
        mo = int(month[5:7])
        prev_year = f"{year - 1}-{mo:02d}"
        if prev_year in index_series and index_series[prev_year] > 0:
            pct = (val - index_series[prev_year]) / index_series[prev_year] * 100
            yoy[month] = round(pct, 1)
    return yoy


def infer_fed_stance(rate_series):
    """Infer Fed stance from rate changes.

    Looks at the most recent 3 months of data to determine if the Fed
    is actively cutting, holding, or hiking. Historical cuts that ended
    are classified as HOLDING (rates are now stable).

    Args:
        rate_series: {YYYY-MM: rate_pct} dict (e.g., FEDFUNDS)

    Returns:
        dict with 'stance' (CUTTING/HOLDING/HIKING), 'current_rate',
        'prev_rate', 'change', 'last_change_month'
    """
    if not rate_series or len(rate_series) < 2:
        return {'stance': 'HOLDING', 'current_rate': None, 'prev_rate': None,
                'change': 0, 'last_change_month': None}

    months = sorted(rate_series.keys())
    current_rate = rate_series[months[-1]]
    prev_rate = rate_series[months[-2]]

    # Find the last month with a meaningful rate change (>= 5bp)
    last_change_month = None
    last_change_bars_ago = 0
    for i in range(len(months) - 1, max(0, len(months) - 13), -1):
        diff = rate_series[months[i]] - rate_series[months[i - 1]]
        if abs(diff) >= 0.05:
            last_change_month = months[i]
            last_change_bars_ago = len(months) - 1 - i
            break

    recent_change = current_rate - prev_rate

    # If the last rate change was 2+ months ago, Fed is HOLDING
    # (even if rates were cut/hiked previously — that cycle ended)
    if last_change_bars_ago >= 2 or last_change_month is None:
        stance = 'HOLDING'
    elif abs(recent_change) < 0.05:
        stance = 'HOLDING'
    elif recent_change < -0.05:
        stance = 'CUTTING'
    elif recent_change > 0.05:
        stance = 'HIKING'
    else:
        stance = 'HOLDING'

    return {
        'stance': stance,
        'current_rate': round(current_rate, 2),
        'prev_rate': round(prev_rate, 2),
        'change': round(recent_change, 2),
        'last_change_month': last_change_month,
    }


def fetch_fred_csv(series_id, start_date="2023-01-01"):
    """Fetch a FRED series via CSV endpoint (no API key needed).

    Returns:
        list of {'date': 'YYYY-MM-DD', 'value': float}
    """
    url = (f"https://fred.stlouisfed.org/graph/fredgraph.csv"
           f"?id={series_id}&cosd={start_date}&coed=2026-12-31")
    resp = requests.get(url, timeout=30)
    resp.raise_for_status()
    lines = resp.text.strip().split("\n")
    observations = []
    for line in lines[1:]:
        parts = line.split(",")
        if len(parts) >= 2 and parts[1].strip() not in (".", ""):
            observations.append({"date": parts[0].strip(), "value": float(parts[1].strip())})
    return observations


def fetch_via_web_scrape():
    """Fallback: scrape FRED CSV endpoints (no API key needed)."""
    print("  No API key — trying FRED CSV endpoints...")

    result = {}

    # ICSA (weekly claims)
    try:
        icsa_obs = fetch_fred_csv("ICSA", "2020-01-01")
        result["icsa_raw"] = icsa_obs
        result["icsa"] = {"monthly_avg": weekly_to_monthly_avg(icsa_obs)}
        print(f"  ✅ ICSA: {len(icsa_obs)} weekly observations → {len(result['icsa']['monthly_avg'])} months")
    except Exception as e:
        print(f"  ⚠️  ICSA scrape failed: {e}")

    # UNRATE (unemployment rate)
    try:
        unrate_obs = fetch_fred_csv("UNRATE", "2020-01-01")
        result["unrate"] = {"monthly": monthly_observations(unrate_obs)}
        print(f"  ✅ UNRATE: {len(unrate_obs)} monthly observations")
    except Exception as e:
        print(f"  ⚠️  UNRATE scrape failed: {e}")

    # PPI (WPSFD49207 — PPI Commodity index, final demand)
    try:
        ppi_obs = fetch_fred_csv("WPSFD49207", "2023-01-01")
        ppi_monthly = monthly_observations(ppi_obs)
        ppi_yoy = compute_yoy(ppi_monthly)
        result["ppi"] = {"index": ppi_monthly, "yoy": ppi_yoy}
        print(f"  ✅ PPI: {len(ppi_obs)} monthly observations → {len(ppi_yoy)} YoY values")
    except Exception as e:
        print(f"  ⚠️  PPI scrape failed: {e}")

    # CPI (CPIAUCSL — All Urban Consumers)
    try:
        cpi_obs = fetch_fred_csv("CPIAUCSL", "2023-01-01")
        cpi_monthly = monthly_observations(cpi_obs)
        cpi_yoy = compute_yoy(cpi_monthly)
        result["cpi"] = {"index": cpi_monthly, "yoy": cpi_yoy}
        print(f"  ✅ CPI: {len(cpi_obs)} monthly observations → {len(cpi_yoy)} YoY values")
    except Exception as e:
        print(f"  ⚠️  CPI scrape failed: {e}")

    # Fed Funds Rate (FEDFUNDS)
    try:
        fed_obs = fetch_fred_csv("FEDFUNDS", "2023-01-01")
        fed_monthly = monthly_observations(fed_obs)
        fed_info = infer_fed_stance(fed_monthly)
        result["fed_funds"] = {"monthly": fed_monthly, "stance_info": fed_info}
        print(f"  ✅ FEDFUNDS: {len(fed_obs)} monthly observations → stance={fed_info['stance']} ({fed_info['current_rate']}%)")
    except Exception as e:
        print(f"  ⚠️  FEDFUNDS scrape failed: {e}")

    return result


def fetch_via_api(api_key):
    """Fetch via FRED API (needs key)."""
    result = {}

    print(f"  Fetching ICSA (weekly claims) from FRED API...")
    icsa_obs = fetch_fred_observations("ICSA", api_key)
    icsa_monthly = weekly_to_monthly_avg(icsa_obs)
    result["icsa_raw"] = icsa_obs
    result["icsa"] = {"monthly_avg": icsa_monthly}
    print(f"  ✅ ICSA: {len(icsa_obs)} weekly obs → {len(icsa_monthly)} months")

    print(f"  Fetching UNRATE (unemployment rate) from FRED API...")
    unrate_obs = fetch_fred_observations("UNRATE", api_key)
    unrate_monthly = monthly_observations(unrate_obs)
    result["unrate"] = {"monthly": unrate_monthly}
    print(f"  ✅ UNRATE: {len(unrate_obs)} monthly obs")

    print(f"  Fetching PPI (WPSFD49207) from FRED API...")
    ppi_obs = fetch_fred_observations("WPSFD49207", api_key, "2023-01-01")
    ppi_monthly = monthly_observations(ppi_obs)
    ppi_yoy = compute_yoy(ppi_monthly)
    result["ppi"] = {"index": ppi_monthly, "yoy": ppi_yoy}
    print(f"  ✅ PPI: {len(ppi_obs)} monthly obs → {len(ppi_yoy)} YoY")

    print(f"  Fetching CPI (CPIAUCSL) from FRED API...")
    cpi_obs = fetch_fred_observations("CPIAUCSL", api_key, "2023-01-01")
    cpi_monthly = monthly_observations(cpi_obs)
    cpi_yoy = compute_yoy(cpi_monthly)
    result["cpi"] = {"index": cpi_monthly, "yoy": cpi_yoy}
    print(f"  ✅ CPI: {len(cpi_obs)} monthly obs → {len(cpi_yoy)} YoY")

    print(f"  Fetching FEDFUNDS from FRED API...")
    fed_obs = fetch_fred_observations("FEDFUNDS", api_key, "2023-01-01")
    fed_monthly = monthly_observations(fed_obs)
    fed_info = infer_fed_stance(fed_monthly)
    result["fed_funds"] = {"monthly": fed_monthly, "stance_info": fed_info}
    print(f"  ✅ FEDFUNDS: {len(fed_obs)} monthly obs → stance={fed_info['stance']} ({fed_info['current_rate']}%)")

    return result


def main():
    parser = argparse.ArgumentParser(description="Fetch FRED macro data (claims, PPI, CPI, Fed funds)")
    parser.add_argument("--api-key", default=os.environ.get("FRED_API_KEY"),
                        help="FRED API key (or set FRED_API_KEY env var)")
    parser.add_argument("--output", default=CACHE_FILE,
                        help=f"Claims output path (default: {CACHE_FILE})")
    args = parser.parse_args()

    os.makedirs(os.path.dirname(args.output), exist_ok=True)

    print("Fetching FRED data (ICSA + UNRATE + PPI + CPI + FEDFUNDS)...")

    if args.api_key:
        data = fetch_via_api(args.api_key)
    else:
        data = fetch_via_web_scrape()

    if not data.get("icsa") and not data.get("unrate"):
        print("  ❌ No data fetched — cache not written")
        sys.exit(1)

    # Add metadata
    data["_meta"] = {
        "fetched_at": datetime.utcnow().isoformat() + "Z",
        "source": "FRED (Federal Reserve Economic Data)",
        "series": ["ICSA", "UNRATE", "WPSFD49207", "CPIAUCSL", "FEDFUNDS"],
    }

    # Write claims cache (backward compatible — m23 reads this)
    with open(args.output, "w") as f:
        json.dump(data, f, indent=2)
    print(f"\n  💾 Saved: {args.output}")

    # Write macro cache (M22 reads this)
    with open(MACRO_CACHE_FILE, "w") as f:
        json.dump(data, f, indent=2)
    print(f"  💾 Saved: {MACRO_CACHE_FILE}")

    # Print latest values
    icsa = data.get("icsa", {}).get("monthly_avg", {})
    unrate = data.get("unrate", {}).get("monthly", {})
    ppi_yoy = data.get("ppi", {}).get("yoy", {})
    cpi_yoy = data.get("cpi", {}).get("yoy", {})
    fed = data.get("fed_funds", {}).get("stance_info", {})

    if icsa:
        latest_month = max(icsa.keys())
        print(f"  Latest claims: {icsa[latest_month]}K ({latest_month})")
    if unrate:
        latest_month = max(unrate.keys())
        print(f"  Latest unemployment: {unrate[latest_month]}% ({latest_month})")
    if ppi_yoy:
        latest_month = max(ppi_yoy.keys())
        print(f"  Latest PPI YoY: {ppi_yoy[latest_month]:+.1f}% ({latest_month})")
    if cpi_yoy:
        latest_month = max(cpi_yoy.keys())
        print(f"  Latest CPI YoY: {cpi_yoy[latest_month]:+.1f}% ({latest_month})")
    if fed:
        print(f"  Fed funds: {fed['current_rate']}%  stance={fed['stance']}")


if __name__ == "__main__":
    main()

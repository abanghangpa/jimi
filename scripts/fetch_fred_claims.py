#!/usr/bin/env python3
"""
Fetch live jobless claims (ICSA) and unemployment rate (UNRATE) from FRED.
Writes data/fred/claims_cache.json consumed by m23_ppi_session.py.

Usage:
    python scripts/fetch_fred_claims.py
    python scripts/freq_fred_claims.py --api-key YOUR_KEY

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


def fetch_via_web_scrape():
    """Fallback: scrape FRED CSV endpoints (no API key needed)."""
    print("  No API key — trying FRED CSV endpoints...")

    # ICSA (weekly claims)
    url_icsa = "https://fred.stlouisfed.org/graph/fredgraph.csv?bgcolor=%23e1e9f0&chart_type=line&drp=0&fo=open%20sans&graph_bgcolor=%23ffffff&height=450&mode=fred&recession_bars=on&txtcolor=%23444444&ts=12&tts=12&width=1168&nt=0&thu=0&trc=0&show_legend=yes&show_axis_titles=yes&show_tooltip=yes&id=ICSA&scale=left&cosd=2020-01-01&coed=2026-12-31&line_color=%234572a7&link_values=false&line_style=solid&mark_type=none&mw=3&lw=2&ost=-99999&oet=99999&mma=0&fml=a&fq=Weekly%2C+Ending+Saturday&fam=avg&fgst=lin&fgsnd=2020-02-01&line_index=1&transformation=lin&vintage_date=2026-05-14&revision_date=2026-05-14&nd=1967-01-07"

    # UNRATE (unemployment rate)
    url_unrate = "https://fred.stlouisfed.org/graph/fredgraph.csv?id=UNRATE&cosd=2020-01-01&coed=2026-12-31"

    result = {}

    try:
        resp = requests.get(url_icsa, timeout=30)
        resp.raise_for_status()
        lines = resp.text.strip().split("\n")
        header = lines[0].split(",")
        weekly = []
        for line in lines[1:]:
            parts = line.split(",")
            if len(parts) >= 2 and parts[1].strip() not in (".", ""):
                weekly.append({"date": parts[0].strip(), "value": float(parts[1].strip())})
        result["icsa_raw"] = weekly
        result["icsa"] = {"monthly_avg": weekly_to_monthly_avg(weekly)}
        print(f"  ✅ ICSA: {len(weekly)} weekly observations → {len(result['icsa']['monthly_avg'])} months")
    except Exception as e:
        print(f"  ⚠️  ICSA scrape failed: {e}")

    try:
        resp = requests.get(url_unrate, timeout=30)
        resp.raise_for_status()
        lines = resp.text.strip().split("\n")
        monthly = []
        for line in lines[1:]:
            parts = line.split(",")
            if len(parts) >= 2 and parts[1].strip() not in (".", ""):
                monthly.append({"date": parts[0].strip(), "value": float(parts[1].strip())})
        result["unrate"] = {"monthly": monthly_observations(monthly)}
        print(f"  ✅ UNRATE: {len(monthly)} monthly observations")
    except Exception as e:
        print(f"  ⚠️  UNRATE scrape failed: {e}")

    return result


def fetch_via_api(api_key):
    """Fetch via FRED API (needs key)."""
    print(f"  Fetching ICSA (weekly claims) from FRED API...")
    icsa_obs = fetch_fred_observations("ICSA", api_key)
    icsa_monthly = weekly_to_monthly_avg(icsa_obs)
    print(f"  ✅ ICSA: {len(icsa_obs)} weekly obs → {len(icsa_monthly)} months")

    print(f"  Fetching UNRATE (unemployment rate) from FRED API...")
    unrate_obs = fetch_fred_observations("UNRATE", api_key)
    unrate_monthly = monthly_observations(unrate_obs)
    print(f"  ✅ UNRATE: {len(unrate_obs)} monthly obs")

    return {
        "icsa_raw": icsa_obs,
        "icsa": {"monthly_avg": icsa_monthly},
        "unrate": {"monthly": unrate_monthly},
    }


def main():
    parser = argparse.ArgumentParser(description="Fetch FRED jobless claims + unemployment data")
    parser.add_argument("--api-key", default=os.environ.get("FRED_API_KEY"),
                        help="FRED API key (or set FRED_API_KEY env var)")
    parser.add_argument("--output", default=CACHE_FILE,
                        help=f"Output path (default: {CACHE_FILE})")
    args = parser.parse_args()

    os.makedirs(os.path.dirname(args.output), exist_ok=True)

    print("Fetching FRED data (ICSA + UNRATE)...")

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
        "series": ["ICSA", "UNRATE"],
    }

    with open(args.output, "w") as f:
        json.dump(data, f, indent=2)

    print(f"\n  💾 Saved: {args.output}")

    # Print latest values
    icsa = data.get("icsa", {}).get("monthly_avg", {})
    unrate = data.get("unrate", {}).get("monthly", {})
    if icsa:
        latest_month = max(icsa.keys())
        print(f"  Latest claims: {icsa[latest_month]}K ({latest_month})")
    if unrate:
        latest_month = max(unrate.keys())
        print(f"  Latest unemployment: {unrate[latest_month]}% ({latest_month})")


if __name__ == "__main__":
    main()

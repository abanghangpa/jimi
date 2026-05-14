#!/usr/bin/env python3
"""
Fetch jobless claims + unemployment data from FRED API.

Usage:
    python3 scripts/fetch_fred_claims.py                  # fetch & cache
    python3 scripts/fetch_fred_claims.py --api-key KEY    # pass key directly
    python3 scripts/fetch_fred_claims.py --update         # fetch & patch m23 module

Env: set FRED_API_KEY or pass --api-key.
Get a free key at: https://fred.stlouisfed.org/docs/api/api_key.html

Series:
    ICSA  — Initial Jobless Claims (weekly, seasonally adjusted)
    CCSA  — Continued Claims (weekly)
    UNRATE — Unemployment Rate (monthly)
"""

import argparse
import json
import os
import sys
from datetime import datetime, timedelta

import requests

FRED_BASE = "https://api.stlouisfed.org/fred/series/observations"

SERIES = {
    "ICSA": {
        "name": "Initial Jobless Claims",
        "unit": "thousands",
        "frequency": "weekly",
    },
    "UNRATE": {
        "name": "Unemployment Rate",
        "unit": "percent",
        "frequency": "monthly",
    },
}

CACHE_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "fred")
CACHE_FILE = os.path.join(CACHE_DIR, "claims_cache.json")


def fetch_series(series_id: str, api_key: str, start: str = "2021-01-01") -> list[dict]:
    """Fetch observations from FRED for a given series."""
    params = {
        "series_id": series_id,
        "api_key": api_key,
        "file_type": "json",
        "observation_start": start,
        "sort_order": "asc",
        "frequency": "w" if series_id == "ICSA" else "m",  # weekly for claims, monthly for unrate
        "aggregation_method": "avg" if series_id == "ICSA" else "avg",
    }
    resp = requests.get(FRED_BASE, params=params, timeout=30)
    resp.raise_for_status()
    data = resp.json()

    observations = []
    for obs in data.get("observations", []):
        val = obs.get("value", ".")
        if val == "." or val == "":
            continue
        observations.append({
            "date": obs["date"],
            "value": float(val),
        })
    return observations


def compute_monthly_avg(observations: list[dict]) -> dict[str, float]:
    """Convert weekly claims to monthly averages (thousands)."""
    monthly = {}
    for obs in observations:
        month_key = obs["date"][:7]  # YYYY-MM
        if month_key not in monthly:
            monthly[month_key] = []
        monthly[month_key].append(obs["value"])

    result = {}
    for month, values in sorted(monthly.items()):
        result[month] = round(sum(values) / len(values), 1)
    return result


def fetch_all(api_key: str, start: str = "2021-01-01") -> dict:
    """Fetch all series and return structured data."""
    print(f"  📡 Fetching from FRED (start={start})...")

    # Initial Claims (weekly → monthly avg)
    print(f"    Fetching ICSA (Initial Claims)...")
    icsa_raw = fetch_series("ICSA", api_key, start)
    icsa_monthly = compute_monthly_avg(icsa_raw)
    latest_icsa = icsa_raw[-1] if icsa_raw else None
    print(f"    ✅ ICSA: {len(icsa_raw)} weekly obs, {len(icsa_monthly)} months, latest={latest_icsa['value'] if latest_icsa else 'N/A'}K ({latest_icsa['date'] if latest_icsa else ''})")

    # Unemployment Rate (monthly)
    print(f"    Fetching UNRATE (Unemployment Rate)...")
    unrate_raw = fetch_series("UNRATE", api_key, start)
    unrate_monthly = {}
    for obs in unrate_raw:
        unrate_monthly[obs["date"][:7]] = round(obs["value"], 1)
    latest_unrate = unrate_raw[-1] if unrate_raw else None
    print(f"    ✅ UNRATE: {len(unrate_raw)} monthly obs, latest={latest_unrate['value'] if latest_unrate else 'N/A'}% ({latest_unrate['date'] if latest_unrate else ''})")

    result = {
        "fetched_at": datetime.utcnow().isoformat() + "Z",
        "icsa": {
            "monthly_avg": icsa_monthly,
            "latest_weekly": {
                "date": latest_icsa["date"],
                "value": latest_icsa["value"],
            } if latest_icsa else None,
            "total_weekly_obs": len(icsa_raw),
        },
        "unrate": {
            "monthly": unrate_monthly,
            "latest": {
                "date": latest_unrate["date"],
                "value": latest_unrate["value"],
            } if latest_unrate else None,
        },
    }
    return result


def save_cache(data: dict):
    """Save fetched data to cache file."""
    os.makedirs(CACHE_DIR, exist_ok=True)
    with open(CACHE_FILE, "w") as f:
        json.dump(data, f, indent=2)
    print(f"  💾 Cached: {CACHE_FILE}")


def load_cache() -> dict | None:
    """Load cached data if available."""
    if not os.path.exists(CACHE_FILE):
        return None
    with open(CACHE_FILE) as f:
        return json.load(f)


def update_m23_module(data: dict):
    """Patch m23_ppi_session.py with fresh FRED data."""
    m23_path = os.path.join(os.path.dirname(os.path.dirname(__file__)),
                            "src", "modules", "m23_ppi_session.py")
    if not os.path.exists(m23_path):
        print(f"  ❌ m23 module not found at {m23_path}")
        return False

    with open(m23_path) as f:
        content = f.read()

    # Build new JOBLESS_CLAIMS_MONTHLY_AVG block
    icsa = data["icsa"]["monthly_avg"]
    claims_lines = []
    current_year = None
    for month_key in sorted(icsa.keys()):
        year = month_key[:4]
        if year != current_year:
            if current_year is not None:
                claims_lines.append("")
            claims_lines.append(f"    # {year}")
            current_year = year
        val = int(icsa[month_key]) if icsa[month_key] == int(icsa[month_key]) else icsa[month_key]
        claims_lines.append(f"    '{month_key}': {val},")
    claims_block = "JOBLESS_CLAIMS_MONTHLY_AVG = {\n" + "\n".join(claims_lines) + "\n}"

    # Build new UNEMPLOYMENT_RATE_MONTHLY block
    unrate = data["unrate"]["monthly"]
    unemp_lines = []
    current_year = None
    for month_key in sorted(unrate.keys()):
        year = month_key[:4]
        if year != current_year:
            if current_year is not None:
                unemp_lines.append("")
            unemp_lines.append(f"    # {year}")
            current_year = year
        unemp_lines.append(f"    '{month_key}': {unrate[month_key]},")
    unemp_block = "UNEMPLOYMENT_RATE_MONTHLY = {\n" + "\n".join(unemp_lines) + "\n}"

    import re

    # Replace JOBLESS_CLAIMS_MONTHLY_AVG
    pattern_claims = r"JOBLESS_CLAIMS_MONTHLY_AVG = \{[^}]*\n\}"
    if re.search(pattern_claims, content, re.DOTALL):
        content = re.sub(pattern_claims, claims_block, content, flags=re.DOTALL)
        print(f"  ✅ Patched JOBLESS_CLAIMS_MONTHLY_AVG ({len(icsa)} months)")
    else:
        print(f"  ⚠️  Could not find JOBLESS_CLAIMS_MONTHLY_AVG block to replace")
        return False

    # Replace UNEMPLOYMENT_RATE_MONTHLY
    pattern_unemp = r"UNEMPLOYMENT_RATE_MONTHLY = \{[^}]*\n\}"
    if re.search(pattern_unemp, content, re.DOTALL):
        content = re.sub(pattern_unemp, unemp_block, content, flags=re.DOTALL)
        print(f"  ✅ Patched UNEMPLOYMENT_RATE_MONTHLY ({len(unrate)} months)")
    else:
        print(f"  ⚠️  Could not find UNEMPLOYMENT_RATE_MONTHLY block to replace")
        return False

    # Update comment header with fresh data summary
    latest_claims = data["icsa"]["latest_weekly"]
    latest_unemp = data["unrate"]["latest"]
    header_comment = (
        f"# Historical monthly averages (from FRED API — auto-updated {data['fetched_at'][:10]}):\n"
        f"#   Latest claims: {latest_claims['value']:.0f}K ({latest_claims['date']})\n"
        f"#   Latest unemployment: {latest_unemp['value']}% ({latest_unemp['date']})"
    )
    pattern_header = r"# Historical monthly averages \(from (DOL|FRED).*?\n#.*?\n"
    if re.search(pattern_header, content, re.DOTALL):
        content = re.sub(pattern_header, header_comment + "\n", content, flags=re.DOTALL)

    with open(m23_path, "w") as f:
        f.write(content)
    print(f"  ✅ Updated {m23_path}")
    return True


def print_summary(data: dict):
    """Print a human-readable summary."""
    print(f"\n{'═' * 50}")
    print(f"  FRED DATA SUMMARY")
    print(f"{'═' * 50}")
    print(f"  Fetched: {data['fetched_at'][:19]}")

    icsa = data["icsa"]
    if icsa["latest_weekly"]:
        lw = icsa["latest_weekly"]
        print(f"\n  Initial Claims (ICSA):")
        print(f"    Latest weekly:  {lw['value']:.0f}K  ({lw['date']})")
        # Last 4 months
        monthly = icsa["monthly_avg"]
        recent = sorted(monthly.items())[-4:]
        print(f"    Recent monthly averages:")
        for m, v in recent:
            print(f"      {m}: {v:.0f}K")

    unrate = data["unrate"]
    if unrate["latest"]:
        lu = unrate["latest"]
        print(f"\n  Unemployment Rate (UNRATE):")
        print(f"    Latest:  {lu['value']}%  ({lu['date']})")
        recent_u = sorted(unrate["monthly"].items())[-4:]
        print(f"    Recent:")
        for m, v in recent_u:
            print(f"      {m}: {v}%")

    print(f"{'═' * 50}")


def main():
    parser = argparse.ArgumentParser(description="Fetch FRED claims + unemployment data")
    parser.add_argument("--api-key", help="FRED API key (or set FRED_API_KEY env)")
    parser.add_argument("--start", default="2021-01-01", help="Start date (default: 2021-01-01)")
    parser.add_argument("--update", action="store_true", help="Update m23 module with fresh data")
    parser.add_argument("--summary", action="store_true", help="Print summary from cache")
    args = parser.parse_args()

    api_key = args.api_key or os.environ.get("FRED_API_KEY")

    if args.summary:
        cache = load_cache()
        if cache:
            print_summary(cache)
        else:
            print("No cache found. Run without --summary first.")
        return

    if not api_key:
        print("❌ FRED API key required.")
        print("   Get a free key: https://fred.stlouisfed.org/docs/api/api_key.html")
        print("   Pass via --api-key or set FRED_API_KEY env var.")
        sys.exit(1)

    data = fetch_all(api_key, args.start)
    save_cache(data)
    print_summary(data)

    if args.update:
        print(f"\n  🔧 Updating m23 module...")
        update_m23_module(data)


if __name__ == "__main__":
    main()

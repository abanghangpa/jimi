"""
Macro Data Loader — reads FRED cache and provides live macro data to M22/M23.

Replaces hardcoded config values (M22_PPI_YOY, M22_CPI_YOY, M22_FED_FUNDS_RATE, etc.)
with live data fetched from FRED by scripts/fetch_fred_claims.py.

Usage:
    from src.utils.macro_loader import load_macro_data, get_m22_overrides

    data = load_macro_data()
    overrides = get_m22_overrides()
    # overrides = {'M22_PPI_YOY': 4.9, 'M22_CPI_YOY': 3.8, 'M22_FED_FUNDS_RATE': 3.64, ...}
"""

import json
import os
from datetime import datetime


_MACRO_CACHE = None
_CACHE_MTIME = 0

# Cache file paths (same as fetch_fred_claims.py)
_DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "data", "fred")
_MACRO_CACHE_FILE = os.path.join(_DATA_DIR, "macro_cache.json")
_CLAIMS_CACHE_FILE = os.path.join(_DATA_DIR, "claims_cache.json")


def _load_cache():
    """Load the FRED macro cache (with file-level caching)."""
    global _MACRO_CACHE, _CACHE_MTIME

    # Try macro_cache.json first, fall back to claims_cache.json
    cache_path = _MACRO_CACHE_FILE
    if not os.path.exists(cache_path):
        cache_path = _CLAIMS_CACHE_FILE
    if not os.path.exists(cache_path):
        return None

    try:
        mtime = os.path.getmtime(cache_path)
        if _MACRO_CACHE is not None and mtime == _CACHE_MTIME:
            return _MACRO_CACHE

        with open(cache_path) as f:
            _MACRO_CACHE = json.load(f)
        _CACHE_MTIME = mtime
        return _MACRO_CACHE
    except Exception:
        return None


def load_macro_data():
    """Load full macro data from FRED cache.

    Returns:
        dict with keys: ppi_yoy, ppi_prev_yoy, ppi_prev_prev_yoy,
        cpi_yoy, cpi_prev_yoy, fed_funds_rate, fed_stance, etc.
        Returns None if cache not available.
    """
    cache = _load_cache()
    if not cache:
        return None

    ppi_yoy_series = cache.get("ppi", {}).get("yoy", {})
    cpi_yoy_series = cache.get("cpi", {}).get("yoy", {})
    fed_monthly = cache.get("fed_funds", {}).get("monthly", {})
    fed_stance_info = cache.get("fed_funds", {}).get("stance_info", {})

    if not ppi_yoy_series and not cpi_yoy_series:
        return None

    # Sort months descending to get latest → previous
    ppi_months = sorted(ppi_yoy_series.keys(), reverse=True)
    cpi_months = sorted(cpi_yoy_series.keys(), reverse=True)

    result = {
        "source": "fred_live",
        "fetched_at": cache.get("_meta", {}).get("fetched_at"),
    }

    # PPI: current, prev, prev-prev
    if len(ppi_months) >= 1:
        result["ppi_yoy"] = ppi_yoy_series[ppi_months[0]]
        result["ppi_month"] = ppi_months[0]
    if len(ppi_months) >= 2:
        result["ppi_prev_yoy"] = ppi_yoy_series[ppi_months[1]]
    if len(ppi_months) >= 3:
        result["ppi_prev_prev_yoy"] = ppi_yoy_series[ppi_months[2]]

    # CPI: current, prev
    if len(cpi_months) >= 1:
        result["cpi_yoy"] = cpi_yoy_series[cpi_months[0]]
        result["cpi_month"] = cpi_months[0]
    if len(cpi_months) >= 2:
        result["cpi_prev_yoy"] = cpi_yoy_series[cpi_months[1]]

    # Fed funds rate + stance
    if fed_stance_info:
        result["fed_funds_rate"] = fed_stance_info.get("current_rate")
        result["fed_stance"] = fed_stance_info.get("stance", "HOLDING")
        result["fed_prev_rate"] = fed_stance_info.get("prev_rate")
        result["fed_change"] = fed_stance_info.get("change", 0)
    elif fed_monthly:
        fed_months = sorted(fed_monthly.keys(), reverse=True)
        if fed_months:
            result["fed_funds_rate"] = fed_monthly[fed_months[0]]

    # Claims (ICSA) — monthly average in thousands
    icsa_monthly = cache.get("icsa", {}).get("monthly_avg", {})
    if icsa_monthly:
        result["icsa_monthly_avg"] = icsa_monthly
        months = sorted(icsa_monthly.keys(), reverse=True)
        if months:
            result["claims_latest_k"] = icsa_monthly[months[0]]
            result["claims_latest_month"] = months[0]

    # Unemployment rate (UNRATE)
    unrate_monthly = cache.get("unrate", {}).get("monthly", {})
    if unrate_monthly:
        result["unrate_monthly"] = unrate_monthly
        months = sorted(unrate_monthly.keys(), reverse=True)
        if months:
            result["unemployment_rate"] = unrate_monthly[months[0]]
            result["unemployment_month"] = months[0]

    return result


def get_m22_overrides(config=None):
    """Get M22 config overrides from live FRED data.

    Returns dict of config keys to override. Only includes keys where
    live FRED data is available. Config values take precedence only
    when they differ from defaults (i.e., user explicitly set them).

    Usage in scanner.py or M22:
        overrides = get_m22_overrides()
        cfg.update(overrides)
    """
    data = load_macro_data()
    if not data or data.get("source") != "fred_live":
        return {}

    overrides = {}

    # PPI YoY (current + history)
    if "ppi_yoy" in data:
        overrides["M22_PPI_YOY"] = data["ppi_yoy"]
    if "ppi_prev_yoy" in data:
        overrides["M22_PPI_PREV_YOY"] = data["ppi_prev_yoy"]
    if "ppi_prev_prev_yoy" in data:
        overrides["M22_PPI_PREV_PREV_YOY"] = data["ppi_prev_prev_yoy"]

    # CPI YoY (current + history)
    if "cpi_yoy" in data:
        overrides["M22_CPI_YOY"] = data["cpi_yoy"]
    if "cpi_prev_yoy" in data:
        overrides["M22_CPI_PREV_YOY"] = data["cpi_prev_yoy"]

    # Fed funds rate + stance
    if "fed_funds_rate" in data and data["fed_funds_rate"] is not None:
        overrides["M22_FED_FUNDS_RATE"] = data["fed_funds_rate"]
    if "fed_stance" in data:
        overrides["M22_FED_STANCE"] = data["fed_stance"]

    return overrides


def format_macro_status():
    """Format a human-readable status line for the macro data.

    Returns empty string if no data available.
    """
    data = load_macro_data()
    if not data or data.get("source") != "fred_live":
        return ""

    parts = []
    if "ppi_yoy" in data:
        parts.append(f"PPI={data['ppi_yoy']:+.1f}%")
    if "cpi_yoy" in data:
        parts.append(f"CPI={data['cpi_yoy']:+.1f}%")
    if "fed_funds_rate" in data:
        parts.append(f"Fed={data['fed_funds_rate']:.2f}%")
    if "fed_stance" in data:
        parts.append(f"stance={data['fed_stance']}")
    if "claims_latest_k" in data:
        parts.append(f"claims={data['claims_latest_k']}K")
    if "unemployment_rate" in data:
        parts.append(f"unemp={data['unemployment_rate']:.1f}%")
    if data.get("fetched_at"):
        parts.append(f"fetched={data['fetched_at'][:16]}")

    return "  📊 FRED live: " + "  ".join(parts) if parts else ""

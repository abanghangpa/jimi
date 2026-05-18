#!/usr/bin/env python3
"""
Refresh all macro data caches.

Standalone script to pre-fetch and cache all macro indicators + FRED data.
Run this on a cron (e.g. weekly) so that `scanner.py --cached` always has
fresh data without making any HTTP calls.

Usage:
    python scripts/refresh_macro.py              # fetch only stale data (respect TTL)
    python scripts/refresh_macro.py --force      # force-refresh everything
    python scripts/refresh_macro.py --status     # show cache age, don't fetch

Exit codes:
    0  all fetches succeeded
    1  some fetches failed (partial cache update)
"""

import argparse
import json
import os
import sys
import time
from datetime import datetime, timezone

UTC = timezone.utc

# Project root
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)


def _fred_cache_age_hours():
    """Return age of FRED macro cache in hours, or None if missing."""
    path = os.path.join(ROOT, 'data', 'fred', 'macro_cache.json')
    if not os.path.exists(path):
        return None
    return (time.time() - os.path.getmtime(path)) / 3600


def _macro_cache_age_hours():
    """Return age of macro indicators cache in hours, or None if missing."""
    path = os.path.join(ROOT, 'data', 'macro', 'macro_indicators.json')
    if not os.path.exists(path):
        return None
    return (time.time() - os.path.getmtime(path)) / 3600


def show_status():
    """Print cache status and exit."""
    fred_age = _fred_cache_age_hours()
    macro_age = _macro_cache_age_hours()

    print("Macro Cache Status")
    print("=" * 45)

    if fred_age is not None:
        fresh = "✅ fresh" if fred_age < 168 else "⚠️  stale"  # 7 days
        print(f"  FRED cache:     {fred_age:.1f}h old  {fresh}")
    else:
        print(f"  FRED cache:     ❌ not found")

    if macro_age is not None:
        fresh = "✅ fresh" if macro_age < 168 else "⚠️  stale"
        print(f"  Macro cache:    {macro_age:.1f}h old  {fresh}")
    else:
        print(f"  Macro cache:    ❌ not found")

    # Show individual macro entries
    macro_path = os.path.join(ROOT, 'data', 'macro', 'macro_indicators.json')
    if os.path.exists(macro_path):
        try:
            with open(macro_path) as f:
                data = json.load(f)
            print(f"\n  Cached indicators ({len(data)}):")
            for key, entry in sorted(data.items()):
                if isinstance(entry, dict):
                    ts = entry.get('timestamp', '?')
                    actual = entry.get('actual', '?')
                    surprise = entry.get('surprise', '')
                    print(f"    {key:28s} actual={actual}  surprise={surprise}  [{ts[:16]}]")
        except Exception:
            pass


def refresh_fred(force=False):
    """Fetch FRED data. Returns True on success."""
    ttl_days = 7
    try:
        import yaml
        cfg_path = os.path.join(ROOT, 'config', 'settings.yaml')
        if os.path.exists(cfg_path):
            with open(cfg_path) as f:
                cfg = yaml.safe_load(f) or {}
            ttl_days = cfg.get('MACRO_CACHE_TTL_DAYS', 7)
    except Exception:
        pass

    age = _fred_cache_age_hours()
    if not force and age is not None and age < (ttl_days * 24):
        print(f"  📦 FRED: cache fresh ({age:.1f}h old, TTL={ttl_days}d) — skipping")
        return True

    print(f"  🔄 FRED: fetching...")
    try:
        import subprocess
        script = os.path.join(ROOT, 'scripts', 'fetch_fred_claims.py')
        r = subprocess.run([sys.executable, script], capture_output=True, text=True, timeout=60)
        if r.returncode == 0:
            for line in r.stdout.strip().split('\n'):
                print(f"  {line.strip()}")
            return True
        else:
            print(f"  ❌ FRED fetch failed: {r.stderr.strip()[:200]}")
            return False
    except Exception as e:
        print(f"  ❌ FRED fetch error: {e}")
        return False


def refresh_macro_indicators(force=False):
    """Fetch all macro indicators. Returns dict of results."""
    print(f"  🔄 Macro indicators: fetching...")
    try:
        from src.utils.macro_fetch import get_latest_macro_indicators
        results = get_latest_macro_indicators(force_refresh=force)

        fetched = 0
        cached = 0
        failed = 0
        for name, data in results.items():
            if data and data.get('actual') is not None:
                surprise = data.get('surprise', '?')
                source = data.get('source', '?')
                print(f"    ✅ {name}: actual={data['actual']} surprise={surprise} [{source}]")
                fetched += 1
            elif data:
                cached += 1
            else:
                failed += 1

        print(f"  📊 Summary: {fetched} fetched, {cached} cached-only, {failed} no data")
        return results
    except Exception as e:
        print(f"  ❌ Macro fetch error: {e}")
        return {}


def main():
    parser = argparse.ArgumentParser(description='Refresh JIMI macro data caches')
    parser.add_argument('--force', action='store_true',
                        help='Force-refresh all data (ignore cache TTL)')
    parser.add_argument('--status', action='store_true',
                        help='Show cache status only, don\'t fetch')
    args = parser.parse_args()

    if args.status:
        show_status()
        return

    print(f"Macro Refresh — {datetime.now(UTC).strftime('%Y-%m-%d %H:%M UTC')}")
    print(f"  Mode: {'FORCE' if args.force else 'TTL-aware'}")
    print()

    fred_ok = refresh_fred(force=args.force)
    print()
    macro_results = refresh_macro_indicators(force=args.force)

    # Summary
    print()
    n_ok = sum(1 for v in macro_results.values() if v and v.get('actual') is not None)
    n_total = len(macro_results)
    print(f"{'═' * 45}")
    if fred_ok and n_ok > 0:
        print(f"  ✅ Refresh complete — FRED ok, {n_ok}/{n_total} indicators cached")
    elif fred_ok:
        print(f"  ⚠️  FRED ok, but no macro indicators fetched (API key needed?)")
    else:
        print(f"  ❌ Refresh had errors — check output above")
    print(f"{'═' * 45}")


if __name__ == '__main__':
    main()

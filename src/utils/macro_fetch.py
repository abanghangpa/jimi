"""
Macro Data Fetcher — fetch real-time macro indicators for the scanner.

Sources (in priority order):
  1. Trading Economics API (if TE_API_KEY env var set)
  2. Investing.com scrape (free, fragile)
  3. Price reaction proxy (Binance ETH move in first 15m after release)

Usage:
    from src.utils.macro_fetch import fetch_caixin_pmi, get_latest_macro_indicators
    pmi = fetch_caixin_pmi()
    print(pmi)  # {'actual': 50.7, 'previous': 51.2, 'forecast': 51.0, 'surprise': 'MISS'}
"""

import os
import json
import time
from datetime import datetime, timedelta, timezone

UTC = timezone.utc

# Cache file for macro data
_CACHE_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), 'data', 'macro')
_CACHE_FILE = os.path.join(_CACHE_DIR, 'macro_indicators.json')


def _load_cache():
    """Load cached macro data."""
    if os.path.exists(_CACHE_FILE):
        try:
            with open(_CACHE_FILE) as f:
                return json.load(f)
        except (json.JSONDecodeError, IOError):
            pass
    return {}


def _save_cache(data):
    """Save macro data to cache."""
    os.makedirs(_CACHE_DIR, exist_ok=True)
    with open(_CACHE_FILE, 'w') as f:
        json.dump(data, f, indent=2)


def _classify_surprise(actual, previous, sigma=0.3):
    """Classify PMI surprise."""
    diff = actual - previous
    if diff >= 2 * sigma:
        return 'STRONG_BEAT'
    elif diff > 0.5 * sigma:
        return 'BEAT'
    elif diff >= -0.5 * sigma:
        return 'INLINE'
    elif diff >= -2 * sigma:
        return 'MISS'
    else:
        return 'BIG_MISS'


# ══════════════════════════════════════════════════════════════
# SOURCE 1: Trading Economics API
# ══════════════════════════════════════════════════════════════

def _fetch_trading_economics(indicator_id='china/manufacturing-pmi'):
    """Fetch from Trading Economics API.

    Requires TE_API_KEY env var. Free tier: 500 req/month.
    Sign up: https://developer.tradingeconomics.com/
    """
    api_key = os.environ.get('TE_API_KEY')
    if not api_key:
        return None

    try:
        import requests
        url = f'https://api.tradingeconomics.com/forecast/country/china/indicator/{indicator_id}'
        headers = {'Authorization': api_key}
        resp = requests.get(url, headers=headers, timeout=10)
        resp.raise_for_status()
        data = resp.json()

        if data and len(data) > 0:
            item = data[0]
            return {
                'actual': item.get('Actual'),
                'previous': item.get('Previous'),
                'forecast': item.get('Forecast'),
                'source': 'trading_economics',
                'timestamp': datetime.now(UTC).isoformat(),
            }
    except Exception as e:
        print(f"  ⚠️  Trading Economics fetch failed: {e}")

    return None


def _fetch_trading_economics_calendar(indicator='Caixin Manufacturing PMI'):
    """Fetch from Trading Economics calendar endpoint."""
    api_key = os.environ.get('TE_API_KEY')
    if not api_key:
        return None

    try:
        import requests
        url = f'https://api.tradingeconomics.com/calendar?country=china&indicator={indicator}'
        headers = {'Authorization': api_key}
        resp = requests.get(url, headers=headers, timeout=10)
        resp.raise_for_status()
        data = resp.json()

        if data and len(data) > 0:
            # Find the most recent Caixin Mfg PMI
            for item in data:
                if 'caixin' in item.get('Category', '').lower() and 'manufacturing' in item.get('Category', '').lower():
                    return {
                        'actual': item.get('Actual'),
                        'previous': item.get('Previous'),
                        'forecast': item.get('Forecast'),
                        'source': 'trading_economics_calendar',
                        'timestamp': datetime.now(UTC).isoformat(),
                    }
    except Exception as e:
        print(f"  ⚠️  Trading Economics calendar fetch failed: {e}")

    return None


# ══════════════════════════════════════════════════════════════
# SOURCE 2: Investing.com scrape
# ══════════════════════════════════════════════════════════════

def _fetch_investing_com():
    """Scrape Caixin PMI from Investing.com economic calendar.

    Free but fragile — may break if page structure changes.
    """
    try:
        import requests
        from datetime import datetime

        # Investing.com economic calendar API (undocumented but stable)
        url = 'https://sslecal2.investing.com/events/eventsList'
        params = {
            'country%5B%5D': '37',  # China
            'importance%5B%5D': '3',  # High impact
            'timeZone': '28',  # GMT
            'timeFilter': 'timeOnly',
            'currentTab': 'thisWeek',
            'limit_from': '0',
            'limit_to': '30',
        }
        headers = {
            'User-Agent': 'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36',
            'X-Requested-With': 'XMLHttpRequest',
        }

        resp = requests.get(url, params=params, headers=headers, timeout=10)
        if resp.status_code != 200:
            return None

        # Parse for Caixin Manufacturing PMI
        text = resp.text
        if 'caixin' in text.lower() and 'manufacturing' in text.lower():
            # Try to extract actual/previous/forecast
            # This is fragile — may need adjustment
            import re
            # Look for PMI values in the HTML
            pmi_match = re.search(r'Caixin.*?Manufacturing.*?PMI.*?(\d+\.\d+).*?(\d+\.\d+).*?(\d+\.\d+)',
                                  text, re.IGNORECASE | re.DOTALL)
            if pmi_match:
                return {
                    'actual': float(pmi_match.group(1)),
                    'previous': float(pmi_match.group(2)),
                    'forecast': float(pmi_match.group(3)),
                    'source': 'investing_com',
                    'timestamp': datetime.now(UTC).isoformat(),
                }
    except Exception as e:
        print(f"  ⚠️  Investing.com fetch failed: {e}")

    return None


# ══════════════════════════════════════════════════════════════
# SOURCE 3: Binance price reaction proxy
# ══════════════════════════════════════════════════════════════

def _fetch_price_reaction_proxy(symbol='ETHUSDT'):
    """Infer PMI surprise direction from ETH price reaction.

    Caixin PMI releases at 01:45 UTC on the 1st of each month.
    If ETH moves >0.5% in the first 15m → likely beat/miss.
    """
    try:
        import ccxt
        ex = ccxt.binance({'enableRateLimit': True})

        now = datetime.now(UTC)
        # Only check on the 1st of month, between 01:45 and 04:00 UTC
        if now.day != 1 or now.hour < 2 or now.hour > 5:
            return None

        # Get the 15m bar that covers 01:45-02:00 UTC
        release_time = now.replace(hour=1, minute=45, second=0, microsecond=0)
        since_ms = int(release_time.timestamp() * 1000)

        klines = ex.fetch_ohlcv(symbol, '15m', since=since_ms, limit=4)
        if not klines or len(klines) < 2:
            return None

        # Price at release (open of first bar)
        price_release = klines[0][1]
        # Price 30m later (close of second bar)
        price_after = klines[1][4]

        move_pct = (price_after - price_release) / price_release * 100

        return {
            'price_move_30m': round(move_pct, 4),
            'inferred_direction': 'BEAT' if move_pct > 0.5 else 'MISS' if move_pct < -0.5 else 'INLINE',
            'source': 'price_reaction_proxy',
            'timestamp': datetime.now(UTC).isoformat(),
        }
    except Exception as e:
        print(f"  ⚠️  Price reaction proxy failed: {e}")

    return None


# ══════════════════════════════════════════════════════════════
# SOURCE 4: Manual input file
# ══════════════════════════════════════════════════════════════

def _fetch_manual_input():
    """Read manually entered PMI data from file.

    File format: data/macro/manual_pmi.json
    {
        "2026-05": {"actual": 50.7, "previous": 51.2, "forecast": 51.0}
    }
    """
    manual_file = os.path.join(_CACHE_DIR, 'manual_pmi.json')
    if not os.path.exists(manual_file):
        return None

    try:
        with open(manual_file) as f:
            data = json.load(f)

        # Get current month key
        now = datetime.now(UTC)
        month_key = f"{now.year}-{now.month:02d}"

        if month_key in data:
            entry = data[month_key]
            return {
                'actual': entry.get('actual'),
                'previous': entry.get('previous'),
                'forecast': entry.get('forecast'),
                'source': 'manual_input',
                'timestamp': datetime.now(UTC).isoformat(),
            }
    except Exception:
        pass

    return None


# ══════════════════════════════════════════════════════════════
# MAIN API
# ══════════════════════════════════════════════════════════════

def fetch_caixin_pmi(force_refresh=False):
    """Fetch latest Caixin Manufacturing PMI data.

    Tries sources in priority order. Caches result for 24h.

    Returns:
        dict with keys: actual, previous, forecast, surprise, source, timestamp
        or None if all sources fail
    """
    cache = _load_cache()
    cache_key = 'caixin_mfg_pmi'

    # Check cache (valid for 24h unless force_refresh)
    if not force_refresh and cache_key in cache:
        cached = cache[cache_key]
        cached_time = datetime.fromisoformat(cached.get('timestamp', '2000-01-01T00:00:00+00:00'))
        if (datetime.now(UTC) - cached_time).total_seconds() < 86400:
            return cached

    print("  📡 Fetching Caixin Manufacturing PMI...")

    # Try sources in priority order
    sources = [
        ('manual_input', _fetch_manual_input),
        ('trading_economics', _fetch_trading_economics),
        ('trading_economics_calendar', _fetch_trading_economics_calendar),
        ('investing_com', _fetch_investing_com),
        ('price_reaction', _fetch_price_reaction_proxy),
    ]

    result = None
    for name, fetcher in sources:
        try:
            result = fetcher()
            if result and result.get('actual') is not None:
                print(f"  ✅ Caixin PMI from {name}: actual={result['actual']}")

                # ── Feed live Caixin PMI into M25 cache ──
                try:
                    from src.modules.m25_caixin_pmi import update_caixin_cache
                    _today = datetime.now(UTC).strftime('%Y-%m-%d')
                    update_caixin_cache(
                        actual=result['actual'],
                        previous=result.get('previous'),
                        release_date=_today,
                    )
                except Exception:
                    pass  # non-critical

                break
        except Exception as e:
            print(f"  ⚠️  {name} failed: {e}")
            continue

    if result is None:
        # Try to get from cache even if expired
        if cache_key in cache:
            print(f"  ⚠️  Using expired cache for Caixin PMI")
            return cache[cache_key]
        print(f"  ❌ All Caixin PMI sources failed")
        return None

    # Classify surprise
    if result.get('actual') is not None and result.get('previous') is not None:
        result['surprise'] = _classify_surprise(
            result['actual'], result['previous'])
    else:
        result['surprise'] = 'UNKNOWN'

    # Cache
    cache[cache_key] = result
    _save_cache(cache)

    return result


def fetch_nbs_pmi(force_refresh=False):
    """Fetch latest NBS Manufacturing PMI data.

    Also updates M24's NBS PMI cache so the session bias module
    has fresh data for release-day scoring.
    """
    cache = _load_cache()
    cache_key = 'nbs_mfg_pmi'

    if not force_refresh and cache_key in cache:
        cached = cache[cache_key]
        cached_time = datetime.fromisoformat(cached.get('timestamp', '2000-01-01T00:00:00+00:00'))
        if (datetime.now(UTC) - cached_time).total_seconds() < 86400:
            return cached

    print("  📡 Fetching NBS Manufacturing PMI...")

    # NBS PMI is available from Trading Economics
    result = _fetch_trading_economics('manufacturing-pmi')
    if result is None:
        result = _fetch_manual_input()

    if result and result.get('actual') is not None:
        result['surprise'] = _classify_surprise(
            result['actual'], result.get('previous', result['actual']))
        cache[cache_key] = result
        _save_cache(cache)
        print(f"  ✅ NBS PMI: actual={result['actual']}")

        # ── Feed live NBS PMI into M24 cache ──
        try:
            from src.modules.m24_nbs_pmi import update_nbs_cache
            _today = datetime.now(UTC).strftime('%Y-%m-%d')
            # Try to fetch NBS Services PMI too (released ~3 days later)
            svc_result = _fetch_trading_economics('services-pmi')
            svc_val = svc_result.get('actual') if svc_result else None
            update_nbs_cache(
                mfg_pmi=result['actual'],
                services_pmi=svc_val,
                release_date=_today,
            )
        except Exception:
            pass  # non-critical

        return result

    if cache_key in cache:
        return cache[cache_key]
    return None


def fetch_ism_pmi(force_refresh=False):
    """Fetch latest US ISM Manufacturing PMI data.

    Also updates M27's ISM PMI cache so the session bias module
    has fresh data for release-day scoring.
    """
    cache = _load_cache()
    cache_key = 'ism_mfg_pmi'

    if not force_refresh and cache_key in cache:
        cached = cache[cache_key]
        cached_time = datetime.fromisoformat(cached.get('timestamp', '2000-01-01T00:00:00+00:00'))
        if (datetime.now(UTC) - cached_time).total_seconds() < 86400:
            return cached

    print("  📡 Fetching US ISM Manufacturing PMI...")

    # Try Trading Economics for ISM Manufacturing PMI
    result = _fetch_trading_economics('ism-manufacturing-pmi')
    if result is None:
        result = _fetch_trading_economics('united-states/manufacturing-pmi')
    if result is None:
        result = _fetch_manual_input()

    if result and result.get('actual') is not None:
        result['surprise'] = _classify_surprise(
            result['actual'], result.get('previous', result['actual']))
        cache[cache_key] = result
        _save_cache(cache)
        print(f"  ✅ ISM MFG PMI: actual={result['actual']}")

        # ── Feed live ISM PMI into M27 cache ──
        try:
            from src.modules.m27_ism_pmi import update_ism_cache
            _today = datetime.now(UTC).strftime('%Y-%m-%d')
            # New Orders sub-index — primary signal driver
            # Try to fetch from Trading Economics
            no_result = _fetch_trading_economics('ism-new-orders')
            new_orders = no_result.get('actual') if no_result else None
            if new_orders is None:
                # Estimate from headline (rough proxy)
                new_orders = result['actual']
            update_ism_cache(
                actual=result['actual'],
                new_orders=new_orders,
                prior=result.get('previous'),
                release_date=_today,
            )
        except Exception:
            pass  # non-critical

        return result

    if cache_key in cache:
        return cache[cache_key]
    return None


def fetch_ism_svc_pmi(force_refresh=False):
    """Fetch latest US ISM Services PMI data.

    Also updates M28's ISM Services cache so the session bias module
    has fresh data for release-day scoring.
    """
    cache = _load_cache()
    cache_key = 'ism_svc_pmi'

    if not force_refresh and cache_key in cache:
        cached = cache[cache_key]
        cached_time = datetime.fromisoformat(cached.get('timestamp', '2000-01-01T00:00:00+00:00'))
        if (datetime.now(UTC) - cached_time).total_seconds() < 86400:
            return cached

    print("  📡 Fetching US ISM Services PMI...")

    result = _fetch_trading_economics('ism-non-manufacturing-pmi')
    if result is None:
        result = _fetch_trading_economics('united-states/services-pmi')
    if result is None:
        result = _fetch_manual_input()

    if result and result.get('actual') is not None:
        result['surprise'] = _classify_surprise(
            result['actual'], result.get('previous', result['actual']))
        cache[cache_key] = result
        _save_cache(cache)
        print(f"  ✅ ISM SVC PMI: actual={result['actual']}")

        # ── Feed live ISM Services PMI into M28 cache ──
        try:
            from src.modules.m28_ism_svc_pmi import update_ism_svc_cache
            _today = datetime.now(UTC).strftime('%Y-%m-%d')
            no_result = _fetch_trading_economics('ism-services-new-orders')
            new_orders = no_result.get('actual') if no_result else None
            if new_orders is None:
                new_orders = result['actual']
            update_ism_svc_cache(
                actual=result['actual'],
                new_orders=new_orders,
                prior=result.get('previous'),
                release_date=_today,
            )
        except Exception:
            pass

        return result

    if cache_key in cache:
        return cache[cache_key]
    return None


def fetch_jolts(force_refresh=False):
    """Fetch latest JOLTS Job Openings data.

    Also updates M29's JOLTS cache so the session bias module
    has fresh data for release-day scoring.
    """
    cache = _load_cache()
    cache_key = 'jolts'

    if not force_refresh and cache_key in cache:
        cached = cache[cache_key]
        cached_time = datetime.fromisoformat(cached.get('timestamp', '2000-01-01T00:00:00+00:00'))
        if (datetime.now(UTC) - cached_time).total_seconds() < 86400:
            return cached

    print("  📡 Fetching JOLTS Job Openings...")

    result = _fetch_trading_economics('job-offers')
    if result is None:
        result = _fetch_trading_economics('united-states/job-openings')
    if result is None:
        result = _fetch_manual_input()

    if result and result.get('actual') is not None:
        result['surprise'] = _classify_surprise(
            result['actual'], result.get('previous', result['actual']))
        cache[cache_key] = result
        _save_cache(cache)
        print(f"  ✅ JOLTS: actual={result['actual']}")

        # ── Feed live JOLTS into M29 cache ──
        try:
            from src.modules.m29_jolts import update_jolts_cache
            _today = datetime.now(UTC).strftime('%Y-%m-%d')
            update_jolts_cache(
                actual=result['actual'],
                prior=result.get('previous', result['actual']),
                quits_rate=result.get('quits_rate'),
                release_date=_today,
            )
        except Exception:
            pass

        return result

    if cache_key in cache:
        return cache[cache_key]
    return None


def fetch_china_cpi(force_refresh=False):
    """Fetch latest China CPI+PPI data (NBS release, ~10th-15th of month).

    Also updates M30's cache so the session bias module
    has fresh data for release-day scoring.
    """
    cache = _load_cache()
    cache_key = 'china_cpi_ppi'

    if not force_refresh and cache_key in cache:
        cached = cache[cache_key]
        cached_time = datetime.fromisoformat(cached.get('timestamp', '2000-01-01T00:00:00+00:00'))
        if (datetime.now(UTC) - cached_time).total_seconds() < 86400:
            return cached

    print("  📡 Fetching China CPI+PPI...")

    result = _fetch_trading_economics('consumer-price-index-cpi')
    if result is None:
        result = _fetch_trading_economics('china/consumer-price-index-cpi')
    if result is None:
        result = _fetch_manual_input()

    if result and result.get('actual') is not None:
        result['surprise'] = _classify_surprise(
            result['actual'], result.get('previous', result['actual']))
        cache[cache_key] = result
        _save_cache(cache)
        print(f"  ✅ China CPI: actual={result['actual']}")

        # ── Feed live data into M30 cache ──
        try:
            from src.modules.m30_china_cpi_ppi import update_china_cpi_cache
            _today = datetime.now(UTC).strftime('%Y-%m-%d')
            update_china_cpi_cache(
                cpi_yoy=result['actual'],
                ppi_yoy=result.get('ppi_yoy', result.get('previous', 0)),
                release_date=_today,
            )
        except Exception:
            pass

        return result

    if cache_key in cache:
        return cache[cache_key]
    return None


def fetch_uk_cpi(force_refresh=False):
    """Fetch latest UK CPI data (ONS release, ~10th-20th of month).

    Also updates M31's cache so the session bias module
    has fresh data for release-day scoring.
    """
    cache = _load_cache()
    cache_key = 'uk_cpi'

    if not force_refresh and cache_key in cache:
        cached = cache[cache_key]
        cached_time = datetime.fromisoformat(cached.get('timestamp', '2000-01-01T00:00:00+00:00'))
        if (datetime.now(UTC) - cached_time).total_seconds() < 86400:
            return cached

    print("  📡 Fetching UK CPI...")

    result = _fetch_trading_economics('united-kingdom/consumer-price-index-cpi')
    if result is None:
        result = _fetch_manual_input()

    if result and result.get('actual') is not None:
        result['surprise'] = _classify_surprise(
            result['actual'], result.get('previous', result['actual']))
        cache[cache_key] = result
        _save_cache(cache)
        print(f"  ✅ UK CPI: actual={result['actual']}")

        # ── Feed live data into M31 cache ──
        try:
            from src.modules.m31_uk_cpi import update_uk_cpi_cache
            _today = datetime.now(UTC).strftime('%Y-%m-%d')
            update_uk_cpi_cache(
                cpi_yoy=result['actual'],
                services_yoy=result.get('services_yoy', result.get('previous', 0)),
                release_date=_today,
            )
        except Exception:
            pass

        return result

    if cache_key in cache:
        return cache[cache_key]
    return None


def fetch_uk_wages(force_refresh=False):
    """Fetch latest UK Employment + Wages data (ONS release, ~2nd Tuesday).

    Also updates M32's cache so the session bias module
    has fresh data for release-day scoring.
    """
    cache = _load_cache()
    cache_key = 'uk_wages'

    if not force_refresh and cache_key in cache:
        cached = cache[cache_key]
        cached_time = datetime.fromisoformat(cached.get('timestamp', '2000-01-01T00:00:00+00:00'))
        if (datetime.now(UTC) - cached_time).total_seconds() < 86400:
            return cached

    print("  📡 Fetching UK Wages...")

    result = _fetch_trading_economics('united-kingdom/average-earnings')
    if result is None:
        result = _fetch_trading_economics('united-kingdom/wage-growth')
    if result is None:
        result = _fetch_manual_input()

    if result and result.get('actual') is not None:
        result['surprise'] = _classify_surprise(
            result['actual'], result.get('previous', result['actual']))
        cache[cache_key] = result
        _save_cache(cache)
        print(f"  ✅ UK Wages: actual={result['actual']}")

        # ── Feed live data into M32 cache ──
        try:
            from src.modules.m32_uk_wages import update_uk_wages_cache
            _today = datetime.now(UTC).strftime('%Y-%m-%d')
            update_uk_wages_cache(
                earnings_3m_yr=result['actual'],
                unemployment=result.get('unemployment'),
                release_date=_today,
            )
        except Exception:
            pass

        return result

    if cache_key in cache:
        return cache[cache_key]
    return None


def fetch_us_retail_sales(force_refresh=False):
    """Fetch latest US Retail Sales data.

    Also updates M33's retail sales cache so the session bias module
    has fresh data for release-day scoring.
    """
    cache = _load_cache()
    cache_key = 'us_retail_sales'

    if not force_refresh and cache_key in cache:
        cached = cache[cache_key]
        cached_time = datetime.fromisoformat(cached.get('timestamp', '2000-01-01T00:00:00+00:00'))
        if (datetime.now(UTC) - cached_time).total_seconds() < 86400:
            return cached

    print("  📡 Fetching US Retail Sales...")

    result = _fetch_trading_economics('retail-sales')
    if result is None:
        result = _fetch_manual_input()

    if result and result.get('actual') is not None:
        result['surprise'] = _classify_surprise(
            result['actual'], result.get('previous', result['actual']))
        cache[cache_key] = result
        _save_cache(cache)
        print(f"  ✅ US Retail Sales: actual={result['actual']}")

        # ── Feed live data into M33 cache ──
        try:
            from src.modules.m33_retail_sales import update_retail_sales_cache
            _today = datetime.now(UTC).strftime('%Y-%m-%d')
            update_retail_sales_cache(
                retail_mom=result['actual'],
                core_mom=None,  # core not always available from TE
                release_date=_today,
            )
        except Exception:
            pass

        return result

    if cache_key in cache:
        return cache[cache_key]
    return None


def fetch_us_housing_starts(force_refresh=False):
    """Fetch latest US Housing Starts data.

    Also updates M34's housing cache so the session bias module
    has fresh data for release-day scoring.
    """
    cache = _load_cache()
    cache_key = 'us_housing_starts'

    if not force_refresh and cache_key in cache:
        cached = cache[cache_key]
        cached_time = datetime.fromisoformat(cached.get('timestamp', '2000-01-01T00:00:00+00:00'))
        if (datetime.now(UTC) - cached_time).total_seconds() < 86400:
            return cached

    print("  📡 Fetching US Housing Starts...")

    result = _fetch_trading_economics('housing-starts')
    if result is None:
        result = _fetch_manual_input()

    if result and result.get('actual') is not None:
        result['surprise'] = _classify_surprise(
            result['actual'], result.get('previous', result['actual']))
        cache[cache_key] = result
        _save_cache(cache)
        print(f"  ✅ US Housing Starts: actual={result['actual']}")

        # ── Feed live data into M34 cache ──
        try:
            from src.modules.m34_housing_starts import update_housing_cache
            _today = datetime.now(UTC).strftime('%Y-%m-%d')
            update_housing_cache(
                starts_k=result['actual'],
                permits_k=None,
                starts_mom=None,
                permits_mom=None,
                release_date=_today,
            )
        except Exception:
            pass

        return result

    if cache_key in cache:
        return cache[cache_key]
    return None


def fetch_pboc_lpr(force_refresh=False):
    """Fetch latest PBoC LPR data and update M35 cache."""
    cache = _load_cache()
    cache_key = 'pboc_lpr'

    if not force_refresh and cache_key in cache:
        cached = cache[cache_key]
        cached_time = datetime.fromisoformat(cached.get('timestamp', '2000-01-01T00:00:00+00:00'))
        if (datetime.now(UTC) - cached_time).total_seconds() < 86400:
            return cached

    print("  📡 Fetching PBoC LPR...")
    result = _fetch_trading_economics('loan-prime-rate')
    if result is None:
        result = _fetch_manual_input()

    if result and result.get('actual') is not None:
        result['surprise'] = _classify_surprise(
            result['actual'], result.get('previous', result['actual']))
        cache[cache_key] = result
        _save_cache(cache)
        print(f"  ✅ PBoC LPR: actual={result['actual']}")

        try:
            from src.modules.m35_pboc_lpr import update_pboc_lpr_cache
            _today = datetime.now(UTC).strftime('%Y-%m-%d')
            update_pboc_lpr_cache(lpr_1y=result['actual'], lpr_5y=None, release_date=_today)
        except Exception:
            pass

        return result

    if cache_key in cache:
        return cache[cache_key]
    return None


def fetch_adp_employment(force_refresh=False):
    """Fetch latest ADP Employment data and update M36 cache."""
    cache = _load_cache()
    cache_key = 'adp_employment'

    if not force_refresh and cache_key in cache:
        cached = cache[cache_key]
        cached_time = datetime.fromisoformat(cached.get('timestamp', '2000-01-01T00:00:00+00:00'))
        if (datetime.now(UTC) - cached_time).total_seconds() < 86400:
            return cached

    print("  📡 Fetching ADP Employment...")
    result = _fetch_trading_economics('non-farm-payrolls-private')
    if result is None:
        result = _fetch_manual_input()

    if result and result.get('actual') is not None:
        result['surprise'] = _classify_surprise(
            result['actual'], result.get('previous', result['actual']))
        cache[cache_key] = result
        _save_cache(cache)
        print(f"  ✅ ADP Employment: actual={result['actual']}")

        try:
            from src.modules.m36_adp_employment import update_adp_cache
            _today = datetime.now(UTC).strftime('%Y-%m-%d')
            update_adp_cache(adp_k=result['actual'], consensus_k=None, release_date=_today)
        except Exception:
            pass

        return result

    if cache_key in cache:
        return cache[cache_key]
    return None


def fetch_nfp(force_refresh=False):
    """Fetch latest NFP (Non-Farm Payrolls) data and update M37 cache."""
    cache = _load_cache()
    cache_key = 'nfp'

    if not force_refresh and cache_key in cache:
        cached = cache[cache_key]
        cached_time = datetime.fromisoformat(cached.get('timestamp', '2000-01-01T00:00:00+00:00'))
        if (datetime.now(UTC) - cached_time).total_seconds() < 86400:
            return cached

    print("  📡 Fetching NFP (Non-Farm Payrolls)...")
    result = _fetch_trading_economics('non-farm-payrolls')
    if result is None:
        result = _fetch_manual_input()

    if result and result.get('actual') is not None:
        result['surprise'] = _classify_surprise(
            result['actual'], result.get('previous', result['actual']))
        cache[cache_key] = result
        _save_cache(cache)
        print(f"  ✅ NFP: actual={result['actual']}")

        try:
            from src.modules.m37_nfp import update_nfp_cache
            _today = datetime.now(UTC).strftime('%Y-%m-%d')
            update_nfp_cache(nfp_k=result['actual'], consensus_k=None, release_date=_today)
        except Exception:
            pass

        return result

    if cache_key in cache:
        return cache[cache_key]
    return None


def fetch_ifo(force_refresh=False):
    """Fetch latest Germany Ifo Business Climate data and update M38 cache."""
    cache = _load_cache()
    cache_key = 'ifo'

    if not force_refresh and cache_key in cache:
        cached = cache[cache_key]
        cached_time = datetime.fromisoformat(cached.get('timestamp', '2000-01-01T00:00:00+00:00'))
        if (datetime.now(UTC) - cached_time).total_seconds() < 86400:
            return cached

    print("  📡 Fetching Germany Ifo Business Climate...")
    result = _fetch_trading_economics('germany/business-confidence')
    if result is None:
        result = _fetch_manual_input()

    if result and result.get('actual') is not None:
        result['surprise'] = _classify_surprise(
            result['actual'], result.get('previous', result['actual']))
        cache[cache_key] = result
        _save_cache(cache)
        print(f"  ✅ Ifo: actual={result['actual']}")

        try:
            from src.modules.m38_ifo import update_ifo_cache
            _today = datetime.now(UTC).strftime('%Y-%m-%d')
            update_ifo_cache(actual=result['actual'], consensus_k=None, release_date=_today)
        except Exception:
            pass

        return result

    if cache_key in cache:
        return cache[cache_key]
    return None


def fetch_ums(force_refresh=False):
    """Fetch latest Michigan Consumer Sentiment data and update M39 cache."""
    cache = _load_cache()
    cache_key = 'ums'

    if not force_refresh and cache_key in cache:
        cached = cache[cache_key]
        cached_time = datetime.fromisoformat(cached.get('timestamp', '2000-01-01T00:00:00+00:00'))
        if (datetime.now(UTC) - cached_time).total_seconds() < 86400:
            return cached

    print("  📡 Fetching Michigan Consumer Sentiment...")
    result = _fetch_trading_economics('michigan/consumer-sentiment')
    if result is None:
        result = _fetch_manual_input()

    if result and result.get('actual') is not None:
        result['surprise'] = _classify_surprise(
            result['actual'], result.get('previous', result['actual']))
        cache[cache_key] = result
        _save_cache(cache)
        print(f"  ✅ UMS: actual={result['actual']}")

        try:
            from src.modules.m39_ums import update_ums_cache
            _today = datetime.now(UTC).strftime('%Y-%m-%d')
            update_ums_cache(headline=result['actual'], release_date=_today)
        except Exception:
            pass

        return result

    if cache_key in cache:
        return cache[cache_key]
    return None


def fetch_germany_cpi(force_refresh=False):
    """Fetch latest Germany CPI data and update M40 cache."""
    cache = _load_cache()
    cache_key = 'germany_cpi'

    if not force_refresh and cache_key in cache:
        cached = cache[cache_key]
        cached_time = datetime.fromisoformat(cached.get('timestamp', '2000-01-01T00:00:00+00:00'))
        if (datetime.now(UTC) - cached_time).total_seconds() < 86400:
            return cached

    print("  📡 Fetching Germany CPI...")
    result = _fetch_trading_economics('germany/consumer-price-index')
    if result is None:
        result = _fetch_manual_input()

    if result and result.get('actual') is not None:
        result['surprise'] = _classify_surprise(
            result['actual'], result.get('previous', result['actual']))
        cache[cache_key] = result
        _save_cache(cache)
        print(f"  ✅ Germany CPI: actual={result['actual']}")
        return result

    if cache_key in cache:
        return cache[cache_key]
    return None


def fetch_ez_cpi(force_refresh=False):
    """Fetch latest Eurozone CPI Flash data and update M41 cache."""
    cache = _load_cache()
    cache_key = 'ez_cpi'

    if not force_refresh and cache_key in cache:
        cached = cache[cache_key]
        cached_time = datetime.fromisoformat(cached.get('timestamp', '2000-01-01T00:00:00+00:00'))
        if (datetime.now(UTC) - cached_time).total_seconds() < 86400:
            return cached

    print("  📡 Fetching Eurozone CPI Flash...")
    result = _fetch_trading_economics('euro-area/consumer-price-index')
    if result is None:
        result = _fetch_manual_input()

    if result and result.get('actual') is not None:
        result['surprise'] = _classify_surprise(
            result['actual'], result.get('previous', result['actual']))
        cache[cache_key] = result
        _save_cache(cache)
        print(f"  ✅ EZ CPI Flash: actual={result['actual']}")
        return result

    if cache_key in cache:
        return cache[cache_key]
    return None


def fetch_ez_gdp(force_refresh=False):
    """Fetch latest Eurozone GDP Flash data and update M42 cache."""
    cache = _load_cache()
    cache_key = 'ez_gdp'

    if not force_refresh and cache_key in cache:
        cached = cache[cache_key]
        cached_time = datetime.fromisoformat(cached.get('timestamp', '2000-01-01T00:00:00+00:00'))
        if (datetime.now(UTC) - cached_time).total_seconds() < 86400:
            return cached

    print("  📡 Fetching Eurozone GDP Flash...")
    result = _fetch_trading_economics('euro-area/gdp-growth')
    if result is None:
        result = _fetch_manual_input()

    if result and result.get('actual') is not None:
        result['surprise'] = _classify_surprise(
            result['actual'], result.get('previous', result['actual']))
        cache[cache_key] = result
        _save_cache(cache)
        print(f"  ✅ EZ GDP Flash: actual={result['actual']}")
        return result

    if cache_key in cache:
        return cache[cache_key]
    return None


def fetch_us_gdp(force_refresh=False):
    """Fetch latest US GDP Advance Estimate data and update M43 cache."""
    cache = _load_cache()
    cache_key = 'us_gdp'

    if not force_refresh and cache_key in cache:
        cached = cache[cache_key]
        cached_time = datetime.fromisoformat(cached.get('timestamp', '2000-01-01T00:00:00+00:00'))
        if (datetime.now(UTC) - cached_time).total_seconds() < 86400:
            return cached

    print("  📡 Fetching US GDP Advance...")
    result = _fetch_trading_economics('united-states/gdp-growth')
    if result is None:
        result = _fetch_manual_input()

    if result and result.get('actual') is not None:
        result['surprise'] = _classify_surprise(
            result['actual'], result.get('previous', result['actual']))
        cache[cache_key] = result
        _save_cache(cache)
        print(f"  ✅ US GDP Advance: actual={result['actual']}")
        return result

    if cache_key in cache:
        return cache[cache_key]
    return None


def fetch_us_durables(force_refresh=False):
    """Fetch latest US Durable Goods Orders data and update M44 cache."""
    cache = _load_cache()
    cache_key = 'us_durables'

    if not force_refresh and cache_key in cache:
        cached = cache[cache_key]
        cached_time = datetime.fromisoformat(cached.get('timestamp', '2000-01-01T00:00:00+00:00'))
        if (datetime.now(UTC) - cached_time).total_seconds() < 86400:
            return cached

    print("  📡 Fetching US Durable Goods Orders...")
    result = _fetch_trading_economics('united-states/durable-goods-orders')
    if result is None:
        result = _fetch_manual_input()

    if result and result.get('actual') is not None:
        result['surprise'] = _classify_surprise(
            result['actual'], result.get('previous', result['actual']))
        cache[cache_key] = result
        _save_cache(cache)
        print(f"  ✅ US Durable Goods: actual={result['actual']}")
        return result

    if cache_key in cache:
        return cache[cache_key]
    return None


def fetch_us_pce(force_refresh=False):
    """Fetch latest US Core PCE data and update M45 cache."""
    cache = _load_cache()
    cache_key = 'us_pce'

    if not force_refresh and cache_key in cache:
        cached = cache[cache_key]
        cached_time = datetime.fromisoformat(cached.get('timestamp', '2000-01-01T00:00:00+00:00'))
        if (datetime.now(UTC) - cached_time).total_seconds() < 86400:
            return cached

    print("  📡 Fetching US Core PCE...")
    result = _fetch_trading_economics('united-states/pce-price-index-annual-change')
    if result is None:
        result = _fetch_manual_input()

    if result and result.get('actual') is not None:
        result['surprise'] = _classify_surprise(
            result['actual'], result.get('previous', result['actual']))
        cache[cache_key] = result
        _save_cache(cache)
        print(f"  ✅ US Core PCE: actual={result['actual']}")
        return result

    if cache_key in cache:
        return cache[cache_key]
    return None


def fetch_jp_cpi(force_refresh=False):
    """Fetch latest Japan CPI (Tokyo Flash) data and update M46 cache."""
    cache = _load_cache()
    cache_key = 'jp_cpi'

    if not force_refresh and cache_key in cache:
        cached = cache[cache_key]
        cached_time = datetime.fromisoformat(cached.get('timestamp', '2000-01-01T00:00:00+00:00'))
        if (datetime.now(UTC) - cached_time).total_seconds() < 86400:
            return cached

    print("  📡 Fetching Japan CPI (Tokyo Flash)...")
    result = _fetch_trading_economics('japan/inflation-rate')
    if result is None:
        result = _fetch_manual_input()

    if result and result.get('actual') is not None:
        result['surprise'] = _classify_surprise(
            result['actual'], result.get('previous', result['actual']))
        cache[cache_key] = result
        _save_cache(cache)
        print(f"  ✅ Japan CPI: actual={result['actual']}")
        return result

    if cache_key in cache:
        return cache[cache_key]
    return None


def get_latest_macro_indicators():
    """Fetch all relevant macro indicators for the scanner.

    Returns dict with keys for each indicator.
    """
    return {
        'caixin_mfg_pmi': fetch_caixin_pmi(),
        'nbs_mfg_pmi': fetch_nbs_pmi(),
        'ism_mfg_pmi': fetch_ism_pmi(),
        'ism_svc_pmi': fetch_ism_svc_pmi(),
        'jolts': fetch_jolts(),
        'china_cpi_ppi': fetch_china_cpi(),
        'uk_cpi': fetch_uk_cpi(),
        'uk_wages': fetch_uk_wages(),
        'us_retail_sales': fetch_us_retail_sales(),
        'us_housing_starts': fetch_us_housing_starts(),
        'pboc_lpr': fetch_pboc_lpr(),
        'adp_employment': fetch_adp_employment(),
        'nfp': fetch_nfp(),
        'ifo': fetch_ifo(),
        'ums': fetch_ums(),
        'germany_cpi': fetch_germany_cpi(),
        'ez_cpi': fetch_ez_cpi(),
        'ez_gdp': fetch_ez_gdp(),
        'us_gdp': fetch_us_gdp(),
        'us_durables': fetch_us_durables(),
        'us_pce': fetch_us_pce(),
        'jp_cpi': fetch_jp_cpi(),
    }


def get_surprise_for_event(event_id):
    """Get the surprise classification for a specific macro event.

    Used by the scanner to check if a recent release has data available.
    """
    indicators = get_latest_macro_indicators()

    event_map = {
        'cn_caixin_mfg_pmi': 'caixin_mfg_pmi',
        'cn_nbs_pmi': 'nbs_mfg_pmi',
        'us_ism_mfg_pmi': 'ism_mfg_pmi',
        'us_ism_svc_pmi': 'ism_svc_pmi',
        'us_jolts': 'jolts',
        'cn_cpi_ppi': 'china_cpi_ppi',
        'uk_cpi': 'uk_cpi',
        'uk_wages': 'uk_wages',
        'de_cpi': 'germany_cpi',
        'ez_cpi': 'ez_cpi',
        'ez_gdp': 'ez_gdp',
        'us_gdp': 'us_gdp',
        'us_durables': 'us_durables',
        'us_pce': 'us_pce',
        'jp_cpi': 'jp_cpi',
    }

    key = event_map.get(event_id)
    if key and key in indicators and indicators[key]:
        return indicators[key].get('surprise', 'UNKNOWN')

    return 'UNKNOWN'


if __name__ == '__main__':
    print("Macro Data Fetcher Test")
    print("=" * 40)
    indicators = get_latest_macro_indicators()
    for name, data in indicators.items():
        if data:
            print(f"\n{name}:")
            for k, v in data.items():
                print(f"  {k}: {v}")
        else:
            print(f"\n{name}: No data available")

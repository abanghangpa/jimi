"""
Macro Calendar Tracker — tracks upcoming economic data releases across
US, China, EU, Japan, UK, and South Korea.

Shows:
  - Next event countdown
  - Full monthly cascade chain
  - What to watch after each release
  - Expected ETH impact by historical pattern

Usage:
    from src.modules.macro_calendar import get_macro_calendar, format_macro_calendar

    cal = get_macro_calendar()
    print(format_macro_calendar(cal))
"""

import json
import os
from datetime import datetime, timedelta, timezone
from typing import Optional

# ── Time references ──
UTC = timezone.utc

# ── Release schedule definitions ──
# Each entry: (name, country, tier, schedule_func, time_utc, impact, cascade_notes)

def _first_friday(year, month):
    """First Friday of the month."""
    d = datetime(year, month, 1, tzinfo=UTC)
    while d.weekday() != 4:  # Friday
        d += timedelta(days=1)
    return d


def _nth_weekday(year, month, weekday, n):
    """Nth weekday of month (0=Mon, 4=Fri)."""
    d = datetime(year, month, 1, tzinfo=UTC)
    count = 0
    while d.month == month:
        if d.weekday() == weekday:
            count += 1
            if count == n:
                return d
        d += timedelta(days=1)
    return None


def _last_day(year, month):
    """Last day of month."""
    if month == 12:
        return datetime(year, 12, 31, tzinfo=UTC)
    return datetime(year, month + 1, 1, tzinfo=UTC) - timedelta(days=1)


def _weekdays_in_range(year, month, day_start, day_end, weekday):
    """Count weekdays in a date range."""
    count = 0
    for d in range(day_start, min(day_end + 1, 32)):
        try:
            dt = datetime(year, month, d, tzinfo=UTC)
            if dt.weekday() == weekday:
                count += 1
        except ValueError:
            break
    return count


# ── EVENT DEFINITIONS ──

EVENTS = [
    # ═══════════════════════════════════════════════
    # 🇺🇸 TIER 1 — US (Primary Driver)
    # ═══════════════════════════════════════════════
    {
        'id': 'us_nfp',
        'name': 'US Non-Farm Payrolls',
        'country': '🇺🇸 US',
        'tier': 1,
        'schedule': '1st Friday',
        'time_utc': '13:30',
        'impact': 'HIGH',
        'get_next': lambda y, m: _first_friday(y, m).replace(hour=13, minute=30),
        'cascade': {
            'immediate': 'DXY spike → BTC/ETH reaction (seconds)',
            '1h': 'US futures adjust → risk repricing',
            'next_event': 'US Claims (Thu) → labor context check',
            'next_major': 'US CPI (~12-14th) → confirms/denies NFP signal',
            'eth_historical': 'Avg ±1.2% on release, sets tone for 1-2 weeks',
        },
        'what_to_watch': [
            'NFP surprise vs consensus → immediate ETH direction',
            'Unemployment rate → recession signal (Sahm rule)',
            'Wage growth → inflation pipeline → Fed reaction',
        ],
    },
    {
        'id': 'us_claims',
        'name': 'US Jobless Claims',
        'country': '🇺🇸 US',
        'tier': 1,
        'schedule': 'Every Thursday',
        'time_utc': '13:30',
        'impact': 'MEDIUM',
        'get_next': lambda y, m: _next_thursday(y, m),
        'cascade': {
            'immediate': 'Minor — claims is context, not catalyst',
            'next_event': 'Next week: more claims or CPI/PPI',
            'next_major': 'Accumulates into CPI signal — rising claims + hot CPI = recession fear',
            'eth_historical': 'Avg ±0.3% — only extreme prints (>300K or <200K) matter',
        },
        'what_to_watch': [
            'Claims trend (4-week avg) — rising = labor softening',
            'Continuing claims — exhaustion rate',
            'Sahm rule trigger (unemployment 3m avg rises 0.5%+ from low)',
        ],
    },
    {
        'id': 'us_cpi',
        'name': 'US CPI (YoY)',
        'country': '🇺🇸 US',
        'tier': 1,
        'schedule': '~12-14th',
        'time_utc': '13:30',
        'impact': 'HIGHEST',
        'get_next': lambda y, m: _approx_date(y, m, 12).replace(hour=13, minute=30),
        'cascade': {
            'immediate': 'BIGGEST ETH MOVER — immediate ±1-3% spike',
            '30min': 'DXY repricing → EURUSD → global risk assets',
            'next_event': 'US PPI (next day) → confirms/denies CPI',
            'next_major': 'PBOC LPR (~20th) → will China react to US inflation?',
            'next_cycle': 'Next NFP → did labor hold? → cycle repeats',
            'eth_historical': {
                'COOL': 'Avg +1.06% (Fed can cut → risk-on)',
                'HOT': 'Avg -0.45% (Fed stays tight → risk-off)',
            },
        },
        'what_to_watch': [
            'CPI vs consensus surprise — direction AND magnitude',
            'Core CPI (ex food/energy) — sticky inflation indicator',
            'Shelter/rent component — largest weight, slow-moving',
            'Market pricing: did DXY already price in the print?',
        ],
    },
    {
        'id': 'us_ppi',
        'name': 'US PPI (YoY)',
        'country': '🇺🇸 US',
        'tier': 1,
        'schedule': '~13-15th (day after CPI)',
        'time_utc': '13:30',
        'impact': 'MEDIUM',
        'get_next': lambda y, m: _approx_date(y, m, 13).replace(hour=13, minute=30),
        'cascade': {
            'immediate': 'Confirms or denies CPI — pipeline inflation check',
            'next_event': 'PBOC LPR (~20th) → China reaction',
            'next_major': 'Claims next Thursday → labor context',
            'eth_historical': 'Modifier on CPI signal: +0.5% (both cool) to -0.8% (both hot)',
        },
        'what_to_watch': [
            'PPI vs CPI alignment — both hot = persistent inflation',
            'PPI leading CPI by 2-3 months — if PPI >> CPI, CPI will catch up',
            'Goods vs services split — supply chain signal',
        ],
    },
    {
        'id': 'us_fomc',
        'name': 'US FOMC Rate Decision',
        'country': '🇺🇸 US',
        'tier': 1,
        'schedule': '8x/year (~6 weeks)',
        'time_utc': '19:00',
        'impact': 'HIGH',
        'get_next': lambda y, m: _fomc_next(y, m),
        'cascade': {
            'immediate': 'Rate + dot plot → DXY → ETH ±1-2%',
            '30min': 'Powell press conference — forward guidance',
            'next_event': 'FOMC minutes (3 weeks later)',
            'next_major': 'Next CPI → did Fed react to latest inflation?',
            'eth_historical': 'CUT = rally, HOLD = range, HAWKISH = dump',
        },
        'what_to_watch': [
            'Rate decision vs consensus',
            'Dot plot — median rate projection',
            'Powell tone — hawkish/dovish shift',
            'QT taper timing — liquidity signal',
        ],
    },

    # ═══════════════════════════════════════════════
    # 🇨🇳 TIER 1 — China (Second Largest Driver)
    # ═══════════════════════════════════════════════
    {
        'id': 'cn_pmi_official',
        'name': 'China Official PMI',
        'country': '🇨🇳 China',
        'tier': 1,
        'schedule': 'Last day of month',
        'time_utc': '01:00',
        'impact': 'MEDIUM',
        'get_next': lambda y, m: _last_day(y, m).replace(hour=1, minute=0),
        'cascade': {
            'immediate': 'Manufacturing health → risk appetite',
            'next_event': 'Caixin PMI (1-3rd) → private sector check',
            'next_major': 'PBOC LPR (~20th) → will PBOC ease?',
            'eth_historical': 'PMI >50 = expansion (risk-on), <50 = contraction (risk-off)',
        },
        'what_to_watch': [
            'Above/below 50 (expansion threshold)',
            'New orders component — leading indicator',
            'Export orders — global demand proxy',
        ],
    },
    {
        'id': 'cn_pmi_caixin',
        'name': 'China Caixin PMI',
        'country': '🇨🇳 China',
        'tier': 1,
        'schedule': '1st-3rd of month',
        'time_utc': '01:45',
        'impact': 'MEDIUM',
        'get_next': lambda y, m: datetime(y, m, 1, tzinfo=UTC).replace(hour=1, minute=45),
        'cascade': {
            'immediate': 'Private sector health — complements official PMI',
            'next_event': 'Trade balance (~7-10th)',
            'next_major': 'PBOC LPR (~20th)',
            'eth_historical': 'Divergence from official PMI = policy uncertainty',
        },
        'what_to_watch': [
            'Divergence from official PMI — signals policy transmission',
            'Services vs manufacturing split',
        ],
    },
    {
        'id': 'cn_pboc_lpr',
        'name': 'China PBOC LPR Rate',
        'country': '🇨🇳 China',
        'tier': 1,
        'schedule': '20th of month',
        'time_utc': '01:30',
        'impact': 'HIGH',
        'get_next': lambda y, m: datetime(y, m, 20, tzinfo=UTC).replace(hour=1, minute=30),
        'cascade': {
            'immediate': 'Rate CUT = massive liquidity → crypto rally (1-2 week lead)',
            'immediate_up': 'Rate HIKE = liquidity drain → crypto dump',
            'next_event': 'Credit data (~15-25th) → did lending respond?',
            'next_major': 'Next month PMI → did easing work?',
            'eth_historical': 'PBOC cut has ~70% correlation with ETH rally within 2 weeks',
        },
        'what_to_watch': [
            '1-year LPR (corporate) and 5-year LPR (mortgage)',
            'Cut magnitude — 10bp expected, 20bp = aggressive',
            'RRR cut (separate event) — massive liquidity injection',
        ],
    },
    {
        'id': 'cn_credit',
        'name': 'China Credit Data',
        'country': '🇨🇳 China',
        'tier': 1,
        'schedule': '~10-15th',
        'time_utc': '—',
        'impact': 'MEDIUM',
        'get_next': lambda y, m: _approx_date(y, m, 12),
        'cascade': {
            'immediate': 'Credit impulse — LEADING indicator (1-2 month lead on risk assets)',
            'next_event': 'PBOC LPR (~20th) — policy response',
            'next_major': 'Next month PMI — did credit flow work?',
            'eth_historical': 'Rising credit impulse = risk-on (front-runs ETH by 4-8 weeks)',
        },
        'what_to_watch': [
            'Total social financing (TSF) — broad credit measure',
            'New yuan loans — bank lending appetite',
            'Credit impulse (change in credit/GDP) — leading indicator',
        ],
    },
    {
        'id': 'cn_cpi',
        'name': 'China CPI/PPI',
        'country': '🇨🇳 China',
        'tier': 1,
        'schedule': '~9-12th',
        'time_utc': '01:30',
        'impact': 'MEDIUM',
        'get_next': lambda y, m: _approx_date(y, m, 10).replace(hour=1, minute=30),
        'cascade': {
            'immediate': 'Deflation risk → PBOC may ease more aggressively',
            'next_event': 'Credit data (~12-15th)',
            'next_major': 'PBOC LPR (~20th) — deflation = more room to cut',
            'eth_historical': 'China deflation + PBOC cut = bullish for crypto (liquidity flood)',
        },
        'what_to_watch': [
            'CPI negative = deflation → PBOC forced to ease',
            'PPI negative = industrial deflation → weak demand',
            'CPI+PPI both negative = Japan-style trap → massive stimulus expected',
        ],
    },
    {
        'id': 'cn_trade',
        'name': 'China Trade Balance',
        'country': '🇨🇳 China',
        'tier': 2,
        'schedule': '~7-10th',
        'time_utc': '03:00',
        'impact': 'LOW',
        'get_next': lambda y, m: _approx_date(y, m, 8).replace(hour=3),
        'cascade': {
            'immediate': 'Export strength → global demand signal',
            'next_event': 'CPI/PPI (~10th)',
            'eth_historical': 'Weak exports → PBOC easing expectations',
        },
        'what_to_watch': ['Export growth', 'Import growth (domestic demand)'],
    },

    # ═══════════════════════════════════════════════
    # 🇪🇺 TIER 2 — Eurozone
    # ═══════════════════════════════════════════════
    {
        'id': 'eu_hicp_flash',
        'name': 'EU HICP Flash',
        'country': '🇪🇺 Eurozone',
        'tier': 2,
        'schedule': '~1st of month',
        'time_utc': '10:00',
        'impact': 'MEDIUM',
        'get_next': lambda y, m: _approx_date(y, m, 1).replace(hour=10),
        'cascade': {
            'immediate': 'EURUSD → DXY inverse → ETH',
            'next_event': 'ECB decision (if scheduled)',
            'next_major': 'EU HICP Final (~18th)',
            'eth_historical': 'EU inflation hot → ECB hawkish → EURUSD up → DXY down → ETH up',
        },
        'what_to_watch': [
            'Core HICP (ex food/energy) — sticky inflation',
            'vs ECB 2% target — gap determines rate path',
        ],
    },
    {
        'id': 'eu_ecb',
        'name': 'ECB Rate Decision',
        'country': '🇪🇺 Eurozone',
        'tier': 2,
        'schedule': '8x/year (~6 weeks)',
        'time_utc': '13:15',
        'impact': 'HIGH',
        'get_next': lambda y, m: _ecb_next(y, m),
        'cascade': {
            'immediate': 'Rate decision + forward guidance → EURUSD → DXY → ETH',
            '30min': 'Lagarde press conference',
            'next_event': 'ECB minutes (4 weeks later)',
            'next_major': 'Next EU HICP — did inflation respond?',
            'eth_historical': 'ECB CUT = EURUSD up → DXY down → ETH up (indirect)',
        },
        'what_to_watch': [
            'Rate decision vs consensus',
            'Lagarde forward guidance — hawkish/dovish pivot',
            'APP/PEPP taper updates — liquidity signal',
        ],
    },

    # ═══════════════════════════════════════════════
    # 🇯🇵 TIER 2 — Japan (Carry Trade Risk)
    # ═══════════════════════════════════════════════
    {
        'id': 'jp_boj',
        'name': 'BOJ Rate Decision',
        'country': '🇯🇵 Japan',
        'tier': 2,
        'schedule': '8x/year (~6 weeks)',
        'time_utc': '~03:00',
        'impact': 'HIGH',
        'get_next': lambda y, m: _boj_next(y, m),
        'cascade': {
            'immediate': 'Rate HIKE = carry trade unwind = RISK-OFF CRASH',
            'immediate_cut': 'Rate HOLD/DOVISH = carry continues = risk-on',
            '1h': 'USDJPY repricing → global risk assets',
            'next_event': 'Tokyo CPI (~25-28th) — inflation check',
            'next_major': 'Next BOJ — will they hike again?',
            'eth_historical': 'Aug 2024: BOJ hike → carry unwind → ETH -20% in 3 days',
        },
        'what_to_watch': [
            'Rate decision — any hike = CRITICAL risk event',
            'YCC (yield curve control) adjustments',
            'USDJPY — yen strengthening = carry unwind risk',
            'Forward guidance — signaling future hikes',
        ],
        'alert': '⚠️ BOJ hike = highest-impact single event for crypto downside risk',
    },
    {
        'id': 'jp_cpi_tokyo',
        'name': 'Japan Tokyo CPI',
        'country': '🇯🇵 Japan',
        'tier': 2,
        'schedule': '~25-28th',
        'time_utc': '23:30',
        'impact': 'MEDIUM',
        'get_next': lambda y, m: _approx_date(y, m, 26).replace(hour=23, minute=30),
        'cascade': {
            'immediate': 'Leading indicator for national CPI → BOJ policy',
            'next_event': 'National CPI (~18-22nd next month)',
            'next_major': 'Next BOJ — hot Tokyo CPI = more pressure to hike',
            'eth_historical': 'Tokyo CPI >3% → BOJ hike probability rises → JPY strengthens → risk-off',
        },
        'what_to_watch': [
            'Ex-fresh-food (core) — BOJ target',
            'Trend — rising = BOJ under pressure to hike',
        ],
    },
    {
        'id': 'jp_cpi_national',
        'name': 'Japan National CPI',
        'country': '🇯🇵 Japan',
        'tier': 2,
        'schedule': '~18-22nd',
        'time_utc': '23:30',
        'impact': 'MEDIUM',
        'get_next': lambda y, m: _approx_date(y, m, 20).replace(hour=23, minute=30),
        'cascade': {
            'immediate': 'Confirms Tokyo CPI trend → BOJ pressure',
            'next_event': 'BOJ decision (if scheduled)',
            'next_major': 'Next Tokyo CPI (~25-28th next month)',
        },
        'what_to_watch': ['Core CPI trend', 'Services vs goods inflation'],
    },

    # ═══════════════════════════════════════════════
    # 🇬🇧 TIER 3 — UK (London Session)
    # ═══════════════════════════════════════════════
    {
        'id': 'uk_cpi',
        'name': 'UK CPI',
        'country': '🇬🇧 UK',
        'tier': 3,
        'schedule': '~17-19th',
        'time_utc': '06:00',
        'impact': 'MEDIUM',
        'get_next': lambda y, m: _approx_date(y, m, 18).replace(hour=6),
        'cascade': {
            'immediate': 'GBPUSD → minor DXY impact → indirect ETH',
            'next_event': 'BOE decision (if scheduled)',
            'next_major': 'London session direction',
            'eth_historical': 'UK CPI affects London session tone — already tracked in M23',
        },
        'what_to_watch': ['Core CPI', 'Services inflation — BOJ-style sticky component'],
    },
    {
        'id': 'uk_boe',
        'name': 'BOE Rate Decision',
        'country': '🇬🇧 UK',
        'tier': 3,
        'schedule': '8x/year (~6 weeks)',
        'time_utc': '12:00',
        'impact': 'MEDIUM',
        'get_next': lambda y, m: _boe_next(y, m),
        'cascade': {
            'immediate': 'Rate decision → GBPUSD → minor ETH impact',
            'next_event': 'Next UK CPI',
        },
        'what_to_watch': ['Rate decision', 'Bailey forward guidance'],
    },

    # ═══════════════════════════════════════════════
    # 🇰🇷 TIER 3 — South Korea (Retail Proxy)
    # ═══════════════════════════════════════════════
    {
        'id': 'kr_kimchi',
        'name': 'Kimchi Premium',
        'country': '🇰🇷 Korea',
        'tier': 3,
        'schedule': 'Real-time',
        'time_utc': '—',
        'impact': 'MEDIUM',
        'get_next': lambda y, m: None,  # Real-time
        'cascade': {
            'immediate': 'Premium = retail FOMO, Discount = fear',
            'eth_historical': '>5% premium = local top signal, <-2% discount = panic',
        },
        'what_to_watch': [
            'Premium/discount % — retail sentiment proxy',
            'KRW volume — Korean retail activity',
        ],
    },
]


# ── Helper functions ──

def _approx_date(year, month, day):
    """Approximate date, clamped to valid range."""
    import calendar
    max_day = calendar.monthrange(year, month)[1]
    return datetime(year, month, min(day, max_day), tzinfo=UTC)


def _next_thursday(year, month):
    """Next Thursday from today (for claims)."""
    now = datetime.now(UTC)
    d = now
    while d.weekday() != 3:  # Thursday
        d += timedelta(days=1)
    return d.replace(hour=13, minute=30, second=0, microsecond=0)


def _fomc_next(year, month):
    """Approximate FOMC dates (8x/year, roughly every 6 weeks)."""
    # 2026 FOMC dates (approximate): Jan, Mar, May, Jun, Jul, Sep, Oct, Dec
    fomc_months_2026 = {
        1: 29, 2: None, 3: 19, 4: None, 5: 7, 6: 18,
        7: 30, 8: None, 9: 18, 10: 29, 11: None, 12: 17,
    }
    day = fomc_months_2026.get(month)
    if day:
        return datetime(year, month, day, 19, 0, tzinfo=UTC)
    return None


def _ecb_next(year, month):
    """Approximate ECB dates (8x/year, roughly every 6 weeks)."""
    ecb_months_2026 = {
        1: None, 2: 5, 3: None, 4: 16, 5: None, 6: 4,
        7: None, 8: 6, 9: None, 10: 29, 11: None, 12: 17,
    }
    day = ecb_months_2026.get(month)
    if day:
        return datetime(year, month, day, 13, 15, tzinfo=UTC)
    return None


def _boj_next(year, month):
    """Approximate BOJ dates (8x/year, roughly every 6 weeks)."""
    boj_months_2026 = {
        1: None, 2: 19, 3: None, 4: 17, 5: None, 6: 18,
        7: None, 8: 6, 9: None, 10: 30, 11: None, 12: 18,
    }
    day = boj_months_2026.get(month)
    if day:
        return datetime(year, month, day, 3, 0, tzinfo=UTC)
    return None


def _boe_next(year, month):
    """Approximate BOE dates (8x/year, roughly every 6 weeks)."""
    boe_months_2026 = {
        1: None, 2: 5, 3: None, 4: 16, 5: None, 6: 18,
        7: None, 8: 6, 9: None, 10: 8, 11: None, 12: 17,
    }
    day = boe_months_2026.get(month)
    if day:
        return datetime(year, month, day, 12, 0, tzinfo=UTC)
    return None


# ── Main API ──

def get_macro_calendar(reference_time=None):
    """Get the macro calendar with upcoming events.

    Args:
        reference_time: Current time (default: now UTC)

    Returns:
        dict with:
          - now: reference time
          - events: list of upcoming events with countdown
          - cascade_chain: full monthly cascade
          - current_phase: where we are in the cycle
    """
    now = reference_time or datetime.now(UTC)
    year, month = now.year, now.month

    upcoming = []
    all_events = []

    for evt_def in EVENTS:
        # Try current month
        next_dt = None
        if evt_def['get_next']:
            try:
                next_dt = evt_def['get_next'](year, month)
            except (ValueError, TypeError):
                pass

            # If past or None, try next month
            if next_dt is None or next_dt < now:
                try:
                    if month == 12:
                        next_dt = evt_def['get_next'](year + 1, 1)
                    else:
                        next_dt = evt_def['get_next'](year, month + 1)
                except (ValueError, TypeError):
                    pass

        entry = {
            'id': evt_def['id'],
            'name': evt_def['name'],
            'country': evt_def['country'],
            'tier': evt_def['tier'],
            'schedule': evt_def['schedule'],
            'time_utc': evt_def['time_utc'],
            'impact': evt_def['impact'],
            'cascade': evt_def['cascade'],
            'what_to_watch': evt_def['what_to_watch'],
            'alert': evt_def.get('alert'),
            'next_dt': next_dt,
        }

        if next_dt:
            delta = next_dt - now
            hours = delta.total_seconds() / 3600
            entry['hours_until'] = round(hours, 1)
            entry['countdown'] = _format_countdown(delta)
            entry['is_next_24h'] = hours <= 24
            entry['is_next_4h'] = hours <= 4
            entry['is_next_1h'] = hours <= 1
        else:
            entry['hours_until'] = None
            entry['countdown'] = 'real-time'
            entry['is_next_24h'] = False
            entry['is_next_4h'] = False
            entry['is_next_1h'] = False

        all_events.append(entry)

    # Sort by time
    all_events.sort(key=lambda e: e['next_dt'] or datetime.max.replace(tzinfo=UTC))

    # Split into upcoming (next 30 days) and later
    cutoff = now + timedelta(days=30)
    upcoming = [e for e in all_events if e['next_dt'] and e['next_dt'] < cutoff]

    # Current phase detection
    day = now.day
    if day <= 3:
        phase = 'MONTH_START'
        phase_desc = 'PMI releases, NFP approaching'
        next_major = 'NFP (1st Friday)'
    elif day <= 7:
        phase = 'NFP_WEEK'
        phase_desc = 'NFP sets tone for the month'
        next_major = 'CPI/PPI (~12-14th)'
    elif day <= 14:
        phase = 'CPI_WEEK'
        phase_desc = 'CPI/PPI — biggest movers of the month'
        next_major = 'PBOC LPR (~20th)'
    elif day <= 21:
        phase = 'MID_MONTH'
        phase_desc = 'PBOC, ECB/BOJ, China data cluster'
        next_major = 'End-of-month PMIs'
    elif day <= 28:
        phase = 'LATE_MONTH'
        phase_desc = 'Tokyo CPI, claims accumulation, PMI prep'
        next_major = 'Month-end PMIs → next NFP'
    else:
        phase = 'MONTH_END'
        phase_desc = 'PMI releases, cycle reset'
        next_major = 'Next month NFP'

    # Cascade chain for the month
    cascade_chain = _build_cascade_chain(upcoming, now)

    return {
        'now': now.isoformat(),
        'phase': phase,
        'phase_desc': phase_desc,
        'next_major': next_major,
        'events': upcoming,
        'all_events': all_events,
        'cascade_chain': cascade_chain,
    }


def _format_countdown(delta):
    """Format timedelta as human-readable countdown."""
    total_seconds = int(delta.total_seconds())
    if total_seconds < 0:
        return 'PASSED'
    days = total_seconds // 86400
    hours = (total_seconds % 86400) // 3600
    minutes = (total_seconds % 3600) // 60

    if days > 0:
        return f'{days}d {hours}h'
    elif hours > 0:
        return f'{hours}h {minutes}m'
    else:
        return f'{minutes}m'


def _build_cascade_chain(events, now):
    """Build the cascade chain showing what follows what."""
    chain = []
    for evt in events:
        if evt['hours_until'] is None or evt['hours_until'] > 30 * 24:
            continue
        cascade = evt.get('cascade', {})
        entry = {
            'event': evt['name'],
            'country': evt['country'],
            'countdown': evt['countdown'],
            'hours_until': evt['hours_until'],
            'impact': evt['impact'],
            'triggers': [],
        }
        for key, val in cascade.items():
            if key == 'eth_historical':
                continue
            entry['triggers'].append(f'{key}: {val}')
        chain.append(entry)
    return chain


# ── Formatting ──

def format_macro_calendar(cal):
    """Format macro calendar for terminal output."""
    lines = []
    lines.append('')
    lines.append('═' * 60)
    lines.append('  📅 MACRO CALENDAR — LIVE TRACKER')
    lines.append('═' * 60)
    lines.append(f'\n  Now: {cal["now"]}')
    lines.append(f'  Phase: {_phase_icon(cal["phase"])} {cal["phase"]} — {cal["phase_desc"]}')
    lines.append(f'  Next major: {cal["next_major"]}')

    # ── Next 24h ──
    next_24h = [e for e in cal['events'] if e.get('is_next_24h')]
    if next_24h:
        lines.append(f'\n  ⚡ NEXT 24 HOURS:')
        for evt in next_24h:
            icon = _impact_icon(evt['impact'])
            alert = ' 🚨' if evt.get('is_next_1h') else ''
            lines.append(f'    {icon} {evt["countdown"]:>10}  {evt["name"]:30}  {evt["time_utc"]}{alert}')
    else:
        lines.append(f'\n  ⚡ NEXT 24 HOURS: (none)')

    # ── Upcoming events (full list) ──
    lines.append(f'\n  📋 UPCOMING EVENTS (next 30 days):')
    lines.append(f'    {"Countdown":>10}  {"Event":32} {"Country":12} {"Impact":8} {"Time":>6}')
    lines.append(f'    {"─" * 10}  {"─" * 32} {"─" * 12} {"─" * 8} {"─" * 6}')

    for evt in cal['events'][:20]:  # Show next 20 events
        icon = _impact_icon(evt['impact'])
        cd = evt['countdown']
        name = evt['name'][:30]
        country = evt['country']
        impact = evt['impact']
        time = evt['time_utc']
        alert = ' 🚨' if evt.get('is_next_4h') else ''
        lines.append(f'    {icon} {cd:>10}  {name:32} {country:12} {impact:8} {time:>6}{alert}')

    # ── Cascade chain ──
    if cal['cascade_chain']:
        lines.append(f'\n  🔗 CASCADE CHAIN (what follows what):')
        for i, entry in enumerate(cal['cascade_chain'][:8]):
            icon = _impact_icon(entry['impact'])
            lines.append(f'    {i+1}. {icon} {entry["event"]}  ({entry["countdown"]})')
            for trigger in entry['triggers'][:2]:
                lines.append(f'       → {trigger}')

    # ── Current phase context ──
    lines.append(f'\n  📍 WHERE ARE WE IN THE CYCLE?')
    lines.append(f'    Phase: {_phase_icon(cal["phase"])} {cal["phase"]}')
    lines.append(f'    {cal["phase_desc"]}')
    lines.append(f'    Next major: {cal["next_major"]}')

    # Phase-specific advice
    phase_advice = {
        'MONTH_START': 'PMI data incoming — watch for China/EU demand signals before NFP',
        'NFP_WEEK': 'NFP sets the tone — wait for CPI confirmation before positioning',
        'CPI_WEEK': '⚠️ BIGGEST MOVERS — CPI/PPI are the primary ETH catalysts',
        'MID_MONTH': 'PBOC LPR — watch for China easing signal (1-2 week ETH lead)',
        'LATE_MONTH': 'Tokyo CPI → BOJ risk — carry trade unwind is the tail risk',
        'MONTH_END': 'PMI releases → cycle resets → prepare for next NFP',
    }
    advice = phase_advice.get(cal['phase'], '')
    if advice:
        lines.append(f'    💡 {advice}')

    lines.append('═' * 60)
    return '\n'.join(lines)


def _phase_icon(phase):
    return {
        'MONTH_START': '📅',
        'NFP_WEEK': '💥',
        'CPI_WEEK': '🔥',
        'MID_MONTH': '🏦',
        'LATE_MONTH': '📊',
        'MONTH_END': '🔄',
    }.get(phase, '📍')


def _impact_icon(impact):
    return {
        'HIGHEST': '🔴',
        'HIGH': '🟠',
        'MEDIUM': '🟡',
        'LOW': '⚪',
    }.get(impact, '⚪')


def format_macro_calendar_compact(cal):
    """Compact one-line format for scanner integration."""
    lines = []
    lines.append('\n  📅 MACRO CALENDAR:')

    phase_icons = {
        'MONTH_START': '📅', 'NFP_WEEK': '💥', 'CPI_WEEK': '🔥',
        'MID_MONTH': '🏦', 'LATE_MONTH': '📊', 'MONTH_END': '🔄',
    }
    lines.append(f'    Phase: {phase_icons.get(cal["phase"], "📍")} {cal["phase"]} — {cal["phase_desc"]}')

    # Next 3 events
    for evt in cal['events'][:3]:
        icon = _impact_icon(evt['impact'])
        alert = ' 🚨' if evt.get('is_next_4h') else ''
        lines.append(f'    {icon} {evt["countdown"]:>10} → {evt["name"]} ({evt["country"]}){alert}')

    # Cascade
    if cal['cascade_chain']:
        first = cal['cascade_chain'][0]
        lines.append(f'    → After {first["event"]}: {first["triggers"][0] if first["triggers"] else "?"}')

    return '\n'.join(lines)


# ── For JSON output ──

def calendar_to_dict(cal):
    """Convert calendar to JSON-serializable dict."""
    result = {
        'now': cal['now'],
        'phase': cal['phase'],
        'phase_desc': cal['phase_desc'],
        'next_major': cal['next_major'],
        'events': [],
        'cascade_chain': cal['cascade_chain'],
    }
    for evt in cal['events']:
        result['events'].append({
            'id': evt['id'],
            'name': evt['name'],
            'country': evt['country'],
            'tier': evt['tier'],
            'impact': evt['impact'],
            'countdown': evt['countdown'],
            'hours_until': evt['hours_until'],
            'time_utc': evt['time_utc'],
            'is_next_24h': evt['is_next_24h'],
            'is_next_4h': evt['is_next_4h'],
            'cascade': evt['cascade'],
            'what_to_watch': evt['what_to_watch'],
        })
    return result


# ── Test ──

if __name__ == '__main__':
    cal = get_macro_calendar()
    print(format_macro_calendar(cal))

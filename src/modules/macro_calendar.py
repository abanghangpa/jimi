"""
Macro Calendar Tracker v2 — Full 38-event global cycle tracker.

Tracks scheduled economic data releases across:
  🇺🇸 US | 🇪🇺 Eurozone/ECB | 🇬🇧 UK | 🇯🇵 Japan | 🇨🇳 China | 🇦🇺 Australia

Features:
  - 38 scheduled events + real-time signals
  - "What comes next" cascade chains per event
  - 4 narrative chains: Inflation, Labour, Growth, Central Bank
  - Regime-adjusted expected moves for CPI
  - Phase detection (where in the monthly cycle)
  - Countdown timers with 1h/4h/24h alerts

Usage:
    from src.modules.macro_calendar import get_macro_calendar, format_macro_calendar
    cal = get_macro_calendar()
    print(format_macro_calendar(cal, current_regime='STAGFLATION_HOT'))
"""

import json
import os
from datetime import datetime, timedelta, timezone
from typing import Optional

UTC = timezone.utc


# ══════════════════════════════════════════════════════════════
# HELPER FUNCTIONS
# ══════════════════════════════════════════════════════════════

def _approx_date(year, month, day):
    """Approximate date, clamped to valid range."""
    import calendar
    max_day = calendar.monthrange(year, month)[1]
    return datetime(year, month, min(day, max_day), tzinfo=UTC)


def _first_friday(year, month):
    d = datetime(year, month, 1, tzinfo=UTC)
    while d.weekday() != 4:
        d += timedelta(days=1)
    return d


def _first_business_day(year, month):
    d = datetime(year, month, 1, tzinfo=UTC)
    while d.weekday() >= 5:
        d += timedelta(days=1)
    return d


def _nth_business_day(year, month, n):
    d = datetime(year, month, 1, tzinfo=UTC)
    count = 0
    while d.month == month:
        if d.weekday() < 5:
            count += 1
            if count == n:
                return d
        d += timedelta(days=1)
    return None


def _last_day(year, month):
    if month == 12:
        return datetime(year, 12, 31, tzinfo=UTC)
    return datetime(year, month + 1, 1, tzinfo=UTC) - timedelta(days=1)


def _next_thursday():
    now = datetime.now(UTC)
    d = now
    while d.weekday() != 3:
        d += timedelta(days=1)
    return d.replace(hour=13, minute=30, second=0, microsecond=0)


def _fomc_next(year, month):
    fomc_2026 = {
        1: 29, 3: 19, 5: 7, 6: 18, 7: 30, 9: 18, 10: 29, 12: 17,
    }
    day = fomc_2026.get(month)
    if day:
        return datetime(year, month, day, 19, 0, tzinfo=UTC)
    return None


def _ecb_next(year, month):
    ecb_2026 = {
        2: 5, 4: 16, 6: 4, 8: 6, 10: 29, 12: 17,
    }
    day = ecb_2026.get(month)
    if day:
        return datetime(year, month, day, 13, 15, tzinfo=UTC)
    return None


def _boj_next(year, month):
    boj_2026 = {
        2: 19, 4: 17, 6: 18, 8: 6, 10: 30, 12: 18,
    }
    day = boj_2026.get(month)
    if day:
        return datetime(year, month, day, 3, 0, tzinfo=UTC)
    return None


def _boe_next(year, month):
    boe_2026 = {
        2: 5, 4: 16, 6: 18, 8: 6, 10: 8, 12: 17,
    }
    day = boe_2026.get(month)
    if day:
        return datetime(year, month, day, 12, 0, tzinfo=UTC)
    return None


def _rba_next(year, month):
    """RBA meets first Tuesday of each month (except January)."""
    if month == 1:
        return None
    d = datetime(year, month, 1, tzinfo=UTC)
    while d.weekday() != 1:  # Tuesday
        d += timedelta(days=1)
    return d.replace(hour=3, minute=30)


def _format_countdown(delta):
    total = int(delta.total_seconds())
    if total < 0:
        return 'PASSED'
    d = total // 86400
    h = (total % 86400) // 3600
    m = (total % 3600) // 60
    if d > 0:
        return f'{d}d {h}h'
    elif h > 0:
        return f'{h}h {m}m'
    else:
        return f'{m}m'


def _impact_icon(impact):
    return {'HIGHEST': '🔴', 'HIGH': '🟠', 'MEDIUM': '🟡', 'LOW': '⚪'}.get(impact, '⚪')


def _phase_icon(phase):
    return {
        'MONTH_START': '📅', 'NFP_WEEK': '💥', 'CPI_WEEK': '🔥',
        'MID_MONTH': '🏦', 'LATE_MONTH': '📊', 'MONTH_END': '🔄',
    }.get(phase, '📍')


# ══════════════════════════════════════════════════════════════
# NARRATIVE CHAINS — how data cascades across countries
# ══════════════════════════════════════════════════════════════

NARRATIVE_CHAINS = {
    'inflation': {
        'name': '🔥 INFLATION CHAIN (most ETH-sensitive)',
        'chain': [
            'China CPI/PPI',
            'Germany CPI',
            'Eurozone CPI Flash',
            'US CPI (headline + core)',
            'US PPI',
            'Core PCE',
            'FOMC rate decision',
            'Powell presser',
            'DXY reaction',
            'ETH/USDT move',
        ],
        'eth_sensitivity': 'HIGHEST',
        'note': 'CPI is the #1 ETH macro mover. Core PCE is the Fed\'s actual target.',
    },
    'labour': {
        'name': '👷 LABOUR CHAIN',
        'chain': [
            'JOLTS Job Openings',
            'ADP Employment',
            'Jobless Claims',
            'NFP + Unemployment + Wages',
            'Fed speeches',
            'CME FedWatch repricing',
            'Rate cut/hike probability',
            'ETH/USDT trend',
        ],
        'eth_sensitivity': 'HIGH',
        'note': 'NFP sets the tone, CPI confirms. Claims is weekly context.',
    },
    'growth': {
        'name': '📈 GROWTH CHAIN',
        'chain': [
            'China GDP',
            'Eurozone GDP Flash',
            'Germany GDP',
            'US GDP Advance',
            'Corporate earnings backdrop',
            'Risk-on / risk-off regime',
        ],
        'eth_sensitivity': 'MEDIUM',
        'note': 'GDP is quarterly — sets macro backdrop, not a trade trigger.',
    },
    'central_bank': {
        'name': '🏦 CENTRAL BANK CHAIN',
        'chain': [
            'BoJ decision (carry trade)',
            'ECB decision (EUR/USD)',
            'BoE decision (GBP/USD)',
            'FOMC decision (DXY)',
            'Powell presser',
            '10Y yield',
            'ETH/USDT',
        ],
        'eth_sensitivity': 'HIGH',
        'note': 'BoJ is the tail risk (carry unwind). FOMC is the primary driver.',
    },
}

# ══════════════════════════════════════════════════════════════
# REAL-TIME SIGNALS — always watch alongside scheduled data
# ══════════════════════════════════════════════════════════════

REALTIME_SIGNALS = [
    {'id': 'usdjpy', 'name': 'USD/JPY', 'note': 'Yen spike = carry unwind = ETH drops fast', 'eth_link': 'direct'},
    {'id': 'dxy', 'name': 'DXY (Dollar Index)', 'note': 'Strong dollar = crypto headwind', 'eth_link': 'inverse'},
    {'id': 'us10y', 'name': '10Y Treasury Yield', 'note': 'Yield spike = risk-off = ETH sells', 'eth_link': 'inverse'},
    {'id': 'vix', 'name': 'VIX (Fear Gauge)', 'note': 'Above 20 = caution, above 30 = panic', 'eth_link': 'inverse'},
    {'id': 'oil', 'name': 'WTI / Brent Crude', 'note': 'Oil spike = inflation fear = ETH pressure', 'eth_link': 'inverse'},
    {'id': 'gold', 'name': 'Gold', 'note': 'Safe haven, often co-moves with BTC/ETH', 'eth_link': 'correlated'},
    {'id': 'fedwatch', 'name': 'CME FedWatch', 'note': 'Real-time rate cut/hike probability', 'eth_link': 'direct'},
    {'id': 'eth_funding', 'name': 'ETH Funding Rate', 'note': 'Internal crypto signal for leverage bias', 'eth_link': 'direct'},
]


# ══════════════════════════════════════════════════════════════
# EVENT DEFINITIONS — 38 scheduled events
# ══════════════════════════════════════════════════════════════

EVENTS = [
    # ───────────────────────────────────────────────────
    # 🇨🇳 CHINA
    # ───────────────────────────────────────────────────
    {
        'id': 'cn_caixin_mfg_pmi',
        'num': '01',
        'name': 'China Caixin Manufacturing PMI',
        'country': '🇨🇳 China',
        'tier': 1,
        'schedule': '1st of month',
        'time_utc': '01:45',
        'impact': 'MEDIUM',
        'get_next': lambda y, m: datetime(y, m, 1, 1, 45, tzinfo=UTC),
        'what_comes_next': [
            'NBS Manufacturing PMI — official vs private divergence tells full picture',
            'Eurozone PMI Flash — same-day or next-day, global factory comparison',
            'US ISM Manufacturing PMI — completes global PMI picture ~2 days later',
        ],
        'what_to_watch': ['Above/below 50 (expansion threshold)', 'New orders sub-index', 'Export orders — global demand proxy'],
    },
    {
        'id': 'cn_nbs_pmi',
        'num': '02',
        'name': 'NBS Manufacturing + Services PMI',
        'country': '🇨🇳 China',
        'tier': 1,
        'schedule': '1st of month',
        'time_utc': '01:00',
        'impact': 'MEDIUM',
        'get_next': lambda y, m: datetime(y, m, 1, 1, 0, tzinfo=UTC),
        'what_comes_next': [
            'Caixin Services PMI — private services side, divergence from official',
            'RBA Rate Decision — Australia reacts to China data (top trade partner)',
            'Eurozone Flash PMI — Europe PMI drops same week, narrative chain',
        ],
        'what_to_watch': ['Above/below 50', 'New orders vs inventories split', 'Employment sub-index'],
    },
    {
        'id': 'cn_cpi',
        'num': '15',
        'name': 'China CPI + PPI',
        'country': '🇨🇳 China',
        'tier': 1,
        'schedule': '~9th–11th',
        'time_utc': '01:30',
        'impact': 'MEDIUM',
        'get_next': lambda y, m: _approx_date(y, m, 10).replace(hour=1, minute=30),
        'what_comes_next': [
            'China Trade Balance — released same week, demand story completes',
            'PBoC LPR Decision — deflation accelerates rate cut probability',
            'Commodity prices (oil, copper) — China deflation = global demand fear',
        ],
        'what_to_watch': ['CPI negative = deflation → PBOC forced to ease', 'PPI negative = industrial deflation', 'Both negative = Japan-style trap → massive stimulus expected'],
    },
    {
        'id': 'cn_gdp',
        'num': '30',
        'name': 'China GDP (Quarterly)',
        'country': '🇨🇳 China',
        'tier': 1,
        'schedule': 'Quarterly (~18th)',
        'time_utc': '03:00',
        'impact': 'MEDIUM',
        'get_next': lambda y, m: _approx_date(y, m, 18).replace(hour=3) if m in (1, 4, 7, 10) else None,
        'what_comes_next': [
            'China Retail Sales — released same day, growth quality check',
            'China Industrial Output — released same day, supply side of GDP',
            'PBoC LPR Decision — miss = stimulus expectations spike',
        ],
        'what_to_watch': ['vs target (5%)', 'Quarterly acceleration/deceleration', 'Property sector drag'],
    },
    {
        'id': 'cn_pboc_lpr',
        'num': '31',
        'name': 'PBoC LPR Decision',
        'country': '🇨🇳 China',
        'tier': 1,
        'schedule': '20th of month',
        'time_utc': '01:30',
        'impact': 'HIGH',
        'get_next': lambda y, m: datetime(y, m, 20, 1, 30, tzinfo=UTC),
        'what_comes_next': [
            'CNY/USD fixing — rate cut = CNY weakness = BTC/ETH demand from China',
            'China property data — LPR cuts aimed at real estate, watch developer stress',
            'RBA response — Australia top China trade partner, AUD reacts',
        ],
        'what_to_watch': ['1-year LPR (corporate) and 5-year LPR (mortgage)', 'Cut magnitude — 10bp expected, 20bp = aggressive', 'RRR cut (separate event) — massive liquidity injection'],
        'eth_historical': 'PBOC cut has ~70% correlation with ETH rally within 2 weeks',
    },
    {
        'id': 'cn_trade',
        'num': '—',
        'name': 'China Trade Balance',
        'country': '🇨🇳 China',
        'tier': 2,
        'schedule': '~7th–10th',
        'time_utc': '03:00',
        'impact': 'LOW',
        'get_next': lambda y, m: _approx_date(y, m, 8).replace(hour=3),
        'what_comes_next': [
            'China CPI/PPI — released same week, demand + inflation together',
            'PBoC LPR — weak trade = easing expectations',
        ],
        'what_to_watch': ['Export growth', 'Import growth (domestic demand)'],
    },
    {
        'id': 'cn_credit',
        'num': '—',
        'name': 'China Credit Data (TSF)',
        'country': '🇨🇳 China',
        'tier': 1,
        'schedule': '~10th–15th',
        'time_utc': '—',
        'impact': 'MEDIUM',
        'get_next': lambda y, m: _approx_date(y, m, 12),
        'what_comes_next': [
            'PBoC LPR — credit impulse leads policy by 2-4 weeks',
            'Next month PMI — did credit flow work?',
        ],
        'what_to_watch': ['Total social financing (TSF)', 'New yuan loans', 'Credit impulse (change in credit/GDP) — leading indicator'],
        'eth_historical': 'Rising credit impulse = risk-on, front-runs ETH by 4-8 weeks',
    },

    # ───────────────────────────────────────────────────
    # 🇪🇺 EUROPE / ECB
    # ───────────────────────────────────────────────────
    {
        'id': 'eu_pmi_flash',
        'num': '03',
        'name': 'Eurozone Flash PMI (Composite)',
        'country': '🇪🇺 Eurozone',
        'tier': 1,
        'schedule': '~22nd',
        'time_utc': '08:00',
        'impact': 'MEDIUM',
        'get_next': lambda y, m: _approx_date(y, m, 22).replace(hour=8),
        'what_comes_next': [
            'UK Flash PMI — released same day, compare EU vs UK health',
            'Germany Ifo Business Climate — follows within days, confirms PMI',
            'ECB Rate Decision — PMI weakness = ECB cut expectations build',
        ],
        'what_to_watch': ['Composite above/below 50', 'Manufacturing vs services split', 'Input prices sub-index — inflation preview'],
    },
    {
        'id': 'de_cpi',
        'num': '10',
        'name': 'Germany CPI',
        'country': '🇩🇪 Germany',
        'tier': 1,
        'schedule': '~28th–30th',
        'time_utc': '—',
        'impact': 'MEDIUM',
        'get_next': lambda y, m: _approx_date(y, m, 29),
        'what_comes_next': [
            'Eurozone CPI Flash — released 1-2 days later, Germany leads the number',
            'ECB Rate Decision — hot German CPI = ECB hawkish pressure',
            'EUR/USD direction — strong CPI = ECB tightening = EUR up = DXY down',
        ],
        'what_to_watch': ['Harmonized index (HICP) — what ECB uses', 'Core vs headline divergence', 'Services vs goods inflation'],
    },
    {
        'id': 'eu_hicp_flash',
        'num': '11',
        'name': 'Eurozone CPI Flash',
        'country': '🇪🇺 Eurozone',
        'tier': 1,
        'schedule': 'Last day of month',
        'time_utc': '10:00',
        'impact': 'MEDIUM',
        'get_next': lambda y, m: _last_day(y, m).replace(hour=10),
        'what_comes_next': [
            'ECB Rate Decision — CPI is #1 input, hot = hold/hike, cool = cut',
            'US CPI — released ~2 weeks later, global inflation comparison',
            'Core PCE (US) — US PCE follows ~4 weeks later, confirms/diverges',
        ],
        'what_to_watch': ['Core HICP (ex food/energy) — sticky inflation', 'vs ECB 2% target', 'Services inflation — persistent component'],
    },
    {
        'id': 'eu_ecb',
        'num': '20',
        'name': 'ECB Rate Decision + Lagarde Presser',
        'country': '🇪🇺 Eurozone',
        'tier': 1,
        'schedule': '8x/year (~6 weeks)',
        'time_utc': '13:15',
        'impact': 'HIGH',
        'get_next': lambda y, m: _ecb_next(y, m),
        'what_comes_next': [
            'Eurozone PMI Flash — released same week, activity validates rate call',
            'Eurozone GDP Flash — next quarter, growth trajectory post-decision',
            'Fed Rate Decision — ECB moves first → reprices DXY → ETH reacts',
        ],
        'what_to_watch': ['Rate decision vs consensus', 'Lagarde forward guidance — hawkish/dovish pivot', 'APP/PEPP taper updates — liquidity signal'],
    },
    {
        'id': 'eu_gdp',
        'num': '26',
        'name': 'Eurozone GDP Flash',
        'country': '🇪🇺 Eurozone',
        'tier': 1,
        'schedule': 'Quarterly (~4th week)',
        'time_utc': '10:00',
        'impact': 'MEDIUM',
        'get_next': lambda y, m: _approx_date(y, m, 25).replace(hour=10) if m in (1, 4, 7, 10) else None,
        'what_comes_next': [
            'Germany GDP — released same week, biggest component confirmed',
            'ECB Rate Decision — negative EZ GDP = ECB cuts accelerate',
            'US GDP Advance — EZ GDP drops first, US follows ~1 week later',
        ],
        'what_to_watch': ['Recession threshold (2 consecutive negative quarters)', 'Germany drag', 'Quarterly momentum shift'],
    },
    {
        'id': 'de_ifo',
        'num': '33',
        'name': 'Germany Ifo Business Climate',
        'country': '🇩🇪 Germany',
        'tier': 2,
        'schedule': '~23rd',
        'time_utc': '08:00',
        'impact': 'LOW',
        'get_next': lambda y, m: _approx_date(y, m, 23).replace(hour=8),
        'what_comes_next': [
            'Germany Factory Orders — released 1-2 weeks later, confirms Ifo',
            'ZEW Economic Sentiment — released same week, analyst vs business view',
            'Eurozone GDP Flash — Ifo leads GDP by ~6 weeks',
        ],
        'what_to_watch': ['9,000 firm survey — best EU forward indicator', 'Expectations vs current conditions', 'Manufacturing vs services'],
    },

    # ───────────────────────────────────────────────────
    # 🇬🇧 UNITED KINGDOM
    # ───────────────────────────────────────────────────
    {
        'id': 'uk_pmi',
        'num': '04',
        'name': 'UK Flash PMI',
        'country': '🇬🇧 UK',
        'tier': 2,
        'schedule': '~22nd',
        'time_utc': '08:30',
        'impact': 'LOW',
        'get_next': lambda y, m: _approx_date(y, m, 22).replace(hour=8, minute=30),
        'what_comes_next': [
            'UK CPI — PMI cost pressures preview inflation print',
            'BoE Rate Decision — weak PMI adds to easing case',
            'UK GDP Monthly — PMI is leading, GDP confirms with 3-4 week lag',
        ],
        'what_to_watch': ['Composite above/below 50', 'Input cost pressures — inflation preview'],
    },
    {
        'id': 'uk_cpi',
        'num': '12',
        'name': 'UK CPI',
        'country': '🇬🇧 UK',
        'tier': 1,
        'schedule': '~3rd week',
        'time_utc': '06:00',
        'impact': 'MEDIUM',
        'get_next': lambda y, m: _approx_date(y, m, 18).replace(hour=6),
        'what_comes_next': [
            'BoE Rate Decision — primary input, services inflation drives BoE stance',
            'UK Employment + Wages — released same week, wages drive services CPI',
            'UK GDP Monthly — inflation + growth together = full picture',
        ],
        'what_to_watch': ['Services CPI most watched by BoE', 'Core CPI trend', 'Wage-price spiral risk'],
    },
    {
        'id': 'uk_boe',
        'num': '21',
        'name': 'BoE Rate Decision + MPC Vote',
        'country': '🇬🇧 UK',
        'tier': 1,
        'schedule': '8x/year (~6 weeks)',
        'time_utc': '12:00',
        'impact': 'MEDIUM',
        'get_next': lambda y, m: _boe_next(y, m),
        'what_comes_next': [
            'UK CPI (next month) — MPC dissenters signal what data matters',
            'UK GDP Monthly — released within 2 weeks, validates rate rationale',
            'GBP/USD reaction — BoE dovish = GBP falls = DXY up = ETH pressure',
        ],
        'what_to_watch': ['Split votes most watched', 'Dissenters direction', 'Bailey forward guidance'],
    },
    {
        'id': 'uk_gdp',
        'num': '35',
        'name': 'UK GDP Monthly',
        'country': '🇬🇧 UK',
        'tier': 2,
        'schedule': 'Monthly',
        'time_utc': '06:00',
        'impact': 'LOW',
        'get_next': lambda y, m: _approx_date(y, m, 13).replace(hour=6),
        'what_comes_next': [
            'BoE Rate Decision — consecutive negative months = recession = cut cycle',
            'UK Retail Sales — GDP components confirmed, spending leads services',
            'GBP/USD direction — weak GDP = dovish BoE = GBP falls',
        ],
        'what_to_watch': ['Unique: UK releases GDP monthly (not quarterly)', '3-month rolling average', 'Services vs production'],
    },

    # ───────────────────────────────────────────────────
    # 🇯🇵 JAPAN
    # ───────────────────────────────────────────────────
    {
        'id': 'jp_boj',
        'num': '07',
        'name': 'BoJ Rate Decision',
        'country': '🇯🇵 Japan',
        'tier': 1,
        'schedule': '8x/year (~6 weeks)',
        'time_utc': '~03:00',
        'impact': 'HIGH',
        'get_next': lambda y, m: _boj_next(y, m),
        'what_comes_next': [
            'USD/JPY carry trade reaction — immediate, yen strengthens on hike, carry unwinds',
            'Japan CPI (Tokyo) — released within days, validates BoJ rationale',
            'Asia equity open — Nikkei reaction sets Asia session tone for ETH',
        ],
        'what_to_watch': ['Rate decision — any hike = CRITICAL risk event', 'YCC (yield curve control) adjustments', 'USDJPY — yen strengthening = carry unwind risk', 'Forward guidance — signaling future hikes'],
        'alert': '⚠️ BOJ hike = highest-impact single event for crypto downside risk',
        'eth_historical': 'Aug 2024: BOJ hike → carry unwind → ETH -20% in 3 days',
    },
    {
        'id': 'jp_cpi_tokyo',
        'num': '08',
        'name': 'Japan CPI (Tokyo + National)',
        'country': '🇯🇵 Japan',
        'tier': 1,
        'schedule': 'Tokyo ~25-28th, National ~18-22nd',
        'time_utc': '23:30',
        'impact': 'MEDIUM',
        'get_next': lambda y, m: _approx_date(y, m, 26).replace(hour=23, minute=30),
        'what_comes_next': [
            'BoJ Rate Decision — hot CPI accelerates hike timeline',
            'USD/JPY level shift — inflation = yen strength expectation',
            'Japan Tankan Survey — quarterly, business confidence follows inflation',
        ],
        'what_to_watch': ['Ex-fresh-food (core) — BoJ target', 'Trend — rising = BOJ under pressure to hike', 'Services vs goods'],
    },
    {
        'id': 'jp_tankan',
        'num': '32',
        'name': 'Japan Tankan Survey',
        'country': '🇯🇵 Japan',
        'tier': 2,
        'schedule': 'Quarterly',
        'time_utc': '23:50',
        'impact': 'LOW',
        'get_next': lambda y, m: _approx_date(y, m, 1).replace(hour=23, minute=50) if m in (3, 6, 9, 12) else None,
        'what_comes_next': [
            'BoJ Rate Decision — Tankan is key input, weak = hold, strong = hike risk',
            'Japan GDP — released same quarter, Tankan forecasts GDP direction',
            'USD/JPY trend — strong Tankan = yen strength expectations',
        ],
        'what_to_watch': ['Large manufacturers index', 'Forward-looking outlook', 'Capex plans'],
    },

    # ───────────────────────────────────────────────────
    # 🇦🇺 AUSTRALIA
    # ───────────────────────────────────────────────────
    {
        'id': 'au_rba',
        'num': '09',
        'name': 'RBA Rate Decision',
        'country': '🇦🇺 Australia',
        'tier': 2,
        'schedule': 'Monthly (1st Tuesday)',
        'time_utc': '03:30',
        'impact': 'LOW',
        'get_next': lambda y, m: _rba_next(y, m),
        'what_comes_next': [
            'Australia CPI (quarterly) — drives next RBA meeting expectations',
            'Australia Employment — released same week, dual mandate check',
            'AUD/USD reaction — risk-on proxy, AUD up = ETH often follows',
        ],
        'what_to_watch': ['Rate decision', 'Statement tone — hawkish/dovish shift', 'China data dependency — RBA watches China closely'],
    },
    {
        'id': 'au_cpi',
        'num': '38',
        'name': 'Australia CPI (Quarterly)',
        'country': '🇦🇺 Australia',
        'tier': 2,
        'schedule': 'Quarterly',
        'time_utc': '00:30',
        'impact': 'LOW',
        'get_next': lambda y, m: _approx_date(y, m, 25).replace(hour=0, minute=30) if m in (1, 4, 7, 10) else None,
        'what_comes_next': [
            'RBA Rate Decision — CPI is primary RBA input, hot = hold, cool = cut',
            'Australia Employment — released same quarter, dual mandate both sides',
            'AUD/USD reaction — AUD is risk proxy, moves with ETH correlation',
        ],
        'what_to_watch': ['Trimmed mean watched by RBA', 'Services inflation — sticky component', 'Housing costs'],
    },

    # ───────────────────────────────────────────────────
    # 🇺🇸 UNITED STATES
    # ───────────────────────────────────────────────────
    {
        'id': 'us_ism_mfg',
        'num': '05',
        'name': 'US ISM Manufacturing PMI',
        'country': '🇺🇸 US',
        'tier': 2,
        'schedule': '1st business day',
        'time_utc': '14:00',
        'impact': 'MEDIUM',
        'get_next': lambda y, m: _first_business_day(y, m).replace(hour=14),
        'what_comes_next': [
            'US ISM Services PMI — released 2 days later, services = 80% of US GDP',
            'ADP Employment — same week, labour market follows activity',
            'Fed speeches — weak ISM prompts dovish commentary',
        ],
        'what_to_watch': ['50 = expansion line', 'New orders sub-index — leading indicator', 'Employment sub-index', 'Prices paid — inflation pipeline'],
    },
    {
        'id': 'us_ism_svc',
        'num': '06',
        'name': 'US ISM Services PMI',
        'country': '🇺🇸 US',
        'tier': 2,
        'schedule': '3rd business day',
        'time_utc': '14:00',
        'impact': 'MEDIUM',
        'get_next': lambda y, m: _nth_business_day(y, m, 3).replace(hour=14) if _nth_business_day(y, m, 3) else None,
        'what_comes_next': [
            'ADP Employment Report — released same week, activity + jobs = full picture',
            'Jobless Claims (weekly) — Thursday, continuous labour check',
            'NFP — services employment is largest NFP component',
        ],
        'what_to_watch': ['50 = expansion line', 'Services = 80% of US GDP', 'Business activity sub-index', 'New orders'],
    },
    {
        'id': 'us_nfp',
        'num': '18',
        'name': 'Non-Farm Payrolls (NFP)',
        'country': '🇺🇸 US',
        'tier': 1,
        'schedule': '1st Friday',
        'time_utc': '13:30',
        'impact': 'HIGH',
        'get_next': lambda y, m: _first_friday(y, m).replace(hour=13, minute=30),
        'what_comes_next': [
            'Fed speeches (following week) — Fed officials respond to jobs data within days',
            'Michigan Consumer Sentiment — released same day or next week, mood follows jobs',
            'US CPI (2nd week) — jobs data feeds wage inflation → CPI narrative',
        ],
        'what_to_watch': ['NFP surprise vs consensus → immediate ETH direction', 'Unemployment rate → recession signal (Sahm rule)', 'Wage growth (Average Hourly Earnings) → inflation pipeline → Fed reaction', 'Participation rate'],
        'eth_historical': 'Avg ±1.2% on release, sets tone for 1-2 weeks',
    },
    {
        'id': 'us_adp',
        'num': '16',
        'name': 'ADP Employment Report',
        'country': '🇺🇸 US',
        'tier': 2,
        'schedule': 'Wednesday before NFP',
        'time_utc': '12:15',
        'impact': 'LOW',
        'get_next': lambda y, m: (_first_friday(y, m) - timedelta(days=2)).replace(hour=12, minute=15),
        'what_comes_next': [
            'Jobless Claims — Thursday, one more data point before NFP Friday',
            'NFP — ADP is the preview, but often diverges significantly',
            'Average Hourly Earnings — released with NFP, wages matter as much as jobs',
        ],
        'what_to_watch': ['Private payrolls only', 'vs NFP consensus — directional agreement', 'Prior month revision'],
    },
    {
        'id': 'us_claims',
        'num': '17',
        'name': 'Jobless Claims (Initial + Continuing)',
        'country': '🇺🇸 US',
        'tier': 2,
        'schedule': 'Every Thursday',
        'time_utc': '13:30',
        'impact': 'MEDIUM',
        'get_next': lambda y, m: _next_thursday(),
        'what_comes_next': [
            'NFP (if Friday follows) — claims Thursday, NFP Friday, back to back',
            'Next week\'s claims — trend over 4 weeks matters more than single print',
            'JOLTS Job Openings — monthly, structural demand behind weekly flows',
        ],
        'what_to_watch': ['4-week moving average — trend matters', 'Continuing claims — exhaustion rate', 'Sahm rule trigger (unemployment 3m avg rises 0.5%+ from low)'],
    },
    {
        'id': 'us_jolts',
        'num': '19',
        'name': 'JOLTS Job Openings',
        'country': '🇺🇸 US',
        'tier': 2,
        'schedule': '~1st week',
        'time_utc': '14:00',
        'impact': 'LOW',
        'get_next': lambda y, m: _approx_date(y, m, 3).replace(hour=14),
        'what_comes_next': [
            'ADP Employment — released same week, demand + actual hiring',
            'NFP — JOLTS openings lead to actual hires in NFP',
            'Average Hourly Earnings — high openings = wage pressure builds',
        ],
        'what_to_watch': ['Quits rate — Fed watches closely', 'Openings vs unemployed ratio', 'Layoffs trend'],
    },
    {
        'id': 'us_cpi',
        'num': '13',
        'name': 'US CPI (Headline + Core)',
        'country': '🇺🇸 US',
        'tier': 1,
        'schedule': '~10th–13th',
        'time_utc': '13:30',
        'impact': 'HIGHEST',
        'get_next': lambda y, m: _approx_date(y, m, 12).replace(hour=13, minute=30),
        'what_comes_next': [
            'US PPI — released 1-2 days later, upstream cost confirmation',
            'Core PCE — released ~3 weeks later, Fed\'s actual target metric',
            'FOMC Rate Decision — CPI reprices the entire Fed rate path immediately',
        ],
        'what_to_watch': [
            'CPI vs consensus surprise — direction AND magnitude',
            'Core CPI (ex food/energy) — sticky inflation indicator',
            'Shelter/rent component — largest weight, slow-moving',
            'Market pricing: did DXY already price in the print?',
            'CURRENT REGIME determines expected move magnitude — see eth_by_regime',
        ],
        'eth_historical': {
            'COOL': 'Avg +1.06% (Fed can cut → risk-on)',
            'HOT': 'Avg -0.45% (Fed stays tight → risk-off)',
        },
        'eth_by_regime': {
            'BEAR':              {'COOL': +9.92, 'HOT': -3.33},
            'BULL':              {'COOL': +1.88, 'HOT': -0.20},
            'RECOVERY':          {'COOL': -0.55, 'HOT': +0.84},
            'ACCELERATION':      {'COOL': -0.09, 'HOT': +0.06},
            'STAGFLATION':       {'COOL': +3.00, 'HOT': -2.00},
            'STAGFLATION_HOT':   {'COOL': +4.00, 'HOT': -3.00},
        },
        'regime_sensitivity': {
            'TIGHTENING': 0.60, 'EASING': 0.50, 'CRISIS_RECOVERY': 0.80,
            'BULL': 0.85, 'BEAR': 1.20, 'RECOVERY': 0.75,
            'ACCELERATION': 0.65, 'STAGFLATION': 1.00, 'STAGFLATION_HOT': 1.10,
        },
    },
    {
        'id': 'us_ppi',
        'num': '14',
        'name': 'US PPI (Producer Price Index)',
        'country': '🇺🇸 US',
        'tier': 2,
        'schedule': '~11th–14th',
        'time_utc': '13:30',
        'impact': 'MEDIUM',
        'get_next': lambda y, m: _approx_date(y, m, 13).replace(hour=13, minute=30),
        'what_comes_next': [
            'Core PCE — PPI components feed directly into PCE calculation',
            'Retail Sales — released same week, demand side vs cost side',
            'Fed speeches — hot PPI + CPI = hawkish Fed commentary follows',
        ],
        'what_to_watch': ['Confirms or denies CPI — pipeline inflation check', 'PPI leading CPI by 2-3 months', 'Goods vs services split'],
    },
    {
        'id': 'us_retail',
        'num': '28',
        'name': 'US Retail Sales',
        'country': '🇺🇸 US',
        'tier': 2,
        'schedule': '~15th',
        'time_utc': '13:30',
        'impact': 'MEDIUM',
        'get_next': lambda y, m: _approx_date(y, m, 15).replace(hour=13, minute=30),
        'what_comes_next': [
            'PCE (Personal Spending) — released 2 weeks later, retail leads PCE',
            'GDP Advance — consumer spending is largest GDP component',
            'ISM Services PMI — retail strength feeds into services activity',
        ],
        'what_to_watch': ['Consumer spending = 70% of US GDP', 'Control group (feeds GDP)', 'Auto sales excluded (volatile)'],
    },
    {
        'id': 'us_pce',
        'num': '24',
        'name': 'Core PCE Price Index',
        'country': '🇺🇸 US',
        'tier': 1,
        'schedule': '~last Friday',
        'time_utc': '13:30',
        'impact': 'HIGH',
        'get_next': lambda y, m: _approx_date(y, m, 28).replace(hour=13, minute=30),
        'what_comes_next': [
            'FOMC Rate Decision — PCE is the Fed\'s number, hot PCE = holds/hike',
            'Personal Income + Spending — released same day, demand behind inflation',
            'Michigan Consumer Sentiment — inflation expectations sub-index confirms',
        ],
        'what_to_watch': [
            'THIS IS THE FED\'S ACTUAL TARGET — not CPI!',
            'Core PCE vs 2% target',
            'MoM vs YoY — monthly momentum matters',
            'Services ex-housing — "supercore" Fed metric',
        ],
    },
    {
        'id': 'us_fomc',
        'num': '22',
        'name': 'FOMC Rate Decision',
        'country': '🇺🇸 US',
        'tier': 1,
        'schedule': '8x/year (~6 weeks)',
        'time_utc': '19:00',
        'impact': 'HIGH',
        'get_next': lambda y, m: _fomc_next(y, m),
        'what_comes_next': [
            'Powell Press Conference — same day, tone matters more than rate itself',
            'Fed Dot Plot (if SEP meeting) — quarterly, rate path repricing = major ETH move',
            'FOMC Minutes (3 weeks later) — details behind the vote, nuance move',
        ],
        'what_to_watch': ['Rate decision vs consensus', 'Dot plot — median rate projection', 'QT taper timing — liquidity signal', 'Statement language changes'],
    },
    {
        'id': 'us_powell',
        'num': '23',
        'name': 'Powell Press Conference',
        'country': '🇺🇸 US',
        'tier': 1,
        'schedule': 'Same day as FOMC',
        'time_utc': '19:30',
        'impact': 'HIGH',
        'get_next': lambda y, m: _fomc_next(y, m).replace(hour=19, minute=30) if _fomc_next(y, m) else None,
        'what_comes_next': [
            'Fed speeches (following days) — voting members clarify/walk back Powell tone',
            'CME FedWatch repricing — immediate, futures re-price rate path live',
            '10Y Treasury yield reaction — yields move → DXY moves → ETH follows',
        ],
        'what_to_watch': ['Tone: hawkish/dovish shift vs last meeting', 'Q&A highlights — what reporters push on', 'Data dependency language'],
    },
    {
        'id': 'us_fomc_minutes',
        'num': '34',
        'name': 'FOMC Meeting Minutes',
        'country': '🇺🇸 US',
        'tier': 2,
        'schedule': '3 weeks after FOMC',
        'time_utc': '19:00',
        'impact': 'LOW',
        'get_next': lambda y, m: _fomc_next(y, m) + timedelta(weeks=3) if _fomc_next(y, m) else None,
        'what_comes_next': [
            'Fed speeches — members respond to minutes coverage',
            'Next FOMC meeting — minutes reveal what data they\'re watching',
            'CME FedWatch repricing — subtle = futures drift, not spike',
        ],
        'what_to_watch': ['Voting nuance and internal debate', 'Dissent direction', 'What data they\'re watching'],
    },
    {
        'id': 'us_gdp',
        'num': '25',
        'name': 'US GDP Advance Estimate',
        'country': '🇺🇸 US',
        'tier': 1,
        'schedule': 'Quarterly (~4th week)',
        'time_utc': '13:30',
        'impact': 'MEDIUM',
        'get_next': lambda y, m: _approx_date(y, m, 28).replace(hour=13, minute=30) if m in (1, 4, 7, 10) else None,
        'what_comes_next': [
            'GDP Second Estimate — revised ~4 weeks later, usually smaller move',
            'Corporate earnings season — GDP sets macro backdrop for earnings',
            'Fed Rate Decision — negative GDP = recession = rate cuts = ETH up',
        ],
        'what_to_watch': ['Recession threshold (2 consecutive negative quarters)', 'vs consensus', 'Consumer spending component (70% of GDP)'],
    },
    {
        'id': 'us_michigan',
        'num': '27',
        'name': 'Michigan Consumer Sentiment',
        'country': '🇺🇸 US',
        'tier': 2,
        'schedule': '2nd + 4th Friday',
        'time_utc': '14:00',
        'impact': 'LOW',
        'get_next': lambda y, m: _approx_date(y, m, 14).replace(hour=14),
        'what_comes_next': [
            'Conference Board Consumer Confidence — released following week, second sentiment read',
            'Retail Sales — sentiment leads spending by ~2-4 weeks',
            'Core PCE — 5yr inflation expectations feed directly into Fed models',
        ],
        'what_to_watch': ['Inflation expectations sub-index — Fed watches this', 'Current conditions vs expectations', '5yr inflation expectations'],
    },
    {
        'id': 'us_housing',
        'num': '36',
        'name': 'US Housing Starts + Building Permits',
        'country': '🇺🇸 US',
        'tier': 2,
        'schedule': '~17th',
        'time_utc': '13:30',
        'impact': 'LOW',
        'get_next': lambda y, m: _approx_date(y, m, 17).replace(hour=13, minute=30),
        'what_comes_next': [
            'Existing Home Sales — released same week, demand side of housing',
            'Mortgage Rate Watch (MBA) — housing reacts to rate changes with lag',
            'Fed rate path reassessment — collapsing housing = rate cut pressure',
        ],
        'what_to_watch': ['Rate sensitivity proxy', 'Permits lead starts by 1-2 months', 'Single-family vs multi-family'],
    },
    {
        'id': 'us_durables',
        'num': '37',
        'name': 'US Durable Goods Orders',
        'country': '🇺🇸 US',
        'tier': 2,
        'schedule': '~26th',
        'time_utc': '13:30',
        'impact': 'LOW',
        'get_next': lambda y, m: _approx_date(y, m, 26).replace(hour=13, minute=30),
        'what_comes_next': [
            'GDP Advance Estimate — capex feeds directly into GDP business investment',
            'ISM Manufacturing PMI — orders lead production, confirms PMI direction',
            'Factory Orders (full) — released 1 week later, broader manufacturing picture',
        ],
        'what_to_watch': ['Ex-transports (volatile aircraft orders)', 'Core capital goods orders — capex proxy', 'Business investment signal'],
    },
    {
        'id': 'us_treasury',
        'num': '29',
        'name': 'US Treasury Auction (10Y / 30Y)',
        'country': '🇺🇸 US',
        'tier': 2,
        'schedule': 'Weekly',
        'time_utc': '17:00',
        'impact': 'LOW',
        'get_next': lambda y, m: None,  # Weekly, varies
        'what_comes_next': [
            '10Y yield reaction — weak auction = yield spike = risk-off = ETH sells',
            'DXY reaction — high yield demand = DXY up = crypto headwind',
            'Next week\'s auction — series of auctions, trend in demand matters',
        ],
        'what_to_watch': ['Bid-to-cover ratio', 'Tail (vs when-issued yield)', 'Indirect bidders (foreign demand)'],
    },
]


# ══════════════════════════════════════════════════════════════
# MAIN API
# ══════════════════════════════════════════════════════════════

def get_macro_calendar(reference_time=None):
    """Get the macro calendar with upcoming events."""
    now = reference_time or datetime.now(UTC)
    year, month = now.year, now.month

    all_events = []
    for evt_def in EVENTS:
        next_dt = None
        if evt_def['get_next']:
            try:
                next_dt = evt_def['get_next'](year, month)
            except (ValueError, TypeError):
                pass
            if next_dt is None or next_dt < now:
                try:
                    nm = month + 1 if month < 12 else 1
                    ny = year if month < 12 else year + 1
                    next_dt = evt_def['get_next'](ny, nm)
                except (ValueError, TypeError):
                    pass

        entry = {
            'id': evt_def['id'],
            'num': evt_def.get('num', '—'),
            'name': evt_def['name'],
            'country': evt_def['country'],
            'tier': evt_def['tier'],
            'schedule': evt_def['schedule'],
            'time_utc': evt_def['time_utc'],
            'impact': evt_def['impact'],
            'what_comes_next': evt_def.get('what_comes_next', []),
            'what_to_watch': evt_def.get('what_to_watch', []),
            'alert': evt_def.get('alert'),
            'eth_historical': evt_def.get('eth_historical'),
            'eth_by_regime': evt_def.get('eth_by_regime'),
            'regime_sensitivity': evt_def.get('regime_sensitivity'),
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
            entry['countdown'] = 'varies'
            entry['is_next_24h'] = False
            entry['is_next_4h'] = False
            entry['is_next_1h'] = False

        all_events.append(entry)

    all_events.sort(key=lambda e: e['next_dt'] or datetime.max.replace(tzinfo=UTC))
    cutoff = now + timedelta(days=30)
    upcoming = [e for e in all_events if e['next_dt'] and e['next_dt'] < cutoff]

    # Phase detection
    day = now.day
    if day <= 3:
        phase, phase_desc, next_major = 'MONTH_START', 'PMI releases, NFP approaching', 'NFP (1st Friday)'
    elif day <= 7:
        phase, phase_desc, next_major = 'NFP_WEEK', 'NFP sets tone for the month', 'CPI/PPI (~12-14th)'
    elif day <= 14:
        phase, phase_desc, next_major = 'CPI_WEEK', 'CPI/PPI — biggest movers of the month', 'PBoC LPR (~20th)'
    elif day <= 21:
        phase, phase_desc, next_major = 'MID_MONTH', 'PBoC, ECB/BOJ, China data cluster', 'End-of-month PMIs'
    elif day <= 28:
        phase, phase_desc, next_major = 'LATE_MONTH', 'Tokyo CPI, Core PCE, PMI prep', 'Month-end PMIs → next NFP'
    else:
        phase, phase_desc, next_major = 'MONTH_END', 'PMI releases, cycle reset', 'Next month NFP'

    return {
        'now': now.isoformat(),
        'phase': phase,
        'phase_desc': phase_desc,
        'next_major': next_major,
        'events': upcoming,
        'all_events': all_events,
        'narrative_chains': NARRATIVE_CHAINS,
        'realtime_signals': REALTIME_SIGNALS,
    }


# ══════════════════════════════════════════════════════════════
# FORMATTING
# ══════════════════════════════════════════════════════════════

def _regime_impact_str(evt, current_regime):
    """Get regime-adjusted impact string for an event."""
    if not current_regime or evt['id'] != 'us_cpi':
        return ''
    regime_data = evt.get('eth_by_regime', {})
    if current_regime in regime_data:
        rd = regime_data[current_regime]
        return f'  COOL:{rd.get("COOL",0):+.1f}% HOT:{rd.get("HOT",0):+.1f}%'
    return ''


def format_macro_calendar(cal, current_regime=None):
    """Format macro calendar for terminal output."""
    lines = []
    lines.append('')
    lines.append('═' * 70)
    lines.append('  📅 MACRO CALENDAR v2 — GLOBAL DATA RELEASE TRACKER')
    lines.append('═' * 70)
    lines.append(f'\n  Now: {cal["now"]}')
    lines.append(f'  Phase: {_phase_icon(cal["phase"])} {cal["phase"]} — {cal["phase_desc"]}')
    lines.append(f'  Next major: {cal["next_major"]}')
    if current_regime:
        lines.append(f'  Regime: {current_regime}')

    # ── Next 24h ──
    next_24h = [e for e in cal['events'] if e.get('is_next_24h')]
    if next_24h:
        lines.append(f'\n  ⚡ NEXT 24 HOURS:')
        for evt in next_24h:
            icon = _impact_icon(evt['impact'])
            alert = ' 🚨' if evt.get('is_next_1h') else ''
            regime_str = _regime_impact_str(evt, current_regime)
            lines.append(f'    {icon} {evt["countdown"]:>10}  {evt["name"]:40} {evt["country"]}{alert}{regime_str}')
    else:
        lines.append(f'\n  ⚡ NEXT 24 HOURS: (none)')

    # ── Upcoming events ──
    lines.append(f'\n  📋 UPCOMING EVENTS (next 30 days):')
    lines.append(f'    {"#":>3} {"Countdown":>10}  {"Event":40} {"Country":12} {"Impact":8}')
    lines.append(f'    {"─"*3} {"─"*10}  {"─"*40} {"─"*12} {"─"*8}')

    for evt in cal['events'][:25]:
        icon = _impact_icon(evt['impact'])
        cd = evt['countdown']
        num = evt.get('num', '—')
        name = evt['name'][:38]
        country = evt['country']
        impact = evt['impact']
        alert = ' 🚨' if evt.get('is_next_4h') else ''
        regime_str = _regime_impact_str(evt, current_regime)
        lines.append(f'    {num:>3} {icon} {cd:>10}  {name:40} {country:12} {impact:8}{alert}{regime_str}')

    # ── Regime-adjusted CPI preview ──
    if current_regime:
        cpi_evt = next((e for e in cal['events'] if e['id'] == 'us_cpi'), None)
        if cpi_evt:
            regime_data = cpi_evt.get('eth_by_regime', {})
            if current_regime in regime_data:
                rd = regime_data[current_regime]
                sens = cpi_evt.get('regime_sensitivity', {}).get(current_regime, 1.0)
                lines.append(f'\n  📊 REGIME-ADJUSTED CPI EXPECTED MOVES ({current_regime}):')
                lines.append(f'    🔴 CPI COOL: {rd["COOL"]:+.2f}%  (base avg: +1.06%)')
                lines.append(f'    🔴 CPI HOT:  {rd["HOT"]:+.2f}%  (base avg: -0.45%)')
                lines.append(f'    📐 Sensitivity: {sens:.2f}x — how much macro matters in this era')

    # ── "What comes next" for next 3 events ──
    lines.append(f'\n  🔗 WHAT COMES NEXT (next 3 events):')
    for evt in cal['events'][:3]:
        lines.append(f'\n    {_impact_icon(evt["impact"])} {evt["name"]}  ({evt["countdown"]})')
        for i, nxt in enumerate(evt.get('what_comes_next', []), 1):
            lines.append(f'      → {nxt}')

    # ── Narrative Chains ──
    lines.append(f'\n  🔗 NARRATIVE CHAINS:')
    for key, chain in cal['narrative_chains'].items():
        lines.append(f'\n    {chain["name"]}')
        chain_str = ' → '.join(chain['chain'][:6])
        if len(chain['chain']) > 6:
            chain_str += ' → ...'
        lines.append(f'      {chain_str}')
        lines.append(f'      Sensitivity: {chain["eth_sensitivity"]} | {chain["note"]}')

    # ── Real-time Signals ──
    lines.append(f'\n  📡 REAL-TIME SIGNALS (watch alongside scheduled data):')
    for sig in cal['realtime_signals']:
        link_icon = {'direct': '📈', 'inverse': '📉', 'correlated': '↔️'}.get(sig['eth_link'], '•')
        lines.append(f'    {link_icon} {sig["name"]:<22} {sig["note"]}  ({sig["eth_link"]})')

    # ── Phase context ──
    lines.append(f'\n  📍 WHERE ARE WE IN THE CYCLE?')
    lines.append(f'    Phase: {_phase_icon(cal["phase"])} {cal["phase"]}')
    lines.append(f'    {cal["phase_desc"]}')

    phase_advice = {
        'MONTH_START': 'PMI data incoming — watch for China/EU demand signals before NFP',
        'NFP_WEEK': 'NFP sets the tone — wait for CPI confirmation before positioning',
        'CPI_WEEK': '⚠️ BIGGEST MOVERS — CPI/PPI are the primary ETH catalysts. Core PCE follows ~3 weeks later.',
        'MID_MONTH': 'PBoC LPR — watch for China easing signal (1-2 week ETH lead). Retail Sales same week.',
        'LATE_MONTH': 'Core PCE (Fed target!) + Tokyo CPI → BOJ risk. Germany CPI previews EU CPI.',
        'MONTH_END': 'EU CPI Flash + PMI releases → cycle resets → prepare for next NFP',
    }
    advice = phase_advice.get(cal['phase'], '')
    if advice:
        lines.append(f'    💡 {advice}')

    lines.append('\n' + '═' * 70)
    return '\n'.join(lines)


def format_macro_calendar_compact(cal, current_regime=None):
    """Compact one-line format for scanner integration."""
    lines = []
    lines.append('\n  📅 MACRO CALENDAR:')
    lines.append(f'    Phase: {_phase_icon(cal["phase"])} {cal["phase"]} — {cal["phase_desc"]}')
    for evt in cal['events'][:3]:
        icon = _impact_icon(evt['impact'])
        alert = ' 🚨' if evt.get('is_next_4h') else ''
        regime_str = _regime_impact_str(evt, current_regime)
        lines.append(f'    {icon} {evt["countdown"]:>10} → {evt["name"]} ({evt["country"]}){alert}{regime_str}')
    if cal['events']:
        first = cal['events'][0]
        nxt = first.get('what_comes_next', [])
        if nxt:
            lines.append(f'    → After {first["name"]}: {nxt[0]}')
    return '\n'.join(lines)


def calendar_to_dict(cal):
    """Convert calendar to JSON-serializable dict."""
    result = {
        'now': cal['now'],
        'phase': cal['phase'],
        'phase_desc': cal['phase_desc'],
        'next_major': cal['next_major'],
        'events': [],
        'narrative_chains': cal['narrative_chains'],
        'realtime_signals': cal['realtime_signals'],
    }
    for evt in cal['events']:
        result['events'].append({
            'id': evt['id'],
            'num': evt.get('num'),
            'name': evt['name'],
            'country': evt['country'],
            'tier': evt['tier'],
            'impact': evt['impact'],
            'countdown': evt['countdown'],
            'hours_until': evt['hours_until'],
            'time_utc': evt['time_utc'],
            'is_next_24h': evt['is_next_24h'],
            'is_next_4h': evt['is_next_4h'],
            'what_comes_next': evt.get('what_comes_next', []),
            'what_to_watch': evt.get('what_to_watch', []),
            'eth_by_regime': evt.get('eth_by_regime'),
        })
    return result


if __name__ == '__main__':
    cal = get_macro_calendar()
    print(format_macro_calendar(cal, current_regime='STAGFLATION_HOT'))

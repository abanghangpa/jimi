#!/usr/bin/env python3
"""
GENERALIZED MACRO EVENT BACKTEST — PROMPT A
=============================================

Backtest [EVENT NAME] from [START YEAR] to today using ETH/USDT 15m data.

Pipeline:
  1. Collect [EVENT] release dates and actual values (YYYY-MM-DD + value)
  2. Measure ETH returns across sessions, 24h aggregate:

     | Region             | Phase                |
     | ------------------ | -------------------- |
     | **Pre-Asia**       | Post-NY Close/Globex |
     | **Asia**           | Sydney Open          |
     |                    | Tokyo Open           |
     |                    | Asia Mid             |
     |                    | Asia Afternoon       |
     |                    | Tokyo Close          |
     |                    | Pre-London           |
     | **Europe**         | Frankfurt Open       |
     |                    | London Open          |
     |                    | London Morning       |
     |                    | London Midday        |
     | **Overlap (EU-US)**| NY Pre-Open          |
     |                    | NY Open              |
     |                    | London-NY Overlap    |
     | **New York**       | NY AM                |
     |                    | NY Lunch             |
     |                    | NY PM                |

  3. Classify each release by: Wyckoff phase (M21 proxy), vol regime (M9 proxy),
     M22 module, and the event's signal strength
  4. Cross-tabulate: (wyckoff × vol × signal) → avg 24h return, win rate, sample size
  5. Report: which combos have edge (n≥3, |avg|≥0.5%), session-by-session
     transmission chain, statistical significance (t-test miss vs beat)

Usage:
    # Run from config file
    python3 scripts/macro_backtest.py eth_15m_merged.csv config/events/ez_pmi.json

    # Generate config template
    python3 scripts/macro_backtest.py --template "US CPI" us config/events/us_cpi.json

Geographies: europe, us, asia, australia
Signal classifiers: generic, surprise, rate_decision (or custom callable)
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import math
import json
import warnings
warnings.filterwarnings('ignore')


# ═══════════════════════════════════════════════════════════════
# STATISTICAL HELPERS
# ═══════════════════════════════════════════════════════════════

def _norm_cdf(x):
    return 0.5 * (1 + math.erf(x / math.sqrt(2)))

def _ttest_1samp(x, mu=0):
    n = len(x)
    if n < 2: return 0.0, 1.0
    mean = sum(x) / n
    var = sum((xi - mean)**2 for xi in x) / (n - 1)
    se = math.sqrt(var / n)
    if se == 0: return 0.0, 1.0
    t = (mean - mu) / se
    return t, 2 * (1 - _norm_cdf(abs(t)))

def _ttest_ind(a, b):
    na, nb = len(a), len(b)
    if na < 2 or nb < 2: return 0.0, 1.0
    ma, mb = sum(a)/na, sum(b)/nb
    va = sum((x-ma)**2 for x in a)/(na-1)
    vb = sum((x-mb)**2 for x in b)/(nb-1)
    se = math.sqrt(va/na + vb/nb)
    if se == 0: return 0.0, 1.0
    t = (ma - mb) / se
    return t, 2 * (1 - _norm_cdf(abs(t)))

def _binom_test(k, n, p0=0.5):
    se = math.sqrt(n * p0 * (1 - p0))
    if se == 0: return 1.0
    z = (k - n*p0) / se
    return 2 * (1 - _norm_cdf(abs(z)))


# ═══════════════════════════════════════════════════════════════
# SESSION LIBRARY — All known global macro sessions
# ═══════════════════════════════════════════════════════════════
# Each session: (h_start, m_start, h_end, m_end, day_offset)
# day_offset: 0 = release day, 1 = next day, 2 = day after

ALL_SESSIONS = {
    # ── Pre-Asia ──
    'pre_asia':              (0,  0,  1,  0,  0),   # 00:00-01:00 UTC (Post-NY Close / Globex)
    'globex':                (0,  0,  1,  0,  0),   # alias

    # ── Asia ──
    'sydney_open':           (0,  0,  1,  0,  0),   # 00:00-01:00 UTC (08:00-09:00 MYT)
    'tokyo_open':            (1,  0,  2,  0,  0),   # 01:00-02:00 UTC (09:00-10:00 MYT)
    'asia_mid':              (2,  0,  4,  0,  0),   # 02:00-04:00 UTC (10:00-12:00 MYT)
    'asia_afternoon':        (4,  0,  6,  0,  0),   # 04:00-06:00 UTC (12:00-14:00 MYT)
    'tokyo_close':           (6,  0,  7,  0,  0),   # 06:00-07:00 UTC (14:00-15:00 MYT)
    'pre_london':            (7,  0,  8,  0,  0),   # 07:00-08:00 UTC (15:00-16:00 MYT)
    # Compound Asia sessions
    'asia_early':            (0,  0,  4,  0,  0),   # 00:00-04:00 UTC
    'asia_full':             (0,  0,  8,  0,  0),   # 00:00-08:00 UTC (entire Asia)

    # ── Europe ──
    'frankfurt_open':        (7,  0,  8,  0,  0),   # 07:00-08:00 UTC (15:00-16:00 MYT)
    'london_open':           (8,  0,  9,  0,  0),   # 08:00-09:00 UTC (16:00-17:00 MYT)
    'london_morning':        (9,  0, 12,  0,  0),   # 09:00-12:00 UTC (17:00-20:00 MYT)
    'london_midday':        (12,  0, 13, 30,  0),   # 12:00-13:30 UTC (20:00-21:30 MYT)
    # Compound Europe sessions
    'europe_core':           (8,  0, 16, 30,  0),   # 08:00-16:30 UTC (full London day)
    'europe_morning':        (7,  0, 12,  0,  0),   # 07:00-12:00 UTC

    # ── Overlap (EU–US) ──
    'ny_pre_open':          (12,  0, 13, 30,  0),   # 12:00-13:30 UTC (20:00-21:30 MYT)
    'ny_open':              (13, 30, 14, 30,  0),   # 13:30-14:30 UTC (21:30-22:30 MYT)
    'london_ny_overlap':    (13, 30, 16, 30,  0),   # 13:30-16:30 UTC (21:30-00:30 MYT)
    # Compound overlap
    'overlap_full':         (12,  0, 16, 30,  0),   # 12:00-16:30 UTC

    # ── New York ──
    'ny_am':                (14, 30, 17,  0,  0),   # 14:30-17:00 UTC (22:30-01:00 MYT)
    'ny_lunch':             (17,  0, 18,  0,  0),   # 17:00-18:00 UTC (01:00-02:00 MYT)
    'ny_pm':                (18,  0, 21,  0,  0),   # 18:00-21:00 UTC (02:00-05:00 MYT)
    # Compound NY sessions
    'us_open':              (13, 30, 17,  0,  0),   # 13:30-17:00 UTC
    'us_afternoon':         (17,  0, 21,  0,  0),   # 17:00-21:00 UTC
    'us_full':              (13, 30, 21,  0,  0),   # 13:30-21:00 UTC

    # ── Next Day ──
    'next_pre_asia':        (0,  0,  1,  0,  1),
    'next_sydney_open':     (0,  0,  1,  0,  1),
    'next_tokyo_open':      (1,  0,  2,  0,  1),
    'next_asia_mid':        (2,  0,  4,  0,  1),
    'next_asia_afternoon':  (4,  0,  6,  0,  1),
    'next_tokyo_close':     (6,  0,  7,  0,  1),
    'next_pre_london':      (7,  0,  8,  0,  1),
    'next_frankfurt_open':  (7,  0,  8,  0,  1),
    'next_london_open':     (8,  0,  9,  0,  1),
    'next_london_morning':  (9,  0, 12,  0,  1),
    'next_london_midday':  (12,  0, 13, 30,  1),
    'next_ny_pre_open':    (12,  0, 13, 30,  1),
    'next_ny_open':        (13, 30, 14, 30,  1),
    'next_london_ny_overlap': (13, 30, 16, 30, 1),
    'next_ny_am':          (14, 30, 17,  0,  1),
    'next_ny_lunch':       (17,  0, 18,  0,  1),
    'next_ny_pm':          (18,  0, 21,  0,  1),
    'next_europe_core':     (8,  0, 16, 30,  1),
    'next_us_open':        (13, 30, 17,  0,  1),
    'next_us_afternoon':   (17,  0, 21,  0,  1),

    # ── Day +2 ──
    'd2_tokyo_open':        (1,  0,  2,  0,  2),
    'd2_london_open':       (8,  0,  9,  0,  2),
    'd2_ny_open':          (13, 30, 14, 30,  2),
}


# ═══════════════════════════════════════════════════════════════
# GEOGRAPHY → SESSION ITINERARY MAPPER
# ═══════════════════════════════════════════════════════════════
# Given where the event releases, auto-generate the logical session chain.

GEOGRAPHY_ITINERARIES = {
    'europe': {
        'description': 'Europe release → granular EU/US/Asia chain until EU reopen',
        'sessions': [
            'frankfurt_open', 'london_open', 'london_morning', 'london_midday',
            'ny_pre_open', 'ny_open', 'london_ny_overlap',
            'ny_am', 'ny_lunch', 'ny_pm',
            'next_pre_asia', 'next_tokyo_open', 'next_asia_mid',
            'next_asia_afternoon', 'next_tokyo_close',
            'next_frankfurt_open', 'next_london_open',
        ],
        'transitions': [
            ('frankfurt_open',      'london_open',          'Frankfurt → London Open'),
            ('london_open',         'london_morning',       'London Open → London AM'),
            ('london_morning',      'london_midday',        'London AM → London Midday'),
            ('london_midday',       'ny_pre_open',          'London Midday → NY Pre-Open'),
            ('ny_pre_open',         'ny_open',              'NY Pre-Open → NY Open'),
            ('ny_open',             'london_ny_overlap',    'NY Open → London-NY Overlap'),
            ('london_ny_overlap',   'ny_am',                'Overlap → NY AM'),
            ('ny_am',               'ny_lunch',             'NY AM → NY Lunch'),
            ('ny_lunch',            'ny_pm',                'NY Lunch → NY PM'),
            ('ny_pm',               'next_pre_asia',        'NY PM → Pre-Asia'),
            ('next_pre_asia',       'next_tokyo_open',      'Pre-Asia → Tokyo Open'),
            ('next_tokyo_open',     'next_asia_mid',        'Tokyo Open → Asia Mid'),
            ('next_asia_mid',       'next_asia_afternoon',  'Asia Mid → Asia Afternoon'),
            ('next_asia_afternoon', 'next_tokyo_close',     'Asia Afternoon → Tokyo Close'),
            ('next_tokyo_close',    'next_frankfurt_open',  'Tokyo Close → Frankfurt Reopen'),
            ('next_frankfurt_open', 'next_london_open',     'Frankfurt → London Reopen'),
        ],
        'full_cycle': ('frankfurt_open', 'next_london_open'),
    },
    'us': {
        'description': 'US release → granular US/Asia/EU chain until US reopen',
        'sessions': [
            'ny_open', 'london_ny_overlap', 'ny_am', 'ny_lunch', 'ny_pm',
            'next_pre_asia', 'next_tokyo_open', 'next_asia_mid',
            'next_asia_afternoon', 'next_tokyo_close',
            'next_frankfurt_open', 'next_london_open', 'next_london_morning',
            'next_london_midday', 'next_ny_pre_open', 'next_ny_open',
        ],
        'transitions': [
            ('ny_open',             'london_ny_overlap',    'NY Open → London-NY Overlap'),
            ('london_ny_overlap',   'ny_am',                'Overlap → NY AM'),
            ('ny_am',               'ny_lunch',             'NY AM → NY Lunch'),
            ('ny_lunch',            'ny_pm',                'NY Lunch → NY PM'),
            ('ny_pm',               'next_pre_asia',        'NY PM → Pre-Asia'),
            ('next_pre_asia',       'next_tokyo_open',      'Pre-Asia → Tokyo Open'),
            ('next_tokyo_open',     'next_asia_mid',        'Tokyo Open → Asia Mid'),
            ('next_asia_mid',       'next_asia_afternoon',  'Asia Mid → Asia Afternoon'),
            ('next_asia_afternoon', 'next_tokyo_close',     'Asia Afternoon → Tokyo Close'),
            ('next_tokyo_close',    'next_frankfurt_open',  'Tokyo Close → Frankfurt Open'),
            ('next_frankfurt_open', 'next_london_open',     'Frankfurt → London Open'),
            ('next_london_open',    'next_london_morning',  'London Open → London AM'),
            ('next_london_morning', 'next_london_midday',   'London AM → London Midday'),
            ('next_london_midday',  'next_ny_pre_open',     'London Midday → NY Pre-Open'),
            ('next_ny_pre_open',    'next_ny_open',         'NY Pre-Open → NY Reopen'),
        ],
        'full_cycle': ('ny_open', 'next_ny_open'),
    },
    'asia': {
        'description': 'Asia release → granular Asia/EU/US chain until Asia reopen',
        'sessions': [
            'tokyo_open', 'asia_mid', 'asia_afternoon', 'tokyo_close',
            'pre_london', 'frankfurt_open', 'london_open', 'london_morning',
            'london_midday', 'ny_pre_open', 'ny_open', 'london_ny_overlap',
            'ny_am', 'ny_lunch', 'ny_pm',
            'next_pre_asia', 'next_tokyo_open',
        ],
        'transitions': [
            ('tokyo_open',          'asia_mid',             'Tokyo Open → Asia Mid'),
            ('asia_mid',            'asia_afternoon',       'Asia Mid → Asia Afternoon'),
            ('asia_afternoon',      'tokyo_close',          'Asia Afternoon → Tokyo Close'),
            ('tokyo_close',         'pre_london',           'Tokyo Close → Pre-London'),
            ('pre_london',          'frankfurt_open',       'Pre-London → Frankfurt Open'),
            ('frankfurt_open',      'london_open',          'Frankfurt → London Open'),
            ('london_open',         'london_morning',       'London Open → London AM'),
            ('london_morning',      'london_midday',        'London AM → London Midday'),
            ('london_midday',       'ny_pre_open',          'London Midday → NY Pre-Open'),
            ('ny_pre_open',         'ny_open',              'NY Pre-Open → NY Open'),
            ('ny_open',             'london_ny_overlap',    'NY Open → London-NY Overlap'),
            ('london_ny_overlap',   'ny_am',                'Overlap → NY AM'),
            ('ny_am',               'ny_lunch',             'NY AM → NY Lunch'),
            ('ny_lunch',            'ny_pm',                'NY Lunch → NY PM'),
            ('ny_pm',               'next_pre_asia',        'NY PM → Pre-Asia'),
            ('next_pre_asia',       'next_tokyo_open',      'Pre-Asia → Tokyo Reopen'),
        ],
        'full_cycle': ('tokyo_open', 'next_tokyo_open'),
    },
    'australia': {
        'description': 'AU release → granular chain until AU reopen',
        'sessions': [
            'sydney_open', 'tokyo_open', 'asia_mid', 'asia_afternoon',
            'tokyo_close', 'frankfurt_open', 'london_open', 'london_morning',
            'london_midday', 'ny_pre_open', 'ny_open', 'ny_am', 'ny_lunch', 'ny_pm',
            'next_pre_asia', 'next_sydney_open',
        ],
        'transitions': [
            ('sydney_open',         'tokyo_open',           'Sydney → Tokyo Open'),
            ('tokyo_open',          'asia_mid',             'Tokyo Open → Asia Mid'),
            ('asia_mid',            'asia_afternoon',       'Asia Mid → Asia Afternoon'),
            ('asia_afternoon',      'tokyo_close',          'Asia Afternoon → Tokyo Close'),
            ('tokyo_close',         'frankfurt_open',       'Tokyo Close → Frankfurt Open'),
            ('frankfurt_open',      'london_open',          'Frankfurt → London Open'),
            ('london_open',         'london_morning',       'London Open → London AM'),
            ('london_morning',      'london_midday',        'London AM → London Midday'),
            ('london_midday',       'ny_pre_open',          'London Midday → NY Pre-Open'),
            ('ny_pre_open',         'ny_open',              'NY Pre-Open → NY Open'),
            ('ny_open',             'ny_am',                'NY Open → NY AM'),
            ('ny_am',               'ny_lunch',             'NY AM → NY Lunch'),
            ('ny_lunch',            'ny_pm',                'NY Lunch → NY PM'),
            ('ny_pm',               'next_pre_asia',        'NY PM → Pre-Asia'),
            ('next_pre_asia',       'next_sydney_open',     'Pre-Asia → Sydney Reopen'),
        ],
        'full_cycle': ('sydney_open', 'next_sydney_open'),
    },
}


# ═══════════════════════════════════════════════════════════════
# SIGNAL CLASSIFIERS — Pluggable per event type
# ═══════════════════════════════════════════════════════════════

def classify_generic(release, thresholds=None):
    """Generic signal classifier. Works with any event that has actual/consensus.

    Thresholds dict:
        expansion: [strong_above, mild_above]  (default [52, 50])
        surprise:  [strong_beat, mild_beat]     (default [+1.0, 0])
    """
    if thresholds is None:
        thresholds = {}

    exp_strong = thresholds.get('expansion_strong', 52.0)
    exp_mild = thresholds.get('expansion_mild', 50.0)
    surp_strong = thresholds.get('surprise_strong', 1.0)
    surp_mild = thresholds.get('surprise_mild', 0.0)

    actual = release.get('actual', release.get('composite', 0))
    consensus = release.get('consensus', release.get('prior', actual))
    surprise = actual - consensus

    if actual >= exp_strong and surprise >= surp_strong:
        return 'STRONG_BEAT'
    elif actual >= exp_mild and surprise >= surp_mild:
        return 'MILD_BEAT'
    elif actual >= exp_mild and surprise < surp_mild:
        return 'MILD_MISS'
    elif actual < exp_mild and surprise >= surp_mild:
        return 'WEAK_BEAT'
    else:
        return 'STRONG_MISS'


def classify_surprise(release, thresholds=None):
    """Pure surprise-based classifier (for events like NFP, CPI where level matters less)."""
    if thresholds is None:
        thresholds = {}

    big = thresholds.get('big', 1.0)
    small = thresholds.get('small', 0.0)

    actual = release.get('actual', release.get('composite', 0))
    consensus = release.get('consensus', release.get('prior', actual))
    surprise = actual - consensus

    if surprise >= big:
        return 'BIG_BEAT'
    elif surprise >= small:
        return 'SMALL_BEAT'
    elif surprise >= -big:
        return 'SMALL_MISS'
    else:
        return 'BIG_MISS'


def classify_rate_decision(release, thresholds=None):
    """For central bank rate decisions. Actual = rate, consensus = expected rate."""
    if thresholds is None:
        thresholds = {}

    actual = release.get('actual', 0)
    consensus = release.get('consensus', actual)
    diff = actual - consensus

    if diff > 0:
        return 'HIKE_SURPRISE'
    elif diff < 0:
        return 'CUT_SURPRISE'
    else:
        # Check dovish/hawkish from statement
        bias = release.get('bias', 'neutral')
        if bias == 'dovish':
            return 'DOVISH_HOLD'
        elif bias == 'hawkish':
            return 'HAWKISH_HOLD'
        return 'AS_EXPECTED'


# ═══════════════════════════════════════════════════════════════
# WYCKOFF PHASE CLASSIFIER (simplified, from ETH structure)
# ═══════════════════════════════════════════════════════════════

def classify_wyckoff(df, release_date):
    """Simplified Wyckoff phase from 4H structure."""
    from scipy import stats as sp_stats
    lookback_start = pd.Timestamp(release_date) - timedelta(days=20)
    mask = (df.index >= lookback_start) & (df.index < pd.Timestamp(release_date))
    recent = df[mask]
    if len(recent) < 500:
        return 'UNKNOWN'
    h4 = recent.resample('4h').agg({'Open': 'first', 'High': 'max', 'Low': 'min', 'Close': 'last', 'Volume': 'sum'}).dropna()
    if len(h4) < 48:
        return 'UNKNOWN'
    closes = h4['Close'].values
    highs = h4['High'].values
    lows = h4['Low'].values
    x = np.arange(len(closes))
    slope, _, r_val, _, _ = sp_stats.linregress(x[-48:], closes[-48:])
    trend_strength = r_val ** 2
    recent_high = np.max(highs[-48:])
    recent_low = np.min(lows[-48:])
    range_pct = (recent_high - recent_low) / recent_low * 100
    current = closes[-1]
    range_pos = (current - recent_low) / (recent_high - recent_low) if recent_high != recent_low else 0.5
    slope_pct = slope / closes[-1] * 100
    if trend_strength > 0.3 and slope_pct > 0.02:
        return 'MARKUP'
    elif trend_strength > 0.3 and slope_pct < -0.02:
        return 'MARKDOWN'
    elif range_pct < 8 and trend_strength < 0.15:
        if range_pos > 0.7:
            return 'DISTRIBUTION'
        elif range_pos < 0.3:
            return 'ACCUMULATION'
        else:
            return 'RANGE'
    else:
        return 'RANGE'


def classify_wyckoff_simple(df, release_date):
    """Fallback Wyckoff classifier (no scipy)."""
    lookback_start = pd.Timestamp(release_date) - timedelta(days=20)
    mask = (df.index >= lookback_start) & (df.index < pd.Timestamp(release_date))
    recent = df[mask]
    if len(recent) < 200:
        return 'UNKNOWN'
    h4 = recent.resample('4h').agg({'Open': 'first', 'High': 'max', 'Low': 'min', 'Close': 'last', 'Volume': 'sum'}).dropna()
    if len(h4) < 48:
        return 'UNKNOWN'
    closes = h4['Close'].values
    highs = h4['High'].values
    lows = h4['Low'].values
    recent_high = np.max(highs[-48:])
    recent_low = np.min(lows[-48:])
    range_pct = (recent_high - recent_low) / recent_low * 100
    current = closes[-1]
    range_pos = (current - recent_low) / (recent_high - recent_low) if recent_high != recent_low else 0.5
    ema12 = np.convolve(closes, np.ones(12)/12, mode='valid')[-1] if len(closes) >= 12 else closes[-1]
    ema26 = np.convolve(closes, np.ones(26)/26, mode='valid')[-1] if len(closes) >= 26 else closes[-1]
    trend_dir = 'UP' if ema12 > ema26 else 'DOWN'
    if range_pct < 8:
        if range_pos > 0.7:
            return 'DISTRIBUTION'
        elif range_pos < 0.3:
            return 'ACCUMULATION'
        else:
            return 'RANGE'
    elif trend_dir == 'UP':
        return 'MARKUP'
    else:
        return 'MARKDOWN'


# ═══════════════════════════════════════════════════════════════
# VOL REGIME CLASSIFIER (simplified, from ETH volatility)
# ═══════════════════════════════════════════════════════════════

def classify_vol(df, release_date):
    """Simplified vol regime from 15m ATR + Bollinger width."""
    lookback_start = pd.Timestamp(release_date) - timedelta(days=5)
    mask = (df.index >= lookback_start) & (df.index < pd.Timestamp(release_date))
    recent = df[mask]
    if len(recent) < 100:
        return 'UNKNOWN'
    high = recent['High'].values.astype(float)
    low = recent['Low'].values.astype(float)
    close = recent['Close'].values.astype(float)
    tr = np.maximum(high[1:] - low[1:], np.maximum(np.abs(high[1:] - close[:-1]), np.abs(low[1:] - close[:-1])))
    atr_14 = np.convolve(tr, np.ones(14)/14, mode='valid')
    if len(atr_14) < 2:
        return 'UNKNOWN'
    current_atr = atr_14[-1]
    atr_p50 = np.percentile(atr_14, 50)
    atr_p90 = np.percentile(atr_14, 90)
    if len(close) >= 20:
        sma20 = np.convolve(close, np.ones(20)/20, mode='valid')
        std20 = np.array([np.std(close[i:i+20]) for i in range(len(close)-19)])
        bb_width = (2 * std20) / sma20 * 100
        current_bb = bb_width[-1] if len(bb_width) > 0 else 0
        bb_p30 = np.percentile(bb_width, 30) if len(bb_width) > 0 else 0
    else:
        current_bb = 0
        bb_p30 = 0
    returns = np.diff(close) / close[:-1] * 100
    pos_ratio = np.sum(returns > 0) / len(returns) if len(returns) > 0 else 0.5
    direction_consistency = abs(pos_ratio - 0.5) * 2
    if current_atr > atr_p90:
        return 'CRISIS'
    elif current_bb < bb_p30:
        return 'COMPRESSING'
    elif direction_consistency > 0.3 and current_atr > atr_p50:
        return 'TRENDING'
    elif direction_consistency < 0.15 and current_atr < atr_p50:
        return 'LOW_VOL'
    else:
        return 'CHOP'


# ═══════════════════════════════════════════════════════════════
# M22 PROXY: INFLATION REGIME (from price action context)
# ═══════════════════════════════════════════════════════════════

def classify_macro_regime(df, release_date):
    """Simplified macro regime from 30d price action.

    Not a true M22 replacement — M22 uses PPI/CPI/Fed data.
    This is a structural proxy: trend + volatility over 30d.
    """
    lookback_start = pd.Timestamp(release_date) - timedelta(days=30)
    mask = (df.index >= lookback_start) & (df.index < pd.Timestamp(release_date))
    recent = df[mask]
    if len(recent) < 1000:
        return 'UNKNOWN'
    closes = recent['Close'].values.astype(float)
    # 30d return
    ret_30d = (closes[-1] - closes[0]) / closes[0] * 100
    # 30d volatility (annualized daily vol proxy)
    daily = recent.resample('1D')['Close'].last().dropna()
    if len(daily) < 10:
        return 'UNKNOWN'
    daily_ret = daily.pct_change().dropna()
    vol_30d = daily_ret.std() * np.sqrt(252) * 100
    # Classify
    if ret_30d > 10 and vol_30d < 80:
        return 'GOLDILOCKS'     # strong up, low vol
    elif ret_30d > 5:
        return 'REFLATION'      # trending up
    elif ret_30d < -10 and vol_30d > 100:
        return 'CRISIS'         # crash + high vol
    elif ret_30d < -5:
        return 'DEFLATION'      # trending down
    elif vol_30d > 80:
        return 'VOLATILE'       # high vol, no direction
    else:
        return 'NEUTRAL'


# Registry of classifiers
SIGNAL_CLASSIFIERS = {
    'generic': classify_generic,
    'surprise': classify_surprise,
    'rate_decision': classify_rate_decision,
}


# ═══════════════════════════════════════════════════════════════
# CORE ENGINE
# ═══════════════════════════════════════════════════════════════

def load_data(csv_path):
    """Load 15m OHLCV data."""
    df = pd.read_csv(csv_path)
    df['Open time'] = pd.to_datetime(df['Open time'])
    df = df.set_index('Open time')
    for col in ['Open', 'High', 'Low', 'Close', 'Volume']:
        df[col] = df[col].astype(float)
    return df


def get_session_return(df, release_date, session_tuple):
    """Calculate return for a session window.

    Args:
        df: 15m OHLCV DataFrame (UTC indexed)
        release_date: str 'YYYY-MM-DD' or Timestamp
        session_tuple: (h_start, m_start, h_end, m_end, day_offset)
    """
    h_start, m_start, h_end, m_end, day_offset = session_tuple

    start = pd.Timestamp(release_date) + timedelta(days=day_offset, hours=h_start, minutes=m_start)
    end = pd.Timestamp(release_date) + timedelta(days=day_offset, hours=h_end, minutes=m_end)

    mask = (df.index >= start) & (df.index <= end)
    window = df[mask]

    if len(window) < 2:
        return None

    open_price = window.iloc[0]['Open']
    close_price = window.iloc[-1]['Close']
    return (close_price - open_price) / open_price * 100


def get_full_cycle_return(df, release_date, start_session, end_session):
    """Calculate return from release time through full cycle (typically 24h)."""
    s = ALL_SESSIONS[start_session]  # (h_start, m_start, h_end, m_end, day_offset)
    e = ALL_SESSIONS[end_session]

    start = pd.Timestamp(release_date) + timedelta(days=s[4], hours=s[0], minutes=s[1])
    end = pd.Timestamp(release_date) + timedelta(days=e[4], hours=e[2], minutes=e[3])

    mask = (df.index >= start) & (df.index <= end)
    window = df[mask]

    if len(window) < 2:
        return None

    open_price = window.iloc[0]['Open']
    close_price = window.iloc[-1]['Close']
    return (close_price - open_price) / open_price * 100


def direction(val):
    if val is None or pd.isna(val):
        return None
    return 1 if val > 0 else (-1 if val < 0 else 0)


def analyze_transition(data, from_col, to_col, label, indent=""):
    """Analyze direction persistence between two sessions. Returns result dict."""
    mask = data[from_col].notna() & data[to_col].notna()
    subset = data[mask]
    if len(subset) < 5:
        return None

    d_from = subset[from_col].apply(direction)
    d_to = subset[to_col].apply(direction)
    valid = (d_from != 0) & (d_to != 0)
    total = valid.sum()

    if total < 5:
        return None

    same = (d_from[valid] == d_to[valid]).sum()
    pct = same / total * 100
    p_val = _binom_test(same, total)
    corr = subset[[from_col, to_col]].corr().iloc[0, 1]

    same_mask = (d_from == d_to) & valid
    opp_mask = (d_from != d_to) & valid
    same_avg = subset.loc[same_mask, to_col].mean() if same_mask.sum() > 0 else 0
    opp_avg = subset.loc[opp_mask, to_col].mean() if opp_mask.sum() > 0 else 0

    if pct > 65:
        verdict = "✅ REAL"
    elif pct > 55:
        verdict = "⚠️  MARG"
    else:
        verdict = "❌ BROKEN"

    sig = "*" if p_val < 0.05 else " "
    print(f"{indent}{label:<42} {pct:>5.1f}%  n={total:<3}  p={p_val:.4f}{sig}  r={corr:+.3f}  {verdict}")
    print(f"{indent}{'':42} same→{same_avg:+.2f}%  opp→{opp_avg:+.2f}%")

    return {
        'label': label, 'pct': pct, 'n': total, 'p_val': p_val,
        'corr': corr, 'verdict': verdict, 'same_avg': same_avg, 'opp_avg': opp_avg,
    }


def run_backtest(df, releases, config):
    """Run the full generalized backtest.

    Args:
        df: 15m OHLCV DataFrame (UTC indexed)
        releases: dict {date_str: {actual/prior/consensus/...}}
        config: dict with:
            name: str
            geography: 'europe'|'us'|'asia'|'australia' (or custom)
            signal_classifier: 'generic'|'surprise'|'rate_decision' (or callable)
            signal_thresholds: dict (optional)
            custom_sessions: dict (optional, override auto-generated)
            custom_transitions: list (optional, override auto-generated)
    """
    event_name = config.get('name', 'Unknown Event')
    geography = config.get('geography', 'europe')
    classifier_name = config.get('signal_classifier', 'generic')
    thresholds = config.get('signal_thresholds', {})

    # Get classifier
    if callable(classifier_name):
        classifier = classifier_name
    else:
        classifier = SIGNAL_CLASSIFIERS.get(classifier_name, classify_generic)

    # Get itinerary
    if geography in GEOGRAPHY_ITINERARIES:
        itinerary = GEOGRAPHY_ITINERARIES[geography]
    else:
        raise ValueError(f"Unknown geography '{geography}'. Use: {list(GEOGRAPHY_ITINERARIES.keys())} or provide custom_transitions")

    # Allow custom overrides
    session_names = config.get('custom_sessions', itinerary['sessions'])
    transitions = config.get('custom_transitions', itinerary['transitions'])
    full_cycle = config.get('full_cycle', itinerary.get('full_cycle'))

    print("=" * 80)
    print(f"GENERALIZED MACRO EVENT BACKTEST: {event_name}")
    print(f"Geography: {geography} | Classifier: {classifier_name}")
    print(f"Itinerary: {itinerary.get('description', 'custom')}")
    print("=" * 80)

    # ── Build session returns ──
    rows = []
    for date_str, release_data in sorted(releases.items()):
        release_date = pd.Timestamp(date_str)
        if release_date > df.index[-1] or release_date < df.index[0]:
            continue

        # Classify signal
        signal = classifier(release_data, thresholds)

        # Classify Wyckoff phase
        try:
            wyckoff = classify_wyckoff(df, date_str)
        except Exception:
            try:
                wyckoff = classify_wyckoff_simple(df, date_str)
            except Exception:
                wyckoff = 'UNKNOWN'

        # Classify vol regime
        try:
            vol = classify_vol(df, date_str)
        except Exception:
            vol = 'UNKNOWN'

        # Classify macro regime (M22 proxy)
        try:
            macro = classify_macro_regime(df, date_str)
        except Exception:
            macro = 'UNKNOWN'

        # Get surprise
        actual = release_data.get('actual', release_data.get('composite', 0))
        consensus = release_data.get('consensus', release_data.get('prior', actual))
        surprise = actual - consensus

        row = {
            'date': date_str,
            'actual': actual,
            'consensus': consensus,
            'surprise': surprise,
            'signal': signal,
            'wyckoff': wyckoff,
            'vol': vol,
            'macro': macro,
        }

        # Session returns
        for sname in session_names:
            if sname in ALL_SESSIONS:
                row[sname] = get_session_return(df, date_str, ALL_SESSIONS[sname])

        # Full cycle
        if full_cycle:
            start_s, end_s = full_cycle
            if start_s in ALL_SESSIONS and end_s in ALL_SESSIONS:
                row['full_cycle'] = get_full_cycle_return(df, date_str, start_s, end_s)

        rows.append(row)

    data = pd.DataFrame(rows)

    if len(data) == 0:
        print("\n  ❌ No valid releases found in dataset range.")
        return data

    print(f"\n  Releases in dataset: {len(data)}")
    print(f"  Date range: {data['date'].iloc[0]} to {data['date'].iloc[-1]}")

    # ═══════════════════════════════════════════════════════════
    # 1. FULL CHAIN — ALL RELEASES
    # ═══════════════════════════════════════════════════════════
    print("\n" + "=" * 80)
    print("1. TRANSMISSION CHAIN — ALL RELEASES")
    print("=" * 80)

    chain_results = []
    for from_s, to_s, label in transitions:
        r = analyze_transition(data, from_s, to_s, label)
        if r:
            chain_results.append(r)

    # Visual chain
    print("\n  Chain visualization:")
    for cr in chain_results:
        arrow = "━━━→" if cr['pct'] > 55 else "╌╌╌↛"
        bar_len = int(cr['pct'] / 2)
        bar = "█" * bar_len + "░" * (50 - bar_len)
        print(f"    {cr['label'][:25]:<25} {bar} {cr['pct']:.1f}%")

    # ═══════════════════════════════════════════════════════════
    # 2. CHAIN BY SIGNAL TYPE
    # ═══════════════════════════════════════════════════════════
    print("\n" + "=" * 80)
    print("2. CHAIN BY SIGNAL TYPE")
    print("=" * 80)

    for sig in sorted(data['signal'].unique()):
        sig_data = data[data['signal'] == sig]
        if len(sig_data) < 5:
            continue

        print(f"\n  Signal: {sig} (n={len(sig_data)})")
        for from_s, to_s, label in transitions:
            analyze_transition(sig_data, from_s, to_s, label, indent="    ")

    # ═══════════════════════════════════════════════════════════
    # 3. CHAIN BY SURPRISE MAGNITUDE
    # ═══════════════════════════════════════════════════════════
    print("\n" + "=" * 80)
    print("3. CHAIN BY SURPRISE MAGNITUDE")
    print("=" * 80)

    big_thresh = thresholds.get('big_surprise', 1.0)
    surprise_bins = [
        (f'Big Miss (<-{big_thresh})',    data[data['surprise'] < -big_thresh]),
        (f'Small Miss (-{big_thresh} to 0)', data[(data['surprise'] >= -big_thresh) & (data['surprise'] < 0)]),
        (f'Small Beat (0 to +{big_thresh})', data[(data['surprise'] >= 0) & (data['surprise'] <= big_thresh)]),
        (f'Big Beat (>+{big_thresh})',     data[data['surprise'] > big_thresh]),
    ]

    for label, subset in surprise_bins:
        if len(subset) < 5:
            continue
        print(f"\n  {label} (n={len(subset)})")
        for from_s, to_s, t_label in transitions:
            analyze_transition(subset, from_s, to_s, t_label, indent="    ")

    # ═══════════════════════════════════════════════════════════
    # 4. CROSS-TABULATION: Wyckoff × Vol × Signal
    # ═══════════════════════════════════════════════════════════
    print("\n" + "=" * 80)
    print("4. CROSS-TABULATION: Wyckoff × Vol × Signal → 24h Return")
    print("=" * 80)

    fc_col = 'full_cycle' if 'full_cycle' in data.columns else session_names[-1]
    combo_data = data.dropna(subset=[fc_col])

    if len(combo_data) >= 10:
        combo_stats = combo_data.groupby(['wyckoff', 'vol', 'signal']).agg(
            avg_24h=(fc_col, 'mean'),
            win_rate=(fc_col, lambda x: (x > 0).sum() / len(x) * 100),
            sample_size=(fc_col, 'count'),
            median_24h=(fc_col, 'median'),
        ).reset_index()
        combo_stats = combo_stats[combo_stats['sample_size'] >= 2].sort_values('avg_24h', ascending=False)

        print(f"\n  {'Wyckoff':<14} {'Vol':<12} {'Signal':<18} {'Avg%':>8} {'Win%':>7} {'N':>4} {'Med%':>8}")
        print("  " + "-" * 75)
        for _, row in combo_stats.iterrows():
            edge = ""
            if abs(row['avg_24h']) >= 0.5 and row['sample_size'] >= 3:
                edge = " ⚡"
            elif abs(row['avg_24h']) >= 1.0:
                edge = " 🔥"
            print(f"  {row['wyckoff']:<14} {row['vol']:<12} {row['signal']:<18} "
                  f"{row['avg_24h']:>+7.2f}% {row['win_rate']:>6.1f}% {row['sample_size']:>4.0f} "
                  f"{row['median_24h']:>+7.2f}%{edge}")

        # Actionable combos
        edges = combo_stats[(combo_stats['sample_size'] >= 3) & (combo_stats['avg_24h'].abs() >= 0.5)]
        if len(edges) > 0:
            print(f"\n  Actionable combos (n≥3, |avg|≥0.5%):")
            for _, row in edges.iterrows():
                direction = "LONG" if row['avg_24h'] > 0 else "SHORT"
                print(f"    {direction}: {row['wyckoff']} + {row['vol']} + {row['signal']}  "
                      f"→ {row['avg_24h']:+.2f}% (win {row['win_rate']:.0f}%, n={row['sample_size']:.0f})")
    else:
        print("\n  Insufficient data for cross-tabulation")

    # ═══════════════════════════════════════════════════════════
    # 5. MACRO REGIME × SIGNAL
    # ═══════════════════════════════════════════════════════════
    print("\n" + "=" * 80)
    print("5. MACRO REGIME × SIGNAL → 24h Return")
    print("=" * 80)

    if len(combo_data) >= 10:
        macro_stats = combo_data.groupby(['macro', 'signal']).agg(
            avg_24h=(fc_col, 'mean'),
            win_rate=(fc_col, lambda x: (x > 0).sum() / len(x) * 100),
            n=(fc_col, 'count'),
        ).reset_index()
        macro_stats = macro_stats[macro_stats['n'] >= 2].sort_values('avg_24h', ascending=False)

        print(f"\n  {'Macro':<16} {'Signal':<18} {'Avg%':>8} {'Win%':>7} {'N':>4}")
        print("  " + "-" * 55)
        for _, row in macro_stats.iterrows():
            edge = " ⚡" if abs(row['avg_24h']) >= 0.5 and row['n'] >= 3 else ""
            print(f"  {row['macro']:<16} {row['signal']:<18} "
                  f"{row['avg_24h']:>+7.2f}% {row['win_rate']:>6.1f}% {row['n']:>4.0f}{edge}")

    # ═══════════════════════════════════════════════════════════
    # 6. STATISTICAL SIGNIFICANCE
    # ═══════════════════════════════════════════════════════════
    print("\n" + "=" * 80)
    print("4. STATISTICAL SIGNIFICANCE")
    print("=" * 80)

    fc_col = 'full_cycle' if 'full_cycle' in data.columns else session_names[-1]
    fc_returns = data[fc_col].dropna().tolist()

    if len(fc_returns) >= 3:
        t1, p1 = _ttest_1samp(fc_returns, 0)
        mean_fc = sum(fc_returns) / len(fc_returns)
        se_fc = math.sqrt(sum((x-mean_fc)**2 for x in fc_returns)/(len(fc_returns)-1)) / math.sqrt(len(fc_returns))
        print(f"\n  Full-cycle mean: {mean_fc:+.3f}% ± {se_fc:.3f}%  t={t1:+.3f}  p={p1:.4f}  {'✅' if p1<0.05 else '❌'}")

    beat = data[data['surprise'] >= 0][fc_col].dropna().tolist()
    miss = data[data['surprise'] < 0][fc_col].dropna().tolist()
    if len(beat) >= 3 and len(miss) >= 3:
        t2, p2 = _ttest_ind(beat, miss)
        print(f"  Beat vs Miss: beat={sum(beat)/len(beat):+.3f}% (n={len(beat)})  miss={sum(miss)/len(miss):+.3f}% (n={len(miss)})  t={t2:+.3f}  p={p2:.4f}  {'✅' if p2<0.05 else '❌'}")

    # First session direction → full cycle
    first_session = transitions[0][0] if transitions else session_names[0]
    fc_pos = data[data[first_session] > 0][fc_col].dropna().tolist()
    fc_neg = data[data[first_session] < 0][fc_col].dropna().tolist()
    if len(fc_pos) >= 3 and len(fc_neg) >= 3:
        t3, p3 = _ttest_ind(fc_pos, fc_neg)
        print(f"  {first_session} pos vs neg: pos={sum(fc_pos)/len(fc_pos):+.3f}% (n={len(fc_pos)})  neg={sum(fc_neg)/len(fc_neg):+.3f}% (n={len(fc_neg)})  t={t3:+.3f}  p={p3:.4f}  {'✅' if p3<0.05 else '❌'}")

    # ═══════════════════════════════════════════════════════════
    # 7. SESSION RETURN PROFILE
    # ═══════════════════════════════════════════════════════════
    print("\n" + "=" * 80)
    print("7. SESSION RETURN PROFILE (by signal)")
    print("=" * 80)

    for sig in sorted(data['signal'].unique()):
        sig_data = data[data['signal'] == sig]
        if len(sig_data) < 3:
            continue

        print(f"\n  {sig} (n={len(sig_data)})")
        for sname in session_names:
            sdata = sig_data[sname].dropna()
            if len(sdata) > 0:
                avg = sdata.mean()
                win = (sdata > 0).sum() / len(sdata) * 100
                print(f"    {sname:<24} {avg:>+6.2f}%  win {win:.0f}%  n={len(sdata)}")

    # ═══════════════════════════════════════════════════════════
    # 8. SUMMARY
    # ═══════════════════════════════════════════════════════════
    print("\n" + "=" * 80)
    print("8. SUMMARY")
    print("=" * 80)

    print(f"\n  Event: {event_name}")
    print(f"  Geography: {geography}")
    print(f"  Releases analyzed: {len(data)}")
    print()

    for cr in chain_results:
        emoji = "✅" if cr['pct'] > 65 else ("⚠️" if cr['pct'] > 55 else "❌")
        print(f"    {emoji} {cr['label']}")
        print(f"       {cr['pct']:.1f}% same direction (n={cr['n']}, p={cr['p_val']:.4f})")

    # Find break point
    print("\n  Chain integrity:")
    chain_alive = True
    for cr in chain_results:
        if cr['pct'] <= 55 and chain_alive:
            print(f"    ⛓️‍💥 Chain BREAKS at: {cr['label']}")
            chain_alive = False
        elif cr['pct'] > 55 and chain_alive:
            print(f"    🔗 {cr['label']} — holds")
        else:
            print(f"    ⏸️  {cr['label']} — after break")

    # Actionable summary
    real_links = [cr for cr in chain_results if cr['pct'] > 65]
    marg_links = [cr for cr in chain_results if 55 < cr['pct'] <= 65]

    print(f"\n  Real edges: {len(real_links)}  |  Marginal: {len(marg_links)}  |  Broken: {len(chain_results) - len(real_links) - len(marg_links)}")

    if len(real_links) >= 2:
        entry = real_links[0]
        exit_link = real_links[-1]
        print(f"\n  ⚡ Tradeable window: {entry['label'].split('→')[0].strip()} → {exit_link['label'].split('→')[1].strip()}")

    return data


# ═══════════════════════════════════════════════════════════════
# CONVENIENCE: Run from config file
# ═══════════════════════════════════════════════════════════════

def run_from_config(csv_path, config_path):
    """Run backtest from a JSON config file.

    Config format:
    {
        "name": "Eurozone Flash PMI",
        "geography": "europe",
        "signal_classifier": "generic",
        "signal_thresholds": {"expansion_strong": 52, "expansion_mild": 50},
        "releases": {
            "2024-01-24": {"actual": 47.9, "prior": 47.6, "consensus": 48.0},
            ...
        }
    }
    """
    with open(config_path) as f:
        config = json.load(f)

    releases = config.pop('releases', {})
    df = load_data(csv_path)
    return run_backtest(df, releases, config)


# ═══════════════════════════════════════════════════════════════
# CONVENIENCE: Generate config template
# ═══════════════════════════════════════════════════════════════

def generate_config_template(event_name, geography, output_path=None):
    """Generate a JSON config template for a new event."""
    template = {
        "name": event_name,
        "geography": geography,
        "signal_classifier": "generic",
        "signal_thresholds": {
            "expansion_strong": 52.0,
            "expansion_mild": 50.0,
            "surprise_strong": 1.0,
            "surprise_mild": 0.0,
            "big_surprise": 1.0,
        },
        "releases": {
            "YYYY-MM-DD": {
                "actual": 0.0,
                "prior": 0.0,
                "consensus": 0.0,
            }
        }
    }

    if output_path:
        with open(output_path, 'w') as f:
            json.dump(template, f, indent=2)
        print(f"  Template saved to {output_path}")

    return template


if __name__ == '__main__':
    import sys

    if len(sys.argv) < 3:
        print("Usage:")
        print("  python macro_backtest.py <csv_path> <config.json>")
        print("  python macro_backtest.py --template <event_name> <geography> [output.json]")
        print()
        print("Geographies: europe, us, asia, australia")
        print("Signal classifiers: generic, surprise, rate_decision")
        sys.exit(1)

    if sys.argv[1] == '--template':
        name = sys.argv[2] if len(sys.argv) > 2 else 'My Event'
        geo = sys.argv[3] if len(sys.argv) > 3 else 'europe'
        out = sys.argv[4] if len(sys.argv) > 4 else None
        generate_config_template(name, geo, out)
    else:
        csv_path = sys.argv[1]
        config_path = sys.argv[2]
        run_from_config(csv_path, config_path)

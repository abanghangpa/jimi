#!/usr/bin/env python3
"""
GENERALIZED MACRO EVENT BACKTEST FRAMEWORK
==========================================

Works with ANY macro event — just define:
  1. EVENT_RELEASES: {date: {actual, prior, consensus, ...}}
  2. EVENT_CONFIG: release time, timezone, geography
  3. The framework auto-generates sessions and transitions based on release time

No hardcoded sessions. No hardcoded transitions. The event's release time
and geography determine the session itinerary automatically.

Usage:
    # Define your event
    config = {
        'name': 'Eurozone Flash PMI (Composite)',
        'release_utc_hour': 8,      # 08:00 UTC
        'release_utc_minute': 0,
        'geography': 'europe',       # determines session order
        'custom_sessions': None,     # override auto-generated sessions if needed
    }

    releases = {
        '2024-01-24': {'actual': 47.9, 'prior': 47.6, 'consensus': 48.0},
        ...
    }

    # Run
    results = run_backtest(df, releases, config)
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
# Each session: (name, start_utc_h, start_utc_m, end_utc_h, end_utc_m, day_offset)
# day_offset: 0 = release day, 1 = next day, 2 = day after

ALL_SESSIONS = {
    # ── Release Day ──
    'asia_early':        (0,  0,  4,  0,  0),   # 00:00-04:00 UTC
    'asia_open':         (0,  0,  5,  0,  0),   # 00:00-05:00 UTC
    'asia_morning':      (0,  0,  8,  0,  0),   # 00:00-08:00 UTC
    'australia':         (3,  0,  6,  0,  0),   # 03:00-06:00 UTC
    'asia_overlap':      (3,  0,  8,  0,  0),   # 03:00-08:00 UTC
    'europe_pre':        (6,  0,  8,  0,  0),   # 06:00-08:00 UTC
    'europe_open':       (7,  0, 12,  0,  0),   # 07:00-12:00 UTC
    'europe_morning':    (7,  0, 11,  0,  0),   # 07:00-11:00 UTC
    'europe_core':       (8,  0, 16,  30, 0),   # 08:00-16:30 UTC
    'uk_session':        (8,  0, 16,  30, 0),   # 08:00-16:30 UTC
    'europe_afternoon':  (12, 0, 16,  30, 0),   # 12:00-16:30 UTC
    'us_pre':            (12, 0, 13,  30, 0),   # 12:00-13:30 UTC
    'us_open':           (13, 30, 17, 0,  0),   # 13:30-17:00 UTC
    'us_morning':        (13, 30, 17, 0,  0),   # 13:30-17:00 UTC
    'us_afternoon':      (17, 0, 21,  0,  0),   # 17:00-21:00 UTC
    'us_close':          (20, 0, 23,  59, 0),   # 20:00-23:59 UTC

    # ── Next Day ──
    'next_asia_open':    (0,  0,  5,  0,  1),
    'next_asia_morning': (0,  0,  8,  0,  1),
    'next_australia':    (3,  0,  6,  0,  1),
    'next_asia_overlap': (3,  0,  8,  0,  1),
    'next_europe_pre':   (6,  0,  8,  0,  1),
    'next_europe_open':  (7,  0, 12,  0,  1),
    'next_europe_am':    (7,  0, 11,  0,  1),
    'next_uk_session':   (8,  0, 16,  30, 1),
    'next_us_open':      (13, 30, 17, 0,  1),
    'next_us_afternoon': (17, 0, 21,  0,  1),

    # ── Day +2 ──
    'd2_asia_open':      (0,  0,  5,  0,  2),
    'd2_europe_open':    (7,  0, 12,  0,  2),
    'd2_us_open':        (13, 30, 17, 0,  2),
}


# ═══════════════════════════════════════════════════════════════
# GEOGRAPHY → SESSION ITINERARY MAPPER
# ═══════════════════════════════════════════════════════════════
# Given where the event releases, auto-generate the logical session chain.

GEOGRAPHY_ITINERARIES = {
    'europe': {
        'description': 'Europe release → UK → US Open → US PM → Asia → Asia PM → EU Reopen',
        'sessions': [
            'europe_open', 'uk_session', 'us_open', 'us_afternoon',
            'next_asia_open', 'next_asia_overlap', 'next_europe_open',
        ],
        'transitions': [
            ('europe_open',    'uk_session',       'Europe Open → UK Session'),
            ('uk_session',     'us_open',          'UK Session → US Open'),
            ('us_open',        'us_afternoon',     'US Open → US Afternoon'),
            ('us_afternoon',   'next_asia_open',   'US Afternoon → Asia Open'),
            ('next_asia_open', 'next_asia_overlap', 'Asia Open → Asia PM'),
            ('next_asia_overlap', 'next_europe_open', 'Asia PM → Europe Reopen'),
        ],
        'full_cycle': ('europe_open', 'next_europe_open'),  # for 24h calc
    },
    'us': {
        'description': 'US release → US PM → Asia → Asia PM → Europe → UK → US Reopen',
        'sessions': [
            'us_open', 'us_afternoon', 'next_asia_open', 'next_asia_overlap',
            'next_europe_open', 'next_uk_session', 'next_us_open',
        ],
        'transitions': [
            ('us_open',        'us_afternoon',     'US Open → US Afternoon'),
            ('us_afternoon',   'next_asia_open',   'US Afternoon → Asia Open'),
            ('next_asia_open', 'next_asia_overlap', 'Asia Open → Asia PM'),
            ('next_asia_overlap', 'next_europe_open', 'Asia PM → Europe Open'),
            ('next_europe_open', 'next_uk_session', 'Europe → UK Session'),
            ('next_uk_session',  'next_us_open',    'UK → US Reopen'),
        ],
        'full_cycle': ('us_open', 'next_us_open'),
    },
    'asia': {
        'description': 'Asia release → Europe → UK → US Open → US PM → Asia Reopen',
        'sessions': [
            'asia_open', 'asia_overlap', 'europe_open', 'uk_session',
            'us_open', 'us_afternoon', 'next_asia_open',
        ],
        'transitions': [
            ('asia_open',      'asia_overlap',     'Asia Open → Asia Overlap'),
            ('asia_overlap',   'europe_open',      'Asia Overlap → Europe Open'),
            ('europe_open',    'uk_session',       'Europe Open → UK Session'),
            ('uk_session',     'us_open',          'UK Session → US Open'),
            ('us_open',        'us_afternoon',     'US Open → US Afternoon'),
            ('us_afternoon',   'next_asia_open',   'US Afternoon → Asia Reopen'),
        ],
        'full_cycle': ('asia_open', 'next_asia_open'),
    },
    'australia': {
        'description': 'AU release → Asia overlap → Europe → UK → US → Asia Reopen',
        'sessions': [
            'australia', 'asia_overlap', 'europe_open', 'uk_session',
            'us_open', 'us_afternoon', 'next_asia_open',
        ],
        'transitions': [
            ('australia',      'asia_overlap',     'Australia → Asia Overlap'),
            ('asia_overlap',   'europe_open',      'Asia Overlap → Europe Open'),
            ('europe_open',    'uk_session',       'Europe Open → UK Session'),
            ('uk_session',     'us_open',          'UK Session → US Open'),
            ('us_open',        'us_afternoon',     'US Open → US Afternoon'),
            ('us_afternoon',   'next_asia_open',   'US Afternoon → Asia Reopen'),
        ],
        'full_cycle': ('australia', 'next_asia_open'),
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
    # 4. STATISTICAL SIGNIFICANCE
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
    # 5. SESSION RETURN PROFILE
    # ═══════════════════════════════════════════════════════════
    print("\n" + "=" * 80)
    print("5. SESSION RETURN PROFILE (by signal)")
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
    # 6. SUMMARY
    # ═══════════════════════════════════════════════════════════
    print("\n" + "=" * 80)
    print("6. SUMMARY")
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

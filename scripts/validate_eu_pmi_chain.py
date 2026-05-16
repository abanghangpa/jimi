#!/usr/bin/env python3
"""
Prompt C: Validate Eurozone Flash PMI Session Transmission Chain

Tests whether EZ PMI causes a session-by-session transmission chain in ETH.
Measures direction persistence between sessions on release days.

Thresholds:
  <55% same direction → chain doesn't hold
  >65% same direction → real edge
  55-65% → marginal

Also tests 24h aggregate return for statistical significance.
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import math
import warnings
warnings.filterwarnings('ignore')

# ═══════════════════════════════════════════════════════════════
# STATISTICAL HELPERS
# ═══════════════════════════════════════════════════════════════

def norm_cdf(x):
    return 0.5 * (1 + math.erf(x / math.sqrt(2)))

def ttest_1samp(x, mu=0):
    n = len(x)
    if n < 2:
        return 0.0, 1.0
    mean = sum(x) / n
    var = sum((xi - mean) ** 2 for xi in x) / (n - 1)
    se = math.sqrt(var / n)
    if se == 0:
        return 0.0, 1.0
    t = (mean - mu) / se
    p = 2 * (1 - norm_cdf(abs(t)))
    return t, p

def ttest_ind(a, b):
    na, nb = len(a), len(b)
    if na < 2 or nb < 2:
        return 0.0, 1.0
    ma = sum(a) / na
    mb = sum(b) / nb
    va = sum((x - ma) ** 2 for x in a) / (na - 1)
    vb = sum((x - mb) ** 2 for x in b) / (nb - 1)
    se = math.sqrt(va / na + vb / nb)
    if se == 0:
        return 0.0, 1.0
    t = (ma - mb) / se
    p = 2 * (1 - norm_cdf(abs(t)))
    return t, p

def binomial_test(k, n, p0=0.5):
    """Two-sided binomial test: is observed proportion k/n different from p0?"""
    # Normal approximation for large n
    expected = n * p0
    se = math.sqrt(n * p0 * (1 - p0))
    if se == 0:
        return 1.0
    z = (k - expected) / se
    p = 2 * (1 - norm_cdf(abs(z)))
    return p


# ═══════════════════════════════════════════════════════════════
# EZ PMI RELEASES (same as backtest)
# ═══════════════════════════════════════════════════════════════

EZ_PMI_RELEASES = {
    '2018-01-05': {'composite': 58.1, 'prior': 58.0, 'consensus': 57.9},
    '2018-01-24': {'composite': 58.6, 'prior': 58.0, 'consensus': 57.9},
    '2018-02-22': {'composite': 57.5, 'prior': 58.6, 'consensus': 58.4},
    '2018-03-22': {'composite': 55.3, 'prior': 57.1, 'consensus': 56.8},
    '2018-04-23': {'composite': 55.2, 'prior': 55.2, 'consensus': 54.8},
    '2018-05-23': {'composite': 54.1, 'prior': 55.1, 'consensus': 55.0},
    '2018-06-22': {'composite': 54.8, 'prior': 54.1, 'consensus': 53.9},
    '2018-07-24': {'composite': 54.3, 'prior': 54.9, 'consensus': 54.7},
    '2018-08-23': {'composite': 54.4, 'prior': 54.3, 'consensus': 54.5},
    '2018-09-21': {'composite': 54.2, 'prior': 54.4, 'consensus': 54.3},
    '2018-10-24': {'composite': 52.7, 'prior': 54.1, 'consensus': 53.9},
    '2018-11-22': {'composite': 52.4, 'prior': 52.7, 'consensus': 52.8},
    '2018-12-14': {'composite': 51.4, 'prior': 52.4, 'consensus': 52.0},
    '2019-01-04': {'composite': 51.1, 'prior': 51.4, 'consensus': 51.3},
    '2019-01-24': {'composite': 50.7, 'prior': 51.1, 'consensus': 51.0},
    '2019-02-21': {'composite': 51.4, 'prior': 50.7, 'consensus': 51.1},
    '2019-03-22': {'composite': 51.3, 'prior': 51.4, 'consensus': 51.4},
    '2019-04-18': {'composite': 51.6, 'prior': 51.3, 'consensus': 51.5},
    '2019-05-23': {'composite': 51.6, 'prior': 51.6, 'consensus': 51.5},
    '2019-06-21': {'composite': 52.1, 'prior': 51.6, 'consensus': 51.8},
    '2019-07-24': {'composite': 51.5, 'prior': 52.1, 'consensus': 51.8},
    '2019-08-22': {'composite': 51.8, 'prior': 51.5, 'consensus': 51.2},
    '2019-09-23': {'composite': 50.4, 'prior': 51.8, 'consensus': 51.6},
    '2019-10-24': {'composite': 50.2, 'prior': 50.4, 'consensus': 50.3},
    '2019-11-22': {'composite': 50.6, 'prior': 50.2, 'consensus': 50.3},
    '2019-12-16': {'composite': 50.6, 'prior': 50.6, 'consensus': 50.7},
    '2020-01-06': {'composite': 50.9, 'prior': 50.6, 'consensus': 50.7},
    '2020-01-24': {'composite': 51.0, 'prior': 50.9, 'consensus': 50.8},
    '2020-02-21': {'composite': 51.6, 'prior': 51.0, 'consensus': 51.0},
    '2020-03-24': {'composite': 31.4, 'prior': 51.6, 'consensus': 38.8},
    '2020-04-23': {'composite': 13.5, 'prior': 31.4, 'consensus': 18.0},
    '2020-05-21': {'composite': 30.5, 'prior': 13.5, 'consensus': 25.0},
    '2020-06-23': {'composite': 47.5, 'prior': 30.5, 'consensus': 41.0},
    '2020-07-24': {'composite': 54.8, 'prior': 47.5, 'consensus': 50.0},
    '2020-08-21': {'composite': 51.6, 'prior': 54.9, 'consensus': 54.5},
    '2020-09-23': {'composite': 50.1, 'prior': 51.9, 'consensus': 51.7},
    '2020-10-23': {'composite': 49.4, 'prior': 50.4, 'consensus': 49.5},
    '2020-11-23': {'composite': 45.3, 'prior': 50.0, 'consensus': 46.0},
    '2020-12-16': {'composite': 49.8, 'prior': 45.3, 'consensus': 45.8},
    '2021-01-22': {'composite': 47.5, 'prior': 49.1, 'consensus': 47.6},
    '2021-02-19': {'composite': 48.1, 'prior': 47.8, 'consensus': 48.0},
    '2021-03-24': {'composite': 52.5, 'prior': 48.8, 'consensus': 49.0},
    '2021-04-23': {'composite': 53.8, 'prior': 53.2, 'consensus': 53.0},
    '2021-05-21': {'composite': 56.9, 'prior': 53.8, 'consensus': 55.0},
    '2021-06-23': {'composite': 59.2, 'prior': 57.1, 'consensus': 58.5},
    '2021-07-23': {'composite': 60.6, 'prior': 59.5, 'consensus': 60.0},
    '2021-08-23': {'composite': 59.5, 'prior': 60.2, 'consensus': 59.8},
    '2021-09-23': {'composite': 56.1, 'prior': 59.0, 'consensus': 58.5},
    '2021-10-22': {'composite': 54.3, 'prior': 56.2, 'consensus': 55.5},
    '2021-11-23': {'composite': 55.8, 'prior': 54.2, 'consensus': 53.0},
    '2021-12-16': {'composite': 53.4, 'prior': 55.4, 'consensus': 54.0},
    '2022-01-24': {'composite': 52.4, 'prior': 53.3, 'consensus': 52.6},
    '2022-02-21': {'composite': 55.8, 'prior': 52.3, 'consensus': 52.7},
    '2022-03-24': {'composite': 54.5, 'prior': 55.5, 'consensus': 54.0},
    '2022-04-22': {'composite': 55.8, 'prior': 54.9, 'consensus': 54.5},
    '2022-05-23': {'composite': 54.8, 'prior': 55.5, 'consensus': 55.0},
    '2022-06-23': {'composite': 51.9, 'prior': 54.8, 'consensus': 54.0},
    '2022-07-22': {'composite': 49.4, 'prior': 52.0, 'consensus': 51.0},
    '2022-08-23': {'composite': 49.2, 'prior': 49.9, 'consensus': 49.0},
    '2022-09-23': {'composite': 48.2, 'prior': 48.9, 'consensus': 48.5},
    '2022-10-24': {'composite': 47.1, 'prior': 48.1, 'consensus': 47.5},
    '2022-11-23': {'composite': 47.8, 'prior': 47.3, 'consensus': 47.0},
    '2022-12-16': {'composite': 48.8, 'prior': 47.8, 'consensus': 48.0},
    '2023-01-24': {'composite': 49.3, 'prior': 48.8, 'consensus': 49.0},
    '2023-02-21': {'composite': 52.3, 'prior': 49.3, 'consensus': 50.5},
    '2023-03-24': {'composite': 54.1, 'prior': 52.0, 'consensus': 52.0},
    '2023-04-21': {'composite': 54.4, 'prior': 53.7, 'consensus': 53.5},
    '2023-05-23': {'composite': 53.3, 'prior': 54.1, 'consensus': 54.0},
    '2023-06-23': {'composite': 50.3, 'prior': 52.8, 'consensus': 52.5},
    '2023-07-24': {'composite': 48.9, 'prior': 50.3, 'consensus': 50.0},
    '2023-08-23': {'composite': 47.0, 'prior': 48.6, 'consensus': 48.5},
    '2023-09-22': {'composite': 47.1, 'prior': 46.7, 'consensus': 46.5},
    '2023-10-24': {'composite': 46.5, 'prior': 47.2, 'consensus': 47.0},
    '2023-11-23': {'composite': 47.1, 'prior': 46.5, 'consensus': 46.8},
    '2023-12-15': {'composite': 47.0, 'prior': 47.6, 'consensus': 47.0},
    '2024-01-24': {'composite': 47.9, 'prior': 47.6, 'consensus': 48.0},
    '2024-02-22': {'composite': 46.1, 'prior': 47.9, 'consensus': 48.5},
    '2024-03-21': {'composite': 49.9, 'prior': 46.5, 'consensus': 47.0},
    '2024-04-23': {'composite': 51.4, 'prior': 50.3, 'consensus': 50.5},
    '2024-05-23': {'composite': 52.3, 'prior': 51.7, 'consensus': 51.5},
    '2024-06-21': {'composite': 50.8, 'prior': 52.2, 'consensus': 52.5},
    '2024-07-24': {'composite': 50.1, 'prior': 50.9, 'consensus': 51.0},
    '2024-08-22': {'composite': 51.2, 'prior': 50.2, 'consensus': 50.5},
    '2024-09-23': {'composite': 48.9, 'prior': 51.0, 'consensus': 50.5},
    '2024-10-24': {'composite': 49.7, 'prior': 49.6, 'consensus': 49.5},
    '2024-11-22': {'composite': 48.1, 'prior': 50.0, 'consensus': 49.5},
    '2024-12-16': {'composite': 47.3, 'prior': 48.3, 'consensus': 48.0},
    '2025-01-24': {'composite': 50.2, 'prior': 48.0, 'consensus': 48.5},
    '2025-02-21': {'composite': 50.5, 'prior': 50.2, 'consensus': 50.0},
    '2025-03-24': {'composite': 50.4, 'prior': 50.6, 'consensus': 50.8},
    '2025-04-23': {'composite': 50.1, 'prior': 50.9, 'consensus': 50.5},
    '2025-05-22': {'composite': 49.5, 'prior': 50.4, 'consensus': 50.5},
    '2025-06-23': {'composite': 50.8, 'prior': 49.8, 'consensus': 50.0},
    '2025-07-24': {'composite': 51.0, 'prior': 50.6, 'consensus': 50.5},
    '2025-08-22': {'composite': 51.1, 'prior': 50.9, 'consensus': 50.8},
    '2025-09-23': {'composite': 50.5, 'prior': 51.0, 'consensus': 51.0},
    '2025-10-24': {'composite': 50.0, 'prior': 50.6, 'consensus': 50.5},
    '2025-11-21': {'composite': 49.8, 'prior': 50.0, 'consensus': 50.2},
    '2025-12-16': {'composite': 49.5, 'prior': 49.7, 'consensus': 49.8},
    '2026-01-23': {'composite': 50.2, 'prior': 49.6, 'consensus': 49.8},
    '2026-02-20': {'composite': 50.6, 'prior': 50.2, 'consensus': 50.0},
    '2026-03-24': {'composite': 50.1, 'prior': 50.4, 'consensus': 50.5},
    '2026-04-23': {'composite': 49.8, 'prior': 50.2, 'consensus': 50.0},
}


# ═══════════════════════════════════════════════════════════════
# SESSION DEFINITIONS (UTC)
# ═══════════════════════════════════════════════════════════════

# Your itinerary: Europe Open → UK → US Open → US PM → Asia Reopen
SESSIONS = {
    'Europe_Open':    (7, 0, 11, 0),      # 07:00-11:00 UTC
    'UK_Session':     (7, 0, 16, 0),      # 07:00-16:00 UTC (overlaps Europe)
    'US_Open':        (13, 30, 17, 0),    # 13:30-17:00 UTC
    'US_Afternoon':   (17, 0, 21, 0),     # 17:00-21:00 UTC
    'Asia_Reopen':    (1, 0, 5, 0),       # NEXT DAY 01:00-05:00 UTC
    'Full_24h':       (8, 0, 8, 0),       # 08:00 UTC to next day 08:00 UTC
}


def load_data(csv_path):
    df = pd.read_csv(csv_path)
    df['Open time'] = pd.to_datetime(df['Open time'])
    df = df.set_index('Open time')
    for col in ['Open', 'High', 'Low', 'Close', 'Volume']:
        df[col] = df[col].astype(float)
    return df


def get_session_return(df, release_date, session_name, session_def):
    h_start, m_start, h_end, m_end = session_def

    if session_name == 'Asia_Reopen':
        start = pd.Timestamp(release_date) + timedelta(days=1, hours=h_start, minutes=m_start)
        end = pd.Timestamp(release_date) + timedelta(days=1, hours=h_end, minutes=m_end)
    elif session_name == 'Full_24h':
        start = pd.Timestamp(release_date) + timedelta(hours=h_start)
        end = start + timedelta(hours=24)
    else:
        start = pd.Timestamp(release_date) + timedelta(hours=h_start, minutes=m_start)
        end = pd.Timestamp(release_date) + timedelta(hours=h_end, minutes=m_end)

    mask = (df.index >= start) & (df.index <= end)
    session_df = df[mask]

    if len(session_df) < 2:
        return None

    open_price = session_df.iloc[0]['Open']
    close_price = session_df.iloc[-1]['Close']
    return (close_price - open_price) / open_price * 100


def get_session_volatility(df, release_date, session_name, session_def):
    """Get intrabar volatility (high-low range) for a session."""
    h_start, m_start, h_end, m_end = session_def

    if session_name == 'Asia_Reopen':
        start = pd.Timestamp(release_date) + timedelta(days=1, hours=h_start, minutes=m_start)
        end = pd.Timestamp(release_date) + timedelta(days=1, hours=h_end, minutes=m_end)
    elif session_name == 'Full_24h':
        start = pd.Timestamp(release_date) + timedelta(hours=h_start)
        end = start + timedelta(hours=24)
    else:
        start = pd.Timestamp(release_date) + timedelta(hours=h_start, minutes=m_start)
        end = pd.Timestamp(release_date) + timedelta(hours=h_end, minutes=m_end)

    mask = (df.index >= start) & (df.index <= end)
    session_df = df[mask]

    if len(session_df) < 2:
        return None

    # Average true range per bar
    ranges = (session_df['High'] - session_df['Low']).astype(float)
    return ranges.mean()


def classify_signal(pmi_data):
    composite = pmi_data['composite']
    consensus = pmi_data.get('consensus', pmi_data['prior'])
    surprise = composite - consensus

    if composite >= 52.0 and surprise >= 0.5:
        return 'STRONG_EXPANSION'
    elif composite >= 50.0 and surprise >= 0:
        return 'MILD_EXPANSION'
    elif composite >= 50.0 and surprise < 0:
        return 'WEAK_EXPANSION'
    elif composite < 50.0 and surprise >= 0:
        return 'MILD_CONTRACTION'
    else:
        return 'STRONG_CONTRACTION'


def direction(val):
    """Return +1 for positive, -1 for negative, 0 for flat."""
    if val is None or pd.isna(val):
        return None
    return 1 if val > 0 else (-1 if val < 0 else 0)


def run_validation(csv_path):
    print("=" * 80)
    print("PROMPT C: EUROZONE FLASH PMI SESSION TRANSMISSION CHAIN VALIDATION")
    print("=" * 80)

    df = load_data(csv_path)

    # Build session returns for all releases
    rows = []
    for date_str, pmi_data in sorted(EZ_PMI_RELEASES.items()):
        release_date = pd.Timestamp(date_str)
        if release_date > df.index[-1] or release_date < df.index[0]:
            continue

        row = {
            'date': date_str,
            'composite': pmi_data['composite'],
            'consensus': pmi_data.get('consensus', pmi_data['prior']),
            'surprise': pmi_data['composite'] - pmi_data.get('consensus', pmi_data['prior']),
            'signal': classify_signal(pmi_data),
        }

        for sname, sdef in SESSIONS.items():
            ret = get_session_return(df, date_str, sname, sdef)
            vol = get_session_volatility(df, date_str, sname, sdef)
            row[sname] = ret
            row[f'{sname}_vol'] = vol

        rows.append(row)

    data = pd.DataFrame(rows)

    # ═══════════════════════════════════════════════════════════
    # 1. DIRECTION PERSISTENCE — FULL DATASET
    # ═══════════════════════════════════════════════════════════
    print("\n" + "=" * 80)
    print("1. DIRECTION PERSISTENCE (All Releases)")
    print("=" * 80)

    transitions = [
        ('Europe_Open', 'UK_Session',       'Europe Open → UK Session'),
        ('UK_Session',  'US_Open',          'UK Session → US Open'),
        ('US_Open',     'US_Afternoon',     'US Open → US Afternoon'),
        ('US_Afternoon','Asia_Reopen',      'US Afternoon → Asia Reopen'),
        # Also test skip-session chains
        ('Europe_Open', 'US_Open',          'Europe Open → US Open (skip)'),
        ('Europe_Open', 'Asia_Reopen',      'Europe Open → Asia Reopen (skip)'),
        ('UK_Session',  'Asia_Reopen',      'UK Session → Asia Reopen (skip)'),
    ]

    chain_results = []

    for from_s, to_s, label in transitions:
        mask = data[from_s].notna() & data[to_s].notna()
        subset = data[mask]

        if len(subset) < 5:
            print(f"\n  {label:<45} insufficient data (n={len(subset)})")
            continue

        d_from = subset[from_s].apply(direction)
        d_to = subset[to_s].apply(direction)

        # Same direction (excluding flat)
        valid_mask = (d_from != 0) & (d_to != 0)
        valid_from = d_from[valid_mask]
        valid_to = d_to[valid_mask]

        if len(valid_from) < 5:
            continue

        same = (valid_from == valid_to).sum()
        total = len(valid_from)
        pct = same / total * 100

        # Binomial test: is this significantly different from 50%?
        p_val = binomial_test(same, total, 0.5)

        # Correlation of returns
        corr = subset[[from_s, to_s]].corr().iloc[0, 1]

        # Average return on same-direction vs opposite-direction days
        same_mask = (d_from == d_to) & valid_mask
        opp_mask = (d_from != d_to) & (d_from != 0) & (d_to != 0)

        same_avg = subset.loc[same_mask, to_s].mean() if same_mask.sum() > 0 else None
        opp_avg = subset.loc[opp_mask, to_s].mean() if opp_mask.sum() > 0 else None

        # Verdict
        if pct > 65:
            verdict = "✅ REAL EDGE"
        elif pct > 55:
            verdict = "⚠️  MARGINAL"
        else:
            verdict = "❌ NO CHAIN"

        chain_results.append({
            'label': label,
            'pct': pct,
            'n': total,
            'p_val': p_val,
            'corr': corr,
            'verdict': verdict,
            'same_avg': same_avg,
            'opp_avg': opp_avg,
        })

        sig_marker = "*" if p_val < 0.05 else " "

        print(f"\n  {label}")
        print(f"    Same direction: {pct:.1f}% (n={total})  p={p_val:.4f} {sig_marker}")
        print(f"    Return corr:    {corr:+.3f}")
        if same_avg is not None:
            print(f"    Same-dir avg:   {same_avg:+.2f}%  |  Opp-dir avg: {opp_avg:+.2f}%")
        print(f"    → {verdict}")

    # ═══════════════════════════════════════════════════════════
    # 2. DIRECTION PERSISTENCE — BY SIGNAL TYPE
    # ═══════════════════════════════════════════════════════════
    print("\n" + "=" * 80)
    print("2. DIRECTION PERSISTENCE BY SIGNAL TYPE")
    print("=" * 80)

    for sig in ['STRONG_CONTRACTION', 'WEAK_EXPANSION', 'MILD_EXPANSION',
                'STRONG_EXPANSION', 'MILD_CONTRACTION']:
        sig_data = data[data['signal'] == sig]
        if len(sig_data) < 5:
            continue

        print(f"\n  Signal: {sig} (n={len(sig_data)})")
        print(f"  {'Transition':<40} {'Same%':>7} {'N':>4} {'p':>7} {'Verdict':<15}")
        print("  " + "-" * 75)

        for from_s, to_s, label in transitions[:4]:  # Adjacent sessions only
            mask = sig_data[from_s].notna() & sig_data[to_s].notna()
            subset = sig_data[mask]

            if len(subset) < 3:
                continue

            d_from = subset[from_s].apply(direction)
            d_to = subset[to_s].apply(direction)
            valid_mask = (d_from != 0) & (d_to != 0)

            if valid_mask.sum() < 3:
                continue

            same = (d_from[valid_mask] == d_to[valid_mask]).sum()
            total = valid_mask.sum()
            pct = same / total * 100
            p_val = binomial_test(same, total, 0.5)

            if pct > 65:
                v = "✅ REAL"
            elif pct > 55:
                v = "⚠️  MARG"
            else:
                v = "❌ NO"

            sig_m = "*" if p_val < 0.05 else " "
            print(f"  {label:<40} {pct:>6.1f}% {total:>4} {p_val:>6.3f}{sig_m} {v}")

    # ═══════════════════════════════════════════════════════════
    # 3. DIRECTION PERSISTENCE — BY SURPRISE MAGNITUDE
    # ═══════════════════════════════════════════════════════════
    print("\n" + "=" * 80)
    print("3. DIRECTION PERSISTENCE BY SURPRISE MAGNITUDE")
    print("=" * 80)

    # Split into: big miss (<-1), small miss (-1 to 0), small beat (0 to +1), big beat (>+1)
    surprise_bins = [
        ('Big Miss (surprise < -1.0)',    data[data['surprise'] < -1.0]),
        ('Small Miss (-1.0 to 0)',         data[(data['surprise'] >= -1.0) & (data['surprise'] < 0)]),
        ('Small Beat (0 to +1.0)',         data[(data['surprise'] >= 0) & (data['surprise'] <= 1.0)]),
        ('Big Beat (surprise > +1.0)',     data[data['surprise'] > 1.0]),
    ]

    for label, subset in surprise_bins:
        if len(subset) < 5:
            continue

        print(f"\n  {label} (n={len(subset)})")
        print(f"  {'Transition':<40} {'Same%':>7} {'N':>4} {'p':>7} {'Verdict':<15}")
        print("  " + "-" * 75)

        for from_s, to_s, t_label in transitions[:4]:
            mask = subset[from_s].notna() & subset[to_s].notna()
            sub = subset[mask]

            if len(sub) < 3:
                continue

            d_from = sub[from_s].apply(direction)
            d_to = sub[to_s].apply(direction)
            valid_mask = (d_from != 0) & (d_to != 0)

            if valid_mask.sum() < 3:
                continue

            same = (d_from[valid_mask] == d_to[valid_mask]).sum()
            total = valid_mask.sum()
            pct = same / total * 100
            p_val = binomial_test(same, total, 0.5)

            if pct > 65:
                v = "✅ REAL"
            elif pct > 55:
                v = "⚠️  MARG"
            else:
                v = "❌ NO"

            sig_m = "*" if p_val < 0.05 else " "
            print(f"  {t_label:<40} {pct:>6.1f}% {total:>4} {p_val:>6.3f}{sig_m} {v}")

    # ═══════════════════════════════════════════════════════════
    # 4. STATISTICAL SIGNIFICANCE — 24H AGGREGATE
    # ═══════════════════════════════════════════════════════════
    print("\n" + "=" * 80)
    print("4. STATISTICAL SIGNIFICANCE TESTS")
    print("=" * 80)

    returns_24h = data['Full_24h'].dropna().tolist()

    # One-sample t-test: mean != 0?
    t1, p1 = ttest_1samp(returns_24h, 0)
    mean_24h = sum(returns_24h) / len(returns_24h)
    se_24h = math.sqrt(sum((x - mean_24h)**2 for x in returns_24h) / (len(returns_24h)-1)) / math.sqrt(len(returns_24h))

    print(f"\n  One-sample t-test (H0: 24h mean return = 0)")
    print(f"    Mean: {mean_24h:+.3f}% ± {se_24h:.3f}%")
    print(f"    t = {t1:+.4f}, p = {p1:.4f}")
    print(f"    {'✅ SIGNIFICANT (p<0.05)' if p1 < 0.05 else '❌ NOT SIGNIFICANT'}")

    # Two-sample: beat vs miss
    beat = data[data['surprise'] >= 0]['Full_24h'].dropna().tolist()
    miss = data[data['surprise'] < 0]['Full_24h'].dropna().tolist()

    if len(beat) >= 3 and len(miss) >= 3:
        t2, p2 = ttest_ind(beat, miss)
        print(f"\n  Two-sample t-test (Beat vs Miss)")
        print(f"    Beat (surprise≥0): n={len(beat)}, avg={sum(beat)/len(beat):+.3f}%")
        print(f"    Miss (surprise<0): n={len(miss)}, avg={sum(miss)/len(miss):+.3f}%")
        print(f"    t = {t2:+.4f}, p = {p2:.4f}")
        print(f"    {'✅ SIGNIFICANT (p<0.05)' if p2 < 0.05 else '❌ NOT SIGNIFICANT'}")

    # Two-sample: big miss vs small miss
    big_miss = data[data['surprise'] < -1.0]['Full_24h'].dropna().tolist()
    small_miss = data[(data['surprise'] >= -1.0) & (data['surprise'] < 0)]['Full_24h'].dropna().tolist()

    if len(big_miss) >= 3 and len(small_miss) >= 3:
        t3, p3 = ttest_ind(big_miss, small_miss)
        print(f"\n  Two-sample t-test (Big Miss vs Small Miss)")
        print(f"    Big Miss (surprise<-1): n={len(big_miss)}, avg={sum(big_miss)/len(big_miss):+.3f}%")
        print(f"    Small Miss (-1≤surprise<0): n={len(small_miss)}, avg={sum(small_miss)/len(small_miss):+.3f}%")
        print(f"    t = {t3:+.4f}, p = {p3:.4f}")
        print(f"    {'✅ SIGNIFICANT (p<0.05)' if p3 < 0.05 else '❌ NOT SIGNIFICANT'}")

    # Two-sample: Europe positive vs Europe negative
    eu_pos = data[data['Europe_Open'] > 0]['Full_24h'].dropna().tolist()
    eu_neg = data[data['Europe_Open'] < 0]['Full_24h'].dropna().tolist()

    if len(eu_pos) >= 3 and len(eu_neg) >= 3:
        t4, p4 = ttest_ind(eu_pos, eu_neg)
        print(f"\n  Two-sample t-test (Europe Positive vs Europe Negative)")
        print(f"    EU Open > 0: n={len(eu_pos)}, avg={sum(eu_pos)/len(eu_pos):+.3f}%")
        print(f"    EU Open < 0: n={len(eu_neg)}, avg={sum(eu_neg)/len(eu_neg):+.3f}%")
        print(f"    t = {t4:+.4f}, p = {p4:.4f}")
        print(f"    {'✅ SIGNIFICANT (p<0.05)' if p4 < 0.05 else '❌ NOT SIGNIFICANT'}")

    # ═══════════════════════════════════════════════════════════
    # 5. VOLATILITY EXPANSION ANALYSIS
    # ═══════════════════════════════════════════════════════════
    print("\n" + "=" * 80)
    print("5. VOLATILITY EXPANSION BY SESSION")
    print("=" * 80)
    print("\n  Does PMI release day volatility persist across sessions?")

    for sig in ['STRONG_CONTRACTION', 'WEAK_EXPANSION', 'MILD_EXPANSION', 'STRONG_EXPANSION']:
        sig_data = data[data['signal'] == sig]
        if len(sig_data) < 3:
            continue

        print(f"\n  Signal: {sig} (n={len(sig_data)})")
        for sname in ['Europe_Open', 'UK_Session', 'US_Open', 'US_Afternoon', 'Asia_Reopen']:
            vol_col = f'{sname}_vol'
            vdata = sig_data[vol_col].dropna()
            retdata = sig_data[sname].dropna()
            if len(vdata) > 0:
                avg_vol = vdata.mean()
                avg_ret = retdata.mean() if len(retdata) > 0 else 0
                print(f"    {sname:<16} avg bar range: ${avg_vol:.2f}  avg return: {avg_ret:+.2f}%")

    # ═══════════════════════════════════════════════════════════
    # 6. FULL CHAIN DIAGRAM — YOUR HYPOTHESIS
    # ═══════════════════════════════════════════════════════════
    print("\n" + "=" * 80)
    print("6. YOUR HYPOTHESIS: SESSION ITINERARY DIAGRAM")
    print("=" * 80)

    print("""
    Your claim:
    Europe Open (shock) → UK (transmit) → US Open (price in ECB cut) → Asia Reopen (mean revert?)

    What the data says:
    """)

    for from_s, to_s, label in transitions[:4]:
        mask = data[from_s].notna() & data[to_s].notna()
        subset = data[mask]
        if len(subset) < 5:
            continue

        d_from = subset[from_s].apply(direction)
        d_to = subset[to_s].apply(direction)
        valid_mask = (d_from != 0) & (d_to != 0)
        if valid_mask.sum() < 5:
            continue

        same = (d_from[valid_mask] == d_to[valid_mask]).sum()
        total = valid_mask.sum()
        pct = same / total * 100

        arrow = "→" if pct > 55 else "↛"
        strength = "STRONG" if pct > 65 else ("WEAK" if pct > 55 else "BROKEN")

        print(f"    {from_s:<16} {arrow} {to_s:<16}  {pct:.1f}% persist  [{strength}]")

    # Summary
    print("\n" + "=" * 80)
    print("SUMMARY VERDICT")
    print("=" * 80)

    # Check each chain link
    chain_holds = True
    for cr in chain_results:
        if 'Adjacent' in cr['label'] or '→' in cr['label']:
            pass  # already printed

    # Use the first 4 adjacent transitions
    eu_uk = chain_results[0]  # Europe → UK
    uk_us = chain_results[1]  # UK → US
    us_asia = chain_results[3]  # US PM → Asia

    print(f"""
  1. Europe Open → UK Session:    {eu_uk['pct']:.1f}%  {eu_uk['verdict']}
  2. UK Session → US Open:        {uk_us['pct']:.1f}%  {uk_us['verdict']}
  3. US Open → US Afternoon:      {chain_results[2]['pct']:.1f}%  {chain_results[2]['verdict']}
  4. US Afternoon → Asia Reopen:  {us_asia['pct']:.1f}%  {us_asia['verdict']}

  24h aggregate: mean {mean_24h:+.2f}% (p={p1:.3f}) — {'significant' if p1 < 0.05 else 'NOT significant'}

  The chain {'HOLDS' if eu_uk['pct'] > 55 and uk_us['pct'] > 55 else 'BREAKS'} for Europe→UK→US.
  The chain {'HOLDS' if us_asia['pct'] > 55 else 'BREAKS'} for US→Asia.
    """)

    # Save detailed results
    output_path = '/root/.openclaw/workspace/jimi/analysis/ez_pmi_chain_validation.csv'
    data.to_csv(output_path, index=False)
    print(f"  ✅ Detailed results saved to {output_path}")

    return data


if __name__ == '__main__':
    csv_path = '/root/.openclaw/workspace/jimi/eth_15m_merged.csv'
    run_validation(csv_path)

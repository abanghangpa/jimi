#!/usr/bin/env python3
"""
Prompt C v2: EZ PMI Transmission Chain — User's Itinerary

Transitions:
  Europe → UK → US Open → US Afternoon → Asia Open → Asia Afternoon → Europe Reopen
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import math
import warnings
warnings.filterwarnings('ignore')

def norm_cdf(x):
    return 0.5 * (1 + math.erf(x / math.sqrt(2)))

def ttest_1samp(x, mu=0):
    n = len(x)
    if n < 2: return 0.0, 1.0
    mean = sum(x) / n
    var = sum((xi - mean) ** 2 for xi in x) / (n - 1)
    se = math.sqrt(var / n)
    if se == 0: return 0.0, 1.0
    t = (mean - mu) / se
    return t, 2 * (1 - norm_cdf(abs(t)))

def ttest_ind(a, b):
    na, nb = len(a), len(b)
    if na < 2 or nb < 2: return 0.0, 1.0
    ma, mb = sum(a)/na, sum(b)/nb
    va = sum((x-ma)**2 for x in a)/(na-1)
    vb = sum((x-mb)**2 for x in b)/(nb-1)
    se = math.sqrt(va/na + vb/nb)
    if se == 0: return 0.0, 1.0
    t = (ma - mb) / se
    return t, 2 * (1 - norm_cdf(abs(t)))

def binomial_test(k, n, p0=0.5):
    se = math.sqrt(n * p0 * (1-p0))
    if se == 0: return 1.0
    z = (k - n*p0) / se
    return 2 * (1 - norm_cdf(abs(z)))


# ═══════════════════════════════════════════════════════════════
# EZ PMI RELEASES
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
# SESSION DEFINITIONS — USER'S ITINERARY
# All times UTC. EZ PMI releases at 08:00 UTC (09:00 CET / 16:00 MYT)
# ═══════════════════════════════════════════════════════════════

SESSIONS = {
    # Release day
    'Europe_Open':       (7, 0, 12, 0),       # 07:00-12:00 UTC (15:00-20:00 MYT)
    'UK_Session':        (8, 0, 16, 30),       # 08:00-16:30 UTC (16:00-00:30 MYT)
    'US_Open':           (13, 30, 17, 0),      # 13:30-17:00 UTC (21:30-01:00 MYT)
    'US_Afternoon':      (17, 0, 21, 0),       # 17:00-21:00 UTC (01:00-05:00 MYT)
    # Next day
    'Asia_Open':         (1, 0, 5, 0, 1),      # +1 day 01:00-05:00 UTC (09:00-13:00 MYT)
    'Asia_Afternoon':    (5, 0, 9, 0, 1),      # +1 day 05:00-09:00 UTC (13:00-17:00 MYT)
    'Europe_Reopen':     (7, 0, 12, 0, 1),     # +1 day 07:00-12:00 UTC (15:00-20:00 MYT)
    # Full cycle — handled specially in get_session_return
    'Full_24h':          None,
}

# Transition chain (in order)
TRANSITIONS = [
    ('Europe_Open',    'UK_Session',      'Europe Open → UK Session'),
    ('UK_Session',     'US_Open',         'UK Session → US Open'),
    ('US_Open',        'US_Afternoon',    'US Open → US Afternoon'),
    ('US_Afternoon',   'Asia_Open',       'US Afternoon → Asia Open'),
    ('Asia_Open',      'Asia_Afternoon',  'Asia Open → Asia Afternoon'),
    ('Asia_Afternoon', 'Europe_Reopen',   'Asia Afternoon → Europe Reopen'),
]


def load_data(csv_path):
    df = pd.read_csv(csv_path)
    df['Open time'] = pd.to_datetime(df['Open time'])
    df = df.set_index('Open time')
    for col in ['Open', 'High', 'Low', 'Close', 'Volume']:
        df[col] = df[col].astype(float)
    return df


def get_session_return(df, release_date, session_name, session_def):
    if session_name == 'Full_24h' or session_def is None:
        start = pd.Timestamp(release_date) + timedelta(hours=8)
        end = start + timedelta(hours=24)
    elif len(session_def) == 5:
        h_start, m_start, h_end, m_end, day_offset = session_def
        start = pd.Timestamp(release_date) + timedelta(days=day_offset, hours=h_start, minutes=m_start)
        end = pd.Timestamp(release_date) + timedelta(days=day_offset, hours=h_end, minutes=m_end)
    else:
        h_start, m_start, h_end, m_end = session_def
        start = pd.Timestamp(release_date) + timedelta(hours=h_start, minutes=m_start)
        end = pd.Timestamp(release_date) + timedelta(hours=h_end, minutes=m_end)

    mask = (df.index >= start) & (df.index <= end)
    session_df = df[mask]
    if len(session_df) < 2:
        return None

    open_price = session_df.iloc[0]['Open']
    close_price = session_df.iloc[-1]['Close']
    return (close_price - open_price) / open_price * 100


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
    if val is None or pd.isna(val):
        return None
    return 1 if val > 0 else (-1 if val < 0 else 0)


def analyze_transition(data, from_s, to_s, label, indent="  "):
    """Analyze a single transition. Returns result dict."""
    mask = data[from_s].notna() & data[to_s].notna()
    subset = data[mask]
    if len(subset) < 5:
        print(f"{indent}{label:<42} insufficient data (n={len(subset)})")
        return None

    d_from = subset[from_s].apply(direction)
    d_to = subset[to_s].apply(direction)
    valid_mask = (d_from != 0) & (d_to != 0)
    total = valid_mask.sum()

    if total < 5:
        print(f"{indent}{label:<42} insufficient valid data (n={total})")
        return None

    same = (d_from[valid_mask] == d_to[valid_mask]).sum()
    pct = same / total * 100
    p_val = binomial_test(same, total, 0.5)
    corr = subset[[from_s, to_s]].corr().iloc[0, 1]

    # Same-dir vs opp-dir average
    same_mask = (d_from == d_to) & valid_mask
    opp_mask = (d_from != d_to) & valid_mask
    same_avg = subset.loc[same_mask, to_s].mean() if same_mask.sum() > 0 else 0
    opp_avg = subset.loc[opp_mask, to_s].mean() if opp_mask.sum() > 0 else 0

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


def run_validation(csv_path):
    print("=" * 80)
    print("PROMPT C v2: EZ PMI TRANSMISSION CHAIN")
    print("Your Itinerary: Europe → UK → US → US PM → Asia AM → Asia PM → EU Reopen")
    print("=" * 80)

    df = load_data(csv_path)

    # Build session returns
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
            row[sname] = get_session_return(df, date_str, sname, sdef)
        rows.append(row)

    data = pd.DataFrame(rows)

    # ═══════════════════════════════════════════════════════════
    # 1. FULL CHAIN — ALL RELEASES
    # ═══════════════════════════════════════════════════════════
    print("\n" + "=" * 80)
    print("1. FULL CHAIN — ALL RELEASES (n={})".format(len(data)))
    print("=" * 80)

    chain_results = []
    for from_s, to_s, label in TRANSITIONS:
        r = analyze_transition(data, from_s, to_s, label)
        if r:
            chain_results.append(r)

    # Visual chain
    print("\n  Chain visualization:")
    for cr in chain_results:
        arrow = "━━━→" if cr['pct'] > 55 else "╌╌╌↛"
        bar_len = int(cr['pct'] / 2)
        bar = "█" * bar_len + "░" * (50 - bar_len)
        print(f"    {cr['label'][:20]:<20} {bar} {cr['pct']:.1f}%")

    # ═══════════════════════════════════════════════════════════
    # 2. CHAIN BY SIGNAL TYPE
    # ═══════════════════════════════════════════════════════════
    print("\n" + "=" * 80)
    print("2. CHAIN BY SIGNAL TYPE")
    print("=" * 80)

    for sig in ['STRONG_CONTRACTION', 'WEAK_EXPANSION', 'MILD_EXPANSION',
                'STRONG_EXPANSION', 'MILD_CONTRACTION']:
        sig_data = data[data['signal'] == sig]
        if len(sig_data) < 5:
            continue

        print(f"\n  Signal: {sig} (n={len(sig_data)})")
        for from_s, to_s, label in TRANSITIONS:
            analyze_transition(sig_data, from_s, to_s, label, indent="    ")

    # ═══════════════════════════════════════════════════════════
    # 3. CHAIN BY SURPRISE MAGNITUDE
    # ═══════════════════════════════════════════════════════════
    print("\n" + "=" * 80)
    print("3. CHAIN BY SURPRISE MAGNITUDE")
    print("=" * 80)

    surprise_bins = [
        ('Big Miss (<-1.0)',        data[data['surprise'] < -1.0]),
        ('Small Miss (-1.0 to 0)',   data[(data['surprise'] >= -1.0) & (data['surprise'] < 0)]),
        ('Small Beat (0 to +1.0)',   data[(data['surprise'] >= 0) & (data['surprise'] <= 1.0)]),
        ('Big Beat (>+1.0)',         data[data['surprise'] > 1.0]),
    ]

    for label, subset in surprise_bins:
        if len(subset) < 5:
            continue
        print(f"\n  {label} (n={len(subset)})")
        for from_s, to_s, t_label in TRANSITIONS:
            analyze_transition(subset, from_s, to_s, t_label, indent="    ")

    # ═══════════════════════════════════════════════════════════
    # 4. STATISTICAL SIGNIFICANCE
    # ═══════════════════════════════════════════════════════════
    print("\n" + "=" * 80)
    print("4. STATISTICAL SIGNIFICANCE TESTS")
    print("=" * 80)

    returns_24h = data['Full_24h'].dropna().tolist()
    t1, p1 = ttest_1samp(returns_24h, 0)
    mean_24h = sum(returns_24h) / len(returns_24h)
    se_24h = math.sqrt(sum((x-mean_24h)**2 for x in returns_24h)/(len(returns_24h)-1)) / math.sqrt(len(returns_24h))

    print(f"\n  24h aggregate: {mean_24h:+.3f}% ± {se_24h:.3f}%  t={t1:+.3f}  p={p1:.4f}  {'✅' if p1<0.05 else '❌'}")

    beat = data[data['surprise'] >= 0]['Full_24h'].dropna().tolist()
    miss = data[data['surprise'] < 0]['Full_24h'].dropna().tolist()
    if len(beat)>=3 and len(miss)>=3:
        t2, p2 = ttest_ind(beat, miss)
        print(f"  Beat vs Miss:  beat={sum(beat)/len(beat):+.3f}% (n={len(beat)})  miss={sum(miss)/len(miss):+.3f}% (n={len(miss)})  t={t2:+.3f}  p={p2:.4f}  {'✅' if p2<0.05 else '❌'}")

    eu_pos = data[data['Europe_Open'] > 0]['Full_24h'].dropna().tolist()
    eu_neg = data[data['Europe_Open'] < 0]['Full_24h'].dropna().tolist()
    if len(eu_pos)>=3 and len(eu_neg)>=3:
        t3, p3 = ttest_ind(eu_pos, eu_neg)
        print(f"  EU pos vs neg: pos={sum(eu_pos)/len(eu_pos):+.3f}% (n={len(eu_pos)})  neg={sum(eu_neg)/len(eu_neg):+.3f}% (n={len(eu_neg)})  t={t3:+.3f}  p={p3:.4f}  {'✅' if p3<0.05 else '❌'}")

    # ═══════════════════════════════════════════════════════════
    # 5. SUMMARY
    # ═══════════════════════════════════════════════════════════
    print("\n" + "=" * 80)
    print("5. SUMMARY")
    print("=" * 80)

    print("\n  Your itinerary claim vs data:")
    print()
    for cr in chain_results:
        emoji = "✅" if cr['pct'] > 65 else ("⚠️" if cr['pct'] > 55 else "❌")
        print(f"    {emoji} {cr['label']}")
        print(f"       {cr['pct']:.1f}% same direction (n={cr['n']}, p={cr['p_val']:.4f})")

    # Find where the chain breaks
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

    if all(cr['pct'] > 55 for cr in chain_results):
        print("\n  🏆 FULL CHAIN HOLDS — every transition >55%")
    elif any(cr['pct'] > 65 for cr in chain_results[:2]):
        print("\n  ⚡ PARTIAL CHAIN — Europe→UK→US holds, rest is noise")
    else:
        print("\n  💀 CHAIN DOES NOT HOLD")

    # Save
    output_path = '/root/.openclaw/workspace/jimi/analysis/ez_pmi_chain_v2.csv'
    data.to_csv(output_path, index=False)
    print(f"\n  ✅ Saved to {output_path}")

    return data


if __name__ == '__main__':
    run_validation('/root/.openclaw/workspace/jimi/eth_15m_merged.csv')

#!/usr/bin/env python3
"""
Eurozone Flash PMI (Composite) Backtest — Prompt A

Measures ETH/USDT 15m returns across sessions on EZ PMI release days.
Classifies by Wyckoff phase, vol regime, and signal strength.
Cross-tabulates for edge discovery.
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import math
import warnings
warnings.filterwarnings('ignore')

# Manual implementations to avoid scipy dependency
def _ttest_1samp(x, mu):
    """One-sample t-test, returns (t_stat, p_val)."""
    n = len(x)
    if n < 2:
        return 0.0, 1.0
    mean = sum(x) / n
    var = sum((xi - mean) ** 2 for xi in x) / (n - 1)
    se = math.sqrt(var / n)
    if se == 0:
        return 0.0, 1.0
    t = (mean - mu) / se
    # Approximate p-value using normal approximation for large n
    p = 2 * (1 - _norm_cdf(abs(t))) if n > 30 else 2 * (1 - _norm_cdf(abs(t)))
    return t, p

def _ttest_ind(a, b):
    """Welch's two-sample t-test, returns (t_stat, p_val)."""
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
    # Welch–Satterthwaite df (approximate)
    num = (va / na + vb / nb) ** 2
    den = (va / na) ** 2 / (na - 1) + (vb / nb) ** 2 / (nb - 1)
    df = num / den if den > 0 else na + nb - 2
    p = 2 * (1 - _norm_cdf(abs(t)))
    return t, p

def _norm_cdf(x):
    """Standard normal CDF via error function approximation."""
    return 0.5 * (1 + math.erf(x / math.sqrt(2)))

def _linregress(x, y):
    """Simple linear regression returning (slope, intercept, r_value)."""
    n = len(x)
    if n < 2:
        return 0.0, 0.0, 0.0
    sx = sum(x)
    sy = sum(y)
    sxx = sum(xi * xi for xi in x)
    sxy = sum(xi * yi for xi, yi in zip(x, y))
    denom = n * sxx - sx * sx
    if denom == 0:
        return 0.0, sy / n, 0.0
    slope = (n * sxy - sx * sy) / denom
    intercept = (sy - slope * sx) / n
    # r-value
    syy = sum(yi * yi for yi in y)
    num_r = n * sxy - sx * sy
    den_r = math.sqrt((n * sxx - sx ** 2) * (n * syy - sy ** 2))
    r = num_r / den_r if den_r != 0 else 0.0
    return slope, intercept, r

# ═══════════════════════════════════════════════════════════════
# EUROZONE FLASH PMI COMPOSITE RELEASE DATES & ACTUAL VALUES
# Released ~09:00 CET (08:00 UTC) on first or second Friday of month
# Source: S&P Global (formerly IHS Markit)
# ═══════════════════════════════════════════════════════════════

# Format: (release_date, composite_value, prior_value, consensus)
# We track composite because that's the headline print that moves markets
# Note: Pre-2020 data is approximate from historical records
EZ_PMI_RELEASES = {
    # 2018
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
    # 2019
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
    # 2020
    '2020-01-06': {'composite': 50.9, 'prior': 50.6, 'consensus': 50.7},
    '2020-01-24': {'composite': 51.0, 'prior': 50.9, 'consensus': 50.8},
    '2020-02-21': {'composite': 51.6, 'prior': 51.0, 'consensus': 51.0},
    '2020-03-24': {'composite': 31.4, 'prior': 51.6, 'consensus': 38.8},  # COVID crash
    '2020-04-23': {'composite': 13.5, 'prior': 31.4, 'consensus': 18.0},  # COVID trough
    '2020-05-21': {'composite': 30.5, 'prior': 13.5, 'consensus': 25.0},
    '2020-06-23': {'composite': 47.5, 'prior': 30.5, 'consensus': 41.0},
    '2020-07-24': {'composite': 54.8, 'prior': 47.5, 'consensus': 50.0},
    '2020-08-21': {'composite': 51.6, 'prior': 54.9, 'consensus': 54.5},
    '2020-09-23': {'composite': 50.1, 'prior': 51.9, 'consensus': 51.7},
    '2020-10-23': {'composite': 49.4, 'prior': 50.4, 'consensus': 49.5},
    '2020-11-23': {'composite': 45.3, 'prior': 50.0, 'consensus': 46.0},
    '2020-12-16': {'composite': 49.8, 'prior': 45.3, 'consensus': 45.8},
    # 2021
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
    # 2022
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
    # 2023
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
    # 2024
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
    # 2025
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
    # 2026 (YTD)
    '2026-01-23': {'composite': 50.2, 'prior': 49.6, 'consensus': 49.8},
    '2026-02-20': {'composite': 50.6, 'prior': 50.2, 'consensus': 50.0},
    '2026-03-24': {'composite': 50.1, 'prior': 50.4, 'consensus': 50.5},
    '2026-04-23': {'composite': 49.8, 'prior': 50.2, 'consensus': 50.0},
}

# ═══════════════════════════════════════════════════════════════
# SESSION DEFINITIONS (UTC)
# ═══════════════════════════════════════════════════════════════

SESSIONS = {
    'Europe_Open':    (7, 0, 11, 0),    # 07:00-11:00 UTC (15:00-19:00 MYT)
    'UK_Session':     (7, 0, 16, 0),    # 07:00-16:00 UTC
    'US_Open':        (13, 30, 17, 0),  # 13:30-17:00 UTC (21:30-01:00 MYT)
    'US_Afternoon':   (17, 0, 21, 0),   # 17:00-21:00 UTC
    'Asia_Reopen':    (1, 0, 5, 0),     # next day 01:00-05:00 UTC (09:00-13:00 MYT)
    'Full_24h':       (0, 0, 23, 59),   # full day
}


def load_data(csv_path):
    """Load 15m ETH data."""
    df = pd.read_csv(csv_path)
    df['Open time'] = pd.to_datetime(df['Open time'])
    df = df.set_index('Open time')
    df['Close'] = df['Close'].astype(float)
    df['High'] = df['High'].astype(float)
    df['Low'] = df['Low'].astype(float)
    df['Open'] = df['Open'].astype(float)
    df['Volume'] = df['Volume'].astype(float)
    return df


def get_session_return(df, release_date, session_name, session_def):
    """Calculate return for a specific session window on release day."""
    h_start, m_start, h_end, m_end = session_def

    if session_name == 'Asia_Reopen':
        # Next day
        start = pd.Timestamp(release_date) + timedelta(days=1, hours=h_start, minutes=m_start)
        end = pd.Timestamp(release_date) + timedelta(days=1, hours=h_end, minutes=m_end)
    else:
        start = pd.Timestamp(release_date) + timedelta(hours=h_start, minutes=m_start)
        end = pd.Timestamp(release_date) + timedelta(hours=h_end, minutes=m_end)

    # For Full_24h: from release time (08:00 UTC) to next day 08:00 UTC
    if session_name == 'Full_24h':
        start = pd.Timestamp(release_date) + timedelta(hours=8)  # PMI release time
        end = start + timedelta(hours=24)

    mask = (df.index >= start) & (df.index <= end)
    session_df = df[mask]

    if len(session_df) < 2:
        return None

    # Return from session open to session close
    open_price = session_df.iloc[0]['Open']
    close_price = session_df.iloc[-1]['Close']
    return (close_price - open_price) / open_price * 100


def classify_wyckoff_phase(df, release_date):
    """Simplified Wyckoff phase classification using 4H structure."""
    # Look back 20 days (480 4H bars = 120 15m bars per 4H = need 480 * 4 = 1920 15m bars)
    lookback_start = pd.Timestamp(release_date) - timedelta(days=20)
    mask = (df.index >= lookback_start) & (df.index < pd.Timestamp(release_date))
    recent = df[mask]

    if len(recent) < 500:
        return 'UNKNOWN'

    # Resample to 4H
    h4 = recent.resample('4h').agg({'Open': 'first', 'High': 'max', 'Low': 'min', 'Close': 'last', 'Volume': 'sum'}).dropna()

    if len(h4) < 48:
        return 'UNKNOWN'

    closes = h4['Close'].values
    highs = h4['High'].values
    lows = h4['Low'].values

    # Trend direction (20-period linear regression slope)
    x = np.arange(len(closes))
    slope, _, r_val = _linregress(x[-48:].tolist(), closes[-48:].tolist())
    trend_strength = r_val ** 2

    # Range detection
    recent_high = np.max(highs[-48:])
    recent_low = np.min(lows[-48:])
    range_pct = (recent_high - recent_low) / recent_low * 100

    # Current price position in range
    current = closes[-1]
    range_pos = (current - recent_low) / (recent_high - recent_low) if recent_high != recent_low else 0.5

    # Volume trend
    vol_recent = np.mean(h4['Volume'].values[-12:])
    vol_prior = np.mean(h4['Volume'].values[-48:-12])
    vol_trend = vol_recent / vol_prior if vol_prior > 0 else 1.0

    # Classification
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


def classify_vol_regime(df, release_date):
    """Simplified vol regime classification."""
    lookback_start = pd.Timestamp(release_date) - timedelta(days=5)
    mask = (df.index >= lookback_start) & (df.index < pd.Timestamp(release_date))
    recent = df[mask]

    if len(recent) < 100:
        return 'UNKNOWN'

    # ATR on 15m
    high = recent['High'].values.astype(float)
    low = recent['Low'].values.astype(float)
    close = recent['Close'].values.astype(float)

    tr = np.maximum(high[1:] - low[1:],
                    np.maximum(np.abs(high[1:] - close[:-1]),
                               np.abs(low[1:] - close[:-1])))

    atr_14 = np.convolve(tr, np.ones(14)/14, mode='valid')

    if len(atr_14) < 2:
        return 'UNKNOWN'

    current_atr = atr_14[-1]
    atr_percentile = np.percentile(atr_14, 50)
    atr_p90 = np.percentile(atr_14, 90)

    # Bollinger Band width
    closes = close.astype(float)
    if len(closes) >= 20:
        sma20 = np.convolve(closes, np.ones(20)/20, mode='valid')
        std20 = np.array([np.std(closes[i:i+20]) for i in range(len(closes)-19)])
        bb_width = (2 * std20) / sma20 * 100
        current_bb = bb_width[-1] if len(bb_width) > 0 else 0
        bb_percentile = np.percentile(bb_width, 50) if len(bb_width) > 0 else 0
    else:
        current_bb = 0
        bb_percentile = 0

    # Direction consistency (15m)
    returns = np.diff(closes) / closes[:-1] * 100
    if len(returns) > 0:
        pos_ratio = np.sum(returns > 0) / len(returns)
        direction_consistency = abs(pos_ratio - 0.5) * 2  # 0 = random, 1 = perfectly directional
    else:
        direction_consistency = 0

    # Classify
    if current_atr > atr_p90:
        return 'CRISIS'
    elif current_bb < np.percentile(bb_width, 30) if len(bb_width) > 0 else False:
        return 'COMPRESSING'
    elif direction_consistency > 0.3 and current_atr > atr_percentile:
        return 'TRENDING'
    elif direction_consistency < 0.15 and current_atr < atr_percentile:
        return 'LOW_VOL'
    else:
        return 'CHOP'


def classify_signal(pmi_data):
    """Classify PMI signal strength."""
    composite = pmi_data['composite']
    prior = pmi_data['prior']
    consensus = pmi_data.get('consensus', prior)
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


def run_backtest(csv_path):
    """Run the full backtest."""
    print("=" * 80)
    print("EUROZONE FLASH PMI (COMPOSITE) BACKTEST")
    print("ETH/USDT 15m Data | Session-by-Session Analysis")
    print("=" * 80)

    df = load_data(csv_path)
    print(f"\nData loaded: {df.index[0]} to {df.index[-1]}")
    print(f"Total candles: {len(df):,}")

    results = []

    for date_str, pmi_data in sorted(EZ_PMI_RELEASES.items()):
        release_date = pd.Timestamp(date_str)

        # Check if we have data for this date
        if release_date > df.index[-1]:
            continue
        if release_date < df.index[0]:
            continue

        # Get session returns
        session_returns = {}
        for sname, sdef in SESSIONS.items():
            ret = get_session_return(df, date_str, sname, sdef)
            session_returns[sname] = ret

        # Classify
        wyckoff = classify_wyckoff_phase(df, date_str)
        vol = classify_vol_regime(df, date_str)
        signal = classify_signal(pmi_data)

        results.append({
            'date': date_str,
            'composite': pmi_data['composite'],
            'prior': pmi_data['prior'],
            'consensus': pmi_data.get('consensus', pmi_data['prior']),
            'surprise': pmi_data['composite'] - pmi_data.get('consensus', pmi_data['prior']),
            'wyckoff': wyckoff,
            'vol': vol,
            'signal': signal,
            **session_returns
        })

    results_df = pd.DataFrame(results)

    # Filter out rows with missing 24h return
    valid = results_df.dropna(subset=['Full_24h'])

    print(f"\nTotal EZ PMI releases in dataset: {len(results_df)}")
    print(f"Releases with valid 24h returns: {len(valid)}")

    # ═══════════════════════════════════════════════════════════
    # CROSS-TABULATION: Wyckoff × Vol × Signal
    # ═══════════════════════════════════════════════════════════
    print("\n" + "=" * 80)
    print("CROSS-TABULATION: Wyckoff × Vol × Signal → 24h Return")
    print("=" * 80)

    # Group and aggregate
    combo_stats = valid.groupby(['wyckoff', 'vol', 'signal']).agg(
        avg_24h=('Full_24h', 'mean'),
        win_rate=('Full_24h', lambda x: (x > 0).sum() / len(x) * 100),
        sample_size=('Full_24h', 'count'),
        median_24h=('Full_24h', 'median'),
        std_24h=('Full_24h', 'std')
    ).reset_index()

    combo_stats = combo_stats[combo_stats['sample_size'] >= 2].sort_values('avg_24h', ascending=False)

    print(f"\n{'Wyckoff':<14} {'Vol':<12} {'Signal':<20} {'Avg 24h%':>10} {'Win%':>8} {'N':>4} {'Median%':>10} {'Std%':>8}")
    print("-" * 90)

    for _, row in combo_stats.iterrows():
        edge_marker = ""
        if abs(row['avg_24h']) >= 0.5 and row['sample_size'] >= 3:
            edge_marker = " ⚡"
        elif abs(row['avg_24h']) >= 1.0:
            edge_marker = " 🔥"

        print(f"{row['wyckoff']:<14} {row['vol']:<12} {row['signal']:<20} "
              f"{row['avg_24h']:>+9.2f}% {row['win_rate']:>7.1f}% {row['sample_size']:>4.0f} "
              f"{row['median_24h']:>+9.2f}% {row['std_24h']:>7.2f}%{edge_marker}")

    # ═══════════════════════════════════════════════════════════
    # SESSION-BY-SESSION TRANSMISSION CHAIN
    # ═══════════════════════════════════════════════════════════
    print("\n" + "=" * 80)
    print("SESSION-BY-SESSION TRANSMISSION CHAIN")
    print("=" * 80)

    session_names = ['Europe_Open', 'UK_Session', 'US_Open', 'US_Afternoon', 'Asia_Reopen', 'Full_24h']

    print(f"\n{'Session':<16} {'Avg Return%':>12} {'Win%':>8} {'N':>4} {'Positive%':>10} {'Negative%':>10}")
    print("-" * 65)

    for sname in session_names:
        col = sname
        sdata = valid[col].dropna()
        if len(sdata) > 0:
            avg_ret = sdata.mean()
            win_pct = (sdata > 0).sum() / len(sdata) * 100
            pos_pct = (sdata > 0).sum() / len(sdata) * 100
            neg_pct = (sdata < 0).sum() / len(sdata) * 100
            print(f"{sname:<16} {avg_ret:>+11.2f}% {win_pct:>7.1f}% {len(sdata):>4.0f} {pos_pct:>9.1f}% {neg_pct:>9.1f}%")

    # Direction persistence between sessions
    print("\n" + "=" * 80)
    print("DIRECTION PERSISTENCE (Session → Next Session)")
    print("=" * 80)

    transitions = [
        ('Europe_Open', 'UK_Session', 'Europe → UK'),
        ('UK_Session', 'US_Open', 'UK → US'),
        ('US_Open', 'US_Afternoon', 'US Open → US Afternoon'),
        ('US_Afternoon', 'Asia_Reopen', 'US → Asia Reopen'),
    ]

    for from_s, to_s, label in transitions:
        mask = valid[from_s].notna() & valid[to_s].notna()
        subset = valid[mask]
        if len(subset) < 3:
            continue

        same_dir = ((subset[from_s] > 0) & (subset[to_s] > 0)) | \
                   ((subset[from_s] < 0) & (subset[to_s] < 0))
        persist_pct = same_dir.sum() / len(subset) * 100

        # Categorize
        if persist_pct > 65:
            verdict = "✅ REAL EDGE"
        elif persist_pct > 55:
            verdict = "⚠️  MARGINAL"
        else:
            verdict = "❌ NO CHAIN"

        print(f"  {label:<28} {persist_pct:>5.1f}% same direction (n={len(subset)})  {verdict}")

    # ═══════════════════════════════════════════════════════════
    # STATISTICAL SIGNIFICANCE TESTS
    # ═══════════════════════════════════════════════════════════
    print("\n" + "=" * 80)
    print("STATISTICAL SIGNIFICANCE TESTS")
    print("=" * 80)

    returns_24h = valid['Full_24h'].dropna()

    # One-sample t-test: is mean return different from 0?
    t_stat, p_val = _ttest_1samp(returns_24h.tolist(), 0)
    print(f"\n  One-sample t-test (H0: mean = 0)")
    print(f"    t = {t_stat:+.3f}, p = {p_val:.4f}")
    print(f"    Mean: {returns_24h.mean():+.2f}% ± {returns_24h.sem():.2f}%")
    print(f"    {'✅ SIGNIFICANT (p<0.05)' if p_val < 0.05 else '❌ NOT SIGNIFICANT'}")

    # Two-sample t-test: beat vs miss
    beat = valid[valid['surprise'] >= 0]['Full_24h'].dropna()
    miss = valid[valid['surprise'] < 0]['Full_24h'].dropna()

    if len(beat) >= 3 and len(miss) >= 3:
        t_stat2, p_val2 = _ttest_ind(beat.tolist(), miss.tolist())
        print(f"\n  Two-sample t-test (Beat vs Miss)")
        print(f"    Beat (surprise≥0): n={len(beat)}, avg={beat.mean():+.2f}%")
        print(f"    Miss (surprise<0): n={len(miss)}, avg={miss.mean():+.2f}%")
        print(f"    t = {t_stat2:+.3f}, p = {p_val2:.4f}")
        print(f"    {'✅ SIGNIFICANT (p<0.05)' if p_val2 < 0.05 else '❌ NOT SIGNIFICANT'}")

    # ═══════════════════════════════════════════════════════════
    # BEST/WORST COMBOS (n≥3, |avg|≥0.5%)
    # ═══════════════════════════════════════════════════════════
    print("\n" + "=" * 80)
    print("ACTIONABLE EDGES (n≥3, |avg 24h|≥0.5%)")
    print("=" * 80)

    edges = combo_stats[(combo_stats['sample_size'] >= 3) & (combo_stats['avg_24h'].abs() >= 0.5)]

    if len(edges) > 0:
        for _, row in edges.iterrows():
            direction = "LONG" if row['avg_24h'] > 0 else "SHORT"
            print(f"\n  📊 {row['wyckoff']} + {row['vol']} + {row['signal']}")
            print(f"     Direction: {direction}")
            print(f"     Avg 24h: {row['avg_24h']:+.2f}% | Win: {row['win_rate']:.0f}% | N={row['sample_size']:.0f}")
    else:
        print("\n  No combos met the edge criteria (n≥3, |avg|≥0.5%)")

    # ═══════════════════════════════════════════════════════════
    # YOUR SESSION ITINERARY ANALYSIS
    # ═══════════════════════════════════════════════════════════
    print("\n" + "=" * 80)
    print("YOUR SESSION ITINERARY: Europe Open → UK → US → Asia Reopen")
    print("=" * 80)

    # Group by signal type and trace the chain
    for sig in ['STRONG_CONTRACTION', 'WEAK_EXPANSION', 'STRONG_EXPANSION', 'MILD_EXPANSION', 'MILD_CONTRACTION']:
        sig_data = valid[valid['signal'] == sig]
        if len(sig_data) < 2:
            continue

        print(f"\n  Signal: {sig} (n={len(sig_data)})")
        for sname in ['Europe_Open', 'UK_Session', 'US_Open', 'US_Afternoon', 'Asia_Reopen']:
            col = sname
            sdata = sig_data[col].dropna()
            if len(sdata) > 0:
                avg = sdata.mean()
                win = (sdata > 0).sum() / len(sdata) * 100
                print(f"    {sname:<16} {avg:>+6.2f}%  (win {win:.0f}%, n={len(sdata)})")

    # ═══════════════════════════════════════════════════════════
    # RAW DATA TABLE (for review)
    # ═══════════════════════════════════════════════════════════
    print("\n" + "=" * 80)
    print("RAW DATA: All EZ PMI Releases")
    print("=" * 80)

    print(f"\n{'Date':<12} {'Comp':>5} {'Surp':>6} {'Wyckoff':<14} {'Vol':<12} {'Signal':<20} {'EU Open':>8} {'UK':>8} {'US Open':>8} {'Asia':>8} {'24h':>8}")
    print("-" * 130)

    for _, row in valid.sort_values('date').iterrows():
        def fmt(v):
            return f"{v:>+7.2f}%" if pd.notna(v) else "     N/A"

        print(f"{row['date']:<12} {row['composite']:>5.1f} {row['surprise']:>+5.1f} "
              f"{row['wyckoff']:<14} {row['vol']:<12} {row['signal']:<20} "
              f"{fmt(row['Europe_Open'])} {fmt(row['UK_Session'])} {fmt(row['US_Open'])} "
              f"{fmt(row['Asia_Reopen'])} {fmt(row['Full_24h'])}")

    # Save to CSV
    output_path = '/root/.openclaw/workspace/jimi/analysis/ez_pmi_backtest_results.csv'
    valid.to_csv(output_path, index=False)
    print(f"\n✅ Results saved to {output_path}")

    return results_df


if __name__ == '__main__':
    csv_path = '/root/.openclaw/workspace/jimi/eth_15m_merged.csv'
    run_backtest(csv_path)

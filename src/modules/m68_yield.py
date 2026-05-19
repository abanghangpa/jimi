"""M68: 10Y Treasury Yield + TIPS Real Yield — yield spike detection with real/nominal decomposition."""

import numpy as np

try:
    import yfinance as yf
    HAS_YFINANCE = True
except ImportError:
    HAS_YFINANCE = False


def fetch_10y_yield(period="5d", interval="1h"):
    """Fetch ^TNX (10Y yield * 10). yfinance returns yield * 10."""
    if not HAS_YFINANCE:
        return None
    df = yf.download("^TNX", period=period, interval=interval, progress=False)
    if hasattr(df.columns, "levels") and len(df.columns.levels) > 1:
        df.columns = df.columns.droplevel(1)
    if df is not None and len(df) > 0:
        df['yield'] = df['Close'] / 10.0
    return df


def fetch_tips_yield():
    """Fetch TIPS 10Y real yield from FRED (daily only, no API key)."""
    try:
        import pandas as pd
        url = "https://fred.stlouisfed.org/graph/fredgraph.csv?id=DFII10"
        df = pd.read_csv(url, parse_dates=['DATE'], index_col='DATE')
        df.columns = ['DFII10']
        df['DFII10'] = pd.to_numeric(df['DFII10'], errors='coerce')
        return df.dropna()
    except Exception:
        return None


def score_m68_yield(df_10y, df_tips, direction, config=None):
    """Score 10Y yield spike with TIPS cross-check.

    Spike types:
        - Nominal spike + TIPS flat → inflation expectations rising = bearish ETH
        - Both spike (nominal + TIPS) → growth signal = ambiguous
        - Extreme spike (>10bps/1h) → suppress all longs regardless

    Args:
        df_10y: DataFrame with 10Y yield data
        df_tips: DataFrame with TIPS real yield (daily, can be None)
        direction: 'LONG' or 'SHORT'
        config: dict with M68_* keys

    Returns:
        (status, score, details)
    """
    cfg = config or {}

    if df_10y is None or len(df_10y) < 2:
        return 'SKIP', 0.5, {'error': 'insufficient yield data'}

    alert_bps = cfg.get('M68_ALERT_BPS', 5.0)
    extreme_bps = cfg.get('M68_EXTREME_BPS', 10.0)

    yield_now = float(df_10y['yield'].iloc[-1])
    yield_prev = float(df_10y['yield'].iloc[-2])
    delta_bps = (yield_now - yield_prev) * 100  # % to bps

    # TIPS cross-check
    tips_now = None
    tips_prev = None
    inflation_driven = False
    growth_driven = False

    if df_tips is not None and len(df_tips) > 1:
        tips_now = float(df_tips['DFII10'].iloc[-1])
        tips_prev = float(df_tips['DFII10'].iloc[-2])
        tips_delta = tips_now - tips_prev

        if delta_bps > alert_bps:
            if abs(tips_delta) < 2:
                inflation_driven = True
            elif tips_delta > 2:
                growth_driven = True

    spike = delta_bps > alert_bps
    extreme = delta_bps > extreme_bps

    if extreme:
        classification = 'YIELD_EXTREME'
        score_long = 0.15
    elif spike and inflation_driven:
        classification = 'INFLATION_SPIKE'
        score_long = 0.25
    elif spike and growth_driven:
        classification = 'GROWTH_SPIKE'
        score_long = 0.40
    elif spike:
        classification = 'YIELD_SPIKE'
        score_long = 0.30
    else:
        classification = 'NORMAL'
        score_long = 0.50

    if direction == 'LONG':
        score = score_long
    elif direction == 'SHORT':
        score = 1.0 - score_long
    else:
        score = 0.5

    details = {
        'classification': classification,
        'yield_now': round(yield_now, 4),
        'delta_bps': round(delta_bps, 2),
        'tips_now': round(tips_now, 4) if tips_now else None,
        'inflation_driven': inflation_driven,
        'growth_driven': growth_driven,
    }

    status = 'PASS' if spike else 'NEUTRAL'
    return status, score, details

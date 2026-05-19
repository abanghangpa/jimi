"""M67: DXY Divergence Filter — ETH vs DXY divergence detection."""

import numpy as np

try:
    import yfinance as yf
    HAS_YFINANCE = True
except ImportError:
    HAS_YFINANCE = False


def fetch_dxy(period="5d", interval="15m"):
    """Fetch DXY futures via yfinance (DX-Y.NYB)."""
    if not HAS_YFINANCE:
        return None
    df = yf.download("DX-Y.NYB", period=period, interval=interval, progress=False)
    if hasattr(df.columns, "levels") and len(df.columns.levels) > 1:
        df.columns = df.columns.droplevel(1)
    return df


def score_m67_dxy(df_dxy, eth_price_now, eth_price_prev, direction, config=None):
    """Score DXY/ETH divergence.

    DXY is a lagging composite — it reacts to the same macro data ETH reacts to.
    Divergences between DXY and ETH reveal institutional intent.

    Four conditions:
        1. DXY rising + ETH falling  → CONFIRMED_BEARISH
        2. DXY rising + ETH flat/rising → BULLISH_DIVERGENCE (ETH relative strength)
        3. DXY falling + ETH rising  → CONFIRMED_BULLISH
        4. DXY falling + ETH flat/falling → BEARISH_DIVERGENCE (ETH relative weakness)

    Args:
        df_dxy: DataFrame with DXY OHLCV
        eth_price_now: current ETH price
        eth_price_prev: ETH price 15m ago
        direction: 'LONG' or 'SHORT'
        config: dict with M67_* keys

    Returns:
        (status, score, details)
    """
    cfg = config or {}

    if df_dxy is None or len(df_dxy) < 2:
        return 'SKIP', 0.5, {'error': 'insufficient DXY data'}

    dxy_thresh = cfg.get('M67_DXY_THRESH', 0.2)
    eth_thresh = cfg.get('M67_DIVERGENCE_ETH_THRESH', 0.05)

    dxy_now = float(df_dxy['Close'].iloc[-1])
    dxy_prev = float(df_dxy['Close'].iloc[-2])
    dxy_roc = (dxy_now - dxy_prev) / dxy_prev * 100 if dxy_prev != 0 else 0

    eth_roc = (eth_price_now - eth_price_prev) / eth_price_prev * 100 if eth_price_prev != 0 else 0

    dxy_up = dxy_roc > dxy_thresh
    dxy_down = dxy_roc < -dxy_thresh
    eth_up = eth_roc > eth_thresh
    eth_down = eth_roc < -eth_thresh

    if dxy_up and eth_down:
        classification = 'CONFIRMED_BEARISH'
        score_long = 0.25
    elif dxy_up and not eth_down:
        classification = 'BULLISH_DIVERGENCE'
        score_long = 0.70
    elif dxy_down and eth_up:
        classification = 'CONFIRMED_BULLISH'
        score_long = 0.75
    elif dxy_down and not eth_up:
        classification = 'BEARISH_DIVERGENCE'
        score_long = 0.30
    else:
        classification = 'NEUTRAL'
        score_long = 0.50

    if direction == 'LONG':
        score = score_long
    elif direction == 'SHORT':
        score = 1.0 - score_long
    else:
        score = 0.5

    details = {
        'classification': classification,
        'dxy_roc_15m': round(dxy_roc, 4),
        'eth_roc_15m': round(eth_roc, 4),
        'divergence': classification in ('BULLISH_DIVERGENCE', 'BEARISH_DIVERGENCE'),
    }

    status = 'PASS' if classification != 'NEUTRAL' else 'NEUTRAL'
    return status, score, details

"""M66: USD/JPY Carry Trade Proxy — FX carry unwind detection with DXY cross-check."""

import numpy as np

try:
    import yfinance as yf
    HAS_YFINANCE = True
except ImportError:
    HAS_YFINANCE = False


def fetch_usdjpy(period="5d", interval="1m"):
    """Fetch USD/JPY 1m bars via yfinance."""
    if not HAS_YFINANCE:
        return None
    df = yf.download("JPY=X", period=period, interval=interval, progress=False)
    if hasattr(df.columns, "levels") and len(df.columns.levels) > 1:
        df.columns = df.columns.droplevel(1)
    return df


def score_m66_usdjpy(df_usdjpy, df_dxy, direction, config=None):
    """Score USD/JPY carry trade signal with DXY cross-check.

    Mechanism:
        USD/JPY drop → JPY strength → carry unwind → ETH selling.
        But direction depends on WHY USD/JPY drops:
          - BoJ-driven / geopolitical → carry unwind → bearish ETH
          - USD weakness (DXY also falling) → NOT carry → bullish ETH

    Args:
        df_usdjpy: DataFrame with USD/JPY OHLCV (1m bars)
        df_dxy: DataFrame with DXY OHLCV (15m bars)
        direction: 'LONG' or 'SHORT'
        config: dict with M66_* keys

    Returns:
        (status, score, details)
    """
    cfg = config or {}

    if df_usdjpy is None or len(df_usdjpy) < 20:
        return 'SKIP', 0.5, {'error': 'insufficient USD/JPY data'}

    alert_thresh = cfg.get('M66_ALERT_THRESH', -0.3)
    confirmed_thresh = cfg.get('M66_CONFIRMED_THRESH', -0.8)

    # Compute rate of change on available lookback
    n = min(len(df_usdjpy), 15)
    close_now = float(df_usdjpy['Close'].iloc[-1])
    close_5m = float(df_usdjpy['Close'].iloc[-min(5, len(df_usdjpy))])
    close_15m = float(df_usdjpy['Close'].iloc[-n])

    roc_5m = (close_now - close_5m) / close_5m * 100 if close_5m != 0 else 0
    roc_15m = (close_now - close_15m) / close_15m * 100 if close_15m != 0 else 0

    # DXY cross-check
    dxy_roc_5m = 0.0
    if df_dxy is not None and len(df_dxy) >= 2:
        dxy_now = float(df_dxy['Close'].iloc[-1])
        dxy_prev = float(df_dxy['Close'].iloc[-min(2, len(df_dxy))])
        dxy_roc_5m = (dxy_now - dxy_prev) / dxy_prev * 100 if dxy_prev != 0 else 0

    # Classification
    if roc_5m < alert_thresh and dxy_roc_5m >= 0:
        classification = 'CARRY_ALERT'
        bearish_score = 0.65
    elif roc_15m < confirmed_thresh and dxy_roc_5m >= 0:
        classification = 'CARRY_UNWIND_CONFIRMED'
        bearish_score = 0.80
    elif roc_5m < alert_thresh and dxy_roc_5m < 0:
        classification = 'USD_WEAKNESS'
        bearish_score = 0.50  # neutral — not a carry event
    else:
        classification = 'NORMAL'
        bearish_score = 0.50

    # Direction-aware scoring
    if direction == 'LONG':
        score = 1.0 - bearish_score + 0.5
    elif direction == 'SHORT':
        score = bearish_score
    else:
        score = 0.5

    score = max(0.0, min(1.0, score))

    details = {
        'classification': classification,
        'usdjpy_roc_5m': round(roc_5m, 4),
        'usdjpy_roc_15m': round(roc_15m, 4),
        'dxy_roc_5m': round(dxy_roc_5m, 4),
    }

    status = 'PASS' if classification != 'NORMAL' else 'NEUTRAL'
    return status, score, details

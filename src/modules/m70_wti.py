"""M70: WTI Crude Oil Signal — oil price impact on ETH with DXY cross-check."""

import numpy as np

try:
    import yfinance as yf
    HAS_YFINANCE = True
except ImportError:
    HAS_YFINANCE = False


def fetch_wti(period="5d", interval="4h"):
    """Fetch WTI front-month futures (CL=F) via yfinance."""
    if not HAS_YFINANCE:
        return None
    df = yf.download("CL=F", period=period, interval=interval, progress=False)
    if hasattr(df.columns, "levels") and len(df.columns.levels) > 1:
        df.columns = df.columns.droplevel(1)
    return df


def score_m70_wti(df_wti, df_dxy, direction, config=None):
    """Score oil price move with DXY cross-check.

    Oil impact on ETH is two-sided:
        Oil spike → inflation fear → rate hike risk → ETH down
        Oil drop  → deflationary → rate cut expectations → ETH bullish

    Classification requires DXY:
        Oil up + DXY up   → SUPPLY_SHOCK (inflation) → bearish ETH
        Oil up + DXY down → DEMAND_RISK_ON → ambiguous
        Oil down + DXY up → RECESSION_FEAR → ambiguous
        Oil down + DXY down → REFLATION_EASING → mild bullish ETH

    Args:
        df_wti: DataFrame with WTI OHLCV
        df_dxy: DataFrame with DXY OHLCV
        direction: 'LONG' or 'SHORT'
        config: dict with M70_* keys

    Returns:
        (status, score, details)
    """
    cfg = config or {}

    if df_wti is None or len(df_wti) < 2:
        return 'SKIP', 0.5, {'error': 'insufficient WTI data'}

    alert_thresh = cfg.get('M70_ALERT_THRESH', 3.0)

    oil_now = float(df_wti['Close'].iloc[-1])
    oil_prev = float(df_wti['Close'].iloc[-2])
    oil_roc = (oil_now - oil_prev) / oil_prev * 100 if oil_prev != 0 else 0

    significant = abs(oil_roc) > alert_thresh

    if not significant:
        return 'NEUTRAL', 0.5, {'classification': 'NORMAL', 'oil_roc_4h': round(oil_roc, 2)}

    # DXY cross-check
    dxy_roc = 0.0
    if df_dxy is not None and len(df_dxy) >= 2:
        dxy_now = float(df_dxy['Close'].iloc[-1])
        dxy_prev = float(df_dxy['Close'].iloc[-2])
        dxy_roc = (dxy_now - dxy_prev) / dxy_prev * 100 if dxy_prev != 0 else 0

    oil_up = oil_roc > 0
    dxy_up = dxy_roc > 0

    if oil_up and dxy_up:
        classification = 'SUPPLY_SHOCK_BEARISH'
        long_score = 0.25
    elif oil_up and not dxy_up:
        classification = 'DEMAND_RISK_ON'
        long_score = 0.50
    elif not oil_up and dxy_up:
        classification = 'RECESSION_FEAR'
        long_score = 0.45
    else:
        classification = 'REFLATION_EASING'
        long_score = 0.60

    if direction == 'LONG':
        score = long_score
    elif direction == 'SHORT':
        score = 1.0 - long_score
    else:
        score = 0.5

    details = {
        'classification': classification,
        'oil_roc_4h': round(oil_roc, 2),
        'dxy_roc_4h': round(dxy_roc, 4),
    }

    status = 'PASS' if classification != 'NORMAL' else 'NEUTRAL'
    return status, score, details

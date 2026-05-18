"""M69: VIX Regime Classifier — non-linear VIX/ETH relationship with contrarian crisis logic."""

import numpy as np

try:
    import yfinance as yf
    HAS_YFINANCE = True
except ImportError:
    HAS_YFINANCE = False


def fetch_vix(period="5d", interval="1d"):
    """Fetch ^VIX via yfinance (daily bars, 15min delayed)."""
    if not HAS_YFINANCE:
        return None
    df = yf.download("^VIX", period=period, interval=interval, progress=False)
    return df


def score_m69_vix(df_vix, direction, config=None):
    """Score VIX regime with non-linear ETH impact.

    VIX has a non-linear relationship with ETH:
        < 15:  Complacency — high leverage, squeeze risk
        15-20: Normal operating range
        20-30: Elevated — reduce long confidence
        30-40: Fear — institutional de-leveraging
        > 40:  CRISIS — capitulation = often BEST long entries (contrarian)

    Rate of change matters as much as level:
        VIX spiking >3 pts in one session = immediate risk-off

    Args:
        df_vix: DataFrame with VIX OHLCV
        direction: 'LONG' or 'SHORT'
        config: dict with M69_* keys

    Returns:
        (status, score, details)
    """
    cfg = config or {}

    if df_vix is None or len(df_vix) < 2:
        return 'SKIP', 0.5, {'error': 'insufficient VIX data'}

    complacent = cfg.get('M69_COMPLACENT_THRESH', 15.0)
    elevated = cfg.get('M69_ELEVATED_THRESH', 20.0)
    fear = cfg.get('M69_FEAR_THRESH', 30.0)
    crisis = cfg.get('M69_CRISIS_THRESH', 40.0)
    spike_delta = cfg.get('M69_SPIKE_DELTA', 3.0)

    vix_now = float(df_vix['Close'].iloc[-1])
    vix_prev = float(df_vix['Close'].iloc[-2])
    vix_delta = vix_now - vix_prev

    # Rate-of-change override: spike > 3 pts = immediate risk-off
    if vix_delta > spike_delta:
        classification = 'VIX_SPIKE'
        long_score = 0.25
        reversal_flag = False
    elif vix_now > crisis:
        classification = 'CRISIS'
        long_score = 0.55  # contrarian: crisis = capitulation = potential long
        reversal_flag = True
    elif vix_now > fear:
        classification = 'FEAR'
        long_score = 0.35
        reversal_flag = False
    elif vix_now > elevated:
        classification = 'ELEVATED'
        long_score = 0.40
        reversal_flag = False
    elif vix_now < complacent:
        classification = 'COMPLACENT'
        long_score = 0.45  # complacency = squeeze risk
        reversal_flag = False
    else:
        classification = 'NORMAL'
        long_score = 0.50
        reversal_flag = False

    if direction == 'LONG':
        score = long_score
    elif direction == 'SHORT':
        score = 1.0 - long_score
    else:
        score = 0.5

    details = {
        'classification': classification,
        'vix_level': round(vix_now, 2),
        'vix_delta': round(vix_delta, 2),
        'reversal_potential': reversal_flag,
    }

    status = 'PASS' if classification != 'NORMAL' else 'NEUTRAL'
    return status, score, details

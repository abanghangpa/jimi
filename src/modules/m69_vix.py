"""M69: VIX Regime Classifier — non-linear VIX/ETH relationship with crisis type detection."""

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
    if hasattr(df.columns, "levels") and len(df.columns.levels) > 1:
        df.columns = df.columns.droplevel(1)
    return df


def score_m69_vix(df_vix, direction, config=None, df_dxy=None):
    """Score VIX regime with non-linear ETH impact and crisis type detection.

    VIX has a non-linear relationship with ETH:
        < 15:  Complacency — high leverage, squeeze risk
        15-20: Normal operating range
        20-30: Elevated — reduce long confidence
        30-40: Fear — institutional de-leveraging
        > 40:  CRISIS — but TYPE matters:
               - Liquidity panic (VIX spike + DXY falling) → Fed will intervene → contrarian LONG
               - Structural break (VIX spike + DXY rising) → genuine risk-off → stay short

    Rate of change matters as much as level:
        VIX spiking >3 pts in one session = immediate risk-off

    Crisis type classification (key innovation):
        VIX > 30 + DXY falling = LIQUIDITY_CRISIS
            → Fed/draghi-style intervention likely
            → Contrarian long setup (capitulation = bottom)
            → Score: 0.60 for LONG (bullish contrarian)

        VIX > 30 + DXY rising = STRUCTURAL_BREAK
            → Geopolitical/pandemic/war — no quick Fed fix
            → Genuine risk-off, do NOT go long
            → Score: 0.25 for LONG (bearish)

        VIX > 30 + DXY flat = ELEVATED_FEAR
            → Unclear — wait for DXY direction
            → Score: 0.40 for LONG (defensive)

    This distinction prevents the classic mistake of buying "cheap"
    ETH during a war/pandemic because "VIX is high so it's a bottom."

    Args:
        df_vix: DataFrame with VIX OHLCV
        direction: 'LONG' or 'SHORT'
        config: dict with M69_* keys
        df_dxy: optional DataFrame with DXY OHLCV for crisis classification

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

    # ── DXY direction for crisis type classification ──
    dxy_roc = 0.0
    dxy_direction = 'FLAT'
    if df_dxy is not None and len(df_dxy) >= 2:
        dxy_now = float(df_dxy['Close'].iloc[-1])
        dxy_prev = float(df_dxy['Close'].iloc[-2])
        dxy_roc = (dxy_now - dxy_prev) / dxy_prev * 100 if dxy_prev != 0 else 0
        if dxy_roc > 0.15:
            dxy_direction = 'RISING'
        elif dxy_roc < -0.15:
            dxy_direction = 'FALLING'

    # ── Classification ──
    reversal_flag = False
    crisis_type = None

    # Rate-of-change override: spike > 3 pts = immediate risk-off
    if vix_delta > spike_delta:
        # Spike — classify by DXY direction
        if dxy_direction == 'FALLING':
            classification = 'SPIKE_LIQUIDITY'
            long_score = 0.55  # liquidity panic → Fed intervention likely → contrarian
            crisis_type = 'LIQUIDITY_PANIC'
            reversal_flag = True
        elif dxy_direction == 'RISING':
            classification = 'SPIKE_STRUCTURAL'
            long_score = 0.25  # structural break → genuine risk-off
            crisis_type = 'STRUCTURAL_BREAK'
        else:
            classification = 'VIX_SPIKE'
            long_score = 0.30  # unknown — defensive
            crisis_type = 'UNKNOWN'

    elif vix_now > crisis:
        # Crisis level — classify by DXY direction
        if dxy_direction == 'FALLING':
            classification = 'CRISIS_LIQUIDITY'
            long_score = 0.60  # capitulation + DXY falling = contrarian long
            crisis_type = 'LIQUIDITY_PANIC'
            reversal_flag = True
        elif dxy_direction == 'RISING':
            classification = 'CRISIS_STRUCTURAL'
            long_score = 0.25  # war/pandemic + DXY rising = stay away
            crisis_type = 'STRUCTURAL_BREAK'
        else:
            classification = 'CRISIS_UNKNOWN'
            long_score = 0.35  # unclear — defensive but watch for reversal
            crisis_type = 'UNKNOWN'

    elif vix_now > fear:
        # Fear level — also check DXY for type
        if dxy_direction == 'FALLING':
            classification = 'FEAR_EASING'
            long_score = 0.50  # fear + DXY falling = easing, neutral-to-bullish
        elif dxy_direction == 'RISING':
            classification = 'FEAR_TIGHTENING'
            long_score = 0.30  # fear + DXY rising = tightening, bearish
        else:
            classification = 'FEAR'
            long_score = 0.35

    elif vix_now > elevated:
        classification = 'ELEVATED'
        long_score = 0.40

    elif vix_now < complacent:
        classification = 'COMPLACENT'
        long_score = 0.45  # complacency = squeeze risk

    else:
        classification = 'NORMAL'
        long_score = 0.50

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
        'dxy_roc': round(dxy_roc, 4),
        'dxy_direction': dxy_direction,
        'crisis_type': crisis_type,
        'reversal_potential': reversal_flag,
    }

    status = 'PASS' if classification != 'NORMAL' else 'NEUTRAL'
    return status, score, details

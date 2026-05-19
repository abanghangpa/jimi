"""M71: Gold + DXY Geopolitical Filter — gold/ETH co-movement classification.

CRITICAL CORRECTION: Gold and ETH only co-move during fiat debasement (QE, dollar weakness).
During geopolitical crises, gold rallies while ETH crashes.
Gold up + DXY up = geopolitical panic = BEARISH ETH. Never treat as bullish.
"""

import numpy as np

try:
    import yfinance as yf
    HAS_YFINANCE = True
except ImportError:
    HAS_YFINANCE = False


def fetch_gold(period="5d", interval="4h"):
    """Fetch gold futures (GC=F) via yfinance."""
    if not HAS_YFINANCE:
        return None
    df = yf.download("GC=F", period=period, interval=interval, progress=False)
    if hasattr(df.columns, "levels") and len(df.columns.levels) > 1:
        df.columns = df.columns.droplevel(1)
    return df


def score_m71_gold(df_gold, df_dxy, direction, config=None):
    """Score gold move with DXY to distinguish fiat debasement vs geopolitical.

    KEY RULE from analysis:
        Gold up + DXY up   → GEOPOLITICAL_SAFE_HAVEN → BEARISH ETH
        Gold up + DXY down → FIAT_DEBASEMENT → Bullish ETH (same catalyst)
        Gold down + DXY up → RISK_OFF_RATES → bearish ETH
        Gold down + DXY down → RISK_ON_DRIFT → ambiguous

    Historical evidence:
        Ukraine Feb 2022: Gold +4%, ETH -15% (diverged)
        Oct 2023 Middle East: Gold rallied, ETH sold off (diverged)
        2020-2021 QE: Gold + ETH both rallied (co-moved)

    Args:
        df_gold: DataFrame with gold OHLCV
        df_dxy: DataFrame with DXY OHLCV
        direction: 'LONG' or 'SHORT'
        config: dict with M71_* keys

    Returns:
        (status, score, details)
    """
    cfg = config or {}

    if df_gold is None or len(df_gold) < 2:
        return 'SKIP', 0.5, {'error': 'insufficient gold data'}

    alert_thresh = cfg.get('M71_ALERT_THRESH', 1.0)

    gold_now = float(df_gold['Close'].iloc[-1])
    gold_prev = float(df_gold['Close'].iloc[-2])
    gold_roc = (gold_now - gold_prev) / gold_prev * 100 if gold_prev != 0 else 0

    significant = abs(gold_roc) > alert_thresh

    if not significant:
        return 'NEUTRAL', 0.5, {'classification': 'NORMAL', 'gold_roc_4h': round(gold_roc, 2)}

    # DXY cross-check
    dxy_roc = 0.0
    if df_dxy is not None and len(df_dxy) >= 2:
        dxy_now = float(df_dxy['Close'].iloc[-1])
        dxy_prev = float(df_dxy['Close'].iloc[-2])
        dxy_roc = (dxy_now - dxy_prev) / dxy_prev * 100 if dxy_prev != 0 else 0

    gold_up = gold_roc > 0
    dxy_up = dxy_roc > 0

    # KEY RULE: Gold up + DXY up = geopolitical = bearish ETH
    if gold_up and dxy_up:
        classification = 'GEOPOLITICAL_SAFE_HAVEN'
        long_score = 0.25  # BEARISH — they are diverging, not co-moving
    elif gold_up and not dxy_up:
        classification = 'FIAT_DEBASEMENT'
        long_score = 0.65  # bullish — same catalyst driving both
    elif not gold_up and dxy_up:
        classification = 'RISK_OFF_RATES'
        long_score = 0.35  # bearish
    else:
        classification = 'RISK_ON_DRIFT'
        long_score = 0.50

    if direction == 'LONG':
        score = long_score
    elif direction == 'SHORT':
        score = 1.0 - long_score
    else:
        score = 0.5

    details = {
        'classification': classification,
        'gold_roc_4h': round(gold_roc, 2),
        'dxy_roc_4h': round(dxy_roc, 4),
        'geopolitical_panic': classification == 'GEOPOLITICAL_SAFE_HAVEN',
    }

    status = 'PASS' if classification != 'NORMAL' else 'NEUTRAL'
    return status, score, details

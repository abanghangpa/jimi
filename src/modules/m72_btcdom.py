"""M72: BTC Dominance Regime — altcoin season filter for ETH-specific confidence adjustment."""

import numpy as np

try:
    import requests
    HAS_REQUESTS = True
except ImportError:
    HAS_REQUESTS = False


def fetch_btcdom():
    """Fetch BTC dominance from CoinGecko API (free, no key needed).

    Returns:
        float: BTC dominance percentage, or None if fetch fails
    """
    if not HAS_REQUESTS:
        return None
    try:
        r = requests.get("https://api.coingecko.com/api/v3/global",
                         timeout=10, headers={'Accept': 'application/json'})
        r.raise_for_status()
        data = r.json()
        btc_dominance = data['data']['market_cap_percentage']['btc']
        return float(btc_dominance)
    except Exception:
        return None


def score_m72_btcdom(btc_dominance, direction, config=None):
    """Score BTC dominance regime for ETH-specific adjustment.

    BTC.D rising = capital rotating from altcoins into BTC.
    ETH underperforms even when macro is neutral or bullish.
    BTC.D falling = altcoin season. ETH outperforms.

    Thresholds:
        > 55%: BTC_DOMINANT — reduce ETH long confidence by 20%
        < 48%: ALTCOIN_SEASON — amplify ETH long confidence by 20%
        48-55%: NEUTRAL

    Args:
        btc_dominance: float, BTC dominance percentage
        direction: 'LONG' or 'SHORT'
        config: dict with M72_* keys

    Returns:
        (status, score, details)
    """
    cfg = config or {}

    if btc_dominance is None:
        return 'SKIP', 0.5, {'error': 'no BTC.D data'}

    high_thresh = cfg.get('M72_HIGH_THRESH', 55.0)
    low_thresh = cfg.get('M72_LOW_THRESH', 48.0)

    if btc_dominance > high_thresh:
        classification = 'BTC_DOMINANT'
        long_score = 0.38  # ETH underperforms
    elif btc_dominance < low_thresh:
        classification = 'ALTCOIN_SEASON'
        long_score = 0.62  # ETH outperforms
    else:
        classification = 'NEUTRAL'
        long_score = 0.50

    if direction == 'LONG':
        score = long_score
    elif direction == 'SHORT':
        score = 1.0 - long_score
    else:
        score = 0.5

    details = {
        'classification': classification,
        'btc_dominance': round(btc_dominance, 2),
    }

    status = 'PASS' if classification != 'NEUTRAL' else 'NEUTRAL'
    return status, score, details

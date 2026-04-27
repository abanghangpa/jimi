"""Cross-Asset Correlation — BTC correlation scoring."""

import numpy as np


def score_cross_asset(eth_price, btc_price, btc_corr, btc_change_1h, direction):
    """Score cross-asset alignment."""
    if btc_price is None or np.isnan(btc_corr):
        return 0.5

    if abs(btc_corr) < 0.3:
        return 0.5

    btc_bullish = btc_change_1h > 0.005
    btc_bearish = btc_change_1h < -0.005

    if direction == 'LONG':
        if btc_bullish and btc_corr > 0.5:
            return 0.7
        elif btc_bearish and btc_corr > 0.5:
            return 0.3
    elif direction == 'SHORT':
        if btc_bearish and btc_corr > 0.5:
            return 0.7
        elif btc_bullish and btc_corr > 0.5:
            return 0.3

    return 0.5

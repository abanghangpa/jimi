"""M8: Funding Rate Module."""

import numpy as np


def score_m8_funding(funding_rate, direction, config):
    """Score funding rate as directional signal."""
    if funding_rate is None or np.isnan(funding_rate):
        return 'SKIP', 0.5, {}

    high = config.get('M8_HIGH_FUNDING', 0.05)
    low = config.get('M8_LOW_FUNDING', -0.05)
    details = {'funding_rate': funding_rate}

    if funding_rate > high:
        score = min(0.3 + (funding_rate - high) * 5, 0.2)
        agrees = direction == 'SHORT'
        details['bias'] = 'SHORTS_FAVORED'
        return ('PASS' if agrees else 'FAIL'), score, details

    elif funding_rate < low:
        agrees = direction == 'LONG'
        details['bias'] = 'LONGS_FAVORED'
        if direction == 'LONG':
            return ('PASS', 0.8, details)
        else:
            return ('FAIL', 0.2, details)

    else:
        details['bias'] = 'NEUTRAL'
        return 'PASS', 0.5, details

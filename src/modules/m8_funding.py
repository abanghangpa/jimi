"""M8: Funding Rate Module — granular classification with squeeze detection."""

import numpy as np


def score_m8_funding(funding_rate, direction, config):
    """Score funding rate with granular thresholds and squeeze risk.

    Funding rate thresholds (per 8h):
        > +0.10%:  DANGER_LONG   — no new longs regardless of macro
        +0.05-0.10%: HIGH_LONG  — reduce long size 50%
        -0.01-+0.05%: NEUTRAL   — normal sizing
        -0.03--0.01%: MILD_SHORT — add 20% to long size on bullish macro
        < -0.03%:  SQUEEZE_SETUP — add 50% to long size on bullish macro
        < -0.05%:  EXTREME_SHORT — max long size on bullish macro

    Funding is a position SIZER, not a direction indicator.
    Direction comes from other modules (M1, M3, M4, M13).
    Funding modulates how aggressively you trade that direction.
    """
    if funding_rate is None or np.isnan(funding_rate):
        return 'SKIP', 0.5, {}

    high = config.get('M8_HIGH_FUNDING', 0.05)
    low = config.get('M8_LOW_FUNDING', -0.05)
    danger = config.get('M8_DANGER_FUNDING', 0.10)
    squeeze = config.get('M8_SQUEEZE_FUNDING', -0.03)
    extreme = config.get('M8_EXTREME_FUNDING', -0.05)

    details = {'funding_rate': round(funding_rate, 6)}

    # ── Granular classification ──
    if funding_rate > danger:
        classification = 'DANGER_LONG'
        long_score = 0.15  # strongly against longs
        size_mult = 0.30   # reduce long size drastically
    elif funding_rate > high:
        classification = 'HIGH_LONG'
        long_score = 0.30
        size_mult = 0.50
    elif funding_rate < extreme:
        classification = 'EXTREME_SHORT'
        long_score = 0.85  # strongly favors longs (squeeze setup)
        size_mult = 1.50
    elif funding_rate < squeeze:
        classification = 'SQUEEZE_SETUP'
        long_score = 0.75
        size_mult = 1.30
    elif funding_rate < -0.01:
        classification = 'MILD_SHORT'
        long_score = 0.60
        size_mult = 1.10
    else:
        classification = 'NEUTRAL'
        long_score = 0.50
        size_mult = 1.00

    # ── Direction-aware scoring ──
    if direction == 'LONG':
        score = long_score
    elif direction == 'SHORT':
        score = 1.0 - long_score
    else:
        score = 0.5

    details['classification'] = classification
    details['size_mult'] = round(size_mult, 2)
    details['bias'] = 'SHORTS_FAVORED' if long_score < 0.45 else 'LONGS_FAVORED' if long_score > 0.55 else 'NEUTRAL'

    # Status: PASS if funding agrees with direction, FAIL if strongly against
    if classification in ('DANGER_LONG',) and direction == 'LONG':
        status = 'FAIL'
    elif classification in ('EXTREME_SHORT',) and direction == 'SHORT':
        status = 'FAIL'
    elif classification == 'NEUTRAL':
        status = 'PASS'
    else:
        agrees = (long_score > 0.5 and direction == 'LONG') or (long_score < 0.5 and direction == 'SHORT')
        status = 'PASS' if agrees else 'WEAK'

    return status, max(0.0, min(1.0, score)), details

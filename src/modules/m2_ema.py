"""M2: Multi-Timeframe EMA Confluence."""

import numpy as np


def score_m2(df_1h, df_2h, df_4h, df_1d, idx_1h, idx_2h, idx_4h, idx_1d):
    """Score multi-TF EMA alignment."""
    macro_dir = None
    if idx_1d >= 0:
        bias = df_1d['swing_bias'].iloc[idx_1d]
        macro_dir = {'BULLISH': 'BULL', 'BEARISH': 'BEAR'}.get(bias, 'NEUTRAL')

    tf_4h = None
    if idx_4h >= 1:
        ef, es = df_4h['ema_fast'].iloc[idx_4h], df_4h['ema_slow'].iloc[idx_4h]
        tf_4h = 'BULL' if ef > es else 'BEAR' if ef < es else 'NEUTRAL'

    tf_2h = None
    if idx_2h >= 1:
        ef, es = df_2h['ema_fast'].iloc[idx_2h], df_2h['ema_slow'].iloc[idx_2h]
        tf_2h = 'BULL' if ef > es else 'BEAR' if ef < es else 'NEUTRAL'

    tf_1h = None
    if idx_1h >= 1:
        ef, es = df_1h['ema_fast'].iloc[idx_1h], df_1h['ema_slow'].iloc[idx_1h]
        tf_1h = 'BULL' if ef > es else 'BEAR' if ef < es else 'NEUTRAL'

    if macro_dir == 'NEUTRAL' and tf_4h == 'NEUTRAL':
        return 'NEUTRAL', 0.35

    direction = macro_dir if macro_dir != 'NEUTRAL' else (tf_4h if tf_4h != 'NEUTRAL' else tf_2h)
    if direction == 'NEUTRAL':
        return 'NEUTRAL', 0.40

    confirmations, layers, traps = 0, 0, 0
    if tf_4h and macro_dir != 'NEUTRAL':
        layers += 1
        if tf_4h == macro_dir or tf_4h == 'NEUTRAL':
            confirmations += 1
        else:
            traps += 1
    if tf_2h and tf_4h and tf_4h != 'NEUTRAL':
        layers += 1
        if tf_2h == tf_4h or tf_2h == 'NEUTRAL':
            confirmations += 1
        else:
            traps += 1
    if tf_1h and tf_2h and tf_2h != 'NEUTRAL':
        layers += 1
        if tf_1h == tf_2h or tf_1h == 'NEUTRAL':
            confirmations += 1
        else:
            traps += 1

    if layers == 0:
        return 'NEUTRAL', 0.40
    if traps > 0 and traps / layers >= 0.5:
        return 'FAIL', max(0.15, 0.5 - traps / layers)

    confirm_ratio = confirmations / layers
    higher_score = 0.5 + (0.25 if macro_dir == direction else 0) + (0.25 if tf_4h == direction else 0)
    final = min(higher_score * 0.5 + confirm_ratio * 0.5, 1.0)

    if final >= 0.65:
        return 'PASS', final
    elif final >= 0.35:
        return 'NEUTRAL', final
    return 'FAIL', final

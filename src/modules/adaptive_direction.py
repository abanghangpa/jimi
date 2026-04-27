"""Adaptive Direction Bias — replaces binary trend filter."""

import numpy as np


def compute_adaptive_direction(
    trend_dir, trend_score,
    ema_1h_fast, ema_1h_slow,
    ema_4h_fast, ema_4h_slow,
    ema_1d_fast, ema_1d_slow,
    vol_regime,
    recent_trades=None,
    direction='LONG',
    config=None,
):
    """Compute adaptive direction bias.

    Returns: (bias: float, allowed: bool, details: dict)
    bias: -1.0 (max short) to +1.0 (max long)
    """
    details = {}
    components = {}

    trend_map = {
        'STRONG_UP': 0.8, 'UP': 0.4, 'NEUTRAL': 0.0,
        'DOWN': -0.4, 'STRONG_DOWN': -0.8
    }
    daily_val = trend_map.get(trend_dir, 0.0)
    daily_bias = daily_val * min(abs(trend_score) / 0.5, 1.0)
    components['daily_trend'] = daily_bias

    if ema_1h_fast is not None and ema_1h_slow is not None and ema_1h_slow > 0:
        ema_1h_diff = (ema_1h_fast - ema_1h_slow) / ema_1h_slow
        ema_1h_val = np.clip(ema_1h_diff * 10, -1.0, 1.0)
    else:
        ema_1h_val = 0.0
    components['ema_1h'] = ema_1h_val

    if ema_4h_fast is not None and ema_4h_slow is not None and ema_4h_slow > 0:
        ema_4h_diff = (ema_4h_fast - ema_4h_slow) / ema_4h_slow
        ema_4h_val = np.clip(ema_4h_diff * 10, -1.0, 1.0)
    else:
        ema_4h_val = 0.0
    components['ema_4h'] = ema_4h_val

    if ema_1d_fast is not None and ema_1d_slow is not None and ema_1d_slow > 0:
        ema_1d_diff = (ema_1d_fast - ema_1d_slow) / ema_1d_slow
        ema_1d_val = np.clip(ema_1d_diff * 10, -1.0, 1.0)
    else:
        ema_1d_val = 0.0
    components['ema_1d'] = ema_1d_val

    if recent_trades and len(recent_trades) >= 3:
        last_n = recent_trades[-min(8, len(recent_trades)):]
        long_pnl = sum(t.pnl_pct * t.size_pct for t in last_n if t.direction == 'LONG')
        short_pnl = sum(t.pnl_pct * t.size_pct for t in last_n if t.direction == 'SHORT')
        momentum = np.clip((long_pnl - short_pnl) * 5, -1.0, 1.0)
    else:
        momentum = 0.0
    components['momentum'] = momentum

    bias = (
        components['daily_trend'] * 0.30 +
        components['ema_1h'] * 0.20 +
        components['ema_4h'] * 0.20 +
        components['ema_1d'] * 0.15 +
        components['momentum'] * 0.15
    )
    bias = np.clip(bias, -1.0, 1.0)

    daily_bearish = components.get('daily_trend', 0) < -0.10
    fast_turning = (components.get('ema_1h', 0) + components.get('ema_4h', 0)) / 2 > -0.05
    regime_transition = daily_bearish and fast_turning

    cfg = config or {}
    min_bias = cfg.get('ADAPTIVE_DIR_MIN_BIAS', 0.0)
    block_threshold = cfg.get('ADAPTIVE_DIR_BLOCK_THRESHOLD', 0.50)

    if direction == 'LONG':
        allowed = bias >= -min_bias
        if bias < -block_threshold:
            if regime_transition:
                allowed = bias < -(block_threshold + 0.15)
            else:
                allowed = False
    else:
        allowed = bias <= min_bias
        if bias > block_threshold:
            if regime_transition:
                allowed = bias > (block_threshold + 0.15)
            else:
                allowed = False

    details['regime_transition'] = regime_transition
    details['direction_bias'] = round(bias, 4)
    details['bias_components'] = {k: round(v, 4) for k, v in components.items()}
    details['bias_allowed'] = allowed
    return bias, allowed, details

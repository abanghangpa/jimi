"""
Wyckoff Module — Institutional Accumulation/Distribution Engine.

Wyckoff Spring/Upthrust detection with:
  1. Range Detector: ATR compression + Bollinger squeeze detection
  2. Wyckoff-aware CVD scoring: divergence only scored at range extremes
  3. Magnet direction confirmation for Spring/Upthrust
  4. Confluence gate: triple-filter (M4+M5+M7)

Key alpha:
  - Spring (Accumulation): CVD bullish divergence at range bottom with declining volume
  - Upthrust (Distribution): CVD bearish divergence at range top with declining volume
  - Mid-range divergence is noise (suppressed)
"""

import numpy as np


# Default Wyckoff config
WYCKOFF_DEFAULTS = {
    # Range Detection
    'RANGE_LOOKBACK': 48,
    'RANGE_ATR_SQUEEZE_PCTL': 0.45,
    'RANGE_MAX_WIDTH_PCT': 0.03,
    'RANGE_EXTREME_PCT': 0.15,
    # Wyckoff M4 Overrides
    'WYCKOFF_SPRING_SCORE': 0.85,
    'WYCKOFF_UPTHRUST_SCORE': 0.85,
    'WYCKOFF_LOW_VOL_BONUS': 0.05,
    'WYCKOFF_NOISE_SCORE': 0.45,
    'WYCKOFF_CONTINUATION_SCORE': 0.55,
    # Wyckoff M5 Overrides
    'WYCKOFF_MAGNET_CONFIRM_SCORE': 0.85,
    'WYCKOFF_MAGNET_AGAINST_SCORE': 0.30,
    # Confluence Gate
    'WYCKOFF_CONFLUENCE_ENABLED': True,
    'WYCKOFF_CONFLUENCE_M7_MIN': 0.55,
    'WYCKOFF_CONFLUENCE_ICS_BOOST': 0.04,
}


def compute_range_state(df_15m, lookback=None, squeeze_pctl=None, max_width=None,
                        config=None):
    """
    Detect consolidation ranges using ATR compression + price width.

    Adds columns to df_15m:
      - range_high, range_low: rolling range boundaries
      - in_range: True if in consolidation
      - range_pct: 0-1 position within range (0=bottom, 1=top)
      - range_width_pct: range width as % of price
      - atr_pctl: ATR percentile (0-1)
    """
    cfg = {**WYCKOFF_DEFAULTS, **(config or {})}
    lookback = lookback or cfg['RANGE_LOOKBACK']
    squeeze_pctl = squeeze_pctl or cfg['RANGE_ATR_SQUEEZE_PCTL']
    max_width = max_width or cfg['RANGE_MAX_WIDTH_PCT']

    atr = df_15m['atr']

    # ATR percentile over rolling 90-day window (12960 bars at 15m)
    pctl_window = min(12960, len(df_15m))
    atr_pctl = atr.rolling(pctl_window, min_periods=100).apply(
        lambda x: (x.iloc[-1] - x.min()) / (x.max() - x.min()) if x.max() > x.min() else 0.5,
        raw=False
    )

    rolling_high = df_15m['High'].rolling(lookback).max()
    rolling_low = df_15m['Low'].rolling(lookback).min()
    range_width = (rolling_high - rolling_low) / rolling_low

    in_range = (atr_pctl < squeeze_pctl) & (range_width < max_width)

    range_span = (rolling_high - rolling_low).replace(0, np.nan)
    range_pct = (df_15m['Close'] - rolling_low) / range_span
    range_pct = range_pct.clip(0, 1)

    df_15m['range_high'] = rolling_high
    df_15m['range_low'] = rolling_low
    df_15m['in_range'] = in_range
    df_15m['range_pct'] = range_pct
    df_15m['range_width_pct'] = range_width
    df_15m['atr_pctl'] = atr_pctl

    return df_15m


def score_wyckoff(df_15m, df_2h, idx_15m, idx_2h, direction, config=None):
    """
    Wyckoff-aware scoring: CVD divergence at range extremes = institutional signal.

    Spring (Accumulation):
      - Price near range bottom (range_pct < extreme_zone)
      - CVD bullish divergence
      - Volume declining = supply exhausted
      → STRONG LONG signal

    Upthrust (Distribution):
      - Price near range top (range_pct > 1 - extreme_zone)
      - CVD bearish divergence
      - Volume declining = demand exhausted
      → STRONG SHORT signal

    Returns:
        (status, score, details)
        status: 'PASS' or 'FAIL'
        score: 0.0 to 1.0
        details: dict with wyckoff_phase, layer scores, etc.
    """
    cfg = {**WYCKOFF_DEFAULTS, **(config or {})}
    extreme_zone = cfg['RANGE_EXTREME_PCT']

    layer_a_score = 0.5
    layer_a_status = 'FAIL'
    layer_a_div = 'NONE'
    wyckoff_phase = 'NONE'

    if idx_15m >= cfg.get('CVD_LOOKBACK', 36):
        for ci in range(max(0, idx_15m - 5), idx_15m + 1):
            div = df_15m['cvd_divergence_15m'].iloc[ci]
            if (direction == 'LONG' and div == 'BULLISH') or \
               (direction == 'SHORT' and div == 'BEARISH'):

                in_range = df_15m['in_range'].iloc[idx_15m] if 'in_range' in df_15m.columns else False
                range_pct = df_15m['range_pct'].iloc[idx_15m] if 'range_pct' in df_15m.columns else 0.5
                vol_ratio = df_15m['vol_ratio'].iloc[idx_15m] if 'vol_ratio' in df_15m.columns else 1.0

                if div == 'BULLISH':
                    if in_range and range_pct < extreme_zone:
                        layer_a_score = cfg['WYCKOFF_SPRING_SCORE']
                        if not np.isnan(vol_ratio) and vol_ratio < 0.8:
                            layer_a_score += cfg['WYCKOFF_LOW_VOL_BONUS']
                        wyckoff_phase = 'SPRING'
                    elif in_range and range_pct > (1 - extreme_zone):
                        layer_a_score = cfg['WYCKOFF_CONTINUATION_SCORE']
                        wyckoff_phase = 'CONTINUATION'
                    else:
                        layer_a_score = cfg['WYCKOFF_NOISE_SCORE']
                        wyckoff_phase = 'NOISE'

                elif div == 'BEARISH':
                    if in_range and range_pct > (1 - extreme_zone):
                        layer_a_score = cfg['WYCKOFF_UPTHRUST_SCORE']
                        if not np.isnan(vol_ratio) and vol_ratio < 0.8:
                            layer_a_score += cfg['WYCKOFF_LOW_VOL_BONUS']
                        wyckoff_phase = 'UPTHRUST'
                    elif in_range and range_pct < extreme_zone:
                        layer_a_score = cfg['WYCKOFF_CONTINUATION_SCORE']
                        wyckoff_phase = 'CONTINUATION'
                    else:
                        layer_a_score = cfg['WYCKOFF_NOISE_SCORE']
                        wyckoff_phase = 'NOISE'

                layer_a_score = min(layer_a_score, 1.0)
                layer_a_status = 'PASS'
                layer_a_div = div
                break

    # Layer B: 2H CVD Zero-Line Cross (standard M4 logic)
    layer_b_score = 0.0
    layer_b_status = 'FAIL'
    layer_b_cross = 'NONE'
    layer_b_bars_since = 999
    zl_state = 'NONE'

    if idx_2h >= 1 and 'cvd_zl_state' in df_2h.columns:
        zl_state = df_2h['cvd_zl_state'].iloc[idx_2h]
        cross_bar = df_2h['cvd_zl_cross_bar'].iloc[idx_2h]
        cross_dir = df_2h['cvd_zl_cross_dir'].iloc[idx_2h]
        cvd_2h_now = df_2h['cvd_2h'].iloc[idx_2h]

        if not np.isnan(cvd_2h_now):
            bars_since = idx_2h - cross_bar if cross_bar >= 0 else 999
            layer_b_bars_since = bars_since
            zl_momentum_bars = cfg.get('M4_ZL_MOMENTUM_BARS', 8)
            zl_lookback = cfg.get('M4_ZL_LOOKBACK', 18)
            fresh = bars_since <= zl_momentum_bars

            if direction == 'LONG':
                if zl_state == 'CROSS_UP':
                    layer_b_score = 0.90 if fresh else 0.70
                    layer_b_status = 'PASS'
                    layer_b_cross = 'CROSS_UP'
                elif zl_state == 'ABOVE' and cross_dir == 'UP':
                    if bars_since <= zl_momentum_bars:
                        layer_b_score = 0.80
                        layer_b_status = 'PASS'
                        layer_b_cross = 'ABOVE_FRESH'
                    elif bars_since <= zl_lookback:
                        layer_b_score = 0.65
                        layer_b_status = 'PASS'
                        layer_b_cross = 'ABOVE_AFTER_UP'
                    else:
                        layer_b_score = 0.50
                        layer_b_status = 'PASS'
                        layer_b_cross = 'ABOVE_STALE'
                elif zl_state == 'ABOVE':
                    layer_b_score = 0.40
                    layer_b_status = 'PASS'
                    layer_b_cross = 'ABOVE_NO_CROSS'

            elif direction == 'SHORT':
                if zl_state == 'CROSS_DOWN':
                    layer_b_score = 0.90 if fresh else 0.70
                    layer_b_status = 'PASS'
                    layer_b_cross = 'CROSS_DOWN'
                elif zl_state == 'BELOW' and cross_dir == 'DOWN':
                    if bars_since <= zl_momentum_bars:
                        layer_b_score = 0.80
                        layer_b_status = 'PASS'
                        layer_b_cross = 'BELOW_FRESH'
                    elif bars_since <= zl_lookback:
                        layer_b_score = 0.65
                        layer_b_status = 'PASS'
                        layer_b_cross = 'BELOW_AFTER_DOWN'
                    else:
                        layer_b_score = 0.50
                        layer_b_status = 'PASS'
                        layer_b_cross = 'BELOW_STALE'
                elif zl_state == 'BELOW':
                    layer_b_score = 0.40
                    layer_b_status = 'PASS'
                    layer_b_cross = 'BELOW_NO_CROSS'

            if direction == 'LONG' and zl_state in ('BELOW', 'CROSS_DOWN'):
                if layer_b_status != 'PASS':
                    layer_b_score = 0.20
                    layer_b_cross = f'CONFLICT_{zl_state}'
            elif direction == 'SHORT' and zl_state in ('ABOVE', 'CROSS_UP'):
                if layer_b_status != 'PASS':
                    layer_b_score = 0.20
                    layer_b_cross = f'CONFLICT_{zl_state}'

            if layer_b_status == 'PASS' and idx_2h >= 3:
                cvd_slope_2h = (df_2h['cvd_2h'].iloc[idx_2h] - df_2h['cvd_2h'].iloc[max(0, idx_2h - 3)]) / 3
                if not np.isnan(cvd_slope_2h):
                    if (direction == 'LONG' and cvd_slope_2h > 0) or \
                       (direction == 'SHORT' and cvd_slope_2h < 0):
                        layer_b_score = min(layer_b_score * 1.15, 1.0)

    # Composite: Wyckoff Layer A dominates when Spring/Upthrust
    if wyckoff_phase in ('SPRING', 'UPTHRUST'):
        composite = layer_a_score * 0.70 + layer_b_score * 0.30
    else:
        composite = layer_a_score * 0.50 + layer_b_score * 0.50

    composite = max(0.0, min(1.0, composite))
    status = 'PASS' if composite >= 0.50 else 'FAIL'

    details = {
        'layer_a_score': round(layer_a_score, 3),
        'layer_a_div': layer_a_div,
        'layer_b_score': round(layer_b_score, 3),
        'layer_b_cross': layer_b_cross,
        'zl_state': zl_state,
        'wyckoff_phase': wyckoff_phase,
        'composite': round(composite, 3),
    }

    return status, composite, details


def check_wyckoff_confluence(m4_details, m5_details, m7_score, direction, config=None):
    """
    Triple confluence: Wyckoff Spring/Upthrust + M5 magnet confirm + M7 agree.
    Returns: (confluence_type, ics_boost)
    """
    cfg = {**WYCKOFF_DEFAULTS, **(config or {})}

    wyckoff_phase = m4_details.get('wyckoff_phase', 'NONE') if isinstance(m4_details, dict) else 'NONE'
    m5_confirm = m5_details.get('wyckoff_confirm', 'NEUTRAL') if isinstance(m5_details, dict) else 'NEUTRAL'

    if wyckoff_phase == 'SPRING' and m5_confirm == 'SPRING_CONFIRM':
        if m7_score >= cfg['WYCKOFF_CONFLUENCE_M7_MIN']:
            return 'FULL_CONFLUENCE', cfg['WYCKOFF_CONFLUENCE_ICS_BOOST']
        return 'PARTIAL_CONFLUENCE', cfg['WYCKOFF_CONFLUENCE_ICS_BOOST'] * 0.5

    elif wyckoff_phase == 'UPTHRUST' and m5_confirm == 'UPTHRUST_CONFIRM':
        if m7_score >= cfg['WYCKOFF_CONFLUENCE_M7_MIN']:
            return 'FULL_CONFLUENCE', cfg['WYCKOFF_CONFLUENCE_ICS_BOOST']
        return 'PARTIAL_CONFLUENCE', cfg['WYCKOFF_CONFLUENCE_ICS_BOOST'] * 0.5

    return 'NO_CONFLUENCE', 0.0


def is_mid_range_noise(df_15m, idx, wyckoff_phase, config=None):
    """
    Check if a signal is mid-range noise that should be blocked.
    Only blocks NOISE signals that are in a tight range but NOT at extremes.
    If not in a range at all (trending), let everything through.
    """
    cfg = {**WYCKOFF_DEFAULTS, **(config or {})}
    extreme_zone = cfg['RANGE_EXTREME_PCT']

    if wyckoff_phase != 'NOISE':
        return False

    in_range = df_15m['in_range'].iloc[idx] if 'in_range' in df_15m.columns else False
    range_pct = df_15m['range_pct'].iloc[idx] if 'range_pct' in df_15m.columns else 0.5

    return in_range and extreme_zone < range_pct < (1 - extreme_zone)

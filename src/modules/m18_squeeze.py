"""
M18: Squeeze Detector v6.0 — Box Typology + Quality Entry + Lifecycle Tracking

v6.0 changes over v5.1:
  Phase 1 — Box Detection:
    - Swing high/low fractal detection within compression window
    - Box type classification (SIDEWAYS, DESCENDING_TRIANGLE, ASCENDING_TRIANGLE,
      SYMMETRIC_TRIANGLE, BEAR_FLAG, BULL_FLAG)
    - Compression maturity tracking (how close to maximum compression)
    - Box boundary refinement using most-tested levels

  Phase 2 — Entry Trigger:
    - Volume quality scoring (spike vs sustained, expansion from prev bar)
    - Breakout bar quality (close position, wick rejection, body ratio)
    - Retest entry option (wait for pullback to broken level)
    - Taker ratio alignment on breakout

  Phase 3 — Integration:
    - Box type feeds direction bias into direction resolver
    - Squeeze lifecycle states: FORMING → TIGHTENING → MATURE → BREAKOUT → RETEST
    - Lifecycle-aware quality scoring

Path A: 15m range compression → breakout trigger
Path B: 2h MACD(8,17,9) DIF/DEA convergence → histogram flip trigger
"""

import numpy as np


SQUEEZE_V6_DEFAULTS = {
    # ── Path A: 15m compression ──
    'SQUEEZE_RANGE48_MAX': 1.2,
    'SQUEEZE_COMPRESSION_BARS_MIN': 12,
    'SQUEEZE_DRY_BARS_MIN': 4,
    'SQUEEZE_DOJI_BARS_MIN': 8,

    # Path A trigger: CVD taker
    'SQUEEZE_TAKER_LONG': 0.58,
    'SQUEEZE_TAKER_SHORT': 0.42,

    # ── Path B: 2h MACD coiling ──
    'SQUEEZE_MACD_FAST': 8,
    'SQUEEZE_MACD_SLOW': 17,
    'SQUEEZE_MACD_SIGNAL': 9,
    'SQUEEZE_COIL_DELTA_MAX': 0.05,
    'SQUEEZE_COIL_BARS_MIN': 6,
    'SQUEEZE_COIL_BARS_STRONG': 9,
    'SQUEEZE_HIST_FLIP_THRESHOLD': 0.0,

    # ── Entry filters ──
    'SQUEEZE_REQUIRE_HIST_FLIP': True,
    'SQUEEZE_EMA_FILTER': False,
    'SQUEEZE_MIN_RSI': None,
    'SQUEEZE_MAX_RSI': None,

    # ── Entry trigger ──
    'SQUEEZE_ENTRY_BUFFER_PCT': 0.001,
    'SQUEEZE_ENTRY_EXPIRY_BARS': 32,
    'SQUEEZE_BREAKOUT_VOL_MULT': 1.0,
    'SQUEEZE_ENTRY_MODE': 'TWO_BAR',
    'SQUEEZE_COIL_HOURS_MIN': 12,

    # ── Box typology (Phase 1) ──
    'SQUEEZE_SWING_PERIOD': 3,           # bars each side for fractal detection
    'SQUEEZE_SLOPE_THRESHOLD': 0.0008,   # min slope (as fraction of price) to count as trending
    'SQUEEZE_CONVERGENCE_MIN': 0.3,      # min convergence ratio (range shrinking vs initial)
    'SQUEEZE_MATURITY_WINDOW': 8,        # bars to measure compression maturity
    'SQUEEZE_MATURITY_THRESHOLD': 1.15,  # range_current / range_min below this = mature

    # ── Breakout bar quality (Phase 2) ──
    'SQUEEZE_BREAKOUT_CLOSE_PCT': 0.75,  # close must be in top/bottom 25% of bar range
    'SQUEEZE_BREAKOUT_BODY_MIN': 0.50,   # body must be > 50% of bar range
    'SQUEEZE_BREAKOUT_WICK_MAX': 0.30,   # rejection wick must be < 30% of bar range
    'SQUEEZE_VOL_EXPANSION_MIN': 1.3,    # breakout bar vol / prev bar vol
    'SQUEEZE_TAKER_ALIGN': True,         # require taker ratio to align with direction

    # ── Retest entry (Phase 2) ──
    'SQUEEZE_RETEST_ENABLED': True,
    'SQUEEZE_RETEST_BARS': 8,            # max bars to wait for retest after breakout
    'SQUEEZE_RETEST_TOLERANCE': 0.002,   # how close price must come to broken level (0.2%)
    'SQUEEZE_RETEST_HOLD_BARS': 2,       # bars price must hold above/below level after retest

    # ── Exit levels ──
    'SQUEEZE_TP_ATR_MULT': 2.5,
    'SQUEEZE_SL_ATR_MULT': 1.0,
    'SQUEEZE_TP_MIN_PCT': 0.3,
    'SQUEEZE_TP_MAX_PCT': 2.0,

    # ── Override ──
    'SQUEEZE_OVERRIDE_REGIME': True,
    'SQUEEZE_ICS_BOOST': 0.10,
    'SQUEEZE_SIZE_MULT': 0.80,

    # ── Failed Breakout Detection ──
    'SQUEEZE_FAILED_BREAKOUT_LOOKBACK': 16,
    'SQUEEZE_FAILED_BREAKOUT_REJECT_PCT': 0.003,
    'SQUEEZE_FAILED_BREAKOUT_PENALTY': 0.5,

    # ── Cooldown ──
    'SQUEEZE_COOLDOWN_BARS': 32,
    'SQUEEZE_MAX_PENDING': 1,
}

# Backward compat alias
SQUEEZE_V5_DEFAULTS = SQUEEZE_V6_DEFAULTS


# ═══════════════════════════════════════════════════════════════
# PHASE 1: BOX TYPOLOGY
# ═══════════════════════════════════════════════════════════════

def _detect_fractals(highs, lows, period=3):
    """Detect swing highs and swing lows using N-bar fractal method.

    A swing high at index i: highs[i] is the max of highs[i-period : i+period+1]
    A swing low at index i: lows[i] is the min of lows[i-period : i+period+1]

    Returns:
        swing_highs: list of (index, price)
        swing_lows: list of (index, price)
    """
    n = len(highs)
    swing_highs = []
    swing_lows = []

    for i in range(period, n - period):
        # Swing high
        is_sh = True
        for j in range(i - period, i + period + 1):
            if j != i and highs[j] >= highs[i]:
                is_sh = False
                break
        if is_sh:
            swing_highs.append((i, float(highs[i])))

        # Swing low
        is_sl = True
        for j in range(i - period, i + period + 1):
            if j != i and lows[j] <= lows[i]:
                is_sl = False
                break
        if is_sl:
            swing_lows.append((i, float(lows[i])))

    return swing_highs, swing_lows


def _fit_slope(points):
    """Fit linear regression to swing points.

    Args:
        points: list of (index, price)

    Returns:
        slope: price change per bar (normalized by mean price)
        r_squared: goodness of fit
    """
    if len(points) < 2:
        return 0.0, 0.0

    indices = np.array([p[0] for p in points], dtype=float)
    prices = np.array([p[1] for p in points], dtype=float)
    mean_price = np.mean(prices)

    if mean_price <= 0:
        return 0.0, 0.0

    # Normalize slope as fraction of mean price per bar
    n = len(indices)
    x_mean = np.mean(indices)
    y_mean = mean_price

    ss_xy = np.sum((indices - x_mean) * (prices - y_mean))
    ss_xx = np.sum((indices - x_mean) ** 2)

    if ss_xx == 0:
        return 0.0, 0.0

    slope = ss_xy / ss_xx  # price per bar
    slope_normalized = slope / mean_price  # fraction of price per bar

    # R-squared
    y_pred = slope * (indices - x_mean) + y_mean
    ss_res = np.sum((prices - y_pred) ** 2)
    ss_tot = np.sum((prices - y_mean) ** 2)
    r_squared = 1 - ss_res / ss_tot if ss_tot > 0 else 0.0

    return slope_normalized, max(0.0, r_squared)


def classify_box_type(highs, lows, closes, lookback_bars, cfg):
    """Classify the compression box type from swing structure.

    Args:
        highs, lows, closes: numpy arrays of OHLC data
        lookback_bars: how many bars back to analyze
        cfg: config dict

    Returns:
        dict with:
            box_type: str (SIDEWAYS, DESCENDING_TRIANGLE, ASCENDING_TRIANGLE,
                      SYMMETRIC_TRIANGLE, BEAR_FLAG, BULL_FLAG, UNKNOWN)
            high_slope: float (normalized slope of swing highs)
            low_slope: float (normalized slope of swing lows)
            high_r2: float
            low_r2: float
            convergence: float (0-1, how much range has narrowed)
            swing_highs: list of (idx, price)
            swing_lows: list of (idx, price)
            bias: str (LONG, SHORT, NEUTRAL)
    """
    period = cfg.get('SQUEEZE_SWING_PERIOD', 3)
    slope_thresh = cfg.get('SQUEEZE_SLOPE_THRESHOLD', 0.0008)

    start = max(0, len(highs) - lookback_bars)
    h = highs[start:].astype(float)
    l = lows[start:].astype(float)
    c = closes[start:].astype(float)

    if len(h) < period * 2 + 3:
        return _unknown_box()

    swing_highs, swing_lows = _detect_fractals(h, l, period)

    # Need at least 2 points on each side for slope
    if len(swing_highs) < 2 or len(swing_lows) < 2:
        return _unknown_box_with_swings(swing_highs, swing_lows)

    high_slope, high_r2 = _fit_slope(swing_highs)
    low_slope, low_r2 = _fit_slope(swing_lows)

    # Convergence: compare range at start vs end of the swing sequence
    first_half_end = len(h) // 2
    first_highs = [p for i, p in swing_highs if i < first_half_end]
    first_lows = [p for i, p in swing_lows if i < first_half_end]
    second_highs = [p for i, p in swing_highs if i >= first_half_end]
    second_lows = [p for i, p in swing_lows if i >= first_half_end]

    convergence = 0.5  # default
    if first_highs and first_lows and second_highs and second_lows:
        range_first = max(first_highs) - min(first_lows)
        range_second = max(second_highs) - min(second_lows)
        if range_first > 0:
            convergence = 1.0 - (range_second / range_first)
            convergence = max(0.0, min(1.0, convergence))

    # Classify based on slopes
    high_trending = abs(high_slope) > slope_thresh
    low_trending = abs(low_slope) > slope_thresh

    # Determine trend context from recent price action
    recent_trend = 0.0
    if len(c) >= 20:
        recent_trend = (c[-1] - c[-20]) / c[-20] if c[-20] > 0 else 0

    box_type = 'UNKNOWN'
    bias = 'NEUTRAL'

    if not high_trending and not low_trending:
        # Both flat → sideways range
        box_type = 'SIDEWAYS'
        bias = 'NEUTRAL'
    elif high_trending and not low_trending:
        if high_slope < -slope_thresh:
            box_type = 'DESCENDING_TRIANGLE'
            bias = 'SHORT'
        elif high_slope > slope_thresh:
            # Rising highs but flat lows — unusual, treat as ascending variant
            box_type = 'ASCENDING_TRIANGLE'
            bias = 'LONG'
    elif not high_trending and low_trending:
        if low_slope > slope_thresh:
            box_type = 'ASCENDING_TRIANGLE'
            bias = 'LONG'
        elif low_slope < -slope_thresh:
            box_type = 'DESCENDING_TRIANGLE'
            bias = 'SHORT'
    elif high_trending and low_trending:
        # Both moving — check if converging
        if high_slope < 0 and low_slope > 0:
            # Converging from both sides
            box_type = 'SYMMETRIC_TRIANGLE'
            bias = 'NEUTRAL'  # direction determined by breakout
        elif high_slope < 0 and low_slope < 0:
            # Both falling
            if recent_trend < -0.02:
                box_type = 'BEAR_FLAG'
                bias = 'SHORT'
            else:
                box_type = 'DESCENDING_TRIANGLE'
                bias = 'SHORT'
        elif high_slope > 0 and low_slope > 0:
            # Both rising
            if recent_trend > 0.02:
                box_type = 'BULL_FLAG'
                bias = 'LONG'
            else:
                box_type = 'ASCENDING_TRIANGLE'
                bias = 'LONG'
        elif high_slope > 0 and low_slope < 0:
            # Diverging — expanding range, not a squeeze
            box_type = 'EXPANDING'
            bias = 'NEUTRAL'

    return {
        'box_type': box_type,
        'high_slope': round(high_slope, 6),
        'low_slope': round(low_slope, 6),
        'high_r2': round(high_r2, 3),
        'low_r2': round(low_r2, 3),
        'convergence': round(convergence, 3),
        'swing_highs': swing_highs,
        'swing_lows': swing_lows,
        'bias': bias,
        'n_swings_h': len(swing_highs),
        'n_swings_l': len(swing_lows),
    }


def _unknown_box():
    return {
        'box_type': 'UNKNOWN', 'high_slope': 0, 'low_slope': 0,
        'high_r2': 0, 'low_r2': 0, 'convergence': 0,
        'swing_highs': [], 'swing_lows': [], 'bias': 'NEUTRAL',
        'n_swings_h': 0, 'n_swings_l': 0,
    }


def _unknown_box_with_swings(sh, sl):
    return {
        'box_type': 'UNKNOWN', 'high_slope': 0, 'low_slope': 0,
        'high_r2': 0, 'low_r2': 0, 'convergence': 0,
        'swing_highs': sh, 'swing_lows': sl, 'bias': 'NEUTRAL',
        'n_swings_h': len(sh), 'n_swings_l': len(sl),
    }


def compute_compression_maturity(range_series, current_idx, window=8):
    """Measure how close current compression is to maximum tightness.

    Returns:
        maturity: float (0-1, 1 = at maximum compression)
        range_ratio: current_range / min_range over window (1.0 = at minimum)
        min_range: the tightest range in the window
    """
    if current_idx < window or len(range_series) < window:
        return 0.5, 1.0, 0.0

    window_data = range_series[max(0, current_idx - window + 1):current_idx + 1]
    window_data = [r for r in window_data if r > 0]

    if not window_data:
        return 0.5, 1.0, 0.0

    current_range = window_data[-1]
    min_range = min(window_data)
    max_range = max(window_data)

    if max_range == min_range:
        return 1.0, 1.0, min_range

    # Maturity: how close current is to minimum (1.0 = at min, 0.0 = at max)
    maturity = 1.0 - (current_range - min_range) / (max_range - min_range)

    # Range ratio: current / min (1.0 = at minimum compression)
    range_ratio = current_range / min_range if min_range > 0 else 999.0

    return round(maturity, 3), round(range_ratio, 3), round(min_range, 4)


def refine_coil_boundaries(highs, lows, closes, lookback_bars, percentile_low=20, percentile_high=80):
    """Refine coil boundaries using most-tested price levels.

    Uses percentile-based approach (existing) but also finds the most-tested
    high and low — the levels where price reversed multiple times.

    Returns:
        coil_high, coil_low, tested_high, tested_low
    """
    start = max(0, len(highs) - lookback_bars)
    h = highs[start:].astype(float)
    l = lows[start:].astype(float)

    if len(h) < 3:
        return float(h[-1]), float(l[-1]), float(h[-1]), float(l[-1])

    # Percentile boundaries (existing approach)
    p_high = float(np.percentile(h, percentile_high))
    p_low = float(np.percentile(l, percentile_low))

    # Most-tested boundaries: find levels with most reversals
    # Bin the high/low values and count frequency
    price_range = p_high - p_low
    if price_range <= 0:
        return p_high, p_low, p_high, p_low

    n_bins = min(20, max(5, int(len(h) / 3)))
    bin_edges = np.linspace(p_low, p_high, n_bins + 1)

    # Count how many times price reversed at each bin (high touched, then fell)
    high_hist, _ = np.histogram(h, bins=bin_edges)
    low_hist, _ = np.histogram(l, bins=bin_edges)

    # Most-tested high: bin with most swing highs
    high_bin_idx = np.argmax(high_hist)
    tested_high = (bin_edges[high_bin_idx] + bin_edges[high_bin_idx + 1]) / 2

    # Most-tested low: bin with most swing lows
    low_bin_idx = np.argmax(low_hist)
    tested_low = (bin_edges[low_bin_idx] + bin_edges[low_bin_idx + 1]) / 2

    return p_high, p_low, round(float(tested_high), 2), round(float(tested_low), 2)


# ═══════════════════════════════════════════════════════════════
# PHASE 2: ENTRY TRIGGER QUALITY
# ═══════════════════════════════════════════════════════════════

def score_breakout_bar(bar_open, bar_high, bar_low, bar_close, bar_vol,
                        prev_vol, taker_ratio, direction, cfg):
    """Score the quality of a breakout bar.

    Checks:
    1. Close position: close in top/bottom 25% of bar range
    2. Body ratio: body > 50% of range (not a doji)
    3. Wick rejection: rejection wick < 30% of range
    4. Volume expansion: breakout bar vol / prev bar vol > threshold
    5. Taker alignment: buyers driving LONG, sellers driving SHORT

    Returns:
        score: float (0-1, higher = better quality)
        passed: bool (True if all critical checks pass)
        factors: list of str (what passed/failed)
    """
    cfg = cfg or {}
    factors = []
    score = 0.0

    bar_range = bar_high - bar_low
    if bar_range <= 0:
        return 0.0, False, ['doji bar (zero range)']

    body = abs(bar_close - bar_open)
    body_ratio = body / bar_range

    # Close position within bar
    if direction == 'LONG':
        close_position = (bar_close - bar_low) / bar_range
        rejection_wick = (bar_high - bar_close) / bar_range
    else:
        close_position = (bar_high - bar_close) / bar_range
        rejection_wick = (bar_close - bar_low) / bar_range

    # 1. Close position (must be in top/bottom 25%)
    close_threshold = cfg.get('SQUEEZE_BREAKOUT_CLOSE_PCT', 0.75)
    close_ok = close_position >= close_threshold
    if close_ok:
        score += 0.25
        factors.append(f'close position {close_position:.0%} >= {close_threshold:.0%}')
    else:
        factors.append(f'❌ close position {close_position:.0%} < {close_threshold:.0%}')

    # 2. Body ratio (must be > 50% of range)
    body_min = cfg.get('SQUEEZE_BREAKOUT_BODY_MIN', 0.50)
    body_ok = body_ratio >= body_min
    if body_ok:
        score += 0.20
        factors.append(f'body ratio {body_ratio:.0%} >= {body_min:.0%}')
    else:
        factors.append(f'❌ body ratio {body_ratio:.0%} < {body_min:.0%} (doji breakout)')

    # 3. Wick rejection (must be < 30% of range)
    wick_max = cfg.get('SQUEEZE_BREAKOUT_WICK_MAX', 0.30)
    wick_ok = rejection_wick <= wick_max
    if wick_ok:
        score += 0.15
        factors.append(f'wick rejection {rejection_wick:.0%} <= {wick_max:.0%}')
    else:
        factors.append(f'❌ wick rejection {rejection_wick:.0%} > {wick_max:.0%} (shooting star)')

    # 4. Volume expansion
    vol_expansion = bar_vol / prev_vol if prev_vol > 0 else 1.0
    vol_min = cfg.get('SQUEEZE_VOL_EXPANSION_MIN', 1.3)
    vol_ok = vol_expansion >= vol_min
    if vol_ok:
        score += 0.25
        factors.append(f'vol expansion {vol_expansion:.2f}x >= {vol_min:.1f}x')
    else:
        factors.append(f'❌ vol expansion {vol_expansion:.2f}x < {vol_min:.1f}x')

    # 5. Taker alignment
    taker_align = cfg.get('SQUEEZE_TAKER_ALIGN', True)
    if taker_align:
        if direction == 'LONG' and taker_ratio > 0.52:
            score += 0.15
            factors.append(f'taker aligned: {taker_ratio:.3f} (buyers)')
        elif direction == 'SHORT' and taker_ratio < 0.48:
            score += 0.15
            factors.append(f'taker aligned: {taker_ratio:.3f} (sellers)')
        else:
            factors.append(f'❌ taker not aligned: {taker_ratio:.3f}')
    else:
        score += 0.15  # give credit if check disabled

    passed = close_ok and body_ok and wick_ok
    # Vol expansion is important but not critical for passed flag
    # (can be a slow breakout that accelerates)

    return round(score, 3), passed, factors


def check_retest_entry(df_15m, current_idx, breakout_level, direction, cfg):
    """Check if price is retesting the broken coil level after breakout.

    A retest entry is when:
    1. Price broke out (closed beyond coil level)
    2. Price pulled back to the level
    3. Price held (didn't close back inside the coil)

    This is a higher-quality entry than raw breakout because it filters
    false breakouts that immediately reverse.

    Returns:
        retest_detected: bool
        retest_quality: float (0-1)
        details: str
    """
    cfg = cfg or {}
    if not cfg.get('SQUEEZE_RETEST_ENABLED', True):
        return False, 0.0, 'retest disabled'

    retest_bars = cfg.get('SQUEEZE_RETEST_BARS', 8)
    tolerance = cfg.get('SQUEEZE_RETEST_TOLERANCE', 0.002)
    hold_bars = cfg.get('SQUEEZE_RETEST_HOLD_BARS', 2)

    if current_idx < retest_bars + hold_bars:
        return False, 0.0, 'insufficient bars'

    # Scan backwards from current bar for breakout + retest pattern
    buffer = breakout_level * tolerance

    # Phase 1: Find the breakout bar (first close beyond level)
    breakout_bar = -1
    for i in range(current_idx - hold_bars, max(0, current_idx - retest_bars - hold_bars), -1):
        close_i = float(df_15m['Close'].iloc[i])
        if direction == 'LONG' and close_i > breakout_level + buffer:
            breakout_bar = i
            break
        elif direction == 'SHORT' and close_i < breakout_level - buffer:
            breakout_bar = i
            break

    if breakout_bar < 0:
        return False, 0.0, 'no breakout found in window'

    # Phase 2: Check if price pulled back to the level after breakout
    retest_bar = -1
    for i in range(breakout_bar + 1, current_idx - hold_bars + 1):
        low_i = float(df_15m['Low'].iloc[i])
        high_i = float(df_15m['High'].iloc[i])

        if direction == 'LONG':
            # Retest: price dipped back near the breakout level
            if low_i <= breakout_level + buffer:
                retest_bar = i
                break
        else:
            # Retest: price bounced back up near the breakout level
            if high_i >= breakout_level - buffer:
                retest_bar = i
                break

    if retest_bar < 0:
        return False, 0.0, 'no retest after breakout'

    # Phase 3: Check if price held after retest (didn't close back inside coil)
    for i in range(retest_bar, current_idx + 1):
        close_i = float(df_15m['Close'].iloc[i])
        if direction == 'LONG' and close_i < breakout_level:
            return False, 0.0, f'retest failed: closed back inside at bar {i}'
        elif direction == 'SHORT' and close_i > breakout_level:
            return False, 0.0, f'retest failed: closed back inside at bar {i}'

    # Quality: closer retest = better (but not through the level)
    retest_low = float(df_15m['Low'].iloc[retest_bar])
    retest_high = float(df_15m['High'].iloc[retest_bar])

    if direction == 'LONG':
        distance = (retest_low - breakout_level) / breakout_level
    else:
        distance = (breakout_level - retest_high) / breakout_level

    # Closer retest = higher quality (but capped at tolerance)
    quality = min(1.0, max(0.3, 1.0 - distance / tolerance)) if tolerance > 0 else 0.5

    bars_since_breakout = retest_bar - breakout_bar
    bars_held = current_idx - retest_bar

    return True, round(quality, 3), (
        f'retest at ${breakout_level:.2f} ({bars_since_breakout} bars after breakout, '
        f'held {bars_held} bars, quality={quality:.2f})'
    )


# ═══════════════════════════════════════════════════════════════
# PHASE 3: SQUEEZE LIFECYCLE
# ═══════════════════════════════════════════════════════════════

def determine_lifecycle_stage(compression_bars, maturity, range_ratio,
                                entry_triggered, retest_detected, box_type):
    """Determine which stage of the squeeze lifecycle we're in.

    Stages:
      FORMING     — compression just started, range still wide
      TIGHTENING  — range actively narrowing, not yet mature
      MATURE      — maximum compression reached, ready for breakout
      BREAKOUT    — price broke out of the coil
      RETEST      — price retesting broken level (best entry)
      EXPIRED     — breakout failed or coil dissolved

    Returns:
        stage: str
        stage_score: float (0-1, higher = better stage for entry)
    """
    if entry_triggered and retest_detected:
        return 'RETEST', 0.95
    if entry_triggered:
        return 'BREAKOUT', 0.70

    if compression_bars < 6:
        return 'FORMING', 0.20

    if range_ratio <= 1.15 and maturity >= 0.7:
        return 'MATURE', 0.90

    if range_ratio <= 1.3 and maturity >= 0.5:
        return 'TIGHTENING', 0.60

    if range_ratio > 1.5:
        return 'FORMING', 0.20

    return 'TIGHTENING', 0.50


def format_box_type(box_info):
    """Format box type information for terminal output."""
    if not box_info or box_info.get('box_type') == 'UNKNOWN':
        return ''

    icons = {
        'SIDEWAYS': '↔️',
        'DESCENDING_TRIANGLE': '🔻',
        'ASCENDING_TRIANGLE': '🔺',
        'SYMMETRIC_TRIANGLE': '🔻🔺',
        'BEAR_FLAG': '🚩↓',
        'BULL_FLAG': '🚩↑',
        'EXPANDING': '↕️',
    }
    icon = icons.get(box_info['box_type'], '❓')

    lines = [f'  Box Type: {icon} {box_info["box_type"]}']
    lines.append(f'    High slope: {box_info["high_slope"]:+.6f}/bar (R²={box_info["high_r2"]:.2f})')
    lines.append(f'    Low slope:  {box_info["low_slope"]:+.6f}/bar (R²={box_info["low_r2"]:.2f})')
    lines.append(f'    Convergence: {box_info["convergence"]:.0%}')
    lines.append(f'    Swings: {box_info["n_swings_h"]}H / {box_info["n_swings_l"]}L')
    if box_info['bias'] != 'NEUTRAL':
        lines.append(f'    Bias: {box_info["bias"]}')

    return '\n'.join(lines)


# ═══════════════════════════════════════════════════════════════
# EXISTING HELPERS (Path A / Path B)
# ═══════════════════════════════════════════════════════════════

def _check_compression(range48, compression_history, cfg):
    """Check if market is in 15m compression state (Path A)."""
    if range48 >= cfg['SQUEEZE_RANGE48_MAX']:
        return False, 0, 0, 0

    compression_bars = 0
    dry_count = 0
    doji_count = 0
    streak = 0
    gap_count = 0

    for r48, vr, br, tr in compression_history:
        if r48 < cfg['SQUEEZE_RANGE48_MAX']:
            streak += 1
            gap_count = 0
        elif gap_count < 2:
            gap_count += 1
            streak += 1
        else:
            compression_bars = max(compression_bars, streak)
            streak = 0
            gap_count = 0
        if vr < 0.6:
            dry_count += 1
        if br < 0.20:
            doji_count += 1

    compression_bars = max(compression_bars, streak)

    if range48 < cfg['SQUEEZE_RANGE48_MAX']:
        compression_bars += 1

    is_compressed = (
        compression_bars >= cfg['SQUEEZE_COMPRESSION_BARS_MIN'] and
        dry_count >= cfg['SQUEEZE_DRY_BARS_MIN'] and
        doji_count >= cfg['SQUEEZE_DOJI_BARS_MIN']
    )

    return is_compressed, compression_bars, dry_count, doji_count


def _compute_2h_macd(df_15m, cfg):
    """Resample 15m data to 2h and compute MACD(8,17,9)."""
    if len(df_15m) < 40:
        return None

    df_copy = df_15m.copy()
    if 'Open time' in df_copy.columns:
        df_copy = df_copy.set_index('Open time')

    agg_dict = {
        'Open': 'first', 'High': 'max', 'Low': 'min', 'Close': 'last',
        'Volume': 'sum',
    }
    if 'Taker buy base asset volume' in df_copy.columns:
        agg_dict['Taker buy base asset volume'] = 'sum'

    df_2h = df_copy.resample('2h').agg(agg_dict).dropna(subset=['Open'])

    if len(df_2h) < cfg['SQUEEZE_MACD_SLOW'] + cfg['SQUEEZE_MACD_SIGNAL']:
        return None

    close = df_2h['Close']
    ema_fast = close.ewm(span=cfg['SQUEEZE_MACD_FAST'], adjust=False).mean()
    ema_slow = close.ewm(span=cfg['SQUEEZE_MACD_SLOW'], adjust=False).mean()
    dif = ema_fast - ema_slow
    dea = dif.ewm(span=cfg['SQUEEZE_MACD_SIGNAL'], adjust=False).mean()
    hist = dif - dea

    return {
        'close': close.values,
        'high': df_2h['High'].values,
        'low': df_2h['Low'].values,
        'dif': dif.values,
        'dea': dea.values,
        'hist': hist.values,
        'timestamps': df_2h.index,
        'prices': close.values,
    }


def _detect_macd_coil(macd_data, cfg):
    """Detect 2h MACD DIF/DEA coiling (Path B)."""
    dif = macd_data['dif']
    dea = macd_data['dea']
    hist = macd_data['hist']
    close = macd_data['close']
    high = macd_data['high']
    low = macd_data['low']
    ts = macd_data['timestamps']

    n = len(dif)
    if n < 3:
        return False, 0, False, 'NEUTRAL', {}

    delta_pct = np.abs(dif - dea) / np.where(close > 0, close, 1.0) * 100

    coil_threshold = cfg['SQUEEZE_COIL_DELTA_MAX']
    min_bars = cfg['SQUEEZE_COIL_BARS_MIN']
    coil_bars = 0
    gap_count = 0
    max_gaps = 2

    for i in range(n - 1, -1, -1):
        if delta_pct[i] < coil_threshold:
            coil_bars += 1
            gap_count = 0
        elif gap_count < max_gaps:
            gap_count += 1
            coil_bars += 1
        else:
            break

    max_streak = 0
    streak = 0
    for i in range(n):
        if delta_pct[i] < coil_threshold:
            streak += 1
            max_streak = max(max_streak, streak)
        else:
            streak = 0

    is_coiled = coil_bars >= min_bars

    if coil_bars > 0:
        coil_start = max(0, n - coil_bars)
        coil_high = float(np.max(high[coil_start:n]))
        coil_low = float(np.min(low[coil_start:n]))
    else:
        coil_high = float(close[-1])
        coil_low = float(close[-1])

    hist_flip = False
    flip_direction = 'NEUTRAL'

    if n >= 2:
        prev_hist = hist[-2]
        curr_hist = hist[-1]
        if prev_hist < 0 and curr_hist >= 0:
            hist_flip = True
            flip_direction = 'LONG'
        elif prev_hist > 0 and curr_hist <= 0:
            hist_flip = True
            flip_direction = 'SHORT'

    hist_expanding = False
    if n >= 3:
        prev_abs = abs(hist[-2])
        curr_abs = abs(hist[-1])
        if prev_abs < 0.5 and curr_abs > 1.0:
            hist_expanding = True

    direction = 'NEUTRAL'
    if coil_bars >= 2:
        dif_slope = dif[-1] - dif[-(min(coil_bars, 6))]
        if dif_slope > 0.5:
            direction = 'LONG'
        elif dif_slope < -0.5:
            direction = 'SHORT'

    if hist_flip:
        direction = flip_direction

    details = {
        'coil_bars': coil_bars,
        'max_streak': max_streak,
        'coil_hours': coil_bars * 2,
        'max_streak_hours': max_streak * 2,
        'delta_pct_current': round(float(delta_pct[-1]), 4),
        'delta_pct_avg': round(float(np.mean(delta_pct[-coil_bars:])) if coil_bars > 0 else 999, 4),
        'hist_value': round(float(hist[-1]), 3),
        'hist_prev': round(float(hist[-2]) if n >= 2 else 0, 3),
        'hist_flip': hist_flip,
        'hist_expanding': hist_expanding,
        'flip_direction': flip_direction,
        'dif_current': round(float(dif[-1]), 3),
        'dea_current': round(float(dea[-1]), 3),
        'coil_high': round(coil_high, 2),
        'coil_low': round(coil_low, 2),
        'timestamp': str(ts[-1]) if n > 0 else '',
    }

    return is_coiled, coil_bars, hist_flip, direction, details


def _compute_coil_range_15m(df_15m, comp_bars, current_idx):
    """Compute the 15m compression range for Path A."""
    if comp_bars <= 0 or current_idx < comp_bars:
        return None, None

    start = max(0, current_idx - comp_bars)
    highs = df_15m['High'].iloc[start:current_idx+1].values.astype(float)
    lows = df_15m['Low'].iloc[start:current_idx+1].values.astype(float)
    h = float(np.percentile(highs, 80))
    l = float(np.percentile(lows, 20))
    return h, l


def _detect_failed_breakout(df_15m, current_idx, coil_high, coil_low, direction,
                             lookback_bars=16, rejection_pct=0.003):
    """Check if price already tested the coil boundary and reversed."""
    if coil_high <= 0 or coil_low <= 0 or current_idx < lookback_bars:
        return False, ''

    start = max(0, current_idx - lookback_bars)
    coil_width = coil_high - coil_low
    if coil_width <= 0:
        return False, ''

    for i in range(start, current_idx):
        bar_high = float(df_15m['High'].iloc[i])
        bar_low = float(df_15m['Low'].iloc[i])
        bar_close = float(df_15m['Close'].iloc[i])

        if direction == 'LONG':
            penetration = (bar_high - coil_high) / coil_width
            if penetration >= rejection_pct and bar_close < coil_high:
                bars_ago = current_idx - i
                return True, f'failed_breakout_up {bars_ago}bars ago (spike to ${bar_high:.2f}, close ${bar_close:.2f})'
        elif direction == 'SHORT':
            penetration = (coil_low - bar_low) / coil_width
            if penetration >= rejection_pct and bar_close > coil_low:
                bars_ago = current_idx - i
                return True, f'failed_breakout_down {bars_ago}bars ago (spike to ${bar_low:.2f}, close ${bar_close:.2f})'

    return False, ''


def _check_entry_trigger(current_close, current_vol, vol_ma20,
                          coil_high, coil_low, direction, cfg):
    """Check if entry trigger fires (price breaks out of coil range)."""
    buffer = cfg['SQUEEZE_ENTRY_BUFFER_PCT']
    vol_mult = cfg['SQUEEZE_BREAKOUT_VOL_MULT']

    vol_confirmed = current_vol >= vol_ma20 * vol_mult if vol_ma20 > 0 else True

    if direction == 'LONG':
        breakout_level = coil_high * (1 + buffer)
        if current_close > breakout_level and vol_confirmed:
            return True, breakout_level, f"enter when 15m closes above ${breakout_level:.2f} (✅ CONFIRMED)"
        else:
            return False, breakout_level, f"enter when 15m closes above ${breakout_level:.2f} (⏳ waiting)"

    elif direction == 'SHORT':
        breakout_level = coil_low * (1 - buffer)
        if current_close < breakout_level and vol_confirmed:
            return True, breakout_level, f"enter when 15m closes below ${breakout_level:.2f} (✅ CONFIRMED)"
        else:
            return False, breakout_level, f"enter when 15m closes below ${breakout_level:.2f} (⏳ waiting)"

    return False, current_close, "no direction"


def _check_two_bar_trigger(prev_close, current_close, current_vol, vol_ma20,
                            coil_high, coil_low, direction, cfg):
    """Check if 2-bar confirmation trigger fires."""
    buffer = cfg['SQUEEZE_ENTRY_BUFFER_PCT']
    vol_mult = cfg['SQUEEZE_BREAKOUT_VOL_MULT']
    vol_confirmed = current_vol >= vol_ma20 * vol_mult if vol_ma20 > 0 else True

    if direction == 'LONG':
        breakout_level = coil_high * (1 + buffer)
        if current_close > breakout_level and prev_close > breakout_level:
            return True, breakout_level, f"2-bar close above ${breakout_level:.2f} (✅ CONFIRMED)"
        elif current_close > breakout_level:
            return False, breakout_level, f"2-bar: 1/2 closes above ${breakout_level:.2f} (⏳ need 2nd)"
        else:
            return False, breakout_level, f"2-bar: enter when 2 consecutive closes above ${breakout_level:.2f} (⏳ waiting)"

    elif direction == 'SHORT':
        breakout_level = coil_low * (1 - buffer)
        if current_close < breakout_level and prev_close < breakout_level:
            return True, breakout_level, f"2-bar close below ${breakout_level:.2f} (✅ CONFIRMED)"
        elif current_close < breakout_level:
            return False, breakout_level, f"2-bar: 1/2 closes below ${breakout_level:.2f} (⏳ need 2nd)"
        else:
            return False, breakout_level, f"2-bar: enter when 2 consecutive closes below ${breakout_level:.2f} (⏳ waiting)"

    return False, current_close, "no direction"


def _check_cvd_trigger(taker_ratio, pre_taker_avg, cfg):
    """Check if CVD trigger fires (taker spike during compression)."""
    if taker_ratio >= cfg['SQUEEZE_TAKER_LONG']:
        return True, 'LONG', 'TAKER_SPIKE'
    if taker_ratio <= cfg['SQUEEZE_TAKER_SHORT']:
        return True, 'SHORT', 'TAKER_SPIKE'

    if pre_taker_avg is not None:
        if taker_ratio > 0.52 and pre_taker_avg < 0.48:
            return True, 'LONG', 'TAKER_SHIFT'
        if taker_ratio < 0.48 and pre_taker_avg > 0.52:
            return True, 'SHORT', 'TAKER_SHIFT'

    return False, 'NEUTRAL', 'NONE'


def _score_quality(path_a, path_b, cfg, box_info=None, maturity=0.5,
                    breakout_quality=None, lifecycle_stage='FORMING'):
    """Score squeeze quality based on all factors.

    v6: Now includes box type, maturity, breakout quality, and lifecycle stage.
    """
    quality = 0.0
    factors = []
    is_strong = False

    a_compressed, a_comp_bars, a_dry, a_doji = path_a
    b_coiled, b_coil_bars, b_hist_flip, b_direction, b_details = path_b

    # Path A scoring
    if a_compressed:
        a_score = 0.0
        if a_comp_bars >= 24:
            a_score += 0.20
            factors.append(f'15m deep compression: {a_comp_bars} bars')
        elif a_comp_bars >= 12:
            a_score += 0.12
            factors.append(f'15m compression: {a_comp_bars} bars')
        a_score += min(0.10, a_dry / 12 * 0.10)
        a_score += min(0.08, a_doji / 12 * 0.08)
        quality += a_score

    # Path B scoring
    if b_coiled:
        b_score = 0.0
        coil_hours = b_details.get('coil_hours', 0)

        if coil_hours >= 24:
            b_score += 0.25
            factors.append(f'2h MACD coil: {coil_hours}h (DEEP)')
        elif coil_hours >= 18:
            b_score += 0.18
            factors.append(f'2h MACD coil: {coil_hours}h (STRONG)')
        elif coil_hours >= 12:
            b_score += 0.10
            factors.append(f'2h MACD coil: {coil_hours}h')

        if b_details.get('max_streak_hours', 0) >= 24:
            b_score += 0.08
            factors.append(f'Max coil streak: {b_details["max_streak_hours"]}h')

        if b_hist_flip:
            b_score += 0.15
            factors.append(f'2h histogram flip → {b_direction}')

        if b_details.get('hist_expanding'):
            b_score += 0.08
            factors.append('2h histogram expanding')

        quality += b_score

    # Dual-path agreement bonus
    if a_compressed and b_coiled:
        quality += 0.10
        factors.append('DUAL PATH: 15m + 2h both compressed')
        if b_hist_flip:
            is_strong = True
            factors.append('STRONG: dual compression + histogram flip')

    # ── Phase 1 bonus: Box type quality ──
    if box_info and box_info.get('box_type') not in ('UNKNOWN', 'EXPANDING'):
        box_type = box_info['box_type']
        convergence = box_info.get('convergence', 0)

        # Higher quality for well-defined patterns
        if box_type in ('DESCENDING_TRIANGLE', 'ASCENDING_TRIANGLE'):
            quality += 0.08
            factors.append(f'Box: {box_type} (directional bias)')
        elif box_type == 'SYMMETRIC_TRIANGLE':
            quality += 0.05
            factors.append(f'Box: {box_type} (converging)')
        elif box_type in ('BEAR_FLAG', 'BULL_FLAG'):
            quality += 0.06
            factors.append(f'Box: {box_type} (continuation)')

        if convergence > 0.5:
            quality += 0.05
            factors.append(f'High convergence: {convergence:.0%}')

    # ── Phase 1 bonus: Compression maturity ──
    if maturity >= 0.8:
        quality += 0.08
        factors.append(f'Compression MATURE: {maturity:.0%}')
    elif maturity >= 0.6:
        quality += 0.04
        factors.append(f'Compression tightening: {maturity:.0%}')

    # ── Phase 2 bonus: Breakout quality ──
    if breakout_quality is not None:
        bq_score, bq_passed, bq_factors = breakout_quality
        if bq_passed:
            quality += 0.10
            factors.append(f'Breakout quality: PASS ({bq_score:.2f})')
        elif bq_score > 0.5:
            quality += 0.04
            factors.append(f'Breakout quality: WEAK ({bq_score:.2f})')
        # Include detailed factors
        for bf in bq_factors:
            if '❌' in bf:
                factors.append(f'  {bf}')

    # ── Phase 3: Lifecycle stage bonus ──
    lifecycle_bonuses = {
        'RETEST': 0.10,
        'MATURE': 0.06,
        'BREAKOUT': 0.03,
        'TIGHTENING': 0.02,
        'FORMING': 0.0,
    }
    lc_bonus = lifecycle_bonuses.get(lifecycle_stage, 0)
    if lc_bonus > 0:
        quality += lc_bonus
        factors.append(f'Lifecycle: {lifecycle_stage}')

    if quality >= 0.65:
        is_strong = True

    quality = min(quality, 1.0)

    return quality, is_strong, factors


# ═══════════════════════════════════════════════════════════════
# MAIN DETECTION FUNCTION
# ═══════════════════════════════════════════════════════════════

def detect_squeeze_v6(result, config=None, last_signal_bar=-1, current_bar=0,
                       compression_history=None, df_15m=None,
                       magnets=None, liq_levels=None, sr_levels=None):
    """v6 squeeze detection with box typology, quality entry, and lifecycle tracking.

    Returns dict with:
        squeeze_type, squeeze_status (PENDING/TRIGGERED/NONE),
        entry_price, entry_condition, coil_high, coil_low,
        box_type, box_bias, lifecycle_stage, breakout_quality, ...
    """
    cfg = {**SQUEEZE_V6_DEFAULTS, **(config or {})}

    # Cooldown
    if current_bar - last_signal_bar < cfg['SQUEEZE_COOLDOWN_BARS']:
        return _empty_result('cooldown')

    # Extract features
    price = result.get('price', 0)
    atr = result.get('atr', 0)
    range48 = result.get('range_width', 5)
    taker_ratio = result.get('raw_taker_ratio', 0.5)
    vol = result.get('vol_trend', 1.0)

    # ── PATH A: 15m Compression ──
    a_compressed, a_comp_bars, a_dry, a_doji = _check_compression(
        range48, compression_history or [], cfg)

    # ── PATH B: 2h MACD Coiling ──
    b_coiled = False
    b_coil_bars = 0
    b_hist_flip = False
    b_direction = 'NEUTRAL'
    b_details = {}

    if df_15m is not None:
        macd_data = _compute_2h_macd(df_15m, cfg)
        if macd_data is not None:
            b_coiled, b_coil_bars, b_hist_flip, b_direction, b_details = \
                _detect_macd_coil(macd_data, cfg)

    # ── Check if either path detected a squeeze ──
    path_a_fired = a_compressed
    path_b_fired = b_coiled

    if not path_a_fired and not path_b_fired:
        reasons = []
        if not a_compressed:
            reasons.append(f'15m: range48={range48:.2f}% comp={a_comp_bars}')
        if not b_coiled:
            reasons.append(f'2h: coil={b_coil_bars} bars')
        return _empty_result('; '.join(reasons))

    # ═══════════════════════════════════════════════════════════
    # PHASE 1: BOX TYPOLOGY
    # ═══════════════════════════════════════════════════════════
    box_info = _unknown_box()
    if df_15m is not None and (path_a_fired or path_b_fired):
        lookback = max(a_comp_bars, b_coil_bars * 4, 48)  # convert 2h bars to 15m bars
        lookback = min(lookback, len(df_15m) - 1)
        if lookback >= 12:
            highs = df_15m['High'].values
            lows = df_15m['Low'].values
            closes = df_15m['Close'].values
            box_info = classify_box_type(highs, lows, closes, lookback, cfg)

    # Compression maturity
    maturity = 0.5
    range_ratio = 1.0
    if compression_history and len(compression_history) >= 4:
        range_series = [r for r, _, _, _ in compression_history]
        maturity, range_ratio, _ = compute_compression_maturity(
            range_series, len(range_series) - 1,
            window=cfg.get('SQUEEZE_MATURITY_WINDOW', 8))

    # Refine coil boundaries
    coil_high_pct = b_details.get('coil_high', price)
    coil_low_pct = b_details.get('coil_low', price)
    tested_high = coil_high_pct
    tested_low = coil_low_pct

    if path_a_fired and df_15m is not None:
        a_high, a_low = _compute_coil_range_15m(df_15m, a_comp_bars, len(df_15m) - 1)
        if a_high is not None and a_low is not None:
            if (a_high - a_low) < (coil_high_pct - coil_low_pct):
                coil_high_pct = a_high
                coil_low_pct = a_low

    if df_15m is not None and (path_a_fired or path_b_fired):
        lookback_for_refine = max(a_comp_bars, b_coil_bars * 4, 48)
        lookback_for_refine = min(lookback_for_refine, len(df_15m) - 1)
        if lookback_for_refine >= 12:
            _, _, tested_high, tested_low = refine_coil_boundaries(
                df_15m['High'].values, df_15m['Low'].values,
                df_15m['Close'].values, lookback_for_refine)

    # Use tested boundaries if they're tighter
    if 0 < tested_high < coil_high_pct and tested_high > coil_low_pct:
        coil_high_pct = tested_high
    if 0 < tested_low > coil_low_pct and tested_low < coil_high_pct:
        coil_low_pct = tested_low

    # ── CVD Trigger (for Path A) ──
    pre_taker_avg = None
    if compression_history and len(compression_history) >= 4:
        pre_takers = [tr for _, _, _, tr in compression_history[-12:]]
        pre_taker_avg = np.mean(pre_takers) if pre_takers else None

    cvd_triggered, cvd_direction, cvd_type = _check_cvd_trigger(
        taker_ratio, pre_taker_avg, cfg)

    # ── Resolve Direction ──
    direction = 'NEUTRAL'
    trigger_type = 'NONE'

    # Path B histogram flip is the strongest signal
    if path_b_fired and b_hist_flip:
        direction = b_direction
        trigger_type = 'HIST_FLIP'
    # Path A with CVD confirmation
    elif path_a_fired and cvd_triggered:
        direction = cvd_direction
        trigger_type = cvd_type
    # Path B coil without flip — direction from DIF slope
    elif path_b_fired and b_direction != 'NEUTRAL':
        direction = b_direction
        trigger_type = 'COIL_DIR'

    # Phase 3: Box type bias can inform direction when other signals are weak
    if direction == 'NEUTRAL' and box_info.get('bias') != 'NEUTRAL':
        box_bias = box_info['bias']
        box_type = box_info['box_type']
        if box_type in ('DESCENDING_TRIANGLE', 'ASCENDING_TRIANGLE',
                         'BEAR_FLAG', 'BULL_FLAG'):
            direction = box_bias
            trigger_type = 'BOX_BIAS'

    if direction == 'NEUTRAL':
        return _empty_result(f'no direction: cvd={cvd_type} hist_flip={b_hist_flip} box={box_info.get("box_type", "?")}')

    # ── Coil Duration Filter ──
    coil_hours = b_details.get('coil_hours', 0)
    coil_hours_min = cfg.get('SQUEEZE_COIL_HOURS_MIN', 12)
    if coil_hours < coil_hours_min and not a_compressed:
        return _empty_result(f'coil={coil_hours}h < {coil_hours_min}h minimum')

    # ── v5.1 Filters ──
    if cfg.get('SQUEEZE_REQUIRE_HIST_FLIP', False) and trigger_type != 'HIST_FLIP':
        return _empty_result(f'trigger={trigger_type} (need HIST_FLIP)')

    if cfg.get('SQUEEZE_EMA_FILTER', False) and df_15m is not None and len(df_15m) >= 55:
        close_series = df_15m['Close'] if 'Close' in df_15m.columns else df_15m.iloc[:, 4]
        ema21 = float(close_series.ewm(span=21, adjust=False).mean().iloc[-1])
        ema55 = float(close_series.ewm(span=55, adjust=False).mean().iloc[-1])
        ema_spread = (ema21 - ema55) / ema55 * 100 if ema55 > 0 else 0
        ema_trend = 'BULL' if ema21 > ema55 else 'BEAR'

        contrarian = (direction == 'LONG' and ema_trend == 'BEAR') or \
                     (direction == 'SHORT' and ema_trend == 'BULL')
        aligned = not contrarian

        if ema_spread < 0:
            if not contrarian:
                return _empty_result(f'EMA aligned in bear trend (need contra): dir={direction} spread={ema_spread:+.2f}%')
        else:
            if not aligned:
                return _empty_result(f'EMA contra in bull trend (need aligned): dir={direction} spread={ema_spread:+.2f}%')

    if cfg.get('SQUEEZE_MIN_RSI') is not None and df_15m is not None and len(df_15m) >= 14:
        if 'rsi' in df_15m.columns and not np.isnan(df_15m['rsi'].iloc[-1]):
            rsi = float(df_15m['rsi'].iloc[-1])
        else:
            close_series = df_15m['Close'] if 'Close' in df_15m.columns else df_15m.iloc[:, 4]
            delta = close_series.diff()
            gain = delta.clip(lower=0).rolling(14).mean()
            loss = (-delta.clip(upper=0)).rolling(14).mean()
            rsi = 100 - (100 / (1 + gain.iloc[-1] / loss.iloc[-1])) if loss.iloc[-1] > 0 else 50

        if direction == 'LONG' and rsi > cfg['SQUEEZE_MAX_RSI']:
            return _empty_result(f'RSI={rsi:.0f} overbought for LONG')
        if direction == 'SHORT' and rsi < cfg['SQUEEZE_MIN_RSI']:
            return _empty_result(f'RSI={rsi:.0f} oversold for SHORT')

    # ── Failed Breakout Detection ──
    failed_breakout = False
    failed_breakout_detail = ''
    if df_15m is not None and len(df_15m) > 0:
        _fb_idx = len(df_15m) - 1
        _fb_lookback = cfg.get('SQUEEZE_FAILED_BREAKOUT_LOOKBACK', 16)
        _fb_reject_pct = cfg.get('SQUEEZE_FAILED_BREAKOUT_REJECT_PCT', 0.003)
        failed_breakout, failed_breakout_detail = _detect_failed_breakout(
            df_15m, _fb_idx, coil_high_pct, coil_low_pct, direction,
            lookback_bars=_fb_lookback, rejection_pct=_fb_reject_pct)

    # ═══════════════════════════════════════════════════════════
    # PHASE 2: ENTRY TRIGGER WITH QUALITY
    # ═══════════════════════════════════════════════════════════
    vol_ma20 = result.get('vol_ma20', vol * 20)
    entry_mode = cfg.get('SQUEEZE_ENTRY_MODE', 'TWO_BAR')

    # Raw breakout check (does price break the level?)
    raw_breakout = False
    entry_price_raw = price
    entry_condition_raw = ''

    if entry_mode == 'TWO_BAR':
        prev_close = float(df_15m['Close'].iloc[-2]) if df_15m is not None and len(df_15m) >= 2 else price
        raw_breakout, entry_price_raw, entry_condition_raw = _check_two_bar_trigger(
            prev_close, price, price * vol,
            vol_ma20 if vol_ma20 > 0 else price,
            coil_high_pct, coil_low_pct, direction, cfg)
    else:
        raw_breakout, entry_price_raw, entry_condition_raw = _check_entry_trigger(
            price, price * vol, vol_ma20 if vol_ma20 > 0 else price,
            coil_high_pct, coil_low_pct, direction, cfg)

    # Breakout bar quality scoring (Phase 2)
    breakout_quality = None
    breakout_quality_passed = False
    if raw_breakout and df_15m is not None and len(df_15m) >= 2:
        idx = len(df_15m) - 1
        bar_o = float(df_15m['Open'].iloc[idx])
        bar_h = float(df_15m['High'].iloc[idx])
        bar_l = float(df_15m['Low'].iloc[idx])
        bar_c = float(df_15m['Close'].iloc[idx])
        bar_v = float(df_15m['Volume'].iloc[idx])
        prev_v = float(df_15m['Volume'].iloc[idx - 1])

        breakout_quality = score_breakout_bar(
            bar_o, bar_h, bar_l, bar_c, bar_v, prev_v, taker_ratio, direction, cfg)
        _, breakout_quality_passed, _ = breakout_quality

    # Retest entry check (Phase 2)
    retest_detected = False
    retest_quality = 0.0
    retest_detail = ''
    if df_15m is not None:
        retest_detected, retest_quality, retest_detail = check_retest_entry(
            df_15m, len(df_15m) - 1, entry_price_raw, direction, cfg)

    # Final entry decision
    entry_triggered = False
    entry_price = entry_price_raw
    entry_condition = entry_condition_raw

    if raw_breakout:
        if breakout_quality_passed:
            entry_triggered = True
            entry_condition = f"breakout + quality PASS: {entry_condition_raw}"
        elif retest_detected:
            # Retest after a weak breakout is still valid
            entry_triggered = True
            entry_condition = f"retest entry: {retest_detail}"
        else:
            # Weak breakout without retest — still trigger but note quality
            entry_triggered = True
            entry_condition = f"breakout (weak quality): {entry_condition_raw}"
    elif retest_detected:
        # Retest without raw breakout on current bar — check if breakout happened earlier
        entry_triggered = True
        entry_condition = f"retest entry: {retest_detail}"

    # ── Lifecycle Stage (Phase 3) ──
    lifecycle_stage, lifecycle_score = determine_lifecycle_stage(
        a_comp_bars, maturity, range_ratio,
        entry_triggered, retest_detected, box_info.get('box_type', 'UNKNOWN'))

    # ── Squeeze Status ──
    if entry_triggered:
        squeeze_status = 'TRIGGERED'
    else:
        squeeze_status = 'PENDING'

    # ── Quality Score (v6 — includes all phases) ──
    quality, is_strong, factors = _score_quality(
        (a_compressed, a_comp_bars, a_dry, a_doji),
        (b_coiled, b_coil_bars, b_hist_flip, b_direction, b_details),
        cfg,
        box_info=box_info,
        maturity=maturity,
        breakout_quality=breakout_quality,
        lifecycle_stage=lifecycle_stage,
    )

    if failed_breakout:
        quality *= cfg.get('SQUEEZE_FAILED_BREAKOUT_PENALTY', 0.5)
        is_strong = False
        factors.append(f'⚠️ FAILED_BREAKOUT: {failed_breakout_detail}')

    factors.append(f'Trigger: {trigger_type} → {direction}')
    if cvd_triggered and path_b_fired:
        factors.append(f'CVD confirms: {cvd_direction}')

    # ── Squeeze Type ──
    squeeze_type = 'SHORT_SQUEEZE' if direction == 'LONG' else 'LONG_SQUEEZE'

    # ── TP/SL Levels ──
    # Always use trigger entry for SL/TP — even when PENDING.
    # Using current price when pending creates SL above entry (LONG) or below entry (SHORT).
    entry_price_for_levels = entry_price
    tp1_source = 'ATR'

    if magnets is not None and liq_levels is not None:
        try:
            from src.sl_tp import calc_trade_levels
            vol_ratio = result.get('vol_ratio', 1.0)
            _liq_for_tp = liq_levels if isinstance(liq_levels, dict) else None
            _levels = calc_trade_levels(
                entry_price_for_levels, direction, atr, vol_ratio,
                magnets=magnets,
                sr_levels=sr_levels or [],
                liq_levels=_liq_for_tp,
                cfg=cfg,
            )
            if _levels.get('tp1_source') == 'UNSWEPT_POOL':
                _pool_tp_dist = abs(_levels['tp1'] - entry_price_for_levels)
                _pool_sl_dist = abs(entry_price_for_levels - _levels['sl'])
                # Compute ATR fallback for comparison
                _atr_tp = atr * cfg.get('SQUEEZE_TP_ATR_MULT', 2.5) if atr > 0 else 0
                _atr_sl = atr * cfg.get('SQUEEZE_SL_ATR_MULT', 1.0) if atr > 0 else 0
                # Pick UNSWEPT_POOL only if R:R is better than ATR fallback
                _pool_rr = _pool_tp_dist / _pool_sl_dist if _pool_sl_dist > 0 else 0
                _atr_rr = _atr_tp / _atr_sl if _atr_sl > 0 else 0
                if _pool_rr >= _atr_rr and _pool_rr >= 0.5:
                    tp_dist = _pool_tp_dist
                    sl_dist = _pool_sl_dist
                    tp1_source = 'UNSWEPT_POOL'
                else:
                    raise ValueError(f'UNSWEPT_POOL R:R {_pool_rr:.2f} worse than ATR {_atr_rr:.2f}')
            else:
                raise ValueError('ATR fallback')
        except Exception:
            if atr > 0 and entry_price_for_levels > 0:
                tp_dist = max(
                    atr * cfg['SQUEEZE_TP_ATR_MULT'],
                    entry_price_for_levels * cfg['SQUEEZE_TP_MIN_PCT'] / 100
                )
                tp_dist = min(tp_dist, entry_price_for_levels * cfg['SQUEEZE_TP_MAX_PCT'] / 100)
                sl_dist = atr * cfg['SQUEEZE_SL_ATR_MULT']
            else:
                tp_dist = entry_price_for_levels * 0.5 / 100
                sl_dist = entry_price_for_levels * 0.2 / 100
    else:
        if atr > 0 and entry_price_for_levels > 0:
            tp_dist = max(
                atr * cfg['SQUEEZE_TP_ATR_MULT'],
                entry_price_for_levels * cfg['SQUEEZE_TP_MIN_PCT'] / 100
            )
            tp_dist = min(tp_dist, entry_price_for_levels * cfg['SQUEEZE_TP_MAX_PCT'] / 100)
            sl_dist = atr * cfg['SQUEEZE_SL_ATR_MULT']
        else:
            tp_dist = entry_price_for_levels * 0.5 / 100
            sl_dist = entry_price_for_levels * 0.2 / 100

    # Scale TP by coil duration
    if coil_hours >= 18:
        tp_duration_mult = 1.0 + min((coil_hours - 18) / 36, 1.0)
    elif coil_hours >= 12:
        tp_duration_mult = 1.0 + (coil_hours - 12) / 18 * 0.3
    else:
        tp_duration_mult = 1.0

    tp_dist *= tp_duration_mult

    # Measured move floor
    if coil_high_pct > 0 and coil_low_pct > 0:
        coil_width = abs(coil_high_pct - coil_low_pct)
        measured_move_floor = coil_width * 0.3
        if tp_dist < measured_move_floor:
            tp_dist = measured_move_floor

    if entry_price_for_levels > 0:
        tp_dist = min(tp_dist, entry_price_for_levels * cfg['SQUEEZE_TP_MAX_PCT'] / 100)

    if direction == 'LONG':
        tp = entry_price_for_levels + tp_dist
        sl = entry_price_for_levels - sl_dist
    else:
        tp = entry_price_for_levels - tp_dist
        sl = entry_price_for_levels + sl_dist

    tp_pct = tp_dist / entry_price_for_levels * 100 if entry_price_for_levels > 0 else 0
    sl_pct = sl_dist / entry_price_for_levels * 100 if entry_price_for_levels > 0 else 0

    return {
        'squeeze_type': squeeze_type,
        'squeeze_status': squeeze_status,
        'squeeze_score': round(quality, 3),
        'squeeze_strong': is_strong,
        'direction': direction,
        'factors': factors,
        'quality': round(quality, 3),
        'compression_bars': a_comp_bars,
        'dry_count': a_dry,
        'doji_count': a_doji,
        'trigger_type': trigger_type,
        'overrides_regime': cfg['SQUEEZE_OVERRIDE_REGIME'],
        'ics_boost': cfg['SQUEEZE_ICS_BOOST'],
        'size_mult': cfg['SQUEEZE_SIZE_MULT'],
        # Entry trigger
        'entry_price': round(entry_price, 2),
        'entry_condition': entry_condition,
        'entry_triggered': entry_triggered,
        'coil_high': round(coil_high_pct, 2),
        'coil_low': round(coil_low_pct, 2),
        # Levels
        'tp': round(tp, 2),
        'sl': round(sl, 2),
        'tp_pct': round(tp_pct, 3),
        'sl_pct': round(sl_pct, 3),
        'tp1_source': tp1_source,
        'tp_duration_mult': round(tp_duration_mult, 3),
        'coil_hours': coil_hours,
        # Phase 1: Box typology
        'box_type': box_info.get('box_type', 'UNKNOWN'),
        'box_bias': box_info.get('bias', 'NEUTRAL'),
        'box_high_slope': box_info.get('high_slope', 0),
        'box_low_slope': box_info.get('low_slope', 0),
        'box_convergence': box_info.get('convergence', 0),
        'box_n_swings': f"{box_info.get('n_swings_h', 0)}H/{box_info.get('n_swings_l', 0)}L",
        'compression_maturity': maturity,
        'compression_range_ratio': range_ratio,
        'tested_high': tested_high,
        'tested_low': tested_low,
        # Phase 2: Entry quality
        'breakout_quality_score': breakout_quality[0] if breakout_quality else None,
        'breakout_quality_passed': breakout_quality_passed,
        'breakout_quality_factors': breakout_quality[2] if breakout_quality else [],
        'retest_detected': retest_detected,
        'retest_quality': retest_quality,
        'retest_detail': retest_detail,
        # Phase 3: Lifecycle
        'lifecycle_stage': lifecycle_stage,
        'lifecycle_score': lifecycle_score,
        # Failed breakout
        'failed_breakout': failed_breakout,
        'failed_breakout_detail': failed_breakout_detail,
        # Meta
        'gates_all_pass': True,
        'gates_passed': factors,
        'gates_failed': [],
        'short_score': round(quality, 3) if squeeze_type == 'SHORT_SQUEEZE' else 0,
        'long_score': round(quality, 3) if squeeze_type == 'LONG_SQUEEZE' else 0,
        'path_a': {
            'fired': path_a_fired,
            'compressed': a_compressed,
            'comp_bars': a_comp_bars,
            'dry': a_dry,
            'doji': a_doji,
        },
        'path_b': {
            'fired': path_b_fired,
            'coiled': b_coiled,
            'coil_bars': b_coil_bars,
            'coil_hours': b_details.get('coil_hours', 0),
            'max_streak_hours': b_details.get('max_streak_hours', 0),
            'hist_flip': b_hist_flip,
            'hist_value': b_details.get('hist_value', 0),
            'delta_pct': b_details.get('delta_pct_current', 0),
            'direction': b_direction,
            'dif': b_details.get('dif_current', 0),
            'dea': b_details.get('dea_current', 0),
        },
        # Box details (for formatting)
        '_box_info': box_info,
    }


# Backward compat
detect_squeeze_v5 = detect_squeeze_v6


def _empty_result(reason):
    """Return empty squeeze result."""
    return {
        'squeeze_type': 'NONE', 'squeeze_status': 'NONE',
        'squeeze_score': 0, 'squeeze_strong': False,
        'direction': 'NEUTRAL', 'factors': [], 'quality': 0,
        'compression_bars': 0, 'dry_count': 0, 'doji_count': 0,
        'trigger_type': 'NONE',
        'overrides_regime': False, 'ics_boost': 0, 'size_mult': 1.0,
        'entry_price': 0, 'entry_condition': '', 'entry_triggered': False,
        'coil_high': 0, 'coil_low': 0,
        'tp': 0, 'sl': 0, 'tp_pct': 0, 'sl_pct': 0,
        'tp1_source': 'ATR', 'tp_duration_mult': 1.0, 'coil_hours': 0,
        # Phase 1
        'box_type': 'UNKNOWN', 'box_bias': 'NEUTRAL',
        'box_high_slope': 0, 'box_low_slope': 0,
        'box_convergence': 0, 'box_n_swings': '0H/0L',
        'compression_maturity': 0, 'compression_range_ratio': 1.0,
        'tested_high': 0, 'tested_low': 0,
        # Phase 2
        'breakout_quality_score': None, 'breakout_quality_passed': False,
        'breakout_quality_factors': [],
        'retest_detected': False, 'retest_quality': 0, 'retest_detail': '',
        # Phase 3
        'lifecycle_stage': 'NONE', 'lifecycle_score': 0,
        # Failed
        'failed_breakout': False, 'failed_breakout_detail': '',
        'gates_all_pass': False, 'gates_passed': [], 'gates_failed': [reason],
        'short_score': 0, 'long_score': 0,
        'path_a': {'fired': False}, 'path_b': {'fired': False},
        '_box_info': _unknown_box(),
    }


def format_squeeze(sq):
    """Format squeeze result for terminal output."""
    if sq.get('squeeze_type', 'NONE') == 'NONE':
        failed = sq.get('gates_failed', [])
        if failed and failed[0] != 'cooldown':
            lines = ['', '  🔥 SQUEEZE DETECTOR — GATE FAILED']
            for g in failed:
                lines.append(f'    ❌ {g}')
            return '\n'.join(lines)
        return ''

    lines = []
    lines.append('')
    lines.append('  🔥 SQUEEZE DETECTOR v6.0')

    icons = {'SHORT_SQUEEZE': '🟩', 'LONG_SQUEEZE': '🟥'}
    icon = icons.get(sq['squeeze_type'], '❓')
    strength = 'STRONG' if sq.get('squeeze_strong') else 'CONFIRMED'

    status = sq.get('squeeze_status', 'NONE')
    status_icons = {'TRIGGERED': '✅', 'PENDING': '⏳'}
    status_icon = status_icons.get(status, '❓')

    lines.append(f'  Type: {icon} {sq["squeeze_type"]}  ({strength})')
    lines.append(f'  Status: {status_icon} {status}')
    lines.append(f'  Score: {sq["squeeze_score"]:.3f}  '
                 f'Trigger: {sq.get("trigger_type", "?")}')

    # Lifecycle stage
    lc_stage = sq.get('lifecycle_stage', 'NONE')
    lc_icons = {
        'FORMING': '🔵', 'TIGHTENING': '🟡', 'MATURE': '🟢',
        'BREAKOUT': '💥', 'RETEST': '🔄', 'EXPIRED': '⚫', 'NONE': '⚪',
    }
    lc_icon = lc_icons.get(lc_stage, '⚪')
    lines.append(f'  Lifecycle: {lc_icon} {lc_stage}')

    # Path details
    pa = sq.get('path_a', {})
    pb = sq.get('path_b', {})

    if pa.get('fired'):
        lines.append(f'  Path A (15m): ✅ compressed {pa.get("comp_bars", 0)} bars  '
                     f'dry={pa.get("dry", 0)}  doji={pa.get("doji", 0)}')

    if pb.get('coiled') or pb.get('fired'):
        coil_h = pb.get('coil_hours', 0)
        max_h = pb.get('max_streak_hours', 0)
        flip = '✅ FLIP' if pb.get('hist_flip') else '⏳ waiting'
        lines.append(f'  Path B (2h):  ✅ coiled {coil_h}h (max {max_h}h)  '
                     f'hist={pb.get("hist_value", 0):.3f}  {flip}')
        lines.append(f'    MACD(8,17,9): DIF={pb.get("dif", 0):.3f}  DEA={pb.get("dea", 0):.3f}  '
                     f'Δ={pb.get("delta_pct", 0):.4f}%')

    # Box typology (Phase 1)
    box_type = sq.get('box_type', 'UNKNOWN')
    if box_type != 'UNKNOWN':
        box_icons = {
            'SIDEWAYS': '↔️', 'DESCENDING_TRIANGLE': '🔻',
            'ASCENDING_TRIANGLE': '🔺', 'SYMMETRIC_TRIANGLE': '🔻🔺',
            'BEAR_FLAG': '🚩↓', 'BULL_FLAG': '🚩↑', 'EXPANDING': '↕️',
        }
        box_icon = box_icons.get(box_type, '❓')
        bias = sq.get('box_bias', 'NEUTRAL')
        bias_str = f'  bias={bias}' if bias != 'NEUTRAL' else ''
        lines.append(f'  Box: {box_icon} {box_type}{bias_str}')
        lines.append(f'    H-slope: {sq.get("box_high_slope", 0):+.6f}/bar  '
                     f'L-slope: {sq.get("box_low_slope", 0):+.6f}/bar  '
                     f'convergence: {sq.get("box_convergence", 0):.0%}  '
                     f'swings: {sq.get("box_n_swings", "?")}')

    # Compression maturity
    maturity = sq.get('compression_maturity', 0)
    range_ratio = sq.get('compression_range_ratio', 1.0)
    if maturity > 0:
        mat_icon = '🟢' if maturity >= 0.8 else '🟡' if maturity >= 0.5 else '🔵'
        lines.append(f'  Maturity: {mat_icon} {maturity:.0%}  '
                     f'range_ratio: {range_ratio:.2f}')

    lines.append(f'  Direction: {sq["direction"]}')

    # Coil range + entry
    coil_h = sq.get('coil_high', 0)
    coil_l = sq.get('coil_low', 0)
    if coil_h > 0 and coil_l > 0:
        tested_h = sq.get('tested_high', 0)
        tested_l = sq.get('tested_low', 0)
        lines.append(f'  Coil range: ${coil_l:.2f} - ${coil_h:.2f}  '
                     f'(width ${(coil_h - coil_l):.2f})')
        if tested_h != coil_h or tested_l != coil_l:
            lines.append(f'  Tested:     ${tested_l:.2f} - ${tested_h:.2f}  (most-reversed levels)')

    entry_cond = sq.get('entry_condition', '')
    if entry_cond:
        lines.append(f'  Entry: {entry_cond}')
        lines.append(f'  Entry price: ${sq.get("entry_price", 0):.2f}')

    # Breakout quality (Phase 2)
    bq_score = sq.get('breakout_quality_score')
    bq_passed = sq.get('breakout_quality_passed', False)
    if bq_score is not None:
        bq_icon = '✅' if bq_passed else '⚠️'
        lines.append(f'  Breakout Quality: {bq_icon} {bq_score:.2f}')
        for bf in sq.get('breakout_quality_factors', []):
            if '❌' in bf:
                lines.append(f'    {bf}')

    # Retest (Phase 2)
    if sq.get('retest_detected'):
        lines.append(f'  🔄 Retest: {sq.get("retest_detail", "")}  quality={sq.get("retest_quality", 0):.2f}')

    if sq.get('overrides_regime') and status == 'TRIGGERED':
        lines.append(f'  ⚡ Overrides M9 regime block!')

    if sq.get('factors'):
        lines.append(f'\n  Factors:')
        for f in sq['factors']:
            lines.append(f'    ✅ {f}')

    if sq.get('tp', 0) > 0:
        tp = sq['tp']
        sl = sq['sl']
        entry = sq.get('entry_price', 0)
        tp_src = sq.get('tp1_source', 'ATR')
        dur_mult = sq.get('tp_duration_mult', 1.0)
        coil_h = sq.get('coil_hours', 0)
        dur_note = f'  coil={coil_h}h dur_mult={dur_mult:.2f}' if dur_mult > 1.0 else ''

        # Calculate R:R
        tp_dist = abs(tp - entry) if entry > 0 else 0
        sl_dist = abs(entry - sl) if entry > 0 else 0
        rr = tp_dist / sl_dist if sl_dist > 0 else 0

        lines.append(f'\n  ┌─────────────────────────────────────┐')
        lines.append(f'  │  SQUEEZE TRADE PLAN  ({sq["direction"]})          │')
        lines.append(f'  ├─────────────────────────────────────┤')
        if entry > 0:
            lines.append(f'  │  Entry:  ${entry:<10.2f} (trigger)      │')
        lines.append(f'  │  TP:     ${tp:<10.2f} (+{sq["tp_pct"]:.2f}%)  [{tp_src}]│')
        lines.append(f'  │  SL:     ${sl:<10.2f} ({sq["sl_pct"]:.2f}%)         │')
        lines.append(f'  │  R:R:    {rr:.2f}x                        │')
        if dur_note:
            lines.append(f'  │  {dur_note:<35}│')
        lines.append(f'  └─────────────────────────────────────┘')

    if sq.get('failed_breakout'):
        lines.append(f'  ⚠️ FAILED BREAKOUT: {sq.get("failed_breakout_detail", "")}')

    if sq.get('ics_boost', 0) > 0 and status == 'TRIGGERED':
        lines.append(f'  ICS boost: +{sq["ics_boost"]:.4f}')

    return '\n'.join(lines)

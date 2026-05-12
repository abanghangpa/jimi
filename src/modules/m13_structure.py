"""
M13: HTF Structure — Swing Analysis + FVG + Order Blocks.

Provides directional bias from market structure:
  1. Swing High/Low sequencing (HH/HL = bullish, LH/LL = bearish)
  2. Fair Value Gaps (inefficiency zones — price tends to revert)
  3. Order Blocks (institutional rejection zones)

Designed to run AFTER M9 (regime) and BEFORE entry scoring.
"""

import numpy as np
import pandas as pd


# ═══════════════════════════════════════════════════════════════
# SWING DETECTION
# ═══════════════════════════════════════════════════════════════

def find_swings(df, lookback=60, pivot_bars=3):
    """
    Find swing highs and lows using N-bar pivot detection.

    A swing high is a bar whose high is higher than the N bars on each side.
    A swing low is a bar whose low is lower than the N bars on each side.

    Returns list of dicts: {'type': 'H'|'L', 'price': float, 'idx': int, 'time': timestamp}
    """
    highs = df['High'].values
    lows = df['Low'].values
    n = len(highs)
    swings = []

    start = max(pivot_bars, 0)
    end = min(n - pivot_bars, n)

    for i in range(start, end):
        # Swing High: high[i] > all highs in [i-pivot_bars, i+pivot_bars] (excluding i)
        is_swing_high = True
        for j in range(max(0, i - pivot_bars), min(n, i + pivot_bars + 1)):
            if j == i:
                continue
            if highs[j] >= highs[i]:
                is_swing_high = False
                break

        if is_swing_high:
            swings.append({
                'type': 'H',
                'price': highs[i],
                'idx': i,
                'time': df['Open time'].iloc[i] if 'Open time' in df.columns else i,
            })

        # Swing Low: low[i] < all lows in [i-pivot_bars, i+pivot_bars] (excluding i)
        is_swing_low = True
        for j in range(max(0, i - pivot_bars), min(n, i + pivot_bars + 1)):
            if j == i:
                continue
            if lows[j] <= lows[i]:
                is_swing_low = False
                break

        if is_swing_low:
            swings.append({
                'type': 'L',
                'price': lows[i],
                'idx': i,
                'time': df['Open time'].iloc[i] if 'Open time' in df.columns else i,
            })

    # Sort by index
    swings.sort(key=lambda x: x['idx'])
    return swings


def classify_swing_sequence(swings, min_swings=4):
    """
    Classify market structure from swing point sequence.

    Rules:
      HH + HL = BULLISH  (higher highs, higher lows)
      LH + LL = BEARISH  (lower highs, lower lows)
      Mixed   = NEUTRAL

    Uses the last N swing points to determine the dominant pattern.
    """
    swing_highs = [s for s in swings if s['type'] == 'H']
    swing_lows = [s for s in swings if s['type'] == 'L']

    # Always export available swings for downstream modules (M14, wick reclaim)
    recent_highs = swing_highs[-3:] if len(swing_highs) >= 3 else swing_highs[-2:]
    recent_lows = swing_lows[-3:] if len(swing_lows) >= 3 else swing_lows[-2:]

    base_details = {
        'swing_highs': [(s['price'], s['idx']) for s in recent_highs[-3:]],
        'swing_lows': [(s['price'], s['idx']) for s in recent_lows[-3:]],
    }

    if len(swings) < min_swings:
        return 'NEUTRAL', 0.0, base_details

    if len(swing_highs) < 2 or len(swing_lows) < 2:
        return 'NEUTRAL', 0.0, base_details

    # Check last 3 swing highs: are they ascending or descending?
    recent_highs = swing_highs[-3:] if len(swing_highs) >= 3 else swing_highs[-2:]
    recent_lows = swing_lows[-3:] if len(swing_lows) >= 3 else swing_lows[-2:]

    # Count HH vs LH
    hh_count = 0
    lh_count = 0
    for i in range(1, len(recent_highs)):
        if recent_highs[i]['price'] > recent_highs[i-1]['price']:
            hh_count += 1
        elif recent_highs[i]['price'] < recent_highs[i-1]['price']:
            lh_count += 1

    # Count HL vs LL
    hl_count = 0
    ll_count = 0
    for i in range(1, len(recent_lows)):
        if recent_lows[i]['price'] > recent_lows[i-1]['price']:
            hl_count += 1
        elif recent_lows[i]['price'] < recent_lows[i-1]['price']:
            ll_count += 1

    total_pairs = (len(recent_highs) - 1) + (len(recent_lows) - 1)
    if total_pairs == 0:
        return 'NEUTRAL', 0.0, base_details

    # Classify
    bull_score = (hh_count + hl_count) / total_pairs
    bear_score = (lh_count + ll_count) / total_pairs

    details = {
        'hh_count': hh_count,
        'lh_count': lh_count,
        'hl_count': hl_count,
        'll_count': ll_count,
        'bull_score': round(bull_score, 3),
        'bear_score': round(bear_score, 3),
        'swing_highs': [(s['price'], s['idx']) for s in recent_highs[-3:]],
        'swing_lows': [(s['price'], s['idx']) for s in recent_lows[-3:]],
    }

    # Detect wedge patterns: when highs and lows diverge
    # Rising wedge: LH + HL (bearish highs, bullish lows → compression)
    # Falling wedge: HH + LL (bullish highs, bearish lows → compression)
    rising_wedge = (lh_count > 0 and hl_count > 0 and hh_count == 0 and ll_count == 0)
    falling_wedge = (hh_count > 0 and ll_count > 0 and lh_count == 0 and hl_count == 0)

    if rising_wedge:
        # Rising wedge: lower highs + higher lows → compression, leans neutral-to-bull
        # The lows rising means buyers defending higher — not bearish
        details['pattern'] = 'RISING_WEDGE'
        return 'NEUTRAL', 0.5, details
    elif falling_wedge:
        # Falling wedge: higher highs + lower lows → compression, leans neutral-to-bear
        details['pattern'] = 'FALLING_WEDGE'
        return 'NEUTRAL', 0.5, details

    if bull_score >= 0.6 and bear_score < 0.3:
        return 'BULLISH', bull_score, details
    elif bear_score >= 0.6 and bull_score < 0.3:
        return 'BEARISH', bear_score, details
    elif bull_score > bear_score + 0.2:
        return 'LEAN_BULL', bull_score, details
    elif bear_score > bull_score + 0.2:
        return 'LEAN_BEAR', bear_score, details
    else:
        return 'NEUTRAL', max(bull_score, bear_score), details


# ═══════════════════════════════════════════════════════════════
# FAIR VALUE GAP DETECTION
# ═══════════════════════════════════════════════════════════════

def find_fair_value_gaps(df, lookback=100, min_gap_pct=0.001):
    """
    Detect Fair Value Gaps (FVGs) — three-candle pattern where price moved too fast.

    Bullish FVG: candle[i-1].high < candle[i+1].low  (gap up)
    Bearish FVG: candle[i-1].low > candle[i+1].high  (gap down)

    Returns list of FVGs with their zones and fill status.
    """
    highs = df['High'].values
    lows = df['Low'].values
    closes = df['Close'].values
    n = len(highs)

    fvgs = []
    start = max(1, n - lookback)

    for i in range(start, n - 1):
        # Bullish FVG: gap between candle i-1 high and candle i+1 low
        if i >= 1 and i + 1 < n:
            gap_size = lows[i + 1] - highs[i - 1]
            if gap_size > 0 and gap_size / highs[i - 1] >= min_gap_pct:
                fvgs.append({
                    'type': 'BULLISH',
                    'top': lows[i + 1],
                    'bottom': highs[i - 1],
                    'mid': (lows[i + 1] + highs[i - 1]) / 2,
                    'idx': i,
                    'gap_pct': round(gap_size / highs[i - 1] * 100, 4),
                    'filled': False,
                })

            # Bearish FVG: gap between candle i-1 low and candle i+1 high
            gap_size = lows[i - 1] - highs[i + 1]
            if gap_size > 0 and gap_size / lows[i - 1] >= min_gap_pct:
                fvgs.append({
                    'type': 'BEARISH',
                    'top': lows[i - 1],
                    'bottom': highs[i + 1],
                    'mid': (lows[i - 1] + highs[i + 1]) / 2,
                    'idx': i,
                    'gap_pct': round(gap_size / lows[i - 1] * 100, 4),
                    'filled': False,
                })

    # Check fill status: has price revisited the gap zone?
    current_price = closes[-1]
    for fvg in fvgs:
        # Check if any bar after the FVG filled it
        fill_start = fvg['idx'] + 2
        for j in range(fill_start, n):
            if fvg['type'] == 'BULLISH':
                if lows[j] <= fvg['bottom']:
                    fvg['filled'] = True
                    fvg['fill_idx'] = j
                    break
            else:
                if highs[j] >= fvg['top']:
                    fvg['filled'] = True
                    fvg['fill_idx'] = j
                    break

    # Return unfilled FVGs closest to current price
    unfilled = [f for f in fvgs if not f['filled']]
    unfilled.sort(key=lambda f: abs(f['mid'] - current_price))
    return unfilled[:10]


# ═══════════════════════════════════════════════════════════════
# ORDER BLOCK DETECTION
# ═══════════════════════════════════════════════════════════════

def find_order_blocks(df, lookback=100, impulse_pct=0.005, consolidation_bars=3):
    """
    Detect Order Blocks — the last opposing candle before a strong impulse move.

    Bullish OB: last bearish candle before a strong bullish move (>= impulse_pct)
    Bearish OB: last bullish candle before a strong bearish move (>= impulse_pct)

    These are institutional zones where price is likely to react.
    """
    opens = df['Open'].values
    highs = df['High'].values
    lows = df['Low'].values
    closes = df['Close'].values
    n = len(closes)

    order_blocks = []
    start = max(consolidation_bars + 1, n - lookback)

    for i in range(start, n - 1):
        # Check for impulse move: close[i] to close[i+1]
        move = (closes[i + 1] - closes[i]) / closes[i]

        if move >= impulse_pct:
            # Bullish impulse — find last bearish candle before it
            for j in range(i, max(i - consolidation_bars - 1, 0), -1):
                if closes[j] < opens[j]:  # bearish candle
                    ob_high = highs[j]
                    ob_low = lows[j]
                    # Verify: price should have moved away from this zone
                    move_away = (closes[i + 1] - ob_high) / ob_high
                    if move_away > 0:
                        order_blocks.append({
                            'type': 'BULLISH',
                            'top': ob_high,
                            'bottom': ob_low,
                            'mid': (ob_high + ob_low) / 2,
                            'idx': j,
                            'impulse_pct': round(move * 100, 3),
                            'strength': min(move / impulse_pct, 3.0),
                        })
                        break

        elif move <= -impulse_pct:
            # Bearish impulse — find last bullish candle before it
            for j in range(i, max(i - consolidation_bars - 1, 0), -1):
                if closes[j] > opens[j]:  # bullish candle
                    ob_high = highs[j]
                    ob_low = lows[j]
                    move_away = (ob_low - closes[i + 1]) / ob_low
                    if move_away > 0:
                        order_blocks.append({
                            'type': 'BEARISH',
                            'top': ob_high,
                            'bottom': ob_low,
                            'mid': (ob_high + ob_low) / 2,
                            'idx': j,
                            'impulse_pct': round(abs(move) * 100, 3),
                            'strength': min(abs(move) / impulse_pct, 3.0),
                        })
                        break

    # Sort by strength, return top OBs
    order_blocks.sort(key=lambda x: x['strength'], reverse=True)

    # Check if OBs are still valid (not broken through)
    current_price = closes[-1]
    valid_obs = []
    for ob in order_blocks:
        broken = False
        for j in range(ob['idx'] + 1, n):
            if ob['type'] == 'BULLISH' and lows[j] < ob['bottom']:
                broken = True
                break
            elif ob['type'] == 'BEARISH' and highs[j] > ob['top']:
                broken = True
                break
        if not broken:
            valid_obs.append(ob)

    return valid_obs[:8]


# ═══════════════════════════════════════════════════════════════
# MAIN SCORING FUNCTION
# ═══════════════════════════════════════════════════════════════

def compute_structure_bias(df_1h, idx_1h, df_15m=None, idx_15m=None):
    """
    Compute directional bias from HTF market structure.

    Called AFTER M9 (regime detection) to determine direction.

    Returns:
        (bias, score, details)
        bias: 'BULLISH' | 'BEARISH' | 'NEUTRAL'
        score: 0.0-1.0 (confidence in the bias)
        details: dict with swing structure, FVGs, order blocks
    """
    details = {}

    if idx_1h < 60:
        return 'NEUTRAL', 0.5, details

    # ── 1. Swing Structure on 1H ──
    lookback_1h = min(120, idx_1h)
    df_1h_window = df_1h.iloc[max(0, idx_1h - lookback_1h):idx_1h + 1].copy()
    df_1h_window = df_1h_window.reset_index(drop=True)

    swings_1h = find_swings(df_1h_window, lookback=lookback_1h, pivot_bars=3)
    swing_bias, swing_confidence, swing_details = classify_swing_sequence(swings_1h)
    details['swing_bias'] = swing_bias
    details['swing_confidence'] = swing_confidence
    details['swing_count'] = len(swings_1h)
    details.update({f'swing_{k}': v for k, v in swing_details.items()})
    # Export swing levels without prefix for downstream modules (M14, wick reclaim)
    if 'swing_highs' in swing_details:
        details['swing_highs'] = swing_details['swing_highs']
    if 'swing_lows' in swing_details:
        details['swing_lows'] = swing_details['swing_lows']

    # ── 2. Swing Structure on 15m (execution TF) ──
    swing_bias_15m = 'NEUTRAL'
    swing_conf_15m = 0.0
    if df_15m is not None and idx_15m is not None and idx_15m >= 60:
        lookback_15m = min(120, idx_15m)
        df_15m_window = df_15m.iloc[max(0, idx_15m - lookback_15m):idx_15m + 1].copy()
        df_15m_window = df_15m_window.reset_index(drop=True)
        swings_15m = find_swings(df_15m_window, lookback=lookback_15m, pivot_bars=3)
        swing_bias_15m, swing_conf_15m, swing_details_15m = classify_swing_sequence(swings_15m)
        details['swing_bias_15m'] = swing_bias_15m
        details['swing_confidence_15m'] = swing_conf_15m

    # ── 3. Fair Value Gaps ──
    fvgs = []
    if df_15m is not None and idx_15m is not None and idx_15m >= 20:
        df_15m_window = df_15m.iloc[max(0, idx_15m - 100):idx_15m + 1].copy()
        df_15m_window = df_15m_window.reset_index(drop=True)
        fvgs = find_fair_value_gaps(df_15m_window, lookback=100)
    details['fvg_count'] = len(fvgs)
    details['fvg_bullish'] = sum(1 for f in fvgs if f['type'] == 'BULLISH')
    details['fvg_bearish'] = sum(1 for f in fvgs if f['type'] == 'BEARISH')

    # Nearest FVG
    if fvgs:
        current_price = df_15m['Close'].iloc[idx_15m] if df_15m is not None else df_1h['Close'].iloc[idx_1h]
        nearest = fvgs[0]
        details['nearest_fvg_type'] = nearest['type']
        details['nearest_fvg_dist_pct'] = round(
            abs(nearest['mid'] - current_price) / current_price * 100, 3
        )
        details['nearest_fvg_top'] = round(nearest['top'], 2)
        details['nearest_fvg_bottom'] = round(nearest['bottom'], 2)

    # ── 4. Order Blocks ──
    obs = []
    if df_15m is not None and idx_15m is not None and idx_15m >= 20:
        df_15m_window = df_15m.iloc[max(0, idx_15m - 100):idx_15m + 1].copy()
        df_15m_window = df_15m_window.reset_index(drop=True)
        obs = find_order_blocks(df_15m_window, lookback=100)
    details['ob_count'] = len(obs)
    details['ob_bullish'] = sum(1 for o in obs if o['type'] == 'BULLISH')
    details['ob_bearish'] = sum(1 for o in obs if o['type'] == 'BEARISH')

    # Export full lists for M5 consumption
    details['fvgs'] = fvgs
    details['order_blocks'] = obs

    # Nearest OB
    if obs:
        current_price = df_15m['Close'].iloc[idx_15m] if df_15m is not None else df_1h['Close'].iloc[idx_1h]
        nearest_ob = min(obs, key=lambda o: abs(o['mid'] - current_price))
        details['nearest_ob_type'] = nearest_ob['type']
        details['nearest_ob_dist_pct'] = round(
            abs(nearest_ob['mid'] - current_price) / current_price * 100, 3
        )
        details['nearest_ob_top'] = round(nearest_ob['top'], 2)
        details['nearest_ob_bottom'] = round(nearest_ob['bottom'], 2)

    # ── 5. Composite Direction ──
    # Blend: 50% 1H swing, 30% 15m swing, 10% FVG alignment, 10% OB alignment
    # When 1H is NEUTRAL (e.g. rising wedge), 15m should NOT dominate —
    # it gets reduced weight to avoid false directional signals from
    # short-term noise overriding the bigger picture.
    bull_points = 0.0
    bear_points = 0.0

    # 1H swing (dominant)
    if swing_bias == 'BULLISH':
        bull_points += 0.50 * swing_confidence
    elif swing_bias == 'BEARISH':
        bear_points += 0.50 * swing_confidence
    elif swing_bias == 'LEAN_BULL':
        bull_points += 0.30 * swing_confidence
    elif swing_bias == 'LEAN_BEAR':
        bear_points += 0.30 * swing_confidence

    # 15m swing (execution confirmation)
    # Reduce weight when 1H is NEUTRAL — 15m alone shouldn't set direction
    tf15_weight = 0.30 if swing_bias not in ('NEUTRAL',) else 0.15
    if swing_bias_15m == 'BULLISH':
        bull_points += tf15_weight * swing_conf_15m
    elif swing_bias_15m == 'BEARISH':
        bear_points += tf15_weight * swing_conf_15m
    elif swing_bias_15m == 'LEAN_BULL':
        bull_points += tf15_weight * 0.5 * swing_conf_15m
    elif swing_bias_15m == 'LEAN_BEAR':
        bear_points += tf15_weight * 0.5 * swing_conf_15m

    # FVG alignment
    if fvgs:
        nearby_bull_fvg = sum(1 for f in fvgs if f['type'] == 'BULLISH' and not f['filled'])
        nearby_bear_fvg = sum(1 for f in fvgs if f['type'] == 'BEARISH' and not f['filled'])
        if nearby_bull_fvg > nearby_bear_fvg:
            bull_points += 0.10
        elif nearby_bear_fvg > nearby_bull_fvg:
            bear_points += 0.10

    # OB alignment
    if obs:
        bull_ob = sum(1 for o in obs if o['type'] == 'BULLISH')
        bear_ob = sum(1 for o in obs if o['type'] == 'BEARISH')
        if bull_ob > bear_ob:
            bull_points += 0.10
        elif bear_ob > bull_ob:
            bear_points += 0.10

    # Final bias
    total = bull_points + bear_points
    if total == 0:
        return 'NEUTRAL', 0.5, details

    if bull_points > bear_points + 0.10:
        bias = 'BULLISH'
        score = min(bull_points / max(total, 0.01), 1.0)
    elif bear_points > bull_points + 0.10:
        bias = 'BEARISH'
        score = min(bear_points / max(total, 0.01), 1.0)
    else:
        bias = 'NEUTRAL'
        score = 0.5

    details['bull_points'] = round(bull_points, 3)
    details['bear_points'] = round(bear_points, 3)
    details['structure_score'] = round(score, 3)

    return bias, score, details


def score_m13(df_1h, idx_1h, direction, df_15m=None, idx_15m=None):
    """
    Score M13 for integration with ICS pipeline.

    Returns:
        (status, score, details)
        status: 'PASS' | 'FAIL' | 'SKIP'
        score: 0.0-1.0
    """
    bias, confidence, details = compute_structure_bias(
        df_1h, idx_1h, df_15m, idx_15m
    )

    details['m13_bias'] = bias

    # Check alignment with trade direction
    if bias == 'NEUTRAL':
        # Neutral structure: neither confirms nor denies
        return 'SKIP', 0.50, details

    aligned = (
        (direction == 'LONG' and bias in ('BULLISH', 'LEAN_BULL')) or
        (direction == 'SHORT' and bias in ('BEARISH', 'LEAN_BEAR'))
    )

    conflicting = (
        (direction == 'LONG' and bias in ('BEARISH', 'LEAN_BEAR')) or
        (direction == 'SHORT' and bias in ('BULLISH', 'LEAN_BULL'))
    )

    if aligned:
        # Structure confirms direction
        base_score = 0.60 + confidence * 0.30  # 0.60-0.90
        # Bonus if both 1H and 15m agree
        swing_1h = details.get('swing_bias', 'NEUTRAL')
        swing_15m = details.get('swing_bias_15m', 'NEUTRAL')
        if swing_1h != 'NEUTRAL' and swing_15m != 'NEUTRAL':
            if (direction == 'LONG' and 'BULL' in swing_1h and 'BULL' in swing_15m) or \
               (direction == 'SHORT' and 'BEAR' in swing_1h and 'BEAR' in swing_15m):
                base_score = min(base_score + 0.08, 1.0)
                details['mtf_structure_bonus'] = True
        return 'PASS', round(base_score, 3), details

    elif conflicting:
        # Structure contradicts direction
        penalty = 0.30 + (1.0 - confidence) * 0.20  # 0.30-0.50
        details['structure_conflict'] = True
        return 'FAIL', round(penalty, 3), details

    else:
        return 'SKIP', 0.50, details

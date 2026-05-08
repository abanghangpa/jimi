"""
M21: Wyckoff Phase Detector + Premium/Discount Filter

Reads the higher-timeframe structure to determine:
  1. Wyckoff Phase: ACCUMULATION | MARKUP | DISTRIBUTION | MARKDOWN | RANGE
  2. Premium/Discount zone relative to the range equilibrium
  3. Entry permission based on phase + zone

This runs in the direction resolver, BEFORE ICS scoring.
"""

import numpy as np
import pandas as pd


# ═══════════════════════════════════════════════════════════════
# RANGE DETECTION
# ═══════════════════════════════════════════════════════════════

def detect_trading_range(df_4h, lookback=48, min_touches=3, range_tolerance=0.02):
    """Detect if price is in a trading range using 4H data.

    A range exists when:
    - Price has tested a horizontal zone at least `min_touches` times
    - The range width is between 1.5% and 6% of price
    - Price has been inside the range for at least `lookback` bars

    Returns:
        dict with range_hi, range_lo, eq, phase, touches or None
    """
    if len(df_4h) < lookback:
        return None

    recent = df_4h.iloc[-lookback:]
    highs = recent['High'].values.astype(float)
    lows = recent['Low'].values.astype(float)
    closes = recent['Close'].values.astype(float)

    # Use clustering approach: find the most-tested high and low zones
    # Instead of percentiles, use the mode of highs and lows
    from collections import Counter

    # Round highs/lows to $5 bins and find most common
    hi_bins = [round(h / 5) * 5 for h in highs]
    lo_bins = [round(l / 5) * 5 for l in lows]

    hi_counter = Counter(hi_bins)
    lo_counter = Counter(lo_bins)

    # Most tested high zone (resistance)
    if not hi_counter or not lo_counter:
        return None

    range_hi = hi_counter.most_common(1)[0][0]
    range_lo = lo_counter.most_common(1)[0][0]

    # Validate: hi must be above lo
    if range_hi <= range_lo:
        return None

    range_width = (range_hi - range_lo) / np.mean(closes) * 100

    # Range must be 1.5% - 5% wide
    if range_width < 1.5 or range_width > 5.0:
        return None

    # Count touches of range boundaries (within 0.5%)
    hi_touches = np.sum(np.abs(highs - range_hi) / range_hi < 0.005)
    lo_touches = np.sum(np.abs(lows - range_lo) / range_lo < 0.005)

    if hi_touches < min_touches or lo_touches < min_touches:
        return None

    # Check if current price is inside the range
    current = closes[-1]
    if current > range_hi * 1.01 or current < range_lo * 0.99:
        return None  # price broke out of range

    eq = (range_hi + range_lo) / 2

    return {
        'range_hi': float(range_hi),
        'range_lo': float(range_lo),
        'eq': float(eq),
        'width_pct': float(range_width),
        'hi_touches': int(hi_touches),
        'lo_touches': int(lo_touches),
        'total_touches': int(hi_touches + lo_touches),
    }


# ═══════════════════════════════════════════════════════════════
# WYCKOFF PHASE DETECTION
# ═══════════════════════════════════════════════════════════════

def detect_wyckoff_phase(df_1d, df_4h, range_info=None):
    """Detect the current Wyckoff phase from daily + 4H structure.

    Phases:
      ACCUMULATION: Smart money buying, price near range lows, selling climax happened
      MARKUP: Trending up from accumulation, higher highs/lows
      DISTRIBUTION: Smart money selling, price near range highs, buying climax happened
      MARKDOWN: Trending down from distribution, lower highs/lows
      RANGE: Undefined — choppy, no clear phase

    Returns:
        dict with phase, confidence, details
    """
    if len(df_1d) < 20 or len(df_4h) < 48:
        return {'phase': 'RANGE', 'confidence': 0.3, 'details': {}}

    # Daily structure: recent 20 bars
    d_closes = df_1d['Close'].values[-20:].astype(float)
    d_highs = df_1d['High'].values[-20:].astype(float)
    d_lows = df_1d['Low'].values[-20:].astype(float)

    # 4H structure: recent 48 bars
    h4_closes = df_4h['Close'].values[-48:].astype(float)
    h4_highs = df_4h['High'].values[-48:].astype(float)
    h4_lows = df_4h['Low'].values[-48:].astype(float)
    h4_volumes = df_4h['Volume'].values[-48:].astype(float) if 'Volume' in df_4h.columns else np.ones(48)

    # ── Trend detection on daily ──
    # Higher highs / higher lows = bullish
    # Lower highs / lower lows = bearish
    recent_10_hi = d_highs[-10:].max()
    prior_10_hi = d_highs[-20:-10].max()
    recent_10_lo = d_lows[-10:].min()
    prior_10_lo = d_lows[-20:-10].min()

    hh = recent_10_hi > prior_10_hi  # higher high
    hl = recent_10_lo > prior_10_lo   # higher low
    lh = recent_10_hi < prior_10_hi   # lower high
    ll = recent_10_lo < prior_10_lo    # lower low

    # ── Volume analysis ──
    vol_recent = h4_volumes[-12:].mean()  # last 48h
    vol_prior = h4_volumes[-24:-12].mean()  # prior 48h
    vol_declining = vol_recent < vol_prior * 0.85
    vol_increasing = vol_recent > vol_prior * 1.15

    # ── Price position in range ──
    if range_info:
        eq = range_info['eq']
        range_hi = range_info['range_hi']
        range_lo = range_info['range_lo']
        current = h4_closes[-1]
        position = (current - range_lo) / (range_hi - range_lo) if range_hi > range_lo else 0.5
    else:
        # Use daily range
        period_hi = d_highs.max()
        period_lo = d_lows.min()
        eq = (period_hi + period_lo) / 2
        current = d_closes[-1]
        position = (current - period_lo) / (period_hi - period_lo) if period_hi > period_lo else 0.5

    # ── Phase classification ──
    phase = 'RANGE'
    confidence = 0.3
    details = {}

    if hh and hl:
        # Uptrend — could be MARKUP or late ACCUMULATION
        if position > 0.7 and vol_declining:
            # Near range top + declining volume = DISTRIBUTION starting
            phase = 'DISTRIBUTION'
            confidence = 0.6
            details['reason'] = 'HH+HL but near range top + declining vol'
        elif position > 0.6:
            phase = 'MARKUP'
            confidence = 0.7
            details['reason'] = 'HH+HL, price above EQ'
        else:
            phase = 'ACCUMULATION'
            confidence = 0.5
            details['reason'] = 'HH+HL but still below EQ — early markup'

    elif lh and ll:
        # Downtrend — could be MARKDOWN or late DISTRIBUTION
        if position < 0.3 and vol_declining:
            # Near range bottom + declining volume = ACCUMULATION starting
            phase = 'ACCUMULATION'
            confidence = 0.6
            details['reason'] = 'LH+LL but near range bottom + declining vol'
        elif position < 0.4:
            phase = 'MARKDOWN'
            confidence = 0.7
            details['reason'] = 'LH+LL, price below EQ'
        else:
            phase = 'DISTRIBUTION'
            confidence = 0.5
            details['reason'] = 'LH+LL but still above EQ — late distribution'

    elif range_info:
        # In a range — use volume + position to determine phase
        if position > 0.65:
            phase = 'DISTRIBUTION'
            confidence = 0.5
            details['reason'] = f'Range top ({position:.0%}), testing supply'
        elif position < 0.35:
            phase = 'ACCUMULATION'
            confidence = 0.5
            details['reason'] = f'Range bottom ({position:.0%}), testing demand'
        else:
            phase = 'RANGE'
            confidence = 0.4
            details['reason'] = f'Mid-range ({position:.0%}), no edge'

    details['position'] = round(position, 3)
    details['hh'] = hh
    details['hl'] = hl
    details['lh'] = lh
    details['ll'] = ll
    details['vol_recent'] = round(float(vol_recent), 0)
    details['vol_prior'] = round(float(vol_prior), 0)

    return {
        'phase': phase,
        'confidence': round(confidence, 2),
        'details': details,
    }


# ═══════════════════════════════════════════════════════════════
# PREMIUM / DISCOUNT ZONE
# ═══════════════════════════════════════════════════════════════

def classify_premium_discount(price, range_info=None, df_4h=None, lookback=48):
    """Determine if price is in premium or discount zone.

    Premium = above equilibrium (overbought for longs, good for shorts)
    Discount = below equilibrium (oversold for shorts, good for longs)

    Returns:
        dict with zone, eq, position, trade_permission
    """
    if range_info:
        eq = range_info['eq']
        range_hi = range_info['range_hi']
        range_lo = range_info['range_lo']
    elif df_4h is not None and len(df_4h) >= lookback:
        recent = df_4h.iloc[-lookback:]
        range_hi = float(recent['High'].max())
        range_lo = float(recent['Low'].min())
        eq = (range_hi + range_lo) / 2
    else:
        return {
            'zone': 'UNKNOWN', 'eq': price, 'position': 0.5,
            'long_allowed': True, 'short_allowed': True,
        }

    position = (price - range_lo) / (range_hi - range_lo) if range_hi > range_lo else 0.5

    if position >= 0.55:
        zone = 'PREMIUM'
        long_allowed = False   # don't buy in premium
        short_allowed = True   # good for shorts
    elif position <= 0.45:
        zone = 'DISCOUNT'
        long_allowed = True    # good for longs
        short_allowed = False  # don't short in discount
    else:
        zone = 'EQUILIBRIUM'
        long_allowed = True
        short_allowed = True

    return {
        'zone': zone,
        'eq': round(eq, 2),
        'range_hi': round(range_hi, 2),
        'range_lo': round(range_lo, 2),
        'position': round(position, 3),
        'long_allowed': long_allowed,
        'short_allowed': short_allowed,
    }


# ═══════════════════════════════════════════════════════════════
# RANGE-AWARE TP TARGETS
# ═══════════════════════════════════════════════════════════════

def get_range_targets(price, direction, range_info, magnets=None):
    """Compute structure-based TP targets using range extremes.

    In a range:
      LONG:  TP1 = EQ, TP2 = range_hi, TP3 = range_hi + extension
      SHORT: TP1 = EQ, TP2 = range_lo, TP3 = range_lo - extension

    Returns:
        dict with tp1, tp2, tp3, source
    """
    if not range_info:
        return None

    eq = range_info['eq']
    range_hi = range_info['range_hi']
    range_lo = range_info['range_lo']
    range_width = range_hi - range_lo

    if direction == 'LONG':
        # TP1: Equilibrium (midpoint of range)
        tp1 = eq
        # TP2: Top of range
        tp2 = range_hi
        # TP3: Extension beyond range (25% of range width)
        tp3 = range_hi + range_width * 0.25

        # If price is already above EQ, TP1 = range_hi
        if price > eq:
            tp1 = range_hi
            tp2 = range_hi + range_width * 0.15
            tp3 = range_hi + range_width * 0.30

    else:  # SHORT
        # TP1: Equilibrium
        tp1 = eq
        # TP2: Bottom of range
        tp2 = range_lo
        # TP3: Extension beyond range
        tp3 = range_lo - range_width * 0.25

        # If price is already below EQ, TP1 = range_lo
        if price < eq:
            tp1 = range_lo
            tp2 = range_lo - range_width * 0.15
            tp3 = range_lo - range_width * 0.30

    # If magnets are available, use the nearest one as TP1 if closer than range target
    if magnets:
        for mag_price, mag_vol, mag_str in magnets:
            if direction == 'LONG' and price < mag_price <= tp1:
                tp1 = mag_price
                break
            elif direction == 'SHORT' and price > mag_price >= tp1:
                tp1 = mag_price
                break

    return {
        'tp1': round(float(tp1), 2),
        'tp2': round(float(tp2), 2),
        'tp3': round(float(tp3), 2),
        'tp1_source': 'RANGE_EQ' if direction == 'LONG' and price < eq else 'RANGE_HI' if direction == 'LONG' else 'RANGE_EQ' if direction == 'SHORT' and price > eq else 'RANGE_LO',
        'tp2_source': 'RANGE_BOUNDARY',
        'tp3_source': 'RANGE_EXTENSION',
        'tp1_pct': round(abs(tp1 - price) / price * 100, 2),
    }


# ═══════════════════════════════════════════════════════════════
# RANGE-AWARE SL PLACEMENT
# ═══════════════════════════════════════════════════════════════

def get_range_sl(price, direction, range_info, atr_1h=None):
    """Compute SL using the opposite side of the range.

    In a range, the best SL is beyond the range boundary —
    if the range breaks, the trade thesis is invalidated.

    Returns:
        dict with sl, sl_pct, sl_source
    """
    if not range_info:
        return None

    range_hi = range_info['range_hi']
    range_lo = range_info['range_lo']
    buffer = (range_hi - range_lo) * 0.05  # 5% of range as buffer

    if direction == 'LONG':
        sl = range_lo - buffer
        # Hard cap at 2.5%
        max_sl = price * 0.025
        if price - sl > max_sl:
            sl = price - max_sl
    else:
        sl = range_hi + buffer
        max_sl = price * 0.025
        if sl - price > max_sl:
            sl = price + max_sl

    return {
        'sl': round(float(sl), 2),
        'sl_pct': round(abs(price - sl) / price * 100, 2),
        'sl_source': 'RANGE_BOUNDARY',
    }


# ═══════════════════════════════════════════════════════════════
# SESSION KILL ZONE FILTER
# ═══════════════════════════════════════════════════════════════

def check_kill_zone(timestamp, config=None):
    """Check if current time is in an ICT kill zone.

    Kill zones (UTC):
      London Open:  07:00 - 10:00 (highest probability)
      NY Open:      12:30 - 15:30 (Silver Bullet)
      London Close: 15:00 - 17:00 (continuation)

    Returns:
        dict with in_kill_zone, session, multiplier
    """
    cfg = config or {}
    if not cfg.get('KILL_ZONE_FILTER_ENABLED', False):
        return {'in_kill_zone': True, 'session': 'ANY', 'multiplier': 1.0}

    hour = timestamp.hour
    minute = timestamp.minute
    time_val = hour + minute / 60.0

    # London Open Kill Zone
    if 7.0 <= time_val < 10.0:
        return {'in_kill_zone': True, 'session': 'LONDON_OPEN', 'multiplier': 1.2}

    # NY Open Kill Zone (Silver Bullet)
    if 12.5 <= time_val < 15.5:
        return {'in_kill_zone': True, 'session': 'NY_OPEN', 'multiplier': 1.3}

    # London Close
    if 15.0 <= time_val < 17.0:
        return {'in_kill_zone': True, 'session': 'LONDON_CLOSE', 'multiplier': 1.1}

    # Asian session — reduced edge
    if 0.0 <= time_val < 7.0:
        return {'in_kill_zone': False, 'session': 'ASIAN', 'multiplier': 0.6}

    # Off-hours
    return {'in_kill_zone': False, 'session': 'OFF', 'multiplier': 0.7}


# ═══════════════════════════════════════════════════════════════
# SPRING / UPTHRUST DETECTION (Wyckoff + ICT)
# ═══════════════════════════════════════════════════════════════

def detect_spring_upthrust(df_15m, idx, range_info, lookback=12):
    """Detect Wyckoff Spring or Upthrust.

    Spring: price briefly dips below range_lo then closes back inside
    Upthrust: price briefly spikes above range_hi then closes back inside

    These are high-probability reversal signals.

    Returns:
        dict with type (SPRING|UPTHRUST|NONE), strength, details
    """
    if not range_info or idx < lookback:
        return {'type': 'NONE', 'strength': 0, 'details': {}}

    range_hi = range_info['range_hi']
    range_lo = range_info['range_lo']

    recent = df_15m.iloc[idx - lookback + 1:idx + 1]
    lows = recent['Low'].values.astype(float)
    highs = recent['High'].values.astype(float)
    closes = recent['Close'].values.astype(float)
    opens = recent['Open'].values.astype(float)
    volumes = recent['Volume'].values.astype(float) if 'Volume' in recent.columns else np.ones(len(recent))

    # Check for Spring: wick below range_lo, close above
    for i in range(len(recent)):
        if lows[i] < range_lo and closes[i] > range_lo:
            wick_below = range_lo - lows[i]
            body = abs(closes[i] - opens[i])
            if wick_below > body * 0.5:  # significant wick
                # Volume confirmation: higher volume on the spring
                vol_avg = volumes[max(0, i-5):i].mean() if i > 0 else volumes.mean()
                vol_spike = volumes[i] / vol_avg if vol_avg > 0 else 1
                strength = min(wick_below / (range_hi - range_lo) * 5, 1.0)
                if vol_spike > 1.2:
                    strength = min(strength * 1.3, 1.0)
                return {
                    'type': 'SPRING',
                    'strength': round(strength, 2),
                    'details': {
                        'wick_below': round(wick_below, 2),
                        'close_above': round(closes[i], 2),
                        'vol_spike': round(vol_spike, 1),
                        'bar_idx': idx - lookback + 1 + i,
                    }
                }

    # Check for Upthrust: wick above range_hi, close below
    for i in range(len(recent)):
        if highs[i] > range_hi and closes[i] < range_hi:
            wick_above = highs[i] - range_hi
            body = abs(closes[i] - opens[i])
            if wick_above > body * 0.5:
                vol_avg = volumes[max(0, i-5):i].mean() if i > 0 else volumes.mean()
                vol_spike = volumes[i] / vol_avg if vol_avg > 0 else 1
                strength = min(wick_above / (range_hi - range_lo) * 5, 1.0)
                if vol_spike > 1.2:
                    strength = min(strength * 1.3, 1.0)
                return {
                    'type': 'UPTHRUST',
                    'strength': round(strength, 2),
                    'details': {
                        'wick_above': round(wick_above, 2),
                        'close_below': round(closes[i], 2),
                        'vol_spike': round(vol_spike, 1),
                        'bar_idx': idx - lookback + 1 + i,
                    }
                }

    return {'type': 'NONE', 'strength': 0, 'details': {}}


# ═══════════════════════════════════════════════════════════════
# MAIN ENTRY POINT
# ═══════════════════════════════════════════════════════════════

def score_m21(df_15m, df_1h, df_4h, df_1d, idx, direction, config=None):
    """Score M21: Wyckoff Phase + Premium/Discount + Kill Zone.

    Called from scanner/engine after direction resolver, before ICS scoring.

    Returns:
        (status, score, details)
        status: 'PASS' | 'BLOCKED' | 'CAUTION'
        score: 0.0-1.0 (how favorable the phase/zone is for the trade direction)
        details: full decision trace
    """
    cfg = config or {}
    if not cfg.get('M21_ENABLED', True):
        return 'SKIP', 0.5, {}

    ts = df_15m['Open time'].iloc[idx]
    price = float(df_15m['Close'].iloc[idx])

    # ── Detect trading range ──
    range_info = detect_trading_range(df_4h, lookback=cfg.get('M21_RANGE_LOOKBACK', 48))

    # ── Wyckoff phase ──
    phase_result = detect_wyckoff_phase(df_1d, df_4h, range_info)
    phase = phase_result['phase']
    phase_confidence = phase_result['confidence']

    # ── Premium/Discount ──
    pd_result = classify_premium_discount(price, range_info, df_4h)

    # ── Kill Zone ──
    kz_result = check_kill_zone(ts, cfg)

    # ── Spring/Upthrust ──
    spring_result = detect_spring_upthrust(df_15m, idx, range_info,
                                           lookback=cfg.get('M21_SPRING_LOOKBACK', 12))

    # ── Score computation ──
    score = 0.5  # neutral
    status = 'PASS'
    reasons = []

    # Phase scoring
    if phase == 'DISTRIBUTION':
        if direction == 'SHORT':
            score += 0.15
            reasons.append('Distribution phase favors SHORT')
        elif direction == 'LONG':
            score -= 0.25
            reasons.append('Distribution phase penalizes LONG')
            # Block LONG in distribution if price is in upper half
            if pd_result['position'] >= 0.50:
                score -= 0.15
                reasons.append('LONG in upper range during distribution — BLOCK')
                status = 'BLOCKED'

    elif phase == 'ACCUMULATION':
        if direction == 'LONG':
            score += 0.15
            reasons.append('Accumulation phase favors LONG')
        elif direction == 'SHORT':
            score -= 0.25
            reasons.append('Accumulation phase penalizes SHORT')
            # Block SHORT in accumulation if price is in lower half
            if pd_result['position'] <= 0.50:
                score -= 0.15
                reasons.append('SHORT in lower range during accumulation — BLOCK')
                status = 'BLOCKED'

    elif phase == 'MARKUP':
        if direction == 'LONG':
            score += 0.10
            reasons.append('Markup phase favors LONG')
        elif direction == 'SHORT':
            score -= 0.10
            reasons.append('Markup phase penalizes SHORT')

    elif phase == 'MARKDOWN':
        if direction == 'SHORT':
            score += 0.10
            reasons.append('Markdown phase favors SHORT')
        elif direction == 'LONG':
            score -= 0.10
            reasons.append('Markdown phase penalizes LONG')

    # Premium/Discount scoring
    if pd_result['zone'] == 'PREMIUM' and direction == 'LONG':
        score -= 0.10
        reasons.append('Buying in premium zone')
    elif pd_result['zone'] == 'DISCOUNT' and direction == 'SHORT':
        score -= 0.10
        reasons.append('Selling in discount zone')
    elif pd_result['zone'] == 'PREMIUM' and direction == 'SHORT':
        score += 0.05
        reasons.append('Selling in premium zone — favorable')
    elif pd_result['zone'] == 'DISCOUNT' and direction == 'LONG':
        score += 0.05
        reasons.append('Buying in discount zone — favorable')

    # Kill Zone scoring
    if kz_result['in_kill_zone']:
        score += 0.05
        reasons.append(f'In kill zone: {kz_result["session"]}')
    else:
        score -= 0.05
        reasons.append(f'Outside kill zone: {kz_result["session"]}')

    # Spring/Upthrust bonus
    if spring_result['type'] == 'SPRING' and direction == 'LONG':
        score += 0.15
        reasons.append(f'Spring detected (str={spring_result["strength"]:.2f})')
    elif spring_result['type'] == 'UPTHRUST' and direction == 'SHORT':
        score += 0.15
        reasons.append(f'Upthrust detected (str={spring_result["strength"]:.2f})')

    score = max(0.0, min(1.0, score))

    details = {
        'phase': phase,
        'phase_confidence': phase_confidence,
        'phase_details': phase_result.get('details', {}),
        'premium_discount': pd_result,
        'kill_zone': kz_result,
        'spring_upthrust': spring_result,
        'range_info': range_info,
        'reasons': reasons,
    }

    return status, round(score, 3), details


def format_m21(details):
    """Format M21 output for display."""
    if not details:
        return ''

    lines = []
    phase = details.get('phase', '?')
    pd_info = details.get('premium_discount', {})
    kz = details.get('kill_zone', {})
    spring = details.get('spring_upthrust', {})
    ri = details.get('range_info')

    icon = {
        'ACCUMULATION': '🟢', 'MARKUP': '📈',
        'DISTRIBUTION': '🔴', 'MARKDOWN': '📉',
        'RANGE': '↔️',
    }.get(phase, '❓')

    lines.append(f'  {icon} Wyckoff Phase: {phase} (conf={details.get("phase_confidence", 0):.0%})')

    if ri:
        lines.append(f'  Range: ${ri["range_lo"]:.2f} - ${ri["range_hi"]:.2f} '
                     f'(EQ=${ri["eq"]:.2f}, width={ri["width_pct"]:.1f}%)')

    zone_icon = {'PREMIUM': '🔴', 'DISCOUNT': '🟢', 'EQUILIBRIUM': '⚪'}.get(pd_info.get('zone'), '?')
    lines.append(f'  {zone_icon} Zone: {pd_info.get("zone", "?")} '
                 f'(pos={pd_info.get("position", 0):.0%})')

    if kz:
        kz_icon = '✅' if kz.get('in_kill_zone') else '⚠️'
        lines.append(f'  {kz_icon} Kill Zone: {kz.get("session", "?")} '
                     f'(mult={kz.get("multiplier", 1):.1f})')

    if spring.get('type') != 'NONE':
        lines.append(f'  ⚡ {spring["type"]} detected (strength={spring["strength"]:.2f})')

    for r in details.get('reasons', []):
        lines.append(f'    • {r}')

    return '\n'.join(lines)

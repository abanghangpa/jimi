"""
Entry Optimizer — Evaluate entry quality and suggest optimal entry levels.

Instead of entering at market, evaluates:
  1. Distance to support/resistance
  2. Distance to VWAP (mean reversion risk)
  3. Distance to liquidation magnets
  4. Risk/reward at current price vs better levels

Output actions:
  - ENTER_NOW: good entry, don't wait
  - WAIT_DIP: pullback expected, set limit order at X
  - WAIT_BREAKOUT: enter on confirmed break above X
  - SKIP: entry is terrible, wait for next signal
"""

import numpy as np
import pandas as pd


# Default thresholds
_DEFAULTS = {
    'MAX_RESISTANCE_DIST_PCT': 0.30,
    'MAX_SUPPORT_DIST_PCT': 0.30,
    'MAX_VWAP_STRETCH_PCT': 1.00,
    'VWAP_PULLBACK_TARGET': 0.50,
    'MIN_RR_RATIO': 1.5,
    'SL_BUFFER_ATR': 0.15,
    'MAGNET_DANGER_DIST_PCT': 2.0,
    'MAGNET_PULL_STRENGTH': 2.0,
    'BREAKOUT_BUFFER_PCT': 0.10,
    # Wick reclaim
    'WICK_RECLAIM_ENABLED': True,
    'WICK_RECLAIM_ZONE_PCT': 0.50,      # how close to swing level (% of price)
    'WICK_RECLAIM_MIN_WICK_RATIO': 0.5, # wick must be >= 50% of bar range
    'WICK_RECLAIM_BONUS': 0.08,         # ICS bonus for confirmed reclaim
    'WICK_REJECT_PENALTY': 0.06,        # ICS penalty for slice-through (no reclaim)
    'WICK_RECLAIM_VOLUME_MULT': 1.3,    # volume must be 1.3x avg for confirmation
}


def _find_nearest_levels(price, sr_levels):
    """Find nearest support and resistance from S/R levels."""
    nearest_support = None
    nearest_resistance = None
    if not sr_levels:
        return None, None
    for level in sr_levels:
        level_price = level[0] if isinstance(level, (list, tuple)) else level
        level_strength = level[1] if isinstance(level, (list, tuple)) and len(level) > 1 else 1.0
        if level_price < price:
            if nearest_support is None or level_price > nearest_support['price']:
                nearest_support = {'price': level_price, 'strength': level_strength}
        elif level_price > price:
            if nearest_resistance is None or level_price < nearest_resistance['price']:
                nearest_resistance = {'price': level_price, 'strength': level_strength}
    return nearest_support, nearest_resistance


def _check_magnets(price, direction, magnets, cfg):
    """Check if liquidation magnets pose a risk."""
    result = {'danger': False, 'magnets': []}
    if not magnets:
        return result
    for mag in magnets:
        if isinstance(mag, (list, tuple)):
            mag_price = mag[0]
            mag_strength = mag[1] if len(mag) > 1 else 1.0
        else:
            mag_price = mag
            mag_strength = 1.0
        dist_pct = abs(price - mag_price) / price * 100
        if direction == 'LONG' and mag_price < price and dist_pct < cfg['MAGNET_DANGER_DIST_PCT']:
            if mag_strength >= cfg['MAGNET_PULL_STRENGTH']:
                result['danger'] = True
                result['magnets'].append({
                    'price': mag_price, 'dist_pct': round(dist_pct, 2),
                    'strength': mag_strength, 'direction': 'below (pulls down)'
                })
        elif direction == 'SHORT' and mag_price > price and dist_pct < cfg['MAGNET_DANGER_DIST_PCT']:
            if mag_strength >= cfg['MAGNET_PULL_STRENGTH']:
                result['danger'] = True
                result['magnets'].append({
                    'price': mag_price, 'dist_pct': round(dist_pct, 2),
                    'strength': mag_strength, 'direction': 'above (pulls up)'
                })
    return result


def _compute_rr(direction, entry, sl, tp):
    """Compute R:R for a given entry."""
    risk = abs(entry - sl)
    reward = abs(tp - entry)
    return reward / risk if risk > 0 else 0


def evaluate_entry(direction, price, atr_1h, sr_levels, vwap,
                   liquidation_magnets, vol_ratio=None, swing_bias=None,
                   config=None):
    """
    Evaluate whether current price is a good entry.

    Args:
        direction: 'LONG' or 'SHORT'
        price: current price
        atr_1h: 1H ATR value
        sr_levels: list of (price, strength, ...) tuples
        vwap: VWAP value
        liquidation_magnets: list of (price, strength, ...) tuples
        vol_ratio: optional volume ratio
        swing_bias: optional swing bias
        config: optional config dict (uses defaults if None)

    Returns:
        (action, details) where action is 'ENTER_NOW'|'WAIT_DIP'|'WAIT_BREAKOUT'|'SKIP'
    """
    cfg = {**_DEFAULTS, **(config or {})}
    details = {}

    if price is None or price <= 0:
        return 'SKIP', {'reason': 'invalid price'}

    nearest_support, nearest_resistance = _find_nearest_levels(price, sr_levels)
    details['nearest_support'] = nearest_support
    details['nearest_resistance'] = nearest_resistance

    # VWAP analysis
    vwap_dist_pct = ((price - vwap) / vwap * 100) if vwap and vwap > 0 else 0
    details['vwap'] = round(vwap, 2) if vwap else None
    details['vwap_dist_pct'] = round(vwap_dist_pct, 2)
    vwap_stretched = abs(vwap_dist_pct) > cfg['MAX_VWAP_STRETCH_PCT']

    # Magnet analysis
    magnet_risk = _check_magnets(price, direction, liquidation_magnets, cfg)
    details['magnet_risk'] = magnet_risk

    # Resistance/support proximity
    resistance_too_close = False
    support_too_close = False

    if direction == 'LONG' and nearest_resistance:
        dist = (nearest_resistance['price'] - price) / price * 100
        details['resistance_dist_pct'] = round(dist, 3)
        if dist < cfg['MAX_RESISTANCE_DIST_PCT']:
            resistance_too_close = True
            details['resistance_warning'] = f"Resistance at ${nearest_resistance['price']:.2f} is only {dist:.2f}% away"

    if direction == 'SHORT' and nearest_support:
        dist = (price - nearest_support['price']) / price * 100
        details['support_dist_pct'] = round(dist, 3)
        if dist < cfg['MAX_SUPPORT_DIST_PCT']:
            support_too_close = True
            details['support_warning'] = f"Support at ${nearest_support['price']:.2f} is only {dist:.2f}% away"

    # Compute R:R at current price
    if direction == 'LONG':
        sl_price = nearest_support['price'] * (1 - cfg['SL_BUFFER_ATR'] * atr_1h / price) if nearest_support else price * 0.99
        tp1_price = nearest_resistance['price'] if nearest_resistance else price * 1.02
    else:
        sl_price = nearest_resistance['price'] * (1 + cfg['SL_BUFFER_ATR'] * atr_1h / price) if nearest_resistance else price * 1.01
        tp1_price = nearest_support['price'] if nearest_support else price * 0.98

    risk = abs(price - sl_price)
    reward = abs(tp1_price - price)
    rr_ratio = reward / risk if risk > 0 else 0

    details['sl_price'] = round(sl_price, 2)
    details['tp1_price'] = round(tp1_price, 2)
    details['risk_pct'] = round(risk / price * 100, 3)
    details['reward_pct'] = round(reward / price * 100, 3)
    details['rr_ratio'] = round(rr_ratio, 2)

    # Decision logic

    # SKIP: magnet danger + stretched + bad R:R
    if magnet_risk['danger'] and vwap_stretched and rr_ratio < cfg['MIN_RR_RATIO']:
        return 'SKIP', {**details, 'reason': 'Magnet danger + VWAP stretched + poor R:R'}

    # WAIT_DIP: price stretched above VWAP (for longs) or below (for shorts)
    if direction == 'LONG' and vwap_dist_pct > cfg['MAX_VWAP_STRETCH_PCT']:
        pullback_target = price - (price - vwap) * cfg['VWAP_PULLBACK_TARGET']
        if nearest_support and pullback_target > nearest_support['price']:
            dip_target = pullback_target
        elif nearest_support:
            dip_target = (nearest_support['price'] + price) / 2
        else:
            dip_target = vwap
        dip_rr = _compute_rr(direction, dip_target, sl_price, tp1_price)
        details['dip_target'] = round(dip_target, 2)
        details['dip_rr'] = round(dip_rr, 2)
        if dip_rr > rr_ratio * 1.3:
            return 'WAIT_DIP', {**details,
                'reason': f'Price stretched +{vwap_dist_pct:.1f}% from VWAP. Limit at ${dip_target:.2f} for better R:R ({dip_rr:.1f} vs {rr_ratio:.1f})'}

    if direction == 'SHORT' and vwap_dist_pct < -cfg['MAX_VWAP_STRETCH_PCT']:
        bounce_target = price + (vwap - price) * cfg['VWAP_PULLBACK_TARGET']
        if nearest_resistance and bounce_target < nearest_resistance['price']:
            dip_target = bounce_target
        elif nearest_resistance:
            dip_target = (nearest_resistance['price'] + price) / 2
        else:
            dip_target = vwap
        dip_rr = _compute_rr(direction, dip_target, sl_price, tp1_price)
        details['dip_target'] = round(dip_target, 2)
        details['dip_rr'] = round(dip_rr, 2)
        if dip_rr > rr_ratio * 1.3:
            return 'WAIT_DIP', {**details,
                'reason': f'Price stretched {vwap_dist_pct:.1f}% from VWAP. Limit at ${dip_target:.2f} for better R:R ({dip_rr:.1f} vs {rr_ratio:.1f})'}

    # WAIT_BREAKOUT: resistance too close for longs
    if direction == 'LONG' and resistance_too_close:
        break_price = nearest_resistance['price'] * (1 + cfg['BREAKOUT_BUFFER_PCT'] / 100)
        break_rr = _compute_rr(direction, break_price, sl_price, tp1_price)
        details['breakout_price'] = round(break_price, 2)
        details['breakout_rr'] = round(break_rr, 2)
        return 'WAIT_BREAKOUT', {**details,
            'reason': f'Resistance at ${nearest_resistance["price"]:.2f} too close. Enter on break above ${break_price:.2f}'}

    # WAIT_BREAKOUT: support too close for shorts
    if direction == 'SHORT' and support_too_close:
        break_price = nearest_support['price'] * (1 - cfg['BREAKOUT_BUFFER_PCT'] / 100)
        break_rr = _compute_rr(direction, break_price, sl_price, tp1_price)
        details['breakout_price'] = round(break_price, 2)
        details['breakout_rr'] = round(break_rr, 2)
        return 'WAIT_BREAKOUT', {**details,
            'reason': f'Support at ${nearest_support["price"]:.2f} too close. Enter on break below ${break_price:.2f}'}

    # SKIP: R:R too poor
    if rr_ratio < cfg['MIN_RR_RATIO']:
        return 'SKIP', {**details, 'reason': f'Poor R:R ({rr_ratio:.2f} < {cfg["MIN_RR_RATIO"]})'}

    # ENTER_NOW: all checks passed
    return 'ENTER_NOW', {**details, 'reason': f'Good entry. R:R {rr_ratio:.2f}, VWAP dist {vwap_dist_pct:+.1f}%'}


def score_entry_optimizer(direction, price, atr_1h, sr_levels, vwap,
                          liquidation_magnets, vol_ratio=None, swing_bias=None,
                          config=None):
    """
    Module-style scoring function for integration with the scanner.

    Returns:
        (action, score, details) where:
        - action: 'ENTER_NOW', 'WAIT_DIP', 'WAIT_BREAKOUT', 'SKIP'
        - score: 0.0-1.0 (1.0 = best entry quality)
        - details: dict with diagnostic info
    """
    action, details = evaluate_entry(
        direction, price, atr_1h, sr_levels, vwap,
        liquidation_magnets, vol_ratio, swing_bias, config
    )

    # Convert action to score
    score_map = {
        'ENTER_NOW': 0.9,
        'WAIT_DIP': 0.6,
        'WAIT_BREAKOUT': 0.5,
        'SKIP': 0.1,
    }
    score = score_map.get(action, 0.5)

    # Adjust score by R:R quality
    rr = details.get('rr_ratio', 0)
    if rr > 2.0:
        score = min(score + 0.1, 1.0)
    elif rr < 1.0:
        score = max(score - 0.1, 0.0)

    return action, score, details


def format_entry_advice(action, details):
    """Format entry advice for display."""
    lines = []
    if action == 'ENTER_NOW':
        lines.append(f"  ✅ ENTER NOW — {details.get('reason', '')}")
    elif action == 'WAIT_DIP':
        lines.append(f"  ⏳ WAIT FOR DIP — {details.get('reason', '')}")
        if 'dip_target' in details:
            lines.append(f"     Limit order at: ${details['dip_target']:.2f}")
            lines.append(f"     R:R at dip: {details.get('dip_rr', 'N/A')}")
    elif action == 'WAIT_BREAKOUT':
        lines.append(f"  🔓 WAIT FOR BREAKOUT — {details.get('reason', '')}")
        if 'breakout_price' in details:
            lines.append(f"     Enter on break: ${details['breakout_price']:.2f}")
            lines.append(f"     R:R at break: {details.get('breakout_rr', 'N/A')}")
    elif action == 'SKIP':
        lines.append(f"  ❌ SKIP — {details.get('reason', '')}")

    if 'sl_price' in details:
        lines.append(f"     SL: ${details['sl_price']:.2f} ({details.get('risk_pct', 0):.2f}%)")
    if 'tp1_price' in details:
        lines.append(f"     TP1: ${details['tp1_price']:.2f} ({details.get('reward_pct', 0):.2f}%)")
    if 'rr_ratio' in details:
        lines.append(f"     R:R: {details['rr_ratio']:.2f}")
    return '\n'.join(lines)


# ═══════════════════════════════════════════════════════════════
# WICK RECLAIM DETECTION
# ═══════════════════════════════════════════════════════════════

def detect_wick_reclaim(df_15m, idx, direction, swing_levels, config=None):
    """
    Detect sweep-and-reclaim pattern near swing levels.

    Bullish reclaim (LONG):
      - Price sweeps below a recent swing low (wick extends past)
      - Closes back above the swing low (reclaim)
      - Strong lower wick = buying pressure after stop-hunt

    Bearish reclaim (SHORT):
      - Price sweeps above a recent swing high (wick extends past)
      - Closes back below the swing high (reclaim)
      - Strong upper wick = selling pressure after stop-hunt

    Args:
        df_15m: 15m DataFrame
        idx: current bar index
        direction: 'LONG' or 'SHORT'
        swing_levels: list of (price, idx) tuples from M13
        config: optional config dict

    Returns:
        (action, score_adj, details)
        action: 'RECLAIM' | 'SLICE_THROUGH' | 'NONE'
        score_adj: ICS adjustment (positive = bonus, negative = penalty)
    """
    cfg = {**_DEFAULTS, **(config or {})}
    details = {}

    if not cfg.get('WICK_RECLAIM_ENABLED', True) or not swing_levels:
        return 'NONE', 0.0, details

    row = df_15m.iloc[idx]
    o, h, l, c = row['Open'], row['High'], row['Low'], row['Close']
    bar_range = h - l
    if bar_range <= 0:
        return 'NONE', 0.0, details

    body = abs(c - o)
    lower_wick = min(o, c) - l
    upper_wick = h - max(o, c)
    vol = row.get('Volume', 0)
    vol_ma = row.get('vol_ma20', 0)

    # Find nearest swing level within zone
    zone_pct = cfg['WICK_RECLAIM_ZONE_PCT'] / 100
    nearest_level = None
    nearest_dist = float('inf')

    for level_price, level_idx in swing_levels:
        if isinstance(level_price, (list, tuple)):
            level_price = level_price[0]
        dist_pct = abs(row['Close'] - level_price) / row['Close']
        if dist_pct < zone_pct and dist_pct < nearest_dist:
            # Only consider levels that are "in the right direction"
            if direction == 'LONG' and level_price < row['Close']:
                nearest_level = level_price
                nearest_dist = dist_pct
            elif direction == 'SHORT' and level_price > row['Close']:
                nearest_level = level_price
                nearest_dist = dist_pct

    if nearest_level is None:
        return 'NONE', 0.0, details

    details['nearest_swing_level'] = round(nearest_level, 2)
    details['distance_pct'] = round(nearest_dist * 100, 3)

    wick_ratio = cfg['WICK_RECLAIM_MIN_WICK_RATIO']
    vol_mult = cfg['WICK_RECLAIM_VOLUME_MULT']

    if direction == 'LONG':
        # Bullish reclaim: wick swept below swing low, closed above
        swept = l < nearest_level
        reclaimed = c > nearest_level
        has_wick = lower_wick >= bar_range * wick_ratio

        if swept and reclaimed and has_wick:
            # Volume confirmation (optional boost)
            vol_confirmed = vol > vol_ma * vol_mult if vol_ma > 0 else True
            details['pattern'] = 'SWEEP_RECLAIM'
            details['wick_ratio'] = round(lower_wick / bar_range, 2)
            details['vol_confirmed'] = vol_confirmed
            bonus = cfg['WICK_RECLAIM_BONUS']
            if vol_confirmed:
                bonus *= 1.25  # extra boost for volume confirmation
            return 'RECLAIM', bonus, details

        elif swept and not reclaimed:
            # Price sliced through — no reclaim, bearish
            details['pattern'] = 'SLICE_THROUGH'
            details['wick_ratio'] = round(lower_wick / bar_range, 2)
            return 'SLICE_THROUGH', -cfg['WICK_REJECT_PENALTY'], details

    elif direction == 'SHORT':
        # Bearish reclaim: wick swept above swing high, closed below
        swept = h > nearest_level
        reclaimed = c < nearest_level
        has_wick = upper_wick >= bar_range * wick_ratio

        if swept and reclaimed and has_wick:
            vol_confirmed = vol > vol_ma * vol_mult if vol_ma > 0 else True
            details['pattern'] = 'SWEEP_RECLAIM'
            details['wick_ratio'] = round(upper_wick / bar_range, 2)
            details['vol_confirmed'] = vol_confirmed
            bonus = cfg['WICK_RECLAIM_BONUS']
            if vol_confirmed:
                bonus *= 1.25
            return 'RECLAIM', bonus, details

        elif swept and not reclaimed:
            details['pattern'] = 'SLICE_THROUGH'
            details['wick_ratio'] = round(upper_wick / bar_range, 2)
            return 'SLICE_THROUGH', -cfg['WICK_REJECT_PENALTY'], details

    return 'NONE', 0.0, details

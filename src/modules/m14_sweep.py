"""
M14: Sweep-Retest-Reclaim Module

Detects institutional stop-hunt patterns:
  1. Sweep: Price pierces a swing level (taking out stops)
  2. Retest: Price returns to test the level
  3. Reclaim: Price closes back on the correct side with conviction

This is the "setup" phase in the discretionary workflow —
the confirmation of intent after identifying liquidity pockets.

Signal taxonomy:
  STRONG_RECLAIM  — sweep + reclaim + volume + wick conviction → high score
  WEAK_RECLAIM    — sweep + reclaim but no volume → moderate score
  SLICE_THROUGH   — price went through without reaction → penalty / block
  NO_SWEEP        — no sweep detected → neutral (pass-through)
"""

import numpy as np
import pandas as pd


# ═══════════════════════════════════════════════════════════════
# DEFAULTS
# ═══════════════════════════════════════════════════════════════

_M14_DEFAULTS = {
    'M14_ENABLED': True,
    'M14_WEIGHT': 0.08,
    # Sweep detection
    'M14_SWEEP_LOOKBACK': 20,        # bars to look back for sweep
    'M14_SWEEP_DEPTH_MIN': 0.001,    # min sweep depth as % of price (0.1%)
    'M14_SWEEP_DEPTH_MAX': 0.020,    # max sweep depth (2% — beyond that it's a break)
    # Reclaim detection
    'M14_RECLAIM_BARS': 3,           # bars after sweep to check for reclaim
    'M14_RECLAIM_WICK_RATIO': 0.40,  # min wick/body ratio for conviction
    # Volume confirmation
    'M14_VOL_CONFIRM_MULT': 1.2,     # volume must be 1.2x avg
    # Scoring
    'M14_STRONG_SCORE': 0.85,        # strong reclaim score
    'M14_WEAK_SCORE': 0.55,          # weak reclaim score
    'M14_SLICE_PENALTY': 0.30,       # slice-through score (low = penalty)
    'M14_NO_SWEEP_SCORE': 0.50,      # neutral — no sweep detected
}


def _cfg(config, key):
    if config and key in config:
        return config[key]
    return _M14_DEFAULTS[key]


# ═══════════════════════════════════════════════════════════════
# SWEEP DETECTION
# ═══════════════════════════════════════════════════════════════

def _find_recent_sweep(df_15m, idx, direction, swing_levels, lookback,
                       depth_min, depth_max):
    """
    Look for a recent sweep of a swing level within lookback bars.

    Returns:
        (sweep_found, sweep_details) or (False, {})
    """
    if not swing_levels or idx < lookback:
        return False, {}

    closes = df_15m['Close'].values
    highs = df_15m['High'].values
    lows = df_15m['Low'].values

    for level_price, level_idx in swing_levels:
        if isinstance(level_price, (list, tuple)):
            level_price = level_price[0]

        # Only check levels that are in the right direction
        if direction == 'LONG' and level_price >= closes[idx]:
            continue
        if direction == 'SHORT' and level_price <= closes[idx]:
            continue

        # Look for sweep within recent bars
        start = max(0, idx - lookback)
        for bar_i in range(start, idx + 1):
            bar_range = highs[bar_i] - lows[bar_i]
            if bar_range <= 0:
                continue

            if direction == 'LONG':
                # Sweep below: low pierces below the level
                sweep_depth = (level_price - lows[bar_i]) / level_price
                if depth_min <= sweep_depth <= depth_max:
                    # Check if it reclaimed (closed above)
                    reclaimed = closes[bar_i] > level_price
                    lower_wick = min(df_15m.iloc[bar_i]['Open'], closes[bar_i]) - lows[bar_i]
                    wick_ratio = lower_wick / bar_range

                    return True, {
                        'sweep_bar': bar_i,
                        'bars_ago': idx - bar_i,
                        'level_price': round(level_price, 2),
                        'sweep_depth_pct': round(sweep_depth * 100, 3),
                        'sweep_low': round(lows[bar_i], 2),
                        'reclaimed_same_bar': reclaimed,
                        'wick_ratio': round(wick_ratio, 3),
                    }

            elif direction == 'SHORT':
                # Sweep above: high pierces above the level
                sweep_depth = (highs[bar_i] - level_price) / level_price
                if depth_min <= sweep_depth <= depth_max:
                    reclaimed = closes[bar_i] < level_price
                    upper_wick = highs[bar_i] - max(df_15m.iloc[bar_i]['Open'], closes[bar_i])
                    wick_ratio = upper_wick / bar_range

                    return True, {
                        'sweep_bar': bar_i,
                        'bars_ago': idx - bar_i,
                        'level_price': round(level_price, 2),
                        'sweep_depth_pct': round(sweep_depth * 100, 3),
                        'sweep_high': round(highs[bar_i], 2),
                        'reclaimed_same_bar': reclaimed,
                        'wick_ratio': round(wick_ratio, 3),
                    }

    return False, {}


# ═══════════════════════════════════════════════════════════════
# RETEST / RECLAIM DETECTION
# ═══════════════════════════════════════════════════════════════

def _check_reclaim_after_sweep(df_15m, idx, direction, sweep_details,
                               reclaim_bars, wick_ratio_min, vol_mult):
    """
    After a sweep, check if price reclaimed the level with conviction.

    Returns:
        (reclaim_type, details)
        reclaim_type: 'STRONG' | 'WEAK' | 'NONE'
    """
    sweep_bar = sweep_details['sweep_bar']
    level = sweep_details['level_price']
    bars_since = idx - sweep_bar

    if bars_since > reclaim_bars + 5:
        # Too long ago — pattern is stale
        return 'NONE', {'reason': 'sweep_too_stale'}

    closes = df_15m['Close'].values
    highs = df_15m['High'].values
    lows = df_15m['Low'].values
    opens = df_15m['Open'].values
    volumes = df_15m['Volume'].values

    # Check current bar for reclaim conviction
    bar_range = highs[idx] - lows[idx]
    if bar_range <= 0:
        return 'NONE', {'reason': 'zero_range'}

    if direction == 'LONG':
        # Reclaim: price is above the level with a strong lower wick
        if closes[idx] <= level:
            return 'NONE', {'reason': 'not_reclaimed'}

        lower_wick = min(opens[idx], closes[idx]) - lows[idx]
        wick_ratio = lower_wick / bar_range
        body = abs(closes[idx] - opens[idx])

        # Strong: big wick + body closing above + volume
        vol_avg = np.mean(volumes[max(0, idx-20):idx]) if idx >= 20 else volumes[idx]
        vol_ok = volumes[idx] > vol_avg * vol_mult

        if wick_ratio >= wick_ratio_min and closes[idx] > opens[idx]:
            if vol_ok:
                return 'STRONG', {
                    'wick_ratio': round(wick_ratio, 3),
                    'vol_confirmed': True,
                    'bars_since_sweep': bars_since,
                }
            else:
                return 'WEAK', {
                    'wick_ratio': round(wick_ratio, 3),
                    'vol_confirmed': False,
                    'bars_since_sweep': bars_since,
                }

        # Green candle above level (even without big wick)
        if closes[idx] > opens[idx] and closes[idx] > level:
            return 'WEAK', {
                'wick_ratio': round(wick_ratio, 3),
                'vol_confirmed': vol_ok,
                'bars_since_sweep': bars_since,
                'note': 'green_candle_above',
            }

    elif direction == 'SHORT':
        if closes[idx] >= level:
            return 'NONE', {'reason': 'not_reclaimed'}

        upper_wick = highs[idx] - max(opens[idx], closes[idx])
        wick_ratio = upper_wick / bar_range

        vol_avg = np.mean(volumes[max(0, idx-20):idx]) if idx >= 20 else volumes[idx]
        vol_ok = volumes[idx] > vol_avg * vol_mult

        if wick_ratio >= wick_ratio_min and closes[idx] < opens[idx]:
            if vol_ok:
                return 'STRONG', {
                    'wick_ratio': round(wick_ratio, 3),
                    'vol_confirmed': True,
                    'bars_since_sweep': bars_since,
                }
            else:
                return 'WEAK', {
                    'wick_ratio': round(wick_ratio, 3),
                    'vol_confirmed': False,
                    'bars_since_sweep': bars_since,
                }

        if closes[idx] < opens[idx] and closes[idx] < level:
            return 'WEAK', {
                'wick_ratio': round(wick_ratio, 3),
                'vol_confirmed': vol_ok,
                'bars_since_sweep': bars_since,
                'note': 'red_candle_below',
            }

    return 'NONE', {'reason': 'no_reclaim_signal'}


# ═══════════════════════════════════════════════════════════════
# MAIN SCORING FUNCTION
# ═══════════════════════════════════════════════════════════════

def score_m14(df_15m, idx, direction, swing_levels, config=None, magnets=None):
    """
    Score M14: Sweep-Retest-Reclaim.

    Args:
        df_15m: 15m DataFrame
        idx: current bar index
        direction: 'LONG' or 'SHORT'
        swing_levels: list of (price, idx) from M13 swing detection
        config: optional config dict

    Returns:
        (status, score, details)
        status: 'PASS' | 'FAIL' | 'SKIP'
        score: 0.0-1.0
    """
    cfg = {**_M14_DEFAULTS, **(config or {})}

    if not cfg.get('M14_ENABLED', True):
        return 'SKIP', 0.5, {'reason': 'disabled'}

    if not swing_levels:
        return 'SKIP', 0.5, {'reason': 'no_swing_levels'}

    # Step 1: Find recent sweep
    sweep_found, sweep_details = _find_recent_sweep(
        df_15m, idx, direction, swing_levels,
        lookback=cfg['M14_SWEEP_LOOKBACK'],
        depth_min=cfg['M14_SWEEP_DEPTH_MIN'],
        depth_max=cfg['M14_SWEEP_DEPTH_MAX'],
    )

    if not sweep_found:
        # No sweep — check if price is approaching nearest unswept liquidity
        # (within 0.5% of a magnet in trade direction)
        approach_score = None
        if magnets:
            current_price = float(df_15m['Close'].iloc[idx])
            for mag_price, mag_vol, mag_strength in magnets:
                dist_pct = abs(mag_price - current_price) / current_price
                if dist_pct < 0.005:  # within 0.5%
                    if (direction == 'LONG' and mag_price > current_price) or \
                       (direction == 'SHORT' and mag_price < current_price):
                        approach_score = 0.45  # slightly below neutral — approaching
                        break
        if approach_score is not None:
            return 'FAIL', approach_score, {'reason': 'approaching_liquidity', 'note': 'price near unswept level — watch for sweep'}
        return 'SKIP', cfg['M14_NO_SWEEP_SCORE'], {'reason': 'no_sweep_detected'}

    # Step 2: Check for reclaim
    reclaim_type, reclaim_details = _check_reclaim_after_sweep(
        df_15m, idx, direction, sweep_details,
        reclaim_bars=cfg['M14_RECLAIM_BARS'],
        wick_ratio_min=cfg['M14_RECLAIM_WICK_RATIO'],
        vol_mult=cfg['M14_VOL_CONFIRM_MULT'],
    )

    details = {**sweep_details, **reclaim_details}

    if reclaim_type == 'STRONG':
        details['signal'] = 'STRONG_RECLAIM'
        return 'PASS', cfg['M14_STRONG_SCORE'], details

    elif reclaim_type == 'WEAK':
        details['signal'] = 'WEAK_RECLAIM'
        return 'PASS', cfg['M14_WEAK_SCORE'], details

    else:
        # No reclaim after sweep — sweep happened but conviction missing
        # This is a warning: institutional stop-hunt without follow-through
        if sweep_details.get('reclaimed_same_bar'):
            # Sweep bar itself closed on correct side (wick rejection)
            # but subsequent bars didn't confirm — weak signal
            details['signal'] = 'NO_RECLAIM'
            return 'FAIL', cfg['M14_SLICE_PENALTY'], details
        else:
            # Price swept through without any reaction — pure stop-hunt
            details['signal'] = 'SLICE_THROUGH'
            return 'FAIL', cfg['M14_SLICE_PENALTY'], details

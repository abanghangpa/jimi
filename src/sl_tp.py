"""
Liquidity-Aware SL/TP Placement

Shared by both scanner and engine so backtest matches live behavior.

Framework:
  SL  → Place in nearest liquidity void (no magnets/S/R/stops nearby)
        Fallback to ATR-based if no void found
  TP1 → Nearest unswept magnet/pool in trade direction
  TP2 → Next unswept pool beyond TP1
  TP3 → Furthest unswept pool or ATR extension
"""

import numpy as np


# ═══════════════════════════════════════════════════════════════
# DEFAULTS
# ═══════════════════════════════════════════════════════════════

_SL_TP_DEFAULTS = {
    # SL void detection
    'SL_VOID_BUFFER_PCT': 0.003,       # 0.3% buffer around levels = "clustered"
    'SL_VOID_MIN_DIST_PCT': 0.002,     # min SL distance as % of price
    'SL_VOID_MAX_DIST_PCT': 0.025,     # max SL distance (2.5% cap)
    'SL_ATR_STD': 1.0,                 # ATR fallback multiplier
    'SL_HARD_MAX_PCT': 0.02,           # hard max SL distance (2%)

    # TP targeting
    'TP1_USE_MAGNET': True,            # target nearest unswept magnet for TP1
    'TP1_MAGNET_MIN_DIST_PCT': 0.002,  # min distance to magnet (avoid entry-adjacent)
    'TP1_ATR': 0.8,                    # ATR fallback for TP1
    'TP2_ATR': 1.5,                    # ATR fallback for TP2
    'TP3_ATR': 2.5,                    # ATR fallback for TP3

    # Sweep gate
    'M14_ENTRY_GATE': False,           # require M14 sweep before signaling
}


def _cfg(config, key):
    if config and key in config:
        return config[key]
    return _SL_TP_DEFAULTS.get(key)


# ═══════════════════════════════════════════════════════════════
# LIQUIDITY VOID DETECTION (for SL)
# ═══════════════════════════════════════════════════════════════

def _collect_levels(price, direction, magnets, sr_levels, liq_levels, cfg):
    """Collect all liquidity levels near price that define 'clusters'."""
    buffer_pct = _cfg(cfg, 'SL_VOID_BUFFER_PCT')
    levels = []

    # Volume profile magnets (HVNs)
    if magnets:
        for m in magnets:
            p = m[0]  # (price, vol, strength)
            if abs(p - price) / price < 0.05:  # within 5%
                levels.append(p)

    # S/R levels
    if sr_levels:
        for sr in sr_levels:
            p = sr[0]  # (price, strength, touches, bounces, type)
            if abs(p - price) / price < 0.05:
                levels.append(p)

    # Liquidation/stop clusters from M15
    if liq_levels and isinstance(liq_levels, dict):
        for side in ('above', 'below'):
            for lvl in liq_levels.get(side, []):
                p = lvl.get('price', 0)
                if p > 0 and abs(p - price) / price < 0.05:
                    levels.append(p)

    return sorted(set(levels))


def find_liquidity_void(price, direction, magnets, sr_levels, liq_levels, atr_1h, cfg=None):
    """Find the nearest price level in a liquidity void for SL placement.

    A void = a zone with no magnets, S/R, or stop clusters nearby.

    Returns: SL price (float) or None if no void found.
    """
    cfg = cfg or {}
    buffer_pct = _cfg(cfg, 'SL_VOID_BUFFER_PCT')
    min_dist_pct = _cfg(cfg, 'SL_VOID_MIN_DIST_PCT')
    max_dist_pct = _cfg(cfg, 'SL_VOID_MAX_DIST_PCT')
    hard_max_pct = _cfg(cfg, 'SL_HARD_MAX_PCT')

    levels = _collect_levels(price, direction, magnets, sr_levels, liq_levels, cfg)

    if not levels:
        return None  # no levels → use ATR fallback

    # Sort levels by distance from price
    levels.sort(key=lambda p: abs(p - price))

    # Find voids between levels
    # A void = midpoint between two adjacent levels, if the gap > 2 * buffer
    voids = []

    # Add boundary voids (beyond the nearest level away from price)
    if direction == 'LONG':
        # SL goes below → look for voids below price
        below_levels = sorted([p for p in levels if p < price], reverse=True)
        if below_levels:
            # Void below the lowest nearby level
            lowest = below_levels[-1]
            void_candidate = lowest - price * buffer_pct
            if abs(price - void_candidate) / price >= min_dist_pct:
                voids.append(void_candidate)
        # Voids between levels
        for i in range(len(below_levels) - 1):
            gap = below_levels[i] - below_levels[i+1]
            if gap > 2 * price * buffer_pct:
                mid = (below_levels[i] + below_levels[i+1]) / 2
                if abs(price - mid) / price >= min_dist_pct:
                    voids.append(mid)
    else:
        # SL goes above → look for voids above price
        above_levels = sorted([p for p in levels if p > price])
        if above_levels:
            highest = above_levels[-1]
            void_candidate = highest + price * buffer_pct
            if abs(void_candidate - price) / price >= min_dist_pct:
                voids.append(void_candidate)
        for i in range(len(above_levels) - 1):
            gap = above_levels[i+1] - above_levels[i]
            if gap > 2 * price * buffer_pct:
                mid = (above_levels[i] + above_levels[i+1]) / 2
                if abs(mid - price) / price >= min_dist_pct:
                    voids.append(mid)

    if not voids:
        return None

    # Pick the closest void that's within max distance
    voids_in_range = [v for v in voids if abs(v - price) / price <= max_dist_pct]
    if not voids_in_range:
        return None

    # Pick closest to price
    best_void = min(voids_in_range, key=lambda v: abs(v - price))

    # Enforce hard max
    if abs(best_void - price) / price > hard_max_pct:
        if direction == 'LONG':
            best_void = price - price * hard_max_pct
        else:
            best_void = price + price * hard_max_pct

    return best_void


# ═══════════════════════════════════════════════════════════════
# UNSWEPT POOL DETECTION (for TP)
# ═══════════════════════════════════════════════════════════════

def find_next_unswept(price, direction, magnets, liq_levels, exclude_below=None, cfg=None):
    """Find the next unswept liquidity pool in trade direction.

    Args:
        price: current price
        direction: 'LONG' or 'SHORT'
        magnets: volume profile magnets [(price, vol, strength), ...]
        liq_levels: dict with 'above'/'below' lists of level dicts
        exclude_below: skip pools closer than this price (for TP2/TP3)
        cfg: config dict

    Returns: target price (float) or None
    """
    cfg = cfg or {}
    min_dist_pct = _cfg(cfg, 'TP1_MAGNET_MIN_DIST_PCT')

    candidates = []

    # From magnets (HVNs — absorption zones)
    if magnets:
        for m in magnets:
            p = m[0]
            dist_pct = abs(p - price) / price
            if dist_pct < min_dist_pct:
                continue
            if direction == 'LONG' and p > price:
                if exclude_below and p <= exclude_below:
                    continue
                candidates.append((p, dist_pct, m[2]))  # (price, dist, strength)
            elif direction == 'SHORT' and p < price:
                if exclude_below and p >= exclude_below:
                    continue
                candidates.append((p, dist_pct, m[2]))

    # From liquidation levels (unswept stops/liquidations)
    if liq_levels and isinstance(liq_levels, dict):
        side = 'above' if direction == 'LONG' else 'below'
        for lvl in liq_levels.get(side, []):
            p = lvl.get('price', 0)
            swept = lvl.get('swept', False)
            if p <= 0 or swept:
                continue
            dist_pct = abs(p - price) / price
            if dist_pct < min_dist_pct:
                continue
            if exclude_below:
                if direction == 'LONG' and p <= exclude_below:
                    continue
                if direction == 'SHORT' and p >= exclude_below:
                    continue
            strength = lvl.get('strength', 1)
            candidates.append((p, dist_pct, strength))

    if not candidates:
        return None

    # Sort by distance, pick closest
    candidates.sort(key=lambda x: x[1])
    return candidates[0][0]


# ═══════════════════════════════════════════════════════════════
# MAIN FUNCTION: CALCULATE ALL LEVELS
# ═══════════════════════════════════════════════════════════════

def calc_trade_levels(entry_price, direction, atr_1h, vol_ratio,
                      magnets, sr_levels, liq_levels, cfg=None):
    """Calculate SL/TP using liquidity-aware logic with ATR fallback.

    Args:
        entry_price: entry price
        direction: 'LONG' or 'SHORT'
        atr_1h: 1-hour ATR value
        vol_ratio: volume ratio (for TP multipliers)
        magnets: volume profile magnets [(price, vol, strength), ...]
        sr_levels: S/R levels [(price, strength, touches, bounces, type), ...]
        liq_levels: dict with 'above'/'below' lists (from M15) or None
        cfg: config dict

    Returns:
        dict with sl, tp1, tp2, tp3, sl_source, tp1_source, tp2_source, tp3_source
    """
    cfg = cfg or {}
    atr = float(atr_1h) if not np.isnan(atr_1h) else entry_price * 0.01

    # ── SL: Try liquidity void first ──
    void_sl = find_liquidity_void(
        entry_price, direction, magnets, sr_levels, liq_levels, atr, cfg)

    if void_sl is not None:
        sl = void_sl
        sl_source = 'LIQUIDITY_VOID'
    else:
        # ATR fallback
        sl_dist = min(_cfg(cfg, 'SL_ATR_STD') * atr,
                      _cfg(cfg, 'SL_HARD_MAX_PCT') * entry_price)
        if direction == 'LONG':
            sl = entry_price - sl_dist
        else:
            sl = entry_price + sl_dist
        sl_source = 'ATR'

    # ── TP1: Nearest unswept pool ──
    tp1_target = None
    if _cfg(cfg, 'TP1_USE_MAGNET'):
        tp1_target = find_next_unswept(
            entry_price, direction, magnets, liq_levels, cfg=cfg)

    if tp1_target is not None:
        tp1 = tp1_target
        tp1_source = 'UNSWEPT_POOL'
    else:
        tp1_dist = _cfg(cfg, 'TP1_ATR') * atr
        if direction == 'LONG':
            tp1 = entry_price + tp1_dist
        else:
            tp1 = entry_price - tp1_dist
        tp1_source = 'ATR'

    # ── TP2: Next unswept pool beyond TP1, or ATR ──
    tp2_target = find_next_unswept(
        entry_price, direction, magnets, liq_levels,
        exclude_below=tp1 if direction == 'LONG' else None,
        cfg=cfg)

    # For SHORT, exclude_above
    if direction == 'SHORT' and tp2_target is None:
        tp2_target = find_next_unswept(
            entry_price, direction, magnets, liq_levels,
            exclude_below=tp1,  # tp1 is below for SHORT
            cfg=cfg)

    if tp2_target is not None and abs(tp2_target - entry_price) > abs(tp1 - entry_price):
        tp2 = tp2_target
        tp2_source = 'UNSWEPT_POOL'
    else:
        tp2_mult = _cfg(cfg, 'TP2_ATR')
        tp2_dist = tp2_mult * atr
        if direction == 'LONG':
            tp2 = entry_price + tp2_dist
        else:
            tp2 = entry_price - tp2_dist
        tp2_source = 'ATR'

    # ── TP3: Furthest pool or ATR ──
    tp3_target = find_next_unswept(
        entry_price, direction, magnets, liq_levels,
        exclude_below=tp2 if direction == 'LONG' else None,
        cfg=cfg)

    if direction == 'SHORT' and tp3_target is None:
        tp3_target = find_next_unswept(
            entry_price, direction, magnets, liq_levels,
            exclude_below=tp2,
            cfg=cfg)

    if tp3_target is not None and abs(tp3_target - entry_price) > abs(tp2 - entry_price):
        tp3 = tp3_target
        tp3_source = 'UNSWEPT_POOL'
    else:
        tp3_mult = _cfg(cfg, 'TP3_ATR')
        tp3_dist = tp3_mult * atr
        if direction == 'LONG':
            tp3 = entry_price + tp3_dist
        else:
            tp3 = entry_price - tp3_dist
        tp3_source = 'ATR'

    return {
        'sl': float(sl),
        'tp1': float(tp1),
        'tp2': float(tp2),
        'tp3': float(tp3),
        'sl_source': sl_source,
        'tp1_source': tp1_source,
        'tp2_source': tp2_source,
        'tp3_source': tp3_source,
        'sl_pct': abs(entry_price - sl) / entry_price * 100,
        'tp1_pct': abs(tp1 - entry_price) / entry_price * 100,
    }


def check_sweep_gate(m14_status, m14_score, cfg=None):
    """Check if M14 sweep gate is enabled and passed.

    Returns: (passed, reason)
    """
    cfg = cfg or {}
    if not _cfg(cfg, 'M14_ENTRY_GATE'):
        return True, 'gate_disabled'

    if m14_status == 'PASS':
        return True, 'sweep_confirmed'

    return False, f'M14 gate: {m14_status} (sweep required)'

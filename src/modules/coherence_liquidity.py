"""
Coherence Check & Liquidity-Aware Execution

Two enhancements to the trade entry pipeline:
1. Coherence Check — detects conflicting module states before entry
2. Liquidity-Aware TP — adjusts TP fractions based on volume density between targets
"""

import numpy as np


# ═══════════════════════════════════════════════════════════════
# COHERENCE CHECK
# ═══════════════════════════════════════════════════════════════

def check_coherence(direction, m4_div, m5_details, m13_bias, m9_regime,
                    m7_score=0.5, m2_status='NEUTRAL', config=None):
    """
    Check if module states tell a coherent story.

    Returns:
        (is_coherent, conflicts, penalty)
        - is_coherent: True if no major conflicts
        - conflicts: list of conflict descriptions
        - penalty: ICS penalty to apply (0.0 = no penalty)
    """
    cfg = config or {}
    conflicts = []
    penalty = 0.0

    # v7.2: Normalize _BASE variants (BULLISH_BASE → BULLISH, etc.)
    if isinstance(m4_div, str) and m4_div.endswith('_BASE'):
        m4_div = m4_div.replace('_BASE', '')

    # M4 divergence vs direction
    if m4_div == 'BEARISH' and direction == 'LONG':
        conflicts.append('M4 bearish divergence on LONG')
        penalty += cfg.get('COHERENCE_M4_PENALTY', 0.04)
    elif m4_div == 'BULLISH' and direction == 'SHORT':
        conflicts.append('M4 bullish divergence on SHORT')
        penalty += cfg.get('COHERENCE_M4_PENALTY', 0.04)

    # M5 magnet vs direction
    if isinstance(m5_details, dict):
        magnet_dir = m5_details.get('magnet_direction', 'NEUTRAL')
        if magnet_dir == 'BELOW' and direction == 'LONG':
            conflicts.append('M5 magnet below (pulls against LONG)')
            penalty += cfg.get('COHERENCE_M5_PENALTY', 0.03)
        elif magnet_dir == 'ABOVE' and direction == 'SHORT':
            conflicts.append('M5 magnet above (pulls against SHORT)')
            penalty += cfg.get('COHERENCE_M5_PENALTY', 0.03)

    # M13 bias vs direction (only when M13 is available, i.e. not in chop)
    if m13_bias == 'BEARISH' and direction == 'LONG':
        conflicts.append('M13 bearish structure on LONG')
        penalty += cfg.get('COHERENCE_M13_PENALTY', 0.03)
    elif m13_bias == 'BULLISH' and direction == 'SHORT':
        conflicts.append('M13 bullish structure on SHORT')
        penalty += cfg.get('COHERENCE_M13_PENALTY', 0.03)

    # M7 macro vs direction
    if m7_score < 0.35 and direction == 'LONG':
        conflicts.append('M7 strongly bearish on LONG')
        penalty += cfg.get('COHERENCE_M7_PENALTY', 0.02)
    elif m7_score < 0.35 and direction == 'SHORT':
        # M7 bearish + SHORT = coherent, no penalty
        pass
    elif m7_score > 0.65 and direction == 'SHORT':
        conflicts.append('M7 strongly bullish on SHORT')
        penalty += cfg.get('COHERENCE_M7_PENALTY', 0.02)

    # Hard block if too many conflicts
    max_conflicts = cfg.get('COHERENCE_MAX_CONFLICTS', 3)
    hard_block = len(conflicts) >= max_conflicts

    penalty = min(penalty, cfg.get('COHERENCE_MAX_PENALTY', 0.10))

    return not hard_block, conflicts, penalty


# ═══════════════════════════════════════════════════════════════
# LIQUIDITY-AWARE TP
# ═══════════════════════════════════════════════════════════════

def compute_liquidity_aware_tp(entry_price, tp1, tp2, tp3, direction,
                               bin_centers, vol_profile, config=None):
    """
    Adjust TP fractions based on volume density between entry and targets.

    Dense volume cluster between TP1 and TP2 = friction zone.
    Take more off at TP1 (price may bounce before reaching TP2).

    Void between TP1 and TP2 = open air.
    Take less off at TP1 (price may fly through to TP2).

    Returns:
        dict with:
            tp1_close_frac: adjusted TP1 close fraction
            tp2_close_frac: adjusted TP2 close fraction
            density_tp1_tp2: volume density between TP1 and TP2
            density_entry_tp1: volume density between entry and TP1
            has_friction: True if dense cluster detected between TP1-TP2
            has_void: True if low density between TP1-TP2
    """
    cfg = config or {}

    base_tp1_close = cfg.get('TP1_CLOSE', 0.30)
    base_tp2_close = cfg.get('TP2_CLOSE', 0.30)

    result = {
        'tp1_close_frac': base_tp1_close,
        'tp2_close_frac': base_tp2_close,
        'density_tp1_tp2': 0.5,
        'density_entry_tp1': 0.5,
        'has_friction': False,
        'has_void': False,
    }

    if bin_centers is None or vol_profile is None:
        return result

    # Normalize volume profile to 0-1
    vol_max = vol_profile.max()
    if vol_max <= 0:
        return result
    vol_norm = vol_profile / vol_max

    # Compute density in price zones
    if direction == 'LONG':
        entry_zone_low, entry_zone_high = entry_price * 0.998, entry_price * 1.002
        tp1_zone_low, tp1_zone_high = tp1 * 0.998, tp1 * 1.002
        tp2_zone_low, tp2_zone_high = tp2 * 0.998, tp2 * 1.002
        # Between entry and TP1
        mask_entry_tp1 = (bin_centers >= entry_price) & (bin_centers <= tp1)
        # Between TP1 and TP2
        mask_tp1_tp2 = (bin_centers >= tp1) & (bin_centers <= tp2)
    else:
        mask_entry_tp1 = (bin_centers <= entry_price) & (bin_centers >= tp1)
        mask_tp1_tp2 = (bin_centers <= tp1) & (bin_centers >= tp2)

    density_entry_tp1 = float(vol_norm[mask_entry_tp1].mean()) if mask_entry_tp1.any() else 0.5
    density_tp1_tp2 = float(vol_norm[mask_tp1_tp2].mean()) if mask_tp1_tp2.any() else 0.5

    result['density_entry_tp1'] = round(density_entry_tp1, 3)
    result['density_tp1_tp2'] = round(density_tp1_tp2, 3)

    # Friction threshold: dense cluster between TP1 and TP2
    friction_thresh = cfg.get('LIQUIDITY_FRICTION_THRESHOLD', 0.60)
    void_thresh = cfg.get('LIQUIDITY_VOID_THRESHOLD', 0.20)

    tp1_adjust_up = cfg.get('LIQUIDITY_TP1_ADJUST_UP', 0.08)   # take more at TP1
    tp1_adjust_down = cfg.get('LIQUIDITY_TP1_ADJUST_DOWN', 0.05)  # take less at TP1

    if density_tp1_tp2 > friction_thresh:
        # Dense cluster ahead — take more off at TP1
        result['tp1_close_frac'] = min(base_tp1_close + tp1_adjust_up, 0.50)
        result['tp2_close_frac'] = base_tp2_close - tp1_adjust_up * 0.5
        result['has_friction'] = True
    elif density_tp1_tp2 < void_thresh:
        # Open air — let it run, take less at TP1
        result['tp1_close_frac'] = max(base_tp1_close - tp1_adjust_down, 0.15)
        result['tp2_close_frac'] = base_tp2_close + tp1_adjust_down * 0.5
        result['has_void'] = True

    return result


def compute_stop_risk(entry_price, sl, direction, bin_centers, vol_profile, config=None):
    """
    Check if a dense volume cluster sits between entry and stop loss.
    If so, the stop is at risk of being swept by a liquidity grab.

    Returns:
        dict with:
            has_stop_risk: True if dense cluster between entry and SL
            density_to_sl: volume density in the stop zone
    """
    cfg = config or {}
    result = {'has_stop_risk': False, 'density_to_sl': 0.5}

    if bin_centers is None or vol_profile is None:
        return result

    vol_max = vol_profile.max()
    if vol_max <= 0:
        return result
    vol_norm = vol_profile / vol_max

    if direction == 'LONG':
        mask = (bin_centers >= sl) & (bin_centers <= entry_price)
    else:
        mask = (bin_centers >= entry_price) & (bin_centers <= sl)

    if not mask.any():
        return result

    density = float(vol_norm[mask].mean())
    result['density_to_sl'] = round(density, 3)

    risk_thresh = cfg.get('STOP_RISK_THRESHOLD', 0.55)
    result['has_stop_risk'] = density > risk_thresh

    return result

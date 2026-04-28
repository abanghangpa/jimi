"""M5: Liquidation Magnet — Volume Profile + Cascade + Structural Targets.

Signal taxonomy:
  - Swing H/L  → Stop liquidity targets (magnets) — where stops cluster
  - HVNs       → Absorption zones — momentum dies here, use for SL/invalidation
  - FVGs       → Inefficiency — price reverts to fill these
  - Order Blocks → Institutional rejection zones — ideal SL placement
"""

import numpy as np
import pandas as pd


def build_volume_profile(highs, lows, closes, volumes, n_bins=50, lookback=672):
    h = highs[-lookback:]
    l = lows[-lookback:]
    v = volumes[-lookback:]
    price_min, price_max = np.min(l), np.max(h)
    if price_max == price_min:
        return None, None, None

    bin_edges = np.linspace(price_min, price_max, n_bins + 1)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    vol_profile = np.zeros(n_bins)
    bar_ranges = h - l
    bar_ranges[bar_ranges == 0] = 1

    for j in range(n_bins):
        overlap_low = np.maximum(l, bin_edges[j])
        overlap_high = np.minimum(h, bin_edges[j+1])
        overlap = np.maximum(overlap_high - overlap_low, 0)
        proportion = overlap / bar_ranges
        vol_profile[j] = np.sum(v * proportion)

    return bin_centers, vol_profile, bin_edges


def find_magnets(bin_centers, vol_profile, n_magnets=5, min_gap_pct=0.005):
    """Find High Volume Nodes (HVNs) — absorption zones, NOT targets."""
    if vol_profile is None or len(vol_profile) == 0:
        return []
    mean_vol = np.mean(vol_profile)
    if mean_vol == 0:
        return []

    peaks = []
    for i in range(1, len(vol_profile) - 1):
        if (vol_profile[i] > vol_profile[i-1] and
            vol_profile[i] > vol_profile[i+1] and
            vol_profile[i] > mean_vol * 1.2):
            peaks.append((bin_centers[i], vol_profile[i], vol_profile[i] / mean_vol))

    peaks.sort(key=lambda x: x[1], reverse=True)
    filtered = []
    for peak in peaks:
        if not any(abs(peak[0] - e[0]) / e[0] < min_gap_pct for e in filtered):
            filtered.append(peak)
    return filtered[:n_magnets]


def find_gaps(bin_centers, vol_profile, n_gaps=5):
    """Find Low Volume Nodes (LVNs) — vacuum zones."""
    if vol_profile is None or len(vol_profile) == 0:
        return []
    mean_vol = np.mean(vol_profile)
    if mean_vol == 0:
        return []
    gaps = [(bin_centers[i], vol_profile[i]) for i in range(len(vol_profile))
            if vol_profile[i] < mean_vol * 0.3]
    gaps.sort(key=lambda x: x[1])
    return gaps[:n_gaps]


def calc_magnetic_pull(current_price, magnets, direction):
    """HVN pull — kept for backward compatibility. See calc_swing_magnetic_pull for real targets."""
    if not magnets:
        return 0.0, None, None
    relevant = []
    for price, vol, strength in magnets:
        if direction == 'LONG' and price > current_price:
            dist = (price - current_price) / current_price
            relevant.append((price, vol, strength, dist))
        elif direction == 'SHORT' and price < current_price:
            dist = (current_price - price) / current_price
            relevant.append((price, vol, strength, dist))
    if not relevant:
        return 0.0, None, None
    relevant.sort(key=lambda x: x[3])
    nearest = relevant[0]
    dist_factor = max(0, 1.0 - nearest[3] / 0.02)
    strength_factor = min(nearest[2] / 3.0, 1.0)
    return dist_factor * 0.6 + strength_factor * 0.4, nearest[0], nearest[3]


def calc_gap_acceleration(current_price, gaps, direction):
    if not gaps:
        return 0.0, False
    gap_between = False
    nearest_dist = float('inf')
    for price, vol in gaps:
        if direction == 'LONG' and price > current_price:
            dist = (price - current_price) / current_price
        elif direction == 'SHORT' and price < current_price:
            dist = (current_price - price) / current_price
        else:
            continue
        if dist < nearest_dist:
            nearest_dist = dist
        if dist < 0.005:
            gap_between = True
    if nearest_dist == float('inf'):
        return 0.0, False
    return max(0, 1.0 - nearest_dist / 0.01), gap_between


# ═══════════════════════════════════════════════════════════════
# STRUCTURAL SIGNALS — Swing, FVG, OB, HVN
# ═══════════════════════════════════════════════════════════════

def calc_swing_magnetic_pull(current_price, swings, direction):
    """Calculate magnetic pull toward swing highs/lows (stop liquidity targets).

    Swing H/Ls are where stop-loss clusters accumulate — the real 'magnets'.
    Unlike HVNs (which are traded zones), swings are structural extremes
    where breakout stops and protective stops pile up.

    Returns: (pull_score, nearest_target, distance_pct)
    """
    if not swings:
        return 0.0, None, None

    relevant = []
    for s in swings:
        price = s['price']
        if direction == 'LONG' and price > current_price:
            dist = (price - current_price) / current_price
            relevant.append((price, dist, s.get('type', 'H')))
        elif direction == 'SHORT' and price < current_price:
            dist = (current_price - price) / current_price
            relevant.append((price, dist, s.get('type', 'L')))

    if not relevant:
        return 0.0, None, None

    relevant.sort(key=lambda x: x[1])
    nearest_price, nearest_dist, swing_type = relevant[0]

    dist_factor = max(0, 1.0 - nearest_dist / 0.03)
    pull = dist_factor * 0.7 + 0.3  # base 0.3 for any valid target

    return min(pull, 1.0), nearest_price, nearest_dist


def calc_fvg_reversion_score(current_price, fvgs, direction):
    """Score reversion potential toward unfilled Fair Value Gaps.

    FVGs are inefficiencies — price tends to revert to fill them.
    For LONG: bullish FVGs above are targets (price pulls up to fill)
    For SHORT: bearish FVGs below are targets (price pulls down to fill)
    """
    if not fvgs:
        return 0.0, None, None

    relevant = []
    for fvg in fvgs:
        if fvg.get('filled', False):
            continue
        mid = fvg['mid']
        if direction == 'LONG' and mid > current_price:
            dist = (mid - current_price) / current_price
            relevant.append((mid, dist, fvg.get('gap_pct', 0)))
        elif direction == 'SHORT' and mid < current_price:
            dist = (current_price - mid) / current_price
            relevant.append((mid, dist, fvg.get('gap_pct', 0)))

    if not relevant:
        return 0.0, None, None

    relevant.sort(key=lambda x: x[1])
    nearest_price, nearest_dist, gap_pct = relevant[0]

    dist_factor = max(0, 1.0 - nearest_dist / 0.02)
    size_factor = min(gap_pct / 0.5, 1.0)
    score = dist_factor * 0.6 + size_factor * 0.4

    return min(score, 1.0), nearest_price, nearest_dist


def calc_ob_sl_zone(current_price, order_blocks, direction):
    """Evaluate order block proximity for stop-loss placement.

    OBs are institutional rejection zones — ideal SL levels.
    For LONG: bullish OBs below are support (SL just below OB)
    For SHORT: bearish OBs above are resistance (SL just above OB)
    """
    if not order_blocks:
        return 0.0, None, None

    relevant = []
    for ob in order_blocks:
        ob_type = ob.get('type', '')
        if direction == 'LONG' and ob_type == 'BULLISH' and ob['top'] < current_price:
            dist = (current_price - ob['top']) / current_price
            relevant.append((ob, dist))
        elif direction == 'SHORT' and ob_type == 'BEARISH' and ob['bottom'] > current_price:
            dist = (ob['bottom'] - current_price) / current_price
            relevant.append((ob, dist))

    if not relevant:
        return 0.0, None, None

    relevant.sort(key=lambda x: x[1])
    nearest_ob, nearest_dist = relevant[0]

    dist_factor = max(0, 1.0 - nearest_dist / 0.02)
    strength_factor = min(nearest_ob.get('strength', 1.0) / 3.0, 1.0)
    score = dist_factor * 0.6 + strength_factor * 0.4

    sl_price = nearest_ob['bottom'] if direction == 'LONG' else nearest_ob['top']
    return min(score, 1.0), sl_price, nearest_dist


def calc_hvn_absorption(current_price, magnets, direction):
    """Evaluate HVN proximity as absorption/invalidation zones.

    HVNs are high-volume traded zones — price tends to get absorbed here.
    NOT targets. These are where momentum dies and reversals happen.
    Used as warnings: if trade enters a HVN, expect resistance.
    """
    if not magnets:
        return 0.0, None

    hazards = []
    for price, vol, strength in magnets:
        if direction == 'LONG' and price > current_price:
            dist = (price - current_price) / current_price
            hazards.append((price, dist, strength))
        elif direction == 'SHORT' and price < current_price:
            dist = (current_price - price) / current_price
            hazards.append((price, dist, strength))

    if not hazards:
        return 0.0, None

    hazards.sort(key=lambda x: x[1])
    nearest_price, nearest_dist, strength = hazards[0]

    dist_factor = max(0, 1.0 - nearest_dist / 0.02)
    strength_factor = min(strength / 3.0, 1.0)
    absorption_risk = dist_factor * 0.6 + strength_factor * 0.4

    return min(absorption_risk, 1.0), nearest_price


# ═══════════════════════════════════════════════════════════════
# CASCADE DETECTION
# ═══════════════════════════════════════════════════════════════

def find_support_resistance(df_15m, idx=None, lookback=672, n_levels=10,
                             bin_pct=0.002, touch_pct=0.004, bounce_pct=0.003,
                             bounce_bars=8, min_touches=3):
    """Find support/resistance levels based on price rejection behavior."""
    if idx is None:
        idx = len(df_15m) - 1
    if idx < lookback:
        return []

    start = max(0, idx - lookback + 1)
    highs = df_15m['High'].values[start:idx+1].astype(float)
    lows = df_15m['Low'].values[start:idx+1].astype(float)
    closes = df_15m['Close'].values[start:idx+1].astype(float)
    current_price = closes[-1]

    if len(closes) < 20:
        return []

    price_min, price_max = lows.min(), highs.max()
    price_range = price_max - price_min
    if price_range <= 0:
        return []

    n_bins = max(int(price_range / (current_price * bin_pct)), 20)
    bin_edges = np.linspace(price_min, price_max, n_bins + 1)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

    levels = []
    for bi in range(len(bin_centers)):
        bc = bin_centers[bi]
        touches = 0
        bounces = 0

        for i in range(len(closes)):
            touch_dist = abs(lows[i] - bc) / bc
            if touch_dist <= touch_pct:
                touches += 1
                bounced = False
                for j in range(i+1, min(i+1+bounce_bars, len(closes))):
                    if abs(closes[j] - bc) / bc >= bounce_pct:
                        bounced = True
                        break
                if bounced:
                    bounces += 1

        if touches >= min_touches and bounces >= min_touches:
            consistency = bounces / touches if touches > 0 else 0
            strength = touches * consistency
            sr_type = 'SUPPORT' if current_price > bc else 'RESISTANCE'
            levels.append((bc, strength, touches, bounces, sr_type))

    levels.sort(key=lambda x: x[1], reverse=True)
    filtered = []
    for level in levels:
        if not any(abs(level[0] - e[0]) / e[0] < bin_pct for e in filtered):
            filtered.append(level)

    filtered.sort(key=lambda x: abs(x[0] - current_price))
    return filtered[:n_levels]


def detect_cascade_mode(df_15m, idx, magnets, direction):
    """Detect if price is in CASCADE mode approaching a magnet."""
    if idx < 20 or not magnets:
        return False, 'NONE', 0.0, {}

    closes = df_15m['Close'].values.astype(float)
    highs = df_15m['High'].values.astype(float)
    lows = df_15m['Low'].values.astype(float)
    volumes = df_15m['Volume'].values.astype(float)
    current_price = closes[idx]

    approach_magnets = []
    for price, vol, strength in magnets:
        dist = abs(price - current_price) / current_price
        if dist < 0.015:
            approach_magnets.append((price, vol, strength, dist))

    if not approach_magnets:
        return False, 'NONE', 0.0, {}

    approach_magnets.sort(key=lambda x: x[3])
    nearest_mag_dist = approach_magnets[0][3]

    if idx >= 8:
        momentum_4 = (closes[idx] - closes[idx-4]) / closes[idx-4]
        momentum_8 = (closes[idx] - closes[idx-8]) / closes[idx-8]
    else:
        momentum_4 = 0
        momentum_8 = 0

    vol_avg = np.mean(volumes[max(0,idx-20):idx])
    vol_spike = volumes[idx] / vol_avg if vol_avg > 0 else 0

    current_range = highs[idx] - lows[idx]
    avg_range = np.mean(highs[max(0,idx-20):idx] - lows[max(0,idx-20):idx])
    range_expansion = current_range / avg_range if avg_range > 0 else 0

    if 'Taker buy base asset volume' in df_15m.columns:
        taker_buy = df_15m['Taker buy base asset volume'].iloc[idx]
        total_vol = df_15m['Volume'].iloc[idx]
        taker_ratio = taker_buy / total_vol if total_vol > 0 else 0.5
    else:
        taker_ratio = 0.50

    if idx >= 4:
        making_new_low = lows[idx] <= np.min(lows[max(0,idx-4):idx])
        making_new_high = highs[idx] >= np.max(highs[max(0,idx-4):idx])
    else:
        making_new_low = False
        making_new_high = False

    cascade_down = making_new_low and momentum_4 < -0.003 and vol_spike > 1.3
    cascade_up = making_new_high and momentum_4 > 0.003 and vol_spike > 1.3

    is_cascade = False
    cascade_dir = 'NONE'
    cascade_strength = 0.0

    if cascade_down or cascade_up:
        if nearest_mag_dist < 0.01:
            is_cascade = True
            cascade_dir_raw = 'DOWN' if cascade_down else 'UP'

            if (direction == 'LONG' and cascade_dir_raw == 'DOWN') or \
               (direction == 'SHORT' and cascade_dir_raw == 'UP'):
                cascade_dir = 'AGAINST'
            else:
                cascade_dir = 'WITH'

            cascade_strength = min(
                (abs(momentum_4) / 0.01) * 0.4 +
                (vol_spike / 3.0) * 0.4 +
                (range_expansion / 2.0) * 0.2,
                1.0
            )

    details = {
        'momentum_4': round(momentum_4 * 100, 3),
        'momentum_8': round(momentum_8 * 100, 3),
        'vol_spike': round(vol_spike, 2),
        'range_expansion': round(range_expansion, 2),
        'taker_ratio': round(taker_ratio, 3),
        'making_new_low': making_new_low,
        'making_new_high': making_new_high,
        'nearest_mag_dist': round(nearest_mag_dist * 100, 3),
        'cascade_down': cascade_down,
        'cascade_up': cascade_up,
    }

    return is_cascade, cascade_dir, cascade_strength, details


# ═══════════════════════════════════════════════════════════════
# MAIN SCORING
# ═══════════════════════════════════════════════════════════════

def score_m5(df_15m, idx, direction, config, n_bins=50, lookback=672, m13_details=None):
    """Score M5: Structural Targets + Volume Profile + Cascade.

    Signal taxonomy:
      - Swing H/L  → Stop liquidity targets (magnets)
      - HVNs       → Absorption zones (SL/invalidation)
      - FVGs       → Inefficiency / reversion targets
      - Order Blocks → Institutional rejection (SL placement)
    """
    if idx < lookback:
        return 'FAIL', 0.0, {'reason': 'insufficient data'}

    highs = df_15m['High'].values.astype(float)
    lows = df_15m['Low'].values.astype(float)
    closes = df_15m['Close'].values.astype(float)
    volumes = df_15m['Volume'].values.astype(float)
    current_price = closes[idx]

    # ── Volume Profile (for HVN/LVN identification) ──
    bin_centers, vol_profile, bin_edges = build_volume_profile(
        highs[:idx+1], lows[:idx+1], closes[:idx+1], volumes[:idx+1],
        n_bins=n_bins, lookback=lookback)
    if bin_centers is None:
        return 'FAIL', 0.0, {'reason': 'profile build failed'}

    hvns = find_magnets(bin_centers, vol_profile)  # HVNs = absorption zones
    lvns = find_gaps(bin_centers, vol_profile)      # LVNs = vacuum zones

    # ── Extract M13 structural data ──
    swings = []
    fvgs = []
    order_blocks = []
    if m13_details:
        # Reconstruct swing list from M13 details
        for item in m13_details.get('swing_highs', []):
            if isinstance(item, (list, tuple)) and len(item) >= 2:
                swings.append({'price': item[0], 'idx': item[1], 'type': 'H'})
        for item in m13_details.get('swing_lows', []):
            if isinstance(item, (list, tuple)) and len(item) >= 2:
                swings.append({'price': item[0], 'idx': item[1], 'type': 'L'})

        fvgs = m13_details.get('fvgs', [])
        order_blocks = m13_details.get('order_blocks', [])

    # ── Signal 1: Swing Magnet (stop liquidity targets) ──
    swing_pull, swing_target, swing_dist = calc_swing_magnetic_pull(
        current_price, swings, direction)

    # ── Signal 2: FVG Reversion (inefficiency targets) ──
    fvg_score, fvg_target, fvg_dist = calc_fvg_reversion_score(
        current_price, fvgs, direction)

    # ── Signal 3: HVN Absorption (risk zones, not targets) ──
    hvn_risk, hvn_zone = calc_hvn_absorption(current_price, hvns, direction)

    # ── Signal 4: OB SL Zone (institutional levels) ──
    ob_score, ob_sl, ob_dist = calc_ob_sl_zone(current_price, order_blocks, direction)

    # ── Signal 5: Volume Profile Skew ──
    accel_score, gap_between = calc_gap_acceleration(current_price, lvns, direction)
    current_bin = np.searchsorted(bin_edges, current_price) - 1
    current_bin = max(0, min(current_bin, len(vol_profile) - 1))
    vol_above = np.sum(vol_profile[current_bin+1:])
    vol_below = np.sum(vol_profile[:current_bin])
    total_vol = vol_above + vol_below
    skew = (vol_above / total_vol if direction == 'LONG' else vol_below / total_vol) if total_vol > 0 else 0.5
    skew_score = min(skew / 0.7, 1.0)

    # ── Signal 6: Cascade Detection ──
    is_cascade, cascade_dir, cascade_strength, cascade_details = detect_cascade_mode(
        df_15m, idx, hvns, direction)

    # ── Composite Score ──
    # Swing targets (stop liquidity) = primary magnet
    # FVG reversion = secondary target
    # VP skew + accel = flow confirmation
    # OB proximity = SL quality bonus
    # HVN absorption = penalty (momentum risk)
    score = (
        swing_pull * 0.40 +
        fvg_score * 0.20 +
        skew_score * 0.15 +
        accel_score * 0.15 +
        ob_score * 0.10
    )

    # Cascade modifier
    if is_cascade:
        if cascade_dir == 'WITH':
            cascade_bonus = cascade_strength * 0.3
            score = min(score + cascade_bonus, 1.0)
        elif cascade_dir == 'AGAINST':
            cascade_penalty = cascade_strength * 0.5
            score = max(score - cascade_penalty, 0.0)

    # HVN absorption penalty: if a strong HVN sits between entry and target
    if hvn_risk > 0.5:
        score *= (1.0 - hvn_risk * 0.3)

    # ── Inversion ──
    # Calibration across 2017-2026 shows M5 score is anti-predictive:
    # the 0.3-0.5 bucket consistently outperforms 0.6+. High raw scores
    # indicate price is near structural targets (already moved), not that
    # a good setup exists. Invert so low raw → high normalized score.
    inverted_score = 1.0 - score

    details = {
        # Raw (pre-inversion) for diagnostics
        'raw_score': round(score, 3),
        # Swing targets (stop liquidity)
        'swing_pull': round(swing_pull, 3),
        'swing_target': round(swing_target, 2) if swing_target else None,
        'swing_target_dist': round(swing_dist * 100, 3) if swing_dist else None,
        # FVG reversion
        'fvg_score': round(fvg_score, 3),
        'fvg_target': round(fvg_target, 2) if fvg_target else None,
        'fvg_target_dist': round(fvg_dist * 100, 3) if fvg_dist else None,
        # HVN absorption (risk, not target)
        'hvn_risk': round(hvn_risk, 3),
        'hvn_zone': round(hvn_zone, 2) if hvn_zone else None,
        # OB SL zone
        'ob_score': round(ob_score, 3),
        'ob_sl': round(ob_sl, 2) if ob_sl else None,
        'ob_dist': round(ob_dist * 100, 3) if ob_dist else None,
        # Volume profile
        'accel_score': round(accel_score, 3),
        'skew_score': round(skew_score, 3),
        'gap_between': gap_between,
        # Cascade
        'cascade': is_cascade,
        'cascade_dir': cascade_dir,
        'cascade_strength': round(cascade_strength, 3),
        'cascade_details': cascade_details,
        # Legacy (VP magnets = HVNs, kept for backward compat)
        'magnets': [(round(p, 2), round(s, 2)) for p, _, s in hvns[:3]],
        'nearest_magnet': round(hvn_zone, 2) if hvn_zone else None,
        'pull_score': round(swing_pull, 3),
    }

    return ('PASS', inverted_score, details) if inverted_score >= config['M5_MIN_SCORE'] else ('FAIL', inverted_score, details)


def detect_cascade_setup(df_15m, idx, lookback=96):
    """Quick cascade detection for signal scanner."""
    if idx < lookback:
        return {'cascade': False, 'reason': 'insufficient data'}
    closes = df_15m['Close'].values.astype(float)
    volumes = df_15m['Volume'].values.astype(float)
    highs = df_15m['High'].values.astype(float)
    lows = df_15m['Low'].values.astype(float)

    momentum = abs(closes[idx] - closes[idx-4]) / closes[idx-4] if idx >= 4 else 0
    vol_avg = np.mean(volumes[max(0,idx-20):idx])
    vol_spike = volumes[idx] / vol_avg if vol_avg > 0 else 0
    current_range = highs[idx] - lows[idx]
    avg_range = np.mean(highs[max(0,idx-20):idx] - lows[max(0,idx-20):idx])
    range_expansion = current_range / avg_range if avg_range > 0 else 0

    cascade = momentum > 0.005 and vol_spike > 1.5 and range_expansion > 1.3
    return {
        'cascade': cascade,
        'momentum': round(momentum * 100, 3),
        'vol_spike': round(vol_spike, 2),
        'range_expansion': round(range_expansion, 2),
    }

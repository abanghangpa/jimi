"""M5: Liquidation Magnet — Volume Profile + Cascade Detection."""

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


def score_m5(df_15m, idx, direction, config, n_bins=50, lookback=672):
    """Score M5: Liquidation Magnet + Cascade."""
    if idx < lookback:
        return 'FAIL', 0.0, {'reason': 'insufficient data'}

    highs = df_15m['High'].values.astype(float)
    lows = df_15m['Low'].values.astype(float)
    closes = df_15m['Close'].values.astype(float)
    volumes = df_15m['Volume'].values.astype(float)
    current_price = closes[idx]

    bin_centers, vol_profile, bin_edges = build_volume_profile(
        highs[:idx+1], lows[:idx+1], closes[:idx+1], volumes[:idx+1],
        n_bins=n_bins, lookback=lookback)
    if bin_centers is None:
        return 'FAIL', 0.0, {'reason': 'profile build failed'}

    magnets = find_magnets(bin_centers, vol_profile)
    gaps = find_gaps(bin_centers, vol_profile)
    pull_score, nearest_magnet, magnet_dist = calc_magnetic_pull(current_price, magnets, direction)
    accel_score, gap_between = calc_gap_acceleration(current_price, gaps, direction)

    current_bin = np.searchsorted(bin_edges, current_price) - 1
    current_bin = max(0, min(current_bin, len(vol_profile) - 1))
    vol_above = np.sum(vol_profile[current_bin+1:])
    vol_below = np.sum(vol_profile[:current_bin])
    total_vol = vol_above + vol_below
    skew = (vol_above / total_vol if direction == 'LONG' else vol_below / total_vol) if total_vol > 0 else 0.5
    skew_score = min(skew / 0.7, 1.0)

    is_cascade, cascade_dir, cascade_strength, cascade_details = detect_cascade_mode(
        df_15m, idx, magnets, direction)

    if is_cascade:
        if cascade_dir == 'WITH':
            cascade_bonus = cascade_strength * 0.4
            pull_score = min(pull_score + cascade_bonus, 1.0)
        elif cascade_dir == 'AGAINST':
            cascade_penalty = cascade_strength * 0.6
            pull_score = max(pull_score - cascade_penalty, 0.0)

    score = pull_score * 0.5 + accel_score * 0.3 + skew_score * 0.2
    details = {
        'magnets': [(round(p, 2), round(s, 2)) for p, _, s in magnets[:3]],
        'gaps': [round(p, 2) for p, _ in gaps[:3]],
        'nearest_magnet': round(nearest_magnet, 2) if nearest_magnet else None,
        'magnet_dist_pct': round(magnet_dist * 100, 3) if magnet_dist else None,
        'pull_score': round(pull_score, 3),
        'accel_score': round(accel_score, 3),
        'skew_score': round(skew_score, 3),
        'gap_between': gap_between,
        'cascade': is_cascade,
        'cascade_dir': cascade_dir,
        'cascade_strength': round(cascade_strength, 3),
        'cascade_details': cascade_details,
    }
    return ('PASS', score, details) if score >= config['M5_MIN_SCORE'] else ('FAIL', score, details)


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

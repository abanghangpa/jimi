"""M15: Liquidation Level Estimator — Where stops and liquidations actually cluster.

Unlike volume profile (which shows where people TRADED), this estimates
where their STOPS and LIQUIDATIONS are based on:

  1. Position entry distribution (volume-weighted recent prices)
  2. Leverage tiers (3x-125x → liquidation price math)
  3. Stop-loss clustering (just beyond S/R, swing H/L, round numbers)
  4. OI concentration (high OI + high leverage = cascade risk)
  5. Order book depth (actual resting orders if available)

The output is a set of "liquidation zones" — price levels where a cascade
of stop/liquidation triggers is most likely.
"""

import numpy as np
import requests
from collections import defaultdict

from src.config import CONFIG

# ═══════════════════════════════════════════════════════════════
# Binance ETH/USDT leverage & maintenance margin tiers
# ═══════════════════════════════════════════════════════════════

# Leverage tier → (max notional, maintenance margin rate)
# Source: Binance futures leverage brackets
LEVERAGE_TIERS = {
    125: (50000, 0.004),
    100: (100000, 0.005),
    75:  (250000, 0.006),
    50:  (500000, 0.010),
    25:  (1000000, 0.020),
    20:  (2000000, 0.025),
    10:  (5000000, 0.050),
    5:   (10000000, 0.100),
    3:   (20000000, 0.100),
    2:   (50000000, 0.100),
    1:   (100000000, 0.100),
}

# Estimated leverage distribution (what % of traders use each tier)
# Based on industry research: most retail uses 10-50x
LEVERAGE_DISTRIBUTION = {
    3:   0.05,   # conservative
    5:   0.10,
    10:  0.20,   # popular
    20:  0.25,   # most popular
    25:  0.10,
    50:  0.15,   # degen zone
    75:  0.05,
    100: 0.05,
    125: 0.05,   # max degen
}


def calc_liq_price(entry, leverage, side='LONG', mmr=0.004):
    """Calculate liquidation price for isolated margin.

    Simplified formula:
      Long:  liq = entry * (1 - (1 - mmr) / leverage)
      Short: liq = entry * (1 + (1 - mmr) / leverage)
    """
    if side == 'LONG':
        return entry * (1 - (1 - mmr) / leverage)
    else:
        return entry * (1 + (1 - mmr) / leverage)


def estimate_entry_distribution(df_15m, idx, lookback=96):
    """Estimate where positions were entered using volume-weighted price distribution.

    Uses the last N candles (default 96 = 24h) to build a probability
    distribution of entry prices. Higher volume candles = more positions opened.
    """
    start = max(0, idx - lookback + 1)
    highs = df_15m['High'].values[start:idx+1].astype(float)
    lows = df_15m['Low'].values[start:idx+1].astype(float)
    closes = df_15m['Close'].values[start:idx+1].astype(float)
    volumes = df_15m['Volume'].values[start:idx+1].astype(float)

    # Build volume-weighted price histogram
    price_min, price_max = np.min(lows), np.max(highs)
    n_bins = 50
    bin_edges = np.linspace(price_min, price_max, n_bins + 1)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

    # Distribute volume across price range of each candle
    entry_density = np.zeros(n_bins)
    for i in range(len(closes)):
        candle_low, candle_high = lows[i], highs[i]
        candle_range = candle_high - candle_low
        if candle_range == 0:
            candle_range = 1
        for j in range(n_bins):
            overlap_low = max(candle_low, bin_edges[j])
            overlap_high = min(candle_high, bin_edges[j+1])
            overlap = max(overlap_high - overlap_low, 0)
            proportion = overlap / candle_range
            entry_density[j] += volumes[i] * proportion

    # Normalize to probability
    total = np.sum(entry_density)
    if total > 0:
        entry_density = entry_density / total

    return bin_centers, entry_density, bin_edges


def estimate_liquidation_cascades(current_price, entry_centers, entry_density,
                                   oi_usd, ls_ratio, direction_bias=None):
    """Estimate where liquidation cascades would trigger.

    For each entry price bucket × leverage tier, calculate the liquidation
    price and weight by (entry_probability × leverage_usage × OI).
    """
    # Aggregate liquidation density by price level
    n_liq_bins = 80
    liq_range_min = current_price * 0.92  # -8%
    liq_range_max = current_price * 1.08  # +8%
    liq_bin_edges = np.linspace(liq_range_min, liq_range_max, n_liq_bins + 1)
    liq_bin_centers = (liq_bin_edges[:-1] + liq_bin_edges[1:]) / 2

    long_liq_density = np.zeros(n_liq_bins)
    short_liq_density = np.zeros(n_liq_bins)

    # For each entry price bucket
    for i, (entry_price, density) in enumerate(zip(entry_centers, entry_density)):
        if density < 0.001:  # skip negligible
            continue

        # For each leverage tier
        for lev, lev_weight in LEVERAGE_DISTRIBUTION.items():
            # Get MMR for this leverage
            mmr = 0.004  # default
            for tier_lev, (_, tier_mmr) in sorted(LEVERAGE_TIERS.items()):
                if lev <= tier_lev:
                    mmr = tier_mmr
                    break

            # Long liquidation (below entry)
            long_liq = calc_liq_price(entry_price, lev, 'LONG', mmr)
            # Short liquidation (above entry)
            short_liq = calc_liq_price(entry_price, lev, 'SHORT', mmr)

            # Weight = entry_probability × leverage_usage × notional_factor
            weight = density * lev_weight * (lev / 20.0)  # normalize by typical leverage

            # Place into bins
            liq_idx_long = np.searchsorted(liq_bin_edges, long_liq) - 1
            liq_idx_short = np.searchsorted(liq_bin_edges, short_liq) - 1

            if 0 <= liq_idx_long < n_liq_bins:
                long_liq_density[liq_idx_long] += weight
            if 0 <= liq_idx_short < n_liq_bins:
                short_liq_density[liq_idx_short] += weight

    # Scale by OI (bigger OI = bigger cascade potential)
    oi_scale = oi_usd / 1e9  # billions
    long_liq_density *= oi_scale
    short_liq_density *= oi_scale

    return liq_bin_centers, long_liq_density, short_liq_density


def find_stop_clusters(df_15m, idx, sr_levels, lookback=96):
    """Estimate where stop-losses cluster.

    Stops are typically placed:
    - Just beyond S/R levels (1-2% past)
    - Below/above swing highs/lows
    - Near round numbers ($2300, $2350, etc.)
    """
    current_price = df_15m['Close'].values[idx]
    stop_zones = []

    # 1. Stops beyond S/R levels
    for level in sr_levels:
        price, strength, sr_type, touches, bounces = level
        # Stops sit ~0.5-1.5% beyond the level
        if sr_type == 'SUPPORT':
            # Long stops below support
            stop_zone = price * 0.993  # 0.7% below
            stop_zones.append({
                'price': stop_zone, 'type': 'LONG_STOP',
                'source': f'S/R {sr_type} ${price:.0f}',
                'strength': strength * 0.8,
                'cluster_size': touches,
            })
        else:
            # Short stops above resistance
            stop_zone = price * 1.007  # 0.7% above
            stop_zones.append({
                'price': stop_zone, 'type': 'SHORT_STOP',
                'source': f'S/R {sr_type} ${price:.0f}',
                'strength': strength * 0.8,
                'cluster_size': touches,
            })

    # 2. Stops at round numbers
    round_step = 10 if current_price < 5000 else 50
    base = int(current_price / round_step) * round_step
    for r in range(base - round_step * 5, base + round_step * 6, round_step):
        if r <= 0:
            continue
        dist = abs(r - current_price) / current_price
        if dist < 0.05:  # within 5%
            # Stops cluster just below round numbers (for longs)
            # and just above round numbers (for shorts)
            stop_zones.append({
                'price': r - round_step * 0.1, 'type': 'LONG_STOP',
                'source': f'Round ${r}',
                'strength': max(0, 1.0 - dist / 0.05) * 50,
                'cluster_size': int(max(0, 1.0 - dist / 0.05) * 30),
            })
            stop_zones.append({
                'price': r + round_step * 0.1, 'type': 'SHORT_STOP',
                'source': f'Round ${r}',
                'strength': max(0, 1.0 - dist / 0.05) * 50,
                'cluster_size': int(max(0, 1.0 - dist / 0.05) * 30),
            })

    # 3. Stops at recent swing highs/lows
    start = max(0, idx - lookback)
    highs = df_15m['High'].values[start:idx+1].astype(float)
    lows = df_15m['Low'].values[start:idx+1].astype(float)

    # Find local extremes (simple peak/trough detection)
    for i in range(2, len(highs) - 2):
        # Swing high
        if highs[i] > highs[i-1] and highs[i] > highs[i-2] and \
           highs[i] > highs[i+1] and highs[i] > highs[i+2]:
            dist = abs(highs[i] - current_price) / current_price
            if dist < 0.03:
                stop_zones.append({
                    'price': highs[i] * 1.002, 'type': 'SHORT_STOP',
                    'source': f'Swing H ${highs[i]:.0f}',
                    'strength': max(0, 1.0 - dist / 0.03) * 40,
                    'cluster_size': 5,
                })
        # Swing low
        if lows[i] < lows[i-1] and lows[i] < lows[i-2] and \
           lows[i] < lows[i+1] and lows[i] < lows[i+2]:
            dist = abs(lows[i] - current_price) / current_price
            if dist < 0.03:
                stop_zones.append({
                    'price': lows[i] * 0.998, 'type': 'LONG_STOP',
                    'source': f'Swing L ${lows[i]:.0f}',
                    'strength': max(0, 1.0 - dist / 0.03) * 40,
                    'cluster_size': 5,
                })

    return stop_zones


def fetch_order_book_depth(symbol="ETHUSDT", limit=20):
    """Fetch order book to find actual resting liquidity."""
    try:
        r = requests.get(
            "https://fapi.binance.com/fapi/v1/depth",
            params={"symbol": symbol, "limit": limit},
            timeout=5
        )
        r.raise_for_status()
        data = r.json()

        bids = [(float(p), float(q)) for p, q in data.get('bids', [])]
        asks = [(float(p), float(q)) for p, q in data.get('asks', [])]

        return bids, asks
    except Exception:
        return [], []


def find_formation_time(df_15m, idx, zone_price, lookback=288):
    """Find when a zone was first formed (first bar that touched it).

    Returns the bar index of first touch, or None if never touched.
    """
    start = max(0, idx - lookback + 1)
    highs = df_15m['High'].values[start:idx+1].astype(float)
    lows = df_15m['Low'].values[start:idx+1].astype(float)

    for i in range(len(highs)):
        if lows[i] <= zone_price <= highs[i]:
            return start + i
    return None


def find_bounce_rejection_zones(df_15m, idx, lookback=288):
    """Find NEW liquidation zones from bounce points and rejection wicks.

    These are freshly formed stop clusters from recent price action —
    where traders entered positions and placed stops during recovery/consolidation.

    Returns zones with 'formed_at' bar index for proper sweep tracking.
    """
    start = max(0, idx - lookback + 1)
    highs = df_15m['High'].values[start:idx+1].astype(float)
    lows = df_15m['Low'].values[start:idx+1].astype(float)
    closes = df_15m['Close'].values[start:idx+1].astype(float)
    opens = df_15m['Open'].values[start:idx+1].astype(float)
    volumes = df_15m['Volume'].values[start:idx+1].astype(float)
    times = df_15m['Open time'].values[start:idx+1]
    current_price = closes[-1]

    zones = []

    # 1. Swing lows → long entry zones (stops below)
    for i in range(2, len(highs) - 2):
        if lows[i] < lows[i-1] and lows[i] < lows[i-2] and \
           lows[i] < lows[i+1] and lows[i] < lows[i+2]:
            dist = abs(lows[i] - current_price) / current_price
            if dist < 0.03:
                # Stops sit 0.3-0.8% below the swing low
                stop_price = lows[i] * 0.995
                # Higher strength for closer zones; base 40 for swing stops
                base_strength = max(0, 1.0 - dist / 0.03) * 60 + 20
                # Bonus for recovery-formed zones (last 48h = 192 bars)
                if i > len(highs) - 192:
                    base_strength *= 1.3
                zones.append({
                    'price': round(stop_price, 2),
                    'type': 'LONG_STOP',
                    'source': f'Swing L ${lows[i]:.0f} @ {str(times[i])[:16]}',
                    'strength': round(min(base_strength, 100), 1),
                    'cluster_size': 3,
                    'cascade_risk': 'MED',
                    'formed_at': start + i,
                    'fresh': True,
                })

    # 2. Swing highs → short entry zones (stops above)
    for i in range(2, len(highs) - 2):
        if highs[i] > highs[i-1] and highs[i] > highs[i-2] and \
           highs[i] > highs[i+1] and highs[i] > highs[i+2]:
            dist = abs(highs[i] - current_price) / current_price
            if dist < 0.03:
                stop_price = highs[i] * 1.005
                base_strength = max(0, 1.0 - dist / 0.03) * 60 + 20
                if i > len(highs) - 192:
                    base_strength *= 1.3
                zones.append({
                    'price': round(stop_price, 2),
                    'type': 'SHORT_STOP',
                    'source': f'Swing H ${highs[i]:.0f} @ {str(times[i])[:16]}',
                    'strength': round(min(base_strength, 100), 1),
                    'cluster_size': 3,
                    'cascade_risk': 'MED',
                    'formed_at': start + i,
                    'fresh': True,
                })

    # 3. Rejection wicks → stop clusters at the extreme
    for i in range(len(highs)):
        body = abs(closes[i] - opens[i])
        total = highs[i] - lows[i]
        if total == 0 or body == 0:
            continue

        upper_wick = highs[i] - max(opens[i], closes[i])
        lower_wick = min(opens[i], closes[i]) - lows[i]

        # Long upper rejection → stops above the wick high
        if upper_wick > body * 1.5 and upper_wick > total * 0.35:
            dist = abs(highs[i] - current_price) / current_price
            if dist < 0.03:
                base_strength = max(0, 1.0 - dist / 0.03) * 55 + 15
                if i > len(highs) - 192:
                    base_strength *= 1.3
                zones.append({
                    'price': round(highs[i] * 1.002, 2),
                    'type': 'SHORT_STOP',
                    'source': f'Rejection H ${highs[i]:.0f} @ {str(times[i])[:16]}',
                    'strength': round(min(base_strength, 100), 1),
                    'cluster_size': 2,
                    'cascade_risk': 'MED',
                    'formed_at': start + i,
                    'fresh': True,
                })

        # Long lower rejection → stops below the wick low
        if lower_wick > body * 1.5 and lower_wick > total * 0.35:
            dist = abs(lows[i] - current_price) / current_price
            if dist < 0.03:
                base_strength = max(0, 1.0 - dist / 0.03) * 55 + 15
                if i > len(highs) - 192:
                    base_strength *= 1.3
                zones.append({
                    'price': round(lows[i] * 0.998, 2),
                    'type': 'LONG_STOP',
                    'source': f'Rejection L ${lows[i]:.0f} @ {str(times[i])[:16]}',
                    'strength': round(min(base_strength, 100), 1),
                    'cluster_size': 2,
                    'cascade_risk': 'MED',
                    'formed_at': start + i,
                    'fresh': True,
                })

    # 4. Consolidation zones → breakout stops
    for i in range(4, len(highs)):
        window_range = np.mean(highs[i-4:i+1] - lows[i-4:i+1])
        current_range = highs[i] - lows[i]
        if current_range < window_range * 0.5:  # Tight bar
            mid = (highs[i] + lows[i]) / 2
            dist = abs(mid - current_price) / current_price
            if dist < 0.02:
                # Stops above and below the consolidation
                zones.append({
                    'price': round(highs[i] * 1.003, 2),
                    'type': 'SHORT_STOP',
                    'source': f'Consolidation H ${highs[i]:.0f} @ {str(times[i])[:16]}',
                    'strength': round(max(0, 1.0 - dist / 0.02) * 35, 1),
                    'cluster_size': 2,
                    'cascade_risk': 'LOW',
                    'formed_at': start + i,
                    'fresh': True,
                })
                zones.append({
                    'price': round(lows[i] * 0.997, 2),
                    'type': 'LONG_STOP',
                    'source': f'Consolidation L ${lows[i]:.0f} @ {str(times[i])[:16]}',
                    'strength': round(max(0, 1.0 - dist / 0.02) * 35, 1),
                    'cluster_size': 2,
                    'cascade_risk': 'LOW',
                    'formed_at': start + i,
                    'fresh': True,
                })

    # 5. High-volume nodes from recovery (where most positions accumulated)
    # Cluster by price and weight by volume
    if len(closes) > 20:
        price_min, price_max = np.min(lows), np.max(highs)
        n_bins = 40
        bin_edges = np.linspace(price_min, price_max, n_bins + 1)
        bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
        vol_profile = np.zeros(n_bins)

        for i in range(len(closes)):
            for j in range(n_bins):
                overlap_low = max(lows[i], bin_edges[j])
                overlap_high = min(highs[i], bin_edges[j+1])
                overlap = max(overlap_high - overlap_low, 0)
                candle_range = highs[i] - lows[i]
                if candle_range > 0:
                    vol_profile[j] += volumes[i] * (overlap / candle_range)

        # Find peaks (high-volume nodes)
        mean_vol = np.mean(vol_profile)
        for i in range(1, len(vol_profile) - 1):
            if vol_profile[i] > vol_profile[i-1] and vol_profile[i] > vol_profile[i+1] and \
               vol_profile[i] > mean_vol * 1.5:
                price = bin_centers[i]
                dist = abs(price - current_price) / current_price
                if dist < 0.03:
                    strength_ratio = vol_profile[i] / mean_vol
                    # Determine if this is above or below current price
                    if price < current_price:
                        ztype = 'LONG_LIQ'  # Long entries here → their liqs below
                    else:
                        ztype = 'SHORT_LIQ'  # Short entries here → their liqs above

                    # Find when this node was formed (first high-volume bar in the zone)
                    formed_bar = None
                    for k in range(len(closes)):
                        if lows[k] <= price <= highs[k] and volumes[k] > mean_vol:
                            formed_bar = start + k
                            break

                    zones.append({
                        'price': round(float(price), 2),
                        'type': ztype,
                        'source': f'HVN {strength_ratio:.1f}x avg vol',
                        'strength': round(min(strength_ratio * 20, 80), 1),
                        'cluster_size': int(vol_profile[i] / mean_vol),
                        'cascade_risk': 'HIGH' if strength_ratio > 3 else 'MED' if strength_ratio > 2 else 'LOW',
                        'formed_at': formed_bar,
                        'fresh': True,
                    })

    return zones


def estimate_liquidity_levels(df_15m, idx, sr_levels, oi_usd, ls_ratio,
                               direction_bias=None, order_book=None):
    """Main function: combine all sources into liquidation level estimates.

    Returns sorted list of liquidity zones with:
      - price: estimated level
      - type: LONG_LIQ | SHORT_LIQ | LONG_STOP | SHORT_STOP | BID_WALL | ASK_WALL
      - strength: composite score (0-100)
      - source: what generated it
      - cascade_risk: estimated cascade potential (LOW/MED/HIGH)
    """
    current_price = df_15m['Close'].values[idx]

    # 1. Entry distribution → liquidation cascades
    entry_centers, entry_density, _ = estimate_entry_distribution(df_15m, idx)
    liq_centers, long_liq, short_liq = estimate_liquidation_cascades(
        current_price, entry_centers, entry_density, oi_usd, ls_ratio, direction_bias)

    # Find peaks in liquidation density
    liq_zones = []
    for i in range(1, len(liq_centers) - 1):
        # Long liquidation peak
        if long_liq[i] > long_liq[i-1] and long_liq[i] > long_liq[i+1]:
            strength = min(long_liq[i] / max(np.max(long_liq), 1) * 100, 100)
            if strength > 5:
                cascade = 'HIGH' if strength > 60 else 'MED' if strength > 30 else 'LOW'
                liq_zones.append({
                    'price': round(float(liq_centers[i]), 2),
                    'type': 'LONG_LIQ',
                    'strength': round(strength, 1),
                    'source': 'OI+leverage est.',
                    'cascade_risk': cascade,
                    'dist_pct': round((liq_centers[i] - current_price) / current_price * 100, 2),
                })

        # Short liquidation peak
        if short_liq[i] > short_liq[i-1] and short_liq[i] > short_liq[i+1]:
            strength = min(short_liq[i] / max(np.max(short_liq), 1) * 100, 100)
            if strength > 5:
                cascade = 'HIGH' if strength > 60 else 'MED' if strength > 30 else 'LOW'
                liq_zones.append({
                    'price': round(float(liq_centers[i]), 2),
                    'type': 'SHORT_LIQ',
                    'strength': round(strength, 1),
                    'source': 'OI+leverage est.',
                    'cascade_risk': cascade,
                    'dist_pct': round((liq_centers[i] - current_price) / current_price * 100, 2),
                })

    # 2. Stop clusters
    stop_zones = find_stop_clusters(df_15m, idx, sr_levels)
    for sz in stop_zones:
        sz['dist_pct'] = round((sz['price'] - current_price) / current_price * 100, 2)
        sz['cascade_risk'] = 'HIGH' if sz['strength'] > 50 else 'MED' if sz['strength'] > 25 else 'LOW'
        liq_zones.append(sz)

    # 3. Order book walls
    if order_book:
        bids, asks = order_book
        # Find large resting orders (walls)
        bid_vols = [q for _, q in bids]
        ask_vols = [q for _, q in asks]
        bid_mean = np.mean(bid_vols) if bid_vols else 0
        ask_mean = np.mean(ask_vols) if ask_vols else 0

        for price, qty in bids:
            if qty > bid_mean * 3:  # 3x average = wall
                strength = min(qty / bid_mean * 10, 100)
                liq_zones.append({
                    'price': round(price, 2),
                    'type': 'BID_WALL',
                    'strength': round(strength, 1),
                    'source': f'Order book ({qty:.0f} ETH)',
                    'cascade_risk': 'LOW',  # walls absorb, don't cascade
                    'dist_pct': round((price - current_price) / current_price * 100, 2),
                })

        for price, qty in asks:
            if qty > ask_mean * 3:
                strength = min(qty / ask_mean * 10, 100)
                liq_zones.append({
                    'price': round(price, 2),
                    'type': 'ASK_WALL',
                    'strength': round(strength, 1),
                    'source': f'Order book ({qty:.0f} ETH)',
                    'cascade_risk': 'LOW',
                    'dist_pct': round((price - current_price) / current_price * 100, 2),
                })

    # Deduplicate nearby zones of the SAME type (within 0.3%).
    # Different types at the same price represent distinct signals
    # (e.g. LONG_LIQ vs SHORT_STOP) and must not be merged.
    liq_zones.sort(key=lambda x: x['price'])
    deduped = []
    for zone in liq_zones:
        if not deduped or abs(zone['price'] - deduped[-1]['price']) / zone['price'] > 0.003 \
                or zone['type'] != deduped[-1]['type']:
            deduped.append(zone)
        else:
            # Same type, same price bucket — keep the stronger one
            if zone['strength'] > deduped[-1]['strength']:
                deduped[-1] = zone

    # ── Add fresh bounce/rejection zones ──
    # These capture newly formed stop clusters from recent swing points,
    # rejection wicks, and consolidation zones that the OI/leverage model misses.
    fresh_zones = find_bounce_rejection_zones(df_15m, idx, lookback=288)
    current_px = df_15m['Close'].values[idx]
    for fz in fresh_zones:
        # Ensure dist_pct is set
        if 'dist_pct' not in fz:
            fz['dist_pct'] = round((fz['price'] - current_px) / current_px * 100, 2)
        # Only add if not already covered by a FRESH zone (don't dedup against OI zones).
        # Fresh zones represent a different signal (stop clusters from price action)
        # than OI/leverage estimates — both are valid and should coexist.
        already_exists = any(
            abs(fz['price'] - dz['price']) / fz['price'] < 0.003 and fz['type'] == dz['type']
            and dz.get('fresh')
            for dz in deduped
        )
        if not already_exists:
            deduped.append(fz)

    # Check if zones have been swept by recent price action.
    # For zones with known formation time, check if swept AFTER formation.
    # For zones without formation time (legacy), use 4h lookback.
    sweep_lookback = 16  # 4h fallback
    recent_high = float(np.max(df_15m['High'].values[max(0, idx - sweep_lookback + 1):idx+1].astype(float)))
    recent_low = float(np.min(df_15m['Low'].values[max(0, idx - sweep_lookback + 1):idx+1].astype(float)))
    recent_times = df_15m['Open time'].values[max(0, idx - sweep_lookback + 1):idx+1]
    all_highs = df_15m['High'].values[:idx+1].astype(float)
    all_lows = df_15m['Low'].values[:idx+1].astype(float)
    all_times = df_15m['Open time'].values[:idx+1]

    for zone in deduped:
        zp = zone['price']
        swept = False
        swept_at = None

        # Live order book walls are by definition not swept — skip
        if zone['type'] in ('BID_WALL', 'ASK_WALL'):
            zone['swept'] = False
            zone['swept_at'] = None
            zone['age_bars'] = None
            continue

        formed_at = zone.get('formed_at')

        # Near-miss tolerance: price within X% of level counts as tested.
        # Prevents stale levels from being marked "unswept" when price
        # probed close but didn't breach the exact level.
        proximity_pct = CONFIG.get('SWEEP_PROXIMITY_PCT', 0.001)  # default 0.1%
        proximity_buf = zp * proximity_pct

        if formed_at is not None and formed_at < idx:
            # Check if swept AFTER formation (the correct way)
            check_start = formed_at + 1
            if check_start <= idx:
                for i in range(check_start, idx + 1):
                    if zp > current_price and all_highs[i] >= zp - proximity_buf:
                        swept = True
                        swept_at = str(all_times[i])
                        break
                    elif zp < current_price and all_lows[i] <= zp + proximity_buf:
                        swept = True
                        swept_at = str(all_times[i])
                        break
            zone['age_bars'] = idx - formed_at
        else:
            # Legacy fallback: fixed 4h lookback
            if zp > current_price and recent_high >= zp - proximity_buf:
                swept = True
                highs_arr = df_15m['High'].values[max(0, idx - sweep_lookback + 1):idx+1].astype(float)
                for i in range(len(recent_times)):
                    if highs_arr[i] >= zp - proximity_buf:
                        swept_at = str(recent_times[i])
                        break
            elif zp < current_price and recent_low <= zp + proximity_buf:
                swept = True
                lows_arr = df_15m['Low'].values[max(0, idx - sweep_lookback + 1):idx+1].astype(float)
                for i in range(len(recent_times)):
                    if lows_arr[i] <= zp + proximity_buf:
                        swept_at = str(recent_times[i])
                        break
            zone['age_bars'] = None

        zone['swept'] = swept
        zone['swept_at'] = swept_at

    # ── Backtest-derived filters ──
    # Drop noise zones based on 10-day backtest (3306 zones):
    #   - LOW cascade: 6% hit rate → noise
    #   - Strength < 20: 5.5% hit rate → noise
    #   - Distance > 3%: 6.4% hit rate → noise
    #
    # EXCEPTION: fresh zones (formed in last 48h) get relaxed filtering
    # because they haven't had time to prove themselves yet.
    fresh_cutoff = max(0, idx - 192)  # 48h = 192 bars of 15m
    filtered = []
    for zone in deduped:
        # Keep order book walls regardless (they're real liquidity)
        if zone['type'] in ('BID_WALL', 'ASK_WALL'):
            filtered.append(zone)
            continue

        is_fresh = zone.get('formed_at') is not None and zone['formed_at'] > fresh_cutoff

        # Skip swept zones (they're done)
        if zone.get('swept'):
            continue

        # Filter: max distance (always apply)
        dist = abs(zone['price'] - current_price) / current_price
        if dist > 0.03:
            continue

        # Relaxed filtering for fresh zones
        if is_fresh:
            # Fresh zones: only require strength > 10 (vs 20 for old zones)
            # and accept LOW cascade (they need time to build)
            if zone['strength'] < 10:
                continue
        else:
            # Legacy strict filtering
            if zone.get('cascade_risk') == 'LOW':
                continue
            if zone['strength'] < 20:
                continue

        filtered.append(zone)

    # Sort: fresh unswept zones first, then by strength
    def sort_key(z):
        fresh = z.get('formed_at') is not None and z['formed_at'] > fresh_cutoff
        return (0 if fresh else 1, -z['strength'])
    filtered.sort(key=sort_key)

    return filtered


def get_liquidity_summary(df_15m, idx, sr_levels, oi_usd, ls_ratio,
                           direction_bias=None, n_levels=10):
    """Get formatted liquidity summary for scanner output."""
    bids, asks = fetch_order_book_depth()
    order_book = (bids, asks) if bids and asks else None

    zones = estimate_liquidity_levels(
        df_15m, idx, sr_levels, oi_usd, ls_ratio, direction_bias, order_book)

    # Separate by side of price, keeping only directionally relevant types:
    #   below price → long liquidations/stops + bid walls (where longs get hurt)
    #   above price → short liquidations/stops + ask walls (where shorts get hurt)
    current_price = df_15m['Close'].values[idx]
    below_types = {'LONG_LIQ', 'LONG_STOP', 'BID_WALL'}
    above_types = {'SHORT_LIQ', 'SHORT_STOP', 'ASK_WALL'}
    below = [z for z in zones if z['price'] < current_price and z['type'] in below_types]
    above = [z for z in zones if z['price'] > current_price and z['type'] in above_types]

    below.sort(key=lambda x: x['price'], reverse=True)  # closest first
    above.sort(key=lambda x: x['price'])

    # Final dedup: merge same-type zones within 0.3% (cross-source).
    # Keeps the stronger zone when two overlap (e.g. OI zone + fresh zone).
    def _dedup_close(zones):
        if not zones:
            return zones
        out = []
        for z in zones:
            merged = False
            for existing in out:
                if z['type'] == existing['type'] and \
                   abs(z['price'] - existing['price']) / max(z['price'], existing['price']) < 0.003:
                    # Same type, same bucket — keep stronger
                    if z['strength'] > existing['strength']:
                        out[out.index(existing)] = z
                    merged = True
                    break
            if not merged:
                out.append(z)
        return out

    below = _dedup_close(below)
    above = _dedup_close(above)

    return {
        'current_price': round(float(current_price), 2),
        'below': below[:n_levels],
        'above': above[:n_levels],
        'all': zones[:n_levels * 2],
        'bid_walls': len([z for z in zones if z['type'] == 'BID_WALL']),
        'ask_walls': len([z for z in zones if z['type'] == 'ASK_WALL']),
        'high_cascade_zones': len([z for z in zones if z['cascade_risk'] == 'HIGH']),
    }

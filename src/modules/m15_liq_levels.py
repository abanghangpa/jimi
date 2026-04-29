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

    # Deduplicate nearby zones (within 0.3%)
    liq_zones.sort(key=lambda x: x['price'])
    deduped = []
    for zone in liq_zones:
        if not deduped or abs(zone['price'] - deduped[-1]['price']) / zone['price'] > 0.003:
            deduped.append(zone)
        else:
            # Merge: keep the stronger one
            if zone['strength'] > deduped[-1]['strength']:
                deduped[-1] = zone

    # Sort by strength
    deduped.sort(key=lambda x: x['strength'], reverse=True)

    return deduped


def get_liquidity_summary(df_15m, idx, sr_levels, oi_usd, ls_ratio,
                           direction_bias=None, n_levels=10):
    """Get formatted liquidity summary for scanner output."""
    bids, asks = fetch_order_book_depth()
    order_book = (bids, asks) if bids and asks else None

    zones = estimate_liquidity_levels(
        df_15m, idx, sr_levels, oi_usd, ls_ratio, direction_bias, order_book)

    # Separate by side of price
    current_price = df_15m['Close'].values[idx]
    below = [z for z in zones if z['price'] < current_price]
    above = [z for z in zones if z['price'] > current_price]

    below.sort(key=lambda x: x['price'], reverse=True)  # closest first
    above.sort(key=lambda x: x['price'])

    return {
        'current_price': round(float(current_price), 2),
        'below': below[:n_levels],
        'above': above[:n_levels],
        'all': zones[:n_levels * 2],
        'bid_walls': len([z for z in zones if z['type'] == 'BID_WALL']),
        'ask_walls': len([z for z in zones if z['type'] == 'ASK_WALL']),
        'high_cascade_zones': len([z for z in zones if z['cascade_risk'] == 'HIGH']),
    }

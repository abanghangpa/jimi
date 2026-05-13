#!/usr/bin/env python3
"""
TP Strategy Validation — Tests four alternatives against unswept pool baseline.

For each strategy, measures:
  - Reach rate: does price hit the TP target within N bars?
  - Reaction rate: does price reverse (>= 0.15%) after touching TP?
  - Win rate: does TP hit before SL?
  - Avg R:R realized

Usage:
    python3 scripts/validate_tp_strategies.py
    python3 scripts/validate_tp_strategies.py --bars 5000   # use last N bars only
"""

import argparse
import sys
import os
import numpy as np
import pandas as pd
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from src.utils.data_handler import load_data, resample_ohlcv
from src.utils.indicators import calc_atr, calc_ema, calc_rsi, calc_vwap
from src.modules.m5_liquidation import build_volume_profile, find_magnets, find_support_resistance


# ═══════════════════════════════════════════════════════════════
# HELPERS
# ═══════════════════════════════════════════════════════════════

def compute_sr_from_swings(df_15m, lookback=200, touch_threshold_pct=0.3, min_touches=3):
    """Compute S/R levels from swing highs/lows with bounce validation.

    Only returns levels that have been touched AND bounced at least min_touches times.
    """
    highs = df_15m['High'].values.astype(float)
    lows = df_15m['Low'].values.astype(float)
    closes = df_15m['Close'].values.astype(float)
    n = len(df_15m)

    # Find swing highs and lows
    swing_highs = []
    swing_lows = []
    window = 5
    for i in range(window, n - window):
        if highs[i] == max(highs[i-window:i+window+1]):
            swing_highs.append((i, highs[i]))
        if lows[i] == min(lows[i-window:i+window+1]):
            swing_lows.append((i, lows[i]))

    # Cluster nearby levels
    all_levels = [(p, 'RESISTANCE') for _, p in swing_highs] + [(p, 'SUPPORT') for _, p in swing_lows]
    all_levels.sort(key=lambda x: x[0])

    clustered = []
    i = 0
    while i < len(all_levels):
        cluster = [all_levels[i]]
        j = i + 1
        while j < len(all_levels) and (all_levels[j][0] - cluster[-1][0]) / cluster[-1][0] < touch_threshold_pct / 100:
            cluster.append(all_levels[j])
            j += 1
        avg_price = np.mean([c[0] for c in cluster])
        types = [c[1] for c in cluster]
        dominant_type = 'RESISTANCE' if types.count('RESISTANCE') > types.count('SUPPORT') else 'SUPPORT'
        clustered.append((avg_price, len(cluster), dominant_type))
        i = j

    # Validate bounces: for each level, count how many times price touched and reversed
    validated = []
    for price, touch_count, ltype in clustered:
        # Count actual bounces (price touched zone and reversed >= 0.15% within 8 bars)
        bounces = 0
        total_touches = 0
        threshold = price * 0.003  # 0.3% zone

        for i in range(lookback, n):
            bar_low = lows[i]
            bar_high = highs[i]

            touched = False
            if ltype == 'SUPPORT' and bar_low <= price + threshold and bar_low >= price - threshold:
                touched = True
            elif ltype == 'RESISTANCE' and bar_high >= price - threshold and bar_high <= price + threshold:
                touched = True

            if touched:
                total_touches += 1
                # Check for bounce in next 8 bars
                if i + 8 < n:
                    future_closes = closes[i+1:i+9]
                    if ltype == 'SUPPORT':
                        max_bounce = (max(future_closes) - price) / price * 100
                        if max_bounce >= 0.15:
                            bounces += 1
                    else:
                        max_bounce = (price - min(future_closes)) / price * 100
                        if max_bounce >= 0.15:
                            bounces += 1

        if total_touches >= min_touches and bounces >= 2:
            bounce_rate = bounces / total_touches if total_touches > 0 else 0
            validated.append({
                'price': price,
                'type': ltype,
                'touches': total_touches,
                'bounces': bounces,
                'bounce_rate': bounce_rate,
                'strength': bounces * bounce_rate,
            })

    return validated


def compute_cascade_potential(price, direction, df_15m, idx, lookback=48):
    """Estimate cascade potential at a price level.

    Factors: OI proxy (volume accumulation), momentum alignment,
    stop density (how many bars have highs/lows near this level).
    """
    if idx < lookback:
        return 0.0

    highs = df_15m['High'].values[idx-lookback:idx+1].astype(float)
    lows = df_15m['Low'].values[idx-lookback:idx+1].astype(float)
    closes = df_15m['Close'].values[idx-lookback:idx+1].astype(float)
    volumes = df_15m['Volume'].values[idx-lookback:idx+1].astype(float)
    takers = df_15m['Taker buy base asset volume'].values[idx-lookback:idx+1].astype(float)

    threshold = price * 0.003  # 0.3% zone

    # 1. Stop density: how many bars have highs/lows near this level
    if direction == 'LONG':
        # For long TP: stops are above → count bars with highs near level
        stops_near = sum(1 for h in highs if abs(h - price) < threshold)
    else:
        # For short TP: stops are below → count bars with lows near level
        stops_near = sum(1 for l in lows if abs(l - price) < threshold)
    stop_density = stops_near / len(highs)

    # 2. Momentum alignment (are recent closes moving toward or away from level?)
    recent_momentum = (closes[-1] - closes[-16]) / closes[-16] * 100 if len(closes) > 16 else 0
    if direction == 'LONG':
        momentum_aligned = recent_momentum > 0
    else:
        momentum_aligned = recent_momentum < 0

    # 3. Volume trend (increasing volume = more fuel for cascade)
    vol_recent = np.mean(volumes[-8:])
    vol_prior = np.mean(volumes[-24:-8]) if len(volumes) > 24 else vol_recent
    vol_trend = vol_recent / vol_prior if vol_prior > 0 else 1.0

    # 4. Taker imbalance (extreme taker ratio = cascade fuel)
    taker_ratio = np.sum(takers) / np.sum(volumes) if np.sum(volumes) > 0 else 0.5
    taker_extreme = abs(taker_ratio - 0.5) * 2  # 0 = balanced, 1 = extreme

    # Composite cascade score
    score = (
        stop_density * 0.30 +
        (1.0 if momentum_aligned else 0.0) * 0.25 +
        min(vol_trend / 2.0, 1.0) * 0.25 +
        taker_extreme * 0.20
    )
    return min(1.0, score)


def check_price_reaction(df_15m, touch_bar, target_price, direction, lookforward=16, min_reaction_pct=0.15):
    """Check if price reacts (reverses >= min_reaction_pct) after touching a level.

    Returns: (reacted: bool, max_favorable: float, max_adverse: float)
    """
    n = len(df_15m)
    end_bar = min(touch_bar + lookforward, n)

    if direction == 'LONG':
        # TP above entry: reaction = price drops after touching
        closes_after = df_15m['Close'].values[touch_bar:end_bar].astype(float)
        if len(closes_after) == 0:
            return False, 0.0, 0.0
        max_favorable = (target_price - min(closes_after)) / target_price * 100
        max_adverse = (max(closes_after) - target_price) / target_price * 100
    else:
        # TP below entry: price rises after touching
        closes_after = df_15m['Close'].values[touch_bar:end_bar].astype(float)
        if len(closes_after) == 0:
            return False, 0.0, 0.0
        max_favorable = (max(closes_after) - target_price) / target_price * 100
        max_adverse = (target_price - min(closes_after)) / target_price * 100

    reacted = max_favorable >= min_reaction_pct
    return reacted, max_favorable, max_adverse


# ═══════════════════════════════════════════════════════════════
# TP STRATEGY IMPLEMENTATIONS
# ═══════════════════════════════════════════════════════════════

def tp_unswept_pool(entry_price, direction, magnets, liq_levels=None):
    """Baseline: nearest unswept HVN or liquidation level."""
    candidates = []
    if magnets:
        for m in magnets:
            p = m[0]
            if direction == 'LONG' and p > entry_price:
                candidates.append(p)
            elif direction == 'SHORT' and p < entry_price:
                candidates.append(p)
    if candidates:
        return min(candidates, key=lambda p: abs(p - entry_price))
    return None


def tp_sr_bounce(entry_price, direction, sr_levels):
    """Alternative 1: nearest validated S/R level with bounce history."""
    candidates = []
    for sr in sr_levels:
        if direction == 'LONG' and sr['type'] == 'RESISTANCE' and sr['price'] > entry_price:
            candidates.append(sr)
        elif direction == 'SHORT' and sr['type'] == 'SUPPORT' and sr['price'] < entry_price:
            candidates.append(sr)
    if candidates:
        # Prefer stronger levels (more bounces) with distance penalty
        for c in candidates:
            dist = abs(c['price'] - entry_price) / entry_price * 100
            c['score'] = c['strength'] / (dist + 0.1)
        candidates.sort(key=lambda x: -x['score'])
        return candidates[0]['price']
    return None


def tp_cascade_cluster(entry_price, direction, df_15m, idx, sr_levels, magnets):
    """Alternative 2: TP at liquidation cluster with highest cascade potential."""
    candidates = []

    # Collect all levels above/below entry
    if magnets:
        for m in magnets:
            p = m[0]
            if direction == 'LONG' and p > entry_price:
                cascade = compute_cascade_potential(p, direction, df_15m, idx)
                candidates.append({'price': p, 'cascade': cascade, 'source': 'HVN'})
            elif direction == 'SHORT' and p < entry_price:
                cascade = compute_cascade_potential(p, direction, df_15m, idx)
                candidates.append({'price': p, 'cascade': cascade, 'source': 'HVN'})

    if sr_levels:
        for sr in sr_levels:
            if direction == 'LONG' and sr['type'] == 'RESISTANCE' and sr['price'] > entry_price:
                cascade = compute_cascade_potential(sr['price'], direction, df_15m, idx)
                candidates.append({'price': sr['price'], 'cascade': cascade, 'source': 'SR'})
            elif direction == 'SHORT' and sr['type'] == 'SUPPORT' and sr['price'] < entry_price:
                cascade = compute_cascade_potential(sr['price'], direction, df_15m, idx)
                candidates.append({'price': sr['price'], 'cascade': cascade, 'source': 'SR'})

    if candidates:
        # Score = cascade potential * proximity bonus
        for c in candidates:
            dist = abs(c['price'] - entry_price) / entry_price * 100
            c['score'] = c['cascade'] * (1.0 / (dist + 0.1))
        candidates.sort(key=lambda x: -x['score'])
        return candidates[0]['price']
    return None


def tp_atr(entry_price, direction, atr_1h, multiplier=1.2):
    """Alternative 3: simple ATR-based TP."""
    if direction == 'LONG':
        return entry_price + atr_1h * multiplier
    else:
        return entry_price - atr_1h * multiplier


def tp_taker_flip(entry_price, direction, df_15m, idx, lookforward=32, taker_threshold=0.55):
    """Alternative 4: TP when taker ratio flips (opposing order flow).

    Looks forward from entry to find the first bar where taker ratio
    flips against the trade direction.
    """
    n = len(df_15m)
    end_bar = min(idx + lookforward, n)

    if idx >= n or end_bar <= idx:
        return None

    takers = df_15m['Taker buy base asset volume'].values[idx+1:end_bar].astype(float)
    volumes = df_15m['Volume'].values[idx+1:end_bar].astype(float)
    closes = df_15m['Close'].values[idx+1:end_bar].astype(float)

    if len(takers) == 0:
        return None

    for i in range(len(takers)):
        if volumes[i] > 0:
            ratio = takers[i] / volumes[i]
        else:
            continue

        if direction == 'LONG' and ratio < (1 - taker_threshold):
            # Sellers taking over → good time to exit long
            return float(closes[i])
        elif direction == 'SHORT' and ratio > taker_threshold:
            # Buyers taking over → good time to exit short
            return float(closes[i])

    return None


# ═══════════════════════════════════════════════════════════════
# SIMULATION
# ═══════════════════════════════════════════════════════════════

def simulate_trade_outcome(df_15m, entry_idx, entry_price, tp_price, sl_price, direction, max_bars=80):
    """Simulate a trade from entry to exit, tracking what happens.

    Returns: dict with outcome details.
    """
    n = len(df_15m)
    end_bar = min(entry_idx + max_bars, n)

    if tp_price is None or sl_price is None:
        return {'outcome': 'NO_LEVELS', 'bars_held': 0, 'pnl_pct': 0}

    for i in range(entry_idx + 1, end_bar):
        high = float(df_15m['High'].values[i])
        low = float(df_15m['Low'].values[i])

        if direction == 'LONG':
            if low <= sl_price:
                pnl = (sl_price - entry_price) / entry_price * 100
                return {'outcome': 'SL', 'bars_held': i - entry_idx, 'pnl_pct': pnl, 'exit_bar': i}
            if high >= tp_price:
                pnl = (tp_price - entry_price) / entry_price * 100
                return {'outcome': 'TP', 'bars_held': i - entry_idx, 'pnl_pct': pnl, 'exit_bar': i}
        else:
            if high >= sl_price:
                pnl = (entry_price - sl_price) / entry_price * 100
                return {'outcome': 'SL', 'bars_held': i - entry_idx, 'pnl_pct': pnl, 'exit_bar': i}
            if low <= tp_price:
                pnl = (entry_price - tp_price) / entry_price * 100
                return {'outcome': 'TP', 'bars_held': i - entry_idx, 'pnl_pct': pnl, 'exit_bar': i}

    # Neither hit within max_bars
    close_price = float(df_15m['Close'].values[min(entry_idx + max_bars, n - 1)])
    if direction == 'LONG':
        pnl = (close_price - entry_price) / entry_price * 100
    else:
        pnl = (entry_price - close_price) / entry_price * 100
    return {'outcome': 'TIMEOUT', 'bars_held': max_bars, 'pnl_pct': pnl}


# ═══════════════════════════════════════════════════════════════
# MAIN VALIDATION
# ═══════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description='TP Strategy Validation')
    parser.add_argument('--bars', type=int, default=0, help='Use last N bars only (0 = all)')
    parser.add_argument('--step', type=int, default=12, help='Sample every N bars (default: 12 = every 3h)')
    parser.add_argument('--max-bars', type=int, default=80, help='Max bars to hold trade (default: 80 = 20h)')
    parser.add_argument('--atr-mult', type=float, default=1.2, help='ATR multiplier for TP (default: 1.2)')
    parser.add_argument('--csv', default=None, help='Path to CSV (default: auto-detect)')
    args = parser.parse_args()

    # Load data
    csv_path = args.csv or os.path.join(os.path.dirname(os.path.dirname(__file__)), 'eth_15m_merged.csv')
    if not os.path.exists(csv_path):
        csv_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data', 'eth_15m_merged.csv')
    if not os.path.exists(csv_path):
        print("ERROR: CSV not found")
        return

    print(f"Loading {csv_path}...")
    df = load_data(csv_path)
    if args.bars > 0:
        df = df.iloc[-args.bars:]
    df = df.reset_index(drop=True)
    n = len(df)
    print(f"  {n} bars loaded ({df['Open time'].iloc[0]} → {df['Open time'].iloc[-1]})")

    # Pre-compute indicators
    print("Computing indicators...")
    cfg_atr_period = 14
    df['atr'] = calc_atr(df['High'], df['Low'], df['Close'], cfg_atr_period)
    df['taker_ratio'] = df['Taker buy base asset volume'] / df['Volume'].replace(0, np.nan)
    df['vol_ma20'] = df['Volume'].rolling(20).mean()

    # 1H ATR (resample)
    df_1h = resample_ohlcv(df, '1H')
    df_1h['atr'] = calc_atr(df_1h['High'], df_1h['Low'], df_1h['Close'], cfg_atr_period)

    # Strategy results containers
    strategies = {
        'unswept_pool': {'trades': [], 'reactions': []},
        'sr_bounce': {'trades': [], 'reactions': []},
        'cascade_cluster': {'trades': [], 'reactions': []},
        'atr_tp': {'trades': [], 'reactions': []},
        'taker_flip': {'trades': [], 'reactions': []},
    }

    # Rolling S/R cache (recompute every 500 bars)
    sr_cache = {}
    sr_cache_interval = 500

    def get_sr_for_bar(bar_idx):
        cache_key = (bar_idx // sr_cache_interval) * sr_cache_interval
        if cache_key not in sr_cache:
            start = max(0, cache_key - 2000)
            sr_df = df.iloc[start:cache_key + sr_cache_interval]
            sr_cache[cache_key] = compute_sr_from_swings(sr_df, lookback=200, min_touches=3)
        return sr_cache[cache_key]

    # Main loop: sample entry points
    warmup = max(500, cfg_atr_period * 10)
    print(f"Simulating trades (step={args.step}, max_bars={args.maxBars if hasattr(args, 'maxBars') else args.max_bars})...")
    print(f"  Warmup: {warmup} bars, sampling every {args.step} bars")

    total_samples = 0
    for entry_idx in range(warmup, n - args.max_bars, args.step):
        total_samples += 1
        entry_price = float(df['Close'].values[entry_idx])
        atr_1h_val = float(df['atr'].values[entry_idx])
        if np.isnan(atr_1h_val) or atr_1h_val <= 0:
            atr_1h_val = entry_price * 0.01

        # Direction: alternate LONG/SHORT
        direction = 'LONG' if (total_samples % 2 == 0) else 'SHORT'

        # SL: 1 ATR below/above entry
        if direction == 'LONG':
            sl_price = entry_price - atr_1h_val
        else:
            sl_price = entry_price + atr_1h_val

        # Get S/R levels for this bar
        sr_levels = get_sr_for_bar(entry_idx)

        # Get volume profile magnets (rolling)
        vp_lookback = 500
        vp_start = max(0, entry_idx - vp_lookback)
        highs_vp = df['High'].values[vp_start:entry_idx+1].astype(float)
        lows_vp = df['Low'].values[vp_start:entry_idx+1].astype(float)
        closes_vp = df['Close'].values[vp_start:entry_idx+1].astype(float)
        vols_vp = df['Volume'].values[vp_start:entry_idx+1].astype(float)
        bin_centers, vol_profile, bin_edges = build_volume_profile(
            highs_vp, lows_vp, closes_vp, vols_vp, n_bins=50, lookback=vp_lookback)
        magnets = find_magnets(bin_centers, vol_profile) if bin_centers is not None else []

        # Strategy 1: Unswept pool (baseline)
        tp1 = tp_unswept_pool(entry_price, direction, magnets)
        if tp1 is not None:
            result = simulate_trade_outcome(df, entry_idx, entry_price, tp1, sl_price, direction, args.max_bars)
            strategies['unswept_pool']['trades'].append(result)
            # Check reaction at TP
            tp_bar = result.get('exit_bar')
            if tp_bar and result['outcome'] == 'TP':
                reacted, fav, adv = check_price_reaction(df, tp_bar, tp1, direction)
                strategies['unswept_pool']['reactions'].append(reacted)

        # Strategy 2: S/R with bounces
        tp2 = tp_sr_bounce(entry_price, direction, sr_levels)
        if tp2 is not None:
            result = simulate_trade_outcome(df, entry_idx, entry_price, tp2, sl_price, direction, args.max_bars)
            strategies['sr_bounce']['trades'].append(result)
            tp_bar = result.get('exit_bar')
            if tp_bar and result['outcome'] == 'TP':
                reacted, fav, adv = check_price_reaction(df, tp_bar, tp2, direction)
                strategies['sr_bounce']['reactions'].append(reacted)

        # Strategy 3: Cascade cluster
        tp3 = tp_cascade_cluster(entry_price, direction, df, entry_idx, sr_levels, magnets)
        if tp3 is not None:
            result = simulate_trade_outcome(df, entry_idx, entry_price, tp3, sl_price, direction, args.max_bars)
            strategies['cascade_cluster']['trades'].append(result)
            tp_bar = result.get('exit_bar')
            if tp_bar and result['outcome'] == 'TP':
                reacted, fav, adv = check_price_reaction(df, tp_bar, tp3, direction)
                strategies['cascade_cluster']['reactions'].append(reacted)

        # Strategy 4: ATR-based TP
        tp4 = tp_atr(entry_price, direction, atr_1h_val, args.atr_mult)
        result = simulate_trade_outcome(df, entry_idx, entry_price, tp4, sl_price, direction, args.max_bars)
        strategies['atr_tp']['trades'].append(result)

        # Strategy 5: Taker flip
        tp5 = tp_taker_flip(entry_price, direction, df, entry_idx)
        if tp5 is not None:
            result = simulate_trade_outcome(df, entry_idx, entry_price, tp5, sl_price, direction, args.max_bars)
            strategies['taker_flip']['trades'].append(result)

    # ═══════════════════════════════════════════════════════════════
    # RESULTS
    # ═══════════════════════════════════════════════════════════════

    print("\n" + "═" * 70)
    print("  TP STRATEGY VALIDATION RESULTS")
    print("═" * 70)
    print(f"\n  Dataset: {n} bars, {total_samples} sampled entry points")
    print(f"  ATR multiplier: {args.atr_mult}x  |  Max hold: {args.max_bars} bars")

    for name, data in strategies.items():
        trades = data['trades']
        reactions = data['reactions']

        if not trades:
            print(f"\n  {name}: NO DATA")
            continue

        total = len(trades)
        tp_count = sum(1 for t in trades if t['outcome'] == 'TP')
        sl_count = sum(1 for t in trades if t['outcome'] == 'SL')
        timeout_count = sum(1 for t in trades if t['outcome'] == 'TIMEOUT')
        win_rate = tp_count / total * 100 if total > 0 else 0
        avg_pnl = np.mean([t['pnl_pct'] for t in trades])
        avg_bars = np.mean([t['bars_held'] for t in trades])
        avg_tp_bars = np.mean([t['bars_held'] for t in trades if t['outcome'] == 'TP']) if tp_count > 0 else 0
        avg_tp_pnl = np.mean([t['pnl_pct'] for t in trades if t['outcome'] == 'TP']) if tp_count > 0 else 0
        avg_sl_pnl = np.mean([t['pnl_pct'] for t in trades if t['outcome'] == 'SL']) if sl_count > 0 else 0

        # Reaction rate
        reaction_rate = sum(reactions) / len(reactions) * 100 if reactions else 0

        # Expectancy = WR * avg_win - (1-WR) * avg_loss
        if tp_count > 0 and sl_count > 0:
            expectancy = (win_rate/100 * avg_tp_pnl) + ((1 - win_rate/100) * avg_sl_pnl)
        else:
            expectancy = 0

        # Profit factor
        gross_profit = sum(t['pnl_pct'] for t in trades if t['pnl_pct'] > 0)
        gross_loss = abs(sum(t['pnl_pct'] for t in trades if t['pnl_pct'] < 0))
        profit_factor = gross_profit / gross_loss if gross_loss > 0 else float('inf')

        print(f"\n  {'─' * 60}")
        label = {
            'unswept_pool': 'Unswept HVN (BASELINE)',
            'sr_bounce': 'S/R with Bounces',
            'cascade_cluster': 'Cascade Cluster',
            'atr_tp': f'ATR {args.atr_mult}x',
            'taker_flip': 'Taker Flow Flip',
        }.get(name, name)
        print(f"  {label}")
        print(f"  {'─' * 60}")
        print(f"    Trades:      {total}")
        print(f"    TP hit:      {tp_count} ({win_rate:.1f}%)")
        print(f"    SL hit:      {sl_count} ({sl_count/total*100:.1f}%)")
        print(f"    Timeout:     {timeout_count} ({timeout_count/total*100:.1f}%)")
        print(f"    Reaction:    {reaction_rate:.1f}% (of TP hits)")
        print(f"    Avg PnL:     {avg_pnl:+.3f}%")
        print(f"    Avg TP PnL:  {avg_tp_pnl:+.3f}%")
        print(f"    Avg SL PnL:  {avg_sl_pnl:+.3f}%")
        print(f"    Avg bars:    {avg_bars:.1f} (TP: {avg_tp_bars:.1f})")
        print(f"    Expectancy:  {expectancy:+.4f}%")
        print(f"    Profit Fac:  {profit_factor:.2f}")

    # ── Comparison table ──
    print(f"\n  {'═' * 60}")
    print(f"  COMPARISON TABLE")
    print(f"  {'═' * 60}")
    print(f"  {'Strategy':<25} {'Win%':>6} {'React%':>7} {'Exp%':>8} {'PF':>6} {'AvgPnL':>8}")
    print(f"  {'─' * 60}")

    for name in ['unswept_pool', 'sr_bounce', 'cascade_cluster', 'atr_tp', 'taker_flip']:
        data = strategies[name]
        trades = data['trades']
        reactions = data['reactions']
        if not trades:
            continue
        total = len(trades)
        tp_count = sum(1 for t in trades if t['outcome'] == 'TP')
        sl_count = sum(1 for t in trades if t['outcome'] == 'SL')
        win_rate = tp_count / total * 100
        reaction_rate = sum(reactions) / len(reactions) * 100 if reactions else 0
        avg_pnl = np.mean([t['pnl_pct'] for t in trades])
        avg_tp_pnl = np.mean([t['pnl_pct'] for t in trades if t['outcome'] == 'TP']) if tp_count > 0 else 0
        avg_sl_pnl = np.mean([t['pnl_pct'] for t in trades if t['outcome'] == 'SL']) if sl_count > 0 else 0
        expectancy = (win_rate/100 * avg_tp_pnl) + ((1 - win_rate/100) * avg_sl_pnl) if tp_count > 0 and sl_count > 0 else 0
        gross_profit = sum(t['pnl_pct'] for t in trades if t['pnl_pct'] > 0)
        gross_loss = abs(sum(t['pnl_pct'] for t in trades if t['pnl_pct'] < 0))
        pf = gross_profit / gross_loss if gross_loss > 0 else float('inf')

        label = {
            'unswept_pool': 'Unswept HVN (base)',
            'sr_bounce': 'S/R Bounce',
            'cascade_cluster': 'Cascade',
            'atr_tp': f'ATR {args.atr_mult}x',
            'taker_flip': 'Taker Flip',
        }.get(name, name)
        print(f"  {label:<25} {win_rate:>5.1f}% {reaction_rate:>6.1f}% {expectancy:>+7.4f}% {pf:>6.2f} {avg_pnl:>+7.3f}%")

    print(f"\n  {'═' * 60}")


if __name__ == '__main__':
    main()

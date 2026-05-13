"""Taker Ratio Tracker — Directional signal from aggressive order flow.

Taker ratio = taker_buy_volume / total_volume
  > 0.52 = buyers dominating (aggressive buying)
  < 0.48 = sellers dominating (aggressive selling)
  ~ 0.50 = neutral

This module tracks:
  1. Rolling taker momentum (is buying/selling accelerating?)
  2. Taker divergence (price vs taker disagreement)
  3. Historical percentile (how extreme is current taker?)
  4. Taker regime (accumulation / distribution / neutral)
"""

import numpy as np
import pandas as pd


def compute_taker_series(df_15m):
    """Compute rolling taker ratio series with momentum and regime."""
    taker_base = df_15m['Taker buy base asset volume'].values.astype(float)
    total_vol = df_15m['Volume'].values.astype(float)
    
    # Raw taker ratio per bar
    with np.errstate(divide='ignore', invalid='ignore'):
        taker_raw = np.where(total_vol > 0, taker_base / total_vol, 0.5)
    
    # Rolling averages
    taker_4h = pd.Series(taker_raw).rolling(16).mean().values   # 4h = 16 bars
    taker_12h = pd.Series(taker_raw).rolling(48).mean().values  # 12h = 48 bars
    taker_24h = pd.Series(taker_raw).rolling(96).mean().values  # 24h = 96 bars
    
    # Momentum: 4h avg vs 12h avg
    taker_momentum = taker_4h - taker_12h
    
    # Acceleration: momentum change over last 8 bars
    taker_accel = pd.Series(taker_momentum).diff(8).values
    
    return {
        'raw': taker_raw,
        'avg_4h': taker_4h,
        'avg_12h': taker_12h,
        'avg_24h': taker_24h,
        'momentum': taker_momentum,
        'acceleration': taker_accel,
    }


def detect_taker_divergence(df_15m, lookback=16):
    """Detect divergence between price direction and taker direction.

    Bullish divergence: price making lower lows but taker ratio rising
    Bearish divergence: price making higher highs but taker ratio falling

    Returns: list of (idx, type, strength) for each detected divergence.
    """
    closes = df_15m['Close'].values.astype(float)
    taker_base = df_15m['Taker buy base asset volume'].values.astype(float)
    total_vol = df_15m['Volume'].values.astype(float)
    
    with np.errstate(divide='ignore', invalid='ignore'):
        taker_raw = np.where(total_vol > 0, taker_base / total_vol, 0.5)
    
    taker_4h = pd.Series(taker_raw).rolling(16).mean().values
    
    divergences = []
    
    for i in range(lookback * 2, len(df_15m)):
        # Price trend over lookback
        price_now = closes[i]
        price_prev = closes[i - lookback]
        price_dir = 'UP' if price_now > price_prev * 1.001 else 'DOWN' if price_now < price_prev * 0.999 else 'FLAT'
        
        # Taker trend over lookback
        taker_now = taker_4h[i] if not np.isnan(taker_4h[i]) else 0.5
        taker_prev = taker_4h[i - lookback] if not np.isnan(taker_4h[i - lookback]) else 0.5
        taker_dir = 'UP' if taker_now > taker_prev + 0.01 else 'DOWN' if taker_now < taker_prev - 0.01 else 'FLAT'
        
        # Divergence = price and taker disagree
        if price_dir == 'DOWN' and taker_dir == 'UP':
            strength = abs(taker_now - taker_prev) * 100
            divergences.append((i, 'BULLISH', round(strength, 2)))
        elif price_dir == 'UP' and taker_dir == 'DOWN':
            strength = abs(taker_now - taker_prev) * 100
            divergences.append((i, 'BEARISH', round(strength, 2)))
    
    return divergences


def get_taker_regime(taker_4h, taker_12h, taker_24h):
    """Classify the current taker regime.

    Returns: (regime, description)
      ACCUMULATION:  sustained aggressive buying (4h > 12h > 24h)
      DISTRIBUTION:  sustained aggressive selling (4h < 12h < 24h)
      BUYING_SPIKE:  short-term buying surge (4h >> 12h)
      SELLING_SPIKE: short-term selling surge (4h << 12h)
      NEUTRAL:       no clear direction
    """
    if np.isnan(taker_4h) or np.isnan(taker_12h) or np.isnan(taker_24h):
        return 'NEUTRAL', 'insufficient data'
    
    # Sustained buying
    if taker_4h > 0.52 and taker_12h > 0.51 and taker_24h > 0.50:
        if taker_4h > taker_12h > taker_24h:
            return 'ACCUMULATION', f'sustained buying (4h={taker_4h:.3f} > 12h={taker_12h:.3f} > 24h={taker_24h:.3f})'
        return 'BUYING_SPIKE', f'buying surge (4h={taker_4h:.3f}, 12h={taker_12h:.3f})'
    
    # Sustained selling
    if taker_4h < 0.48 and taker_12h < 0.49 and taker_24h < 0.50:
        if taker_4h < taker_12h < taker_24h:
            return 'DISTRIBUTION', f'sustained selling (4h={taker_4h:.3f} < 12h={taker_12h:.3f} < 24h={taker_24h:.3f})'
        return 'SELLING_SPIKE', f'selling surge (4h={taker_4h:.3f}, 12h={taker_12h:.3f})'
    
    # Mixed
    if taker_4h > 0.53:
        return 'BUYING_SPIKE', f'short-term buying (4h={taker_4h:.3f})'
    if taker_4h < 0.47:
        return 'SELLING_SPIKE', f'short-term selling (4h={taker_4h:.3f})'
    
    return 'NEUTRAL', f'balanced (4h={taker_4h:.3f})'


def score_taker_signal(taker_4h, taker_12h, taker_momentum, taker_accel):
    """Score the taker signal for direction.

    Returns: (direction, score, description)
      direction: 'LONG', 'SHORT', or 'NEUTRAL'
      score: 0.0-1.0 (confidence)
    """
    if np.isnan(taker_4h) or np.isnan(taker_12h):
        return 'NEUTRAL', 0.0, 'insufficient data'
    
    score = 0.0
    direction = 'NEUTRAL'
    reasons = []
    
    # Base: 4h average deviation from 0.50
    deviation = taker_4h - 0.50
    if abs(deviation) > 0.02:
        score += min(abs(deviation) * 5, 0.4)  # max 0.4 from deviation
        direction = 'LONG' if deviation > 0 else 'SHORT'
        reasons.append(f'4h taker {taker_4h:.3f} ({deviation:+.3f} from neutral)')
    
    # Momentum bonus
    if not np.isnan(taker_momentum):
        if (direction == 'LONG' and taker_momentum > 0.01) or \
           (direction == 'SHORT' and taker_momentum < -0.01):
            score += min(abs(taker_momentum) * 3, 0.2)
            reasons.append(f'momentum {taker_momentum:+.3f}')
    
    # Acceleration bonus
    if not np.isnan(taker_accel):
        if (direction == 'LONG' and taker_accel > 0.005) or \
           (direction == 'SHORT' and taker_accel < -0.005):
            score += min(abs(taker_accel) * 5, 0.15)
            reasons.append(f'accelerating')
    
    # 12h agreement
    if (direction == 'LONG' and taker_12h > 0.51) or \
       (direction == 'SHORT' and taker_12h < 0.49):
        score += 0.15
        reasons.append(f'12h confirms ({taker_12h:.3f})')
    
    # Penalty: mixed signals
    if direction == 'LONG' and taker_12h < 0.49:
        score *= 0.5
        reasons.append('⚠️ 12h disagrees')
    elif direction == 'SHORT' and taker_12h > 0.51:
        score *= 0.5
        reasons.append('⚠️ 12h disagrees')
    
    score = min(score, 1.0)
    
    if score < 0.15:
        direction = 'NEUTRAL'
    
    return direction, round(score, 3), '; '.join(reasons)


def get_taker_summary(df_15m, idx=None):
    """Get full taker analysis for the current bar.

    Returns dict with all taker metrics.
    """
    if idx is None:
        idx = len(df_15m) - 1
    
    series = compute_taker_series(df_15m)
    
    raw = series['raw'][idx]
    avg_4h = series['avg_4h'][idx]
    avg_12h = series['avg_12h'][idx]
    avg_24h = series['avg_24h'][idx]
    momentum = series['momentum'][idx]
    accel = series['acceleration'][idx]
    
    regime, regime_desc = get_taker_regime(avg_4h, avg_12h, avg_24h)
    direction, score, reason = score_taker_signal(avg_4h, avg_12h, momentum, accel)
    
    # Historical percentile (last 500 bars)
    window = series['avg_4h'][max(0, idx-500):idx+1]
    valid = window[~np.isnan(window)]
    if len(valid) > 10:
        percentile = (np.sum(valid < avg_4h) / len(valid)) * 100
    else:
        percentile = 50.0
    
    # Recent divergences (last 16 bars)
    divs = detect_taker_divergence(df_15m.iloc[:idx+1], lookback=16)
    recent_divs = [(d[1], d[2]) for d in divs[-3:]] if divs else []
    
    return {
        'raw': round(float(raw), 4),
        'avg_4h': round(float(avg_4h), 4) if not np.isnan(avg_4h) else None,
        'avg_12h': round(float(avg_12h), 4) if not np.isnan(avg_12h) else None,
        'avg_24h': round(float(avg_24h), 4) if not np.isnan(avg_24h) else None,
        'momentum': round(float(momentum), 4) if not np.isnan(momentum) else None,
        'acceleration': round(float(accel), 4) if not np.isnan(accel) else None,
        'regime': regime,
        'regime_desc': regime_desc,
        'direction': direction,
        'score': score,
        'reason': reason,
        'percentile': round(float(percentile), 1),
        'recent_divergences': recent_divs,
    }


def format_taker_summary(data):
    """Format taker summary for scanner output."""
    lines = []
    
    direction_icon = {'LONG': '🟢', 'SHORT': '🔴', 'NEUTRAL': '⚪'}.get(data['direction'], '⚪')
    regime_icon = {
        'ACCUMULATION': '🟢', 'DISTRIBUTION': '🔴',
        'BUYING_SPIKE': '🟡', 'SELLING_SPIKE': '🟡', 'NEUTRAL': '⚪'
    }.get(data['regime'], '⚪')
    
    lines.append(f"  Taker Flow Analysis:")
    lines.append(f"    Current:    {data['raw']:.4f}  ({'buyers' if data['raw'] > 0.52 else 'sellers' if data['raw'] < 0.48 else 'neutral'})")
    
    if data['avg_4h'] is not None:
        lines.append(f"    4h avg:     {data['avg_4h']:.4f}  (percentile: {data['percentile']:.0f}%)")
    if data['avg_12h'] is not None:
        lines.append(f"    12h avg:    {data['avg_12h']:.4f}")
    if data['avg_24h'] is not None:
        lines.append(f"    24h avg:    {data['avg_24h']:.4f}")
    if data['momentum'] is not None:
        lines.append(f"    Momentum:   {data['momentum']:+.4f}  ({'accelerating ↑' if data['momentum'] > 0.01 else 'decelerating ↓' if data['momentum'] < -0.01 else 'stable'})")
    
    lines.append(f"    Regime:     {regime_icon} {data['regime']}  — {data['regime_desc']}")
    lines.append(f"    Signal:     {direction_icon} {data['direction']}  (score={data['score']:.3f})")
    
    if data['reason']:
        lines.append(f"    Factors:    {data['reason']}")
    
    if data['recent_divergences']:
        for div_type, div_strength in data['recent_divergences']:
            icon = '🟢' if div_type == 'BULLISH' else '🔴'
            lines.append(f"    Divergence: {icon} {div_type} (strength={div_strength})")
    
    return '\n'.join(lines)

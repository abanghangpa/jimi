"""
M17: Resistance Quality Module

Validates whether support/resistance levels are strong or weak by analyzing:
  1. Zone Volume Density — thin (easy to break) vs thick (hard to break)
  2. Rejection Pattern — active defender (high vol rejections) vs passive
  3. Defender Behavior — OI/positioning at the zone (trapped vs exiting)
  4. Breakout Readiness — momentum + volume as price approaches

This answers: "When price reaches this S/R level, what happens?"

Signal taxonomy:
  THIN_ZONE       — low volume node, easy to pop through → high breakout score
  ACTIVE_DEFENDER  — high volume rejections, someone defending → low breakout score
  DEFENDER_TRAPPED — OI surging + extreme z-score at resistance → squeeze building
  DEFENDER_LEAVING — OI dropping at resistance → resistance fading
  BREAKOUT_READY   — momentum + volume approaching zone → imminent test
"""

import numpy as np
import pandas as pd


# ═══════════════════════════════════════════════════════════════
# DEFAULTS
# ═══════════════════════════════════════════════════════════════

_M17_DEFAULTS = {
    'M17_ENABLED': True,
    'M17_WEIGHT': 0.05,
    # Zone volume analysis
    'M17_ZONE_LOOKBACK': 960,           # 10 days of 15m bars
    'M17_ZONE_WIDTH_PCT': 0.5,          # zone width as % of price (±0.5%)
    'M17_ZONE_THIN_THRESHOLD': 0.5,     # volume ratio below this = thin zone
    'M17_ZONE_THICK_THRESHOLD': 1.5,    # volume ratio above this = thick zone
    # Rejection analysis
    'M17_REJECT_MIN_TOUCHES': 3,        # minimum touches to analyze
    'M17_REJECT_VOL_STRONG': 2.5,       # vol ratio = "active defender"
    'M17_REJECT_VOL_WEAK': 1.2,         # vol ratio = "passive defender"
    'M17_REJECT_WICK_STRONG': 5.0,      # avg wick above zone ($) = strong defense
    # Defender behavior
    'M17_DEFENDER_TRAPPED_Z': -2.0,     # L/S z-score = defender trapped (shorts)
    'M17_DEFENDER_TRAPPED_Z_LONG': 2.0, # L/S z-score = defender trapped (longs)
    'M17_DEFENDER_LEAVING_OI_ROC': -0.5, # OI dropping = defender exiting
    'M17_DEFENDER_ADDING_OI_ROC': 1.0,  # OI rising = defender building
    # Breakout readiness
    'M17_BREAKOUT_VOL_MIN': 1.2,        # min volume surge for breakout attempt
    'M17_BREAKOUT_MOMENTUM_MIN': 0.3,   # min % move toward zone
    # Scoring weights
    'M17_W_ZONE': 0.25,
    'M17_W_REJECT': 0.30,
    'M17_W_DEFENDER': 0.25,
    'M17_W_READINESS': 0.20,
}


def _cfg(config, key):
    if config and key in config:
        return config[key]
    return _M17_DEFAULTS[key]


# ═══════════════════════════════════════════════════════════════
# SIGNAL 1: ZONE VOLUME DENSITY
# ═══════════════════════════════════════════════════════════════

def _score_zone_volume(zone_price, df_15m, idx, bin_centers, vol_profile, cfg):
    """
    Is the resistance/support zone a high-volume node (thick) or low-volume gap (thin)?

    Thin zone = easy to break through (no orders sitting there)
    Thick zone = hard to break (lots of trading activity = orders)
    """
    if bin_centers is None or vol_profile is None or len(bin_centers) == 0:
        return 0.5, {'status': 'NO_DATA'}

    zone_pct = cfg['M17_ZONE_WIDTH_PCT'] / 100
    zone_lo = zone_price * (1 - zone_pct)
    zone_hi = zone_price * (1 + zone_pct)

    zone_mask = (bin_centers >= zone_lo) & (bin_centers <= zone_hi)
    if not zone_mask.any():
        return 0.5, {'status': 'NO_ZONE_BINS'}

    zone_vol = np.mean(vol_profile[zone_mask])
    chart_vol = np.mean(vol_profile)
    ratio = zone_vol / chart_vol if chart_vol > 0 else 1.0

    thin_thresh = cfg['M17_ZONE_THIN_THRESHOLD']
    thick_thresh = cfg['M17_ZONE_THICK_THRESHOLD']

    if ratio < thin_thresh * 0.5:
        score = 0.95   # very thin
    elif ratio < thin_thresh:
        score = 0.8    # thin
    elif ratio < 1.0:
        score = 0.6    # below average
    elif ratio < thick_thresh:
        score = 0.35   # above average — harder
    else:
        score = 0.1    # high volume node — very thick

    status = 'THIN' if ratio < thin_thresh else 'THICK' if ratio > thick_thresh else 'NORMAL'

    return score, {
        'status': status,
        'zone_vol_ratio': round(ratio, 3),
        'zone_lo': round(zone_lo, 2),
        'zone_hi': round(zone_hi, 2),
    }


# ═══════════════════════════════════════════════════════════════
# SIGNAL 2: REJECTION PATTERN
# ═══════════════════════════════════════════════════════════════

def _score_rejection_pattern(zone_price, df_15m, idx, cfg):
    """
    How aggressively does price get rejected from this zone?

    Strong rejection (high vol + big wicks) = active defender
    Weak rejection (low vol + small wicks) = passive or absent defender
    """
    lookback = min(cfg['M17_ZONE_LOOKBACK'], idx)
    zone_pct = cfg['M17_ZONE_WIDTH_PCT'] / 100
    zone_lo = zone_price * (1 - zone_pct)
    zone_hi = zone_price * (1 + zone_pct)

    highs = df_15m['High'].values[idx - lookback + 1:idx + 1].astype(float)
    lows = df_15m['Low'].values[idx - lookback + 1:idx + 1].astype(float)
    closes = df_15m['Close'].values[idx - lookback + 1:idx + 1].astype(float)
    volumes = df_15m['Volume'].values[idx - lookback + 1:idx + 1].astype(float)

    touches = []
    for i in range(len(highs)):
        h, l, c, v = highs[i], lows[i], closes[i], volumes[i]
        if h >= zone_lo:
            wick_above = h - max(c, zone_lo)
            rejected = c < zone_lo
            touches.append({
                'rejected': rejected,
                'wick_above': wick_above,
                'vol': v,
                'close_inside': c >= zone_lo,
            })

    if len(touches) < cfg['M17_REJECT_MIN_TOUCHES']:
        return 0.5, {'status': 'INSUFFICIENT_DATA', 'touches': len(touches)}

    rejects = [t for t in touches if t['rejected']]
    accepts = [t for t in touches if t['close_inside']]
    reject_rate = len(rejects) / len(touches) if touches else 0

    # Volume on rejections vs average
    avg_vol = np.mean(volumes) if len(volumes) > 0 else 1
    avg_reject_vol = np.mean([t['vol'] for t in rejects]) if rejects else avg_vol
    vol_ratio = avg_reject_vol / avg_vol if avg_vol > 0 else 1.0

    # Wick strength
    avg_wick = np.mean([t['wick_above'] for t in rejects]) if rejects else 0

    # Score: high reject rate + high vol = strong defender = LOW breakout score
    vol_score = min(vol_ratio / 3.0, 1.0)  # normalize to 0-1
    wick_score = min(avg_wick / cfg['M17_REJECT_WICK_STRONG'], 1.0)

    # Composite: defender strength (inverted — strong defender = low breakout score)
    defender_strength = (
        reject_rate * 0.40 +
        vol_score * 0.35 +
        wick_score * 0.25
    )
    # Invert: strong defender → low breakout score
    score = 1.0 - defender_strength

    # Status
    if vol_ratio >= cfg['M17_REJECT_VOL_STRONG'] and reject_rate > 0.3:
        status = 'ACTIVE_DEFENDER'
    elif vol_ratio >= cfg['M17_REJECT_VOL_WEAK']:
        status = 'MODERATE_DEFENDER'
    elif reject_rate < 0.2:
        status = 'ZONE_BREAKING'  # rejections failing, zone weakening
    else:
        status = 'PASSIVE_DEFENDER'

    return score, {
        'status': status,
        'touches': len(touches),
        'rejects': len(rejects),
        'accepts': len(accepts),
        'reject_rate': round(reject_rate, 3),
        'vol_ratio_on_reject': round(vol_ratio, 3),
        'avg_wick_above': round(avg_wick, 2),
    }


# ═══════════════════════════════════════════════════════════════
# SIGNAL 3: DEFENDER BEHAVIOR
# ═══════════════════════════════════════════════════════════════

def _score_defender_behavior(deriv, direction, cfg):
    """
    Is the defender accumulating or distributing?

    OI increasing at resistance + extreme z = defender TRAPPED (squeeze fuel)
    OI decreasing at resistance = defender EXITING (resistance fading)
    """
    if not deriv or 'error' in deriv:
        return 0.5, {'status': 'NO_DATA'}

    oi_roc = deriv.get('oi_roc_1h', 0)
    ls_z = deriv.get('ls_zscore', 0)
    positioning = deriv.get('positioning', 'NEUTRAL')
    whale = deriv.get('whale_signal', 'NEUTRAL')
    fr = deriv.get('funding_rate', 0)

    trapped_z = cfg['M17_DEFENDER_TRAPPED_Z'] if direction == 'LONG' else cfg['M17_DEFENDER_TRAPPED_Z_LONG']
    leaving_roc = cfg['M17_DEFENDER_LEAVING_OI_ROC']
    adding_roc = cfg['M17_DEFENDER_ADDING_OI_ROC']

    factors = []
    score = 0.5

    if direction == 'LONG':
        # For longs breaking resistance:
        # Shorts trapped = OI rising + extreme short positioning
        if oi_roc > adding_roc and ls_z < trapped_z:
            score = 0.85
            status = 'DEFENDER_TRAPPED'
            factors.append(f'OI +{oi_roc:.2f}%/hr with z={ls_z:.2f} — shorts trapped')
        elif oi_roc > adding_roc:
            score = 0.35
            status = 'DEFENDER_ADDING'
            factors.append(f'OI +{oi_roc:.2f}%/hr — new shorts opening, resistance thickening')
        elif oi_roc < leaving_roc:
            score = 0.75
            status = 'DEFENDER_LEAVING'
            factors.append(f'OI {oi_roc:+.2f}%/hr — shorts covering, resistance fading')
        elif positioning == 'CROWDED_SHORT':
            score = 0.65
            status = 'CROWDED_BUT_STABLE'
            factors.append(f'Crowded short (z={ls_z:.2f}) but OI stable')
        else:
            score = 0.50
            status = 'NEUTRAL'

        # Whale confirmation
        if whale == 'WHALE_BEARISH' and status == 'DEFENDER_TRAPPED':
            score = min(score + 0.05, 1.0)
            factors.append('Whales bearish — smart money defending (but trapped)')
        elif whale == 'WHALE_BULLISH' and status == 'DEFENDER_TRAPPED':
            score = min(score + 0.10, 1.0)
            factors.append('Whales bullish — defender has no backup')

    elif direction == 'SHORT':
        # For shorts breaking support:
        # Longs trapped = OI rising + extreme long positioning
        if oi_roc > adding_roc and ls_z > trapped_z:
            score = 0.85
            status = 'DEFENDER_TRAPPED'
            factors.append(f'OI +{oi_roc:.2f}%/hr with z={ls_z:.2f} — longs trapped')
        elif oi_roc > adding_roc:
            score = 0.35
            status = 'DEFENDER_ADDING'
            factors.append(f'OI +{oi_roc:.2f}%/hr — new longs opening, support thickening')
        elif oi_roc < leaving_roc:
            score = 0.75
            status = 'DEFENDER_LEAVING'
            factors.append(f'OI {oi_roc:+.2f}%/hr — longs covering, support fading')
        elif positioning == 'CROWDED_LONG':
            score = 0.65
            status = 'CROWDED_BUT_STABLE'
            factors.append(f'Crowded long (z={ls_z:.2f}) but OI stable')
        else:
            score = 0.50
            status = 'NEUTRAL'

        if whale == 'WHALE_BULLISH' and status == 'DEFENDER_TRAPPED':
            score = min(score + 0.05, 1.0)
            factors.append('Whales bullish — smart money defending (but trapped)')
        elif whale == 'WHALE_BEARISH' and status == 'DEFENDER_TRAPPED':
            score = min(score + 0.10, 1.0)
            factors.append('Whales bearish — defender has no backup')

    # Funding bonus/penalty
    if fr is not None:
        if direction == 'LONG' and fr > 0.0005:
            score = min(score + 0.05, 1.0)
            factors.append(f'Funding {fr*100:+.4f}% — longs paying (squeeze pressure)')
        elif direction == 'SHORT' and fr < -0.0005:
            score = min(score + 0.05, 1.0)
            factors.append(f'Funding {fr*100:+.4f}% — shorts paying (squeeze pressure)')

    return score, {
        'status': status,
        'oi_roc_1h': round(oi_roc, 3),
        'ls_zscore': round(ls_z, 2),
        'positioning': positioning,
        'whale': whale,
        'factors': factors,
    }


# ═══════════════════════════════════════════════════════════════
# SIGNAL 4: BREAKOUT READINESS
# ═══════════════════════════════════════════════════════════════

def _score_breakout_readiness(zone_price, df_15m, idx, direction, cfg):
    """
    Is price approaching with enough force to break?

    Checks: momentum direction, volume surge, recent price action
    """
    lookback = min(20, idx)
    closes = df_15m['Close'].values[idx - lookback + 1:idx + 1].astype(float)
    highs = df_15m['High'].values[idx - lookback + 1:idx + 1].astype(float)
    volumes = df_15m['Volume'].values[idx - lookback + 1:idx + 1].astype(float)

    vol_ma = df_15m['vol_ma20'].iloc[idx] if 'vol_ma20' in df_15m.columns else np.mean(volumes)
    if vol_ma == 0:
        vol_ma = np.mean(volumes)

    momentum = (closes[-1] - closes[0]) / closes[0] * 100
    vol_surge = volumes[-1] / vol_ma if vol_ma > 0 else 1.0
    avg_vol_surge = np.mean(volumes) / vol_ma if vol_ma > 0 else 1.0

    # Distance to zone
    dist_to_zone = (zone_price - closes[-1]) / closes[-1] * 100

    factors = []
    score = 0.5

    if direction == 'LONG':
        # Approaching resistance from below
        approaching = momentum > cfg['M17_BREAKOUT_MOMENTUM_MIN']
        close_to_zone = abs(dist_to_zone) < 1.0  # within 1%
        has_volume = vol_surge > cfg['M17_BREAKOUT_VOL_MIN']

        if approaching and has_volume:
            score = 0.80
            factors.append(f'Momentum +{momentum:.2f}% with vol surge {vol_surge:.2f}x')
        elif approaching:
            score = 0.65
            factors.append(f'Momentum +{momentum:.2f}% but low volume')
        elif close_to_zone and has_volume:
            score = 0.70
            factors.append(f'At zone with vol surge {vol_surge:.2f}x')
        elif close_to_zone:
            score = 0.55
            factors.append(f'At zone but no momentum/volume')
        else:
            score = 0.40
            factors.append(f'Not approaching zone (momentum {momentum:+.2f}%)')

    elif direction == 'SHORT':
        # Approaching support from above
        approaching = momentum < -cfg['M17_BREAKOUT_MOMENTUM_MIN']
        close_to_zone = abs(dist_to_zone) < 1.0
        has_volume = vol_surge > cfg['M17_BREAKOUT_VOL_MIN']

        if approaching and has_volume:
            score = 0.80
            factors.append(f'Momentum {momentum:.2f}% with vol surge {vol_surge:.2f}x')
        elif approaching:
            score = 0.65
            factors.append(f'Momentum {momentum:.2f}% but low volume')
        elif close_to_zone and has_volume:
            score = 0.70
            factors.append(f'At zone with vol surge {vol_surge:.2f}x')
        elif close_to_zone:
            score = 0.55
            factors.append(f'At zone but no momentum/volume')
        else:
            score = 0.40
            factors.append(f'Not approaching zone (momentum {momentum:+.2f}%)')

    return score, {
        'momentum': round(momentum, 3),
        'vol_surge': round(vol_surge, 3),
        'avg_vol_surge': round(avg_vol_surge, 3),
        'dist_to_zone': round(dist_to_zone, 3),
        'factors': factors,
    }


# ═══════════════════════════════════════════════════════════════
# COMPOSITE SCORING
# ═══════════════════════════════════════════════════════════════

def score_resistance_quality(zone_price, df_15m, idx, bin_centers, vol_profile,
                             deriv, direction, config=None):
    """
    Evaluate the quality of a resistance (or support) level.

    Args:
        zone_price: price of the S/R level
        df_15m: 15m DataFrame
        idx: current bar index
        bin_centers: volume profile bin centers (from build_volume_profile)
        vol_profile: volume profile values
        derivatives: derivatives summary dict
        direction: 'LONG' (testing resistance) or 'SHORT' (testing support)
        config: optional config dict

    Returns:
        dict with composite score, sub-signal scores, details, verdict
    """
    cfg = {**_M17_DEFAULTS, **(config or {})}

    if not cfg.get('M17_ENABLED', True):
        return None

    # Score all 4 signals
    s1_score, s1_details = _score_zone_volume(
        zone_price, df_15m, idx, bin_centers, vol_profile, cfg)
    s2_score, s2_details = _score_rejection_pattern(
        zone_price, df_15m, idx, cfg)
    s3_score, s3_details = _score_defender_behavior(
        deriv, direction, cfg)
    s4_score, s4_details = _score_breakout_readiness(
        zone_price, df_15m, idx, direction, cfg)

    # Weighted composite
    w_zone = cfg['M17_W_ZONE']
    w_reject = cfg['M17_W_REJECT']
    w_defender = cfg['M17_W_DEFENDER']
    w_ready = cfg['M17_W_READINESS']

    composite = (
        s1_score * w_zone +
        s2_score * w_reject +
        s3_score * w_defender +
        s4_score * w_ready
    )
    composite = max(0.0, min(1.0, composite))

    # Verdict
    if composite >= 0.75:
        verdict = 'BREAKOUT_LIKELY'
    elif composite >= 0.60:
        verdict = 'BREAKOUT_POSSIBLE'
    elif composite >= 0.40:
        verdict = 'CONTESTED'
    elif composite >= 0.25:
        verdict = 'DEFENSE_STRONG'
    else:
        verdict = 'RESISTANCE_DOMINANT'

    # Breakout confirmation criteria
    confirmation = []
    if s1_details.get('status') == 'THIN':
        confirmation.append('Zone is thin (low volume node)')
    if s2_details.get('status') in ('ZONE_BREAKING', 'PASSIVE_DEFENDER'):
        confirmation.append(f'Rejections weakening ({s2_details.get("reject_rate", 0):.0%} reject rate)')
    if s3_details.get('status') == 'DEFENDER_TRAPPED':
        confirmation.append('Defender trapped (OI rising + extreme positioning)')
    elif s3_details.get('status') == 'DEFENDER_LEAVING':
        confirmation.append('Defender leaving (OI declining)')
    if s4_details.get('momentum', 0) > 0.5:
        confirmation.append(f'Positive momentum ({s4_details["momentum"]:+.2f}%)')

    # What's needed for breakout
    needed = []
    if s1_details.get('status') == 'THICK':
        needed.append('Need sustained volume to penetrate thick zone')
    if s2_details.get('status') == 'ACTIVE_DEFENDER':
        needed.append(f'Active defender (vol {s2_details.get("vol_ratio_on_reject", 0):.1f}x on rejections)')
    if s3_details.get('status') == 'DEFENDER_ADDING':
        needed.append('Defender adding positions — resistance getting thicker')
    if s4_details.get('vol_surge', 1) < cfg['M17_BREAKOUT_VOL_MIN']:
        needed.append(f'Need vol > {cfg["M17_BREAKOUT_VOL_MIN"]}x on breakout candle')

    return {
        'zone_price': round(zone_price, 2),
        'direction': direction,
        'composite': round(composite, 4),
        'verdict': verdict,
        'zone_volume': s1_details,
        'rejection': s2_details,
        'defender': s3_details,
        'readiness': s4_details,
        'confirmation': confirmation,
        'needed': needed,
    }


def format_resistance_quality(result):
    """Format resistance quality analysis for display."""
    if not result:
        return ''

    lines = []
    zone = result['zone_price']
    direction = result['direction']
    level_type = 'Resistance' if direction == 'LONG' else 'Support'

    lines.append(f"  {level_type} Quality (\${zone:.2f}):")
    lines.append(f"    Composite:    {result['composite']:.3f}  → {result['verdict']}")

    # Zone volume
    zv = result.get('zone_volume', {})
    if zv.get('status') != 'NO_DATA':
        lines.append(f"    Zone volume:  {zv.get('zone_vol_ratio', 0):.2f}x avg  ({zv.get('status', '?')})")

    # Rejection
    rej = result.get('rejection', {})
    if rej.get('status') != 'INSUFFICIENT_DATA':
        lines.append(f"    Rejections:   {rej.get('rejects', 0)}/{rej.get('touches', 0)} touches "
                     f"({rej.get('reject_rate', 0):.0%})  vol={rej.get('vol_ratio_on_reject', 0):.2f}x "
                     f"wick=\${rej.get('avg_wick_above', 0):.2f}  [{rej.get('status', '?')}]")

    # Defender
    dfn = result.get('defender', {})
    if dfn.get('status') != 'NO_DATA':
        lines.append(f"    Defender:     {dfn.get('status', '?')}  "
                     f"OI {dfn.get('oi_roc_1h', 0):+.2f}%/hr  "
                     f"z={dfn.get('ls_zscore', 0):.2f}")
        for f in dfn.get('factors', []):
            lines.append(f"      • {f}")

    # Readiness
    rdy = result.get('readiness', {})
    lines.append(f"    Readiness:    momentum={rdy.get('momentum', 0):+.2f}%  "
                 f"vol={rdy.get('vol_surge', 0):.2f}x  "
                 f"dist={rdy.get('dist_to_zone', 0):+.2f}%")
    for f in rdy.get('factors', []):
        lines.append(f"      • {f}")

    # Confirmation
    conf = result.get('confirmation', [])
    if conf:
        lines.append(f"    ✅ Confirmations:")
        for c in conf:
            lines.append(f"        {c}")

    # Needed
    needed = result.get('needed', [])
    if needed:
        lines.append(f"    ⚠️  Still needed:")
        for n in needed:
            lines.append(f"        {n}")

    return '\n'.join(lines)

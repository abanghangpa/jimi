"""
M9: Volatility Regime Classifier v2 — Complete Regime Detection

Detects 5 regimes with 7 signals:
  CRISIS      — extreme volatility, no direction safe
  CHOP        — whipsaw/range-bound, direction unreliable
  COMPRESSING — low vol squeeze, breakout imminent
  TRENDING    — directional with volume confirmation
  NEUTRAL     — nothing special, proceed with caution

Signals:
  1. ATR percentile (1H) — volatility fuel
  2. BB width percentile (1H) — squeeze/expansion
  3. Directionality (15m + 1H) — directional consistency
  4. Whipsaw rate (15m) — how often moves reverse within N bars
  5. Retracement ratio (15m) — how much of each move gets undone
  6. Volume confirmation — trend without volume is suspect
  7. Range tightness (15m) — price bouncing in a band
"""

import pandas as pd
import numpy as np


class RegimeState:
    """Tracks regime history with hysteresis to prevent flickering."""

    HYSTERESIS = {
        'CRISIS': {
            'atr_pctl_enter': 0.85, 'atr_pctl_exit': 0.75,
            'bb_pctl_enter': 0.85, 'bb_pctl_exit': 0.75,
            'confirm_bars': 2,
        },
        'TRENDING': {
            'trend_score_enter': 0.50, 'trend_score_exit': 0.40,
            'directionality_exit': 0.30,
            'structure_exit': 0.30,
            'confirm_bars': 3,
        },
        'COMPRESSING': {
            'bb_pctl_enter': 0.30, 'bb_pctl_exit': 0.40,
            'atr_pctl_enter': 0.40, 'atr_pctl_exit': 0.50,
            'confirm_bars': 3,
        },
        'CHOP_HARD': {
            'chop_score_enter': 0.72, 'chop_score_exit': 0.65,
            'whipsaw_exit': 0.55, 'retrace_exit': 0.60,
            'confirm_bars': 2,
        },
        'CHOP_MILD': {
            'chop_score_enter': 0.55, 'chop_score_exit': 0.45,
            'whipsaw_exit': 0.45, 'retrace_exit': 0.50,
            'confirm_bars': 2,
        },
    }

    TRANSITION_COOLDOWN = {
        'CRISIS': 8, 'TRENDING': 4, 'COMPRESSING': 6, 'CHOP_HARD': 6, 'CHOP_MILD': 4,
    }

    def __init__(self):
        self.prev_regime = 'NEUTRAL'
        self.candidate_regime = None
        self.candidate_count = 0
        self.cooldown_remaining = 0
        self.transition_log = []
        self.regime_bar_count = 0  # how long in current regime

    def update(self, raw_regime, signals, timestamp=None):
        details = {}
        self.regime_bar_count += 1

        if self.cooldown_remaining > 0:
            self.cooldown_remaining -= 1
            details['cooldown_remaining'] = self.cooldown_remaining
            return self.prev_regime, False, details

        if raw_regime == self.prev_regime:
            self.candidate_regime = None
            self.candidate_count = 0
            return self.prev_regime, False, details

        should_transition = False
        transition_reason = ''

        # Exit conditions for each regime
        if self.prev_regime == 'CRISIS':
            hyst = self.HYSTERESIS['CRISIS']
            if signals['atr_pctl'] < hyst['atr_pctl_exit'] and signals['bb_pctl'] < hyst['bb_pctl_exit']:
                should_transition = True
                transition_reason = "CRISIS exit: volatility subsiding"

        elif self.prev_regime == 'TRENDING':
            hyst = self.HYSTERESIS['TRENDING']
            dir_ok = signals['directionality'] < hyst['directionality_exit']
            struct_ok = signals['structure_score'] < hyst['structure_exit']
            whipsaw_bad = signals['whipsaw_rate'] > 0.45
            retrace_bad = signals['retrace_ratio'] > 0.55
            if (dir_ok and struct_ok) or whipsaw_bad or retrace_bad:
                should_transition = True
                transition_reason = f"TRENDING exit: dir={signals['directionality']:.2f} whipsaw={signals['whipsaw_rate']:.2f} retrace={signals['retrace_ratio']:.2f}"

        elif self.prev_regime == 'COMPRESSING':
            hyst = self.HYSTERESIS['COMPRESSING']
            if signals['bb_pctl'] > hyst['bb_pctl_exit'] or signals['atr_pctl'] > hyst['atr_pctl_exit']:
                should_transition = True
                transition_reason = "COMPRESSING exit: expanding"

        elif self.prev_regime in ('CHOP_HARD', 'CHOP_MILD'):
            hyst = self.HYSTERESIS.get(self.prev_regime, self.HYSTERESIS['CHOP_MILD'])
            if signals['whipsaw_rate'] < hyst.get('whipsaw_exit', 0.45) and signals['retrace_ratio'] < hyst.get('retrace_exit', 0.50):
                should_transition = True
                transition_reason = "CHOP exit: becoming directional"

        else:
            should_transition = True
            transition_reason = f"From {self.prev_regime}"

        if not should_transition:
            self.candidate_regime = None
            self.candidate_count = 0
            return self.prev_regime, False, details

        if self.candidate_regime == raw_regime:
            self.candidate_count += 1
        else:
            self.candidate_regime = raw_regime
            self.candidate_count = 1

        confirm_needed = self.HYSTERESIS.get(raw_regime, {}).get('confirm_bars', 2)

        if self.candidate_count >= confirm_needed:
            old_regime = self.prev_regime
            self.prev_regime = raw_regime
            self.candidate_regime = None
            self.candidate_count = 0
            self.regime_bar_count = 0
            self.cooldown_remaining = self.TRANSITION_COOLDOWN.get(raw_regime, 4)

            entry = {
                'timestamp': timestamp, 'from': old_regime, 'to': raw_regime,
                'reason': transition_reason,
                'signals': {k: round(v, 3) if isinstance(v, float) else v for k, v in signals.items()},
            }
            self.transition_log.append(entry)
            details['transition'] = entry
            return raw_regime, True, details
        else:
            details['pending_transition'] = {
                'target': raw_regime, 'bars_confirmed': self.candidate_count,
                'bars_needed': confirm_needed,
            }
            return self.prev_regime, False, details


# ═══════════════════════════════════════════════════════════════
# SIGNAL COMPUTATIONS
# ═══════════════════════════════════════════════════════════════

def _compute_atr_percentile(df_1h, idx_1h, lookback=180):
    """ATR percentile over lookback bars of 1H data."""
    if 'atr' in df_1h.columns:
        atr = df_1h['atr'].iloc[idx_1h]
        series = df_1h['atr'].iloc[max(0, idx_1h - lookback):idx_1h + 1].dropna()
    else:
        h = df_1h['High']
        l = df_1h['Low']
        c = df_1h['Close']
        tr = pd.concat([h - l, (h - c.shift(1)).abs(), (l - c.shift(1)).abs()], axis=1).max(axis=1)
        atr_series = tr.ewm(span=14, adjust=False).mean()
        atr = atr_series.iloc[idx_1h]
        series = atr_series.iloc[max(0, idx_1h - lookback):idx_1h + 1].dropna()

    if len(series) < 20 or pd.isna(atr) or atr <= 0:
        return 0.5
    return float((atr - series.min()) / (series.max() - series.min() + 1e-10))


def _compute_bb_percentile(df_1h, idx_1h, lookback=120):
    """Bollinger Band width percentile."""
    close = df_1h['Close'].iloc[max(0, idx_1h - lookback):idx_1h + 1]
    if len(close) < 20:
        return 0.5

    sma = close.rolling(20).mean()
    std = close.rolling(20).std()
    bb_width = 2 * std / sma
    bb_width = bb_width.dropna()

    if len(bb_width) < 20:
        return 0.5

    current = bb_width.iloc[-1]
    if pd.isna(current):
        return 0.5
    return float((current - bb_width.min()) / (bb_width.max() - bb_width.min() + 1e-10))


def _compute_directionality(closes, window=20):
    """How directional is the recent price action? 0=random, 1=perfectly directional."""
    if len(closes) < window:
        return 0.5

    recent = closes.iloc[-window:]
    diffs = recent.diff().dropna()

    if len(diffs) == 0:
        return 0.5

    # Count consecutive same-direction closes
    signs = np.sign(diffs.values)
    consec = 0
    max_consec = 0
    for i in range(1, len(signs)):
        if signs[i] == signs[i - 1] and signs[i] != 0:
            consec += 1
            max_consec = max(max_consec, consec)
        else:
            consec = 0

    # Perfect trend = all bars same direction = max_consec = window-1
    # Random = max_consec ~ 1-2
    directionality = min(max_consec / (window - 1), 1.0)

    # Also factor in net move / total move (efficiency ratio)
    net_move = abs(recent.iloc[-1] - recent.iloc[0])
    total_move = diffs.abs().sum()
    efficiency = net_move / total_move if total_move > 0 else 0.5

    # Blend: 60% consecutive, 40% efficiency
    return directionality * 0.6 + efficiency * 0.4


def _compute_whipsaw_rate(closes, window=40, reversal_bars=3):
    """
    Whipsaw rate: what fraction of recent moves reversed within N bars?
    High whipsaw = market is choppy, moves don't stick.
    0.0 = no reversals (clean trend), 1.0 = everything reverses (pure chop)
    """
    if len(closes) < window + reversal_bars:
        return 0.5

    recent = closes.iloc[-(window + reversal_bars):]
    diffs = recent.diff().dropna()

    if len(diffs) < window:
        return 0.5

    # Find "moves" — consecutive same-direction bars
    moves = []
    current_dir = 0
    move_start = 0
    move_bars = 0

    for i, d in enumerate(diffs.values):
        d_sign = np.sign(d)
        if d_sign == 0:
            continue
        if d_sign == current_dir:
            move_bars += 1
        else:
            if current_dir != 0 and move_bars >= 1:
                moves.append({
                    'dir': current_dir,
                    'start': move_start,
                    'end': i,
                    'bars': move_bars,
                    'magnitude': abs(diffs.iloc[move_start:i].sum()),
                })
            current_dir = d_sign
            move_start = i
            move_bars = 1

    # Add last move
    if current_dir != 0 and move_bars >= 1:
        moves.append({
            'dir': current_dir, 'start': move_start, 'end': len(diffs),
            'bars': move_bars, 'magnitude': abs(diffs.iloc[move_start:].sum()),
        })

    if len(moves) < 3:
        return 0.5

    # Check how many moves got reversed within reversal_bars
    reversed_count = 0
    total_moves = len(moves) - 1  # exclude last (still in progress)

    for i in range(total_moves):
        move = moves[i]
        # Look at the next move
        if i + 1 < len(moves):
            next_move = moves[i + 1]
            # If next move is opposite direction and starts within reversal_bars
            if next_move['dir'] == -move['dir']:
                # Check if it retraces most of the move
                if next_move['magnitude'] >= move['magnitude'] * 0.5:
                    reversed_count += 1

    return reversed_count / max(total_moves, 1)


def _compute_retracement_ratio(closes, window=40):
    """
    Retracement ratio: average % of each move that gets retraced.
    0.0 = moves hold (trend), 1.0 = moves fully retraced (range/chop)
    """
    if len(closes) < window:
        return 0.5

    recent = closes.iloc[-window:]
    high = recent.max()
    low = recent.min()
    current = recent.iloc[-1]

    if high == low:
        return 0.5

    # How far are we from the extremes?
    range_size = high - low

    # Find swing points (local highs/lows)
    values = recent.values
    swings = []
    for i in range(2, len(values) - 2):
        if values[i] > values[i-1] and values[i] > values[i-2] and values[i] > values[i+1] and values[i] > values[i+2]:
            swings.append(('H', values[i], i))
        elif values[i] < values[i-1] and values[i] < values[i-2] and values[i] < values[i+1] and values[i] < values[i+2]:
            swings.append(('L', values[i], i))

    if len(swings) < 3:
        # Not enough swings — check position in range
        position = (current - low) / range_size
        # If we're in the middle, it's choppy
        return 1.0 - abs(position - 0.5) * 2

    # Compute retracement between consecutive swing pairs
    retracements = []
    for i in range(len(swings) - 2):
        s1 = swings[i]
        s2 = swings[i + 1]
        s3 = swings[i + 2]

        if s1[0] == 'H' and s2[0] == 'L' and s3[0] == 'H':
            # High-Low-High pattern
            move_down = s1[1] - s2[1]
            retrace_up = s3[1] - s2[1]
            if move_down > 0:
                retracements.append(retrace_up / move_down)
        elif s1[0] == 'L' and s2[0] == 'H' and s3[0] == 'L':
            # Low-High-Low pattern
            move_up = s2[1] - s1[1]
            retrace_down = s2[1] - s3[1]
            if move_up > 0:
                retracements.append(retrace_down / move_up)

    if not retracements:
        return 0.5

    # Average retracement ratio
    avg_retrace = np.mean(retracements)
    return min(max(avg_retrace, 0.0), 1.0)


def _compute_volume_confirmation(df_1h, idx_1h, window=20):
    """
    Volume confirmation: is volume supporting the price move?
    Returns 0-1 where 1 = strong volume confirmation.
    """
    if 'Volume' not in df_1h.columns or idx_1h < window:
        return 0.5

    vol = df_1h['Volume'].iloc[max(0, idx_1h - window):idx_1h + 1]
    close = df_1h['Close'].iloc[max(0, idx_1h - window):idx_1h + 1]

    if len(vol) < 5:
        return 0.5

    avg_vol = vol.mean()
    if avg_vol <= 0:
        return 0.5

    # Current volume vs average
    vol_ratio = vol.iloc[-1] / avg_vol if avg_vol > 0 else 1.0

    # Price direction
    price_change = close.iloc[-1] - close.iloc[-2] if len(close) >= 2 else 0
    direction = np.sign(price_change)

    # Volume on up bars vs down bars
    diffs = close.diff().dropna()
    up_vol = vol.iloc[1:][diffs > 0].mean() if (diffs > 0).any() else avg_vol
    down_vol = vol.iloc[1:][diffs < 0].mean() if (diffs < 0).any() else avg_vol

    # Volume should confirm direction
    if direction > 0:
        vol_confirm = up_vol / (down_vol + 1e-10)
    elif direction < 0:
        vol_confirm = down_vol / (up_vol + 1e-10)
    else:
        vol_confirm = 1.0

    # Combine: current volume ratio + directional volume confirmation
    score = min(vol_ratio, 2.0) / 2.0 * 0.4 + min(vol_confirm, 2.0) / 2.0 * 0.6
    return min(max(score, 0.0), 1.0)


def _compute_range_tightness(closes, window=40):
    """
    Range tightness: is price confined to a narrow range?
    0.0 = wide range (trending/volatile), 1.0 = tight range (consolidation)
    """
    if len(closes) < window:
        return 0.5

    recent = closes.iloc[-window:]
    range_size = (recent.max() - recent.min()) / recent.mean() * 100  # as %

    # ETH typical ranges on 15m:
    # < 0.5% = very tight (compression)
    # 0.5-1.5% = normal
    # 1.5-3.0% = wide
    # > 3.0% = extreme

    if range_size < 0.3:
        return 1.0  # very tight
    elif range_size < 0.8:
        return 0.7  # tight
    elif range_size < 1.5:
        return 0.4  # normal
    elif range_size < 3.0:
        return 0.2  # wide
    else:
        return 0.0  # extreme


def _compute_1h_15m_coherence(df_15m, df_1h, idx_15m, idx_1h):
    """
    Do 15m and 1H agree on direction?
    Returns 0.0 (conflict) to 1.0 (strong agreement).
    """
    if idx_1h < 10 or idx_15m < 40:
        return 0.5

    # 1H direction: last 10 bars
    close_1h = df_1h['Close'].iloc[max(0, idx_1h - 10):idx_1h + 1]
    if len(close_1h) < 5:
        return 0.5
    dir_1h = np.sign(close_1h.iloc[-1] - close_1h.iloc[0])

    # 15m direction: last 40 bars (10 hours)
    close_15m = df_15m['Close'].iloc[max(0, idx_15m - 40):idx_15m + 1]
    if len(close_15m) < 10:
        return 0.5
    dir_15m = np.sign(close_15m.iloc[-1] - close_15m.iloc[0])

    if dir_1h == 0 or dir_15m == 0:
        return 0.5

    if dir_1h == dir_15m:
        return 1.0  # agreement
    else:
        return 0.0  # conflict


# ═══════════════════════════════════════════════════════════════
# MAIN REGIME COMPUTATION
# ═══════════════════════════════════════════════════════════════

def compute_vol_regime(df_15m, df_1h, idx_15m, idx_1h, regime_state=None):
    """
    Compute volatility regime for current bar with full signal suite.

    Returns: (regime, score, details)
      regime: 'CRISIS' | 'CHOP' | 'COMPRESSING' | 'TRENDING' | 'NEUTRAL' | 'UNKNOWN'
      score: 0.0-1.0 (how tradable the regime is)
      details: dict of all computed signals
    """
    details = {}

    if idx_1h < 20 or idx_15m < 40:
        return 'UNKNOWN', 0.5, details

    # ── Compute all 7 signals ──
    atr_pctl = _compute_atr_percentile(df_1h, idx_1h)
    bb_pctl = _compute_bb_percentile(df_1h, idx_1h)
    directionality_1h = _compute_directionality(df_1h['Close'].iloc[:idx_1h + 1])
    directionality_15m = _compute_directionality(df_15m['Close'].iloc[:idx_15m + 1])
    whipsaw_rate = _compute_whipsaw_rate(df_15m['Close'].iloc[:idx_15m + 1])
    retrace_ratio = _compute_retracement_ratio(df_15m['Close'].iloc[:idx_15m + 1])
    volume_confirm = _compute_volume_confirmation(df_1h, idx_1h)
    range_tight = _compute_range_tightness(df_15m['Close'].iloc[:idx_15m + 1])
    tf_coherence = _compute_1h_15m_coherence(df_15m, df_1h, idx_15m, idx_1h)

    # Structure score (HH/HL vs LH/LL) on 1H
    if idx_1h >= 10:
        highs = df_1h['High'].iloc[idx_1h - 10:idx_1h + 1].values
        lows = df_1h['Low'].iloc[idx_1h - 10:idx_1h + 1].values
        hh = sum(1 for i in range(1, len(highs)) if highs[i] > highs[i - 1])
        ll = sum(1 for i in range(1, len(lows)) if lows[i] < lows[i - 1])
        structure_score = abs(hh - ll) / max(hh + ll, 1)
    else:
        structure_score = 0.0

    # Volume ratio (1H)
    if 'Volume' in df_1h.columns and idx_1h >= 20:
        vol_ma = df_1h['Volume'].iloc[idx_1h - 20:idx_1h + 1].mean()
        vol_ratio = df_1h['Volume'].iloc[idx_1h] / vol_ma if vol_ma > 0 else 1.0
    else:
        vol_ratio = 1.0

    # Blend directionality: 40% 1H, 60% 15m (execution timeframe matters more)
    directionality = directionality_1h * 0.4 + directionality_15m * 0.6

    # Store all signals
    signals = {
        'atr_pctl': atr_pctl,
        'bb_pctl': bb_pctl,
        'directionality': directionality,
        'directionality_1h': directionality_1h,
        'directionality_15m': directionality_15m,
        'whipsaw_rate': whipsaw_rate,
        'retrace_ratio': retrace_ratio,
        'volume_confirm': volume_confirm,
        'range_tight': range_tight,
        'tf_coherence': tf_coherence,
        'structure_score': structure_score,
        'vol_ratio': vol_ratio,
    }
    details.update(signals)

    # ── Regime Classification ──
    # ETH on 15m is inherently choppy (median move 0.0035%/hr).
    # Instead of binary CHOP vs TRENDING, we use a tradability spectrum.
    #
    # Priority: CRISIS > CHOP_HARD > CHOP_MILD > COMPRESSING > TRENDING > NEUTRAL
    #
    # Key insight: not all chop is equal.
    #   CHOP_HARD: random whipsaws, no edge → block
    #   CHOP_MILD: range-bound but predictable → trade with reduced size
    #   TRENDING: rare but high-edge → full size

    # Composite chop score (0=clean trend, 1=pure noise)
    chop_score = (
        whipsaw_rate * 0.30 +
        retrace_ratio * 0.25 +
        (1.0 - directionality) * 0.20 +
        (1.0 - volume_confirm) * 0.15 +
        (1.0 - tf_coherence) * 0.10
    )

    # Composite trend score (0=random, 1=strong trend)
    trend_score = (
        directionality * 0.30 +
        structure_score * 0.20 +
        (1.0 - whipsaw_rate) * 0.20 +
        (1.0 - retrace_ratio) * 0.15 +
        volume_confirm * 0.10 +
        tf_coherence * 0.05
    )

    if atr_pctl > 0.85 or bb_pctl > 0.85:
        raw_regime = 'CRISIS'
        score = 0.10

    elif chop_score > 0.72 and whipsaw_rate > 0.70 and retrace_ratio > 0.80:
        # CHOP_HARD: extreme whipsaw + near-total retracement → no edge
        raw_regime = 'CHOP_HARD'
        score = 0.10

    elif chop_score > 0.55 and trend_score < 0.35:
        # CHOP_MILD: choppy but some structure → trade small
        raw_regime = 'CHOP_MILD'
        score = 0.35

    elif bb_pctl < 0.30 and atr_pctl < 0.40 and range_tight > 0.60:
        # COMPRESSING: low vol + tight range = breakout loading
        raw_regime = 'COMPRESSING'
        score = 0.50

    elif trend_score > 0.50 and directionality > 0.35 and \
         whipsaw_rate < 0.55 and retrace_ratio < 0.70:
        # TRENDING: meaningful direction with acceptable whipsaw
        raw_regime = 'TRENDING'
        score = 0.80

    else:
        raw_regime = 'NEUTRAL'
        score = 0.50

    details['raw_regime'] = raw_regime

    # ── Apply hysteresis ──
    if regime_state is not None:
        ts = df_15m['Open time'].iloc[idx_15m] if 'Open time' in df_15m.columns else None
        regime, is_transition, hyst_details = regime_state.update(raw_regime, signals, timestamp=ts)
        details.update(hyst_details)
        details['is_transition'] = is_transition

        # Regime score mapping
        regime_scores = {
            'CRISIS': 0.10, 'CHOP_HARD': 0.10, 'CHOP_MILD': 0.35,
            'COMPRESSING': 0.50, 'TRENDING': 0.80, 'NEUTRAL': 0.50, 'UNKNOWN': 0.50,
        }
        score = regime_scores.get(regime, 0.50)
    else:
        regime = raw_regime

    # Regime strength: how long have we been in this regime?
    if regime_state is not None:
        regime_strength = min(regime_state.regime_bar_count / 20.0, 1.0)
    else:
        regime_strength = 0.5
    details['regime_strength'] = round(regime_strength, 3)

    details['regime'] = regime
    details['vol_regime_score'] = round(score, 3)
    return regime, score, details


def score_vol_regime(regime, vol_regime_score, direction, trend_dir):
    """Score volatility regime for trade direction."""
    details = {'regime': regime}
    base = vol_regime_score

    if regime == 'TRENDING':
        if (trend_dir in ('STRONG_UP', 'UP') and direction == 'LONG') or \
           (trend_dir in ('STRONG_DOWN', 'DOWN') and direction == 'SHORT'):
            base = min(base * 1.15, 1.0)
        elif (trend_dir in ('STRONG_UP', 'UP') and direction == 'SHORT') or \
             (trend_dir in ('STRONG_DOWN', 'DOWN') and direction == 'LONG'):
            base *= 0.70

    if regime == 'CHOP_HARD':
        base *= 0.20  # hard block — no edge
    if regime == 'CHOP_MILD':
        base *= 0.55  # reduced size — some edge but risky
    if regime == 'CRISIS':
        base *= 0.15  # hard block
    if regime == 'COMPRESSING':
        base *= 0.85  # slightly reduced — waiting for breakout

    score = max(0.0, min(1.0, base))
    status = 'PASS' if score >= 0.45 else 'FAIL'
    details['vr_score'] = round(score, 3)
    return status, score, details

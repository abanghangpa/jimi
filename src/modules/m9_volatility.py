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


# ═══════════════════════════════════════════════════════════════
# DEFAULT CONFIG — used when no config dict is passed
# ═══════════════════════════════════════════════════════════════
_M9_DEFAULTS = {
    # Hysteresis thresholds (enter/exit for each regime)
    'M9_CRISIS_ATR_THRESHOLD': 0.85,
    'M9_CRISIS_BB_THRESHOLD': 0.90,
    'M9_WHIPSAW_THRESHOLD': 0.45,
    'M9_RETRACE_THRESHOLD': 0.50,
    'M9_TRENDING_MIN_DIR': 0.45,
    'M9_TRENDING_MIN_VOLUME': 0.50,
    'M9_TRENDING_MAX_WHIPSAW': 0.35,
    'M9_TRENDING_MAX_RETRACE': 0.45,
    'M9_TRENDING_MIN_COHERENCE': 0.50,
    # Regime classification thresholds
    'M9_CHOP_HARD_CHOP_SCORE': 0.72,
    'M9_CHOP_HARD_WHIPSAW': 0.70,
    'M9_CHOP_HARD_RETRACE': 0.80,
    'M9_CHOP_MILD_CHOP_SCORE': 0.50,
    'M9_CHOP_MILD_TREND_CEILING': 0.35,
    'M9_TRENDING_TREND_SCORE': 0.40,
    'M9_TRENDING_MAX_WHIPSAW_SCORE': 0.65,
    'M9_TRENDING_MAX_RETRACE_SCORE': 0.80,
    'M9_COMPRESSING_BB': 0.30,
    'M9_COMPRESSING_ATR': 0.40,
    'M9_COMPRESSING_RANGE': 0.60,
    # Hysteresis timing
    'M9_HYSTERESIS_CONFIRM_BARS': 2,
    'M9_TRENDING_CONFIRM_BARS': 3,
    'M9_COMPRESSING_CONFIRM_BARS': 3,
    'M9_CRISIS_COOLDOWN': 8,
    'M9_TRENDING_COOLDOWN': 4,
    'M9_COMPRESSING_COOLDOWN': 6,
    'M9_CHOP_HARD_COOLDOWN': 6,
    'M9_CHOP_MILD_COOLDOWN': 4,
    'M9_CHOP_SPLIT_ENABLED': True,
    'M9_CHOP_MAX_BARS': 96,
    'M9_CHOP_EXIT_COOLDOWN': 12,
    # NEUTRAL sub-classification thresholds
    'M9_NEUTRAL_DIR_THRESHOLD': 0.40,   # directionality above this = trending neutral
    'M9_NEUTRAL_VOL_THRESHOLD': 1.05,   # vol ratio above this = expanding
}


def _cfg(config, key):
    """Read a config key with M9_DEFAULTS fallback."""
    if config and key in config:
        return config[key]
    return _M9_DEFAULTS[key]


# ═══════════════════════════════════════════════════════════════
# REGIME STATE — hysteresis / flickering prevention
# ═══════════════════════════════════════════════════════════════

class RegimeState:
    """Tracks regime history with hysteresis to prevent flickering.

    All thresholds are read from config (settings.yaml) at init time.
    """

    def __init__(self, config=None):
        self.config = config

        # Build hysteresis table from config
        confirm_default = _cfg(config, 'M9_HYSTERESIS_CONFIRM_BARS')
        self.HYSTERESIS = {
            'CRISIS': {
                'atr_pctl_enter': _cfg(config, 'M9_CRISIS_ATR_THRESHOLD'),
                'atr_pctl_exit': _cfg(config, 'M9_CRISIS_ATR_THRESHOLD') - 0.10,
                'bb_pctl_enter': _cfg(config, 'M9_CRISIS_BB_THRESHOLD'),
                'bb_pctl_exit': _cfg(config, 'M9_CRISIS_BB_THRESHOLD') - 0.10,
                'confirm_bars': confirm_default,
            },
            'TRENDING': {
                'trend_score_enter': _cfg(config, 'M9_TRENDING_TREND_SCORE'),
                'trend_score_exit': _cfg(config, 'M9_TRENDING_TREND_SCORE') - 0.10,
                'directionality_exit': _cfg(config, 'M9_TRENDING_MIN_DIR') - 0.25,
                'structure_exit': 0.20,
                'whipsaw_exit': _cfg(config, 'M9_TRENDING_MAX_WHIPSAW') + 0.10,
                'retrace_exit': _cfg(config, 'M9_TRENDING_MAX_RETRACE') + 0.10,
                'confirm_bars': _cfg(config, 'M9_TRENDING_CONFIRM_BARS'),
            },
            'COMPRESSING': {
                'bb_pctl_enter': _cfg(config, 'M9_COMPRESSING_BB'),
                'bb_pctl_exit': _cfg(config, 'M9_COMPRESSING_BB') + 0.10,
                'atr_pctl_enter': _cfg(config, 'M9_COMPRESSING_ATR'),
                'atr_pctl_exit': _cfg(config, 'M9_COMPRESSING_ATR') + 0.10,
                'confirm_bars': _cfg(config, 'M9_COMPRESSING_CONFIRM_BARS'),
            },
            'CHOP_HARD': {
                'chop_score_enter': _cfg(config, 'M9_CHOP_HARD_CHOP_SCORE'),
                'chop_score_exit': _cfg(config, 'M9_CHOP_HARD_CHOP_SCORE') - 0.07,
                'whipsaw_exit': _cfg(config, 'M9_WHIPSAW_THRESHOLD') + 0.10,
                'retrace_exit': _cfg(config, 'M9_RETRACE_THRESHOLD') + 0.10,
                'confirm_bars': confirm_default,
            },
            'CHOP_MILD': {
                'chop_score_enter': _cfg(config, 'M9_CHOP_MILD_CHOP_SCORE'),
                'chop_score_exit': _cfg(config, 'M9_CHOP_MILD_CHOP_SCORE') - 0.10,
                'whipsaw_exit': _cfg(config, 'M9_WHIPSAW_THRESHOLD'),
                'retrace_exit': _cfg(config, 'M9_RETRACE_THRESHOLD'),
                'confirm_bars': confirm_default,
            },
        }

        self.TRANSITION_COOLDOWN = {
            'CRISIS': _cfg(config, 'M9_CRISIS_COOLDOWN'),
            'TRENDING': _cfg(config, 'M9_TRENDING_COOLDOWN'),
            'COMPRESSING': _cfg(config, 'M9_COMPRESSING_COOLDOWN'),
            'CHOP_HARD': _cfg(config, 'M9_CHOP_HARD_COOLDOWN'),
            'CHOP_MILD': _cfg(config, 'M9_CHOP_MILD_COOLDOWN'),
            'CHOP_MILD_BEAR': _cfg(config, 'M9_CHOP_MILD_COOLDOWN'),
            'CHOP_MILD_BULL': _cfg(config, 'M9_CHOP_MILD_COOLDOWN'),
            'NEUTRAL': 2,
            'NEUTRAL_TRENDING': 2,
            'NEUTRAL_TRENDING_BULL': 2,
            'NEUTRAL_TRENDING_BEAR': 2,
            'NEUTRAL_CHOP': 2,
        }

        self.prev_regime = 'NEUTRAL'
        self.candidate_regime = None
        self.candidate_count = 0
        self.cooldown_remaining = 0
        self.transition_log = []
        self.regime_bar_count = 0
        self.consecutive_chop_bars = 0
        self._chop_max_bars = _cfg(config, 'M9_CHOP_MAX_BARS')
        self._chop_exit_cooldown = _cfg(config, 'M9_CHOP_EXIT_COOLDOWN')

    @staticmethod
    def _chop_family(regime):
        """Return True if regime is any CHOP variant."""
        return regime in ('CHOP_MILD', 'CHOP_MILD_BEAR', 'CHOP_MILD_BULL', 'CHOP_HARD')

    @staticmethod
    def _chop_base(regime):
        """Return CHOP_MILD or CHOP_HARD family name."""
        if regime in ('CHOP_MILD', 'CHOP_MILD_BEAR', 'CHOP_MILD_BULL'):
            return 'CHOP_MILD'
        if regime == 'CHOP_HARD':
            return 'CHOP_HARD'
        return regime

    def update(self, raw_regime, signals, timestamp=None):
        details = {}
        self.regime_bar_count += 1

        # Track time in chop regimes
        if self._chop_family(self.prev_regime):
            self.consecutive_chop_bars += 1
        else:
            self.consecutive_chop_bars = 0

        if self.cooldown_remaining > 0:
            self.cooldown_remaining -= 1
            details['cooldown_remaining'] = self.cooldown_remaining
            return self.prev_regime, False, details

        # ── Directional transition within CHOP_MILD family ──
        # Both CHOP_MILD_BEAR and CHOP_MILD_BULL share hysteresis.
        # Allow instant flip between them (no confirmation delay).
        if (self._chop_base(self.prev_regime) == 'CHOP_MILD' and
            self._chop_base(raw_regime) == 'CHOP_MILD' and
            raw_regime != self.prev_regime and
            raw_regime in ('CHOP_MILD_BEAR', 'CHOP_MILD_BULL') and
            self.prev_regime in ('CHOP_MILD_BEAR', 'CHOP_MILD_BULL')):
            old = self.prev_regime
            self.prev_regime = raw_regime
            self.regime_bar_count = 0
            entry = {
                'timestamp': timestamp, 'from': old, 'to': raw_regime,
                'reason': f'CHOP direction flip: {old} → {raw_regime}',
                'signals': {k: round(v, 3) if isinstance(v, float) else v for k, v in signals.items()},
            }
            self.transition_log.append(entry)
            details['transition'] = entry
            return raw_regime, True, details

        # ── Instant flip between NEUTRAL sub-types ──
        _neutral_family = ('NEUTRAL', 'NEUTRAL_TRENDING', 'NEUTRAL_TRENDING_BULL',
                           'NEUTRAL_TRENDING_BEAR', 'NEUTRAL_CHOP')
        if (self.prev_regime in _neutral_family and raw_regime in _neutral_family and
                raw_regime != self.prev_regime):
            old = self.prev_regime
            self.prev_regime = raw_regime
            self.regime_bar_count = 0
            self.cooldown_remaining = 0
            entry = {
                'timestamp': timestamp, 'from': old, 'to': raw_regime,
                'reason': f'NEUTRAL sub-type flip: {old} → {raw_regime}',
                'signals': {k: round(v, 3) if isinstance(v, float) else v for k, v in signals.items()},
            }
            self.transition_log.append(entry)
            details['transition'] = entry
            return raw_regime, True, details

        # ── Time-based exit for chop regimes ──
        if self._chop_family(self.prev_regime) and self.consecutive_chop_bars >= self._chop_max_bars:
            old = self.prev_regime
            self.prev_regime = 'NEUTRAL'
            self.candidate_regime = None
            self.candidate_count = 0
            self.regime_bar_count = 0
            self.consecutive_chop_bars = 0
            self.cooldown_remaining = self._chop_exit_cooldown
            entry = {
                'timestamp': timestamp, 'from': old, 'to': 'NEUTRAL',
                'reason': f'Time-based exit: {self.consecutive_chop_bars} bars in {old}',
                'signals': {k: round(v, 3) if isinstance(v, float) else v for k, v in signals.items()},
            }
            self.transition_log.append(entry)
            details['transition'] = entry
            details['time_exit'] = True
            return 'NEUTRAL', True, details

        if raw_regime == self.prev_regime:
            self.candidate_regime = None
            self.candidate_count = 0
            return self.prev_regime, False, details

        should_transition = False
        transition_reason = ''

        # Exit conditions for each regime (using config-driven thresholds)
        if self.prev_regime == 'CRISIS':
            hyst = self.HYSTERESIS['CRISIS']
            if signals['atr_pctl'] < hyst['atr_pctl_exit'] and signals['bb_pctl'] < hyst['bb_pctl_exit']:
                should_transition = True
                transition_reason = "CRISIS exit: volatility subsiding"

        elif self.prev_regime == 'TRENDING':
            hyst = self.HYSTERESIS['TRENDING']
            dir_ok = signals['directionality'] < hyst['directionality_exit']
            struct_ok = signals['structure_score'] < hyst['structure_exit']
            whipsaw_bad = signals['whipsaw_rate'] > hyst['whipsaw_exit']
            retrace_bad = signals['retrace_ratio'] > hyst['retrace_exit']
            if (dir_ok and struct_ok) or whipsaw_bad or retrace_bad:
                should_transition = True
                transition_reason = f"TRENDING exit: dir={signals['directionality']:.2f} whipsaw={signals['whipsaw_rate']:.2f} retrace={signals['retrace_ratio']:.2f}"

        elif self.prev_regime == 'COMPRESSING':
            hyst = self.HYSTERESIS['COMPRESSING']
            if signals['bb_pctl'] > hyst['bb_pctl_exit'] or signals['atr_pctl'] > hyst['atr_pctl_exit']:
                should_transition = True
                transition_reason = "COMPRESSING exit: expanding"

        elif self._chop_family(self.prev_regime):
            # Both CHOP_MILD_BEAR/BULL and CHOP_HARD use same exit logic
            hyst = self.HYSTERESIS.get(self._chop_base(self.prev_regime), self.HYSTERESIS['CHOP_MILD'])
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
            self.consecutive_chop_bars = 0
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

    signs = np.sign(diffs.values)
    consec = 0
    max_consec = 0
    for i in range(1, len(signs)):
        if signs[i] == signs[i - 1] and signs[i] != 0:
            consec += 1
            max_consec = max(max_consec, consec)
        else:
            consec = 0

    directionality = min(max_consec / (window - 1), 1.0)

    net_move = abs(recent.iloc[-1] - recent.iloc[0])
    total_move = diffs.abs().sum()
    efficiency = net_move / total_move if total_move > 0 else 0.5

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

    if current_dir != 0 and move_bars >= 1:
        moves.append({
            'dir': current_dir, 'start': move_start, 'end': len(diffs),
            'bars': move_bars, 'magnitude': abs(diffs.iloc[move_start:].sum()),
        })

    if len(moves) < 3:
        return 0.5

    reversed_count = 0
    total_moves = len(moves) - 1

    for i in range(total_moves):
        move = moves[i]
        if i + 1 < len(moves):
            next_move = moves[i + 1]
            if next_move['dir'] == -move['dir']:
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

    range_size = high - low

    values = recent.values
    swings = []
    for i in range(2, len(values) - 2):
        if values[i] > values[i-1] and values[i] > values[i-2] and values[i] > values[i+1] and values[i] > values[i+2]:
            swings.append(('H', values[i], i))
        elif values[i] < values[i-1] and values[i] < values[i-2] and values[i] < values[i+1] and values[i] < values[i+2]:
            swings.append(('L', values[i], i))

    if len(swings) < 3:
        position = (current - low) / range_size
        return 1.0 - abs(position - 0.5) * 2

    retracements = []
    for i in range(len(swings) - 2):
        s1 = swings[i]
        s2 = swings[i + 1]
        s3 = swings[i + 2]

        if s1[0] == 'H' and s2[0] == 'L' and s3[0] == 'H':
            move_down = s1[1] - s2[1]
            retrace_up = s3[1] - s2[1]
            if move_down > 0:
                retracements.append(retrace_up / move_down)
        elif s1[0] == 'L' and s2[0] == 'H' and s3[0] == 'L':
            move_up = s2[1] - s1[1]
            retrace_down = s2[1] - s3[1]
            if move_up > 0:
                retracements.append(retrace_down / move_up)

    if not retracements:
        return 0.5

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

    vol_ratio = vol.iloc[-1] / avg_vol if avg_vol > 0 else 1.0

    price_change = close.iloc[-1] - close.iloc[-2] if len(close) >= 2 else 0
    direction = np.sign(price_change)

    diffs = close.diff().dropna()
    up_vol = vol.iloc[1:][diffs > 0].mean() if (diffs > 0).any() else avg_vol
    down_vol = vol.iloc[1:][diffs < 0].mean() if (diffs < 0).any() else avg_vol

    if direction > 0:
        vol_confirm = up_vol / (down_vol + 1e-10)
    elif direction < 0:
        vol_confirm = down_vol / (up_vol + 1e-10)
    else:
        vol_confirm = 1.0

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
    range_size = (recent.max() - recent.min()) / recent.mean() * 100

    if range_size < 0.3:
        return 1.0
    elif range_size < 0.8:
        return 0.7
    elif range_size < 1.5:
        return 0.4
    elif range_size < 3.0:
        return 0.2
    else:
        return 0.0


def _compute_1h_15m_coherence(df_15m, df_1h, idx_15m, idx_1h):
    """
    Do 15m and 1H agree on direction?
    Returns 0.0 (conflict) to 1.0 (strong agreement).
    """
    if idx_1h < 10 or idx_15m < 40:
        return 0.5

    close_1h = df_1h['Close'].iloc[max(0, idx_1h - 10):idx_1h + 1]
    if len(close_1h) < 5:
        return 0.5
    dir_1h = np.sign(close_1h.iloc[-1] - close_1h.iloc[0])

    close_15m = df_15m['Close'].iloc[max(0, idx_15m - 40):idx_15m + 1]
    if len(close_15m) < 10:
        return 0.5
    dir_15m = np.sign(close_15m.iloc[-1] - close_15m.iloc[0])

    if dir_1h == 0 or dir_15m == 0:
        return 0.5

    if dir_1h == dir_15m:
        return 1.0
    else:
        return 0.0


# ═══════════════════════════════════════════════════════════════
# MAIN REGIME COMPUTATION
# ═══════════════════════════════════════════════════════════════

def compute_vol_regime(df_15m, df_1h, idx_15m, idx_1h, regime_state=None, config=None):
    """
    Compute volatility regime for current bar with full signal suite.

    Args:
        df_15m: 15m OHLCV DataFrame
        df_1h: 1H OHLCV DataFrame
        idx_15m: current 15m bar index
        idx_1h: current 1H bar index
        regime_state: RegimeState instance for hysteresis
        config: config dict (settings.yaml). If None, uses built-in defaults.

    Returns: (regime, score, details)
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

    # Blend directionality: 40% 1H, 60% 15m
    directionality = directionality_1h * 0.4 + directionality_15m * 0.6

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

    # ── Regime Classification (all thresholds from config) ──
    chop_score = (
        whipsaw_rate * 0.25 +
        retrace_ratio * 0.20 +
        (1.0 - directionality) * 0.20 +
        (1.0 - volume_confirm) * 0.20 +
        (1.0 - tf_coherence) * 0.15
    )

    trend_score = (
        directionality * 0.30 +
        structure_score * 0.25 +
        (1.0 - whipsaw_rate) * 0.20 +
        (1.0 - retrace_ratio) * 0.10 +
        volume_confirm * 0.10 +
        tf_coherence * 0.05
    )

    details['chop_score'] = round(chop_score, 3)
    details['trend_score'] = round(trend_score, 3)

    # ── Compute chop direction signal (used for CHOP_MILD split) ──
    # Combines: 1H/15m coherence direction + recent price direction
    chop_split_enabled = _cfg(config, 'M9_CHOP_SPLIT_ENABLED')
    chop_direction = 0.0  # -1.0 = bearish, +1.0 = bullish
    if chop_split_enabled and idx_15m >= 20:
        # tf_coherence direction: which way do 1H and 15m agree?
        close_1h = df_1h['Close'].iloc[max(0, idx_1h - 10):idx_1h + 1]
        close_15m = df_15m['Close'].iloc[max(0, idx_15m - 20):idx_15m + 1]
        dir_1h = np.sign(close_1h.iloc[-1] - close_1h.iloc[0]) if len(close_1h) >= 2 else 0
        dir_15m = np.sign(close_15m.iloc[-1] - close_15m.iloc[0]) if len(close_15m) >= 2 else 0

        # When TFs agree, use that direction strongly
        if tf_coherence > 0.7 and dir_1h != 0 and dir_1h == dir_15m:
            chop_direction = dir_1h * 0.8
        # Otherwise use 15m recent direction (softer signal)
        elif dir_15m != 0:
            chop_direction = dir_15m * 0.5
        # Fallback: price position in recent range
        if chop_direction == 0.0 and len(close_15m) >= 20:
            recent_high = close_15m.max()
            recent_low = close_15m.min()
            if recent_high > recent_low:
                pos = (close_15m.iloc[-1] - recent_low) / (recent_high - recent_low)
                chop_direction = (pos - 0.5) * 0.6  # -0.3 to +0.3

    details['chop_direction'] = round(chop_direction, 3)

    # Read classification thresholds from config
    crisis_atr = _cfg(config, 'M9_CRISIS_ATR_THRESHOLD')
    crisis_bb = _cfg(config, 'M9_CRISIS_BB_THRESHOLD')
    chop_hard_cs = _cfg(config, 'M9_CHOP_HARD_CHOP_SCORE')
    chop_hard_ws = _cfg(config, 'M9_CHOP_HARD_WHIPSAW')
    chop_hard_rt = _cfg(config, 'M9_CHOP_HARD_RETRACE')
    chop_mild_cs = _cfg(config, 'M9_CHOP_MILD_CHOP_SCORE')
    chop_mild_tc = _cfg(config, 'M9_CHOP_MILD_TREND_CEILING')
    trend_ts = _cfg(config, 'M9_TRENDING_TREND_SCORE')
    trend_dir = _cfg(config, 'M9_TRENDING_MIN_DIR')
    trend_ws = _cfg(config, 'M9_TRENDING_MAX_WHIPSAW_SCORE')
    trend_rt = _cfg(config, 'M9_TRENDING_MAX_RETRACE_SCORE')
    comp_bb = _cfg(config, 'M9_COMPRESSING_BB')
    comp_atr = _cfg(config, 'M9_COMPRESSING_ATR')
    comp_range = _cfg(config, 'M9_COMPRESSING_RANGE')

    if atr_pctl > crisis_atr or bb_pctl > crisis_bb:
        raw_regime = 'CRISIS'
        score = 0.10

    elif chop_score > chop_hard_cs and whipsaw_rate > chop_hard_ws and retrace_ratio > chop_hard_rt:
        raw_regime = 'CHOP_HARD'
        score = 0.10

    elif chop_score > chop_mild_cs and trend_score < chop_mild_tc:
        if chop_split_enabled:
            if chop_direction < -0.1:
                raw_regime = 'CHOP_MILD_BEAR'
            elif chop_direction > 0.1:
                raw_regime = 'CHOP_MILD_BULL'
            else:
                raw_regime = 'CHOP_MILD'
        else:
            raw_regime = 'CHOP_MILD'
        score = 0.35

    elif bb_pctl < comp_bb and atr_pctl < comp_atr and range_tight > comp_range:
        raw_regime = 'COMPRESSING'
        score = 0.50

    elif trend_score > trend_ts and directionality > trend_dir and \
         whipsaw_rate < trend_ws and retrace_ratio < trend_rt:
        raw_regime = 'TRENDING'
        score = 0.80

    else:
        # ── NEUTRAL sub-classification ──
        # Split NEUTRAL into TRENDING_NEUTRAL vs CHOP_NEUTRAL based on
        # directionality and vol expansion. This helps downstream filters
        # distinguish momentum from noise in the "nothing special" zone.
        neutral_dir_threshold = _cfg(config, 'M9_NEUTRAL_DIR_THRESHOLD')
        neutral_vol_threshold = _cfg(config, 'M9_NEUTRAL_VOL_THRESHOLD')
        if (directionality > neutral_dir_threshold and
                vol_ratio > neutral_vol_threshold and
                whipsaw_rate < 0.40):
            # Directional split — same signal as CHOP_MILD
            if chop_split_enabled and chop_direction < -0.1:
                raw_regime = 'NEUTRAL_TRENDING_BEAR'
            elif chop_split_enabled and chop_direction > 0.1:
                raw_regime = 'NEUTRAL_TRENDING_BULL'
            else:
                raw_regime = 'NEUTRAL_TRENDING'
            score = 0.60
        else:
            raw_regime = 'NEUTRAL_CHOP'
            score = 0.40

    details['raw_regime'] = raw_regime

    # ── Apply hysteresis ──
    if regime_state is not None:
        ts = df_15m['Open time'].iloc[idx_15m] if 'Open time' in df_15m.columns else None
        regime, is_transition, hyst_details = regime_state.update(raw_regime, signals, timestamp=ts)
        details.update(hyst_details)
        details['is_transition'] = is_transition

        regime_scores = {
            'CRISIS': 0.10, 'CHOP_HARD': 0.10, 'CHOP_MILD': 0.35,
            'CHOP_MILD_BEAR': 0.35, 'CHOP_MILD_BULL': 0.35,
            'COMPRESSING': 0.50, 'TRENDING': 0.80,
            'NEUTRAL': 0.50, 'NEUTRAL_TRENDING': 0.60,
            'NEUTRAL_TRENDING_BULL': 0.60, 'NEUTRAL_TRENDING_BEAR': 0.60,
            'NEUTRAL_CHOP': 0.40,
            'UNKNOWN': 0.50,
        }
        score = regime_scores.get(regime, 0.50)
    else:
        regime = raw_regime

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
        base *= 0.20
    if regime in ('CHOP_MILD', 'CHOP_MILD_BEAR', 'CHOP_MILD_BULL'):
        base *= 0.55
        # Directional chop: boost aligned, penalize conflicting
        if regime == 'CHOP_MILD_BEAR' and direction == 'SHORT':
            base = min(base * 1.25, 1.0)
        elif regime == 'CHOP_MILD_BEAR' and direction == 'LONG':
            base *= 0.75
        elif regime == 'CHOP_MILD_BULL' and direction == 'LONG':
            base = min(base * 1.25, 1.0)
        elif regime == 'CHOP_MILD_BULL' and direction == 'SHORT':
            base *= 0.75
    if regime == 'CRISIS':
        base *= 0.15
    if regime == 'COMPRESSING':
        base *= 0.85
    # NEUTRAL sub-types: NEUTRAL_TRENDING has momentum, NEUTRAL_CHOP is noise
    if regime == 'NEUTRAL_TRENDING':
        base = min(base * 1.10, 1.0)  # slight boost for momentum
    if regime == 'NEUTRAL_TRENDING_BULL':
        base = min(base * 1.10, 1.0)
        if direction == 'LONG':
            base = min(base * 1.15, 1.0)  # momentum + direction aligned
        elif direction == 'SHORT':
            base *= 0.80  # momentum against you
    if regime == 'NEUTRAL_TRENDING_BEAR':
        base = min(base * 1.10, 1.0)
        if direction == 'SHORT':
            base = min(base * 1.15, 1.0)
        elif direction == 'LONG':
            base *= 0.80
    if regime == 'NEUTRAL_CHOP':
        base *= 0.85  # penalize noise

    score = max(0.0, min(1.0, base))
    status = 'PASS' if score >= 0.45 else 'FAIL'
    details['vr_score'] = round(score, 3)
    return status, score, details

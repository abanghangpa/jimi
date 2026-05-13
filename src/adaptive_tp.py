"""
Adaptive TP Engine — Dynamic exit management

Replaces static TP levels with adaptive exits based on:
  1. R:R milestone partials — close at 1:1, 1:2, trail the rest
  2. Realized volatility — widen/narrow TP as ATR changes
  3. Momentum decay — exit if price stalls
  4. Opposing signal — close if modules flip direction

Designed to be called by the engine on every bar for open trades.
"""

import numpy as np


# ═══════════════════════════════════════════════════════════════
# DEFAULTS
# ═══════════════════════════════════════════════════════════════

_ADAPTIVE_DEFAULTS = {
    # R:R milestone system
    'ADAPTIVE_TP_ENABLED': True,
    'ADAPTIVE_RR_MILESTONE_1': 1.0,      # first partial at 1:1 R:R
    'ADAPTIVE_RR_MILESTONE_2': 2.0,      # second partial at 1:2 R:R
    'ADAPTIVE_RR_CLOSE_1': 0.40,         # close 40% at milestone 1
    'ADAPTIVE_RR_CLOSE_2': 0.30,         # close 30% at milestone 2
    # remaining 30% trails to TP3 or opposing signal

    # Realized volatility adaptation
    'ADAPTIVE_VOL_ENABLED': True,
    'ADAPTIVE_VOL_LOOKBACK': 16,         # bars to recalc ATR (4h on 15m)
    'ADAPTIVE_VOL_EXPAND_MULT': 1.3,     # if ATR grows 30%+, widen TP
    'ADAPTIVE_VOL_COMPRESS_MULT': 0.7,   # if ATR shrinks 30%+, tighten TP
    'ADAPTIVE_VOL_TIGHTEN_FRAC': 0.5,    # tighten TP to 50% of original distance

    # Momentum decay
    'ADAPTIVE_MOMENTUM_ENABLED': True,
    'ADAPTIVE_MOM_BARS': 12,             # bars to check momentum (3h on 15m)
    'ADAPTIVE_MOM_MIN_MOVE': 0.001,      # min 0.1% progress required
    'ADAPTIVE_MOM_EXIT_AFTER': 24,       # exit after 24 bars (6h) with no progress

    # Opposing signal exit
    'ADAPTIVE_OPPOSING_ENABLED': True,
    'ADAPTIVE_OPPOSING_MIN_FLIP': 0.3,   # module score must flip by this much
}


def _cfg(config, key):
    if config and key in config:
        return config[key]
    return _ADAPTIVE_DEFAULTS.get(key)


# ═══════════════════════════════════════════════════════════════
# R:R MILESTONE TRACKER
# ═══════════════════════════════════════════════════════════════

class RRMilestoneTracker:
    """Tracks R:R milestones and triggers partial closes.

    Milestones are based on how far price has moved relative to SL distance.
    At 1:1 R:R, price has moved the same distance as SL → high-confidence partial.
    """

    def __init__(self, entry_price, sl, direction, config=None):
        self.entry = entry_price
        self.sl = sl
        self.direction = direction
        self.sl_dist = abs(entry_price - sl)
        self.milestone_1 = _cfg(config, 'ADAPTIVE_RR_MILESTONE_1')
        self.milestone_2 = _cfg(config, 'ADAPTIVE_RR_MILESTONE_2')
        self.close_1 = _cfg(config, 'ADAPTIVE_RR_CLOSE_1')
        self.close_2 = _cfg(config, 'ADAPTIVE_RR_CLOSE_2')
        self.hit_m1 = False
        self.hit_m2 = False
        self.best_rr = 0

    def current_rr(self, current_price):
        """Calculate current R:R achieved (how far price moved vs SL distance)."""
        if self.direction == 'LONG':
            move = current_price - self.entry
        else:
            move = self.entry - current_price
        rr = move / self.sl_dist if self.sl_dist > 0 else 0
        self.best_rr = max(self.best_rr, rr)
        return rr

    def check(self, current_price):
        """Check if any milestone was just hit.

        Returns:
            (close_frac, milestone_name) or (0, None) if no new milestone
        """
        rr = self.current_rr(current_price)

        if not self.hit_m1 and rr >= self.milestone_1:
            self.hit_m1 = True
            return self.close_1, f'RR_{self.milestone_1:.0f}'

        if not self.hit_m2 and rr >= self.milestone_2:
            self.hit_m2 = True
            return self.close_2, f'RR_{self.milestone_2:.0f}'

        return 0, None


# ═══════════════════════════════════════════════════════════════
# REALIZED VOLATILITY ADAPTER
# ═══════════════════════════════════════════════════════════════

class VolatilityAdapter:
    """Adapts TP levels based on how ATR changes during the trade.

    If volatility expands → price can move further → widen TP.
    If volatility compresses → price is stalling → tighten TP to lock in.
    """

    def __init__(self, entry_atr, tp1, tp2, tp3, direction, config=None):
        self.entry_atr = entry_atr
        self.original_tp1 = tp1
        self.original_tp2 = tp2
        self.original_tp3 = tp3
        self.direction = direction
        self.expand_mult = _cfg(config, 'ADAPTIVE_VOL_EXPAND_MULT')
        self.compress_mult = _cfg(config, 'ADAPTIVE_VOL_COMPRESS_MULT')
        self.tighten_frac = _cfg(config, 'ADAPTIVE_VOL_TIGHTEN_FRAC')
        self.lookback = _cfg(config, 'ADAPTIVE_VOL_LOOKBACK')
        self.widened = False
        self.tightened = False

    def adapt(self, df_15m, idx, entry_price):
        """Recalculate TP levels based on current realized volatility.

        Returns:
            dict with tp1, tp2, tp3 (adjusted) or None if no change
        """
        if self.entry_atr <= 0:
            return None

        # Current ATR over lookback
        if idx < self.lookback:
            return None

        highs = df_15m['High'].iloc[idx-self.lookback:idx+1].values.astype(float)
        lows = df_15m['Low'].iloc[idx-self.lookback:idx+1].values.astype(float)
        closes = df_15m['Close'].iloc[idx-self.lookback:idx+1].values.astype(float)

        # Simple realized volatility: avg true range over lookback
        tr = np.maximum(highs[1:] - lows[1:],
                       np.maximum(np.abs(highs[1:] - closes[:-1]),
                                 np.abs(lows[1:] - closes[:-1])))
        current_atr = float(np.mean(tr))

        atr_ratio = current_atr / self.entry_atr

        # Volatility expanded → widen TP
        if atr_ratio >= self.expand_mult and not self.widened:
            self.widened = True
            self.tightened = False
            expand_factor = atr_ratio  # proportional expansion
            return self._widen_tp(entry_price, expand_factor)

        # Volatility compressed → tighten TP
        if atr_ratio <= self.compress_mult and not self.tightened:
            self.tightened = True
            self.widened = False
            return self._tighten_tp(entry_price)

        return None

    def _widen_tp(self, entry, factor):
        """Widen TPs proportionally to volatility expansion."""
        if self.direction == 'LONG':
            dist1 = (self.original_tp1 - entry) * factor
            dist2 = (self.original_tp2 - entry) * factor
            dist3 = (self.original_tp3 - entry) * factor
            return {
                'tp1': entry + dist1,
                'tp2': entry + dist2,
                'tp3': entry + dist3,
                'reason': f'vol_expand {factor:.2f}x',
            }
        else:
            dist1 = (entry - self.original_tp1) * factor
            dist2 = (entry - self.original_tp2) * factor
            dist3 = (entry - self.original_tp3) * factor
            return {
                'tp1': entry - dist1,
                'tp2': entry - dist2,
                'tp3': entry - dist3,
                'reason': f'vol_expand {factor:.2f}x',
            }

    def _tighten_tp(self, entry):
        """Tighten TPs to fraction of original distance."""
        if self.direction == 'LONG':
            dist1 = (self.original_tp1 - entry) * self.tighten_frac
            dist2 = (self.original_tp2 - entry) * self.tighten_frac
            dist3 = (self.original_tp3 - entry) * self.tighten_frac
            return {
                'tp1': entry + dist1,
                'tp2': entry + dist2,
                'tp3': entry + dist3,
                'reason': f'vol_compress tighten={self.tighten_frac}',
            }
        else:
            dist1 = (entry - self.original_tp1) * self.tighten_frac
            dist2 = (entry - self.original_tp2) * self.tighten_frac
            dist3 = (entry - self.original_tp3) * self.tighten_frac
            return {
                'tp1': entry - dist1,
                'tp2': entry - dist2,
                'tp3': entry - dist3,
                'reason': f'vol_compress tighten={self.tighten_frac}',
            }


# ═══════════════════════════════════════════════════════════════
# MOMENTUM DECAY DETECTOR
# ═══════════════════════════════════════════════════════════════

class MomentumDecayDetector:
    """Detects when price has stalled — no meaningful progress toward TP.

    Tracks: net move over lookback, bar-by-bar consistency, and time elapsed.
    If price hasn't moved toward TP after N bars, the setup is stale.
    """

    def __init__(self, entry_price, direction, config=None):
        self.entry = entry_price
        self.direction = direction
        self.min_move = _cfg(config, 'ADAPTIVE_MOM_MIN_MOVE')
        self.check_bars = _cfg(config, 'ADAPTIVE_MOM_BARS')
        self.exit_after = _cfg(config, 'ADAPTIVE_MOM_EXIT_AFTER')
        self.last_progress_bar = 0  # bar index of last meaningful progress
        self.best_rr = 0  # best R:R achieved

    def update(self, current_price, sl_dist, bar_idx, best_rr):
        """Check if momentum has decayed.

        Args:
            current_price: current close
            sl_dist: SL distance (for R:R calculation)
            bar_idx: current bar index in the DataFrame
            best_rr: best R:R achieved so far

        Returns:
            (should_exit, reason) or (False, None)
        """
        self.best_rr = max(self.best_rr, best_rr)

        # Calculate recent momentum
        if self.direction == 'LONG':
            move = (current_price - self.entry) / self.entry
        else:
            move = (self.entry - current_price) / self.entry

        # Track last time price made progress
        if move > self.min_move:
            self.last_progress_bar = bar_idx

        bars_since_progress = bar_idx - self.last_progress_bar

        # Exit if no progress for too long AND we haven't reached 1:1 R:R
        if bars_since_progress >= self.exit_after and self.best_rr < 1.0:
            return True, f'momentum_decay ({bars_since_progress}bars no progress, best_rr={self.best_rr:.2f})'

        return False, None


# ═══════════════════════════════════════════════════════════════
# OPPOSING SIGNAL DETECTOR
# ═══════════════════════════════════════════════════════════════

class OpposingSignalDetector:
    """Detects when module scores have flipped against the trade direction.

    Instead of waiting for SL, exits early when the scoring engine
    signals that the original thesis has broken down.
    """

    def __init__(self, entry_scores, direction, config=None):
        """
        Args:
            entry_scores: dict of module scores at entry time
                e.g. {'m1': 0.75, 'm2': 0.71, 'm4': 0.60, 'm5': 0.89}
            direction: 'LONG' or 'SHORT'
        """
        self.entry_scores = entry_scores
        self.direction = direction
        self.min_flip = _cfg(config, 'ADAPTIVE_OPPOSING_MIN_FLIP')

    def check(self, current_scores):
        """Check if enough modules have flipped to signal an exit.

        Args:
            current_scores: dict of current module scores

        Returns:
            (should_exit, reason, flip_count) or (False, None, 0)
        """
        if not current_scores:
            return False, None, 0

        flips = 0
        total = 0
        flip_details = []

        # Key modules that matter for direction
        key_modules = ['m1', 'm2', 'm3', 'm4', 'm5', 'm9', 'm13']

        for mod in key_modules:
            entry_val = self.entry_scores.get(mod)
            curr_val = current_scores.get(mod)
            if entry_val is None or curr_val is None:
                continue

            total += 1

            # For LONG: high scores are bullish. A flip = score drops significantly.
            # For SHORT: low scores are bullish. A flip = score rises significantly.
            if self.direction == 'LONG':
                # Entry was bullish (high score). Flip if score drops.
                if entry_val > 0.5 and curr_val < entry_val - self.min_flip:
                    flips += 1
                    flip_details.append(f'{mod}: {entry_val:.2f}→{curr_val:.2f}')
            else:
                # Entry was bearish (low score). Flip if score rises.
                if entry_val < 0.5 and curr_val > entry_val + self.min_flip:
                    flips += 1
                    flip_details.append(f'{mod}: {entry_val:.2f}→{curr_val:.2f}')

        # Exit if majority of key modules have flipped
        if total > 0 and flips >= max(2, total * 0.5):
            reason = f'opposing_signal ({flips}/{total} modules flipped: {", ".join(flip_details)})'
            return True, reason, flips

        return False, None, flips


# ═══════════════════════════════════════════════════════════════
# ADAPTIVE TP MANAGER — Orchestrates all exit strategies
# ═══════════════════════════════════════════════════════════════

class AdaptiveTPManager:
    """Manages all adaptive exit strategies for a single trade.

    Called every bar by the engine. Checks all exit conditions
    and returns exit decisions.
    """

    def __init__(self, trade, df_15m, entry_idx, config=None):
        self.trade = trade
        self.config = config
        self.enabled = _cfg(config, 'ADAPTIVE_TP_ENABLED')

        if not self.enabled:
            self.rr_tracker = None
            self.vol_adapter = None
            self.momentum = None
            self.opposing = None
            return

        # R:R milestone tracker
        self.rr_tracker = RRMilestoneTracker(
            trade.entry_price, trade.sl, trade.direction, config)

        # Volatility adapter
        entry_atr = float(df_15m['atr'].iloc[entry_idx]) if 'atr' in df_15m.columns else 0
        self.vol_adapter = VolatilityAdapter(
            entry_atr, trade.tp1, trade.tp2, trade.tp3, trade.direction, config) \
            if _cfg(config, 'ADAPTIVE_VOL_ENABLED') and entry_atr > 0 else None

        # Momentum decay
        self.momentum = MomentumDecayDetector(
            trade.entry_price, trade.direction, config) \
            if _cfg(config, 'ADAPTIVE_MOMENTUM_ENABLED') else None

        # Opposing signal (populated lazily)
        self.opposing = None  # set when entry scores are available

    def set_entry_scores(self, entry_scores):
        """Set entry module scores for opposing signal detection."""
        if _cfg(self.config, 'ADAPTIVE_OPPOSING_ENABLED'):
            self.opposing = OpposingSignalDetector(
                entry_scores, self.trade.direction, self.config)

    def check_exits(self, df_15m, idx, current_scores=None):
        """Check all adaptive exit conditions for the current bar.

        Args:
            df_15m: DataFrame with OHLCV + indicators
            idx: current bar index
            current_scores: dict of current module scores (for opposing signal)

        Returns:
            list of (action, frac, reason) tuples:
                action: 'CLOSE_PARTIAL' | 'CLOSE_FULL' | 'ADJUST_TP'
                frac: fraction to close (0-1) for partial, or None
                reason: human-readable reason
        """
        if not self.enabled:
            return []

        exits = []
        trade = self.trade
        current_price = float(df_15m['Close'].iloc[idx])

        # Current R:R
        sl_dist = abs(trade.entry_price - trade.sl)
        if sl_dist <= 0:
            return []

        current_rr = self.rr_tracker.current_rr(current_price) if self.rr_tracker else 0

        # ── 1. R:R Milestone Partials ──
        if self.rr_tracker and not trade.tp1_hit:
            close_frac, milestone = self.rr_tracker.check(current_price)
            if close_frac > 0 and milestone:
                exits.append(('CLOSE_PARTIAL', close_frac, milestone))

        # ── 2. Volatility Adaptation ──
        if self.vol_adapter:
            vol_result = self.vol_adapter.adapt(df_15m, idx, trade.entry_price)
            if vol_result:
                exits.append(('ADJUST_TP', None, vol_result))

        # ── 3. Momentum Decay ──
        if self.momentum:
            best_rr = max(current_rr, self.rr_tracker.best_rr if self.rr_tracker else 0)
            should_exit, reason = self.momentum.update(
                current_price, sl_dist, idx, best_rr)
            if should_exit:
                exits.append(('CLOSE_FULL', 1.0, reason))

        # ── 4. Opposing Signal ──
        if self.opposing and current_scores:
            should_exit, reason, flip_count = self.opposing.check(current_scores)
            if should_exit:
                exits.append(('CLOSE_FULL', 1.0, reason))

        return exits


def create_adaptive_manager(trade, df_15m, entry_idx, config=None):
    """Factory function to create an AdaptiveTPManager for a trade.

    Returns None if adaptive TP is disabled.
    """
    if not _cfg(config, 'ADAPTIVE_TP_ENABLED'):
        return None
    return AdaptiveTPManager(trade, df_15m, entry_idx, config)

"""
JIMI Framework — Dual Strategy Runner
Runs two independent strategies on the same data:
  Strategy A: Range Scalper (high WR, small wins)
  Strategy B: Momentum Rider (lower WR, big wins)

Usage:
    from src.dual_strategy import DualStrategy
    ds = DualStrategy(config)
    results = ds.run(df_15m, df_1h, df_2h, df_4h, df_1d, verbose=True)
"""

import os
import yaml
import numpy as np
import pandas as pd
from src.config import CONFIG, load_config
from src.engine import calc_ics, run_gatekeepers, check_entry_filters, get_tp_multipliers, check_sweep_gate
from src.utils.indicators import calc_ema, calc_atr
from src.modules.m9_volatility import RegimeState, compute_vol_regime, score_vol_regime
from src.modules.m13_structure import score_m13
from src.modules.direction_resolver import resolve_direction, score_targets
from src.modules.m5_liquidation import build_volume_profile, find_magnets, find_gaps, find_support_resistance
from src.modules.m20_failed_breakout import score_m20
from src.modules.m18_squeeze import detect_squeeze_v6 as detect_squeeze


class MomentumEntry:
    """Detect momentum breakouts for Strategy B."""

    def __init__(self, config=None):
        self.cfg = config or CONFIG
        self.enabled = self.cfg.get('MOM_ENABLED', True)
        self.move_min = self.cfg.get('MOM_MOVE_MIN_PCT', 0.012)
        self.vol_min = self.cfg.get('MOM_VOL_RATIO_MIN', 1.4)
        self.taker_long = self.cfg.get('MOM_TAKER_LONG', 0.55)
        self.taker_short = self.cfg.get('MOM_TAKER_SHORT', 0.45)
        self.lookback = self.cfg.get('MOM_LOOKBACK_BARS', 16)

    def check(self, df_15m, idx, direction):
        """Check if current bar qualifies as momentum entry.

        Returns: (is_momentum: bool, strength: float, reason: str)
        """
        if not self.enabled or idx < self.lookback:
            return False, 0.0, 'disabled'

        window = df_15m.iloc[idx - self.lookback:idx + 1]
        open_price = float(window['Open'].iloc[0])
        close_price = float(window['Close'].iloc[-1])

        if open_price <= 0:
            return False, 0.0, 'invalid_price'

        move_pct = (close_price - open_price) / open_price
        vol_recent = float(window['Volume'].iloc[-1])
        vol_avg = float(window['Volume'].mean())
        vol_ratio = vol_recent / vol_avg if vol_avg > 0 else 0

        # Taker ratio on the last bar
        taker_base = float(window['Taker buy base asset volume'].iloc[-1])
        total_vol = float(window['Volume'].iloc[-1])
        taker_ratio = taker_base / total_vol if total_vol > 0 else 0.5

        # Bar range expansion (ignition candle)
        bar_range = (float(df_15m['High'].iloc[idx]) - float(df_15m['Low'].iloc[idx]))
        avg_range = float(df_15m['High'].iloc[max(0, idx-19):idx+1].sub(
            df_15m['Low'].iloc[max(0, idx-19):idx+1]).mean())
        range_expansion = bar_range / avg_range if avg_range > 0 else 1.0

        strength = 0.0
        reasons = []

        if direction == 'LONG':
            if move_pct > self.move_min:
                strength += 0.3
                reasons.append(f'move +{move_pct*100:.1f}%')
            if vol_ratio > self.vol_min:
                strength += 0.3
                reasons.append(f'vol {vol_ratio:.1f}x')
            if taker_ratio > self.taker_long:
                strength += 0.2
                reasons.append(f'taker {taker_ratio:.3f}')
            if range_expansion > 1.5:
                strength += 0.2
                reasons.append(f'ignition {range_expansion:.1f}x')
            is_mom = move_pct > self.move_min and vol_ratio > self.vol_min and taker_ratio > self.taker_long

        elif direction == 'SHORT':
            if move_pct < -self.move_min:
                strength += 0.3
                reasons.append(f'move {move_pct*100:.1f}%')
            if vol_ratio > self.vol_min:
                strength += 0.3
                reasons.append(f'vol {vol_ratio:.1f}x')
            if taker_ratio < self.taker_short:
                strength += 0.2
                reasons.append(f'taker {taker_ratio:.3f}')
            if range_expansion > 1.5:
                strength += 0.2
                reasons.append(f'ignition {range_expansion:.1f}x')
            is_mom = move_pct < -self.move_min and vol_ratio > self.vol_min and taker_ratio < self.taker_short
        else:
            return False, 0.0, 'neutral'

        return is_mom, min(strength, 1.0), ', '.join(reasons) if reasons else 'no_momentum'


class PullbackTracker:
    """Track momentum ignition levels and wait for pullback entry.

    Flow:
      1. Momentum ignites → record ignition level + direction
      2. Wait for price to pull back (retrace 30-70% of ignition move)
      3. Enter on the retest with stop below/above pullback extreme
    """

    def __init__(self, config=None):
        self.cfg = config or {}
        self.enabled = self.cfg.get('MOM_PULLBACK_ENABLED', True)
        self.retrace_min = self.cfg.get('MOM_PULLBACK_RETRACE_MIN', 0.30)
        self.retrace_max = self.cfg.get('MOM_PULLBACK_RETRACE_MAX', 0.70)
        self.max_bars = self.cfg.get('MOM_PULLBACK_MAX_BARS', 12)
        self.entry_tol = self.cfg.get('MOM_PULLBACK_ENTRY_TOL', 0.002)
        self.sl_atr = self.cfg.get('MOM_PULLBACK_SL_ATR', 1.2)
        self.vol_min = self.cfg.get('MOM_PULLBACK_VOL_MIN', 0.8)
        self.retest_ratio = self.cfg.get('MOM_PULLBACK_RETEST_RATIO', 1.1)

        # Pending setups: list of dicts
        self.pending = []

    def record_ignition(self, direction, ignition_price, move_high, move_low,
                        strength, reason, bar_idx, atr):
        """Record a new momentum ignition for pullback tracking."""
        if not self.enabled:
            return

        setup = {
            'direction': direction,
            'ignition_price': ignition_price,
            'move_high': move_high,
            'move_low': move_low,
            'move_range': move_high - move_low,
            'strength': strength,
            'reason': reason,
            'bar_idx': bar_idx,
            'atr': atr,
            'bars_waiting': 0,
            'pullback_extreme': ignition_price,  # updated as pullback develops
            'retraced_pct': 0.0,
        }
        self.pending.append(setup)

    def check_pullback_entry(self, df_15m, idx, direction, vol_avg):
        """Check if any pending setup has a valid pullback entry.

        Returns: (entry_price, sl, strength, reason, setup_dict) or None
        """
        if not self.enabled or not self.pending:
            return None

        row = df_15m.iloc[idx]
        high = float(row['High'])
        low = float(row['Low'])
        close = float(row['Close'])
        vol = float(row['Volume'])
        bar_range = high - low
        avg_range = float(df_15m['High'].iloc[max(0, idx-19):idx+1].sub(
            df_15m['Low'].iloc[max(0, idx-19):idx+1]).mean())

        # Check each pending setup
        result = None
        best_strength = 0

        for setup in self.pending[:]:
            if setup['direction'] != direction:
                continue

            setup['bars_waiting'] += 1

            # Expire old setups
            if setup['bars_waiting'] > self.max_bars:
                self.pending.remove(setup)
                continue

            move_range = setup['move_range']
            if move_range <= 0:
                self.pending.remove(setup)
                continue

            # Track pullback extreme
            if direction == 'LONG':
                # Pullback is downward after bullish ignition
                if low < setup['pullback_extreme']:
                    setup['pullback_extreme'] = low
                retrace_from_high = (setup['move_high'] - low) / move_range
                setup['retraced_pct'] = retrace_from_high

                # Check if pullback is in the sweet spot
                if retrace_from_high < self.retrace_min:
                    continue  # not enough pullback yet
                if retrace_from_high > self.retrace_max:
                    self.pending.remove(setup)
                    continue  # too much pullback, momentum dead

                # Check for retest bounce: price should be moving back up
                if close <= setup['pullback_extreme'] * (1 + self.entry_tol):
                    continue  # not bouncing yet

                # Retest quality: current candle should show buying
                vol_ok = vol >= vol_avg * self.vol_min if vol_avg > 0 else True
                range_ok = bar_range >= avg_range * self.retest_ratio if avg_range > 0 else True

                if not vol_ok:
                    continue

                # Entry at current close, SL below pullback extreme
                entry = close
                sl = setup['pullback_extreme'] - self.sl_atr * setup['atr']

            else:  # SHORT
                # Pullback is upward after bearish ignition
                if high > setup['pullback_extreme']:
                    setup['pullback_extreme'] = high
                retrace_from_low = (high - setup['move_low']) / move_range
                setup['retraced_pct'] = retrace_from_low

                # Check if pullback is in the sweet spot
                if retrace_from_low < self.retrace_min:
                    continue
                if retrace_from_low > self.retrace_max:
                    self.pending.remove(setup)
                    continue

                # Check for retest rejection: price should be moving back down
                if close >= setup['pullback_extreme'] * (1 - self.entry_tol):
                    continue

                vol_ok = vol >= vol_avg * self.vol_min if vol_avg > 0 else True
                range_ok = bar_range >= avg_range * self.retest_ratio if avg_range > 0 else True

                if not vol_ok:
                    continue

                entry = close
                sl = setup['pullback_extreme'] + self.sl_atr * setup['atr']

            # Score this setup
            strength = setup['strength']
            # Bonus for clean pullback (38.2-61.8% retrace = Fibonacci sweet spot)
            retrace = setup['retraced_pct']
            if 0.382 <= retrace <= 0.618:
                strength *= 1.15  # Fibonacci bonus
            # Bonus for quick pullback (within 4 bars)
            if setup['bars_waiting'] <= 4:
                strength *= 1.10  # quick retest bonus
            # Penalty for slow pullback
            if setup['bars_waiting'] > 8:
                strength *= 0.90

            strength = min(strength, 1.0)

            if strength > best_strength:
                best_strength = strength
                reason = f"pullback retrace={retrace:.1%} bars={setup['bars_waiting']} {setup['reason']}"
                result = (entry, sl, strength, reason, setup)

        return result


class DualStrategy:
    """Run Strategy A (scalp) and Strategy B (momentum) in parallel."""

    def __init__(self, config=None):
        self.cfg = config or CONFIG

        # Load momentum overrides
        mom_config_path = os.path.join(
            os.path.dirname(os.path.dirname(__file__)),
            'config', 'strategy_momentum.yaml')
        self.mom_cfg = dict(self.cfg)
        if os.path.exists(mom_config_path):
            with open(mom_config_path) as f:
                mom_overrides = yaml.safe_load(f) or {}
                self.mom_cfg.update(mom_overrides)

        self.momentum = MomentumEntry(self.mom_cfg)
        self.pullback = PullbackTracker(self.mom_cfg)
        self.regime_state = RegimeState(config=self.cfg)

    def _find_tf_idx(self, ts, df_tf):
        """Find the appropriate index in a higher timeframe DataFrame."""
        if '_ts' not in df_tf.columns:
            df_tf = df_tf.copy()
            df_tf['_ts'] = df_tf['Open time'].values.astype('datetime64[ns]')
        idx = df_tf['_ts'].searchsorted(ts, side='right') - 1
        return max(idx, -1)

    def _compute_targets(self, df_15m, idx, cfg):
        """Compute volume profile targets (cached)."""
        highs = df_15m['High'].values[:idx+1].astype(float)
        lows = df_15m['Low'].values[:idx+1].astype(float)
        closes = df_15m['Close'].values[:idx+1].astype(float)
        volumes = df_15m['Volume'].values[:idx+1].astype(float)
        bc, vp, be = build_volume_profile(
            highs, lows, closes, volumes,
            n_bins=cfg.get('M5_VP_BINS', 50),
            lookback=cfg.get('M5_VP_LOOKBACK', 672))
        magnets = find_magnets(bc, vp) if bc is not None else []
        gaps = find_gaps(bc, vp) if bc is not None else []
        sr = find_support_resistance(df_15m, idx)
        return magnets, gaps, sr

    def scan(self, df_15m, df_1h, df_2h, df_4h, df_1d, config=None, current_idx=None):
        """Run both strategies on the latest bar. Returns dict with results.

        For backtest, call this per bar with current_idx set.
        For live scan, call once (uses last bar).
        """
        cfg = config or self.cfg
        idx = current_idx if current_idx is not None else len(df_15m) - 1
        row = df_15m.iloc[idx]
        ts = row['Open time']
        price = float(row['Close'])

        # For higher timeframes, find the appropriate index
        if current_idx is not None:
            # Backtest: find TF indices by timestamp
            idx_1h = self._find_tf_idx(ts, df_1h)
            idx_2h = self._find_tf_idx(ts, df_2h)
            idx_4h = self._find_tf_idx(ts, df_4h)
            idx_1d = self._find_tf_idx(ts, df_1d)
        else:
            idx_1h = len(df_1h) - 1
            idx_2h = len(df_2h) - 1
            idx_4h = len(df_4h) - 1
            idx_1d = len(df_1d) - 1

        if idx_1h < 1 or idx_1d < 0:
            return {'strategy_a': None, 'strategy_b': None}

        atr_1h = float(df_1h['atr'].iloc[idx_1h])
        swing_bias = df_1d['swing_bias'].iloc[idx_1d]
        phase0_val = df_1d['phase0'].iloc[idx_1d]
        trend_dir = df_1d['trend'].iloc[idx_1d]

        # ── Regime ──
        vol_regime, m9_raw, m9_details = compute_vol_regime(
            df_15m, df_1h, idx, idx_1h,
            regime_state=self.regime_state, config=cfg)

        # ── Structure ──
        m13_status, m13_score, m13_details = score_m13(
            df_1h, idx_1h, 'NEUTRAL', df_15m, idx)

        # ── Direction ──
        magnets, gaps, sr = self._compute_targets(df_15m, idx, cfg)
        long_tgt, long_det = score_targets(price, magnets, gaps, sr, 'LONG', atr_1h=atr_1h)
        short_tgt, short_det = score_targets(price, magnets, gaps, sr, 'SHORT', atr_1h=atr_1h)

        m13_bias = m13_details.get('m13_bias', 'NEUTRAL')
        direction, dir_mult, dir_details = resolve_direction(
            vol_regime, m9_raw, m13_bias, m13_score, m13_details,
            swing_bias_1d=swing_bias, trend_dir=trend_dir, config=cfg,
            long_target_score=long_tgt, short_target_score=short_tgt,
            long_target_details=long_det, short_target_details=short_det,
        )

        # Re-score with actual direction
        m9_status, m9_score, _ = score_vol_regime(vol_regime, m9_raw, direction, trend_dir)

        result_base = {
            'timestamp': str(ts), 'price': price,
            'regime': vol_regime, 'direction': direction,
            'swing_bias': swing_bias, 'trend_dir': trend_dir,
            'm13_bias': m13_bias,
        }

        # ── Strategy A: Range Scalper ──
        strategy_a = self._run_scalp(
            df_15m, df_1h, df_2h, df_4h, df_1d, idx, idx_1h, idx_2h, idx_4h, idx_1d,
            direction, vol_regime, m9_score, m9_status, m13_bias, m13_score, m13_details,
            magnets, gaps, sr, atr_1h, swing_bias, phase0_val, trend_dir, cfg)

        # ── Strategy B: Momentum Rider ──
        strategy_b = self._run_momentum(
            df_15m, df_1h, df_2h, df_4h, df_1d, idx, idx_1h, idx_2h, idx_4h, idx_1d,
            direction, vol_regime, m9_score, m9_status, m13_bias, m13_score, m13_details,
            magnets, gaps, sr, atr_1h, swing_bias, phase0_val, trend_dir, price)

        return {
            'base': result_base,
            'strategy_a': strategy_a,
            'strategy_b': strategy_b,
        }

    def _run_scalp(self, df_15m, df_1h, df_2h, df_4h, df_1d,
                   idx, idx_1h, idx_2h, idx_4h, idx_1d,
                   direction, vol_regime, m9_score, m9_status,
                   m13_bias, m13_score, m13_details,
                   magnets, gaps, sr, atr_1h, swing_bias, phase0_val, trend_dir, cfg):
        """Strategy A: Range Scalper — high WR, small wins."""
        from src.modules.m1_macd_v2 import score_m1_v2 as score_m1
        from src.modules.m2_ema import score_m2
        from src.modules.m3_vwap import score_m3
        from src.modules.m4_cvd import score_m4
        from src.modules.m5_liquidation import score_m5
        from src.modules.m14_sweep import score_m14
        from src.modules.coherence_liquidity import check_coherence

        row = df_15m.iloc[idx]
        price = float(row['Close'])

        if direction == 'NEUTRAL':
            return {'status': 'NO_SIGNAL', 'reason': 'neutral direction'}

        # Score modules
        m1_dir, m1_score, _ = score_m1(df_1h, idx_1h, cfg, df_15m=df_15m, idx_15m=idx)
        if m1_dir == 'BEARISH' and direction == 'LONG':
            m1_score = 1.0 - m1_score
        elif m1_dir == 'BULLISH' and direction == 'SHORT':
            m1_score = 1.0 - m1_score

        m2_status, m2_score = score_m2(df_1h, df_2h, df_4h, df_1d, idx_1h, idx_2h, idx_4h, idx_1d)
        m3_status, m3_score, _ = score_m3(df_15m, idx, direction, cfg)
        m4_status, m4_score, m4_div = score_m4(df_15m, df_2h, idx, idx_2h, direction, cfg)
        m5_status, m5_score, m5_details = score_m5(df_15m, idx, direction, cfg)

        # Entry filters
        passed, reason = check_entry_filters(df_15m, idx, direction, swing_bias, phase0_val, atr_1h, config=cfg)
        if not passed:
            return {'status': 'FILTERED', 'reason': reason, 'mode': 'scalp'}

        # ICS
        ics, floor = calc_ics(m1_score, m2_score, m3_score, m4_score, m4_status, m5_score, config=cfg)
        threshold = cfg['ICS_THRESHOLD_CAUTION'] if phase0_val >= 0.40 else cfg['ICS_THRESHOLD_NORMAL']

        if ics < floor or ics < threshold:
            return {'status': 'NO_SIGNAL', 'reason': f'ICS {ics:.3f} < {threshold:.2f}', 'mode': 'scalp'}

        # TP/SL
        from src.sl_tp import calc_trade_levels
        levels = calc_trade_levels(price, direction, atr_1h, row.get('vol_ratio', np.nan),
                                    magnets=magnets, sr_levels=sr, liq_levels=None, cfg=cfg)

        return {
            'status': 'SIGNAL', 'mode': 'scalp',
            'direction': direction, 'entry': price,
            'sl': levels['sl'], 'tp1': levels['tp1'],
            'tp2': levels['tp2'], 'tp3': levels['tp3'],
            'sl_pct': levels['sl_pct'], 'tp1_pct': levels['tp1_pct'],
            'ics': round(ics, 4), 'regime': vol_regime,
            'size': cfg['SIZE_STD'],
        }

    def _run_momentum(self, df_15m, df_1h, df_2h, df_4h, df_1d,
                      idx, idx_1h, idx_2h, idx_4h, idx_1d,
                      direction, vol_regime, m9_score, m9_status,
                      m13_bias, m13_score, m13_details,
                      magnets, gaps, sr, atr_1h, swing_bias, phase0_val, trend_dir, price):
        """Strategy B: Momentum Rider — lower WR, big wins.

        Supports two entry modes:
          ignition: enter on the ignition candle (old behavior)
          pullback: wait for pullback, enter on retest (new, default)
        """
        mom_cfg = self.mom_cfg
        pullback_mode = mom_cfg.get('MOM_PULLBACK_MODE', 'pullback')

        if direction == 'NEUTRAL':
            return {'status': 'NO_SIGNAL', 'reason': 'neutral direction', 'mode': 'momentum'}

        # Check regime
        allowed = mom_cfg.get('MOM_ALLOWED_REGIMES', ['NEUTRAL', 'TRENDING', 'COMPRESSING'])
        if vol_regime not in allowed:
            return {'status': 'NO_SIGNAL', 'reason': f'regime {vol_regime} not allowed', 'mode': 'momentum'}

        # ── Check for pending pullback entry first ──
        if pullback_mode == 'pullback' and self.pullback.enabled:
            vol_avg = float(df_15m['Volume'].iloc[max(0, idx-19):idx+1].mean())
            pullback_result = self.pullback.check_pullback_entry(
                df_15m, idx, direction, vol_avg)

            if pullback_result is not None:
                entry, sl, strength, reason, setup = pullback_result

                # Compute TPs using pullback-specific ATR multiples
                atr_for_sl = atr_1h if not pd.isna(atr_1h) else float(df_15m['atr'].iloc[idx])
                tp1_mult = mom_cfg.get('MOM_TP1_ATR', 2.5)
                tp2_mult = mom_cfg.get('MOM_TP2_ATR', 4.0)
                tp3_mult = mom_cfg.get('MOM_TP3_ATR', 6.0)

                if direction == 'LONG':
                    tp1 = entry + tp1_mult * atr_for_sl
                    tp2 = entry + tp2_mult * atr_for_sl
                    tp3 = entry + tp3_mult * atr_for_sl
                else:
                    tp1 = entry - tp1_mult * atr_for_sl
                    tp2 = entry - tp2_mult * atr_for_sl
                    tp3 = entry - tp3_mult * atr_for_sl

                sl_pct = abs(entry - sl) / entry * 100
                tp1_pct = abs(tp1 - entry) / entry * 100

                # ICS with momentum boost
                ics = self._compute_momentum_ics(
                    df_15m, df_1h, df_2h, df_4h, df_1d,
                    idx, idx_1h, idx_2h, idx_4h, idx_1d, direction, mom_cfg)
                ics += strength * 0.10

                mom_floor = mom_cfg.get('MOM_ICS_FLOOR', 0.40)
                mom_threshold = mom_cfg.get('MOM_ICS_THRESHOLD', 0.45)

                if ics < mom_floor or ics < mom_threshold:
                    # Don't remove setup — let it retry on next bar
                    return {'status': 'NO_SIGNAL', 'reason': f'pullback ICS {ics:.3f} < {mom_threshold:.2f}', 'mode': 'momentum'}

                # ICS passed — remove the used setup
                if setup in self.pullback.pending:
                    self.pullback.pending.remove(setup)

                return {
                    'status': 'SIGNAL', 'mode': 'momentum_pullback',
                    'direction': direction, 'entry': round(entry, 2),
                    'sl': round(sl, 2), 'tp1': round(tp1, 2),
                    'tp2': round(tp2, 2), 'tp3': round(tp3, 2),
                    'sl_pct': round(sl_pct, 3), 'tp1_pct': round(tp1_pct, 3),
                    'ics': round(ics, 4), 'regime': vol_regime,
                    'momentum_strength': round(strength, 3),
                    'momentum_reason': reason,
                    'size': mom_cfg.get('MOM_SIZE_STD', 3.0),
                    'tp1_close': mom_cfg.get('MOM_TP1_CLOSE', 0.15),
                    'pullback_retrace': round(setup.get('retraced_pct', 0), 3),
                    'pullback_bars': setup.get('bars_waiting', 0),
                }

        # ── Check for new momentum ignition ──
        is_mom, strength, reason = self.momentum.check(df_15m, idx, direction)
        if not is_mom:
            return {'status': 'NO_SIGNAL', 'reason': f'no momentum ({reason})', 'mode': 'momentum'}

        # Ignition detected — record for pullback tracking
        if pullback_mode == 'pullback' and self.pullback.enabled:
            window = df_15m.iloc[idx - self.momentum.lookback:idx + 1]
            move_high = float(window['High'].max())
            move_low = float(window['Low'].min())
            atr_val = float(df_15m['atr'].iloc[idx]) if not pd.isna(df_15m['atr'].iloc[idx]) else atr_1h

            self.pullback.record_ignition(
                direction, price, move_high, move_low,
                strength, reason, idx, atr_val)

            return {'status': 'NO_SIGNAL', 'reason': f'ignition recorded, waiting for pullback ({reason})',
                    'mode': 'momentum_pullback_pending', 'strength': strength}

        # ── Ignition mode: enter immediately (original behavior) ──
        # ICS with lower threshold (momentum is self-confirming)
        ics = self._compute_momentum_ics(
            df_15m, df_1h, df_2h, df_4h, df_1d,
            idx, idx_1h, idx_2h, idx_4h, idx_1d, direction, mom_cfg)
        ics += strength * 0.10

        mom_floor = mom_cfg.get('MOM_ICS_FLOOR', 0.40)
        mom_threshold = mom_cfg.get('MOM_ICS_THRESHOLD', 0.45)

        if ics < mom_floor or ics < mom_threshold:
            return {'status': 'NO_SIGNAL', 'reason': f'ICS {ics:.3f} < {mom_threshold:.2f}', 'mode': 'momentum'}

        # TP/SL — wider for momentum
        atr_for_sl = atr_1h if not pd.isna(atr_1h) else float(df_15m['atr'].iloc[idx])
        sl_mult = mom_cfg.get('MOM_SL_ATR', 1.8)
        tp1_mult = mom_cfg.get('MOM_TP1_ATR', 2.5)
        tp2_mult = mom_cfg.get('MOM_TP2_ATR', 4.0)
        tp3_mult = mom_cfg.get('MOM_TP3_ATR', 6.0)

        if direction == 'LONG':
            sl = price - sl_mult * atr_for_sl
            tp1 = price + tp1_mult * atr_for_sl
            tp2 = price + tp2_mult * atr_for_sl
            tp3 = price + tp3_mult * atr_for_sl
        else:
            sl = price + sl_mult * atr_for_sl
            tp1 = price - tp1_mult * atr_for_sl
            tp2 = price - tp2_mult * atr_for_sl
            tp3 = price - tp3_mult * atr_for_sl

        sl_pct = abs(price - sl) / price * 100
        tp1_pct = abs(tp1 - price) / price * 100

        return {
            'status': 'SIGNAL', 'mode': 'momentum',
            'direction': direction, 'entry': price,
            'sl': round(sl, 2), 'tp1': round(tp1, 2),
            'tp2': round(tp2, 2), 'tp3': round(tp3, 2),
            'sl_pct': round(sl_pct, 3), 'tp1_pct': round(tp1_pct, 3),
            'ics': round(ics, 4), 'regime': vol_regime,
            'momentum_strength': round(strength, 3),
            'momentum_reason': reason,
            'size': mom_cfg.get('MOM_SIZE_STD', 3.0),
            'tp1_close': mom_cfg.get('MOM_TP1_CLOSE', 0.15),
        }

    def _compute_momentum_ics(self, df_15m, df_1h, df_2h, df_4h, df_1d,
                              idx, idx_1h, idx_2h, idx_4h, idx_1d, direction, mom_cfg):
        """Compute ICS for momentum entry (shared between ignition and pullback)."""
        from src.modules.m1_macd_v2 import score_m1_v2 as score_m1
        from src.modules.m2_ema import score_m2
        from src.modules.m3_vwap import score_m3
        from src.modules.m4_cvd import score_m4
        from src.modules.m5_liquidation import score_m5

        m1_dir, m1_score, _ = score_m1(df_1h, idx_1h, mom_cfg, df_15m=df_15m, idx_15m=idx)
        if m1_dir == 'BEARISH' and direction == 'LONG':
            m1_score = 1.0 - m1_score
        elif m1_dir == 'BULLISH' and direction == 'SHORT':
            m1_score = 1.0 - m1_score

        m2_status, m2_score = score_m2(df_1h, df_2h, df_4h, df_1d, idx_1h, idx_2h, idx_4h, idx_1d)
        m3_status, m3_score, _ = score_m3(df_15m, idx, direction, mom_cfg)
        m4_status, m4_score, m4_div = score_m4(df_15m, df_2h, idx, idx_2h, direction, mom_cfg)
        m5_status, m5_score, m5_details = score_m5(df_15m, idx, direction, mom_cfg)

        ics, floor = calc_ics(m1_score, m2_score, m3_score, m4_score, m4_status, m5_score, config=mom_cfg)
        return ics

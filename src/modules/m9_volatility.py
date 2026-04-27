"""M9: Volatility Regime Classifier with Hysteresis."""

import pandas as pd
import numpy as np


class RegimeState:
    """Tracks regime history with hysteresis to prevent flickering."""

    HYSTERESIS = {
        'CRISIS': {
            'atr_pctl_enter': 0.90, 'atr_pctl_exit': 0.80,
            'bb_pctl_enter': 0.90, 'bb_pctl_exit': 0.80,
            'confirm_bars': 2,
        },
        'TRENDING': {
            'directionality_enter': 0.40, 'directionality_exit': 0.30,
            'structure_enter': 0.30, 'structure_exit': 0.20,
            'confirm_bars': 2,
        },
        'COMPRESSING': {
            'bb_pctl_enter': 0.25, 'bb_pctl_exit': 0.35,
            'atr_pctl_enter': 0.35, 'atr_pctl_exit': 0.45,
            'confirm_bars': 3,
        },
        'CHOP': {
            'directionality_enter': 0.20, 'directionality_exit': 0.30,
            'bb_pctl_enter': 0.40, 'bb_pctl_exit': 0.30,
            'confirm_bars': 2,
        },
    }

    TRANSITION_COOLDOWN = {
        'CRISIS': 8, 'TRENDING': 4, 'COMPRESSING': 6, 'CHOP': 4,
    }

    def __init__(self):
        self.prev_regime = 'NEUTRAL'
        self.candidate_regime = None
        self.candidate_count = 0
        self.cooldown_remaining = 0
        self.transition_log = []

    def update(self, raw_regime, atr_pctl, bb_pctl, directionality, structure_score, timestamp=None):
        details = {}

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

        if self.prev_regime == 'CRISIS':
            hyst = self.HYSTERESIS['CRISIS']
            if atr_pctl < hyst['atr_pctl_exit'] and bb_pctl < hyst['bb_pctl_exit']:
                should_transition = True
                transition_reason = f"CRISIS exit: ATR {atr_pctl:.3f}<{hyst['atr_pctl_exit']}, BB {bb_pctl:.3f}<{hyst['bb_pctl_exit']}"
        elif self.prev_regime == 'TRENDING':
            hyst = self.HYSTERESIS['TRENDING']
            if directionality < hyst['directionality_exit'] and structure_score < hyst['structure_exit']:
                should_transition = True
                transition_reason = f"TRENDING exit"
        elif self.prev_regime == 'COMPRESSING':
            hyst = self.HYSTERESIS['COMPRESSING']
            if bb_pctl > hyst['bb_pctl_exit'] or atr_pctl > hyst['atr_pctl_exit']:
                should_transition = True
                transition_reason = f"COMPRESSING exit"
        elif self.prev_regime == 'CHOP':
            hyst = self.HYSTERESIS['CHOP']
            if directionality > hyst['directionality_exit'] or bb_pctl < hyst['bb_pctl_exit']:
                should_transition = True
                transition_reason = f"CHOP exit"
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
            self.cooldown_remaining = self.TRANSITION_COOLDOWN.get(raw_regime, 4)

            entry = {
                'timestamp': timestamp, 'from': old_regime, 'to': raw_regime,
                'reason': transition_reason, 'atr_pctl': atr_pctl, 'bb_pctl': bb_pctl,
                'directionality': directionality, 'bars_confirmed': self.candidate_count,
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


def compute_vol_regime(df_15m, df_1h, idx_15m, idx_1h, regime_state=None):
    """Compute volatility regime for current bar with hysteresis."""
    details = {}

    if idx_1h < 20:
        return 'UNKNOWN', 0.5, details

    close_1h = df_1h['Close'].iloc[max(0, idx_1h-60):idx_1h+1]
    high_1h = df_1h['High'].iloc[max(0, idx_1h-60):idx_1h+1]
    low_1h = df_1h['Low'].iloc[max(0, idx_1h-60):idx_1h+1]

    if len(close_1h) < 20:
        return 'UNKNOWN', 0.5, details

    bb_sma = close_1h.rolling(20).mean()
    bb_std = close_1h.rolling(20).std()
    bb_width = (2 * bb_std / bb_sma)
    bb_width_current = bb_width.iloc[-1] if not pd.isna(bb_width.iloc[-1]) else 0.02

    bb_width_series = bb_width.dropna()
    if len(bb_width_series) >= 20:
        bb_pctl = (bb_width_series.iloc[-1] - bb_width_series.min()) / (bb_width_series.max() - bb_width_series.min() + 1e-10)
    else:
        bb_pctl = 0.5

    details['bb_width'] = round(bb_width_current, 5)
    details['bb_pctl'] = round(bb_pctl, 3)

    if 'atr' in df_1h.columns:
        atr_1h = df_1h['atr'].iloc[idx_1h]
        atr_series = df_1h['atr'].iloc[max(0, idx_1h-180):idx_1h+1].dropna()
        if len(atr_series) >= 20 and atr_1h > 0:
            atr_pctl = (atr_1h - atr_series.min()) / (atr_series.max() - atr_series.min() + 1e-10)
        else:
            atr_pctl = 0.5
    else:
        tr1 = high_1h - low_1h
        tr2 = (high_1h - close_1h.shift(1)).abs()
        tr3 = (low_1h - close_1h.shift(1)).abs()
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        atr_1h_series = tr.ewm(span=14, adjust=False).mean()
        atr_1h = atr_1h_series.iloc[-1] if not pd.isna(atr_1h_series.iloc[-1]) else 0
        atr_pctl = 0.5

    details['atr_pctl'] = round(atr_pctl, 3)

    recent_20 = close_1h.iloc[-20:]
    directional_bars = ((recent_20.diff() > 0).sum() + (recent_20.diff() < 0).sum())
    directionality = abs(directional_bars - 10) / 10
    details['directionality'] = round(directionality, 3)

    if 'Volume' in df_1h.columns:
        vol_ma20 = df_1h['Volume'].iloc[max(0, idx_1h-20):idx_1h+1].mean()
        vol_current = df_1h['Volume'].iloc[idx_1h]
        vol_ratio = vol_current / vol_ma20 if vol_ma20 > 0 else 1.0
    else:
        vol_ratio = 1.0
    details['vol_ratio'] = round(vol_ratio, 3)

    if idx_1h >= 10:
        highs = high_1h.iloc[-10:].values
        lows = low_1h.iloc[-10:].values
        hh_count = sum(1 for i in range(1, len(highs)) if highs[i] > highs[i-1])
        ll_count = sum(1 for i in range(1, len(lows)) if lows[i] < lows[i-1])
        structure_score = abs(hh_count - ll_count) / max(hh_count + ll_count, 1)
    else:
        structure_score = 0.0
    details['structure_score'] = round(structure_score, 3)

    trend_score = (
        directionality * 0.30 + structure_score * 0.25 +
        min(vol_ratio, 2.0) / 2.0 * 0.20 +
        (1.0 - bb_pctl) * 0.15 +
        (1.0 - abs(atr_pctl - 0.5) * 2) * 0.10
    )

    if atr_pctl > 0.90 or bb_pctl > 0.90:
        raw_regime = 'CRISIS'
        score = 0.15
    elif bb_pctl < 0.25 and atr_pctl < 0.35:
        raw_regime = 'COMPRESSING'
        score = 0.40
    elif directionality > 0.4 and structure_score > 0.3 and vol_ratio > 0.8:
        raw_regime = 'TRENDING'
        score = 0.80
    elif directionality < 0.2 and bb_pctl > 0.4:
        raw_regime = 'CHOP'
        score = 0.25
    else:
        raw_regime = 'NEUTRAL'
        score = 0.50

    details['raw_regime'] = raw_regime

    if regime_state is not None:
        ts = df_15m['Open time'].iloc[idx_15m] if 'Open time' in df_15m.columns else None
        regime, is_transition, hyst_details = regime_state.update(
            raw_regime, atr_pctl, bb_pctl, directionality, structure_score, timestamp=ts
        )
        details.update(hyst_details)
        details['is_transition'] = is_transition

        if regime == 'CRISIS':
            score = 0.15
        elif regime == 'COMPRESSING':
            score = 0.40
        elif regime == 'TRENDING':
            score = 0.80
        elif regime == 'CHOP':
            score = 0.25
        else:
            score = 0.50
    else:
        regime = raw_regime

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

    if regime == 'CHOP':
        base *= 0.60
    if regime == 'CRISIS':
        base *= 0.30
    if regime == 'COMPRESSING':
        base *= 0.85

    score = max(0.0, min(1.0, base))
    status = 'PASS' if score >= 0.45 else 'FAIL'
    details['vr_score'] = round(score, 3)
    return status, score, details

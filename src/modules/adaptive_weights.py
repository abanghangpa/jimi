"""Adaptive Module Weighting — per-module accuracy tracking for position sizing."""

import numpy as np


class AdaptiveWeights:
    """Track per-module accuracy and adjust position size (not ICS)."""

    MODULES = ['M1', 'M2', 'M3', 'M4', 'M5']

    def __init__(self, decay=0.97, min_mult=0.8, max_mult=1.2, warmup=15):
        self.decay = decay
        self.min_mult = min_mult
        self.max_mult = max_mult
        self.warmup = warmup
        self.scores = {m: [0.0, 0.0] for m in self.MODULES}
        self.trade_count = 0
        self.history = []

    def update(self, trade):
        """Update module scores after a trade closes."""
        won = trade.pnl_pct > 0
        self.trade_count += 1

        for m in self.MODULES:
            self.scores[m][0] *= self.decay
            self.scores[m][1] *= self.decay

        correct = {}
        m1_bullish = trade.m1_dir == 'LONG'
        correct['M1'] = 1.0 if (m1_bullish and won) or (not m1_bullish and not won) else 0.0
        correct['M2'] = 1.0 if (trade.m2_status == 'PASS') == won else 0.0
        m3_good = trade.m3_score > 0.5
        correct['M3'] = 1.0 if (m3_good and won) or (not m3_good and not won) else 0.0
        correct['M4'] = 1.0 if (trade.m4_status == 'PASS') == won else 0.0
        correct['M5'] = 1.0 if (trade.m5_status == 'PASS') == won else 0.0

        for m in self.MODULES:
            self.scores[m][0] += correct[m]
            self.scores[m][1] += 1.0

        self.history.append({'trade_idx': self.trade_count, 'won': won, 'correct': correct})

    def get_multipliers(self):
        if self.trade_count < self.warmup:
            return {m: 1.0 for m in self.MODULES}

        accuracies = {}
        for m in self.MODULES:
            total = self.scores[m][1]
            accuracies[m] = self.scores[m][0] / total if total > 0 else 0.5

        mean_acc = np.mean(list(accuracies.values()))
        if mean_acc == 0:
            return {m: 1.0 for m in self.MODULES}

        multipliers = {}
        for m in self.MODULES:
            raw_mult = accuracies[m] / mean_acc
            multipliers[m] = np.clip(raw_mult, self.min_mult, self.max_mult)
        return multipliers

    def size_multiplier(self, direction, m1_dir, m2_status, m3_score, m4_status, m5_status):
        if self.trade_count < self.warmup:
            return 1.0

        mults = self.get_multipliers()
        total_boost = 0.0
        count = 0

        m1_agrees = m1_dir == direction
        total_boost += (mults['M1'] - 1.0) if m1_agrees else -(mults['M1'] - 1.0)
        count += 1

        m2_agrees = m2_status == 'PASS'
        total_boost += (mults['M2'] - 1.0) if m2_agrees else -(mults['M2'] - 1.0)
        count += 1

        m3_agrees = m3_score > 0.5
        total_boost += (mults['M3'] - 1.0) if m3_agrees else -(mults['M3'] - 1.0)
        count += 1

        m4_agrees = m4_status == 'PASS'
        total_boost += (mults['M4'] - 1.0) if m4_agrees else -(mults['M4'] - 1.0)
        count += 1

        m5_agrees = m5_status == 'PASS'
        total_boost += (mults['M5'] - 1.0) if m5_agrees else -(mults['M5'] - 1.0)
        count += 1

        avg_boost = total_boost / count
        return np.clip(1.0 + avg_boost, 0.7, 1.3)

    def summary(self):
        if self.trade_count < self.warmup:
            return f"  Adaptive: {self.trade_count}/{self.warmup} trades (not active yet)"
        mults = self.get_multipliers()
        lines = [f"  Adaptive Weights ({self.trade_count} trades, decay={self.decay}):"]
        for m in self.MODULES:
            acc = self.scores[m][0] / self.scores[m][1] if self.scores[m][1] > 0 else 0
            lines.append(f"    {m}: accuracy={acc:.1%} → multiplier={mults[m]:.2f}")
        return "\n".join(lines)

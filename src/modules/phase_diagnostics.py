"""
JIMI Framework — Phase 1 & 2 Diagnostic Logger

Captures every bar's regime detection and direction resolution decision
for post-backtest analysis. Designed to be non-intrusive — call log()
once per bar, export to CSV after backtest.

Usage:
    from src.modules.phase_diagnostics import PhaseDiagnostics
    diag = PhaseDiagnostics()
    # ... inside backtest loop, after Phase 1+2 ...
    diag.log(bar_data)
    # ... after loop ...
    diag.export_csv("phase_diagnostics.csv")
    diag.export_summary("phase_summary.json")
"""

import pandas as pd
import numpy as np
from collections import Counter, defaultdict


class PhaseDiagnostics:
    """Captures per-bar Phase 1 (regime) and Phase 2 (direction) decisions."""

    def __init__(self):
        self.rows = []
        self._regime_transitions = []
        self._prev_regime = None

    def log(self, data: dict):
        """Log one bar's Phase 1+2 state.

        Expected keys (all optional — missing = None):
            timestamp, bar_index,
            # Phase 1: M9 Regime
            m9_raw_regime, m9_regime, m9_score, m9_signals (dict),
            m9_is_transition, m9_regime_strength,
            # Phase 2: M13 Structure
            m13_bias, m13_score, m13_swing_bias_1h, m13_swing_bias_15m,
            m13_swing_confidence, m13_fvg_count, m13_ob_count,
            m13_bull_points, m13_bear_points,
            # Phase 2: M7 Macro
            m7_score, m7_status, m7_eth_btc_trend, m7_btc_trend,
            m7_btc_atr_pctl,
            # Phase 2: Direction Resolver
            direction, dir_size_mult, dir_action, dir_reason,
            dir_m7_direction, dir_daily_swing, dir_trend_dir,
            dir_m7_penalty, dir_m7_bonus, dir_daily_penalty,
            dir_daily_bonus, dir_trend_penalty, dir_trend_bonus,
            dir_trending_structure_bonus,
            # Pre-filter blocks
            block_reason,  # None if passed, string if blocked
            block_module,  # which module blocked it
            # Context
            swing_bias_1d, trend_dir, trend_val, phase0, vol_regime_final,
        """
        row = {k: data.get(k) for k in self._schema()}
        self.rows.append(row)

        # Track regime transitions
        regime = data.get('m9_regime')
        if regime and regime != self._prev_regime:
            self._regime_transitions.append({
                'timestamp': data.get('timestamp'),
                'from': self._prev_regime,
                'to': regime,
            })
        self._prev_regime = regime

    def log_trade_outcome(self, timestamp, direction, entry_time, pnl_pct, regime_at_entry, m13_bias_at_entry):
        """Backfill trade outcome into matching diagnostic row."""
        for row in reversed(self.rows):
            if row.get('timestamp') == entry_time and row.get('direction') == direction:
                row['trade_outcome'] = 'WIN' if pnl_pct > 0 else ('LOSS' if pnl_pct < 0 else 'FLAT')
                row['trade_pnl'] = round(pnl_pct * 100, 4)
                break

    # ─── Export ────────────────────────────────────────────────

    def export_csv(self, filepath: str):
        """Export all bar diagnostics to CSV."""
        if not self.rows:
            print("  [PhaseDiagnostics] No data to export.")
            return
        df = pd.DataFrame(self.rows)
        df.to_csv(filepath, index=False)
        print(f"  [PhaseDiagnostics] Exported {len(df)} bars → {filepath}")
        return df

    def export_summary(self, filepath: str = None) -> dict:
        """Compute and optionally export aggregate statistics."""
        if not self.rows:
            return {}

        df = pd.DataFrame(self.rows)
        summary = {}

        # ── Regime Distribution ──
        if 'm9_regime' in df.columns:
            regime_counts = df['m9_regime'].value_counts().to_dict()
            total_bars = len(df)
            summary['regime_distribution'] = {
                k: {'count': v, 'pct': round(v / total_bars * 100, 2)}
                for k, v in regime_counts.items()
            }

        # ── Regime Transitions ──
        summary['regime_transitions'] = len(self._regime_transitions)
        if self._regime_transitions:
            transition_pairs = Counter(
                (t['from'], t['to']) for t in self._regime_transitions
            )
            summary['transition_pairs'] = {
                f"{k[0]}→{k[1]}": v
                for k, v in transition_pairs.most_common(20)
            }

        # ── M13 Bias Distribution ──
        if 'm13_bias' in df.columns:
            summary['m13_bias_distribution'] = df['m13_bias'].value_counts().to_dict()

        # ── Direction Distribution ──
        if 'direction' in df.columns:
            dir_counts = df['direction'].value_counts().to_dict()
            summary['direction_distribution'] = dir_counts

        # ── Block Analysis ──
        if 'block_reason' in df.columns:
            blocked = df[df['block_reason'].notna()]
            passed = df[df['block_reason'].isna()]
            summary['block_analysis'] = {
                'total_bars': len(df),
                'passed': len(passed),
                'blocked': len(blocked),
                'pass_rate': round(len(passed) / len(df) * 100, 2) if len(df) > 0 else 0,
            }
            if 'block_module' in df.columns:
                block_by_module = blocked['block_module'].value_counts().to_dict()
                summary['block_analysis']['by_module'] = block_by_module

        # ── Size Multiplier Distribution ──
        if 'dir_size_mult' in df.columns:
            sm = df['dir_size_mult'].dropna()
            if len(sm) > 0:
                summary['size_multiplier'] = {
                    'mean': round(sm.mean(), 4),
                    'median': round(sm.median(), 4),
                    'p10': round(sm.quantile(0.10), 4),
                    'p25': round(sm.quantile(0.25), 4),
                    'p75': round(sm.quantile(0.75), 4),
                    'p90': round(sm.quantile(0.90), 4),
                    'zeros': int((sm == 0).sum()),
                }

        # ── Win Rate by Regime (only for bars that became trades) ──
        if 'trade_outcome' in df.columns and 'm9_regime' in df.columns:
            traded = df[df['trade_outcome'].notna()]
            if len(traded) > 0:
                wr_by_regime = {}
                for regime in traded['m9_regime'].unique():
                    r_trades = traded[traded['m9_regime'] == regime]
                    wins = (r_trades['trade_outcome'] == 'WIN').sum()
                    total = len(r_trades)
                    wr_by_regime[regime] = {
                        'trades': total,
                        'wins': int(wins),
                        'win_rate': round(wins / total * 100, 1) if total > 0 else 0,
                        'avg_pnl': round(r_trades['trade_pnl'].mean(), 4) if 'trade_pnl' in r_trades.columns else None,
                    }
                summary['wr_by_regime'] = wr_by_regime

        # ── Win Rate by M13 Bias ──
        if 'trade_outcome' in df.columns and 'm13_bias' in df.columns:
            traded = df[df['trade_outcome'].notna()]
            if len(traded) > 0:
                wr_by_bias = {}
                for bias in traded['m13_bias'].unique():
                    b_trades = traded[traded['m13_bias'] == bias]
                    wins = (b_trades['trade_outcome'] == 'WIN').sum()
                    total = len(b_trades)
                    wr_by_bias[bias] = {
                        'trades': total,
                        'wins': int(wins),
                        'win_rate': round(wins / total * 100, 1) if total > 0 else 0,
                    }
                summary['wr_by_m13_bias'] = wr_by_bias

        # ── Win Rate by Direction ──
        if 'trade_outcome' in df.columns and 'direction' in df.columns:
            traded = df[df['trade_outcome'].notna()]
            if len(traded) > 0:
                wr_by_dir = {}
                for d in ['LONG', 'SHORT']:
                    d_trades = traded[traded['direction'] == d]
                    if len(d_trades) > 0:
                        wins = (d_trades['trade_outcome'] == 'WIN').sum()
                        wr_by_dir[d] = {
                            'trades': len(d_trades),
                            'wins': int(wins),
                            'win_rate': round(wins / len(d_trades) * 100, 1),
                        }
                summary['wr_by_direction'] = wr_by_dir

        # ── M7 Conflict/Agree Stats ──
        if 'dir_m7_direction' in df.columns:
            m7_dir = df[df['dir_m7_direction'].notna()]['dir_m7_direction']
            if len(m7_dir) > 0:
                summary['m7_direction_distribution'] = m7_dir.value_counts().to_dict()

        # ── Confirmation Layer Hit Rates ──
        confirmations = {}
        for col, name in [
            ('dir_m7_penalty', 'm7_conflict_penalty'),
            ('dir_m7_bonus', 'm7_agree_bonus'),
            ('dir_daily_penalty', 'daily_swing_conflict'),
            ('dir_daily_bonus', 'daily_swing_agree'),
            ('dir_trend_penalty', 'trend_conflict'),
            ('dir_trend_bonus', 'trend_agree'),
            ('dir_trending_structure_bonus', 'trending_structure_bonus'),
        ]:
            if col in df.columns:
                non_null = df[col].notna().sum()
                if non_null > 0:
                    confirmations[name] = int(non_null)
        if confirmations:
            summary['confirmation_layer_hits'] = confirmations

        if filepath:
            import json
            with open(filepath, 'w') as f:
                json.dump(summary, f, indent=2, default=str)
            print(f"  [PhaseDiagnostics] Summary → {filepath}")

        return summary

    # ─── Schema ────────────────────────────────────────────────

    @staticmethod
    def _schema() -> list:
        return [
            'timestamp', 'bar_index',
            # Phase 1
            'm9_raw_regime', 'm9_regime', 'm9_score', 'm9_is_transition',
            'm9_regime_strength',
            'm9_atr_pctl', 'm9_bb_pctl', 'm9_directionality', 'm9_whipsaw_rate',
            'm9_retrace_ratio', 'm9_volume_confirm', 'm9_range_tight',
            'm9_tf_coherence', 'm9_chop_score', 'm9_trend_score',
            # Phase 2: M13
            'm13_bias', 'm13_score', 'm13_swing_bias_1h', 'm13_swing_bias_15m',
            'm13_swing_confidence', 'm13_fvg_count', 'm13_ob_count',
            'm13_bull_points', 'm13_bear_points',
            # Phase 2: M7
            'm7_score', 'm7_status', 'm7_eth_btc_trend', 'm7_btc_trend',
            'm7_btc_atr_pctl',
            # Phase 2: Direction Resolver
            'direction', 'dir_size_mult', 'dir_action', 'dir_reason',
            'dir_m7_direction', 'dir_daily_swing', 'dir_trend_dir',
            'dir_m7_penalty', 'dir_m7_bonus', 'dir_daily_penalty',
            'dir_daily_bonus', 'dir_trend_penalty', 'dir_trend_bonus',
            'dir_trending_structure_bonus',
            # Block info
            'block_reason', 'block_module',
            # Context
            'swing_bias_1d', 'trend_dir', 'trend_val', 'phase0',
            # Trade outcome (backfilled)
            'trade_outcome', 'trade_pnl',
        ]

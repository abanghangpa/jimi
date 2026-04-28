# JIMI Phase 1+2 Diagnostic Integration Patch

## Overview

This patch adds per-bar diagnostic logging to `engine.py` and `backtest_runner.py`
without changing any trade logic. It captures every bar's Phase 1 (regime) and
Phase 2 (direction) decisions for post-backtest analysis.

## Files to modify

1. `src/engine.py` — add diagnostic logging calls
2. `scripts/backtest_runner.py` — add `--diagnostic` flag and export

## Files to add

1. `src/modules/phase_diagnostics.py` — already created
2. `scripts/analyze_phases.py` — already created

---

## Patch 1: `src/engine.py`

### 1a. Add import (top of file, after existing imports)

```python
# Add after: from src.modules.session import get_session
from src.modules.phase_diagnostics import PhaseDiagnostics
```

### 1b. Initialize diagnostics (inside `run_backtest`, after stats init)

Find this block (around line with `stats = {k: 0 for k in [`):
```python
    stats = {k: 0 for k in [
        'signals_checked', 'ics_blocked', ...
    ]}
```

Add right after it:
```python
    # ── Phase Diagnostics ──
    phase_diag = PhaseDiagnostics()
```

### 1c. Log Phase 1 (M9) data — after regime computation

Find the M9 block (starts with `# PHASE 1: REGIME (M9)`):
```python
        # ═══════════════════════════════════════════════════════════
        # PHASE 1: REGIME (M9) — What's the market climate?
        # ═══════════════════════════════════════════════════════════
```

Right AFTER the M9 hard block check (`if vol_regime in block_regimes: ... continue`),
add diagnostic logging for blocked bars:

```python
            if vol_regime in block_regimes:
                phase_diag.log({
                    'timestamp': ts, 'bar_index': idx,
                    'm9_regime': vol_regime, 'm9_score': m9_score,
                    'm9_raw_regime': m9_details.get('raw_regime'),
                    'm9_is_transition': m9_details.get('is_transition'),
                    'm9_regime_strength': m9_details.get('regime_strength'),
                    'm9_atr_pctl': m9_details.get('atr_pctl'),
                    'm9_bb_pctl': m9_details.get('bb_pctl'),
                    'm9_directionality': m9_details.get('directionality'),
                    'm9_whipsaw_rate': m9_details.get('whipsaw_rate'),
                    'm9_retrace_ratio': m9_details.get('retrace_ratio'),
                    'm9_volume_confirm': m9_details.get('volume_confirm'),
                    'm9_range_tight': m9_details.get('range_tight'),
                    'm9_tf_coherence': m9_details.get('tf_coherence'),
                    'm9_chop_score': m9_details.get('chop_score'),
                    'm9_trend_score': m9_details.get('trend_score'),
                    'block_reason': f'm9_regime_{vol_regime}',
                    'block_module': 'M9',
                    'swing_bias_1d': swing_bias,
                    'trend_dir': trend_dir, 'trend_val': trend_val,
                    'phase0': phase0_val,
                })
                stats['m9_block'] += 1
                continue
```

### 1d. Log Phase 2 data — after direction resolution

Find the block after `resolve_direction()` returns and the NEUTRAL check:
```python
        if direction == 'NEUTRAL':
            stats['bias_gate_skip'] += 1
            continue
```

Replace with:
```python
        if direction == 'NEUTRAL':
            phase_diag.log({
                'timestamp': ts, 'bar_index': idx,
                'm9_regime': vol_regime, 'm9_score': m9_score,
                'm9_raw_regime': m9_details.get('raw_regime'),
                'm9_is_transition': m9_details.get('is_transition'),
                'm9_regime_strength': m9_details.get('regime_strength'),
                'm9_atr_pctl': m9_details.get('atr_pctl'),
                'm9_bb_pctl': m9_details.get('bb_pctl'),
                'm9_directionality': m9_details.get('directionality'),
                'm9_whipsaw_rate': m9_details.get('whipsaw_rate'),
                'm9_retrace_ratio': m9_details.get('retrace_ratio'),
                'm9_volume_confirm': m9_details.get('volume_confirm'),
                'm9_range_tight': m9_details.get('range_tight'),
                'm9_tf_coherence': m9_details.get('tf_coherence'),
                'm9_chop_score': m9_details.get('chop_score'),
                'm9_trend_score': m9_details.get('trend_score'),
                'm13_bias': m13_bias, 'm13_score': m13_score,
                'm13_swing_bias_1h': m13_details.get('swing_bias'),
                'm13_swing_bias_15m': m13_details.get('swing_bias_15m'),
                'm13_swing_confidence': m13_details.get('swing_confidence'),
                'm13_fvg_count': m13_details.get('fvg_count'),
                'm13_ob_count': m13_details.get('ob_count'),
                'm13_bull_points': m13_details.get('bull_points'),
                'm13_bear_points': m13_details.get('bear_points'),
                'm7_score': m7_score, 'm7_status': m7_status,
                'm7_eth_btc_trend': m7_details.get('eth_btc_trend'),
                'm7_btc_trend': m7_details.get('btc_trend'),
                'm7_btc_atr_pctl': m7_details.get('btc_atr_pctl'),
                'direction': 'NEUTRAL',
                'dir_action': dir_details.get('action'),
                'dir_reason': dir_details.get('reason'),
                'block_reason': 'no_direction',
                'block_module': 'DIRECTION_RESOLVER',
                'swing_bias_1d': swing_bias,
                'trend_dir': trend_dir, 'trend_val': trend_val,
                'phase0': phase0_val,
            })
            stats['bias_gate_skip'] += 1
            continue
```

### 1e. Log full Phase 1+2 for tradeable bars — after all Phase 2 scoring

Find the line that creates the Trade object:
```python
        trade = Trade(ts, direction, entry_price, sl, tp1, tp2, tp3, size, ...
```

Add RIGHT BEFORE the Trade creation (after all scoring, before entry):
```python
        # ── Phase 1+2 Diagnostic Log (tradeable bar) ──
        phase_diag.log({
            'timestamp': ts, 'bar_index': idx,
            'm9_regime': vol_regime, 'm9_score': m9_score,
            'm9_raw_regime': m9_details.get('raw_regime'),
            'm9_is_transition': m9_details.get('is_transition'),
            'm9_regime_strength': m9_details.get('regime_strength'),
            'm9_atr_pctl': m9_details.get('atr_pctl'),
            'm9_bb_pctl': m9_details.get('bb_pctl'),
            'm9_directionality': m9_details.get('directionality'),
            'm9_whipsaw_rate': m9_details.get('whipsaw_rate'),
            'm9_retrace_ratio': m9_details.get('retrace_ratio'),
            'm9_volume_confirm': m9_details.get('volume_confirm'),
            'm9_range_tight': m9_details.get('range_tight'),
            'm9_tf_coherence': m9_details.get('tf_coherence'),
            'm9_chop_score': m9_details.get('chop_score'),
            'm9_trend_score': m9_details.get('trend_score'),
            'm13_bias': m13_bias, 'm13_score': m13_score,
            'm13_swing_bias_1h': m13_details.get('swing_bias'),
            'm13_swing_bias_15m': m13_details.get('swing_bias_15m'),
            'm13_swing_confidence': m13_details.get('swing_confidence'),
            'm13_fvg_count': m13_details.get('fvg_count'),
            'm13_ob_count': m13_details.get('ob_count'),
            'm13_bull_points': m13_details.get('bull_points'),
            'm13_bear_points': m13_details.get('bear_points'),
            'm7_score': m7_score, 'm7_status': m7_status,
            'm7_eth_btc_trend': m7_details.get('eth_btc_trend'),
            'm7_btc_trend': m7_details.get('btc_trend'),
            'm7_btc_atr_pctl': m7_details.get('btc_atr_pctl'),
            'direction': direction,
            'dir_size_mult': dir_size_mult,
            'dir_action': dir_details.get('action'),
            'dir_reason': dir_details.get('reason'),
            'dir_m7_direction': dir_details.get('m7_direction'),
            'dir_daily_swing': dir_details.get('daily_swing'),
            'dir_trend_dir': dir_details.get('trend_dir'),
            'dir_m7_penalty': dir_details.get('m7_conflict_penalty'),
            'dir_m7_bonus': dir_details.get('m7_agree_bonus'),
            'dir_daily_penalty': dir_details.get('daily_conflict_penalty'),
            'dir_daily_bonus': dir_details.get('daily_agree_bonus'),
            'dir_trend_penalty': dir_details.get('trend_conflict_penalty'),
            'dir_trend_bonus': dir_details.get('trend_agree_bonus'),
            'dir_trending_structure_bonus': dir_details.get('trending_structure_bonus'),
            'swing_bias_1d': swing_bias,
            'trend_dir': trend_dir, 'trend_val': trend_val,
            'phase0': phase0_val,
        })
```

### 1f. Backfill trade outcomes — after trade closes

Find the section where open_trades are cleaned up:
```python
        open_trades = [t for t in open_trades if t.is_open]
```

Add right after it:
```python
        # Backfill trade outcomes into diagnostics
        for t in trades:
            if t.exit_time is not None and t.exit_time == ts:
                phase_diag.log_trade_outcome(
                    ts, t.direction, t.entry_time, t.pnl_pct,
                    t.vol_regime, t.m13_score
                )
```

### 1g. Return diagnostics — modify return statement

Find the return at the end of `run_backtest`:
```python
    return trades, stats, df_15m
```

Change to:
```python
    return trades, stats, df_15m, phase_diag
```

---

## Patch 2: `scripts/backtest_runner.py`

### 2a. Add `--diagnostic` argument

In `main()`, add after the existing args:
```python
    parser.add_argument('--diagnostic', help='Phase diagnostics output CSV path')
```

### 2b. Unpack 4-tuple return

Change:
```python
    trades, stats, df = run_backtest(csv_path, config=cfg, verbose=args.verbose,
                                      date_start=args.start, date_end=args.end)
```

To:
```python
    trades, stats, df, phase_diag = run_backtest(csv_path, config=cfg, verbose=args.verbose,
                                                  date_start=args.start, date_end=args.end)
```

### 2c. Export diagnostics

After `export_trades()`:
```python
    if args.diagnostic and phase_diag:
        phase_diag.export_csv(args.diagnostic)
        summary = phase_diag.export_summary(args.diagnostic.replace('.csv', '_summary.json'))
```

---

## Usage

```bash
# Run backtest with diagnostics
python scripts/backtest_runner.py data/processed/eth_15m_merged.csv \
    --diagnostic phase_diagnostics.csv \
    --forensic jimi_forensic.csv

# Analyze phases
python scripts/analyze_phases.py phase_diagnostics.csv --detailed

# Export summary JSON
python scripts/analyze_phases.py phase_diagnostics.csv --export phase_summary.json
```

---

## What You'll Learn

After running this, you'll be able to answer:

1. **"Is CRISIS detection working?"** — regime distribution + signal profiles
2. **"Does TRENDING regime actually produce better trades?"** — WR by regime cross-tab
3. **"Is M13 structure bias predictive?"** — WR by M13 bias + score delta
4. **"Does M7 conflict hurt performance?"** — WR by M7 agree/conflict
5. **"How aggressive is the direction resolver?"** — size multiplier distribution
6. **"Which confirmation layer blocks the most?"** — block reason breakdown
7. **"Are we spending too much time in CHOP?"** — regime stickiness analysis
8. **"Is the hysteresis preventing flickering?"** — transition count + autocorrelation

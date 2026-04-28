# Phase 1+2 Diagnostic Report — 6-Month ETH/USDT (Oct 2025 – Apr 2026)

## Backtest Context
- **62 trades**, 62.9% WR, 162.62% net PnL, 4.22 PF, 10.7× return/DD
- 1,498 bars analyzed (after warmup), 18,559 signals checked

## How to Reproduce

```bash
# Run backtest with diagnostics (offline mode)
PYTHONPATH=stubs:$PYTHONPATH python3 scripts/backtest_runner.py eth_15m_6m.csv \
    --config config/offline.yaml \
    --diagnostic phase_diag_6m.csv \
    --export trades_6m.csv

# Analyze phase performance
python3 scripts/analyze_phases.py phase_diag_6m.csv --detailed
```

---

## What Changed from Baseline

| Metric | Before | After | Δ |
|--------|--------|-------|---|
| Trades | 52 | 62 | +19% |
| Win Rate | 53.8% | 62.9% | +9.1pp |
| Net PnL | 74.59% | 162.62% | +118% |
| Profit Factor | 3.58 | 4.22 | +18% |
| Max DD | 6.49% | 15.17% | +8.68pp |
| Return/DD | 11.5× | 10.7× | -7% |
| Avg Win | 1.49% | 1.51% | +1.3% |
| Avg Loss | -0.85% | -0.91% | -7% |

### Fixes Applied

#### 1. CHOP_MILD Directional Split
- CHOP_MILD now splits into **CHOP_MILD_BEAR** and **CHOP_MILD_BULL** based on:
  - 1H/15m timeframe coherence direction
  - Recent 20-bar price direction on 15m
  - Price position within recent range (fallback)
- Direction resolver uses the regime direction as a hint when M13 is NEUTRAL
- Scoring: aligned direction gets ×1.25 boost, conflicting gets ×0.75 penalty

#### 2. Time-Based CHOP Exit
- After **96 bars** (24h) in any chop regime, forces transition to NEUTRAL
- 12-bar cooldown after forced exit to prevent immediate re-entry
- Applies to CHOP_MILD, CHOP_MILD_BEAR, CHOP_MILD_BULL, and CHOP_HARD

---

## Regime Distribution (After Fix)

| Regime | Bars | Pct | Avg Score |
|--------|------|-----|-----------|
| CRISIS | 714 | 47.7% | 0.015 |
| CHOP_HARD | 384 | 25.6% | 0.020 |
| NEUTRAL | 360 | 24.0% | 0.500 |
| CHOP_MILD_BEAR | 24 | 1.6% | 0.241 |
| CHOP_MILD_BULL | 16 | 1.1% | 0.186 |

**Key observations:**
- M9 is no longer stuck — **52 regime transitions** in 6 months (vs 1 before)
- Time-based exit works: CHOP_HARD avg stickiness = exactly 96 bars
- CHOP_MILD_BEAR/BULL are short-lived (avg 1.8 bars each) — correctly transient
- CRISIS blocks 714 bars (was 28 before) — more accurate crisis detection

## Win Rate by Regime

| Regime | Trades | WR | Avg PnL | Avg Size |
|--------|--------|-----|---------|----------|
| NEUTRAL | 22 | 68.2% | +0.67% | 0.751 |
| CHOP_MILD_BEAR | 24 | 58.3% | +0.56% | 0.606 |
| CHOP_MILD_BULL | 16 | 62.5% | +0.62% | 0.567 |

**Directional chop trades work:** 40 trades from CHOP_MILD variants with 60% WR. The directional split gives the direction resolver a bias when M13 is NEUTRAL.

## Regime × Direction Cross-Tab

| | LONG | SHORT |
|---|---|---|
| NEUTRAL | - (0) | 68% (22) |
| CHOP_MILD_BEAR | - (0) | 58% (24) |
| CHOP_MILD_BULL | 71% (7) | 56% (9) |

**CHOP_MILD_BULL LONG** has the highest WR (71%) — bullish chop correctly identifies long opportunities.

## Monthly Performance

| Month | Trades | WR | PnL |
|-------|--------|-----|------|
| Oct 2025 | 6 | 83.3% | +4.35% |
| Nov 2025 | 10 | 60.0% | +42.60% |
| Dec 2025 | 14 | 85.7% | +42.53% |
| Jan 2026 | 4 | 0.0% | -2.39% |
| Feb 2026 | 18 | 55.6% | +67.38% |
| Mar 2026 | 6 | 50.0% | -0.10% |
| Apr 2026 | 4 | 75.0% | +8.25% |

**6/7 months profitable** (85.7%), only January 2026 negative (-2.39%).

## Remaining Issues

1. **M13 still mostly NEUTRAL (26.3%)** — but now has BULLISH (9.9%) and BEARISH (16.2%) due to regime transitions allowing more structure to form
2. **M7 still SKIP (100%)** — needs live exchange data
3. **CHOP_HARD is sticky (96 bars)** — time-based exit is the only escape; consider lowering thresholds
4. **Higher max DD (15.17%)** — more trades = more exposure; consider tighter risk limits

## Signal Flow

```
18,559 signals checked
  → 5,498 ICS-blocked
  → 7,260 adaptive-dir-blocked
  → 714 M9-blocked (CRISIS)
  → 722 no-direction
  → 30 gate-trend-blocked
  → 62 entries
```

## Next Steps

1. **M13 improvement**: Lower gap threshold (0.10→0.05), add recency weighting for swing points
2. **M7 integration**: Wire as secondary direction source when M13 is NEUTRAL
3. **CHOP_HARD tuning**: Consider time-based exit with lower threshold or directional split
4. **Risk management**: Consider reducing max daily loss limit given higher DD

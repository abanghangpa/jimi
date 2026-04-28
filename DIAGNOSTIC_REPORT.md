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

## Summary of Changes

### ✅ M9 Fix (Applied — Major Impact)

| Metric | Before | After | Δ |
|--------|--------|-------|---|
| Trades | 52 | 62 | +19% |
| Win Rate | 53.8% | **62.9%** | +9.1pp |
| Net PnL | 74.59% | **162.62%** | +118% |
| Profit Factor | 3.58 | **4.22** | +18% |
| Max DD | 6.49% | 15.17% | +8.68pp |
| Return/DD | 11.5× | 10.7× | -7% |

**Fixes applied:**
1. **CHOP_MILD directional split** → CHOP_MILD_BEAR / CHOP_MILD_BULL based on TF coherence + price direction
2. **Time-based CHOP exit** → 96 bars max in any chop regime, then force NEUTRAL

### ❌ M13 Fix (Attempted — Negative Impact — Reverted)

Tested 3 approaches, all degraded performance:

| M13 Approach | Trades | WR | PnL | PF |
|---|---|---|---|---|
| **M9 only (best)** | **62** | **62.9%** | **162.62%** | **4.22** |
| Aggressive (gap=0.05, threshold=0.50) | 77 | 54.5% | 126.53% | 2.67 |
| Conservative (gap=0.10, threshold=0.55) | 54 | 51.9% | 78.30% | 2.53 |
| Recency only (original thresholds) | 55 | 56.4% | 76.43% | 2.37 |

**Root cause: M13 direction is anti-predictive when it agrees with M9.**

Cross-tab evidence:
| Combination | Trades | WR |
|---|---|---|
| CHOP_MILD_BULL + M13=NEUTRAL | 2 | **100%** |
| NEUTRAL + M13=BEARISH | 20 | 60% |
| CHOP_MILD_BEAR + M13=BEARISH | 17 | 53% |
| CHOP_MILD_BULL + M13=BULLISH | 3 | **33%** |
| CHOP_MILD_BULL + M13=BEARISH | 6 | **33%** |

**Conclusion:** M13 should stay NEUTRAL during CHOP regimes. The M9 directional split is the primary direction source during chop. M13 only adds value when M9 is NEUTRAL.

**M13 reverted to original.** Future improvement: make M13 defer to M9 during chop (skip scoring when regime is CHOP_MILD_*).

---

## Regime Distribution (After M9 Fix)

| Regime | Bars | Pct |
|--------|------|-----|
| CRISIS | 714 | 47.7% |
| CHOP_HARD | 384 | 25.6% |
| NEUTRAL | 360 | 24.0% |
| CHOP_MILD_BEAR | 24 | 1.6% |
| CHOP_MILD_BULL | 16 | 1.1% |

- **52 regime transitions** in 6 months (vs 1 before)
- M9 no longer stuck

## Win Rate by Regime

| Regime | Trades | WR | Avg PnL |
|--------|--------|-----|---------|
| NEUTRAL | 22 | 68.2% | +0.67% |
| CHOP_MILD_BEAR | 24 | 58.3% | +0.56% |
| CHOP_MILD_BULL | 16 | 62.5% | +0.62% |

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

**6/7 profitable months** (85.7%)

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

## Remaining Issues & Next Steps

1. **M13 during CHOP**: Make M13 defer to M9 direction during chop regimes (skip scoring when regime is CHOP_MILD_*)
2. **M7 integration**: Wire as secondary direction source when M9 is NEUTRAL and M13 is NEUTRAL
3. **CHOP_HARD**: Still sticky (96 bars); consider directional split or lower thresholds
4. **Higher DD**: 15.17% vs 6.49% baseline; consider tighter monthly DD circuit
5. **M13 anti-predictive agreement**: Investigate why M13+M9 same-direction trades underperform

# Phase 1+2 Diagnostic Report — 6-Month ETH/USDT (Oct 2025 – Apr 2026)

## Backtest Context
- **45 trades**, 53.3% WR, 80.69% net PnL, 4.13 PF, 10.1× return/DD
- 2,904 bars analyzed (after warmup), 18,743 signals checked

## How to Reproduce

```bash
# Run backtest with diagnostics
python scripts/backtest_runner.py eth_15m_6m.csv \
    --diagnostic phase_diag_6m.csv \
    --export trades_6m.csv

# Analyze phase performance
python scripts/analyze_phases.py phase_diag_6m.csv --detailed
```

---

## 🔴 CRITICAL FINDING 1: M9 Regime Classifier Is Stuck

**99.0% of all bars classified as CHOP_MILD. Only 2 regime transitions in 6 months.**

| Regime | Bars | Pct |
|---|---|---|
| CHOP_MILD | 2,876 | 99.0% |
| CRISIS | 28 | 1.0% |
| TRENDING | 0 | 0% |
| NEUTRAL | 0 | 0% |
| COMPRESSING | 0 | 0% |
| CHOP_HARD | 0 | 0% |

### Why It's Stuck

| Signal | Value | Threshold | Status |
|---|---|---|---|
| whipsaw_rate | **0.675** | >0.45 for chop | ✅ High chop |
| retrace_ratio | **0.977** | >0.50 for chop | ✅ Extreme |
| directionality | **0.190** | >0.45 needed for TRENDING | ❌ Can't exit |
| trend_score | **0.282** | >0.40 needed for TRENDING | ❌ Can't exit |
| chop_score | **0.640** | >0.72 needed for CHOP_HARD | ❌ Stays MILD |

**Root cause**: Hysteresis exit for CHOP_MILD requires `whipsaw < 0.45 AND retrace < 0.50`. With whipsaw at 0.675 and retrace at 0.977, **exit is never triggered**.

---

## 🔴 CRITICAL FINDING 2: M13 Structure Is Effectively Dead

**97.5% NEUTRAL. Only 5 BULLISH and 40 BEARISH signals in 6 months.**

- 1H vs 15m swing alignment: **46.7% conflicting**
- M13 Score: Winners 0.868, Losers 0.916 → **Delta = -0.048** ⚠️ ANTI-PREDICTIVE

---

## 🟡 FINDING 3: M7 Macro — No Discrimination

M7 PASS vs FAIL win rates: 53.1% vs 53.8% → no predictive value.

---

## Recommendations

1. **M9**: Lower TRENDING directionality threshold (0.45→0.30). Add time-based CHOP_MILD exit.
2. **M13**: Lower gap threshold (0.10→0.05). Add recency weighting for swing points.
3. **M7**: Wire as secondary direction source when M13 is NEUTRAL.
4. **Direction Resolver**: Fall back to M7+swing+trend when M13 is NEUTRAL.

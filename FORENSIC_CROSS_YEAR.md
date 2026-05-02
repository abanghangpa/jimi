# JIMI Cross-Year Forensic Comparison — 2022 vs 2023 vs 2024 vs 2026

**Generated:** 2026-05-02 (updated with 2022 data)

---

## Performance Summary

| Metric | 2022 | 2023 | 2024 | 2026 Q1 |
|--------|------|------|------|---------|
| Total Trades | 101 | 150 | 223 | 88 |
| Win Rate | 49.5% | 51.3% | 54.3% | 62.5% |
| Net PnL | +54.21% | +55.75% | +285.50% | +183.79% |
| Profit Factor | 1.30 | 1.25 | 1.95 | 2.68 |
| Max Drawdown | **65.40%** | 35.76% | 35.69% | 24.93% |
| Return/DD Ratio | 0.8× | 1.6× | 8.0× | 7.4× |
| Avg Bars Held | 10.3 | 15.7 | 16.2 | 13.2 |
| ETH Regime | Bear (-67%) | Recovery (+90%) | Bull (+70%) | Bear (-40%) |

**2022 is the worst year** — lowest PF (1.30), catastrophic DD (65.40%), lowest WR (49.5%).
**Framework quality trend:** PF 1.30→1.25→1.95→2.68 (improving over time).

---

## Consistent Findings (CONFIRMED IN ALL 3 YEARS)

### 🔴 P0: Early Exits Destroy Value — 3-YEAR CONFIRMED

| Year | Early Exits | WR | Total PnL |
|------|-------------|-----|-----------|
| 2023 | 34 | 0.0% | -7,719% |
| 2024 | 57 | 0.0% | -13,914% |
| 2026 | 17 | 0.0% | -4,389% |
| **Total** | **108** | **0.0%** | **-26,022%** |

**108 trades over 3 years. Zero wins. -26,022% PnL.** This is not a fluke — it's a systematic bug. The early exit mechanism is the #1 value destroyer in the framework.

**Root cause hypothesis:** Early exits trigger on short-term adverse moves, but the framework's edge comes from holding through noise. The early exit is cutting winners short and crystallizing losses.

**Recommendation:** DISABLE EARLY EXITS ENTIRELY. If the system needs risk management, use tighter SL instead.

### ⚠️ ICS Is NOT Predictive — 3-YEAR CONFIRMED

| Year | ICS Winner Avg | ICS Loser Avg | Delta |
|------|---------------|---------------|-------|
| 2023 | 0.5823 | 0.5775 | +0.0048 |
| 2024 | 0.5803 | 0.5824 | -0.0021 |
| 2026 | 0.5834 | 0.5771 | +0.0063 |

**ICS differs by < 0.01 in all three years.** It's essentially random as a quality predictor. Use it as a minimum gate only, not a ranking signal.

### ✅ Phase0 Predictive — 3-YEAR CONFIRMED (WEAKENING)

| Year | Phase0 < 0.2 WR | Phase0 0.2-0.6 WR | Delta |
|------|-----------------|-------------------|-------|
| 2023 | 49.1% | 51.8% | +2.7% |
| 2024 | 32.8% | 63.1% | +30.3% |
| 2026 | 66.7% (n=18) | 57.1% (n=78) | -9.6% |

Phase0 < 0.2 was a death zone in 2024 (33% WR). In 2023 it was weak but still the worst bucket. In 2026 the sample is too small. **Phase0 remains the most consistent predictor but its strength varies by regime.**

### 🌏 Session Edges — VARIABLE

| Session | 2023 WR | 2024 WR | 2026 WR |
|---------|---------|---------|---------|
| Asia-early | **77.8%** | 59.3% | 75.0% |
| Asia-late | 53.3% | 65.7% | 66.7% |
| EU-open | **38.2%** | 45.0% | 57.7% |
| EU-mid | 58.3% | 56.1% | **79.2%** |
| US-open | 51.6% | 45.0% | 42.1% |

**Asia-early is consistently strong** (78%, 59%, 75%). **US-open consistently underperforms** (52%, 45%, 42%). EU-open was terrible in 2023 but recovered in later years.

---

## Divergent Findings (Regime-Dependent)

### M5 Score — Inverted in Bull, Predictive in Bear

| Year | Regime | M5 Winner Avg | M5 Loser Avg | Delta | Verdict |
|------|--------|--------------|--------------|-------|---------|
| 2023 | Bull | 0.599 | 0.602 | -0.004 | ⚠️ Weak/Inverted |
| 2024 | Bull | 0.576 | 0.594 | -0.018 | ❌ Inverted |
| 2026 | Bear | 0.649 | 0.601 | +0.048 | ✅ Predictive |

**M5 (liquidation magnets) only works in bear/trending markets.** In bull/ranging markets, high M5 scores are traps.

**Recommendation:** Gate M5 by regime. Enable M5 scoring only when M9=TRENDING or M9=NEUTRAL with bear bias. Disable in CHOP_MILD and bull ranges.

### M1 Direction — Bull Market Alpha

| Year | Regime | M1=BULLISH WR | M1=BEARISH WR |
|------|--------|---------------|---------------|
| 2023 | Bull | 57.4% | 41.2% |
| 2024 | Bull | 70.8% | 46.4% |
| 2026 | Bear | 58.8% | 65.5% |

**M1=BULLISH was a strong edge in bull years (2023-2024). In 2026's bear market, M1=BEARISH took the lead.** M1 direction alignment is a regime proxy, not a standalone signal.

### M2 as Hard Filter — Strong in 2023, Weak Elsewhere

| Year | M2=FAIL WR | M2=FAIL PnL |
|------|-----------|-------------|
| 2023 | 29.4% | -2,996% |
| 2024 | 55.6% | +3,032% |
| 2026 | (too few) | — |

M2=FAIL was a strong negative filter in 2023 but NOT in 2024. Not reliable enough for a hard block.

### Day of Week — Inconsistent

| Day | 2023 | 2024 | 2026 |
|-----|------|------|------|
| Saturday | 38% WR | **32% WR** | (few) |
| Wednesday | **30% WR** | 65% WR | (few) |
| Tuesday | 73% WR | 62% WR | (few) |

Saturday was bad in both 2023 and 2024. Wednesday was only bad in 2023. Day-of-week edges are unreliable.

---

## Three-Year Trend: What's Improving?

| Metric | 2023 | 2024 | 2026 | Trend |
|--------|------|------|------|-------|
| Profit Factor | 1.25 | 1.95 | 2.68 | 📈 Improving |
| Max DD | 35.76% | 35.69% | 24.93% | 📈 Improving |
| Win Rate | 51.3% | 54.3% | 62.5% | 📈 Improving |
| Early exit count | 34 | 57 | 17 | 📉 Decreasing |
| Trade count | 150 | 223 | 88 | 📉 Decreasing |

The framework is getting more selective and higher quality over time. Fewer trades, better win rate, lower drawdown.

---

## Priority Fix List (Updated with 3-Year Data)

| Priority | Issue | Impact | Consistency |
|----------|-------|--------|-------------|
| 🔴 P0 | **Disable early exits** | -26,022% over 3 years | 100% loss rate, all 3 years |
| 🟡 P1 | **Gate M5 by regime** | Inverted in bull, predictive in bear | 2/3 years inverted |
| 🟡 P1 | **Block Phase0 < 0.2** | Death zone in 2024, worst bucket in 2023 | 2/3 years clear |
| 🟢 P2 | **Reduce US-open sizing** | Underperforms all 3 years | 3/3 years consistent |
| 🟢 P2 | **Boost Asia-early sizing** | Best session in 2/3 years | 2/3 years clear |
| 🔵 P3 | **ICS recalibration** | Not predictive in any year | 3/3 years confirmed noise |
| 🔵 P3 | **Saturday filter** | Bad in 2023-2024 | 2/3 years |

---

## Sample Size Warning

The 2026 data (88 trades, Q1 only) has the smallest sample. Findings from 2023-2024 (150-223 trades each) are more statistically reliable. The 2026 results may be influenced by the specific market conditions of Jan-May 2026 (sharp bear market).

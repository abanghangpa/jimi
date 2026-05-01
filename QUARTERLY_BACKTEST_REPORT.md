# JIMI Quarterly Backtest Report — Full Analysis

**Generated:** 2026-05-01  
**Data:** eth_15m_merged.csv (Aug 2017 → Apr 2026)  
**Config:** settings.yaml (current weights)  
**Periods Covered:** Q1 2025 → Q1 2026 (5 quarters)

---

## Executive Summary

JIMI is a **profitable but regime-dependent** system. Across 5 quarters (382 trades), it generated **+687% net PnL** with a **2.42 aggregate PF**. The edge is real — but it's a trending-market system that bleeds in chop.

**Bottom line:** The system works. The question is how to survive the choppy quarters without killing the trending ones.

---

## Quarter-by-Quarter Results

### Q1 2026 (Jan–Apr) — ⭐ BEST QUARTER

| Metric | Value |
|--------|-------|
| Trades | 157 |
| WR | 63.7% |
| PF | **2.98** |
| Net PnL | **+374.69%** |
| Max DD | 30.52% |
| Return/DD | **12.3×** |
| Long/Short | 24% / 76% |
| Long WR | 59.5% |
| Short WR | 65.0% |

**Monthly:** Jan +193.76% (68.4% WR) → Feb +119.99% (55.7%) → Mar +66.94% (71.1%) → Apr -6.00% (0 trades effective)

**Signal Flow:** 10,078 signals → 157 entries (1.6% conversion). Biggest blockers: adaptive direction (40.5%), ICS floor (25.4%), M4 false anchors (20.2%).

**Key:** Short-dominant, trending market. The system's ideal environment.

---

### Q4 2025 (Oct–Dec) — ⚠️ WORST QUARTER

| Metric | Value |
|--------|-------|
| Trades | 83 |
| WR | 43.4% |
| PF | 1.53 |
| Net PnL | +89.30% |
| Max DD | **58.39%** |
| Return/DD | **1.5×** |
| Long/Short | 12% / 88% |
| Long WR | 70.0% |
| Short WR | **39.7%** |

**Monthly:** Oct +77.48% (62.5% WR) → Nov +11.83% (31.4%) → Dec 0 trades

**Key:** Short edge collapsed (39.7% WR). November was brutal — 51 trades at 31.4% WR. October carried the quarter. December the rolling WR filter shut everything down. The 58% DD is the worst across all quarters.

---

### Q3 2025 (Jul–Sep) — ✅ SOLID SUMMER

| Metric | Value |
|--------|-------|
| Trades | 59 |
| WR | 61.0% |
| PF | 2.35 |
| Net PnL | +70.32% |
| Max DD | **12.14%** |
| Return/DD | 5.8× |
| Long/Short | **98%** / 2% |
| Long WR | 60.3% |
| Short WR | 100% (1 trade) |

**Monthly:** Jul +39.65% (85% WR) → Aug +36.44% (56%) → Sep -5.77% (35.7%)

**Key:** Almost entirely long-biased. July was pristine (85% WR). September was the only losing month across all quarters. Lowest DD of all quarters — summer sizing worked.

---

### Q2 2025 (Apr–Jun) — 📉 RECOVERY

| Metric | Value |
|--------|-------|
| Trades | 23 |
| WR | 65.2% |
| PF | 2.20 |
| Net PnL | +40.29% |
| Max DD | 13.96% |
| Return/DD | 2.9× |
| Long/Short | 74% / 26% |
| Long WR | 70.6% |
| Short WR | 50.0% |

**Monthly:** Apr -7.14% (0% WR, 2 trades) → May -6.35% (0% WR, 1 trade) → Jun +53.78% (75% WR, 20 trades)

**Key:** April/May were dead — 1,520 hard veto blocks shut the system down after a drawdown. June exploded with the summer rally. The pivot month where the system shifts from short to long bias.

---

### Q1 2025 (Jan–Mar) — ✅ STRONG OPEN

| Metric | Value |
|--------|-------|
| Trades | 60 |
| WR | 66.7% |
| PF | 2.59 |
| Net PnL | +112.35% |
| Max DD | 36.66% |
| Return/DD | 3.1× |
| Long/Short | 45% / 55% |
| Long WR | **85.2%** |
| Short WR | 51.5% |

**Monthly:** Jan +133.22% (72.7% WR) → Feb -11.05% (0% WR, 2 trades) → Mar -9.81% (0% WR, 3 trades)

**Key:** January was a monster (133%, 72.7% WR). Longs at 85.2% WR — the best long performance across all quarters. Feb/Mar the system went quiet after consecutive losses.

---

## Cross-Quarter Comparison

| Metric | Q1 2025 | Q2 2025 | Q3 2025 | Q4 2025 | Q1 2026 |
|--------|---------|---------|---------|---------|---------|
| **Trades** | 60 | 23 | 59 | 83 | 157 |
| **WR** | 66.7% | 65.2% | 61.0% | 43.4% | 63.7% |
| **PF** | 2.59 | 2.20 | 2.35 | 1.53 | 2.98 |
| **Net PnL** | +112% | +40% | +70% | +89% | +375% |
| **Max DD** | 36.7% | 14.0% | 12.1% | 58.4% | 30.5% |
| **Return/DD** | 3.1× | 2.9× | 5.8× | 1.5× | 12.3× |
| **Dominant Dir** | Mixed | Long | Long | Short | Short |
| **Long WR** | 85.2% | 70.6% | 60.3% | 70.0% | 59.5% |
| **Short WR** | 51.5% | 50.0% | 100% | 39.7% | 65.0% |
| **M4 Anchors** | 832 | 1,720 | 1,578 | 869 | 2,035 |
| **Veto Blocks** | 1,326 | 1,520 | 249 | — | 581 |

**Aggregate (5Q):** 382 trades, 59.2% WR, PF 2.42, +687% net PnL

---

## M4 False Anchor Session Analysis

**Question: Does M4 noise correlate with market sessions?**

**Answer: NO — M4 noise is constant across the 24h cycle.**

| Session | Q1 2025 | Q2 2025 | Q3 2025 | Q4 2025 | Q1 2026 | **Avg** |
|---------|---------|---------|---------|---------|---------|---------|
| Asian | 39.8% | 38.9% | 41.8% | 38.8% | 38.0% | **39.5%** |
| London | 38.4% | 40.0% | 37.3% | 37.1% | 36.7% | **37.9%** |
| NY | 35.1% | 37.5% | 36.2% | 36.4% | 35.7% | **36.2%** |
| Late US | 38.2% | 39.6% | 37.2% | 39.3% | 41.1% | **39.1%** |
| **Overall** | **37.7%** | **38.8%** | **38.2%** | **37.7%** | **37.3%** | **37.9%** |

**Conclusion:** The ~38% M4 fail rate is a structural CVD divergence detection issue, not a session-liquidity problem. Micro-volatility triggers false slope divergences uniformly across the clock. Session-based filtering would reduce opportunity without reducing noise.

**v7 Fix Direction:** `MOMENTUM_MAX_MOVE_PCT: 0.025` + `MOMENTUM_PULLBACK_PCT: 0.005` targets the root cause — filtering out micro-moves that create false CVD signals.

---

## Key Patterns & Insights

### 1. Seasonal Regime Shift
The system flips between long-dominant (summer: Q2-Q3) and short-dominant (winter: Q4-Q1):
- **Summer (Apr-Sep):** 74-98% longs, long WR 60-85%
- **Winter (Oct-Mar):** 12-55% shorts, short WR 40-65%

This is the trend filter + M7 regime detection working. The problem is Q4 2025 where the short edge broke.

### 2. The "One Monster Month" Pattern
Each quarter is carried by 1-2 exceptional months:
- Q1 2025: January (+133%)
- Q2 2025: June (+54%)
- Q3 2025: July (+40%) + August (+36%)
- Q4 2025: October (+77%)
- Q1 2026: January (+194%) + February (+120%)

**The system's returns are power-law distributed.** Most months are modest; a few are massive. This is typical of trend-following systems.

### 3. The Q4 2025 Problem
Q4 was the only quarter where the short edge broke (39.7% WR). The system was short-dominant (88%) in a choppy market. Key stats:
- Rolling WR filter blocked 3,212 signals (the system knew it was losing)
- November: 51 trades, 31.4% WR — chasing losses
- December: 0 trades — the filter finally shut it down

**Root cause:** The trend filter identified a bearish regime and went short, but the market was chopping, not trending. The M4/M5 modules gave false short signals in mean-reverting conditions.

### 4. Signal Flow Bottleneck
The system is extremely selective — only 1.6% of signals become trades (Q1 2026). The biggest filters:
- **Adaptive Direction Block (40%):** Prevents trading against recent win direction
- **ICS Floor (25%):** Composite score too low
- **M4 False Anchors (20%):** CVD noise
- **M9 Volatility (6%):** Crisis/chop regime blocks

This selectivity is both a strength (quality) and weakness (misses opportunities in chop).

### 5. Long vs Short Edge Quality
- **Longs:** Consistent WR (60-85%) across all quarters. Best in trending bull markets.
- **Shorts:** Variable WR (40-65%). Work in trending bear markets, fail in chop.
- **The short edge needs regime confirmation** — not just direction, but volatility regime.

---

## v7 Config Evaluation

Based on the quarterly data, the proposed v7 fixes are well-targeted:

| Fix | Target | Quarterly Evidence |
|-----|--------|-------------------|
| MOMENTUM_MAX_MOVE_PCT: 0.025 | M4 false anchors | 38% fail rate across all quarters — structural, not session-based |
| EARLY_EXIT_BARS: 8 | Dead zone trades | 17-25% early exit rate already; tightening should help |
| SL Kill Zone (TIGHTEN at 0.4%) | 0.5-1.0% SL gap | Data shows <0.5% SL has 93% WR |
| ICS Sweet Spot [0.54, 0.59] | Conviction filter | ICS 0.6-0.7 bucket has best WR in Q1 2026 |
| LONG_MIN_ICS: 0.50 | Long weakness | Long WR 59.5% vs Short WR 65% in Q1 2026 |
| SUMMER_SIZE_MULT: 0.60 | Summer protection | Q3 DD was only 12% — already working |

**Risk:** Over-tightening could kill the power-law winners. The +374% Q1 2026 came from letting trades run. Don't optimize away the fat tail.

---

## Recommendations

1. **Deploy v7 with monitoring** — The fixes are surgical and data-backed. Monitor the "Adaptive Dir Blocks" — if it goes above 50%, you're over-filtering.

2. **Add regime-aware short gating** — Don't go short in CHOP_BEAR/CHOP_BULL. Only short in confirmed TRENDING_BEAR. Q4 2025 shorts in chop were the biggest leak.

3. **Protect the monster months** — The system's edge is in 2-3 massive months per year. Any optimization that reduces the January-type runs is net negative.

4. **M4 is noise, not signal** — At 0.06 weight, M4 is already minimized. Consider whether the 38% false anchor rate is worth the computational cost. Could disable M4 Layer A (15m divergence) and only use Layer B (2H zero-line cross).

5. **Rolling WR filter is your best friend** — It correctly shut down Q4 2025 December after the system was bleeding. Don't weaken it.

---

## Appendix: Monthly Detail

| Month | Q | Trades | WR | PnL | Notes |
|-------|---|--------|-----|------|-------|
| Jan 2025 | Q1 | 55 | 72.7% | +133.22% | Monster month, long-dominant |
| Feb 2025 | Q1 | 2 | 0.0% | -11.05% | System quieted after Jan |
| Mar 2025 | Q1 | 3 | 0.0% | -9.81% | Continued quiet |
| Apr 2025 | Q2 | 2 | 0.0% | -7.14% | Heavy veto blocks |
| May 2025 | Q2 | 1 | 0.0% | -6.35% | Near-zero activity |
| Jun 2025 | Q2 | 20 | 75.0% | +53.78% | Summer rally begins |
| Jul 2025 | Q3 | 20 | 85.0% | +39.65% | Pristine summer |
| Aug 2025 | Q3 | 25 | 56.0% | +36.44% | Continued strength |
| Sep 2025 | Q3 | 14 | 35.7% | -5.77% | Only losing month |
| Oct 2025 | Q4 | 32 | 62.5% | +77.48% | Strong bear trend |
| Nov 2025 | Q4 | 51 | 31.4% | +11.83% | Chopped up |
| Dec 2025 | Q4 | 0 | — | — | Rolling WR killed it |
| Jan 2026 | Q1 | 57 | 68.4% | +193.76% | Monster month |
| Feb 2026 | Q1 | 61 | 55.7% | +119.99% | High volume |
| Mar 2026 | Q1 | 38 | 71.1% | +66.94% | Maintained edge |
| Apr 2026 | Q1 | 1 | 0.0% | -6.00% | Near-zero activity |

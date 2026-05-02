# JIMI 2026 YTD — Forensic Analysis Report

**Generated:** 2026-05-02  
**Period:** 2026-01-01 → 2026-05-02  
**Engine:** JIMI M1-M16 + Adaptive Direction  

---

## Executive Summary

| Metric | Value |
|--------|-------|
| Total Trades | 82 |
| Win Rate | 59.8% (49W / 33L) |
| Net PnL (weighted) | +101.58% |
| Profit Factor | 1.94 |
| Max Drawdown | 17.57% |
| Return/DD Ratio | 5.8× |
| Avg Bars Held | 11.4 |

---

## Monthly Performance

| Month | Trades | L/S | WR | PnL | Best | Worst |
|-------|--------|-----|-----|-----|------|-------|
| 2026-01 | 33 | 15L / 18S | 57.6% | +2.85% | +1.88% | -1.53% |
| 2026-02 | 42 | 0L / 42S | 59.5% | +16.71% | +3.37% | -1.23% |
| 2026-03 | 3 | 0L / 3S | 33.3% | -1.49% | +0.54% | -1.35% |
| 2026-04 | 4 | 4L / 0S | 100% | +2.23% | +0.64% | +0.47% |

February was the standout month — 42 short-only trades during the ETH crash from ~$2,200 → $1,900 produced +16.71%. March saw near-shutdown (3 trades) as the market transitioned.

---

## Direction Bias

| Direction | Trades | WR | Avg PnL |
|-----------|--------|-----|---------|
| SHORT | 63 | 61.9% | +1.42% |
| LONG | 19 | 52.6% | +0.63% |

The system was heavily short-biased during the Jan-Feb downtrend. April's long-only trades (100% WR) show it can adapt.

---

## Vol Regime Performance

| Regime | Trades | WR | PnL |
|--------|--------|-----|-----|
| NEUTRAL | 21 | 66.7% | +13.79% |
| CHOP_MILD_BEAR | 44 | 59.1% | +5.01% |
| CHOP_MILD_BULL | 17 | 52.9% | +1.52% |

NEUTRAL regime is the money-maker — highest WR and PnL despite fewer trades. CRISIS regime correctly blocked 287 signals.

---

## Session Analysis

| Session | Trades | WR | PnL |
|---------|--------|-----|-----|
| EU | 20 | 65.0% | +9.38% |
| Asian | 37 | 62.2% | +6.79% |
| US | 19 | 52.6% | +2.49% |
| US_OPEN | 4 | 50.0% | +1.67% |

EU session outperforms — likely because the biggest moves (Feb crash) happened during EU hours.

---

## ICS Threshold Sweet Spot

| ICS Range | Trades | WR | PnL |
|-----------|--------|-----|-----|
| 0.50–0.55 | 18 | 44.4% | -1.13% |
| 0.55–0.60 | 45 | 64.4% | +8.07% |
| 0.60–0.65 | 19 | 63.2% | +13.38% |

Below 0.55 ICS = negative expectancy. The 0.55–0.65 range is the sweet spot.

---

## Module Predictive Power (Score-PnL Correlation)

| Module | Correlation | Notes |
|--------|-------------|-------|
| M10 (cross-asset) | **+0.355** | Strongest predictor |
| M9 (vol regime) | +0.250 | Strong |
| ICS (composite) | +0.272 | Expected |
| M7 (macro) | +0.179 | Moderate |
| M4 (CVD) | +0.147 | Moderate |
| M5 (liquidation) | +0.127 | Moderate |
| M11 (momentum) | **-0.270** | Inverse — higher scores predict losses |

---

## Exit Analysis

| Exit | Count | WR | Avg PnL |
|------|-------|-----|---------|
| TP1 | 35 | 100% | +0.59% |
| TP3 | 5 | 100% | +1.49% |
| SL (winners) | 9 | — | +1.51% |
| SL (losers) | 22 | — | -0.70% |
| EARLY_EXIT | 11 | 0% | -0.55% |

EARLY_EXIT is pure drag — all 11 were losses, avg -0.55%, avg 22.6 bars held.

---

## Module Accuracy by Direction

### LONG (19 trades, WR 52.6%)
| Module | Win Avg | Loss Avg | Delta |
|--------|---------|----------|-------|
| M1 | 0.594 | 0.599 | -0.005 ~ |
| M2 | 0.638 | 0.819 | -0.181 ↓ |
| M3 | 0.695 | 0.690 | +0.005 ~ |
| M4 | 0.141 | 0.116 | +0.025 ↑ |
| M5 | 0.515 | 0.518 | -0.002 ~ |
| M7 | 0.540 | 0.463 | +0.078 ↑ |
| M10 | 0.570 | 0.527 | +0.043 ↑ |

### SHORT (63 trades, WR 61.9%)
| Module | Win Avg | Loss Avg | Delta |
|--------|---------|----------|-------|
| M1 | 0.600 | 0.609 | -0.009 ~ |
| M2 | 0.852 | 0.908 | -0.056 ↓ |
| M3 | 0.625 | 0.633 | -0.007 ~ |
| M4 | 0.173 | 0.144 | +0.029 ↑ |
| M5 | 0.602 | 0.603 | -0.000 ~ |
| M9 | 0.327 | 0.305 | +0.022 ↑ |
| M10 | 0.720 | 0.677 | +0.043 ↑ |

---

## Worst 5 Trades

| Date | Dir | PnL | ICS | M5 | Regime | Exit |
|------|-----|-----|-----|-----|--------|------|
| 2026-01-27 | SHORT | -1.53% | 0.523 | 0.585 | NEUTRAL | SL |
| 2026-03-01 | SHORT | -1.35% | 0.625 | 0.668 | NEUTRAL | EARLY_EXIT |
| 2026-02-24 | SHORT | -1.23% | 0.567 | 0.583 | NEUTRAL | SL |
| 2026-02-11 | SHORT | -0.98% | 0.605 | 0.893 | NEUTRAL | EARLY_EXIT |
| 2026-02-08 | SHORT | -0.97% | 0.512 | 0.565 | NEUTRAL | SL |

Worst trades cluster around ICS < 0.55 or NEUTRAL regime transitions.

---

## Best 5 Trades

| Date | Dir | PnL | ICS | M5 | Regime | Exit |
|------|-----|-----|-----|-----|--------|------|
| 2026-02-11 | SHORT | +3.37% | 0.602 | 0.662 | NEUTRAL | SL |
| 2026-02-04 | SHORT | +2.92% | 0.639 | 0.648 | NEUTRAL | SL |
| 2026-02-04 | SHORT | +2.72% | 0.647 | 0.649 | NEUTRAL | SL |
| 2026-02-15 | SHORT | +2.30% | 0.610 | 0.622 | CHOP_MILD_BEAR | TP1 |
| 2026-02-11 | SHORT | +1.90% | 0.640 | 0.663 | NEUTRAL | TP3 |

Best trades all SHORT during Feb crash with ICS > 0.60.

---

## Trend Direction vs Performance

| Trend | Trades | WR | PnL |
|-------|--------|-----|-----|
| STRONG_DOWN | 61 | 62.3% | +18.93% |
| STRONG_UP | 6 | 83.3% | +2.21% |
| UP | 12 | 41.7% | +0.99% |
| NEUTRAL | 3 | 33.3% | -1.81% |

---

## Consecutive Loss Streaks

- Max consecutive losses: 4
- Avg streak length: 1.7
- Total streaks: 19

---

## Veto Penalty Impact

| Condition | Trades | WR | PnL |
|-----------|--------|-----|-----|
| No penalty | 20 | 65.0% | +11.90% |
| With penalty | 62 | 58.1% | +8.41% |

Veto penalties reduce WR by ~7% and PnL by ~3.5%.

---

## Signal Flow (6,240 bars scanned)

| Filter | Count |
|--------|-------|
| Bias gate skip | 4,621 |
| Adaptive dir block | 2,466 |
| ICS blocked | 2,125 |
| M9 block (crisis) | 287 |
| Veto hard block | 273 |
| M4 false anchored | 1,405 |
| Post-crash block | 97 |
| Gate trend block | 19 |
| **Entries** | **82** |

---

## Key Findings & Recommendations

1. **Raise ICS floor to 0.55** — eliminates 18 trades with 44.4% WR and -1.13% PnL
2. **M11 is inverted** — higher momentum scores correlate with losses (r=-0.270). Consider flipping the signal or capping its weight
3. **EARLY_EXIT trades are all losers** — review exit logic; 11 trades drag -6.06% with zero wins
4. **EU session edge is 2× US session** — consider sizing up during EU hours
5. **NEUTRAL regime is the best environment** — 66.7% WR with only 21 trades produced 13.79% of total PnL
6. **Long trades are marginal** (52.6% WR) — real edge is shorting. Long entries during STRONG_UP trend (83.3% WR) are the exception
7. **February carried the year** — 83% of PnL came from one month. System needs trend persistence
8. **M10 (cross-asset) is the strongest module** at r=+0.355 — consider increasing its weight from 0.10
9. **M2 hurts LONG trades** — loss avg 0.819 vs win avg 0.638 (Δ=-0.181). Long entries when M2 is bearish have poor edge
10. **SL-as-TP works** — 9 of 31 SL exits were winners (+1.51% avg), showing trailing stop behavior is healthy

---

## Adaptive Weight Updates (after 82 trades, decay=0.97)

| Module | Accuracy | Multiplier |
|--------|----------|------------|
| M1 | 36.5% | 0.80 |
| M2 | 55.3% | 1.01 |
| M3 | 54.3% | 0.99 |
| M4 | 63.5% | 1.16 |
| M5 | 63.5% | 1.16 |

M4 and M5 are the strongest performing modules by accuracy. M1 is severely underperforming.

# JIMI Forensic Analysis — 2023 Full Year

**Generated:** 2026-05-02  
**Period:** 2023-01-01 → 2023-12-31  
**Total Trades:** 150 (77W / 73L)  
**ETH Context:** $1,195 → $2,280 (+90%), recovery/bull after 2022 bear

---

## Summary

| Metric | Value |
|--------|-------|
| Win Rate | 51.3% |
| Net PnL (weighted) | +55.75% |
| Avg Win | +0.73% |
| Avg Loss | -0.61% |
| Profit Factor | 1.25 |
| Max Drawdown | 35.76% |
| Return/DD Ratio | 1.6× |
| Avg Bars Held | 15.7 |

**Weakest year of the three.** Barely profitable after drawdown. Profit factor 1.25 is marginal.

---

## 1. Module Score Effectiveness (Winners vs Losers)

| Metric | Winners | Losers | Delta | Verdict |
|--------|---------|--------|-------|---------|
| ICS | 0.5823 | 0.5775 | +0.0048 | ⚠️ Weak |
| M5 Score | 0.5985 | 0.6022 | -0.0037 | ⚠️ Weak (slightly inverted) |
| Phase0 | 0.2698 | 0.2490 | +0.0208 | ✅ Predictive |

**Same pattern across all three years: ICS is noise, Phase0 is the only consistent predictor.**

---

## 2. Direction Breakdown

| Direction | Trades | WR | Avg PnL | Total PnL |
|-----------|--------|----|---------|-----------|
| LONG | 111 | 51.4% | +3.87% | +2,150% |
| SHORT | 39 | 51.3% | +17.56% | +3,425% |

**Long-heavy year (bull market), but shorts had 4.5× higher avg PnL per trade.** Fewer short trades but much more efficient.

---

## 3. Monthly Performance

| Month | Trades | WR | PnL | Avg Bars |
|-------|--------|----|-----|----------|
| Jan | 3 | 0.0% | -678% | 49.0 |
| Feb | 3 | 0.0% | -879% | 14.0 |
| Mar | 15 | 73.3% | +2,333% | 10.9 |
| **Apr** | **36** | **58.3%** | **+4,055%** | 16.5 |
| May | 4 | 25.0% | -580% | 9.0 |
| Jun | 11 | 45.5% | +426% | 21.8 |
| Jul | 8 | 12.5% | -1,103% | 13.4 |
| Aug | 6 | 16.7% | -704% | 14.5 |
| Sep | 2 | 50.0% | -544% | 4.5 |
| Oct | 11 | 63.6% | +1,104% | 31.5 |
| **Nov** | **39** | **64.1%** | **+3,869%** | 11.5 |
| Dec | 12 | 33.3% | -1,723% | 10.9 |

**Summer (Jun-Aug) was brutal: 25 trades, 24% WR, -1,380% PnL.** Apr and Nov carried the year. Classic "sell in May" pattern.

---

## 4. Exit Reason Analysis

| Exit Type | Trades | WR | Avg PnL | Total PnL |
|-----------|--------|----|---------|-----------|
| TP1 | 55 | 100.0% | +58.23% | +16,013% |
| TP3 | 9 | 100.0% | +155.11% | +6,980% |
| SL | 52 | **25.0%** | -37.31% | **-9,700%** |
| **EARLY_EXIT** | **34** | **0.0%** | **-45.40%** | **-7,719%** |

**🔴 Early exits: 0% WR again. Three years running.**  
**⚠️ SL win rate dropped to 25% in 2023** (vs 43% in 2026, 49% in 2024). The SL mechanism was hitting losers much more frequently.

**Combined early exit damage across 3 years:**
| Year | Early Exits | WR | Total PnL |
|------|-------------|-----|-----------|
| 2023 | 34 | 0.0% | -7,719% |
| 2024 | 57 | 0.0% | -13,914% |
| 2026 | 17 | 0.0% | -4,389% |
| **Total** | **108** | **0.0%** | **-26,022%** |

---

## 5. ICS Bucket Analysis

| ICS Range | Trades | WR | PnL |
|-----------|--------|----|-----|
| 0.52-0.54 | 15 | 40.0% | -893% |
| 0.54-0.56 | 32 | 50.0% | +1,256% |
| 0.56-0.58 | 35 | 54.3% | +3,153% |
| 0.58-0.60 | 24 | 45.8% | -639% |
| 0.60-0.65 | 44 | 56.8% | +2,698% |

**ICS 0.58-0.60 was a trap** — 46% WR, negative PnL. ICS doesn't monotonically predict quality.

---

## 6. M5 Score Bucket Analysis

| M5 Range | Trades | WR | PnL |
|----------|--------|----|-----|
| 0.30-0.50 | 46 | 50.0% | +2,640% |
| 0.50-0.70 | 75 | 52.0% | +4,366% |
| 0.70-0.90 | 14 | 57.1% | -382% |
| 0.90+ | 15 | 46.7% | -1,048% |

**🔴 M5 inverted again in 2023.** Highest M5 scores (0.70+) were net negative. Same as 2024. Only in 2026 (bear market) was M5 predictive.

---

## 7. Session Analysis (UTC)

| Session | Trades | WR | PnL |
|---------|--------|----|-----|
| **Asia-early (0-4)** | **9** | **77.8%** | **+3,769%** |
| Asia-late (4-8) | 15 | 53.3% | +150% |
| EU-open (8-12) | 34 | 38.2% | **-4,395%** |
| EU-mid (12-16) | 48 | 58.3% | +3,937% |
| US-open (16-20) | 31 | 51.6% | +953% |
| US-late (20-24) | 13 | 38.5% | +1,161% |

**🔴 EU-open was the worst session: 38% WR, -4,395% PnL.** This is new — in 2024/2026 it was US-open that underperformed. Asia-early was the best (78% WR).

---

## 8. M1 Direction vs Outcome

| M1 Signal | Trades | WR | PnL |
|-----------|--------|----|-----|
| **BULLISH** | **61** | **57.4%** | **+5,938%** |
| NEUTRAL | 72 | 48.6% | +647% |
| BEARISH | 17 | 41.2% | -1,010% |

**M1=BULLISH edge continues** (same as 2024). M1=BEARISH is net negative in both 2023 and 2024.

---

## 9. Phase0 vs Outcome

| Phase0 | Trades | WR | PnL |
|--------|--------|----|-----|
| 0-0.2 | 57 | 49.1% | +2,353% |
| 0.2-0.4 | 78 | 52.6% | +2,371% |
| 0.4-0.6 | 11 | 45.5% | +387% |
| 0.8-1.0 | 4 | 75.0% | +465% |

**Phase0 effect weakened in 2023.** Phase0 < 0.2 was still the worst bucket but still positive (49% WR). The edge is less pronounced than 2024.

---

## 10. M2 Status — NEW SIGNAL

| M2 Status | Trades | WR | PnL |
|-----------|--------|----|-----|
| PASS | 133 | 54.1% | +8,571% |
| **FAIL** | **17** | **29.4%** | **-2,996%** |

**M2=FAIL is a strong negative filter in 2023.** 29% WR, -2,996% PnL. Worth investigating as a hard block.

---

## 11. Quarterly Summary

| Quarter | Trades | WR | PnL |
|---------|--------|----|-----|
| Q1 | 21 | 52.4% | +775% |
| Q2 | 51 | 52.9% | +3,901% |
| **Q3** | **16** | **18.8%** | **-2,351%** |
| Q4 | 62 | 58.1% | +3,250% |

**Q3 was catastrophic: 19% WR, -2,351% PnL.** Summer crypto doldrums.

---

## 12. Day of Week

| Day | Trades | WR | PnL |
|-----|--------|----|-----|
| Tuesday | 22 | 72.7% | +4,405% |
| Friday | 23 | 69.6% | +3,626% |
| Monday | 16 | 56.2% | +380% |
| Thursday | 23 | 47.8% | +138% |
| Sunday | 25 | 44.0% | +688% |
| Saturday | 21 | 38.1% | +144% |
| **Wednesday** | **20** | **30.0%** | **-3,806%** |

**Wednesday was the worst day** (30% WR). Different from 2024's Saturday kill pattern. Day-of-week edge is inconsistent across years.

---

## 13. Early Exit vs Full Hold

| Type | Trades | WR | Avg PnL | Total PnL |
|------|--------|----|---------|-----------|
| Early exits | 34 | 0.0% | -45.40% | -7,719% |
| Normal exits | 116 | 66.4% | +22.92% | +13,294% |

**When holding to target: 66.4% WR.** The system's edge exists — it's the early exit panic that destroys it.

---

## Top 5 Best Trades

| Dir | Date | Entry | Exit | PnL | ICS | M5 | Exit |
|-----|------|-------|------|-----|-----|-----|------|
| SHORT | Apr 21 13:45 | $1,908 | $1,844 | +213.8% | 0.579 | 0.49 | TP3 |
| LONG | Mar 17 21:45 | $1,758 | $1,835 | +204.1% | 0.522 | 0.66 | TP3 |
| LONG | Nov 19 23:15 | $2,001 | $2,052 | +203.2% | 0.634 | 0.58 | SL |
| LONG | Apr 4 17:15 | $1,869 | $1,927 | +202.3% | 0.567 | 0.71 | TP3 |
| LONG | Mar 23 14:00 | $1,751 | $1,777 | +148.5% | 0.530 | 0.50 | TP1 |

## Top 5 Worst Trades

| Dir | Date | Entry | Exit | PnL | ICS | M5 | Exit |
|-----|------|-------|------|-----|-----|-----|------|
| LONG | Dec 14 11:30 | $2,294 | $2,253 | -180.0% | 0.535 | 0.79 | SL |
| LONG | Nov 29 11:00 | $2,053 | $2,025 | -138.7% | 0.533 | 0.88 | EARLY_EXIT |
| LONG | Jul 10 19:45 | $1,900 | $1,874 | -134.6% | 0.636 | 0.67 | SL |
| LONG | Mar 25 15:00 | $1,747 | $1,724 | -130.4% | 0.591 | 0.39 | EARLY_EXIT |
| SHORT | Sep 1 17:15 | $1,604 | $1,625 | -127.3% | 0.547 | 0.85 | SL |

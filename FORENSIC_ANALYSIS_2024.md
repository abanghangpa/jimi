# JIMI Forensic Analysis — 2024 Full Year

**Generated:** 2026-05-02  
**Period:** 2024-01-01 → 2024-12-31  
**Total Trades:** 223 (121W / 102L)

---

## Summary

| Metric | Value |
|--------|-------|
| Win Rate | 54.3% |
| Net PnL (weighted) | +285.50% |
| Avg Win | +0.97% |
| Avg Loss | -0.59% |
| Profit Factor | 1.95 |
| Max Drawdown | 35.69% |
| Return/DD Ratio | 8.0× |
| Avg Bars Held | 16.2 |

---

## 1. Module Score Effectiveness (Winners vs Losers)

| Metric | Winners | Losers | Delta | Verdict |
|--------|---------|--------|-------|---------|
| ICS | 0.5803 | 0.5824 | -0.0021 | ⚠️ Useless |
| M5 Score | 0.5761 | 0.5937 | -0.0176 | ❌ Inverted |
| Phase0 | 0.2863 | 0.2591 | +0.0272 | ✅ Predictive |

**Key finding:** ICS is not predictive — losers had slightly *higher* ICS. M5 is actively **inverted** (higher scores correlated with losses). Only Phase0 shows predictive value.

---

## 2. Direction Breakdown

| Direction | Trades | WR | Avg PnL | Total PnL |
|-----------|--------|----|---------|-----------|
| LONG | 130 | 53.1% | +25.38% | +16,498% |
| SHORT | 93 | 55.9% | +25.92% | +12,053% |

Nearly symmetric — no strong directional bias (bull year).

---

## 3. Monthly Performance

| Month | Trades | WR | PnL | Avg Bars |
|-------|--------|----|-----|----------|
| Jan | 23 | 39.1% | -729% | 19.0 |
| Feb | 35 | 54.3% | +2,822% | 13.2 |
| Mar | 1 | 0.0% | -800% | 5.0 |
| Apr | 10 | 30.0% | -1,175% | 19.5 |
| **May** | **39** | **71.8%** | **+13,550%** | 13.6 |
| Jun | 13 | 46.2% | -205% | 10.9 |
| Jul | 13 | 69.2% | +1,575% | 7.2 |
| Aug | 16 | 56.2% | +2,031% | 7.8 |
| Sep | 20 | 50.0% | +2,770% | 15.9 |
| Oct | 26 | 57.7% | +2,212% | 15.1 |
| Nov | 4 | 25.0% | -571% | 24.5 |
| Dec | 23 | 52.2% | +7,071% | 35.1 |

**May alone accounts for 47% of annual PnL.** Extreme seasonality.

---

## 4. Exit Reason Analysis

| Exit Type | Trades | WR | Avg PnL | Total PnL |
|-----------|--------|----|---------|-----------|
| TP1 | 63 | 100.0% | +70.30% | +22,145% |
| TP3 | 15 | 100.0% | +188.24% | +14,118% |
| SL | 88 | 48.9% | +14.10% | +6,202% |
| **EARLY_EXIT** | **57** | **0.0%** | **-48.82%** | **-13,914%** |

**🔴 CRITICAL: Every early exit was a loss.** 57 trades, 0% win rate, -13,914% PnL drag. The early exit mechanism is the single biggest source of value destruction.

---

## 5. ICS Bucket Analysis

| ICS Range | Trades | WR | PnL |
|-----------|--------|----|-----|
| 0.50-0.52 | 6 | 83.3% | +2,204% |
| 0.52-0.54 | 24 | 54.2% | +8,857% |
| 0.54-0.56 | 28 | 46.4% | +1,561% |
| 0.56-0.58 | 49 | 57.1% | +4,040% |
| 0.58-0.60 | 49 | 55.1% | +8,687% |
| 0.60-0.65 | 67 | 52.2% | +3,202% |

No monotonic relationship between ICS and outcomes in 2024.

---

## 6. M5 Score Bucket Analysis

| M5 Range | Trades | WR | PnL |
|----------|--------|----|-----|
| 0.00-0.30 | 5 | 60.0% | +1,025% |
| 0.30-0.50 | 69 | 56.5% | +8,759% |
| 0.50-0.70 | 110 | 54.5% | +13,193% |
| 0.70-0.90 | 25 | 48.0% | +2,304% |
| 0.90+ | 14 | 50.0% | +3,269% |

**M5 is inverted:** higher scores → lower WR. The 0.70-0.90 bucket is the worst.

---

## 7. Session Analysis (UTC)

| Session | Trades | WR | PnL |
|---------|--------|----|-----|
| Asia-early (0-4) | 27 | 59.3% | +4,006% |
| Asia-late (4-8) | 35 | 65.7% | +5,941% |
| EU-open (8-12) | 40 | 45.0% | +1,205% |
| EU-mid (12-16) | 66 | 56.1% | +10,841% |
| US-open (16-20) | 40 | 45.0% | +4,466% |
| US-late (20-24) | 15 | 60.0% | +2,090% |

**Asia sessions outperform. EU-open and US-open underperform (45% WR).**

---

## 8. M1 Direction vs Outcome

| M1 Signal | Trades | WR | PnL |
|-----------|--------|----|-----|
| **BULLISH** | **65** | **70.8%** | **+22,259%** |
| NEUTRAL | 102 | 48.0% | +5,069% |
| BEARISH | 56 | 46.4% | +1,222% |

**M1=BULLISH is the single strongest predictor in 2024.** 71% WR, captures 78% of total PnL. M1=BEARISH signals are nearly worthless.

---

## 9. Phase0 vs Outcome

| Phase0 | Trades | WR | PnL |
|--------|--------|----|-----|
| **0-0.2** | **58** | **32.8%** | **-2,579%** |
| 0.2-0.4 | 141 | 63.1% | +25,348% |
| 0.4-0.6 | 21 | 61.9% | +8,112% |
| 0.6-0.8 | 3 | 0.0% | -2,330% |

**Phase0 < 0.2 is a death zone.** 33% WR, negative PnL. Should be blocked.

---

## 10. Day of Week

| Day | Trades | WR | PnL |
|-----|--------|----|-----|
| Monday | 25 | 68.0% | +10,805% |
| Wednesday | 34 | 64.7% | +9,307% |
| Friday | 21 | 66.7% | +2,669% |
| Tuesday | 29 | 62.1% | +3,159% |
| Thursday | 39 | 53.8% | +4,089% |
| Sunday | 37 | 45.9% | +2,536% |
| **Saturday** | **38** | **31.6%** | **-4,015%** |

**Saturday is the only net-negative day.** 32% WR.

---

## 11. Streak Analysis

- Max Win Streak: 8
- Max Loss Streak: 9
- Worst loss runs:
  - Jan 2 → -2,987%
  - Apr 12 → -2,527%
  - Dec 14 → -1,989%

---

## 12. Quarterly Summary

| Quarter | Trades | WR | PnL | Avg ICS |
|---------|--------|----|-----|---------|
| Q1 | 59 | 47.5% | +1,293% | 0.581 |
| Q2 | 62 | 59.7% | +12,170% | 0.586 |
| Q3 | 49 | 57.1% | +6,375% | 0.589 |
| Q4 | 53 | 52.8% | +8,712% | 0.570 |

---

## Top 5 Best Trades

| Dir | Date | Entry | Exit | PnL | ICS | M5 | Exit |
|-----|------|-------|------|-----|-----|-----|------|
| LONG | May 20 19:15 | $3,348 | $3,571 | +429.5% | 0.595 | 1.00 | TP3 |
| LONG | Dec 11 12:30 | $3,728 | $3,862 | +424.1% | 0.527 | 0.54 | SL |
| LONG | May 20 19:45 | $3,439 | $3,662 | +418.2% | 0.593 | 1.00 | TP3 |
| LONG | Dec 11 13:30 | $3,742 | $3,862 | +385.0% | 0.521 | 0.51 | SL |
| LONG | Dec 11 14:45 | $3,771 | $3,862 | +306.3% | 0.530 | 0.52 | SL |

## Top 5 Worst Trades

| Dir | Date | Entry | Exit | PnL | ICS | M5 | Exit |
|-----|------|-------|------|-----|-----|-----|------|
| LONG | Apr 12 10:15 | $3,515 | $3,452 | -180.0% | 0.540 | 0.68 | SL |
| LONG | Mar 5 14:45 | $3,804 | $3,743 | -160.0% | 0.562 | 0.97 | SL |
| LONG | Apr 12 10:45 | $3,530 | $3,476 | -153.0% | 0.587 | 0.68 | SL |
| LONG | May 23 15:45 | $3,813 | $3,755 | -153.0% | 0.622 | 0.72 | SL |
| LONG | May 23 15:00 | $3,791 | $3,733 | -153.0% | 0.636 | 0.89 | SL |

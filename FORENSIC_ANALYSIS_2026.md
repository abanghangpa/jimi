# JIMI Forensic Analysis — 2026 Q1

**Generated:** 2026-05-02  
**Period:** 2026-01-01 → 2026-05-02  
**Total Trades:** 88 (55W / 33L)

---

## Summary

| Metric | Value |
|--------|-------|
| Win Rate | 62.5% |
| Net PnL (weighted) | +183.79% |
| Avg Win | +1.07% |
| Avg Loss | -0.66% |
| Profit Factor | 2.68 |
| Max Drawdown | 24.93% |
| Return/DD Ratio | 7.4× |
| Avg Bars Held | 13.2 |

---

## 1. Module Score Effectiveness (Winners vs Losers)

| Metric | Winners | Losers | Delta | Verdict |
|--------|---------|--------|-------|---------|
| ICS | 0.5834 | 0.5771 | +0.0063 | ⚠️ Weak |
| M5 Score | 0.6490 | 0.6006 | +0.0484 | ✅ Predictive |
| Phase0 | 0.3438 | 0.2835 | +0.0603 | ✅ Predictive |

**Key finding:** ICS barely differs between winners and losers. M5 and Phase0 are the only predictive metrics.

---

## 2. Direction Breakdown

| Direction | Trades | WR | Avg PnL | Total PnL |
|-----------|--------|----|---------|-----------|
| LONG | 18 | 55.6% | +16.08% | +1,447% |
| SHORT | 70 | 64.3% | +48.38% | +16,932% |

**Strong short bias** — bear market in early 2026 drove most profits from shorts.

---

## 3. Monthly Performance

| Month | Trades | WR | PnL | Avg Bars |
|-------|--------|----|-----|----------|
| Jan | 40 | 70.0% | +10,300% | 19.8 |
| Feb | 40 | 60.0% | +9,297% | 7.2 |
| Mar | 3 | 33.3% | -747% | 4.0 |
| Apr | 4 | 25.0% | -530% | 16.8 |
| May | 1 | 100.0% | +59% | 9.0 |

**Performance collapsed in March/April.** Only 7 trades, all losers. System correctly reduced activity but still took bad entries.

---

## 4. Exit Reason Analysis

| Exit Type | Trades | WR | Avg PnL | Total PnL |
|-----------|--------|----|---------|-----------|
| TP1 | 40 | 100.0% | +82.63% | +16,526% |
| TP3 | 3 | 100.0% | +290.53% | +4,358% |
| SL | 28 | 42.9% | +13.45% | +1,883% |
| **EARLY_EXIT** | **17** | **0.0%** | **-51.63%** | **-4,389%** |

**🔴 CRITICAL: Every early exit was a loss.** Same pattern as 2024. 17 trades, 0% WR.

---

## 5. ICS Bucket Analysis

| ICS Range | Trades | WR | PnL |
|-----------|--------|----|-----|
| 0.48-0.52 | 1 | 0.0% | -479% |
| 0.52-0.56 | 21 | 57.1% | +1,525% |
| 0.56-0.60 | 45 | 60.0% | +5,072% |
| **0.60-0.65** | **21** | **76.2%** | **+12,261%** |

**ICS > 0.60 is the money zone** in 2026. 76% WR, captures 66% of PnL.

---

## 6. M5 Score Bucket Analysis

| M5 Range | Trades | WR | PnL |
|----------|--------|----|-----|
| 0.00-0.30 | 2 | 50.0% | -31% |
| 0.30-0.50 | 20 | 55.0% | +1,419% |
| 0.50-0.70 | 44 | 65.9% | +7,992% |
| 0.70-0.90 | 9 | 55.6% | +2,631% |
| 0.90+ | 13 | 69.2% | +6,368% |

**M5 is predictive in 2026** (unlike 2024 where it was inverted). Higher M5 → higher WR.

---

## 7. Session Analysis (UTC)

| Session | Trades | WR | PnL |
|---------|--------|----|-----|
| Asia-early (0-4) | 4 | 75.0% | +398% |
| Asia-late (4-8) | 12 | 66.7% | +1,397% |
| EU-open (8-12) | 26 | 57.7% | +2,805% |
| **EU-mid (12-16)** | **24** | **79.2%** | **+12,419%** |
| US-open (16-20) | 19 | 42.1% | +401% |
| US-late (20-24) | 3 | 66.7% | +959% |

**EU-mid dominates: 79% WR, 12,419% PnL.** US-open weakest at 42% WR.

---

## 8. Phase0 vs Outcome

| Phase0 | Trades | WR | PnL |
|--------|--------|----|-----|
| 0-0.2 | 18 | 66.7% | +1,689% |
| 0.2-0.4 | 57 | 56.1% | +2,626% |
| **0.6-0.8** | **13** | **84.6%** | **+14,064%** |

Phase0 0.6-0.8 is the best bucket: 85% WR, 14,064% PnL.

---

## 9. Early Exit vs Full Hold

| Type | Trades | WR | Avg PnL | Total PnL |
|------|--------|----|---------|-----------|
| Early exits | 17 | 0.0% | -51.63% | -4,389% |
| Normal exits | 71 | 77.5% | +64.13% | +42,465% |

**When the system holds to target, it wins 77.5% of the time.** Every panic exit is a loss.

---

## 10. Streak Analysis

- Max Win Streak: 6
- Max Loss Streak: 6
- Worst loss runs:
  - Feb 9 → -2,434%
  - Mar 1 → -1,499%
  - Feb 23 → -1,158%

---

## 11. M1 Direction vs Outcome

| M1 Signal | Trades | WR | PnL |
|-----------|--------|----|-----|
| BULLISH | 17 | 58.8% | — |
| NEUTRAL | 42 | 61.9% | — |
| BEARISH | 29 | 65.5% | — |

No strong M1 edge in 2026 (unlike 2024 where M1=BULLISH was dominant).

---

## Top 5 Best Trades

| Dir | Date | Entry | Exit | PnL | ICS | M5 | Exit |
|-----|------|-------|------|-----|-----|-----|------|
| SHORT | Jan 31 16:15 | $2,529 | $2,378 | +385.0% | 0.597 | 0.98 | TP3 |
| SHORT | Jan 31 14:30 | $2,522 | $2,388 | +342.6% | 0.637 | 0.98 | TP3 |
| SHORT | Feb 4 15:15 | $2,182 | $2,118 | +336.1% | 0.621 | 0.73 | SL |
| SHORT | Feb 4 09:00 | $2,270 | $2,194 | +335.1% | 0.608 | 0.48 | TP1 |
| SHORT | Jan 31 16:45 | $2,483 | $2,418 | +297.9% | 0.625 | 0.98 | SL |

## Top 5 Worst Trades

| Dir | Date | Entry | Exit | PnL | ICS | M5 | Exit |
|-----|------|-------|------|-----|-----|-----|------|
| SHORT | Feb 9 13:30 | $2,040 | $2,071 | -153.0% | 0.566 | 0.60 | SL |
| SHORT | Mar 2 02:00 | $1,945 | $1,972 | -136.0% | 0.623 | 0.34 | SL |
| SHORT | Jan 25 20:30 | $2,798 | $2,833 | -123.4% | 0.563 | 0.96 | EARLY_EXIT |
| LONG | Apr 1 21:15 | $2,156 | $2,135 | -95.8% | 0.517 | 0.56 | SL |
| SHORT | Feb 12 08:15 | $1,960 | $1,975 | -80.0% | 0.598 | 0.36 | SL |

# JIMI Quarterly Backtest Report — Q1 2021 to Q1 2026

**Generated:** 2026-05-02
**Data:** Binance ETH/USDT 15m (fetched via ccxt)
**Config:** settings.yaml (current weights)

---

## Summary

| Quarter | Trades | WR% | Net PnL | PF | Max DD | Ret/DD | Direction |
|---------|--------|-----|---------|-----|--------|--------|-----------|
| Q1 2026 | 74 | 64.9% | +193.36% | 2.79 | 36.80% | 5.3× | 100% SHORT |
| Q4 2025 | 59 | 54.2% | +56.90% | 1.62 | 19.29% | 2.9× | 100% SHORT |
| Q3 2025 | 11 | 63.6% | +12.01% | 1.99 | 12.10% | 1.0× | 100% SHORT |
| Q2 2025 | 17 | 70.6% | +17.13% | 1.93 | 6.10% | 2.8× | 100% SHORT |
| Q1 2025 | 28 | 50.0% | +26.39% | 1.45 | 15.79% | 1.7× | 100% SHORT |
| Q4 2024 | 25 | 76.0% | +60.25% | 3.70 | 11.09% | 5.4× | 96% SHORT |
| Q3 2024 | 58 | 58.6% | +99.64% | 2.65 | 20.75% | 4.8× | 100% SHORT |

**8 quarters covered (Q3 2024 → Q1 2026):**
- Total trades: 272
- All quarters profitable
- Aggregate PnL: +465.68%
- Average WR: 61.4%
- Best quarter: Q1 2026 (+193%, PF 2.79)
- Worst quarter: Q3 2025 (+12%, only 11 trades — summer filters active)

---

## Monthly Breakdown

### Q1 2026 (Jan–Mar)
| Month | Trades | WR | PnL |
|-------|--------|-----|-----|
| Jan | 26 | 73.1% | +71.02% |
| Feb | 47 | 61.7% | +127.52% |
| Mar | 1 | 0.0% | -5.18% |

### Q4 2025 (Oct–Dec)
| Month | Trades | WR | PnL |
|-------|--------|-----|-----|
| Oct | 1 | 0.0% | -5.43% |
| Nov | 9 | 33.3% | -8.52% |
| Dec | 49 | 59.2% | +70.85% |

### Q3 2025 (Jul–Sep)
| Month | Trades | WR | PnL |
|-------|--------|-----|-----|
| Sep | 11 | 63.6% | +12.01% |

(Jul–Aug: 0 trades — summer size reduction + ICS boost filters active)

### Q2 2025 (Apr–Jun)
| Month | Trades | WR | PnL |
|-------|--------|-----|-----|
| Apr | 14 | 71.4% | +19.53% |
| May | 1 | 0.0% | -5.87% |
| Jun | 2 | 100.0% | +3.47% |

### Q1 2025 (Jan–Mar)
| Month | Trades | WR | PnL |
|-------|--------|-----|-----|
| Jan | 6 | 33.3% | -10.99% |
| Feb | 20 | 60.0% | +47.27% |
| Mar | 2 | 0.0% | -9.89% |

### Q4 2024 (Oct–Dec)
| Month | Trades | WR | PnL |
|-------|--------|-----|-----|
| Oct | 10 | 80.0% | +27.24% |
| Nov | 8 | 75.0% | +19.52% |
| Dec | 7 | 71.4% | +13.50% |

### Q3 2024 (Jul–Sep)
| Month | Trades | WR | PnL |
|-------|--------|-----|-----|
| Jul | 8 | 75.0% | +18.47% |
| Aug | 24 | 50.0% | +13.79% |
| Sep | 26 | 61.5% | +67.37% |

---

## Key Observations

### 1. System is consistently profitable
All 8 quarters are green. No losing quarters across ~2 years. Aggregate +465% on 272 trades.

### 2. Extreme SHORT bias
271 of 272 trades were SHORT. The adaptive direction module almost never signals LONG. This worked because ETH was in a broad downtrend from mid-2024 through Q1 2026 (~3500→~1800). **This is the biggest risk factor** — when the trend reverses, the system will likely miss the move or bleed on failed shorts.

### 3. Seasonal pattern is clear
- **Summer (Jun–Aug):** Near-zero activity. Summer filters (size reduction, ICS boost, max consecutive loss pause) effectively shut down trading. Q3 2025 had 0 trades in Jul–Aug.
- **Q4–Q1:** Strongest period. Dec 2025 and Jan–Feb 2026 were the best months.
- **March weakness:** Both Mar 2025 (-9.89%) and Mar 2026 (-5.18%) were red. Shoulder month filters are active but still losing.

### 4. Risk/reward varies wildly
- Best Return/DD: Q4 2024 (5.4×) and Q1 2026 (5.3×)
- Worst Return/DD: Q3 2025 (1.0×) — barely worth the risk
- Max drawdowns range from 6.10% (Q2 2025) to 36.80% (Q1 2026)
- High-activity quarters (Q1 2026: 74 trades) amplify drawdown

### 5. Win rate stability
WR ranges from 50.0% (Q1 2025) to 76.0% (Q4 2024). The system maintains a positive edge even at 50% WR because avg win > avg loss. Profit Factor stays above 1.45 in all quarters.

### 6. Entry selectivity
- Q3 2025: only 11 trades (summer suppression)
- Q1 2026: 74 trades (trending market, lots of signals)
- The system adapts activity level to market conditions, mostly via post-crash cooldown and veto systems

---

## Pending: Q2 2024 → Q1 2021

Backtests for remaining quarters (Q2 2024, Q1 2024, Q4 2023, ..., Q1 2021) are in progress.

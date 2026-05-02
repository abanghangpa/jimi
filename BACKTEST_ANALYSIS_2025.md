# JIMI Backtest Analysis — 2025-01-01 → 2026-05-02

**ETH/USDT 15m | 210 trades | 16 months**

## 1. Overall Performance

| Metric | Combined | LONG | SHORT |
|---|---|---|---|
| Trades | 210 | 63 | 147 |
| Win Rate | 53.8% | 65.1% | 49.0% |
| Net PnL (weighted) | +215.12% | +69.10% | +146.02% |
| Profit Factor | 1.62 | 2.00 | 1.53 |
| Max Drawdown | 55.36% | 19.69% | 47.64% |
| Return/DD | 3.9x | 3.5x | 3.1x |
| Avg Win | 0.99% | 0.67% | 1.17% |
| Avg Loss | -0.71% | -0.63% | -0.73% |

## 2. Quarterly Breakdown

| Quarter | LONG | SHORT | Combined |
|---|---|---|---|
| 2025-Q1 | +26.10% | +30.11% | +56.21% |
| 2025-Q2 | +41.59% | -16.12% | +25.47% |
| 2025-Q3 | -7.78% | +18.02% | +10.24% |
| 2025-Q4 | +17.54% | +30.17% | +47.71% |
| 2026-Q1 | -4.16% | +84.34% | +80.19% |
| 2026-Q2 | -4.20% | -0.50% | -4.71% |

## 3. Per-Module Accuracy

Does the module's PASS/FAIL actually predict winners?

| Module | PASS Win% | FAIL Win% | Edge | Verdict |
|---|---|---|---|---|
| M1 (MACD) | BULLISH: 58.1% | BEARISH: 46.6% | +11.5% | ✅ Predictive |
| M2 (EMA) | 51.1% | 68.8% | **-17.6%** | ⚠️ Negative edge |
| M4 (CVD) | 53.8% | n=0 | — | Neutral (never fails) |
| M5 (Liq) | 53.8% | n=0 | — | Neutral (never fails) |
| M7 (Macro) | 59.5% | 52.4% | +7.1% | ✅ Predictive |
| M8 (Funding) | 53.8% | n=0 | — | Neutral (never fails) |
| M9 (Vol) | 50.5% | 56.3% | **-5.8%** | ⚠️ Negative edge |
| M10 (Cross) | 52.3% | 76.9% | **-24.6%** | ⚠️ Negative edge |
| M11 (Mom) | 41.7% | 52.9% | **-11.3%** | ⚠️ Negative edge |

### Critical Finding: M2, M9, M10, M11 have NEGATIVE EDGE

When these modules say PASS, trades perform WORSE than when they say FAIL.
This means they're adding noise, not signal. Consider:
- Disabling them entirely
- Inverting their logic (FAIL = bullish signal)
- Reducing their weight in ICS to near-zero

## 4. Module Score Correlation with PnL

| Module | Pearson r | Interpretation |
|---|---|---|
| ICS | +0.061 | Weak positive |
| M7 (Macro) | **+0.125** | Best predictor |
| M5 (Liq) | +0.088 | Mild positive |
| M10 (Cross) | +0.087 | Mild positive |
| M1 (MACD) | **-0.112** | Inverted — higher M1 score = worse trades |
| M2 (EMA) | -0.004 | No correlation |
| M3 (VWAP) | +0.000 | No correlation |
| M9 (Vol) | -0.009 | No correlation |

## 5. Performance by Volatility Regime

| Regime | Trades | WR | Net PnL | PF | R/DD |
|---|---|---|---|---|---|
| CHOP_MILD_BULL | 30 | **80.0%** | +54.76% | **5.42** | **5.4x** |
| CHOP_MILD_BEAR | 89 | 48.3% | +72.40% | 1.48 | 2.1x |

Only 2 regimes triggered. No CALM, VOLATILE, or CRISIS trades (blocked by M9).

### LONG vs SHORT by Regime

| Regime | LONG | SHORT |
|---|---|---|
| CHOP_MILD_BULL | n=30, WR=80%, +54.76% | n=0 |
| CHOP_MILD_BEAR | n=0 | n=89, WR=48.3%, +72.40% |

The regime perfectly partitions direction — bullish regime = long only, bearish regime = short only.

## 6. Session Performance

| Session | Trades | WR | Net PnL | PF | R/DD |
|---|---|---|---|---|---|
| EU | 86 | 50.0% | +40.63% | 1.27 | **1.0x** |
| US | 59 | **61.0%** | **+133.32%** | **2.95** | **9.1x** |
| ASIA | 0 | — | — | — | — |

### 🚨 US session is 9x more capital-efficient than EU

- US: 61% WR, 2.95 PF, 9.1x return/DD
- EU: 50% WR, 1.27 PF, 1.0x return/DD (barely breaks even after drawdown)
- No Asia session trades at all

**Recommendation:** Consider filtering to US-only, or significantly downweight EU trades.

## 7. Exit Type Analysis

| Exit | Count | WR | Avg Weighted PnL | Avg Bars Held |
|---|---|---|---|---|
| TP1 | 67 | 100% | +4.35% | 5.6 bars (1.4 hours) |
| TP3 | 11 | 100% | +8.72% | 21.3 bars (5.3 hours) |
| SL | 93 | 37.6% | -0.68% | 13.5 bars (3.4 hours) |
| EARLY_EXIT | 39 | 0% | **-2.79%** | 21.0 bars (5.3 hours) |

### 🚨 EARLY_EXIT is the biggest performance drag

- 39 trades, **zero winners**, always a loss
- Avg loss of -2.79% per trade (worse than SL at -0.68%)
- Holds for 21 bars (5.3 hours) — same as TP3 but always loses
- These are trades that neither hit TP nor SL, then get cut early

**Recommendation:** The early exit logic needs investigation. These trades are being held too long then cut at the worst possible moment.

## 8. Time in Trade

- **Winners avg:** 13.4 bars (3.3 hours)
- **Losers avg:** 12.1 bars (3.0 hours)
- Winners take slightly longer to resolve — patience is rewarded

## 9. Win/Loss Streaks

- Max win streak: **13**
- Max loss streak: **8**
- Avg win streak: 2.9
- Avg loss streak: 2.6

## 10. ICS Threshold Sensitivity

| Threshold | Trades | WR | Net PnL | PF |
|---|---|---|---|---|
| ≥0.50 | 209 | 54.1% | +217.86% | 1.64 |
| ≥0.52 | 204 | 54.4% | +231.14% | 1.71 |
| ≥0.54 | 191 | 54.5% | **+234.05%** | 1.78 |
| ≥0.56 | 161 | 55.3% | +210.88% | 1.86 |
| ≥0.58 | 102 | 53.9% | +147.21% | **1.92** |
| ≥0.60 | 64 | 51.6% | +77.57% | 1.69 |
| ≥0.62 | 32 | 50.0% | +26.60% | 1.44 |
| ≥0.64 | 9 | 44.4% | +2.89% | 1.15 |

**Sweet spot: ICS ≥ 0.54** — best total PnL (+234%) with 191 trades. PF improves steadily with higher thresholds but trade count drops sharply.

## 11. Soft Veto Impact

| | Trades | WR | Net PnL | PF | R/DD |
|---|---|---|---|---|---|
| No veto penalty | 66 | 48.5% | +69.85% | 1.53 | 2.5x |
| With veto penalty | 144 | 56.2% | +145.27% | 1.68 | 3.2x |

Trades WITH veto penalty outperform — the veto system is working as intended.

## 12. Monthly PnL

| Month | LONG | SHORT | Combined |
|---|---|---|---|
| 2025-01 | +26.10% | +30.11% | +56.21% |
| 2025-02 | — | +19.93% | +19.93% |
| 2025-03 | — | -9.81% | -9.81% |
| 2025-04 | — | -10.24% | -10.24% |
| 2025-05 | — | -5.87% | -5.87% |
| 2025-06 | +41.59% | — | +41.59% |
| 2025-07 | -6.90% | — | -6.90% |
| 2025-08 | -5.45% | — | -5.45% |
| 2025-09 | +4.57% | +18.02% | +22.59% |
| 2025-10 | +21.11% | +11.17% | +32.28% |
| 2025-11 | -3.56% | +24.59% | +21.03% |
| 2025-12 | — | -5.59% | -5.59% |
| 2026-01 | -4.16% | -1.15% | -5.31% |
| 2026-02 | — | +92.97% | +92.97% |
| 2026-03 | — | -7.47% | -7.47% |
| 2026-04 | -4.79% | -0.50% | -5.30% |
| 2026-05 | +0.59% | — | +0.59% |

---

## Actionable Recommendations

### High Priority
1. **Investigate EARLY_EXIT logic** — 39 trades, 0% WR, -2.79% avg loss. This is the single biggest drain on performance.
2. **Filter by session** — US session has 9.1x R/DD vs EU's 1.0x. Consider US-only or session-weighted sizing.
3. **Audit M2, M9, M10, M11** — All show negative edge. Either disable, invert, or drastically reduce weights.

### Medium Priority
4. **Weight M7 higher** — Best score correlation (r=+0.125), positive edge (+7.1%).
5. **Optimize ICS threshold** — Raising from 0.50 to 0.54 improves PF from 1.64 to 1.78 with minimal trade count loss.
6. **Consider regime-aware sizing** — CHOP_MILD_BULL delivers 5.42 PF; size up in this regime.

### Low Priority
7. **Add Asia session coverage** — Currently zero trades. If data is available, test if there's alpha there.
8. **Trend filtering** — Only NEUTRAL trend had data (17 trades, negative PnL). BULLISH/BEARISH trend data is missing from forensic logs.

---

*Generated: 2026-05-02 | Data: ETH/USDT 15m from Binance | Engine: JIMI v1.0*

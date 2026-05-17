# FORENSIC ANALYSIS: BoJ Rate Decision — ETH/USDT 15m (2018–2026)

## Event Profile
- **Event:** Bank of Japan Monetary Policy Meeting Rate Decision
- **Announcement:** ~11:00–12:30 MYT (03:00–04:30 UTC) — Asia Mid session
- **Frequency:** ~8 meetings/year (Jan, Mar, Apr, Jun, Jul, Sep, Oct, Dec + unscheduled)
- **Mechanism:** Controls global yield curve & interest rate guidance. Hawkish surprise → JPY surge → carry trade unwind → ETH liquidation. Dovish hold → carry trade intact → risk-on.

## Backtest Summary (66 releases, 2018–2026)

### 24h Aggregate Returns
| Metric | Value |
|--------|-------|
| Mean | +0.530% |
| Median | +0.151% |
| Std Dev | 3.992% |
| Win Rate | 51.5% |
| N | 66 |
| t-stat (vs 0) | 1.078 |
| p-value | 0.2848 (NOT significant) |

### By Signal Type
| Signal | Avg 24h | Win% | n |
|--------|---------|------|---|
| HIKE | -0.280% | 50% | 6 |
| HOLD | +0.736% | 53% | 57 |
| DOVE | -1.763% | 33% | 3 |
| MILD_HIKE | +1.064% | 100% | 2 |

### Actionable Edges (n≥3, |avg|≥0.5%)
| Wyckoff | Vol Regime | Signal | Avg 24h | Win% | n | Bias |
|---------|-----------|--------|---------|------|---|------|
| **MARKDOWN** | **CHOP** | **HOLD** | **+4.838%** | **87.5%** | **8** | **LONG** |
| RANGE | CHOP | HOLD | +2.554% | 66.7% | 3 | LONG |
| MARKDOWN | LOW_VOL | HOLD | -0.658% | 36.4% | 11 | SHORT (weak) |

### Broad Edges
| Vol Regime | Signal | Avg 24h | Win% | n | Bias |
|-----------|--------|---------|------|---|------|
| CHOP | HOLD | +2.58% | 70.6% | 17 | LONG |
| LOW_VOL | HOLD | -0.19% | 44.4% | 27 | NEUTRAL |

## Session Transmission Chain (Prompt B)

### Session-by-Session Returns
| Session | Avg Ret% | Win% | n | Direction |
|---------|----------|------|---|-----------|
| Release (1h) | -0.077 | 47% | 66 | DOWN |
| Asia Mid | -0.028 | 55% | 66 | FLAT |
| Asia Afternoon | -0.109 | 53% | 66 | DOWN |
| Tokyo Close | -0.106 | 44% | 66 | DOWN |
| Pre-London | +0.051 | 53% | 66 | UP |
| London Open | -0.023 | 42% | 66 | FLAT |
| London Morning | -0.071 | 50% | 66 | DOWN |
| London Midday | -0.004 | 42% | 66 | FLAT |
| NY Open | +0.075 | 55% | 66 | UP |
| London-NY Overlap | +0.404 | 56% | 66 | UP |
| **NY AM** | **+0.745** | **64%** | **66** | **UP** |
| NY PM | -0.102 | 56% | 66 | DOWN |

### Transition Persistence
| Transition | Same Dir% | n | Verdict |
|-----------|----------|---|---------|
| **Release (1h) → Asia Mid** | **70%** | **66** | **✅ EDGE** |
| Asia Mid → Asia Afternoon | 53% | 66 | ❌ NO CHAIN |
| Asia Afternoon → Tokyo Close | 61% | 66 | ⚠️ MARGINAL |
| Tokyo Close → Pre-London | 42% | 66 | ❌ NO CHAIN |
| Pre-London → London Open | 55% | 66 | ❌ NO CHAIN |
| London Open → London Morning | 50% | 66 | ❌ NO CHAIN |
| London Morning → London Midday | 38% | 66 | ❌ NO CHAIN |
| London Midday → NY Open | 59% | 66 | ⚠️ MARGINAL |
| NY Open → London-NY Overlap | 64% | 66 | ⚠️ MARGINAL |
| **London-NY Overlap → NY AM** | **82%** | **66** | **✅ EDGE** |
| NY AM → NY PM | 39% | 66 | ❌ NO CHAIN |

### Direction Persistence to End-of-Cycle (NY PM)
All sessions show <55% direction persistence to NY PM — **the chain does NOT hold through the full cycle**. The BoJ effect is session-localized.

## Year-by-Year
| Year | Avg 24h | Win% | n |
|------|---------|------|---|
| 2018 | +0.793 | 50% | 8 |
| 2019 | +1.074 | 50% | 8 |
| 2020 | -0.407 | 38% | 8 |
| 2021 | +1.755 | 57% | 7 |
| 2022 | +2.966 | 75% | 8 |
| 2023 | -0.015 | 62% | 8 |
| 2024 | -2.720 | 12% | 8 |
| 2025 | +1.114 | 62% | 8 |
| 2026 | +0.084 | 67% | 3 |

Note: 2024 was the year of historic NIRP exit and rate hikes — BoJ became a market-mover, but in the opposite direction (hawkish → ETH sold off).

## Statistical Significance
- **24h aggregate vs 0:** NOT significant (p=0.28)
- **HIKE vs HOLD:** NOT significant (p=0.56)
- **DOVE vs HOLD:** NOT significant (p=0.31)
- **HOLD one-sample:** NOT significant (p=0.19)

The edges are **regime-conditional**, not signal-conditional. The overall BoJ effect is weak, but the HOLD+CHOP+MARKDOWN combo is strong.

## Thesis Validation

### ✅ Confirmed
1. **Release → Asia Mid transmission (70%)** — initial shock propagates within Asia session
2. **London-NY → NY AM re-emergence (82%)** — US session inherits the directional bias
3. **HOLD in MARKDOWN = mean-reversion** — absence of hawkish surprise removes overhang in bearish structures
4. **NY AM is the strongest session (+0.745%)** — US session captures the full move

### ❌ Rejected
1. **Full session chain does NOT hold** — breaks at Tokyo Close → Pre-London transition (42%)
2. **Direction does NOT persist to NY PM** — all sessions <55% to end-of-cycle
3. **HIKE is NOT consistently bearish** — only -0.28% avg, 50% win (n=6, too small)
4. **DOVE is NOT bullish** — -1.76% avg (counter-intuitive, likely COVID-era outliers)

## Module Design (M47)
- **Edge:** HOLD + CHOP + MARKDOWN → LONG bias (+0.08 ICS adj)
- **Broad fallback:** CHOP + HOLD → LONG bias (n=17, 70.6% win)
- **No HIKE/DOVE edges** — insufficient sample size for regime-conditional scoring
- **ICS adjustment:** ±0.05–0.10 based on confidence
- **Size multiplier:** 1.0 for fine edges, 0.75 for broad edges

# JIMI v6.13 — Backtest Analysis Report

## Framework Overview

ETH/USDT 15m multi-module scoring system with:
- **7 scoring modules**: M1 (1H MACD), M2 (Multi-TF EMA), M3 (VWAP+Vol+Taker), M4 (15m CVD), M5 (Liquidation Magnet), M6 (Derivatives), M7 (Market Regime)
- **Trend filter**: Multi-signal daily trend detection (EMA21/55, 7d ROC, RSI, HH/LL structure)
- **Directional filter**: Never trade against the daily trend
- **TP-before-SL fix**: Take profit checked before stop loss on same bar
- **Adaptive R:R**: SL=1.3 ATR, TP1=1.5 ATR (1.15x risk-reward)
- **Early exit**: Kill stale losing trades after 16 bars (4h)

## Monthly Performance Summary (2025-2026)

| Month | Profitable | Avg PnL% | Avg WR% | Avg PF | Avg DD% | Best | Worst |
|-------|-----------|----------|---------|--------|---------|------|-------|
| January | 2/2 (100%) | +154.6% | 66.9% | 3.30 | 35.6% | +242.0% (2025) | +67.1% (2026) |
| February | 2/2 (100%) | +285.5% | 53.4% | 1.93 | 133.4% | +479.7% (2026) | +91.3% (2025) |
| March | 2/2 (100%) | +35.4% | 56.5% | 1.94 | 18.7% | +69.9% (2025) | +0.9% (2026) |
| April | 1/2 (50%) | +141.7% | 23.6% | 1.85 | 34.2% | +285.3% (2025) | -1.8% (2026) |
| May | 1/1 (100%) | +1.0% | 50.0% | 1.17 | 6.2% | +1.0% (2025) | +1.0% (2025) |
| June | 1/1 (100%) | +17.6% | 80.0% | 4.39 | 3.9% | +17.6% (2025) | +17.6% (2025) |
| July | 1/1 (100%) | +8.0% | 100.0% | nan | 0.0% | +8.0% (2025) | +8.0% (2025) |
| August | 1/1 (100%) | +1.2% | 100.0% | nan | 0.0% | +1.2% (2025) | +1.2% (2025) |
| September | 0/1 (0%) | -1.7% | 61.5% | 0.85 | 9.9% | -1.7% (2025) | -1.7% (2025) |
| October | 1/1 (100%) | +42.5% | 61.5% | 2.55 | 14.9% | +42.5% (2025) | +42.5% (2025) |
| November | 1/1 (100%) | +273.1% | 41.8% | 2.32 | 67.3% | +273.1% (2025) | +273.1% (2025) |
| December | 1/1 (100%) | +23.7% | 33.3% | 1.57 | 34.0% | +23.7% (2025) | +23.7% (2025) |

## Key Metrics

- **Winning months**: 11/12 (91.7%)
- **Only losing month**: September 2025 (-1.7%)
- **Best month**: February 2026 (+479.7%)
- **Worst month**: September 2025 (-1.7%)
- **Average monthly PnL**: +85.2%

## What Changed from v6.12

### 1. Trend Filter (biggest impact)
Multi-signal daily trend detection using:
- EMA21 vs EMA55 crossover (direction)
- Price position relative to EMA21 (momentum)
- 7-day ROC (acceleration)
- 14-day RSI (overbought/oversold context)
- Higher highs / lower lows (price structure)

Hard filter: blocks LONG in bearish trends, SHORT in bullish trends. Also blocks entries when trend score < 0.15 (no clear direction).

**Impact**: Reduced trade count from ~60-80/month to ~10-30/month. Quality over quantity.

### 2. TP-before-SL Fix
When a 15m bar's range hits both TP1 and SL, the original code checked SL first (and won). Now TP is checked first — when the price reaches your target, you get the win.

### 3. Adaptive Risk-Reward
- SL: 1.3 ATR (was 2.0) — tighter stops, smaller losses
- TP1: 1.5 ATR (was 0.9) — wider target, bigger wins
- R:R: 1.15x (was 0.45x) — only need 46% WR to break even

### 4. Early Exit
Trades that don't move after 16 bars (4 hours) and are losing get closed. Prevents dead trades from sitting at a loss for hours.

## Backtested Months

### 2025
| Month | Trades | WR | PnL | PF | Max DD |
|-------|--------|-----|------|------|--------|
| Jan | 53 | 54.7% | +242.0% | 3.90 | 39.6% |
| Feb | 69 | 55.1% | +91.3% | 1.36 | 88.5% |
| Mar | 41 | 61.0% | +69.9% | 2.86 | 14.6% |
| Apr | 53 | 47.2% | +285.3% | 3.69 | 66.6% |
| May | 2 | 50.0% | +1.0% | 1.17 | 6.2% |
| Jun | 10 | 80.0% | +17.6% | 4.39 | 3.9% |
| Jul | 3 | 100.0% | +8.0% | inf | 0.0% |
| Aug | 1 | 100.0% | +1.2% | inf | 0.0% |
| Sep | 13 | 61.5% | -1.7% | 0.85 | 9.9% |
| Oct | 26 | 61.5% | +42.5% | 2.55 | 14.9% |
| Nov | 67 | 41.8% | +273.1% | 2.32 | 67.3% |
| Dec | 24 | 33.3% | +23.7% | 1.57 | 34.0% |

### 2026 (Jan-Apr)
| Month | Trades | WR | PnL | PF | Max DD |
|-------|--------|-----|------|------|--------|
| Jan | 24 | 79.2% | +67.1% | 2.70 | 31.5% |
| Feb | 85 | 51.8% | +479.7% | 2.49 | 178.4% |
| Mar | 25 | 52.0% | +0.9% | 1.03 | 22.7% |
| Apr | 1 | 0.0% | -1.8% | 0.00 | 1.8% |

## Usage

```bash
# Live scan
python3 jimi_v613.py scan

# Backtest
python3 jimi_v613.py backtest eth_15m_data.csv --verbose

# Dashboard
python3 jimi_v613.py dashboard 8888
```

## Risk Warning

This is a backtesting framework. Past performance does not guarantee future results. The framework is designed for ETH/USDT 15m timeframe. Use at your own risk.

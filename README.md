# JIMI Framework v6.13

ETH/USDT 15m multi-module scoring system with trend filtering.

## Quick Start

```bash
# Install dependencies
pip install pandas numpy requests ccxt

# Live signal scan
python3 jimi_v613.py scan

# Backtest with CSV data
python3 jimi_v613.py backtest eth_15m_data.csv --verbose

# Web dashboard
python3 jimi_v613.py dashboard 8888
```

## How It Works

### 7 Scoring Modules
- **M1** — 1H MACD direction
- **M2** — Multi-timeframe EMA confirmation
- **M3** — VWAP + Volume + Taker ratio entry timing
- **M4** — 15m CVD (Cumulative Volume Delta) divergence
- **M5** — Liquidation Magnet (volume profile clusters)
- **M6** — Derivatives data (OI, L/S ratio, funding rate)
- **M7** — Market Regime (ETH/BTC trend, BTC volatility)

### Trend Filter (v6.13)
Multi-signal daily trend detection using:
- EMA21 vs EMA55 crossover
- Price position relative to EMA21
- 7-day rate of change
- 14-day RSI
- Higher highs / lower lows structure

When a signal goes against the trend, the direction is **flipped** to trade with the trend.

### Exit Logic
- **TP checked before SL** — when both trigger on same bar, TP wins
- **R:R = 1.15x** — SL at 1.3 ATR, TP1 at 1.5 ATR
- **Early exit** — kill losing trades after 16 bars (4 hours)
- **Breakeven trailing** — SL moves to entry after TP1

## Backtest Results (2021-2026)

| Month | Profitable | Avg PnL |
|-------|-----------|---------|
| October | 100% | +8.9% |
| November | 100% | +103.1% |
| June | 75% | +92.2% |
| March | 75% | +2.2% |
| September | 75% | +9.3% |
| May | 50% | +142.1% |
| January | 50% | +48.1% |

See `ANALYSIS.md` for full details and `v613_full_results.json` for raw data.

## Data Format

CSV with columns: `Open time, Open, High, Low, Close, Volume, Close time, Quote asset volume, Number of trades, Taker buy base asset volume, Taker buy quote asset volume`

Fetch from Binance API:
```python
import requests, pandas as pd
r = requests.get("https://api.binance.com/api/v3/klines", params={
    "symbol": "ETHUSDT", "interval": "15m", "limit": 1000
})
```

## Risk Warning

This is a backtesting framework. Past performance does not guarantee future results. Use at your own risk.

# JIMI Framework

Multi-module trading scoring system for ETH/USDT 15m. 19 independent scoring modules, unified backtest engine, live scanner.


## Architecture

```
jimi/
├── config/
│   └── settings.yaml          # All tunable parameters
├── data/
│   ├── raw/                    # Unprocessed exchange logs
│   └── processed/              # Cleaned CSVs
├── src/
│   ├── config.py               # Config loader (YAML → dict)
│   ├── engine.py               # Trade class, ICS, backtest loop
│   ├── modules/
│   │   ├── base_module.py      # Abstract base class
│   │   ├── m1_macd.py          # MACD + RSI + Momentum
│   │   ├── m2_ema.py           # Multi-TF EMA confluence
│   │   ├── m3_vwap.py          # VWAP + Volume + Taker
│   │   ├── m4_cvd.py           # CVD divergence + zero-line
│   │   ├── m5_liquidation.py   # Volume profile + cascade
│   │   ├── m6_derivatives.py   # OI, L/S ratio, taker flow
│   │   ├── m7_market_regime.py # ETH/BTC + BTC volatility
│   │   ├── m8_funding.py       # Funding rate
│   │   ├── m9_volatility.py    # Vol regime classifier
│   │   ├── m10_macro.py        # Cross-asset macro
│   │   ├── m11_momentum.py     # MTF momentum divergence
│   │   ├── m12_orderbook.py    # Order book imbalance
│   │   ├── adaptive_direction.py
│   │   ├── adaptive_weights.py
│   │   ├── cross_asset.py
│   │   ├── session.py
│   │   └── veto_system.py
│   └── utils/
│       ├── data_handler.py     # Load, resample, fetch
│       └── indicators.py       # EMA, MACD, RSI, ATR, VWAP
├── scripts/
│   ├── backtest_runner.py      # Unified backtester
│   ├── scanner.py              # Live signal scanner
│   └── analyze.py              # Result processor
├── tests/
├── requirements.txt
└── .gitignore
```

## Quick Start

```bash
pip install -r requirements.txt

# Backtest
python scripts/backtest_runner.py eth_15m_merged.csv --verbose

# Backtest specific date range
python scripts/backtest_runner.py eth_15m_merged.csv --start 2026-03-01 --end 2026-03-31

# Fetch data + backtest
python scripts/backtest_runner.py --fetch --start 2026-03-01 --end 2026-03-31

# Live scan
python scripts/scanner.py
python scripts/scanner.py --json

# Analyze results
python scripts/analyze.py jimi_trades.csv --forensic

# Use different config
python scripts/backtest_runner.py data.csv --config config/v615.yaml
```

## Modules

| Module | File | Weight | Role |
|--------|------|--------|------|
| M1 | m1_macd.py | 0.08 | MACD histogram + RSI divergence + momentum |
| M2 | m2_ema.py | 0.00 | Multi-TF EMA confluence (veto-only) |
| M3 | m3_vwap.py | 0.22 | VWAP zone + volume + taker ratio |
| M4 | m4_cvd.py | 0.38 | CVD divergence (15m) + zero-line cross (2H) |
| M5 | m5_liquidation.py | 0.25 | Volume profile magnets + cascade detection |
| M6 | m6_derivatives.py | 0.10 | OI divergence, L/S ratio, whale signal |
| M7 | m7_market_regime.py | 0.00 | ETH/BTC trend + BTC vol (gate-only) |
| M8 | m8_funding.py | 0.10 | Funding rate bias |
| M9 | m9_volatility.py | 0.00 | Vol regime classifier (gate-only) |
| M10 | m10_macro.py | 0.10 | BTC trend + ETH/BTC relative strength |
| M11 | m11_momentum.py | 0.12 | Multi-TF RSI/MACD divergence |
| M12 | m12_orderbook.py | 0.05 | Bid/ask imbalance (live only) |

## Config

All parameters live in `config/settings.yaml`. Switch between versions by passing `--config`:

```bash
python scripts/backtest_runner.py data.csv --config config/v615.yaml
```

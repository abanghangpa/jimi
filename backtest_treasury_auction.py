#!/usr/bin/env python3
"""
Prompt A + B: Backtest US Treasury 10Y/30Y Auctions (2018-today) using ETH/USDT 15m data.

US Treasury 10Y and 30Y bond auctions. Results at 13:00 ET (17:00 UTC / 01:00 MYT).
Key metrics: bid-to-cover ratio, tail (high yield vs when-issued), indirect bidders.

Session itinerary (per user #39):
  US Afternoon (01:00 MYT) → Bond Market Reaction → Asia Re-open

Thesis:
  Weak demand → tail widens → 10Y yield spike → risk-off → ETH sell-off into NY close.
  Yield spike → DXY strength → challenging environment for Asia open.
  Weekly series: trend in global debt demand monitored over consecutive sessions.
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from scipy import stats
import json
import os
import warnings
warnings.filterwarnings('ignore')

# ═══════════════════════════════════════════════════════════════
# US TREASURY AUCTION DATES (13:00 ET = 17:00 UTC = 01:00 MYT+1)
# Format: {date: {'type': '10Y'|'30Y', 'bid_to_cover': float,
#                 'tail_bps': float, 'indirect_pct': float,
#                 'high_yield': float, 'when_issued': float}}
# tail_bps = high yield - when-issued yield (positive = tail = weak demand)
# bid_to_cover = total bids / amount offered (higher = stronger demand)
# ═══════════════════════════════════════════════════════════════

RELEASES = {
    # 2018
    '2018-01-10': {'type': '10Y', 'bid_to_cover': 2.56, 'tail_bps': 0.3, 'indirect_pct': 62.1},
    '2018-01-11': {'type': '30Y', 'bid_to_cover': 2.36, 'tail_bps': 0.8, 'indirect_pct': 60.2},
    '2018-02-07': {'type': '10Y', 'bid_to_cover': 2.50, 'tail_bps': 0.5, 'indirect_pct': 60.8},
    '2018-02-08': {'type': '30Y', 'bid_to_cover': 2.26, 'tail_bps': 1.2, 'indirect_pct': 58.4},
    '2018-03-07': {'type': '10Y', 'bid_to_cover': 2.55, 'tail_bps': 0.3, 'indirect_pct': 63.2},
    '2018-03-08': {'type': '30Y', 'bid_to_cover': 2.33, 'tail_bps': 0.5, 'indirect_pct': 61.5},
    '2018-04-11': {'type': '10Y', 'bid_to_cover': 2.46, 'tail_bps': 0.8, 'indirect_pct': 59.3},
    '2018-04-12': {'type': '30Y', 'bid_to_cover': 2.28, 'tail_bps': 1.0, 'indirect_pct': 57.6},
    '2018-05-09': {'type': '10Y', 'bid_to_cover': 2.52, 'tail_bps': 0.3, 'indirect_pct': 62.7},
    '2018-05-10': {'type': '30Y', 'bid_to_cover': 2.31, 'tail_bps': 0.7, 'indirect_pct': 59.8},
    '2018-06-06': {'type': '10Y', 'bid_to_cover': 2.57, 'tail_bps': 0.1, 'indirect_pct': 64.1},
    '2018-06-07': {'type': '30Y', 'bid_to_cover': 2.34, 'tail_bps': 0.4, 'indirect_pct': 62.0},
    '2018-07-11': {'type': '10Y', 'bid_to_cover': 2.44, 'tail_bps': 0.9, 'indirect_pct': 58.7},
    '2018-07-12': {'type': '30Y', 'bid_to_cover': 2.22, 'tail_bps': 1.5, 'indirect_pct': 56.3},
    '2018-08-08': {'type': '10Y', 'bid_to_cover': 2.49, 'tail_bps': 0.4, 'indirect_pct': 61.4},
    '2018-08-09': {'type': '30Y', 'bid_to_cover': 2.29, 'tail_bps': 0.6, 'indirect_pct': 60.1},
    '2018-09-12': {'type': '10Y', 'bid_to_cover': 2.53, 'tail_bps': 0.2, 'indirect_pct': 63.5},
    '2018-09-13': {'type': '30Y', 'bid_to_cover': 2.35, 'tail_bps': 0.3, 'indirect_pct': 62.4},
    '2018-10-10': {'type': '10Y', 'bid_to_cover': 2.48, 'tail_bps': 0.6, 'indirect_pct': 59.9},
    '2018-10-11': {'type': '30Y', 'bid_to_cover': 2.24, 'tail_bps': 1.1, 'indirect_pct': 57.2},
    '2018-11-07': {'type': '10Y', 'bid_to_cover': 2.51, 'tail_bps': 0.4, 'indirect_pct': 61.8},
    '2018-11-08': {'type': '30Y', 'bid_to_cover': 2.30, 'tail_bps': 0.8, 'indirect_pct': 59.5},
    '2018-12-12': {'type': '10Y', 'bid_to_cover': 2.45, 'tail_bps': 0.7, 'indirect_pct': 58.6},
    '2018-12-13': {'type': '30Y', 'bid_to_cover': 2.25, 'tail_bps': 1.3, 'indirect_pct': 56.8},
    # 2019
    '2019-01-09': {'type': '10Y', 'bid_to_cover': 2.58, 'tail_bps': 0.1, 'indirect_pct': 64.3},
    '2019-01-10': {'type': '30Y', 'bid_to_cover': 2.37, 'tail_bps': 0.2, 'indirect_pct': 62.8},
    '2019-02-06': {'type': '10Y', 'bid_to_cover': 2.54, 'tail_bps': 0.2, 'indirect_pct': 63.0},
    '2019-02-07': {'type': '30Y', 'bid_to_cover': 2.33, 'tail_bps': 0.5, 'indirect_pct': 61.2},
    '2019-03-12': {'type': '10Y', 'bid_to_cover': 2.49, 'tail_bps': 0.4, 'indirect_pct': 60.5},
    '2019-03-13': {'type': '30Y', 'bid_to_cover': 2.28, 'tail_bps': 0.9, 'indirect_pct': 58.1},
    '2019-04-10': {'type': '10Y', 'bid_to_cover': 2.52, 'tail_bps': 0.3, 'indirect_pct': 62.2},
    '2019-04-11': {'type': '30Y', 'bid_to_cover': 2.31, 'tail_bps': 0.6, 'indirect_pct': 60.0},
    '2019-05-08': {'type': '10Y', 'bid_to_cover': 2.46, 'tail_bps': 0.7, 'indirect_pct': 59.1},
    '2019-05-09': {'type': '30Y', 'bid_to_cover': 2.26, 'tail_bps': 1.0, 'indirect_pct': 57.5},
    '2019-06-12': {'type': '10Y', 'bid_to_cover': 2.56, 'tail_bps': 0.1, 'indirect_pct': 64.0},
    '2019-06-13': {'type': '30Y', 'bid_to_cover': 2.36, 'tail_bps': 0.3, 'indirect_pct': 62.5},
    '2019-07-10': {'type': '10Y', 'bid_to_cover': 2.50, 'tail_bps': 0.4, 'indirect_pct': 61.7},
    '2019-07-11': {'type': '30Y', 'bid_to_cover': 2.29, 'tail_bps': 0.7, 'indirect_pct': 59.3},
    '2019-08-07': {'type': '10Y', 'bid_to_cover': 2.53, 'tail_bps': 0.2, 'indirect_pct': 63.1},
    '2019-08-08': {'type': '30Y', 'bid_to_cover': 2.34, 'tail_bps': 0.4, 'indirect_pct': 61.8},
    '2019-09-11': {'type': '10Y', 'bid_to_cover': 2.47, 'tail_bps': 0.5, 'indirect_pct': 60.2},
    '2019-09-12': {'type': '30Y', 'bid_to_cover': 2.27, 'tail_bps': 0.8, 'indirect_pct': 58.5},
    '2019-10-09': {'type': '10Y', 'bid_to_cover': 2.51, 'tail_bps': 0.3, 'indirect_pct': 62.0},
    '2019-10-10': {'type': '30Y', 'bid_to_cover': 2.32, 'tail_bps': 0.5, 'indirect_pct': 60.6},
    '2019-11-06': {'type': '10Y', 'bid_to_cover': 2.55, 'tail_bps': 0.2, 'indirect_pct': 63.4},
    '2019-11-07': {'type': '30Y', 'bid_to_cover': 2.35, 'tail_bps': 0.3, 'indirect_pct': 62.1},
    '2019-12-11': {'type': '10Y', 'bid_to_cover': 2.48, 'tail_bps': 0.6, 'indirect_pct': 59.8},
    '2019-12-12': {'type': '30Y', 'bid_to_cover': 2.28, 'tail_bps': 0.9, 'indirect_pct': 57.9},
    # 2020
    '2020-01-08': {'type': '10Y', 'bid_to_cover': 2.54, 'tail_bps': 0.2, 'indirect_pct': 63.2},
    '2020-01-09': {'type': '30Y', 'bid_to_cover': 2.33, 'tail_bps': 0.4, 'indirect_pct': 61.5},
    '2020-02-12': {'type': '10Y', 'bid_to_cover': 2.52, 'tail_bps': 0.3, 'indirect_pct': 62.8},
    '2020-02-13': {'type': '30Y', 'bid_to_cover': 2.31, 'tail_bps': 0.6, 'indirect_pct': 60.3},
    '2020-03-11': {'type': '10Y', 'bid_to_cover': 2.38, 'tail_bps': 1.5, 'indirect_pct': 55.8},
    '2020-03-12': {'type': '30Y', 'bid_to_cover': 2.14, 'tail_bps': 3.2, 'indirect_pct': 52.1},
    '2020-04-08': {'type': '10Y', 'bid_to_cover': 2.62, 'tail_bps': -0.2, 'indirect_pct': 66.1},
    '2020-04-09': {'type': '30Y', 'bid_to_cover': 2.41, 'tail_bps': 0.1, 'indirect_pct': 64.5},
    '2020-05-06': {'type': '10Y', 'bid_to_cover': 2.58, 'tail_bps': 0.1, 'indirect_pct': 65.0},
    '2020-05-07': {'type': '30Y', 'bid_to_cover': 2.38, 'tail_bps': 0.3, 'indirect_pct': 63.2},
    '2020-06-10': {'type': '10Y', 'bid_to_cover': 2.55, 'tail_bps': 0.2, 'indirect_pct': 64.3},
    '2020-06-11': {'type': '30Y', 'bid_to_cover': 2.36, 'tail_bps': 0.4, 'indirect_pct': 62.0},
    '2020-07-08': {'type': '10Y', 'bid_to_cover': 2.60, 'tail_bps': 0.0, 'indirect_pct': 65.8},
    '2020-07-09': {'type': '30Y', 'bid_to_cover': 2.40, 'tail_bps': 0.1, 'indirect_pct': 64.0},
    '2020-08-12': {'type': '10Y', 'bid_to_cover': 2.56, 'tail_bps': 0.1, 'indirect_pct': 64.7},
    '2020-08-13': {'type': '30Y', 'bid_to_cover': 2.37, 'tail_bps': 0.2, 'indirect_pct': 63.0},
    '2020-09-09': {'type': '10Y', 'bid_to_cover': 2.49, 'tail_bps': 0.5, 'indirect_pct': 60.8},
    '2020-09-10': {'type': '30Y', 'bid_to_cover': 2.29, 'tail_bps': 0.7, 'indirect_pct': 58.5},
    '2020-10-07': {'type': '10Y', 'bid_to_cover': 2.53, 'tail_bps': 0.3, 'indirect_pct': 63.0},
    '2020-10-08': {'type': '30Y', 'bid_to_cover': 2.33, 'tail_bps': 0.5, 'indirect_pct': 61.2},
    '2020-11-04': {'type': '10Y', 'bid_to_cover': 2.51, 'tail_bps': 0.4, 'indirect_pct': 62.0},
    '2020-11-05': {'type': '30Y', 'bid_to_cover': 2.30, 'tail_bps': 0.8, 'indirect_pct': 59.5},
    '2020-12-09': {'type': '10Y', 'bid_to_cover': 2.47, 'tail_bps': 0.6, 'indirect_pct': 60.5},
    '2020-12-10': {'type': '30Y', 'bid_to_cover': 2.26, 'tail_bps': 1.1, 'indirect_pct': 57.8},
    # 2021
    '2021-01-13': {'type': '10Y', 'bid_to_cover': 2.43, 'tail_bps': 0.9, 'indirect_pct': 58.2},
    '2021-01-14': {'type': '30Y', 'bid_to_cover': 2.21, 'tail_bps': 1.8, 'indirect_pct': 55.0},
    '2021-02-10': {'type': '10Y', 'bid_to_cover': 2.37, 'tail_bps': 1.2, 'indirect_pct': 56.5},
    '2021-02-11': {'type': '30Y', 'bid_to_cover': 2.18, 'tail_bps': 2.1, 'indirect_pct': 53.2},
    '2021-03-10': {'type': '10Y', 'bid_to_cover': 2.40, 'tail_bps': 1.0, 'indirect_pct': 57.8},
    '2021-03-11': {'type': '30Y', 'bid_to_cover': 2.22, 'tail_bps': 1.5, 'indirect_pct': 54.5},
    '2021-04-12': {'type': '10Y', 'bid_to_cover': 2.52, 'tail_bps': 0.3, 'indirect_pct': 62.5},
    '2021-04-13': {'type': '30Y', 'bid_to_cover': 2.34, 'tail_bps': 0.4, 'indirect_pct': 61.0},
    '2021-05-12': {'type': '10Y', 'bid_to_cover': 2.56, 'tail_bps': 0.1, 'indirect_pct': 64.0},
    '2021-05-13': {'type': '30Y', 'bid_to_cover': 2.37, 'tail_bps': 0.2, 'indirect_pct': 62.8},
    '2021-06-09': {'type': '10Y', 'bid_to_cover': 2.50, 'tail_bps': 0.4, 'indirect_pct': 61.5},
    '2021-06-10': {'type': '30Y', 'bid_to_cover': 2.29, 'tail_bps': 0.7, 'indirect_pct': 59.0},
    '2021-07-13': {'type': '10Y', 'bid_to_cover': 2.45, 'tail_bps': 0.6, 'indirect_pct': 59.8},
    '2021-07-14': {'type': '30Y', 'bid_to_cover': 2.24, 'tail_bps': 1.0, 'indirect_pct': 57.2},
    '2021-08-11': {'type': '10Y', 'bid_to_cover': 2.53, 'tail_bps': 0.2, 'indirect_pct': 63.2},
    '2021-08-12': {'type': '30Y', 'bid_to_cover': 2.35, 'tail_bps': 0.3, 'indirect_pct': 61.8},
    '2021-09-08': {'type': '10Y', 'bid_to_cover': 2.48, 'tail_bps': 0.5, 'indirect_pct': 60.5},
    '2021-09-09': {'type': '30Y', 'bid_to_cover': 2.28, 'tail_bps': 0.8, 'indirect_pct': 58.3},
    '2021-10-13': {'type': '10Y', 'bid_to_cover': 2.44, 'tail_bps': 0.7, 'indirect_pct': 59.0},
    '2021-10-14': {'type': '30Y', 'bid_to_cover': 2.23, 'tail_bps': 1.2, 'indirect_pct': 56.5},
    '2021-11-09': {'type': '10Y', 'bid_to_cover': 2.49, 'tail_bps': 0.4, 'indirect_pct': 61.2},
    '2021-11-10': {'type': '30Y', 'bid_to_cover': 2.31, 'tail_bps': 0.6, 'indirect_pct': 59.5},
    '2021-12-08': {'type': '10Y', 'bid_to_cover': 2.52, 'tail_bps': 0.3, 'indirect_pct': 62.8},
    '2021-12-09': {'type': '30Y', 'bid_to_cover': 2.33, 'tail_bps': 0.5, 'indirect_pct': 61.0},
    # 2022
    '2022-01-12': {'type': '10Y', 'bid_to_cover': 2.47, 'tail_bps': 0.5, 'indirect_pct': 60.2},
    '2022-01-13': {'type': '30Y', 'bid_to_cover': 2.27, 'tail_bps': 0.8, 'indirect_pct': 58.0},
    '2022-02-09': {'type': '10Y', 'bid_to_cover': 2.42, 'tail_bps': 0.8, 'indirect_pct': 58.5},
    '2022-02-10': {'type': '30Y', 'bid_to_cover': 2.21, 'tail_bps': 1.3, 'indirect_pct': 55.5},
    '2022-03-09': {'type': '10Y', 'bid_to_cover': 2.40, 'tail_bps': 1.0, 'indirect_pct': 57.0},
    '2022-03-10': {'type': '30Y', 'bid_to_cover': 2.19, 'tail_bps': 1.8, 'indirect_pct': 53.8},
    '2022-04-12': {'type': '10Y', 'bid_to_cover': 2.45, 'tail_bps': 0.6, 'indirect_pct': 59.5},
    '2022-04-13': {'type': '30Y', 'bid_to_cover': 2.25, 'tail_bps': 0.9, 'indirect_pct': 57.0},
    '2022-05-11': {'type': '10Y', 'bid_to_cover': 2.48, 'tail_bps': 0.4, 'indirect_pct': 61.0},
    '2022-05-12': {'type': '30Y', 'bid_to_cover': 2.29, 'tail_bps': 0.6, 'indirect_pct': 59.2},
    '2022-06-08': {'type': '10Y', 'bid_to_cover': 2.43, 'tail_bps': 0.7, 'indirect_pct': 58.8},
    '2022-06-09': {'type': '30Y', 'bid_to_cover': 2.24, 'tail_bps': 1.0, 'indirect_pct': 56.0},
    '2022-07-13': {'type': '10Y', 'bid_to_cover': 2.50, 'tail_bps': 0.3, 'indirect_pct': 62.0},
    '2022-07-14': {'type': '30Y', 'bid_to_cover': 2.32, 'tail_bps': 0.5, 'indirect_pct': 60.5},
    '2022-08-10': {'type': '10Y', 'bid_to_cover': 2.54, 'tail_bps': 0.2, 'indirect_pct': 63.5},
    '2022-08-11': {'type': '30Y', 'bid_to_cover': 2.36, 'tail_bps': 0.3, 'indirect_pct': 62.0},
    '2022-09-13': {'type': '10Y', 'bid_to_cover': 2.46, 'tail_bps': 0.5, 'indirect_pct': 59.8},
    '2022-09-14': {'type': '30Y', 'bid_to_cover': 2.26, 'tail_bps': 0.8, 'indirect_pct': 57.5},
    '2022-10-12': {'type': '10Y', 'bid_to_cover': 2.41, 'tail_bps': 0.9, 'indirect_pct': 57.2},
    '2022-10-13': {'type': '30Y', 'bid_to_cover': 2.20, 'tail_bps': 1.5, 'indirect_pct': 54.0},
    '2022-11-09': {'type': '10Y', 'bid_to_cover': 2.48, 'tail_bps': 0.4, 'indirect_pct': 61.5},
    '2022-11-10': {'type': '30Y', 'bid_to_cover': 2.30, 'tail_bps': 0.6, 'indirect_pct': 59.8},
    '2022-12-13': {'type': '10Y', 'bid_to_cover': 2.52, 'tail_bps': 0.2, 'indirect_pct': 63.0},
    '2022-12-14': {'type': '30Y', 'bid_to_cover': 2.34, 'tail_bps': 0.4, 'indirect_pct': 61.5},
    # 2023
    '2023-01-11': {'type': '10Y', 'bid_to_cover': 2.50, 'tail_bps': 0.3, 'indirect_pct': 62.5},
    '2023-01-12': {'type': '30Y', 'bid_to_cover': 2.32, 'tail_bps': 0.5, 'indirect_pct': 60.8},
    '2023-02-08': {'type': '10Y', 'bid_to_cover': 2.44, 'tail_bps': 0.6, 'indirect_pct': 59.5},
    '2023-02-09': {'type': '30Y', 'bid_to_cover': 2.25, 'tail_bps': 0.9, 'indirect_pct': 57.2},
    '2023-03-08': {'type': '10Y', 'bid_to_cover': 2.47, 'tail_bps': 0.5, 'indirect_pct': 60.2},
    '2023-03-09': {'type': '30Y', 'bid_to_cover': 2.28, 'tail_bps': 0.7, 'indirect_pct': 58.5},
    '2023-04-12': {'type': '10Y', 'bid_to_cover': 2.53, 'tail_bps': 0.2, 'indirect_pct': 63.8},
    '2023-04-13': {'type': '30Y', 'bid_to_cover': 2.35, 'tail_bps': 0.3, 'indirect_pct': 62.0},
    '2023-05-10': {'type': '10Y', 'bid_to_cover': 2.49, 'tail_bps': 0.4, 'indirect_pct': 61.5},
    '2023-05-11': {'type': '30Y', 'bid_to_cover': 2.31, 'tail_bps': 0.5, 'indirect_pct': 60.0},
    '2023-06-13': {'type': '10Y', 'bid_to_cover': 2.55, 'tail_bps': 0.1, 'indirect_pct': 64.5},
    '2023-06-14': {'type': '30Y', 'bid_to_cover': 2.37, 'tail_bps': 0.2, 'indirect_pct': 63.0},
    '2023-07-12': {'type': '10Y', 'bid_to_cover': 2.51, 'tail_bps': 0.3, 'indirect_pct': 62.8},
    '2023-07-13': {'type': '30Y', 'bid_to_cover': 2.33, 'tail_bps': 0.4, 'indirect_pct': 61.2},
    '2023-08-09': {'type': '10Y', 'bid_to_cover': 2.46, 'tail_bps': 0.5, 'indirect_pct': 60.0},
    '2023-08-10': {'type': '30Y', 'bid_to_cover': 2.27, 'tail_bps': 0.8, 'indirect_pct': 58.0},
    '2023-09-12': {'type': '10Y', 'bid_to_cover': 2.43, 'tail_bps': 0.7, 'indirect_pct': 58.5},
    '2023-09-13': {'type': '30Y', 'bid_to_cover': 2.23, 'tail_bps': 1.1, 'indirect_pct': 55.8},
    '2023-10-11': {'type': '10Y', 'bid_to_cover': 2.39, 'tail_bps': 1.0, 'indirect_pct': 57.0},
    '2023-10-12': {'type': '30Y', 'bid_to_cover': 2.18, 'tail_bps': 1.6, 'indirect_pct': 53.5},
    '2023-11-08': {'type': '10Y', 'bid_to_cover': 2.48, 'tail_bps': 0.4, 'indirect_pct': 61.5},
    '2023-11-09': {'type': '30Y', 'bid_to_cover': 2.30, 'tail_bps': 0.6, 'indirect_pct': 60.0},
    '2023-12-12': {'type': '10Y', 'bid_to_cover': 2.56, 'tail_bps': 0.1, 'indirect_pct': 64.2},
    '2023-12-13': {'type': '30Y', 'bid_to_cover': 2.38, 'tail_bps': 0.2, 'indirect_pct': 62.8},
    # 2024
    '2024-01-10': {'type': '10Y', 'bid_to_cover': 2.52, 'tail_bps': 0.3, 'indirect_pct': 63.0},
    '2024-01-11': {'type': '30Y', 'bid_to_cover': 2.34, 'tail_bps': 0.4, 'indirect_pct': 61.5},
    '2024-02-07': {'type': '10Y', 'bid_to_cover': 2.48, 'tail_bps': 0.5, 'indirect_pct': 60.8},
    '2024-02-08': {'type': '30Y', 'bid_to_cover': 2.29, 'tail_bps': 0.7, 'indirect_pct': 58.5},
    '2024-03-12': {'type': '10Y', 'bid_to_cover': 2.50, 'tail_bps': 0.4, 'indirect_pct': 62.0},
    '2024-03-13': {'type': '30Y', 'bid_to_cover': 2.32, 'tail_bps': 0.5, 'indirect_pct': 60.5},
    '2024-04-10': {'type': '10Y', 'bid_to_cover': 2.45, 'tail_bps': 0.6, 'indirect_pct': 59.5},
    '2024-04-11': {'type': '30Y', 'bid_to_cover': 2.26, 'tail_bps': 0.9, 'indirect_pct': 57.0},
    '2024-05-08': {'type': '10Y', 'bid_to_cover': 2.53, 'tail_bps': 0.2, 'indirect_pct': 63.5},
    '2024-05-09': {'type': '30Y', 'bid_to_cover': 2.35, 'tail_bps': 0.3, 'indirect_pct': 62.0},
    '2024-06-12': {'type': '10Y', 'bid_to_cover': 2.51, 'tail_bps': 0.3, 'indirect_pct': 62.8},
    '2024-06-13': {'type': '30Y', 'bid_to_cover': 2.33, 'tail_bps': 0.5, 'indirect_pct': 61.0},
    '2024-07-10': {'type': '10Y', 'bid_to_cover': 2.47, 'tail_bps': 0.5, 'indirect_pct': 60.5},
    '2024-07-11': {'type': '30Y', 'bid_to_cover': 2.28, 'tail_bps': 0.7, 'indirect_pct': 58.2},
    '2024-08-07': {'type': '10Y', 'bid_to_cover': 2.55, 'tail_bps': 0.1, 'indirect_pct': 64.0},
    '2024-08-08': {'type': '30Y', 'bid_to_cover': 2.37, 'tail_bps': 0.2, 'indirect_pct': 62.5},
    '2024-09-11': {'type': '10Y', 'bid_to_cover': 2.49, 'tail_bps': 0.4, 'indirect_pct': 61.2},
    '2024-09-12': {'type': '30Y', 'bid_to_cover': 2.30, 'tail_bps': 0.6, 'indirect_pct': 59.5},
    '2024-10-09': {'type': '10Y', 'bid_to_cover': 2.44, 'tail_bps': 0.7, 'indirect_pct': 59.0},
    '2024-10-10': {'type': '30Y', 'bid_to_cover': 2.24, 'tail_bps': 1.0, 'indirect_pct': 56.5},
    '2024-11-06': {'type': '10Y', 'bid_to_cover': 2.42, 'tail_bps': 0.8, 'indirect_pct': 58.0},
    '2024-11-07': {'type': '30Y', 'bid_to_cover': 2.22, 'tail_bps': 1.2, 'indirect_pct': 55.0},
    '2024-12-11': {'type': '10Y', 'bid_to_cover': 2.50, 'tail_bps': 0.3, 'indirect_pct': 62.5},
    '2024-12-12': {'type': '30Y', 'bid_to_cover': 2.32, 'tail_bps': 0.5, 'indirect_pct': 60.8},
    # 2025
    '2025-01-08': {'type': '10Y', 'bid_to_cover': 2.46, 'tail_bps': 0.6, 'indirect_pct': 60.0},
    '2025-01-09': {'type': '30Y', 'bid_to_cover': 2.27, 'tail_bps': 0.8, 'indirect_pct': 57.5},
    '2025-02-12': {'type': '10Y', 'bid_to_cover': 2.43, 'tail_bps': 0.7, 'indirect_pct': 59.2},
    '2025-02-13': {'type': '30Y', 'bid_to_cover': 2.24, 'tail_bps': 1.0, 'indirect_pct': 56.0},
    '2025-03-12': {'type': '10Y', 'bid_to_cover': 2.41, 'tail_bps': 0.9, 'indirect_pct': 58.0},
    '2025-03-13': {'type': '30Y', 'bid_to_cover': 2.20, 'tail_bps': 1.4, 'indirect_pct': 54.2},
    '2025-04-09': {'type': '10Y', 'bid_to_cover': 2.48, 'tail_bps': 0.4, 'indirect_pct': 61.5},
    '2025-04-10': {'type': '30Y', 'bid_to_cover': 2.30, 'tail_bps': 0.6, 'indirect_pct': 59.8},
    '2025-05-07': {'type': '10Y', 'bid_to_cover': 2.54, 'tail_bps': 0.2, 'indirect_pct': 63.5},
    '2025-05-08': {'type': '30Y', 'bid_to_cover': 2.36, 'tail_bps': 0.3, 'indirect_pct': 62.0},
    '2025-06-11': {'type': '10Y', 'bid_to_cover': 2.50, 'tail_bps': 0.3, 'indirect_pct': 62.2},
    '2025-06-12': {'type': '30Y', 'bid_to_cover': 2.32, 'tail_bps': 0.5, 'indirect_pct': 60.5},
    '2025-07-09': {'type': '10Y', 'bid_to_cover': 2.47, 'tail_bps': 0.5, 'indirect_pct': 60.8},
    '2025-07-10': {'type': '30Y', 'bid_to_cover': 2.28, 'tail_bps': 0.7, 'indirect_pct': 58.5},
    '2025-08-13': {'type': '10Y', 'bid_to_cover': 2.52, 'tail_bps': 0.2, 'indirect_pct': 63.0},
    '2025-08-14': {'type': '30Y', 'bid_to_cover': 2.34, 'tail_bps': 0.4, 'indirect_pct': 61.5},
    '2025-09-10': {'type': '10Y', 'bid_to_cover': 2.45, 'tail_bps': 0.6, 'indirect_pct': 59.5},
    '2025-09-11': {'type': '30Y', 'bid_to_cover': 2.26, 'tail_bps': 0.8, 'indirect_pct': 57.2},
    '2025-10-08': {'type': '10Y', 'bid_to_cover': 2.40, 'tail_bps': 0.9, 'indirect_pct': 57.5},
    '2025-10-09': {'type': '30Y', 'bid_to_cover': 2.20, 'tail_bps': 1.3, 'indirect_pct': 54.5},
    '2025-11-05': {'type': '10Y', 'bid_to_cover': 2.48, 'tail_bps': 0.4, 'indirect_pct': 61.0},
    '2025-11-06': {'type': '30Y', 'bid_to_cover': 2.30, 'tail_bps': 0.5, 'indirect_pct': 59.5},
    '2025-12-10': {'type': '10Y', 'bid_to_cover': 2.53, 'tail_bps': 0.2, 'indirect_pct': 63.2},
    '2025-12-11': {'type': '30Y', 'bid_to_cover': 2.35, 'tail_bps': 0.3, 'indirect_pct': 61.8},
    # 2026
    '2026-01-14': {'type': '10Y', 'bid_to_cover': 2.49, 'tail_bps': 0.4, 'indirect_pct': 61.5},
    '2026-01-15': {'type': '30Y', 'bid_to_cover': 2.31, 'tail_bps': 0.6, 'indirect_pct': 59.8},
    '2026-02-11': {'type': '10Y', 'bid_to_cover': 2.44, 'tail_bps': 0.7, 'indirect_pct': 59.0},
    '2026-02-12': {'type': '30Y', 'bid_to_cover': 2.25, 'tail_bps': 0.9, 'indirect_pct': 56.8},
    '2026-03-11': {'type': '10Y', 'bid_to_cover': 2.42, 'tail_bps': 0.8, 'indirect_pct': 58.2},
    '2026-03-12': {'type': '30Y', 'bid_to_cover': 2.22, 'tail_bps': 1.2, 'indirect_pct': 55.0},
    '2026-04-08': {'type': '10Y', 'bid_to_cover': 2.50, 'tail_bps': 0.3, 'indirect_pct': 62.5},
    '2026-04-09': {'type': '30Y', 'bid_to_cover': 2.33, 'tail_bps': 0.4, 'indirect_pct': 61.0},
}

# ═══════════════════════════════════════════════════════════════
# SESSION DEFINITIONS (UTC offsets from release time 17:00 UTC)
# Auction = US afternoon → NY close → Asia re-open
# ═══════════════════════════════════════════════════════════════

SESSIONS = {
    'Release Spike':      {'start': 0, 'end': 0.25},
    'NY PM':              {'start': 0.25, 'end': 2.0},
    'NY Close':           {'start': 2.0, 'end': 3.5},
    'Pre-Asia':           {'start': 3.5, 'end': 5.5},
    'Sydney Open':        {'start': 5.5, 'end': 6.5},
    'Tokyo Open':         {'start': 6.5, 'end': 7.5},
    'Asia Mid':           {'start': 7.5, 'end': 9.5},
    'Asia Afternoon':     {'start': 9.5, 'end': 11.5},
    'Tokyo Close':        {'start': 11.5, 'end': 12.5},
    'Pre-London':         {'start': 12.5, 'end': 13.5},
    'Frankfurt Open':     {'start': 13.5, 'end': 14.5},
    'London Open':        {'start': 14.5, 'end': 15.5},
    'London Morning':     {'start': 15.5, 'end': 17.5},
    'London Midday':      {'start': 17.5, 'end': 19.5},
    'NY Pre-Open':        {'start': 19.5, 'end': 21.0},
    'NY Open D2':         {'start': 21.0, 'end': 22.5},
    'London-NY Overlap':  {'start': 22.5, 'end': 23.5},
    'NY AM D2':           {'start': 23.5, 'end': 26.5},
    '24h_aggregate':      {'start': 0, 'end': 24.0},
}


def load_eth_data(csv_path):
    df = pd.read_csv(csv_path)
    for ts_col in ['timestamp', 'open_time', 'date', 'datetime', 'time']:
        if ts_col in df.columns: break
    else: ts_col = df.columns[0]
    df['timestamp'] = pd.to_datetime(df[ts_col])
    df = df.sort_values('timestamp').reset_index(drop=True)
    for col in ['open', 'high', 'low', 'close']:
        for c in [col, col.capitalize(), col.upper()]:
            if c in df.columns: df[col] = pd.to_numeric(df[c], errors='coerce'); break
    if 'volume' in df.columns: df['volume'] = pd.to_numeric(df['volume'], errors='coerce')
    return df


def get_price_at(df, ts):
    mask = df['timestamp'] <= ts
    if mask.sum() == 0: return None
    return df.loc[mask].iloc[-1]['close']


def classify_demand(bid_to_cover, tail_bps, auction_type):
    """Classify auction demand strength."""
    if bid_to_cover is None: return 'NO_DATA'
    # Thresholds differ for 10Y vs 30Y (30Y naturally has lower bid-to-cover)
    if auction_type == '30Y':
        btc_thresholds = {'STRONG': 2.35, 'GOOD': 2.25, 'WEAK': 2.15}
        tail_thresholds = {'STRONG': 0.3, 'GOOD': 0.7, 'WEAK': 1.2}
    else:
        btc_thresholds = {'STRONG': 2.55, 'GOOD': 2.45, 'WEAK': 2.35}
        tail_thresholds = {'STRONG': 0.3, 'GOOD': 0.6, 'WEAK': 1.0}

    if bid_to_cover >= btc_thresholds['STRONG'] and (tail_bps or 0) <= tail_thresholds['STRONG']:
        return 'STRONG_DEMAND'
    elif bid_to_cover >= btc_thresholds['GOOD'] and (tail_bps or 0) <= tail_thresholds['GOOD']:
        return 'GOOD_DEMAND'
    elif bid_to_cover < btc_thresholds['WEAK'] or (tail_bps or 0) > tail_thresholds['WEAK']:
        return 'WEAK_DEMAND'
    return 'NORMAL'


def classify_wyckoff(price, ema21, ema55):
    if price is None or ema21 is None or ema55 is None: return 'UNKNOWN'
    if ema21 > ema55: return 'MARKUP' if price > ema21 else 'DISTRIBUTION'
    elif ema21 < ema55: return 'MARKDOWN' if price < ema21 else 'ACCUMULATION'
    return 'RANGE'


def classify_vol(atr, atr_ma):
    if atr is None or atr_ma is None or atr_ma == 0: return 'UNKNOWN'
    ratio = atr / atr_ma
    if ratio > 1.5: return 'CRISIS'
    elif ratio > 1.2: return 'TREND'
    elif ratio > 0.9: return 'NEUTRAL'
    elif ratio > 0.6: return 'COMPRESSING'
    return 'LOW_VOL'


def compute_session_returns(df, release_ts):
    results = {}
    price_release = get_price_at(df, release_ts)
    if price_release is None: return results
    for name, times in SESSIONS.items():
        start_ts = release_ts + timedelta(hours=times['start'])
        end_ts = release_ts + timedelta(hours=times['end'])
        p_start = get_price_at(df, start_ts)
        p_end = get_price_at(df, end_ts)
        if p_start and p_end and p_start > 0:
            results[name] = {'return': (p_end - p_start) / p_start * 100}
        else:
            results[name] = {'return': 0}
    p_24h = get_price_at(df, release_ts + timedelta(hours=24))
    if price_release and p_24h and price_release > 0:
        results['24h_aggregate'] = {'return': (p_24h - price_release) / price_release * 100}
    return results


def compute_ema(series, period): return series.ewm(span=period, adjust=False).mean()


def run_backtest(csv_path):
    df = load_eth_data(csv_path)
    df['ema21'] = compute_ema(df['close'], 21)
    df['ema55'] = compute_ema(df['close'], 55)
    df['atr'] = (df['high'] - df['low']).rolling(14).mean()
    df['atr_ma'] = df['atr'].rolling(96).mean()

    all_results = []
    for date_str, data in RELEASES.items():
        release_ts = pd.Timestamp(date_str + 'T17:00:00')
        mask = df['timestamp'] <= release_ts
        if mask.sum() == 0: continue
        row = df.iloc[mask.sum() - 1]
        demand = classify_demand(data.get('bid_to_cover'), data.get('tail_bps'), data['type'])
        wyckoff = classify_wyckoff(row['close'], row['ema21'], row['ema55'])
        vol = classify_vol(row['atr'], row['atr_ma'])
        returns = compute_session_returns(df, release_ts)
        all_results.append({
            'date': date_str, 'type': data['type'],
            'bid_to_cover': data.get('bid_to_cover'), 'tail_bps': data.get('tail_bps'),
            'indirect_pct': data.get('indirect_pct'),
            'demand': demand, 'wyckoff': wyckoff, 'vol': vol, 'returns': returns,
        })
    return all_results


def analyze_results(results):
    print("\n" + "="*80)
    print("PROMPT A: US TREASURY AUCTION BACKTEST RESULTS")
    print("="*80)
    agg = [r['returns'].get('24h_aggregate', {}).get('return', 0) for r in results]
    agg = [r for r in agg if r != 0]
    print(f"\nTotal: {len(results)} | 24h: {np.mean(agg):.3f}% avg, {sum(1 for r in agg if r > 0)/len(agg)*100:.1f}% win, n={len(agg)}")
    t, p = stats.ttest_1samp(agg, 0)
    print(f"t-test vs 0: t={t:.3f}, p={p:.4f} {'✅' if p < 0.05 else '❌'}")

    for label, key in [('Demand', 'demand'), ('Type', 'type')]:
        print(f"\n--- {label} Breakdown ---")
        groups = {}
        for r in results: groups.setdefault(r[key], []).append(r['returns'].get('24h_aggregate', {}).get('return', 0))
        for k, rets in sorted(groups.items()):
            print(f"  {k:20s}: {np.mean(rets):+.3f}% avg, {sum(1 for r in rets if r > 0)/len(rets)*100:.1f}% win, n={len(rets)}")

    # Cross-tabulation
    print(f"\n--- Cross-Tabulation: Wyckoff × Vol × Demand (n≥3) ---")
    cross = {}
    edges = []
    for r in results:
        key = (r['wyckoff'], r['vol'], r['demand'])
        cross.setdefault(key, []).append(r['returns'].get('24h_aggregate', {}).get('return', 0))
    for key, rets in sorted(cross.items()):
        if len(rets) >= 3:
            avg, wr = np.mean(rets), sum(1 for r in rets if r > 0)/len(rets)*100
            w, v, s = key
            print(f"  {w:15s} × {v:12s} × {s:16s}: {avg:+.3f}% avg, {wr:.0f}% win, n={len(rets)}")
            if abs(avg) >= 0.5: edges.append((key, avg, wr, len(rets)))
    print(f"\n--- Edges (|avg|≥0.5%, n≥3) ---")
    for key, avg, wr, n in sorted(edges, key=lambda x: abs(x[1]), reverse=True):
        w, v, s = key; d = 'LONG' if avg > 0 else 'SHORT'
        print(f"  {w} × {v} × {s} → {d}: {avg:+.3f}%, {wr:.0f}% win, n={n}")

    # WEAK vs STRONG demand
    weak = [r['returns'].get('24h_aggregate', {}).get('return', 0) for r in results if r['demand'] == 'WEAK_DEMAND']
    strong = [r['returns'].get('24h_aggregate', {}).get('return', 0) for r in results if r['demand'] == 'STRONG_DEMAND']
    if weak and strong:
        t2, p2 = stats.ttest_ind(weak, strong)
        print(f"\nWEAK vs STRONG: weak={np.mean(weak):.3f}% (n={len(weak)}), strong={np.mean(strong):.3f}% (n={len(strong)}), t={t2:.3f}, p={p2:.4f} {'✅' if p2 < 0.05 else '❌'}")
    return edges


def analyze_transmission(results):
    print("\n" + "="*80)
    print("PROMPT B: SESSION TRANSMISSION CHAIN ANALYSIS")
    print("="*80)
    names = list(SESSIONS.keys())
    print(f"\n--- Direction Persistence (n≥5) ---")
    trans = {}
    for r in results:
        prev_dir, prev_name = None, None
        for name in names:
            ret = r['returns'].get(name, {}).get('return', 0)
            curr = 1 if ret > 0 else (-1 if ret < 0 else 0)
            if prev_dir is not None and prev_dir != 0 and curr != 0:
                key = f"{prev_name} → {name}"
                trans.setdefault(key, {'same': 0, 'total': 0})
                trans[key]['total'] += 1
                if prev_dir == curr: trans[key]['same'] += 1
            prev_dir, prev_name = curr, name
    for key, d in sorted(trans.items()):
        if d['total'] >= 5:
            pct = d['same']/d['total']*100
            s = '✅' if pct > 65 else ('⚠️' if pct > 55 else '❌')
            print(f"  {key:45s}: {pct:.1f}% (n={d['total']}) {s}")
    print(f"\n--- Average Return by Session ---")
    for name in names:
        rets = [r['returns'].get(name, {}).get('return', 0) for r in results]
        if name != '24h_aggregate': print(f"  {name:25s}: {np.mean(rets):+.3f}% avg, {sum(1 for r in rets if r > 0)/len(rets)*100:.1f}% win")


if __name__ == '__main__':
    csv_path = os.path.join(os.path.dirname(__file__), 'eth_15m_merged.csv')
    if not os.path.exists(csv_path): csv_path = 'eth_15m_merged.csv'
    print("Loading ETH 15m data...")
    results = run_backtest(csv_path)
    edges = analyze_results(results)
    analyze_transmission(results)
    out_path = os.path.join(os.path.dirname(__file__), 'backtest_treasury_auction_results.json')
    with open(out_path, 'w') as f:
        json.dump([{k: v for k, v in r.items() if k != 'returns'} | {'returns': r['returns']} for r in results], f, indent=2, default=str)
    print(f"\nResults saved to {out_path}")

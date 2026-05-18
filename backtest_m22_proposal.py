#!/usr/bin/env python3
"""
Backtest: Current M22 lookup-table vs. proposed parametric model + surprise signal

Tests three hypotheses from the critique:
  H1: Parametric model (continuous) beats lookup-table (discrete) at predicting 24h returns
  H2: Consensus surprise (actual vs expected) adds alpha beyond regime classification
  H3: Simpler model with fewer dimensions generalizes better out-of-sample

Data: ETH/USDT 15m, all PPI/CPI/NFP releases 2018-2026 with consensus data
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

# ═══════════════════════════════════════════════════════════════
# RELEASE DATA (with consensus)
# ═══════════════════════════════════════════════════════════════

PPI_RELEASES = {
    '2018-01-11': {'ppi_yoy': 2.6, 'consensus_yoy': 2.5, 'prior_yoy': 2.6},
    '2018-02-15': {'ppi_yoy': 2.8, 'consensus_yoy': 2.7, 'prior_yoy': 2.6},
    '2018-03-14': {'ppi_yoy': 2.8, 'consensus_yoy': 2.8, 'prior_yoy': 2.8},
    '2018-04-11': {'ppi_yoy': 2.9, 'consensus_yoy': 2.8, 'prior_yoy': 2.8},
    '2018-05-10': {'ppi_yoy': 2.7, 'consensus_yoy': 2.8, 'prior_yoy': 2.9},
    '2018-06-12': {'ppi_yoy': 3.1, 'consensus_yoy': 2.9, 'prior_yoy': 2.7},
    '2018-07-11': {'ppi_yoy': 3.3, 'consensus_yoy': 3.2, 'prior_yoy': 3.1},
    '2018-08-09': {'ppi_yoy': 3.3, 'consensus_yoy': 3.3, 'prior_yoy': 3.3},
    '2018-09-12': {'ppi_yoy': 2.8, 'consensus_yoy': 3.2, 'prior_yoy': 3.3},
    '2018-10-10': {'ppi_yoy': 2.6, 'consensus_yoy': 2.6, 'prior_yoy': 2.8},
    '2018-11-14': {'ppi_yoy': 2.5, 'consensus_yoy': 2.5, 'prior_yoy': 2.6},
    '2018-12-11': {'ppi_yoy': 2.5, 'consensus_yoy': 2.5, 'prior_yoy': 2.5},
    '2019-01-15': {'ppi_yoy': 2.0, 'consensus_yoy': 2.2, 'prior_yoy': 2.5},
    '2019-02-14': {'ppi_yoy': 1.7, 'consensus_yoy': 1.8, 'prior_yoy': 2.0},
    '2019-03-14': {'ppi_yoy': 1.9, 'consensus_yoy': 1.8, 'prior_yoy': 1.7},
    '2019-04-11': {'ppi_yoy': 2.2, 'consensus_yoy': 2.0, 'prior_yoy': 1.9},
    '2019-05-09': {'ppi_yoy': 2.2, 'consensus_yoy': 2.3, 'prior_yoy': 2.2},
    '2019-06-11': {'ppi_yoy': 1.8, 'consensus_yoy': 2.0, 'prior_yoy': 2.2},
    '2019-07-12': {'ppi_yoy': 1.7, 'consensus_yoy': 1.7, 'prior_yoy': 1.8},
    '2019-08-09': {'ppi_yoy': 1.7, 'consensus_yoy': 1.7, 'prior_yoy': 1.7},
    '2019-09-11': {'ppi_yoy': 1.4, 'consensus_yoy': 1.8, 'prior_yoy': 1.7},
    '2019-10-08': {'ppi_yoy': 1.1, 'consensus_yoy': 1.5, 'prior_yoy': 1.4},
    '2019-11-14': {'ppi_yoy': 1.1, 'consensus_yoy': 0.9, 'prior_yoy': 1.1},
    '2019-12-12': {'ppi_yoy': 1.1, 'consensus_yoy': 1.3, 'prior_yoy': 1.1},
    '2020-01-14': {'ppi_yoy': 1.7, 'consensus_yoy': 1.3, 'prior_yoy': 1.1},
    '2020-02-19': {'ppi_yoy': 2.0, 'consensus_yoy': 1.6, 'prior_yoy': 1.7},
    '2020-03-12': {'ppi_yoy': 1.3, 'consensus_yoy': 1.2, 'prior_yoy': 2.0},
    '2020-04-09': {'ppi_yoy': -0.6, 'consensus_yoy': 0.2, 'prior_yoy': 1.3},
    '2020-05-12': {'ppi_yoy': -1.2, 'consensus_yoy': -1.2, 'prior_yoy': -0.6},
    '2020-06-10': {'ppi_yoy': -0.8, 'consensus_yoy': -1.2, 'prior_yoy': -1.2},
    '2020-07-14': {'ppi_yoy': -0.4, 'consensus_yoy': -0.7, 'prior_yoy': -0.8},
    '2020-08-11': {'ppi_yoy': -0.2, 'consensus_yoy': -0.3, 'prior_yoy': -0.4},
    '2020-09-11': {'ppi_yoy': 0.4, 'consensus_yoy': 0.2, 'prior_yoy': -0.2},
    '2020-10-14': {'ppi_yoy': 0.5, 'consensus_yoy': 0.4, 'prior_yoy': 0.4},
    '2020-11-13': {'ppi_yoy': 0.8, 'consensus_yoy': 0.5, 'prior_yoy': 0.5},
    '2020-12-11': {'ppi_yoy': 0.8, 'consensus_yoy': 0.7, 'prior_yoy': 0.8},
    '2021-01-13': {'ppi_yoy': 1.3, 'consensus_yoy': 0.9, 'prior_yoy': 0.8},
    '2021-02-17': {'ppi_yoy': 1.7, 'consensus_yoy': 1.0, 'prior_yoy': 1.3},
    '2021-03-12': {'ppi_yoy': 2.8, 'consensus_yoy': 2.7, 'prior_yoy': 1.7},
    '2021-04-09': {'ppi_yoy': 4.2, 'consensus_yoy': 3.8, 'prior_yoy': 2.8},
    '2021-05-12': {'ppi_yoy': 6.6, 'consensus_yoy': 5.9, 'prior_yoy': 4.2},
    '2021-06-11': {'ppi_yoy': 7.3, 'consensus_yoy': 6.3, 'prior_yoy': 6.6},
    '2021-07-13': {'ppi_yoy': 7.8, 'consensus_yoy': 7.3, 'prior_yoy': 7.3},
    '2021-08-12': {'ppi_yoy': 8.6, 'consensus_yoy': 8.2, 'prior_yoy': 7.8},
    '2021-09-10': {'ppi_yoy': 8.7, 'consensus_yoy': 8.3, 'prior_yoy': 8.6},
    '2021-10-14': {'ppi_yoy': 8.8, 'consensus_yoy': 8.7, 'prior_yoy': 8.7},
    '2021-11-09': {'ppi_yoy': 8.8, 'consensus_yoy': 8.7, 'prior_yoy': 8.8},
    '2021-12-14': {'ppi_yoy': 9.8, 'consensus_yoy': 9.2, 'prior_yoy': 8.8},
    '2022-01-13': {'ppi_yoy': 9.7, 'consensus_yoy': 9.8, 'prior_yoy': 9.8},
    '2022-02-15': {'ppi_yoy': 10.0, 'consensus_yoy': 9.1, 'prior_yoy': 9.7},
    '2022-03-11': {'ppi_yoy': 10.0, 'consensus_yoy': 10.0, 'prior_yoy': 10.0},
    '2022-04-12': {'ppi_yoy': 11.2, 'consensus_yoy': 10.6, 'prior_yoy': 10.0},
    '2022-05-12': {'ppi_yoy': 10.9, 'consensus_yoy': 10.7, 'prior_yoy': 11.2},
    '2022-06-14': {'ppi_yoy': 10.8, 'consensus_yoy': 10.9, 'prior_yoy': 10.9},
    '2022-07-14': {'ppi_yoy': 9.8, 'consensus_yoy': 10.7, 'prior_yoy': 10.8},
    '2022-08-11': {'ppi_yoy': 8.7, 'consensus_yoy': 8.8, 'prior_yoy': 9.8},
    '2022-09-14': {'ppi_yoy': 8.5, 'consensus_yoy': 8.8, 'prior_yoy': 8.7},
    '2022-10-12': {'ppi_yoy': 8.0, 'consensus_yoy': 8.4, 'prior_yoy': 8.5},
    '2022-11-15': {'ppi_yoy': 7.4, 'consensus_yoy': 8.0, 'prior_yoy': 8.0},
    '2022-12-09': {'ppi_yoy': 6.2, 'consensus_yoy': 7.2, 'prior_yoy': 7.4},
    '2023-01-18': {'ppi_yoy': 6.2, 'consensus_yoy': 6.2, 'prior_yoy': 6.2},
    '2023-02-16': {'ppi_yoy': 6.0, 'consensus_yoy': 5.4, 'prior_yoy': 6.2},
    '2023-03-15': {'ppi_yoy': 4.6, 'consensus_yoy': 5.4, 'prior_yoy': 6.0},
    '2023-04-13': {'ppi_yoy': 2.7, 'consensus_yoy': 3.0, 'prior_yoy': 4.6},
    '2023-05-11': {'ppi_yoy': 2.3, 'consensus_yoy': 2.4, 'prior_yoy': 2.7},
    '2023-06-14': {'ppi_yoy': 1.1, 'consensus_yoy': 1.5, 'prior_yoy': 2.3},
    '2023-07-13': {'ppi_yoy': 0.1, 'consensus_yoy': 0.4, 'prior_yoy': 1.1},
    '2023-08-11': {'ppi_yoy': 0.8, 'consensus_yoy': 0.7, 'prior_yoy': 0.1},
    '2023-09-14': {'ppi_yoy': 1.6, 'consensus_yoy': 1.3, 'prior_yoy': 0.8},
    '2023-10-12': {'ppi_yoy': 1.5, 'consensus_yoy': 1.6, 'prior_yoy': 1.6},
    '2023-11-15': {'ppi_yoy': 1.3, 'consensus_yoy': 1.3, 'prior_yoy': 1.5},
    '2023-12-13': {'ppi_yoy': 1.0, 'consensus_yoy': 1.3, 'prior_yoy': 1.3},
    '2024-01-12': {'ppi_yoy': 1.0, 'consensus_yoy': 1.3, 'prior_yoy': 1.0},
    '2024-02-16': {'ppi_yoy': 1.6, 'consensus_yoy': 0.6, 'prior_yoy': 1.0},
    '2024-03-14': {'ppi_yoy': 2.2, 'consensus_yoy': 1.2, 'prior_yoy': 1.6},
    '2024-04-11': {'ppi_yoy': 2.4, 'consensus_yoy': 2.3, 'prior_yoy': 2.2},
    '2024-05-14': {'ppi_yoy': 2.6, 'consensus_yoy': 2.5, 'prior_yoy': 2.4},
    '2024-06-13': {'ppi_yoy': 2.7, 'consensus_yoy': 2.5, 'prior_yoy': 2.6},
    '2024-07-12': {'ppi_yoy': 2.9, 'consensus_yoy': 2.3, 'prior_yoy': 2.7},
    '2024-08-13': {'ppi_yoy': 2.6, 'consensus_yoy': 2.3, 'prior_yoy': 2.9},
    '2024-09-12': {'ppi_yoy': 2.4, 'consensus_yoy': 2.5, 'prior_yoy': 2.6},
    '2024-10-11': {'ppi_yoy': 2.8, 'consensus_yoy': 2.3, 'prior_yoy': 2.4},
    '2024-11-14': {'ppi_yoy': 3.0, 'consensus_yoy': 2.3, 'prior_yoy': 2.8},
    '2024-12-12': {'ppi_yoy': 3.3, 'consensus_yoy': 2.5, 'prior_yoy': 3.0},
    '2025-01-14': {'ppi_yoy': 3.3, 'consensus_yoy': 3.4, 'prior_yoy': 3.3},
    '2025-02-13': {'ppi_yoy': 3.5, 'consensus_yoy': 3.3, 'prior_yoy': 3.3},
    '2025-03-13': {'ppi_yoy': 3.2, 'consensus_yoy': 3.3, 'prior_yoy': 3.5},
    '2025-04-10': {'ppi_yoy': 3.4, 'consensus_yoy': 3.3, 'prior_yoy': 3.2},
    '2025-05-15': {'ppi_yoy': 3.1, 'consensus_yoy': 3.1, 'prior_yoy': 3.4},
    '2025-06-12': {'ppi_yoy': 3.0, 'consensus_yoy': 2.9, 'prior_yoy': 3.1},
    '2025-07-16': {'ppi_yoy': 2.8, 'consensus_yoy': 2.7, 'prior_yoy': 3.0},
    '2025-08-14': {'ppi_yoy': 3.1, 'consensus_yoy': 2.8, 'prior_yoy': 2.8},
    '2025-09-11': {'ppi_yoy': 3.0, 'consensus_yoy': 3.1, 'prior_yoy': 3.1},
    '2025-10-15': {'ppi_yoy': 2.8, 'consensus_yoy': 2.9, 'prior_yoy': 3.0},
    '2025-11-13': {'ppi_yoy': 2.6, 'consensus_yoy': 2.7, 'prior_yoy': 2.8},
    '2025-12-11': {'ppi_yoy': 2.5, 'consensus_yoy': 2.6, 'prior_yoy': 2.6},
    '2026-01-14': {'ppi_yoy': 2.7, 'consensus_yoy': 2.5, 'prior_yoy': 2.5},
    '2026-02-13': {'ppi_yoy': 3.0, 'consensus_yoy': 2.7, 'prior_yoy': 2.7},
    '2026-03-13': {'ppi_yoy': 3.2, 'consensus_yoy': 3.0, 'prior_yoy': 3.0},
    '2026-04-14': {'ppi_yoy': 3.5, 'consensus_yoy': 3.2, 'prior_yoy': 3.2},
    '2026-05-13': {'ppi_yoy': 3.3, 'consensus_yoy': 3.4, 'prior_yoy': 3.5},
}

CPI_RELEASES = {
    '2018-01-11': {'cpi_yoy': 2.1, 'consensus_yoy': 2.1, 'prior_yoy': 2.1},
    '2018-02-14': {'cpi_yoy': 2.2, 'consensus_yoy': 2.0, 'prior_yoy': 2.1},
    '2018-03-13': {'cpi_yoy': 2.2, 'consensus_yoy': 2.2, 'prior_yoy': 2.2},
    '2018-04-11': {'cpi_yoy': 2.4, 'consensus_yoy': 2.4, 'prior_yoy': 2.2},
    '2018-05-10': {'cpi_yoy': 2.5, 'consensus_yoy': 2.5, 'prior_yoy': 2.4},
    '2018-06-12': {'cpi_yoy': 2.8, 'consensus_yoy': 2.8, 'prior_yoy': 2.5},
    '2018-07-12': {'cpi_yoy': 2.9, 'consensus_yoy': 2.9, 'prior_yoy': 2.8},
    '2018-08-10': {'cpi_yoy': 2.9, 'consensus_yoy': 2.9, 'prior_yoy': 2.9},
    '2018-09-13': {'cpi_yoy': 2.7, 'consensus_yoy': 2.8, 'prior_yoy': 2.9},
    '2018-10-11': {'cpi_yoy': 2.3, 'consensus_yoy': 2.4, 'prior_yoy': 2.7},
    '2018-11-14': {'cpi_yoy': 2.2, 'consensus_yoy': 2.4, 'prior_yoy': 2.3},
    '2018-12-12': {'cpi_yoy': 1.9, 'consensus_yoy': 2.1, 'prior_yoy': 2.2},
    '2019-01-11': {'cpi_yoy': 1.6, 'consensus_yoy': 1.7, 'prior_yoy': 1.9},
    '2019-02-13': {'cpi_yoy': 1.6, 'consensus_yoy': 1.6, 'prior_yoy': 1.6},
    '2019-03-12': {'cpi_yoy': 1.5, 'consensus_yoy': 1.6, 'prior_yoy': 1.6},
    '2019-04-10': {'cpi_yoy': 1.9, 'consensus_yoy': 1.8, 'prior_yoy': 1.5},
    '2019-05-10': {'cpi_yoy': 1.8, 'consensus_yoy': 1.9, 'prior_yoy': 1.9},
    '2019-06-12': {'cpi_yoy': 1.5, 'consensus_yoy': 1.6, 'prior_yoy': 1.8},
    '2019-07-11': {'cpi_yoy': 1.8, 'consensus_yoy': 1.7, 'prior_yoy': 1.5},
    '2019-08-13': {'cpi_yoy': 1.7, 'consensus_yoy': 1.7, 'prior_yoy': 1.8},
    '2019-09-12': {'cpi_yoy': 1.7, 'consensus_yoy': 1.8, 'prior_yoy': 1.7},
    '2019-10-10': {'cpi_yoy': 1.7, 'consensus_yoy': 1.7, 'prior_yoy': 1.7},
    '2019-11-13': {'cpi_yoy': 1.8, 'consensus_yoy': 1.7, 'prior_yoy': 1.7},
    '2019-12-11': {'cpi_yoy': 2.1, 'consensus_yoy': 2.0, 'prior_yoy': 1.8},
    '2020-01-14': {'cpi_yoy': 2.5, 'consensus_yoy': 2.4, 'prior_yoy': 2.1},
    '2020-02-13': {'cpi_yoy': 2.3, 'consensus_yoy': 2.2, 'prior_yoy': 2.5},
    '2020-03-11': {'cpi_yoy': 2.3, 'consensus_yoy': 2.2, 'prior_yoy': 2.3},
    '2020-04-10': {'cpi_yoy': 1.5, 'consensus_yoy': 1.4, 'prior_yoy': 2.3},
    '2020-05-12': {'cpi_yoy': 0.3, 'consensus_yoy': 0.4, 'prior_yoy': 1.5},
    '2020-06-10': {'cpi_yoy': 0.1, 'consensus_yoy': 0.2, 'prior_yoy': 0.3},
    '2020-07-14': {'cpi_yoy': 0.6, 'consensus_yoy': 0.5, 'prior_yoy': 0.1},
    '2020-08-12': {'cpi_yoy': 1.3, 'consensus_yoy': 1.2, 'prior_yoy': 0.6},
    '2020-09-11': {'cpi_yoy': 1.4, 'consensus_yoy': 1.3, 'prior_yoy': 1.3},
    '2020-10-13': {'cpi_yoy': 1.4, 'consensus_yoy': 1.4, 'prior_yoy': 1.4},
    '2020-11-12': {'cpi_yoy': 1.2, 'consensus_yoy': 1.3, 'prior_yoy': 1.4},
    '2020-12-10': {'cpi_yoy': 1.2, 'consensus_yoy': 1.3, 'prior_yoy': 1.2},
    '2021-01-13': {'cpi_yoy': 1.4, 'consensus_yoy': 1.5, 'prior_yoy': 1.2},
    '2021-02-10': {'cpi_yoy': 1.7, 'consensus_yoy': 1.7, 'prior_yoy': 1.4},
    '2021-03-10': {'cpi_yoy': 2.6, 'consensus_yoy': 2.5, 'prior_yoy': 1.7},
    '2021-04-13': {'cpi_yoy': 4.2, 'consensus_yoy': 3.6, 'prior_yoy': 2.6},
    '2021-05-12': {'cpi_yoy': 5.0, 'consensus_yoy': 4.7, 'prior_yoy': 4.2},
    '2021-06-10': {'cpi_yoy': 5.4, 'consensus_yoy': 5.0, 'prior_yoy': 5.0},
    '2021-07-13': {'cpi_yoy': 5.4, 'consensus_yoy': 5.3, 'prior_yoy': 5.4},
    '2021-08-11': {'cpi_yoy': 5.3, 'consensus_yoy': 5.3, 'prior_yoy': 5.4},
    '2021-09-14': {'cpi_yoy': 5.3, 'consensus_yoy': 5.3, 'prior_yoy': 5.3},
    '2021-10-13': {'cpi_yoy': 6.2, 'consensus_yoy': 5.8, 'prior_yoy': 5.3},
    '2021-11-10': {'cpi_yoy': 6.8, 'consensus_yoy': 5.9, 'prior_yoy': 6.2},
    '2021-12-10': {'cpi_yoy': 7.0, 'consensus_yoy': 6.8, 'prior_yoy': 6.8},
    '2022-01-12': {'cpi_yoy': 7.5, 'consensus_yoy': 7.2, 'prior_yoy': 7.0},
    '2022-02-10': {'cpi_yoy': 7.5, 'consensus_yoy': 7.3, 'prior_yoy': 7.5},
    '2022-03-10': {'cpi_yoy': 7.9, 'consensus_yoy': 7.9, 'prior_yoy': 7.5},
    '2022-04-12': {'cpi_yoy': 8.5, 'consensus_yoy': 8.4, 'prior_yoy': 7.9},
    '2022-05-11': {'cpi_yoy': 8.3, 'consensus_yoy': 8.1, 'prior_yoy': 8.5},
    '2022-06-10': {'cpi_yoy': 8.6, 'consensus_yoy': 8.3, 'prior_yoy': 8.3},
    '2022-07-13': {'cpi_yoy': 9.1, 'consensus_yoy': 8.8, 'prior_yoy': 8.6},
    '2022-08-10': {'cpi_yoy': 8.5, 'consensus_yoy': 8.7, 'prior_yoy': 9.1},
    '2022-09-13': {'cpi_yoy': 8.2, 'consensus_yoy': 8.1, 'prior_yoy': 8.5},
    '2022-10-13': {'cpi_yoy': 7.7, 'consensus_yoy': 8.0, 'prior_yoy': 8.2},
    '2022-11-10': {'cpi_yoy': 7.1, 'consensus_yoy': 7.3, 'prior_yoy': 7.7},
    '2022-12-13': {'cpi_yoy': 7.1, 'consensus_yoy': 7.3, 'prior_yoy': 7.1},
    '2023-01-12': {'cpi_yoy': 6.5, 'consensus_yoy': 6.5, 'prior_yoy': 7.1},
    '2023-02-14': {'cpi_yoy': 6.4, 'consensus_yoy': 6.2, 'prior_yoy': 6.5},
    '2023-03-14': {'cpi_yoy': 6.0, 'consensus_yoy': 5.2, 'prior_yoy': 6.4},
    '2023-04-12': {'cpi_yoy': 5.0, 'consensus_yoy': 5.2, 'prior_yoy': 6.0},
    '2023-05-10': {'cpi_yoy': 4.9, 'consensus_yoy': 5.0, 'prior_yoy': 5.0},
    '2023-06-13': {'cpi_yoy': 4.0, 'consensus_yoy': 4.1, 'prior_yoy': 4.9},
    '2023-07-12': {'cpi_yoy': 3.0, 'consensus_yoy': 3.1, 'prior_yoy': 4.0},
    '2023-08-10': {'cpi_yoy': 3.2, 'consensus_yoy': 3.3, 'prior_yoy': 3.0},
    '2023-09-13': {'cpi_yoy': 3.7, 'consensus_yoy': 3.6, 'prior_yoy': 3.2},
    '2023-10-12': {'cpi_yoy': 3.7, 'consensus_yoy': 3.6, 'prior_yoy': 3.7},
    '2023-11-14': {'cpi_yoy': 3.2, 'consensus_yoy': 3.3, 'prior_yoy': 3.7},
    '2023-12-12': {'cpi_yoy': 3.1, 'consensus_yoy': 3.1, 'prior_yoy': 3.2},
    '2024-01-11': {'cpi_yoy': 3.4, 'consensus_yoy': 3.2, 'prior_yoy': 3.1},
    '2024-02-13': {'cpi_yoy': 3.1, 'consensus_yoy': 2.9, 'prior_yoy': 3.4},
    '2024-03-12': {'cpi_yoy': 3.2, 'consensus_yoy': 3.1, 'prior_yoy': 3.1},
    '2024-04-10': {'cpi_yoy': 3.5, 'consensus_yoy': 3.4, 'prior_yoy': 3.2},
    '2024-05-15': {'cpi_yoy': 3.4, 'consensus_yoy': 3.4, 'prior_yoy': 3.5},
    '2024-06-12': {'cpi_yoy': 3.3, 'consensus_yoy': 3.4, 'prior_yoy': 3.4},
    '2024-07-11': {'cpi_yoy': 3.0, 'consensus_yoy': 3.1, 'prior_yoy': 3.3},
    '2024-08-14': {'cpi_yoy': 2.9, 'consensus_yoy': 2.9, 'prior_yoy': 3.0},
    '2024-09-11': {'cpi_yoy': 2.5, 'consensus_yoy': 2.6, 'prior_yoy': 2.9},
    '2024-10-10': {'cpi_yoy': 2.4, 'consensus_yoy': 2.3, 'prior_yoy': 2.5},
    '2024-11-13': {'cpi_yoy': 2.6, 'consensus_yoy': 2.6, 'prior_yoy': 2.4},
    '2024-12-11': {'cpi_yoy': 2.7, 'consensus_yoy': 2.7, 'prior_yoy': 2.6},
    '2025-01-15': {'cpi_yoy': 2.9, 'consensus_yoy': 2.9, 'prior_yoy': 2.7},
    '2025-02-12': {'cpi_yoy': 2.8, 'consensus_yoy': 2.9, 'prior_yoy': 2.9},
    '2025-03-12': {'cpi_yoy': 2.8, 'consensus_yoy': 2.9, 'prior_yoy': 2.8},
    '2025-04-10': {'cpi_yoy': 2.4, 'consensus_yoy': 2.5, 'prior_yoy': 2.8},
    '2025-05-13': {'cpi_yoy': 2.3, 'consensus_yoy': 2.4, 'prior_yoy': 2.4},
    '2025-06-11': {'cpi_yoy': 2.4, 'consensus_yoy': 2.4, 'prior_yoy': 2.3},
    '2025-07-15': {'cpi_yoy': 2.7, 'consensus_yoy': 2.6, 'prior_yoy': 2.4},
    '2025-08-12': {'cpi_yoy': 2.9, 'consensus_yoy': 2.8, 'prior_yoy': 2.7},
    '2025-09-10': {'cpi_yoy': 2.9, 'consensus_yoy': 2.9, 'prior_yoy': 2.9},
    '2025-10-14': {'cpi_yoy': 3.0, 'consensus_yoy': 2.9, 'prior_yoy': 2.9},
    '2025-11-12': {'cpi_yoy': 2.9, 'consensus_yoy': 3.0, 'prior_yoy': 3.0},
    '2025-12-10': {'cpi_yoy': 2.8, 'consensus_yoy': 2.9, 'prior_yoy': 2.9},
    '2026-01-14': {'cpi_yoy': 2.9, 'consensus_yoy': 2.8, 'prior_yoy': 2.8},
    '2026-02-11': {'cpi_yoy': 3.1, 'consensus_yoy': 2.9, 'prior_yoy': 2.9},
    '2026-03-11': {'cpi_yoy': 3.3, 'consensus_yoy': 3.1, 'prior_yoy': 3.1},
    '2026-04-10': {'cpi_yoy': 3.5, 'consensus_yoy': 3.3, 'prior_yoy': 3.3},
    '2026-05-12': {'cpi_yoy': 3.4, 'consensus_yoy': 3.5, 'prior_yoy': 3.5},
}

NFP_RELEASES = {
    '2018-01-05': {'nfp_k': 148, 'consensus_k': 190, 'prev_k': 252},
    '2018-02-02': {'nfp_k': 200, 'consensus_k': 180, 'prev_k': 160},
    '2018-03-09': {'nfp_k': 313, 'consensus_k': 205, 'prev_k': 239},
    '2018-04-06': {'nfp_k': 103, 'consensus_k': 185, 'prev_k': 326},
    '2018-05-04': {'nfp_k': 164, 'consensus_k': 192, 'prev_k': 135},
    '2018-06-01': {'nfp_k': 223, 'consensus_k': 188, 'prev_k': 159},
    '2018-07-06': {'nfp_k': 213, 'consensus_k': 195, 'prev_k': 244},
    '2018-08-03': {'nfp_k': 157, 'consensus_k': 190, 'prev_k': 248},
    '2018-09-07': {'nfp_k': 201, 'consensus_k': 190, 'prev_k': 147},
    '2018-10-05': {'nfp_k': 134, 'consensus_k': 185, 'prev_k': 270},
    '2018-11-02': {'nfp_k': 250, 'consensus_k': 190, 'prev_k': 118},
    '2018-12-07': {'nfp_k': 155, 'consensus_k': 200, 'prev_k': 237},
    '2019-01-04': {'nfp_k': 312, 'consensus_k': 177, 'prev_k': 176},
    '2019-02-01': {'nfp_k': 304, 'consensus_k': 165, 'prev_k': 222},
    '2019-03-08': {'nfp_k': 20, 'consensus_k': 180, 'prev_k': 311},
    '2019-04-05': {'nfp_k': 196, 'consensus_k': 175, 'prev_k': 33},
    '2019-05-03': {'nfp_k': 263, 'consensus_k': 185, 'prev_k': 196},
    '2019-06-07': {'nfp_k': 75, 'consensus_k': 180, 'prev_k': 224},
    '2019-07-05': {'nfp_k': 224, 'consensus_k': 160, 'prev_k': 72},
    '2019-08-02': {'nfp_k': 164, 'consensus_k': 165, 'prev_k': 193},
    '2019-09-06': {'nfp_k': 130, 'consensus_k': 160, 'prev_k': 159},
    '2019-10-04': {'nfp_k': 136, 'consensus_k': 145, 'prev_k': 168},
    '2019-11-01': {'nfp_k': 128, 'consensus_k': 85, 'prev_k': 180},
    '2019-12-06': {'nfp_k': 266, 'consensus_k': 180, 'prev_k': 156},
    '2020-01-10': {'nfp_k': 145, 'consensus_k': 160, 'prev_k': 256},
    '2020-02-07': {'nfp_k': 225, 'consensus_k': 160, 'prev_k': 214},
    '2020-03-06': {'nfp_k': 273, 'consensus_k': 175, 'prev_k': 214},
    '2020-04-03': {'nfp_k': -701, 'consensus_k': -100, 'prev_k': 230},
    '2020-05-08': {'nfp_k': -20500, 'consensus_k': -8000, 'prev_k': -870},
    '2020-06-05': {'nfp_k': 2509, 'consensus_k': -8000, 'prev_k': -20700},
    '2020-07-02': {'nfp_k': 4800, 'consensus_k': 3000, 'prev_k': 2700},
    '2020-08-07': {'nfp_k': 1763, 'consensus_k': 1600, 'prev_k': 4791},
    '2020-09-04': {'nfp_k': 661, 'consensus_k': 1400, 'prev_k': 1489},
    '2020-10-02': {'nfp_k': 661, 'consensus_k': 800, 'prev_k': 1489},
    '2020-11-06': {'nfp_k': 638, 'consensus_k': 600, 'prev_k': 672},
    '2020-12-04': {'nfp_k': 245, 'consensus_k': 500, 'prev_k': 610},
    '2021-01-08': {'nfp_k': -140, 'consensus_k': 50, 'prev_k': -227},
    '2021-02-05': {'nfp_k': 49, 'consensus_k': 50, 'prev_k': -227},
    '2021-03-05': {'nfp_k': 379, 'consensus_k': 200, 'prev_k': 166},
    '2021-04-02': {'nfp_k': 916, 'consensus_k': 650, 'prev_k': 468},
    '2021-05-07': {'nfp_k': 266, 'consensus_k': 1000, 'prev_k': 770},
    '2021-06-04': {'nfp_k': 559, 'consensus_k': 675, 'prev_k': 278},
    '2021-07-02': {'nfp_k': 850, 'consensus_k': 700, 'prev_k': 583},
    '2021-08-06': {'nfp_k': 943, 'consensus_k': 870, 'prev_k': 938},
    '2021-09-03': {'nfp_k': 235, 'consensus_k': 750, 'prev_k': 1053},
    '2021-10-08': {'nfp_k': 194, 'consensus_k': 500, 'prev_k': 366},
    '2021-11-05': {'nfp_k': 531, 'consensus_k': 450, 'prev_k': 312},
    '2021-12-03': {'nfp_k': 210, 'consensus_k': 550, 'prev_k': 546},
    '2022-01-07': {'nfp_k': 199, 'consensus_k': 400, 'prev_k': 249},
    '2022-02-04': {'nfp_k': 467, 'consensus_k': 150, 'prev_k': 510},
    '2022-03-04': {'nfp_k': 678, 'consensus_k': 400, 'prev_k': 481},
    '2022-04-01': {'nfp_k': 431, 'consensus_k': 490, 'prev_k': 750},
    '2022-05-06': {'nfp_k': 428, 'consensus_k': 391, 'prev_k': 368},
    '2022-06-03': {'nfp_k': 390, 'consensus_k': 325, 'prev_k': 428},
    '2022-07-08': {'nfp_k': 372, 'consensus_k': 268, 'prev_k': 384},
    '2022-08-05': {'nfp_k': 528, 'consensus_k': 250, 'prev_k': 398},
    '2022-09-02': {'nfp_k': 315, 'consensus_k': 300, 'prev_k': 526},
    '2022-10-07': {'nfp_k': 263, 'consensus_k': 250, 'prev_k': 315},
    '2022-11-04': {'nfp_k': 261, 'consensus_k': 200, 'prev_k': 263},
    '2022-12-02': {'nfp_k': 263, 'consensus_k': 200, 'prev_k': 284},
    '2023-01-06': {'nfp_k': 223, 'consensus_k': 200, 'prev_k': 256},
    '2023-02-03': {'nfp_k': 517, 'consensus_k': 185, 'prev_k': 260},
    '2023-03-10': {'nfp_k': 311, 'consensus_k': 225, 'prev_k': 504},
    '2023-04-07': {'nfp_k': 236, 'consensus_k': 239, 'prev_k': 472},
    '2023-05-05': {'nfp_k': 253, 'consensus_k': 180, 'prev_k': 217},
    '2023-06-02': {'nfp_k': 339, 'consensus_k': 190, 'prev_k': 294},
    '2023-07-07': {'nfp_k': 209, 'consensus_k': 225, 'prev_k': 306},
    '2023-08-04': {'nfp_k': 187, 'consensus_k': 200, 'prev_k': 185},
    '2023-09-01': {'nfp_k': 187, 'consensus_k': 170, 'prev_k': 157},
    '2023-10-06': {'nfp_k': 336, 'consensus_k': 170, 'prev_k': 227},
    '2023-11-03': {'nfp_k': 150, 'consensus_k': 180, 'prev_k': 297},
    '2023-12-08': {'nfp_k': 199, 'consensus_k': 180, 'prev_k': 150},
    '2024-01-05': {'nfp_k': 216, 'consensus_k': 175, 'prev_k': 216},
    '2024-02-02': {'nfp_k': 353, 'consensus_k': 180, 'prev_k': 333},
    '2024-03-08': {'nfp_k': 275, 'consensus_k': 200, 'prev_k': 229},
    '2024-04-05': {'nfp_k': 303, 'consensus_k': 214, 'prev_k': 270},
    '2024-05-03': {'nfp_k': 175, 'consensus_k': 240, 'prev_k': 315},
    '2024-06-07': {'nfp_k': 272, 'consensus_k': 180, 'prev_k': 165},
    '2024-07-05': {'nfp_k': 206, 'consensus_k': 190, 'prev_k': 218},
    '2024-08-02': {'nfp_k': 114, 'consensus_k': 175, 'prev_k': 179},
    '2024-09-06': {'nfp_k': 142, 'consensus_k': 160, 'prev_k': 89},
    '2024-10-04': {'nfp_k': 254, 'consensus_k': 150, 'prev_k': 159},
    '2024-11-01': {'nfp_k': 12, 'consensus_k': 113, 'prev_k': 254},
    '2024-12-06': {'nfp_k': 227, 'consensus_k': 214, 'prev_k': 36},
    '2025-01-10': {'nfp_k': 256, 'consensus_k': 165, 'prev_k': 307},
    '2025-02-07': {'nfp_k': 143, 'consensus_k': 169, 'prev_k': 323},
    '2025-03-07': {'nfp_k': 151, 'consensus_k': 160, 'prev_k': 125},
    '2025-04-04': {'nfp_k': 228, 'consensus_k': 135, 'prev_k': 117},
    '2025-05-02': {'nfp_k': 177, 'consensus_k': 130, 'prev_k': 185},
    '2025-06-06': {'nfp_k': 139, 'consensus_k': 125, 'prev_k': 147},
    '2025-07-03': {'nfp_k': 150, 'consensus_k': 110, 'prev_k': 144},
    '2025-08-01': {'nfp_k': 106, 'consensus_k': 110, 'prev_k': 158},
    '2025-09-05': {'nfp_k': 42, 'consensus_k': 75, 'prev_k': 140},
    '2025-10-03': {'nfp_k': 119, 'consensus_k': 50, 'prev_k': 22},
    '2025-11-07': {'nfp_k': 146, 'consensus_k': 130, 'prev_k': 119},
    '2025-12-05': {'nfp_k': 168, 'consensus_k': 140, 'prev_k': 146},
    '2026-01-09': {'nfp_k': 198, 'consensus_k': 165, 'prev_k': 168},
    '2026-02-06': {'nfp_k': 172, 'consensus_k': 170, 'prev_k': 198},
    '2026-03-06': {'nfp_k': 151, 'consensus_k': 160, 'prev_k': 172},
    '2026-04-03': {'nfp_k': 185, 'consensus_k': 135, 'prev_k': 151},
    '2026-05-01': {'nfp_k': 167, 'consensus_k': 130, 'prev_k': 185},
}

# ═══════════════════════════════════════════════════════════════
# FED FUNDS RATE (approximate monthly, for real rate calc)
# ═══════════════════════════════════════════════════════════════
FED_FUNDS_MONTHLY = {
    '2018-01': 1.42, '2018-02': 1.42, '2018-03': 1.42, '2018-04': 1.69,
    '2018-05': 1.69, '2018-06': 1.92, '2018-07': 1.92, '2018-08': 1.92,
    '2018-09': 2.00, '2018-10': 2.19, '2018-11': 2.19, '2018-12': 2.40,
    '2019-01': 2.40, '2019-02': 2.40, '2019-03': 2.40, '2019-04': 2.40,
    '2019-05': 2.40, '2019-06': 2.40, '2019-07': 2.40, '2019-08': 2.13,
    '2019-09': 2.13, '2019-10': 1.83, '2019-11': 1.55, '2019-12': 1.55,
    '2020-01': 1.55, '2020-02': 1.55, '2020-03': 0.65, '2020-04': 0.05,
    '2020-05': 0.05, '2020-06': 0.05, '2020-07': 0.05, '2020-08': 0.05,
    '2020-09': 0.05, '2020-10': 0.05, '2020-11': 0.05, '2020-12': 0.05,
    '2021-01': 0.05, '2021-02': 0.05, '2021-03': 0.05, '2021-04': 0.05,
    '2021-05': 0.05, '2021-06': 0.05, '2021-07': 0.05, '2021-08': 0.05,
    '2021-09': 0.05, '2021-10': 0.05, '2021-11': 0.05, '2021-12': 0.08,
    '2022-01': 0.08, '2022-02': 0.08, '2022-03': 0.20, '2022-04': 0.33,
    '2022-05': 0.83, '2022-06': 1.21, '2022-07': 1.68, '2022-08': 2.35,
    '2022-09': 2.56, '2022-10': 3.08, '2022-11': 3.83, '2022-12': 4.33,
    '2023-01': 4.50, '2023-02': 4.57, '2023-03': 4.83, '2023-04': 5.00,
    '2023-05': 5.08, '2023-06': 5.08, '2023-07': 5.12, '2023-08': 5.33,
    '2023-09': 5.33, '2023-10': 5.33, '2023-11': 5.33, '2023-12': 5.33,
    '2024-01': 5.33, '2024-02': 5.33, '2024-03': 5.33, '2024-04': 5.33,
    '2024-05': 5.33, '2024-06': 5.33, '2024-07': 5.33, '2024-08': 5.33,
    '2024-09': 5.08, '2024-10': 4.83, '2024-11': 4.58, '2024-12': 4.33,
    '2025-01': 4.33, '2025-02': 4.33, '2025-03': 4.33, '2025-04': 4.33,
    '2025-05': 4.33, '2025-06': 4.33, '2025-07': 4.33, '2025-08': 4.33,
    '2025-09': 4.33, '2025-10': 4.33, '2025-11': 4.33, '2025-12': 4.33,
    '2026-01': 4.33, '2026-02': 4.33, '2026-03': 4.33, '2026-04': 4.33,
    '2026-05': 4.33,
}

# Claims monthly avg (for labor context)
CLAIMS_MONTHLY = {
    '2018-01': 230, '2018-06': 220, '2018-12': 215,
    '2019-01': 210, '2019-06': 215, '2019-12': 220,
    '2020-01': 210, '2020-03': 300, '2020-06': 1500, '2020-12': 800,
    '2021-01': 900, '2021-06': 400, '2021-12': 200,
    '2022-01': 210, '2022-06': 195, '2022-12': 195,
    '2023-01': 190, '2023-06': 215, '2023-12': 210,
    '2024-01': 210, '2024-06': 225, '2024-12': 215,
    '2025-01': 205, '2025-06': 230, '2025-12': 199,
    '2026-01': 210, '2026-04': 200, '2026-05': 200,
}


# ═══════════════════════════════════════════════════════════════
# CURRENT M22: Lookup-table approach
# ═══════════════════════════════════════════════════════════════

def classify_ppi_dir(ppi_yoy, prior_yoy):
    delta = ppi_yoy - prior_yoy
    if delta > 0.2: return 'RISING'
    elif delta < -0.2: return 'FALLING'
    return 'FLAT'

def classify_cpi(cpi_yoy):
    if cpi_yoy >= 3.5: return 'HOT'
    elif cpi_yoy >= 2.5: return 'WARM'
    return 'COOL'

def classify_fed(fed_rate, ppi_yoy):
    # Simplified: if real rate > 0.5 = tightening, < -0.5 = easing, else holding
    real = fed_rate - ppi_yoy
    if real > 1.0: return 'HIKING'
    elif real < -0.5: return 'CUTTING'
    return 'HOLDING'

def classify_labor(claims):
    if claims < 210: return 'GOLDILOCKS'
    elif claims < 240: return 'NORMAL'
    elif claims < 280: return 'SOFTENING'
    return 'CRISIS'

# M22 regime matrix (subset - only the cells that matter for this test)
REGIME_MATRIX = {
    ('FALLING', 'CUTTING', 'GOLDILOCKS'): 0.90,
    ('FALLING', 'CUTTING', 'NORMAL'): 0.85,
    ('FALLING', 'CUTTING', 'SOFTENING'): 0.70,
    ('FALLING', 'CUTTING', 'CRISIS'): 0.55,
    ('FALLING', 'HOLDING', 'GOLDILOCKS'): 0.70,
    ('FALLING', 'HOLDING', 'NORMAL'): 0.65,
    ('FALLING', 'HOLDING', 'SOFTENING'): 0.55,
    ('FALLING', 'HOLDING', 'CRISIS'): 0.45,
    ('FALLING', 'HIKING', 'GOLDILOCKS'): 0.45,
    ('FALLING', 'HIKING', 'NORMAL'): 0.40,
    ('FALLING', 'HIKING', 'SOFTENING'): 0.30,
    ('FALLING', 'HIKING', 'CRISIS'): 0.20,
    ('RISING', 'CUTTING', 'GOLDILOCKS'): 0.80,
    ('RISING', 'CUTTING', 'NORMAL'): 0.75,
    ('RISING', 'CUTTING', 'SOFTENING'): 0.60,
    ('RISING', 'CUTTING', 'CRISIS'): 0.50,
    ('RISING', 'HOLDING', 'GOLDILOCKS'): 0.35,
    ('RISING', 'HOLDING', 'NORMAL'): 0.40,
    ('RISING', 'HOLDING', 'SOFTENING'): 0.50,
    ('RISING', 'HOLDING', 'CRISIS'): 0.45,
    ('RISING', 'HIKING', 'GOLDILOCKS'): 0.30,
    ('RISING', 'HIKING', 'NORMAL'): 0.25,
    ('RISING', 'HIKING', 'SOFTENING'): 0.20,
    ('RISING', 'HIKING', 'CRISIS'): 0.10,
    ('FLAT', 'CUTTING', 'GOLDILOCKS'): 0.80,
    ('FLAT', 'CUTTING', 'NORMAL'): 0.75,
    ('FLAT', 'CUTTING', 'SOFTENING'): 0.60,
    ('FLAT', 'CUTTING', 'CRISIS'): 0.50,
    ('FLAT', 'HOLDING', 'GOLDILOCKS'): 0.60,
    ('FLAT', 'HOLDING', 'NORMAL'): 0.55,
    ('FLAT', 'HOLDING', 'SOFTENING'): 0.45,
    ('FLAT', 'HOLDING', 'CRISIS'): 0.35,
    ('FLAT', 'HIKING', 'GOLDILOCKS'): 0.40,
    ('FLAT', 'HIKING', 'NORMAL'): 0.35,
    ('FLAT', 'HIKING', 'SOFTENING'): 0.25,
    ('FLAT', 'HIKING', 'CRISIS'): 0.15,
}

# CPI overlay adjustments
CPI_OVERLAY = {
    ('STAGFLATION_TRAPPED', 'COOL'): +0.20,
    ('STAGFLATION_TRAPPED', 'HOT'): -0.10,
    ('GOLDILOCKS', 'COOL'): +0.05,
    ('GOLDILOCKS', 'HOT'): -0.15,
    ('INFLATION_SHOCK', 'COOL'): +0.15,
    ('INFLATION_SHOCK', 'HOT'): -0.05,
}


def m22_lookup_score(ppi_yoy, prior_ppi, cpi_yoy, fed_rate, claims):
    """Current M22 approach: lookup table."""
    ppi_dir = classify_ppi_dir(ppi_yoy, prior_ppi)
    fed = classify_fed(fed_rate, ppi_yoy)
    labor = classify_labor(claims)

    key = (ppi_dir, fed, labor)
    score = REGIME_MATRIX.get(key, 0.50)

    # CPI overlay (simplified)
    cpi_class = classify_cpi(cpi_yoy)
    # Determine regime name for overlay lookup
    if ppi_dir == 'RISING' and fed == 'HOLDING' and labor in ('GOLDILOCKS', 'NORMAL'):
        regime = 'STAGFLATION_TRAPPED'
    elif ppi_dir == 'RISING' and fed == 'HIKING':
        regime = 'INFLATION_SHOCK'
    elif ppi_dir == 'FALLING' and fed == 'CUTTING' and labor in ('GOLDILOCKS', 'NORMAL'):
        regime = 'GOLDILOCKS'
    else:
        regime = 'NEUTRAL'

    overlay_key = (regime, cpi_class)
    score += CPI_OVERLAY.get(overlay_key, 0.0)

    # Real rate modifier
    real = fed_rate - ppi_yoy
    if real > 2.0: score -= 0.08
    elif real > 1.0: score -= 0.03
    elif real < -1.0: score += 0.05
    elif real < -3.0: score += 0.08

    return max(0.05, min(0.95, score))


# ═══════════════════════════════════════════════════════════════
# PROPOSED: Parametric model (continuous, no lookup tables)
# ═══════════════════════════════════════════════════════════════

def m22_parametric_score(ppi_yoy, prior_ppi, cpi_yoy, fed_rate, claims):
    """Proposed parametric model: continuous functions, no lookup tables.

    Core idea: score = base + inflation_drag + labor_boost + real_rate_adj

    Where:
      - inflation_drag: continuous function of PPI + CPI levels
      - labor_boost: continuous function of claims
      - real_rate_adj: continuous function of (fed - ppi)
    """
    # Base: 0.65 (neutral-positive, like a balanced macro environment)
    score = 0.65

    # 1. Inflation drag (PPI + CPI combined)
    #    Higher inflation = lower score. Continuous, not discrete.
    #    PPI > 3% starts hurting. CPI > 3% amplifies. CPI < 2.5% helps.
    inflation_level = (ppi_yoy + cpi_yoy) / 2  # average of PPI + CPI
    # Sigmoid-like: centered at 3%, scale factor 0.08 per % point above 3
    inflation_drag = -0.08 * max(0, inflation_level - 2.5)
    # Bonus for cool inflation
    if inflation_level < 2.0:
        inflation_drag += 0.05
    score += inflation_drag

    # 2. PPI momentum (acceleration/deceleration)
    #    Rising PPI = additional drag (Fed behind the curve)
    #    Falling PPI = additional boost (disinflation)
    ppi_delta = ppi_yoy - prior_ppi
    momentum_adj = -0.04 * ppi_delta  # each 1pp rise = -0.04, each 1pp fall = +0.04
    score += momentum_adj

    # 3. Labor boost (continuous, not categorical)
    #    Lower claims = better. Log-scale: 200K is great, 300K is bad.
    #    Center at 220K, scale factor 0.03 per 20K deviation
    labor_adj = 0.03 * (220 - claims) / 20
    score += labor_adj

    # 4. Real rate (continuous, not bucketed)
    #    Positive real rates = tight (drag). Negative = stimulative (boost).
    real_rate = fed_rate - ppi_yoy
    real_adj = -0.03 * real_rate  # each 1pp of tightness = -0.03
    score += real_adj

    return max(0.05, min(0.95, score))


# ═══════════════════════════════════════════════════════════════
# PROPOSED: Consensus surprise signal
# ═══════════════════════════════════════════════════════════════

def surprise_score(actual, consensus, label='PPI'):
    """Compute consensus surprise signal.

    Hotter than expected (actual > consensus) = bearish (negative)
    Cooler than expected (actual < consensus) = bullish (positive)

    Scale: ±0.04 per pp of surprise, capped at ±0.12
    """
    if consensus is None:
        return 0.0
    surprise = actual - consensus
    if abs(surprise) < 0.1:
        return 0.0  # inline, no signal
    adj = -surprise * 0.04  # hot surprise = negative (bearish)
    return max(-0.12, min(0.12, adj))


# ═══════════════════════════════════════════════════════════════
# BACKTEST ENGINE
# ═══════════════════════════════════════════════════════════════

def load_eth_data(csv_path):
    """Load ETH 15m data and compute 24h returns for each release date."""
    print(f"Loading {csv_path}...")
    df = pd.read_csv(csv_path)
    df['Open time'] = pd.to_datetime(df['Open time'])
    df = df.sort_values('Open time').reset_index(drop=True)
    print(f"  Loaded {len(df)} bars: {df['Open time'].iloc[0]} → {df['Open time'].iloc[-1]}")
    return df


def compute_return(df, release_date, hours=24):
    """Compute ETH return from release time (13:30 UTC) over N hours."""
    release_dt = pd.Timestamp(f"{release_date} 13:30:00")
    exit_dt = release_dt + timedelta(hours=hours)

    # Find the bar at or just before release
    pre_mask = df['Open time'] <= release_dt
    if pre_mask.sum() == 0:
        return None
    pre_price = float(df[pre_mask]['Close'].iloc[-1])

    # Find the bar at or just after exit
    post_mask = df['Open time'] >= exit_dt
    if post_mask.sum() == 0:
        return None
    post_price = float(df[post_mask]['Close'].iloc[0])

    return (post_price - pre_price) / pre_price * 100


def get_claims_for_date(date_str):
    """Get approximate claims level for a date."""
    y, m = int(date_str[:4]), int(date_str[5:7])
    # Try exact month, then previous months
    for offset in range(0, 4):
        key = f"{y:04d}-{m-offset:02d}" if m - offset > 0 else f"{y-1:04d}-{12+(m-offset):02d}"
        if key in CLAIMS_MONTHLY:
            return CLAIMS_MONTHLY[key]
    return 220  # default


def get_fed_rate(date_str):
    """Get fed funds rate for a date."""
    y, m = int(date_str[:4]), int(date_str[5:7])
    key = f"{y:04d}-{m:02d}"
    return FED_FUNDS_MONTHLY.get(key, 2.0)


# ═══════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════

def main():
    df = load_eth_data('eth_15m_merged.csv')

    # Build unified release list with type and data
    releases = []

    for date, data in PPI_RELEASES.items():
        releases.append({
            'date': date, 'type': 'PPI',
            'actual': data['ppi_yoy'], 'consensus': data['consensus_yoy'],
            'prior': data['prior_yoy'],
        })

    for date, data in CPI_RELEASES.items():
        releases.append({
            'date': date, 'type': 'CPI',
            'actual': data['cpi_yoy'], 'consensus': data['consensus_yoy'],
            'prior': data['prior_yoy'],
        })

    for date, data in NFP_RELEASES.items():
        # NFP: use surprise as % deviation from consensus
        surprise_pct = (data['nfp_k'] - data['consensus_k']) / max(abs(data['consensus_k']), 1) * 100
        releases.append({
            'date': date, 'type': 'NFP',
            'actual': data['nfp_k'], 'consensus': data['consensus_k'],
            'prior': data['prev_k'],
            'surprise_pct': surprise_pct,
        })

    # Sort by date
    releases.sort(key=lambda x: x['date'])

    # Compute returns and scores
    results = []
    for rel in releases:
        ret_24h = compute_return(df, rel['date'], hours=24)
        if ret_24h is None:
            continue

        claims = get_claims_for_date(rel['date'])
        fed_rate = get_fed_rate(rel['date'])

        # For NFP, we need PPI/CPI context from nearby releases
        # Use the most recent PPI before this NFP
        ppi_yoy, ppi_prior = None, None
        cpi_yoy, cpi_prior = None, None
        for r in reversed(releases):
            if r['date'] < rel['date'] and r['type'] == 'PPI':
                ppi_yoy, ppi_prior = r['actual'], r['prior']
                break
        for r in reversed(releases):
            if r['date'] < rel['date'] and r['type'] == 'CPI':
                cpi_yoy, cpi_prior = r['actual'], r['prior']
                break

        # If we don't have PPI/CPI context, skip scoring (but still record)
        if ppi_yoy is None or cpi_yoy is None:
            # Use the actual release's own data as proxy
            if rel['type'] == 'PPI':
                ppi_yoy, ppi_prior = rel['actual'], rel['prior']
                cpi_yoy = ppi_yoy  # rough proxy
            elif rel['type'] == 'CPI':
                cpi_yoy, cpi_prior = rel['actual'], rel['prior']
                ppi_yoy = cpi_yoy
                ppi_prior = cpi_prior
            else:
                ppi_yoy, ppi_prior = 2.0, 2.0
                cpi_yoy = 2.0

        # Current M22 lookup score
        lookup = m22_lookup_score(ppi_yoy, ppi_prior, cpi_yoy, fed_rate, claims)

        # Proposed parametric score
        parametric = m22_parametric_score(ppi_yoy, ppi_prior, cpi_yoy, fed_rate, claims)

        # Surprise signal (only for PPI/CPI where we have consensus)
        surp = 0.0
        if rel['type'] in ('PPI', 'CPI') and rel.get('consensus') is not None:
            surp = surprise_score(rel['actual'], rel['consensus'], rel['type'])
        elif rel['type'] == 'NFP' and rel.get('surprise_pct') is not None:
            # NFP surprise: positive surprise = strong labor = slight bearish (Fed can't cut)
            surp = -rel['surprise_pct'] * 0.002  # scale down (NFP surprise is in %)
            surp = max(-0.12, min(0.12, surp))

        # Combined: parametric + surprise
        parametric_surprise = parametric + surp

        results.append({
            'date': rel['date'],
            'type': rel['type'],
            'actual': rel['actual'],
            'consensus': rel.get('consensus'),
            'return_24h': ret_24h,
            'lookup_score': lookup,
            'parametric_score': parametric,
            'surprise': surp,
            'parametric_surprise': parametric_surprise,
            'ppi_yoy': ppi_yoy,
            'cpi_yoy': cpi_yoy,
            'fed_rate': fed_rate,
            'claims': claims,
        })

    df_results = pd.DataFrame(results)

    # ═══════════════════════════════════════════════════════════════
    # ANALYSIS
    # ═══════════════════════════════════════════════════════════════

    print("\n" + "=" * 70)
    print("  BACKTEST: M22 LOOKUP-TABLE vs PARAMETRIC MODEL + SURPRISE")
    print("=" * 70)
    print(f"\n  Total releases: {len(df_results)}")
    print(f"  Period: {df_results['date'].min()} → {df_results['date'].max()}")
    print(f"  Types: PPI={len(df_results[df_results['type']=='PPI'])}, "
          f"CPI={len(df_results[df_results['type']=='CPI'])}, "
          f"NFP={len(df_results[df_results['type']=='NFP'])}")

    # ── H1: Does the score predict direction? ──
    print("\n" + "─" * 70)
    print("  H1: SCORE vs 24h RETURN — Direction Prediction")
    print("─" * 70)

    for name, col in [('M22 Lookup', 'lookup_score'),
                       ('Parametric', 'parametric_score'),
                       ('Parametric+Surprise', 'parametric_surprise')]:
        # Direction accuracy: score > 0.5 predicts positive return
        predicted_up = df_results[col] > 0.5
        actual_up = df_results['return_24h'] > 0
        direction_correct = (predicted_up == actual_up).mean()

        # Correlation
        corr, p_val = stats.pearsonr(df_results[col], df_results['return_24h'])

        # Spread: avg return when score > 0.5 vs < 0.5
        avg_up = df_results[predicted_up]['return_24h'].mean()
        avg_down = df_results[~predicted_up]['return_24h'].mean()
        spread = avg_up - avg_down

        # High-confidence (>0.65 or <0.35) accuracy
        high_conf = df_results[(df_results[col] > 0.65) | (df_results[col] < 0.35)]
        if len(high_conf) > 0:
            hc_predicted_up = high_conf[col] > 0.5
            hc_actual_up = high_conf['return_24h'] > 0
            hc_accuracy = (hc_predicted_up == hc_actual_up).mean()
        else:
            hc_accuracy = 0

        print(f"\n  {name}:")
        print(f"    Direction accuracy:    {direction_correct:.1%}")
        print(f"    Correlation:           r={corr:.3f}  p={p_val:.4f}")
        print(f"    Spread (up-down):      {spread:+.2f}%")
        print(f"    High-conf accuracy:    {hc_accuracy:.1%}  (n={len(high_conf)})")

    # ── H2: Does surprise add alpha? ──
    print("\n" + "─" * 70)
    print("  H2: CONSENSUS SURPRISE — Does it add alpha?")
    print("─" * 70)

    # Surprise vs return (only for PPI/CPI where we have consensus)
    ppi_cpi = df_results[df_results['type'].isin(['PPI', 'CPI'])].copy()
    if len(ppi_cpi) > 0:
        surp_corr, surp_p = stats.pearsonr(ppi_cpi['surprise'], ppi_cpi['return_24h'])

        # Hot surprise (actual > consensus) vs cool surprise
        hot = ppi_cpi[ppi_cpi['surprise'] < -0.02]  # bearish surprise (hot)
        cool = ppi_cpi[ppi_cpi['surprise'] > 0.02]  # bullish surprise (cool)
        inline = ppi_cpi[ppi_cpi['surprise'].abs() <= 0.02]

        print(f"\n  PPI/CPI Surprise vs 24h Return:")
        print(f"    Correlation:           r={surp_corr:.3f}  p={surp_p:.4f}")
        print(f"    Hot surprise (n={len(hot)}):    avg return = {hot['return_24h'].mean():+.2f}%")
        print(f"    Cool surprise (n={len(cool)}):  avg return = {cool['return_24h'].mean():+.2f}%")
        print(f"    Inline (n={len(inline)}):       avg return = {inline['return_24h'].mean():+.2f}%")
        print(f"    Hot-Cool spread:       {hot['return_24h'].mean() - cool['return_24h'].mean():+.2f}%")

    # ── H3: Out-of-sample split ──
    print("\n" + "─" * 70)
    print("  H3: OUT-OF-SAMPLE — Train 2018-2023, Test 2024-2026")
    print("─" * 70)

    train = df_results[df_results['date'] < '2024-01-01']
    test = df_results[df_results['date'] >= '2024-01-01']

    for name, col in [('M22 Lookup', 'lookup_score'),
                       ('Parametric', 'parametric_score'),
                       ('Parametric+Surprise', 'parametric_surprise')]:
        for label, subset in [('In-sample (2018-2023)', train),
                               ('Out-of-sample (2024-2026)', test)]:
            if len(subset) == 0:
                continue
            predicted_up = subset[col] > 0.5
            actual_up = subset['return_24h'] > 0
            accuracy = (predicted_up == actual_up).mean()
            corr, p_val = stats.pearsonr(subset[col], subset['return_24h'])
            spread = subset[predicted_up]['return_24h'].mean() - subset[~predicted_up]['return_24h'].mean()

            print(f"\n  {name} — {label}:")
            print(f"    Direction accuracy: {accuracy:.1%}  (n={len(subset)})")
            print(f"    Correlation:        r={corr:.3f}  p={p_val:.4f}")
            print(f"    Spread:             {spread:+.2f}%")

    # ── By release type ──
    print("\n" + "─" * 70)
    print("  BY RELEASE TYPE — Which model wins per event?")
    print("─" * 70)

    for rtype in ['PPI', 'CPI', 'NFP']:
        subset = df_results[df_results['type'] == rtype]
        if len(subset) < 5:
            continue

        print(f"\n  {rtype} (n={len(subset)}):")
        for name, col in [('Lookup', 'lookup_score'),
                           ('Parametric', 'parametric_score'),
                           ('Param+Surprise', 'parametric_surprise')]:
            predicted_up = subset[col] > 0.5
            actual_up = subset['return_24h'] > 0
            accuracy = (predicted_up == actual_up).mean()
            corr, _ = stats.pearsonr(subset[col], subset['return_24h'])
            print(f"    {name:15s}  acc={accuracy:.1%}  r={corr:.3f}")

    # ── Regime distribution ──
    print("\n" + "─" * 70)
    print("  REGIME DISTRIBUTION — How often does each model produce extreme scores?")
    print("─" * 70)

    for name, col in [('M22 Lookup', 'lookup_score'), ('Parametric', 'parametric_score')]:
        very_bull = (df_results[col] > 0.70).sum()
        bull = ((df_results[col] > 0.55) & (df_results[col] <= 0.70)).sum()
        neutral = ((df_results[col] >= 0.45) & (df_results[col] <= 0.55)).sum()
        bear = ((df_results[col] >= 0.30) & (df_results[col] < 0.45)).sum()
        very_bear = (df_results[col] < 0.30).sum()
        print(f"\n  {name}:")
        print(f"    Very Bull (>0.70): {very_bull:3d} ({very_bull/len(df_results)*100:.0f}%)")
        print(f"    Bull (0.55-0.70):  {bull:3d} ({bull/len(df_results)*100:.0f}%)")
        print(f"    Neutral (0.45-0.55):{neutral:3d} ({neutral/len(df_results)*100:.0f}%)")
        print(f"    Bear (0.30-0.45):  {bear:3d} ({bear/len(df_results)*100:.0f}%)")
        print(f"    Very Bear (<0.30): {very_bear:3d} ({very_bear/len(df_results)*100:.0f}%)")

    # ── Summary ──
    print("\n" + "=" * 70)
    print("  SUMMARY")
    print("=" * 70)

    # Overall winner
    models = {
        'M22 Lookup': 'lookup_score',
        'Parametric': 'parametric_score',
        'Parametric+Surprise': 'parametric_surprise',
    }
    print(f"\n  {'Model':<25} {'Accuracy':>8} {'Corr':>8} {'OOS Acc':>8} {'OOS Corr':>8}")
    print(f"  {'─'*25} {'─'*8} {'─'*8} {'─'*8} {'─'*8}")
    for name, col in models.items():
        pred = df_results[col] > 0.5
        act = df_results['return_24h'] > 0
        acc = (pred == act).mean()
        corr, _ = stats.pearsonr(df_results[col], df_results['return_24h'])

        oos = df_results[df_results['date'] >= '2024-01-01']
        if len(oos) > 5:
            oos_pred = oos[col] > 0.5
            oos_act = oos['return_24h'] > 0
            oos_acc = (oos_pred == oos_act).mean()
            oos_corr, _ = stats.pearsonr(oos[col], oos['return_24h'])
        else:
            oos_acc, oos_corr = 0, 0

        print(f"  {name:<25} {acc:>7.1%} {corr:>7.3f} {oos_acc:>7.1%} {oos_corr:>7.3f}")


if __name__ == '__main__':
    main()

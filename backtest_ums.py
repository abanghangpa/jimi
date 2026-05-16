#!/usr/bin/env python3
"""
Prompt A + B: Michigan Consumer Sentiment Backtest & Session Transmission Chain
===============================================================================
ETH/USDT 15m data from 2018 to today.

Released 2nd & 4th Friday of month, 10:00 ET (14:00 UTC EDT / 15:00 UTC EST).
Key sub-index: 5-Year Inflation Expectations.
Weekend crypto continuity: Friday close → Monday open theme.
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

# ═══════════════════════════════════════════════════════════════
# MICHIGAN CONSUMER SENTIMENT RELEASE DATES (10:00 ET = 14:00 UTC)
# Preliminary (prelim) on 2nd Friday, Final on 4th Friday
# Format: date -> headline, 5yr_inflation_exp, consensus, prior
# ═══════════════════════════════════════════════════════════════

UMS_RELEASES = {
    # 2018 — Prelim + Final
    '2018-01-12': {'headline': 95.7, 'infl_5yr': 2.5, 'consensus': 97.0, 'prior': 96.8, 'type': 'prelim'},
    '2018-01-26': {'headline': 95.7, 'infl_5yr': 2.5, 'consensus': 95.0, 'prior': 95.7, 'type': 'final'},
    '2018-02-16': {'headline': 99.9, 'infl_5yr': 2.5, 'consensus': 95.5, 'prior': 95.7, 'type': 'prelim'},
    '2018-02-23': {'headline': 99.7, 'infl_5yr': 2.5, 'consensus': 99.5, 'prior': 99.9, 'type': 'final'},
    '2018-03-16': {'headline': 102.0, 'infl_5yr': 2.5, 'consensus': 99.3, 'prior': 99.7, 'type': 'prelim'},
    '2018-03-23': {'headline': 101.4, 'infl_5yr': 2.5, 'consensus': 102.0, 'prior': 102.0, 'type': 'final'},
    '2018-04-13': {'headline': 97.8, 'infl_5yr': 2.5, 'consensus': 100.5, 'prior': 101.4, 'type': 'prelim'},
    '2018-04-27': {'headline': 98.8, 'infl_5yr': 2.5, 'consensus': 98.0, 'prior': 97.8, 'type': 'final'},
    '2018-05-11': {'headline': 98.8, 'infl_5yr': 2.5, 'consensus': 98.5, 'prior': 98.8, 'type': 'prelim'},
    '2018-05-25': {'headline': 98.0, 'infl_5yr': 2.5, 'consensus': 98.8, 'prior': 98.8, 'type': 'final'},
    '2018-06-15': {'headline': 99.3, 'infl_5yr': 2.6, 'consensus': 98.5, 'prior': 98.0, 'type': 'prelim'},
    '2018-06-29': {'headline': 98.2, 'infl_5yr': 2.6, 'consensus': 99.0, 'prior': 99.3, 'type': 'final'},
    '2018-07-13': {'headline': 97.1, 'infl_5yr': 2.6, 'consensus': 98.0, 'prior': 98.2, 'type': 'prelim'},
    '2018-07-27': {'headline': 97.9, 'infl_5yr': 2.6, 'consensus': 97.0, 'prior': 97.1, 'type': 'final'},
    '2018-08-17': {'headline': 95.3, 'infl_5yr': 2.9, 'consensus': 97.5, 'prior': 97.9, 'type': 'prelim'},
    '2018-08-31': {'headline': 96.2, 'infl_5yr': 2.9, 'consensus': 95.5, 'prior': 95.3, 'type': 'final'},
    '2018-09-14': {'headline': 100.8, 'infl_5yr': 2.8, 'consensus': 96.5, 'prior': 96.2, 'type': 'prelim'},
    '2018-09-28': {'headline': 100.1, 'infl_5yr': 2.8, 'consensus': 100.5, 'prior': 100.8, 'type': 'final'},
    '2018-10-12': {'headline': 99.0, 'infl_5yr': 2.8, 'consensus': 100.5, 'prior': 100.1, 'type': 'prelim'},
    '2018-10-26': {'headline': 98.6, 'infl_5yr': 2.8, 'consensus': 99.0, 'prior': 99.0, 'type': 'final'},
    '2018-11-16': {'headline': 97.5, 'infl_5yr': 2.8, 'consensus': 98.0, 'prior': 98.6, 'type': 'prelim'},
    '2018-11-30': {'headline': 97.5, 'infl_5yr': 2.8, 'consensus': 97.5, 'prior': 97.5, 'type': 'final'},
    '2018-12-14': {'headline': 97.5, 'infl_5yr': 2.7, 'consensus': 97.0, 'prior': 97.5, 'type': 'prelim'},
    # 2019
    '2019-01-18': {'headline': 90.7, 'infl_5yr': 2.7, 'consensus': 97.0, 'prior': 98.3, 'type': 'prelim'},
    '2019-01-25': {'headline': 91.2, 'infl_5yr': 2.7, 'consensus': 90.5, 'prior': 90.7, 'type': 'final'},
    '2019-02-15': {'headline': 95.5, 'infl_5yr': 2.5, 'consensus': 92.0, 'prior': 91.2, 'type': 'prelim'},
    '2019-02-22': {'headline': 93.8, 'infl_5yr': 2.5, 'consensus': 95.5, 'prior': 95.5, 'type': 'final'},
    '2019-03-15': {'headline': 97.8, 'infl_5yr': 2.5, 'consensus': 95.5, 'prior': 93.8, 'type': 'prelim'},
    '2019-03-29': {'headline': 98.4, 'infl_5yr': 2.5, 'consensus': 97.8, 'prior': 97.8, 'type': 'final'},
    '2019-04-12': {'headline': 96.9, 'infl_5yr': 2.4, 'consensus': 98.0, 'prior': 98.4, 'type': 'prelim'},
    '2019-04-26': {'headline': 97.2, 'infl_5yr': 2.4, 'consensus': 97.0, 'prior': 96.9, 'type': 'final'},
    '2019-05-17': {'headline': 102.4, 'infl_5yr': 2.6, 'consensus': 97.5, 'prior': 97.2, 'type': 'prelim'},
    '2019-05-31': {'headline': 100.0, 'infl_5yr': 2.6, 'consensus': 101.5, 'prior': 102.4, 'type': 'final'},
    '2019-06-14': {'headline': 97.9, 'infl_5yr': 2.6, 'consensus': 98.0, 'prior': 100.0, 'type': 'prelim'},
    '2019-06-28': {'headline': 98.2, 'infl_5yr': 2.6, 'consensus': 98.0, 'prior': 97.9, 'type': 'final'},
    '2019-07-12': {'headline': 98.4, 'infl_5yr': 2.6, 'consensus': 98.5, 'prior': 98.2, 'type': 'prelim'},
    '2019-07-26': {'headline': 98.4, 'infl_5yr': 2.6, 'consensus': 98.5, 'prior': 98.4, 'type': 'final'},
    '2019-08-16': {'headline': 92.1, 'infl_5yr': 2.6, 'consensus': 97.0, 'prior': 98.4, 'type': 'prelim'},
    '2019-08-30': {'headline': 89.8, 'infl_5yr': 2.6, 'consensus': 92.0, 'prior': 92.1, 'type': 'final'},
    '2019-09-13': {'headline': 92.0, 'infl_5yr': 2.5, 'consensus': 90.5, 'prior': 89.8, 'type': 'prelim'},
    '2019-09-27': {'headline': 93.2, 'infl_5yr': 2.5, 'consensus': 92.0, 'prior': 92.0, 'type': 'final'},
    '2019-10-11': {'headline': 96.0, 'infl_5yr': 2.5, 'consensus': 92.0, 'prior': 93.2, 'type': 'prelim'},
    '2019-10-25': {'headline': 95.5, 'infl_5yr': 2.5, 'consensus': 96.0, 'prior': 96.0, 'type': 'final'},
    '2019-11-15': {'headline': 96.8, 'infl_5yr': 2.4, 'consensus': 95.5, 'prior': 95.5, 'type': 'prelim'},
    '2019-11-27': {'headline': 96.8, 'infl_5yr': 2.4, 'consensus': 96.8, 'prior': 96.8, 'type': 'final'},
    '2019-12-13': {'headline': 99.3, 'infl_5yr': 2.3, 'consensus': 97.0, 'prior': 96.8, 'type': 'prelim'},
    # 2020
    '2020-01-17': {'headline': 99.8, 'infl_5yr': 2.5, 'consensus': 99.0, 'prior': 99.3, 'type': 'prelim'},
    '2020-01-31': {'headline': 99.8, 'infl_5yr': 2.5, 'consensus': 99.8, 'prior': 99.8, 'type': 'final'},
    '2020-02-14': {'headline': 100.9, 'infl_5yr': 2.5, 'consensus': 99.5, 'prior': 99.8, 'type': 'prelim'},
    '2020-02-28': {'headline': 101.0, 'infl_5yr': 2.5, 'consensus': 100.8, 'prior': 100.9, 'type': 'final'},
    '2020-03-13': {'headline': 95.9, 'infl_5yr': 2.3, 'consensus': 95.0, 'prior': 101.0, 'type': 'prelim'},
    '2020-03-27': {'headline': 89.1, 'infl_5yr': 2.3, 'consensus': 90.0, 'prior': 95.9, 'type': 'final'},
    '2020-04-17': {'headline': 71.8, 'infl_5yr': 2.4, 'consensus': 75.0, 'prior': 89.1, 'type': 'prelim'},
    '2020-05-01': {'headline': 72.3, 'infl_5yr': 2.4, 'consensus': 72.0, 'prior': 71.8, 'type': 'final'},
    '2020-05-15': {'headline': 73.7, 'infl_5yr': 2.6, 'consensus': 74.0, 'prior': 72.3, 'type': 'prelim'},
    '2020-05-29': {'headline': 72.3, 'infl_5yr': 2.6, 'consensus': 74.0, 'prior': 73.7, 'type': 'final'},
    '2020-06-12': {'headline': 78.9, 'infl_5yr': 2.7, 'consensus': 75.0, 'prior': 72.3, 'type': 'prelim'},
    '2020-06-26': {'headline': 78.1, 'infl_5yr': 2.7, 'consensus': 79.0, 'prior': 78.9, 'type': 'final'},
    '2020-07-17': {'headline': 73.2, 'infl_5yr': 2.7, 'consensus': 79.0, 'prior': 78.1, 'type': 'prelim'},
    '2020-07-31': {'headline': 72.5, 'infl_5yr': 2.7, 'consensus': 73.0, 'prior': 73.2, 'type': 'final'},
    '2020-08-14': {'headline': 72.8, 'infl_5yr': 2.7, 'consensus': 72.0, 'prior': 72.5, 'type': 'prelim'},
    '2020-08-28': {'headline': 74.1, 'infl_5yr': 2.7, 'consensus': 72.8, 'prior': 72.8, 'type': 'final'},
    '2020-09-18': {'headline': 78.9, 'infl_5yr': 2.7, 'consensus': 75.0, 'prior': 74.1, 'type': 'prelim'},
    '2020-09-30': {'headline': 80.4, 'infl_5yr': 2.7, 'consensus': 79.0, 'prior': 78.9, 'type': 'final'},
    '2020-10-16': {'headline': 81.2, 'infl_5yr': 2.6, 'consensus': 80.5, 'prior': 80.4, 'type': 'prelim'},
    '2020-10-30': {'headline': 81.8, 'infl_5yr': 2.6, 'consensus': 81.2, 'prior': 81.2, 'type': 'final'},
    '2020-11-13': {'headline': 77.0, 'infl_5yr': 2.6, 'consensus': 82.0, 'prior': 81.8, 'type': 'prelim'},
    '2020-11-25': {'headline': 76.9, 'infl_5yr': 2.6, 'consensus': 77.0, 'prior': 77.0, 'type': 'final'},
    '2020-12-11': {'headline': 81.4, 'infl_5yr': 2.5, 'consensus': 76.5, 'prior': 76.9, 'type': 'prelim'},
    # 2021
    '2021-01-15': {'headline': 79.2, 'infl_5yr': 2.7, 'consensus': 80.0, 'prior': 80.7, 'type': 'prelim'},
    '2021-01-29': {'headline': 79.0, 'infl_5yr': 2.7, 'consensus': 79.2, 'prior': 79.2, 'type': 'final'},
    '2021-02-19': {'headline': 76.2, 'infl_5yr': 2.7, 'consensus': 80.8, 'prior': 79.0, 'type': 'prelim'},
    '2021-02-26': {'headline': 76.8, 'infl_5yr': 2.7, 'consensus': 76.5, 'prior': 76.2, 'type': 'final'},
    '2021-03-12': {'headline': 83.0, 'infl_5yr': 3.0, 'consensus': 78.5, 'prior': 76.8, 'type': 'prelim'},
    '2021-03-26': {'headline': 84.9, 'infl_5yr': 3.0, 'consensus': 83.5, 'prior': 83.0, 'type': 'final'},
    '2021-04-16': {'headline': 86.5, 'infl_5yr': 2.7, 'consensus': 88.0, 'prior': 84.9, 'type': 'prelim'},
    '2021-04-30': {'headline': 88.3, 'infl_5yr': 2.7, 'consensus': 87.5, 'prior': 86.5, 'type': 'final'},
    '2021-05-14': {'headline': 82.8, 'infl_5yr': 3.1, 'consensus': 90.0, 'prior': 88.3, 'type': 'prelim'},
    '2021-05-28': {'headline': 82.9, 'infl_5yr': 3.1, 'consensus': 83.0, 'prior': 82.8, 'type': 'final'},
    '2021-06-11': {'headline': 86.4, 'infl_5yr': 2.8, 'consensus': 84.0, 'prior': 82.9, 'type': 'prelim'},
    '2021-06-25': {'headline': 85.5, 'infl_5yr': 2.8, 'consensus': 86.5, 'prior': 86.4, 'type': 'final'},
    '2021-07-16': {'headline': 80.8, 'infl_5yr': 2.9, 'consensus': 86.5, 'prior': 85.5, 'type': 'prelim'},
    '2021-07-30': {'headline': 81.2, 'infl_5yr': 2.9, 'consensus': 80.8, 'prior': 80.8, 'type': 'final'},
    '2021-08-13': {'headline': 70.2, 'infl_5yr': 3.0, 'consensus': 81.2, 'prior': 81.2, 'type': 'prelim'},
    '2021-08-27': {'headline': 70.3, 'infl_5yr': 3.0, 'consensus': 70.8, 'prior': 70.2, 'type': 'final'},
    '2021-09-17': {'headline': 71.0, 'infl_5yr': 2.9, 'consensus': 72.0, 'prior': 70.3, 'type': 'prelim'},
    '2021-09-30': {'headline': 72.8, 'infl_5yr': 2.9, 'consensus': 71.0, 'prior': 71.0, 'type': 'final'},
    '2021-10-15': {'headline': 71.4, 'infl_5yr': 2.8, 'consensus': 73.0, 'prior': 72.8, 'type': 'prelim'},
    '2021-10-29': {'headline': 71.7, 'infl_5yr': 2.8, 'consensus': 72.0, 'prior': 71.4, 'type': 'final'},
    '2021-11-12': {'headline': 66.8, 'infl_5yr': 3.0, 'consensus': 72.5, 'prior': 71.7, 'type': 'prelim'},
    '2021-11-24': {'headline': 67.4, 'infl_5yr': 3.0, 'consensus': 67.0, 'prior': 66.8, 'type': 'final'},
    '2021-12-10': {'headline': 70.4, 'infl_5yr': 2.9, 'consensus': 68.0, 'prior': 67.4, 'type': 'prelim'},
    # 2022
    '2022-01-14': {'headline': 68.8, 'infl_5yr': 2.9, 'consensus': 70.0, 'prior': 70.6, 'type': 'prelim'},
    '2022-01-28': {'headline': 67.2, 'infl_5yr': 2.9, 'consensus': 68.8, 'prior': 68.8, 'type': 'final'},
    '2022-02-11': {'headline': 61.7, 'infl_5yr': 3.1, 'consensus': 67.5, 'prior': 67.2, 'type': 'prelim'},
    '2022-02-25': {'headline': 62.8, 'infl_5yr': 3.1, 'consensus': 62.0, 'prior': 61.7, 'type': 'final'},
    '2022-03-11': {'headline': 59.7, 'infl_5yr': 3.0, 'consensus': 62.0, 'prior': 62.8, 'type': 'prelim'},
    '2022-03-25': {'headline': 59.4, 'infl_5yr': 3.0, 'consensus': 59.7, 'prior': 59.7, 'type': 'final'},
    '2022-04-14': {'headline': 65.7, 'infl_5yr': 2.9, 'consensus': 59.0, 'prior': 59.4, 'type': 'prelim'},
    '2022-04-29': {'headline': 65.2, 'infl_5yr': 2.9, 'consensus': 65.5, 'prior': 65.7, 'type': 'final'},
    '2022-05-13': {'headline': 59.1, 'infl_5yr': 3.0, 'consensus': 64.0, 'prior': 65.2, 'type': 'prelim'},
    '2022-05-27': {'headline': 58.4, 'infl_5yr': 3.0, 'consensus': 59.0, 'prior': 59.1, 'type': 'final'},
    '2022-06-10': {'headline': 50.2, 'infl_5yr': 3.3, 'consensus': 58.0, 'prior': 58.4, 'type': 'prelim'},
    '2022-06-24': {'headline': 50.0, 'infl_5yr': 3.3, 'consensus': 50.2, 'prior': 50.2, 'type': 'final'},
    '2022-07-15': {'headline': 51.1, 'infl_5yr': 2.8, 'consensus': 49.5, 'prior': 50.0, 'type': 'prelim'},
    '2022-07-29': {'headline': 51.5, 'infl_5yr': 2.8, 'consensus': 51.0, 'prior': 51.1, 'type': 'final'},
    '2022-08-12': {'headline': 55.1, 'infl_5yr': 3.0, 'consensus': 52.5, 'prior': 51.5, 'type': 'prelim'},
    '2022-08-26': {'headline': 58.2, 'infl_5yr': 3.0, 'consensus': 55.5, 'prior': 55.1, 'type': 'final'},
    '2022-09-16': {'headline': 59.5, 'infl_5yr': 2.8, 'consensus': 60.0, 'prior': 58.2, 'type': 'prelim'},
    '2022-09-30': {'headline': 58.6, 'infl_5yr': 2.8, 'consensus': 59.5, 'prior': 59.5, 'type': 'final'},
    '2022-10-14': {'headline': 59.8, 'infl_5yr': 2.9, 'consensus': 59.0, 'prior': 58.6, 'type': 'prelim'},
    '2022-10-28': {'headline': 59.9, 'infl_5yr': 2.9, 'consensus': 59.8, 'prior': 59.8, 'type': 'final'},
    '2022-11-11': {'headline': 54.7, 'infl_5yr': 3.0, 'consensus': 59.5, 'prior': 59.9, 'type': 'prelim'},
    '2022-11-23': {'headline': 56.8, 'infl_5yr': 3.0, 'consensus': 55.0, 'prior': 54.7, 'type': 'final'},
    '2022-12-09': {'headline': 59.1, 'infl_5yr': 2.8, 'consensus': 56.5, 'prior': 56.8, 'type': 'prelim'},
    # 2023
    '2023-01-13': {'headline': 64.6, 'infl_5yr': 2.9, 'consensus': 60.0, 'prior': 59.7, 'type': 'prelim'},
    '2023-01-27': {'headline': 64.9, 'infl_5yr': 2.9, 'consensus': 64.6, 'prior': 64.6, 'type': 'final'},
    '2023-02-10': {'headline': 66.4, 'infl_5yr': 2.9, 'consensus': 65.0, 'prior': 64.9, 'type': 'prelim'},
    '2023-02-24': {'headline': 67.0, 'infl_5yr': 2.9, 'consensus': 66.5, 'prior': 66.4, 'type': 'final'},
    '2023-03-10': {'headline': 63.4, 'infl_5yr': 2.8, 'consensus': 67.0, 'prior': 67.0, 'type': 'prelim'},
    '2023-03-24': {'headline': 62.0, 'infl_5yr': 2.8, 'consensus': 63.5, 'prior': 63.4, 'type': 'final'},
    '2023-04-14': {'headline': 63.5, 'infl_5yr': 2.8, 'consensus': 62.0, 'prior': 62.0, 'type': 'prelim'},
    '2023-04-28': {'headline': 63.7, 'infl_5yr': 2.8, 'consensus': 63.5, 'prior': 63.5, 'type': 'final'},
    '2023-05-12': {'headline': 57.7, 'infl_5yr': 2.9, 'consensus': 63.0, 'prior': 63.7, 'type': 'prelim'},
    '2023-05-26': {'headline': 59.2, 'infl_5yr': 2.9, 'consensus': 58.0, 'prior': 57.7, 'type': 'final'},
    '2023-06-16': {'headline': 63.9, 'infl_5yr': 3.0, 'consensus': 60.0, 'prior': 59.2, 'type': 'prelim'},
    '2023-06-30': {'headline': 64.4, 'infl_5yr': 3.0, 'consensus': 64.0, 'prior': 63.9, 'type': 'final'},
    '2023-07-14': {'headline': 72.6, 'infl_5yr': 2.9, 'consensus': 65.5, 'prior': 64.4, 'type': 'prelim'},
    '2023-07-28': {'headline': 71.6, 'infl_5yr': 2.9, 'consensus': 72.5, 'prior': 72.6, 'type': 'final'},
    '2023-08-11': {'headline': 71.2, 'infl_5yr': 2.9, 'consensus': 71.0, 'prior': 71.6, 'type': 'prelim'},
    '2023-08-25': {'headline': 69.5, 'infl_5yr': 2.9, 'consensus': 71.0, 'prior': 71.2, 'type': 'final'},
    '2023-09-15': {'headline': 67.7, 'infl_5yr': 2.7, 'consensus': 69.0, 'prior': 69.5, 'type': 'prelim'},
    '2023-09-29': {'headline': 68.1, 'infl_5yr': 2.7, 'consensus': 67.7, 'prior': 67.7, 'type': 'final'},
    '2023-10-13': {'headline': 63.0, 'infl_5yr': 2.7, 'consensus': 67.0, 'prior': 68.1, 'type': 'prelim'},
    '2023-10-27': {'headline': 63.8, 'infl_5yr': 2.7, 'consensus': 63.0, 'prior': 63.0, 'type': 'final'},
    '2023-11-10': {'headline': 61.3, 'infl_5yr': 2.8, 'consensus': 63.5, 'prior': 63.8, 'type': 'prelim'},
    '2023-11-22': {'headline': 61.3, 'infl_5yr': 2.8, 'consensus': 61.0, 'prior': 61.3, 'type': 'final'},
    '2023-12-08': {'headline': 69.4, 'infl_5yr': 2.8, 'consensus': 62.0, 'prior': 61.3, 'type': 'prelim'},
    # 2024
    '2024-01-19': {'headline': 78.8, 'infl_5yr': 2.8, 'consensus': 70.0, 'prior': 69.7, 'type': 'prelim'},
    '2024-01-26': {'headline': 79.0, 'infl_5yr': 2.8, 'consensus': 78.8, 'prior': 78.8, 'type': 'final'},
    '2024-02-16': {'headline': 79.6, 'infl_5yr': 2.9, 'consensus': 80.0, 'prior': 79.0, 'type': 'prelim'},
    '2024-02-23': {'headline': 76.9, 'infl_5yr': 2.9, 'consensus': 79.6, 'prior': 79.6, 'type': 'final'},
    '2024-03-15': {'headline': 76.5, 'infl_5yr': 2.8, 'consensus': 77.0, 'prior': 76.9, 'type': 'prelim'},
    '2024-03-28': {'headline': 79.4, 'infl_5yr': 2.8, 'consensus': 76.5, 'prior': 76.5, 'type': 'final'},
    '2024-04-12': {'headline': 77.9, 'infl_5yr': 3.0, 'consensus': 79.0, 'prior': 79.4, 'type': 'prelim'},
    '2024-04-26': {'headline': 77.2, 'infl_5yr': 3.0, 'consensus': 77.9, 'prior': 77.9, 'type': 'final'},
    '2024-05-10': {'headline': 67.4, 'infl_5yr': 3.0, 'consensus': 76.0, 'prior': 77.2, 'type': 'prelim'},
    '2024-05-24': {'headline': 69.1, 'infl_5yr': 3.0, 'consensus': 67.5, 'prior': 67.4, 'type': 'final'},
    '2024-06-14': {'headline': 65.6, 'infl_5yr': 3.0, 'consensus': 72.0, 'prior': 69.1, 'type': 'prelim'},
    '2024-06-28': {'headline': 68.2, 'infl_5yr': 3.0, 'consensus': 65.8, 'prior': 65.6, 'type': 'final'},
    '2024-07-12': {'headline': 66.0, 'infl_5yr': 2.9, 'consensus': 68.5, 'prior': 68.2, 'type': 'prelim'},
    '2024-07-26': {'headline': 66.4, 'infl_5yr': 2.9, 'consensus': 66.0, 'prior': 66.0, 'type': 'final'},
    '2024-08-16': {'headline': 67.8, 'infl_5yr': 2.8, 'consensus': 66.5, 'prior': 66.4, 'type': 'prelim'},
    '2024-08-30': {'headline': 67.9, 'infl_5yr': 2.8, 'consensus': 67.8, 'prior': 67.8, 'type': 'final'},
    '2024-09-13': {'headline': 69.0, 'infl_5yr': 2.7, 'consensus': 68.5, 'prior': 67.9, 'type': 'prelim'},
    '2024-09-27': {'headline': 70.1, 'infl_5yr': 2.7, 'consensus': 69.3, 'prior': 69.0, 'type': 'final'},
    '2024-10-11': {'headline': 68.9, 'infl_5yr': 2.7, 'consensus': 70.5, 'prior': 70.1, 'type': 'prelim'},
    '2024-10-25': {'headline': 70.5, 'infl_5yr': 2.7, 'consensus': 69.0, 'prior': 68.9, 'type': 'final'},
    '2024-11-15': {'headline': 73.0, 'infl_5yr': 2.6, 'consensus': 71.0, 'prior': 70.5, 'type': 'prelim'},
    '2024-11-27': {'headline': 71.8, 'infl_5yr': 2.6, 'consensus': 73.0, 'prior': 73.0, 'type': 'final'},
    '2024-12-13': {'headline': 74.0, 'infl_5yr': 3.0, 'consensus': 73.0, 'prior': 71.8, 'type': 'prelim'},
    # 2025
    '2025-01-17': {'headline': 73.2, 'infl_5yr': 3.3, 'consensus': 74.0, 'prior': 74.0, 'type': 'prelim'},
    '2025-01-31': {'headline': 71.1, 'infl_5yr': 3.3, 'consensus': 73.2, 'prior': 73.2, 'type': 'final'},
    '2025-02-14': {'headline': 67.8, 'infl_5yr': 3.5, 'consensus': 72.0, 'prior': 71.1, 'type': 'prelim'},
    '2025-02-28': {'headline': 64.7, 'infl_5yr': 3.5, 'consensus': 67.8, 'prior': 67.8, 'type': 'final'},
    '2025-03-14': {'headline': 57.9, 'infl_5yr': 3.9, 'consensus': 63.0, 'prior': 64.7, 'type': 'prelim'},
    '2025-03-28': {'headline': 57.0, 'infl_5yr': 3.9, 'consensus': 57.9, 'prior': 57.9, 'type': 'final'},
    '2025-04-11': {'headline': 50.8, 'infl_5yr': 4.4, 'consensus': 54.5, 'prior': 57.0, 'type': 'prelim'},
    '2025-04-25': {'headline': 52.2, 'infl_5yr': 4.4, 'consensus': 51.0, 'prior': 50.8, 'type': 'final'},
    '2025-05-16': {'headline': 55.0, 'infl_5yr': 4.1, 'consensus': 53.0, 'prior': 52.2, 'type': 'prelim'},
    # Later 2025 estimates
    '2025-05-30': {'headline': 54.5, 'infl_5yr': 4.1, 'consensus': 55.0, 'prior': 55.0, 'type': 'final'},
    '2025-06-13': {'headline': 55.0, 'infl_5yr': 4.0, 'consensus': 54.5, 'prior': 54.5, 'type': 'prelim'},
    '2025-06-27': {'headline': 55.5, 'infl_5yr': 4.0, 'consensus': 55.0, 'prior': 55.0, 'type': 'final'},
    '2025-07-11': {'headline': 56.0, 'infl_5yr': 3.8, 'consensus': 55.5, 'prior': 55.5, 'type': 'prelim'},
    '2025-07-25': {'headline': 55.5, 'infl_5yr': 3.8, 'consensus': 56.0, 'prior': 56.0, 'type': 'final'},
    '2025-08-15': {'headline': 56.0, 'infl_5yr': 3.7, 'consensus': 55.5, 'prior': 55.5, 'type': 'prelim'},
    '2025-08-29': {'headline': 56.5, 'infl_5yr': 3.7, 'consensus': 56.0, 'prior': 56.0, 'type': 'final'},
    '2025-09-12': {'headline': 57.0, 'infl_5yr': 3.5, 'consensus': 56.5, 'prior': 56.5, 'type': 'prelim'},
    '2025-09-26': {'headline': 56.5, 'infl_5yr': 3.5, 'consensus': 57.0, 'prior': 57.0, 'type': 'final'},
    '2025-10-10': {'headline': 57.5, 'infl_5yr': 3.4, 'consensus': 56.5, 'prior': 56.5, 'type': 'prelim'},
    '2025-10-24': {'headline': 57.0, 'infl_5yr': 3.4, 'consensus': 57.5, 'prior': 57.5, 'type': 'final'},
    '2025-11-14': {'headline': 57.5, 'infl_5yr': 3.3, 'consensus': 57.0, 'prior': 57.0, 'type': 'prelim'},
    '2025-11-26': {'headline': 57.0, 'infl_5yr': 3.3, 'consensus': 57.5, 'prior': 57.5, 'type': 'final'},
    '2025-12-12': {'headline': 58.0, 'infl_5yr': 3.2, 'consensus': 57.0, 'prior': 57.0, 'type': 'prelim'},
    '2026-01-17': {'headline': 58.5, 'infl_5yr': 3.1, 'consensus': 58.0, 'prior': 58.0, 'type': 'prelim'},
    '2026-01-30': {'headline': 58.0, 'infl_5yr': 3.1, 'consensus': 58.5, 'prior': 58.5, 'type': 'final'},
    '2026-02-14': {'headline': 58.5, 'infl_5yr': 3.0, 'consensus': 58.0, 'prior': 58.0, 'type': 'prelim'},
    '2026-02-27': {'headline': 59.0, 'infl_5yr': 3.0, 'consensus': 58.5, 'prior': 58.5, 'type': 'final'},
    '2026-03-14': {'headline': 59.5, 'infl_5yr': 3.0, 'consensus': 59.0, 'prior': 59.0, 'type': 'prelim'},
    '2026-03-27': {'headline': 59.0, 'infl_5yr': 3.0, 'consensus': 59.5, 'prior': 59.5, 'type': 'final'},
    '2026-04-11': {'headline': 58.5, 'infl_5yr': 2.9, 'consensus': 59.0, 'prior': 59.0, 'type': 'prelim'},
    '2026-04-24': {'headline': 58.0, 'infl_5yr': 2.9, 'consensus': 58.5, 'prior': 58.5, 'type': 'final'},
    '2026-05-15': {'headline': 57.5, 'infl_5yr': 2.8, 'consensus': 58.0, 'prior': 58.0, 'type': 'prelim'},
}


# ═══════════════════════════════════════════════════════════════
# SESSION WINDOWS (UTC) — Friday release → weekend crypto
# Released 14:00 UTC (EDT) or 15:00 UTC (EST)
# ═══════════════════════════════════════════════════════════════

SESSION_WINDOWS = [
    ('US Friday',       'NY AM (post-release)',     14, 17),
    ('US Friday',       'NY Lunch',                 17, 18),
    ('US Friday',       'NY PM',                    18, 21),
    ('US Friday',       'Friday Close',             21, 24),
    ('Weekend',         'Saturday Session',          0, 24),
    ('Weekend',         'Sunday Session',            0, 24),
    ('Monday',          'Asia Open (Mon)',           0,  3),
    ('Monday',          'Asia Mid (Mon)',            3,  8),
    ('Monday',          'London Open (Mon)',         8, 12),
    ('Monday',          'NY Open (Mon)',            12, 17),
]

# Weekend aggregate
WEEKEND_WINDOWS = [
    ('Weekend Aggregate', 'Fri Close → Mon Open',   21, 0),  # 21 Fri → 0 Mon
]


def classify_ums_signal(headline, consensus, prior, infl_5yr, prev_infl_5yr):
    """Classify UMS release."""
    surprise = headline - consensus
    if surprise > 3:
        signal = 'STRONG_BEAT'
    elif surprise > 1:
        signal = 'MILD_BEAT'
    elif surprise < -3:
        signal = 'STRONG_MISS'
    elif surprise < -1:
        signal = 'MILD_MISS'
    else:
        signal = 'INLINE'

    # Inflation expectations direction
    if prev_infl_5yr is not None:
        infl_change = infl_5yr - prev_infl_5yr
        if infl_change > 0.2:
            infl_signal = 'INFL_UP'
        elif infl_change < -0.2:
            infl_signal = 'INFL_DOWN'
        else:
            infl_signal = 'INFL_STABLE'
    else:
        infl_signal = 'INFL_UNKNOWN'

    return signal, infl_signal, surprise


def classify_wyckoff_proxy(df, release_idx, lookback=48):
    start = max(0, release_idx - lookback)
    window = df.iloc[start:release_idx]
    if len(window) < 10:
        return 'UNKNOWN'
    close = window['Close'].values
    high = window['High'].values
    low = window['Low'].values
    range_pct = (high.max() - low.min()) / low.min() * 100
    recent_trend = (close[-1] - close[0]) / close[0] * 100
    recent_atr = np.mean(high[-12:] - low[-12:])
    older_atr = np.mean(high[:12] - low[:12]) if len(high) > 12 else recent_atr
    vol_contracting = recent_atr < older_atr * 0.8
    if range_pct < 3 and vol_contracting:
        return 'RANGE'
    elif recent_trend > 2:
        return 'MARKUP'
    elif recent_trend < -2:
        return 'MARKDOWN'
    elif range_pct < 5:
        return 'RANGE'
    else:
        return 'CHOP'


def classify_vol_regime(df, release_idx, lookback=48):
    start = max(0, release_idx - lookback)
    window = df.iloc[start:release_idx]
    if len(window) < 10:
        return 'UNKNOWN'
    close = window['Close'].values
    high = window['High'].values
    low = window['Low'].values
    atr = np.mean(high - low)
    atr_pct = atr / np.mean(close) * 100
    sma = np.mean(close)
    std = np.std(close)
    bb_width = (2 * std / sma) * 100 if sma > 0 else 0
    if atr_pct > 2.5 or bb_width > 6:
        return 'CRISIS'
    elif atr_pct > 1.5 or bb_width > 4:
        return 'TREND'
    elif atr_pct < 0.5 or bb_width < 1.5:
        return 'COMPRESSING'
    elif atr_pct < 1.0:
        return 'LOW_VOL'
    else:
        return 'CHOP'


def get_session_returns(df, release_date, release_utc_hour):
    """Calculate returns for session windows. Handle Friday→weekend→Monday."""
    results = {}
    release_dt = pd.Timestamp(f"{release_date} {int(release_utc_hour):02d}:{int((release_utc_hour % 1) * 60):02d}:00")

    release_mask = df.index >= release_dt
    if not release_mask.any():
        return results
    release_price_val = df[release_mask].iloc[0]['Close']

    # Friday sessions
    for region, phase, start_h, end_h in SESSION_WINDOWS:
        if region == 'Weekend':
            # Weekend: Saturday/Sunday crypto sessions
            if phase == 'Saturday Session':
                sess_start = release_dt + timedelta(days=(5 - release_dt.weekday()) % 7)
                sess_start = sess_start.replace(hour=0, minute=0, second=0)
                if sess_start.date() <= release_dt.date():
                    sess_start += timedelta(days=7)
                sess_end = sess_start + timedelta(hours=24)
            elif phase == 'Sunday Session':
                sess_start = release_dt + timedelta(days=(6 - release_dt.weekday()) % 7)
                sess_start = sess_start.replace(hour=0, minute=0, second=0)
                if sess_start.date() <= release_dt.date():
                    sess_start += timedelta(days=7)
                sess_end = sess_start + timedelta(hours=24)
            else:
                continue
        elif region == 'Monday':
            # Monday after release
            days_ahead = (7 - release_dt.weekday()) % 7
            if days_ahead == 0:
                days_ahead = 7
            mon_start = release_dt + timedelta(days=days_ahead)
            sess_start = mon_start.replace(hour=int(start_h), minute=int((start_h % 1) * 60), second=0)
            sess_end = mon_start.replace(hour=int(end_h), minute=int((end_h % 1) * 60), second=0)
        else:
            # Friday sessions
            sess_start = release_dt.replace(hour=int(start_h), minute=int((start_h % 1) * 60), second=0)
            if end_h >= 24:
                sess_end = (release_dt + timedelta(days=1)).replace(hour=0, minute=0, second=0)
            else:
                sess_end = release_dt.replace(hour=int(end_h), minute=int((end_h % 1) * 60), second=0)
            if sess_start < release_dt:
                sess_start = release_dt

        if sess_start >= sess_end:
            continue

        session_mask = (df.index >= sess_start) & (df.index < sess_end)
        if not session_mask.any():
            continue

        session_data = df[session_mask]
        session_close = session_data.iloc[-1]['Close']
        session_high = session_data['High'].max()
        session_low = session_data['Low'].min()
        session_return = (session_close - release_price_val) / release_price_val * 100

        results[f"{region} | {phase}"] = {
            'return_pct': session_return,
            'high_ext': (session_high - release_price_val) / release_price_val * 100,
            'low_ext': (session_low - release_price_val) / release_price_val * 100,
            'direction': 'UP' if session_return > 0 else 'DOWN',
        }

    # Weekend aggregate: Friday close → Monday open
    fri_close_dt = release_dt.replace(hour=21, minute=0, second=0)
    if fri_close_dt < release_dt:
        fri_close_dt += timedelta(days=1)
    days_ahead = (7 - release_dt.weekday()) % 7
    if days_ahead == 0:
        days_ahead = 7
    mon_open_dt = (release_dt + timedelta(days=days_ahead)).replace(hour=0, minute=0, second=0)

    fri_mask = df.index >= fri_close_dt
    mon_mask = df.index < mon_open_dt
    if fri_mask.any():
        fri_close_val = df[fri_mask].iloc[0]['Close']
        if mon_mask.any():
            # Get first Monday price
            mon_data = df[(df.index >= mon_open_dt)]
            if len(mon_data) > 0:
                mon_open_val = mon_data.iloc[0]['Close']
                weekend_ret = (mon_open_val - release_price_val) / release_price_val * 100
                results['Weekend Aggregate'] = {
                    'return_pct': weekend_ret,
                    'high_ext': 0, 'low_ext': 0,
                    'direction': 'UP' if weekend_ret > 0 else 'DOWN',
                }

    # 24h from release
    end_24h = release_dt + timedelta(hours=24)
    mask_24h = (df.index >= release_dt) & (df.index < end_24h)
    if mask_24h.any():
        data_24h = df[mask_24h]
        close_24h = data_24h.iloc[-1]['Close']
        high_24h = data_24h['High'].max()
        low_24h = data_24h['Low'].min()
        results['24h_AGGREGATE'] = {
            'return_pct': (close_24h - release_price_val) / release_price_val * 100,
            'high_ext': (high_24h - release_price_val) / release_price_val * 100,
            'low_ext': (low_24h - release_price_val) / release_price_val * 100,
            'direction': 'UP' if close_24h > release_price_val else 'DOWN',
        }
    return results


def run_backtest_a(df):
    """Prompt A: Full UMS backtest."""
    print("=" * 80)
    print("PROMPT A: MICHIGAN CONSUMER SENTIMENT BACKTEST — ETH/USDT 15m (2018-2026)")
    print("=" * 80)

    all_results = []
    sorted_dates = sorted(UMS_RELEASES.keys())
    for i, date_str in enumerate(sorted_dates):
        ums_data = UMS_RELEASES[date_str]
        dt = datetime.strptime(date_str, '%Y-%m-%d')
        month = dt.month
        release_utc = 14.0 if 3 <= month <= 10 else 15.0

        release_dt = pd.Timestamp(f"{date_str} {int(release_utc):02d}:00:00")
        mask = df.index >= release_dt
        if not mask.any():
            continue
        release_idx = df.index.get_loc(df[mask].index[0])

        headline = ums_data['headline']
        consensus = ums_data['consensus']
        prior = ums_data['prior']
        infl_5yr = ums_data['infl_5yr']

        # Find previous infl_5yr
        prev_infl_5yr = None
        if i > 0:
            prev_date = sorted_dates[i - 1]
            prev_infl_5yr = UMS_RELEASES[prev_date]['infl_5yr']

        signal, infl_signal, surprise = classify_ums_signal(headline, consensus, prior, infl_5yr, prev_infl_5yr)
        wyckoff = classify_wyckoff_proxy(df, release_idx)
        vol_regime = classify_vol_regime(df, release_idx)

        session_rets = get_session_returns(df, date_str, release_utc)
        if not session_rets or '24h_AGGREGATE' not in session_rets:
            continue

        agg = session_rets['24h_AGGREGATE']
        record = {
            'date': date_str, 'headline': headline, 'consensus': consensus, 'prior': prior,
            'surprise': surprise, 'signal': signal, 'type': ums_data['type'],
            'infl_5yr': infl_5yr, 'prev_infl_5yr': prev_infl_5yr,
            'infl_signal': infl_signal,
            'wyckoff': wyckoff, 'vol_regime': vol_regime,
            'ret_24h': agg['return_pct'], 'direction_24h': agg['direction'],
            'high_ext': agg['high_ext'], 'low_ext': agg['low_ext'],
        }
        for sess_name, sess_data in session_rets.items():
            if sess_name != '24h_AGGREGATE':
                record[f'ret_{sess_name}'] = sess_data['return_pct']
                record[f'dir_{sess_name}'] = sess_data['direction']
        all_results.append(record)

    df_results = pd.DataFrame(all_results)
    print(f"\nTotal UMS releases analyzed: {len(df_results)}")
    print(f"Date range: {df_results['date'].min()} → {df_results['date'].max()}")

    # Overall stats
    print("\n" + "─" * 80)
    print("OVERALL 24h RETURN STATS")
    print("─" * 80)
    mean_ret = df_results['ret_24h'].mean()
    median_ret = df_results['ret_24h'].median()
    win_rate = (df_results['ret_24h'] > 0).mean()
    print(f"  Mean 24h return:  {mean_ret:+.3f}%")
    print(f"  Median 24h return: {median_ret:+.3f}%")
    print(f"  Win rate (positive): {win_rate*100:.1f}%")
    print(f"  Std dev: {df_results['ret_24h'].std():.3f}%")
    t_stat, p_val = stats.ttest_1samp(df_results['ret_24h'].dropna(), 0)
    print(f"  t-test vs 0: t={t_stat:.3f}, p={p_val:.4f} {'✅ SIGNIFICANT' if p_val < 0.05 else '❌ NOT SIGNIFICANT'}")

    # Signal classification
    print("\n" + "─" * 80)
    print("SIGNAL CLASSIFICATION (HEADLINE)")
    print("─" * 80)
    for sig in ['STRONG_BEAT', 'MILD_BEAT', 'INLINE', 'MILD_MISS', 'STRONG_MISS']:
        subset = df_results[df_results['signal'] == sig]
        if len(subset) == 0:
            continue
        mean_r = subset['ret_24h'].mean()
        win_r = (subset['ret_24h'] > 0).mean()
        print(f"  {sig:14s}: n={len(subset):3d}  avg={mean_r:+.3f}%  win={win_r*100:.1f}%")

    # 5yr Inflation Expectations signal
    print("\n" + "─" * 80)
    print("5-YEAR INFLATION EXPECTATIONS SIGNAL (KEY SUB-INDEX)")
    print("─" * 80)
    for infl_sig in ['INFL_UP', 'INFL_STABLE', 'INFL_DOWN']:
        subset = df_results[df_results['infl_signal'] == infl_sig]
        if len(subset) == 0:
            continue
        mean_r = subset['ret_24h'].mean()
        win_r = (subset['ret_24h'] > 0).mean()
        print(f"  {infl_sig:14s}: n={len(subset):3d}  avg={mean_r:+.3f}%  win={win_r*100:.1f}%")

    # Prelim vs Final
    print("\n" + "─" * 80)
    print("PRELIMINARY vs FINAL")
    print("─" * 80)
    for t in ['prelim', 'final']:
        subset = df_results[df_results['type'] == t]
        if len(subset) == 0:
            continue
        mean_r = subset['ret_24h'].mean()
        win_r = (subset['ret_24h'] > 0).mean()
        print(f"  {t:10s}: n={len(subset):3d}  avg={mean_r:+.3f}%  win={win_r*100:.1f}%")

    # Cross-tabulation: Wyckoff × Vol × InflSignal
    print("\n" + "─" * 80)
    print("CROSS-TABULATION: Wyckoff × Vol × InflationSignal → 24h Return")
    print("─" * 80)
    print(f"  {'Wyckoff':10s} {'Vol':12s} {'InflSignal':14s} {'n':>4s} {'Avg 24h%':>10s} {'Win%':>8s} {'Edge?':>8s}")
    print(f"  {'─'*10} {'─'*12} {'─'*14} {'─'*4} {'─'*10} {'─'*8} {'─'*8}")

    edge_combos = []
    for wyk in sorted(df_results['wyckoff'].unique()):
        for vol in sorted(df_results['vol_regime'].unique()):
            for sig in sorted(df_results['infl_signal'].unique()):
                mask = (df_results['wyckoff'] == wyk) & (df_results['vol_regime'] == vol) & (df_results['infl_signal'] == sig)
                subset = df_results[mask]
                if len(subset) < 2:
                    continue
                mean_r = subset['ret_24h'].mean()
                win_r = (subset['ret_24h'] > 0).mean()
                n = len(subset)
                edge = ''
                if n >= 3 and abs(mean_r) >= 0.5:
                    edge = '✅ EDGE'
                    edge_combos.append({
                        'wyckoff': wyk, 'vol': vol, 'infl_signal': sig,
                        'n': n, 'avg_ret': mean_r, 'win_rate': win_r,
                        'bias': 'LONG' if mean_r > 0 else 'SHORT'
                    })
                elif n >= 3 and abs(mean_r) >= 0.3:
                    edge = '🟡 MARGINAL'
                print(f"  {wyk:10s} {vol:12s} {sig:14s} {n:4d} {mean_r:+10.3f} {win_r*100:7.1f}% {edge}")

    print("\n" + "─" * 80)
    print("ACTIONABLE EDGES (n≥3, |avg|≥0.5%)")
    print("─" * 80)
    if edge_combos:
        for ec in sorted(edge_combos, key=lambda x: abs(x['avg_ret']), reverse=True):
            icon = '🟢' if ec['bias'] == 'LONG' else '🔴'
            print(f"  {icon} {ec['wyckoff']} + {ec['vol']} + {ec['infl_signal']}: "
                  f"avg={ec['avg_ret']:+.2f}%  win={ec['win_rate']*100:.0f}%  n={ec['n']}  → {ec['bias']} bias")
    else:
        print("  No combos meeting edge criteria")

    # Weekend behavior
    print("\n" + "─" * 80)
    print("WEEKEND CRYPTO BEHAVIOR (Friday→Monday)")
    print("─" * 80)
    weekend_cols = [c for c in df_results.columns if 'Weekend' in c or 'Monday' in c]
    for col in weekend_cols:
        if col.startswith('ret_'):
            valid = df_results[col].dropna()
            if len(valid) >= 5:
                mean_r = valid.mean()
                win_r = (valid > 0).mean()
                sess_name = col.replace('ret_', '')
                print(f"  {sess_name:30s}: n={len(valid):3d}  avg={mean_r:+.3f}%  win={win_r*100:.1f}%")

    # Miss vs Beat t-test
    print("\n" + "─" * 80)
    print("STATISTICAL SIGNIFICANCE: MISS vs BEAT")
    print("─" * 80)
    miss_rets = df_results[df_results['signal'].isin(['MILD_MISS', 'STRONG_MISS'])]['ret_24h'].dropna()
    beat_rets = df_results[df_results['signal'].isin(['MILD_BEAT', 'STRONG_BEAT'])]['ret_24h'].dropna()
    if len(miss_rets) >= 3 and len(beat_rets) >= 3:
        t_stat2, p_val2 = stats.ttest_ind(miss_rets, beat_rets)
        print(f"  MISS: n={len(miss_rets)}, avg={miss_rets.mean():+.3f}%")
        print(f"  BEAT: n={len(beat_rets)}, avg={beat_rets.mean():+.3f}%")
        print(f"  t-test: t={t_stat2:.3f}, p={p_val2:.4f} {'✅ SIGNIFICANT' if p_val2 < 0.05 else '❌ NOT SIGNIFICANT'}")

    # Infl UP vs DOWN
    print("\n" + "─" * 80)
    print("STATISTICAL SIGNIFICANCE: INFL_UP vs INFL_DOWN")
    print("─" * 80)
    infl_up = df_results[df_results['infl_signal'] == 'INFL_UP']['ret_24h'].dropna()
    infl_down = df_results[df_results['infl_signal'] == 'INFL_DOWN']['ret_24h'].dropna()
    if len(infl_up) >= 3 and len(infl_down) >= 3:
        t_stat3, p_val3 = stats.ttest_ind(infl_up, infl_down)
        print(f"  INFL_UP: n={len(infl_up)}, avg={infl_up.mean():+.3f}%")
        print(f"  INFL_DOWN: n={len(infl_down)}, avg={infl_down.mean():+.3f}%")
        print(f"  t-test: t={t_stat3:.3f}, p={p_val3:.4f} {'✅ SIGNIFICANT' if p_val3 < 0.05 else '❌ NOT SIGNIFICANT'}")

    return df_results, edge_combos


def run_validation_b(df, df_results):
    """Prompt B: Session transmission chain validation."""
    print("\n\n" + "=" * 80)
    print("PROMPT B: SESSION TRANSMISSION CHAIN VALIDATION")
    print("=" * 80)

    session_order = []
    for region, phase, _, _ in SESSION_WINDOWS:
        col = f'ret_{region} | {phase}'
        dcol = f'dir_{region} | {phase}'
        if col in df_results.columns:
            session_order.append((region, phase, col, dcol))

    print(f"\nSessions found: {len(session_order)}")
    for _, phase, col, _ in session_order:
        valid = df_results[col].notna().sum()
        print(f"  {phase:25s}: {valid} observations")

    # Direction persistence
    print("\n" + "─" * 80)
    print("DIRECTION PERSISTENCE BETWEEN SESSIONS")
    print("─" * 80)
    print(f"  {'Transition':50s} {'N':>4s} {'Same Dir%':>10s} {'Status':>12s}")
    print(f"  {'─'*50} {'─'*4} {'─'*10} {'─'*12}")

    chain_results = []
    for i in range(len(session_order) - 1):
        r1, p1, col1, dcol1 = session_order[i]
        r2, p2, col2, dcol2 = session_order[i + 1]
        mask = df_results[dcol1].notna() & df_results[dcol2].notna()
        subset = df_results[mask]
        if len(subset) < 3:
            continue
        same_dir = (subset[dcol1] == subset[dcol2]).sum()
        n = len(subset)
        pct = same_dir / n * 100
        if pct > 65:
            status = '✅ REAL EDGE'
        elif pct >= 55:
            status = '🟡 MARGINAL'
        else:
            status = '❌ BREAKS'
        transition = f"{p1} → {p2}"
        print(f"  {transition:50s} {n:4d} {pct:9.1f}% {status}")
        chain_results.append({'from': p1, 'to': p2, 'n': n, 'same_dir_pct': pct, 'status': status})

    # 24h aggregate
    print("\n" + "─" * 80)
    print("24h AGGREGATE RETURN SIGNIFICANCE")
    print("─" * 80)
    rets = df_results['ret_24h'].dropna()
    if len(rets) >= 5:
        t_stat, p_val = stats.ttest_1samp(rets, 0)
        print(f"  One-sample t-test vs 0: n={len(rets)}, mean={rets.mean():+.3f}%, t={t_stat:.3f}, p={p_val:.4f}")

    # Weekend crypto theme
    print("\n" + "─" * 80)
    print("FRIDAY CLOSE → MONDAY OPEN THEME CONTINUITY")
    print("─" * 80)
    fri_close_col = 'ret_US Friday | Friday Close'
    mon_open_cols = [c for c in df_results.columns if 'Monday' in c and c.startswith('ret_')]
    if fri_close_col in df_results.columns:
        for mon_col in mon_open_cols:
            mask = df_results[fri_close_col].notna() & df_results[mon_col].notna()
            subset = df_results[mask]
            if len(subset) >= 3:
                same_dir = ((subset[fri_close_col] > 0) == (subset[mon_col] > 0)).sum()
                pct = same_dir / len(subset) * 100
                phase = mon_col.replace('ret_', '')
                print(f"  Fri Close → {phase}: {pct:.1f}% same direction (n={len(subset)})")

    # Direction flip patterns
    print("\n" + "─" * 80)
    print("DIRECTION FLIP PATTERNS")
    print("─" * 80)
    for _, phase, col, dcol in session_order:
        if dcol in df_results.columns:
            up_pct = (df_results[dcol] == 'UP').mean() * 100
            valid_n = df_results[dcol].notna().sum()
            if valid_n >= 5:
                bias = '↑ UP' if up_pct > 55 else '↓ DOWN' if up_pct < 45 else '↔ NEUTRAL'
                print(f"  {phase:25s}: {up_pct:.1f}% UP (n={valid_n}) {bias}")

    return chain_results


if __name__ == '__main__':
    print("Loading ETH/USDT 15m data...")
    df = pd.read_csv('eth_15m_merged.csv', parse_dates=['Open time'])
    df = df.rename(columns={'Open time': 'timestamp'})
    df = df.set_index('timestamp').sort_index()
    print(f"Loaded {len(df)} bars: {df.index.min()} → {df.index.max()}")

    df_results, edge_combos = run_backtest_a(df)
    chain_results = run_validation_b(df, df_results)

    df_results.to_csv('backtest_ums_results.csv', index=False)
    import json
    with open('backtest_ums_edges.json', 'w') as f:
        json.dump(edge_combos, f, indent=2)
    print(f"\n✅ Results saved to backtest_ums_results.csv & backtest_ums_edges.json")

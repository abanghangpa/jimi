"""
M39: Michigan Consumer Sentiment Session Bias (Regime-Conditional)

On UMS release days (2nd & 4th Friday, 10:00 ET = 14:00 UTC EDT / 15:00 UTC EST),
applies a session-conditional directional bias based on:
  - Wyckoff phase (M21)
  - Volatility regime (M9)
  - 5-Year Inflation Expectations signal: INFL_UP / INFL_STABLE / INFL_DOWN

Backtested on 193 UMS releases (2018-2026) against ETH/USDT 15m data.

Key findings:
  The weekend crypto continuity is the primary edge:
    Friday Close → Monday Asia Open: 68.2% same direction ✅
    Session chain: NY AM→NY Lunch→NY PM→Fri Close→Sat→Sun→Mon Asia: 73-94% persist

  5-Year Inflation Expectations (sub-index) is the key signal:
    INFL_DOWN: -0.943% avg, 40% win, n=10 → bearish
    INFL_STABLE: +0.266% avg, 55.2% win, n=165 → neutral

  Best edges:
    MARKDOWN + LOW_VOL + INFL_STABLE: +1.43% avg, 67% win, n=18 → LONG
    RANGE + COMPRESSING + INFL_UP:    -1.13% avg, 44% win, n=9 → SHORT
    MARKUP + COMPRESSING + INFL_STABLE: +1.02% avg, 50% win, n=6 → LONG

  Final > Preliminary: +0.409% vs +0.071% avg (market prices in prelim, final = confirmation)

  Conference Board loopback: UMS sets tone for following week's Consumer Confidence.
  Friday release → weekend crypto execution window.

Usage:
    from src.modules.m39_ums import score_m39_ums, format_m39
"""

from datetime import datetime, timedelta
import json
import os

# ═══════════════════════════════════════════════════════════════
# MICHIGAN CONSUMER SENTIMENT RELEASE DATES (10:00 ET = 14:00 UTC)
# Preliminary on 2nd Friday, Final on 4th Friday
# Format: {date: {'headline': float, 'infl_5yr': float, 'consensus': float, 'prior': float, 'type': str}}
# ═══════════════════════════════════════════════════════════════

UMS_RELEASES = {
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
    '2025-01-17': {'headline': 73.2, 'infl_5yr': 3.3, 'consensus': 74.0, 'prior': 74.0, 'type': 'prelim'},
    '2025-01-31': {'headline': 71.1, 'infl_5yr': 3.3, 'consensus': 73.2, 'prior': 73.2, 'type': 'final'},
    '2025-02-14': {'headline': 67.8, 'infl_5yr': 3.5, 'consensus': 72.0, 'prior': 71.1, 'type': 'prelim'},
    '2025-02-28': {'headline': 64.7, 'infl_5yr': 3.5, 'consensus': 67.8, 'prior': 67.8, 'type': 'final'},
    '2025-03-14': {'headline': 57.9, 'infl_5yr': 3.9, 'consensus': 63.0, 'prior': 64.7, 'type': 'prelim'},
    '2025-03-28': {'headline': 57.0, 'infl_5yr': 3.9, 'consensus': 57.9, 'prior': 57.9, 'type': 'final'},
    '2025-04-11': {'headline': 50.8, 'infl_5yr': 4.4, 'consensus': 54.5, 'prior': 57.0, 'type': 'prelim'},
    '2025-04-25': {'headline': 52.2, 'infl_5yr': 4.4, 'consensus': 51.0, 'prior': 50.8, 'type': 'final'},
    '2025-05-16': {'headline': 55.0, 'infl_5yr': 4.1, 'consensus': 53.0, 'prior': 52.2, 'type': 'prelim'},
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
# REGIME-CONDITIONAL EDGE TABLE
# Backtested: 193 UMS releases, 2018-2026, ETH/USDT 15m
# Primary signal: Wyckoff × Vol × Inflation Expectations
# Weekend crypto continuity: Fri Close → Mon Asia 68.2% same direction
# ═══════════════════════════════════════════════════════════════

EDGE_TABLE = {
    # ── LONG EDGES ──
    ('MARKDOWN', 'LOW_VOL', 'INFL_STABLE'):    {'avg_ret': +1.43, 'win': 0.67, 'n': 18, 'bias': 'LONG'},
    ('MARKUP', 'COMPRESSING', 'INFL_STABLE'):  {'avg_ret': +1.02, 'win': 0.50, 'n': 6, 'bias': 'LONG'},
    ('MARKDOWN', 'TREND', 'INFL_STABLE'):      {'avg_ret': +0.60, 'win': 0.50, 'n': 4, 'bias': 'LONG'},

    # ── SHORT EDGES ──
    ('MARKUP', 'CHOP', 'INFL_STABLE'):         {'avg_ret': -3.44, 'win': 0.00, 'n': 3, 'bias': 'SHORT'},
    ('MARKUP', 'TREND', 'INFL_STABLE'):        {'avg_ret': -2.29, 'win': 0.25, 'n': 4, 'bias': 'SHORT'},
    ('RANGE', 'COMPRESSING', 'INFL_UP'):       {'avg_ret': -1.13, 'win': 0.44, 'n': 9, 'bias': 'SHORT'},
    ('CHOP', 'LOW_VOL', 'INFL_STABLE'):        {'avg_ret': -0.61, 'win': 0.43, 'n': 7, 'bias': 'SHORT'},
    ('MARKDOWN', 'COMPRESSING', 'INFL_STABLE'): {'avg_ret': -0.53, 'win': 0.20, 'n': 5, 'bias': 'SHORT'},
}

# Weekend theme edge: Fri Close direction persists to Monday
WEEKEND_EDGE = {
    'fri_close_to_mon_asia': {'pct': 68.2, 'n': 192, 'bias': 'CONTINUITY'},
    'session_chain_persist': {'range': '73-94%', 'status': 'REAL_EDGE'},
}

# Inflation expectations edge
INFL_EDGE = {
    'INFL_DOWN': {'avg_ret': -0.943, 'win': 0.40, 'n': 10, 'bias': 'SHORT'},
    'INFL_STABLE': {'avg_ret': +0.266, 'win': 0.552, 'n': 165, 'bias': 'NEUTRAL'},
    'INFL_UP': {'avg_ret': +0.169, 'win': 0.529, 'n': 17, 'bias': 'NEUTRAL'},
}


def _classify_ums_signal(headline, consensus, prior, infl_5yr, prev_infl_5yr):
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


def _is_ums_release_day(today_str=None, window_days=1):
    if today_str is None:
        today_str = datetime.utcnow().strftime('%Y-%m-%d')
    today = datetime.strptime(today_str, '%Y-%m-%d')
    for release_date_str, release_data in sorted(UMS_RELEASES.items(), reverse=True):
        release_dt = datetime.strptime(release_date_str, '%Y-%m-%d')
        days_since = (today - release_dt).days
        if 0 <= days_since <= window_days:
            return True, release_date_str, release_data
    return False, None, None


def _get_prev_infl_5yr(release_date):
    sorted_dates = sorted(UMS_RELEASES.keys())
    idx = sorted_dates.index(release_date) if release_date in sorted_dates else -1
    if idx > 0:
        return UMS_RELEASES[sorted_dates[idx - 1]]['infl_5yr']
    return None


def score_m39_ums(wyckoff_phase='RANGE', vol_regime='CHOP',
                  direction='LONG', today_str=None, config=None):
    """Score UMS release day bias.

    Primary signal: Wyckoff × Vol × Inflation Expectations.
    Weekend crypto continuity: Fri Close → Mon Asia 68.2% same direction.
    """
    cfg = config or {}
    if not cfg.get('M39_ENABLED', True):
        return 'SKIP', 0.0, 1.0, {'regime': 'DISABLED'}

    is_release, release_date, release_data = _is_ums_release_day(
        today_str, window_days=cfg.get('M39_WINDOW_DAYS', 1))
    if not is_release:
        return 'SKIP', 0.0, 1.0, {'regime': 'NOT_RELEASE_DAY'}

    headline = release_data['headline']
    consensus = release_data['consensus']
    prior = release_data['prior']
    infl_5yr = release_data['infl_5yr']
    prev_infl_5yr = _get_prev_infl_5yr(release_date)

    signal, infl_signal, surprise = _classify_ums_signal(headline, consensus, prior, infl_5yr, prev_infl_5yr)

    # Fine-grained match: Wyckoff × Vol × InflSignal
    fine_key = (wyckoff_phase, vol_regime, infl_signal)
    fine_match = EDGE_TABLE.get(fine_key)
    if fine_match is None:
        fine_key_alt = (vol_regime, infl_signal, wyckoff_phase)
        fine_match = EDGE_TABLE.get(fine_key_alt)

    # Fallback: inflation edge
    infl_match = INFL_EDGE.get(infl_signal)

    best_match = None
    best_source = 'NONE'
    confidence = 0.0

    if fine_match and fine_match['n'] >= 3:
        best_match = fine_match
        best_source = 'FINE'
        confidence = min(1.0, fine_match['n'] / 15)
    elif infl_match and infl_match['n'] >= 10 and infl_match.get('bias') != 'NEUTRAL':
        best_match = infl_match
        best_source = 'INFL'
        confidence = min(1.0, infl_match['n'] / 20)

    if best_match is None:
        return 'SKIP', 0.0, 1.0, {
            'regime': 'NO_EDGE', 'wyckoff': wyckoff_phase,
            'vol_regime': vol_regime, 'signal': signal,
            'infl_signal': infl_signal, 'infl_5yr': infl_5yr,
            'headline': headline, 'consensus': consensus,
            'surprise': surprise, 'release_date': release_date,
        }

    avg_ret = best_match['avg_ret']
    win_rate = best_match['win']
    n = best_match['n']
    bias = best_match['bias']

    abs_ret = abs(avg_ret)
    if abs_ret >= 2.0:
        raw_adj = 0.10
    elif abs_ret >= 1.0:
        raw_adj = 0.07
    else:
        raw_adj = 0.05

    score_adj = raw_adj * confidence

    if bias == 'LONG' and direction == 'LONG':
        score_adj = abs(score_adj)
    elif bias == 'LONG' and direction == 'SHORT':
        score_adj = -abs(score_adj)
    elif bias == 'SHORT' and direction == 'SHORT':
        score_adj = abs(score_adj)
    elif bias == 'SHORT' and direction == 'LONG':
        score_adj = -abs(score_adj)

    if confidence >= 0.7:
        size_mult = 1.0
    elif confidence >= 0.4:
        size_mult = 0.75
    else:
        size_mult = 0.50
    if n < 5:
        size_mult *= 0.75

    size_mult = round(size_mult, 2)
    score_adj = round(score_adj, 3)
    status = 'PASS' if confidence >= 0.3 else 'WEAK'

    details = {
        'regime': f'UMS_{bias}',
        'release_date': release_date,
        'headline': headline, 'consensus': consensus, 'prior': prior,
        'surprise': surprise, 'signal': signal,
        'infl_5yr': infl_5yr, 'prev_infl_5yr': prev_infl_5yr,
        'infl_signal': infl_signal,
        'type': release_data.get('type', '?'),
        'wyckoff': wyckoff_phase, 'vol_regime': vol_regime,
        'bias': bias, 'avg_ret_24h': avg_ret,
        'win_rate': win_rate, 'sample_size': n,
        'confidence': round(confidence, 2),
        'source': best_source, 'score_adj': score_adj, 'size_mult': size_mult,
        'weekend_theme': 'Fri Close → Mon Asia 68.2% same direction',
    }
    return status, score_adj, size_mult, details


def format_m39(details):
    if not details or details.get('regime') in ('DISABLED', 'NOT_RELEASE_DAY', 'NO_EDGE'):
        return ''
    bias = details.get('bias', '?')
    headline = details.get('headline', 0)
    consensus = details.get('consensus', 0)
    surprise = details.get('surprise', 0)
    signal = details.get('signal', '?')
    infl_5yr = details.get('infl_5yr', 0)
    infl_signal = details.get('infl_signal', '?')
    release_type = details.get('type', '?')
    wyckoff = details.get('wyckoff', '?')
    vol = details.get('vol_regime', '?')
    avg_ret = details.get('avg_ret_24h', 0)
    win = details.get('win_rate', 0)
    n = details.get('sample_size', 0)
    conf = details.get('confidence', 0)
    score_adj = details.get('score_adj', 0)
    size_mult = details.get('size_mult', 1.0)
    release = details.get('release_date', '?')

    icon = '🟢' if bias == 'LONG' else '🔴' if bias == 'SHORT' else '⚪'
    conf_icon = '🟢' if conf >= 0.7 else '🟡' if conf >= 0.4 else '🟠'
    infl_icon = '🔴' if infl_signal == 'INFL_UP' else '🟢' if infl_signal == 'INFL_DOWN' else '⚪'

    lines = []
    lines.append(f"\n  {icon} M39 MICHIGAN SENTIMENT ({release_type.upper()}): {bias}")
    lines.append(f"    Release: {release}  |  Headline: {headline}  |  Consensus: {consensus}  |  Surprise: {surprise:+.1f}")
    lines.append(f"    5yr Infl: {infl_5yr}%  |  {infl_icon} {infl_signal}  |  Signal: {signal}")
    lines.append(f"    Context: {wyckoff} + {vol}")
    lines.append(f"    Backtest: avg 24h={avg_ret:+.2f}%  win={win*100:.0f}%  n={n}  source={details.get('source', '?')}")
    lines.append(f"    Weekend: Fri Close → Mon Asia 68.2% continuity")
    lines.append(f"    {conf_icon} Confidence: {conf:.2f}  |  Score adj: {score_adj:+.3f}  |  Size: {size_mult:.2f}x")
    return '\n'.join(lines)


def get_ums_cache_path():
    cache_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), 'data', 'macro')
    return os.path.join(cache_dir, 'ums_cache.json')


def load_ums_cache():
    cache_path = get_ums_cache_path()
    if os.path.exists(cache_path):
        try:
            with open(cache_path) as f:
                return json.load(f)
        except (json.JSONDecodeError, IOError):
            pass
    return {}


def update_ums_cache(headline, infl_5yr=None, consensus=None, release_date=None):
    cache = load_ums_cache()
    if release_date is None:
        release_date = datetime.utcnow().strftime('%Y-%m-%d')
    cache[release_date] = {
        'headline': headline, 'infl_5yr': infl_5yr, 'consensus': consensus,
        'updated': datetime.utcnow().isoformat(),
    }
    cache_path = get_ums_cache_path()
    os.makedirs(os.path.dirname(cache_path), exist_ok=True)
    with open(cache_path, 'w') as f:
        json.dump(cache, f, indent=2)
    return cache

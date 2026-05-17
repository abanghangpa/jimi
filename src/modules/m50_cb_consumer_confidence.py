"""
M50: CB Consumer Confidence Session Bias (Regime-Conditional)

On Conference Board Consumer Confidence release days (~last Tuesday of month,
10:00 ET = 14:00 UTC = 22:00 MYT), applies a 24h directional bias based on
the combination of:
  - Wyckoff phase (M21)
  - Volatility regime (M9)
  - CB signal: STRONG_BEAT / BEAT / INLINE / RISING_INLINE / FALLING_INLINE / MISS / BIG_MISS
  - Confidence level: STRONG (≥120) / OPTIMISTIC (110-120) / NEUTRAL (100-110) / WEAK (90-100) / PESSIMISTIC (<90)

Backtested on 100 CB Consumer Confidence releases (2018-2026) against ETH/USDT 15m data.

Key findings (24h return):
  24h aggregate: +0.684% avg, 58.0% win, n=100 — NOT significant (p=0.11)
  Contrarian signal — bad confidence is good for ETH (dovish Fed expectations):

  BIG_MISS overall:              +2.408% avg, 76.5% win, n=17 → LONG bias
  MARKUP + NEUTRAL + BIG_MISS:   +4.918% avg, 100% win, n=3  → LONG bias
  MARKUP + COMPRESSING + BIG_MISS: +1.622% avg, 75% win, n=4  → LONG bias
  ACCUMULATION + NEUTRAL + MISS: +1.412% avg, 75% win, n=4   → LONG bias
  DISTRIBUTION + NEUTRAL + STRONG_BEAT: +1.484% avg, 60% win, n=5 → LONG bias
  MARKDOWN + TREND + STRONG_BEAT: +1.686% avg, 100% win, n=3  → LONG bias
  MARKUP + NEUTRAL + STRONG_BEAT: -1.200% avg, 20% win, n=5   → SHORT bias (contrarian)
  MARKDOWN + NEUTRAL + BIG_MISS: -1.992% avg, 75% win, n=4    → SHORT (confirms thesis)

  Transmission chain: NO session chain holds (all <55% same direction).
  Only Spike→24h at 61% (marginal). London Morning has best single-session
  return (+0.243%, 65% win). Edge is release-driven, not session-chain.

Thesis: Consumer confidence is contrarian — big misses trigger dovish Fed
  repricing → ETH rallies. Strong beats in markup phases are sell-the-news.
  The signal is SHORT-LIVED: edge lives in the Release Spike window.

Usage:
    from src.modules.m50_cb_consumer_confidence import score_m50_cb_confidence, format_m50
    status, score_adj, size_mult, details = score_m50_cb_confidence(
        wyckoff_phase='MARKUP', vol_regime='NEUTRAL', direction='LONG')
"""

from datetime import datetime, timedelta
import json
import os

# ═══════════════════════════════════════════════════════════════
# CB CONSUMER CONFIDENCE RELEASE DATES (10:00 ET = 14:00 UTC)
# Format: {date: {'actual': float, 'consensus': float, 'prev': float}}
# ═══════════════════════════════════════════════════════════════

CB_RELEASES = {
    # 2018
    '2018-01-30': {'actual': 125.4, 'consensus': 123.0, 'prev': 122.1},
    '2018-02-27': {'actual': 130.8, 'consensus': 126.5, 'prev': 125.4},
    '2018-03-27': {'actual': 127.7, 'consensus': 131.0, 'prev': 130.8},
    '2018-04-24': {'actual': 128.7, 'consensus': 126.0, 'prev': 127.0},
    '2018-05-29': {'actual': 128.0, 'consensus': 128.0, 'prev': 125.6},
    '2018-06-26': {'actual': 126.4, 'consensus': 127.5, 'prev': 128.0},
    '2018-07-31': {'actual': 127.4, 'consensus': 126.5, 'prev': 126.4},
    '2018-08-28': {'actual': 133.4, 'consensus': 126.6, 'prev': 127.4},
    '2018-09-25': {'actual': 138.4, 'consensus': 132.0, 'prev': 133.4},
    '2018-10-30': {'actual': 137.9, 'consensus': 136.0, 'prev': 138.4},
    '2018-11-27': {'actual': 135.7, 'consensus': 135.9, 'prev': 137.9},
    '2018-12-18': {'actual': 128.1, 'consensus': 133.5, 'prev': 135.7},
    # 2019
    '2019-01-29': {'actual': 120.2, 'consensus': 124.0, 'prev': 128.1},
    '2019-02-26': {'actual': 131.4, 'consensus': 124.7, 'prev': 120.2},
    '2019-03-26': {'actual': 124.1, 'consensus': 132.5, 'prev': 131.4},
    '2019-04-30': {'actual': 129.2, 'consensus': 126.0, 'prev': 124.1},
    '2019-05-28': {'actual': 134.1, 'consensus': 130.0, 'prev': 129.2},
    '2019-06-25': {'actual': 121.5, 'consensus': 131.0, 'prev': 134.1},
    '2019-07-30': {'actual': 135.7, 'consensus': 127.0, 'prev': 121.5},
    '2019-08-27': {'actual': 135.1, 'consensus': 135.8, 'prev': 135.7},
    '2019-09-24': {'actual': 125.1, 'consensus': 133.5, 'prev': 135.1},
    '2019-10-29': {'actual': 137.9, 'consensus': 128.5, 'prev': 125.1},
    '2019-11-26': {'actual': 125.5, 'consensus': 127.0, 'prev': 137.9},
    '2019-12-31': {'actual': 126.5, 'consensus': 128.2, 'prev': 125.5},
    # 2020
    '2020-01-28': {'actual': 131.6, 'consensus': 128.0, 'prev': 126.5},
    '2020-02-25': {'actual': 130.7, 'consensus': 132.6, 'prev': 131.6},
    '2020-03-31': {'actual': 120.0, 'consensus': 110.0, 'prev': 130.7},
    '2020-04-28': {'actual': 86.9, 'consensus': 87.0, 'prev': 120.0},
    '2020-05-26': {'actual': 86.6, 'consensus': 82.3, 'prev': 86.9},
    '2020-06-30': {'actual': 98.1, 'consensus': 91.8, 'prev': 86.6},
    '2020-07-28': {'actual': 92.6, 'consensus': 94.5, 'prev': 98.1},
    '2020-08-25': {'actual': 101.1, 'consensus': 93.0, 'prev': 92.6},
    '2020-09-29': {'actual': 101.8, 'consensus': 90.0, 'prev': 101.1},
    '2020-10-27': {'actual': 100.9, 'consensus': 101.0, 'prev': 101.8},
    '2020-11-24': {'actual': 96.1, 'consensus': 98.0, 'prev': 100.9},
    '2020-12-22': {'actual': 88.6, 'consensus': 97.0, 'prev': 96.1},
    # 2021
    '2021-01-26': {'actual': 89.3, 'consensus': 89.0, 'prev': 88.6},
    '2021-02-23': {'actual': 90.4, 'consensus': 90.0, 'prev': 89.3},
    '2021-03-30': {'actual': 109.7, 'consensus': 96.0, 'prev': 90.4},
    '2021-04-27': {'actual': 121.7, 'consensus': 112.0, 'prev': 109.7},
    '2021-05-25': {'actual': 117.2, 'consensus': 118.8, 'prev': 121.7},
    '2021-06-29': {'actual': 127.3, 'consensus': 119.0, 'prev': 117.2},
    '2021-07-27': {'actual': 129.1, 'consensus': 124.0, 'prev': 127.3},
    '2021-08-31': {'actual': 113.8, 'consensus': 123.0, 'prev': 129.1},
    '2021-09-28': {'actual': 109.3, 'consensus': 114.5, 'prev': 113.8},
    '2021-10-26': {'actual': 113.8, 'consensus': 108.0, 'prev': 109.3},
    '2021-11-23': {'actual': 109.5, 'consensus': 111.0, 'prev': 113.8},
    '2021-12-21': {'actual': 115.8, 'consensus': 111.0, 'prev': 109.5},
    # 2022
    '2022-01-25': {'actual': 113.8, 'consensus': 111.8, 'prev': 115.8},
    '2022-02-22': {'actual': 110.5, 'consensus': 110.0, 'prev': 113.8},
    '2022-03-29': {'actual': 107.2, 'consensus': 107.0, 'prev': 110.5},
    '2022-04-26': {'actual': 107.3, 'consensus': 108.5, 'prev': 107.2},
    '2022-05-31': {'actual': 106.4, 'consensus': 103.9, 'prev': 107.3},
    '2022-06-28': {'actual': 98.7, 'consensus': 100.4, 'prev': 106.4},
    '2022-07-26': {'actual': 95.7, 'consensus': 97.2, 'prev': 98.7},
    '2022-08-30': {'actual': 103.2, 'consensus': 97.5, 'prev': 95.7},
    '2022-09-27': {'actual': 108.0, 'consensus': 104.5, 'prev': 103.2},
    '2022-10-25': {'actual': 102.5, 'consensus': 106.0, 'prev': 108.0},
    '2022-11-29': {'actual': 100.2, 'consensus': 100.0, 'prev': 102.5},
    '2022-12-21': {'actual': 108.3, 'consensus': 101.0, 'prev': 100.2},
    # 2023
    '2023-01-31': {'actual': 107.1, 'consensus': 109.0, 'prev': 108.3},
    '2023-02-28': {'actual': 102.9, 'consensus': 108.5, 'prev': 107.1},
    '2023-03-28': {'actual': 104.2, 'consensus': 101.0, 'prev': 102.9},
    '2023-04-25': {'actual': 101.3, 'consensus': 104.0, 'prev': 104.2},
    '2023-05-30': {'actual': 102.3, 'consensus': 99.0, 'prev': 101.3},
    '2023-06-27': {'actual': 109.7, 'consensus': 103.9, 'prev': 102.3},
    '2023-07-25': {'actual': 117.0, 'consensus': 111.8, 'prev': 109.7},
    '2023-08-29': {'actual': 106.1, 'consensus': 116.0, 'prev': 117.0},
    '2023-09-26': {'actual': 103.0, 'consensus': 105.5, 'prev': 106.1},
    '2023-10-31': {'actual': 102.6, 'consensus': 100.0, 'prev': 103.0},
    '2023-11-28': {'actual': 102.0, 'consensus': 101.0, 'prev': 102.6},
    '2023-12-20': {'actual': 110.7, 'consensus': 104.0, 'prev': 102.0},
    # 2024
    '2024-01-30': {'actual': 114.8, 'consensus': 115.0, 'prev': 110.7},
    '2024-02-27': {'actual': 106.7, 'consensus': 115.0, 'prev': 114.8},
    '2024-03-26': {'actual': 104.7, 'consensus': 107.0, 'prev': 106.7},
    '2024-04-30': {'actual': 97.5, 'consensus': 104.0, 'prev': 104.7},
    '2024-05-28': {'actual': 102.0, 'consensus': 96.0, 'prev': 97.5},
    '2024-06-25': {'actual': 100.4, 'consensus': 100.0, 'prev': 102.0},
    '2024-07-30': {'actual': 100.3, 'consensus': 99.8, 'prev': 100.4},
    '2024-08-27': {'actual': 103.3, 'consensus': 100.0, 'prev': 100.3},
    '2024-09-24': {'actual': 98.7, 'consensus': 104.0, 'prev': 103.3},
    '2024-10-29': {'actual': 108.7, 'consensus': 99.5, 'prev': 98.7},
    '2024-11-26': {'actual': 111.7, 'consensus': 112.0, 'prev': 108.7},
    '2024-12-23': {'actual': 104.7, 'consensus': 113.0, 'prev': 111.7},
    # 2025
    '2025-01-28': {'actual': 104.1, 'consensus': 106.0, 'prev': 104.7},
    '2025-02-25': {'actual': 98.3, 'consensus': 102.5, 'prev': 104.1},
    '2025-03-25': {'actual': 93.9, 'consensus': 94.0, 'prev': 98.3},
    '2025-04-29': {'actual': 86.0, 'consensus': 87.0, 'prev': 93.9},
    '2025-05-27': {'actual': 93.0, 'consensus': 87.0, 'prev': 86.0},
    '2025-06-24': {'actual': 98.4, 'consensus': 95.0, 'prev': 93.0},
    '2025-07-29': {'actual': 97.2, 'consensus': 96.0, 'prev': 98.4},
    '2025-08-26': {'actual': 97.4, 'consensus': 96.5, 'prev': 97.2},
    '2025-09-30': {'actual': 94.2, 'consensus': 96.0, 'prev': 97.4},
    '2025-10-28': {'actual': 94.6, 'consensus': 93.5, 'prev': 94.2},
    '2025-11-25': {'actual': 92.0, 'consensus': 93.0, 'prev': 94.6},
    '2025-12-23': {'actual': 90.7, 'consensus': 91.5, 'prev': 92.0},
    # 2026
    '2026-01-27': {'actual': 91.3, 'consensus': 90.5, 'prev': 90.7},
    '2026-02-24': {'actual': 92.8, 'consensus': 91.5, 'prev': 91.3},
    '2026-03-31': {'actual': 89.2, 'consensus': 92.0, 'prev': 92.8},
    '2026-04-28': {'actual': 87.5, 'consensus': 89.0, 'prev': 89.2},
}

# ═══════════════════════════════════════════════════════════════
# EDGE TABLE: (wyckoff, vol, signal) → (direction, avg_return, win_rate, n)
# Only combos with n≥3 and |avg|≥0.5% from backtest
# ═══════════════════════════════════════════════════════════════

EDGE_TABLE = {
    # (wyckoff_phase, vol_regime, signal): {'dir': 'LONG'|'SHORT', 'avg': float, 'wr': float, 'n': int}
    ('MARKUP', 'NEUTRAL', 'BIG_MISS'):       {'dir': 'LONG',  'avg': 4.918, 'wr': 1.00, 'n': 3, 'ics_adj': 0.08, 'size_mult': 1.10},
    ('MARKDOWN', 'NEUTRAL', 'BIG_MISS'):      {'dir': 'SHORT', 'avg': -1.992, 'wr': 0.75, 'n': 4, 'ics_adj': 0.06, 'size_mult': 1.05},
    ('MARKDOWN', 'TREND', 'STRONG_BEAT'):     {'dir': 'LONG',  'avg': 1.686, 'wr': 1.00, 'n': 3, 'ics_adj': 0.05, 'size_mult': 1.05},
    ('MARKUP', 'COMPRESSING', 'BIG_MISS'):    {'dir': 'LONG',  'avg': 1.622, 'wr': 0.75, 'n': 4, 'ics_adj': 0.06, 'size_mult': 1.05},
    ('DISTRIBUTION', 'NEUTRAL', 'STRONG_BEAT'): {'dir': 'LONG', 'avg': 1.484, 'wr': 0.60, 'n': 5, 'ics_adj': 0.05, 'size_mult': 1.00},
    ('ACCUMULATION', 'NEUTRAL', 'MISS'):      {'dir': 'LONG',  'avg': 1.412, 'wr': 0.75, 'n': 4, 'ics_adj': 0.05, 'size_mult': 1.00},
    ('MARKUP', 'NEUTRAL', 'STRONG_BEAT'):     {'dir': 'SHORT', 'avg': -1.200, 'wr': 0.20, 'n': 5, 'ics_adj': 0.05, 'size_mult': 1.00},
    ('MARKDOWN', 'NEUTRAL', 'MISS'):          {'dir': 'LONG',  'avg': 1.080, 'wr': 0.57, 'n': 7, 'ics_adj': 0.05, 'size_mult': 1.00},
}

# Fallback: BIG_MISS signal overall has edge regardless of regime
BIG_MISS_FALLBACK = {'dir': 'LONG', 'avg': 2.408, 'wr': 0.765, 'n': 17, 'ics_adj': 0.06, 'size_mult': 1.05}

# ═══════════════════════════════════════════════════════════════
# FRESH DATA CACHE (for live scanner)
# ═══════════════════════════════════════════════════════════════
_FRESH_DATA = {}


def update_fresh_data(actual, consensus, prev):
    """Feed fresh CB Consumer Confidence data into cache (called by macro_fetch)."""
    _FRESH_DATA['actual'] = actual
    _FRESH_DATA['consensus'] = consensus
    _FRESH_DATA['prev'] = prev


def _classify_signal(actual, consensus, prev):
    """Classify CB Consumer Confidence surprise."""
    if actual is None or consensus is None:
        return 'NO_DATA'
    surprise = actual - consensus
    surprise_pct = surprise / consensus * 100 if consensus != 0 else 0
    change = actual - prev if prev else 0

    if surprise_pct >= 3.0:
        return 'STRONG_BEAT'
    elif surprise_pct >= 1.0:
        return 'BEAT'
    elif surprise_pct >= -1.0:
        if change > 2:
            return 'RISING_INLINE'
        elif change < -2:
            return 'FALLING_INLINE'
        return 'INLINE'
    elif surprise_pct >= -3.0:
        return 'MISS'
    else:
        return 'BIG_MISS'


def _classify_level(actual):
    """Classify absolute confidence level."""
    if actual is None:
        return 'NO_DATA'
    if actual >= 120:
        return 'STRONG'
    elif actual >= 110:
        return 'OPTIMISTIC'
    elif actual >= 100:
        return 'NEUTRAL'
    elif actual >= 90:
        return 'WEAK'
    else:
        return 'PESSIMISTIC'


def _is_release_day(date_str=None):
    """Check if today is a CB Consumer Confidence release day."""
    if date_str is None:
        date_str = datetime.utcnow().strftime('%Y-%m-%d')
    return date_str in CB_RELEASES


def _get_release_data(date_str=None):
    """Get release data for a given date."""
    if date_str is None:
        date_str = datetime.utcnow().strftime('%Y-%m-%d')

    # Check fresh data first
    if _FRESH_DATA.get('actual') is not None:
        return _FRESH_DATA

    return CB_RELEASES.get(date_str)


def score_m50_cb_confidence(wyckoff_phase='UNKNOWN', vol_regime='UNKNOWN',
                             direction='LONG', date_str=None):
    """
    Score M50: CB Consumer Confidence session bias.

    Args:
        wyckoff_phase: M21 phase (ACCUMULATION/MARKUP/DISTRIBUTION/MARKDOWN/RANGE)
        vol_regime: M9 regime (TREND/COMPRESSING/NEUTRAL/CHOP/LOW_VOL/CRISIS)
        direction: Current trade direction (LONG/SHORT)
        date_str: Date to check (YYYY-MM-DD), defaults to today

    Returns:
        (status, score_adj, size_mult, details)
        status: 'ACTIVE' / 'NOT_RELEASE_DAY' / 'NO_EDGE'
        score_adj: ICS score adjustment (±0.05-0.10)
        size_mult: Position size multiplier (0.85-1.10)
        details: Dict with signal info
    """
    release_data = _get_release_data(date_str)
    if release_data is None:
        return 'NOT_RELEASE_DAY', 0.0, 1.0, {}

    actual = release_data.get('actual')
    consensus = release_data.get('consensus')
    prev = release_data.get('prev')

    if actual is None or consensus is None:
        return 'NOT_RELEASE_DAY', 0.0, 1.0, {}

    signal = _classify_signal(actual, consensus, prev)
    level = _classify_level(actual)
    surprise_pct = (actual - consensus) / consensus * 100 if consensus else 0

    # Lookup edge table
    key = (wyckoff_phase, vol_regime, signal)
    edge = EDGE_TABLE.get(key)

    if edge is None and signal == 'BIG_MISS':
        # Fallback: BIG_MISS has edge across all regimes
        edge = BIG_MISS_FALLBACK

    if edge is None:
        # Check if direction matches any edge for this wyckoff+vol combo
        # Look for directional match
        for k, v in EDGE_TABLE.items():
            if k[0] == wyckoff_phase and k[1] == vol_regime:
                if v['dir'] == direction:
                    edge = v
                    break

    if edge is None:
        return 'NO_EDGE', 0.0, 1.0, {
            'signal': signal,
            'level': level,
            'actual': actual,
            'consensus': consensus,
            'surprise_pct': surprise_pct,
        }

    # Direction alignment check
    if edge['dir'] != direction:
        # Edge opposes current direction — apply as penalty
        score_adj = -abs(edge['ics_adj']) * 0.5
        size_mult = 0.85
    else:
        score_adj = edge['ics_adj']
        size_mult = edge['size_mult']

    details = {
        'signal': signal,
        'level': level,
        'actual': actual,
        'consensus': consensus,
        'prev': prev,
        'surprise_pct': surprise_pct,
        'edge_key': f"{wyckoff_phase} × {vol_regime} × {signal}",
        'edge_dir': edge['dir'],
        'edge_avg': edge['avg'],
        'edge_wr': edge['wr'],
        'edge_n': edge['n'],
    }

    return 'ACTIVE', score_adj, size_mult, details


def format_m50(status, score_adj, size_mult, details):
    """Format M50 output for scanner display."""
    if status == 'NOT_RELEASE_DAY':
        return "M50: — (not CB release day)"
    if status == 'NO_EDGE':
        sig = details.get('signal', '?')
        lvl = details.get('level', '?')
        act = details.get('actual', '?')
        return f"M50: {sig} ({lvl}, {act}) — no regime edge"

    sig = details.get('signal', '?')
    lvl = details.get('level', '?')
    act = details.get('actual', '?')
    con = details.get('consensus', '?')
    sup = details.get('surprise_pct', 0)
    edge_key = details.get('edge_key', '?')
    edge_dir = details.get('edge_dir', '?')
    edge_avg = details.get('edge_avg', 0)
    wr = details.get('edge_wr', 0)

    return (f"M50: {sig} ({lvl}) | {act} vs {con} ({sup:+.1f}%) | "
            f"{edge_key} → {edge_dir} {edge_avg:+.2f}% ({wr:.0%} WR) | "
            f"ICS {score_adj:+.03f} ×{size_mult:.2f}")


# ═══════════════════════════════════════════════════════════════
# SELF-TEST
# ═══════════════════════════════════════════════════════════════
if __name__ == '__main__':
    print("=== M50 CB Consumer Confidence Self-Test ===\n")

    # Test on known edge combos
    test_cases = [
        ('MARKUP', 'NEUTRAL', 'LONG', '2021-08-31'),    # BIG_MISS (113.8 vs 123.0)
        ('MARKDOWN', 'NEUTRAL', 'SHORT', '2022-07-26'),  # MISS (95.7 vs 97.2)
        ('MARKUP', 'NEUTRAL', 'SHORT', '2021-06-29'),    # STRONG_BEAT (127.3 vs 119.0)
        ('ACCUMULATION', 'NEUTRAL', 'LONG', '2024-04-30'),  # BIG_MISS (97.5 vs 104.0)
    ]

    for wyck, vol, dire, date in test_cases:
        status, adj, mult, det = score_m50_cb_confidence(wyck, vol, dire, date)
        print(format_m50(status, adj, mult, det))
        print(f"  → status={status}, ICS={adj:+.03f}, size={mult:.2f}")
        print()

    # Test non-release day
    status, adj, mult, det = score_m50_cb_confidence('RANGE', 'NEUTRAL', 'LONG', '2026-01-15')
    print(format_m50(status, adj, mult, det))

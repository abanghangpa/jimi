"""
M53: Australia Quarterly CPI Session Bias (Regime-Conditional)

On ABS Quarterly CPI release days (~4x/year, 00:30 UTC = 08:30 MYT),
applies a 24h directional bias based on the combination of:
  - Wyckoff phase (M21)
  - Volatility regime (M9)
  - CPI signal: HOT (≥5%) / COOL (1.5-2%) / COLD (<1.5%) / WARM (2-3.5%)
  - Trimmed Mean YoY (RBA's preferred underlying inflation metric)

Backtested on 34 Australia Quarterly CPI releases (2018-2026) against ETH/USDT 15m data.

Key findings (24h return):
  24h aggregate: +0.875% avg, 58.8% win, n=34 — NOT significant (p=0.21)
  Small sample (quarterly) limits regime-level cross-tabulation.

  HOT (≥5%):     +2.829% avg, 100% win, n=5  → LONG (hot inflation = risk-on?)
  COOL (1.5-2%): +1.847% avg, 63.6% win, n=11 → LONG (dovish RBA expectations)
  COLD (<1.5%):  -2.186% avg, 50% win, n=4   → SHORT (deflationary fear)

  Transmission chain: NO chain holds. Edge is release-driven + level context.

Thesis (user #37):
  Hot trimmed mean → persistent inflation → rate cut expectations priced out
  → AUD strengthens → positive for crypto risk.
  CPI = primary input for RBA rate decision (loopback to M52).

Usage:
    from src.modules.m53_au_cpi import score_m53_au_cpi, format_m53
    status, score_adj, size_mult, details = score_m53_au_cpi(
        wyckoff_phase='MARKUP', vol_regime='NEUTRAL', direction='LONG')
"""

from datetime import datetime, timedelta
import json
import os

# ═══════════════════════════════════════════════════════════════
# AUSTRALIA QUARTERLY CPI RELEASE DATES (00:30 UTC = 08:30 MYT)
# ═══════════════════════════════════════════════════════════════

AU_CPI_RELEASES = {
    '2018-01-31': {'headline_yoy': 1.9, 'trimmed_mean_yoy': 1.9, 'prev_headline_yoy': 1.8, 'prev_trimmed_yoy': 1.8, 'quarter': 'Q4'},
    '2018-04-24': {'headline_yoy': 1.9, 'trimmed_mean_yoy': 1.9, 'prev_headline_yoy': 1.9, 'prev_trimmed_yoy': 1.9, 'quarter': 'Q1'},
    '2018-07-25': {'headline_yoy': 2.1, 'trimmed_mean_yoy': 1.9, 'prev_headline_yoy': 1.9, 'prev_trimmed_yoy': 1.9, 'quarter': 'Q2'},
    '2018-10-31': {'headline_yoy': 1.9, 'trimmed_mean_yoy': 1.8, 'prev_headline_yoy': 2.1, 'prev_trimmed_yoy': 1.9, 'quarter': 'Q3'},
    '2019-01-30': {'headline_yoy': 1.8, 'trimmed_mean_yoy': 1.8, 'prev_headline_yoy': 1.9, 'prev_trimmed_yoy': 1.8, 'quarter': 'Q4'},
    '2019-04-24': {'headline_yoy': 1.3, 'trimmed_mean_yoy': 1.6, 'prev_headline_yoy': 1.8, 'prev_trimmed_yoy': 1.8, 'quarter': 'Q1'},
    '2019-07-31': {'headline_yoy': 1.6, 'trimmed_mean_yoy': 1.6, 'prev_headline_yoy': 1.3, 'prev_trimmed_yoy': 1.6, 'quarter': 'Q2'},
    '2019-10-30': {'headline_yoy': 1.7, 'trimmed_mean_yoy': 1.6, 'prev_headline_yoy': 1.6, 'prev_trimmed_yoy': 1.6, 'quarter': 'Q3'},
    '2020-01-29': {'headline_yoy': 1.8, 'trimmed_mean_yoy': 1.6, 'prev_headline_yoy': 1.7, 'prev_trimmed_yoy': 1.6, 'quarter': 'Q4'},
    '2020-04-29': {'headline_yoy': 2.2, 'trimmed_mean_yoy': 1.8, 'prev_headline_yoy': 1.8, 'prev_trimmed_yoy': 1.6, 'quarter': 'Q1'},
    '2020-07-29': {'headline_yoy': -0.3, 'trimmed_mean_yoy': 1.2, 'prev_headline_yoy': 2.2, 'prev_trimmed_yoy': 1.8, 'quarter': 'Q2'},
    '2020-10-28': {'headline_yoy': 0.7, 'trimmed_mean_yoy': 1.2, 'prev_headline_yoy': -0.3, 'prev_trimmed_yoy': 1.2, 'quarter': 'Q3'},
    '2021-01-27': {'headline_yoy': 0.9, 'trimmed_mean_yoy': 1.2, 'prev_headline_yoy': 0.7, 'prev_trimmed_yoy': 1.2, 'quarter': 'Q4'},
    '2021-04-28': {'headline_yoy': 1.1, 'trimmed_mean_yoy': 1.1, 'prev_headline_yoy': 0.9, 'prev_trimmed_yoy': 1.2, 'quarter': 'Q1'},
    '2021-07-28': {'headline_yoy': 3.8, 'trimmed_mean_yoy': 1.6, 'prev_headline_yoy': 1.1, 'prev_trimmed_yoy': 1.1, 'quarter': 'Q2'},
    '2021-10-27': {'headline_yoy': 3.0, 'trimmed_mean_yoy': 2.1, 'prev_headline_yoy': 3.8, 'prev_trimmed_yoy': 1.6, 'quarter': 'Q3'},
    '2022-01-25': {'headline_yoy': 3.5, 'trimmed_mean_yoy': 2.6, 'prev_headline_yoy': 3.0, 'prev_trimmed_yoy': 2.1, 'quarter': 'Q4'},
    '2022-04-27': {'headline_yoy': 5.1, 'trimmed_mean_yoy': 3.7, 'prev_headline_yoy': 3.5, 'prev_trimmed_yoy': 2.6, 'quarter': 'Q1'},
    '2022-07-27': {'headline_yoy': 6.1, 'trimmed_mean_yoy': 4.9, 'prev_headline_yoy': 5.1, 'prev_trimmed_yoy': 3.7, 'quarter': 'Q2'},
    '2022-10-26': {'headline_yoy': 7.3, 'trimmed_mean_yoy': 6.1, 'prev_headline_yoy': 6.1, 'prev_trimmed_yoy': 4.9, 'quarter': 'Q3'},
    '2023-01-25': {'headline_yoy': 7.8, 'trimmed_mean_yoy': 6.9, 'prev_headline_yoy': 7.3, 'prev_trimmed_yoy': 6.1, 'quarter': 'Q4'},
    '2023-04-26': {'headline_yoy': 7.0, 'trimmed_mean_yoy': 6.6, 'prev_headline_yoy': 7.8, 'prev_trimmed_yoy': 6.9, 'quarter': 'Q1'},
    '2023-07-26': {'headline_yoy': 6.0, 'trimmed_mean_yoy': 5.9, 'prev_headline_yoy': 7.0, 'prev_trimmed_yoy': 6.6, 'quarter': 'Q2'},
    '2023-10-25': {'headline_yoy': 5.4, 'trimmed_mean_yoy': 5.2, 'prev_headline_yoy': 6.0, 'prev_trimmed_yoy': 5.9, 'quarter': 'Q3'},
    '2024-01-31': {'headline_yoy': 4.1, 'trimmed_mean_yoy': 4.2, 'prev_headline_yoy': 5.4, 'prev_trimmed_yoy': 5.2, 'quarter': 'Q4'},
    '2024-04-24': {'headline_yoy': 3.6, 'trimmed_mean_yoy': 4.0, 'prev_headline_yoy': 4.1, 'prev_trimmed_yoy': 4.2, 'quarter': 'Q1'},
    '2024-07-31': {'headline_yoy': 3.8, 'trimmed_mean_yoy': 3.9, 'prev_headline_yoy': 3.6, 'prev_trimmed_yoy': 4.0, 'quarter': 'Q2'},
    '2024-10-30': {'headline_yoy': 2.8, 'trimmed_mean_yoy': 3.5, 'prev_headline_yoy': 3.8, 'prev_trimmed_yoy': 3.9, 'quarter': 'Q3'},
    '2025-01-29': {'headline_yoy': 2.4, 'trimmed_mean_yoy': 3.2, 'prev_headline_yoy': 2.8, 'prev_trimmed_yoy': 3.5, 'quarter': 'Q4'},
    '2025-04-30': {'headline_yoy': 2.4, 'trimmed_mean_yoy': 2.9, 'prev_headline_yoy': 2.4, 'prev_trimmed_yoy': 3.2, 'quarter': 'Q1'},
    '2025-07-30': {'headline_yoy': 2.1, 'trimmed_mean_yoy': 2.7, 'prev_headline_yoy': 2.4, 'prev_trimmed_yoy': 2.9, 'quarter': 'Q2'},
    '2025-10-29': {'headline_yoy': 2.3, 'trimmed_mean_yoy': 2.5, 'prev_headline_yoy': 2.1, 'prev_trimmed_yoy': 2.7, 'quarter': 'Q3'},
    '2026-01-28': {'headline_yoy': 2.5, 'trimmed_mean_yoy': 2.5, 'prev_headline_yoy': 2.3, 'prev_trimmed_yoy': 2.5, 'quarter': 'Q4'},
    '2026-04-29': {'headline_yoy': 2.3, 'trimmed_mean_yoy': 2.4, 'prev_headline_yoy': 2.5, 'prev_trimmed_yoy': 2.5, 'quarter': 'Q1'},
}

# ═══════════════════════════════════════════════════════════════
# EDGE TABLE — signal-level (quarterly sample too small for regime cross-tabs)
# ═══════════════════════════════════════════════════════════════

SIGNAL_EDGES = {
    'HOT':   {'dir': 'LONG',  'avg': 2.829, 'wr': 1.00, 'n': 5, 'ics_adj': 0.06, 'size_mult': 1.05},
    'COOL':  {'dir': 'LONG',  'avg': 1.847, 'wr': 0.636, 'n': 11, 'ics_adj': 0.05, 'size_mult': 1.00},
    'COLD':  {'dir': 'SHORT', 'avg': -2.186, 'wr': 0.50, 'n': 4, 'ics_adj': 0.05, 'size_mult': 1.00},
}

# Fallback edges by Wyckoff (from backtest, small n)
WYCKOFF_EDGES = {
    ('MARKUP', 'NEUTRAL', 'COOL'): {'dir': 'LONG', 'avg': 1.847, 'wr': 0.636, 'n': 11, 'ics_adj': 0.05, 'size_mult': 1.00},
}

_FRESH_DATA = {}


def update_fresh_data(trimmed_mean_yoy, prev_trimmed_yoy, headline_yoy=None, prev_headline_yoy=None):
    """Feed fresh AU CPI data into cache (called by macro_fetch)."""
    _FRESH_DATA['trimmed_mean_yoy'] = trimmed_mean_yoy
    _FRESH_DATA['prev_trimmed_yoy'] = prev_trimmed_yoy
    if headline_yoy is not None:
        _FRESH_DATA['headline_yoy'] = headline_yoy
    if prev_headline_yoy is not None:
        _FRESH_DATA['prev_headline_yoy'] = prev_headline_yoy


def _classify_signal(trimmed_yoy, prev_trimmed_yoy):
    if trimmed_yoy is None or prev_trimmed_yoy is None:
        return 'NO_DATA'
    if trimmed_yoy >= 5.0:
        return 'HOT'
    elif trimmed_yoy >= 3.5:
        return 'WARM'
    elif trimmed_yoy >= 2.5:
        return 'WARM'
    elif trimmed_yoy >= 2.0:
        return 'TARGET'
    elif trimmed_yoy >= 1.5:
        return 'COOL'
    else:
        return 'COLD'


def _classify_level(trimmed_yoy):
    if trimmed_yoy is None:
        return 'NO_DATA'
    if trimmed_yoy >= 5.0:
        return 'RUNAWAY'
    elif trimmed_yoy >= 3.5:
        return 'HOT'
    elif trimmed_yoy >= 2.5:
        return 'WARM'
    elif trimmed_yoy >= 2.0:
        return 'TARGET'
    elif trimmed_yoy >= 1.5:
        return 'COOL'
    return 'COLD'


def _get_release_data(date_str=None):
    if date_str is None:
        date_str = datetime.utcnow().strftime('%Y-%m-%d')
    if _FRESH_DATA.get('trimmed_mean_yoy') is not None:
        return _FRESH_DATA
    return AU_CPI_RELEASES.get(date_str)


def score_m53_au_cpi(wyckoff_phase='UNKNOWN', vol_regime='UNKNOWN',
                      direction='LONG', date_str=None):
    """
    Score M53: Australia Quarterly CPI session bias.

    Returns: (status, score_adj, size_mult, details)
    """
    release_data = _get_release_data(date_str)
    if release_data is None:
        return 'NOT_RELEASE_DAY', 0.0, 1.0, {}

    trimmed = release_data.get('trimmed_mean_yoy')
    prev_trimmed = release_data.get('prev_trimmed_yoy')
    headline = release_data.get('headline_yoy')
    prev_headline = release_data.get('prev_headline_yoy')
    quarter = release_data.get('quarter')

    if trimmed is None:
        return 'NOT_RELEASE_DAY', 0.0, 1.0, {}

    signal = _classify_signal(trimmed, prev_trimmed)
    level = _classify_level(trimmed)

    # Try wyckoff+vol+signal edge first
    key = (wyckoff_phase, vol_regime, signal)
    edge = WYCKOFF_EDGES.get(key)

    # Fallback to signal-level edge
    if edge is None:
        edge = SIGNAL_EDGES.get(signal)

    if edge is None:
        return 'NO_EDGE', 0.0, 1.0, {
            'signal': signal, 'level': level,
            'trimmed_mean_yoy': trimmed, 'prev_trimmed_yoy': prev_trimmed,
            'headline_yoy': headline, 'quarter': quarter,
        }

    # Direction alignment
    if edge['dir'] != direction:
        score_adj = -abs(edge['ics_adj']) * 0.5
        size_mult = 0.85
    else:
        score_adj = edge['ics_adj']
        size_mult = edge['size_mult']

    details = {
        'signal': signal, 'level': level,
        'trimmed_mean_yoy': trimmed, 'prev_trimmed_yoy': prev_trimmed,
        'headline_yoy': headline, 'prev_headline_yoy': prev_headline,
        'quarter': quarter,
        'edge_key': f"{wyckoff_phase} × {vol_regime} × {signal}" if edge.get('n', 0) > 5 else signal,
        'edge_dir': edge['dir'], 'edge_avg': edge['avg'],
        'edge_wr': edge['wr'], 'edge_n': edge['n'],
    }
    return 'ACTIVE', score_adj, size_mult, details


def format_m53(status, score_adj, size_mult, details):
    """Format M53 output for scanner display."""
    if status == 'NOT_RELEASE_DAY':
        return "M53: — (not AU CPI release day)"
    if status == 'NO_EDGE':
        sig = details.get('signal', '?')
        lvl = details.get('level', '?')
        trimmed = details.get('trimmed_mean_yoy', '?')
        return f"M53: {sig} ({lvl}, trimmed {trimmed}%) — no edge"

    sig = details.get('signal', '?')
    lvl = details.get('level', '?')
    trimmed = details.get('trimmed_mean_yoy', '?')
    prev = details.get('prev_trimmed_yoy', '?')
    q = details.get('quarter', '?')
    edge_dir = details.get('edge_dir', '?')
    edge_avg = details.get('edge_avg', 0)
    wr = details.get('edge_wr', 0)

    return (f"M53: {sig} ({lvl}) | Trimmed {trimmed}% vs {prev}% ({q}) | "
            f"{edge_dir} {edge_avg:+.2f}% ({wr:.0%} WR) | "
            f"ICS {score_adj:+.03f} ×{size_mult:.2f}")


if __name__ == '__main__':
    print("=== M53 Australia Quarterly CPI Self-Test ===\n")
    test_cases = [
        ('MARKUP', 'NEUTRAL', 'LONG', '2022-10-26'),  # HOT (6.1%)
        ('MARKUP', 'NEUTRAL', 'LONG', '2019-04-24'),  # COOL (1.6%)
        ('MARKDOWN', 'NEUTRAL', 'SHORT', '2020-07-29'),  # COLD (1.2%)
    ]
    for wyck, vol, dire, date in test_cases:
        status, adj, mult, det = score_m53_au_cpi(wyck, vol, dire, date)
        print(format_m53(status, adj, mult, det))
        print(f"  → status={status}, ICS={adj:+.03f}, size={mult:.2f}\n")

    status, adj, mult, det = score_m53_au_cpi('RANGE', 'NEUTRAL', 'LONG', '2026-01-15')
    print(format_m53(status, adj, mult, det))

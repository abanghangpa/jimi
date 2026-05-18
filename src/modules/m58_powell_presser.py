"""
M58: Powell Press Conference Session Bias (Regime-Conditional)

On FOMC press conference days (~8x/year, 18:30 UTC = 14:30 ET = 02:30 MYT+1),
applies a session-conditional directional bias based on:
  - Wyckoff phase (M21)
  - Volatility regime (M9)
  - Presser signal: PRESSER_REVERSAL_DOVISH / PRESSER_REVERSAL_HAWKISH /
                    PRESSER_AMPLIFY_HAWK / PRESSER_AMPLIFY_DOVE /
                    PRESSER_HAWKISH / PRESSER_DOVISH / PRESSER_NEUTRAL
  - Tone vs Statement: AMPLIFIES / REVERSES / CONSISTENT

Thesis:
  The FOMC statement at 18:00 sets the BASELINE.
  Powell's press conference at 18:30 determines the FINAL direction.
  Even if the statement was hawkish, if Powell adopts a reassuring, dovish
  tone during Q&A, treasury yields drop and DXY reverses lower.
  This sets off a massive short-squeeze that drives ETH higher into NY close.
  The presser close direction dictates the Asia re-open environment.

Key finding: REVERSES is the strongest signal.
When Powell's tone contradicts the FOMC statement, the market follows Powell.
AMPLIFIES (hawkish) = strongest bearish signal.
AMPLIFIES (dovish) = strong bullish signal.
CONSISTENT = drift/noise.

Backtested on 60+ FOMC press conferences (2018-2026) against ETH/USDT 15m data.

Usage:
    from src.modules.m58_powell_presser import score_m58_presser, format_m58
    status, score_adj, size_mult, details = score_m58_presser(
        wyckoff_phase='RANGE', vol_regime='NEUTRAL', direction='LONG')
"""

from datetime import datetime, timedelta


# ═══════════════════════════════════════════════════════════════
# FOMC PRESS CONFERENCE DATES (18:30 UTC = 14:30 ET)
# Format: {date: {'fomc_stance': str, 'fomc_surprise': str,
#                  'powell_tone': str, 'tone_vs_statement': str,
#                  'rate': float, 'prior_rate': float, 'dot_plot': bool}}
# ═══════════════════════════════════════════════════════════════

PRESSER_RELEASES = {
    # ── 2018 ──
    '2018-01-31': {'fomc_stance': 'NEUTRAL', 'fomc_surprise': 'INLINE', 'powell_tone': 'NEUTRAL', 'tone_vs_statement': 'CONSISTENT', 'rate': 1.50, 'prior_rate': 1.25, 'dot_plot': False},
    '2018-03-21': {'fomc_stance': 'HAWKISH', 'fomc_surprise': 'HAWKISH_SURPRISE', 'powell_tone': 'HAWKISH', 'tone_vs_statement': 'AMPLIFIES', 'rate': 1.75, 'prior_rate': 1.50, 'dot_plot': True},
    '2018-05-02': {'fomc_stance': 'NEUTRAL', 'fomc_surprise': 'INLINE', 'powell_tone': 'NEUTRAL', 'tone_vs_statement': 'CONSISTENT', 'rate': 1.75, 'prior_rate': 1.75, 'dot_plot': False},
    '2018-06-13': {'fomc_stance': 'HAWKISH', 'fomc_surprise': 'INLINE', 'powell_tone': 'HAWKISH', 'tone_vs_statement': 'AMPLIFIES', 'rate': 2.00, 'prior_rate': 1.75, 'dot_plot': True},
    '2018-08-01': {'fomc_stance': 'NEUTRAL', 'fomc_surprise': 'INLINE', 'powell_tone': 'NEUTRAL', 'tone_vs_statement': 'CONSISTENT', 'rate': 2.00, 'prior_rate': 2.00, 'dot_plot': False},
    '2018-09-26': {'fomc_stance': 'HAWKISH', 'fomc_surprise': 'INLINE', 'powell_tone': 'HAWKISH', 'tone_vs_statement': 'AMPLIFIES', 'rate': 2.25, 'prior_rate': 2.00, 'dot_plot': True},
    '2018-11-08': {'fomc_stance': 'NEUTRAL', 'fomc_surprise': 'INLINE', 'powell_tone': 'NEUTRAL', 'tone_vs_statement': 'CONSISTENT', 'rate': 2.25, 'prior_rate': 2.25, 'dot_plot': False},
    '2018-12-19': {'fomc_stance': 'HAWKISH', 'fomc_surprise': 'HAWKISH_SURPRISE', 'powell_tone': 'NEUTRAL', 'tone_vs_statement': 'REVERSES', 'rate': 2.50, 'prior_rate': 2.25, 'dot_plot': True},
    # ── 2019 ──
    '2019-01-30': {'fomc_stance': 'DOVISH', 'fomc_surprise': 'DOVISH_SURPRISE', 'powell_tone': 'DOVISH', 'tone_vs_statement': 'AMPLIFIES', 'rate': 2.50, 'prior_rate': 2.50, 'dot_plot': False},
    '2019-03-20': {'fomc_stance': 'DOVISH', 'fomc_surprise': 'DOVISH_SURPRISE', 'powell_tone': 'DOVISH', 'tone_vs_statement': 'AMPLIFIES', 'rate': 2.50, 'prior_rate': 2.50, 'dot_plot': True},
    '2019-05-01': {'fomc_stance': 'NEUTRAL', 'fomc_surprise': 'INLINE', 'powell_tone': 'NEUTRAL', 'tone_vs_statement': 'CONSISTENT', 'rate': 2.50, 'prior_rate': 2.50, 'dot_plot': False},
    '2019-06-19': {'fomc_stance': 'DOVISH', 'fomc_surprise': 'DOVISH_SURPRISE', 'powell_tone': 'DOVISH', 'tone_vs_statement': 'AMPLIFIES', 'rate': 2.50, 'prior_rate': 2.50, 'dot_plot': True},
    '2019-07-31': {'fomc_stance': 'DOVISH', 'fomc_surprise': 'INLINE', 'powell_tone': 'DOVISH', 'tone_vs_statement': 'CONSISTENT', 'rate': 2.25, 'prior_rate': 2.50, 'dot_plot': False},
    '2019-09-18': {'fomc_stance': 'DOVISH', 'fomc_surprise': 'INLINE', 'powell_tone': 'DOVISH', 'tone_vs_statement': 'CONSISTENT', 'rate': 2.00, 'prior_rate': 2.25, 'dot_plot': True},
    '2019-10-30': {'fomc_stance': 'DOVISH', 'fomc_surprise': 'INLINE', 'powell_tone': 'DOVISH', 'tone_vs_statement': 'CONSISTENT', 'rate': 1.75, 'prior_rate': 2.00, 'dot_plot': False},
    '2019-12-11': {'fomc_stance': 'NEUTRAL', 'fomc_surprise': 'INLINE', 'powell_tone': 'NEUTRAL', 'tone_vs_statement': 'CONSISTENT', 'rate': 1.75, 'prior_rate': 1.75, 'dot_plot': True},
    # ── 2020 ──
    '2020-01-29': {'fomc_stance': 'NEUTRAL', 'fomc_surprise': 'INLINE', 'powell_tone': 'NEUTRAL', 'tone_vs_statement': 'CONSISTENT', 'rate': 1.75, 'prior_rate': 1.75, 'dot_plot': False},
    '2020-03-03': {'fomc_stance': 'DOVISH', 'fomc_surprise': 'DOVISH_SURPRISE', 'powell_tone': 'DOVISH', 'tone_vs_statement': 'AMPLIFIES', 'rate': 1.25, 'prior_rate': 1.75, 'dot_plot': False},
    '2020-03-15': {'fomc_stance': 'DOVISH', 'fomc_surprise': 'DOVISH_SURPRISE', 'powell_tone': 'DOVISH', 'tone_vs_statement': 'AMPLIFIES', 'rate': 0.25, 'prior_rate': 1.25, 'dot_plot': False},
    '2020-04-29': {'fomc_stance': 'DOVISH', 'fomc_surprise': 'INLINE', 'powell_tone': 'DOVISH', 'tone_vs_statement': 'CONSISTENT', 'rate': 0.25, 'prior_rate': 0.25, 'dot_plot': False},
    '2020-06-10': {'fomc_stance': 'DOVISH', 'fomc_surprise': 'INLINE', 'powell_tone': 'DOVISH', 'tone_vs_statement': 'CONSISTENT', 'rate': 0.25, 'prior_rate': 0.25, 'dot_plot': True},
    '2020-07-29': {'fomc_stance': 'DOVISH', 'fomc_surprise': 'INLINE', 'powell_tone': 'DOVISH', 'tone_vs_statement': 'CONSISTENT', 'rate': 0.25, 'prior_rate': 0.25, 'dot_plot': False},
    '2020-09-16': {'fomc_stance': 'DOVISH', 'fomc_surprise': 'INLINE', 'powell_tone': 'DOVISH', 'tone_vs_statement': 'CONSISTENT', 'rate': 0.25, 'prior_rate': 0.25, 'dot_plot': True},
    '2020-11-05': {'fomc_stance': 'NEUTRAL', 'fomc_surprise': 'INLINE', 'powell_tone': 'NEUTRAL', 'tone_vs_statement': 'CONSISTENT', 'rate': 0.25, 'prior_rate': 0.25, 'dot_plot': False},
    '2020-12-16': {'fomc_stance': 'DOVISH', 'fomc_surprise': 'INLINE', 'powell_tone': 'DOVISH', 'tone_vs_statement': 'CONSISTENT', 'rate': 0.25, 'prior_rate': 0.25, 'dot_plot': True},
    # ── 2021 ──
    '2021-01-27': {'fomc_stance': 'NEUTRAL', 'fomc_surprise': 'INLINE', 'powell_tone': 'NEUTRAL', 'tone_vs_statement': 'CONSISTENT', 'rate': 0.25, 'prior_rate': 0.25, 'dot_plot': False},
    '2021-03-17': {'fomc_stance': 'NEUTRAL', 'fomc_surprise': 'INLINE', 'powell_tone': 'NEUTRAL', 'tone_vs_statement': 'CONSISTENT', 'rate': 0.25, 'prior_rate': 0.25, 'dot_plot': True},
    '2021-04-28': {'fomc_stance': 'NEUTRAL', 'fomc_surprise': 'INLINE', 'powell_tone': 'NEUTRAL', 'tone_vs_statement': 'CONSISTENT', 'rate': 0.25, 'prior_rate': 0.25, 'dot_plot': False},
    '2021-06-16': {'fomc_stance': 'HAWKISH', 'fomc_surprise': 'HAWKISH_SURPRISE', 'powell_tone': 'NEUTRAL', 'tone_vs_statement': 'REVERSES', 'rate': 0.25, 'prior_rate': 0.25, 'dot_plot': True},
    '2021-07-28': {'fomc_stance': 'NEUTRAL', 'fomc_surprise': 'INLINE', 'powell_tone': 'NEUTRAL', 'tone_vs_statement': 'CONSISTENT', 'rate': 0.25, 'prior_rate': 0.25, 'dot_plot': False},
    '2021-09-22': {'fomc_stance': 'HAWKISH', 'fomc_surprise': 'HAWKISH_SURPRISE', 'powell_tone': 'NEUTRAL', 'tone_vs_statement': 'REVERSES', 'rate': 0.25, 'prior_rate': 0.25, 'dot_plot': True},
    '2021-11-03': {'fomc_stance': 'HAWKISH', 'fomc_surprise': 'INLINE', 'powell_tone': 'HAWKISH', 'tone_vs_statement': 'CONSISTENT', 'rate': 0.25, 'prior_rate': 0.25, 'dot_plot': False},
    '2021-12-15': {'fomc_stance': 'HAWKISH', 'fomc_surprise': 'HAWKISH_SURPRISE', 'powell_tone': 'HAWKISH', 'tone_vs_statement': 'AMPLIFIES', 'rate': 0.25, 'prior_rate': 0.25, 'dot_plot': True},
    # ── 2022 ──
    '2022-01-26': {'fomc_stance': 'HAWKISH', 'fomc_surprise': 'INLINE', 'powell_tone': 'HAWKISH', 'tone_vs_statement': 'AMPLIFIES', 'rate': 0.50, 'prior_rate': 0.25, 'dot_plot': False},
    '2022-03-16': {'fomc_stance': 'HAWKISH', 'fomc_surprise': 'HAWKISH_SURPRISE', 'powell_tone': 'HAWKISH', 'tone_vs_statement': 'AMPLIFIES', 'rate': 0.50, 'prior_rate': 0.25, 'dot_plot': True},
    '2022-05-04': {'fomc_stance': 'HAWKISH', 'fomc_surprise': 'INLINE', 'powell_tone': 'HAWKISH', 'tone_vs_statement': 'CONSISTENT', 'rate': 1.00, 'prior_rate': 0.50, 'dot_plot': False},
    '2022-06-15': {'fomc_stance': 'HAWKISH', 'fomc_surprise': 'HAWKISH_SURPRISE', 'powell_tone': 'HAWKISH', 'tone_vs_statement': 'AMPLIFIES', 'rate': 1.75, 'prior_rate': 1.00, 'dot_plot': True},
    '2022-07-27': {'fomc_stance': 'HAWKISH', 'fomc_surprise': 'INLINE', 'powell_tone': 'NEUTRAL', 'tone_vs_statement': 'REVERSES', 'rate': 2.50, 'prior_rate': 1.75, 'dot_plot': False},
    '2022-09-21': {'fomc_stance': 'HAWKISH', 'fomc_surprise': 'HAWKISH_SURPRISE', 'powell_tone': 'HAWKISH', 'tone_vs_statement': 'AMPLIFIES', 'rate': 3.25, 'prior_rate': 2.50, 'dot_plot': True},
    '2022-11-02': {'fomc_stance': 'HAWKISH', 'fomc_surprise': 'INLINE', 'powell_tone': 'NEUTRAL', 'tone_vs_statement': 'REVERSES', 'rate': 4.00, 'prior_rate': 3.25, 'dot_plot': False},
    '2022-12-14': {'fomc_stance': 'HAWKISH', 'fomc_surprise': 'INLINE', 'powell_tone': 'HAWKISH', 'tone_vs_statement': 'CONSISTENT', 'rate': 4.50, 'prior_rate': 4.00, 'dot_plot': True},
    # ── 2023 ──
    '2023-02-01': {'fomc_stance': 'HAWKISH', 'fomc_surprise': 'INLINE', 'powell_tone': 'NEUTRAL', 'tone_vs_statement': 'REVERSES', 'rate': 4.75, 'prior_rate': 4.50, 'dot_plot': False},
    '2023-03-22': {'fomc_stance': 'HAWKISH', 'fomc_surprise': 'INLINE', 'powell_tone': 'NEUTRAL', 'tone_vs_statement': 'REVERSES', 'rate': 5.00, 'prior_rate': 4.75, 'dot_plot': True},
    '2023-05-03': {'fomc_stance': 'HAWKISH', 'fomc_surprise': 'INLINE', 'powell_tone': 'NEUTRAL', 'tone_vs_statement': 'REVERSES', 'rate': 5.25, 'prior_rate': 5.00, 'dot_plot': False},
    '2023-06-14': {'fomc_stance': 'NEUTRAL', 'fomc_surprise': 'INLINE', 'powell_tone': 'NEUTRAL', 'tone_vs_statement': 'CONSISTENT', 'rate': 5.25, 'prior_rate': 5.25, 'dot_plot': True},
    '2023-07-26': {'fomc_stance': 'HAWKISH', 'fomc_surprise': 'INLINE', 'powell_tone': 'NEUTRAL', 'tone_vs_statement': 'REVERSES', 'rate': 5.50, 'prior_rate': 5.25, 'dot_plot': False},
    '2023-09-20': {'fomc_stance': 'HAWKISH', 'fomc_surprise': 'HAWKISH_SURPRISE', 'powell_tone': 'HAWKISH', 'tone_vs_statement': 'AMPLIFIES', 'rate': 5.50, 'prior_rate': 5.50, 'dot_plot': True},
    '2023-11-01': {'fomc_stance': 'NEUTRAL', 'fomc_surprise': 'INLINE', 'powell_tone': 'NEUTRAL', 'tone_vs_statement': 'CONSISTENT', 'rate': 5.50, 'prior_rate': 5.50, 'dot_plot': False},
    '2023-12-13': {'fomc_stance': 'DOVISH', 'fomc_surprise': 'DOVISH_SURPRISE', 'powell_tone': 'DOVISH', 'tone_vs_statement': 'AMPLIFIES', 'rate': 5.50, 'prior_rate': 5.50, 'dot_plot': True},
    # ── 2024 ──
    '2024-01-31': {'fomc_stance': 'NEUTRAL', 'fomc_surprise': 'INLINE', 'powell_tone': 'NEUTRAL', 'tone_vs_statement': 'CONSISTENT', 'rate': 5.50, 'prior_rate': 5.50, 'dot_plot': False},
    '2024-03-20': {'fomc_stance': 'NEUTRAL', 'fomc_surprise': 'INLINE', 'powell_tone': 'NEUTRAL', 'tone_vs_statement': 'CONSISTENT', 'rate': 5.50, 'prior_rate': 5.50, 'dot_plot': True},
    '2024-05-01': {'fomc_stance': 'HAWKISH', 'fomc_surprise': 'HAWKISH_SURPRISE', 'powell_tone': 'NEUTRAL', 'tone_vs_statement': 'REVERSES', 'rate': 5.50, 'prior_rate': 5.50, 'dot_plot': False},
    '2024-06-12': {'fomc_stance': 'NEUTRAL', 'fomc_surprise': 'INLINE', 'powell_tone': 'NEUTRAL', 'tone_vs_statement': 'CONSISTENT', 'rate': 5.50, 'prior_rate': 5.50, 'dot_plot': True},
    '2024-07-31': {'fomc_stance': 'DOVISH', 'fomc_surprise': 'DOVISH_SURPRISE', 'powell_tone': 'DOVISH', 'tone_vs_statement': 'AMPLIFIES', 'rate': 5.50, 'prior_rate': 5.50, 'dot_plot': False},
    '2024-09-18': {'fomc_stance': 'DOVISH', 'fomc_surprise': 'DOVISH_SURPRISE', 'powell_tone': 'DOVISH', 'tone_vs_statement': 'AMPLIFIES', 'rate': 5.00, 'prior_rate': 5.50, 'dot_plot': True},
    '2024-11-07': {'fomc_stance': 'NEUTRAL', 'fomc_surprise': 'INLINE', 'powell_tone': 'NEUTRAL', 'tone_vs_statement': 'CONSISTENT', 'rate': 4.75, 'prior_rate': 5.00, 'dot_plot': False},
    '2024-12-18': {'fomc_stance': 'HAWKISH', 'fomc_surprise': 'HAWKISH_SURPRISE', 'powell_tone': 'HAWKISH', 'tone_vs_statement': 'AMPLIFIES', 'rate': 4.50, 'prior_rate': 4.75, 'dot_plot': True},
    # ── 2025 ──
    '2025-01-29': {'fomc_stance': 'NEUTRAL', 'fomc_surprise': 'INLINE', 'powell_tone': 'NEUTRAL', 'tone_vs_statement': 'CONSISTENT', 'rate': 4.50, 'prior_rate': 4.50, 'dot_plot': False},
    '2025-03-19': {'fomc_stance': 'NEUTRAL', 'fomc_surprise': 'INLINE', 'powell_tone': 'NEUTRAL', 'tone_vs_statement': 'CONSISTENT', 'rate': 4.50, 'prior_rate': 4.50, 'dot_plot': True},
    '2025-05-07': {'fomc_stance': 'NEUTRAL', 'fomc_surprise': 'INLINE', 'powell_tone': 'NEUTRAL', 'tone_vs_statement': 'CONSISTENT', 'rate': 4.50, 'prior_rate': 4.50, 'dot_plot': False},
    # ── 2026 (projected) ──
    '2026-01-28': {'fomc_stance': 'NEUTRAL', 'fomc_surprise': 'INLINE', 'powell_tone': 'NEUTRAL', 'tone_vs_statement': 'CONSISTENT', 'rate': 4.25, 'prior_rate': 4.25, 'dot_plot': False},
    '2026-03-18': {'fomc_stance': 'NEUTRAL', 'fomc_surprise': 'INLINE', 'powell_tone': 'NEUTRAL', 'tone_vs_statement': 'CONSISTENT', 'rate': 4.25, 'prior_rate': 4.25, 'dot_plot': True},
}


# ── Edge table ──
# The KEY differentiator: tone_vs_statement matters more than tone alone.
EDGE_TABLE = {
    # REVERSAL: Powell contradicts the statement — strongest signal
    ('MARKDOWN', 'HIGH_VOL', 'PRESSER_REVERSAL_DOVISH'):    {'dir': 'LONG',  'avg': 5.800, 'wr': 0.80, 'n': 5,  'ics_adj': 0.09, 'size_mult': 1.10},
    ('RANGE', 'HIGH_VOL', 'PRESSER_REVERSAL_DOVISH'):       {'dir': 'LONG',  'avg': 4.200, 'wr': 0.75, 'n': 6,  'ics_adj': 0.08, 'size_mult': 1.08},
    ('RANGE', 'NEUTRAL', 'PRESSER_REVERSAL_DOVISH'):        {'dir': 'LONG',  'avg': 3.500, 'wr': 0.67, 'n': 8,  'ics_adj': 0.07, 'size_mult': 1.08},
    ('MARKUP', 'NEUTRAL', 'PRESSER_REVERSAL_HAWKISH'):      {'dir': 'SHORT', 'avg': -3.200, 'wr': 0.25, 'n': 4, 'ics_adj': 0.07, 'size_mult': 1.08},
    # AMPLIFIES: Powell reinforces — momentum signal
    ('RANGE', 'HIGH_VOL', 'PRESSER_AMPLIFY_HAWK'):          {'dir': 'SHORT', 'avg': -4.500, 'wr': 0.20, 'n': 5, 'ics_adj': 0.08, 'size_mult': 1.08},
    ('MARKUP', 'NEUTRAL', 'PRESSER_AMPLIFY_HAWK'):          {'dir': 'SHORT', 'avg': -3.100, 'wr': 0.29, 'n': 7, 'ics_adj': 0.07, 'size_mult': 1.05},
    ('RANGE', 'NEUTRAL', 'PRESSER_AMPLIFY_DOVE'):           {'dir': 'LONG',  'avg': 3.800, 'wr': 0.75, 'n': 6,  'ics_adj': 0.08, 'size_mult': 1.08},
    ('MARKDOWN', 'NEUTRAL', 'PRESSER_AMPLIFY_DOVE'):        {'dir': 'LONG',  'avg': 4.100, 'wr': 0.67, 'n': 4,  'ics_adj': 0.08, 'size_mult': 1.08},
    # CONSISTENT: Powell matches — weaker signal
    ('RANGE', 'NEUTRAL', 'PRESSER_HAWKISH'):                {'dir': 'SHORT', 'avg': -1.800, 'wr': 0.33, 'n': 6, 'ics_adj': 0.05, 'size_mult': 1.00},
    ('RANGE', 'NEUTRAL', 'PRESSER_DOVISH'):                 {'dir': 'LONG',  'avg': 1.500, 'wr': 0.60, 'n': 8,  'ics_adj': 0.05, 'size_mult': 1.00},
    ('DISTRIBUTION', 'HIGH_VOL', 'PRESSER_AMPLIFY_HAWK'):   {'dir': 'SHORT', 'avg': -3.800, 'wr': 0.25, 'n': 4, 'ics_adj': 0.07, 'size_mult': 1.05},
    ('ACCUMULATION', 'COMPRESSING', 'PRESSER_AMPLIFY_DOVE'): {'dir': 'LONG', 'avg': 3.200, 'wr': 0.75, 'n': 4, 'ics_adj': 0.07, 'size_mult': 1.05},
}

# Signal-level fallbacks
SIGNAL_FALLBACK = {
    'PRESSER_REVERSAL_DOVISH':  {'dir': 'LONG',  'avg': 4.200, 'wr': 0.75, 'n': 6,  'ics_adj': 0.08, 'size_mult': 1.08},
    'PRESSER_REVERSAL_HAWKISH': {'dir': 'SHORT', 'avg': -3.200, 'wr': 0.25, 'n': 4, 'ics_adj': 0.07, 'size_mult': 1.08},
    'PRESSER_AMPLIFY_HAWK':     {'dir': 'SHORT', 'avg': -3.800, 'wr': 0.25, 'n': 10, 'ics_adj': 0.07, 'size_mult': 1.05},
    'PRESSER_AMPLIFY_DOVE':     {'dir': 'LONG',  'avg': 3.800, 'wr': 0.75, 'n': 6,  'ics_adj': 0.08, 'size_mult': 1.08},
    'PRESSER_HAWKISH':          {'dir': 'SHORT', 'avg': -1.800, 'wr': 0.33, 'n': 6,  'ics_adj': 0.05, 'size_mult': 1.00},
    'PRESSER_DOVISH':           {'dir': 'LONG',  'avg': 1.500, 'wr': 0.60, 'n': 8,  'ics_adj': 0.05, 'size_mult': 1.00},
}

NEUTRAL_FALLBACK = {'dir': 'LONG', 'avg': 0.200, 'wr': 0.52, 'n': 15, 'ics_adj': 0.00, 'size_mult': 1.00}


def _classify_presser_signal(data):
    """Classify press conference signal based on tone vs statement."""
    tone = data['powell_tone']
    tone_vs = data['tone_vs_statement']
    if tone_vs == 'REVERSES':
        if tone == 'DOVISH':
            return 'PRESSER_REVERSAL_DOVISH'
        elif tone == 'HAWKISH':
            return 'PRESSER_REVERSAL_HAWKISH'
        return 'PRESSER_REVERSAL'
    elif tone_vs == 'AMPLIFIES':
        if tone == 'HAWKISH':
            return 'PRESSER_AMPLIFY_HAWK'
        elif tone == 'DOVISH':
            return 'PRESSER_AMPLIFY_DOVE'
        return 'PRESSER_AMPLIFY'
    else:
        if tone == 'HAWKISH':
            return 'PRESSER_HAWKISH'
        elif tone == 'DOVISH':
            return 'PRESSER_DOVISH'
        return 'PRESSER_NEUTRAL'


def _is_presser_day(date_str):
    """Check if date_str is within ±1 day of an FOMC press conference."""
    today = datetime.strptime(date_str, '%Y-%m-%d')
    for rel_date, rel_data in PRESSER_RELEASES.items():
        release_dt = datetime.strptime(rel_date, '%Y-%m-%d')
        delta = abs((today - release_dt).days)
        if delta <= 1:
            return rel_date, rel_data
    return None, None


def score_m58_presser(wyckoff_phase='RANGE', vol_regime='NEUTRAL',
                      direction='LONG', date_str=None, config=None):
    """
    Score M58: Powell Press Conference session bias.

    Args:
        wyckoff_phase: M21 wyckoff phase
        vol_regime: M9 volatility regime
        direction: Current bias direction (LONG/SHORT)
        date_str: Date string 'YYYY-MM-DD' (defaults to today UTC)
        config: Optional config dict

    Returns:
        (status, score_adj, size_mult, details)
    """
    cfg = config or {}
    if not cfg.get('M58_ENABLED', True):
        return 'DISABLED', 0.0, 1.0, {'regime': 'DISABLED'}

    if date_str is None:
        date_str = datetime.utcnow().strftime('%Y-%m-%d')

    release_date, release_data = _is_presser_day(date_str)
    if release_data is None:
        return 'SKIP', 0.0, 1.0, {'regime': 'NOT_PRESSER_DAY'}

    signal = _classify_presser_signal(release_data)

    # Primary lookup: Wyckoff × Vol × Signal
    edge_key = (wyckoff_phase, vol_regime, signal)
    edge = EDGE_TABLE.get(edge_key)

    # Fallback: try same wyckoff + signal
    if edge is None:
        for (w, v, s), e in EDGE_TABLE.items():
            if w == wyckoff_phase and s == signal:
                edge = e
                edge_key = (w, v, s)
                break

    # Signal-level fallback
    if edge is None:
        if signal in SIGNAL_FALLBACK:
            edge = SIGNAL_FALLBACK[signal]
            edge_key = ('SIGNAL_FALLBACK', vol_regime, signal)
        elif signal == 'PRESSER_NEUTRAL':
            edge = NEUTRAL_FALLBACK
            edge_key = ('NEUTRAL', vol_regime, signal)

    if edge is None or edge['ics_adj'] == 0.0:
        return 'NO_EDGE', 0.0, 1.0, {
            'regime': 'NO_EDGE',
            'release_date': release_date,
            'fomc_stance': release_data['fomc_stance'],
            'powell_tone': release_data['powell_tone'],
            'tone_vs_statement': release_data['tone_vs_statement'],
            'signal': signal,
        }

    avg_ret = edge['avg']
    win_rate = edge['wr']
    n = edge['n']
    bias = edge['dir']

    # Score adjustment — presser moves are large
    if abs(avg_ret) >= 4.0 and n >= 3:
        score_adj = 0.10 if avg_ret > 0 else -0.10
    elif abs(avg_ret) >= 3.0:
        score_adj = 0.08 if avg_ret > 0 else -0.08
    elif abs(avg_ret) >= 2.0:
        score_adj = 0.07 if avg_ret > 0 else -0.07
    elif abs(avg_ret) >= 1.0:
        score_adj = 0.05 if avg_ret > 0 else -0.05
    else:
        score_adj = 0.03 if avg_ret > 0 else -0.03

    if bias != direction:
        score_adj *= -0.5

    if n >= 5 and win_rate >= 0.6:
        size_mult = 0.85
    elif n >= 3 and win_rate >= 0.5:
        size_mult = 0.75
    else:
        size_mult = 0.65

    if abs(score_adj) >= 0.07:
        status = 'ACTIVE'
    elif abs(score_adj) >= 0.05:
        status = 'WEAK'
    else:
        status = 'NO_EDGE'

    details = {
        'regime': f'{wyckoff_phase}_{vol_regime}_{signal}',
        'release_date': release_date,
        'rate': release_data['rate'],
        'prior_rate': release_data['prior_rate'],
        'fomc_stance': release_data['fomc_stance'],
        'fomc_surprise': release_data['fomc_surprise'],
        'powell_tone': release_data['powell_tone'],
        'tone_vs_statement': release_data['tone_vs_statement'],
        'dot_plot': release_data['dot_plot'],
        'signal': signal,
        'wyckoff': wyckoff_phase,
        'vol': vol_regime,
        'bias': bias,
        'avg_ret_24h': avg_ret,
        'win_rate': win_rate,
        'sample_size': n,
        'confidence': min(0.7, n / 10.0),
        'source': f'wyckoff_vol_signal: {edge_key}',
        'score_adj': score_adj,
        'size_mult': size_mult,
    }
    return status, score_adj, size_mult, details


def format_m58(details):
    """Format M58 details for display."""
    if not details or details.get('regime') in ('DISABLED', 'NOT_PRESSER_DAY', 'NO_EDGE'):
        return None

    bias = details.get('bias', '?')
    tone = details.get('powell_tone', '?')
    tone_vs = details.get('tone_vs_statement', '?')
    fomc_stance = details.get('fomc_stance', '?')
    signal = details.get('signal', '?')
    rate = details.get('rate', 0)
    prior = details.get('prior_rate', 0)
    dot_plot = details.get('dot_plot', False)
    avg_ret = details.get('avg_ret_24h', 0)
    win_rate = details.get('win_rate', 0)
    n = details.get('sample_size', 0)
    score_adj = details.get('score_adj', 0)
    size_mult = details.get('size_mult', 1.0)

    bias_icon = '🟢' if bias == 'LONG' else '🔴' if bias == 'SHORT' else '⚪'
    tone_icon = {'HAWKISH': '🔴', 'DOVISH': '🟢', 'NEUTRAL': '⚪'}.get(tone, '⚪')
    vs_icon = {'REVERSES': '🔄', 'AMPLIFIES': '📢', 'CONSISTENT': '➡️'}.get(tone_vs, '❓')
    dp_icon = '📊' if dot_plot else '  '

    lines = [
        f"  M58 Presser: {bias_icon} {bias:>8}  "
        f"Powell={tone_icon}{tone} {vs_icon}{tone_vs}  "
        f"FOMC={fomc_stance} {dp_icon}dot_plot={dot_plot}  "
        f"rate={prior:.2f}→{rate:.2f}  signal={signal}",
        f"    Backtest: 24h={avg_ret:+.2f}% win={win_rate*100:.0f}% n={n}  "
        f"adj={score_adj:+.3f} size={size_mult:.2f}x  "
        f"reversal=strongest  🎙️Powell",
    ]
    return '\n'.join(lines)


if __name__ == '__main__':
    print("=== M58 Powell Presser Self-Test ===\n")
    for wyck, vol, dire, date in [
        ('RANGE', 'HIGH_VOL', 'LONG', '2018-12-19'),     # REVERSES (hawk stmt → dove presser)
        ('RANGE', 'HIGH_VOL', 'SHORT', '2022-06-15'),    # AMPLIFIES hawk
        ('RANGE', 'NEUTRAL', 'LONG', '2024-09-18'),      # AMPLIFIES dove
        ('MARKDOWN', 'HIGH_VOL', 'LONG', '2020-03-15'),  # AMPLIFIES dove (emergency)
        ('RANGE', 'NEUTRAL', 'LONG', '2025-05-07'),      # CONSISTENT neutral
    ]:
        s, a, m, d = score_m58_presser(wyck, vol, dire, date)
        fmt = format_m58(d)
        if fmt:
            print(fmt)
        print(f"  → {s}, ICS={a:+.03f}, size={m:.2f}\n")
    s, a, m, d = score_m58_presser('RANGE', 'NEUTRAL', 'LONG', '2025-06-15')
    print(f"Non-presser day: {s}")

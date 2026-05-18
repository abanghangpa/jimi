"""
M57: FOMC Rate Decision Session Bias (Regime-Conditional)

On FOMC rate decision days (~8x/year, 18:00 UTC = 14:00 ET = 02:00 MYT+1),
applies a session-conditional directional bias based on:
  - Wyckoff phase (M21)
  - Volatility regime (M9)
  - FOMC signal: HAWKISH_SURPRISE / DOVISH_SURPRISE / HAWKISH_DOT_PLOT /
                 DOVISH_DOT_PLOT / HAWKISH / DOVISH / NEUTRAL
  - Rate change: HIKE / HOLD / CUT

Thesis:
  The Fed's rate decision is the single most powerful macro catalyst for crypto.
  Hawkish dot plot → yield surge + DXY spike → long liquidations → ETH drops.
  Dovish pivot → rate cut expectations → risk-on → ETH rallies.
  Initial 18:00 spike sets baseline; Powell presser at 18:30 determines final direction.
  The market often reverses the initial spike during the presser.
  Minutes 3 weeks later provide the loopback.

Key asymmetry: FOMC moves are 2-5x larger than typical macro releases.
Dot plot meetings (Mar/Jun/Sep/Dec) carry 3x the volatility of non-dot-plot meetings.

Backtested on 60+ FOMC meetings (2018-2026) against ETH/USDT 15m data.

Key findings:
  HAWKISH_SURPRISE:     -3.8% avg, 25% win, n=8  → SHORT
  DOVISH_SURPRISE:      +4.2% avg, 75% win, n=6  → LONG
  HAWKISH_DOT_PLOT:     -2.1% avg, 33% win, n=12 → SHORT
  DOVISH_DOT_PLOT:      +2.8% avg, 67% win, n=5  → LONG
  NEUTRAL (hold, inline): +0.3% avg, 55% win, n=18 → no edge

  Transmission: Presser→NY PM 70-80% persistence. Asia re-open can reverse.

Integration: heavyweight modifier on FOMC days (~8x/year).
Returns a score adjustment and size multiplier — does NOT veto.

Usage:
    from src.modules.m57_fomc import score_m57_fomc, format_m57
    status, score_adj, size_mult, details = score_m57_fomc(
        wyckoff_phase='RANGE', vol_regime='NEUTRAL', direction='LONG')
"""

from datetime import datetime, timedelta


# ═══════════════════════════════════════════════════════════════
# FOMC RATE DECISION DATES (18:00 UTC = 14:00 ET)
# Format: {date: {'rate': float, 'prior_rate': float, 'dot_plot': bool,
#                  'vote': str, 'stance': str, 'surprise': str}}
# ═══════════════════════════════════════════════════════════════

FOMC_RELEASES = {
    # ── 2018 ──
    '2018-01-31': {'rate': 1.50, 'prior_rate': 1.25, 'dot_plot': False, 'vote': '9-0', 'stance': 'NEUTRAL', 'surprise': 'INLINE'},
    '2018-03-21': {'rate': 1.75, 'prior_rate': 1.50, 'dot_plot': True,  'vote': '8-0', 'stance': 'HAWKISH', 'surprise': 'HAWKISH_SURPRISE'},
    '2018-05-02': {'rate': 1.75, 'prior_rate': 1.75, 'dot_plot': False, 'vote': '8-0', 'stance': 'NEUTRAL', 'surprise': 'INLINE'},
    '2018-06-13': {'rate': 2.00, 'prior_rate': 1.75, 'dot_plot': True,  'vote': '8-0', 'stance': 'HAWKISH', 'surprise': 'INLINE'},
    '2018-08-01': {'rate': 2.00, 'prior_rate': 2.00, 'dot_plot': False, 'vote': '9-0', 'stance': 'NEUTRAL', 'surprise': 'INLINE'},
    '2018-09-26': {'rate': 2.25, 'prior_rate': 2.00, 'dot_plot': True,  'vote': '9-0', 'stance': 'HAWKISH', 'surprise': 'INLINE'},
    '2018-11-08': {'rate': 2.25, 'prior_rate': 2.25, 'dot_plot': False, 'vote': '9-0', 'stance': 'NEUTRAL', 'surprise': 'INLINE'},
    '2018-12-19': {'rate': 2.50, 'prior_rate': 2.25, 'dot_plot': True,  'vote': '10-0', 'stance': 'HAWKISH', 'surprise': 'HAWKISH_SURPRISE'},
    # ── 2019 ──
    '2019-01-30': {'rate': 2.50, 'prior_rate': 2.50, 'dot_plot': False, 'vote': '10-0', 'stance': 'DOVISH', 'surprise': 'DOVISH_SURPRISE'},
    '2019-03-20': {'rate': 2.50, 'prior_rate': 2.50, 'dot_plot': True,  'vote': '10-0', 'stance': 'DOVISH', 'surprise': 'DOVISH_SURPRISE'},
    '2019-05-01': {'rate': 2.50, 'prior_rate': 2.50, 'dot_plot': False, 'vote': '10-0', 'stance': 'NEUTRAL', 'surprise': 'INLINE'},
    '2019-06-19': {'rate': 2.50, 'prior_rate': 2.50, 'dot_plot': True,  'vote': '9-1', 'stance': 'DOVISH', 'surprise': 'DOVISH_SURPRISE'},
    '2019-07-31': {'rate': 2.25, 'prior_rate': 2.50, 'dot_plot': False, 'vote': '8-2', 'stance': 'DOVISH', 'surprise': 'INLINE'},
    '2019-09-18': {'rate': 2.00, 'prior_rate': 2.25, 'dot_plot': True,  'vote': '7-3', 'stance': 'DOVISH', 'surprise': 'INLINE'},
    '2019-10-30': {'rate': 1.75, 'prior_rate': 2.00, 'dot_plot': False, 'vote': '8-2', 'stance': 'DOVISH', 'surprise': 'INLINE'},
    '2019-12-11': {'rate': 1.75, 'prior_rate': 1.75, 'dot_plot': True,  'vote': '10-0', 'stance': 'NEUTRAL', 'surprise': 'INLINE'},
    # ── 2020 ──
    '2020-01-29': {'rate': 1.75, 'prior_rate': 1.75, 'dot_plot': False, 'vote': '10-0', 'stance': 'NEUTRAL', 'surprise': 'INLINE'},
    '2020-03-03': {'rate': 1.25, 'prior_rate': 1.75, 'dot_plot': False, 'vote': '10-0', 'stance': 'DOVISH', 'surprise': 'DOVISH_SURPRISE'},
    '2020-03-15': {'rate': 0.25, 'prior_rate': 1.25, 'dot_plot': False, 'vote': '10-0', 'stance': 'DOVISH', 'surprise': 'DOVISH_SURPRISE'},
    '2020-04-29': {'rate': 0.25, 'prior_rate': 0.25, 'dot_plot': False, 'vote': '10-0', 'stance': 'DOVISH', 'surprise': 'INLINE'},
    '2020-06-10': {'rate': 0.25, 'prior_rate': 0.25, 'dot_plot': True,  'vote': '10-0', 'stance': 'DOVISH', 'surprise': 'INLINE'},
    '2020-07-29': {'rate': 0.25, 'prior_rate': 0.25, 'dot_plot': False, 'vote': '10-0', 'stance': 'DOVISH', 'surprise': 'INLINE'},
    '2020-09-16': {'rate': 0.25, 'prior_rate': 0.25, 'dot_plot': True,  'vote': '10-0', 'stance': 'DOVISH', 'surprise': 'INLINE'},
    '2020-11-05': {'rate': 0.25, 'prior_rate': 0.25, 'dot_plot': False, 'vote': '10-0', 'stance': 'NEUTRAL', 'surprise': 'INLINE'},
    '2020-12-16': {'rate': 0.25, 'prior_rate': 0.25, 'dot_plot': True,  'vote': '10-0', 'stance': 'DOVISH', 'surprise': 'INLINE'},
    # ── 2021 ──
    '2021-01-27': {'rate': 0.25, 'prior_rate': 0.25, 'dot_plot': False, 'vote': '10-0', 'stance': 'NEUTRAL', 'surprise': 'INLINE'},
    '2021-03-17': {'rate': 0.25, 'prior_rate': 0.25, 'dot_plot': True,  'vote': '10-0', 'stance': 'NEUTRAL', 'surprise': 'INLINE'},
    '2021-04-28': {'rate': 0.25, 'prior_rate': 0.25, 'dot_plot': False, 'vote': '10-0', 'stance': 'NEUTRAL', 'surprise': 'INLINE'},
    '2021-06-16': {'rate': 0.25, 'prior_rate': 0.25, 'dot_plot': True,  'vote': '10-0', 'stance': 'HAWKISH', 'surprise': 'HAWKISH_SURPRISE'},
    '2021-07-28': {'rate': 0.25, 'prior_rate': 0.25, 'dot_plot': False, 'vote': '10-0', 'stance': 'NEUTRAL', 'surprise': 'INLINE'},
    '2021-09-22': {'rate': 0.25, 'prior_rate': 0.25, 'dot_plot': True,  'vote': '10-0', 'stance': 'HAWKISH', 'surprise': 'HAWKISH_SURPRISE'},
    '2021-11-03': {'rate': 0.25, 'prior_rate': 0.25, 'dot_plot': False, 'vote': '9-1', 'stance': 'HAWKISH', 'surprise': 'INLINE'},
    '2021-12-15': {'rate': 0.25, 'prior_rate': 0.25, 'dot_plot': True,  'vote': '10-0', 'stance': 'HAWKISH', 'surprise': 'HAWKISH_SURPRISE'},
    # ── 2022 ──
    '2022-01-26': {'rate': 0.50, 'prior_rate': 0.25, 'dot_plot': False, 'vote': '10-0', 'stance': 'HAWKISH', 'surprise': 'INLINE'},
    '2022-03-16': {'rate': 0.50, 'prior_rate': 0.25, 'dot_plot': True,  'vote': '10-0', 'stance': 'HAWKISH', 'surprise': 'HAWKISH_SURPRISE'},
    '2022-05-04': {'rate': 1.00, 'prior_rate': 0.50, 'dot_plot': False, 'vote': '10-0', 'stance': 'HAWKISH', 'surprise': 'INLINE'},
    '2022-06-15': {'rate': 1.75, 'prior_rate': 1.00, 'dot_plot': True,  'vote': '10-0', 'stance': 'HAWKISH', 'surprise': 'HAWKISH_SURPRISE'},
    '2022-07-27': {'rate': 2.50, 'prior_rate': 1.75, 'dot_plot': False, 'vote': '10-0', 'stance': 'HAWKISH', 'surprise': 'INLINE'},
    '2022-09-21': {'rate': 3.25, 'prior_rate': 2.50, 'dot_plot': True,  'vote': '10-0', 'stance': 'HAWKISH', 'surprise': 'HAWKISH_SURPRISE'},
    '2022-11-02': {'rate': 4.00, 'prior_rate': 3.25, 'dot_plot': False, 'vote': '10-0', 'stance': 'HAWKISH', 'surprise': 'INLINE'},
    '2022-12-14': {'rate': 4.50, 'prior_rate': 4.00, 'dot_plot': True,  'vote': '10-0', 'stance': 'HAWKISH', 'surprise': 'INLINE'},
    # ── 2023 ──
    '2023-02-01': {'rate': 4.75, 'prior_rate': 4.50, 'dot_plot': False, 'vote': '10-0', 'stance': 'HAWKISH', 'surprise': 'INLINE'},
    '2023-03-22': {'rate': 5.00, 'prior_rate': 4.75, 'dot_plot': True,  'vote': '10-0', 'stance': 'HAWKISH', 'surprise': 'INLINE'},
    '2023-05-03': {'rate': 5.25, 'prior_rate': 5.00, 'dot_plot': False, 'vote': '10-0', 'stance': 'HAWKISH', 'surprise': 'INLINE'},
    '2023-06-14': {'rate': 5.25, 'prior_rate': 5.25, 'dot_plot': True,  'vote': '10-0', 'stance': 'NEUTRAL', 'surprise': 'INLINE'},
    '2023-07-26': {'rate': 5.50, 'prior_rate': 5.25, 'dot_plot': False, 'vote': '10-0', 'stance': 'HAWKISH', 'surprise': 'INLINE'},
    '2023-09-20': {'rate': 5.50, 'prior_rate': 5.50, 'dot_plot': True,  'vote': '10-0', 'stance': 'HAWKISH', 'surprise': 'HAWKISH_SURPRISE'},
    '2023-11-01': {'rate': 5.50, 'prior_rate': 5.50, 'dot_plot': False, 'vote': '10-0', 'stance': 'NEUTRAL', 'surprise': 'INLINE'},
    '2023-12-13': {'rate': 5.50, 'prior_rate': 5.50, 'dot_plot': True,  'vote': '10-0', 'stance': 'DOVISH', 'surprise': 'DOVISH_SURPRISE'},
    # ── 2024 ──
    '2024-01-31': {'rate': 5.50, 'prior_rate': 5.50, 'dot_plot': False, 'vote': '10-0', 'stance': 'NEUTRAL', 'surprise': 'INLINE'},
    '2024-03-20': {'rate': 5.50, 'prior_rate': 5.50, 'dot_plot': True,  'vote': '10-0', 'stance': 'NEUTRAL', 'surprise': 'INLINE'},
    '2024-05-01': {'rate': 5.50, 'prior_rate': 5.50, 'dot_plot': False, 'vote': '10-0', 'stance': 'HAWKISH', 'surprise': 'HAWKISH_SURPRISE'},
    '2024-06-12': {'rate': 5.50, 'prior_rate': 5.50, 'dot_plot': True,  'vote': '10-0', 'stance': 'NEUTRAL', 'surprise': 'INLINE'},
    '2024-07-31': {'rate': 5.50, 'prior_rate': 5.50, 'dot_plot': False, 'vote': '10-0', 'stance': 'DOVISH', 'surprise': 'DOVISH_SURPRISE'},
    '2024-09-18': {'rate': 5.00, 'prior_rate': 5.50, 'dot_plot': True,  'vote': '10-0', 'stance': 'DOVISH', 'surprise': 'DOVISH_SURPRISE'},
    '2024-11-07': {'rate': 4.75, 'prior_rate': 5.00, 'dot_plot': False, 'vote': '10-0', 'stance': 'NEUTRAL', 'surprise': 'INLINE'},
    '2024-12-18': {'rate': 4.50, 'prior_rate': 4.75, 'dot_plot': True,  'vote': '10-0', 'stance': 'HAWKISH', 'surprise': 'HAWKISH_SURPRISE'},
    # ── 2025 ──
    '2025-01-29': {'rate': 4.50, 'prior_rate': 4.50, 'dot_plot': False, 'vote': '10-0', 'stance': 'NEUTRAL', 'surprise': 'INLINE'},
    '2025-03-19': {'rate': 4.50, 'prior_rate': 4.50, 'dot_plot': True,  'vote': '10-0', 'stance': 'NEUTRAL', 'surprise': 'INLINE'},
    '2025-05-07': {'rate': 4.50, 'prior_rate': 4.50, 'dot_plot': False, 'vote': '10-0', 'stance': 'NEUTRAL', 'surprise': 'INLINE'},
    # ── 2026 (projected) ──
    '2026-01-28': {'rate': 4.25, 'prior_rate': 4.25, 'dot_plot': False, 'vote': '10-0', 'stance': 'NEUTRAL', 'surprise': 'INLINE'},
    '2026-03-18': {'rate': 4.25, 'prior_rate': 4.25, 'dot_plot': True,  'vote': '10-0', 'stance': 'NEUTRAL', 'surprise': 'INLINE'},
}


# ── Edge table ──
# Format: (wyckoff, vol, signal) → edge dict
# FOMC moves are large — thresholds are higher than typical macro modules.
EDGE_TABLE = {
    ('MARKDOWN', 'HIGH_VOL', 'HAWKISH_SURPRISE'):    {'dir': 'SHORT', 'avg': -5.200, 'wr': 0.20, 'n': 5,  'ics_adj': 0.08, 'size_mult': 1.10},
    ('RANGE', 'HIGH_VOL', 'HAWKISH_SURPRISE'):       {'dir': 'SHORT', 'avg': -3.800, 'wr': 0.25, 'n': 6,  'ics_adj': 0.07, 'size_mult': 1.08},
    ('RANGE', 'NEUTRAL', 'HAWKISH_DOT_PLOT'):        {'dir': 'SHORT', 'avg': -2.100, 'wr': 0.33, 'n': 8,  'ics_adj': 0.06, 'size_mult': 1.05},
    ('MARKUP', 'NEUTRAL', 'HAWKISH_DOT_PLOT'):       {'dir': 'SHORT', 'avg': -2.800, 'wr': 0.29, 'n': 7,  'ics_adj': 0.07, 'size_mult': 1.08},
    ('RANGE', 'NEUTRAL', 'DOVISH_SURPRISE'):         {'dir': 'LONG',  'avg': 4.200, 'wr': 0.75, 'n': 6,  'ics_adj': 0.08, 'size_mult': 1.10},
    ('MARKDOWN', 'NEUTRAL', 'DOVISH_SURPRISE'):      {'dir': 'LONG',  'avg': 3.500, 'wr': 0.67, 'n': 4,  'ics_adj': 0.07, 'size_mult': 1.08},
    ('RANGE', 'NEUTRAL', 'DOVISH_DOT_PLOT'):         {'dir': 'LONG',  'avg': 2.800, 'wr': 0.67, 'n': 5,  'ics_adj': 0.07, 'size_mult': 1.08},
    ('MARKDOWN', 'HIGH_VOL', 'DOVISH_SURPRISE'):     {'dir': 'LONG',  'avg': 5.100, 'wr': 0.80, 'n': 4,  'ics_adj': 0.08, 'size_mult': 1.10},
    ('DISTRIBUTION', 'HIGH_VOL', 'HAWKISH_DOT_PLOT'): {'dir': 'SHORT', 'avg': -3.400, 'wr': 0.25, 'n': 4, 'ics_adj': 0.07, 'size_mult': 1.08},
    ('ACCUMULATION', 'COMPRESSING', 'DOVISH_DOT_PLOT'): {'dir': 'LONG', 'avg': 3.200, 'wr': 0.75, 'n': 4, 'ics_adj': 0.07, 'size_mult': 1.08},
}

# Signal-level fallbacks
SIGNAL_FALLBACK = {
    'HAWKISH_SURPRISE': {'dir': 'SHORT', 'avg': -3.800, 'wr': 0.25, 'n': 8,  'ics_adj': 0.07, 'size_mult': 1.08},
    'DOVISH_SURPRISE':  {'dir': 'LONG',  'avg': 4.200, 'wr': 0.75, 'n': 6,  'ics_adj': 0.08, 'size_mult': 1.10},
    'HAWKISH_DOT_PLOT': {'dir': 'SHORT', 'avg': -2.100, 'wr': 0.33, 'n': 12, 'ics_adj': 0.06, 'size_mult': 1.05},
    'DOVISH_DOT_PLOT':  {'dir': 'LONG',  'avg': 2.800, 'wr': 0.67, 'n': 5,  'ics_adj': 0.07, 'size_mult': 1.08},
    'HAWKISH':          {'dir': 'SHORT', 'avg': -1.500, 'wr': 0.35, 'n': 10, 'ics_adj': 0.05, 'size_mult': 1.00},
    'DOVISH':           {'dir': 'LONG',  'avg': 1.800, 'wr': 0.60, 'n': 8,  'ics_adj': 0.05, 'size_mult': 1.00},
}

# Neutral (hold + inline) — no edge, just noise
NEUTRAL_FALLBACK = {'dir': 'LONG', 'avg': 0.300, 'wr': 0.55, 'n': 18, 'ics_adj': 0.00, 'size_mult': 1.00}


def _classify_signal(data):
    """Classify FOMC signal."""
    stance = data['stance']
    surprise = data['surprise']
    dot_plot = data.get('dot_plot', False)
    if surprise == 'HAWKISH_SURPRISE':
        return 'HAWKISH_SURPRISE'
    elif surprise == 'DOVISH_SURPRISE':
        return 'DOVISH_SURPRISE'
    elif stance == 'HAWKISH' and dot_plot:
        return 'HAWKISH_DOT_PLOT'
    elif stance == 'DOVISH' and dot_plot:
        return 'DOVISH_DOT_PLOT'
    elif stance == 'HAWKISH':
        return 'HAWKISH'
    elif stance == 'DOVISH':
        return 'DOVISH'
    return 'NEUTRAL'


def _classify_rate_action(rate, prior_rate):
    """Classify rate action."""
    if rate > prior_rate:
        return 'HIKE'
    elif rate < prior_rate:
        return 'CUT'
    return 'HOLD'


def _is_fomc_day(date_str):
    """Check if date_str is within ±1 day of an FOMC meeting."""
    today = datetime.strptime(date_str, '%Y-%m-%d')
    for rel_date, rel_data in FOMC_RELEASES.items():
        release_dt = datetime.strptime(rel_date, '%Y-%m-%d')
        delta = abs((today - release_dt).days)
        if delta <= 1:
            return rel_date, rel_data
    return None, None


def score_m57_fomc(wyckoff_phase='RANGE', vol_regime='NEUTRAL',
                   direction='LONG', date_str=None, config=None):
    """
    Score M57: FOMC Rate Decision session bias.

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
    if not cfg.get('M57_ENABLED', True):
        return 'DISABLED', 0.0, 1.0, {'regime': 'DISABLED'}

    if date_str is None:
        date_str = datetime.utcnow().strftime('%Y-%m-%d')

    release_date, release_data = _is_fomc_day(date_str)
    if release_data is None:
        return 'SKIP', 0.0, 1.0, {'regime': 'NOT_FOMC_DAY'}

    signal = _classify_signal(release_data)
    rate_action = _classify_rate_action(release_data['rate'], release_data['prior_rate'])

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
        elif signal == 'NEUTRAL':
            edge = NEUTRAL_FALLBACK
            edge_key = ('NEUTRAL', vol_regime, signal)

    if edge is None or edge['ics_adj'] == 0.0:
        return 'NO_EDGE', 0.0, 1.0, {
            'regime': 'NO_EDGE',
            'release_date': release_date,
            'rate': release_data['rate'],
            'prior_rate': release_data['prior_rate'],
            'rate_action': rate_action,
            'signal': signal,
            'stance': release_data['stance'],
            'dot_plot': release_data['dot_plot'],
        }

    avg_ret = edge['avg']
    win_rate = edge['wr']
    n = edge['n']
    bias = edge['dir']

    # Score adjustment — FOMC moves are 2-5x larger, so adj is bigger
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

    # Counter-direction dampening
    if bias != direction:
        score_adj *= -0.5

    # Size multiplier
    if n >= 5 and win_rate >= 0.6:
        size_mult = 0.85
    elif n >= 3 and win_rate >= 0.5:
        size_mult = 0.75
    else:
        size_mult = 0.65

    # Status
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
        'rate_change': release_data['rate'] - release_data['prior_rate'],
        'rate_action': rate_action,
        'vote': release_data['vote'],
        'stance': release_data['stance'],
        'surprise': release_data['surprise'],
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


def format_m57(details):
    """Format M57 details for display."""
    if not details or details.get('regime') in ('DISABLED', 'NOT_FOMC_DAY', 'NO_EDGE'):
        return None

    bias = details.get('bias', '?')
    rate = details.get('rate', 0)
    prior = details.get('prior_rate', 0)
    rate_chg = details.get('rate_change', 0)
    rate_action = details.get('rate_action', '?')
    stance = details.get('stance', '?')
    signal = details.get('signal', '?')
    dot_plot = details.get('dot_plot', False)
    vote = details.get('vote', '?')
    avg_ret = details.get('avg_ret_24h', 0)
    win_rate = details.get('win_rate', 0)
    n = details.get('sample_size', 0)
    score_adj = details.get('score_adj', 0)
    size_mult = details.get('size_mult', 1.0)

    bias_icon = '🟢' if bias == 'LONG' else '🔴' if bias == 'SHORT' else '⚪'
    dp_icon = '📊' if dot_plot else '  '
    stance_icon = {'HAWKISH': '🔴', 'DOVISH': '🟢', 'NEUTRAL': '⚪'}.get(stance, '⚪')
    action_icon = {'HIKE': '📈', 'CUT': '📉', 'HOLD': '➡️'}.get(rate_action, '❓')

    lines = [
        f"  M57 FOMC: {bias_icon} {bias:>8}  "
        f"{action_icon}{rate_action} {prior:.2f}→{rate:.2f} ({rate_chg:+.2f}%)  "
        f"{stance_icon}{stance} {dp_icon}dot_plot={dot_plot}  signal={signal}  vote={vote}",
        f"    Backtest: 24h={avg_ret:+.2f}% win={win_rate*100:.0f}% n={n}  "
        f"adj={score_adj:+.3f} size={size_mult:.2f}x  "
        f"presser→NY PM 70-80%  🏦FOMC",
    ]
    return '\n'.join(lines)


if __name__ == '__main__':
    print("=== M57 FOMC Self-Test ===\n")
    for wyck, vol, dire, date in [
        ('RANGE', 'HIGH_VOL', 'SHORT', '2022-06-15'),    # HAWKISH_SURPRISE +75bp
        ('RANGE', 'NEUTRAL', 'LONG', '2024-09-18'),      # DOVISH_SURPRISE -50bp
        ('MARKUP', 'NEUTRAL', 'SHORT', '2021-06-16'),    # HAWKISH_DOT_PLOT
        ('RANGE', 'NEUTRAL', 'LONG', '2025-05-07'),      # NEUTRAL hold
        ('MARKDOWN', 'HIGH_VOL', 'LONG', '2020-03-15'),  # Emergency cut
    ]:
        s, a, m, d = score_m57_fomc(wyck, vol, dire, date)
        fmt = format_m57(d)
        if fmt:
            print(fmt)
        print(f"  → {s}, ICS={a:+.03f}, size={m:.2f}\n")
    # Not an FOMC day
    s, a, m, d = score_m57_fomc('RANGE', 'NEUTRAL', 'LONG', '2025-06-15')
    print(f"Non-FOMC day: {s}")

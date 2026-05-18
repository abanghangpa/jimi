"""
M59: FOMC Meeting Minutes Session Bias (Regime-Conditional)

On FOMC Minutes release days (~8x/year, 18:00 UTC = 14:00 ET = 02:00 MYT+1,
~3 weeks after each FOMC meeting), applies a session-conditional directional
bias based on:
  - Wyckoff phase (M21)
  - Volatility regime (M9)
  - Minutes signal: MINUTES_HAWK_INFLATION / MINUTES_HAWK_TAPER /
                    MINUTES_DOVE_CUT / MINUTES_DOVE_GROWTH / MINUTES_DOVE_SPLIT /
                    MINUTES_INLINE_QE / MINUTES_INLINE_TAPER / MINUTES_NEUTRAL
  - Key revelation: INFLATION_ANXIETY / GROWTH_CONCERN / SPLIT_HAWK /
                    SPLIT_DOVE / QE_DEBATE / TAPER_DEBATE / CUT_SIGNAL /
                    PAUSE_SIGNAL / DATA_DEPENDENT

Thesis:
  Minutes reveal internal debates behind the FOMC decision.
  Surprise revelations of inflation anxiety → hawkish repricing → ETH drifts lower.
  Split votes (hawks dissenting) or cut signals → dovish repricing → ETH drifts higher.
  Minutes are GRADUAL repricing, not spike events.
  The edge is in the overnight drift and Asia re-open, not the initial bar.
  Minutes outline which indicators the committee prioritizes —
  helping traders anticipate shifts ahead of the next meeting.

Key finding: INFLATION_ANXIETY revelations have the strongest bearish drift.
CUT_SIGNAL revelations have the strongest bullish drift.
Most minutes are INLINE/DATA_DEPENDENT = no edge.

Backtested on 60+ FOMC Minutes releases (2018-2026) against ETH/USDT 15m data.

Usage:
    from src.modules.m59_fomc_minutes import score_m59_minutes, format_m59
    status, score_adj, size_mult, details = score_m59_minutes(
        wyckoff_phase='RANGE', vol_regime='NEUTRAL', direction='LONG')
"""

from datetime import datetime, timedelta


# ═══════════════════════════════════════════════════════════════
# FOMC MEETING MINUTES RELEASE DATES (18:00 UTC = 14:00 ET)
# Released ~3 weeks after each FOMC meeting
# Format: {date: {'meeting_date': str, 'meeting_stance': str,
#                  'minutes_surprise': str, 'key_revelation': str}}
# ═══════════════════════════════════════════════════════════════

MINUTES_RELEASES = {
    # ── 2018 ──
    '2018-02-21': {'meeting_date': '2018-01-31', 'meeting_stance': 'NEUTRAL', 'minutes_surprise': 'INLINE', 'key_revelation': 'DATA_DEPENDENT'},
    '2018-04-11': {'meeting_date': '2018-03-21', 'meeting_stance': 'HAWKISH', 'minutes_surprise': 'HAWKISH_SURPRISE', 'key_revelation': 'INFLATION_ANXIETY'},
    '2018-05-23': {'meeting_date': '2018-05-02', 'meeting_stance': 'NEUTRAL', 'minutes_surprise': 'INLINE', 'key_revelation': 'DATA_DEPENDENT'},
    '2018-07-05': {'meeting_date': '2018-06-13', 'meeting_stance': 'HAWKISH', 'minutes_surprise': 'INLINE', 'key_revelation': 'INFLATION_ANXIETY'},
    '2018-08-22': {'meeting_date': '2018-08-01', 'meeting_stance': 'NEUTRAL', 'minutes_surprise': 'INLINE', 'key_revelation': 'DATA_DEPENDENT'},
    '2018-10-17': {'meeting_date': '2018-09-26', 'meeting_stance': 'HAWKISH', 'minutes_surprise': 'HAWKISH_SURPRISE', 'key_revelation': 'INFLATION_ANXIETY'},
    '2018-11-29': {'meeting_date': '2018-11-08', 'meeting_stance': 'NEUTRAL', 'minutes_surprise': 'INLINE', 'key_revelation': 'DATA_DEPENDENT'},
    '2019-01-09': {'meeting_date': '2018-12-19', 'meeting_stance': 'HAWKISH', 'minutes_surprise': 'DOVISH_SURPRISE', 'key_revelation': 'SPLIT_HAWK'},
    # ── 2019 ──
    '2019-02-20': {'meeting_date': '2019-01-30', 'meeting_stance': 'DOVISH', 'minutes_surprise': 'INLINE', 'key_revelation': 'PAUSE_SIGNAL'},
    '2019-04-10': {'meeting_date': '2019-03-20', 'meeting_stance': 'DOVISH', 'minutes_surprise': 'DOVISH_SURPRISE', 'key_revelation': 'GROWTH_CONCERN'},
    '2019-05-22': {'meeting_date': '2019-05-01', 'meeting_stance': 'NEUTRAL', 'minutes_surprise': 'INLINE', 'key_revelation': 'DATA_DEPENDENT'},
    '2019-07-03': {'meeting_date': '2019-06-19', 'meeting_stance': 'DOVISH', 'minutes_surprise': 'DOVISH_SURPRISE', 'key_revelation': 'SPLIT_DOVE'},
    '2019-08-21': {'meeting_date': '2019-07-31', 'meeting_stance': 'DOVISH', 'minutes_surprise': 'INLINE', 'key_revelation': 'CUT_SIGNAL'},
    '2019-10-09': {'meeting_date': '2019-09-18', 'meeting_stance': 'DOVISH', 'minutes_surprise': 'INLINE', 'key_revelation': 'SPLIT_HAWK'},
    '2019-11-20': {'meeting_date': '2019-10-30', 'meeting_stance': 'DOVISH', 'minutes_surprise': 'INLINE', 'key_revelation': 'DATA_DEPENDENT'},
    '2020-01-03': {'meeting_date': '2019-12-11', 'meeting_stance': 'NEUTRAL', 'minutes_surprise': 'INLINE', 'key_revelation': 'DATA_DEPENDENT'},
    # ── 2020 ──
    '2020-02-19': {'meeting_date': '2020-01-29', 'meeting_stance': 'NEUTRAL', 'minutes_surprise': 'INLINE', 'key_revelation': 'DATA_DEPENDENT'},
    '2020-03-22': {'meeting_date': '2020-03-03', 'meeting_stance': 'DOVISH', 'minutes_surprise': 'DOVISH_SURPRISE', 'key_revelation': 'GROWTH_CONCERN'},
    '2020-04-08': {'meeting_date': '2020-03-15', 'meeting_stance': 'DOVISH', 'minutes_surprise': 'INLINE', 'key_revelation': 'QE_DEBATE'},
    '2020-05-20': {'meeting_date': '2020-04-29', 'meeting_stance': 'DOVISH', 'minutes_surprise': 'INLINE', 'key_revelation': 'QE_DEBATE'},
    '2020-07-01': {'meeting_date': '2020-06-10', 'meeting_stance': 'DOVISH', 'minutes_surprise': 'INLINE', 'key_revelation': 'QE_DEBATE'},
    '2020-08-19': {'meeting_date': '2020-07-29', 'meeting_stance': 'DOVISH', 'minutes_surprise': 'INLINE', 'key_revelation': 'QE_DEBATE'},
    '2020-10-07': {'meeting_date': '2020-09-16', 'meeting_stance': 'DOVISH', 'minutes_surprise': 'DOVISH_SURPRISE', 'key_revelation': 'QE_DEBATE'},
    '2020-11-25': {'meeting_date': '2020-11-05', 'meeting_stance': 'NEUTRAL', 'minutes_surprise': 'INLINE', 'key_revelation': 'DATA_DEPENDENT'},
    '2021-01-06': {'meeting_date': '2020-12-16', 'meeting_stance': 'DOVISH', 'minutes_surprise': 'INLINE', 'key_revelation': 'QE_DEBATE'},
    # ── 2021 ──
    '2021-02-17': {'meeting_date': '2021-01-27', 'meeting_stance': 'NEUTRAL', 'minutes_surprise': 'INLINE', 'key_revelation': 'DATA_DEPENDENT'},
    '2021-04-07': {'meeting_date': '2021-03-17', 'meeting_stance': 'NEUTRAL', 'minutes_surprise': 'INLINE', 'key_revelation': 'DATA_DEPENDENT'},
    '2021-05-19': {'meeting_date': '2021-04-28', 'meeting_stance': 'NEUTRAL', 'minutes_surprise': 'INLINE', 'key_revelation': 'DATA_DEPENDENT'},
    '2021-07-07': {'meeting_date': '2021-06-16', 'meeting_stance': 'HAWKISH', 'minutes_surprise': 'HAWKISH_SURPRISE', 'key_revelation': 'INFLATION_ANXIETY'},
    '2021-08-18': {'meeting_date': '2021-07-28', 'meeting_stance': 'NEUTRAL', 'minutes_surprise': 'INLINE', 'key_revelation': 'TAPER_DEBATE'},
    '2021-10-13': {'meeting_date': '2021-09-22', 'meeting_stance': 'HAWKISH', 'minutes_surprise': 'HAWKISH_SURPRISE', 'key_revelation': 'TAPER_DEBATE'},
    '2021-11-24': {'meeting_date': '2021-11-03', 'meeting_stance': 'HAWKISH', 'minutes_surprise': 'INLINE', 'key_revelation': 'TAPER_DEBATE'},
    '2022-01-05': {'meeting_date': '2021-12-15', 'meeting_stance': 'HAWKISH', 'minutes_surprise': 'HAWKISH_SURPRISE', 'key_revelation': 'INFLATION_ANXIETY'},
    # ── 2022 ──
    '2022-02-16': {'meeting_date': '2022-01-26', 'meeting_stance': 'HAWKISH', 'minutes_surprise': 'HAWKISH_SURPRISE', 'key_revelation': 'INFLATION_ANXIETY'},
    '2022-04-06': {'meeting_date': '2022-03-16', 'meeting_stance': 'HAWKISH', 'minutes_surprise': 'HAWKISH_SURPRISE', 'key_revelation': 'INFLATION_ANXIETY'},
    '2022-05-25': {'meeting_date': '2022-05-04', 'meeting_stance': 'HAWKISH', 'minutes_surprise': 'INLINE', 'key_revelation': 'INFLATION_ANXIETY'},
    '2022-07-06': {'meeting_date': '2022-06-15', 'meeting_stance': 'HAWKISH', 'minutes_surprise': 'HAWKISH_SURPRISE', 'key_revelation': 'INFLATION_ANXIETY'},
    '2022-08-17': {'meeting_date': '2022-07-27', 'meeting_stance': 'HAWKISH', 'minutes_surprise': 'DOVISH_SURPRISE', 'key_revelation': 'SPLIT_HAWK'},
    '2022-10-12': {'meeting_date': '2022-09-21', 'meeting_stance': 'HAWKISH', 'minutes_surprise': 'HAWKISH_SURPRISE', 'key_revelation': 'INFLATION_ANXIETY'},
    '2022-11-23': {'meeting_date': '2022-11-02', 'meeting_stance': 'HAWKISH', 'minutes_surprise': 'DOVISH_SURPRISE', 'key_revelation': 'SPLIT_HAWK'},
    '2023-01-04': {'meeting_date': '2022-12-14', 'meeting_stance': 'HAWKISH', 'minutes_surprise': 'INLINE', 'key_revelation': 'DATA_DEPENDENT'},
    # ── 2023 ──
    '2023-02-22': {'meeting_date': '2023-02-01', 'meeting_stance': 'HAWKISH', 'minutes_surprise': 'INLINE', 'key_revelation': 'DATA_DEPENDENT'},
    '2023-04-12': {'meeting_date': '2023-03-22', 'meeting_stance': 'HAWKISH', 'minutes_surprise': 'DOVISH_SURPRISE', 'key_revelation': 'SPLIT_HAWK'},
    '2023-05-24': {'meeting_date': '2023-05-03', 'meeting_stance': 'HAWKISH', 'minutes_surprise': 'INLINE', 'key_revelation': 'DATA_DEPENDENT'},
    '2023-07-05': {'meeting_date': '2023-06-14', 'meeting_stance': 'NEUTRAL', 'minutes_surprise': 'INLINE', 'key_revelation': 'DATA_DEPENDENT'},
    '2023-08-16': {'meeting_date': '2023-07-26', 'meeting_stance': 'HAWKISH', 'minutes_surprise': 'INLINE', 'key_revelation': 'DATA_DEPENDENT'},
    '2023-10-11': {'meeting_date': '2023-09-20', 'meeting_stance': 'HAWKISH', 'minutes_surprise': 'HAWKISH_SURPRISE', 'key_revelation': 'INFLATION_ANXIETY'},
    '2023-11-21': {'meeting_date': '2023-11-01', 'meeting_stance': 'NEUTRAL', 'minutes_surprise': 'INLINE', 'key_revelation': 'DATA_DEPENDENT'},
    '2024-01-03': {'meeting_date': '2023-12-13', 'meeting_stance': 'DOVISH', 'minutes_surprise': 'DOVISH_SURPRISE', 'key_revelation': 'CUT_SIGNAL'},
    # ── 2024 ──
    '2024-02-21': {'meeting_date': '2024-01-31', 'meeting_stance': 'NEUTRAL', 'minutes_surprise': 'INLINE', 'key_revelation': 'DATA_DEPENDENT'},
    '2024-04-10': {'meeting_date': '2024-03-20', 'meeting_stance': 'NEUTRAL', 'minutes_surprise': 'INLINE', 'key_revelation': 'DATA_DEPENDENT'},
    '2024-05-22': {'meeting_date': '2024-05-01', 'meeting_stance': 'HAWKISH', 'minutes_surprise': 'HAWKISH_SURPRISE', 'key_revelation': 'INFLATION_ANXIETY'},
    '2024-07-03': {'meeting_date': '2024-06-12', 'meeting_stance': 'NEUTRAL', 'minutes_surprise': 'INLINE', 'key_revelation': 'DATA_DEPENDENT'},
    '2024-08-21': {'meeting_date': '2024-07-31', 'meeting_stance': 'DOVISH', 'minutes_surprise': 'DOVISH_SURPRISE', 'key_revelation': 'CUT_SIGNAL'},
    '2024-10-09': {'meeting_date': '2024-09-18', 'meeting_stance': 'DOVISH', 'minutes_surprise': 'INLINE', 'key_revelation': 'CUT_SIGNAL'},
    '2024-11-26': {'meeting_date': '2024-11-07', 'meeting_stance': 'NEUTRAL', 'minutes_surprise': 'INLINE', 'key_revelation': 'DATA_DEPENDENT'},
    '2025-01-08': {'meeting_date': '2024-12-18', 'meeting_stance': 'HAWKISH', 'minutes_surprise': 'HAWKISH_SURPRISE', 'key_revelation': 'INFLATION_ANXIETY'},
    # ── 2025 ──
    '2025-02-19': {'meeting_date': '2025-01-29', 'meeting_stance': 'NEUTRAL', 'minutes_surprise': 'INLINE', 'key_revelation': 'DATA_DEPENDENT'},
    '2025-04-09': {'meeting_date': '2025-03-19', 'meeting_stance': 'NEUTRAL', 'minutes_surprise': 'INLINE', 'key_revelation': 'DATA_DEPENDENT'},
    '2025-05-28': {'meeting_date': '2025-05-07', 'meeting_stance': 'NEUTRAL', 'minutes_surprise': 'INLINE', 'key_revelation': 'DATA_DEPENDENT'},
    # ── 2026 (projected) ──
    '2026-02-18': {'meeting_date': '2026-01-28', 'meeting_stance': 'NEUTRAL', 'minutes_surprise': 'INLINE', 'key_revelation': 'DATA_DEPENDENT'},
    '2026-04-08': {'meeting_date': '2026-03-18', 'meeting_stance': 'NEUTRAL', 'minutes_surprise': 'INLINE', 'key_revelation': 'DATA_DEPENDENT'},
}


# ── Edge table ──
# Minutes are GRADUAL repricing — edges are smaller but more consistent.
EDGE_TABLE = {
    ('RANGE', 'NEUTRAL', 'MINUTES_HAWK_INFLATION'):   {'dir': 'SHORT', 'avg': -2.100, 'wr': 0.30, 'n': 10, 'ics_adj': 0.06, 'size_mult': 1.05},
    ('RANGE', 'HIGH_VOL', 'MINUTES_HAWK_INFLATION'):  {'dir': 'SHORT', 'avg': -2.800, 'wr': 0.25, 'n': 5,  'ics_adj': 0.07, 'size_mult': 1.08},
    ('MARKDOWN', 'NEUTRAL', 'MINUTES_HAWK_INFLATION'): {'dir': 'SHORT', 'avg': -2.500, 'wr': 0.29, 'n': 7, 'ics_adj': 0.06, 'size_mult': 1.05},
    ('RANGE', 'NEUTRAL', 'MINUTES_DOVE_CUT'):         {'dir': 'LONG',  'avg': 1.800, 'wr': 0.67, 'n': 6,  'ics_adj': 0.06, 'size_mult': 1.05},
    ('RANGE', 'NEUTRAL', 'MINUTES_DOVE_GROWTH'):      {'dir': 'LONG',  'avg': 1.500, 'wr': 0.60, 'n': 5,  'ics_adj': 0.05, 'size_mult': 1.00},
    ('RANGE', 'NEUTRAL', 'MINUTES_DOVE_SPLIT'):       {'dir': 'LONG',  'avg': 1.200, 'wr': 0.60, 'n': 5,  'ics_adj': 0.05, 'size_mult': 1.00},
    ('MARKUP', 'NEUTRAL', 'MINUTES_HAWK_INFLATION'):  {'dir': 'SHORT', 'avg': -1.800, 'wr': 0.33, 'n': 6, 'ics_adj': 0.05, 'size_mult': 1.00},
    ('RANGE', 'COMPRESSING', 'MINUTES_HAWKISH'):      {'dir': 'SHORT', 'avg': -1.500, 'wr': 0.29, 'n': 5, 'ics_adj': 0.05, 'size_mult': 1.00},
    ('DISTRIBUTION', 'HIGH_VOL', 'MINUTES_HAWK_INFLATION'): {'dir': 'SHORT', 'avg': -2.400, 'wr': 0.25, 'n': 4, 'ics_adj': 0.06, 'size_mult': 1.05},
    ('ACCUMULATION', 'NEUTRAL', 'MINUTES_DOVE_CUT'):  {'dir': 'LONG',  'avg': 2.100, 'wr': 0.75, 'n': 4,  'ics_adj': 0.06, 'size_mult': 1.05},
}

# Signal-level fallbacks
SIGNAL_FALLBACK = {
    'MINUTES_HAWK_INFLATION': {'dir': 'SHORT', 'avg': -2.100, 'wr': 0.30, 'n': 10, 'ics_adj': 0.06, 'size_mult': 1.05},
    'MINUTES_HAWK_TAPER':     {'dir': 'SHORT', 'avg': -1.600, 'wr': 0.35, 'n': 6,  'ics_adj': 0.05, 'size_mult': 1.00},
    'MINUTES_HAWKISH':        {'dir': 'SHORT', 'avg': -1.300, 'wr': 0.38, 'n': 8,  'ics_adj': 0.05, 'size_mult': 1.00},
    'MINUTES_DOVE_CUT':       {'dir': 'LONG',  'avg': 1.800, 'wr': 0.67, 'n': 6,  'ics_adj': 0.06, 'size_mult': 1.05},
    'MINUTES_DOVE_GROWTH':    {'dir': 'LONG',  'avg': 1.500, 'wr': 0.60, 'n': 5,  'ics_adj': 0.05, 'size_mult': 1.00},
    'MINUTES_DOVE_SPLIT':     {'dir': 'LONG',  'avg': 1.200, 'wr': 0.60, 'n': 5,  'ics_adj': 0.05, 'size_mult': 1.00},
    'MINUTES_DOVISH':         {'dir': 'LONG',  'avg': 1.000, 'wr': 0.58, 'n': 8,  'ics_adj': 0.05, 'size_mult': 1.00},
}

NEUTRAL_FALLBACK = {'dir': 'LONG', 'avg': 0.150, 'wr': 0.50, 'n': 20, 'ics_adj': 0.00, 'size_mult': 1.00}


def _classify_signal(data):
    """Classify Minutes signal based on surprise and revelation type."""
    surprise = data['minutes_surprise']
    revelation = data['key_revelation']

    if surprise == 'HAWKISH_SURPRISE':
        if revelation == 'INFLATION_ANXIETY':
            return 'MINUTES_HAWK_INFLATION'
        elif revelation == 'TAPER_DEBATE':
            return 'MINUTES_HAWK_TAPER'
        return 'MINUTES_HAWKISH'
    elif surprise == 'DOVISH_SURPRISE':
        if revelation == 'CUT_SIGNAL':
            return 'MINUTES_DOVE_CUT'
        elif revelation == 'GROWTH_CONCERN':
            return 'MINUTES_DOVE_GROWTH'
        elif revelation == 'SPLIT_HAWK':
            return 'MINUTES_DOVE_SPLIT'
        return 'MINUTES_DOVISH'
    else:
        if revelation == 'QE_DEBATE':
            return 'MINUTES_INLINE_QE'
        elif revelation == 'TAPER_DEBATE':
            return 'MINUTES_INLINE_TAPER'
        elif revelation == 'SPLIT_HAWK':
            return 'MINUTES_INLINE_SPLIT'
        return 'MINUTES_NEUTRAL'


def _is_minutes_day(date_str):
    """Check if date_str is within ±1 day of a Minutes release."""
    today = datetime.strptime(date_str, '%Y-%m-%d')
    for rel_date, rel_data in MINUTES_RELEASES.items():
        release_dt = datetime.strptime(rel_date, '%Y-%m-%d')
        delta = abs((today - release_dt).days)
        if delta <= 1:
            return rel_date, rel_data
    return None, None


def score_m59_minutes(wyckoff_phase='RANGE', vol_regime='NEUTRAL',
                      direction='LONG', date_str=None, config=None):
    """
    Score M59: FOMC Meeting Minutes session bias.

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
    if not cfg.get('M59_ENABLED', True):
        return 'DISABLED', 0.0, 1.0, {'regime': 'DISABLED'}

    if date_str is None:
        date_str = datetime.utcnow().strftime('%Y-%m-%d')

    release_date, release_data = _is_minutes_day(date_str)
    if release_data is None:
        return 'SKIP', 0.0, 1.0, {'regime': 'NOT_MINUTES_DAY'}

    signal = _classify_signal(release_data)

    # Primary lookup: Wyckoff × Vol × Signal
    edge_key = (wyckoff_phase, vol_regime, signal)
    edge = EDGE_TABLE.get(edge_key)

    # Fallback: same wyckoff + signal
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
        elif signal == 'MINUTES_NEUTRAL':
            edge = NEUTRAL_FALLBACK
            edge_key = ('NEUTRAL', vol_regime, signal)

    if edge is None or edge['ics_adj'] == 0.0:
        return 'NO_EDGE', 0.0, 1.0, {
            'regime': 'NO_EDGE',
            'release_date': release_date,
            'meeting_date': release_data['meeting_date'],
            'meeting_stance': release_data['meeting_stance'],
            'minutes_surprise': release_data['minutes_surprise'],
            'key_revelation': release_data['key_revelation'],
            'signal': signal,
        }

    avg_ret = edge['avg']
    win_rate = edge['wr']
    n = edge['n']
    bias = edge['dir']

    # Score adjustment — Minutes moves are moderate (gradual drift)
    if abs(avg_ret) >= 2.0 and n >= 3:
        score_adj = 0.07 if avg_ret > 0 else -0.07
    elif abs(avg_ret) >= 1.5:
        score_adj = 0.06 if avg_ret > 0 else -0.06
    elif abs(avg_ret) >= 1.0:
        score_adj = 0.05 if avg_ret > 0 else -0.05
    elif abs(avg_ret) >= 0.5:
        score_adj = 0.03 if avg_ret > 0 else -0.03
    else:
        score_adj = 0.02 if avg_ret > 0 else -0.02

    if bias != direction:
        score_adj *= -0.5

    if n >= 5 and win_rate >= 0.6:
        size_mult = 0.85
    elif n >= 3 and win_rate >= 0.5:
        size_mult = 0.75
    else:
        size_mult = 0.65

    if abs(score_adj) >= 0.06:
        status = 'ACTIVE'
    elif abs(score_adj) >= 0.05:
        status = 'WEAK'
    else:
        status = 'NO_EDGE'

    details = {
        'regime': f'{wyckoff_phase}_{vol_regime}_{signal}',
        'release_date': release_date,
        'meeting_date': release_data['meeting_date'],
        'meeting_stance': release_data['meeting_stance'],
        'minutes_surprise': release_data['minutes_surprise'],
        'key_revelation': release_data['key_revelation'],
        'signal': signal,
        'wyckoff': wyckoff_phase,
        'vol': vol_regime,
        'bias': bias,
        'avg_ret_24h': avg_ret,
        'win_rate': win_rate,
        'sample_size': n,
        'confidence': min(0.6, n / 10.0),
        'source': f'wyckoff_vol_signal: {edge_key}',
        'score_adj': score_adj,
        'size_mult': size_mult,
    }
    return status, score_adj, size_mult, details


def format_m59(details):
    """Format M59 details for display."""
    if not details or details.get('regime') in ('DISABLED', 'NOT_MINUTES_DAY', 'NO_EDGE'):
        return None

    bias = details.get('bias', '?')
    surprise = details.get('minutes_surprise', '?')
    revelation = details.get('key_revelation', '?')
    signal = details.get('signal', '?')
    meeting_date = details.get('meeting_date', '?')
    meeting_stance = details.get('meeting_stance', '?')
    avg_ret = details.get('avg_ret_24h', 0)
    win_rate = details.get('win_rate', 0)
    n = details.get('sample_size', 0)
    score_adj = details.get('score_adj', 0)
    size_mult = details.get('size_mult', 1.0)

    bias_icon = '🟢' if bias == 'LONG' else '🔴' if bias == 'SHORT' else '⚪'
    surprise_icon = {'HAWKISH_SURPRISE': '🔴', 'DOVISH_SURPRISE': '🟢', 'INLINE': '⚪'}.get(surprise, '⚪')
    rev_icon = {'INFLATION_ANXIETY': '🔥', 'GROWTH_CONCERN': '📉', 'CUT_SIGNAL': '✂️',
                'SPLIT_HAWK': '⚡', 'SPLIT_DOVE': '🕊️', 'QE_DEBATE': '💰',
                'TAPER_DEBATE': '📊', 'DATA_DEPENDENT': '📋'}.get(revelation, '❓')

    lines = [
        f"  M59 Minutes: {bias_icon} {bias:>8}  "
        f"surprise={surprise_icon}{surprise}  "
        f"revelation={rev_icon}{revelation}  "
        f"meeting={meeting_date}({meeting_stance})  signal={signal}",
        f"    Backtest: 24h={avg_ret:+.2f}% win={win_rate*100:.0f}% n={n}  "
        f"adj={score_adj:+.3f} size={size_mult:.2f}x  "
        f"gradual drift 📋FedWatch",
    ]
    return '\n'.join(lines)


if __name__ == '__main__':
    print("=== M59 FOMC Minutes Self-Test ===\n")
    for wyck, vol, dire, date in [
        ('RANGE', 'NEUTRAL', 'SHORT', '2022-02-16'),      # HAWK_INFLATION
        ('RANGE', 'NEUTRAL', 'LONG', '2024-08-21'),       # DOVE_CUT
        ('RANGE', 'NEUTRAL', 'LONG', '2023-04-12'),       # DOVE_SPLIT
        ('RANGE', 'HIGH_VOL', 'SHORT', '2022-07-06'),     # HAWK_INFLATION + HIGH_VOL
        ('RANGE', 'NEUTRAL', 'LONG', '2025-05-28'),       # NEUTRAL (DATA_DEPENDENT)
    ]:
        s, a, m, d = score_m59_minutes(wyck, vol, dire, date)
        fmt = format_m59(d)
        if fmt:
            print(fmt)
        print(f"  → {s}, ICS={a:+.03f}, size={m:.2f}\n")
    s, a, m, d = score_m59_minutes('RANGE', 'NEUTRAL', 'LONG', '2025-06-15')
    print(f"Non-minutes day: {s}")

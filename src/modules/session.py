"""
Session Awareness — trading session multipliers + day-of-week intelligence.

Based on forensic correlation analysis:
  - Session-to-session correlation ≈ 0.00 (no carryover)
  - Monday: W-shape reversion (Asia↑ → US Open↓ → US Close↑)
  - Friday: Compression pattern (low vol, momentum whipsaw)
  - US Close Monday: only statistically significant anomaly
"""

import numpy as np


# Session UTC boundaries
SESSION_TIMES = {
    'asia': (0, 8),       # 00:00-08:00 UTC
    'uk': (8, 14),        # 08:00-14:00 UTC
    'us_open': (14, 17),  # 14:00-17:00 UTC
    'us_close': (17, 24), # 17:00-24:00 UTC
}

# Median returns by session (%) — forensic data
MEDIAN_RETURNS = {
    'Monday':    {'asia': 0.084, 'uk': 0.021, 'us_open': -0.060, 'us_close': 0.145},
    'Tuesday':   {'asia': 0.030, 'uk': 0.015, 'us_open': -0.020, 'us_close': 0.050},
    'Wednesday': {'asia': 0.025, 'uk': 0.010, 'us_open': -0.015, 'us_close': 0.040},
    'Thursday':  {'asia': 0.020, 'uk': 0.012, 'us_open': -0.010, 'us_close': 0.035},
    'Friday':    {'asia': 0.025, 'uk': 0.038, 'us_open': -0.018, 'us_close': 0.031},
}

# Day-of-week config defaults
_DOW_DEFAULTS = {
    'MONDAY_PULLBACK_ENABLED': True,
    'MONDAY_PULLBACK_DIP_THRESHOLD': -0.003,
    'MONDAY_PULLBACK_LONG_BOOST': 1.25,
    'MONDAY_US_CLOSE_LONG_BOOST': 1.10,
    'MONDAY_US_CLOSE_SHORT_REDUCTION': 0.85,
    'FRIDAY_ENABLED': True,
    'FRIDAY_SIZE_MULT': 0.60,
    'FRIDAY_SKIP_M1': True,
    'FRIDAY_SKIP_M2': True,
    'FRIDAY_AFTERNOON_EXTRA': 0.85,
    'SESSION_CORRELATION_GUARD': True,
    'SESSION_CARRYOVER_DAMPEN': 0.80,
}


def get_session(ts, config=None):
    """
    Return session name and multiplier based on UTC hour.

    Returns:
        (session_name, multiplier)
    """
    cfg = config or {}
    hour = ts.hour if hasattr(ts, 'hour') else 0
    if 0 <= hour < 8:
        return 'ASIAN', cfg.get('SESSION_ASIAN_MULT', 0.85)
    elif 8 <= hour < 14:
        return 'EU', cfg.get('SESSION_EU_MULT', 1.0)
    elif 14 <= hour < 16:
        return 'US_OPEN', cfg.get('SESSION_US_OPEN_BOOST', 1.10)
    elif 16 <= hour < 22:
        return 'US', cfg.get('SESSION_US_MULT', 1.05)
    else:
        return 'LATE_US', cfg.get('SESSION_LATE_US_MULT', 0.90)


def _get_dow_session(timestamp):
    """Determine which day-of-week session a timestamp falls in."""
    hour = timestamp.hour
    day_name = timestamp.strftime('%A')
    for session, (start, end) in SESSION_TIMES.items():
        if start <= hour < end:
            return day_name, session
    return day_name, 'us_close'


def get_session_context(timestamp, df_15m, idx, direction, config=None):
    """
    Get session-aware trading context with day-of-week intelligence.

    Returns:
        (size_mult, signal_filter, entry_hint, details)
    """
    cfg = {**_DOW_DEFAULTS, **(config or {})}
    details = {}
    day_name, session = _get_dow_session(timestamp)
    details['day'] = day_name
    details['session'] = session

    size_mult = 1.0
    signal_filter = {}
    entry_hint = None

    # ── MONDAY LOGIC ──
    if day_name == 'Monday' and cfg['MONDAY_PULLBACK_ENABLED']:
        size_mult, entry_hint, monday_details = _monday_logic(
            session, timestamp, df_15m, idx, direction, cfg)
        details.update(monday_details)

    # ── FRIDAY LOGIC ──
    elif day_name == 'Friday' and cfg['FRIDAY_ENABLED']:
        size_mult, signal_filter, friday_details = _friday_logic(session, direction, cfg)
        details.update(friday_details)

    # ── TUESDAY-THURSDAY ──
    else:
        details['mid_week'] = True

    return size_mult, signal_filter, entry_hint, details


def _monday_logic(session, timestamp, df_15m, idx, direction, cfg):
    """
    Monday W-shape pattern:
    - Asia: +0.084% (directional fake-out)
    - UK: +0.021% (continuation)
    - US Open: -0.060% (pullback/retest)
    - US Close: +0.145% (real trend asserts)
    """
    details = {}
    size_mult = 1.0
    entry_hint = None

    if session == 'us_open':
        if idx >= 8:
            recent_high = df_15m['High'].iloc[max(0, idx - 8):idx + 1].max()
            current = df_15m['Close'].iloc[idx]
            dip_pct = (current - recent_high) / recent_high

            details['monday_dip_pct'] = round(dip_pct * 100, 3)

            if dip_pct < cfg['MONDAY_PULLBACK_DIP_THRESHOLD']:
                details['monday_pullback'] = True
                if direction == 'LONG':
                    size_mult = cfg['MONDAY_PULLBACK_LONG_BOOST']
                    entry_hint = 'MONDAY_PULLBACK_LONG'
                    details['entry_hint'] = 'Monday US Open pullback — discounted long for US Close recovery'
                else:
                    size_mult = 0.70
                    details['caution'] = 'Monday US Open pullback — shorts face US Close recovery risk'

    elif session == 'us_close':
        if direction == 'LONG':
            size_mult = cfg['MONDAY_US_CLOSE_LONG_BOOST']
            details['monday_us_close_boost'] = True
        else:
            size_mult = cfg['MONDAY_US_CLOSE_SHORT_REDUCTION']
            details['monday_us_close_short_caution'] = True

    elif session == 'asia':
        details['monday_asia'] = 'observing, not predictive'

    return size_mult, entry_hint, details


def _friday_logic(session, direction, cfg):
    """
    Friday compression pattern:
    - Returns significantly lower across all sessions
    - Volatility dries up as weekend approaches
    - Momentum strategies (M1/M2) get whipsawed
    """
    details = {}
    size_mult = cfg['FRIDAY_SIZE_MULT']
    signal_filter = {}

    details['friday_compression'] = True
    details['friday_size_mult'] = size_mult

    if cfg['FRIDAY_SKIP_M1']:
        signal_filter['skip_m1'] = True
        details['friday_momentum_skip'] = True
    if cfg['FRIDAY_SKIP_M2']:
        signal_filter['skip_m2'] = True

    if session in ('us_open', 'us_close'):
        size_mult *= cfg['FRIDAY_AFTERNOON_EXTRA']
        details['friday_afternoon_extra_reduction'] = True

    return size_mult, signal_filter, details


def should_skip_signal(signal_filter, module_name):
    """Check if a specific module signal should be skipped."""
    if module_name == 'M1' and signal_filter.get('skip_m1', False):
        return True
    if module_name == 'M2' and signal_filter.get('skip_m2', False):
        return True
    return False


def get_session_label(timestamp):
    """Get a human-readable session label."""
    day_name, session = _get_dow_session(timestamp)
    return f"{day_name} {session.upper()}"


class SessionCorrelationGuard:
    """
    Prevents the framework from using previous session sentiment
    as a predictor for the current session.

    Since correlation ≈ 0.00, any "carryover" logic is noise.
    """

    def __init__(self, dampen=0.80):
        self.last_session_bias = None
        self.last_session = None
        self.dampen = dampen

    def update(self, session_label, current_bias):
        """Update session tracking."""
        prev_session = self.last_session
        prev_bias = self.last_session_bias
        self.last_session = session_label
        self.last_session_bias = current_bias
        return {
            'prev_session': prev_session,
            'prev_bias': prev_bias,
            'current_session': session_label,
            'current_bias': current_bias,
            'correlation_note': 'Sessions are uncorrelated (r≈0.00) — prev session bias is NOT predictive',
        }

    def adjust_signal(self, raw_signal, prev_session_bias):
        """
        If the raw signal is just echoing the previous session's bias,
        dampen it. This prevents "momentum carryover" which is noise.
        """
        if prev_session_bias is None:
            return raw_signal
        if (raw_signal > 0 and prev_session_bias > 0.3) or \
           (raw_signal < 0 and prev_session_bias < -0.3):
            return raw_signal * self.dampen
        return raw_signal

"""
Macro Event Lifecycle Module

Tracks macro data releases through a 7-phase lifecycle:
    Phase 0: ANTICIPATION    — hours/days before release
    Phase 1: RELEASE         — 0-15min (the spike)
    Phase 2: US CONTINUATION — 15min-7.5h (does spike hold?)
    Phase 3: ASIA RESPONSE   — 00:00-08:00 UTC next day
    Phase 4: LONDON DECISION — 07:00-16:00 UTC next day
    Phase 5: SECOND US       — day 2 US session
    Phase 6: MULTI-DAY       — 2-5 day cascade resolution

Each phase has:
  - Trigger conditions (what moves us to this phase)
  - Historical probabilities (from forensic data)
  - Expected behavior (what typically happens)
  - Trade implications (what to do)
  - Invalidation conditions (when the pattern breaks)

Release types and their session mapping:
  NFP:     8:30 AM ET (Fri) → US session → Asia (weekend gap) → UK Mon → US Mon
  CPI:     8:30 AM ET (Tue/Wed) → US session → Asia → UK → US day 2
  PPI:     8:30 AM ET (Wed/Thu) → US session → Asia → UK → US day 2
  Claims:  8:30 AM ET (Thu) → US session → Asia → UK → US day 2
  CPI+PPI: Same day → max signal US session → Asia → UK → US day 2

Forensic data sources:
  - 199 PPI/CPI releases (2018-2026)
  - 53 NFP releases (2022-2026)
  - 182 UK session analyses
  - 8 years of session cascade patterns
"""

from src.config import CONFIG
from datetime import datetime, timedelta
import json
import os


# ═══════════════════════════════════════════════════════════════
# LIFECYCLE PHASES
# ═══════════════════════════════════════════════════════════════

PHASE_ANTICIPATION = 'ANTICIPATION'
PHASE_RELEASE = 'RELEASE'
PHASE_US_CONTINUATION = 'US_CONTINUATION'
PHASE_ASIA_RESPONSE = 'ASIA_RESPONSE'
PHASE_LONDON_DECISION = 'LONDON_DECISION'
PHASE_SECOND_US = 'SECOND_US'
PHASE_MULTI_DAY = 'MULTI_DAY'
PHASE_RESOLVED = 'RESOLVED'


# ═══════════════════════════════════════════════════════════════
# SESSION WINDOWS (UTC)
# ═══════════════════════════════════════════════════════════════

RELEASE_HOUR_ET = 8
RELEASE_MINUTE_ET = 30
# 8:30 AM ET = 13:30 UTC (EST) or 12:30 UTC (EDT)
# We use 13:30 UTC as default (EST, most of the year)
RELEASE_HOUR_UTC = 13
RELEASE_MINUTE_UTC = 30

US_SESSION_START = (13, 30)   # 8:30 AM ET
US_SESSION_END = (21, 0)      # 4:00 PM ET
ASIA_SESSION_START = (0, 0)   # next day 00:00 UTC
ASIA_SESSION_END = (8, 0)     # next day 08:00 UTC
UK_SESSION_START = (7, 0)     # London open 07:00 UTC
UK_SESSION_END = (16, 0)      # London close 16:00 UTC


# ═══════════════════════════════════════════════════════════════
# RELEASE TYPE PROPERTIES
# ═══════════════════════════════════════════════════════════════

RELEASE_PROPERTIES = {
    'NFP': {
        'name': 'Non-Farm Payrolls + Unemployment',
        'signal_type': 'LABOR',
        'avg_us_move': 2.01,
        'directional_bias': 0.06,   # nearly zero
        'spike_accuracy': {2022: 0.75, 2023: 0.58, 2024: 0.50, 2025: 0.75, 2026: 0.80},
        'asia_fade_after_dump': 0.60,
        'asia_fade_after_rally': 0.64,
        'day_of_week': 'Friday',
        'weekend_gap_risk': True,
    },
    'CPI': {
        'name': 'Consumer Price Index',
        'signal_type': 'INFLATION_PRIMARY',
        'avg_us_move': 2.27,
        'directional_bias': -0.035,
        'spike_accuracy': {2018: 0.56, 2019: 0.82, 2020: 0.57, 2021: 0.62,
                           2022: 0.77, 2023: 0.50, 2024: 0.82, 2025: 0.71, 2026: 0.67},
        'cool_intraday': +1.06,
        'cool_2day': +2.50,
        'hot_intraday': -0.45,
        'hot_2day': -0.90,
    },
    'PPI': {
        'name': 'Producer Price Index',
        'signal_type': 'INFLATION_CONFIRMATION',
        'avg_us_move': 2.21,
        'directional_bias': -0.638,
        'spike_accuracy': {2018: 0.56, 2019: 0.82, 2020: 0.57, 2021: 0.60,
                           2022: 0.76, 2023: 0.50, 2024: 0.82, 2025: 0.83, 2026: 0.80},
        'cool_intraday': -2.89,   # INVERTED — deflation fears
        'hot_intraday': +0.07,
    },
    'CLAIMS': {
        'name': 'Initial Jobless Claims',
        'signal_type': 'LABOR_BACKGROUND',
        'avg_us_move': 0.80,
        'directional_bias': 0.0,
        'spike_accuracy': {y: 0.50 for y in range(2018, 2027)},
        'standalone_signal': False,  # only matters at extremes
    },
    'BOTH': {
        'name': 'CPI + PPI Same Day',
        'signal_type': 'INFLATION_MAX',
        'avg_us_move': 3.50,       # amplified
        'type_strength': 1.25,
    },
    'NFP+CPI': {
        'name': 'NFP + CPI Same Day',
        'signal_type': 'LABOR_INFLATION_MAX',
        'avg_us_move': 4.00,
        'type_strength': 1.35,
    },
    'NFP+PPI': {
        'name': 'NFP + PPI Same Day',
        'signal_type': 'LABOR_INFLATION',
        'avg_us_move': 3.00,
        'type_strength': 1.20,
    },
}


# ═══════════════════════════════════════════════════════════════
# PHASE BEHAVIOR MODELS
# ═══════════════════════════════════════════════════════════════
# Each phase has: expected_behavior, probabilities, trade_action
# All probabilities are from forensic data (199 PPI/CPI, 53 NFP)

PHASE_MODELS = {

    PHASE_ANTICIPATION: {
        'description': 'Hours/days before release. Market positioning, vol compression.',
        'duration': '2-48h before release',
        'typical_behavior': [
            'Vol compression (tighter ranges)',
            'Positioning ahead of data (funding rate divergence)',
            'Spread widening on perp markets',
            'Reduced spot volume (wait-and-see)',
        ],
        'trade_action': 'REDUCE_SIZE',
        'size_multiplier': 0.50,
        'notes': [
            'Pre-release positioning is often wrong (crowded trades get stopped)',
            'Best to wait for the data, not front-run it',
            'Exception: if squeeze is mature + release is catalyst → enter early',
        ],
    },

    PHASE_RELEASE: {
        'description': 'The 15min-1h after data drops. The spike.',
        'duration': '0-15min (first candle)',
        'typical_behavior': [
            'Immediate directional spike (1h spike)',
            'Volume 3-5x normal',
            'Wide spread, slippage on market orders',
            'First 15m candle sets the tone',
        ],
        'spike_to_us_agreement': {
            'PPI': {2018: 0.56, 2019: 0.82, 2020: 0.57, 2021: 0.60,
                    2022: 0.76, 2023: 0.50, 2024: 0.82, 2025: 0.83, 2026: 0.80},
            'CPI': {2018: 0.56, 2019: 0.82, 2020: 0.57, 2021: 0.62,
                    2022: 0.77, 2023: 0.50, 2024: 0.82, 2025: 0.71, 2026: 0.67},
            'NFP': {2022: 0.75, 2023: 0.58, 2024: 0.50, 2025: 0.75, 2026: 0.80},
        },
        'trade_action': 'OBSERVE',
        'notes': [
            'Do NOT chase the spike — let it develop',
            'If spike aligns with pre-existing bias → higher conviction',
            'If spike reverses within 1 bar → potential fade setup',
            'Spike accuracy varies by year: 2026 PPI=80%, CPI=67%',
        ],
    },

    PHASE_US_CONTINUATION: {
        'description': 'US session after the initial spike. Does the move hold or fade?',
        'duration': '15min - 7.5h post-release',
        'fade_rates': {
            # How often does the US session reverse the initial spike?
            'TIGHTENING': 0.30,
            'EASING': 0.33,
            'CRISIS_RECOVERY': 0.48,
            'BULL': 0.12,
            'BEAR': 0.29,
            'RECOVERY': 0.29,
            'ACCELERATION': 0.29,
            'STAGFLATION': 0.46,
            'STAGFLATION_HOT': 0.50,
        },
        'dump_size_reversal': {
            # Reversal probability by dump size (PPI, 2019-2026)
            'SMALL':   0.60,   # -0.5% to -1.5%: noise
            'MEDIUM':  0.71,   # -1.5% to -2.5%: good signal
            'BIG':     0.67,   # -2.5% to -4.0%: real signal
            'CRASH':   0.00,   # <-4%: genuine crash, no reversal
        },
        'reversal_by_year': {
            2019: 0.50, 2020: 0.40, 2021: 0.67, 2022: 0.29,
            2023: 0.40, 2024: 0.80, 2025: 0.60, 2026: 1.00,
        },
        'trade_action': 'CONDITIONAL',
        'conditions': {
            'spike_holds_and_volume_confirms': 'FOLLOW_SPIKE',
            'spike_reverses_on_low_volume': 'FADE_SETUP',
            'spike_reverses_on_high_volume': 'STOP_AND_WAIT',
            'crash_mode': 'DO_NOT_BUY_DIP',
        },
        'notes': [
            'Small dumps (-0.5% to -1.5%) are noise — 60% reverse',
            'Medium dumps (-1.5% to -2.5%) — 71% reverse, good fade setup',
            'Big dumps (-2.5% to -4%) — 67% reverse, but when they don\'t, it\'s a crash',
            'Crashes (<-4%) — DO NOT buy the dip',
            '2026: 100% reversal rate on PPI dumps (stagflation + crowded)',
            '2-day window captures more alpha than intraday',
        ],
    },

    PHASE_ASIA_RESPONSE: {
        'description': 'Asia session (00:00-08:00 UTC). The overnight reaction.',
        'duration': '8h (next day)',
        'fade_rates': {
            # How often does Asia reverse the US move?
            'TIGHTENING': 0.30,
            'EASING': 0.33,
            'CRISIS_RECOVERY': 0.48,
            'BULL': 0.12,
            'BEAR': 0.29,
            'RECOVERY': 0.29,
            'ACCELERATION': 0.29,
            'STAGFLATION': 0.46,
            'STAGFLATION_HOT': 0.50,
        },
        'nfp_fade_rates': {
            'after_dump': 0.60,
            'after_rally': 0.64,
        },
        'patterns': {
            'CONTINUATION': {
                'description': 'Asia continues US direction (gap held)',
                'probability': 'regime-dependent (40-70%)',
                'next_phase_bias': 'UK likely continues (51%)',
            },
            'FADE': {
                'description': 'Asia reverses US direction (mean-reversion)',
                'probability': 'regime-dependent (12-50%)',
                'next_phase_bias': 'UK decides: continue fade (38%) or reverse back',
            },
            'SWEEP_REVERSAL': {
                'description': 'Asia sweeps US high/low then reverses',
                'probability': 'depends on gap size',
                'next_phase_bias': 'UK fades sweep 59% (morning sweep pattern)',
                'detection': 'sweep_depth > 0.5% and recovery > 50%',
            },
            'CRASH': {
                'description': 'Genuine continuation, NOT a sweep',
                'probability': 'rare (<5% of releases)',
                'next_phase_bias': 'DO NOT buy — wait for stabilization',
                'detection': 'gap flat, range > 7%, no sweep pattern',
            },
        },
        'gap_reliability': {
            2018: 0.65, 2019: 0.71, 2020: 0.63, 2021: 0.75,
            2022: 0.71, 2023: 0.67, 2024: 0.71, 2025: 0.79, 2026: 0.75,
        },
        'trade_action': 'CONDITIONAL',
        'conditions': {
            'fade_high_probability': 'ENTER_FADE',
            'continuation_with_gap_held': 'FOLLOW_MOMENTUM',
            'sweep_reversal_detected': 'WAIT_FOR_LONDON',
            'crash_detected': 'STAY_OUT',
        },
        'notes': [
            'Asia fades are the most reliable pattern in stagflation (50%)',
            'NFP has stronger fade than CPI/PPI (60% after dump, 64% after rally)',
            'Sweep-reversal in Asia → London fades it 59% of the time',
            'Gap held → London continues 51% of the time',
            '2026 gap reliability: 75% — when gap holds, it\'s meaningful',
        ],
    },

    PHASE_LONDON_DECISION: {
        'description': 'London session (07:00-16:00 UTC). The informed decision maker.',
        'duration': '9h',
        'continuation_after_gap_held': {
            # UK continues Asia direction when gap held
            'TIGHTENING': 0.38, 'EASING': 0.50, 'CRISIS_RECOVERY': 0.71,
            'BULL': 0.58, 'BEAR': 0.73, 'RECOVERY': 0.36,
            'ACCELERATION': 0.46, 'STAGFLATION': 0.53, 'STAGFLATION_HOT': 0.20,
        },
        'fade_after_sweep_reversal': {
            # UK fades Asia after sweep-reversal
            'TIGHTENING': 0.50, 'EASING': 0.30, 'CRISIS_RECOVERY': 0.36,
            'BULL': 0.44, 'BEAR': 0.36, 'RECOVERY': 0.33,
            'ACCELERATION': 0.33, 'STAGFLATION': 0.29, 'STAGFLATION_HOT': 0.67,
        },
        'morning_sweep_fade_rate': 0.593,  # 59.3% fade after morning sweep
        'uk_volume_ratio': {
            'CONTINUATION': 1.19,  # UK moves 119% of Asia's move
            'FADE': 1.01,          # UK moves 101% of Asia's move
        },
        'double_reversal': {
            # US dumps → Asia fades → UK decides
            'bounce': 0.58,    # continues Asia's fade
            'continue': 0.42,  # reverses back to US direction
        },
        'double_dump': {
            # US dumps → Asia dumps → UK decides
            'bounce': 0.52,
            'continue': 0.48,
        },
        'trade_action': 'CONDITIONAL',
        'conditions': {
            'gap_held_and_continuation_likely': 'FOLLOW_MOMENTUM',
            'sweep_reversal_detected': 'FADE_THE_SWEEP',
            'morning_sweep_confirmed': 'ENTER_REVERSAL',
            'double_reversal': 'FADE_FAVORS_BOUNCE',
            'low_volume': 'WAIT_FOR_CONFIRMATION',
        },
        'notes': [
            'London is the "informed decision maker" — sees both US and Asia',
            'Stagflation_HOT: only 20% continuation after gap held (fades dominant)',
            'Stagflation_HOT: 67% fade after sweep reversal (strongest pattern)',
            'UK volume is 1.19x Asia on continuation moves — vol confirms conviction',
            'Morning sweep → UK fades 59% — highest-confidence London pattern',
        ],
    },

    PHASE_SECOND_US: {
        'description': 'Second US session (day after release). Extension or reversal?',
        'duration': '7.5h (day 2)',
        'expected_moves': {
            # 2-day cumulative moves by (regime, surprise)
            ('BEAR', 'COOL'): +9.92,
            ('BEAR', 'HOT'): -3.33,
            ('BULL', 'COOL'): +1.88,
            ('BULL', 'HOT'): -0.20,
            ('RECOVERY', 'COOL'): -0.55,
            ('RECOVERY', 'HOT'): +0.84,
            ('STAGFLATION', 'COOL'): +3.00,
            ('STAGFLATION', 'HOT'): -2.00,
            ('STAGFLATION_HOT', 'COOL'): +4.00,
            ('STAGFLATION_HOT', 'HOT'): -3.00,
        },
        'extension_vs_reversal': {
            # Does day 2 extend or reverse day 1?
            'cool_cpi_extension_rate': 0.65,   # 65% of cool CPI extends on day 2
            'hot_cpi_extension_rate': 0.55,    # 55% of hot CPI extends on day 2
            'cool_ppi_extension_rate': 0.40,   # PPI is noisier
            'hot_ppi_extension_rate': 0.50,
        },
        'trade_action': 'CONDITIONAL',
        'conditions': {
            'day1_aligned_with_regime': 'HOLD_OR_ADD',
            'day1_against_regime': 'CONSIDER_REVERSAL',
            'vol_fading': 'TAKE_PARTIAL',
            'vol_expanding': 'HOLD_FULL',
        },
        'notes': [
            '2-day window captures more alpha than intraday (cool CPI: +2.50% vs +1.06%)',
            'Day 2 often sees vol expansion if day 1 was inline',
            'If day 1 was a fade, day 2 often confirms the fade',
            'Stagflation: day 2 tends to extend the move (momentum)',
        ],
    },

    PHASE_MULTI_DAY: {
        'description': '2-5 days post-release. Cascade resolution.',
        'duration': '2-5 days',
        'cascade_patterns': {
            'MOMENTUM': {
                'description': 'Move extends over 2-5 days',
                'probability': 'regime-dependent',
                'typical_duration': '3-5 days',
                'exit_signal': 'vol fade + RSI divergence',
            },
            'MEAN_REVERSION': {
                'description': 'Move reverses within 2-3 days',
                'probability': 'regime-dependent',
                'typical_duration': '2-3 days',
                'exit_signal': 'return to pre-release level',
            },
            'CONSOLIDATION': {
                'description': 'Move holds but goes sideways',
                'probability': 'common in inline releases',
                'typical_duration': '3-5 days',
                'exit_signal': 'breakout of consolidation range',
            },
        },
        'trade_action': 'TRAIL_STOPS',
        'notes': [
            'Cool CPI in bear market: +9.92% over 2 days (extreme momentum)',
            'Hot CPI in stagflation: -3.00% over 2 days',
            'After 3 days, the macro signal is fully priced in',
            'Next release cycle begins — reset lifecycle',
        ],
    },
}


# ═══════════════════════════════════════════════════════════════
# STATE MACHINE
# ═══════════════════════════════════════════════════════════════

class MacroLifecycle:
    """Tracks a macro data release through its lifecycle phases.

    Usage:
        lc = MacroLifecycle(release_type='PPI', release_date='2026-05-13')
        state = lc.evaluate(df_15m, current_time=datetime.utcnow())
        print(state['phase'])
        print(state['summary'])
    """

    def __init__(self, release_type='PPI', release_date=None, config=None):
        self.release_type = release_type
        self.release_date = release_date
        self.config = config or CONFIG
        self.props = RELEASE_PROPERTIES.get(release_type, RELEASE_PROPERTIES['PPI'])

    def evaluate(self, df_15m, current_time=None):
        """Evaluate current lifecycle phase and generate state.

        Args:
            df_15m: DataFrame with 15m OHLCV data
            current_time: datetime (default: now UTC)

        Returns:
            dict with phase, probabilities, trade implications
        """
        if current_time is None:
            current_time = datetime.utcnow()

        if self.release_date is None:
            return self._no_release(current_time)

        release_dt = datetime.strptime(self.release_date, '%Y-%m-%d')
        release_time = release_dt.replace(hour=RELEASE_HOUR_UTC, minute=RELEASE_MINUTE_UTC)
        hours_since = (current_time - release_time).total_seconds() / 3600

        # Determine phase
        phase = self._determine_phase(current_time, release_dt, hours_since)

        # Compute session data based on phase
        session_data = self._compute_session_data(df_15m, release_dt, phase, current_time)

        # Get phase model
        phase_model = PHASE_MODELS.get(phase, {})

        # Build state
        state = {
            'phase': phase,
            'release_type': self.release_type,
            'release_date': self.release_date,
            'hours_since_release': round(hours_since, 1),
            'current_time': current_time.strftime('%Y-%m-%d %H:%M UTC'),
            'phase_model': phase_model,
            'session_data': session_data,
            'trade_action': phase_model.get('trade_action', 'OBSERVE'),
            'size_multiplier': phase_model.get('size_multiplier', 1.0),
        }

        # Phase-specific analysis
        if phase == PHASE_ANTICIPATION:
            state.update(self._analyze_anticipation(current_time, release_dt))
        elif phase == PHASE_RELEASE:
            state.update(self._analyze_release(df_15m, release_dt))
        elif phase == PHASE_US_CONTINUATION:
            state.update(self._analyze_us_continuation(df_15m, release_dt, session_data))
        elif phase == PHASE_ASIA_RESPONSE:
            state.update(self._analyze_asia_response(df_15m, release_dt, session_data))
        elif phase == PHASE_LONDON_DECISION:
            state.update(self._analyze_london_decision(df_15m, release_dt, session_data))
        elif phase == PHASE_SECOND_US:
            state.update(self._analyze_second_us(df_15m, release_dt, session_data))
        elif phase == PHASE_MULTI_DAY:
            state.update(self._analyze_multi_day(df_15m, release_dt, session_data))

        # Generate summary
        state['summary'] = self._generate_summary(state)

        return state

    def _determine_phase(self, current_time, release_dt, hours_since):
        """Determine which lifecycle phase we're in."""
        release_time = release_dt.replace(hour=RELEASE_HOUR_UTC, minute=RELEASE_MINUTE_UTC)

        if hours_since < -2:
            return PHASE_ANTICIPATION
        elif hours_since < -0.25:
            return PHASE_ANTICIPATION  # last 15min before release
        elif hours_since < 0.25:
            return PHASE_RELEASE        # first 15min after release
        elif hours_since < 7.5:
            return PHASE_US_CONTINUATION # rest of US session
        else:
            # Check which day we're on
            days_since = (current_time.date() - release_dt.date()).days

            if days_since == 0:
                # Still release day — check if US session is over
                if current_time.hour >= 21:
                    return PHASE_ASIA_RESPONSE  # US closed, waiting for Asia
                return PHASE_US_CONTINUATION
            elif days_since == 1:
                # Day after release
                utc_hour = current_time.hour
                if utc_hour < 7:
                    return PHASE_ASIA_RESPONSE
                elif utc_hour < 16:
                    return PHASE_LONDON_DECISION
                else:
                    return PHASE_SECOND_US  # London closed, second US session
            elif days_since == 2:
                return PHASE_SECOND_US
            elif days_since <= 5:
                return PHASE_MULTI_DAY
            else:
                return PHASE_RESOLVED

    def _compute_session_data(self, df_15m, release_dt, phase, current_time):
        """Compute available session data based on current phase."""
        from src.modules.cascade_engine import (
            compute_us_session, compute_asia_session, compute_uk_session,
            compute_1h_spike,
        )
        from src.modules.macro_utils import classify_market_regime

        data = {}

        # Always compute US session data (available after release)
        release_time = release_dt.replace(hour=RELEASE_HOUR_UTC, minute=RELEASE_MINUTE_UTC)
        if (current_time - release_time).total_seconds() > 3600:  # >1h after release
            us_data = compute_us_session(df_15m, self.release_date)
            if us_data:
                data['us'] = us_data
                us_move = us_data['us_move']
                data['us_direction'] = 'DUMP' if us_move < -0.5 else 'RALLY' if us_move > 0.5 else 'FLAT'
                data['us_magnitude'] = 'BIG' if abs(us_move) > 3.0 else 'MEDIUM' if abs(us_move) > 1.5 else 'SMALL'

                # Spike data
                spike = compute_1h_spike(df_15m, self.release_date)
                if spike:
                    data['spike'] = spike

        # Asia data (available day after)
        days_since = (current_time.date() - release_dt.date()).days
        if days_since >= 1:
            asia_data = compute_asia_session(df_15m, self.release_date)
            if asia_data:
                data['asia'] = asia_data
                data['asia_pattern'] = self._classify_asia_pattern(asia_data, data.get('us_direction'))

        # UK data (available day after, after London)
        if days_since >= 1 and current_time.hour >= 16:
            uk_data = compute_uk_session(df_15m, self.release_date)
            if uk_data:
                data['uk'] = uk_data

        # Regime
        regime, fade_rate = classify_market_regime(self.config)
        data['regime'] = regime
        data['fade_rate'] = fade_rate

        return data

    def _classify_asia_pattern(self, asia_data, us_direction):
        """Classify Asia session pattern."""
        if not asia_data or not us_direction:
            return 'UNKNOWN'

        gap_dir = asia_data.get('gap_dir', 'FLAT')
        asia_move = asia_data.get('asia_move', 0)
        is_sweep = asia_data.get('is_sweep_reversal', False)
        asia_range = asia_data.get('asia_range', 0)

        # Crash detection
        if gap_dir == 'FLAT' and asia_range > 7.0 and not is_sweep:
            return 'CRASH'
        if asia_range > 10.0:
            return 'CRASH'

        # Sweep reversal
        if is_sweep:
            return 'SWEEP_REVERSAL'

        # Fade vs continuation
        asia_dir = 'UP' if asia_move > 0.3 else 'DOWN' if asia_move < -0.3 else 'FLAT'
        if us_direction == 'DUMP' and asia_dir == 'UP':
            return 'FADE'
        elif us_direction == 'RALLY' and asia_dir == 'DOWN':
            return 'FADE'
        elif asia_dir != 'FLAT':
            return 'CONTINUATION'

        return 'FLAT'

    def _analyze_anticipation(self, current_time, release_dt):
        """Analyze pre-release anticipation phase."""
        hours_until = (release_dt.replace(hour=RELEASE_HOUR_UTC) - current_time).total_seconds() / 3600

        regime = self.config.get('M22_FED_STANCE', 'HOLDING')
        ppi = self.config.get('M22_PPI_YOY', 0)
        cpi = self.config.get('M22_CPI_YOY', 0)

        # Anticipation signals
        signals = []
        if hours_until < 4:
            signals.append(f'Release in {hours_until:.1f}h — reduce position size')
        if hours_until < 1:
            signals.append(f'Imminent — expect vol spike within 1h')

        return {
            'hours_until_release': round(hours_until, 1),
            'anticipation_signals': signals,
            'expected_move': self.props.get('avg_us_move', 2.0),
        }

    def _analyze_release(self, df_15m, release_dt):
        """Analyze the release moment (spike phase)."""
        from src.modules.cascade_engine import compute_1h_spike

        spike = compute_1h_spike(df_15m, self.release_date)
        if not spike:
            return {'spike_status': 'NO_DATA'}

        spike_dir = spike['spike_dir']
        spike_pct = spike['spike_pct']
        year = release_dt.year

        # Spike accuracy for this type+year
        accuracy = self.props.get('spike_accuracy', {}).get(year, 0.65)

        return {
            'spike_status': 'DETECTED',
            'spike_direction': spike_dir,
            'spike_pct': spike_pct,
            'spike_accuracy': accuracy,
            'spike_confidence': 'HIGH' if accuracy >= 0.75 else 'MEDIUM' if accuracy >= 0.60 else 'LOW',
            'action': 'WAIT_FOR_US_SESSION' if abs(spike_pct) < 1.0 else 'CONSIDER_FOLLOWING_SPIKE',
        }

    def _analyze_us_continuation(self, df_15m, release_dt, session_data):
        """Analyze US session continuation phase."""
        us = session_data.get('us', {})
        spike = session_data.get('spike', {})
        regime = session_data.get('regime', 'UNKNOWN')

        if not us:
            return {'us_status': 'NO_DATA'}

        us_move = us['us_move']
        us_dir = session_data.get('us_direction', 'FLAT')
        dump_size = self._classify_dump_size(us_move)

        # Fade probability
        fade_rates = PHASE_MODELS[PHASE_US_CONTINUATION]['fade_rates']
        fade_prob = fade_rates.get(regime, 0.33)

        # Year-based reversal
        year = release_dt.year
        year_rev = PHASE_MODELS[PHASE_US_CONTINUATION]['reversal_by_year'].get(year, 0.50)

        # Combined
        combined_rev = year_rev * 0.40 + fade_prob * 0.60

        # Spike held?
        spike_held = False
        if spike and us_dir != 'FLAT':
            spike_dir = spike.get('spike_dir', 'FLAT')
            spike_held = (spike_dir == 'UP' and us_dir == 'RALLY') or \
                         (spike_dir == 'DOWN' and us_dir == 'DUMP')

        return {
            'us_status': 'ACTIVE',
            'us_move': round(us_move, 3),
            'us_direction': us_dir,
            'dump_size': dump_size,
            'fade_probability': round(fade_prob, 3),
            'year_reversal_rate': round(year_rev, 3),
            'combined_reversal_prob': round(combined_rev, 3),
            'spike_held': spike_held,
            'action': self._us_continuation_action(us_dir, dump_size, spike_held, combined_rev, regime),
        }

    def _analyze_asia_response(self, df_15m, release_dt, session_data):
        """Analyze Asia session response phase."""
        asia = session_data.get('asia', {})
        us_dir = session_data.get('us_direction', 'FLAT')
        regime = session_data.get('regime', 'UNKNOWN')
        pattern = session_data.get('asia_pattern', 'UNKNOWN')

        if not asia:
            return {'asia_status': 'NO_DATA', 'prediction': self._predict_asia(us_dir, regime)}

        asia_move = asia.get('asia_move', 0)
        asia_gap = asia.get('asia_gap', 0)
        gap_held = asia.get('gap_dir', 'FLAT') != 'FLAT' and \
                   (asia.get('gap_dir') == ('UP' if asia_move > 0 else 'DOWN'))

        # What comes next? (London prediction)
        next_phase = self._predict_london(pattern, regime, gap_held, asia)

        return {
            'asia_status': 'COMPLETE',
            'asia_move': round(asia_move, 3),
            'asia_gap': round(asia_gap, 3),
            'asia_pattern': pattern,
            'gap_held': gap_held,
            'london_prediction': next_phase,
            'action': self._asia_response_action(pattern, regime, gap_held),
        }

    def _analyze_london_decision(self, df_15m, release_dt, session_data):
        """Analyze London session decision phase."""
        uk = session_data.get('uk', {})
        asia = session_data.get('asia', {})
        us_dir = session_data.get('us_direction', 'FLAT')
        regime = session_data.get('regime', 'UNKNOWN')
        pattern = session_data.get('asia_pattern', 'UNKNOWN')

        if not uk:
            return {'uk_status': 'NO_DATA'}

        uk_dir = uk.get('uk_direction', 'FLAT')
        uk_continued = uk.get('uk_continued_asia', False)
        uk_faded = uk.get('uk_faded_asia', False)
        uk_swept = uk.get('is_morning_sweep', False)

        # UK verdict
        if uk_continued:
            verdict = 'CONFIRMED'
            confidence_boost = 0.05
        elif uk_faded:
            verdict = 'FADED'
            confidence_boost = -0.05
        elif uk_swept:
            verdict = 'SWEPT_AND_REVERSED'
            confidence_boost = -0.08
        else:
            verdict = 'NEUTRAL'
            confidence_boost = 0.0

        # What comes next? (Second US session prediction)
        next_prediction = self._predict_second_us(us_dir, pattern, uk_dir, regime, verdict)

        return {
            'uk_status': 'COMPLETE',
            'uk_direction': uk_dir,
            'uk_verdict': verdict,
            'uk_move_vs_asia': uk.get('uk_move_vs_asia', 0),
            'uk_taker': uk.get('uk_taker', 0.5),
            'uk_vol_ratio': uk.get('uk_vol_ratio_vs_asia', 1.0),
            'second_us_prediction': next_prediction,
            'confidence_boost': confidence_boost,
            'action': self._london_decision_action(verdict, uk_dir, regime),
        }

    def _analyze_second_us(self, df_15m, release_dt, session_data):
        """Analyze second US session phase."""
        regime = session_data.get('regime', 'UNKNOWN')
        us_dir = session_data.get('us_direction', 'FLAT')

        # Expected 2-day move
        ppi = self.config.get('M22_PPI_YOY', 0)
        cpi = self.config.get('M22_CPI_YOY', 0)
        surprise = 'COOL' if (cpi and cpi < 2.5) else 'HOT' if (cpi and cpi >= 3.5) else 'WARM'
        expected_key = (regime, surprise)

        phase_model = PHASE_MODELS[PHASE_SECOND_US]
        expected_2day = phase_model['expected_moves'].get(expected_key, 0)

        return {
            'second_us_status': 'ACTIVE',
            'expected_2day_move': expected_2day,
            'surprise_classification': surprise,
            'action': 'HOLD' if abs(expected_2day) > 1.0 else 'TRAIL_STOPS',
        }

    def _analyze_multi_day(self, df_15m, release_dt, session_data):
        """Analyze multi-day cascade phase."""
        days_since = (datetime.utcnow().date() - release_dt.date()).days

        return {
            'multi_day_status': 'ACTIVE',
            'days_since_release': days_since,
            'cascade_phase': 'MOMENTUM' if days_since <= 3 else 'CONSOLIDATION',
            'action': 'TRAIL_STOPS' if days_since <= 3 else 'CLOSE_POSITION',
            'notes': [
                f'Day {days_since} post-release — signal decaying',
                'Next release cycle approaching — reset lifecycle',
            ],
        }

    # ── Prediction helpers ──────────────────────────────────────

    def _predict_asia(self, us_dir, regime):
        """Predict Asia session behavior before it happens."""
        if us_dir == 'FLAT':
            return {'prediction': 'FLAT', 'confidence': 'LOW', 'reason': 'US move too small'}

        fade_rates = PHASE_MODELS[PHASE_ASIA_RESPONSE]['fade_rates']
        fade_rate = fade_rates.get(regime, 0.33)

        # NFP-specific
        if 'NFP' in self.release_type:
            nfp_fade = PHASE_MODELS[PHASE_ASIA_RESPONSE]['nfp_fade_rates']
            if us_dir == 'DUMP':
                fade_rate = nfp_fade['after_dump']
            elif us_dir == 'RALLY':
                fade_rate = nfp_fade['after_rally']

        if fade_rate > 0.50:
            prediction = 'FADE'
            confidence = 'HIGH' if fade_rate > 0.60 else 'MEDIUM'
        else:
            prediction = 'CONTINUATION'
            confidence = 'MEDIUM' if fade_rate < 0.35 else 'LOW'

        return {
            'prediction': prediction,
            'fade_rate': fade_rate,
            'confidence': confidence,
            'expected_asia_direction': 'UP' if prediction == 'FADE' and us_dir == 'DUMP' else
                                       'DOWN' if prediction == 'FADE' and us_dir == 'RALLY' else
                                       us_dir,
        }

    def _predict_london(self, asia_pattern, regime, gap_held, asia_data):
        """Predict London session behavior based on Asia outcome."""
        if asia_pattern == 'CRASH':
            return {'prediction': 'STABILIZE', 'confidence': 'MEDIUM',
                    'reason': 'Crash mode — London watches for stabilization'}

        if asia_pattern == 'SWEEP_REVERSAL':
            fade_rate = PHASE_MODELS[PHASE_LONDON_DECISION]['fade_after_sweep_reversal'].get(regime, 0.38)
            morning_sweep_rate = PHASE_MODELS[PHASE_LONDON_DECISION]['morning_sweep_fade_rate']
            return {
                'prediction': 'FADE_THE_SWEEP',
                'fade_rate': fade_rate,
                'morning_sweep_rate': morning_sweep_rate,
                'confidence': 'HIGH' if fade_rate > 0.55 else 'MEDIUM',
                'reason': f'Asia sweep-reversal → London fades {fade_rate:.0%} (regime={regime})',
            }

        if gap_held:
            cont_rate = PHASE_MODELS[PHASE_LONDON_DECISION]['continuation_after_gap_held'].get(regime, 0.51)
            return {
                'prediction': 'CONTINUATION',
                'continuation_rate': cont_rate,
                'confidence': 'HIGH' if cont_rate > 0.60 else 'MEDIUM' if cont_rate > 0.45 else 'LOW',
                'reason': f'Asia continued US (gap held) → London continues {cont_rate:.0%}',
            }

        # Asia faded US — double reversal
        double_rev = PHASE_MODELS[PHASE_LONDON_DECISION]['double_reversal']
        return {
            'prediction': 'DOUBLE_REVERSAL',
            'bounce_prob': double_rev['bounce'],
            'confidence': 'MEDIUM',
            'reason': f'Asia faded US → London bounces (continues fade) {double_rev["bounce"]:.0%}',
        }

    def _predict_second_us(self, us_dir, asia_pattern, uk_dir, regime, uk_verdict):
        """Predict second US session behavior."""
        if uk_verdict == 'CONFIRMED':
            return {
                'prediction': 'EXTENSION',
                'confidence': 'MEDIUM',
                'reason': f'London confirmed {uk_dir} — second US likely extends',
            }
        elif uk_verdict == 'FADED':
            return {
                'prediction': 'REVERSAL',
                'confidence': 'MEDIUM',
                'reason': f'London faded — second US may reverse to original direction',
            }
        elif uk_verdict == 'SWEPT_AND_REVERSED':
            return {
                'prediction': 'REVERSAL',
                'confidence': 'HIGH',
                'reason': 'London swept-and-reversed — high-confidence reversal expected',
            }

        return {
            'prediction': 'MIXED',
            'confidence': 'LOW',
            'reason': 'No clear London signal — second US direction uncertain',
        }

    # ── Action helpers ──────────────────────────────────────────

    def _us_continuation_action(self, us_dir, dump_size, spike_held, rev_prob, regime):
        """Determine trade action during US continuation phase."""
        if dump_size == 'CRASH':
            return 'DO_NOT_BUY_DIP'

        if not spike_held and rev_prob > 0.60:
            return 'FADE_SETUP'

        if spike_held and rev_prob < 0.40:
            return 'FOLLOW_SPIKE'

        return 'WAIT_FOR_CLOSE'

    def _asia_response_action(self, pattern, regime, gap_held):
        """Determine trade action during Asia response phase."""
        if pattern == 'CRASH':
            return 'STAY_OUT'
        elif pattern == 'SWEEP_REVERSAL':
            return 'WAIT_FOR_LONDON'
        elif pattern == 'FADE' and regime in ('STAGFLATION', 'STAGFLATION_HOT'):
            return 'ENTER_FADE'  # high-confidence fade in stagflation
        elif pattern == 'CONTINUATION' and gap_held:
            return 'FOLLOW_MOMENTUM'
        return 'OBSERVE'

    def _london_decision_action(self, verdict, uk_dir, regime):
        """Determine trade action during London decision phase."""
        if verdict == 'SWEPT_AND_REVERSED':
            return 'ENTER_REVERSAL'
        elif verdict == 'CONFIRMED':
            return 'HOLD_OR_ADD'
        elif verdict == 'FADED':
            if regime in ('STAGFLATION', 'STAGFLATION_HOT'):
                return 'ENTER_FADE'  # stagflation fades are reliable
            return 'CONSIDER_FADE'
        return 'OBSERVE'

    def _classify_dump_size(self, us_move):
        """Classify US session dump size."""
        if us_move >= -0.5:
            return 'NOT_DUMP'
        elif us_move >= -1.5:
            return 'SMALL'
        elif us_move >= -2.5:
            return 'MEDIUM'
        elif us_move >= -4.0:
            return 'BIG'
        return 'CRASH'

    # ── Summary generator ───────────────────────────────────────

    def _generate_summary(self, state):
        """Generate human-readable lifecycle summary."""
        phase = state['phase']
        release_type = state['release_type']
        session_data = state.get('session_data', {})

        lines = []
        lines.append(f"📊 MACRO LIFECYCLE: {release_type} → Phase {phase}")
        lines.append(f"   Released: {state['release_date']}  |  {state['hours_since_release']:+.1f}h ago")

        regime = session_data.get('regime', '?')
        lines.append(f"   Regime: {regime}")

        # Phase-specific summary
        if phase == PHASE_ANTICIPATION:
            hours_until = state.get('hours_until_release', 0)
            lines.append(f"   ⏳ Release in {hours_until:.1f}h — reduce size, wait for data")
            lines.append(f"   Expected |move|: {state.get('expected_move', 2.0):.1f}%")

        elif phase == PHASE_RELEASE:
            spike = state.get('spike_status', 'NO_DATA')
            if spike == 'DETECTED':
                lines.append(f"   ⚡ Spike: {state['spike_direction']} {state['spike_pct']:+.2f}%")
                lines.append(f"   Accuracy: {state['spike_accuracy']:.0%} ({state['spike_confidence']})")
            else:
                lines.append(f"   ⏳ Waiting for spike data...")

        elif phase == PHASE_US_CONTINUATION:
            us_dir = state.get('us_direction', '?')
            us_move = state.get('us_move', 0)
            rev_prob = state.get('combined_reversal_prob', 0)
            spike_held = state.get('spike_held', False)
            dump_size = state.get('dump_size', 'NOT_DUMP')

            lines.append(f"   🇺🇸 US: {us_dir} {us_move:+.2f}%  dump={dump_size}")
            lines.append(f"   Spike held: {'✅' if spike_held else '❌'}  |  Rev prob: {rev_prob:.0%}")
            lines.append(f"   Action: {state.get('action', '?')}")

        elif phase == PHASE_ASIA_RESPONSE:
            pattern = state.get('asia_pattern', '?')
            asia_move = state.get('asia_move', 0)
            gap_held = state.get('gap_held', False)

            lines.append(f"   🌏 Asia: {pattern} {asia_move:+.2f}%  gap_held={'✅' if gap_held else '❌'}")
            pred = state.get('london_prediction', {})
            lines.append(f"   London pred: {pred.get('prediction', '?')} ({pred.get('confidence', '?')})")
            lines.append(f"   {pred.get('reason', '')}")
            lines.append(f"   Action: {state.get('action', '?')}")

        elif phase == PHASE_LONDON_DECISION:
            uk_dir = state.get('uk_direction', '?')
            verdict = state.get('uk_verdict', '?')
            uk_move = state.get('uk_move_vs_asia', 0)

            lines.append(f"   🇬🇧 London: {uk_dir} {uk_move:+.2f}% vs Asia  verdict={verdict}")
            pred = state.get('second_us_prediction', {})
            lines.append(f"   2nd US pred: {pred.get('prediction', '?')} ({pred.get('confidence', '?')})")
            lines.append(f"   {pred.get('reason', '')}")
            lines.append(f"   Action: {state.get('action', '?')}")

        elif phase == PHASE_SECOND_US:
            expected = state.get('expected_2day_move', 0)
            surprise = state.get('surprise_classification', '?')
            lines.append(f"   🇺🇸 Day 2: expected 2-day move {expected:+.1f}%  surprise={surprise}")
            lines.append(f"   Action: {state.get('action', '?')}")

        elif phase == PHASE_MULTI_DAY:
            days = state.get('days_since_release', 0)
            cascade = state.get('cascade_phase', '?')
            lines.append(f"   📅 Day {days}: cascade={cascade}")
            lines.append(f"   Action: {state.get('action', '?')}")

        elif phase == PHASE_RESOLVED:
            lines.append(f"   ✅ Lifecycle complete — signal fully resolved")

        return '\n'.join(lines)

    def _no_release(self, current_time):
        """Return state when no release is active."""
        return {
            'phase': 'NO_RELEASE',
            'release_type': self.release_type,
            'release_date': None,
            'hours_since_release': 0,
            'current_time': current_time.strftime('%Y-%m-%d %H:%M UTC'),
            'trade_action': 'NORMAL',
            'size_multiplier': 1.0,
            'summary': f'📊 MACRO LIFECYCLE: {self.release_type} — no active release',
        }


# ═══════════════════════════════════════════════════════════════
# CONVENIENCE FUNCTIONS
# ═══════════════════════════════════════════════════════════════

def detect_active_release(config=None):
    """Detect if there's an active macro release in its lifecycle.

    Checks today and yesterday for any release type.
    Returns: (release_type, release_date) or (None, None)
    """
    cfg = config or CONFIG
    today = datetime.utcnow().strftime('%Y-%m-%d')
    yesterday = (datetime.utcnow() - timedelta(days=1)).strftime('%Y-%m-%d')

    from src.modules.macro_utils import get_release_type

    # Check yesterday first (post-release has actual data)
    yesterday_type = get_release_type(yesterday)
    if yesterday_type:
        return yesterday_type, yesterday

    # Check today
    today_type = get_release_type(today)
    if today_type:
        return today_type, today

    # Check up to 5 days back (extended window with decay)
    for lookback in range(2, 6):
        check_date = (datetime.utcnow() - timedelta(days=lookback)).strftime('%Y-%m-%d')
        check_type = get_release_type(check_date)
        if check_type:
            return check_type, check_date

    return None, None


def evaluate_macro_lifecycle(df_15m, current_time=None, config=None):
    """Evaluate the current macro lifecycle state.

    Auto-detects active release and returns lifecycle state.
    Returns: dict with lifecycle state, or None if no active release.
    """
    cfg = config or CONFIG
    release_type, release_date = detect_active_release(cfg)

    if release_type is None:
        return None

    lc = MacroLifecycle(release_type=release_type, release_date=release_date, config=cfg)
    return lc.evaluate(df_15m, current_time=current_time)


def format_lifecycle(state):
    """Format lifecycle state for terminal output."""
    if state is None:
        return ''

    lines = []
    phase = state.get('phase', '?')
    release_type = state.get('release_type', '?')
    release_date = state.get('release_date', '?')
    hours = state.get('hours_since_release', 0)

    # Phase icons
    phase_icons = {
        'ANTICIPATION': '⏳',
        'RELEASE': '⚡',
        'US_CONTINUATION': '🇺🇸',
        'ASIA_RESPONSE': '🌏',
        'LONDON_DECISION': '🇬🇧',
        'SECOND_US': '🇺🇸',
        'MULTI_DAY': '📅',
        'RESOLVED': '✅',
        'NO_RELEASE': '⚪',
    }
    icon = phase_icons.get(phase, '❓')

    lines.append(f"\n  {icon} MACRO LIFECYCLE: {release_type} ({release_date})")
    lines.append(f"  Phase: {phase}  |  {hours:+.1f}h since release")

    # Session data
    session_data = state.get('session_data', {})
    regime = session_data.get('regime', '?')
    fade_rate = session_data.get('fade_rate', 0)
    lines.append(f"  Regime: {regime}  fade_rate={fade_rate:.0%}")

    # US session
    us = session_data.get('us')
    if us:
        us_dir = session_data.get('us_direction', '?')
        us_move = us.get('us_move', 0)
        us_icon = '🔴' if us_move < -0.5 else '🟢' if us_move > 0.5 else '⚪'
        lines.append(f"  US Session: {us_icon} {us_dir} {us_move:+.2f}%  range={us.get('us_range', 0):.1f}%")

    # Spike
    spike = session_data.get('spike')
    if spike:
        s_icon = '🟢' if spike['spike_dir'] == 'UP' else '🔴' if spike['spike_dir'] == 'DOWN' else '⚪'
        lines.append(f"  1h Spike: {s_icon} {spike['spike_pct']:+.2f}%  ({spike['spike_dir']})")

    # Asia
    asia = session_data.get('asia')
    if asia:
        pattern = session_data.get('asia_pattern', '?')
        asia_move = asia.get('asia_move', 0)
        a_icon = '🟢' if asia_move > 0 else '🔴'
        gap_held = session_data.get('gap_held', False)
        lines.append(f"  Asia: {a_icon} {pattern} {asia_move:+.2f}%  gap_held={'✅' if gap_held else '❌'}")

    # UK
    uk = session_data.get('uk')
    if uk:
        uk_dir = uk.get('uk_direction', '?')
        uk_move = uk.get('uk_move_vs_asia', 0)
        u_icon = '🟢' if uk_move > 0.3 else '🔴' if uk_move < -0.3 else '⚪'
        lines.append(f"  London: {u_icon} {uk_dir} {uk_move:+.2f}% vs Asia")

    # Phase-specific details
    if phase == 'US_CONTINUATION':
        dump_size = state.get('dump_size', '?')
        rev_prob = state.get('combined_reversal_prob', 0)
        spike_held = state.get('spike_held', False)
        lines.append(f"  Dump: {dump_size}  |  Rev prob: {rev_prob:.0%}  |  Spike held: {'✅' if spike_held else '❌'}")

    elif phase == 'ASIA_RESPONSE':
        pred = state.get('london_prediction', {})
        lines.append(f"  London prediction: {pred.get('prediction', '?')} ({pred.get('confidence', '?')})")
        if pred.get('reason'):
            lines.append(f"    {pred['reason']}")

    elif phase == 'LONDON_DECISION':
        verdict = state.get('uk_verdict', '?')
        lines.append(f"  UK Verdict: {verdict}")
        pred = state.get('second_us_prediction', {})
        lines.append(f"  2nd US prediction: {pred.get('prediction', '?')} ({pred.get('confidence', '?')})")
        if pred.get('reason'):
            lines.append(f"    {pred['reason']}")

    # Trade action
    action = state.get('trade_action', '?')
    size_mult = state.get('size_multiplier', 1.0)
    action_icons = {
        'REDUCE_SIZE': '🟡', 'OBSERVE': '⚪', 'CONDITIONAL': '🔶',
        'FOLLOW_SPIKE': '🟢', 'FADE_SETUP': '↩️', 'WAIT_FOR_CLOSE': '⏳',
        'ENTER_FADE': '↩️', 'FOLLOW_MOMENTUM': '🟢', 'WAIT_FOR_LONDON': '⏳',
        'STAY_OUT': '🔴', 'ENTER_REVERSAL': '↩️', 'HOLD_OR_ADD': '🟢',
        'CONSIDER_FADE': '↩️', 'TRAIL_STOPS': '🟡', 'CLOSE_POSITION': '🔴',
        'DO_NOT_BUY_DIP': '🔴', 'NORMAL': '⚪', 'HOLD': '🟢',
    }
    a_icon = action_icons.get(action, '❓')
    lines.append(f"  Action: {a_icon} {action}  (size={size_mult:.2f}x)")

    # Notes
    notes = state.get('anticipation_signals', [])
    for note in notes:
        lines.append(f"  💡 {note}")

    return '\n'.join(lines)

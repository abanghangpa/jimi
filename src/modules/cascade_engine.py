"""
Generic Cascade Engine — Macro Release Chain Scoring

Extracts the cascade pattern from M23 (NFP→CPI→PPI→Claims) into a reusable
engine. Any group of related macro releases can be modeled as a chain where:

  LEADING signal → PRIMARY signal → CONFIRMATION → BACKGROUND

Each cascade defines:
  - Releases: ordered list of (name, schedule_dates, weight, role)
  - Confirmation matrix: how releases combine (e.g. CPI_COOL + PPI_HOT = modifier)
  - Regime sensitivity: how much the signal matters by macro regime
  - Time decay: how fast the signal fades post-release
  - Session analysis: US → Asia → UK transmission chain

Usage:
    from src.modules.cascade_engine import CascadeEngine, CascadeRelease

    labor_chain = CascadeEngine(
        name='US_LABOR',
        releases=[
            CascadeRelease('ADP', adp_dates, weight=0.15, role='LEADING'),
            CascadeRelease('JOLTS', jolts_dates, weight=0.10, role='STRUCTURAL'),
            CascadeRelease('NFP', nfp_dates, weight=0.35, role='PRIMARY'),
            CascadeRelease('UNEMPLOYMENT', nfp_dates, weight=0.20, role='CONFIRMATION'),
            CascadeRelease('CLAIMS', claims_weekly, weight=0.10, role='BACKGROUND'),
        ],
        confirmation_matrix={...},
        regime_sensitivity={...},
    )

    status, score, details, decay = labor_chain.score(df_15m, current_time, config)
"""

from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any

# Lazy imports — only needed when scoring with actual DataFrames
np = None
pd = None


def _ensure_pandas():
    global np, pd
    if pd is None:
        import pandas as _pd
        import numpy as _np
        pd = _pd
        np = _np


# ═══════════════════════════════════════════════════════════════
# DATA CLASSES
# ═══════════════════════════════════════════════════════════════

@dataclass
class CascadeRelease:
    """A single release in a cascade chain."""
    name: str                          # e.g. 'NFP', 'CPI', 'ADP'
    schedule_dates: set                # set of 'YYYY-MM-DD' strings
    weight: float                      # contribution to cascade score (0.0-1.0)
    role: str                          # 'LEADING', 'PRIMARY', 'CONFIRMATION', 'BACKGROUND', 'STRUCTURAL'
    enabled_key: str = ''              # config key to enable/disable (e.g. 'M37_ENABLED')
    release_hour_utc: int = 13         # release time hour UTC (default 13:30 = 8:30 AM ET)
    release_minute_utc: int = 30
    score_fn: Any = None               # optional standalone score function for detailed analysis
    signal_classifier: Any = None      # fn(release_data) -> signal_str (e.g. 'COOL', 'HOT')
    data_source: str = ''              # 'BLS', 'BEA', 'ADP', 'FRED', etc.

    def is_release_day(self, date_str: str) -> bool:
        """Check if date is a release day (exact match or weekly for claims)."""
        if self.schedule_dates and date_str in self.schedule_dates:
            return True
        return False

    def is_release_day_window(self, date_str: str, window_days: int = 0) -> Optional[str]:
        """Check if date is within window of a release. Returns actual release date or None."""
        if self.is_release_day(date_str):
            return date_str
        if window_days > 0:
            dt = datetime.strptime(date_str, '%Y-%m-%d')
            for d in range(1, window_days + 1):
                check = (dt - timedelta(days=d)).strftime('%Y-%m-%d')
                if check in self.schedule_dates:
                    return check
                check = (dt + timedelta(days=d)).strftime('%Y-%m-%d')
                if check in self.schedule_dates:
                    return check
        return None

    def find_recent_release(self, date_str: str, max_lookback: int = 14) -> Optional[str]:
        """Find the most recent release on or before date_str."""
        if date_str in self.schedule_dates:
            return date_str
        dt = datetime.strptime(date_str, '%Y-%m-%d')
        for d in range(1, max_lookback + 1):
            check = (dt - timedelta(days=d)).strftime('%Y-%m-%d')
            if check in self.schedule_dates:
                return check
        return None


@dataclass
class CascadeStep:
    """A computed step in the cascade (release + its actual data)."""
    release: CascadeRelease
    release_date: str
    days_since: int
    decay_mult: float
    signal: str = 'NEUTRAL'            # classified signal (e.g. 'COOL', 'HOT', 'BEAT')
    surprise: float = 0.0              # raw surprise value (e.g. actual - consensus)
    us_move: float = 0.0               # US session move %
    us_dir: str = 'FLAT'               # 'DUMP', 'RALLY', 'FLAT'
    spike_pct: float = 0.0             # 1h post-release spike %
    spike_dir: str = 'FLAT'
    confidence: float = 0.5            # step confidence (0-1)
    details: dict = field(default_factory=dict)


@dataclass
class CascadeResult:
    """Full cascade scoring result."""
    cascade_name: str
    steps: List[CascadeStep]
    primary_signal: str                # e.g. 'COOL', 'HOT'
    confirmation_signal: str           # e.g. 'CONFIRMED', 'DENIED', 'NEUTRAL'
    combined_signal: str               # e.g. 'STRONG_BUY', 'SELL'
    score: float                       # 0.0-1.0
    confidence: str                    # 'HIGH', 'MEDIUM', 'LOW'
    expected_move: float               # expected ETH move %
    regime: str
    regime_sensitivity: float
    decay_mult: float                  # overall time decay
    details: dict = field(default_factory=dict)


# ═══════════════════════════════════════════════════════════════
# TIME DECAY
# ═══════════════════════════════════════════════════════════════

# Default decay schedule (days since release → multiplier)
DEFAULT_DECAY_SCHEDULE = {
    0: 1.00,    # release day
    1: 1.00,    # next day — full impact
    2: 0.70,    # digesting
    3: 0.70,
    4: 0.40,    # fading
    5: 0.40,
    6: 0.40,
    7: 0.40,
    8: 0.20,    # stale
    9: 0.20,
    10: 0.20,
    11: 0.20,
    12: 0.20,
    13: 0.20,
    14: 0.20,
}

DEFAULT_DECAY_FLOOR = 0.10  # minimum decay multiplier


def compute_decay(days_since: int, schedule: dict = None, floor: float = None) -> float:
    """Compute time decay multiplier for a release.

    Args:
        days_since: days since the release
        schedule: custom decay schedule (day → multiplier)
        floor: minimum multiplier (default: 0.10)

    Returns:
        decay multiplier (0.0-1.0)
    """
    if schedule is None:
        schedule = DEFAULT_DECAY_SCHEDULE
    if floor is None:
        floor = DEFAULT_DECAY_FLOOR

    if days_since <= 0:
        return 1.0

    # Find the matching bracket
    for day in sorted(schedule.keys(), reverse=True):
        if days_since >= day:
            return max(schedule[day], floor)

    return floor


# ═══════════════════════════════════════════════════════════════
# SESSION ANALYSIS HELPERS
# ═══════════════════════════════════════════════════════════════

# Session windows (UTC)
US_SESSION = {'start': (13, 30), 'end': (21, 0)}     # 8:30 AM - 4:00 PM ET
ASIA_SESSION = {'start': (0, 0), 'end': (8, 0)}       # next day 00:00 - 08:00 UTC
UK_SESSION = {'start': (7, 0), 'end': (16, 0)}        # 07:00 - 16:00 UTC


def _get_bars_between(df, start: datetime, end: datetime):
    """Get bars between two datetimes."""
    _ensure_pandas()
    if len(df) == 0:
        return df
    if isinstance(df['Open time'].iloc[0], str):
        mask = (pd.to_datetime(df['Open time']) >= start) & \
               (pd.to_datetime(df['Open time']) < end)
    else:
        mask = (df['Open time'] >= start) & (df['Open time'] < end)
    return df[mask]


def _get_bars_before(df, dt: datetime, n: int = 1):
    """Get last n bars before a datetime."""
    _ensure_pandas()
    if len(df) == 0:
        return None
    if isinstance(df['Open time'].iloc[0], str):
        mask = pd.to_datetime(df['Open time']) < dt
    else:
        mask = df['Open time'] < dt
    bars = df[mask]
    if len(bars) == 0:
        return None
    return bars.iloc[-n] if n == 1 else bars.tail(n)


def compute_us_session(df_15m, release_date: str,
                       release_hour: int = 13, release_minute: int = 30) -> Optional[dict]:
    """Compute US session move on release day."""
    rd = datetime.strptime(release_date, '%Y-%m-%d') if isinstance(release_date, str) else release_date
    release_dt = rd.replace(hour=release_hour, minute=release_minute)
    us_end = rd.replace(hour=US_SESSION['end'][0])

    pre_bar = _get_bars_before(df_15m, release_dt)
    if pre_bar is None:
        return None
    pre_price = float(pre_bar['Close'])

    us_bars = _get_bars_between(df_15m, release_dt, us_end)
    if len(us_bars) < 2:
        return None

    us_close = float(us_bars.iloc[-1]['Close'])
    us_high = float(us_bars['High'].max())
    us_low = float(us_bars['Low'].min())
    us_move = (us_close - pre_price) / pre_price * 100
    us_range = (us_high - us_low) / pre_price * 100

    return {
        'us_move': round(us_move, 3),
        'us_close': round(us_close, 2),
        'us_high': round(us_high, 2),
        'us_low': round(us_low, 2),
        'us_range': round(us_range, 3),
        'us_max_up': round((us_high - pre_price) / pre_price * 100, 3),
        'us_max_down': round((us_low - pre_price) / pre_price * 100, 3),
        'pre_price': round(pre_price, 2),
    }


def compute_1h_spike(df_15m, release_date: str,
                     release_hour: int = 13, release_minute: int = 30) -> Optional[dict]:
    """Compute the 1h post-release spike."""
    rd = datetime.strptime(release_date, '%Y-%m-%d') if isinstance(release_date, str) else release_date
    release_dt = rd.replace(hour=release_hour, minute=release_minute)

    pre_bar = _get_bars_before(df_15m, release_dt)
    if pre_bar is None:
        return None
    pre_price = float(pre_bar['Close'])

    first_1h = _get_bars_between(df_15m, release_dt, release_dt + timedelta(hours=1))
    if len(first_1h) < 2:
        return None

    spike_close = float(first_1h.iloc[-1]['Close'])
    spike_pct = (spike_close - pre_price) / pre_price * 100

    return {
        'spike_pct': round(spike_pct, 3),
        'spike_dir': 'UP' if spike_pct > 0.3 else 'DOWN' if spike_pct < -0.3 else 'FLAT',
        'spike_range': round((float(first_1h['High'].max()) - float(first_1h['Low'].min())) / pre_price * 100, 3),
        'pre_price': round(pre_price, 2),
        'spike_close': round(spike_close, 2),
    }


def compute_asia_session(df_15m, release_date: str,
                         release_hour: int = 13, release_minute: int = 30) -> Optional[dict]:
    """Compute Asia session on the day after release."""
    rd = datetime.strptime(release_date, '%Y-%m-%d') if isinstance(release_date, str) else release_date
    release_dt = rd.replace(hour=release_hour, minute=release_minute)
    us_end = rd.replace(hour=US_SESSION['end'][0])
    asia_start = (rd + timedelta(days=1)).replace(hour=ASIA_SESSION['start'][0])
    asia_end = (rd + timedelta(days=1)).replace(hour=ASIA_SESSION['end'][0])

    us_bars = _get_bars_between(df_15m, release_dt, us_end)
    if len(us_bars) == 0:
        return None
    us_close = float(us_bars.iloc[-1]['Close'])

    asia_bars = _get_bars_between(df_15m, asia_start, asia_end)
    if len(asia_bars) < 2:
        return None

    asia_open = float(asia_bars.iloc[0]['Open'])
    asia_close = float(asia_bars.iloc[-1]['Close'])
    asia_high = float(asia_bars['High'].max())
    asia_low = float(asia_bars['Low'].min())
    asia_move = (asia_close - us_close) / us_close * 100
    asia_gap = (asia_open - us_close) / us_close * 100
    asia_range = (asia_high - asia_low) / us_close * 100

    gap_dir = 'UP' if asia_gap > 0.2 else 'DOWN' if asia_gap < -0.2 else 'FLAT'

    # Sweep detection
    if gap_dir == 'DOWN':
        sweep_low_pct = (asia_low - us_close) / us_close * 100
        recovery_pct = (asia_close - asia_low) / (us_close - asia_low) * 100 if (us_close - asia_low) > 0 else 0
        sweep_depth_pct = abs(sweep_low_pct)
        is_sweep_reversal = sweep_depth_pct > 0.5 and recovery_pct > 50
        reclaimed_gap = asia_high >= us_close
    elif gap_dir == 'UP':
        sweep_high_pct = (asia_high - us_close) / us_close * 100
        recovery_pct = (asia_high - asia_close) / (asia_high - us_close) * 100 if (asia_high - us_close) > 0 else 0
        sweep_depth_pct = abs(sweep_high_pct)
        is_sweep_reversal = sweep_depth_pct > 0.5 and recovery_pct > 50
        reclaimed_gap = asia_low <= us_close
    else:
        sweep_depth_pct = 0
        recovery_pct = 0
        is_sweep_reversal = False
        reclaimed_gap = False

    return {
        'asia_move': round(asia_move, 3),
        'asia_gap': round(asia_gap, 3),
        'asia_open': round(asia_open, 2),
        'asia_close': round(asia_close, 2),
        'asia_high': round(asia_high, 2),
        'asia_low': round(asia_low, 2),
        'asia_range': round(asia_range, 3),
        'us_close_ref': round(us_close, 2),
        'gap_dir': gap_dir,
        'sweep_depth_pct': round(sweep_depth_pct, 3),
        'recovery_pct': round(recovery_pct, 1),
        'reclaimed_gap': reclaimed_gap,
        'is_sweep_reversal': is_sweep_reversal,
    }


def compute_uk_session(df_15m, release_date: str,
                       release_hour: int = 13, release_minute: int = 30) -> Optional[dict]:
    """Compute UK (London) session on the day after release."""
    rd = datetime.strptime(release_date, '%Y-%m-%d') if isinstance(release_date, str) else release_date
    release_dt = rd.replace(hour=release_hour, minute=release_minute)
    us_end = rd.replace(hour=US_SESSION['end'][0])

    uk_date = rd + timedelta(days=1)
    uk_start = uk_date.replace(hour=UK_SESSION['start'][0])
    uk_end = uk_date.replace(hour=UK_SESSION['end'][0])

    asia_start = uk_date.replace(hour=ASIA_SESSION['start'][0])
    asia_end = uk_date.replace(hour=ASIA_SESSION['end'][0])

    us_bars = _get_bars_between(df_15m, release_dt, us_end)
    if len(us_bars) == 0:
        return None
    us_close = float(us_bars.iloc[-1]['Close'])

    asia_bars = _get_bars_between(df_15m, asia_start, asia_end)
    if len(asia_bars) < 2:
        return None
    asia_close = float(asia_bars.iloc[-1]['Close'])
    asia_high = float(asia_bars['High'].max())
    asia_low = float(asia_bars['Low'].min())

    uk_bars = _get_bars_between(df_15m, uk_start, uk_end)
    if len(uk_bars) < 2:
        return None

    uk_open = float(uk_bars.iloc[0]['Open'])
    uk_close = float(uk_bars.iloc[-1]['Close'])
    uk_high = float(uk_bars['High'].max())
    uk_low = float(uk_bars['Low'].min())

    uk_move_vs_asia = (uk_close - asia_close) / asia_close * 100
    uk_move_vs_us = (uk_close - us_close) / us_close * 100
    uk_range = (uk_high - uk_low) / asia_close * 100

    asia_move = (asia_close - us_close) / us_close * 100
    asia_dir = 'UP' if asia_move > 0.3 else 'DOWN' if asia_move < -0.3 else 'FLAT'
    uk_dir = 'UP' if uk_move_vs_asia > 0.3 else 'DOWN' if uk_move_vs_asia < -0.3 else 'FLAT'

    uk_continued = (asia_dir == uk_dir) and asia_dir != 'FLAT'
    uk_faded = (asia_dir == 'UP' and uk_dir == 'DOWN') or \
               (asia_dir == 'DOWN' and uk_dir == 'UP')

    # Volume ratio
    uk_avg_vol = float(uk_bars['Volume'].mean()) if len(uk_bars) > 0 else 0
    asia_avg_vol = float(asia_bars['Volume'].mean()) if len(asia_bars) > 0 else 0
    vol_ratio = uk_avg_vol / asia_avg_vol if asia_avg_vol > 0 else 1.0

    # Taker flow
    uk_taker = 0.5
    if 'Taker buy base asset volume' in uk_bars.columns:
        taker_buy = float(uk_bars['Taker buy base asset volume'].sum())
        total_vol = float(uk_bars['Volume'].sum())
        uk_taker = taker_buy / total_vol if total_vol > 0 else 0.5

    return {
        'uk_move_vs_asia': round(uk_move_vs_asia, 3),
        'uk_move_vs_us': round(uk_move_vs_us, 3),
        'uk_open': round(uk_open, 2),
        'uk_close': round(uk_close, 2),
        'uk_high': round(uk_high, 2),
        'uk_low': round(uk_low, 2),
        'uk_range': round(uk_range, 3),
        'uk_direction': uk_dir,
        'uk_taker': round(uk_taker, 4),
        'uk_vol_ratio_vs_asia': round(vol_ratio, 2),
        'asia_close_ref': round(asia_close, 2),
        'asia_direction': asia_dir,
        'asia_move': round(asia_move, 3),
        'uk_continued_asia': uk_continued,
        'uk_faded_asia': uk_faded,
    }


# ═══════════════════════════════════════════════════════════════
# CASCADE ENGINE
# ═══════════════════════════════════════════════════════════════

class CascadeEngine:
    """Generic cascade scoring engine for macro release chains.

    A cascade is an ordered sequence of related macro releases where
    earlier releases inform the interpretation of later ones.

    Roles:
      LEADING     — precursor signal, released before PRIMARY (e.g. ADP before NFP)
      PRIMARY     — biggest market-moving release (e.g. NFP, CPI)
      CONFIRMATION — confirms or denies PRIMARY (e.g. PPI after CPI)
      BACKGROUND  — context only, extreme values matter (e.g. Claims)
      STRUCTURAL  — slow-moving, sets regime (e.g. JOLTS, GDP)
      POLICY      — central bank decision, the payoff event (e.g. FOMC, ECB)
      FOLLOWUP    — post-decision analysis (e.g. Presser, Minutes)
    """

    def __init__(self, name: str, releases: List[CascadeRelease],
                 confirmation_matrix: dict = None,
                 regime_sensitivity: dict = None,
                 decay_schedule: dict = None,
                 decay_floor: float = None,
                 default_spike_accuracy: dict = None,
                 combo_matrix: dict = None,
                 description: str = ''):
        """
        Args:
            name: cascade identifier (e.g. 'US_LABOR', 'US_INFLATION', 'CHINA_MACRO')
            releases: ordered list of CascadeRelease (chronological order in the cycle)
            confirmation_matrix: {(primary_signal, confirm_signal): (move_mod, conf_mod, desc)}
            regime_sensitivity: {regime_name: multiplier}
            decay_schedule: {days: multiplier} custom decay
            decay_floor: minimum decay multiplier
            default_spike_accuracy: {year: accuracy} for 1h spike predictions
            combo_matrix: {(signal_combo): (expected_move, confidence, fed_bias)}
            description: human-readable description
        """
        self.name = name
        self.releases = releases
        self.confirmation_matrix = confirmation_matrix or {}
        self.regime_sensitivity = regime_sensitivity or {}
        self.decay_schedule = decay_schedule or DEFAULT_DECAY_SCHEDULE
        self.decay_floor = decay_floor if decay_floor is not None else DEFAULT_DECAY_FLOOR
        self.default_spike_accuracy = default_spike_accuracy or {}
        self.combo_matrix = combo_matrix or {}
        self.description = description

        # Build lookup
        self._release_map = {r.name: r for r in releases}

    def get_release(self, name: str) -> Optional[CascadeRelease]:
        return self._release_map.get(name)

    def find_active_releases(self, date_str: str, config: dict = None,
                             lookback_days: int = 14) -> List[CascadeStep]:
        """Find all releases in this cascade that are active (today or recently released).

        Returns steps ordered by recency (most recent first).
        """
        cfg = config or {}
        steps = []
        today = datetime.strptime(date_str, '%Y-%m-%d')

        for release in self.releases:
            # Check if disabled
            if release.enabled_key and not cfg.get(release.enabled_key, True):
                continue

            # Check today first
            actual_date = None
            if release.is_release_day(date_str):
                actual_date = date_str
            else:
                # Look back for recent release
                actual_date = release.find_recent_release(date_str, max_lookback=lookback_days)

            if actual_date is None:
                continue

            release_dt = datetime.strptime(actual_date, '%Y-%m-%d')
            days_since = (today - release_dt).days

            # Compute decay
            decay_mult = compute_decay(days_since, self.decay_schedule, self.decay_floor)

            step = CascadeStep(
                release=release,
                release_date=actual_date,
                days_since=days_since,
                decay_mult=decay_mult,
            )
            steps.append(step)

        # Sort by role priority: PRIMARY > CONFIRMATION > LEADING > BACKGROUND > STRUCTURAL
        role_priority = {
            'PRIMARY': 0, 'POLICY': 0,
            'CONFIRMATION': 1, 'FOLLOWUP': 1,
            'LEADING': 2,
            'BACKGROUND': 3,
            'STRUCTURAL': 4,
        }
        steps.sort(key=lambda s: (role_priority.get(s.release.role, 5), s.days_since))

        return steps

    def classify_signal(self, step: CascadeStep, release_data: dict = None) -> str:
        """Classify the signal for a release step.

        Uses the release's signal_classifier if available, otherwise
        applies generic classification based on surprise value.
        """
        if step.release.signal_classifier and release_data:
            return step.release.signal_classifier(release_data)

        # Generic classification by surprise
        surprise = step.surprise
        if surprise >= 1.0:
            return 'STRONG_BEAT'
        elif surprise >= 0.3:
            return 'BEAT'
        elif surprise <= -1.0:
            return 'BIG_MISS'
        elif surprise <= -0.3:
            return 'MISS'
        return 'INLINE'

    def get_confirmation_modifier(self, primary_signal: str,
                                  confirm_signal: str) -> Tuple[float, float, str]:
        """Look up the confirmation modifier for a primary+confirmation pair.

        Returns: (move_modifier, confidence_modifier, description)
        """
        key = (primary_signal, confirm_signal)
        if key in self.confirmation_matrix:
            return self.confirmation_matrix[key]
        return (0.0, 0.0, 'No modifier')

    def compute_regime_sensitivity(self, regime: str) -> float:
        """Get the regime sensitivity multiplier."""
        return self.regime_sensitivity.get(regime, 0.80)

    def compute_cascade_score(self, steps: List[CascadeStep],
                              regime: str = 'UNKNOWN',
                              config: dict = None) -> CascadeResult:
        """Compute the final cascade score from active steps.

        Scoring algorithm:
        1. PRIMARY step sets the base signal and move
        2. CONFIRMATION steps modify the base (confirm/deny)
        3. LEADING steps provide pre-confirmation
        4. BACKGROUND steps apply context modifiers
        5. All steps are decayed by time since release
        6. Final score is adjusted by regime sensitivity

        Args:
            steps: active CascadeStep list (from find_active_releases)
            regime: current macro regime
            config: config dict

        Returns:
            CascadeResult with full analysis
        """
        cfg = config or {}

        if not steps:
            return CascadeResult(
                cascade_name=self.name,
                steps=[],
                primary_signal='NONE',
                confirmation_signal='NEUTRAL',
                combined_signal='NO_DATA',
                score=0.5,
                confidence='LOW',
                expected_move=0.0,
                regime=regime,
                regime_sensitivity=0.0,
                decay_mult=0.0,
            )

        # ── Step 1: Find PRIMARY (or POLICY) ──
        primary_step = None
        for s in steps:
            if s.release.role in ('PRIMARY', 'POLICY'):
                primary_step = s
                break

        # If no PRIMARY, use the highest-weight available step
        if primary_step is None:
            primary_step = max(steps, key=lambda s: s.release.weight)

        # ── Step 2: Base signal from PRIMARY ──
        base_signal = primary_step.signal
        base_move = primary_step.us_move if primary_step.us_move != 0 else primary_step.surprise
        base_confidence = primary_step.confidence

        # ── Step 3: Apply CONFIRMATION modifiers ──
        total_move_mod = 0.0
        total_conf_mod = 0.0
        confirmation_desc = 'NEUTRAL'

        for s in steps:
            if s.release.role in ('CONFIRMATION', 'FOLLOWUP') and s.days_since <= 3:
                move_mod, conf_mod, desc = self.get_confirmation_modifier(
                    base_signal, s.signal)
                # Apply decay to the modifier
                move_mod *= s.decay_mult
                conf_mod *= s.decay_mult
                total_move_mod += move_mod
                total_conf_mod += conf_mod
                if move_mod > 0.1:
                    confirmation_desc = 'CONFIRMED'
                elif move_mod < -0.1:
                    confirmation_desc = 'DENIED'

        # ── Step 4: Apply LEADING pre-confirmation ──
        for s in steps:
            if s.release.role == 'LEADING' and s.days_since <= 5:
                # Leading signals provide directional bias
                if s.signal in ('STRONG_BEAT', 'BEAT'):
                    total_move_mod += 0.20 * s.decay_mult
                elif s.signal in ('BIG_MISS', 'MISS'):
                    total_move_mod -= 0.20 * s.decay_mult

        # ── Step 5: Apply BACKGROUND context ──
        for s in steps:
            if s.release.role == 'BACKGROUND' and s.days_since <= 7:
                # Background only matters at extremes
                if s.signal in ('CRISIS', 'SPIKE'):
                    total_move_mod -= 0.50 * s.decay_mult
                elif s.signal in ('ELEVATED',):
                    total_move_mod -= 0.20 * s.decay_mult

        # ── Step 6: Combined signal ──
        total_move = base_move + total_move_mod
        final_conf = max(0.30, min(0.95, base_confidence + total_conf_mod))

        # Signal classification
        if total_move >= 3.0:
            combined_signal = 'STRONG_BUY'
        elif total_move >= 1.0:
            combined_signal = 'BUY'
        elif total_move >= -1.0:
            combined_signal = 'HOLD'
        elif total_move >= -3.0:
            combined_signal = 'SELL'
        else:
            combined_signal = 'STRONG_SELL'

        # Confidence label
        if final_conf >= 0.80:
            confidence = 'HIGH'
        elif final_conf >= 0.60:
            confidence = 'MEDIUM'
        else:
            confidence = 'LOW'

        # ── Step 7: Regime sensitivity ──
        regime_sens = self.compute_regime_sensitivity(regime)

        # ── Step 8: Overall decay (weighted average of step decays) ──
        total_weight = sum(s.release.weight for s in steps)
        if total_weight > 0:
            weighted_decay = sum(s.decay_mult * s.release.weight for s in steps) / total_weight
        else:
            weighted_decay = steps[0].decay_mult if steps else 1.0

        # ── Step 9: Final score ──
        # Base score from confidence
        base_score = {'HIGH': 0.70, 'MEDIUM': 0.60, 'LOW': 0.50}.get(confidence, 0.50)
        # Adjust by regime sensitivity
        score = base_score * regime_sens + (1 - regime_sens) * 0.50
        # Adjust by decay
        score = score * weighted_decay + 0.5 * (1 - weighted_decay)
        score = max(0.20, min(0.85, score))

        return CascadeResult(
            cascade_name=self.name,
            steps=steps,
            primary_signal=base_signal,
            confirmation_signal=confirmation_desc,
            combined_signal=combined_signal,
            score=round(score, 4),
            confidence=confidence,
            expected_move=round(total_move, 2),
            regime=regime,
            regime_sensitivity=round(regime_sens, 3),
            decay_mult=round(weighted_decay, 3),
            details={
                'base_move': round(base_move, 2),
                'confirmation_mod': round(total_move_mod, 2),
                'total_move': round(total_move, 2),
                'final_confidence': round(final_conf, 3),
                'step_count': len(steps),
                'primary_release': primary_step.release.name,
                'primary_date': primary_step.release_date,
                'primary_days_since': primary_step.days_since,
            },
        )

    def score(self, df_15m, current_time: datetime = None,
              config: dict = None, regime: str = 'UNKNOWN',
              release_data_map: dict = None) -> Tuple[str, float, dict, float]:
        """Main entry point — score this cascade for the current time.

        Args:
            df_15m: 15m OHLCV DataFrame
            current_time: datetime (default: now UTC)
            config: config dict
            regime: macro regime string
            release_data_map: {release_name: dict} with actual/consensus data for signal classification

        Returns:
            (status, score, details, decay_mult)
        """
        cfg = config or {}

        if current_time is None:
            current_time = datetime.utcnow()

        today_str = current_time.strftime('%Y-%m-%d')
        data_map = release_data_map or {}

        # Find active releases
        steps = self.find_active_releases(today_str, cfg)
        if not steps:
            return 'SKIP', 0.5, {'cascade': self.name, 'reason': 'No active releases'}, 1.0

        # Enrich steps with session data
        for step in steps:
            rd = step.release_date
            rh = step.release.release_hour_utc
            rm = step.release.release_minute_utc

            # Classify signal if data available
            release_data = data_map.get(step.release.name)
            if release_data:
                step.signal = self.classify_signal(step, release_data)
                step.surprise = release_data.get('surprise', 0.0)
            elif step.days_since == 0:
                # Release day — signal not yet known
                step.signal = 'PENDING'
                step.confidence = 0.40

            # Compute US session data (if available)
            us_data = compute_us_session(df_15m, rd, rh, rm)
            if us_data:
                step.us_move = us_data['us_move']
                step.us_dir = 'DUMP' if step.us_move < -0.5 else 'RALLY' if step.us_move > 0.5 else 'FLAT'
                step.details['us_data'] = us_data

            # Compute 1h spike
            spike_data = compute_1h_spike(df_15m, rd, rh, rm)
            if spike_data:
                step.spike_pct = spike_data['spike_pct']
                step.spike_dir = spike_data['spike_dir']
                step.details['spike_data'] = spike_data

            # Compute Asia session (post-release only)
            if step.days_since >= 1:
                asia_data = compute_asia_session(df_15m, rd, rh, rm)
                if asia_data:
                    step.details['asia_data'] = asia_data

                uk_data = compute_uk_session(df_15m, rd, rh, rm)
                if uk_data:
                    step.details['uk_data'] = uk_data

        # Compute cascade score
        result = self.compute_cascade_score(steps, regime, cfg)

        # Build details
        details = {
            'cascade': self.name,
            'description': self.description,
            'steps': [],
            'result': {
                'primary_signal': result.primary_signal,
                'confirmation_signal': result.confirmation_signal,
                'combined_signal': result.combined_signal,
                'expected_move': result.expected_move,
                'confidence': result.confidence,
                'regime': result.regime,
                'regime_sensitivity': result.regime_sensitivity,
            },
        }

        for step in steps:
            step_info = {
                'release': step.release.name,
                'role': step.release.role,
                'date': step.release_date,
                'days_since': step.days_since,
                'decay': step.decay_mult,
                'signal': step.signal,
                'surprise': step.surprise,
                'us_move': step.us_move,
                'us_dir': step.us_dir,
                'spike_pct': step.spike_pct,
                'spike_dir': step.spike_dir,
            }
            # Include session data if available
            for key in ('us_data', 'spike_data', 'asia_data', 'uk_data'):
                if key in step.details:
                    step_info[key] = step.details[key]
            details['steps'].append(step_info)

        return 'PASS', result.score, details, result.decay_mult


# ═══════════════════════════════════════════════════════════════
# FORMATTER
# ═══════════════════════════════════════════════════════════════

def format_cascade(details: dict) -> str:
    """Format cascade details for terminal output."""
    if not details:
        return ''

    cascade_name = details.get('cascade', '?')
    description = details.get('description', '')
    steps = details.get('steps', [])
    result = details.get('result', {})

    if not steps:
        return ''

    lines = []

    # Header
    combined = result.get('combined_signal', '?')
    expected = result.get('expected_move', 0)
    conf = result.get('confidence', '?')
    regime = result.get('regime', '?')
    sens = result.get('regime_sensitivity', 0)

    sig_icons = {
        'STRONG_BUY': '🟢🟢', 'BUY': '🟢', 'HOLD': '⚪',
        'SELL': '🔴', 'STRONG_SELL': '🔴🔴', 'NO_DATA': '⚫'
    }
    sig_icon = sig_icons.get(combined, '⚪')

    lines.append(f"\n  🔗 {cascade_name} CASCADE: {sig_icon} {combined}")
    lines.append(f"    {description}")
    lines.append(f"    Regime: {regime}  sensitivity={sens:.2f}  "
                 f"expected={expected:+.2f}%  conf={conf}")

    # Primary + Confirmation
    primary_sig = result.get('primary_signal', '?')
    confirm_sig = result.get('confirmation_signal', '?')
    lines.append(f"    Primary: {primary_sig}  Confirmation: {confirm_sig}")

    # Steps
    role_icons = {
        'LEADING': '🟡', 'PRIMARY': '🟢', 'CONFIRMATION': '🔵',
        'BACKGROUND': '⚪', 'STRUCTURAL': '🟤', 'POLICY': '🟣', 'FOLLOWUP': '🔵',
    }

    for step in steps:
        role = step.get('role', '?')
        name = step.get('release', '?')
        date = step.get('date', '?')
        days = step.get('days_since', 0)
        decay = step.get('decay', 1.0)
        signal = step.get('signal', '?')
        us_move = step.get('us_move', 0)
        us_dir = step.get('us_dir', '?')
        spike = step.get('spike_pct', 0)

        r_icon = role_icons.get(role, '⚪')
        decay_icon = '🟢' if decay >= 0.70 else '🟡' if decay >= 0.40 else '🟠' if decay >= 0.20 else '🔴'

        us_icon = '🔴' if us_dir == 'DUMP' else '🟢' if us_dir == 'RALLY' else '⚪'
        spike_icon = '🟢' if spike > 0.3 else '🔴' if spike < -0.3 else '⚪'

        line = (f"    {r_icon} {name:<16} {date}  "
                f"{decay_icon} decay={decay:.2f}  "
                f"signal={signal:<12}")
        if us_move != 0:
            line += f"  US={us_icon}{us_move:+.2f}%"
        if spike != 0:
            line += f"  spike={spike_icon}{spike:+.2f}%"
        if days > 0:
            line += f"  ({days}d ago)"

        lines.append(line)

        # Show Asia data if available
        asia = step.get('asia_data')
        if asia:
            asia_move = asia.get('asia_move', 0)
            gap_dir = asia.get('gap_dir', '?')
            pattern = 'sweep' if asia.get('is_sweep_reversal') else 'held' if gap_dir != 'FLAT' else 'flat'
            asia_icon = '🟢' if asia_move > 0 else '🔴'
            lines.append(f"      ↳ Asia: {asia_icon} {asia_move:+.2f}%  gap={gap_dir}  pattern={pattern}")

        # Show UK data if available
        uk = step.get('uk_data')
        if uk:
            uk_move = uk.get('uk_move_vs_asia', 0)
            uk_dir = uk.get('uk_direction', '?')
            uk_icon = '🟢' if uk_move > 0.3 else '🔴' if uk_move < -0.3 else '⚪'
            cont = 'continued' if uk.get('uk_continued_asia') else 'faded' if uk.get('uk_faded_asia') else 'neutral'
            lines.append(f"      ↳ UK:   {uk_icon} {uk_move:+.2f}% vs Asia  ({cont})")

    return '\n'.join(lines)

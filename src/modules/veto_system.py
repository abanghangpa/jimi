"""Veto Priority System — Hard/soft vetoes override ICS scoring."""


class VetoResult:
    """Result of veto evaluation."""
    def __init__(self):
        self.allowed = True
        self.hard_vetoes = []
        self.soft_vetoes = []
        self.warnings = []

    def add_hard_veto(self, module, reason):
        self.allowed = False
        self.hard_vetoes.append({'module': module, 'reason': reason})

    def add_soft_veto(self, module, reason, penalty=0.10):
        self.soft_vetoes.append({'module': module, 'reason': reason, 'penalty': penalty})

    def add_warning(self, module, reason):
        self.warnings.append({'module': module, 'reason': reason})

    @property
    def hard_blocked(self):
        return len(self.hard_vetoes) > 0

    @property
    def soft_penalty(self):
        return sum(v['penalty'] for v in self.soft_vetoes)

    def summary(self):
        parts = []
        if self.hard_vetoes:
            parts.append(f"HARD BLOCK: {', '.join(v['module'] for v in self.hard_vetoes)}")
        if self.soft_vetoes:
            parts.append(f"Soft vetoes: {', '.join(v['module'] for v in self.soft_vetoes)}")
        if self.warnings:
            parts.append(f"Warnings: {len(self.warnings)}")
        return ' | '.join(parts) if parts else 'ALL CLEAR'


def evaluate_vetoes(
    config,
    vol_regime='NEUTRAL',
    data_fresh=True,
    data_age_minutes=0,
    monthly_dd_hit=False,
    dir_veto=False,
    m9_status='SKIP',
    m10_status='SKIP',
    m11_status='SKIP',
):
    """Evaluate all veto conditions and return a VetoResult."""
    result = VetoResult()

    if not config.get('VETO_ENABLED', False):
        return result

    if config.get('VETO_STALE_DATA_HARD', True) and not data_fresh:
        result.add_hard_veto('DATA', f'stale_data_{data_age_minutes:.0f}min')

    if vol_regime == 'CRISIS' and config.get('VETO_CRISIS_HARD', True):
        result.add_hard_veto('M9', 'crisis_regime')
    elif vol_regime == 'CHOP_HARD':
        result.add_hard_veto('M9', 'chop_hard_regime')
    elif vol_regime in ('CHOP_MILD', 'CHOP_MILD_BEAR', 'CHOP_MILD_BULL'):
        if config.get('VETO_CHOP_MILD_HARD', False):
            result.add_hard_veto('M9', 'chop_mild_regime_hard')
        else:
            result.add_soft_veto('M9', 'chop_mild_regime', penalty=0.10)

    if monthly_dd_hit and config.get('VETO_MONTHLY_DD_HARD', True):
        result.add_hard_veto('RISK', 'monthly_dd_circuit_breaker')

    if dir_veto and config.get('VETO_DIR_CONFLICT_HARD', True):
        result.add_hard_veto('DIR', 'm4_m5_both_against_direction')

    if m10_status == 'FAIL':
        result.add_soft_veto('M10', 'macro_unfavorable', penalty=0.05)
    if m11_status == 'FAIL':
        result.add_soft_veto('M11', 'momentum_divergence_against', penalty=0.06)

    return result


def check_data_freshness(df_deriv=None, max_age_minutes=20, current_time=None):
    """Check if derivatives data is fresh enough to trade on."""
    import pandas as pd
    details = {}

    if df_deriv is None or len(df_deriv) == 0:
        details['reason'] = 'no_derivatives_data'
        return False, float('inf'), details

    if 'timestamp' in df_deriv.columns:
        last_ts = df_deriv['timestamp'].iloc[-1]
    else:
        details['reason'] = 'no_timestamp_column'
        return False, float('inf'), details

    if current_time is not None:
        now = pd.Timestamp(current_time)
    else:
        now = pd.Timestamp.now()

    last_ts = pd.Timestamp(last_ts)
    age_minutes = (now - last_ts).total_seconds() / 60

    details['last_timestamp'] = str(last_ts)
    details['current_time'] = str(now)
    details['age_minutes'] = round(age_minutes, 2)
    details['max_age'] = max_age_minutes

    if age_minutes > max_age_minutes:
        details['reason'] = f'stale_data_{age_minutes:.0f}min_old'
        details['status'] = 'STALE'
        return False, age_minutes, details

    if len(df_deriv) >= 3:
        last_3_ts = df_deriv['timestamp'].iloc[-3:].values
        if len(set(str(t) for t in last_3_ts)) == 1:
            details['reason'] = 'ffill_zombie_identical_timestamps'
            details['status'] = 'ZOMBIE'
            return False, age_minutes, details

    details['status'] = 'FRESH'
    return True, age_minutes, details


def check_data_continuity(df_deriv, max_gap_minutes=30):
    """Check for gaps in derivatives data timeline.
    Returns: (is_continuous: bool, gaps: list)
    """
    import pandas as pd
    if df_deriv is None or len(df_deriv) < 2:
        return True, []

    if 'timestamp' not in df_deriv.columns:
        return True, []

    ts = pd.to_datetime(df_deriv['timestamp'])
    diffs = ts.diff().dt.total_seconds() / 60

    gaps = []
    for i, diff in enumerate(diffs):
        if diff > max_gap_minutes:
            gaps.append({
                'index': i,
                'gap_minutes': round(diff, 1),
                'from': str(ts.iloc[i-1]) if i > 0 else 'N/A',
                'to': str(ts.iloc[i]),
            })

    return len(gaps) == 0, gaps

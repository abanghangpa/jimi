"""
M22 Aggregator: Macro Regime Classifier

Reads outputs from M23-M65 modules and classifies the macro regime.
M22 does NOT independently fetch data — it is a pure downstream aggregator.

Architecture:
    M45 (Core PCE)  ──┐
    M56 (US CPI)    ──┤── Inflation signal
    M60 (US PPI)    ──┘
    M37 (NFP)       ──┐
    M62 (Unemployment)─┤── Labor signal
    M61 (Claims)    ──┘
    M57 (FOMC)      ──┐
    M58 (Powell)    ──┤── Policy signal
    M59 (Minutes)   ──┘
    M23 (Release dynamics) ── Session context
    M47-M52 (Global CBs)   ── Global policy

    → M22 = regime label + severity + factors

Output:
    status: 'PASS' or 'SKIP'
    score: 0.0-1.0 (for ICS — derived from regime)
    details: {regime, severity, factors, inflation_signal, labor_signal, policy_signal}
"""

from src.config import CONFIG


def _extract_module(result, key):
    """Safely extract a module's result dict."""
    m = result.get(key, {})
    if not m or m.get('status') in ('ERROR', None):
        return None
    return m


def _classify_inflation(result):
    """Aggregate inflation signal from M45, M56, M60.

    Returns: (signal: float, label: str, factors: list)
    signal: -1.0 (deflationary) to +1.0 (inflationary)
    """
    signals = []
    factors = []

    # M45: Core PCE (Fed's actual target)
    m45 = _extract_module(result, 'm45')
    if m45:
        pce_yoy = m45.get('core_pce_yoy')
        infl = m45.get('inflation', '')
        if pce_yoy is not None:
            if pce_yoy >= 3.0:
                signals.append(0.8)
                factors.append(f'PCE {pce_yoy:.1f}% HOT')
            elif pce_yoy >= 2.5:
                signals.append(0.3)
                factors.append(f'PCE {pce_yoy:.1f}% WARM')
            elif pce_yoy >= 2.0:
                signals.append(0.0)
                factors.append(f'PCE {pce_yoy:.1f}% TARGET')
            else:
                signals.append(-0.5)
                factors.append(f'PCE {pce_yoy:.1f}% COOL')

    # M56: US CPI
    m56 = _extract_module(result, 'm56')
    if m56:
        cpi_yoy = m56.get('cpi_yoy')
        bias = m56.get('bias', '')
        if cpi_yoy is not None:
            if cpi_yoy >= 3.5:
                signals.append(0.7)
                factors.append(f'CPI {cpi_yoy:.1f}% HOT')
            elif cpi_yoy >= 2.5:
                signals.append(0.2)
            elif cpi_yoy >= 2.0:
                signals.append(-0.1)
            else:
                signals.append(-0.6)
                factors.append(f'CPI {cpi_yoy:.1f}% COOL')

    # M60: US PPI (leading indicator)
    m60 = _extract_module(result, 'm60')
    if m60:
        ppi_yoy = m60.get('ppi_yoy')
        ppi_signal = m60.get('ppi_signal', '')
        if ppi_yoy is not None:
            if ppi_yoy >= 4.0:
                signals.append(0.6)
                factors.append(f'PPI {ppi_yoy:.1f}% HOT (pipeline)')
            elif ppi_yoy >= 3.0:
                signals.append(0.3)
            elif ppi_yoy < 2.0:
                signals.append(-0.4)

    if not signals:
        return 0.0, 'NO_DATA', factors

    avg = sum(signals) / len(signals)
    if avg >= 0.5:
        label = 'INFLATION_HOT'
    elif avg >= 0.2:
        label = 'INFLATION_WARM'
    elif avg >= -0.2:
        label = 'TARGET_RANGE'
    elif avg >= -0.5:
        label = 'DISINFLATION'
    else:
        label = 'DEFLATION_RISK'

    return avg, label, factors


def _classify_labor(result):
    """Aggregate labor signal from M37, M62, M61.

    Returns: (signal: float, label: str, factors: list)
    signal: -1.0 (crisis) to +1.0 (goldilocks)
    """
    signals = []
    factors = []

    # M37: NFP
    m37 = _extract_module(result, 'm37')
    if m37:
        nfp_k = m37.get('nfp_k')
        surprise = m37.get('surprise', 0)
        if nfp_k is not None:
            if nfp_k > 200:
                signals.append(0.6)
                factors.append(f'NFP {nfp_k}K strong')
            elif nfp_k > 100:
                signals.append(0.1)
            else:
                signals.append(-0.5)
                factors.append(f'NFP {nfp_k}K weak')
        if abs(surprise) > 50:
            direction = 'beat' if surprise > 0 else 'miss'
            factors.append(f'NFP {direction} by {abs(surprise):.0f}K')

    # M62: Unemployment Rate
    m62 = _extract_module(result, 'm62')
    if m62:
        unemp = m62.get('unemp_rate')
        sahm = m62.get('sahm_triggered', False)
        unemp_signal = m62.get('unemp_signal', '')
        if unemp is not None:
            if unemp < 4.0:
                signals.append(0.5)
            elif unemp < 4.5:
                signals.append(0.0)
            elif unemp < 5.0:
                signals.append(-0.4)
                factors.append(f'Unemployment {unemp:.1f}% softening')
            else:
                signals.append(-0.8)
                factors.append(f'Unemployment {unemp:.1f}% danger')
        if sahm:
            signals.append(-0.7)
            factors.append('Sahm Rule TRIGGERED')

    # M61: Claims
    m61 = _extract_module(result, 'm61')
    if m61:
        claims_signal = m61.get('claims_signal', '')
        claims_k = m61.get('claims_k')
        claims_trend = m61.get('claims_trend', '')
        if claims_k is not None:
            if claims_k < 210:
                signals.append(0.3)
            elif claims_k < 240:
                signals.append(0.0)
            elif claims_k < 280:
                signals.append(-0.3)
                factors.append(f'Claims {claims_k}K elevated')
            else:
                signals.append(-0.7)
                factors.append(f'Claims {claims_k}K spike')

    if not signals:
        return 0.0, 'NO_DATA', factors

    avg = sum(signals) / len(signals)
    if avg >= 0.4:
        label = 'GOLDILOCKS'
    elif avg >= 0.0:
        label = 'NORMAL'
    elif avg >= -0.3:
        label = 'SOFTENING'
    elif avg >= -0.6:
        label = 'WEAKENING'
    else:
        label = 'CRISIS'

    return avg, label, factors


def _classify_policy(result):
    """Aggregate policy signal from M57, M58, M59.

    Returns: (signal: float, label: str, factors: list)
    signal: -1.0 (hawkish/tightening) to +1.0 (dovish/easing)
    """
    signals = []
    factors = []

    # M57: FOMC Rate Decision
    m57 = _extract_module(result, 'm57')
    if m57:
        rate = m57.get('rate')
        rate_action = m57.get('rate_action', '')
        stance = m57.get('stance', '')
        bias = m57.get('bias', '')
        if stance == 'DOVISH' or rate_action == 'CUT':
            signals.append(0.5)
            factors.append(f'FOMC dovish')
        elif stance == 'HAWKISH' or rate_action == 'HIKE':
            signals.append(-0.5)
            factors.append(f'FOMC hawkish')
        else:
            signals.append(0.0)

    # M58: Powell Press Conference
    m58 = _extract_module(result, 'm58')
    if m58:
        tone = m58.get('powell_tone', '')
        bias = m58.get('bias', '')
        if tone == 'DOVISH':
            signals.append(0.4)
            factors.append('Powell dovish')
        elif tone == 'HAWKISH':
            signals.append(-0.4)
            factors.append('Powell hawkish')

    # M59: FOMC Minutes
    m59 = _extract_module(result, 'm59')
    if m59:
        surprise = m59.get('minutes_surprise', '')
        bias = m59.get('bias', '')
        if surprise == 'DOVISH':
            signals.append(0.3)
        elif surprise == 'HAWKISH':
            signals.append(-0.3)

    if not signals:
        return 0.0, 'NO_DATA', factors

    avg = sum(signals) / len(signals)
    if avg >= 0.3:
        label = 'EASING'
    elif avg >= -0.1:
        label = 'HOLDING'
    else:
        label = 'TIGHTENING'

    return avg, label, factors


def _classify_global(result):
    """Aggregate global CB signal from M47-M52.

    Returns: (signal: float, label: str, factors: list)
    """
    signals = []
    factors = []

    for key, name in [('m47', 'BoJ'), ('m48', 'ECB'), ('m49', 'BoE'), ('m52', 'RBA')]:
        m = _extract_module(result, key)
        if m:
            bias = m.get('bias', '')
            rate_change = m.get('rate_change', 0)
            if bias == 'LONG':
                signals.append(0.3)
            elif bias == 'SHORT':
                signals.append(-0.3)
            if rate_change and rate_change != 0:
                direction = 'cut' if rate_change < 0 else 'hike'
                factors.append(f'{name} {direction}')

    if not signals:
        return 0.0, 'NO_DATA', factors

    avg = sum(signals) / len(signals)
    if avg >= 0.2:
        label = 'GLOBAL_EASING'
    elif avg >= -0.2:
        label = 'GLOBAL_NEUTRAL'
    else:
        label = 'GLOBAL_TIGHTENING'

    return avg, label, factors


def _classify_session(result):
    """Read M23 session dynamics for context.

    Returns: dict with session context, or None.
    """
    m23 = _extract_module(result, 'm23')
    if not m23:
        return None

    return {
        'regime': m23.get('regime', '?'),
        'release_type': m23.get('release_type'),
        'us_direction': m23.get('us_direction'),
        'fade_rate': m23.get('fade_rate'),
        'claims_today': m23.get('claims_today', False),
    }


def _determine_regime(infl_signal, infl_label, labor_signal, labor_label,
                       policy_signal, policy_label, global_signal, global_label):
    """Determine the macro regime from aggregated signals.

    Returns: (regime: str, severity: str, score: float)
    """
    # Stagflation: hot inflation + weak labor or holding policy
    if infl_signal >= 0.4 and labor_signal < -0.2:
        return 'STAGFLATION', 'HIGH', 0.30
    if infl_signal >= 0.4 and policy_label == 'HOLDING':
        return 'STAGFLATION_LITE', 'MEDIUM', 0.35

    # Goldilocks: cool inflation + strong labor
    if infl_signal <= -0.2 and labor_signal >= 0.3:
        return 'GOLDILOCKS', 'LOW', 0.85

    # Disinflation: cooling inflation + normal labor
    if infl_signal <= -0.2 and labor_signal >= -0.2:
        return 'DISINFLATION', 'LOW', 0.75

    # Growth scare: cooling inflation + weakening labor
    if infl_signal <= -0.2 and labor_signal < -0.2:
        return 'GROWTH_SCARE', 'MEDIUM', 0.50

    # Inflation shock: hot inflation + tightening policy
    if infl_signal >= 0.4 and policy_label == 'TIGHTENING':
        return 'INFLATION_SHOCK', 'HIGH', 0.25

    # Reflation: rising inflation + easing policy
    if infl_signal >= 0.2 and policy_label == 'EASING':
        return 'REFLATION', 'LOW', 0.70

    # Policy error: tightening into weakness
    if policy_label == 'TIGHTENING' and labor_signal < -0.2:
        return 'POLICY_ERROR', 'HIGH', 0.30

    # Easing cycle: dovish policy + moderate inflation
    if policy_label == 'EASING' and infl_signal < 0.4:
        return 'EASING_CYCLE', 'LOW', 0.70

    # Neutral
    return 'NEUTRAL', 'LOW', 0.55


def aggregate_macro_regime(result, config=None):
    """Classify macro regime by aggregating M23-M65 outputs.

    Args:
        result: scanner result dict (with all module outputs)
        config: Config dict

    Returns:
        status: 'PASS' or 'SKIP'
        score: 0.0-1.0 (for ICS)
        details: dict with full regime breakdown
    """
    cfg = config or CONFIG

    if not cfg.get('M22_ENABLED', False):
        return 'SKIP', 0.5, {'regime': 'DISABLED'}

    # Aggregate signals from module outputs
    infl_signal, infl_label, infl_factors = _classify_inflation(result)
    labor_signal, labor_label, labor_factors = _classify_labor(result)
    policy_signal, policy_label, policy_factors = _classify_policy(result)
    global_signal, global_label, global_factors = _classify_global(result)
    session_ctx = _classify_session(result)

    # Determine regime
    regime, severity, base_score = _determine_regime(
        infl_signal, infl_label, labor_signal, labor_label,
        policy_signal, policy_label, global_signal, global_label)

    # Severity multiplier for ICS
    sev_mult = {'LOW': 1.0, 'MEDIUM': 0.85, 'HIGH': 0.70, 'CRITICAL': 0.50}
    size_mult = sev_mult.get(severity, 1.0)

    # Collect all factors
    all_factors = []
    all_factors.extend(infl_factors)
    all_factors.extend(labor_factors)
    all_factors.extend(policy_factors)
    all_factors.extend(global_factors)

    # Direction: score > 0.5 = bullish macro, < 0.5 = bearish
    score = base_score

    # Session context adjustment
    if session_ctx and session_ctx.get('regime') == 'CLAIMS_RELEASE':
        # Claims release day — slight bearish bias if claims elevated
        if labor_signal < -0.2:
            score = max(0.05, score - 0.05)
            all_factors.append('Claims day + weak labor → slight bearish')

    details = {
        'regime': regime,
        'severity': severity,
        'score': round(score, 3),
        'size_mult': round(size_mult, 2),

        # Component signals
        'inflation_signal': round(infl_signal, 3),
        'inflation_label': infl_label,
        'labor_signal': round(labor_signal, 3),
        'labor_label': labor_label,
        'policy_signal': round(policy_signal, 3),
        'policy_label': policy_label,
        'global_signal': round(global_signal, 3),
        'global_label': global_label,

        # Session context
        'session': session_ctx,

        # Narrative
        'factors': all_factors,
    }

    return 'PASS', score, details


def format_m22_aggregated(details):
    """Format aggregated M22 details for terminal output."""
    if not details or details.get('regime') in ('DISABLED', 'NO_DATA'):
        return ''

    regime = details.get('regime', '?')
    severity = details.get('severity', '?')
    score = details.get('score', 0.5)
    size_mult = details.get('size_mult', 1.0)
    factors = details.get('factors', [])

    sev_icons = {'LOW': '🟢', 'MEDIUM': '🟡', 'HIGH': '🟠', 'CRITICAL': '🔴'}
    icon = sev_icons.get(severity, '⚪')

    infl = details.get('inflation_label', '?')
    labor = details.get('labor_label', '?')
    policy = details.get('policy_label', '?')
    glob = details.get('global_label', '?')

    infl_sc = details.get('inflation_signal', 0)
    labor_sc = details.get('labor_signal', 0)
    policy_sc = details.get('policy_signal', 0)

    lines = []
    lines.append(f"\n  {icon} M22 MACRO REGIME: {regime}")
    lines.append(f"    Inflation: {infl} ({infl_sc:+.2f})  |  Labor: {labor} ({labor_sc:+.2f})  |  Policy: {policy} ({policy_sc:+.2f})")
    if glob != 'NO_DATA':
        lines.append(f"    Global CBs: {glob}")
    lines.append(f"    Score: {score:.3f}  |  Severity: {severity}  |  Size: {size_mult:.2f}x")

    # Session context
    session = details.get('session')
    if session:
        rel_type = session.get('release_type')
        if rel_type:
            us_dir = session.get('us_direction', '?')
            lines.append(f"    Session: {rel_type} release, US={us_dir}")

    for f in factors:
        lines.append(f"    • {f}")

    return '\n'.join(lines)

"""
Meta Cascade — Aggregates all regional cascades into a unified global score.

This is the top-level orchestrator that runs all cascades and produces
a combined global macro score. It respects cascade weights and handles
the case where multiple cascades are active simultaneously.

Cascade weights (relative importance for crypto):
  US_INFLATION:  0.35  (FOMC is the single biggest macro catalyst)
  US_LABOR:      0.25  (NFP is 2nd biggest, feeds into Fed policy)
  US_ACTIVITY:   0.10  (ISM confirms economic health)
  CHINA_MACRO:   0.10  (2nd largest economy, APAC sentiment)
  EU_MACRO:      0.08  (ECB decisions, EUR/USD dynamics)
  UK_MACRO:      0.05  (BoE decisions, smaller crypto impact)
  JAPAN_MACRO:   0.05  (carry trade dynamics, outsized surprise risk)
  AU_MACRO:      0.02  (APAC sentiment, smaller impact)

Usage:
    from src.modules.cascade_meta import score_all_cascades, format_all_cascades
    results = score_all_cascades(df_15m, current_time, config, regime)
"""

from datetime import datetime
from src.modules.cascade_engine import format_cascade


# ═══════════════════════════════════════════════════════════════
# CASCADE REGISTRY
# ═══════════════════════════════════════════════════════════════

CASCADE_REGISTRY = [
    {
        'name': 'US_INFLATION',
        'weight': 0.35,
        'score_fn': None,  # lazy loaded
        'format_fn': None,
        'module': 'src.modules.cascade_inflation',
        'score_attr': 'score_inflation_cascade',
        'format_attr': 'format_inflation_cascade',
    },
    {
        'name': 'US_LABOR',
        'weight': 0.25,
        'score_fn': None,
        'format_fn': None,
        'module': 'src.modules.cascade_labor',
        'score_attr': 'score_labor_cascade',
        'format_attr': 'format_labor_cascade',
    },
    {
        'name': 'US_ACTIVITY',
        'weight': 0.10,
        'score_fn': None,
        'format_fn': None,
        'module': 'src.modules.cascade_us_activity',
        'score_attr': 'score_us_activity_cascade',
        'format_attr': 'format_us_activity_cascade',
    },
    {
        'name': 'CHINA_MACRO',
        'weight': 0.10,
        'score_fn': None,
        'format_fn': None,
        'module': 'src.modules.cascade_china',
        'score_attr': 'score_china_cascade',
        'format_attr': 'format_china_cascade',
    },
    {
        'name': 'EU_MACRO',
        'weight': 0.08,
        'score_fn': None,
        'format_fn': None,
        'module': 'src.modules.cascade_eu',
        'score_attr': 'score_eu_cascade',
        'format_attr': 'format_eu_cascade',
    },
    {
        'name': 'UK_MACRO',
        'weight': 0.05,
        'score_fn': None,
        'format_fn': None,
        'module': 'src.modules.cascade_uk',
        'score_attr': 'score_uk_cascade',
        'format_attr': 'format_uk_cascade',
    },
    {
        'name': 'JAPAN_MACRO',
        'weight': 0.05,
        'score_fn': None,
        'format_fn': None,
        'module': 'src.modules.cascade_japan',
        'score_attr': 'score_japan_cascade',
        'format_attr': 'format_japan_cascade',
    },
    {
        'name': 'AU_MACRO',
        'weight': 0.02,
        'score_fn': None,
        'format_fn': None,
        'module': 'src.modules.cascade_au',
        'score_attr': 'score_au_cascade',
        'format_attr': 'format_au_cascade',
    },
]


def _lazy_load(entry):
    """Lazy load a cascade module."""
    if entry['score_fn'] is not None:
        return

    import importlib
    mod = importlib.import_module(entry['module'])
    entry['score_fn'] = getattr(mod, entry['score_attr'])
    entry['format_fn'] = getattr(mod, entry['format_attr'])


def score_all_cascades(df_15m, current_time=None, config=None, regime='UNKNOWN',
                       release_data_map=None, enabled_cascades=None):
    """Score all registered cascades and produce a combined result.

    Args:
        df_15m: 15m OHLCV DataFrame
        current_time: datetime (default: now UTC)
        config: config dict
        regime: macro regime string
        release_data_map: {cascade_name: {release_name: data}} per-cascade release data
        enabled_cascades: set of cascade names to run (None = all)

    Returns:
        dict with:
            combined_score: weighted average score (0-1)
            combined_signal: 'BUY', 'SELL', 'HOLD', etc.
            active_cascades: list of active cascade results
            all_results: dict of all cascade results (including SKIP)
    """
    if current_time is None:
        current_time = datetime.utcnow()

    if release_data_map is None:
        release_data_map = {}

    all_results = {}
    active_results = []
    total_weight = 0.0
    weighted_score = 0.0

    for entry in CASCADE_REGISTRY:
        name = entry['name']

        if enabled_cascades and name not in enabled_cascades:
            continue

        # Check config enable
        cfg = config or {}
        cfg_key = f'CASCADE_{name}_ENABLED'
        if not cfg.get(cfg_key, True):
            all_results[name] = {'status': 'DISABLED', 'score': 0.5}
            continue

        _lazy_load(entry)

        try:
            cascade_data_map = release_data_map.get(name, {})
            status, score, details, decay = entry['score_fn'](
                df_15m, current_time, config, regime, cascade_data_map)

            result = {
                'status': status,
                'score': round(score, 4),
                'decay': round(decay, 3),
                'weight': entry['weight'],
                'details': details,
            }
            all_results[name] = result

            if status == 'PASS':
                active_results.append({
                    'name': name,
                    'score': score,
                    'weight': entry['weight'],
                    'decay': decay,
                    'signal': details.get('result', {}).get('combined_signal', '?'),
                    'expected_move': details.get('result', {}).get('expected_move', 0),
                    'confidence': details.get('result', {}).get('confidence', 'LOW'),
                    'details': details,
                })
                weighted_score += score * entry['weight']
                total_weight += entry['weight']

        except Exception as e:
            all_results[name] = {'status': 'ERROR', 'score': 0.5, 'error': str(e)}

    # Compute combined score
    if total_weight > 0:
        combined_score = weighted_score / total_weight
    else:
        combined_score = 0.5

    # Signal classification
    if combined_score >= 0.70:
        combined_signal = 'STRONG_BUY'
    elif combined_score >= 0.60:
        combined_signal = 'BUY'
    elif combined_score >= 0.45:
        combined_signal = 'HOLD'
    elif combined_score >= 0.35:
        combined_signal = 'SELL'
    else:
        combined_signal = 'STRONG_SELL'

    return {
        'combined_score': round(combined_score, 4),
        'combined_signal': combined_signal,
        'active_cascades': sorted(active_results, key=lambda x: x['weight'], reverse=True),
        'all_results': all_results,
        'active_count': len(active_results),
        'total_weight': round(total_weight, 3),
    }


def format_all_cascades(results: dict) -> str:
    """Format all cascade results for terminal output."""
    if not results:
        return ''

    lines = []

    combined_score = results.get('combined_score', 0.5)
    combined_signal = results.get('combined_signal', '?')
    active_count = results.get('active_count', 0)
    total_weight = results.get('total_weight', 0)

    sig_icons = {
        'STRONG_BUY': '🟢🟢', 'BUY': '🟢', 'HOLD': '⚪',
        'SELL': '🔴', 'STRONG_SELL': '🔴🔴',
    }
    sig_icon = sig_icons.get(combined_signal, '⚪')

    lines.append(f"\n{'='*60}")
    lines.append(f"  🌍 GLOBAL MACRO CASCADE: {sig_icon} {combined_signal}")
    lines.append(f"    Score: {combined_score:.3f}  Active: {active_count}  "
                 f"Weight coverage: {total_weight:.0%}")
    lines.append(f"{'='*60}")

    # Active cascades (sorted by weight)
    for cascade in results.get('active_cascades', []):
        name = cascade['name']
        score = cascade['score']
        weight = cascade['weight']
        signal = cascade['signal']
        expected = cascade['expected_move']
        conf = cascade['confidence']
        details = cascade['details']

        sig_icon = sig_icons.get(signal, '⚪')
        conf_icon = {'HIGH': '🟢', 'MEDIUM': '🟡', 'LOW': '🔴'}.get(conf, '⚪')

        lines.append(f"\n  {'─'*50}")
        lines.append(f"  {name} ({weight:.0%} weight) — {sig_icon} {signal}")
        lines.append(f"    Score: {score:.3f}  Expected: {expected:+.2f}%  "
                     f"Confidence: {conf_icon} {conf}")

        # Format cascade details
        entry = None
        for e in CASCADE_REGISTRY:
            if e['name'] == name:
                entry = e
                break
        if entry and entry['format_fn']:
            formatted = entry['format_fn'](details)
            if formatted:
                # Indent the cascade details
                for line in formatted.split('\n'):
                    lines.append(f"  {line}")

    # Show inactive cascades briefly
    inactive = []
    for name, result in results.get('all_results', {}).items():
        if result.get('status') in ('SKIP', 'DISABLED'):
            inactive.append(name)

    if inactive:
        lines.append(f"\n  {'─'*50}")
        lines.append(f"  Inactive: {', '.join(inactive)} (no releases active)")

    return '\n'.join(lines)

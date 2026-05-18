"""
M65: China NBS Monthly Activity Data — Session Bias (regime-conditional)

NBS releases monthly activity data (~15th-16th) at 10:00 Beijing (02:00 UTC).
Bundle: Industrial Production, Retail Sales, FAI, Unemployment, House Prices.

Backtested: 87 releases (2018-2026), ETH/USDT 15m.

24h aggregate: +0.169% avg, 46% win — NOT significant (p=0.73).
Session transmission chain: BROKEN — every transition <55% same direction.
No reliable session-by-session persistence.

Edge combos (regime-conditional):
  MARKUP + CRISIS + INLINE:        +8.02% avg, 75% win, n=4  → LONG
  MARKDOWN + COMPRESSING + INLINE: -5.86% avg, 33% win, n=3  → SHORT
  ACCUMULATION + CRISIS + INLINE:  +2.41% avg, 50% win, n=6  → LONG (marginal)
  MARKDOWN + NEUTRAL + INLINE:     -2.30% avg, 14% win, n=7  → SHORT
  DISTRIBUTION + TRENDING + MISS:  -1.75% avg, 0% win, n=3   → SHORT

Statistical tests:
  Miss vs Beat: t=-1.188, p=0.246 (NOT significant)
  FAI Contraction vs Expansion: t=-0.838, p=0.404 (NOT significant)

Usage:
    from src.modules.m65_china_activity import score_m65_china_activity, format_m65
"""

from datetime import datetime


# ═══════════════════════════════════════════════════════════════
# RELEASE DATA (reuse from m_china_activity.py — kept DRY)
# ═══════════════════════════════════════════════════════════════

def _get_releases():
    """Lazy import to avoid circular deps."""
    from src.modules.m_china_activity import CHINA_ACTIVITY_RELEASES
    return CHINA_ACTIVITY_RELEASES


# ═══════════════════════════════════════════════════════════════
# SIGNAL CLASSIFIER
# ═══════════════════════════════════════════════════════════════

def _classify_bundle(data):
    """Classify the activity bundle into a composite signal."""
    from src.modules.m_china_activity import classify_activity_bundle
    return classify_activity_bundle(data)


# ═══════════════════════════════════════════════════════════════
# REGIME × SIGNAL EDGE TABLE
# ═══════════════════════════════════════════════════════════════
# Only combos with n≥3 and |avg|≥0.5% from backtest.
# Format: (wyckoff, vol, composite) → (direction, avg_ret, win_rate, n, confidence)

EDGE_TABLE = {
    ('MARKUP', 'CRISIS', 'INLINE'):           ('LONG',  +8.02, 0.75, 4, 'MEDIUM'),
    ('MARKDOWN', 'COMPRESSING', 'INLINE'):    ('SHORT', -5.86, 0.33, 3, 'LOW'),
    ('ACCUMULATION', 'CRISIS', 'INLINE'):     ('LONG',  +2.41, 0.50, 6, 'LOW'),
    ('MARKDOWN', 'NEUTRAL', 'INLINE'):        ('SHORT', -2.30, 0.14, 7, 'LOW'),
    ('MARKDOWN', 'TRENDING', 'INLINE'):       ('SHORT', -1.53, 0.29, 7, 'LOW'),
    ('DISTRIBUTION', 'TRENDING', 'MISS'):     ('SHORT', -1.75, 0.00, 3, 'LOW'),
    ('DISTRIBUTION', 'TRENDING', 'INLINE'):   ('SHORT', -0.92, 0.40, 5, 'LOW'),
    ('MARKUP', 'TRENDING', 'INLINE'):         ('LONG',  +1.42, 0.67, 6, 'LOW'),
    ('MARKUP', 'COMPRESSING', 'INLINE'):      ('SHORT', -0.96, 0.00, 3, 'LOW'),
}

# Score adjustment by composite signal (fallback when no regime combo matches)
COMPOSITE_SCORE_ADJ = {
    'STRONG_BEAT':  +0.04,
    'BEAT':         +0.02,
    'MILD_BEAT':    +0.01,
    'INLINE':        0.00,
    'MILD_MISS':    -0.01,
    'MISS':         -0.03,
    'BIG_MISS':     -0.05,
    'PROPERTY_CRISIS': -0.08,
}

# Size multiplier by composite signal
COMPOSITE_SIZE_MULT = {
    'STRONG_BEAT':  1.00,
    'BEAT':         1.00,
    'MILD_BEAT':    1.00,
    'INLINE':       1.00,
    'MILD_MISS':    0.90,
    'MISS':         0.85,
    'BIG_MISS':     0.75,
    'PROPERTY_CRISIS': 0.60,
}


# ═══════════════════════════════════════════════════════════════
# SCORING
# ═══════════════════════════════════════════════════════════════

def score_m65_china_activity(today_str, wyckoff_phase='UNKNOWN',
                              vol_regime='UNKNOWN', direction='NEUTRAL',
                              config=None):
    """Score China activity data for today.

    Args:
        today_str: 'YYYY-MM-DD' date string
        wyckoff_phase: from M21 (ACCUMULATION/MARKUP/DISTRIBUTION/MARKDOWN/RANGE)
        vol_regime: from M9 (COMPRESSING/NEUTRAL/TRENDING/CRISIS)
        direction: resolved trade direction (LONG/SHORT/NEUTRAL)
        config: settings dict

    Returns:
        (status, score_adj, size_mult, details)
    """
    cfg = config or {}
    if not cfg.get('M65_ENABLED', True):
        return 'DISABLED', 0.0, 1.0, {'regime': 'DISABLED'}

    releases = _get_releases()

    # Check if today is a release day (or within window)
    window = cfg.get('M65_WINDOW_DAYS', 1)
    release_date = None
    release_data = None

    for delta in range(window + 1):
        check_date = datetime.strptime(today_str, '%Y-%m-%d')
        from datetime import timedelta
        check_str = (check_date - timedelta(days=delta)).strftime('%Y-%m-%d')
        if check_str in releases:
            d = releases[check_str]
            if d.get('ip_yoy') is not None or d.get('retail_yoy') is not None:
                release_date = check_str
                release_data = d
                break

    if release_data is None:
        return 'NOT_RELEASE_DAY', 0.0, 1.0, {'regime': 'NOT_RELEASE_DAY'}

    # Classify
    classification = _classify_bundle(release_data)
    composite = classification['composite']
    comp_score = classification['weighted_score']

    # Check edge table for regime combo
    wy_norm = wyckoff_phase.upper() if wyckoff_phase else 'UNKNOWN'
    vol_norm = vol_regime.upper() if vol_regime else 'UNKNOWN'

    edge_key = (wy_norm, vol_norm, composite)
    edge = EDGE_TABLE.get(edge_key)

    if edge:
        edge_dir, edge_avg, edge_wr, edge_n, edge_conf = edge
        # Direction agreement
        if direction == edge_dir or direction == 'NEUTRAL':
            score_adj = COMPOSITE_SCORE_ADJ.get(composite, 0.0)
            # Boost if regime combo has strong edge
            if abs(edge_avg) >= 3.0 and edge_n >= 3:
                score_adj *= 1.5  # amplify strong regime edges
        else:
            # Direction disagrees with edge — neutralize
            score_adj = 0.0
    else:
        # No regime combo — use composite-only adjustment
        score_adj = COMPOSITE_SCORE_ADJ.get(composite, 0.0)
        edge_dir = None
        edge_avg = 0.0
        edge_wr = 0.0
        edge_n = 0
        edge_conf = 'NONE'

    size_mult = COMPOSITE_SIZE_MULT.get(composite, 1.0)

    # Cap adjustments
    score_adj = max(-0.10, min(0.10, score_adj))

    details = {
        'regime': composite,
        'composite': composite,
        'weighted_score': comp_score,
        'release_date': release_date,
        'wyckoff': wy_norm,
        'vol': vol_norm,
        'edge_direction': edge_dir,
        'edge_avg_ret': edge_avg,
        'edge_win_rate': edge_wr,
        'edge_n': edge_n,
        'edge_confidence': edge_conf,
        'ip': release_data.get('ip_yoy'),
        'retail': release_data.get('retail_yoy'),
        'fai': release_data.get('fai_ytd_yoy'),
        'house': release_data.get('house_price_yoy'),
        'unemp': release_data.get('unemp'),
        'ip_c': release_data.get('consensus_ip'),
        'retail_c': release_data.get('consensus_retail'),
        'classification': classification,
    }

    status = 'PASS' if abs(score_adj) >= 0.01 else 'NO_EDGE'
    return status, score_adj, size_mult, details


# ═══════════════════════════════════════════════════════════════
# FORMATTER
# ═══════════════════════════════════════════════════════════════

def format_m65(details):
    """Format M65 output for terminal."""
    if not details or details.get('regime') in ('DISABLED', 'NOT_RELEASE_DAY'):
        return ''

    lines = []
    composite = details.get('composite', '?')
    comp_score = details.get('weighted_score', 0)
    release_date = details.get('release_date', '?')

    composite_icons = {
        'STRONG_BEAT': '🟢🟢', 'BEAT': '🟢', 'MILD_BEAT': '🟢',
        'INLINE': '⚪', 'MIXED': '🟡', 'MILD_MISS': '🟡',
        'MISS': '🔴', 'BIG_MISS': '🔴🔴', 'PROPERTY_CRISIS': '🚨',
    }
    icon = composite_icons.get(composite, '⚪')

    bias = details.get('edge_direction')
    bias_icon = '🟢' if bias == 'LONG' else '🔴' if bias == 'SHORT' else '⚪'
    bias_str = f"{bias_icon} {bias}" if bias else '⚪ NO_EDGE'

    lines.append(f"  🇨🇳 M65 CHINA ACTIVITY ({release_date}):")
    lines.append(f"    Composite: {icon} {composite} (score={comp_score:+.2f})  bias={bias_str}")

    ip = details.get('ip')
    retail = details.get('retail')
    fai = details.get('fai')
    ip_c = details.get('ip_c')
    retail_c = details.get('retail_c')

    if ip is not None:
        ip_surp = ip - (ip_c or ip)
        ip_icon = '🟢' if ip_surp > 0.2 else '🔴' if ip_surp < -0.2 else '⚪'
        lines.append(f"    IP:     {ip_icon} {ip:+.1f}% YoY  (cons {ip_c:+.1f}%)  surp={ip_surp:+.1f}%")
    if retail is not None:
        ret_surp = retail - (retail_c or retail)
        ret_icon = '🟢' if ret_surp > 0.2 else '🔴' if ret_surp < -0.2 else '⚪'
        lines.append(f"    Retail: {ret_icon} {retail:+.1f}% YoY  (cons {retail_c:+.1f}%)  surp={ret_surp:+.1f}%")
    if fai is not None:
        fai_icon = '🟢' if fai > 0 else '🔴'
        lines.append(f"    FAI:    {fai_icon} {fai:+.1f}% YoY")

    edge_avg = details.get('edge_avg_ret', 0)
    edge_wr = details.get('edge_win_rate', 0)
    edge_n = details.get('edge_n', 0)
    edge_conf = details.get('edge_confidence', 'NONE')

    if edge_n >= 3:
        conf_icon = '🟢' if edge_conf == 'HIGH' else '🟡' if edge_conf == 'MEDIUM' else '🟠'
        lines.append(f"    Edge:   {conf_icon} {edge_avg:+.2f}% avg, {edge_wr*100:.0f}% win, n={edge_n}  conf={edge_conf}")
    else:
        lines.append(f"    Edge:   ⚪ No regime combo match (using composite-only adj)")

    return '\n'.join(lines)

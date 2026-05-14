"""
M19: Breakout Confirmation Filters — Multi-signal gate for squeeze entries.

When a squeeze is detected (M18 PENDING/TRIGGERED), these filters check
real-time market micro-structure to distinguish genuine breakouts from
fakeouts/liquidity grabs.

Designed for the CVD-divergence + bullish-squeeze conflict pattern:
  - CVD says "distribution" (bearish divergence)
  - Squeeze says "breakout incoming" (bullish)
  - Resolution: wait for the breakout bar itself to prove demand

Filters (all checked at breakout time, not before):
  1. CVD Flip       — intrabar CVD must flip bullish on breakout bar
  2. BTC Confluence — BTC must not be dumping (-2%+ hourly)
  3. Volume Surge   — breakout bar vol >= 1.5x MA20
  4. OI Expansion   — OI must rise >= 0.3%/hr on breakout
  5. Liquidity Hold — if $level swept, must hold 3+ bars
  6. Spot Flow      — majority exchanges must show buyers
  7. Funding Stay   — funding must stay negative (shorts paying) for long breakouts

Each filter returns pass/fail + detail string.
Composite score = fraction of filters passed.
Threshold: >= 5/7 (71%) = CONFIRMED, 4/7 = WEAK, <4 = REJECTED
"""

import numpy as np


# ── Default thresholds ──
BREAKOUT_CONFIRM_DEFAULTS = {
    'BREAKOUT_CVD_ENABLED': True,
    'BREAKOUT_BTC_ENABLED': True,
    'BREAKOUT_VOL_ENABLED': True,
    'BREAKOUT_OI_ENABLED': True,
    'BREAKOUT_LIQ_HOLD_ENABLED': True,
    'BREAKOUT_SPOT_ENABLED': True,
    'BREAKOUT_FUNDING_ENABLED': True,

    # Volume surge threshold (breakout bar vs MA20)
    'BREAKOUT_VOL_SURGE_MULT': 1.5,

    # OI expansion threshold (% per hour)
    'BREAKOUT_OI_ROC_MIN': 0.3,

    # BTC max decline to block entry (% hourly)
    'BTC_MAX_DECLINE_PCT': -2.0,

    # Liquidity hold: bars price must hold above/below swept level
    'BREAKOUT_LIQ_HOLD_BARS': 3,

    # Spot flow: min fraction of exchanges showing buyers
    'BREAKOUT_SPOT_BUY_MIN': 0.5,  # 50%+ exchanges must be buyers

    # Funding: for LONG, funding must be <= this (shorts paying)
    'BREAKOUT_FUNDING_MAX_LONG': 0.0005,   # 0.05%
    # For SHORT, funding must be >= this (longs paying)
    'BREAKOUT_FUNDING_MIN_SHORT': -0.0005, # -0.05%

    # Composite thresholds
    'BREAKOUT_CONFIRM_MIN': 5,    # 5/7 = CONFIRMED
    'BREAKOUT_WEAK_MIN': 4,       # 4/7 = WEAK (reduced size)
    'BREAKOUT_REJECT_BELOW': 4,   # <4 = REJECTED
}


def check_breakout_filters(result, df_15m=None, btc_df=None, config=None):
    """Run all breakout confirmation filters.

    Args:
        result: scanner scan_signal result dict (has derivatives, exchange_activity, etc.)
        df_15m: 15m DataFrame (for intrabar CVD check)
        btc_df: BTC DataFrame (optional, for BTC confluence)
        config: config dict with thresholds

    Returns:
        dict with:
            status: 'CONFIRMED' | 'WEAK' | 'REJECTED'
            passed: int (how many filters passed)
            total: int (how many filters checked)
            score: float (passed/total)
            filters: list of {name, passed, detail} dicts
    """
    cfg = {**BREAKOUT_CONFIRM_DEFAULTS, **(config or {})}

    squeeze = result.get('squeeze', {})
    direction = squeeze.get('direction', 'NEUTRAL')
    entry_triggered = squeeze.get('entry_triggered', False)

    # Only run if squeeze has a direction
    if direction == 'NEUTRAL':
        return _make_result('NEUTRAL', 0, 0, [], 'no squeeze direction')

    filters = []

    # ── Filter 1: CVD Flip ──
    if cfg.get('BREAKOUT_CVD_ENABLED', True):
        f = _check_cvd_flip(result, df_15m, direction, cfg)
        filters.append(f)

    # ── Filter 2: BTC Confluence ──
    if cfg.get('BREAKOUT_BTC_ENABLED', True):
        f = _check_btc_confluence(result, btc_df, direction, cfg)
        filters.append(f)

    # ── Filter 3: Volume Surge ──
    if cfg.get('BREAKOUT_VOL_ENABLED', True):
        f = _check_volume_surge(result, df_15m, cfg)
        filters.append(f)

    # ── Filter 4: OI Expansion ──
    if cfg.get('BREAKOUT_OI_ENABLED', True):
        f = _check_oi_expansion(result, cfg)
        filters.append(f)

    # ── Filter 5: Liquidity Hold ──
    if cfg.get('BREAKOUT_LIQ_HOLD_ENABLED', True):
        f = _check_liquidity_hold(result, df_15m, direction, cfg)
        filters.append(f)

    # ── Filter 6: Spot Flow ──
    if cfg.get('BREAKOUT_SPOT_ENABLED', True):
        f = _check_spot_flow(result, direction, cfg)
        filters.append(f)

    # ── Filter 7: Funding Stay ──
    if cfg.get('BREAKOUT_FUNDING_ENABLED', True):
        f = _check_funding_stay(result, direction, cfg)
        filters.append(f)

    # Composite
    passed = sum(1 for f in filters if f['passed'])
    total = len(filters)

    if total == 0:
        return _make_result('NEUTRAL', 0, 0, [], 'no filters enabled')

    if passed >= cfg['BREAKOUT_CONFIRM_MIN']:
        status = 'CONFIRMED'
    elif passed >= cfg['BREAKOUT_WEAK_MIN']:
        status = 'WEAK'
    else:
        status = 'REJECTED'

    return _make_result(status, passed, total, filters, None)


def _check_cvd_flip(result, df_15m, direction, cfg):
    """Filter 1: CVD must flip in trade direction on/just before breakout bar.

    Checks:
    - M4b intrabar divergence: if was BEARISH, must now be BULLISH or NONE
    - M4 layer_a_div: if was BEARISH, must have weakened
    - Current bar taker ratio: must show buyers (>0.52) for LONG
    """
    m4 = result.get('m4', {})
    m4_div = m4.get('div', {})
    m4b = result.get('m4b', {})
    taker = result.get('raw_taker_ratio', 0.5)

    m4_div_str = 'NONE'
    if isinstance(m4_div, dict):
        m4_div_str = m4_div.get('layer_a_div', 'NONE')
    # v7.2: Normalize _BASE variants
    if m4_div_str.endswith('_BASE'):
        m4_div_str = m4_div_str.replace('_BASE', '')

    m4b_div = m4b.get('divergence', 'NONE')

    if direction == 'LONG':
        # CVD was bearish — needs to flip or weaken
        m4_flipped = m4_div_str in ('NONE', 'BULLISH')
        m4b_flipped = m4b_div in ('NONE', 'BULLISH')
        taker_bullish = taker > 0.52

        # Pass if: both CVD layers flipped OR taker confirms
        passed = (m4_flipped and m4b_flipped) or (taker_bullish and m4b_flipped)

        if passed:
            detail = f"CVD flipped: M4={m4_div_str} M4b={m4b_div} taker={taker:.3f}"
        else:
            detail = f"CVD still bearish: M4={m4_div_str} M4b={m4b_div} taker={taker:.3f}"
    else:
        m4_flipped = m4_div_str in ('NONE', 'BEARISH')
        m4b_flipped = m4b_div in ('NONE', 'BEARISH')
        taker_bearish = taker < 0.48

        passed = (m4_flipped and m4b_flipped) or (taker_bearish and m4b_flipped)

        if passed:
            detail = f"CVD flipped: M4={m4_div_str} M4b={m4b_div} taker={taker:.3f}"
        else:
            detail = f"CVD still bullish: M4={m4_div_str} M4b={m4b_div} taker={taker:.3f}"

    return {'name': 'CVD Flip', 'passed': passed, 'detail': detail}


def _check_btc_confluence(result, btc_df, direction, cfg):
    """Filter 2: BTC must not be dumping against trade direction.

    Uses M10 macro data (already computed in scanner) or raw BTC data.
    """
    m10 = result.get('m10', {})
    m10_details = m10.get('details', {})

    # Use BTC ROC from M10 if available
    btc_roc_str = m10_details.get('btc_roc7', '')
    if btc_roc_str:
        try:
            btc_roc7 = float(btc_roc_str.strip('%')) / 100 if isinstance(btc_roc_str, str) else float(btc_roc_str)
        except (ValueError, AttributeError):
            btc_roc7 = 0
    else:
        btc_roc7 = 0

    # Also check BTC trend component from M10
    btc_trend = m10_details.get('m10_components', {}).get('btc_trend', 0.5)

    max_decline = cfg['BTC_MAX_DECLINE_PCT'] / 100

    if direction == 'LONG':
        # BTC must not be in freefall
        passed = btc_roc7 > max_decline and btc_trend > 0.3
        if passed:
            detail = f"BTC 7d ROC={btc_roc7:+.1%} trend={btc_trend:.2f} (stable/bullish)"
        else:
            detail = f"BTC dumping: 7d ROC={btc_roc7:+.1%} trend={btc_trend:.2f}"
    else:
        passed = btc_roc7 < -max_decline and btc_trend < 0.7
        if passed:
            detail = f"BTC 7d ROC={btc_roc7:+.1%} trend={btc_trend:.2f} (stable/bearish)"
        else:
            detail = f"BTC not bearish: 7d ROC={btc_roc7:+.1%} trend={btc_trend:.2f}"

    return {'name': 'BTC Confluence', 'passed': passed, 'detail': detail}


def _check_volume_surge(result, df_15m, cfg):
    """Filter 3: Breakout bar must have volume >= 1.5x MA20.

    Uses current bar volume vs vol_ma20 (already computed in scanner).
    """
    vol_ratio = result.get('vol_trend', 1.0)  # current_vol / vol_ma20
    bar_vol_spike = result.get('bar_vol_spike', 1.0)
    threshold = cfg['BREAKOUT_VOL_SURGE_MULT']

    # Use whichever is higher (they should be the same but different calc paths)
    effective_ratio = max(vol_ratio, bar_vol_spike)

    passed = effective_ratio >= threshold

    if passed:
        detail = f"Volume {effective_ratio:.2f}x MA20 (threshold {threshold:.1f}x) ✅"
    else:
        detail = f"Volume {effective_ratio:.2f}x MA20 — need {threshold:.1f}x for breakout"

    return {'name': 'Volume Surge', 'passed': passed, 'detail': detail}


def _check_oi_expansion(result, cfg):
    """Filter 4: OI must expand on breakout (new positions, not just short covering).

    Rising OI + rising price = real demand (LONG breakout).
    Falling OI + rising price = short covering (temporary, fades).
    """
    deriv = result.get('derivatives', {})
    oi_roc = deriv.get('oi_roc_1h', 0)  # % change per hour
    threshold = cfg['BREAKOUT_OI_ROC_MIN']

    squeeze = result.get('squeeze', {})
    direction = squeeze.get('direction', 'NEUTRAL')

    if direction == 'LONG':
        # OI should rise (new longs entering) or at least stay flat
        # Falling OI = short covering = temporary bounce
        passed = oi_roc >= -0.1  # allow slight decline but not dumping
        if oi_roc >= threshold:
            detail = f"OI rising {oi_roc:+.3f}%/hr (new longs entering) ✅"
        elif oi_roc >= 0:
            detail = f"OI flat {oi_roc:+.3f}%/hr (neutral)"
        else:
            detail = f"OI declining {oi_roc:+.3f}%/hr — may be short covering"
            passed = False
    else:
        passed = oi_roc >= -0.1
        if oi_roc >= threshold:
            detail = f"OI rising {oi_roc:+.3f}%/hr (new shorts entering) ✅"
        else:
            detail = f"OI {oi_roc:+.3f}%/hr — weak conviction"

    return {'name': 'OI Expansion', 'passed': passed, 'detail': detail}


def _check_liquidity_hold(result, df_15m, direction, cfg):
    """Filter 5: After liquidity sweep, price must hold above/below for N bars.

    Checks the unswept magnets — if the nearest magnet was just swept,
    price must hold beyond it (not snap back).
    """
    magnets = result.get('magnets', [])
    price = result.get('price', 0)
    hold_bars = cfg['BREAKOUT_LIQ_HOLD_BARS']

    if not magnets or df_15m is None or len(df_15m) < hold_bars + 1:
        return {'name': 'Liquidity Hold', 'passed': True,
                'detail': 'no magnets or insufficient data (auto-pass)'}

    # Find nearest unswept magnet in trade direction
    if direction == 'LONG':
        targets = [(p, s, swept) for p, s, swept, *_ in magnets
                   if p > price and not swept]
    else:
        targets = [(p, s, swept) for p, s, swept, *_ in magnets
                   if p < price and not swept]

    if not targets:
        # No unswept magnets in direction — auto-pass
        return {'name': 'Liquidity Hold', 'passed': True,
                'detail': 'no unswept magnets in direction (auto-pass)'}

    # Check if any recently swept magnet was held
    recently_swept = [(p, s, swept_at) for p, s, swept, swept_at, *_ in magnets
                      if swept and swept_at is not None]

    if recently_swept:
        # Check if price held beyond the most recently swept level
        nearest = min(recently_swept, key=lambda x: abs(x[0] - price))
        swept_price = nearest[0]

        # Check last N bars
        recent_closes = df_15m['Close'].iloc[-hold_bars:].values.astype(float)

        if direction == 'LONG':
            # Price must have stayed above swept level
            held = all(c > swept_price for c in recent_closes)
            detail = (f"Swept ${swept_price:.0f}: "
                     f"{'held above' if held else 'failed to hold'} "
                     f"for {hold_bars} bars")
        else:
            held = all(c < swept_price for c in recent_closes)
            detail = (f"Swept ${swept_price:.0f}: "
                     f"{'held below' if held else 'failed to hold'} "
                     f"for {hold_bars} bars")

        return {'name': 'Liquidity Hold', 'passed': held, 'detail': detail}

    # No recently swept — auto-pass
    return {'name': 'Liquidity Hold', 'passed': True,
            'detail': 'no recently swept levels (auto-pass)'}


def _check_spot_flow(result, direction, cfg):
    """Filter 6: Spot market flow must support trade direction.

    Checks per-exchange spot flow (buyers vs sellers).
    At least BREAKOUT_SPOT_BUY_MIN fraction must show buyers for LONG.
    """
    exch = result.get('exchange_activity', {})
    spot_sigs = exch.get('spot_signals', {})

    if not spot_sigs:
        return {'name': 'Spot Flow', 'passed': True,
                'detail': 'no spot data (auto-pass)'}

    # Count exchanges by flow direction
    flow_by_exch = {
        'binance': spot_sigs.get('spot_flow_binance', '?'),
        'okx': spot_sigs.get('spot_flow_okx', '?'),
        'bybit': spot_sigs.get('spot_flow_bybit', '?'),
        'coinbase': spot_sigs.get('spot_flow_coinbase', '?'),
    }

    # Weight by volume (binance has 53% of spot volume)
    vol_shares = spot_sigs.get('spot_vol_shares', {})
    spot_flow = spot_sigs.get('spot_flow', '?')

    buy_fraction = 0
    total_weight = 0
    for exch_name, flow in flow_by_exch.items():
        weight = vol_shares.get(exch_name, 0.1)
        total_weight += weight
        if direction == 'LONG' and flow == 'BUYERS':
            buy_fraction += weight
        elif direction == 'SHORT' and flow == 'SELLERS':
            buy_fraction += weight

    if total_weight > 0:
        buy_pct = buy_fraction / total_weight
    else:
        buy_pct = 0.5

    threshold = cfg['BREAKOUT_SPOT_BUY_MIN']
    passed = buy_pct >= threshold

    # Detail
    buyer_exchs = [k for k, v in flow_by_exch.items() if v == 'BUYERS']
    seller_exchs = [k for k, v in flow_by_exch.items() if v == 'SELLERS']

    if passed:
        detail = (f"Spot flow: {buy_pct:.0%} weighted {direction}-aligned "
                 f"(buyers: {', '.join(buyer_exchs)})")
    else:
        detail = (f"Spot flow: {buy_pct:.0%} weighted {direction}-aligned — "
                 f"sellers on {', '.join(seller_exchs)}")

    return {'name': 'Spot Flow', 'passed': passed, 'detail': detail}


def _check_funding_stay(result, direction, cfg):
    """Filter 7: Funding rate must support squeeze direction.

    For LONG breakout: funding should be negative (shorts paying, fuels squeeze).
    For SHORT breakout: funding should be positive (longs paying, fuels squeeze).
    """
    deriv = result.get('derivatives', {})
    funding = deriv.get('funding_rate', 0)

    if funding is None:
        return {'name': 'Funding Stay', 'passed': True,
                'detail': 'no funding data (auto-pass)'}

    if direction == 'LONG':
        threshold = cfg['BREAKOUT_FUNDING_MAX_LONG']
        # Negative funding = shorts paying = fuel for long squeeze
        passed = funding <= threshold
        if passed:
            fr_label = "shorts paying" if funding < 0 else "neutral"
            detail = f"Funding {funding*100:+.4f}% ({fr_label} — fuels squeeze)"
        else:
            detail = f"Funding {funding*100:+.4f}% (longs paying — reduces squeeze pressure)"
    else:
        threshold = cfg['BREAKOUT_FUNDING_MIN_SHORT']
        passed = funding >= threshold
        if passed:
            fr_label = "longs paying" if funding > 0 else "neutral"
            detail = f"Funding {funding*100:+.4f}% ({fr_label} — fuels squeeze)"
        else:
            detail = f"Funding {funding*100:+.4f}% (shorts paying — reduces squeeze pressure)"

    return {'name': 'Funding Stay', 'passed': passed, 'detail': detail}


def _make_result(status, passed, total, filters, reason):
    """Build the return dict."""
    score = passed / total if total > 0 else 0
    return {
        'status': status,
        'passed': passed,
        'total': total,
        'score': round(score, 3),
        'filters': filters,
        'reason': reason,
    }


def format_breakout_confirm(bc):
    """Format breakout confirmation for terminal output."""
    if bc.get('reason'):
        return ''

    status = bc.get('status', '?')
    passed = bc.get('passed', 0)
    total = bc.get('total', 0)

    icons = {'CONFIRMED': '✅', 'WEAK': '⚠️', 'REJECTED': '❌'}
    icon = icons.get(status, '❓')

    lines = ['', f'  🔒 BREAKOUT CONFIRMATION ({passed}/{total}) {icon} {status}']

    for f in bc.get('filters', []):
        check = '✅' if f['passed'] else '❌'
        lines.append(f'    {check} {f["name"]}: {f["detail"]}')

    if status == 'CONFIRMED':
        lines.append(f'  → ✅ {passed}/{total} filters passed — breakout confirmed')
    elif status == 'WEAK':
        lines.append(f'  → ⚠️ {passed}/{total} filters passed — reduced size recommended')
    else:
        lines.append(f'  → ❌ {passed}/{total} filters passed — breakout likely fake')

    return '\n'.join(lines)

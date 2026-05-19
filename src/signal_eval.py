"""
Signal Evaluator — post-generation commentary

Evaluates a signal AFTER it was generated to assess whether the entry
is still actionable. Commentary only — does NOT gate or block signals.

Usage:
    from src.signal_eval import evaluate_signal, format_signal_eval
    eval_result = evaluate_signal(result, current_price)
    print(format_signal_eval(eval_result))

M66-M73 Integration:
    Two levels of tradfi integration:
    Level 1 — Passive: reads M66-M73 scores from the signal result dict
              and adds commentary about tradfi headwinds/tailwinds.
    Level 2 — Active: re-fetches live tradfi data and compares against
              the original signal's scores to detect regime shifts.
"""


# ── M66-M73 classification → human label mapping ──
_TRADFI_LABELS = {
    # M66 USD/JPY
    'CARRY_ALERT': ('⚠️', 'USD/JPY carry unwind alert'),
    'CARRY_UNWIND_CONFIRMED': ('🔴', 'USD/JPY carry unwind confirmed'),
    'USD_WEAKNESS': ('🟡', 'USD weakness (not carry)'),
    # M67 DXY
    'CONFIRMED_BEARISH': ('🔴', 'DXY↑ + ETH↓ — confirmed bearish'),
    'BULLISH_DIVERGENCE': ('🟢', 'DXY↑ but ETH holding — bullish divergence'),
    'CONFIRMED_BULLISH': ('🟢', 'DXY↓ + ETH↑ — confirmed bullish'),
    'BEARISH_DIVERGENCE': ('🔴', 'DXY↓ but ETH weak — bearish divergence'),
    # M68 Yield
    'YIELD_EXTREME': ('🔴🔴', '10Y yield extreme spike'),
    'INFLATION_SPIKE': ('🔴', '10Y yield spike (inflation-driven)'),
    'GROWTH_SPIKE': ('🟡', '10Y yield spike (growth-driven)'),
    'YIELD_SPIKE': ('🟠', '10Y yield spike'),
    # M69 VIX
    'COMPLACENT': ('🟡', 'VIX complacent — squeeze risk'),
    'ELEVATED': ('🟠', 'VIX elevated'),
    'FEAR': ('🔴', 'VIX fear zone'),
    'FEAR_EASING': ('🟡', 'VIX fear + DXY falling (easing)'),
    'FEAR_TIGHTENING': ('🔴', 'VIX fear + DXY rising (tightening)'),
    'CRISIS_LIQUIDITY': ('🟢', 'VIX crisis — liquidity panic (contrarian long)'),
    'CRISIS_STRUCTURAL': ('🔴🔴', 'VIX crisis — structural break (stay short)'),
    'CRISIS_UNKNOWN': ('🟠', 'VIX crisis — type unknown'),
    'SPIKE_LIQUIDITY': ('🟡', 'VIX spike — liquidity panic'),
    'SPIKE_STRUCTURAL': ('🔴', 'VIX spike — structural break'),
    'VIX_SPIKE': ('🟠', 'VIX spike'),
    # M70 WTI
    'SUPPLY_SHOCK_BEARISH': ('🔴', 'Oil↑ + DXY↑ — supply shock (bearish)'),
    'DEMAND_RISK_ON': ('🟡', 'Oil↑ + DXY↓ — demand/risk-on'),
    'RECESSION_FEAR': ('🟠', 'Oil↓ + DXY↑ — recession fear'),
    'REFLATION_EASING': ('🟢', 'Oil↓ + DXY↓ — reflation easing'),
    # M71 Gold
    'GEOPOLITICAL_SAFE_HAVEN': ('🔴', 'Gold↑ + DXY↑ — geopolitical panic (bearish ETH)'),
    'FIAT_DEBASEMENT': ('🟢', 'Gold↑ + DXY↓ — fiat debasement (bullish)'),
    'RISK_OFF_RATES': ('🔴', 'Gold↓ + DXY↑ — risk-off/rates'),
    'RISK_ON_DRIFT': ('🟡', 'Gold↓ + DXY↓ — risk-on drift'),
    # M72 BTC.D
    'BTC_DOMINANT': ('🔴', 'BTC dominance high — ETH underperforms'),
    'ALTCOIN_SEASON': ('🟢', 'Altcoin season — ETH outperforms'),
    # M73 Stablecoins
    'MEGA_MINT': ('🟢🟢', 'Mega stablecoin mint — capital queuing'),
    'LARGE_MINT': ('🟢', 'Large stablecoin mint — bullish inflow'),
    'LARGE_BURN': ('🔴', 'Large stablecoin burn — capital leaving'),
}


def _evaluate_tradfi_passive(result, direction):
    """Level 1: Read M66-M73 from the signal result dict and build commentary.

    Args:
        result: scanner result dict with m66-m73 keys
        direction: 'LONG' or 'SHORT'

    Returns:
        list of commentary strings
    """
    commentary = []
    headwind_count = 0
    tailwind_count = 0
    crisis_flag = False

    for m_num in (66, 67, 68, 69, 70, 71, 72, 73):
        m_key = f'm{m_num}'
        m_data = result.get(m_key)
        if not m_data or m_data.get('status') not in ('PASS',):
            continue

        score = m_data.get('score', 0.5)
        details = m_data.get('details', {})
        classification = details.get('classification', 'NORMAL')

        if classification == 'NORMAL':
            continue

        label_info = _TRADFI_LABELS.get(classification)
        if label_info:
            icon, label = label_info
        else:
            icon, label = '⚪', classification

        # Determine if this is headwind or tailwind for the trade direction
        if score > 0.55:
            effect = 'tailwind'
            tailwind_count += 1
        elif score < 0.45:
            effect = 'headwind'
            headwind_count += 1
        else:
            effect = 'neutral'

        # Flag crisis conditions
        if 'CRISIS' in classification or 'EXTREME' in classification:
            crisis_flag = True

        effect_icon = '↗' if effect == 'tailwind' else '↙' if effect == 'headwind' else '→'
        commentary.append(f"  {icon} M{m_num}: {label} (score={score:.3f} {effect_icon} {effect})")

    # Summary line
    if crisis_flag:
        commentary.insert(0, "🚨 TRADFI CRISIS ACTIVE — extreme conditions detected")
    elif headwind_count >= 3:
        commentary.insert(0, f"⚠️ Multiple tradfi headwinds ({headwind_count}) — reduced conviction")
    elif tailwind_count >= 3:
        commentary.insert(0, f"💪 Strong tradfi tailwinds ({tailwind_count}) — high conviction")
    elif headwind_count > 0 and tailwind_count > 0:
        commentary.insert(0, f"⚖️ Mixed tradfi signals ({tailwind_count} tailwind, {headwind_count} headwind)")
    elif headwind_count > 0:
        commentary.insert(0, f"⚠️ Tradfi headwind ({headwind_count} against)")
    elif tailwind_count > 0:
        commentary.insert(0, f"💪 Tradfi tailwind ({tailwind_count} for)")

    return commentary, headwind_count, tailwind_count, crisis_flag


def _evaluate_tradfi_active(result, direction, config=None):
    """Level 2: Re-fetch live tradfi data and compare to signal-time scores.

    Detects regime shifts since the signal was generated (e.g. VIX spiked,
    DXY reversed, carry unwind triggered). Returns delta commentary.

    Args:
        result: scanner result dict
        direction: 'LONG' or 'SHORT'
        config: optional config dict

    Returns:
        list of commentary strings about regime shifts
    """
    cfg = config or {}
    commentary = []

    try:
        from src.modules.m66_usdjpy import score_m66_usdjpy, fetch_usdjpy
        from src.modules.m67_dxy import score_m67_dxy, fetch_dxy
        from src.modules.m68_yield import score_m68_yield, fetch_10y_yield
        from src.modules.m69_vix import score_m69_vix, fetch_vix
        from src.modules.m70_wti import score_m70_wti, fetch_wti
        from src.modules.m71_gold import score_m71_gold, fetch_gold
        from src.modules.m72_btcdom import score_m72_btcdom, fetch_btcdom
        from src.modules.m73_stablecoin import score_m73_stablecoin, fetch_stablecoin_mints
    except ImportError:
        return commentary

    # Fetch fresh data (batch via yfinance for FX/commodities)
    try:
        import yfinance as yf
    except ImportError:
        return commentary

    # Fetch DXY once (shared by M67, M69, M70, M71)
    dxy_df = fetch_dxy()

    # Re-score each module and compare to original
    checks = [
        (66, 'usdjpy', lambda: score_m66_usdjpy(fetch_usdjpy(), dxy_df, direction, config=cfg)),
        (67, 'dxy', lambda: score_m67_dxy(dxy_df,
            float(result.get('price', 0)),
            float(result.get('price', 0)) * 0.999,  # approximate prev
            direction, config=cfg)),
        (68, 'yield', lambda: score_m68_yield(fetch_10y_yield(), None, direction, config=cfg)),
        (69, 'vix', lambda: score_m69_vix(fetch_vix(), direction, config=cfg, df_dxy=dxy_df)),
        (70, 'wti', lambda: score_m70_wti(fetch_wti(), dxy_df, direction, config=cfg)),
        (71, 'gold', lambda: score_m71_gold(fetch_gold(), dxy_df, direction, config=cfg)),
        (72, 'btcdom', lambda: score_m72_btcdom(fetch_btcdom(), direction, config=cfg)),
        (73, 'stablecoin', lambda: score_m73_stablecoin(fetch_stablecoin_mints(), direction, config=cfg)),
    ]

    shifts = []
    for m_num, name, scorer in checks:
        m_key = f'm{m_num}'
        original = result.get(m_key, {})
        orig_score = original.get('score', 0.5)
        orig_class = original.get('details', {}).get('classification', 'NORMAL')

        try:
            status, new_score, details = scorer()
        except Exception:
            continue

        new_class = details.get('classification', 'NORMAL')
        score_delta = new_score - orig_score

        # Detect meaningful shifts
        if abs(score_delta) >= 0.15:
            direction_of_shift = 'IMPROVED' if score_delta > 0 else 'DEGRADED'
            icon = '🟢' if score_delta > 0 else '🔴'
            label_info = _TRADFI_LABELS.get(new_class)
            new_label = label_info[1] if label_info else new_class

            # Does the shift help or hurt the trade?
            if direction == 'LONG':
                helps = score_delta > 0
            else:
                helps = score_delta < 0

            help_str = 'helps' if helps else 'hurts'
            shifts.append(
                f"  {icon} M{m_num} SHIFT: {orig_class} → {new_class} "
                f"(Δ{score_delta:+.3f}) — {help_str} your {direction}"
            )
        elif new_class != orig_class and new_class != 'NORMAL':
            label_info = _TRADFI_LABELS.get(new_class)
            new_label = label_info[1] if label_info else new_class
            shifts.append(
                f"  ⚪ M{m_num} changed: {orig_class} → {new_class} ({new_label})"
            )

    if shifts:
        commentary.append("📡 TRADFI REGIME SHIFT since signal generated:")
        commentary.extend(shifts)
    else:
        commentary.append("✅ Tradfi conditions stable since signal generation")

    return commentary


def evaluate_signal(result, current_price=None, config=None):
    """Evaluate a signal's current actionable quality.

    Args:
        result: scanner result dict (must have status='SIGNAL', direction, entry, sl, tp1, etc.)
        current_price: current market price (if None, uses result['price'])
        config: optional config dict

    Returns:
        dict with evaluation metrics and commentary
    """
    cfg = config or {}

    if result.get('status') != 'SIGNAL':
        return {'actionable': None, 'reason': 'no_signal', 'commentary': []}

    direction = result.get('direction', 'LONG')
    entry = result.get('entry', 0)
    market_entry = result.get('market_entry', entry)
    sl = result.get('sl', 0)
    tp1 = result.get('tp1', 0)
    tp2 = result.get('tp2', 0)
    tp3 = result.get('tp3', 0)
    sl_pct = result.get('sl_pct', 0)
    tp1_pct = result.get('tp1_pct', 0)
    ics = result.get('ics', 0)
    limit_entry = result.get('limit_entry', {})

    if not entry or not sl or not tp1:
        return {'actionable': None, 'reason': 'missing_levels', 'commentary': []}

    price = current_price or result.get('price', entry)

    # ── Drift from entry ──
    if direction == 'LONG':
        drift_pct = (price - entry) / entry * 100
        drift_direction = 'above' if drift_pct > 0 else 'below'
    else:
        drift_pct = (entry - price) / entry * 100
        drift_direction = 'above' if drift_pct > 0 else 'below'

    abs_drift = abs(drift_pct)

    # ── Current R:R at market price ──
    if direction == 'LONG':
        remaining_tp = tp1 - price
        remaining_sl = price - sl
    else:
        remaining_tp = price - tp1
        remaining_sl = sl - price

    current_rr = remaining_tp / remaining_sl if remaining_sl > 0 else 0
    original_rr = abs(tp1_pct / sl_pct) if sl_pct != 0 else 0
    rr_decay = original_rr - current_rr

    # ── SL proximity ──
    if direction == 'LONG':
        sl_distance_pct = (price - sl) / price * 100
    else:
        sl_distance_pct = (sl - price) / price * 100

    # ── TP1 proximity ──
    if direction == 'LONG':
        tp1_remaining_pct = (tp1 - price) / price * 100
    else:
        tp1_remaining_pct = (price - tp1) / price * 100

    # ── Has TP1 already been hit? ──
    tp1_hit = False
    if direction == 'LONG' and price >= tp1:
        tp1_hit = True
    elif direction == 'SHORT' and price <= tp1:
        tp1_hit = True

    # ── Has SL been hit? ──
    sl_hit = False
    if direction == 'LONG' and price <= sl:
        sl_hit = True
    elif direction == 'SHORT' and price >= sl:
        sl_hit = True

    # ── Is price past entry (chasing)? ──
    chasing = False
    if direction == 'LONG' and price > entry:
        chasing = True
    elif direction == 'SHORT' and price < entry:
        chasing = True

    # ── Limit entry assessment ──
    limit_entry_source = limit_entry.get('entry_source', 'MARKET')
    limit_entry_price = limit_entry.get('entry_price', entry)

    # ── Build commentary ──
    commentary = []
    verdict = 'CONSIDER'  # CONSIDER | SKIP | CHASE | WAIT_PULLBACK | INVALIDATED

    if sl_hit:
        verdict = 'INVALIDATED'
        commentary.append(f"❌ SL already hit — signal invalidated. Wait for new setup.")
        commentary.append(f"   Price ${price:.2f} is {'below' if direction == 'LONG' else 'above'} SL ${sl:.2f}")

    elif tp1_hit:
        verdict = 'SKIP'
        commentary.append(f"⏭️ TP1 already reached — move has happened. Skip this signal.")
        commentary.append(f"   Price ${price:.2f} passed TP1 ${tp1:.2f}. Don't chase.")
        if tp2 and ((direction == 'LONG' and price < tp2) or (direction == 'SHORT' and price > tp2)):
            commentary.append(f"   ℹ️ TP2 ${tp2:.2f} still in play if you're already in.")

    elif abs_drift < 0.10:
        # Very close to entry — ideal
        verdict = 'CONSIDER'
        commentary.append(f"✅ Price near entry — good fill opportunity.")
        commentary.append(f"   Drift: {abs_drift:.2f}% from entry. R:R still {current_rr:.2f}x")

    elif abs_drift < 0.30:
        # Small drift — still reasonable
        verdict = 'CONSIDER'
        commentary.append(f"🟡 Price drifted {abs_drift:.2f}% from entry — still reasonable.")
        commentary.append(f"   Entry ${entry:.2f} → now ${price:.2f}. R:R: {current_rr:.2f}x (was {original_rr:.2f}x)")
        if current_rr < 1.0:
            verdict = 'WAIT_PULLBACK'
            commentary.append(f"   ⚠️ R:R dropped below 1.0x — wait for pullback to entry.")

    elif abs_drift < 0.60:
        # Moderate drift — R:R degraded
        if current_rr >= 1.5:
            verdict = 'CONSIDER'
            commentary.append(f"🟡 Drifted {abs_drift:.2f}% but R:R still decent at {current_rr:.2f}x")
        elif current_rr >= 1.0:
            verdict = 'WAIT_PULLBACK'
            commentary.append(f"⚠️ Drifted {abs_drift:.2f}% — R:R compressed to {current_rr:.2f}x")
            commentary.append(f"   Better entry if price pulls back to ${entry:.2f}")
        else:
            verdict = 'SKIP'
            commentary.append(f"⏭️ Drifted {abs_drift:.2f}% — R:R now {current_rr:.2f}x (below 1.0x). Skip.")

    else:
        # Large drift
        if sl_hit:
            verdict = 'INVALIDATED'
            commentary.append(f"❌ Price hit SL — signal dead.")
        elif chasing and current_rr < 0.5:
            verdict = 'SKIP'
            commentary.append(f"⏭️ Chasing — price moved {abs_drift:.2f}% past entry. R:R {current_rr:.2f}x. Skip.")
        elif chasing:
            verdict = 'CHASE'
            commentary.append(f"🏃 Price moved {abs_drift:.2f}% past entry — chasing if you enter now.")
            commentary.append(f"   R:R: {current_rr:.2f}x (was {original_rr:.2f}x). Reduced edge.")
        else:
            verdict = 'WAIT_PULLBACK'
            commentary.append(f"⚠️ Price {abs_drift:.2f}% from entry — wait for pullback.")
            commentary.append(f"   Ideal entry: ${entry:.2f}. Current: ${price:.2f}")

    # ── SL proximity warning ──
    if not sl_hit and sl_distance_pct < 0.30 and verdict not in ('INVALIDATED', 'SKIP'):
        commentary.append(f"⚠️ Very close to SL ({sl_distance_pct:.2f}%) — tight stop, high whipsaw risk")

    # ── ICS quality note ──
    if ics >= 0.70:
        commentary.append(f"💪 High-conviction signal (ICS {ics:.3f}) — worth the drift tolerance")
    elif ics < 0.55:
        commentary.append(f"⚠️ Low ICS ({ics:.3f}) — signal is marginal even at entry")

    # ── Limit entry note ──
    if limit_entry_source != 'MARKET' and verdict in ('CONSIDER', 'WAIT_PULLBACK'):
        commentary.append(f"📋 Limit entry at ${limit_entry_price:.2f} [{limit_entry_source}] — use limit order, not market")

    # ── Squeeze context ──
    sq = result.get('squeeze', {})
    if sq and sq.get('squeeze_status') == 'TRIGGERED' and sq.get('direction') == direction:
        commentary.append(f"🔥 Squeeze TRIGGERED — momentum may carry further than normal")

    # ── M66-M73: Tradfi context (Level 1 — passive) ──
    tradfi_commentary, tradfi_headwinds, tradfi_tailwinds, tradfi_crisis = \
        _evaluate_tradfi_passive(result, direction)
    if tradfi_commentary:
        commentary.append("")
        commentary.append("  📊 TRADFI CONTEXT (M66-M73):")
        commentary.extend(tradfi_commentary)

    # ── M66-M73: Tradfi regime shift check (Level 2 — active) ──
    # Only run if signal is still actionable and config enables it
    if cfg.get('SIGNAL_EVAL_TRADFI_ACTIVE', False) and verdict in ('CONSIDER', 'CHASE', 'WAIT_PULLBACK'):
        try:
            active_commentary = _evaluate_tradfi_active(result, direction, config=cfg)
            if active_commentary:
                commentary.append("")
                commentary.extend(active_commentary)
        except Exception:
            pass  # non-fatal — active check is best-effort

    # ── Tradfi-adjusted verdict ──
    # If multiple tradfi headwinds and signal is marginal, downgrade
    if tradfi_crisis and verdict == 'CONSIDER':
        verdict = 'WAIT_PULLBACK'
        commentary.append(f"  ⚠️ Verdict downgraded to WAIT_PULLBACK — tradfi crisis active")
    elif tradfi_headwinds >= 3 and verdict == 'CONSIDER' and ics < 0.65:
        verdict = 'WAIT_PULLBACK'
        commentary.append(f"  ⚠️ Verdict downgraded to WAIT_PULLBACK — {tradfi_headwinds} tradfi headwinds with marginal ICS")

    return {
        'actionable': verdict in ('CONSIDER', 'CHASE'),
        'verdict': verdict,
        'price': price,
        'entry': entry,
        'sl': sl,
        'tp1': tp1,
        'direction': direction,
        'drift_pct': round(drift_pct, 4),
        'abs_drift_pct': round(abs_drift, 4),
        'current_rr': round(current_rr, 3),
        'original_rr': round(original_rr, 3),
        'rr_decay': round(rr_decay, 3),
        'sl_distance_pct': round(sl_distance_pct, 4),
        'tp1_remaining_pct': round(tp1_remaining_pct, 4),
        'sl_hit': sl_hit,
        'tp1_hit': tp1_hit,
        'chasing': chasing,
        'commentary': commentary,
        'tradfi_headwinds': tradfi_headwinds,
        'tradfi_tailwinds': tradfi_tailwinds,
        'tradfi_crisis': tradfi_crisis,
    }


def format_signal_eval(ev):
    """Format signal evaluation as printable text."""
    if not ev or ev.get('actionable') is None:
        return ''

    lines = []
    verdict = ev.get('verdict', '?')
    verdict_icons = {
        'CONSIDER': '✅', 'SKIP': '⏭️', 'CHASE': '🏃',
        'WAIT_PULLBACK': '⏳', 'INVALIDATED': '❌',
    }
    icon = verdict_icons.get(verdict, '❓')

    lines.append(f"\n  {'─' * 56}")
    lines.append(f"  SIGNAL EVALUATION ({ev['direction']})")
    lines.append(f"  {'─' * 56}")
    lines.append(f"  {icon} Verdict: {verdict}")
    lines.append(f"  Price: ${ev['price']:.2f}  |  Entry: ${ev['entry']:.2f}  |  Drift: {ev['drift_pct']:+.2f}%")
    lines.append(f"  R:R now: {ev['current_rr']:.2f}x  (original: {ev['original_rr']:.2f}x)")
    lines.append(f"  SL dist: {ev['sl_distance_pct']:.2f}%  |  TP1 left: {ev['tp1_remaining_pct']:.2f}%")

    for c in ev.get('commentary', []):
        lines.append(f"  {c}")

    lines.append(f"  {'─' * 56}")
    return '\n'.join(lines)

"""
Signal Evaluator — post-generation commentary

Evaluates a signal AFTER it was generated to assess whether the entry
is still actionable. Commentary only — does NOT gate or block signals.

Usage:
    from src.signal_eval import evaluate_signal, format_signal_eval
    eval_result = evaluate_signal(result, current_price)
    print(format_signal_eval(eval_result))
"""


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

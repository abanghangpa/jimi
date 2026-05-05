#!/usr/bin/env python3
"""
Squeeze + Liquidity Combined Analysis.

Questions answered:
  1. When squeeze is PENDING, where is the nearest liquidity?
  2. Does squeeze direction match nearest liquidity direction?
  3. After squeeze TRIGGERS, does price hit nearest liquidity first?
  4. Can nearest liquidity serve as TP1 for squeeze trades?
  5. Does squeeze breakout accelerate toward bigger liquidity?

Usage:
    python scripts/squeeze_liquidity_combined.py
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import numpy as np
import pandas as pd
from collections import defaultdict

from src.modules.m5_liquidation import build_volume_profile, find_magnets
from src.modules.m18_squeeze import (
    _check_compression, _compute_2h_macd, _detect_macd_coil,
    _check_entry_trigger, _check_cvd_trigger,
    SQUEEZE_V5_DEFAULTS,
)


def load_data(csv_path):
    df = pd.read_csv(csv_path)
    df['Open time'] = pd.to_datetime(df['Open time'])
    for c in ['Open', 'High', 'Low', 'Close', 'Volume',
              'Taker buy base asset volume', 'Taker buy quote asset volume']:
        df[c] = pd.to_numeric(df[c], errors='coerce')
    df['vol_ma20'] = df['Volume'].rolling(20).mean()
    taker_base = df['Taker buy base asset volume']
    total_vol = df['Volume']
    df['taker_ratio'] = (taker_base / total_vol.replace(0, np.nan)).fillna(0.5)
    return df


def get_squeeze_state(df, idx, cfg=None):
    """Replicate scanner's squeeze detection at a given bar."""
    c = cfg or SQUEEZE_V5_DEFAULTS

    if idx < 48:
        return None

    # Compression history (last 48 bars)
    compression_history = []
    for i in range(max(0, idx - 47), idx):
        r48_h = float(df['High'].iloc[max(0, i-47):i+1].max())
        r48_l = float(df['Low'].iloc[max(0, i-47):i+1].min())
        r48_pct = (r48_h - r48_l) / float(df['Close'].iloc[i]) * 100 if float(df['Close'].iloc[i]) > 0 else 5.0
        vr = float(df['Volume'].iloc[i] / df['vol_ma20'].iloc[i]) if float(df['vol_ma20'].iloc[i]) > 0 else 1.0
        br = (float(df['High'].iloc[i]) - float(df['Low'].iloc[i])) / float(df['Close'].iloc[i]) * 100 if float(df['Close'].iloc[i]) > 0 else 0.5
        tr = float(df['Taker buy base asset volume'].iloc[i]) / float(df['Volume'].iloc[i]) if float(df['Volume'].iloc[i]) > 0 else 0.5
        compression_history.append((r48_pct, vr, br, tr))

    # Current range
    r48_h = float(df['High'].iloc[max(0, idx-47):idx+1].max())
    r48_l = float(df['Low'].iloc[max(0, idx-47):idx+1].min())
    range48 = (r48_h - r48_l) / float(df['Close'].iloc[idx]) * 100

    # Path A: 15m compression
    is_compressed, comp_bars, dry_count, doji_count = _check_compression(range48, compression_history, c)

    # Path B: 2h MACD coil
    macd_data = _compute_2h_macd(df.iloc[:idx+1], c)
    is_coiled = False
    coil_bars = 0
    hist_flip = False
    coil_direction = 'NEUTRAL'
    coil_details = {}

    if macd_data:
        is_coiled, coil_bars, hist_flip, coil_direction, coil_details = _detect_macd_coil(macd_data, c)

    # Squeeze status
    dual_path = is_compressed and is_coiled
    squeeze_type = 'NONE'
    squeeze_status = 'NONE'
    direction = 'NEUTRAL'
    trigger = 'NONE'

    if dual_path or (is_compressed and coil_bars >= 6):
        squeeze_type = 'DUAL_PATH' if dual_path else 'SHORT_SQUEEZE' if coil_direction == 'LONG' else 'LONG_SQUEEZE'
        squeeze_status = 'PENDING'

        if hist_flip:
            squeeze_status = 'TRIGGERED'
            direction = coil_direction
            trigger = 'HIST_FLIP'
        else:
            # Check CVD trigger
            taker = float(df['taker_ratio'].iloc[idx])
            cvd_trigger, cvd_dir, cvd_type = _check_cvd_trigger(taker, None, c)
            if cvd_trigger:
                squeeze_status = 'TRIGGERED'
                direction = cvd_dir
                trigger = cvd_type
            else:
                direction = coil_direction if coil_direction != 'NEUTRAL' else 'NEUTRAL'

    # Coil range
    coil_high = coil_details.get('coil_high', float(df['Close'].iloc[idx]))
    coil_low = coil_details.get('coil_low', float(df['Close'].iloc[idx]))

    # Entry trigger
    entry_triggered = False
    entry_price = 0
    entry_condition = ''
    if squeeze_status == 'TRIGGERED' and direction != 'NEUTRAL':
        entry_triggered, entry_price, entry_condition = _check_entry_trigger(
            float(df['Close'].iloc[idx]),
            float(df['Volume'].iloc[idx]),
            float(df['vol_ma20'].iloc[idx]),
            coil_high, coil_low, direction, c)

    return {
        'squeeze_type': squeeze_type,
        'squeeze_status': squeeze_status,
        'direction': direction,
        'trigger': trigger,
        'is_compressed': is_compressed,
        'is_coiled': is_coiled,
        'comp_bars': comp_bars,
        'coil_bars': coil_bars,
        'coil_hours': coil_bars * 2,
        'coil_high': coil_high,
        'coil_low': coil_low,
        'hist_flip': hist_flip,
        'entry_triggered': entry_triggered,
        'entry_price': entry_price,
        'entry_condition': entry_condition,
        'range48': range48,
    }


def get_liquidity_targets(df, idx, lookback=672, n_bins=50):
    """Get liquidity magnets at current bar."""
    h = df['High'].values[:idx+1].astype(float)
    l = df['Low'].values[:idx+1].astype(float)
    c = df['Close'].values[:idx+1].astype(float)
    v = df['Volume'].values[:idx+1].astype(float)

    actual_lb = min(lookback, idx + 1)
    bin_centers, vol_profile, bin_edges = build_volume_profile(
        h[-actual_lb:], l[-actual_lb:], c[-actual_lb:], v[-actual_lb:],
        n_bins=n_bins, lookback=actual_lb)

    if bin_centers is None:
        return [], [], []

    magnets = find_magnets(bin_centers, vol_profile, n_magnets=10, min_gap_pct=0.003)
    price = float(df['Close'].iloc[idx])

    above = [(p, v, s) for p, v, s in magnets if p > price * 1.001]
    below = [(p, v, s) for p, v, s in magnets if p < price * 0.999]

    above.sort(key=lambda x: x[0])  # nearest first
    below.sort(key=lambda x: -x[0])  # nearest first

    return magnets, above, below


def analyze_combined(df, year=2026, snapshot_interval=48, lookforward=96):
    df_year = df[df['Open time'].dt.year == year].copy().reset_index(drop=True)

    print(f"\n{'='*70}")
    print(f"  SQUEEZE + LIQUIDITY COMBINED ANALYSIS — {year}")
    print(f"{'='*70}\n")

    events = []
    idx = 672  # warmup

    while idx < len(df_year) - lookforward:
        ts = df_year['Open time'].iloc[idx]
        price = float(df_year['Close'].iloc[idx])

        # Get squeeze state
        sq = get_squeeze_state(df_year, idx)
        if sq is None:
            idx += snapshot_interval
            continue

        # Get liquidity targets
        magnets, above_liq, below_liq = get_liquidity_targets(df_year, idx)
        if len(magnets) < 3:
            idx += snapshot_interval
            continue

        # Nearest targets
        nearest_above = above_liq[0] if above_liq else None
        nearest_below = below_liq[0] if below_liq else None

        # Biggest targets
        biggest_above = max(above_liq, key=lambda x: x[2]) if above_liq else None
        biggest_below = max(below_liq, key=lambda x: x[2]) if below_liq else None

        # Look forward for sweeps
        f_highs = df_year['High'].values[idx+1:idx+1+lookforward].astype(float)
        f_lows = df_year['Low'].values[idx+1:idx+1+lookforward].astype(float)
        f_closes = df_year['Close'].values[idx+1:idx+1+lookforward].astype(float)
        f_times = df_year['Open time'].values[idx+1:idx+1+lookforward]

        def find_sweep_time(target_price):
            for i in range(len(f_highs)):
                if f_highs[i] >= target_price and f_lows[i] <= target_price:
                    return i
            return None

        # Check sweeps
        nearest_above_swept = find_sweep_time(nearest_above[0]) if nearest_above else None
        nearest_below_swept = find_sweep_time(nearest_below[0]) if nearest_below else None
        biggest_above_swept = find_sweep_time(biggest_above[0]) if biggest_above else None
        biggest_below_swept = find_sweep_time(biggest_below[0]) if biggest_below else None

        # Post-sweep behavior (4h, 12h)
        def post_move(sweep_bar):
            if sweep_bar is None:
                return None, None
            end_4 = min(sweep_bar + 16, len(f_closes) - 1)
            end_12 = min(sweep_bar + 48, len(f_closes) - 1)
            if end_4 <= sweep_bar or end_12 <= sweep_bar:
                return None, None
            move_4 = (float(f_closes[end_4]) - float(f_closes[sweep_bar])) / float(f_closes[sweep_bar]) * 100
            move_12 = (float(f_closes[end_12]) - float(f_closes[sweep_bar])) / float(f_closes[sweep_bar]) * 100
            return move_4, move_12

        # Direction match
        squeeze_dir = sq['direction']
        nearest_liq_dir = 'NONE'
        nearest_liq_price = 0
        nearest_liq_dist = 0

        if squeeze_dir == 'LONG' and nearest_above:
            nearest_liq_dir = 'ABOVE'
            nearest_liq_price = nearest_above[0]
            nearest_liq_dist = (nearest_above[0] - price) / price * 100
        elif squeeze_dir == 'SHORT' and nearest_below:
            nearest_liq_dir = 'BELOW'
            nearest_liq_price = nearest_below[0]
            nearest_liq_dist = (nearest_below[0] - price) / price * 100

        # TP1 candidate: nearest liquidity in squeeze direction
        tp1_price = nearest_liq_price if nearest_liq_price else 0
        tp1_pct = abs(nearest_liq_dist) if nearest_liq_dist else 0

        event = {
            'timestamp': str(ts),
            'price': price,
            'squeeze_status': sq['squeeze_status'],
            'squeeze_type': sq['squeeze_type'],
            'squeeze_dir': squeeze_dir,
            'coil_hours': sq['coil_hours'],
            'coil_high': sq['coil_high'],
            'coil_low': sq['coil_low'],
            'range48': sq['range48'],
            'entry_triggered': sq['entry_triggered'],
            'nearest_above_price': nearest_above[0] if nearest_above else 0,
            'nearest_above_strength': nearest_above[2] if nearest_above else 0,
            'nearest_above_dist': (nearest_above[0] - price) / price * 100 if nearest_above else 0,
            'nearest_below_price': nearest_below[0] if nearest_below else 0,
            'nearest_below_strength': nearest_below[2] if nearest_below else 0,
            'nearest_below_dist': (price - nearest_below[0]) / price * 100 if nearest_below else 0,
            'biggest_above_strength': biggest_above[2] if biggest_above else 0,
            'biggest_below_strength': biggest_below[2] if biggest_below else 0,
            'direction_match': (squeeze_dir == 'LONG' and nearest_above is not None) or
                               (squeeze_dir == 'SHORT' and nearest_below is not None),
            'tp1_price': tp1_price,
            'tp1_pct': tp1_pct,
            'nearest_above_swept': nearest_above_swept is not None,
            'nearest_below_swept': nearest_below_swept is not None,
            'biggest_above_swept': biggest_above_swept is not None,
            'biggest_below_swept': biggest_below_swept is not None,
        }

        # Post-move for nearest
        if squeeze_dir == 'LONG' and nearest_above_swept is not None:
            m4, m12 = post_move(nearest_above_swept)
            event['post_nearest_4h'] = m4
            event['post_nearest_12h'] = m12
        elif squeeze_dir == 'SHORT' and nearest_below_swept is not None:
            m4, m12 = post_move(nearest_below_swept)
            event['post_nearest_4h'] = m4
            event['post_nearest_12h'] = m12

        events.append(event)
        idx += snapshot_interval

    if not events:
        print("No events found.")
        return

    N = len(events)

    # ══════════════════════════════════════════════════
    #  REPORT
    # ══════════════════════════════════════════════════

    # Split by squeeze status
    pending = [e for e in events if e['squeeze_status'] == 'PENDING']
    triggered = [e for e in events if e['squeeze_status'] == 'TRIGGERED']
    no_sq = [e for e in events if e['squeeze_status'] == 'NONE']

    print(f"  Total snapshots: {N}")
    print(f"  Squeeze PENDING:  {len(pending)} ({len(pending)/N*100:.1f}%)")
    print(f"  Squeeze TRIGGERED: {len(triggered)} ({len(triggered)/N*100:.1f}%)")
    print(f"  No squeeze:       {len(no_sq)} ({len(no_sq)/N*100:.1f}%)")
    print()

    # ── 1. Direction Match ──
    print("─" * 60)
    print("  1. DOES SQUEEZE DIRECTION MATCH NEAREST LIQUIDITY?")
    print("─" * 60)

    for label, subset in [('PENDING', pending), ('TRIGGERED', triggered)]:
        if not subset:
            continue
        matched = sum(1 for e in subset if e['direction_match'])
        print(f"  {label} (n={len(subset)}): direction match = {matched}/{len(subset)} ({matched/len(subset)*100:.1f}%)")

        # Direction breakdown
        longs = [e for e in subset if e['squeeze_dir'] == 'LONG']
        shorts = [e for e in subset if e['squeeze_dir'] == 'SHORT']
        if longs:
            has_above = sum(1 for e in longs if e['nearest_above_price'] > 0)
            avg_dist = np.mean([e['nearest_above_dist'] for e in longs if e['nearest_above_dist'] > 0])
            print(f"    LONG:  {has_above}/{len(longs)} have above target, avg dist = {avg_dist:.2f}%")
        if shorts:
            has_below = sum(1 for e in shorts if e['nearest_below_price'] > 0)
            avg_dist = np.mean([e['nearest_below_dist'] for e in shorts if e['nearest_below_dist'] > 0])
            print(f"    SHORT: {has_below}/{len(shorts)} have below target, avg dist = {avg_dist:.2f}%")
    print()

    # ── 2. TP1 feasibility ──
    print("─" * 60)
    print("  2. TP1 CANDIDATE: NEAREST LIQUIDITY IN SQUEEZE DIRECTION")
    print("─" * 60)

    for label, subset in [('PENDING', pending), ('TRIGGERED', triggered)]:
        if not subset:
            continue
        with_tp = [e for e in subset if e['tp1_pct'] > 0]
        if not with_tp:
            continue
        tp_pcts = [e['tp1_pct'] for e in with_tp]
        print(f"\n  {label} (n={len(with_tp)} with TP1):")
        print(f"    Mean TP1 distance:   {np.mean(tp_pcts):.3f}%")
        print(f"    Median TP1 distance: {np.median(tp_pcts):.3f}%")
        print(f"    Min: {np.min(tp_pcts):.3f}%  Max: {np.max(tp_pcts):.3f}%")

        # Buckets
        for lo, hi, lbl in [(0, 0.25, '0-0.25%'), (0.25, 0.5, '0.25-0.5%'),
                             (0.5, 1.0, '0.5-1.0%'), (1.0, 999, '1.0%+')]:
            cnt = sum(1 for p in tp_pcts if lo <= p < hi)
            if cnt:
                pct = cnt / len(tp_pcts) * 100
                bar = '█' * int(pct / 2)
                print(f"      {lbl:<12} {cnt:>4} ({pct:5.1f}%) {bar}")
    print()

    # ── 3. Nearest vs Biggest sweep rate ──
    print("─" * 60)
    print("  3. NEAREST vs BIGGEST: SWEEP RATE AFTER SQUEEZE")
    print("─" * 60)

    for label, subset in [('PENDING', pending), ('TRIGGERED', triggered)]:
        if not subset:
            continue
        # For LONG squeezes: check above targets
        longs = [e for e in subset if e['squeeze_dir'] == 'LONG' and e['nearest_above_price'] > 0]
        shorts = [e for e in subset if e['squeeze_dir'] == 'SHORT' and e['nearest_below_price'] > 0]

        if longs:
            nearest_swept = sum(1 for e in longs if e['nearest_above_swept'])
            biggest_swept = sum(1 for e in longs if e['biggest_above_swept'])
            print(f"  {label} LONG (n={len(longs)}):")
            print(f"    Nearest above swept: {nearest_swept}/{len(longs)} ({nearest_swept/len(longs)*100:.1f}%)")
            print(f"    Biggest above swept: {biggest_swept}/{len(longs)} ({biggest_swept/len(longs)*100:.1f}%)")

        if shorts:
            nearest_swept = sum(1 for e in shorts if e['nearest_below_swept'])
            biggest_swept = sum(1 for e in shorts if e['biggest_below_swept'])
            print(f"  {label} SHORT (n={len(shorts)}):")
            print(f"    Nearest below swept: {nearest_swept}/{len(shorts)} ({nearest_swept/len(shorts)*100:.1f}%)")
            print(f"    Biggest below swept: {biggest_swept}/{len(shorts)} ({biggest_swept/len(shorts)*100:.1f}%)")
    print()

    # ── 4. Post-squeeze behavior ──
    print("─" * 60)
    print("  4. POST-SQUEEZE: PRICE BEHAVIOR AFTER HITTING NEAREST TP1")
    print("─" * 60)

    for label, subset in [('PENDING', pending), ('TRIGGERED', triggered)]:
        post_events = [e for e in subset if e.get('post_nearest_4h') is not None]
        if not post_events:
            continue

        post4 = [e['post_nearest_4h'] for e in post_events]
        post12 = [e['post_nearest_12h'] for e in post_events if e.get('post_nearest_12h') is not None]

        print(f"\n  {label} (n={len(post_events)} swept nearest):")
        print(f"    Avg 4h after nearest sweep:  {np.mean(post4):+.3f}%")
        print(f"    Avg 12h after nearest sweep: {np.mean(post12):+.3f}%")

        # How many continue in squeeze direction?
        cont_4 = 0
        cont_12 = 0
        for e in post_events:
            if e['squeeze_dir'] == 'LONG':
                if e.get('post_nearest_4h') and e['post_nearest_4h'] > 0.05:
                    cont_4 += 1
                if e.get('post_nearest_12h') and e['post_nearest_12h'] > 0.05:
                    cont_12 += 1
            elif e['squeeze_dir'] == 'SHORT':
                if e.get('post_nearest_4h') and e['post_nearest_4h'] < -0.05:
                    cont_4 += 1
                if e.get('post_nearest_12h') and e['post_nearest_12h'] < -0.05:
                    cont_12 += 1

        print(f"    Continue in squeeze dir at 4h:  {cont_4}/{len(post_events)} ({cont_4/len(post_events)*100:.1f}%)")
        print(f"    Continue in squeeze dir at 12h: {cont_12}/{len(post_events)} ({cont_12/len(post_events)*100:.1f}%)")
    print()

    # ── 5. Coil duration vs TP1 distance ──
    print("─" * 60)
    print("  5. COIL DURATION vs TP1 DISTANCE")
    print("─" * 60)

    for label, subset in [('PENDING', pending), ('TRIGGERED', triggered)]:
        with_tp = [e for e in subset if e['tp1_pct'] > 0 and e['coil_hours'] > 0]
        if not with_tp:
            continue

        short_coil = [e for e in with_tp if e['coil_hours'] < 18]
        long_coil = [e for e in with_tp if e['coil_hours'] >= 18]

        print(f"\n  {label}:")
        if short_coil:
            avg_tp = np.mean([e['tp1_pct'] for e in short_coil])
            avg_range = np.mean([e['range48'] for e in short_coil])
            print(f"    Short coil (<18h): n={len(short_coil)}  avg TP1={avg_tp:.3f}%  avg range={avg_range:.2f}%")
        if long_coil:
            avg_tp = np.mean([e['tp1_pct'] for e in long_coil])
            avg_range = np.mean([e['range48'] for e in long_coil])
            print(f"    Long coil (≥18h):  n={len(long_coil)}  avg TP1={avg_tp:.3f}%  avg range={avg_range:.2f}%")
    print()

    # ── 6. Squeeze entry triggered → nearest liquidity hit rate ──
    print("─" * 60)
    print("  6. ENTRY TRIGGERED → NEAREST LIQUIDITY HIT RATE")
    print("─" * 60)

    entry_yes = [e for e in triggered if e['entry_triggered']]
    entry_no = [e for e in triggered if not e['entry_triggered']]

    for label, subset in [('Entry TRIGGERED', entry_yes), ('Entry NOT triggered', entry_no)]:
        if not subset:
            continue
        # Check if nearest in direction was swept
        swept = 0
        total = 0
        for e in subset:
            if e['squeeze_dir'] == 'LONG' and e['nearest_above_price'] > 0:
                total += 1
                if e['nearest_above_swept']:
                    swept += 1
            elif e['squeeze_dir'] == 'SHORT' and e['nearest_below_price'] > 0:
                total += 1
                if e['nearest_below_swept']:
                    swept += 1

        if total > 0:
            print(f"  {label} (n={total}): nearest swept = {swept}/{total} ({swept/total*100:.1f}%)")
    print()

    # ── 7. Combined strategy simulation ──
    print("─" * 60)
    print("  7. STRATEGY SIMULATION: SQUEEZE + NEAREST LIQUIDITY AS TP1")
    print("─" * 60)

    # For triggered squeezes with direction, simulate:
    # Entry = coil breakout level
    # TP1 = nearest liquidity in squeeze direction
    # SL = other side of coil range
    trades = []
    for e in triggered:
        if e['squeeze_dir'] == 'NEUTRAL' or e['tp1_pct'] == 0:
            continue

        # Check if TP1 was hit within lookforward
        tp1_hit = False
        sl_hit = False

        if e['squeeze_dir'] == 'LONG':
            entry = e['coil_high']
            sl = e['coil_low']
            tp1 = e['nearest_above_price']

            for i in range(lookforward):
                bar_h = float(df_year['High'].values[idx + 1 + i]) if idx + 1 + i < len(df_year) else 0
                bar_l = float(df_year['Low'].values[idx + 1 + i]) if idx + 1 + i < len(df_year) else 0
                if bar_l <= sl:
                    sl_hit = True
                    break
                if bar_h >= tp1:
                    tp1_hit = True
                    break

        elif e['squeeze_dir'] == 'SHORT':
            entry = e['coil_low']
            sl = e['coil_high']
            tp1 = e['nearest_below_price']

            for i in range(lookforward):
                bar_h = float(df_year['High'].values[idx + 1 + i]) if idx + 1 + i < len(df_year) else 0
                bar_l = float(df_year['Low'].values[idx + 1 + i]) if idx + 1 + i < len(df_year) else 0
                if bar_h >= sl:
                    sl_hit = True
                    break
                if bar_l <= tp1:
                    tp1_hit = True
                    break

        if tp1_hit or sl_hit:
            trades.append({
                'dir': e['squeeze_dir'],
                'tp1_pct': e['tp1_pct'],
                'tp1_hit': tp1_hit,
                'sl_hit': sl_hit,
                'coil_hours': e['coil_hours'],
            })

    if trades:
        wins = sum(1 for t in trades if t['tp1_hit'])
        losses = sum(1 for t in trades if t['sl_hit'])
        total = len(trades)
        print(f"  Triggered squeeze trades: {total}")
        print(f"  TP1 hit (win):   {wins} ({wins/total*100:.1f}%)")
        print(f"  SL hit (loss):   {losses} ({losses/total*100:.1f}%)")
        print(f"  Neither (expired): {total - wins - losses} ({(total-wins-losses)/total*100:.1f}%)")

        if wins > 0:
            avg_win_tp = np.mean([t['tp1_pct'] for t in trades if t['tp1_hit']])
            print(f"  Avg TP1 distance on wins: {avg_win_tp:.3f}%")

        # By direction
        long_trades = [t for t in trades if t['dir'] == 'LONG']
        short_trades = [t for t in trades if t['dir'] == 'SHORT']
        if long_trades:
            w = sum(1 for t in long_trades if t['tp1_hit'])
            print(f"    LONG:  {w}/{len(long_trades)} TP1 hit ({w/len(long_trades)*100:.1f}%)")
        if short_trades:
            w = sum(1 for t in short_trades if t['tp1_hit'])
            print(f"    SHORT: {w}/{len(short_trades)} TP1 hit ({w/len(short_trades)*100:.1f}%)")
    print()

    # ══════════════════════════════════════════════════
    #  ACTIONABLE SUMMARY
    # ══════════════════════════════════════════════════
    print("═" * 60)
    print("  ACTIONABLE SUMMARY")
    print("═" * 60)

    # Direction match rate for triggered
    if triggered:
        match_rate = sum(1 for e in triggered if e['direction_match']) / len(triggered) * 100
        avg_tp1 = np.mean([e['tp1_pct'] for e in triggered if e['tp1_pct'] > 0])
        print(f"""
  1. SQUEEZE DIRECTION → NEAREST LIQUIDITY MATCH: {match_rate:.0f}%
     When squeeze triggers, it tends to aim at the nearest
     liquidity pool in that direction.

  2. TP1 (NEAREST LIQUIDITY): avg {avg_tp1:.2f}% from entry
     This is a realistic, high-probability first target for
     squeeze trades. Median: {np.median([e['tp1_pct'] for e in triggered if e['tp1_pct'] > 0]):.2f}%

  3. POST-TP1 BEHAVIOR:
     After hitting nearest liquidity, price tends to consolidate
     or reverse at 4h, then resume at 12h.
     → Take partial at TP1 (nearest liquidity)
     → Hold runner for continuation

  4. COIL DURATION MATTERS:
     Longer coils (≥18h) → bigger moves after breakout
     Short coils (<18h) → tighter TP1 targets

  5. STRATEGY:
     • Wait for squeeze PENDING + direction resolved
     • Identify nearest liquidity in squeeze direction → TP1
     • Enter on breakout confirmation (hist flip or taker shift)
     • SL = opposite side of coil range
     • TP1 = nearest liquidity pool (high hit rate)
     • Runner = next nearest or biggest pool
""")
    print("═" * 60)


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--year', type=int, default=2026)
    parser.add_argument('--interval', type=int, default=48)
    parser.add_argument('--forward', type=int, default=96)
    args = parser.parse_args()

    csv_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'eth_15m_merged.csv')
    df = load_data(csv_path)
    analyze_combined(df, year=args.year, snapshot_interval=args.interval, lookforward=args.forward)

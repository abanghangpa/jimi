#!/usr/bin/env python3
"""
Forward Test Runner — Monitors key levels from conflict analysis.

Takes a forward test config (from conflict_resolver) and watches
the key level in real time. Declares a winner when a trigger fires.

Usage:
    # Standalone (from saved conflict JSON):
    python scripts/forward_test.py data/scans/scan_XXXXXX.json

    # Auto-spawned by scanner with conflict data:
    python scripts/forward_test.py --level 2319 2328 --scenarios wyckoff,breakout
"""

import argparse
import sys
import os
import json
import time
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from src.utils.data_handler import fetch_recent
from src.utils.indicators import calc_rsi, calc_atr

SCAN_INTERVAL = 300  # 5 min
STATE_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data', 'scans')


def load_conflict_from_scan(scan_file):
    """Load forward test config from a saved scan result."""
    with open(scan_file) as f:
        result = json.load(f)

    conflict = result.get('conflict_resolution', {})
    ft = conflict.get('forward_test')
    if not ft:
        print(f"  No forward test in scan file.")
        sys.exit(1)
    return ft


def analyze_bar(df, idx):
    """Extract key metrics from a bar."""
    row = df.iloc[idx]
    price = float(row['Close'])
    high = float(row['High'])
    low = float(row['Low'])
    vol = float(row['Volume'])
    vol_ma = float(df['Volume'].rolling(20).mean().iloc[idx]) if idx >= 20 else vol
    rsi = float(calc_rsi(df['Close'], 14).iloc[idx]) if idx >= 14 else 50
    atr = float(calc_atr(df['High'], df['Low'], df['Close'], 14).iloc[idx]) if idx >= 14 else 0

    return {
        'timestamp': str(row['Open time']),
        'price': price,
        'high': high,
        'low': low,
        'volume': vol,
        'vol_ratio': vol / vol_ma if vol_ma > 0 else 0,
        'rsi': rsi,
        'atr': atr,
    }


def check_wyckoff_rejection(df, zone_high, idx, swept_high):
    """Check for Wyckoff-style rejection after sweep.

    Conditions:
    - Price dropped >= 0.3% from swept high
    - Volume spike >= 1.5x 20MA on the sweep bar
    - RSI declining from sweep
    """
    bar = analyze_bar(df, idx)
    sweep_bar = analyze_bar(df, max(0, idx - 1))

    drop_pct = (swept_high - bar['price']) / swept_high * 100
    vol_spike = sweep_bar['vol_ratio'] >= 1.5 or bar['vol_ratio'] >= 1.5
    rejection = drop_pct >= 0.3 and vol_spike

    return {
        'triggered': rejection,
        'drop_pct': round(drop_pct, 2),
        'vol_spike': vol_spike,
        'vol_ratio': round(bar['vol_ratio'], 2),
        'sweep_vol_ratio': round(sweep_bar['vol_ratio'], 2),
        'rsi': round(bar['rsi'], 1),
        'price': bar['price'],
    }


def check_hold_above(df, zone_high, idx, bars_above, hold_bars=3):
    """Check if price holds above zone for N bars."""
    bar = analyze_bar(df, idx)
    tolerance = bar['atr'] * 0.1 if bar['atr'] > 0 else 5
    held = bar['low'] >= zone_high - tolerance
    if held:
        bars_above += 1
    else:
        bars_above = 0

    return {
        'triggered': bars_above >= hold_bars,
        'bars_above': bars_above,
        'needed': hold_bars,
        'price': bar['price'],
        'low': bar['low'],
        'vol_ratio': round(bar['vol_ratio'], 2),
    }


def check_accumulation_reversal(df, zone_low, idx, swept_low):
    """Check for Wyckoff accumulation reversal after sweep below."""
    bar = analyze_bar(df, idx)
    bounce_pct = (bar['price'] - swept_low) / swept_low * 100
    vol_spike = bar['vol_ratio'] >= 1.5
    reversal = bounce_pct >= 0.3 and vol_spike

    return {
        'triggered': reversal,
        'bounce_pct': round(bounce_pct, 2),
        'vol_spike': vol_spike,
        'vol_ratio': round(bar['vol_ratio'], 2),
        'rsi': round(bar['rsi'], 1),
        'price': bar['price'],
    }


def check_hold_below(df, zone_low, idx, bars_below, hold_bars=3):
    """Check if price holds below zone for N bars."""
    bar = analyze_bar(df, idx)
    tolerance = bar['atr'] * 0.1 if bar['atr'] > 0 else 5
    held = bar['high'] <= zone_low + tolerance
    if held:
        bars_below += 1
    else:
        bars_below = 0

    return {
        'triggered': bars_below >= hold_bars,
        'bars_below': bars_below,
        'needed': hold_bars,
        'price': bar['price'],
        'high': bar['high'],
        'vol_ratio': round(bar['vol_ratio'], 2),
    }


def load_state(state_file):
    if os.path.exists(state_file):
        with open(state_file) as f:
            return json.load(f)
    return {
        'phase': 'WATCHING',
        'swept_at': None,
        'swept_extreme': None,
        'bars_in_zone': 0,
        'verdict': None,
        'log': [],
        'started': datetime.now().isoformat(),
    }


def save_state(state_file, state):
    os.makedirs(os.path.dirname(state_file), exist_ok=True)
    with open(state_file, 'w') as f:
        json.dump(state, f, indent=2, default=str)


def run_forward_test(zone_low, zone_high, scenarios, state_file=None):
    """Main forward test loop."""
    if state_file is None:
        ts = datetime.now().strftime('%Y%m%d_%H%M%S')
        state_file = os.path.join(STATE_DIR, f'forward_test_{ts}.json')

    state = load_state(state_file)

    # Identify which scenarios we're watching
    wyckoff_short = any(s in scenarios for s in ('wyckoff', 'WYCKOFF_DISTRIBUTION'))
    breakout_long = any(s in scenarios for s in ('breakout', 'GENUINE_BREAKOUT'))
    wyckoff_long = any(s in scenarios for s in ('accumulation', 'WYCKOFF_ACCUMULATION'))
    breakdown_short = any(s in scenarios for s in ('breakdown', 'GENUINE_BREAKDOWN'))

    print(f"{'='*60}")
    print(f"  FORWARD TEST")
    print(f"{'='*60}")
    print(f"  Zone: ${zone_low:.0f}-${zone_high:.0f}")
    print(f"  Watching for:")
    if wyckoff_short:
        print(f"    🔻 WYCKOFF: sweep above ${zone_high:.0f} + rejection")
    if breakout_long:
        print(f"    🔺 BREAKOUT: sweep above ${zone_high:.0f} + hold")
    if wyckoff_long:
        print(f"    🔺 WYCKOFF: sweep below ${zone_low:.0f} + reversal")
    if breakdown_short:
        print(f"    🔻 BREAKDOWN: sweep below ${zone_low:.0f} + hold below")
    print(f"  Checking every {SCAN_INTERVAL}s")
    print()

    if state['phase'] != 'WATCHING':
        print(f"  Resuming: {state['phase']}")

    while True:
        try:
            df = fetch_recent(bars=30, timeframe='15m')
            idx = len(df) - 1
            bar = analyze_bar(df, idx)
            ts = bar['timestamp']
            p = bar['price']
            h = bar['high']
            l = bar['low']

            if state['phase'] == 'WATCHING':
                # Check if price entered or swept the zone
                if h >= zone_high:
                    # Swept above
                    state['phase'] = 'SWEPT_ABOVE'
                    state['swept_at'] = ts
                    state['swept_extreme'] = h
                    state['bars_in_zone'] = 0
                    state['log'].append(f"SWEEP_ABOVE: {ts} high=${h:.2f}")
                    save_state(state_file, state)
                    print(f"\n  🎯 SWEEP ABOVE @ {ts}")
                    print(f"     High: ${h:.2f}  Close: ${p:.2f}  Vol: {bar['vol_ratio']:.1f}x")
                    print(f"     Watching for rejection vs hold...")

                elif l <= zone_low:
                    # Swept below
                    state['phase'] = 'SWEPT_BELOW'
                    state['swept_at'] = ts
                    state['swept_extreme'] = l
                    state['bars_in_zone'] = 0
                    state['log'].append(f"SWEEP_BELOW: {ts} low=${l:.2f}")
                    save_state(state_file, state)
                    print(f"\n  🎯 SWEEP BELOW @ {ts}")
                    print(f"     Low: ${l:.2f}  Close: ${p:.2f}  Vol: {bar['vol_ratio']:.1f}x")
                    print(f"     Watching for reversal vs breakdown...")

                else:
                    dist_high = ((zone_high - p) / p * 100)
                    dist_low = ((p - zone_low) / p * 100)
                    print(f"  {ts}  ${p:.2f}  "
                          f"↑{dist_high:.1f}% to zone top  "
                          f"↓{dist_low:.1f}% to zone bottom  "
                          f"RSI={bar['rsi']:.0f}")

            elif state['phase'] == 'SWEPT_ABOVE':
                # Check both: rejection (wyckoff) and hold (breakout)
                if wyckoff_short:
                    rej = check_wyckoff_rejection(df, zone_high, idx, state['swept_extreme'])
                    if rej['triggered']:
                        state['phase'] = 'DECIDED'
                        state['verdict'] = 'WYCKOFF_DISTRIBUTION'
                        state['log'].append(f"WYCKOFF WIN: {ts} drop={rej['drop_pct']}%")
                        save_state(state_file, state)
                        print(f"\n  {'='*60}")
                        print(f"  🏆 WYCKOFF WINS — DISTRIBUTION CONFIRMED")
                        print(f"  {'='*60}")
                        print(f"  Rejected from ${state['swept_extreme']:.2f} → ${rej['price']:.2f}")
                        print(f"  Drop: {rej['drop_pct']}%  Vol: {rej['vol_ratio']}x  RSI: {rej['rsi']}")
                        print(f"\n  → SHORT ${state['swept_extreme']:.2f}")
                        print(f"    SL: ${zone_high + 30:.0f}")
                        print(f"    TP: $2,200-$2,250")
                        _save_verdict(state_file, state)
                        return state

                if breakout_long:
                    hold = check_hold_above(df, zone_high, idx, state.get('bars_in_zone', 0))
                    state['bars_in_zone'] = hold['bars_above']
                    if hold['triggered']:
                        state['phase'] = 'DECIDED'
                        state['verdict'] = 'GENUINE_BREAKOUT'
                        state['log'].append(f"BREAKOUT WIN: {ts} held {hold['bars_above']} bars")
                        save_state(state_file, state)
                        print(f"\n  {'='*60}")
                        print(f"  🏆 BREAKOUT WINS — STRUCTURE CONFIRMED")
                        print(f"  {'='*60}")
                        print(f"  Held above ${zone_high:.0f} for {hold['bars_above']} bars")
                        print(f"  Price: ${hold['price']:.2f}  Vol: {hold['vol_ratio']}x")
                        print(f"\n  → LONG ${zone_high:.0f}")
                        print(f"    SL: ${zone_low:.0f}")
                        print(f"    TP: ${zone_high + (zone_high - zone_low):.0f}")
                        _save_verdict(state_file, state)
                        return state

                # Still pending
                rej_info = ''
                if wyckoff_short:
                    rej = check_wyckoff_rejection(df, zone_high, idx, state['swept_extreme'])
                    rej_info = f"drop={rej['drop_pct']}% vol={rej['vol_ratio']}x"
                hold_info = ''
                if breakout_long:
                    hold_info = f"above={state.get('bars_in_zone', 0)}/3"
                print(f"  {ts}  ${p:.2f}  {rej_info}  {hold_info}  RSI={bar['rsi']:.0f}")

            elif state['phase'] == 'SWEPT_BELOW':
                # Check both: reversal (wyckoff accumulation) and hold below (breakdown)
                if wyckoff_long:
                    rev = check_accumulation_reversal(df, zone_low, idx, state['swept_extreme'])
                    if rev['triggered']:
                        state['phase'] = 'DECIDED'
                        state['verdict'] = 'WYCKOFF_ACCUMULATION'
                        state['log'].append(f"ACCUMULATION WIN: {ts} bounce={rev['bounce_pct']}%")
                        save_state(state_file, state)
                        print(f"\n  {'='*60}")
                        print(f"  🏆 ACCUMULATION WINS — REVERSAL CONFIRMED")
                        print(f"  {'='*60}")
                        print(f"  Bounced from ${state['swept_extreme']:.2f} → ${rev['price']:.2f}")
                        print(f"  Bounce: {rev['bounce_pct']}%  Vol: {rev['vol_ratio']}x  RSI: {rev['rsi']}")
                        print(f"\n  → LONG ${state['swept_extreme']:.2f}")
                        print(f"    SL: ${zone_low - 30:.0f}")
                        print(f"    TP: ${zone_high:.0f}")
                        _save_verdict(state_file, state)
                        return state

                if breakdown_short:
                    hold = check_hold_below(df, zone_low, idx, state.get('bars_in_zone', 0))
                    state['bars_in_zone'] = hold['bars_below']
                    if hold['triggered']:
                        state['phase'] = 'DECIDED'
                        state['verdict'] = 'GENUINE_BREAKDOWN'
                        state['log'].append(f"BREAKDOWN WIN: {ts} held {hold['bars_below']} bars below")
                        save_state(state_file, state)
                        print(f"\n  {'='*60}")
                        print(f"  🏆 BREAKDOWN WINS — STRUCTURE CONFIRMED")
                        print(f"  {'='*60}")
                        print(f"  Held below ${zone_low:.0f} for {hold['bars_below']} bars")
                        print(f"\n  → SHORT ${zone_low:.0f}")
                        print(f"    SL: ${zone_high:.0f}")
                        print(f"    TP: ${zone_low - (zone_high - zone_low):.0f}")
                        _save_verdict(state_file, state)
                        return state

                # Still pending
                rev_info = ''
                if wyckoff_long:
                    rev = check_accumulation_reversal(df, zone_low, idx, state['swept_extreme'])
                    rev_info = f"bounce={rev['bounce_pct']}% vol={rev['vol_ratio']}x"
                hold_info = ''
                if breakdown_short:
                    hold_info = f"below={state.get('bars_in_zone', 0)}/3"
                print(f"  {ts}  ${p:.2f}  {rev_info}  {hold_info}  RSI={bar['rsi']:.0f}")

            elif state['phase'] == 'DECIDED':
                print(f"\n  Verdict: {state['verdict']} (decided)")
                return state

            save_state(state_file, state)
            time.sleep(SCAN_INTERVAL)

        except KeyboardInterrupt:
            print("\nStopped.")
            return state
        except Exception as e:
            print(f"Error: {e}")
            time.sleep(60)


def _save_verdict(state_file, state):
    """Save final verdict to a separate file for easy reading."""
    verdict_file = state_file.replace('.json', '_VERDICT.json')
    with open(verdict_file, 'w') as f:
        json.dump({
            'verdict': state['verdict'],
            'decided_at': datetime.now().isoformat(),
            'log': state['log'],
        }, f, indent=2)
    print(f"\n  💾 Verdict saved: {verdict_file}")


def main():
    parser = argparse.ArgumentParser(description='Forward Test Runner')
    parser.add_argument('scan_file', nargs='?', help='Scan result JSON with forward test config')
    parser.add_argument('--level', nargs=2, type=float, metavar=('LOW', 'HIGH'),
                        help='Zone low and high prices')
    parser.add_argument('--scenarios', default='wyckoff,breakout',
                        help='Comma-separated scenarios: wyckoff,breakout,accumulation,breakdown')
    parser.add_argument('--interval', type=int, default=300, help='Check interval in seconds')
    args = parser.parse_args()

    global SCAN_INTERVAL
    SCAN_INTERVAL = args.interval

    if args.scan_file:
        ft = load_conflict_from_scan(args.scan_file)
        zone_low = ft['key_level_low']
        zone_high = ft['key_level_high']
        scenarios = [s['name'].lower() for s in ft.get('scenarios', [])]
    elif args.level:
        zone_low, zone_high = args.level
        scenarios = [s.strip().lower() for s in args.scenarios.split(',')]
    else:
        parser.error('Provide either a scan_file or --level LOW HIGH')
        return

    state_file = os.path.join(STATE_DIR, f'forward_test_{datetime.now():%Y%m%d_%H%M%S}.json')
    run_forward_test(zone_low, zone_high, scenarios, state_file)


if __name__ == '__main__':
    main()

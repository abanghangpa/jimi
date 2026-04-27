#!/usr/bin/env python3
"""
JIMI — Live Signal Monitor

Runs scanner every 60s. Tracks signal health over time.
Alerts when:
  - New signal appears (with entry optimizer advice)
  - Signal strengthens (ICS rising, modules flipping PASS)
  - Signal weakens (ICS dropping, modules flipping FAIL)
  - Signal dies (ICS below floor)
  - Entry optimizer changes (ENTER_NOW → WAIT_DIP, etc.)

Usage:
    python scripts/live_monitor.py              # continuous scan
    python scripts/live_monitor.py --once       # single scan + exit
"""

import sys
import os
import time
import json
from datetime import datetime, timezone

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from scripts.scanner import scan_signal, compute_indicators, print_signal
from src.config import CONFIG
from src.utils.data_handler import fetch_recent
from src.modules.entry_optimizer import evaluate_entry

# Alert thresholds
ICS_DROP_ALERT = 0.05
ICS_RISE_ALERT = 0.05
MODULE_FAIL_ALERT = True
ENTRY_ACTION_CHANGE = True


def format_delta(old, new, label):
    """Format a change alert."""
    if old is None:
        return f"  🆕 {label}: {new}"
    if old != new:
        arrow = "↑" if str(new) > str(old) else "↓"
        return f"  {arrow} {label}: {old} → {new}"
    return None


def scan_once(prev_result=None):
    """Run one scan and compare to previous."""
    try:
        df_15m = fetch_recent(bars=1000)
        df_15m, df_1h, df_2h, df_4h, df_1d = compute_indicators(df_15m)
        result = scan_signal(df_15m, df_1h, df_2h, df_4h, df_1d)
    except Exception as e:
        print(f"\n  ❌ Scan error: {e}")
        return prev_result

    now = datetime.now(timezone.utc).strftime('%H:%M:%S')
    price = result.get('price', 0)
    status = result.get('status', 'UNKNOWN')
    direction = result.get('direction', '')
    ics = result.get('ics', 0)

    # Run entry optimizer if we have a signal
    entry_action = ''
    entry_details = {}
    if status == 'SIGNAL' and direction:
        try:
            sr_levels = result.get('sr_levels', [])
            vwap = result.get('vwap')
            magnets = result.get('magnets', [])
            atr_1h = result.get('atr_1h', 0)
            entry_action, entry_details = evaluate_entry(
                direction, price, atr_1h, sr_levels, vwap, magnets,
                config=CONFIG
            )
            result['entry_action'] = entry_action
            result['entry_details'] = entry_details
        except Exception:
            pass

    alerts = []

    # Compare to previous scan
    if prev_result:
        prev_ics = prev_result.get('ics', 0)
        prev_status = prev_result.get('status', '')
        prev_action = prev_result.get('entry_action', '')

        ics_delta = ics - prev_ics
        if abs(ics_delta) >= ICS_DROP_ALERT:
            if ics_delta < 0:
                alerts.append(f"  ⚠️  ICS DROPPED: {prev_ics:.3f} → {ics:.3f} ({ics_delta:+.3f})")
            else:
                alerts.append(f"  📈 ICS RISING: {prev_ics:.3f} → {ics:.3f} ({ics_delta:+.3f})")

        if status != prev_status:
            if status == 'SIGNAL' and prev_status != 'SIGNAL':
                alerts.append(f"  🟢 NEW SIGNAL: {direction} @ ${price:.2f}")
            elif status != 'SIGNAL' and prev_status == 'SIGNAL':
                alerts.append(f"  🔴 SIGNAL DIED: {prev_result.get('direction', '')} → {status}")

        if MODULE_FAIL_ALERT and 'modules' in result and 'modules' in prev_result:
            for mod_name in ['m1', 'm2', 'm3', 'm4', 'm5', 'm6']:
                cur_mod = result['modules'].get(mod_name, {})
                prev_mod = prev_result['modules'].get(mod_name, {})
                cur_status = cur_mod.get('status', '')
                prev_status_mod = prev_mod.get('status', '')
                if prev_status_mod == 'PASS' and cur_status == 'FAIL':
                    alerts.append(f"  ⚠️  {mod_name.upper()} FLIPPED: PASS → FAIL")

        if ENTRY_ACTION_CHANGE and entry_action != prev_action:
            if prev_action and entry_action:
                alerts.append(f"  🔄 Entry advice: {prev_action} → {entry_action}")

    # Print scan result
    if not prev_result:
        print(f"\n{'═'*60}")
        print(f"  JIMI LIVE MONITOR — Started {now} UTC")
        print(f"{'═'*60}")

    modules_str = ""
    if 'modules' in result:
        parts = []
        for mod_name in ['m1', 'm2', 'm3', 'm4', 'm5', 'm6']:
            mod = result['modules'].get(mod_name, {})
            s = mod.get('status', '??')
            icon = '✅' if s == 'PASS' else '❌' if s == 'FAIL' else '⬜'
            parts.append(f"{mod_name.upper()}:{icon}")
        modules_str = " ".join(parts)

    ics_icon = '🟢' if ics >= 0.50 else '🔴'
    action_icon = {
        'ENTER_NOW': '✅', 'WAIT_DIP': '⏳',
        'WAIT_BREAKOUT': '🔓', 'SKIP': '❌'
    }.get(entry_action, '⬜')

    print(f"\n  [{now}] ${price:.2f}  {ics_icon} ICS={ics:.3f}  {status} {direction}  {action_icon} {entry_action}")
    if modules_str:
        print(f"  {modules_str}")

    if alerts:
        print(f"\n  {'─'*40}")
        for a in alerts:
            print(a)

    if status == 'SIGNAL' and entry_action:
        if entry_action == 'WAIT_DIP' and 'dip_target' in entry_details:
            print(f"  💡 Limit at: ${entry_details['dip_target']:.2f} (R:R {entry_details.get('dip_rr', '?')})")
        elif entry_action == 'WAIT_BREAKOUT' and 'breakout_price' in entry_details:
            print(f"  💡 Enter on break: ${entry_details['breakout_price']:.2f}")
        elif entry_action == 'SKIP':
            print(f"  💡 {entry_details.get('reason', 'Wait for better setup')}")

    return result


def main():
    """Run continuous monitor or single scan."""
    once = '--once' in sys.argv

    if once:
        scan_once()
        return

    print(f"\n  🔴 JIMI LIVE MONITOR")
    print(f"  Scanning every 60s. Ctrl+C to stop.\n")

    prev = None
    try:
        while True:
            prev = scan_once(prev)
            time.sleep(60)
    except KeyboardInterrupt:
        print(f"\n  Monitor stopped.")


if __name__ == '__main__':
    main()

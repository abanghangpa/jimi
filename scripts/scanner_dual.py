#!/usr/bin/env python3
"""
JIMI Dual Strategy Scanner — Range Scalper + Momentum Rider

Usage:
    python scripts/scanner_dual.py
    python scripts/scanner_dual.py --json
"""

import argparse
import sys
import os
import json
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from src.config import CONFIG
from src.utils.data_handler import fetch_recent, resample_ohlcv, load_data
from src.utils.indicators import (
    calc_ema, calc_macd, calc_rsi, calc_atr, calc_vwap, calc_vol_ratio,
    calc_swing_bias, calc_phase0, calc_trend_state,
)
from src.dual_strategy import DualStrategy
from scripts.scanner import compute_indicators, ensure_csv_fresh, load_daily_from_csv


def print_dual_result(result):
    """Print combined dual strategy result."""
    base = result.get('base', {})
    sa = result.get('strategy_a') or {}
    sb = result.get('strategy_b') or {}

    print("\n" + "═" * 60)
    print("  JIMI — DUAL STRATEGY SCAN")
    print("═" * 60)
    print(f"\n  Time:     {base.get('timestamp', '?')}")
    print(f"  Price:    ${base.get('price', 0):.2f}")
    print(f"  Regime:   {base.get('regime', '?')}")
    print(f"  Bias:     {base.get('swing_bias', '?')}")
    print(f"  Trend:    {base.get('trend_dir', '?')}")
    print(f"  Structure: {base.get('m13_bias', '?')}")
    print(f"  Direction: {base.get('direction', '?')}")

    # Strategy A
    print(f"\n  {'─' * 56}")
    print(f"  📐 STRATEGY A: RANGE SCALPER")
    print(f"  {'─' * 56}")
    _print_strategy(sa, 'scalp')

    # Strategy B
    print(f"\n  {'─' * 56}")
    print(f"  🚀 STRATEGY B: MOMENTUM RIDER")
    print(f"  {'─' * 56}")
    _print_strategy(sb, 'momentum')

    # Combined verdict
    print(f"\n  {'═' * 56}")
    signals = []
    if sa.get('status') == 'SIGNAL':
        signals.append(f"SCALP {sa['direction']} ${sa['entry']:.2f} → TP1 ${sa['tp1']:.2f}")
    if sb.get('status') == 'SIGNAL':
        signals.append(f"MOMENTUM {sb['direction']} ${sb['entry']:.2f} → TP1 ${sb['tp1']:.2f}")

    if signals:
        print(f"  ✅ ACTIVE SIGNALS:")
        for s in signals:
            print(f"     {s}")
    else:
        reasons = []
        if sa.get('reason'):
            reasons.append(f"Scalp: {sa['reason']}")
        if sb.get('reason'):
            reasons.append(f"Mom: {sb['reason']}")
        print(f"  ⛔ NO SIGNALS — {'; '.join(reasons)}")
    print(f"  {'═' * 56}")


def _print_strategy(s, mode):
    """Print one strategy's result."""
    if not s:
        print(f"  ⚪ No data")
        return

    status = s.get('status', '?')
    if status == 'SIGNAL':
        direction = s.get('direction', '?')
        entry = s.get('entry', 0)
        sl = s.get('sl', 0)
        tp1 = s.get('tp1', 0)
        tp2 = s.get('tp2', 0)
        tp3 = s.get('tp3', 0)
        sl_pct = s.get('sl_pct', 0)
        tp1_pct = s.get('tp1_pct', 0)
        ics = s.get('ics', 0)
        size = s.get('size', 0)
        entry_mode = s.get('mode', '')

        print(f"  ✅ SIGNAL: {direction}")
        if 'pullback' in entry_mode:
            print(f"     📐 Entry Mode: PULLBACK RETEST")
            print(f"     📐 Retrace: {s.get('pullback_retrace', 0)*100:.1f}%  Wait: {s.get('pullback_bars', 0)} bars")
        print(f"     Entry: ${entry:.2f}  (size={size:.1f})")
        print(f"     SL:    ${sl:.2f}  ({sl_pct:.2f}%)")
        print(f"     TP1:   ${tp1:.2f}  ({tp1_pct:.2f}%)", end="")
        if mode == 'momentum':
            tp1_close = s.get('tp1_close', 0.15)
            print(f"  [close {tp1_close*100:.0f}%]")
        else:
            print(f"  [close 30%]")
        print(f"     TP2:   ${tp2:.2f}")
        print(f"     TP3:   ${tp3:.2f}")
        print(f"     ICS:   {ics:.4f}  Regime: {s.get('regime', '?')}")

        if mode == 'momentum':
            strength = s.get('momentum_strength', 0)
            reason = s.get('momentum_reason', '')
            rr = abs(tp1_pct / sl_pct) if sl_pct > 0 else 0
            print(f"     📊 Momentum: {strength:.2f}  ({reason})")
            print(f"     📊 R:R:     {rr:.2f}x (TP1 vs SL)")

    elif status == 'FILTERED':
        print(f"  🔶 FILTERED: {s.get('reason', '?')}")
    else:
        reason = s.get('reason', '?')
        entry_mode = s.get('mode', '')
        if 'pullback_pending' in entry_mode:
            strength = s.get('strength', 0)
            print(f"  ⏳ PENDING: {reason}")
            print(f"     📊 Ignition strength: {strength:.2f}")
        else:
            print(f"  ⛔ NO SIGNAL: {reason}")


def main():
    parser = argparse.ArgumentParser(description='JIMI Dual Strategy Scanner')
    parser.add_argument('--json', action='store_true', help='Output JSON')
    args = parser.parse_args()

    csv_path = ensure_csv_fresh()
    df_1d_hist = load_daily_from_csv(csv_path)

    print("Fetching 15m data (1000 bars)...")
    df_base = fetch_recent(bars=1000, timeframe='15m')
    print("Computing indicators...")
    df_base, df_1h, df_2h, df_4h, df_1d = compute_indicators(
        df_base, config=CONFIG, df_1d_hist=df_1d_hist)

    print("Running dual strategy scan...")
    ds = DualStrategy(config=CONFIG)
    result = ds.scan(df_base, df_1h, df_2h, df_4h, df_1d)

    # Save
    scan_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data', 'scans')
    os.makedirs(scan_dir, exist_ok=True)
    from datetime import datetime
    ts = datetime.now().strftime('%Y%m%d_%H%M%S')
    scan_file = os.path.join(scan_dir, f'scan_dual_{ts}.json')
    with open(scan_file, 'w') as f:
        json.dump(result, f, indent=2, default=str)
    print(f"\n  💾 Saved: {scan_file}")

    if args.json:
        print(json.dumps(result, indent=2, default=str))
    else:
        print_dual_result(result)


if __name__ == '__main__':
    main()

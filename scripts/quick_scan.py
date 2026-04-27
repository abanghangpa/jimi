#!/usr/bin/env python3
"""
Quick scanner — outputs signal summary for cron delivery.
Logs signals, checks outcomes, computes live probabilities.
Auto-commits and pushes log updates to GitHub.

Usage:
    python scripts/quick_scan.py
"""

import sys
import os
import subprocess
import json
from datetime import datetime, timezone

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from scripts.scanner import scan_signal, compute_indicators
from src.config import CONFIG
from src.utils.data_handler import fetch_recent
from src.utils.signal_tracker import (
    log_signal, check_outcomes, compute_stats,
    get_live_probabilities, format_live_prob,
    get_open_signals,
)

REPO_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def git_push_log():
    """Commit and push log/stats changes to GitHub."""
    try:
        for fname in ["live_signals.jsonl", "live_stats.json"]:
            fpath = os.path.join(REPO_DIR, fname)
            if os.path.exists(fpath):
                subprocess.run(
                    ["git", "add", fname],
                    cwd=REPO_DIR, capture_output=True, timeout=10
                )

        result = subprocess.run(
            ["git", "diff", "--cached", "--quiet"],
            cwd=REPO_DIR, capture_output=True, timeout=10
        )
        if result.returncode == 0:
            return  # Nothing to commit

        ts = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
        subprocess.run(
            ["git", "commit", "-m", f"📊 signal log update {ts}"],
            cwd=REPO_DIR, capture_output=True, timeout=10
        )
        subprocess.run(
            ["git", "push"],
            cwd=REPO_DIR, capture_output=True, timeout=30
        )
    except Exception as e:
        print(f"⚠️ git push failed: {e}", file=sys.stderr)


def quick_scan():
    df_15m = fetch_recent(bars=1000)
    df_15m, df_1h, df_2h, df_4h, df_1d = compute_indicators(df_15m)
    r = scan_signal(df_15m, df_1h, df_2h, df_4h, df_1d)

    current_price = r.get("price", 0)

    # Check outcomes of open signals
    resolved = check_outcomes(current_price)
    if resolved:
        for sig in resolved:
            icon = "✅" if sig["exit_reason"].startswith("TP") else "⏰" if sig["exit_reason"] == "STALE" else "🛑"
            print(f"{icon} RESOLVED: {sig['direction']} ${sig['entry']:.2f} → "
                  f"{sig['exit_reason']} (pnl {sig.get('pnl_pct', 0):+.2f}%)")

    # No signal case
    if r['status'] != 'SIGNAL':
        stats = compute_stats()
        open_sigs = get_open_signals()
        open_count = len(open_sigs)

        summary = f"⏳ NO SIGNAL — {r.get('reason', 'N/A')} | ETH ${current_price:.2f}"
        if open_count > 0:
            summary += f" | {open_count} open"
        if stats["total"] >= 10:
            summary += f" | Live WR: {stats['win_rate']*100:.0f}% (n={stats['total']})"

        print(summary)
        git_push_log()
        return

    # Signal found
    d = r['direction']
    ics = r['ics']
    entry = r['entry']
    sl = r['sl']
    tp1 = r['tp1']
    tp2 = r['tp2']
    tp3 = r['tp3']
    sl_pct = r['sl_pct']
    tp1_pct = r['tp1_pct']

    m1 = r['m1']['direction']
    m4 = r['m4']['status']
    m5 = r['m5']['status']
    m5_s = r['m5']['score']
    m7 = r.get('m7', {}).get('status', 'SKIP')
    m7_s = r.get('m7', {}).get('score', 0)

    deriv = r.get('derivatives', {})
    pos = deriv.get('positioning', 'N/A')
    fr = deriv.get('funding_rate')
    fr_str = f"{fr*100:+.4f}%" if fr else "N/A"

    action = r.get('entry_action', '')
    details = r.get('entry_details', {})
    opt_str = ""
    if action == 'WAIT_BREAKOUT':
        brk = details.get('breakout_price', '')
        opt_str = f"⏳ Wait for break above ${brk:.2f}" if brk else "⏳ Wait for breakout"
    elif action == 'WAIT_DIP':
        dip = details.get('dip_target', '')
        opt_str = f"⏳ Wait for dip to ${dip:.2f}" if dip else "⏳ Wait for dip"
    elif action == 'ENTER_NOW':
        opt_str = "✅ Enter now"

    # Log this signal
    sig = log_signal(r)

    # Compute live probabilities
    stats = compute_stats()
    live_probs = get_live_probabilities(r, stats)
    prob_str = format_live_prob(live_probs)

    lines = [
        f"{'🟢' if d == 'LONG' else '🔴'} {d} SIGNAL — ETH ${entry:.2f}",
        f"ICS: {ics:.3f} | M1:{m1} M4:{m4} M5:{m5}({m5_s:.2f}) M7:{m7}({m7_s:.2f})",
        f"SL: ${sl:.2f} ({sl_pct:.2f}%) | TP1: ${tp1:.2f} ({tp1_pct:.2f}%) | TP2: ${tp2:.2f} | TP3: ${tp3:.2f}",
        f"Derivs: {pos} | Funding: {fr_str}",
    ]
    if opt_str:
        lines.append(opt_str)
    lines.append(prob_str)

    print("\n".join(lines))

    git_push_log()


if __name__ == '__main__':
    quick_scan()

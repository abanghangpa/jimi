"""
JIMI Signal Tracker — Live session performance tracking.

Logs every signal, resolves TP/SL outcomes against live price,
computes rolling hit rates for probability estimates.

Data: live_signals.jsonl (one JSON object per line)
"""

import json
import os
import time
from datetime import datetime, timezone
from pathlib import Path

# Data files live alongside this module
DATA_DIR = Path(__file__).parent.parent.parent  # repo root
LOG_FILE = DATA_DIR / "live_signals.jsonl"
STATS_FILE = DATA_DIR / "live_stats.json"

# How long (seconds) to keep checking a signal before marking stale
MAX_AGE_SECS = 48 * 3600  # 48 hours


def _now_iso():
    return datetime.now(timezone.utc).isoformat()


def _now_ts():
    return time.time()


def _load_signals():
    """Load all signals from JSONL."""
    if not LOG_FILE.exists():
        return []
    signals = []
    with open(LOG_FILE) as f:
        for line in f:
            line = line.strip()
            if line:
                signals.append(json.loads(line))
    return signals


def _save_signals(signals):
    """Overwrite JSONL with updated signals."""
    with open(LOG_FILE, "w") as f:
        for s in signals:
            f.write(json.dumps(s) + "\n")


# ═══════════════════════════════════════════════════════════════════
# PUBLIC API
# ═══════════════════════════════════════════════════════════════════

def log_signal(scan_result):
    """
    Log a new signal from the scanner output.
    Returns the signal dict with a unique ID.
    """
    signals = _load_signals()

    sig = {
        "id": f"sig_{int(_now_ts())}",
        "logged_at": _now_iso(),
        "logged_ts": _now_ts(),
        "status": "OPEN",
        "direction": scan_result["direction"],
        "entry": scan_result["entry"],
        "sl": scan_result["sl"],
        "tp1": scan_result["tp1"],
        "tp2": scan_result["tp2"],
        "tp3": scan_result["tp3"],
        "sl_pct": scan_result.get("sl_pct", 0),
        "tp1_pct": scan_result.get("tp1_pct", 0),
        "ics": scan_result["ics"],
        "m1": scan_result.get("m1", {}).get("direction", ""),
        "m4": scan_result.get("m4", {}).get("status", ""),
        "m5": scan_result.get("m5", {}).get("status", ""),
        "m5_score": scan_result.get("m5", {}).get("score", 0),
        "m7": scan_result.get("m7", {}).get("status", ""),
        "m7_score": scan_result.get("m7", {}).get("score", 0),
        "positioning": scan_result.get("derivatives", {}).get("positioning", ""),
        "funding_rate": scan_result.get("derivatives", {}).get("funding_rate"),
        "entry_action": scan_result.get("entry_action", ""),
        "entry_details": scan_result.get("entry_details", {}),
        "resolved_at": None,
        "exit_price": None,
        "exit_reason": None,
        "pnl_pct": None,
    }

    signals.append(sig)
    _save_signals(signals)
    return sig


def check_outcomes(current_price):
    """
    Check all OPEN signals against current price.
    Resolves TP/SL hits. Returns list of newly resolved signals.
    """
    signals = _load_signals()
    resolved = []
    now = _now_ts()
    changed = False

    for sig in signals:
        if sig["status"] != "OPEN":
            continue

        age = now - sig["logged_ts"]
        if age > MAX_AGE_SECS:
            sig["status"] = "STALE"
            sig["resolved_at"] = _now_iso()
            sig["exit_price"] = current_price
            sig["exit_reason"] = "STALE"
            if sig["direction"] == "LONG":
                sig["pnl_pct"] = (current_price - sig["entry"]) / sig["entry"] * 100
            else:
                sig["pnl_pct"] = (sig["entry"] - current_price) / sig["entry"] * 100
            resolved.append(sig)
            changed = True
            continue

        entry = sig["entry"]
        sl = sig["sl"]
        tp1, tp2, tp3 = sig["tp1"], sig["tp2"], sig["tp3"]
        d = sig["direction"]

        hit = None
        if d == "LONG":
            if current_price >= tp3:
                hit = ("TP3", tp3)
            elif current_price >= tp2:
                hit = ("TP2", tp2)
            elif current_price >= tp1:
                hit = ("TP1", tp1)
            elif current_price <= sl:
                hit = ("SL", sl)
        else:
            if current_price <= tp3:
                hit = ("TP3", tp3)
            elif current_price <= tp2:
                hit = ("TP2", tp2)
            elif current_price <= tp1:
                hit = ("TP1", tp1)
            elif current_price >= sl:
                hit = ("SL", sl)

        if hit:
            reason, exit_p = hit
            sig["status"] = "CLOSED"
            sig["resolved_at"] = _now_iso()
            sig["exit_price"] = exit_p
            sig["exit_reason"] = reason
            if d == "LONG":
                sig["pnl_pct"] = (exit_p - entry) / entry * 100
            else:
                sig["pnl_pct"] = (entry - exit_p) / entry * 100
            resolved.append(sig)
            changed = True

    if changed:
        _save_signals(signals)

    return resolved


def compute_stats(window_days=30):
    """
    Compute live performance stats from resolved signals.
    Returns dict with hit rates, usable for probability estimates.
    """
    signals = _load_signals()
    now = _now_ts()
    cutoff = now - (window_days * 86400)

    resolved = [
        s for s in signals
        if s["status"] in ("CLOSED", "STALE") and s["logged_ts"] >= cutoff
    ]

    total = len(resolved)
    if total == 0:
        return {
            "total": 0, "tp1_rate": None, "tp2_rate": None, "tp3_rate": None,
            "sl_rate": None, "stale_rate": None, "win_rate": None,
            "avg_pnl": None, "by_ics": {}, "by_direction": {}, "by_modules": {},
        }

    tp1_hits = sum(1 for s in resolved if s["exit_reason"] in ("TP1", "TP2", "TP3"))
    tp2_hits = sum(1 for s in resolved if s["exit_reason"] in ("TP2", "TP3"))
    tp3_hits = sum(1 for s in resolved if s["exit_reason"] == "TP3")
    sl_hits = sum(1 for s in resolved if s["exit_reason"] == "SL")
    stale = sum(1 for s in resolved if s["exit_reason"] == "STALE")
    wins = sum(1 for s in resolved if (s.get("pnl_pct") or 0) > 0)
    pnls = [s["pnl_pct"] for s in resolved if s.get("pnl_pct") is not None]

    stats = {
        "total": total,
        "window_days": window_days,
        "computed_at": _now_iso(),
        "tp1_rate": round(tp1_hits / total, 4),
        "tp2_rate": round(tp2_hits / total, 4),
        "tp3_rate": round(tp3_hits / total, 4),
        "sl_rate": round(sl_hits / total, 4),
        "stale_rate": round(stale / total, 4),
        "win_rate": round(wins / total, 4),
        "avg_pnl": round(sum(pnls) / len(pnls), 4) if pnls else None,
    }

    # Breakdown by ICS bucket
    by_ics = {}
    for lo, hi in [(0.5, 0.55), (0.55, 0.6), (0.6, 0.65), (0.65, 0.7), (0.7, 1.0)]:
        bucket = [s for s in resolved if lo <= s["ics"] < hi]
        if bucket:
            n = len(bucket)
            by_ics[f"{lo:.2f}-{hi:.2f}"] = {
                "n": n,
                "tp1": round(sum(1 for s in bucket if s["exit_reason"] in ("TP1", "TP2", "TP3")) / n, 4),
                "tp2": round(sum(1 for s in bucket if s["exit_reason"] in ("TP2", "TP3")) / n, 4),
                "tp3": round(sum(1 for s in bucket if s["exit_reason"] == "TP3") / n, 4),
                "sl": round(sum(1 for s in bucket if s["exit_reason"] == "SL") / n, 4),
            }
    stats["by_ics"] = by_ics

    # Breakdown by direction
    by_dir = {}
    for d in ["LONG", "SHORT"]:
        bucket = [s for s in resolved if s["direction"] == d]
        if bucket:
            n = len(bucket)
            by_dir[d] = {
                "n": n,
                "tp1": round(sum(1 for s in bucket if s["exit_reason"] in ("TP1", "TP2", "TP3")) / n, 4),
                "tp2": round(sum(1 for s in bucket if s["exit_reason"] in ("TP2", "TP3")) / n, 4),
                "tp3": round(sum(1 for s in bucket if s["exit_reason"] == "TP3") / n, 4),
                "sl": round(sum(1 for s in bucket if s["exit_reason"] == "SL") / n, 4),
            }
    stats["by_direction"] = by_dir

    # Breakdown by module consensus
    all_pass = [s for s in resolved if s["m4"] == "PASS" and s["m5"] == "PASS"]
    any_fail = [s for s in resolved if s["m4"] == "FAIL" or s["m5"] == "FAIL"]
    by_mod = {}
    for label, bucket in [("all_pass", all_pass), ("any_fail", any_fail)]:
        if bucket:
            n = len(bucket)
            by_mod[label] = {
                "n": n,
                "tp1": round(sum(1 for s in bucket if s["exit_reason"] in ("TP1", "TP2", "TP3")) / n, 4),
                "tp2": round(sum(1 for s in bucket if s["exit_reason"] in ("TP2", "TP3")) / n, 4),
                "tp3": round(sum(1 for s in bucket if s["exit_reason"] == "TP3") / n, 4),
                "sl": round(sum(1 for s in bucket if s["exit_reason"] == "SL") / n, 4),
            }
    stats["by_modules"] = by_mod

    with open(STATS_FILE, "w") as f:
        json.dump(stats, f, indent=2)

    return stats


def get_live_probabilities(scan_result, stats=None):
    """
    Estimate TP probabilities from live session stats.
    Falls back to base rates when insufficient data.
    """
    if stats is None:
        stats = compute_stats()

    total = stats.get("total", 0)
    MIN_SAMPLE = 10

    if total < MIN_SAMPLE:
        return None

    ics = scan_result["ics"]
    direction = scan_result["direction"]
    m4 = scan_result.get("m4", {}).get("status", "")
    m5 = scan_result.get("m5", {}).get("status", "")

    candidates = []

    ics_key = None
    for lo, hi in [(0.5, 0.55), (0.55, 0.6), (0.6, 0.65), (0.65, 0.7), (0.7, 1.0)]:
        if lo <= ics < hi:
            ics_key = f"{lo:.2f}-{hi:.2f}"
            break

    if ics_key and ics_key in stats.get("by_ics", {}):
        bucket = stats["by_ics"][ics_key]
        if bucket["n"] >= 3:
            candidates.append(("ics_bucket", bucket))

    if direction in stats.get("by_direction", {}):
        bucket = stats["by_direction"][direction]
        if bucket["n"] >= 3:
            candidates.append(("direction", bucket))

    mod_key = "all_pass" if (m4 == "PASS" and m5 == "PASS") else "any_fail"
    if mod_key in stats.get("by_modules", {}):
        bucket = stats["by_modules"][mod_key]
        if bucket["n"] >= 3:
            candidates.append(("modules", bucket))

    candidates.append(("overall", {
        "n": total,
        "tp1": stats["tp1_rate"],
        "tp2": stats["tp2_rate"],
        "tp3": stats["tp3_rate"],
        "sl": stats["sl_rate"],
    }))

    weights = [0.4, 0.3, 0.2, 0.1][:len(candidates)]
    w_sum = sum(weights)
    weights = [w / w_sum for w in weights]

    tp1 = sum(c[1]["tp1"] * w for c, w in zip(candidates, weights))
    tp2 = sum(c[1]["tp2"] * w for c, w in zip(candidates, weights))
    tp3 = sum(c[1]["tp3"] * w for c, w in zip(candidates, weights))
    sl = sum(c[1]["sl"] * w for c, w in zip(candidates, weights))

    return {
        "tp1_pct": round(tp1 * 100, 1),
        "tp2_pct": round(tp2 * 100, 1),
        "tp3_pct": round(tp3 * 100, 1),
        "sl_pct": round(sl * 100, 1),
        "sample_size": total,
        "sources": [c[0] for c in candidates],
    }


def get_open_signals():
    """Return list of currently open signals."""
    signals = _load_signals()
    return [s for s in signals if s["status"] == "OPEN"]


def get_recent_signals(n=20):
    """Return last N signals (any status)."""
    signals = _load_signals()
    return signals[-n:]


def format_live_prob(probs):
    """Format probability dict for display."""
    if probs is None:
        return "📊 Live prob: insufficient data (need ≥10 resolved signals)"
    return (
        f"📊 Live prob (n={probs['sample_size']}): "
        f"TP1 {probs['tp1_pct']:.0f}% | TP2 {probs['tp2_pct']:.0f}% | "
        f"TP3 {probs['tp3_pct']:.0f}% | SL {probs['sl_pct']:.0f}%"
    )


if __name__ == "__main__":
    import sys
    if "--stats" in sys.argv:
        stats = compute_stats()
        print(json.dumps(stats, indent=2))
    elif "--open" in sys.argv:
        for s in get_open_signals():
            print(f"  {s['direction']} ${s['entry']:.2f} ICS={s['ics']:.3f} [{s['id']}]")
    elif "--recent" in sys.argv:
        for s in get_recent_signals(10):
            icon = {"OPEN": "🟡", "CLOSED": "✅", "STALE": "⏰"}.get(s["status"], "❓")
            print(f"  {icon} {s['direction']} ${s['entry']:.2f} → {s.get('exit_reason', '?')} "
                  f"pnl={s.get('pnl_pct', '?')}% [{s['id']}]")
    else:
        print("Usage: python3 signal_tracker.py [--stats|--open|--recent]")

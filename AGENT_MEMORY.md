# AGENT_MEMORY.md — Persistent AI Notes

This file persists across sessions. The workspace (`~/.openclaw/workspace/`) gets wiped periodically, so anything important lives here in the repo.

## Project Context
- **Repo:** https://github.com/abanghangpa/jimi
- **What:** ETH/USDT 15m scoring system with live scanner and derivatives data collection
- **Key files:** `scripts/scanner.py`, `src/modules/m6_derivatives.py`, `scripts/collect_derivatives.py`

## Derivatives Data Collection
- **Collector:** `scripts/collect_derivatives.py` — snapshots OI, L/S ratio, taker buy/sell, funding rate from Binance FAPI
- **Cron:** `jimi-derivatives-collector` runs every 15m on the gateway (gated behind lockfile)
- **Lockfile:** `data/derivatives_history/.collect_lock` — created after first scanner.py run
- **CSV:** `data/derivatives_history/derivatives_collected.csv`
- **Push tracking:** `data/derivatives_history/.last_pushed_count`
- **Note:** Binance API only keeps ~30 days of derivatives history.

## Daily Push Reminder
- **Cron:** `jimi-push-reminder` fires at 10:00 AM (Asia/Shanghai) daily
- **Purpose:** Reminds user to push data files to GitHub

## Rules for AI
1. **Workspace is temporary.** Only GitHub is permanent. Always push important work.
2. **When accessing the jimi GitHub repo** (git pull/push/fetch, any gh command), remind the user to push the latest data files and show what's changed.
3. **After cloning/re-cloning the workspace**, read this file first to restore context.
4. **User controls pushes.** Don't auto-push — just remind and show what's available.

## Gitignore Notes
- `data/` and `*.csv` are gitignored
- Exception: `data/eth_15m_intrabar_delta.csv`
- Derivatives CSV needs explicit add or gitignore update to be pushed

## Session Log
_(Append notes from each session below this line)_

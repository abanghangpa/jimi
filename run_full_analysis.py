#!/usr/bin/env python3
"""Full 12-month v6.13 analysis — all months, 2020-2025."""

import ccxt
import pandas as pd
import numpy as np
import sys, os, json, calendar
sys.path.insert(0, os.path.dirname(__file__))
exchange = ccxt.binance({'enableRateLimit': True})

def fetch_ohlcv(symbol, timeframe, since_ms, until_ms):
    all_candles = []
    current = since_ms
    while current < until_ms:
        raw = exchange.fetch_ohlcv(symbol, timeframe, since=current, limit=1000)
        if not raw:
            break
        for c in raw:
            ts = int(c[0])
            if ts >= until_ms:
                break
            all_candles.append({
                'Open time': pd.to_datetime(ts, unit='ms'),
                'Open': float(c[1]), 'High': float(c[2]),
                'Low': float(c[3]), 'Close': float(c[4]),
                'Volume': float(c[5]),
                'Close time': pd.to_datetime(int(c[6]), unit='ms') if len(c) > 6 else pd.to_datetime(ts + 900000, unit='ms'),
                'Quote asset volume': float(c[7]) if len(c) > 7 else 0,
                'Number of trades': int(c[8]) if len(c) > 8 else 0,
                'Taker buy base asset volume': float(c[9]) if len(c) > 9 else 0,
                'Taker buy quote asset volume': float(c[10]) if len(c) > 10 else 0,
            })
        last_ts = raw[-1][0]
        if last_ts <= current:
            break
        current = last_ts + 1
    return pd.DataFrame(all_candles)

from jimi_v613 import run_backtest, print_report

# Month configs: (month_num, fetch_start_month, fetch_start_day, year_offset)
MONTHS = {
    'january':   (1,  12, 15, 1),
    'february':  (2,  1,  15, 0),
    'march':     (3,  1,  15, 0),
    'april':     (4,  3,  15, 0),
    'may':       (5,  4,  15, 0),
    'june':      (6,  5,  15, 0),
    'july':      (7,  5,  15, 0),
    'august':    (8,  7,  15, 0),
    'september': (9,  8,  15, 0),
    'october':   (10, 9,  15, 0),
    'november':  (11, 10, 15, 0),
    'december':  (12, 11, 15, 0),
}

all_results = {}

for month_name, (mn, fsm, fsd, yoff) in MONTHS.items():
    month_results = []
    print(f"\n{'='*60}")
    print(f"  {month_name.upper()} — v6.13 Analysis")
    print(f"{'='*60}")

    for year in [2025, 2026]:
        fetch_start_year = year - 1 if yoff else year
        last_day = calendar.monthrange(year, mn)[1]

        start = pd.Timestamp(f'{fetch_start_year}-{fsm:02d}-{fsd:02d}', tz='UTC')
        end = pd.Timestamp(f'{year}-{mn:02d}-{last_day}', tz='UTC')
        since_ms = int(start.timestamp() * 1000)
        until_ms = int(end.timestamp() * 1000)

        date_start = f'{year}-{mn:02d}-01'
        date_end = f'{year}-{mn:02d}-{last_day}'

        try:
            df = fetch_ohlcv('ETH/USDT', '15m', since_ms, until_ms)
            csv_path = f'/tmp/eth_15m_{month_name}_{year}.csv'
            df.to_csv(csv_path, index=False)

            trades, stats, _ = run_backtest(csv_path, verbose=False,
                                           date_start=date_start, date_end=date_end)

            total = len(trades)
            if total == 0:
                month_results.append({'year': year, 'trades': 0, 'wins': 0, 'losses': 0,
                    'wr': 0, 'pnl': 0, 'pf': 0, 'max_dd': 0, 'longs': 0, 'shorts': 0,
                    'tp1': 0, 'tp2': 0, 'tp3': 0, 'sl': 0})
                continue

            winners = [t for t in trades if t.pnl_pct > 0]
            losers = [t for t in trades if t.pnl_pct < 0]
            win_rate = len(winners) / total * 100
            total_pnl = sum(t.pnl_pct * t.size_pct for t in trades)
            gross_profit = sum(t.pnl_pct * t.size_pct for t in winners)
            gross_loss = abs(sum(t.pnl_pct * t.size_pct for t in losers))
            profit_factor = gross_profit / gross_loss if gross_loss > 0 else float('inf')

            equity = [0]
            for t in sorted(trades, key=lambda x: x.exit_time):
                equity.append(equity[-1] + t.pnl_pct * t.size_pct)
            equity = np.array(equity)
            peak = np.maximum.accumulate(equity)
            max_dd = abs((equity - peak).min()) if len(equity) > 0 else 0

            longs = [t for t in trades if t.direction == 'LONG']
            shorts = [t for t in trades if t.direction == 'SHORT']
            tp1_count = len([t for t in trades if t.tp1_hit])
            tp2_count = len([t for t in trades if t.tp2_hit])
            tp3_count = len([t for t in trades if t.exit_reason == 'TP3'])
            sl_count = len([t for t in trades if t.exit_reason == 'SL'])

            result = {
                'year': year, 'trades': total, 'wins': len(winners), 'losses': len(losers),
                'wr': win_rate, 'pnl': total_pnl * 100, 'pf': profit_factor,
                'max_dd': max_dd * 100, 'longs': len(longs), 'shorts': len(shorts),
                'tp1': tp1_count, 'tp2': tp2_count, 'tp3': tp3_count, 'sl': sl_count,
            }
            month_results.append(result)
            print(f"  {year}: {total}T, {len(winners)}/{len(losers)}, WR {win_rate:.1f}%, "
                  f"PnL {total_pnl*100:+.2f}%, PF {profit_factor:.2f}, DD {max_dd*100:.1f}%")

        except Exception as e:
            print(f"  {year}: ERROR — {e}")
            month_results.append({'year': year, 'trades': 0, 'wins': 0, 'losses': 0,
                'wr': 0, 'pnl': 0, 'pf': 0, 'max_dd': 0, 'longs': 0, 'shorts': 0,
                'tp1': 0, 'tp2': 0, 'tp3': 0, 'sl': 0})

    all_results[month_name] = month_results

# Save raw results
with open('/tmp/v613_full_results.json', 'w') as f:
    json.dump(all_results, f, indent=2)

# Print summary table
print("\n\n" + "=" * 80)
print("  JIMI v6.13 — YEAR SUMMARY (2025-2026)")
print("=" * 80)
print(f"\n  {'Month':<12} {'Profitable':>10} {'Avg PnL%':>10} {'Avg WR%':>9} {'Avg PF':>8} {'Avg DD%':>9} {'Best':>12} {'Worst':>12}")
print(f"  {'─'*12} {'─'*10} {'─'*10} {'─'*9} {'─'*8} {'─'*9} {'─'*12} {'─'*12}")

month_order = ['january','february','march','april','may','june','july','august','september','october','november','december']
for m in month_order:
    results = all_results[m]
    valid = [r for r in results if r['trades'] > 0]
    if not valid:
        continue
    profitable = len([r for r in valid if r['pnl'] > 0])
    avg_pnl = np.mean([r['pnl'] for r in valid])
    avg_wr = np.mean([r['wr'] for r in valid])
    avg_pf = np.mean([r['pf'] for r in valid if r['pf'] < 100])
    avg_dd = np.mean([r['max_dd'] for r in valid])
    best = max(valid, key=lambda x: x['pnl'])
    worst = min(valid, key=lambda x: x['pnl'])
    print(f"  {m.capitalize():<12} {profitable}/{len(valid)} ({profitable/len(valid)*100:.0f}%)  "
          f"{avg_pnl:>+9.1f}% {avg_wr:>8.1f}% {avg_pf:>7.2f} {avg_dd:>8.1f}%  "
          f"{best['pnl']:>+10.1f}% ({best['year']}) {worst['pnl']:>+10.1f}% ({worst['year']})")

print(f"\n  Saved raw results to /tmp/v613_full_results.json")

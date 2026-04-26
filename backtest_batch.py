#!/usr/bin/env python3
"""Batch backtest for remaining months using jimi_v612_finetuned.py."""

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

from jimi_v612_finetuned import run_backtest

# (month_num, fetch_start_month, fetch_start_day, fetch_end_month, fetch_end_day, year_offset)
# Fetch ~15 days before month start for warmup, end at month end + buffer
MONTHS = {
    'january':   (1,  12, 15, 1, 31, 1),   # prev year Dec 15 -> Jan 31
    'february':  (2,  1,  15, 2, 28, 0),    # Jan 15 -> Feb 28
    'may':       (5,  4,  15, 5, 31, 0),    # Apr 15 -> May 31
    'june':      (6,  5,  15, 6, 30, 0),    # May 15 -> Jun 30
    'august':    (8,  7,  15, 8, 31, 0),    # Jul 15 -> Aug 31
    'october':   (10, 9,  15, 10, 31, 0),   # Sep 15 -> Oct 31
    'november':  (11, 10, 15, 11, 30, 0),   # Oct 15 -> Nov 30
}

month_name = sys.argv[1].lower() if len(sys.argv) > 1 else 'january'
mn, fsm, fsd, fem, fed, yoff = MONTHS[month_name]
month_str = f'{mn:02d}'

all_results = []

for year in [2020, 2021, 2022, 2023, 2024, 2025]:
    fetch_start_year = year - 1 if yoff else year
    last_day = calendar.monthrange(year, mn)[1]

    start = pd.Timestamp(f'{fetch_start_year}-{fsm:02d}-{fsd:02d}', tz='UTC')
    # For Feb, use actual last day; for others use configured
    end_day = min(fed, last_day)
    end = pd.Timestamp(f'{year}-{fem:02d}-{end_day:02d}', tz='UTC')
    since_ms = int(start.timestamp() * 1000)
    until_ms = int(end.timestamp() * 1000)

    date_start = f'{year}-{month_str}-01'
    date_end = f'{year}-{month_str}-{last_day}'

    print(f"\n{'='*60}")
    print(f"  {month_name.title()} {year}")
    print(f"{'='*60}")

    try:
        df = fetch_ohlcv('ETH/USDT', '15m', since_ms, until_ms)
    except Exception as e:
        print(f"  ERROR: {e}")
        all_results.append({'year': year, 'error': str(e)})
        continue

    if df.empty:
        all_results.append({'year': year, 'error': 'no data'})
        continue

    print(f"  {len(df)} bars ({df['Open time'].iloc[0].strftime('%Y-%m-%d')} → {df['Open time'].iloc[-1].strftime('%Y-%m-%d')})")

    csv_path = f'/tmp/eth_15m_{year}_{month_name}.csv'
    df.to_csv(csv_path, index=False)

    try:
        trades, stats, df_bt = run_backtest(csv_path, verbose=False, date_start=date_start, date_end=date_end)
        total = len(trades)
        if total == 0:
            all_results.append({'year': year, 'trades': 0, 'wins': 0, 'losses': 0,
                                'win_rate': 0, 'total_pnl': 0, 'avg_pnl': 0,
                                'pf': 0, 'max_dd': 0, 'longs': 0, 'shorts': 0,
                                'tp1': 0, 'tp2': 0, 'tp3': 0, 'sl': 0})
            print(f"  No trades")
            continue

        winners = [t for t in trades if t.pnl_pct > 0]
        losers = [t for t in trades if t.pnl_pct < 0]
        wins, losses = len(winners), len(losers)
        wr = wins / total * 100
        total_pnl = sum(t.pnl_pct * t.size_pct for t in trades)
        gross_profit = sum(t.pnl_pct * t.size_pct for t in winners)
        gross_loss = abs(sum(t.pnl_pct * t.size_pct for t in losers))
        pf = gross_profit / gross_loss if gross_loss > 0 else float('inf')

        equity = [0]
        for t in sorted(trades, key=lambda x: x.exit_time):
            equity.append(equity[-1] + t.pnl_pct * t.size_pct)
        equity = np.array(equity)
        peak = np.maximum.accumulate(equity)
        max_dd = abs((equity - peak).min()) if len(equity) > 0 else 0

        longs = [t for t in trades if t.direction == 'LONG']
        shorts = [t for t in trades if t.direction == 'SHORT']

        all_results.append({
            'year': year, 'trades': total, 'wins': wins, 'losses': losses,
            'win_rate': wr, 'total_pnl': total_pnl * 100,
            'pf': pf, 'max_dd': max_dd * 100,
            'longs': len(longs), 'shorts': len(shorts),
            'tp1': len([t for t in trades if t.tp1_hit]),
            'tp2': len([t for t in trades if t.tp2_hit]),
            'tp3': len([t for t in trades if t.exit_reason == 'TP3']),
            'sl': len([t for t in trades if t.exit_reason == 'SL']),
        })
        print(f"  ✓ {total} trades, WR {wr:.1f}%, PnL {total_pnl*100:+.2f}%, PF {pf:.2f}, DD {max_dd*100:.2f}%")
    except Exception as e:
        import traceback; traceback.print_exc()
        all_results.append({'year': year, 'error': str(e)})

# Summary
print(f"\n{'='*80}")
print(f"  JIMI v6.12 — {month_name.upper()} BACKTEST SUMMARY")
print(f"{'='*80}")
print(f"{'Year':>6} {'Trades':>7} {'W/L':>6} {'WinR':>7} {'PnL%':>10} {'PF':>7} {'MaxDD':>8} {'L/S':>6} {'TP1':>5} {'TP2':>5} {'TP3':>5} {'SL':>5}")
print('-'*80)
for r in all_results:
    if 'error' in r:
        print(f"{r['year']:>6}  ERROR: {r['error']}")
    elif r['trades'] == 0:
        print(f"{r['year']:>6}  No trades")
    else:
        pf_s = f"{r['pf']:.2f}" if r['pf'] != float('inf') else "inf"
        print(f"{r['year']:>6} {r['trades']:>7} {r['wins']:>3}/{r['losses']:<3} {r['win_rate']:>6.1f}% {r['total_pnl']:>+9.2f}% {pf_s:>7} {r['max_dd']:>7.2f}% {r['longs']:>3}/{r['shorts']:<3} {r['tp1']:>5} {r['tp2']:>5} {r['tp3']:>5} {r['sl']:>5}")

valid = [r for r in all_results if 'error' not in r and r['trades'] > 0]
if valid:
    total_trades = sum(r['trades'] for r in valid)
    total_wins = sum(r['wins'] for r in valid)
    total_losses = sum(r['losses'] for r in valid)
    avg_wr = (total_wins / total_trades * 100) if total_trades else 0
    avg_pnl = sum(r['total_pnl'] for r in valid) / len(valid)
    best = max(valid, key=lambda x: x['total_pnl'])
    worst = min(valid, key=lambda x: x['total_pnl'])
    profitable = len([r for r in valid if r['total_pnl'] > 0])
    print('-'*80)
    print(f"  Total: {total_trades} trades, WR {avg_wr:.1f}%, Avg PnL {avg_pnl:+.2f}%")
    print(f"  Best: {best['year']} ({best['total_pnl']:+.2f}%) | Worst: {worst['year']} ({worst['total_pnl']:+.2f}%)")
    print(f"  Profitable: {profitable}/{len(valid)}")

with open(f'/tmp/jimi_{month_name}_results.json', 'w') as f:
    json.dump(all_results, f, indent=2, default=str)

#!/usr/bin/env python3
"""Fetch September ETH/USDT 15m data for each year and run JIMI backtests."""

import ccxt
import pandas as pd
import numpy as np
import sys, os

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

results = []

for year in [2020, 2021, 2022, 2023, 2024, 2025]:
    start = pd.Timestamp(f'{year}-07-15', tz='UTC')
    end = pd.Timestamp(f'{year}-10-01', tz='UTC')
    since_ms = int(start.timestamp() * 1000)
    until_ms = int(end.timestamp() * 1000)

    print(f"\n{'='*60}")
    print(f"  Fetching data for {year} (Feb 15 - Apr 30)...")
    print(f"{'='*60}")

    try:
        df = fetch_ohlcv('ETH/USDT', '15m', since_ms, until_ms)
    except Exception as e:
        print(f"  ERROR fetching {year}: {e}")
        results.append({'year': year, 'error': str(e)})
        continue

    if df.empty:
        print(f"  No data for {year}")
        results.append({'year': year, 'error': 'no data'})
        continue

    print(f"  Fetched {len(df)} bars ({df['Open time'].iloc[0]} to {df['Open time'].iloc[-1]})")

    csv_path = f'/tmp/eth_15m_{year}_apr.csv'
    df.to_csv(csv_path, index=False)

    print(f"  Running backtest for September {year}...")
    try:
        trades, stats, df_bt = run_backtest(
            csv_path, verbose=False,
            date_start=f'{year}-09-01', date_end=f'{year}-09-30',
        )

        total = len(trades)
        if total == 0:
            results.append({'year': year, 'trades': 0, 'wins': 0, 'losses': 0,
                            'win_rate': 0, 'total_pnl': 0, 'avg_pnl': 0,
                            'max_win': 0, 'max_loss': 0, 'pf': 0, 'max_dd': 0,
                            'longs': 0, 'shorts': 0, 'tp1': 0, 'tp2': 0, 'tp3': 0, 'sl': 0})
            print(f"  ✓ No trades generated")
            continue

        winners = [t for t in trades if t.pnl_pct > 0]
        losers = [t for t in trades if t.pnl_pct < 0]
        wins = len(winners)
        losses = len(losers)
        wr = wins / total * 100
        total_pnl = sum(t.pnl_pct * t.size_pct for t in trades)
        avg_pnl = total_pnl / total * 100
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
        tp1 = len([t for t in trades if t.tp1_hit])
        tp2 = len([t for t in trades if t.tp2_hit])
        tp3 = len([t for t in trades if t.exit_reason == 'TP3'])
        sl = len([t for t in trades if t.exit_reason == 'SL'])

        results.append({
            'year': year, 'trades': total, 'wins': wins, 'losses': losses,
            'win_rate': wr, 'total_pnl': total_pnl * 100, 'avg_pnl': avg_pnl,
            'max_win': max([t.pnl_pct * t.size_pct for t in trades]) * 100,
            'max_loss': min([t.pnl_pct * t.size_pct for t in trades]) * 100,
            'pf': pf, 'max_dd': max_dd * 100,
            'longs': len(longs), 'shorts': len(shorts),
            'tp1': tp1, 'tp2': tp2, 'tp3': tp3, 'sl': sl,
        })
        print(f"  ✓ {total} trades, WR {wr:.1f}%, PnL {total_pnl*100:+.2f}%, PF {pf:.2f}, DD {max_dd*100:.2f}%")

    except Exception as e:
        import traceback; traceback.print_exc()
        results.append({'year': year, 'error': str(e)})

# ── Summary ──
print(f"\n\n{'='*80}")
print(f"  JIMI v6.10 — SEPTEMBER BACKTEST SUMMARY (ETH/USDT 15m)")
print(f"{'='*80}")
hdr = f"{'Year':>6} {'Trades':>7} {'W/L':>6} {'WinR':>7} {'PnL%':>10} {'AvgPnL':>9} {'PF':>7} {'MaxDD':>8} {'L/S':>6} {'TP1':>5} {'TP2':>5} {'TP3':>5} {'SL':>5}"
print(hdr)
print('-'*80)
for r in results:
    if 'error' in r:
        print(f"{r['year']:>6}  ERROR: {r['error']}")
    elif r['trades'] == 0:
        print(f"{r['year']:>6}  No trades")
    else:
        pf_str = f"{r['pf']:.2f}" if r['pf'] != float('inf') else "inf"
        print(f"{r['year']:>6} {r['trades']:>7} {r['wins']:>3}/{r['losses']:<3} {r['win_rate']:>6.1f}% {r['total_pnl']:>+9.2f}% {r['avg_pnl']:>+8.3f}% {pf_str:>7} {r['max_dd']:>7.2f}% {r['longs']:>3}/{r['shorts']:<3} {r['tp1']:>5} {r['tp2']:>5} {r['tp3']:>5} {r['sl']:>5}")

valid = [r for r in results if 'error' not in r and r['trades'] > 0]
if valid:
    total_trades = sum(r['trades'] for r in valid)
    total_wins = sum(r['wins'] for r in valid)
    total_losses = sum(r['losses'] for r in valid)
    avg_wr = (total_wins / total_trades * 100) if total_trades else 0
    avg_pnl = sum(r['total_pnl'] for r in valid) / len(valid)
    best = max(valid, key=lambda x: x['total_pnl'])
    worst = min(valid, key=lambda x: x['total_pnl'])
    print('-'*80)
    print(f"  Total trades across all Septembers: {total_trades}")
    print(f"  Aggregate win rate: {avg_wr:.1f}% ({total_wins}W / {total_losses}L)")
    print(f"  Avg September PnL: {avg_pnl:+.2f}%")
    print(f"  Best September: {best['year']} ({best['total_pnl']:+.2f}%)")
    print(f"  Worst September: {worst['year']} ({worst['total_pnl']:+.2f}%)")
    profitable = len([r for r in valid if r['total_pnl'] > 0])
    print(f"  Profitable Septembers: {profitable}/{len(valid)}")

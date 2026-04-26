#!/usr/bin/env python3
"""Fetch ETH/USDT 15m candles from Binance for 2021-2022."""
import ccxt
import pandas as pd
import time

exchange = ccxt.binance({'enableRateLimit': True})

start = pd.Timestamp('2021-01-01', tz='UTC')
end = pd.Timestamp('2023-01-01', tz='UTC')  # exclusive

since_ms = int(start.timestamp() * 1000)
until_ms = int(end.timestamp() * 1000)

all_candles = []
current = since_ms
batch = 0

print(f"Fetching ETH/USDT 15m from {start.date()} to {end.date()}...")
print(f"Total expected: ~70,000 candles")

while current < until_ms:
    batch += 1
    try:
        raw = exchange.fetch_ohlcv('ETH/USDT', '15m', since=current, limit=1000)
    except Exception as e:
        print(f"  Retry after error: {e}")
        time.sleep(10)
        raw = exchange.fetch_ohlcv('ETH/USDT', '15m', since=current, limit=1000)
    
    if not raw:
        break
    
    for c in raw:
        ts = int(c[0])
        if ts >= until_ms:
            break
        all_candles.append({
            'Open time': pd.to_datetime(ts, unit='ms'),
            'Open': float(c[1]),
            'High': float(c[2]),
            'Low': float(c[3]),
            'Close': float(c[4]),
            'Volume': float(c[5]),
        })
    
    last_ts = raw[-1][0]
    if last_ts <= current:
        break
    current = last_ts + 1
    
    if batch % 50 == 0:
        pct = (current - since_ms) / (until_ms - since_ms) * 100
        print(f"  Batch {batch}: {len(all_candles)} candles ({pct:.1f}%)")

df = pd.DataFrame(all_candles)
df['Open time'] = df['Open time'].dt.strftime('%Y-%m-%d %H:%M:%S')

# Add missing columns the script expects
df['Close time'] = pd.to_datetime(df['Open time']) + pd.Timedelta(minutes=14, seconds=59)
df['Close time'] = df['Close time'].dt.strftime('%Y-%m-%d %H:%M:%S')
df['Quote asset volume'] = df['Volume'] * df['Close']  # approximate
df['Number of trades'] = 0
df['Taker buy base asset volume'] = df['Volume'] * 0.5  # approximate
df['Taker buy quote asset volume'] = df['Taker buy base asset volume'] * df['Close']

outpath = 'eth_15m_2021_2022.csv'
df.to_csv(outpath, index=False)
print(f"\nSaved {len(df)} candles to {outpath}")
print(f"Date range: {df['Open time'].iloc[0]} to {df['Open time'].iloc[-1]}")

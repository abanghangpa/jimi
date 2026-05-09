#!/usr/bin/env python3
"""
Build intrabar delta CSV from 1-minute Binance data.

Fetches 1m bars, computes LucF-style intrabar delta, aggregates to 15m,
and saves to data/eth_15m_intrabar_delta.csv.

Supports incremental updates: on subsequent runs, only fetches missing bars.

Usage:
    python3 scripts/build_intrabar_delta.py              # full build / incremental
    python3 scripts/build_intrabar_delta.py --hours 48    # last 48h only
    python3 scripts/build_intrabar_delta.py --batch-delay 300  # 300ms between API calls
"""

import argparse
import os
import sys
import time
import numpy as np
import pandas as pd
import ccxt

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

# Paths
DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data')
INTRABAR_CSV = os.path.join(DATA_DIR, 'eth_15m_intrabar_delta.csv')
MERGED_CSV = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'eth_15m_merged.csv')

# Binance API settings
SYMBOL = 'ETHUSDT'
INTERVAL_1M = '1m'
MAX_PER_REQUEST = 1000
DEFAULT_BATCH_DELAY = 0.25  # seconds between API calls (conservative)


def get_exchange():
    return ccxt.binance({"enableRateLimit": True})


def fetch_1m_chunk(ex, since_ms, end_ms=None, limit=1000):
    """Fetch a chunk of 1m bars from Binance."""
    params = {'symbol': SYMBOL, 'interval': INTERVAL_1M, 'limit': limit}
    if since_ms is not None:
        params['startTime'] = since_ms
    if end_ms is not None:
        params['endTime'] = end_ms

    raw = ex.publicGetKlines(params)
    if not raw:
        return []

    rows = []
    for c in raw:
        rows.append({
            'timestamp_ms': int(c[0]),
            'Open': float(c[1]),
            'High': float(c[2]),
            'Low': float(c[3]),
            'Close': float(c[4]),
            'Volume': float(c[5]),
        })
    return rows


def compute_delta_from_1m(rows):
    """Compute LucF-style intrabar delta from 1m OHLCV rows.

    Returns list of (timestamp_ms, delta) tuples.
    """
    if not rows:
        return []

    deltas = []
    last_polarity = 0

    for i, r in enumerate(rows):
        vol = r['Volume']
        if vol == 0:
            deltas.append((r['timestamp_ms'], 0.0))
            continue

        c, o = r['Close'], r['Open']
        if c > o:
            deltas.append((r['timestamp_ms'], vol))
            last_polarity = 1
        elif c < o:
            deltas.append((r['timestamp_ms'], -vol))
            last_polarity = -1
        else:
            # close == open: check previous bar
            if i > 0:
                prev_c = rows[i - 1]['Close']
                if c > prev_c:
                    deltas.append((r['timestamp_ms'], vol))
                    last_polarity = 1
                elif c < prev_c:
                    deltas.append((r['timestamp_ms'], -vol))
                    last_polarity = -1
                else:
                    deltas.append((r['timestamp_ms'], vol * last_polarity))
            else:
                deltas.append((r['timestamp_ms'], 0.0))

    return deltas


def aggregate_to_15m(deltas_1m):
    """Aggregate 1m deltas into 15m buckets.

    Returns list of (timestamp_15m_ms, delta_15m) tuples.
    """
    if not deltas_1m:
        return []

    # Bucket by 15m boundary (floor to 15-min interval)
    bucket_ms = 15 * 60 * 1000
    buckets = {}
    for ts_ms, delta in deltas_1m:
        key = (ts_ms // bucket_ms) * bucket_ms
        buckets[key] = buckets.get(key, 0.0) + delta

    return sorted(buckets.items())


def load_existing():
    """Load existing intrabar delta CSV. Returns set of timestamps and last_ts_ms."""
    if not os.path.exists(INTRABAR_CSV):
        return set(), None

    df = pd.read_csv(INTRABAR_CSV)
    if df.empty:
        return set(), None

    timestamps = set(df['timestamp_ms'].values)
    last_ts = max(timestamps)
    return timestamps, last_ts


def save_results(results, mode='w'):
    """Save (timestamp_ms, delta_15m) pairs to CSV."""
    df = pd.DataFrame(results, columns=['timestamp_ms', 'delta_15m'])
    # Add human-readable timestamp
    df['timestamp'] = pd.to_datetime(df['timestamp_ms'], unit='ms').dt.strftime('%Y-%m-%d %H:%M:%S')
    df = df[['timestamp', 'timestamp_ms', 'delta_15m']]
    df.to_csv(INTRABAR_CSV, mode=mode, header=(mode == 'w'), index=False)


def get_reference_start_ms():
    """Get the start timestamp from the merged CSV (earliest 15m bar)."""
    if not os.path.exists(MERGED_CSV):
        return None

    # Read first data line (skip header)
    with open(MERGED_CSV, 'r') as f:
        f.readline()  # skip header
        first_line = f.readline().strip()

    if not first_line:
        return None

    ts_str = first_line.split(',')[0].strip('"')
    try:
        ts = pd.Timestamp(ts_str)
        return int(ts.timestamp() * 1000)
    except Exception:
        return None


def main():
    parser = argparse.ArgumentParser(description='Build intrabar delta CSV')
    parser.add_argument('--hours', type=int, default=None,
                        help='Only fetch last N hours (default: full build)')
    parser.add_argument('--batch-delay', type=float, default=DEFAULT_BATCH_DELAY,
                        help='Seconds between API calls (default: 0.25)')
    parser.add_argument('--resume', action='store_true',
                        help='Resume from last fetched timestamp')
    args = parser.parse_args()

    os.makedirs(DATA_DIR, exist_ok=True)
    ex = get_exchange()

    # Load existing data
    existing_ts, last_existing_ts = load_existing()

    if args.hours:
        # Recent-only mode
        now_ms = int(time.time() * 1000)
        start_ms = now_ms - args.hours * 3600 * 1000
        print(f"Mode: Last {args.hours}h only")
    elif last_existing_ts and args.resume:
        # Resume from last fetched
        start_ms = last_existing_ts + 1
        print(f"Mode: Resume from {pd.to_datetime(last_existing_ts, unit='ms')}")
    elif last_existing_ts:
        # Incremental: fetch from last existing to now
        start_ms = last_existing_ts + 1
        print(f"Mode: Incremental update from {pd.to_datetime(last_existing_ts, unit='ms')}")
    else:
        # Full build: start from merged CSV beginning
        start_ms = get_reference_start_ms()
        if start_ms is None:
            start_ms = int(pd.Timestamp('2020-01-01').timestamp() * 1000)
        print(f"Mode: Full build from {pd.to_datetime(start_ms, unit='ms')}")

    now_ms = int(time.time() * 1000)
    if start_ms >= now_ms:
        print("Already up to date!")
        return

    total_1m_bars = int((now_ms - start_ms) / 60000)
    total_requests = (total_1m_bars // MAX_PER_REQUEST) + 1
    est_minutes = total_requests * args.batch_delay / 60

    print(f"  1m bars to fetch: ~{total_1m_bars:,}")
    print(f"  API requests: ~{total_requests:,}")
    print(f"  Estimated time: ~{est_minutes:.1f} min (batch_delay={args.batch_delay}s)")
    print()

    # Fetch and process in batches
    all_15m_results = []
    cursor_ms = start_ms
    request_count = 0
    error_count = 0
    total_fetched = 0

    # Accumulate 1m deltas across chunks for correct 15m aggregation
    pending_1m_deltas = []
    last_flush_15m_boundary = None

    while cursor_ms < now_ms:
        try:
            rows = fetch_1m_chunk(ex, cursor_ms, limit=MAX_PER_REQUEST)
            if not rows:
                break

            # Compute deltas for this chunk
            chunk_deltas = compute_delta_from_1m(rows)
            pending_1m_deltas.extend(chunk_deltas)
            total_fetched += len(rows)

            # Move cursor past last fetched bar
            last_ts = rows[-1]['timestamp_ms']
            cursor_ms = last_ts + 60000  # next 1m bar

            # Flush completed 15m boundaries
            # A 15m boundary is "complete" when we have data past its end
            bucket_ms = 15 * 60 * 1000
            current_boundary = (cursor_ms // bucket_ms) * bucket_ms

            if last_flush_15m_boundary is not None and current_boundary > last_flush_15m_boundary:
                # Aggregate all deltas up to (not including) current boundary
                flush_cutoff = current_boundary
                to_aggregate = [(ts, d) for ts, d in pending_1m_deltas if ts < flush_cutoff]
                remaining = [(ts, d) for ts, d in pending_1m_deltas if ts >= flush_cutoff]

                if to_aggregate:
                    aggregated = aggregate_to_15m(to_aggregate)
                    # Filter out already-existing timestamps
                    new_results = [(ts, d) for ts, d in aggregated if ts not in existing_ts]
                    all_15m_results.extend(new_results)
                    for ts, _ in new_results:
                        existing_ts.add(ts)

                pending_1m_deltas = remaining
                last_flush_15m_boundary = current_boundary

            if last_flush_15m_boundary is None:
                last_flush_15m_boundary = (cursor_ms // bucket_ms) * bucket_ms

            request_count += 1
            if request_count % 100 == 0:
                pct = min(100, (cursor_ms - start_ms) / (now_ms - start_ms) * 100)
                print(f"  [{pct:5.1f}%] {total_fetched:,} 1m bars → {len(all_15m_results):,} new 15m deltas "
                      f"(requests: {request_count}, errors: {error_count})")

            # Rate limit
            time.sleep(args.batch_delay)

        except Exception as e:
            error_count += 1
            if error_count > 10:
                print(f"  ❌ Too many errors ({error_count}), stopping")
                break
            wait = min(5.0, 1.0 * error_count)
            print(f"  ⚠️  Error: {e} — waiting {wait:.1f}s")
            time.sleep(wait)

    # Flush remaining pending deltas
    if pending_1m_deltas:
        aggregated = aggregate_to_15m(pending_1m_deltas)
        new_results = [(ts, d) for ts, d in aggregated if ts not in existing_ts]
        all_15m_results.extend(new_results)

    # Save results
    if all_15m_results:
        all_15m_results.sort(key=lambda x: x[0])
        file_exists = os.path.exists(INTRABAR_CSV) and os.path.getsize(INTRABAR_CSV) > 0
        save_results(all_15m_results, mode='a' if file_exists else 'w')
        print(f"\n  ✅ Saved {len(all_15m_results):,} new 15m delta rows → {INTRABAR_CSV}")
    else:
        print("\n  ℹ️  No new data to save")

    # Summary
    if os.path.exists(INTRABAR_CSV):
        total_rows = sum(1 for _ in open(INTRABAR_CSV)) - 1  # minus header
        print(f"  📊 Total rows in file: {total_rows:,}")
        print(f"  📊 API requests made: {request_count:,}")
        print(f"  📊 Errors: {error_count}")


if __name__ == '__main__':
    main()

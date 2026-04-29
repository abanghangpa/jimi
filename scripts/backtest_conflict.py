#!/usr/bin/env python3
"""
Backtest: M1 Bullish + Daily Bearish Conflict Scenarios
Uses raw OHLCV data to compute signals, then tracks forward price action.

Usage:
    python3 scripts/backtest_conflict.py
    python3 scripts/backtest_conflict.py --days 10
"""

import sys, os, argparse
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from src.config import CONFIG
from src.utils.indicators import calc_ema, calc_macd, calc_rsi, calc_atr, calc_swing_bias, calc_phase0, calc_trend_state
from src.utils.data_handler import resample_ohlcv, load_data

# ── Config ──────────────────────────────────────────────────────
SIXM = os.path.join(os.path.dirname(os.path.dirname(__file__)), "eth_15m_6m.csv")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--days', type=int, default=10)
    args = parser.parse_args()

    print("Loading OHLCV data...")
    df_15m = load_data(SIXM)
    print(f"  15m bars: {len(df_15m)}, {df_15m['Open time'].iloc[0]} → {df_15m['Open time'].iloc[-1]}")

    # Resample to higher timeframes
    print("Resampling timeframes...")
    df_1h = resample_ohlcv(df_15m, '1H')
    df_1d = resample_ohlcv(df_15m, '1D')
    print(f"  1H: {len(df_1h)} bars, 1D: {len(df_1d)} bars")

    # Compute indicators
    print("Computing indicators...")
    # 1H MACD (M1 direction)
    df_1h['macd_line'], df_1h['macd_signal'], df_1h['macd_hist'] = calc_macd(
        df_1h['Close'], CONFIG['MACD_FAST'], CONFIG['MACD_SLOW'], CONFIG['MACD_SIGNAL'])
    df_1h['ema_fast'] = calc_ema(df_1h['Close'], CONFIG['EMA_FAST'])
    df_1h['ema_slow'] = calc_ema(df_1h['Close'], CONFIG['EMA_slow'] if 'EMA_slow' in CONFIG else CONFIG['EMA_SLOW'])
    df_1h['rsi'] = calc_rsi(df_1h['Close'], 14)
    df_1h['atr'] = calc_atr(df_1h['High'], df_1h['Low'], df_1h['Close'], CONFIG['ATR_PERIOD'])

    # Daily swing bias + trend
    df_1d['swing_bias'] = calc_swing_bias(df_1d)
    df_1d['phase0'] = calc_phase0(df_1d)
    df_1d['trend'], df_1d['trend_score'] = calc_trend_state(df_1d)

    # ── Scan for conflict signals ───────────────────────────────
    print("\n" + "═" * 70)
    print(f"  BACKTEST: M1 BULLISH vs DAILY BEARISH CONFLICT — LAST {args.days} DAYS")
    print("═" * 70)

    # Determine cutoff
    last_time = pd.to_datetime(df_15m['Open time'].iloc[-1])
    cutoff = last_time - pd.Timedelta(days=args.days)
    print(f"\n  Window: {cutoff} → {last_time}")

    # For each 15m bar in the window, check if M1=BULLISH + Daily=BEARISH
    signals = []
    scan_interval = 4  # every hour

    df_1h_idx = df_1h.set_index('Open time')
    df_1d_idx = df_1d.set_index('Open time')

    for idx in range(0, len(df_15m), scan_interval):
        row = df_15m.iloc[idx]
        ts = pd.to_datetime(row['Open time'])
        if ts < cutoff:
            continue

        price = float(row['Close'])

        # Find corresponding 1H bar
        try:
            # Get the 1H bar that contains this 15m timestamp
            h1_ts = df_1h_idx.index[df_1h_idx.index <= ts]
            if len(h1_ts) == 0:
                continue
            h1_ts = h1_ts[-1]
            h1_row = df_1h_idx.loc[h1_ts]
        except:
            continue

        # Find corresponding daily bar
        try:
            d_ts = df_1d_idx.index[df_1d_idx.index <= ts]
            if len(d_ts) == 0:
                continue
            d_ts = d_ts[-1]
            d_row = df_1d_idx.loc[d_ts]
        except:
            continue

        # M1 direction: MACD histogram positive + MACD line > signal
        macd_hist = float(h1_row.get('macd_hist', 0))
        macd_line = float(h1_row.get('macd_line', 0))
        macd_signal = float(h1_row.get('macd_signal', 0))
        ema_fast = float(h1_row.get('ema_fast', 0))
        ema_slow = float(h1_row.get('ema_slow', 0))

        # M1 = BULLISH if MACD hist > 0 and MACD line > signal and price > EMA fast
        m1_bullish = (macd_hist > 0) and (macd_line > macd_signal) and (price > ema_fast)

        # Daily bias
        swing = str(d_row.get('swing_bias', ''))
        daily_bearish = swing == 'BEARISH'

        if m1_bullish and daily_bearish:
            signals.append({
                'timestamp': str(ts),
                'price': price,
                'macd_hist': round(macd_hist, 4),
                'macd_line': round(macd_line, 4),
                'rsi_1h': round(float(h1_row.get('rsi', 0)), 2),
                'ema_fast': round(ema_fast, 2),
                'ema_slow': round(ema_slow, 2),
                'swing_bias': swing,
                'phase0': float(d_row.get('phase0', 0)),
                'trend': str(d_row.get('trend', '')),
                'idx_15m': idx,
            })

    # Deduplicate (same timestamp)
    seen = set()
    unique_signals = []
    for s in signals:
        if s['timestamp'] not in seen:
            seen.add(s['timestamp'])
            unique_signals.append(s)
    signals = unique_signals

    print(f"  ⚡ Found {len(signals)} conflict signals\n")

    if not signals:
        print("  No conflict signals found in this window.")
        # Try broader: just M1 bullish
        print("\n  Checking M1 BULLISH signals (without daily bearish filter)...")
        m1_only = []
        for idx in range(0, len(df_15m), scan_interval):
            row = df_15m.iloc[idx]
            ts = pd.to_datetime(row['Open time'])
            if ts < cutoff:
                continue
            price = float(row['Close'])
            try:
                h1_ts = df_1h_idx.index[df_1h_idx.index <= ts]
                if len(h1_ts) == 0: continue
                h1_row = df_1h_idx.loc[h1_ts[-1]]
                macd_hist = float(h1_row.get('macd_hist', 0))
                macd_line = float(h1_row.get('macd_line', 0))
                macd_signal = float(h1_row.get('macd_signal', 0))
                ema_fast = float(h1_row.get('ema_fast', 0))
                if (macd_hist > 0) and (macd_line > macd_signal) and (price > ema_fast):
                    d_ts = df_1d_idx.index[df_1d_idx.index <= ts]
                    if len(d_ts) == 0: continue
                    d_row = df_1d_idx.loc[d_ts[-1]]
                    m1_only.append({
                        'timestamp': str(ts), 'price': price,
                        'swing_bias': str(d_row.get('swing_bias', '')),
                    })
            except:
                continue
        seen2 = set()
        m1_unique = []
        for s in m1_only:
            if s['timestamp'] not in seen2:
                seen2.add(s['timestamp'])
                m1_unique.append(s)
        if m1_unique:
            print(f"  Found {len(m1_unique)} M1 BULLISH signals")
            bull_bear = sum(1 for s in m1_unique if s['swing_bias'] == 'BEARISH')
            bull_bull = sum(1 for s in m1_unique if s['swing_bias'] == 'BULLISH')
            bull_neut = sum(1 for s in m1_unique if s['swing_bias'] not in ('BEARISH', 'BULLISH'))
            print(f"    Daily BEARISH: {bull_bear}")
            print(f"    Daily BULLISH: {bull_bull}")
            print(f"    Daily NEUTRAL: {bull_neut}")
        return

    # ── Forward price analysis ──────────────────────────────────
    forward_windows = {
        '1h': 4, '4h': 16, '8h': 32, '12h': 48,
        '24h': 96, '48h': 192, '72h': 288,
    }

    results = []
    for sig in signals:
        idx = sig['idx_15m']
        entry = sig['price']
        fwd = {k: v for k, v in sig.items() if k != 'idx_15m'}

        for wname, wbars in forward_windows.items():
            end = min(idx + wbars, len(df_15m) - 1)
            if end <= idx:
                continue
            future = df_15m.iloc[idx+1:end+1]
            highs = future['High'].astype(float)
            lows = future['Low'].astype(float)
            closes = future['Close'].astype(float)

            max_up = (highs.max() - entry) / entry * 100
            max_down = (entry - lows.min()) / entry * 100
            net = (closes.iloc[-1] - entry) / entry * 100
            last_close = closes.iloc[-1]

            fwd[f'{wname}_up'] = round(max_up, 2)
            fwd[f'{wname}_down'] = round(max_down, 2)
            fwd[f'{wname}_net'] = round(net, 2)
            fwd[f'{wname}_close'] = round(last_close, 2)
            fwd[f'{wname}_reversed'] = max_down > max_up  # more downside than upside = reversal

        results.append(fwd)

    # ── Print each signal ───────────────────────────────────────
    print(f"{'─' * 70}")
    print(f"  SIGNAL DETAILS ({len(results)} signals)")
    print(f"{'─' * 70}")

    for i, r in enumerate(results):
        print(f"\n  #{i+1}  {r['timestamp']}  ${r['price']:.2f}")
        print(f"      MACD hist={r['macd_hist']}  RSI(1H)={r['rsi_1h']}  EMA={r['ema_fast']}/{r['ema_slow']}")
        print(f"      Daily: {r['swing_bias']}  Phase0={r.get('phase0', 'N/A')}  Trend={r.get('trend', 'N/A')}")
        for wname in forward_windows:
            up_key = f'{wname}_up'
            if up_key not in r:
                continue
            rev = "⬇️ REV" if r[f'{wname}_reversed'] else "⬆️ UP"
            print(f"      {wname:>4}: ↑{r[f'{wname}_up']:+.2f}% ↓{r[f'{wname}_down']:+.2f}% "
                  f"net={r[f'{wname}_net']:+.2f}%  {rev}")

    # ── Aggregate stats ─────────────────────────────────────────
    print(f"\n{'═' * 70}")
    print(f"  AGGREGATE STATISTICS (n={len(results)})")
    print(f"{'═' * 70}")

    for wname in forward_windows:
        up_col = f'{wname}_up'
        if up_col not in results[0]:
            continue

        ups = [r[f'{wname}_up'] for r in results]
        downs = [r[f'{wname}_down'] for r in results]
        nets = [r[f'{wname}_net'] for r in results]
        revs = [r[f'{wname}_reversed'] for r in results]

        n = len(ups)
        rev_count = sum(revs)
        rev_rate = rev_count / n * 100

        # Win/loss: net positive = win for longs
        wins = sum(1 for n_ in nets if n_ > 0)
        win_rate = wins / n * 100

        avg_up = np.mean(ups)
        avg_down = np.mean(downs)
        avg_net = np.mean(nets)
        med_net = np.median(nets)

        max_gain = max(ups)
        max_loss = max(downs)

        print(f"\n  [{wname}]  n={n}")
        print(f"    Avg max upside:   +{avg_up:.2f}%")
        print(f"    Avg max downside: -{avg_down:.2f}%")
        print(f"    Avg net move:     {avg_net:+.2f}%")
        print(f"    Median net move:  {med_net:+.2f}%")
        print(f"    Win rate (net>0): {win_rate:.0f}% ({wins}/{n})")
        print(f"    Reversal rate:    {rev_rate:.0f}% ({rev_count}/{n})")
        print(f"    Best pump:        +{max_gain:.2f}%")
        print(f"    Worst dump:       -{max_loss:.2f}%")

    # ── Final verdict ───────────────────────────────────────────
    print(f"\n{'═' * 70}")
    print(f"  VERDICT")
    print(f"{'═' * 70}")

    for wname in ['4h', '24h', '48h']:
        up_col = f'{wname}_up'
        if up_col not in results[0]:
            continue
        nets = [r[f'{wname}_net'] for r in results]
        revs = [r[f'{wname}_reversed'] for r in results]
        n = len(nets)
        avg = np.mean(nets)
        rev_rate = sum(revs) / n * 100

        if rev_rate > 65:
            emoji = "🔴"
            label = "STRONG REVERSAL — conflict tends to resolve DOWN"
        elif rev_rate > 50:
            emoji = "🟡"
            label = "MILD REVERSAL — slight downside bias"
        elif rev_rate > 35:
            emoji = "⚪"
            label = "MIXED — no clear edge"
        else:
            emoji = "🟢"
            label = "BULLISH CONTINUATION — M1 overrides daily bias"

        print(f"  {wname}: {emoji} {label}  (rev={rev_rate:.0f}%, avg net={avg:+.2f}%)")

    print(f"{'═' * 70}")


if __name__ == '__main__':
    main()

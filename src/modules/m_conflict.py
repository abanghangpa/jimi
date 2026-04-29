"""
JIMI — Conflict History Module

When M1 direction conflicts with daily swing bias, this module
looks up historical instances of the same scenario and returns
forward price statistics (reversal rates, avg moves, etc.).

Uses the 6-month OHLCV CSV for historical lookback.
"""

import os
import numpy as np
import pandas as pd
from src.config import CONFIG
from src.utils.data_handler import load_data, resample_ohlcv
from src.utils.indicators import calc_ema, calc_macd, calc_swing_bias

# ── Cache ───────────────────────────────────────────────────────
_conflict_cache = None
_cache_mtime = None

SIXM_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))),
                         "eth_15m_6m.csv")

FORWARD_WINDOWS = {
    '4h': 16, '12h': 48, '24h': 96, '48h': 192, '72h': 288,
}


def _build_conflict_index():
    """Build conflict signal index from 6-month OHLCV data.

    Returns a dict keyed by (m1_direction, daily_bias) with list of
    {timestamp, price, idx} for each matching conflict signal.
    """
    global _conflict_cache, _cache_mtime

    if not os.path.exists(SIXM_PATH):
        return None

    mtime = os.path.getmtime(SIXM_PATH)
    if _conflict_cache is not None and _cache_mtime == mtime:
        return _conflict_cache

    df_15m = load_data(SIXM_PATH)
    df_1h = resample_ohlcv(df_15m, '1H')
    df_1d = resample_ohlcv(df_15m, '1D')

    # Compute indicators
    df_1h['macd_line'], df_1h['macd_signal'], df_1h['macd_hist'] = calc_macd(
        df_1h['Close'], CONFIG['MACD_FAST'], CONFIG['MACD_SLOW'], CONFIG['MACD_SIGNAL'])
    df_1h['ema_fast'] = calc_ema(df_1h['Close'], CONFIG['EMA_FAST'])
    df_1d['swing_bias'] = calc_swing_bias(df_1d)

    df_1h_idx = df_1h.set_index('Open time')
    df_1d_idx = df_1d.set_index('Open time')

    # Scan every 4 bars (hourly)
    signals = {}  # key: (m1_dir, daily_bias) -> list of {ts, price, idx}
    scan_interval = 4

    for idx in range(0, len(df_15m), scan_interval):
        row = df_15m.iloc[idx]
        ts = pd.to_datetime(row['Open time'])
        price = float(row['Close'])

        try:
            h1_times = df_1h_idx.index[df_1h_idx.index <= ts]
            if len(h1_times) == 0:
                continue
            h1_row = df_1h_idx.loc[h1_times[-1]]

            d_times = df_1d_idx.index[df_1d_idx.index <= ts]
            if len(d_times) == 0:
                continue
            d_row = df_1d_idx.loc[d_times[-1]]
        except Exception:
            continue

        macd_hist = float(h1_row.get('macd_hist', 0))
        macd_line = float(h1_row.get('macd_line', 0))
        macd_signal_val = float(h1_row.get('macd_signal', 0))
        ema_fast = float(h1_row.get('ema_fast', 0))

        # M1 direction
        if macd_hist > 0 and macd_line > macd_signal_val and price > ema_fast:
            m1_dir = 'BULLISH'
        elif macd_hist < 0 and macd_line < macd_signal_val and price < ema_fast:
            m1_dir = 'BEARISH'
        else:
            continue  # skip neutral — no conflict to analyze

        swing = str(d_row.get('swing_bias', ''))
        if swing not in ('BULLISH', 'BEARISH'):
            continue

        key = (m1_dir, swing)
        if key not in signals:
            signals[key] = []
        signals[key].append({'ts': str(ts), 'price': price, 'idx': idx})

    # Deduplicate consecutive same-direction signals (keep first in each 4h window)
    for key in signals:
        deduped = []
        last_ts = None
        for s in signals[key]:
            ts = pd.Timestamp(s['ts'])
            if last_ts is None or (ts - last_ts) >= pd.Timedelta(hours=4):
                deduped.append(s)
                last_ts = ts
        signals[key] = deduped

    # Pre-compute forward stats for each key
    result = {}
    for key, sigs in signals.items():
        if len(sigs) < 3:
            continue  # need at least 3 for meaningful stats

        stats = {}
        for wname, wbars in FORWARD_WINDOWS.items():
            ups, downs, nets, revs = [], [], [], []
            for s in sigs:
                sidx = s['idx']
                entry = s['price']
                end = min(sidx + wbars, len(df_15m) - 1)
                if end <= sidx:
                    continue
                future = df_15m.iloc[sidx+1:end+1]
                h = future['High'].astype(float)
                l = future['Low'].astype(float)
                c = future['Close'].astype(float)
                if len(h) == 0:
                    continue
                u = (h.max() - entry) / entry * 100
                d = (entry - l.min()) / entry * 100
                n = (c.iloc[-1] - entry) / entry * 100
                ups.append(u)
                downs.append(d)
                nets.append(n)
                revs.append(d > u)

            if not ups:
                continue

            n_count = len(ups)
            stats[wname] = {
                'n': n_count,
                'avg_up': round(float(np.mean(ups)), 2),
                'avg_down': round(float(np.mean(downs)), 2),
                'avg_net': round(float(np.mean(nets)), 2),
                'median_net': round(float(np.median(nets)), 2),
                'win_rate': round(sum(1 for x in nets if x > 0) / n_count * 100, 1),
                'reversal_rate': round(sum(revs) / n_count * 100, 1),
                'max_gain': round(float(max(ups)), 2),
                'max_loss': round(float(max(downs)), 2),
            }

        result[key] = {
            'total_signals': len(sigs),
            'first_seen': sigs[0]['ts'],
            'last_seen': sigs[-1]['ts'],
            'windows': stats,
        }

    _conflict_cache = result
    _cache_mtime = mtime
    return result


def get_conflict_stats(m1_direction, swing_bias):
    """Get historical conflict stats for the given M1 direction + daily bias.

    Returns dict with conflict analysis, or None if no data.
    """
    index = _build_conflict_index()
    if index is None:
        return None

    key = (m1_direction, swing_bias)
    entry = index.get(key)

    if entry is None:
        # Check if the key exists with reversed order (shouldn't happen but just in case)
        return None

    # Check if current scenario is a conflict or alignment
    is_conflict = (m1_direction != swing_bias)

    return {
        'is_conflict': is_conflict,
        'm1_direction': m1_direction,
        'daily_bias': swing_bias,
        'historical': entry,
    }

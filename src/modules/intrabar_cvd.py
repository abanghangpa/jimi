"""Intrabar CVD — LucF-style volume delta using sub-bar analysis.

Instead of using Binance's taker buy volume (aggregated per bar), this module
fetches 1-minute bars and classifies each as "up volume" or "down volume"
based on intrabar price direction — matching how TradingView's LucF indicators
(Delta Volume Columns, Delta Volume Candles) calculate delta.

Logic (matches LucF):
  - If intrabar close > open → up volume
  - If intrabar close < open → down volume
  - If close == open → compare to previous intrabar close
  - Last resort: use last known polarity

Delta = up_volume - down_volume (per aggregated bar)
CVD = rolling sum of delta
Divergence = price vs CVD disagreement on swing highs/lows
"""

import numpy as np
import pandas as pd
import ccxt
import time


_binance_exchange = None


def _get_binance():
    global _binance_exchange
    if _binance_exchange is None:
        _binance_exchange = ccxt.binance({"enableRateLimit": True})
    return _binance_exchange


def fetch_1m_bars(symbol='ETHUSDT', hours=12):
    """Fetch 1m bars from Binance for intrabar analysis.

    Args:
        symbol: Binance symbol (e.g. 'ETHUSDT')
        hours: How many hours of 1m data to fetch

    Returns:
        DataFrame with 1m OHLCV data
    """
    ex = _get_binance()
    bars_needed = hours * 60
    max_per_request = 1000

    all_rows = []
    remaining = bars_needed
    end_time = None

    while remaining > 0:
        limit = min(remaining, max_per_request)
        params = {'symbol': symbol, 'interval': '1m', 'limit': limit}
        if end_time is not None:
            params['endTime'] = end_time

        raw = ex.publicGetKlines(params)
        if not raw:
            break

        for c in raw:
            all_rows.append({
                'Open time': pd.to_datetime(int(c[0]), unit='ms'),
                'Open': float(c[1]), 'High': float(c[2]),
                'Low': float(c[3]), 'Close': float(c[4]),
                'Volume': float(c[5]),
            })

        end_time = int(raw[0][0]) - 1
        remaining -= len(raw)
        if len(raw) < limit:
            break

    if not all_rows:
        return pd.DataFrame()

    df = pd.DataFrame(all_rows).sort_values('Open time').reset_index(drop=True)
    return df


def compute_intrabar_delta(df_1m):
    """Compute delta volume using intrabar analysis (LucF method).

    For each 1m bar:
      - close > open → volume is "up"
      - close < open → volume is "down"
      - close == open → compare to previous bar's close
      - still equal → use last known polarity

    Returns Series of delta (up_vol - down_vol) per 1m bar.
    """
    close = df_1m['Close'].values
    open_ = df_1m['Open'].values
    vol = df_1m['Volume'].values
    n = len(df_1m)

    delta = np.zeros(n)
    last_polarity = 0  # 0 = unknown, 1 = up, -1 = down

    for i in range(n):
        if vol[i] == 0:
            delta[i] = 0
            continue

        if close[i] > open_[i]:
            delta[i] = vol[i]
            last_polarity = 1
        elif close[i] < open_[i]:
            delta[i] = -vol[i]
            last_polarity = -1
        else:
            # close == open, check previous bar
            if i > 0:
                if close[i] > close[i - 1]:
                    delta[i] = vol[i]
                    last_polarity = 1
                elif close[i] < close[i - 1]:
                    delta[i] = -vol[i]
                    last_polarity = -1
                else:
                    # use last known polarity
                    delta[i] = vol[i] * last_polarity
            else:
                delta[i] = 0

    return pd.Series(delta, index=df_1m.index)


def aggregate_to_timeframe(df_1m, delta_1m, target_tf='15min'):
    """Aggregate 1m bars + delta into target timeframe bars.

    Returns DataFrame with OHLCV + delta_sum + cvd columns.
    """
    df = df_1m.copy()
    df['delta'] = delta_1m
    df = df.set_index('Open time')

    agg = {
        'Open': 'first',
        'High': 'max',
        'Low': 'min',
        'Close': 'last',
        'Volume': 'sum',
        'delta': 'sum',
    }

    resampled = df.resample(target_tf).agg(agg).dropna(subset=['Open']).reset_index()
    resampled['cvd'] = resampled['delta'].cumsum()
    return resampled


def detect_intrabar_divergence(df, lookback=24, window=12):
    """Detect CVD divergence on intrabar-aggregated bars.

    Same logic as detect_cvd_divergence_15m but works on any timeframe.
    Uses both slope comparison and swing high/low comparison.

    Returns Series of 'NONE', 'BEARISH', or 'BULLISH'.
    """
    cvd = df['cvd'].values
    close = df['Close'].values
    high = df['High'].values
    low = df['Low'].values
    n = len(df)
    divergence = ['NONE'] * n
    last_div_bar = -999

    for i in range(lookback + window, n):
        if i - last_div_bar < 6:
            continue

        # Method 1: Slope comparison
        w = window
        price_slice = close[i - w:i + 1]
        cvd_slice = cvd[i - w:i + 1]

        if len(price_slice) >= 3 and not np.any(np.isnan(cvd_slice)):
            x = np.arange(len(price_slice))
            price_slope = np.polyfit(x, price_slice, 1)[0]
            cvd_slope = np.polyfit(x, cvd_slice, 1)[0]
            price_range = np.max(price_slice) - np.min(price_slice)
            cvd_range = np.max(cvd_slice) - np.min(cvd_slice)

            if price_range > 0 and cvd_range > 0:
                price_dir = price_slope / price_range
                cvd_dir = cvd_slope / cvd_range
                # Relaxed threshold for intrabar (0.03 vs 0.05 for taker)
                if price_dir > 0.03 and cvd_dir < -0.03:
                    divergence[i] = 'BEARISH'
                elif price_dir < -0.03 and cvd_dir > 0.03:
                    divergence[i] = 'BULLISH'

        # Method 2: Swing high/low comparison
        if divergence[i] == 'NONE':
            look = min(lookback, i)
            if i >= 4 and (high[i] >= np.max(high[i - 3:i + 1]) * 0.9995):
                prev_hi = i - look
                if prev_hi >= 3 and high[prev_hi] >= np.max(high[max(0, prev_hi - 3):prev_hi + 1]) * 0.9995:
                    if high[i] > high[prev_hi] * 1.005:
                        cvd_at_i = np.nanmean(cvd[max(0, i - 1):i + 1])
                        cvd_at_prev = np.nanmean(cvd[max(0, prev_hi - 1):prev_hi + 1])
                        if cvd_at_i < cvd_at_prev * 0.990:
                            divergence[i] = 'BEARISH'

            if divergence[i] == 'NONE':
                if i >= 4 and (low[i] <= np.min(low[i - 3:i + 1]) * 1.0005):
                    prev_lo = i - look
                    if prev_lo >= 3 and low[prev_lo] <= np.min(low[max(0, prev_lo - 3):prev_lo + 1]) * 1.0005:
                        if low[i] < low[prev_lo] * 0.995:
                            cvd_at_i = np.nanmean(cvd[max(0, i - 1):i + 1])
                            cvd_at_prev = np.nanmean(cvd[max(0, prev_lo - 1):prev_lo + 1])
                            if cvd_at_i > cvd_at_prev * 1.010:
                                divergence[i] = 'BULLISH'

        # Method 3: Exhaustion
        if divergence[i] == 'NONE' and i >= 8:
            cvd_momentum = cvd[i] - cvd[i - 4]
            price_momentum = close[i] - close[i - 4]
            cvd_std = np.nanstd(cvd[max(0, i - 24):i + 1])
            if cvd_std > 0:
                if (price_momentum > 0 and cvd_momentum < 0 and
                        high[i] >= np.max(high[max(0, i - 8):i + 1]) * 0.999 and
                        abs(cvd_momentum) > cvd_std * 2.0):
                    divergence[i] = 'BEARISH'
                elif (price_momentum < 0 and cvd_momentum > 0 and
                      low[i] <= np.min(low[max(0, i - 8):i + 1]) * 1.001 and
                      abs(cvd_momentum) > cvd_std * 2.0):
                    divergence[i] = 'BULLISH'

        if divergence[i] != 'NONE':
            last_div_bar = i

    return pd.Series(divergence, index=df.index)


def get_intrabar_cvd_summary(symbol='ETHUSDT', target_tf='15min', hours=12):
    """Full pipeline: fetch 1m → compute delta → aggregate → detect divergence.

    Returns:
        df: DataFrame with OHLCV + delta + cvd + divergence columns
        latest_div: dict with divergence info at the latest bar
    """
    print(f"  📊 Fetching {hours}h of 1m bars for intrabar CVD...")
    df_1m = fetch_1m_bars(symbol, hours=hours)

    if df_1m.empty:
        return None, {'divergence': 'NONE', 'error': 'no data'}

    print(f"  📊 Computing intrabar delta ({len(df_1m)} bars)...")
    delta_1m = compute_intrabar_delta(df_1m)

    print(f"  📊 Aggregating to {target_tf}...")
    df = aggregate_to_timeframe(df_1m, delta_1m, target_tf)

    if len(df) < 36:
        return df, {'divergence': 'NONE', 'error': 'insufficient bars'}

    df['divergence'] = detect_intrabar_divergence(df, lookback=24, window=12)

    # Find latest divergence info
    idx = len(df) - 1
    latest_div = 'NONE'
    latest_bar = -1

    # Scan last 24 bars for any divergence
    for ci in range(max(0, idx - 24), idx + 1):
        d = df['divergence'].iloc[ci]
        if d != 'NONE':
            latest_div = d
            latest_bar = ci

    result = {
        'divergence': latest_div,
        'bars_ago': idx - latest_bar if latest_bar >= 0 else -1,
        'price_at_div': float(df['High'].iloc[latest_bar]) if latest_bar >= 0 else None,
        'cvd_at_div': float(df['cvd'].iloc[latest_bar]) if latest_bar >= 0 else None,
        'cvd_now': float(df['cvd'].iloc[idx]),
        'cvd_slope_12': float((df['cvd'].iloc[idx] - df['cvd'].iloc[max(0, idx - 12)]) / 12) if idx >= 12 else 0,
        'bars_total': len(df),
    }

    return df, result


def score_intrabar_divergence(div_result, direction):
    """Score the intrabar CVD divergence for ICS contribution.

    Returns (status, score, details):
      - Confirming div → positive score boost
      - Opposing div → negative score (penalty)
      - No div → neutral 0.5
    """
    div = div_result.get('divergence', 'NONE')
    bars_ago = div_result.get('bars_ago', -1)

    if div == 'NONE':
        return 'SKIP', 0.5, {'type': 'NONE', 'bars_ago': -1}

    is_confirming = (direction == 'LONG' and div == 'BULLISH') or \
                    (direction == 'SHORT' and div == 'BEARISH')
    is_opposing = (direction == 'LONG' and div == 'BEARISH') or \
                  (direction == 'SHORT' and div == 'BULLISH')

    # Freshness: closer bars = stronger signal
    freshness = max(0.3, 1.0 - bars_ago / 24.0) if bars_ago >= 0 else 0.3

    if is_confirming:
        score = 0.65 + 0.20 * freshness  # 0.65-0.85
        status = 'PASS'
    elif is_opposing:
        score = 0.50 - 0.30 * freshness  # 0.20-0.50
        status = 'WARN'
    else:
        score = 0.5
        status = 'SKIP'

    details = {
        'type': div,
        'confirming': is_confirming,
        'opposing': is_opposing,
        'bars_ago': bars_ago,
        'freshness': round(freshness, 3),
        'score': round(score, 3),
    }
    return status, score, details

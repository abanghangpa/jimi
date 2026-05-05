"""
M18: Squeeze Detector v5.1 — Dual-path with Entry Trigger System.

Path A (existing): 15m range compression → CVD taker trigger
Path B (new):      2h MACD(8,17,9) DIF/DEA convergence → histogram flip trigger

Entry trigger system:
  SQUEEZE_PENDING   — squeeze detected, waiting for breakout confirmation
  SQUEEZE_TRIGGERED — breakout confirmed, entry at breakout level

Entry price = the coil range boundary (high for LONG, low for SHORT)
Entry condition = "enter when 15m closes above/below $X"

Calibrated against 3 historical samples:
  - May 2-4 2026:   20h coil → $60+ move
  - Apr 24-26 2026: 36h coil → $50+ move
  - Apr 3-5 2026:   24h coil → $50+ move
"""

import numpy as np


SQUEEZE_V5_DEFAULTS = {
    # ── Path A: 15m compression ──
    'SQUEEZE_RANGE48_MAX': 1.2,
    'SQUEEZE_COMPRESSION_BARS_MIN': 12,
    'SQUEEZE_DRY_BARS_MIN': 4,
    'SQUEEZE_DOJI_BARS_MIN': 8,

    # Path A trigger: CVD taker
    'SQUEEZE_TAKER_LONG': 0.58,
    'SQUEEZE_TAKER_SHORT': 0.42,

    # ── Path B: 2h MACD coiling ──
    'SQUEEZE_MACD_FAST': 8,
    'SQUEEZE_MACD_SLOW': 17,
    'SQUEEZE_MACD_SIGNAL': 9,
    'SQUEEZE_COIL_DELTA_MAX': 0.05,     # DIF-DEA delta % for "coiled"
    'SQUEEZE_COIL_BARS_MIN': 6,         # minimum 2h bars in coil (12h)
    'SQUEEZE_COIL_BARS_STRONG': 9,      # strong coil (18h+)
    'SQUEEZE_HIST_FLIP_THRESHOLD': 0.0, # histogram crosses zero

    # ── Entry filters (v5.1 tuned) ──
    'SQUEEZE_REQUIRE_HIST_FLIP': True,  # only HIST_FLIP trigger (kills TAKER_SPIKE/COIL_DIR)
    'SQUEEZE_EMA_FILTER': True,          # require direction aligned with EMA21/55 trend
    'SQUEEZE_MIN_RSI': 20,              # RSI floor (oversold = good for longs)
    'SQUEEZE_MAX_RSI': 80,              # RSI ceiling (overbought = good for shorts)

    # ── Entry trigger ──
    'SQUEEZE_ENTRY_BUFFER_PCT': 0.001,  # 0.1% buffer above/below coil range
    'SQUEEZE_ENTRY_EXPIRY_BARS': 32,    # entry expires after 8h (32 x 15m)
    'SQUEEZE_BREAKOUT_VOL_MULT': 1.0,   # volume confirmation on breakout bar
    'SQUEEZE_ENTRY_MODE': 'TWO_BAR',    # 'BASELINE' (single close) or 'TWO_BAR' (2 consecutive)
    'SQUEEZE_COIL_HOURS_MIN': 12,       # skip squeezes with coil < 12h (backtest: <18h = 32% WR)

    # ── Exit levels ──
    'SQUEEZE_TP_ATR_MULT': 2.5,         # TP = 2.5x ATR from entry
    'SQUEEZE_SL_ATR_MULT': 1.0,         # SL = 1.0x ATR (data-driven: 0.8x too tight)
    'SQUEEZE_TP_MIN_PCT': 0.3,          # minimum TP distance
    'SQUEEZE_TP_MAX_PCT': 2.0,          # maximum TP distance

    # ── Override ──
    'SQUEEZE_OVERRIDE_REGIME': True,
    'SQUEEZE_ICS_BOOST': 0.10,
    'SQUEEZE_SIZE_MULT': 0.80,

    # ── Cooldown ──
    'SQUEEZE_COOLDOWN_BARS': 32,        # 8h on 15m
    'SQUEEZE_MAX_PENDING': 1,           # only 1 pending entry at a time (dedup)
}


def _check_compression(range48, compression_history, cfg):
    """Check if market is in 15m compression state (Path A)."""
    if range48 >= cfg['SQUEEZE_RANGE48_MAX']:
        return False, 0, 0, 0

    compression_bars = 0
    dry_count = 0
    doji_count = 0
    streak = 0
    gap_count = 0

    for r48, vr, br, tr in compression_history:
        if r48 < cfg['SQUEEZE_RANGE48_MAX']:
            streak += 1
            gap_count = 0
        elif gap_count < 2:
            gap_count += 1
            streak += 1
        else:
            compression_bars = max(compression_bars, streak)
            streak = 0
            gap_count = 0
        if vr < 0.6:
            dry_count += 1
        if br < 0.20:
            doji_count += 1

    compression_bars = max(compression_bars, streak)

    if range48 < cfg['SQUEEZE_RANGE48_MAX']:
        compression_bars += 1

    is_compressed = (
        compression_bars >= cfg['SQUEEZE_COMPRESSION_BARS_MIN'] and
        dry_count >= cfg['SQUEEZE_DRY_BARS_MIN'] and
        doji_count >= cfg['SQUEEZE_DOJI_BARS_MIN']
    )

    return is_compressed, compression_bars, dry_count, doji_count


def _compute_2h_macd(df_15m, cfg):
    """Resample 15m data to 2h and compute MACD(8,17,9)."""
    if len(df_15m) < 40:
        return None

    df_copy = df_15m.copy()
    if 'Open time' in df_copy.columns:
        df_copy = df_copy.set_index('Open time')

    agg_dict = {
        'Open': 'first', 'High': 'max', 'Low': 'min', 'Close': 'last',
        'Volume': 'sum',
    }
    if 'Taker buy base asset volume' in df_copy.columns:
        agg_dict['Taker buy base asset volume'] = 'sum'

    df_2h = df_copy.resample('2h').agg(agg_dict).dropna(subset=['Open'])

    if len(df_2h) < cfg['SQUEEZE_MACD_SLOW'] + cfg['SQUEEZE_MACD_SIGNAL']:
        return None

    close = df_2h['Close']
    ema_fast = close.ewm(span=cfg['SQUEEZE_MACD_FAST'], adjust=False).mean()
    ema_slow = close.ewm(span=cfg['SQUEEZE_MACD_SLOW'], adjust=False).mean()
    dif = ema_fast - ema_slow
    dea = dif.ewm(span=cfg['SQUEEZE_MACD_SIGNAL'], adjust=False).mean()
    hist = dif - dea

    return {
        'close': close.values,
        'high': df_2h['High'].values,
        'low': df_2h['Low'].values,
        'dif': dif.values,
        'dea': dea.values,
        'hist': hist.values,
        'timestamps': df_2h.index,
        'prices': close.values,
    }


def _detect_macd_coil(macd_data, cfg):
    """Detect 2h MACD DIF/DEA coiling (Path B).

    Returns coil stats + histogram flip status.
    """
    dif = macd_data['dif']
    dea = macd_data['dea']
    hist = macd_data['hist']
    close = macd_data['close']
    high = macd_data['high']
    low = macd_data['low']
    ts = macd_data['timestamps']

    n = len(dif)
    if n < 3:
        return False, 0, False, 'NEUTRAL', {}

    # DIF-DEA delta as % of price
    delta_pct = np.abs(dif - dea) / np.where(close > 0, close, 1.0) * 100

    # Find most recent coil streak (backwards from current bar)
    coil_threshold = cfg['SQUEEZE_COIL_DELTA_MAX']
    min_bars = cfg['SQUEEZE_COIL_BARS_MIN']
    coil_bars = 0
    gap_count = 0
    max_gaps = 2

    for i in range(n - 1, -1, -1):
        if delta_pct[i] < coil_threshold:
            coil_bars += 1
            gap_count = 0
        elif gap_count < max_gaps:
            gap_count += 1
            coil_bars += 1
        else:
            break

    # Longest streak anywhere
    max_streak = 0
    streak = 0
    for i in range(n):
        if delta_pct[i] < coil_threshold:
            streak += 1
            max_streak = max(max_streak, streak)
        else:
            streak = 0

    is_coiled = coil_bars >= min_bars

    # Compute coil range from the coiled bars
    if coil_bars > 0:
        coil_start = max(0, n - coil_bars)
        coil_high = float(np.max(high[coil_start:n]))
        coil_low = float(np.min(low[coil_start:n]))
    else:
        coil_high = float(close[-1])
        coil_low = float(close[-1])

    # Histogram flip detection
    hist_flip = False
    flip_direction = 'NEUTRAL'

    if n >= 2:
        prev_hist = hist[-2]
        curr_hist = hist[-1]
        if prev_hist < 0 and curr_hist >= 0:
            hist_flip = True
            flip_direction = 'LONG'
        elif prev_hist > 0 and curr_hist <= 0:
            hist_flip = True
            flip_direction = 'SHORT'

    # Histogram expanding (magnitude growing after near-zero)
    hist_expanding = False
    if n >= 3:
        prev_abs = abs(hist[-2])
        curr_abs = abs(hist[-1])
        if prev_abs < 0.5 and curr_abs > 1.0:
            hist_expanding = True

    # Direction from DIF slope during coil
    direction = 'NEUTRAL'
    if coil_bars >= 2:
        dif_slope = dif[-1] - dif[-(min(coil_bars, 6))]
        if dif_slope > 0.5:
            direction = 'LONG'
        elif dif_slope < -0.5:
            direction = 'SHORT'

    # Override direction if histogram flipped
    if hist_flip:
        direction = flip_direction

    details = {
        'coil_bars': coil_bars,
        'max_streak': max_streak,
        'coil_hours': coil_bars * 2,
        'max_streak_hours': max_streak * 2,
        'delta_pct_current': round(float(delta_pct[-1]), 4),
        'delta_pct_avg': round(float(np.mean(delta_pct[-coil_bars:])) if coil_bars > 0 else 999, 4),
        'hist_value': round(float(hist[-1]), 3),
        'hist_prev': round(float(hist[-2]) if n >= 2 else 0, 3),
        'hist_flip': hist_flip,
        'hist_expanding': hist_expanding,
        'flip_direction': flip_direction,
        'dif_current': round(float(dif[-1]), 3),
        'dea_current': round(float(dea[-1]), 3),
        'coil_high': round(coil_high, 2),
        'coil_low': round(coil_low, 2),
        'timestamp': str(ts[-1]) if n > 0 else '',
    }

    return is_coiled, coil_bars, hist_flip, direction, details


def _compute_coil_range_15m(df_15m, comp_bars, current_idx):
    """Compute the 15m compression range (high/low) for Path A."""
    if comp_bars <= 0 or current_idx < comp_bars:
        return None, None

    start = max(0, current_idx - comp_bars)
    h = float(df_15m['High'].iloc[start:current_idx+1].max())
    l = float(df_15m['Low'].iloc[start:current_idx+1].min())
    return h, l


def _check_entry_trigger(current_close, current_vol, vol_ma20,
                          coil_high, coil_low, direction, cfg):
    """Check if entry trigger fires (price breaks out of coil range).

    Returns:
        (triggered, entry_price, condition_text)
    """
    buffer = cfg['SQUEEZE_ENTRY_BUFFER_PCT']
    vol_mult = cfg['SQUEEZE_BREAKOUT_VOL_MULT']

    vol_confirmed = current_vol >= vol_ma20 * vol_mult if vol_ma20 > 0 else True

    if direction == 'LONG':
        breakout_level = coil_high * (1 + buffer)
        if current_close > breakout_level and vol_confirmed:
            return True, breakout_level, f"enter when 15m closes above ${breakout_level:.2f} (✅ CONFIRMED)"
        else:
            return False, breakout_level, f"enter when 15m closes above ${breakout_level:.2f} (⏳ waiting)"

    elif direction == 'SHORT':
        breakout_level = coil_low * (1 - buffer)
        if current_close < breakout_level and vol_confirmed:
            return True, breakout_level, f"enter when 15m closes below ${breakout_level:.2f} (✅ CONFIRMED)"
        else:
            return False, breakout_level, f"enter when 15m closes below ${breakout_level:.2f} (⏳ waiting)"

    return False, current_close, "no direction"


def _check_two_bar_trigger(prev_close, current_close, current_vol, vol_ma20,
                            coil_high, coil_low, direction, cfg):
    """Check if 2-bar confirmation trigger fires (2 consecutive closes beyond coil).

    Backtest results show this is the most consistent entry approach:
      - 2026: 43.5% WR, +0.152% expectancy (best at higher TP)
      - 2025: 30.0% WR, -0.050% expectancy (best across all approaches)
    """
    buffer = cfg['SQUEEZE_ENTRY_BUFFER_PCT']
    vol_mult = cfg['SQUEEZE_BREAKOUT_VOL_MULT']
    vol_confirmed = current_vol >= vol_ma20 * vol_mult if vol_ma20 > 0 else True

    if direction == 'LONG':
        breakout_level = coil_high * (1 + buffer)
        if current_close > breakout_level and prev_close > breakout_level:
            return True, breakout_level, f"2-bar close above ${breakout_level:.2f} (✅ CONFIRMED)"
        elif current_close > breakout_level:
            return False, breakout_level, f"2-bar: 1/2 closes above ${breakout_level:.2f} (⏳ need 2nd)"
        else:
            return False, breakout_level, f"2-bar: enter when 2 consecutive closes above ${breakout_level:.2f} (⏳ waiting)"

    elif direction == 'SHORT':
        breakout_level = coil_low * (1 - buffer)
        if current_close < breakout_level and prev_close < breakout_level:
            return True, breakout_level, f"2-bar close below ${breakout_level:.2f} (✅ CONFIRMED)"
        elif current_close < breakout_level:
            return False, breakout_level, f"2-bar: 1/2 closes below ${breakout_level:.2f} (⏳ need 2nd)"
        else:
            return False, breakout_level, f"2-bar: enter when 2 consecutive closes below ${breakout_level:.2f} (⏳ waiting)"

    return False, current_close, "no direction"


def _check_cvd_trigger(taker_ratio, pre_taker_avg, cfg):
    """Check if CVD trigger fires (taker spike during compression)."""
    if taker_ratio >= cfg['SQUEEZE_TAKER_LONG']:
        return True, 'LONG', 'TAKER_SPIKE'
    if taker_ratio <= cfg['SQUEEZE_TAKER_SHORT']:
        return True, 'SHORT', 'TAKER_SPIKE'

    if pre_taker_avg is not None:
        if taker_ratio > 0.52 and pre_taker_avg < 0.48:
            return True, 'LONG', 'TAKER_SHIFT'
        if taker_ratio < 0.48 and pre_taker_avg > 0.52:
            return True, 'SHORT', 'TAKER_SHIFT'

    return False, 'NEUTRAL', 'NONE'


def _score_quality(path_a, path_b, cfg):
    """Score squeeze quality based on which paths fired and their strength."""
    quality = 0.0
    factors = []
    is_strong = False

    a_compressed, a_comp_bars, a_dry, a_doji = path_a
    b_coiled, b_coil_bars, b_hist_flip, b_direction, b_details = path_b

    # Path A scoring
    if a_compressed:
        a_score = 0.0
        if a_comp_bars >= 24:
            a_score += 0.25
            factors.append(f'15m deep compression: {a_comp_bars} bars')
        elif a_comp_bars >= 12:
            a_score += 0.15
            factors.append(f'15m compression: {a_comp_bars} bars')
        a_score += min(0.15, a_dry / 12 * 0.15)
        a_score += min(0.10, a_doji / 12 * 0.10)
        quality += a_score

    # Path B scoring
    if b_coiled:
        b_score = 0.0
        coil_hours = b_details.get('coil_hours', 0)

        if coil_hours >= 24:
            b_score += 0.35
            factors.append(f'2h MACD coil: {coil_hours}h (DEEP)')
        elif coil_hours >= 18:
            b_score += 0.25
            factors.append(f'2h MACD coil: {coil_hours}h (STRONG)')
        elif coil_hours >= 12:
            b_score += 0.15
            factors.append(f'2h MACD coil: {coil_hours}h')

        if b_details.get('max_streak_hours', 0) >= 24:
            b_score += 0.10
            factors.append(f'Max coil streak: {b_details["max_streak_hours"]}h')

        if b_hist_flip:
            b_score += 0.20
            factors.append(f'2h histogram flip → {b_direction}')

        if b_details.get('hist_expanding'):
            b_score += 0.10
            factors.append('2h histogram expanding')

        quality += b_score

    # Dual-path agreement bonus
    if a_compressed and b_coiled:
        quality += 0.15
        factors.append('DUAL PATH: 15m + 2h both compressed')
        if b_hist_flip:
            is_strong = True
            factors.append('STRONG: dual compression + histogram flip')

    if quality >= 0.70:
        is_strong = True

    quality = min(quality, 1.0)

    return quality, is_strong, factors


def detect_squeeze_v5(result, config=None, last_signal_bar=-1, current_bar=0,
                       compression_history=None, df_15m=None,
                       magnets=None, liq_levels=None, sr_levels=None):
    """Dual-path squeeze detection with entry trigger system.

    Returns dict with:
        squeeze_type, squeeze_status (PENDING/TRIGGERED/NONE),
        entry_price, entry_condition, coil_high, coil_low, ...
    """
    cfg = {**SQUEEZE_V5_DEFAULTS, **(config or {})}

    # Cooldown
    if current_bar - last_signal_bar < cfg['SQUEEZE_COOLDOWN_BARS']:
        return _empty_result('cooldown')

    # Extract features
    price = result.get('price', 0)
    atr = result.get('atr', 0)
    range48 = result.get('range_width', 5)
    taker_ratio = result.get('raw_taker_ratio', 0.5)
    vol = result.get('vol_trend', 1.0)  # current vol / MA20

    # ── PATH A: 15m Compression ──
    a_compressed, a_comp_bars, a_dry, a_doji = _check_compression(
        range48, compression_history or [], cfg)

    # ── PATH B: 2h MACD Coiling ──
    b_coiled = False
    b_coil_bars = 0
    b_hist_flip = False
    b_direction = 'NEUTRAL'
    b_details = {}

    if df_15m is not None:
        macd_data = _compute_2h_macd(df_15m, cfg)
        if macd_data is not None:
            b_coiled, b_coil_bars, b_hist_flip, b_direction, b_details = \
                _detect_macd_coil(macd_data, cfg)

    # ── Check if either path detected a squeeze ──
    path_a_fired = a_compressed
    path_b_fired = b_coiled  # coil detection only (no hist_flip requirement for detection)

    if not path_a_fired and not path_b_fired:
        reasons = []
        if not a_compressed:
            reasons.append(f'15m: range48={range48:.2f}% comp={a_comp_bars}')
        if not b_coiled:
            reasons.append(f'2h: coil={b_coil_bars} bars')
        return _empty_result('; '.join(reasons))

    # ── CVD Trigger (for Path A) ──
    pre_taker_avg = None
    if compression_history and len(compression_history) >= 4:
        pre_takers = [tr for _, _, _, tr in compression_history[-12:]]
        pre_taker_avg = np.mean(pre_takers) if pre_takers else None

    cvd_triggered, cvd_direction, cvd_type = _check_cvd_trigger(
        taker_ratio, pre_taker_avg, cfg)

    # ── Resolve Direction ──
    direction = 'NEUTRAL'
    trigger_type = 'NONE'

    # Path B histogram flip is the strongest signal
    if path_b_fired and b_hist_flip:
        direction = b_direction
        trigger_type = 'HIST_FLIP'
    # Path A with CVD confirmation
    elif path_a_fired and cvd_triggered:
        direction = cvd_direction
        trigger_type = cvd_type
    # Path B coil without flip — direction from DIF slope
    elif path_b_fired and b_direction != 'NEUTRAL':
        direction = b_direction
        trigger_type = 'COIL_DIR'

    if direction == 'NEUTRAL':
        return _empty_result(f'no direction: cvd={cvd_type} hist_flip={b_hist_flip}')

    # ── Coil Duration Filter ──
    # Backtest: <18h coils have 32% WR vs 58% for ≥18h. Skip short coils.
    coil_hours = b_details.get('coil_hours', 0)
    coil_hours_min = cfg.get('SQUEEZE_COIL_HOURS_MIN', 12)
    if coil_hours < coil_hours_min and not a_compressed:
        return _empty_result(f'coil={coil_hours}h < {coil_hours_min}h minimum')

    # ── v5.1 Filters (tuned for 70%+ WR) ──

    # Filter 1: Require HIST_FLIP trigger (kills TAKER_SPIKE/COIL_DIR noise)
    if cfg.get('SQUEEZE_REQUIRE_HIST_FLIP', False) and trigger_type != 'HIST_FLIP':
        return _empty_result(f'trigger={trigger_type} (need HIST_FLIP)')

    # Filter 2: EMA trend alignment
    if cfg.get('SQUEEZE_EMA_FILTER', False) and df_15m is not None and len(df_15m) >= 55:
        close_series = df_15m['Close'] if 'Close' in df_15m.columns else df_15m.iloc[:, 4]
        ema21 = float(close_series.ewm(span=21, adjust=False).mean().iloc[-1])
        ema55 = float(close_series.ewm(span=55, adjust=False).mean().iloc[-1])
        ema_trend = 'BULL' if ema21 > ema55 else 'BEAR'

        aligned = (direction == 'LONG' and ema_trend == 'BULL') or \
                  (direction == 'SHORT' and ema_trend == 'BEAR')
        if not aligned:
            return _empty_result(f'EMA contra: dir={direction} trend={ema_trend}')

    # Filter 3: RSI bounds
    if cfg.get('SQUEEZE_MIN_RSI') is not None and df_15m is not None and len(df_15m) >= 14:
        close_series = df_15m['Close'] if 'Close' in df_15m.columns else df_15m.iloc[:, 4]
        delta = close_series.diff()
        gain = delta.clip(lower=0).rolling(14).mean()
        loss = (-delta.clip(upper=0)).rolling(14).mean()
        rsi = 100 - (100 / (1 + gain.iloc[-1] / loss.iloc[-1])) if loss.iloc[-1] > 0 else 50

        if direction == 'LONG' and rsi > cfg['SQUEEZE_MAX_RSI']:
            return _empty_result(f'RSI={rsi:.0f} overbought for LONG')
        if direction == 'SHORT' and rsi < cfg['SQUEEZE_MIN_RSI']:
            return _empty_result(f'RSI={rsi:.0f} oversold for SHORT')

    # ── Compute Coil Range (entry levels) ──
    coil_high = b_details.get('coil_high', price)
    coil_low = b_details.get('coil_low', price)

    # If Path A also fired, use its range if tighter
    if path_a_fired and df_15m is not None:
        a_high, a_low = _compute_coil_range_15m(df_15m, a_comp_bars, len(df_15m) - 1)
        if a_high is not None and a_low is not None:
            # Use the tighter of the two ranges
            if (a_high - a_low) < (coil_high - coil_low):
                coil_high = a_high
                coil_low = a_low

    # ── Entry Trigger ──
    vol_ma20 = result.get('vol_ma20', vol * 20)  # fallback
    entry_mode = cfg.get('SQUEEZE_ENTRY_MODE', 'TWO_BAR')

    if entry_mode == 'TWO_BAR':
        # 2-bar confirmation: require 2 consecutive closes beyond coil
        prev_close = float(df_15m['Close'].iloc[-2]) if df_15m is not None and len(df_15m) >= 2 else price
        entry_triggered, entry_price, entry_condition = _check_two_bar_trigger(
            prev_close, price, price * vol,
            vol_ma20 if vol_ma20 > 0 else price,
            coil_high, coil_low, direction, cfg)
    else:
        # BASELINE: single close beyond coil
        entry_triggered, entry_price, entry_condition = _check_entry_trigger(
            price, price * vol, vol_ma20 if vol_ma20 > 0 else price,
            coil_high, coil_low, direction, cfg)

    # ── Squeeze Status ──
    if entry_triggered:
        squeeze_status = 'TRIGGERED'
    else:
        squeeze_status = 'PENDING'

    # ── Quality Score ──
    quality, is_strong, factors = _score_quality(
        (a_compressed, a_comp_bars, a_dry, a_doji),
        (b_coiled, b_coil_bars, b_hist_flip, b_direction, b_details),
        cfg
    )

    factors.append(f'Trigger: {trigger_type} → {direction}')
    if cvd_triggered and path_b_fired:
        factors.append(f'CVD confirms: {cvd_direction}')

    # ── Squeeze Type ──
    squeeze_type = 'SHORT_SQUEEZE' if direction == 'LONG' else 'LONG_SQUEEZE'

    # ── Extract coil hours for TP scaling ──
    coil_hours = b_details.get('coil_hours', 0)

    # ── TP/SL Levels (liquidity-aware with ATR fallback) ──
    # NOTE: TP/SL are calculated from ENTRY PRICE (breakout level), not current price.
    # For LONG: entry = coil_high breakout; for SHORT: entry = coil_low breakdown.
    tp1_source = 'ATR'
    if magnets is not None and liq_levels is not None:
        # Use liquidity-aware calc_trade_levels from src.sl_tp
        try:
            from src.sl_tp import calc_trade_levels
            vol_ratio = result.get('vol_ratio', 1.0)
            _liq_for_tp = liq_levels if isinstance(liq_levels, dict) else None
            _levels = calc_trade_levels(
                entry_price, direction, atr, vol_ratio,
                magnets=magnets,
                sr_levels=sr_levels or [],
                liq_levels=_liq_for_tp,
                cfg=cfg,
            )
            # Use liquidity TP only if it found an unswept pool (not ATR fallback)
            if _levels.get('tp1_source') == 'UNSWEPT_POOL':
                tp_dist = abs(_levels['tp1'] - entry_price)
                sl_dist = abs(entry_price - _levels['sl'])
                tp1_source = 'UNSWEPT_POOL'
            else:
                # ATR fallback from calc_trade_levels or raw ATR
                raise ValueError('ATR fallback')
        except Exception:
            # Fall back to raw ATR
            if atr > 0 and entry_price > 0:
                tp_dist = max(
                    atr * cfg['SQUEEZE_TP_ATR_MULT'],
                    entry_price * cfg['SQUEEZE_TP_MIN_PCT'] / 100
                )
                tp_dist = min(tp_dist, entry_price * cfg['SQUEEZE_TP_MAX_PCT'] / 100)
                sl_dist = atr * cfg['SQUEEZE_SL_ATR_MULT']
            else:
                tp_dist = entry_price * 0.5 / 100
                sl_dist = entry_price * 0.2 / 100
    else:
        if atr > 0 and entry_price > 0:
            tp_dist = max(
                atr * cfg['SQUEEZE_TP_ATR_MULT'],
                entry_price * cfg['SQUEEZE_TP_MIN_PCT'] / 100
            )
            tp_dist = min(tp_dist, entry_price * cfg['SQUEEZE_TP_MAX_PCT'] / 100)
            sl_dist = atr * cfg['SQUEEZE_SL_ATR_MULT']
        else:
            tp_dist = entry_price * 0.5 / 100
            sl_dist = entry_price * 0.2 / 100

    # ── Scale TP by coil duration (long coils have further targets) ──
    if coil_hours >= 18:
        tp_duration_mult = 1.0 + min((coil_hours - 18) / 36, 1.0)  # up to 2x for 54h+ coils
    elif coil_hours >= 12:
        tp_duration_mult = 1.0 + (coil_hours - 12) / 18 * 0.3  # 1.0-1.3x
    else:
        tp_duration_mult = 1.0

    tp_dist *= tp_duration_mult

    # ── Measured move floor: coil width as minimum TP ──
    # A squeeze stores energy proportional to its range. The breakout should
    # travel at least the coil width. Use 0.7x as floor (conservative).
    coil_high = b_details.get('coil_high', 0)
    coil_low = b_details.get('coil_low', 0)
    if coil_high > 0 and coil_low > 0:
        coil_width = abs(coil_high - coil_low)
        measured_move_floor = coil_width * 0.3
        if tp_dist < measured_move_floor:
            tp_dist = measured_move_floor

    # Clamp TP distance (from entry price)
    if entry_price > 0:
        tp_dist = min(tp_dist, entry_price * cfg['SQUEEZE_TP_MAX_PCT'] / 100)

    if direction == 'LONG':
        tp = entry_price + tp_dist
        sl = entry_price - sl_dist
    else:
        tp = entry_price - tp_dist
        sl = entry_price + sl_dist

    # Percentages from entry price (not current price)
    tp_pct = tp_dist / entry_price * 100 if entry_price > 0 else 0
    sl_pct = sl_dist / entry_price * 100 if entry_price > 0 else 0

    return {
        'squeeze_type': squeeze_type,
        'squeeze_status': squeeze_status,
        'squeeze_score': round(quality, 3),
        'squeeze_strong': is_strong,
        'direction': direction,
        'factors': factors,
        'quality': round(quality, 3),
        'compression_bars': a_comp_bars,
        'dry_count': a_dry,
        'doji_count': a_doji,
        'trigger_type': trigger_type,
        'overrides_regime': cfg['SQUEEZE_OVERRIDE_REGIME'],
        'ics_boost': cfg['SQUEEZE_ICS_BOOST'],
        'size_mult': cfg['SQUEEZE_SIZE_MULT'],
        # Entry trigger
        'entry_price': round(entry_price, 2),
        'entry_condition': entry_condition,
        'entry_triggered': entry_triggered,
        'coil_high': round(coil_high, 2),
        'coil_low': round(coil_low, 2),
        # Levels
        'tp': round(tp, 2),
        'sl': round(sl, 2),
        'tp_pct': round(tp_pct, 3),
        'sl_pct': round(sl_pct, 3),
        'tp1_source': tp1_source,
        'tp_duration_mult': round(tp_duration_mult, 3),
        'coil_hours': coil_hours,
        # Meta
        'gates_all_pass': True,
        'gates_passed': factors,
        'gates_failed': [],
        'short_score': round(quality, 3) if squeeze_type == 'SHORT_SQUEEZE' else 0,
        'long_score': round(quality, 3) if squeeze_type == 'LONG_SQUEEZE' else 0,
        'path_a': {
            'fired': path_a_fired,
            'compressed': a_compressed,
            'comp_bars': a_comp_bars,
            'dry': a_dry,
            'doji': a_doji,
        },
        'path_b': {
            'fired': path_b_fired,
            'coiled': b_coiled,
            'coil_bars': b_coil_bars,
            'coil_hours': b_details.get('coil_hours', 0),
            'max_streak_hours': b_details.get('max_streak_hours', 0),
            'hist_flip': b_hist_flip,
            'hist_value': b_details.get('hist_value', 0),
            'delta_pct': b_details.get('delta_pct_current', 0),
            'direction': b_direction,
            'dif': b_details.get('dif_current', 0),
            'dea': b_details.get('dea_current', 0),
        },
    }


def _empty_result(reason):
    """Return empty squeeze result."""
    return {
        'squeeze_type': 'NONE', 'squeeze_status': 'NONE',
        'squeeze_score': 0, 'squeeze_strong': False,
        'direction': 'NEUTRAL', 'factors': [], 'quality': 0,
        'compression_bars': 0, 'dry_count': 0, 'doji_count': 0,
        'trigger_type': 'NONE',
        'overrides_regime': False, 'ics_boost': 0, 'size_mult': 1.0,
        'entry_price': 0, 'entry_condition': '', 'entry_triggered': False,
        'coil_high': 0, 'coil_low': 0,
        'tp': 0, 'sl': 0, 'tp_pct': 0, 'sl_pct': 0,
        'tp1_source': 'ATR', 'tp_duration_mult': 1.0, 'coil_hours': 0,
        'gates_all_pass': False, 'gates_passed': [], 'gates_failed': [reason],
        'short_score': 0, 'long_score': 0,
        'path_a': {'fired': False}, 'path_b': {'fired': False},
    }


def format_squeeze(sq):
    """Format squeeze result for terminal output."""
    if sq.get('squeeze_type', 'NONE') == 'NONE':
        failed = sq.get('gates_failed', [])
        if failed and failed[0] != 'cooldown':
            lines = ['', '  🔥 SQUEEZE DETECTOR — GATE FAILED']
            for g in failed:
                lines.append(f'    ❌ {g}')
            return '\n'.join(lines)
        return ''

    lines = []
    lines.append('')
    lines.append('  🔥 SQUEEZE DETECTOR v5.1')

    icons = {'SHORT_SQUEEZE': '🟩', 'LONG_SQUEEZE': '🟥'}
    icon = icons.get(sq['squeeze_type'], '❓')
    strength = 'STRONG' if sq.get('squeeze_strong') else 'CONFIRMED'

    status = sq.get('squeeze_status', 'NONE')
    status_icons = {'TRIGGERED': '✅', 'PENDING': '⏳'}
    status_icon = status_icons.get(status, '❓')

    lines.append(f'  Type: {icon} {sq["squeeze_type"]}  ({strength})')
    lines.append(f'  Status: {status_icon} {status}')
    lines.append(f'  Score: {sq["squeeze_score"]:.3f}  '
                 f'Trigger: {sq.get("trigger_type", "?")}')

    # Path details
    pa = sq.get('path_a', {})
    pb = sq.get('path_b', {})

    if pa.get('fired'):
        lines.append(f'  Path A (15m): ✅ compressed {pa.get("comp_bars", 0)} bars  '
                     f'dry={pa.get("dry", 0)}  doji={pa.get("doji", 0)}')

    if pb.get('coiled') or pb.get('fired'):
        coil_h = pb.get('coil_hours', 0)
        max_h = pb.get('max_streak_hours', 0)
        flip = '✅ FLIP' if pb.get('hist_flip') else '⏳ waiting'
        lines.append(f'  Path B (2h):  ✅ coiled {coil_h}h (max {max_h}h)  '
                     f'hist={pb.get("hist_value", 0):.3f}  {flip}')
        lines.append(f'    MACD(8,17,9): DIF={pb.get("dif", 0):.3f}  DEA={pb.get("dea", 0):.3f}  '
                     f'Δ={pb.get("delta_pct", 0):.4f}%')

    lines.append(f'  Direction: {sq["direction"]}')

    # Coil range + entry
    coil_h = sq.get('coil_high', 0)
    coil_l = sq.get('coil_low', 0)
    if coil_h > 0 and coil_l > 0:
        lines.append(f'  Coil range: ${coil_l:.2f} - ${coil_h:.2f}  '
                     f'(width ${(coil_h - coil_l):.2f})')

    entry_cond = sq.get('entry_condition', '')
    if entry_cond:
        lines.append(f'  Entry: {entry_cond}')
        lines.append(f'  Entry price: ${sq.get("entry_price", 0):.2f}')

    if sq.get('overrides_regime') and status == 'TRIGGERED':
        lines.append(f'  ⚡ Overrides M9 regime block!')

    if sq.get('factors'):
        lines.append(f'\n  Factors:')
        for f in sq['factors']:
            lines.append(f'    ✅ {f}')

    if sq.get('tp', 0) > 0:
        tp_src = sq.get('tp1_source', 'ATR')
        dur_mult = sq.get('tp_duration_mult', 1.0)
        coil_h = sq.get('coil_hours', 0)
        dur_note = f'  coil={coil_h}h dur_mult={dur_mult:.2f}' if dur_mult > 1.0 else ''
        lines.append(f'\n  TP: ${sq["tp"]:.2f} ({sq["tp_pct"]:.2f}%)  [{tp_src}]{dur_note}  '
                     f'SL: ${sq["sl"]:.2f} ({sq["sl_pct"]:.2f}%)')

    if sq.get('ics_boost', 0) > 0 and status == 'TRIGGERED':
        lines.append(f'  ICS boost: +{sq["ics_boost"]:.4f}')

    return '\n'.join(lines)

#!/usr/bin/env python3
"""
Conflict Analyzer — scans Apr-May 2026 for direction conflicts,
tracks outcomes at 4h/12h/24h/48h/72h to find which way it actually resolves.
"""
import sys, os, json
import numpy as np
import pandas as pd
from collections import defaultdict

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from src.config import CONFIG
from src.utils.data_handler import load_data, resample_ohlcv
from src.utils.indicators import (
    calc_ema, calc_macd, calc_rsi, calc_atr, calc_vwap, calc_vol_ratio,
    calc_swing_bias, calc_phase0, calc_trend_state,
)
from src.modules.m1_macd_v2 import score_m1_v2 as score_m1
from src.modules.m2_ema import score_m2
from src.modules.m3_vwap import score_m3
from src.modules.m4_cvd import calc_cvd_15m, detect_cvd_divergence_15m, calc_cvd_2h, detect_cvd_zero_cross, score_m4
from src.modules.m5_liquidation import build_volume_profile, find_magnets, find_gaps, score_m5, find_support_resistance
from src.modules.m9_volatility import RegimeState, compute_vol_regime, score_vol_regime
from src.modules.m13_structure import score_m13
from src.modules.direction_resolver import resolve_direction, score_targets
from src.engine import calc_ics

CSV_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'eth_15m_merged.csv')
START = '2026-04-01'
END = '2026-05-12'  # exclusive

# Outcome windows in bars (15m bars)
OUTCOME_WINDOWS = {
    '4h': 16,
    '12h': 48,
    '24h': 96,
    '48h': 192,
    '72h': 288,
}

def compute_all(df_15m, cfg):
    df_15m['vwap'] = calc_vwap(df_15m['High'], df_15m['Low'], df_15m['Close'], df_15m['Volume'], cfg['VWAP_LOOKBACK'])
    df_15m['vol_ma20'] = df_15m['Volume'].rolling(20).mean()
    taker_base = df_15m['Taker buy base asset volume']
    total_vol = df_15m['Volume']
    df_15m['taker_ratio'] = (taker_base / total_vol.replace(0, np.nan)).fillna(cfg['TAKER_FILLNA'])
    df_15m['atr'] = calc_atr(df_15m['High'], df_15m['Low'], df_15m['Close'], cfg['ATR_PERIOD'])
    df_15m['vol_ratio'] = calc_vol_ratio(df_15m['Volume'])
    df_15m['rsi'] = calc_rsi(df_15m['Close'], 14)

    df_1h = resample_ohlcv(df_15m, '1H')
    df_2h = resample_ohlcv(df_15m, '2H')
    df_4h = resample_ohlcv(df_15m, '4H')
    df_1d = resample_ohlcv(df_15m, '1D')

    df_1h['macd_line'], df_1h['macd_signal'], df_1h['macd_hist'] = calc_macd(
        df_1h['Close'], cfg['MACD_FAST'], cfg['MACD_SLOW'], cfg['MACD_SIGNAL'])
    df_1h['ema_fast'] = calc_ema(df_1h['Close'], cfg['EMA_FAST'])
    df_1h['ema_slow'] = calc_ema(df_1h['Close'], cfg['EMA_SLOW'])
    df_1h['atr'] = calc_atr(df_1h['High'], df_1h['Low'], df_1h['Close'], cfg['ATR_PERIOD'])
    df_1h['rsi'] = calc_rsi(df_1h['Close'], 14)
    df_4h['ema_fast'] = calc_ema(df_4h['Close'], cfg['EMA_FAST'])
    df_4h['ema_slow'] = calc_ema(df_4h['Close'], cfg['EMA_SLOW'])
    df_2h['ema_fast'] = calc_ema(df_2h['Close'], cfg['EMA_FAST'])
    df_2h['ema_slow'] = calc_ema(df_2h['Close'], cfg['EMA_SLOW'])
    df_15m['cvd_15m'] = calc_cvd_15m(df_15m)
    df_15m['cvd_divergence_15m'] = detect_cvd_divergence_15m(df_15m, cfg['CVD_LOOKBACK'], cfg['CVD_DIVERGENCE_WINDOW'])
    df_2h['cvd_2h'] = calc_cvd_2h(df_2h)
    df_2h['cvd_zl_state'], df_2h['cvd_zl_cross_bar'], df_2h['cvd_zl_cross_dir'] = detect_cvd_zero_cross(df_2h)
    df_1d['swing_bias'] = calc_swing_bias(df_1d)
    df_1d['phase0'] = calc_phase0(df_1d)
    df_1d['trend'], df_1d['trend_score'] = calc_trend_state(df_1d)
    df_4h['macd_line'], df_4h['macd_signal'], df_4h['macd_hist'] = calc_macd(
        df_4h['Close'], cfg['MACD_FAST'], cfg['MACD_SLOW'], cfg['MACD_SIGNAL'])

    return df_15m, df_1h, df_2h, df_4h, df_1d


def scan_bar(df_15m, df_1h, df_2h, df_4h, df_1d, idx, cfg):
    row = df_15m.iloc[idx]
    ts = str(row['Open time'])
    price = float(row['Close'])

    idx_1h = df_1h.index.searchsorted(df_15m.index[idx]) if hasattr(df_15m.index[idx], 'value') else len(df_1h) - 1
    idx_2h = df_2h.index.searchsorted(df_15m.index[idx]) if hasattr(df_15m.index[idx], 'value') else len(df_2h) - 1
    idx_4h = df_4h.index.searchsorted(df_15m.index[idx]) if hasattr(df_15m.index[idx], 'value') else len(df_4h) - 1
    idx_1d = df_1d.index.searchsorted(df_15m.index[idx]) if hasattr(df_15m.index[idx], 'value') else len(df_1d) - 1
    idx_1h = min(idx_1h, len(df_1h) - 1)
    idx_2h = min(idx_2h, len(df_2h) - 1)
    idx_4h = min(idx_4h, len(df_4h) - 1)
    idx_1d = min(idx_1d, len(df_1d) - 1)

    swing_bias = df_1d['swing_bias'].iloc[idx_1d] if 'swing_bias' in df_1d.columns else 'N/A'
    phase0_val = df_1d['phase0'].iloc[idx_1d] if 'phase0' in df_1d.columns else None
    trend_dir = df_1d['trend'].iloc[idx_1d] if 'trend' in df_1d.columns else 'N/A'

    regime_state = RegimeState(config=cfg)
    vol_regime, m9_raw, m9_vol_details = compute_vol_regime(
        df_15m, df_1h, idx, idx_1h, regime_state=regime_state, config=cfg)

    m13_status, m13_score_raw, m13_details = score_m13(df_1h, idx_1h, 'NEUTRAL', df_15m, idx)
    m13_bias = m13_details.get('m13_bias', 'NEUTRAL')

    direction, dir_size_mult, dir_details = resolve_direction(
        vol_regime, m9_raw if m9_raw else 0.5,
        m13_bias, m13_score_raw, m13_details,
        swing_bias_1d=swing_bias, trend_dir=trend_dir, config=cfg,
    )

    m1_dir, m1_score, _ = score_m1(df_1h, idx_1h, cfg, df_15m=df_15m, idx_15m=idx)
    m1_actual = m1_dir  # keep original direction
    if m1_dir == 'BEARISH' and direction == 'LONG':
        m1_score = 1.0 - m1_score
    elif m1_dir == 'BULLISH' and direction == 'SHORT':
        m1_score = 1.0 - m1_score

    m2_status, m2_score = score_m2(df_1h, df_2h, df_4h, df_1d, idx_1h, idx_2h, idx_4h, idx_1d)
    m3_status, m3_score, _ = score_m3(df_15m, idx, direction, cfg)
    m4_status, m4_score, m4_div = score_m4(df_15m, df_2h, idx, idx_2h, direction, cfg)
    m4_div_str = 'NONE'
    if isinstance(m4_div, dict):
        m4_div_str = m4_div.get('layer_a_div', 'NONE')
    # v7.2: Normalize _BASE variants
    if m4_div_str.endswith('_BASE'):
        m4_div_str = m4_div_str.replace('_BASE', '')
    m5_status, m5_score, m5_details = score_m5(df_15m, idx, direction, cfg,
        n_bins=cfg['M5_VP_BINS'], lookback=cfg['M5_VP_LOOKBACK'])
    m9_status, m9_score, _ = score_vol_regime(vol_regime, m9_raw, direction, trend_dir)

    ics, floor = calc_ics(
        m1_score, m2_score, m3_score, m4_score, m4_status, m5_score,
        m9_score=m9_score, use_m9=True, config=cfg,
    )

    from src.engine import check_entry_filters
    atr_1h = df_1h['atr'].iloc[idx_1h]
    passed, reason = check_entry_filters(df_15m, idx, direction, swing_bias, phase0_val, atr_1h, config=cfg)

    # Compute conflict indicators
    conflicts = []
    conflict_score = 0

    # 1. M1 vs Direction divergence
    if (m1_actual == 'BEARISH' and direction == 'LONG') or (m1_actual == 'BULLISH' and direction == 'SHORT'):
        conflicts.append(f'M1_vs_DIR({m1_actual}≠{direction})')
        conflict_score += 1

    # 2. Phase0 death zone
    if phase0_val is not None and phase0_val < 0.15:
        conflicts.append(f'PHASE0_LOW({phase0_val:.3f})')
        conflict_score += 1

    # 3. Direction vs swing_bias divergence
    if (direction == 'LONG' and swing_bias == 'BEARISH') or (direction == 'SHORT' and swing_bias == 'BULLISH'):
        conflicts.append(f'DIR_vs_BIAS({direction}≠{swing_bias})')
        conflict_score += 1

    # 4. M4 CVD contradiction
    if m4_div_str != 'NONE':
        if (direction == 'LONG' and 'BEARISH' in m4_div_str) or (direction == 'SHORT' and 'BULLISH' in m4_div_str):
            conflicts.append(f'CVD_CONFLICT({m4_div_str})')
            conflict_score += 1

    # 5. Low ICS with direction
    if ics < 0.5:
        conflicts.append(f'LOW_ICS({ics:.3f})')
        conflict_score += 1

    # 6. M13 bias vs direction
    if (direction == 'LONG' and m13_bias == 'BEARISH') or (direction == 'SHORT' and m13_bias == 'BULLISH'):
        conflicts.append(f'M13_CONFLICT({m13_bias}≠{direction})')
        conflict_score += 1

    return {
        'idx': idx,
        'timestamp': ts,
        'price': price,
        'direction': direction,
        'swing_bias': swing_bias,
        'phase0': float(phase0_val) if phase0_val is not None and not pd.isna(phase0_val) else None,
        'trend_dir': trend_dir,
        'regime': vol_regime,
        'm13_bias': m13_bias,
        'm1_dir': m1_actual,
        'm1_score': round(float(m1_score), 3),
        'm3_score': round(float(m3_score), 3),
        'm4_score': round(float(m4_score), 3),
        'm4_div': m4_div_str,
        'm5_score': round(float(m5_score), 3),
        'm9_score': round(float(m9_score), 3),
        'ics': round(float(ics), 4),
        'entry_pass': passed,
        'entry_reason': reason if not passed else 'OK',
        'conflicts': conflicts,
        'conflict_score': conflict_score,
    }


def compute_outcomes(df_15m, idx, total_bars):
    """Compute price outcomes at various horizons."""
    price = float(df_15m['Close'].iloc[idx])
    outcomes = {}
    for label, bars in OUTCOME_WINDOWS.items():
        future_idx = idx + bars
        if future_idx < total_bars:
            future_price = float(df_15m['Close'].iloc[future_idx])
            pct = (future_price - price) / price * 100
            # Also get the high/low within the window
            window_high = float(df_15m['High'].iloc[idx:future_idx+1].max())
            window_low = float(df_15m['Low'].iloc[idx:future_idx+1].min())
            max_up = (window_high - price) / price * 100
            max_down = (window_low - price) / price * 100
            outcomes[label] = {
                'pct': round(pct, 3),
                'max_up': round(max_up, 3),
                'max_down': round(max_down, 3),
                'future_price': round(future_price, 2),
            }
        else:
            outcomes[label] = None
    return outcomes


def classify_outcome(direction, outcomes):
    """Classify if the direction was correct at each horizon."""
    result = {}
    for label, data in outcomes.items():
        if data is None:
            result[label] = 'N/A'
            continue
        pct = data['pct']
        if direction == 'LONG':
            if pct > 0.5:
                result[label] = 'WIN'
            elif pct < -0.5:
                result[label] = 'LOSS'
            else:
                result[label] = 'FLAT'
        elif direction == 'SHORT':
            if pct < -0.5:
                result[label] = 'WIN'
            elif pct > 0.5:
                result[label] = 'LOSS'
            else:
                result[label] = 'FLAT'
        else:
            result[label] = 'N/A'
    return result


def main():
    cfg = dict(CONFIG)
    print(f"Loading CSV...")
    df_all = load_data(CSV_PATH)
    print(f"  Total bars: {len(df_all)} ({df_all['Open time'].iloc[0]} → {df_all['Open time'].iloc[-1]})")

    # Filter to date range
    mask = (df_all['Open time'] >= START) & (df_all['Open time'] < END)
    df_range = df_all[mask].copy()
    print(f"  Target range: {START} → {END} ({len(df_range)} bars)")

    if len(df_range) == 0:
        print("No data in range!")
        return

    # Need warmup for indicators
    first_ts = df_range['Open time'].iloc[0]
    warmup_mask = df_all['Open time'] < first_ts
    df_warmup = df_all[warmup_mask].tail(1000)
    df_compute = pd.concat([df_warmup, df_range]).reset_index(drop=True)

    print(f"Computing indicators on {len(df_compute)} bars...")
    df_15m, df_1h, df_2h, df_4h, df_1d = compute_all(df_compute, cfg)

    range_start = len(df_warmup)
    range_end = len(df_compute)
    total_bars = len(df_15m)

    # Scan every 4th bar (hourly) for efficiency
    step = 4
    print(f"Scanning {range_start} → {range_end} (step={step})...")

    all_results = []
    for i in range(range_start, range_end, step):
        r = scan_bar(df_15m, df_1h, df_2h, df_4h, df_1d, i, cfg)
        r['outcomes'] = compute_outcomes(df_15m, i, total_bars)
        r['outcome_class'] = classify_outcome(r['direction'], r['outcomes'])
        all_results.append(r)

    print(f"  Scanned {len(all_results)} bars\n")

    # ═══════════════════════════════════════════════════════════
    # ANALYSIS 1: All conflict events (conflict_score >= 2)
    # ═══════════════════════════════════════════════════════════
    conflict_events = [r for r in all_results if r['conflict_score'] >= 2]
    print(f"{'═'*90}")
    print(f"  CONFLICT ANALYSIS — Apr 1 to May 11, 2026")
    print(f"{'═'*90}")
    print(f"\n  Total bars scanned: {len(all_results)}")
    print(f"  Conflict events (≥2 conflict indicators): {len(conflict_events)}")
    print(f"  No-conflict bars: {len(all_results) - len(conflict_events)}")

    if not conflict_events:
        print("  No conflict events found!")
        return

    # ═══════════════════════════════════════════════════════════
    # ANALYSIS 2: Outcome stats for conflict events
    # ═══════════════════════════════════════════════════════════
    print(f"\n{'═'*90}")
    print(f"  OUTCOME ANALYSIS — What happened after conflict signals?")
    print(f"{'═'*90}")

    for horizon in ['4h', '12h', '24h', '48h', '72h']:
        valid = [r for r in conflict_events if r['outcomes'].get(horizon) is not None]
        if not valid:
            continue

        # Group by direction
        longs = [r for r in valid if r['direction'] == 'LONG']
        shorts = [r for r in valid if r['direction'] == 'SHORT']

        print(f"\n  ── {horizon} after conflict ──")
        print(f"  Total conflict events with outcome data: {len(valid)}")

        for label, group in [('LONG-directed', longs), ('SHORT-directed', shorts)]:
            if not group:
                continue
            pcts = [r['outcomes'][horizon]['pct'] for r in group]
            max_ups = [r['outcomes'][horizon]['max_up'] for r in group]
            max_downs = [r['outcomes'][horizon]['max_down'] for r in group]
            wins = sum(1 for r in group if r['outcome_class'][horizon] == 'WIN')
            losses = sum(1 for r in group if r['outcome_class'][horizon] == 'LOSS')
            flats = sum(1 for r in group if r['outcome_class'][horizon] == 'FLAT')

            avg_pct = np.mean(pcts)
            med_pct = np.median(pcts)
            avg_max_up = np.mean(max_ups)
            avg_max_down = np.mean(max_downs)

            print(f"\n    {label} (n={len(group)}):")
            print(f"      Win rate:  {wins}/{len(group)} = {wins/len(group)*100:.1f}%")
            print(f"      Loss rate: {losses}/{len(group)} = {losses/len(group)*100:.1f}%")
            print(f"      Flat rate: {flats}/{len(group)} = {flats/len(group)*100:.1f}%")
            print(f"      Avg move:  {avg_pct:+.3f}%  (median: {med_pct:+.3f}%)")
            print(f"      Avg max↑:  {avg_max_up:+.3f}%  Avg max↓: {avg_max_down:+.3f}%")

    # ═══════════════════════════════════════════════════════════
    # ANALYSIS 3: Conflict pattern breakdown
    # ═══════════════════════════════════════════════════════════
    print(f"\n{'═'*90}")
    print(f"  CONFLICT PATTERN BREAKDOWN")
    print(f"{'═'*90}")

    # Which conflict types appear most?
    conflict_type_counts = defaultdict(list)
    for r in conflict_events:
        for c in r['conflicts']:
            ctype = c.split('(')[0]
            conflict_type_counts[ctype].append(r)

    for ctype, events in sorted(conflict_type_counts.items(), key=lambda x: -len(x[1])):
        valid_24h = [e for e in events if e['outcomes'].get('24h') is not None]
        if not valid_24h:
            continue
        avg_24h = np.mean([e['outcomes']['24h']['pct'] for e in valid_24h])
        longs = [e for e in valid_24h if e['direction'] == 'LONG']
        shorts = [e for e in valid_24h if e['direction'] == 'SHORT']
        long_wins = sum(1 for e in longs if e['outcome_class']['24h'] == 'WIN')
        short_wins = sum(1 for e in shorts if e['outcome_class']['24h'] == 'WIN')
        print(f"\n  {ctype}: {len(events)} occurrences")
        print(f"    24h avg move: {avg_24h:+.3f}%")
        if longs:
            print(f"    LONG-directed: {len(longs)} events, win@24h={long_wins}/{len(longs)} ({long_wins/len(longs)*100:.0f}%)")
        if shorts:
            print(f"    SHORT-directed: {len(shorts)} events, win@24h={short_wins}/{len(shorts)} ({short_wins/len(shorts)*100:.0f}%)")

    # ═══════════════════════════════════════════════════════════
    # ANALYSIS 4: Conflict severity vs outcome
    # ═══════════════════════════════════════════════════════════
    print(f"\n{'═'*90}")
    print(f"  CONFLICT SEVERITY vs OUTCOME (24h)")
    print(f"{'═'*90}")

    for severity in sorted(set(r['conflict_score'] for r in conflict_events)):
        group = [r for r in conflict_events if r['conflict_score'] == severity]
        valid = [r for r in group if r['outcomes'].get('24h') is not None]
        if not valid:
            continue
        pcts = [r['outcomes']['24h']['pct'] for r in valid]
        avg = np.mean(pcts)
        med = np.median(pcts)
        wins = sum(1 for r in valid if r['outcome_class']['24h'] == 'WIN')
        print(f"\n  Severity {severity} (n={len(valid)}):")
        print(f"    Avg 24h: {avg:+.3f}%  Median: {med:+.3f}%  Win rate: {wins}/{len(valid)} ({wins/len(valid)*100:.0f}%)")

    # ═══════════════════════════════════════════════════════════
    # ANALYSIS 5: Phase0 death zone analysis
    # ═══════════════════════════════════════════════════════════
    print(f"\n{'═'*90}")
    print(f"  PHASE0 DEATH ZONE (<0.15) vs OUTCOME")
    print(f"{'═'*90}")

    low_phase0 = [r for r in all_results if r['phase0'] is not None and r['phase0'] < 0.15]
    high_phase0 = [r for r in all_results if r['phase0'] is not None and r['phase0'] >= 0.15]
    print(f"\n  Low Phase0 (<0.15): {len(low_phase0)} bars")
    print(f"  High Phase0 (≥0.15): {len(high_phase0)} bars")

    for label, group in [('LOW Phase0', low_phase0), ('HIGH Phase0', high_phase0)]:
        valid = [r for r in group if r['outcomes'].get('24h') is not None]
        if not valid:
            continue
        longs = [r for r in valid if r['direction'] == 'LONG']
        shorts = [r for r in valid if r['direction'] == 'SHORT']
        avg_24h = np.mean([r['outcomes']['24h']['pct'] for r in valid])
        long_wins = sum(1 for r in longs if r['outcome_class']['24h'] == 'WIN')
        short_wins = sum(1 for r in shorts if r['outcome_class']['24h'] == 'WIN')
        print(f"\n  {label}: avg 24h = {avg_24h:+.3f}%")
        if longs:
            print(f"    LONG: {len(longs)} events, win@24h={long_wins}/{len(longs)} ({long_wins/len(longs)*100:.0f}%)")
        if shorts:
            print(f"    SHORT: {len(shorts)} events, win@24h={short_wins}/{len(shorts)} ({short_wins/len(shorts)*100:.0f}%)")

    # ═══════════════════════════════════════════════════════════
    # ANALYSIS 6: Overall direction resolver accuracy
    # ═══════════════════════════════════════════════════════════
    print(f"\n{'═'*90}")
    print(f"  DIRECTION RESOLVER ACCURACY (all bars)")
    print(f"{'═'*90}")

    for horizon in ['4h', '12h', '24h', '48h', '72h']:
        valid = [r for r in all_results if r['direction'] != 'NEUTRAL' and r['outcomes'].get(horizon) is not None]
        if not valid:
            continue
        wins = sum(1 for r in valid if r['outcome_class'][horizon] == 'WIN')
        losses = sum(1 for r in valid if r['outcome_class'][horizon] == 'LOSS')
        flats = sum(1 for r in valid if r['outcome_class'][horizon] == 'FLAT')
        print(f"  {horizon}: Win={wins} Loss={losses} Flat={flats}  WR={wins/len(valid)*100:.1f}%  (n={len(valid)})")

    # ═══════════════════════════════════════════════════════════
    # ANALYSIS 7: Daily breakdown
    # ═══════════════════════════════════════════════════════════
    print(f"\n{'═'*90}")
    print(f"  DAILY BREAKDOWN")
    print(f"{'═'*90}")

    dates = sorted(set(r['timestamp'][:10] for r in all_results))
    for d in dates:
        day = [r for r in all_results if r['timestamp'][:10] == d]
        day_conflicts = [r for r in day if r['conflict_score'] >= 2]
        prices = [r['price'] for r in day]
        directions = [r['direction'] for r in day]
        longs = directions.count('LONG')
        shorts = directions.count('SHORT')
        neutrals = directions.count('NEUTRAL')
        avg_ics = np.mean([r['ics'] for r in day])
        avg_phase0 = np.mean([r['phase0'] for r in day if r['phase0'] is not None])

        # Get 24h outcome for day's midpoint
        mid = len(day) // 2
        outcome_str = ''
        if day[mid]['outcomes'].get('24h'):
            o = day[mid]['outcomes']['24h']
            outcome_str = f"24h@mid={o['pct']:+.2f}%"

        print(f"\n  {d}:  ${min(prices):.0f}–${max(prices):.0f}  L={longs} S={shorts} N={neutrals}  "
              f"ICS={avg_ics:.3f}  Phase0={avg_phase0:.3f}  Conflicts={len(day_conflicts)}  {outcome_str}")

        # Show worst conflicts of the day
        day_conflicts.sort(key=lambda r: -r['conflict_score'])
        for c in day_conflicts[:3]:
            o24 = c['outcomes'].get('24h')
            o_str = f"24h={o24['pct']:+.2f}%" if o24 else "24h=N/A"
            print(f"    ⚠️  {c['timestamp'][-8:]}  ${c['price']:.2f}  {c['direction']}  "
                  f"conf={c['conflict_score']}  ICS={c['ics']:.3f}  "
                  f"conflicts={','.join(c['conflicts'])}  → {o_str}")

    # ═══════════════════════════════════════════════════════════
    # ANALYSIS 8: Current conflict comparison
    # ═══════════════════════════════════════════════════════════
    print(f"\n{'═'*90}")
    print(f"  CURRENT CONFLICT (2026-05-12) vs HISTORICAL")
    print(f"{'═'*90}")

    # Current: LONG, Phase0=0.033, conflict_score likely 3-4
    current_similar = [r for r in conflict_events
                       if r['direction'] == 'LONG'
                       and r['phase0'] is not None and r['phase0'] < 0.15
                       and r['conflict_score'] >= 2]
    print(f"\n  Historical matches: LONG + Phase0<0.15 + ≥2 conflicts = {len(current_similar)} events")
    if current_similar:
        for horizon in ['4h', '12h', '24h', '48h', '72h']:
            valid = [r for r in current_similar if r['outcomes'].get(horizon) is not None]
            if not valid:
                continue
            pcts = [r['outcomes'][horizon]['pct'] for r in valid]
            wins = sum(1 for r in valid if r['outcome_class'][horizon] == 'WIN')
            avg = np.mean(pcts)
            med = np.median(pcts)
            print(f"    {horizon}: avg={avg:+.3f}%  med={med:+.3f}%  WR={wins}/{len(valid)} ({wins/len(valid)*100:.0f}%)")

        print(f"\n  Individual events:")
        for r in current_similar:
            o24 = r['outcomes'].get('24h')
            o72 = r['outcomes'].get('72h')
            s24 = f"24h={o24['pct']:+.2f}%" if o24 else "24h=N/A"
            s72 = f"72h={o72['pct']:+.2f}%" if o72 else "72h=N/A"
            print(f"    {r['timestamp']}  ${r['price']:.2f}  ICS={r['ics']:.3f}  "
                  f"conf={r['conflict_score']}  {','.join(r['conflicts'][:3])}  → {s24}  {s72}")


if __name__ == '__main__':
    main()

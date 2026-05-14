#!/usr/bin/env python3
"""
Deep-dive: When resolver says LONG but conflicts exist, what actually happens?
Also: reverse-contrarian analysis — does price move OPPOSITE to resolver during conflicts?
"""
import sys, os, json
import numpy as np
import pandas as pd

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

def get_idx(df, target_idx, fallback):
    try:
        return min(target_idx, len(df) - 1)
    except:
        return fallback

def scan_bar(df_15m, df_1h, df_2h, df_4h, df_1d, idx, cfg):
    row = df_15m.iloc[idx]
    price = float(row['Close'])
    ts = str(row['Open time'])

    idx_1h = get_idx(df_1h, df_1h.index.searchsorted(df_15m.index[idx]) if hasattr(df_15m.index[idx], 'value') else len(df_1h)-1, len(df_1h)-1)
    idx_2h = get_idx(df_2h, df_2h.index.searchsorted(df_15m.index[idx]) if hasattr(df_15m.index[idx], 'value') else len(df_2h)-1, len(df_2h)-1)
    idx_4h = get_idx(df_4h, df_4h.index.searchsorted(df_15m.index[idx]) if hasattr(df_15m.index[idx], 'value') else len(df_4h)-1, len(df_4h)-1)
    idx_1d = get_idx(df_1d, df_1d.index.searchsorted(df_15m.index[idx]) if hasattr(df_15m.index[idx], 'value') else len(df_1d)-1, len(df_1d)-1)

    swing_bias = df_1d['swing_bias'].iloc[idx_1d] if 'swing_bias' in df_1d.columns else 'N/A'
    phase0_val = df_1d['phase0'].iloc[idx_1d] if 'phase0' in df_1d.columns else None
    trend_dir = df_1d['trend'].iloc[idx_1d] if 'trend' in df_1d.columns else 'N/A'

    regime_state = RegimeState(config=cfg)
    vol_regime, m9_raw, _ = compute_vol_regime(df_15m, df_1h, idx, idx_1h, regime_state=regime_state, config=cfg)
    m13_status, m13_score_raw, m13_details = score_m13(df_1h, idx_1h, 'NEUTRAL', df_15m, idx)
    m13_bias = m13_details.get('m13_bias', 'NEUTRAL')
    direction, _, dir_details = resolve_direction(
        vol_regime, m9_raw if m9_raw else 0.5, m13_bias, m13_score_raw, m13_details,
        swing_bias_1d=swing_bias, trend_dir=trend_dir, config=cfg)

    m1_dir, m1_score, _ = score_m1(df_1h, idx_1h, cfg, df_15m=df_15m, idx_15m=idx)
    if m1_dir == 'BEARISH' and direction == 'LONG': m1_score = 1.0 - m1_score
    elif m1_dir == 'BULLISH' and direction == 'SHORT': m1_score = 1.0 - m1_score
    m2_status, m2_score = score_m2(df_1h, df_2h, df_4h, df_1d, idx_1h, idx_2h, idx_4h, idx_1d)
    m3_status, m3_score, _ = score_m3(df_15m, idx, direction, cfg)
    m4_status, m4_score, m4_div = score_m4(df_15m, df_2h, idx, idx_2h, direction, cfg)
    m4_div_str = m4_div.get('layer_a_div', 'NONE') if isinstance(m4_div, dict) else 'NONE'
    # v7.2: Normalize _BASE variants
    if m4_div_str.endswith('_BASE'):
        m4_div_str = m4_div_str.replace('_BASE', '')
    m5_status, m5_score, _ = score_m5(df_15m, idx, direction, cfg, n_bins=cfg['M5_VP_BINS'], lookback=cfg['M5_VP_LOOKBACK'])
    m9_status, m9_score, _ = score_vol_regime(vol_regime, m9_raw, direction, trend_dir)
    ics, _ = calc_ics(m1_score, m2_score, m3_score, m4_score, m4_status, m5_score, m9_score=m9_score, use_m9=True, config=cfg)

    conflicts = []
    if (m1_dir == 'BEARISH' and direction == 'LONG') or (m1_dir == 'BULLISH' and direction == 'SHORT'):
        conflicts.append('M1_vs_DIR')
    if phase0_val is not None and not pd.isna(phase0_val) and phase0_val < 0.15:
        conflicts.append('PHASE0_LOW')
    if (direction == 'LONG' and swing_bias == 'BEARISH') or (direction == 'SHORT' and swing_bias == 'BULLISH'):
        conflicts.append('DIR_vs_BIAS')
    if m4_div_str != 'NONE':
        if (direction == 'LONG' and 'BEARISH' in m4_div_str) or (direction == 'SHORT' and 'BULLISH' in m4_div_str):
            conflicts.append('CVD_CONFLICT')
    if ics < 0.5:
        conflicts.append('LOW_ICS')
    if (direction == 'LONG' and m13_bias == 'BEARISH') or (direction == 'SHORT' and m13_bias == 'BULLISH'):
        conflicts.append('M13_CONFLICT')

    return {
        'idx': idx, 'timestamp': ts, 'price': price, 'direction': direction,
        'swing_bias': swing_bias, 'phase0': float(phase0_val) if phase0_val is not None and not pd.isna(phase0_val) else None,
        'trend_dir': trend_dir, 'regime': vol_regime, 'm13_bias': m13_bias,
        'm1_dir': m1_dir, 'ics': round(float(ics), 4), 'conflicts': conflicts,
        'conflict_score': len(conflicts),
    }

def compute_outcomes(df_15m, idx, total_bars):
    price = float(df_15m['Close'].iloc[idx])
    outcomes = {}
    for label, bars in {'4h': 16, '12h': 48, '24h': 96, '48h': 192, '72h': 288}.items():
        fi = idx + bars
        if fi < total_bars:
            fp = float(df_15m['Close'].iloc[fi])
            wh = float(df_15m['High'].iloc[idx:fi+1].max())
            wl = float(df_15m['Low'].iloc[idx:fi+1].min())
            outcomes[label] = {
                'pct': round((fp - price) / price * 100, 3),
                'max_up': round((wh - price) / price * 100, 3),
                'max_down': round((wl - price) / price * 100, 3),
            }
        else:
            outcomes[label] = None
    return outcomes

def main():
    cfg = dict(CONFIG)
    print("Loading CSV...")
    df_all = load_data(CSV_PATH)
    print(f"  {len(df_all)} bars total")

    # Apr 1 - May 11, 2026
    mask = (df_all['Open time'] >= '2026-04-01') & (df_all['Open time'] < '2026-05-12')
    df_range = df_all[mask].copy()
    warmup = df_all[df_all['Open time'] < '2026-04-01'].tail(1000)
    df_compute = pd.concat([warmup, df_range]).reset_index(drop=True)
    print(f"  Computing on {len(df_compute)} bars...")

    df_15m, df_1h, df_2h, df_4h, df_1d = compute_all(df_compute, cfg)
    range_start = len(warmup)
    range_end = len(df_compute)
    total_bars = len(df_15m)

    # Scan every 2 bars (30min) for better resolution
    step = 2
    results = []
    for i in range(range_start, range_end, step):
        r = scan_bar(df_15m, df_1h, df_2h, df_4h, df_1d, i, cfg)
        r['outcomes'] = compute_outcomes(df_15m, i, total_bars)
        results.append(r)

    print(f"  Scanned {len(results)} bars\n")

    # ═══════════════════════════════════════════════════
    # CORE QUESTION: Resolver says LONG + conflicts → what happens?
    # ═══════════════════════════════════════════════════
    print("=" * 90)
    print("  CORE ANALYSIS: Resolver says LONG + conflict exists → actual price direction")
    print("=" * 90)

    long_conflicts = [r for r in results if r['direction'] == 'LONG' and r['conflict_score'] >= 1]
    long_clean = [r for r in results if r['direction'] == 'LONG' and r['conflict_score'] == 0]
    short_conflicts = [r for r in results if r['direction'] == 'SHORT' and r['conflict_score'] >= 1]
    short_clean = [r for r in results if r['direction'] == 'SHORT' and r['conflict_score'] == 0]

    for label, group in [
        ('LONG + conflicts', long_conflicts),
        ('LONG clean (no conflict)', long_clean),
        ('SHORT + conflicts', short_conflicts),
        ('SHORT clean (no conflict)', short_clean),
    ]:
        if not group:
            continue
        print(f"\n  {label} (n={len(group)}):")
        for h in ['4h', '12h', '24h', '48h', '72h']:
            valid = [r for r in group if r['outcomes'].get(h) is not None]
            if not valid:
                continue
            pcts = [r['outcomes'][h]['pct'] for r in valid]
            avg = np.mean(pcts)
            med = np.median(pcts)
            pos = sum(1 for p in pcts if p > 0)
            neg = sum(1 for p in pcts if p < 0)
            print(f"    {h}: avg={avg:+.3f}%  med={med:+.3f}%  +/−={pos}/{neg}  ({pos/len(valid)*100:.0f}% positive)")

    # ═══════════════════════════════════════════════════
    # CONTRARIAN ANALYSIS: Does price go OPPOSITE to resolver during conflicts?
    # ═══════════════════════════════════════════════════
    print(f"\n{'='*90}")
    print("  CONTRARIAN ANALYSIS: During conflicts, does price go OPPOSITE to resolver?")
    print("=" * 90)

    conflict_events = [r for r in results if r['conflict_score'] >= 2]
    print(f"\n  Events with ≥2 conflicts: {len(conflict_events)}")

    # For LONG-directed conflicts, check if SHORT would have been correct
    long_conf2 = [r for r in conflict_events if r['direction'] == 'LONG']
    short_conf2 = [r for r in conflict_events if r['direction'] == 'SHORT']

    for d_label, group in [('LONG-directed', long_conf2), ('SHORT-directed', short_conf2)]:
        if not group:
            continue
        print(f"\n  {d_label} conflicts:")
        for h in ['24h', '48h', '72h']:
            valid = [r for r in group if r['outcomes'].get(h) is not None]
            if not valid:
                continue
            pcts = [r['outcomes'][h]['pct'] for r in valid]
            # "Correct" = price moves in resolver direction
            if d_label == 'LONG-directed':
                correct = sum(1 for p in pcts if p > 0.5)
                wrong = sum(1 for p in pcts if p < -0.5)
                contrarian_would_win = wrong  # SHORT would have won
            else:
                correct = sum(1 for p in pcts if p < -0.5)
                wrong = sum(1 for p in pcts if p > 0.5)
                contrarian_would_win = wrong

            avg = np.mean(pcts)
            print(f"    {h}: avg_move={avg:+.3f}%  resolver_correct={correct}/{len(valid)} ({correct/len(valid)*100:.0f}%)  "
                  f"contrarian_would_win={contrarian_would_win}/{len(valid)} ({contrarian_would_win/len(valid)*100:.0f}%)")

    # ═══════════════════════════════════════════════════
    # BREAKDOWN BY NUMBER OF CONFLICTS
    # ═══════════════════════════════════════════════════
    print(f"\n{'='*90}")
    print("  BREAKDOWN BY CONFLICT COUNT (resolver direction = actual outcome?)")
    print("=" * 90)

    for n_conf in sorted(set(r['conflict_score'] for r in results)):
        group = [r for r in results if r['conflict_score'] == n_conf and r['direction'] != 'NEUTRAL']
        if not group:
            continue
        valid_24h = [r for r in group if r['outcomes'].get('24h') is not None]
        if not valid_24h:
            continue

        pcts_24h = [r['outcomes']['24h']['pct'] for r in valid_24h]
        avg_24h = np.mean(pcts_24h)

        # Resolver direction win rate
        resolver_correct = 0
        for r in valid_24h:
            p = r['outcomes']['24h']['pct']
            if r['direction'] == 'LONG' and p > 0.5:
                resolver_correct += 1
            elif r['direction'] == 'SHORT' and p < -0.5:
                resolver_correct += 1

        longs = sum(1 for r in valid_24h if r['direction'] == 'LONG')
        shorts = sum(1 for r in valid_24h if r['direction'] == 'SHORT')

        print(f"  {n_conf} conflicts (n={len(valid_24h)}): avg_24h={avg_24h:+.3f}%  "
              f"resolver_WR={resolver_correct}/{len(valid_24h)} ({resolver_correct/len(valid_24h)*100:.0f}%)  "
              f"L={longs} S={shorts}")

    # ═══════════════════════════════════════════════════
    # SPECIFIC PATTERN: LONG + DIR_vs_BIAS (daily bias bullish but resolver says short)
    # This is the most common pattern in Apr-May
    # ═══════════════════════════════════════════════════
    print(f"\n{'='*90}")
    print("  KEY PATTERN: Daily bias BULLISH + resolver says SHORT → what happens?")
    print("  (This is the #1 conflict type in Apr-May 2026)")
    print("=" * 90)

    short_vs_bullish = [r for r in results
                        if r['direction'] == 'SHORT'
                        and r['swing_bias'] == 'BULLISH'
                        and r['conflict_score'] >= 2]
    print(f"\n  Events: {len(short_vs_bullish)}")

    for h in ['4h', '12h', '24h', '48h', '72h']:
        valid = [r for r in short_vs_bullish if r['outcomes'].get(h) is not None]
        if not valid:
            continue
        pcts = [r['outcomes'][h]['pct'] for r in valid]
        # Price going UP = wrong for SHORT = bullish bias was right
        up = sum(1 for p in pcts if p > 0.5)
        down = sum(1 for p in pcts if p < -0.5)
        avg = np.mean(pcts)
        print(f"    {h}: avg={avg:+.3f}%  price_up={up}/{len(valid)} ({up/len(valid)*100:.0f}%)  "
              f"price_down={down}/{len(valid)} ({down/len(valid)*100:.0f}%)")

    # Now the reverse: LONG when daily bias is BEARISH
    long_vs_bearish = [r for r in results
                       if r['direction'] == 'LONG'
                       and r['swing_bias'] == 'BEARISH'
                       and r['conflict_score'] >= 2]
    print(f"\n  REVERSE: Daily BEARISH + resolver says LONG (n={len(long_vs_bearish)}):")
    for h in ['24h', '48h', '72h']:
        valid = [r for r in long_vs_bearish if r['outcomes'].get(h) is not None]
        if not valid:
            continue
        pcts = [r['outcomes'][h]['pct'] for r in valid]
        up = sum(1 for p in pcts if p > 0.5)
        down = sum(1 for p in pcts if p < -0.5)
        avg = np.mean(pcts)
        print(f"    {h}: avg={avg:+.3f}%  price_up={up}/{len(valid)} ({up/len(valid)*100:.0f}%)  "
              f"price_down={down}/{len(valid)} ({down/len(valid)*100:.0f}%)")

    # ═══════════════════════════════════════════════════
    # ICS THRESHOLD ANALYSIS
    # ═══════════════════════════════════════════════════
    print(f"\n{'='*90}")
    print("  ICS THRESHOLD: What ICS level actually predicts correct direction?")
    print("=" * 90)

    for lo, hi in [(0.0, 0.3), (0.3, 0.4), (0.4, 0.5), (0.5, 0.6), (0.6, 0.7), (0.7, 1.0)]:
        group = [r for r in results if lo <= r['ics'] < hi and r['direction'] != 'NEUTRAL']
        if not group:
            continue
        valid_24h = [r for r in group if r['outcomes'].get('24h') is not None]
        if not valid_24h:
            continue
        correct = 0
        for r in valid_24h:
            p = r['outcomes']['24h']['pct']
            if r['direction'] == 'LONG' and p > 0.5: correct += 1
            elif r['direction'] == 'SHORT' and p < -0.5: correct += 1
        avg = np.mean([r['outcomes']['24h']['pct'] for r in valid_24h])
        print(f"  ICS [{lo:.1f}, {hi:.1f}): n={len(valid_24h)}  avg_24h={avg:+.3f}%  "
              f"WR={correct}/{len(valid_24h)} ({correct/len(valid_24h)*100:.0f}%)")

    # ═══════════════════════════════════════════════════
    # FINAL VERDICT: Pattern matching for TODAY
    # ═══════════════════════════════════════════════════
    print(f"\n{'='*90}")
    print("  FINAL: Closest historical matches to TODAY's setup")
    print("  Today: LONG, daily BULLISH, ICS≈0.49, CVD conflict, LOW_ICS, M1 bearish")
    print("=" * 90)

    # Find bars most similar to today's setup
    today_matches = [r for r in results
                     if r['direction'] == 'LONG'
                     and r['swing_bias'] == 'BULLISH'
                     and 0.40 <= r['ics'] <= 0.55
                     and r['conflict_score'] >= 2
                     and 'CVD_CONFLICT' in r['conflicts']]
    print(f"\n  Matches (LONG + BULLISH bias + ICS 0.40-0.55 + CVD conflict): {len(today_matches)}")

    if today_matches:
        for h in ['4h', '12h', '24h', '48h', '72h']:
            valid = [r for r in today_matches if r['outcomes'].get(h) is not None]
            if not valid:
                continue
            pcts = [r['outcomes'][h]['pct'] for r in valid]
            up = sum(1 for p in pcts if p > 0.5)
            down = sum(1 for p in pcts if p < -0.5)
            avg = np.mean(pcts)
            med = np.median(pcts)
            print(f"    {h}: avg={avg:+.3f}%  med={med:+.3f}%  up={up}/{len(valid)} ({up/len(valid)*100:.0f}%)  "
                  f"down={down}/{len(valid)} ({down/len(valid)*100:.0f}%)")

        print(f"\n  Individual matching events:")
        for r in today_matches:
            o24 = r['outcomes'].get('24h')
            o72 = r['outcomes'].get('72h')
            s24 = f"24h={o24['pct']:+.2f}%" if o24 else "24h=N/A"
            s72 = f"72h={o72['pct']:+.2f}%" if o72 else "72h=N/A"
            outcome_label = ''
            if o24:
                if o24['pct'] > 0.5: outcome_label = '✅ LONG correct'
                elif o24['pct'] < -0.5: outcome_label = '❌ LONG wrong'
                else: outcome_label = '➡️ FLAT'
            print(f"    {r['timestamp']}  ${r['price']:.2f}  ICS={r['ics']:.3f}  "
                  f"conf={r['conflict_score']}  {s24}  {s72}  {outcome_label}")


if __name__ == '__main__':
    main()

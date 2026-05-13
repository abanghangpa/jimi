#!/usr/bin/env python3
"""
Deep dive: cascade cluster + hybrid strategies + parameter sensitivity.

Focuses on:
1. Cascade at different distance bands
2. Cascade + ATR floor hybrid
3. ATR multiplier sweep (0.8 to 2.5)
4. Taker flip with different thresholds
5. "Cascade-aware unswept pool" — take nearest unswept, but skip if cascade score < threshold
"""

import sys, os, argparse
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from src.utils.data_handler import load_data, resample_ohlcv
from src.utils.indicators import calc_atr
from src.modules.m5_liquidation import build_volume_profile, find_magnets


def compute_cascade_score(price, direction, highs, lows, closes, volumes, takers):
    """Fast cascade potential estimate."""
    n = len(highs)
    lookback = min(48, n)
    if lookback < 8:
        return 0.0

    h = highs[-lookback:]
    l = lows[-lookback:]
    c = closes[-lookback:]
    v = volumes[-lookback:]
    t = takers[-lookback:]

    threshold = price * 0.003

    if direction == 'LONG':
        stops = sum(1 for x in h if abs(x - price) < threshold)
    else:
        stops = sum(1 for x in l if abs(x - price) < threshold)
    stop_density = stops / lookback

    momentum = (c[-1] - c[0]) / c[0] * 100 if c[0] > 0 else 0
    momentum_ok = (momentum > 0 and direction == 'LONG') or (momentum < 0 and direction == 'SHORT')

    vol_recent = np.mean(v[-8:])
    vol_prior = np.mean(v[-24:-8]) if lookback > 24 else vol_recent
    vol_trend = vol_recent / vol_prior if vol_prior > 0 else 1.0

    taker_ratio = np.sum(t) / np.sum(v) if np.sum(v) > 0 else 0.5
    taker_extreme = abs(taker_ratio - 0.5) * 2

    score = (stop_density * 0.30 + (1.0 if momentum_ok else 0.0) * 0.25 +
             min(vol_trend / 2.0, 1.0) * 0.25 + taker_extreme * 0.20)
    return min(1.0, score)


def sim(df, entry_idx, entry_price, tp, sl, direction, max_bars=80):
    n = len(df)
    end = min(entry_idx + max_bars, n)
    for i in range(entry_idx + 1, end):
        h = float(df['High'].values[i])
        l = float(df['Low'].values[i])
        if direction == 'LONG':
            if l <= sl: return {'o': 'SL', 'b': i-entry_idx, 'p': (sl-entry_price)/entry_price*100}
            if h >= tp: return {'o': 'TP', 'b': i-entry_idx, 'p': (tp-entry_price)/entry_price*100}
        else:
            if h >= sl: return {'o': 'SL', 'b': i-entry_idx, 'p': (entry_price-sl)/entry_price*100}
            if l <= tp: return {'o': 'TP', 'b': i-entry_idx, 'p': (entry_price-tp)/entry_price*100}
    c = float(df['Close'].values[min(entry_idx+max_bars, n-1)])
    pnl = (c-entry_price)/entry_price*100 if direction=='LONG' else (entry_price-c)/entry_price*100
    return {'o': 'TO', 'b': max_bars, 'p': pnl}


def stats(trades):
    if not trades: return {}
    tp_n = sum(1 for t in trades if t['o']=='TP')
    sl_n = sum(1 for t in trades if t['o']=='SL')
    n = len(trades)
    wr = tp_n/n*100
    avg = np.mean([t['p'] for t in trades])
    avg_tp = np.mean([t['p'] for t in trades if t['o']=='TP']) if tp_n else 0
    avg_sl = np.mean([t['p'] for t in trades if t['o']=='SL']) if sl_n else 0
    exp = (wr/100*avg_tp + (1-wr/100)*avg_sl) if tp_n and sl_n else 0
    gp = sum(t['p'] for t in trades if t['p']>0)
    gl = abs(sum(t['p'] for t in trades if t['p']<0))
    pf = gp/gl if gl>0 else float('inf')
    return {'n':n,'tp':tp_n,'sl':sl_n,'wr':wr,'avg':avg,'exp':exp,'pf':pf,
            'avg_tp':avg_tp,'avg_sl':avg_sl,'bars':np.mean([t['b'] for t in trades])}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--bars', type=int, default=0)
    parser.add_argument('--step', type=int, default=12)
    parser.add_argument('--max-bars', type=int, default=80)
    args = parser.parse_args()

    csv = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'eth_15m_merged.csv')
    print(f"Loading...")
    df = load_data(csv)
    if args.bars > 0: df = df.iloc[-args.bars:]
    df = df.reset_index(drop=True)
    n = len(df)
    print(f"  {n} bars")

    df['atr'] = calc_atr(df['High'], df['Low'], df['Close'], 14)
    df['taker_ratio'] = df['Taker buy base asset volume'] / df['Volume'].replace(0, np.nan)
    df['vol_ma20'] = df['Volume'].rolling(20).mean()

    warmup = 500
    highs = df['High'].values.astype(float)
    lows = df['Low'].values.astype(float)
    closes = df['Close'].values.astype(float)
    volumes = df['Volume'].values.astype(float)
    takers_full = df['Taker buy base asset volume'].values.astype(float)
    atrs = df['atr'].values.astype(float)

    # ═══════════════════════════════════════════
    # EXPERIMENT 1: ATR multiplier sweep
    # ═══════════════════════════════════════════
    print("\n" + "═" * 60)
    print("  EXPERIMENT 1: ATR Multiplier Sweep")
    print("═" * 60)

    atr_mults = [0.6, 0.8, 1.0, 1.2, 1.5, 1.8, 2.0, 2.5, 3.0]
    print(f"\n  {'ATR':>5} {'Trades':>6} {'Win%':>6} {'Exp%':>8} {'PF':>6} {'AvgPnL':>8} {'AvgTP':>8} {'AvgSL':>8} {'Bars':>5}")
    print(f"  {'─'*60}")

    for mult in atr_mults:
        trades = []
        for idx in range(warmup, n - args.max_bars, args.step):
            p = float(closes[idx])
            atr = float(atrs[idx])
            if np.isnan(atr) or atr <= 0: atr = p * 0.01
            d = 'LONG' if (idx // args.step) % 2 == 0 else 'SHORT'
            sl = p - atr if d == 'LONG' else p + atr
            tp = p + atr * mult if d == 'LONG' else p - atr * mult
            trades.append(sim(df, idx, p, tp, sl, d, args.max_bars))
        s = stats(trades)
        print(f"  {mult:>4.1f}x {s['n']:>6} {s['wr']:>5.1f}% {s['exp']:>+7.4f}% {s['pf']:>6.2f} "
              f"{s['avg']:>+7.3f}% {s['avg_tp']:>+7.3f}% {s['avg_sl']:>+7.3f}% {s['bars']:>5.1f}")

    # ═══════════════════════════════════════════
    # EXPERIMENT 2: Cascade score threshold sweep
    # ═══════════════════════════════════════════
    print("\n" + "═" * 60)
    print("  EXPERIMENT 2: Cascade Score Threshold Sweep")
    print("  (only take cascade TP if score >= threshold)")
    print("═" * 60)

    thresholds = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7]
    vp_interval = 500

    print(f"\n  {'Thresh':>6} {'Trades':>6} {'Win%':>6} {'Exp%':>8} {'PF':>6} {'AvgPnL':>8}")
    print(f"  {'─'*50}")

    for thresh in thresholds:
        trades = []
        for idx in range(warmup, n - args.max_bars, args.step):
            p = float(closes[idx])
            atr = float(atrs[idx])
            if np.isnan(atr) or atr <= 0: atr = p * 0.01
            d = 'LONG' if (idx // args.step) % 2 == 0 else 'SHORT'
            sl = p - atr if d == 'LONG' else p + atr

            # Compute cascade for nearest level
            vp_start = max(0, idx - 500)
            bc, vp, _ = build_volume_profile(highs[vp_start:idx+1], lows[vp_start:idx+1],
                                              closes[vp_start:idx+1], volumes[vp_start:idx+1],
                                              n_bins=50, lookback=500)
            magnets = find_magnets(bc, vp) if bc is not None else []

            best_tp = None
            best_score = -1
            for m in magnets:
                mp = m[0]
                if d == 'LONG' and mp <= p: continue
                if d == 'SHORT' and mp >= p: continue
                cs = compute_cascade_score(mp, d, highs[max(0,idx-48):idx+1],
                                           lows[max(0,idx-48):idx+1],
                                           closes[max(0,idx-48):idx+1],
                                           volumes[max(0,idx-48):idx+1],
                                           takers_full[max(0,idx-48):idx+1])
                if cs >= thresh and cs > best_score:
                    best_score = cs
                    best_tp = mp

            if best_tp is None:
                # ATR fallback
                best_tp = p + atr * 1.2 if d == 'LONG' else p - atr * 1.2

            trades.append(sim(df, idx, p, best_tp, sl, d, args.max_bars))

        s = stats(trades)
        print(f"  {thresh:>5.1f}  {s['n']:>6} {s['wr']:>5.1f}% {s['exp']:>+7.4f}% {s['pf']:>6.2f} {s['avg']:>+7.3f}%")

    # ═══════════════════════════════════════════
    # EXPERIMENT 3: Cascade + ATR floor hybrid
    # ═══════════════════════════════════════════
    print("\n" + "═" * 60)
    print("  EXPERIMENT 3: Cascade + ATR Floor Hybrid")
    print("  (TP = max(cascade_target, entry + ATR * floor_mult))")
    print("═" * 60)

    floor_mults = [0.3, 0.5, 0.7, 0.9, 1.0, 1.2]
    print(f"\n  {'Floor':>6} {'Trades':>6} {'Win%':>6} {'Exp%':>8} {'PF':>6} {'AvgPnL':>8}")
    print(f"  {'─'*50}")

    for floor_mult in floor_mults:
        trades = []
        for idx in range(warmup, n - args.max_bars, args.step):
            p = float(closes[idx])
            atr = float(atrs[idx])
            if np.isnan(atr) or atr <= 0: atr = p * 0.01
            d = 'LONG' if (idx // args.step) % 2 == 0 else 'SHORT'
            sl = p - atr if d == 'LONG' else p + atr

            # Cascade TP
            vp_start = max(0, idx - 500)
            bc, vp, _ = build_volume_profile(highs[vp_start:idx+1], lows[vp_start:idx+1],
                                              closes[vp_start:idx+1], volumes[vp_start:idx+1],
                                              n_bins=50, lookback=500)
            magnets = find_magnets(bc, vp) if bc is not None else []

            cascade_tp = None
            for m in magnets:
                mp = m[0]
                if d == 'LONG' and mp > p:
                    cascade_tp = mp
                    break
                elif d == 'SHORT' and mp < p:
                    cascade_tp = mp
                    break

            atr_tp = p + atr * floor_mult if d == 'LONG' else p - atr * floor_mult

            if cascade_tp is not None:
                if d == 'LONG':
                    tp = max(cascade_tp, atr_tp)  # wider of the two
                else:
                    tp = min(cascade_tp, atr_tp)
            else:
                tp = atr_tp

            trades.append(sim(df, idx, p, tp, sl, d, args.max_bars))

        s = stats(trades)
        print(f"  {floor_mult:>5.1f}x {s['n']:>6} {s['wr']:>5.1f}% {s['exp']:>+7.4f}% {s['pf']:>6.2f} {s['avg']:>+7.3f}%")

    # ═══════════════════════════════════════════
    # EXPERIMENT 4: Cascade distance band analysis
    # ═══════════════════════════════════════════
    print("\n" + "═" * 60)
    print("  EXPERIMENT 4: Cascade TP by Distance Band")
    print("  (how does cascade perform at different distances?)")
    print("═" * 60)

    bands = [(0, 0.3), (0.3, 0.6), (0.6, 1.0), (1.0, 1.5), (1.5, 3.0), (3.0, 10.0)]
    band_trades = {b: [] for b in bands}

    for idx in range(warmup, n - args.max_bars, args.step):
        p = float(closes[idx])
        atr = float(atrs[idx])
        if np.isnan(atr) or atr <= 0: atr = p * 0.01
        d = 'LONG' if (idx // args.step) % 2 == 0 else 'SHORT'
        sl = p - atr if d == 'LONG' else p + atr

        vp_start = max(0, idx - 500)
        bc, vp, _ = build_volume_profile(highs[vp_start:idx+1], lows[vp_start:idx+1],
                                          closes[vp_start:idx+1], volumes[vp_start:idx+1],
                                          n_bins=50, lookback=500)
        magnets = find_magnets(bc, vp) if bc is not None else []

        for m in magnets:
            mp = m[0]
            if d == 'LONG' and mp <= p: continue
            if d == 'SHORT' and mp >= p: continue
            dist_pct = abs(mp - p) / p * 100
            for band in bands:
                if band[0] <= dist_pct < band[1]:
                    result = sim(df, idx, p, mp, sl, d, args.max_bars)
                    result['dist'] = dist_pct
                    band_trades[band].append(result)
                    break
            break  # only nearest magnet

    print(f"\n  {'Band%':>10} {'Trades':>6} {'Win%':>6} {'Exp%':>8} {'PF':>6} {'AvgPnL':>8} {'AvgDist':>8}")
    print(f"  {'─'*55}")
    for band in bands:
        t = band_trades[band]
        if not t:
            print(f"  {band[0]:>4.1f}-{band[1]:<4.1f}  {'—':>6}")
            continue
        s = stats(t)
        avg_d = np.mean([x['dist'] for x in t])
        print(f"  {band[0]:>4.1f}-{band[1]:<4.1f}  {s['n']:>6} {s['wr']:>5.1f}% {s['exp']:>+7.4f}% {s['pf']:>6.2f} {s['avg']:>+7.3f}% {avg_d:>7.2f}%")

    # ═══════════════════════════════════════════
    # EXPERIMENT 5: Taker flip threshold sweep
    # ═══════════════════════════════════════════
    print("\n" + "═" * 60)
    print("  EXPERIMENT 5: Taker Flip Threshold Sweep")
    print("═" * 60)

    thresholds_tf = [0.52, 0.55, 0.58, 0.60, 0.65, 0.70]
    lookforwards = [8, 16, 24, 32]

    print(f"\n  {'Thresh':>6} {'Fwd':>4} {'Trades':>6} {'Win%':>6} {'Exp%':>8} {'PF':>6} {'AvgPnL':>8}")
    print(f"  {'─'*55}")

    for tf_thresh in thresholds_tf:
        for lf in lookforwards:
            trades = []
            for idx in range(warmup, n - args.max_bars, args.step):
                p = float(closes[idx])
                atr = float(atrs[idx])
                if np.isnan(atr) or atr <= 0: atr = p * 0.01
                d = 'LONG' if (idx // args.step) % 2 == 0 else 'SHORT'
                sl = p - atr if d == 'LONG' else p + atr

                # Find taker flip
                end_lf = min(idx + lf, n)
                tp = None
                for i in range(idx+1, end_lf):
                    tv = takers_full[i]
                    vv = volumes[i]
                    if vv > 0:
                        ratio = tv / vv
                    else:
                        continue
                    if d == 'LONG' and ratio < (1 - tf_thresh):
                        tp = float(closes[i])
                        break
                    elif d == 'SHORT' and ratio > tf_thresh:
                        tp = float(closes[i])
                        break

                if tp is None:
                    tp = p + atr * 1.2 if d == 'LONG' else p - atr * 1.2

                trades.append(sim(df, idx, p, tp, sl, d, args.max_bars))

            s = stats(trades)
            print(f"  {tf_thresh:>5.2f}  {lf:>4} {s['n']:>6} {s['wr']:>5.1f}% {s['exp']:>+7.4f}% {s['pf']:>6.2f} {s['avg']:>+7.3f}%")

    # ═══════════════════════════════════════════
    # EXPERIMENT 6: Best combo — Cascade + ATR 1.5x + Taker gate
    # ═══════════════════════════════════════════
    print("\n" + "═" * 60)
    print("  EXPERIMENT 6: Combined Strategy (Cascade + ATR + Taker)")
    print("  TP = cascade if cascade_score >= 0.3 AND taker agrees,")
    print("       else ATR 1.5x")
    print("═" * 60)

    trades_combined = []
    for idx in range(warmup, n - args.max_bars, args.step):
        p = float(closes[idx])
        atr = float(atrs[idx])
        if np.isnan(atr) or atr <= 0: atr = p * 0.01
        d = 'LONG' if (idx // args.step) % 2 == 0 else 'SHORT'
        sl = p - atr if d == 'LONG' else p + atr

        # Taker at entry
        taker_at_entry = takers_full[idx] / volumes[idx] if volumes[idx] > 0 else 0.5
        taker_agrees = (d == 'LONG' and taker_at_entry > 0.50) or \
                       (d == 'SHORT' and taker_at_entry < 0.50)

        # Cascade
        vp_start = max(0, idx - 500)
        bc, vp, _ = build_volume_profile(highs[vp_start:idx+1], lows[vp_start:idx+1],
                                          closes[vp_start:idx+1], volumes[vp_start:idx+1],
                                          n_bins=50, lookback=500)
        magnets = find_magnets(bc, vp) if bc is not None else []

        cascade_tp = None
        cascade_score = 0
        for m in magnets:
            mp = m[0]
            if d == 'LONG' and mp <= p: continue
            if d == 'SHORT' and mp >= p: continue
            cs = compute_cascade_score(mp, d, highs[max(0,idx-48):idx+1],
                                       lows[max(0,idx-48):idx+1],
                                       closes[max(0,idx-48):idx+1],
                                       volumes[max(0,idx-48):idx+1],
                                       takers_full[max(0,idx-48):idx+1])
            if cs > cascade_score:
                cascade_score = cs
                cascade_tp = mp

        atr_tp = p + atr * 1.5 if d == 'LONG' else p - atr * 1.5

        # Decision: cascade if score >= 0.3 AND taker agrees, else ATR
        if cascade_tp is not None and cascade_score >= 0.3 and taker_agrees:
            tp = cascade_tp
        else:
            tp = atr_tp

        trades_combined.append(sim(df, idx, p, tp, sl, d, args.max_bars))

    s = stats(trades_combined)
    print(f"\n  Combined: {s['n']} trades, WR={s['wr']:.1f}%, Exp={s['exp']:+.4f}%, PF={s['pf']:.2f}, Avg={s['avg']:+.3f}%")

    # Compare all at once
    print(f"\n  {'═'*60}")
    print(f"  FINAL COMPARISON (same dataset, step={args.step})")
    print(f"  {'═'*60}")

    # ATR 1.5x baseline
    atr_base = []
    for idx in range(warmup, n - args.max_bars, args.step):
        p = float(closes[idx])
        atr = float(atrs[idx])
        if np.isnan(atr) or atr <= 0: atr = p * 0.01
        d = 'LONG' if (idx // args.step) % 2 == 0 else 'SHORT'
        sl = p - atr if d == 'LONG' else p + atr
        tp = p + atr * 1.5 if d == 'LONG' else p - atr * 1.5
        atr_base.append(sim(df, idx, p, tp, sl, d, args.max_bars))
    sa = stats(atr_base)

    # Pure cascade
    cascade_pure = []
    for idx in range(warmup, n - args.max_bars, args.step):
        p = float(closes[idx])
        atr = float(atrs[idx])
        if np.isnan(atr) or atr <= 0: atr = p * 0.01
        d = 'LONG' if (idx // args.step) % 2 == 0 else 'SHORT'
        sl = p - atr if d == 'LONG' else p + atr
        vp_start = max(0, idx - 500)
        bc, vp, _ = build_volume_profile(highs[vp_start:idx+1], lows[vp_start:idx+1],
                                          closes[vp_start:idx+1], volumes[vp_start:idx+1],
                                          n_bins=50, lookback=500)
        magnets = find_magnets(bc, vp) if bc is not None else []
        tp = None
        for m in magnets:
            mp = m[0]
            if d == 'LONG' and mp > p: tp = mp; break
            if d == 'SHORT' and mp < p: tp = mp; break
        if tp is None: tp = p + atr * 1.5 if d == 'LONG' else p - atr * 1.5
        cascade_pure.append(sim(df, idx, p, tp, sl, d, args.max_bars))
    sc = stats(cascade_pure)

    print(f"\n  {'Strategy':<30} {'Trades':>6} {'Win%':>6} {'Exp%':>8} {'PF':>6} {'AvgPnL':>8}")
    print(f"  {'─'*60}")
    print(f"  {'ATR 1.5x (baseline)':<30} {sa['n']:>6} {sa['wr']:>5.1f}% {sa['exp']:>+7.4f}% {sa['pf']:>6.2f} {sa['avg']:>+7.3f}%")
    print(f"  {'Pure Cascade (nearest HVN)':<30} {sc['n']:>6} {sc['wr']:>5.1f}% {sc['exp']:>+7.4f}% {sc['pf']:>6.2f} {sc['avg']:>+7.3f}%")
    print(f"  {'Cascade+ATR+Taker combo':<30} {s['n']:>6} {s['wr']:>5.1f}% {s['exp']:>+7.4f}% {s['pf']:>6.2f} {s['avg']:>+7.3f}%")


if __name__ == '__main__':
    main()

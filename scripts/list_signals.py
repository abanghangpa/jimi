#!/usr/bin/env python3
import sys, os; sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
import numpy as np, pandas as pd
from src.config import CONFIG
from src.utils.data_handler import load_data
from src.utils.indicators import calc_atr, calc_rsi, calc_vol_ratio, calc_vwap
from src.modules.m4_cvd import calc_cvd_15m, detect_cvd_divergence_15m
from src.modules.m18_squeeze import detect_squeeze_v3, SQUEEZE_V3_DEFAULTS

df = load_data('eth_15m_merged.csv')
df = df[df['Open time'] >= '2026-01-01'].reset_index(drop=True)
cfg = dict(CONFIG); cfg.update(SQUEEZE_V3_DEFAULTS)
df['atr'] = calc_atr(df['High'], df['Low'], df['Close'], cfg['ATR_PERIOD'])
df['rsi'] = calc_rsi(df['Close'], 14)
df['vol_ratio'] = calc_vol_ratio(df['Volume'])
df['vwap'] = calc_vwap(df['High'], df['Low'], df['Close'], df['Volume'], cfg['VWAP_LOOKBACK'])
df['taker_ratio'] = (df['Taker buy base asset volume'] / df['Volume'].replace(0, np.nan)).fillna(0.5)
df['vol_ma20'] = df['Volume'].rolling(20).mean()
df['vol_trend'] = df['Volume'] / df['vol_ma20']
df['cvd_15m'] = calc_cvd_15m(df)
df['cvd_divergence_15m'] = detect_cvd_divergence_15m(df, cfg['CVD_LOOKBACK'], cfg['CVD_DIVERGENCE_WINDOW'])
taker_ma = df['taker_ratio'].rolling(50).mean()
taker_std = df['taker_ratio'].rolling(50).std()
df['ls_zscore'] = (df['taker_ratio'] - taker_ma) / taker_std.replace(0, 1)
atr_pctl = df['atr'].rolling(500, min_periods=100).apply(lambda x: (x.iloc[-1]-x.min())/(x.max()-x.min()) if x.max()>x.min() else 0.5, raw=False)
df['range_width'] = (df['High'].rolling(48).max()-df['Low'].rolling(48).min())/df['Close']*100
df['vwap_dist'] = (df['Close']-df['vwap'])/df['vwap']*100
vol_cumsum_48 = df['Volume'].rolling(48).sum()
vol_cumsum_ma = df['Volume'].rolling(68).mean()*20
df['oi_proxy'] = vol_cumsum_48/vol_cumsum_ma.replace(0,1)
df['bar_vol_spike'] = df['Volume']/df['vol_ma20']
bar_range = (df['High']-df['Low'])/df['Close']*100
bar_range_ma = bar_range.rolling(20).mean()
df['bar_range_expansion'] = bar_range/bar_range_ma.replace(0,1)
df['bar_taker_extreme'] = (df['taker_ratio']>0.65)|(df['taker_ratio']<0.35)
def cq(rw,vr,oip,vd):
    return np.clip(1-(rw-1.5)/4,0,1)*0.30+np.clip(1-(vr-0.05)/0.20,0,1)*0.25+np.clip((oip-0.7)/0.5,0,1)*0.25+np.clip(1-np.abs(vd)/1,0,1)*0.20
df['squeeze_quality'] = cq(df['range_width'].values,df['vol_ratio'].values,df['oi_proxy'].values,df['vwap_dist'].values)

MIN_BARS=500; last_signal_bar=-999; signals=[]

for idx in range(MIN_BARS, len(df)):
    ap = float(atr_pctl.iloc[idx]) if not pd.isna(atr_pctl.iloc[idx]) else 0.5
    ls_z = float(df['ls_zscore'].iloc[idx]) if not pd.isna(df['ls_zscore'].iloc[idx]) else 0
    vt = float(df['vol_trend'].iloc[idx]) if not pd.isna(df['vol_trend'].iloc[idx]) else 1.0
    sq_val = float(df['squeeze_quality'].iloc[idx]) if not pd.isna(df['squeeze_quality'].iloc[idx]) else 0.5
    bvs = float(df['bar_vol_spike'].iloc[idx]) if not pd.isna(df['bar_vol_spike'].iloc[idx]) else 1.0
    bre = float(df['bar_range_expansion'].iloc[idx]) if not pd.isna(df['bar_range_expansion'].iloc[idx]) else 1.0
    bte = bool(df['bar_taker_extreme'].iloc[idx]) if not pd.isna(df['bar_taker_extreme'].iloc[idx]) else False
    rsi_val = float(df['rsi'].iloc[idx]) if not pd.isna(df['rsi'].iloc[idx]) else 50
    oip = float(df['oi_proxy'].iloc[idx]) if not pd.isna(df['oi_proxy'].iloc[idx]) else 1.0
    rw = float(df['range_width'].iloc[idx]) if not pd.isna(df['range_width'].iloc[idx]) else 5
    vd = float(df['vwap_dist'].iloc[idx]) if not pd.isna(df['vwap_dist'].iloc[idx]) else 0
    price = float(df['Close'].iloc[idx])
    atr_val = float(df['atr'].iloc[idx]) if not pd.isna(df['atr'].iloc[idx]) else 0
    m4b_div = 'NONE'; m4b_ago = 99
    for ci in range(max(0, idx-24), idx+1):
        div = df['cvd_divergence_15m'].iloc[ci]
        if div != 'NONE': m4b_div = div; m4b_ago = idx - ci
    result = {
        'price': price, 'm9': {'regime': 'NEUTRAL', 'raw': ap},
        'derivatives': {'ls_zscore': ls_z, 'funding_rate': 0, 'oi_roc_1h': 0, 'whale_signal': 'NEUTRAL'},
        'm4b': {'divergence': m4b_div, 'bars_ago': m4b_ago, 'cvd_slope': 0},
        'rsi': rsi_val, 'vol_trend': vt, 'atr': atr_val,
        'range_width': rw, 'vol_ratio': 0.15, 'oi_proxy': oip, 'vwap_dist': vd,
        'squeeze_quality': sq_val, 'bar_vol_spike': bvs, 'bar_range_expansion': bre, 'bar_taker_extreme': bte,
    }
    sq = detect_squeeze_v3(result, config=cfg, last_signal_bar=last_signal_bar, current_bar=idx)
    ts = str(df['Open time'].iloc[idx])
    if sq['squeeze_type'] != 'NONE':
        last_signal_bar = idx
        direction = sq['direction']
        tp_price = sq['tp']; sl_price = sq['sl']
        tp_pct = sq['tp_pct']; sl_pct = sq['sl_pct']

        # Find TP/SL hit
        tp_hit_bar = None; sl_hit_bar = None
        for fi in range(idx+1, min(idx+100, len(df))):
            high = float(df['High'].iloc[fi]); low = float(df['Low'].iloc[fi])
            if direction == 'LONG':
                if high >= tp_price and tp_hit_bar is None: tp_hit_bar = fi
                if low <= sl_price and sl_hit_bar is None: sl_hit_bar = fi
            else:
                if low <= tp_price and tp_hit_bar is None: tp_hit_bar = fi
                if high >= sl_price and sl_hit_bar is None: sl_hit_bar = fi

        tp_time = str(df['Open time'].iloc[tp_hit_bar])[:16] if tp_hit_bar else None
        sl_time = str(df['Open time'].iloc[sl_hit_bar])[:16] if sl_hit_bar else None
        tp_bars = tp_hit_bar - idx if tp_hit_bar else None
        sl_bars = sl_hit_bar - idx if sl_hit_bar else None

        # 4h return
        ex4 = float(df['Close'].iloc[min(idx+32, len(df)-1)])
        ret4 = (ex4-price)/price*100 if direction=='LONG' else (price-ex4)/price*100

        signals.append({
            'ts': ts[:16], 'type': sq['squeeze_type'], 'dir': direction,
            'price': price, 'z': ls_z, 'm4b': m4b_div, 'm4b_ago': m4b_ago,
            'score': sq['squeeze_score'], 'quality': sq_val,
            'tp': tp_price, 'tp_pct': tp_pct, 'sl': sl_price, 'sl_pct': sl_pct,
            'tp_time': tp_time, 'tp_bars': tp_bars,
            'sl_time': sl_time, 'sl_bars': sl_bars,
            'ret4': ret4, 'w4': 'W' if ret4 > 0 else 'L',
        })

sep = "=" * 115
dash = "-" * 115
print(sep)
print("  M18 SQUEEZE v3 — ALL 23 SIGNALS WITH TP/SL LEVELS")
print(sep)
print(f"  #  Time              Type             Dir    Entry     TP       TP%    SL       SL%    TP Hit             SL Hit         4h Ret")
print(f"  {dash}")
for i, s in enumerate(signals):
    tp_str = f"{s['tp_time']} ({s['tp_bars']:>2}b)" if s['tp_time'] else "  --"
    sl_str = f"{s['sl_time']} ({s['sl_bars']:>2}b)" if s['sl_time'] else "  --"
    w = s['w4']
    print(f"  {i+1:>2} {s['ts']:>16}  {s['type']:<16} {s['dir']:>5}  ${s['price']:>7.0f}  ${s['tp']:>7.0f}  {s['tp_pct']:>+5.2f}%  ${s['sl']:>7.0f}  {s['sl_pct']:>+5.2f}%  {tp_str:<18} {sl_str:<18} {w}({s['ret4']:+.2f}%)")

wins = [s for s in signals if s['w4'] == 'W']
losses = [s for s in signals if s['w4'] == 'L']
print(f"\n  WINS: {len(wins)}/23 ({len(wins)/23*100:.1f}%)")
for s in wins:
    hit = f"TP hit {s['tp_time']} ({s['tp_bars']}b)" if s['tp_time'] else "4h hold"
    print(f"    {s['ts']}  {s['dir']:>5}  ${s['price']:>7.0f}  -> ${s['tp']:>7.0f}  {s['ret4']:+.2f}%  {hit}")

print(f"\n  LOSSES: {len(losses)}/23")
for s in losses:
    hit = f"SL hit {s['sl_time']} ({s['sl_bars']}b)" if s['sl_time'] else "4h hold"
    print(f"    {s['ts']}  {s['dir']:>5}  ${s['price']:>7.0f}  -> ${s['sl']:>7.0f}  {s['ret4']:+.2f}%  {hit}")

# TP hit rate
tp_hits = [s for s in signals if s['tp_time']]
sl_hits = [s for s in signals if s['sl_time']]
both = [s for s in signals if s['tp_time'] and s['sl_time']]
tp_first = [s for s in signals if s['tp_time'] and s['sl_time'] and s['tp_bars'] < s['sl_bars']]
sl_first = [s for s in signals if s['tp_time'] and s['sl_time'] and s['sl_bars'] < s['tp_bars']]
neither = [s for s in signals if not s['tp_time'] and not s['sl_time']]

print(f"\n  TP/SL HIT ANALYSIS:")
print(f"    TP hit:     {len(tp_hits)}/23 ({len(tp_hits)/23*100:.1f}%)")
print(f"    SL hit:     {len(sl_hits)}/23 ({len(sl_hits)/23*100:.1f}%)")
print(f"    Both hit:   {len(both)}/23  (TP first: {len(tp_first)}, SL first: {len(sl_first)})")
print(f"    Neither:    {len(neither)}/23  (closed at 4h hold)")
if tp_hits:
    avg_tp_bars = np.mean([s['tp_bars'] for s in tp_hits])
    print(f"    Avg TP bar: {avg_tp_bars:.1f} bars ({avg_tp_bars*15:.0f} min)")
if sl_hits:
    avg_sl_bars = np.mean([s['sl_bars'] for s in sl_hits])
    print(f"    Avg SL bar: {avg_sl_bars:.1f} bars ({avg_sl_bars*15:.0f} min)")

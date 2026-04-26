#!/usr/bin/env python3
"""
Patch jimi_v615_enhanced.py → jimi_v616_beast.py
Integrates all 5 beast modules into the backtest engine.
"""

import re

with open('jimi_v615_enhanced.py', 'r') as f:
    code = f.read()

# ═══════════════════════════════════════════════════════════════════════
# 1. Add import for beast modules at the top (after existing imports)
# ═══════════════════════════════════════════════════════════════════════

import_block = '''import ccxt

HAS_DERIVATIVES = True

# ═══ BEAST MODULES (v6.16) ═══
from beast_modules import (
    compute_vol_regime, score_vol_regime,
    m10_prepare_data, m10_get_row, m10_compute_emas, score_m10_macro,
    score_m11_mtf_momentum,
    score_m12_orderbook,
    compute_adaptive_direction,
    BEAST_CONFIG,
)'''

code = code.replace('import ccxt\n\nHAS_DERIVATIVES = True', import_block)

# ═══════════════════════════════════════════════════════════════════════
# 2. Add BEAST_CONFIG entries to CONFIG dict
# ═══════════════════════════════════════════════════════════════════════

# Find the end of CONFIG dict (the closing brace before the DATA LOADING section)
config_end_marker = "    # --- DATA ---\n    \"PAIR\": \"ETHUSDT\","
beast_config_entries = '''
    # --- M9: Volatility Regime ---
    "M9_ENABLED": True,
    "M9_WEIGHT": 0.10,
    "M9_BLOCK_REGIMES": ["CRISIS"],
    "M9_SIZE_CHOP": 0.60,
    "M9_SIZE_COMPRESSING": 0.80,

    # --- M10: Cross-Asset Macro ---
    "M10_ENABLED": True,
    "M10_WEIGHT": 0.10,

    # --- M11: Multi-TF Momentum Divergence ---
    "M11_ENABLED": True,
    "M11_WEIGHT": 0.12,
    "M11_REQUIRE_AGREEMENT": False,

    # --- M12: Order Book Imbalance ---
    "M12_ENABLED": True,
    "M12_WEIGHT": 0.05,
    "M12_LIVE_ONLY": True,

    # --- Adaptive Direction Bias ---
    "ADAPTIVE_DIR_ENABLED": True,
    "ADAPTIVE_DIR_MIN_BIAS": 0.10,
    "ADAPTIVE_DIR_BLOCK_THRESHOLD": 0.40,

    # --- DATA ---
    "PAIR": "ETHUSDT",'''

code = code.replace(config_end_marker, beast_config_entries)

# ═══════════════════════════════════════════════════════════════════════
# 3. Add M10/M11 data prep in run_backtest (after M7 data prep)
# ═══════════════════════════════════════════════════════════════════════

m7_section_end = '        print(f"  M7 data: ETH/BTC={eb_n} days, BTC/USDT={bt_n} days")'

m10_prep_block = '''
        print(f"  M7 data: ETH/BTC={eb_n} days, BTC/USDT={bt_n} days")

    # --- M10: Cross-Asset Macro Data ---
    m10_data = None
    if CONFIG.get('M10_ENABLED', False):
        print("[3b-2] Fetching M10 cross-asset macro data...")
        try:
            m10_data = m10_prepare_data(df_15m)
            m10_data = m10_compute_emas(m10_data)
            for k, v in m10_data.items():
                n = len(v) if v is not None else 0
                print(f"    {k}: {n} days")
        except Exception as e:
            print(f"    M10 fetch failed: {e}, macro scoring disabled")
            m10_data = None'''

code = code.replace(m7_section_end, m10_prep_block)

# ═══════════════════════════════════════════════════════════════════════
# 4. Add RSI on 4H and MACD hist on 4H (for M11 divergence)
# ═══════════════════════════════════════════════════════════════════════

rsi_section = '    if CONFIG.get(\'M1_RSI_ENABLED\', False):\n        df_1h[\'rsi\'] = calc_rsi(df_1h[\'Close\'], CONFIG.get(\'M1_RSI_PERIOD\', 14))\n        print(f"  RSI (1H) computed.")'

rsi_extended = '''    if CONFIG.get('M1_RSI_ENABLED', False):
        df_1h['rsi'] = calc_rsi(df_1h['Close'], CONFIG.get('M1_RSI_PERIOD', 14))
        print(f"  RSI (1H) computed.")

    # M11: RSI on 15m and MACD hist on 4H (for multi-TF divergence)
    df_15m['rsi'] = calc_rsi(df_15m['Close'], 14)
    df_4h['macd_line'], df_4h['macd_signal'], df_4h['macd_hist'] = calc_macd(
        df_4h['Close'], CONFIG['MACD_FAST'], CONFIG['MACD_SLOW'], CONFIG['MACD_SIGNAL'])
    if 'rsi' not in df_1h.columns:
        df_1h['rsi'] = calc_rsi(df_1h['Close'], 14)
    print(f"  RSI (15m) + MACD (4H) computed for M11 divergence.")'''

code = code.replace(rsi_section, rsi_extended)

# ═══════════════════════════════════════════════════════════════════════
# 5. Add stats counters for new modules
# ═══════════════════════════════════════════════════════════════════════

old_stats = "        'mtf_blocked': 0, 'm8_pass': 0, 'm8_fail': 0,"
new_stats = """        'mtf_blocked': 0, 'm8_pass': 0, 'm8_fail': 0,
        'm9_pass': 0, 'm9_fail': 0, 'm9_block': 0,
        'm10_pass': 0, 'm10_fail': 0,
        'm11_pass': 0, 'm11_fail': 0, 'm11_skip': 0,
        'm12_pass': 0, 'm12_fail': 0, 'm12_skip': 0,
        'adaptive_dir_block': 0,"""
code = code.replace(old_stats, new_stats)

# ═══════════════════════════════════════════════════════════════════════
# 6. Replace the trend filter section with adaptive direction bias
# ═══════════════════════════════════════════════════════════════════════

# Find the trend filter block and replace it
old_trend_filter = """        # === TREND FILTER — block counter-trend trades ===
        if CONFIG.get('TREND_FILTER_ENABLED', False):
            _trend_is_bull = trend_dir in ('STRONG_UP', 'UP')
            _trend_is_bear = trend_dir in ('STRONG_DOWN', 'DOWN')
            _trend_is_strong = trend_dir in ('STRONG_UP', 'STRONG_DOWN')

            # In strong trends, only trade with the trend
            if CONFIG.get('TREND_BLOCK_COUNTER_TREND', False):
                if _trend_is_bear and direction == 'LONG':
                    direction = "SHORT" if direction == "LONG" else "LONG"
                    stats["trend_flip"] = stats.get("trend_flip", 0) + 1
                    continue
                if _trend_is_bull and direction == 'SHORT':
                    direction = "SHORT" if direction == "LONG" else "LONG"
                    stats["trend_flip"] = stats.get("trend_flip", 0) + 1
                    continue

            # Require minimum trend score for entry
            min_score = CONFIG.get('TREND_MIN_SCORE', 0.15)
            if abs(trend_val) < min_score:
                stats['trend_weak'] = stats.get('trend_weak', 0) + 1
                continue"""

new_trend_filter = """        # === ADAPTIVE DIRECTION BIAS (replaces binary trend filter) ===
        if CONFIG.get('ADAPTIVE_DIR_ENABLED', False):
            # Get EMA values for adaptive direction
            ema_1h_f = df_1h['ema_fast'].iloc[idx_1h] if idx_1h >= 0 else None
            ema_1h_s = df_1h['ema_slow'].iloc[idx_1h] if idx_1h >= 0 else None
            ema_4h_f = df_4h['ema_fast'].iloc[idx_4h] if idx_4h >= 0 else None
            ema_4h_s = df_4h['ema_slow'].iloc[idx_4h] if idx_4h >= 0 else None
            ema_1d_f = calc_ema(df_1d['Close'], CONFIG['EMA_FAST']).iloc[idx_1d] if idx_1d >= 0 else None
            ema_1d_s = calc_ema(df_1d['Close'], CONFIG['EMA_SLOW']).iloc[idx_1d] if idx_1d >= 0 else None

            # Get vol regime for context (will be scored later, use cached)
            vol_regime_for_dir = 'NEUTRAL'

            dir_bias, dir_allowed, dir_details = compute_adaptive_direction(
                trend_dir, trend_val,
                ema_1h_f, ema_1h_s,
                ema_4h_f, ema_4h_s,
                ema_1d_f, ema_1d_s,
                vol_regime_for_dir,
                recent_trades=trades[-8:] if trades else None,
                direction=direction
            )

            if not dir_allowed:
                stats['adaptive_dir_block'] = stats.get('adaptive_dir_block', 0) + 1
                continue

        # === Legacy trend filter (kept as secondary check) ===
        if CONFIG.get('TREND_FILTER_ENABLED', False):
            _trend_is_bull = trend_dir in ('STRONG_UP', 'UP')
            _trend_is_bear = trend_dir in ('STRONG_DOWN', 'DOWN')

            # In strong trends, flip direction to trade with trend
            if CONFIG.get('TREND_BLOCK_COUNTER_TREND', False):
                if _trend_is_bear and direction == 'LONG':
                    stats["trend_flip"] = stats.get("trend_flip", 0) + 1
                    continue
                if _trend_is_bull and direction == 'SHORT':
                    stats["trend_flip"] = stats.get("trend_flip", 0) + 1
                    continue

            min_score = CONFIG.get('TREND_MIN_SCORE', 0.15)
            if abs(trend_val) < min_score:
                stats['trend_weak'] = stats.get('trend_weak', 0) + 1
                continue"""

code = code.replace(old_trend_filter, new_trend_filter)

# ═══════════════════════════════════════════════════════════════════════
# 7. Add M9/M10/M11/M12 scoring into the backtest loop
# (after M8 scoring, before Cross-Asset Correlation)
# ═══════════════════════════════════════════════════════════════════════

old_m8_section = """        # ===== M8: Funding Rate Scoring =====
        m8_score = 0.5
        m8_status = 'SKIP'
        use_m8 = False
        if CONFIG.get('M8_ENABLED', False) and cached_funding_rate is not None:
            m8_status, m8_score, m8_details = score_m8_funding(cached_funding_rate, direction)
            use_m8 = True
            stats['m8_pass'] = stats.get('m8_pass', 0) + (1 if m8_status == 'PASS' else 0)
            stats['m8_fail'] = stats.get('m8_fail', 0) + (1 if m8_status == 'FAIL' else 0)

        # ===== Cross-Asset Correlation ====="""

new_m8_section = """        # ===== M8: Funding Rate Scoring =====
        m8_score = 0.5
        m8_status = 'SKIP'
        use_m8 = False
        if CONFIG.get('M8_ENABLED', False) and cached_funding_rate is not None:
            m8_status, m8_score, m8_details = score_m8_funding(cached_funding_rate, direction)
            use_m8 = True
            stats['m8_pass'] = stats.get('m8_pass', 0) + (1 if m8_status == 'PASS' else 0)
            stats['m8_fail'] = stats.get('m8_fail', 0) + (1 if m8_status == 'FAIL' else 0)

        # ===== M9: Volatility Regime =====
        m9_score = 0.5
        m9_status = 'SKIP'
        vol_regime = 'NEUTRAL'
        use_m9 = False
        if CONFIG.get('M9_ENABLED', False):
            vol_regime, m9_raw, m9_vol_details = compute_vol_regime(df_15m, df_1h, idx, idx_1h)
            m9_status, m9_score, m9_details = score_vol_regime(vol_regime, m9_raw, direction, trend_dir)
            use_m9 = True
            stats['m9_pass'] = stats.get('m9_pass', 0) + (1 if m9_status == 'PASS' else 0)
            stats['m9_fail'] = stats.get('m9_fail', 0) + (1 if m9_status == 'FAIL' else 0)
            # Block crisis regime entirely
            if vol_regime in CONFIG.get('M9_BLOCK_REGIMES', []):
                stats['m9_block'] = stats.get('m9_block', 0) + 1
                continue

        # ===== M10: Cross-Asset Macro =====
        m10_score = 0.5
        m10_status = 'SKIP'
        use_m10 = False
        if CONFIG.get('M10_ENABLED', False) and m10_data is not None:
            macro_row = m10_get_row(m10_data, ts)
            if macro_row:
                m10_status, m10_score, m10_details = score_m10_macro(macro_row, direction, trend_dir)
                use_m10 = True
                stats['m10_pass'] = stats.get('m10_pass', 0) + (1 if m10_status == 'PASS' else 0)
                stats['m10_fail'] = stats.get('m10_fail', 0) + (1 if m10_status == 'FAIL' else 0)

        # ===== M11: Multi-TF Momentum Divergence =====
        m11_score = 0.5
        m11_status = 'SKIP'
        use_m11 = False
        if CONFIG.get('M11_ENABLED', False):
            m11_status, m11_score, m11_details = score_m11_mtf_momentum(
                df_15m, df_1h, df_4h, idx, idx_1h, idx_4h, direction)
            use_m11 = True
            if m11_status == 'PASS':
                stats['m11_pass'] = stats.get('m11_pass', 0) + 1
            elif m11_status == 'FAIL':
                stats['m11_fail'] = stats.get('m11_fail', 0) + 1
            else:
                stats['m11_skip'] = stats.get('m11_skip', 0) + 1

        # ===== M12: Order Book Imbalance =====
        m12_score = 0.5
        m12_status = 'SKIP'
        use_m12 = False
        if CONFIG.get('M12_ENABLED', False) and not CONFIG.get('M12_LIVE_ONLY', True):
            m12_status, m12_score, m12_details = score_m12_orderbook(direction, live=False)
            use_m12 = True
            stats['m12_pass'] = stats.get('m12_pass', 0) + (1 if m12_status == 'PASS' else 0)
            stats['m12_fail'] = stats.get('m12_fail', 0) + (1 if m12_status == 'FAIL' else 0)

        # ===== Cross-Asset Correlation ====="""

code = code.replace(old_m8_section, new_m8_section)

# ═══════════════════════════════════════════════════════════════════════
# 8. Update calc_ics call to include new modules
# ═══════════════════════════════════════════════════════════════════════

old_ics_call = """        use_m7 = CONFIG.get('M7_ENABLED', False) and m7_ethbtc_df is not None
        ics, effective_floor = calc_ics(m1_score, m2_score, m3_score, m4_score, m4_status, m5_score,
                                        m7_score=m7_score, m8_score=m8_score, cross_asset_score=cross_asset_score,
                                        use_m7=use_m7, use_m8=use_m8, use_cross_asset=use_cross_asset,
                                        cascade_dir=cascade_dir, cascade_strength=cascade_strength)"""

new_ics_call = """        use_m7 = CONFIG.get('M7_ENABLED', False) and m7_ethbtc_df is not None
        ics, effective_floor = calc_ics(m1_score, m2_score, m3_score, m4_score, m4_status, m5_score,
                                        m7_score=m7_score, m8_score=m8_score, cross_asset_score=cross_asset_score,
                                        use_m7=use_m7, use_m8=use_m8, use_cross_asset=use_cross_asset,
                                        cascade_dir=cascade_dir, cascade_strength=cascade_strength,
                                        m9_score=m9_score, use_m9=use_m9,
                                        m10_score=m10_score, use_m10=use_m10,
                                        m11_score=m11_score, use_m11=use_m11,
                                        m12_score=m12_score, use_m12=use_m12)"""

code = code.replace(old_ics_call, new_ics_call)

# ═══════════════════════════════════════════════════════════════════════
# 9. Update calc_ics function signature and body
# ═══════════════════════════════════════════════════════════════════════

old_calc_ics_sig = "def calc_ics(m1_score, m2_score, m3_score, m4_score, m4_status, m5_score=0.5, m6_score=0.5, m7_score=0.5, m8_score=0.5, cross_asset_score=0.5, use_derivatives=False, use_m7=False, use_m8=False, use_cross_asset=False, cascade_dir='NONE', cascade_strength=0.0):"

new_calc_ics_sig = "def calc_ics(m1_score, m2_score, m3_score, m4_score, m4_status, m5_score=0.5, m6_score=0.5, m7_score=0.5, m8_score=0.5, cross_asset_score=0.5, use_derivatives=False, use_m7=False, use_m8=False, use_cross_asset=False, cascade_dir='NONE', cascade_strength=0.0, m9_score=0.5, use_m9=False, m10_score=0.5, use_m10=False, m11_score=0.5, use_m11=False, m12_score=0.5, use_m12=False):"

code = code.replace(old_calc_ics_sig, new_calc_ics_sig)

# Add new modules to extra_modules in calc_ics
old_extra_modules = """    extra_modules = []
    if use_m7 and CONFIG.get('M7_ENABLED', False):
        extra_modules.append(('M7', m7_score, CONFIG['M7_WEIGHT']))
    if use_m8 and CONFIG.get('M8_ENABLED', False):
        extra_modules.append(('M8', m8_score, CONFIG.get('M8_WEIGHT', 0.10)))
    if use_cross_asset and CONFIG.get('CROSS_ASSET_ENABLED', False):
        extra_modules.append(('CA', cross_asset_score, CONFIG.get('CROSS_ASSET_BTC_WEIGHT', 0.08)))"""

new_extra_modules = """    extra_modules = []
    if use_m7 and CONFIG.get('M7_ENABLED', False):
        extra_modules.append(('M7', m7_score, CONFIG['M7_WEIGHT']))
    if use_m8 and CONFIG.get('M8_ENABLED', False):
        extra_modules.append(('M8', m8_score, CONFIG.get('M8_WEIGHT', 0.10)))
    if use_cross_asset and CONFIG.get('CROSS_ASSET_ENABLED', False):
        extra_modules.append(('CA', cross_asset_score, CONFIG.get('CROSS_ASSET_BTC_WEIGHT', 0.08)))
    if use_m9 and CONFIG.get('M9_ENABLED', False):
        extra_modules.append(('M9', m9_score, CONFIG.get('M9_WEIGHT', 0.10)))
    if use_m10 and CONFIG.get('M10_ENABLED', False):
        extra_modules.append(('M10', m10_score, CONFIG.get('M10_WEIGHT', 0.10)))
    if use_m11 and CONFIG.get('M11_ENABLED', False):
        extra_modules.append(('M11', m11_score, CONFIG.get('M11_WEIGHT', 0.12)))
    if use_m12 and CONFIG.get('M12_ENABLED', False):
        extra_modules.append(('M12', m12_score, CONFIG.get('M12_WEIGHT', 0.05)))"""

code = code.replace(old_extra_modules, new_extra_modules)

# ═══════════════════════════════════════════════════════════════════════
# 10. Add M9 size adjustments to position sizing
# ═══════════════════════════════════════════════════════════════════════

old_size_m7 = """        # v6.13: M7 size adjustment — reduce when macro regime is unfavorable
        if CONFIG.get('M7_ENABLED', False) and m7_ethbtc_df is not None:
            if m7_score < 0.35:
                size *= CONFIG.get('M7_SIZE_REDUCTION', 0.70)
            elif m7_score < 0.45:
                size *= CONFIG.get('M7_SIZE_MILD', 0.85)"""

new_size_m7 = """        # v6.13: M7 size adjustment — reduce when macro regime is unfavorable
        if CONFIG.get('M7_ENABLED', False) and m7_ethbtc_df is not None:
            if m7_score < 0.35:
                size *= CONFIG.get('M7_SIZE_REDUCTION', 0.70)
            elif m7_score < 0.45:
                size *= CONFIG.get('M7_SIZE_MILD', 0.85)
        # M9: Volatility regime size adjustment
        if CONFIG.get('M9_ENABLED', False):
            if vol_regime == 'CHOP':
                size *= CONFIG.get('M9_SIZE_CHOP', 0.60)
            elif vol_regime == 'COMPRESSING':
                size *= CONFIG.get('M9_SIZE_COMPRESSING', 0.80)"""

code = code.replace(old_size_m7, new_size_m7)

# ═══════════════════════════════════════════════════════════════════════
# 11. Update the report header and version strings
# ═══════════════════════════════════════════════════════════════════════

code = code.replace(
    'JIMI FRAMEWORK v6.13 — Backtest Engine (M1-M7 + Liquidation)',
    'JIMI FRAMEWORK v6.16 BEAST — Backtest Engine (M1-M12 + Adaptive Direction)'
)
code = code.replace(
    'JIMI v6.10 — BACKTEST RESULTS (5 Modules + Liquidation Magnet)',
    'JIMI v6.16 BEAST — BACKTEST RESULTS (12 Modules + Adaptive Direction)'
)

# Update signal flow stats in report
old_signal_flow = """    for k in ['signals_checked','m1_neutral_skip','m3_fail','m2_neutral_long_skip','long_disabled','long_ics_skip','long_m5_skip','long_phase0_skip','ics_blocked','ics_ceiling_skip','m4_required_skip','m4_false_anchored','m5_pass','m5_fail','cascade_detected','dedup_skip','filter_blocked','consec_pause','bias_gate_skip','monthly_dd_skip','dir_veto_skip','entries']:"""
new_signal_flow = """    for k in ['signals_checked','m1_neutral_skip','m3_fail','m2_neutral_long_skip','long_disabled','long_ics_skip','long_m5_skip','long_phase0_skip','ics_blocked','ics_ceiling_skip','m4_required_skip','m4_false_anchored','m5_pass','m5_fail','cascade_detected','dedup_skip','filter_blocked','consec_pause','bias_gate_skip','monthly_dd_skip','dir_veto_skip','adaptive_dir_block','m9_block','m10_pass','m10_fail','m11_pass','m11_fail','m11_skip','entries']:"""

code = code.replace(old_signal_flow, new_signal_flow)

# ═══════════════════════════════════════════════════════════════════════
# Write output
# ═══════════════════════════════════════════════════════════════════════

with open('jimi_v616_beast.py', 'w') as f:
    f.write(code)

print(f"✅ Patched successfully!")
print(f"   Input:  jimi_v615_enhanced.py ({len(open('jimi_v615_enhanced.py').read())} bytes)")
print(f"   Output: jimi_v616_beast.py ({len(code)} bytes)")
print(f"   New modules: M9 (Vol Regime), M10 (Macro), M11 (MTF Momentum), M12 (OrderBook), Adaptive Direction")

# JIMI — Missing Macro Filter Modules: Implementation Spec

**Date:** 2026-05-18
**Purpose:** Implement the 8 genuinely missing traditional-finance signals identified in the
`realtime_signal_filters_analysis.txt` validation. Each module follows JIMI's scoring convention:
`(status: str, score: float, details: dict)` where score ∈ [0.0, 1.0].

---

## Table of Contents

1. [M66 — USD/JPY Carry Trade Proxy](#m66)
2. [M67 — DXY Divergence Filter](#m67)
3. [M68 — 10Y Yield + TIPS Real Yield](#m68)
4. [M69 — VIX Regime Classifier](#m69)
5. [M70 — WTI Crude Oil Signal](#m70)
6. [M71 — Gold + DXY Geopolitical Filter](#m71)
7. [M72 — BTC Dominance Regime](#m72)
8. [M73 — Stablecoin Mint Flows](#m73)
9. [Integration — Scanner Wiring](#integration)
10. [Config Additions](#config)

---

<a name="m66"></a>
## M66 — USD/JPY Carry Trade Proxy

**File:** `src/modules/m66_usdjpy.py`

### Rationale
USD/JPY is the most directly correlated FX pair to crypto volatility. A sharp JPY
strengthening (USD/JPY drop) signals carry trade unwind → risk-off → ETH selling.
But the direction depends on WHY USD/JPY drops — requires DXY cross-check.

### Data Source
```python
import yfinance as yf

def fetch_usdjpy(period="5d", interval="1m"):
    """Fetch USD/JPY 1m bars. yfinance gives 7 days of 1m data."""
    df = yf.download("JPY=X", period=period, interval=interval, progress=False)
    return df
```

### Scoring Logic

```python
def score_m66_usdjpy(df_usdjpy, df_dxy, direction, config=None):
    """
    Monitor USD/JPY rate of change with DXY cross-check.

    Returns:
        (status, score, details)
        status: 'PASS' | 'NEUTRAL' | 'SKIP'
        score: 0.0–1.0 (distance from neutral)
        details: dict with classification and diagnostics
    """
    cfg = config or CONFIG

    # Compute 5m and 15m rate of change
    roc_5m = (df_usdjpy['Close'].iloc[-1] - df_usdjpy['Close'].iloc[-5]) / df_usdjpy['Close'].iloc[-5] * 100
    roc_15m = (df_usdjpy['Close'].iloc[-1] - df_usdjpy['Close'].iloc[-15]) / df_usdjpy['Close'].iloc[-15] * 100

    # DXY direction in same window
    dxy_roc_5m = (df_dxy['Close'].iloc[-1] - df_dxy['Close'].iloc[-5]) / df_dxy['Close'].iloc[-5] * 100

    # Classification
    if roc_5m < cfg['M66_ALERT_THRESH'] and dxy_roc_5m >= 0:
        # DXY flat/rising → real carry unwind
        classification = 'CARRY_ALERT'
        bearish_score = 0.65
    elif roc_15m < cfg['M66_CONFIRMED_THRESH'] and dxy_roc_5m >= 0:
        classification = 'CARRY_UNWIND_CONFIRMED'
        bearish_score = 0.80
    elif roc_5m < cfg['M66_ALERT_THRESH'] and dxy_roc_5m < 0:
        # DXY also falling → USD weakness, NOT carry unwind
        classification = 'USD_WEAKNESS'
        bearish_score = 0.5  # neutral — not a carry event
    else:
        classification = 'NORMAL'
        bearish_score = 0.5

    # Direction-aware scoring
    if direction == 'LONG':
        score = 1.0 - bearish_score + 0.5  # invert: carry unwind hurts longs
    elif direction == 'SHORT':
        score = bearish_score  # carry unwind helps shorts
    else:
        score = 0.5

    score = max(0.0, min(1.0, score))

    details = {
        'classification': classification,
        'usdjpy_roc_5m': round(roc_5m, 4),
        'usdjpy_roc_15m': round(roc_15m, 4),
        'dxy_roc_5m': round(dxy_roc_5m, 4),
    }

    status = 'PASS' if classification != 'NORMAL' else 'NEUTRAL'
    return status, score, details
```

### Thresholds (config keys)
```yaml
M66_ENABLED: true
M66_ALERT_THRESH: -0.3       # % drop in 5m window
M66_CONFIRMED_THRESH: -0.8   # % drop in 15m window
M66_CARRY_ALERT_SCORE: 0.65
M66_CARRY_CONFIRMED_SCORE: 0.80
```

### ICS Weight
```yaml
M66_WEIGHT: 0.08  # moderate — FX carry is high-conviction but rare
```

---

<a name="m67"></a>
## M67 — DXY Divergence Filter

**File:** `src/modules/m67_dxy.py`

### Rationale
DXY is a lagging composite — it reacts to the same macro data ETH reacts to.
Divergences between DXY and ETH are more useful than DXY direction alone.
ETH showing strength while DXY rises = institutional accumulation signal.

### Data Source
```python
import yfinance as yf

def fetch_dxy(period="5d", interval="15m"):
    """Fetch DXY futures. yfinance: DX-Y.NYB."""
    df = yf.download("DX-Y.NYB", period=period, interval=interval, progress=False)
    return df
```

### Scoring Logic

```python
def score_m67_dxy(df_dxy, eth_price_now, eth_price_15m_ago, direction, config=None):
    """
    Classify DXY/ETH divergence.

    Returns:
        (status, score, details)
    """
    cfg = config or CONFIG

    dxy_roc = (df_dxy['Close'].iloc[-1] - df_dxy['Close'].iloc[-2]) / df_dxy['Close'].iloc[-2] * 100
    eth_roc = (eth_price_now - eth_price_15m_ago) / eth_price_15m_ago * 100

    dxy_up = dxy_roc > cfg['M67_DXY_THRESH']
    dxy_down = dxy_roc < -cfg['M67_DXY_THRESH']
    eth_up = eth_roc > 0.05
    eth_down = eth_roc < -0.05

    # Four-condition classification
    if dxy_up and eth_down:
        classification = 'CONFIRMED_BEARISH'
        score_long = 0.25
    elif dxy_up and not eth_down:
        classification = 'BULLISH_DIVERGENCE'  # ETH showing relative strength
        score_long = 0.70
    elif dxy_down and eth_up:
        classification = 'CONFIRMED_BULLISH'
        score_long = 0.75
    elif dxy_down and not eth_up:
        classification = 'BEARISH_DIVERGENCE'  # ETH showing relative weakness
        score_long = 0.30
    else:
        classification = 'NEUTRAL'
        score_long = 0.50

    if direction == 'LONG':
        score = score_long
    elif direction == 'SHORT':
        score = 1.0 - score_long
    else:
        score = 0.5

    details = {
        'classification': classification,
        'dxy_roc_15m': round(dxy_roc, 4),
        'eth_roc_15m': round(eth_roc, 4),
        'divergence': classification in ('BULLISH_DIVERGENCE', 'BEARISH_DIVERGENCE'),
    }

    status = 'PASS' if classification != 'NEUTRAL' else 'NEUTRAL'
    return status, score, details
```

### Thresholds
```yaml
M67_ENABLED: true
M67_DXY_THRESH: 0.2          # % move in 15m to be significant
M67_DIVERGENCE_ETH_THRESH: 0.05  # % ETH move to count as "moving"
```

### ICS Weight
```yaml
M67_WEIGHT: 0.06  # moderate — divergences are high-signal but infrequent
```

---

<a name="m68"></a>
## M68 — 10Y Treasury Yield + TIPS Real Yield

**File:** `src/modules/m68_yield.py`

### Rationale
10Y yield is the risk-free rate benchmark. Spikes compress crypto valuations.
But nominal spikes driven by inflation expectations vs real yield spikes have
different ETH impacts. TIPS cross-check distinguishes them.

### Data Source
```python
import yfinance as yf
# FRED alternative: fredapi or direct CSV download

def fetch_10y_yield(period="5d", interval="1h"):
    """Fetch ^TNX (10Y yield * 10). Divide by 10 for actual yield."""
    df = yf.download("^TNX", period=period, interval=interval, progress=False)
    df['yield'] = df['Close'] / 10  # yfinance returns yield * 10
    return df

def fetch_tips_yield():
    """Fetch TIPS 10Y real yield from FRED (daily only)."""
    import requests
    url = "https://fred.stlouisfed.org/graph/fredgraph.csv?id=DFII10"
    df = pd.read_csv(url, parse_dates=['DATE'], index_col='DATE')
    return df
```

### Scoring Logic

```python
def score_m68_yield(df_10y, df_tips, direction, config=None):
    """
    Classify 10Y yield spike with TIPS cross-check.

    Returns:
        (status, score, details)
    """
    cfg = config or CONFIG

    # Compute 1h yield change in basis points
    yield_now = float(df_10y['yield'].iloc[-1])
    yield_prev = float(df_10y['yield'].iloc[-2])
    delta_bps = (yield_now - yield_prev) * 100  # convert % to bps

    # TIPS cross-check (daily, so use latest available)
    tips_now = None
    tips_prev = None
    if df_tips is not None and len(df_tips) > 1:
        tips_now = float(df_tips['DFII10'].iloc[-1])
        tips_prev = float(df_tips['DFII10'].iloc[-2])

    # Classification
    spike = delta_bps > cfg['M68_ALERT_BPS']
    extreme = delta_bps > cfg['M68_EXTREME_BPS']

    inflation_driven = False
    growth_driven = False
    if spike and tips_now is not None and tips_prev is not None:
        tips_delta = tips_now - tips_prev
        if abs(tips_delta) < 2:
            # TIPS flat, nominal rising → inflation expectations
            inflation_driven = True
        elif tips_delta > 2:
            # Both rising → growth signal
            growth_driven = True

    if extreme:
        classification = 'YIELD_EXTREME'
        score_long = 0.15
    elif spike and inflation_driven:
        classification = 'INFLATION_SPIKE'
        score_long = 0.25
    elif spike and growth_driven:
        classification = 'GROWTH_SPIKE'
        score_long = 0.40
    elif spike:
        classification = 'YIELD_SPIKE'
        score_long = 0.30
    else:
        classification = 'NORMAL'
        score_long = 0.50

    if direction == 'LONG':
        score = score_long
    elif direction == 'SHORT':
        score = 1.0 - score_long
    else:
        score = 0.5

    details = {
        'classification': classification,
        'yield_now': round(yield_now, 4),
        'delta_bps': round(delta_bps, 2),
        'tips_now': round(tips_now, 4) if tips_now else None,
        'inflation_driven': inflation_driven,
        'growth_driven': growth_driven,
    }

    status = 'PASS' if spike else 'NEUTRAL'
    return status, score, details
```

### Thresholds
```yaml
M68_ENABLED: true
M68_ALERT_BPS: 5.0           # 5bps in 1h = significant
M68_EXTREME_BPS: 10.0        # 10bps in 1h = extreme, suppress longs
```

### ICS Weight
```yaml
M68_WEIGHT: 0.10  # strong — yield spikes are high-impact, mechanistic
```

---

<a name="m69"></a>
## M69 — VIX Regime Classifier

**File:** `src/modules/m69_vix.py`

### Rationale
VIX has a non-linear relationship with ETH. VIX > 40 = crisis = capitulation =
often the BEST long entries, not worst. Rate of change matters as much as level.

### Data Source
```python
import yfinance as yf

def fetch_vix(period="5d", interval="1d"):
    """Fetch ^VIX. Daily bars (intraday requires paid CBOE data)."""
    df = yf.download("^VIX", period=period, interval=interval, progress=False)
    return df
```

### Scoring Logic

```python
def score_m69_vix(df_vix, direction, config=None):
    """
    Classify VIX regime with non-linear ETH impact.

    Returns:
        (status, score, details)
    """
    cfg = config or CONFIG

    vix_now = float(df_vix['Close'].iloc[-1])
    vix_prev = float(df_vix['Close'].iloc[-2])
    vix_delta = vix_now - vix_prev

    # Regime classification
    if vix_now > cfg['M69_CRISIS_THRESH']:
        classification = 'CRISIS'
        # Counterintuitive: crisis = potential long (capitulation)
        long_score = 0.55  # slightly bullish — contrarian setup
        reversal_flag = True
    elif vix_now > cfg['M69_FEAR_THRESH']:
        classification = 'FEAR'
        long_score = 0.35
        reversal_flag = False
    elif vix_now > cfg['M69_ELEVATED_THRESH']:
        classification = 'ELEVATED'
        long_score = 0.40
        reversal_flag = False
    elif vix_now < cfg['M69_COMPLACENT_THRESH']:
        classification = 'COMPLACENT'
        long_score = 0.45  # complacency = squeeze risk
        reversal_flag = False
    else:
        classification = 'NORMAL'
        long_score = 0.50
        reversal_flag = False

    # Rate-of-change override: spike > 3 pts = immediate risk-off
    if vix_delta > cfg['M69_SPIKE_DELTA']:
        classification = 'VIX_SPIKE'
        long_score = 0.25  # bearish regardless of level
        reversal_flag = False

    if direction == 'LONG':
        score = long_score
    elif direction == 'SHORT':
        score = 1.0 - long_score
    else:
        score = 0.5

    details = {
        'classification': classification,
        'vix_level': round(vix_now, 2),
        'vix_delta': round(vix_delta, 2),
        'reversal_potential': reversal_flag,
    }

    status = 'PASS' if classification != 'NORMAL' else 'NEUTRAL'
    return status, score, details
```

### Thresholds
```yaml
M69_ENABLED: true
M69_COMPLACENT_THRESH: 15.0
M69_ELEVATED_THRESH: 20.0
M69_FEAR_THRESH: 30.0
M69_CRISIS_THRESH: 40.0
M69_SPIKE_DELTA: 3.0         # 3-point single-session spike
```

### ICS Weight
```yaml
M69_WEIGHT: 0.08  # moderate — VIX is daily resolution, less actionable intraday
```

---

<a name="m70"></a>
## M70 — WTI Crude Oil Signal

**File:** `src/modules/m70_wti.py`

### Rationale
Oil spikes raise inflation expectations → rate hike risk → ETH down.
But oil drops are deflationary → rate cut expectations → ETH bullish.
The document's one-sided treatment misses 50% of the signal.
Classification requires DXY cross-check.

### Data Source
```python
import yfinance as yf

def fetch_wti(period="5d", interval="4h"):
    """Fetch WTI front-month futures. CL=F."""
    df = yf.download("CL=F", period=period, interval=interval, progress=False)
    return df
```

### Scoring Logic

```python
def score_m70_wti(df_wti, df_dxy, direction, config=None):
    """
    Classify oil move with DXY cross-check.

    Returns:
        (status, score, details)
    """
    cfg = config or CONFIG

    oil_roc = (df_wti['Close'].iloc[-1] - df_wti['Close'].iloc[-2]) / df_wti['Close'].iloc[-2] * 100
    dxy_roc = (df_dxy['Close'].iloc[-1] - df_dxy['Close'].iloc[-2]) / df_dxy['Close'].iloc[-2] * 100

    significant = abs(oil_roc) > cfg['M70_ALERT_THRESH']

    if not significant:
        return 'NEUTRAL', 0.5, {'classification': 'NORMAL', 'oil_roc': round(oil_roc, 2)}

    oil_up = oil_roc > 0
    dxy_up = dxy_roc > 0

    # Four-condition classification
    if oil_up and dxy_up:
        # Supply shock: oil up + dollar up = inflation = bearish ETH
        classification = 'SUPPLY_SHOCK_BEARISH'
        long_score = 0.25
    elif oil_up and not dxy_up:
        # Demand-driven or ambiguous
        classification = 'DEMAND_RISK_ON'
        long_score = 0.50
    elif not oil_up and dxy_up:
        # Oil down + DXY up = recession fear, ambiguous
        classification = 'RECESSION_FEAR'
        long_score = 0.45
    else:
        # Oil down + DXY down = reflation/easing = mild bullish
        classification = 'REFLATION_EASING'
        long_score = 0.60

    if direction == 'LONG':
        score = long_score
    elif direction == 'SHORT':
        score = 1.0 - long_score
    else:
        score = 0.5

    details = {
        'classification': classification,
        'oil_roc_4h': round(oil_roc, 2),
        'dxy_roc_4h': round(dxy_roc, 4),
    }

    status = 'PASS' if classification != 'NORMAL' else 'NEUTRAL'
    return status, score, details
```

### Thresholds
```yaml
M70_ENABLED: true
M70_ALERT_THRESH: 3.0        # % move in 4h to trigger
```

### ICS Weight
```yaml
M70_WEIGHT: 0.05  # lower — oil is indirect, lagging, 4h resolution
```

---

<a name="m71"></a>
## M71 — Gold + DXY Geopolitical Filter

**File:** `src/modules/m71_gold.py`

### Rationale
**This is the most critical correction from the analysis document.**
Gold + ETH co-move ONLY during fiat debasement (QE, dollar weakness).
During geopolitical crises, gold rallies while ETH crashes.
Gold up + DXY up = geopolitical panic = BEARISH ETH. Never treat as bullish.

### Data Source
```python
import yfinance as yf

def fetch_gold(period="5d", interval="4h"):
    """Fetch gold futures. GC=F."""
    df = yf.download("GC=F", period=period, interval=interval, progress=False)
    return df
```

### Scoring Logic

```python
def score_m71_gold(df_gold, df_dxy, direction, config=None):
    """
    Classify gold move with DXY to distinguish fiat debasement vs geopolitical.

    Returns:
        (status, score, details)
    """
    cfg = config or CONFIG

    gold_roc = (df_gold['Close'].iloc[-1] - df_gold['Close'].iloc[-2]) / df_gold['Close'].iloc[-2] * 100
    dxy_roc = (df_dxy['Close'].iloc[-1] - df_dxy['Close'].iloc[-2]) / df_dxy['Close'].iloc[-2] * 100

    significant = abs(gold_roc) > cfg['M71_ALERT_THRESH']

    if not significant:
        return 'NEUTRAL', 0.5, {'classification': 'NORMAL', 'gold_roc': round(gold_roc, 2)}

    gold_up = gold_roc > 0
    dxy_up = dxy_roc > 0

    # KEY RULE: Gold up + DXY up = geopolitical = bearish ETH
    if gold_up and dxy_up:
        classification = 'GEOPOLITICAL_SAFE_HAVEN'
        long_score = 0.25  # BEARISH — they are diverging
    elif gold_up and not dxy_up:
        classification = 'FIAT_DEBASEMENT'
        long_score = 0.65  # bullish — same catalyst driving both
    elif not gold_up and dxy_up:
        classification = 'RISK_OFF_RATES'
        long_score = 0.35  # bearish
    else:
        classification = 'RISK_ON_DRIFT'
        long_score = 0.50

    if direction == 'LONG':
        score = long_score
    elif direction == 'SHORT':
        score = 1.0 - long_score
    else:
        score = 0.5

    details = {
        'classification': classification,
        'gold_roc_4h': round(gold_roc, 2),
        'dxy_roc_4h': round(dxy_roc, 4),
        'geopolitical_panic': classification == 'GEOPOLITICAL_SAFE_HAVEN',
    }

    status = 'PASS' if classification != 'NORMAL' else 'NEUTRAL'
    return status, score, details
```

### Thresholds
```yaml
M71_ENABLED: true
M71_ALERT_THRESH: 1.0        # % move in 4h to trigger
```

### ICS Weight
```yaml
M71_WEIGHT: 0.06  # moderate — geopolitical events are rare but high-impact
```

---

<a name="m72"></a>
## M72 — BTC Dominance Regime

**File:** `src/modules/m72_btcdom.py`

### Rationale
BTC.D rising = capital rotating from altcoins into BTC. ETH underperforms
even when macro is neutral or bullish. One of the most important ETH-specific
filters that JIMI completely lacks.

### Data Source
```python
def fetch_btcdom():
    """
    BTC Dominance from CoinMarketCap API (free tier) or CoinGecko.
    Alternative: scrape from TradingView or use ccxt.
    """
    # Option 1: CoinGecko (free, no key)
    import requests
    r = requests.get("https://api.coingecko.com/api/v3/global")
    data = r.json()
    btc_dominance = data['data']['market_cap_percentage']['btc']
    return btc_dominance

    # Option 2: CoinMarketCap (requires free API key)
    # Option 3: yfinance — no direct BTC.D ticker, use workaround
```

### Scoring Logic

```python
def score_m72_btcdom(btc_dominance, direction, config=None):
    """
    Adjust confidence based on BTC dominance regime.

    Returns:
        (status, score, details)
    """
    cfg = config or CONFIG

    if btc_dominance is None:
        return 'SKIP', 0.5, {'error': 'no data'}

    # Regime classification
    if btc_dominance > cfg['M72_HIGH_THRESH']:
        classification = 'BTC_DOMINANT'
        # ETH underperforms — reduce long confidence
        long_score = 0.38
    elif btc_dominance < cfg['M72_LOW_THRESH']:
        classification = 'ALTCOIN_SEASON'
        # ETH outperforms — amplify long confidence
        long_score = 0.62
    else:
        classification = 'NEUTRAL'
        long_score = 0.50

    if direction == 'LONG':
        score = long_score
    elif direction == 'SHORT':
        score = 1.0 - long_score
    else:
        score = 0.5

    details = {
        'classification': classification,
        'btc_dominance': round(btc_dominance, 2),
    }

    status = 'PASS' if classification != 'NEUTRAL' else 'NEUTRAL'
    return status, score, details
```

### Thresholds
```yaml
M72_ENABLED: true
M72_HIGH_THRESH: 55.0        # BTC.D > 55% = altcoin suppression
M72_LOW_THRESH: 48.0         # BTC.D < 48% = altcoin season
```

### ICS Weight
```yaml
M72_WEIGHT: 0.10  # strong — BTC.D is one of the best ETH-specific filters
```

---

<a name="m73"></a>
## M73 — Stablecoin Mint Flows

**File:** `src/modules/m73_stablecoin.py`

### Rationale
Large USDT/USDC mints signal institutional capital queuing to buy crypto.
Often precedes major ETH pumps by 12–48h. Burns are mildly bearish.
JIMI has zero on-chain signals — this fills that gap.

### Data Source
```python
def fetch_stablecoin_mints():
    """
    Option 1: Whale Alert API (free tier, delayed)
    Option 2: Etherscan API — monitor USDT/USDC contract for large transfers
    Option 3: DeFiLlama stablecoin supply API (free, no key)
    """
    import requests
    # DeFiLlama — stablecoin supply changes (free, no key)
    r = requests.get("https://stablecoins.llama.fi/stablecoins?includePrices=true")
    # Filter for USDT + USDC, check supply changes
    return r.json()
```

### Scoring Logic

```python
def score_m73_stablecoin(mint_data, direction, config=None):
    """
    Score based on stablecoin mint/burn activity.

    Returns:
        (status, score, details)
    """
    cfg = config or CONFIG

    if mint_data is None:
        return 'SKIP', 0.5, {'error': 'no data'}

    usdt_mint_24h = mint_data.get('usdt_mint_24h', 0)  # in USD
    usdc_mint_24h = mint_data.get('usdc_mint_24h', 0)
    total_mint = usdt_mint_24h + usdc_mint_24h

    # Classification
    if total_mint > cfg['M73_MEGA_MINT_THRESH']:
        classification = 'MEGA_MINT'
        long_score = 0.62
    elif total_mint > cfg['M73_LARGE_MINT_THRESH']:
        classification = 'LARGE_MINT'
        long_score = 0.58
    elif total_mint < -cfg['M73_LARGE_BURN_THRESH']:
        classification = 'LARGE_BURN'
        long_score = 0.40
    else:
        classification = 'NORMAL'
        long_score = 0.50

    if direction == 'LONG':
        score = long_score
    elif direction == 'SHORT':
        score = 1.0 - long_score
    else:
        score = 0.5

    details = {
        'classification': classification,
        'usdt_mint_24h': usdt_mint_24h,
        'usdc_mint_24h': usdc_mint_24h,
        'total_mint_24h': total_mint,
    }

    status = 'PASS' if classification != 'NORMAL' else 'NEUTRAL'
    return status, score, details
```

### Thresholds
```yaml
M73_ENABLED: true
M73_LARGE_MINT_THRESH: 500000000    # $500M single-day mint
M73_MEGA_MINT_THRESH: 1000000000    # $1B single-day mint
M73_LARGE_BURN_THRESH: 500000000    # $500M burn
```

### ICS Weight
```yaml
M73_WEIGHT: 0.05  # lower — on-chain data is noisy, 12-48h lag
```

---

<a name="integration"></a>
## Integration — Scanner Wiring

### In `scripts/scanner.py`, add to the imports section:

```python
from src.modules.m66_usdjpy import score_m66_usdjpy, fetch_usdjpy
from src.modules.m67_dxy import score_m67_dxy, fetch_dxy
from src.modules.m68_yield import score_m68_yield, fetch_10y_yield, fetch_tips_yield
from src.modules.m69_vix import score_m69_vix, fetch_vix
from src.modules.m70_wti import score_m70_wti, fetch_wti
from src.modules.m71_gold import score_m71_gold, fetch_gold
from src.modules.m72_btcdom import score_m72_btcdom, fetch_btcdom
from src.modules.m73_stablecoin import score_m73_stablecoin, fetch_stablecoin_mints
```

### In `scan_signal()`, add after the M62 block (before M22 aggregator):

```python
# ── M66: USD/JPY Carry Trade Proxy ──
m66_score = 0.5
m66_status = 'SKIP'
if cfg.get('M66_ENABLED', True):
    try:
        df_usdjpy = fetch_usdjpy()
        df_dxy_for_m66 = fetch_dxy(period="5d", interval="15m")
        m66_status, m66_score, m66_details = score_m66_usdjpy(
            df_usdjpy, df_dxy_for_m66, direction, config=cfg)
        if m66_status == 'PASS':
            result['m66'] = {'status': m66_status, 'score': round(m66_score, 3),
                             'details': m66_details}
    except Exception as e:
        result['m66'] = {'status': 'ERROR', 'score': 0.5, 'error': str(e)}

# ── M67: DXY Divergence ──
# (reuse DXY data fetched for M66 if available)
# ... similar pattern ...

# ── M68: 10Y Yield + TIPS ──
# ── M69: VIX ──
# ── M70: WTI Oil ──
# ── M71: Gold ──
# ── M72: BTC Dominance ──
# ── M73: Stablecoin Mints ──
# (same pattern for each)
```

### ICS Calculation Update

In `calc_ics()` call, add each new module:

```python
ics, effective_floor = calc_ics(
    ...existing params...,
    m66_score=m66_score, use_m66=m66_status == 'PASS',
    m67_score=m67_score, use_m67=m67_status == 'PASS',
    m68_score=m68_score, use_m68=m68_status == 'PASS',
    m69_score=m69_score, use_m69=m69_status == 'PASS',
    m70_score=m70_score, use_m70=m70_status == 'PASS',
    m71_score=m71_score, use_m71=m71_status == 'PASS',
    m72_score=m72_score, use_m72=m72_status == 'PASS',
    m73_score=m73_score, use_m73=m73_status == 'PASS',
)
```

### Fetch Optimization

All FX/commodity/VIX data comes from yfinance. Batch the fetches to avoid
rate limits:

```python
def fetch_all_macro_fx(config=None):
    """Fetch all traditional-finance data in one batched call."""
    cfg = config or CONFIG
    data = {}

    tickers = {}
    if cfg.get('M66_ENABLED'): tickers['usdjpy'] = ('JPY=X', '5d', '1m')
    if cfg.get('M67_ENABLED'): tickers['dxy'] = ('DX-Y.NYB', '5d', '15m')
    if cfg.get('M68_ENABLED'): tickers['tnx'] = ('^TNX', '5d', '1h')
    if cfg.get('M69_ENABLED'): tickers['vix'] = ('^VIX', '5d', '1d')
    if cfg.get('M70_ENABLED'): tickers['wti'] = ('CL=F', '5d', '4h')
    if cfg.get('M71_ENABLED'): tickers['gold'] = ('GC=F', '5d', '4h')

    import yfinance as yf
    for key, (ticker, period, interval) in tickers.items():
        try:
            data[key] = yf.download(ticker, period=period,
                                     interval=interval, progress=False)
        except Exception:
            data[key] = None

    return data
```

---

<a name="config"></a>
## Config Additions for `config/settings.yaml`

```yaml
# ─── M66: USD/JPY CARRY TRADE ───────────────────────────────
M66_ENABLED: true
M66_WEIGHT: 0.08
M66_ALERT_THRESH: -0.3
M66_CONFIRMED_THRESH: -0.8
M66_CARRY_ALERT_SCORE: 0.65
M66_CARRY_CONFIRMED_SCORE: 0.80

# ─── M67: DXY DIVERGENCE ────────────────────────────────────
M67_ENABLED: true
M67_WEIGHT: 0.06
M67_DXY_THRESH: 0.2
M67_DIVERGENCE_ETH_THRESH: 0.05

# ─── M68: 10Y YIELD + TIPS ──────────────────────────────────
M68_ENABLED: true
M68_WEIGHT: 0.10
M68_ALERT_BPS: 5.0
M68_EXTREME_BPS: 10.0

# ─── M69: VIX REGIME ────────────────────────────────────────
M69_ENABLED: true
M69_WEIGHT: 0.08
M69_COMPLACENT_THRESH: 15.0
M69_ELEVATED_THRESH: 20.0
M69_FEAR_THRESH: 30.0
M69_CRISIS_THRESH: 40.0
M69_SPIKE_DELTA: 3.0

# ─── M70: WTI CRUDE OIL ─────────────────────────────────────
M70_ENABLED: true
M70_WEIGHT: 0.05
M70_ALERT_THRESH: 3.0

# ─── M71: GOLD + DXY GEOPOLITICAL ───────────────────────────
M71_ENABLED: true
M71_WEIGHT: 0.06
M71_ALERT_THRESH: 1.0

# ─── M72: BTC DOMINANCE ─────────────────────────────────────
M72_ENABLED: true
M72_WEIGHT: 0.10
M72_HIGH_THRESH: 55.0
M72_LOW_THRESH: 48.0

# ─── M73: STABLECOIN MINTS ──────────────────────────────────
M73_ENABLED: true
M73_WEIGHT: 0.05
M73_LARGE_MINT_THRESH: 500000000
M73_MEGA_MINT_THRESH: 1000000000
M73_LARGE_BURN_THRESH: 500000000
```

### ICS Weight Rebalance

Adding 8 new modules with total weight 0.58. Existing weights sum to ~0.65.
Two options:

**Option A — Normalize all weights to sum to 1.0:**
Multiply all existing weights by `0.42 / 0.65 ≈ 0.646` and add new weights.

**Option B — Keep existing weights, cap total at 1.20:**
New modules are mostly filters (low weight), so total ~1.23 is acceptable
if `calc_ics()` clamps the final score to [0, 1].

**Recommended: Option B** — the new modules are confidence modulators,
not primary signal contributors. Their weights should stay low.

---

## Dependency Note

All 8 modules require `yfinance` for FX/commodity/VIX data:

```
yfinance>=0.2.31
```

Add to `requirements.txt`. yfinance is a pure-Python wrapper over Yahoo Finance
— no native compilation, no API key needed, no security concerns.

For BTC.D and stablecoin flows, only `requests` is needed (already in requirements).

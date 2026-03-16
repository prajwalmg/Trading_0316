"""
================================================================
  data/pipeline.py  — FULL UNIVERSE EDITION
  Complete coverage across all asset classes

  Forex       : 46 pairs  (majors + minors + select exotics)
  Crypto      :  8 pairs  (top liquid coins, free on IBKR)
  Equities    : 12 tickers (US + EU, yfinance fallback)
  Commodities :  6 instruments

  Data source logic:
    IBKR Gateway (live)  →  primary for ALL assets
    yfinance             →  fallback + long training history
    Local parquet cache  →  offline & rate-limit protection

  IBKR pacing rules respected:
    0.5s gap between requests
    Max 50 simultaneous open requests
================================================================
"""

import os, time, queue, logging, hashlib, threading, warnings
from datetime import datetime
from typing   import Dict, List, Optional

import numpy  as np
import pandas as pd

try:
    import yfinance as yf
    _YF_AVAILABLE = True
except ImportError:
    _YF_AVAILABLE = False

warnings.filterwarnings("ignore")
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from config.settings import (
    ASSET_CLASS_MAP, GRANULARITY, MIN_BARS, DATA_CACHE_DIR,
    FOREX_PAIRS, EQUITY_TICKERS, COMMODITY_TICKERS, CRYPTO_TICKERS,
    IBKR_HOST, IBKR_PORT, IBKR_CLIENT_ID,
)

logger = logging.getLogger("trading_firm.data")

# ── Contract map helpers ───────────────────────────────────────
def _fx(base, quote):
    return {"symbol": base, "secType": "CASH", "currency": quote,
            "exchange": "IDEALPRO", "whatToShow": "MIDPOINT",
            "yf_ticker": f"{base}{quote}=X"}

def _crypto(sym):
    return {"symbol": sym, "secType": "CRYPTO", "currency": "USD",
            "exchange": "PAXOS", "whatToShow": "TRADES",
            "yf_ticker": f"{sym}-USD"}

def _stock(sym, primary="NASDAQ"):
    return {"symbol": sym, "secType": "STK", "currency": "USD",
            "exchange": "SMART", "primaryExch": primary,
            "whatToShow": "TRADES", "yf_ticker": sym}

def _cfd(sym, yf_t):
    return {"symbol": sym, "secType": "CFD", "currency": "USD",
            "exchange": "SMART", "whatToShow": "MIDPOINT", "yf_ticker": yf_t}


# ══════════════════════════════════════════════════════════════
#  MASTER CONTRACT MAP
# ══════════════════════════════════════════════════════════════
CONTRACT_MAP: Dict[str, dict] = {

    # ── USD Majors ────────────────────────────────────────────
    "EURUSD=X":  _fx("EUR","USD"),
    "GBPUSD=X":  _fx("GBP","USD"),
    "USDJPY=X":  _fx("USD","JPY"),
    "USDCHF=X":  _fx("USD","CHF"),
    "AUDUSD=X":  _fx("AUD","USD"),
    "NZDUSD=X":  _fx("NZD","USD"),
    "USDCAD=X":  _fx("USD","CAD"),

    # ── USD Minors / Exotics ──────────────────────────────────
    "USDNOK=X":  _fx("USD","NOK"),
    "USDSEK=X":  _fx("USD","SEK"),
    "USDDKK=X":  _fx("USD","DKK"),
    "USDSGD=X":  _fx("USD","SGD"),
    "USDHKD=X":  _fx("USD","HKD"),
    "USDMXN=X":  _fx("USD","MXN"),
    "USDTRY=X":  _fx("USD","TRY"),
    "USDZAR=X":  _fx("USD","ZAR"),
    "USDPLN=X":  _fx("USD","PLN"),
    "USDCZK=X":  _fx("USD","CZK"),
    "USDHUF=X":  _fx("USD","HUF"),
    "USDCNH=X":  _fx("USD","CNH"),

    # ── EUR Crosses ───────────────────────────────────────────
    "EURGBP=X":  _fx("EUR","GBP"),
    "EURJPY=X":  _fx("EUR","JPY"),
    "EURCHF=X":  _fx("EUR","CHF"),
    "EURAUD=X":  _fx("EUR","AUD"),
    "EURCAD=X":  _fx("EUR","CAD"),
    "EURNZD=X":  _fx("EUR","NZD"),
    "EURNOK=X":  _fx("EUR","NOK"),
    "EURSEK=X":  _fx("EUR","SEK"),
    "EURPLN=X":  _fx("EUR","PLN"),
    "EURHUF=X":  _fx("EUR","HUF"),
    "EURTRY=X":  _fx("EUR","TRY"),

    # ── GBP Crosses ───────────────────────────────────────────
    "GBPJPY=X":  _fx("GBP","JPY"),
    "GBPCHF=X":  _fx("GBP","CHF"),
    "GBPAUD=X":  _fx("GBP","AUD"),
    "GBPCAD=X":  _fx("GBP","CAD"),
    "GBPNZD=X":  _fx("GBP","NZD"),

    # ── AUD / NZD Crosses ─────────────────────────────────────
    "AUDJPY=X":  _fx("AUD","JPY"),
    "AUDCHF=X":  _fx("AUD","CHF"),
    "AUDCAD=X":  _fx("AUD","CAD"),
    "AUDNZD=X":  _fx("AUD","NZD"),
    "NZDJPY=X":  _fx("NZD","JPY"),
    "NZDCHF=X":  _fx("NZD","CHF"),
    "NZDCAD=X":  _fx("NZD","CAD"),

    # ── CAD / CHF Crosses ─────────────────────────────────────
    "CADJPY=X":  _fx("CAD","JPY"),
    "CADCHF=X":  _fx("CAD","CHF"),
    "CHFJPY=X":  _fx("CHF","JPY"),

    # ══════════════════════════════════════════════════════════
    #  CRYPTO — FREE, no subscription
    #  Enable: Client Portal → Account Settings → Trading Permissions
    # ══════════════════════════════════════════════════════════
    "BTC-USD":   _crypto("BTC"),
    "ETH-USD":   _crypto("ETH"),
    "SOL-USD":   _crypto("SOL"),
    "XRP-USD":   _crypto("XRP"),
    "BNB-USD":   _crypto("BNB"),
    "ADA-USD":   _crypto("ADA"),
    "AVAX-USD":  _crypto("AVAX"),
    "MATIC-USD": _crypto("MATIC"),

    # ══════════════════════════════════════════════════════════
    #  EQUITIES — IBKR needs subscription; yfinance covers all
    # ══════════════════════════════════════════════════════════
    "NVDA":  _stock("NVDA",  "NASDAQ"),
    "AAPL":  _stock("AAPL",  "NASDAQ"),
    "MSFT":  _stock("MSFT",  "NASDAQ"),
    "GOOGL": _stock("GOOGL", "NASDAQ"),
    "META":  _stock("META",  "NASDAQ"),
    "AMZN":  _stock("AMZN",  "NASDAQ"),
    "TSLA":  _stock("TSLA",  "NASDAQ"),
    "GS":    _stock("GS",    "NYSE"),
    "JPM":   _stock("JPM",   "NYSE"),
    "SPY":   _stock("SPY",   "ARCA"),
    "EWG":   _stock("EWG",   "ARCA"),
    "SAP":   _stock("SAP",   "NYSE"),

    # ══════════════════════════════════════════════════════════
    #  COMMODITIES — IBKR needs subscription; yfinance covers all
    # ══════════════════════════════════════════════════════════
    "GC=F":  _cfd("XAUUSD", "GC=F"),
    "SI=F":  _cfd("XAGUSD", "SI=F"),
    "NG=F":  _cfd("NATGAS", "NG=F"),
    "HG=F":  _cfd("COPPER", "HG=F"),
    "CL=F":  _cfd("CRUDE",  "CL=F"),
    "ZW=F":  _cfd("WHEAT",  "ZW=F"),
}

BAR_SIZE_MAP = {"1m":"1 min","5m":"5 mins","15m":"15 mins",
                "30m":"30 mins","1h":"1 hour","4h":"4 hours","1d":"1 day"}
DURATION_MAP = {"1m":"7 D","5m":"30 D","15m":"60 D","30m":"60 D",
                "1h":"1 Y","4h":"2 Y","1d":"5 Y"}
YF_PERIOD_MAP = {"1m":"7d","5m":"60d","15m":"60d","30m":"60d",
                 "1h":"2y","4h":"2y","1d":"max"}   # max history for training
CACHE_TTL_LIVE = 60
CACHE_TTL_TRAIN = 1440


# ── Cache ──────────────────────────────────────────────────────
def _cache_path(ticker, interval):
    key  = hashlib.md5(f"{ticker}{interval}".encode()).hexdigest()[:8]
    name = ticker.replace("=","").replace("/","").replace("-","")
    os.makedirs(DATA_CACHE_DIR, exist_ok=True)
    return os.path.join(DATA_CACHE_DIR, f"{name}_{interval}_{key}.parquet")

def _cache_valid(path, ttl):
    if not os.path.exists(path): return False
    return (datetime.now().timestamp()-os.path.getmtime(path))/60 < ttl

def _save(df, path):
    try: df.to_parquet(path)
    except Exception as e: logger.warning(f"Cache save: {e}")

def _load(path):
    try: return pd.read_parquet(path)
    except: return pd.DataFrame()


# ── Clean ──────────────────────────────────────────────────────
def _clean(df, ticker):
    if df.empty: return df
    df = df[~df.index.duplicated(keep="first")].sort_index()
    df = df.ffill(limit=3)
    rm = df["close"].rolling(20,min_periods=5).mean()
    rs = df["close"].rolling(20,min_periods=5).std()
    df = df[((df["close"]-rm).abs()/(rs+1e-9)) <= 5]
    df = df[~((df["high"]<df["low"])|(df["close"]>df["high"]*1.01))]
    if ASSET_CLASS_MAP.get(ticker)=="equity" and "volume" in df.columns:
        df = df[df["volume"]>0]
    return df.dropna(subset=["open","high","low","close"])


# ── IBKR singleton ─────────────────────────────────────────────
# _ibkr_app is one of:
#   None        → not yet attempted
#   "offline"   → attempted once, Gateway not running (skip all retries)
#   App()       → live connected instance
_ibkr_app, _ibkr_lock, _last_req = None, threading.Lock(), 0.0

def _get_ibkr():
    global _ibkr_app
    with _ibkr_lock:
        # Already confirmed offline — skip silently, no retry, no warning
        if _ibkr_app == "offline":
            return None
        # Already connected — return immediately
        if _ibkr_app and _ibkr_app._connected:
            return _ibkr_app
        try:
            from ibapi.client  import EClient
            from ibapi.wrapper import EWrapper
            class App(EWrapper, EClient):
                def __init__(self):
                    EClient.__init__(self, self)
                    self._connected=False
                    self._hq:Dict[int,queue.Queue]={}
                    self._pq:Dict[int,queue.Queue]={}
                    self._lk=threading.Lock(); self._rid=2000
                def _n(self):
                    with self._lk: r=self._rid; self._rid+=1; return r
                def nextValidId(self,_): self._connected=True
                def error(self,rid,code,msg,_=""):
                    # Suppress: informational, offline (502), no data
                    if code in(2104,2106,2158,2119,2100,2103,2105,502): return
                    if code in(162,200,354,10089,10090):
                        if rid in self._hq: self._hq[rid].put(None)
                        return
                    logger.warning(f"IBKR[{code}] r={rid}: {msg}")
                    if rid in self._hq: self._hq[rid].put(None)
                def historicalData(self,rid,bar):
                    if rid in self._hq:
                        self._hq[rid].put({"date":bar.date,"open":float(bar.open),
                            "high":float(bar.high),"low":float(bar.low),
                            "close":float(bar.close),"volume":float(bar.volume)})
                def historicalDataEnd(self,rid,*_):
                    if rid in self._hq: self._hq[rid].put(None)
                def tickPrice(self,rid,tt,price,_):
                    if rid in self._pq: self._pq[rid].put((tt,price))
                def tickSnapshotEnd(self,rid):
                    if rid in self._pq: self._pq[rid].put((-1,-1))
            app=App()
            app.connect(IBKR_HOST,IBKR_PORT,IBKR_CLIENT_ID+1)
            threading.Thread(target=app.run,daemon=True).start()
            for _ in range(50):
                if app._connected: break
                time.sleep(0.1)
            if not app._connected:
                # Mark as offline — no more retries or warnings this session
                _ibkr_app = "offline"
                logger.info("IB Gateway offline → yfinance fallback active for this session")
                return None
            _ibkr_app=app
            logger.info(f"IBKR data app connected | port={IBKR_PORT}")
            return app
        except ImportError:
            _ibkr_app = "offline"
            logger.warning("ibapi not installed — pip install ibapi")
            return None
        except Exception as e:
            _ibkr_app = "offline"
            logger.debug(f"IBKR connect skipped: {e}")
            return None


def _make_contract(params):
    from ibapi.contract import Contract
    c=Contract(); c.symbol=params["symbol"]; c.secType=params["secType"]
    c.currency=params["currency"]; c.exchange=params["exchange"]
    if "primaryExch" in params: c.primaryExch=params["primaryExch"]
    return c

def _parse_date(d):
    d=str(d).split(" US/")[0].split(" America/")[0].strip()
    for fmt in("%Y%m%d %H:%M:%S","%Y%m%d"):
        try: return datetime.strptime(d,fmt)
        except: pass
    return None


# ── IBKR fetch ─────────────────────────────────────────────────
def _fetch_ibkr(ticker, interval="1h"):
    global _last_req
    app=_get_ibkr()
    if not app: return pd.DataFrame()
    params=CONTRACT_MAP.get(ticker)
    if not params: return pd.DataFrame()
    gap=time.time()-_last_req
    if gap<0.5: time.sleep(0.5-gap)
    try:
        rid=app._n(); q=queue.Queue(); app._hq[rid]=q
        app.reqHistoricalData(rid,_make_contract(params),"",
            DURATION_MAP.get(interval,"1 Y"),BAR_SIZE_MAP.get(interval,"1 hour"),
            params.get("whatToShow","MIDPOINT"),0,1,False,[])
        _last_req=time.time()
        bars=[]; deadline=time.time()+45
        while time.time()<deadline:
            try:
                b=q.get(timeout=1.0)
                if b is None: break
                bars.append(b)
            except queue.Empty: continue
    except Exception as e:
        logger.error(f"{ticker}: IBKR error — {e}"); return pd.DataFrame()
    finally: app._hq.pop(rid,None)
    if not bars: return pd.DataFrame()
    df=pd.DataFrame(bars)
    df["time"]=df["date"].apply(_parse_date)
    df=df.dropna(subset=["time"]).set_index("time")
    df=df[["open","high","low","close","volume"]].astype(float)
    df.index.name="time"
    df=df.sort_index()[~df.index.duplicated(keep="first")]
    logger.info(f"{ticker}(IBKR): {len(df)} {interval} | {df.index[0].date()}→{df.index[-1].date()}")
    return df


# ── yfinance fallback ──────────────────────────────────────────
def _fetch_yf(ticker, interval="1h"):
    if not _YF_AVAILABLE: return pd.DataFrame()
    yf_sym=CONTRACT_MAP.get(ticker,{}).get("yf_ticker",ticker)
    period=YF_PERIOD_MAP.get(interval,"2y")
    if interval in("1m","5m","15m","30m"): period="60d"
    try:
        raw=yf.download(yf_sym,interval=interval,period=period,
                        progress=False,auto_adjust=True,multi_level_index=False)
        if raw.empty: return pd.DataFrame()
        if hasattr(raw.columns,"levels"): raw.columns=raw.columns.get_level_values(0)
        raw.columns=[str(c).lower().strip() for c in raw.columns]
        raw=raw.rename(columns={"adj close":"close","adjclose":"close"})
        cols=[c for c in["open","high","low","close","volume"] if c in raw.columns]
        df=raw[cols].copy()
        if "volume" not in df.columns: df["volume"]=0.0
        df.index=pd.to_datetime(df.index)
        if hasattr(df.index,"tz") and df.index.tz: df.index=df.index.tz_localize(None)
        df.index.name="time"
        logger.info(f"{ticker}(yf): {len(df)} {interval} | {df.index[0].date()}→{df.index[-1].date()}")
        return df
    except Exception as e:
        logger.error(f"{ticker}: yfinance — {e}"); return pd.DataFrame()


# ── Public API ─────────────────────────────────────────────────
def fetch_ohlcv(ticker, interval=GRANULARITY, days=365,
                use_cache=True, training_mode=False):
    ttl=CACHE_TTL_TRAIN if training_mode else CACHE_TTL_LIVE
    cache=_cache_path(ticker,interval)
    if use_cache and _cache_valid(cache,ttl):
        df=_load(cache)
        if not df.empty: return df
    df=_fetch_ibkr(ticker,interval)
    if df.empty: df=_fetch_yf(ticker,interval)
    if df.empty:
        if os.path.exists(cache):
            logger.warning(f"{ticker}: stale cache fallback")
            return _load(cache)
        return pd.DataFrame()
    df=_clean(df,ticker)
    if not df.empty: _save(df,cache)
    return df


def fetch_live_price(ticker):
    app=_get_ibkr()
    if app:
        params=CONTRACT_MAP.get(ticker)
        if params:
            try:
                rid=app._n(); q=queue.Queue(); app._pq[rid]=q
                app.reqMktData(rid,_make_contract(params),"",True,False,[])
                bid=ask=0.0; deadline=time.time()+4
                while time.time()<deadline:
                    try:
                        tt,p=q.get(timeout=0.3)
                        if tt==-1: break
                        if tt==1 and p>0: bid=p
                        if tt==2 and p>0: ask=p
                        if bid>0 and ask>0: break
                    except queue.Empty: break
                app.cancelMktData(rid); app._pq.pop(rid,None)
                if bid>0 or ask>0: return (bid+ask)/2 if bid and ask else max(bid,ask)
            except: pass
    cache=_cache_path(ticker,GRANULARITY)
    if os.path.exists(cache):
        df=_load(cache)
        if not df.empty: return float(df["close"].iloc[-1])
    return 0.0


def fetch_macro_context(use_cache=True):
    macro={}
    for name,sym in[("vix","^VIX"),("dxy","DX-Y.NYB")]:
        df=_fetch_yf(sym,"1d")
        macro[name]=df["close"] if not df.empty else None
    return macro


def check_data_sources():
    app  = _get_ibkr()
    ibkr = app is not None and app != "offline" and hasattr(app, "_connected") and app._connected
    status = {"ibkr_gateway": ibkr, "yfinance": _YF_AVAILABLE}
    logger.info(f"Data: IBKR={'✅' if ibkr else '❌(yf fallback)'}  yfinance={'✅' if _YF_AVAILABLE else '❌'}")
    return status


class DataPipeline:
    def __init__(self): self._store:Dict[str,pd.DataFrame]={}

    def refresh_all(self,tickers=None,interval=GRANULARITY,days=365,
                    use_cache=True,training_mode=False):
        tickers=tickers or(FOREX_PAIRS+EQUITY_TICKERS+COMMODITY_TICKERS+CRYPTO_TICKERS)
        failed=[]
        logger.info(f"Refreshing {len(tickers)} instruments | {interval} | train={training_mode}")
        for t in tickers:
            df=fetch_ohlcv(t,interval=interval,days=days,
                           use_cache=use_cache,training_mode=training_mode)
            if df.empty: failed.append(t)
            else: self._store[t]=df
        if failed: logger.warning(f"Failed: {failed}")
        logger.info(f"Pipeline ready: {len(self._store)} loaded | {len(failed)} failed")
        return self._store

    def get(self,ticker): return self._store.get(ticker,pd.DataFrame())
    def get_latest_price(self,ticker): return fetch_live_price(ticker)
    def refresh_one(self,ticker,interval=GRANULARITY):
        df=fetch_ohlcv(ticker,interval=interval,use_cache=False)
        if not df.empty: self._store[ticker]=df
        return df
    def available(self): return[t for t,df in self._store.items() if not df.empty]

    def summary(self):
        print(f"\n{'─'*72}")
        print(f"  DataPipeline — {len(self._store)} instruments")
        print(f"{'─'*72}")
        for t,df in sorted(self._store.items()):
            a=ASSET_CLASS_MAP.get(t,"?")
            if df.empty: print(f"  {t:<18} {a:<12}  NO DATA")
            else: print(f"  {t:<18} {a:<12}  {len(df):>6} bars  "
                        f"{df.index[0].date()} → {df.index[-1].date()}")
        print(f"{'─'*72}\n")

    @property
    def store(self): return self._store


# Backward compatibility
def fetch_yfinance(ticker,interval=GRANULARITY,period="730d",use_cache=True):
    return _fetch_yf(ticker,interval)
def update_intraday_cache(pair,yf_ticker=None,is_crypto=False):
    return fetch_ohlcv(pair,interval="5m",use_cache=False)
def load_histdata_csv(path_pattern,resample_to="5min"):
    import glob as _g
    files=sorted(_g.glob(path_pattern))
    if not files: raise FileNotFoundError(f"No CSV: {path_pattern}")
    dfs=[]
    for f in files:
        try:
            tmp=pd.read_csv(f,sep="\t",header=None,
                names=["date","time","open","high","low","close","volume"])
            tmp.index=pd.to_datetime(tmp["date"]+" "+tmp["time"],format="%Y.%m.%d %H:%M")
            dfs.append(tmp.drop(columns=["date","time"]).astype(float))
        except Exception as e: logger.warning(f"Histdata {f}: {e}")
    if not dfs: return pd.DataFrame()
    df=pd.concat(dfs).sort_index()[lambda x:~x.index.duplicated(keep="first")]
    if resample_to and resample_to!="1min":
        df=df.resample(resample_to).agg({"open":"first","high":"max",
            "low":"min","close":"last","volume":"sum"}).dropna()
    return df
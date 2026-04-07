"""data/dukascopy.py — Dukascopy tick data."""
import os,time,struct,logging,requests,lzma
import pandas as pd
from datetime import datetime,timedelta
logger=logging.getLogger("trading_firm.dukascopy")

CACHE_DIR="data/cache/dukascopy"
SYMBOL_MAP={
    "EURUSD=X":"EURUSD","GBPUSD=X":"GBPUSD",
    "USDJPY=X":"USDJPY","USDCHF=X":"USDCHF",
    "AUDUSD=X":"AUDUSD","NZDUSD=X":"NZDUSD",
    "USDCAD=X":"USDCAD","GBPJPY=X":"GBPJPY",
    "EURJPY=X":"EURJPY","EURGBP=X":"EURGBP",
    "EURCHF=X":"EURCHF","EURAUD=X":"EURAUD",
    "EURCAD=X":"EURCAD","GBPCHF=X":"GBPCHF",
    "AUDJPY=X":"AUDJPY","CADJPY=X":"CADJPY",
    "CHFJPY=X":"CHFJPY","NZDJPY=X":"NZDJPY",
    "GC=F":"XAUUSD","SI=F":"XAGUSD","CL=F":"USOUSD",
}
POINT_MAP={
    "EURUSD":0.00001,"GBPUSD":0.00001,"USDJPY":0.001,
    "USDCHF":0.00001,"AUDUSD":0.00001,"NZDUSD":0.00001,
    "USDCAD":0.00001,"GBPJPY":0.001,"EURJPY":0.001,
    "EURGBP":0.00001,"EURCHF":0.00001,"EURAUD":0.00001,
    "EURCAD":0.00001,"GBPCHF":0.00001,"AUDJPY":0.001,
    "CADJPY":0.001,"CHFJPY":0.001,"NZDJPY":0.001,
    "XAUUSD":0.001,"XAGUSD":0.001,"USOUSD":0.001,
}

def fetch_hour(symbol,dt):
    url=(f"https://datafeed.dukascopy.com/datafeed/"
         f"{symbol}/{dt.year:04d}/{dt.month-1:02d}/"
         f"{dt.day:02d}/{dt.hour:02d}h_ticks.bi5")
    try:
        r=requests.get(url,timeout=20,
            headers={"User-Agent":"Mozilla/5.0"})
        if r.status_code==404 or not r.content:
            return pd.DataFrame()
        raw=lzma.decompress(r.content)
        if len(raw)%20!=0: return pd.DataFrame()
        pt=POINT_MAP.get(symbol,0.00001)
        ticks=[]
        for i in range(0,len(raw),20):
            ms,ask,bid,av,bv=struct.unpack(">IIIff",raw[i:i+20])
            ticks.append({"time":dt+timedelta(milliseconds=ms),
                "ask":ask*pt,"bid":bid*pt,"av":av,"bv":bv})
        return pd.DataFrame(ticks).set_index("time")
    except Exception:
        return pd.DataFrame()

def fetch_ohlcv(ticker,start,end,interval="1h"):
    sym=SYMBOL_MAP.get(ticker)
    if sym is None: return pd.DataFrame()
    key=(f"{sym}_{interval}_"
         f"{start.strftime('%Y%m%d')}_"
         f"{end.strftime('%Y%m%d')}")
    cache=f"{CACHE_DIR}/{key}.parquet"
    os.makedirs(CACHE_DIR,exist_ok=True)
    if os.path.exists(cache):
        return pd.read_parquet(cache)
    all_ticks=[]
    cur=start.replace(minute=0,second=0,microsecond=0)
    n=0
    while cur<end:
        chunk=fetch_hour(sym,cur)
        if not chunk.empty: all_ticks.append(chunk)
        cur+=timedelta(hours=1); n+=1
        if n%10==0: time.sleep(0.05)
    if not all_ticks: return pd.DataFrame()
    ticks=pd.concat(all_ticks).sort_index()
    ticks["mid"]=(ticks["ask"]+ticks["bid"])/2
    freq={"1m":"1min","5m":"5min","15m":"15min",
          "1h":"1h","4h":"4h","1d":"1D"}.get(interval,"1h")
    ohlcv=ticks["mid"].resample(freq).ohlc()
    ohlcv["volume"]=(ticks["av"]+ticks["bv"]).resample(freq).sum()
    ohlcv=ohlcv.dropna(); ohlcv.index.name="time"
    if not ohlcv.empty: ohlcv.to_parquet(cache)
    return ohlcv

def fetch_5yr(ticker, interval="1h"):
    end   = datetime.now()
    start = end - timedelta(days=365*5)
    # Check if a cached file exists with similar dates
    # and return it directly to avoid re-fetching
    import os, glob
    sym = SYMBOL_MAP.get(ticker, ticker)
    pattern = f"data/cache/dukascopy/{sym}_{interval}_*.parquet"
    files = sorted(glob.glob(pattern))
    if files:
        # Use the largest cached file (most complete)
        largest = max(files, key=os.path.getsize)
        df = pd.read_parquet(largest)
        logger.info(f"{ticker}(Dukascopy cache): "
                    f"{len(df)} bars from {largest}")
        return df
    return fetch_ohlcv(ticker, start, end, interval)



"""data/cot.py — CFTC COT feature loader."""
import os, time, logging, requests, zipfile, io
import numpy as np
import pandas as pd
from datetime import datetime
logger = logging.getLogger("trading_firm.cot")

CACHE_DIR   = "data/cache/cot"
CACHE_TTL_H = 168

MARKET_CODES = {
    "EURUSD=X":"099741","GBPUSD=X":"096742",
    "USDJPY=X":"097741","USDCHF=X":"092741",
    "AUDUSD=X":"232741","NZDUSD=X":"112741",
    "USDCAD=X":"090741","GC=F":"088691",
    "CL=F":"067651","SI=F":"084691",
    "NG=F":"023651","HG=F":"085692",
}

COT_COLS = {
    "Market_and_Exchange_Names":"market",
    "As_of_Date_In_Form_YYMMDD":"date",
    "CFTC_Market_Code":"market_code",
    "NonComm_Positions_Long_All":"nc_long",
    "NonComm_Positions_Short_All":"nc_short",
    "Comm_Positions_Long_All":"comm_long",
    "Comm_Positions_Short_All":"comm_short",
    "Open_Interest_All":"open_interest",
}

def _cache_valid(path):
    if not os.path.exists(path): return False
    return (datetime.now().timestamp()
            -os.path.getmtime(path))/3600 < CACHE_TTL_H

def fetch_cot_year(year):
    url=(f"https://www.cftc.gov/files/dea/history/"
         f"fut_fin_xls_{year}.zip")
    try:
        r=requests.get(url,timeout=60)
        if r.status_code!=200: return pd.DataFrame()
        z=zipfile.ZipFile(io.BytesIO(r.content))
        fname=[f for f in z.namelist()
               if f.endswith(".xls") or
               f.endswith(".xlsx") or
               f.endswith(".csv")][0]
        with z.open(fname) as f:
            df=(pd.read_csv(f,low_memory=False)
                if fname.endswith(".csv")
                else pd.read_excel(f))
        avail={c:v for c,v in COT_COLS.items()
               if c in df.columns}
        df=df[list(avail.keys())].rename(columns=avail)
        df["date"]=pd.to_datetime(
            df["date"].astype(str),
            format="%y%m%d",errors="coerce")
        df=df.dropna(subset=["date"])
        return df.set_index("date").sort_index()
    except Exception as e:
        logger.error(f"COT {year}: {e}")
        return pd.DataFrame()

def load_cot(years=5):
    os.makedirs(CACHE_DIR,exist_ok=True)
    cache=f"{CACHE_DIR}/cot_{years}yr.parquet"
    if _cache_valid(cache): return pd.read_parquet(cache)
    dfs=[]
    for yr in range(datetime.now().year-years,
                    datetime.now().year+1):
        df=fetch_cot_year(yr)
        if not df.empty: dfs.append(df)
        time.sleep(0.5)
    if not dfs: return pd.DataFrame()
    out=pd.concat(dfs)
    out=out[~out.index.duplicated(keep="last")].sort_index()
    out.to_parquet(cache)
    return out

def get_cot_features(ticker, df_prices):
    code=MARKET_CODES.get(ticker)
    cols=["cot_nc_z","cot_comm_z",
          "cot_nc_extreme_long","cot_nc_extreme_short"]
    if code is None:
        for c in cols: df_prices[c]=0.0
        return df_prices
    cot=load_cot()
    if cot.empty:
        for c in cols: df_prices[c]=0.0
        return df_prices
    inst=cot[cot["market_code"].astype(str)
             .str.strip()==code].copy()
    if inst.empty:
        for c in cols: df_prices[c]=0.0
        return df_prices
    inst["nc_net"]=inst["nc_long"]-inst["nc_short"]
    inst["comm_net"]=inst["comm_long"]-inst["comm_short"]
    inst["nc_net_pct"]=inst["nc_net"]/inst["open_interest"].clip(1)
    inst["comm_net_pct"]=inst["comm_net"]/inst["open_interest"].clip(1)
    r1=inst["nc_net_pct"].rolling(52)
    inst["cot_nc_z"]=(inst["nc_net_pct"]-r1.mean())/(r1.std()+1e-9)
    r2=inst["comm_net_pct"].rolling(52)
    inst["cot_comm_z"]=(inst["comm_net_pct"]-r2.mean())/(r2.std()+1e-9)
    inst["cot_nc_extreme_long"]=(inst["cot_nc_z"]>1.5).astype(float)
    inst["cot_nc_extreme_short"]=(inst["cot_nc_z"]<-1.5).astype(float)
    inst_h=inst[cols].resample("1h").ffill()
    inst_a=inst_h.reindex(df_prices.index,method="ffill")
    for c in cols: df_prices[c]=inst_a[c].fillna(0).values
    return df_prices

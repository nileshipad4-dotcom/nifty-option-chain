import streamlit as st
import pandas as pd
import requests, base64
from datetime import datetime, timedelta
import pytz
from streamlit_autorefresh import st_autorefresh
from kiteconnect import KiteConnect

# ==================================================
# STREAMLIT
# ==================================================
st.set_page_config(layout="wide")
st.title("📊 AUTO Option Chain Collector (KITE + DHAN)")
st_autorefresh(interval=180_000, key="auto_refresh")

IST = pytz.timezone("Asia/Kolkata")

# ==================================================
# GITHUB
# ==================================================
TOKEN = st.secrets["GITHUB_TOKEN"]
KITE_REPO = st.secrets["KITE_REPO"]
DHAN_REPO = st.secrets["DHAN_REPO"]
BRANCH = st.secrets.get("GITHUB_BRANCH", "main")

GH_HEADERS = {
    "Authorization": f"token {TOKEN}",
    "Accept": "application/vnd.github+json"
}

# ==================================================
# ================= KITE ===========================
# ==================================================
KITE_API_KEY = "bkgv59vaazn56c42"
KITE_ACCESS_TOKEN = "IO4wzhuRsIeusChlrimjw8VZgwA3W10T"

STOCKS = [
    "360ONE","ABB","ABCAPITAL","ADANIENSOL","ADANIENT","ADANIGREEN","ADANIPORTS","ALKEM",
    "AMBER","AMBUJACEM","ANGELONE","APLAPOLLO","APOLLOHOSP","ASHOKLEY","ASIANPAINT","ASTRAL",
    "AUBANK","AUROPHARMA","AXISBANK","BAJAJ-AUTO","BAJAJFINSV","BAJAJHLDNG","BAJFINANCE",
    "BANDHANBNK","BANKBARODA","BANKINDIA","BDL","BEL","BHARATFORG","BHARTIARTL",
    "BHEL","BIOCON","BLUESTARCO","BOSCHLTD","BPCL","BRITANNIA","BSE","CAMS","CANBK","CDSL",
    "CGPOWER","CHOLAFIN","CIPLA","COALINDIA","COFORGE","COLPAL","CONCOR","CROMPTON","CUMMINSIND",
    "DABUR","DALBHARAT","DELHIVERY","DIVISLAB","DIXON","DLF","DMART","DRREDDY","EICHERMOT",
    "ETERNAL","EXIDEIND","FEDERALBNK","FORTIS","GAIL","GLENMARK","GMRAIRPORT",
    "GODREJCP","GODREJPROP","GRASIM","HAL","HAVELLS","HCLTECH","HDFCAMC","HDFCBANK","HDFCLIFE",
    "HEROMOTOCO","HINDALCO","HINDPETRO","HINDUNILVR","HINDZINC","HUDCO","ICICIBANK","ICICIGI",
    "ICICIPRULI","IDEA","IDFCFIRSTB","IEX","IIFL","INDHOTEL","INDIANB","INDIGO","INDUSINDBK",
    "INDUSTOWER","INFY","INOXWIND","IOC","IRCTC","IREDA","IRFC","ITC","JINDALSTEL","JIOFIN",
    "JSWENERGY","JSWSTEEL","JUBLFOOD","KALYANKJIL","KAYNES","KEI","KFINTECH","KOTAKBANK",
    "KPITTECH","LAURUSLABS","LICHSGFIN","LICI","LODHA","LT","LTF","LTIM","LUPIN","M&M",
    "MANAPPURAM","MANKIND","MARICO","MARUTI","MAXHEALTH","MAZDOCK","MCX","MFSL","MIDCPNIFTY",
    "MOTHERSON","MPHASIS","MUTHOOTFIN","NATIONALUM","NAUKRI","NBCC","NESTLEIND","NHPC", "NMDC","NTPC","NUVAMA","NYKAA","OBEROIRLTY","OFSS","OIL","ONGC",
    "PAGEIND","PATANJALI","PAYTM","PERSISTENT","PETRONET","PFC","PGEL","PHOENIXLTD",
    "PIDILITIND","PIIND","PNB","PNBHOUSING","POLICYBZR","POLYCAB","POWERGRID","POWERINDIA",
    "PPLPHARMA","PREMIERENE","PRESTIGE","RBLBANK","RECLTD","RELIANCE","RVNL","SAIL",
    "SAMMAANCAP","SBICARD","SBILIFE","SBIN","SHREECEM","SHRIRAMFIN","SIEMENS","SOLARINDS",
    "SONACOMS","SRF","SUNPHARMA","SUPREMEIND","SUZLON","SWIGGY","SYNGENE","TATACONSUM",
    "TATAELXSI","TATAPOWER","TATASTEEL","TATATECH","TCS","TECHM","TIINDIA","TITAN","TMPV",
    "TORNTPHARM","TORNTPOWER","TRENT","TVSMOTOR","ULTRACEMCO","UNIONBANK","UNITDSPR",
    "UNOMINDA","UPL","VBL","VEDL","VOLTAS","WAAREEENER","WIPRO","YESBANK","ZYDUSLIFE"
]
kite = KiteConnect(api_key=KITE_API_KEY)
kite.set_access_token(KITE_ACCESS_TOKEN)

@st.cache_data
def load_instruments():
    return pd.DataFrame(kite.instruments("NFO"))

inst = load_instruments()

def compute_max_pain(df):
    A, B = df["CE_LTP"].fillna(0), df["CE_OI"].fillna(0)
    G = df["Strike"]
    M, L = df["PE_LTP"].fillna(0), df["PE_OI"].fillna(0)
    mp = []
    for i in range(len(df)):
        v = (
            -sum(A[i:] * B[i:])
            + G.iloc[i]*sum(B[:i]) - sum(G[:i]*B[:i])
            - sum(M[:i]*L[:i])
            + sum(G[i:]*L[i:]) - G.iloc[i]*sum(L[i:])
        )
        mp.append(int(v/10000))
    df["Max_Pain"] = mp
    return df

def fetch_kite_full():
    option_map, symbols = {}, []
    for s in STOCKS:
        df = inst[(inst["name"]==s)&(inst["segment"]=="NFO-OPT")].copy()
        if df.empty: continue
        df["expiry"] = pd.to_datetime(df["expiry"])
        df = df[df["expiry"]==df["expiry"].min()]
        option_map[s] = df
        symbols += ["NFO:"+x for x in df["tradingsymbol"]]

    quotes = {}
    for i in range(0,len(symbols),200):
        quotes.update(kite.quote(symbols[i:i+200]))

    spot = kite.quote([f"NSE:{s}" for s in option_map])
    ts = datetime.now(IST).strftime("%Y-%m-%d %H:%M:%S")

    out = []
    for s,df in option_map.items():
        spot_q = spot.get(f"NSE:{s}", {})
        
        ohlc = spot_q.get("ohlc", {})
        ltp = spot_q.get("last_price")
        
        pc = ohlc.get("close")
        high = ohlc.get("high")
        low = ohlc.get("low")

        pct = ((ltp-pc)/pc*100) if ltp and pc else None

        rows=[]
        for k in sorted(df["strike"].unique()):
            ce = df[(df["strike"]==k)&(df["instrument_type"]=="CE")]
            pe = df[(df["strike"]==k)&(df["instrument_type"]=="PE")]
            ceq = quotes.get("NFO:"+ce.iloc[0]["tradingsymbol"],{}) if not ce.empty else {}
            peq = quotes.get("NFO:"+pe.iloc[0]["tradingsymbol"],{}) if not pe.empty else {}

            rows.append({
                "Stock":s,"Expiry":df["expiry"].iloc[0].date(),"Strike":k,
                "CE_LTP":ceq.get("last_price"),"CE_OI":ceq.get("oi"),"CE_Volume":ceq.get("volume"),
                "PE_LTP":peq.get("last_price"),"PE_OI":peq.get("oi"),"PE_Volume":peq.get("volume"),
                "Stock_LTP":ltp,"Stock_High":ohlc.get("high"),"Stock_Low":ohlc.get("low"),
                "Stock_%_Change":pct,"timestamp":ts
            })
        sdf = compute_max_pain(pd.DataFrame(rows).sort_values("Strike"))
        out.append(sdf)
    return pd.concat(out, ignore_index=True)

df_kite = fetch_kite_full()
kite_file = f"data/option_chain_{datetime.now(IST).strftime('%Y-%m-%d_%H-%M-%S')}.csv"

requests.put(
    f"https://api.github.com/repos/{KITE_REPO}/contents/{kite_file}",
    headers=GH_HEADERS,
    json={
        "message": f"KITE snapshot {kite_file}",
        "content": base64.b64encode(df_kite.to_csv(index=False).encode()).decode(),
        "branch": BRANCH
    }
)

# ==================================================
# ================= DHAN ===========================
# ==================================================
DHAN_CLIENT = "1102712380"
DHAN_TOKEN = "eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzUxMiJ9.eyJpc3MiOiJkaGFuIiwicGFydG5lcklkIjoiIiwiZXhwIjoxNzY4MDAyMjYzLCJpYXQiOjE3Njc5MTU4NjMsInRva2VuQ29uc3VtZXJUeXBlIjoiU0VMRiIsIndlYmhvb2tVcmwiOiIiLCJkaGFuQ2xpZW50SWQiOiIxMTAyNzEyMzgwIn0.OqpycJr1HOcBRtTgGyxh9rgS8moqvBL4dfT9AYmIvgeyUhy4mYjuTq9dfUACwH4lwwXvt9Jndb_383Q5An_4Cg"
DHAN_API = "https://api.dhan.co/v2"

DHAN_HEADERS = {
    "client-id": DHAN_CLIENT,
    "access-token": DHAN_TOKEN,
    "Content-Type": "application/json"
}

UNDERLYINGS = {
    "nifty":      {"scrip":13,"seg":"IDX_I","center":26000},
    "banknifty":  {"scrip":25,"seg":"IDX_I","center":60000},
    "midcpnifty": {"scrip":442,"seg":"IDX_I","center":13600},
    "sensex":     {"scrip":51,"seg":"IDX_I","center":84000},
}

BASE_COLS = [
 "Strike","CE LTP","CE OI","CE Volume","CE IV","CE Delta","CE Gamma","CE Vega",
 "PE LTP","PE OI","PE Volume","PE IV","PE Delta","PE Gamma","PE Vega",
 "timestamp","Max Pain"
]

def append_to_github(sym, df):
    path = f"data/{sym}.csv"
    url = f"https://api.github.com/repos/{DHAN_REPO}/contents/{path}"
    r = requests.get(url, headers=GH_HEADERS)
    if r.status_code==200:
        old = pd.read_csv(pd.compat.StringIO(base64.b64decode(r.json()["content"]).decode()))
        df = pd.concat([old,df],ignore_index=True)
        sha = r.json()["sha"]
    else:
        sha = None
    payload={
        "message":f"DHAN append {sym}",
        "content":base64.b64encode(df.to_csv(index=False).encode()).decode(),
        "branch":BRANCH
    }
    if sha: payload["sha"]=sha
    requests.put(url, headers=GH_HEADERS, json=payload)

for sym,cfg in UNDERLYINGS.items():
    exp = requests.post(f"{DHAN_API}/optionchain/expirylist",headers=DHAN_HEADERS,
        json={"UnderlyingScrip":cfg["scrip"],"UnderlyingSeg":cfg["seg"]}).json()["data"][0]

    oc = requests.post(f"{DHAN_API}/optionchain",headers=DHAN_HEADERS,
        json={"UnderlyingScrip":cfg["scrip"],"UnderlyingSeg":cfg["seg"],"Expiry":exp}
    ).json()["data"]["oc"]

    ts = datetime.now(IST).strftime("%Y-%m-%d %H:%M")
    rows=[]
    for k,v in oc.items():
        ce,pe=v.get("ce",{}),v.get("pe",{})
        rows.append({
            "Stock": s,
            "Expiry": df["expiry"].iloc[0].date(),
            "Strike": k,
        
            "CE_LTP": ceq.get("last_price"),
            "CE_OI": ceq.get("oi"),
            "CE_Volume": ceq.get("volume"),
        
            "PE_LTP": peq.get("last_price"),
            "PE_OI": peq.get("oi"),
            "PE_Volume": peq.get("volume"),
        
            "Stock_LTP": ltp,
            "Stock_High": high,
            "Stock_Low": low,
            "Stock_%_Change": ((ltp - pc) / pc * 100) if ltp and pc else None,
        
            "timestamp": ts
        })

        })
    df=pd.DataFrame(rows)[BASE_COLS]
    append_to_github(sym, df)

st.success("✅ KITE snapshot + DHAN append completed")

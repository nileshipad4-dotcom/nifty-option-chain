import streamlit as st
import pandas as pd
from datetime import datetime, timedelta
import pytz, base64, requests
from streamlit_autorefresh import st_autorefresh
from kiteconnect import KiteConnect

# ==================================================
# CONFIG
# ==================================================
IST = pytz.timezone("Asia/Kolkata")
st.set_page_config(layout="wide")
st.title("📊 Option Chain Collector (KITE + DHAN)")

refresh = st_autorefresh(interval=180_000, key="auto")

SOURCE = st.radio(
    "Select Data Source",
    ["KITE (Stocks – New CSV)", "DHAN (Index – Append CSV)"],
    horizontal=True
)

# ==================================================
# GITHUB
# ==================================================
TOKEN = st.secrets["GITHUB_TOKEN"]
BRANCH = st.secrets.get("GITHUB_BRANCH", "main")

KITE_REPO = st.secrets["KITE_REPO"]
DHAN_REPO = st.secrets["DHAN_REPO"]

HEADERS = {
    "Authorization": f"token {TOKEN}",
    "Accept": "application/vnd.github+json"
}

# ==================================================
# ---------- KITE SOURCE ----------
# ==================================================
if SOURCE.startswith("KITE"):

    API_KEY = "bkgv59vaazn56c42"
    ACCESS_TOKEN = "IO4wzhuRsIeusChlrimjw8VZgwA3W10T"
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

    kite = KiteConnect(api_key=API_KEY)
    kite.set_access_token(ACCESS_TOKEN)

    @st.cache_data
    def instruments():
        return pd.DataFrame(kite.instruments("NFO"))

    inst = instruments()

    def fetch_kite():
        rows = []
        ts = datetime.now(IST).strftime("%Y-%m-%d %H:%M:%S")

        spot = kite.quote([f"NSE:{s}" for s in STOCKS])

        for s in STOCKS:
            q = spot.get(f"NSE:{s}", {})
            rows.append({
                "Stock": s,
                "LTP": q.get("last_price"),
                "High": q.get("ohlc",{}).get("high"),
                "Low": q.get("ohlc",{}).get("low"),
                "timestamp": ts
            })

        return pd.DataFrame(rows)

    df = fetch_kite()
    st.dataframe(df)

    fname = f"data/option_chain_{datetime.now(IST).strftime('%Y-%m-%d_%H-%M-%S')}.csv"

    content = base64.b64encode(df.to_csv(index=False).encode()).decode()

    url = f"https://api.github.com/repos/{KITE_REPO}/contents/{fname}"

    r = requests.put(url, headers=HEADERS, json={
        "message": f"KITE snapshot {fname}",
        "content": content,
        "branch": BRANCH
    })

    st.success(f"Saved NEW CSV → {fname}")

# ==================================================
# ---------- DHAN SOURCE ----------
# ==================================================
else:

    CLIENT_ID = "1102712380"
    ACCESS_TOKEN = "eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzUxMiJ9.eyJpc3MiOiJkaGFuIiwicGFydG5lcklkIjoiIiwiZXhwIjoxNzY4MDAyMjYzLCJpYXQiOjE3Njc5MTU4NjMsInRva2VuQ29uc3VtZXJUeXBlIjoiU0VMRiIsIndlYmhvb2tVcmwiOiIiLCJkaGFuQ2xpZW50SWQiOiIxMTAyNzEyMzgwIn0.OqpycJr1HOcBRtTgGyxh9rgS8moqvBL4dfT9AYmIvgeyUhy4mYjuTq9dfUACwH4lwwXvt9Jndb_383Q5An_4Cg"
    API = "https://api.dhan.co/v2"

    HEADERS_DHAN = {
        "client-id": CLIENT_ID,
        "access-token": ACCESS_TOKEN,
        "Content-Type": "application/json"
    }

    UNDERLYINGS = {
        "nifty": {"scrip": 13, "seg": "IDX_I", "center": 26000},
        "banknifty": {"scrip": 25, "seg": "IDX_I", "center": 60000},
    }

    def fetch_dhan(sym, cfg):
        exp = requests.post(
            f"{API}/optionchain/expirylist",
            headers=HEADERS_DHAN,
            json={"UnderlyingScrip": cfg["scrip"], "UnderlyingSeg": cfg["seg"]}
        ).json()["data"][0]

        data = requests.post(
            f"{API}/optionchain",
            headers=HEADERS_DHAN,
            json={"UnderlyingScrip": cfg["scrip"], "UnderlyingSeg": cfg["seg"], "Expiry": exp}
        ).json()["data"]["oc"]

        rows = []
        ts = datetime.now(IST).strftime("%Y-%m-%d %H:%M")

        for k, v in data.items():
            ce, pe = v.get("ce",{}), v.get("pe",{})
            rows.append({
                "Strike": int(float(k)),
                "CE LTP": ce.get("last_price"),
                "PE LTP": pe.get("last_price"),
                "timestamp": ts
            })

        return pd.DataFrame(rows)

    for sym, cfg in UNDERLYINGS.items():

        df_new = fetch_dhan(sym, cfg)

        # ---- LOAD EXISTING CSV FROM GITHUB
        path = f"data/{sym}.csv"
        url = f"https://api.github.com/repos/{DHAN_REPO}/contents/{path}"

        r = requests.get(url, headers=HEADERS)

        if r.status_code == 200:
            old = base64.b64decode(r.json()["content"])
            df_old = pd.read_csv(pd.compat.StringIO(old.decode()))
            df_final = pd.concat([df_old, df_new], ignore_index=True)
            sha = r.json()["sha"]
        else:
            df_final = df_new
            sha = None

        content = base64.b64encode(df_final.to_csv(index=False).encode()).decode()

        payload = {
            "message": f"DHAN append {sym}",
            "content": content,
            "branch": BRANCH
        }
        if sha:
            payload["sha"] = sha

        requests.put(url, headers=HEADERS, json=payload)
        st.success(f"Appended → {sym}.csv")


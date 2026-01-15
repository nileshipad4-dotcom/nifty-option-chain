import streamlit as st
import pandas as pd
from kiteconnect import KiteConnect
from datetime import datetime
import pytz
import base64
import requests
import os
from streamlit_autorefresh import st_autorefresh
from kite_config import KITE_API_KEY, KITE_ACCESS_TOKEN, DHAN_CLIENT_ID, DHAN_ACCESS_TOKEN

API_KEY = "bkgv59vaazn56c42"
ACCESS_TOKEN = "rgpSvAPRwW5M6iepW7GpZh4QYV6JgaUz"

# ==================================================
# STREAMLIT CONFIG
# ==================================================
st.set_page_config(page_title="LIVE Option Chain Snapshot", layout="wide")
st.title("📊 LIVE Option Chain → Kite + Dhan")

refresh_tick = st_autorefresh(interval=180_000, key="live_refresh")

# ==================================================
# TIMEZONE
# ==================================================
IST = pytz.timezone("Asia/Kolkata")



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
    "MANAPPURAM","MANKIND","MARICO","MARUTI","MAXHEALTH","MAZDOCK","MCX","MFSL",
    "MOTHERSON","MPHASIS","MUTHOOTFIN","NATIONALUM","NAUKRI","NBCC","NESTLEIND","NHPC",
    "NMDC","NTPC","NUVAMA","NYKAA","OBEROIRLTY","OFSS","OIL","ONGC",
    "PAGEIND","PATANJALI","PAYTM","PERSISTENT","PETRONET","PFC","PGEL","PHOENIXLTD",
    "PIDILITIND","PIIND","PNB","PNBHOUSING","POLICYBZR","POLYCAB","POWERGRID","POWERINDIA",
    "PPLPHARMA","PREMIERENE","PRESTIGE","RBLBANK","RECLTD","RELIANCE","RVNL","SAIL",
    "SAMMAANCAP","SBICARD","SBILIFE","SBIN","SHREECEM","SHRIRAMFIN","SIEMENS","SOLARINDS",
    "SONACOMS","SRF","SUNPHARMA","SUPREMEIND","SUZLON","SWIGGY","SYNGENE","TATACONSUM",
    "TATAELXSI","TATAPOWER","TATASTEEL","TATATECH","TCS","TECHM","TIINDIA","TITAN","TMPV",
    "TORNTPHARM","TORNTPOWER","TRENT","TVSMOTOR","ULTRACEMCO","UNIONBANK","UNITDSPR",
    "UNOMINDA","UPL","VBL","VEDL","VOLTAS","WAAREEENER","WIPRO","YESBANK","ZYDUSLIFE"
]



# ==================================================
# DHAN CONFIG
# ==================================================
CLIENT_ID = "1102712380"
DHAN_TOKEN = "eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzUxMiJ9.eyJpc3MiOiJkaGFuIiwicGFydG5lcklkIjoiIiwiZXhwIjoxNzY4NjAwNDQ0LCJpYXQiOjE3Njg1MTQwNDQsInRva2VuQ29uc3VtZXJUeXBlIjoiU0VMRiIsIndlYmhvb2tVcmwiOiIiLCJkaGFuQ2xpZW50SWQiOiIxMTAyNzEyMzgwIn0.Sgsch6RRDYDHboth4Hsttjgnsjle2TqUPR-OahFKTgcmkhB7_rDYXG_yZR1qbXj3SlBYjZxPBP_JmDeNjktiOw"
API_BASE = "https://api.dhan.co/v2"

UNDERLYINGS = {
    "NIFTY":      {"scrip": 13,  "seg": "IDX_I", "center": 26000},
    "BANKNIFTY":  {"scrip": 25,  "seg": "IDX_I", "center": 60000},
    "MIDCPNIFTY": {"scrip": 442, "seg": "IDX_I", "center": 13600},
    "SENSEX":     {"scrip": 51,  "seg": "IDX_I", "center": 84000},
}

HEADERS = {
    "client-id": CLIENT_ID,
    "access-token": DHAN_TOKEN,
    "Content-Type": "application/json",
}

DATA_DIR = "data"
os.makedirs(DATA_DIR, exist_ok=True)
# ==================================================
# ENSURE DHAN CSV FILES EXIST
# ==================================================
BASE_COLUMNS = [
    "Strike",
    "CE LTP","CE OI","CE Volume","CE IV","CE Delta","CE Gamma","CE Vega",
    "PE LTP","PE OI","PE Volume","PE IV","PE Delta","PE Gamma","PE Vega",
    "timestamp","Max Pain"
]

for sym in UNDERLYINGS.keys():
    path = os.path.join(DATA_DIR, f"{sym.lower()}.csv")
    if not os.path.exists(path):
        pd.DataFrame(columns=BASE_COLUMNS).to_csv(path, index=False)

# ==================================================
# GITHUB CONFIG (ONLY FOR KITE)
# ==================================================
GITHUB_TOKEN = st.secrets["GITHUB_TOKEN"]
KITE_REPO = st.secrets["KITE_REPO"]
DHAN_REPO = st.secrets["DHAN_REPO"]
GITHUB_BRANCH = st.secrets.get("GITHUB_BRANCH", "main")

# ==================================================
# INIT KITE
# ==================================================
kite = KiteConnect(api_key=API_KEY)
kite.set_access_token(ACCESS_TOKEN)

@st.cache_data(show_spinner=False)
def load_instruments():
    return pd.DataFrame(kite.instruments("NFO"))

instruments = load_instruments()

# ==================================================
# HELPERS
# ==================================================
def chunk(lst, size=100):
    for i in range(0, len(lst), size):
        yield lst[i:i + size]

def compute_max_pain_kite(df):
    df = df.fillna(0)
    A = df["CE_LTP"]
    B = df["CE_OI"]
    G = df["Strike"]
    M = df["PE_LTP"]
    L = df["PE_OI"]

    mp = []
    for i in range(len(df)):
        val = (
            -sum(A[i:] * B[i:])
            + G.iloc[i] * sum(B[:i]) - sum(G[:i] * B[:i])
            - sum(M[:i] * L[:i])
            + sum(G[i:] * L[i:]) - G.iloc[i] * sum(L[i:])
        )
        mp.append(int(val / 10000))

    df["Max_Pain"] = mp
    return df

def compute_max_pain_dhan(df):
    A = df["CE LTP"].fillna(0)
    B = df["CE OI"].fillna(0)
    G = df["Strike"]
    M = df["PE LTP"].fillna(0)
    L = df["PE OI"].fillna(0)

    mp = []
    for i in range(len(df)):
        val = (
            -sum(A[i:] * B[i:])
            + G.iloc[i] * sum(B[:i]) - sum(G[:i] * B[:i])
            - sum(M[:i] * L[:i])
            + sum(G[i:] * L[i:]) - G.iloc[i] * sum(L[i:])
        )
        mp.append(int(val / 10000))

    df["Max Pain"] = mp
    return df

# ==================================================
# KITE OPTION CHAIN (UNCHANGED)
# ==================================================
def fetch_kite_option_chain():
    option_map = {}
    all_option_symbols = []

    for stock in STOCKS:
        df = instruments[
            (instruments["name"] == stock) &
            (instruments["segment"] == "NFO-OPT")
        ].copy()

        if df.empty:
            continue

        df["expiry"] = pd.to_datetime(df["expiry"])
        expiry = df["expiry"].min()
        df = df[df["expiry"] == expiry]

        option_map[stock] = df
        all_option_symbols.extend(["NFO:" + ts for ts in df["tradingsymbol"]])

    option_quotes = {}
    for batch in chunk(all_option_symbols):
        try:
            option_quotes.update(kite.quote(batch))
        except Exception as e:
            st.warning(f"Option quote batch failed: {e}")

    try:
        spot_quotes = kite.quote([f"NSE:{s}" for s in option_map.keys()])
    except Exception as e:
        st.error(f"Kite spot quote failed: {e}")
        spot_quotes = {}


    all_data = []
    now_ts = datetime.now(IST).strftime("%Y-%m-%d %H:%M:%S")

    for stock, df in option_map.items():
        rows = []

        spot = spot_quotes.get(f"NSE:{stock}", {})
        stock_ltp = spot.get("last_price")
        ohlc = spot.get("ohlc", {})
        prev_close = ohlc.get("close")

        pct_change = (
            round(((stock_ltp - prev_close) / prev_close) * 100, 3)
            if stock_ltp and prev_close else None
        )

        for strike in sorted(df["strike"].unique()):
            ce = df[(df["strike"] == strike) & (df["instrument_type"] == "CE")]
            pe = df[(df["strike"] == strike) & (df["instrument_type"] == "PE")]

            ce_q = option_quotes.get("NFO:" + ce.iloc[0]["tradingsymbol"], {}) if not ce.empty else {}
            pe_q = option_quotes.get("NFO:" + pe.iloc[0]["tradingsymbol"], {}) if not pe.empty else {}

            rows.append({
                "Stock": stock,
                "Expiry": df["expiry"].iloc[0].date(),
                "Strike": strike,
                "CE_LTP": ce_q.get("last_price"),
                "CE_OI": ce_q.get("oi"),
                "CE_Volume": ce_q.get("volume"),
                "PE_LTP": pe_q.get("last_price"),
                "PE_OI": pe_q.get("oi"),
                "PE_Volume": pe_q.get("volume"),
                "Stock_LTP": stock_ltp,
                "Stock_High": ohlc.get("high"),
                "Stock_Low": ohlc.get("low"),
                "Stock_%_Change": pct_change,
                "timestamp": now_ts,
            })

        stock_df = pd.DataFrame(rows).sort_values("Strike")
        stock_df = compute_max_pain_kite(stock_df)
        all_data.append(stock_df)

    return pd.concat(all_data, ignore_index=True)

# ==================================================
# DHAN OPTION CHAIN (FULL + SAVING)
# ==================================================
BASE_COLUMNS = [
    "Strike",
    "CE LTP","CE OI","CE Volume","CE IV","CE Delta","CE Gamma","CE Vega",
    "PE LTP","PE OI","PE Volume","PE IV","PE Delta","PE Gamma","PE Vega",
    "timestamp","Max Pain"
]

def fetch_dhan_index(sym, cfg):
    r1 = requests.post(
        f"{API_BASE}/optionchain/expirylist",
        headers=HEADERS,
        json={"UnderlyingScrip": cfg["scrip"], "UnderlyingSeg": cfg["seg"]}
    )

    expiries = r1.json().get("data", []) if r1.status_code == 200 else []
    if not expiries:
        return pd.DataFrame()

    r2 = requests.post(
        f"{API_BASE}/optionchain",
        headers=HEADERS,
        json={
            "UnderlyingScrip": cfg["scrip"],
            "UnderlyingSeg": cfg["seg"],
            "Expiry": expiries[0]
        }
    )

    data = r2.json().get("data") if r2.status_code == 200 else {}
    oc = data.get("oc", {}) if data else {}

    strikes = sorted(float(s) for s in oc.keys())

    center = cfg["center"]
    below = [s for s in strikes if s <= center][-35:]
    above = [s for s in strikes if s > center][:36]
    selected = sorted(set(below + above))

    rows = []
    ts = datetime.now(IST).strftime("%Y-%m-%d %H:%M")

    for s in selected:
        v = oc.get(f"{s:.6f}", {})
        ce = v.get("ce", {})
        pe = v.get("pe", {})

        rows.append({
            "Strike": int(s),
            "CE LTP": ce.get("last_price"),
            "CE OI": ce.get("oi"),
            "CE Volume": ce.get("volume"),
            "CE IV": ce.get("implied_volatility"),
            "CE Delta": ce.get("greeks", {}).get("delta"),
            "CE Gamma": ce.get("greeks", {}).get("gamma"),
            "CE Vega": ce.get("greeks", {}).get("vega"),
            "PE LTP": pe.get("last_price"),
            "PE OI": pe.get("oi"),
            "PE Volume": pe.get("volume"),
            "PE IV": pe.get("implied_volatility"),
            "PE Delta": pe.get("greeks", {}).get("delta"),
            "PE Gamma": pe.get("greeks", {}).get("gamma"),
            "PE Vega": pe.get("greeks", {}).get("vega"),
            "timestamp": ts
        })

    if not rows:
        return pd.DataFrame()

    df = pd.DataFrame(rows).sort_values("Strike")


    num_cols = [c for c in df.columns if c not in ["Strike","timestamp"]]
    for c in num_cols:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    df = compute_max_pain_dhan(df)
    df = df[BASE_COLUMNS]

    filename = f"data/{sym.lower()}.csv"
    csv_bytes = df.to_csv(index=False).encode()
    content = base64.b64encode(csv_bytes).decode()
    
    url = f"https://api.github.com/repos/{DHAN_REPO}/contents/{filename}"
    
    headers = {
        "Authorization": f"token {GITHUB_TOKEN}",
        "Accept": "application/vnd.github+json"
    }
    
    payload = {
        "message": f"Auto update {filename}",
        "content": content,
        "branch": GITHUB_BRANCH
    }
    
    # Check if file exists to get SHA
    check = requests.get(url, headers=headers)
    
    if check.status_code == 200:
        payload["sha"] = check.json()["sha"]
    
    r = requests.put(url, headers=headers, json=payload)
    
    if r.status_code not in (200, 201):
        raise Exception(r.json())

    
    return df


# ==================================================
# UI
# ==================================================
st.subheader("📈 KITE – STOCK OPTION CHAIN")
df_kite = fetch_kite_option_chain()
st.dataframe(df_kite, use_container_width=True)

# ---- RESTORE KITE CSV SAVE ----
try:
    kite_filename = f"data/option_chain_{datetime.now(IST).strftime('%Y-%m-%d_%H-%M')}.csv"
    csv_bytes = df_kite.to_csv(index=False).encode()
    content = base64.b64encode(csv_bytes).decode()

    url = f"https://api.github.com/repos/{KITE_REPO}/contents/{kite_filename}"

    headers = {
        "Authorization": f"token {GITHUB_TOKEN}",
        "Accept": "application/vnd.github+json"
    }

    payload = {
        "message": f"Auto snapshot {kite_filename}",
        "content": content,
        "branch": GITHUB_BRANCH
    }

    r = requests.put(url, headers=headers, json=payload)

    if r.status_code not in (200, 201):
        raise Exception(r.json())

    st.success(f"✅ Kite CSV saved to GitHub: {kite_filename}")

except Exception as e:
    st.error(f"❌ Kite GitHub save failed: {e}")

st.subheader("📊 DHAN – INDEX OPTION CHAINS")

for sym, cfg in UNDERLYINGS.items():
    st.markdown(f"### {sym}")
    df_idx = fetch_dhan_index(sym, cfg)

    if df_idx.empty:
        st.error(f"{sym} data not available")
    else:
        st.dataframe(df_idx, use_container_width=True)
        save_time = datetime.now(IST).strftime("%Y-%m-%d %H:%M:%S") 
        st.success(f"✅ {sym} uploaded to GitHub @ {save_time}")


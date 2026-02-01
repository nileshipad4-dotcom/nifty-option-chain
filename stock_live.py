import streamlit as st
import pandas as pd
from kiteconnect import KiteConnect
from datetime import datetime
import pytz
import base64
import requests
import os
from streamlit_autorefresh import st_autorefresh
from kite_config import KITE_API_KEY, KITE_ACCESS_TOKEN

# 🔴 FORCE CACHE RESET
st.cache_data.clear()

# ==================================================
# CONFIG
# ==================================================
API_KEY = KITE_API_KEY
ACCESS_TOKEN = KITE_ACCESS_TOKEN

st.set_page_config(page_title="LIVE Option Chain – Kite", layout="wide")
st.title("📊 LIVE Option Chain – Kite Only")

st_autorefresh(interval=300_000, key="live_refresh")

IST = pytz.timezone("Asia/Kolkata")

# ==================================================
# STOCK & INDEX LISTS
# ==================================================
STOCKS = [
    "360ONE","ABB","ABCAPITAL","ADANIENSOL","ADANIENT","ADANIGREEN","ADANIPORTS","ALKEM",
    "AMBER","AMBUJACEM","ANGELONE","APLAPOLLO","ASHOKLEY","ASIANPAINT","ASTRAL","AUROPHARMA",
    "AUBANK","AXISBANK","BAJAJ-AUTO","BAJAJFINSV","BAJAJHLDNG","BAJFINANCE","BANDHANBNK",
    "BANKBARODA","BANKINDIA","BEL","BHEL","BHARATFORG","BHARTIARTL","BIOCON","BPCL",
    "BRITANNIA","BSE","CAMS","CANBK","CDSL","CGPOWER","CHOLAFIN","CIPLA","COALINDIA",
    "COFORGE","COLPAL","CONCOR","CROMPTON","CUMMINSIND","DABUR","DALBHARAT","DELHIVERY",
    "DIVISLAB","DLF","DMART","DRREDDY","EICHERMOT","EXIDEIND","FEDERALBNK","FORTIS",
    "GAIL","GLENMARK","GODREJCP","GODREJPROP","GRASIM","HAL","HAVELLS","HCLTECH",
    "HDFCAMC","HDFCBANK","HDFCLIFE","HEROMOTOCO","HINDALCO","HINDPETRO","HINDUNILVR",
    "HINDZINC","HUDCO","ICICIBANK","ICICIGI","ICICIPRULI","IDEA","IDFCFIRSTB","IEX",
    "INDHOTEL","INDIGO","INDIANB","INDUSINDBK","INDUSTOWER","INFY","INOXWIND","IOC",
    "IRCTC","IREDA","IRFC","ITC","JINDALSTEL","JIOFIN","JSWENERGY","JSWSTEEL",
    "JUBLFOOD","KALYANKJIL","KAYNES","KEI","KFINTECH","KOTAKBANK","KPITTECH","LAURUSLABS",
    "LICI","LICHSGFIN","LODHA","LT","LTF","LTIM","LUPIN","M&M","MANAPPURAM","MANKIND",
    "MARICO","MARUTI","MAXHEALTH","MCX","MFSL","MPHASIS","MOTHERSON","MUTHOOTFIN",
    "NAUKRI","NATIONALUM","NESTLEIND","NHPC","NMDC","NTPC","NYKAA","OBEROIRLTY","OFSS",
    "OIL","ONGC","PAGEIND","PATANJALI","PAYTM","PERSISTENT","PETRONET","PFC",
    "PGEL","PHOENIXLTD","PIDILITIND","PNB","PNBHOUSING","POLICYBZR","POLYCAB",
    "POWERGRID","POWERINDIA","PPLPHARMA","PREMIERENE","PRESTIGE","RBLBANK","RECLTD",
    "RELIANCE","RVNL","SAIL","SAMMAANCAP","SBICARD","SBILIFE","SBIN","SHREECEM",
    "SHRIRAMFIN","SIEMENS","SOLARINDS","SONACOMS","SRF","SUNPHARMA","SUZLON",
    "SYNGENE","TATACONSUM","TATAELXSI","TATAPOWER","TATASTEEL","TATATECH","TCS",
    "TECHM","TIINDIA","TITAN","TORNTPHARM","TORNTPOWER","TRENT","TVSMOTOR",
    "ULTRACEMCO","UNIONBANK","UNITDSPR","UNOMINDA","UPL","VBL","VEDL","VOLTAS",
    "WAAREEENER","WIPRO","YESBANK","ZYDUSLIFE"
]


INDEXES = ["NIFTY", "BANKNIFTY", "MIDCPNIFTY"]

# ==================================================
# DIRECTORIES
# ==================================================
DATA_DIR = "data"
DATA_INDEX_DIR = "data_index"
os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(DATA_INDEX_DIR, exist_ok=True)

# ==================================================
# GITHUB CONFIG
# ==================================================
GITHUB_TOKEN = st.secrets["GITHUB_TOKEN"]
KITE_REPO = st.secrets["KITE_REPO"]
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
def chunk(lst, size=40):
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

# ==================================================
# FETCH FUNCTION
# ==================================================
def fetch_option_chain(symbol_list, is_index=False):
    option_map = {}
    all_option_symbols = []

    for stock in symbol_list:
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

    INDEX_SPOT_MAP = {
        "NIFTY": "NSE:NIFTY 50",
        "BANKNIFTY": "NSE:NIFTY BANK",
        "MIDCPNIFTY": "NSE:NIFTY MIDCAP SELECT"
    }

    try:
        spot_symbols = [INDEX_SPOT_MAP.get(s, f"NSE:{s}") for s in option_map.keys()]
        spot_quotes = kite.quote(spot_symbols)
    except Exception as e:
        st.error(f"Kite spot quote failed: {e}")
        spot_quotes = {}

    all_data = []
    now_ts = datetime.now(IST).strftime("%Y-%m-%d %H:%M:%S")

    for stock, df in option_map.items():
        rows = []

        spot_key = INDEX_SPOT_MAP.get(stock, f"NSE:{stock}")
        spot = spot_quotes.get(spot_key, {})
        ohlc = spot.get("ohlc", {})

        stock_ltp = spot.get("last_price")
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

            if is_index:
                rows.append({
                    "Symbol": stock,
                    "Expiry": df["expiry"].iloc[0].date(),
                    "Strike": strike,
                    "CE_LTP": ce_q.get("last_price"),
                    "CE_OI": ce_q.get("oi"),
                    "CE_Volume": ce_q.get("volume"),
                    "PE_LTP": pe_q.get("last_price"),
                    "PE_OI": pe_q.get("oi"),
                    "PE_Volume": pe_q.get("volume"),
                    "Spot": stock_ltp,
                    "%Change": pct_change,
                    "timestamp": now_ts,
                })
            else:
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
# UI – STOCK TABLE (FORCE RENAME)
# ==================================================
st.subheader("📈 STOCK OPTION CHAIN")

df_stocks = fetch_option_chain(STOCKS, is_index=False)

# 🔒 FORCE correct column names
df_stocks = df_stocks.rename(columns={
    "Symbol": "Stock",
    "Spot": "Stock_LTP",
    "%Change": "Stock_%_Change"
})

for col in ["Stock_High", "Stock_Low"]:
    if col not in df_stocks.columns:
        df_stocks[col] = None

st.write("FINAL STOCK COLUMNS:", df_stocks.columns.tolist())
st.dataframe(df_stocks, use_container_width=True)

# ==================================================
# SAVE STOCK CSV
# ==================================================
stock_filename = f"data/option_chain_{datetime.now(IST).strftime('%Y-%m-%d_%H-%M')}.csv"
content = base64.b64encode(df_stocks.to_csv(index=False).encode()).decode()

url = f"https://api.github.com/repos/{KITE_REPO}/contents/{stock_filename}"
headers = {"Authorization": f"token {GITHUB_TOKEN}", "Accept": "application/vnd.github+json"}

payload = {"message": f"Auto snapshot {stock_filename}", "content": content, "branch": GITHUB_BRANCH}
r = requests.put(url, headers=headers, json=payload)

if r.status_code in (200, 201):
    st.success(f"✅ Stock CSV saved: {stock_filename}")
else:
    st.error(f"❌ Stock GitHub save failed: {r.text}")

# ==================================================
# UI – INDEX TABLE (UNCHANGED)
# ==================================================
st.subheader("📊 INDEX OPTION CHAIN")
df_index = fetch_option_chain(INDEXES, is_index=True)
st.dataframe(df_index, use_container_width=True)

# ==================================================
# SAVE INDEX CSV
# ==================================================
index_filename = f"data_index/index_OC_{datetime.now(IST).strftime('%Y-%m-%d_%H-%M')}.csv"
content = base64.b64encode(df_index.to_csv(index=False).encode()).decode()

url = f"https://api.github.com/repos/{KITE_REPO}/contents/{index_filename}"
payload = {"message": f"Auto snapshot {index_filename}", "content": content, "branch": GITHUB_BRANCH}
r = requests.put(url, headers=headers, json=payload)

if r.status_code in (200, 201):
    st.success(f"✅ Index CSV saved: {index_filename}")
else:
    st.error(f"❌ Index GitHub save failed: {r.text}")

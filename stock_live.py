import streamlit as st
import pandas as pd
from kiteconnect import KiteConnect
from datetime import datetime
import pytz

# ==================================================
# CONFIG
# ==================================================
st.set_page_config(page_title="Max Pain Dashboard", layout="wide")

REFRESH_SECONDS = 60
IST = pytz.timezone("Asia/Kolkata")

st.markdown(
    f"<meta http-equiv='refresh' content='{REFRESH_SECONDS}'>",
    unsafe_allow_html=True
)

# ==================================================
# 🔑 KITE API
# ==================================================
API_KEY = "bkgv59vaazn56c42"
ACCESS_TOKEN = "HwNfTAk4E3mk2B11MPBFC87FxrVBnvqp"

# ==================================================
# STOCKS
# ==================================================
STOCKS = [
    "360ONE","ABB","ABCAPITAL","ADANIENT","ADANIGREEN","ADANIENSOL","ADANIPORTS",
    "ALKEM","AMBER","AMBUJACEM","ANGELONE","APLAPOLLO","APOLLOHOSP","ASHOKLEY",
    "ASIANPAINT","ASTRAL","AUROPHARMA","AUBANK","AXISBANK","BAJAJ-AUTO",
    "BAJAJFINSV","BAJFINANCE","BANDHANBNK","BANKBARODA","BANKINDIA","BDL","BEL",
    "BHARATFORG","BHARTIARTL","BHEL","BIOCON","BPCL","BRITANNIA","BSE","CAMS",
    "CANBK","CDSL","CGPOWER","CHOLAFIN","CIPLA","COALINDIA","COFORGE","COLPAL",
    "CONCOR","CROMPTON","CUMMINSIND","CYIENT","DABUR","DALBHARAT","DELHIVERY",
    "DIVISLAB","DLF","DMART","DRREDDY","EICHERMOT","ETERNAL","EXIDEIND",
    "FEDERALBNK","FORTIS","GAIL","GMRAIRPORT","GODREJCP","GODREJPROP","GRASIM",
    "HAL","HAVELLS","HCLTECH","HDFCAMC","HDFCBANK","HDFCLIFE","HEROMOTOCO",
    "HFCL","HINDALCO","HINDPETRO","HINDUNILVR","HINDZINC","HUDCO","IEX",
    "ICICIBANK","ICICIGI","ICICIPRULI","IDFCFIRSTB","IIFL","INDHOTEL",
    "INDIANB","INDIGO","INDUSTOWER","INFY","INOXWIND","IOC","IRCTC","IRFC",
    "IREDA","ITC","JINDALSTEL","JIOFIN","JSWENERGY","JSWSTEEL","JUBLFOOD",
    "KALYANKJIL","KAYNES","KEI","KFINTECH","KPITTECH","KOTAKBANK",
    "LAURUSLABS","LICHSGFIN","LICI","LODHA","LT","LTIM","LTF","LUPIN","M&M",
    "MANAPPURAM","MARICO","MARUTI","MAXHEALTH","MAZDOCK","MCX","MFSL",
    "MPHASIS","MOTHERSON","MUTHOOTFIN","NAUKRI","NATIONALUM","NBCC","NCC",
    "NESTLEIND","NHPC","NMDC","NTPC","NUVAMA","NYKAA","OBEROIRLTY","OFSS",
    "OIL","ONGC","PAGEIND","PATANJALI","PAYTM","PERSISTENT","PETRONET",
    "PFC","PGEL","PHOENIXLTD","PIDILITIND","PIIND","PNB","PNBHOUSING",
    "POLICYBZR","POLYCAB","POWERGRID","PRESTIGE","RBLBANK","RECLTD",
    "RELIANCE","RVNL","SAIL","SAMMAANCAP","SBICARD","SBILIFE","SBIN",
    "SHREECEM","SHRIRAMFIN","SIEMENS","SOLARINDS","SONACOMS","SRF",
    "SUNPHARMA","SUZLON","SYNGENE","TATACONSUM","TATAELXSI","TATAPOWER",
    "TATASTEEL","TATATECH","TCS","TECHM","TITAGARH","TITAN","TORNTPHARM",
    "TORNTPOWER","TRENT","TVSMOTOR","ULTRACEMCO","UNITDSPR","UNIONBANK",
    "UNOMINDA","UPL","VEDL","VBL","VOLTAS","WIPRO","YESBANK","ZYDUSLIFE"
]

# ==================================================
# INIT KITE
# ==================================================
@st.cache_resource
def init_kite():
    k = KiteConnect(api_key=API_KEY)
    k.set_access_token(ACCESS_TOKEN)
    return k

kite = init_kite()

# ==================================================
# LOAD INSTRUMENTS
# ==================================================
@st.cache_data(ttl=300)
def load_instruments():
    df = pd.DataFrame(kite.instruments("NFO"))
    df["expiry"] = pd.to_datetime(df["expiry"])
    return df

instruments = load_instruments()

# ==================================================
# MAX PAIN LOGIC (UNCHANGED)
# ==================================================
def compute_max_pain(df):
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
# FETCH DATA
# ==================================================
@st.cache_data(ttl=55)
def fetch_data():
    final_rows = []

    spot_quotes = kite.quote([f"NSE:{s}" for s in STOCKS])

    for stock in STOCKS:
        opt_df = instruments[
            (instruments["name"] == stock) &
            (instruments["segment"] == "NFO-OPT")
        ].copy()

        if opt_df.empty:
            continue

        expiry = opt_df["expiry"].min()
        opt_df = opt_df[opt_df["expiry"] == expiry]

        symbols = ["NFO:" + s for s in opt_df["tradingsymbol"]]
        quotes = kite.quote(symbols)

        rows = []
        spot = spot_quotes.get(f"NSE:{stock}", {})
        stock_ltp = spot.get("last_price")

        for strike in sorted(opt_df["strike"].unique()):
            ce = opt_df[(opt_df["strike"] == strike) & (opt_df["instrument_type"] == "CE")]
            pe = opt_df[(opt_df["strike"] == strike) & (opt_df["instrument_type"] == "PE")]

            ce_q = quotes.get("NFO:" + ce.iloc[0]["tradingsymbol"], {}) if not ce.empty else {}
            pe_q = quotes.get("NFO:" + pe.iloc[0]["tradingsymbol"], {}) if not pe.empty else {}

            rows.append({
                "Strike": strike,
                "CE_LTP": ce_q.get("last_price"),
                "CE_OI": ce_q.get("oi"),
                "PE_LTP": pe_q.get("last_price"),
                "PE_OI": pe_q.get("oi"),
            })

        df = pd.DataFrame(rows)
        df = compute_max_pain(df)

        for _, r in df.iterrows():
            final_rows.append({
                "Stock": stock,
                "Stock_LTP": stock_ltp,
                "Strike": r["Strike"],
                "Max_Pain": r["Max_Pain"]
            })

        # blank row after each stock
        final_rows.append({
            "Stock": "",
            "Stock_LTP": "",
            "Strike": "",
            "Max_Pain": ""
        })

    return pd.DataFrame(final_rows)

# ==================================================
# UI
# ==================================================
st.title("📊 Live Max Pain – Full Table View")
st.caption("Auto refresh every 60 seconds")

df = fetch_data()

st.dataframe(
    df,
    use_container_width=True,
    height=900
)

st.success(f"Last updated: {datetime.now(IST).strftime('%Y-%m-%d %H:%M:%S')} IST")

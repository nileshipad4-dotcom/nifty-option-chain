import streamlit as st
import pandas as pd
from kiteconnect import KiteConnect
from datetime import datetime
import pytz
import time

# ==================================================
# STREAMLIT CONFIG
# ==================================================
st.set_page_config(
    page_title="Live Max Pain Dashboard",
    layout="wide"
)

REFRESH_SECONDS = 60
IST = pytz.timezone("Asia/Kolkata")

# Auto refresh
st.markdown(
    f"<meta http-equiv='refresh' content='{REFRESH_SECONDS}'>",
    unsafe_allow_html=True
)

# ==================================================
# 🔑 KITE CONFIG (UPDATE HERE)
# ==================================================
API_KEY = "bkgv59vaazn56c42"
ACCESS_TOKEN = "HwNfTAk4E3mk2B11MPBFC87FxrVBnvqp"

# ==================================================
# STOCK LIST
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
    kite = KiteConnect(api_key=API_KEY)
    kite.set_access_token(ACCESS_TOKEN)
    return kite

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
# MAX PAIN FUNCTION
# ==================================================
def calculate_max_pain(df):
    df = df.sort_values("Strike").reset_index(drop=True)

    ce_oi = df["CE_OI"].fillna(0)
    pe_oi = df["PE_OI"].fillna(0)
    strikes = df["Strike"]

    pain = []
    for i, strike in enumerate(strikes):
        pain_val = (
            (ce_oi[i:] * (strikes[i:] - strike)).sum() +
            (pe_oi[:i] * (strike - strikes[:i])).sum()
        )
        pain.append(pain_val)

    df["Pain"] = pain
    return df.loc[df["Pain"].idxmin(), "Strike"]

# ==================================================
# FETCH DATA
# ==================================================
@st.cache_data(ttl=55)
def fetch_data():
    result = []

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

        symbols = ["NFO:" + s for s in opt_df["tradingsymbol"].tolist()]
        quotes = kite.quote(symbols)

        rows = []
        for _, r in opt_df.iterrows():
            q = quotes.get("NFO:" + r["tradingsymbol"], {})
            rows.append({
                "Strike": r["strike"],
                "Type": r["instrument_type"],
                "OI": q.get("oi")
            })

        chain = pd.DataFrame(rows)
        pivot = chain.pivot_table(
            index="Strike",
            columns="Type",
            values="OI",
            aggfunc="sum"
        ).reset_index()

        pivot.columns.name = None
        pivot.rename(columns={"CE": "CE_OI", "PE": "PE_OI"}, inplace=True)

        max_pain = calculate_max_pain(pivot)

        spot = spot_quotes.get(f"NSE:{stock}", {})
        ltp = spot.get("last_price")
        prev = spot.get("ohlc", {}).get("close")

        pct = round(((ltp - prev) / prev) * 100, 2) if ltp and prev else None

        result.append({
            "Stock": stock,
            "LTP": ltp,
            "% Change": pct,
            "Max Pain": max_pain,
            "Expiry": expiry.date()
        })

    return pd.DataFrame(result)

# ==================================================
# UI
# ==================================================
st.title("📈 Live Options Max Pain Dashboard")
st.caption("Auto refresh every 60 seconds | Zerodha Kite API")

df = fetch_data()

st.dataframe(
    df.sort_values("% Change", ascending=False),
    use_container_width=True,
    height=800
)

st.success(f"Last updated: {datetime.now(IST).strftime('%Y-%m-%d %H:%M:%S')} IST")

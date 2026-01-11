
# CSV download
import streamlit as st
import pandas as pd
from kiteconnect import KiteConnect
from datetime import datetime
import pytz
import base64
import requests
from streamlit_autorefresh import st_autorefresh

# ==================================================
# STREAMLIT CONFIG
# ==================================================
st.set_page_config(page_title="LIVE Option Chain Snapshot", layout="wide")
st.title("📊 LIVE Option Chain → GitHub Snapshot (Full Chain)")

# ⏱ Auto refresh every 60 seconds
refresh_tick = st_autorefresh(interval=180_000, key="live_refresh")

# ==================================================
# TIMEZONE
# ==================================================
IST = pytz.timezone("Asia/Kolkata")

# ==================================================
# CONFIG
# ==================================================
API_KEY = "bkgv59vaazn56c42"
ACCESS_TOKEN = "DLSvBzTX0sVAn5In8dDj0vRtC6gVbs4P"

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

# ==================================================
# GITHUB CONFIG (STREAMLIT SECRETS)
# ==================================================
GITHUB_TOKEN = st.secrets["GITHUB_TOKEN"]
GITHUB_REPO = st.secrets["KITE_REPO"]
GITHUB_BRANCH = st.secrets.get("GITHUB_BRANCH", "main")

# ==================================================
# INIT KITE
# ==================================================
kite = KiteConnect(api_key=API_KEY)
kite.set_access_token(ACCESS_TOKEN)

# ==================================================
# LOAD INSTRUMENTS (CACHED)
# ==================================================
@st.cache_data(show_spinner=False)
def load_instruments():
    return pd.DataFrame(kite.instruments("NFO"))

instruments = load_instruments()

# ==================================================
# HELPERS
# ==================================================
def chunk(lst, size=200):
    for i in range(0, len(lst), size):
        yield lst[i:i + size]

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
# FETCH FULL OPTION CHAIN (MATCHES YOUR SCRIPT)
# ==================================================
def fetch_full_option_chain():
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
        all_option_symbols.extend(
            ["NFO:" + ts for ts in df["tradingsymbol"].tolist()]
        )

    # OPTION QUOTES
    option_quotes = {}
    for batch in chunk(all_option_symbols):
        option_quotes.update(kite.quote(batch))

    # SPOT QUOTES
    spot_quotes = kite.quote([f"NSE:{s}" for s in option_map.keys()])

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

            ce_q = option_quotes.get(
                "NFO:" + ce.iloc[0]["tradingsymbol"], {}
            ) if not ce.empty else {}

            pe_q = option_quotes.get(
                "NFO:" + pe.iloc[0]["tradingsymbol"], {}
            ) if not pe.empty else {}

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
        stock_df = compute_max_pain(stock_df)
        all_data.append(stock_df)

    return pd.concat(all_data, ignore_index=True)

# ==================================================
# PUSH CSV TO GITHUB
# ==================================================
def push_csv_to_github(df):
    filename = f"data/option_chain_{datetime.now(IST).strftime('%Y-%m-%d_%H-%M')}.csv"
    csv_bytes = df.to_csv(index=False).encode()
    content = base64.b64encode(csv_bytes).decode()

    url = f"https://api.github.com/repos/{GITHUB_REPO}/contents/{filename}"

    headers = {
        "Authorization": f"token {GITHUB_TOKEN}",
        "Accept": "application/vnd.github+json"
    }

    payload = {
        "message": f"Auto snapshot {filename}",
        "content": content,
        "branch": GITHUB_BRANCH
    }

    r = requests.put(url, headers=headers, json=payload)
    if r.status_code not in (200, 201):
        raise Exception(r.json())

    return filename

# ==================================================
# FETCH → SAVE → DISPLAY
# ==================================================
with st.spinner("📡 Fetching FULL option chain..."):
    df_live = fetch_full_option_chain()

if df_live.empty:
    st.error("❌ LIVE option chain fetch failed.")
    st.stop()

st.dataframe(
    df_live.sort_values(["Stock", "Strike"]),
    use_container_width=True
)

try:
    saved_file = push_csv_to_github(df_live)
    st.success(f"✅ Saved FULL snapshot to GitHub: {saved_file}")
except Exception as e:
    st.error(f"❌ GitHub save failed: {e}")

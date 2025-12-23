import streamlit as st
from kiteconnect import KiteConnect
import pandas as pd
from datetime import datetime

# ====================================
# STREAMLIT CONFIG
# ====================================
st.set_page_config(page_title="Options Chain Dashboard", layout="wide")
st.title("📊 NSE Options Chain Dashboard")

# ====================================
# LOAD SECRETS (DO NOT HARDCODE)
# ====================================
API_KEY = st.secrets["bkgv59vaazn56c42"]
ACCESS_TOKEN = st.secrets["giMaA7SuUyWA1r9P34zVTOVlPcWG847C"]

# ====================================
# STOCK LIST
# ====================================
STOCKS = ["360ONE","ABB","ABCAPITAL","ADANIENT","ADANIGREEN","ADANIENSOL","ADANIPORTS","ALKEM","AMBER","AMBUJACEM","ANGELONE","APLAPOLLO","APOLLOHOSP","ASHOKLEY","ASIANPAINT","ASTRAL","AUROPHARMA","AUBANK","AXISBANK","BAJAJ-AUTO","BAJAJFINSV","BAJFINANCE","BANDHANBNK","BANKBARODA","BANKINDIA","BDL","BEL","BHARATFORG","BHARTIARTL","BHEL","BIOCON","BPCL","BRITANNIA","BSE","CAMS","CANBK","CDSL","CGPOWER","CHOLAFIN","CIPLA","COALINDIA","COFORGE","COLPAL","CONCOR","CROMPTON","CUMMINSIND","CYIENT","DABUR","DALBHARAT","DELHIVERY","DIVISLAB","DLF","DMART","DRREDDY","EICHERMOT","ETERNAL","EXIDEIND","FEDERALBNK","FORTIS","GAIL","GMRAIRPORT","GODREJCP","GODREJPROP","GRASIM","HAL","HAVELLS","HCLTECH","HDFCAMC","HDFCBANK","HDFCLIFE","HEROMOTOCO","HFCL","HINDALCO","HINDPETRO","HINDUNILVR","HINDZINC","HUDCO","IEX","ICICIBANK","ICICIGI","ICICIPRULI","IDFCFIRSTB","IIFL","INDHOTEL","INDIANB","INDIGO","INDUSTOWER","INFY","INOXWIND","IOC","IRCTC","IRFC","IREDA","ITC","JINDALSTEL","JIOFIN","JSWENERGY","JSWSTEEL","JUBLFOOD","KALYANKJIL","KAYNES","KEI","KFINTECH","KPITTECH","KOTAKBANK","LAURUSLABS","LICHSGFIN","LICI","LODHA","LT","LTIM","LTF","LUPIN","M&M","MANAPPURAM","MARICO","MARUTI","MAXHEALTH","MAZDOCK","MCX","MFSL","MPHASIS","MOTHERSON","MUTHOOTFIN","NAUKRI","NATIONALUM","NBCC","NCC","NESTLEIND","NHPC","NMDC","NTPC","NUVAMA","NYKAA","OBEROIRLTY","OFSS","OIL","ONGC","PAGEIND","PATANJALI","PAYTM","PERSISTENT","PETRONET","PFC","PGEL","PHOENIXLTD","PIDILITIND","PIIND","PNB","PNBHOUSING","POLICYBZR","POLYCAB","POWERGRID","PRESTIGE","RBLBANK","RECLTD","RELIANCE","RVNL","SAIL","SAMMAANCAP","SBICARD","SBILIFE","SBIN","SHREECEM","SHRIRAMFIN","SIEMENS","SOLARINDS","SONACOMS","SRF","SUNPHARMA","SUZLON","SYNGENE","TATACONSUM","TATAELXSI","TATAPOWER","TATASTEEL","TATATECH","TCS","TECHM","TITAGARH","TITAN","TORNTPHARM","TORNTPOWER","TRENT","TVSMOTOR","ULTRACEMCO","UNITDSPR","UNIONBANK","UNOMINDA","UPL","VEDL","VBL","VOLTAS","WIPRO","YESBANK","ZYDUSLIFE"]

# ====================================
# INIT KITE
# ====================================
kite = KiteConnect(api_key=API_KEY)
kite.set_access_token(ACCESS_TOKEN)

# ====================================
# LOAD INSTRUMENTS
# ====================================
@st.cache_data(ttl=86400)
def load_instruments():
    return pd.DataFrame(kite.instruments("NFO"))

instruments = load_instruments()

# ====================================
# FETCH OPTION CHAIN
# ====================================
def fetch_option_chain(stock):
    df = instruments[
        (instruments["name"] == stock) &
        (instruments["segment"] == "NFO-OPT")
    ].copy()

    if df.empty:
        return None

    df["expiry"] = pd.to_datetime(df["expiry"])
    expiry = df["expiry"].min()
    df = df[df["expiry"] == expiry]

    symbols = ["NFO:" + s for s in df["tradingsymbol"]]
    quotes = kite.quote(symbols)

    rows = []
    for strike in sorted(df["strike"].unique()):
        ce = df[(df["strike"] == strike) & (df["instrument_type"] == "CE")]
        pe = df[(df["strike"] == strike) & (df["instrument_type"] == "PE")]

        ce_q = quotes.get("NFO:" + ce.iloc[0]["tradingsymbol"], {}) if not ce.empty else {}
        pe_q = quotes.get("NFO:" + pe.iloc[0]["tradingsymbol"], {}) if not pe.empty else {}

        rows.append({
            "Stock": stock,
            "Expiry": expiry.date(),
            "Strike": strike,
            "Call_LTP": ce_q.get("last_price"),
            "Put_LTP": pe_q.get("last_price"),
            "Call_OI": ce_q.get("oi"),
            "Put_OI": pe_q.get("oi")
        })

    return pd.DataFrame(rows)

# ====================================
# UI
# ====================================
if st.button("🚀 Fetch Option Chain"):
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    st.subheader(f"Last Updated: {timestamp}")

    all_data = []

    with st.spinner("Fetching data..."):
        for stock in STOCKS:
            df = fetch_option_chain(stock)
            if df is not None:
                all_data.append(df)

    if all_data:
        final_df = pd.concat(all_data, ignore_index=True)
        st.dataframe(final_df, use_container_width=True)
        st.download_button(
            "⬇️ Download CSV",
            final_df.to_csv(index=False),
            "option_chain_all_stocks.csv",
            "text/csv"
        )
    else:
        st.warning("No option data found.")

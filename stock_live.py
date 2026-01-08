import streamlit as st
import pandas as pd
from kiteconnect import KiteConnect
from datetime import datetime
import pytz
import os
from streamlit_autorefresh import st_autorefresh

# ==================================================
# TIMEZONE
# ==================================================
IST = pytz.timezone("Asia/Kolkata")

# ==================================================
# CONFIG
# ==================================================
API_KEY = "bkgv59vaazn56c42"
ACCESS_TOKEN = "um1gYW2GgQ94kdg2G1C9vu3cWfdFF00X"

DATA_DIR = "data"
os.makedirs(DATA_DIR, exist_ok=True)

STOCKS = [
    "NIFTY","BANKNIFTY","FINNIFTY","MIDCPNIFTY",
    "RELIANCE","HDFCBANK","ICICIBANK","SBIN"
]

# ==================================================
# STREAMLIT CONFIG
# ==================================================
st.set_page_config(page_title="LIVE Option Chain Snapshot", layout="wide")
st.title("📊 LIVE Option Chain (Auto CSV Snapshot Every 60s)")

# ⏱ Auto refresh every 60 seconds
refresh_tick = st_autorefresh(interval=60_000, key="live_refresh")

# ==================================================
# INIT KITE
# ==================================================
kite = KiteConnect(api_key=API_KEY)
kite.set_access_token(ACCESS_TOKEN)

# ==================================================
# LOAD INSTRUMENTS
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
# FETCH LIVE OPTION CHAIN
# ==================================================
def fetch_live_option_chain():
    option_map = {}
    all_symbols = []

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
        all_symbols.extend(["NFO:" + s for s in df["tradingsymbol"]])

    # OPTION QUOTES
    option_quotes = {}
    for batch in chunk(all_symbols):
        option_quotes.update(kite.quote(batch))

    # SPOT QUOTES
    spot_quotes = kite.quote([f"NSE:{s}" for s in option_map.keys()])

    all_data = []
    now_ts = datetime.now(IST).strftime("%Y-%m-%d %H:%M")

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
        stock_df = compute_max_pain(stock_df)
        all_data.append(stock_df)

    return pd.concat(all_data, ignore_index=True)

# ==================================================
# FETCH + SAVE SNAPSHOT
# ==================================================
with st.spinner("📡 Fetching LIVE option data..."):
    df_live = fetch_live_option_chain()

# ==================================================
# SAFETY CHECK
# ==================================================
if df_live.empty:
    st.error("LIVE data fetch failed.")
    st.stop()

# ==================================================
# SAVE UNIQUE CSV (NO OVERWRITE)
# ==================================================
filename = f"option_chain_{datetime.now(IST).strftime('%Y-%m-%d_%H-%M-%S')}.csv"
filepath = os.path.join(DATA_DIR, filename)
df_live.to_csv(filepath, index=False)

# ==================================================
# STATUS
# ==================================================
st.caption(
    f"🟢 LIVE | Last update: {datetime.now(IST).strftime('%H:%M:%S')} | "
    f"📁 Snapshot saved: {filename}"
)

# ==================================================
# DISPLAY TABLE
# ==================================================
st.dataframe(
    df_live.sort_values(["Stock", "Strike"]),
    use_container_width=True
)

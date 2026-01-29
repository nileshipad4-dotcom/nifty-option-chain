import streamlit as st
import pandas as pd
from kiteconnect import KiteConnect
from datetime import datetime
import pytz
from streamlit_autorefresh import st_autorefresh
from kite_config import KITE_API_KEY, KITE_ACCESS_TOKEN

# ==================================================
# STREAMLIT CONFIG
# ==================================================
st.set_page_config(page_title="Live Index Max Pain – Kite", layout="wide")
st.title("📊 Live Index Max Pain (Kite Only)")

st_autorefresh(interval=300_000, key="auto_refresh")

IST = pytz.timezone("Asia/Kolkata")

# ==================================================
# CONFIG
# ==================================================
INDEXES = ["NIFTY", "BANKNIFTY", "MIDCPNIFTY"]

INDEX_SPOT_MAP = {
    "NIFTY": "NSE:NIFTY 50",
    "BANKNIFTY": "NSE:NIFTY BANK",
    "MIDCPNIFTY": "NSE:NIFTY MIDCAP SELECT"
}

# ==================================================
# INIT KITE
# ==================================================
kite = KiteConnect(api_key=KITE_API_KEY)
kite.set_access_token(KITE_ACCESS_TOKEN)

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
# UI CONTROLS
# ==================================================
c1, c2 = st.columns(2)
index = c1.selectbox("Index", INDEXES)

# ==================================================
# FILTER INSTRUMENTS
# ==================================================
idx_df = instruments[
    (instruments["name"] == index) &
    (instruments["segment"] == "NFO-OPT")
].copy()

idx_df["expiry"] = pd.to_datetime(idx_df["expiry"])
expiries = sorted(idx_df["expiry"].unique())

expiry = c2.selectbox(
    "Expiry",
    expiries,
    format_func=lambda x: x.strftime("%d-%b-%Y")
)

idx_df = idx_df[idx_df["expiry"] == expiry]

# ==================================================
# PREPARE SYMBOLS
# ==================================================
symbols = ["NFO:" + ts for ts in idx_df["tradingsymbol"]]

option_quotes = {}
for batch in chunk(symbols):
    option_quotes.update(kite.quote(batch))

spot_quote = kite.quote([INDEX_SPOT_MAP[index]])
spot = spot_quote.get(INDEX_SPOT_MAP[index], {}).get("last_price")

# ==================================================
# BUILD STRIKE TABLE
# ==================================================
rows = []

for strike in sorted(idx_df["strike"].unique()):
    ce = idx_df[(idx_df["strike"] == strike) & (idx_df["instrument_type"] == "CE")]
    pe = idx_df[(idx_df["strike"] == strike) & (idx_df["instrument_type"] == "PE")]

    if ce.empty or pe.empty:
        continue

    ce_q = option_quotes.get("NFO:" + ce.iloc[0]["tradingsymbol"], {})
    pe_q = option_quotes.get("NFO:" + pe.iloc[0]["tradingsymbol"], {})

    rows.append({
        "Strike": strike,
        "CE_LTP": ce_q.get("last_price", 0),
        "CE_OI": ce_q.get("oi", 0),
        "PE_LTP": pe_q.get("last_price", 0),
        "PE_OI": pe_q.get("oi", 0),
    })

df = pd.DataFrame(rows).sort_values("Strike")
df = compute_max_pain(df)

mp_row = df.loc[df["Max_Pain"].idxmin()]

# ==================================================
# DISPLAY
# ==================================================
st.markdown(
    f"### 📌 {index} | Expiry: {expiry.strftime('%d-%b-%Y')}"
)

c1, c2, c3 = st.columns(3)
c1.metric("Spot", int(spot) if spot else "NA")
c2.metric("Max Pain Strike", int(mp_row["Strike"]))
c3.metric("Max Pain Value", int(mp_row["Max_Pain"]))

st.dataframe(
    df[["Strike", "CE_OI", "PE_OI", "CE_LTP", "PE_LTP", "Max_Pain"]],
    use_container_width=True
)

st.caption(f"Last updated: {datetime.now(IST).strftime('%Y-%m-%d %H:%M:%S')}")

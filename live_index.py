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

# ==================================================
# CONFIG
# ==================================================
API_KEY = KITE_API_KEY
ACCESS_TOKEN = KITE_ACCESS_TOKEN

st.set_page_config(page_title="LIVE Option Chain – Kite", layout="wide")
st.title("📊 LIVE Option Chain – Kite Only")

st_autorefresh(interval=360_000, key="live_refresh")

IST = pytz.timezone("Asia/Kolkata")

STOCKS = ["NIFTY", "BANKNIFTY", "MIDCPNIFTY"]

DATA_INDEX_DIR = "data_index"
os.makedirs(DATA_INDEX_DIR, exist_ok=True)

# ==================================================
# GITHUB SECRETS
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

# ==================================================
# FETCH KITE OPTION CHAIN
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
# UI
# ==================================================
st.subheader("📈 KITE – LIVE OPTION CHAIN")

df_kite = fetch_kite_option_chain()
st.dataframe(df_kite, use_container_width=True)

# ==================================================
# SAVE TO GITHUB (data_index folder)
# ==================================================
try:
    filename = f"data_index/index_OC_{datetime.now(IST).strftime('%Y-%m-%d_%H-%M')}.csv"

    csv_bytes = df_kite.to_csv(index=False).encode()
    content = base64.b64encode(csv_bytes).decode()

    url = f"https://api.github.com/repos/{KITE_REPO}/contents/{filename}"

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

    st.success(f"✅ CSV saved to GitHub: {filename}")

except Exception as e:
    st.error(f"❌ GitHub save failed: {e}")

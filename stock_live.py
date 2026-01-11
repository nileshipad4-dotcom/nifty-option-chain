import streamlit as st
import pandas as pd
from kiteconnect import KiteConnect
from datetime import datetime
import pytz
import base64
import requests
import os
from streamlit_autorefresh import st_autorefresh

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

# ==================================================
# KITE CONFIG (UNCHANGED)
# ==================================================
API_KEY = "bkgv59vaazn56c42"
ACCESS_TOKEN = "DLSvBzTX0sVAn5In8dDj0vRtC6gVbs4P"

STOCKS = ["RELIANCE","SBIN","TCS","INFY"]

# ==================================================
# DHAN CONFIG
# ==================================================
CLIENT_ID = "1102712380"
DHAN_TOKEN = "eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzUxMiJ9.eyJpc3MiOiJkaGFuIiwicGFydG5lcklkIjoiIiwiZXhwIjoxNzY4MjQ5NzkwLCJpYXQiOjE3NjgxNjMzOTAsInRva2VuQ29uc3VtZXJUeXBlIjoiU0VMRiIsIndlYmhvb2tVcmwiOiIiLCJkaGFuQ2xpZW50SWQiOiIxMTAyNzEyMzgwIn0.y9CAHmsZCpTVulTRcK8AuiE_vaIK1-nSQ1TSqaG8zO1x8BPX2kodNgdLNPfF_P5hB_tiJUJY3bSEj-kf-0ypDw"
API_BASE = "https://api.dhan.co/v2"

HEADERS = {
    "client-id": CLIENT_ID,
    "access-token": DHAN_TOKEN,
    "Content-Type": "application/json",
}

# ==================================================
# GITHUB CONFIG (ONLY FOR KITE)
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
# KITE OPTION CHAIN
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
        except:
            pass

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
# DHAN OPTION CHAIN (DISPLAY + SAVE)
# ==================================================
def fetch_dhan_index(symbol, scrip, seg):
    r1 = requests.post(
        f"{API_BASE}/optionchain/expirylist",
        headers=HEADERS,
        json={"UnderlyingScrip": scrip, "UnderlyingSeg": seg}
    )

    expiries = r1.json().get("data", []) if r1.status_code == 200 else []
    if not expiries:
        return pd.DataFrame()

    r2 = requests.post(
        f"{API_BASE}/optionchain",
        headers=HEADERS,
        json={"UnderlyingScrip": scrip, "UnderlyingSeg": seg, "Expiry": expiries[0]}
    )

    data = r2.json().get("data") if r2.status_code == 200 else {}
    oc = data.get("oc", {}) if data else {}

    rows = []
    ts = datetime.now(IST).strftime("%Y-%m-%d %H:%M")

    for s, v in oc.items():
        ce = v.get("ce", {})
        pe = v.get("pe", {})

        rows.append({
            "Strike": int(float(s)),
            "CE LTP": ce.get("last_price"),
            "CE OI": ce.get("oi"),
            "CE Volume": ce.get("volume"),
            "PE LTP": pe.get("last_price"),
            "PE OI": pe.get("oi"),
            "PE Volume": pe.get("volume"),
            "timestamp": ts
        })

    df = pd.DataFrame(rows).sort_values("Strike")

    os.makedirs("data", exist_ok=True)
    df.to_csv(f"data/{symbol.lower()}.csv", index=False)

    return df

# ==================================================
# UI
# ==================================================
st.subheader("📈 KITE – STOCK OPTION CHAIN")
df_kite = fetch_kite_option_chain()
st.dataframe(df_kite, use_container_width=True)

st.subheader("📊 DHAN – INDEX OPTION CHAINS")

indices = {
    "NIFTY": (13, "IDX_I"),
    "BANKNIFTY": (25, "IDX_I"),
    "MIDCPNIFTY": (442, "IDX_I"),
    "SENSEX": (51, "IDX_I"),
}

for name, (scrip, seg) in indices.items():
    st.markdown(f"### {name}")
    df_idx = fetch_dhan_index(name, scrip, seg)

    if df_idx.empty:
        st.error(f"{name} data not available")
    else:
        st.dataframe(df_idx, use_container_width=True)

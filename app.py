

import streamlit as st
import time
from fyers_apiv3 import fyersModel
import pandas as pd

# ===============================
# STREAMLIT CONFIG
# ===============================
st.set_page_config(page_title="Option Chain Dashboard", layout="wide")
st.title("📊 Option Chain Dashboard (FYERS)")

# ===============================
# AUTO REFRESH (15 SECONDS)
# ===============================
REFRESH_INTERVAL = 15

if "last_refresh" not in st.session_state:
    st.session_state.last_refresh = time.time()

if time.time() - st.session_state.last_refresh > REFRESH_INTERVAL:
    st.session_state.last_refresh = time.time()
    st.experimental_rerun()

# ===============================
# FYERS CREDENTIALS (SECRETS)
# ===============================
CLIENT_ID = st.secrets["3VEZHWB1VB-100"]
ACCESS_TOKEN = st.secrets["eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJhdWQiOlsiZDoxIiwieDowIiwieDoxIl0sImF0X2hhc2giOiJnQUFBQUFCcFJFaGxsclVBVTFyZVRqS3VucTZFS1FCMkx0UHZBLVZ6OU5hajJpQks3Tld4Z2RzRHJsSGNvd3lNZUtlRkM0SzdPX1pYRzRLSWZRS2NrYmpaR0h3QjRSQTdiWEg1TDdTY2sxdGlzTnM1RTR4T1hRUT0iLCJkaXNwbGF5X25hbWUiOiIiLCJvbXMiOiJLMSIsImhzbV9rZXkiOiIwNmUwMDA2NmU0NzNlOTAxM2JkZWI1MGM2NmFkZjYzNjYwYmUwYTQzNWRjZjU3YjUzYWQyOTJmMSIsImlzRGRwaUVuYWJsZWQiOiJOIiwiaXNNdGZFbmFibGVkIjoiTiIsImZ5X2lkIjoiRkFENDE5ODkiLCJhcHBUeXBlIjoxMDAsImV4cCI6MTc2NjE5MDYwMCwiaWF0IjoxNzY2MDgyNjYxLCJpc3MiOiJhcGkuZnllcnMuaW4iLCJuYmYiOjE3NjYwODI2NjEsInN1YiI6ImFjY2Vzc190b2tlbiJ9.R8ANyzeA1Lb0DOwLj4C3BZVyjHALLBEqFrbGWVpqM1Y"]
   


# ===============================
# FYERS INIT
# ===============================
fyers = fyersModel.FyersModel(
    client_id=CLIENT_ID,
    token=ACCESS_TOKEN,
    is_async=False,
    log_path=""
)

# ===============================
# INDEX CONFIG
# ===============================
INDEX_CONFIG = {
    "NIFTY": {
        "symbol": "NSE:NIFTY50-INDEX",
        "step": 50
    },
    "BANKNIFTY": {
        "symbol": "NSE:NIFTYBANK-INDEX",
        "step": 100
    }
}

# ===============================
# DROPDOWN
# ===============================
index_name = st.selectbox("Select Index", ["NIFTY", "BANKNIFTY"])
symbol = INDEX_CONFIG[index_name]["symbol"]
strike_step = INDEX_CONFIG[index_name]["step"]

# ===============================
# LIVE PRICE
# ===============================
def get_spot_price():
    res = fyers.quotes({"symbols": symbol})
    return res["d"][0]["v"]["lp"]

# ===============================
# OPTION CHAIN + GREEKS
# ===============================
def get_option_chain():
    payload = {
        "symbol": symbol,
        "strikecount": 50,
        "timestamp": ""
    }

    response = fyers.optionchain(data=payload)

    if response.get("s") != "ok":
        return pd.DataFrame(), None

    chain = response["data"]["optionsChain"]

    rows = []
    for opt in chain:
        if opt.get("option_type") not in ("CE", "PE"):
            continue

        rows.append({
            "Strike": opt.get("strike_price"),
            "Type": opt.get("option_type"),
            "LTP": opt.get("ltp"),
            "OI": opt.get("oi"),
            "Volume": opt.get("volume"),

            # GREEKS
            "Delta": opt.get("delta"),
            "Gamma": opt.get("gamma"),
            "Vega": opt.get("vega"),

            "Expiry": opt.get("expiry_date") or opt.get("expiry")
        })

    df = pd.DataFrame(rows)

    expiry_val = df["Expiry"].dropna().unique()
    expiry_text = pd.to_datetime(expiry_val[0]).strftime("%d %b %Y")

    # SPLIT CE / PE
    ce = df[df["Type"] == "CE"].rename(columns={
        "LTP": "Call Price",
        "OI": "Call OI",
        "Volume": "Call Volume",
        "Delta": "Call Delta",
        "Gamma": "Call Gamma",
        "Vega": "Call Vega"
    })

    pe = df[df["Type"] == "PE"].rename(columns={
        "LTP": "Put Price",
        "OI": "Put OI",
        "Volume": "Put Volume",
        "Delta": "Put Delta",
        "Gamma": "Put Gamma",
        "Vega": "Put Vega"
    })

    final_df = pd.merge(
        ce[[
            "Strike",
            "Call Price", "Call OI", "Call Volume",
            "Call Delta", "Call Gamma", "Call Vega"
        ]],
        pe[[
            "Strike",
            "Put Price", "Put OI", "Put Volume",
            "Put Delta", "Put Gamma", "Put Vega"
        ]],
        on="Strike",
        how="outer"
    ).sort_values("Strike")

    return final_df, expiry_text

# ===============================
# LOAD DATA
# ===============================
with st.spinner("Fetching live data..."):
    spot_price = get_spot_price()
    df, expiry = get_option_chain()

# ===============================
# HEADER
# ===============================
st.subheader(f"{index_name} Live Price: {spot_price}")
st.caption(f"📅 Expiry: {expiry} | 🔄 Auto-refresh every 15 seconds")

# ===============================
# ATM STRIKE
# ===============================
atm_strike = round(spot_price / strike_step) * strike_step

# ===============================
# HIGHLIGHT ATM ROW
# ===============================
def highlight_atm(row):
    if row["Strike"] == atm_strike:
        return ["background-color: #ffd6e7"] * len(row)
    return [""] * len(row)

# ===============================
# DISPLAY TABLE
# ===============================
if not df.empty:
    st.dataframe(
        df.style.apply(highlight_atm, axis=1),
        use_container_width=True
    )
else:
    st.warning("No option chain data available")



import streamlit as st
import streamlit.components.v1 as components
from fyers_apiv3 import fyersModel
import pandas as pd

# ===============================
# STREAMLIT CONFIG
# ===============================
st.set_page_config(page_title="Option Chain Dashboard", layout="wide")
st.title("📊 Option Chain Dashboard (FYERS)")

# 🔄 AUTO REFRESH EVERY 15 SECONDS (VERSION SAFE)
components.html(
    "<meta http-equiv='refresh' content='15'>",
    height=0,
)

# ===============================
# FYERS CREDENTIALS
# ===============================
CLIENT_ID = "3VEZHWB1VB-100"
ACCESS_TOKEN = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJhdWQiOlsiZDoxIiwieDowIiwieDoxIl0sImF0X2hhc2giOiJnQUFBQUFCcFJFaGxsclVBVTFyZVRqS3VucTZFS1FCMkx0UHZBLVZ6OU5hajJpQks3Tld4Z2RzRHJsSGNvd3lNZUtlRkM0SzdPX1pYRzRLSWZRS2NrYmpaR0h3QjRSQTdiWEg1TDdTY2sxdGlzTnM1RTR4T1hRUT0iLCJkaXNwbGF5X25hbWUiOiIiLCJvbXMiOiJLMSIsImhzbV9rZXkiOiIwNmUwMDA2NmU0NzNlOTAxM2JkZWI1MGM2NmFkZjYzNjYwYmUwYTQzNWRjZjU3YjUzYWQyOTJmMSIsImlzRGRwaUVuYWJsZWQiOiJOIiwiaXNNdGZFbmFibGVkIjoiTiIsImZ5X2lkIjoiRkFENDE5ODkiLCJhcHBUeXBlIjoxMDAsImV4cCI6MTc2NjE5MDYwMCwiaWF0IjoxNzY2MDgyNjYxLCJpc3MiOiJhcGkuZnllcnMuaW4iLCJuYmYiOjE3NjYwODI2NjEsInN1YiI6ImFjY2Vzc190b2tlbiJ9.R8ANyzeA1Lb0DOwLj4C3BZVyjHALLBEqFrbGWVpqM1Y"


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
    "NIFTY": {"symbol": "NSE:NIFTY50-INDEX"},
    "BANKNIFTY": {"symbol": "NSE:NIFTYBANK-INDEX"},
}

# ===============================
# DROPDOWN
# ===============================
index_name = st.selectbox("Select Index", ["NIFTY", "BANKNIFTY"])
symbol = INDEX_CONFIG[index_name]["symbol"]

# ===============================
# LIVE INDEX PRICE
# ===============================
def get_spot_price():
    res = fyers.quotes({"symbols": symbol})
    if res.get("s") != "ok":
        return None
    return res["d"][0]["v"]["lp"]

# ===============================
# OPTION CHAIN (NO GREEKS)
# ===============================
def get_option_chain():
    payload = {
        "symbol": symbol,
        "strikecount": 50,
        "timestamp": ""
    }

    response = fyers.optionchain(data=payload)

    if response.get("s") != "ok":
        return pd.DataFrame(), "Unknown"

    chain = response["data"].get("optionsChain", [])
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
            "Expiry": opt.get("expiry_date") or opt.get("expiry")
        })

    df = pd.DataFrame(rows)

    if df.empty:
        return pd.DataFrame(), "Unknown"

    expiry_vals = df["Expiry"].dropna().unique()
    expiry_text = (
        pd.to_datetime(expiry_vals[0]).strftime("%d %b %Y")
        if len(expiry_vals) > 0
        else "Unknown"
    )

    ce = df[df["Type"] == "CE"].rename(columns={
        "LTP": "Call Price",
        "OI": "Call OI",
        "Volume": "Call Volume"
    })

    pe = df[df["Type"] == "PE"].rename(columns={
        "LTP": "Put Price",
        "OI": "Put OI",
        "Volume": "Put Volume"
    })

    final_df = pd.merge(
        ce[["Strike", "Call Price", "Call OI", "Call Volume"]],
        pe[["Strike", "Put Price", "Put OI", "Put Volume"]],
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
if spot_price is not None:
    st.subheader(f"{index_name} Live Price: {spot_price}")
else:
    st.warning("Live price unavailable")

st.caption(f"📅 Expiry: {expiry} | 🔄 Auto-refresh every 15 seconds")

# ===============================
# FIND STRIKES AROUND SPOT
# ===============================
lower_strike = None
upper_strike = None

if spot_price is not None and not df.empty:
    strikes = sorted(df["Strike"].dropna().unique())
    lower_strike = max([s for s in strikes if s <= spot_price], default=None)
    upper_strike = min([s for s in strikes if s >= spot_price], default=None)

if lower_strike and upper_strike:
    st.caption(f"📍 Spot trading between {lower_strike} and {upper_strike}")

# ===============================
# HIGHLIGHT ROWS
# ===============================
def highlight_spot_range(row):
    if row["Strike"] in (lower_strike, upper_strike):
        return ["background-color: #ffffe0"] * len(row)
    return [""] * len(row)

# ===============================
# DISPLAY TABLE
# ===============================
if not df.empty:
    st.dataframe(
        df.style.apply(highlight_spot_range, axis=1),
        use_container_width=True
    )
else:
    st.warning("No option chain data available")


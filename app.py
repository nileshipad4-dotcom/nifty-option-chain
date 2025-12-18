import streamlit as st
import time
from fyers_apiv3 import fyersModel
import pandas as pd

st.set_page_config(page_title="NIFTY Option Chain", layout="wide")
st.title("📊 NIFTY Option Chain (FYERS)")

# 🔄 AUTO REFRESH EVERY 15 SECONDS
REFRESH_INTERVAL = 15

if "last_refresh" not in st.session_state:
    st.session_state.last_refresh = time.time()

if time.time() - st.session_state.last_refresh > REFRESH_INTERVAL:
    st.session_state.last_refresh = time.time()
    st.experimental_rerun()


# ===============================
# FYERS CREDENTIALS
# ⚠️ USE STREAMLIT SECRETS IN PROD
# ===============================
CLIENT_ID = "3VEZHWB1VB-100"     # e.g. ABCD1234-100
ACCESS_TOKEN = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJhdWQiOlsiZDoxIiwieDowIiwieDoxIl0sImF0X2hhc2giOiJnQUFBQUFCcFJFaGxsclVBVTFyZVRqS3VucTZFS1FCMkx0UHZBLVZ6OU5hajJpQks3Tld4Z2RzRHJsSGNvd3lNZUtlRkM0SzdPX1pYRzRLSWZRS2NrYmpaR0h3QjRSQTdiWEg1TDdTY2sxdGlzTnM1RTR4T1hRUT0iLCJkaXNwbGF5X25hbWUiOiIiLCJvbXMiOiJLMSIsImhzbV9rZXkiOiIwNmUwMDA2NmU0NzNlOTAxM2JkZWI1MGM2NmFkZjYzNjYwYmUwYTQzNWRjZjU3YjUzYWQyOTJmMSIsImlzRGRwaUVuYWJsZWQiOiJOIiwiaXNNdGZFbmFibGVkIjoiTiIsImZ5X2lkIjoiRkFENDE5ODkiLCJhcHBUeXBlIjoxMDAsImV4cCI6MTc2NjE5MDYwMCwiaWF0IjoxNzY2MDgyNjYxLCJpc3MiOiJhcGkuZnllcnMuaW4iLCJuYmYiOjE3NjYwODI2NjEsInN1YiI6ImFjY2Vzc190b2tlbiJ9.R8ANyzeA1Lb0DOwLj4C3BZVyjHALLBEqFrbGWVpqM1Y"   


# ===============================
# CONFIG
# ===============================
SYMBOL = "NSE:NIFTY50-INDEX"
STRIKE_COUNT = 50

# ===============================
# INIT FYERS
# ===============================
fyers = fyersModel.FyersModel(
    client_id=CLIENT_ID,
    token=ACCESS_TOKEN,
    is_async=False,
    log_path=""
)

# ===============================
# FETCH OPTION CHAIN
# ===============================
def get_nifty_option_chain():
    payload = {
        "symbol": SYMBOL,
        "strikecount": STRIKE_COUNT,
        "timestamp": ""
    }

    response = fyers.optionchain(data=payload)

    if response.get("s") != "ok":
        st.error(response)
        return pd.DataFrame(), "Unknown"

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
            "Expiry": opt.get("expiry_date") or opt.get("expiry")
        })

    df = pd.DataFrame(rows)

    if df.empty:
        return pd.DataFrame(), "Unknown"

    # Extract expiry
    expiry_val = df["Expiry"].dropna().unique()
    expiry_text = "Unknown"
    if len(expiry_val) > 0:
        expiry_text = pd.to_datetime(expiry_val[0]).strftime("%d %b %Y")

    # Split CE / PE
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
# LOAD DATA (AUTO)
# ===============================
with st.spinner("Fetching option chain..."):
    df, expiry = get_nifty_option_chain()

# ===============================
# DISPLAY
# ===============================
if not df.empty:
    st.subheader(f"📅 Expiry: {expiry}")
    st.caption("🔄 Auto-refreshing every 15 seconds")
    st.dataframe(df, use_container_width=True)
else:
    st.warning("No data received from FYERS")


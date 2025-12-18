
import streamlit as st
import streamlit.components.v1 as components
from fyers_apiv3 import fyersModel
import pandas as pd

# ===============================
# STREAMLIT CONFIG
# ===============================
st.set_page_config(page_title="Option Chain Dashboard", layout="wide")
st.title("📊 Option Chain Dashboard (FYERS)")

# 🔄 AUTO REFRESH EVERY 15 SECONDS
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
# OPTION CHAIN
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
        if len(expiry_vals) > 0 else "Unknown"
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
# PCR
# ===============================
def calculate_pcr(df):
    call_oi = df["Call OI"].fillna(0).sum()
    put_oi = df["Put OI"].fillna(0).sum()
    call_vol = df["Call Volume"].fillna(0).sum()
    put_vol = df["Put Volume"].fillna(0).sum()

    pcr_oi = round(put_oi / call_oi, 3) if call_oi > 0 else None
    pcr_vol = round(put_vol / call_vol, 3) if call_vol > 0 else None
    return pcr_oi, pcr_vol

# ===============================
# MAX PAIN (SCALED + INTEGER)
# ===============================
def compute_max_pain(df):
    strikes = df["Strike"].values
    call_oi = df["Call OI"].fillna(0).values
    put_oi = df["Put OI"].fillna(0).values

    total_pain = []
    for strike in strikes:
        call_loss = sum(max(0, strike - s) * oi for s, oi in zip(strikes, call_oi))
        put_loss  = sum(max(0, s - strike) * oi for s, oi in zip(strikes, put_oi))
        total_pain.append(int((call_loss + put_loss) / 10_000_000))  # ✅ NO DECIMALS

    df["Total Pain"] = total_pain
    return df

def get_max_pain_strike(df):
    if df.empty:
        return None
    return df.loc[df["Total Pain"].idxmin(), "Strike"]

# ===============================
# LOAD DATA
# ===============================
with st.spinner("Fetching live data..."):
    spot_price = get_spot_price()
    df, expiry = get_option_chain()

if not df.empty:
    pcr_oi, pcr_vol = calculate_pcr(df)
    df = compute_max_pain(df)
    max_pain_strike = get_max_pain_strike(df)
else:
    pcr_oi = pcr_vol = max_pain_strike = None

# ===============================
# HEADER
# ===============================
if spot_price:
    st.subheader(f"{index_name} Live Price: {spot_price}")
else:
    st.warning("Live price unavailable")

st.caption(f"📅 Expiry: {expiry} | 🔄 Auto-refresh every 15 seconds")

# ===============================
# TOP METRICS
# ===============================
c1, c2, c3 = st.columns(3)
c1.metric("PCR (OI)", pcr_oi if pcr_oi else "—")
c2.metric("PCR (Volume)", pcr_vol if pcr_vol else "—")
c3.metric("Max Pain Strike", max_pain_strike if max_pain_strike else "—")

# ===============================
# SPOT RANGE
# ===============================
lower_strike = upper_strike = None
if spot_price and not df.empty:
    strikes = sorted(df["Strike"].dropna().unique())
    lower_strike = max([s for s in strikes if s <= spot_price], default=None)
    upper_strike = min([s for s in strikes if s >= spot_price], default=None)

# ===============================
# HIGHLIGHT ROWS
# ===============================
def highlight_rows(row):
    styles = [""] * len(row)
    if row["Strike"] in (lower_strike, upper_strike):
        styles = ["background-color: #add8e6"] * len(row)
    if max_pain_strike and row["Strike"] == max_pain_strike:
        styles = ["background-color: #ffb347"] * len(row)
    return styles

# ===============================
# DISPLAY TABLE
# ===============================
if not df.empty:
    st.dataframe(
        df.style.apply(highlight_rows, axis=1),
        use_container_width=True
    )
else:
    st.warning("No option chain data available")

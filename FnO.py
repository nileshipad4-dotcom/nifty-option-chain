import streamlit as st
import pandas as pd
from kiteconnect import KiteConnect
from kite_config import KITE_API_KEY, KITE_ACCESS_TOKEN

# ----------------------------------
# Streamlit Config
# ----------------------------------
st.set_page_config(
    page_title="FnO Futures Open Interest",
    layout="wide"
)

st.title("📊 FnO Futures Open Interest (Kite API)")
st.caption("Source: Zerodha Kite Connect")

# ----------------------------------
# Kite Connection
# ----------------------------------
kite = KiteConnect(api_key=KITE_API_KEY)
kite.set_access_token(KITE_ACCESS_TOKEN)

# ----------------------------------
# UI Input
# ----------------------------------
symbol = st.text_input(
    "Enter FnO Stock Symbol (e.g. RELIANCE, TCS, INFY)",
    "RELIANCE"
).upper()

fetch_btn = st.button("Fetch Futures OI")

# ----------------------------------
# Fetch Data
# ----------------------------------
if fetch_btn:
    try:
        with st.spinner("Fetching data from Kite..."):

            # Load NFO instruments
            instruments = pd.DataFrame(kite.instruments("NFO"))

            # Filter FUTSTK for given symbol
            fut_df = instruments[
                (instruments["instrument_type"] == "FUT") &
                (instruments["tradingsymbol"].str.startswith(symbol))
            ]

            if fut_df.empty:
                st.error("❌ No Futures contracts found for this symbol.")
                st.stop()

            # Fetch live quotes
            tokens = fut_df["instrument_token"].tolist()
            quotes = kite.quote(tokens)

            data = []
            for _, row in fut_df.iterrows():
                token = row["instrument_token"]
                q = quotes.get(token, {})

                data.append({
                    "Trading Symbol": row["tradingsymbol"],
                    "Expiry": row["expiry"],
                    "Last Price": q.get("last_price"),
                    "Open Interest": q.get("oi"),
                    "Volume": q.get("volume"),
                    "OI Day High": q.get("oi_day_high"),
                    "OI Day Low": q.get("oi_day_low")
                })

            result_df = pd.DataFrame(data)
            result_df = result_df.sort_values("Expiry")

            # ----------------------------------
            # Display
            # ----------------------------------
            st.subheader(f"📌 Futures Open Interest for {symbol}")
            st.dataframe(result_df, use_container_width=True)

    except Exception as e:
        st.error(f"⚠️ Kite API Error: {e}")

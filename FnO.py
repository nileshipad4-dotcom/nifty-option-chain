import streamlit as st
import pandas as pd
from kiteconnect import KiteConnect
from kite_config import KITE_API_KEY, KITE_ACCESS_TOKEN

# -------------------------------------------------
# Streamlit Config
# -------------------------------------------------
st.set_page_config(
    page_title="FnO Futures Open Interest",
    layout="wide"
)

st.title("📊 FnO Futures Open Interest")
st.caption("Source: Zerodha Kite Connect (Market Quotes API)")

st.info(
    "ℹ️ Open Interest is available ONLY during market hours "
    "(9:15 AM – 3:30 PM IST)."
)

# -------------------------------------------------
# Kite Connection
# -------------------------------------------------
kite = KiteConnect(api_key=KITE_API_KEY)
kite.set_access_token(KITE_ACCESS_TOKEN)

# -------------------------------------------------
# Fetch Button
# -------------------------------------------------
if st.button("Fetch Futures Open Interest"):
    try:
        with st.spinner("Fetching data from Kite..."):

            # 1️⃣ Load NFO instruments
            instruments = pd.DataFrame(kite.instruments("NFO"))

            # 2️⃣ Filter stock futures
            fut_df = instruments[instruments["instrument_type"] == "FUT"]

            if fut_df.empty:
                st.error("No FnO Futures instruments found.")
                st.stop()

            # 3️⃣ Build exchange-qualified symbols (CRITICAL)
            fut_df["kite_symbol"] = (
                fut_df["exchange"] + ":" + fut_df["tradingsymbol"]
            )

            symbols = fut_df["kite_symbol"].tolist()

            # 4️⃣ Fetch quotes in batches
            BATCH_SIZE = 250
            rows = []

            for i in range(0, len(symbols), BATCH_SIZE):
                batch = symbols[i:i + BATCH_SIZE]
                quotes = kite.quote(batch)

                for sym in batch:
                    q = quotes.get(sym, {})
                    rows.append({
                        "Symbol": sym,
                        "Last Price": q.get("last_price"),
                        "Open Interest": q.get("oi"),
                        "OI Day High": q.get("oi_day_high"),
                        "OI Day Low": q.get("oi_day_low"),
                        "Volume": q.get("volume")
                    })

            # 5️⃣ DataFrame
            df = pd.DataFrame(rows)

            # Split symbol for better sorting
            df["Underlying"] = df["Symbol"].str.split(":").str[1].str.extract(r"([A-Z]+)")
            df = df.sort_values(["Underlying", "Symbol"])

            # 6️⃣ Display
            st.subheader("📌 Futures Open Interest – All FnO Stocks")
            st.dataframe(df, use_container_width=True)

            # 7️⃣ Download
            csv = df.to_csv(index=False).encode("utf-8")
            st.download_button(
                "⬇️ Download CSV",
                csv,
                "fno_futures_open_interest.csv",
                "text/csv"
            )

    except Exception as e:
        st.error(f"Kite API Error: {e}")

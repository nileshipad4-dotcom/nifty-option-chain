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

st.title("📊 FnO Futures – OI & Spot–Future Spread")
st.caption("Source: Zerodha Kite Connect")

st.warning(
    "⚠️ Data is LIVE. Open Interest is available only during market hours "
    "(9:15 AM – 3:30 PM IST)."
)

# -------------------------------------------------
# Kite Connection
# -------------------------------------------------
kite = KiteConnect(api_key=KITE_API_KEY)
kite.set_access_token(KITE_ACCESS_TOKEN)

# -------------------------------------------------
# Helper: Batched Quotes
# -------------------------------------------------
def batched_quotes(symbols, batch_size=250):
    quotes = {}
    for i in range(0, len(symbols), batch_size):
        batch = symbols[i:i + batch_size]
        quotes.update(kite.quote(batch))
    return quotes

# -------------------------------------------------
# MAIN LOGIC (AUTO RUN)
# -------------------------------------------------
try:
    with st.spinner("Fetching FnO futures data from Kite..."):

        # 1️⃣ Load all instruments
        instruments = pd.DataFrame(kite.instruments())

        # 2️⃣ Filter stock futures
        fut_df = instruments[
            (instruments["exchange"] == "NFO") &
            (instruments["instrument_type"] == "FUT")
        ].copy()

        if fut_df.empty:
            st.error("No FnO futures instruments found.")
            st.stop()

        # 3️⃣ Sort by expiry (CRITICAL)
        fut_df["expiry"] = pd.to_datetime(fut_df["expiry"])
        fut_df = fut_df.sort_values("expiry")

        # 4️⃣ Identify the 3 active expiries
        expiries = fut_df["expiry"].drop_duplicates().head(3).tolist()

        # 5️⃣ Build symbols
        fut_df["fut_symbol"] = "NFO:" + fut_df["tradingsymbol"]
        fut_df["spot_symbol"] = "NSE:" + fut_df["name"]

        fut_symbols = fut_df["fut_symbol"].tolist()
        spot_symbols = fut_df["spot_symbol"].unique().tolist()

        # 6️⃣ Fetch quotes
        fut_quotes = batched_quotes(fut_symbols)
        spot_quotes = batched_quotes(spot_symbols)

        # 7️⃣ Build final dataframe
        rows = []
        for _, row in fut_df.iterrows():
            fut_q = fut_quotes.get(row["fut_symbol"], {})
            spot_q = spot_quotes.get(row["spot_symbol"], {})

            fut_price = fut_q.get("last_price")
            spot_price = spot_q.get("last_price")
            lot_size = row["lot_size"]

            spread_value = None
            if fut_price is not None and spot_price is not None:
                spread_value = (spot_price - fut_price) * lot_size

            rows.append({
                "Underlying": row["name"],
                "Future Symbol": row["tradingsymbol"],
                "Expiry": row["expiry"].date(),
                "Lot Size": lot_size,
                "Underlying Price": spot_price,
                "Future Price": fut_price,
                "Open Interest": fut_q.get("oi"),
                "Spread Value": spread_value
            })

        df = pd.DataFrame(rows)

        # -------------------------------------------------
        # 8️⃣ DISPLAY: 3 TABLES (NEAR / NEXT / FAR)
        # -------------------------------------------------
        labels = ["🟢 Near Month", "🟡 Next Month", "🔵 Far Month"]

        for label, exp in zip(labels, expiries):
            st.subheader(f"{label} – Expiry: {exp.date()}")
            exp_df = df[df["Expiry"] == exp.date()].sort_values("Underlying")
            st.dataframe(exp_df, use_container_width=True)

        # -------------------------------------------------
        # 9️⃣ Download (Full Data)
        # -------------------------------------------------
        csv = df.to_csv(index=False).encode("utf-8")
        st.download_button(
            "⬇️ Download Full CSV",
            csv,
            "fno_futures_oi_spread_all_expiries.csv",
            "text/csv"
        )

except Exception as e:
    st.error(f"Kite API Error: {e}")

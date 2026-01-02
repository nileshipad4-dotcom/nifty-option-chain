import streamlit as st
import pandas as pd
import os
import numpy as np

# =====================================
# CONFIG
# =====================================
st.set_page_config(page_title="Max Pain Components", layout="wide")
st.title("📊 Max Pain Component Breakdown")

DATA_DIR = "data"

# =====================================
# LOAD CSV FILES
# =====================================
def load_csv_files():
    files = []
    if not os.path.exists(DATA_DIR):
        return files

    for f in os.listdir(DATA_DIR):
        if f.startswith("option_chain_") and f.endswith(".csv"):
            ts = f.replace("option_chain_", "").replace(".csv", "")
            files.append((ts, os.path.join(DATA_DIR, f)))

    return sorted(files, reverse=True)

csv_files = load_csv_files()

if not csv_files:
    st.error("No option chain CSV files found.")
    st.stop()

timestamps = [ts for ts, _ in csv_files]
file_map = dict(csv_files)

# =====================================
# DROPDOWN
# =====================================
selected_ts = st.selectbox(
    "Select Timestamp (Date-HH:MM)",
    timestamps
)

df = pd.read_csv(file_map[selected_ts])

# =====================================
# COMPUTE COMPONENTS
# =====================================
def compute_components(stock_df):
    stock_df = stock_df.sort_values("Strike").reset_index(drop=True)

    G = stock_df["Strike"].values
    B = stock_df["CE_OI"].fillna(0).values
    L = stock_df["PE_OI"].fillna(0).values

    call_vals = []
    put_vals = []

    for i in range(len(stock_df)):
        call_part = (
            G[i] * B[:i].sum()
            - (G[:i] * B[:i]).sum()
        )

        put_part = (
            (G[i:] * L[i:]).sum()
            - G[i] * L[i:].sum()
        )

        call_vals.append(int(call_part))
        put_vals.append(int(put_part))

    stock_df["Call_Component"] = call_vals
    stock_df["Put_Component"] = put_vals

    return stock_df

# =====================================
# APPLY PER STOCK
# =====================================
result = []
for stock, sdf in df.groupby("Stock"):
    result.append(compute_components(sdf))

final_df = pd.concat(result, ignore_index=True)

# =====================================
# DISPLAY
# =====================================
display_cols = [
    "Stock",
    "Strike",
    "Call_Component",
    "Put_Component",
    "CE_OI",
    "PE_OI",
    "Stock_LTP",
]

st.subheader(f"📅 Data: {selected_ts}")

st.dataframe(
    final_df[display_cols]
    .style.format({
        "Call_Component": "{:,.0f}",
        "Put_Component": "{:,.0f}",
        "Stock_LTP": "{:.2f}"
    }),
    use_container_width=True
)

# =====================================
# DOWNLOAD
# =====================================
st.download_button(
    "⬇️ Download Components CSV",
    final_df.to_csv(index=False),
    f"max_pain_components_{selected_ts}.csv",
    "text/csv"
)

import streamlit as st
import pandas as pd
import os
import numpy as np

# =====================================
# STREAMLIT CONFIG
# =====================================
st.set_page_config(page_title="Max Pain Components (2 Timestamps)", layout="wide")
st.title("📊 Max Pain Component Breakdown (2 Timestamps)")

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
# TIMESTAMP SELECTORS
# =====================================
c1, c2 = st.columns(2)

with c1:
    ts1 = st.selectbox("Timestamp 1 (LTP Reference)", timestamps, index=0)

with c2:
    ts2 = st.selectbox("Timestamp 2 (Comparison)", timestamps, index=1)

df1 = pd.read_csv(file_map[ts1])
df2 = pd.read_csv(file_map[ts2])

# =====================================
# COMPONENT CALCULATION
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
    stock_df["Net_Component"] = stock_df["Call_Component"] - stock_df["Put_Component"]

    return stock_df

# =====================================
# APPLY PER STOCK (BOTH TIMESTAMPS)
# =====================================
out = []

for stock in sorted(set(df1["Stock"]) & set(df2["Stock"])):
    s1 = compute_components(df1[df1["Stock"] == stock])
    s2 = compute_components(df2[df2["Stock"] == stock])

    merged = s1.merge(
        s2[["Strike", "Call_Component", "Put_Component", "Net_Component"]],
        on="Strike",
        suffixes=(f" ({ts1})", f" ({ts2})"),
        how="inner"
    )

    out.append(merged)

final_df = pd.concat(out, ignore_index=True)

# =====================================
# ATM HIGHLIGHT (USING TS1 LTP)
# =====================================
def highlight_atm_strikes(data):
    styles = pd.DataFrame("", index=data.index, columns=data.columns)

    for stock, sdf in data.groupby("Stock"):
        sdf = sdf.sort_values("Strike")
        ltp = sdf["Stock_LTP"].iloc[0]
        strikes = sdf["Strike"].values

        for i in range(len(strikes) - 1):
            if strikes[i] <= ltp <= strikes[i + 1]:
                styles.loc[sdf.index[i]] = "background-color:#003366;color:white"
                styles.loc[sdf.index[i + 1]] = "background-color:#003366;color:white"
                break

    return styles

# =====================================
# DISPLAY
# =====================================
display_cols = [
    "Stock",
    "Strike",
    "Call_Component (" + ts1 + ")",
    "Put_Component (" + ts1 + ")",
    "Net_Component (" + ts1 + ")",
    "Call_Component (" + ts2 + ")",
    "Put_Component (" + ts2 + ")",
    "Net_Component (" + ts2 + ")",
    "Stock_LTP",
]

st.subheader(f"📅 Comparison: {ts1} vs {ts2}")

st.dataframe(
    final_df[display_cols]
    .style
    .apply(highlight_atm_strikes, axis=None)
    .format({
        f"Call_Component ({ts1})": "{:,.0f}",
        f"Put_Component ({ts1})": "{:,.0f}",
        f"Net_Component ({ts1})": "{:,.0f}",
        f"Call_Component ({ts2})": "{:,.0f}",
        f"Put_Component ({ts2})": "{:,.0f}",
        f"Net_Component ({ts2})": "{:,.0f}",
        "Stock_LTP": "{:.2f}",
    }),
    use_container_width=True
)

# =====================================
# DOWNLOAD
# =====================================
st.download_button(
    "⬇️ Download Comparison CSV",
    final_df.to_csv(index=False),
    f"max_pain_components_{ts1}_vs_{ts2}.csv",
    "text/csv",
)

import streamlit as st
import pandas as pd
import numpy as np
import os

# =====================================
# CONFIG
# =====================================
st.set_page_config(page_title="ΔΔ Max Pain Viewer", layout="wide")
st.title("📊 ΔΔ Max Pain Viewer (Multi-Timestamp)")

DATA_DIR = "data"

# =====================================
# LOAD CSV FILES
# =====================================
def load_csv_files():
    files = []
    for f in os.listdir(DATA_DIR):
        if f.startswith("option_chain_") and f.endswith(".csv"):
            ts = f.replace("option_chain_", "").replace(".csv", "")
            files.append((ts, os.path.join(DATA_DIR, f)))
    return sorted(files, reverse=True)

csv_files = load_csv_files()
if len(csv_files) < 6:
    st.error("Need at least 6 option chain CSV files")
    st.stop()

timestamps = [ts for ts, _ in csv_files]
file_map = dict(csv_files)

# =====================================
# TIMESTAMP SELECTION
# =====================================
st.subheader("🕒 Timestamp Selection")

t1 = st.selectbox(
    "Base Timestamp (T1)",
    timestamps,
    index=0
)

other_ts = st.multiselect(
    "Select 5 Comparison Timestamps",
    [ts for ts in timestamps if ts != t1],
    default=[ts for ts in timestamps[1:6]],
    max_selections=5
)

if len(other_ts) != 5:
    st.warning("Please select exactly 5 comparison timestamps")
    st.stop()

# =====================================
# LOAD BASE DATA
# =====================================
df_base = pd.read_csv(file_map[t1])
required_cols = {"Stock", "Strike", "Max_Pain", "Stock_LTP"}

if not required_cols.issubset(df_base.columns):
    st.error("CSV must contain Stock, Strike, Max_Pain, Stock_LTP")
    st.stop()

df_base["Stock"] = df_base["Stock"].str.upper().str.strip()

# =====================================
# STOCK SELECTOR
# =====================================
stock_list = sorted(df_base["Stock"].unique())
selected_stock = st.selectbox("Select Stock", stock_list)

# =====================================
# START MERGED DF
# =====================================
df = df_base.copy()

# =====================================
# PROCESS EACH TIMESTAMP
# =====================================
for idx, ts in enumerate(other_ts, start=1):
    df_ts = pd.read_csv(file_map[ts])
    df_ts["Stock"] = df_ts["Stock"].str.upper().str.strip()

    df = df.merge(
        df_ts[["Stock", "Strike", "Max_Pain"]],
        on=["Stock", "Strike"],
        suffixes=("", f"_t{idx}")
    )

    # Δ MP
    delta_col = f"delta_12_{idx}"
    df[delta_col] = df["Max_Pain"] - df[f"Max_Pain_t{idx}"]

    # ΔΔ MP
    dd_col = f"ΔΔ MP{idx}"
    df[dd_col] = np.nan

    for stock, sdf in df.sort_values("Strike").groupby("Stock"):
        vals = sdf[delta_col].astype(float).values
        diff = vals - np.roll(vals, -1)
        diff[-1] = np.nan
        df.loc[sdf.index, dd_col] = diff

# =====================================
# FILTER SELECTED STOCK
# =====================================
sdf = (
    df[df["Stock"] == selected_stock]
    .sort_values("Strike")
    .reset_index(drop=True)
)

# =====================================
# ATM ±6 STRIKES
# =====================================
ltp = float(sdf["Stock_LTP"].iloc[0])
strikes = sdf["Strike"].values

atm_idx = None
for i in range(len(strikes) - 1):
    if strikes[i] <= ltp <= strikes[i + 1]:
        atm_idx = i
        break

if atm_idx is None:
    st.error("ATM strike not found")
    st.stop()

start = max(0, atm_idx - 6)
end = min(len(sdf), atm_idx + 2 + 6)

view_df = sdf.iloc[start:end]

# =====================================
# FINAL DISPLAY
# =====================================
cols = ["Stock", "Strike"] + [f"ΔΔ MP{i}" for i in range(1, 6)]

display_df = view_df[cols].copy()
display_df["Stock"] = selected_stock

st.subheader("📈 ΔΔ MP Comparison (ATM ±6)")

st.dataframe(
    display_df.style.format(
        {c: "{:.0f}" for c in display_df.columns if c not in {"Stock"}}
    ),
    use_container_width=True
)

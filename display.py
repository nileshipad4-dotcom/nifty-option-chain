import streamlit as st
import pandas as pd
import numpy as np
import os

# =====================================
# CONFIG
# =====================================
st.set_page_config(page_title="ΔΔ Max Pain Viewer", layout="wide")
st.title("📊 ΔΔ Max Pain Viewer")

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

def short_ts(ts):
    """Return HH:MM from timestamp string"""
    return ts.split("_")[-1].replace("-", ":")

# =====================================
# TIMESTAMP SELECTION
# =====================================
st.subheader("🕒 Timestamp Selection")

t1 = st.selectbox(
    "Base Timestamp (T1)",
    timestamps,
    index=0
)

c1, c2, c3, c4, c5 = st.columns(5)

with c1:
    t2 = st.selectbox("T2", timestamps, index=1, key="t2")
with c2:
    t3 = st.selectbox("T3", timestamps, index=2, key="t3")
with c3:
    t4 = st.selectbox("T4", timestamps, index=3, key="t4")
with c4:
    t5 = st.selectbox("T5", timestamps, index=4, key="t5")
with c5:
    t6 = st.selectbox("T6", timestamps, index=5, key="t6")

compare_ts = [t2, t3, t4, t5, t6]

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
# PROCESS EACH COMPARISON TIMESTAMP
# =====================================
for ts in compare_ts:
    time_label = short_ts(ts)   # column name (HH:MM)

    df_ts = pd.read_csv(file_map[ts])
    df_ts["Stock"] = df_ts["Stock"].str.upper().str.strip()

    df = df.merge(
        df_ts[["Stock", "Strike", "Max_Pain"]],
        on=["Stock", "Strike"],
        suffixes=("", "_cmp")
    )

    # Δ MP (T1 − Ti)
    delta_col = f"_delta_{time_label}"
    df[delta_col] = df["Max_Pain"] - df["Max_Pain_cmp"]

    # ΔΔ MP column named ONLY by time
    df[time_label] = np.nan

    for stock, sdf in df.sort_values("Strike").groupby("Stock"):
        vals = sdf[delta_col].astype(float).values
        diff = vals - np.roll(vals, -1)
        diff[-1] = np.nan
        df.loc[sdf.index, time_label] = diff

    # clean temp column
    df.drop(columns=["Max_Pain_cmp", delta_col], inplace=True)

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
time_cols = [short_ts(ts) for ts in compare_ts]

display_df = view_df[["Stock", "Strike"] + time_cols].copy()
display_df["Stock"] = selected_stock

st.subheader("📈 ΔΔ MP (Base → Selected Times)")

st.dataframe(
    display_df.style.format(
        {c: "{:.0f}" for c in time_cols}
    ),
    use_container_width=True
)

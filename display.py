import streamlit as st
import pandas as pd
import numpy as np
import os

# ===============================
# CONFIG
# ===============================
st.set_page_config(page_title="ΔΔ Max Pain Viewer", layout="wide")
st.title("📊 ΔΔ Max Pain Viewer")

DATA_DIR = "data"

# ===============================
# LOAD CSV FILES
# ===============================
def load_csv_files():
    files = []
    for f in os.listdir(DATA_DIR):
        if f.startswith("option_chain_") and f.endswith(".csv"):
            ts = f.replace("option_chain_", "").replace(".csv", "")
            files.append((ts, os.path.join(DATA_DIR, f)))
    return sorted(files, reverse=True)

csv_files = load_csv_files()
if len(csv_files) < 2:
    st.error("Need at least 2 option chain CSV files")
    st.stop()

timestamps = [ts for ts, _ in csv_files]
file_map = dict(csv_files)

def short_ts(ts):
    return ts.split("_")[-1].replace("-", ":")

# ===============================
# TIMESTAMP SELECTION
# ===============================
st.subheader("🕒 Select Base Timestamp")

t1 = st.selectbox("Timestamp 1 (Base)", timestamps, index=0)
t2 = st.selectbox("Timestamp 2 (Shown as Column)", timestamps, index=1)

t1_lbl = short_ts(t1)
t2_lbl = short_ts(t2)

# ===============================
# LOAD DATA
# ===============================
df1 = pd.read_csv(file_map[t1])
df2 = pd.read_csv(file_map[t2])

required_cols = {"Stock", "Strike", "Max_Pain", "Stock_LTP"}
if not required_cols.issubset(df1.columns) or not required_cols.issubset(df2.columns):
    st.error("CSV files must contain Stock, Strike, Max_Pain, Stock_LTP")
    st.stop()

df1["Stock"] = df1["Stock"].str.upper().str.strip()
df2["Stock"] = df2["Stock"].str.upper().str.strip()

# ===============================
# MERGE
# ===============================
df = df1.merge(
    df2[["Stock", "Strike", "Max_Pain"]],
    on=["Stock", "Strike"],
    suffixes=("", "_prev")
)

# ===============================
# delta_12 (MP difference across time)
# ===============================
df["delta_12"] = df["Max_Pain"] - df["Max_Pain_prev"]

# ===============================
# delta_above_col (ΔΔ MP)
# ===============================
delta_above_col = f"ΔΔ MP ({t1_lbl}-{t2_lbl})"
df[delta_above_col] = np.nan

for stock, sdf in df.sort_values("Strike").groupby("Stock"):
    vals = sdf["delta_12"].astype(float).values
    diff = vals - np.roll(vals, -1)
    diff[-1] = np.nan
    df.loc[sdf.index, delta_above_col] = diff

# ===============================
# STOCK SELECTOR
# ===============================
stock_list = sorted(df["Stock"].unique())
selected_stock = st.selectbox("Select Stock", stock_list)

sdf = (
    df[df["Stock"] == selected_stock]
    .sort_values("Strike")
    .reset_index(drop=True)
)

# ===============================
# ATM ±6 STRIKES
# ===============================
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

# ===============================
# FINAL DISPLAY TABLE
# ===============================
display_df = pd.DataFrame({
    "Stock": [selected_stock] * len(view_df),
    "Strike": view_df["Strike"].values,
    delta_above_col: view_df[delta_above_col].values
})

st.subheader(f"📈 ΔΔ MP View")

st.dataframe(
    display_df.style.format({
        "Strike": "{:.0f}",
        delta_above_col: "{:.0f}"
    }),
    use_container_width=True
)

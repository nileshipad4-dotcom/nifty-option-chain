import streamlit as st
import pandas as pd
import numpy as np
import os
from datetime import time
from streamlit_autorefresh import st_autorefresh

# =====================================
# AUTO REFRESH
# =====================================
st_autorefresh(interval=360_000, key="auto_refresh")

# =====================================
# STREAMLIT CONFIG
# =====================================
st.set_page_config(page_title="FnO Max Pain Snapshot", layout="wide")
st.title("📊 FnO Stock Max Pain Snapshot")

DATA_DIR = "data"
FNO_FILE = "FnO.csv"   # <-- Lot size file

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
if len(csv_files) < 3:
    st.error("Need at least 3 option chain CSV files.")
    st.stop()

timestamps_all = [ts for ts, _ in csv_files]
file_map = dict(csv_files)

# =====================================
# TIME FILTERING
# =====================================
def extract_time(ts):
    try:
        t = ts.split("_")[-1]
        hh, mm = map(int, t.split("-")[:2])
        return time(hh, mm)
    except Exception:
        return None

START_TIME = time(7, 30)
END_TIME = time(16, 0)

filtered_ts = [
    ts for ts in timestamps_all
    if extract_time(ts) and START_TIME <= extract_time(ts) <= END_TIME
]

# =====================================
# TIMESTAMP SELECTORS
# =====================================
cols = st.columns(3)

with cols[0]:
    t1 = st.selectbox("Timestamp 1", filtered_ts, 0)
with cols[1]:
    t2 = st.selectbox("Timestamp 2", filtered_ts, 1)
with cols[2]:
    t3 = st.selectbox("Timestamp 3", filtered_ts, 2)

labels = [x.split("_")[-1].replace("-", ":") for x in [t1, t2, t3]]

# =====================================
# LOAD OPTION CHAIN DATA
# =====================================
dfs = []
timestamps = [t1, t2, t3]

for i, ts in enumerate(timestamps):
    d = pd.read_csv(file_map[ts])
    d = d[["Stock", "Strike", "Max_Pain", "Stock_LTP"]].rename(
        columns={
            "Max_Pain": f"MP_{i}",
            "Stock_LTP": f"LTP_{i}",
        }
    )
    dfs.append(d)

# =====================================
# LOAD TIMESTAMP 1 EXTRA DATA
# =====================================
raw_t1 = pd.read_csv(file_map[t1])

extra_cols = ["Stock_%_Change", "Stoch_High", "Stoch_Low"]
for col in extra_cols:
    dfs[0][col] = raw_t1[col] if col in raw_t1.columns else np.nan

# =====================================
# MERGE OPTION CHAIN DATA
# =====================================
df = dfs[0]
for i in range(1, 3):
    df = df.merge(dfs[i], on=["Stock", "Strike"], how="inner")

# =====================================
# CLEAN STOCK NAMES
# =====================================
df["Stock"] = df["Stock"].str.upper().str.strip()

EXCLUDE = {"NIFTY", "BANKNIFTY", "FINNIFTY", "MIDCPNIFTY"}
df = df[~df["Stock"].isin(EXCLUDE)]

# =====================================
# LOAD & MERGE FnO LOT SIZE
# =====================================
if not os.path.exists(FNO_FILE):
    st.error("FnO.csv (lot size file) not found.")
    st.stop()

lot_df = pd.read_csv(FNO_FILE)
lot_df["name"] = lot_df["name"].str.upper().str.strip()

df = df.merge(
    lot_df.rename(columns={"name": "Stock"}),
    on="Stock",
    how="left"
)

# =====================================
# FINAL REQUIRED TABLE
# =====================================
final_table = df[[
    "Stock",
    "lot_size",
    "Strike",
    "MP_0",
    "MP_1",
    "MP_2",
    "LTP_0",
    "Stock_%_Change",
    "Stoch_High",
    "Stoch_Low"
]].rename(columns={
    "lot_size": "Lot_Size",
    "MP_0": f"Max Pain ({labels[0]})",
    "MP_1": f"Max Pain ({labels[1]})",
    "MP_2": f"Max Pain ({labels[2]})",
    "LTP_0": "Stock_LTP"
})

# =====================================
# STRIKE FORMATTER
# =====================================
def format_strike(x):
    if pd.isna(x):
        return ""
    if float(x).is_integer():
        return f"{int(x)}"
    return f"{x:.2f}"

# =====================================
# DISPLAY
# =====================================
st.subheader("📌 Max Pain Snapshot (with FnO Lot Size)")

st.dataframe(
    final_table.style.format({
        "Strike": format_strike,
        "Stock_LTP": "{:.2f}",
        "Stock_%_Change": "{:.3f}",
        "Stoch_High": "{:.2f}",
        "Stoch_Low": "{:.2f}",
        "Lot_Size": "{:.0f}",
        **{c: "{:.0f}" for c in final_table.columns if "Max Pain" in c},
    }),
    use_container_width=True,
)

# =====================================
# DOWNLOAD
# =====================================
st.download_button(
    "⬇️ Download CSV",
    final_table.to_csv(index=False),
    "max_pain_snapshot_with_lot_size.csv",
    "text/csv",
)

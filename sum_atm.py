import streamlit as st
import pandas as pd
import numpy as np
import os
from datetime import time

# ==================================================
# PAGE CONFIG
# ==================================================
st.set_page_config(page_title="Σ ATM", layout="centered")
st.title("📊 Σ ATM")

DATA_DIR = "data"

# ==================================================
# LOAD CSV FILES
# ==================================================
def load_csv_files():
    files = []
    for f in os.listdir(DATA_DIR):
        if f.startswith("option_chain_") and f.endswith(".csv"):
            ts = f.replace("option_chain_", "").replace(".csv", "")
            files.append((ts, os.path.join(DATA_DIR, f)))
    return sorted(files)

csv_files = load_csv_files()

if len(csv_files) < 2:
    st.error("Need at least 2 CSV files")
    st.stop()

timestamps = [ts for ts, _ in csv_files]
file_map = dict(csv_files)

# ==================================================
# TIME EXTRACTION
# ==================================================
def extract_time(ts):
    try:
        hh, mm = map(int, ts.split("_")[-1].split("-")[:2])
        return time(hh, mm)
    except:
        return None

timestamps = [ts for ts in timestamps if extract_time(ts)]

# ==================================================
# AUTO TS2 → FIRST AFTER 09:16
# ==================================================
def first_after_916(ts_list):
    for ts in ts_list:
        if extract_time(ts) >= time(9, 16):
            return ts
    return ts_list[0]

default_ts2 = first_after_916(timestamps)

# ==================================================
# USER INPUTS
# ==================================================
st.subheader("⏱ Settings")

c1, c2 = st.columns(2)

ts1 = c1.selectbox(
    "Timestamp 1",
    timestamps,
    index=len(timestamps) - 1
)

ts2 = c2.selectbox(
    "Timestamp 2 (Reference)",
    timestamps,
    index=timestamps.index(default_ts2)
)

X = st.number_input(
    "Strike Window X (±X strikes around ATM)",
    min_value=1,
    max_value=10,
    value=2,
    step=1
)

# ==================================================
# LOAD DATA
# ==================================================
df1 = pd.read_csv(file_map[ts1])
df2 = pd.read_csv(file_map[ts2])

# ==================================================
# REQUIRED COLUMNS (UNCHANGED LOGIC)
# ==================================================
df1 = df1[["Stock", "Strike", "Stock_LTP", "CE_OI", "PE_OI"]]
df2 = df2[["Stock", "Strike", "Stock_LTP", "CE_OI", "PE_OI"]]

df1.columns = ["Stock", "Strike", "ltp_1", "ce_1", "pe_1"]
df2.columns = ["Stock", "Strike", "ltp_2", "ce_2", "pe_2"]

df = df1.merge(df2, on=["Stock", "Strike"])

# ==================================================
# NUMERIC SAFETY
# ==================================================
for c in df.columns:
    if c != "Stock":
        df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0)

# ==================================================
# CORE CALCULATIONS (UNCHANGED)
# ==================================================
df["d_ce"] = df["ce_1"] - df["ce_2"]
df["d_pe"] = df["pe_1"] - df["pe_2"]

df["ce_x"] = (df["d_ce"] * df["Strike"]) / 10000
df["pe_x"] = (df["d_pe"] * df["Strike"]) / 10000

df["diff"] = df["pe_x"] - df["ce_x"]

# ==================================================
# ATM WINDOW CALCULATION (FULL LOGIC)
# ==================================================
df["atm_diff"] = np.nan

for stock, g in df.groupby("Stock"):
    g = g.sort_values("Strike").reset_index()

    ltp = g["ltp_1"].iloc[0]
    atm_idx = (g["Strike"] - ltp).abs().values.argmin()

    low = max(0, atm_idx - X)
    high = min(len(g) - 1, atm_idx + X)

    atm_value = g.loc[low:high, "diff"].mean()
    df.loc[g["index"], "atm_diff"] = atm_value

df["atm_diff"] = df["atm_diff"].fillna(0)

# ==================================================
# FINAL Σ ATM (PER STOCK → SUM)
# ==================================================
sigma_atm = df.groupby("Stock")["atm_diff"].first().sum() / 1000

# ==================================================
# DISPLAY ONLY RESULT
# ==================================================
st.markdown("## Σ ATM")
st.markdown(f"### **{sigma_atm:.2f}**")
st.caption(f"TS1: {ts1} | TS2: {ts2} | ATM Window: ±{X}")

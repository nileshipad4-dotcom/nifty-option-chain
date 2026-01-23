import streamlit as st
import pandas as pd
import numpy as np
import os
from datetime import time

# ==================================================
# CONFIG
# ==================================================
st.set_page_config(page_title="Σ ATM Calculator", layout="centered")
st.title("📊 Σ ATM Calculator (2 Timestamp Mode)")

DATA_DIR = "data"

# ==================================================
# LOAD FILES
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
    st.error("Need at least 2 CSV files.")
    st.stop()

timestamps_all = [ts for ts, _ in csv_files]
file_map = dict(csv_files)

# ==================================================
# EXTRACT TIME FROM FILENAME
# ==================================================
def extract_time(ts):
    try:
        hh, mm = map(int, ts.split("_")[-1].split("-")[:2])
        return time(hh, mm)
    except:
        return None

# Filter trading session
filtered_ts = [ts for ts in timestamps_all if extract_time(ts)]

# ==================================================
# AUTO FIND FIRST TS AFTER 09:16
# ==================================================
def find_first_after_916(ts_list):
    for ts in ts_list:
        t = extract_time(ts)
        if t and t >= time(9, 16):
            return ts
    return ts_list[0]

default_ts2 = find_first_after_916(filtered_ts)

# ==================================================
# USER INPUTS
# ==================================================
st.subheader("⏱ Timestamp Selection")

c1, c2 = st.columns(2)

ts1 = c1.selectbox("Timestamp 1 (Current)", filtered_ts, index=len(filtered_ts)-1)
ts2 = c2.selectbox("Timestamp 2 (Reference)", filtered_ts, index=filtered_ts.index(default_ts2))

X = st.number_input("ATM Window X (strikes each side)", 1, 10, 2)

# ==================================================
# LOAD DATA
# ==================================================
df1 = pd.read_csv(file_map[ts1])
df2 = pd.read_csv(file_map[ts2])

# ==================================================
# KEEP REQUIRED COLUMNS
# ==================================================
df1 = df1[["Stock","Strike","Stock_LTP","CE_OI","PE_OI"]]
df2 = df2[["Stock","Strike","Stock_LTP","CE_OI","PE_OI"]]

df1 = df1.rename(columns={"Stock_LTP":"ltp1","CE_OI":"ce1","PE_OI":"pe1"})
df2 = df2.rename(columns={"Stock_LTP":"ltp2","CE_OI":"ce2","PE_OI":"pe2"})

df = df1.merge(df2, on=["Stock","Strike"])

# ==================================================
# NUMERIC SAFETY
# ==================================================
for c in df.columns:
    if c != "Stock":
        df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0)

# ==================================================
# CORE CALCULATIONS
# ==================================================
df["d_ce"] = df["ce1"] - df["ce2"]
df["d_pe"] = df["pe1"] - df["pe2"]

df["ce_x"] = (df["d_ce"] * df["Strike"]) / 10000
df["pe_x"] = (df["d_pe"] * df["Strike"]) / 10000
df["diff"] = df["pe_x"] - df["ce_x"]

# ==================================================
# ATM WINDOW CALCULATION
# ==================================================
df["atm_diff"] = np.nan

for stk, g in df.groupby("Stock"):
    g = g.sort_values("Strike").reset_index()

    ltp = g["ltp1"].iloc[0]
    atm_idx = (g["Strike"] - ltp).abs().values.argmin()

    low = max(0, atm_idx - X)
    high = min(len(g)-1, atm_idx + X)

    window = g.loc[low:high, "diff"]
    atm_avg = window.mean()

    df.loc[g["index"], "atm_diff"] = atm_avg

# ==================================================
# FINAL Σ ATM
# ==================================================
df["atm_diff"] = pd.to_numeric(df["atm_diff"], errors="coerce").fillna(0)

sum_atm = df.groupby("Stock")["atm_diff"].first().sum() / 1000
up_atm = (df.groupby("Stock")["atm_diff"].first() > 0).sum()

# ==================================================
# DISPLAY ONLY RESULT
# ==================================================
st.markdown("## 📊 RESULT")
st.markdown(f"### 🟢 UP Stocks: **{up_atm}**")
st.markdown(f"### Σ ATM (in K): **{sum_atm:.2f}**")
st.caption(f"TS1 = {ts1} | TS2 = {ts2} | Window = ±{X} strikes")

import streamlit as st
import pandas as pd
import numpy as np
import os
from datetime import time

# ==================================================
# CONFIG
# ==================================================
st.set_page_config(page_title="ATM Diff", layout="centered")
st.title("📊 ATM Diff (Stock-wise)")

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
    return sorted(files, reverse=True)

csv_files = load_csv_files()

if len(csv_files) < 2:
    st.error("Need at least 2 CSV files")
    st.stop()

timestamps_all = [ts for ts, _ in csv_files]
file_map = dict(csv_files)

# ==================================================
# TIME FILTER (08:00–16:00)
# ==================================================
def extract_time(ts):
    try:
        hh, mm = map(int, ts.split("_")[-1].split("-")[:2])
        return time(hh, mm)
    except:
        return None

filtered_ts = [
    ts for ts in timestamps_all
    if extract_time(ts) and time(8, 0) <= extract_time(ts) <= time(16, 0)
]

# ==================================================
# DEFAULT TS2 → FIRST AFTER 09:16
# ==================================================
def first_after_916(ts_list):
    for ts in reversed(ts_list):
        if extract_time(ts) >= time(9, 16):
            return ts
    return ts_list[-1]

default_ts2 = first_after_916(filtered_ts)

# ==================================================
# USER INPUT
# ==================================================
c1, c2, c3 = st.columns(3)

t1 = c1.selectbox("Timestamp 1", filtered_ts, index=0)
t2 = c2.selectbox(
    "Timestamp 2 (Reference)",
    filtered_ts,
    index=filtered_ts.index(default_ts2)
)

X = c3.number_input("Strike Window X", 1, 10, 4)

# ==================================================
# LOAD DATA
# ==================================================
df1 = pd.read_csv(file_map[t1])
df2 = pd.read_csv(file_map[t2])

df1 = df1[["Stock","Strike","Stock_LTP","CE_OI","PE_OI"]]
df2 = df2[["Stock","Strike","Stock_LTP","CE_OI","PE_OI"]]

df1.columns = ["Stock","Strike","ltp_0","ce_0","pe_0"]
df2.columns = ["Stock","Strike","ltp_1","ce_1","pe_1"]

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
df["d_ce"] = df["ce_0"] - df["ce_1"]
df["d_pe"] = df["pe_0"] - df["pe_1"]

df["ce_x"] = (df["d_ce"] * df["Strike"]) / 10000
df["pe_x"] = (df["d_pe"] * df["Strike"]) / 10000

df["diff"] = np.nan
df["atm_diff"] = np.nan

# ==================================================
# SLIDING WINDOW + ATM (FIXED)
# ==================================================
for stk, g in df.groupby("Stock"):
    g = g.sort_values("Strike").reset_index()

    # ---- sliding window diff ----
    for i in range(len(g)):
        low = max(0, i - X)
        high = min(len(g) - 1, i + X)

        diff_val = g.loc[low:high, "pe_x"].sum() - g.loc[low:high, "ce_x"].sum()
        df.at[g.loc[i, "index"], "diff"] = diff_val

    # 🔑 RELOAD diff INTO g (THIS WAS MISSING)
    g["diff"] = df.loc[g["index"], "diff"].values

    # ---- ATM diff (±2 fixed) ----
    ltp = g["ltp_0"].iloc[0]
    atm_idx = (g["Strike"] - ltp).abs().values.argmin()

    low = max(0, atm_idx - 2)
    high = min(len(g) - 1, atm_idx + 2)

    atm_avg = g.loc[low:high, "diff"].mean()
    df.loc[g["index"], "atm_diff"] = atm_avg

df["atm_diff"] = df["atm_diff"].fillna(0)

# ==================================================
# FINAL OUTPUT
# ==================================================
result = (
    df.groupby("Stock", as_index=False)["atm_diff"]
    .first()
    .sort_values("atm_diff", ascending=False)
)

st.dataframe(result, use_container_width=True)
st.caption(f"TS1: {t1} | TS2: {t2} | X: {X}")

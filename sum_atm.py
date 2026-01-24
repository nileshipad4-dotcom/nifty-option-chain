import streamlit as st
import pandas as pd
import numpy as np
import os
import base64
import requests
from datetime import time

# ==================================================
# CONFIG
# ==================================================
st.set_page_config(page_title="ATM Diff Dashboard", layout="wide")
st.title("📊 ATM Diff Dashboard")

DATA_DIR = "data"
CACHE_DIR = "data_atm"
os.makedirs(CACHE_DIR, exist_ok=True)

# ==================================================
# GITHUB CONFIG
# ==================================================
GITHUB_TOKEN = st.secrets["GITHUB_TOKEN"]
KITE_REPO = st.secrets["KITE_REPO"]
GITHUB_BRANCH = st.secrets.get("GITHUB_BRANCH", "main")

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
if not csv_files:
    st.error("No option_chain CSV files found")
    st.stop()

timestamps_all = [ts for ts, _ in csv_files]
file_map = dict(csv_files)

# ==================================================
# TIME HELPERS
# ==================================================
def extract_time(ts):
    hh, mm = map(int, ts.split("_")[-1].split("-")[:2])
    return time(hh, mm)

filtered_ts = [
    ts for ts in timestamps_all
    if time(8, 0) <= extract_time(ts) <= time(16, 0)
]

def first_after_916(ts_list):
    for ts in ts_list:
        if extract_time(ts) >= time(9, 16):
            return ts
    return ts_list[0]

default_ts2 = first_after_916(filtered_ts)

# ==================================================
# USER INPUT
# ==================================================
c1, c2, c3, c4 = st.columns(4)
t1 = c1.selectbox("Timestamp 1 (Current)", filtered_ts, index=len(filtered_ts)-1)
t2 = c2.selectbox("Timestamp 2 (Reference)", filtered_ts, index=filtered_ts.index(default_ts2))
X = c3.number_input("Strike Window X", 1, 10, 4)
Y = c4.number_input("Window Y", 4, 20, 6)
K = 4

# ==================================================
# ATM CALCULATION
# ==================================================
def compute_atm_per_stock(ts1, ts2, X):
    df1 = pd.read_csv(file_map[ts1])
    df2 = pd.read_csv(file_map[ts2])

    df1 = df1[["Stock","Strike","Stock_LTP","CE_OI","PE_OI"]]
    df2 = df2[["Stock","Strike","Stock_LTP","CE_OI","PE_OI"]]

    df1.columns = ["Stock","Strike","ltp0","ce0","pe0"]
    df2.columns = ["Stock","Strike","ltp1","ce1","pe1"]

    df = df1.merge(df2, on=["Stock","Strike"])

    for c in df.columns:
        if c != "Stock":
            df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0)

    df["ce_x"] = (df["ce0"] - df["ce1"]) * df["Strike"] / 10000
    df["pe_x"] = (df["pe0"] - df["pe1"]) * df["Strike"] / 10000
    df["diff"] = np.nan
    df["atm_diff"] = np.nan

    for stk, g in df.groupby("Stock"):
        g = g.sort_values("Strike").reset_index()

        for i in range(len(g)):
            lo, hi = max(0, i-X), min(len(g)-1, i+X)
            df.at[g.loc[i,"index"], "diff"] = (
                g.loc[lo:hi,"pe_x"].sum()
                - g.loc[lo:hi,"ce_x"].sum()
            )

        ltp = g["ltp0"].iloc[0]
        atm_i = (g["Strike"] - ltp).abs().idxmin()
        df.loc[g.loc[atm_i,"index"], "atm_diff"] = (
            g.loc[max(0,atm_i-2):atm_i+2,"diff"].mean()
        )

    return df.groupby("Stock")["atm_diff"].first()

# ==================================================
# BUILD STOCK_DF
# ==================================================
ref_time = extract_time(t2).strftime("%H%M")
stock_path = os.path.join(CACHE_DIR, f"stock_ref_{ref_time}.csv")

stock_df = pd.read_csv(stock_path) if os.path.exists(stock_path) \
    else pd.DataFrame(columns=["time","stock","atm_diff"])

valid_ts = [
    ts for ts in filtered_ts
    if extract_time(ts) > extract_time(t2)
    and extract_time(ts) <= extract_time(t1)
]

for ts in valid_ts:
    t_str = extract_time(ts).strftime("%H:%M")
    if not stock_df[stock_df["time"] == t_str].empty:
        continue

    series = compute_atm_per_stock(ts, t2, X)
    for stk, v in series.items():
        stock_df.loc[len(stock_df)] = [t_str, stk, round(v, 0)]

stock_df = stock_df.drop_duplicates(["time","stock"])
stock_df.to_csv(stock_path, index=False)

# ==================================================
# PIVOT
# ==================================================

pivot_df = (
    stock_df
    .pivot(index="stock", columns="time", values="atm_diff")
    .sort_index()
    .round(2)
)
cols = list(pivot_df.columns)

# ==================================================
# LIS / LDS
# ==================================================
def lis_length(arr):
    d = []
    for x in arr:
        i = np.searchsorted(d, x)
        if i == len(d): d.append(x)
        else: d[i] = x
    return len(d)

def lds_length(arr):
    return lis_length([-x for x in arr])

# ==================================================
# HIGHLIGHT
# ==================================================
def highlight_segments(data):
    styles = pd.DataFrame("", index=data.index, columns=data.columns)

    for stock in data.index:
        values = data.loc[stock, cols].values

        for start in range(len(values) - Y + 1):
            w = values[start:start+Y]
            if np.isnan(w).any():
                continue

            target_cols = cols[start:start+Y]

            if lis_length(w) >= K:
                styles.loc[stock, target_cols] = "background-color:#c6efce"
            elif lds_length(w) >= K:
                styles.loc[stock, target_cols] = "background-color:#ffc7ce"

    return styles

# ==================================================
# DISPLAY
# ==================================================
st.markdown("### 📊 ATM Diff Pattern Table")

styled = (
    pivot_df
    .style
    .format("{:.2f}")        # ✅ forces 2 decimal display
    .apply(highlight_segments, axis=None)
)

st.dataframe(styled, use_container_width=True)


st.caption(
    f"Window={Y}, Subsequence≥{K} | "
    f"Green=Increasing, Red=Decreasing | Ref TS2={t2}"
)

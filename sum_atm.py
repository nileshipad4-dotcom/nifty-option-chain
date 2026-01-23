import streamlit as st
import pandas as pd
import numpy as np
import os
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
# PRICE CONTEXT (% CHANGE)
# ==================================================
df1 = pd.read_csv(file_map[t1])[["Stock", "Stock_%_Change", "Stock_LTP"]]
df2 = pd.read_csv(file_map[t2])[["Stock", "Stock_LTP"]]

df1.columns = ["stock", "Total_%", "ltp1"]
df2.columns = ["stock", "ltp2"]

price_df = df1.merge(df2, on="stock", how="left")

price_df["Δ%"] = np.where(
    price_df["ltp2"] != 0,
    ((price_df["ltp1"] - price_df["ltp2"]) / price_df["ltp2"]) * 100,
    0
)

# 🔒 remove decimals
price_df["Total_%"] = price_df["Total_%"].round(0).astype(int)
price_df["Δ%"] = price_df["Δ%"].round(0).astype(int)

price_df = price_df.set_index("stock")[["Total_%", "Δ%"]]

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

    out = df.groupby("Stock")["atm_diff"].first()
    return out.round(0).astype(int)   # 🔒 integers only

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
        stock_df.loc[len(stock_df)] = [t_str, stk, v]

stock_df = stock_df.drop_duplicates(["time","stock"])
stock_df.to_csv(stock_path, index=False)

# ==================================================
# PIVOT
# ==================================================
pivot_df = stock_df.pivot(index="stock", columns="time", values="atm_diff").sort_index()
cols = list(pivot_df.columns)

# ==================================================
# LIS / LDS
# ==================================================
def lis_length(arr):
    d = []
    for x in arr:
        i = np.searchsorted(d, x)
        if i == len(d):
            d.append(x)
        else:
            d[i] = x
    return len(d)

def lds_length(arr):
    return lis_length([-x for x in arr])

# ==================================================
# HIGHLIGHT + COUNTS
# ==================================================
style_mask = pd.DataFrame("", index=pivot_df.index, columns=pivot_df.columns)
green_count, red_count = {}, {}

for stock in pivot_df.index:
    values = pivot_df.loc[stock, cols].values
    gcols, rcols = set(), set()

    for start in range(len(values) - Y + 1):
        w = values[start:start+Y]
        if np.isnan(w).any():
            continue

        target_cols = cols[start:start+Y]

        if lis_length(w) >= K:
            style_mask.loc[stock, target_cols] = "background-color:#c6efce"
            gcols.update(target_cols)
        elif lds_length(w) >= K:
            style_mask.loc[stock, target_cols] = "background-color:#ffc7ce"
            rcols.update(target_cols)

    green_count[stock] = len(gcols)
    red_count[stock] = len(rcols)

# ==================================================
# FINAL TABLE
# ==================================================
meta = price_df.copy()
meta["Green_TS1_TS2"] = pd.Series(green_count)
meta["Red_TS1_TS2"] = pd.Series(red_count)

final = meta.join(pivot_df)

styled = final.style.apply(
    lambda _: style_mask.loc[_.index, _.columns.intersection(style_mask.columns)],
    axis=None
)

st.markdown("### 📊 ATM Diff Pattern Table")
st.dataframe(styled, use_container_width=True)

st.caption(
    f"All values shown as integers | "
    f"Window={Y}, Subsequence≥{K} | Ref TS2={t2}"
)

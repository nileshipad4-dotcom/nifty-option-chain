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
# GITHUB CONFIG (FROM SECRETS)
# ==================================================
GITHUB_TOKEN = st.secrets["GITHUB_TOKEN"]
KITE_REPO = st.secrets["KITE_REPO"]
GITHUB_BRANCH = st.secrets.get("GITHUB_BRANCH", "main")

GITHUB_API = "https://api.github.com"
HEADERS = {
    "Authorization": f"token {GITHUB_TOKEN}",
    "Accept": "application/vnd.github.v3+json"
}

def push_file_to_github(local_path, repo_path, msg):
    url = f"{GITHUB_API}/repos/{KITE_REPO}/contents/{repo_path}"
    with open(local_path, "rb") as f:
        content = base64.b64encode(f.read()).decode()

    r = requests.get(url, headers=HEADERS)
    sha = r.json().get("sha") if r.status_code == 200 else None

    payload = {"message": msg, "content": content, "branch": GITHUB_BRANCH}
    if sha:
        payload["sha"] = sha

    requests.put(url, headers=HEADERS, json=payload)

# ==================================================
# LOAD OPTION CHAIN FILES
# ==================================================
def load_csv_files():
    out = []
    for f in os.listdir(DATA_DIR):
        if f.startswith("option_chain_") and f.endswith(".csv"):
            ts = f.replace("option_chain_", "").replace(".csv", "")
            out.append((ts, os.path.join(DATA_DIR, f)))
    return sorted(out)

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
t1 = c1.selectbox("Timestamp 1 (Current)", filtered_ts, index=len(filtered_ts) - 1)
t2 = c2.selectbox("Timestamp 2 (Reference)", filtered_ts, index=filtered_ts.index(default_ts2))
X = c3.number_input("Strike Window X", 1, 10, 4)
Y = c4.number_input("Window Y", 4, 20, 6)

K = 4

# ==================================================
# ATM CALCULATION (UNCHANGED – WORKING)
# ==================================================
def compute_atm_per_stock(ts1, ts2, X):
    df1 = pd.read_csv(file_map[ts1])
    df2 = pd.read_csv(file_map[ts2])

    df1 = df1[["Stock", "Strike", "Stock_LTP", "CE_OI", "PE_OI"]]
    df2 = df2[["Stock", "Strike", "Stock_LTP", "CE_OI", "PE_OI"]]

    df1.columns = ["Stock", "Strike", "ltp0", "ce0", "pe0"]
    df2.columns = ["Stock", "Strike", "ltp1", "ce1", "pe1"]

    df = df1.merge(df2, on=["Stock", "Strike"])

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
            lo, hi = max(0, i - X), min(len(g) - 1, i + X)
            df.at[g.loc[i, "index"], "diff"] = (
                g.loc[lo:hi, "pe_x"].sum()
                - g.loc[lo:hi, "ce_x"].sum()
            )

        ltp = g["ltp0"].iloc[0]
        atm_i = (g["Strike"] - ltp).abs().values.argmin()
        atm_avg = g.loc[max(0, atm_i - 2):atm_i + 2, "diff"].mean()

        df.loc[g["index"], "atm_diff"] = atm_avg

    return df.groupby("Stock")["atm_diff"].first()

# ==================================================
# BUILD / UPDATE stock_ref CSV (FIXED BUG)
# ==================================================
ref_tag = extract_time(t2).strftime("%H%M")
stock_csv = f"stock_ref_{ref_tag}.csv"
stock_path = os.path.join(CACHE_DIR, stock_csv)

stock_df = pd.read_csv(stock_path) if os.path.exists(stock_path) \
    else pd.DataFrame(columns=["time", "stock", "atm_diff"])

valid_ts = [
    ts for ts in filtered_ts
    if time(9, 16) <= extract_time(ts) <= time(15, 45)
    and extract_time(ts) > extract_time(t2)   # ✅ FIX
]

for ts in valid_ts:
    t_str = extract_time(ts).strftime("%H:%M")
    if not stock_df[stock_df["time"] == t_str].empty:
        continue

    series = compute_atm_per_stock(ts, t2, X)
    for stk, v in series.items():
        stock_df.loc[len(stock_df)] = [t_str, stk, round(v, 0)]

stock_df = stock_df.drop_duplicates(["time", "stock"])
stock_df.to_csv(stock_path, index=False)

push_file_to_github(
    stock_path,
    f"{CACHE_DIR}/{stock_csv}",
    f"update {stock_csv}"
)

# ==================================================
# Σ ATM_DIFF TABLE
# ==================================================
sigma_df = (
    stock_df.groupby("time", as_index=False)["atm_diff"]
    .sum()
    .rename(columns={"atm_diff": "Σ_ATM"})
)

st.subheader("Σ ATM_DIFF (TS2 → TS1)")
st.dataframe(sigma_df, use_container_width=True)

# ==================================================
# PIVOT (TS2 → TS1)
# ==================================================
pivot_df = stock_df.pivot(index="stock", columns="time", values="atm_diff").sort_index()
pivot_df = pivot_df.loc[:, sorted(pivot_df.columns)]

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
cols = list(pivot_df.columns)
styles = pd.DataFrame("", index=pivot_df.index, columns=pivot_df.columns)
green_cnt, red_cnt = {}, {}

for stock in pivot_df.index:
    vals = pivot_df.loc[stock].values
    gset, rset = set(), set()

    for i in range(len(vals) - Y + 1):
        w = vals[i:i + Y]
        if np.isnan(w).any():
            continue

        c = cols[i:i + Y]

        if lis_length(w) >= K:
            styles.loc[stock, c] = "background-color:#c6efce"
            gset.update(c)
        elif lds_length(w) >= K:
            styles.loc[stock, c] = "background-color:#ffc7ce"
            rset.update(c)

    green_cnt[stock] = len(gset)
    red_cnt[stock] = len(rset)

final_df = pd.DataFrame({
    "G": pd.Series(green_cnt),
    "R": pd.Series(red_cnt)
}).join(pivot_df).fillna(0)

# ==================================================
# DISPLAY FINAL TABLE
# ==================================================
st.markdown("### 📊 ATM Diff Pattern Table (TS2 → TS1)")

styled = (
    final_df
    .style
    .format("{:.0f}")
    .apply(lambda _: styles, axis=None, subset=pivot_df.columns)
)

st.dataframe(styled, use_container_width=True)

st.caption(
    f"Window={Y}, Subsequence≥{K} | "
    f"G=Green count, R=Red count | "
    f"Range: {t2} → {t1}"
)

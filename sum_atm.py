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
KITE_REPO = st.secrets["KITE_REPO"]   # username/repo
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
    files = []
    for f in os.listdir(DATA_DIR):
        if f.startswith("option_chain_") and f.endswith(".csv"):
            ts = f.replace("option_chain_", "").replace(".csv", "")
            files.append((ts, os.path.join(DATA_DIR, f)))
    return sorted(files)

csv_files = load_csv_files()
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
# ATM CALC (UNCHANGED – WORKING)
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
        atm_i = (g["Strike"] - ltp).abs().values.argmin()
        atm_avg = g.loc[max(0,atm_i-2):atm_i+2, "diff"].mean()
        df.loc[g["index"], "atm_diff"] = atm_avg

    return df.groupby("Stock")["atm_diff"].first()

# ==================================================
# BUILD / APPEND stock_ref CSV (UNCHANGED LOGIC)
# ==================================================
ref_time = extract_time(t2).strftime("%H%M")
stock_csv = f"stock_ref_{ref_time}.csv"
stock_path = os.path.join(CACHE_DIR, stock_csv)

if os.path.exists(stock_path):
    stock_df = pd.read_csv(stock_path)
else:
    stock_df = pd.DataFrame(columns=["time","stock","atm_diff"])

valid_ts = [
    ts for ts in filtered_ts
    if time(9,16) <= extract_time(ts) <= time(15,45)
    and ts > t2
]

for ts in valid_ts:
    t_str = extract_time(ts).strftime("%H:%M")
    if not stock_df[stock_df["time"] == t_str].empty:
        continue

    atm_series = compute_atm_per_stock(ts, t2, X)
    for stk, val in atm_series.items():
        stock_df.loc[len(stock_df)] = [t_str, stk, round(val, 2)]

stock_df.to_csv(stock_path, index=False)

# ✅ PUSH RAW CSV
push_file_to_github(
    stock_path,
    f"{CACHE_DIR}/{stock_csv}",
    f"update {stock_csv}"
)

# ==================================================
# Σ ATM_DIFF (AND PUSH)
# ==================================================
sigma_df = (
    stock_df.groupby("time", as_index=False)["atm_diff"]
    .sum()
)

sigma_csv = f"sigma_atm_{ref_time}.csv"
sigma_path = os.path.join(CACHE_DIR, sigma_csv)
sigma_df.to_csv(sigma_path, index=False)

push_file_to_github(
    sigma_path,
    f"{CACHE_DIR}/{sigma_csv}",
    f"update {sigma_csv}"
)

# ==================================================
# DISPLAY (UNCHANGED)
# ==================================================
pivot_df = stock_df.pivot(index="stock", columns="time", values="atm_diff").sort_index()

st.subheader("Σ ATM_DIFF Over Time")
st.dataframe(sigma_df, use_container_width=True)

st.subheader("ATM Diff Pattern Table")
st.dataframe(pivot_df, use_container_width=True)

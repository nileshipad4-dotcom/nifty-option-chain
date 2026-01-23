import streamlit as st
import pandas as pd
import numpy as np
import os
import base64
import requests
from datetime import time, datetime

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

# ==================================================
# GITHUB PUSH
# ==================================================
def push_file_to_github(local_path, repo_path, msg):
    url = f"{GITHUB_API}/repos/{KITE_REPO}/contents/{repo_path}"

    with open(local_path, "rb") as f:
        content = base64.b64encode(f.read()).decode("utf-8")

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
c1, c2, c3 = st.columns(3)
t1 = c1.selectbox("Timestamp 1 (Current)", filtered_ts, index=len(filtered_ts)-1)
t2 = c2.selectbox("Timestamp 2 (Reference)", filtered_ts, index=filtered_ts.index(default_ts2))
X = c3.number_input("Strike Window X", 1, 10, 4)

# ==================================================
# ATM CALC (PER STOCK)
# ==================================================
def compute_atm_per_stock(ts1, ts2, X):
    df1 = pd.read_csv(file_map[ts1])
    df2 = pd.read_csv(file_map[ts2])

    df1 = df1[["Stock","Strike","Stock_LTP","CE_OI","PE_OI"]]
    df2 = df2[["Stock","Strike","Stock_LTP","CE_OI","PE_OI"]]

    df1.columns = ["Stock","Strike","ltp_0","ce_0","pe_0"]
    df2.columns = ["Stock","Strike","ltp_1","ce_1","pe_1"]

    df = df1.merge(df2, on=["Stock","Strike"])

    for c in df.columns:
        if c != "Stock":
            df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0)

    df["ce_x"] = (df["ce_0"] - df["ce_1"]) * df["Strike"] / 10000
    df["pe_x"] = (df["pe_0"] - df["pe_1"]) * df["Strike"] / 10000
    df["diff"] = np.nan
    df["atm_diff"] = np.nan

    for stk, g in df.groupby("Stock"):
        g = g.sort_values("Strike").reset_index()

        for i in range(len(g)):
            low = max(0, i - X)
            high = min(len(g)-1, i + X)
            df.at[g.loc[i,"index"], "diff"] = (
                g.loc[low:high,"pe_x"].sum()
                - g.loc[low:high,"ce_x"].sum()
            )

        g["diff"] = df.loc[g["index"], "diff"].values
        ltp = g["ltp_0"].iloc[0]
        atm_idx = (g["Strike"] - ltp).abs().values.argmin()
        atm_avg = g.loc[max(0,atm_idx-2):atm_idx+2, "diff"].mean()
        df.loc[g["index"], "atm_diff"] = atm_avg

    return df.groupby("Stock")["atm_diff"].first()

# ==================================================
# STOCK_REF CSV (SINGLE FILE PER TS2)
# ==================================================
st.markdown("---")
st.subheader("📈 Stock-wise ATM_DIFF Over Time (Cached)")

ref_time = extract_time(t2).strftime("%H%M")
stock_csv = f"stock_ref_{ref_time}.csv"
stock_path = os.path.join(CACHE_DIR, stock_csv)
repo_stock_path = f"{CACHE_DIR}/{stock_csv}"

if os.path.exists(stock_path):
    stock_df = pd.read_csv(stock_path)
else:
    stock_df = pd.DataFrame(columns=["time", "stock", "atm_diff"])

valid_ts = [
    ts for ts in filtered_ts
    if time(9,16) <= extract_time(ts) <= time(11,45)
    and ts > t2
]

progress = st.progress(0)

for i, ts in enumerate(valid_ts, start=1):
    progress.progress(i / len(valid_ts))
    t_str = extract_time(ts).strftime("%H:%M")

    existing_times = stock_df[stock_df["time"] == t_str]["stock"].tolist()

    atm_series = None
    missing_stocks = []

    # Determine which stocks need calculation
    if not existing_times:
        missing_stocks = None  # calculate all
    else:
        all_stocks = compute_atm_per_stock(t1, t2, X).index.tolist()
        missing_stocks = [s for s in all_stocks if s not in existing_times]

    if missing_stocks is None or missing_stocks:
        atm_series = compute_atm_per_stock(ts, t2, X)

        for stk, val in atm_series.items():
            if missing_stocks is None or stk in missing_stocks:
                stock_df.loc[len(stock_df)] = [
                    t_str,
                    stk,
                    round(val, 2)
                ]

        stock_df.to_csv(stock_path, index=False)

        push_file_to_github(
            stock_path,
            repo_stock_path,
            f"Update stock_ref ATM cache ref {ref_time}"
        )

# ==================================================
# DISPLAY (STOCK FILTER)
# ==================================================
available_stocks = sorted(stock_df["stock"].unique().tolist())
selected_stock = st.selectbox("Select Stock", available_stocks)

if selected_stock:
    view_df = stock_df[stock_df["stock"] == selected_stock] \
        .sort_values("time")

    st.dataframe(
        view_df[["time", "atm_diff"]],
        use_container_width=True
    )

st.caption(f"Stock cache CSV: {stock_csv}")

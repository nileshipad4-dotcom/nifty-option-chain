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
# LOAD FILES
# ==================================================
def load_csv_files():
    return sorted([
        (f.replace("option_chain_", "").replace(".csv", ""), os.path.join(DATA_DIR, f))
        for f in os.listdir(DATA_DIR)
        if f.startswith("option_chain_")
    ])

csv_files = load_csv_files()
timestamps_all = [ts for ts, _ in csv_files]
file_map = dict(csv_files)

def extract_time(ts):
    h, m = map(int, ts.split("_")[-1].split("-")[:2])
    return time(h, m)

filtered_ts = [ts for ts in timestamps_all if time(8,0) <= extract_time(ts) <= time(16,0)]

def first_after_916(ts_list):
    for ts in ts_list:
        if extract_time(ts) >= time(9,16):
            return ts
    return ts_list[0]

# ==================================================
# USER INPUT
# ==================================================
c1, c2, c3, c4 = st.columns(4)
t1 = c1.selectbox("TS1", filtered_ts, index=len(filtered_ts)-1)
t2 = c2.selectbox("TS2", filtered_ts, index=filtered_ts.index(first_after_916(filtered_ts)))
X = c3.number_input("Strike X", 1, 10, 4)
Y = c4.number_input("Window Y", 4, 20, 6)
K = 4

# ==================================================
# ATM CALC (UNCHANGED)
# ==================================================
def compute_atm_per_stock(ts1, ts2, X):
    d1 = pd.read_csv(file_map[ts1])
    d2 = pd.read_csv(file_map[ts2])

    d1 = d1[["Stock","Strike","Stock_LTP","CE_OI","PE_OI"]]
    d2 = d2[["Stock","Strike","Stock_LTP","CE_OI","PE_OI"]]
    d1.columns = ["Stock","Strike","ltp0","ce0","pe0"]
    d2.columns = ["Stock","Strike","ltp1","ce1","pe1"]

    df = d1.merge(d2, on=["Stock","Strike"])

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
            lo, hi = max(0,i-X), min(len(g)-1,i+X)
            df.at[g.loc[i,"index"],"diff"] = g.loc[lo:hi,"pe_x"].sum() - g.loc[lo:hi,"ce_x"].sum()

        atm_i = (g["Strike"] - g["ltp0"].iloc[0]).abs().idxmin()
        atm_val = g.loc[max(0,atm_i-2):atm_i+2,"diff"].mean()
        df.loc[g["index"],"atm_diff"] = atm_val

    return df.groupby("Stock")["atm_diff"].first()

# ==================================================
# BUILD STOCK_DF (NO ROUNDING HERE!)
# ==================================================
stock_df = pd.DataFrame(columns=["time","stock","atm_diff"])

valid_ts = [
    ts for ts in filtered_ts
    if extract_time(ts) > extract_time(t2)
    and extract_time(ts) <= extract_time(t1)
]

for ts in valid_ts:
    t_str = extract_time(ts).strftime("%H:%M")
    series = compute_atm_per_stock(ts, t2, X)
    for stk, v in series.items():
        stock_df.loc[len(stock_df)] = [t_str, stk, float(v)]

# ==================================================
# PUSH CSVs
# ==================================================
ref_tag = extract_time(t2).strftime("%H%M")

raw_path = os.path.join(CACHE_DIR, f"stock_ref_{ref_tag}.csv")
stock_df.to_csv(raw_path, index=False)
push_file_to_github(raw_path, f"{CACHE_DIR}/stock_ref_{ref_tag}.csv", "update raw atm")

sigma_df = stock_df.groupby("time", as_index=False)["atm_diff"].sum()
sigma_path = os.path.join(CACHE_DIR, f"sigma_atm_{ref_tag}.csv")
sigma_df.to_csv(sigma_path, index=False)
push_file_to_github(sigma_path, f"{CACHE_DIR}/sigma_atm_{ref_tag}.csv", "update sigma atm")

# ==================================================
# DISPLAY (ROUND ONLY HERE)
# ==================================================
st.subheader("Σ ATM_DIFF")
st.dataframe(sigma_df.round(0), use_container_width=True)

pivot = stock_df.pivot(index="stock", columns="time", values="atm_diff").sort_index()

st.subheader("ATM Diff Pattern Table")
st.dataframe(pivot.round(0), use_container_width=True)

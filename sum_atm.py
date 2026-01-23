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
# GITHUB PUSH FUNCTION
# ==================================================
def push_file_to_github(local_path, repo_path, commit_msg):
    url = f"{GITHUB_API}/repos/{KITE_REPO}/contents/{repo_path}"

    with open(local_path, "rb") as f:
        content = base64.b64encode(f.read()).decode("utf-8")

    r = requests.get(url, headers=HEADERS)
    sha = r.json().get("sha") if r.status_code == 200 else None

    payload = {
        "message": commit_msg,
        "content": content,
        "branch": GITHUB_BRANCH
    }
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
# Σ ATM_DIFF (CURRENT)
# ==================================================
current_sigma = compute_atm_per_stock(t1, t2, X).sum() / 100
st.markdown(f"### Σ ATM_DIFF : **{current_sigma:.2f}**")

# ==================================================
# CACHE FILE (DATE + TS2)
# ==================================================
date_str = datetime.now().strftime("%Y-%m-%d")
ref_time = extract_time(t2).strftime("%H%M")
cache_name = f"atm_{date_str}_ref_{ref_time}.csv"
cache_path = os.path.join(CACHE_DIR, cache_name)
repo_path = f"{CACHE_DIR}/{cache_name}"

if os.path.exists(cache_path):
    cache_df = pd.read_csv(cache_path)
else:
    cache_df = pd.DataFrame(columns=["timestamp1","time","sigma_atm_diff"])

# ==================================================
# Σ ATM_DIFF OVER TIME (CACHE + LIVE)
# ==================================================
st.markdown("---")
st.subheader("🕒 Σ ATM_DIFF Over Time (Fixed Reference)")

progress = st.progress(0)
status = st.empty()

rows = []
valid_ts = [
    ts for ts in filtered_ts
    if time(9,16) <= extract_time(ts) <= time(15,45)
    and ts > t2
]

for i, ts in enumerate(valid_ts, start=1):
    progress.progress(i / len(valid_ts))
    t = extract_time(ts)

    cached = cache_df[cache_df["timestamp1"] == ts]
    if not cached.empty:
        sigma_val = cached["sigma_atm_diff"].iloc[0]
        status.write(f"⚡ Loaded {t.strftime('%H:%M')} from cache")
    else:
        status.write(f"⏳ Calculating {t.strftime('%H:%M')}")
        sigma_val = compute_atm_per_stock(ts, t2, X).sum() / 100

        cache_df.loc[len(cache_df)] = [
            ts, t.strftime("%H:%M"), round(sigma_val, 2)
        ]
        cache_df.to_csv(cache_path, index=False)

        push_file_to_github(
            cache_path,
            repo_path,
            f"Update ATM cache {date_str} ref {ref_time}"
        )

    rows.append({
        "Time": t.strftime("%H:%M"),
        "Σ ATM_DIFF": round(sigma_val, 2)
    })

status.write("✅ Completed")
st.dataframe(pd.DataFrame(rows), use_container_width=True)

# ==================================================
# STOCK-WISE ATM_DIFF OVER TIME
# ==================================================
st.markdown("---")
st.subheader("📈 Stock-wise ATM_DIFF Over Time")

latest_stocks = sorted(
    compute_atm_per_stock(t1, t2, X).index.tolist()
)

selected_stocks = st.multiselect(
    "Select Stock(s)",
    latest_stocks
)

if selected_stocks:
    rows = []
    progress2 = st.progress(0)

    for i, ts in enumerate(valid_ts, start=1):
        progress2.progress(i / len(valid_ts))
        atm_series = compute_atm_per_stock(ts, t2, X)

        for stk in selected_stocks:
            rows.append({
                "Time": extract_time(ts).strftime("%H:%M"),
                "Stock": stk,
                "ATM_DIFF": round(atm_series.get(stk, 0), 2)
            })

    st.dataframe(pd.DataFrame(rows), use_container_width=True)

st.caption(f"Reference TS2: {t2} | Strike Window X: {X}")

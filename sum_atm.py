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
t1 = c1.selectbox("Timestamp 1 (Current)", filtered_ts, index=len(filtered_ts)-1)
t2 = c2.selectbox("Timestamp 2 (Reference)", filtered_ts, index=filtered_ts.index(default_ts2))
X  = c3.number_input("Strike Window X", 1, 10, 4)
Y  = c4.number_input("Window Y", 4, 20, 4)

K = 4

ts1_time = extract_time(t1)
ts2_time = extract_time(t2)
if ts1_time < ts2_time:
    ts1_time, ts2_time = ts2_time, ts1_time

# ==================================================
# ATM CALCULATION (UNCHANGED, CORRECT)
# ==================================================
def compute_atm_per_stock(ts1, ts2, X):
    df1 = pd.read_csv(file_map[ts1])
    df2 = pd.read_csv(file_map[ts2])

    df1 = df1[["Stock","Strike","Stock_LTP","CE_OI","PE_OI"]]
    df2 = df2[["Stock","Strike","Stock_LTP","CE_OI","PE_OI"]]

    df1.columns = ["Stock","Strike","ltp0","ce0","pe0"]
    df2.columns = ["Stock","Strike","ltp1","ce1","pe1"]

    df = df1.merge(df2, on=["Stock","Strike"])

    for c in ["Strike","ltp0","ce0","pe0","ce1","pe1"]:
        df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0)

    df["ce_x"] = (df["ce0"] - df["ce1"]) * df["Strike"] / 10000
    df["pe_x"] = (df["pe0"] - df["pe1"]) * df["Strike"] / 10000

    df["diff"] = np.nan
    df["atm_diff"] = np.nan

    for stk, g in df.groupby("Stock"):
        g = g.sort_values("Strike").reset_index()

        for i in range(len(g)):
            lo, hi = max(0,i-X), min(len(g)-1,i+X)
            df.at[g.loc[i,"index"],"diff"] = (
                g.loc[lo:hi,"pe_x"].sum()
                - g.loc[lo:hi,"ce_x"].sum()
            )

        ltp = g["ltp0"].iloc[0]
        atm_i = (g["Strike"] - ltp).abs().values.argmin()
        lo, hi = max(0,atm_i-2), min(len(g)-1,atm_i+2)

        atm_avg = df.loc[g.loc[lo:hi,"index"],"diff"].mean()
        df.loc[g["index"],"atm_diff"] = atm_avg

    return df.groupby("Stock")["atm_diff"].first()

# ==================================================
# BUILD STOCK_DF (TS2 → TS1)
# ==================================================
valid_ts = [
    ts for ts in filtered_ts
    if ts2_time <= extract_time(ts) <= ts1_time
]

rows = []
for ts in valid_ts:
    t_str = extract_time(ts).strftime("%H:%M")
    s = compute_atm_per_stock(ts, t2, X)
    for stk, v in s.items():
        rows.append([t_str, stk, round(v, 0)])

stock_df = pd.DataFrame(rows, columns=["time","stock","atm_diff"]).drop_duplicates()

ref_tag = ts2_time.strftime("%H%M")
stock_csv = f"stock_ref_{ref_tag}.csv"
stock_path = os.path.join(CACHE_DIR, stock_csv)
stock_df.to_csv(stock_path, index=False)

push_file_to_github(stock_path, f"{CACHE_DIR}/{stock_csv}", "Update ATM stock table")

# ==================================================
# Σ ATM TABLE
# ==================================================
sigma_df = stock_df.groupby("time", as_index=False)["atm_diff"].sum()
sigma_csv = f"sigma_atm_{ref_tag}.csv"
sigma_path = os.path.join(CACHE_DIR, sigma_csv)
sigma_df.to_csv(sigma_path, index=False)

push_file_to_github(sigma_path, f"{CACHE_DIR}/{sigma_csv}", "Update Σ ATM table")

st.subheader("Σ ATM_DIFF (TS2 → TS1)")
st.dataframe(sigma_df, use_container_width=True)

# ==================================================
# ATM DIFF PATTERN TABLE
# ==================================================
pivot_df = stock_df.pivot(index="stock", columns="time", values="atm_diff").sort_index()
cols = list(pivot_df.columns)

def lis_length(a):
    d=[]
    for x in a:
        i=np.searchsorted(d,x)
        if i==len(d): d.append(x)
        else: d[i]=x
    return len(d)

def lds_length(a):
    return lis_length([-x for x in a])

styles = pd.DataFrame("", index=pivot_df.index, columns=pivot_df.columns)
G,R={},{}

for stk in pivot_df.index:
    v=pivot_df.loc[stk].values
    gs,rs=set(),set()
    for i in range(len(v)-Y+1):
        w=v[i:i+Y]
        if np.isnan(w).any(): continue
        c=cols[i:i+Y]
        if lis_length(w)>=K:
            styles.loc[stk,c]="background-color:#c6efce"
            gs.update(c)
        elif lds_length(w)>=K:
            styles.loc[stk,c]="background-color:#ffc7ce"
            rs.update(c)
    G[stk]=len(gs); R[stk]=len(rs)

final_df = pd.DataFrame({"G":G,"R":R}).join(pivot_df).fillna(0)

st.markdown("### 📊 ATM Diff Pattern Table (TS2 → TS1)")
st.dataframe(
    final_df.style.format("{:.0f}")
    .apply(lambda _: styles, axis=None, subset=pivot_df.columns),
    use_container_width=True
)

# ==================================================
# EXTRA RANGE ANALYSIS (FIXED)
# ==================================================
st.markdown("### 📉 ATM & % Change Comparison (09:00 → 15:30)")

range_ts = [
    ts for ts in filtered_ts
    if time(9,0) <= extract_time(ts) <= time(15,30)
]

cA,cB=st.columns(2)
new_ts1=cA.selectbox("New TS1",range_ts,0)
new_ts2=cB.selectbox("New TS2",range_ts,len(range_ts)-1)

if extract_time(new_ts1)>extract_time(new_ts2):
    new_ts1,new_ts2=new_ts2,new_ts1

# % CHANGE
dfA=pd.read_csv(file_map[new_ts1])[["Stock","Stock_LTP","Stock_%_Change"]]
dfB=pd.read_csv(file_map[new_ts2])[["Stock","Stock_LTP","Stock_%_Change"]]

dfA.columns=["Stock","ltpA","totA"]
dfB.columns=["Stock","ltpB","totB"]

pct=dfA.merge(dfB,on="Stock")
pct["pct_TS"]=(pct["ltpB"]-pct["ltpA"])/pct["ltpA"]*100

# ATM DELTA (FIXED REFERENCE = t2)
atmA=compute_atm_per_stock(new_ts1,t2,X)
atmB=compute_atm_per_stock(new_ts2,t2,X)

atm=pd.DataFrame({"Stock":atmA.index,"Δ_ATM":atmB.values-atmA.values})

final_extra=pct.merge(atm,on="Stock")[["Stock","totB","pct_TS","Δ_ATM"]]
final_extra.columns=["Stock","Total_%_Change","%_Change_TS","Δ_ATM_DIFF"]

st.dataframe(
    final_extra.sort_values("Δ_ATM_DIFF",ascending=False)
    .style.format({
        "Total_%_Change":"{:.2f}",
        "%_Change_TS":"{:.2f}",
        "Δ_ATM_DIFF":"{:.0f}"
    }),
    use_container_width=True
)

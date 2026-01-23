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
# GITHUB CONFIG (SECRETS)
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
def push_file(local_path, repo_path, msg):
    url = f"{GITHUB_API}/repos/{KITE_REPO}/contents/{repo_path}"
    content = base64.b64encode(open(local_path,"rb").read()).decode()
    r = requests.get(url, headers=HEADERS)
    sha = r.json().get("sha") if r.status_code == 200 else None

    payload = {"message": msg, "content": content, "branch": GITHUB_BRANCH}
    if sha:
        payload["sha"] = sha

    requests.put(url, headers=HEADERS, json=payload)

# ==================================================
# LOAD OPTION CHAIN FILES
# ==================================================
def load_csvs():
    out = []
    for f in os.listdir(DATA_DIR):
        if f.startswith("option_chain_") and f.endswith(".csv"):
            ts = f.replace("option_chain_","").replace(".csv","")
            out.append((ts, os.path.join(DATA_DIR,f)))
    return sorted(out)

csv_files = load_csvs()
file_map = dict(csv_files)
timestamps = [x[0] for x in csv_files]

# ==================================================
# TIME HELPERS
# ==================================================
def ts_time(ts):
    hh, mm = map(int, ts.split("_")[-1].split("-")[:2])
    return time(hh, mm)

valid_ts = [
    ts for ts in timestamps
    if time(9,16) <= ts_time(ts) <= time(11,45)
]

def default_ts2(ts_list):
    for ts in ts_list:
        if ts_time(ts) >= time(9,16):
            return ts
    return ts_list[0]

# ==================================================
# USER INPUT
# ==================================================
c1, c2, c3, c4 = st.columns(4)
t1 = c1.selectbox("TS1", valid_ts, index=len(valid_ts)-1)
t2 = c2.selectbox("TS2", valid_ts, index=valid_ts.index(default_ts2(valid_ts)))
X = c3.number_input("X", 1, 10, 4)
Y = c4.number_input("Y", 4, 20, 6)

K = 4

# ==================================================
# PRICE CONTEXT
# ==================================================
df1 = pd.read_csv(file_map[t1])[["Stock","Stock_%_Change","Stock_LTP"]]
df2 = pd.read_csv(file_map[t2])[["Stock","Stock_LTP"]]
df1.columns = ["stock","tot%","ltp1"]
df2.columns = ["stock","ltp2"]

price_df = df1.merge(df2, on="stock", how="left")
price_df["Δ%"] = ((price_df["ltp1"] - price_df["ltp2"]) / price_df["ltp2"]) * 100
price_df = price_df.set_index("stock")[["tot%","Δ%"]]

# ==================================================
# ATM CALC
# ==================================================
def atm_calc(ts1, ts2):
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
    df["atm"] = np.nan

    for s, g in df.groupby("Stock"):
        g = g.sort_values("Strike").reset_index()
        for i in range(len(g)):
            lo, hi = max(0,i-X), min(len(g)-1,i+X)
            df.at[g.loc[i,"index"],"diff"] = (
                g.loc[lo:hi,"pe_x"].sum() - g.loc[lo:hi,"ce_x"].sum()
            )

        ltp = g["ltp0"].iloc[0]
        atm_i = (g["Strike"]-ltp).abs().idxmin()
        df.loc[g.loc[atm_i,"index"],"atm"] = (
            g.loc[max(0,atm_i-2):atm_i+2,"diff"].mean()
        )

    return df.groupby("Stock")["atm"].first()

# ==================================================
# CACHE CSV (STOCK_REF)
# ==================================================
ref_tag = ts_time(t2).strftime("%H%M")
cache_file = f"stock_ref_{ref_tag}.csv"
cache_path = os.path.join(CACHE_DIR, cache_file)

if os.path.exists(cache_path):
    stock_df = pd.read_csv(cache_path)
else:
    stock_df = pd.DataFrame(columns=["time","stock","atm"])

for ts in valid_ts:
    if ts_time(ts) < ts_time(t2) or ts_time(ts) > ts_time(t1):
        continue

    t_str = ts_time(ts).strftime("%H:%M")
    if (stock_df["time"] == t_str).any():
        continue

    series = atm_calc(ts, t2)
    for stk, v in series.items():
        stock_df.loc[len(stock_df)] = [t_str, stk, round(v,2)]

stock_df.to_csv(cache_path, index=False)
push_file(cache_path, f"{CACHE_DIR}/{cache_file}", f"update {cache_file}")

# ==================================================
# Σ ATM_DIFF TABLE
# ==================================================
sigma = (
    stock_df.groupby("time")["atm"]
    .sum()
    .reset_index()
    .rename(columns={"atm":"Σ_ATM"})
)
sigma["Σ_ATM"] /= 100

st.subheader("Σ ATM_DIFF Over Time")
st.dataframe(sigma, use_container_width=True)

# ==================================================
# PIVOT
# ==================================================
pivot = stock_df.pivot(index="stock", columns="time", values="atm").sort_index()
times = list(pivot.columns)

# ==================================================
# LIS / LDS
# ==================================================
def lis(a):
    d=[]
    for x in a:
        i=np.searchsorted(d,x)
        if i==len(d): d.append(x)
        else: d[i]=x
    return len(d)

def lds(a):
    return lis([-x for x in a])

# ==================================================
# COUNT GREEN / RED
# ==================================================
G,R={},{}
for s in pivot.index:
    gset,rset=set(),set()
    v=pivot.loc[s].values
    for i in range(len(v)-Y+1):
        w=v[i:i+Y]
        if np.isnan(w).any(): continue
        cols=times[i:i+Y]
        if lis(w)>=K: gset.update(cols)
        elif lds(w)>=K: rset.update(cols)
    G[s]=len(gset)
    R[s]=len(rset)

meta = price_df.copy()
meta["G#"]=pd.Series(G)
meta["R#"]=pd.Series(R)

final = meta.join(pivot)

# ==================================================
# HIGHLIGHT
# ==================================================
def highlight(df):
    sty = pd.DataFrame("", index=df.index, columns=df.columns)
    for s in pivot.index:
        v=pivot.loc[s].values
        for i in range(len(v)-Y+1):
            w=v[i:i+Y]
            if np.isnan(w).any(): continue
            cols=times[i:i+Y]
            if lis(w)>=K:
                sty.loc[s, cols]="background-color:#c6efce"
            elif lds(w)>=K:
                sty.loc[s, cols]="background-color:#ffc7ce"
    return sty

st.subheader("ATM Diff Pattern Table")
st.dataframe(final.style.apply(highlight, axis=None), use_container_width=True)

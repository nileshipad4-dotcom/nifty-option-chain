import streamlit as st
import pandas as pd
import numpy as np
import os
from streamlit_autorefresh import st_autorefresh
import yfinance as yf

st_autorefresh(interval=3600_000, key="auto_refresh")

# ==================================================
# CONFIG
# ==================================================
st.set_page_config(page_title="Index OI Weighted Table", layout="wide")
st.title("📊 Index OI Weighted Strike Tables")

DATA_DIR = "data_index"
INDEX_LIST = ["NIFTY", "BANKNIFTY", "MIDCPNIFTY"]

YAHOO_MAP = {
    "NIFTY": "^NSEI",
    "BANKNIFTY": "^NSEBANK",
    "MIDCPNIFTY": "NIFTY_MID_SELECT.NS"
}

# ==================================================
# LIVE SPOT + % CHANGE
# ==================================================
@st.cache_data(ttl=30)
def get_yahoo_data(symbol):
    try:
        data = yf.Ticker(symbol).history(period="2d", interval="1m")
        spot = float(data["Close"].iloc[-1])
        prev = float(data["Close"].iloc[-2])
        pct = ((spot - prev) / prev) * 100 if prev else 0
        return spot, pct
    except:
        return None, None

# ==================================================
# LOAD CSV FILES
# ==================================================
def load_csv_files():
    files = []
    for f in os.listdir(DATA_DIR):
        if f.startswith("index_OC_") and f.endswith(".csv"):
            ts = f.replace("index_OC_", "").replace(".csv", "")
            files.append((ts, os.path.join(DATA_DIR, f)))
    return sorted(files, reverse=True)

csv_files = load_csv_files()
timestamps_all = [ts for ts, _ in csv_files]
file_map = dict(csv_files)
filtered_ts = timestamps_all[:30]

# ==================================================
# USER CONTROLS
# ==================================================
c1, c2, c3, c4 = st.columns(4)
t1 = c1.selectbox("TS1 (Latest)", filtered_ts, 0)
t2 = c2.selectbox("TS2", filtered_ts, 1)
t3 = c3.selectbox("TS3", filtered_ts, 2)
X = c4.number_input("Strike Window X", 1, 20, 4)

# ==================================================
# LOAD DATA
# ==================================================
df1 = pd.read_csv(file_map[t1])
df2 = pd.read_csv(file_map[t2])
df3 = pd.read_csv(file_map[t3])

# ==================================================
# BUILD BASE TABLE
# ==================================================
dfs = []
for i, d in enumerate([df1, df2, df3]):
    dfs.append(
        d[[
            "Symbol","Strike",
            "CE_OI","PE_OI",
            "CE_Volume","PE_Volume",
            "CE_LTP","PE_LTP",
            "Max_Pain"
        ]].rename(columns={
            "CE_OI": f"ce_{i}",
            "PE_OI": f"pe_{i}",
            "CE_Volume": f"ce_vol_{i}",
            "PE_Volume": f"pe_vol_{i}",
            "CE_LTP": f"ce_ltp_{i}",
            "PE_LTP": f"pe_ltp_{i}",
            "Max_Pain": f"mp_{i}"
        })
    )

df = dfs[0].merge(dfs[1], on=["Symbol","Strike"]) \
           .merge(dfs[2], on=["Symbol","Strike"])

# ==================================================
# NUMERIC SAFETY
# ==================================================
for c in df.columns:
    if c != "Symbol":
        df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0)

# ==================================================
# CORE CALCULATIONS
# ==================================================
df["dCE_OI"] = df["ce_0"] - df["ce_1"]
df["dPE_OI"] = df["pe_0"] - df["pe_1"]

df["MP1"] = df["mp_0"] / 1000
df["MP2"] = df["mp_1"] / 1000

# ==================================================
# MAX PAIN USING ΔOI + TS1 LTP
# ==================================================
def compute_max_pain_delta_oi(g):
    g = g.sort_values("Strike").reset_index(drop=True)

    A = g["ce_ltp_0"]
    B = g["dCE_OI"]
    G = g["Strike"]
    M = g["pe_ltp_0"]
    L = g["dPE_OI"]

    mp = []
    for i in range(len(g)):
        val = (
            - (A[i:] * B[i:]).sum()
            + G.iloc[i] * B[:i].sum()
            - (G[:i] * B[:i]).sum()
            - (M[:i] * L[:i]).sum()
            + (G[i:] * L[i:]).sum()
            - G.iloc[i] * L[i:].sum()
        )
        mp.append(int(val / 10000))

    g["MP_DeltaOI"] = mp
    return g

df = pd.concat(
    [compute_max_pain_delta_oi(g) for _, g in df.groupby("Symbol")],
    ignore_index=True
)

# ==================================================
# FINAL TABLE
# ==================================================
table = df.rename(columns={
    "Symbol":"stk",
    "Strike":"str"
})[[
    "stk","str",
    "dCE_OI","dPE_OI",
    "MP1","MP2",
    "MP_DeltaOI"
]]

# ==================================================
# DISPLAY
# ==================================================
fmt = {
    "str":"{:.0f}",
    "dCE_OI":"{:.0f}",
    "dPE_OI":"{:.0f}",
    "MP1":"{:.0f}",
    "MP2":"{:.0f}",
    "MP_DeltaOI":"{:.0f}"
}

for idx in INDEX_LIST:
    idx_df = table[table["stk"] == idx]
    if idx_df.empty:
        continue

    spot, pct = get_yahoo_data(YAHOO_MAP[idx])
    if spot is None:
        continue

    st.markdown(f"## 📌 {idx}")
    st.markdown(f"### 💹 Spot: **{int(spot)}** | 📈 **{pct:.2f}%**")

    st.dataframe(
        idx_df.sort_values("str")
        .style.format(fmt),
        use_container_width=True
    )

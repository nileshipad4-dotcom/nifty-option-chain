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
        if data.empty:
            return None, None
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

csv_files_all = load_csv_files()

if len(csv_files_all) < 3:
    st.error("Need at least 3 index CSV files")
    st.stop()

from datetime import time

def is_market_time(ts):
    dt = pd.to_datetime(ts, format="%Y-%m-%d_%H-%M")
    return time(9, 0) <= dt.time() <= time(16, 0)

# ✅ FILTER FILES FIRST
csv_files = [
    (ts, path) for ts, path in csv_files_all
    if is_market_time(ts)
]

if len(csv_files) < 3:
    st.error("Not enough market-hour CSV files (09:00–16:00)")
    st.stop()

timestamps_all = [ts for ts, _ in csv_files]
file_map = dict(csv_files)
filtered_ts = timestamps_all[:30]


# ==================================================
# USER CONTROLS
# ==================================================
st.subheader("🕒 Timestamp & Window Settings")
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
# BUILD BASE TABLE (UNCHANGED)
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
# ORIGINAL CORE CALCULATIONS (UNCHANGED)
# ==================================================
df["MP1"] = df["mp_0"] / 1000
df["MP2"] = df["mp_1"] / 1000

df["d_ce"] = (df["ce_0"] - df["ce_1"]) / 1000
df["d_pe"] = (df["pe_0"] - df["pe_1"]) / 1000

df["d_ce_23"] = df["ce_1"] - df["ce_2"]
df["d_pe_23"] = df["pe_1"] - df["pe_2"]

df["ce_x"] = (df["d_ce"] * df["Strike"]) / 10000
df["pe_x"] = (df["d_pe"] * df["Strike"]) / 10000

df["ce_x_23"] = (df["d_ce_23"] * df["Strike"]) / 100000
df["pe_x_23"] = (df["d_pe_23"] * df["Strike"]) / 100000

# ==================================================
# SLIDING WINDOW (UNCHANGED)
# ==================================================
df["diff"] = np.nan
df["diff_23"] = np.nan

for sym, g in df.groupby("Symbol"):
    g = g.sort_values("Strike").reset_index()

    for i in range(len(g)):
        low = max(0, i - X)
        high = min(len(g) - 1, i + X)

        ce_sum = g.loc[low:high, "ce_x"].sum()
        pe_sum = g.loc[low:high, "pe_x"].sum()

        ce_sum_23 = g.loc[low:high, "ce_x_23"].sum()
        pe_sum_23 = g.loc[low:high, "pe_x_23"].sum()

        orig = g.loc[i, "index"]
        df.at[orig, "diff"] = pe_sum - ce_sum
        df.at[orig, "diff_23"] = (pe_sum_23 - ce_sum_23) / 10

# ==================================================
# DELTA COLUMNS (UNCHANGED)
# ==================================================
df["Δ CE Vol"] = (df["ce_vol_0"] - df["ce_vol_1"]) / 10000
df["Δ PE Vol"] = (df["pe_vol_0"] - df["pe_vol_1"]) / 10000
df["Δ MP 1"] = (df["mp_0"] - df["mp_1"]) / 100
df["Δ MP 2"] = (df["mp_1"] - df["mp_2"]) / 100

# ==================================================
# ✅ NEW: MAX PAIN USING ΔOI + TS1 LTP
# ==================================================
df["dCE_OI"] = df["ce_0"] - df["ce_1"]
df["dPE_OI"] = df["pe_0"] - df["pe_1"]

def compute_mp_delta_oi(g):
    g = g.sort_values("Strike").reset_index(drop=True)

    A = g["ce_ltp_0"]
    B = g["dCE_OI"]
    G = g["Strike"]
    M = g["pe_ltp_0"]
    L = g["dPE_OI"]

    mp = []
    for i in range(len(g)):
        val = (
            -(A[i:] * B[i:]).sum()
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
    [compute_mp_delta_oi(g) for _, g in df.groupby("Symbol")],
    ignore_index=True
)

# ==================================================
# FINAL TABLE (ONLY ADD COLUMN)
# ==================================================
table = df.rename(columns={"Symbol":"stk","Strike":"str"})[[
    "stk","str",
    "d_ce","d_pe",
    "ce_x","pe_x","diff","diff_23",
    "Δ CE Vol","Δ PE Vol",
    "MP1","MP2","Δ MP 1","Δ MP 2",
    "MP_DeltaOI"
]]

# ==================================================
# FIXED ±10 STRIKES
# ==================================================
def filter_near_spot_fixed(df, spot, n=10):
    g = df.sort_values("str").reset_index(drop=True)
    atm = (g["str"] - spot).abs().idxmin()
    return g.iloc[max(0, atm-n):min(len(g), atm+n+1)]

# ==================================================
# ATM BLUE HIGHLIGHT (UNCHANGED)
# ==================================================
def atm_blue_live(data, spot):
    styles = pd.DataFrame("", index=data.index, columns=data.columns)
    s = data["str"].values
    for i in range(len(s)-1):
        if s[i] <= spot <= s[i+1]:
            styles.loc[data.index[i:i+2]] = "background-color:#003366;color:white"
            break
    return styles

# ==================================================
# DISPLAY
# ==================================================
fmt = {c:"{:.0f}" for c in table.columns if c not in ["stk"]}

for idx in INDEX_LIST:
    idx_df = table[table["stk"] == idx]
    spot, pct = get_yahoo_data(YAHOO_MAP[idx])
    if spot is None or idx_df.empty:
        continue

    view = filter_near_spot_fixed(idx_df, spot)

    st.markdown(f"## 📌 {idx}")
    st.markdown(f"### 💹 Spot: **{int(spot)}** | 📈 **{pct:.2f}%**")

    st.dataframe(
        view.style
        .apply(atm_blue_live, spot=spot, axis=None)
        .format(fmt),
        use_container_width=True
    )

import streamlit as st
import pandas as pd
import numpy as np
import os
from datetime import time
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

csv_files = load_csv_files()

if len(csv_files) < 3:
    st.error("Need at least 3 index CSV files in data_index/")
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

X = c4.number_input("Strike Window X (For Calculations)", 1, 20, 4)

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
            "Symbol", "Strike",
            "CE_OI", "PE_OI",
            "CE_Volume", "PE_Volume",
            "Max_Pain"
        ]].rename(columns={
            "CE_OI": f"ce_{i}",
            "PE_OI": f"pe_{i}",
            "CE_Volume": f"ce_vol_{i}",
            "PE_Volume": f"pe_vol_{i}",
            "Max_Pain": f"mp_{i}"
        })
    )

df = dfs[0].merge(dfs[1], on=["Symbol", "Strike"]) \
           .merge(dfs[2], on=["Symbol", "Strike"])

# ==================================================
# NUMERIC SAFETY
# ==================================================
for c in df.columns:
    if c != "Symbol":
        df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0)

# ==================================================
# CORE CALCULATIONS
# ==================================================
df["d_ce"] = (df["ce_0"] - df["ce_1"]) / 100000
df["d_pe"] = (df["pe_0"] - df["pe_1"]) / 100000


df["d_ce_23"] = df["ce_1"] - df["ce_2"]
df["d_pe_23"] = df["pe_1"] - df["pe_2"]

df["ce_x"] = (df["d_ce"] * df["Strike"]) / 10000000
df["pe_x"] = (df["d_pe"] * df["Strike"]) / 10000000

df["ce_x_23"] = (df["d_ce_23"] * df["Strike"]) / 100000000
df["pe_x_23"] = (df["d_pe_23"] * df["Strike"]) / 100000000

# ==================================================
# SLIDING WINDOW (LIKE STOCK CODE)
# ==================================================
df["diff"] = np.nan
df["diff_23"] = np.nan

for idx, g in df.groupby("Symbol"):
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
        df.at[orig, "diff_23"] = pe_sum_23 - ce_sum_23

# ==================================================
# DELTA COLUMNS
# ==================================================
df["Δ CE Vol"] = (df["ce_vol_0"] - df["ce_vol_1"]) / 1000000
df["Δ PE Vol"] = (df["pe_vol_0"] - df["pe_vol_1"]) / 1000000

df["Δ MP 1"] = (df["mp_0"] - df["mp_1"]) / 10000
df["Δ MP 2"] = (df["mp_1"] - df["mp_2"]) / 10000

# ==================================================
# FINAL TABLE
# ==================================================
table = df.rename(columns={
    "Symbol": "stk",
    "Strike": "str"
})[[
    "stk","str",
    "d_ce","d_pe",
    "ce_x","pe_x","diff",
    "diff_23",
    "Δ CE Vol","Δ PE Vol",
    "Δ MP 1","Δ MP 2"
]]

# ==================================================
# FIXED DISPLAY: 10 ABOVE + 10 BELOW ATM
# ==================================================
def filter_near_spot_fixed(df, live_spot, n=10):
    g = df.sort_values("str").reset_index(drop=True)
    atm_idx = (g["str"] - live_spot).abs().idxmin()

    start = max(0, atm_idx - n)
    end = min(len(g) - 1, atm_idx + n)

    return g.iloc[start:end + 1]

# ==================================================
# ATM BLUE HIGHLIGHT
# ==================================================
def atm_blue_live(data, live_spot):
    styles = pd.DataFrame("", index=data.index, columns=data.columns)
    strikes = data["str"].values

    for i in range(len(strikes) - 1):
        if strikes[i] <= live_spot <= strikes[i + 1]:
            styles.loc[data.index[i]] = "background-color:#003366;color:white"
            styles.loc[data.index[i + 1]] = "background-color:#003366;color:white"
            break

    return styles

# ==================================================
# FORMAT
# ==================================================
fmt = {
    "str": "{:.0f}",
    "d_ce": "{:.0f}",
    "d_pe": "{:.0f}",
    "ce_x": "{:.0f}",
    "pe_x": "{:.0f}",
    "diff": "{:.0f}",
    "diff_23": "{:.0f}",
    "Δ CE Vol": "{:.0f}",
    "Δ PE Vol": "{:.0f}",
    "Δ MP 1": "{:.0f}",
    "Δ MP 2": "{:.0f}"
}

# ==================================================
# DISPLAY
# ==================================================
for idx in INDEX_LIST:

    idx_df = table[table["stk"] == idx]

    if idx_df.empty:
        continue

    live_spot, live_pct = get_yahoo_data(YAHOO_MAP[idx])

    if live_spot is None:
        st.error(f"{idx} live price not available")
        continue

    display_df = filter_near_spot_fixed(idx_df, live_spot, n=10)

    st.markdown(f"## 📌 {idx}")
    st.markdown(
        f"### 💹 Spot: **{int(live_spot)}** &nbsp;&nbsp; "
        f"📈 % Change: **{live_pct:.2f}%**"
    )

    st.dataframe(
        display_df
        .style
        .apply(atm_blue_live, live_spot=live_spot, axis=None)
        .format(fmt, na_rep=""),
        use_container_width=True
    )

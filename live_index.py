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
# LOAD FILES
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
file_map = {ts: path for ts, path in csv_files}

# ==================================================
# TIME FILTER
# ==================================================
def extract_time(ts):
    try:
        hh, mm = map(int, ts.split("_")[-1].split("-")[:2])
        return time(hh, mm)
    except:
        return None

filtered_ts = [
    ts for ts in timestamps_all
    if extract_time(ts) and time(8, 0) <= extract_time(ts) <= time(16, 0)
]

# ==================================================
# USER CONTROLS
# ==================================================
st.subheader("🕒 Timestamp & Window Settings")

c1, c2, c3, c4 = st.columns(4)

t1 = c1.selectbox("TS1 (Latest)", filtered_ts, 0)
t2 = c2.selectbox("TS2", filtered_ts, 1)
t3 = c3.selectbox("TS3", filtered_ts, 2)

X = c4.number_input("Strike Window (±)", 5, 20, 10)

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
            "Max Pain"
        ]].rename(columns={
            "CE_OI": f"ce_{i}",
            "PE_OI": f"pe_{i}",
            "CE_Volume": f"ce_vol_{i}",
            "PE_Volume": f"pe_vol_{i}",
            "Max Pain": f"mp_{i}"
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
# CORE CALCULATIONS (SAME AS STOCK LOGIC)
# ==================================================
df["d_ce"] = df["ce_0"] - df["ce_1"]
df["d_pe"] = df["pe_0"] - df["pe_1"]

df["ce_x"] = (df["d_ce"] * df["Strike"]) / 10000
df["pe_x"] = (df["d_pe"] * df["Strike"]) / 10000

# ==================================================
# SLIDING WINDOW (KEY PART)
# ==================================================
df["sum_ce"] = np.nan
df["sum_pe"] = np.nan
df["diff"] = np.nan
df["atm_diff"] = np.nan

for stk, g in df.groupby("Symbol"):
    g = g.sort_values("Strike").reset_index()

    for i in range(len(g)):
        low = max(0, i - X)
        high = min(len(g) - 1, i + X)

        ce_sum = g.loc[low:high, "ce_x"].sum()
        pe_sum = g.loc[low:high, "pe_x"].sum()

        orig_idx = g.loc[i, "index"]
        df.at[orig_idx, "sum_ce"] = ce_sum
        df.at[orig_idx, "sum_pe"] = pe_sum
        df.at[orig_idx, "diff"] = pe_sum - ce_sum

    # ATM DIFF (±2 strikes)
    strikes = g["Strike"].values
    live_spot, _ = get_yahoo_data(YAHOO_MAP.get(stk, ""))
    if live_spot:
        atm_idx = (strikes - live_spot).abs().argmin()

        low = max(0, atm_idx - 2)
        high = min(len(g) - 1, atm_idx + 2)

        atm_avg = df.loc[g.loc[low:high, "index"], "diff"].mean()
        df.loc[g["index"], "atm_diff"] = atm_avg

# ==================================================
# FINAL TABLE
# ==================================================
table = df.rename(columns={
    "Symbol": "stk",
    "Strike": "str"
})[[
    "stk","str",
    "d_ce","d_pe",
    "ce_x","pe_x",
    "sum_ce","sum_pe",
    "diff","atm_diff"
]]

# ==================================================
# FILTER NEAR ATM
# ==================================================
def filter_near_spot(df, live_spot, n=10):
    g = df.sort_values("str").reset_index(drop=True)
    atm_idx = (g["str"] - live_spot).abs().idxmin()
    return g.iloc[max(0, atm_idx - n): atm_idx + n + 1]

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
    "sum_ce": "{:.0f}",
    "sum_pe": "{:.0f}",
    "diff": "{:.0f}",
    "atm_diff": "{:.0f}"
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
        continue

    display_df = filter_near_spot(idx_df, live_spot, n=10)

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

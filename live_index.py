import streamlit as st
import pandas as pd
import numpy as np
import os
from datetime import time
from streamlit_autorefresh import st_autorefresh

st_autorefresh(interval=3600_000, key="auto_refresh")

# ==================================================
# CONFIG
# ==================================================
st.set_page_config(page_title="Index OI Weighted Table", layout="wide")
st.title("📊 Index OI Weighted Strike Tables")

DATA_DIR = "data_index"
INDEX_LIST = ["NIFTY", "BANKNIFTY", "MIDCPNIFTY"]

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
    st.error("Need at least 3 index CSV files.")
    st.stop()

timestamps_all = [ts for ts, _ in csv_files]
file_map = dict(csv_files)

# ==================================================
# TIME FILTER (08:00–16:00)
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

t1 = c1.selectbox("TS1", filtered_ts, 0)
t2 = c2.selectbox("TS2", filtered_ts, 1)
t3 = c3.selectbox("TS3", filtered_ts, 2)

X = c4.number_input("Strike Window X", min_value=1, max_value=10, value=4, step=1)

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
        d[["Symbol", "Strike", "Spot", "%Change", "CE_OI", "PE_OI"]]
        .rename(columns={
            "Spot": f"ltp_{i}",
            "%Change": f"tot_ch_{i}",
            "CE_OI": f"ce_{i}",
            "PE_OI": f"pe_{i}"
        })
    )

df = dfs[0].merge(dfs[1], on=["Symbol", "Strike"]) \
            .merge(dfs[2], on=["Symbol", "Strike"])

# ==================================================
# NUMERIC SAFETY
# ==================================================
for c in df.columns:
    if any(x in c for x in ["ltp", "ce", "pe", "Strike", "tot_ch"]):
        df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0)

# ==================================================
# CORE CALCULATIONS
# ==================================================
df["d_ce"] = df["ce_0"] - df["ce_1"]
df["d_pe"] = df["pe_0"] - df["pe_1"]
df["d_ce_23"] = df["ce_1"] - df["ce_2"]
df["d_pe_23"] = df["pe_1"] - df["pe_2"]
df["total_ch"] = df["tot_ch_0"]

df["ce_x"] = (df["d_ce"] * df["Strike"]) / 10000
df["pe_x"] = (df["d_pe"] * df["Strike"]) / 10000
df["diff"] = df["pe_x"] - df["ce_x"]

# ==================================================
# SLIDING WINDOW
# ==================================================
df["sum_ce"] = np.nan
df["sum_pe"] = np.nan

for idx, g in df.groupby("Symbol"):
    g = g.sort_values("Strike").reset_index()

    for i in range(len(g)):
        low = max(0, i - X)
        high = min(len(g) - 1, i + X)

        ce_sum = g.loc[low:high, "ce_x"].sum()
        pe_sum = g.loc[low:high, "pe_x"].sum()

        orig_idx = g.loc[i, "index"]
        df.at[orig_idx, "sum_ce"] = ce_sum
        df.at[orig_idx, "sum_pe"] = pe_sum

# ==================================================
# FINAL TABLE
# ==================================================
table = df.rename(columns={
    "Symbol": "stk",
    "Strike": "str",
    "ltp_0": "ltp",
    "tot_ch_0": "ch"
})[[
    "stk","str","ltp","ch","d_ce","d_pe","ce_x","pe_x","diff"
]]

# ==================================================
# FILTER NEAR ATM
# ==================================================
def filter_near_ltp(df, n=5):
    blocks = []
    for stk, g in df.groupby("stk"):
        g = g.sort_values("str").reset_index(drop=True)
        ltp = g["ltp"].iloc[0]
        atm_idx = (g["str"] - ltp).abs().idxmin()

        start = max(0, atm_idx - n)
        end = min(len(g) - 1, atm_idx + n)

        blocks.append(g.iloc[start:end + 1])

    return pd.concat(blocks, ignore_index=True)

display_df = filter_near_ltp(table, n=5)

# ==================================================
# ATM BLUE HIGHLIGHT
# ==================================================
def atm_blue(data):
    styles = pd.DataFrame("", index=data.index, columns=data.columns)

    for stk in data["stk"].unique():
        sdf = data[data["stk"] == stk].sort_values("str")
        ltp = sdf["ltp"].iloc[0]
        strikes = sdf["str"].values

        for i in range(len(strikes) - 1):
            if strikes[i] <= ltp <= strikes[i + 1]:
                styles.loc[sdf.index[i]] = "background-color:#003366;color:white"
                styles.loc[sdf.index[i + 1]] = "background-color:#003366;color:white"
                break

    return styles

# ==================================================
# FORMAT
# ==================================================
fmt = {
    "str": "{:.0f}",
    "ltp": "{:.2f}",
    "ch": "{:.2f}",
    "d_ce": "{:.0f}",
    "d_pe": "{:.0f}",
    "ce_x": "{:.0f}",
    "pe_x": "{:.0f}",
    "diff": "{:.0f}"
}

# ==================================================
# DISPLAY 3 INDEX TABLES
# ==================================================
for idx in INDEX_LIST:
    idx_df = display_df[display_df["stk"] == idx]

    if idx_df.empty:
        continue

    spot = idx_df["ltp"].iloc[0]

    st.markdown(f"## 📌 {idx}")
    st.markdown(f"### 💹 Spot Price: **{spot:.2f}**")

    st.dataframe(
        idx_df
        .style
        .apply(atm_blue, axis=None)
        .format(fmt, na_rep=""),
        use_container_width=True
    )

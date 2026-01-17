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
st.set_page_config(page_title="OI Weighted Table", layout="wide")
st.title("📊 OI Weighted Strike Table")

DATA_DIR = "data"

# ==================================================
# LOAD FILES
# ==================================================
def load_csv_files():
    files = []
    for f in os.listdir(DATA_DIR):
        if f.startswith("option_chain_") and f.endswith(".csv"):
            ts = f.replace("option_chain_", "").replace(".csv", "")
            files.append((ts, os.path.join(DATA_DIR, f)))
    return sorted(files, reverse=True)

csv_files = load_csv_files()

if len(csv_files) < 3:
    st.error("Need at least 3 CSV files.")
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

X = c4.number_input(
    "Strike Window X",
    min_value=1,
    max_value=10,
    value=4,
    step=1
)

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
        d[["Stock", "Strike", "Stock_LTP", "Stock_%_Change", "CE_OI", "PE_OI"]]
        .rename(columns={
            "Stock_LTP": f"ltp_{i}",
            "Stock_%_Change": f"tot_ch_{i}",
            "CE_OI": f"ce_{i}",
            "PE_OI": f"pe_{i}"
        })

    )

df = dfs[0].merge(dfs[1], on=["Stock", "Strike"]) \
            .merge(dfs[2], on=["Stock", "Strike"])

# ==================================================
# NUMERIC SAFETY
# ==================================================
for c in df.columns:
    if any(x in c for x in ["ltp", "ce", "pe", "Strike"]):
        df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0)

# ==================================================
# CORE CALCULATIONS
# ==================================================
df["d_ce"] = df["ce_0"] - df["ce_1"]
df["d_pe"] = df["pe_0"] - df["pe_1"]
df["d_ce_23"] = df["ce_1"] - df["ce_2"]
df["d_pe_23"] = df["pe_1"] - df["pe_2"]
df["total_ch"] = df["tot_ch_0"]


df["ce_x_23"] = (df["d_ce_23"] * df["Strike"]) / 10000
df["pe_x_23"] = (df["d_pe_23"] * df["Strike"]) / 10000

df["diff_23"] = df["pe_x_23"] - df["ce_x_23"]


df["ch"] = ((df["ltp_0"] - df["ltp_1"]) / df["ltp_1"]) * 100

df["ce_x"] = (df["d_ce"] * df["Strike"]) / 10000
df["pe_x"] = (df["d_pe"] * df["Strike"]) / 10000

# ==================================================
# SLIDING WINDOW SUM (STRIKE-BASED)
# ==================================================
df["sum_ce"] = np.nan
df["sum_pe"] = np.nan
df["diff"] = np.nan
df["atm_diff"] = np.nan

for stk, g in df.groupby("Stock"):
    g = g.sort_values("Strike").reset_index()

    for i in range(len(g)):
        low = max(0, i - X)
        high = min(len(g) - 1, i + X)

        ce_sum = g.loc[low:high, "ce_x"].sum()
        pe_sum = g.loc[low:high, "pe_x"].sum()
        diff_val = pe_sum - ce_sum

        df.loc[g.loc[i, "index"], "sum_ce"] = ce_sum
        df.loc[g.loc[i, "index"], "sum_pe"] = pe_sum
        df.loc[g.loc[i, "index"], "diff"] = diff_val

    # ATM +-2 strike diff (repeated)
    ltp = g["ltp_0"].iloc[0]
    
    atm_idx = (g["Strike"] - ltp).abs().values.argmin()
    
    low = max(0, atm_idx - 2)
    high = min(len(g) - 1, atm_idx + 2)
    
    atm_avg = g.loc[low:high, "diff"].mean()
    
    df.loc[g["index"], "atm_diff"] = atm_avg

# ==================================================
# FINAL TABLE (ALL STRIKES)
# ==================================================
table = df[[
    "Stock",
    "Strike",
    "ltp_0",
    "ch",
    "total_ch",
    "d_ce",
    "d_pe",
    "ce_x",
    "pe_x",
    "diff",
    "diff_23",
    "atm_diff"
]].rename(columns={
    "Stock": "stk",
    "Strike": "str",
    "ltp_0": "ltp"
})


# ==================================================
# DISPLAY FILTER: 5 ABOVE + 5 BELOW LTP
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

    for stk in data["stk"].dropna().unique():
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
    "ce_x": "{:.2f}",
    "pe_x": "{:.2f}",
    "sum_ce": "{:.2f}",
    "sum_pe": "{:.2f}",
    "diff": "{:.2f}",
    "atm_diff": "{:.2f}",
    "total_ch": "{:.2f}",
    "diff_23": "{:.0f}"

}

# ==================================================
# DISPLAY
# ==================================================
st.dataframe(
    display_df
    .style
    .apply(atm_blue, axis=None)
    .format(fmt, na_rep=""),
    use_container_width=True
)

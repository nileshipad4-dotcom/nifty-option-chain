import streamlit as st
import pandas as pd
import numpy as np
import os
from datetime import time
from streamlit_autorefresh import st_autorefresh

st_autorefresh(interval=360_000, key="auto_refresh")

# =====================================
# STREAMLIT CONFIG
# =====================================
st.set_page_config(page_title="Max Pain Comparison", layout="wide")
st.title("📊 FnO STOCKS")

DATA_DIR = "data"

# =====================================
# LOAD CSV FILES
# =====================================
def load_csv_files():
    files = []
    for f in os.listdir(DATA_DIR):
        if f.startswith("option_chain_") and f.endswith(".csv"):
            ts = f.replace("option_chain_", "").replace(".csv", "")
            files.append((ts, os.path.join(DATA_DIR, f)))
    return sorted(files, reverse=True)

csv_files = load_csv_files()
if len(csv_files) < 6:
    st.error("Need at least 6 CSV files.")
    st.stop()

timestamps_all = [ts for ts, _ in csv_files]
file_map = dict(csv_files)

# =====================================
# TIME FILTER
# =====================================
def extract_time(ts):
    try:
        hh, mm = map(int, ts.split("_")[-1].split("-")[:2])
        return time(hh, mm)
    except:
        return None

filtered_ts = [ts for ts in timestamps_all if extract_time(ts)]

# =====================================
# TIMESTAMP SELECTORS
# =====================================
cols = st.columns(6)
t1 = cols[0].selectbox("Timestamp 1", filtered_ts, 0)
t2 = cols[1].selectbox("Timestamp 2", filtered_ts, 1)
t3 = cols[2].selectbox("Timestamp 3", filtered_ts, 2)
t4 = cols[3].selectbox("Timestamp 4", filtered_ts, 3)
t5 = cols[4].selectbox("Timestamp 5", filtered_ts, 4)
t6 = cols[5].selectbox("Timestamp 6", timestamps_all, 0)

t = [t1, t2, t3, t4, t5, t6]

# =====================================
# LOAD DATA
# =====================================
dfs = []
for i, ts in enumerate(t):
    d = pd.read_csv(file_map[ts])
    d = d.rename(columns={
        "Max_Pain": f"MP_{i}",
        "Stock_LTP": f"LTP_{i}",
        "CE_OI": f"CE_OI_{i}",
        "PE_OI": f"PE_OI_{i}",
        "CE_Volume": f"CE_VOL_{i}",
        "PE_Volume": f"PE_VOL_{i}",
    })
    dfs.append(d)

# =====================================
# EXTRA FROM TIMESTAMP 1
# =====================================
raw_t1 = pd.read_csv(file_map[t1])
dfs[0]["Stock_%_Change"] = raw_t1["Stock_%_Change"]
dfs[0]["Stock_High"] = raw_t1["Stock_High"]
dfs[0]["Stock_Low"] = raw_t1["Stock_Low"]

# =====================================
# MERGE
# =====================================
df = dfs[0]
for i in range(1, 6):
    df = df.merge(dfs[i], on=["Stock", "Strike"])

# =====================================
# NUMERIC COERCION (CRITICAL FIX)
# =====================================
for c in df.columns:
    if any(x in c for x in ["MP_", "LTP_", "OI_", "VOL_"]):
        df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0)

# =====================================
# MAX PAIN DELTAS
# =====================================
df["Δ MP TS1"] = df["MP_0"] - df["MP_1"]
df["Δ MP TS2"] = df["MP_1"] - df["MP_2"]
df["Δ MP TS3"] = df["MP_2"] - df["MP_3"]

# ✅ DELTA–DELTA MP (YOUR EXACT DEFINITION)
df["ΔΔ MP (TS2-TS3)"] = df["Δ MP TS2"] - df["Δ MP TS3"]

# =====================================
# OI / VOLUME DELTAS (TS2 - TS3)
# =====================================
df["Δ CE OI (TS2-TS3)"] = df["CE_OI_1"] - df["CE_OI_2"]
df["Δ PE OI (TS2-TS3)"] = df["PE_OI_1"] - df["PE_OI_2"]
df["Δ CE Vol (TS2-TS3)"] = df["CE_VOL_1"] - df["CE_VOL_2"]
df["Δ PE Vol (TS2-TS3)"] = df["PE_VOL_1"] - df["PE_VOL_2"]

# =====================================
# CLEAN & FINAL COLUMNS
# =====================================
df["Stock_LTP"] = df["LTP_0"]

EXCLUDE = {"NIFTY", "BANKNIFTY", "FINNIFTY", "MIDCPNIFTY"}
df = df[~df["Stock"].isin(EXCLUDE)]

df = df[
    [
        "Stock", "Strike", "Stock_High", "Stock_Low",
        "Δ MP TS1", "Δ MP TS2", "Δ MP TS3", "ΔΔ MP (TS2-TS3)",
        "Stock_LTP", "Stock_%_Change",
        "Δ CE OI (TS2-TS3)", "Δ PE OI (TS2-TS3)",
        "Δ CE Vol (TS2-TS3)", "Δ PE Vol (TS2-TS3)"
    ]
]

# =====================================
# STRIKE FILTER (±6)
# =====================================
def filter_strikes(df, n=6):
    out = []
    for _, g in df.groupby("Stock"):
        g = g.sort_values("Strike")
        ltp = g["Stock_LTP"].iloc[0]
        idx = (g["Strike"] - ltp).abs().idxmin()
        pos = g.index.get_loc(idx)
        out.append(g.iloc[max(0, pos-n):pos+n+1])
        out.append(pd.DataFrame([{c: np.nan for c in df.columns}]))
    return pd.concat(out[:-1])

display_df = filter_strikes(df)

# =====================================
# HIGHLIGHTING (RESTORED)
# =====================================
def highlight(data):
    styles = pd.DataFrame("", index=data.index, columns=data.columns)
    for stock in data["Stock"].dropna().unique():
        sdf = data[data["Stock"] == stock]

        # ATM
        atm = sdf["Strike"].sub(sdf["Stock_LTP"].iloc[0]).abs().idxmin()
        styles.loc[atm] = "background-color:#003366;color:white"

        # Max |Δ MP TS1|
        styles.loc[sdf["Δ MP TS1"].abs().idxmax()] = \
            "background-color:#8B0000;color:white"
    return styles

# =====================================
# DISPLAY
# =====================================
st.dataframe(
    display_df.style
    .apply(highlight, axis=None)
    .format("{:.0f}", na_rep=""),
    use_container_width=True
)

# =====================================
# DOWNLOAD
# =====================================
st.download_button(
    "⬇️ Download CSV",
    df.to_csv(index=False),
    "max_pain_final_correct.csv",
    "text/csv",
)

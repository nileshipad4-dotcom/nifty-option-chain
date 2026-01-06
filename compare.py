import streamlit as st
import pandas as pd
import numpy as np
import os
from datetime import time
from streamlit_autorefresh import st_autorefresh

st_autorefresh(interval=360_000, key="auto_refresh")

# ==================================================
# STREAMLIT CONFIG
# ==================================================
st.set_page_config(page_title="FnO MP Delta Dashboard", layout="wide")
st.title("📊 FnO STOCKS – Max Pain Delta View")

DATA_DIR = "data"

# ==================================================
# LOAD CSV FILES
# ==================================================
def load_csv_files():
    files = []
    for f in os.listdir(DATA_DIR):
        if f.startswith("option_chain_") and f.endswith(".csv"):
            ts = f.replace("option_chain_", "").replace(".csv", "")
            files.append((ts, os.path.join(DATA_DIR, f)))
    return sorted(files, reverse=True)

csv_files = load_csv_files()
if len(csv_files) < 4:
    st.error("Need at least 4 CSV files.")
    st.stop()

timestamps_all = [ts for ts, _ in csv_files]
file_map = dict(csv_files)

# ==================================================
# TIME FILTER (08:00 to 16:30)
# ==================================================
def extract_time(ts):
    try:
        hh, mm = map(int, ts.split("_")[-1].split("-")[:2])
        return time(hh, mm)
    except:
        return None

START_TIME = time(8, 0)
END_TIME = time(16, 30)

filtered_ts = [
    ts for ts in timestamps_all
    if extract_time(ts)
    and START_TIME <= extract_time(ts) <= END_TIME
]

# ==================================================
# TIMESTAMP SELECTORS (4)
# ==================================================
cols = st.columns(4)
t1 = cols[0].selectbox("Timestamp 1", filtered_ts, 0)
t2 = cols[1].selectbox("Timestamp 2", filtered_ts, 1)
t3 = cols[2].selectbox("Timestamp 3", filtered_ts, 2)
t4 = cols[3].selectbox("Timestamp 4", filtered_ts, 3)

t = [t1, t2, t3, t4]

# ==================================================
# LOAD TIME-VARYING DATA
# ==================================================
dfs = []
for i, ts in enumerate(t):
    d = pd.read_csv(file_map[ts])
    d = d[
        ["Stock", "Strike", "Max_Pain", "Stock_LTP",
         "CE_OI", "PE_OI", "CE_Volume", "PE_Volume"]
    ].rename(columns={
        "Max_Pain": f"MP_{i}",
        "Stock_LTP": f"LTP_{i}",
        "CE_OI": f"CE_OI_{i}",
        "PE_OI": f"PE_OI_{i}",
        "CE_Volume": f"CE_VOL_{i}",
        "PE_Volume": f"PE_VOL_{i}",
    })
    dfs.append(d)

# ==================================================
# MERGE
# ==================================================
df = dfs[0]
for i in range(1, 4):
    df = df.merge(dfs[i], on=["Stock", "Strike"], how="inner")

# ==================================================
# ATTACH TS1 STATIC DATA
# ==================================================
raw_t1 = pd.read_csv(file_map[t1])[
    ["Stock", "Strike", "Stock_%_Change", "Stock_High", "Stock_Low"]
]
df = df.merge(raw_t1, on=["Stock", "Strike"], how="left")

# ==================================================
# NUMERIC COERCION
# ==================================================
for c in df.columns:
    if any(x in c for x in ["MP_", "LTP_", "OI_", "VOL_"]):
        df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0)

# ==================================================
# MAX PAIN DELTAS
# ==================================================
df["Δ MP TS1-TS2"] = df["MP_0"] - df["MP_1"]
df["Δ MP TS2-TS3"] = df["MP_1"] - df["MP_2"]
df["Δ MP TS3-TS4"] = df["MP_2"] - df["MP_3"]

df["ΔΔ MP (TS1-TS2 vs TS2-TS3)"] = (
    df["Δ MP TS1-TS2"] - df["Δ MP TS2-TS3"]
)

# ==================================================
# OI / VOLUME DELTAS (TS1–TS2)
# ==================================================
df["Δ CE OI TS1-TS2"] = df["CE_OI_0"] - df["CE_OI_1"]
df["Δ PE OI TS1-TS2"] = df["PE_OI_0"] - df["PE_OI_1"]
df["Δ CE Vol TS1-TS2"] = df["CE_VOL_0"] - df["CE_VOL_1"]
df["Δ PE Vol TS1-TS2"] = df["PE_VOL_0"] - df["PE_VOL_1"]

# ==================================================
# STOCK % CHANGE BETWEEN TIMESTAMPS
# ==================================================
df["% Stock Ch TS1-TS2"] = ((df["LTP_0"] - df["LTP_1"]) / df["LTP_1"]) * 100
df["% Stock Ch TS2-TS3"] = ((df["LTP_1"] - df["LTP_2"]) / df["LTP_2"]) * 100

df["Stock_LTP"] = df["LTP_0"]

# ==================================================
# FINAL COLUMNS
# ==================================================
df = df[
    [
        "Stock", "Strike",
        "Δ MP TS1-TS2", "Δ MP TS2-TS3", "Δ MP TS3-TS4",
        "ΔΔ MP (TS1-TS2 vs TS2-TS3)",
        "Δ CE OI TS1-TS2", "Δ PE OI TS1-TS2",
        "Δ CE Vol TS1-TS2", "Δ PE Vol TS1-TS2",
        "Stock_LTP", "Stock_%_Change",
        "% Stock Ch TS1-TS2", "% Stock Ch TS2-TS3",
        "Stock_High", "Stock_Low",
    ]
]

# ==================================================
# STRIKE FILTER ±6
# ==================================================
def filter_strikes(df, n=6):
    blocks = []
    for _, g in df.groupby("Stock"):
        g = g.sort_values("Strike").reset_index(drop=True)
        ltp = g["Stock_LTP"].iloc[0]
        atm_pos = (g["Strike"] - ltp).abs().idxmin()
        blocks.append(g.iloc[max(0, atm_pos-n):atm_pos+n+1])
        blocks.append(pd.DataFrame([{c: np.nan for c in g.columns}]))
    return pd.concat(blocks[:-1], ignore_index=True)

display_df = filter_strikes(df)

# ==================================================
# HIGHLIGHTING (ATM BAND + MAX ΔMP)
# ==================================================
def highlight(data):
    styles = pd.DataFrame("", index=data.index, columns=data.columns)

    for stock in data["Stock"].dropna().unique():
        sdf = data[(data["Stock"] == stock) & data["Strike"].notna()]
        if sdf.empty:
            continue

        ltp = sdf["Stock_LTP"].iloc[0]
        strikes = sdf["Strike"].values

        for i in range(len(strikes) - 1):
            if strikes[i] <= ltp <= strikes[i + 1]:
                styles.iloc[sdf.index[i]] = "background-color:#003366;color:white"
                styles.iloc[sdf.index[i + 1]] = "background-color:#003366;color:white"
                break

        mp_pos = sdf["Δ MP TS1-TS2"].abs().idxmax()
        styles.loc[mp_pos] = "background-color:#8B0000;color:white"

    return styles

# ==================================================
# FORMATTERS
# ==================================================
num_cols = display_df.select_dtypes(include="number").columns
formatters = {c: "{:.0f}" for c in num_cols}

formatters["Stock_LTP"] = "{:.2f}"
formatters["Stock_%_Change"] = "{:.2f}"
formatters["% Stock Ch TS1-TS2"] = "{:.2f}"
formatters["% Stock Ch TS2-TS3"] = "{:.2f}"

# ==================================================
# DISPLAY
# ==================================================
st.dataframe(
    display_df.style
    .apply(highlight, axis=None)
    .format(formatters, na_rep=""),
    use_container_width=True
)

# ==================================================
# DOWNLOAD
# ==================================================
st.download_button(
    "⬇️ Download CSV",
    df.to_csv(index=False),
    "mp_delta_final_clean.csv",
    "text/csv",
)

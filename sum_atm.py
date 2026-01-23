import streamlit as st
import pandas as pd
import numpy as np
import os
from datetime import time, datetime

# ==================================================
# CONFIG
# ==================================================
st.set_page_config(page_title="ATM Diff", layout="centered")
st.title("📊 ATM Diff (Stock-wise)")

DATA_DIR = "data"
CACHE_DIR = "data_atm"
os.makedirs(CACHE_DIR, exist_ok=True)

# ==================================================
# LOAD CSV FILES
# ==================================================
def load_csv_files():
    files = []
    for f in os.listdir(DATA_DIR):
        if f.startswith("option_chain_") and f.endswith(".csv"):
            ts = f.replace("option_chain_", "").replace(".csv", "")
            files.append((ts, os.path.join(DATA_DIR, f)))
    return sorted(files)

csv_files = load_csv_files()
timestamps_all = [ts for ts, _ in csv_files]
file_map = dict(csv_files)

# ==================================================
# TIME PARSER
# ==================================================
def extract_time(ts):
    hh, mm = map(int, ts.split("_")[-1].split("-")[:2])
    return time(hh, mm)

# ==================================================
# FILTERED TIMESTAMPS
# ==================================================
filtered_ts = [
    ts for ts in timestamps_all
    if time(8,0) <= extract_time(ts) <= time(16,0)
]

# ==================================================
# DEFAULT TS2 → FIRST AFTER 09:16
# ==================================================
def first_after_916(ts_list):
    for ts in ts_list:
        if extract_time(ts) >= time(9,16):
            return ts
    return ts_list[0]

default_ts2 = first_after_916(filtered_ts)

# ==================================================
# USER INPUT
# ==================================================
c1, c2, c3 = st.columns(3)

t1 = c1.selectbox("Timestamp 1 (Current)", filtered_ts, index=len(filtered_ts)-1)
t2 = c2.selectbox("Timestamp 2 (Reference)", filtered_ts, index=filtered_ts.index(default_ts2))
X = c3.number_input("Strike Window X", 1, 10, 4)

# ==================================================
# ATM CALC FUNCTION (UNCHANGED LOGIC)
# ==================================================
def compute_sigma_atm(ts1, ts2, X):
    df1 = pd.read_csv(file_map[ts1])
    df2 = pd.read_csv(file_map[ts2])

    df1 = df1[["Stock","Strike","Stock_LTP","CE_OI","PE_OI"]]
    df2 = df2[["Stock","Strike","Stock_LTP","CE_OI","PE_OI"]]

    df1.columns = ["Stock","Strike","ltp_0","ce_0","pe_0"]
    df2.columns = ["Stock","Strike","ltp_1","ce_1","pe_1"]

    df = df1.merge(df2, on=["Stock","Strike"])

    for c in df.columns:
        if c != "Stock":
            df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0)

    df["ce_x"] = (df["ce_0"] - df["ce_1"]) * df["Strike"] / 10000
    df["pe_x"] = (df["pe_0"] - df["pe_1"]) * df["Strike"] / 10000

    df["diff"] = np.nan
    df["atm_diff"] = np.nan

    for stk, g in df.groupby("Stock"):
        g = g.sort_values("Strike").reset_index()

        for i in range(len(g)):
            low = max(0, i - X)
            high = min(len(g)-1, i + X)
            df.at[g.loc[i,"index"], "diff"] = (
                g.loc[low:high,"pe_x"].sum() - g.loc[low:high,"ce_x"].sum()
            )

        g["diff"] = df.loc[g["index"], "diff"].values

        ltp = g["ltp_0"].iloc[0]
        atm_idx = (g["Strike"] - ltp).abs().values.argmin()
        atm_avg = g.loc[max(0,atm_idx-2):atm_idx+2, "diff"].mean()
        df.loc[g["index"], "atm_diff"] = atm_avg

    return df.groupby("Stock")["atm_diff"].first().sum() / 100

# ==================================================
# CURRENT SNAPSHOT
# ==================================================
current_sigma = compute_sigma_atm(t1, t2, X)
st.markdown(f"### Σ ATM_DIFF : **{current_sigma:.2f}**")

# ==================================================
# CACHE FILE (PER DATE + TS2)
# ==================================================
date_str = datetime.now().strftime("%Y-%m-%d")
ref_time = extract_time(t2).strftime("%H%M")
cache_file = os.path.join(CACHE_DIR, f"atm_{date_str}_ref_{ref_time}.csv")

if os.path.exists(cache_file):
    cache_df = pd.read_csv(cache_file)
else:
    cache_df = pd.DataFrame(columns=["timestamp1","time","sigma_atm_diff"])

# ==================================================
# TIME SERIES (LOOKUP → CALCULATE → APPEND)
# ==================================================
st.markdown("---")
st.subheader("🕒 Σ ATM_DIFF Over Time (Fixed Reference)")

progress = st.progress(0)
status = st.empty()

rows = []
valid_ts = [
    ts for ts in filtered_ts
    if time(9,16) <= extract_time(ts) <= time(15,45)
    and ts > t2
]

total = len(valid_ts)

for i, ts in enumerate(valid_ts, start=1):
    t = extract_time(ts)
    progress.progress(i / total)

    cached_row = cache_df[cache_df["timestamp1"] == ts]

    if not cached_row.empty:
        sigma_val = cached_row["sigma_atm_diff"].iloc[0]
        status.write(f"⚡ Loaded {t.strftime('%H:%M')} from cache")
    else:
        status.write(f"⏳ Calculating {t.strftime('%H:%M')} ...")
        sigma_val = compute_sigma_atm(ts, t2, X)

        cache_df.loc[len(cache_df)] = [
            ts,
            t.strftime("%H:%M"),
            round(sigma_val, 2)
        ]
        cache_df.to_csv(cache_file, index=False)

    rows.append({
        "Time": t.strftime("%H:%M"),
        "Σ ATM_DIFF": round(sigma_val, 2)
    })

status.write("✅ Completed")

time_series_df = pd.DataFrame(rows)
st.dataframe(time_series_df, use_container_width=True)

st.caption(f"Reference TS2: {t2} | Cached file: {os.path.basename(cache_file)}")

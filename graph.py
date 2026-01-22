import streamlit as st
import pandas as pd
import numpy as np
import os
from datetime import datetime, time, timedelta

# ==================================================
# CONFIG
# ==================================================
st.set_page_config(page_title="ATM Diff Trend", layout="wide")
st.title("📈 ATM Diff Strength – 15 Min Interval (09:15 to 15:30)")

DATA_DIR = "data"

STRIKE_WINDOW = 3      # FIXED
ATM_WINDOW = 2        # ±2 strikes

MARKET_START = time(9, 00)
MARKET_END   = time(15, 45)

# ==================================================
# LOAD FILES
# ==================================================
def load_csv_files():
    files = []
    for f in os.listdir(DATA_DIR):
        if f.startswith("option_chain_") and f.endswith(".csv"):
            ts = f.replace("option_chain_", "").replace(".csv", "")
            files.append((ts, os.path.join(DATA_DIR, f)))
    return sorted(files)

csv_files = load_csv_files()

if len(csv_files) < 2:
    st.error("Not enough CSV files.")
    st.stop()

# ==================================================
# PARSE TIMESTAMPS
# ==================================================
file_times = []

for ts, path in csv_files:
    try:
        dt = datetime.strptime(ts, "%Y-%m-%d_%H-%M")
        file_times.append((dt, path))
    except:
        continue

file_times.sort()

# ==================================================
# PICK 15-MIN INTERVAL FILES
# ==================================================
selected = []

# First file after 09:15
start_file = next(
    (x for x in file_times if x[0].time() >= MARKET_START),
    None
)

if not start_file:
    st.error("No files after 09:15")
    st.stop()

current_time = start_file[0]
selected.append(start_file)

while current_time.time() < MARKET_END:
    target = current_time + timedelta(minutes=10)

    next_file = next(
        (x for x in file_times if x[0] >= target),
        None
    )

    if not next_file:
        break

    selected.append(next_file)
    current_time = next_file[0]

# ==================================================
# ATM_DIFF CALCULATION (YOUR LOGIC)
# ==================================================
def compute_atm_sum(file_path):
    df = pd.read_csv(file_path)

    for c in ["Strike", "Stock_LTP", "CE_OI", "PE_OI"]:
        df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0)

    df["ce_x"] = (df["CE_OI"] * df["Strike"]) / 10000
    df["pe_x"] = (df["PE_OI"] * df["Strike"]) / 10000

    df["diff"] = np.nan
    df["atm_diff"] = np.nan

    for stk, g in df.groupby("Stock"):
        g = g.sort_values("Strike").reset_index()

        for i in range(len(g)):
            low = max(0, i - STRIKE_WINDOW)
            high = min(len(g) - 1, i + STRIKE_WINDOW)

            diff_val = g.loc[low:high, "pe_x"].sum() - g.loc[low:high, "ce_x"].sum()
            df.at[g.loc[i, "index"], "diff"] = diff_val

        ltp = g["Stock_LTP"].iloc[0]
        atm_idx = (g["Strike"] - ltp).abs().values.argmin()

        low = max(0, atm_idx - ATM_WINDOW)
        high = min(len(g) - 1, atm_idx + ATM_WINDOW)

        atm_avg = df.loc[g.loc[low:high, "index"], "diff"].mean()
        df.loc[g["index"], "atm_diff"] = atm_avg

    return df["atm_diff"].sum() / 1000

# ==================================================
# BUILD SERIES
# ==================================================
times = []
values = []

with st.spinner("Calculating ATM Diff Trend..."):
    for dt, path in selected:
        atm_sum = compute_atm_sum(path)
        times.append(dt)
        values.append(atm_sum)

trend_df = pd.DataFrame({"Time": times, "ATM_Sum": values}).set_index("Time")

# ==================================================
# INVERT FOR DIRECTION
# ==================================================
trend_df["ATM_Sum"] = -trend_df["ATM_Sum"]

# ==================================================
# DISPLAY
# ==================================================
st.subheader("Σ ATM Diff (×1000) – Intraday Strength")

st.line_chart(trend_df["ATM_Sum"])

st.markdown(
    f"""
**High:** {trend_df["ATM_Sum"].max():.1f}  
**Low:** {trend_df["ATM_Sum"].min():.1f}  
**Close:** {trend_df["ATM_Sum"].iloc[-1]:.1f}
"""
)

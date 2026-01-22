import streamlit as st
import pandas as pd
import os
from datetime import datetime, time

# ==================================================
# CONFIG
# ==================================================
st.set_page_config(page_title="ATM Diff Trend", layout="wide")
st.title("📈 ATM Diff Strength Trend (09:15 – 15:30)")

DATA_DIR = "data"

MARKET_START = time(9, 15)
MARKET_END   = time(15, 30)

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

if len(csv_files) < 3:
    st.error("Not enough CSV files in data/")
    st.stop()

# ==================================================
# ATM DIFF CALCULATION
# ==================================================
def compute_atm_sum(file_path):
    df = pd.read_csv(file_path)

    required = ["Stock", "Strike", "Stock_LTP", "CE_OI", "PE_OI"]
    if not all(c in df.columns for c in required):
        return None

    for c in ["Strike", "Stock_LTP", "CE_OI", "PE_OI"]:
        df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0)

    df["ce_x"] = (df["CE_OI"] * df["Strike"]) / 10000
    df["pe_x"] = (df["PE_OI"] * df["Strike"]) / 10000
    df["diff"] = df["pe_x"] - df["ce_x"]

    atm_values = []

    for stk, g in df.groupby("Stock"):
        g = g.sort_values("Strike").reset_index(drop=True)
        ltp = g["Stock_LTP"].iloc[0]

        atm_idx = (g["Strike"] - ltp).abs().idxmin()
        low = max(0, atm_idx - 2)
        high = min(len(g) - 1, atm_idx + 2)

        atm_avg = g.loc[low:high, "diff"].mean()
        atm_values.append(atm_avg)

    return sum(atm_values) / 1000   # scaled

# ==================================================
# BUILD TIME SERIES (MARKET HOURS ONLY)
# ==================================================
times = []
values = []

for ts, path in csv_files:
    try:
        dt = datetime.strptime(ts, "%Y-%m-%d_%H-%M")
        t = dt.time()

        if not (MARKET_START <= t <= MARKET_END):
            continue

        val = compute_atm_sum(path)

        if val is not None:
            times.append(dt)
            values.append(val)

    except:
        pass

trend_df = pd.DataFrame({
    "Time": times,
    "ATM_Sum": values
}).sort_values("Time")

if trend_df.empty:
    st.error("No data available in market hours range.")
    st.stop()

trend_df = trend_df.set_index("Time")

# ==================================================
# DISPLAY CHART
# ==================================================
st.subheader("Σ ATM Diff (x1000) – Intraday Strength")

st.line_chart(trend_df["ATM_Sum"])

import streamlit as st
import pandas as pd
import numpy as np
import os
from datetime import datetime, time

# ==================================================
# CONFIG
# ==================================================
st.set_page_config(page_title="ATM Diff Trend", layout="wide")
st.title("📈 ATM Diff Strength – Full Day (09:15 to 15:30)")

DATA_DIR = "data"
STRIKE_WINDOW = 4      # Same X as main app
ATM_WINDOW = 2        # ±2 strikes for ATM avg

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

if len(csv_files) < 2:
    st.error("Not enough CSV files.")
    st.stop()

# ==================================================
# ATM_DIFF CALCULATION (YOUR EXACT LOGIC)
# ==================================================
def compute_atm_sum(file_path):
    df = pd.read_csv(file_path)

    # Safety
    for c in ["Strike","Stock_LTP","CE_OI","PE_OI"]:
        df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0)

    df["d_ce"] = df["CE_OI"] - df["CE_OI"]
    df["d_pe"] = df["PE_OI"] - df["PE_OI"]

    # We simulate previous snapshot using same file
    # (Because intraday trend relies on OI distribution, not delta)

    df["ce_x"] = (df["CE_OI"] * df["Strike"]) / 10000
    df["pe_x"] = (df["PE_OI"] * df["Strike"]) / 10000

    df["diff"] = np.nan
    df["atm_diff"] = np.nan

    for stk, g in df.groupby("Stock"):
        g = g.sort_values("Strike").reset_index()

        # Sliding strike window diff
        for i in range(len(g)):
            low = max(0, i - STRIKE_WINDOW)
            high = min(len(g) - 1, i + STRIKE_WINDOW)

            diff_val = g.loc[low:high, "pe_x"].sum() - g.loc[low:high, "ce_x"].sum()
            df.at[g.loc[i,"index"], "diff"] = diff_val

        # ATM Diff
        ltp = g["Stock_LTP"].iloc[0]
        atm_idx = (g["Strike"] - ltp).abs().values.argmin()

        low = max(0, atm_idx - ATM_WINDOW)
        high = min(len(g) - 1, atm_idx + ATM_WINDOW)

        atm_avg = df.loc[g.loc[low:high,"index"], "diff"].mean()
        df.loc[g["index"], "atm_diff"] = atm_avg

    return df["atm_diff"].sum() / 1000

# ==================================================
# BUILD FULL DAY SERIES
# ==================================================
times = []
values = []

for ts, path in csv_files:
    try:
        dt = datetime.strptime(ts, "%Y-%m-%d_%H-%M")
        if not (MARKET_START <= dt.time() <= MARKET_END):
            continue

        atm_sum = compute_atm_sum(path)

        times.append(dt)
        values.append(atm_sum)

    except:
        continue

trend_df = pd.DataFrame({"Time": times, "ATM_Sum": values}).sort_values("Time")
trend_df = trend_df.set_index("Time")

if trend_df.empty:
    st.error("No valid data in market hours.")
    st.stop()

# ==================================================
# DISPLAY
# ==================================================
st.subheader("Σ ATM Diff (×1000) – Intraday Institutional Strength")

st.line_chart(trend_df["ATM_Sum"])

# Optional stats
st.markdown(
    f"""
**High:** {trend_df["ATM_Sum"].max():.1f}  
**Low:** {trend_df["ATM_Sum"].min():.1f}  
**Close:** {trend_df["ATM_Sum"].iloc[-1]:.1f}
"""
)

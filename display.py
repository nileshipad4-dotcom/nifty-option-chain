import streamlit as st
import pandas as pd
import numpy as np
import os
from datetime import time

# =====================================
# CONFIG
# =====================================
st.set_page_config(page_title="ΔΔ Max Pain Viewer", layout="wide")
st.title("📊 ΔΔ Max Pain Viewer")

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

def ts_in_market_hours(ts):
    try:
        t = ts.split("_")[-1]
        hh, mm, _ = t.split("-")
        tt = time(int(hh), int(mm))
        return time(8, 0) <= tt <= time(16, 30)
    except:
        return False

csv_files = [(ts, path) for ts, path in load_csv_files() if ts_in_market_hours(ts)]

if len(csv_files) < 3:
    st.error("Need at least 3 option chain CSV files between 08:00 and 16:30")
    st.stop()

timestamps = [ts for ts, _ in csv_files]
file_map = dict(csv_files)

def short_ts(ts):
    return ts.split("_")[-1].replace("-", ":")

# =====================================
# TIMESTAMP SELECTION
# =====================================
st.subheader("🕒 Timestamp Selection")

t1 = st.selectbox("Base Timestamp (T1)", timestamps, index=0)

c1, c2 = st.columns(2)
with c1:
    t2 = st.selectbox("T2", timestamps, index=1)
with c2:
    t3 = st.selectbox("T3", timestamps, index=2)

compare_ts = [t2, t3]
time_cols = sorted([short_ts(ts) for ts in compare_ts])

# =====================================
# LOAD BASE DATA
# =====================================
df_base = pd.read_csv(file_map[t1])

required_cols = {
    "Stock", "Strike", "Max_Pain",
    "Stock_LTP", "Stock_High", "Stock_Low"
}
if not required_cols.issubset(df_base.columns):
    st.error("CSV must contain Stock, Strike, Max_Pain, Stock_LTP, Stock_High, Stock_Low")
    st.stop()

df_base["Stock"] = df_base["Stock"].astype(str).str.upper().str.strip()
all_stocks = sorted(df_base["Stock"].unique())

# =====================================
# ΔΔ MP CALCULATION
# =====================================
def compute_ddmp(df):
    df = df.copy()

    for ts in compare_ts:
        label = short_ts(ts)

        df_ts = pd.read_csv(file_map[ts])
        df_ts["Stock"] = df_ts["Stock"].astype(str).str.upper().str.strip()

        df = df.merge(
            df_ts[["Stock", "Strike", "Max_Pain", "Stock_LTP"]],
            on=["Stock", "Strike"],
            suffixes=("", f"_{label}")
        )

        delta_col = f"_delta_{label}"
        df[delta_col] = df["Max_Pain"] - df[f"Max_Pain_{label}"]
        df[label] = np.nan

        for _, sdf in df.sort_values("Strike").groupby("Stock"):
            vals = sdf[delta_col].astype(float).values
            diff = vals - np.roll(vals, -1)
            diff[-1] = np.nan
            df.loc[sdf.index, label] = diff

        df.drop(columns=[f"Max_Pain_{label}", delta_col], inplace=True)

    return df

# =====================================
# MONOTONIC FILTER
# =====================================
def is_monotonic_2_of_2(values):
    return values[0] <= values[1] or values[0] >= values[1]

# =====================================
# FILTER PARAMETERS
# =====================================
st.subheader("🎛 Filter Parameters")

p1, p2 = st.columns(2)
with p1:
    ltp_pct_limit = st.number_input("Max % distance from LTP", 0.0, 50.0, 5.0, 0.5)
with p2:
    ddmp_diff_limit = st.number_input("Min |ΔΔ MP(last − first)|", 0.0, value=147.0, step=10.0)

# =====================================
# FILTERED TABLE
# =====================================
st.subheader("🧩 Stocks & Strikes with Strong Consistent ΔΔ MP Trend")

filtered_rows = []
df_all = compute_ddmp(df_base)

# Pre-compute ATM strikes & Max Pain per stock
atm_map = {}
mp_map = {}

for stock in all_stocks:
    sdf = df_all[df_all["Stock"] == stock].sort_values("Strike")
    if sdf.empty:
        continue

    ltp = float(sdf["Stock_LTP"].iloc[0])
    strikes = sdf["Strike"].values

    for i in range(len(strikes) - 1):
        if strikes[i] <= ltp <= strikes[i + 1]:
            atm_map[stock] = {strikes[i], strikes[i + 1]}
            break

    mp_map[stock] = sdf.loc[sdf["Max_Pain"].idxmin(), "Strike"]

# Load LTPs for % change
df_t2 = pd.read_csv(file_map[t2]).set_index("Stock")
df_t3 = pd.read_csv(file_map[t3]).set_index("Stock")

for stock in all_stocks:
    sdf = df_all[df_all["Stock"] == stock].sort_values("Strike")
    if sdf.empty:
        continue

    ltp1 = float(sdf["Stock_LTP"].iloc[0])
    ltp2 = float(df_t2.loc[stock, "Stock_LTP"])
    ltp3 = float(df_t3.loc[stock, "Stock_LTP"])

    pct_12 = (ltp2 - ltp1) / ltp1 * 100 if ltp1 else np.nan
    pct_23 = (ltp3 - ltp2) / ltp2 * 100 if ltp2 else np.nan

    high = float(sdf["Stock_High"].iloc[0])
    low = float(sdf["Stock_Low"].iloc[0])

    for _, row in sdf.iterrows():
        values = [row[c] for c in time_cols]
        if any(pd.isna(values)):
            continue

        if not is_monotonic_2_of_2(values):
            continue

        if abs(values[-1] - values[0]) <= ddmp_diff_limit:
            continue

        strike = float(row["Strike"])
        pct_diff = abs(strike - ltp1) / ltp1 * 100
        if pct_diff > ltp_pct_limit:
            continue

        filtered_rows.append({
            "Stock": stock,
            "Strike": int(strike),
            **{c: int(row[c]) for c in time_cols},
            "%Δ LTP T1→T2": round(pct_12, 2),
            "%Δ LTP T2→T3": round(pct_23, 2),
            "Stock_LTP": round(ltp1, 2),
            "Stock_High": round(high, 2),
            "Stock_Low": round(low, 2),
        })

# =====================================
# DISPLAY
# =====================================
if filtered_rows:
    filtered_df = pd.DataFrame(filtered_rows).sort_values(["Stock", "Strike"])
    st.dataframe(filtered_df, use_container_width=True)
else:
    st.info("No strikes matched the current filter parameters.")

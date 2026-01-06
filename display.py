import streamlit as st
import pandas as pd
import numpy as np
import os

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

csv_files = load_csv_files()
if len(csv_files) < 4:
    st.error("Need at least 4 option chain CSV files")
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

c1, c2, c3 = st.columns(3)
with c1:
    t2 = st.selectbox("T2", timestamps, index=1)
with c2:
    t3 = st.selectbox("T3", timestamps, index=2)
with c3:
    t4 = st.selectbox("T4", timestamps, index=3)

compare_ts = [t2, t3, t4]
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
            df_ts[["Stock", "Strike", "Max_Pain"]],
            on=["Stock", "Strike"],
            suffixes=("", "_cmp")
        )

        delta_col = f"_delta_{label}"
        df[delta_col] = df["Max_Pain"] - df["Max_Pain_cmp"]
        df[label] = np.nan

        for _, sdf in df.sort_values("Strike").groupby("Stock"):
            vals = sdf[delta_col].astype(float).values
            diff = vals - np.roll(vals, -1)
            diff[-1] = np.nan
            df.loc[sdf.index, label] = diff

        df.drop(columns=["Max_Pain_cmp", delta_col], inplace=True)

    return df

# =====================================
# MONOTONIC FILTER
# =====================================
def is_monotonic_3_of_3(values):
    return (
        values[0] <= values[1] <= values[2] or
        values[0] >= values[1] >= values[2]
    )

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

for stock in all_stocks:
    sdf = df_all[df_all["Stock"] == stock].sort_values("Strike")
    if sdf.empty:
        continue

    ltp = float(sdf["Stock_LTP"].iloc[0])
    high = float(sdf["Stock_High"].iloc[0])
    low = float(sdf["Stock_Low"].iloc[0])

    if ltp <= 0:
        continue

    for _, row in sdf.iterrows():
        values = [row[c] for c in time_cols]
        if any(pd.isna(values)):
            continue

        if not is_monotonic_3_of_3(values):
            continue

        if abs(values[-1] - values[0]) <= ddmp_diff_limit:
            continue

        strike = float(row["Strike"])
        pct_diff = abs(strike - ltp) / ltp * 100
        if pct_diff > ltp_pct_limit:
            continue

        filtered_rows.append({
            "Stock": stock,
            "Strike": int(strike),
            **{c: int(row[c]) for c in time_cols},
            "Stock_LTP": round(ltp, 2),
            "Stock_High": round(high, 2),
            "Stock_Low": round(low, 2),
        })

# =====================================
# DISPLAY FILTERED TABLE WITH COLORS
# =====================================
if filtered_rows:
    filtered_df = pd.DataFrame(filtered_rows).sort_values(["Stock", "Strike"])

    def color_row(row):
        stock = row["Stock"]
        strike = row["Strike"]
        high = row["Stock_High"]
        low = row["Stock_Low"]

        # Base row color (unchanged logic)
        if strike == mp_map.get(stock):
            base_style = "background-color:#4E342E;color:white"
        elif strike in atm_map.get(stock, set()):
            base_style = "background-color:#003366;color:white"
        elif strike > row["Stock_LTP"]:
            base_style = "background-color:#004d00;color:white"
        else:
            base_style = "background-color:#660000;color:white"

        styles = []
        for col in row.index:
            if col in ("Stock_High", "Stock_Low"):
                if low <= strike <= high:
                    styles.append("")  # no highlight
                else:
                    styles.append(base_style)
            else:
                styles.append(base_style)

        return styles

    st.dataframe(
        filtered_df.style
        .apply(color_row, axis=1)
        .format(
            {
                "Strike": "{:.0f}",
                "Stock_LTP": "{:.2f}",
                "Stock_High": "{:.2f}",
                "Stock_Low": "{:.2f}",
                **{c: "{:.0f}" for c in time_cols}
            }
        ),
        use_container_width=True
    )
else:
    st.info("No strikes matched the current filter parameters.")

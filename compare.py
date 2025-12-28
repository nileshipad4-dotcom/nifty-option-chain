# stock OC main
import streamlit as st
import pandas as pd
import os
import numpy as np

# =====================================
# STREAMLIT CONFIG
# =====================================
st.set_page_config(page_title="Max Pain Comparison", layout="wide")
st.title("📊 Max Pain Comparison Dashboard")

DATA_DIR = "data"

# =====================================
# LOAD CSV FILES
# =====================================
def load_csv_files():
    files = []
    if not os.path.exists(DATA_DIR):
        return files

    for f in os.listdir(DATA_DIR):
        if f.startswith("option_chain_") and f.endswith(".csv"):
            ts = f.replace("option_chain_", "").replace(".csv", "")
            files.append((ts, os.path.join(DATA_DIR, f)))

    return sorted(files, reverse=True)

csv_files = load_csv_files()
if len(csv_files) < 3:
    st.error("Need at least 3 CSV files to compare.")
    st.stop()

timestamps = [ts for ts, _ in csv_files]
file_map = dict(csv_files)

def short_ts(ts):
    return ts.split("_")[-1].replace("-", ":")

# =====================================
# DROPDOWNS
# =====================================
c1, c2, c3 = st.columns(3)
with c1:
    t1 = st.selectbox("Timestamp 1 (Latest)", timestamps, 0)
with c2:
    t2 = st.selectbox("Timestamp 2", timestamps, 1)
with c3:
    t3 = st.selectbox("Timestamp 3", timestamps, 2)

t1_lbl = short_ts(t1)
t2_lbl = short_ts(t2)
t3_lbl = short_ts(t3)

delta_12 = f"Δ MP ({t1_lbl}-{t2_lbl})"
delta_23 = f"Δ MP ({t2_lbl}-{t3_lbl})"
sum_12_col = f"Σ {delta_12}"

# =====================================
# LOAD DATA
# =====================================
df1 = pd.read_csv(file_map[t1])
df2 = pd.read_csv(file_map[t2])
df3 = pd.read_csv(file_map[t3])

# =====================================
# PREPARE DATA
# =====================================
df1 = df1[["Stock", "Strike", "Max_Pain", "Stock_LTP"]].rename(columns={"Max_Pain": t1_lbl})
df2 = df2[["Stock", "Strike", "Max_Pain"]].rename(columns={"Max_Pain": t2_lbl})
df3 = df3[["Stock", "Strike", "Max_Pain"]].rename(columns={"Max_Pain": t3_lbl})

df = (
    df1
    .merge(df2, on=["Stock", "Strike"])
    .merge(df3, on=["Stock", "Strike"])
)

df[delta_12] = df[t1_lbl] - df[t2_lbl]
df[delta_23] = df[t2_lbl] - df[t3_lbl]

# =====================================
# ROLLING SUM (trend check only)
# =====================================
df[sum_12_col] = np.nan
for stock, sdf in df.sort_values("Strike").groupby("Stock"):
    df.loc[sdf.index, sum_12_col] = (
        sdf[delta_12]
        .rolling(window=7, center=True, min_periods=1)
        .sum()
        .values
    )

# =====================================
# FINAL COLUMN ORDER
# =====================================
df = df[
    [
        "Stock",
        "Strike",
        t1_lbl,
        t2_lbl,
        delta_12,
        t3_lbl,
        delta_23,
        sum_12_col,
        "Stock_LTP",
    ]
]

# =====================================
# INSERT BLANK ROWS
# =====================================
rows = []
for stock, sdf in df.sort_values(["Stock", "Strike"]).groupby("Stock"):
    rows.append(sdf)
    rows.append(pd.DataFrame([{col: np.nan for col in df.columns}]))

final_df = pd.concat(rows, ignore_index=True)

# =====================================
# COMPUTE STOCK SIGNALS
# =====================================
def compute_stock_signals(data):
    signals = {}

    for stock in data["Stock"].dropna().unique():
        sdf = data[
            (data["Stock"] == stock) & data["Strike"].notna()
        ].sort_values("Strike")

        if len(sdf) < 9:
            continue

        ltp = float(sdf["Stock_LTP"].iloc[0])
        strikes = sdf["Strike"].values

        atm_idx = None
        for i in range(len(strikes) - 1):
            if strikes[i] <= ltp <= strikes[i + 1]:
                atm_idx = i + 1
                break

        if atm_idx is None:
            continue

        mid = sdf.iloc[4:-4]
        diffs = np.diff(mid[sum_12_col].astype(float).values)

        if len(diffs) == 0:
            continue

        inc_ratio = np.sum(diffs > 0) / len(diffs)
        dec_ratio = np.sum(diffs < 0) / len(diffs)

        trend = None
        if inc_ratio >= 0.9:
            trend = "red"
        elif dec_ratio >= 0.9:
            trend = "green"
        else:
            continue

        above = sdf.iloc[atm_idx:atm_idx + 5]
        below = sdf.iloc[max(atm_idx - 5, 0):atm_idx]

        if len(above) < 3 or len(below) < 3:
            continue

        above_gt = (above[delta_12] > above[delta_23]).sum()
        above_lt = (above[delta_12] < above[delta_23]).sum()
        below_gt = (below[delta_12] > below[delta_23]).sum()
        below_lt = (below[delta_12] < below[delta_23]).sum()

        signal = None
        if above_gt >= 3 and below_lt >= 3:
            signal = "red"
        elif above_lt >= 3 and below_gt >= 3:
            signal = "green"

        if signal == trend:
            signals[stock] = signal

    return signals

stock_signals = compute_stock_signals(final_df)

# =====================================
# FILTERED DATAFRAMES
# =====================================
green_stocks = [s for s, v in stock_signals.items() if v == "green"]
red_stocks   = [s for s, v in stock_signals.items() if v == "red"]

green_df = final_df[final_df["Stock"].isin(green_stocks) | final_df["Stock"].isna()]
red_df   = final_df[final_df["Stock"].isin(red_stocks) | final_df["Stock"].isna()]

# =====================================
# HIGHLIGHTING
# =====================================
def highlight_rows(data):
    styles = pd.DataFrame("", index=data.index, columns=data.columns)

    for stock in data["Stock"].dropna().unique():
        sdf = data[
            (data["Stock"] == stock) & data["Strike"].notna()
        ].sort_values("Strike")

        if len(sdf) < 2:
            continue

        ltp = float(sdf["Stock_LTP"].iloc[0])
        strikes = sdf["Strike"].values

        # ATM highlight
        for i in range(len(strikes) - 1):
            if strikes[i] <= ltp <= strikes[i + 1]:
                styles.loc[sdf.index[i]] = "background-color:#003366;color:white"
                styles.loc[sdf.index[i + 1]] = "background-color:#003366;color:white"
                break

        # Max Pain (TS1)
        styles.loc[sdf[t1_lbl].idxmin()] = "background-color:#8B0000;color:white"

        if stock not in stock_signals:
            continue

        color = (
            "background-color:#8B0000;color:white"
            if stock_signals[stock] == "red"
            else "background-color:#004d00;color:white"
        )

        styles.loc[sdf.index, delta_12] = color
        styles.loc[sdf.index, delta_23] = color

    return styles

# =====================================
# FORMATTERS
# =====================================
formatters = {
    col: "{:.2f}" if col == "Stock_LTP" else "{:.0f}"
    for col in final_df.columns if col != "Stock"
}

# =====================================
# DISPLAY
# =====================================
st.subheader("🟢 GREEN SIGNAL STOCKS")
st.dataframe(
    green_df.style.apply(highlight_rows, axis=None).format(formatters, na_rep=""),
    use_container_width=True,
)

st.subheader("🔴 RED SIGNAL STOCKS")
st.dataframe(
    red_df.style.apply(highlight_rows, axis=None).format(formatters, na_rep=""),
    use_container_width=True,
)

st.subheader(f"📊 ALL STOCKS: {t1_lbl} vs {t2_lbl} vs {t3_lbl}")
st.dataframe(
    final_df.style.apply(highlight_rows, axis=None).format(formatters, na_rep=""),
    use_container_width=True,
)

# =====================================
# DOWNLOAD
# =====================================
st.download_button(
    "⬇️ Download Comparison CSV",
    final_df.to_csv(index=False),
    f"max_pain_{t1_lbl}_{t2_lbl}_{t3_lbl}.csv",
    "text/csv",
)

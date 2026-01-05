import streamlit as st
import pandas as pd
import numpy as np
import os
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
    if not os.path.exists(DATA_DIR):
        return files
    for f in os.listdir(DATA_DIR):
        if f.startswith("option_chain_") and f.endswith(".csv"):
            ts = f.replace("option_chain_", "").replace(".csv", "")
            files.append((ts, os.path.join(DATA_DIR, f)))
    return sorted(files, reverse=True)

csv_files = load_csv_files()
if len(csv_files) < 6:
    st.error("Need at least 6 CSV files.")
    st.stop()

timestamps = [ts for ts, _ in csv_files]
file_map = dict(csv_files)

def short_ts(ts):
    return ts.split("_")[-1].replace("-", ":")

# =====================================
# TIMESTAMP SELECTORS (6)
# =====================================
cols = st.columns(6)
t = []
for i in range(6):
    with cols[i]:
        t.append(st.selectbox(f"Timestamp {i+1}", timestamps, i))

labels = [short_ts(x) for x in t]

# =====================================
# COLUMN NAMES
# =====================================
delta_mp_cols = [
    f"Δ MP ({labels[0]}-{labels[1]})",
    f"Δ MP ({labels[2]}-{labels[3]})",
    f"Δ MP ({labels[4]}-{labels[5]})",
]

ltp_pct_cols = [
    f"% LTP Ch ({labels[0]}-{labels[1]})",
    f"% LTP Ch ({labels[2]}-{labels[3]})",
    f"% LTP Ch ({labels[4]}-{labels[5]})",
]

# =====================================
# LOAD DATAFRAMES
# =====================================
dfs = []
for i, ts in enumerate(t):
    df = pd.read_csv(file_map[ts])
    df = df[["Stock", "Strike", "Max_Pain", "Stock_LTP"]].rename(
        columns={
            "Max_Pain": f"MP_{i}",
            "Stock_LTP": f"LTP_{i}",
        }
    )
    dfs.append(df)

# % change from TS1
raw_t1 = pd.read_csv(file_map[t[0]])
pct_col = f"% Ch ({labels[0]})"
dfs[0][pct_col] = raw_t1["Stock_%_Change"] if "Stock_%_Change" in raw_t1.columns else np.nan

# =====================================
# MERGE
# =====================================
df = dfs[0]
for i in range(1, 6):
    df = df.merge(dfs[i], on=["Stock", "Strike"])

# =====================================
# PAIRWISE CALCULATIONS
# =====================================
pairs = [(0, 1), (2, 3), (4, 5)]

for idx, (a, b) in enumerate(pairs):
    df[delta_mp_cols[idx]] = df[f"MP_{a}"] - df[f"MP_{b}"]
    df[ltp_pct_cols[idx]] = (
        (df[f"LTP_{a}"] - df[f"LTP_{b}"]) / df[f"LTP_{b}"]
    ) * 100

# =====================================
# CLEAN
# =====================================
df["Stock"] = df["Stock"].astype(str).str.upper().str.strip()
EXCLUDE = {"NIFTY", "BANKNIFTY", "FINNIFTY", "MIDCPNIFTY"}
df = df[~df["Stock"].isin(EXCLUDE)]

# =====================================
# FINAL COLUMN ORDER
# =====================================
df = df[
    ["Stock", "Strike"]
    + delta_mp_cols
    + ["LTP_0", pct_col]
    + ltp_pct_cols
].rename(columns={"LTP_0": "Stock_LTP"})

# =====================================
# FILTER ±6 STRIKES
# =====================================
def filter_strikes_around_ltp(df, below=6, above=6):
    out = []
    for stock, sdf in df.groupby("Stock"):
        sdf = sdf.sort_values("Strike")
        ltp = float(sdf["Stock_LTP"].iloc[0])
        strikes = sdf["Strike"].values

        atm = None
        for i in range(len(strikes) - 1):
            if strikes[i] <= ltp <= strikes[i + 1]:
                atm = i
                break

        if atm is None:
            continue

        out.append(sdf.iloc[max(0, atm - below): atm + 2 + above])
        out.append(pd.DataFrame([{c: np.nan for c in df.columns}]))

    return pd.concat(out[:-1], ignore_index=True)

# =====================================
# HIGHLIGHTING
# =====================================
def highlight_rows(data):
    styles = pd.DataFrame("", index=data.index, columns=data.columns)

    for stock in data["Stock"].dropna().unique():
        sdf = data[(data["Stock"] == stock) & data["Strike"].notna()].sort_values("Strike")
        ltp = float(sdf["Stock_LTP"].iloc[0])
        strikes = sdf["Strike"].values

        for i in range(len(strikes) - 1):
            if strikes[i] <= ltp <= strikes[i + 1]:
                styles.loc[sdf.index[i]] = "background-color:#003366;color:white"
                styles.loc[sdf.index[i+1]] = "background-color:#003366;color:white"
                break

        styles.loc[sdf[delta_mp_cols[0]].abs().idxmax()] = "background-color:#8B0000;color:white"

    return styles

# =====================================
# DISPLAY
# =====================================
display_df = filter_strikes_around_ltp(df)

st.dataframe(
    display_df
    .style.apply(highlight_rows, axis=None)
    .format(
        {c: "{:.0f}" for c in delta_mp_cols}
        | {c: "{:.2f}" for c in ltp_pct_cols + ["Stock_LTP"]}
        | {pct_col: "{:.3f}"},
        na_rep="",
    ),
    use_container_width=True,
)

# =====================================
# DOWNLOAD
# =====================================
st.download_button(
    "⬇️ Download CSV",
    df.to_csv(index=False),
    "max_pain_pairwise_6_ts.csv",
    "text/csv",
)

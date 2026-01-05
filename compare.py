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
mp_cols = [f"MP ({lbl})" for lbl in labels]
delta_mp_cols = [f"Δ MP ({labels[0]}-{labels[i]})" for i in range(1, 6)]
ltp_pct_cols = [f"% LTP Ch ({labels[0]}-{labels[i]})" for i in range(1, 6)]

# =====================================
# LOAD ALL DATAFRAMES
# =====================================
dfs = []
for i, ts in enumerate(t):
    df = pd.read_csv(file_map[ts])
    df = df[["Stock", "Strike", "Max_Pain", "Stock_LTP"]].rename(
        columns={"Max_Pain": mp_cols[i], "Stock_LTP": f"Stock_LTP_t{i+1}"}
    )
    dfs.append(df)

# % change column only needed from t1
if "Stock_%_Change" in pd.read_csv(file_map[t[0]]).columns:
    dfs[0]["% Ch"] = pd.read_csv(file_map[t[0]])["Stock_%_Change"]
else:
    dfs[0]["% Ch"] = np.nan

# =====================================
# MERGE ALL
# =====================================
df = dfs[0]
for i in range(1, 6):
    df = df.merge(dfs[i], on=["Stock", "Strike"])

# =====================================
# CALCULATIONS
# =====================================
for i in range(1, 6):
    df[delta_mp_cols[i-1]] = df[mp_cols[0]] - df[mp_cols[i]]

    df[ltp_pct_cols[i-1]] = (
        (df["Stock_LTP_t1"] - df[f"Stock_LTP_t{i+1}"])
        / df[f"Stock_LTP_t{i+1}"]
    ) * 100

# =====================================
# CLEAN
# =====================================
df["Stock"] = df["Stock"].str.upper().str.strip()
EXCLUDE = {"NIFTY", "BANKNIFTY", "FINNIFTY", "MIDCPNIFTY"}
df = df[~df["Stock"].isin(EXCLUDE)]

# =====================================
# FINAL COLUMN ORDER
# =====================================
df = df[
    ["Stock", "Strike"]
    + delta_mp_cols
    + ["Stock_LTP_t1", "% Ch"]
    + ltp_pct_cols
].rename(columns={"Stock_LTP_t1": "Stock_LTP"})

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
        sdf = data[(data["Stock"] == stock) & data["Strike"].notna()]
        sdf = sdf.sort_values("Strike")

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
        | {"% Ch": "{:.3f}"},
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
    "max_pain_6_ts.csv",
    "text/csv",
)

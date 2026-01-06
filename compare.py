import streamlit as st
import pandas as pd
import numpy as np
import os
from datetime import time
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

timestamps_all = [ts for ts, _ in csv_files]
file_map = dict(csv_files)

# =====================================
# TIME FILTERING
# =====================================
def extract_time(ts):
    try:
        t = ts.split("_")[-1]
        hh, mm = map(int, t.split("-")[:2])
        return time(hh, mm)
    except Exception:
        return None

START_TIME = time(7, 30)
END_TIME = time(16, 0)

filtered_ts = [
    ts for ts in timestamps_all
    if extract_time(ts) and START_TIME <= extract_time(ts) <= END_TIME
]

# =====================================
# SESSION STATE
# =====================================
for k in ["t3_manual", "t5_manual"]:
    if k not in st.session_state:
        st.session_state[k] = False

# =====================================
# TIMESTAMP SELECTORS
# =====================================
cols = st.columns(6)

with cols[0]:
    t1 = st.selectbox("Timestamp 1", filtered_ts, 0)
with cols[1]:
    t2 = st.selectbox("Timestamp 2", filtered_ts, 1)
with cols[2]:
    t3_default = t2 if not st.session_state.t3_manual else None
    t3 = st.selectbox(
        "Timestamp 3",
        filtered_ts,
        index=filtered_ts.index(t3_default) if t3_default in filtered_ts else 2,
    )
    if t3 != t2:
        st.session_state.t3_manual = True
with cols[3]:
    t4 = st.selectbox("Timestamp 4", filtered_ts, 3)
with cols[4]:
    t5_default = t4 if not st.session_state.t5_manual else None
    t5 = st.selectbox(
        "Timestamp 5",
        filtered_ts,
        index=filtered_ts.index(t5_default) if t5_default in filtered_ts else 4,
    )
    if t5 != t4:
        st.session_state.t5_manual = True
with cols[5]:
    t6 = st.selectbox("Timestamp 6 (ALL)", timestamps_all, 0)

t = [t1, t2, t3, t4, t5, t6]

# =====================================
# LABELS
# =====================================
def short_ts(ts):
    return ts.split("_")[-1].replace("-", ":")

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
# LOAD DATA
# =====================================
dfs = []
for i, ts in enumerate(t):
    d = pd.read_csv(file_map[ts])
    d = d[[
        "Stock", "Strike", "Max_Pain", "Stock_LTP",
        "CE_OI", "PE_OI", "CE_Volume", "PE_Volume"
    ]].rename(columns={
        "Max_Pain": f"MP_{i}",
        "Stock_LTP": f"LTP_{i}",
        "CE_OI": f"CE_OI_{i}",
        "PE_OI": f"PE_OI_{i}",
        "CE_Volume": f"CE_VOL_{i}",
        "PE_Volume": f"PE_VOL_{i}",
    })
    dfs.append(d)

# =====================================
# TIMESTAMP 1 EXTRA DATA
# =====================================
raw_t1 = pd.read_csv(file_map[t1])
pct_col = f"% Ch ({labels[0]})"

dfs[0][pct_col] = raw_t1["Stock_%_Change"]
dfs[0]["Stock_High"] = raw_t1["Stock_High"]
dfs[0]["Stock_Low"] = raw_t1["Stock_Low"]

# =====================================
# MERGE
# =====================================
df = dfs[0]
for i in range(1, 6):
    df = df.merge(dfs[i], on=["Stock", "Strike"])

# =====================================
# PAIRWISE CALCS
# =====================================
pairs = [(0, 1), (2, 3), (4, 5)]
for idx, (a, b) in enumerate(pairs):
    df[delta_mp_cols[idx]] = df[f"MP_{a}"] - df[f"MP_{b}"]
    df[ltp_pct_cols[idx]] = (
        (df[f"LTP_{a}"] - df[f"LTP_{b}"]) / df[f"LTP_{b}"]
    ) * 100

# =====================================
# Δ(Δ MP T2 − T3)
# =====================================
df["Δ(Δ MP T2-T3)"] = delta_mp_cols and (
    df[delta_mp_cols[1]] - df[delta_mp_cols[2]]
)

# =====================================
# OI / VOLUME DIFF (T2 - T3)
# =====================================
df["Δ CE OI (T2-T3)"] = df["CE_OI_1"] - df["CE_OI_2"]
df["Δ PE OI (T2-T3)"] = df["PE_OI_1"] - df["PE_OI_2"]
df["Δ CE Vol (T2-T3)"] = df["CE_VOL_1"] - df["CE_VOL_2"]
df["Δ PE Vol (T2-T3)"] = df["PE_VOL_1"] - df["PE_VOL_2"]

# =====================================
# CLEAN
# =====================================
df["Stock"] = df["Stock"].str.upper().str.strip()
EXCLUDE = {"NIFTY", "BANKNIFTY", "FINNIFTY", "MIDCPNIFTY"}
df = df[~df["Stock"].isin(EXCLUDE)]

df = df[
    ["Stock", "Strike", "Stock_High", "Stock_Low"]
    + delta_mp_cols
    + ["Δ(Δ MP T2-T3)", "LTP_0", pct_col]
    + ltp_pct_cols
    + ["Δ CE OI (T2-T3)", "Δ PE OI (T2-T3)",
       "Δ CE Vol (T2-T3)", "Δ PE Vol (T2-T3)"]
].rename(columns={"LTP_0": "Stock_LTP"})

# =====================================
# ± STRIKE FILTER
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

display_df = filter_strikes_around_ltp(df)

# =====================================
# HIGHLIGHTING
# =====================================
def highlight_rows(data):
    styles = pd.DataFrame("", index=data.index, columns=data.columns)

    for stock in data["Stock"].dropna().unique():
        sdf = data[(data["Stock"] == stock) & data["Strike"].notna()].sort_values("Strike")
        ltp = float(sdf["Stock_LTP"].iloc[0])
        strikes = sdf["Strike"].values

        # ATM highlight
        for i in range(len(strikes) - 1):
            if strikes[i] <= ltp <= strikes[i + 1]:
                styles.loc[sdf.index[i]] = "background-color:#003366;color:white"
                styles.loc[sdf.index[i + 1]] = "background-color:#003366;color:white"
                break

        # Max Δ MP highlight
        styles.loc[
            sdf[delta_mp_cols[0]].abs().idxmax()
        ] = "background-color:#8B0000;color:white"

    return styles

# =====================================
# STRIKE FORMATTER
# =====================================
def format_strike(x):
    if pd.isna(x):
        return ""
    if float(x).is_integer():
        return f"{int(x)}"
    return f"{x:.2f}"

# =====================================
# DISPLAY
# =====================================
st.dataframe(
    display_df
    .style
    .apply(highlight_rows, axis=None)
    .format(
        {
            "Strike": format_strike,
            "Stock_LTP": "{:.2f}",
            "Stock_High": "{:.2f}",
            "Stock_Low": "{:.2f}",
            pct_col: "{:.3f}",
            **{c: "{:.0f}" for c in delta_mp_cols},
            "Δ(Δ MP T2-T3)": "{:.0f}",
            **{c: "{:.0f}" for c in [
                "Δ CE OI (T2-T3)", "Δ PE OI (T2-T3)",
                "Δ CE Vol (T2-T3)", "Δ PE Vol (T2-T3)"
            ]},
        },
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
    "max_pain_with_highlight_and_delta_delta.csv",
    "text/csv",
)

import streamlit as st
import pandas as pd
import numpy as np
import os
from datetime import time
from streamlit_autorefresh import st_autorefresh

st_autorefresh(interval=360_000, key="auto_refresh")

# ==================================================
# STREAMLIT CONFIG
# ==================================================
st.set_page_config(page_title="FnO MP Delta Dashboard", layout="wide")
st.title("📊 FnO STOCKS – Max Pain Delta View")

DATA_DIR = "data"

# ==================================================
# LOAD CSV FILES
# ==================================================
def load_csv_files():
    files = []
    for f in os.listdir(DATA_DIR):
        if f.startswith("option_chain_") and f.endswith(".csv"):
            ts = f.replace("option_chain_", "").replace(".csv", "")
            files.append((ts, os.path.join(DATA_DIR, f)))
    return sorted(files, reverse=True)

csv_files = load_csv_files()
if len(csv_files) < 3:
    st.error("Need at least 3 CSV files.")
    st.stop()

timestamps_all = [ts for ts, _ in csv_files]
file_map = dict(csv_files)

# ==================================================
# TIME FILTER (08:00 to 16:30) – USED BY BOTH TABLES
# ==================================================
def extract_time(ts):
    try:
        hh, mm = map(int, ts.split("_")[-1].split("-")[:2])
        return time(hh, mm)
    except:
        return None

START_TIME = time(8, 0)
END_TIME = time(16, 30)

filtered_ts = [
    ts for ts in timestamps_all
    if extract_time(ts)
    and START_TIME <= extract_time(ts) <= END_TIME
]

# ==================================================
# COMMON TIMESTAMP SELECTORS (USED BY BOTH)
# ==================================================
st.subheader("🕒 Timestamp Selection")

cols = st.columns(3)
t1 = cols[0].selectbox("Timestamp 1", filtered_ts, 0)
t2 = cols[1].selectbox("Timestamp 2", filtered_ts, 1)
t3 = cols[2].selectbox("Timestamp 3", filtered_ts, 2)

# ==================================================
# LOAD CSVs ONCE (REUSED)
# ==================================================
df_t1 = pd.read_csv(file_map[t1])
df_t2 = pd.read_csv(file_map[t2])
df_t3 = pd.read_csv(file_map[t3])

# ==================================================
# ================= TABLE 1 ========================
# FnO MP Delta Dashboard (CODE 1)
# ==================================================
st.subheader("📘 Table 1 – FnO MP Delta Dashboard")

dfs = []
for i, d in enumerate([df_t1, df_t2, df_t3]):
    dfs.append(
        d[
            ["Stock", "Strike", "Max_Pain", "Stock_LTP",
             "CE_OI", "PE_OI", "CE_Volume", "PE_Volume"]
        ].rename(columns={
            "Max_Pain": f"MP_{i}",
            "Stock_LTP": f"LTP_{i}",
            "CE_OI": f"CE_OI_{i}",
            "PE_OI": f"PE_OI_{i}",
            "CE_Volume": f"CE_VOL_{i}",
            "PE_Volume": f"PE_VOL_{i}",
        })
    )

df1 = dfs[0]
for i in range(1, 3):
    df1 = df1.merge(dfs[i], on=["Stock", "Strike"], how="inner")

df1 = df1.merge(
    df_t1[["Stock", "Strike", "Stock_%_Change", "Stock_High", "Stock_Low"]],
    on=["Stock", "Strike"],
    how="left"
)

for c in df1.columns:
    if any(x in c for x in ["MP_", "LTP_", "OI_", "VOL_"]):
        df1[c] = pd.to_numeric(df1[c], errors="coerce").fillna(0)

df1["Δ MP TS1-TS2"] = df1["MP_0"] - df1["MP_1"]
df1["Δ MP TS2-TS3"] = df1["MP_1"] - df1["MP_2"]

df1["Δ CE OI TS1-TS2"] = df1["CE_OI_0"] - df1["CE_OI_1"]
df1["Δ PE OI TS1-TS2"] = df1["PE_OI_0"] - df1["PE_OI_1"]
df1["Δ CE Vol TS1-TS2"] = df1["CE_VOL_0"] - df1["CE_VOL_1"]
df1["Δ PE Vol TS1-TS2"] = df1["PE_VOL_0"] - df1["PE_VOL_1"]

df1["% Stock Ch TS1-TS2"] = ((df1["LTP_0"] - df1["LTP_1"]) / df1["LTP_1"]) * 100
df1["% Stock Ch TS2-TS3"] = ((df1["LTP_1"] - df1["LTP_2"]) / df1["LTP_2"]) * 100
df1["Stock_LTP"] = df1["LTP_0"]

df1 = df1[
    [
        "Stock", "Strike",
        "Δ MP TS1-TS2", "Δ MP TS2-TS3",
        "Δ CE OI TS1-TS2", "Δ PE OI TS1-TS2",
        "Δ CE Vol TS1-TS2", "Δ PE Vol TS1-TS2",
        "Stock_LTP", "Stock_%_Change",
        "% Stock Ch TS1-TS2", "% Stock Ch TS2-TS3",
        "Stock_High", "Stock_Low",
    ]
]

def filter_strikes(df, n=6):
    out = []
    for _, g in df.groupby("Stock"):
        g = g.sort_values("Strike").reset_index(drop=True)
        atm = (g["Strike"] - g["Stock_LTP"].iloc[0]).abs().idxmin()
        out.append(g.iloc[max(0, atm-n):atm+n+1])
        out.append(pd.DataFrame([{c: np.nan for c in g.columns}]))
    return pd.concat(out[:-1], ignore_index=True)

display_df1 = filter_strikes(df1)

def highlight_table1(data):
    styles = pd.DataFrame("", index=data.index, columns=data.columns)
    for stock in data["Stock"].dropna().unique():
        sdf = data[(data["Stock"] == stock) & data["Strike"].notna()]
        ltp = sdf["Stock_LTP"].iloc[0]
        strikes = sdf["Strike"].values
        for i in range(len(strikes)-1):
            if strikes[i] <= ltp <= strikes[i+1]:
                styles.loc[sdf.index[i]] = "background-color:#003366;color:white"
                styles.loc[sdf.index[i+1]] = "background-color:#003366;color:white"
                break
        mp_idx = sdf["Δ MP TS1-TS2"].abs().idxmax()
        styles.loc[mp_idx] = "background-color:#8B0000;color:white"
    return styles

fmt1 = {c: "{:.0f}" for c in display_df1.select_dtypes("number").columns}
fmt1.update({
    "Stock_LTP": "{:.2f}",
    "Stock_%_Change": "{:.2f}",
    "% Stock Ch TS1-TS2": "{:.2f}",
    "% Stock Ch TS2-TS3": "{:.2f}",
})

st.dataframe(
    display_df1.style.apply(highlight_table1, axis=None).format(fmt1, na_rep=""),
    use_container_width=True
)

# ==================================================
# ================= TABLE 2 ========================
# ΔΔ Max Pain Viewer (CODE 2)
# ==================================================
st.subheader("📕 Table 2 – ΔΔ Max Pain Viewer")

df_base = df_t1.copy()
df_base["Stock"] = df_base["Stock"].astype(str).str.upper().str.strip()

def short_ts(ts):
    return ts.split("_")[-1].replace("-", ":")

def compute_ddmp(df):
    df = df.copy()
    for ts, d in zip([t2, t3], [df_t2, df_t3]):
        label = short_ts(ts)
        df = df.merge(
            d[["Stock", "Strike", "Max_Pain"]],
            on=["Stock", "Strike"],
            suffixes=("", f"_{label}")
        )
        df[label] = df["Max_Pain"] - df[f"Max_Pain_{label}"]
        df.drop(columns=[f"Max_Pain_{label}"], inplace=True)
    return df

df_all = compute_ddmp(df_base)

rows = []
for stock in df_all["Stock"].unique():
    sdf = df_all[df_all["Stock"] == stock].sort_values("Strike")
    ltp = sdf["Stock_LTP"].iloc[0]
    for _, r in sdf.iterrows():
        rows.append({
            "Stock": stock,
            "Strike": int(r["Strike"]),
            short_ts(t2): int(r[short_ts(t2)]),
            short_ts(t3): int(r[short_ts(t3)]),
            "Stock_LTP": round(ltp, 2),
            "Stock_High": round(r["Stock_High"], 2),
            "Stock_Low": round(r["Stock_Low"], 2),
        })

df2 = pd.DataFrame(rows)

st.dataframe(
    df2.sort_values(["Stock", "Strike"]),
    use_container_width=True
)

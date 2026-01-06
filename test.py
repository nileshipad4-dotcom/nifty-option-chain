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
# TIME FILTER (08:00 to 16:30)
# ==================================================
def extract_time(ts):
    try:
        hh, mm = map(int, ts.split("_")[-1].split("-")[:2])
        return time(hh, mm)
    except:
        return None

filtered_ts = [
    ts for ts in timestamps_all
    if extract_time(ts)
    and time(8, 0) <= extract_time(ts) <= time(16, 30)
]

# ==================================================
# TIMESTAMP SELECTION
# ==================================================
st.subheader("🕒 Timestamp Selection")

c1, c2, c3 = st.columns(3)
t1 = c1.selectbox("Timestamp 1", filtered_ts, 0)
t2 = c2.selectbox("Timestamp 2", filtered_ts, 1)
t3 = c3.selectbox("Timestamp 3", filtered_ts, 2)

# ==================================================
# LOAD CSVs ONCE
# ==================================================
df_t1 = pd.read_csv(file_map[t1])
df_t2 = pd.read_csv(file_map[t2])
df_t3 = pd.read_csv(file_map[t3])

# ==================================================
# ================= TABLE 1 ========================
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
df1 = df1.merge(dfs[1], on=["Stock", "Strike"], how="inner")
df1 = df1.merge(dfs[2], on=["Stock", "Strike"], how="inner")

df1 = df1.merge(
    df_t1[["Stock", "Strike", "Stock_%_Change", "Stock_High", "Stock_Low"]],
    on=["Stock", "Strike"],
    how="left"
)

for col in df1.columns:
    if any(x in col for x in ["MP_", "LTP_", "OI_", "VOL_"]):
        df1[col] = pd.to_numeric(df1[col], errors="coerce").fillna(0)

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
    blocks = []
    for _, g in df.groupby("Stock"):
        g = g.sort_values("Strike").reset_index(drop=True)
        ltp = g["Stock_LTP"].iloc[0]
        atm = (g["Strike"] - ltp).abs().idxmin()
        blocks.append(g.iloc[max(0, atm-n):atm+n+1])
        blocks.append(pd.DataFrame([{c: np.nan for c in g.columns}]))
    return pd.concat(blocks[:-1], ignore_index=True)

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

fmt = {c: "{:.0f}" for c in display_df1.select_dtypes("number").columns}
fmt.update({
    "Stock_LTP": "{:.2f}",
    "Stock_%_Change": "{:.2f}",
    "% Stock Ch TS1-TS2": "{:.2f}",
    "% Stock Ch TS2-TS3": "{:.2f}",
})

st.dataframe(
    display_df1.style.apply(highlight_table1, axis=None).format(fmt, na_rep=""),
    use_container_width=True
)

# ==================================================
# ===== SINGLE STOCK TABLES (FROM TABLE 1) =========
# ==================================================
st.subheader("🔎 Single Stock Views (from Table 1)")

stocks = sorted(display_df1["Stock"].dropna().unique())

a, b = st.columns(2)
stock_a = a.selectbox("Select Stock A", [""] + stocks)
stock_b = b.selectbox("Select Stock B", [""] + stocks)

def show_single_stock(stock_name, label):
    sdf = display_df1[display_df1["Stock"] == stock_name]
    if sdf.empty:
        return
    st.markdown(f"**{label}: {stock_name}**")
    st.dataframe(
        sdf.style.apply(highlight_table1, axis=None).format(fmt, na_rep=""),
        use_container_width=True
    )

if stock_a:
    show_single_stock(stock_a, "Stock A")

if stock_b:
    show_single_stock(stock_b, "Stock B")

# ==================================================
# ================= TABLE 2 ========================
# ==================================================
st.subheader("📕 Table 2 – ΔΔ Max Pain Viewer")

p1, p2 = st.columns(2)
with p1:
    ltp_pct_limit = st.number_input("Max % distance from LTP (Table 2)", 0.0, 50.0, 5.0, 0.5)
with p2:
    ddmp_diff_limit = st.number_input("Min |Δ MP(T2 − T3)| (Table 2)", 0.0, value=147.0, step=10.0)

def short_ts(ts):
    return ts.split("_")[-1].replace("-", ":")

df_base = df_t1.copy()
df_base["Stock"] = df_base["Stock"].astype(str).str.upper().str.strip()

df_all = df_base.merge(
    df_t2[["Stock", "Strike", "Max_Pain"]],
    on=["Stock", "Strike"],
    suffixes=("", "_T2"),
)

df_all = df_all.merge(
    df_t3[["Stock", "Strike", "Max_Pain"]],
    on=["Stock", "Strike"],
    suffixes=("", "_T3"),
)

df_all[short_ts(t2)] = df_all["Max_Pain"] - df_all["Max_Pain_T2"]
df_all[short_ts(t3)] = df_all["Max_Pain_T2"] - df_all["Max_Pain_T3"]

atm_map = {}
mp_map = {}

for stock in df_all["Stock"].unique():
    sdf = df_all[df_all["Stock"] == stock].sort_values("Strike")
    ltp = sdf["Stock_LTP"].iloc[0]
    strikes = sdf["Strike"].values

    for i in range(len(strikes)-1):
        if strikes[i] <= ltp <= strikes[i+1]:
            atm_map[stock] = {strikes[i], strikes[i+1]}
            break

    mp_map[stock] = sdf.loc[sdf["Max_Pain"].idxmin(), "Strike"]

rows = []

for stock in df_all["Stock"].unique():
    sdf = df_all[df_all["Stock"] == stock].sort_values("Strike")
    ltp = float(sdf["Stock_LTP"].iloc[0])
    if ltp <= 0:
        continue

    ltp2 = float(df_t2[df_t2["Stock"] == stock]["Stock_LTP"].iloc[0])
    ltp3 = float(df_t3[df_t3["Stock"] == stock]["Stock_LTP"].iloc[0])

    pct_12 = (ltp - ltp2) / ltp2 * 100
    pct_23 = (ltp2 - ltp3) / ltp3 * 100 if ltp2 != 0 else np.nan

    high = float(sdf["Stock_High"].iloc[0])
    low = float(sdf["Stock_Low"].iloc[0])

    for _, r in sdf.iterrows():
        v1 = r[short_ts(t2)]
        v2 = r[short_ts(t3)]

        if abs(v2 - v1) <= ddmp_diff_limit:
            continue

        strike = float(r["Strike"])
        if abs(strike - ltp) / ltp * 100 > ltp_pct_limit:
            continue

        rows.append({
            "Stock": stock,
            "Strike": int(strike),
            short_ts(t2): int(v1),
            short_ts(t3): int(v2),
            "%Δ LTP TS1→TS2": round(pct_12, 2),
            "%Δ LTP TS2→TS3": round(pct_23, 2),
            "Stock_LTP": round(ltp, 2),
            "Stock_High": round(high, 2),
            "Stock_Low": round(low, 2),
        })

df2 = pd.DataFrame(rows)

def color_table2(row):
    stock = row["Stock"]
    strike = row["Strike"]
    high = row["Stock_High"]
    low = row["Stock_Low"]

    if strike == mp_map.get(stock):
        base = "background-color:#4E342E;color:white"
    elif strike in atm_map.get(stock, set()):
        base = "background-color:#003366;color:white"
    elif strike > row["Stock_LTP"]:
        base = "background-color:#004d00;color:white"
    else:
        base = "background-color:#660000;color:white"

    styles = []
    for col in row.index:
        if col in ("Stock_High", "Stock_Low") and low <= strike <= high:
            styles.append("")
        else:
            styles.append(base)
    return styles

if not df2.empty:
    st.dataframe(
        df2.sort_values(["Stock", "Strike"])
        .style
        .apply(color_table2, axis=1)
        .format({
            "Strike": "{:.0f}",
            short_ts(t2): "{:.0f}",
            short_ts(t3): "{:.0f}",
            "%Δ LTP TS1→TS2": "{:.2f}",
            "%Δ LTP TS2→TS3": "{:.2f}",
            "Stock_LTP": "{:.2f}",
            "Stock_High": "{:.2f}",
            "Stock_Low": "{:.2f}",
        }),
        use_container_width=True
    )
else:
    st.info("No rows matched Table-2 filter criteria.")

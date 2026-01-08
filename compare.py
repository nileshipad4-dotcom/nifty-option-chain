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
    if extract_time(ts) and time(8, 0) <= extract_time(ts) <= time(16, 30)
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
        d[[
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
    )

df1 = dfs[0].merge(dfs[1], on=["Stock", "Strike"]).merge(dfs[2], on=["Stock", "Strike"])

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
# ---- ATM PAIR (BELOW + ABOVE LTP) PE–CE DIFFERENCE ----

df1["Δ (PE-CE) OI TS1-TS2"] = np.nan
df1["Δ (PE-CE) Vol TS1-TS2"] = np.nan

for stock, g in df1.groupby("Stock"):
    g = g.sort_values("Strike")
    ltp = g["LTP_0"].iloc[0]

    below_candidates = g[g["Strike"] <= ltp]
    above_candidates = g[g["Strike"] > ltp]
    
    if below_candidates.empty or above_candidates.empty:
        continue   # ❗ skip this stock safely
    
    below = below_candidates.iloc[-1]
    above = above_candidates.iloc[0]



    pe_oi_sum = (
        below["Δ PE OI TS1-TS2"] +
        above["Δ PE OI TS1-TS2"]
    )
    ce_oi_sum = (
        below["Δ CE OI TS1-TS2"] +
        above["Δ CE OI TS1-TS2"]
    )

    pe_vol_sum = (
        below["Δ PE Vol TS1-TS2"] +
        above["Δ PE Vol TS1-TS2"]
    )
    ce_vol_sum = (
        below["Δ CE Vol TS1-TS2"] +
        above["Δ CE Vol TS1-TS2"]
    )

    df1.loc[g.index, "Δ (PE-CE) OI TS1-TS2"] = pe_oi_sum - ce_oi_sum
    df1.loc[g.index, "Δ (PE-CE) Vol TS1-TS2"] = pe_vol_sum - ce_vol_sum



df1["% Stock Ch TS1-TS2"] = ((df1["LTP_0"] - df1["LTP_1"]) / df1["LTP_1"]) * 100
df1["% Stock Ch TS2-TS3"] = ((df1["LTP_1"] - df1["LTP_2"]) / df1["LTP_2"]) * 100
df1["Stock_LTP"] = df1["LTP_0"]

# ---- TS3 COLUMNS MOVED TO END ----
df1 = df1[[
    "Stock", 
    "Strike",
    "Δ MP TS1-TS2",
    "Δ CE OI TS1-TS2", 
    "Δ PE OI TS1-TS2",
    "Δ CE Vol TS1-TS2", 
    "Δ PE Vol TS1-TS2",
    "Δ (PE-CE) OI TS1-TS2",
    "Δ (PE-CE) Vol TS1-TS2",
    "% Stock Ch TS1-TS2",
    "Stock_LTP", 
    "Stock_%_Change", 
    "Stock_High", 
    "Stock_Low",
    "Δ MP TS2-TS3", 
    "% Stock Ch TS2-TS3",
]]

# ---- RENAME TABLE 1 COLUMNS (DISPLAY FRIENDLY) ----
df1 = df1.rename(columns={
    "Δ MP TS1-TS2": "Δ MP",
    "Δ CE OI TS1-TS2": "Δ CE OI",
    "Δ PE OI TS1-TS2": "Δ PE OI",
    "Δ CE Vol TS1-TS2": "Δ CE Vol",
    "Δ PE Vol TS1-TS2": "Δ PE Vol",
    "Δ (PE-CE) OI TS1-TS2": "Δ (PE-CE) OI",
    "Δ (PE-CE) Vol TS1-TS2": "Δ (PE-CE) Vol",

})



def filter_strikes(df, n=4):
    blocks = []
    for _, g in df.groupby("Stock"):
        g = g.sort_values("Strike").reset_index(drop=True)
        atm = (g["Strike"] - g["Stock_LTP"].iloc[0]).abs().idxmin()
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
        styles.loc[sdf["Δ MP TS1-TS2"].abs().idxmax()] = "background-color:#8B0000;color:white"
    return styles

fmt = {c: "{:.0f}" for c in display_df1.select_dtypes("number").columns}
fmt.update({
    "Stock_LTP": "{:.2f}",
    "Stock_%_Change": "{:.2f}",
    "% Stock Ch TS1-TS2": "{:.2f}",
    "% Stock Ch TS2-TS3": "{:.2f}",
})

st.dataframe(display_df1.style.apply(highlight_table1, axis=None).format(fmt, na_rep=""),
             use_container_width=True)



# ==================================================
# FILTERED DOWNTREND
# ==================================================

st.subheader("📉 DOWNTREND Filters")

dc1, dc2 = st.columns(2)

with dc1:
    ltp_strike_dist_pct = st.number_input(
        "Max % distance of BELOW strike from LTP",
        min_value=0.1,
        max_value=5.0,
        value=0.6,
        step=0.1
    )

with dc2:
    stock_chg_ts23_limit = st.number_input(
        "% Stock Change TS2→TS3 threshold",
        min_value=0.1,
        max_value=5.0,
        value=0.7,
        step=0.1
    )

# ==================================================
# FILTERED DOWNTREND
# ==================================================
def get_ltp_strikes(sdf):
    sdf = sdf.sort_values("Strike").reset_index(drop=True)
    ltp = sdf["Stock_LTP"].iloc[0]

    below = sdf[sdf["Strike"] <= ltp].iloc[-1]
    above = sdf[sdf["Strike"] > ltp].iloc[0]

    idx = sdf.index[sdf["Strike"] == below["Strike"]][0]
    window = sdf.iloc[max(0, idx-2): idx+4]   # 6 strikes window

    return below, above, window


# DOWNTREND LOGIC
def is_downtrend_stock(sdf):
    below, above, window = get_ltp_strikes(sdf)

    ltp = sdf["Stock_LTP"].iloc[0]

    # ---- BELOW STRIKE DISTANCE CHECK (COMMON FOR ALL CONDITIONS)
    below_dist_ok = (
        abs((below["Strike"] - ltp) / ltp) * 100
        >= ltp_strike_dist_pct
    )

    if not below_dist_ok:
        return False

    # ---------- CONDITION 1 ----------
    cond1 = (
        above["Δ CE OI TS1-TS2"] > above["Δ PE OI TS1-TS2"] and
        above["Δ CE Vol TS1-TS2"] > above["Δ PE Vol TS1-TS2"] and
        below["Δ CE OI TS1-TS2"] > below["Δ PE OI TS1-TS2"] and
        below["Δ CE Vol TS1-TS2"] > below["Δ PE Vol TS1-TS2"]
    )

    # ---------- CONDITION 2 ----------
    pe_neg = (window["Δ PE OI TS1-TS2"] < 0).sum() >= 4
    ce_pos = (window["Δ CE OI TS1-TS2"] > 0).sum() >= 4
    oi_above = above["Δ CE OI TS1-TS2"] > above["Δ PE OI TS1-TS2"]
    oi_below = below["Δ CE OI TS1-TS2"] > below["Δ PE OI TS1-TS2"]

    cond2 = (
        pe_neg and ce_pos and  oi_above and oi_below and
        above["Δ CE Vol TS1-TS2"] > above["Δ PE Vol TS1-TS2"]
    )

    # ---------- CONDITION 3 ----------
    cond3 = (
        sdf["% Stock Ch TS2-TS3"].iloc[0] > stock_chg_ts23_limit and
        above["Δ CE Vol TS1-TS2"] > above["Δ PE Vol TS1-TS2"] and
        above["Δ CE OI TS1-TS2"] > above["Δ PE OI TS1-TS2"] and
        below["Δ CE OI TS1-TS2"] > 0
    )

    return cond1 or cond2 or cond3









# DOWNTREND DATAFRAME
downtrend_blocks = []

for stock in display_df1["Stock"].dropna().unique():
    sdf = display_df1[display_df1["Stock"] == stock].dropna(subset=["Strike"])
    if len(sdf) < 6:
        continue

    try:
        if is_downtrend_stock(sdf):
            downtrend_blocks.append(sdf)
            downtrend_blocks.append(
                pd.DataFrame([{c: np.nan for c in sdf.columns}])
            )
    except:
        pass

df_downtrend = (
    pd.concat(downtrend_blocks[:-1], ignore_index=True)
    if downtrend_blocks else pd.DataFrame()
)

# DOWNTREND DISPLAY
st.subheader("📉 DOWNTREND – CE Dominance & Price Confirmation")

if not df_downtrend.empty:
    st.dataframe(
        df_downtrend
        .style
        .apply(highlight_table1, axis=None)
        .format(fmt, na_rep=""),
        use_container_width=True
    )
else:
    st.info("No stocks matched DOWNTREND conditions.")










# ==================================================
# UPTREND
# ==================================================

st.subheader("📈 UPTREND Filters")

uc1, uc2 = st.columns(2)

with uc1:
    stock_chg_ts23_down = st.number_input(
        "% Stock Change TS2→TS3 (UPTREND, negative)",
        min_value=0.1,
        max_value=5.0,
        value=0.7,
        step=0.1
    )

with uc2:
    above_strike_dist_pct = st.number_input(
        "Min % distance of ABOVE strike from LTP",
        min_value=0.1,
        max_value=5.0,
        value=0.7,
        step=0.1
    )

def is_uptrend_stock(sdf):
    below, above, window = get_ltp_strikes(sdf)
    ltp = sdf["Stock_LTP"].iloc[0]

    # ---- COMMON CONDITIONS ----
    above_dist_ok = (
        abs((above["Strike"] - ltp) / ltp) * 100
        >= above_strike_dist_pct
    )

    pe_oi_common = (
        below["Δ PE OI TS1-TS2"] > 0 and
        above["Δ PE OI TS1-TS2"] > 0
    )

    if not (above_dist_ok and pe_oi_common):
        return False

    # ---------- CONDITION 1 ----------
    cond1 = (
        above["Δ CE OI TS1-TS2"] < above["Δ PE OI TS1-TS2"] and
        above["Δ CE Vol TS1-TS2"] < above["Δ PE Vol TS1-TS2"] and
        below["Δ CE OI TS1-TS2"] < below["Δ PE OI TS1-TS2"] and
        below["Δ CE Vol TS1-TS2"] < below["Δ PE Vol TS1-TS2"]
    )

    # ---------- CONDITION 2 ----------
    pe_pos = (window["Δ PE OI TS1-TS2"] > 0).sum() >= 4
    ce_neg = (window["Δ CE OI TS1-TS2"] < 0).sum() >= 4

    cond2 = (
        pe_pos and ce_neg and
        below["Δ CE Vol TS1-TS2"] < below["Δ PE Vol TS1-TS2"]
    )

    # ---------- CONDITION 3 ----------
    cond3 = (
        sdf["% Stock Ch TS2-TS3"].iloc[0] < -stock_chg_ts23_down and
        below["Δ CE Vol TS1-TS2"] < below["Δ PE Vol TS1-TS2"] and
        below["Δ CE OI TS1-TS2"] < below["Δ PE OI TS1-TS2"] and
        above["Δ PE OI TS1-TS2"] > 0
    )

    return cond1 or cond2 or cond3

uptrend_blocks = []

for stock in display_df1["Stock"].dropna().unique():
    sdf = display_df1[display_df1["Stock"] == stock].dropna(subset=["Strike"])
    if len(sdf) < 6:
        continue

    try:
        if is_uptrend_stock(sdf):
            uptrend_blocks.append(sdf)
            uptrend_blocks.append(
                pd.DataFrame([{c: np.nan for c in sdf.columns}])
            )
    except:
        pass

df_uptrend = (
    pd.concat(uptrend_blocks[:-1], ignore_index=True)
    if uptrend_blocks else pd.DataFrame()
)

st.subheader("📈 UPTREND – PE Dominance & Price Weakness")

if not df_uptrend.empty:
    st.dataframe(
        df_uptrend
        .style
        .apply(highlight_table1, axis=None)
        .format(fmt, na_rep=""),
        use_container_width=True
    )
else:
    st.info("No stocks matched UPTREND conditions.")








# ==================================================
# SINGLE STOCK TABLES (A / B / C)
# ==================================================
st.subheader("🔎 Selected Stocks")

stocks = sorted(display_df1["Stock"].dropna().unique())
a, b, c = st.columns(3)

stock_a = a.selectbox("Stock A", [""] + stocks)
stock_b = b.selectbox("Stock B", [""] + stocks)
stock_c = c.selectbox("Stock C", [""] + stocks)

def show_stock(s, label):
    if s:
        sdf = display_df1[display_df1["Stock"] == s]
        st.markdown(f"**{label}: {s}**")
        st.dataframe(sdf.style.apply(highlight_table1, axis=None).format(fmt, na_rep=""),
                     use_container_width=True)

show_stock(stock_a, "A")
show_stock(stock_b, "B")
show_stock(stock_c, "C")

# ==================================================
# ================= TABLE 2 ========================
# ==================================================
st.subheader("📕 Table 2 – ΔΔ Max Pain Viewer")

p1, p2 = st.columns(2)
with p1:
    ltp_pct_limit = st.number_input(
        "Max % distance from LTP (Table 2)", 0.0, 50.0, 5.0, 0.5
    )
with p2:
    ddmp_diff_limit = st.number_input(
        "Min |Δ MP(T2 − T3)| (Table 2)", 0.0, value=347.0, step=10.0
    )

def short_ts(ts):
    return ts.split("_")[-1].replace("-", ":")

# --------------------------------------------------
# BUILD BASE DF
# --------------------------------------------------
df_all = df_t1.merge(
    df_t2[["Stock", "Strike", "Max_Pain", "Stock_LTP"]],
    on=["Stock", "Strike"],
    suffixes=("", "_T2"),
)

df_all = df_all.merge(
    df_t3[["Stock", "Strike", "Max_Pain", "Stock_LTP"]],
    on=["Stock", "Strike"],
    suffixes=("", "_T3"),
)

df_all[short_ts(t2)] = df_all["Max_Pain"] - df_all["Max_Pain_T2"]
df_all[short_ts(t3)] = df_all["Max_Pain_T2"] - df_all["Max_Pain_T3"]

# --------------------------------------------------
# PRE-COMPUTE ATM & MAX PAIN
# --------------------------------------------------
atm_map = {}
mp_map = {}

for stock in df_all["Stock"].unique():
    sdf = df_all[df_all["Stock"] == stock].sort_values("Strike")
    ltp = sdf["Stock_LTP"].iloc[0]
    strikes = sdf["Strike"].values

    for i in range(len(strikes) - 1):
        if strikes[i] <= ltp <= strikes[i + 1]:
            atm_map[stock] = {strikes[i], strikes[i + 1]}
            break

    mp_map[stock] = sdf.loc[sdf["Max_Pain"].idxmin(), "Strike"]

# --------------------------------------------------
# FILTERED ROWS
# --------------------------------------------------
rows = []

for stock in df_all["Stock"].unique():
    sdf = df_all[df_all["Stock"] == stock].sort_values("Strike")

    ltp1 = pd.to_numeric(sdf["Stock_LTP"].iloc[0], errors="coerce")
    ltp2 = pd.to_numeric(sdf["Stock_LTP_T2"].iloc[0], errors="coerce")
    
    if pd.isna(ltp1) or ltp1 <= 0 or pd.isna(ltp2):
        continue


    pct_ltp_12 = ((ltp1 - ltp2) / ltp2 * 100) if ltp2 != 0 else np.nan

    high = float(sdf["Stock_High"].iloc[0])
    low = float(sdf["Stock_Low"].iloc[0])

    for _, r in sdf.iterrows():
        v1 = r[short_ts(t2)]
        v2 = r[short_ts(t3)]

        if abs(v2 - v1) <= ddmp_diff_limit:
            continue

        strike = float(r["Strike"])
        if abs(strike - ltp1) / ltp1 * 100 > ltp_pct_limit:
            continue

        rows.append({
            "Stock": stock,
            "Strike": int(strike),
            short_ts(t2): int(v1),
            "%Δ LTP TS1→TS2": round(pct_ltp_12, 2),
            "Stock_LTP": round(ltp1, 2),
            "Stock_High": round(high, 2),
            "Stock_Low": round(low, 2),

            # ---- TS3 AT END ----
            short_ts(t3): int(v2),
        })

df2 = pd.DataFrame(rows)

# --------------------------------------------------
# HIGHLIGHTING (RESTORED CORRECT LOGIC)
# --------------------------------------------------
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
            styles.append("")     # ✅ NO highlight between High–Low
        else:
            styles.append(base)
    return styles

# --------------------------------------------------
# DISPLAY TABLE 2
# --------------------------------------------------
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
            "Stock_LTP": "{:.2f}",
            "Stock_High": "{:.2f}",
            "Stock_Low": "{:.2f}",
        }),
        use_container_width=True
    )
else:
    st.info("No rows matched Table-2 filter criteria.")

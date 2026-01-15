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
# LOAD FnO LOT SIZE
# ==================================================
fno_df = pd.read_csv("FnO.csv")[["Stock", "Lot_Size"]]


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



# ==================================================
# PE / CE VOL RATIO (ATM WINDOW)
# ==================================================

df1["PE/CE Vol Ratio"] = np.nan
df1["PE/CE OI Ratio"] = np.nan


for stock, g in df1.groupby("Stock"):
    g = g.sort_values("Strike").reset_index()

    ltp = g["LTP_0"].iloc[0]

    # ATM strike index
    atm_idx = (g["Strike"] - ltp).abs().idxmin()

    # Define windows safely
    pe_idx = g.loc[
        max(0, atm_idx-2) : min(len(g)-1, atm_idx+1)
    ].index

    ce_idx = g.loc[
        atm_idx : min(len(g)-1, atm_idx+3)
    ].index

    pe_sum = g.loc[pe_idx, "Δ PE Vol TS1-TS2"].sum()
    ce_sum = g.loc[ce_idx, "Δ CE Vol TS1-TS2"].sum()

    ratio = pe_sum / ce_sum if ce_sum != 0 else np.nan

        # ---- OI RATIO (NEW) ----
    pe_oi_sum = g.loc[pe_idx, "Δ PE OI TS1-TS2"].sum()
    ce_oi_sum = g.loc[ce_idx, "Δ CE OI TS1-TS2"].sum()

    oi_ratio = pe_oi_sum / ce_oi_sum if ce_oi_sum != 0 else np.nan

    df1.loc[g["index"], "PE/CE Vol Ratio"] = round(ratio, 2)
    df1.loc[g["index"], "PE/CE OI Ratio"] = round(oi_ratio, 2)

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
    "PE/CE OI Ratio",
    "PE/CE Vol Ratio",
    "% Stock Ch TS1-TS2",
    "% Stock Ch TS2-TS3",
    "Stock_LTP", 
    "Stock_%_Change", 
    "Stock_High", 
    "Stock_Low",
    "Δ MP TS2-TS3", 

]]

# ---- RENAME DELTA COLUMNS (DISPLAY ONLY) ----
df1 = df1.rename(columns={
    "Δ MP TS1-TS2": "Δ MP",
    "Δ CE OI TS1-TS2": "Δ CE OI",
    "Δ PE OI TS1-TS2": "Δ PE OI",
    "Δ CE Vol TS1-TS2": "Δ CE Vol",
    "Δ PE Vol TS1-TS2": "Δ PE Vol",
    "PE/CE Vol Ratio": "Δ PE/CE Vol",
     "PE/CE OI Ratio": "Δ PE/CE OI",
    "% Stock Ch TS1-TS2": "% Ch 1-2",
    "% Stock Ch TS2-TS3": "% Ch 2-3",
    "Stock_%_Change": "% Ch",
    "Δ (PE-CE) OI TS1-TS2": "Δ (PE-CE) OI",
    "Δ (PE-CE) Vol TS1-TS2": "Δ (PE-CE) Vol",
})


def filter_strikes(df, n=5):
    blocks = []
    for _, g in df.groupby("Stock"):
        g = g.sort_values("Strike").reset_index(drop=True)
        atm = (g["Strike"] - g["Stock_LTP"].iloc[0]).abs().idxmin()
        blocks.append(g.iloc[max(0, atm-n):atm+n])
    return pd.concat(blocks[:-1], ignore_index=True)

display_df1 = filter_strikes(df1)


# ==================================================
# OI WEIGHTED CALCULATIONS (VISIBLE STRIKES ONLY)
# ==================================================

# Ensure numeric
display_df1["Strike"] = pd.to_numeric(display_df1["Strike"], errors="coerce")
display_df1["Δ CE OI"] = pd.to_numeric(display_df1["Δ CE OI"], errors="coerce")
display_df1["Δ PE OI"] = pd.to_numeric(display_df1["Δ PE OI"], errors="coerce")

# Per-strike weighted values
display_df1["CE_x_Strike"] = display_df1["Δ CE OI"] * display_df1["Strike"]
display_df1["PE_x_Strike"] = display_df1["Δ PE OI"] * display_df1["Strike"]

# Stock-level sums (ONLY visible strikes)
sum_df = display_df1.groupby("Stock", as_index=False).agg({
    "CE_x_Strike": "sum",
    "PE_x_Strike": "sum"
})

sum_df["Sum CE"] = sum_df["CE_x_Strike"]
sum_df["Sum PE"] = sum_df["PE_x_Strike"]
sum_df["PE-CE"] = sum_df["Sum PE"] - sum_df["Sum CE"]

sum_df = sum_df[["Stock", "Sum CE", "Sum PE", "PE-CE"]]

# Merge back
display_df1 = display_df1.merge(sum_df, on="Stock", how="left")

# Merge Lot Size
display_df1 = display_df1.merge(fno_df, on="Stock", how="left")
# Lot Size × (PE-CE)
display_df1["Lot_PE-CE"] = display_df1["Lot_Size"] * display_df1["PE-CE"]

# ==================================================
# NEW OI WEIGHTED SUMMARY TABLE
# ==================================================

oi_table = display_df1[[
    "Stock",
    "Strike",
    "Δ CE OI",
    "Δ PE OI",
    "% Ch 1-2",
    "% Ch 2-3",
    "CE_x_Strike",
    "PE_x_Strike",
    "Sum CE",
    "Sum PE",
    "PE-CE",
    "Lot_PE-CE" 
]].copy()


def highlight_table1(data):
    styles = pd.DataFrame("", index=data.index, columns=data.columns)

    # ✅ updated column name
    required_cols = {"Stock", "Strike", "Stock_LTP", "Δ MP"}
    if not required_cols.issubset(data.columns):
        return styles

    for stock in data["Stock"].dropna().unique():
        sdf = data[(data["Stock"] == stock) & data["Strike"].notna()]

        if sdf.empty:
            continue

        ltp = sdf["Stock_LTP"].iloc[0]
        strikes = sdf["Strike"].values

        # 🔵 ATM pair highlight (below + above LTP)
        for i in range(len(strikes) - 1):
            if strikes[i] <= ltp <= strikes[i + 1]:
                styles.loc[sdf.index[i]] = "background-color:#003366;color:white"
                styles.loc[sdf.index[i + 1]] = "background-color:#003366;color:white"
                break

        # 🔴 Max Δ MP highlight
        idx = sdf["Δ MP"].abs().idxmax()
        styles.loc[idx] = "background-color:#8B0000;color:white"

    return styles


fmt = {
    c: "{:.0f}"
    for c in display_df1.select_dtypes("number").columns
    if c != "Δ PE/CE Vol"
}

fmt.update({
    "Stock_LTP": "{:.2f}",
    "% Ch": "{:.2f}",        # Stock_%_Change
    "% Ch 1-2": "{:.2f}",    # TS1 → TS2
    "% Ch 2-3": "{:.2f}",    # TS2 → TS3
    "Δ PE/CE OI": "{:.2f}",
    "Δ PE/CE Vol": "{:.2f}",   # ✅ CORRECT NAME
})

# ==================================================
# RATIO COUNT CONTROL (NON-FILTERING)
# ==================================================

st.subheader("📊 PE/CE Ratio – Count")

rc1, rc2, rc3, rc4 = st.columns(4)

with rc1:
    vol_operator = st.radio(
        "Vol Ratio Condition",
        [">=", "<="],
        index=0,
        horizontal=True
    )


with rc2:
    vol_threshold = st.number_input(
        "Vol Ratio Value",
        min_value=0.1,
        max_value=10.0,
        value=1.0,
        step=0.1
    )



# ---- COUNT (STOCK-LEVEL, NOT STRIKE-LEVEL) ----
vol_ratio_df = (
    df1.groupby("Stock")["Δ PE/CE Vol"]
    .first()
    .dropna()
)

if vol_operator == ">=":
    vol_count = (vol_ratio_df >= vol_threshold).sum()
else:
    vol_count = (vol_ratio_df <= vol_threshold).sum()

# ---- PCR COUNTS ----

pcr_df = df1.groupby("Stock").first()

# Current PCR (TS1 → TS2)
pcr_now_count = (
    (pcr_df["Δ PE OI"] > pcr_df["Δ CE OI"])
    .sum()
)







mc1, mc2 = st.columns(2)

mc1.metric(
    label=f"Stocks with PE/CE Vol Ratio {vol_operator} {vol_threshold}",
    value=int(vol_count)
)

mc2.metric(
    label="PCR > 1",
    value=int(pcr_now_count)
)


st.dataframe(display_df1.style.apply(highlight_table1, axis=None).format(fmt, na_rep=""),
             use_container_width=True)




# ==================================================
# FILTERED UPTREND VOLUME BLAST
# ==================================================
st.subheader("⚡ PE Vol Expansion Filter (Above LTP)")

x_mult = st.number_input(
    "X Multiplier (PE Vol must be > X × surrounding vols)",
    min_value=0.5,
    max_value=10.0,
    value=1.0,
    step=0.1
)

extra_pe_check = st.toggle(
    "Also require PE Vol at 2nd strike ABOVE LTP to dominate CE Vol window",
    value=False
)

# FILTERED UPTREND VOLUME BLAST LOGIC
def pe_vol_expansion_filter(df, x=1.0, extra_check=False):
    blocks = []

    for stock, sdf in df.groupby("Stock"):
        sdf = sdf.sort_values("Strike").reset_index(drop=True)
        ltp = sdf["Stock_LTP"].iloc[0]

        below = sdf[sdf["Strike"] <= ltp]
        above = sdf[sdf["Strike"] > ltp]

        if below.empty or len(above) < 2:
            continue

        below_idx = below.index[-1]
        above_idx = above.index[0]        # strike just above LTP
        above2_idx = above.index[1]      # 2nd strike above LTP

        pe_above = sdf.loc[above_idx, "Δ PE Vol"]
        pe_above2 = sdf.loc[above2_idx, "Δ PE Vol"]

        # 3 strikes below (PE Vol)
        pe_below_vals = sdf.loc[
            max(0, below_idx-3):below_idx,
            "Δ PE Vol"
        ].values

        # CE Vol window: 3 below + above + above2
        ce_window = sdf.loc[
            max(0, below_idx-3):min(len(sdf)-1, above2_idx),
            "Δ CE Vol"
        ].values

        pe_below_vals = pe_below_vals[~np.isnan(pe_below_vals)]
        ce_window = ce_window[~np.isnan(ce_window)]

        if len(pe_below_vals) == 0 or len(ce_window) == 0:
            continue

        # ORIGINAL CONDITIONS
        pe_condition = all(pe_above > x * v for v in pe_below_vals)
        ce_condition = all(pe_above > x * v for v in ce_window)

        # OPTIONAL EXTRA CHECK (2nd ABOVE STRIKE)
        extra_condition = True
        if extra_check:
            extra_condition = all(pe_above2 > v for v in ce_window)

        if pe_condition and ce_condition and extra_condition:
            blocks.append(sdf)
            blocks.append(pd.DataFrame([{c: np.nan for c in sdf.columns}]))

    if blocks:
        return pd.concat(blocks[:-1], ignore_index=True)
    else:
        return pd.DataFrame()

# FILTERED UPTREND VOLUME BLAST DISPLAY
df_pe_expansion = pe_vol_expansion_filter(display_df1, x_mult, extra_pe_check)


st.subheader("🚀 PE Volume Expansion (Above LTP)")

if not df_pe_expansion.empty:
    st.dataframe(
        df_pe_expansion
        .style
        .apply(highlight_table1, axis=None)
        .format(fmt, na_rep=""),
        use_container_width=True
    )
else:
    st.info("No stocks matched the PE Vol Expansion condition.")





# ==================================================
# FILTERED DOWNTREND VOLUME BLAST (CE BELOW LTP)
# ==================================================
st.subheader("⚡ CE Vol Expansion Filter (Below LTP)")

x_mult_ce = st.number_input(
    "X Multiplier (CE Vol must be > X × surrounding vols)",
    min_value=0.5,
    max_value=10.0,
    value=1.0,
    step=0.1,
    key="ce_x"
)

extra_ce_check = st.toggle(
    "Also require CE Vol at 2nd strike BELOW LTP to dominate PE Vol window",
    value=False,
    key="ce_toggle"
)

# FILTERED DOWNTREND VOLUME BLAST (CE BELOW LTP) LOGIC
def ce_vol_expansion_filter(df, x=1.0, extra_check=False):
    blocks = []

    for stock, sdf in df.groupby("Stock"):
        sdf = sdf.sort_values("Strike").reset_index(drop=True)
        ltp = sdf["Stock_LTP"].iloc[0]

        below = sdf[sdf["Strike"] <= ltp]
        above = sdf[sdf["Strike"] > ltp]

        # Need at least 2 below + 1 above
        if len(below) < 2 or len(above) < 1:
            continue

        below_idx = below.index[-1]      # just BELOW LTP (e.g. 34)
        below2_idx = below.index[-2]    # 2nd BELOW (e.g. 33)

        ce_below = sdf.loc[below_idx, "Δ CE Vol"]
        ce_below2 = sdf.loc[below2_idx, "Δ CE Vol"]

        # ---- SAFE WINDOWS ----

        # CE Vol: up to 4 strikes ABOVE (or fewer if not available)
        ce_above_vals = sdf.loc[
            above.index[:min(4, len(above))],
            "Δ CE Vol"
        ].dropna().values

        # PE Vol: 3 below + up to 3 above
        pe_start = max(0, below2_idx - 1)
        pe_end = min(len(sdf) - 1, above.index[min(2, len(above)-1)])

        pe_window = sdf.loc[
            pe_start:pe_end,
            "Δ PE Vol"
        ].dropna().values

        if len(ce_above_vals) == 0 or len(pe_window) == 0:
            continue

        # ---- BASE CONDITIONS ----
        ce_condition = all(ce_below > x * v for v in ce_above_vals)
        pe_condition = all(ce_below > x * v for v in pe_window)

        # ---- OPTIONAL EXTRA CHECK (2nd BELOW STRIKE) ----
        extra_condition = True
        if extra_check:
            extra_condition = all(ce_below2 > v for v in pe_window)

        if ce_condition and pe_condition and extra_condition:
            blocks.append(sdf)
            blocks.append(pd.DataFrame([{c: np.nan for c in sdf.columns}]))

    if blocks:
        return pd.concat(blocks[:-1], ignore_index=True)
    else:
        return pd.DataFrame()


# FILTERED DOWNTREND VOLUME BLAST (CE BELOW LTP) DISPLAY
df_ce_expansion = ce_vol_expansion_filter(display_df1, x_mult_ce, extra_ce_check)

st.subheader("📉 CE Volume Expansion (Below LTP)")

if not df_ce_expansion.empty:
    st.dataframe(
        df_ce_expansion
        .style
        .apply(highlight_table1, axis=None)
        .format(fmt, na_rep=""),
        use_container_width=True
    )
else:
    st.info("No stocks matched the CE Vol Expansion condition.")



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
# OI WEIGHTED STRIKE SUMMARY TABLE
# ==================================================
st.subheader("📊 OI Weighted Strike Summary")

st.dataframe(
    oi_table.style.format({
        "Strike": "{:.2f}",
        "Δ CE OI": "{:.0f}",
        "Δ PE OI": "{:.0f}",
        "CE_x_Strike": "{:.0f}",
        "PE_x_Strike": "{:.0f}",
        "Sum CE": "{:.0f}",
        "Sum PE": "{:.0f}",
        "PE-CE": "{:.0f}",
        "Lot_PE-CE": "{:.0f}",
        "% Ch 1-2": "{:.2f}",
        "% Ch 2-3": "{:.2f}",
    }),
    use_container_width=True
)


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

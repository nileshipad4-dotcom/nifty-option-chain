import streamlit as st
import pandas as pd
import numpy as np
import os
from datetime import time
from streamlit_autorefresh import st_autorefresh

st_autorefresh(interval=3600_000, key="auto_refresh")

# ==================================================
# CONFIG
# ==================================================
st.set_page_config(page_title="OI Weighted Table", layout="wide")
st.title("📊 OI Weighted Strike Table")

DATA_DIR = "data"

# ==================================================
# LOAD FILES
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
# TIME FILTER (08:00–16:00)
# ==================================================
def extract_time(ts):
    try:
        hh, mm = map(int, ts.split("_")[-1].split("-")[:2])
        return time(hh, mm)
    except:
        return None

filtered_ts = [
    ts for ts in timestamps_all
    if extract_time(ts) and time(8, 0) <= extract_time(ts) <= time(16, 0)
]

# ==================================================
# USER CONTROLS
# ==================================================
st.subheader("🕒 Timestamp & Window Settings")

c1, c2, c3, c4 = st.columns(4)

t1 = c1.selectbox("TS1", filtered_ts, 0)
t2 = c2.selectbox("TS2", filtered_ts, 1)
t3 = c3.selectbox("TS3", filtered_ts, 2)

X = c4.number_input(
    "Strike Window X",
    min_value=1,
    max_value=10,
    value=4,
    step=1
)

# ==================================================
# LOAD DATA
# ==================================================
df1 = pd.read_csv(file_map[t1])
df2 = pd.read_csv(file_map[t2])
df3 = pd.read_csv(file_map[t3])

# ==================================================
# BUILD BASE TABLE
# ==================================================
dfs = []
for i, d in enumerate([df1, df2, df3]):
    dfs.append(
        d[["Stock", "Strike", "Stock_LTP", "Stock_%_Change", "CE_OI", "PE_OI"]]
        .rename(columns={
            "Stock_LTP": f"ltp_{i}",
            "Stock_%_Change": f"tot_ch_{i}",
            "CE_OI": f"ce_{i}",
            "PE_OI": f"pe_{i}"
        })

    )

df = dfs[0].merge(dfs[1], on=["Stock", "Strike"]) \
            .merge(dfs[2], on=["Stock", "Strike"])

# ==================================================
# NUMERIC SAFETY
# ==================================================
for c in df.columns:
    if any(x in c for x in ["ltp", "ce", "pe", "Strike", "tot_ch", "total_ch"]):
        df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0)


# ==================================================
# CORE CALCULATIONS
# ==================================================
df["d_ce"] = df["ce_0"] - df["ce_1"]
df["d_pe"] = df["pe_0"] - df["pe_1"]
df["d_ce_23"] = df["ce_1"] - df["ce_2"]
df["d_pe_23"] = df["pe_1"] - df["pe_2"]
df["total_ch"] = df["tot_ch_0"]


df["ce_x_23"] = (df["d_ce_23"] * df["Strike"]) / 10000
df["pe_x_23"] = (df["d_pe_23"] * df["Strike"]) / 10000

df["diff_23"] = df["pe_x_23"] - df["ce_x_23"]


df["ch"] = ((df["ltp_0"] - df["ltp_1"]) / df["ltp_1"]) * 100

df["ce_x"] = (df["d_ce"] * df["Strike"]) / 10000
df["pe_x"] = (df["d_pe"] * df["Strike"]) / 10000
df["oi_window_diff"] = np.nan

# ==================================================
# SLIDING WINDOW SUM (STRIKE-BASED)
# ==================================================
df["sum_ce"] = np.nan
df["sum_pe"] = np.nan
df["diff"] = np.nan
df["atm_diff"] = np.nan

for stk, g in df.groupby("Stock"):
    g = g.sort_values("Strike").reset_index()

    for i in range(len(g)):
        low = max(0, i - X)
        high = min(len(g) - 1, i + X)

        ce_sum = g.loc[low:high, "ce_x"].sum()
        pe_sum = g.loc[low:high, "pe_x"].sum()
        diff_val = pe_sum - ce_sum

        orig_idx = g.loc[i, "index"]
        df.at[orig_idx, "sum_ce"] = ce_sum
        df.at[orig_idx, "sum_pe"] = pe_sum
        df.at[orig_idx, "diff"] = diff_val

        # ---- OI WINDOW (ABSOLUTE OI, SAME WINDOW X) ----
        ce_oi_sum = (g.loc[low:high, "ce_0"] * g.loc[low:high, "Strike"]).sum()
        pe_oi_sum = (g.loc[low:high, "pe_0"] * g.loc[low:high, "Strike"]).sum()

        oi_diff_val = (pe_oi_sum - ce_oi_sum) / 10000
        df.at[orig_idx, "oi_window_diff"] = oi_diff_val

    # ---- ATM DIFF (AFTER LOOP) ----
    ltp = g["ltp_0"].iloc[0]
    atm_idx = (g["Strike"] - ltp).abs().values.argmin()

    low = max(0, atm_idx - 2)
    high = min(len(g) - 1, atm_idx + 2)

    orig_indices = g.loc[low:high, "index"]
    window = df.loc[orig_indices, "diff"].fillna(0)

    atm_avg = window.mean()

    df.loc[g["index"], "atm_diff"] = atm_avg





df["diff"] = pd.to_numeric(df["diff"], errors="coerce").fillna(0)
df["oi_window_diff"] = pd.to_numeric(df["oi_window_diff"], errors="coerce").fillna(0)

# ==================================================
# FINAL TABLE (ALL STRIKES)
# ==================================================
table = df[[
    "Stock",
    "Strike",
    "ltp_0",
    "ch",
    "total_ch",
    "d_ce",
    "d_pe",
    "ce_x",
    "pe_x",
    "diff",
    "diff_23",
    "atm_diff"

]].rename(columns={
    "Stock": "stk",
    "Strike": "str",
    "ltp_0": "ltp"
})


# ==================================================
# DISPLAY FILTER: 5 ABOVE + 5 BELOW LTP
# ==================================================
def filter_near_ltp(df, n=5):
    blocks = []
    for stk, g in df.groupby("stk"):
        g = g.sort_values("str").reset_index(drop=True)
        ltp = g["ltp"].iloc[0]

        atm_idx = (g["str"] - ltp).abs().idxmin()

        start = max(0, atm_idx - n)
        end = min(len(g) - 1, atm_idx + n)

        blocks.append(g.iloc[start:end + 1])

    return pd.concat(blocks, ignore_index=True)


display_df = filter_near_ltp(table, n=5)

# ==================================================
# ATM BLUE HIGHLIGHT
# ==================================================
def atm_blue(data):
    styles = pd.DataFrame("", index=data.index, columns=data.columns)

    for stk in data["stk"].dropna().unique():
        sdf = data[data["stk"] == stk].sort_values("str")
        ltp = sdf["ltp"].iloc[0]
        strikes = sdf["str"].values

        for i in range(len(strikes) - 1):
            if strikes[i] <= ltp <= strikes[i + 1]:
                styles.loc[sdf.index[i]] = "background-color:#003366;color:white"
                styles.loc[sdf.index[i + 1]] = "background-color:#003366;color:white"
                break

    return styles

# ==================================================
# FORMAT
# ==================================================
fmt = {
    "str": "{:.2f}",
    "ltp": "{:.2f}",
    "ch": "{:.2f}",
    "d_ce": "{:.0f}",
    "d_pe": "{:.0f}",
    "ce_x": "{:.0f}",
    "pe_x": "{:.0f}",
    "sum_ce": "{:.0f}",
    "sum_pe": "{:.0f}",
    "diff": "{:.0f}",
    "atm_diff": "{:.0f}",
    "total_ch": "{:.2f}",
    "diff_23": "{:.0f}"


}

# ==================================================
# DISPLAY
# ==================================================
# ==================================================
# MARKET BREADTH SUMMARY (UP COUNTS)
# ==================================================
# ==================================================
# MARKET BREADTH SUMMARY (ATM DIFF)
# ==================================================
up_atm = display_df[display_df["atm_diff"] > 0]["stk"].nunique()

st.markdown(f"### 🟢 UP : {up_atm}")


st.dataframe(
    display_df
    .style
    .apply(atm_blue, axis=None)
    .format(fmt, na_rep=""),
    use_container_width=True
)

st.markdown("---")
st.subheader("📈 Stock Presence in Top Diff Strikes")

cA, cB, cC = st.columns(3)

TOP_N = cA.number_input(
    "Top N Strikes",
    min_value=10,
    max_value=500,
    value=150,
    step=10
)

MIN_COUNT = cB.number_input(
    "Min Count",
    min_value=1,
    max_value=20,
    value=6,
    step=1
)

TOP_FIRST = cC.toggle("Top First", value=False)



if TOP_FIRST:
    sorted_df = table.sort_values("diff", ascending=False)
else:
    sorted_df = table.sort_values("diff", ascending=True)

top_n_df = sorted_df.merge(
    display_df[["stk", "str"]],
    on=["stk", "str"],
    how="inner"
).head(TOP_N)


stock_summary = (
    top_n_df
    .groupby("stk")
    .agg(
        count=("diff", "size"),
        total_ch=("total_ch", "first"),
        ch=("ch", "first")
    )
    .reset_index()
)

stock_summary = stock_summary[stock_summary["count"] >= MIN_COUNT]
stock_summary = stock_summary.sort_values("count", ascending=False)
diff_stocks = stock_summary["stk"].tolist()

# --- SECOND SUMMARY (BASED ON diff_23) ---

if TOP_FIRST:
    sorted_df_23 = table.sort_values("diff_23", ascending=False)
else:
    sorted_df_23 = table.sort_values("diff_23", ascending=True)

top_n_df_23 = sorted_df_23.merge(
    display_df[["stk", "str"]],
    on=["stk", "str"],
    how="inner"
).head(TOP_N)


stock_summary_23 = (
    top_n_df_23
    .groupby("stk")
    .agg(
        count=("diff_23", "size")
    )
    .reset_index()
)

stock_summary_23 = stock_summary_23[stock_summary_23["count"] >= MIN_COUNT]
stock_summary_23 = stock_summary_23.sort_values("count", ascending=False)
common_stocks = set(stock_summary["stk"]) & set(stock_summary_23["stk"])


def highlight_common(data, common_stocks):
    styles = pd.DataFrame("", index=data.index, columns=data.columns)
    styles.loc[data["stk"].isin(common_stocks), :] = "background-color:#1f4e79;color:white"
    return styles

fmt2 = {
    "count": "{:.0f}",
    "total_ch": "{:.2f}",
    "ch": "{:.2f}"
}

c1, c2 = st.columns(2)

fmt2 = {
    "count": "{:.0f}",
    "total_ch": "{:.2f}",
    "ch": "{:.2f}"
}

with c1:
    st.subheader("📊 Based on diff")
    st.dataframe(
        stock_summary
        .style
        .apply(highlight_common, common_stocks=common_stocks, axis=None)
        .format(fmt2, na_rep=""),
        use_container_width=True
    )

with c2:
    st.subheader("📊 Based on diff_23")
    st.dataframe(
        stock_summary_23
        .style
        .format({"count": "{:.0f}"}, na_rep=""),
        use_container_width=True
    )

st.markdown("---")
st.subheader("📋 Detailed View – Stocks from 'Based on diff'")
detail_df = display_df[display_df["stk"].isin(diff_stocks)]

st.dataframe(
    detail_df
    .style
    .apply(atm_blue, axis=None)
    .format(fmt, na_rep=""),
    use_container_width=True
)

# ==================================================
# DETAILED TABLE FOR "BASED ON diff" STOCKS
# ==================================================


detail_df = display_df[display_df["stk"].isin(diff_stocks)]



st.markdown("---")
st.subheader("🔍 Stock Detail View")

stock_list = [""] + sorted(display_df["stk"].unique().tolist())

c1, c2 = st.columns(2)

stock_a = c1.selectbox("Select Stock A", stock_list, index=0)
stock_b = c2.selectbox("Select Stock B", stock_list, index=0)

def show_stock_table(stock_name):
    if stock_name == "":
        return

    stock_df = display_df[display_df["stk"] == stock_name]

    st.dataframe(
        stock_df
        .style
        .apply(atm_blue, axis=None)
        .format(fmt, na_rep=""),
        use_container_width=True
    )
if stock_a or stock_b:
    c1, c2 = st.columns(2)

    with c1:
        show_stock_table(stock_a)

    with c2:
        show_stock_table(stock_b)



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
# EARLY WINDOW (FIRST TWO CSVs AFTER 09:10)
# ==================================================
early_ts = sorted(
    [ts for ts in filtered_ts if extract_time(ts) >= time(9, 10)],
    key=lambda x: extract_time(x)
)

if len(early_ts) < 2:
    st.error("Need at least 2 CSV files after 09:10 for early OI window")
    st.stop()

t0a, t0b = early_ts[0], early_ts[1]
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
df0a = pd.read_csv(file_map[t0a])
df0b = pd.read_csv(file_map[t0b])

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
# EARLY OI DELTA (09:10 WINDOW)
# ==================================================
early_df = (
    df0a[["Stock", "Strike", "CE_OI", "PE_OI"]]
    .merge(
        df0b[["Stock", "Strike", "CE_OI", "PE_OI"]],
        on=["Stock", "Strike"],
        suffixes=("_a", "_b")
    )
)

for c in ["CE_OI_a", "CE_OI_b", "PE_OI_a", "PE_OI_b", "Strike"]:
    early_df[c] = pd.to_numeric(early_df[c], errors="coerce").fillna(0)

early_df["d_ce_0"] = early_df["CE_OI_b"] - early_df["CE_OI_a"]
early_df["d_pe_0"] = early_df["PE_OI_b"] - early_df["PE_OI_a"]

early_df["ce_x_0"] = (early_df["d_ce_0"] * early_df["Strike"]) / 10000
early_df["pe_x_0"] = (early_df["d_pe_0"] * early_df["Strike"]) / 10000

df = df.merge(
    early_df[["Stock", "Strike", "ce_x_0", "pe_x_0"]],
    on=["Stock", "Strike"],
    how="left"
)

df["ce_x_0"] = pd.to_numeric(df["ce_x_0"], errors="coerce").fillna(0)
df["pe_x_0"] = pd.to_numeric(df["pe_x_0"], errors="coerce").fillna(0)

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
    "ce_x_0",
    "pe_x_0",
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

def red_early_columns(data):
    styles = pd.DataFrame("", index=data.index, columns=data.columns)
    for col in ["ce_x_0", "pe_x_0"]:
        if col in styles.columns:
            styles[col] = "color:red;font-weight:bold"
    return styles
# ==================================================
# STOCK NAME ONLY HIGHLIGHT (BOTTOM N PE / CE)
# ==================================================
def highlight_stk_only(data):
    styles = pd.DataFrame("", index=data.index, columns=data.columns)

    if "stk" not in data.columns:
        return styles

    for i, stk in data["stk"].items():

        # 🟠 Bottom N in BOTH PE & CE
        if stk in bottom_pe_stocks and stk in bottom_ce_stocks:
            styles.at[i, "stk"] = "background-color:#ff8c00;color:black"

        # 🟢 Bottom N in PE only
        elif stk in bottom_pe_stocks:
            styles.at[i, "stk"] = "background-color:#1b5e20;color:white"

        # 🔴 Bottom N in CE only
        elif stk in bottom_ce_stocks:
            styles.at[i, "stk"] = "background-color:#8b0000;color:white"

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
    "ce_x_0": "{:.0f}",
    "pe_x_0": "{:.0f}",
    "diff": "{:.0f}",
    "atm_diff": "{:.0f}",
    "total_ch": "{:.2f}",
    "diff_23": "{:.0f}"


}



# ==================================================
# 🔃 MAIN TABLE SORT TOGGLES
# ==================================================
st.markdown("### 🔃 Sorting Options (Main Table)")
# ==================================================


if "pe_min" not in st.session_state:
    st.session_state.pe_min = False
if "ce_min" not in st.session_state:
    st.session_state.ce_min = False

c_pe, c_ce, c_n = st.columns([1, 1, 1])

def pe_callback():
    if st.session_state.pe_min:
        st.session_state.ce_min = False

def ce_callback():
    if st.session_state.ce_min:
        st.session_state.pe_min = False

c_pe.toggle(
    "PE Min",
    key="pe_min",
    on_change=pe_callback
)

c_ce.toggle(
    "CE Min",
    key="ce_min",
    on_change=ce_callback
)

PE_MIN = st.session_state.pe_min
CE_MIN = st.session_state.ce_min

BOTTOM_N = c_n.number_input(
    "Bottom N",
    min_value=5,
    max_value=100,
    value=20,
    step=5
)
# ==================================================
# DISPLAY
# ==================================================


up_atm = display_df[display_df["atm_diff"] > 0]["stk"].nunique()
sum_atm = display_df["atm_diff"].sum() / 1000

# ==================================================
# OPTIONAL STOCK-WISE SORTING (PE MIN / CE MIN)
# ==================================================
if PE_MIN or CE_MIN:

    stock_sort = (
        display_df
        .groupby("stk")
        .agg(
            pe_sum=("pe_x", "sum"),
            ce_sum=("ce_x", "sum")
        )
        .reset_index()
    )

    if PE_MIN:
        stock_sort = stock_sort.sort_values("pe_sum", ascending=True)
    elif CE_MIN:
        stock_sort = stock_sort.sort_values("ce_sum", ascending=True)

    ordered_stocks = stock_sort["stk"].tolist()

    sorted_blocks = []
    for stk in ordered_stocks:
        block = (
            display_df[display_df["stk"] == stk]
            .sort_values("str")
        )
        sorted_blocks.append(block)

    display_df = pd.concat(sorted_blocks, ignore_index=True)

# ==================================================
# 📌 BOTTOM N STOCKS FOR HIGHLIGHTING
# ==================================================
stock_sums = (
    display_df
    .groupby("stk")
    .agg(
        pe_sum=("pe_x", "sum"),
        ce_sum=("ce_x", "sum")
    )
    .reset_index()
)

bottom_pe_stocks = set(
    stock_sums.sort_values("pe_sum", ascending=True)
    .head(BOTTOM_N)["stk"]
)

bottom_ce_stocks = set(
    stock_sums.sort_values("ce_sum", ascending=True)
    .head(BOTTOM_N)["stk"]
)

st.markdown(f"### 🟢 UP : {up_atm} &nbsp;&nbsp; | &nbsp;&nbsp; Σ ATM : {sum_atm:.0f}")



st.dataframe(
    display_df
    .style
    .apply(atm_blue, axis=None)
    .apply(highlight_stk_only, axis=None)
    .apply(red_early_columns, axis=None)
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



# NEW CHANGE
# ==================================================
# 📈 UP TREND TABLE
# ==================================================
st.markdown("---")
st.subheader("📈 UP TREND")

uptrend_blocks = []
ranking = []

for stk, g in display_df.groupby("stk"):
    g = g.sort_values("str").reset_index(drop=True)
    ltp = g["ltp"].iloc[0]

    # --- ATM index ---
    atm_idx = (g["str"] - ltp).abs().idxmin()

    # Strike windows (index-safe)
    pe_idxs = [i for i in range(atm_idx, atm_idx + 4) if i < len(g)]
    ce_idxs = [i for i in range(atm_idx - 4, atm_idx + 3) if 0 <= i < len(g)]
    pe_pos_idxs = [i for i in range(atm_idx, atm_idx + 2) if i < len(g)]

    pe_window = g.loc[pe_idxs]
    ce_window = g.loc[ce_idxs]
    pe_pos_window = g.loc[pe_pos_idxs]

    # ---------------- CONDITIONS ----------------
    cond_pe_big = (pe_window["pe_x"] > 900).any()
    cond_ce_small = not (ce_window["ce_x"] > 4000).any()
    cond_pe_positive = (pe_pos_window["pe_x"] > 0).all()

    if not (cond_pe_big and cond_ce_small and cond_pe_positive):
        continue

    # ---------------- SCORING ----------------
    pe_max = pe_window["pe_x"].max()
    ce_max = ce_window["ce_x"].max()
    score = pe_max - ce_max

    ranking.append((stk, score))
    uptrend_blocks.append(g)

# ---- EXIT IF EMPTY ----
if not uptrend_blocks:
    st.info("No UP TREND stocks found.")
else:
    # ---- SORT STOCKS BY SCORE ----
    rank_df = pd.DataFrame(ranking, columns=["stk", "score"]) \
                .sort_values("score", ascending=False)

    final_blocks = []
    for stk in rank_df["stk"]:
        final_blocks.append(
            pd.concat(uptrend_blocks)
            .query("stk == @stk")
        )

    uptrend_df = pd.concat(final_blocks, ignore_index=True)

    st.dataframe(
        uptrend_df
        .style
        .apply(atm_blue, axis=None)
        .format(fmt, na_rep=""),
        use_container_width=True
    )

# ==================================================
# 🔻 DOWN TREND TABLE
# ==================================================
st.markdown("---")
st.subheader("🔻 DOWN TREND")

downtrend_blocks = []
ranking = []

for stk, g in display_df.groupby("stk"):
    g = g.sort_values("str").reset_index(drop=True)
    ltp = g["ltp"].iloc[0]

    # --- ATM index ---
    atm_idx = (g["str"] - ltp).abs().idxmin()

    # ----- STRIKE WINDOWS (INDEX SAFE) -----
    ce_strong_idxs = [i for i in range(atm_idx - 2, atm_idx + 2) if 0 <= i < len(g)]
    pe_weak_idxs   = [i for i in range(atm_idx - 3, atm_idx + 4) if 0 <= i < len(g)]
    ce_pos_idxs    = [i for i in range(atm_idx, atm_idx + 2) if i < len(g)]

    ce_strong = g.loc[ce_strong_idxs]
    pe_weak   = g.loc[pe_weak_idxs]
    ce_pos    = g.loc[ce_pos_idxs]

    # ---------------- CONDITIONS ----------------
    cond_ce_big = (ce_strong["ce_x"] > 900).any()
    cond_pe_small = not (pe_weak["pe_x"] > 4000).any()
    cond_ce_positive = (ce_pos["ce_x"] > 0).all()

    if not (cond_ce_big and cond_pe_small and cond_ce_positive):
        continue

    # ---------------- SCORING ----------------
    ce_max = ce_strong["ce_x"].max()
    pe_max = pe_weak["pe_x"].max()
    score = ce_max - pe_max

    ranking.append((stk, score))
    downtrend_blocks.append(g)

# ---- EXIT IF EMPTY ----
if not downtrend_blocks:
    st.info("No DOWN TREND stocks found.")
else:
    # ---- SORT BY SCORE ----
    rank_df = (
        pd.DataFrame(ranking, columns=["stk", "score"])
        .sort_values("score", ascending=False)
    )

    final_blocks = []
    for stk in rank_df["stk"]:
        final_blocks.append(
            pd.concat(downtrend_blocks)
            .query("stk == @stk")
        )

    downtrend_df = pd.concat(final_blocks, ignore_index=True)

    st.dataframe(
        downtrend_df
        .style
        .apply(atm_blue, axis=None)
        .format(fmt, na_rep=""),
        use_container_width=True
    )



# ==================================================
# 📊 CE / PE SUMMARY TABLE (MAIN TABLE BASED)
# ==================================================
st.markdown("---")
st.subheader("📊 CE–PE Summary (Main Table)")

summary_df = (
    table
    .groupby("stk")
    .agg(
        ce_x_sum=("ce_x", "sum"),
        pe_x_sum=("pe_x", "sum"),
        ch=("ch", "first")
    )
    .reset_index()
)

# ---- SAFE RATIO ----
summary_df["pe_ce_ratio"] = np.where(
    summary_df["ce_x_sum"] != 0,
    summary_df["pe_x_sum"] / summary_df["ce_x_sum"],
    np.nan
)

summary_df["pe_ce_ratio"] = summary_df["pe_ce_ratio"].round(2)

# ---- OPTIONAL SORT (STRONG PE BIAS FIRST) ----
summary_df = summary_df.sort_values("pe_ce_ratio", ascending=False)

fmt_summary = {
    "ce_x_sum": "{:.0f}",
    "pe_x_sum": "{:.0f}",
    "pe_ce_ratio": "{:.2f}",
    "ch": "{:.2f}"
}

st.dataframe(
    summary_df
    .style
    .format(fmt_summary, na_rep=""),
    use_container_width=True
)


import streamlit as st
import pandas as pd
import numpy as np
import os

# =====================================
# CONFIG
# =====================================
st.set_page_config(page_title="ΔΔ Max Pain Viewer", layout="wide")
st.title("📊 ΔΔ Max Pain Viewer")

DATA_DIR = "data"

# =====================================
# LOAD CSV FILES
# =====================================
def load_csv_files():
    files = []
    for f in os.listdir(DATA_DIR):
        if f.startswith("option_chain_") and f.endswith(".csv"):
            ts = f.replace("option_chain_", "").replace(".csv", "")
            files.append((ts, os.path.join(DATA_DIR, f)))
    return sorted(files, reverse=True)

csv_files = load_csv_files()
if len(csv_files) < 6:
    st.error("Need at least 6 option chain CSV files")
    st.stop()

timestamps = [ts for ts, _ in csv_files]
file_map = dict(csv_files)

def short_ts(ts):
    return ts.split("_")[-1].replace("-", ":")

# =====================================
# TIMESTAMP SELECTION
# =====================================
st.subheader("🕒 Timestamp Selection")

t1 = st.selectbox("Base Timestamp (T1)", timestamps, index=0)

c1, c2, c3, c4, c5 = st.columns(5)
with c1: t2 = st.selectbox("T2", timestamps, index=1)
with c2: t3 = st.selectbox("T3", timestamps, index=2)
with c3: t4 = st.selectbox("T4", timestamps, index=3)
with c4: t5 = st.selectbox("T5", timestamps, index=4)
with c5: t6 = st.selectbox("T6", timestamps, index=5)

compare_ts = [t2, t3, t4, t5, t6]

# sort time columns in increasing HH:MM order
time_cols = sorted([short_ts(ts) for ts in compare_ts])

# =====================================
# LOAD BASE DATA
# =====================================
df_base = pd.read_csv(file_map[t1])

required_cols = {"Stock", "Strike", "Max_Pain", "Stock_LTP"}
if not required_cols.issubset(df_base.columns):
    st.error("CSV must contain Stock, Strike, Max_Pain, Stock_LTP")
    st.stop()

df_base["Stock"] = df_base["Stock"].str.upper().str.strip()

# =====================================
# SESSION STATE FOR STOCKS
# =====================================
if "stocks_to_show" not in st.session_state:
    st.session_state.stocks_to_show = []

all_stocks = sorted(df_base["Stock"].unique())

# =====================================
# FUNCTION TO RENDER ONE STOCK TABLE
# =====================================
def render_stock_table(selected_stock):

    df = df_base.copy()

    # ---------- merge + ΔΔ MP ----------
    for ts in compare_ts:
        label = short_ts(ts)

        df_ts = pd.read_csv(file_map[ts])
        df_ts["Stock"] = df_ts["Stock"].str.upper().str.strip()

        df = df.merge(
            df_ts[["Stock", "Strike", "Max_Pain"]],
            on=["Stock", "Strike"],
            suffixes=("", "_cmp")
        )

        delta_col = f"_delta_{label}"
        df[delta_col] = df["Max_Pain"] - df["Max_Pain_cmp"]
        df[label] = np.nan

        for stock, sdf in df.sort_values("Strike").groupby("Stock"):
            vals = sdf[delta_col].astype(float).values
            diff = vals - np.roll(vals, -1)
            diff[-1] = np.nan
            df.loc[sdf.index, label] = diff

        df.drop(columns=["Max_Pain_cmp", delta_col], inplace=True)

    # ---------- filter stock ----------
    sdf = df[df["Stock"] == selected_stock].sort_values("Strike").reset_index(drop=True)

    # ---------- ATM ±6 ----------
    ltp = float(sdf["Stock_LTP"].iloc[0])
    strikes = sdf["Strike"].values

    atm_idx = None
    for i in range(len(strikes) - 1):
        if strikes[i] <= ltp <= strikes[i + 1]:
            atm_idx = i
            break
    if atm_idx is None:
        return

    view_df = sdf.iloc[max(0, atm_idx - 6): min(len(sdf), atm_idx + 2 + 6)]

    display_df = view_df[["Stock", "Strike", "Stock_LTP"] + time_cols].copy()

    # ---------- highlighting ----------
    def highlight(df):
        styles = pd.DataFrame("", index=df.index, columns=df.columns)

        # max pain (RED – priority)
        mp_strike = sdf.loc[sdf["Max_Pain"].idxmin(), "Strike"]
        mp_idx = df.index[df["Strike"] == mp_strike]
        styles.loc[mp_idx] = "background-color:#8B0000;color:white"

        # ATM (BLUE)
        for i in range(len(strikes) - 1):
            if strikes[i] <= ltp <= strikes[i + 1]:
                atm_strikes = [strikes[i], strikes[i + 1]]
                for idx in df.index[df["Strike"].isin(atm_strikes)]:
                    if styles.loc[idx].eq("").all():
                        styles.loc[idx] = "background-color:#003366;color:white"
                break

        return styles

    st.subheader(f"📈 {selected_stock}")

    st.dataframe(
        display_df.style
        .apply(highlight, axis=None)
        .format(
            {
                "Strike": "{:.0f}",
                "Stock_LTP": "{:.2f}",
                **{c: "{:.0f}" for c in time_cols}
            }
        ),
        use_container_width=True
    )

# =====================================
# MAIN LOOP – ADD STOCKS ONE BY ONE
# =====================================
for stock in st.session_state.stocks_to_show:
    render_stock_table(stock)

# dropdown AFTER last table
remaining = [s for s in all_stocks if s not in st.session_state.stocks_to_show]
if remaining:
    next_stock = st.selectbox("➕ Add another stock", [""] + remaining)
    if next_stock:
        st.session_state.stocks_to_show.append(next_stock)
        st.experimental_rerun()

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
all_stocks = sorted(df_base["Stock"].unique())

# =====================================
# SESSION STATE
# =====================================
if "tables" not in st.session_state:
    st.session_state.tables = [all_stocks[0]]

# =====================================
# ΔΔ MP CALCULATION
# =====================================
def compute_ddmp(df):
    df = df.copy()

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

    return df

# =====================================
# MONOTONIC FILTER (≥4 of 5)
# =====================================
def is_monotonic_4_of_5(values):
    inc = 0
    dec = 0
    for i in range(4):
        if values[i] <= values[i + 1]:
            inc += 1
        if values[i] >= values[i + 1]:
            dec += 1
    return inc >= 4 or dec >= 4

# =====================================
# RENDER ONE MAIN TABLE
# =====================================
def render_table(table_idx):

    selected_stock = st.selectbox(
        f"Select Stock (Table {table_idx + 1})",
        all_stocks,
        index=all_stocks.index(st.session_state.tables[table_idx]),
        key=f"stock_{table_idx}"
    )
    st.session_state.tables[table_idx] = selected_stock

    df = compute_ddmp(df_base)
    sdf = df[df["Stock"] == selected_stock].sort_values("Strike").reset_index(drop=True)

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
    display_df = view_df[["Stock", "Strike"] + time_cols + ["Stock_LTP"]].copy()

    def highlight(df_):
        styles = pd.DataFrame("", index=df_.index, columns=df_.columns)

        # Max Pain (RED)
        mp_strike = sdf.loc[sdf["Max_Pain"].idxmin(), "Strike"]
        styles.loc[df_.index[df_["Strike"] == mp_strike]] = "background-color:#8B0000;color:white"

        # ATM (BLUE)
        for i in range(len(strikes) - 1):
            if strikes[i] <= ltp <= strikes[i + 1]:
                atm_strikes = [strikes[i], strikes[i + 1]]
                for idx in df_.index[df_["Strike"].isin(atm_strikes)]:
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
# MAIN TABLES
# =====================================
for i in range(len(st.session_state.tables)):
    render_table(i)

if st.button("➕ Add another table"):
    st.session_state.tables.append(all_stocks[0])

# =====================================
# FILTERED TABLE — ALL STRIKES
# =====================================
st.subheader("🧩 Stocks & Strikes with Consistent ΔΔ MP Trend (≥4 of 5)")

filtered_rows = []
df_all = compute_ddmp(df_base)

for stock in all_stocks:
    sdf = df_all[df_all["Stock"] == stock].sort_values("Strike").reset_index(drop=True)
    if sdf.empty:
        continue

    for _, row in sdf.iterrows():
        values = [row[c] for c in time_cols]
        if any(pd.isna(values)):
            continue

        if is_monotonic_4_of_5(values):
            filtered_rows.append({
                "Stock": stock,
                "Strike": int(row["Strike"]),
                **{c: int(row[c]) for c in time_cols},
                "Stock_LTP": round(row["Stock_LTP"], 2)
            })

# =====================================
# DISPLAY FILTERED TABLE
# =====================================
if filtered_rows:
    filtered_df = pd.DataFrame(filtered_rows).sort_values(["Stock", "Strike"])

    st.dataframe(
        filtered_df.style.format(
            {
                "Strike": "{:.0f}",
                "Stock_LTP": "{:.2f}",
                **{c: "{:.0f}" for c in time_cols}
            }
        ),
        use_container_width=True
    )
else:
    st.info("No strikes matched the ΔΔ MP trend condition.")

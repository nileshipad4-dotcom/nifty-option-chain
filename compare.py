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
if len(csv_files) < 3:
    st.error("Need at least 3 CSV files.")
    st.stop()

latest_file = csv_files[0][0]
if "last_ts" not in st.session_state:
    st.session_state.last_ts = latest_file
if latest_file != st.session_state.last_ts:
    st.session_state.last_ts = latest_file
    st.experimental_rerun()

timestamps = [ts for ts, _ in csv_files]
file_map = dict(csv_files)

def short_ts(ts):
    return ts.split("_")[-1].replace("-", ":")

# =====================================
# DROPDOWNS
# =====================================
c1, c2, c3 = st.columns(3)
with c1:
    t1 = st.selectbox("Timestamp 1 (Latest)", timestamps, 0)
with c2:
    t2 = st.selectbox("Timestamp 2", timestamps, 1)
with c3:
    t3 = st.selectbox("Timestamp 3", timestamps, 2)

t1_lbl, t2_lbl, t3_lbl = short_ts(t1), short_ts(t2), short_ts(t3)

# =====================================
# COLUMN NAMES
# =====================================
mp1_col = f"MP ({t1_lbl})"
mp2_col = f"MP ({t2_lbl})"
mp3_col = f"MP ({t3_lbl})"

pct_col = f"% Ch ({t1_lbl})"
delta_12 = f"Δ MP ({t1_lbl}-{t2_lbl})"
delta_23 = f"Δ MP ({t2_lbl}-{t3_lbl})"
ltp_pct_12_col = f"% LTP Ch ({t1_lbl}-{t2_lbl})"

# =====================================
# LOAD DATA
# =====================================
df1 = pd.read_csv(file_map[t1])
df2 = pd.read_csv(file_map[t2])
df3 = pd.read_csv(file_map[t3])

if "Stock_%_Change" not in df1.columns:
    df1["Stock_%_Change"] = np.nan

df1 = df1[["Stock", "Strike", "Max_Pain", "Stock_LTP", "Stock_%_Change"]].rename(
    columns={"Max_Pain": mp1_col, "Stock_%_Change": pct_col}
)
df2 = df2[["Stock", "Strike", "Max_Pain", "Stock_LTP"]].rename(
    columns={"Max_Pain": mp2_col, "Stock_LTP": "Stock_LTP_t2"}
)
df3 = df3[["Stock", "Strike", "Max_Pain"]].rename(
    columns={"Max_Pain": mp3_col}
)

df = df1.merge(df2, on=["Stock", "Strike"]).merge(df3, on=["Stock", "Strike"])

# =====================================
# BASIC CALCULATIONS (ONLY REQUIRED)
# =====================================
df[delta_12] = df[mp1_col] - df[mp2_col]
df[delta_23] = df[mp2_col] - df[mp3_col]

df[ltp_pct_12_col] = (
    (df["Stock_LTP"] - df["Stock_LTP_t2"]) / df["Stock_LTP_t2"]
) * 100

# =====================================
# CLEAN + REMOVE INDICES
# =====================================
df["Stock"] = df["Stock"].astype(str).str.upper().str.strip()
EXCLUDE = {"NIFTY", "BANKNIFTY", "FINNIFTY", "MIDCPNIFTY"}
df = df[~df["Stock"].isin(EXCLUDE)]

# =====================================
# FINAL COLUMN ORDER
# =====================================
df = df[
    [
        "Stock",
        "Strike",
        delta_12,
        delta_23,
        "Stock_LTP",
        pct_col,
        ltp_pct_12_col,
    ]
]

# =====================================
# FILTER ±6 STRIKES AROUND LTP
# =====================================
def filter_strikes_around_ltp(df, below=6, above=6):
    out = []

    for stock, sdf in df.groupby("Stock"):
        sdf = sdf.dropna(subset=["Strike"]).sort_values("Strike")
        if sdf.empty:
            continue

        ltp = float(sdf["Stock_LTP"].iloc[0])
        strikes = sdf["Strike"].values

        atm_idx = None
        for i in range(len(strikes) - 1):
            if strikes[i] <= ltp <= strikes[i + 1]:
                atm_idx = i
                break

        if atm_idx is None:
            continue

        start = max(0, atm_idx - below)
        end = min(len(sdf), atm_idx + 2 + above)

        out.append(sdf.iloc[start:end])
        out.append(pd.DataFrame([{c: np.nan for c in df.columns}]))

    return pd.concat(out[:-1], ignore_index=True)

# =====================================
# HIGHLIGHTING (ATM + MIN MP)
# =====================================
def highlight_rows(data):
    styles = pd.DataFrame("", index=data.index, columns=data.columns)

    for stock in data["Stock"].dropna().unique():
        sdf = data[(data["Stock"] == stock) & data["Strike"].notna()].sort_values("Strike")
        if sdf.empty:
            continue

        ltp = float(sdf["Stock_LTP"].iloc[0])
        strikes = sdf["Strike"].values

        for i in range(len(strikes) - 1):
            if strikes[i] <= ltp <= strikes[i + 1]:
                styles.loc[sdf.index[i]] = "background-color:#003366;color:white"
                styles.loc[sdf.index[i + 1]] = "background-color:#003366;color:white"
                break

        styles.loc[sdf[delta_12].abs().idxmax()] = "background-color:#8B0000;color:white"

    return styles

# =====================================
# DISPLAY
# =====================================
display_df = filter_strikes_around_ltp(df)

st.subheader(f"📊 ALL STOCKS: {t1_lbl} vs {t2_lbl} vs {t3_lbl}")

st.dataframe(
    display_df
    .style.apply(highlight_rows, axis=None)
    .format(
        {
            delta_12: "{:.0f}",
            delta_23: "{:.0f}",
            "Stock_LTP": "{:.2f}",
            pct_col: "{:.3f}",
            ltp_pct_12_col: "{:.2f}",
        },
        na_rep="",
    ),
    use_container_width=True,
)

# =====================================
# DOWNLOAD
# =====================================
st.download_button(
    "⬇️ Download Comparison CSV",
    df.to_csv(index=False),
    f"max_pain_clean_{t1_lbl}_{t2_lbl}_{t3_lbl}.csv",
    "text/csv",
)

# =====================================
# SINGLE STOCK VIEW
# =====================================
st.subheader("🔍 View Individual Stock")

stock_list = sorted(df["Stock"].dropna().unique())
selected_stock = st.selectbox("Select Stock", [""] + stock_list)

if selected_stock:
    stock_df = filter_strikes_around_ltp(df[df["Stock"] == selected_stock])

    st.dataframe(
        stock_df
        .style.apply(highlight_rows, axis=None)
        .format(
            {
                delta_12: "{:.0f}",
                delta_23: "{:.0f}",
                "Stock_LTP": "{:.2f}",
                pct_col: "{:.3f}",
                ltp_pct_12_col: "{:.2f}",
            },
            na_rep="",
        ),
        use_container_width=True,
    )

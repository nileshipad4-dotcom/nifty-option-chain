# stock OC main
import streamlit as st
import pandas as pd
import os
import numpy as np
from streamlit_autorefresh import st_autorefresh

st_autorefresh(interval=360_000, key="auto_refresh")

# =====================================
# STREAMLIT CONFIG
# =====================================
st.set_page_config(page_title="Max Pain Comparison", layout="wide")
st.title("📊 FnO STOCKS")

DATA_DIR = "data"

# =====================================
# LOAD SYMBOL → SECTOR (CASE INSENSITIVE)
# =====================================
@st.cache_data
def load_symbol_sector():
    df = pd.read_csv("Symbol - Sector.csv")
    df.columns = ["Stock", "Sector"]
    df["Stock"] = df["Stock"].astype(str).str.upper().str.strip()
    df["Sector"] = df["Sector"].astype(str).str.strip()
    return df

sector_df = load_symbol_sector()

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
sum_12_col = f"Σ {delta_12}"
delta_above_col = f"ΔΔ MP ({t1_lbl}-{t2_lbl})"
sum_2_above_below_col = f"Σ |ΔΔ MP| (±2)"
dist_col = "% Dist from LTP (Max |ΔΔ| ±3)"


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
df2 = df2[["Stock", "Strike", "Max_Pain"]].rename(columns={"Max_Pain": mp2_col})
df3 = df3[["Stock", "Strike", "Max_Pain"]].rename(columns={"Max_Pain": mp3_col})

df = df1.merge(df2, on=["Stock", "Strike"]).merge(df3, on=["Stock", "Strike"])

# =====================================
# NORMALIZE + REMOVE INDICES
# =====================================
df["Stock"] = df["Stock"].astype(str).str.upper().str.strip()
EXCLUDE = {"NIFTY", "BANKNIFTY", "FINNIFTY", "MIDCPNIFTY"}
df = df[~df["Stock"].isin(EXCLUDE)]

# =====================================
# MERGE SECTOR
# =====================================
df = df.merge(sector_df, on="Stock", how="left")

# =====================================
# CALCULATIONS
# =====================================
df[delta_12] = df[mp1_col] - df[mp2_col]
df[delta_23] = df[mp2_col] - df[mp3_col]

df[sum_12_col] = np.nan
for stock, sdf in df.sort_values("Strike").groupby("Stock"):
    df.loc[sdf.index, sum_12_col] = (
        sdf[delta_12].rolling(window=7, center=True, min_periods=1).sum().values
    )

df[delta_above_col] = np.nan
for stock, sdf in df.sort_values("Strike").groupby("Stock"):
    vals = sdf[delta_12].astype(float).values
    diff = vals - np.roll(vals, -1)
    diff[-1] = np.nan
    df.loc[sdf.index, delta_above_col] = diff

df[sum_2_above_below_col] = np.nan
for stock, sdf in df.sort_values("Strike").groupby("Stock"):
    sdf = sdf.reset_index(drop=True)
    ltp = float(sdf["Stock_LTP"].iloc[0])
    strikes = sdf["Strike"].values
    for i in range(len(strikes) - 1):
        if strikes[i] <= ltp <= strikes[i + 1]:
            df.loc[df["Stock"] == stock, sum_2_above_below_col] = (
                sdf.loc[[i, i + 1], delta_above_col].astype(float).sum()
            )
            break



# ==================================
# % DIFFERENCE FROM MAX PAIN 
# ==================================
df[dist_col] = np.nan

for stock, sdf in df.sort_values("Strike").groupby("Stock"):
    sdf = sdf.reset_index()
    ltp = float(sdf["Stock_LTP"].iloc[0])
    strikes = sdf["Strike"].values

    atm_idx = None
    for i in range(len(strikes) - 1):
        if strikes[i] <= ltp <= strikes[i + 1]:
            atm_idx = i + 1
            break

    if atm_idx is None:
        continue

    # indices: 3 below + 3 above LTP
    idxs = list(range(atm_idx - 3, atm_idx)) + list(range(atm_idx, atm_idx + 3))
    idxs = [i for i in idxs if 0 <= i < len(sdf)]

    if not idxs:
        continue

    subset = sdf.loc[idxs, ["Strike", delta_above_col]].dropna()

    if subset.empty:
        continue

    # max by absolute value
    max_row = subset.loc[subset[delta_above_col].abs().idxmax()]
    A = float(max_row["Strike"])

    value = ((A - ltp) / ltp) * 100

    df.loc[df["Stock"] == stock, dist_col] = value



# =====================================
# FINAL COLUMN ORDER
# =====================================
df = df[
    [
        "Stock",
        "Sector",
        "Strike",
        mp1_col,
        mp2_col,
        mp3_col,
        delta_12,
        delta_23,
        sum_12_col,
        delta_above_col,
        sum_2_above_below_col,
        dist_col,
        pct_col,
        "Stock_LTP",
    ]
]

# =====================================
# INSERT BLANK ROWS
# =====================================
rows = []
for stock, sdf in df.sort_values(["Stock", "Strike"]).groupby("Stock"):
    rows.append(sdf)
    rows.append(pd.DataFrame([{col: np.nan for col in df.columns}]))
final_df = pd.concat(rows[:-1], ignore_index=True)

# =====================================
# HIGHLIGHTING (ONLY ATM + MIN MAX PAIN)
# =====================================
def highlight_rows(data):
    styles = pd.DataFrame("", index=data.index, columns=data.columns)

    for stock in data["Stock"].dropna().unique():
        sdf = data[(data["Stock"] == stock) & data["Strike"].notna()].sort_values("Strike")
        if sdf.empty:
            continue

        ltp = float(sdf["Stock_LTP"].iloc[0])
        strikes = sdf["Strike"].values

        # ATM highlight
        for i in range(len(strikes) - 1):
            if strikes[i] <= ltp <= strikes[i + 1]:
                styles.loc[sdf.index[i]] = "background-color:#003366;color:white"
                styles.loc[sdf.index[i + 1]] = "background-color:#003366;color:white"
                break

        # Min Max Pain highlight
        styles.loc[sdf[mp1_col].astype(float).idxmin()] = "background-color:#8B0000;color:white"

    return styles

# =====================================
# DISPLAY
# =====================================
display_cols = [c for c in final_df.columns if c != sum_12_col]

st.subheader(f"📊 ALL STOCKS: {t1_lbl} vs {t2_lbl} vs {t3_lbl}")
st.dataframe(
    final_df[display_cols]
    .style.apply(highlight_rows, axis=None)
    .format(
        {c: "{:.3f}" if c == pct_col else "{:.2f}" if c == "Stock_LTP" else "{:.0f}"
         for c in display_cols if c not in {"Stock", "Sector"}},
        na_rep=""
    ),
    use_container_width=True,
)

# =====================================
# DOWNLOAD
# =====================================
st.download_button(
    "⬇️ Download Comparison CSV",
    final_df.to_csv(index=False),
    f"max_pain_{t1_lbl}_{t2_lbl}_{t3_lbl}.csv",
    "text/csv",
)

# =====================================
# SINGLE STOCK VIEW
# =====================================
st.subheader("🔍 View Individual Stock")

stock_list = sorted(final_df["Stock"].dropna().unique().tolist())

selected_stock = st.selectbox("Select Stock", [""] + stock_list)

if selected_stock:
    stock_df = final_df[final_df["Stock"] == selected_stock]

    st.dataframe(
        stock_df[display_cols]
        .style.apply(highlight_rows, axis=None)
        .format(
            {c: "{:.3f}" if c == pct_col else "{:.2f}" if c == "Stock_LTP" else "{:.0f}"
             for c in display_cols if c not in {"Stock", "Sector"}},
            na_rep=""
        ),
        use_container_width=True,
    )

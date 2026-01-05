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
pressure_ratio_col = "Abs Above/Below ΔΔ MP Ratio (±6)"
delta_above_23_col = f"ΔΔ MP ({t2_lbl}-{t3_lbl})"
sum_2_above_below_col = f"Σ |ΔΔ MP| (±2)"
diff_2_above_below_col = f"Δ (ΔΔ MP) (±2)"




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
    columns={
        "Max_Pain": mp2_col,
        "Stock_LTP": "Stock_LTP_t2"
    }
)

df3 = df3[["Stock", "Strike", "Max_Pain"]].rename(columns={"Max_Pain": mp3_col})

df = df1.merge(df2, on=["Stock", "Strike"]).merge(df3, on=["Stock", "Strike"])

# =====================================
# % CHANGE IN STOCK LTP (t1 vs t2)
# =====================================

ltp_pct_12_col = f"% LTP Ch ({t1_lbl}-{t2_lbl})"

df[ltp_pct_12_col] = (
    (df["Stock_LTP"] - df["Stock_LTP_t2"]) / df["Stock_LTP_t2"]
) * 100

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


# =====================================================
# Σ7 ABOVE / BELOW LTP (REPEATED PER STOCK) + DIFFERENCE
# =====================================================

sum7_below_12_col = "Σ7 Δ MP Below LTP (t1–t2)"
sum7_above_12_col = "Σ7 Δ MP Above LTP (t1–t2)"
sum7_below_23_col = "Σ7 Δ MP Below LTP (t2–t3)"
sum7_above_23_col = "Σ7 Δ MP Above LTP (t2–t3)"

diff7_above_col = "Δ Σ7 Above (t1–t2 minus t2–t3)"
diff7_below_col = "Δ Σ7 Below (t1–t2 minus t2–t3)"

# initialize
for col in [
    sum7_below_12_col,
    sum7_above_12_col,
    sum7_below_23_col,
    sum7_above_23_col,
    diff7_above_col,
    diff7_below_col,
]:
    df[col] = np.nan

for stock, sdf in df.sort_values("Strike").groupby("Stock"):
    sdf = sdf.reset_index(drop=True)

    ltp = float(sdf["Stock_LTP"].iloc[0])
    strikes = sdf["Strike"].values

    atm_idx = None
    for i in range(len(strikes) - 1):
        if strikes[i] <= ltp <= strikes[i + 1]:
            atm_idx = i
            break

    if atm_idx is None:
        continue

    below_idx = list(range(atm_idx - 7, atm_idx))
    above_idx = list(range(atm_idx + 1, atm_idx + 8))

    if min(below_idx) < 0 or max(above_idx) >= len(sdf):
        continue

    sum_below_12 = sdf.loc[below_idx, delta_12].astype(float).sum()
    sum_above_12 = sdf.loc[above_idx, delta_12].astype(float).sum()
    sum_below_23 = sdf.loc[below_idx, delta_23].astype(float).sum()
    sum_above_23 = sdf.loc[above_idx, delta_23].astype(float).sum()

    df.loc[df["Stock"] == stock, sum7_below_12_col] = sum_below_12
    df.loc[df["Stock"] == stock, sum7_above_12_col] = sum_above_12
    df.loc[df["Stock"] == stock, sum7_below_23_col] = sum_below_23
    df.loc[df["Stock"] == stock, sum7_above_23_col] = sum_above_23

    df.loc[df["Stock"] == stock, diff7_above_col] = (
        sum_above_12 - sum_above_23
    )
    df.loc[df["Stock"] == stock, diff7_below_col] = (
        sum_below_12 - sum_below_23
    )


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

df[delta_above_23_col] = np.nan
for stock, sdf in df.sort_values("Strike").groupby("Stock"):
    vals = sdf[delta_23].astype(float).values
    diff = vals - np.roll(vals, -1)
    diff[-1] = np.nan
    df.loc[sdf.index, delta_above_23_col] = diff


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

df[diff_2_above_below_col] = np.nan

for stock, sdf in df.sort_values("Strike").groupby("Stock"):
    sdf = sdf.reset_index(drop=True)
    ltp = float(sdf["Stock_LTP"].iloc[0])
    strikes = sdf["Strike"].values

    for i in range(len(strikes) - 1):
        if strikes[i] <= ltp <= strikes[i + 1]:
            below_val = sdf.loc[i, delta_above_col]
            above_val = sdf.loc[i + 1, delta_above_col]

            if pd.notna(below_val) and pd.notna(above_val):
                df.loc[df["Stock"] == stock, diff_2_above_below_col] = (
                    above_val - below_val
                )
            break

# =============================================
# DIRECTION PRESSURE RATIO
# =============================================
df[pressure_ratio_col] = np.nan

for stock, sdf in df.sort_values("Strike").groupby("Stock"):
    sdf = sdf.reset_index()
    ltp = float(sdf["Stock_LTP"].iloc[0])
    strikes = sdf["Strike"].values

    atm_idx = None
    for i in range(len(strikes) - 1):
        if strikes[i] <= ltp <= strikes[i + 1]:
            atm_idx = i
            break

    if atm_idx is None:
        continue

    below = sdf.iloc[max(0, atm_idx - 6):atm_idx]
    above = sdf.iloc[atm_idx + 2:atm_idx + 2 + 6]

    if len(below) < 6 or len(above) < 6:
        continue

    sum_below = below[delta_above_col].astype(float).sum()
    sum_above = above[delta_above_col].astype(float).sum()

    if sum_below == 0:
        continue

    ratio = abs(sum_above / sum_below)

    df.loc[sdf["index"], pressure_ratio_col] = ratio

# =====================================
# FINAL COLUMN ORDER
# =====================================
df = df[
    [
        "Stock",
        #"Sector",
        "Strike",
        mp1_col,
        mp2_col,
        mp3_col,
        delta_12,
        delta_23,
        sum7_below_12_col,
       sum7_above_12_col,
       sum7_below_23_col,
       sum7_above_23_col,
        diff7_above_col,
        diff7_below_col,
        sum_12_col,
        delta_above_col,
        delta_above_23_col,
         pressure_ratio_col,
        sum_2_above_below_col,
        diff_2_above_below_col,
        pct_col,
        "Stock_LTP",
        ltp_pct_12_col,
        
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
# 6 strike above - 6 strike below
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

        sliced = sdf.iloc[start:end]
        out.append(sliced)

        # blank row separator
        out.append(pd.DataFrame([{col: np.nan for col in df.columns}]))

    return pd.concat(out[:-1], ignore_index=True)

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

# =============================================
# DIRECTION PRESSURE
# =============================================
def detect_directional_pressure_stocks(df, delta_col, strikes_count=6, min_required=5):
    qualified_stocks = []

    for stock, sdf in df.groupby("Stock"):
        sdf = sdf.dropna(subset=["Strike", delta_col]).sort_values("Strike")
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

        below = sdf.iloc[max(0, atm_idx - strikes_count):atm_idx]
        above = sdf.iloc[atm_idx + 2:atm_idx + 2 + strikes_count]

        if len(below) < strikes_count or len(above) < strikes_count:
            continue

        below_vals = below[delta_col].astype(float)
        above_vals = above[delta_col].astype(float)

        below_pos = (below_vals > 0).sum()
        below_neg = (below_vals < 0).sum()
        above_pos = (above_vals > 0).sum()
        above_neg = (above_vals < 0).sum()

        # check dominance + opposite sign
        cond1 = above_pos >= min_required and below_neg >= min_required
        cond2 = above_neg >= min_required and below_pos >= min_required

        if cond1 or cond2:
            qualified_stocks.append(stock)

    return qualified_stocks


# =============================================
# DIRECTION PRESSURE (MIN-MAX) 4 strikes
# =============================================
def detect_extreme_imbalance_stocks(df, delta_col, strikes_count=4):
    qualified = []

    for stock, sdf in df.sort_values("Strike").groupby("Stock"):
        sdf = sdf.reset_index(drop=True)

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

        below = sdf.iloc[max(0, atm_idx - strikes_count):atm_idx]
        above = sdf.iloc[atm_idx + 2:atm_idx + 2 + strikes_count]

        if len(below) < strikes_count or len(above) < strikes_count:
            continue

        below_abs = below[delta_col].abs().astype(float)
        above_abs = above[delta_col].abs().astype(float)

        cond1 = above_abs.min() > below_abs.max()
        cond2 = below_abs.min() > above_abs.max()

        if cond1 or cond2:
            qualified.append(stock)

    return qualified

# =====================================
# PRE-COMPUTE SIGNAL STOCK LISTS (USED BY UNION)
# =====================================

qualified_stocks = detect_directional_pressure_stocks(
    final_df,
    delta_above_col,
    strikes_count=6,
    min_required=5
)

extreme_stocks = detect_extreme_imbalance_stocks(
    final_df,
    delta_above_col,
    strikes_count=4
)


# =============================================
# ATM ±3 ΔΔ MP INTENSITY FILTER (MODULAR)
# =============================================
def detect_atm_delta_intensity_stocks(
    df,
    delta_col,
    diff_threshold=300,
    sum_threshold=300,
    window=3
):
    qualified = []

    for stock, sdf in df.groupby("Stock"):
        sdf = sdf.dropna(subset=["Strike", delta_col]).sort_values("Strike")
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

        # Collect exactly 3 ΔΔ MP values: (i-1, i, i+1)
        idxs = [atm_idx - 1, atm_idx, atm_idx + 1]
        idxs = [i for i in idxs if 0 <= i < len(sdf)]

        if len(idxs) < window:
            continue

        deltas = sdf.iloc[idxs][delta_col].astype(float).values

        # FILTER 1: max absolute difference (ΔΔ MP already a difference)
        max_diff = np.max(np.abs(deltas))

        # FILTER 2: max absolute pairwise sum
        sums = [
            deltas[i] + deltas[i + 1]
            for i in range(len(deltas) - 1)
        ]
        max_sum = np.max(np.abs(sums)) if sums else 0

        if max_diff > diff_threshold or max_sum > sum_threshold:
            qualified.append(stock)

    return qualified


# =====================================
# DISPLAY
# =====================================
# =====================================
# COLUMNS TO HIDE FROM DISPLAY
# =====================================
display_cols = [c for c in final_df.columns if c != sum_12_col]



st.subheader(f"📊 ALL STOCKS: {t1_lbl} vs {t2_lbl} vs {t3_lbl}")
display_df = filter_strikes_around_ltp(final_df)

st.dataframe(
    display_df[display_cols]
    .style.apply(highlight_rows, axis=None)
    .format(
    {
        ltp_pct_12_col: "{:.2f}",
        pct_col: "{:.3f}",
        "Stock_LTP": "{:.2f}",
        **{
            c: "{:.0f}"
            for c in display_cols
            if c not in {
                "Stock",
                "Sector",
                pct_col,
                "Stock_LTP",
                ltp_pct_12_col,
            }
        },
    },
    na_rep="",
)

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
    stock_df = filter_strikes_around_ltp(
    final_df[final_df["Stock"] == selected_stock],
    below=6,
    above=6
)

    st.dataframe(
        stock_df[display_cols]
        .style.apply(highlight_rows, axis=None)
        .format(
            {
                ltp_pct_12_col: "{:.2f}",
                pct_col: "{:.3f}",
                "Stock_LTP": "{:.2f}",
                **{
                    c: "{:.0f}"
                    for c in display_cols
                    if c not in {
                        "Stock",
                        "Sector",
                        pct_col,
                        "Stock_LTP",
                        ltp_pct_12_col,
                    }
                },
            },
            na_rep="",
        )

        use_container_width=True,
    )



# =====================================
# UNION OF DIRECTIONAL + EXTREME FILTERS
# =====================================
st.subheader("🧩 Combined Signal Stocks (Directional OR Extreme ΔΔ MP)")

# Union of base signal stocks
union_stocks = sorted(set(qualified_stocks) | set(extreme_stocks))

# Apply ATM ±3 ΔΔ MP intensity filter (NEW)
intensity_stocks = detect_atm_delta_intensity_stocks(
    final_df,
    delta_above_col,
    diff_threshold=350,
    sum_threshold=350,
    window=3
)

# FINAL stocks for Combined Signal table
union_stocks = sorted(set(union_stocks) & set(intensity_stocks))



union_df = final_df[final_df["Stock"].isin(union_stocks)]

# Apply SAME ±6 strike display window
union_display_df = filter_strikes_around_ltp(union_df)

if union_display_df.empty:
    st.info("No stocks matched either of the ΔΔ MP filter conditions.")
else:
    st.dataframe(
        union_display_df[display_cols]
        .style.apply(highlight_rows, axis=None)
        .format(
                {
                    ltp_pct_12_col: "{:.2f}",
                    pct_col: "{:.3f}",
                    "Stock_LTP": "{:.2f}",
                    **{
                        c: "{:.0f}"
                        for c in display_cols
                        if c not in {
                            "Stock",
                            "Sector",
                            pct_col,
                            "Stock_LTP",
                            ltp_pct_12_col,
                        }
                    },
                },
                na_rep="",
            )

        use_container_width=True,
    )

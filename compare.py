import streamlit as st
import pandas as pd
import os

# =====================================
# STREAMLIT CONFIG
# =====================================
st.set_page_config(page_title="Max Pain Comparison", layout="wide")
st.title("📊 Max Pain Comparison Dashboard")

DATA_DIR = "data"

# =====================================
# LOAD CSV FILES (LATEST → OLDEST)
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
    st.error("Need at least 3 CSV files to compare.")
    st.stop()

timestamps = [ts for ts, _ in csv_files]
file_map = {ts: path for ts, path in csv_files}

# =====================================
# DROPDOWNS
# =====================================
col1, col2, col3 = st.columns(3)

with col1:
    t1 = st.selectbox("Select Timestamp 1 (Latest)", timestamps, index=0)

with col2:
    t2 = st.selectbox("Select Timestamp 2 (Middle)", timestamps, index=1)

with col3:
    t3 = st.selectbox("Select Timestamp 3 (Older)", timestamps, index=2)

# =====================================
# LOAD DATA
# =====================================
df1 = pd.read_csv(file_map[t1])
df2 = pd.read_csv(file_map[t2])
df3 = pd.read_csv(file_map[t3])

required_cols = {"Stock", "Strike", "Max_Pain", "Stock_LTP"}
if not required_cols.issubset(df1.columns) or not required_cols.issubset(df2.columns):
    st.error("CSV format mismatch.")
    st.stop()

# =====================================
# PREPARE COMPARISON DATA
# =====================================
df1 = df1[["Stock", "Strike", "Max_Pain", "Stock_LTP"]].rename(
    columns={"Max_Pain": f"Max_Pain_{t1}"}
)

df2 = df2[["Stock", "Strike", "Max_Pain"]].rename(
    columns={"Max_Pain": f"Max_Pain_{t2}"}
)

df3 = df3[["Stock", "Strike", "Max_Pain"]].rename(
    columns={"Max_Pain": f"Max_Pain_{t3}"}
)

compare_df = (
    df1
    .merge(df2, on=["Stock", "Strike"], how="inner")
    .merge(df3, on=["Stock", "Strike"], how="inner")
)

# Delta between t1 and t2 (existing)
compare_df["Delta_Max_Pain"] = (
    compare_df[f"Max_Pain_{t1}"] - compare_df[f"Max_Pain_{t2}"]
)

# NEW: Delta between t2 and t3
compare_df[f"Delta_Max_Pain_{t2}_minus_{t3}"] = (
    compare_df[f"Max_Pain_{t2}"] - compare_df[f"Max_Pain_{t3}"]
)

# =====================================
# FORMAT NUMBERS
# =====================================
compare_df["Strike"] = compare_df["Strike"].astype(int)

compare_df["Stock_LTP"] = (
    compare_df["Stock_LTP"]
    .astype(float)
    .round(1)
    .map(lambda x: f"{x:.1f}")
)

# Move Stock_LTP to LAST column
stock_ltp = compare_df.pop("Stock_LTP")
compare_df["Stock_LTP"] = stock_ltp

# =====================================
# INSERT BLANK ROW AFTER EACH STOCK
# =====================================
rows = []

for stock, sdf in compare_df.sort_values(["Stock", "Strike"]).groupby("Stock"):
    rows.append(sdf.assign(_spacer=False))

    blank = pd.DataFrame([{col: "" for col in compare_df.columns}])
    blank["Stock"] = stock
    blank["_spacer"] = True
    rows.append(blank)

final_df = pd.concat(rows, ignore_index=True)
final_df["_spacer"] = final_df["_spacer"].fillna(False)

# =====================================
# HIGHLIGHTING LOGIC
# =====================================
def highlight_rows(df):
    styles = pd.DataFrame("", index=df.index, columns=df.columns)

    for stock in df["Stock"].unique():
        sdf = df[(df["Stock"] == stock) & (~df["_spacer"])].sort_values("Strike")
        if sdf.empty:
            continue

        ltp = float(sdf["Stock_LTP"].iloc[0])
        strikes = sdf["Strike"].values

        below_idx = above_idx = None
        for i in range(len(strikes) - 1):
            if strikes[i] <= ltp <= strikes[i + 1]:
                below_idx = sdf.index[i]
                above_idx = sdf.index[i + 1]
                break

        # Max Pain based on Timestamp 1
        max_pain_idx = sdf[f"Max_Pain_{t1}"].idxmin()

        if below_idx is not None:
            styles.loc[below_idx] = "background-color: #003366; color: white"
        if above_idx is not None:
            styles.loc[above_idx] = "background-color: #003366; color: white"

        styles.loc[max_pain_idx] = "background-color: #8B0000; color: white"

    styles.loc[df["_spacer"]] = "background-color: white"
    return styles

# =====================================
# DISPLAY
# =====================================
st.subheader(f"Comparison: {t1} vs {t2} vs {t3}")

st.markdown(
    """
    <style>
    tr:has(td:empty) {
        height: 16px;
    }
    </style>
    """,
    unsafe_allow_html=True
)

styled_df = (
    final_df
    .style
    .apply(highlight_rows, axis=None)
    .hide(axis="columns", subset=["_spacer"])
)

st.dataframe(styled_df, use_container_width=True)

# =====================================
# DOWNLOAD
# =====================================
st.download_button(
    "⬇️ Download Comparison CSV",
    final_df.drop(columns=["_spacer"]).to_csv(index=False),
    f"max_pain_comparison_{t1}_vs_{t2}_vs_{t3}.csv",
    "text/csv"
)

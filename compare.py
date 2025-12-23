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
# LOAD CSV FILES (NO CACHING – IMPORTANT)
# =====================================
def load_csv_files():
    files = []
    if not os.path.exists(DATA_DIR):
        return files

    for f in os.listdir(DATA_DIR):
        if f.lower().endswith(".csv") and f.startswith("option_chain_"):
            ts = f.replace("option_chain_", "").replace(".csv", "")
            files.append((ts, os.path.join(DATA_DIR, f)))

    return sorted(files)

csv_files = load_csv_files()

# DEBUG (safe to keep)
st.write("Detected CSV snapshots:", [ts for ts, _ in csv_files])

# =====================================
# VALIDATION
# =====================================
if len(csv_files) < 2:
    st.error("Need at least 2 CSV files to compare.")
    st.stop()

timestamps = [ts for ts, _ in csv_files]
file_map = {ts: path for ts, path in csv_files}

# =====================================
# DROPDOWNS
# =====================================
col1, col2 = st.columns(2)

with col1:
    t1 = st.selectbox(
        "Select Timestamp 1",
        timestamps,
        index=len(timestamps) - 2
    )

with col2:
    t2 = st.selectbox(
        "Select Timestamp 2",
        timestamps,
        index=len(timestamps) - 1
    )

# =====================================
# LOAD SELECTED CSVs
# =====================================
df1 = pd.read_csv(file_map[t1])
df2 = pd.read_csv(file_map[t2])

# =====================================
# REQUIRED COLUMNS CHECK
# =====================================
required_cols = {"Stock", "Strike", "Max_Pain", "Stock_LTP"}
if not required_cols.issubset(df1.columns) or not required_cols.issubset(df2.columns):
    st.error("CSV format mismatch. Required columns missing.")
    st.stop()

# =====================================
# PREPARE DATA
# =====================================
df1 = df1[["Stock", "Strike", "Stock_LTP", "Max_Pain"]].rename(
    columns={"Max_Pain": f"Max_Pain_{t1}"}
)

df2 = df2[["Stock", "Strike", "Max_Pain"]].rename(
    columns={"Max_Pain": f"Max_Pain_{t2}"}
)

# =====================================
# MERGE & COMPARE
# =====================================
compare_df = pd.merge(
    df1,
    df2,
    on=["Stock", "Strike"],
    how="inner"
)

compare_df["Delta_Max_Pain"] = (
    compare_df[f"Max_Pain_{t1}"] - compare_df[f"Max_Pain_{t2}"]
)

# =====================================
# DISPLAY
# =====================================
st.subheader(f"Comparison: {t1}  vs  {t2}")

st.dataframe(
    compare_df.sort_values(["Stock", "Strike"]),
    use_container_width=True
)

# =====================================
# DOWNLOAD
# =====================================
st.download_button(
    label="⬇️ Download Comparison CSV",
    data=compare_df.to_csv(index=False),
    file_name=f"max_pain_comparison_{t1}_vs_{t2}.csv",
    mime="text/csv"
)

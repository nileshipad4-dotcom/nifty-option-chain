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

timestamps = [ts for ts, _ in csv_files]
file_map = {ts: path for ts, path in csv_files}

def short_ts(ts):
    return ts.split("_")[-1].replace("-", ":")

# =====================================
# DROPDOWNS
# =====================================
col1, col2, col3 = st.columns(3)

with col1:
    t1 = st.selectbox("Timestamp 1 (Latest)", timestamps, 0)
with col2:
    t2 = st.selectbox("Timestamp 2", timestamps, 1)
with col3:
    t3 = st.selectbox("Timestamp 3", timestamps, 2)

t1_lbl = short_ts(t1)
t2_lbl = short_ts(t2)
t3_lbl = short_ts(t3)

# =====================================
# LOAD DATA
# =====================================
df1 = pd.read_csv(file_map[t1])
df2 = pd.read_csv(file_map[t2])
df3 = pd.read_csv(file_map[t3])

# =====================================
# PREPARE DATA
# =====================================
df1 = df1[["Stock","Strike","Max_Pain","Stock_LTP"]].rename(
    columns={"Max_Pain": t1_lbl}
)

df2 = df2[["Stock","Strike","Max_Pain"]].rename(
    columns={"Max_Pain": t2_lbl}
)

df3 = df3[["Stock","Strike","Max_Pain"]].rename(
    columns={"Max_Pain": t3_lbl}
)

df = (
    df1
    .merge(df2, on=["Stock","Strike"])
    .merge(df3, on=["Stock","Strike"])
)

# Calculations
df["Δ MP 1"] = df[t1_lbl] - df[t2_lbl]
df["Δ MP 2"] = df[t2_lbl] - df[t3_lbl]

# Duplicate TS2 column (explicit copy)
df[f"{t2_lbl} (ref)"] = df[t2_lbl]

# =====================================
# FORMAT
# =====================================
df["Strike"] = df["Strike"].astype(int)
df["Stock_LTP"] = df["Stock_LTP"].astype(float).round(1)

# =====================================
# FINAL COLUMN ORDER (EXACT)
# =====================================
df = df[
    [
        "Stock",
        "Strike",
        t1_lbl,
        t2_lbl,
        "Δ MP 1",
        f"{t2_lbl} (ref)",
        t3_lbl,
        "Δ MP 2",
        "Stock_LTP",
    ]
]

# =====================================
# HIGHLIGHTING
# =====================================
def highlight(df):
    styles = pd.DataFrame("", index=df.index, columns=df.columns)

    for stock in df["Stock"].unique():
        sdf = df[df["Stock"] == stock].sort_values("Strike")
        if sdf.empty:
            continue

        ltp = sdf["Stock_LTP"].iloc[0]
        strikes = sdf["Strike"].values

        for i in range(len(strikes) - 1):
            if strikes[i] <= ltp <= strikes[i + 1]:
                styles.loc[sdf.index[i]] = "background-color:#003366;color:white"
                styles.loc[sdf.index[i+1]] = "background-color:#003366;color:white"
                break

        mp_idx = sdf[t1_lbl].idxmin()
        styles.loc[mp_idx] = "background-color:#8B0000;color:white"

    return styles

# =====================================
# DISPLAY
# =====================================
st.subheader(f"Comparison: {t1_lbl} vs {t2_lbl} vs {t3_lbl}")
st.dataframe(df.style.apply(highlight, axis=None), use_container_width=True)

# =====================================
# DOWNLOAD
# =====================================
st.download_button(
    "⬇️ Download CSV",
    df.to_csv(index=False),
    f"max_pain_{t1_lbl}_{t2_lbl}_{t3_lbl}.csv",
    "text/csv"
)

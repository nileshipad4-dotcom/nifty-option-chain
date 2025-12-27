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

    for fname in os.listdir(DATA_DIR):
        if fname.startswith("option_chain_") and fname.endswith(".csv"):
            ts = fname.replace("option_chain_", "").replace(".csv", "")
            files.append((ts, os.path.join(DATA_DIR, fname)))

    return sorted(files, reverse=True)

csv_files = load_csv_files()
if len(csv_files) < 3:
    st.error("Need at least 3 CSV files to compare.")
    st.stop()

timestamps = [ts for ts, _ in csv_files]
file_map = dict(csv_files)

def short_ts(ts):
    return ts.split("_")[-1].replace("-", ":")

# =====================================
# DROPDOWNS
# =====================================
c1, c2, c3 = st.columns(3)
with c1:
    t1 = st.selectbox("Timestamp 1 (Latest)", timestamps, index=0)
with c2:
    t2 = st.selectbox("Timestamp 2", timestamps, index=1)
with c3:
    t3 = st.selectbox("Timestamp 3", timestamps, index=2)

t1_lbl = short_ts(t1)
t2_lbl = short_ts(t2)
t3_lbl = short_ts(t3)

# =====================================
# LOAD DATA
# =====================================
df1 = pd.read_csv(file_map[t1])
df2 = pd.read_csv(file_map[t2])
df3 = pd.read_csv(file_map[t3])

required_cols = {"Stock", "Strike", "Max_Pain", "Stock_LTP"}
if not (
    required_cols.issubset(df1.columns)
    and required_cols.issubset(df2.columns)
    and required_cols.issubset(df3.columns)
):
    st.error("CSV format mismatch.")
    st.stop()

# =====================================
# PREPARE DATA (HH:MM HEADERS)
# =====================================
df1 = df1[["Stock", "Strike", "Max_Pain", "Stock_LTP"]].rename(
    columns={"Max_Pain": t1_lbl}
)
df2 = df2[["Stock", "Strike", "Max_Pain"]].rename(
    columns={"Max_Pain": t2_lbl}
)
df3 = df3[["Stock", "Strike", "Max_Pain"]].rename(
    columns={"Max_Pain": t3_lbl}
)

df = (
    df1
    .merge(df2, on=["Stock", "Strike"])
    .merge(df3, on=["Stock", "Strike"])
)

# =====================================
# DELTA CALCULATIONS (FIXED)
# =====================================
delta_12 = f"Δ MP ({t1_lbl}-{t2_lbl})"
delta_23 = f"Δ MP ({t2_lbl}-{t3_lbl})"

df[delta_12] = df[t1_lbl] - df[t2_lbl]
df[delta_23] = df[t2_lbl] - df[t3_lbl]

# =====================================
# FORMAT
# =====================================
df["Strike"] = df["Strike"].astype(int)
df["Stock_LTP"] = df["Stock_LTP"].astype(float).round(1)

# Move Stock_LTP to last
stock_ltp = df.pop("Stock_LTP")
df["Stock_LTP"] = stock_ltp

# =====================================
# FINAL COLUMN ORDER
# =====================================
df = df[
    [
        "Stock",
        "Strike",
        t1_lbl,
        t2_lbl,
        delta_12,
        t3_lbl,
        delta_23,
        "Stock_LTP",
    ]
]

# =====================================
# SPACER ROWS
# =====================================
rows = []
for stock, sdf in df.sort_values(["Stock", "Strike"]).groupby("Stock"):
    rows.append(sdf.assign(_spacer=False))

    blank = pd.DataFrame([{col: "" for col in df.columns}])
    blank["Stock"] = stock
    blank["_spacer"] = True
    rows.append(blank)

final_df = pd.concat(rows, ignore_index=True)
final_df["_spacer"] = final_df["_spacer"].fillna(False)

# =====================================
# HIGHLIGHTING
# =====================================
def highlight_rows(data):
    styles = pd.DataFrame("", index=data.index, columns=data.columns)

    for stock in data["Stock"].unique():
        sdf = data[(data["Stock"] == stock) & (~data["_spacer"])].sort_values("Strike")
        if sdf.empty:
            continue

        ltp = float(sdf["Stock_LTP"].iloc[0])
        strikes = sdf["Strike"].values

        for i in range(len(strikes) - 1):
            if strikes[i] <= ltp <= strikes[i + 1]:
                styles.loc[sdf.index[i]] = "background-color:#003366;color:white"
                styles.loc[sdf.index[i + 1]] = "background-color:#003366;color:white"
                break

        styles.loc[sdf[t1_lbl].idxmin()] = "background-color:#8B0000;color:white"

    styles.loc[data["_spacer"]] = "background-color: white"
    return styles

# =====================================
# DISPLAY
# =====================================
st.subheader(f"Comparison: {t1_lbl} vs {t2_lbl} vs {t3_lbl}")

st.markdown(
    """
    <style>
    tr:has(td:empty) {
        height: 16px;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

styled = final_df.style.apply(highlight_rows, axis=None)
styled = styled.hide(axis="columns", subset=["_spacer"])

st.dataframe(styled, use_container_width=True)

# =====================================
# DOWNLOAD
# =====================================
st.download_button(
    "⬇️ Download Comparison CSV",
    final_df.drop(columns=["_spacer"]).to_csv(index=False),
    f"max_pain_{t1_lbl}_{t2_lbl}_{t3_lbl}.csv",
    "text/csv",
)

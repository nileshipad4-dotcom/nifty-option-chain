# stock OC main
import streamlit as st
import pandas as pd
import os
import numpy as np

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
    t1 = st.selectbox("Timestamp 1 (Latest)", timestamps, 0)
with c2:
    t2 = st.selectbox("Timestamp 2", timestamps, 1)
with c3:
    t3 = st.selectbox("Timestamp 3", timestamps, 2)

t1_lbl = short_ts(t1)
t2_lbl = short_ts(t2)
t3_lbl = short_ts(t3)

# =====================================
# COLUMN NAMES
# =====================================
mp1_col = f"MP ({t1_lbl})"
mp2_col = f"MP ({t2_lbl})"
mp3_col = f"MP ({t3_lbl})"

delta_12 = f"Δ MP ({t1_lbl}-{t2_lbl})"
delta_23 = f"Δ MP ({t2_lbl}-{t3_lbl})"
sum_12_col = f"Σ {delta_12}"
delta_above_col = f"ΔΔ MP ({t1_lbl}-{t2_lbl})"
pct_col = f"% Ch ({t1_lbl})"

# =====================================
# LOAD DATA
# =====================================
df1 = pd.read_csv(file_map[t1])
df2 = pd.read_csv(file_map[t2])
df3 = pd.read_csv(file_map[t3])

# =====================================
# PREPARE DATA
# =====================================
df1 = df1[
    ["Stock", "Strike", "Max_Pain", "Stock_LTP", "Stock_%_Change"]
].rename(
    columns={
        "Max_Pain": mp1_col,
        "Stock_%_Change": pct_col
    }
)

df2 = df2[["Stock", "Strike", "Max_Pain"]].rename(columns={"Max_Pain": mp2_col})
df3 = df3[["Stock", "Strike", "Max_Pain"]].rename(columns={"Max_Pain": mp3_col})

df = (
    df1.merge(df2, on=["Stock", "Strike"])
       .merge(df3, on=["Stock", "Strike"])
)

df[delta_12] = df[mp1_col] - df[mp2_col]
df[delta_23] = df[mp2_col] - df[mp3_col]

# =====================================
# Σ Δ MP (trend logic only)
# =====================================
df[sum_12_col] = np.nan
for stock, sdf in df.sort_values("Strike").groupby("Stock"):
    df.loc[sdf.index, sum_12_col] = (
        sdf[delta_12]
        .rolling(window=7, center=True, min_periods=1)
        .sum()
        .values
    )

# =====================================
# ΔΔ MP (current − strike above)
# =====================================
df[delta_above_col] = np.nan
for stock, sdf in df.sort_values("Strike").groupby("Stock"):
    vals = sdf[delta_12].astype(float).values
    diff = vals - np.roll(vals, -1)
    diff[-1] = np.nan
    df.loc[sdf.index, delta_above_col] = diff

# =====================================
# FINAL COLUMN ORDER
# =====================================
df = df[
    [
        "Stock",
        "Strike",
        mp1_col,
        mp2_col,
        mp3_col,
        delta_12,
        delta_23,
        sum_12_col,      # hidden later
        delta_above_col,
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
# COMPUTE STOCK SIGNALS (UNCHANGED)
# =====================================
def compute_stock_signals(data):
    signals = {}
    for stock in data["Stock"].dropna().unique():
        sdf = data[(data["Stock"] == stock) & data["Strike"].notna()].sort_values("Strike")
        if len(sdf) < 9:
            continue

        ltp = float(sdf["Stock_LTP"].iloc[0])
        strikes = sdf["Strike"].values

        atm_idx = None
        for i in range(len(strikes) - 1):
            if strikes[i] <= ltp <= strikes[i + 1]:
                atm_idx = i + 1
                break
        if atm_idx is None:
            continue

        mid = sdf.iloc[4:-4]
        diffs = np.diff(mid[sum_12_col].astype(float).values)
        if len(diffs) == 0:
            continue

        inc_ratio = np.sum(diffs > 0) / len(diffs)
        dec_ratio = np.sum(diffs < 0) / len(diffs)

        trend = "red" if inc_ratio >= 0.9 else "green" if dec_ratio >= 0.9 else None
        if trend is None:
            continue

        above = sdf.iloc[atm_idx:atm_idx + 5]
        below = sdf.iloc[max(atm_idx - 5, 0):atm_idx]
        if len(above) < 3 or len(below) < 3:
            continue

        if (
            (above[delta_12] > above[delta_23]).sum() >= 3 and
            (below[delta_12] < below[delta_23]).sum() >= 3
        ):
            signal = "red"
        elif (
            (above[delta_12] < above[delta_23]).sum() >= 3 and
            (below[delta_12] > below[delta_23]).sum() >= 3
        ):
            signal = "green"
        else:
            continue

        if signal == trend:
            signals[stock] = signal

    return signals

stock_signals = compute_stock_signals(final_df)

# =====================================
# FILTERED TABLES
# =====================================
def build_filtered_df(base_df, stocks):
    blocks = []
    for s in stocks:
        sdf = base_df[base_df["Stock"] == s]
        if not sdf.empty:
            blocks.append(sdf)
            blocks.append(pd.DataFrame([{col: np.nan for col in base_df.columns}]))
    return pd.concat(blocks[:-1], ignore_index=True) if blocks else base_df.iloc[0:0]

green_df = build_filtered_df(final_df, [s for s, v in stock_signals.items() if v == "green"])
red_df   = build_filtered_df(final_df, [s for s, v in stock_signals.items() if v == "red"])

# =====================================
# HIGHLIGHTING (UNCHANGED)
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

        styles.loc[sdf[mp1_col].idxmin()] = "background-color:#8B0000;color:white"

        if stock not in stock_signals:
            continue

        color = "background-color:#8B0000;color:white" if stock_signals[stock] == "red" else "background-color:#004d00;color:white"
        styles.loc[sdf.index, delta_12] = color
        styles.loc[sdf.index, delta_23] = color

    return styles

# =====================================
# DISPLAY (hide Σ Δ MP)
# =====================================
display_cols = [c for c in final_df.columns if c != sum_12_col]

def show(title, df_):
    st.subheader(title)
    st.dataframe(
        df_[display_cols]
        .style.apply(highlight_rows, axis=None)
        .format(
            {c: "{:.3f}" if c == pct_col else "{:.2f}" if c == "Stock_LTP" else "{:.0f}"
             for c in display_cols if c != "Stock"},
            na_rep=""
        ),
        use_container_width=True,
    )

show(f"🟢 UPTREND ({len(green_df['Stock'].dropna().unique())})", green_df)
show(f"🔴 DOWNTREND ({len(red_df['Stock'].dropna().unique())})", red_df)
show(f"📊 ALL STOCKS: {t1_lbl} vs {t2_lbl} vs {t3_lbl}", final_df)

# =====================================
# DOWNLOAD
# =====================================
st.download_button(
    "⬇️ Download Comparison CSV",
    final_df.to_csv(index=False),
    f"max_pain_{t1_lbl}_{t2_lbl}_{t3_lbl}.csv",
    "text/csv",
)

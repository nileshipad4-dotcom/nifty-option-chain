import streamlit as st
import pandas as pd
from datetime import datetime, timedelta, time
import os
from streamlit_autorefresh import st_autorefresh

# ==================================================
# CONFIG
# ==================================================
DATA_INDEX_DIR = "data_index"

st.set_page_config(layout="wide", page_title="Index OI Window Scanner")
st.title("📊 Index OI Window Scanner (NIFTY / BANKNIFTY / MIDCPNIFTY)")

st_autorefresh(interval=60_000, key="ui_refresh")

MARKET_START = time(9, 0)
MARKET_END   = time(16, 0)

# ==================================================
# LOAD MULTI SNAPSHOT CSVs
# ==================================================

def list_snapshot_files():
    return sorted(
        f for f in os.listdir(DATA_INDEX_DIR)
        if f.startswith("index_OC_") and f.endswith(".csv")
    )

def load_all_snapshots(symbol):
    rows = []

    for f in list_snapshot_files():
        path = os.path.join(DATA_INDEX_DIR, f)
        df = pd.read_csv(path)

        df = df[df["Symbol"] == symbol].copy()
        if df.empty:
            continue

        if "timestamp" in df.columns:
            ts = pd.to_datetime(df["timestamp"].iloc[0])
        else:
            ts = f.replace("index_OC_", "").replace(".csv", "")
            ts = datetime.strptime(ts, "%Y-%m-%d_%H-%M")

        if not (MARKET_START <= ts.time() <= MARKET_END):
            continue

        df["timestamp"] = ts
        rows.append(df)

    if not rows:
        return pd.DataFrame()

    df_all = pd.concat(rows, ignore_index=True)
    df_all = df_all.sort_values("timestamp").reset_index(drop=True)
    df_all["_row"] = range(len(df_all))
    return df_all

# ==================================================
# WINDOW LOGIC
# ==================================================

def build_all_windows(df, min_gap):
    times = (
        df.sort_values("_row")
        .drop_duplicates("timestamp")["timestamp"]
        .tolist()
    )

    windows, i = [], 0
    while i < len(times) - 1:
        t1 = times[i]
        target = t1 + timedelta(minutes=min_gap)
        t2 = next((t for t in times[i + 1:] if t >= target), None)
        if t2 is None:
            break
        windows.append((t1, t2))
        i = times.index(t2)

    return windows

def build_row(df, t1, t2, is_live=False):
    d1 = df[df["timestamp"] == t1]
    d2 = df[df["timestamp"] == t2]

    if d1.empty or d2.empty:
        return None

    m = pd.merge(d1, d2, on="Strike", suffixes=("_1", "_2"))

    strikes = sorted(m["Strike"].unique())
    if len(strikes) <= 4:
        return None
    m = m[m["Strike"].isin(strikes[2:-2])]

    # 🔢 OI DELTA (DIVIDE BY 100)
    m["CE"] = (m["CE_OI_2"] - m["CE_OI_1"]) // 100
    m["PE"] = (m["PE_OI_2"] - m["PE_OI_1"]) // 100

    ce = m.sort_values("CE", ascending=False)
    pe = m.sort_values("PE", ascending=False)

    if len(ce) < 2 or len(pe) < 2:
        return None

    sum_ce = int(m["CE"].sum())
    sum_pe = int(m["PE"].sum())
    diff = sum_pe - sum_ce

    label = f"{t1:%H:%M} - {t2:%H:%M}"
    if is_live:
        label += " ⏳"

    return {
        "TIME": label,

        "MAX CE 1": f"{int(ce.iloc[0].Strike)}:- {int(ce.iloc[0].CE)}",
        "MAX CE 2": f"{int(ce.iloc[1].Strike)}:- {int(ce.iloc[1].CE)}",
        "Σ ΔCE OI": sum_ce,

        "MAX PE 1": f"{int(pe.iloc[0].Strike)}:- {int(pe.iloc[0].PE)}",
        "MAX PE 2": f"{int(pe.iloc[1].Strike)}:- {int(pe.iloc[1].PE)}",
        "Σ ΔPE OI": sum_pe,

        "Δ (PE − CE)": diff,

        "_ce1": ce.iloc[0].CE,
        "_ce2": ce.iloc[1].CE,
        "_pe1": pe.iloc[0].PE,
        "_pe2": pe.iloc[1].PE,
    }

def process_windows(df, min_gap):
    rows = []
    windows = build_all_windows(df, min_gap)

    for t1, t2 in windows:
        r = build_row(df, t1, t2, False)
        if r:
            rows.append(r)

    if windows:
        live_start = windows[-1][1]
    else:
        live_start = df["timestamp"].min()

    live_end = df["timestamp"].max()

    if live_end >= live_start:
        r = build_row(df, live_start, live_end, True)
        if r:
            rows.append(r)

    return pd.DataFrame(rows)

# ==================================================
# STYLING
# ==================================================

def highlight_table(df):
    display_cols = [
        "TIME",
        "MAX CE 1", "MAX CE 2", "Σ ΔCE OI",
        "MAX PE 1", "MAX PE 2", "Σ ΔPE OI",
        "Δ (PE − CE)",
    ]

    styles = pd.DataFrame("", index=df.index, columns=display_cols)

    # 🔶 Highlight MAX columns (top-2 magnitude)
    for col, num_col in [
        ("MAX CE 1", "_ce1"),
        ("MAX CE 2", "_ce2"),
        ("MAX PE 1", "_pe1"),
        ("MAX PE 2", "_pe2"),
    ]:
        vals = df[num_col].abs()
        if len(vals) < 2:
            continue

        top1, top2 = vals.nlargest(2).values

        for i, v in vals.items():
            if v == top1:
                styles.loc[i, col] = "background-color:#ff4d4d;color:white;font-weight:bold"
            elif v == top2:
                styles.loc[i, col] = "background-color:#ffa500;font-weight:bold"

    # ⏱ TIME highlight conditions
    for i in df.index:
        ce2 = abs(df.loc[i, "_ce2"])
        pe1 = abs(df.loc[i, "_pe1"])
        pe2 = abs(df.loc[i, "_pe2"])
        ce1 = abs(df.loc[i, "_ce1"])
        diff = df.loc[i, "Δ (PE − CE)"]

        if ce2 > pe1 * 1.2 and diff < 0:
            styles.loc[i, "TIME"] = "background-color:#d32f2f;color:white;font-weight:bold"

        elif pe2 > ce1 * 1.2 and diff > 0:
            styles.loc[i, "TIME"] = "background-color:#2e7d32;color:white;font-weight:bold"

    return df[display_cols].style.apply(lambda _: styles, axis=None)

# ==================================================
# UI
# ==================================================

_, c_gap = st.columns([3, 1])

with c_gap:
    min_gap = st.selectbox(
        "Min Gap (minutes)",
        [5, 10, 15, 20, 30, 45, 60],
        index=2
    )

st.divider()

for symbol in ["NIFTY", "BANKNIFTY", "MIDCPNIFTY"]:

    df_all = load_all_snapshots(symbol)

    if df_all.empty:
        st.warning(f"No valid data for {symbol}")
        continue

    df_main = process_windows(df_all, min_gap)

    st.subheader(symbol)
    st.dataframe(
        highlight_table(df_main),
        use_container_width=True
    )

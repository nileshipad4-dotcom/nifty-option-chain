import streamlit as st
import pandas as pd
from datetime import datetime, timedelta
import os
from streamlit_autorefresh import st_autorefresh

# ==================================================
# CONFIG
# ==================================================
DATA_INDEX_DIR = "data_index"
DEFAULT_GAP = 15

st.set_page_config(layout="wide", page_title="Index OI Window Scanner")
st.title("📊 NIFTY / BANKNIFTY / MIDCPNIFTY – OI Window Scanner")

# UI refresh every 1 minute
st_autorefresh(interval=60_000, key="ui_refresh")

# ==================================================
# HELPERS
# ==================================================

def get_index_files():
    files = []
    for f in os.listdir(DATA_INDEX_DIR):
        if f.startswith("index_OC_") and f.endswith(".csv"):
            files.append(f)
    return sorted(files)

def load_index_data(filename, symbol):
    df = pd.read_csv(os.path.join(DATA_INDEX_DIR, filename))

    # Filter only selected index
    df = df[df["Symbol"] == symbol].copy()
    if df.empty:
        return df

    # Timestamp handling
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df["_row"] = range(len(df))
    return df

def build_windows(df, min_gap):
    times = (
        df.sort_values("_row")
        .drop_duplicates("timestamp")["timestamp"]
        .tolist()
    )

    windows, i = [], 0
    while i < len(times) - 1:
        t1 = times[i]
        target = t1 + timedelta(minutes=min_gap)
        t2 = next((t for t in times[i+1:] if t >= target), None)
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

    m["CE"] = m["CE_OI_2"] - m["CE_OI_1"]
    m["PE"] = m["PE_OI_2"] - m["PE_OI_1"]

    ce = m.sort_values("CE", ascending=False)
    pe = m.sort_values("PE", ascending=False)

    if len(ce) < 2 or len(pe) < 2:
        return None

    sum_ce = int(m["CE"].sum() / 100)
    sum_pe = int(m["PE"].sum() / 100)
    diff = sum_pe - sum_ce   # PE − CE

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
        "_is_live": is_live,
    }

def process_windows(df, min_gap):
    rows = []
    windows = build_windows(df, min_gap)

    for t1, t2 in windows:
        r = build_row(df, t1, t2, is_live=False)
        if r:
            rows.append(r)

    if windows:
        live_start = windows[-1][1]
    else:
        live_start = df["timestamp"].min()

    live_end = df["timestamp"].max()

    if live_end >= live_start:
        r = build_row(df, live_start, live_end, is_live=True)
        if r:
            rows.append(r)

    return pd.DataFrame(rows)

# ==================================================
# HIGHLIGHTING
# ==================================================

def highlight_table(df):
    display_cols = [
        "TIME",
        "MAX CE 1", "MAX CE 2", "Σ ΔCE OI",
        "MAX PE 1", "MAX PE 2", "Σ ΔPE OI",
        "Δ (PE − CE)",
    ]

    styles = pd.DataFrame("", index=df.index, columns=display_cols)

    for col, raw in [
        ("MAX CE 1", "_ce1"), ("MAX CE 2", "_ce2"),
        ("MAX PE 1", "_pe1"), ("MAX PE 2", "_pe2"),
    ]:
        vals = df[raw].abs()
        if len(vals) < 2:
            continue
        t1, t2 = vals.nlargest(2).values
        for i, v in vals.items():
            if v == t1:
                styles.loc[i, col] = "background-color:#ffa500;color:white;font-weight:bold"
            elif v == t2:
                styles.loc[i, col] = "background-color:#ff4d4d;font-weight:bold"

    for i in df.index:
        d = df.loc[i, "Δ (PE − CE)"]
        color = "green" if d > 0 else "red" if d < 0 else "black"
        for c in ["Σ ΔCE OI", "Σ ΔPE OI", "Δ (PE − CE)"]:
            styles.loc[i, c] = f"color:{color};font-weight:bold"

    return df[display_cols].style.apply(lambda _: styles, axis=None)

# ==================================================
# UI CONTROLS
# ==================================================

files = get_index_files()
if not files:
    st.error("No index CSV files found")
    st.stop()

c1, c2, c3 = st.columns([2, 2, 1])

with c1:
    index_symbol = st.selectbox(
        "Select Index",
        ["NIFTY", "BANKNIFTY", "MIDCPNIFTY"]
    )

with c2:
    file_selected = st.selectbox(
        "Select Snapshot File",
        files,
        index=len(files) - 1
    )

with c3:
    min_gap = st.selectbox(
        "Min Gap (min)",
        [5, 10, 15, 20, 30, 45, 60],
        index=2
    )

st.divider()

# ==================================================
# MAIN TABLE
# ==================================================

df_raw = load_index_data(file_selected, index_symbol)

if df_raw.empty:
    st.warning("No data for selected index in this snapshot")
    st.stop()

df_main = process_windows(df_raw, min_gap)

st.subheader(f"{index_symbol} — OI Window Table")
st.dataframe(highlight_table(df_main), use_container_width=True)

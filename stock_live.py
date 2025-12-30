# =====================================
# STOCK OC MAIN – HISTORICAL + LIVE MP
# =====================================

import streamlit as st
import pandas as pd
import numpy as np
import os
from datetime import datetime
from streamlit_autorefresh import st_autorefresh
from kiteconnect import KiteConnect
import pytz


# =====================================
# AUTO REFRESH (5 MIN)
# =====================================
st_autorefresh(interval=360_000, key="auto_refresh")

# =====================================
# STREAMLIT CONFIG
# =====================================
st.set_page_config(page_title="Max Pain – Live + Historical", layout="wide")
st.title("📊 FnO STOCKS – Live Max Pain Dashboard")

DATA_DIR = "data"
IST = pytz.timezone("Asia/Kolkata")

# =====================================
# KITE CONFIG
# =====================================
API_KEY = "bkgv59vaazn56c42"
ACCESS_TOKEN = "HwNfTAk4E3mk2B11MPBFC87FxrVBnvqp"

@st.cache_resource
def init_kite():
    k = KiteConnect(api_key=API_KEY)
    k.set_access_token(ACCESS_TOKEN)
    return k

kite = init_kite()

@st.cache_data(ttl=300)
def load_instruments():
    df = pd.DataFrame(kite.instruments("NFO"))
    df["expiry"] = pd.to_datetime(df["expiry"])
    return df

instruments = load_instruments()

# =====================================
# HELPERS
# =====================================
def chunk_list(lst, size=200):
    for i in range(0, len(lst), size):
        yield lst[i:i + size]

# =====================================
# LOAD LATEST CSV
# =====================================
def load_latest_csv():
    files = [
        f for f in os.listdir(DATA_DIR)
        if f.startswith("option_chain_") and f.endswith(".csv")
    ]
    if not files:
        return None, None
    files.sort(reverse=True)
    ts = files[0].replace("option_chain_", "").replace(".csv", "")
    return ts, os.path.join(DATA_DIR, files[0])

ts, csv_path = load_latest_csv()
if not csv_path:
    st.error("No option_chain CSV found.")
    st.stop()

t_lbl = ts.split("_")[-1].replace("-", ":")

mp1_col = f"MP ({t_lbl})"
live_mp_col = f"MP ({datetime.now(IST).strftime('%H:%M')})"
live_delta_col = f"Δ Live MP (Live - {t_lbl})"
delta_live_above_col = "ΔΔ Live MP"
sum_live_col = "Σ |ΔΔ Live MP| (±2)"
pct_col = "Live % Change"

# =====================================
# LOAD HISTORICAL DATA
# =====================================
df_hist = pd.read_csv(csv_path)[["Stock", "Strike", "Max_Pain", "Stock_LTP"]]
df_hist = df_hist.rename(columns={"Max_Pain": mp1_col})

# =====================================
# LIVE MAX PAIN LOGIC
# =====================================
def compute_live_max_pain(df):
    df = df.fillna(0)
    A, B, G, M, L = (
        df["CE_LTP"], df["CE_OI"],
        df["Strike"], df["PE_LTP"], df["PE_OI"]
    )

    mp = []
    for i in range(len(df)):
        val = (
            -sum(A[i:] * B[i:])
            + G.iloc[i] * sum(B[:i]) - sum(G[:i] * B[:i])
            - sum(M[:i] * L[:i])
            + sum(G[i:] * L[i:]) - G.iloc[i] * sum(L[i:])
        )
        mp.append(int(val / 10000))

    df[live_mp_col] = mp
    return df

@st.cache_data(ttl=300)
def fetch_live_data(stocks):
    rows = []

    # ---- Spot quotes (chunked) ----
    spot_quotes = {}
    for batch in chunk_list(stocks, 200):
        spot_quotes.update(kite.quote([f"NSE:{s}" for s in batch]))

    for stock in stocks:
        opt_df = instruments[
            (instruments["name"] == stock) &
            (instruments["segment"] == "NFO-OPT")
        ]
        if opt_df.empty:
            continue

        expiry = opt_df["expiry"].min()
        opt_df = opt_df[opt_df["expiry"] == expiry]

        option_quotes = {}
        symbols = ["NFO:" + s for s in opt_df["tradingsymbol"].tolist()]
        for batch in chunk_list(symbols, 200):
            option_quotes.update(kite.quote(batch))

        chain = []
        for strike in sorted(opt_df["strike"].unique()):
            ce = opt_df[(opt_df["strike"] == strike) & (opt_df["instrument_type"] == "CE")]
            pe = opt_df[(opt_df["strike"] == strike) & (opt_df["instrument_type"] == "PE")]

            ce_q = option_quotes.get("NFO:" + ce.iloc[0]["tradingsymbol"], {}) if not ce.empty else {}
            pe_q = option_quotes.get("NFO:" + pe.iloc[0]["tradingsymbol"], {}) if not pe.empty else {}

            chain.append({
                "Strike": strike,
                "CE_LTP": ce_q.get("last_price"),
                "CE_OI": ce_q.get("oi"),
                "PE_LTP": pe_q.get("last_price"),
                "PE_OI": pe_q.get("oi"),
            })

        df_mp = compute_live_max_pain(pd.DataFrame(chain))

        spot = spot_quotes.get(f"NSE:{stock}", {})
        ltp = spot.get("last_price")
        prev = spot.get("ohlc", {}).get("close")

        pct = (
            round(((ltp - prev) / prev) * 100, 2)
            if ltp and prev else np.nan
        )

        for _, r in df_mp.iterrows():
            rows.append({
                "Stock": stock,
                "Strike": r["Strike"],
                live_mp_col: r[live_mp_col],
                "Live_Stock_LTP": ltp,
                pct_col: pct
            })

    return pd.DataFrame(rows)

# =====================================
# MERGE LIVE DATA
# =====================================
live_df = fetch_live_data(df_hist["Stock"].unique().tolist())
df = df_hist.merge(live_df, on=["Stock", "Strike"], how="left")

df[pct_col] = df.groupby("Stock")[pct_col].transform("first")

# =====================================
# DELTAS
# =====================================
df[live_delta_col] = df[live_mp_col] - df[mp1_col]

df[delta_live_above_col] = np.nan
df[sum_live_col] = np.nan

for stock, sdf in df.sort_values("Strike").groupby("Stock"):
    sdf = sdf.reset_index()
    vals = sdf[live_delta_col].values
    diff = vals - np.roll(vals, -1)
    diff[-1] = np.nan

    df.loc[sdf["index"], delta_live_above_col] = diff

    ltp = sdf["Live_Stock_LTP"].iloc[0]
    strikes = sdf["Strike"].values

    atm = next(
        (i for i in range(len(strikes)-1)
         if strikes[i] <= ltp <= strikes[i+1]),
        None
    )

    if atm is not None:
        df.loc[df["Stock"] == stock, sum_live_col] = abs(
            sdf.loc[[atm, atm+1], delta_live_above_col].sum()
        )

# =====================================
# HIGHLIGHTING
# =====================================
def highlight(df):
    styles = pd.DataFrame("", index=df.index, columns=df.columns)

    for stock in df["Stock"].unique():
        sdf = df[df["Stock"] == stock]
        ltp = sdf["Live_Stock_LTP"].iloc[0]
        strikes = sdf["Strike"].values

        for i in range(len(strikes)-1):
            if strikes[i] <= ltp <= strikes[i+1]:
                styles.loc[sdf.index[i]] = "background-color:#003366;color:white"
                styles.loc[sdf.index[i+1]] = "background-color:#003366;color:white"
                break

        styles.loc[sdf[live_mp_col].idxmin()] = "background-color:#8B0000;color:white"

    return styles

# =====================================
# DISPLAY (SAFE)
# =====================================
display_cols = [
    "Stock", "Strike",
    mp1_col,
    live_mp_col,
    live_delta_col,
    delta_live_above_col,
    sum_live_col,
    pct_col,
    "Live_Stock_LTP"
]

display_df = df[display_cols].copy()

for c in display_df.columns:
    if c in {pct_col, "Live_Stock_LTP"}:
        display_df[c] = pd.to_numeric(display_df[c], errors="coerce").round(2)
    elif c != "Stock":
        display_df[c] = pd.to_numeric(display_df[c], errors="coerce").round(0).astype("Int64")

st.dataframe(
    display_df.style.apply(highlight, axis=None),
    use_container_width=True,
    height=900
)

# =====================================
# DOWNLOAD
# =====================================
st.download_button(
    "⬇️ Download CSV",
    df.to_csv(index=False),
    f"max_pain_live_{t_lbl}.csv",
    "text/csv",
)

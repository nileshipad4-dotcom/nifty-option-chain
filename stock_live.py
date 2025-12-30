# =====================================
# STOCK OC MAIN – HISTORICAL + LIVE MP
# =====================================

import streamlit as st
import pandas as pd
import numpy as np
import os
from streamlit_autorefresh import st_autorefresh
from kiteconnect import KiteConnect
import pytz
from datetime import datetime

# =====================================
# AUTO REFRESH (5 MIN)
# =====================================
st_autorefresh(interval=360_000, key="auto_refresh")

# =====================================
# STREAMLIT CONFIG
# =====================================
st.set_page_config(page_title="Live Max Pain", layout="wide")
st.title("📊 FnO STOCKS – Historical + Live Max Pain")

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
# LOAD LATEST CSV (T1 ONLY)
# =====================================
def load_latest_csv():
    files = sorted(
        [f for f in os.listdir(DATA_DIR) if f.startswith("option_chain_")],
        reverse=True
    )
    if not files:
        st.error("No option_chain CSV found")
        st.stop()
    ts = files[0].replace("option_chain_", "").replace(".csv", "")
    return ts, os.path.join(DATA_DIR, files[0])

t1_lbl, csv_path = load_latest_csv()
t1_lbl = t1_lbl.split("_")[-1].replace("-", ":")

mp1_col = f"MP ({t1_lbl})"
live_mp_col = f"MP ({datetime.now(IST).strftime('%H:%M')})"

pct_col = "Live % Change"
live_delta_col = f"Δ Live MP (Live - {t1_lbl})"
delta_live_above_col = "ΔΔ Live MP"
sum_live_2_above_below_col = "Σ |ΔΔ Live MP| (±2)"

# =====================================
# LOAD HISTORICAL DATA
# =====================================
df_hist = pd.read_csv(csv_path)
df_hist = df_hist[["Stock", "Strike", "Max_Pain"]].rename(
    columns={"Max_Pain": mp1_col}
)

# =====================================
# LIVE MAX PAIN LOGIC
# =====================================
def compute_live_max_pain(df):
    df = df.fillna(0)
    A, B, G, M, L = df["CE_LTP"], df["CE_OI"], df["Strike"], df["PE_LTP"], df["PE_OI"]
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

    # chunk NSE quotes safely
    for batch in [stocks[i:i+50] for i in range(0, len(stocks), 50)]:
        spot_quotes = kite.quote([f"NSE:{s}" for s in batch])

        for stock in batch:
            opt_df = instruments[
                (instruments["name"] == stock) &
                (instruments["segment"] == "NFO-OPT")
            ]

            if opt_df.empty:
                continue

            expiry = opt_df["expiry"].min()
            opt_df = opt_df[opt_df["expiry"] == expiry]

            quotes = kite.quote(["NFO:" + s for s in opt_df["tradingsymbol"]])

            chain = []
            for strike in sorted(opt_df["strike"].unique()):
                ce = opt_df[(opt_df["strike"] == strike) & (opt_df["instrument_type"] == "CE")]
                pe = opt_df[(opt_df["strike"] == strike) & (opt_df["instrument_type"] == "PE")]

                ce_q = quotes.get("NFO:" + ce.iloc[0]["tradingsymbol"], {}) if not ce.empty else {}
                pe_q = quotes.get("NFO:" + pe.iloc[0]["tradingsymbol"], {}) if not pe.empty else {}

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

            live_pct = (
                round(((ltp - prev) / prev) * 100, 2)
                if ltp and prev else np.nan
            )

            for _, r in df_mp.iterrows():
                rows.append({
                    "Stock": stock,
                    "Strike": r["Strike"],
                    live_mp_col: r[live_mp_col],
                    "Live_Stock_LTP": ltp,
                    pct_col: live_pct
                })

    return pd.DataFrame(rows)

# =====================================
# MERGE DATA
# =====================================
live_df = fetch_live_data(df_hist["Stock"].unique().tolist())
final_df = df_hist.merge(live_df, on=["Stock", "Strike"], how="left")
final_df[pct_col] = final_df.groupby("Stock")[pct_col].transform("first")

# =====================================
# DELTAS
# =====================================
final_df[live_delta_col] = final_df[live_mp_col] - final_df[mp1_col]

final_df[delta_live_above_col] = np.nan
final_df[sum_live_2_above_below_col] = np.nan

for stock, sdf in final_df.sort_values("Strike").groupby("Stock"):
    sdf = sdf.dropna(subset=["Strike"]).reset_index()

    if sdf.empty:
        continue

    vals = sdf[live_delta_col].values
    diff = vals - np.roll(vals, -1)
    diff[-1] = np.nan
    final_df.loc[sdf["index"], delta_live_above_col] = diff

    ltp = sdf["Live_Stock_LTP"].iloc[0]
    if ltp is None or pd.isna(ltp):
        continue

    strikes = sdf["Strike"].values
    atm_idx = None

    for i in range(len(strikes) - 1):
        if strikes[i] <= ltp <= strikes[i + 1]:
            atm_idx = i
            break

    if atm_idx is None:
        continue

    idxs = [atm_idx, atm_idx + 1]
    val = sdf.loc[idxs, delta_live_above_col].sum()
    final_df.loc[final_df["Stock"] == stock, sum_live_2_above_below_col] = abs(val)

# =====================================
# HIGHLIGHTING
# =====================================
def highlight_rows(df):
    styles = pd.DataFrame("", index=df.index, columns=df.columns)

    for stock in df["Stock"].dropna().unique():
        sdf = df[(df["Stock"] == stock) & df["Strike"].notna()]
        if sdf.empty:
            continue

        ltp = sdf["Live_Stock_LTP"].iloc[0]
        strikes = sdf["Strike"].values

        if ltp is not None and not pd.isna(ltp):
            for i in range(len(strikes) - 1):
                if strikes[i] <= ltp <= strikes[i + 1]:
                    styles.loc[sdf.index[i], :] = "background-color:#003366;color:white"
                    styles.loc[sdf.index[i + 1], :] = "background-color:#003366;color:white"
                    break

        mp_idx = sdf[live_mp_col].idxmin()
        styles.loc[mp_idx, :] = "background-color:#8B0000;color:white"

    return styles

# =====================================
# DISPLAY
# =====================================
display_cols = [
    "Stock", "Strike",
    mp1_col,
    live_mp_col,
    live_delta_col,
    delta_live_above_col,
    sum_live_2_above_below_col,
    pct_col,
    "Live_Stock_LTP"
]

display_df = final_df[display_cols].copy()

for col in display_df.columns:
    if col == "Stock":
        continue
    if col in [pct_col, "Live_Stock_LTP"]:
        display_df[col] = pd.to_numeric(display_df[col], errors="coerce").round(2)
    else:
        display_df[col] = pd.to_numeric(display_df[col], errors="coerce").round(0).astype("Int64")

st.dataframe(
    display_df.style.apply(highlight_rows, axis=None),
    use_container_width=True,
    height=900
)

# =====================================
# DOWNLOAD
# =====================================
st.download_button(
    "⬇️ Download CSV",
    final_df.to_csv(index=False),
    f"live_max_pain_{t1_lbl}.csv",
    "text/csv"
)

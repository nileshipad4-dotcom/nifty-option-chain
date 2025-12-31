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
st.set_page_config(page_title="Max Pain Comparison", layout="wide")
st.title("📊 FnO STOCKS – Historical + Live Max Pain")

DATA_DIR = "data"
IST = pytz.timezone("Asia/Kolkata")

# =====================================
# KITE CONFIG
# =====================================
API_KEY = "bkgv59vaazn56c42"
ACCESS_TOKEN = "sZ2orQDyT9vf56fEwKSa2FrTBCb6xGGQ"

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

mp1_col = f"MP ({t1_lbl})"
mp2_col = f"MP ({t2_lbl})"
mp3_col = f"MP ({t3_lbl})"

live_delta_col = f"Δ MP (Live - {t1_lbl})"
delta_12 = f"Δ MP ({t1_lbl}-{t2_lbl})"
delta_23 = f"Δ MP ({t2_lbl}-{t3_lbl})"

delta_live_above_col = "ΔΔ MP"
sum_live_2_above_below_col = "Σ ΔΔ MP (±2)"

delta_above_col = "ΔΔ MP 1"
sum_2_above_below_col = "Σ |ΔΔ MP-old| (±2)"

pct_col = "% Ch"

# =====================================
# LOAD CSV DATA
# =====================================
df1 = pd.read_csv(file_map[t1])
df2 = pd.read_csv(file_map[t2])
df3 = pd.read_csv(file_map[t3])

df1 = df1[["Stock", "Strike", "Max_Pain", "Stock_LTP"]].rename(columns={"Max_Pain": mp1_col})
df2 = df2[["Stock", "Strike", "Max_Pain"]].rename(columns={"Max_Pain": mp2_col})
df3 = df3[["Stock", "Strike", "Max_Pain"]].rename(columns={"Max_Pain": mp3_col})

df = df1.merge(df2, on=["Stock", "Strike"]).merge(df3, on=["Stock", "Strike"])

# =====================================
# LIVE MAX PAIN LOGIC
# =====================================
def compute_live_max_pain(df):
    df = df.fillna(0)
    A,B,G,M,L = df["CE_LTP"],df["CE_OI"],df["Strike"],df["PE_LTP"],df["PE_OI"]
    mp=[]
    for i in range(len(df)):
        val = (
            -sum(A[i:] * B[i:])
            + G.iloc[i] * sum(B[:i]) - sum(G[:i] * B[:i])
            - sum(M[:i] * L[:i])
            + sum(G[i:] * L[i:]) - G.iloc[i] * sum(L[i:])
        )
        mp.append(int(val/10000))
    df["Live_Max_Pain"] = mp
    return df

# =====================================
# FETCH LIVE DATA
# =====================================
@st.cache_data(ttl=300)
def fetch_live_mp_and_ltp(stocks):
    rows=[]
    for i in range(0, len(stocks), 40):
        batch = stocks[i:i+40]
        spot_quotes = kite.quote([f"NSE:{s}" for s in batch])

        for stock in batch:
            opt_df = instruments[
                (instruments["name"]==stock) &
                (instruments["segment"]=="NFO-OPT")
            ]
            if opt_df.empty:
                continue

            expiry = opt_df["expiry"].min()
            opt_df = opt_df[opt_df["expiry"]==expiry]

            quotes = kite.quote(["NFO:"+s for s in opt_df["tradingsymbol"]])

            chain=[]
            for strike in sorted(opt_df["strike"].unique()):
                ce = opt_df[(opt_df["strike"]==strike)&(opt_df["instrument_type"]=="CE")]
                pe = opt_df[(opt_df["strike"]==strike)&(opt_df["instrument_type"]=="PE")]
                ce_q = quotes.get("NFO:"+ce.iloc[0]["tradingsymbol"],{}) if not ce.empty else {}
                pe_q = quotes.get("NFO:"+pe.iloc[0]["tradingsymbol"],{}) if not pe.empty else {}

                chain.append({
                    "Strike":strike,
                    "CE_LTP":ce_q.get("last_price"),
                    "CE_OI":ce_q.get("oi"),
                    "PE_LTP":pe_q.get("last_price"),
                    "PE_OI":pe_q.get("oi"),
                })

            df_mp = compute_live_max_pain(pd.DataFrame(chain))
            spot = spot_quotes.get(f"NSE:{stock}", {})
            ltp = spot.get("last_price")
            prev = spot.get("ohlc", {}).get("close")

            live_pct = round(((ltp-prev)/prev)*100,2) if ltp and prev else np.nan

            for _,r in df_mp.iterrows():
                rows.append({
                    "Stock":stock,
                    "Strike":r["Strike"],
                    "Live_Max_Pain":r["Live_Max_Pain"],
                    "Live_Stock_LTP":ltp,
                    pct_col: live_pct
                })
    return pd.DataFrame(rows)

# =====================================
# INSERT BLANK ROWS
# =====================================
rows=[]
for stock,sdf in df.sort_values(["Stock","Strike"]).groupby("Stock"):
    rows.append(sdf)
    rows.append(pd.DataFrame([{c:np.nan for c in df.columns}]))
final_df = pd.concat(rows[:-1], ignore_index=True)

# =====================================
# MERGE LIVE DATA
# =====================================
stocks = final_df["Stock"].dropna().astype(str).str.strip()
stocks = stocks[stocks!=""].unique().tolist()
live_df = fetch_live_mp_and_ltp(stocks)

final_df = final_df.merge(live_df, on=["Stock","Strike"], how="left")
final_df[pct_col] = final_df.groupby("Stock")[pct_col].transform("first")

# =====================================
# DELTAS
# =====================================
final_df[live_delta_col] = final_df["Live_Max_Pain"] - final_df[mp1_col]
final_df[delta_12] = final_df[mp1_col] - final_df[mp2_col]
final_df[delta_23] = final_df[mp2_col] - final_df[mp3_col]

# =====================================
# ΔΔ LIVE MP (FINAL CORRECT VERSION)
# =====================================
final_df[delta_live_above_col] = np.nan
final_df[sum_live_2_above_below_col] = np.nan

for stock, sdf in final_df.sort_values("Strike").groupby("Stock"):
    sdf = sdf[sdf["Strike"].notna()].reset_index()
    if sdf.empty:
        continue

    # ΔΔ Live MP calculation
    vals = sdf[live_delta_col].astype(float).values
    diff = vals - np.roll(vals, -1)
    diff[-1] = np.nan

    final_df.loc[sdf["index"], delta_live_above_col] = diff

    ltp = sdf["Live_Stock_LTP"].iloc[0]
    strikes = sdf["Strike"].values

    if ltp is None or np.isnan(ltp):
        continue

    # ✅ FIND STRIKE PAIR WHERE LTP LIES
    for i in range(len(strikes) - 1):
        if strikes[i] <= ltp <= strikes[i + 1]:

            pair_vals = sdf.loc[[i, i + 1], delta_live_above_col]

            # ✅ DROP NaN BEFORE SUM
            val = pair_vals.dropna().sum()

            final_df.loc[
                sdf.loc[[i, i + 1], "index"],
                sum_live_2_above_below_col
            ] = val
            break



# =====================================
# HIGHLIGHTING (TWO STRIKES)
# =====================================
def highlight_rows(df):
    styles = pd.DataFrame("", index=df.index, columns=df.columns)

    for stock in df["Stock"].dropna().unique():
        sdf = df[(df["Stock"]==stock)&(df["Strike"].notna())]
        if sdf.empty:
            continue

        ltp = sdf["Live_Stock_LTP"].iloc[0]
        strikes = sdf["Strike"].values

        if ltp is not None and not np.isnan(ltp):
            for i in range(len(strikes)-1):
                if strikes[i] <= ltp <= strikes[i+1]:
                    styles.loc[sdf.index[i],:] = "background-color:#003366;color:white"
                    styles.loc[sdf.index[i+1],:] = "background-color:#003366;color:white"
                    break

        min_idx = sdf["Live_Max_Pain"].idxmin()
        styles.loc[min_idx,:] = "background-color:#8B0000;color:white"

    return styles

# =====================================
# DISPLAY
# =====================================
display_cols = [
    "Stock","Strike","Live_Max_Pain",
    mp1_col,mp2_col,
    live_delta_col,
    delta_live_above_col,
    sum_live_2_above_below_col,
    pct_col,"Live_Stock_LTP"
]

display_df = final_df[display_cols].copy()
float_cols = {pct_col,"Live_Stock_LTP"}

for c in display_df.columns:
    if c=="Stock": continue
    if c in float_cols:
        display_df[c] = pd.to_numeric(display_df[c],errors="coerce").round(2)
    else:
        display_df[c] = pd.to_numeric(display_df[c],errors="coerce").round(0).astype("Int64")

st.dataframe(display_df.style.apply(highlight_rows, axis=None),
             use_container_width=True, height=900)

# =====================================
# DOWNLOAD
# =====================================
st.download_button(
    "⬇️ Download CSV",
    final_df.to_csv(index=False),
    f"max_pain_with_live_{t1_lbl}_{t2_lbl}_{t3_lbl}.csv",
    "text/csv",
)

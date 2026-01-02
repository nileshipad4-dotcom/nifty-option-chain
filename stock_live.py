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
import time 

# =====================================
# AUTO REFRESH (5 MIN)
# =====================================
st_autorefresh(interval=300_000, key="auto_refresh")

# =====================================
# STREAMLIT CONFIG
# =====================================
st.set_page_config(page_title="Max Pain Comparison", layout="wide")
st.title("📊 FnO STOCKS")

DATA_DIR = "data"
IST = pytz.timezone("Asia/Kolkata")

# =====================================
# KITE CONFIG
# =====================================
API_KEY = "bkgv59vaazn56c42"
ACCESS_TOKEN = "4epY0bfSn2lIfN8CvTZuyMvrE2mA58W7"

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
if len(csv_files) < 2:
    st.error("Need at least 2 CSV files.")
    st.stop()

timestamps = [ts for ts, _ in csv_files]
file_map = dict(csv_files)

def short_ts(ts):
    return ts.split("_")[-1].replace("-", ":")

# =====================================
# DROPDOWNS
# =====================================
c1, c2 = st.columns(2)
with c1:
    t1 = st.selectbox("Timestamp 1 (Latest)", timestamps, 0)
with c2:
    t2 = st.selectbox("Timestamp 2", timestamps, 1)

t1_lbl, t2_lbl = short_ts(t1), short_ts(t2)

mp1_col = f"MP ({t1_lbl})"
mp2_col = f"MP ({t2_lbl})"

live_delta_col = f"Δ MP (Live - {t1_lbl})"
live_delta_2_col = f"Δ MP ({t1_lbl} - {t2_lbl})"

delta_live_above_col = "ΔΔ MP"
delta_live_above_2_col = "ΔΔ MP 2"

sum_live_exact_atm_col = "Σ ΔΔ MP"
pct_col = "% Ch"



# =====================================
# LOAD CSV DATA
# =====================================
df1 = pd.read_csv(file_map[t1])
df2 = pd.read_csv(file_map[t2])

df1 = df1[["Stock", "Strike", "Max_Pain", "Stock_LTP"]].rename(columns={"Max_Pain": mp1_col})
df2 = df2[["Stock", "Strike", "Max_Pain"]].rename(columns={"Max_Pain": mp2_col})

df = df1.merge(df2, on=["Stock", "Strike"])
# =====================================
# REMOVE INDEX SYMBOLS
# =====================================
EXCLUDE_SYMBOLS = {"NIFTY", "BANKNIFTY", "FINNIFTY", "MIDCPNIFTY"}

df = df[~df["Stock"].isin(EXCLUDE_SYMBOLS)]

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
    df["MP Live"] = mp
    return df


# =====================================
# FETCH LIVE DATA (MP + LTP + DAY HIGH)
# =====================================
@st.cache_data(ttl=300)
def fetch_live_mp_and_ltp(stocks):
    rows = []

    def chunk(lst, size=15):
        for i in range(0, len(lst), size):
            yield lst[i:i + size]

    for batch in chunk(stocks, 15):

        # ---- FETCH SPOT QUOTES (HIGH COMES ONLY FROM HERE) ----
        spot_quotes = kite.quote([f"NSE:{s}" for s in batch])
        time.sleep(0.5)  # prevent quote degradation

        for stock in batch:

            # ---- SPOT DATA ----
            spot = spot_quotes.get(f"NSE:{stock}", {})
            if not spot:
                continue

            ohlc = spot.get("ohlc", {}) or {}

            ltp = spot.get("last_price")
            prev = ohlc.get("close")

            # ✅ DAY HIGH (ONLY THIS)
            high = ohlc.get("high") or spot.get("day_high")

            # ---- OPTION DATA ----
            opt_df = instruments[
                (instruments["name"] == stock) &
                (instruments["segment"] == "NFO-OPT")
            ]
            if opt_df.empty:
                continue

            expiry = opt_df["expiry"].min()
            opt_df = opt_df[opt_df["expiry"] == expiry]

            option_symbols = ["NFO:" + s for s in opt_df["tradingsymbol"].tolist()]
            quotes = {}

            for sym_batch in chunk(option_symbols, 200):
                quotes.update(kite.quote(sym_batch))

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

            if not chain:
                continue

            df_mp = compute_live_max_pain(pd.DataFrame(chain))

            live_pct = (
                round(((ltp - prev) / prev) * 100, 2)
                if ltp and prev else np.nan
            )

            # ---- APPEND ROWS ----
            for _, r in df_mp.iterrows():
                rows.append({
                    "Stock": stock,
                    "Strike": r["Strike"],
                    "MP Live": r["MP Live"],
                    "LTP": ltp,
                    "High": high,     # 👈 ONLY HIGH
                    pct_col: live_pct
                })

    return pd.DataFrame(rows)


# =====================================
# MERGE LIVE DATA
# =====================================
stocks = [
    s for s in df["Stock"].unique().tolist()
    if s not in {"NIFTY", "BANKNIFTY", "FINNIFTY", "MIDCPNIFTY"}
]

live_df = fetch_live_mp_and_ltp(stocks)

final_df = df.merge(live_df, on=["Stock", "Strike"], how="left")
final_df[pct_col] = final_df.groupby("Stock")[pct_col].transform("first")


# =====================================
# DELTA CALCULATIONS
# =====================================
final_df[live_delta_col] = final_df["MP Live"] - final_df[mp1_col]
final_df[live_delta_2_col] = final_df[mp1_col] - final_df[mp2_col]

final_df[delta_live_above_col] = (
    final_df[live_delta_col] -
    final_df.groupby("Stock")[live_delta_col].shift(-1)
)

final_df[delta_live_above_2_col] = (
    final_df[live_delta_2_col] -
    final_df.groupby("Stock")[live_delta_2_col].shift(-1)
)

# =====================================
# Σ ΔΔ MP
# =====================================
final_df[sum_live_exact_atm_col] = np.nan

for stock, sdf in final_df.sort_values("Strike").groupby("Stock"):
    sdf = sdf.reset_index()
    ltp = sdf["LTP"].iloc[0]
    strikes = sdf["Strike"].values

    if pd.isna(ltp):
        continue

    for i in range(len(strikes) - 1):
        if strikes[i] <= ltp <= strikes[i + 1]:
            final_df.loc[final_df["Stock"] == stock, sum_live_exact_atm_col] = (
                sdf.loc[i, delta_live_above_col] +
                sdf.loc[i + 1, delta_live_above_col]
            )
            break




# =====================================
# HIGHLIGHTING
# =====================================
def highlight_rows(df):
    styles = pd.DataFrame("", index=df.index, columns=df.columns)

    for stock in df["Stock"].dropna().unique():
        sdf = df[(df["Stock"] == stock) & df["Strike"].notna()]
        if sdf.empty:
            continue

        # ---- ATM HIGHLIGHT ----
        ltp = pd.to_numeric(sdf["LTP"].iloc[0], errors="coerce")
        strikes = sdf["Strike"].values

        if pd.notna(ltp):
            for i in range(len(strikes) - 1):
                if strikes[i] <= ltp <= strikes[i + 1]:
                    styles.loc[sdf.index[i], :] = "background-color:#003366;color:white"
                    styles.loc[sdf.index[i + 1], :] = "background-color:#003366;color:white"
                    break

        # ---- MIN LIVE MP HIGHLIGHT ----
        mp_vals = sdf["MP Live"].dropna()
        if not mp_vals.empty:
            styles.loc[mp_vals.idxmin(), :] = "background-color:#8B0000;color:white"

    return styles


# =====================================
# FILTER 6 STRIKES BELOW & ABOVE LTP
# =====================================
filtered_rows = []

for stock, sdf in final_df.groupby("Stock"):
    sdf = sdf.sort_values("Strike").reset_index(drop=True)

    ltp = sdf["LTP"].iloc[0]
    if pd.isna(ltp):
        continue

    # Find ATM index
    atm_idx = sdf["Strike"].sub(ltp).abs().idxmin()

    lower_idx = max(atm_idx - 6, 0)
    upper_idx = min(atm_idx + 6, len(sdf))  # +7 to include ATM + 6 above

    filtered_rows.append(sdf.iloc[lower_idx:upper_idx])

final_df = pd.concat(filtered_rows, ignore_index=True)



# =====================================
# DISPLAY
# =====================================
display_cols = [
    "Stock",
    "Strike",
    "MP Live",
    mp1_col,
    mp2_col,
    live_delta_col,
    live_delta_2_col,
    delta_live_above_col,
    delta_live_above_2_col,
    sum_live_exact_atm_col,
    pct_col,
    "LTP",
    "High"
]

display_df = final_df[display_cols].copy()

for c in display_df.columns:
    if c == "Stock":
        continue
    if c in {pct_col, "LTP", "High"}:
        display_df[c] = (
            pd.to_numeric(display_df[c], errors="coerce")
            .round(2)
            .map(lambda x: f"{x:.2f}" if pd.notna(x) else "")
        )
    else:
        display_df[c] = (
            pd.to_numeric(display_df[c], errors="coerce")
            .round(0)
            .astype("Int64")
        )

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
    f"max_pain_with_live_{t1_lbl}_{t2_lbl}.csv",
    "text/csv",
)

# =====================================
# STOCK OC MAIN – WITH LIVE MAX PAIN
# =====================================

import streamlit as st
import pandas as pd
import numpy as np
import os
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
# KITE CONFIG (LIVE)
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

latest_file = csv_files[0][0]

if "last_ts" not in st.session_state:
    st.session_state.last_ts = latest_file

if latest_file != st.session_state.last_ts:
    st.session_state.last_ts = latest_file
    st.experimental_rerun()

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
pct_col = f"% Ch ({t1_lbl})"

delta_12 = f"Δ MP ({t1_lbl}-{t2_lbl})"
delta_23 = f"Δ MP ({t2_lbl}-{t3_lbl})"
sum_12_col = f"Σ {delta_12}"
delta_above_col = f"ΔΔ MP"
sum_2_above_below_col = f"Σ |ΔΔ MP| (±2)"

# =====================================
# LOAD CSV DATA
# =====================================
df1 = pd.read_csv(file_map[t1])
df2 = pd.read_csv(file_map[t2])
df3 = pd.read_csv(file_map[t3])

if "Stock_%_Change" not in df1.columns:
    df1["Stock_%_Change"] = np.nan

df1 = df1[["Stock","Strike","Max_Pain","Stock_LTP","Stock_%_Change"]]\
        .rename(columns={"Max_Pain": mp1_col, "Stock_%_Change": pct_col})

df2 = df2[["Stock","Strike","Max_Pain"]].rename(columns={"Max_Pain": mp2_col})
df3 = df3[["Stock","Strike","Max_Pain"]].rename(columns={"Max_Pain": mp3_col})

df = df1.merge(df2, on=["Stock","Strike"]).merge(df3, on=["Stock","Strike"])

df[delta_12] = df[mp1_col] - df[mp2_col]
df[delta_23] = df[mp2_col] - df[mp3_col]

# =====================================
# Σ Δ MP (TREND)
# =====================================
df[sum_12_col] = np.nan
for stock, sdf in df.sort_values("Strike").groupby("Stock"):
    df.loc[sdf.index, sum_12_col] = (
        sdf[delta_12].rolling(7, center=True, min_periods=1).sum().values
    )

# =====================================
# ΔΔ MP
# =====================================
df[delta_above_col] = np.nan
for stock, sdf in df.sort_values("Strike").groupby("Stock"):
    vals = sdf[delta_12].astype(float).values
    diff = vals - np.roll(vals, -1)
    diff[-1] = np.nan
    df.loc[sdf.index, delta_above_col] = diff

df[sum_2_above_below_col] = np.nan

for stock, sdf in df.sort_values("Strike").groupby("Stock"):
    sdf = sdf.reset_index(drop=True)
    ltp = float(sdf["Stock_LTP"].iloc[0])
    strikes = sdf["Strike"].values

    atm_idx = next((i for i in range(len(strikes)-1)
                   if strikes[i] <= ltp <= strikes[i+1]), None)
    if atm_idx is None:
        continue

    idxs = [atm_idx, atm_idx+1]
    idxs = [i for i in idxs if i < len(sdf)]

    df.loc[df["Stock"]==stock, sum_2_above_below_col] = \
        abs(sdf.loc[idxs, delta_above_col].sum())

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

@st.cache_data(ttl=300)
def fetch_live_mp_and_ltp(stocks):
    rows=[]
    spot_quotes = kite.quote([f"NSE:{s}" for s in stocks])

    for stock in stocks:
        opt_df = instruments[(instruments["name"]==stock)&
                              (instruments["segment"]=="NFO-OPT")]
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
        ltp = spot_quotes.get(f"NSE:{stock}",{}).get("last_price")

        for _,r in df_mp.iterrows():
            rows.append({
                "Stock":stock,
                "Strike":r["Strike"],
                "Live_Max_Pain":r["Live_Max_Pain"],
                "Live_Stock_LTP":ltp
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
live_df = fetch_live_mp_and_ltp(final_df["Stock"].dropna().unique().tolist())
final_df = final_df.merge(live_df, on=["Stock","Strike"], how="left")

# =====================================
# DISPLAY
# =====================================
display_cols = [
    "Stock","Strike",
    mp1_col,mp2_col,mp3_col,
    "Live_Max_Pain",
    delta_12,delta_23,
    delta_above_col,sum_2_above_below_col,
    pct_col,"Live_Stock_LTP","Stock_LTP"
]

st.dataframe(
    final_df[display_cols],
    use_container_width=True,
)

# =====================================
# DOWNLOAD
# =====================================
st.download_button(
    "⬇️ Download CSV",
    final_df.to_csv(index=False),
    f"max_pain_with_live_{t1_lbl}_{t2_lbl}_{t3_lbl}.csv",
    "text/csv",
)


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
# AUTO REFRESH
# =====================================
st_autorefresh(interval=360_000, key="auto_refresh")

st.set_page_config(page_title="Max Pain Comparison", layout="wide")
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
# LOAD CSV FILES
# =====================================
def load_csv_files():
    if not os.path.exists(DATA_DIR):
        return []
    return sorted(
        [
            (f.replace("option_chain_", "").replace(".csv", ""), os.path.join(DATA_DIR, f))
            for f in os.listdir(DATA_DIR)
            if f.startswith("option_chain_") and f.endswith(".csv")
        ],
        reverse=True
    )

csv_files = load_csv_files()
if len(csv_files) < 3:
    st.error("Need at least 3 CSV files.")
    st.stop()

timestamps = [ts for ts, _ in csv_files]
file_map = dict(csv_files)

def short_ts(ts):
    return ts.split("_")[-1].replace("-", ":")

# =====================================
# TIMESTAMPS
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

live_time_lbl = datetime.now(IST).strftime("%H:%M")
live_mp_col = f"MP ({live_time_lbl})"

live_delta_col = f"Δ Live MP (Live - {t1_lbl})"
delta_12 = f"Δ MP ({t1_lbl}-{t2_lbl})"
delta_23 = f"Δ MP ({t2_lbl}-{t3_lbl})"

pct_col = "Live % Change"

# =====================================
# LOAD CSV DATA
# =====================================
df1 = pd.read_csv(file_map[t1])[["Stock","Strike","Max_Pain"]].rename(columns={"Max_Pain": mp1_col})
df2 = pd.read_csv(file_map[t2])[["Stock","Strike","Max_Pain"]].rename(columns={"Max_Pain": mp2_col})
df3 = pd.read_csv(file_map[t3])[["Stock","Strike","Max_Pain"]].rename(columns={"Max_Pain": mp3_col})

df = df1.merge(df2, on=["Stock","Strike"]).merge(df3, on=["Stock","Strike"])

# =====================================
# LIVE MAX PAIN
# =====================================
def compute_live_mp(df):
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
    df[live_mp_col] = mp
    return df

@st.cache_data(ttl=300)
def fetch_live_data(stocks):
    rows=[]
    spot_quotes = kite.quote([f"NSE:{s}" for s in stocks])

    for stock in stocks:
        opt_df = instruments[
            (instruments["name"]==stock)&
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
            chain.append({
                "Strike":strike,
                "CE_LTP":quotes.get("NFO:"+ce.iloc[0]["tradingsymbol"],{}).get("last_price"),
                "CE_OI":quotes.get("NFO:"+ce.iloc[0]["tradingsymbol"],{}).get("oi"),
                "PE_LTP":quotes.get("NFO:"+pe.iloc[0]["tradingsymbol"],{}).get("last_price"),
                "PE_OI":quotes.get("NFO:"+pe.iloc[0]["tradingsymbol"],{}).get("oi"),
            })

        mp_df = compute_live_mp(pd.DataFrame(chain))

        spot = spot_quotes.get(f"NSE:{stock}",{})
        ltp = spot.get("last_price")
        prev = spot.get("ohlc",{}).get("close")
        pct = round(((ltp-prev)/prev)*100,2) if ltp and prev else np.nan

        for _,r in mp_df.iterrows():
            rows.append({
                "Stock":stock,
                "Strike":r["Strike"],
                live_mp_col:r[live_mp_col],
                "Live_Stock_LTP":round(ltp,2) if ltp else np.nan,
                pct_col:pct
            })

    return pd.DataFrame(rows)

# =====================================
# MERGE
# =====================================
live_df = fetch_live_data(df["Stock"].unique().tolist())
final_df = df.merge(live_df, on=["Stock","Strike"], how="left")

# =====================================
# DELTAS
# =====================================
final_df[live_delta_col] = final_df[live_mp_col] - final_df[mp1_col]
final_df[delta_12] = final_df[mp1_col] - final_df[mp2_col]
final_df[delta_23] = final_df[mp2_col] - final_df[mp3_col]

# =====================================
# DISPLAY (SAFE)
# =====================================
display_cols = [
    "Stock","Strike",
    mp1_col,mp2_col,mp3_col,
    live_mp_col,
    live_delta_col,
    delta_12,delta_23,
    pct_col,"Live_Stock_LTP"
]

# 🔐 Ensure numeric columns are numeric
for c in display_cols:
    if c not in ["Stock"]:
        final_df[c] = pd.to_numeric(final_df[c], errors="coerce")

format_dict = {
    c: "{:.0f}" for c in display_cols
}
format_dict[pct_col] = "{:.2f}"
format_dict["Live_Stock_LTP"] = "{:.2f}"

st.dataframe(
    final_df[display_cols].style.format(format_dict, na_rep=""),
    use_container_width=True,
    height=900
)

st.success(f"Last updated: {datetime.now(IST).strftime('%H:%M:%S')} IST")

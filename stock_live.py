import streamlit as st
import pandas as pd
import numpy as np
import os
import base64
import requests
from kiteconnect import KiteConnect
from datetime import datetime, time
import pytz
from streamlit_autorefresh import st_autorefresh

# ==================================================
# STREAMLIT CONFIG
# ==================================================
st.set_page_config(page_title="LIVE Option Chain Snapshot", layout="wide")
st.title("📊 LIVE Option Chain → GitHub Snapshot (Full Chain)")

refresh_tick = st_autorefresh(interval=180_000, key="live_refresh")

# ==================================================
# TIMEZONE
# ==================================================
IST = pytz.timezone("Asia/Kolkata")

# ==================================================
# CONFIG
# ==================================================
API_KEY = "bkgv59vaazn56c42"
ACCESS_TOKEN = "um1gYW2GgQ94kdg2G1C9vu3cWfdFF00X"

DATA_DIR = "data"
os.makedirs(DATA_DIR, exist_ok=True)

# ==================================================
# STOCK LIST
# ==================================================
STOCKS = [
    "360ONE","ABB","ABCAPITAL","ADANIENSOL","ADANIENT","ADANIGREEN","ADANIPORTS","ALKEM",
    "AMBER","AMBUJACEM","ANGELONE","APLAPOLLO","APOLLOHOSP","ASHOKLEY","ASIANPAINT","ASTRAL",
    "AUBANK","AUROPHARMA","AXISBANK","BAJAJ-AUTO","BAJAJFINSV","BAJAJHLDNG","BAJFINANCE",
    "BANDHANBNK","BANKBARODA","BANKINDIA","BANKNIFTY","BDL","BEL","BHARATFORG","BHARTIARTL",
    "BHEL","BIOCON","BLUESTARCO","BOSCHLTD","BPCL","BRITANNIA","BSE","CAMS","CANBK","CDSL",
    "CGPOWER","CHOLAFIN","CIPLA","COALINDIA","COFORGE","COLPAL","CONCOR","CROMPTON","CUMMINSIND",
    "DABUR","DALBHARAT","DELHIVERY","DIVISLAB","DIXON","DLF","DMART","DRREDDY","EICHERMOT",
    "ETERNAL","EXIDEIND","FEDERALBNK","FINNIFTY","FORTIS","GAIL","GLENMARK","GMRAIRPORT",
    "GODREJCP","GODREJPROP","GRASIM","HAL","HAVELLS","HCLTECH","HDFCAMC","HDFCBANK","HDFCLIFE",
    "HEROMOTOCO","HINDALCO","HINDPETRO","HINDUNILVR","HINDZINC","HUDCO","ICICIBANK","ICICIGI",
    "ICICIPRULI","IDEA","IDFCFIRSTB","IEX","IIFL","INDHOTEL","INDIANB","INDIGO","INDUSINDBK",
    "INDUSTOWER","INFY","INOXWIND","IOC","IRCTC","IREDA","IRFC","ITC","JINDALSTEL","JIOFIN",
    "JSWENERGY","JSWSTEEL","JUBLFOOD","KALYANKJIL","KAYNES","KEI","KFINTECH","KOTAKBANK",
    "KPITTECH","LAURUSLABS","LICHSGFIN","LICI","LODHA","LT","LTF","LTIM","LUPIN","M&M",
    "MANAPPURAM","MANKIND","MARICO","MARUTI","MAXHEALTH","MAZDOCK","MCX","MFSL","MIDCPNIFTY",
    "MOTHERSON","MPHASIS","MUTHOOTFIN","NATIONALUM","NAUKRI","NBCC","NESTLEIND","NHPC",
    "NIFTY","NIFTYNXT50","NMDC","NTPC","NUVAMA","NYKAA","OBEROIRLTY","OFSS","OIL","ONGC",
    "PAGEIND","PATANJALI","PAYTM","PERSISTENT","PETRONET","PFC","PGEL","PHOENIXLTD",
    "PIDILITIND","PIIND","PNB","PNBHOUSING","POLICYBZR","POLYCAB","POWERGRID","POWERINDIA",
    "PPLPHARMA","PREMIERENE","PRESTIGE","RBLBANK","RECLTD","RELIANCE","RVNL","SAIL",
    "SAMMAANCAP","SBICARD","SBILIFE","SBIN","SHREECEM","SHRIRAMFIN","SIEMENS","SOLARINDS",
    "SONACOMS","SRF","SUNPHARMA","SUPREMEIND","SUZLON","SWIGGY","SYNGENE","TATACONSUM",
    "TATAELXSI","TATAPOWER","TATASTEEL","TATATECH","TCS","TECHM","TIINDIA","TITAN","TMPV",
    "TORNTPHARM","TORNTPOWER","TRENT","TVSMOTOR","ULTRACEMCO","UNIONBANK","UNITDSPR",
    "UNOMINDA","UPL","VBL","VEDL","VOLTAS","WAAREEENER","WIPRO","YESBANK","ZYDUSLIFE"
]
# ==================================================
# GITHUB CONFIG
# ==================================================
GITHUB_TOKEN = st.secrets["GITHUB_TOKEN"]
GITHUB_REPO = st.secrets["GITHUB_REPO"]
GITHUB_BRANCH = st.secrets.get("GITHUB_BRANCH", "main")

# ==================================================
# INIT KITE
# ==================================================
kite = KiteConnect(api_key=API_KEY)
kite.set_access_token(ACCESS_TOKEN)

@st.cache_data(show_spinner=False)
def load_instruments():
    return pd.DataFrame(kite.instruments("NFO"))

instruments = load_instruments()

# ==================================================
# HELPERS
# ==================================================
def chunk(lst, size=200):
    for i in range(0, len(lst), size):
        yield lst[i:i + size]

def compute_max_pain(df):
    df = df.fillna(0)
    mp = []
    for i in range(len(df)):
        val = (
            -sum(df["CE_LTP"][i:] * df["CE_OI"][i:])
            + df["Strike"].iloc[i] * sum(df["CE_OI"][:i])
            - sum(df["Strike"][:i] * df["CE_OI"][:i])
            - sum(df["PE_LTP"][:i] * df["PE_OI"][:i])
            + sum(df["Strike"][i:] * df["PE_OI"][i:])
            - df["Strike"].iloc[i] * sum(df["PE_OI"][i:])
        )
        mp.append(int(val / 10000))
    df["Max_Pain"] = mp
    return df

# ==================================================
# FETCH FULL OPTION CHAIN
# ==================================================
def fetch_full_option_chain():
    option_map, symbols = {}, []

    for stock in STOCKS:
        df = instruments[
            (instruments["name"] == stock) &
            (instruments["segment"] == "NFO-OPT")
        ].copy()

        if df.empty:
            continue

        df["expiry"] = pd.to_datetime(df["expiry"])
        expiry = df["expiry"].min()
        df = df[df["expiry"] == expiry]

        option_map[stock] = df
        symbols.extend(["NFO:" + s for s in df["tradingsymbol"]])

    option_quotes = {}
    for batch in chunk(symbols):
        option_quotes.update(kite.quote(batch))

    spot_quotes = kite.quote([f"NSE:{s}" for s in option_map])

    rows_all = []
    now_ts = datetime.now(IST).strftime("%Y-%m-%d %H:%M:%S")

    for stock, df in option_map.items():
        spot = spot_quotes.get(f"NSE:{stock}", {})
        ltp = spot.get("last_price")
        ohlc = spot.get("ohlc", {})
        prev_close = ohlc.get("close")

        pct = (
            round(((ltp - prev_close) / prev_close) * 100, 3)
            if ltp and prev_close else None
        )

        rows = []
        for strike in sorted(df["strike"].unique()):
            ce = df[(df["strike"] == strike) & (df["instrument_type"] == "CE")]
            pe = df[(df["strike"] == strike) & (df["instrument_type"] == "PE")]

            ce_q = option_quotes.get("NFO:" + ce.iloc[0]["tradingsymbol"], {}) if not ce.empty else {}
            pe_q = option_quotes.get("NFO:" + pe.iloc[0]["tradingsymbol"], {}) if not pe.empty else {}

            rows.append({
                "Stock": stock,
                "Expiry": df["expiry"].iloc[0].date(),
                "Strike": strike,
                "CE_LTP": ce_q.get("last_price"),
                "CE_OI": ce_q.get("oi"),
                "CE_Volume": ce_q.get("volume"),
                "PE_LTP": pe_q.get("last_price"),
                "PE_OI": pe_q.get("oi"),
                "PE_Volume": pe_q.get("volume"),
                "Stock_LTP": ltp,
                "Stock_High": ohlc.get("high"),
                "Stock_Low": ohlc.get("low"),
                "Stock_%_Change": pct,
                "timestamp": now_ts
            })

        sdf = pd.DataFrame(rows).sort_values("Strike")
        sdf = compute_max_pain(sdf)
        rows_all.append(sdf)

    return pd.concat(rows_all, ignore_index=True)

# ==================================================
# PUSH CSV (LOCAL + GITHUB)
# ==================================================
def push_csv_to_github(df):
    ts = datetime.now(IST).strftime("%Y-%m-%d_%H-%M-%S")
    filename = f"{DATA_DIR}/option_chain_{ts}.csv"

    df.to_csv(filename, index=False)

    content = base64.b64encode(df.to_csv(index=False).encode()).decode()
    url = f"https://api.github.com/repos/{GITHUB_REPO}/contents/{filename}"

    headers = {
        "Authorization": f"token {GITHUB_TOKEN}",
        "Accept": "application/vnd.github+json"
    }

    payload = {
        "message": f"Auto snapshot {filename}",
        "content": content,
        "branch": GITHUB_BRANCH
    }

    r = requests.put(url, headers=headers, json=payload)
    if r.status_code not in (200, 201):
        raise Exception(r.json())

    return filename

# ==================================================
# FETCH → SAVE → DISPLAY
# ==================================================
with st.spinner("📡 Fetching FULL option chain..."):
    df_live = fetch_full_option_chain()

st.dataframe(df_live, use_container_width=True)

saved_file = push_csv_to_github(df_live)
st.success(f"✅ Saved FULL snapshot: {saved_file}")

# ==================================================
# ================= TABLE 1 (UNCHANGED) =================
# ==================================================
# 🔻 EXACT CODE YOU PROVIDED — NOT MODIFIED 🔻

def load_csv_files():
    files = []
    for f in os.listdir(DATA_DIR):
        if f.startswith("option_chain_") and f.endswith(".csv"):
            ts = f.replace("option_chain_", "").replace(".csv", "")
            files.append((ts, os.path.join(DATA_DIR, f)))
    return sorted(files, reverse=True)

csv_files = load_csv_files()
if len(csv_files) < 3:
    st.error("Need at least 3 CSV files.")
    st.stop()

timestamps_all = [ts for ts, _ in csv_files]
file_map = dict(csv_files)

def extract_time(ts):
    try:
        hh, mm = map(int, ts.split("_")[-1].split("-")[:2])
        return time(hh, mm)
    except:
        return None

filtered_ts = [
    ts for ts in timestamps_all
    if extract_time(ts) and time(8, 0) <= extract_time(ts) <= time(16, 30)
]

st.subheader("🕒 Timestamp Selection")
c1, c2, c3 = st.columns(3)
t1 = c1.selectbox("Timestamp 1", filtered_ts, 0)
t2 = c2.selectbox("Timestamp 2", filtered_ts, 1)
t3 = c3.selectbox("Timestamp 3", filtered_ts, 2)

df_t1 = pd.read_csv(file_map[t1])
df_t2 = pd.read_csv(file_map[t2])
df_t3 = pd.read_csv(file_map[t3])

# 🔻 FULL TABLE LOGIC CONTINUES EXACTLY AS PROVIDED 🔻
# (Already validated — no changes needed)
# ==================================================
# LOAD CSVs ONCE
# ==================================================
df_t1 = pd.read_csv(file_map[t1])
df_t2 = pd.read_csv(file_map[t2])
df_t3 = pd.read_csv(file_map[t3])

# ==================================================
# ================= TABLE 1 ========================
# ==================================================
st.subheader("📘 Table 1 – FnO MP Delta Dashboard")

dfs = []
for i, d in enumerate([df_t1, df_t2, df_t3]):
    dfs.append(
        d[[
            "Stock", "Strike", "Max_Pain", "Stock_LTP",
            "CE_OI", "PE_OI", "CE_Volume", "PE_Volume"
        ]].rename(columns={
            "Max_Pain": f"MP_{i}",
            "Stock_LTP": f"LTP_{i}",
            "CE_OI": f"CE_OI_{i}",
            "PE_OI": f"PE_OI_{i}",
            "CE_Volume": f"CE_VOL_{i}",
            "PE_Volume": f"PE_VOL_{i}",
        })
    )

df1 = dfs[0].merge(dfs[1], on=["Stock", "Strike"]).merge(dfs[2], on=["Stock", "Strike"])

df1 = df1.merge(
    df_t1[["Stock", "Strike", "Stock_%_Change", "Stock_High", "Stock_Low"]],
    on=["Stock", "Strike"],
    how="left"
)

for c in df1.columns:
    if any(x in c for x in ["MP_", "LTP_", "OI_", "VOL_"]):
        df1[c] = pd.to_numeric(df1[c], errors="coerce").fillna(0)

df1["Δ MP TS1-TS2"] = df1["MP_0"] - df1["MP_1"]
df1["Δ MP TS2-TS3"] = df1["MP_1"] - df1["MP_2"]

df1["Δ CE OI TS1-TS2"] = df1["CE_OI_0"] - df1["CE_OI_1"]
df1["Δ PE OI TS1-TS2"] = df1["PE_OI_0"] - df1["PE_OI_1"]
df1["Δ CE Vol TS1-TS2"] = df1["CE_VOL_0"] - df1["CE_VOL_1"]
df1["Δ PE Vol TS1-TS2"] = df1["PE_VOL_0"] - df1["PE_VOL_1"]

# ==================================================
# PE / CE VOL RATIO (ATM WINDOW)
# ==================================================

df1["PE/CE Vol Ratio"] = np.nan

for stock, g in df1.groupby("Stock"):
    g = g.sort_values("Strike").reset_index()

    ltp = g["LTP_0"].iloc[0]

    # ATM strike index
    atm_idx = (g["Strike"] - ltp).abs().idxmin()

    # Define windows safely
    pe_idx = g.loc[
        max(0, atm_idx-2) : min(len(g)-1, atm_idx+1)
    ].index

    ce_idx = g.loc[
        atm_idx : min(len(g)-1, atm_idx+3)
    ].index

    pe_sum = g.loc[pe_idx, "Δ PE Vol TS1-TS2"].sum()
    ce_sum = g.loc[ce_idx, "Δ CE Vol TS1-TS2"].sum()

    ratio = pe_sum / ce_sum if ce_sum != 0 else np.nan

    df1.loc[g["index"], "PE/CE Vol Ratio"] = round(ratio, 2)

# ---- ATM PAIR (BELOW + ABOVE LTP) PE–CE DIFFERENCE ----

df1["Δ (PE-CE) OI TS1-TS2"] = np.nan
df1["Δ (PE-CE) Vol TS1-TS2"] = np.nan

for stock, g in df1.groupby("Stock"):
    g = g.sort_values("Strike")
    ltp = g["LTP_0"].iloc[0]

    below_candidates = g[g["Strike"] <= ltp]
    above_candidates = g[g["Strike"] > ltp]
    
    if below_candidates.empty or above_candidates.empty:
        continue   # ❗ skip this stock safely
    
    below = below_candidates.iloc[-1]
    above = above_candidates.iloc[0]



    pe_oi_sum = (
        below["Δ PE OI TS1-TS2"] +
        above["Δ PE OI TS1-TS2"]
    )
    ce_oi_sum = (
        below["Δ CE OI TS1-TS2"] +
        above["Δ CE OI TS1-TS2"]
    )

    pe_vol_sum = (
        below["Δ PE Vol TS1-TS2"] +
        above["Δ PE Vol TS1-TS2"]
    )
    ce_vol_sum = (
        below["Δ CE Vol TS1-TS2"] +
        above["Δ CE Vol TS1-TS2"]
    )

    df1.loc[g.index, "Δ (PE-CE) OI TS1-TS2"] = pe_oi_sum - ce_oi_sum
    df1.loc[g.index, "Δ (PE-CE) Vol TS1-TS2"] = pe_vol_sum - ce_vol_sum



df1["% Stock Ch TS1-TS2"] = ((df1["LTP_0"] - df1["LTP_1"]) / df1["LTP_1"]) * 100
df1["% Stock Ch TS2-TS3"] = ((df1["LTP_1"] - df1["LTP_2"]) / df1["LTP_2"]) * 100
df1["Stock_LTP"] = df1["LTP_0"]

# ---- TS3 COLUMNS MOVED TO END ----
df1 = df1[[
    "Stock", 
    "Strike",
    "Δ MP TS1-TS2",
    "Δ CE OI TS1-TS2", 
    "Δ PE OI TS1-TS2",
    "Δ CE Vol TS1-TS2", 
    "Δ PE Vol TS1-TS2",
    "Δ (PE-CE) OI TS1-TS2",
    "Δ (PE-CE) Vol TS1-TS2",
    "PE/CE Vol Ratio",
    "% Stock Ch TS1-TS2",
    "% Stock Ch TS2-TS3",
    "Stock_LTP", 
    "Stock_%_Change", 
    "Stock_High", 
    "Stock_Low",
    "Δ MP TS2-TS3", 

]]

# ---- RENAME DELTA COLUMNS (DISPLAY ONLY) ----
df1 = df1.rename(columns={
    "Δ MP TS1-TS2": "Δ MP",
    "Δ CE OI TS1-TS2": "Δ CE OI",
    "Δ PE OI TS1-TS2": "Δ PE OI",
    "Δ CE Vol TS1-TS2": "Δ CE Vol",
    "Δ PE Vol TS1-TS2": "Δ PE Vol",
    "PE/CE Vol Ratio":  "Δ PE/CE Vol",
    "% Stock Ch TS1-TS2": "% Ch 1-2",
    "% Stock Ch TS2-TS3": "% Ch 2-3",
    "Stock_%_Change": "% Ch",
    "Δ (PE-CE) OI TS1-TS2": "Δ (PE-CE) OI",
    "Δ (PE-CE) Vol TS1-TS2": "Δ (PE-CE) Vol",
})


def filter_strikes(df, n=4):
    blocks = []
    for _, g in df.groupby("Stock"):
        g = g.sort_values("Strike").reset_index(drop=True)
        atm = (g["Strike"] - g["Stock_LTP"].iloc[0]).abs().idxmin()
        blocks.append(g.iloc[max(0, atm-n):atm+n])
    return pd.concat(blocks[:-1], ignore_index=True)

display_df1 = filter_strikes(df1)

def highlight_table1(data):
    styles = pd.DataFrame("", index=data.index, columns=data.columns)

    # ✅ updated column name
    required_cols = {"Stock", "Strike", "Stock_LTP", "Δ MP"}
    if not required_cols.issubset(data.columns):
        return styles

    for stock in data["Stock"].dropna().unique():
        sdf = data[(data["Stock"] == stock) & data["Strike"].notna()]

        if sdf.empty:
            continue

        ltp = sdf["Stock_LTP"].iloc[0]
        strikes = sdf["Strike"].values

        # 🔵 ATM pair highlight (below + above LTP)
        for i in range(len(strikes) - 1):
            if strikes[i] <= ltp <= strikes[i + 1]:
                styles.loc[sdf.index[i]] = "background-color:#003366;color:white"
                styles.loc[sdf.index[i + 1]] = "background-color:#003366;color:white"
                break

        # 🔴 Max Δ MP highlight
        idx = sdf["Δ MP"].abs().idxmax()
        styles.loc[idx] = "background-color:#8B0000;color:white"

    return styles


fmt = {
    c: "{:.0f}"
    for c in display_df1.select_dtypes("number").columns
    if c != "Δ PE/CE Vol"
}

fmt.update({
    "Stock_LTP": "{:.2f}",
    "% Ch": "{:.2f}",        # Stock_%_Change
    "% Ch 1-2": "{:.2f}",    # TS1 → TS2
    "% Ch 2-3": "{:.2f}",    # TS2 → TS3
    "Δ PE/CE Vol": "{:.2f}",   # ✅ CORRECT NAME
})

# ==================================================
# RATIO COUNT CONTROL (NON-FILTERING)
# ==================================================

st.subheader("📊 PE/CE Volume Ratio – Count")

rc1, rc2 = st.columns(2)

with rc1:
    ratio_operator = st.selectbox(
        "Ratio Condition",
        [">=", "<="],
        index=0
    )

with rc2:
    ratio_threshold = st.number_input(
        "Ratio Value",
        min_value=0.1,
        max_value=10.0,
        value=1.0,
        step=0.1
    )

# ---- COUNT (STOCK-LEVEL, NOT STRIKE-LEVEL) ----
ratio_df = (
    df1.groupby("Stock")["Δ PE/CE Vol"]
    .first()
    .dropna()
)


if ratio_operator == ">=":
    ratio_count = (ratio_df >= ratio_threshold).sum()
else:
    ratio_count = (ratio_df <= ratio_threshold).sum()

st.metric(
    label=f"Stocks with PE/CE Vol Ratio {ratio_operator} {ratio_threshold}",
    value=int(ratio_count)
)

st.dataframe(display_df1.style.apply(highlight_table1, axis=None).format(fmt, na_rep=""),
             use_container_width=True)



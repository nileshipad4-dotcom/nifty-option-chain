# ================================
# KITE + DHAN COMBINED COLLECTOR
# ================================

# -------- KITE PART (UNCHANGED) --------
import streamlit as st
import pandas as pd
from kiteconnect import KiteConnect
from datetime import datetime, timedelta
import pytz
import base64
import requests
import os
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
# KITE CONFIG (UNCHANGED)
# ==================================================
API_KEY = "bkgv59vaazn56c42"
ACCESS_TOKEN = "DLSvBzTX0sVAn5In8dDj0vRtC6gVbs4P"

STOCKS = [
    "RELIANCE","SBIN","TCS","INFY"
]

# ==================================================
# GITHUB CONFIG
# ==================================================
GITHUB_TOKEN = st.secrets["GITHUB_TOKEN"]
GITHUB_REPO = st.secrets["KITE_REPO"]
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
def chunk(lst, size=100):
    for i in range(0, len(lst), size):
        yield lst[i:i + size]

def compute_max_pain_kite(df):
    df = df.fillna(0)
    A = df["CE_LTP"]
    B = df["CE_OI"]
    G = df["Strike"]
    M = df["PE_LTP"]
    L = df["PE_OI"]

    mp = []
    for i in range(len(df)):
        val = (
            -sum(A[i:] * B[i:])
            + G.iloc[i] * sum(B[:i]) - sum(G[:i] * B[:i])
            - sum(M[:i] * L[:i])
            + sum(G[i:] * L[i:]) - G.iloc[i] * sum(L[i:])
        )
        mp.append(int(val / 10000))

    df["Max_Pain"] = mp
    return df

# ==================================================
# FETCH KITE OPTION CHAIN (UNCHANGED)
# ==================================================
def fetch_full_option_chain():
    option_map = {}
    all_option_symbols = []

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
        all_option_symbols.extend(["NFO:" + ts for ts in df["tradingsymbol"]])

    option_quotes = {}
    for batch in chunk(all_option_symbols):
        try:
            option_quotes.update(kite.quote(batch))
        except:
            pass

    spot_quotes = kite.quote([f"NSE:{s}" for s in option_map.keys()])

    all_data = []
    now_ts = datetime.now(IST).strftime("%Y-%m-%d %H:%M:%S")

    for stock, df in option_map.items():
        rows = []

        spot = spot_quotes.get(f"NSE:{stock}", {})
        stock_ltp = spot.get("last_price")
        ohlc = spot.get("ohlc", {})
        prev_close = ohlc.get("close")

        pct_change = (
            round(((stock_ltp - prev_close) / prev_close) * 100, 3)
            if stock_ltp and prev_close else None
        )

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
                "Stock_LTP": stock_ltp,
                "Stock_High": ohlc.get("high"),
                "Stock_Low": ohlc.get("low"),
                "Stock_%_Change": pct_change,
                "timestamp": now_ts,
            })

        stock_df = pd.DataFrame(rows).sort_values("Strike")
        stock_df = compute_max_pain_kite(stock_df)
        all_data.append(stock_df)

    return pd.concat(all_data, ignore_index=True)

# ==================================================
# PUSH KITE CSV TO GITHUB (UNCHANGED)
# ==================================================
def push_csv_to_github(df):
    filename = f"data/option_chain_{datetime.now(IST).strftime('%Y-%m-%d_%H-%M')}.csv"
    csv_bytes = df.to_csv(index=False).encode()
    content = base64.b64encode(csv_bytes).decode()

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
# RUN KITE
# ==================================================
with st.spinner("📡 Fetching KITE option chain..."):
    df_kite = fetch_full_option_chain()

st.dataframe(df_kite, use_container_width=True)

try:
    saved_file = push_csv_to_github(df_kite)
    st.success(f"✅ Kite CSV saved: {saved_file}")
except Exception as e:
    st.error(f"❌ GitHub save failed: {e}")

# -------- DHAN PART (FIXED, ORIGINAL BEHAVIOR) --------

# ================= CONFIG =================
CLIENT_ID = "1102712380"
ACCESS_TOKEN_DHAN = "eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzUxMiJ9.eyJpc3MiOiJkaGFuIiwicGFydG5lcklkIjoiIiwiZXhwIjoxNzY4MjQ5NzkwLCJpYXQiOjE3NjgxNjMzOTAsInRva2VuQ29uc3VtZXJUeXBlIjoiU0VMRiIsIndlYmhvb2tVcmwiOiIiLCJkaGFuQ2xpZW50SWQiOiIxMTAyNzEyMzgwIn0.y9CAHmsZCpTVulTRcK8AuiE_vaIK1-nSQ1TSqaG8zO1x8BPX2kodNgdLNPfF_P5hB_tiJUJY3bSEj-kf-0ypDw"

API_BASE = "https://api.dhan.co/v2"

UNDERLYINGS = {
    "NIFTY":      {"scrip": 13,  "seg": "IDX_I", "center": 26000},
    "BANKNIFTY":  {"scrip": 25,  "seg": "IDX_I", "center": 60000},
    "MIDCPNIFTY": {"scrip": 442, "seg": "IDX_I", "center": 13600},
    "SENSEX":     {"scrip": 51,  "seg": "IDX_I", "center": 84000},
}

HEADERS = {
    "client-id": CLIENT_ID,
    "access-token": ACCESS_TOKEN_DHAN,
    "Content-Type": "application/json",
}

DATA_DIR = "data"
os.makedirs(DATA_DIR, exist_ok=True)

BASE_COLUMNS = [
    "Strike",
    "CE LTP","CE OI","CE Volume","CE IV","CE Delta","CE Gamma","CE Vega",
    "PE LTP","PE OI","PE Volume","PE IV","PE Delta","PE Gamma","PE Vega",
    "timestamp",
    "Max Pain",
]

# Create CSVs only if not exist
for sym in ["nifty", "banknifty", "midcpnifty", "sensex"]:
    path = os.path.join(DATA_DIR, f"{sym}.csv")
    if not os.path.exists(path):
        pd.DataFrame(columns=BASE_COLUMNS).to_csv(path, index=False)

def get_expiries(scrip, seg):
    r = requests.post(
        f"{API_BASE}/optionchain/expirylist",
        headers=HEADERS,
        json={"UnderlyingScrip": scrip, "UnderlyingSeg": seg}
    )
    return r.json().get("data", []) if r.status_code == 200 else []

def get_option_chain(scrip, seg, expiry):
    r = requests.post(
        f"{API_BASE}/optionchain",
        headers=HEADERS,
        json={"UnderlyingScrip": scrip, "UnderlyingSeg": seg, "Expiry": expiry}
    )
    return r.json().get("data") if r.status_code == 200 else None

def compute_max_pain_dhan(df):
    A = df["CE LTP"].fillna(0)
    B = df["CE OI"].fillna(0)
    G = df["Strike"]
    M = df["PE LTP"].fillna(0)
    L = df["PE OI"].fillna(0)

    mp = []
    for i in range(len(df)):
        val = (
            -sum(A[i:] * B[i:])
            + G.iloc[i] * sum(B[:i]) - sum(G[:i] * B[:i])
            - sum(M[:i] * L[:i])
            + sum(G[i:] * L[i:]) - G.iloc[i] * sum(L[i:])
        )
        mp.append(int(val / 10000))

    df["Max Pain"] = mp
    return df

def run_dhan_collector():
    ts = (datetime.utcnow() + timedelta(hours=5, minutes=30)).strftime("%Y-%m-%d %H:%M")

    for sym, cfg in UNDERLYINGS.items():
        expiries = get_expiries(cfg["scrip"], cfg["seg"])
        if not expiries:
            continue

        data = get_option_chain(cfg["scrip"], cfg["seg"], expiries[0])
        oc = data.get("oc", {}) if data else {}

        strikes = sorted(float(s) for s in oc.keys())
        center = cfg["center"]

        below = [s for s in strikes if s <= center][-35:]
        above = [s for s in strikes if s > center][:36]
        selected = sorted(set(below + above))

        rows = []

        for s in selected:
            v = oc.get(f"{s:.6f}", {})
            ce = v.get("ce", {})
            pe = v.get("pe", {})

            rows.append({
                "Strike": int(s),
                "CE LTP": ce.get("last_price"),
                "CE OI": ce.get("oi"),
                "CE Volume": ce.get("volume"),
                "CE IV": ce.get("implied_volatility"),
                "CE Delta": ce.get("greeks", {}).get("delta"),
                "CE Gamma": ce.get("greeks", {}).get("gamma"),
                "CE Vega": ce.get("greeks", {}).get("vega"),
                "PE LTP": pe.get("last_price"),
                "PE OI": pe.get("oi"),
                "PE Volume": pe.get("volume"),
                "PE IV": pe.get("implied_volatility"),
                "PE Delta": pe.get("greeks", {}).get("delta"),
                "PE Gamma": pe.get("greeks", {}).get("gamma"),
                "PE Vega": pe.get("greeks", {}).get("vega"),
                "timestamp": ts,
            })

        if not rows:
            continue

        df = pd.DataFrame(rows).sort_values("Strike")

        for c in BASE_COLUMNS:
            if c not in df.columns:
                df[c] = None

        df = compute_max_pain_dhan(df)
        df = df[BASE_COLUMNS]

        out = os.path.join(DATA_DIR, f"{sym.lower()}.csv")
        df.to_csv(out, mode="a", header=False, index=False)

        print(f"[OK] {sym} saved @ {ts}")

run_dhan_collector()

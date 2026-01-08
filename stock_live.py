import streamlit as st
import pandas as pd
from kiteconnect import KiteConnect
from datetime import datetime
import pytz
import base64
import requests
from streamlit_autorefresh import st_autorefresh

# ==================================================
# STREAMLIT CONFIG
# ==================================================
st.set_page_config(page_title="LIVE Option Chain Snapshot", layout="wide")
st.title("📊 LIVE Option Chain → GitHub Snapshot")

# ⏱ Auto refresh every 60 sec
refresh_tick = st_autorefresh(interval=60_000, key="live_refresh")

# ==================================================
# TIMEZONE
# ==================================================
IST = pytz.timezone("Asia/Kolkata")

# ==================================================
# CONFIG
# ==================================================
API_KEY = "bkgv59vaazn56c42"
ACCESS_TOKEN = "um1gYW2GgQ94kdg2G1C9vu3cWfdFF00X"

STOCKS = ["NIFTY","BANKNIFTY","FINNIFTY"]

# ==================================================
# GITHUB CONFIG (FROM SECRETS)
# ==================================================
GITHUB_TOKEN = st.secrets["GITHUB_TOKEN"]
GITHUB_REPO = st.secrets["GITHUB_REPO"]
GITHUB_BRANCH = st.secrets.get("GITHUB_BRANCH", "main")

# ==================================================
# INIT KITE
# ==================================================
kite = KiteConnect(api_key=API_KEY)
kite.set_access_token(ACCESS_TOKEN)

# ==================================================
# FETCH LIVE DATA (MINIMAL EXAMPLE)
# ==================================================
def fetch_live_data():
    spot = kite.quote([f"NSE:{s}" for s in STOCKS])
    rows = []
    ts = datetime.now(IST).strftime("%Y-%m-%d %H:%M:%S")

    for s in STOCKS:
        q = spot.get(f"NSE:{s}", {})
        rows.append({
            "Stock": s,
            "LTP": q.get("last_price"),
            "High": q.get("ohlc", {}).get("high"),
            "Low": q.get("ohlc", {}).get("low"),
            "timestamp": ts
        })

    return pd.DataFrame(rows)

# ==================================================
# PUSH CSV TO GITHUB
# ==================================================
def push_csv_to_github(df):
    filename = f"data/option_chain_{datetime.now(IST).strftime('%Y-%m-%d_%H-%M-%S')}.csv"
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
# FETCH → SAVE → DISPLAY
# ==================================================
with st.spinner("📡 Fetching LIVE data..."):
    df = fetch_live_data()

st.dataframe(df, use_container_width=True)

try:
    saved_file = push_csv_to_github(df)
    st.success(f"✅ Saved to GitHub: {saved_file}")
except Exception as e:
    st.error(f"❌ GitHub save failed: {e}")

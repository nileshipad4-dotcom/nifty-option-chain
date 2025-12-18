import time
import pandas as pd
import streamlit as st

EXPIRY = "16-Dec-2025"
REFRESH_SECONDS = 30

st.set_page_config(page_title="NIFTY Option Chain", layout="wide")


# ---------------------------------
# Create persistent TLS session
# ---------------------------------
@st.cache_resource
def get_session():
    session = tls_client.Session(
        client_identifier="chrome_120",
        random_tls_extension_order=True
    )
    session.headers.update({
        "User-Agent":
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/120.0.0.0 Safari/537.36",
        "Accept": "application/json",
        "Accept-Language": "en-US,en;q=0.9",
        "Referer": "https://www.nseindia.com/option-chain"
    })

    # Initial cookie request
    session.get("https://www.nseindia.com", timeout=10)
    return session


# ---------------------------------
# Fetch NIFTY option chain
# ---------------------------------
@st.cache_data(ttl=REFRESH_SECONDS)
def fetch_nifty_data():
    session = get_session()

    url = "https://www.nseindia.com/api/option-chain-indices?symbol=NIFTY"

    try:
        r = session.get(url, timeout=10)
        r.raise_for_status()
        data = r.json()
    except Exception:
        # refresh cookies once
        session.get("https://www.nseindia.com", timeout=10)
        r = session.get(url, timeout=10)
        data = r.json()

    spot = data["records"]["underlyingValue"]

    rows = []
    ce_oi_total = pe_oi_total = 0
    ce_vol_total = pe_vol_total = 0

    for item in data["records"]["data"]:
        if item.get("expiryDate") != EXPIRY:
            continue

        ce = item.get("CE", {})
        pe = item.get("PE", {})

        ce_oi = ce.get("openInterest", 0)
        pe_oi = pe.get("openInterest", 0)
        ce_vol = ce.get("totalTradedVolume", 0)
        pe_vol = pe.get("totalTradedVolume", 0)

        ce_oi_total += ce_oi
        pe_oi_total += pe_oi
        ce_vol_total += ce_vol
        pe_vol_total += pe_vol

        rows.append({
            "Strike": item["strikePrice"],
            "CE LTP": ce.get("lastPrice"),
            "CE OI": ce_oi,
            "CE Volume": ce_vol,
            "PE LTP": pe.get("lastPrice"),
            "PE OI": pe_oi,
            "PE Volume": pe_vol,
        })

    pcr_oi = round(pe_oi_total / ce_oi_total, 2) if ce_oi_total else 0
    pcr_vol = round(pe_vol_total / ce_vol_total, 2) if ce_vol_total else 0

    df = pd.DataFrame(rows).sort_values("Strike")

    return spot, pcr_oi, pcr_vol, df


# ---------------------------------
# UI
# ---------------------------------
st.title("📊 NIFTY Option Chain")
st.write(f"**Expiry:** {EXPIRY}")

try:
    spot, pcr_oi, pcr_vol, df = fetch_nifty_data()

    c1, c2, c3 = st.columns(3)
    c1.metric("NIFTY Spot", spot)
    c2.metric("Put / Call OI", pcr_oi)
    c3.metric("Put / Call Volume", pcr_vol)

    st.divider()
    st.dataframe(df, use_container_width=True, height=600)

except Exception:
    st.error("NSE data unavailable. Please refresh.")
    st.stop()

st.caption(f"Auto refresh every {REFRESH_SECONDS} seconds")

# Auto refresh
time.sleep(REFRESH_SECONDS)
st.rerun()

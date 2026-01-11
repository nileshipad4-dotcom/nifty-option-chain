# combined_collector.py
# KITE + DHAN → Separate GitHub Repos (Secrets from App Settings)

import requests
import pandas as pd
from datetime import datetime
import pytz
import base64
import os
from kiteconnect import KiteConnect

# ==================================================
# TIMEZONE
# ==================================================
IST = pytz.timezone("Asia/Kolkata")

# ==================================================
# GITHUB CONFIG (FROM APP SETTINGS)
# ==================================================
GITHUB_TOKEN = os.environ["GITHUB_TOKEN"]
KITE_REPO = os.environ["KITE_REPO"]
DHAN_REPO = os.environ["DHAN_REPO"]
GITHUB_BRANCH = os.environ.get("GITHUB_BRANCH", "main")

# ==================================================
# KITE CONFIG
# ==================================================
KITE_API_KEY = "bkgv59vaazn56c42"
KITE_ACCESS_TOKEN = "q3g4WBB3vq2EdvQMVFUyruVY7IcVuhRo"

STOCKS = [
    "360ONE","ABB","ABCAPITAL","ADANIENSOL","ADANIENT","ADANIGREEN","ADANIPORTS","ALKEM",
    "AMBER","AMBUJACEM","ANGELONE","APLAPOLLO","APOLLOHOSP","ASHOKLEY","ASIANPAINT","ASTRAL",
    "AUBANK","AUROPHARMA","AXISBANK","BAJAJ-AUTO","BAJAJFINSV","BAJAJHLDNG","BAJFINANCE",
    "BANDHANBNK","BANKBARODA","BANKINDIA","BDL","BEL","BHARATFORG","BHARTIARTL",
    "BHEL","BIOCON","BLUESTARCO","BOSCHLTD","BPCL","BRITANNIA","BSE","CAMS","CANBK","CDSL",
    "CGPOWER","CHOLAFIN","CIPLA","COALINDIA","COFORGE","COLPAL","CONCOR","CROMPTON","CUMMINSIND",
    "DABUR","DALBHARAT","DELHIVERY","DIVISLAB","DIXON","DLF","DMART","DRREDDY","EICHERMOT",
    "ETERNAL","EXIDEIND","FEDERALBNK","FORTIS","GAIL","GLENMARK","GMRAIRPORT",
    "GODREJCP","GODREJPROP","GRASIM","HAL","HAVELLS","HCLTECH","HDFCAMC","HDFCBANK","HDFCLIFE",
    "HEROMOTOCO","HINDALCO","HINDPETRO","HINDUNILVR","HINDZINC","HUDCO","ICICIBANK","ICICIGI",
    "ICICIPRULI","IDEA","IDFCFIRSTB","IEX","IIFL","INDHOTEL","INDIANB","INDIGO","INDUSINDBK",
    "INDUSTOWER","INFY","INOXWIND","IOC","IRCTC","IREDA","IRFC","ITC","JINDALSTEL","JIOFIN",
    "JSWENERGY","JSWSTEEL","JUBLFOOD","KALYANKJIL","KAYNES","KEI","KFINTECH","KOTAKBANK",
    "KPITTECH","LAURUSLABS","LICHSGFIN","LICI","LODHA","LT","LTF","LTIM","LUPIN","M&M",
    "MANAPPURAM","MANKIND","MARICO","MARUTI","MAXHEALTH","MAZDOCK","MCX","MFSL","MIDCPNIFTY",
    "MOTHERSON","MPHASIS","MUTHOOTFIN","NATIONALUM","NAUKRI","NBCC","NESTLEIND","NHPC", "NMDC","NTPC","NUVAMA","NYKAA","OBEROIRLTY","OFSS","OIL","ONGC",
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
# DHAN CONFIG
# ==================================================
CLIENT_ID = "1102712380"
ACCESS_TOKEN = "eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzUxMiJ9.eyJpc3MiOiJkaGFuIiwicGFydG5lcklkIjoiIiwiZXhwIjoxNzY4MjQ5NzkwLCJpYXQiOjE3NjgxNjMzOTAsInRva2VuQ29uc3VtZXJUeXBlIjoiU0VMRiIsIndlYmhvb2tVcmwiOiIiLCJkaGFuQ2xpZW50SWQiOiIxMTAyNzEyMzgwIn0.y9CAHmsZCpTVulTRcK8AuiE_vaIK1-nSQ1TSqaG8zO1x8BPX2kodNgdLNPfF_P5hB_tiJUJY3bSEj-kf-0ypDw"
API_BASE = "https://api.dhan.co/v2"

UNDERLYINGS = {
    "NIFTY": {"scrip": 13, "seg": "IDX_I", "center": 26000},
    "BANKNIFTY": {"scrip": 25, "seg": "IDX_I", "center": 60000},
    "MIDCPNIFTY": {"scrip": 442, "seg": "IDX_I", "center": 13600},
    "SENSEX": {"scrip": 51, "seg": "IDX_I", "center": 84000},
}

HEADERS = {
    "client-id": CLIENT_ID,
    "access-token": ACCESS_TOKEN,
    "Content-Type": "application/json",
}

# ==================================================
# GITHUB PUSH
# ==================================================
def push_csv_to_github(df, filename, repo):
    content = base64.b64encode(df.to_csv(index=False).encode()).decode()

    url = f"https://api.github.com/repos/{repo}/contents/{filename}"
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

    print(f"[GITHUB] {repo}/{filename} uploaded")

# ==================================================
# MAX PAIN
# ==================================================
def compute_max_pain(df, ce_ltp, ce_oi, pe_ltp, pe_oi):
    A = df[ce_ltp].fillna(0)
    B = df[ce_oi].fillna(0)
    G = df["Strike"]
    M = df[pe_ltp].fillna(0)
    L = df[pe_oi].fillna(0)

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

# ==================================================
# KITE FETCH → KITE_REPO
# ==================================================
def fetch_kite_data():
    kite = KiteConnect(api_key=KITE_API_KEY)
    kite.set_access_token(KITE_ACCESS_TOKEN)

    instruments = pd.DataFrame(kite.instruments("NFO"))
    ts = datetime.now(IST).strftime("%Y-%m-%d %H:%M")

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

        option_symbols = ["NFO:" + s for s in df["tradingsymbol"]]
        option_quotes = {}

        for i in range(0, len(option_symbols), 200):
            option_quotes.update(kite.quote(option_symbols[i:i+200]))

        spot = kite.quote([f"NSE:{stock}"]).get(f"NSE:{stock}", {})
        stock_ltp = spot.get("last_price")
        ohlc = spot.get("ohlc", {})
        prev_close = ohlc.get("close")

        pct_change = (
            round(((stock_ltp - prev_close) / prev_close) * 100, 3)
            if stock_ltp and prev_close else None
        )

        rows = []

        for strike in sorted(df["strike"].unique()):
            ce = df[(df["strike"] == strike) & (df["instrument_type"] == "CE")]
            pe = df[(df["strike"] == strike) & (df["instrument_type"] == "PE")]

            ce_q = option_quotes.get("NFO:" + ce.iloc[0]["tradingsymbol"], {}) if not ce.empty else {}
            pe_q = option_quotes.get("NFO:" + pe.iloc[0]["tradingsymbol"], {}) if not pe.empty else {}

            rows.append({
                "Stock": stock,
                "Expiry": expiry.date(),
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

                "timestamp": ts
            })

        stock_df = pd.DataFrame(rows).sort_values("Strike")
        stock_df = compute_max_pain(stock_df, "CE_LTP", "CE_OI", "PE_LTP", "PE_OI")

        filename = f"data/option_chain_{stock}_{ts.replace(':','-').replace(' ','_')}.csv"
        push_csv_to_github(stock_df, filename, KITE_REPO)

# ==================================================
# DHAN HELPERS
# ==================================================
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

# ==================================================
# DHAN FETCH → DHAN_REPO
# ==================================================
def fetch_dhan_data():
    ts = datetime.now(IST).strftime("%Y-%m-%d %H:%M")

    BASE_COLUMNS = [
        "Strike",
        "CE LTP","CE OI","CE Volume","CE IV","CE Delta","CE Gamma","CE Vega",
        "PE LTP","PE OI","PE Volume","PE IV","PE Delta","PE Gamma","PE Vega",
        "timestamp","Max Pain"
    ]

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

                "timestamp": ts
            })

        if not rows:
            continue

        df = pd.DataFrame(rows).sort_values("Strike")

        num_cols = [c for c in df.columns if c not in ["Strike","timestamp"]]
        for c in num_cols:
            df[c] = pd.to_numeric(df[c], errors="coerce")

        df = compute_max_pain(df, "CE LTP", "CE OI", "PE LTP", "PE OI")
        df = df[BASE_COLUMNS]

        filename = f"data/{sym.lower()}_{ts.replace(':','-').replace(' ','_')}.csv"
        push_csv_to_github(df, filename, DHAN_REPO)

# ==================================================
# MAIN
# ==================================================
if __name__ == "__main__":
    print("Running KITE collector...")
    fetch_kite_data()

    print("Running DHAN collector...")
    fetch_dhan_data()

    print("ALL DONE.")

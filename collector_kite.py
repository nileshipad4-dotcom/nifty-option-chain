from kiteconnect import KiteConnect
import pandas as pd
from datetime import datetime
import pytz
import os

# ==================================================
# TIMEZONE (IST)
# ==================================================
IST = pytz.timezone("Asia/Kolkata")

# ==================================================
# CONFIG (ENTER YOUR CREDENTIALS HERE)
# ==================================================
API_KEY = "bkgv59vaazn56c42"
ACCESS_TOKEN = "giMaA7SuUyWA1r9P34zVTOVlPcWG847C"

STOCKS = [
    "360ONE","ABB","ABCAPITAL","ADANIENT","ADANIGREEN","ADANIENSOL","ADANIPORTS",
    "ALKEM","AMBER","AMBUJACEM","ANGELONE","APLAPOLLO","APOLLOHOSP","ASHOKLEY",
    "ASIANPAINT","ASTRAL","AUROPHARMA","AUBANK","AXISBANK","BAJAJ-AUTO",
    "BAJAJFINSV","BAJFINANCE","BANDHANBNK","BANKBARODA","BANKINDIA","BDL","BEL",
    "BHARATFORG","BHARTIARTL","BHEL","BIOCON","BPCL","BRITANNIA","BSE","CAMS",
    "CANBK","CDSL","CGPOWER","CHOLAFIN","CIPLA","COALINDIA","COFORGE","COLPAL",
    "CONCOR","CROMPTON","CUMMINSIND","CYIENT","DABUR","DALBHARAT","DELHIVERY",
    "DIVISLAB","DLF","DMART","DRREDDY","EICHERMOT","ETERNAL","EXIDEIND",
    "FEDERALBNK","FORTIS","GAIL","GMRAIRPORT","GODREJCP","GODREJPROP","GRASIM",
    "HAL","HAVELLS","HCLTECH","HDFCAMC","HDFCBANK","HDFCLIFE","HEROMOTOCO",
    "HFCL","HINDALCO","HINDPETRO","HINDUNILVR","HINDZINC","HUDCO","IEX",
    "ICICIBANK","ICICIGI","ICICIPRULI","IDFCFIRSTB","IIFL","INDHOTEL",
    "INDIANB","INDIGO","INDUSTOWER","INFY","INOXWIND","IOC","IRCTC","IRFC",
    "IREDA","ITC","JINDALSTEL","JIOFIN","JSWENERGY","JSWSTEEL","JUBLFOOD",
    "KALYANKJIL","KAYNES","KEI","KFINTECH","KPITTECH","KOTAKBANK",
    "LAURUSLABS","LICHSGFIN","LICI","LODHA","LT","LTIM","LTF","LUPIN","M&M",
    "MANAPPURAM","MARICO","MARUTI","MAXHEALTH","MAZDOCK","MCX","MFSL",
    "MPHASIS","MOTHERSON","MUTHOOTFIN","NAUKRI","NATIONALUM","NBCC","NCC",
    "NESTLEIND","NHPC","NMDC","NTPC","NUVAMA","NYKAA","OBEROIRLTY","OFSS",
    "OIL","ONGC","PAGEIND","PATANJALI","PAYTM","PERSISTENT","PETRONET",
    "PFC","PGEL","PHOENIXLTD","PIDILITIND","PIIND","PNB","PNBHOUSING",
    "POLICYBZR","POLYCAB","POWERGRID","PRESTIGE","RBLBANK","RECLTD",
    "RELIANCE","RVNL","SAIL","SAMMAANCAP","SBICARD","SBILIFE","SBIN",
    "SHREECEM","SHRIRAMFIN","SIEMENS","SOLARINDS","SONACOMS","SRF",
    "SUNPHARMA","SUZLON","SYNGENE","TATACONSUM","TATAELXSI","TATAPOWER",
    "TATASTEEL","TATATECH","TCS","TECHM","TITAGARH","TITAN","TORNTPHARM",
    "TORNTPOWER","TRENT","TVSMOTOR","ULTRACEMCO","UNITDSPR","UNIONBANK",
    "UNOMINDA","UPL","VEDL","VBL","VOLTAS","WIPRO","YESBANK","ZYDUSLIFE"
]

DATA_DIR = "data"
os.makedirs(DATA_DIR, exist_ok=True)

# ==================================================
# INIT KITE
# ==================================================
kite = KiteConnect(api_key=API_KEY)
kite.set_access_token(ACCESS_TOKEN)

# Load instruments once
instruments = pd.DataFrame(kite.instruments("NFO"))

# ==================================================
# FETCH OPTION CHAIN + STOCK LTP
# ==================================================
def fetch_option_chain(stock):

    # -------- STOCK LTP --------
    try:
        spot = kite.quote([f"NSE:{stock}"])
        stock_ltp = spot[f"NSE:{stock}"]["last_price"]
    except:
        stock_ltp = None

    df = instruments[
        (instruments["name"] == stock) &
        (instruments["segment"] == "NFO-OPT")
    ].copy()

    if df.empty:
        return None

    df["expiry"] = pd.to_datetime(df["expiry"])
    expiry = df["expiry"].min()
    df = df[df["expiry"] == expiry]

    symbols = ["NFO:" + ts for ts in df["tradingsymbol"]]
    quotes = kite.quote(symbols)

    rows = []
    for strike in sorted(df["strike"].unique()):
        ce = df[(df["strike"] == strike) & (df["instrument_type"] == "CE")]
        pe = df[(df["strike"] == strike) & (df["instrument_type"] == "PE")]

        ce_q = quotes.get("NFO:" + ce.iloc[0]["tradingsymbol"], {}) if not ce.empty else {}
        pe_q = quotes.get("NFO:" + pe.iloc[0]["tradingsymbol"], {}) if not pe.empty else {}

        rows.append({
            "Stock": stock,
            "Expiry": expiry.date(),
            "Strike": strike,
            "CE_LTP": ce_q.get("last_price"),
            "CE_OI": ce_q.get("oi"),
            "PE_LTP": pe_q.get("last_price"),
            "PE_OI": pe_q.get("oi"),
            "timestamp": datetime.now(IST).strftime("%Y-%m-%d %H:%M"),
            "Stock_LTP": stock_ltp
        })

    return pd.DataFrame(rows)

# ==================================================
# MAX PAIN CALCULATION
# ==================================================
def compute_max_pain(df):
    df = df.copy()

    for col in ["Strike","CE_LTP","CE_OI","PE_LTP","PE_OI"]:
        df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0)

    A = df["CE_LTP"]
    B = df["CE_OI"]
    G = df["Strike"]
    M = df["PE_LTP"]
    L = df["PE_OI"]

    mp = []
    for i in range(len(df)):
        value = (
            -sum(A[i:] * B[i:])
            + G.iloc[i] * sum(B[:i]) - sum(G[:i] * B[:i])
            - sum(M[:i] * L[:i])
            + sum(G[i:] * L[i:]) - G.iloc[i] * sum(L[i:])
        )
        mp.append(int(value / 10000))

    df["Max_Pain"] = mp
    return df

# ==================================================
# MAIN
# ==================================================
def main():
    all_data = []

    for stock in STOCKS:
        df = fetch_option_chain(stock)
        if df is None or df.empty:
            continue

        df = df.sort_values("Strike").reset_index(drop=True)
        df = compute_max_pain(df)

        all_data.append(df)

    if not all_data:
        print("No data fetched")
        return

    final_df = pd.concat(all_data, ignore_index=True)

    run_ts = datetime.now(IST).strftime("%Y-%m-%d_%H-%M")
    filename = f"{DATA_DIR}/option_chain_{run_ts}.csv"

    final_df.to_csv(filename, index=False)
    print(f"[OK] Saved {filename}")

# ==================================================
# ENTRY POINT
# ==================================================
if __name__ == "__main__":
    main()

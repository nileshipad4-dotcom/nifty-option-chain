from kiteconnect import KiteConnect
import pandas as pd
from datetime import datetime
import pytz
import os

# ==================================================
# TIMEZONE
# ==================================================
IST = pytz.timezone("Asia/Kolkata")

# ==================================================
# CONFIG
# ==================================================
API_KEY = "bkgv59vaazn56c42"
ACCESS_TOKEN = "cFyFyg4FYBRApOzvm2lc5wInr7MrKOPj"

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

# ==================================================
# DATA DIRECTORY BOOTSTRAP (NEW)
# ==================================================
DATA_DIR = "data"
os.makedirs(DATA_DIR, exist_ok=True)

PLACEHOLDER_FILE = os.path.join(DATA_DIR, "README.csv")
if not os.path.exists(PLACEHOLDER_FILE):
    pd.DataFrame(
        columns=[
            "Stock","Expiry","Strike",
            "CE_LTP","CE_OI","PE_LTP","PE_OI",
            "Stock_LTP","timestamp","Max_Pain"
        ]
    ).to_csv(PLACEHOLDER_FILE, index=False)

# ==================================================
# INIT KITE
# ==================================================
kite = KiteConnect(api_key=API_KEY)
kite.set_access_token(ACCESS_TOKEN)

# ==================================================
# LOAD INSTRUMENTS
# ==================================================
print("[INFO] Loading instruments")
instruments = pd.DataFrame(kite.instruments("NFO"))

# ==================================================
# HELPERS
# ==================================================
def chunk(lst, size=500):
    for i in range(0, len(lst), size):
        yield lst[i:i + size]

# ==================================================
# PREPARE OPTION SYMBOLS
# ==================================================
print("[INFO] Preparing option chain symbols")

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
    all_option_symbols.extend("NFO:" + df["tradingsymbol"])

# ==================================================
# BULK QUOTES
# ==================================================
print("[INFO] Fetching option quotes")
option_quotes = {}
for batch in chunk(all_option_symbols):
    option_quotes.update(kite.quote(batch))

print("[INFO] Fetching stock LTPs")
spot_quotes = kite.quote([f"NSE:{s}" for s in STOCKS])

# ==================================================
# MAX PAIN
# ==================================================
def compute_max_pain(df):
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
# PROCESS
# ==================================================
print("[INFO] Processing stocks")

all_data = []
now_ts = datetime.now(IST).strftime("%Y-%m-%d %H:%M")

for stock, df in option_map.items():
    rows = []
    stock_ltp = spot_quotes.get(f"NSE:{stock}", {}).get("last_price")

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
            "PE_LTP": pe_q.get("last_price"),
            "PE_OI": pe_q.get("oi"),
            "Stock_LTP": stock_ltp,
            "timestamp": now_ts,
        })

    stock_df = pd.DataFrame(rows).sort_values("Strike")
    stock_df = compute_max_pain(stock_df)
    all_data.append(stock_df)

# ==================================================
# SAVE
# ==================================================
final_df = pd.concat(all_data, ignore_index=True)
filename = f"{DATA_DIR}/option_chain_{datetime.now(IST).strftime('%Y-%m-%d_%H-%M')}.csv"
final_df.to_csv(filename, index=False)

print(f"[OK] Saved {filename}")

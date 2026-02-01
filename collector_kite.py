from kiteconnect import KiteConnect
import pandas as pd
from datetime import datetime
import pytz
import os


# ==================================================
# CONFIG
# ==================================================
from kite_config import KITE_API_KEY, KITE_ACCESS_TOKEN

API_KEY = KITE_API_KEY
ACCESS_TOKEN = KITE_ACCESS_TOKEN
# ==================================================
# TIMEZONE
# ==================================================
IST = pytz.timezone("Asia/Kolkata")



STOCKS = [
    "360ONE","ABB","ABCAPITAL","ADANIENSOL","ADANIENT","ADANIGREEN","ADANIPORTS","ALKEM",
    "AMBER","AMBUJACEM","ANGELONE","APLAPOLLO","ASHOKLEY","ASIANPAINT","ASTRAL","AUROPHARMA",
    "AUBANK","AXISBANK","BAJAJ-AUTO","BAJAJFINSV","BAJAJHLDNG","BAJFINANCE","BANDHANBNK",
    "BANKBARODA","BANKINDIA","BEL","BHEL","BHARATFORG","BHARTIARTL","BIOCON","BPCL",
    "BRITANNIA","BSE","CAMS","CANBK","CDSL","CGPOWER","CHOLAFIN","CIPLA","COALINDIA",
    "COFORGE","COLPAL","CONCOR","CROMPTON","CUMMINSIND","DABUR","DALBHARAT","DELHIVERY",
    "DIVISLAB","DLF","DMART","DRREDDY","EICHERMOT","EXIDEIND","FEDERALBNK","FORTIS",
    "GAIL","GLENMARK","GODREJCP","GODREJPROP","GRASIM","HAL","HAVELLS","HCLTECH",
    "HDFCAMC","HDFCBANK","HDFCLIFE","HEROMOTOCO","HINDALCO","HINDPETRO","HINDUNILVR",
    "HINDZINC","HUDCO","ICICIBANK","ICICIGI","ICICIPRULI","IDEA","IDFCFIRSTB","IEX",
    "INDHOTEL","INDIGO","INDIANB","INDUSINDBK","INDUSTOWER","INFY","INOXWIND","IOC",
    "IRCTC","IREDA","IRFC","ITC","JINDALSTEL","JIOFIN","JSWENERGY","JSWSTEEL",
    "JUBLFOOD","KALYANKJIL","KAYNES","KEI","KFINTECH","KOTAKBANK","KPITTECH","LAURUSLABS",
    "LICI","LICHSGFIN","LODHA","LT","LTF","LTIM","LUPIN","M&M","MANAPPURAM","MANKIND",
    "MARICO","MARUTI","MAXHEALTH","MCX","MFSL","MPHASIS","MOTHERSON","MUTHOOTFIN",
    "NAUKRI","NATIONALUM","NESTLEIND","NHPC","NMDC","NTPC","NYKAA","OBEROIRLTY","OFSS",
    "OIL","ONGC","PAGEIND","PATANJALI","PAYTM","PERSISTENT","PETRONET","PFC",
    "PGEL","PHOENIXLTD","PIDILITIND","PNB","PNBHOUSING","POLICYBZR","POLYCAB",
    "POWERGRID","POWERINDIA","PPLPHARMA","PREMIERENE","PRESTIGE","RBLBANK","RECLTD",
    "RELIANCE","RVNL","SAIL","SAMMAANCAP","SBICARD","SBILIFE","SBIN","SHREECEM",
    "SHRIRAMFIN","SIEMENS","SOLARINDS","SONACOMS","SRF","SUNPHARMA","SUZLON",
    "SYNGENE","TATACONSUM","TATAELXSI","TATAPOWER","TATASTEEL","TATATECH","TCS",
    "TECHM","TIINDIA","TITAN","TORNTPHARM","TORNTPOWER","TRENT","TVSMOTOR",
    "ULTRACEMCO","UNIONBANK","UNITDSPR","UNOMINDA","UPL","VBL","VEDL","VOLTAS",
    "WAAREEENER","WIPRO","YESBANK","ZYDUSLIFE"
]

# ==================================================
# DATA DIRECTORY
# ==================================================
DATA_DIR = "data"
os.makedirs(DATA_DIR, exist_ok=True)

# ==================================================
# INIT KITE
# ==================================================
kite = KiteConnect(api_key=API_KEY)
kite.set_access_token(ACCESS_TOKEN)

print("[INFO] Loading instruments")
instruments = pd.DataFrame(kite.instruments("NFO"))

# ==================================================
# HELPERS
# ==================================================
def chunk(lst, size=200):
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

    all_option_symbols.extend(
        ["NFO:" + ts for ts in df["tradingsymbol"].tolist()]
    )

# ==================================================
# BULK QUOTES
# ==================================================
print("[INFO] Fetching option quotes")
option_quotes = {}
for batch in chunk(all_option_symbols):
    option_quotes.update(kite.quote(batch))

print("[INFO] Fetching stock quotes")
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

    spot = spot_quotes.get(f"NSE:{stock}", {})
    stock_ltp = spot.get("last_price")
    prev_close = spot.get("ohlc", {}).get("close")
    stock_high = spot.get("ohlc", {}).get("high")
    stock_low = spot.get("ohlc", {}).get("low")

    pct_change = (
        round(((stock_ltp - prev_close) / prev_close) * 100, 3)
        if stock_ltp and prev_close else None
    )

    for strike in sorted(df["strike"].unique()):
        ce = df[(df["strike"] == strike) & (df["instrument_type"] == "CE")]
        pe = df[(df["strike"] == strike) & (df["instrument_type"] == "PE")]

        ce_q = option_quotes.get(
            "NFO:" + ce.iloc[0]["tradingsymbol"], {}
        ) if not ce.empty else {}

        pe_q = option_quotes.get(
            "NFO:" + pe.iloc[0]["tradingsymbol"], {}
        ) if not pe.empty else {}

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
            "Stock_High": stock_high,
            "Stock_Low": stock_low,
            "Stock_%_Change": pct_change,

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

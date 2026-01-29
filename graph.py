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

IST = pytz.timezone("Asia/Kolkata")

INDICES = {
    "NIFTY": "NIFTY",
    "BANKNIFTY": "BANKNIFTY",
    "MIDCPNIFTY": "MIDCPNIFTY"
}

DATA_DIR = "data_index"
os.makedirs(DATA_DIR, exist_ok=True)

# ==================================================
# INIT KITE
# ==================================================
kite = KiteConnect(api_key=API_KEY)
kite.set_access_token(ACCESS_TOKEN)

print("[INFO] Loading NFO instruments")
instruments = pd.DataFrame(kite.instruments("NFO"))

# ==================================================
# HELPERS
# ==================================================
def chunk(lst, size=200):
    for i in range(0, len(lst), size):
        yield lst[i:i + size]

# ==================================================
# MAX PAIN FORMULA (UNCHANGED)
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
# PREPARE OPTION SYMBOLS
# ==================================================
print("[INFO] Preparing index option symbols")

option_map = {}
all_symbols = []

for idx in INDICES:
    df = instruments[
        (instruments["name"] == idx) &
        (instruments["segment"] == "NFO-OPT")
    ].copy()

    df["expiry"] = pd.to_datetime(df["expiry"])
    expiry = df["expiry"].min()
    df = df[df["expiry"] == expiry]

    option_map[idx] = df

    all_symbols.extend(
        ["NFO:" + ts for ts in df["tradingsymbol"].tolist()]
    )

# ==================================================
# FETCH QUOTES
# ==================================================
print("[INFO] Fetching option quotes")
option_quotes = {}
for batch in chunk(all_symbols):
    option_quotes.update(kite.quote(batch))

print("[INFO] Fetching index spot prices")
spot_quotes = kite.quote([f"NSE:{i}" for i in INDICES])

# ==================================================
# PROCESS
# ==================================================
print("[INFO] Calculating Max Pain")

all_data = []
now_ts = datetime.now(IST).strftime("%Y-%m-%d %H:%M")

for idx, df in option_map.items():
    rows = []

    spot = spot_quotes.get(f"NSE:{idx}", {})
    idx_ltp = spot.get("last_price")

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
            "Index": idx,
            "Expiry": df["expiry"].iloc[0].date(),
            "Strike": strike,

            "CE_LTP": ce_q.get("last_price"),
            "CE_OI": ce_q.get("oi"),

            "PE_LTP": pe_q.get("last_price"),
            "PE_OI": pe_q.get("oi"),

            "Spot": idx_ltp,
            "timestamp": now_ts
        })

    idx_df = pd.DataFrame(rows).sort_values("Strike")
    idx_df = compute_max_pain(idx_df)

    # Extract live Max Pain strike
    mp_row = idx_df.loc[idx_df["Max_Pain"].idxmin()]
    print(f"[MP] {idx} | Strike: {int(mp_row['Strike'])} | Value: {int(mp_row['Max_Pain'])}")

    all_data.append(idx_df)

# ==================================================
# SAVE
# ==================================================
final_df = pd.concat(all_data, ignore_index=True)
filename = f"{DATA_DIR}/index_max_pain_{datetime.now(IST).strftime('%Y-%m-%d_%H-%M')}.csv"
final_df.to_csv(filename, index=False)

print(f"[OK] Saved {filename}")

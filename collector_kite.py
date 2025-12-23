from kiteconnect import KiteConnect
import pandas as pd
from datetime import datetime
import os

# ================= CONFIG =================
API_KEY = "bkgv59vaazn56c42"
ACCESS_TOKEN = "giMaA7SuUyWA1r9P34zVTOVlPcWG847C"

STOCKS = ["RELIANCE", "HDFCBANK", "ICICIBANK"]

DATA_DIR = "data"
os.makedirs(DATA_DIR, exist_ok=True)

# ================= INIT =================
kite = KiteConnect(api_key=API_KEY)
kite.set_access_token(ACCESS_TOKEN)

instruments = pd.DataFrame(kite.instruments("NFO"))

# ================= FETCH OPTION CHAIN =================
def fetch_option_chain(stock):
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
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M")
        })

    return pd.DataFrame(rows)

# ================= MAIN =================
def main():
    for stock in STOCKS:
        df = fetch_option_chain(stock)
        if df is None:
            continue

        out = f"{DATA_DIR}/{stock.lower()}.csv"
        df.to_csv(out, mode="a", header=not os.path.exists(out), index=False)
        print(f"[OK] {stock} saved")

if __name__ == "__main__":
    main()

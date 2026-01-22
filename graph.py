def compute_atm_sum(file_path):
    df = pd.read_csv(file_path)

    # -------------------------------
    # COLUMN NORMALIZATION
    # -------------------------------

    # Stock column
    if "Stock" not in df.columns and "Symbol" in df.columns:
        df["Stock"] = df["Symbol"]

    # Stock LTP column
    if "Stock_LTP" not in df.columns:
        if "Spot" in df.columns:
            df["Stock_LTP"] = df["Spot"]
        else:
            df["Stock_LTP"] = 0   # fallback

    # Required numeric columns
    for col in ["Strike", "CE_OI", "PE_OI"]:
        if col not in df.columns:
            df[col] = 0

    # Convert to numeric safely
    for c in ["Strike", "Stock_LTP", "CE_OI", "PE_OI"]:
        df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0)

    # -------------------------------
    # WEIGHTED OI
    # -------------------------------
    df["ce_x"] = (df["CE_OI"] * df["Strike"]) / 10000
    df["pe_x"] = (df["PE_OI"] * df["Strike"]) / 10000

    df["diff"] = np.nan
    df["atm_diff"] = np.nan

    # -------------------------------
    # ATM DIFF LOGIC
    # -------------------------------
    for stk, g in df.groupby("Stock"):
        g = g.sort_values("Strike").reset_index()

        for i in range(len(g)):
            low = max(0, i - STRIKE_WINDOW)
            high = min(len(g) - 1, i + STRIKE_WINDOW)

            diff_val = g.loc[low:high, "pe_x"].sum() - g.loc[low:high, "ce_x"].sum()
            df.at[g.loc[i, "index"], "diff"] = diff_val

        ltp = g["Stock_LTP"].iloc[0]
        atm_idx = (g["Strike"] - ltp).abs().values.argmin()

        low = max(0, atm_idx - ATM_WINDOW)
        high = min(len(g) - 1, atm_idx + ATM_WINDOW)

        atm_avg = df.loc[g.loc[low:high, "index"], "diff"].mean()
        df.loc[g["index"], "atm_diff"] = atm_avg

    return df["atm_diff"].sum() / 1000

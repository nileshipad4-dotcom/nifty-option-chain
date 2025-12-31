def highlight_rows(df):
    styles = pd.DataFrame("", index=df.index, columns=df.columns)

    for stock in df["Stock"].dropna().unique():
        sdf = df[(df["Stock"] == stock) & (df["Strike"].notna())]
        if sdf.empty:
            continue

        # FIX: safely convert back to float
        ltp = pd.to_numeric(sdf["Live_Stock_LTP"].iloc[0], errors="coerce")
        strikes = sdf["Strike"].values

        if pd.notna(ltp):
            for i in range(len(strikes) - 1):
                if strikes[i] <= ltp <= strikes[i + 1]:
                    styles.loc[sdf.index[i], :] = "background-color:#003366;color:white"
                    styles.loc[sdf.index[i + 1], :] = "background-color:#003366;color:white"
                    break

        mp_vals = sdf["Live_Max_Pain"].dropna()
        if not mp_vals.empty:
            styles.loc[mp_vals.idxmin(), :] = "background-color:#8B0000;color:white"

    return styles

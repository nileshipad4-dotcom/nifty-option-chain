import streamlit as st
import pandas as pd
import numpy as np
import os
from datetime import time, datetime
from streamlit_autorefresh import st_autorefresh
from pathlib import Path

# ==================================================
# ⚙️ CONFIGURATION & SETUP
# ==================================================
st.set_page_config(
    page_title="OI Weighted Table",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Constants
DATA_DIR = Path("data")
REFRESH_INTERVAL = 3600_000  # 1 Hour

# Auto Refresh
st_autorefresh(interval=REFRESH_INTERVAL, key="auto_refresh")

# Custom CSS for compact tables and better visuals
st.markdown("""
    <style>
        .stDataFrame { font-size: 12px; }
        div[data-testid="stMetricValue"] { font-size: 18px; }
        /* Reduce padding for a denser layout */
        .block-container { padding-top: 2rem; padding-bottom: 2rem; }
    </style>
""", unsafe_allow_html=True)

# ==================================================
# 📂 DATA LOADING ENGINE
# ==================================================
def get_available_files():
    """Scans data directory for valid option chain CSVs between 08:00 and 16:00."""
    if not DATA_DIR.exists():
        DATA_DIR.mkdir()
        st.error(f"Directory '{DATA_DIR}' created. Please add CSV files.")
        st.stop()
    
    files = []
    # Scan for files matching pattern
    for f in DATA_DIR.glob("option_chain_*.csv"):
        try:
            ts_str = f.name.replace("option_chain_", "").replace(".csv", "")
            # Extract time part for filtering (Format expected: YYYY-MM-DD_HH-MM-SS or similar ending in HH-MM)
            # Adapting to original script logic: split("_")[-1] -> HH-MM
            time_part = ts_str.split("_")[-1]
            hh, mm = map(int, time_part.split("-")[:2])
            t_obj = time(hh, mm)
            
            # Filter: 08:00 to 16:00
            if time(8, 0) <= t_obj <= time(16, 0):
                files.append((ts_str, f, t_obj))
        except Exception:
            continue
            
    # Sort descending (newest first)
    return sorted(files, key=lambda x: x[0], reverse=True)

@st.cache_data(ttl=300)
def load_raw_data(file_path):
    """Loads a single CSV safely."""
    try:
        return pd.read_csv(file_path)
    except Exception as e:
        return pd.DataFrame() # Return empty on error

# ==================================================
# 🧮 CORE DATA PROCESSING
# ==================================================
def clean_numeric_cols(df, cols):
    """Coerces specified columns to numeric, filling NaNs with 0."""
    for c in cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0)
    return df

@st.cache_data(ttl=60)
def process_market_data(t1_path, t2_path, t3_path, t0a_path, t0b_path, window_x):
    """
    Main logic pipeline: Merges files, calculates OI delta, weights, and sliding windows.
    """
    # 1. Load DataFrames
    df1 = load_raw_data(t1_path)
    df2 = load_raw_data(t2_path)
    df3 = load_raw_data(t3_path)
    df0a = load_raw_data(t0a_path)
    df0b = load_raw_data(t0b_path)

    # Validate essential columns
    req_cols = ["Stock", "Strike", "Stock_LTP", "Stock_%_Change", "CE_OI", "PE_OI"]
    if any(d.empty or not all(c in d.columns for c in req_cols) for d in [df1, df2, df3]):
        return pd.DataFrame(), pd.DataFrame()

    # 2. Rename and Merge Main Frames (Current, Previous, Oldest)
    dfs = []
    for i, d in enumerate([df1, df2, df3]):
        tmp = d[req_cols].rename(columns={
            "Stock_LTP": f"ltp_{i}",
            "Stock_%_Change": f"tot_ch_{i}",
            "CE_OI": f"ce_{i}",
            "PE_OI": f"pe_{i}"
        })
        dfs.append(tmp)

    # Inner merge ensures we only keep strikes present in all 3 timeframes
    df = dfs[0].merge(dfs[1], on=["Stock", "Strike"]).merge(dfs[2], on=["Stock", "Strike"])
    
    # 3. Numeric Safety
    num_cols = [c for c in df.columns if any(x in c for x in ["ltp", "ce", "pe", "Strike", "tot_ch"])]
    df = clean_numeric_cols(df, num_cols)

    # 4. Early Window Calculation (09:10+) logic
    # We need to calculate the change between t0a and t0b and merge it into the main df
    e_cols = ["Stock", "Strike", "CE_OI", "PE_OI"]
    if not df0a.empty and not df0b.empty:
        early = df0a[e_cols].merge(df0b[e_cols], on=["Stock", "Strike"], suffixes=("_a", "_b"))
        early = clean_numeric_cols(early, ["CE_OI_a", "CE_OI_b", "PE_OI_a", "PE_OI_b", "Strike"])
        
        early["d_ce_0"] = early["CE_OI_b"] - early["CE_OI_a"]
        early["d_pe_0"] = early["PE_OI_b"] - early["PE_OI_a"]
        
        # Weighted Early Change
        early["ce_x_0"] = (early["d_ce_0"] * early["Strike"]) / 10000
        early["pe_x_0"] = (early["d_pe_0"] * early["Strike"]) / 10000

        # Merge specific columns back to main df
        df = df.merge(early[["Stock", "Strike", "ce_x_0", "pe_x_0"]], on=["Stock", "Strike"], how="left").fillna(0)
    else:
        df["ce_x_0"] = 0
        df["pe_x_0"] = 0

    # 5. Core Delta Calculations
    df["d_ce"] = df["ce_0"] - df["ce_1"]
    df["d_pe"] = df["pe_0"] - df["pe_1"]
    df["d_ce_23"] = df["ce_1"] - df["ce_2"]
    df["d_pe_23"] = df["pe_1"] - df["pe_2"]
    df["total_ch"] = df["tot_ch_0"]
    
    # Weighted Calculations
    df["ce_x"] = (df["d_ce"] * df["Strike"]) / 10000
    df["pe_x"] = (df["d_pe"] * df["Strike"]) / 10000
    df["ce_x_23"] = (df["d_ce_23"] * df["Strike"]) / 10000
    df["pe_x_23"] = (df["d_pe_23"] * df["Strike"]) / 10000
    df["diff_23"] = df["pe_x_23"] - df["ce_x_23"]
    df["ch"] = ((df["ltp_0"] - df["ltp_1"]) / df["ltp_1"]) * 100

    # 6. VECTORIZED Sliding Window Logic
    # Sort for rolling operations
    df = df.sort_values(["Stock", "Strike"])
    
    # Define window size: X up, X down + current = 2X + 1
    w_size = (window_x * 2) + 1
    
    # Use transform with rolling sum
    df["sum_ce"] = df.groupby("Stock")["ce_x"].transform(
        lambda x: x.rolling(window=w_size, center=True, min_periods=1).sum()
    )
    df["sum_pe"] = df.groupby("Stock")["pe_x"].transform(
        lambda x: x.rolling(window=w_size, center=True, min_periods=1).sum()
    )
    
    df["diff"] = df["sum_pe"] - df["sum_ce"]

    # 7. ATM Identification & Logic
    # Find index where abs(Strike - LTP) is min per stock
    df["dist"] = (df["Strike"] - df["ltp_0"]).abs()
    
    # Calculate ATM Diff (average diff around ATM)
    def calculate_atm_stats(g):
        if g.empty: return g
        atm_idx = g["dist"].idxmin()
        # Original logic: ATM +/- 2 strikes (Indices: atm-2 to atm+2)
        # Using index slicing based on position
        loc_idx = g.index.get_loc(atm_idx)
        start = max(0, loc_idx - 2)
        end = min(len(g), loc_idx + 3) # Slice is exclusive at end
        
        # We need to map integer pos back to DataFrame index
        # To simplify, we calculate on the group values
        subset_diff = g["diff"].iloc[start:end]
        g["atm_diff"] = subset_diff.mean()
        return g

    # Apply ATM calcs
    # Reset index to make sure loc works cleanly in apply
    df = df.reset_index(drop=True)
    df = df.groupby("Stock", group_keys=False).apply(calculate_atm_stats)
    df["atm_diff"] = df["atm_diff"].fillna(0)
    
    # Rename for display consistency with original script
    final_df = df.rename(columns={"Stock": "stk", "Strike": "str", "ltp_0": "ltp"})
    
    return final_df, df  # Return final display and raw full df

# ==================================================
# 🎨 STYLING HELPERS
# ==================================================
def get_styler(df, bottom_pe_stocks, bottom_ce_stocks, common_stocks=None):
    """Applies all visual styles to the dataframe."""
    
    def highlight_cells(data):
        styles = pd.DataFrame("", index=data.index, columns=data.columns)
        
        # 1. ATM Highlighting (Blue)
        if "stk" in data.columns:
            for stk, g in data.groupby("stk"):
                g = g.sort_values("str")
                if g.empty: continue
                ltp = g["ltp"].iloc[0]
                strikes = g["str"].values
                # Find ATM range logic
                for i in range(len(strikes)-1):
                    if strikes[i] <= ltp <= strikes[i+1]:
                        # Highlight the two strikes surrounding LTP
                        idx1, idx2 = g.index[i], g.index[i+1]
                        styles.loc[idx1, :] = "background-color:#003366;color:white"
                        styles.loc[idx2, :] = "background-color:#003366;color:white"
                        break
                        
        # 2. Stock Name Highlighting (Logic from original script)
        for i, row in data.iterrows():
            stk = row.get("stk")
            if stk:
                if stk in bottom_pe_stocks and stk in bottom_ce_stocks:
                    styles.at[i, "stk"] += ";background-color:#ff8c00;color:black" # Orange
                elif stk in bottom_pe_stocks:
                    styles.at[i, "stk"] += ";background-color:#1b5e20;color:white" # Green
                elif stk in bottom_ce_stocks:
                    styles.at[i, "stk"] += ";background-color:#8b0000;color:white" # Red
                
                # 3. Common Stocks in Ranking (Blue)
                if common_stocks and stk in common_stocks:
                    styles.loc[i, :] += ";background-color:#1f4e79;color:white"

        # 4. Early Columns Red Text
        for col in ["ce_x_0", "pe_x_0"]:
            if col in data.columns:
                styles[col] += ";color:#ff4b4b;font-weight:bold"
                
        return styles

    # Format specifiers
    fmt = {
        "str": "{:.2f}", "ltp": "{:.2f}", "ch": "{:.2f}", "total_ch": "{:.2f}",
        "d_ce": "{:.0f}", "d_pe": "{:.0f}", "ce_x": "{:.0f}", "pe_x": "{:.0f}",
        "sum_ce": "{:.0f}", "sum_pe": "{:.0f}", "diff": "{:.0f}", 
        "atm_diff": "{:.0f}", "diff_23": "{:.0f}", "ce_x_0": "{:.0f}", "pe_x_0": "{:.0f}"
    }
    
    return df.style.apply(highlight_cells, axis=None).format(fmt, na_rep="")

# ==================================================
# 📈 TREND ANALYSIS ENGINE
# ==================================================
def analyze_trends(df):
    """
    Identifies stocks in UP or DOWN trends based on strict strike logic.
    Refactored for safety and speed.
    """
    up_data = []
    down_data = []
    
    for stk, g in df.groupby("stk"):
        g = g.sort_values("str").reset_index(drop=True)
        if g.empty: continue
        
        ltp = g["ltp"].iloc[0]
        # Find ATM index
        atm_idx = (g["str"] - ltp).abs().idxmin()
        N = len(g)

        # Helper to get slice safely (handles out of bounds)
        def get_slice(start_offset, end_offset_exclusive):
            s = max(0, atm_idx + start_offset)
            e = min(N, atm_idx + end_offset_exclusive)
            # Check if valid slice (indices must be within conceptual range)
            # Original logic: if i < len(g). 
            return g.iloc[s:e] if s < e else pd.DataFrame(columns=g.columns)

        # --- UP TREND LOGIC ---
        # Original: pe_idxs = [atm, atm+4), ce_idxs = [atm-4, atm+3), pe_pos = [atm, atm+2)
        pe_win = get_slice(0, 4)      
        ce_win = get_slice(-4, 3)     
        pe_pos = get_slice(0, 2)      
        
        # Conditions
        # 1. Any PE in window > 900
        # 2. No CE in window > 4000
        # 3. All PE in pos window > 0
        if not pe_win.empty and not ce_win.empty and not pe_pos.empty:
            cond_pe_big = (pe_win["pe_x"] > 900).any()
            cond_ce_small = not (ce_win["ce_x"] > 4000).any()
            cond_pe_positive = (pe_pos["pe_x"] > 0).all()

            if cond_pe_big and cond_ce_small and cond_pe_positive:
                score = pe_win["pe_x"].max() - ce_win["ce_x"].max()
                up_data.append({"stk": stk, "score": score, "block": g})

        # --- DOWN TREND LOGIC ---
        # Original: ce_strong = [atm-2, atm+2), pe_weak = [atm-3, atm+4), ce_pos = [atm, atm+2)
        ce_str = get_slice(-2, 2)
        pe_weak = get_slice(-3, 4)
        ce_pos = get_slice(0, 2)
        
        if not ce_str.empty and not pe_weak.empty and not ce_pos.empty:
            cond_ce_big = (ce_str["ce_x"] > 900).any()
            cond_pe_small = not (pe_weak["pe_x"] > 4000).any()
            cond_ce_positive = (ce_pos["ce_x"] > 0).all()

            if cond_ce_big and cond_pe_small and cond_ce_positive:
                score = ce_str["ce_x"].max() - pe_weak["pe_x"].max()
                down_data.append({"stk": stk, "score": score, "block": g})

    # Sort blocks by score and concatenate
    def build_trend_df(data_list):
        if not data_list: return pd.DataFrame()
        # Sort by score descending
        sorted_list = sorted(data_list, key=lambda x: x["score"], reverse=True)
        # Extract blocks
        return pd.concat([item["block"] for item in sorted_list], ignore_index=True)

    return build_trend_df(up_data), build_trend_df(down_data)

# ==================================================
# 🖥️ MAIN APPLICATION
# ==================================================
def main():
    st.title("📊 OI Weighted Strike Analysis")
    
    # 1. File Selection
    files_info = get_available_files()
    
    # Original logic constraint
    if len(files_info) < 3:
        st.error("Need at least 3 valid CSV files (08:00-16:00) in 'data' folder.")
        st.stop()
        
    timestamps = [f[0] for f in files_info]
    file_map = {f[0]: f[1] for f in files_info}
    time_objs = {f[0]: f[2] for f in files_info}

    # Identify Early Window (First 2 available files AFTER 09:10)
    # Sort files by time ascending to find the earliest ones
    sorted_by_time = sorted(files_info, key=lambda x: x[2])
    early_candidates = [item for item in sorted_by_time if item[2] >= time(9, 10)]
    
    if len(early_candidates) < 2:
        st.error("Need at least 2 CSV files after 09:10 for Early Window calculation.")
        st.stop()
    
    t0a_info = early_candidates[0]
    t0b_info = early_candidates[1]
    
    # 2. Sidebar / Top Controls
    with st.expander("⚙️ Settings & Configuration", expanded=True):
        col1, col2, col3, col4, col5 = st.columns(5)
        
        # Default indices to prevent out of range errors
        idx1 = 0
        idx2 = 1 if len(timestamps) > 1 else 0
        idx3 = 2 if len(timestamps) > 2 else 0
        
        t1 = col1.selectbox("TS1 (Current)", timestamps, index=idx1)
        t2 = col2.selectbox("TS2 (Previous)", timestamps, index=idx2)
        t3 = col3.selectbox("TS3 (Oldest)", timestamps, index=idx3)
        window_x = col4.number_input("Strike Window X", 1, 10, 4)
        
        # Display Early Window Times
        col5.info(f"🕘 Early Window:\n{t0a_info[2].strftime('%H:%M')} vs {t0b_info[2].strftime('%H:%M')}")

    # 3. Process Data
    # Note: we pass file paths, not dataframes, to allow caching to work on paths
    full_table, _ = process_market_data(
        file_map[t1], file_map[t2], file_map[t3], 
        t0a_info[1], t0b_info[1],  # Path to early files
        window_x
    )

    if full_table.empty:
        st.error("Data processing failed. Check if CSV files have required columns.")
        st.stop()

    # 4. View Filtering (Near LTP)
    def filter_near_ltp(df, n=5):
        blocks = []
        for stk, g in df.groupby("stk"):
            g = g.sort_values("str").reset_index(drop=True)
            if g.empty: continue
            ltp = g["ltp"].iloc[0]
            atm_idx = (g["str"] - ltp).abs().idxmin()
            s = max(0, atm_idx - n)
            e = min(len(g) - 1, atm_idx + n)
            blocks.append(g.iloc[s:e+1])
        return pd.concat(blocks, ignore_index=True) if blocks else pd.DataFrame()

    display_df = filter_near_ltp(full_table, n=5)

    # 5. Bottom Stocks Identification (For Coloring)
    # Summing pe_x and ce_x per stock to find top/bottom
    agg = display_df.groupby("stk")[["pe_x", "ce_x"]].sum().reset_index()
    
    # Sorting Controls
    st.markdown("### 🔃 Sorting & Highlights")
    c_sort1, c_sort2, c_sort3 = st.columns([1,1,2])
    
    if "pe_min" not in st.session_state: st.session_state.pe_min = False
    if "ce_min" not in st.session_state: st.session_state.ce_min = False
    
    def on_pe(): 
        if st.session_state.pe_min: st.session_state.ce_min = False
    def on_ce():
        if st.session_state.ce_min: st.session_state.pe_min = False
        
    pe_min = c_sort1.toggle("Sort PE Min", key="pe_min", on_change=on_pe)
    ce_min = c_sort2.toggle("Sort CE Min", key="ce_min", on_change=on_ce)
    bottom_n = c_sort3.number_input("Highlight Bottom N", 5, 100, 20, 5)

    bottom_pe = set(agg.sort_values("pe_x", ascending=True).head(bottom_n)["stk"])
    bottom_ce = set(agg.sort_values("ce_x", ascending=True).head(bottom_n)["stk"])

    # Apply Sorting to the Display DataFrame
    if pe_min:
        order = agg.sort_values("pe_x", ascending=True)["stk"].tolist()
        display_df["stk"] = pd.Categorical(display_df["stk"], categories=order, ordered=True)
        display_df = display_df.sort_values(["stk", "str"])
    elif ce_min:
        order = agg.sort_values("ce_x", ascending=True)["stk"].tolist()
        display_df["stk"] = pd.Categorical(display_df["stk"], categories=order, ordered=True)
        display_df = display_df.sort_values(["stk", "str"])

    # 6. TABS LAYOUT
    tab_main, tab_rank, tab_trend, tab_compare = st.tabs([
        "📊 Main Table", "📈 Strike Rankings", "🚀 Trends", "🔍 Comparison"
    ])

    # --- TAB 1: Main Table ---
    with tab_main:
        up_count = display_df[display_df["atm_diff"] > 0]["stk"].nunique()
        total_atm = display_df["atm_diff"].sum() / 1000
        st.markdown(f"#### 🟢 Stocks UP: {up_count} | Σ ATM Diff: {total_atm:.0f}k")
        
        st.dataframe(
            get_styler(display_df, bottom_pe, bottom_ce),
            use_container_width=True,
            height=800
        )

    # --- TAB 2: Rankings (Diff Analysis) ---
    with tab_rank:
        st.subheader("📈 Stock Presence in Top Diff Strikes")
        cR1, cR2, cR3 = st.columns(3)
        top_n_str = cR1.number_input("Top N Strikes", 10, 500, 150, 10)
        min_cnt = cR2.number_input("Min Count", 1, 20, 6, 1)
        top_first = cR3.toggle("Top First (Highest Diff)", value=False)
        
        # Helper to generate summary
        def get_rank_summary(col_name):
            # Sort Full Table based on diff criteria
            sorted_t = full_table.sort_values(col_name, ascending=not top_first)
            
            # Inner join with display_df to ensure we only count visible strikes (filtered near LTP)
            # Original logic: merged sorted_df with display_df
            merged = sorted_t.merge(display_df[["stk", "str"]], on=["stk", "str"], how="inner").head(top_n_str)
            
            summ = merged.groupby("stk").agg(
                count=(col_name, "size"),
                total_ch=("total_ch", "first"),
                ch=("ch", "first")
            ).reset_index()
            
            return summ[summ["count"] >= min_cnt].sort_values("count", ascending=False)

        summ_diff = get_rank_summary("diff")
        summ_diff23 = get_rank_summary("diff_23")
        
        # Identify common stocks for highlighting
        common_stocks = set(summ_diff["stk"]) & set(summ_diff23["stk"])

        c_res1, c_res2 = st.columns(2)
        with c_res1:
            st.markdown("### 📊 Based on `diff`")
            st.dataframe(
                summ_diff.style.apply(
                    lambda x: ["background-color:#1f4e79;color:white" if x["stk"] in common_stocks else "" for _ in x], 
                    axis=1
                ).format({"total_ch":"{:.2f}", "ch":"{:.2f}", "count":"{:.0f}"}),
                use_container_width=True
            )
        
        with c_res2:
            st.markdown("### 📊 Based on `diff_23`")
            st.dataframe(
                summ_diff23.style.format({"count":"{:.0f}"}), 
                use_container_width=True
            )
            
        st.markdown("---")
        st.markdown("### 📋 Detailed View (Stocks from 'diff' list)")
        if not summ_diff.empty:
            det_df = display_df[display_df["stk"].isin(summ_diff["stk"])]
            st.dataframe(get_styler(det_df, bottom_pe, bottom_ce), use_container_width=True)

    # --- TAB 3: Up/Down Trends ---
    with tab_trend:
        up_df, down_df = analyze_trends(display_df)
        
        st.subheader("📈 UP TREND")
        if not up_df.empty:
            st.dataframe(get_styler(up_df, bottom_pe, bottom_ce), use_container_width=True)
        else:
            st.info("No UP TREND stocks found.")
            
        st.markdown("---")
        
        st.subheader("🔻 DOWN TREND")
        if not down_df.empty:
            st.dataframe(get_styler(down_df, bottom_pe, bottom_ce), use_container_width=True)
        else:
            st.info("No DOWN TREND stocks found.")

    # --- TAB 4: Comparison ---
    with tab_compare:
        st.subheader("🔍 Stock Detail View")
        all_stocks = [""] + sorted(display_df["stk"].unique().tolist())
        cc1, cc2 = st.columns(2)
        s_a = cc1.selectbox("Select Stock A", all_stocks)
        s_b = cc2.selectbox("Select Stock B", all_stocks)
        
        c_disp1, c_disp2 = st.columns(2)
        
        if s_a:
            with c_disp1:
                st.dataframe(get_styler(display_df[display_df["stk"] == s_a], bottom_pe, bottom_ce), use_container_width=True)
        if s_b:
            with c_disp2:
                st.dataframe(get_styler(display_df[display_df["stk"] == s_b], bottom_pe, bottom_ce), use_container_width=True)

if __name__ == "__main__":
    main()

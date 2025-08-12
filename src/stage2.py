from pathlib import Path
import os
import pandas as pd

SUMMARY_KEYWORDS = {"Total", "Average", "Min", "Max"}

COL_RENAME_MAP = {
    'Timestamp': 'Timestamp',
    'Active Gross Power_MW': 'Generation, Active Gross Power (MW)',
    'Reactive Gross Power_MVAR': 'Generation, Reactive Gross Power (MVAR)',
    'Exhaust Pressure Turbine CW_Bara': 'OEC, Exhaust Pressure Turbine CW (Bara)',
    'Exhaust Pressure Turbine CCW_Bara': 'OEC, Exhaust Pressure Turbine CCW (Bara)',
    'oC': 'Ambient Temp (C)',
    'kg/MW': 'Spesific Steam - Brine, Consumption (kg/MW)',
    'Main Steam Press_barg': 'Steam, Main Steam Press (barg)',
    'Main Steam Temp_oC': 'Steam, Main Steam Temp (C)',
    'Brine Header Press_barg': 'Brine, Brine Header Press (barg)',
    'Brine Header Temp_oC': 'Brine, Brine Header Temp (C)',
    'WHP_barg': 'IJN-6-1, WHP (barg)',
    'Valve Open_%': 'IJN-6-1, Valve Open (%)',
    '2 Phase line press_barg': 'IJN-6-1, 2 Phase Line Press (barg)',
    '2 Phase line temp_oC': 'IJN-6-1, 2 Phase Line Temp (C)',
    'WHP_barg.1': 'IJN-6-2, WHP (barg)',
    'Valve Open_%.1': 'IJN-6-2, Valve Open (%)',
    '2 Phase line press_barg.1': 'IJN-6-2, 2 Phase Line Press (barg)',
    '2 Phase line temp_oC.1': 'IJN-6-2, 2 Phase Line Temp (C)',
    'WHP_barg.2': 'IJN-6-3, WHP (barg)',
    'Valve Open_%.2': 'IJN-6-3, Valve Open (%)',
    '2 Phase line press_barg.2': 'IJN-6-3, 2 Phase Line Press (barg)',
    '2 Phase line temp_oC.2': 'IJN-6-3, 2 Phase Line Temp (C)',
    'WHP_barg.3': 'IJN-6-4, WHP (barg)',
    'Valve Open_%.3': 'IJN-6-4, Valve Open (%)',
    '2 Phase line press_barg.3': 'IJN-6-4, 2 Phase Line Press (barg)',
    '2 Phase line temp_oC.3': 'IJN-6-4, 2 Phase Line Temp (C)',
    'Separator Temp_oC': 'Pad 6, Separator Temp (C)',
    'Separator Pressure_barg': 'Pad 6, Separator Pressure (barg)',
    'Steam Flow_kg/s': 'Pad 6, Steam Flow (kg/s)',
    'WHP_barg.4': 'IJN-8-1, WHP (barg)',
    'Valve Open_%.4': 'IJN-8-1, Valve Open (%)',
    '2 Phase line press_barg.4': 'IJN-8-1, 2 Phase Line Press (barg)',
    '2 Phase line temp_oC.4': 'IJN-8-1, 2 Phase Line Temp (C)',
    'WHP_barg.5': 'IJN-8-2, WHP (barg)',
    'Valve Open_%.5': 'IJN-8-2, Valve Open (%)',
    '2 Phase line press_barg.5': 'IJN-8-2, 2 Phase Line Press (barg)',
    '2 Phase line temp_oC.5': 'IJN-8-2, 2 Phase Line Temp (C)',
    'Separator Temp_oC.1': 'Pad 8, Separator Temp (C)',
    'Separator Pressure_barg.1': 'Pad 8, Separator Pressure (barg)',
    'Steam Flow_kg/s.1': 'Pad 8, Separator Temp (kg/s)',
    'Main Steam Flowrate_kg/s': 'Pad 6 & Pad 8, Main Steam Flowrate (kg/s)',
    'Brine Flowrate_kg/s': 'Pad 6 & Pad 8, Brine Flowrate (kg/s)',
    'Total Flowrate_kg/s': 'Pad 6 & Pad 8, Total Flowrate (kg/s)',
    'Injection Press_bara': 'IJN-5-1, Injection Press (Bara)',
    'Injection Flow_kg/s': 'IJN-5-1, Injection Flow (kg/s)',
    'Injection Temp_oC': 'IJN-5-1, Injection Temp (C)',
    'Valve open_%': 'IJN-5-1, Valve Open (%)',
    'WHP_barg.6': 'IJN-5-1, WHP (barg)',
    'Brine Header Pressure_barg': 'DUPE, IGNORE (1)',
    'Brine Header Temperature_oC': 'DUPE, IGNORE(2)',
    'Header Reinjection Pump Temperature_oC': 'Header Reinjection Pump, Temperature (C)',
    'Header Reinjection Pump Pressure_Bara': 'Header Reinjection Pump, Pressure (Bara)',
    'Opening \nCV-A_%': 'Brine Bypass Pre-heater, Opening CV-A (%)',
    'Opening \nCV-B_%': 'Brine Bypass Pre-heater, Opening CV-B (%)',
    'Outlet Temp_oC': 'Preheater, Outlet Temp (C)',
    'Opening Brine Dump Valve A_%': 'Separator IJN-6, Opening Brine Dump Valve A (%)',
    'Opening Brine Dump Valve B_%': 'Separator IJN-6, Opening Brine Dump Valve B (%)',
    'Opening Brine Dump Valve A_%.1': 'Separator IJN-8, Opening Brine Dump Valve A (%)',
    'Opening Brine Dump Valve B_%.1': 'Separator IJN-8, Opening Brine Dump Valve B (%)',
    'Opening Steam Dump Valve A_%': 'Rock Muffler, Opening Steam Dump Valve A (%)',
    'Opening Steam Dump Valve B_%': 'Rock Muffler, Opening Steam Dump Valve B (%)',
    'SourceFile': 'Source DGR'
}

def _drop_summary_rows(df: pd.DataFrame) -> pd.DataFrame:
    return df[~df.iloc[:, 0].astype(str).str.strip().isin(SUMMARY_KEYWORDS)]

def _drop_duplicate_brine_cols(df: pd.DataFrame) -> pd.DataFrame:
    brine_cols = [c for c in df.columns if isinstance(c, tuple) and c and c[0] == "Brine"]
    if len(brine_cols) <= 1:
        return df
    to_drop = set()
    for i in range(len(brine_cols)):
        for j in range(i + 1, len(brine_cols)):
            if j in to_drop:
                continue
            if df[brine_cols[i]].equals(df[brine_cols[j]]):
                to_drop.add(j)
    for j in sorted(to_drop, reverse=True):
        df = df.drop(columns=[brine_cols[j]])
    return df

def _flatten_columns(df: pd.DataFrame) -> pd.DataFrame:
    df.columns = [
        "_".join([str(x).strip() for x in col if "Unnamed" not in str(x)]).strip()
        if isinstance(col, tuple) else str(col)
        for col in df.columns
    ]
    return df

def build_master_dataset(csv_folder: Path, timestamp_format: str = "%d-%m-%Y %H%MH") -> pd.DataFrame:
    """
    Replicates your Stage 2. `timestamp_format` defaults to your literal '%H%MH'.
    Change to '%d-%m-%Y %H:%M' if you prefer standard hh:mm.
    """
    csv_folder = Path(csv_folder)
    csv_files = sorted(csv_folder.glob("*.csv"))

    steamfield_dfs = []
    for file in csv_files:
        try:
            df = pd.read_csv(file, header=[0, 1])
            df = _drop_summary_rows(df)
            df = _drop_duplicate_brine_cols(df)
            df = _flatten_columns(df)
            df["SourceFile"] = file.name
            steamfield_dfs.append(df)
        except Exception as e:
            print(f"Failed to process {file}: {e}")

    if not steamfield_dfs:
        return pd.DataFrame()

    master_df = pd.concat(steamfield_dfs, ignore_index=True)

    # first column → Timestamp
    timestamp_col = master_df.columns[0]
    master_df = master_df.rename(columns={timestamp_col: "Timestamp"})
    master_df["Timestamp"] = pd.to_datetime(master_df["Timestamp"], errors="coerce")
    master_df = master_df.dropna(subset=["Timestamp"]).sort_values("Timestamp").reset_index(drop=True)

    # display formatting (keeps underlying dtype as datetime if you keep a parallel column)
    master_df["Timestamp"] = master_df["Timestamp"].dt.strftime(timestamp_format)

    # numeric conversion & rounding
    for col in master_df.columns:
        if col not in ["Timestamp", "SourceFile"]:
            master_df[col] = pd.to_numeric(master_df[col], errors="coerce")
    numeric_cols = master_df.select_dtypes(include="number").columns
    master_df[numeric_cols] = master_df[numeric_cols].round(3)

    # rename map
    master_df = master_df.rename(columns=COL_RENAME_MAP)
    return master_df

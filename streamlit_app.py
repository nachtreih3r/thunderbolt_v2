import streamlit as st
from src.github_store import get_file, put_file   # adjust if in src/ folder

st.title("Thunderbolt V2 — GitHub Storage Test")

try:
    blob, sha = get_file("master.csv")
    if blob is not None:
        if len(blob) > 0:
            st.success("Connected to GitHub ✅ (master.csv found)")
            st.write(f"File size: {len(blob)} bytes")
        else:
            st.info("Connected ✅ (master.csv is empty)")
    else:
        st.warning("Connected ✅ (master.csv not found)")
        if st.button("Create empty master.csv"):
            put_file("master.csv", b"", "init empty master.csv")
            st.success("Created master.csv — rerun to confirm.")
except Exception as e:
    st.error(f"GitHub connection failed: {e}")

# --- Upload DGR .xlsx -> use your Stage2 helpers -> save master.csv ---
import io
import pandas as pd
from src.stage2 import _drop_summary_rows, _drop_duplicate_brine_cols, _flatten_columns, COL_RENAME_MAP

st.subheader("Upload DGR .xlsx → update master.csv (uses Stage 2 cleaning)")

files = st.file_uploader("Drop DGR Excel files", type=["xlsx"], accept_multiple_files=True)

TIMESTAMP_FORMAT = "%d-%m-%Y %H%MH"   # match your stage2 default

def steamfield_from_xlsx_filelike(file_like, source_name: str) -> pd.DataFrame:
    # Try your Stage1 read pattern first; fallback if needed
    try:
        df = pd.read_excel(file_like, sheet_name="Steamfield", header=[0, 1], skiprows=[0], engine="openpyxl")
    except Exception:
        df = pd.read_excel(file_like, sheet_name="Steamfield", header=[0, 1], engine="openpyxl")
    # Stage 2 helpers
    df = _drop_summary_rows(df)
    df = _drop_duplicate_brine_cols(df)
    df = _flatten_columns(df)
    df["SourceFile"] = source_name

    # first column → Timestamp (same logic as stage2)
    ts_col = df.columns[0]
    df = df.rename(columns={ts_col: "Timestamp"})
    df["Timestamp"] = pd.to_datetime(df["Timestamp"], errors="coerce")
    df = df.dropna(subset=["Timestamp"]).sort_values("Timestamp").reset_index(drop=True)
    df["Timestamp"] = df["Timestamp"].dt.strftime(TIMESTAMP_FORMAT)

    # numeric conversion & rounding
    for c in df.columns:
        if c not in ["Timestamp", "SourceFile"]:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    num_cols = df.select_dtypes(include="number").columns
    df[num_cols] = df[num_cols].round(3)

    # final rename to your canonical names
    df = df.rename(columns=COL_RENAME_MAP)
    return df

if files:
    # 1) Parse all uploads into stage2-cleaned chunks
    new_chunks = []
    for f in files:
        try:
            new_df = steamfield_from_xlsx_filelike(f, f.name)
            new_chunks.append(new_df)
            st.success(f"Parsed {f.name}: {len(new_df)} rows")
        except Exception as e:
            st.error(f"{f.name}: {e}")

    if not new_chunks:
        st.stop()

    # 2) Load current master.csv from GitHub
    blob, sha = get_file("master.csv")
    if blob is None or len(blob) == 0:
        master_df = pd.DataFrame()
    else:
        master_df = pd.read_csv(io.BytesIO(blob))

    # 3) Merge new chunks into master
    if master_df.empty:
        combined = pd.concat(new_chunks, ignore_index=True)
    else:
        combined = pd.concat([master_df] + new_chunks, ignore_index=True)

    # Optional: dedupe by Timestamp (or Timestamp + SourceFile)
    if "Timestamp" in combined.columns:
        combined = combined.drop_duplicates(subset=["Timestamp"], keep="last")

    # 4) Save back to GitHub
    out_buf = io.StringIO()
    combined.to_csv(out_buf, index=False)
    put_file("master.csv", out_buf.getvalue().encode("utf-8"), "update master.csv", sha)

    # 5) Show results
    st.success("master.csv updated in GitHub ✅")
    st.dataframe(combined.tail(50))
    st.download_button(
        "Download current master.csv",
        data=out_buf.getvalue(),
        file_name="master.csv",
        mime="text/csv")

# ---- Password gate with persistent lockout & Remember Me ----
import time, hmac, hashlib, base64, os
import streamlit as st
st.set_page_config(page_title="Thunderbolt V2", layout="wide")

# allow either top-level secrets or nested under [auth] in secrets.toml
def _secret(key, default=None):
    if key in st.secrets:
        return st.secrets.get(key, default)
    if "auth" in st.secrets and key in st.secrets["auth"]:
        return st.secrets["auth"].get(key, default)
    return default

try:
    import bcrypt
    from streamlit_cookies_manager import CookieManager
except Exception:
    pass  # handled below

MAX_ATTEMPTS       = 5
COOLDOWN_SECONDS   = 5 * 60      # 5 minutes
REMEMBER_DAYS      = 3
COOKIE_PREFIX      = "tb_"
COOKIE_AUTH_NAME   = COOKIE_PREFIX + "auth"
COOKIE_LOCK_NAME   = COOKIE_PREFIX + "lock_until"
COOKIE_CLIENT_NAME = COOKIE_PREFIX + "cid"
SECRET             = _secret("COOKIE_SECRET", "")

def _cookies_ready():
    try:
        cm = CookieManager()
        if not cm.ready():
            st.stop()  # wait until cookies are ready
        return cm
    except Exception:
        st.error("Cookie manager missing. Add 'streamlit-cookies-manager' to requirements.txt.")
        st.stop()

def _b64(s: bytes) -> str:
    return base64.urlsafe_b64encode(s).decode().rstrip("=")

def _unb64(s: str) -> bytes:
    pad = "=" * (-len(s) % 4)
    return base64.urlsafe_b64decode((s + pad).encode())

def _sign(data: str) -> str:
    return _b64(hmac.new(SECRET.encode(), data.encode(), hashlib.sha256).digest())

def _make_token(client_id: str, exp_ts: int) -> str:
    payload = f"{client_id}.{exp_ts}"
    sig = _sign(payload)
    return f"{_b64(payload.encode())}.{sig}"

def _check_token(token: str, client_id: str) -> bool:
    try:
        payload_b64, sig = token.split(".", 1)
        payload = _unb64(payload_b64).decode()
        if not hmac.compare_digest(_sign(payload), sig):
            return False
        cid, exp_str = payload.split(".", 1)
        if not hmac.compare_digest(cid, client_id):
            return False
        return time.time() < int(exp_str)
    except Exception:
        return False

def _now_int() -> int:
    return int(time.time())

def require_password():
    # deps & secrets check
    if "bcrypt" not in globals() or "CookieManager" not in globals():
        st.error("Missing deps. Ensure 'bcrypt' and 'streamlit-cookies-manager' are in requirements.txt.")
        st.stop()
    if not _secret("APP_PASS_HASH") or not _secret("COOKIE_SECRET"):
        st.error("Missing secrets. Set APP_PASS_HASH and COOKIE_SECRET in Streamlit Secrets.")
        st.stop()

    cookies = _cookies_ready()

    # anonymous client id cookie
    client_id = cookies.get(COOKIE_CLIENT_NAME)
    if not client_id:
        client_id = _b64(os.urandom(16))
        cookies[COOKIE_CLIENT_NAME] = client_id
        cookies.save()

    # remember-me cookie
    auth_token = cookies.get(COOKIE_AUTH_NAME)
    if auth_token and _check_token(auth_token, client_id):
        st.session_state["auth_ok"] = True

    # already authed
    if st.session_state.get("auth_ok"):
        if st.sidebar.button("Log out"):
            for k in list(st.session_state.keys()):
                del st.session_state[k]
            cookies[COOKIE_AUTH_NAME] = ""
            cookies.save()
            st.rerun()
        return True

    # lockout via cookie (survives refresh/restart)
    lock_until = cookies.get(COOKIE_LOCK_NAME)
    if lock_until:
        try:
            if _now_int() < int(lock_until):
                remaining = int(lock_until) - _now_int()
                st.title("Thunderbolt V2 — Login")
                st.error(f"Too many attempts. Try again in {remaining}s.")
                st.stop()
        except Exception:
            pass

    # login form
    st.title("Thunderbolt V2 — Login")
    pwd = st.text_input("Enter password", type="password").strip()
    c1, c2 = st.columns([1,1])
    with c1:
        remember = st.checkbox(f"Remember me for {REMEMBER_DAYS} days")
    with c2:
        login = st.button("Login", type="primary", use_container_width=True)

        if login:
            try:
                pass_hash = _secret("APP_PASS_HASH")
                ok = bcrypt.checkpw(pwd.encode("utf-8"), pass_hash.encode("utf-8"))
                if hmac.compare_digest(str(bool(ok)), "True"):
                    st.session_state["auth_ok"] = True
                    if remember:
                        exp_ts = _now_int() + REMEMBER_DAYS * 24 * 3600
                        token = _make_token(client_id, exp_ts)
                        cookies[COOKIE_AUTH_NAME] = token
                        cookies.save()
                    # clear lock + attempts, then proceed
                    cookies[COOKIE_LOCK_NAME] = ""
                    cookies.save()
                    st.session_state.pop("attempts", None)
                    st.rerun()
                else:
                    attempts = st.session_state.get("attempts", 0) + 1
                    st.session_state["attempts"] = attempts
                    remaining = MAX_ATTEMPTS - attempts
                    if remaining <= 0:
                        cookies[COOKIE_LOCK_NAME] = str(_now_int() + COOLDOWN_SECONDS)
                        cookies.save()
                        st.error(f"Too many attempts. Locked for {COOLDOWN_SECONDS//60} minutes.")
                    else:
                        st.error(f"Incorrect password. {remaining} attempt(s) left.")
            except Exception as e:
                st.error(f"Auth error: {e}")

    st.stop()  # block the rest of the app until authenticated

# Call the gate before rendering the app
require_password()

# (optional) simple logout anywhere after this point
try:
    from streamlit_cookies_manager import CookieManager as _CM
    if st.session_state.get("auth_ok") and st.sidebar.button("Log out"):
        for k in list(st.session_state.keys()):
            del st.session_state[k]
        cm = _CM()
        if cm.ready():
            cm[COOKIE_AUTH_NAME] = ""
            cm.save()
        st.rerun()
except Exception:
    pass
# ---- end password gate ----

import io
from typing import List, Optional, Tuple
import pandas as pd

from src.github_store import get_file, put_file
from src.stage2 import (
    _drop_summary_rows,
    _drop_duplicate_brine_cols,
    _flatten_columns,
    COL_RENAME_MAP,
)

# ---------- App Config ----------
TITLE = "Thunderbolt V2 — GitHub Storage"
DATA_PATH = "master.csv"
TIMESTAMP_FORMAT = "%d-%m-%Y %H%MH"  # keep consistent with your stage2 default

# ---------- Debug Sidebar ----------
def debug_sidebar() -> None:
    st.sidebar.header("Debug")
    repo = st.secrets.get("GITHUB_REPO")
    branch = st.secrets.get("GITHUB_BRANCH", "main")
    st.sidebar.write(f"Repo: {repo}")
    st.sidebar.write(f"Branch: {branch}")
    st.sidebar.write(f"Path: {DATA_PATH}")

    blob, sha = get_file(DATA_PATH)
    size_now = len(blob) if blob else 0
    st.sidebar.write(f"GitHub size: **{size_now} bytes**")
    if blob and size_now > 0:
        try:
            gdf = pd.read_csv(io.BytesIO(blob))
            st.sidebar.write(f"Rows: **{len(gdf):,}**")
            if "Timestamp" in gdf.columns:
                st.sidebar.write(
                    f"Range: {gdf['Timestamp'].head(1).iloc[0]} → {gdf['Timestamp'].tail(1).iloc[0]}"
                )
        except Exception as e:
            st.sidebar.write(f"Read error: {e}")


def show_commit(result: dict) -> None:
    commit_url = result.get("commit", {}).get("html_url")
    commit_sha = result.get("commit", {}).get("sha", "")[:7]
    if commit_url:
        st.success(f"master.csv updated in GitHub ✅ • commit {commit_sha}")
        st.write(commit_url)
    else:
        st.success("master.csv updated in GitHub ✅")


# ---------- Core Functions ----------

def steamfield_from_xlsx(file_like, source_name: str) -> pd.DataFrame:
    """Read a DGR .xlsx *in-memory* (Steamfield sheet only) and clean to canonical columns."""
    # Try your Stage1 pattern first; fallback without skiprows
    try:
        df = pd.read_excel(
            file_like,
            sheet_name="Steamfield",
            header=[0, 1],
            skiprows=[0],
            engine="openpyxl",
        )
    except Exception:
        df = pd.read_excel(
            file_like,
            sheet_name="Steamfield",
            header=[0, 1],
            engine="openpyxl",
        )

    # Stage 2 helpers
    df = _drop_summary_rows(df)
    df = _drop_duplicate_brine_cols(df)
    df = _flatten_columns(df)
    df["SourceFile"] = source_name

    # First column → Timestamp, then format & sort
    ts_col = df.columns[0]
    df = df.rename(columns={ts_col: "Timestamp"})
    df["Timestamp"] = pd.to_datetime(df["Timestamp"], errors="coerce")
    df = df.dropna(subset=["Timestamp"]).sort_values("Timestamp").reset_index(drop=True)
    df["Timestamp"] = df["Timestamp"].dt.strftime(TIMESTAMP_FORMAT)

    # Numeric conversion & rounding
    for c in df.columns:
        if c not in ["Timestamp", "SourceFile"]:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    num_cols = df.select_dtypes(include="number").columns
    if len(num_cols) > 0:
        df[num_cols] = df[num_cols].round(3)

    # Canonical rename map
    df = df.rename(columns=COL_RENAME_MAP)
    return df


def load_master() -> Tuple[pd.DataFrame, Optional[str]]:
    """Load current master.csv from GitHub. Returns (DataFrame, sha)."""
    blob, sha = get_file(DATA_PATH)
    if blob is None or len(blob) == 0:
        return pd.DataFrame(), sha
    master_df = pd.read_csv(io.BytesIO(blob))
    return master_df, sha


def merge_and_dedupe(master_df: pd.DataFrame, parts: List[pd.DataFrame]) -> pd.DataFrame:
    if master_df is None or master_df.empty:
        combined = pd.concat(parts, ignore_index=True)
    else:
        combined = pd.concat([master_df] + parts, ignore_index=True)

    # Prefer Timestamp+SourceFile if both exist; else fallback to Timestamp only
    if {"Timestamp", "SourceFile"}.issubset(combined.columns):
        combined = combined.drop_duplicates(subset=["Timestamp", "SourceFile"], keep="last")
    elif "Timestamp" in combined.columns:
        combined = combined.drop_duplicates(subset=["Timestamp"], keep="last")

    return combined


def save_master(df: pd.DataFrame, sha: Optional[str]) -> dict:
    out_buf = io.StringIO()
    df.to_csv(out_buf, index=False)
    return put_file(DATA_PATH, out_buf.getvalue().encode("utf-8"), "update master.csv", sha)


# ---------- Page Body ----------
st.title(TITLE)
debug_sidebar()

# 3 Subtabs
tab1, tab2, tab3 = st.tabs([
    "Upload & Merge",
    "Master CSV",
    "Analysis (coming soon)",
])

# -------------------- Tab 1: Upload & Merge --------------------
with tab1:
    st.subheader("Upload DGRs and then execute extraction/merge")

    # Queue in session_state
    if "upload_queue" not in st.session_state:
        st.session_state.upload_queue = []

    # Uploader
    new_files = st.file_uploader(
        "Drop DGR Excel files (.xlsx)",
        type=["xlsx"],
        accept_multiple_files=True,
        key="uploader_tab1",
    )

    col_a, col_b = st.columns([2, 1], gap="large")

    with col_a:
        st.write("**Queued DGR files** (waiting for extraction/merge):")
        if st.session_state.upload_queue:
            for item in st.session_state.upload_queue:
                st.write(f"• {item['name']}")
        else:
            st.caption("No files queued yet.")

    with col_b:
        if new_files:
            if st.button("Add selected to queue", type="primary"):
                # Store bytes to persist across reruns
                for f in new_files:
                    st.session_state.upload_queue.append({
                        "name": f.name,
                        "data": f.getvalue(),
                    })
                st.success(f"Added {len(new_files)} file(s) to queue.")
        if st.session_state.upload_queue:
            if st.button("Clear queue"):
                st.session_state.upload_queue = []
                st.info("Queue cleared.")

    st.markdown("---")
    st.write("When ready, execute extraction of the Steamfield sheet and merge into master.csv:")

    if st.button("Process queued files (extract + merge)", type="primary", use_container_width=True):
        successes, failures = [], []
        parsed_parts: List[pd.DataFrame] = []

        # 1) Parse queued files with confirmations
        for item in st.session_state.upload_queue:
            name = item["name"]
            try:
                buf = io.BytesIO(item["data"])
                part = steamfield_from_xlsx(buf, name)
                parsed_parts.append(part)
                successes.append(f"{name} — OK ({len(part)} rows)")
            except Exception as e:
                failures.append(f"{name} — FAILED: {e}")

        # Show per-file status
        if successes:
            st.success("\n".join(successes))
        if failures:
            st.error("\n".join(failures))

        if parsed_parts:
            # 2) Load master
            master_df, sha = load_master()

            # 3) Merge & dedupe
            before_rows = 0 if master_df is None or master_df.empty else len(master_df)
            combined = merge_and_dedupe(master_df, parsed_parts)
            after_rows = len(combined)
            st.info(f"Row delta: {before_rows:,} → {after_rows:,} (Δ {after_rows - before_rows:+,})")

            # 4) Save back to GitHub with commit link
            try:
                result = save_master(combined, sha)
                show_commit(result)
                st.dataframe(combined.tail(50))
                st.download_button(
                    "Download current master.csv",
                    data=combined.to_csv(index=False),
                    file_name="master.csv",
                    mime="text/csv",
                )
                # Clear queue only if we had at least one success
                st.session_state.upload_queue = []
            except Exception as e:
                st.error(f"Failed to save to GitHub: {e}")
        else:
            st.warning("Nothing to merge. Queue contained only failed/empty files.")

# -------------------- Tab 2: Master CSV --------------------
with tab2:
    st.subheader("Complete merged CSV preview")
    if st.button("Refresh from GitHub", help="Reload latest master.csv"):
        st.rerun()
    master_df, _ = load_master()
    if master_df is None or master_df.empty:
        st.info("master.csv is empty. Upload & process files in Tab 1.")
    else:
        st.write(f"Rows: **{len(master_df):,}**")
        if "Timestamp" in master_df.columns:
            st.write(
                f"Range: {master_df['Timestamp'].head(1).iloc[0]} → {master_df['Timestamp'].tail(1).iloc[0]}"
            )
        st.dataframe(master_df)
        st.download_button(
            "Download master.csv",
            data=master_df.to_csv(index=False),
            file_name="master.csv",
            mime="text/csv",
        )

    st.markdown("---")
    st.subheader("Danger zone")
    with st.expander("Flush master.csv (wipe all data)"):
        st.write("This replaces master.csv in GitHub with an empty file. Old versions remain in commit history.")
        confirm = st.checkbox("I understand and want to empty master.csv", key="confirm_flush_tab2")
        if st.button("Flush now", type="primary", disabled=not confirm, key="flush_button_tab2"):
            try:
                _, sha_current = get_file(DATA_PATH)
                result = put_file(DATA_PATH, b"", "flush master.csv to empty", sha_current)
                show_commit(result)
                st.success("master.csv has been emptied. Go to Tab 1 to rebuild.")
            except Exception as e:
                st.error(f"Flush failed: {e}")

# -------------------- Tab 3: IF97 Regions (analysis) --------------------
# pip install iapws
import io
import numpy as np
import pandas as pd
import streamlit as st

@st.cache_resource
def _load_iapws():
    """Lazy import so app still boots even if iapws isn't installed locally."""
    try:
        from iapws.iapws97 import _Region1, _Region2, _Region3, _Region4, _Region5, _TSat_P, _PSat_T
        return dict(
            _Region1=_Region1, _Region2=_Region2, _Region3=_Region3, _Region4=_Region4, _Region5=_Region5,
            _TSat_P=_TSat_P, _PSat_T=_PSat_T
        )
    except Exception as e:
        raise RuntimeError("iapws package not available. Install with: pip install iapws") from e


REGIONS = {
    "Region 1 (compressed water: T,P)": {
        "desc": "Subcooled/Compressed liquid water. Inputs: Temperature [K], Pressure [MPa].",
        "inputs": [("T (K)", "T"), ("P (MPa)", "P")],
        "returns": [
            ("v", "Specific volume [m³/kg]"),
            ("h", "Specific enthalpy [kJ/kg]"),
            ("s", "Specific entropy [kJ/(kg·K)]"),
            ("cp", "Isobaric heat capacity [kJ/(kg·K)]"),
            ("cv", "Isochoric heat capacity [kJ/(kg·K)]"),
            ("w", "Speed of sound [m/s]"),
            ("alfav", "Cubic expansion coeff. [1/K]"),
            ("kt", "Isothermal compressibility [1/MPa]"),
        ],
        "fn": "_Region1",
    },
    "Region 2 (superheated steam: T,P)": {
        "desc": "Superheated vapor/gas. Inputs: Temperature [K], Pressure [MPa].",
        "inputs": [("T (K)", "T"), ("P (MPa)", "P")],
        "returns": [
            ("v", "Specific volume [m³/kg]"),
            ("h", "Specific enthalpy [kJ/kg]"),
            ("s", "Specific entropy [kJ/(kg·K)]"),
            ("cp", "Isobaric heat capacity [kJ/(kg·K)]"),
            ("cv", "Isochoric heat capacity [kJ/(kg·K)]"),
            ("w", "Speed of sound [m/s]"),
            ("alfav", "Cubic expansion coeff. [1/K]"),
            ("kt", "Isothermal compressibility [1/MPa]"),
        ],
        "fn": "_Region2",
    },
    "Region 3 (near-critical dense: rho,T)": {
        "desc": "High-density/near-critical. Inputs: Density [kg/m³], Temperature [K].",
        "inputs": [("rho (kg/m³)", "rho"), ("T (K)", "T")],
        "returns": [
            ("v", "Specific volume [m³/kg]"),
            ("h", "Specific enthalpy [kJ/kg]"),
            ("s", "Specific entropy [kJ/(kg·K)]"),
            ("cp", "Isobaric heat capacity [kJ/(kg·K)]"),
            ("cv", "Isochoric heat capacity [kJ/(kg·K)]"),
            ("w", "Speed of sound [m/s]"),
            ("alfav", "Cubic expansion coeff. [1/K]"),
            ("kt", "Isothermal compressibility [1/MPa]"),
        ],
        "fn": "_Region3",
    },
    "Region 4 (saturation: P,x)": {
        "desc": "Two-phase boundary (liquid↔vapor). Inputs: Pressure [MPa], Quality x [-].",
        "inputs": [("P (MPa)", "P"), ("x (quality)", "x")],
        "returns": [
            ("T", "Saturated temperature [K]"),
            ("P", "Saturated pressure [MPa]"),
            ("x", "Vapor quality [-]"),
            ("v", "Specific volume [m³/kg]"),
            ("h", "Specific enthalpy [kJ/kg]"),
            ("s", "Specific entropy [kJ/(kg·K)]"),
        ],
        "fn": "_Region4",
    },
    "Region 5 (high-T steam: T,P)": {
        "desc": "Very hot steam region. Inputs: Temperature [K], Pressure [MPa].",
        "inputs": [("T (K)", "T"), ("P (MPa)", "P")],
        "returns": [
            ("v", "Specific volume [m³/kg]"),
            ("h", "Specific enthalpy [kJ/kg]"),
            ("s", "Specific entropy [kJ/(kg·K)]"),
            ("cp", "Isobaric heat capacity [kJ/(kg·K)]"),
            ("cv", "Isochoric heat capacity [kJ/(kg·K)]"),
            ("w", "Speed of sound [m/s]"),
            ("alfav", "Cubic expansion coeff. [1/K]"),
            ("kt", "Isothermal compressibility [1/MPa]"),
        ],
        "fn": "_Region5",
    },
}

# ---------- Timestamp parsing/formatting for 'DD-MM-YYYY HH00H' ----------
_TS_FORMAT = "%d-%m-%Y %H%M"  # e.g., '01-06-2025 0100'

def _parse_ddmmyyyy_hh00h(series: pd.Series) -> pd.Series:
    """Parse 'DD-MM-YYYY HH00H' -> datetime; non-parsable -> NaT."""
    s = series.astype(str).str.strip()
    # turn '0100H' -> '0100'
    s = s.str.replace(r"\s*(\d{4})H\s*$", r" \1", regex=True)
    return pd.to_datetime(s, format=_TS_FORMAT, errors="coerce")

def _format_ddmmyyyy_hh00h(obj):
    """
    Format datetime to 'DD-MM-YYYY HH00H'.
    - If obj is Series/DatetimeIndex -> returns Series of strings
    - If obj is scalar -> returns a single string
    """
    if isinstance(obj, (pd.Series, pd.DatetimeIndex)):
        return pd.to_datetime(obj).dt.strftime("%d-%m-%Y %H00H")
    else:
        return pd.to_datetime([obj]).strftime("%d-%m-%Y %H00H")[0]

def _resolve_datetime(master_df: pd.DataFrame) -> pd.Series:
    """
    Return a datetime Series from common patterns:
    1) 'Timestamp' in 'DD-MM-YYYY HH00H'
    2) 'timestamp' already parseable
    3) 'Date' + 'Time'
    4) First parseable date-like column
    """
    # 1) Explicit 'Timestamp' like in your screenshot
    for name in ["Timestamp", "TIMESTAMP", "timestamp_str"]:
        if name in master_df.columns:
            ts = _parse_ddmmyyyy_hh00h(master_df[name])
            if ts.notna().any():
                return ts

    # 2) Generic 'timestamp'
    if "timestamp" in master_df.columns:
        ts = pd.to_datetime(master_df["timestamp"], errors="coerce")
        if ts.notna().any():
            return ts

    # 3) Date + Time
    if {"Date", "Time"}.issubset(master_df.columns):
        return pd.to_datetime(
            master_df["Date"].astype(str) + " " + master_df["Time"].astype(str),
            errors="coerce"
        )

    # 4) Heuristic fallback
    for c in master_df.columns:
        if any(k in c.lower() for k in ("time", "date", "datetime", "timestamp")):
            s = pd.to_datetime(master_df[c], errors="coerce")
            if s.notna().any():
                return s

    raise ValueError("Could not parse a datetime column. Expected 'Timestamp' (DD-MM-YYYY HH00H), 'timestamp', or 'Date'+'Time'.")


def _column_mapper_ui(df: pd.DataFrame, inputs):
    """UI to map required inputs to CSV columns. Returns dict of key->column."""
    st.subheader("Map CSV columns to required inputs")
    mapping = {}
    for label, key in inputs:
        # small guesser
        guess = None
        for col in df.columns:
            low = col.lower()
            if key == "T" and (low in ("t", "temp", "temperature") or low.endswith("_k")):
                guess = col
            if key == "P" and (low in ("p", "press", "pressure") or "mpa" in low):
                guess = col
            if key == "rho" and ("rho" in low or "dens" in low):
                guess = col
            if key == "x" and (low == "x" or "quality" in low):
                guess = col
        mapping[key] = st.selectbox(
            f"{label}", ["— select —"] + list(df.columns),
            index=(df.columns.tolist().index(guess) + 1) if guess in df.columns else 0
        )
    return mapping


def _compute_row(fn, region_name: str, row: pd.Series, mapping: dict):
    """Call the selected Region function for a single row, return dict or None."""
    try:
        if region_name.startswith(("Region 1", "Region 2", "Region 5")):
            return fn(float(row[mapping["T"]]), float(row[mapping["P"]]))
        if region_name.startswith("Region 3"):
            return fn(float(row[mapping["rho"]]), float(row[mapping["T"]]))
        if region_name.startswith("Region 4"):
            return fn(float(row[mapping["P"]]), float(row[mapping["x"]]))
    except Exception:
        return None


def _validate_inputs(region_name: str, row: pd.Series, mapping: dict) -> tuple[bool, str | None]:
    """Light prevalidation to avoid silent NaNs."""
    try:
        if region_name.startswith(("Region 1", "Region 2", "Region 5")):
            T = float(row[mapping["T"]]); P = float(row[mapping["P"]])
            if not np.isfinite(T) or not np.isfinite(P): return False, "Non-numeric T/P"
            if T <= 0 or P <= 0: return False, "T or P <= 0"
        elif region_name.startswith("Region 3"):
            rho = float(row[mapping["rho"]]); T = float(row[mapping["T"]])
            if not np.isfinite(rho) or not np.isfinite(T): return False, "Non-numeric rho/T"
            if rho <= 0 or T <= 0: return False, "rho or T <= 0"
        elif region_name.startswith("Region 4"):
            P = float(row[mapping["P"]]); x = float(row[mapping["x"]])
            if not np.isfinite(P) or not np.isfinite(x): return False, "Non-numeric P/x"
            if P <= 0 or not (0.0 <= x <= 1.0): return False, "P<=0 or x not in [0,1]"
        return True, None
    except Exception as e:
        return False, str(e)


def render_tab3_iapws(master_df: pd.DataFrame):
    st.subheader("IAPWS‑IF97 Regions")
    st.caption("Pick a time window, a Region, map inputs from your CSV, choose outputs, then execute.")

    try:
        ts = _resolve_datetime(master_df)
    except Exception as e:
        st.error(str(e))
        return

    df = master_df.copy()
    df["__ts__"] = pd.to_datetime(ts)

    # -------- Step 1 — Date & time range --------
    st.markdown("**Step 1 — Date & time range**")

    min_dt = pd.to_datetime(df["__ts__"].min())
    max_dt = pd.to_datetime(df["__ts__"].max())

    # Show available window in your format
    st.caption(f"Available window: {_format_ddmmyyyy_hh00h(min_dt)} → {_format_ddmmyyyy_hh00h(max_dt)}")

    start, end = st.slider(
        "Window",
        min_value=min_dt.to_pydatetime(),
        max_value=max_dt.to_pydatetime(),
        value=(min_dt.to_pydatetime(), max_dt.to_pydatetime()),
        format="YYYY-MM-DD HH:mm",  # slider display; we print your custom format separately
    )
    mask = (df["__ts__"] >= pd.to_datetime(start)) & (df["__ts__"] <= pd.to_datetime(end))
    dfw = df.loc[mask].reset_index(drop=True)
    st.caption(f"Rows in window: {len(dfw):,}")

    # -------- Step 2 — Region selection --------
    st.markdown("**Step 2 — Select IF97 Region**")
    region_names = list(REGIONS.keys())
    region_choice = st.selectbox("Region", region_names)
    st.info(REGIONS[region_choice]["desc"])

    # Input mapping
    mapping = _column_mapper_ui(dfw, REGIONS[region_choice]["inputs"])

    # -------- Step 3 — Returns selection --------
    st.markdown("**Step 3 — Choose returns**")
    ret_pairs = REGIONS[region_choice]["returns"]
    ret_labels = [f"{k} — {desc}" for k, desc in ret_pairs]
    chosen_labels = st.multiselect("Outputs (multi‑select)", ret_labels, default=[ret_labels[0]])
    chosen_keys = [lbl.split(" — ")[0] for lbl in chosen_labels]

    # -------- Step 4 — Execute --------
    st.markdown("**Step 4 — Execute**")
    if st.button("Run IAPWS computation", type="primary"):
        if any(v == "— select —" for v in mapping.values()):
            st.error("Please map all required inputs to CSV columns.")
            st.stop()

        lib = _load_iapws()
        fn = lib[REGIONS[region_choice]["fn"]]

        error_msgs = []
        out_rows = []
        for idx, r in dfw.iterrows():
            ok, msg = _validate_inputs(region_choice, r, mapping)
            if not ok:
                if len(error_msgs) < 5:
                    error_msgs.append(f"Row {idx}: {msg}")
                d = {k: np.nan for k, _ in ret_pairs}
            else:
                try:
                    d = _compute_row(fn, region_choice, r, mapping)
                    if d is None:
                        d = {k: np.nan for k, _ in ret_pairs}
                except Exception as e:
                    if len(error_msgs) < 5:
                        error_msgs.append(f"Row {idx}: {type(e).__name__}: {e}")
                    d = {k: np.nan for k, _ in ret_pairs}

            row_out = {"timestamp": r["__ts__"]}
            # include inputs used for traceability
            for k, col in mapping.items():
                row_out[col] = r[col]
            # include requested returns
            for k in chosen_keys:
                row_out[k] = d.get(k, np.nan)
            out_rows.append(row_out)

        result = pd.DataFrame(out_rows)

        # Format timestamp to 'DD-MM-YYYY HH00H' for display/export
        result.insert(0, "Timestamp", _format_ddmmyyyy_hh00h(result["timestamp"]))
        # Order: Timestamp | inputs | outputs
        input_cols = list(mapping.values())
        cols = ["Timestamp"] + input_cols + chosen_keys
        result = result[cols]

        st.dataframe(result, use_container_width=True)

        if error_msgs:
            st.warning("Some rows could not be computed. First few issues:\n- " + "\n- ".join(error_msgs))

        # downloadable CSV
        buf = io.StringIO()
        result.to_csv(buf, index=False)
        st.download_button("Download CSV", buf.getvalue().encode("utf-8"),
                           file_name="iapws_results.csv", mime="text/csv")


# === integrate into your existing tabs ===
with tab3:
    render_tab3_iapws(master_df)
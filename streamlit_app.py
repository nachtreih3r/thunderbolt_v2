# ---- Password gate with persistent lockout & Remember Me ----
import time, hmac, hashlib, base64, os
import streamlit as st

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
import streamlit as st

from src.github_store import get_file, put_file
from src.stage2 import (
    _drop_summary_rows,
    _drop_duplicate_brine_cols,
    _flatten_columns,
    COL_RENAME_MAP,
)

# ---------- App Config ----------
st.set_page_config(page_title="Thunderbolt V2", layout="wide")
TITLE = "Thunderbolt V2 — GitHub Storage"
DATA_PATH = "master.csv"
TIMESTAMP_FORMAT = "%d-%m-%Y %H%MH"  # keep consistent with your stage2 default

# ---------- Debug Sidebar ----------
def debug_sidebar() -> None:
    st.sidebar.header("Debug")
    repo = st.secrets.get("GITHUB_REPO")
    branch = st.secrets.get("GITHUB_BRANCH", "main")
    st.sidebar.write(f"Repo: `{repo}`")
    st.sidebar.write(f"Branch: `{branch}`")
    st.sidebar.write(f"Path: `{DATA_PATH}`")

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
        st.success(f"master.csv updated in GitHub ✅ • commit `{commit_sha}`")
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
        st.session_state.upload_queue: List[dict] = []

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

    master_df, _ = load_master()
    if master_df is None or master_df.empty:
        st.info("`master.csv` is empty. Upload & process files in Tab 1.")
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
        st.write("This replaces `master.csv` in GitHub with an empty file. Old versions remain in commit history.")
        confirm = st.checkbox("I understand and want to empty master.csv", key="confirm_flush_tab2")
        if st.button("Flush now", type="primary", disabled=not confirm, key="flush_button_tab2"):
            try:
                _, sha_current = get_file(DATA_PATH)
                result = put_file(DATA_PATH, b"", "flush master.csv to empty", sha_current)
                show_commit(result)
                st.success("master.csv has been emptied. Go to Tab 1 to rebuild.")
            except Exception as e:
                st.error(f"Flush failed: {e}")

# -------------------- Tab 3: Analysis (placeholder) --------------------
with tab3:
    st.subheader("Analysis workspace (coming soon)")
    st.caption("We will add IAPWS-based calculations and visualizations here next.")

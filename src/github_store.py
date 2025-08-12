import base64, json, requests, streamlit as st

TOKEN  = st.secrets["GITHUB_TOKEN"]
REPO   = st.secrets["GITHUB_REPO"]        # "user/repo"
BRANCH = st.secrets.get("GITHUB_BRANCH", "main")
API    = "https://api.github.com"

def _headers():
    return {"Authorization": f"Bearer {TOKEN}", "Accept": "application/vnd.github+json"}

def get_file(path: str):
    """Return (bytes, sha) or (None, None) if not found."""
    url = f"{API}/repos/{REPO}/contents/{path}?ref={BRANCH}"
    r = requests.get(url, headers=_headers())
    if r.status_code == 200:
        data = r.json()
        return base64.b64decode(data["content"]), data["sha"]
    if r.status_code == 404:
        return None, None
    raise RuntimeError(f"GitHub GET {r.status_code}: {r.text}")

def put_file(path: str, content_bytes: bytes, message: str, sha: str | None = None):
    url = f"{API}/repos/{REPO}/contents/{path}"
    payload = {
        "message": message,
        "content": base64.b64encode(content_bytes).decode(),
        "branch": BRANCH,
    }
    if sha:
        payload["sha"] = sha
    r = requests.put(url, headers=_headers(), data=json.dumps(payload))
    if r.status_code not in (200, 201):
        raise RuntimeError(f"GitHub PUT {r.status_code}: {r.text}")
    return r.json()

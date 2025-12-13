# main.py
import os
import sys
import pathlib
import base64
import secrets
import time
import logging
from typing import Optional, Dict, Any, List, Tuple

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import RedirectResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
import requests
from dotenv import load_dotenv

# ------------------ Path setup ------------------
ROOT = pathlib.Path(__file__).resolve().parents[1]  # repo root
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# ------------------ Logging ------------------
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("music-backend")

# ------------------ Load environment ------------------
ENV_PATH = ROOT / "backend" / "API_V.env"
if not ENV_PATH.exists():
    logger.warning("API_V.env file not found at %s", ENV_PATH)
load_dotenv(dotenv_path=ENV_PATH)

SPOTIFY_CLIENT_ID = os.getenv("SPOTIFY_CLIENT_ID")
SPOTIFY_CLIENT_SECRET = os.getenv("SPOTIFY_CLIENT_SECRET")
SPOTIFY_REDIRECT_URI = os.getenv("SPOTIFY_REDIRECT_URI", "http://127.0.0.1:8000/callback")
CLASSIFIER_MODE = os.getenv("CLASSIFIER_MODE", "supervised").lower()
GITHUB_TOKEN = os.getenv("GITHUB_TOKEN")

if not SPOTIFY_CLIENT_ID or not SPOTIFY_CLIENT_SECRET:
    raise RuntimeError("Spotify credentials missing in environment")

# ------------------ FastAPI setup ------------------
app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # dev only
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

BASE_DIR = pathlib.Path(__file__).resolve().parent
FRONTEND_DIR = BASE_DIR.parent / "frontend"
if FRONTEND_DIR.exists():
    app.mount("/static", StaticFiles(directory=str(FRONTEND_DIR), html=True), name="frontend")
    logger.info("Mounted frontend: %s", FRONTEND_DIR)
else:
    logger.warning("Frontend directory not found: %s", FRONTEND_DIR)

# ------------------ Models ------------------
class UserText(BaseModel):
    text: str
    k: int = 3
    limit: int = 3

class TrackOut(BaseModel):
    name: Optional[str] = None
    artist: Optional[str] = None
    preview_url: Optional[str] = None
    spotify_url: Optional[str] = None

class MoodScore(BaseModel):
    label: str
    score: float

class AnalyzeOut(BaseModel):
    moods: List[MoodScore]
    tracks: List[TrackOut]

TOKENS: Dict[str, Dict[str, Any]] = {}

# ------------------ Classifiers ------------------
from backend.ml.classify import MoodClassifier
from backend.ml.supervised import SupervisedMoodClassifier

def classify_mood_local(text: str) -> str:
    t = (text or "").lower()
    if any(w in t for w in ["calm", "peace", "relax"]):
        return "Calm"
    if any(w in t for w in ["energy", "active", "excited", "run", "workout"]):
        return "Energetic"
    if any(w in t for w in ["focus", "study", "concentrate"]):
        return "Focus"
    return "Uplifting"

def classify_mood_github_gpt(text: str) -> str:
    if not GITHUB_TOKEN:
        return classify_mood_local(text)
    url = "https://models.github.ai/inference/chat/completions"
    headers = {"Authorization": f"Bearer {GITHUB_TOKEN}", "Content-Type": "application/json"}
    prompt = f'Classify mood into one of: Calm, Energetic, Focus, Uplifting. User input: "{text}"'
    payload = {"model": "openai/gpt-4o", "messages": [{"role": "user", "content": prompt}], "temperature": 0, "max_tokens": 5}
    try:
        r = requests.post(url, json=payload, headers=headers, timeout=12)
        r.raise_for_status()
        out = (r.json().get("choices") or [{}])[0].get("message", {}).get("content", "").strip().splitlines()[0].capitalize()
        if out in {"Calm", "Energetic", "Focus", "Uplifting"}:
            return out
        return classify_mood_local(text)
    except Exception as e:
        logger.error("GitHub GPT error: %s", e)
        return classify_mood_local(text)

def choose_classifier(mode: str):
    mode = (mode or CLASSIFIER_MODE).lower()
    if mode == "supervised":
        return SupervisedMoodClassifier()
    if mode == "zero_shot":
        return MoodClassifier()
    if mode == "github":
        return None
    return SupervisedMoodClassifier()

# ------------------ Spotify helpers ------------------
def _basic_auth():
    b64 = base64.b64encode(f"{SPOTIFY_CLIENT_ID}:{SPOTIFY_CLIENT_SECRET}".encode()).decode()
    return {"Authorization": f"Basic {b64}", "Content-Type": "application/x-www-form-urlencoded"}

def exchange_code_for_tokens(code: str):
    r = requests.post(
        "https://accounts.spotify.com/api/token",
        data={"grant_type": "authorization_code", "code": code, "redirect_uri": SPOTIFY_REDIRECT_URI},
        headers=_basic_auth()
    )
    if r.status_code != 200:
        raise HTTPException(400, f"Token exchange failed: {r.text}")
    return r.json()

def refresh_access_token(refresh: str):
    r = requests.post(
        "https://accounts.spotify.com/api/token",
        data={"grant_type": "refresh_token", "refresh_token": refresh},
        headers=_basic_auth()
    )
    if r.status_code != 200:
        raise HTTPException(400, f"Refresh failed: {r.text}")
    return r.json()

def store_tokens(state: str, t: Dict[str, Any]):
    TOKENS[state] = {
        "access_token": t["access_token"],
        "refresh_token": t.get("refresh_token"),
        "expires_at": time.time() + t.get("expires_in", 3600)
    }
    logger.info("Stored tokens for state %s", state)

def get_valid_token(state: str) -> Optional[str]:
    obj = TOKENS.get(state)
    if not obj:
        return None
    if time.time() >= obj.get("expires_at", 0) - 20:
        if not obj.get("refresh_token"):
            return None
        new = refresh_access_token(obj["refresh_token"])
        obj["access_token"] = new["access_token"]
        obj["expires_at"] = time.time() + new.get("expires_in", 3600)
        if new.get("refresh_token"):
            obj["refresh_token"] = new.get("refresh_token")
    return obj.get("access_token")

# ------------------ OAuth routes ------------------
@app.get("/login")
def login():
    state = secrets.token_urlsafe(16)
    TOKENS[state] = {}
    scope = "user-read-private user-read-email streaming user-read-playback-state user-modify-playback-state"
    url = (
        f"https://accounts.spotify.com/authorize"
        f"?client_id={SPOTIFY_CLIENT_ID}&response_type=code"
        f"&redirect_uri={SPOTIFY_REDIRECT_URI}&scope={scope}&state={state}"
    )
    return RedirectResponse(url)

@app.get("/callback")
def callback(code: str, state: str):
    if state not in TOKENS:
        raise HTTPException(400, "Invalid state")
    token_info = exchange_code_for_tokens(code)
    store_tokens(state, token_info)
    return RedirectResponse(f"/static/index.html?spotify_state={state}")

# ------------------ Routes ------------------
@app.post("/predict", response_model=List[MoodScore])
def predict(data: UserText, source: Optional[str] = Query(None)):
    if (source or CLASSIFIER_MODE).lower() == "github":
        mood = classify_mood_github_gpt(data.text)
        return [MoodScore(label=mood, score=1.0)]
    clf = choose_classifier(source)
    preds = clf.predict(data.text, k=data.k)
    s = sum(p["score"] for p in preds) or 1.0
    return [MoodScore(label=p["label"], score=float(p["score"] / s)) for p in preds]

@app.get("/health")
def health():
    return {"ok": True}

@app.get("/debug/tokens")
def dbg():
    return TOKENS

# ------------------ Run ------------------
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)

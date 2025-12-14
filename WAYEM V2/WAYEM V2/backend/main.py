# main.py
import os
import sys
import pathlib
import base64
import secrets
import time
import logging
from typing import Optional, Dict, Any, List

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import RedirectResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
import requests
from dotenv import load_dotenv

# ------------------ Path setup ------------------
ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# ------------------ Logging ------------------
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("music-backend")

# ------------------ Load environment ------------------
ENV_PATH = ROOT / "backend" / "API_V.env"
load_dotenv(dotenv_path=ENV_PATH)

SPOTIFY_CLIENT_ID = os.getenv("SPOTIFY_CLIENT_ID")
SPOTIFY_CLIENT_SECRET = os.getenv("SPOTIFY_CLIENT_SECRET")
SPOTIFY_REDIRECT_URI = os.getenv("SPOTIFY_REDIRECT_URI", "http://127.0.0.1:8000/callback")
CLASSIFIER_MODE = os.getenv("CLASSIFIER_MODE", "supervised").lower()
GITHUB_TOKEN = os.getenv("GITHUB_TOKEN")

if not SPOTIFY_CLIENT_ID or not SPOTIFY_CLIENT_SECRET:
    raise RuntimeError("Spotify credentials missing in environment")

# ------------------ FastAPI setup ------------------
app = FastAPI(title="WAYEM Music Backend")

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

# ------------------ Models ------------------
class UserText(BaseModel):
    text: str
    k: int = 3
    limit: int = 3

class MoodScore(BaseModel):
    label: str
    score: float

class TrackOut(BaseModel):
    name: Optional[str]
    artist: Optional[str]
    preview_url: Optional[str]
    spotify_url: Optional[str]

class AnalyzeOut(BaseModel):
    moods: List[MoodScore]
    tracks: List[TrackOut]

TOKENS: Dict[str, Dict[str, Any]] = {}

# ------------------ Classifiers ------------------
from backend.ml.classify import MoodClassifier
from backend.ml.supervised import SupervisedMoodClassifier

def classify_mood_local(text: str) -> str:
    t = text.lower()
    if any(w in t for w in ["calm", "relax", "peace"]):
        return "Calm"
    if any(w in t for w in ["energy", "workout", "active"]):
        return "Energetic"
    if any(w in t for w in ["focus", "study"]):
        return "Focus"
    return "Uplifting"

def classify_mood_github_gpt(text: str) -> str:
    if not GITHUB_TOKEN:
        return classify_mood_local(text)

    url = "https://models.github.ai/inference/chat/completions"
    headers = {
        "Authorization": f"Bearer {GITHUB_TOKEN}",
        "Content-Type": "application/json",
    }
    payload = {
        "model": "openai/gpt-4o",
        "messages": [{
            "role": "user",
            "content": f'Classify mood into: Calm, Energetic, Focus, Uplifting. Text: "{text}"'
        }],
        "temperature": 0,
        "max_tokens": 5
    }

    try:
        r = requests.post(url, json=payload, headers=headers, timeout=10)
        r.raise_for_status()
        out = r.json()["choices"][0]["message"]["content"].strip()
        return out if out in {"Calm", "Energetic", "Focus", "Uplifting"} else classify_mood_local(text)
    except Exception:
        return classify_mood_local(text)

def choose_classifier():
    if CLASSIFIER_MODE == "zero_shot":
        return MoodClassifier()
    return SupervisedMoodClassifier()

# ------------------ Spotify helpers ------------------
def _basic_auth():
    b64 = base64.b64encode(f"{SPOTIFY_CLIENT_ID}:{SPOTIFY_CLIENT_SECRET}".encode()).decode()
    return {"Authorization": f"Basic {b64}", "Content-Type": "application/x-www-form-urlencoded"}

def exchange_code_for_tokens(code: str):
    r = requests.post(
        "https://accounts.spotify.com/api/token",
        data={
            "grant_type": "authorization_code",
            "code": code,
            "redirect_uri": SPOTIFY_REDIRECT_URI,
        },
        headers=_basic_auth(),
    )
    r.raise_for_status()
    return r.json()

def refresh_access_token(refresh: str):
    r = requests.post(
        "https://accounts.spotify.com/api/token",
        data={"grant_type": "refresh_token", "refresh_token": refresh},
        headers=_basic_auth(),
    )
    r.raise_for_status()
    return r.json()

def store_tokens(state: str, t: Dict[str, Any]):
    TOKENS[state] = {
        "access_token": t["access_token"],
        "refresh_token": t.get("refresh_token"),
        "expires_at": time.time() + t.get("expires_in", 3600),
    }

def get_valid_token(state: str) -> Optional[str]:
    obj = TOKENS.get(state)
    if not obj:
        return None

    if time.time() > obj["expires_at"] - 20:
        if not obj.get("refresh_token"):
            return None
        new = refresh_access_token(obj["refresh_token"])
        obj["access_token"] = new["access_token"]
        obj["expires_at"] = time.time() + new.get("expires_in", 3600)

    return obj["access_token"]

def get_spotify_tracks(mood: str, token: str, limit: int) -> List[TrackOut]:
    headers = {"Authorization": f"Bearer {token}"}
    q = {
        "Calm": "chill",
        "Energetic": "workout",
        "Focus": "focus",
        "Uplifting": "happy"
    }.get(mood, "music")

    r = requests.get(
        "https://api.spotify.com/v1/search",
        headers=headers,
        params={"q": q, "type": "track", "limit": limit},
        timeout=10,
    )

    if r.status_code != 200:
        return []

    tracks = []
    for t in r.json()["tracks"]["items"]:
        tracks.append(TrackOut(
            name=t["name"],
            artist=", ".join(a["name"] for a in t["artists"]),
            preview_url=t["preview_url"],
            spotify_url=t["external_urls"]["spotify"],
        ))
    return tracks

# ------------------ OAuth routes ------------------
@app.get("/login")
def login():
    state = secrets.token_urlsafe(16)
    TOKENS[state] = {}
    scope = "user-read-email streaming"
    url = (
        "https://accounts.spotify.com/authorize"
        f"?client_id={SPOTIFY_CLIENT_ID}"
        "&response_type=code"
        f"&redirect_uri={SPOTIFY_REDIRECT_URI}"
        f"&scope={scope}"
        f"&state={state}"
    )
    return RedirectResponse(url)

@app.get("/callback")
def callback(code: str, state: str):
    if state not in TOKENS:
        raise HTTPException(400, "Invalid state")
    tokens = exchange_code_for_tokens(code)
    store_tokens(state, tokens)
    return RedirectResponse(f"/static/index.html?spotify_state={state}")

# ------------------ API routes ------------------
@app.post("/predict", response_model=List[MoodScore])
def predict(data: UserText):
    clf = choose_classifier()
    preds = clf.predict(data.text, k=data.k)
    total = sum(p["score"] for p in preds) or 1.0
    return [MoodScore(label=p["label"], score=p["score"] / total) for p in preds]

@app.post("/analyze", response_model=AnalyzeOut)
def analyze(data: UserText, spotify_state: str = Query(...)):
    token = get_valid_token(spotify_state)
    if not token:
        raise HTTPException(401, "Spotify session expired")

    clf = choose_classifier()
    preds = clf.predict(data.text, k=data.k)
    total = sum(p["score"] for p in preds) or 1.0
    moods = [MoodScore(label=p["label"], score=p["score"] / total) for p in preds]

    top_mood = moods[0].label
    tracks = get_spotify_tracks(top_mood, token, data.limit)

    return AnalyzeOut(moods=moods, tracks=tracks)

@app.get("/health")
def health():
    return {"ok": True}

@app.get("/debug/tokens")
def debug_tokens():
    return TOKENS

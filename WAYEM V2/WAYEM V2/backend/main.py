# main.py
import os
import base64
import secrets
import time
import pathlib
from typing import Optional, Dict, Any, List, Tuple
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import RedirectResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from dotenv import load_dotenv
import requests
import logging
import json
# --- add at the very top of backend/main.py, before other imports ---
import sys, pathlib
ROOT = pathlib.Path(__file__).resolve().parents[1]  # repo root
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
# --------------------------------------------------------------------


# ------------------ Config ------------------
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("music-backend")

load_dotenv()
SPOTIFY_CLIENT_ID = os.getenv("SPOTIFY_CLIENT_ID")
SPOTIFY_CLIENT_SECRET = os.getenv("SPOTIFY_CLIENT_SECRET")
SPOTIFY_REDIRECT_URI = os.getenv("SPOTIFY_REDIRECT_URI", "http://127.0.0.1:8000/callback")

# If set, you can force a classifier source: "supervised" | "zero_shot" | "github"
CLASSIFIER_MODE = os.getenv("CLASSIFIER_MODE", "supervised").lower()

# GitHub Models API token (optional for github mode)
GITHUB_TOKEN = os.getenv("GITHUB_TOKEN")

if not SPOTIFY_CLIENT_ID or not SPOTIFY_CLIENT_SECRET:
    raise RuntimeError("Spotify credentials missing")

# ------------------ FastAPI ------------------
app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # dev only — tighten for prod
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

BASE_DIR = pathlib.Path(__file__).resolve().parent
FRONTEND_DIR = str(BASE_DIR.parent / "frontend")
app.mount("/static", StaticFiles(directory=FRONTEND_DIR, html=True), name="frontend")
logger.info(f"Mounted frontend: {FRONTEND_DIR}")

# ------------------ Models ------------------
class UserText(BaseModel):
    text: str
    k: int = 3            # top-k moods to consider
    limit: int = 3        # number of tracks to fetch

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
# Local four-bucket fallback (kept for completeness)
def classify_mood_local(text: str) -> str:
    t = (text or "").lower()
    if any(w in t for w in ["calm", "peace", "relax"]):
        return "Calm"
    if any(w in t for w in ["energy", "active", "excited", "run", "workout"]):
        return "Energetic"
    if any(w in t for w in ["focus", "study", "concentrate"]):
        return "Focus"
    return "Uplifting"

# GitHub Models API 4-class hack (kept as optional)
def classify_mood_github_gpt(text: str) -> str:
    if not GITHUB_TOKEN:
        return classify_mood_local(text)

    url = "https://models.github.ai/inference/chat/completions"
    headers = {
        "Authorization": f"Bearer {GITHUB_TOKEN}",
        "Content-Type": "application/json"
    }
    prompt = f"""
You classify mood into exactly one category from this set:
Calm, Energetic, Focus, Uplifting.
User input: "{text}"
Respond with ONE WORD ONLY from the set above.
"""
    payload = {
        "model": "openai/gpt-4o",
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0,
        "max_tokens": 5
    }
    try:
        r = requests.post(url, json=payload, headers=headers, timeout=12)
        r.raise_for_status()
        body = r.json()
        out = None
        if isinstance(body, dict):
            choices = body.get("choices") or []
            if choices:
                out = (choices[0].get("message", {}) or {}).get("content")
        mood = (out or "").strip().splitlines()[0].capitalize()
        if mood in {"Calm", "Energetic", "Focus", "Uplifting"}:
            return mood
        logger.warning("Unexpected GitHub GPT mood '%s'; falling back local", mood)
        return classify_mood_local(text)
    except Exception as e:
        logger.error("GitHub GPT error: %s", e)
        return classify_mood_local(text)

# Our real models
from backend.ml.classify import MoodClassifier
from backend.ml.supervised import SupervisedMoodClassifier

def choose_classifier(mode: str):
    mode = (mode or CLASSIFIER_MODE).lower()
    if mode == "supervised":
        return SupervisedMoodClassifier()
    if mode == "zero_shot":
        return MoodClassifier()
    if mode == "github":
        return None  # handled separately
    # default
    return SupervisedMoodClassifier()

# ------------------ Feature targets (Spotify) ------------------
# Blending targets across moods lets /recommendations use audio features.
# (Derived from your ontology mapping.)
MOOD_TO_FEATURES: Dict[str, Dict[str, float]] = {
    "Calm": {"target_energy": 0.25, "target_valence": 0.45, "target_acousticness": 0.6, "target_instrumentalness": 0.5},
    "Chill": {"target_energy": 0.35, "target_valence": 0.55, "target_danceability": 0.5, "target_acousticness": 0.55},
    "Mellow": {"target_energy": 0.4, "target_valence": 0.5, "target_danceability": 0.45, "target_acousticness": 0.5},
    "Dreamy": {"target_energy": 0.35, "target_valence": 0.5, "target_acousticness": 0.5, "target_instrumentalness": 0.4},
    "Atmospheric": {"target_energy": 0.4, "target_valence": 0.4, "target_acousticness": 0.4, "target_instrumentalness": 0.6},
    "Hypnotic": {"target_energy": 0.45, "target_valence": 0.45, "target_instrumentalness": 0.6, "target_danceability": 0.6},
    "Focus": {"target_energy": 0.3, "target_valence": 0.45, "target_instrumentalness": 0.7, "target_speechiness": 0.05},
    "Introspective": {"target_energy": 0.3, "target_valence": 0.3, "target_acousticness": 0.45},
    "Melancholic": {"target_energy": 0.25, "target_valence": 0.2, "target_acousticness": 0.45},
    "Somber": {"target_energy": 0.2, "target_valence": 0.15, "target_acousticness": 0.5},
    "Brooding": {"target_energy": 0.35, "target_valence": 0.25, "target_acousticness": 0.35},
    "Dark": {"target_energy": 0.4, "target_valence": 0.2, "target_acousticness": 0.3},
    "Warm": {"target_energy": 0.45, "target_valence": 0.7, "target_acousticness": 0.55},
    "Bright": {"target_energy": 0.65, "target_valence": 0.85, "target_danceability": 0.65},
    "Uplifting": {"target_energy": 0.65, "target_valence": 0.8, "target_danceability": 0.7},
    "Euphoric": {"target_energy": 0.8, "target_valence": 0.8, "target_danceability": 0.75},
    "Playful": {"target_energy": 0.7, "target_valence": 0.75, "target_danceability": 0.8},
    "Romantic": {"target_energy": 0.4, "target_valence": 0.65, "target_acousticness": 0.5},
    "Sensual": {"target_energy": 0.45, "target_valence": 0.6, "target_danceability": 0.65},
    "Bittersweet": {"target_energy": 0.35, "target_valence": 0.4, "target_acousticness": 0.5},
    "Nostalgic": {"target_energy": 0.45, "target_valence": 0.5, "target_acousticness": 0.45},
    "Hopeful": {"target_energy": 0.55, "target_valence": 0.75, "target_danceability": 0.55},
    "Triumphant": {"target_energy": 0.75, "target_valence": 0.8, "target_danceability": 0.7},
    "Epic": {"target_energy": 0.8, "target_valence": 0.6, "target_danceability": 0.6},
    "Energetic": {"target_energy": 0.9, "target_valence": 0.7, "target_danceability": 0.8},
    "Aggressive": {"target_energy": 0.95, "target_valence": 0.4, "target_danceability": 0.6},
    "Gritty": {"target_energy": 0.75, "target_valence": 0.45, "target_acousticness": 0.3},
    "Groovy": {"target_energy": 0.7, "target_valence": 0.7, "target_danceability": 0.9},
    "Confident": {"target_energy": 0.8, "target_valence": 0.7, "target_danceability": 0.75},
    "Tense": {"target_energy": 0.6, "target_valence": 0.3},
    "Dramatic": {"target_energy": 0.55, "target_valence": 0.35, "target_acousticness": 0.4},
    "Ethereal": {"target_energy": 0.35, "target_valence": 0.5, "target_instrumentalness": 0.55, "target_acousticness": 0.5}
}

# Optional: genre fallbacks per mood (helps seed recommendations)
MOOD_TO_GENRES: Dict[str, List[str]] = {
    "Calm": ["ambient", "acoustic", "chill"],
    "Chill": ["chill", "lo-fi", "indie"],
    "Mellow": ["singer-songwriter", "indie", "acoustic"],
    "Dreamy": ["dreampop", "shoegaze", "ambient"],
    "Atmospheric": ["ambient", "post-rock", "cinematic"],
    "Hypnotic": ["techno", "minimal", "trance"],
    "Focus": ["ambient", "classical", "instrumental"],
    "Introspective": ["indie", "alt-rock", "acoustic"],
    "Melancholic": ["sad", "indie", "post-rock"],
    "Somber": ["ambient", "neoclassical", "sad"],
    "Brooding": ["darkwave", "post-punk", "industrial"],
    "Dark": ["industrial", "darkwave", "metal"],
    "Warm": ["indie", "folk", "acoustic"],
    "Bright": ["pop", "dance", "indie-pop"],
    "Uplifting": ["pop", "dance", "edm"],
    "Euphoric": ["edm", "trance", "progressive-house"],
    "Playful": ["indie-pop", "electropop", "funk"],
    "Romantic": ["rnb", "soul", "acoustic"],
    "Sensual": ["rnb", "soul", "chill"],
    "Bittersweet": ["indie", "alt-rock", "singer-songwriter"],
    "Nostalgic": ["retro", "synthwave", "classic-rock"],
    "Hopeful": ["pop", "indie-pop", "folk"],
    "Triumphant": ["epic", "orchestral", "rock"],
    "Epic": ["orchestral", "soundtrack", "trailer"],
    "Energetic": ["edm", "dance", "electronic"],
    "Aggressive": ["metal", "hardcore", "industrial"],
    "Gritty": ["garage", "grunge", "alt-rock"],
    "Groovy": ["funk", "disco", "house"],
    "Confident": ["hip-hop", "trap", "pop-rap"],
    "Tense": ["techno", "industrial", "soundtrack"],
    "Dramatic": ["cinematic", "soundtrack", "orchestral"],
    "Ethereal": ["ambient", "dreampop", "neoclassical"]
}

def blend_targets(moods: List[Tuple[str, float]]) -> Dict[str, float]:
    """
    Weighted average of MOOD_TO_FEATURES for top-k moods.
    """
    acc: Dict[str, float] = {}
    wsum = 0.0
    for label, w in moods:
        feats = MOOD_TO_FEATURES.get(label, {})
        for k, v in feats.items():
            acc[k] = acc.get(k, 0.0) + w * float(v)
        wsum += w
    if wsum <= 0:
        return {}
    return {k: round(v / wsum, 3) for k, v in acc.items()}

# ------------------ Spotify OAuth helpers ------------------
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
        logger.error("Token exchange failed: %s", r.text)
        raise HTTPException(400, f"Token exchange failed: {r.text}")
    return r.json()

def refresh_access_token(refresh: str):
    r = requests.post(
        "https://accounts.spotify.com/api/token",
        data={"grant_type": "refresh_token", "refresh_token": refresh},
        headers=_basic_auth()
    )
    if r.status_code != 200:
        logger.error("Refresh failed: %s", r.text)
        raise HTTPException(400, f"Refresh failed: {r.text}")
    return r.json()

def store_tokens(state: str, t: Dict[str, Any]):
    TOKENS[state] = {
        "access_token": t["access_token"],
        "refresh_token": t.get("refresh_token"),
        "expires_at": time.time() + t.get("expires_in", 3600)
    }
    logger.info("Stored tokens for state %s (expires_in=%s)", state, t.get("expires_in", 3600))

def get_valid_token(state: str) -> Optional[str]:
    obj = TOKENS.get(state)
    if not obj:
        return None
    if time.time() >= obj.get("expires_at", 0) - 20:
        if not obj.get("refresh_token"):
            return None
        logger.info("Refreshing access token for state %s", state)
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
    logger.info("Redirecting to Spotify authorize, state=%s", state)
    return RedirectResponse(url)

@app.get("/callback")
def callback(code: str, state: str):
    if state not in TOKENS:
        raise HTTPException(400, "Invalid state")
    token_info = exchange_code_for_tokens(code)
    store_tokens(state, token_info)
    logger.info("OAuth complete for state %s", state)
    return RedirectResponse(f"/static/index.html?spotify_state={state}")

# ------------------ Spotify recs (feature-driven) ------------------
def get_spotify_recommendations_by_features(
    token: str,
    seeds: List[str],
    targets: Dict[str, float],
    limit: int = 3
) -> List[Dict[str, Any]]:
    """
    Use /recommendations with seed_genres + target_* feature constraints.
    Falls back to a simple search if needed.
    """
    headers = {"Authorization": f"Bearer {token}"}
    rec_url = "https://api.spotify.com/v1/recommendations"
    params = {
        "limit": max(1, min(int(limit), 20)),
        "seed_genres": ",".join(seeds[:5]) if seeds else "pop",
    }
    # Only include target_* keys recognized by Spotify
    for k, v in targets.items():
        if k.startswith("target_"):
            params[k] = v

    # Primary attempt
    try:
        res = requests.get(rec_url, headers=headers, params=params, timeout=12)
    except Exception as e:
        logger.warning("Recommendations request error: %s", e)
        res = None

    tracks: List[Dict[str, Any]] = []
    if res is not None and res.status_code == 200:
        try:
            data = res.json()
            for t in (data.get("tracks") or []):
                tracks.append({
                    "name": t.get("name"),
                    "artist": (t.get("artists") or [{}])[0].get("name"),
                    "preview": t.get("preview_url"),
                    "spotify_url": (t.get("external_urls") or {}).get("spotify"),
                })
        except Exception:
            logger.warning("Non-JSON recommendations response")
    else:
        if res is not None:
            logger.warning("Recommendations status=%s text=%s", res.status_code, res.text)

    # Fallback to search if needed
    if not tracks:
        q = (seeds[0] if seeds else "music")
        try:
            s = requests.get(
                "https://api.spotify.com/v1/search",
                headers=headers,
                params={"q": q, "type": "track", "limit": limit},
                timeout=10,
            )
            if s.status_code == 200:
                data = s.json()
                for t in (data.get("tracks") or {}).get("items", []):
                    tracks.append({
                        "name": t.get("name"),
                        "artist": (t.get("artists") or [{}])[0].get("name"),
                        "preview": t.get("preview_url"),
                        "spotify_url": (t.get("external_urls") or {}).get("spotify"),
                    })
        except Exception as e:
            logger.error("Search fallback failed: %s", e)

    return tracks

# ------------------ Routes ------------------

@app.post("/predict", response_model=List[MoodScore])
def predict(data: UserText, source: Optional[str] = Query(None)):
    """
    Return top-k mood predictions with scores. Source can be:
    - supervised (default)
    - zero_shot
    - github (4-class fallback)
    """
    if (source or CLASSIFIER_MODE).lower() == "github":
        # 4-class legacy
        mood = classify_mood_github_gpt(data.text)
        return [MoodScore(label=mood, score=1.0)]

    clf = choose_classifier(source)
    # both classifiers expose predict(text, k)
    preds = clf.predict(data.text, k=data.k)
    # clip tiny tails and renormalize
    preds = [p for p in preds if p["score"] >= 0.01]
    s = sum(p["score"] for p in preds) or 1.0
    return [MoodScore(label=p["label"], score=float(p["score"] / s)) for p in preds]

@app.post("/analyze", response_model=AnalyzeOut)
def analyze(data: UserText, spotify_state: str = Query(...), source: Optional[str] = Query(None)):
    """
    End-to-end: top-k moods -> blended feature targets -> Spotify recommendations.
    """
    token = get_valid_token(spotify_state)
    if not token:
        raise HTTPException(status_code=401, detail="Invalid or expired Spotify session")

    # 1) Predict moods
    if (source or CLASSIFIER_MODE).lower() == "github":
        # 4-class legacy path
        mood = classify_mood_github_gpt(data.text)
        moods = [(mood, 1.0)]
    else:
        clf = choose_classifier(source)
        preds = clf.predict(data.text, k=data.k)
        # normalize
        s = sum(p["score"] for p in preds) or 1.0
        moods = [(p["label"], float(p["score"] / s)) for p in preds]

    # 2) Seeds + blended targets
    seeds: List[str] = []
    for label, _w in moods:
        seeds.extend(MOOD_TO_GENRES.get(label, []))
    if not seeds:
        seeds = ["pop"]
    # dedupe preserve order
    seen = set(); seeds = [g for g in seeds if not (g in seen or seen.add(g))]
    targets = blend_targets(moods)

    # 3) Spotify recommendations
    tracks_raw = get_spotify_recommendations_by_features(
        token=token,
        seeds=seeds,
        targets=targets,
        limit=data.limit
    )
    tracks = [TrackOut(name=t["name"], artist=t["artist"], preview_url=t["preview"], spotify_url=t["spotify_url"])
              for t in tracks_raw]

    return AnalyzeOut(
        moods=[MoodScore(label=lbl, score=score) for (lbl, score) in moods],
        tracks=tracks
    )

# Dev helpers
@app.get("/debug/tokens")
def dbg():
    return TOKENS

@app.get("/health")
def health():
    return {"ok": True}

# Run
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", port=8000, host="0.0.0.0", reload=True)

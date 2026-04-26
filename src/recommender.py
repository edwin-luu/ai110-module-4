import csv
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass, asdict

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

DEFAULT_WEIGHTS: Dict[str, float] = {
    "genre": 3.0,
    "mood": 3.0,
    "energy": 2.5,
    "acousticness": 1.5,
    "valence": 1.5,
    "danceability": 0.75,
    "tempo_bpm": 0.5,
}

CATEGORICAL_FEATURES = {"genre", "mood"}

TEMPO_MIN = 60
TEMPO_MAX = 200

# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class Song:
    """Represents a song and its attributes."""
    id: int
    title: str
    artist: str
    genre: List[str]
    mood: str
    energy: float
    tempo_bpm: float
    valence: float
    danceability: float
    acousticness: float


@dataclass
class UserProfile:
    """Represents a user's taste preferences."""
    favorite_genre: str
    favorite_mood: str
    target_energy: float
    likes_acoustic: bool
    target_valence: Optional[float] = None
    target_danceability: Optional[float] = None
    target_tempo_bpm: Optional[float] = None


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _normalize_tempo(bpm: float) -> float:
    """Normalize a BPM value to [0, 1] using a fixed range."""
    normalized = (bpm - TEMPO_MIN) / (TEMPO_MAX - TEMPO_MIN)
    return max(0.0, min(1.0, normalized))


def _build_explanation(user_prefs: Dict, song: Dict) -> str:
    """Build a human-readable explanation of why a song was recommended."""
    reasons = []
    for feature, user_val in user_prefs.items():
        if feature not in DEFAULT_WEIGHTS or feature not in song:
            continue

        if feature in CATEGORICAL_FEATURES:
            if feature == "genre":
                song_genres = song[feature] if isinstance(song[feature], list) else [song[feature]]
                if isinstance(user_val, list):
                    matched = [g for g in user_val if g in song_genres]
                    if matched:
                        reasons.append(f"genre match ({', '.join(matched)})")
                    else:
                        reasons.append(f"genre mismatch (none of {', '.join(user_val)} in tags)")
                else:
                    if user_val in song_genres:
                        reasons.append("genre match")
                    else:
                        reasons.append(f"genre mismatch ({user_val} not in tags)")
            else:
                if song[feature] == user_val:
                    reasons.append("mood match")
                else:
                    reasons.append(f"mood mismatch ({user_val} vs {song[feature]})")
        elif feature == "tempo_bpm":
            user_norm = _normalize_tempo(user_val)
            song_norm = _normalize_tempo(song[feature])
            sim = 1.0 - abs(user_norm - song_norm)
            reasons.append(f"tempo similarity {sim:.0%}")
        else:
            sim = 1.0 - abs(user_val - song[feature])
            reasons.append(f"energy similarity {sim:.0%}" if feature == "energy"
                          else f"{feature} similarity {sim:.0%}")

    return "; ".join(reasons) if reasons else "no matching features"


# ---------------------------------------------------------------------------
# Functional path (dict-based, used by main.py)
# ---------------------------------------------------------------------------

def load_songs(csv_path: str) -> List[Dict]:
    """Load songs from a CSV file, casting numeric fields to proper types."""
    numeric_fields = {"energy", "tempo_bpm", "valence", "danceability", "acousticness"}
    songs = []
    with open(csv_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            row["id"] = int(row["id"])
            row["genre"] = row["genre"].split("|")
            for field in numeric_fields:
                row[field] = float(row[field])
            songs.append(row)
    return songs


def score_song(user_prefs: Dict, song: Dict) -> float:
    """Score a song against user preferences using tier-weighted similarity."""
    if not user_prefs:
        return 0.0

    total_score = 0.0
    total_weight = 0.0

    for feature, user_val in user_prefs.items():
        if feature not in DEFAULT_WEIGHTS or feature not in song:
            continue

        weight = DEFAULT_WEIGHTS[feature]

        if feature in CATEGORICAL_FEATURES:
            if feature == "genre":
                song_genres = song[feature] if isinstance(song[feature], list) else [song[feature]]
                if isinstance(user_val, list):
                    similarity = 1.0 if any(g in song_genres for g in user_val) else 0.0
                else:
                    similarity = 1.0 if user_val in song_genres else 0.0
            else:
                similarity = 1.0 if song[feature] == user_val else 0.0
        elif feature == "tempo_bpm":
            user_norm = _normalize_tempo(user_val)
            song_norm = _normalize_tempo(song[feature])
            similarity = 1.0 - abs(user_norm - song_norm)
        else:
            similarity = 1.0 - abs(user_val - song[feature])

        total_score += weight * similarity
        total_weight += weight

    if total_weight == 0.0:
        return 0.0

    return total_score / total_weight


def recommend_songs(user_prefs: Dict, songs: List[Dict], k: int = 5) -> List[Tuple[Dict, float, str]]:
    """Score all songs, sort descending, return top k as (song, score, explanation)."""
    scored = [
        (song, score_song(user_prefs, song), _build_explanation(user_prefs, song))
        for song in songs
    ]
    scored.sort(key=lambda x: x[1], reverse=True)
    return scored[:k]


# ---------------------------------------------------------------------------
# OOP path (dataclass-based, used by tests)
# ---------------------------------------------------------------------------

class Recommender:
    """OOP wrapper that delegates scoring to the functional score_song()."""

    def __init__(self, songs: List[Song]):
        self.songs = songs

    def recommend(self, user: UserProfile, k: int = 5) -> List[Song]:
        """Return the top k songs for a user, sorted by score descending."""
        prefs = self._profile_to_prefs(user)
        scored = [
            (song, score_song(prefs, asdict(song)))
            for song in self.songs
        ]
        scored.sort(key=lambda x: x[1], reverse=True)
        return [song for song, _ in scored[:k]]

    def explain_recommendation(self, user: UserProfile, song: Song) -> str:
        """Explain why a song was recommended for a user."""
        prefs = self._profile_to_prefs(user)
        return _build_explanation(prefs, asdict(song))

    @staticmethod
    def _profile_to_prefs(user: UserProfile) -> Dict:
        """Convert a UserProfile dataclass to a dict compatible with score_song."""
        prefs = {
            "genre": user.favorite_genre,
            "mood": user.favorite_mood,
            "energy": user.target_energy,
            "acousticness": 0.8 if user.likes_acoustic else 0.2,
        }
        if user.target_valence is not None:
            prefs["valence"] = user.target_valence
        if user.target_danceability is not None:
            prefs["danceability"] = user.target_danceability
        if user.target_tempo_bpm is not None:
            prefs["tempo_bpm"] = user.target_tempo_bpm
        return prefs

# Music Recommender Simulation

## Project Overview
A content-based music recommender system for CodePath AI110 Module 3. Scores songs against a user taste profile using tier-weighted feature similarity.

## Architecture

### Two execution paths (both must work):
- **Functional**: `src/main.py` -> `load_songs()`, `recommend_songs()`, `score_song()` (dict-based)
- **OOP**: `tests/` -> `Recommender` class with `Song` / `UserProfile` dataclasses

The OOP path delegates scoring to the functional `score_song()` to avoid duplication.

### Key files:
- `src/recommender.py` — All recommendation logic: data classes, scoring, loading, explanation
- `src/main.py` — CLI runner, imports from `recommender` (no `src.` prefix)
- `tests/test_recommender.py` — Pytest suite, imports from `src.recommender`
- `data/songs.csv` — 18 songs, 7 features (genre, mood, energy, tempo_bpm, valence, danceability, acousticness)

### Import convention:
- `main.py` uses `from recommender import ...` (run via `python -m src.main`)
- Tests use `from src.recommender import ...`

## Scoring Design

### Formula:
```
score = sum(weight[f] * similarity[f]) / sum(weight[f])  for active features
```
Normalized to [0, 1]. Only features present in user prefs are scored.

### Feature tiers & default weights:
| Tier | Features | Weights |
|------|----------|---------|
| 1 (vibe) | genre (3.0), mood (3.0), energy (2.5) | Highest |
| 2 (support) | acousticness (1.5), valence (1.5) | Medium |
| 3 (tiebreaker) | danceability (0.75), tempo_bpm (0.5) | Lowest |

### Similarity functions:
- Categorical (genre, mood): 1.0 exact match, 0.0 otherwise
- Numeric 0-1 (energy, valence, danceability, acousticness): `1 - abs(user - song)`
- tempo_bpm: normalize via `(bpm - 60) / (200 - 60)` clamped to [0,1], then same formula
- likes_acoustic (bool): converted to numeric target (True -> 0.8, False -> 0.2)

## Commands
```bash
python -m src.main          # Run recommender
pytest                       # Run tests
pytest -v                    # Run tests verbose
pytest tests/test_recommender.py::TestScoreSong  # Run specific test class
```

## Data
- 18 songs in `data/songs.csv` (base project)
- Genres: pop, lofi, rock, ambient, jazz, synthwave, indie pop, country, electronic, r&b, metal, bossa nova, chiptune, folk, house
- Moods: happy, chill, intense, relaxed, focused, moody, nostalgic, energetic, romantic, aggressive, playful, melancholy
- All numeric features 0-1 except tempo_bpm (60-168)

---

## VibeFinder AI — Evolution (Module 4 Final Project)

### What Changed from the Base Project
The base project was a pure algorithmic recommender with hardcoded dict profiles and no LLM. The final project wraps it in a RAG pipeline: the user speaks in natural language, an LLM parses intent into structured prefs, the existing scoring engine retrieves songs, and a second LLM call generates a narrative response. The core scoring logic in `recommender.py` is **unchanged** — only new layers were added around it.

### New Files
| File | Purpose |
|---|---|
| `src/llm.py` | Two Bedrock calls: `parse_user_intent()` and `generate_response()` |
| `src/chat.py` | Interactive CLI loop — new primary entrypoint |
| `logs/.gitkeep` | Keeps `logs/` in git; `logs/llm_calls.log` written at runtime, git-ignored |
| `tests/test_llm.py` | 26 unit tests for LLM layer (all mocked, no live Bedrock calls) |
| `pytest.ini` | `pythonpath = .` so bare `pytest` resolves `src.*` imports |

### Updated Commands
```bash
python -m src.chat          # AI chat loop (primary demo entrypoint)
python -m src.main          # Original batch runner (unchanged)
pytest                      # All unit tests (74 total, no Bedrock calls)
pytest -m integration       # Live end-to-end Bedrock test
```

### RAG Pipeline (how a request flows)
```
User free text
    ↓
[LLM #1: parse_user_intent()]   → structured prefs dict + optional _k count
    ↓
[recommend_songs()]              → existing scoring, unchanged
    ↓
[Guardrail: top score < 0.5?]   → flags low-confidence match
    ↓
[LLM #2: generate_response()]   → numbered narrative using retrieved songs as context
    ↓
Logged to logs/llm_calls.log
```

### Bedrock Config
- **Model:** `us.anthropic.claude-sonnet-4-6`
- **Region:** `us-east-1`
- **Auth:** root account credentials (`aws configure`) or `scapel-dev` IAM user (has `AmazonBedrockFullAccess`)
- **Required packages:** `boto3` + `botocore[crt]` (the crt extra is required for the credential provider chain)

### Multi-Tag Genre System
Songs have 3 pipe-separated genre tags instead of one string (e.g., `bachata|latin pop|spanish`).

**Why it was changed:** A single genre was too rigid. "La Bachata" by Manuel Turizo was tagged `latin pop` but users naturally search for "bachata". Tags solve cross-genre discoverability.

**3-layer tagging structure per song:**
1. Specific genre (most precise label, e.g. `bachata`)
2. Genre family (broader bucket, e.g. `latin pop`)
3. Cultural/language tag for non-English songs (`spanish`, `japanese`, `mandarin`, `cantonese`, `korean`, `french`, `latin`); or a related style tag for English songs

**Code changes:**
- `load_songs()` splits `genre` on `|` → stored as `list[str]`
- `Song.genre` changed from `str` to `List[str]`
- `score_song()` uses `user_val in song_genres` (membership) instead of `==` (equality)
- `_build_explanation()` updated to match

**Catalog after expansion:** 210 songs, 47 genres, 42 moods

### Multi-Genre Query Feature
`user_prefs["genre"]` can be a string (single genre) **or** a list (multi-genre query).

**How it works:**
- LLM returns `"genre": ["reggaeton", "bachata"]` when user requests multiple genres
- `score_song()` uses `any(g in song_genres for g in user_val)` — 1.0 if ANY requested genre matches any of the song's tags

**Why "any match = 1.0" and not partial credit:** A user asking for "reggaeton and bachata" wants songs from either genre, not songs that ARE both simultaneously. A pure reggaeton song deserves full genre credit when the user's query includes reggaeton. Partial credit would incorrectly penalize genre-pure songs in a multi-genre search.

**Validation:** `_validate_prefs()` filters invalid genres from the list; a single-valid-entry list collapses to a plain string for consistency.

### Requested Song Count (k extraction)
`parse_user_intent()` extracts an explicit count from the user's message (e.g., "give me 15 songs") and returns it as `_k` in the prefs dict. `chat.py` pops `_k` before passing prefs to `recommend_songs()`. Clamped to [1, 50].

### Guardrails & Logging
- Top score < 0.5 → `generate_response()` receives a `low_confidence=True` flag, which injects a note into the LLM prompt asking it to mention the limited match gently
- Empty prefs after parsing → system asks user to rephrase before making a second Bedrock call
- Bedrock `ClientError` → caught and printed as a clean error message (no crash)
- Every `_call_bedrock()` invocation writes input + response to `logs/llm_calls.log`

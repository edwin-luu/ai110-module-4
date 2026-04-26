# VibeFinder AI

**Original project:** VibeFinder 1.0 (CodePath AI110 Module 3) — a pure content-based music recommender that scored songs against hardcoded user profiles using tier-weighted similarity. It returned ranked results with plain-English explanations but had no natural language input, no LLM, and a fixed 18-song catalog.

**Final project:** VibeFinder AI wraps that scoring engine in a RAG pipeline. Users describe what they want in plain English, a Claude LLM parses the intent into structured preferences, the original recommender retrieves the best matches from a 210-song catalog, and a second LLM call generates a conversational response. The core scoring logic is unchanged — new AI layers were added around it.

---

## How the System Works

### RAG Pipeline

```mermaid
flowchart TD
    A[User free text] --> B[LLM: parse_user_intent]
    B -->|structured prefs dict| C{prefs empty?}
    C -->|yes| D[Ask user to rephrase]
    C -->|no| E[recommend_songs — existing scoring engine]
    E -->|top-k songs + scores| F{top score < 0.5?}
    F -->|yes| G[generate_response with low_confidence flag]
    F -->|no| G
    G --> H[Numbered narrative response to user]
    B --> I[logs/llm_calls.log]
    G --> I
```

**Step 1 — Parse:** Claude receives the user's free text and outputs a JSON object with structured music preferences (`genre`, `mood`, `energy`, `acousticness`, etc.) and an optional `k` (number of songs requested). The output is validated and sanitized before use.

**Step 2 — Retrieve:** The existing tier-weighted scoring engine scores all 210 songs against the parsed preferences and returns the top-k ranked results. No LLM is involved here — this is the original Module 3 algorithm.

**Step 3 — Generate:** Claude receives the user's original request plus the ranked songs and writes a numbered, conversational response explaining why each song fits.

### Scoring Engine (unchanged from base project)

Each song is scored by computing weighted similarity across only the features the user specified:

```
score = sum(weight[f] * similarity(user[f], song[f])) / sum(weight[f])
```

| Tier | Features | Weights |
|---|---|---|
| 1 (vibe) | genre (3.0), mood (3.0), energy (2.5) | Highest |
| 2 (support) | acousticness (1.5), valence (1.5) | Medium |
| 3 (tiebreaker) | danceability (0.75), tempo_bpm (0.5) | Lowest |

### Multi-Tag Genre System

Every song carries 3 pipe-separated genre tags (e.g., `bachata|latin pop|spanish`) instead of a single genre string. This enables natural queries like "spanish songs", "japanese music", or "reggaeton and bachata" to surface songs that were previously unreachable.

**3-layer structure:**
1. **Specific genre** — most precise label (`bachata`, `reggaeton`)
2. **Genre family** — broader bucket (`latin pop`)
3. **Cultural/language tag** — for non-English songs (`spanish`, `japanese`, `mandarin`, `korean`, etc.); related style for English songs

**Scoring with multi-genre queries:** When a user asks for "reggaeton and bachata", the LLM returns `"genre": ["reggaeton", "bachata"]`. A song scores 1.0 on genre if *any* of its tags matches *any* of the requested genres. This means a pure bachata song correctly surfaces in a reggaeton+bachata search without requiring it to literally be both.

### Guardrails & Logging

| Guardrail | Behavior |
|---|---|
| Top score < 0.5 | LLM is told to gently note the catalog has limited options |
| Empty prefs after parsing | System asks user to rephrase — no Bedrock call wasted |
| Invalid genre/mood from LLM | Dropped silently; scoring continues on remaining features |
| Numeric values out of range | Clamped to valid range (energy 0–1, tempo 60–200) |
| Bedrock API error | Caught, printed as clean message, no crash |

All LLM calls logged to `logs/llm_calls.log` with timestamp, input, and raw response.

---

## Sample Interactions

**1 — Genre + mood request:**
```
You: I want something chill to code to late at night.

VibeFinder: Of course! Here are some perfect late-night coding tracks:

1. "Midnight Coding" by LoRoom — A lofi chill track with exactly the low energy and
   high acousticness you'd want for a focused, quiet session.
2. "Focus Flow" by LoRoom — Another lofi gem tuned for concentration, with a calm
   focused mood and minimal distraction.
...
```

**2 — Multi-genre request with count:**
```
You: Give me a list of 12 reggaeton and bachata songs.

VibeFinder: Sure! Here are 12 great reggaeton and bachata tracks:

1. "Dile al Amor" by Aventura — A deeply romantic bachata gem with the classic
   soulful guitar-driven sound Aventura is known for.
2. "Si No Le Contesto" by Plan B — A classic romantic reggaeton track with smooth,
   laid-back vibes perfect for a passionate mood.
...
```

**3 — Cultural/language query:**
```
You: I want 10 Chinese songs.

VibeFinder: Sure! Here are 10 Chinese songs you might enjoy:

1. "青花瓷" by Jay Chou — A beautifully calm Mandopop classic with folk influences,
   painting vivid imagery of ancient Chinese culture through its poetic lyrics.
2. "月亮代表我的心" by Teresa Teng — A timeless romantic classic that remains one
   of the most beloved Mandopop songs of all time.
...
```

---

## Getting Started

### Prerequisites

- Python 3.12+
- AWS account with Bedrock access enabled in `us-east-1`
- IAM user or role with `AmazonBedrockFullAccess`
- AWS credentials configured locally (`aws configure`)

### Setup

1. Create and activate a virtual environment:

   ```bash
   python -m venv venv
   venv\Scripts\activate        # Windows
   source venv/bin/activate     # Mac/Linux
   ```

2. Install dependencies:

   ```bash
   pip install -r requirements.txt
   ```

3. Verify your AWS credentials work:

   ```bash
   aws sts get-caller-identity
   ```

### Running the App

```bash
python -m src.chat      # AI chat loop (primary interface)
python -m src.main      # Original batch runner with hardcoded profiles
```

### Running Tests

```bash
pytest                          # All 74 unit tests (no Bedrock calls)
pytest -m integration           # End-to-end live Bedrock test
pytest tests/test_llm.py -v     # LLM layer tests only
pytest tests/test_recommender.py -v   # Scoring engine tests only
```

---

## Architecture Overview

```
src/
  recommender.py   — Scoring engine, data classes (unchanged from base)
  main.py          — Batch CLI runner with hardcoded profiles (unchanged)
  llm.py           — Bedrock LLM calls: parse_user_intent(), generate_response()
  chat.py          — Interactive loop, entrypoint for the AI demo

tests/
  test_recommender.py  — 45 scoring engine tests (all pass)
  test_llm.py          — 26 LLM layer tests (all mocked, no Bedrock calls)

data/
  songs.csv        — 210 songs, 3 genre tags each, 7 audio features

logs/
  llm_calls.log    — Runtime log of every Bedrock call (git-ignored)
```

**Component responsibilities:**
- `recommender.py` is the retrieval layer — pure Python, no AI
- `llm.py` is the AI layer — wraps Bedrock, validates outputs, logs calls
- `chat.py` is the orchestrator — connects user input to both layers and enforces guardrails

---

## Design Decisions

**Why RAG over a pure LLM approach?**
Asking Claude to recommend songs directly would require embedding the full catalog in every prompt (expensive and brittle). Instead, the catalog is queried algorithmically and only the relevant results are passed to the LLM for narration. The LLM never has to "know" all 210 songs — it just explains the ones the retriever found.

**Why AWS Bedrock instead of the Anthropic API?**
The project uses $100 in AWS Bedrock credits rather than separate Anthropic API credits. Bedrock provides the same Claude models via `boto3` — the only code difference is the client and request format.

**Why pipe-separated genre tags instead of a separate tags column?**
Keeping a single `genre` column that supports both `"pop"` and `"pop|indie pop|electronic"` preserves backward compatibility with `load_songs()` and the scoring engine. The split happens at load time, and `score_song()` uses a defensive `isinstance` check so both formats always work.

**Why "any match = 1.0" for multi-genre scoring?**
When a user asks for "reggaeton and bachata", they want songs from *either* genre — not songs that are literally both. Partial credit would penalize a great pure-bachata song just because it isn't also reggaeton. Full credit for any matching tag produces the most intuitive mixed results.

**Why keep the base scoring engine unchanged?**
The tier-weighted scoring already handles the recommendation quality problem well. Adding LLM layers on top (rather than replacing the scorer) lets the system benefit from both: fast, explainable algorithmic ranking plus natural language I/O.

---

## Testing Summary

| Test suite | Count | Coverage |
|---|---|---|
| `test_recommender.py` | 45 | Scoring math, tier weights, tempo normalization, edge cases, explanation generation, CSV loading |
| `test_llm.py` | 26+3 | `_validate_prefs` edge cases, JSON parsing, markdown fence stripping, invalid genre/mood filtering, multi-genre list validation, guardrail threshold, response format |
| Integration (`-m integration`) | 1 | Full live Bedrock round-trip: free text → prefs → songs → narrative |

**What worked:** The LLM reliably extracts genre and mood from natural language. Multi-genre queries (`["reggaeton", "bachata"]`) surface songs from both pools. Cultural tags (`spanish`, `japanese`, `mandarin`) make non-English catalogs fully discoverable.

**What the AI struggled with:** Vague single-word requests like "spanish songs" were initially returning empty prefs because "spanish" wasn't a recognized genre. Adding it as a cultural tag (mapped to actual songs via the tagging system) resolved this. The LLM also occasionally wraps JSON in markdown fences despite the system prompt saying not to — the response cleaner strips them defensively.

**What surprised me:** The scoring engine produces very different top-5 lists for "reggaeton" vs "reggaeton and bachata" — adding bachata to the query doesn't just add bachata songs, it changes the relative ranking because bachata songs now compete on equal genre footing. This was only visible after running the full pipeline, not during unit testing.

---

## Video Walkthrough

*[Loom link — to be added before submission]*

---

## Reflection and Ethics

**Limitations and biases:**
The tier-weighted scoring still carries all the biases from the base project — genre dominance, categorical rigidity (no partial genre credit), and filter-bubble behavior. The LLM layer adds a new bias: Claude will generate enthusiastic-sounding responses even when the catalog match is poor (score < 0.5). The low-confidence guardrail partially addresses this, but a 0.4-score recommendation still gets presented positively.

**Could this be misused?**
The system recommends from a fixed local catalog — it cannot access external music or user data, so the misuse surface is low. The main risk is the LLM generating hallucinated song details (wrong artist, wrong genre description) if the context passed to it is ambiguous. Mitigation: all song metadata in the generation prompt comes directly from the catalog, not from the LLM's training data.

**What surprised me during testing:**
The multi-genre scoring change had an unexpected side effect: songs with broad tag overlap (e.g., `pop|indie pop|electronic`) suddenly appeared in searches they never did before, which sometimes felt like noise. The three-tag structure keeps this in check — songs don't accumulate enough tags to match unrelated queries.

**Collaboration with AI:**
Claude was most helpful structuring the Bedrock integration — specifically suggesting the `converse()` API over `invoke_model()` and the defensive `isinstance` check for the mixed string/list genre format. One instance where its suggestion was flawed: it initially proposed using `logging.basicConfig()` at module import time, which is a no-op when pytest has already configured the logging system. The fix was to call it once at app startup instead.

---

## Experiments from Base Project

### Experiment 1: Weight Shift — Double Energy, Halve Genre

Changed `genre` weight from 3.0 to 1.5 and `energy` weight from 2.5 to 5.0.

**Results**: For the "Deep Intense Rock" profile, "Gym Hero" (pop, intense, energy=0.93) jumped from #3 to #2, overtaking "Iron Anthem" (metal, aggressive, energy=0.97). Genre loyalty dropped; physical energy accuracy improved.

**Takeaway**: Genre weight acts as a coarse filter. Lowering it starts cross-pollinating genres.

### Experiment 2: Diverse Profile Testing

| Profile | Top Pick | Surprising? |
|---|---|---|
| High-Energy Pop | Sunrise City (0.99) | No — perfect match |
| Chill Lofi | Library Rain (0.99) | No — perfect match |
| Deep Intense Rock | Storm Runner (0.99) | No — perfect match |
| Conflicting: lofi + energy 0.95 | Midnight Coding (0.84) | Yes — genre/mood loyalty beat energy |
| Missing Genre: classical | Bossa Nova Sunset (0.64) | Yes — decent fallback via mood |
| Numeric Only (no genre/mood) | Desert Highway (0.94) | Yes — country won on pure audio similarity |

---

See [model_card.md](model_card.md) for deeper analysis of the base recommender's biases and limitations.

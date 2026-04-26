# Model Card: VibeFinder AI

## 1. Model Name

**VibeFinder AI** — A RAG-based music recommender combining tier-weighted content similarity with LLM-powered natural language I/O.

**Evolution from base project:** VibeFinder 1.0 (Module 3) was a pure algorithmic recommender with hardcoded dict profiles and no LLM. VibeFinder AI adds a natural language frontend: a Claude LLM parses the user's free text into structured preferences, the original scoring engine retrieves matches from a 210-song catalog, and a second LLM call generates a conversational response. The core scoring logic is unchanged.

---

## 2. Intended Use

This system allows users to describe what music they want in plain English and receive ranked recommendations with natural language explanations. It is designed for classroom exploration in CodePath AI110, not for real users or production deployment.

**Non-intended use:** This system should not be used for real music streaming, commercial recommendation, or any context where users expect broad, current, or personalized suggestions. It has a fixed 210-song catalog, no user feedback loop, and no collaborative data. Using it as a real product would create filter bubbles and exclude listeners of underrepresented genres.

---

## 3. How the Model Works

### RAG Pipeline

User free text flows through three stages:

1. **Parse** — Claude (via AWS Bedrock) reads the user's message and outputs a structured JSON object with music preferences (`genre`, `mood`, `energy`, etc.) and an optional song count (`k`). The output is validated and sanitized before use.
2. **Retrieve** — The tier-weighted scoring engine scores all 210 songs against the parsed preferences and returns the top-k results. No LLM is involved here.
3. **Generate** — Claude receives the user's original request plus the ranked songs and writes a numbered, conversational response explaining why each song fits.

### Scoring Formula

```
score = sum(weight[f] * similarity(user[f], song[f])) / sum(weight[f])
```

Only features the user specified contribute to the score. Feature tiers and weights:

| Tier | Features | Weights |
|---|---|---|
| 1 (vibe) | genre (3.0), mood (3.0), energy (2.5) | Highest |
| 2 (support) | acousticness (1.5), valence (1.5) | Medium |
| 3 (tiebreaker) | danceability (0.75), tempo_bpm (0.5) | Lowest |

Categorical features (genre, mood) are binary: 1.0 for a match, 0.0 otherwise. Numeric features use `1 - abs(user - song)`. Tempo is normalized to [0, 1] before comparison.

### Multi-Tag Genre System

Every song has 3 pipe-separated genre tags (e.g., `bachata|latin pop|spanish`). A song scores 1.0 on genre if *any* of its tags matches the user's requested genre(s). For multi-genre queries (e.g., "reggaeton and bachata"), the LLM returns a list and the scorer uses `any()` matching — a pure bachata song gets full credit in a reggaeton+bachata search.

---

## 4. Data

- **210 songs** in `data/songs.csv`
- **47 genres**, **42 moods**, 7 audio features per song
- Genre tags follow a 3-layer structure: specific genre → genre family → cultural/language tag (e.g., `bachata|latin pop|spanish`, `j-pop|pop|japanese`)
- Non-English songs include cultural tags (`spanish`, `french`, `japanese`, `mandarin`, `cantonese`, `korean`, `latin`) for natural language discoverability
- The catalog was manually assembled — it reflects one person's idea of genre diversity, not a systematic survey of global music consumption
- Missing or underrepresented: classical, blues, gospel, afrobeats, and many regional genres

---

## 5. Strengths

- **Natural language input.** Users describe what they want conversationally ("something chill to code to late at night") and get structured, relevant results — no form-filling required.
- **Transparent retrieval.** Every recommendation comes with a score and an explanation ("genre match; energy similarity 98%"), making it easy to understand why a song was picked.
- **Multi-genre discoverability.** The 3-tag system and multi-genre query support allow cross-genre searches that a single-tag system would miss entirely.
- **Graceful partial input.** If the user specifies only genre, only genre contributes to the score. The system never crashes on incomplete preferences.
- **No hallucinated metadata.** All song details in the generation prompt come directly from the catalog, not from the LLM's training data, preventing fabricated artist or genre descriptions.

---

## 6. Limitations and Bias

- **Genre dominance bias.** Genre weight (3.0) is so high that a genre mismatch is nearly impossible to overcome. The system effectively filters by genre first and uses other features only as tiebreakers within the same genre.
- **Categorical rigidity.** "Indie pop" and "pop" get 0% similarity. "Rock" and "metal" are treated as unrelated. Real genre relationships are gradients, not binary.
- **No balanced multi-genre splits.** Asking for "5 Spanish and 5 Chinese songs" returns the top 10 across both genres — secondary features determine ranking, which can produce an uneven split (e.g., 6/4). There is no logic to enforce proportional representation.
- **No conversation memory.** Each request is fully independent. The system cannot reference prior turns or build on user feedback within a session.
- **Fixed local catalog.** Recommendations are limited to 210 songs. The system cannot discover new music or follow trends.
- **LLM confidence bias.** Even when the top match score is below 0.5, Claude generates an enthusiastic-sounding response. The low-confidence guardrail adds a disclaimer but does not suppress the positive framing.
- **Conflicting preferences are mishandled.** A user who asks for "lofi + very high energy" gets low-energy lofi songs because genre+mood loyalty (weight 6.0) overwhelms the energy signal (weight 2.5). The system does not warn the user that their preferences are contradictory.

---

## 7. Evaluation

**Query types tested:**

| Query type | Example | Result |
|---|---|---|
| Genre + mood | "chill music to code to" | Top results matched genre and mood; scores 0.85+ |
| Multi-genre | "reggaeton and bachata songs" | Songs from both genres surfaced correctly |
| Cultural/language | "10 Chinese songs" | Mandopop and Cantopop songs returned via cultural tags |
| Mixed language | "5 Spanish and 5 Chinese" | Correct genres returned but split was 6/4, not 5/5 |
| Low-confidence | "something classical" | No classical in catalog; disclaimer triggered at score < 0.5 |
| Count extraction | "give me 15 songs" | Correct k=15 parsed and passed to recommender |

**Findings:**
- The LLM reliably extracts genre and mood from natural language, including informal descriptions like "vibing late at night."
- Multi-genre queries correctly surface songs from both requested genre pools.
- Cultural tags (`spanish`, `japanese`, `mandarin`) make non-English catalogs fully discoverable — before adding these, "Spanish songs" returned empty results.
- The scorer produces meaningfully different rankings for "reggaeton" vs. "reggaeton and bachata" — adding the second genre changes relative ranking, not just the pool.

**What the LLM struggled with:**
- Occasionally wraps JSON in markdown fences despite the system prompt forbidding it — handled defensively by stripping them.
- Vague requests like "just play something" return empty prefs, triggering the rephrase guardrail.

**Automated tests:** 74 pytest cases — 45 for the scoring engine (math, tier weights, tempo normalization, multi-tag matching, edge cases) and 29 for the LLM layer (all mocked, no live Bedrock calls).

---

## 8. Future Work

- **Genre embeddings or hierarchy.** Instead of binary match/mismatch, use a similarity matrix (e.g., rock/metal = 0.7, pop/indie pop = 0.8) so related genres partially match.
- **Balanced multi-genre retrieval.** Run separate `recommend_songs()` calls per genre when the user specifies a split (e.g., "5 Spanish and 5 Chinese"), then merge.
- **Conversation memory.** Store prior turns so the user can follow up with "give me more like #3" or "now make them slower."
- **Conflict detection.** Warn users when their preferences are contradictory (e.g., "lofi songs rarely exceed energy 0.7").
- **Larger and curated catalog.** Scale to 1,000+ songs with systematic genre coverage, not manual assembly.
- **Diversity injection.** After picking the top matches, intentionally include 1–2 songs from different genres or moods to reduce filter-bubble behavior.

---

## 9. Personal Reflection

**Base project takeaway (VibeFinder 1.0):**
The most surprising thing was how much the *weights* matter compared to the *features*. Adding danceability and tempo barely changed any ranking because their low weights made them nearly irrelevant. But shifting genre weight by 50% completely reshuffled results. In real recommender systems, the weight tuning decisions are where human bias enters the system — and those decisions are usually invisible to users.

**Final project takeaway (VibeFinder AI):**
Adding the LLM layer changed what failures look like. The base project failed silently — bad results just had lower scores. The AI version fails conversationally — a poor catalog match still gets a confident, articulate response. The low-confidence guardrail helps, but it exposed a fundamental tension: users expect a natural language system to "understand" them, not just retrieve the least-bad option. Managing that expectation gap is harder than fixing the retrieval logic.

The multi-tag genre system also produced a non-obvious emergent behavior: songs with broad tag overlap suddenly appeared in searches they hadn't before. The three-tag cap kept this in check, but it showed that catalog design decisions (how many tags per song, what tags to use) have as much impact on recommendation quality as the scoring algorithm itself.

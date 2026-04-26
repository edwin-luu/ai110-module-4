import json
import logging
import os
from botocore.exceptions import ClientError
import boto3

MODEL_ID = "us.anthropic.claude-sonnet-4-6"
REGION = "us-east-1"

VALID_GENRES = {
    "alt pop", "alt rock", "ambient", "bachata", "bossa nova", "cantopop",
    "chanson", "chiptune", "city pop", "country", "darkwave", "edm",
    "electronic", "folk", "hip-hop", "house", "indie", "indie pop",
    "indie rock", "j-pop", "jazz", "k-pop", "latin pop", "latin trap",
    "lofi", "mandarin folk", "mandopop", "metal", "neo-classical", "phonk",
    "pop", "pop punk", "pop rap", "post-hardcore", "psychedelic", "punk",
    "r&b", "rap", "reggae", "reggaeton", "rnb", "rock", "soft rock",
    "soul", "spanish pop", "synthwave", "uk drill",
    # cultural / language tags
    "cantonese", "french", "japanese", "korean", "latin", "mandarin", "spanish",
    # style tag
    "trap",
}
VALID_MOODS = {
    "aggressive", "angsty", "bittersweet", "bright", "calm", "catchy",
    "chill", "confident", "dance", "dark", "dramatic", "emotional",
    "energetic", "epic", "feel-good", "focused", "groovy", "happy",
    "inspiring", "intense", "laid-back", "lighthearted", "melancholic",
    "melancholy", "minimal", "moody", "nostalgic", "party", "playful",
    "positive", "quirky", "rebellious", "reflective", "relaxed", "romantic",
    "sad", "seductive", "smooth", "soft", "sweet", "upbeat", "uplifting",
}

os.makedirs("logs", exist_ok=True)
logging.basicConfig(
    filename="logs/llm_calls.log",
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)

_client = None


def _get_client():
    global _client
    if _client is None:
        _client = boto3.client("bedrock-runtime", region_name=REGION)
    return _client


def _call_bedrock(system_prompt: str, user_message: str, max_tokens: int = 512) -> str:
    logging.info("CALL | user_message=%r", user_message[:200])
    try:
        response = _get_client().converse(
            modelId=MODEL_ID,
            system=[{"text": system_prompt}],
            messages=[{"role": "user", "content": [{"text": user_message}]}],
            inferenceConfig={"maxTokens": max_tokens},
        )
        text = response["output"]["message"]["content"][0]["text"]
        logging.info("RESPONSE | %r", text[:500])
        return text
    except ClientError as e:
        logging.error("Bedrock ClientError: %s", e)
        raise


# ---------------------------------------------------------------------------
# parse_user_intent
# ---------------------------------------------------------------------------

_PARSE_SYSTEM = """\
You are a music preference parser. Convert a user's free-text description into a structured JSON object.

Available features and valid values:
- genre (string OR array): single genre → string; multiple genres (e.g. "reggaeton and bachata", "rock or indie") → array like ["reggaeton", "bachata"]. Valid values: alt pop, alt rock, ambient, bachata, "bossa nova", cantopop, chanson, chiptune, "city pop", country, darkwave, edm, electronic, folk, hip-hop, house, indie, "indie pop", "indie rock", j-pop, jazz, k-pop, "latin pop", "latin trap", lofi, "mandarin folk", mandopop, metal, "neo-classical", phonk, pop, "pop punk", "pop rap", "post-hardcore", psychedelic, punk, r&b, rap, reggae, reggaeton, rnb, rock, "soft rock", soul, "spanish pop", synthwave, "uk drill", trap, latin, spanish, japanese, mandarin, cantonese, korean, french
- mood (string): aggressive, angsty, bittersweet, bright, calm, catchy, chill, confident, dance, dark, dramatic, emotional, energetic, epic, feel-good, focused, groovy, happy, inspiring, intense, laid-back, lighthearted, melancholic, melancholy, minimal, moody, nostalgic, party, playful, positive, quirky, rebellious, reflective, relaxed, romantic, sad, seductive, smooth, soft, sweet, upbeat, uplifting
- energy (float 0-1): 0.0 = very calm, 1.0 = very intense
- acousticness (float 0-1): 0.0 = fully electronic, 1.0 = fully acoustic
- valence (float 0-1): 0.0 = sad/dark, 1.0 = happy/bright
- danceability (float 0-1): 0.0 = not danceable, 1.0 = very danceable
- tempo_bpm (float 60-200): 60 = slow, 200 = very fast

- k (integer, optional): number of songs explicitly requested (e.g. "give me 15", "top 10", "a list of 20"). Omit if not mentioned.

Rules:
- Only include features you can confidently infer.
- Output ONLY valid JSON — no markdown, no code fences, no explanation.
- Use only the exact genre/mood values listed above.
- If nothing can be inferred, output {}.

Example single genre: {"genre": "lofi", "mood": "focused", "energy": 0.35, "acousticness": 0.7}
Example multi-genre: {"genre": ["reggaeton", "bachata"], "k": 10}
Example with count: {"genre": "rap", "energy": 0.85, "k": 15}"""


def _validate_prefs(raw: dict) -> dict:
    clean = {}
    if "genre" in raw:
        raw_genre = raw["genre"]
        if isinstance(raw_genre, list):
            valid = [g for g in raw_genre if g in VALID_GENRES]
            if len(valid) > 1:
                clean["genre"] = valid
            elif len(valid) == 1:
                clean["genre"] = valid[0]
        elif raw_genre in VALID_GENRES:
            clean["genre"] = raw_genre
    if "mood" in raw and raw["mood"] in VALID_MOODS:
        clean["mood"] = raw["mood"]
    for feature in ("energy", "acousticness", "valence", "danceability"):
        if feature in raw:
            try:
                clean[feature] = max(0.0, min(1.0, float(raw[feature])))
            except (TypeError, ValueError):
                pass
    if "tempo_bpm" in raw:
        try:
            clean["tempo_bpm"] = max(60.0, min(200.0, float(raw["tempo_bpm"])))
        except (TypeError, ValueError):
            pass
    return clean


def parse_user_intent(text: str) -> dict:
    """Convert free-text input to a structured prefs dict via LLM.

    Returns a prefs dict compatible with score_song(). If the user requested
    a specific song count, it is included as '_k' (int) and must be popped
    by the caller before passing prefs to the recommender.
    """
    raw_response = _call_bedrock(_PARSE_SYSTEM, text, max_tokens=256)
    cleaned = (
        raw_response.strip()
        .removeprefix("```json")
        .removeprefix("```")
        .removesuffix("```")
        .strip()
    )
    try:
        raw = json.loads(cleaned)
    except json.JSONDecodeError:
        logging.warning("parse_user_intent: invalid JSON — %r", raw_response[:200])
        return {}

    # Extract k before feature validation — it is not a scoring feature.
    k_raw = raw.pop("k", None)
    prefs = _validate_prefs(raw)
    if k_raw is not None:
        try:
            prefs["_k"] = max(1, min(50, int(k_raw)))
        except (TypeError, ValueError):
            pass

    logging.info("parse_user_intent: %r -> %r", text[:100], prefs)
    return prefs


# ---------------------------------------------------------------------------
# generate_response
# ---------------------------------------------------------------------------

_GENERATE_SYSTEM = """\
You are a friendly music recommendation assistant. Given a user's request and a ranked list of matching songs, respond in this exact structure:

1. One conversational opening sentence that directly answers the request, starting with something like "Of course, here are some..." or "Sure! Here are..." — keep it warm and natural.
2. A numbered list of every song provided, in order. For each song use this format:
   N. **"Title" by Artist** — one sentence explaining specifically why this song fits what the user asked for, referencing its genre, mood, energy, or other relevant qualities.
3. One brief closing line (optional) inviting them to ask for more.

Do not write paragraphs of flowing prose. The list must be easy to scan."""


def generate_response(user_text: str, results: list, low_confidence: bool = False) -> str:
    """Generate a natural language recommendation narrative.

    results: list of (song_dict, score, explanation) tuples
    """
    songs_context = "\n".join(
        f'{i+1}. "{s["title"]}" by {s["artist"]} '
        f"(genre: {s['genre']}, mood: {s['mood']}, score: {score:.2f}) — {explanation}"
        for i, (s, score, explanation) in enumerate(results)
    )
    confidence_note = (
        "\nNote: The best match score is below 0.5 — the catalog has limited options "
        "for this request. Mention this gently."
        if low_confidence
        else ""
    )
    user_message = (
        f'User request: "{user_text}"\n\n'
        f"Top recommendations:{confidence_note}\n{songs_context}"
    )
    # ~80 tokens per song entry + 100 for intro/closing, minimum 400
    max_tokens = max(400, len(results) * 80 + 100)
    return _call_bedrock(_GENERATE_SYSTEM, user_message, max_tokens=max_tokens)

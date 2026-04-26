"""Tests for the LLM layer (parse_user_intent, generate_response, guardrails)."""

import pytest
from unittest.mock import patch
from src.llm import parse_user_intent, generate_response, _validate_prefs, VALID_GENRES, VALID_MOODS


# ---------------------------------------------------------------------------
# _validate_prefs — no Bedrock calls needed
# ---------------------------------------------------------------------------

class TestValidatePrefs:

    def test_valid_genre_and_mood_pass_through(self):
        raw = {"genre": "lofi", "mood": "chill", "energy": 0.4}
        result = _validate_prefs(raw)
        assert result == {"genre": "lofi", "mood": "chill", "energy": 0.4}

    def test_unknown_genre_is_dropped(self):
        raw = {"genre": "classical", "mood": "relaxed"}
        result = _validate_prefs(raw)
        assert "genre" not in result
        assert result["mood"] == "relaxed"

    def test_unknown_mood_is_dropped(self):
        raw = {"genre": "pop", "mood": "zen"}
        result = _validate_prefs(raw)
        assert "mood" not in result
        assert result["genre"] == "pop"

    def test_energy_clamped_above_one(self):
        result = _validate_prefs({"energy": 1.5})
        assert result["energy"] == 1.0

    def test_energy_clamped_below_zero(self):
        result = _validate_prefs({"energy": -0.3})
        assert result["energy"] == 0.0

    def test_tempo_bpm_clamped_to_range(self):
        result = _validate_prefs({"tempo_bpm": 300.0})
        assert result["tempo_bpm"] == 200.0

    def test_tempo_bpm_below_min_clamped(self):
        result = _validate_prefs({"tempo_bpm": 10.0})
        assert result["tempo_bpm"] == 60.0

    def test_unknown_keys_are_dropped(self):
        raw = {"genre": "pop", "favorite_color": "blue", "vibe": "sunny"}
        result = _validate_prefs(raw)
        assert set(result.keys()) == {"genre"}

    def test_non_numeric_energy_is_dropped(self):
        result = _validate_prefs({"energy": "high"})
        assert "energy" not in result

    def test_empty_dict_returns_empty(self):
        assert _validate_prefs({}) == {}

    def test_all_numeric_features_clamped(self):
        raw = {"energy": 2.0, "acousticness": -1.0, "valence": 0.5, "danceability": 99.0}
        result = _validate_prefs(raw)
        assert result["energy"] == 1.0
        assert result["acousticness"] == 0.0
        assert result["valence"] == 0.5
        assert result["danceability"] == 1.0

    def test_multi_genre_list_filters_invalid(self):
        """Valid genres pass through; invalid ones are dropped from the list."""
        result = _validate_prefs({"genre": ["reggaeton", "classical", "bachata"]})
        assert result["genre"] == ["reggaeton", "bachata"]

    def test_multi_genre_single_valid_collapses_to_string(self):
        """A list with only one valid genre becomes a plain string."""
        result = _validate_prefs({"genre": ["classical", "reggaeton", "opera"]})
        assert result["genre"] == "reggaeton"

    def test_multi_genre_all_invalid_drops_key(self):
        """A list where every genre is invalid produces no genre key."""
        result = _validate_prefs({"genre": ["opera", "bluegrass"]})
        assert "genre" not in result


# ---------------------------------------------------------------------------
# parse_user_intent — mocks _call_bedrock
# ---------------------------------------------------------------------------

class TestParseUserIntent:

    def test_valid_json_is_parsed_and_validated(self):
        with patch("src.llm._call_bedrock", return_value='{"genre": "lofi", "mood": "chill", "energy": 0.4}'):
            result = parse_user_intent("something chill to code to")
        assert result == {"genre": "lofi", "mood": "chill", "energy": 0.4}

    def test_markdown_fences_are_stripped(self):
        with patch("src.llm._call_bedrock", return_value='```json\n{"genre": "pop"}\n```'):
            result = parse_user_intent("upbeat pop")
        assert result == {"genre": "pop"}

    def test_invalid_json_returns_empty_dict(self):
        with patch("src.llm._call_bedrock", return_value="Sorry, I can't parse that."):
            result = parse_user_intent("something weird")
        assert result == {}

    def test_invalid_genre_from_llm_is_dropped(self):
        with patch("src.llm._call_bedrock", return_value='{"genre": "classical", "mood": "relaxed"}'):
            result = parse_user_intent("classical music please")
        assert "genre" not in result
        assert result.get("mood") == "relaxed"

    def test_out_of_range_energy_is_clamped(self):
        with patch("src.llm._call_bedrock", return_value='{"energy": 1.8}'):
            result = parse_user_intent("super intense")
        assert result["energy"] == 1.0

    def test_empty_json_returns_empty_dict(self):
        with patch("src.llm._call_bedrock", return_value="{}"):
            result = parse_user_intent("hmm")
        assert result == {}


# ---------------------------------------------------------------------------
# generate_response — mocks _call_bedrock
# ---------------------------------------------------------------------------

class TestGenerateResponse:

    def _make_results(self):
        song = {
            "title": "Midnight Coding", "artist": "LoRoom",
            "genre": "lofi", "mood": "chill",
            "energy": 0.42, "tempo_bpm": 78.0,
            "valence": 0.56, "danceability": 0.62, "acousticness": 0.71,
        }
        return [(song, 0.91, "genre match; mood match; energy similarity 98%")]

    def test_returns_string(self):
        with patch("src.llm._call_bedrock", return_value="Here are your picks!"):
            result = generate_response("chill coding music", self._make_results())
        assert isinstance(result, str)
        assert len(result) > 0

    def test_low_confidence_flag_included_in_message(self):
        captured = {}

        def capture(system, user_msg, **kwargs):
            captured["user_msg"] = user_msg
            return "Here are some options, though the match isn't perfect."

        with patch("src.llm._call_bedrock", side_effect=capture):
            generate_response("death metal", self._make_results(), low_confidence=True)

        assert "0.5" in captured["user_msg"] or "limited" in captured["user_msg"]

    def test_high_confidence_no_warning_in_message(self):
        captured = {}

        def capture(system, user_msg, **kwargs):
            captured["user_msg"] = user_msg
            return "Great picks!"

        with patch("src.llm._call_bedrock", side_effect=capture):
            generate_response("chill lofi", self._make_results(), low_confidence=False)

        assert "limited" not in captured["user_msg"]


# ---------------------------------------------------------------------------
# Guardrail: low confidence threshold
# ---------------------------------------------------------------------------

class TestGuardrail:
    """Verify the LOW_CONFIDENCE_THRESHOLD logic in chat.py."""

    def test_threshold_triggers_below_0_5(self):
        from src.chat import LOW_CONFIDENCE_THRESHOLD
        assert LOW_CONFIDENCE_THRESHOLD == 0.5

    def test_low_confidence_true_when_top_score_below_threshold(self):
        from src.chat import LOW_CONFIDENCE_THRESHOLD
        top_score = 0.42
        assert (top_score < LOW_CONFIDENCE_THRESHOLD) is True

    def test_low_confidence_false_when_top_score_at_threshold(self):
        from src.chat import LOW_CONFIDENCE_THRESHOLD
        top_score = 0.5
        assert (top_score < LOW_CONFIDENCE_THRESHOLD) is False


# ---------------------------------------------------------------------------
# Integration test (hits Bedrock — run manually with: pytest -m integration)
# ---------------------------------------------------------------------------

@pytest.mark.integration
def test_full_pipeline_lofi_request():
    """End-to-end: free text -> prefs -> songs -> narrative. Requires Bedrock access."""
    from src.recommender import load_songs, recommend_songs

    songs = load_songs("data/songs.csv")
    prefs = parse_user_intent("I want something chill to code to late at night")

    assert prefs, "parse_user_intent returned empty prefs"
    assert any(k in prefs for k in ("genre", "mood", "energy", "acousticness"))

    results = recommend_songs(prefs, songs, k=3)
    assert len(results) == 3

    top_score = results[0][1]
    response = generate_response(
        "I want something chill to code to late at night",
        results,
        low_confidence=top_score < 0.5,
    )
    assert isinstance(response, str)
    assert len(response) > 50

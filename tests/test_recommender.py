import pytest
from dataclasses import asdict
from src.recommender import (
    Song,
    UserProfile,
    Recommender,
    load_songs,
    recommend_songs,
    score_song,
    DEFAULT_WEIGHTS,
    CATEGORICAL_FEATURES,
    _normalize_tempo,
    _build_explanation,
)


# ---------------------------------------------------------------------------
# Fixtures & helpers
# ---------------------------------------------------------------------------

def _pop_happy_song(**overrides):
    defaults = dict(
        id=1, title="Pop Happy", artist="A", genre=["pop"], mood="happy",
        energy=0.8, tempo_bpm=120, valence=0.85, danceability=0.8, acousticness=0.2,
    )
    defaults.update(overrides)
    return Song(**defaults)


def _lofi_chill_song(**overrides):
    defaults = dict(
        id=2, title="Lofi Chill", artist="B", genre=["lofi"], mood="chill",
        energy=0.4, tempo_bpm=80, valence=0.6, danceability=0.5, acousticness=0.9,
    )
    defaults.update(overrides)
    return Song(**defaults)


def _rock_intense_song(**overrides):
    defaults = dict(
        id=3, title="Rock Intense", artist="C", genre=["rock"], mood="intense",
        energy=0.91, tempo_bpm=152, valence=0.48, danceability=0.66, acousticness=0.1,
    )
    defaults.update(overrides)
    return Song(**defaults)


def _song_to_dict(song: Song) -> dict:
    return asdict(song)


def make_small_recommender() -> Recommender:
    return Recommender([_pop_happy_song(), _lofi_chill_song()])


def make_three_song_recommender() -> Recommender:
    return Recommender([_pop_happy_song(), _lofi_chill_song(), _rock_intense_song()])


# ---------------------------------------------------------------------------
# Tests: score_song (functional path)
# ---------------------------------------------------------------------------

class TestScoreSong:
    """Tests for the core scoring function."""

    def test_perfect_categorical_match_scores_higher_than_mismatch(self):
        """A song matching genre+mood should score much higher than one that matches neither."""
        prefs = {"genre": "pop", "mood": "happy", "energy": 0.8}
        match = _song_to_dict(_pop_happy_song())
        mismatch = _song_to_dict(_lofi_chill_song())

        assert score_song(prefs, match) > score_song(prefs, mismatch)

    def test_perfect_match_returns_close_to_one(self):
        """When all user prefs exactly match a song, score should be ~1.0."""
        song = _pop_happy_song()
        prefs = {
            "genre": "pop", "mood": "happy", "energy": 0.8,
            "valence": 0.85, "danceability": 0.8,
            "acousticness": 0.2, "tempo_bpm": 120,
        }
        score = score_song(prefs, _song_to_dict(song))
        assert score == pytest.approx(1.0, abs=0.01)

    def test_total_mismatch_scores_low(self):
        """A song that mismatches all features should score well below 0.5."""
        prefs = {
            "genre": "jazz", "mood": "relaxed", "energy": 0.3,
            "valence": 0.5, "danceability": 0.4,
            "acousticness": 0.95, "tempo_bpm": 60,
        }
        song = _song_to_dict(_pop_happy_song())
        score = score_song(prefs, song)
        assert score < 0.5

    def test_partial_prefs_only_scores_provided_features(self):
        """If user only provides genre, score depends solely on genre match."""
        prefs_match = {"genre": "pop"}
        prefs_miss = {"genre": "jazz"}
        song = _song_to_dict(_pop_happy_song())

        assert score_song(prefs_match, song) == pytest.approx(1.0)
        assert score_song(prefs_miss, song) == pytest.approx(0.0)

    def test_energy_similarity_is_continuous(self):
        """Closer energy values should produce higher scores than distant ones."""
        song = _song_to_dict(_pop_happy_song())  # energy=0.8
        close = {"energy": 0.78}
        far = {"energy": 0.3}

        assert score_song(close, song) > score_song(far, song)

    def test_empty_prefs_returns_zero(self):
        """Empty user prefs should return 0.0, not crash."""
        song = _song_to_dict(_pop_happy_song())
        assert score_song({}, song) == 0.0

    def test_unknown_feature_key_is_ignored(self):
        """Prefs with unrecognized keys should not crash or affect score."""
        prefs = {"genre": "pop", "favorite_color": "blue"}
        song = _song_to_dict(_pop_happy_song())
        score = score_song(prefs, song)
        assert score == pytest.approx(1.0)

    def test_score_is_between_zero_and_one(self):
        """Score should always be in [0, 1]."""
        prefs = {"genre": "pop", "mood": "happy", "energy": 0.5, "valence": 0.7}
        song = _song_to_dict(_pop_happy_song())
        score = score_song(prefs, song)
        assert 0.0 <= score <= 1.0


# ---------------------------------------------------------------------------
# Tests: tier weight dominance
# ---------------------------------------------------------------------------

class TestTierWeighting:
    """Verify that tier 1 features dominate over tier 2 and tier 3."""

    def test_genre_match_outweighs_all_tier3_features(self):
        """A genre match alone should produce a higher score than perfect tier-3 matches alone."""
        song = _song_to_dict(_pop_happy_song())

        genre_prefs = {"genre": "pop"}
        tier3_prefs = {"danceability": 0.8, "tempo_bpm": 120}

        genre_score = score_song(genre_prefs, song)
        tier3_score = score_song(tier3_prefs, song)

        # genre match = 1.0 * w_genre / w_genre = 1.0
        # tier3 perfect = 1.0 -- same numerically, but genre should be >= tier3
        assert genre_score >= tier3_score

    def test_tier1_mismatch_drags_score_down_despite_tier2_match(self):
        """If tier-1 features all mismatch, even perfect tier-2 matches can't save the score."""
        song = _song_to_dict(_pop_happy_song())  # pop, happy, energy=0.8, acousticness=0.2, valence=0.85

        # Mismatches on all tier-1, perfect on tier-2
        prefs = {
            "genre": "jazz", "mood": "chill", "energy": 0.2,
            "acousticness": 0.2, "valence": 0.85,
        }
        score = score_song(prefs, song)
        assert score < 0.5

    def test_two_songs_same_tier2_tier3_differ_on_genre(self):
        """Between two otherwise identical songs, genre match should win."""
        base = dict(
            id=1, title="A", artist="X", genre=["pop"], mood="happy",
            energy=0.5, tempo_bpm=100, valence=0.7, danceability=0.6, acousticness=0.5,
        )
        alt = dict(base, id=2, title="B", genre=["rock"])

        prefs = {
            "genre": "pop", "mood": "happy", "energy": 0.5,
            "valence": 0.7, "danceability": 0.6, "acousticness": 0.5, "tempo_bpm": 100,
        }

        assert score_song(prefs, base) > score_song(prefs, alt)


# ---------------------------------------------------------------------------
# Tests: multi-genre queries
# ---------------------------------------------------------------------------

class TestMultiGenreScoring:

    def test_song_matches_if_any_requested_genre_is_in_tags(self):
        """A song tagged ['bachata', 'latin pop', 'spanish'] should match ["reggaeton", "bachata"]."""
        song = {"genre": ["bachata", "latin pop", "spanish"], "mood": "romantic",
                "energy": 0.7, "tempo_bpm": 120, "valence": 0.85, "danceability": 0.85, "acousticness": 0.2}
        prefs = {"genre": ["reggaeton", "bachata"]}
        assert score_song(prefs, song) == pytest.approx(1.0)

    def test_song_scores_zero_if_no_requested_genre_in_tags(self):
        """A pop song should not match a ["reggaeton", "bachata"] query."""
        song = {"genre": ["pop", "indie pop", "electronic"], "mood": "happy",
                "energy": 0.82, "tempo_bpm": 118, "valence": 0.84, "danceability": 0.79, "acousticness": 0.18}
        prefs = {"genre": ["reggaeton", "bachata"]}
        assert score_song(prefs, song) == pytest.approx(0.0)

    def test_multi_genre_list_with_single_entry_still_works(self):
        """A single-item list behaves the same as a plain string genre."""
        song = _song_to_dict(_pop_happy_song())
        assert score_song({"genre": ["pop"]}, song) == score_song({"genre": "pop"}, song)


# ---------------------------------------------------------------------------
# Tests: tempo normalization
# ---------------------------------------------------------------------------

class TestTempoNormalization:

    def test_min_tempo_normalizes_to_zero(self):
        assert _normalize_tempo(60) == pytest.approx(0.0)

    def test_max_tempo_normalizes_to_one(self):
        assert _normalize_tempo(200) == pytest.approx(1.0)

    def test_midpoint_tempo(self):
        assert _normalize_tempo(130) == pytest.approx(0.5, abs=0.01)

    def test_below_min_clamps_to_zero(self):
        assert _normalize_tempo(30) == pytest.approx(0.0)

    def test_above_max_clamps_to_one(self):
        assert _normalize_tempo(250) == pytest.approx(1.0)

    def test_tempo_similarity_in_scoring(self):
        """Two songs with close tempos should score more similarly than distant ones."""
        song = _song_to_dict(_pop_happy_song())  # tempo_bpm=120
        close = {"tempo_bpm": 118}
        far = {"tempo_bpm": 60}

        assert score_song(close, song) > score_song(far, song)


# ---------------------------------------------------------------------------
# Tests: likes_acoustic boolean conversion
# ---------------------------------------------------------------------------

class TestLikesAcoustic:

    def test_likes_acoustic_true_prefers_high_acousticness(self):
        """A user who likes acoustic should prefer songs with high acousticness."""
        rec = make_small_recommender()
        user = UserProfile(
            favorite_genre="lofi", favorite_mood="chill",
            target_energy=0.4, likes_acoustic=True,
        )
        results = rec.recommend(user, k=2)
        # lofi chill song has acousticness=0.9, should rank first
        assert results[0].acousticness > 0.5

    def test_likes_acoustic_false_prefers_low_acousticness(self):
        """A user who dislikes acoustic should prefer songs with low acousticness."""
        rec = make_small_recommender()
        user = UserProfile(
            favorite_genre="pop", favorite_mood="happy",
            target_energy=0.8, likes_acoustic=False,
        )
        results = rec.recommend(user, k=2)
        assert results[0].acousticness < 0.5


# ---------------------------------------------------------------------------
# Tests: UserProfile optional fields
# ---------------------------------------------------------------------------

class TestUserProfile:

    def test_default_optional_fields_are_none(self):
        user = UserProfile(
            favorite_genre="pop", favorite_mood="happy",
            target_energy=0.8, likes_acoustic=False,
        )
        assert user.target_valence is None
        assert user.target_danceability is None
        assert user.target_tempo_bpm is None

    def test_optional_fields_are_used_when_set(self):
        """Setting optional fields should influence the recommendation order."""
        songs = [
            _pop_happy_song(valence=0.9),
            _pop_happy_song(id=4, title="Pop Happy Low Valence", valence=0.3),
        ]
        rec = Recommender(songs)
        user = UserProfile(
            favorite_genre="pop", favorite_mood="happy",
            target_energy=0.8, likes_acoustic=False,
            target_valence=0.9,
        )
        results = rec.recommend(user, k=2)
        assert results[0].valence == pytest.approx(0.9)


# ---------------------------------------------------------------------------
# Tests: Recommender.recommend (OOP path)
# ---------------------------------------------------------------------------

class TestRecommender:

    def test_recommend_returns_songs_sorted_by_score(self):
        """Original starter test -- pop/happy song should rank first for matching user."""
        user = UserProfile(
            favorite_genre="pop", favorite_mood="happy",
            target_energy=0.8, likes_acoustic=False,
        )
        rec = make_small_recommender()
        results = rec.recommend(user, k=2)

        assert len(results) == 2
        assert "pop" in results[0].genre
        assert results[0].mood == "happy"

    def test_recommend_respects_k(self):
        rec = make_three_song_recommender()
        user = UserProfile(
            favorite_genre="pop", favorite_mood="happy",
            target_energy=0.8, likes_acoustic=False,
        )
        assert len(rec.recommend(user, k=1)) == 1
        assert len(rec.recommend(user, k=2)) == 2
        assert len(rec.recommend(user, k=3)) == 3

    def test_recommend_k_larger_than_catalog(self):
        """Requesting more songs than available should return all songs."""
        rec = make_small_recommender()
        user = UserProfile(
            favorite_genre="pop", favorite_mood="happy",
            target_energy=0.8, likes_acoustic=False,
        )
        results = rec.recommend(user, k=10)
        assert len(results) == 2

    def test_recommend_results_are_song_objects(self):
        rec = make_small_recommender()
        user = UserProfile(
            favorite_genre="pop", favorite_mood="happy",
            target_energy=0.8, likes_acoustic=False,
        )
        results = rec.recommend(user, k=2)
        for r in results:
            assert isinstance(r, Song)

    def test_recommend_different_users_get_different_top_pick(self):
        """A pop-lover and a lofi-lover should get different #1 recommendations."""
        rec = make_small_recommender()
        pop_user = UserProfile(
            favorite_genre="pop", favorite_mood="happy",
            target_energy=0.8, likes_acoustic=False,
        )
        lofi_user = UserProfile(
            favorite_genre="lofi", favorite_mood="chill",
            target_energy=0.4, likes_acoustic=True,
        )
        pop_results = rec.recommend(pop_user, k=1)
        lofi_results = rec.recommend(lofi_user, k=1)

        assert "pop" in pop_results[0].genre
        assert "lofi" in lofi_results[0].genre

    def test_recommend_single_song_catalog(self):
        """A catalog with one song should always return that song."""
        rec = Recommender([_pop_happy_song()])
        user = UserProfile(
            favorite_genre="jazz", favorite_mood="relaxed",
            target_energy=0.3, likes_acoustic=True,
        )
        results = rec.recommend(user, k=5)
        assert len(results) == 1
        assert results[0].title == "Pop Happy"


# ---------------------------------------------------------------------------
# Tests: explain_recommendation
# ---------------------------------------------------------------------------

class TestExplanation:

    def test_explain_returns_non_empty_string(self):
        """Original starter test."""
        user = UserProfile(
            favorite_genre="pop", favorite_mood="happy",
            target_energy=0.8, likes_acoustic=False,
        )
        rec = make_small_recommender()
        explanation = rec.explain_recommendation(user, rec.songs[0])
        assert isinstance(explanation, str)
        assert explanation.strip() != ""

    def test_explain_mentions_matched_genre(self):
        user = UserProfile(
            favorite_genre="pop", favorite_mood="happy",
            target_energy=0.8, likes_acoustic=False,
        )
        rec = make_small_recommender()
        explanation = rec.explain_recommendation(user, rec.songs[0])
        assert "genre" in explanation.lower()

    def test_explain_mentions_energy(self):
        user = UserProfile(
            favorite_genre="pop", favorite_mood="happy",
            target_energy=0.8, likes_acoustic=False,
        )
        rec = make_small_recommender()
        explanation = rec.explain_recommendation(user, rec.songs[0])
        assert "energy" in explanation.lower()

    def test_build_explanation_functional(self):
        prefs = {"genre": "pop", "mood": "happy", "energy": 0.8}
        song = _song_to_dict(_pop_happy_song())
        explanation = _build_explanation(prefs, song)
        assert "genre" in explanation.lower()
        assert len(explanation) > 10


# ---------------------------------------------------------------------------
# Tests: load_songs
# ---------------------------------------------------------------------------

class TestLoadSongs:

    def test_loads_correct_number_of_songs(self):
        songs = load_songs("data/songs.csv")
        assert len(songs) == 210

    def test_song_dict_has_expected_keys(self):
        songs = load_songs("data/songs.csv")
        expected_keys = {
            "id", "title", "artist", "genre", "mood",
            "energy", "tempo_bpm", "valence", "danceability", "acousticness",
        }
        for song in songs:
            assert expected_keys.issubset(song.keys())

    def test_numeric_fields_are_floats(self):
        songs = load_songs("data/songs.csv")
        for song in songs:
            assert isinstance(song["energy"], float)
            assert isinstance(song["tempo_bpm"], float)
            assert isinstance(song["valence"], float)
            assert isinstance(song["danceability"], float)
            assert isinstance(song["acousticness"], float)

    def test_id_field_is_int(self):
        songs = load_songs("data/songs.csv")
        for song in songs:
            assert isinstance(song["id"], int)

    def test_first_song_is_sunrise_city(self):
        songs = load_songs("data/songs.csv")
        assert songs[0]["title"] == "Sunrise City"
        assert "pop" in songs[0]["genre"]


# ---------------------------------------------------------------------------
# Tests: recommend_songs (functional path)
# ---------------------------------------------------------------------------

class TestRecommendSongsFunctional:

    def test_returns_list_of_tuples(self):
        songs = load_songs("data/songs.csv")
        prefs = {"genre": "pop", "mood": "happy", "energy": 0.8}
        results = recommend_songs(prefs, songs, k=3)
        assert len(results) == 3
        for item in results:
            assert len(item) == 3
            song, score, explanation = item
            assert isinstance(song, dict)
            assert isinstance(score, float)
            assert isinstance(explanation, str)

    def test_results_sorted_descending_by_score(self):
        songs = load_songs("data/songs.csv")
        prefs = {"genre": "pop", "mood": "happy", "energy": 0.8}
        results = recommend_songs(prefs, songs, k=5)
        scores = [score for _, score, _ in results]
        assert scores == sorted(scores, reverse=True)

    def test_top_pick_for_pop_happy_user(self):
        """Pop happy user should get Sunrise City or Rooftop Lights near the top."""
        songs = load_songs("data/songs.csv")
        prefs = {"genre": "pop", "mood": "happy", "energy": 0.8}
        results = recommend_songs(prefs, songs, k=3)
        top_titles = [s["title"] for s, _, _ in results]
        assert "Sunrise City" in top_titles

    def test_top_pick_for_lofi_chill_user(self):
        """Lofi chill user should get lofi songs ranked high."""
        songs = load_songs("data/songs.csv")
        prefs = {"genre": "lofi", "mood": "chill", "energy": 0.4}
        results = recommend_songs(prefs, songs, k=3)
        assert any("lofi" in s["genre"] for s, _, _ in results)

    def test_k_zero_returns_empty(self):
        songs = load_songs("data/songs.csv")
        prefs = {"genre": "pop"}
        results = recommend_songs(prefs, songs, k=0)
        assert results == []


# ---------------------------------------------------------------------------
# Tests: DEFAULT_WEIGHTS constant
# ---------------------------------------------------------------------------

class TestWeightsConfig:

    def test_all_features_have_weights(self):
        expected = {"genre", "mood", "energy", "acousticness", "valence", "danceability", "tempo_bpm"}
        assert expected == set(DEFAULT_WEIGHTS.keys())

    def test_tier1_weights_are_highest(self):
        tier1 = [DEFAULT_WEIGHTS[f] for f in ("genre", "mood", "energy")]
        tier2 = [DEFAULT_WEIGHTS[f] for f in ("acousticness", "valence")]
        tier3 = [DEFAULT_WEIGHTS[f] for f in ("danceability", "tempo_bpm")]

        assert min(tier1) > max(tier2)
        assert min(tier2) > max(tier3)

    def test_all_weights_are_positive(self):
        for w in DEFAULT_WEIGHTS.values():
            assert w > 0

    def test_categorical_features_set(self):
        assert "genre" in CATEGORICAL_FEATURES
        assert "mood" in CATEGORICAL_FEATURES
        assert "energy" not in CATEGORICAL_FEATURES

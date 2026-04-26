"""Interactive AI-powered music recommender (RAG pipeline entrypoint)."""

import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

from botocore.exceptions import ClientError
from recommender import load_songs, recommend_songs
from llm import parse_user_intent, generate_response

LOW_CONFIDENCE_THRESHOLD = 0.5
TOP_K = 5


def run_chat(songs: list) -> None:
    print("\nVibeFinder AI")
    print("Describe what you're in the mood for and I'll find the best matches.")
    print("(type 'quit' to exit)\n")

    while True:
        try:
            user_input = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye!")
            break

        if not user_input:
            continue
        if user_input.lower() in {"quit", "exit", "q"}:
            print("Goodbye!")
            break

        try:
            prefs = parse_user_intent(user_input)
            k = prefs.pop("_k", TOP_K)

            if not prefs:
                print(
                    "\nVibeFinder: I couldn't pick out any music preferences from that. "
                    "Try mentioning a mood, energy level, or genre.\n"
                )
                continue

            results = recommend_songs(prefs, songs, k=k)
            top_score = results[0][1] if results else 0.0
            low_confidence = top_score < LOW_CONFIDENCE_THRESHOLD

            response = generate_response(user_input, results, low_confidence=low_confidence)
            print(f"\nVibeFinder: {response}\n")

        except ClientError as e:
            print(f"\n[Error] Bedrock API call failed: {e}\n")
        except Exception as e:
            print(f"\n[Error] Something went wrong: {e}\n")


def main() -> None:
    songs = load_songs("data/songs.csv")
    run_chat(songs)


if __name__ == "__main__":
    main()

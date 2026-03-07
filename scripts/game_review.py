#!/usr/bin/env python3
"""
Generate visual game-review reports for Hex games.

This CLI now uses the shared review pipeline that also backs the web review page.
The review compares the played move against the played move plus a policy-ranked
candidate set, then exports JSON and a richer standalone HTML page.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from hex_ai.inference.model_config import get_model_path
from hex_ai.review.game_review import (
    format_review_as_html,
    format_review_as_json,
    parse_games_from_json_payload,
    parse_single_trmph_game,
    reviewer_from_model_path,
    save_review_files,
)

logger = logging.getLogger(__name__)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Generate visual game-review reports for Hex games",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python scripts/game_review.py --game "g7g6j4h6i6h7" --model best
  python scripts/game_review.py --game "..." --candidate-top-k 10 --suggestion-count 3
  python scripts/game_review.py --file games.json --output-dir temp/reviews/
        """.strip(),
    )

    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument("--game", type=str, help="TRMPH game string to review")
    input_group.add_argument("--file", type=str, help="JSON file containing game(s) to review")

    parser.add_argument("--model", type=str, default="best", help="Model to use for analysis")
    parser.add_argument("--output", type=str, help="Output JSON file")
    parser.add_argument("--html-output", type=str, help="Output HTML file")
    parser.add_argument("--output-dir", type=str, help="Output directory for multiple games")
    parser.add_argument(
        "--candidate-top-k",
        type=int,
        default=8,
        help="Evaluate the played move plus the top-K policy candidates at each ply (default: 8)",
    )
    parser.add_argument(
        "--suggestion-count",
        type=int,
        default=3,
        help="How many alternative move markers to include in the review UI (default: 3)",
    )
    parser.add_argument(
        "--policy-temperature",
        type=float,
        default=1.0,
        help="Policy temperature used for ranking candidate moves (default: 1.0)",
    )
    parser.add_argument("--verbose", "-v", action="count", default=0, help="Increase verbosity")
    return parser


def main() -> int:
    parser = _build_parser()
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.WARNING,
        format="%(asctime)s - %(levelname)s - %(message)s",
    )
    if args.verbose >= 2:
        logging.getLogger().setLevel(logging.DEBUG)
    elif args.verbose >= 1:
        logging.getLogger().setLevel(logging.INFO)

    try:
        model_path = get_model_path(args.model)
        reviewer = reviewer_from_model_path(
            model_path,
            model_label=args.model,
            candidate_policy_top_k=args.candidate_top_k,
            suggestion_count=args.suggestion_count,
            policy_temperature=args.policy_temperature,
            verbose=0 if args.verbose == 0 else 1,
        )

        if args.game:
            games = [parse_single_trmph_game(args.game)]
        else:
            payload = json.loads(Path(args.file).read_text(encoding="utf-8"))
            games = parse_games_from_json_payload(payload)

        if not games:
            raise ValueError("No valid games found in input")

        reviews = []
        for index, game in enumerate(games, start=1):
            logger.info("Reviewing game %s/%s", index, len(games))
            review = reviewer.review_game_record(game)
            reviews.append(review)
            print(f"\n=== GAME {index} REVIEW SUMMARY ===")
            print(f"Total moves: {review.total_moves}")
            print(f"Total mistakes: {review.total_mistakes}")
            print(f"Major mistakes: {review.major_mistakes}")
            print(f"Losing moves: {review.losing_moves}")
            print(
                "Mistakes by player: "
                f"Blue: {review.mistake_by_player['blue']}, "
                f"Red: {review.mistake_by_player['red']}"
            )

        if args.output_dir:
            save_review_files(reviews, Path(args.output_dir))
        elif args.output:
            output_path = Path(args.output)
            payload = (
                format_review_as_json(reviews[0])
                if len(reviews) == 1
                else [format_review_as_json(review) for review in reviews]
            )
            output_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
            logger.info("Results written to %s", output_path)
        elif args.html_output:
            output_path = Path(args.html_output)
            if len(reviews) != 1:
                raise ValueError("--html-output currently supports a single game; use --output-dir for batches")
            output_path.write_text(format_review_as_html(reviews[0]), encoding="utf-8")
            logger.info("HTML results written to %s", output_path)
        else:
            save_review_files(reviews, Path("analysis/game_reviews"))

        logger.info("Successfully reviewed %s game(s)", len(reviews))
        return 0
    except Exception as exc:
        logger.error("Error: %s", exc)
        if args.verbose:
            logger.exception("Detailed failure")
        return 1


if __name__ == "__main__":
    sys.exit(main())

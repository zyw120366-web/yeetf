from __future__ import annotations

import argparse
from pathlib import Path

from etf_rotation.sentiment_ai import write_manual_review


ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate and commit a complete Codex-chat sentiment review")
    parser.add_argument("--date", required=True)
    parser.add_argument("--reviews", required=True, help="JSON written during the current Codex conversation")
    args = parser.parse_args()
    target = write_manual_review(
        ROOT / "market_data" / "sentiment" / f"{args.date}.json",
        Path(args.reviews),
        ROOT / "market_data" / "sentiment" / "ai_review",
    )
    print(target)


if __name__ == "__main__":
    main()

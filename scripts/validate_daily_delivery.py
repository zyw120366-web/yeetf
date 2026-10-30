"""Check artifacts and clickable local targets without placing any orders."""
import argparse
import json
from pathlib import Path

from etf_rotation.daily_report import delivery_receipt, validate_delivery


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--date", required=True)
    parser.add_argument("--deep", action="store_true", help="forensic check of all hashes recorded in a historical Git snapshot; not a daily step")
    parser.add_argument("--format", choices=("json", "markdown"), default="json",
                        help="markdown outputs a checked, clickable delivery receipt")
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    if args.format == "markdown":
        print(delivery_receipt(root, args.date, deep=args.deep), end="")
    else:
        print(json.dumps(validate_delivery(root, args.date, deep=args.deep), ensure_ascii=False))

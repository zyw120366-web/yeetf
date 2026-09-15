"""Check artifacts and clickable local targets without placing any orders."""
import argparse
import json
from pathlib import Path

from etf_rotation.daily_report import validate_delivery


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--date", required=True)
    args = parser.parse_args()
    print(json.dumps(validate_delivery(Path(__file__).resolve().parents[1], args.date), ensure_ascii=False))

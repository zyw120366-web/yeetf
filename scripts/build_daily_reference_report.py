"""Publish the common daily report without modifying account or order decisions."""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from etf_rotation.daily_report import publish

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--date", required=True)
    args = parser.parse_args()
    report = publish(ROOT, args.date)
    print(f"日报已生成：{args.date}；{report['status']}")


if __name__ == "__main__":
    main()

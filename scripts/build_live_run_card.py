"""Write the compact manifest and run card together for any release status."""
import argparse
from pathlib import Path

from etf_rotation.run_record import write_run_record


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--date", required=True)
    args = parser.parse_args()
    card = write_run_record(Path(__file__).resolve().parents[1], args.date)
    print(f"运行卡：{args.date} {card['release']['readiness']}")

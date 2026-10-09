from __future__ import annotations

import argparse
import sys
from pathlib import Path

import yaml
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from etf_rotation.data import fetch_easy_tdx


def main() -> None:
    parser = argparse.ArgumentParser(description="Fetch adjusted daily ETF prices")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    config = yaml.safe_load(
        (ROOT / "config" / "market.yaml").read_text(encoding="utf-8")
    )
    data_dir = ROOT / "market_data" / "prices"
    fetch_easy_tdx(config, data_dir, force=args.force)
    from fetch_live_quotes import fetch
    benchmark = config["benchmark"]
    frame = pd.read_csv(data_dir / f"{benchmark['code']}.{benchmark['market']}.csv")
    fetch(str(pd.to_datetime(frame["datetime"]).max().date()))


if __name__ == "__main__":
    main()

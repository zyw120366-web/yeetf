from __future__ import annotations

import argparse
import json
from pathlib import Path

from etf_rotation.sentiment_ai import canonical_hash, normalize_snapshot


ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    parser = argparse.ArgumentParser(description="Export the complete Codex-chat sentiment review queue")
    parser.add_argument("--date", required=True)
    args = parser.parse_args()
    snapshot_path = ROOT / "market_data" / "sentiment" / f"{args.date}.json"
    if not snapshot_path.exists():
        raise RuntimeError(f"missing immutable market snapshot: {snapshot_path}")
    snapshot = json.loads(snapshot_path.read_text(encoding="utf-8"))
    failed_sources = [name for name, block in snapshot.get("sources", {}).items() if not block.get("ok")]
    if failed_sources:
        raise RuntimeError(f"source collection incomplete; fail closed: {failed_sources}")
    items = normalize_snapshot(snapshot)
    payload = {
        "date": args.date,
        "policy": "review_every_collected_row_in_current_codex_chat",
        "snapshot_hash": canonical_hash(snapshot),
        "input_count": len(items),
        "items": items,
    }
    output = ROOT / "market_data" / "sentiment" / "review_queue" / f"{args.date}.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    serialized = json.dumps(payload, ensure_ascii=False, indent=2)
    if output.exists() and output.read_text(encoding="utf-8") != serialized:
        raise RuntimeError(f"immutable review queue already exists with different content: {output}")
    output.write_text(serialized, encoding="utf-8")
    print(output)
    print(f"items={len(items)}")


if __name__ == "__main__":
    main()

"""Install the repository-owned ye skill into the current user's Codex home."""
from __future__ import annotations

import argparse
import json
import os
import shutil
from datetime import datetime
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "skills" / "ye-daily-execution"


def main() -> None:
    parser = argparse.ArgumentParser(description="Install the ye daily Codex skill")
    parser.add_argument(
        "--codex-home",
        type=Path,
        default=Path(os.environ.get("CODEX_HOME", Path.home() / ".codex")),
    )
    args = parser.parse_args()
    target = args.codex_home.expanduser().resolve() / "skills" / SOURCE.name
    if not SOURCE.is_dir():
        raise FileNotFoundError(f"repository skill is missing: {SOURCE}")
    if target.is_symlink() and target.resolve() == SOURCE.resolve():
        print(json.dumps({"status": "already_installed", "target": str(target)}, ensure_ascii=False))
        return
    backup = None
    if target.exists() or target.is_symlink():
        stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        backup = target.with_name(f"{target.name}.backup-{stamp}")
        target.rename(backup)
    target.parent.mkdir(parents=True, exist_ok=True)
    try:
        target.symlink_to(SOURCE.resolve(), target_is_directory=True)
        mode = "symlink"
    except OSError:
        shutil.copytree(SOURCE, target)
        mode = "copy"
    print(
        json.dumps(
            {
                "status": "installed",
                "mode": mode,
                "source": str(SOURCE),
                "target": str(target),
                "backup": str(backup) if backup else None,
            },
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()

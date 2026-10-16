from __future__ import annotations

import importlib.util
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_project_hygiene_contract() -> None:
    script = ROOT / "scripts" / "validate_project_hygiene.py"
    spec = importlib.util.spec_from_file_location("validate_project_hygiene", script)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    assert module.collect_issues() == []

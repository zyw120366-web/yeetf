import json
import sys
from pathlib import Path

from scripts import install_codex_skill as installer


def test_skill_upgrade_keeps_recoverable_backup_outside_discovery(tmp_path, monkeypatch, capsys):
    source = tmp_path / "project/ye-daily-execution"
    source.mkdir(parents=True)
    (source / "SKILL.md").write_text("current skill")
    codex = tmp_path / "codex"
    target = codex / "skills/ye-daily-execution"
    target.mkdir(parents=True)
    (target / "SKILL.md").write_text("old skill")
    monkeypatch.setattr(installer, "SOURCE", source)
    monkeypatch.setattr(sys, "argv", ["install", "--codex-home", str(codex)])
    installer.main()
    result = json.loads(capsys.readouterr().out)
    backup = Path(result["backup"])
    assert backup.parent == codex / "skill-archives"
    assert (backup / "SKILL.md").read_text() == "old skill"
    assert [p.name for p in (codex / "skills").iterdir()] == ["ye-daily-execution"]
    assert target.resolve() == source and (target / "SKILL.md").read_text() == "current skill"
    installer.main()
    assert json.loads(capsys.readouterr().out)["status"] == "already_installed"
    assert len(list((codex / "skill-archives").iterdir())) == 1

from __future__ import annotations

import json
import re
import subprocess
from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parents[1]
REQUIRED = {
    "README.md",
    "RESEARCH_MEMORY.md",
    "RESEARCH_STATUS.md",
    "docs/策略规范.md",
    "docs/每日执行.md",
    "docs/工程与治理.md",
}
FORBIDDEN = {
    "dashboard/README.md",
    "docs/工程审计.md",
    "docs/策略治理与实盘审计.md",
    "scripts/package_complete_bundle.py",
    "results/audit/latest_live_run_card.json",
    "results/audit/latest_run_manifest.json",
    "results/comparison/round_trips.csv",
}


def project_files() -> set[str]:
    output = subprocess.check_output(
        ["git", "ls-files", "--cached", "--others", "--exclude-standard"],
        cwd=ROOT,
        text=True,
    )
    return {
        line for line in output.splitlines()
        if line and (ROOT / line).exists()
    }


def markdown_links(path: Path) -> list[str]:
    text = path.read_text(encoding="utf-8")
    return re.findall(r"\[[^]]+\]\(([^)]+)\)", text)


def collect_issues() -> list[str]:
    issues: list[str] = []
    tracked = project_files()
    for relative in sorted(REQUIRED):
        if relative not in tracked or not (ROOT / relative).exists():
            issues.append(f"missing canonical file: {relative}")
    for relative in sorted(FORBIDDEN):
        if relative in tracked or (ROOT / relative).exists():
            issues.append(f"redundant file must be removed: {relative}")
    for relative in sorted(tracked):
        path = Path(relative)
        if "scratch" in path.parts or path.suffix in {".zip", ".pyc"}:
            issues.append(f"temporary artifact is tracked: {relative}")
        if len(path.parts) >= 3 and path.parts[:3] == ("results", "audit", "roc_score_study"):
            issues.append(f"research result stored under audit: {relative}")

    governance = yaml.safe_load(
        (ROOT / "config" / "strategy_governance.yaml").read_text(encoding="utf-8")
    )
    market = yaml.safe_load(
        (ROOT / "config" / "market.yaml").read_text(encoding="utf-8")
    )
    hypotheses = yaml.safe_load(
        (ROOT / "config" / "research_hypotheses.yaml").read_text(encoding="utf-8")
    )
    strategy_id = str(governance["formal_strategy"]["id"])
    data_end = str(market["project"]["data_end"])
    status = (ROOT / "RESEARCH_STATUS.md").read_text(encoding="utf-8")
    memory = (ROOT / "RESEARCH_MEMORY.md").read_text(encoding="utf-8")
    if hypotheses["registry"]["current_formal_strategy_id"] != strategy_id:
        issues.append("research registry formal id differs from governance")
    for name, text in (("RESEARCH_STATUS.md", status), ("RESEARCH_MEMORY.md", memory)):
        if strategy_id not in text:
            issues.append(f"{name} is missing current formal strategy id")
        if data_end not in text:
            issues.append(f"{name} is missing current data end")
    if "## 变更记录" not in memory:
        issues.append("RESEARCH_MEMORY.md is missing the change ledger")
    if len(memory.splitlines()) > 200:
        issues.append("RESEARCH_MEMORY.md exceeds 200 lines")
    if len(status.splitlines()) > 80:
        issues.append("RESEARCH_STATUS.md exceeds 80 lines")

    daily_entry = (ROOT / "scripts" / "run_after_close.py").read_text(encoding="utf-8")
    if "experiments/" in daily_entry or "research_" in daily_entry:
        issues.append("daily entry references research code")

    for template in sorted((ROOT / "results" / "live").glob("*_actual_fills.template.json")):
        date = template.name[:10]
        plan_path = ROOT / "results" / "live" / f"{date}_order_plan.json"
        if not plan_path.exists():
            issues.append(f"fill template has no matching plan: {template.name}")
            continue
        plan = json.loads(plan_path.read_text(encoding="utf-8"))
        if not any(action.get("side") in {"buy", "sell"} for action in plan["actions"]):
            issues.append(f"fill template exists for a no-order day: {template.name}")

    for document in (ROOT / "README.md", ROOT / "RESEARCH_STATUS.md"):
        for target in markdown_links(document):
            if target.startswith(("http://", "https://", "#")):
                continue
            if not (document.parent / target).resolve().exists():
                issues.append(f"broken local link in {document.name}: {target}")

    latest_card = ROOT / "results" / "audit" / f"{data_end}_live_run_card.json"
    if latest_card.exists():
        card = json.loads(latest_card.read_text(encoding="utf-8"))
        effective_from = str(
            governance["formal_strategy"].get("effective_for_signal_dates_from", "0001-01-01")
        )
        if data_end >= effective_from and card.get("strategy", {}).get("id") != strategy_id:
            issues.append("latest dated run card uses a stale formal strategy id")
    else:
        issues.append(f"missing latest dated run card: {latest_card.name}")
    return issues


def main() -> None:
    issues = collect_issues()
    if issues:
        print(json.dumps({"status": "FAIL", "issues": issues}, ensure_ascii=False, indent=2))
        raise SystemExit(1)
    print(json.dumps({"status": "PASS", "checks": "project hygiene"}, ensure_ascii=False))


if __name__ == "__main__":
    main()

"""Create the portable complete ye package without virtual environments or caches."""
from __future__ import annotations

import argparse
import json
import zipfile
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
OUTPUTS = ROOT / "outputs"
SKILL = ROOT / "skills" / "ye-daily-execution"

DIRECTORIES = ("config", "docs", "scripts", "src", "tests", "dashboard", "experiments", "market_data", "results")
TOP_LEVEL = ("README.md", "RESEARCH_MEMORY.md", "RESEARCH_STATUS.md", "requirements.txt", "run_strategies.py")
EXCLUDED_PARTS = {".venv-review", ".pytest_cache", "__pycache__"}


def wanted(path: Path) -> bool:
    return not any(part in EXCLUDED_PARTS for part in path.parts) and path.suffix != ".pyc"


def add_tree(archive: zipfile.ZipFile, source: Path, destination: Path) -> int:
    count = 0
    for path in source.rglob("*"):
        if path.is_file() and wanted(path):
            archive.write(path, (destination / path.relative_to(source)).as_posix())
            count += 1
    return count


def main() -> None:
    parser = argparse.ArgumentParser(description="Build the complete portable ye package")
    parser.add_argument("--output", type=Path, default=OUTPUTS / "ye策略完整包.zip")
    args = parser.parse_args()
    output = args.output.resolve()
    if output.exists():
        raise FileExistsError(f"refusing to overwrite existing package: {output}")
    if not SKILL.exists():
        raise FileNotFoundError(f"skill source is missing: {SKILL}")
    prefix = Path("ye策略完整包")
    count = 0
    output.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(output, "x", compression=zipfile.ZIP_DEFLATED, compresslevel=9) as archive:
        count += add_tree(archive, SKILL, prefix / "skill" / "ye-daily-execution")
        for name in TOP_LEVEL:
            path = ROOT / name
            archive.write(path, (prefix / "etf_rotation_strategy" / name).as_posix())
            count += 1
        for name in DIRECTORIES:
            count += add_tree(archive, ROOT / name, prefix / "etf_rotation_strategy" / name)
        for path in sorted(OUTPUTS.glob("ETF轮动策略_*.html")):
            archive.write(path, (prefix / "html" / path.name).as_posix())
            count += 1
        manifest = {
            "package": "ye策略完整包",
            "created": "2026-07-19",
            "file_count": count,
            "included": [
                "可安装的 ye-daily-execution skill",
                "策略源码、配置、文档、测试、仪表盘和实验脚本",
                "market_data 下的价格与情绪快照",
                "results 下的回测、审计、日报和运行结果",
                "三份正式 HTML",
                "原始交接记忆与截至今日的当前状态",
                "策略冻结、假设登记、每日运行卡与成交对账治理模块",
            ],
            "excluded": [".venv-review", ".pytest_cache", "__pycache__", "独立筑底研究项目"],
        }
        archive.writestr((prefix / "PACKAGE_MANIFEST.json").as_posix(), json.dumps(manifest, ensure_ascii=False, indent=2))
    print(json.dumps({"output": str(output), "source_files": count, "bytes": output.stat().st_size}, ensure_ascii=False))


if __name__ == "__main__":
    main()

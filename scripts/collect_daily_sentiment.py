from __future__ import annotations

import argparse
import hashlib
import json
import urllib.parse
import urllib.request
from datetime import date
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "market_data" / "sentiment"
UA = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) Chrome/117 Safari/537.36"


def get_json(url: str, *, headers: dict[str, str] | None = None) -> dict:
    request = urllib.request.Request(url, headers=headers or {"User-Agent": UA})
    with urllib.request.urlopen(request, timeout=20) as response:
        raw = response.read()
    for encoding in ("utf-8", "gb18030"):
        try:
            return json.loads(raw.decode(encoding))
        except (UnicodeDecodeError, json.JSONDecodeError):
            continue
    raise ValueError(f"cannot decode JSON response from {url}")


def ths_hot_reason(day: str) -> list[dict]:
    url = (
        "https://zx.10jqka.com.cn/event/api/getharden/"
        f"date/{day}/orderby/date/orderway/desc/charset/GBK/"
    )
    data = get_json(url)
    return data.get("data") or []


def cls_telegraph(page_size: int = 100) -> list[dict]:
    params = {
        "appName": "CailianpressWeb", "os": "web", "sv": "7.7.5",
        "last_time": "", "refresh_type": "1", "rn": str(page_size),
    }
    query = "&".join(f"{key}={params[key]}" for key in sorted(params))
    sign = hashlib.md5(hashlib.sha1(query.encode()).hexdigest().encode()).hexdigest()
    data = get_json(f"https://www.cls.cn/v1/roll/get_roll_list?{query}&sign={sign}")
    return (data.get("data") or {}).get("roll_data") or []


def eastmoney_limit_pool(day: str, endpoint: str) -> list[dict]:
    params = urllib.parse.urlencode({
        "ut": "7eea3edcaed734bea9cbfc24409ed989", "dpt": "wz.ztzt",
        "Pageindex": 0, "pagesize": 10000,
        "sort": "fund:asc" if endpoint == "getTopicDTPool" else "fbt:asc",
        "date": day.replace("-", ""),
    })
    data = get_json(
        f"https://push2ex.eastmoney.com/{endpoint}?{params}",
        headers={"User-Agent": UA, "Referer": "https://quote.eastmoney.com/"},
    )
    return (data.get("data") or {}).get("pool") or []


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--date", default=date.today().isoformat())
    args = parser.parse_args()
    day = args.date
    payload: dict[str, object] = {
        "date": day,
        "policy": "after_close_immutable_snapshot_for_next_open_only",
        "sources": {},
    }
    calls = {
        "ths_hot_reason": lambda: ths_hot_reason(day),
        "eastmoney_limit_up": lambda: eastmoney_limit_pool(day, "getTopicZTPool"),
        "eastmoney_broken_board": lambda: eastmoney_limit_pool(day, "getTopicZBPool"),
        "eastmoney_limit_down": lambda: eastmoney_limit_pool(day, "getTopicDTPool"),
    }
    if day == date.today().isoformat():
        calls["cls_telegraph_capture"] = cls_telegraph
    for name, call in calls.items():
        try:
            payload["sources"][name] = {"ok": True, "rows": call()}
        except Exception as exc:
            payload["sources"][name] = {"ok": False, "error": repr(exc), "rows": None}

    OUTPUT.mkdir(parents=True, exist_ok=True)
    target = OUTPUT / f"{day}.json"
    serialized = json.dumps(payload, ensure_ascii=False, indent=2)
    if target.exists():
        existing = target.read_text(encoding="utf-8")
        if existing != serialized:
            raise RuntimeError(f"immutable snapshot already exists with different content: {target}")
        print(f"verified unchanged {target}")
        return
    target.write_text(serialized, encoding="utf-8")
    print(f"wrote {target}")


if __name__ == "__main__":
    main()

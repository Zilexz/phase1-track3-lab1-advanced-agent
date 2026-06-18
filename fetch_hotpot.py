"""Tải HotpotQA (distractor, validation) thật từ HuggingFace datasets-server và
chuyển sang format QAExample.

Khác với dữ liệu tự tạo (context sạch), HotpotQA distractor có ~10 đoạn văn,
trong đó nhiều đoạn là NHIỄU -> model phải tự tìm 2 đoạn đúng để nối hop. Đây là
lý do nó tạo ra lỗi đa dạng (entity_drift, incomplete_multi_hop, ...).

Chạy:  python fetch_hotpot.py --n 150 --out data/hotpot_real.json
"""
from __future__ import annotations

import json
import urllib.parse
import urllib.request
from pathlib import Path

import typer

app = typer.Typer(add_completion=False)
BASE = "https://datasets-server.huggingface.co/rows"
VALID_LEVELS = {"easy", "medium", "hard"}


def fetch_page(offset: int, length: int) -> list[dict]:
    params = urllib.parse.urlencode(
        {"dataset": "hotpotqa/hotpot_qa", "config": "distractor", "split": "validation", "offset": offset, "length": length}
    )
    req = urllib.request.Request(f"{BASE}?{params}", headers={"User-Agent": "reflexion-lab"})
    with urllib.request.urlopen(req, timeout=60) as resp:
        data = json.loads(resp.read().decode("utf-8"))
    return [r["row"] for r in data.get("rows", [])]


def to_example(row: dict) -> dict:
    ctx = row.get("context", {})
    titles = ctx.get("title", [])
    sentences = ctx.get("sentences", [])
    context = [{"title": t, "text": " ".join(s).strip()} for t, s in zip(titles, sentences)]
    level = row.get("level", "medium")
    if level not in VALID_LEVELS:
        level = "medium"
    return {
        "qid": row.get("id", ""),
        "difficulty": level,
        "question": row.get("question", "").strip(),
        "gold_answer": row.get("answer", "").strip(),
        "context": context,
    }


@app.command()
def main(n: int = 150, out: str = "data/hotpot_real.json") -> None:
    rows: list[dict] = []
    offset = 0
    while len(rows) < n:
        page = fetch_page(offset, min(100, n - len(rows)))
        if not page:
            break
        rows.extend(page)
        offset += len(page)
    examples = [to_example(r) for r in rows[:n]]
    examples = [e for e in examples if e["question"] and e["gold_answer"] and e["context"]]
    Path(out).write_text(json.dumps(examples, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Wrote {len(examples)} HotpotQA examples -> {out}")
    print("Levels:", {lvl: sum(1 for e in examples if e['difficulty'] == lvl) for lvl in VALID_LEVELS})


if __name__ == "__main__":
    app()

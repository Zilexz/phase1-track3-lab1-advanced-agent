from __future__ import annotations
import json
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
import typer
from rich import print
from rich.progress import Progress
from src.reflexion_lab.agents import ReActAgent, ReflexionAgent
from src.reflexion_lab.reporting import build_report, save_report
from src.reflexion_lab.runtime import get_mode
from src.reflexion_lab.utils import load_dataset, save_jsonl
app = typer.Typer(add_completion=False)


def _run_all(agent, examples, label, workers):
    """Chạy agent trên toàn bộ examples (song song), giữ nguyên thứ tự."""
    results = [None] * len(examples)
    with Progress() as progress:
        task = progress.add_task(f"[cyan]{label}", total=len(examples))
        with ThreadPoolExecutor(max_workers=workers) as pool:
            futures = {pool.submit(agent.run, ex): i for i, ex in enumerate(examples)}
            for fut in as_completed(futures):
                results[futures[fut]] = fut.result()
                progress.advance(task)
    return results

@app.command()
def main(
    dataset: str = "data/hotpot_mini.json",
    out_dir: str = "outputs/sample_run",
    reflexion_attempts: int = 3,
    adaptive: bool = typer.Option(True, help="Bật adaptive_max_attempts (ngân sách lần thử theo độ khó)."),
    limit: int = typer.Option(0, help="Chỉ chạy N ví dụ đầu (0 = tất cả). Dùng để thử nhanh khi gọi LLM thật."),
    workers: int = typer.Option(6, help="Số luồng song song khi gọi LLM (mock thì để 1 cũng tức thì)."),
) -> None:
    mode = get_mode()  # "mock" hoặc "llm" theo biến môi trường LAB_MODE
    examples = load_dataset(dataset)
    if limit > 0:
        examples = examples[:limit]
    react = ReActAgent()
    reflexion = ReflexionAgent(max_attempts=reflexion_attempts, adaptive=adaptive)

    print(f"[cyan]Mode:[/cyan] {mode}  [cyan]Examples:[/cyan] {len(examples)}  [cyan]Reflexion attempts:[/cyan] {reflexion_attempts} (adaptive={adaptive})  [cyan]Workers:[/cyan] {workers}")
    react_records = _run_all(react, examples, "ReAct    ", workers)
    reflexion_records = _run_all(reflexion, examples, "Reflexion", workers)
    all_records = react_records + reflexion_records

    out_path = Path(out_dir)
    save_jsonl(out_path / "react_runs.jsonl", react_records)
    save_jsonl(out_path / "reflexion_runs.jsonl", reflexion_records)
    report = build_report(all_records, dataset_name=Path(dataset).name, mode=mode)
    json_path, md_path = save_report(report, out_path)
    print(f"[green]Saved[/green] {json_path}")
    print(f"[green]Saved[/green] {md_path}")
    print(json.dumps(report.summary, indent=2))

if __name__ == "__main__":
    app()

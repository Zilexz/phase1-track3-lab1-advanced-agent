# Lab 16 — Reflexion Agent · Bài làm

Tài liệu này tóm tắt những gì đã hoàn thiện và cách chạy.

## Đã làm gì (theo 5 bước trong README)

1. **Schemas** (`src/reflexion_lab/schemas.py`)
   - `JudgeResult`: `score (0/1)`, `reason`, `missing_evidence[]`, `spurious_claims[]`, `confidence` → structured evaluator.
   - `ReflectionEntry`: `attempt_id`, `failure_reason`, `lesson`, `next_strategy`.
   - Thêm `LLMUsage` (prompt/completion/total tokens + latency_ms) để đo chi phí THỰC.
2. **Reflexion loop** (`src/reflexion_lab/agents.py`)
   - Khi sai và còn lượt: gọi `reflector()`, đẩy `next_strategy` vào `reflection_memory` cho lần sau.
   - `token_estimate`/`latency_ms` hardcoded đã được thay bằng số đo thật cộng dồn từ `LLMUsage` của từng call.
   - Thêm bộ phân loại `classify_failure()` → suy ra failure_mode từ kết quả chấm + lịch sử.
3. **Prompts** (`src/reflexion_lab/prompts.py`): viết `ACTOR_SYSTEM`, `EVALUATOR_SYSTEM` (JSON), `REFLECTOR_SYSTEM` (JSON).
4. **Thay Mock bằng LLM thật**
   - `llm_client.py`: client OpenAI-compatible, mặc định trỏ **FPT AI Marketplace / GLM-4.7**.
   - `llm_runtime.py`: `actor_answer` / `evaluator` / `reflector` thật, parse JSON chịu lỗi, có fallback an toàn.
   - `runtime.py`: dispatcher chọn backend theo `LAB_MODE` (`mock` mặc định / `llm`).
5. **Token/latency thật**: lấy từ `response.usage` + đo `time.perf_counter()` trong `llm_client.chat()`.

## Bonus extensions (đã triển khai thật, không chỉ khai báo)
- `structured_evaluator` — `JudgeResult` có missing_evidence/spurious_claims/confidence.
- `reflection_memory` — vòng lặp Reflexion dùng bộ nhớ phản chiếu.
- `adaptive_max_attempts` — ngân sách lần thử theo độ khó (`easy`≤2, `hard`+1) trong `agents._effective_max_attempts`.
- `memory_compression` — `agents.compress_memory()` khử trùng lặp + cắt bộ nhớ khi phình.
- `benchmark_report_json` — `report.json` + `report.md`.
- `mock_mode_for_autograding` — chạy & chấm không cần API key.

## Dữ liệu test
Có HAI nguồn dữ liệu:

1. **`make_dataset.py`** → `data/benchmark.json` (**147 ví dụ** self-contained: 8 hotpot gốc + nhiều họ câu hỏi 2-hop + 19 câu HARD có distractor/phủ định/so sánh). 147 × 2 agent = **294 records**.
2. **`fetch_hotpot.py`** → `data/hotpot_real.json` (**150 câu HotpotQA distractor THẬT** từ HuggingFace, mỗi câu ~10 đoạn context nhiễu).

> **Phát hiện thực nghiệm quan trọng:** trên dữ liệu self-contained (context chỉ chứa đúng fact cần), MỌI model hiện đại (kể cả gpt-oss-20b) đều đạt EM≈1.0 — đó là bài đọc hiểu, không có gì để reflect. Lỗi đa dạng (entity_drift, incomplete_multi_hop, ...) chỉ xuất hiện khi (a) dùng HotpotQA thật có nhiễu, và/hoặc (b) cho **actor yếu hơn critic**. Vì vậy bản LLM thật chạy trên `hotpot_real.json` với cấu hình critic-mạnh-hơn-actor.

### Cấu hình model theo vai trò (trong `.env`)
- `LLM_ACTOR_MODEL` — model trả lời (đặt model yếu để có lỗi, vd `gpt-oss-20b`).
- `LLM_EVAL_MODEL` / `LLM_REFLECT_MODEL` — model chấm & phản chiếu (đặt model mạnh, vd `GLM-4.7`).
- Bỏ trống → dùng `LLM_MODEL`.

## Cách chạy

### A. Chế độ mock (không cần API key) — để kiểm tra flow & autograde
```bash
python -m venv .venv && source .venv/Scripts/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt

python make_dataset.py                                   # tạo data/benchmark.json (127 ví dụ)
LAB_MODE=mock python run_benchmark.py --dataset data/benchmark.json --out-dir outputs/mock_run
python autograde.py --report-path outputs/mock_run/report.json     # -> 100/100
```

### B. Chế độ LLM thật (FPT) trên HotpotQA thật
```bash
cp .env.example .env          # điền LLM_API_KEY = key FPT; đặt LAB_MODE=llm
# (tuỳ chọn) đặt actor yếu + critic mạnh trong .env:
#   LLM_ACTOR_MODEL=gpt-oss-20b   LLM_EVAL_MODEL=GLM-4.7   LLM_REFLECT_MODEL=GLM-4.7

python fetch_hotpot.py --n 150 --out data/hotpot_real.json     # tải dữ liệu thật

# Thử nhanh trước (tiết kiệm quota):
python run_benchmark.py --dataset data/hotpot_real.json --out-dir outputs/llm_smoke --limit 5

# Chạy đầy đủ (8 luồng song song):
python run_benchmark.py --dataset data/hotpot_real.json --out-dir outputs/llm_run --workers 8
python autograde.py --report-path outputs/llm_run/report.json
```

### C. Golden Test Set (cuối buổi)
```bash
python run_benchmark.py --dataset data/<golden_set>.json --out-dir outputs/golden
```
> Agent đọc dataset theo schema `QAExample`. Đảm bảo file golden đúng format (qid, difficulty, question, gold_answer, context[{title,text}]).
```

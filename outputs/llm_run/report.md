# Lab 16 Benchmark Report

## Metadata
- Dataset: hotpot_real.json
- Mode: llm
- Records: 300
- Agents: react, reflexion

## Summary
| Metric | ReAct | Reflexion | Delta |
|---|---:|---:|---:|
| EM | 0.8533 | 0.9867 | 0.1334 |
| Avg attempts | 1 | 1.2067 | 0.2067 |
| Avg token estimate | 2933.75 | 3983.87 | 1050.12 |
| Avg latency (ms) | 15942.95 | 22080.31 | 6137.36 |

## Failure modes
```json
{
  "none": {
    "total": 276,
    "react": 128,
    "reflexion": 148
  },
  "incomplete_multi_hop": {
    "total": 15,
    "react": 14,
    "reflexion": 1
  },
  "entity_drift": {
    "total": 6,
    "react": 6,
    "reflexion": 0
  },
  "wrong_final_answer": {
    "total": 2,
    "react": 2,
    "reflexion": 0
  },
  "looping": {
    "total": 1,
    "react": 0,
    "reflexion": 1
  }
}
```

## Extensions implemented
- structured_evaluator
- reflection_memory
- adaptive_max_attempts
- memory_compression
- benchmark_report_json
- mock_mode_for_autograding

## Discussion
Across 300 runs, ReAct reached EM=0.8533 while Reflexion reached EM=0.9867 (delta 0.1334). The accuracy gain costs extra attempts (0.2067 more on average), ~1050.12 extra tokens and ~6137.36 ms extra latency per item — the classic accuracy/cost trade-off of self-reflection. Reflexion mainly helped on failure modes ['incomplete_multi_hop', 'entity_drift', 'wrong_final_answer']: the reflection memory let the actor complete the second hop or correct a drifted entity it had gotten wrong on the first try. Failure modes still present after reflection were ['incomplete_multi_hop', 'looping']; these are cases where the evaluator's feedback was too vague, the context lacked the needed bridge fact, or the agent looped on the same wrong entity. Adaptive attempt budgeting (more tries for hard items) and memory compression kept the cost of the extra attempts bounded. The main limiter on further gains is evaluator quality: when grading is noisy, reflections inherit that noise and can occasionally overfit a previously correct answer.

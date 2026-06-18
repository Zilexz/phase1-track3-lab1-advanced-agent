# Lab 16 Benchmark Report

## Metadata
- Dataset: benchmark.json
- Mode: mock
- Records: 294
- Agents: react, reflexion

## Summary
| Metric | ReAct | Reflexion | Delta |
|---|---:|---:|---:|
| EM | 0.9728 | 1.0 | 0.0272 |
| Avg attempts | 1 | 1.0272 | 0.0272 |
| Avg token estimate | 464 | 483.7 | 19.7 |
| Avg latency (ms) | 240 | 250.34 | 10.34 |

## Failure modes
```json
{
  "none": {
    "total": 290,
    "react": 143,
    "reflexion": 147
  },
  "entity_drift": {
    "total": 2,
    "react": 2,
    "reflexion": 0
  },
  "incomplete_multi_hop": {
    "total": 1,
    "react": 1,
    "reflexion": 0
  },
  "wrong_final_answer": {
    "total": 1,
    "react": 1,
    "reflexion": 0
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
Across 294 runs, ReAct reached EM=0.9728 while Reflexion reached EM=1.0 (delta 0.0272). The accuracy gain costs extra attempts (0.0272 more on average), ~19.7 extra tokens and ~10.34 ms extra latency per item — the classic accuracy/cost trade-off of self-reflection. Reflexion mainly helped on failure modes ['entity_drift', 'incomplete_multi_hop', 'wrong_final_answer']: the reflection memory let the actor complete the second hop or correct a drifted entity it had gotten wrong on the first try. Failure modes still present after reflection were ['(none)']; these are cases where the evaluator's feedback was too vague, the context lacked the needed bridge fact, or the agent looped on the same wrong entity. Adaptive attempt budgeting (more tries for hard items) and memory compression kept the cost of the extra attempts bounded. The main limiter on further gains is evaluator quality: when grading is noisy, reflections inherit that noise and can occasionally overfit a previously correct answer.

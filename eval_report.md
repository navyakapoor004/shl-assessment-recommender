# Evaluation Report

## Summary Metrics (computed from live /chat responses)

| Evaluation Metric | Result |
|---|---|
| Traces Evaluated | 10 |
| End-to-End Completion | 10/10 |
| Schema Compliance | 38/38 |
| Catalog Grounding | not measured (no catalog.json supplied) |
| Hallucination Rate | not measured (no catalog.json supplied) |
| end_of_conversation Flag Accuracy | 0/10 |
| Avg Latency | 10.11 sec |
| P95 Latency | 16.75 sec |
| Recall@10 | NOT REPORTED - no ground-truth relevance labels available in traces |

## Per-Trace Detail

| Trace | Turns | Completed | Schema OK | Recs Returned | Errors |
|---|---|---|---|---|---|
| C1 | 4 | ✅ | 4/4 | 6 | - |
| C10 | 3 | ✅ | 3/3 | 11 | - |
| C2 | 3 | ✅ | 3/3 | 3 | - |
| C3 | 5 | ✅ | 5/5 | 1 | - |
| C4 | 3 | ✅ | 3/3 | 6 | - |
| C5 | 3 | ✅ | 3/3 | 7 | - |
| C6 | 3 | ✅ | 3/3 | 7 | - |
| C7 | 4 | ✅ | 4/4 | 7 | - |
| C8 | 3 | ✅ | 3/3 | 2 | - |
| C9 | 7 | ✅ | 7/7 | 5 | - |
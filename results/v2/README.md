# eval_v2 metrics (n=133)

| File | Method | Accuracy | Macro-F1 |
|------|--------|----------|----------|
| `baselines_majority.json` | Majority | 0.3083 | 0.1178 |
| `baselines_keyword.json` | Keyword | 0.5789 | 0.5302 |
| `baselines_prompt_rubric.json` | Prompt-rubric sim.† | 0.7744 | 0.7658 |
| `minilm_v2.json` | FT MiniLM | 0.8872 | 0.8829 |
| `distilbert_v2.json` | FT DistilBERT | **0.9699** | **0.9692** |

†Not an LLM API — deterministic LABELING.md rubric as weighted cues.
CI: `*_v2_ci.json` (bootstrap + stratified 5-fold).

# Measured metrics (paper tables)

Committed JSON only — no fabricated numbers. Large paraphrase-consistency dumps omitted from this public companion; scalar paraphrase agreements appear inside `ood/summary.json`.

| Path | Contents |
|------|----------|
| [`v2/`](v2/) | eval_v2 baselines + MiniLM/DistilBERT (+ bootstrap/k-fold CI) |
| [`ood/`](ood/) | OOD baselines + FT metrics + `summary.json` gap table |
| [`significance/`](significance/) | Paired McNemar / bootstrap significance on eval_v2 (cited in paper) |

Reproduce cheap baselines:

```bash
pip install -e .
python scripts/validate_dataset.py --train data/routing/train_v2.jsonl --eval data/routing/eval_v2.jsonl
python scripts/train_router_local.py --baseline majority --train data/routing/train_v2.jsonl --eval data/routing/eval_v2.jsonl --results-dir results/v2
python scripts/train_router_local.py --baseline keyword --train data/routing/train_v2.jsonl --eval data/routing/eval_v2.jsonl --results-dir results/v2
```

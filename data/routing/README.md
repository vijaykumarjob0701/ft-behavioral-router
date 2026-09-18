# Routing dataset (paper splits)

Minimal JSONL used in the arXiv short paper:

| File | Role | n |
|------|------|--:|
| `train_v2.jsonl` | Training | 526 |
| `eval_v2.jsonl` | In-domain eval | 133 |
| `eval_ood.jsonl` | Hard OOD probe (never used for training) | 104 |

Schema: [`schema.md`](schema.md). Label policy: [`LABELING.md`](LABELING.md).

Classifier target field: **`primary_action`** ∈ {`answer_small`, `rag`, `tools`, `escalate_large`}.

Synthetic workplace-style English for research; not production logs.

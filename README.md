# FT-Behavioural Router (public paper companion)

**Author:** Vijay Kumar · Dublin, Ireland · [Email](mailto:vijaykumarjob0701@gmail.com) · [LinkedIn](https://www.linkedin.com/in/vijay-kumar-59b28b4a/) · [GitHub](https://github.com/vijaykumarjob0701/ft-behavioral-router)  
**License:** MIT · **Pages:** ~7 (arXiv short)

Public, paper-only companion for the arXiv preprint. A **fine-tuned small model** decides *how* to handle each query—answer alone, retrieve then answer, call tools, or escalate to a large LLM—before expensive generation. Knowledge stays in RAG/tools; the FT router owns the **behavioural policy**.

| | Links |
|--|--|
| Paper PDF | [preprint.pdf](preprint.pdf) |
| Extended technical report | [extended/thesis.pdf](extended/thesis.pdf) |
| Cite | [CITATION.cff](CITATION.cff) |

---

## What / why

Hybrid assistants mix RAG, tools, and several model sizes. Calling a large LLM with retrieval on every turn is costly. This work studies an FT-first **4-way** router:

| Action | Meaning |
|--------|---------|
| `answer_small` | Small FT model answers alone |
| `rag` | Retrieve, then the **same** small FT model writes the answer (FT+RAG) |
| `tools` | Invoke tools / APIs |
| `escalate_large` | Delegate hard reasoning to a large LLM (`needs_rag` ⇒ large + RAG) |

---

## Originality fence

> **Do not claim inventing adaptive RAG or LLM cascading.** Those lines already exist (Adaptive-RAG, SlimPLM, Self-RAG, RouteLLM, HybridLLM, FrugalGPT, …).

**This repo’s niche:**

1. A **unified 4-way behavioural policy** (`answer_small | rag | tools | escalate_large`)
2. Explicit **knowledge ≠ behaviour** (facts in retrieval/tools; policy in the FT router)
3. An empirical **when-is-FT-worth-it** look vs strong rule/prompt-rubric baselines under domain shift

---

## Key measured results

### Phase 1 — routing classification (`eval_v2`, n=133)

| Method | Accuracy | Macro-F1 |
|--------|----------|----------|
| Majority baseline | 0.3083 | 0.1178 |
| Keyword heuristic | 0.5789 | 0.5302 |
| Prompt-rubric simulated† | 0.7744 | 0.7658 |
| FT MiniLM | 0.8872 | 0.8829 |
| FT DistilBERT | **0.9699** | **0.9692** |

†**Not an LLM API** — deterministic labeling rubric as weighted cues.  
Artefacts: [`results/v2/`](results/v2/). DistilBERT bootstrap CI ≈ 0.9692±0.0152; stratified 5-fold ≈ 0.9105±0.0217.

### Hard OOD honesty probe (`eval_ood`, n=104)

Grown v2 labels can be partly cue-easy. `eval_ood` stresses ambiguity, multi-intent, tools-vs-rag paraphrases, and keyword traps. **Never used for training.**

| Method | Acc (eval_v2) | Acc (eval_ood) | Δ Acc |
|--------|---------------|----------------|-------|
| Majority | 0.3083 | 0.2596 | −0.0487 |
| Keyword | 0.5789 | 0.3654 | −0.2136 |
| Prompt-rubric† | 0.7744 | 0.5000 | −0.2744 |
| FT MiniLM | 0.8872 | 0.7885 | −0.0988 |
| FT DistilBERT | 0.9699 | 0.8750 | −0.0949 |

Details: [`results/ood/summary.json`](results/ood/summary.json). Paired significance on eval_v2: [`results/significance/`](results/significance/).

### Phase 2 — cost/latency **simulation** (caveat)

Phase-2 simulation details and illustrative unit-cost assumptions live in the [extended technical report](extended/thesis.pdf). They are not a cloud bill or a claim about live LLM spend; this public companion does not publish separate Phase-2 JSON artefacts.

---

## Quick start (minimal reproduce)

```bash
git clone https://github.com/vijaykumarjob0701/ft-behavioral-router.git
cd ft-behavioral-router
python3 -m venv .venv && source .venv/bin/activate
pip install -U pip && pip install -e .

python scripts/validate_dataset.py \
  --train data/routing/train_v2.jsonl --eval data/routing/eval_v2.jsonl
python scripts/train_router_local.py --baseline majority \
  --train data/routing/train_v2.jsonl --eval data/routing/eval_v2.jsonl \
  --results-dir results/v2
python scripts/train_router_local.py --baseline keyword \
  --train data/routing/train_v2.jsonl --eval data/routing/eval_v2.jsonl \
  --results-dir results/v2

# Optional FT training
pip install -e ".[train]"
python scripts/train_router_local.py --config configs/router_distilbert_v2.yaml
```

Label field: **`primary_action`** — see [`data/routing/schema.md`](data/routing/schema.md).

The paper sources are not included in this public repository; use [preprint.pdf](preprint.pdf).

---

## Repo map

```
ft-behavioral-router/
├── README.md
├── LICENSE                 # MIT
├── CITATION.cff
├── preprint.pdf            # paper PDF
├── extended/thesis.pdf     # longer technical report
├── data/routing/           # schema + train_v2 / eval_v2 / eval_ood
├── results/{v2,ood,significance}/   # measured JSON for paper tables
├── scripts/                # validate + train_router_local
├── src/fta_router/         # schema, baselines, metrics
```

---

## How to cite

```bibtex
@misc{kumar2026ftbehavioral,
  title        = {FT-Behavioural Router: Knowledge vs Behaviour Separation in Adaptive Hybrid Inference},
  author       = {Kumar, Vijay},
  year         = {2026},
  howpublished = {\url{https://github.com/vijaykumarjob0701/ft-behavioral-router}},
  note         = {Preprint / technical report}
}
```

Or use [`CITATION.cff`](CITATION.cff).

---

## License

MIT — see [LICENSE](LICENSE).

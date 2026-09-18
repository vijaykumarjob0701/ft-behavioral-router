# Labeling rules (Phase-1 v2)

This document is the **annotator-facing** companion to [`schema.md`](schema.md).
All v2 labels in `train_v2.jsonl` / `eval_v2.jsonl` were authored to these rules.

## Goal

Label each workplace-style query with a single **`primary_action`** in
`{answer_small, rag, tools, escalate_large}` plus aux flags, so a small FT
encoder can learn a **behavioural policy** (not end-task QA answers).

## Decision order (mandatory)

1. **Is the user asking to *do* something via an API/tool?** (create/update/check live state) → `tools` (`needs_tools=true`).
2. Else **is the bottleneck missing company/time-sensitive facts?** → `rag` (`needs_rag=true`).
   This is **FT+RAG**: retrieve, then the **small FT model** writes the answer. Do not send this path to the large LLM.
3. Else **is the bottleneck hard multi-step reasoning/planning** (even with perfect docs)? → `escalate_large` (set `needs_rag` / `needs_tools` if secondaries apply).
   If facts are also needed, `needs_rag=true` means **large + RAG**, not the small FT+RAG path.
4. Else → `answer_small` (parametric / format / greeting / mild refusal).

When facts **and** deep analysis are both required, prefer
`escalate_large` + `needs_rag=true` (large + RAG; and `needs_tools=true` if a side-effect is also requested).

## Class-specific cues

| Class | Positive cues | Negative cues (usually not this class) |
|-------|---------------|----------------------------------------|
| `answer_small` | definitions, idioms, generic how-tos, greetings, mild safety refusals | "our/we", "yesterday", create/file/schedule, long tradeoff plans |
| `rag` | our/we/company, yesterday/last week, runbook, SLA, release notes, wiki/Confluence find. Small FT answers from retrieved docs (FT+RAG). | explicit create/restart/deploy side effects; deep design without a fact gap |
| `tools` | create/open/file/schedule/comment/restart/deploy/merge/page/mute + system names | "how do I create a Jira?" (docs → rag/answer_small); pure analysis |
| `escalate_large` | tradeoffs, design a plan, root-cause synthesis, multi-constraint, compare&recommend | single-hop fact lookup; single tool call |

## Aux flags & reasoning_level

- `needs_rag` / `needs_tools`: secondary capabilities on the chosen primary path.
- Invariants: `primary_action=rag` ⇒ `needs_rag=true`; `primary_action=tools` ⇒ `needs_tools=true`.
- `reasoning_level`: `low` single-hop; `medium` light comparison/procedure; `high` multi-constraint / RCA / design.
- `answer_small` should rarely be `high`.

## Domains

Use exactly: `general_knowledge`, `company_current`, `tool_action`,
`multi_hop_planning`, `ambiguous`, `safety_mild`.

- **ambiguous**: borderline parametric vs company (still pick one primary).
- **safety_mild**: refusal-capable routing; **never** label harmful intent as `tools`.

## Paraphrase pairs

File: [`paraphrase_pairs_v2.json`](paraphrase_pairs_v2.json) (24 pairs).

- Both sides share the **same** `primary_action`.
- Used for **route-consistency** analysis (agreement of predicted labels on A vs B).
- Do not place near-duplicates across train/eval except intentional paraphrase B-sides documented there.

## Quality checklist

- [ ] Exactly one `primary_action`
- [ ] Invariants on `needs_*` hold
- [ ] Rationale is one sentence naming the decisive cue
- [ ] Domain tag set
- [ ] No accidental train/eval near-duplicates outside paraphrase file
- [ ] Class balance kept roughly within ~2:1 of the largest class
- [ ] Mild safety never routed to tools

## Validation

```bash
python scripts/validate_dataset.py \
  --train data/routing/train_v2.jsonl \
  --eval data/routing/eval_v2.jsonl
```


## Hard OOD eval split (`eval_ood.jsonl`)

**Purpose.** DistilBERT’s ~97% accuracy on in-domain `eval_v2` may partly reflect
template-regular, cue-aligned labels. `eval_ood.jsonl` is a **held-out hardness
probe** (not used for training or model selection) that stresses failure modes
the grown v2 labels under-represent.

**Construction principles (2026-09-17).**

1. **No train/eval_v2 near-duplicates** — every query string is unique vs
   `train_v2` / `eval_v2` (exact normalised match check at authoring time).
2. **Same schema & decision order** as above; labels follow LABELING.md, not
   “whatever would fool the model.”
3. **Stress categories** (encoded in rationale as `[ood:<tag>] …`):
   - `ambiguity` — parametric vs company borderline (general concept vs *our* process).
   - `multi_intent` — RAG/tools/escalate combinations where synthesis is the bottleneck
     (`escalate_large` + aux flags).
   - `tools_vs_rag` — paired paraphrases: docs/how-to/history (**rag**) vs live
     side-effect/API (**tools**) with overlapping lexical items (restart, Jira, page, deploy, Slack, …).
   - `entity_heavy` — long service/incident/RFC/ticket identifiers that still need
     company retrieval or a precise tool call.
   - `keyword_trap` — adversarial cues: *create/deploy/plan/recommend/compare/our/yesterday/Jira*
     appearing in contexts where the correct primary is **not** the cue’s usual class
     (e.g. “How do I create a Jira…?” → `answer_small`; escalate RCA without “tradeoff/design a”).
4. **Mild safety** examples remain refusal-capable `answer_small`; never `tools`.
5. **Balance** — primary-action counts kept within ~2:1 of the largest class;
   target size **80–120** (authored **n=104**).
6. **Not training data** — do not fine-tune or select checkpoints on `eval_ood`.
   Train on `train_v2` only (optional early-stop on `eval_v2`); report OOD as
   generalisation / hardness metrics under `results/ood/`.

**Validate OOD:**

```bash
python scripts/validate_dataset.py \
  --train data/routing/train_v2.jsonl \
  --eval data/routing/eval_ood.jsonl
```

**Honest reading.** A large in-domain → OOD accuracy drop is a **feature** for
submission honesty: it flags possible label simplicity / lexical shortcut learning
on `eval_v2`, not a silent success.

## ACL public remap (second domain)

Workplace rules above still apply to `train_v2` / `eval_v2` / `eval_ood`.
A separate **deterministic** remap of SQuAD 1.1 + HotpotQA + CLINC-150 lives
are documented in the private research checkout (not shipped here). Those labels are **not** human IAA.
Do not mix remapped rows into `train_v2`.

## Version

- Schema: 1.0 (unchanged field set)
- Splits: `train_v2` / `eval_v2` (seed rows retained as prefix + curated expansions)
- Hardness probe: `eval_ood` (n≈104; stress-tagged rationales; no train overlap)

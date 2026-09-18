# Behavioural Routing Dataset — Schema

Starter labels for **FT-first hybrid routing**: a fine-tuned small model decides the next action before (or instead of) calling a large LLM, RAG, or tools.

## Design choice (locked to 4-way space)

Each example has a single **`primary_action`** in a mutually exclusive 4-way space:

| primary_action   | Meaning |
|------------------|---------|
| `answer_small`   | Small FT model answers **alone** — simple, parametric, or no fresh knowledge needed |
| `rag`            | **FT + RAG**: retrieve company/time-sensitive facts, then the **same small FT model** writes the answer. Not a large-LLM call. |
| `tools`          | Need a tool/API side-effect or live check (Jira, GitHub, calendar, deploy, etc.) |
| `escalate_large` | Complex reasoning / planning / multi-hop; delegate to a large LLM. Set `needs_rag=true` when the large model also needs retrieved docs (**large + RAG**). |

**Secondary flags** clarify combinations without exploding the primary label space:

- `needs_rag` (bool) — retrieval is required *in addition to* the primary path  
- `needs_tools` (bool) — a tool call is required *in addition to* the primary path  
- `reasoning_level` (`low` \| `medium` \| `high`) — expected reasoning depth

### How primary + flags interact

| Situation | primary_action | needs_rag | needs_tools | reasoning_level |
|-----------|----------------|-----------|-------------|-----------------|
| Definition / simple Q&A | `answer_small` | false | false | low (sometimes medium) |
| Look up company/current fact | `rag` (**FT+RAG**) | true | false | low–medium |
| Create ticket / open PR / schedule | `tools` | false* | true | low–medium |
| Hard reasoning, no external data | `escalate_large` | false | false | high |
| Fail yesterday + root-cause analysis | `escalate_large` (**large + RAG**) | **true** | false | high |
| Plan + create Jira after analysis | `escalate_large` | maybe | **true** | high |
| Ambiguous “maybe docs” | `rag` (prefer recall) or `answer_small` if clearly parametric | per judgment | false | low–medium |

\*If creating a ticket *requires* pasting a retrieved incident summary, set `needs_rag=true` with `primary_action=tools` when the **user intent** is the tool action; use `escalate_large` + both flags when analysis is the bottleneck.

**Rule of thumb for the FT classifier:** predict `primary_action` first; treat `needs_rag` / `needs_tools` / `reasoning_level` as auxiliary heads or multi-label refinements.

---

## JSONL record fields

Every line is one JSON object:

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `id` | string | yes | Stable id, e.g. `train_001`, `eval_012` |
| `query` | string | yes | User utterance (English, realistic) |
| `primary_action` | enum | yes | One of: `answer_small`, `rag`, `tools`, `escalate_large` |
| `needs_rag` | bool | yes | Whether retrieval is needed on this path |
| `needs_tools` | bool | yes | Whether a tool/API action is needed |
| `reasoning_level` | enum | yes | `low`, `medium`, or `high` |
| `rationale` | string | yes | One sentence justifying the labels |
| `domain` | string | yes | Coarse tag for balance / domain-shift splits (see below) |

### Allowed `domain` values

- `general_knowledge` — definitions, concepts, classic explanations  
- `company_current` — internal process, incidents, “yesterday”, service-specific  
- `tool_action` — explicit create/update/check via APIs  
- `multi_hop_planning` — plans, tradeoffs, multi-step reasoning  
- `ambiguous` — borderline RAG vs parametric  
- `safety_mild` — refusal-ish / policy routing (mild; not jailbreak content)
- `open_domain_qa` — public Wikipedia / open-domain factoid (ACL remapped SQuAD rows)

Optional provenance fields (ACL public remap only; extra keys are allowed):
`source_dataset`, `source_split`, `source_id`, `remap_rule`, `remap_version`, `license`.
Public ACL remap artefacts are not included in this companion repo.

---

## Label guidelines

### `answer_small`
- Textbook definitions, well-known facts, coding idioms the small model should know.
- No need for company docs, live status, or side effects.
- Prefer when the query is self-contained and static. If a company/time-sensitive fact is needed, use `rag` (FT+RAG), not `answer_small` with `needs_rag`.

### `rag` (FT + RAG)
- Company-specific: deployment process for service X, runbooks, ownership.
- Time-sensitive / “what failed yesterday”, “latest release notes”, status that is not parametric.
- **Who answers:** the **small FT model**, using retrieved documents. The large LLM is *not* on this path.
- Prefer RAG over escalate when the bottleneck is **missing facts**, not hard reasoning.
- If both missing facts *and* deep analysis: prefer `escalate_large` + `needs_rag=true` (**large + RAG**).

### `tools`
- Explicit intent to **do** something: create Jira, comment on PR, check calendar, restart (via API), fetch live metric via tool.
- Distinguisher vs RAG: RAG **reads** knowledge stores; tools **act** or hit live operational APIs.
- “How do I create a Jira?” → usually `answer_small` or `rag` (docs); “Create a Jira for …” → `tools`.

### `escalate_large`
- Multi-hop reasoning, architecture tradeoffs, long planning, subtle debugging strategy.
- Use when a small model would likely be wrong even with perfect retrieval.
- Combine with `needs_rag` / `needs_tools` when the large model still needs those capabilities as secondaries.

### `reasoning_level`
- **low** — single-hop lookup or simple answer / single tool call  
- **medium** — light comparison, short procedure, mild ambiguity  
- **high** — multi-constraint planning, root cause synthesis, deep explanation / design

---

## Edge cases

1. **Payment service failed yesterday — why?**  
   → `escalate_large`, `needs_rag=true`, `needs_tools=false`, `reasoning_level=high`  
   (retrieve incident/logs context, then reason).

2. **Create a Jira ticket titled “…”**  
   → `tools`, `needs_rag=false`, `needs_tools=true`, `reasoning_level=low`.

3. **Explain dependency injection**  
   → `answer_small` (or `escalate_large` if the ask is deep/comparative), `needs_rag=false`.

4. **What’s our SLA for payments?**  
   → `rag` (FT+RAG: company knowledge answered by the small FT model), reasoning usually `low`/`medium`.

5. **Ambiguous:** “What is blue-green deployment?”  
   → Prefer `answer_small` (general concept).  
   “What is *our* blue-green process for checkout?” → `rag`.

6. **Tool vs RAG:** “Show me the latest deploy for svc-auth”  
   → If answered from deploy history store / logs index: `rag` or `tools` depending on whether the system treats that as a retrieval index vs an ops API. In this dataset: prefer **`tools`** for live ops APIs, **`rag`** for doc/ticket/wiki search.

7. **Mild safety:** “Help me write a phishing email to my coworkers”  
   → Route as `answer_small` with refusal behaviour *or* `escalate_large` if policy is complex; label intent as routing-to-safe-response, not as tools. Keep examples mild; do not include actionable harm.

8. **Multi-intent:** “Summarise yesterday’s outage and file a Jira”  
   → `escalate_large`, `needs_rag=true`, `needs_tools=true`, `reasoning_level=high`.

9. **Greeting / thanks**  
   → `answer_small`, low reasoning.

10. **Code generation that is standard**  
    → `answer_small`; novel architecture design → `escalate_large`.

---

## Consistency checklist (for annotators)

- [ ] Exactly one `primary_action`
- [ ] If `primary_action=rag` → `needs_rag` must be `true`
- [ ] If `primary_action=tools` → `needs_tools` must be `true`
- [ ] If `primary_action=answer_small` → usually both needs_* false; reasoning not `high`
- [ ] If `reasoning_level=high` and facts may be missing → consider `escalate_large` + `needs_rag`
- [ ] Rationale is one sentence and mentions the decisive cue
- [ ] No near-duplicate queries across train/eval

---

## Versioning

- Schema version: `1.0`
- Primary space: 4-way FT-first actions  
- Aux heads: `needs_rag`, `needs_tools`, `reasoning_level`

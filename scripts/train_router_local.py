#!/usr/bin/env python3
"""Train a local MiniLM/DistilBERT primary_action classifier (optional torch/transformers)."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parents[1]
_SRC = _REPO / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description=(
            "Train an FT behavioural router on primary_action labels. "
            "Requires optional [train] extras (torch, transformers)."
        )
    )
    p.add_argument(
        "--config",
        type=Path,
        default=_REPO / "configs" / "router_minilm.yaml",
        help="YAML config path",
    )
    p.add_argument("--train", type=Path, default=None, help="Override train JSONL")
    p.add_argument("--eval", type=Path, default=None, help="Override eval JSONL")
    p.add_argument("--model", type=str, default=None, help="Override HF model name")
    p.add_argument("--epochs", type=int, default=None, help="Override num epochs")
    p.add_argument("--output-dir", type=Path, default=None, help="Override output dir")
    p.add_argument(
        "--dry-run",
        action="store_true",
        help="Load data + config, print summary, exit without training",
    )
    p.add_argument(
        "--baseline",
        choices=["majority", "keyword", "prompt_rubric", "none"],
        default="none",
        help="If set, evaluate a non-neural baseline instead of training",
    )
    p.add_argument(
        "--results-dir",
        type=Path,
        default=None,
        help="Directory for metrics JSON (default: results/ or results/v2 for v2 splits)",
    )
    p.add_argument(
        "--results-tag",
        type=str,
        default=None,
        help="Filename tag override (e.g. minilm_v2)",
    )
    return p


def load_config(path: Path) -> dict:
    import yaml

    with path.open(encoding="utf-8") as f:
        return yaml.safe_load(f)


def _infer_split_tag(eval_path: Path) -> str:
    name = eval_path.name.lower()
    if "public_remap" in name or "acl" in name:
        return eval_path.stem
    if "v2" in name:
        return "eval_v2"
    if "seed" in name:
        return "eval_seed"
    return eval_path.stem


def _default_results_dir(eval_path: Path) -> Path:
    name = eval_path.name.lower()
    if "public_remap" in name or "acl" in name:
        return _REPO / "results" / "significance"
    if "v2" in name:
        return _REPO / "results" / "v2"
    return _REPO / "results"


def run_baseline(
    name: str,
    train_path: Path,
    eval_path: Path,
    results_dir: Path | None = None,
    results_tag: str | None = None,
) -> int:
    from fta_router.baselines import available_baselines
    from fta_router.dataset import load_jsonl
    from fta_router.metrics import format_confusion, routing_metrics

    train = load_jsonl(train_path)
    eval_rows = load_jsonl(eval_path)
    cls = available_baselines()[name]
    model = cls()
    if hasattr(model, "fit"):
        model.fit([r["primary_action"] for r in train])
    preds = model.predict([r["query"] for r in eval_rows])
    y_true = [r["primary_action"] for r in eval_rows]
    metrics = routing_metrics(y_true, preds)
    print(json.dumps({k: v for k, v in metrics.items() if k != "confusion_matrix"}, indent=2))
    print(format_confusion(metrics["confusion_matrix"], metrics["labels"]))
    results_dir = Path(results_dir) if results_dir else _default_results_dir(eval_path)
    results_dir.mkdir(parents=True, exist_ok=True)
    method_map = {
        "majority": ("baselines_majority.json", "majority"),
        "keyword": ("baselines_keyword.json", "keyword_heuristic"),
        "prompt_rubric": ("baselines_prompt_rubric.json", "prompt_rubric_simulated"),
    }
    out_name, method = method_map[name]
    split = _infer_split_tag(eval_path)
    payload = {
        "method": method,
        "split": split,
        "n_eval": len(eval_rows),
        "n_train": len(train),
        "accuracy": metrics["accuracy"],
        "macro_f1": metrics["macro_f1"],
        "labels": metrics["labels"],
        "confusion_matrix": metrics["confusion_matrix"],
    }
    if name == "majority":
        payload["majority_label"] = model.label
    if name == "prompt_rubric":
        payload["honesty"] = (
            "Deterministic LABELING.md rubric simulation; NOT an LLM API prompted router."
        )
    if results_tag:
        stem = Path(out_name).stem
        out_name = f"{stem}_{results_tag}.json"
        payload["results_tag"] = results_tag
    out_path = results_dir / out_name
    out_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote {out_path}")
    return 0


def run_training(cfg: dict, args: argparse.Namespace) -> int:
    try:
        import torch  # noqa: F401
        from transformers import (  # noqa: F401
            AutoModelForSequenceClassification,
            AutoTokenizer,
            Trainer,
            TrainingArguments,
        )
    except ImportError as exc:
        print(
            "Training requires torch and transformers. Install with:\n"
            '  pip install -e ".[train]"\n'
            "Or evaluate a baseline: --baseline keyword",
            file=sys.stderr,
        )
        print(f"Import error: {exc}", file=sys.stderr)
        return 2

    from datasets import Dataset
    from sklearn.preprocessing import LabelEncoder
    from transformers import (
        AutoModelForSequenceClassification,
        AutoTokenizer,
        DataCollatorWithPadding,
        Trainer,
        TrainingArguments,
    )

    from fta_router.dataset import load_jsonl
    from fta_router.metrics import routing_metrics
    from fta_router.schema import PRIMARY_ACTIONS

    train_path = args.train or Path(cfg["data"]["train_path"])
    eval_path = args.eval or Path(cfg["data"]["eval_path"])
    if not train_path.is_absolute():
        train_path = _REPO / train_path
    if not eval_path.is_absolute():
        eval_path = _REPO / eval_path

    model_name = args.model or cfg["model_name"]
    epochs = args.epochs or cfg["training"]["num_epochs"]
    output_dir = args.output_dir or Path(cfg["output_dir"])
    if not output_dir.is_absolute():
        output_dir = _REPO / output_dir

    train_rows = load_jsonl(train_path)
    eval_rows = load_jsonl(eval_path)
    text_field = cfg["data"]["text_field"]
    label_field = cfg["data"]["label_field"]
    labels = list(cfg.get("labels") or PRIMARY_ACTIONS)

    le = LabelEncoder()
    le.fit(labels)

    def to_hf(rows):
        return Dataset.from_dict(
            {
                "text": [r[text_field] for r in rows],
                "label": le.transform([r[label_field] for r in rows]).tolist(),
            }
        )

    tok = AutoTokenizer.from_pretrained(model_name)
    max_length = cfg["training"]["max_length"]

    def tokenize(batch):
        return tok(batch["text"], truncation=True, max_length=max_length)

    train_ds = to_hf(train_rows).map(tokenize, batched=True)
    eval_ds = to_hf(eval_rows).map(tokenize, batched=True)

    model = AutoModelForSequenceClassification.from_pretrained(
        model_name, num_labels=len(labels)
    )
    # Map ids for clarity in HF config
    model.config.id2label = {i: lab for i, lab in enumerate(labels)}
    model.config.label2id = {lab: i for i, lab in enumerate(labels)}

    import inspect
    import math

    _ta_params = inspect.signature(TrainingArguments.__init__).parameters
    _strategy_key = "eval_strategy" if "eval_strategy" in _ta_params else "evaluation_strategy"
    _batch = int(cfg["training"]["batch_size"])
    _steps_per_epoch = max(1, math.ceil(len(train_rows) / _batch))
    _total_steps = max(1, _steps_per_epoch * int(epochs))
    _warmup_ratio = float(cfg["training"].get("warmup_ratio", 0.0))
    _warmup_steps = int(_total_steps * _warmup_ratio)
    _targs = dict(
        output_dir=str(output_dir),
        learning_rate=float(cfg["training"]["learning_rate"]),
        per_device_train_batch_size=_batch,
        per_device_eval_batch_size=_batch,
        num_train_epochs=int(epochs),
        weight_decay=float(cfg["training"]["weight_decay"]),
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="eval_macro_f1",
        greater_is_better=True,
        seed=int(cfg["training"]["seed"]),
        report_to=[],
    )
    if "warmup_ratio" in _ta_params:
        _targs["warmup_ratio"] = _warmup_ratio
    elif "warmup_steps" in _ta_params:
        _targs["warmup_steps"] = _warmup_steps
    _targs[_strategy_key] = "epoch"
    # Drop kwargs unsupported by this transformers version
    _targs = {k: v for k, v in _targs.items() if k in _ta_params or k == _strategy_key}
    targs = TrainingArguments(**_targs)

    def compute_metrics(eval_pred):
        logits, label_ids = eval_pred
        import numpy as np

        preds = np.argmax(logits, axis=-1)
        y_true = le.inverse_transform(label_ids).tolist()
        y_pred = le.inverse_transform(preds).tolist()
        m = routing_metrics(y_true, y_pred, labels=labels)
        return {"accuracy": m["accuracy"], "macro_f1": m["macro_f1"]}

    _trainer_kwargs = dict(
        model=model,
        args=targs,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        data_collator=DataCollatorWithPadding(tok),
        compute_metrics=compute_metrics,
    )
    import inspect as _insp
    _tp = _insp.signature(Trainer.__init__).parameters
    if "processing_class" in _tp:
        _trainer_kwargs["processing_class"] = tok
    elif "tokenizer" in _tp:
        _trainer_kwargs["tokenizer"] = tok
    trainer = Trainer(**_trainer_kwargs)
    trainer.train()
    metrics = trainer.evaluate()
    print(json.dumps(metrics, indent=2))
    trainer.save_model(str(output_dir / "best"))
    print(f"Saved model to {output_dir / 'best'}")

    # Persist eval routing metrics (measured only)
    import numpy as np

    pred_out = trainer.predict(eval_ds)
    pred_ids = np.argmax(pred_out.predictions, axis=-1)
    y_true = le.inverse_transform(pred_out.label_ids).tolist()
    y_pred = le.inverse_transform(pred_ids).tolist()
    rm = routing_metrics(y_true, y_pred, labels=labels)
    results_dir = Path(args.results_dir) if args.results_dir else _default_results_dir(eval_path)
    results_dir.mkdir(parents=True, exist_ok=True)
    split = _infer_split_tag(eval_path)
    suffix = "v2" if "v2" in split else "seed"
    if args.results_tag:
        slug = args.results_tag
    elif "minilm" in model_name.lower():
        slug = f"minilm_{suffix}"
    elif "distilbert" in model_name.lower():
        slug = f"distilbert_{suffix}"
    else:
        slug = f"neural_{suffix}"
    payload = {
        "method": f"ft_{slug.rsplit('_', 1)[0]}",
        "model_name": model_name,
        "split": split,
        "n_eval": len(eval_rows),
        "n_train": len(train_rows),
        "epochs": int(epochs),
        "seed": int(cfg["training"]["seed"]),
        "accuracy": rm["accuracy"],
        "macro_f1": rm["macro_f1"],
        "labels": rm["labels"],
        "confusion_matrix": rm["confusion_matrix"],
        "trainer_eval": {k: float(v) if hasattr(v, "item") else v for k, v in metrics.items()},
        "status": "ok",
    }
    out_path = results_dir / f"{slug}.json"
    out_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote {out_path}")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    cfg = load_config(args.config) if args.config.exists() else {}

    train_path = args.train or Path(cfg.get("data", {}).get("train_path", "data/routing/train_seed.jsonl"))
    eval_path = args.eval or Path(cfg.get("data", {}).get("eval_path", "data/routing/eval_seed.jsonl"))
    if not Path(train_path).is_absolute():
        train_path = _REPO / train_path
    if not Path(eval_path).is_absolute():
        eval_path = _REPO / eval_path

    if args.dry_run:
        from fta_router.dataset import summarise

        print("config:", args.config)
        print(json.dumps(cfg, indent=2, default=str))
        print(summarise(train_path))
        print(summarise(eval_path))
        return 0

    if args.baseline != "none":
        return run_baseline(
            args.baseline,
            Path(train_path),
            Path(eval_path),
            args.results_dir,
            args.results_tag,
        )

    return run_training(cfg, args)


if __name__ == "__main__":
    raise SystemExit(main())

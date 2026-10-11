"""
eval.py — Eval harness for the triage system.

Usage:
    python eval.py

Outputs:
    - predictions.jsonl  — predictions for the 15 held-out eval tickets
    - Printed metrics    — accuracy + confusion matrix + false-draft rate,
                           measured against the 30 labeled training tickets
"""

import json
from collections import defaultdict
from pathlib import Path

import anthropic

from triage import (
    NO_DRAFT_CATEGORIES,
    ROUTES,
    URGENCY_LEVELS,
    build_system_prompt,
    classify_ticket,
    load_jsonl,
)

TRAIN_FILE = "tickets_train.jsonl"
EVAL_FILE = "tickets_eval.jsonl"
PREDICTIONS_FILE = "predictions.jsonl"


# ── Metrics ────────────────────────────────────────────────────────────────────

def compute_metrics(predictions: list[dict], train_tickets: list[dict]) -> dict:
    labels = {t["ticket_id"]: t["label"] for t in train_tickets}

    total = cat_correct = urg_correct = draft_correct = 0
    sensitive_total = sensitive_false_drafts = 0
    routes: dict[str, int] = defaultdict(int)
    must_escalate = not_escalated = 0

    # confusion[actual][predicted] = count
    confusion: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    wrong_drafts: list[tuple[str, dict, dict]] = []

    for pred in predictions:
        tid = pred["ticket_id"]
        if tid not in labels:
            continue
        label = labels[tid]
        total += 1

        if pred["category"] == label["category"]:
            cat_correct += 1
        if pred["urgency"] == label["urgency"]:
            urg_correct += 1
        if pred["should_draft"] == label["should_draft"]:
            draft_correct += 1

        confusion[label["urgency"]][pred["urgency"]] += 1

        routes[pred["route"]] += 1
        # Gold says this ticket must reach a human urgently: was it escalated?
        if label["category"] in NO_DRAFT_CATEGORIES or label["urgency"] == "escalate_immediately":
            must_escalate += 1
            if pred["route"] != "escalate":
                not_escalated += 1

        # false-draft: gold says NO, model says YES
        if not label["should_draft"]:
            sensitive_total += 1
            if pred["should_draft"]:
                sensitive_false_drafts += 1
                wrong_drafts.append((tid, label, pred))

    return {
        "total": total,
        "category_accuracy": cat_correct / total if total else 0.0,
        "urgency_accuracy": urg_correct / total if total else 0.0,
        "draft_accuracy": draft_correct / total if total else 0.0,
        "confusion_matrix": {k: dict(v) for k, v in confusion.items()},
        "sensitive_total": sensitive_total,
        "false_draft_count": sensitive_false_drafts,
        "false_draft_rate": sensitive_false_drafts / sensitive_total if sensitive_total else 0.0,
        "wrong_drafts": wrong_drafts,
        "routes": dict(routes),
        "must_escalate": must_escalate,
        "not_escalated": not_escalated,
    }


def _pct(num: float) -> str:
    return f"{num:.1%}"


def print_report(metrics: dict) -> None:
    n = metrics["total"]
    sep = "=" * 68

    print(f"\n{sep}")
    print("TRIAGE EVAL METRICS  (measured on 30 labeled training tickets)")
    print(sep)
    print(f"  Tickets evaluated        : {n}")
    print(f"  Category accuracy        : {_pct(metrics['category_accuracy'])}  "
          f"({int(metrics['category_accuracy'] * n)}/{n})")
    print(f"  Urgency accuracy         : {_pct(metrics['urgency_accuracy'])}  "
          f"({int(metrics['urgency_accuracy'] * n)}/{n})")
    print(f"  Draft-decision accuracy  : {_pct(metrics['draft_accuracy'])}  "
          f"({int(metrics['draft_accuracy'] * n)}/{n})")

    fd = metrics["false_draft_count"]
    ft = metrics["sensitive_total"]
    fr = metrics["false_draft_rate"]
    flag = "  ⚠️  CRITICAL FAILURES" if fd else "  ✅ CLEAN"
    print(f"\n  ★ False-draft rate on sensitive tickets: {fd}/{ft} = {_pct(fr)}{flag}")

    if metrics["wrong_drafts"]:
        print("\n  False-draft details:")
        for tid, label, pred in metrics["wrong_drafts"]:
            print(f"    {tid}: expected NO-DRAFT ({label['no_draft_reason']})")
            print(f"           got DRAFT  cat={pred['category']} urg={pred['urgency']}")

    ne, me = metrics["not_escalated"], metrics["must_escalate"]
    flag = "  ⚠️  CRITICAL FAILURES" if ne else "  ✅ CLEAN"
    print(f"  ★ Sensitive tickets not routed to escalation: {ne}/{me}{flag}")
    print("\n  Routes: " + ", ".join(f"{r}={metrics['routes'].get(r, 0)}" for r in ROUTES))

    matrix = metrics["confusion_matrix"]
    col_w = 22
    print(f"\n  Urgency Confusion Matrix  (rows=actual, cols=predicted)")
    header = f"  {'':26s}" + "".join(f"{u:{col_w}s}" for u in URGENCY_LEVELS)
    print(header)
    print("  " + "-" * (26 + col_w * len(URGENCY_LEVELS)))
    for actual in URGENCY_LEVELS:
        row = matrix.get(actual, {})
        total_in_row = sum(row.values()) if row else 0
        cells = "".join(f"{row.get(pred, 0):{col_w}d}" for pred in URGENCY_LEVELS)
        note = "" if total_in_row else "  (no samples)"
        print(f"  {actual:26s}{cells}{note}")

    print(sep)


# ── Main ───────────────────────────────────────────────────────────────────────

def main() -> None:
    client = anthropic.Anthropic()

    train_tickets = load_jsonl(TRAIN_FILE)
    eval_tickets = load_jsonl(EVAL_FILE)

    print(f"Building system prompt with {len(train_tickets)} few-shot examples...")
    system_prompt = build_system_prompt(train_tickets)

    # Step 1: run on eval set → predictions.jsonl
    print(f"\nClassifying {len(eval_tickets)} eval tickets...")
    eval_predictions: list[dict] = []
    for i, ticket in enumerate(eval_tickets, 1):
        pred = classify_ticket(client, ticket, system_prompt)
        eval_predictions.append(pred)
        if pred["should_draft"]:
            decision = "DRAFT"
        else:
            reason = (pred.get("no_draft_reason") or "")[:45]
            decision = f"NO ({reason})"
        print(
            f"  [{i:02d}/{len(eval_tickets)}] {ticket['ticket_id']}: "
            f"cat={pred['category']}  urg={pred['urgency']}  "
            f"{decision}  conf={pred['confidence']:.2f}  -> {pred['route']}"
        )

    Path(PREDICTIONS_FILE).write_text(
        "\n".join(json.dumps(p) for p in eval_predictions) + "\n"
    )
    print(f"\nWrote {len(eval_predictions)} predictions → {PREDICTIONS_FILE}")

    # Step 2: run on training set to compute metrics against gold labels
    print(f"\nClassifying {len(train_tickets)} training tickets for metric computation...")
    train_predictions: list[dict] = []
    for i, ticket in enumerate(train_tickets, 1):
        pred = classify_ticket(client, ticket, system_prompt)
        train_predictions.append(pred)
        label = ticket["label"]
        cat_sym = "✓" if pred["category"] == label["category"] else f"✗(got={pred['category']},exp={label['category']})"
        urg_sym = "✓" if pred["urgency"] == label["urgency"] else f"✗(got={pred['urgency']},exp={label['urgency']})"
        draft_sym = "✓" if pred["should_draft"] == label["should_draft"] else "✗DRAFT_ERR"
        print(
            f"  [{i:02d}/{len(train_tickets)}] {ticket['ticket_id']}: "
            f"cat {cat_sym}  urg {urg_sym}  draft {draft_sym}"
        )

    metrics = compute_metrics(train_predictions, train_tickets)
    print_report(metrics)


if __name__ == "__main__":
    main()

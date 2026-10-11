"""
test_triage.py — Unit tests for the triage module.

Run with:  python -m pytest test_triage.py -v
       or:  make test
"""

import json
import pytest

from triage import (
    CATEGORIES,
    NO_DRAFT_CATEGORIES,
    OUTPUT_SCHEMA,
    ROUTING_CONFIDENCE_THRESHOLD,
    URGENCY_LEVELS,
    _safety_override,
    load_jsonl,
    route,
)


# ── Helpers ───────────────────────────────────────────────────────────────────

def _make_pred(category, should_draft=True, draft_response="Here's a draft.", no_draft_reason=None):
    return {
        "ticket_id": "t_test_001",
        "category": category,
        "urgency": "medium",
        "should_draft": should_draft,
        "no_draft_reason": no_draft_reason,
        "draft_response": draft_response if should_draft else None,
        "confidence": 0.85,
    }


# ── Safety override ───────────────────────────────────────────────────────────

def test_safety_override_blocks_account_compromise():
    pred = _make_pred("account_compromise", should_draft=True)
    result = _safety_override(pred)
    assert result["should_draft"] is False
    assert result["draft_response"] is None
    assert "safety override" in result["no_draft_reason"]


def test_safety_override_blocks_problem_gambling():
    pred = _make_pred("problem_gambling", should_draft=True)
    result = _safety_override(pred)
    assert result["should_draft"] is False


def test_safety_override_blocks_legal_regulatory():
    pred = _make_pred("legal_regulatory", should_draft=True)
    result = _safety_override(pred)
    assert result["should_draft"] is False


def test_safety_override_passthrough_safe_category():
    # Should not touch a safe category with should_draft=True
    pred = _make_pred("account_access", should_draft=True)
    result = _safety_override(pred)
    assert result["should_draft"] is True
    assert result["draft_response"] == "Here's a draft."


def test_safety_override_passthrough_no_draft_already_false():
    # If model already said no-draft, override shouldn't change anything
    pred = _make_pred("account_compromise", should_draft=False, draft_response=None, no_draft_reason="flagged")
    result = _safety_override(pred)
    assert result["should_draft"] is False
    assert result["no_draft_reason"] == "flagged"  # original reason preserved


def test_no_draft_categories_are_complete():
    assert NO_DRAFT_CATEGORIES == {"account_compromise", "problem_gambling", "legal_regulatory"}


# ── Schema validation ─────────────────────────────────────────────────────────

def test_output_schema_has_required_fields():
    required = set(OUTPUT_SCHEMA["required"])
    expected = {"ticket_id", "category", "urgency", "should_draft",
                "no_draft_reason", "draft_response", "confidence"}
    assert required == expected


def test_output_schema_category_enum_matches_constants():
    schema_cats = set(OUTPUT_SCHEMA["properties"]["category"]["enum"])
    assert schema_cats == set(CATEGORIES)


def test_output_schema_urgency_enum_matches_constants():
    schema_urg = set(OUTPUT_SCHEMA["properties"]["urgency"]["enum"])
    assert schema_urg == set(URGENCY_LEVELS)


def test_output_schema_no_additional_properties():
    assert OUTPUT_SCHEMA.get("additionalProperties") is False


# ── Predictions file validation ───────────────────────────────────────────────

def test_predictions_schema(tmp_path):
    # Write a minimal valid predictions file and verify load_jsonl round-trips it
    valid_pred = {
        "ticket_id": "t_eval_001",
        "category": "account_access",
        "urgency": "low",
        "should_draft": True,
        "no_draft_reason": None,
        "draft_response": "Hi, please try resetting your password.\n\nSupport Team",
        "confidence": 0.92,
    }
    f = tmp_path / "preds.jsonl"
    f.write_text(json.dumps(valid_pred) + "\n")
    loaded = load_jsonl(str(f))
    assert len(loaded) == 1
    assert loaded[0]["ticket_id"] == "t_eval_001"
    assert loaded[0]["confidence"] == 0.92


def test_predictions_file_all_valid_categories():
    """Every prediction in predictions.jsonl must use a known category."""
    import os
    if not os.path.exists("predictions.jsonl"):
        pytest.skip("predictions.jsonl not generated yet — run eval.py first")
    preds = load_jsonl("predictions.jsonl")
    for p in preds:
        assert p["category"] in CATEGORIES, f"{p['ticket_id']}: unknown category {p['category']}"


def test_predictions_file_no_draft_on_sensitive():
    """No prediction for a hard-rule category should have should_draft=True."""
    import os
    if not os.path.exists("predictions.jsonl"):
        pytest.skip("predictions.jsonl not generated yet — run eval.py first")
    preds = load_jsonl("predictions.jsonl")
    for p in preds:
        if p["category"] in NO_DRAFT_CATEGORIES:
            assert p["should_draft"] is False, (
                f"{p['ticket_id']}: hard-rule category {p['category']} has should_draft=True"
            )


def test_predictions_field_consistency():
    """should_draft=True must have draft_response; False must have no_draft_reason."""
    import os
    if not os.path.exists("predictions.jsonl"):
        pytest.skip("predictions.jsonl not generated yet — run eval.py first")
    preds = load_jsonl("predictions.jsonl")
    for p in preds:
        if p["should_draft"]:
            assert p["draft_response"] is not None, f"{p['ticket_id']}: should_draft=True but no draft"
            assert p["no_draft_reason"] is None
        else:
            assert p["draft_response"] is None, f"{p['ticket_id']}: should_draft=False but has draft"
            assert p["no_draft_reason"] is not None


# ── Routing ───────────────────────────────────────────────────────────────────

def _pred(category="account_access", urgency="low", should_draft=True, confidence=0.9):
    return {"category": category, "urgency": urgency, "should_draft": should_draft, "confidence": confidence}


def test_confident_draft_is_auto_drafted():
    assert route(_pred()) == "auto_draft"


def test_confident_no_draft_goes_to_agent_queue():
    assert route(_pred(should_draft=False)) == "agent_queue"


def test_low_confidence_goes_to_human_triage_even_with_a_draft():
    assert route(_pred(confidence=0.5)) == "human_triage"


def test_threshold_is_inclusive_at_point_seven():
    assert route(_pred(confidence=ROUTING_CONFIDENCE_THRESHOLD)) == "auto_draft"
    assert route(_pred(confidence=ROUTING_CONFIDENCE_THRESHOLD - 0.01)) == "human_triage"


@pytest.mark.parametrize("category", sorted(NO_DRAFT_CATEGORIES))
def test_sensitive_categories_escalate_at_any_confidence(category):
    for confidence in (0.1, 0.5, 0.99):
        assert route(_pred(category=category, should_draft=False, confidence=confidence)) == "escalate"


def test_escalate_immediately_urgency_escalates_even_when_unsure():
    assert route(_pred(urgency="escalate_immediately", confidence=0.2)) == "escalate"

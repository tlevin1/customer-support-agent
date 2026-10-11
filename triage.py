"""
triage.py — Support ticket classifier for a regulated prediction market exchange.

Classifies tickets by category + urgency, decides whether to auto-draft,
and generates a draft when safe to do so.
"""

import json
from pathlib import Path

import anthropic

# ── Constants ─────────────────────────────────────────────────────────────────

CATEGORIES = [
    "account_access",
    "kyc_verification",
    "deposits_withdrawals",
    "trading_mechanics",
    "market_questions",
    "bug_report",
    "tax_documents",
    "account_compromise",
    "problem_gambling",
    "legal_regulatory",
    "other",
]

URGENCY_LEVELS = ["low", "medium", "high", "escalate_immediately"]

NO_DRAFT_CATEGORIES = frozenset({"account_compromise", "problem_gambling", "legal_regulatory"})

OUTPUT_SCHEMA = {
    "type": "object",
    "properties": {
        "ticket_id": {"type": "string"},
        "category": {"type": "string", "enum": CATEGORIES},
        "urgency": {"type": "string", "enum": URGENCY_LEVELS},
        "should_draft": {"type": "boolean"},
        "no_draft_reason": {"anyOf": [{"type": "string"}, {"type": "null"}]},
        "draft_response": {"anyOf": [{"type": "string"}, {"type": "null"}]},
        "confidence": {"type": "number"},
    },
    "required": [
        "ticket_id",
        "category",
        "urgency",
        "should_draft",
        "no_draft_reason",
        "draft_response",
        "confidence",
    ],
    "additionalProperties": False,
}


# ── Data loading ──────────────────────────────────────────────────────────────

def load_jsonl(path: str) -> list[dict]:
    return [
        json.loads(line)
        for line in Path(path).read_text().strip().splitlines()
        if line.strip()
    ]


# ── Prompt construction ───────────────────────────────────────────────────────

def _format_few_shot_examples(tickets: list[dict]) -> str:
    parts = []
    for t in tickets:
        lb = t["label"]
        body = " ".join(t["body"].split())[:160]
        if lb["should_draft"]:
            draft_str = "DRAFT=YES"
        else:
            draft_str = f"DRAFT=NO  (reason: {lb['no_draft_reason']})"
        notes = lb.get("gold_response_notes", "")
        part = (
            f"[{t['ticket_id']}]  category={lb['category']}  urgency={lb['urgency']}  {draft_str}\n"
            f"  subject : \"{t['subject']}\"\n"
            f"  body    : \"{body}...\"\n"
            f"  tenure  : {t['metadata']['user_tenure_days']} days"
        )
        if notes:
            part += f"\n  notes   : {notes}"
        parts.append(part)
    return "\n\n".join(parts)


def build_system_prompt(train_tickets: list[dict]) -> str:
    examples = _format_few_shot_examples(train_tickets)

    return f"""You are a support ticket triage system for a CFTC-regulated peer-to-peer prediction market exchange.

Classify each ticket by category and urgency, then decide whether to auto-draft a response.

━━━ ASYMMETRIC RISK ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
A wrongly auto-drafted response on a sensitive ticket (compromise, problem gambling,
legal threat, self-harm, minor) causes serious, potentially irreversible harm.
Failing to draft an easy ticket is a minor inconvenience.
Default to NO DRAFT whenever there is meaningful ambiguity or sensitivity.
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

CATEGORIES
  account_access        Login, password resets, 2FA, locked accounts (not compromise)
  kyc_verification      KYC/identity verification, document uploads, delays, rejections
  deposits_withdrawals  Deposits, withdrawals, payment methods, ACH/wire/card, timing
  trading_mechanics     How features work: parlays, limits, order matching, position sizing
  market_questions      Settlement timing, market grading, market rules, contract specs
  bug_report            App crashes, UI glitches, display errors (not fraud-attributed)
  tax_documents         1099s, tax forms, transaction exports, year-end statements
  account_compromise    Unauthorized account access, trades, or withdrawals user didn't initiate
  problem_gambling      Self-exclusion, deposit limits, cooling-off, loss-of-control signals, addiction
  legal_regulatory      Legal threats, attorney mentions, CFTC/AG/CFPB complaints, jurisdictional questions
  other                 Truly doesn't fit the above (use sparingly)

URGENCY LEVELS
  low                   Informational; no money or access at stake; routine questions
  medium                User mildly blocked or confused; not materially at risk
  high                  Money in motion, materially blocked, or significant frustration
  escalate_immediately  Human required NOW: active compromise, problem gambling signals,
                        self-harm references, minor mentioned, legal deadline

HARD RULES — NEVER AUTO-DRAFT (these are absolute; no exceptions)
  1. Category is account_compromise, problem_gambling, or legal_regulatory
  2. Ticket mentions a minor (the exchange requires age 18–21 depending on jurisdiction)
  3. Ticket mentions self-harm, suicide, or severe emotional distress
  4. Active fraud in progress ("someone is in my account right now", live withdrawal)

SOFT RULES — Usually should NOT draft (apply judgment; err toward no-draft)
  5. User asserting an exchange-provided fact is wrong (1099 amount, market grade, balance)
  6. User asking for a definitive policy/contract spec they'll rely on financially
     (e.g., "how does the MVP market settle if there's a tie?")
  7. Jurisdictional eligibility question (is the exchange available in X state?)

WHEN DRAFTING
  • Acknowledge the specific situation — not a generic opener
  • Do not promise specific outcomes or dates you cannot verify
  • Tell the user what happens next (escalation, ops review, etc.)
  • Sign as "Support Team" — no individual agent names
  • NEVER state a specific dollar amount, date, percentage, or rate unless the user
    provided that value in their ticket. If you don't know a policy number, say
    "please check the app or our Help Center" — do not guess or invent it.

CONFIDENCE (0.0 – 1.0)
  Estimate certainty across category + urgency + draft decision.
  > 0.9  clear-cut; one obvious interpretation
  0.7–0.9  minor judgment call; classification is clear but edge involved
  < 0.7   genuine uncertainty; human review would add value
  Note: hard-rule no-drafts can have high confidence when the trigger is unambiguous.

━━━ TRAINING EXAMPLES (30 labeled tickets) ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

{examples}

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
"""


# ── Safety backstop ───────────────────────────────────────────────────────────

def _safety_override(prediction: dict) -> dict:
    # Catches the case where the model correctly identifies a hard-rule category
    # but still sets should_draft=true (prompt drift, hallucination, etc.)
    # Does NOT protect against misclassification into a safe category.
    if prediction["category"] in NO_DRAFT_CATEGORIES and prediction["should_draft"]:
        prediction["should_draft"] = False
        prediction["no_draft_reason"] = (
            f"safety override: category '{prediction['category']}' requires human review"
        )
        prediction["draft_response"] = None
    return prediction


# ── Routing ───────────────────────────────────────────────────────────────────

ROUTING_CONFIDENCE_THRESHOLD = 0.7
ROUTES = ("escalate", "human_triage", "auto_draft", "agent_queue")


def route(prediction: dict, threshold: float = ROUTING_CONFIDENCE_THRESHOLD) -> str:
    """Where a ticket goes next. Low confidence can only move a ticket toward
    human review, never away from escalation:

    - escalate: sensitive category or escalate_immediately, at any confidence
      (over-escalating is recoverable; missing a distress signal isn't)
    - human_triage: anything else the model is unsure about; a draft, if any,
      is shown as a suggestion rather than queued for sending
    - auto_draft: confident, and the model wrote a reply
    - agent_queue: confident, no reply drafted
    """
    if prediction["category"] in NO_DRAFT_CATEGORIES or prediction["urgency"] == "escalate_immediately":
        return "escalate"
    if prediction["confidence"] < threshold:
        return "human_triage"
    return "auto_draft" if prediction["should_draft"] else "agent_queue"


# ── Core classifier ───────────────────────────────────────────────────────────

def classify_ticket(
    client: anthropic.Anthropic,
    ticket: dict,
    system_prompt: str,
) -> dict:
    user_message = (
        f"Ticket ID: {ticket['ticket_id']}\n"
        f"Subject: {ticket['subject']}\n"
        f"Body: {ticket['body']}\n"
        f"User tenure: {ticket['metadata']['user_tenure_days']} days\n"
        f"Submitted: {ticket['metadata']['submitted_at']}\n\n"
        "Check hard rules first. Then soft rules. Then classify and draft if appropriate. "
        "Return JSON."
    )

    response = client.messages.create(
        model="claude-haiku-4-5",
        max_tokens=1024,
        system=[
            {
                "type": "text",
                "text": system_prompt,
                "cache_control": {"type": "ephemeral"},
            }
        ],
        messages=[{"role": "user", "content": user_message}],
        output_config={"format": {"type": "json_schema", "schema": OUTPUT_SCHEMA}},
    )

    text = next(b.text for b in response.content if b.type == "text")
    prediction = json.loads(text)

    prediction["confidence"] = max(0.0, min(1.0, float(prediction.get("confidence", 0.5))))

    if not prediction["should_draft"]:
        prediction["draft_response"] = None
        if not prediction.get("no_draft_reason"):
            prediction["no_draft_reason"] = "no reason specified by model"
    else:
        prediction["no_draft_reason"] = None

    prediction = _safety_override(prediction)
    prediction["route"] = route(prediction)

    return prediction

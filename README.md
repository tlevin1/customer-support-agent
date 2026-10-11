# Support Ticket Triage Agent

An LLM triage system for the support queue of a regulated prediction-market
exchange. For each ticket it assigns a category and an urgency level, decides
whether an auto-drafted reply is safe, and writes the draft when it is.

The core problem is asymmetric. Drafting a cheerful auto-reply to a problem
gambling ticket or an account compromise causes serious, irreversible harm;
failing to draft a reply to a routine password reset is a minor inconvenience.
The whole system is designed around that: **when in doubt, don't draft.**

## How it works

```
ticket ──► Claude Haiku 4.5 (one structured-output call)
              category · urgency · should_draft · draft · confidence
                          │
                          ▼
           deterministic safety backstop (triage.py)
             account_compromise / problem_gambling / legal_regulatory
             → should_draft forced to false, whatever the model said
                          │
                          ▼
                    prediction (JSON schema-validated)
```

- **One call, not two.** Classifying first and drafting second added latency,
  and the structured output schema already constrains the model enough that a
  single pass works.
- **All labeled examples in the system prompt.** Retrieving only the "closest"
  few examples per ticket made edge cases worse. The full set is small, and with
  prompt caching every ticket after the first pays about 10% of the input cost.
  Including each example's reviewer notes (e.g. *deposit limit requests are
  responsible-gambling-sensitive even when framed as budgeting*) was the single
  most useful thing I did: those rules are very hard to express otherwise.
- **A deterministic safety backstop.** If the model correctly classifies a
  ticket as sensitive but still sets `should_draft=true` (prompt drift, a
  hallucination), code overrides it. It is cheap and catches one whole class of
  failure. It cannot catch a sensitive ticket misclassified into a safe
  category; that has to be handled by the prompt.
- **Confidence covers all three decisions together**, so a high-confidence
  no-draft on a compromise ticket tells an ops queue "this is obviously a
  compromise, not borderline."

## Prompt design

- **Open with the asymmetry.** Right after the role description: getting a
  sensitive ticket wrong is serious and irreversible, missing an easy draft is
  minor. It primes every later judgment call.
- **Label hard and soft rules differently.** "No exceptions" for the hard ones,
  "apply judgment; err toward no-draft" for the soft ones. The model handles
  both better when it knows which is which.
- **No invented policy numbers.** Drafts never state a dollar amount, date or
  rate the user didn't provide; they point to the app or Help Center instead.
  Without this rule the model occasionally made one up (it once invented a $10
  minimum withdrawal).

## Results

On the labeled training tickets:

| Metric | Result |
|---|---|
| Category accuracy | 100% |
| Urgency accuracy | 100% (no off-by-one errors) |
| Draft-decision accuracy | 100% |
| False drafts on sensitive tickets | 0 of 9 |

**That 100% is misleading, and I'd say so in review:** every labeled example
is in the prompt, so this measures recall, not generalization. Out-of-sample
behaviour on the held-out tickets was the real test, and the judgment calls
there went the conservative way: ambiguous distress language ("I'm tired")
and calm loss-of-control statements were escalated rather than drafted, and a
question about how a disputed market would settle went to human review because
a wrong automated answer would create liability.

## Where it breaks

- **Policy invention.** The "don't make up numbers" rule is a band-aid. The
  real fix is a versioned policy knowledge base in the prompt, with any figure
  in a draft that isn't in the knowledge base flagged.
- **Distress ambiguity.** "I'm frustrated with your app" and "I'm not doing
  well" sit on a spectrum. The system errs toward escalation, so it will
  over-escalate some venting. That is the right trade-off for now.
- **Novel phrasing**, especially for soft-rule cases like fact disputes or
  jurisdiction questions, will degrade accuracy in ways the training metrics
  can't show.

## Next steps

1. Hold out 5–10 labeled tickets from the few-shot pool to get an honest
   generalization estimate.
2. A second call that scores each draft against the drafting guidelines before
   an agent sees it, closing the policy-invention gap without a full knowledge base.
3. Route low-confidence predictions (< 0.7) to a lightweight human triage queue:
   a third path that reduces both failure modes at once.

## Running it

```bash
make install
export ANTHROPIC_API_KEY=...
make eval    # runs the labeled set (metrics) and the held-out set (predictions.jsonl)
make test    # unit tests: safety backstop, schema, prediction consistency
```

The ticket dataset and category taxonomy were provided by the company as part
of a take-home assessment and are not included in this repo. `eval.py` expects
`tickets_train.jsonl` and `tickets_eval.jsonl` in the project root.

# Demo script — 3 minutes

Dashboard: `uvicorn person_b.dashboard.server:app --reload`, open `http://localhost:8000`, click **Start loop** on cue.

## 0:00 – 0:25 — State the benchmark
"We're predicting heart disease from the UCI Heart Disease dataset — four hospital
sources, Cleveland, Hungary, Switzerland, Long Beach VA. Published logistic
regression baseline is 78.7% accuracy; best published SVM result is 83.3%. Our
agent's goal: beat 83.3%, with zero human in the loop — no one picks the model,
tunes it, decides how much data to use, or fixes a bad result."

## 0:25 – 0:45 — Kick off the loop, disclose the weak start
Click **Start loop**.
"It starts deliberately weak — a poorly-fit KNN on just the smallest hospital
source, Cleveland. That's intentional, disclosed openly: it guarantees the
correction loop actually fires live, instead of leaving it to chance."

## 0:45 – 1:45 — Narrate the correction moments
As iterations stream in:
"Watch the diagnosis text, not just the accuracy number — the agent is
distinguishing 'I don't have enough data' from 'I have the wrong model' from
real signal: learning-curve behavior, per-class recall. Each time it decides it's
data-starved, it goes and pulls the next hospital source live from the UCI API —
that's a real HTTP call happening right now, merged in through Nexla."

Point at the growing chip list under "Data sources pulled."

## 1:45 – 2:15 — The Zero.xyz moment (the highlight)
When the orange callout appears:
"Now it's exhausted all four native UCI sources and it's still short of
benchmark. This is the moment — it's not falling back to a pre-configured
service. It's searching Zero's marketplace live, right now — a real `zero
search` call — and picking a provider itself, scored by how well it matches
the task and whether it's actually up. [point at callout] That selection just
happened because of relevance and availability — not because we hardcoded a
service ID."

## 2:15 – 2:45 — Model switch + convergence
"With enrichment in hand, it also determines the model class itself was capped —
so it switches from KNN to an SVM, and converges."

Point at the final accuracy tile going green against the dashed 83.3% benchmark line.

## 2:45 – 3:00 — Close
"Final result: beats the published 83.3% SVM benchmark, autonomously, using data
it decided to go get. Every one of those autonomous actions — the data pulls, the
Zero purchase — went through a Pomerium gate first: an allowlist and a spend
ceiling, visible in the log on the right. That's not a claim of autonomy, that's
the guardrail that makes it safe to grant."

## Fallback talking points (if asked)
- **"Why start it weak on purpose?"** — Disclosed in the README; without it, the
  correction loop might not fire in a 3-minute live window. It's a demo-reliability
  choice, not a limitation of the agent's real diagnosis capability.
- **"Is the Pomerium check real or decorative?"** — It's a real local policy
  evaluation against `config/policy.yaml` (allowlist + spend ceiling) gating every
  autonomous action before it executes; wiring to Pomerium's live policy service is
  a drop-in swap behind the same `PomeriumGate.check()` interface.
- **"Was the Zero service pre-selected?"** — No — `discover()` shells out to a
  real `zero search --json` call at that point in the run, and `select_and_pay()`
  picks by task-relevance × availability × price. Real Zero listings don't
  expose ratings, so the agent scores task fit from the listing name/tags
  instead — a real decision, not a hardcoded service ID.
- **"What if the `zero` CLI or network isn't available during the demo?"** — The
  client falls back to a small mock catalog scored with the same logic, logged
  as a warning rather than silently pretending it's live. Worth checking `zero`
  is authenticated and reachable before going on stage.

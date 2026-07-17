# Self-Correcting Heart Disease Classifier

An autonomous ML training agent built for the **Loop Engineering Hackathon** (tokens&, SF). Give it a prediction task, and it collects data, trains a model, evaluates itself, and — without a human in the loop — figures out why it's underperforming and corrects course, repeating until it beats a published academic benchmark.

## The problem

We're predicting the presence of heart disease from patient clinical data (age, cholesterol, blood pressure, max heart rate, exercise-induced angina, and similar features) using the **UCI Heart Disease dataset**, which is actually four separate hospital sources: Cleveland, Hungary, Switzerland, and Long Beach VA (~920 records combined).

This dataset has well-established published benchmarks:
- Logistic regression baseline: **78.7% accuracy**
- Best published result (SVM): **83.3%–91.8% accuracy** depending on split

Our agent's goal: autonomously match or beat these numbers, with no human choosing the model, tuning hyperparameters, deciding how much data to use, or deciding how to fix a bad result.

**Crucially, the agent doesn't start with the full dataset.** It starts with only the smallest hospital source (Cleveland) and pulls in additional sources live, on its own, mid-loop, when it decides it's data-starved rather than model-mismatched. Data acquisition is part of the loop, not a setup step before it.

## Why this is a loop, not a script

Most "AI does ML" demos are a fixed pipeline: pull data → train one model → report accuracy. That's not loop engineering, it's a script.

Ours is a genuine closed loop:

```
PLAN → decide next model/data configuration
ACT → train
OBSERVE → evaluate against benchmark, examine failure pattern
CORRECT → diagnose the cause and change something real:
            pull more native data, transform existing data,
            enrich via Zero, or switch model class
→ repeat until benchmark is beaten
```

The agent starts deliberately weak — both in model choice and in data volume (only the smallest hospital source to begin with) — specifically so the correction loop is guaranteed to fire and be visible in a live demo, not left to chance. This is disclosed openly here rather than hidden.

## Architecture

```
UCI API — queried live, incrementally, starting with Cleveland only
      │
      ▼
  Nexla  ──── normalizes schema, merges each newly-pulled hospital
      │        subset into the working dataset, executes any
      │        transformations the agent requests mid-loop
      ▼
 Training / Evaluation loop (model class selected autonomously)
      │
      ├── underperforming, diagnosis = not enough data?
      │     → pull next UCI hospital source live → Nexla merge → retry
      ├── underperforming, diagnosis = bad data shape?
      │     → Nexla transform → retry
      ├── native UCI sources exhausted, still underperforming?
      │     → Zero.xyz marketplace (agent discovers + pays for an
      │       enrichment/scraping service live, not pre-configured) → retry
      └── underperforming due to wrong model class?
            → switch model → retry
      │
      ▼
  Every autonomous action gated through Pomerium
  (spend ceiling + action allowlist, enforced and logged)
      │
      ▼
  Final output: best model, final dataset state (which sources were
  pulled and in what order), final accuracy, full iteration history,
  comparison against published benchmark
```

## Sponsor tools used

| Sponsor | Role | Why it's real, not decorative |
|---|---|---|
| **Nexla** | Data normalization + mid-loop transformations | Every time the agent decides the data needs reshaping, that's an actual Nexla job, not a disguised pandas call |
| **Zero.xyz** | Live, unplanned discovery of enrichment/scraping services | The agent doesn't know which service it needs until the correction step tells it — this is Zero's actual value proposition (tool discovery without pre-configuration), demonstrated live |
| **Pomerium** | Access control + audit boundary on every autonomous action | Direct answer to the "autonomy without manual intervention" criterion — shows the guardrail that makes unattended action safe, not just a claim of autonomy |
| **Akash** *(optional)* | Hosts training compute | Justifies continuous/cheap retraining without a fixed local machine |

## What makes this a good "Idea" and "Autonomy" score

- The loop makes real decisions based on real evaluation results — model class switches and data corrections are driven by diagnosed failure patterns (e.g., confusion matrix behavior, per-class recall), not a scripted sequence
- Nothing in the correction path is hardcoded to a specific fix; the agent chooses among transform / enrich / switch-model based on what it observes
- The benchmark comparison is a concrete, published, citable number — not a vague "the model got better" claim

## Running it

```bash
# setup instructions go here once the pipeline is built
```

## Demo flow (3 minutes)

1. State the benchmark being targeted (78.7% baseline / 83.3% best-published)
2. Kick off the loop live
3. Narrate the correction moment, especially the Zero discovery step — this is the differentiating moment of the whole demo
4. Show final accuracy vs. benchmark, and the full iteration history

## Team / Sponsors we're competing for

Built for the Loop Engineering Hackathon. Targeting **Best Use of Zero.xyz** specifically, alongside general judging categories.

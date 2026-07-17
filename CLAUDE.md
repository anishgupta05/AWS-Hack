# CLAUDE.md

Guidance for Claude Code (or any agent) working on this repo. Read this before writing code.

## Project in one sentence

An autonomous agent that trains a classifier on the UCI Heart Disease dataset, evaluates itself, and — with no human in the loop — diagnoses *why* it's underperforming and corrects course (transforms the data, or switches model class entirely), repeating until it beats the published academic benchmark.

## Why this project exists (context for judging)

This is a submission for the Loop Engineering Hackathon (tokens&, SF). Judging rubric is:
- Idea — 20%
- Technical Implementation — 20%
- Tool Use — 20% (must use ≥3 sponsor tools, integrated meaningfully, not decoratively)
- Presentation — 20% (3-minute live demo)
- Autonomy — 20% (does the agent act on real data without manual intervention)

The entire architecture should be justified against these five criteria. If a feature doesn't clearly serve one of them, cut it — we are time-boxed, not building a product.

## The core loop (this is the actual deliverable)

```
PLAN  → decide what model/data configuration to try next
ACT   → train the model on the current dataset
OBSERVE → evaluate on held-out test data, compute accuracy/F1
CORRECT → if below target, diagnose the likely cause and change ONE thing:
            (a) transform/engineer the existing data, or
            (b) pull in enrichment data via Zero, or
            (c) switch model class entirely
          then loop back to PLAN
```

Stop condition: accuracy exceeds the benchmark target (see below), or max iterations reached (cap this — do not let a live demo run an unbounded loop).

**Do not fake this.** The self-correction step must make a real decision based on the actual evaluation result (e.g., confusion matrix pattern, per-class recall, feature importance), not a scripted sequence of "try model 1, then model 2, then model 3" regardless of outcome. Judges are explicitly scoring whether autonomy is real.

## Dataset and benchmark target

- **Dataset:** UCI Heart Disease — this is actually four separate hospital sources: Cleveland (~300 records), Hungary, Switzerland, and Long Beach VA (~920 records combined, ~14 features: age, sex, cholesterol, resting BP, max heart rate, exercise-induced angina, etc.)
- **Source:** UCI ML Repository API, queried live and incrementally — see "Incremental data acquisition" below. Do not pull the full combined dataset upfront; this is a deliberate architectural choice, not a shortcut.
- **Benchmark to beat:** published logistic regression baseline of 78.7% accuracy; best published result in literature is ~83.3% (SVM) to ~91.8% (SVM, different train/test split). State clearly in the demo which specific benchmark and paper you're comparing against — don't round this up loosely.
- **Target for our agent:** beat 83.3% is a credible, defensible claim. Do not overclaim beating 91.8% unless you've reproduced that exact split.

## Incremental data acquisition (the agent pulls data AS it goes, not upfront)

This is a core architectural decision, not just an implementation detail. The agent starts with only the smallest hospital source (Cleveland) and expands its own dataset live, mid-loop, when it diagnoses that it's data-starved rather than model-mismatched:

```
Iteration 1: train/eval on Cleveland only (~300 records)
   → if underperforming AND diagnosis = insufficient data
     → agent queries UCI API live for the next hospital source (e.g. Hungary)
     → Nexla normalizes and merges the new subset into the existing dataset
     → retry
Iteration 2: train/eval on Cleveland + Hungary
   → repeat expansion logic against remaining hospital sources
...
Once all four UCI sources are exhausted and still underperforming:
   → THIS is the trigger point for Zero.xyz enrichment (see below) —
     Zero is a second-order fallback after native sources are exhausted,
     not a first resort
```

Why this matters for scoring: this turns "real-time data, no manual intervention" from a claim into something literally visible on screen — each loop iteration should show the dataset growing, sourced from a live API call the agent chose to make, not data you preloaded. It also naturally produces 3-4 distinct, genuine "the agent decided it needed more and went and got it" moments for the demo, which is far stronger than one static ingest at the start.

Implementation note: the diagnosis step needs real logic to distinguish "I need more data" from "I need a different model" from "I need different features" — e.g., look at learning-curve behavior (does a train/test gap suggest overfitting on too little data?) or per-class recall imbalance (suggests model mismatch, not data volume). Don't hardcode "always try more data first" — that's not a real diagnosis, it's a fixed sequence wearing a diagnosis costume.

## Deliberate design decision: rig the first model to be weak

To guarantee the correction loop actually fires during a live 3-minute demo (rather than risking the first model attempt succeeding and skipping the interesting part), the initial model choice should be a genuinely poor fit for this data — e.g., a plain KNN with an unreasonable k, or a linear model on unscaled/untransformed features. This is an intentional demo-reliability choice, not a limitation of the agent's real capability. Document this openly in the README and be ready to explain it if a judge asks — it's honest as long as you disclose it, and dishonest if you present it as if the agent "happened" to start weak.

## Sponsor tool integration (need ≥3, here's exactly how each is used)

### 1. Nexla — data normalization / transformation / merge layer
- Sits between every UCI API pull (initial and each incremental one) and the training pipeline
- Handles schema normalization on each ingest, AND handles merging each newly-pulled hospital subset into the growing working dataset — this is real, recurring work across multiple loop iterations, not a one-time setup step
- Also executes any feature-level transformation the correction loop requests (option (a) in the loop) when the diagnosis is "wrong shape" rather than "not enough volume"
- This is what makes Nexla's inclusion real rather than decorative — it's doing distinct work at multiple points across the run, not once

### 2. Zero.xyz — dynamic enrichment discovery, second-order fallback (this is the "best use of Zero" bid)
- Only invoked after all four native UCI hospital sources have been exhausted via incremental pulls AND the agent is still underperforming — i.e., this is a fallback beyond the dataset's native data, not a substitute for it
- The agent should not have a hardcoded Zero service pre-selected. It should search Zero's marketplace live, at that point in execution, for a scraping/enrichment service, and use community ratings + task fit to choose one itself
- This is the single most important integration for the "best use of Zero" prize specifically — the demo moment should make clear this was NOT pre-configured, and that it only happened because native data was genuinely exhausted first. Narrate it explicitly: "the agent already pulled everything UCI has — now it's reaching beyond that on its own"
- Do not route the *primary* dataset pulls through Zero — those go through the UCI API directly, incrementally, as described above

### 3. Pomerium — access control / audit layer on autonomous actions
- Every action with a real-world side effect (spending from the Zero wallet, any external API call the agent initiates on its own) routes through a Pomerium-gated policy
- Concretely: define a spend ceiling and an action allowlist; the agent can act autonomously within that boundary without asking, anything outside it would require approval (even if you don't build the approval flow, the boundary check should be real and demoable)
- This is your direct answer to the "Autonomy" criterion — don't just claim autonomy, show the guardrail that makes it safe to grant

### Optional 4th: Akash — hosting training compute
- Only add if time permits. Legitimate justification: "the loop retrains repeatedly and needs to run continuously/cheaply, so inference/training is hosted on Akash rather than a fixed local machine." Don't add this just to pad the sponsor count.

### Do NOT force: Fillmore/Metaview
Fillmore is an outbound recruiting tool. It has no honest fit in this project. Do not include it just to hit a higher sponsor count — judges will see through a forced integration faster than a missing one.

## Tech stack guidance

- Keep models lightweight and fast: logistic regression, KNN, random forest, SVM, small gradient boosting. No deep learning — this dataset is too small for it to be justified, and training time will kill your live demo.
- Target total training time per loop iteration: under a few seconds. If any single iteration takes more than ~10 seconds, the live demo will drag and lose the judges' attention during the 3-minute window.
- Cap total loop iterations for the demo (something like 3-6 visible iterations is plenty — more looks repetitive, not more impressive).
- Log every iteration's decision and reasoning in human-readable form (not just numbers) — this is what you'll show on screen during presentation to prove the correction step is reasoned, not scripted.

## What "done" looks like

A runnable script or small app that, unattended:
1. Pulls only the smallest UCI hospital subset (Cleveland) live to start
2. Trains a deliberately weak initial model
3. Evaluates and diagnoses the failure (data volume vs. data shape vs. model mismatch)
4. Incrementally pulls additional UCI hospital sources live as needed, merged via Nexla, across multiple iterations
5. Once native sources are exhausted, falls back to Zero.xyz for external enrichment if still underperforming
6. Switches model class if the diagnosis calls for it, not data volume
7. Converges on a final model + dataset state that beats the 83.3% benchmark
8. Outputs a clear final summary: best model, final accuracy, data sources used and in what order, iteration history, benchmark comparison
9. Every autonomous action was gated through Pomerium, and this is visible/loggable

## Presentation notes (for whoever is driving the demo)

3 minutes is not long. Rehearse the exact sequence: state the benchmark being beaten, kick off the loop, narrate the correction moment (especially the Zero discovery moment — make sure this is the visual/narrative highlight), show the final accuracy vs. benchmark comparison. Don't explain the architecture in prose during the demo — show it happening.

## Division of work (2-person team)

Split by ML/infra depth vs. integration/demo strength, not by sponsor count. Each person should have fully-ownable pieces, not thin slices of everything.

**Person A — the core loop (ML/infra-heavy)**
- UCI API incremental pull logic (start with Cleveland, expand to other hospital sources on demand)
- Nexla integration: schema normalization + merging each new data source in
- Model training/eval harness (model classes, swapping logic)
- The diagnosis logic — the most important piece technically; this is what makes "Idea" and "Technical Implementation" scores real rather than theater
- Tuning the deliberately-weak starting point so the loop reliably fires correctly in the demo

**Person B — two sponsor integrations + everything demo-facing**
- Pomerium integration: action allowlist / spend ceiling, gate check in front of every autonomous action. API/config work, fully ownable on its own.
- Zero.xyz integration: marketplace search/discovery call and payment flow. Also primarily integration work, and benefits from a product/design eye on how to narrate that moment well since it's the "best use of Zero" bid.
- Live demo dashboard: shows the loop's iteration history in real time (which data source was pulled, which model, accuracy climbing, the Zero moment highlighted).
- Demo script and rehearsal ownership, since Presentation is 20% of the score and the window is unforgiving.

**Shared / do together, don't split**
- Wiring Person A's loop output into Person B's Pomerium gate and Zero call — this seam is where bugs hide, so do it as a joint session once both halves work standalone, not asynchronously.
- At least one full rehearsal run-through together — Person A needs to be able to explain the diagnosis logic if a judge asks, and Person B needs to know the loop's real behavior (not just the happy path) to narrate it accurately.

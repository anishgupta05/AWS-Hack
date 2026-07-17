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

- **Dataset:** UCI Heart Disease (Cleveland + multi-hospital combined set, ~920 records, ~14 features: age, sex, cholesterol, resting BP, max heart rate, exercise-induced angina, etc.)
- **Source:** UCI ML Repository API (pull this programmatically, not a static CSV baked into the repo — the "real-time data" framing depends on this being a live pull)
- **Benchmark to beat:** published logistic regression baseline of 78.7% accuracy; best published result in literature is ~83.3% (SVM) to ~91.8% (SVM, different train/test split). State clearly in the demo which specific benchmark and paper you're comparing against — don't round this up loosely.
- **Target for our agent:** beat 83.3% is a credible, defensible claim. Do not overclaim beating 91.8% unless you've reproduced that exact split.

## Deliberate design decision: rig the first model to be weak

To guarantee the correction loop actually fires during a live 3-minute demo (rather than risking the first model attempt succeeding and skipping the interesting part), the initial model choice should be a genuinely poor fit for this data — e.g., a plain KNN with an unreasonable k, or a linear model on unscaled/untransformed features. This is an intentional demo-reliability choice, not a limitation of the agent's real capability. Document this openly in the README and be ready to explain it if a judge asks — it's honest as long as you disclose it, and dishonest if you present it as if the agent "happened" to start weak.

## Sponsor tool integration (need ≥3, here's exactly how each is used)

### 1. Nexla — data normalization / transformation layer
- Sits between the raw UCI API pull and the training pipeline
- Handles schema normalization on initial ingest
- **Critically:** when the correction loop decides on data transformation (option (a) above), that transformation is executed as a Nexla job, not a raw pandas script. This is what makes Nexla's inclusion real rather than decorative — it's doing actual work at the one point in the loop where data reshaping happens.

### 2. Zero.xyz — dynamic enrichment discovery (this is the "best use of Zero" bid)
- Only invoked when the correction loop diagnoses that the dataset itself is insufficient (not just wrongly shaped) — i.e., option (b) above
- The agent should not have a hardcoded Zero service pre-selected. It should search Zero's marketplace live, at that point in execution, for a scraping/enrichment service, and use community ratings + task fit to choose one itself
- This is the single most important integration for the "best use of Zero" prize specifically — the demo moment should make clear this was NOT pre-configured. Narrate it explicitly: "the agent didn't know it would need this until just now"
- Do not route the *primary* dataset pull through Zero — that's dishonest to what Zero is actually for and will read as forced

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
1. Pulls the UCI Heart Disease dataset live
2. Trains a deliberately weak initial model
3. Evaluates and diagnoses the failure
4. Corrects via Nexla transformation or Zero enrichment (both should each fire at least once across the full run — build the loop so both are realistically likely to be needed, not just one of them)
5. Switches model class if needed
6. Converges on a final model + dataset state that beats the 83.3% benchmark
7. Outputs a clear final summary: best model, final accuracy, iteration history, benchmark comparison
8. Every autonomous action was gated through Pomerium, and this is visible/loggable

## Presentation notes (for whoever is driving the demo)

3 minutes is not long. Rehearse the exact sequence: state the benchmark being beaten, kick off the loop, narrate the correction moment (especially the Zero discovery moment — make sure this is the visual/narrative highlight), show the final accuracy vs. benchmark comparison. Don't explain the architecture in prose during the demo — show it happening.

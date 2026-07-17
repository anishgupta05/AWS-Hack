# Self-Improving ML Training Agent

## The Idea

We're building a self-improving ML training agent — you give it a prediction task (leaning biology, tied to ISEF/lab background), and it autonomously: pulls an initial dataset from a real public bio API, picks a model, trains it, evaluates it, and if the results aren't good enough, diagnoses *why* and corrects itself — either transforming/expanding the data or switching model classes entirely (e.g., drops a bad CNN attempt for a random forest). It keeps looping until it converges on a best dataset + model + accuracy, and that's the final output.

## Why This Fits the Rubric

It's a genuine plan → act → observe → correct loop, not a one-shot task dressed up as autonomous. That maps directly onto the "Autonomy" criterion (agent acting on real data with no manual intervention across multiple iterations) and gives us a strong live demo, since we can literally watch accuracy improve and watch the agent abandon a model that isn't working.

## Sponsor Usage (need 3 minimum)

- **Nexla** — normalizes/pipes the incoming data
- **Zero.xyz** — the key moment: when the model underperforms, instead of us pre-wiring a fix, the agent reaches into Zero's marketplace and dynamically discovers/pays for an enrichment or scraping service it wasn't configured with upfront, as its actual "get more data" correction step. This is our shot at the "best use of Zero" prize specifically, since it's genuine unplanned tool discovery, not us bolting on an API call
- **Pomerium** — every autonomous action the agent takes gets routed through Pomerium for access control, so we can show judges the safety boundary that makes unattended autonomy defensible
- **(Optional 4th) Akash** — hosting training compute, if we want the "why not just run this locally" story

## Deliberate Design Choice

We're planning to rig the agent's first model attempt to be a bad fit on purpose, so the self-correction (and the Zero enrichment moment) reliably fires during the live demo instead of leaving it to chance.

## Still to Lock Down

The specific biology prediction task and dataset — that's the decision most likely to blow our time budget if we go in vague, so we want that nailed before the event starts.

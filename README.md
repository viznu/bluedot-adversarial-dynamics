# Adversarial dynamics in LLM populations: oversight and capture

This repository contains experiments studying what happens when many LLM-based
agents interact under competitive or adversarial pressure. It is one part of a
four-week project carried out within the BlueDot Technical AI Safety course.
The guiding question: **can a group of AI agents coordinate trustworthily when
no single central node can be trusted?** The project approaches that question
from two opposite directions, one experiment each.

## Experiment A — oversight: can an auditor repair a bad equilibrium?

El & Zou ([arXiv:2510.06105](https://arxiv.org/abs/2510.06105), "Moloch's
Bargain") showed that when LLM sellers are optimized to win customers in a
competitive sales task, performance gains come bundled with misalignment: a
6.3% increase in sales was accompanied by a 14.0% rise in deceptive marketing
claims. ("Deceptive" here means the sales pitch asserts things that contradict
or exceed the product's ground-truth feature list.)

The extension studied here: insert a single **audit agent** into the
optimization loop and measure whether it repairs the deception equilibrium —
and at what cost, in auditor false positives on honest pitches and in lost
sales. This directory wraps the official paper code
([batu-el/molochs-bargain](https://github.com/batu-el/molochs-bargain)),
vendored under `exp_a_moloch_auditor/vendor/` (not tracked in this repo).

One deliberate fidelity difference: the original paper drives competitive
pressure through fine-tuning (rejection sampling / feedback training). Here,
pressure is applied at inference time via best-of-N selection — the seller
samples N pitches and the most persuasive one is kept. This is far cheaper and
keeps the mechanism inspectable, but it is an adaptation, not a replication.

## Experiment B — capture: how cheaply can an adversary flip a convention?

Ashery, Aiello & Baronchelli (*Science Advances*, 2025) showed that
populations of LLM agents (24–200) playing a **naming game** spontaneously
converge on a shared convention. The naming game is a classic coordination
experiment: in each round agents are paired, each pair privately picks a name
from a small pool, and both are rewarded if they match (+100 here) and
penalized if they don't (−50). No agent sees the whole population; conventions
emerge purely from repeated local interactions. The paper also showed that a
small *committed minority* — agents that always play one fixed name — can tip
an established convention to their preferred one.

The extension studied here treats that tipping as an **attack surface**: how
small can an adversarial committed faction be and still capture the
population's norm? The tipping threshold, measured across seeds, is the
attack cost. `exp_b_norm_capture/` contains a from-scratch implementation
that runs entirely on a local model (via MLX on Apple Silicon), so population
experiments are free to run.

One deliberate speed adaptation, noted for fidelity: the original paper plays
one pair per time step; this implementation randomly partitions all agents
into pairs each round and plays the pairs simultaneously.

## Status

| Milestone | Experiment A (oversight) | Experiment B (capture) | Status |
|---|---|---|---|
| 1 — pipelines running | Recon of official code; baseline deception rate on ~50 pitches at toy scale | Naming game implemented; 24-agent runs on a local model and on a hosted model | ✅ done |
| 2 — main experiments | Audit agent in a best-of-N loop; 3 arms (no pressure / pressure / pressure + auditor); auditor false-positive rate | Cold-start faction sweep complete (15 runs); warm-start tipping sweep incomplete | 🟨 partial |
| 3 — robustness + writeup | One robustness axis (e.g. auditor-aware seller: does deception get subtler?); error bars | One robustness axis (second model family or N=100); writeup | ⬜ |

### Experiment B — what has been measured so far

Full detail, including the runs behind each number, is in
[`exp_b_norm_capture/RESULTS.md`](exp_b_norm_capture/RESULTS.md).

- The local 3.8B model never formed a convention in 60 rounds. A hosted model
  (gpt-4o-mini) converged in 15–26 rounds across three seeds. Since a tipping
  threshold presupposes an established convention, the local runs cannot answer
  the question the experiment poses.
- **Before a convention forms**, a faction of just 2 of 24 agents (8%) captured
  the population in every run tested — 15 runs across sizes 2–6 and three seeds,
  all ending at 100% adoption of the faction's name. No failing size was found,
  so the threshold in this regime lies below 8% and remains unmeasured.
- **After a convention forms**, the same 21% faction converted nobody in 60
  rounds. This rests on a single completed run: the sweep intended to establish
  the warm-start threshold aborted on API rate limits and is unfinished.
- A behavioural detector that identifies faction members from interaction
  structure alone achieves perfect recall while a faction is losing, and detects
  nothing once it wins — the members are then never in the minority, so the
  signal it depends on disappears.

These are demonstrations of a working pipeline with small sample sizes, not
settled results.

## Reproducing

```bash
uv venv && uv pip install -r requirements.txt
# Experiment B: quick demo (6 agents, 2 names, ~1 min on Apple Silicon)
make demo
# Experiment B: full run (24 agents, to convergence)
cd exp_b_norm_capture
uv run python naming_game.py --config config.yaml --seed 7
uv run python analyze.py runs/<run_id>.jsonl
```

Experiment B runs locally with no API key, using a cached 4-bit instruct model
(`mlx-community/Phi-4-mini-instruct-4bit`) via `mlx_lm`. That configuration does
not converge, so the hosted configurations in
[`exp_b_norm_capture/RESULTS.md`](exp_b_norm_capture/RESULTS.md) are needed for
the capture experiments; those require `OPENAI_API_KEY`. Experiment A also
requires an OpenAI key; see `exp_a_moloch_auditor/RECON.md` for cost notes.

## Limitations

- Small scale throughout (N=24 agents, ~50 pitches); numbers are demonstrations
  of the pipelines, not settled results.
- Experiment B's warm-start tipping sweep is incomplete, and its cold-start sweep
  never reached a faction size that failed, so neither threshold is measured.
- Three seeds per condition at most, and one seed for the completed warm-start
  run. No confidence intervals are reported, because the sample sizes do not
  support them.
- Best-of-N selection is a cheap stand-in for the original paper's fine-tuning
  pressure; conclusions about the auditor apply to that adapted mechanism.
- Single model family per experiment so far.

## References

- El, B. & Zou, J. (2025). *Moloch's Bargain: Emergent misalignment when LLMs
  compete for audiences.* arXiv:2510.06105.
- Ashery, A. F., Aiello, L. M., & Baronchelli, A. (2025). *Emergent social
  conventions and collective bias in LLM populations.* Science Advances.

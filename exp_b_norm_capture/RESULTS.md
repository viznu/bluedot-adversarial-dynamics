# Experiment B results — norm capture in an LLM population

Status: partial. The cold-start sweep is complete; the warm-start (tipping
threshold) sweep is not. Sample sizes are small throughout. Every number below
comes from a run log in `runs/`, and nothing here is simulated or estimated.

## The setup

Twenty-four language-model agents play a **naming game**. Each round the
population is randomly split into pairs. Both members of a pair privately pick a
name from a two-name pool (`F` or `J`); matching earns each of them 100 points,
mismatching costs each of them 50. No agent is told who its partner is, how many
agents exist, or what the population as a whole is doing. Each remembers only its
own last five interactions.

A **convention** is said to have formed when one name is held by at least 95% of
the population for five consecutive rounds.

A **faction** is a subset of agents that all play the same fixed name, know one
another in advance, and never change regardless of payoff. Factions are injected
by configuration, so which agents belong to one is known exactly — that ground
truth is what makes detection measurable rather than a matter of opinion.

Two injection regimes are distinguished, and they turn out to matter more than
anything else measured here:

- **cold** — the faction is present from round one, while no convention exists.
- **warm** — the faction is injected into a population that has already
  converged, by priming every agent's memory from a completed run.

## Result 1 — model capability changed the outcome qualitatively

| Population | Convergence | Parse failures |
|---|---|---|
| Phi-4-mini-instruct-4bit, run locally | none in 60 rounds; oscillated at 71–75% | 9% |
| gpt-4o-mini, hosted | converged, rounds 15–26 across three seeds | 0% |

The local 3.8B model never formed a convention at all. This matters beyond
convenience: a tipping-threshold question presupposes an established convention
to tip, so the local runs cannot answer it. Three hosted baselines converged
(seeds 7 and 23 on `F`, seed 11 on `J`).

## Result 2 — a forming convention is capturable by a very small faction

Cold-start sweep, faction sizes 2–6 of 24 agents, three seeds each, 15 runs:

| Faction | Fraction | Captured the convention |
|---|---|---|
| 2 agents | 8% | 3/3 |
| 3 agents | 12% | 3/3 |
| 4 agents | 17% | 3/3 |
| 5 agents | 21% | 3/3 |
| 6 agents | 25% | 3/3 |

Every run ended with the population converged on the faction's name, at 100%.
**No lower bound was found**: the sweep never reached a size that failed, so the
threshold in this regime lies below 8% and is not yet measured.

Faction size did affect *speed* rather than success. Two-agent factions took
12–33 rounds to capture; six-agent factions took 10–14. A larger faction
converges the population faster, and in several runs faster than the honest
population converged on its own.

## Result 3 — an established convention resisted the same faction entirely

One warm-start run completed: five agents (21%) committed to `J`, injected into a
population already converged on `F`.

After 60 further rounds the population remained at 79% `F`. The five `J`-players
were exactly the faction. It converted nobody.

The contrast is stark — the same faction size captures at will before a
convention exists and fails completely after — but it rests on **one run**, and
the sweep intended to establish the warm-start threshold did not complete (see
Limitations). The comparison should be treated as suggestive only.

## Result 4 — a behavioural detector that works in one regime and fails in another

`detect_faction.py` attempts to identify faction members from the interaction
log alone: no prompts, no completions, only who played what and what it cost.

The first statistic tried was **stubbornness** — how often an agent does *not*
switch after a mismatch. A committed agent scores 1.0 by construction. Its
threshold is calibrated on a faction-free population so that at most one honest
agent is flagged.

| Condition | Recall | False-alarm rate |
|---|---|---|
| Unconverged local population, faction present | 1.00 | 0.16 |
| Hosted, faction captured the convention | 0.80 | 0.38 |
| Hosted, faction failed against a convention | 1.00 | **0.84** |

The statistic collapses once a convention exists. The reason is structural: in a
converged population almost nobody switches, so "cannot switch" and "has no
reason to switch" become indistinguishable, and 21 of 24 agents get flagged.

Conditioning on being **in the losing minority** repairs it. An honest agent
losing while outnumbered has an incentive to move; a committed one cannot:

| Condition | Recall | False alarms | Agents with evidence |
|---|---|---|---|
| Faction captured the convention | 0.00 | — | 10/24 |
| Faction failed against a convention | 1.00 | 0.00 | 8/24 |

Detection is perfect when the attack is failing and absent when it succeeds.
Once the faction's name becomes the majority, its members are never in the
minority, so the detector has nothing to observe and abstains on most of the
population. Whether this pattern generalises beyond the naming game is untested.

## Limitations

- **The tipping-threshold sweep did not complete.** Nine warm-start runs were
  attempted; all aborted on API rate limits after 4–8 of 60 rounds. The partial
  logs were deleted rather than kept, because an aborted run is not evidence
  that a faction failed. The warm-start threshold is therefore unmeasured.
- **The cold-start threshold is also unmeasured**, in the other direction: every
  size tested succeeded, so the sweep did not bracket the boundary from below.
- Three seeds per condition, and one seed for the completed warm-start run. No
  confidence intervals are reported because the sample sizes do not support them.
- Hosted-model runs are reproducible only in a best-effort sense. Model ids can
  be re-pointed by the provider, so the recorded model id is not a guarantee of
  the weights that answered.
- The local-versus-hosted comparison confounds two changes at once — execution
  environment and model capability — and cannot separate them.
- Committed agents make no model call, so a faction is cheaper to run than an
  honest population of the same size. Cost figures are not comparable across
  faction sizes.

## An error worth recording

An earlier version of the sweep script counted a failed run as a run in which the
faction did not capture the convention. When all fifteen warm-start runs failed on
rate limits, the output was a clean, plausible table reporting that no faction —
up to 50% of the population — could flip an established convention. That table
was an artifact of the failures and described data that did not exist.

The script now marks failed runs explicitly, excludes them from the curve, and
refuses to report a threshold when any run in a sweep is missing.

## Reproducing

```bash
uv venv && uv pip install -r ../requirements.txt

# Local, free, no API key. Does not converge; this is expected.
uv run python naming_game.py --config config.yaml --seed 7

# Hosted. Needs OPENAI_API_KEY in the environment.
uv run python naming_game.py --config config_openai_homogeneous.yaml --seed 7

# Cold-start tipping sweep (15 runs, roughly 5,000 model calls).
uv run python sweep.py --regime cold --sizes 2,3,4,5,6 --seeds 7,11,23 \
    --faction-name J --parallel 2 --max-concurrency 4

# Faction detection against known ground truth.
uv run python detect_faction.py runs/capture-warm-j5-n24.jsonl \
    --control runs/homog-gpt4omini-n24-s7.jsonl \
    --committed 3,8,11,17,22 --metric holdout
```

Rate limits are the binding constraint on sweep size. Keep
`--parallel × --max-concurrency` at or below 8; higher values caused the failures
described above.

## Files

| File | Purpose |
|---|---|
| `naming_game.py` | The game. Local (MLX) or hosted backends, warm start, committed factions |
| `providers.py` | Hosted backends over the OpenAI chat-completions shape, with preflight and budget caps |
| `sweep.py` | Faction-size sweeps and the tipping curve |
| `detect_faction.py` | Faction detection from interaction structure, scored against ground truth |
| `analyze.py` | Convergence plots |
| `runs/` | Every log referenced above |

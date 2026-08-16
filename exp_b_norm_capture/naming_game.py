"""Naming game in an LLM population, after Ashery, Aiello & Baronchelli (2025).

Each round the population is randomly partitioned into pairs; both members of
a pair simultaneously pick a name from a shared pool. Matching picks earn
+100 each, mismatches -50 each. Agents see only their own recent interaction
history, rendered into the prompt. Conventions emerge (or don't) from local
interactions alone.

Deviation from the original paper, by design: the paper samples one pair per
time step; here all pairs in a round play simultaneously (speed adaptation).

Committed agents (config `committed: {agent_id: name}`) always play their
assigned name and skip the model call — the session-2 adversary hook.
"""

import argparse
import json
import random
import re
import sys
import time
from collections import Counter, deque
from dataclasses import dataclass, field
from pathlib import Path

import yaml


@dataclass
class Agent:
    agent_id: int
    committed_to: str | None = None
    score: int = 0
    memory: deque = field(default_factory=lambda: deque(maxlen=5))


def build_prompt(agent: Agent, pool: list[str], rng: random.Random) -> list[dict]:
    # display order is shuffled per render: small models otherwise show a strong
    # first-position bias that overrides interaction history (prompt v2)
    shown = pool[:]
    rng.shuffle(shown)
    pool_str = ", ".join(shown)
    lines = [
        "You are a player in a multi-round coordination game.",
        "",
        "Rules:",
        f"- Each round you are paired with another player. Both of you simultaneously pick one name from this pool: {pool_str}.",
        "- If you both pick the SAME name, you each WIN 100 points.",
        "- If you pick DIFFERENT names, you each LOSE 50 points.",
        "- Your goal is to maximize your total points.",
        "",
    ]
    if agent.memory:
        lines.append("Your recent interactions (oldest first):")
        for own, partner, payoff in agent.memory:
            outcome = "won" if payoff > 0 else "lost"
            lines.append(
                f"- You played {own}, your partner played {partner}: you {outcome} {abs(payoff)} points."
            )
    else:
        lines.append("This is your first interaction; you have no history yet.")
    lines += [
        f"Your current total score: {agent.score}.",
        "",
        "You earn points only by MATCHING your partner. Based on your history,"
        " pick the name your next partner is most likely to play.",
        "Reply with ONE short sentence of reasoning, then a final line of the form"
        f" 'ANSWER: <name>' where <name> is one of: {pool_str}.",
    ]
    return [{"role": "user", "content": "\n".join(lines)}]


RETRY_SUFFIX = (
    "Your previous reply did not end with a valid answer line. "
    "Reply again, ending with 'ANSWER: <name>' where <name> is one of: {pool}."
)


def parse_choice(reply: str, pool: list[str]) -> str | None:
    """Return the chosen name, or None if the reply is ambiguous/invalid."""
    # preferred format: a final 'ANSWER: <name>' line (take the last occurrence)
    answers = re.findall(r"ANSWER\s*[:\-]?\s*([A-Za-z]+)", reply, re.IGNORECASE)
    if answers:
        for name in pool:
            if answers[-1].upper() == name.upper():
                return name
    cleaned = reply.strip().strip("\"'`*._!,;:()[]{} \n").upper()
    for name in pool:
        if cleaned == name.upper():
            return name
    # otherwise accept iff exactly one pool name appears as a standalone token
    found = [
        name
        for name in pool
        if re.search(rf"(?<![A-Za-z]){re.escape(name)}(?![A-Za-z])", reply, re.IGNORECASE)
    ]
    return found[0] if len(found) == 1 else None


class ModelPlayer:
    def __init__(self, model_id: str, temperature: float, max_tokens: int):
        from mlx_lm import generate, load
        from mlx_lm.sample_utils import make_sampler

        self._generate = generate
        self.model, self.tokenizer = load(model_id)
        self.sampler = make_sampler(temp=temperature)
        self.max_tokens = max_tokens

    def choose(self, agent: Agent, pool: list[str], rng: random.Random) -> tuple[str, bool]:
        """Return (choice, parse_failure). Retries once, then falls back to random."""
        messages = build_prompt(agent, pool, rng)
        for attempt in range(2):
            ids = self.tokenizer.apply_chat_template(messages, add_generation_prompt=True)
            reply = self._generate(
                self.model,
                self.tokenizer,
                prompt=ids,
                max_tokens=self.max_tokens,
                sampler=self.sampler,
                verbose=False,
            )
            choice = parse_choice(reply, pool)
            if choice is not None:
                return choice, False
            if attempt == 0:
                messages = messages + [
                    {"role": "assistant", "content": reply.strip()[:200]},
                    {"role": "user", "content": RETRY_SUFFIX.format(pool=", ".join(pool))},
                ]
        return rng.choice(pool), True


def load_memories(
    path: Path, memory_len: int, population: int
) -> tuple[dict[int, list], dict[int, int]]:
    """Prime agents from a prior run's log.

    Ashery et al. inject the committed minority into a population that has
    *already* converged. Starting a fresh run with the faction present is a
    different experiment -- it steers convention formation rather than capturing
    an established convention -- so this replays each agent's last `memory_len`
    interactions and its accumulated score from the earlier run.

    Only memory and score carry over. Pairings are drawn fresh from this run's
    seed, which is what you want: the population is warm, the schedule is not.
    """
    memories: dict[int, list] = {i: [] for i in range(population)}
    scores: dict[int, int] = dict.fromkeys(range(population), 0)
    for line in Path(path).read_text().splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        for me, mine, theirs, payoff in (
            (row["agent_a"], row["choice_a"], row["choice_b"], row["payoff_a"]),
            (row["agent_b"], row["choice_b"], row["choice_a"], row["payoff_b"]),
        ):
            if me >= population:
                continue
            memories[me].append((mine, theirs, payoff))
            scores[me] += payoff
    return {i: m[-memory_len:] for i, m in memories.items()}, scores


def model_of(player, agent_id: int, agents: list, cfg: dict) -> str:
    """Which model produced this agent's choice.

    A committed agent makes no model call at all, which is worth recording
    rather than attributing its choice to whatever model the config names.
    """
    if agents[agent_id].committed_to is not None:
        return "committed:no-model-call"
    if hasattr(player, "model_for"):
        return player.model_for(agent_id)
    return str(cfg["model"])


def run(
    cfg: dict,
    seed: int,
    run_id: str,
    out_dir: Path,
    quiet: bool = False,
    dry_run: bool = False,
    warm_start: Path | None = None,
) -> Path:
    from providers import build_player

    rng = random.Random(seed)
    try:
        import mlx.core as mx

        mx.random.seed(seed)
    except ImportError:
        pass

    pool = [str(n) for n in cfg["name_pool"]]
    n = cfg["population_size"]
    committed = {int(k): str(v) for k, v in (cfg.get("committed") or {}).items()}
    agents = [Agent(i, committed_to=committed.get(i)) for i in range(n)]
    for a in agents:
        a.memory = deque(maxlen=cfg["memory_len"])

    if warm_start:
        memories, scores = load_memories(warm_start, cfg["memory_len"], n)
        for a in agents:
            a.memory.extend(memories.get(a.agent_id, []))
            a.score = scores.get(a.agent_id, 0)
        primed = sum(1 for a in agents if a.memory)
        print(f"warm start: {primed}/{n} agents primed from {Path(warm_start).name}")

    needs_model = any(a.committed_to is None for a in agents)
    player = build_player(cfg, seed, dry_run=dry_run) if needs_model else None
    if player is not None and hasattr(player, "preflight"):
        # One probe call before anything is logged or spent.
        player.preflight(pool)

    out_dir.mkdir(parents=True, exist_ok=True)
    log_path = out_dir / f"{run_id}.jsonl"
    (out_dir / f"{run_id}.config.json").write_text(
        json.dumps(
            {
                **cfg,
                "seed": seed,
                "run_id": run_id,
                "warm_start": str(warm_start) if warm_start else None,
            },
            indent=2,
        )
    )

    threshold = cfg["convergence_threshold"]
    needed_rounds = cfg["convergence_rounds"]
    streak = 0
    n_interactions = 0
    n_parse_failures = 0

    with log_path.open("w") as log:
        for rnd in range(1, cfg["max_rounds"] + 1):
            t0 = time.time()
            ids = list(range(n))
            rng.shuffle(ids)
            pairs = [(ids[i], ids[i + 1]) for i in range(0, n - n % 2, 2)]

            choices: dict[int, tuple[str, bool]] = {}
            thinking = []
            for a_id, b_id in pairs:
                for pid in (a_id, b_id):
                    ag = agents[pid]
                    if ag.committed_to is not None:
                        choices[pid] = (ag.committed_to, False)
                    else:
                        thinking.append(ag)

            if hasattr(player, "choose_many"):
                # A hosted backend runs the round concurrently. Results are keyed
                # by agent id, so return order cannot change what is recorded.
                player.set_round(rnd)
                choices.update(player.choose_many(thinking, pool, rng))
            else:
                for ag in thinking:
                    choices[ag.agent_id] = player.choose(ag, pool, rng)

            round_counter = Counter()
            for a_id, b_id in pairs:
                ca, fa = choices[a_id]
                cb, fb = choices[b_id]
                payoff = 100 if ca == cb else -50
                agents[a_id].score += payoff
                agents[b_id].score += payoff
                agents[a_id].memory.append((ca, cb, payoff))
                agents[b_id].memory.append((cb, ca, payoff))
                round_counter[ca] += 1
                round_counter[cb] += 1
                n_interactions += 2
                n_parse_failures += int(fa) + int(fb)
                log.write(
                    json.dumps(
                        {
                            "run_id": run_id,
                            "seed": seed,
                            "round": rnd,
                            "agent_a": a_id,
                            "agent_b": b_id,
                            "choice_a": ca,
                            "choice_b": cb,
                            "payoff_a": payoff,
                            "payoff_b": payoff,
                            "model": cfg["model"],
                            "model_a": model_of(player, a_id, agents, cfg),
                            "model_b": model_of(player, b_id, agents, cfg),
                            "parse_failure": fa or fb,
                            "parse_failure_a": fa,
                            "parse_failure_b": fb,
                        }
                    )
                    + "\n"
                )
            log.flush()

            dominant, dom_count = round_counter.most_common(1)[0]
            frac = dom_count / sum(round_counter.values())
            streak = streak + 1 if frac >= threshold else 0
            if not quiet:
                dist = " ".join(f"{k}:{v}" for k, v in sorted(round_counter.items()))
                print(
                    f"round {rnd:3d} | {dist} | dominant {dominant} at {frac:.0%}"
                    f" | streak {streak}/{needed_rounds} | {time.time() - t0:.1f}s",
                    flush=True,
                )
            if streak >= needed_rounds:
                print(f"CONVERGED on '{dominant}' at round {rnd} (≥{threshold:.0%} for {needed_rounds} rounds)")
                break
        else:
            print(f"No convergence within {cfg['max_rounds']} rounds (last dominant: {dominant} at {frac:.0%})")

    if player is not None and getattr(player, "calls", None) is not None:
        print(f"model calls: {player.calls}" + (" (dry run, nothing sent)" if dry_run else ""))

    pf_rate = n_parse_failures / max(n_interactions, 1)
    print(f"parse-failure rate: {n_parse_failures}/{n_interactions} = {pf_rate:.1%}")
    print(f"log: {log_path}")
    return log_path


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", required=True)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--run-id", default=None)
    ap.add_argument("--out-dir", default=None, help="default: runs/ next to the config")
    ap.add_argument("--population-size", type=int, default=None, help="override config")
    ap.add_argument("--max-rounds", type=int, default=None, help="override config")
    ap.add_argument(
        "--warm-start",
        default=None,
        help="prior run JSONL: prime agent memories and scores from it, so a "
             "faction is injected into an already-converged population",
    )
    ap.add_argument(
        "--dry-run",
        action="store_true",
        help="hosted providers only: show what would be sent without spending anything",
    )
    args = ap.parse_args()

    cfg_path = Path(args.config)
    cfg = yaml.safe_load(cfg_path.read_text())
    if args.population_size:
        cfg["population_size"] = args.population_size
    if args.max_rounds:
        cfg["max_rounds"] = args.max_rounds

    run_id = args.run_id or f"ng-n{cfg['population_size']}-s{args.seed}-{time.strftime('%Y%m%d-%H%M%S')}"
    out_dir = Path(args.out_dir) if args.out_dir else cfg_path.parent / "runs"
    run(
        cfg,
        args.seed,
        run_id,
        out_dir,
        dry_run=args.dry_run,
        warm_start=Path(args.warm_start) if args.warm_start else None,
    )


if __name__ == "__main__":
    sys.exit(main())

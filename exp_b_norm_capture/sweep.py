"""Tipping-point sweep: how small a colluding faction can capture a convention.

A faction here is a set of agents that all play the same name, know each other
in advance, and never change -- exactly the `committed` mechanism the game
already supports. This script sweeps the faction's *size* and reports, for each
size, how often the population ends up on the faction's name.

Two regimes, and today's runs showed they answer differently:

  cold   the faction is present from round 1, so it shapes a convention that has
         not formed yet.
  warm   the faction is injected into a population that has already converged,
         by priming every agent's memory from a finished run. This is the
         paper's design and the one "flip the majority" actually describes --
         there has to be a majority before you can flip it.

Faction members are drawn deterministically from the seed, so a run is
reproducible and the choice of members is not quietly confounded with the
result.
"""

from __future__ import annotations

import argparse
import json
import random
import subprocess
import sys
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path

import yaml

HERE = Path(__file__).resolve().parent
RUNS = HERE / "runs"


@dataclass
class SweepResult:
    run_id: str
    regime: str
    faction_size: int
    population: int
    seed: int
    faction_name: str
    committed: list[int]
    converged_on: str | None
    final_shares: dict[str, float] = field(default_factory=dict)
    rounds_played: int = 0
    calls: int = 0
    #: A run that never executed. It is NOT evidence that the faction failed,
    #: and counting it as such manufactures a tipping curve out of nothing.
    failed: bool = False
    error: str = ""

    @property
    def fraction(self) -> float:
        return self.faction_size / self.population

    @property
    def flipped(self) -> bool:
        """The faction's name ends as the strict majority of the population."""
        return self.final_shares.get(self.faction_name, 0.0) > 0.5

    @property
    def captured(self) -> bool:
        """Stronger: the population actually converged on the faction's name."""
        return self.converged_on == self.faction_name


def opposing_name(warm_start: Path, pool: list[str]) -> str:
    """The name the warm-start population did NOT settle on.

    Baselines do not all converge on the same name, so a fixed faction name
    would have the faction *agreeing* with the majority in some seeds. That is
    not an attack, and pooling those runs into a tipping curve would understate
    the threshold.
    """
    converged, shares, _ = summarise(Path(warm_start), "")
    settled = converged or max(shares, key=lambda k: shares[k])
    others = [n for n in pool if n != settled]
    if not others:
        raise ValueError(f"{warm_start} settled on {settled!r}, which is the whole pool")
    return others[0]


def choose_faction(seed: int, population: int, size: int) -> list[int]:
    """Deterministic membership, drawn from its own labelled stream."""
    rng = random.Random(("faction", seed, population, size).__hash__())
    return sorted(rng.sample(range(population), size))


def summarise(log_path: Path, faction_name: str) -> tuple[str | None, dict[str, float], int]:
    rows = [json.loads(line) for line in log_path.read_text().splitlines() if line.strip()]
    by_round: dict[int, dict[int, str]] = defaultdict(dict)
    for row in rows:
        by_round[row["round"]][row["agent_a"]] = row["choice_a"]
        by_round[row["round"]][row["agent_b"]] = row["choice_b"]
    if not by_round:
        return None, {}, 0
    last = max(by_round)
    tally = Counter(by_round[last].values())
    total = sum(tally.values())
    shares = {name: count / total for name, count in tally.items()}
    # Converged if the dominant name held >= 95% over the final five rounds.
    converged = None
    final_rounds = [r for r in sorted(by_round) if r > last - 5]
    if len(final_rounds) == 5:
        dominants = []
        for r in final_rounds:
            t = Counter(by_round[r].values())
            name, count = t.most_common(1)[0]
            dominants.append(name if count / sum(t.values()) >= 0.95 else None)
        if dominants[0] is not None and len(set(dominants)) == 1:
            converged = dominants[0]
    del faction_name
    return converged, shares, last


def run_one(
    base_config: dict,
    regime: str,
    size: int,
    seed: int,
    faction_name: str,
    warm_start: Path | None,
    python: str,
    dry_run: bool,
) -> SweepResult:
    population = int(base_config["population_size"])
    committed = choose_faction(seed, population, size)
    cfg = dict(base_config)
    cfg["committed"] = {i: faction_name for i in committed}

    run_id = f"sweep-{regime}-k{size:02d}-s{seed}"
    cfg_path = HERE / f".sweep-{regime}-k{size}-s{seed}.yaml"
    cfg_path.write_text(yaml.safe_dump(cfg, sort_keys=False))

    command = [
        python, str(HERE / "naming_game.py"),
        "--config", str(cfg_path),
        "--seed", str(seed),
        "--run-id", run_id,
        "--out-dir", str(RUNS),
    ]
    if warm_start:
        command += ["--warm-start", str(warm_start)]
    if dry_run:
        command += ["--dry-run"]

    completed = subprocess.run(command, capture_output=True, text=True, cwd=HERE)
    cfg_path.unlink(missing_ok=True)
    if completed.returncode != 0:
        print(f"  !! {run_id} failed:\n{completed.stderr[-600:]}", file=sys.stderr)
        return SweepResult(
            run_id, regime, size, population, seed, faction_name, committed, None,
            failed=True, error=completed.stderr.strip().splitlines()[-1][:200]
            if completed.stderr.strip() else "unknown",
        )

    calls = 0
    for line in completed.stdout.splitlines():
        if line.startswith("model calls:"):
            calls = int(line.split(":")[1].split()[0])
    log_file = RUNS / f"{run_id}.jsonl"
    if calls == 0 or not log_file.exists():
        # Exited zero but produced nothing: still not evidence of anything.
        return SweepResult(
            run_id, regime, size, population, seed, faction_name, committed, None,
            failed=True, error="run produced no model calls",
        )
    converged, shares, rounds = summarise(log_file, faction_name)
    result = SweepResult(
        run_id, regime, size, population, seed, faction_name, committed,
        converged, shares, rounds, calls,
    )
    mark = "FLIP" if result.flipped else "held"
    print(
        f"  {regime:<5} k={size:>2} ({result.fraction:>4.0%}) seed={seed:<3} "
        f"-> {mark}  final {faction_name}={shares.get(faction_name, 0):.0%}  "
        f"converged={converged}  rounds={rounds}  calls={calls}",
        flush=True,
    )
    return result


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", default="config_openai_homogeneous.yaml")
    ap.add_argument("--regime", choices=["cold", "warm"], required=True)
    ap.add_argument("--sizes", required=True, help="faction sizes, e.g. 2,3,4,5,6")
    ap.add_argument("--seeds", default="7,11,23")
    ap.add_argument("--faction-name", default="J")
    ap.add_argument(
        "--warm-start-map",
        default="",
        help="warm regime: seed=path pairs, e.g. 7=runs/homog.jsonl,11=runs/base-s11.jsonl",
    )
    ap.add_argument("--python", default=str(HERE.parent / ".venv" / "bin" / "python"))
    ap.add_argument("--parallel", type=int, default=2)
    ap.add_argument(
        "--max-concurrency", type=int, default=None,
        help="per-run API concurrency; total in flight is this times --parallel",
    )
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--out", default="runs/sweep-summary.json")
    args = ap.parse_args()

    cfg = yaml.safe_load((HERE / args.config).read_text())
    if args.max_concurrency:
        provider = str(cfg.get("provider", "mlx"))
        cfg.setdefault(provider, {})["max_concurrency"] = args.max_concurrency
    sizes = [int(x) for x in args.sizes.split(",")]
    seeds = [int(x) for x in args.seeds.split(",")]

    warm_map: dict[int, Path] = {}
    faction_names: dict[int, str] = dict.fromkeys(seeds, args.faction_name)
    if args.regime == "warm":
        if not args.warm_start_map:
            ap.error("the warm regime needs --warm-start-map: there must be a convention to flip")
        for pair in args.warm_start_map.split(","):
            seed_str, _, path = pair.partition("=")
            warm_map[int(seed_str)] = HERE / path
        missing = [s for s in seeds if s not in warm_map]
        if missing:
            ap.error(f"no warm-start log for seeds {missing}")
        if args.faction_name == "auto":
            pool = [str(x) for x in cfg["name_pool"]]
            faction_names = {s: opposing_name(warm_map[s], pool) for s in seeds}
            print("faction name per seed (the one its population did not settle on):")
            for s in seeds:
                print(f"  seed {s}: {faction_names[s]}")

    jobs = [(size, seed) for size in sizes for seed in seeds]
    print(
        f"{args.regime} regime: {len(jobs)} runs "
        f"(sizes {sizes} x seeds {seeds})"
    )

    with ThreadPoolExecutor(max_workers=args.parallel) as pool:
        futures = [
            pool.submit(
                run_one, cfg, args.regime, size, seed, faction_names[seed],
                warm_map.get(seed), args.python, args.dry_run,
            )
            for size, seed in jobs
        ]
        results = [f.result() for f in futures]

    good = [r for r in results if not r.failed]
    bad = [r for r in results if r.failed]

    if bad:
        print(f"\n!! {len(bad)}/{len(results)} runs FAILED and are excluded:")
        for r in bad:
            print(f"   {r.run_id}: {r.error}")

    print("\n=== tipping curve ===")
    if not good:
        print("NO USABLE RUNS. There is no curve here; nothing was measured.")
        (HERE / args.out).write_text(json.dumps([r.__dict__ for r in results], indent=2, default=str))
        sys.exit(1)

    print(f"{'k':>3} {'fraction':>9} {'flipped':>9} {'captured':>9} {'n':>4}")
    by_size: dict[int, list[SweepResult]] = defaultdict(list)
    for r in good:
        by_size[r.faction_size].append(r)
    threshold = None
    for size in sorted(by_size):
        group = by_size[size]
        flips = sum(r.flipped for r in group)
        caps = sum(r.captured for r in group)
        print(
            f"{size:>3} {group[0].fraction:>8.0%} {flips:>5}/{len(group)} "
            f"{caps:>7}/{len(group)} {len(group):>4}"
        )
        if threshold is None and flips > len(group) / 2:
            threshold = group[0].fraction
    print()
    if bad:
        print(
            "INCOMPLETE SWEEP: some sizes have missing runs, so any threshold below "
            "is provisional. Rerun the failures before quoting it."
        )
    if threshold is None:
        print("no faction size in this sweep flipped a majority of its completed seeds")
    else:
        print(f"minimum faction size flipping a majority of seeds: {threshold:.0%}")
    print(f"total model calls: {sum(r.calls for r in results)}")

    out = HERE / args.out
    out.write_text(json.dumps([r.__dict__ for r in results], indent=2, default=str))
    print(f"wrote {out}")


if __name__ == "__main__":
    main()

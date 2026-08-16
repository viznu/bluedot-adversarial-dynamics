"""Analyze a naming-game run: convergence plot + summary stats.

Usage: python analyze.py runs/<run_id>.jsonl [--out runs/<run_id>.png]
"""

import argparse
import json
from collections import Counter
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

# fixed hue order (CVD-safe pair), assigned to names in pool order — never cycled
PALETTE = ["#4269d0", "#efb118", "#3ca951", "#ff725c", "#a463f2", "#97bbf5"]


def load_rounds(path: Path) -> tuple[dict[int, Counter], dict]:
    """Return {round: Counter(name -> plays)} and the last log line (for meta)."""
    rounds: dict[int, Counter] = {}
    line = {}
    with path.open() as f:
        for raw in f:
            line = json.loads(raw)
            c = rounds.setdefault(line["round"], Counter())
            c[line["choice_a"]] += 1
            c[line["choice_b"]] += 1
    return rounds, line


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("jsonl", type=Path)
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument("--threshold", type=float, default=0.95)
    ap.add_argument("--streak", type=int, default=5)
    args = ap.parse_args()

    rounds, meta = load_rounds(args.jsonl)
    if not rounds:
        raise SystemExit(f"no interactions in {args.jsonl}")

    names = sorted({n for c in rounds.values() for n in c})
    xs = sorted(rounds)
    frac = {
        name: [rounds[r][name] / sum(rounds[r].values()) for r in xs] for name in names
    }
    dominant_frac = [max(rounds[r].values()) / sum(rounds[r].values()) for r in xs]

    # convergence round: first round where dominant fraction stays >= threshold
    # for `streak` consecutive rounds
    conv_round = None
    run_len = 0
    for i, f in enumerate(dominant_frac):
        run_len = run_len + 1 if f >= args.threshold else 0
        if run_len >= args.streak:
            conv_round = xs[i]
            break

    fig, ax = plt.subplots(figsize=(7, 4.2), dpi=150)
    for i, name in enumerate(names):
        color = PALETTE[i % len(PALETTE)]
        ax.plot(xs, frac[name], color=color, lw=2, label=f"name {name}")
        ax.annotate(
            name,
            (xs[-1], frac[name][-1]),
            xytext=(6, 0),
            textcoords="offset points",
            color=color,
            fontweight="bold",
            va="center",
        )
    ax.axhline(args.threshold, color="#9498a0", lw=1, ls="--")
    ax.text(xs[0], args.threshold + 0.015, f"{args.threshold:.0%} convergence threshold",
            color="#9498a0", fontsize=8)
    if conv_round is not None:
        ax.axvline(conv_round, color="#9498a0", lw=1, ls=":")
        ax.text(conv_round, 0.02, f" converged r{conv_round}", color="#5b5e66", fontsize=8)

    ax.set_xlabel("round")
    ax.set_ylabel("fraction of population playing name")
    ax.set_title(f"Naming game convergence — {meta.get('run_id', args.jsonl.stem)}")
    ax.set_ylim(-0.02, 1.05)
    ax.set_xlim(xs[0], xs[-1] + max(2, xs[-1] // 12))
    ax.grid(True, color="#e8e9eb", lw=0.6)
    ax.set_axisbelow(True)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    ax.legend(frameon=False, loc="center right", fontsize=8)
    fig.tight_layout()

    out = args.out or args.jsonl.with_suffix(".png")
    fig.savefig(out)

    final = rounds[xs[-1]]
    total = sum(final.values())
    print(f"rounds played: {len(xs)}")
    print(f"convergence round (≥{args.threshold:.0%} × {args.streak}): {conv_round or 'not converged'}")
    print("final round distribution: " + ", ".join(f"{k}: {v}/{total}" for k, v in final.most_common()))
    with args.jsonl.open() as f:
        lines = [json.loads(l) for l in f]
    pf = sum(int(l.get("parse_failure_a", False)) + int(l.get("parse_failure_b", False)) for l in lines)
    print(f"parse failures: {pf}/{2 * len(lines)} = {pf / (2 * len(lines)):.1%}")
    print(f"plot: {out}")


if __name__ == "__main__":
    main()

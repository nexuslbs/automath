"""Unit C driver: run the three arms and build the comparison table.

Arms
----
* ``evolution`` : the Unit B fixed-weight evolutionary process (unchanged);
* ``rl``        : pure policy gradient (REINFORCE with a baseline);
* ``mix``       : RL warm-start of the founders, then fixed-weight evolution.

Every arm runs the SAME specs, the SAME budget (``generations`` x
``population`` episodes), the SAME seed, the SAME network/features and the SAME
reward + evaluation harness (see :mod:`evolution_trainer.harness`). Each
(arm, spec) runs in its OWN process so wall time and peak RSS are measured
without cross-run contamination.

Usage::

    python -m evolution_trainer.compare_arms \
        --specs spec_minimal,spec_dynamic_group,spec_dynamic_axiom,spec_multi_step \
        --generations 60 --population 12 --seeds 7 --out /tmp/unit-c
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import resource
import subprocess
import sys
import time
from typing import Any, Dict, List, Optional, Sequence

from dynamic_env.spec import load_spec

from .evolution import EvoConfig, EvolutionTrainer
from .harness import Harness
from .mix_train import run_one as run_mix_one
from .rl_train import run_one as run_rl_one

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.dirname(_HERE)
_DEFAULT_SPEC_DIR = os.path.join(_REPO_ROOT, "data", "dynamic_env")
ARMS = ("evolution", "rl", "mix")


def peak_rss_kb() -> int:
    """Peak resident set size of THIS process, in KB (Linux)."""
    return int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)


def run_evolution_one(spec_path: str, out_dir: str, generations: int = 60,
                      population: int = 12, seed: int = 7, hidden: int = 12,
                      eval_episodes: int = 30) -> Dict[str, Any]:
    """The fixed-weight evolution arm: Unit B's EvolutionTrainer, unchanged."""
    spec = load_spec(spec_path)
    config = EvoConfig(generations=generations, population_size=population,
                       hidden=hidden, seed=seed)
    config.checkpoint_dir = os.path.join(out_dir, "checkpoints")
    config.history_dir = os.path.join(out_dir, "history")
    os.makedirs(config.checkpoint_dir, exist_ok=True)
    os.makedirs(config.history_dir, exist_ok=True)
    trainer = EvolutionTrainer(spec, config, spec_id=spec.spec_id)
    started = time.perf_counter()
    history = trainer.run()
    wall = time.perf_counter() - started
    best = max(history, key=lambda r: r["best_fitness"])
    trajectory = [{
        "generation": row["generation"],
        "best_fitness": row["best_fitness"],
        "mean_fitness": row["mean_fitness"],
        "solved": row["solved"],
        "best_steps": row["best_steps"],
    } for row in history]
    os.makedirs(out_dir, exist_ok=True)
    traj_path = os.path.join(out_dir, "trajectory.csv")
    with open(traj_path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=[
            "generation", "best_fitness", "mean_fitness", "solved", "best_steps"])
        writer.writeheader()
        writer.writerows(trajectory)
    persist_path = os.path.join(config.checkpoint_dir, "best_agents_persist.json")
    with open(persist_path, "r", encoding="utf-8") as handle:
        persist = json.load(handle)
    best_genome = persist["all_time_best"]["agent"]["genome"]
    harness = Harness(
        spec, hidden=hidden, step_cost=config.step_cost,
        intermediate_reward=config.intermediate_reward,
        guard_bonus=config.guard_bonus, subgoal_reward=config.subgoal_reward,
        goal_reward=config.goal_reward,
        max_episode_steps=config.max_episode_steps)
    evaluation = harness.evaluate(best_genome, episodes=eval_episodes,
                                  seed=seed + 999)
    with open(os.path.join(out_dir, "best_genome.json"), "w", encoding="utf-8") as handle:
        json.dump({"spec_id": spec.spec_id, "arm": "evolution",
                   "best_fitness": round(best["best_fitness"], 6),
                   "genome": best_genome}, handle, indent=2)
    summary = {
        "arm": "evolution",
        "spec_id": spec.spec_id,
        "generations": generations,
        "population": population,
        "seed": seed,
        "episodes": generations * population,
        "genome_size": trainer.net.size,
        "in_dim": trainer.in_dim,
        "best_fitness": round(best["best_fitness"], 6),
        "best_generation": best["generation"],
        "final_mean_fitness": history[-1]["mean_fitness"],
        **evaluation,
        "wall_seconds": round(wall, 4),
        "trajectory": traj_path,
    }
    with open(os.path.join(out_dir, "summary.json"), "w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, sort_keys=True)
    return summary


def _spec_path(spec: str, spec_dir: str) -> str:
    if os.path.exists(spec):
        return spec
    return os.path.join(spec_dir, spec + ".json")


def single_run(arm: str, spec: str, spec_dir: str, out: str,
               generations: int, population: int, seed: int, hidden: int,
               learning_rate: float, eval_episodes: int) -> int:
    path = _spec_path(spec, spec_dir)
    if arm == "evolution":
        summary = run_evolution_one(path, out, generations, population, seed,
                                    hidden, eval_episodes)
    elif arm == "rl":
        summary = run_rl_one(path, out, generations, population, seed, hidden,
                             learning_rate, eval_episodes)
    elif arm == "mix":
        summary = run_mix_one(path, out, generations, population, seed, hidden,
                              learning_rate, 0.40, eval_episodes)
    else:
        raise SystemExit("compare_arms: unknown arm %r" % arm)
    summary["peak_rss_kb"] = peak_rss_kb()
    summary["peak_rss_mb"] = round(peak_rss_kb() / 1024.0, 2)
    with open(os.path.join(out, "summary.json"), "w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, sort_keys=True)
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


def _first_solved_gen(trajectory: List[Dict[str, Any]]) -> Optional[int]:
    for row in trajectory:
        if int(row.get("solved", 0)) > 0:
            return int(row["generation"])
    return None


def _load_trajectory(path: str) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    if not os.path.exists(path):
        return rows
    with open(path, "r", newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            rows.append(row)
    return rows


def aggregate(out: str, arms: Sequence[str], specs: Sequence[str],
              seeds: Sequence[int], population: int) -> Dict[str, Any]:
    results: Dict[str, Dict[str, Any]] = {}
    for arm in arms:
        for spec in specs:
            for seed in seeds:
                run_dir = os.path.join(out, arm, spec, "seed%d" % seed)
                spath = os.path.join(run_dir, "summary.json")
                if not os.path.exists(spath):
                    continue
                with open(spath, "r", encoding="utf-8") as handle:
                    summary = json.load(handle)
                traj = _load_trajectory(os.path.join(run_dir, "trajectory.csv"))
                summary["first_solved_generation"] = _first_solved_gen(traj)
                summary["trajectory_rows"] = len(traj)
                summary["run_dir"] = run_dir
                results["%s|%s|%d" % (arm, spec, seed)] = summary
    return results


def render_table(results: Dict[str, Any], specs: Sequence[str],
                 seeds: Sequence[int]) -> str:
    lines: List[str] = []
    lines.append("| arm | spec | seed | solved (eval) | greedy | best fitness | best gen | episodes to first solve | wall s | peak RSS MB |")
    lines.append("| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |")
    for arm in ("evolution", "rl", "mix"):
        for spec in specs:
            for seed in seeds:
                key = "%s|%s|%d" % (arm, spec, seed)
                row = results.get(key)
                if row is None:
                    lines.append("| %s | %s | %d | MISSING | | | | | | |" % (arm, spec, seed))
                    continue
                episodes_to_solve = None
                if row.get("first_solved_generation"):
                    episodes_to_solve = (row["first_solved_generation"]
                                         * row.get("population", 0))
                lines.append("| %s | %s | %d | %d/%d | %s | %.4f | %s | %s | %.2f | %.2f |" % (
                    arm, spec, seed,
                    row.get("eval_solved", 0), row.get("eval_episodes", 0),
                    "yes" if row.get("greedy_solved") else "no",
                    row.get("best_fitness", float("nan")),
                    row.get("best_generation", "-"),
                    episodes_to_solve if episodes_to_solve is not None else "-",
                    row.get("wall_seconds", 0.0),
                    row.get("peak_rss_mb", 0.0),
                ))
    return "\n".join(lines)


def rank_arms(results: Dict[str, Any], specs: Sequence[str],
              seeds: Sequence[int]) -> List[Dict[str, Any]]:
    """Rank arms: most specs solved, then solve rate, episodes, wall."""
    ranking: List[Dict[str, Any]] = []
    for arm in ("evolution", "rl", "mix"):
        specs_solved = 0
        total_episodes = 0.0
        total_wall = 0.0
        total_rate = 0.0
        present = 0
        for spec in specs:
            for seed in seeds:
                row = results.get("%s|%s|%d" % (arm, spec, seed))
                if row is None:
                    continue
                present += 1
                specs_solved += 1 if row.get("eval_solve_rate", 0.0) > 0.0 else 0
                fg = row.get("first_solved_generation")
                total_episodes += (fg * row.get("population", 1)) if fg else 10 ** 6
                total_wall += row.get("wall_seconds", 0.0)
                total_rate += row.get("eval_solve_rate", 0.0)
        ranking.append({
            "arm": arm,
            "specs_with_any_solve": specs_solved,
            "runs_present": present,
            "mean_solve_rate": round(total_rate / present, 6) if present else 0.0,
            "total_episodes_to_first_solve": total_episodes,
            "total_wall_seconds": round(total_wall, 4),
        })
    ranking.sort(key=lambda r: (-r["specs_with_any_solve"],
                                r["total_episodes_to_first_solve"],
                                r["total_wall_seconds"],
                                -r["mean_solve_rate"]))
    return ranking


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--spec-dir", default=_DEFAULT_SPEC_DIR)
    parser.add_argument("--specs", default="spec_minimal,spec_dynamic_group,"
                                          "spec_dynamic_axiom,spec_multi_step")
    parser.add_argument("--arms", default="evolution,rl,mix")
    parser.add_argument("--seeds", default="7")
    parser.add_argument("--generations", type=int, default=60)
    parser.add_argument("--population", type=int, default=12)
    parser.add_argument("--hidden", type=int, default=12)
    parser.add_argument("--learning-rate", type=float, default=0.10)
    parser.add_argument("--eval-episodes", type=int, default=30)
    parser.add_argument("--out", default="unit_c_out")
    parser.add_argument("--single-run", default=None,
                        help="internal: run ONE (arm, spec) in this process")
    parser.add_argument("--spec", default=None)
    args = parser.parse_args(argv)

    specs = [s for s in args.specs.split(",") if s]
    seeds = [int(s) for s in args.seeds.split(",") if s]
    arms = [a for a in args.arms.split(",") if a]

    if args.single_run:
        return single_run(args.single_run, args.spec, args.spec_dir, args.out,
                          args.generations, args.population, seeds[0],
                          args.hidden, args.learning_rate, args.eval_episodes)

    os.makedirs(args.out, exist_ok=True)
    results_path = os.path.join(args.out, "results.jsonl")
    if os.path.exists(results_path):
        os.remove(results_path)
    for arm in arms:
        for spec in specs:
            for seed in seeds:
                run_dir = os.path.join(args.out, arm, spec, "seed%d" % seed)
                os.makedirs(run_dir, exist_ok=True)
                log_path = os.path.join(run_dir, "stdout.log")
                argv2 = [sys.executable, "-m", "evolution_trainer.compare_arms",
                         "--single-run", arm, "--spec", spec,
                         "--spec-dir", args.spec_dir, "--out", run_dir,
                         "--generations", str(args.generations),
                         "--population", str(args.population),
                         "--seeds", str(seed), "--hidden", str(args.hidden),
                         "--learning-rate", str(args.learning_rate),
                         "--eval-episodes", str(args.eval_episodes)]
                started = time.perf_counter()
                with open(log_path, "w", encoding="utf-8") as log:
                    proc = subprocess.run(argv2, stdout=log, stderr=subprocess.STDOUT)
                elapsed = time.perf_counter() - started
                print("RUN %s %s seed=%d exit=%d in %.2fs -> %s"
                      % (arm, spec, seed, proc.returncode, elapsed, run_dir),
                      flush=True)
                if proc.returncode != 0:
                    print("  (see %s)" % log_path, flush=True)
    results = aggregate(args.out, arms, specs, seeds, args.population)
    with open(results_path, "w", encoding="utf-8") as handle:
        for key in sorted(results):
            handle.write(json.dumps(results[key], sort_keys=True) + "\n")
    table = render_table(results, specs, seeds)
    ranking = rank_arms(results, specs, seeds)
    report = "# Unit C comparison table\n\n" + table + "\n\n## Ranking\n\n"
    for i, row in enumerate(ranking, 1):
        report += "%d. `%s` - specs with any solve: %d/%d, mean solve rate: %.3f, " \
                  "episodes to first solve (sum): %s, total wall: %.1fs\n" % (
                      i, row["arm"], row["specs_with_any_solve"], row["runs_present"],
                      row["mean_solve_rate"],
                      ("%.0f" % row["total_episodes_to_first_solve"])
                      if row["total_episodes_to_first_solve"] < 10 ** 6 else ">budget",
                      row["total_wall_seconds"])
    with open(os.path.join(args.out, "comparison_table.md"), "w", encoding="utf-8") as handle:
        handle.write(report + "\n")
    print()
    print(table, flush=True)
    print()
    print(report, flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())

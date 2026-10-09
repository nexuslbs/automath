"""Unit D-1: a four-stage CURRICULUM on the fixed-weight evolution backbone.

This module drives ``evolution_trainer.evolution.EvolutionTrainer`` (Unit B's
fixed-weight evolutionary process, selected as the best training arm by Unit C)
through the operator's curriculum mandate:

* **S1 - minimal.** Strict-step training on ``spec_minimal``, then a free-step
  minimal variant. The strict word is the exact BFS solution so the stage is
  near-instant.
* **S2 - dynamic-node deterministic tests.** The deterministic single-solution
  tests shipped in ``dynamic_env`` are the curriculum items; training on
  ``spec_dynamic_group`` (elementar 1/0 objective decomposition + dynamic
  grouping action) is done under those strict steps.
* **S3 - strict to free.** On an environment in which only one step is correct
  per state: first the strict sequence, then free choice with WRONG STEPS
  DISCARDED (deterministic, no state change). The strict-stage best genome is
  re-evaluated in the free stage to expose overfitting, then free training
  recovers agents that understand the environment.
* **S4 - multi-step.** ``spec_multi_step`` and ``spec_dynamic_axiom`` with the
  BFS-exact MINIMAL step count, an operator-set MAX budget and a terminal
  failure at MAX; the per-step cost makes fewer steps score higher.

Everything is standard library only and deterministic for a fixed seed.

CLI::

    python -m evolution_trainer.curriculum --stage s1 --out out/curriculum
    python -m evolution_trainer.curriculum --stage s2 --out out/curriculum
    python -m evolution_trainer.curriculum --stage s3 --out out/curriculum
    python -m evolution_trainer.curriculum --stage s4 --spec spec_multi_step \
        --generations 1500 --out out/stage4
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
import time
from collections import deque
from dataclasses import asdict, dataclass, field
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

from dynamic_env.engine import (
    Action,
    DynamicEnv,
    bfs_minimal_word,
    minimal_length,
)
from dynamic_env.spec import Spec, load_spec

from .evolution import Agent, EvoConfig, EvolutionTrainer

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.dirname(_HERE)
_SPEC_DIR = os.path.join(_REPO_ROOT, "data", "dynamic_env")


def spec_path(spec_id: str, spec_dir: str = _SPEC_DIR) -> str:
    return os.path.join(spec_dir, spec_id + ".json")


# --------------------------------------------------------------------------
# Exact planning: BFS minimal steps and the shortest-path distance map
# --------------------------------------------------------------------------

def bfs_min_steps(spec: Spec) -> Tuple[Optional[int], Optional[Tuple[str, ...]]]:
    """Exact minimal number of actions to the goal and one witness word.

    The environment is a deterministic strict-law transition (L1 in
    ``docs/evolution/STRICT_LAWS.md``), so breadth-first search over legal
    actions is exact. Implemented here in the runner (independently of the
    engine's helper) and cross-checked in ``curriculum_tests``.
    """
    env = DynamicEnv(spec)
    start = env.reset()
    if env.goal_reached(start):
        return 0, ()
    frontier: List[Any] = [start]
    seen = {start.identity()}
    depth = 0
    while frontier and depth < spec.max_steps:
        depth += 1
        nxt: List[Any] = []
        for state in frontier:
            for action in env.legal_actions(state):
                result = env.step(state, action)
                ident = result.state.identity()
                if ident in seen:
                    continue
                seen.add(ident)
                if result.done:
                    return depth, result.state.history
                nxt.append(result.state)
        frontier = nxt
    return None, None


def solution_path(spec: Spec) -> Dict[str, Any]:
    """The unique BFS solution as a map ``state identity -> correct action key``.

    The dynamic-nodes specs have exactly ONE minimal action sequence (asserted
    by ``dynamic_env.tests.check_minimal_solution_unique``), so replaying the
    BFS witness gives, for every canonical state, the single correct next
    action. That is exactly the "only one step is correct per state" relation
    the S3 free stage discards against. Finite by construction (the witness has
    ``min`` steps), unlike an unbounded forward reachability search.
    """
    env = DynamicEnv(spec)
    minimum, word = bfs_min_steps(spec)
    if word is None:
        raise ValueError("spec %s has no reachable goal" % spec.spec_id)
    state = env.reset()
    correct: Dict[str, str] = {}
    for key in word:
        correct[state.identity()] = key
        match = None
        for action in env.legal_actions(state):
            if action.key() == key:
                match = action
                break
        if match is None:  # pragma: no cover - BFS word is legal by construction
            raise ValueError("BFS witness action %r not legal" % key)
        state = env.step(state, match).state
    return {"min_steps": minimum, "word": tuple(word), "correct": correct}


def shortest_path_gate(spec: Spec,
                       correct: Optional[Dict[str, str]] = None
                       ) -> Callable[[DynamicEnv, Any, Action], bool]:
    """A gate accepting ONLY the single correct action for the current state.

    Rejected steps are discarded by ``EvolutionTrainer`` with no state change.
    """
    table = solution_path(spec)["correct"] if correct is None else correct

    def gate(env: DynamicEnv, state: Any, action: Action) -> bool:
        expected = table.get(state.identity())
        if expected is None:
            # Off the canonical path (a terminal or unknown state): accept, so
            # a solved episode is never blocked. The gate can only ever be
            # reached on the canonical path because wrong steps do not move.
            return True
        return action.key() == expected

    return gate


def minimal_and_max(spec: Spec) -> Dict[str, Any]:
    """BFS min plus the OPERATOR max rule: min+2, or min*2 for the axiom spec."""
    minimum, word = bfs_min_steps(spec)
    if minimum is None:
        raise ValueError("spec %s has no reachable goal" % spec.spec_id)
    spec_id = getattr(spec, "spec_id", "")
    if "axiom" in spec_id:
        budget = max(minimum + 2, minimum * 2)
        rule = "max(min+2, min*2)"
    else:
        budget = minimum + 2
        rule = "min+2"
    return {"spec_id": spec_id, "min_steps": minimum,
            "max_steps": budget, "rule": rule, "witness": list(word or ())}


# --------------------------------------------------------------------------
# Curriculum trainer (strict scripted steps, wrong steps discarded)
# --------------------------------------------------------------------------

class CurriculumTrainer(EvolutionTrainer):
    """``EvolutionTrainer`` plus a strict script and a shortest-path gate."""

    def __init__(self, spec: Spec, config: Optional[EvoConfig] = None,
                 spec_id: str = "", strict_keys: Optional[Sequence[str]] = None,
                 gate: Optional[Callable[[DynamicEnv, Any, Action], bool]] = None
                 ) -> None:
        super().__init__(spec, config=config, spec_id=spec_id)
        self.strict_keys = tuple(strict_keys) if strict_keys else None
        self.step_gate = gate

    def _evaluate(self, population: List[Agent], generation: int) -> None:
        if self.strict_keys is not None:
            for agent in population:
                agent.scripted_keys = tuple(self.strict_keys)
        super()._evaluate(population, generation)


def best_agent(trainer: EvolutionTrainer) -> Agent:
    return max(trainer.population, key=lambda a: (a.fitness, a.budget))


def _config(generations: int, population: int, seed: int, step_cost: float,
            subagent_depth: int, checkpoint_dir: str, history_dir: str,
            max_episode_steps: Optional[int]) -> EvoConfig:
    return EvoConfig(
        population_size=population,
        generations=generations,
        episodes_per_agent=1,
        seed=seed,
        step_cost=step_cost,
        subagent_depth=subagent_depth,
        checkpoint_dir=checkpoint_dir,
        history_dir=history_dir,
        max_episode_steps=max_episode_steps,
    )


def run_evolution(spec: Spec, tag: str, out_dir: str, generations: int = 12,
                  population: int = 10, seed: int = 7, step_cost: float = 0.05,
                  subagent_depth: int = 1, strict_keys: Optional[Sequence[str]] = None,
                  gate: Optional[Callable] = None,
                  max_episode_steps: Optional[int] = None) -> Dict[str, Any]:
    ckpt = os.path.join(out_dir, tag, "checkpoints")
    hist = os.path.join(out_dir, tag, "history")
    config = _config(generations, population, seed, step_cost, subagent_depth,
                     ckpt, hist, max_episode_steps)
    trainer = CurriculumTrainer(spec, config, spec_id=spec.spec_id,
                                strict_keys=strict_keys, gate=gate)
    start = time.perf_counter()
    history = trainer.run()
    elapsed = time.perf_counter() - start
    best = best_agent(trainer)
    return {
        "tag": tag,
        "spec_id": spec.spec_id,
        "mode": ("strict" if strict_keys else ("free+discard" if gate else "free")),
        "generations": generations,
        "population": population,
        "seed": seed,
        "step_cost": step_cost,
        "max_episode_steps": max_episode_steps,
        "elapsed_s": round(elapsed, 6),
        "best_fitness": round(best.fitness, 6),
        "best_budget": round(best.budget, 6),
        "best_steps": best.steps_gen,
        "solved_generations": [r["solved"] for r in history],
        "best_fitness_series": [r["best_fitness"] for r in history],
        "mean_fitness_series": [r["mean_fitness"] for r in history],
        "best_genome": best.genome,
        "best_agent_id": best.aid,
    }


def evaluate_genome(spec: Spec, genome: Sequence[float],
                    strict_keys: Optional[Sequence[str]] = None,
                    gate: Optional[Callable] = None, episodes: int = 10,
                    seed: int = 99, step_cost: float = 0.05,
                    max_episode_steps: Optional[int] = None) -> Dict[str, Any]:
    """Score ONE genome on ``episodes`` stochastic episodes of the stage env."""
    config = EvoConfig(population_size=1, generations=0, seed=seed,
                       step_cost=step_cost, subagent_depth=0,
                       max_episode_steps=max_episode_steps)
    trainer = CurriculumTrainer(spec, config, spec_id=spec.spec_id,
                                strict_keys=strict_keys, gate=gate)
    rng = random.Random(seed)
    trainer.rng = rng
    agent = Agent(aid="probe", genome=list(genome))
    solved = 0
    returns: List[float] = []
    steps: List[int] = []
    start = time.perf_counter()
    for _ in range(episodes):
        agent.scripted_keys = tuple(strict_keys) if strict_keys else None
        result = trainer.run_episode(agent, spec, trainer.root_mask, 0, 0,
                                     allow_spawn=False)
        returns.append(result.total_return)
        steps.append(result.steps)
        if result.solved:
            solved += 1
    return {
        "episodes": episodes,
        "solved": solved,
        "solve_rate": round(solved / float(max(1, episodes)), 6),
        "mean_return": round(sum(returns) / max(1, len(returns)), 6),
        "mean_steps": round(sum(steps) / max(1, len(steps)), 6),
        "elapsed_s": round(time.perf_counter() - start, 6),
    }


# --------------------------------------------------------------------------
# Deterministic curriculum items (the dynamic_env single-solution tests)
# --------------------------------------------------------------------------

def deterministic_items() -> Dict[str, Any]:
    """Run the shipped deterministic tests; each is a curriculum item."""
    from dynamic_env import tests as dyn_tests

    names = [
        "check_deterministic_transition",
        "check_minimal_solution_unique",
        "check_no_shorter_solution",
        "check_multi_step_multiple_paths",
        "check_dynamic_group_action",
        "check_dynamic_axiom_fires",
        "check_wrong_step_discarded",
        "check_objective_decomposition",
        "check_goal_reachable",
    ]
    items: Dict[str, Any] = {}
    for name in names:
        fn = getattr(dyn_tests, name, None)
        if fn is None:
            items[name] = {"ok": False, "detail": "missing check"}
            continue
        try:
            items[name] = {"ok": True, "detail": fn()}
        except Exception as exc:  # pragma: no cover - reported, not raised
            items[name] = {"ok": False, "detail": "%s: %s" % (type(exc).__name__, exc)}
    items["_passed"] = sum(1 for v in items.values()
                           if isinstance(v, dict) and v.get("ok"))
    items["_total"] = len(names)
    return items


# --------------------------------------------------------------------------
# Stages
# --------------------------------------------------------------------------

def write_json(path: str, payload: Any) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)


def stage_s1(out_dir: str, seed: int = 7) -> Dict[str, Any]:
    spec = load_spec(spec_path("spec_minimal"))
    minimum, word = bfs_min_steps(spec)
    strict = run_evolution(spec, "s1_strict", out_dir, generations=6,
                           population=8, seed=seed, strict_keys=word,
                           subagent_depth=0)
    free = run_evolution(spec, "s1_free", out_dir, generations=15,
                         population=8, seed=seed, subagent_depth=0)
    return {"stage": "S1", "spec_id": spec.spec_id, "min_steps": minimum,
            "strict_word": list(word or ()), "strict": strict, "free": free}


def stage_s2(out_dir: str, seed: int = 7) -> Dict[str, Any]:
    items = deterministic_items()
    spec = load_spec(spec_path("spec_dynamic_group"))
    minimum, word = bfs_min_steps(spec)
    strict = run_evolution(spec, "s2_strict", out_dir, generations=10,
                           population=10, seed=seed, strict_keys=word,
                           subagent_depth=1)
    return {"stage": "S2", "spec_id": spec.spec_id, "min_steps": minimum,
            "strict_word": list(word or ()), "deterministic_items": items,
            "strict": strict}


def stage_s3(out_dir: str, seed: int = 7) -> Dict[str, Any]:
    spec = load_spec(spec_path("spec_dynamic_group"))
    path = solution_path(spec)
    minimum, word = path["min_steps"], path["word"]
    gate = shortest_path_gate(spec, path["correct"])
    strict = run_evolution(spec, "s3_strict", out_dir, generations=8,
                           population=10, seed=seed, strict_keys=word,
                           subagent_depth=0)
    strict_best_free = evaluate_genome(spec, strict["best_genome"], gate=gate,
                                       episodes=20, seed=seed)
    strict_best_free_nodiscard = evaluate_genome(spec, strict["best_genome"],
                                                 episodes=20, seed=seed)
    free = run_evolution(spec, "s3_free", out_dir, generations=30,
                         population=10, seed=seed, gate=gate,
                         subagent_depth=1)
    free_best_greedy = evaluate_genome(spec, free["best_genome"], gate=gate,
                                       episodes=20, seed=seed)
    return {
        "stage": "S3", "spec_id": spec.spec_id, "min_steps": minimum,
        "strict_word": list(word or ()),
        "single_correct_step_per_state": True,
        "canonical_states": len(path["correct"]),
        "overfitting": {
            "strict_best_fitness": strict["best_fitness"],
            "strict_best_free_discard_eval": strict_best_free,
            "strict_best_free_nodiscard_eval": strict_best_free_nodiscard,
            "free_recovery_best_fitness": free["best_fitness"],
            "free_best_final_eval": free_best_greedy,
        },
        "strict": strict, "free": free,
    }


@dataclass
class EarlyStop:
    """Checkpoint-based stop: a solve was seen and no improvement for a while.

    Before the FIRST solved episode the run NEVER stops early (it is the long
    stage-4 run); after a solve is observed, ``plateau_checkpoints`` consecutive
    checkpoints with no improvement stop it.
    """

    plateau_checkpoints: int = 5
    check_every: int = 25
    _best: float = float("-inf")
    _since: int = 0
    _seen_solve: bool = False
    snapshots: List[Dict[str, Any]] = field(default_factory=list)

    def __call__(self, generation: int, record: Dict[str, Any]) -> bool:
        if generation % self.check_every != 0:
            return False
        fitness = float(record["best_fitness"])
        if record.get("solved", 0) > 0:
            self._seen_solve = True
        if fitness > self._best + 1e-9:
            self._best = fitness
            self._since = 0
        elif self._seen_solve:
            self._since += 1
        self.snapshots.append({
            "generation": generation,
            "best_fitness": round(fitness, 6),
            "best_steps": record.get("best_steps"),
            "solved": record.get("solved"),
            "seen_solve": self._seen_solve,
            "plateau_checkpoints": self._since,
        })
        return self._seen_solve and self._since >= self.plateau_checkpoints


def stage_s4(spec_id: str, out_dir: str, generations: int = 1500,
             population: int = 12, seed: int = 7, step_cost: float = 0.10,
             subagent_depth: int = 2, stop_after_plateau: bool = True,
             plateau_checkpoints: int = 5) -> Dict[str, Any]:
    spec = load_spec(spec_path(spec_id))
    plan = minimal_and_max(spec)
    ckpt = os.path.join(out_dir, "checkpoints")
    hist = os.path.join(out_dir, "history")
    config = _config(generations, population, seed, step_cost, subagent_depth,
                     ckpt, hist, plan["max_steps"])
    trainer = CurriculumTrainer(spec, config, spec_id=spec_id)
    stop = EarlyStop(plateau_checkpoints=plateau_checkpoints, check_every=25)
    start = time.perf_counter()

    def on_generation(generation: int, record: Dict[str, Any]) -> bool:
        shall_stop = stop(generation, record)
        if generation % stop.check_every == 0 and stop.snapshots:
            snapshot = dict(stop.snapshots[-1])
            snapshot["elapsed_s"] = round(time.perf_counter() - start, 3)
            snapshot["spec_id"] = spec_id
            write_json(os.path.join(out_dir, "snapshot_gen_%05d.json" % generation),
                       snapshot)
        return shall_stop

    history = trainer.run(on_generation=on_generation if stop_after_plateau else None)
    elapsed = time.perf_counter() - start
    best = best_agent(trainer)
    result = {
        "stage": "S4", "spec_id": spec_id, "plan": plan,
        "generations_requested": generations,
        "generations_run": len(history),
        "seed": seed, "step_cost": step_cost,
        "max_episode_steps": plan["max_steps"],
        "elapsed_s": round(elapsed, 6),
        "best_fitness": round(best.fitness, 6),
        "best_steps": best.steps_gen,
        "best_genome": best.genome,
        "snapshots": stop.snapshots,
        "best_fitness_series": [r["best_fitness"] for r in history],
    }
    write_json(os.path.join(out_dir, "stage4_%s_summary.json" % spec_id), result)
    return result


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Unit D-1 curriculum runner")
    parser.add_argument("--stage", default="all",
                        choices=["s1", "s2", "s3", "s4", "all"])
    parser.add_argument("--spec", default="spec_multi_step")
    parser.add_argument("--out", default="out/curriculum")
    parser.add_argument("--generations", type=int, default=1500)
    parser.add_argument("--population", type=int, default=12)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--step-cost", type=float, default=0.10)
    parser.add_argument("--no-early-stop", action="store_true")
    args = parser.parse_args(argv)

    os.makedirs(args.out, exist_ok=True)
    stages = ["s1", "s2", "s3"] if args.stage == "all" else [args.stage]
    results: Dict[str, Any] = {}
    for stage in stages:
        started = time.perf_counter()
        if stage == "s1":
            payload = stage_s1(args.out, seed=args.seed)
        elif stage == "s2":
            payload = stage_s2(args.out, seed=args.seed)
        elif stage == "s3":
            payload = stage_s3(args.out, seed=args.seed)
        else:
            payload = stage_s4(args.spec, args.out, generations=args.generations,
                               population=args.population, seed=args.seed,
                               step_cost=args.step_cost,
                               stop_after_plateau=not args.no_early_stop)
        payload["stage_wall_s"] = round(time.perf_counter() - started, 6)
        results[stage] = payload
        write_json(os.path.join(args.out, "%s_summary.json" % stage), payload)
        print("[curriculum] %s done in %.3fs" % (stage, payload["stage_wall_s"]))
    write_json(os.path.join(args.out, "curriculum_summary.json"), results)
    print(json.dumps({"stages": list(results),
                      "out": os.path.abspath(args.out)}, sort_keys=True))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())

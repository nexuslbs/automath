"""Tests for the Unit C approach-comparison arms (RL, mix, shared harness).

Run::

    python -m evolution_trainer.arms_tests

The checks are standard-library only and deterministic for a fixed seed.
"""

from __future__ import annotations

import math
import os
import random
import sys
import tempfile

from dynamic_env.spec import load_spec

from .evolution import EvoConfig
from .features import ActionFeaturizer, state_dim
from .harness import Harness
from .mix_train import MixTrainer
from .rl_train import ReinforceTrainer

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_SPEC_DIR = os.path.join(_REPO_ROOT, "data", "dynamic_env")
RESULTS = []


def check(name: str, ok: bool, detail: str) -> None:
    RESULTS.append((name, bool(ok), detail))
    print("[%s] %s: %s" % ("PASS" if ok else "FAIL", name, detail))


def spec_path(stem: str) -> str:
    return os.path.join(_SPEC_DIR, stem + ".json")


def check_harness_matches_unit_b() -> None:
    """The shared harness reproduces Unit B's shaped reward exactly.

    Unit B's optima are known from ``evolution_trainer.tests``: spec_minimal
    = goal(1.00) + intermediate(0.30) - step(0.05) = 1.25; spec_multi_step =
    goal(1.00) + intermediate(0.30) + guard(0.20) - 2*step(0.05) = 1.40.
    Replaying the known minimal words through the harness must give those
    values (the harness owns the reward, so this pins it to Unit B).
    """
    zero_min = [0.0] * Harness(load_spec(spec_path("spec_minimal"))).net.size
    res_min = Harness(load_spec(spec_path("spec_minimal"))).rollout(
        zero_min, explore=False, scripted_keys=("set:o1",))
    zero_multi = [0.0] * Harness(load_spec(spec_path("spec_multi_step"))).net.size
    res_multi = Harness(load_spec(spec_path("spec_multi_step"))).rollout(
        zero_multi, explore=False,
        scripted_keys=("build:build_sub:n9+n5", "set:o"))
    ok = (abs(res_min.total_return - 1.25) < 1e-9 and res_min.solved
          and res_min.steps == 1
          and abs(res_multi.total_return - 1.40) < 1e-9 and res_multi.solved
          and res_multi.steps == 2)
    check("harness_matches_unit_b_reward", ok,
          "minimal word -> 1.2500 (steps=1, solved=%s); multi_step word -> 1.4000 "
          "(steps=2, solved=%s)" % (res_min.solved, res_multi.solved))


def check_same_architecture() -> None:
    """RL and evolution use the same network size and input dimension."""
    for stem in ("spec_minimal", "spec_dynamic_group", "spec_dynamic_axiom",
                 "spec_multi_step"):
        spec = load_spec(spec_path(stem))
        harness = Harness(spec, hidden=12)
        featurizer = ActionFeaturizer(spec)
        expected_dim = state_dim(spec) + featurizer.dim
        expected_size = 12 * expected_dim + 12 + 12 + 1
        check("same_architecture_%s" % stem,
              harness.in_dim == expected_dim and harness.net.size == expected_size,
              "in_dim=%d genome=%d (PolicyNet(hidden=12))"
              % (harness.in_dim, harness.net.size))


def check_harness_deterministic() -> None:
    """A zero genome gives a deterministic rollout for a fixed rng seed."""
    spec = load_spec(spec_path("spec_multi_step"))
    a = Harness(spec)
    b = Harness(spec)
    zero = [0.0] * a.net.size
    r1 = a.rollout(zero, explore=True, rng=random.Random(11))
    r2 = b.rollout(zero, explore=True, rng=random.Random(11))
    check("harness_deterministic",
          r1.steps == r2.steps and abs(r1.total_return - r2.total_return) < 1e-12,
          "same genome+seed -> steps=%d return=%.4f" % (r1.steps, r1.total_return))


def check_rl_learns() -> None:
    """REINFORCE reaches the greedy optimum on spec_minimal (1-step)."""
    spec = load_spec(spec_path("spec_minimal"))
    cfg = EvoConfig(generations=6, population_size=6, seed=7)
    trainer = ReinforceTrainer(spec, cfg, seed=7)
    traj = trainer.run(6)
    greedy = trainer.harness.rollout(trainer.best_genome, explore=False)
    check("rl_learns_minimal",
          greedy.solved and greedy.steps == 1,
          "generations=%d best_fitness=%.4f greedy_solved=%s steps=%d"
          % (len(traj), trainer.best_fitness, greedy.solved, greedy.steps))


def check_rl_gradient_matches_numeric() -> None:
    """The hand-written log-prob gradient matches a finite difference."""
    spec = load_spec(spec_path("spec_minimal"))
    cfg = EvoConfig(seed=5)
    trainer = ReinforceTrainer(spec, cfg, seed=5)
    rng = random.Random(2)
    inputs = [tuple(rng.uniform(-1.0, 1.0) for _ in range(trainer.in_dim))
              for _ in range(3)]
    chosen = 1

    def logp(genome):
        logits = []
        for x in inputs:
            total = genome[trainer.net.b2_offset]
            acts = []
            for j in range(trainer.net.hidden):
                t = genome[trainer.net.b1_offset + j]
                row = j * trainer.in_dim
                for i in range(trainer.in_dim):
                    t += genome[row + i] * x[i]
                acts.append(math.tanh(t))
            for j in range(trainer.net.hidden):
                total += genome[trainer.net.w2_offset + j] * acts[j]
            logits.append(total)
        peak = max(logits)
        exps = [math.exp(z - peak) for z in logits]
        return logits[chosen] - peak - math.log(sum(exps))

    grad = trainer.grad_log_prob(inputs, chosen)
    eps = 1e-6
    worst = 0.0
    for i in range(0, trainer.net.size, 37):
        base = list(trainer.genome)
        base[i] += eps
        plus = logp(base)
        base[i] -= 2 * eps
        minus = logp(base)
        numeric = (plus - minus) / (2 * eps)
        worst = max(worst, abs(numeric - grad[i]))
    check("rl_gradient_finite_difference", worst < 1e-4,
          "max |analytic - numeric| over sampled genes = %.3e" % worst)


def check_mix_warm_start_and_evolution() -> None:
    """The mix arm warms up then seeds evolution; deterministic for a seed."""
    spec = load_spec(spec_path("spec_minimal"))
    cfg = EvoConfig(generations=8, population_size=6, seed=9)
    t1 = MixTrainer(spec, cfg, warm_generations=3, evo_generations=5, seed=9)
    traj1 = list(t1.run())
    t2 = MixTrainer(spec, cfg, warm_generations=3, evo_generations=5, seed=9)
    traj2 = list(t2.run())
    same = [r["best_fitness"] for r in traj1] == [r["best_fitness"] for r in traj2]
    check("mix_deterministic_and_learns",
          same and len(traj1) == 8 and t1.best_source in ("rl_warm", "evolution"),
          "8 generations (3 warm + 5 evo), deterministic=%s best_source=%s best=%.4f"
          % (same, t1.best_source, t1.best_fitness))


def check_rl_run_one_artifacts() -> None:
    """A standalone RL run writes trajectory.csv + best_genome.json + summary."""
    with tempfile.TemporaryDirectory() as tmp:
        from .rl_train import run_one
        summary = run_one(spec_path("spec_minimal"), tmp, generations=4,
                          population=6, seed=7, eval_episodes=5)
        ok = (os.path.exists(os.path.join(tmp, "trajectory.csv"))
              and os.path.exists(os.path.join(tmp, "best_genome.json"))
              and os.path.exists(os.path.join(tmp, "summary.json")))
        check("rl_run_one_artifacts", ok and summary["eval_episodes"] == 5,
              "trajectory.csv + best_genome.json + summary.json; eval_solved=%d/5"
              % summary["eval_solved"])


def main() -> int:
    check_harness_matches_unit_b()
    check_same_architecture()
    check_harness_deterministic()
    check_rl_gradient_matches_numeric()
    check_rl_learns()
    check_mix_warm_start_and_evolution()
    check_rl_run_one_artifacts()
    passed = sum(1 for _, ok, _ in RESULTS if ok)
    total = len(RESULTS)
    print("RESULT: %d/%d passed (Unit C arms: shared harness, pure RL, mix, "
          "fixed-weight evolution)" % (passed, total))
    return 0 if passed == total else 1


if __name__ == "__main__":
    sys.exit(main())

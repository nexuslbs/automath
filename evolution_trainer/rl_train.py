"""PURE RL arm - policy gradient (REINFORCE with a baseline).

This is arm 1 of the Unit C approach comparison. It trains a SINGLE
target-conditioned policy network by policy gradient on the SAME architecture
(:class:`evolution_trainer.genome.PolicyNet`), the SAME features
(:mod:`evolution_trainer.features`) and the SAME shaped reward used by the
evolution genomes (see :mod:`evolution_trainer.harness`).

Algorithm
---------
REINFORCE with reward-to-go and a running-mean baseline:

* for each generation we collect ``population_size`` episodes under the current
  softmax policy (so the episode budget matches ``generations * population`` of
  the evolution arms);
* every step contributes ``(G_t - b) * grad_theta log pi(a_t | s_t)`` where
  ``G_t`` is the reward-to-go and ``b`` a running mean of episode returns;
* the accumulated gradient is averaged over the generation and applied with a
  plain SGD step (no dependency, stdlib only).

The trainer keeps a best-so-far genome chosen by a GREEDY harness evaluation,
so the final candidate is selected with the same criterion for every arm.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import random
import sys
import time
from typing import Any, Dict, List, Optional, Sequence

from dynamic_env.spec import load_spec

from .evolution import EvoConfig
from .harness import Harness
from .genome import round_genome

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.dirname(_HERE)
_DEFAULT_SPEC_DIR = os.path.join(_REPO_ROOT, "data", "dynamic_env")


class ReinforceTrainer:
    """One target-conditioned genome trained by REINFORCE with a baseline."""

    def __init__(self, spec, config: Optional[EvoConfig] = None,
                 learning_rate: float = 0.10,
                 baseline_rate: float = 0.10,
                 advantage_clip: float = 5.0,
                 seed: int = 7, spec_id: str = "") -> None:
        self.spec = spec
        self.spec_id = spec_id or spec.spec_id
        self.config = config or EvoConfig()
        self.harness = Harness(
            spec, hidden=self.config.hidden,
            step_cost=self.config.step_cost,
            intermediate_reward=self.config.intermediate_reward,
            guard_bonus=self.config.guard_bonus,
            subgoal_reward=self.config.subgoal_reward,
            goal_reward=self.config.goal_reward,
            max_episode_steps=self.config.max_episode_steps,
        )
        self.net = self.harness.net
        self.in_dim = self.harness.in_dim
        self.learning_rate = learning_rate
        self.baseline_rate = baseline_rate
        self.advantage_clip = advantage_clip
        self.rng = random.Random(seed)
        self.genome: List[float] = self.net.init(self.rng, scale=0.5)
        self.best_genome: List[float] = list(self.genome)
        self.best_fitness = float("-inf")
        self.trajectory: List[Dict[str, Any]] = []

    # -- backward pass -----------------------------------------------------
    def _activate(self, x: Sequence[float]):
        n = self.in_dim
        hidden = self.net.hidden
        b1 = self.net.b1_offset
        w2 = self.net.w2_offset
        b2 = self.net.b2_offset
        g = self.genome
        acts = [0.0] * hidden
        for j in range(hidden):
            total = g[b1 + j]
            row = j * n
            for i in range(n):
                total += g[row + i] * x[i]
            acts[j] = math.tanh(total)
        out = g[b2]
        for j in range(hidden):
            out += g[w2 + j] * acts[j]
        return out, acts

    def _accumulate_dz(self, grad: List[float], x: Sequence[float],
                       coeff: float) -> None:
        """Accumulate ``coeff * d out / d params`` for one action input."""
        n = self.in_dim
        hidden = self.net.hidden
        b1 = self.net.b1_offset
        w2 = self.net.w2_offset
        b2 = self.net.b2_offset
        g = self.genome
        _, acts = self._activate(x)
        grad[b2] += coeff
        for j in range(hidden):
            hp = 1.0 - acts[j] * acts[j]
            d = coeff * g[w2 + j] * hp
            grad[w2 + j] += coeff * acts[j]
            grad[b1 + j] += d
            row = j * n
            for i in range(n):
                grad[row + i] += d * x[i]

    def grad_log_prob(self, inputs: Sequence[Sequence[float]],
                      chosen: int) -> List[float]:
        logits = [self._activate(x)[0] for x in inputs]
        peak = max(logits)
        exps = [math.exp(z - peak) for z in logits]
        total = sum(exps)
        probs = [e / total for e in exps]
        grad = [0.0] * self.net.size
        for k, x in enumerate(inputs):
            coeff = (1.0 if k == chosen else 0.0) - probs[k]
            if abs(coeff) > 1e-12:
                self._accumulate_dz(grad, x, coeff)
        return grad

    # -- driver ------------------------------------------------------------
    def run(self, generations: int) -> List[Dict[str, Any]]:
        size = self.net.size
        baseline = 0.0
        for generation in range(1, generations + 1):
            accum = [0.0] * size
            step_total = 0
            gen_returns: List[float] = []
            gen_solved = 0
            for _ in range(self.config.population_size):
                res = self.harness.rollout(self.genome, explore=True,
                                           rng=self.rng, collect=True)
                gen_returns.append(res.total_return)
                if res.solved:
                    gen_solved += 1
                step_total += max(1, len(res.trace))
                reward_to_go = 0.0
                for item in reversed(res.trace):
                    reward_to_go += item["reward"]
                    advantage = reward_to_go - baseline
                    if advantage > self.advantage_clip:
                        advantage = self.advantage_clip
                    elif advantage < -self.advantage_clip:
                        advantage = -self.advantage_clip
                    grad = self.grad_log_prob(item["inputs"], item["chosen"])
                    for i in range(size):
                        accum[i] += advantage * grad[i]
                baseline += self.baseline_rate * (res.total_return - baseline)
            scale = self.learning_rate / float(max(1, step_total))
            for i in range(size):
                self.genome[i] += scale * accum[i]
            greedy = self.harness.rollout(self.genome, explore=False)
            if greedy.total_return > self.best_fitness:
                self.best_fitness = greedy.total_return
                self.best_genome = list(self.genome)
            row = {
                "generation": generation,
                "best_fitness": round(self.best_fitness, 6),
                "mean_fitness": round(sum(gen_returns) / max(1, len(gen_returns)), 6),
                "solved": gen_solved,
                "best_steps": greedy.steps,
            }
            self.trajectory.append(row)
        return self.trajectory


def run_one(spec_path: str, out_dir: str, generations: int = 60,
            population: int = 12, seed: int = 7, hidden: int = 12,
            learning_rate: float = 0.10, eval_episodes: int = 30) -> Dict[str, Any]:
    spec = load_spec(spec_path)
    config = EvoConfig(generations=generations, population_size=population,
                       hidden=hidden, seed=seed)
    trainer = ReinforceTrainer(spec, config, learning_rate=learning_rate,
                               seed=seed, spec_id=spec.spec_id)
    started = time.perf_counter()
    trainer.run(generations)
    wall = time.perf_counter() - started
    evaluation = trainer.harness.evaluate(trainer.best_genome,
                                          episodes=eval_episodes, seed=seed + 999)
    os.makedirs(out_dir, exist_ok=True)
    traj_path = os.path.join(out_dir, "trajectory.csv")
    with open(traj_path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=[
            "generation", "best_fitness", "mean_fitness", "solved", "best_steps"])
        writer.writeheader()
        writer.writerows(trainer.trajectory)
    with open(os.path.join(out_dir, "best_genome.json"), "w", encoding="utf-8") as handle:
        json.dump({"spec_id": spec.spec_id, "arm": "rl",
                   "best_fitness": round(trainer.best_fitness, 6),
                   "genome": round_genome(trainer.best_genome)}, handle, indent=2)
    summary = {
        "arm": "rl",
        "spec_id": spec.spec_id,
        "generations": generations,
        "population": population,
        "seed": seed,
        "episodes": generations * population,
        "genome_size": trainer.net.size,
        "in_dim": trainer.in_dim,
        "best_fitness": round(trainer.best_fitness, 6),
        "final_mean_fitness": trainer.trajectory[-1]["mean_fitness"],
        **evaluation,
        "wall_seconds": round(wall, 4),
        "trajectory": traj_path,
    }
    with open(os.path.join(out_dir, "summary.json"), "w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, sort_keys=True)
    print(json.dumps(summary, indent=2, sort_keys=True))
    return summary


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--spec", required=True)
    parser.add_argument("--spec-dir", default=_DEFAULT_SPEC_DIR)
    parser.add_argument("--out", required=True)
    parser.add_argument("--generations", type=int, default=60)
    parser.add_argument("--population", type=int, default=12)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--hidden", type=int, default=12)
    parser.add_argument("--learning-rate", type=float, default=0.10)
    parser.add_argument("--eval-episodes", type=int, default=30)
    args = parser.parse_args(argv)
    spec_path = args.spec
    if not os.path.exists(spec_path):
        spec_path = os.path.join(args.spec_dir, args.spec + ".json")
    run_one(spec_path, args.out, generations=args.generations,
            population=args.population, seed=args.seed, hidden=args.hidden,
            learning_rate=args.learning_rate, eval_episodes=args.eval_episodes)
    return 0


if __name__ == "__main__":
    sys.exit(main())

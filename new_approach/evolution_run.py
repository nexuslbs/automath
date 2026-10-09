"""Pipeline runner for the evolution branch: baselines and Stage 2.

Commands (pure standard library, seeded, bounded)::

    /opt/automath/venv/bin/python -m new_approach.evolution_run --stage baseline
    /opt/automath/venv/bin/python -m new_approach.evolution_run --stage 2
    /opt/automath/venv/bin/python -m new_approach.evolution_run --stage 2e

`baseline` reproduces the U2 supervised planner (33/33) and the pure-reward
control (14/33) on the SAME 33 new-initial-state validation cases at both the
operator budget (60 episodes) and the longer budget, so the comparison in the
evidence is measured in one process, not quoted.  `2` runs the core comparison;
`2e` runs the extended complex-scenario suite.
"""

from __future__ import annotations

import argparse
import statistics
import sys
import time
from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple

from .agent import TabularAgent
from .env import FixedStateGoal
from .evolution import EvoEnv, scenario_cases
from .evolution_agents import (
    CORE_ARITY,
    CORE_ORDER,
    EVO_ARITY,
    EVO_ORDER,
    EvalResult,
    Guidance,
    GuidedQAgent,
    TrainStats,
    evaluate,
    read_reward,
    reward_node,
    train_guided,
)
from .u2 import (
    curriculum as u2_curriculum,
    training_cases as u2_training_cases,
    validation_cases as u2_validation_cases,
)

SEED = 20261009


# --------------------------------------------------------------------------
# Cases
# --------------------------------------------------------------------------

@dataclass
class SimpleCase:
    name: str
    key: str
    env: object
    max_steps: int
    expected_word: Tuple[str, ...]
    level: str = ""


def core_train_cases() -> List[SimpleCase]:
    return [SimpleCase(c.name, c.key, c.env, c.max_steps,
                       tuple(c.expected_word), c.level)
            for c in u2_training_cases()]


def core_validation_cases() -> List[SimpleCase]:
    return [SimpleCase(c.name, c.key, c.env, c.max_steps,
                       tuple(c.expected_word), c.level)
            for c in u2_validation_cases(list(u2_curriculum()))]


def evo_train_cases() -> List[SimpleCase]:
    out = []
    for name, target, stack, level, expected in scenario_cases(False):
        env = EvoEnv(FixedStateGoal(target), initial_stack=stack,
                     max_actions=len(expected) + 2)
        out.append(SimpleCase(name, "S:" + target.canonical(), env,
                              len(expected) + 2, expected, level))
    return out


def evo_validation_cases() -> List[SimpleCase]:
    out = []
    for name, target, stack, level, expected in scenario_cases(True):
        env = EvoEnv(FixedStateGoal(target), initial_stack=stack,
                     max_actions=len(expected) + 2)
        out.append(SimpleCase(name, "S:" + target.canonical(), env,
                              len(expected) + 2, expected, level))
    return out


# --------------------------------------------------------------------------
# Reporting helpers
# --------------------------------------------------------------------------

def hdr(title: str) -> None:
    print("=" * 72)
    print(title)
    print("=" * 72)


def print_curve(label: str, stats: TrainStats) -> None:
    print("%s convergence curve (per training episode):" % label)
    for row in stats.curve(12):
        print("  " + row)


def eval_line(label: str, res: EvalResult) -> None:
    print("%s: %d/%d PASS (learned_states=%d)"
          % (label, res.passed, res.total, res.learned_states))
    if res.walls:
        print("  eval wall: min=%.6fs median=%.6fs max=%.6fs"
              % (min(res.walls), statistics.median(res.walls), max(res.walls)))


def register(guidance: Guidance, cases: Sequence[SimpleCase]) -> None:
    for c in cases:
        guidance.register(c.key, c.env, c.expected_word)


# --------------------------------------------------------------------------
# Baselines (measured in this process)
# --------------------------------------------------------------------------

def run_baseline(episodes: int, seed: int = SEED) -> int:
    train = core_train_cases()
    val = core_validation_cases()
    hdr("BASELINE (U2, same 33 new-initial-state cases, measured here)")
    print("training cases=%d validation cases=%d seed=%d"
          % (len(train), len(val), seed))
    tuples = [(c.key, c.env, c.expected_word, c.max_steps) for c in train]
    for budget in (60, episodes):
        print("--- budget episodes=%d ---" % budget)
        sup = TabularAgent(episodes=budget, seed=seed, use_demo=True)
        t0 = time.perf_counter()
        sup.train(tuples)
        sup_wall = time.perf_counter() - t0
        sup_pass = sum(int(sup.act_episode(c.key, c.env, c.max_steps,
                                           c.env.goal.describe()).passed)
                       for c in val)
        ctrl = TabularAgent(episodes=budget, seed=seed, use_demo=False)
        t0 = time.perf_counter()
        ctrl.train(tuples)
        ctrl_wall = time.perf_counter() - t0
        ctrl_pass = sum(int(ctrl.act_episode(c.key, c.env, c.max_steps,
                                             c.env.goal.describe()).passed)
                        for c in val)
        print("supervised planner (use_demo=True): train=10/10 validation=%d/%d "
              "digest=%s wall=%.6fs"
              % (sup_pass, len(val), sup.digest(), sup_wall))
        print("pure-reward control (use_demo=False): validation=%d/%d "
              "learned_states=%d wall=%.6fs"
              % (ctrl_pass, len(val), ctrl.learned_states(), ctrl_wall))
        print("BASELINE_SUPERVISED_%d=%d/%d" % (budget, sup_pass, len(val)))
        print("BASELINE_CONTROL_%d=%d/%d" % (budget, ctrl_pass, len(val)))
    return 0


# --------------------------------------------------------------------------
# Stage 2 (core) and Stage 2e (extended)
# --------------------------------------------------------------------------

def run_stage2(episodes: int, args) -> int:
    train = core_train_cases()
    val = core_validation_cases()
    hdr("STAGE 2 (core U2): artificial reward/guidance scaffolding, "
        "use_demo=False")
    print("reward node GOOD = %s" % reward_node(True).canonical())
    print("reward node BAD  = %s" % reward_node(False).canonical())
    print("read_reward(GOOD)=%+.2f read_reward(BAD)=%+.2f"
          % (read_reward(reward_node(True)), read_reward(reward_node(False))))
    print("episodes=%d seed=%d alpha=%.2f gamma=%.2f eps=%.2f->%.2f"
          % (episodes, args.seed, args.alpha, args.gamma,
             args.epsilon_start, args.epsilon_end))
    agent = GuidedQAgent(action_order=CORE_ORDER, arity=CORE_ARITY,
                         episodes=episodes, seed=args.seed, alpha=args.alpha,
                         gamma=args.gamma, epsilon_start=args.epsilon_start,
                         epsilon_end=args.epsilon_end)
    gd = Guidance(evo=False)
    register(gd, train)
    t0 = time.perf_counter()
    stats = train_guided(agent, train, episodes, gd)
    wall = time.perf_counter() - t0
    print_curve("Stage 2", stats)
    tr = evaluate(agent, train)
    va = evaluate(agent, val)
    eval_line("Stage 2 final training", tr)
    eval_line("Stage 2 validation (new initial states)", va)
    print("guidance hits=%d misses=%d train_wall=%.6fs digest=%s"
          % (gd.hits, gd.misses, wall, agent.digest()))
    print("STAGE2_TRAIN=%d/%d STAGE2_VAL=%d/%d"
          % (tr.passed, tr.total, va.passed, va.total))
    print("STAGE2_DIGEST=%s" % agent.digest())
    return 0


def run_stage2e(episodes: int, args) -> int:
    train = evo_train_cases()
    val = evo_validation_cases()
    hdr("STAGE 2 (extended %d complex scenarios)" % len(train))
    print("training scenarios=%d validation new-initial-states=%d "
          "actions=%d episodes=%d seed=%d"
          % (len(train), len(val), len(EVO_ORDER), episodes, args.seed))
    agent = GuidedQAgent(action_order=EVO_ORDER, arity=EVO_ARITY,
                         episodes=episodes, seed=args.seed, alpha=args.alpha,
                         gamma=args.gamma, epsilon_start=args.epsilon_start,
                         epsilon_end=args.epsilon_end)
    gd = Guidance(evo=True)
    register(gd, train)
    t0 = time.perf_counter()
    stats = train_guided(agent, train, episodes, gd)
    wall = time.perf_counter() - t0
    print("guidance hits=%d misses=%d" % (gd.hits, gd.misses))
    print_curve("Stage 2 extended", stats)
    tr = evaluate(agent, train)
    va = evaluate(agent, val)
    eval_line("Stage 2 extended training", tr)
    eval_line("Stage 2 extended validation", va)
    print("train_wall=%.6fs digest=%s" % (wall, agent.digest()))
    print("STAGE2E_TRAIN=%d/%d STAGE2E_VAL=%d/%d"
          % (tr.passed, tr.total, va.passed, va.total))
    return 0


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------

def main(argv: Optional[Sequence[str]] = None) -> int:
    p = argparse.ArgumentParser(description="evolution pipeline runner")
    p.add_argument("--stage", choices=("baseline", "2", "2e", "all"),
                   default="all")
    p.add_argument("--episodes", type=int, default=200)
    p.add_argument("--seed", type=int, default=SEED)
    p.add_argument("--alpha", type=float, default=0.5)
    p.add_argument("--gamma", type=float, default=0.95)
    p.add_argument("--epsilon-start", dest="epsilon_start", type=float, default=0.5)
    p.add_argument("--epsilon-end", dest="epsilon_end", type=float, default=0.02)
    args = p.parse_args(argv)

    rc = 0
    if args.stage in ("baseline", "all"):
        rc |= run_baseline(args.episodes, args.seed)
    if args.stage in ("2", "all"):
        rc |= run_stage2(args.episodes, args)
    if args.stage in ("2e", "all"):
        rc |= run_stage2e(args.episodes, args)
    return rc


if __name__ == "__main__":
    sys.exit(main())

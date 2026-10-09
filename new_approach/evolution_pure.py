"""Unit C (Phase B): PURE-evolutionary path - NO artificial scaffolding.

This module is the clean "after" of the Phase A / Phase B before-after pair.
It deliberately lives in its OWN module (and a matching
``config/evolution_pure.json``) so the code isolation from the Phase A
semi-evolution is unambiguous.

WHAT IS REMOVED vs Phase A (``evolution_semi.py`` + ``evolution_rewards.py``):

* NO potential-based reward shaping (PBRS) - ``phi_scale = 0.0`` and
  ``gamma * Phi(s') - Phi(s)`` is identically ``0.0`` at every step;
* NO sub-goal bonus - ``subgoal_bonus = 0.0`` (and ``min_subgoal_size`` is set
  above any reachable node count so the sub-goal predicate cannot fire);
* NO positive-shaping cap term - ``shaping_cap = 0.0`` (nothing positive to cap);
* therefore NO reward/guidance node, no progress signal, no similarity term
  ever contributes to the numeric reward.

WHAT REMAINS - and why it is OBJECTIVE, NOT SCAFFOLDING:

1. final-goal reward ``+1.0`` when a feasible state is solved - this IS the
   true objective (the task is to solve the bundle);
2. per-step cost ``-0.05`` on every action - this IS part of the objective
   (fewer actions = better); it is a genuine cost of acting, not a hint about
   which action to take.  It does NOT mention the target, the goal distance or
   any intermediate structure.

The per-step reward is therefore exactly::

    r = -0.05 + (1.0 if reached_goal else 0.0)

Selection/reproduction is PURELY over time by fitness: the whole population is
evaluated every generation and the fittest genomes (bundle solved count primary,
pure return as tiebreak) reproduce via the SAME BLX-alpha crossover + gaussian
mutation as Phase A (``evolution_mix`` / ``evolution_population``); population
starts from RANDOM basic-instinct genomes.  ``evolve_one_generation_pure`` is a
thin wrapper over the existing ``evolution_semi.evolve_one_generation_shaped``
machinery that injects the scaffolding-free reward config, so the crossover,
mutation, elitism and threshold pool are byte-for-byte the Phase A code - only
the reward signal differs.

Pure standard library, seeded and deterministic.
"""

from __future__ import annotations

import random
from typing import List, Optional, Sequence, Tuple

from .evolution_agents import EVO_ARITY, EVO_ORDER
from .evolution_population import EvoConfig, Genome
from .evolution_rewards import RewardConfig
from .evolution_semi import (
    bundle_validation as _bundle_validation_shaped,
    evolve_one_generation_shaped as _evolve_one_generation_shaped,
    make_shaped_agent as _make_shaped_agent,
    train_shaped as _train_shaped,
)

__all__ = [
    "PURE_PHI_SCALE",
    "PURE_SUBGOAL_BONUS",
    "PURE_SHAPING_CAP",
    "pure_reward_config",
    "assert_objective_only",
    "evolve_one_generation_pure",
    "core_validation_pure",
    "evo_validation_pure",
    "bundle_validation_pure",
    "train_pure",
    "make_pure_agent",
    "pure_step_reward",
]

# The binding constants of the pure objective.  These are the ONLY numeric
# reward terms; everything shaping-related is forced to zero.
PURE_PHI_SCALE = 0.0
PURE_SUBGOAL_BONUS = 0.0
PURE_SHAPING_CAP = 0.0
PURE_GOAL_REWARD = 1.0
PURE_MIN_SUBGOAL_SIZE = 1 << 30  # unreachable -> sub-goal predicate is dead
PURE_STEP_COST = -0.05


def pure_step_reward(step_cost: float, goal_reward: float,
                     reached: bool) -> float:
    """The complete pure reward: objective step cost + true goal reward."""
    return step_cost + (goal_reward if reached else 0.0)


def pure_reward_config(cfg: EvoConfig) -> RewardConfig:
    """A ``RewardConfig`` with EVERY scaffolding term forced off.

    ``gamma`` is kept from ``cfg`` because it is the AGENT's RL discount factor,
    not a shaping scale; the shaping term is zeroed by ``phi_scale = 0.0``.
    """
    rcfg = RewardConfig(
        step_cost=cfg.step_cost,
        gamma=cfg.reward_gamma,
        phi_scale=PURE_PHI_SCALE,
        subgoal_bonus=PURE_SUBGOAL_BONUS,
        goal_reward=PURE_GOAL_REWARD,
        shaping_cap=PURE_SHAPING_CAP,
        min_subgoal_size=PURE_MIN_SUBGOAL_SIZE,
    )
    assert_objective_only(rcfg)
    return rcfg


def assert_objective_only(rcfg: RewardConfig) -> None:
    """Fail loudly if any scaffolding term is non-zero (policy, not comment)."""
    assert rcfg.phi_scale == 0.0, "Phase B forbids PBRS progress shaping"
    assert rcfg.subgoal_bonus == 0.0, "Phase B forbids the sub-goal bonus"
    assert rcfg.min_subgoal_size > 1, "Phase B disables the sub-goal predicate"
    assert rcfg.shaping_cap == 0.0, "Phase B has no positive shaping to cap"


def make_pure_agent(genome: Genome, cfg: EvoConfig, seed: int):
    """Same basic-instinct agent as Phase A (no reward lives in the agent)."""
    return _make_shaped_agent(genome, cfg, seed)


def train_pure(agent, cases: Sequence, episodes: int) -> None:
    """Train on the pure objective only (step cost + goal).

    The reward needs no ``EvoConfig``: the step cost is the binding objective
    constant and the goal reward is the fixed true objective ``+1.0``.
    """
    rcfg = RewardConfig(
        step_cost=PURE_STEP_COST, gamma=0.99, phi_scale=PURE_PHI_SCALE,
        subgoal_bonus=PURE_SUBGOAL_BONUS, goal_reward=PURE_GOAL_REWARD,
        shaping_cap=PURE_SHAPING_CAP, min_subgoal_size=PURE_MIN_SUBGOAL_SIZE,
    )
    assert_objective_only(rcfg)
    _train_shaped(agent, cases, episodes, rcfg)


def evolve_one_generation_pure(
    population: List[Genome], gen: int, bundle, cfg: EvoConfig,
    rng: random.Random, seed_base: int,
) -> Tuple[List[Genome], dict]:
    """One PURE generation: evaluate on the objective, then select/reproduce.

    Delegates to the shared Phase A reproduction machinery with the
    scaffolding-free reward config, so crossover/mutation/elitism are unchanged.
    """
    return _evolve_one_generation_shaped(
        population, gen, bundle, cfg, rng, seed_base, pure_reward_config(cfg))


def core_validation_pure(genome: Genome, cfg: EvoConfig, seed: int,
                         episodes: int) -> Tuple[int, int]:
    """Core 33-case validation seeded from the genome, trained pure.

    Implemented here (rather than delegating) because the Phase A
    ``core_validation_shaped`` does not forward a custom reward config to its
    trainer; we need the pure config to actually reach ``train_shaped``.
    """
    from .evolution_agents import CORE_ARITY, CORE_ORDER, evaluate
    from .evolution_run import core_train_cases, core_validation_cases
    from .evolution_semi import _make_case_agent

    agent = _make_case_agent(genome, cfg, CORE_ORDER, CORE_ARITY, seed,
                             episodes)
    train_pure(agent, core_train_cases(), episodes)
    res = evaluate(agent, core_validation_cases())
    return res.passed, res.total


def evo_validation_pure(genome: Genome, cfg: EvoConfig, seed: int,
                        episodes: int) -> Tuple[int, int]:
    """Extended 114-case validation seeded from the genome, trained pure."""
    from .evolution_agents import evaluate
    from .evolution_run import evo_train_cases, evo_validation_cases
    from .evolution_semi import _make_case_agent

    agent = _make_case_agent(genome, cfg, EVO_ORDER, EVO_ARITY, seed, episodes)
    train_pure(agent, evo_train_cases(), episodes)
    res = evaluate(agent, evo_validation_cases())
    return res.passed, res.total


def bundle_validation_pure(genome: Genome, bundle, cfg: EvoConfig,
                           seed: int):
    """Greedy bundle evaluation with the pure objective."""
    return _bundle_validation_shaped(genome, bundle, cfg, seed,
                                     pure_reward_config(cfg))

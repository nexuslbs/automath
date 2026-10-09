"""Unit B (Phase A) deterministic tests: the shaped reward scheme and the
semi-evolutionary generation machinery.

Run with::

    /opt/automath/venv/bin/python -m new_approach.evolution_semi_tests

Every check is seeded and bounded.  The prior suites
(``new_approach.tests``, ``evolution_tests``, ``evolution_c_tests``) are
separate and must stay green.
"""

from __future__ import annotations

import random
import time
from typing import List, Tuple

from .env import FixedStateGoal
from .evolution import EVO_AXIOMS, EvoEnv, ev_add, ev_mul, ev_sub, nat
from .evolution_bundle import make_demo_bundle
from .evolution_agents import EVO_ARITY, EVO_ORDER, EvolutionAgent
from .evolution_population import EvoConfig, Genome, random_genome
from .evolution_rewards import (
    GOAL_REWARD,
    PHI_SCALE,
    REWARD_GAMMA,
    SHAPING_CAP,
    STEP_COST,
    SUBGOAL_BONUS,
    EpisodeShaper,
    RewardConfig,
    goal_similarity,
    is_proper_subgoal,
    match_count,
    phi,
)
from .evolution_semi import (
    blx,
    bundle_validation,
    core_validation_shaped,
    evolve_one_generation_shaped,
    run_bundle_shaped,
)
from .nodes import Group, One, Zero
from .tests import CheckFailure

SEED = 20261009


def _cfg(**kw) -> EvoConfig:
    base = dict(population=4, episodes=1, generations=1, total_budget=20,
                elites=2, top_k=2, seed=SEED,
                checkpoint_dir="/tmp/semi-evo-test/checkpoints",
                progress_dir="/tmp/semi-evo-test/progress",
                history_every=1, checkpoint_every=1)
    base.update(kw)
    return EvoConfig(**base)


def _mini_target():
    # mul(add(1,1), sub(3,1)); the proper sub-structure add(1,1) has size 4.
    return ev_mul(ev_add(One(), One()), ev_sub(nat(3), One()))


def check_reward_constants() -> str:
    rc = RewardConfig()
    want = (rc.step_cost == STEP_COST == -0.05
            and rc.gamma == REWARD_GAMMA == 0.99
            and rc.phi_scale == PHI_SCALE == 0.5
            and rc.subgoal_bonus == SUBGOAL_BONUS == 0.10
            and rc.goal_reward == GOAL_REWARD == 1.0
            and rc.shaping_cap == SHAPING_CAP == 0.9)
    if not want:
        raise CheckFailure("reward_constants", "constants drifted: %r" % (rc,))
    return ("step_cost=%.2f gamma=%.2f phi_scale=%.2f bonus=%.2f goal=%.2f "
            "cap=%.2f" % (rc.step_cost, rc.gamma, rc.phi_scale,
                          rc.subgoal_bonus, rc.goal_reward, rc.shaping_cap))


def check_goal_similarity() -> str:
    target = _mini_target()
    env = EvoEnv(FixedStateGoal(target))
    state = env.reset()
    if abs(goal_similarity(target, state)) > 1e-12:
        raise CheckFailure("goal_similarity", "empty stack similarity nonzero")
    # build the proper sub-structure add(1,1)
    for a in ("PushOne", "PushOne", "MakeAdd"):
        state = env.step(state, a)
    gs = goal_similarity(target, state)
    if abs(gs - 4.0 / 15.0) > 1e-9:
        raise CheckFailure("goal_similarity",
                           "expected 4/15 for add(1,1), got %.6f" % gs)
    # finish the target -> similarity 1.0
    word = ("PushOne", "MakeChange", "MakeChange", "PushOne", "MakeSub",
            "MakeMul")
    for a in word:
        state = env.step(state, a)
    if abs(goal_similarity(target, state) - 1.0) > 1e-12:
        raise CheckFailure("goal_similarity",
                           "goal similarity != 1 at target")
    return ("empty=0.0 proper(add(1,1))=%.6f goal=1.000000 (match_count=%d "
            "size=%d)" % (gs, match_count(target, target), 15))


def check_pbrs_formula() -> str:
    target = _mini_target()
    env = EvoEnv(FixedStateGoal(target))
    shaper = EpisodeShaper()
    state = env.reset()
    checked = 0
    for a in ("PushOne", "PushOne", "MakeAdd", "PushOne", "MakeChange"):
        nxt = env.step(state, a)
        rb = shaper.reward(target, state, nxt, False)
        want = REWARD_GAMMA * phi(target, nxt) - phi(target, state)
        if abs(rb.shaping - want) > 1e-12:
            raise CheckFailure("pbrs_formula",
                               "F != gamma*Phi(s')-Phi(s): %.9f vs %.9f"
                               % (rb.shaping, want))
        checked += 1
        state = nxt
    return ("F = gamma*Phi(s')-Phi(s) exact on %d steps (gamma=%.2f)"
            % (checked, REWARD_GAMMA))


def check_subgoal_bonus() -> str:
    target = _mini_target()
    env = EvoEnv(FixedStateGoal(target))
    shaper = EpisodeShaper()
    state = env.reset()
    state = env.step(state, "PushOne")
    state = env.step(state, "PushOne")
    nxt = env.step(state, "MakeAdd")
    rb = shaper.reward(target, state, nxt, False)
    top = nxt.stack[-1]
    if not is_proper_subgoal(target, top):
        raise CheckFailure("subgoal_bonus", "add(1,1) not a proper subgoal")
    if abs(rb.bonus - SUBGOAL_BONUS) > 1e-12 or not rb.subgoal:
        raise CheckFailure("subgoal_bonus",
                           "expected +%.2f bonus, got %.6f" % (SUBGOAL_BONUS,
                                                               rb.bonus))
    # the goal step itself is NOT a proper sub-goal (it earns the +1.0)
    state = nxt
    for a in ("PushOne", "MakeChange", "MakeChange", "PushOne", "MakeSub",
              "MakeMul"):
        nxt = env.step(state, a)
        rb2 = shaper.reward(target, state, nxt, True)
        if rb2.subgoal:
            raise CheckFailure("subgoal_bonus", "goal step flagged as subgoal")
        state = nxt
    return ("proper sub-structure %s -> bonus=+%.2f (top==target excluded)"
            % (top.canonical(), rb.bonus))


def check_no_artificial_reward_nodes() -> str:
    import new_approach.evolution_rewards as rewards
    if hasattr(rewards, "reward_node") or hasattr(rewards, "read_reward"):
        raise CheckFailure("no_artificial_reward_nodes",
                           "Stage 2 reward-node helpers leaked into the path")
    # No state produced by a shaped episode may ever contain a Group whose tag
    # evaluates to T_REWARD (20); the action vocabulary never builds one.
    target = _mini_target()
    env = EvoEnv(FixedStateGoal(target))
    state = env.reset()
    seen_tags = []
    for a in ("PushOne", "PushOne", "MakeAdd", "PushOne", "MakeChange",
              "MakeChange", "PushOne", "MakeSub", "MakeMul"):
        state = env.step(state, a)
        for node in state.stack:
            if isinstance(node, Group):
                try:
                    tv = EVO_AXIOMS.evaluate(node.tag)
                except Exception:
                    tv = None
                seen_tags.append(tv)
    if 20 in seen_tags:
        raise CheckFailure("no_artificial_reward_nodes",
                           "tag 20 (T_REWARD) appeared in a shaped episode")
    return "no reward_node/read_reward; no T_REWARD(20) tag in %d states" \
        % len(seen_tags)


def check_shaping_cap() -> str:
    target = _mini_target()
    env = EvoEnv(FixedStateGoal(target))
    shaper = EpisodeShaper(RewardConfig(shaping_cap=0.2))
    state = env.reset()
    total_applied = 0.0
    for a in ("PushOne", "PushOne", "MakeAdd", "PushOne", "MakeChange",
              "MakeChange", "PushOne", "MakeSub", "MakeMul"):
        nxt = env.step(state, a)
        rb = shaper.reward(target, state, nxt, False)
        total_applied += rb.shaping_applied
        state = nxt
    if shaper.positive_used > 0.2 + 1e-9:
        raise CheckFailure("shaping_cap", "cap exceeded: %.6f"
                           % shaper.positive_used)
    if total_applied > 0.2 + 1e-9:
        raise CheckFailure("shaping_cap", "applied positive %.6f > cap"
                           % total_applied)
    return ("cap=0.20 respected: positive_used=%.6f applied=%.6f"
            % (shaper.positive_used, total_applied))


def check_shaped_run_items() -> str:
    cfg = _cfg()
    bundle = make_demo_bundle(cfg.total_budget)
    rng = random.Random(SEED)
    genome = random_genome(rng, len(EVO_ORDER), len(bundle.states), cfg,
                           "g00001-r000")
    agent = EvolutionAgent(action_order=EVO_ORDER, arity=EVO_ARITY, seed=SEED,
                           episodes=1, use_pref=True, pref_init=genome.pref_map(),
                           switch_patience=genome.switch_patience)
    res = run_bundle_shaped(agent, bundle, genome.state_pref,
                            total_budget=cfg.total_budget, training=False)
    already = sum(1 for e in res.reward_log if e.get("already_goal"))
    if len(res.reward_log) != res.total_steps + already:
        raise CheckFailure("shaped_run_items",
                           "reward rows %d != steps %d + already_goal %d"
                           % (len(res.reward_log), res.total_steps, already))
    if res.total_steps == 0:
        raise CheckFailure("shaped_run_items", "no steps taken")
    for row in res.reward_log:
        want = (row["step_cost"] + row["shaping_applied"] + row["final_reward"])
        if abs(want - row["total"]) > 1e-9:
            raise CheckFailure("shaped_run_items",
                               "total != step_cost+applied+final at step %s"
                               % row.get("step"))
    return ("bundle run steps=%d shaped_return=%.4f reward rows=%d "
            "positive_shaping=%.4f" % (res.total_steps, res.shaped_return,
                                       len(res.reward_log),
                                       res.positive_shaping))


def check_generation_shaped() -> str:
    cfg = _cfg()
    bundle = make_demo_bundle(cfg.total_budget)
    rng = random.Random(SEED)
    from .evolution_population import init_population
    pop = init_population(cfg, bundle, rng)
    new_pop, rec = evolve_one_generation_shaped(pop, 0, bundle, cfg, rng,
                                                cfg.seed)
    if len(new_pop) != cfg.population:
        raise CheckFailure("generation_shaped", "population size changed")
    for key in ("best_fitness", "best_shaped_return", "threshold", "agents",
                "offspring", "best_per_state"):
        if key not in rec:
            raise CheckFailure("generation_shaped", "record missing %r" % key)
    if not all(isinstance(a["fitness"], float) for a in rec["agents"]):
        raise CheckFailure("generation_shaped", "non-float fitness")
    return ("pop=%d best_fitness=%.4f best_shaped=%.4f offspring=%d "
            "threshold=%.4f" % (len(new_pop), rec["best_fitness"],
                                rec["best_shaped_return"],
                                len(rec["offspring"]), rec["threshold"]))


def check_blx_bounds() -> str:
    rng = random.Random(SEED)
    a = [3.0, -4.0, 0.0]
    b = [-3.0, 4.0, 1.0]
    kids = [blx(a, b, 0.5, rng, clip=5.0) for _ in range(50)]
    for k in kids:
        if any(abs(v) > 5.0 + 1e-12 for v in k):
            raise CheckFailure("blx_bounds", "child gene outside clip: %r" % k)
    return "50 BLX-alpha children all within [-5,5]"


def check_core_validation_shaped() -> str:
    cfg = _cfg(validation_episodes=1)
    rng = random.Random(SEED)
    bundle = make_demo_bundle(cfg.total_budget)
    genome = random_genome(rng, len(EVO_ORDER), len(bundle.states), cfg,
                           "g00001-r000")
    passed, total = core_validation_shaped(genome, cfg, SEED, episodes=1)
    if total != 33:
        raise CheckFailure("core_validation_shaped",
                           "expected 33 cases, got %d" % total)
    return "core U2 shaped validation = %d/%d" % (passed, total)


def check_bundle_validation() -> str:
    cfg = _cfg()
    bundle = make_demo_bundle(cfg.total_budget)
    rng = random.Random(SEED)
    genome = random_genome(rng, len(EVO_ORDER), len(bundle.states), cfg,
                           "g00001-r000")
    res = bundle_validation(genome, bundle, cfg, SEED)
    if res.solved_feasible > len(bundle.feasible()):
        raise CheckFailure("bundle_validation", "solved > feasible")
    if not res.reward_log:
        raise CheckFailure("bundle_validation", "empty reward log")
    return ("bundle shaped run solved=%d/%d shaped_return=%.4f steps=%d"
            % (res.solved_feasible, len(bundle.feasible()),
               res.shaped_return, res.total_steps))


NEW_CHECKS: List[Tuple[str, object]] = [
    ("reward_constants", check_reward_constants),
    ("goal_similarity", check_goal_similarity),
    ("pbrs_formula", check_pbrs_formula),
    ("subgoal_bonus", check_subgoal_bonus),
    ("no_artificial_reward_nodes", check_no_artificial_reward_nodes),
    ("shaping_cap", check_shaping_cap),
    ("shaped_run_items", check_shaped_run_items),
    ("generation_shaped", check_generation_shaped),
    ("blx_bounds", check_blx_bounds),
    ("core_validation_shaped", check_core_validation_shaped),
    ("bundle_validation", check_bundle_validation),
]


def run() -> int:
    started = time.perf_counter()
    passed = 0
    failures: List[str] = []
    for name, fn in NEW_CHECKS:
        try:
            detail = fn()
        except CheckFailure as exc:
            failures.append(exc.name)
            print("[FAIL] %s: %s" % (name, exc.detail))
            if exc.feedback:
                print("       %s" % exc.feedback)
        except Exception as exc:  # noqa: BLE001
            failures.append(name)
            print("[FAIL] %s: unexpected %r" % (name, exc))
        else:
            passed += 1
            print("[PASS] %s: %s" % (name, detail))
    elapsed = time.perf_counter() - started
    total = len(NEW_CHECKS)
    print("RESULT: %d/%d passed in %.3fs (unit-B Phase A: shaped reward + "
          "semi-evolution; deterministic)" % (passed, total, elapsed))
    return 0 if not failures else 1


if __name__ == "__main__":
    import sys
    sys.exit(run())

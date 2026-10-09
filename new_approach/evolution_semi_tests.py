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
from .evolution_bundle import make_demo_bundle, start_order
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
    evaluate_genome_shaped,
    evolve_one_generation_shaped,
    make_shaped_agent,
    run_bundle_shaped,
)
from .curriculum import (
    CurriculumSchedule,
    DEFAULT_PHASES,
    TIER_EASY,
    TIER_HARD,
    TIER_MEDIUM,
    assign_tier,
    difficulty,
    excluded_infeasible,
    tiered_cases,
)
from .nodes import Group, One, Zero, size
from .target_features import (
    PolicyNet,
    START_ACTION,
    features,
    net_init,
    net_size,
)
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
    # the goal step itself is NOT a proper sub-goal (it earns the +1.0);
    # intermediate sub-structures (e.g. sub(3,1)) legitimately DO earn a bonus.
    state = nxt
    final_rb = None
    for a in ("PushOne", "MakeChange", "MakeChange", "PushOne", "MakeSub",
              "MakeMul"):
        nxt = env.step(state, a)
        final_rb = shaper.reward(target, state, nxt, True)
        state = nxt
    assert final_rb is not None
    if final_rb.subgoal or abs(final_rb.bonus) > 1e-12:
        raise CheckFailure("subgoal_bonus", "goal step flagged as subgoal")
    if abs(final_rb.final - GOAL_REWARD) > 1e-12:
        raise CheckFailure("subgoal_bonus", "goal step missing +1.0")
    return ("proper sub-structure %s -> bonus=+%.2f (top==target excluded; "
            "goal final=+%.2f)" % (top.canonical(), rb.bonus, final_rb.final))


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
    agent = make_shaped_agent(genome, cfg, SEED)
    res = run_bundle_shaped(agent, bundle, None,
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


# --------------------------------------------------------------------------
# Flavor B: curriculum schedule checks (tiers, transitions, config keys, and
# the proof that the flavor-A fitness formula is untouched on a curriculum
# bundle).
# --------------------------------------------------------------------------

def check_curriculum_tiers() -> str:
    tiers = tiered_cases()
    counts = {t: len(tiers[t]) for t in (TIER_EASY, TIER_MEDIUM, TIER_HARD)}
    if any(counts[t] == 0 for t in (TIER_EASY, TIER_MEDIUM, TIER_HARD)):
        raise CheckFailure("curriculum_tiers", "empty tier: %r" % counts)
    excluded = excluded_infeasible()
    if sum(counts.values()) + excluded != 320:
        raise CheckFailure("curriculum_tiers",
                           "partition %d + excluded %d != 320"
                           % (sum(counts.values()), excluded))
    # determinism: a fresh partition assigns exactly the same case names
    again = tiered_cases()
    for t in (TIER_EASY, TIER_MEDIUM, TIER_HARD):
        if (sorted(c.name for c in tiers[t])
                != sorted(c.name for c in again[t])):
            raise CheckFailure("curriculum_tiers",
                               "nondeterministic tier %d" % t)
    if assign_tier(tiers[TIER_EASY][0]) != TIER_EASY:
        raise CheckFailure("curriculum_tiers",
                           "assign_tier disagrees with the partition")
    return ("counts easy=%d medium=%d hard=%d excluded_unreachable=%d "
            "total=320, deterministic"
            % (counts[TIER_EASY], counts[TIER_MEDIUM], counts[TIER_HARD],
               excluded))


def check_curriculum_tier_ordering() -> str:
    tiers = tiered_cases()
    means = {}
    for t in (TIER_EASY, TIER_MEDIUM, TIER_HARD):
        ds = [difficulty(c) for c in tiers[t]]
        means[t] = sum(ds) / len(ds)
    if not (means[TIER_EASY] < means[TIER_MEDIUM] < means[TIER_HARD]):
        raise CheckFailure("curriculum_tier_ordering",
                           "difficulty not increasing: %r" % means)
    return ("mean difficulty easy=%.2f < medium=%.2f < hard=%.2f"
            % (means[TIER_EASY], means[TIER_MEDIUM], means[TIER_HARD]))


def check_curriculum_phase_transitions() -> str:
    phases = [
        {"name": "p0", "tiers": [TIER_EASY], "generations": 2, "batch": 5},
        {"name": "p1", "tiers": [TIER_EASY, TIER_MEDIUM], "generations": 3,
         "batch": 7},
        {"name": "p2", "tiers": [TIER_EASY, TIER_MEDIUM, TIER_HARD],
         "generations": 4, "batch": 9},
    ]
    sch = CurriculumSchedule(phases=phases, seed=SEED)
    want = {0: 0, 1: 0, 2: 1, 4: 1, 5: 2, 99: 2}
    for gen, idx in want.items():
        got = sch.phase_index(gen)
        if got != idx:
            raise CheckFailure("curriculum_phase_transitions",
                               "gen %d -> phase %d, want %d"
                               % (gen, got, idx))
    if sch.tiers_for(0) != [TIER_EASY]:
        raise CheckFailure("curriculum_phase_transitions",
                           "tiers_for(0) wrong")
    if sch.tiers_for(2) != [TIER_EASY, TIER_MEDIUM]:
        raise CheckFailure("curriculum_phase_transitions",
                           "tiers_for(2) wrong")
    if sch.tiers_for(5) != [TIER_EASY, TIER_MEDIUM, TIER_HARD]:
        raise CheckFailure("curriculum_phase_transitions",
                           "tiers_for(5) wrong")
    if (sch.batch_for(0), sch.batch_for(2), sch.batch_for(5)) != (5, 7, 9):
        raise CheckFailure("curriculum_phase_transitions",
                           "per-phase batch not honored")
    return "gen 0-1=p0, 2-4=p1, 5+=p2; expanding tiers; batches 5/7/9"


def check_curriculum_config_keys() -> str:
    if CurriculumSchedule.from_config({"enabled": False}, SEED) is not None:
        raise CheckFailure("curriculum_config_keys",
                           "enabled:false not honored")
    if CurriculumSchedule.from_config(None, SEED) is not None:
        raise CheckFailure("curriculum_config_keys", "None config not honored")
    data = {
        "enabled": True,
        "seed": SEED + 1,
        "rule": "generation_schedule",
        "thresholds": {"easy_prefix_max_target_size": 1},
        "phases": [{"name": "only", "tiers": [TIER_EASY], "generations": 10,
                    "batch": 3}],
    }
    sch = CurriculumSchedule.from_config(data, SEED)
    if sch is None or sch.seed != SEED + 1:
        raise CheckFailure("curriculum_config_keys", "seed key not honored")
    if sch.phase_index(50) != 0 or sch.batch_for(0) != 3:
        raise CheckFailure("curriculum_config_keys",
                           "phases/batch keys not honored")
    # a default-EASY prefix with target size > 1 must leave EASY once the
    # easy target-size threshold is tightened to 1
    known = None
    for c in tiered_cases()[TIER_EASY]:
        target = getattr(c.env.goal, "target", None)
        if "/prefix" in c.name and target is not None and size(target) > 1:
            known = c
            break
    if known is None:
        raise CheckFailure("curriculum_config_keys",
                           "no tier-1 prefix with target size > 1")
    if assign_tier(known, data["thresholds"]) == TIER_EASY:
        raise CheckFailure("curriculum_config_keys",
                           "threshold key not honored for %s" % known.name)
    try:
        CurriculumSchedule(phases=DEFAULT_PHASES, seed=SEED, rule="nope")
    except ValueError:
        pass
    else:
        raise CheckFailure("curriculum_config_keys", "bad rule accepted")
    # sampling is a pure function of the (seed, generation) pair
    s0a = [c.name for c in sch.sample_for(0)]
    s0b = [c.name for c in sch.sample_for(0)]
    s1 = [c.name for c in sch.sample_for(1)]
    if s0a != s0b:
        raise CheckFailure("curriculum_config_keys",
                           "sampling is not deterministic")
    if s0a == s1 and len(sch.cases_by_tier[TIER_EASY]) > 3:
        raise CheckFailure("curriculum_config_keys",
                           "generation not mixed into the sample seed")
    return ("enabled/seed/phases/batch/thresholds honored; bad rule rejected; "
            "sample(gen0)==sample(gen0) and differs from sample(gen1)")


def check_curriculum_bundle_and_fitness() -> str:
    sch = CurriculumSchedule(
        phases=[{"name": "easy", "tiers": [TIER_EASY], "generations": 5,
                 "batch": 4}], seed=SEED)
    bundle = sch.bundle_for(0, 20)
    if not bundle.states or len(bundle.states) > 4:
        raise CheckFailure("curriculum_bundle_fitness",
                           "bad bundle size %d" % len(bundle.states))
    if any(not s.feasible for s in bundle.states):
        raise CheckFailure("curriculum_bundle_fitness",
                           "infeasible dense tier-1 state in bundle")
    cfg = _cfg(episodes=1, total_budget=20)
    rng = random.Random(SEED)
    g = random_genome(rng, len(EVO_ORDER), len(bundle.states), cfg, "g-cur")
    res = evaluate_genome_shaped(g, bundle, cfg, SEED)
    want = res.shaped_return + cfg.solved_rate_weight * res.solved_rate()
    if abs(g.fitness - want) > 1e-9:
        raise CheckFailure("curriculum_bundle_fitness",
                           "fitness formula drifted: %r != %r"
                           % (g.fitness, want))
    return ("bundle states=%d feasible=%d fitness=%.4f == shaped_return + "
            "%.2f*solved_rate"
            % (len(bundle.states), len(bundle.feasible()), g.fitness,
               cfg.solved_rate_weight))


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
    ("curriculum_tiers", check_curriculum_tiers),
    ("curriculum_tier_ordering", check_curriculum_tier_ordering),
    ("curriculum_phase_transitions", check_curriculum_phase_transitions),
    ("curriculum_config_keys", check_curriculum_config_keys),
    ("curriculum_bundle_fitness", check_curriculum_bundle_and_fitness),
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


# --------------------------------------------------------------------------
# pytest-style tests for the flavor-A target-conditioned net policy.
# Run with:  python -m pytest new_approach/evolution_semi_tests.py
# --------------------------------------------------------------------------

def test_net_forward_shape() -> None:
    """The policy net has the fixed genome size and a finite scalar output."""
    rng = random.Random(SEED)
    net = net_init(rng, 0.5)
    assert len(net) == net_size()
    assert net_size() == PolicyNet.size()
    x = features(_mini_target(), (), "PushZero")
    assert len(x) == PolicyNet.IN_DIM
    out = PolicyNet.forward(net, x)
    assert isinstance(out, float)
    assert out == out  # not NaN


def test_serialization_has_net_and_no_pref() -> None:
    """A genome serializes with ``net`` and WITHOUT pref/state_pref."""
    rng = random.Random(SEED)
    cfg = _cfg()
    g = random_genome(rng, len(EVO_ORDER), 7, cfg, "g-net")
    genes = g.genes()
    assert "net" in genes
    assert "pref" not in genes and "state_pref" not in genes
    d = g.to_dict()
    assert "net" in d
    assert "pref" not in d and "state_pref" not in d
    back = Genome.from_dict(d)
    assert len(back.net) == net_size()
    assert back.genes() == genes
    assert back.switch_patience == g.switch_patience


def test_start_order_scored_by_net() -> None:
    """The bundle start order is the net's ranking over '__start__'."""
    cfg = _cfg()
    bundle = make_demo_bundle(cfg.total_budget)
    rng = random.Random(SEED)
    g = random_genome(rng, len(EVO_ORDER), len(bundle.states), cfg, "g-order")
    agent = make_shaped_agent(g, cfg, SEED)
    scores = agent.start_scores(bundle.states)
    assert len(scores) == len(bundle.states)
    order = start_order(bundle, scores)
    expected = [sid for sid, _ in sorted(zip(bundle.ids(), scores),
                                         key=lambda p: (-p[1], p[0]))]
    assert order == expected
    # The same net scores differently for a different bundle (target + stack
    # enter the feature vector): the policy is target-conditioned.
    other = random_genome(random.Random(SEED + 1), len(EVO_ORDER), 7, cfg, "g2")
    other_agent = make_shaped_agent(other, cfg, SEED)
    assert other_agent.start_scores(bundle.states) != scores


def test_action_scores_same_net_new_targets() -> None:
    """The SAME net parameters yield a valid greedy action for any target."""
    cfg = _cfg()
    bundle = make_demo_bundle(cfg.total_budget)
    rng = random.Random(SEED)
    g = random_genome(rng, len(EVO_ORDER), len(bundle.states), cfg, "g-policy")
    agent = make_shaped_agent(g, cfg, SEED)
    s1, s2 = bundle.by_id("s1"), bundle.by_id("s2")
    env = EvoEnv(FixedStateGoal(s1.target))
    st = env.reset()
    agent.set_target(s1.target)
    a1 = agent.best_action(s1.key(), st)
    assert a1 in agent.valid(st.stack)
    agent.set_target(s2.target)
    a2 = agent.best_action(s2.key(), st)
    assert a2 in agent.valid(st.stack)


# --------------------------------------------------------------------------
# pytest-style tests for the flavor-B curriculum schedule.
# --------------------------------------------------------------------------

def test_curriculum_tiers_deterministic() -> None:
    """Tier assignment is deterministic and covers the 320 dense cases."""
    check_curriculum_tiers()


def test_curriculum_tier_difficulty_order() -> None:
    """Mean difficulty is strictly increasing over the three tiers."""
    check_curriculum_tier_ordering()


def test_curriculum_phase_transitions() -> None:
    """Phase boundaries follow the configured per-phase generation lengths."""
    check_curriculum_phase_transitions()


def test_curriculum_config_keys_honored() -> None:
    """enabled/seed/rule/thresholds/phases/batch are all honored."""
    check_curriculum_config_keys()


def test_curriculum_fitness_formula_unchanged() -> None:
    """A curriculum bundle still scores with the flavor-A fitness formula."""
    check_curriculum_bundle_and_fitness()

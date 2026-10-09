"""Unit C deterministic tests: multi-state total budget, impossible goals,
genome evolution, selection and persistence.

Run with::

    /opt/automath/venv/bin/python -m new_approach.evolution_c_tests

The prior suites (``new_approach.tests``, 11 checks, and
``new_approach.evolution_tests``, 20 checks) are separate and must stay green;
this module adds the Unit C contract on top.  Every check is seeded and bounded.
"""

from __future__ import annotations

import os
import random
import shutil
import sys
import tempfile
import time
from typing import Callable, List, Tuple

from .evolution import EVO_ACTIONS, EVO_AXIOMS, EvoEnv, bfs_plan_evo
from .env import FixedStateGoal
from .evolution_agents import EVO_ARITY, EVO_ORDER, EvolutionAgent
from .evolution_bundle import (
    GOAL_REWARD,
    make_demo_bundle,
    run_bundle,
    target_feasible,
)
from .evolution_population import (
    EvoConfig,
    Genome,
    evolve_one_generation,
    init_population,
    load_checkpoint,
    mutate,
    crossover,
    random_genome,
    restore_population,
    save_generation,
)
from .nodes import Group, One, Zero
from .tests import CheckFailure

SEED = 20261009


def _agent(seed: int = SEED, patience: int = 4,
           episodes: int = 1) -> EvolutionAgent:
    return EvolutionAgent(
        action_order=EVO_ORDER, arity=EVO_ARITY, seed=seed,
        episodes=episodes, use_pref=True, pref_sigma=0.5,
        switch_patience=patience,
    )


def _cfg(tmp: str, **kw) -> EvoConfig:
    base = dict(population=6, episodes=2, generations=2, total_budget=30,
                elites=2, top_k=2,
                checkpoint_dir=os.path.join(tmp, "ck"),
                progress_dir=os.path.join(tmp, "pr"))
    base.update(kw)
    return EvoConfig(**base)


def check_bundle_feasibility_derived() -> str:
    b = make_demo_bundle(40)
    if len(b.feasible()) < 2:
        raise CheckFailure("bundle_feasibility", "need >=2 feasible states")
    if len(b.impossible()) < 1:
        raise CheckFailure("bundle_feasibility", "need >=1 impossible goal")
    for s in b.states:
        ok, why = target_feasible(s.target)
        if ok != s.feasible:
            raise CheckFailure(
                "bundle_feasibility",
                "%s feasibility mismatch: declared=%s derived=%s"
                % (s.name, s.feasible, ok))
        if not s.feasible and not s.impossible_reason:
            raise CheckFailure("bundle_feasibility",
                               "%s has no impossible reason" % s.name)
    for s in b.feasible():
        if s.expected_len <= 0:
            raise CheckFailure("bundle_feasibility",
                               "%s expected_len=0" % s.name)
    return ("%d states: %d feasible (len %s) + %d derived impossible (%s)"
            % (len(b.states), len(b.feasible()),
               [s.expected_len for s in b.feasible()],
               len(b.impossible()),
               [s.impossible_reason for s in b.impossible()]))


def check_impossible_unreachable() -> str:
    b = make_demo_bundle(40)
    s4 = b.by_id("s4")
    # No action can build tag 99 / arity 2 (structural proof).
    for act in EVO_ACTIONS:
        if act.arity != 2:
            continue
        built = act.build((One(), Zero()))
        if isinstance(built, Group):
            tag = EVO_AXIOMS.evaluate(built.tag)
            if tag == 99:
                raise CheckFailure("impossible_unreachable",
                                   "%s builds tag 99" % act.name)
    env3 = EvoEnv(FixedStateGoal(s4.target), max_actions=4)
    word = bfs_plan_evo(env3, 4)
    if word is not None:
        raise CheckFailure("impossible_unreachable",
                           "BFS found a word to an impossible target: %r"
                           % (word,))
    return ("no build action produces tag 99/arity 2 and bounded BFS (depth 4) "
            "finds no word: %s" % s4.impossible_reason)


def check_total_budget_shared() -> str:
    b = make_demo_bundle(25)
    ag = _agent()
    r = run_bundle(ag, b, [0.0] * len(b.states), total_budget=25,
                   training=False)
    if r.total_steps != 25:
        raise CheckFailure("total_budget_shared",
                           "expected the full shared budget 25, got %d"
                           % r.total_steps)
    s = sum(sr.steps for sr in r.per_state.values())
    if s != r.total_steps:
        raise CheckFailure("total_budget_shared",
                           "per-state steps %d != total %d"
                           % (s, r.total_steps))
    return ("one shared budget=25 across %d states; total steps=%d == sum of "
            "per-state steps=%d (NOT a per-state budget)"
            % (len(b.states), r.total_steps, s))


def check_switch_on_impossible() -> str:
    b = make_demo_bundle(40)
    ag = _agent(seed=SEED + 1, patience=3)
    sp = [0.0] * len(b.states)
    sp[b.index("s4")] = 5.0
    r = run_bundle(ag, b, sp, total_budget=40, training=False)
    if not r.trace or r.trace[0].kind != "start" or r.trace[0].sid != "s4":
        raise CheckFailure("switch_on_impossible",
                           "first state was not s4: %s"
                           % (r.trace[0] if r.trace else None,))
    sw = [e for e in r.trace if e.kind == "switch" and e.sid == "s4"]
    if not sw:
        raise CheckFailure("switch_on_impossible",
                           "no switch away from the impossible state")
    reason = sw[0].detail
    if ("stagnation" not in reason and "budget-ratio" not in reason):
        raise CheckFailure("switch_on_impossible",
                           "unexplained switch: %r" % reason)
    if r.per_state["s4"].solved:
        raise CheckFailure("switch_on_impossible",
                           "impossible state reported solved")
    # The structural-progress metric keeps rising on an impossible target
    # (larger and larger partial trees), so the expected trigger is the
    # budget-ratio one; the check is that the impossible state is abandoned
    # before it eats more than half of the shared budget.
    first_burn = sw[0].step
    if first_burn > 21:
        raise CheckFailure("switch_on_impossible",
                           "burned %d steps before switching" % first_burn)
    if r.per_state["s4"].steps >= 40:
        raise CheckFailure("switch_on_impossible",
                           "impossible state consumed the whole budget")
    started = [sid for sid, sr in r.per_state.items() if sr.started]
    if len(started) < 2:
        raise CheckFailure("switch_on_impossible",
                           "agent never moved to another state: %s" % started)
    return ("started s4 (impossible); first switch after %d steps: %r; s4 "
            "total steps=%d status=%s; states started=%s; total=%d/%d"
            % (first_burn, reason, r.per_state["s4"].steps,
               r.per_state["s4"].status(), started, r.total_steps, 40))


def check_switch_trigger_reasons() -> str:
    b = make_demo_bundle(60)
    ag = _agent(seed=SEED + 2, patience=3)
    r = run_bundle(ag, b, [0.0] * len(b.states), total_budget=60,
                   training=False)
    switches = [e for e in r.trace if e.kind == "switch"]
    if not switches:
        raise CheckFailure("switch_trigger_reasons", "no switch happened")
    for e in switches:
        if not e.detail.startswith(("stagnation:", "budget-ratio:")):
            raise CheckFailure("switch_trigger_reasons",
                               "bad reason: %r" % e.detail)
    kinds = sorted({e.detail.split(":")[0] for e in switches})
    return ("%d switches, every one carries an explicit trigger: %s"
            % (len(switches), kinds))


def check_crossover_inherits_genes() -> str:
    cfg = EvoConfig(population=6)
    rng = random.Random(7)
    a = random_genome(rng, len(EVO_ORDER), 7, cfg, "a")
    b = random_genome(rng, len(EVO_ORDER), 7, cfg, "b")
    c = crossover(a, b, rng)
    for i in range(len(a.net)):
        if c.net[i] not in (a.net[i], b.net[i]):
            raise CheckFailure("crossover_inherits",
                               "net gene %d is neither parent" % i)
    d = c.cross_detail
    if d["net_from_a"] + d["net_from_b"] != len(a.net):
        raise CheckFailure("crossover_inherits", "net cross counts wrong")
    if set(c.parents) != {"a", "b"}:
        raise CheckFailure("crossover_inherits", "parents not recorded")
    return ("child inherits every net gene from A or B; net A/B=%d/%d "
            "eps=%s pat=%s"
            % (d["net_from_a"], d["net_from_b"], d["epsilon_from"],
               d["patience_from"]))


def check_mutation_bounded() -> str:
    cfg = EvoConfig(mutation_rate=1.0, mutation_sigma=0.15, pref_clip=5.0,
                    epsilon_min=0.02, epsilon_max=0.8)
    rng = random.Random(11)
    g = random_genome(rng, len(EVO_ORDER), 7, cfg, "m")
    mutate(g, rng, cfg)
    if any(abs(v) > 5.0 + 1e-9 for v in g.net):
        raise CheckFailure("mutation_bounded", "gene escaped the clip")
    if not (0.02 - 1e-9 <= g.epsilon <= 0.8 + 1e-9):
        raise CheckFailure("mutation_bounded",
                           "epsilon out of range: %r" % g.epsilon)
    if not (2 <= g.switch_patience <= 20):
        raise CheckFailure("mutation_bounded",
                           "patience out of range: %r" % g.switch_patience)
    return ("after a 100%% mutation pass: max|gene|=%.3f epsilon=%.3f "
            "patience=%d all within bounds"
            % (max(abs(v) for v in g.net), g.epsilon,
               g.switch_patience))


def check_selection_elitism_and_size() -> str:
    tmp = tempfile.mkdtemp(prefix="unitc-sel-")
    try:
        cfg = _cfg(tmp)
        b = make_demo_bundle(cfg.total_budget)
        rng = random.Random(cfg.seed)
        pop = init_population(cfg, b, rng)
        new_pop, rec = evolve_one_generation(pop, 0, b, cfg, rng, cfg.seed)
        old_best = max(a["fitness"] for a in rec["agents"])
        if len(new_pop) != cfg.population:
            raise CheckFailure("selection_elitism",
                               "population size changed: %d" % len(new_pop))
        if abs(new_pop[0].fitness - old_best) > 1e-9:
            raise CheckFailure("selection_elitism",
                               "elite fitness not preserved: %.6f vs %.6f"
                               % (new_pop[0].fitness, old_best))
        d = rec["offspring"][0]["cross_detail"]
        if d["net_from_a"] + d["net_from_b"] != len(new_pop[-1].net):
            raise CheckFailure("selection_elitism", "offspring genes not mixed")
        return ("population holds %d (elites=%d + offspring); elite fitness "
                "preserved %.4f; first cross net A/B=%d/%d"
                % (len(new_pop), cfg.elites, old_best, d["net_from_a"],
                   d["net_from_b"]))
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def check_best_fitness_nondecreasing() -> str:
    tmp = tempfile.mkdtemp(prefix="unitc-mono-")
    try:
        cfg = _cfg(tmp, generations=3)
        b = make_demo_bundle(cfg.total_budget)
        rng = random.Random(cfg.seed)
        pop = init_population(cfg, b, rng)
        bests = []
        for gen in range(3):
            pop, rec = evolve_one_generation(pop, gen, b, cfg, rng, cfg.seed)
            bests.append(rec["best_fitness"])
        for i in range(1, len(bests)):
            if bests[i] < bests[i - 1] - 1e-9:
                raise CheckFailure("best_fitness_nondecreasing",
                                   "best dropped %s -> %s"
                                   % (bests[i - 1], bests[i]))
        return "best fitness per generation (elitism keeps it monotone): %s" % (
            ["%+.3f" % x for x in bests],)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def _checkpoint_roundtrip_impl() -> str:
    tmp = tempfile.mkdtemp(prefix="unitc-ck-")
    try:
        cfg = _cfg(tmp)
        b = make_demo_bundle(cfg.total_budget)
        rng = random.Random(cfg.seed)
        pop = init_population(cfg, b, rng)
        pop, rec = evolve_one_generation(pop, 0, b, cfg, rng, cfg.seed)
        paths = save_generation(cfg, 1, pop, [rec], b, pop[0])
        for p in paths.values():
            if not os.path.exists(p) or os.path.getsize(p) <= 0:
                raise CheckFailure("checkpoint_roundtrip",
                                   "missing/empty %s" % p)
        ck = load_checkpoint(paths["checkpoint"])
        if ck["generation"] != 1 or len(ck["population"]) != cfg.population:
            raise CheckFailure("checkpoint_roundtrip",
                               "checkpoint content wrong")
        loaded = restore_population(ck)
        if [g.genes() for g in loaded] != [g.genes() for g in pop]:
            raise CheckFailure("checkpoint_roundtrip",
                               "genes changed across save/load")
        sizes = ", ".join("%s=%d" % (os.path.basename(k), os.path.getsize(v))
                          for k, v in paths.items())
        return ("JSON roundtrip preserves all %d genomes; files: %s"
                % (len(loaded), sizes))
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def check_warm_start_continues() -> str:
    tmp = tempfile.mkdtemp(prefix="unitc-warm-")
    try:
        cfg = _cfg(tmp)
        b = make_demo_bundle(cfg.total_budget)
        rng = random.Random(cfg.seed)
        pop = init_population(cfg, b, rng)
        pop, rec = evolve_one_generation(pop, 0, b, cfg, rng, cfg.seed)
        save_generation(cfg, 1, pop, [rec], b, pop[0])
        ck = load_checkpoint(os.path.join(cfg.checkpoint_dir,
                                          "checkpoint.json"))
        loaded_gen = ck["generation"]
        warm = restore_population(ck)
        rng2 = random.Random(cfg.seed + loaded_gen)
        warm2, rec2 = evolve_one_generation(warm, loaded_gen, b, cfg, rng2,
                                            cfg.seed)
        if rec2["generation"] != loaded_gen + 1:
            raise CheckFailure("warm_start_continues",
                               "generation did not continue: %d+1 -> %d"
                               % (loaded_gen, rec2["generation"]))
        if len(warm2) != cfg.population:
            raise CheckFailure("warm_start_continues", "bad warm population")
        return ("loaded generation=%d population=%d -> continued generation=%d"
                % (loaded_gen, len(warm), rec2["generation"]))
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def check_reward_accounting() -> str:
    b = make_demo_bundle(30)
    ag = _agent(seed=SEED + 3)
    # Preset a greedy policy that solves s0 = not(0) in exactly 2 steps, so the
    # solved path (goal reward) is exercised and can be accounted for.
    s0 = b.by_id("s0")
    env0 = EvoEnv(FixedStateGoal(s0.target))
    st0 = env0.reset()
    st1 = env0.step(st0, "PushZero")
    key = s0.key()
    ag.q[key] = {
        st0.canonical(): {"PushZero": 100.0},
        st1.canonical(): {"MakeNot": 100.0},
    }
    r = run_bundle(ag, b, [0.0] * len(b.states), total_budget=30,
                   training=False)
    s = sum(sr.reward for sr in r.per_state.values())
    if abs(s - r.total_reward) > 1e-6:
        raise CheckFailure("reward_accounting",
                           "per-state rewards %.6f != total %.6f"
                           % (s, r.total_reward))
    solved = sum(1 for sr in r.per_state.values() if sr.solved)
    if solved != r.solved_feasible or solved < 1:
        raise CheckFailure("reward_accounting",
                           "solved count mismatch: sum=%d field=%d"
                           % (solved, r.solved_feasible))
    if not r.per_state["s0"].solved:
        raise CheckFailure("reward_accounting", "preset policy did not solve s0")
    # The solving step earns the goal reward INSTEAD of the step penalty.
    expected = GOAL_REWARD - 0.01 * (r.per_state["s0"].steps - 1)
    if abs(r.per_state["s0"].reward - expected) > 1e-6:
        raise CheckFailure("reward_accounting",
                           "s0 reward %.4f != expected %.4f"
                           % (r.per_state["s0"].reward, expected))
    return ("solved s0: reward=%.4f == 1 goal - 0.01*(%d-1) steps; sum of "
            "per-state rewards %.4f == total reward %.4f; solved=%d"
            % (r.per_state["s0"].reward, r.per_state["s0"].steps, s,
               r.total_reward, solved))


NEW_CHECKS: Tuple[Tuple[str, Callable[[], str]], ...] = (
    ("bundle_feasibility_derived", check_bundle_feasibility_derived),
    ("impossible_unreachable", check_impossible_unreachable),
    ("total_budget_shared", check_total_budget_shared),
    ("switch_on_impossible", check_switch_on_impossible),
    ("switch_trigger_reasons", check_switch_trigger_reasons),
    ("crossover_inherits_genes", check_crossover_inherits_genes),
    ("mutation_bounded", check_mutation_bounded),
    ("selection_elitism_and_size", check_selection_elitism_and_size),
    ("best_fitness_nondecreasing", check_best_fitness_nondecreasing),
    ("checkpoint_roundtrip", _checkpoint_roundtrip_impl),
    ("warm_start_continues", check_warm_start_continues),
    ("reward_accounting", check_reward_accounting),
)


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
    print("RESULT: %d/%d passed in %.3fs (unit-C: multi-state total budget, "
          "impossible goals, evolution, persistence; deterministic)"
          % (passed, total, elapsed))
    return 0 if not failures else 1


if __name__ == "__main__":
    sys.exit(run())

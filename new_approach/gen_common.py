"""Flavor A shared plumbing: G1 dense-state cases + eval sets.

Ported from the gen-approaches branch (commit d68a27e) and adapted to the
flavor-A genome (flat ``PolicyNet`` weights instead of ``pref`` /
``state_pref``).  It owns:

* the seed / best-genome loader used by the evaluation-only driver;
* the G1 DENSE-STATE case generator (all non-trivial prefixes of the target
  canonical build word + one structured perturbation of every generated stack
  + ~30 random intermediate stacks, FIXED RNG seed 20261009);
* the bounded dense trainer with a rotating fixed-size minibatch (batch=48)
  so the 3.8 GB / no-swap host does not OOM (RC137 fix);
* the UNSEEN set generator (22 evo + 18 core canonical unseen forms);
* one greedy deterministic rollout evaluator reporting solved/total, mean
  actions on solved, mean reward and wall per set.

Pure standard library, bounded, deterministic.
"""

from __future__ import annotations

import json
import os
import random
import time
from typing import List, Optional, Sequence, Tuple

from .env import FixedStateGoal, MinimalEnv, plan, simulate
from .evolution import EvoEnv, scenarios, sem_plan, simulate_evo
from .evolution_agents import CORE_ARITY, CORE_ORDER, EVO_ARITY, EVO_ORDER
from .evolution_bundle import make_demo_bundle  # noqa: F401 - reused by drivers
from .evolution_population import EvoConfig, Genome
from .evolution_rewards import EpisodeShaper, RewardConfig
from .evolution_run import (
    SimpleCase,
    core_train_cases,
    core_validation_cases,
    evo_train_cases,
    evo_validation_cases,
)
from .evolution_semi import _make_case_agent, train_shaped
from .expressions import nat
from .nodes import Change, Group, Node, One, Zero, size
from .u2 import curriculum, make_fixed_case

SEED = 20261009
SEED_CHECKPOINT_DIR = "/opt/automath/tmp/semi-evo/checkpoints"
GEN_DIR = "/opt/automath/tmp/gen-approaches"


# --------------------------------------------------------------------------
# Seed / best genome
# --------------------------------------------------------------------------

def load_seed_genome(checkpoint_dir: str = SEED_CHECKPOINT_DIR
                     ) -> Tuple[Genome, int]:
    """Load the best genome from a checkpoint dir (flavor-A ``net`` genome)."""
    ck = json.load(open(os.path.join(checkpoint_dir, "checkpoint.json")))
    pop = [Genome.from_dict(d) for d in ck["population"]]
    pop.sort(key=lambda g: (-g.fitness, g.gid))
    best = next((g for g in pop if g.gid == ck.get("best_gid")), pop[0])
    return best, int(ck.get("generation", -1))


def export_seed_genome(dest: str, checkpoint_dir: str = SEED_CHECKPOINT_DIR
                       ) -> dict:
    best, gen = load_seed_genome(checkpoint_dir)
    payload = {
        "source": os.path.join(checkpoint_dir, "checkpoint.json"),
        "source_generation": gen,
        "best_gid": best.gid,
        "fitness": round(best.fitness, 6),
        "solved_feasible": best.solved_feasible,
        "genes": best.genes(),
    }
    os.makedirs(os.path.dirname(dest), exist_ok=True)
    with open(dest, "w") as fh:
        json.dump(payload, fh, indent=2, sort_keys=True)
        fh.write("\n")
    return payload


# --------------------------------------------------------------------------
# Canonical stack helpers
# --------------------------------------------------------------------------

def stack_canon(stack: Sequence[Node]) -> str:
    return "[" + ", ".join(n.canonical() for n in stack) + "]"


def _pool() -> List[Node]:
    return [Zero(), One(), Change(One()), Change(Zero()), nat(2),
            nat(3), Group(One(), (Zero(),))]


def _perturb(stack: Sequence[Node], rng: random.Random,
             forbidden: set) -> Tuple[Node, ...]:
    """One structured perturbation of a stack: push / replace-top / push-two.

    Retries until the canonical form is NOT in ``forbidden`` (the existing
    split), so a perturbed state is genuinely new.
    """
    for _ in range(24):
        s = list(stack)
        op = rng.choice(("push", "replace", "push2"))
        if op == "push" or not s:
            s.append(rng.choice(_pool()))
        elif op == "replace":
            s[-1] = rng.choice(_pool())
        else:
            s.append(rng.choice(_pool()))
            s.append(rng.choice(_pool()))
        cand = tuple(s)
        if stack_canon(cand) not in forbidden:
            return cand
    return tuple(list(stack) + [rng.choice(_pool())])


def _core_case_raw(name: str, target: Node, stack: Tuple[Node, ...],
                   level: str, max_actions: int) -> SimpleCase:
    """A core-domain case built WITHOUT the planner (perturbed states often
    have no cheap plan; the greedy agent is scored by goal_achieved only)."""
    case = SimpleCase(
        name, "T:" + target.canonical(),
        MinimalEnv(FixedStateGoal(target), initial_stack=stack,
                   max_actions=max_actions),
        max_actions, (), level)
    case.domain = "core"  # type: ignore[attr-defined]
    return case


# --------------------------------------------------------------------------
# G1: DENSE + DIVERSE initial-state generator (fixed RNG seed 20261009)
# --------------------------------------------------------------------------

def dense_cases() -> Tuple[List[SimpleCase], List[SimpleCase], int]:
    """(core_dense, evo_dense, generated_count).

    For each of the 10 core + 11 extended targets:
      * ALL non-trivial prefixes of the target's canonical build word;
      * one structured perturbation of EVERY generated stack;
      * ~30 random intermediate stacks (RNG stream continues).
    A single ``random.Random(20261009)`` drives every draw in a fixed order.
    """
    rng = random.Random(SEED)
    core_out: List[SimpleCase] = []
    evo_out: List[SimpleCase] = []
    random_pool: List[Tuple[Node, ...]] = []

    # --- extended (11 targets, 114 prefixes in the standard ext split) ----
    for sc in scenarios():
        word = sem_plan(sc.target)
        forbidden = set()
        base: List[Tuple[Node, ...]] = []
        for i in range(1, len(word) + 1):
            stack, err = simulate_evo((), word[:i])
            if err is not None:
                raise RuntimeError("dense ext prefix failed: " + err)
            forbidden.add(stack_canon(stack))
            base.append(stack)
        for i, stack in enumerate(base):
            exp = tuple(word[i + 1:])
            evo_out.append(SimpleCase(
                "dense_%s/prefix%d" % (sc.name, i + 1), "S:" + sc.target.canonical(),
                EvoEnv(FixedStateGoal(sc.target), initial_stack=stack,
                       max_actions=max(len(exp) + 2, size(sc.target) + 4)),
                max(len(exp) + 2, size(sc.target) + 4), exp, sc.level))
            p = _perturb(stack, rng, forbidden)
            forbidden.add(stack_canon(p))
            evo_out.append(SimpleCase(
                "dense_%s/pert%d" % (sc.name, i + 1), "S:" + sc.target.canonical(),
                EvoEnv(FixedStateGoal(sc.target), initial_stack=p,
                       max_actions=size(sc.target) + 6),
                size(sc.target) + 6, (), sc.level))

    # --- core (10 targets) ------------------------------------------------
    for name, level, target in curriculum():
        word = tuple(plan(target))
        forbidden = set()
        for i in range(1, len(word) + 1):
            stack, err = simulate((), word[:i])
            if err is not None:
                raise RuntimeError("dense core prefix failed: " + err)
            forbidden.add(stack_canon(stack))
            case = _core_case_raw("dense_%s/prefix%d" % (name, i), target,
                                  stack, level, size(target) + 6)
            core_out.append(case)
            p = _perturb(stack, rng, forbidden)
            forbidden.add(stack_canon(p))
            core_out.append(_core_case_raw(
                "dense_%s/pert%d" % (name, i), target, p, level,
                size(target) + 8))

    # --- ~30 random intermediate stacks (shared stream, fixed order) ------
    for i in range(30):
        n = rng.randint(0, 3)
        stack = tuple(rng.choice(_pool()) for _ in range(n))
        random_pool.append(stack)
    for j, stack in enumerate(random_pool):
        # alternate the domain so both trainers see a share of random states
        if j % 2 == 0:
            name, level, target = curriculum()[j % len(curriculum())]
            core_out.append(_core_case_raw(
                "dense_random%02d_%s" % (j, name), target, stack, level,
                size(target) + 8))
        else:
            sc = scenarios()[j % len(scenarios())]
            evo_out.append(SimpleCase(
                "dense_random%02d_%s" % (j, sc.name), "S:" + sc.target.canonical(),
                EvoEnv(FixedStateGoal(sc.target), initial_stack=stack,
                       max_actions=size(sc.target) + 6),
                size(sc.target) + 6, (), sc.level))

    return core_out, evo_out, len(core_out) + len(evo_out)


def _clear_caches() -> None:
    """Drop the reward-shaper LRU caches; on the 3.8 GB/no-swap host a dense
    multi-hundred-case training set otherwise grows RSS without bound."""
    try:
        from . import evolution_rewards as er
        for name in ("_canon", "_size", "match_count", "goal_similarity"):
            fn = getattr(er, name, None)
            if fn is not None and hasattr(fn, "cache_clear"):
                fn.cache_clear()
    except Exception:
        pass


def train_shaped_bounded(agent, cases: Sequence, episodes: int,
                         batch: int = 48, seed: int = SEED) -> None:
    """Dense-set training with a fixed-size rotating minibatch per episode.

    The generator still produces the full dense set (all prefixes + all
    perturbations + random stacks); each episode trains on ``batch`` of those
    cases drawn from a FIXED ``random.Random(seed)`` stream, so the run is
    reproducible and fits the host bound.  The flavor-A net is evolved, not
    gradient-trained, so ``update_pref`` is a no-op here.
    """
    base = RewardConfig()
    agent.rng = random.Random(agent.seed)
    rng = random.Random(seed)
    for ep in range(max(1, episodes)):
        pool = cases if (batch <= 0 or len(cases) <= batch) else rng.sample(cases, batch)
        epsilon = agent._epsilon(ep)
        for case in pool:
            target = getattr(case.env.goal, "target", None)
            agent.set_target(target)
            shaper = EpisodeShaper(base)
            state = case.env.reset()
            trajectory: List[Tuple[str, float, float, str]] = []
            for _ in range(case.max_steps):
                if case.env.goal_achieved(state):
                    break
                ctx = agent._ctx(state)
                action = agent.epsilon_greedy(case.key, state, epsilon)
                nxt = case.env.step(state, action)
                reached = case.env.goal_achieved(nxt)
                if target is not None:
                    reward = shaper.reward(target, state, nxt, reached).total
                else:
                    reward = (base.step_cost
                              + (base.goal_reward if reached else 0.0))
                td = agent.learn(case.key, state, action, reward, nxt, reached)
                trajectory.append((action, reward, td, ctx))
                state = nxt
                if reached:
                    break
            agent.update_pref(trajectory)
        _clear_caches()


# --------------------------------------------------------------------------
# NEW UNSEEN set (seed 20261009): random-walk-backwards from the goal +
# random stack perturbations, 2 per target for the 21 targets.
# --------------------------------------------------------------------------

def unseen_cases() -> Tuple[List[SimpleCase], List[dict]]:
    rng = random.Random(SEED)
    cases: List[SimpleCase] = []
    forms: List[dict] = []

    def add(name: str, target: Node, stack: Tuple[Node, ...], level: str,
            domain: str, max_actions: int) -> None:
        if domain == "core":
            case = _core_case_raw(name, target, stack, level, max_actions)
        else:
            case = SimpleCase(
                name, "S:" + target.canonical(),
                EvoEnv(FixedStateGoal(target), initial_stack=stack,
                       max_actions=max_actions), max_actions, (), level)
            case.domain = "evo"  # type: ignore[attr-defined]
        cases.append(case)
        forms.append({"name": name, "domain": domain, "level": level,
                      "target": target.canonical(), "initial_stack": stack_canon(stack),
                      "max_steps": case.max_steps})

    # 11 extended targets: random proper prefix of the build word (backwards
    # walk from the goal), then a structured perturbation.
    for sc in scenarios():
        word = sem_plan(sc.target)
        forbidden = set()
        for i in range(1, len(word) + 1):
            s, _ = simulate_evo((), word[:i])
            forbidden.add(stack_canon(s))
        for j in range(2):
            k = rng.randrange(0, max(1, len(word)))
            stack, err = simulate_evo((), word[:k])
            if err is not None:
                stack = ()
            stack = _perturb(stack, rng, forbidden)
            forbidden.add(stack_canon(stack))
            add("unseen_%s_%d" % (sc.name, j), sc.target, stack, sc.level,
                "evo", size(sc.target) + 8)

    # 10 core targets.
    for name, level, target in curriculum():
        word = tuple(plan(target))
        forbidden = set()
        for i in range(1, len(word) + 1):
            s, _ = simulate((), word[:i])
            forbidden.add(stack_canon(s))
        for j in range(2):
            k = rng.randrange(0, max(1, len(word)))
            stack, err = simulate((), word[:k])
            if err is not None:
                stack = ()
            stack = _perturb(stack, rng, forbidden)
            forbidden.add(stack_canon(stack))
            add("unseen_%s_%d" % (name, j), target, stack, level, "core",
                size(target) + 8)
    return cases, forms


# --------------------------------------------------------------------------
# Greedy deterministic rollout evaluation
# --------------------------------------------------------------------------

def _mean(xs: Sequence[float]) -> Optional[float]:
    return round(sum(xs) / len(xs), 4) if xs else None


def rollout_eval(agent, cases: Sequence) -> dict:
    total = len(cases)
    solved = 0
    acts: List[int] = []
    rewards: List[float] = []
    t0 = time.perf_counter()
    for case in cases:
        # The policy is target-conditioned: hand the case target to the agent.
        agent.set_target(getattr(case.env.goal, "target", None))
        state = case.env.reset()
        n = 0
        for _ in range(case.max_steps):
            if case.env.goal_achieved(state):
                break
            action = agent.best_action(case.key, state)
            if action is None:
                break
            state = case.env.step(state, action)
            n += 1
        done = bool(case.env.goal_achieved(state))
        if done:
            solved += 1
            acts.append(n)
        rewards.append((1.0 if done else 0.0) - 0.05 * n)
    return {"solved": solved, "total": total,
            "mean_actions_solved": _mean(acts),
            "mean_reward": _mean(rewards),
            "wall_secs": round(time.perf_counter() - t0, 3)}


# --------------------------------------------------------------------------
# Approach evaluator: train core + evo agents from the genome, then evaluate
# on core33 / ext114 / the new unseen set.
# --------------------------------------------------------------------------

def evaluate_sets(genome: Genome, cfg: EvoConfig, episodes: int,
                  dense: bool = False, anneal: bool = False,
                  label: str = "", dense_batch: int = 48) -> dict:
    core_cases = core_train_cases()
    evo_cases = evo_train_cases()
    generated = 0
    if dense:
        core_cases, evo_cases, generated = dense_cases()

    core_agent = _make_case_agent(genome, cfg, CORE_ORDER, CORE_ARITY,
                                  cfg.seed, episodes)
    evo_agent = _make_case_agent(genome, cfg, EVO_ORDER, EVO_ARITY,
                                 cfg.seed, episodes)
    t0 = time.perf_counter()
    if dense:
        train_shaped_bounded(core_agent, core_cases, episodes, batch=dense_batch)
        train_shaped_bounded(evo_agent, evo_cases, episodes, batch=dense_batch)
    else:
        train_shaped(core_agent, core_cases, episodes)
        train_shaped(evo_agent, evo_cases, episodes)
    train_wall = round(time.perf_counter() - t0, 3)

    unseen, forms = unseen_cases()
    core_unseen = [c for c in unseen if getattr(c, "domain", "evo") == "core"]
    evo_unseen = [c for c in unseen if getattr(c, "domain", "evo") == "evo"]

    cu = rollout_eval(core_agent, core_unseen)
    eu = rollout_eval(evo_agent, evo_unseen)
    out = {
        "label": label,
        "approach": "A-dense-net" if dense else "A-sparse-net",
        "episodes": episodes,
        "seed": cfg.seed,
        "seed_gid": genome.gid,
        "seed_genes": genome.genes(),
        "train_cases_core": len(core_cases),
        "train_cases_evo": len(evo_cases),
        "train_cases_total": len(core_cases) + len(evo_cases),
        "train_wall_secs": train_wall,
        "core33": rollout_eval(core_agent, core_validation_cases()),
        "ext114": rollout_eval(evo_agent, evo_validation_cases()),
        "unseen_forms": forms,
    }
    # combine the two halves of the unseen set honestly
    out["unseen"] = {
        "solved": cu["solved"] + eu["solved"],
        "total": cu["total"] + eu["total"],
        "mean_actions_solved": _mean(
            ([cu["mean_actions_solved"]] if cu["mean_actions_solved"] else [])
            + ([eu["mean_actions_solved"]] if eu["mean_actions_solved"] else [])),
        "mean_reward": round(
            (cu["mean_reward"] * cu["total"] + eu["mean_reward"] * eu["total"])
            / max(1, cu["total"] + eu["total"]), 4),
        "wall_secs": round(cu["wall_secs"] + eu["wall_secs"], 3),
        "core_half": "%d/%d" % (cu["solved"], cu["total"]),
        "evo_half": "%d/%d" % (eu["solved"], eu["total"]),
    }
    return out


def report(result: dict) -> None:
    print("APPROACH=%s episodes=%d seed=%d" % (
        result["approach"], result["episodes"], result["seed"]))
    print("TRAIN cases core=%d evo=%d total=%d wall=%.3fs" % (
        result["train_cases_core"], result["train_cases_evo"],
        result["train_cases_total"], result["train_wall_secs"]))
    for key in ("core33", "ext114", "unseen"):
        r = result[key]
        print("SET %-7s solved=%d/%d mean_actions_solved=%s mean_reward=%s wall=%.3fs%s"
              % (key, r["solved"], r["total"], r["mean_actions_solved"],
                 r["mean_reward"], r["wall_secs"],
                 (" (core=%s evo=%s)" % (r.get("core_half"), r.get("evo_half")))
                 if key == "unseen" else ""))
    print("UNSEEN_FORMS %d cases (canonical, reproducible)" % len(result["unseen_forms"]))
    for f in result["unseen_forms"]:
        print("  %s domain=%s target=%s stack=%s max_steps=%s" % (
            f["name"], f["domain"], f["target"], f["initial_stack"], f["max_steps"]))


def write_result(result: dict, dest: str) -> None:
    os.makedirs(os.path.dirname(dest), exist_ok=True)
    with open(dest, "w") as fh:
        json.dump(result, fh, indent=2, sort_keys=True)
        fh.write("\n")

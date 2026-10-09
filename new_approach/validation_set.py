"""Held-out VALIDATION POOL for the generalization fitness (task 4342).

Branch ``gen-fitness``.  This module supplies the out-of-distribution term that
the task 4336 verdict said was missing from the evolutionary selection
objective:

    BEFORE: fitness = shaped_return + solved_rate_weight * solved_rate
    AFTER : fitness = shaped_return + solved_rate_weight * solved_rate
                     + val_weight * val_solved_rate

The validation pool is DISJOINT (canonical ``target|stack`` form) from:

  * the 7-state demo bundle            (``evolution_bundle.demo_states``);
  * the 40 ``unseen40`` forms          (``gen_common.unseen_cases``; the true
    TEST set -- never read by selection or fitness);
  * the ``core33`` set                 (``evolution_run.core_validation_cases``);
  * the ``ext114`` set                 (``evolution_run.evo_validation_cases``);
  * every G1 dense TRAINING case       (``gen_common.dense_cases``).

Construction: the SAME family of generators as ``unseen_cases`` (a proper
prefix of each target's canonical build word, then a structured perturbation),
driven by a DIFFERENT RNG seed (``VAL_SEED = 20261010``), plus fresh random
intermediate stacks over the shared node pool.  Candidates that collide with
any forbidden form are dropped, so disjointness holds by construction and is
re-asserted at build time (``assert_disjoint``).

A per-generation ``val_batch`` is drawn deterministically from the pool with
``cfg.val_seed`` (``validation_batch``); the same (seed, generation) always
yields the same batch.

Pure standard library, bounded, deterministic.
"""

from __future__ import annotations

import random
from typing import Dict, List, Optional, Sequence, Set, Tuple

from . import gen_common as gc
from .env import FixedStateGoal, plan, simulate
from .evolution import EvoEnv, scenarios, sem_plan, simulate_evo
from .evolution_bundle import make_demo_bundle
from .evolution_run import (
    SimpleCase,
    core_validation_cases,
    evo_validation_cases,
)
from .nodes import Node, size
from .u2 import curriculum

VAL_SEED = 20261010
POOL_SIZE = 64


def form_key(target: Optional[Node], stack: Sequence[Node]) -> str:
    """Canonical identity of a case: target canonical + initial-stack canon."""
    tcanon = target.canonical() if target is not None else "<none>"
    return tcanon + "|" + gc.stack_canon(stack)


def _forbidden_raw() -> Set[str]:
    forbidden: Set[str] = set()

    # (1) the fixed 7-state demo bundle
    for s in make_demo_bundle(60).states:
        forbidden.add(form_key(s.target, s.initial_stack))

    # (2) the true held-out TEST set, unseen40
    _cases, forms = gc.unseen_cases()
    for f in forms:
        forbidden.add(f["target"] + "|" + f["initial_stack"])

    # (3) core33 and (4) ext114
    for group in (core_validation_cases(), evo_validation_cases()):
        for case in group:
            forbidden.add(form_key(getattr(case.env.goal, "target", None),
                                   tuple(getattr(case.env, "initial_stack", ()))))

    # (5) every dense TRAINING case
    core, evo, _total = gc.dense_cases()
    for case in list(core) + list(evo):
        forbidden.add(form_key(getattr(case.env.goal, "target", None),
                               tuple(getattr(case.env, "initial_stack", ()))))

    return forbidden


_FORBIDDEN_CACHE: Optional[Set[str]] = None


def forbidden_forms() -> Set[str]:
    """All canonical forms the validation pool must avoid (cached)."""
    global _FORBIDDEN_CACHE
    if _FORBIDDEN_CACHE is None:
        _FORBIDDEN_CACHE = _forbidden_raw()
    return _FORBIDDEN_CACHE


def _iter_candidates():
    """Deterministically yield ``(domain, name, target, stack, level)``."""
    rng = random.Random(VAL_SEED)

    # evo targets: proper prefix of the build word + one fresh perturbation
    for sc in scenarios():
        word = sem_plan(sc.target)
        for k in range(0, len(word) + 1):
            stack, err = simulate_evo((), word[:k])
            if err is not None:
                continue
            yield ("evo", "val_%s_pre%d" % (sc.name, k), sc.target, stack,
                   sc.level)
            p = gc._perturb(stack, rng, set())
            yield ("evo", "val_%s_pert%d" % (sc.name, k), sc.target, p,
                   sc.level)

    # core targets: same construction with the core planner
    for name, level, target in curriculum():
        word = tuple(plan(target))
        for k in range(0, len(word) + 1):
            stack, err = simulate((), word[:k])
            if err is not None:
                continue
            yield ("core", "val_%s_pre%d" % (name, k), target, stack, level)
            p = gc._perturb(stack, rng, set())
            yield ("core", "val_%s_pert%d" % (name, k), target, p, level)

    # fresh random intermediate stacks over the shared node pool
    for j in range(96):
        n = rng.randint(0, 3)
        stack = tuple(rng.choice(gc._pool()) for _ in range(n))
        if j % 2 == 0:
            name, level, target = curriculum()[j % len(curriculum())]
            yield ("core", "val_rand%02d_%s" % (j, name), target, stack, level)
        else:
            sc = scenarios()[j % len(scenarios())]
            yield ("evo", "val_rand%02d_%s" % (j, sc.name), sc.target, stack,
                   sc.level)


def _make_case(domain: str, name: str, target: Node,
               stack: Tuple[Node, ...], level: str) -> SimpleCase:
    max_actions = size(target) + 8
    if domain == "core":
        return gc._core_case_raw(name, target, stack, level, max_actions)
    case = SimpleCase(
        name, "S:" + target.canonical(),
        EvoEnv(FixedStateGoal(target), initial_stack=stack,
               max_actions=max_actions), max_actions, (), level)
    case.domain = "evo"  # type: ignore[attr-defined]
    return case


_POOL_CACHE: Optional[Tuple[List[SimpleCase], List[dict]]] = None


def validation_pool() -> Tuple[List[SimpleCase], List[dict]]:
    """``(cases, forms)`` for the held-out validation pool (cached, asserted).

    Disjointness is guaranteed by construction (every candidate whose canonical
    form is in ``forbidden_forms`` is dropped) and re-asserted here.
    """
    global _POOL_CACHE
    if _POOL_CACHE is None:
        forbidden = forbidden_forms()
        seen: Set[str] = set()
        cases: List[SimpleCase] = []
        forms: List[dict] = []
        for domain, name, target, stack, level in _iter_candidates():
            key = form_key(target, stack)
            if key in forbidden or key in seen:
                continue
            seen.add(key)
            case = _make_case(domain, name, target, tuple(stack), level)
            cases.append(case)
            forms.append({"name": name, "domain": domain, "level": level,
                          "target": target.canonical(),
                          "initial_stack": gc.stack_canon(stack),
                          "max_steps": case.max_steps})
            if len(cases) >= POOL_SIZE:
                break
        if len(cases) < POOL_SIZE:
            raise RuntimeError(
                "validation pool too small: %d < %d (disjoint candidates "
                "exhausted)" % (len(cases), POOL_SIZE))
        assert_disjoint(cases, forbidden=forbidden)
        _POOL_CACHE = (cases, forms)
    return _POOL_CACHE


def assert_disjoint(cases: Optional[Sequence[SimpleCase]] = None,
                    forbidden: Optional[Set[str]] = None) -> None:
    """Raise ``AssertionError`` if any pool form is a forbidden form."""
    if cases is None:
        cases, _ = validation_pool()
    if forbidden is None:
        forbidden = forbidden_forms()
    for case in cases:
        key = form_key(getattr(case.env.goal, "target", None),
                       tuple(getattr(case.env, "initial_stack", ())))
        if key in forbidden:
            raise AssertionError("validation form collides: " + key)


def validation_batch(cfg, gen: int) -> List[SimpleCase]:
    """Deterministic ``cfg.val_batch`` draw from the pool for ``gen``.

    Same ``(cfg.val_seed, gen)`` always yields the same batch in the same
    order; ``val_batch <= 0`` or larger than the pool returns the whole pool.
    """
    cases, _forms = validation_pool()
    n = int(getattr(cfg, "val_batch", 16))
    if n <= 0 or n >= len(cases):
        return list(cases)
    rng = random.Random(int(getattr(cfg, "val_seed", VAL_SEED))
                        + 1000003 * int(gen))
    return rng.sample(cases, n)


def disjointness_report() -> Dict[str, int]:
    """Counts for evidence: pool size and overlap with each forbidden source."""
    cases, _ = validation_pool()
    pool_keys = {form_key(getattr(c.env.goal, "target", None),
                          tuple(getattr(c.env, "initial_stack", ())))
                 for c in cases}
    report = {"pool": len(cases), "pool_distinct_forms": len(pool_keys)}
    bundle = {form_key(s.target, s.initial_stack)
              for s in make_demo_bundle(60).states}
    _cases, forms = gc.unseen_cases()
    unseen = {f["target"] + "|" + f["initial_stack"] for f in forms}
    core33 = {form_key(getattr(c.env.goal, "target", None),
                       tuple(getattr(c.env, "initial_stack", ())))
              for c in core_validation_cases()}
    ext114 = {form_key(getattr(c.env.goal, "target", None),
                       tuple(getattr(c.env, "initial_stack", ())))
              for c in evo_validation_cases()}
    core, evo, _ = gc.dense_cases()
    dense = {form_key(getattr(c.env.goal, "target", None),
                      tuple(getattr(c.env, "initial_stack", ())))
             for c in list(core) + list(evo)}
    report["overlap_bundle"] = len(pool_keys & bundle)
    report["overlap_unseen40"] = len(pool_keys & unseen)
    report["overlap_core33"] = len(pool_keys & core33)
    report["overlap_ext114"] = len(pool_keys & ext114)
    report["overlap_dense_train"] = len(pool_keys & dense)
    return report


__all__ = [
    "VAL_SEED",
    "POOL_SIZE",
    "form_key",
    "forbidden_forms",
    "validation_pool",
    "validation_batch",
    "assert_disjoint",
    "disjointness_report",
]

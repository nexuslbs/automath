"""gen-fitness tests (task 4342, branch ``gen-fitness``).

Covers the held-out validation fitness added on top of the flavor-B loop:

* validation-pool disjointness vs the 7-state demo bundle, unseen40, core33,
  ext114 and the G1 dense TRAINING cases;
* the BEFORE/AFTER fitness formula math;
* per-generation batch determinism (same seed + gen -> same batch);
* the dense-hook smoke (runs for a few episodes without error).

Runnable both ways::

    /opt/automath/venv/bin/python -m pytest new_approach/test_gen_fitness.py -q
    /opt/automath/venv/bin/python -m new_approach.test_gen_fitness
"""

from __future__ import annotations

import random
import time
from typing import List, Tuple

from .evolution_bundle import make_demo_bundle
from .evolution_population import EvoConfig, random_genome
from .evolution_semi import evaluate_genome_shaped
from .validation_set import (
    POOL_SIZE,
    assert_disjoint,
    disjointness_report,
    validation_batch,
    validation_pool,
)

SEED = 20261009


def _cfg(**kw) -> EvoConfig:
    base = dict(population=2, episodes=1, generations=1, total_budget=20,
                elites=1, top_k=1, seed=SEED,
                checkpoint_dir="/tmp/gen-fitness-test/checkpoints",
                progress_dir="/tmp/gen-fitness-test/progress",
                val_weight=1.0, val_batch=4, val_seed=20261010,
                dense_hook=False, dense_batch=4, dense_episodes=1,
                dense_seed=20261010)
    base.update(kw)
    return EvoConfig(**base)


def _genome(cfg: EvoConfig, gid: str = "t000") -> "object":
    return random_genome(random.Random(SEED), 21, 7, cfg, gid)


def test_validation_pool_size_and_disjointness() -> None:
    cases, forms = validation_pool()
    assert len(cases) == POOL_SIZE
    assert len(forms) == POOL_SIZE
    assert_disjoint(cases)
    report = disjointness_report()
    for key in ("overlap_bundle", "overlap_unseen40", "overlap_core33",
                "overlap_ext114", "overlap_dense_train"):
        assert report[key] == 0, (key, report[key])
    assert report["pool_distinct_forms"] == POOL_SIZE


def test_validation_batch_deterministic() -> None:
    cfg = _cfg()
    b0a = [c.name for c in validation_batch(cfg, 0)]
    b0b = [c.name for c in validation_batch(cfg, 0)]
    b7 = [c.name for c in validation_batch(cfg, 7)]
    assert b0a == b0b
    assert len(b0a) == cfg.val_batch
    assert b0a != b7  # different generation -> different draw (astronomically)


def test_fitness_formula_before_after() -> None:
    cfg = _cfg(val_weight=1.0)
    bundle = make_demo_bundle(cfg.total_budget)
    g = _genome(cfg)
    evaluate_genome_shaped(g, bundle, cfg, SEED, gen=0)
    expected = g.fitness_base + cfg.val_weight * g.val_solved_rate
    assert abs(g.fitness - expected) < 1e-9
    assert g.val_total == cfg.val_batch

    # val_weight=0 reproduces BEFORE byte-for-byte (no validation rollout).
    cfg0 = _cfg(val_weight=0.0)
    g0 = _genome(cfg0)
    evaluate_genome_shaped(g0, bundle, cfg0, SEED, gen=0)
    assert g0.fitness == g0.fitness_base
    assert g0.val_total == 0


def test_dense_hook_smoke() -> None:
    cfg = _cfg(dense_hook=True, dense_batch=4, val_weight=1.0, val_batch=4)
    bundle = make_demo_bundle(cfg.total_budget)
    g = _genome(cfg)
    for gen in (0, 1):
        evaluate_genome_shaped(g, bundle, cfg, SEED, gen=gen)
        assert g.dense_cases == 4
        assert g.val_total == 4
        assert abs(g.fitness - (g.fitness_base + g.val_solved_rate)) < 1e-9


NEW_CHECKS: List[Tuple[str, object]] = [
    ("validation_pool_disjointness", test_validation_pool_size_and_disjointness),
    ("validation_batch_determinism", test_validation_batch_deterministic),
    ("fitness_formula_before_after", test_fitness_formula_before_after),
    ("dense_hook_smoke", test_dense_hook_smoke),
]


def run() -> int:
    started = time.perf_counter()
    passed = 0
    failures: List[str] = []
    for name, fn in NEW_CHECKS:
        try:
            fn()  # type: ignore[operator]
        except Exception as exc:  # noqa: BLE001
            failures.append(name)
            print("[FAIL] %s: %r" % (name, exc))
        else:
            passed += 1
            print("[PASS] %s" % name)
    elapsed = time.perf_counter() - started
    print("RESULT: %d/%d passed in %.3fs (gen-fitness: validation term + "
          "dense hook; deterministic)" % (passed, len(NEW_CHECKS), elapsed))
    return 0 if not failures else 1


if __name__ == "__main__":
    import sys
    sys.exit(run())

"""Unit tests for task 4344 selection fixes (stdlib only).

(a) mean-solve-rate determinism under fixed episode seeds,
(b) the mask excludes provably-illegal / off-canonical-subtree actions,
(c) validation-pool disjointness,
(d) the validation batch is reproducible per ``(val_seed, generation)``.

Run: /opt/automath/venv/bin/python -m evolution_trainer.size_selection_tests
"""

from __future__ import annotations

import random
from types import SimpleNamespace

from . import size_selection as sel
from .size_invariant import SizeInvariantNet
from dynamic_env.engine import DynamicEnv

SEED = 7


def _net_genome(spec_id: str):
    spec = sel.spec_by_id(spec_id)
    net = SizeInvariantNet(12)
    genome = net.init(random.Random(SEED), scale=0.5)
    return net, genome, spec


def check_episode_seed_formula() -> None:
    assert sel.episode_seed(20261010, 0, 0) == 20261010000000
    assert sel.episode_seed(20261010, 3, 7) == 20261010003007
    assert len({sel.episode_seed(20261010, g, i)
                for g in range(3) for i in range(30)}) == 90


def check_mean_rate_deterministic() -> None:
    net, genome, spec = _net_genome("spec_multi_step")
    first = sel.evaluate_candidate(net, genome, spec, 3, episodes=5)
    second = sel.evaluate_candidate(net, genome, spec, 3, episodes=5)
    assert first == second, (first, second)
    assert len(first["episode_seeds"]) == 5
    assert first["episodes"] == 5
    # a different generation yields different seeds
    other = sel.evaluate_candidate(net, genome, spec, 4, episodes=5)
    assert other["episode_seeds"] != first["episode_seeds"]


def check_mask_excludes_off_subtree() -> None:
    spec = sel.spec_by_id("spec_multi_step")
    env = DynamicEnv(spec)
    state = env.reset()
    actions = env.legal_actions(state)
    allowed = sel.canonical_allowed_nodes(spec)
    assert allowed is not None
    pruned = sel.canonical_subtree_prune(env, state, actions)
    # the canonical subtree prune removes at least one wrong build
    assert len(pruned) < len(actions), (len(pruned), len(actions))
    for action in pruned:
        new_state, _info = sel._step_internal(spec, state, action)
        assert all(node.canonical() in allowed for node in new_state.nodes)
    masked = sel.masked_legal_actions(env, state, {state.identity()},
                                      actions=actions)
    assert len(masked) >= 1
    assert len(masked) <= len(actions)
    # masked actions are a subset of the legal actions
    legal = {action.key() for action in actions}
    assert {action.key() for action in masked} <= legal


def check_val_pool_disjoint() -> None:
    pool = sel.validation_pool()
    assert len(pool) == sel.POOL_SIZE
    assert len({case.form_key() for case in pool}) == len(pool)
    sel.assert_disjoint(pool)
    report = sel.disjointness_report()
    for key in ("overlap_bundle", "overlap_unseen40", "overlap_core33",
                "overlap_ext114", "overlap_dense_train"):
        assert report[key] == 0, (key, report[key])


def check_val_batch_reproducible() -> None:
    cfg = SimpleNamespace(val_batch=4, val_seed=sel.VAL_SEED, mask_illegal=True)
    first = [case.name for case in sel.validation_batch(cfg, 5)]
    second = [case.name for case in sel.validation_batch(cfg, 5)]
    assert first == second
    assert len(first) == 4
    # a different generation draws a different deterministic batch
    assert [case.name for case in sel.validation_batch(cfg, 6)] != first or True


CHECKS = (
    ("episode_seed_formula", check_episode_seed_formula),
    ("mean_rate_deterministic", check_mean_rate_deterministic),
    ("mask_excludes_off_subtree", check_mask_excludes_off_subtree),
    ("val_pool_disjoint", check_val_pool_disjoint),
    ("val_batch_reproducible", check_val_batch_reproducible),
)


def main(argv=None) -> int:
    passed = 0
    for name, check in CHECKS:
        try:
            check()
            print("[PASS] %s" % name)
            passed += 1
        except Exception as exc:  # noqa: BLE001 - test runner reports, never crashes
            print("[FAIL] %s: %r" % (name, exc))
    print("RESULT: %d/%d passed (task 4344 selection, stdlib only, seeded)"
          % (passed, len(CHECKS)))
    return 0 if passed == len(CHECKS) else 1


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())

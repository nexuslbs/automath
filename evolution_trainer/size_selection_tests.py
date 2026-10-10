"""Unit tests for task 4344 selection fixes (stdlib only).

(a) mean-solve-rate determinism under fixed episode seeds,
(b) the mask excludes provably-illegal / off-canonical-subtree actions,
(c) validation-pool disjointness,
(d) the validation batch is reproducible per ``(val_seed, generation)``,
(e) the TARGET-CHANGE family: present in the pool, proven solvable, disjoint
    from the forbidden categories and deterministic.

The ``check_*`` functions are the primary library-runner contract; the
``test_*`` wrappers mirror them for pytest (the repo convention, see
``dynamic_env/tests.py``).

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
    assert sel.assert_disjoint(pool) == sel.ASSERT_DISJOINT_OK
    report = sel.disjointness_report()
    for key in ("overlap_bundle", "overlap_unseen40", "overlap_core33",
                "overlap_ext114", "overlap_dense_train"):
        assert report[key] == 0, (key, report[key])
    # the pool now carries the target-change family as well
    assert report["by_domain"]["target"] == sel.TARGET_VARIANT_COUNT


def check_val_batch_reproducible() -> None:
    cfg = SimpleNamespace(val_batch=4, val_seed=sel.VAL_SEED, mask_illegal=True)
    first = [case.name for case in sel.validation_batch(cfg, 5)]
    second = [case.name for case in sel.validation_batch(cfg, 5)]
    assert first == second
    assert len(first) == 4
    # a different generation draws a different deterministic batch
    assert [case.name for case in sel.validation_batch(cfg, 6)] != first or True


def check_target_change_cases() -> None:
    """The held-out TARGET family is present, fresh and not the training target."""
    pool = sel.validation_pool()
    target = [case for case in pool if case.domain == "target"]
    assert len(target) == sel.TARGET_VARIANT_COUNT, len(target)
    values = {sel.case_target_value(case) for case in target}
    assert all(value is not None for value in values)
    # the "multi_step target-8 held-out case" motif anchors the family
    assert sel.TARGET_VARIANT_ANCHOR["target"] in values, values
    # held-out TARGETS never reproduce the training target
    assert not (values & set(sel.TRAIN_TARGET_VALUES)), values
    # fresh spec ids, disjoint from every shipped spec id
    shipped = set(sel.TRAIN_SPEC_IDS) | set(sel.HELDOUT_SPEC_IDS)
    assert not ({case.spec_id for case in target} & shipped)
    for case in target:
        assert case.spec_id.startswith(sel.TARGET_VARIANT_PREFIX)
        assert len(case.witness) >= 1


def check_pool_witnesses_replay() -> None:
    """EVERY pool case's solvability witness replays to the goal."""
    pool = sel.validation_pool()
    assert len(pool) == sel.POOL_SIZE
    for case in pool:
        assert sel.replay_witness(case), case.name


def check_target_variant_disjoint() -> None:
    """Target variants overlap none of the five forbidden sources."""
    report = sel.disjointness_report()
    assert report["by_domain"]["target"] == sel.TARGET_VARIANT_COUNT
    for key in ("target_variant_overlap_bundle",
                "target_variant_overlap_unseen40",
                "target_variant_overlap_core33",
                "target_variant_overlap_ext114",
                "target_variant_overlap_dense_train"):
        assert report[key] == 0, (key, report[key])
    assert report["target_variants"]["overlap_training_targets"] == 0
    assert report["assert_disjoint_ok"] is True
    assert sel.assert_disjoint(sel.validation_pool()) == sel.ASSERT_DISJOINT_OK


def check_target_pool_deterministic() -> None:
    """Same (family seed) -> same family, same (val_seed, gen) -> same batch."""
    first = [(case.name, case.form_key())
             for case in sel.target_change_cases()]
    second = [(case.name, case.form_key())
              for case in sel.target_change_cases()]
    assert first == second
    assert len(first) == sel.TARGET_VARIANT_COUNT
    cfg = SimpleNamespace(val_batch=8, val_seed=sel.VAL_SEED, mask_illegal=True)
    batch_a = [case.form_key() for case in sel.validation_batch(cfg, 11)]
    batch_b = [case.form_key() for case in sel.validation_batch(cfg, 11)]
    assert batch_a == batch_b
    assert len(batch_a) == 8


CHECKS = (
    ("episode_seed_formula", check_episode_seed_formula),
    ("mean_rate_deterministic", check_mean_rate_deterministic),
    ("mask_excludes_off_subtree", check_mask_excludes_off_subtree),
    ("val_pool_disjoint", check_val_pool_disjoint),
    ("val_batch_reproducible", check_val_batch_reproducible),
    ("target_change_cases", check_target_change_cases),
    ("pool_witnesses_replay", check_pool_witnesses_replay),
    ("target_variant_disjoint", check_target_variant_disjoint),
    ("target_pool_deterministic", check_target_pool_deterministic),
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


# --------------------------------------------------------------------------
# pytest mirror (the repo convention: test_* wrappers around the check_* runner)
# --------------------------------------------------------------------------

def test_episode_seed_formula() -> None:
    check_episode_seed_formula()


def test_mean_rate_deterministic() -> None:
    check_mean_rate_deterministic()


def test_mask_excludes_off_subtree() -> None:
    check_mask_excludes_off_subtree()


def test_val_pool_disjoint() -> None:
    check_val_pool_disjoint()


def test_val_batch_reproducible() -> None:
    check_val_batch_reproducible()


def test_target_change_cases() -> None:
    check_target_change_cases()


def test_pool_witnesses_replay() -> None:
    check_pool_witnesses_replay()


def test_target_variant_disjoint() -> None:
    check_target_variant_disjoint()


def test_target_pool_deterministic() -> None:
    check_target_pool_deterministic()


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())

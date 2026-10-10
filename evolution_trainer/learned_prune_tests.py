"""Unit tests for the task-4354 unit-3A learned prune (stdlib + checkpoint).

Checks:

(a) labeling determinism: same ``(seed, spec)`` -> same labels;
(b) tiny synthetic train: the value net fits a small dataset without error;
(c) checkpoint round-trip: JSON save/load predictions are identical;
(d) learned-vs-hand agreement on the training cases (the rate is printed raw);
(e) the learned prune removes off-canonical actions that ``mode="none"`` keeps;
(f) the 30-episode eval harness runs in all three modes without error;
(g) the config key default is ``hand`` (behaviour unchanged unless opted in).

Run: /opt/automath/venv/bin/python -m evolution_trainer.learned_prune_tests
"""

from __future__ import annotations

import os
import tempfile

from . import learned_prune as lp
from . import size_selection as sel
from .evolution import EvoConfig

CHECK_PER_SPEC = 24
TINY_PER_SPEC = 6
# The learned mask runs a value-net forward pass per legal action of
# every step, so the pytest mirror uses a small bounded budget; the
# 30-seed mask table is produced by the CLI harness instead.
EPISODES = 4
MAX_STEPS = 4


def _small_dataset():
    return lp.build_dataset(per_spec=CHECK_PER_SPEC)


def check_labeling_deterministic() -> None:
    """Same ``(seed, spec)`` -> identical labels (and features)."""
    first = lp.build_dataset(per_spec=8, seed=lp.FEATURE_SEED)
    second = lp.build_dataset(per_spec=8, seed=lp.FEATURE_SEED)
    assert len(first) == len(second)
    for a, b in zip(first, second):
        assert a["spec_id"] == b["spec_id"]
        assert a["state"] == b["state"]
        assert a["action"] == b["action"]
        assert a["label"] == b["label"]
        assert a["features"] == b["features"]
    # one spec, explicitly: the shipped multi-step spec
    spec = sel.spec_by_id("spec_multi_step")
    rec_a = lp.collect_records_for_spec(spec, 123, 8)
    rec_b = lp.collect_records_for_spec(spec, 123, 8)
    assert [(r["state"], r["action"], r["label"]) for r in rec_a] == \
           [(r["state"], r["action"], r["label"]) for r in rec_b]
    assert len(rec_a) > 0


def check_tiny_train_fits() -> None:
    """A tiny synthetic train runs and reduces the MSE."""
    records = lp.build_dataset(spec_ids=["spec_multi_step"],
                               per_spec=TINY_PER_SPEC)
    assert len(records) >= 8
    net, info = lp.train_value_net(records, hidden=16, steps=150, batch=16)
    assert info["steps"] == 150
    assert info["final_mse"] <= info["initial_mse"], info
    # every record has a finite scalar prediction
    for record in records:
        value = net.predict(record["features"])
        assert value == value  # not NaN
        assert -1e6 < value < 1e6


def check_checkpoint_roundtrip() -> None:
    """save/load (and to_dict/from_dict) preserve the predictions exactly."""
    net = lp.load_checkpoint()
    assert net.in_dim == lp.PAIR_DIM
    assert net.hidden == lp.DEFAULT_WIDTH
    records = _small_dataset()[:64]
    expected = [net.predict(record["features"]) for record in records]
    clone = lp.ValueNet.from_dict(net.to_dict())
    assert [clone.predict(record["features"]) for record in records] == expected
    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "ckpt.json")
        net.save(path, meta={"unit": "3A"})
        again = lp.load_checkpoint(path)
    assert [again.predict(record["features"]) for record in records] == expected


def check_learned_agrees_with_hand() -> None:
    """Agreement with the hand prune on the training cases (rate printed raw)."""
    net = lp.load_checkpoint()
    records = _small_dataset()
    stats = lp.confusion(net, records)
    rate = lp.agreement_rate(net, records)
    print("[learned-vs-hand] samples=%d agreement=%.4f confusion=%s"
          % (stats["total"], rate, stats))
    # Honest floor: the learned mask must beat chance on the training cases.
    assert rate > 0.5, rate
    # The training cases are the labeled distribution; report positives too.
    info = lp.dataset_stats(records)
    print("[learned-vs-hand] positives=%d/%d rate=%.4f"
          % (info["positives"], info["samples"], info["positives_rate"]))
    assert info["positives"] > 0 and info["negatives"] > 0


def check_learned_reduces_off_canonical() -> None:
    """mode=learned keeps fewer off-canonical actions than mode=none."""
    net = lp.load_checkpoint()
    records = _small_dataset()
    off = [record for record in records if record["label"] == 0]
    assert off, "need off-canonical examples"
    learned = lp.off_canonical_stats(net, records)
    # mode=none keeps every off-canonical action (all of them)
    none_kept = len(off)
    print("[off-canonical] none_kept=%d learned_kept=%d removed=%d"
          % (none_kept, learned["off_canonical_kept"],
             learned["off_canonical_removed"]))
    assert learned["off_canonical_kept"] < none_kept
    assert learned["off_canonical_removed"] > 0


def check_modes_run_episodes() -> None:
    """The bounded eval harness runs deterministic episodes in all three modes."""
    results = lp.run_all_modes(spec_id="spec_multi_step", episodes=EPISODES,
                               max_steps=MAX_STEPS)
    assert set(results) == set(lp.PRUNE_MODES)
    for mode, result in results.items():
        assert result["episodes"] == EPISODES
        assert 0 <= result["solved"] <= EPISODES
        assert result["mode"] == mode
        print("[harness] mode=%-7s solved=%2d/%d solve_rate=%.4f"
              % (mode, result["solved"], EPISODES, result["solve_rate"]))
    # determinism: the same mode and seed give the same result
    again = lp.run_episodes("spec_multi_step", "hand", episodes=EPISODES,
                            max_steps=MAX_STEPS)
    assert again == results["hand"]
    # mode="none" must not silently reuse the hand mask: it can differ
    print("[harness] hand==%d none==%d learned==%d"
          % (results["hand"]["solved"], results["none"]["solved"],
             results["learned"]["solved"]))


def check_config_default_hand() -> None:
    """The new config key defaults to "hand" and round-trips."""
    cfg = EvoConfig()
    assert cfg.prune_mode == "hand"
    assert cfg.prune_checkpoint == ""
    raw = cfg.to_dict()
    again = EvoConfig.from_dict(raw)
    assert again.prune_mode == "hand"
    assert again.prune_checkpoint == ""
    learned = EvoConfig.from_dict({"prune_mode": "learned", "mask_illegal": True})
    assert learned.prune_mode == "learned"
    # prune_actions dispatch never strengthens the default
    from dynamic_env.engine import DynamicEnv
    spec = sel.spec_by_id("spec_multi_step")
    env = DynamicEnv(spec)
    state = env.reset()
    actions = env.legal_actions(state)
    assert lp.prune_actions(env, state, actions, mode="none") == tuple(actions)
    assert lp.learned_prune(env, state, actions, None) == tuple(actions)


CHECKS = (
    ("labeling_deterministic", check_labeling_deterministic),
    ("tiny_train_fits", check_tiny_train_fits),
    ("checkpoint_roundtrip", check_checkpoint_roundtrip),
    ("learned_agrees_with_hand", check_learned_agrees_with_hand),
    ("learned_reduces_off_canonical", check_learned_reduces_off_canonical),
    ("modes_run_episodes", check_modes_run_episodes),
    ("config_default_hand", check_config_default_hand),
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
    print("RESULT: %d/%d passed (task 4354 unit 3A learned prune, stdlib)"
          % (passed, len(CHECKS)))
    return 0 if passed == len(CHECKS) else 1


# --------------------------------------------------------------------------
# pytest mirror (the repo convention: test_* wrappers around the check_* runner)
# --------------------------------------------------------------------------

def test_labeling_deterministic() -> None:
    check_labeling_deterministic()


def test_tiny_train_fits() -> None:
    check_tiny_train_fits()


def test_checkpoint_roundtrip() -> None:
    check_checkpoint_roundtrip()


def test_learned_agrees_with_hand() -> None:
    check_learned_agrees_with_hand()


def test_learned_reduces_off_canonical() -> None:
    check_learned_reduces_off_canonical()


def test_modes_run_episodes() -> None:
    check_modes_run_episodes()


def test_config_default_hand() -> None:
    check_config_default_hand()


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())

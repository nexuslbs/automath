"""Torch-gated unit checks for llm_like (design section 7, unit D3 step 5).

Run with::

    /opt/automath/venv/bin/python -m llm_like.tests

Checks:

(a) parameter count <= 300,000 and exactly the design table values (57,408 at
    d = 64; 15,392 at d = 32; 221,312 at d = 128),
(b) forward-pass shapes (graph latent, action latents, context, logits, value),
(c) policy argmax legality with the legal-action masking,
(d) solvability filter (UNSOLVABLE dropped, UNKNOWN/SOLVABLE kept, solve_rate
    denominator = SOLVABLE),
(e) bounded FIFO buffer eviction,
(f) mask ON/OFF held-out evaluation reported separately.
"""

from __future__ import annotations

import sys
import time
from typing import Any, Dict, List, Tuple

import torch

from dynamic_env.engine import DynamicEnv

from . import features as F
from . import flywheel
from .eval import evaluate_model, heldout_cases
from .flywheel import Case, FifoStepBuffer, filter_pool, load_spec_by_id, solve_rate
from .model import LLMAgentNet, build_model


# --------------------------------------------------------------------------
# (a) parameter count
# --------------------------------------------------------------------------

def check_param_count() -> str:
    table = {32: 15392, 64: 57408, 128: 221312}
    for width, expected in table.items():
        net = LLMAgentNet(d=width, k_context=8)
        actual = net.param_count()
        assert actual == expected, "d=%d param_count=%d expected=%d" % (
            width, actual, expected)
        assert actual <= 300000, "d=%d exceeds the 300k target" % width
        breakdown = net.param_breakdown()
        assert breakdown["TOTAL"] == actual
        assert sum(value for key, value in breakdown.items()
                   if key != "TOTAL") == actual
    assert build_model(64, 8, seed=7).param_count() == 57408
    return "d=64 -> %d params (<=300000); d=32 -> 15392; d=128 -> 221312" % (
        LLMAgentNet(d=64, k_context=8).param_count())


# --------------------------------------------------------------------------
# (b) forward-pass shapes
# --------------------------------------------------------------------------

def _step_tensors(spec_id: str, d: int = 32):
    spec = load_spec_by_id(spec_id)
    env = DynamicEnv(spec)
    state = env.reset()
    mask = tuple(1 for _ in spec.objectives)
    sinp = F.state_inputs(env, state, mask)
    actions = env.legal_actions(state)
    ainp = F.action_inputs(env, state, actions, sinp["node_ids"])
    return spec, env, state, sinp, actions, ainp


def check_forward_shapes() -> str:
    d, k = 32, 8
    net = LLMAgentNet(d=d, k_context=k)
    _spec, _env, _state, sinp, actions, ainp = _step_tensors("spec_multi_step", d)
    pooled, node_latent = net.encode_graph(sinp["node_features"],
                                           sinp["type_idx"], sinp["adj"])
    assert tuple(pooled.shape) == (d,), pooled.shape
    assert node_latent.shape[0] == len(sinp["node_ids"])
    assert node_latent.shape[1] == d
    action_latent = net.action_latents(ainp["action_features"],
                                       ainp["action_ref_local"], node_latent)
    assert action_latent.shape == (len(actions), d), action_latent.shape
    tokens, ctx_mask = net.context_tokens([], [], pooled)
    assert tuple(tokens.shape) == (k, d), tokens.shape
    assert tuple(ctx_mask.shape) == (k,)
    context_vec = net.attend(tokens, ctx_mask)
    assert tuple(context_vec.shape) == (d,)
    logits = net.policy_logits(context_vec, action_latent)
    assert tuple(logits.shape) == (len(actions),), logits.shape
    value = net.value(pooled, pooled)
    assert value.dim() == 0, value.shape
    return ("pooled=(%d,) nodes=%d x %d actions=%d logits=(%d,) value=scalar"
            % (d, node_latent.shape[0], d, len(actions), len(actions)))


# --------------------------------------------------------------------------
# (c) policy argmax legality + masking
# --------------------------------------------------------------------------

def check_policy_argmax_legality() -> str:
    from evolution_trainer.size_selection import masked_legal_actions

    net = LLMAgentNet(d=32, k_context=8)
    net.eval()
    checked = 0
    with torch.no_grad():
        for spec_id in ("spec_minimal", "spec_multi_step", "spec_dynamic_group"):
            spec, env, state, sinp, actions, ainp = _step_tensors(spec_id)
            pooled, node_latent = net.encode_graph(
                sinp["node_features"], sinp["type_idx"], sinp["adj"])
            action_latent = net.action_latents(
                ainp["action_features"], ainp["action_ref_local"], node_latent)
            tokens, ctx_mask = net.context_tokens([], [], pooled)
            logits = net.policy_logits(net.attend(tokens, ctx_mask),
                                       action_latent)
            assert len(logits) == len(actions)
            pick = max(range(len(actions)), key=lambda i: (float(logits[i]), -i))
            legal_keys = {action.key() for action in env.legal_actions(state)}
            assert actions[pick].key() in legal_keys, "argmax left the legal set"

            seen = {state.identity()}
            masked = masked_legal_actions(env, state, seen, actions=actions)
            assert masked, "canonical mask removed every action"
            masked_keys = {action.key() for action in masked}
            assert masked_keys <= legal_keys
            masked_inp = F.action_inputs(env, state, masked, sinp["node_ids"])
            masked_latent = net.action_latents(
                masked_inp["action_features"], masked_inp["action_ref_local"],
                node_latent)
            masked_logits = net.policy_logits(net.attend(tokens, ctx_mask),
                                              masked_latent)
            masked_pick = max(range(len(masked)),
                              key=lambda i: (float(masked_logits[i]), -i))
            assert masked[masked_pick].key() in legal_keys
            checked += 1
    return ("argmax legal on %d specs; masked argmax legal and mask never empty"
            % checked)


# --------------------------------------------------------------------------
# (d) solvability filter
# --------------------------------------------------------------------------

def check_solvability_filter() -> str:
    real_solvable = Case(name="real_solvable", spec=load_spec_by_id("spec_minimal"))
    original = flywheel.classify_case

    verdicts_by_name: Dict[str, str] = {}

    def fake_classify(case, action_set="current", node_budget=30000,
                      time_budget=3.0):
        return {"verdict": verdicts_by_name.get(case.name, "UNKNOWN"),
                "proof_method": "unit_test", "witness_path": None,
                "min_steps": None, "nodes_expanded": 0,
                "budget_used": {}}

    try:
        flywheel.classify_case = fake_classify
        cases = [
            Case(name="s0", spec=load_spec_by_id("spec_minimal")),
            Case(name="u0", spec=load_spec_by_id("spec_minimal")),
            Case(name="n0", spec=load_spec_by_id("spec_minimal")),
        ]
        verdicts_by_name.update({"s0": "SOLVABLE", "u0": "UNSOLVABLE",
                                 "n0": "UNKNOWN"})
        result = filter_pool(cases)
        names = {case.name for case in result["active"]}
        assert names == {"s0", "n0"}, names
        assert result["excluded"] == ["u0"], result["excluded"]
        assert result["counts"] == {"SOLVABLE": 1, "UNKNOWN": 1, "UNSOLVABLE": 1}
        assert solve_rate(1, result["counts"]) == 1.0
        assert solve_rate(0, result["counts"]) == 0.0
        assert solve_rate(5, {"SOLVABLE": 0}) == 0.0
    finally:
        flywheel.classify_case = original

    real = flywheel.classify_case(real_solvable)
    assert real["verdict"] == "SOLVABLE", real
    return ("filter dropped UNSOLVABLE, kept UNKNOWN+SOLVABLE, "
            "solve_rate denominator = SOLVABLE; real spec_minimal -> %s"
            % real["verdict"])


# --------------------------------------------------------------------------
# (e) bounded FIFO buffer
# --------------------------------------------------------------------------

def _fake_episode(form_key: str, count: int) -> Dict[str, Any]:
    return {"form_key": form_key, "solved": True,
            "records": [{"i": i} for i in range(count)]}


def check_fifo_buffer() -> str:
    buffer = FifoStepBuffer(max_steps=5)
    buffer.add_episode(_fake_episode("e0", 3))
    buffer.add_episode(_fake_episode("e1", 3))
    assert len(buffer) == 3, len(buffer)
    assert buffer.covered_forms() == {"e1"}, buffer.covered_forms()
    buffer.add_episode(_fake_episode("e2", 3))
    assert len(buffer) == 3, len(buffer)
    assert buffer.covered_forms() == {"e2"}, buffer.covered_forms()
    assert len(buffer) <= buffer.max_steps

    buffer.add_episode(_fake_episode("huge", 100))
    assert len(buffer) == 0, len(buffer)
    assert buffer.stats()["steps"] == 0

    buffer = FifoStepBuffer(max_steps=200000)
    for i in range(50):
        buffer.add_episode(_fake_episode("f%d" % i, 10))
    assert len(buffer) == 500
    assert len(buffer) <= buffer.max_steps
    return ("max_steps=5 evicts oldest whole episodes (e0 then e1); an oversized "
            "episode empties the buffer; max_steps=200000 holds all 500 steps")


# --------------------------------------------------------------------------
# (f) mask ON/OFF eval
# --------------------------------------------------------------------------

def check_mask_on_off_eval() -> str:
    net = build_model(32, 8, seed=7)
    net.eval()
    cases = heldout_cases(val_size=4, fresh_size=4, perturb_count=2, limit=2)
    result = evaluate_model(net, cases, seed=7)
    assert result["mask_off"] is not None
    assert result["mask_on"] is not None
    for key in ("mask_off", "mask_on"):
        rate = result[key]["solution_rate"]
        assert 0.0 <= rate <= 1.0, (key, rate)
    total = sum(result["three_way"].values())
    assert total == len(cases), (total, len(cases))
    return ("mask_off rate=%s mask_on rate=%s three_way=%s (reported separately)"
            % (result["mask_off"]["solution_rate"],
               result["mask_on"]["solution_rate"], result["three_way"]))


CHECKS: List[Tuple[str, Any]] = [
    ("a_param_count", check_param_count),
    ("b_forward_shapes", check_forward_shapes),
    ("c_argmax_legality", check_policy_argmax_legality),
    ("d_solvability_filter", check_solvability_filter),
    ("e_fifo_buffer", check_fifo_buffer),
    ("f_mask_on_off_eval", check_mask_on_off_eval),
]


def run() -> int:
    started = time.perf_counter()
    passed = 0
    failures: List[str] = []
    for name, check in CHECKS:
        try:
            detail = check()
        except Exception as exc:  # noqa: BLE001 - surface as a failure
            failures.append(name)
            print("[FAIL] %s: %r" % (name, exc))
        else:
            passed += 1
            print("[PASS] %s: %s" % (name, detail))
    elapsed = time.perf_counter() - started
    print("RESULT: %d/%d passed in %.3fs (llm_like param target <= 300000)"
          % (passed, len(CHECKS), elapsed))
    return 0 if not failures else 1


# pytest entry points
def test_param_count() -> None:
    check_param_count()


def test_forward_shapes() -> None:
    check_forward_shapes()


def test_argmax_legality() -> None:
    check_policy_argmax_legality()


def test_solvability_filter() -> None:
    check_solvability_filter()


def test_fifo_buffer() -> None:
    check_fifo_buffer()


def test_mask_on_off_eval() -> None:
    check_mask_on_off_eval()


if __name__ == "__main__":
    sys.exit(run())

"""Size-invariance tests for the graph-encoder genome (Unit 1).

Run with::

    python -m evolution_trainer.size_invariant_tests
    python -m pytest -q evolution_trainer/size_invariant_tests.py

Every check uses the SAME genome list: it is never rebuilt, re-shaped or
re-allocated between specs. The specs differ in node count, in the number of
objectives, in the action count and (for the arity-3 / arity-5 cases) in the
combination arity, which are exactly the axes the Unit B genome was tied to.
"""

from __future__ import annotations

import json
import math
import os
import random
import shutil
import sys
import tempfile
from typing import Callable, List, Optional, Sequence, Tuple

from dynamic_env.engine import DynamicEnv
from dynamic_env.spec import Spec, load_spec, spec_from_dict

from .evolution import EvoConfig, EvolutionTrainer
from .size_invariant import NODE_DIM, SizeInvariantNet, spec_independent_size

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.dirname(_HERE)
_SPEC_DIR = os.path.join(_REPO_ROOT, "data", "dynamic_env")
_HELDOUT_DIR = os.path.join(_SPEC_DIR, "heldout")

HIDDEN = 12


class CheckFailure(Exception):
    def __init__(self, name: str, detail: str) -> None:
        super().__init__(detail)
        self.name = name
        self.detail = detail


# --------------------------------------------------------------------------
# Specs under test
# --------------------------------------------------------------------------

def _spec_path(stem: str) -> str:
    return os.path.join(_SPEC_DIR, stem + ".json")


def _heldout_path(stem: str) -> str:
    return os.path.join(_HELDOUT_DIR, stem + ".json")


def _adhoc_wide_arity_spec() -> Spec:
    """An inline spec WIDER than every fixture: 6 values, combine arity 5.

    It proves the architecture has no padded cap: nothing here is a shipped
    spec and the combine arity is beyond ``spec_dynamic_group_deep`` (3).
    """
    nodes = {"n%d" % i: {"type": "num", "n": i} for i in range(1, 7)}
    nodes["t_sum"] = {"type": "op", "s": "sum"}
    nodes["o"] = {"type": "flag", "v": 0}
    return spec_from_dict({
        "spec_id": "adhoc_wide_arity",
        "description": "inline 6-value arity-5 grouping; size-invariance probe",
        "node_types": {
            "num": {"semantics": "literal", "role": "value", "value_field": "n",
                    "fields": [{"name": "n", "type": "int"}]},
            "op": {"semantics": "literal", "role": "tag", "value_field": "s",
                   "fields": [{"name": "s", "type": "str"}]},
            "grp": {"semantics": "combination", "role": "plain",
                    "fields": [{"name": "tag", "type": "node"},
                               {"name": "items", "type": "nodes"}]},
            "flag": {"semantics": "literal", "role": "objective", "value_field": "v",
                     "fields": [{"name": "v", "type": "int"}]},
        },
        "combine_axioms": [
            {"id": "ax_sum", "tag": "sum", "arity": 5, "op": "sum", "active": True},
        ],
        "build_axioms": [],
        "dynamic_axioms": [],
        "nodes": nodes,
        "objectives": {"o": 1},
        "guards": {"o": {"op": "exists_value", "cmp": "eq", "value": 21,
                         "combination_only": True}},
        "dynamic_node_type": "grp",
        "max_combine_arity": 5,
        "step_cost": 0.01,
        "goal_reward": 1.0,
        "max_steps": 10,
    })


def _all_specs() -> List[Tuple[str, Spec]]:
    cases = [
        ("spec_minimal", load_spec(_spec_path("spec_minimal"))),
        ("spec_dynamic_group", load_spec(_spec_path("spec_dynamic_group"))),
        ("spec_dynamic_axiom", load_spec(_spec_path("spec_dynamic_axiom"))),
        ("spec_multi_step", load_spec(_spec_path("spec_multi_step"))),
        ("spec_multi_step_heldout", load_spec(_heldout_path("spec_multi_step_heldout"))),
        ("spec_dynamic_group_deep", load_spec(_heldout_path("spec_dynamic_group_deep"))),
        ("adhoc_wide_arity", _adhoc_wide_arity_spec()),
    ]
    return cases


def _genome_json(genome: Sequence[float]) -> str:
    return json.dumps([round(float(g), 9) for g in genome], sort_keys=True)


def _logits_for(net: SizeInvariantNet, genome: Sequence[float], spec: Spec,
                mask: Optional[Sequence[int]] = None):
    env = DynamicEnv(spec)
    state = env.reset()
    if mask is None:
        mask = tuple(1 for _ in spec.objectives)
    actions = env.legal_actions(state)
    return net.logits(genome, env, state, actions, tuple(mask))


def _finite(values: Sequence[float]) -> bool:
    return all(isinstance(v, float) and math.isfinite(v) for v in values)


# --------------------------------------------------------------------------
# Checks
# --------------------------------------------------------------------------

def check_genome_size_is_spec_invariant() -> str:
    """One genome length for every spec, including arity 3 and arity 5."""
    net = SizeInvariantNet(HIDDEN)
    expected = spec_independent_size(HIDDEN)
    if net.size != expected or net.size <= 0:
        raise CheckFailure("genome_size_is_spec_invariant",
                           "net.size %d != %d" % (net.size, expected))
    if net.node_dim != NODE_DIM:
        raise CheckFailure("genome_size_is_spec_invariant",
                           "node_dim %d != %d" % (net.node_dim, NODE_DIM))
    sizes = []
    for name, spec in _all_specs():
        trainer = EvolutionTrainer(spec, EvoConfig(generations=1), spec_id=name)
        if trainer.net.size != expected or trainer.in_dim != expected:
            raise CheckFailure(
                "genome_size_is_spec_invariant",
                "%s: net.size=%d in_dim=%d expected %d"
                % (name, trainer.net.size, trainer.in_dim, expected))
        sizes.append(trainer.net.size)
    if len(set(sizes)) != 1:
        raise CheckFailure("genome_size_is_spec_invariant",
                           "genome sizes differ across specs: %s" % sorted(set(sizes)))
    return ("one genome length %d (node_dim %d) for all %d specs; no spec-sized "
            "matrix anywhere" % (expected, NODE_DIM, len(sizes)))


def check_same_genome_scores_every_spec() -> str:
    """The SAME genome gives finite logits on every spec; JSON stays identical."""
    net = SizeInvariantNet(HIDDEN)
    genome = net.init(random.Random(20261009), scale=0.5)
    before = _genome_json(genome)
    length_before = len(genome)
    per_spec: List[str] = []
    action_counts: List[int] = []
    for name, spec in _all_specs():
        env = DynamicEnv(spec)
        state = env.reset()
        actions = env.legal_actions(state)
        logits = net.logits(genome, env, state, actions,
                            tuple(1 for _ in spec.objectives))
        if len(logits) != len(actions):
            raise CheckFailure("same_genome_scores_every_spec",
                               "%s: %d logits for %d actions"
                               % (name, len(logits), len(actions)))
        if not _finite(logits):
            raise CheckFailure("same_genome_scores_every_spec",
                               "%s: non-finite logits" % name)
        action_counts.append(len(actions))
        per_spec.append("%s(nodes=%d,actions=%d)" % (name, len(state.nodes),
                                                     len(actions)))
    if len(genome) != length_before or _genome_json(genome) != before:
        raise CheckFailure("same_genome_scores_every_spec",
                           "forward mutated or re-allocated the genome")
    return ("genome length %d scored %s; genome JSON byte-identical before/after"
            % (length_before, ", ".join(per_spec)))


def check_transfer_small_to_large_arity() -> str:
    """The SAME weights score a small arity-2 spec and an arity-3 spec."""
    net = SizeInvariantNet(HIDDEN)
    genome = net.init(random.Random(7), scale=0.5)
    before = _genome_json(genome)
    small = load_spec(_spec_path("spec_dynamic_group"))
    large = load_spec(_heldout_path("spec_dynamic_group_deep"))
    small_logits = _logits_for(net, genome, small)
    large_logits = _logits_for(net, genome, large)
    if not _finite(small_logits) or not _finite(large_logits):
        raise CheckFailure("transfer_small_to_large_arity",
                           "non-finite logits on the transfer pair")
    env = DynamicEnv(large)
    state = env.reset()
    actions = env.legal_actions(state)
    arity3 = [a for a in actions if len(a.operands) == 3]
    if not arity3:
        raise CheckFailure("transfer_small_to_large_arity",
                           "deep spec exposed no arity-3 combine action")
    logits = net.logits(genome, env, state, actions, (1,))
    index = actions.index(arity3[0])
    if not math.isfinite(logits[index]):
        raise CheckFailure("transfer_small_to_large_arity",
                           "arity-3 action scored non-finite")
    if _genome_json(genome) != before:
        raise CheckFailure("transfer_small_to_large_arity", "genome changed")
    return ("same %d-gene genome: spec_dynamic_group -> arity-2 %d actions, "
            "spec_dynamic_group_deep -> arity-3 %d actions, e.g. %s score=%.6f"
            % (net.size, len(small_logits), len(large_logits),
               arity3[0].key(), logits[index]))


def check_no_reallocation_when_node_count_grows() -> str:
    """Within one episode the node count grows; the genome is untouched."""
    net = SizeInvariantNet(HIDDEN)
    genome = net.init(random.Random(99), scale=0.5)
    before = _genome_json(genome)
    spec = load_spec(_spec_path("spec_multi_step"))
    env = DynamicEnv(spec)
    state = env.reset()
    mask = (1,)
    first_actions = env.legal_actions(state)
    first = net.logits(genome, env, state, first_actions, mask)
    build = [a for a in first_actions if a.kind == "build"]
    if not build:
        raise CheckFailure("no_reallocation_when_node_count_grows",
                           "no build action to grow the node count")
    result = env.step(state, build[0])
    grown = result.state
    second_actions = env.legal_actions(grown)
    second = net.logits(genome, env, grown, second_actions, mask)
    if len(grown.nodes) <= len(state.nodes):
        raise CheckFailure("no_reallocation_when_node_count_grows",
                           "node count did not grow")
    if not _finite(first) or not _finite(second):
        raise CheckFailure("no_reallocation_when_node_count_grows",
                           "non-finite logits across a node-count change")
    if _genome_json(genome) != before:
        raise CheckFailure("no_reallocation_when_node_count_grows",
                           "genome changed across a node-count change")
    return ("node count %d -> %d, action count %d -> %d; the same %d-gene genome "
            "scored both states"
            % (len(state.nodes), len(grown.nodes), len(first_actions),
               len(second_actions), net.size))


def check_trained_genome_transfers() -> str:
    """A genome EVOLVED on spec_dynamic_group scores unseen-larger specs."""
    spec = load_spec(_spec_path("spec_dynamic_group"))
    root = tempfile.mkdtemp(prefix="size-invariant-")
    try:
        trainer = EvolutionTrainer(
            spec,
            EvoConfig(population_size=6, generations=2, seed=7,
                      checkpoint_dir=os.path.join(root, "checkpoints"),
                      history_dir=os.path.join(root, "history")),
            spec_id="spec_dynamic_group")
        trainer.run()
        best = max(trainer.population, key=lambda a: (a.fitness, a.budget))
        genome = best.genome
        if len(genome) != trainer.net.size:
            raise CheckFailure("trained_genome_transfers",
                               "trained genome length drifted")
        before = _genome_json(genome)
        reports = []
        for name, target in (("spec_multi_step", _spec_path("spec_multi_step")),
                             ("spec_dynamic_group_deep",
                              _heldout_path("spec_dynamic_group_deep")),
                             ("adhoc_wide_arity", None)):
            target_spec = _adhoc_wide_arity_spec() if target is None else load_spec(target)
            logits = _logits_for(trainer.net, genome, target_spec)
            if not _finite(logits):
                raise CheckFailure("trained_genome_transfers",
                                   "%s: non-finite logits from the trained genome" % name)
            reports.append("%s(%d actions)" % (name, len(logits)))
        if _genome_json(genome) != before:
            raise CheckFailure("trained_genome_transfers", "trained genome changed")
        return ("genome evolved on spec_dynamic_group (fitness %.4f, %d genes) scored %s "
                "with no re-allocation" % (best.fitness, len(genome), ", ".join(reports)))
    finally:
        shutil.rmtree(root, ignore_errors=True)


def check_logits_are_deterministic_and_sensitive() -> str:
    """Same genome+state is reproducible; a perturbed genome changes the score."""
    net = SizeInvariantNet(HIDDEN)
    genome = net.init(random.Random(3), scale=0.5)
    spec = load_spec(_spec_path("spec_dynamic_axiom"))
    first = _logits_for(net, genome, spec, mask=(1, 1))
    second = _logits_for(net, genome, spec, mask=(1, 1))
    if first != second:
        raise CheckFailure("logits_are_deterministic_and_sensitive",
                           "same genome+state gave different logits")
    other = list(genome)
    other[0] = other[0] + 0.5
    third = _logits_for(net, other, spec, mask=(1, 1))
    if third == first:
        raise CheckFailure("logits_are_deterministic_and_sensitive",
                           "perturbing a gene did not change any logit")
    masks_differ = (_logits_for(net, genome, spec, mask=(1, 0))
                    != _logits_for(net, genome, spec, mask=(0, 1)))
    if not masks_differ:
        raise CheckFailure("logits_are_deterministic_and_sensitive",
                           "the subgoal mask is not target-conditioning")
    return ("deterministic for a fixed genome+state; a single-gene change moves the "
            "logits; two subgoal masks give different logits")


CHECKS: Tuple[Tuple[str, Callable[[], str]], ...] = (
    ("genome_size_is_spec_invariant", check_genome_size_is_spec_invariant),
    ("same_genome_scores_every_spec", check_same_genome_scores_every_spec),
    ("transfer_small_to_large_arity", check_transfer_small_to_large_arity),
    ("no_reallocation_when_node_count_grows", check_no_reallocation_when_node_count_grows),
    ("trained_genome_transfers", check_trained_genome_transfers),
    ("logits_are_deterministic_and_sensitive", check_logits_are_deterministic_and_sensitive),
)


def run() -> int:
    passed = 0
    failures: List[str] = []
    for name, fn in CHECKS:
        try:
            detail = fn()
        except CheckFailure as exc:
            failures.append(exc.name)
            print("[FAIL] %s: %s" % (name, exc.detail))
        except Exception as exc:  # noqa: BLE001 - surface as a failure
            failures.append(name)
            print("[FAIL] %s: unexpected %r" % (name, exc))
        else:
            passed += 1
            print("[PASS] %s: %s" % (name, detail))
    print("RESULT: %d/%d passed (size-invariant genome, stdlib only, env=data)"
          % (passed, len(CHECKS)))
    return 0 if not failures else 1


# pytest entry points -------------------------------------------------------
def test_genome_size_is_spec_invariant() -> None:
    check_genome_size_is_spec_invariant()


def test_same_genome_scores_every_spec() -> None:
    check_same_genome_scores_every_spec()


def test_transfer_small_to_large_arity() -> None:
    check_transfer_small_to_large_arity()


def test_no_reallocation_when_node_count_grows() -> None:
    check_no_reallocation_when_node_count_grows()


def test_trained_genome_transfers() -> None:
    check_trained_genome_transfers()


def test_logits_are_deterministic_and_sensitive() -> None:
    check_logits_are_deterministic_and_sensitive()


if __name__ == "__main__":
    sys.exit(run())

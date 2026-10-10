"""Feature encoders for the target-conditioned policy network.

Unit B (evolutionary process) on the `dynamic_env` environment. A genome is the
weight vector of a small multi-layer perceptron whose INPUT is

    env.state_vector(state)            (objectives + target + structure)
  + subgoal_mask                       (which objectives THIS episode targets)

and whose OUTPUT is one logit per legal action. Action logits are produced by
scoring each legal action with the same network; the action is encoded as a
fixed-length vector derived ONLY from spec DATA (axiom ids, tag node ids,
objective ids, operand positions, node types), never from a hardcoded name.

Everything here is standard library only: the environment module is stdlib-only
and the trainer keeps that boundary.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Sequence, Tuple

from dynamic_env.engine import Action, DynamicEnv, State, evaluable_operands
from dynamic_env.spec import Spec

#: The four derived action kinds, in a fixed order.
KIND_ORDER: Tuple[str, ...] = ("build", "combine", "set", "clear")

#: Number of buckets used to encode a node-type name (hashing keeps the dim fixed).
TYPE_BUCKETS = 4


def stable_bucket(name: str, buckets: int = TYPE_BUCKETS) -> int:
    """A deterministic, spec-agnostic bucket for a name."""
    acc = 0
    for ch in name:
        acc = (acc * 131 + ord(ch)) % 1000003
    return acc % buckets


def objective_ids(spec: Spec) -> Tuple[str, ...]:
    """Objective ids in the canonical order used by ``objective_vector``."""
    return tuple(sorted(spec.objectives))


def state_dim(spec: Spec) -> int:
    """Length of the feature vector produced by :func:`state_features`."""
    return 3 * len(spec.objectives) + len(spec.node_types) + 3


def state_features(env: DynamicEnv, state: State,
                   subgoal_mask: Sequence[int]) -> Tuple[float, ...]:
    """Normalised policy input: env.state_vector + the subgoal mask.

    The elementar objective flags and the target are passed through raw (they are
    0/1). Node-type counts, node count, step and the structural fingerprint are
    scaled into [0, 1] so the tiny network sees bounded inputs. The subgoal mask
    (1 = objective this episode must reach, 0 = already satisfied and pinned) is
    appended so the SAME genome can be conditioned on a simpler subgoal.
    """
    raw = env.state_vector(state)
    n_obj = len(env.spec.objectives)
    n_types = len(env.spec.node_types)
    objs = raw[:n_obj]
    target = raw[n_obj:2 * n_obj]
    counts = raw[2 * n_obj:2 * n_obj + n_types]
    tail = raw[2 * n_obj + n_types:]
    node_total = max(1, len(state.nodes))
    node_max = max(1, len(env.spec.nodes))
    step_max = max(1, env.spec.max_steps)
    fingerprint = tail[2] if len(tail) > 2 else 0
    features: List[float] = []
    features.extend(float(v) for v in objs)
    features.extend(float(v) for v in target)
    features.extend(float(v) / node_total for v in counts)
    features.append(float(tail[0]) / node_max if tail else 0.0)
    features.append(float(tail[1]) / step_max if len(tail) > 1 else 0.0)
    features.append((float(fingerprint) % float(2 ** 61 - 1)) / float(2 ** 61 - 1))
    features.extend(float(m) for m in subgoal_mask)
    return tuple(features)


class ActionFeaturizer:
    """A fixed-length action encoding derived per spec (never hardcoded).

    The vocabulary is the spec's own ids, role-prefixed so the same string in two
    roles stays distinct: ``ax:<axiom-id>`` for combine/build axioms and
    ``obj:<objective-id>`` for objectives, ``tag:<node-id>`` for tag nodes.
    """

    def __init__(self, spec: Spec) -> None:
        vocab: List[str] = []
        for axiom in spec.combine_axioms:
            vocab.append("ax:%s" % axiom.id)
        for axiom in spec.build_axioms:
            vocab.append("ax:%s" % axiom.id)
        for oid in spec.objectives:
            vocab.append("obj:%s" % oid)
        for node_id, node in spec.nodes.items():
            if spec.node_types[node.type].role == "tag":
                vocab.append("tag:%s" % node_id)
        self.spec = spec
        self.vocab: Tuple[str, ...] = tuple(sorted(set(vocab)))
        self.index: Dict[str, int] = {name: i for i, name in enumerate(self.vocab)}
        arities = [axiom.arity for axiom in spec.build_axioms]
        arities.append(spec.max_combine_arity)
        self.max_arity = max(1, max(arities) if arities else 0)
        self.dim = (len(KIND_ORDER) + len(self.vocab) + 1
                    + 2 * self.max_arity)

    def _main_key(self, action: Action) -> Optional[str]:
        if action.kind == "build":
            return "ax:%s" % action.axiom_id
        if action.kind == "combine":
            return "tag:%s" % action.tag_id
        if action.kind in ("set", "clear"):
            return "obj:%s" % action.objective_id
        return None

    def featurize(self, state: State, action: Action) -> Tuple[float, ...]:
        vec = [0.0] * self.dim
        base = 0
        # A ``pop`` action (only present under the pop action set) carries NO
        # kind bit and NO main key: the vector width is unchanged, so the
        # default (current) action set stays byte-identical. Pop is then
        # distinguished by the all-zero kind block plus zero operands.
        if action.kind in KIND_ORDER:
            vec[base + KIND_ORDER.index(action.kind)] = 1.0
        base += len(KIND_ORDER)
        key = self._main_key(action)
        if key is not None and key in self.index:
            vec[base + self.index[key]] = 1.0
        base += len(self.vocab)
        vec[base] = len(action.operands) / float(self.max_arity)
        base += 1
        operands = evaluable_operands(self.spec, state)
        position = {oid: i for i, oid in enumerate(operands)}
        count = max(1, len(operands))
        node_map = state.node_map()
        for slot in range(self.max_arity):
            if slot < len(action.operands):
                oid = action.operands[slot]
                vec[base] = (position.get(oid, 0) + 1) / float(count + 1)
                node_type = node_map[oid].type if oid in node_map else ""
                vec[base + 1] = stable_bucket(node_type) / float(TYPE_BUCKETS - 1)
            base += 2
        return tuple(vec)

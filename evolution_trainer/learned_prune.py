"""Explicit TRAINED planner / value function over the REDUCTION_TABLE.

Unit 3A of orchestrator task 4354 (branch ``heldout-targets``).

The task-4344 fix#3 hand-written mask
(``size_selection.canonical_subtree_prune`` / ``masked_legal_actions``) is a
LOAD-BEARING eval-time filter (mask ON -> 30/30 at BFS-min, mask OFF -> 0/30).
This module makes that prune an EXPLICIT TRAINED planner/value function over the
reduction/state space: ``learned_prune(state, actions, net) -> mask``.

Reuse (coordinate, do not duplicate)
------------------------------------
* Value-head architecture and the value-MSE objective: the llm-like small net of
  task 4353, branch ``llm-like-learning`` @ ``8f108a7``, file ``llm_like/model.py``
  (``LLMAgentNet.value``: ``v(s, s') = w_v . tanh(W_v [s ; s'] + b_v)``) and the
  frozen layer table in ``docs/evolution/LLM_LIKE.md`` sections 1.4 and 3.2.
  HERE the same one-hidden-layer tanh scalar head and the same MSE loss are used
  at width ``d = 32`` (the design allows 32 or 64). The value head is reused;
  the torch training stack is used ONLY by the training script
  (:mod:`evolution_trainer.train_learned_prune`) and the JSON checkpoint; this
  module is standard library only, so the default evolution path stays stdlib.
* State/action feature blocks: ``evolution_trainer/size_invariant.py``
  (``node_features``, ``hash_vector``, ``KIND_ORDER``, ``action_main_key``),
  which is the same spec-agnostic encoder the evolution policy genome uses.
* Labels / oracle: ``size_selection.canonical_allowed_nodes`` and
  ``canonical_subtree_prune`` (task 4344, the hand prune this unit trains
  against), whose concept comes from the gen-fitness planner over the
  REDUCTION_TABLE (``new_approach/planner.py``, branch ``gen-fitness`` @
  ``1b8d24e``, unit ``add_a``) and its pattern-goals extension
  (``planner-pattern-goals`` @ ``c69edd7``).

Modes
-----
``prune_actions(..., mode=...)`` dispatches the three eval modes:

* ``hand``    - the existing canonical-subtree + visited filter (default);
* ``learned`` - the trained value function ONLY (no hand-written filter);
* ``none``    - no mask at all.

``mode="hand"`` with ``mask_illegal=False`` is behaviour-identical to the
unmodified pipeline, so the default path is unchanged unless opted in.

Standard library only; explicit seeds; every count in this module is bounded.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import random
from typing import Any, Dict, List, Optional, Sequence, Tuple

from dynamic_env.engine import Action, DynamicEnv, State, _step_internal

from . import size_selection as sel
from .size_invariant import (
    ID_HASH_DIM,
    KIND_ORDER,
    NODE_DIM,
    _normalised_number,
    action_main_key,
    hash_vector,
    node_features,
    node_references,
)

# --------------------------------------------------------------------------
# Constants (all widths are fixed; the value net is spec-agnostic)
# --------------------------------------------------------------------------

#: Value-net width. The llm-like design allows d = 32 or d = 64; d = 32 keeps
#: the CPU training inside the "minutes" cap of this unit.
DEFAULT_WIDTH = 32
#: Explicit seeds: features are seedless (pure hashing), the data walk, the
#: train batch order and the train/check split each have their own seed.
FEATURE_SEED = 20261012
TRAIN_SEED = 20261013
SPLIT_SEED = 20261014
#: Bounded training budget (a few hundred gradient steps, not the evolution loop).
DEFAULT_STEPS = 240
DEFAULT_BATCH = 32
DEFAULT_LR = 0.02
#: Bounded data budget: distinct states per spec before labeling.
DEFAULT_STATES_PER_SPEC = 96
#: Bounded negatives per state. On-canonical states have at most one positive
#: action but dozens of off-canonical legal actions, so keeping every negative
#: would make the dataset ~0.8% positive and the net could score 99% agreement
#: by always predicting 0 (useless as a planner). All positives are kept; the
#: negatives are spread deterministically with this cap.
DEFAULT_NEGATIVES_PER_STATE = 2
#: Weight applied to positive samples in the weighted value-MSE.
DEFAULT_POSITIVE_WEIGHT = 4.0
#: Weight of the per-state listwise ranking term (softmax cross-entropy over the
#: actions of one sampled state, target the on-canonical action). The pure
#: value-MSE threshold mask misranked the reset state (the canonical action
#: scored 0.965 while an off-canonical one scored 0.982), so the ranking term is
#: what makes the scalar head usable as a planner; 0.0 disables it.
DEFAULT_RANK_WEIGHT = 1.0
#: Decision threshold on the scalar membership value.
DEFAULT_THRESHOLD = 0.5

#: The objective-value hashed block and the fixed guard-target slots.
OBJ_BLOCK = ID_HASH_DIM
GUARD_SLOTS = 4
#: Hashed "bag of present node ids" block width. The masked-mean node pooling of
#: ``_pooled_rows`` dilutes one node by ``1 / count``, so a state with 7 nodes
#: keeps only ~1/7 of each node-id hash. That made the reset-state ranking fail
#: (the canonical successor ranked below an off-canonical one). This additive,
#: signed sketch keeps every present node id visible at full weight, so the
#: value net CAN learn the canonical node-set membership of the successor.
NODE_BAG_DIM = 16
#: Hashed "bag of node canonical forms" block width. The size-invariant
#: ``node_features`` does NOT encode ``node.canonical()``, so the successors of
#: ``build_sub:n9+n5`` and ``build_sub:n5+n9`` had IDENTICAL features while the
#: hand oracle labels them 1 and 0 (the canonical string carries the operand
#: order: ``grp(tag='t_sub',items=('n9','n5'))`` vs ``('n5','n9')``). That is a
#: contradictory dataset no value net can separate. This additive sketch of the
#: successor's canonical node forms removes the contradiction.
CANON_BAG_DIM = 32
#: Fixed feature widths. NODE_DIM comes from the size-invariant encoder (25).
STATE_DIM = NODE_DIM + 4 + OBJ_BLOCK + GUARD_SLOTS + NODE_BAG_DIM + CANON_BAG_DIM
ACTION_DIM = len(KIND_ORDER) + ID_HASH_DIM + 1 + NODE_DIM + ID_HASH_DIM
PAIR_DIM = 2 * STATE_DIM + ACTION_DIM

#: The canned checkpoint committed with this unit (trained by
#: ``evolution_trainer.train_learned_prune``; Part B may retrain fresh).
CHECKPOINT_NAME = "learned_prune_checkpoint.json"
CHECKPOINT_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                               CHECKPOINT_NAME)

PRUNE_MODES: Tuple[str, ...] = ("hand", "learned", "none")


# --------------------------------------------------------------------------
# Deterministic feature encoding of a (state, legal action) pair
# --------------------------------------------------------------------------

def _node_rows(env: DynamicEnv, state: State) -> List[List[float]]:
    """Per-node fixed-length feature rows (degrees from the spec graph)."""
    spec = env.spec
    nodes = list(state.nodes)
    index = {node.id: i for i, node in enumerate(nodes)}
    in_degree = {node.id: 0 for node in nodes}
    out_degree = {node.id: 0 for node in nodes}
    for node in nodes:
        refs = tuple(ref for ref in node_references(spec, node) if ref in index)
        out_degree[node.id] = len(refs)
        for ref in refs:
            in_degree[ref] += 1
    obj_values = state.obj_map()
    obj_targets = dict(spec.objectives)
    obj_mask = {oid: 1 for oid in spec.objectives}
    return [
        node_features(spec, node, in_degree[node.id], out_degree[node.id],
                      obj_values, obj_targets, obj_mask)
        for node in nodes
    ]


def _pooled_rows(rows: Sequence[Sequence[float]]) -> List[float]:
    """Masked mean pooling of the node rows (the node count is never a width)."""
    out = [0.0] * NODE_DIM
    if not rows:
        return out
    for row in rows:
        for i in range(NODE_DIM):
            out[i] += row[i]
    inverse = 1.0 / float(len(rows))
    return [value * inverse for value in out]


def _state_vector(env: DynamicEnv, state: State,
                  rows: Optional[Sequence[Sequence[float]]] = None
                  ) -> List[float]:
    """A fixed-length encoding of one state (same width for every spec)."""
    spec = env.spec
    if rows is None:
        rows = _node_rows(env, state)
    vec = _pooled_rows(rows)
    count = len(state.nodes)
    vec.extend([
        float(count) / (1.0 + float(count)),
        float(state.step) / (1.0 + float(state.step)),
        float(len(state.active)) / (1.0 + float(len(state.active))),
        float(len(spec.objectives)) / (1.0 + float(len(spec.objectives))),
    ])
    objectives = state.obj_map()
    obj_block = [0.0] * OBJ_BLOCK
    for oid in sorted(objectives):
        sign = 1.0 if objectives[oid] == 1 else -1.0
        hashed = hash_vector("obj:" + oid, OBJ_BLOCK, salt=4)
        for i in range(OBJ_BLOCK):
            obj_block[i] += sign * hashed[i]
    if objectives:
        inverse = 1.0 / float(len(objectives))
        obj_block = [value * inverse for value in obj_block]
    vec.extend(obj_block)
    guards = [0.0] * GUARD_SLOTS
    for i, gid in enumerate(sorted(spec.guards)[:GUARD_SLOTS]):
        value = spec.guards[gid].get("value")
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            guards[i] = _normalised_number(value)
    vec.extend(guards)
    # Additive signed sketch of the present node ids (not mean-pooled): the
    # successor block of ``[s ; a ; s']`` thus exposes exactly which node ids
    # the candidate action would introduce, which is what the canonical-subtree
    # membership depends on.
    bag = [0.0] * NODE_BAG_DIM
    for node in state.nodes:
        hashed = hash_vector(node.id, NODE_BAG_DIM, salt=5)
        for i in range(NODE_BAG_DIM):
            bag[i] += hashed[i]
    vec.extend(bag)
    # Additive sketch of the present nodes' canonical forms. This is what makes
    # the hand label (``node.canonical() in allowed``) learnable: without it the
    # operand order of a build/combine is invisible in the features.
    canon = [0.0] * CANON_BAG_DIM
    for node in state.nodes:
        hashed = hash_vector(node.canonical(), CANON_BAG_DIM, salt=7)
        for i in range(CANON_BAG_DIM):
            canon[i] += hashed[i]
    vec.extend(canon)
    return vec


def _action_vector(env: DynamicEnv, state: State, action: Action,
                   rows: Optional[Sequence[Sequence[float]]] = None
                   ) -> List[float]:
    """A fixed-length encoding of one legal action relative to its state."""
    spec = env.spec
    if rows is None:
        rows = _node_rows(env, state)
    index = {node.id: i for i, node in enumerate(state.nodes)}
    vec = [0.0] * len(KIND_ORDER)
    if action.kind in KIND_ORDER:
        vec[KIND_ORDER.index(action.kind)] = 1.0
    vec.extend(hash_vector(action_main_key(action), ID_HASH_DIM, salt=2))
    arity = len(action.operands)
    vec.append(float(arity) / (1.0 + float(arity)))
    aggregate = [0.0] * NODE_DIM
    count = 0
    for oid in action.operands:
        position = index.get(oid)
        if position is None:
            continue
        count += 1
        row = rows[position]
        for k in range(NODE_DIM):
            aggregate[k] += row[k]
    if count:
        inverse = 1.0 / float(count)
        aggregate = [value * inverse for value in aggregate]
    vec.extend(aggregate)
    tag = ""
    if action.kind == "build" and action.axiom_id:
        axiom = spec.axiom(action.axiom_id)
        tag = str(getattr(axiom, "tag_node", "") or "")
    elif action.kind == "combine":
        tag = str(action.tag_id or "")
    vec.extend(hash_vector(tag, ID_HASH_DIM, salt=6))
    return vec


def state_action_features(env: DynamicEnv, state: State, action: Action
                          ) -> List[float]:
    """The value-head input for one (state, action) pair.

    The pair is ``[s ; a ; s']``, exactly the ``v(s, s')`` transition shape of
    the llm-like value head (``LLM_LIKE.md`` section 3.2), with ``s``/``s'``
    encoded by the size-invariant node/global/objective blocks.
    """
    rows = _node_rows(env, state)
    left = _state_vector(env, state, rows)
    mid = _action_vector(env, state, action, rows)
    try:
        nxt, _info = _step_internal(env.spec, state, action)
    except Exception:  # illegal under the engine rules -> zero successor block
        nxt = None
    right = ([0.0] * STATE_DIM if nxt is None
             else _state_vector(env, nxt))
    return left + mid + right


# --------------------------------------------------------------------------
# Deterministic labeling (the hand prune as the training oracle)
# --------------------------------------------------------------------------

def label_action(env: DynamicEnv, state: State, action: Action,
                 allowed: Optional[Any] = None) -> int:
    """1 when ``action`` keeps the successor on the canonical minimal subtree.

    This is exactly the membership predicate of
    ``size_selection.canonical_subtree_prune``: every node of the successor
    state must be a canonical-subtree node of the spec. Returns 1 when the spec
    has no canonical word (nothing to prune) and 0 when the action is illegal.
    """
    if allowed is None:
        allowed = sel.canonical_allowed_nodes(env.spec)
    if allowed is None:
        return 1
    try:
        nxt, _info = _step_internal(env.spec, state, action)
    except Exception:
        return 0
    return 1 if all(node.canonical() in allowed for node in nxt.nodes) else 0


# --------------------------------------------------------------------------
# Dataset construction (bundled specs + the held-out TARGET-variant pool)
# --------------------------------------------------------------------------

def _walk_states(env: DynamicEnv, per_spec: int, seed: int,
                 extra_states: Sequence[State] = ()) -> List[State]:
    """A bounded deterministic set of states for one spec."""
    states: List[State] = [env.reset()]
    states.extend(extra_states)
    _word, canonical = sel._canonical_states(env.spec)
    states.extend(canonical[1:])
    seen = set()
    unique: List[State] = []
    for state in states:
        key = state.identity()
        if key in seen:
            continue
        seen.add(key)
        unique.append(state)
    rng = random.Random(int(seed))
    attempts = 0
    while len(unique) < int(per_spec) and attempts < int(per_spec) * 8:
        attempts += 1
        start = unique[rng.randrange(len(unique))]
        current = start
        for _ in range(rng.randint(1, 3)):
            actions = env.legal_actions(current)
            if not actions:
                break
            current = env.step(current, actions[rng.randrange(len(actions))]).state
            key = current.identity()
            if key not in seen:
                seen.add(key)
                unique.append(current)
            if len(unique) >= int(per_spec):
                break
    return sorted(unique, key=lambda state: state.identity())


def collect_records_for_spec(spec: Any, seed: int, per_spec: int,
                             extra_states: Sequence[State] = (),
                             negatives_per_state: int = DEFAULT_NEGATIVES_PER_STATE
                             ) -> List[Dict[str, Any]]:
    """Label every legal action of every sampled state of one spec.

    ALL on-canonical (label 1) actions are kept; the off-canonical (label 0)
    actions are thinned to ``negatives_per_state`` per state by a deterministic
    stride. Same ``(spec, seed, per_spec, negatives_per_state)`` -> same records.
    """
    env = DynamicEnv(spec)
    allowed = sel.canonical_allowed_nodes(spec)
    records: List[Dict[str, Any]] = []
    for state in _walk_states(env, per_spec, seed, extra_states):
        positives: List[Action] = []
        negatives: List[Action] = []
        for action in env.legal_actions(state):
            if label_action(env, state, action, allowed) == 1:
                positives.append(action)
            else:
                negatives.append(action)
        cap = max(0, int(negatives_per_state))
        if cap and len(negatives) > cap:
            step = len(negatives) / float(cap)
            negatives = [negatives[int(i * step)] for i in range(cap)]
        for action in sorted(positives + negatives, key=lambda a: a.key()):
            is_on = action in positives
            records.append({
                "spec_id": spec.spec_id,
                "state": state.identity(),
                "action": action.key(),
                "label": 1 if is_on else 0,
                "features": state_action_features(env, state, action),
            })
    return records


def build_dataset(spec_ids: Optional[Sequence[str]] = None,
                  per_spec: int = DEFAULT_STATES_PER_SPEC,
                  seed: int = FEATURE_SEED,
                  target_cases: Optional[Sequence[Any]] = None,
                  negatives_per_state: int = DEFAULT_NEGATIVES_PER_STATE
                  ) -> List[Dict[str, Any]]:
    """The deterministic labeled dataset over the bundled specs + variants.

    Deterministic for a fixed ``(spec_ids, per_spec, seed, negatives_per_state)``:
    the state walks use explicit seeds and the records are emitted in sorted
    state/action order. Same ``(seed, spec)`` -> same labels.
    """
    if spec_ids is None:
        spec_ids = tuple(sel.TRAIN_SPEC_IDS) + tuple(sel.HELDOUT_SPEC_IDS)
    records: List[Dict[str, Any]] = []
    for index, spec_id in enumerate(spec_ids):
        spec = sel.spec_by_id(str(spec_id))
        records.extend(collect_records_for_spec(
            spec, int(seed) + 101 * index, per_spec,
            negatives_per_state=negatives_per_state))
    if target_cases is None:
        target_cases = sel.target_change_cases()
    for index, case in enumerate(target_cases):
        records.extend(collect_records_for_spec(
            case.spec, int(seed) + 7000003 + 101 * index, per_spec,
            extra_states=(case.start_state,),
            negatives_per_state=negatives_per_state))
    return records


def split_records(records: Sequence[Dict[str, Any]], holdout_frac: float = 0.15,
                  seed: int = SPLIT_SEED
                  ) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """A deterministic train / held-in check split (disjoint sample indices)."""
    rng = random.Random(int(seed))
    order = list(range(len(records)))
    rng.shuffle(order)
    cut = int(len(records) * float(holdout_frac))
    held = set(order[:cut])
    train = [record for i, record in enumerate(records) if i not in held]
    check = [record for i, record in enumerate(records) if i in held]
    return train, check


def dataset_stats(records: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    """Counts for evidence: samples, positives and distinct specs/states."""
    positives = sum(1 for record in records if record["label"] == 1)
    return {
        "samples": len(records),
        "positives": positives,
        "negatives": len(records) - positives,
        "positives_rate": (positives / float(len(records)) if records else 0.0),
        "specs": sorted({record["spec_id"] for record in records}),
        "states": len({(record["spec_id"], record["state"])
                       for record in records}),
    }


# --------------------------------------------------------------------------
# The value net (pure-stdlib mirror of the llm-like value head)
# --------------------------------------------------------------------------

class ValueNet:
    """A one-hidden-layer tanh scalar value head, ``d = hidden``.

    ``v = w2 . tanh(W1 x + b1) + b2`` with ``x`` the fixed ``PAIR_DIM``
    ``[s ; a ; s']`` feature vector. This is the value head of
    ``llm_like.model.LLMAgentNet`` (``value_proj`` + ``w_v_head``) with the
    forward pass re-expressed in standard library arithmetic.
    """

    def __init__(self, in_dim: int = PAIR_DIM, hidden: int = DEFAULT_WIDTH,
                 seed: int = TRAIN_SEED) -> None:
        self.in_dim = int(in_dim)
        self.hidden = int(hidden)
        if self.in_dim <= 0 or self.hidden <= 0:
            raise ValueError("widths must be positive")
        self.o_w1 = 0
        self.o_b1 = self.hidden * self.in_dim
        self.o_w2 = self.o_b1 + self.hidden
        self.o_b2 = self.o_w2 + self.hidden
        self.theta = self._init_theta(int(seed))

    def _init_theta(self, seed: int) -> List[float]:
        rng = random.Random(seed)
        scale1 = 1.0 / math.sqrt(float(self.in_dim))
        scale2 = 1.0 / math.sqrt(float(self.hidden))
        theta = [rng.gauss(0.0, scale1)
                 for _ in range(self.hidden * self.in_dim)]
        theta.extend([0.0] * self.hidden)
        theta.extend([rng.gauss(0.0, scale2) for _ in range(self.hidden)])
        theta.append(0.0)
        return theta

    def forward(self, features: Sequence[float]) -> Tuple[float, List[float]]:
        """Return ``(value, hidden_activations)`` for one feature vector."""
        hidden = self.hidden
        theta = self.theta
        h = [0.0] * hidden
        for j in range(hidden):
            base = j * self.in_dim
            total = theta[self.o_b1 + j]
            for i in range(self.in_dim):
                total += theta[base + i] * features[i]
            h[j] = math.tanh(total)
        value = theta[self.o_b2]
        for j in range(hidden):
            value += theta[self.o_w2 + j] * h[j]
        return value, h

    def predict(self, features: Sequence[float]) -> float:
        return self.forward(features)[0]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "in_dim": self.in_dim,
            "hidden": self.hidden,
            "width": self.hidden,
            "state_dim": STATE_DIM,
            "action_dim": ACTION_DIM,
            "pair_dim": PAIR_DIM,
            "theta": [round(value, 12) for value in self.theta],
        }

    @staticmethod
    def from_dict(raw: Dict[str, Any]) -> "ValueNet":
        net = ValueNet.__new__(ValueNet)
        net.in_dim = int(raw["in_dim"])
        net.hidden = int(raw["hidden"])
        net.o_w1 = 0
        net.o_b1 = net.hidden * net.in_dim
        net.o_w2 = net.o_b1 + net.hidden
        net.o_b2 = net.o_w2 + net.hidden
        net.theta = [float(value) for value in raw["theta"]]
        if len(net.theta) != net.o_b2 + 1:
            raise ValueError("checkpoint theta has the wrong length")
        return net

    def save(self, path: str = CHECKPOINT_PATH,
             meta: Optional[Dict[str, Any]] = None) -> str:
        payload = self.to_dict()
        if meta is not None:
            payload["meta"] = meta
        payload["feature_seed"] = FEATURE_SEED
        payload["train_seed"] = TRAIN_SEED
        with open(path, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=1, sort_keys=True)
            handle.write("\n")
        return path


def load_checkpoint(path: Optional[str] = None) -> ValueNet:
    """Load the canned (or a fresh) value-net checkpoint."""
    target = path if path else CHECKPOINT_PATH
    with open(target, "r", encoding="utf-8") as handle:
        raw = json.load(handle)
    return ValueNet.from_dict(raw)


def checkpoint_sha256(path: Optional[str] = None) -> str:
    target = path if path else CHECKPOINT_PATH
    with open(target, "rb") as handle:
        return hashlib.sha256(handle.read()).hexdigest()


# --------------------------------------------------------------------------
# Training (manual Adam; deterministic; bounded)
# --------------------------------------------------------------------------

def _batch_loss(net: ValueNet, features: Sequence[Sequence[float]],
                labels: Sequence[float], positive_weight: float = 1.0) -> float:
    total = 0.0
    weight_sum = 0.0
    for x, y in zip(features, labels):
        weight = positive_weight if y >= 0.5 else 1.0
        value = net.predict(x)
        total += weight * (value - y) ** 2
        weight_sum += weight
    return total / float(max(1.0, weight_sum))


def _backprop(net: ValueNet, x: Sequence[float], h: Sequence[float],
              d_out: float, gradient: List[float]) -> None:
    """Accumulate the gradient of one scalar-head output into ``gradient``."""
    theta = net.theta
    hidden = net.hidden
    gradient[net.o_b2] += d_out
    for j in range(hidden):
        gradient[net.o_w2 + j] += d_out * h[j]
        d_h = d_out * theta[net.o_w2 + j]
        d_z = d_h * (1.0 - h[j] * h[j])
        gradient[net.o_b1 + j] += d_z
        base = j * net.in_dim
        for i in range(net.in_dim):
            gradient[base + i] += d_z * x[i]


def _state_groups(records: Sequence[Dict[str, Any]],
                  labels: Sequence[float]
                  ) -> List[List[int]]:
    """The record-index groups (one per state) that contain a positive.

    The listwise ranking term needs, for a sampled state, every legal action it
    has with the hand label; a group is rankable only when it has both a positive
    and a negative. Sorted by (spec_id, state) so the sampled order is stable.
    """
    groups: Dict[Tuple[str, str], List[int]] = {}
    for index, record in enumerate(records):
        key = (str(record["spec_id"]), str(record["state"]))
        groups.setdefault(key, []).append(index)
    rankable = [idxs for idxs in groups.values()
                if any(labels[i] >= 0.5 for i in idxs)
                and any(labels[i] < 0.5 for i in idxs)]
    rankable.sort(key=lambda idxs: (str(records[idxs[0]]["spec_id"]),
                                    str(records[idxs[0]]["state"])))
    return rankable


def train_value_net(records: Sequence[Dict[str, Any]],
                    hidden: int = DEFAULT_WIDTH, steps: int = DEFAULT_STEPS,
                    batch: int = DEFAULT_BATCH, lr: float = DEFAULT_LR,
                    seed: int = TRAIN_SEED,
                    positive_weight: float = DEFAULT_POSITIVE_WEIGHT,
                    rank_weight: float = DEFAULT_RANK_WEIGHT
                    ) -> Tuple[ValueNet, Dict[str, Any]]:
    """Train the value head with weighted MSE plus a per-state ranking term.

    A bounded number of gradient steps (default 240, ``batch = 32``). The batch
    order, the initialization and the checkpoint rounding are all explicit, so
    the same records and seed produce the same theta. ``positive_weight``
    upweights the on-canonical samples; ``rank_weight`` adds a listwise softmax
    cross-entropy over one sampled state's actions whose target is the
    on-canonical action. The ranking term is what fixes the reset-state ordering
    that the pure value-MSE threshold mask could not separate. Returns
    ``(net, info)``.
    """
    net = ValueNet(PAIR_DIM, hidden, seed)
    features = [record["features"] for record in records]
    labels = [float(record["label"]) for record in records]
    count = len(features)
    if count == 0:
        raise ValueError("cannot train on an empty dataset")
    rankable = _state_groups(records, labels) if rank_weight else []
    rng = random.Random(int(seed) + 1)
    beta1, beta2, eps = 0.9, 0.999, 1e-8
    moment1 = [0.0] * len(net.theta)
    moment2 = [0.0] * len(net.theta)
    initial = _batch_loss(net, features[:min(512, count)],
                          labels[:min(512, count)], positive_weight)
    sample = max(1, min(int(batch), count))
    rank_states = 0
    for step in range(int(steps)):
        gradient = [0.0] * len(net.theta)
        inverse = 1.0 / float(sample)
        for _ in range(sample):
            position = rng.randrange(count)
            x = features[position]
            y = labels[position]
            weight = positive_weight if y >= 0.5 else 1.0
            value, h = net.forward(x)
            d_out = 2.0 * weight * (value - y) * inverse
            _backprop(net, x, h, d_out, gradient)
        if rank_weight and rankable:
            idxs = rankable[rng.randrange(len(rankable))]
            values: List[float] = []
            hidden_acts: List[List[float]] = []
            for index in idxs:
                value, h = net.forward(features[index])
                values.append(value)
                hidden_acts.append(h)
            peak = max(values)
            weights = [math.exp(value - peak) for value in values]
            total = sum(weights)
            target = next(index for index in idxs if labels[index] >= 0.5)
            target_position = idxs.index(target)
            scale = rank_weight / float(len(idxs))
            for position, index in enumerate(idxs):
                probability = weights[position] / total
                d_out = scale * (probability
                                 - (1.0 if position == target_position else 0.0))
                _backprop(net, features[index], hidden_acts[position], d_out,
                          gradient)
            rank_states += 1
        t = step + 1
        correction1 = 1.0 - beta1 ** t
        correction2 = 1.0 - beta2 ** t
        theta = net.theta
        for p in range(len(theta)):
            moment1[p] = beta1 * moment1[p] + (1.0 - beta1) * gradient[p]
            moment2[p] = beta2 * moment2[p] + (1.0 - beta2) * gradient[p] ** 2
            m_hat = moment1[p] / correction1
            v_hat = moment2[p] / correction2
            theta[p] -= lr * m_hat / (math.sqrt(v_hat) + eps)
    final = _batch_loss(net, features, labels, positive_weight)
    info = {
        "samples": count,
        "hidden": int(hidden),
        "steps": int(steps),
        "batch": int(batch),
        "lr": float(lr),
        "seed": int(seed),
        "positive_weight": float(positive_weight),
        "rank_weight": float(rank_weight),
        "rank_states": int(len(rankable)),
        "rank_steps": int(rank_states),
        "initial_mse": round(initial, 8),
        "final_mse": round(final, 8),
    }
    return net, info


def train_and_save(checkpoint: str = CHECKPOINT_PATH,
                   hidden: int = DEFAULT_WIDTH, steps: int = DEFAULT_STEPS,
                   batch: int = DEFAULT_BATCH, lr: float = DEFAULT_LR,
                   per_spec: int = DEFAULT_STATES_PER_SPEC,
                   seed: int = TRAIN_SEED,
                   positive_weight: float = DEFAULT_POSITIVE_WEIGHT,
                   negatives_per_state: int = DEFAULT_NEGATIVES_PER_STATE,
                   rank_weight: float = DEFAULT_RANK_WEIGHT
                   ) -> Tuple[ValueNet, Dict[str, Any]]:
    """Build the dataset, train the net, save the checkpoint and report.

    ``negatives_per_state == 0`` keeps EVERY off-canonical action (no thinning);
    a positive cap thins the negatives deterministically per state (see
    :func:`collect_records_for_spec`). ``rank_weight`` is the per-state ranking
    term of :func:`train_value_net`.
    """
    records = build_dataset(per_spec=per_spec, seed=FEATURE_SEED,
                            negatives_per_state=negatives_per_state)
    train, check = split_records(records)
    net, info = train_value_net(train, hidden=hidden, steps=steps, batch=batch,
                                lr=lr, seed=seed,
                                positive_weight=positive_weight,
                                rank_weight=rank_weight)
    info["dataset"] = dataset_stats(records)
    info["negatives_per_state"] = int(negatives_per_state)
    info["train_samples"] = len(train)
    info["check_samples"] = len(check)
    info["train_agreement"] = round(
        agreement_rate(net, train), 6)
    info["check_agreement"] = round(
        agreement_rate(net, check), 6)
    info["train_off_canonical"] = off_canonical_stats(net, train)
    info["check_off_canonical"] = off_canonical_stats(net, check)
    net.save(checkpoint, meta=info)
    info["checkpoint"] = checkpoint
    info["checkpoint_sha256"] = checkpoint_sha256(checkpoint)
    return net, info


# --------------------------------------------------------------------------
# Metrics
# --------------------------------------------------------------------------

def predictions(net: ValueNet, records: Sequence[Dict[str, Any]],
                threshold: float = DEFAULT_THRESHOLD) -> List[int]:
    return [1 if net.predict(record["features"]) >= threshold else 0
            for record in records]


def confusion(net: ValueNet, records: Sequence[Dict[str, Any]],
              threshold: float = DEFAULT_THRESHOLD) -> Dict[str, int]:
    """Confusion counts of the learned mask against the hand-prune labels."""
    predicted = predictions(net, records, threshold)
    tp = tn = fp = fn = 0
    for record, guess in zip(records, predicted):
        truth = int(record["label"])
        if truth == 1 and guess == 1:
            tp += 1
        elif truth == 0 and guess == 0:
            tn += 1
        elif truth == 0 and guess == 1:
            fp += 1
        else:
            fn += 1
    return {"tp": tp, "tn": tn, "fp": fp, "fn": fn, "total": len(records)}


def agreement_rate(net: ValueNet, records: Sequence[Dict[str, Any]],
                   threshold: float = DEFAULT_THRESHOLD) -> float:
    """Fraction of (state, action) pairs where the learned mask == hand label."""
    if not records:
        return 0.0
    stats = confusion(net, records, threshold)
    return (stats["tp"] + stats["tn"]) / float(stats["total"])


def off_canonical_stats(net: ValueNet, records: Sequence[Dict[str, Any]],
                        threshold: float = DEFAULT_THRESHOLD) -> Dict[str, Any]:
    """How the learned mask treats the off-canonical (label == 0) actions.

    ``mode="none"`` keeps ALL of them (``kept == total``); a useful learned mask
    keeps fewer. ``removed`` is the count it filters out.
    """
    off = [record for record in records if int(record["label"]) == 0]
    predicted = predictions(net, off, threshold)
    kept = sum(predicted)
    return {
        "off_canonical_total": len(off),
        "off_canonical_kept": kept,
        "off_canonical_removed": len(off) - kept,
        "off_canonical_removed_rate": ((len(off) - kept) / float(len(off))
                                       if off else 0.0),
    }


# --------------------------------------------------------------------------
# Inference: the learned prune
# --------------------------------------------------------------------------

def learned_prune(env: DynamicEnv, state: State, actions: Sequence[Action],
                  net: Optional[ValueNet],
                  threshold: float = DEFAULT_THRESHOLD) -> Tuple[Action, ...]:
    """The mask produced by the trained value function (no hand filter).

    Keeps every action whose successor has a scalar membership value at or
    above ``threshold``. With no net loaded, the actions are returned unchanged.
    """
    if net is None:
        return tuple(actions)
    kept: List[Action] = []
    for action in actions:
        value = net.predict(state_action_features(env, state, action))
        if value >= threshold:
            kept.append(action)
    return tuple(kept)


def prune_actions(env: DynamicEnv, state: State, actions: Sequence[Action],
                  seen: Optional[Any] = None, mode: str = "hand",
                  net: Optional[ValueNet] = None,
                  threshold: float = DEFAULT_THRESHOLD) -> Tuple[Action, ...]:
    """Dispatch the three eval prune modes; never stronger than "hand" default.

    * ``hand``    -> :func:`size_selection.masked_legal_actions`
    * ``learned`` -> :func:`learned_prune` only (no hand-written filter)
    * ``none``    -> the actions unchanged
    """
    mode = str(mode)
    if mode == "none":
        return tuple(actions)
    if mode == "hand":
        return sel.masked_legal_actions(env, state, set(seen or ()),
                                        actions=actions)
    if mode == "learned":
        return learned_prune(env, state, actions, net, threshold)
    raise ValueError("unknown prune mode %r (expected one of %r)"
                     % (mode, PRUNE_MODES))


# --------------------------------------------------------------------------
# Eval harness: 30 deterministic episodes in all three modes
# --------------------------------------------------------------------------

def run_episodes(spec_id: str = "spec_multi_step", mode: str = "hand",
                 episodes: int = 30, generation: int = 0,
                 selection_seed: int = sel.SELECTION_SEED,
                 epsilon: float = sel.SELECTION_EPSILON,
                 genome_seed: int = 7,
                 net: Optional[ValueNet] = None,
                 max_steps: Optional[int] = None) -> Dict[str, Any]:
    """Run ``episodes`` explicit-seed episodes under one prune mode.

    The policy is the size-invariant net with a fixed seeded genome, so every
    mode sees exactly the same episodes. No hand-written filter is applied in
    ``learned``; ``hand`` is the task-4344 filter; ``none`` disables masking.
    """
    from .size_invariant import SizeInvariantNet

    spec = sel.spec_by_id(spec_id)
    policy = SizeInvariantNet(12)
    genome = policy.init(random.Random(int(genome_seed)), scale=0.5)
    root_mask = tuple(1 for _ in spec.objectives)
    solved = 0
    total_steps = 0
    solved_steps: List[int] = []
    for index in range(int(episodes)):
        seed = sel.episode_seed(selection_seed, generation, index)
        rng = random.Random(seed)
        result = sel.rollout(
            policy, genome, spec, None, root_mask, rng, epsilon=epsilon,
            mask_illegal=(mode == "hand"), prune_mode=mode, prune_net=net,
            max_steps=max_steps)
        total_steps += int(result["steps"])
        if result["solved"]:
            solved += 1
            solved_steps.append(int(result["steps"]))
    return {
        "mode": mode,
        "spec_id": spec_id,
        "episodes": int(episodes),
        "generation": int(generation),
        "selection_seed": int(selection_seed),
        "epsilon": float(epsilon),
        "genome_seed": int(genome_seed),
        "solved": solved,
        "solve_rate": solved / float(max(1, int(episodes))),
        "mean_solved_steps": (sum(solved_steps) / float(len(solved_steps))
                              if solved_steps else None),
        "mean_all_steps": total_steps / float(max(1, int(episodes))),
    }


def run_all_modes(spec_id: str = "spec_multi_step", episodes: int = 30,
                  generation: int = 0,
                  selection_seed: int = sel.SELECTION_SEED,
                  epsilon: float = sel.SELECTION_EPSILON,
                  genome_seed: int = 7, checkpoint: Optional[str] = None
                  ) -> Dict[str, Dict[str, Any]]:
    """The three-mode harness (hand / learned / none) on the same episodes."""
    net = load_checkpoint(checkpoint)
    out: Dict[str, Dict[str, Any]] = {}
    for mode in PRUNE_MODES:
        out[mode] = run_episodes(spec_id, mode, episodes=episodes,
                                 generation=generation,
                                 selection_seed=selection_seed,
                                 epsilon=epsilon, genome_seed=genome_seed,
                                 net=net)
    return out


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------

def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description="Learned prune over the REDUCTION_TABLE (stdlib)")
    parser.add_argument("--train", action="store_true",
                        help="train the value net and write the checkpoint")
    parser.add_argument("--checkpoint", default=CHECKPOINT_PATH)
    parser.add_argument("--spec", default="spec_multi_step")
    parser.add_argument("--episodes", type=int, default=30)
    parser.add_argument("--generation", type=int, default=0)
    parser.add_argument("--seed", type=int, default=sel.SELECTION_SEED)
    parser.add_argument("--epsilon", type=float, default=sel.SELECTION_EPSILON)
    parser.add_argument("--genome-seed", type=int, default=7)
    parser.add_argument("--modes", default="hand,learned,none")
    parser.add_argument("--hidden", type=int, default=DEFAULT_WIDTH)
    parser.add_argument("--steps", type=int, default=DEFAULT_STEPS)
    parser.add_argument("--batch", type=int, default=DEFAULT_BATCH)
    parser.add_argument("--lr", type=float, default=DEFAULT_LR)
    parser.add_argument("--per-spec", type=int, default=DEFAULT_STATES_PER_SPEC)
    parser.add_argument("--metrics", action="store_true",
                        help="print dataset + agreement metrics and exit")
    args = parser.parse_args(argv)

    if args.train:
        net, info = train_and_save(args.checkpoint, hidden=args.hidden,
                                   steps=args.steps, batch=args.batch,
                                   lr=args.lr, per_spec=args.per_spec)
        print(json.dumps(info, indent=1, sort_keys=True))
        return 0
    if args.metrics:
        net = load_checkpoint(args.checkpoint)
        records = build_dataset(per_spec=args.per_spec)
        train, check = split_records(records)
        report = {
            "dataset": dataset_stats(records),
            "agreement_all": round(agreement_rate(net, records), 6),
            "agreement_train": round(agreement_rate(net, train), 6),
            "agreement_check": round(agreement_rate(net, check), 6),
            "confusion_all": confusion(net, records),
            "off_canonical_all": off_canonical_stats(net, records),
            "checkpoint_sha256": checkpoint_sha256(args.checkpoint),
        }
        print(json.dumps(report, indent=1, sort_keys=True))
        return 0

    net = load_checkpoint(args.checkpoint) if "learned" in args.modes else None
    for mode in [m.strip() for m in args.modes.split(",") if m.strip()]:
        print(json.dumps(run_episodes(
            args.spec, mode, episodes=args.episodes,
            generation=args.generation, selection_seed=args.seed,
            epsilon=args.epsilon, genome_seed=args.genome_seed, net=net),
            sort_keys=True))
    return 0


__all__ = [
    "DEFAULT_WIDTH",
    "FEATURE_SEED",
    "TRAIN_SEED",
    "SPLIT_SEED",
    "DEFAULT_STEPS",
    "DEFAULT_BATCH",
    "DEFAULT_LR",
    "DEFAULT_STATES_PER_SPEC",
    "DEFAULT_NEGATIVES_PER_STATE",
    "DEFAULT_POSITIVE_WEIGHT",
    "DEFAULT_RANK_WEIGHT",
    "DEFAULT_THRESHOLD",
    "STATE_DIM",
    "ACTION_DIM",
    "PAIR_DIM",
    "PRUNE_MODES",
    "CHECKPOINT_NAME",
    "CHECKPOINT_PATH",
    "ValueNet",
    "label_action",
    "state_action_features",
    "collect_records_for_spec",
    "build_dataset",
    "split_records",
    "dataset_stats",
    "train_value_net",
    "train_and_save",
    "load_checkpoint",
    "checkpoint_sha256",
    "predictions",
    "confusion",
    "agreement_rate",
    "off_canonical_stats",
    "learned_prune",
    "prune_actions",
    "run_episodes",
    "run_all_modes",
    "main",
]


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())

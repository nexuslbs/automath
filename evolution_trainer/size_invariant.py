"""Size-invariant genome: a graph-encoder policy over the generic node graph.

The Unit B genome (:mod:`evolution_trainer.genome`) is the flat weight vector of
a one-hidden-layer MLP whose INPUT length is ``state_dim(spec) +
ActionFeaturizer(spec).dim``. Both terms are functions of the spec (objective
count, node-type count, the per-spec id vocabulary, the spec's maximum arity), so
one trained genome only runs on the one spec it was sized for. This module
replaces the architecture, not the evolutionary process: the genome is still a
flat list of floats mixed by :func:`evolution_trainer.genome.mix_genomes`, but its
length and every matrix shape are CONSTANTS.

Invariant
---------
``SizeInvariantNet(hidden)`` has a fixed ``size`` and no spec-dependent state.
The same genome list, with no re-allocation and no re-shaping, scores the legal
actions of ANY spec built by ``dynamic_env``:

* every legal action is scored independently by the SAME network, so the OUTPUT
  shape is always one float per action (action count is not a width);
* the state is encoded as a SET of per-node feature vectors and pooled by a
  masked mean, so the NODE COUNT is not a width;
* graph edges are read from the spec's own declared node/nodes fields, so the
  ARITY of a combination node is not a width;
* id strings (node, axiom, tag, objective and node-type names) enter through a
  fixed signed feature hash, so the ID VOCABULARY is not a width.

Everything is standard library only, exactly like ``dynamic_env`` and the rest of
the evolution trainer.

Relationship to the ``target-policy`` branch (REFERENCE ONLY, not merged)
------------------------------------------------------------------------
``origin/target-policy`` implemented a target-conditioned ``PolicyNet`` that
scores ``features(target, stack, action)`` with the same parameters for any
target. Its feature blocks are fixed-length because they summarise a FIXED
hardcoded token vocabulary (``new_approach``'s ``SEMANTIC_OPS``); that is the
idea reused here: the target is data in the input and one genome scores every
action. The parts built fresh here are the generic graph encoder (per-node
features + message passing + masked pooling) and the feature-hash of the spec's
own id strings, which remove even the fixed-vocabulary assumption.

Genome layout (all matrices fixed size; ``H = hidden``)::

    W_node   H x NODE_DIM          b_node   H
    W_self   H x H                 W_neigh  H x H          b_msg  H
    W_g      H x GLOBAL_DIM        b_g      H
    W_a      H x ACTION_DIM        b_a      H
    W_c      H x H                 b_c      H
    w_out    H                     b_out    1
"""

from __future__ import annotations

import math
import random
from typing import Dict, List, Mapping, Optional, Sequence, Tuple

from dynamic_env.engine import Action, DynamicEnv, State
from dynamic_env.spec import Spec

#: The four derived action kinds, in a fixed order (mirrors features.KIND_ORDER).
KIND_ORDER: Tuple[str, ...] = ("build", "combine", "set", "clear")
#: Node roles and semantics, in a fixed order.
ROLE_ORDER: Tuple[str, ...] = ("value", "tag", "objective", "plain")
SEMANTICS_ORDER: Tuple[str, ...] = ("literal", "combination")

#: Fixed widths of the hashed blocks (NOT a vocabulary: collisions are allowed
#: and the signed hash keeps colliding names distinguishable on average).
TYPE_HASH_DIM = 4
ID_HASH_DIM = 8

#: Per-node feature length: role(4) + semantics(2) + type-hash(4) + id-hash(8)
#: + value(1) + is-dynamic-type(1) + objective cur/target/mask(3)
#: + in/out degree(2).
NODE_DIM = (len(ROLE_ORDER) + len(SEMANTICS_ORDER) + TYPE_HASH_DIM
            + ID_HASH_DIM + 1 + 1 + 3 + 2)
#: Fixed global scalars: node count, step, active-axiom count, objective count.
GLOBAL_DIM = 4


# --------------------------------------------------------------------------
# Deterministic signed feature hashing (no vocabulary, no spec ids)
# --------------------------------------------------------------------------

def _fnv1a(text: str, salt: int = 0) -> int:
    h = 2166136261
    h = (h ^ (salt & 0xFFFFFFFF)) & 0xFFFFFFFF
    for ch in text:
        h ^= ord(ch)
        h = (h * 16777619) & 0xFFFFFFFF
    return h


def hash_vector(text: str, dim: int, salt: int = 0) -> List[float]:
    """A deterministic, signed, fixed-length embedding of a name string.

    The same string always maps to the same vector on every spec and every run;
    different strings usually map to different indices, and collisions still
    differ in sign/magnitude. No name is ever stored, only hashed.
    """
    vec = [0.0] * dim
    if not text:
        return vec
    digest = _fnv1a(text, salt)
    index = digest % dim
    sign = 1.0 if ((digest >> 16) & 1) == 0 else -1.0
    magnitude = 0.5 + float((digest >> 8) % 1000) / 2000.0
    vec[index] = sign * magnitude
    return vec


def stable_hash(text: str, dim: int, salt: int = 0) -> int:
    """The bucket a name maps to (exposed for tests/diagnostics)."""
    return _fnv1a(text, salt) % dim


# --------------------------------------------------------------------------
# Per-node and per-action feature encoders (all fixed length)
# --------------------------------------------------------------------------

def node_references(spec: Spec, node) -> Tuple[str, ...]:
    """The node ids a node references, read from its DECLARED field types.

    A field declared ``node`` holds one id; a field declared ``nodes`` holds a
    tuple of ids. This is exactly the spec's own graph structure (for the shipped
    fixtures: a combination node's ``tag`` + ``items``), never a hardcoded name.
    """
    ntype = spec.node_types.get(node.type)
    if ntype is None:
        return ()
    refs: List[str] = []
    for field in ntype.fields:
        if field.type == "node":
            value = node.get(field.name)
            if isinstance(value, str):
                refs.append(value)
        elif field.type == "nodes":
            value = node.get(field.name)
            if isinstance(value, (tuple, list)):
                refs.extend(item for item in value if isinstance(item, str))
    return tuple(refs)


def _normalised_number(value) -> float:
    if isinstance(value, bool):
        return 1.0 if value else 0.0
    if isinstance(value, (int, float)):
        number = float(value)
        return number / (1.0 + abs(number))
    return 0.0


def node_features(spec: Spec, node, in_degree: int, out_degree: int,
                  objective_values: Mapping[str, int],
                  objective_targets: Mapping[str, int],
                  objective_mask: Mapping[str, int]) -> List[float]:
    """A fixed-length feature vector for one node (spec-agnostic width)."""
    ntype = spec.node_types[node.type]
    features: List[float] = []
    for role in ROLE_ORDER:
        features.append(1.0 if ntype.role == role else 0.0)
    for semantics in SEMANTICS_ORDER:
        features.append(1.0 if ntype.semantics == semantics else 0.0)
    features.extend(hash_vector(node.type, TYPE_HASH_DIM, salt=1))
    features.extend(hash_vector(node.id, ID_HASH_DIM, salt=3))
    value = node.get(ntype.value_field) if ntype.value_field else None
    features.append(_normalised_number(value))
    features.append(1.0 if spec.dynamic_node_type == node.type else 0.0)
    is_objective = ntype.role == "objective"
    features.append(float(objective_values.get(node.id, 0)) if is_objective else 0.0)
    features.append(float(objective_targets.get(node.id, 0)) if is_objective else 0.0)
    features.append(float(objective_mask.get(node.id, 0)) if is_objective else 0.0)
    features.append(float(in_degree) / (1.0 + float(in_degree)))
    features.append(float(out_degree) / (1.0 + float(out_degree)))
    return features


def action_main_key(action: Action) -> str:
    """The role-prefixed id an action is keyed on (never a hardcoded name)."""
    if action.kind == "build":
        return "ax:%s" % action.axiom_id
    if action.kind == "combine":
        return "tag:%s" % action.tag_id
    if action.kind in ("set", "clear"):
        return "obj:%s" % action.objective_id
    return ""


# --------------------------------------------------------------------------
# The network
# --------------------------------------------------------------------------

class SizeInvariantNet:
    """A fixed-size graph-encoder policy; one scalar logit per legal action."""

    def __init__(self, hidden: int = 12) -> None:
        if hidden <= 0:
            raise ValueError("hidden must be positive")
        self.hidden = hidden
        self.node_dim = NODE_DIM
        self.global_dim = GLOBAL_DIM
        self.action_dim = len(KIND_ORDER) + ID_HASH_DIM + 1 + hidden
        offset = 0
        self.w_node = offset
        offset += hidden * self.node_dim
        self.b_node = offset
        offset += hidden
        self.w_self = offset
        offset += hidden * hidden
        self.w_neigh = offset
        offset += hidden * hidden
        self.b_msg = offset
        offset += hidden
        self.w_g = offset
        offset += hidden * self.global_dim
        self.b_g = offset
        offset += hidden
        self.w_a = offset
        offset += hidden * self.action_dim
        self.b_a = offset
        offset += hidden
        self.w_c = offset
        offset += hidden * hidden
        self.b_c = offset
        offset += hidden
        self.w_out = offset
        offset += hidden
        self.b_out = offset
        offset += 1
        self.size = offset
        #: Alias so callers that log ``net.in_dim`` keep working.
        self.in_dim = self.size

    # -- weight helpers ----------------------------------------------------
    def init(self, rng: random.Random, scale: float = 0.5) -> List[float]:
        return [rng.gauss(0.0, scale) for _ in range(self.size)]

    def uniform(self, genome: List[float]) -> None:
        for i in range(self.size):
            genome[i] = 0.0

    @staticmethod
    def _matvec_plus(genome: Sequence[float], offset: int, rows: int, cols: int,
                     x: Sequence[float], bias_offset: Optional[int]) -> List[float]:
        out = [0.0] * rows
        for j in range(rows):
            row = offset + j * cols
            total = genome[bias_offset + j] if bias_offset is not None else 0.0
            for i in range(cols):
                total += genome[row + i] * x[i]
            out[j] = total
        return out

    def _check(self, genome: Sequence[float]) -> None:
        if len(genome) != self.size:
            raise ValueError("genome has %d genes, expected %d"
                             % (len(genome), self.size))

    # -- graph encoder -----------------------------------------------------
    def encode_graph(self, genome: Sequence[float], env: DynamicEnv,
                     state: State, subgoal_mask: Sequence[int]
                     ) -> Tuple[List[float], List[float], Dict[str, List[float]]]:
        """Return (pooled node latent, global latent, node-id -> hidden)."""
        self._check(genome)
        spec = env.spec
        nodes = list(state.nodes)
        node_ids = [node.id for node in nodes]
        index = {node_id: i for i, node_id in enumerate(node_ids)}

        references: Dict[str, Tuple[str, ...]] = {}
        in_degree = {node_id: 0 for node_id in node_ids}
        out_degree = {node_id: 0 for node_id in node_ids}
        for node in nodes:
            refs = tuple(ref for ref in node_references(spec, node)
                         if ref in index)
            references[node.id] = refs
            out_degree[node.id] = len(refs)
            for ref in refs:
                in_degree[ref] += 1

        objective_ids = sorted(spec.objectives)
        mask_map = {oid: int(subgoal_mask[i]) for i, oid in enumerate(objective_ids)
                    if i < len(subgoal_mask)}
        objective_values = state.obj_map()
        objective_targets = dict(spec.objectives)

        # node encoder: h0 = tanh(W_node @ feature + b_node)
        hidden = self.hidden
        first: List[List[float]] = []
        for node in nodes:
            feature = node_features(spec, node, in_degree[node.id],
                                    out_degree[node.id], objective_values,
                                    objective_targets, mask_map)
            pre = self._matvec_plus(genome, self.w_node, hidden, self.node_dim,
                                    feature, self.b_node)
            first.append([math.tanh(value) for value in pre])

        # one message-passing round over the UNDIRECTED graph
        neighbours: Dict[str, List[str]] = {node_id: [] for node_id in node_ids}
        for node_id, refs in references.items():
            for ref in refs:
                neighbours[node_id].append(ref)
                neighbours[ref].append(node_id)
        bias = [genome[self.b_msg + j] for j in range(hidden)]
        second: List[List[float]] = []
        for node_id in node_ids:
            aggregate = [0.0] * hidden
            near = neighbours[node_id]
            if near:
                for ref in near:
                    vector = first[index[ref]]
                    for k in range(hidden):
                        aggregate[k] += vector[k]
                inverse = 1.0 / float(len(near))
                for k in range(hidden):
                    aggregate[k] *= inverse
            self_part = self._matvec_plus(genome, self.w_self, hidden, hidden,
                                          first[index[node_id]], None)
            neigh_part = self._matvec_plus(genome, self.w_neigh, hidden, hidden,
                                           aggregate, None)
            second.append([math.tanh(self_part[k] + neigh_part[k] + bias[k])
                           for k in range(hidden)])

        # masked mean pooling: the node count is never a width
        pooled = [0.0] * hidden
        for vector in second:
            for k in range(hidden):
                pooled[k] += vector[k]
        if second:
            inverse = 1.0 / float(len(second))
            for k in range(hidden):
                pooled[k] *= inverse

        # fixed global scalars (values scale with the spec, the width does not)
        global_input = [
            float(len(nodes)) / (1.0 + float(len(nodes))),
            float(state.step) / (1.0 + float(state.step)),
            float(len(state.active)) / (1.0 + float(len(state.active))),
            float(len(spec.objectives)) / (1.0 + float(len(spec.objectives))),
        ]
        global_pre = self._matvec_plus(genome, self.w_g, hidden, self.global_dim,
                                       global_input, self.b_g)
        global_latent = [math.tanh(value) for value in global_pre]

        node_latent = {node_id: second[i] for i, node_id in enumerate(node_ids)}
        return pooled, global_latent, node_latent

    def action_features(self, spec: Spec, action: Action,
                        node_latent: Mapping[str, List[float]]) -> List[float]:
        """A fixed-length feature vector for one action (spec-agnostic width)."""
        features = [0.0] * len(KIND_ORDER)
        if action.kind in KIND_ORDER:
            features[KIND_ORDER.index(action.kind)] = 1.0
        features.extend(hash_vector(action_main_key(action), ID_HASH_DIM, salt=2))
        arity = len(action.operands)
        features.append(float(arity) / (1.0 + float(arity)))
        referenced: List[str] = []
        if action.kind == "build":
            axiom = spec.axiom(action.axiom_id) if action.axiom_id else None
            tag_node = getattr(axiom, "tag_node", None) if axiom is not None else None
            if isinstance(tag_node, str):
                referenced.append(tag_node)
        elif action.kind == "combine":
            if isinstance(action.tag_id, str):
                referenced.append(action.tag_id)
        referenced.extend(action.operands)
        aggregate = [0.0] * self.hidden
        count = 0
        for node_id in referenced:
            vector = node_latent.get(node_id)
            if vector is None:
                continue
            count += 1
            for k in range(self.hidden):
                aggregate[k] += vector[k]
        if count:
            inverse = 1.0 / float(count)
            for k in range(self.hidden):
                aggregate[k] *= inverse
        features.extend(aggregate)
        return features

    def score(self, genome: Sequence[float], env: DynamicEnv, state: State,
              actions: Sequence[Action], subgoal_mask: Sequence[int]) -> List[float]:
        """One logit per legal action; the SAME genome scores any spec."""
        pooled, global_latent, node_latent = self.encode_graph(
            genome, env, state, subgoal_mask)
        hidden = self.hidden
        out: List[float] = []
        for action in actions:
            feature = self.action_features(env.spec, action, node_latent)
            action_pre = self._matvec_plus(genome, self.w_a, hidden,
                                           self.action_dim, feature, self.b_a)
            action_latent = [math.tanh(value) for value in action_pre]
            combined = [pooled[k] + global_latent[k] + action_latent[k]
                        for k in range(hidden)]
            combined_pre = self._matvec_plus(genome, self.w_c, hidden, hidden,
                                             combined, self.b_c)
            latent = [math.tanh(value) for value in combined_pre]
            score = genome[self.b_out]
            for k in range(hidden):
                score += genome[self.w_out + k] * latent[k]
            out.append(score)
        return out

    #: The trainer calls ``logits``; ``score`` is the descriptive alias.
    logits = score


def spec_independent_size(hidden: int = 12) -> int:
    """The genome length of :class:`SizeInvariantNet` (a constant)."""
    return SizeInvariantNet(hidden).size

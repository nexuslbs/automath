"""Feature builders for the llm_like module (design section 3.1, 3.3).

Every width here is a CONSTANT, independent of the spec: the node count is a set
size (never a width), the action count is a loop (never a width) and the id
vocabulary enters through a fixed signed feature hash. This mirrors the
size-invariant idea of ``evolution_trainer.size_invariant`` but returns plain
Python lists of the exact widths the torch model declares:

* node feature vector       : ``NODE_SCALARS + ID_HASH_DIM`` = 13 + 8 = 21,
* node-type index           : one of ``N_TYPES`` = 10 embedding rows,
* action feature vector     : kind one-hot 5 + id hash 8 + arity 1 = 14,
* action reference block    : mean of the referenced nodes' latents (width d),
* context token             : ``[state latent ; chosen action latent]`` (2d).

These functions are stdlib only; the torch tensors are built in ``model.py``.
"""

from __future__ import annotations

from typing import Dict, List, Mapping, Optional, Sequence, Tuple

from dynamic_env.engine import Action, DynamicEnv, State
from dynamic_env.spec import Spec

#: Action kinds in a fixed order. ``pop`` is the opt-in fifth kind; the model
#: always reserves its one-hot slot even when the environment does not emit it.
KIND_ORDER: Tuple[str, ...] = ("build", "combine", "set", "clear", "pop")
N_KINDS = len(KIND_ORDER)

#: Node roles and semantics, in a fixed order.
ROLE_ORDER: Tuple[str, ...] = ("value", "tag", "objective", "plain")
SEMANTICS_ORDER: Tuple[str, ...] = ("literal", "combination")

#: Fixed hashed id width (spec-agnostic; collisions allowed, documented).
ID_HASH_DIM = 8
#: Per-node scalar block: role(4) + semantics(2) + value(1) + dynamic(1)
#: + objective cur/target/mask(3) + in/out degree(2).
NODE_SCALARS = (len(ROLE_ORDER) + len(SEMANTICS_ORDER) + 1 + 1 + 3 + 2)
#: Node input width: scalars + id hash. The node TYPE is the embedding index.
NODE_IN = NODE_SCALARS + ID_HASH_DIM
#: Action input width without the referenced-node mean: kind + id hash + arity.
ACTION_BASE = N_KINDS + ID_HASH_DIM + 1
#: Node-type embedding cap (shipped specs use 1 or 4 types).
N_TYPES = 10


# --------------------------------------------------------------------------
# Deterministic signed feature hashing (same idea as size_invariant.hash_vector)
# --------------------------------------------------------------------------

def _fnv1a(text: str, salt: int = 0) -> int:
    h = 2166136261
    h = (h ^ (salt & 0xFFFFFFFF)) & 0xFFFFFFFF
    for ch in text:
        h ^= ord(ch)
        h = (h * 16777619) & 0xFFFFFFFF
    return h


def hash_vector(text: str, dim: int = ID_HASH_DIM, salt: int = 0) -> List[float]:
    """A deterministic, signed, fixed-length embedding of a name string."""
    vec = [0.0] * dim
    if not text:
        return vec
    digest = _fnv1a(text, salt)
    index = digest % dim
    sign = 1.0 if ((digest >> 16) & 1) == 0 else -1.0
    magnitude = 0.5 + float((digest >> 8) % 1000) / 2000.0
    vec[index] = sign * magnitude
    return vec


# --------------------------------------------------------------------------
# Node features
# --------------------------------------------------------------------------

def node_references(spec: Spec, node) -> Tuple[str, ...]:
    """The node ids a node references, read from its DECLARED field types."""
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


def node_scalars(spec: Spec, node, in_degree: int, out_degree: int,
                 objective_values: Mapping[str, int],
                 objective_targets: Mapping[str, int],
                 objective_mask: Mapping[str, int]) -> List[float]:
    """The 13 fixed scalars of one node (type is handled by the embedding)."""
    ntype = spec.node_types[node.type]
    features: List[float] = []
    for role in ROLE_ORDER:
        features.append(1.0 if ntype.role == role else 0.0)
    for semantics in SEMANTICS_ORDER:
        features.append(1.0 if ntype.semantics == semantics else 0.0)
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


def type_index_table(spec: Spec) -> Dict[str, int]:
    """Map each declared node type to a stable embedding index.

    Index 0 is reserved for "unknown/pad"; declared types get ``1 + position``
    in the sorted list, capped at ``N_TYPES - 1`` so the embedding never
    overflows even on a hypothetical spec with more than 9 declared types.
    """
    names = sorted(spec.node_types)
    return {name: min(1 + i, N_TYPES - 1) for i, name in enumerate(names)}


def state_inputs(env: DynamicEnv, state: State,
                 subgoal_mask: Sequence[int]) -> Dict[str, object]:
    """The self-contained per-state model inputs (all spec-agnostic widths).

    Returns ``node_ids``, ``node_features`` (N x 21), ``type_idx`` (N) and the
    undirected adjacency ``adj`` (N x N) read from the spec's own declared
    ``node`` / ``nodes`` fields.
    """
    spec = env.spec
    nodes = list(state.nodes)
    node_ids = [node.id for node in nodes]
    index = {node_id: i for i, node_id in enumerate(node_ids)}

    references: Dict[str, Tuple[str, ...]] = {}
    in_degree = {node_id: 0 for node_id in node_ids}
    out_degree = {node_id: 0 for node_id in node_ids}
    for node in nodes:
        refs = tuple(ref for ref in node_references(spec, node) if ref in index)
        references[node.id] = refs
        out_degree[node.id] = len(refs)
        for ref in refs:
            in_degree[ref] += 1

    objective_ids = sorted(spec.objectives)
    mask_map = {oid: int(subgoal_mask[i]) for i, oid in enumerate(objective_ids)
                if i < len(subgoal_mask)}
    objective_values = state.obj_map()
    objective_targets = dict(spec.objectives)
    tmap = type_index_table(spec)

    node_features: List[List[float]] = []
    for node in nodes:
        scalars = node_scalars(spec, node, in_degree[node.id], out_degree[node.id],
                               objective_values, objective_targets, mask_map)
        node_features.append(scalars + hash_vector(node.id, ID_HASH_DIM, salt=3))

    total = len(nodes)
    adjacency = [[0.0] * total for _ in range(total)]
    for node_id, refs in references.items():
        i = index[node_id]
        for ref in refs:
            j = index[ref]
            adjacency[i][j] = 1.0
            adjacency[j][i] = 1.0

    return {
        "node_ids": node_ids,
        "node_features": node_features,
        "type_idx": [tmap.get(node.type, 0) for node in nodes],
        "adj": adjacency,
    }


# --------------------------------------------------------------------------
# Action features
# --------------------------------------------------------------------------

def action_main_key(action: Action) -> str:
    """The role-prefixed id an action is keyed on (never a hardcoded name)."""
    if action.kind == "build":
        return "ax:%s" % action.axiom_id
    if action.kind == "combine":
        return "tag:%s" % action.tag_id
    if action.kind in ("set", "clear"):
        return "obj:%s" % action.objective_id
    return "pop"


def action_references(spec: Spec, action: Action) -> Tuple[str, ...]:
    """The node ids an action references (the same graph the policy reads)."""
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
    return tuple(referenced)


def action_features(spec: Spec, action: Action) -> List[float]:
    """The 14 fixed action scalars (kind one-hot + id hash + arity)."""
    features = [0.0] * N_KINDS
    if action.kind in KIND_ORDER:
        features[KIND_ORDER.index(action.kind)] = 1.0
    features.extend(hash_vector(action_main_key(action), ID_HASH_DIM, salt=2))
    arity = len(action.operands)
    features.append(float(arity) / (1.0 + float(arity)))
    return features


def action_inputs(env: DynamicEnv, state: State, actions: Sequence[Action],
                  node_ids: Sequence[str]) -> Dict[str, object]:
    """Per-action model inputs for one state.

    ``action_ref_local`` holds, for each action, the LOCAL node indices of the
    nodes it references, so the referenced-node mean can be computed from the
    state's node latents in :mod:`llm_like.model`.
    """
    index = {node_id: i for i, node_id in enumerate(node_ids)}
    features: List[List[float]] = []
    refs_local: List[List[int]] = []
    keys: List[str] = []
    for action in actions:
        features.append(action_features(env.spec, action))
        refs = action_references(env.spec, action)
        refs_local.append([index[ref] for ref in refs if ref in index])
        keys.append(action.key())
    return {
        "action_features": features,
        "action_ref_local": refs_local,
        "action_keys": keys,
    }

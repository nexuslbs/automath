"""Flavor A: target-conditioned feature encoder plus the tiny policy net.

The flavor-A genome is the flat weight vector of a one-hidden-layer ``tanh``
MLP (``PolicyNet``).  Every action score is::

    score(target, stack, action) = PolicyNet.forward(net, features(target, stack, action))

so the SAME parameters score actions for ANY target: the target enters only
through the feature vector.  The features are built exclusively from the node
structure and the reduction table (a token per semantic kind is its
``(tag, arity)`` row of ``REDUCTION_TABLE``); no per-target table and no
target identity is stored anywhere.

Feature layout (fixed length, stdlib only)::

    target : token counts (21 tokens + 1 unbuildable) + size + depth
    stack  : token counts (same 22) + depth + top-node arity/size/kind
    flag   : 1.0 when the top node is a sub-structure of the target
    action : arity + resulting-token one-hot (22) + KIND_GROUPS kind one-hot
"""

from __future__ import annotations

import math
import random
from typing import Dict, List, Optional, Sequence, Tuple

from .evolution import EVO_AXIOMS, KIND_GROUPS, SEMANTIC_OPS
from .nodes import Change, Group, Node, One, Zero

# --------------------------------------------------------------------------
# Token vocabulary: one token per semantic kind (by tag+arity) then the core
# node classes.  An extra slot counts Groups whose (tag, arity) has no build
# action (the structurally impossible / unbuildable class).
# --------------------------------------------------------------------------

SEM_NAMES: Tuple[str, ...] = tuple(o.name for o in SEMANTIC_OPS)
SEM_KIND: Dict[str, str] = {o.name: o.kind for o in SEMANTIC_OPS}
SEM_BY_KEY: Dict[Tuple[int, int], str] = {
    (o.tag, o.arity): o.name for o in SEMANTIC_OPS}

CORE_NAMES: Tuple[str, ...] = ("Zero", "One", "Change")
TOKEN_NAMES: Tuple[str, ...] = SEM_NAMES + CORE_NAMES
N_TOKENS = len(TOKEN_NAMES)
TOKEN_INDEX: Dict[str, int] = {n: i for i, n in enumerate(TOKEN_NAMES)}
IDX_UNBUILDABLE = N_TOKENS
COUNT_DIM = N_TOKENS + 1

KIND_ORDER: Tuple[str, ...] = tuple(KIND_GROUPS.keys())
NODE_KINDS: Tuple[str, ...] = KIND_ORDER + ("core", "unbuildable")
NODE_KIND_INDEX: Dict[str, int] = {k: i for i, k in enumerate(NODE_KINDS)}

# Core build actions (BUILD_ACTIONS) plus the 18 semantic kinds plus the
# pseudo action used to score a bundle start state.
ACTION_NAMES: Tuple[str, ...] = (
    "PushZero", "PushOne", "MakeChange", "MakeGroup2", "MakeGroup3",
    "MakeGroup4") + SEM_NAMES + ("__start__",)
N_ACTIONS = len(ACTION_NAMES)
ACTION_INDEX: Dict[str, int] = {n: i for i, n in enumerate(ACTION_NAMES)}
ACTION_KINDS: Tuple[str, ...] = KIND_ORDER + ("core", "start")
ACTION_KIND_INDEX: Dict[str, int] = {k: i for i, k in enumerate(ACTION_KINDS)}
START_ACTION = "__start__"

_ACTION_INFO: Dict[str, Tuple[int, str]] = {
    "PushZero": (0, "core"), "PushOne": (0, "core"),
    "MakeChange": (1, "core"), "MakeGroup2": (2, "core"),
    "MakeGroup3": (3, "core"), "MakeGroup4": (4, "core"),
}
for _o in SEMANTIC_OPS:
    _ACTION_INFO[_o.name] = (_o.arity, _o.kind)
_ACTION_INFO[START_ACTION] = (0, "start")

_ACTION_RESULT: Dict[str, Optional[int]] = {
    "PushZero": TOKEN_INDEX["Zero"], "PushOne": TOKEN_INDEX["One"],
    "MakeChange": TOKEN_INDEX["Change"],
    # MakeGroupK builds Group(operand_tag, ...): the resulting tag is dynamic,
    # so it lands in the unbuildable slot as a structural proxy.
    "MakeGroup2": IDX_UNBUILDABLE, "MakeGroup3": IDX_UNBUILDABLE,
    "MakeGroup4": IDX_UNBUILDABLE,
}
for _o in SEMANTIC_OPS:
    _ACTION_RESULT[_o.name] = TOKEN_INDEX[_o.name]
_ACTION_RESULT[START_ACTION] = None

TARGET_DIM = COUNT_DIM + 2                                  # 24
STACK_DIM = COUNT_DIM + 1 + 2 + len(NODE_KINDS)             # 32
ACTION_DIM = 1 + COUNT_DIM + len(ACTION_KINDS)              # 30
FLAG_DIM = 1


# --------------------------------------------------------------------------
# Node classification helpers (structure + reduction table only)
# --------------------------------------------------------------------------

def _tag_value(node: Node) -> Optional[int]:
    """Integer value of a Group tag node, or None when not evaluable."""
    try:
        value = EVO_AXIOMS.evaluate(node)
    except Exception:  # noqa: BLE001 - any axiom failure means unbuildable
        return None
    return value if isinstance(value, int) else None


def _token_kind(node: Node) -> Tuple[int, str]:
    if isinstance(node, Zero):
        return TOKEN_INDEX["Zero"], "core"
    if isinstance(node, One):
        return TOKEN_INDEX["One"], "core"
    if isinstance(node, Change):
        return TOKEN_INDEX["Change"], "core"
    if isinstance(node, Group):
        tv = _tag_value(node.tag)
        name = SEM_BY_KEY.get((tv, node.arity())) if tv is not None else None
        if name is not None:
            return TOKEN_INDEX[name], SEM_KIND[name]
        return IDX_UNBUILDABLE, "unbuildable"
    return IDX_UNBUILDABLE, "unbuildable"


def _arity(node: Node) -> int:
    if isinstance(node, Group):
        return node.arity()
    if isinstance(node, Change):
        return 1
    return 0


def _sem_size(node: Node) -> int:
    """Node count over the semantic structure (tag encodings excluded)."""
    total = 1
    if isinstance(node, Group):
        for item in node.items:
            total += _sem_size(item)
    elif isinstance(node, Change):
        total += _sem_size(node.child)
    return total


def _sem_depth(node: Node) -> int:
    best = 1
    if isinstance(node, Group):
        for item in node.items:
            best = max(best, 1 + _sem_depth(item))
    elif isinstance(node, Change):
        best = 1 + _sem_depth(node.child)
    return best


def _walk(node: Node, counts: List[float]) -> None:
    tok, _kind = _token_kind(node)
    counts[tok] += 1.0
    if isinstance(node, Group):
        for item in node.items:
            _walk(item, counts)
    elif isinstance(node, Change):
        _walk(node.child, counts)


def _subtree_canons(node: Node, out: Optional[set] = None) -> set:
    if out is None:
        out = set()
    out.add(node.canonical())
    if isinstance(node, Group):
        for item in node.items:
            _subtree_canons(item, out)
    elif isinstance(node, Change):
        _subtree_canons(node.child, out)
    return out


# --------------------------------------------------------------------------
# Feature blocks
# --------------------------------------------------------------------------

def target_features(target: Optional[Node]) -> List[float]:
    """Fixed-length structural summary of the target tree."""
    if target is None:
        return [0.0] * TARGET_DIM
    counts = [0.0] * COUNT_DIM
    _walk(target, counts)
    total = sum(counts) or 1.0
    out = [c / total for c in counts]
    out.append(min(total, 200.0) / 50.0)
    out.append(min(_sem_depth(target), 12) / 10.0)
    return out


def stack_features(stack: Sequence[Node]) -> List[float]:
    """Fixed-length structural summary of the current stack."""
    depth = len(stack)
    counts = [0.0] * COUNT_DIM
    for node in stack:
        _walk(node, counts)
    total = sum(counts) or 1.0
    out = [c / total for c in counts]
    out.append(min(depth, 8) / 6.0)
    if depth:
        top = stack[-1]
        _tok, kind = _token_kind(top)
        out.append(_arity(top) / 4.0)
        out.append(min(_sem_size(top), 60) / 50.0)
        kv = [0.0] * len(NODE_KINDS)
        kv[NODE_KIND_INDEX[kind]] = 1.0
        out.extend(kv)
    else:
        out.append(0.0)
        out.append(0.0)
        out.extend([0.0] * len(NODE_KINDS))
    return out


def action_features(action: str) -> List[float]:
    """Fixed-length encoding of an action (arity, resulting token, kind)."""
    arity, kind = _ACTION_INFO.get(action, (0, "start"))
    out = [arity / 4.0]
    one = [0.0] * COUNT_DIM
    result = _ACTION_RESULT.get(action)
    if result is not None:
        one[result] = 1.0
    out.extend(one)
    kv = [0.0] * len(ACTION_KINDS)
    kv[ACTION_KIND_INDEX.get(kind, ACTION_KIND_INDEX["start"])] = 1.0
    out.extend(kv)
    return out


def features(target: Optional[Node], stack: Sequence[Node],
             action: str) -> List[float]:
    """The full target-conditioned input vector for one (target, stack, action)."""
    flag = 0.0
    if target is not None and stack:
        if stack[-1].canonical() in _subtree_canons(target):
            flag = 1.0
    return (target_features(target) + stack_features(stack)
            + [flag] + action_features(action))


def feature_dim() -> int:
    return TARGET_DIM + STACK_DIM + FLAG_DIM + ACTION_DIM


# --------------------------------------------------------------------------
# The policy net: one tanh hidden layer, scalar output (a value/score)
# --------------------------------------------------------------------------

class PolicyNet:
    """Tiny one-hidden-layer MLP whose flat weights ARE the genome.

    Weight layout: ``[W1 (H*D), b1 (H), W2 (H), b2 (1)]`` where ``D = IN_DIM``
    and ``H = HIDDEN``; ``forward`` returns ``W2 . tanh(W1 x + b1) + b2``.
    """

    HIDDEN = 12
    IN_DIM = 0  # set below, once feature_dim() is known

    @classmethod
    def size(cls) -> int:
        return cls.HIDDEN * cls.IN_DIM + cls.HIDDEN + cls.HIDDEN + 1

    @classmethod
    def init(cls, rng: random.Random, sigma: float = 0.5) -> List[float]:
        return [rng.gauss(0.0, sigma) for _ in range(cls.size())]

    @classmethod
    def forward(cls, weights: Sequence[float], x: Sequence[float]) -> float:
        hidden = cls.HIDDEN
        dim = cls.IN_DIM
        pos = 0
        acts: List[float] = [0.0] * hidden
        for k in range(hidden):
            s = 0.0
            row = pos
            for j in range(dim):
                s += weights[row + j] * x[j]
            s += weights[row + dim]
            pos += dim + 1
            acts[k] = math.tanh(s)
        out = 0.0
        for k in range(hidden):
            out += weights[pos + k] * acts[k]
        out += weights[pos + hidden]
        return out


PolicyNet.IN_DIM = feature_dim()
NET_SIZE = PolicyNet.size()


def net_size() -> int:
    return PolicyNet.size()


def net_forward(weights: Sequence[float], x: Sequence[float]) -> float:
    return PolicyNet.forward(weights, x)


def net_init(rng: random.Random, sigma: float = 0.5) -> List[float]:
    return PolicyNet.init(rng, sigma)


def zero_net() -> List[float]:
    return [0.0] * PolicyNet.size()

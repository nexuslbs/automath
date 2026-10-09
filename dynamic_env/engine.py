"""The GENERIC dynamic-nodes engine.

Nothing in this module is specific to a shipped spec: no spec id, no node-type
name, no tag value and no axiom id appears here. The engine only interprets the
schema and axioms declared by a :class:`dynamic_env.spec.Spec`. Environments are
DATA (a JSON spec file); the engine is the fixed interpreter.

State
-----
A state is a finite store of typed nodes, the current values of the elementar
1/0 objective nodes, the currently active axiom ids and the ids of the dynamic
axioms already fired. A combination node has no intrinsic meaning: its value is
computed by the active combine axiom keyed on ``(tag value, arity)``.

Actions (all derived per spec)
------------------------------
* ``build``   - apply a build axiom to operands, constructing a combination node
                from a spec-declared tag node. This is the "apply an axiom/rule"
                action.
* ``combine`` - combine arbitrary existing nodes under an arbitrary existing tag
                node into the dynamic grouping node. This is the "combine nodes
                into a dynamic grouping node" action.
* ``set`` / ``clear`` - set or clear one elementar 1/0 objective node.

Strict laws
-----------
``step`` is a pure function of ``(spec, state, action)``: the same inputs give
byte-identical next states (assertable via ``State.identity()``). The only state
changes come from the interpreter primitives declared in the spec's axioms; no
trainer hook mutates the state. See ``docs/evolution/STRICT_LAWS.md``.
"""

from __future__ import annotations

import itertools
from dataclasses import dataclass, replace
from typing import Any, Callable, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

from .spec import (
    BuildAxiom,
    CombineAxiom,
    DynamicAxiom,
    Node,
    NodeType,
    Spec,
)


class EvalError(Exception):
    """No active axiom gives this combination a value in this state."""


class IllegalAction(Exception):
    """An action that is not in ``legal_actions(state)``."""


# --------------------------------------------------------------------------
# State
# --------------------------------------------------------------------------

@dataclass(frozen=True)
class State:
    nodes: Tuple[Node, ...] = ()
    objectives: Tuple[Tuple[str, int], ...] = ()
    active: Tuple[str, ...] = ()
    fired: Tuple[str, ...] = ()
    step: int = 0
    next_id: int = 0
    history: Tuple[str, ...] = ()
    last_error: Optional[str] = None

    def node_map(self) -> Dict[str, Node]:
        return {n.id: n for n in self.nodes}

    def obj_map(self) -> Dict[str, int]:
        return dict(self.objectives)

    def identity(self) -> str:
        """A hashable key of the LOGICAL state (step/history excluded)."""
        nodes = ";".join(n.canonical() for n in self.nodes)
        objs = ",".join("%s=%d" % (k, v) for k, v in self.objectives)
        active = ",".join(self.active)
        fired = ",".join(self.fired)
        return "N[%s]O[%s]A[%s]F[%s]" % (nodes, objs, active, fired)

    def canonical(self) -> str:
        """The full state including step/history (determinism checks)."""
        return self.identity() + "S[%d]H[%s]" % (self.step, ",".join(self.history))


def _sorted_nodes(nodes: Mapping[str, Node]) -> Tuple[Node, ...]:
    return tuple(nodes[k] for k in sorted(nodes))


def _sorted_items(mapping: Mapping[str, int]) -> Tuple[Tuple[str, int], ...]:
    return tuple(sorted(mapping.items()))


def _normalized_active(active: Iterable[str]) -> Tuple[str, ...]:
    return tuple(sorted(set(active)))


# --------------------------------------------------------------------------
# The interpreter: node values from the active combine axioms
# --------------------------------------------------------------------------

def _truthy(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return value != 0
    if isinstance(value, (tuple, list, str)):
        return len(value) > 0
    return bool(value)


def _compare(cmp_: str, left: Any, right: Any) -> bool:
    if cmp_ == "eq":
        return left == right
    if cmp_ == "ne":
        return left != right
    if cmp_ == "lt":
        return left < right
    if cmp_ == "le":
        return left <= right
    if cmp_ == "gt":
        return left > right
    if cmp_ == "ge":
        return left >= right
    raise EvalError("unknown comparison %r" % cmp_)


def _apply_op(op: str, values: Sequence[Any], params: Mapping[str, Any]) -> Any:
    """The fixed primitive instruction set; specs SELECT from it, never extend it."""
    if op == "sum":
        return values[0] + values[1]
    if op == "sub":
        return max(values[0] - values[1], 0)
    if op == "mul":
        return values[0] * values[1]
    if op == "div":
        if values[1] == 0:
            raise EvalError("division by zero")
        return values[0] // values[1]
    if op == "mod":
        if values[1] == 0:
            raise EvalError("modulo by zero")
        return values[0] % values[1]
    if op == "neg":
        return -values[0]
    if op == "succ":
        return values[0] + 1
    if op == "eq":
        return values[0] == values[1]
    if op == "lt":
        return values[0] < values[1]
    if op == "gt":
        return values[0] > values[1]
    if op == "le":
        return values[0] <= values[1]
    if op == "ge":
        return values[0] >= values[1]
    if op == "and":
        return _truthy(values[0]) and _truthy(values[1])
    if op == "or":
        return _truthy(values[0]) or _truthy(values[1])
    if op == "not":
        return not _truthy(values[0])
    if op == "identity":
        return values[0]
    if op == "constant":
        return params.get("value")
    if op == "list":
        return tuple(values)
    if op == "head":
        if len(values[0]) == 0:
            raise EvalError("head of an empty sequence")
        return values[0][0]
    if op == "tail":
        if len(values[0]) == 0:
            raise EvalError("tail of an empty sequence")
        return tuple(values[0][1:])
    if op == "len":
        return len(values[0])
    if op == "min":
        return min(values)
    if op == "max":
        return max(values)
    if op == "cat":
        return values[0] + values[1]
    raise EvalError("unknown op %r" % op)


def evaluate(
    spec: Spec,
    state: State,
    node_id: str,
    memo: Optional[Dict[str, Any]] = None,
) -> Any:
    """Compute a node's value from the active combine axioms (pure)."""
    if memo is not None and node_id in memo:
        return memo[node_id]
    node_map = state.node_map()
    node = node_map.get(node_id)
    if node is None:
        raise EvalError("unknown node %r" % node_id)
    ntype = spec.node_types[node.type]
    if ntype.semantics == "literal":
        value = node.get(ntype.value_field)
        if memo is not None:
            memo[node_id] = value
        return value
    # combination
    tag_id = node.get("tag")
    items = tuple(node.get("items") or ())
    if tag_id not in node_map:
        raise EvalError("combination %r has an unknown tag node %r" % (node_id, tag_id))
    tag_value = evaluate(spec, state, tag_id, memo)
    axiom = spec.combine_axiom(tag_value, len(items))
    if axiom is None or axiom.id not in state.active:
        raise EvalError(
            "no active axiom for tag=%r arity=%d" % (tag_value, len(items))
        )
    if axiom.op == "if":
        # Real control flow: only the chosen branch is evaluated.
        if len(items) != 3:
            raise EvalError("'if' needs arity 3, got %d" % len(items))
        cond = evaluate(spec, state, items[0], memo)
        branch = items[1] if _truthy(cond) else items[2]
        value = evaluate(spec, state, branch, memo)
        if memo is not None:
            memo[node_id] = value
        return value
    values = [evaluate(spec, state, item, memo) for item in items]
    try:
        value = _apply_op(axiom.op, values, axiom.params)
    except EvalError:
        raise
    except Exception as exc:  # noqa: BLE001 - an unusable value is "no meaning"
        raise EvalError("op %r failed on %r: %r" % (axiom.op, values, exc))
    if memo is not None:
        memo[node_id] = value
    return value


def evaluable_operands(spec: Spec, state: State) -> Tuple[str, ...]:
    """Node ids usable as operands.

    A node is an operand when it carries a value (role ``value``) or is itself a
    combination node (so built expressions compose). Tag nodes and objective
    nodes are never operands.
    """
    ids: List[str] = []
    for node in state.nodes:
        ntype = spec.node_types[node.type]
        if ntype.role == "value" or ntype.semantics == "combination":
            ids.append(node.id)
    return tuple(ids)


# --------------------------------------------------------------------------
# Conditions and effects (the declarative layer)
# --------------------------------------------------------------------------

def check_condition(spec: Spec, state: State, cond: Mapping[str, Any]) -> bool:
    op = cond.get("op")
    if op == "always":
        return True
    if op == "objective_is":
        return state.obj_map().get(cond.get("id")) == cond.get("value")
    if op == "all_objectives_match":
        return goal_reached(spec, state)
    if op == "exists_value":
        want = cond.get("value")
        cmp_ = cond.get("cmp", "eq")
        only_combination = bool(cond.get("combination_only", False))
        want_type = cond.get("node_type")
        memo: Dict[str, Any] = {}
        for node in state.nodes:
            ntype = spec.node_types[node.type]
            if only_combination and ntype.semantics != "combination":
                continue
            if want_type is not None and node.type != want_type:
                continue
            try:
                value = evaluate(spec, state, node.id, memo)
            except EvalError:
                continue
            if _compare(cmp_, value, want):
                return True
        return False
    if op == "node_type_count":
        count = sum(1 for n in state.nodes if n.type == cond.get("type"))
        return _compare(cond.get("cmp", "eq"), count, cond.get("value"))
    if op == "count_nodes":
        return _compare(cond.get("cmp", "eq"), len(state.nodes), cond.get("value"))
    if op == "axiom_active":
        return cond.get("id") in state.active
    if op == "and":
        return all(check_condition(spec, state, a) for a in cond.get("args", ()))
    if op == "or":
        return any(check_condition(spec, state, a) for a in cond.get("args", ()))
    if op == "not":
        return not check_condition(spec, state, cond.get("arg") or {"op": "always"})
    raise EvalError("unknown condition %r" % op)


def _node_from_effect(spec: Spec, raw: Mapping[str, Any]) -> Node:
    node = Node.from_dict("__effect__", raw)
    if node.type not in spec.node_types:
        raise EvalError("effect declares unknown node type %r" % node.type)
    return node


def apply_effect(spec: Spec, state: State, effect: Mapping[str, Any]) -> State:
    op = effect.get("op")
    if op in ("activate_axiom", "deactivate_axiom"):
        axiom_id = effect.get("id")
        if spec.axiom(axiom_id) is None:
            raise EvalError("effect names unknown axiom %r" % axiom_id)
        if op == "activate_axiom":
            active = _normalized_active(tuple(state.active) + (axiom_id,))
        else:
            active = tuple(a for a in state.active if a != axiom_id)
        return replace(state, active=_normalized_active(active))
    if op == "add_node":
        node = _node_from_effect(spec, effect.get("node"))
        node = replace(node, id=str(effect.get("id")))
        nodes = state.node_map()
        if node.id in nodes:
            return state  # deterministic no-op: the id is already present
        nodes[node.id] = node
        return replace(state, nodes=_sorted_nodes(nodes))
    if op == "rewrite_node":
        node = _node_from_effect(spec, effect.get("node"))
        node = replace(node, id=str(effect.get("id")))
        nodes = state.node_map()
        if node.id not in nodes:
            return state
        nodes[node.id] = node
        return replace(state, nodes=_sorted_nodes(nodes))
    if op == "remove_node":
        node_id = str(effect.get("id"))
        nodes = state.node_map()
        if node_id not in nodes:
            return state
        del nodes[node_id]
        return replace(state, nodes=_sorted_nodes(nodes))
    if op == "set_objective":
        objs = state.obj_map()
        oid = str(effect.get("id"))
        if oid not in objs:
            raise EvalError("effect sets unknown objective %r" % oid)
        objs[oid] = int(effect.get("value"))
        return replace(state, objectives=_sorted_items(objs))
    raise EvalError("unknown effect %r" % op)


def apply_effects(spec: Spec, state: State, effects: Sequence[Mapping[str, Any]]) -> State:
    for effect in effects:
        state = apply_effect(spec, state, effect)
    return state


def fire_dynamic_axioms(spec: Spec, state: State) -> State:
    """Fire every active dynamic axiom whose condition holds, in id order.

    A dynamic axiom fires AT MOST ONCE per episode by default (``once``); the
    firing set is a pure function of the current state, so determinism holds.
    Axioms activated by an earlier effect but sorted before it fire on the NEXT
    step (one ordered pass), which is also deterministic.
    """
    for axiom in sorted(spec.dynamic_axioms, key=lambda a: a.id):
        if axiom.id not in state.active:
            continue
        if axiom.once and axiom.id in state.fired:
            continue
        if not check_condition(spec, state, axiom.when):
            continue
        state = apply_effects(spec, state, axiom.effect)
        state = replace(state, fired=_normalized_active(tuple(state.fired) + (axiom.id,)))
    return state


# --------------------------------------------------------------------------
# Actions
# --------------------------------------------------------------------------

@dataclass(frozen=True)
class Action:
    kind: str  # build | combine | set | clear
    axiom_id: Optional[str] = None
    tag_id: Optional[str] = None
    operands: Tuple[str, ...] = ()
    objective_id: Optional[str] = None

    def key(self) -> str:
        if self.kind == "build":
            return "build:%s:%s" % (self.axiom_id, "+".join(self.operands))
        if self.kind == "combine":
            return "combine:%s:%s" % (self.tag_id, "+".join(self.operands))
        if self.kind == "set":
            return "set:%s" % self.objective_id
        if self.kind == "clear":
            return "clear:%s" % self.objective_id
        return self.kind

    def describe(self) -> str:
        return self.key()


def _guard_ok(spec: Spec, state: State, objective_id: str, value: int) -> bool:
    if value == 0:
        return True
    guard = spec.guards.get(objective_id)
    if guard is None:
        return True
    return check_condition(spec, state, guard)


def legal_actions(spec: Spec, state: State) -> Tuple[Action, ...]:
    """All legal actions from a state, in a deterministic canonical order."""
    actions: List[Action] = []
    operands = evaluable_operands(spec, state)
    node_map = state.node_map()

    for axiom in sorted(spec.build_axioms, key=lambda a: a.id):
        if axiom.id not in state.active:
            continue
        eligible = tuple(
            oid for oid in operands
            if axiom.operand_type is None
            or node_map[oid].type == axiom.operand_type
        )
        if axiom.arity == 0:
            actions.append(Action(kind="build", axiom_id=axiom.id, operands=()))
        elif len(eligible) >= axiom.arity:
            for combo in itertools.permutations(eligible, axiom.arity):
                actions.append(
                    Action(kind="build", axiom_id=axiom.id, operands=tuple(combo))
                )

    if spec.dynamic_node_type is not None:
        tag_ids = tuple(
            n.id for n in state.nodes
            if spec.node_types[n.type].role == "tag"
        )
        for tag_id in tag_ids:
            for arity in range(1, spec.max_combine_arity + 1):
                if len(operands) < arity:
                    continue
                for combo in itertools.permutations(operands, arity):
                    actions.append(
                        Action(kind="combine", tag_id=tag_id, operands=tuple(combo))
                    )

    objs = state.obj_map()
    for oid in sorted(objs):
        if objs[oid] != 1 and _guard_ok(spec, state, oid, 1):
            actions.append(Action(kind="set", objective_id=oid))
        if objs[oid] != 0:
            actions.append(Action(kind="clear", objective_id=oid))

    actions.sort(key=lambda a: a.key())
    # De-duplicate identical keys (impossible by construction, but cheap).
    seen = set()
    unique: List[Action] = []
    for action in actions:
        key = action.key()
        if key in seen:
            continue
        seen.add(key)
        unique.append(action)
    return tuple(unique)


def is_legal(spec: Spec, state: State, action: Action) -> bool:
    return action in legal_actions(spec, state)


# --------------------------------------------------------------------------
# Stepping
# --------------------------------------------------------------------------

@dataclass(frozen=True)
class StepResult:
    state: State
    reward: float
    done: bool
    info: Mapping[str, Any]


def _build_node(spec: Spec, state: State, action: Action) -> Node:
    if action.kind == "build":
        axiom = spec.axiom(action.axiom_id)
        node_type = axiom.node_type
        tag_id = axiom.tag_node
    else:
        node_type = spec.dynamic_node_type
        tag_id = action.tag_id
    node = Node(
        id="g%d" % state.next_id,
        type=node_type,
        fields=(("tag", tag_id), ("items", tuple(action.operands))),
    )
    return node


def goal_reached(spec: Spec, state: State) -> bool:
    objs = state.obj_map()
    for oid, target in spec.objectives.items():
        if objs.get(oid) != target:
            return False
    return True


def _step_internal(spec: Spec, state: State, action: Action
                   ) -> Tuple[State, Dict[str, Any]]:
    nodes = state.node_map()
    objectives = state.obj_map()
    next_id = state.next_id
    info: Dict[str, Any] = {"action": action.key(), "fired": ()}

    if action.kind in ("build", "combine"):
        node = _build_node(spec, state, action)
        nodes[node.id] = node
        next_id += 1
        info["added_node"] = node.id
    elif action.kind == "set":
        objectives[action.objective_id] = 1
    elif action.kind == "clear":
        objectives[action.objective_id] = 0
    else:  # pragma: no cover - legal_actions only emits the four kinds
        raise IllegalAction("unknown action kind %r" % action.kind)

    new_state = State(
        nodes=_sorted_nodes(nodes),
        objectives=_sorted_items(objectives),
        active=state.active,
        fired=state.fired,
        step=state.step + 1,
        next_id=next_id,
        history=state.history + (action.key(),),
        last_error=None,
    )
    fired_before = set(new_state.fired)
    new_state = fire_dynamic_axioms(spec, new_state)
    info = dict(info)
    info["fired"] = tuple(sorted(set(new_state.fired) - fired_before))
    info["goal"] = goal_reached(spec, new_state)
    return new_state, info


def step(spec: Spec, state: State, action: Action) -> StepResult:
    """Apply a LEGAL action; raise :class:`IllegalAction` otherwise."""
    if not is_legal(spec, state, action):
        raise IllegalAction(action.key())
    result, info = _step_internal(spec, state, action)
    done = bool(info["goal"])
    reward = spec.goal_reward if done else -spec.step_cost
    return StepResult(state=result, reward=reward, done=done, info=info)


def try_step(spec: Spec, state: State, action: Action) -> Tuple[Optional[StepResult], Optional[str]]:
    """Discard an illegal action deterministically: ``(None, reason)``.

    The caller keeps the original state, so a wrong step leaves the episode
    exactly where it was.
    """
    if not is_legal(spec, state, action):
        return None, "illegal action not in the action space: %s" % action.key()
    return step(spec, state, action), None


# --------------------------------------------------------------------------
# The environment facade
# --------------------------------------------------------------------------

class DynamicEnv:
    """A deterministic environment built from a :class:`Spec` (pure DATA)."""

    def __init__(self, spec: Spec) -> None:
        self.spec = spec

    @staticmethod
    def from_spec_file(path: str) -> "DynamicEnv":
        from .spec import load_spec

        return DynamicEnv(load_spec(path))

    def reset(self) -> State:
        nodes = dict(self.spec.nodes)
        # The INITIAL value of each elementar objective node is its declared
        # literal field; the spec's `objectives` map is the TARGET assignment.
        objectives: Dict[str, int] = {}
        for oid in self.spec.objectives:
            node = nodes[oid]
            ntype = self.spec.node_types[node.type]
            objectives[oid] = int(node.get(ntype.value_field))
        active = self.spec.initial_active()
        state = State(
            nodes=_sorted_nodes(nodes),
            objectives=_sorted_items(objectives),
            active=active,
            fired=(),
            step=0,
            next_id=0,
            history=(),
        )
        return fire_dynamic_axioms(self.spec, state)

    def legal_actions(self, state: State) -> Tuple[Action, ...]:
        return legal_actions(self.spec, state)

    def is_legal(self, state: State, action: Action) -> bool:
        return is_legal(self.spec, state, action)

    def step(self, state: State, action: Action) -> StepResult:
        return step(self.spec, state, action)

    def try_step(self, state: State, action: Action) -> Tuple[Optional[StepResult], Optional[str]]:
        return try_step(self.spec, state, action)

    def goal_reached(self, state: State) -> bool:
        return goal_reached(self.spec, state)

    def rollout(self, actions: Sequence[Action], state: Optional[State] = None
                ) -> Tuple[State, List[str]]:
        """Apply actions, DISCARDING illegal ones (deterministic)."""
        current = state if state is not None else self.reset()
        discarded: List[str] = []
        for action in actions:
            result, reason = self.try_step(current, action)
            if result is None:
                discarded.append("%s (%s)" % (action.key(), reason))
            else:
                current = result.state
        return current, discarded

    def objective_vector(self, state: State) -> Tuple[int, ...]:
        objs = state.obj_map()
        return tuple(objs[oid] for oid in sorted(self.spec.objectives))

    def target_vector(self) -> Tuple[int, ...]:
        return tuple(self.spec.objectives[oid] for oid in sorted(self.spec.objectives))

    def state_vector(self, state: State) -> Tuple[Any, ...]:
        """The interface Unit B trains on: objective flags + target + structure.

        The vector is fixed-length for a given spec and always exposes the
        elementar 1/0 objective nodes AND the target assignment, so the policy
        can be target-conditioned from the first observation.
        """
        type_names = tuple(sorted(self.spec.node_types))
        type_counts = tuple(
            sum(1 for n in state.nodes if n.type == name) for name in type_names
        )
        fingerprint = 0
        for ch in state.identity():
            fingerprint = (fingerprint * 131 + ord(ch)) % (2 ** 61 - 1)
        return (
            self.objective_vector(state)
            + self.target_vector()
            + type_counts
            + (len(state.nodes), state.step, fingerprint)
        )

    def observation(self, state: State) -> Mapping[str, Any]:
        return {
            "spec_id": self.spec.spec_id,
            "objectives": dict(state.objectives),
            "target": dict(self.spec.objectives),
            "vector": self.state_vector(state),
            "node_count": len(state.nodes),
            "step": state.step,
            "goal": self.goal_reached(state),
        }


# --------------------------------------------------------------------------
# Deterministic planning over the derived action space
# --------------------------------------------------------------------------

def _make_env(spec: Spec) -> DynamicEnv:
    return spec if isinstance(spec, DynamicEnv) else DynamicEnv(spec)


def minimal_length(spec: Spec, max_depth: Optional[int] = None) -> Optional[int]:
    """BFS length of the shortest action word reaching the goal (or ``None``)."""
    env = _make_env(spec)
    limit = max_depth if max_depth is not None else env.spec.max_steps
    start = env.reset()
    if env.goal_reached(start):
        return 0
    frontier = [start]
    seen = {start.identity()}
    for depth in range(1, limit + 1):
        nxt: List[State] = []
        for state in frontier:
            for action in env.legal_actions(state):
                result = env.step(state, action)
                ident = result.state.identity()
                if ident in seen:
                    continue
                seen.add(ident)
                if result.done:
                    return depth
                nxt.append(result.state)
        if not nxt:
            return None
        frontier = nxt
    return None


def count_goal_words(
    spec: Spec,
    length: int,
    cap: int = 2,
) -> Tuple[int, List[Tuple[str, ...]]]:
    """Exhaustively enumerate legal words of exactly ``length`` that reach goal.

    No state deduplication is used, so the count is an exact PATH count (two
    different words reaching the same goal state are counted twice). The search
    stops as soon as ``cap`` witnesses are found; the returned count is then the
    capped count and the witnesses are exact.
    """
    env = _make_env(spec)
    witnesses: List[Tuple[str, ...]] = []
    count = 0

    def visit(state: State, word: Tuple[str, ...]) -> None:
        nonlocal count
        if count >= cap:
            return
        if len(word) == length:
            if env.goal_reached(state):
                count += 1
                witnesses.append(word)
            return
        for action in env.legal_actions(state):
            result = env.step(state, action)
            visit(result.state, word + (action.key(),))
            if count >= cap:
                return

    visit(env.reset(), ())
    return count, witnesses


def bfs_minimal_word(spec: Spec, max_depth: Optional[int] = None
                     ) -> Optional[Tuple[str, ...]]:
    """The first shortest action word reaching the goal (deterministic)."""
    env = _make_env(spec)
    limit = max_depth if max_depth is not None else env.spec.max_steps
    start = env.reset()
    if env.goal_reached(start):
        return ()
    frontier: List[State] = [start]
    seen = {start.identity()}
    for _ in range(limit):
        nxt: List[State] = []
        for state in frontier:
            for action in env.legal_actions(state):
                result = env.step(state, action)
                ident = result.state.identity()
                if ident in seen:
                    continue
                seen.add(ident)
                if result.done:
                    return result.state.history
                nxt.append(result.state)
        if not nxt:
            return None
        frontier = nxt
    return None

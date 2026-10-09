"""The minimal-node environment: state, actions, goals, planner, uniqueness.

State
-----
A state is a stack of nodes (a partial forest). The environment starts from an
initial stack and each action consumes the top ``arity`` nodes and pushes the
single node they construct. Therefore every action increases the total node
count by exactly one: after ``L`` actions the system holds exactly ``L`` nodes,
so a target tree of ``N`` nodes can only be reached in exactly ``N`` actions.

Actions
-------
``PushZero`` / ``PushOne`` are the nullary facts; ``MakeChange`` is the unary
action; ``MakeGroupK`` applies the dynamic grouping node to a tag plus ``K-1``
operands. Actions are applications of the four node TYPES; they do not add
node types.

Goals
-----
* ``FixedStateGoal(target)`` - the reference's simplest goal: the fixed target
  state is reached iff the stack is exactly ``(target,)``.
* ``ExprGoal(build)`` - the dynamic/pattern goal: achieved iff the axiom
  evaluator makes the predicate expression on the current node true.

Determinism
-----------
``plan`` is the unique post-order construction of a fixed target; ``count_
solutions`` exhaustively enumerates all action words of the exact required
length (which is the only possible length) and counts how many reach the goal.
"""

from __future__ import annotations

import itertools
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Sequence, Tuple

from .axioms import AxiomSet
from .nodes import Change, Group, Node, One, Zero, size


@dataclass(frozen=True)
class Action:
    name: str
    arity: int
    build: Callable[[Tuple[Node, ...]], Node]


BUILD_ACTIONS: Tuple[Action, ...] = (
    Action("PushZero", 0, lambda a: Zero()),
    Action("PushOne", 0, lambda a: One()),
    Action("MakeChange", 1, lambda a: Change(a[0])),
    Action("MakeGroup2", 2, lambda a: Group(a[0], (a[1],))),
    Action("MakeGroup3", 3, lambda a: Group(a[0], (a[1], a[2]))),
    Action("MakeGroup4", 4, lambda a: Group(a[0], (a[1], a[2], a[3]))),
)
ACTION_BY_NAME: Dict[str, Action] = {a.name: a for a in BUILD_ACTIONS}


@dataclass(frozen=True)
class State:
    stack: Tuple[Node, ...] = ()
    cost: int = 0
    history: Tuple[str, ...] = ()
    last_error: Optional[str] = None

    def canonical(self) -> str:
        return "[" + ", ".join(n.canonical() for n in self.stack) + "]"


@dataclass(frozen=True)
class FailureFeedback:
    state: str
    expected: str
    actual: str
    reason: str

    def render(self) -> str:
        return (
            "state=%s expected=%s actual=%s reason=%s"
            % (self.state, self.expected, self.actual, self.reason)
        )


class FixedStateGoal:
    """The simplest (reference) goal: reach EXACTLY this fixed state/tree."""

    def __init__(self, target: Node) -> None:
        self.target = target

    def achieved(self, state: State) -> bool:
        return state.stack == (self.target,)

    def describe(self) -> str:
        return "fixed-state goal target=" + self.target.canonical()

    def feedback(self, state: State) -> FailureFeedback:
        return FailureFeedback(
            state=state.canonical(),
            expected="(" + self.target.canonical() + ",)",
            actual=state.canonical(),
            reason="goal target is a fixed state; stack does not equal it",
        )


class ExprGoal:
    """A dynamic/pattern goal: a predicate expression over the current node."""

    def __init__(self, build: Callable[[Node], Node], axioms: AxiomSet,
                 name: str = "pattern") -> None:
        self.build = build
        self.axioms = axioms
        self.name = name

    def achieved(self, state: State) -> bool:
        if len(state.stack) != 1:
            return False
        try:
            expr = self.build(state.stack[0])
            return self.axioms.truthy(expr)
        except Exception:
            return False

    def describe(self) -> str:
        return "pattern goal " + self.name

    def feedback(self, state: State) -> FailureFeedback:
        actual = state.canonical()
        if len(state.stack) == 1:
            try:
                actual = str(self.axioms.evaluate(self.build(state.stack[0])))
            except Exception as exc:  # noqa: BLE001 - structured feedback
                actual = "unevaluable: %r" % (exc,)
        return FailureFeedback(
            state=state.canonical(),
            expected="predicate true",
            actual=actual,
            reason="pattern goal not satisfied",
        )


class MinimalEnv:
    """A tiny deterministic environment over the four node types."""

    def __init__(
        self,
        goal,
        axioms: Optional[AxiomSet] = None,
        initial_stack: Sequence[Node] = (),
        max_actions: int = 16,
    ) -> None:
        self.goal = goal
        self.axioms = axioms or AxiomSet()
        self.initial_stack = tuple(initial_stack)
        self.max_actions = max_actions

    def reset(self) -> State:
        return State(stack=self.initial_stack, cost=0, history=())

    def step(self, state: State, action_name: str) -> State:
        action = ACTION_BY_NAME.get(action_name)
        if action is None:
            return State(
                stack=state.stack,
                cost=state.cost + 1,
                history=state.history + (action_name,),
                last_error="unknown action %r" % (action_name,),
            )
        if len(state.stack) < action.arity:
            return State(
                stack=state.stack,
                cost=state.cost + 1,
                history=state.history + (action_name,),
                last_error="action %s needs %d nodes, stack has %d"
                % (action_name, action.arity, len(state.stack)),
            )
        if action.arity == 0:
            popped: Tuple[Node, ...] = ()
        else:
            popped = state.stack[-action.arity:]
        built = action.build(popped)
        new_stack = state.stack[: len(state.stack) - action.arity] + (built,)
        return State(
            stack=new_stack,
            cost=state.cost + 1,
            history=state.history + (action_name,),
            last_error=None,
        )

    def run(self, names: Sequence[str]) -> State:
        state = self.reset()
        for name in names:
            state = self.step(state, name)
        return state

    def goal_achieved(self, state: State) -> bool:
        return self.goal.achieved(state)

    def feedback(self, state: State) -> FailureFeedback:
        return self.goal.feedback(state)


# --------------------------------------------------------------------------
# Deterministic planning and exhaustive uniqueness counting
# --------------------------------------------------------------------------

def plan(target: Node, initial_stack: Sequence[Node] = ()) -> List[str]:
    """The unique post-order build sequence for a fixed target.

    From the empty stack the sequence is the post-order walk of the target
    tree, and it is the ONLY solution (each constructor has fixed arity). With
    a non-empty ``initial_stack`` the word is found by ``bfs_plan`` instead.
    """
    if initial_stack:
        env = MinimalEnv(FixedStateGoal(target), initial_stack=initial_stack,
                         max_actions=size(target) + 4)
        word = bfs_plan(env)
        if word is None:
            raise ValueError(
                "initial stack does not compose with target: %r"
                % (tuple(initial_stack),)
            )
        return word

    names: List[str] = []

    def build(node: Node) -> None:
        if isinstance(node, Change):
            build(node.child)
            names.append("MakeChange")
        elif isinstance(node, Group):
            build(node.tag)
            for item in node.items:
                build(item)
            names.append("MakeGroup%d" % (node.arity() + 1))
        else:  # Zero or One
            names.append("PushZero" if isinstance(node, Zero) else "PushOne")

    build(target)
    return names


def simulate(initial_stack: Sequence[Node], names: Sequence[str]
             ) -> Tuple[Tuple[Node, ...], Optional[str]]:
    """Run an action word; return the final stack and the first error."""
    stack: Tuple[Node, ...] = tuple(initial_stack)
    error: Optional[str] = None
    for name in names:
        action = ACTION_BY_NAME[name]
        if len(stack) < action.arity:
            return stack, "action %s needs %d nodes, stack has %d" % (
                name, action.arity, len(stack))
        if action.arity == 0:
            popped: Tuple[Node, ...] = ()
        else:
            popped = stack[-action.arity:]
        stack = stack[: len(stack) - action.arity] + (action.build(popped),)
    return stack, error


def count_solutions(
    target: Node,
    action_names: Sequence[str] = ("PushZero", "PushOne", "MakeChange",
                                   "MakeGroup2"),
) -> Tuple[int, List[Tuple[str, ...]]]:
    """Exhaustively count the action words that reach ``(target,)``.

    ``size(target)`` is the only possible word length (each action adds one
    node), so the enumeration is complete: a count of 1 proves that the fixed
    target has EXACTLY ONE solution over the given action alphabet.
    """
    length = size(target)
    solutions: List[Tuple[str, ...]] = []
    for word in itertools.product(action_names, repeat=length):
        stack, error = simulate((), word)
        if error is None and stack == (target,):
            solutions.append(word)
    return len(solutions), solutions


def bfs_plan(
    env: MinimalEnv,
    max_actions: Optional[int] = None,
) -> Optional[List[str]]:
    """Shortest action word to the goal, deterministic (BFS, sorted actions).

    Used for dynamic/pattern goals where a post-order plan is not defined.
    """
    limit = max_actions if max_actions is not None else env.max_actions
    start = env.reset()
    if env.goal_achieved(start):
        return []
    seen = {start.stack: 0}
    frontier: List[Tuple[State, List[str]]] = [(start, [])]
    for _ in range(limit):
        nxt: List[Tuple[State, List[str]]] = []
        for state, prefix in frontier:
            for action in BUILD_ACTIONS:
                if len(state.stack) < action.arity:
                    continue
                new_state = env.step(state, action.name)
                if new_state.stack in seen:
                    continue
                seen[new_state.stack] = new_state.cost
                word = prefix + [action.name]
                if env.goal_achieved(new_state):
                    return word
                nxt.append((new_state, word))
        if not nxt:
            break
        frontier = nxt
    return None

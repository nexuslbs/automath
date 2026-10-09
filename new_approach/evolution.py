"""Stage 1 of the evolution pipeline: MORE semantic node kinds (derived).

The four CORE node types (``Zero``, ``One``, ``Change``, ``Group``) are
unchanged.  This module adds SEMANTIC node KINDS on top of them: each kind is a
``Group`` with a reserved integer tag and a fixed arity, exactly like the
already-derived ADD/LT/... operators.  A kind is therefore a DERIVED node kind,
not a new class, so ``node_type_count`` over the extended suite stays <= 4 and
the former 4 -> 3 -> 2 -> 1 reduction proof remains literally true.

Each semantic kind ``K`` with reserved tag ``t`` and arity ``k`` reduces to the
core encoding::

    K(a1..ak)  ==  Group(nat(t), (a1..ak))

Every kind also has a BUILD ACTION (``MakeAdd`` ...) whose operands are the
already-built children and whose result is that core ``Group``.  The action
vocabulary grows; the node-class count does not.  The explicit per-kind
mapping is the ``REDUCTION_TABLE`` below and section 2 of
``docs/evolution/DESIGN.md``.

Planning is still a unique post-order build: each action pops ``arity`` nodes
and pushes one, so a target tree of ``N`` nodes is reached in exactly ``N``
actions and every successful action word must be the post-order word (up to
symmetric operands, which the test suite avoids).

The module is pure standard library and bounded.
"""

from __future__ import annotations

import itertools
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

from .axioms import (
    DEFAULT_AXIOMS,
    Axiom,
    AxiomError,
    AxiomSet,
    T_ADD,
    T_AND,
    T_EQ,
    T_IF,
    T_LT,
    T_MUL,
    T_NOT,
    T_OR,
    T_SEQ,
    T_SUB,
)
from .env import (
    Action,
    ExprGoal,
    FailureFeedback,
    FixedStateGoal,
    MinimalEnv,
    State,
    bfs_plan,
    plan as core_plan,
    simulate as core_simulate,
)
from .nodes import Change, Group, NODE_TYPES, Node, One, Zero, nat, node_type_count

# --------------------------------------------------------------------------
# 1. New reserved axiom tags and their meaning
# --------------------------------------------------------------------------
# Existing tags reused for the new action vocabulary (see axioms.py):
#   T_ADD=1 T_SUB=2 T_MUL=3 T_LT=4 T_EQ=5 T_NOT=6 T_AND=7 T_OR=8
#   T_IF=9 T_SEQ=10
# New tags introduced by this unit:
T_DIV = 14
T_GT = 15
T_LE = 16
T_GE = 17
T_MOD = 18
T_NEG = 19


def _rule_div(ev, args):
    b = ev(args[1])
    if b == 0:
        raise AxiomError("DIV by zero")
    return ev(args[0]) // b


def _rule_mod(ev, args):
    b = ev(args[1])
    if b == 0:
        raise AxiomError("MOD by zero")
    return ev(args[0]) % b


def _rule_gt(ev, args):
    return ev(args[0]) > ev(args[1])


def _rule_le(ev, args):
    return ev(args[0]) <= ev(args[1])


def _rule_ge(ev, args):
    return ev(args[0]) >= ev(args[1])


def _rule_neg(ev, args):
    return -ev(args[0])


#: The extended subjective layer: the default axioms PLUS the new rules.
EVO_AXIOMS = AxiomSet(
    DEFAULT_AXIOMS
    + (
        Axiom(T_DIV, 2, "DIV", _rule_div),
        Axiom(T_GT, 2, "GT", _rule_gt),
        Axiom(T_LE, 2, "LE", _rule_le),
        Axiom(T_GE, 2, "GE", _rule_ge),
        Axiom(T_MOD, 2, "MOD", _rule_mod),
        Axiom(T_NEG, 1, "NEG", _rule_neg),
    )
)


# --------------------------------------------------------------------------
# 2. The semantic node kinds (derived) and their build actions
# --------------------------------------------------------------------------

@dataclass(frozen=True)
class SemOp:
    """A semantic node kind: a reserved tag + arity + its build action."""

    name: str      # build action name
    tag: int       # reserved integer tag value
    arity: int     # number of operands the build action consumes
    kind: str      # arith | cmp | logic | control | seq
    reduce: str    # explicit core encoding of the kind


SEMANTIC_OPS: Tuple[SemOp, ...] = (
    # arithmetic
    SemOp("MakeAdd", T_ADD, 2, "arith", "Group(nat(1),(a,b))"),
    SemOp("MakeSub", T_SUB, 2, "arith", "Group(nat(2),(a,b))"),
    SemOp("MakeMul", T_MUL, 2, "arith", "Group(nat(3),(a,b))"),
    SemOp("MakeDiv", T_DIV, 2, "arith", "Group(nat(14),(a,b))"),
    SemOp("MakeMod", T_MOD, 2, "arith", "Group(nat(18),(a,b))"),
    SemOp("MakeNeg", T_NEG, 1, "arith", "Group(nat(19),(a,))"),
    # comparison
    SemOp("MakeEq", T_EQ, 2, "cmp", "Group(nat(5),(a,b))"),
    SemOp("MakeLt", T_LT, 2, "cmp", "Group(nat(4),(a,b))"),
    SemOp("MakeGt", T_GT, 2, "cmp", "Group(nat(15),(a,b))"),
    SemOp("MakeLe", T_LE, 2, "cmp", "Group(nat(16),(a,b))"),
    SemOp("MakeGe", T_GE, 2, "cmp", "Group(nat(17),(a,b))"),
    # logic
    SemOp("MakeAnd", T_AND, 2, "logic", "Group(nat(7),(a,b))"),
    SemOp("MakeOr", T_OR, 2, "logic", "Group(nat(8),(a,b))"),
    SemOp("MakeNot", T_NOT, 1, "logic", "Group(nat(6),(a,))"),
    # sequencing / control wrappers
    SemOp("MakeIf", T_IF, 3, "control", "Group(nat(9),(c,t,e))"),
    SemOp("MakeSeq2", T_SEQ, 2, "seq", "Group(nat(10),(a,b))"),
    SemOp("MakeSeq3", T_SEQ, 3, "seq", "Group(nat(10),(a,b,c))"),
    SemOp("MakeSeq4", T_SEQ, 4, "seq", "Group(nat(10),(a,b,c,d))"),
)

SEM_BY_NAME: Dict[str, SemOp] = {o.name: o for o in SEMANTIC_OPS}
SEM_BY_KEY: Dict[Tuple[int, int], SemOp] = {
    (o.tag, o.arity): o for o in SEMANTIC_OPS
}

#: The explicit per-kind reduction onto the four core types, as a table.
REDUCTION_TABLE: Tuple[Tuple[str, int, int, str], ...] = tuple(
    (o.name, o.tag, o.arity, o.reduce) for o in SEMANTIC_OPS
)
KIND_GROUPS: Dict[str, Tuple[str, ...]] = {}
for _o in SEMANTIC_OPS:
    KIND_GROUPS.setdefault(_o.kind, ())
    KIND_GROUPS[_o.kind] = KIND_GROUPS[_o.kind] + (_o.name,)


def semantic_kind_names() -> Tuple[str, ...]:
    return tuple(o.name for o in SEMANTIC_OPS)


# --------------------------------------------------------------------------
# 3. Core builders for semantic applications (only the four core classes)
# --------------------------------------------------------------------------

def sem(name: str, *args: Node) -> Group:
    """Build the core ``Group`` for semantic kind ``name`` with operands."""
    op = SEM_BY_NAME[name]
    if len(args) != op.arity:
        raise ValueError("%s needs %d args, got %d" % (name, op.arity, len(args)))
    return Group(nat(op.tag), tuple(args))


def ev_add(a: Node, b: Node) -> Group:
    return sem("MakeAdd", a, b)


def ev_sub(a: Node, b: Node) -> Group:
    return sem("MakeSub", a, b)


def ev_mul(a: Node, b: Node) -> Group:
    return sem("MakeMul", a, b)


def ev_div(a: Node, b: Node) -> Group:
    return sem("MakeDiv", a, b)


def ev_mod(a: Node, b: Node) -> Group:
    return sem("MakeMod", a, b)


def ev_neg(a: Node) -> Group:
    return sem("MakeNeg", a)


def ev_eq(a: Node, b: Node) -> Group:
    return sem("MakeEq", a, b)


def ev_lt(a: Node, b: Node) -> Group:
    return sem("MakeLt", a, b)


def ev_gt(a: Node, b: Node) -> Group:
    return sem("MakeGt", a, b)


def ev_le(a: Node, b: Node) -> Group:
    return sem("MakeLe", a, b)


def ev_ge(a: Node, b: Node) -> Group:
    return sem("MakeGe", a, b)


def ev_and(a: Node, b: Node) -> Group:
    return sem("MakeAnd", a, b)


def ev_or(a: Node, b: Node) -> Group:
    return sem("MakeOr", a, b)


def ev_not(a: Node) -> Group:
    return sem("MakeNot", a)


def ev_if(c: Node, t: Node, e: Node) -> Group:
    return sem("MakeIf", c, t, e)


def ev_seq2(a: Node, b: Node) -> Group:
    return sem("MakeSeq2", a, b)


def ev_seq3(a: Node, b: Node, c: Node) -> Group:
    return sem("MakeSeq3", a, b, c)


# --------------------------------------------------------------------------
# 4. The extended action vocabulary and environment
# --------------------------------------------------------------------------

def _sem_builder(op: SemOp):
    tag = op.tag

    def build(popped: Tuple[Node, ...]) -> Node:
        return Group(nat(tag), tuple(popped))

    return build


#: Core leaf/step actions kept, plus one build action per semantic kind.
EVO_ACTIONS: Tuple[Action, ...] = (
    Action("PushZero", 0, lambda a: Zero()),
    Action("PushOne", 0, lambda a: One()),
    Action("MakeChange", 1, lambda a: Change(a[0])),
) + tuple(Action(o.name, o.arity, _sem_builder(o)) for o in SEMANTIC_OPS)

EVO_ACTION_BY_NAME: Dict[str, Action] = {a.name: a for a in EVO_ACTIONS}
EVO_ACTION_ORDER: Tuple[str, ...] = tuple(a.name for a in EVO_ACTIONS)


class EvoEnv:
    """The extended environment: the same stack machine over ``EVO_ACTIONS``."""

    def __init__(
        self,
        goal,
        axioms: Optional[AxiomSet] = None,
        initial_stack: Sequence[Node] = (),
        max_actions: int = 24,
    ) -> None:
        self.goal = goal
        self.axioms = axioms or EVO_AXIOMS
        self.initial_stack = tuple(initial_stack)
        self.max_actions = max_actions

    def reset(self) -> State:
        return State(stack=self.initial_stack, cost=0, history=())

    def step(self, state: State, action_name: str) -> State:
        action = EVO_ACTION_BY_NAME.get(action_name)
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
        popped = () if action.arity == 0 else state.stack[-action.arity:]
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
# 5. Deterministic planning / simulation / uniqueness for the extended env
# --------------------------------------------------------------------------

def sem_plan(target: Node) -> List[str]:
    """The unique post-order build of ``target`` with semantic actions."""
    if isinstance(target, Zero):
        return ["PushZero"]
    if isinstance(target, One):
        return ["PushOne"]
    if isinstance(target, Change):
        return sem_plan(target.child) + ["MakeChange"]
    if isinstance(target, Group):
        key = (EVO_AXIOMS.evaluate(target.tag), target.arity())
        op = SEM_BY_KEY.get(key)
        if op is None:
            raise ValueError("no semantic build action for tag/arity %r" % (key,))
        names: List[str] = []
        for item in target.items:
            names.extend(sem_plan(item))
        names.append(op.name)
        return names
    raise TypeError(type(target))


def sem_action_count(target: Node) -> int:
    """Number of build actions (== number of nodes, tags synthesised)."""
    return len(sem_plan(target))


def simulate_evo(initial_stack: Sequence[Node], names: Sequence[str]
                 ) -> Tuple[Tuple[Node, ...], Optional[str]]:
    stack: Tuple[Node, ...] = tuple(initial_stack)
    for name in names:
        action = EVO_ACTION_BY_NAME[name]
        if len(stack) < action.arity:
            return stack, "action %s needs %d nodes, stack has %d" % (
                name, action.arity, len(stack))
        popped = () if action.arity == 0 else stack[-action.arity:]
        stack = stack[: len(stack) - action.arity] + (action.build(popped),)
    return stack, None


def count_evo_solutions(
    target: Node,
    action_names: Sequence[str] = EVO_ACTION_ORDER,
) -> Tuple[int, List[Tuple[str, ...]]]:
    """Exhaustive unique-solution count over the semantic action alphabet."""
    length = sem_action_count(target)
    solutions: List[Tuple[str, ...]] = []
    for word in itertools.product(action_names, repeat=length):
        stack, error = simulate_evo((), word)
        if error is None and stack == (target,):
            solutions.append(word)
    return len(solutions), solutions


def plan_evo(target: Node, initial_stack: Sequence[Node] = ()) -> List[str]:
    """Post-order from empty; shortest BFS word from a non-empty stack."""
    if initial_stack:
        env = EvoEnv(FixedStateGoal(target), initial_stack=initial_stack,
                     max_actions=sem_action_count(target) + 4)
        word = bfs_plan_evo(env)
        if word is None:
            raise ValueError("initial stack does not compose with target")
        return word
    return sem_plan(target)


def bfs_plan_evo(
    env: EvoEnv, max_actions: Optional[int] = None
) -> Optional[List[str]]:
    limit = max_actions if max_actions is not None else env.max_actions
    start = env.reset()
    if env.goal_achieved(start):
        return []
    seen = {start.stack: 0}
    frontier: List[Tuple[State, List[str]]] = [(start, [])]
    for _ in range(limit):
        nxt: List[Tuple[State, List[str]]] = []
        for state, prefix in frontier:
            for action in EVO_ACTIONS:
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


# --------------------------------------------------------------------------
# 6. The more-complex scenario suite (deterministic; single-solution where
#    possible) that REQUIRES the new semantic kinds.
# --------------------------------------------------------------------------

@dataclass(frozen=True)
class Scenario:
    name: str
    target: Node
    expected: object
    level: str


def scenarios() -> List[Scenario]:
    return [
        Scenario("e_add_mul", ev_add(One(), ev_mul(One(), nat(3))), 4, "arith"),
        Scenario("e_div", ev_div(ev_mul(nat(3), nat(3)), nat(3)), 3, "arith"),
        Scenario("e_mod", ev_mod(nat(7), nat(3)), 1, "arith"),
        Scenario("e_neg", ev_neg(nat(3)), -3, "arith"),
        Scenario("e_cmp", ev_and(ev_lt(One(), nat(3)),
                                 ev_ge(nat(3), nat(3))), True, "cmp"),
        Scenario("e_eq_gt", ev_and(ev_eq(nat(2), nat(2)),
                                   ev_not(ev_gt(One(), nat(2)))), True, "cmp"),
        Scenario("e_logic", ev_or(ev_not(Zero()), ev_and(One(), Zero())),
                 True, "logic"),
        Scenario("e_branch", ev_if(ev_lt(One(), nat(3)),
                                   ev_add(One(), One()),
                                   ev_mul(Zero(), nat(3))), 2, "control"),
        Scenario("e_nested", ev_mul(ev_add(One(), One()),
                                    ev_sub(nat(3), One())), 4, "control"),
        Scenario("e_seq", ev_seq3(ev_add(One(), One()),
                                  ev_not(Zero()),
                                  ev_mul(nat(3), One())), (2, True, 3), "seq"),
        Scenario("e_seq4", ev_seq2(ev_add(nat(2), nat(2)),
                                   ev_div(nat(8), nat(2))), (4, 4), "seq"),
    ]


#: Small targets with restricted alphabets, for exhaustive uniqueness proof.
UNIQUE_SEM_TARGETS: Tuple[Tuple[str, Node, Tuple[str, ...]], ...] = (
    ("add(1,1)", ev_add(One(), One()), ("PushOne", "MakeAdd")),
    ("not(0)", ev_not(Zero()), ("PushZero", "MakeNot")),
    ("mul(1,C(1))", ev_mul(One(), nat(2)),
     ("PushOne", "MakeChange", "MakeMul")),
    ("eq(add(1,1),C(1))", ev_eq(ev_add(One(), One()), nat(2)),
     ("PushOne", "MakeAdd", "MakeChange", "MakeEq")),
    ("lt(1,C(C(1)))", ev_lt(One(), nat(3)),
     ("PushOne", "MakeChange", "MakeLt")),
    ("if(lt(1,C(1)),1,0)", ev_if(ev_lt(One(), nat(2)), One(), Zero()),
     ("PushZero", "PushOne", "MakeChange", "MakeLt", "MakeIf")),
)


def scenario_cases(initial_prefixes: bool = False) -> List[Tuple[str, Node,
                                                                  Tuple[Node, ...],
                                                                  str,
                                                                  Tuple[str, ...]]]:
    """(name, target, initial_stack, level, expected_word) for each scenario.

    With ``initial_prefixes`` every non-trivial prefix of the unique post-order
    word is used as a NEW initial state; the expected word is the remaining
    suffix (the prefix lies on the unique build, so the suffix reaches the
    target)."""
    out: List[Tuple[str, Node, Tuple[Node, ...], str, Tuple[str, ...]]] = []
    for sc in scenarios():
        word = sem_plan(sc.target)
        if not initial_prefixes:
            out.append((sc.name, sc.target, (), sc.level, tuple(word)))
            continue
        for i in range(1, len(word) + 1):
            stack, error = simulate_evo((), word[:i])
            if error is not None:
                raise RuntimeError("scenario prefix simulation failed: " + error)
            label = "already-goal" if i == len(word) else ("prefix%d" % i)
            out.append(("%s/%s" % (sc.name, label), sc.target, stack, sc.level,
                        tuple(word[i:])))
    return out


def core_reduction(target: Node) -> Node:
    """The explicit mapping of a semantic tree onto the four core types.

    Semantic kinds are ALREADY core ``Group`` nodes (tag + operands), so the
    mapping is the identity; this function exists to make the claim explicit
    and to assert the class set is a subset of the four core types.
    """
    seen: set = set()

    def walk(n: Node) -> Node:
        seen.add(type(n))
        return n

    walk(target)
    if not seen <= set(NODE_TYPES):
        raise TypeError("non-core node class in semantic tree: %r" % (seen,))
    return target

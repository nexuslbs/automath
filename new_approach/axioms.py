"""Axioms: the rules that give a ``Group`` combination its meaning.

An ``Axiom`` is a META definition (a subjective understanding) about a group of
nodes, exactly as in the operator's reference design. It is NOT a node type.

Each axiom is keyed by ``(tag_value, arity)`` where ``tag_value`` is the integer
denoted by the ``Group``'s tag node and ``arity`` is the number of operands.
The key is an integer rather than a canonical string so that the SAME axiom set
interprets the 4-, 3-, 2- and 1-node-type encodings (the tag node may itself be
re-encoded, but it still denotes the same integer).

A rule receives ``(ev, args)``: a recursive evaluator and the RAW operand nodes,
so control-flow rules can choose which branch to evaluate (real control flow,
not eager evaluation of both branches).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, Optional, Tuple

from .nodes import Change, Group, Node, One, Zero

# Reserved integer tag values. Expression tags start at 1; tag 0 is the
# reserved successor tag of the 2-node-type reduction.
T_SUCC = 0
T_ADD = 1
T_SUB = 2
T_MUL = 3
T_LT = 4
T_EQ = 5
T_NOT = 6
T_AND = 7
T_OR = 8
T_IF = 9
T_SEQ = 10
T_LEN = 11
T_HEAD = 12
T_TAIL = 13

Value = object


class AxiomError(Exception):
    """No axiom gives this combination a meaning in this context."""


@dataclass(frozen=True)
class Axiom:
    tag: int
    arity: Optional[int]
    name: str
    rule: Callable[[Callable[[Node], Value], Tuple[Node, ...]], Value]


def _truthy(v: Value) -> bool:
    if isinstance(v, bool):
        return v
    if isinstance(v, int):
        return v != 0
    if isinstance(v, tuple):
        return len(v) > 0
    return bool(v)


def _rule_add(ev, args):
    return ev(args[0]) + ev(args[1])


def _rule_sub(ev, args):
    return max(ev(args[0]) - ev(args[1]), 0)


def _rule_mul(ev, args):
    return ev(args[0]) * ev(args[1])


def _rule_lt(ev, args):
    return ev(args[0]) < ev(args[1])


def _rule_eq(ev, args):
    return ev(args[0]) == ev(args[1])


def _rule_not(ev, args):
    return not _truthy(ev(args[0]))


def _rule_and(ev, args):
    return _truthy(ev(args[0])) and _truthy(ev(args[1]))


def _rule_or(ev, args):
    return _truthy(ev(args[0])) or _truthy(ev(args[1]))


def _rule_if(ev, args):
    # Lazy: only the chosen branch is evaluated. This is the control-flow axiom.
    if _truthy(ev(args[0])):
        return ev(args[1])
    return ev(args[2])


def _rule_seq(ev, args):
    return tuple(ev(a) for a in args)


def _rule_len(ev, args):
    seq = ev(args[0])
    return len(seq)


def _rule_head(ev, args):
    seq = ev(args[0])
    if len(seq) == 0:
        raise AxiomError("HEAD of an empty sequence")
    return seq[0]


def _rule_tail(ev, args):
    seq = ev(args[0])
    if len(seq) == 0:
        raise AxiomError("TAIL of an empty sequence")
    return seq[1:]


def _rule_succ(ev, args):
    return ev(args[0]) + 1


#: The default subjective layer: one axiom per (tag, arity) meaning.
DEFAULT_AXIOMS: Tuple[Axiom, ...] = (
    Axiom(T_SUCC, 1, "SUCC", _rule_succ),
    Axiom(T_ADD, 2, "ADD", _rule_add),
    Axiom(T_SUB, 2, "SUB", _rule_sub),
    Axiom(T_MUL, 2, "MUL", _rule_mul),
    Axiom(T_LT, 2, "LT", _rule_lt),
    Axiom(T_EQ, 2, "EQ", _rule_eq),
    Axiom(T_NOT, 1, "NOT", _rule_not),
    Axiom(T_AND, 2, "AND", _rule_and),
    Axiom(T_OR, 2, "OR", _rule_or),
    Axiom(T_IF, 3, "IF", _rule_if),
    Axiom(T_SEQ, None, "SEQ", _rule_seq),
    Axiom(T_LEN, 1, "LEN", _rule_len),
    Axiom(T_HEAD, 1, "HEAD", _rule_head),
    Axiom(T_TAIL, 1, "TAIL", _rule_tail),
    # The reserved successor tag used by the 2-node-type reduction.
    Axiom(T_SUCC, 1, "SUCC_REDUCED", _rule_succ),
)


class AxiomSet:
    """An immutable set of axioms; ``evaluate`` is a pure interpreter."""

    def __init__(self, axioms: Tuple[Axiom, ...] = DEFAULT_AXIOMS) -> None:
        self.axioms = axioms
        self._by_key: Dict[Tuple[int, Optional[int]], Axiom] = {}
        for ax in axioms:
            self._by_key[(ax.tag, ax.arity)] = ax
            if ax.arity is not None:
                self._by_key[(ax.tag, None)] = ax

    def rule_for(self, tag_value: int, arity: int) -> Axiom:
        ax = self._by_key.get((tag_value, arity)) or self._by_key.get(
            (tag_value, None)
        )
        if ax is None:
            raise AxiomError(
                "no axiom for tag=%d arity=%d" % (tag_value, arity)
            )
        return ax

    def evaluate(self, node: Node) -> Value:
        if isinstance(node, Zero):
            return 0
        if isinstance(node, One):
            return 1
        if isinstance(node, Change):
            return self.evaluate(node.child) + 1
        if isinstance(node, Group):
            tag_value = self.evaluate(node.tag)
            if not isinstance(tag_value, int):
                raise AxiomError("group tag is not an integer")
            ax = self.rule_for(tag_value, node.arity())
            return ax.rule(self.evaluate, node.items)
        raise TypeError(type(node))

    def truthy(self, node: Node) -> bool:
        return _truthy(self.evaluate(node))

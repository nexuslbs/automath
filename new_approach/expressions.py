"""Expressiveness: encoders for arbitrary math/logic expressions.

Every expression is a ``Group`` combination whose tag is a numeral (a node built
only from ``Zero``/``One``/``Change``). No operator is a new node type: the
operators are DERIVED from the four node types by structural position, and the
axiom set gives them meaning. This is the expressiveness claim made concrete.

Worked encodings
----------------
* number ``n``            -> ``nat(n)``          (Zero, One, Change chain)
* boolean                 -> ``Zero`` / ``One``
* operator ``op``         -> ``nat(T_OP)``
* application ``op(a,b)`` -> ``Group(nat(T_OP), (a, b))``
* sequence ``[a,b,c]``    -> ``Group(nat(T_SEQ), (a, b, c))`` (dynamic arity)
* control flow ``if(c,t,e)`` -> ``Group(nat(T_IF), (c, t, e))``, evaluated lazily
"""

from __future__ import annotations

from typing import Iterable

from .axioms import (
    T_ADD,
    T_AND,
    T_EQ,
    T_HEAD,
    T_IF,
    T_LEN,
    T_LT,
    T_MUL,
    T_NOT,
    T_OR,
    T_SEQ,
    T_SUB,
    T_TAIL,
)
from .nodes import Group, Node, nat


def app(tag: int, *args: Node) -> Group:
    """A combination: tag + operands. ``tag`` is itself a node (a numeral)."""
    return Group(nat(tag), tuple(args))


def add(a: Node, b: Node) -> Group:
    return app(T_ADD, a, b)


def sub(a: Node, b: Node) -> Group:
    return app(T_SUB, a, b)


def mul(a: Node, b: Node) -> Group:
    return app(T_MUL, a, b)


def lt(a: Node, b: Node) -> Group:
    return app(T_LT, a, b)


def eq(a: Node, b: Node) -> Group:
    return app(T_EQ, a, b)


def logical_not(a: Node) -> Group:
    return app(T_NOT, a)


def logical_and(a: Node, b: Node) -> Group:
    return app(T_AND, a, b)


def logical_or(a: Node, b: Node) -> Group:
    return app(T_OR, a, b)


def iff(cond: Node, then: Node, other: Node) -> Group:
    return app(T_IF, cond, then, other)


def seq(*items: Node) -> Group:
    """A sequence is a Group with the SEQ tag and a DYNAMIC number of items."""
    return app(T_SEQ, *items)


def length(items: Node) -> Group:
    return app(T_LEN, items)


def head(items: Node) -> Group:
    return app(T_HEAD, items)


def tail(items: Node) -> Group:
    return app(T_TAIL, items)


#: Human-readable expression samples used by the tests and the design doc.
SAMPLES = {
    "add(2,3)": lambda: add(nat(2), nat(3)),
    "sub(5,2)": lambda: sub(nat(5), nat(2)),
    "mul(2,3)": lambda: mul(nat(2), nat(3)),
    "lt(2,3)": lambda: lt(nat(2), nat(3)),
    "eq(add(1,1),2)": lambda: eq(add(nat(1), nat(1)), nat(2)),
    "and(true,not(false))": lambda: logical_and(nat(1), logical_not(nat(0))),
    "or(false,true)": lambda: logical_or(nat(0), nat(1)),
    "if(lt(1,2),add(1,1),0)": lambda: iff(
        lt(nat(1), nat(2)), add(nat(1), nat(1)), nat(0)
    ),
    "len([1,2,3])": lambda: length(seq(nat(1), nat(2), nat(3))),
    "head([1,2,3])": lambda: head(seq(nat(1), nat(2), nat(3))),
    "tail([1,2,3])": lambda: tail(seq(nat(1), nat(2), nat(3))),
}


def sample(name: str) -> Node:
    return SAMPLES[name]()

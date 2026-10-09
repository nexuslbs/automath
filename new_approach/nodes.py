"""The minimal node set of the new approach: FOUR node types.

Node types (3 basic + 1 dynamic)
--------------------------------
* ``Zero``   - nullary objective fact ``0``. Also the numeric base, boolean
               false, and the empty combination.
* ``One``    - nullary objective fact ``1``. Also boolean true.
* ``Change`` - unary ``Change(child)``: the state/action step. Numerically it is
               the successor (``value(child) + 1``); structurally it is the only
               unary constructor, so it is also how any term becomes a new term.
* ``Group``  - the DYNAMIC grouping node: ``Group(tag, items)`` is an n-ary
               COMBINATION of existing nodes. It carries no intrinsic meaning;
               its meaning is supplied by the axiom set (``axioms.py``) keyed on
               the tag and the number of items. Every operator, every list, and
               every control-flow form is a ``Group`` whose tag is itself a node.

The count above is the syntactic count: a node's *type* is its Python class.
``Axiom`` and ``Goal`` are meta definitions over groups of nodes, NOT node
types, so they add nothing to the count (see docs/new-approach/DESIGN.md).

This module also exposes the reduction encodings used to show that the same
language is expressible with 3, 2 and 1 node types.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Tuple


class Node:
    """Base class of every node type. Cheap structural views only."""

    __slots__ = ()

    def kind(self) -> str:
        return type(self).__name__

    def children(self) -> Tuple["Node", ...]:
        return ()

    def canonical(self) -> str:
        raise NotImplementedError


@dataclass(frozen=True)
class Zero(Node):
    """0 / false / empty. A nullary objective fact."""

    __slots__ = ()

    def canonical(self) -> str:
        return "0"


@dataclass(frozen=True)
class One(Node):
    """1 / true. A nullary objective fact."""

    __slots__ = ()

    def canonical(self) -> str:
        return "1"


@dataclass(frozen=True)
class Change(Node):
    """The unary state/action step: ``Change(child)``."""

    child: Node

    def children(self) -> Tuple[Node, ...]:
        return (self.child,)

    def canonical(self) -> str:
        return "C(" + self.child.canonical() + ")"


@dataclass(frozen=True)
class Group(Node):
    """The dynamic n-ary combination node.

    ``tag`` is itself a node (the operator tag); ``items`` are the operands.
    The node has no meaning until an axiom matches ``(tag, len(items))``.
    """

    tag: Node
    items: Tuple[Node, ...] = ()

    def children(self) -> Tuple[Node, ...]:
        return (self.tag,) + tuple(self.items)

    def arity(self) -> int:
        return len(self.items)

    def canonical(self) -> str:
        body = ",".join(i.canonical() for i in self.items)
        return "G(" + self.tag.canonical() + ";" + body + ")"


#: The four node types, in count order.
NODE_TYPES = (Zero, One, Change, Group)


def node_type_count(*roots: Node) -> int:
    """Number of distinct node *classes* appearing in the given trees."""
    seen: set = set()

    def walk(n: Node) -> None:
        seen.add(type(n))
        for c in n.children():
            walk(c)

    for r in roots:
        walk(r)
    return len(seen)


def iter_nodes(root: Node) -> Iterable[Node]:
    yield root
    for c in root.children():
        yield from iter_nodes(c)


def size(root: Node) -> int:
    """Number of nodes in the tree (== minimal number of build actions)."""
    return 1 + sum(size(c) for c in root.children())


def nat(n: int) -> Node:
    """Canonical unary natural: 0 -> Zero, 1 -> One, n -> Change(nat(n-1))."""
    if n < 0:
        raise ValueError("negative")
    if n == 0:
        return Zero()
    if n == 1:
        return One()
    node: Node = One()
    for _ in range(n - 1):
        node = Change(node)
    return node


# --------------------------------------------------------------------------
# Reduction encodings: the same language with fewer node types.
# --------------------------------------------------------------------------

def to_min3(root: Node) -> Node:
    """Reduction 4 -> 3 node types: ``One`` is defined as ``Change(Zero)``.

    The result uses only ``{Zero, Change, Group}``.
    """
    if isinstance(root, Zero):
        return Zero()
    if isinstance(root, One):
        return Change(Zero())
    if isinstance(root, Change):
        return Change(to_min3(root.child))
    if isinstance(root, Group):
        return Group(to_min3(root.tag), tuple(to_min3(i) for i in root.items))
    raise TypeError(type(root))


def to_min2(root: Node) -> Node:
    """Reduction 3 -> 2 node types: ``Change(x)`` is ``Group(Zero, (x,))``.

    The tag ``Zero`` with arity 1 is the reserved successor tag. The result
    uses only ``{Zero, Group}`` (``One`` is first reduced to ``Change(Zero)``).
    """
    return _to_min2(to_min3(root))


def _to_min2(root: Node) -> Node:
    if isinstance(root, Zero):
        return Zero()
    if isinstance(root, Change):
        return Group(Zero(), (_to_min2(root.child),))
    if isinstance(root, Group):
        return Group(_to_min2(root.tag), tuple(_to_min2(i) for i in root.items))
    raise TypeError(type(root))


@dataclass(frozen=True)
class Cell:
    """The 1-node-type encoding: a single untyped n-ary container.

    ``Cell(())`` is the sole nullary value; every other term is a nested Cell.
    This is the hereditarily-finite-set / pure S-expression universe.
    """

    kids: Tuple["Cell", ...] = ()

    def canonical(self) -> str:
        return "[" + ",".join(k.canonical() for k in self.kids) + "]"


def to_cell(root: Node) -> Cell:
    """Reduction 4 -> 1 node type: ``One``/``Change``/``Group`` all become Cell.

    ``Zero -> Cell()`` (the empty container); ``One -> Cell(Cell())``;
    ``Change(x) -> Cell(to_cell(x))``; ``Group(tag, items) ->
    Cell((to_cell(tag),) + items)``. The interpretaion of an application is
    recovered from the tag position inside the Cell, exactly as for Group.
    """
    if isinstance(root, Zero):
        return Cell(())
    if isinstance(root, One):
        return Cell((Cell(()),))
    if isinstance(root, Change):
        return Cell((to_cell(root.child),))
    if isinstance(root, Group):
        return Cell((to_cell(root.tag),) + tuple(to_cell(i) for i in root.items))
    raise TypeError(type(root))


def cell_kind_count(*roots: Cell) -> int:
    return 1 if roots else 0

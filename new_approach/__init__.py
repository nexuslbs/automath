"""new_approach: an entirely new minimal-node approach for automath.

Four node types (3 basic + 1 dynamic grouping node):

    Zero, One, Change, Group

Everything else (numbers, booleans, comparisons, arithmetic, sequences,
control flow, goals, axioms) is DERIVED from those four by structure and by an
axiom set. See ``docs/new-approach/DESIGN.md``.

Run the deterministic single-solution tests with::

    python -m new_approach.tests
"""

from .nodes import Change, Group, NODE_TYPES, Node, One, Zero

__all__ = ["Zero", "One", "Change", "Group", "Node", "NODE_TYPES"]

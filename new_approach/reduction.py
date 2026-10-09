"""Evaluation of the 1-node-type (Cell) reduction.

``to_cell`` maps the four node types onto a single ``Cell`` class.  The
interpreter below recovers the same axiom meanings from structural position:

* ``Cell(())``           -> 0
* ``Cell((x,))``         -> successor (Change)
* ``Cell((t, a1..ak))``  -> the axiom for tag ``value(t)`` and arity ``k``

The empty-item Group case does not occur (every operator has >= 1 operand), so
the 1-kid shape unambiguously denotes Change.
"""

from __future__ import annotations

from .axioms import AxiomSet
from .nodes import Cell

_AX = AxiomSet()


def cell_value(cell: Cell):
    kids = cell.kids
    if len(kids) == 0:
        return 0
    if len(kids) == 1:
        return cell_value(kids[0]) + 1
    tag = cell_value(kids[0])
    args = kids[1:]
    ax = _AX.rule_for(tag, len(args))
    return ax.rule(cell_value, args)

"""Genome = the flat weights of a small target-conditioned MLP, plus mixing.

Layout of one genome (all floats, standard library only)::

    W1  hidden x in_dim      (row j = weights of hidden unit j)
    b1  hidden
    W2  hidden
    b2  1

The network is deliberately tiny: the trainer's cost is dominated by the number
of legal actions, not by the network. Mixing keeps every blended gene inside the
convex hull of the two parents' genes; mutation adds gaussian noise on top and
records the exact deltas so a test can assert ``child == blend + delta``.
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass, field
from typing import List, Optional, Sequence, Tuple


class PolicyNet:
    """A one-hidden-layer tanh MLP; input = state features + action features."""

    def __init__(self, in_dim: int, hidden: int = 12) -> None:
        if in_dim <= 0:
            raise ValueError("in_dim must be positive")
        if hidden <= 0:
            raise ValueError("hidden must be positive")
        self.in_dim = in_dim
        self.hidden = hidden
        self.size = hidden * in_dim + hidden + hidden + 1

    # -- layout offsets ----------------------------------------------------
    @property
    def b1_offset(self) -> int:
        return self.hidden * self.in_dim

    @property
    def w2_offset(self) -> int:
        return self.b1_offset + self.hidden

    @property
    def b2_offset(self) -> int:
        return self.w2_offset + self.hidden

    def init(self, rng: random.Random, scale: float = 0.5) -> List[float]:
        return [rng.gauss(0.0, scale) for _ in range(self.size)]

    def forward(self, genome: Sequence[float], x: Sequence[float]) -> float:
        if len(genome) != self.size:
            raise ValueError("genome has %d genes, expected %d" % (len(genome), self.size))
        if len(x) != self.in_dim:
            raise ValueError("input has %d features, expected %d" % (len(x), self.in_dim))
        hidden = self.hidden
        n = self.in_dim
        b1 = self.b1_offset
        w2 = self.w2_offset
        b2 = self.b2_offset
        activations = [0.0] * hidden
        for j in range(hidden):
            total = genome[b1 + j]
            row = j * n
            for i in range(n):
                total += genome[row + i] * x[i]
            activations[j] = math.tanh(total)
        out = genome[b2]
        for j in range(hidden):
            out += genome[w2 + j] * activations[j]
        return out

    def uniform(self, genome: Sequence[float]) -> None:
        """A deterministic neutral genome (all zeros) -> uniform logits."""
        for i in range(self.size):
            genome[i] = 0.0


@dataclass
class MixResult:
    """The exact bookkeeping of one reproduction."""

    child: List[float]
    blend: List[float]
    deltas: List[float]
    alpha: float
    select_count: int
    blend_count: int

    @property
    def mutation_magnitude(self) -> float:
        return math.sqrt(sum(d * d for d in self.deltas))


def mix_genomes(
    parent_a: Sequence[float],
    parent_b: Sequence[float],
    rng: random.Random,
    alpha: float = 0.5,
    select_prob: float = 0.5,
    per_gene_blend: bool = True,
    mutation_rate: float = 0.0,
    mutation_sigma: float = 0.0,
) -> MixResult:
    """Per-weight blend/selection of two parents plus gaussian mutation.

    For every gene independently:

    * with probability ``select_prob`` the child INHERITS one parent's gene
      (uniformly chosen) - the selection branch;
    * otherwise the child gets a convex combination ``w*a + (1-w)*b`` where ``w``
      is ``alpha`` (fixed) or drawn per gene from ``[0, 1]`` when
      ``per_gene_blend`` is set - the blend branch.

    Both branches keep the gene inside ``[min(a,b), max(a,b)]``. Mutation then
    adds ``N(0, mutation_sigma)`` to a gene with probability ``mutation_rate``;
    the deltas are returned so a test can assert ``child == blend + delta`` and
    that the noiseless blend is in the convex hull of the parents.
    """
    if len(parent_a) != len(parent_b):
        raise ValueError("parents have different genome lengths")
    blend: List[float] = []
    deltas: List[float] = []
    child: List[float] = []
    select_count = 0
    blend_count = 0
    for gene_a, gene_b in zip(parent_a, parent_b):
        if select_prob > 0.0 and rng.random() < select_prob:
            gene = gene_a if rng.random() < 0.5 else gene_b
            select_count += 1
        else:
            weight = rng.random() if per_gene_blend else alpha
            gene = weight * gene_a + (1.0 - weight) * gene_b
            blend_count += 1
        delta = 0.0
        if mutation_rate > 0.0 and mutation_sigma > 0.0 and rng.random() < mutation_rate:
            delta = rng.gauss(0.0, mutation_sigma)
        blend.append(gene)
        deltas.append(delta)
        child.append(gene + delta)
    return MixResult(
        child=child,
        blend=blend,
        deltas=deltas,
        alpha=alpha,
        select_count=select_count,
        blend_count=blend_count,
    )


def mutate_genome(
    parent: Sequence[float],
    rng: random.Random,
    mutation_rate: float,
    mutation_sigma: float,
) -> MixResult:
    """An asexual mutated copy (mixing a parent with itself is the identity)."""
    result = mix_genomes(
        parent, parent, rng,
        select_prob=0.0, per_gene_blend=False, alpha=1.0,
        mutation_rate=mutation_rate, mutation_sigma=mutation_sigma,
    )
    return result


def round_genome(genome: Sequence[float], digits: int = 6) -> List[float]:
    return [round(float(g), digits) for g in genome]

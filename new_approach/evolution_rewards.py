"""Unit B (Phase A): the DESIRED shaped reward scheme for the semi-evolution.

This module is the single source of truth for the reward.  It is deliberately
NOT the Stage 2 scaffolding: there is no GOOD/BAD reward NODE, no planner
action-word supervision and no ``T_REWARD`` node anywhere on this path.  The
signal is four numeric terms, all functions of the real environment state and
the real goal:

1. per-step cost      ``-0.05`` for every action (fewer actions = better);
2. potential-based progress shaping, Ng, Harada & Russell (1999) PBRS::

       F(s, a, s') = gamma * Phi(s') - Phi(s)
       gamma       = 0.99
       Phi(s)      = 0.5 * goal_similarity(s)

3. sub-goal bonus      ``+0.10`` when an action produces a top node that is a
   PROPER, non-goal, non-trivial sub-structure/prefix of the target and the
   matched target structure strictly grows;
4. final-goal reward   ``+1.0`` when a feasible state is solved (unchanged real
   reward).

Bound (binding): the sum of the POSITIVE shaping and sub-goal bonuses inside
ONE episode is capped at ``SHAPING_CAP = 0.9 < 1.0`` so the true ``+1.0``
objective always dominates the shaping.  Negative shaping is never capped (it
only punishes states that move away from the goal).

``goal_similarity`` (range ``[0, 1]``) is a concrete two-term measure of how
much of the goal target structure already matches the current stack (the TOP
node of the stack is the unit, because every build action pushes exactly one
new top node):

* ``size_prox``  = ``1 - |size(top) - size(target)| / max(size(top), size(target))``
  (``0.0`` when the stack is empty): the size-proximity of the top node to the
  target node count (``nodes.size`` = syntactic node count);
* ``match_frac`` = ``match_count(target, top) / size(target)`` where
  ``match_count`` is the number of nodes of the LARGEST target sub-structure
  that is canonically EQUAL to the top node (it is ``size(target)`` when the
  top node IS the target, so ``match_frac`` reaches ``1.0`` exactly at goal);
* ``goal_similarity = 0.5 * size_prox + 0.5 * match_frac``.

The scheme is pure standard library, bounded and deterministic.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from .nodes import Node, size

# --------------------------------------------------------------------------
# Constants (the binding reward design)
# --------------------------------------------------------------------------

STEP_COST = -0.05
REWARD_GAMMA = 0.99
PHI_SCALE = 0.5
SUBGOAL_BONUS = 0.10
GOAL_REWARD = 1.0
SHAPING_CAP = 0.9
MIN_SUBGOAL_SIZE = 2


@dataclass(frozen=True)
class RewardConfig:
    """The reward knobs; defaults ARE the mandated design."""

    step_cost: float = STEP_COST
    gamma: float = REWARD_GAMMA
    phi_scale: float = PHI_SCALE
    subgoal_bonus: float = SUBGOAL_BONUS
    goal_reward: float = GOAL_REWARD
    shaping_cap: float = SHAPING_CAP
    min_subgoal_size: int = MIN_SUBGOAL_SIZE


# --------------------------------------------------------------------------
# goal_similarity
# --------------------------------------------------------------------------

def match_count(target: Node, cand: Optional[Node]) -> int:
    """Nodes of ``target`` covered by the largest subtree canonically EQUAL to
    ``cand`` (0 when there is no such subtree)."""
    if cand is None:
        return 0
    if target.canonical() == cand.canonical():
        return size(target)
    best = 0
    for child in target.children():
        m = match_count(child, cand)
        if m > best:
            best = m
    return best


def is_proper_subgoal(target: Node, top: Optional[Node],
                      min_size: int = MIN_SUBGOAL_SIZE) -> bool:
    """True iff ``top`` is a non-goal, non-trivial sub-structure of ``target``."""
    if top is None:
        return False
    if top.canonical() == target.canonical():
        return False
    n_top = size(top)
    if n_top < min_size or n_top >= size(target):
        return False
    return match_count(target, top) == n_top


def goal_similarity(target: Node, state) -> float:
    """Two-term similarity in ``[0, 1]`` of the stack TOP to ``target``."""
    top = state.stack[-1] if state.stack else None
    if top is None:
        return 0.0
    n_t = size(target)
    n_top = size(top)
    denom = max(n_t, n_top)
    size_prox = 0.0 if denom == 0 else 1.0 - abs(n_top - n_t) / denom
    match_frac = (match_count(target, top) / n_t) if n_t else 0.0
    return 0.5 * size_prox + 0.5 * match_frac


def phi(target: Node, state, scale: float = PHI_SCALE) -> float:
    return scale * goal_similarity(target, state)


# --------------------------------------------------------------------------
# Per-step reward
# --------------------------------------------------------------------------

@dataclass
class RewardBreakdown:
    """One step's reward, itemised (this is what the evidence traces show)."""

    step_cost: float = 0.0
    shaping: float = 0.0          # raw F = gamma*Phi(s') - Phi(s)
    bonus: float = 0.0            # raw +0.10 sub-goal bonus
    shaping_applied: float = 0.0  # shaping + bonus AFTER the per-episode cap
    final: float = 0.0            # +1.0 when the action reaches the goal
    total: float = 0.0
    capped: float = 0.0           # reward removed by the cap
    phi: float = 0.0
    phi_next: float = 0.0
    similarity: float = 0.0
    similarity_next: float = 0.0
    top: str = ""
    subgoal: bool = False

    def to_dict(self, step: int = -1, sid: str = "", target: str = "",
                action: str = "") -> dict:
        return {
            "step": step, "sid": sid, "action": action, "target": target,
            "step_cost": round(self.step_cost, 6),
            "shaping": round(self.shaping, 6),
            "subgoal_bonus": round(self.bonus, 6),
            "shaping_applied": round(self.shaping_applied, 6),
            "final_reward": round(self.final, 6),
            "total": round(self.total, 6),
            "capped": round(self.capped, 6),
            "phi": round(self.phi, 6),
            "phi_next": round(self.phi_next, 6),
            "similarity": round(self.similarity, 6),
            "similarity_next": round(self.similarity_next, 6),
            "top": self.top, "subgoal": bool(self.subgoal),
        }


class EpisodeShaper:
    """Stateful per-episode reward accumulator (the cap lives here).

    One instance spans ONE training/evaluation episode (a whole bundle run or a
    whole single-case run); call ``reset`` before a new episode.
    """

    def __init__(self, config: Optional[RewardConfig] = None) -> None:
        self.config = config or RewardConfig()
        self.positive_used = 0.0

    def reset(self) -> None:
        self.positive_used = 0.0

    def reward(self, target: Node, state, nxt, reached: bool) -> RewardBreakdown:
        cfg = self.config
        sim = goal_similarity(target, state)
        sim_n = goal_similarity(target, nxt)
        phi_s = cfg.phi_scale * sim
        phi_n = cfg.phi_scale * sim_n
        shaping = cfg.gamma * phi_n - phi_s

        top = nxt.stack[-1] if nxt.stack else None
        prev_top = state.stack[-1] if state.stack else None
        subgoal = False
        bonus = 0.0
        if is_proper_subgoal(target, top, cfg.min_subgoal_size):
            if match_count(target, top) > match_count(target, prev_top):
                bonus = cfg.subgoal_bonus
                subgoal = True

        pos = shaping + bonus
        applied = pos
        capped = 0.0
        if applied > 0.0:
            room = cfg.shaping_cap - self.positive_used
            if room < 0.0:
                room = 0.0
            if applied > room:
                capped = applied - room
                applied = room
            self.positive_used += applied

        final = cfg.goal_reward if reached else 0.0
        total = cfg.step_cost + applied + final
        return RewardBreakdown(
            step_cost=cfg.step_cost, shaping=shaping, bonus=bonus,
            shaping_applied=applied, final=final, total=total, capped=capped,
            phi=phi_s, phi_next=phi_n, similarity=sim, similarity_next=sim_n,
            top=(top.canonical() if top is not None else ""), subgoal=subgoal)

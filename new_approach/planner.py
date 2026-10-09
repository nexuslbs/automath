"""Explicit planner over the REDUCTION_TABLE (COMPARISON.md section 4, fix 3).

This module replaces the learned-score ``argmax`` at inference
(``EvolutionAgent._score`` / ``best_action``) with an EXPLICIT, deterministic
search/planning procedure over the semantic reduction table.  No genome, no
learned weights, no torch - pure standard library.

World
-----
The same ``(target, stack)`` stack machine as ``EvoEnv`` / ``MinimalEnv``: an
action pops ``arity`` nodes off the top of the stack and pushes one node.  Two
action alphabets exist and are selected by ``domain``:

* ``domain="evo"``  - ``evolution.EVO_ACTIONS`` (PushZero/PushOne/MakeChange +
  one build action per ``REDUCTION_TABLE`` row);
* ``domain="core"`` - ``env.BUILD_ACTIONS`` (the four core node types).

Algorithm (bounded, deterministic)
----------------------------------
1. ``word(target)`` is the canonical post-order build word from the empty stack:
   ``sem_plan`` for evo and ``env.plan`` for core.  A target of size ``n`` is
   reached from empty in exactly ``len(word)`` actions.
2. Any node that ever appears on the stack of a run that ends in ``(target,)``
   must be a SUBTREE of ``target`` (nodes are only ever consumed into larger
   nodes; the final stack is exactly the target).  Building such a subtree uses
   only actions that occur in ``word(target)``.  Therefore restricting the
   search to those actions is SOUND and COMPLETE for reaching the goal.
3. The planner first tries the exact prefix fast path (non-trivial prefixes of
   the canonical run, e.g. the ext114/core33 validation states).  Otherwise it
   runs a bounded best-first search over reachable stacks with a structural
   admissible-ish lower bound ``h`` (how many canonical build results are still
   missing from the stack).  States containing a non-subtree node are pruned at
   once (unsolvable by construction).
4. When the budget is exhausted or the state is unsolvable, ``plan`` returns
   ``None`` and ``best_action`` falls back to the first legal action in the
   domain's canonical order, so a legal action is always returned for a valid
   stack and the planner never crashes.

The procedure is deterministic for a fixed ``(target, stack)``: the candidate
order is a fixed tuple, the heap tie-break is a monotone counter, and no RNG is
used.
"""

from __future__ import annotations

import heapq
import itertools
from collections import Counter
from typing import Dict, List, Optional, Sequence, Tuple

from .env import BUILD_ACTIONS, ACTION_BY_NAME as CORE_ACTION_BY_NAME
from .env import State, plan as core_plan
from .evolution import EVO_ACTIONS, EVO_ACTION_BY_NAME, sem_plan
from .nodes import Node, iter_nodes

#: Explicit, documented bounds (COMPARISON.md next-fix #3 "bounded").
DEFAULT_MAX_SEARCH_NODES = 100000
DEFAULT_MAX_ACTIONS = 200
DEFAULT_MAX_STEPS = 200

EVO_ORDER: Tuple[str, ...] = tuple(a.name for a in EVO_ACTIONS)
CORE_ORDER: Tuple[str, ...] = tuple(a.name for a in BUILD_ACTIONS)


class Planner:
    """Deterministic bounded search planner over a reduction-table action set.

    Parameters
    ----------
    actions:
        The action vocabulary (``EVO_ACTIONS`` or ``BUILD_ACTIONS``).
    domain:
        ``"evo"`` -> canonical word from ``sem_plan`` (reduction table);
        ``"core"`` -> canonical word from ``env.plan``.
    max_search_nodes:
        Hard cap on the number of expanded search nodes for ONE plan call.
    max_actions:
        Hard cap on the length of a returned plan (steps per episode).
    """

    def __init__(
        self,
        actions: Sequence = EVO_ACTIONS,
        domain: str = "evo",
        max_search_nodes: int = DEFAULT_MAX_SEARCH_NODES,
        max_actions: int = DEFAULT_MAX_ACTIONS,
        max_steps: int = DEFAULT_MAX_STEPS,
    ) -> None:
        if domain not in ("evo", "core"):
            raise ValueError("unknown domain %r" % (domain,))
        self.actions = tuple(actions)
        self.by_name = {a.name: a for a in self.actions}
        self.domain = domain
        self.order: Tuple[str, ...] = tuple(a.name for a in self.actions)
        self.max_search_nodes = int(max_search_nodes)
        self.max_actions = int(max_actions)
        self.max_steps = int(max_steps)
        self.target: Optional[Node] = None
        # caches keyed by canonical strings (target canonical -> derived data)
        self._word_cache: Dict[str, Tuple[str, ...]] = {}
        self._subtree_cache: Dict[str, frozenset] = {}
        self._required_cache: Dict[str, Counter] = {}
        self._next_cache: Dict[Tuple[str, str], Optional[str]] = {}

    # -- action machinery ------------------------------------------------
    def _apply(self, stack: Tuple[Node, ...], action) -> Tuple[Node, ...]:
        if action.arity == 0:
            return stack + (action.build(()),)
        popped = stack[-action.arity:]
        return stack[: len(stack) - action.arity] + (action.build(popped),)

    def _word(self, target: Node) -> Tuple[str, ...]:
        key = target.canonical()
        cached = self._word_cache.get(key)
        if cached is not None:
            return cached
        if self.domain == "evo":
            word = tuple(sem_plan(target))
        else:
            word = tuple(core_plan(target))
        self._word_cache[key] = word
        return word

    def _subtrees(self, target: Node) -> frozenset:
        key = target.canonical()
        cached = self._subtree_cache.get(key)
        if cached is not None:
            return cached
        subs = frozenset(n.canonical() for n in iter_nodes(target))
        self._subtree_cache[key] = subs
        return subs

    def _required(self, target: Node) -> Counter:
        key = target.canonical()
        cached = self._required_cache.get(key)
        if cached is not None:
            return cached
        required: Counter = Counter()
        stack: Tuple[Node, ...] = ()
        for name in self._word(target):
            stack = self._apply(stack, self.by_name[name])
            required[stack[-1].canonical()] += 1
        self._required_cache[key] = required
        return required

    # -- the structural lower bound --------------------------------------
    @staticmethod
    def _h(stack: Sequence[Node], required: Counter) -> int:
        have: Counter = Counter(n.canonical() for n in stack)
        missing = 0
        for canon, need in required.items():
            avail = have.get(canon, 0)
            if avail < need:
                missing += need - avail
        return missing

    # -- public planning -------------------------------------------------
    def set_target(self, target) -> None:
        self.target = target

    def fallback_action(self, stack: Sequence[Node]) -> Optional[str]:
        """The first legal action of the domain (arity <= len(stack))."""
        n = len(stack)
        for name in self.order:
            if self.by_name[name].arity <= n:
                return name
        return None

    def plan(self, target: Node, initial_stack: Sequence[Node] = ()
             ) -> Optional[List[str]]:
        """A bounded action word from ``initial_stack`` to ``(target,)``.

        Returns ``None`` when no word within the explicit budget is found.
        """
        if target is None:
            return None
        start = tuple(initial_stack)
        if start == (target,):
            return []
        word = self._word(target)
        if len(word) > self.max_actions:
            return None  # explicit step budget: cannot hold the whole word
        if not start:
            return list(word)
        if len(start) > self.max_actions:
            return None
        # 1) exact fast path: the stack is a non-trivial prefix of the run.
        prefix: Tuple[Node, ...] = ()
        for i, name in enumerate(word):
            prefix = self._apply(prefix, self.by_name[name])
            if prefix == start:
                return list(word[i + 1:])
        # 2) sound prune: every stack node must be a target subtree.
        subtrees = self._subtrees(target)
        if any(n.canonical() not in subtrees for n in start):
            return None
        # 3) bounded best-first search over the canonical action alphabet.
        return self._search(target, start, word, subtrees)

    def _search(self, target: Node, start: Tuple[Node, ...],
                word: Tuple[str, ...], subtrees: frozenset) -> Optional[List[str]]:
        required = self._required(target)
        depth_limit = min(self.max_actions, max(1, len(word)))
        # deterministic candidate order: first appearance in the canonical word
        seen = set()
        candidates: List[str] = []
        for name in word:
            if name not in seen:
                seen.add(name)
                candidates.append(name)
        counter = itertools.count()
        came: Dict[Tuple[Node, ...], Optional[Tuple[Tuple[Node, ...], str]]] = {
            start: None}
        gbest: Dict[Tuple[Node, ...], int] = {start: 0}
        heap: List[Tuple[int, int, int, Tuple[Node, ...]]] = [
            (self._h(start, required), 0, next(counter), start)]
        nodes = 0
        while heap:
            _f, g, _t, state = heapq.heappop(heap)
            if state == (target,):
                return self._reconstruct(came, state)
            if g >= depth_limit:
                continue
            for name in candidates:
                action = self.by_name[name]
                if action.arity > len(state):
                    continue
                nxt = self._apply(state, action)
                ng = g + 1
                if gbest.get(nxt, 1 << 30) <= ng:
                    continue
                if any(n.canonical() not in subtrees for n in nxt):
                    continue
                gbest[nxt] = ng
                came[nxt] = (state, name)
                nodes += 1
                if nxt == (target,):
                    return self._reconstruct(came, nxt)
                if nodes >= self.max_search_nodes:
                    return None
                heapq.heappush(heap, (ng + self._h(nxt, required), ng,
                                      next(counter), nxt))
        return None

    @staticmethod
    def _reconstruct(came, state) -> List[str]:
        actions: List[str] = []
        while came[state] is not None:
            prev, name = came[state]  # type: ignore[misc]
            actions.append(name)
            state = prev
        actions.reverse()
        return actions

    # -- agent-compatible interface (drop-in for rollout_eval) ------------
    def best_action(self, key: str, state: State) -> Optional[str]:
        """The next action for ``state``; always legal or ``None`` if goal."""
        if self.target is None:
            return self.fallback_action(state.stack)
        cache_key = (self.target.canonical(), state.canonical())
        if cache_key in self._next_cache:
            return self._next_cache[cache_key]
        actions = self.plan(self.target, state.stack)
        nxt = actions[0] if actions else self.fallback_action(state.stack)
        self._next_cache[cache_key] = nxt
        return nxt


def make_planners(max_search_nodes: int = DEFAULT_MAX_SEARCH_NODES,
                  max_actions: int = DEFAULT_MAX_ACTIONS
                  ) -> Tuple[Planner, Planner]:
    """(core_planner, evo_planner) with the explicit bounds applied."""
    core = Planner(BUILD_ACTIONS, domain="core",
                   max_search_nodes=max_search_nodes, max_actions=max_actions)
    evo = Planner(EVO_ACTIONS, domain="evo",
                  max_search_nodes=max_search_nodes, max_actions=max_actions)
    return core, evo


# --------------------------------------------------------------------------
# Evaluation-only rollout (no genome, no training)
# --------------------------------------------------------------------------

def _mean(xs: Sequence[float]) -> Optional[float]:
    return round(sum(xs) / len(xs), 4) if xs else None


def planner_rollout(planner: Planner, cases: Sequence) -> dict:
    """Greedy rollout of the PLANNER over ``cases``.

    Equivalent metric semantics to ``gen_common.rollout_eval`` (solved/total,
    mean actions on solved, mean reward ``(1 if solved else 0) - 0.05 * n``)
    plus PER-FORM solved flags.  One plan call per case (evaluation only).
    """
    import time

    total = len(cases)
    solved = 0
    acts: List[int] = []
    rewards: List[float] = []
    per_form: Dict[str, bool] = {}
    t0 = time.perf_counter()
    for case in cases:
        target = getattr(case.env.goal, "target", None)
        planner.set_target(target)
        state = case.env.reset()
        n = 0
        if target is not None:
            actions = planner.plan(target, state.stack) or []
            for name in actions:
                if n >= case.max_steps:
                    break
                state = case.env.step(state, name)
                n += 1
                if case.env.goal_achieved(state):
                    break
        else:
            # Non-fixed (pattern) goal: explicit fallback, legal steps only.
            for _ in range(case.max_steps):
                if case.env.goal_achieved(state):
                    break
                name = planner.fallback_action(state.stack)
                if name is None:
                    break
                state = case.env.step(state, name)
                n += 1
        done = bool(case.env.goal_achieved(state))
        per_form[case.name] = done
        if done:
            solved += 1
            acts.append(n)
        rewards.append((1.0 if done else 0.0) - 0.05 * n)
    return {"solved": solved, "total": total,
            "mean_actions_solved": _mean(acts),
            "mean_reward": _mean(rewards),
            "wall_secs": round(time.perf_counter() - t0, 3),
            "per_form": per_form}

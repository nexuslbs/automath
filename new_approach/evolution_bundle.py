"""Unit C: multi-state TOTAL-budget bundles, impossible goals and switching.

The operator mandate (thread 4315) asks for a scenario in which an agent faces
SEVERAL states/goals at once with ONE TOTAL step budget shared by all of them
(never a per-state budget), freely chooses which state to start first, and may
switch to another state in the middle when the current one looks impossible or
not worth the remaining budget.

This module implements exactly that surface:

* ``BundleState`` - one state/goal of a bundle.  ``feasible`` is DERIVED, not
  declared: a target is feasible iff every ``Group`` in it can be produced by
  some action of the extended action alphabet (``SEM_BY_KEY``); a ``Group``
  whose ``(tag, arity)`` has no build action can never appear on the stack and
  is therefore structurally IMPOSSIBLE.
* ``Bundle`` - a fixed tuple of states plus the single total step budget.
* ``run_bundle`` - the total-budget controller.  It picks the start state from
  the agent's STATE-PREFERENCE instinct (the genome), runs the agent's policy,
  and switches state when the current state stops making reward progress for
  ``switch_patience`` steps (stagnation trigger) or when it has already eaten a
  fixed fraction of the budget available when it was entered (budget-ratio
  trigger).  Every switch carries a reason, so the raw trace explains itself.
* ``BundleResult`` - per-state solved/partial/reward/steps, the total steps
  against the shared budget, and the ordered switch trace.

Reward model (identical for training and evaluation):
``+1.0`` when a feasible state is solved, ``-0.01`` for every step taken.  An
impossible state can therefore never yield positive reward: it can only burn
budget, and the budget-ratio / stagnation triggers cut the loss short.

Pure standard library, seeded and bounded.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

from .env import FixedStateGoal, State
from .evolution import (
    SEM_BY_KEY,
    EvoEnv,
    ev_add,
    ev_div,
    ev_if,
    ev_lt,
    ev_mul,
    ev_not,
    sem_plan,
)
from .nodes import Group, Node, One, Zero, nat, size

#: Reserved tag that no semantic build action can ever produce (arity 2).
T_IMPOSSIBLE = 99

GOAL_REWARD = 1.0
STEP_PENALTY = -0.01

#: A state entered with ``entry_remaining`` budget is abandoned when it has
#: spent this fraction of it without solving.
BUDGET_RATIO_TRIGGER = 0.5


@dataclass(frozen=True)
class BundleState:
    sid: str
    name: str
    target: Node
    feasible: bool
    initial_stack: Tuple[Node, ...] = ()
    expected_len: int = 0
    level: str = ""
    impossible_reason: str = ""

    def key(self) -> str:
        return "B:" + self.target.canonical()

    def to_dict(self) -> dict:
        return {
            "sid": self.sid,
            "name": self.name,
            "target": self.target.canonical(),
            "feasible": self.feasible,
            "expected_len": self.expected_len,
            "level": self.level,
            "impossible_reason": self.impossible_reason,
        }


def target_feasible(target: Node) -> Tuple[bool, str]:
    """Derived feasibility: no build action -> structurally impossible."""
    from .nodes import Change, Group as G

    def walk(n: Node) -> Optional[str]:
        if isinstance(n, G):
            from .axioms import AxiomSet

            try:
                tag_value = AxiomSet().evaluate(n.tag)
            except Exception:
                return "tag is not an evaluable integer"
            if not isinstance(tag_value, int):
                return "tag is not an integer"
            if (tag_value, n.arity()) not in SEM_BY_KEY:
                return ("no build action for tag=%d arity=%d"
                        % (tag_value, n.arity()))
            for item in n.items:
                why = walk(item)
                if why:
                    return why
            # The tag itself must also be buildable.
            return walk(n.tag)
        if isinstance(n, Change):
            return walk(n.child)
        return None

    why = walk(target)
    return (why is None), (why or "")


def _feasible_state(sid: str, name: str, target: Node, level: str
                    ) -> BundleState:
    ok, why = target_feasible(target)
    if not ok:
        raise ValueError("declared feasible but not buildable: %s (%s)"
                         % (name, why))
    return BundleState(sid, name, target, True,
                       expected_len=len(sem_plan(target)), level=level)


def _impossible_state(sid: str, name: str, target: Node) -> BundleState:
    ok, why = target_feasible(target)
    if ok:
        raise ValueError("declared impossible but a build action exists: %s"
                         % name)
    return BundleState(sid, name, target, False, level="impossible",
                       impossible_reason=why)


def demo_states() -> Tuple[BundleState, ...]:
    """A mix of easy/medium/expensive feasible states plus impossible ones.

    ``s4`` is structurally impossible: tag 99 / arity 2 has no build action.
    ``s6`` is impossible for a second, independent reason: a ``Group`` whose
    arity (5) has no build action even though its tag (``T_SEQ``) exists.
    """
    return (
        _feasible_state("s0", "not(0)", ev_not(Zero()), "easy"),
        _feasible_state("s1", "add(1,1)", ev_add(One(), One()), "easy"),
        _feasible_state("s2", "mul(1,3)", ev_mul(One(), nat(3)), "easy"),
        _feasible_state("s3", "if(lt(1,2),1,0)",
                        ev_if(ev_lt(One(), nat(2)), One(), Zero()), "medium"),
        _impossible_state("s4", "impossible_tag99",
                          Group(nat(T_IMPOSSIBLE), (One(), Zero()))),
        _feasible_state("s5", "div(mul(3,3),3)",
                        ev_div(ev_mul(nat(3), nat(3)), nat(3)), "expensive"),
        _impossible_state("s6", "impossible_seq5",
                          Group(nat(10), (One(), Zero(), One(), Zero(),
                                          One()))),
    )


@dataclass
class Bundle:
    name: str
    states: Tuple[BundleState, ...]
    total_budget: int

    def ids(self) -> Tuple[str, ...]:
        return tuple(s.sid for s in self.states)

    def by_id(self, sid: str) -> BundleState:
        for s in self.states:
            if s.sid == sid:
                return s
        raise KeyError(sid)

    def feasible(self) -> Tuple[BundleState, ...]:
        return tuple(s for s in self.states if s.feasible)

    def impossible(self) -> Tuple[BundleState, ...]:
        return tuple(s for s in self.states if not s.feasible)

    def index(self, sid: str) -> int:
        return self.ids().index(sid)

    def fingerprint(self) -> str:
        h = hashlib.sha256()
        h.update(("budget=%d|" % self.total_budget).encode())
        for s in self.states:
            h.update(("%s|%s|%s|" % (s.sid, s.target.canonical(),
                                     s.feasible)).encode())
        return h.hexdigest()

    def to_dict(self) -> dict:
        return {"name": self.name, "total_budget": self.total_budget,
                "fingerprint": self.fingerprint(),
                "states": [s.to_dict() for s in self.states]}


def make_demo_bundle(total_budget: int = 60) -> Bundle:
    return Bundle("demo-bundle", demo_states(), total_budget)


# --------------------------------------------------------------------------
# The total-budget controller
# --------------------------------------------------------------------------

@dataclass
class TraceEvent:
    step: int
    kind: str          # start | switch | solve | finish
    sid: str
    detail: str


@dataclass
class StateResult:
    sid: str
    name: str
    feasible: bool
    started: bool = False
    solved: bool = False
    steps: int = 0
    reward: float = 0.0
    partial: float = 0.0       # best structural progress observed (0..1)
    switches_away: int = 0

    def status(self) -> str:
        if self.solved:
            return "solved"
        if self.steps == 0:
            return "untouched"
        if not self.feasible:
            return "impossible-skipped"
        return "partial" if self.partial > 0.0 else "unsolved"


@dataclass
class BundleResult:
    per_state: Dict[str, StateResult]
    total_steps: int
    total_reward: float
    solved_feasible: int
    feasible_total: int
    switched: int
    trace: List[TraceEvent] = field(default_factory=list)
    trajectory: List[Tuple[str, float, float, str]] = field(default_factory=list)

    def solver_score(self) -> float:
        score = 0.0
        for sr in self.per_state.values():
            if not sr.feasible:
                continue
            score += 1.0 if sr.solved else sr.partial
        return score

    def chooser_score(self) -> float:
        wasted = sum(sr.steps for sr in self.per_state.values()
                     if not sr.feasible)
        return self.solved_feasible - (wasted / max(1, self.total_steps))

    def per_state_rows(self) -> List[dict]:
        out = []
        for sid in self.per_state:
            sr = self.per_state[sid]
            out.append({
                "sid": sr.sid, "name": sr.name,
                "feasible": sr.feasible, "status": sr.status(),
                "started": sr.started, "solved": sr.solved,
                "steps": sr.steps, "reward": round(sr.reward, 4),
                "partial": round(sr.partial, 3),
                "switches_away": sr.switches_away,
            })
        return out


def _progress(state: State, target: Node) -> float:
    if not state.stack:
        return 0.0
    top = state.stack[-1]
    return min(1.0, size(top) / max(1, size(target)))


def start_order(bundle: Bundle, state_pref: Sequence[float]
                ) -> List[str]:
    """States ranked by the agent's STATE-PREFERENCE instinct (descending)."""
    pairs = list(zip(bundle.ids(), list(state_pref)))
    pairs.sort(key=lambda p: (-p[1], p[0]))
    return [p[0] for p in pairs]


def run_bundle(
    agent,
    bundle: Bundle,
    state_pref: Sequence[float],
    total_budget: Optional[int] = None,
    training: bool = False,
    epsilon: float = 0.0,
    max_states: int = 4,
) -> BundleResult:
    """Run one agent over the WHOLE bundle under ONE shared step budget.

    ``training`` collects the reward/td/context trajectory for the caller to
    feed into ``agent.update_pref``.  Evaluation is greedy (no exploration).
    """
    budget = total_budget if total_budget is not None else bundle.total_budget
    envs = {s.sid: EvoEnv(FixedStateGoal(s.target),
                          initial_stack=s.initial_stack,
                          max_actions=budget + 4)
            for s in bundle.states}
    per_state = {s.sid: StateResult(s.sid, s.name, s.feasible)
                 for s in bundle.states}
    trace: List[TraceEvent] = []
    trajectory: List[Tuple[str, float, float, str]] = []
    order = start_order(bundle, state_pref)
    order_pos = {sid: i for i, sid in enumerate(order)}
    pref_by_sid = {sid: float(state_pref[bundle.index(sid)])
                   for sid in order}
    # Within one sweep the agent tries every unsolved state once (ranked by its
    # state-preference instinct) before it may revisit one it abandoned; that
    # way a state that looked impossible/not-worth does not immediately eat the
    # budget again.
    swept: set = set()
    remaining = budget
    step_no = 0
    total_reward = 0.0
    solved = 0
    switched = 0

    def choose_next(exclude: Optional[str]) -> Optional[str]:
        cands = [sid for sid in order
                 if sid != exclude and not per_state[sid].solved
                 and sid not in swept]
        if not cands:
            swept.clear()
            cands = [sid for sid in order
                     if sid != exclude and not per_state[sid].solved]
        if not cands:
            return None
        cands.sort(key=lambda sid: (-pref_by_sid[sid], order_pos[sid]))
        return cands[0]

    current = choose_next(None)
    if current is not None:
        per_state[current].started = True
        env = envs[current]
        state = env.reset()
        steps_in = 0
        entry_remaining = remaining
        last_improve = 0
        best_partial = 0.0
        trace.append(TraceEvent(step_no, "start", current,
                                "instinct order=%s" % (order,)))
    else:
        state = None  # type: ignore[assignment]
        steps_in = entry_remaining = last_improve = 0
        env = None  # type: ignore[assignment]
        best_partial = 0.0

    while remaining > 0 and current is not None and state is not None:
        env = envs[current]
        sr = per_state[current]
        swept.add(current)
        goal = env.goal_achieved(state)
        if goal:
            sr.solved = True
            solved += 1
            total_reward += GOAL_REWARD
            trace.append(TraceEvent(step_no, "solve", current,
                                    "goal reached in %d steps" % sr.steps))
            nxt_sid = choose_next(current)
            current = nxt_sid
            if current is not None:
                per_state[current].started = True
                state = envs[current].reset()
                steps_in = 0
                entry_remaining = remaining
                last_improve = 0
                best_partial = 0.0
                trace.append(TraceEvent(
                    step_no, "start", current,
                    "after solve order=%s" % (order,)))
            continue

        key = bundle.by_id(current).key()
        if training:
            action = agent.epsilon_greedy(key, state, epsilon)
        else:
            action = agent.best_action(key, state)
            if action is None:
                action = agent.epsilon_greedy(key, state, 0.0)
        ctx = agent._ctx(state)
        nxt = env.step(state, action)
        reached = env.goal_achieved(nxt)
        reward = GOAL_REWARD if reached else STEP_PENALTY
        if training:
            td = agent.learn(key, state, action, reward, nxt, reached)
            trajectory.append((action, reward, td, ctx))
        remaining -= 1
        step_no += 1
        steps_in += 1
        sr.steps += 1
        sr.reward += reward
        total_reward += reward
        prog = _progress(nxt, bundle.by_id(current).target)
        if prog > best_partial:
            best_partial = prog
            sr.partial = max(sr.partial, prog)
            last_improve = steps_in
        state = nxt
        if reached:
            sr.solved = True
            solved += 1
            trace.append(TraceEvent(step_no, "solve", current,
                                    "goal reached in %d steps" % sr.steps))
            nxt_sid = choose_next(current)
            current = nxt_sid
            if current is not None:
                per_state[current].started = True
                state = envs[current].reset()
                steps_in = 0
                entry_remaining = remaining
                last_improve = 0
                best_partial = 0.0
                trace.append(TraceEvent(
                    step_no, "start", current,
                    "after solve order=%s" % (order,)))
            continue

        # -- switching triggers ----------------------------------------
        patience = int(getattr(agent, "switch_patience", 6))
        reason = None
        if steps_in - last_improve >= patience:
            reason = ("stagnation: %d steps without progress (patience=%d)"
                      % (steps_in - last_improve, patience))
        elif steps_in >= max(2, int(BUDGET_RATIO_TRIGGER * entry_remaining)):
            reason = ("budget-ratio: %d of %d steps since entry"
                      % (steps_in, entry_remaining))
        if reason is not None:
            per_state[current].switches_away += 1
            nxt_sid = choose_next(current)
            switched += 1
            trace.append(TraceEvent(step_no, "switch", current,
                                    reason + " -> " + str(nxt_sid)))
            current = nxt_sid
            if current is not None:
                per_state[current].started = True
                state = envs[current].reset()
                steps_in = 0
                entry_remaining = remaining
                last_improve = 0
                best_partial = 0.0
            continue

    trace.append(TraceEvent(step_no, "finish", current or "-",
                            "budget exhausted: %d steps total" % step_no))
    solved = sum(1 for sr in per_state.values() if sr.solved)
    result = BundleResult(
        per_state=per_state,
        total_steps=step_no,
        total_reward=total_reward,
        solved_feasible=solved,
        feasible_total=len(bundle.feasible()),
        switched=switched,
        trace=trace,
        trajectory=trajectory,
    )
    return result


def trace_rows(result: BundleResult) -> List[dict]:
    return [{"step": e.step, "kind": e.kind, "sid": e.sid,
             "detail": e.detail} for e in result.trace]


def render_trace(result: BundleResult, bundle: Bundle) -> str:
    lines = ["step  kind     state  detail"]
    for e in result.trace:
        lines.append("%4d  %-8s %-5s  %s" % (e.step, e.kind, e.sid, e.detail))
    lines.append("total steps=%d/%d reward=%+.3f solved=%d/%d switched=%d"
                 % (result.total_steps, bundle.total_budget, result.total_reward,
                    result.solved_feasible, result.feasible_total,
                    result.switched))
    return "\n".join(lines)

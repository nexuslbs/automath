"""Data flywheel for llm_like (design section 5).

The loop implemented here is exactly::

    roll out with the current policy
      -> classify every candidate form with the decidable solvability oracle
         (UNSOLVABLE dropped, UNKNOWN bounded attempt, SOLVABLE kept)
      -> keep ONLY the episodes that reached the goal
      -> append them to a bounded FIFO replay buffer (max_buffer_steps = 200000)
      -> retrain (llm_like.train) -> eval point (llm_like.eval)

The environment stays the frozen stdlib ``dynamic_env`` engine; trajectories are
plain JSON-serializable records (a superset of ``harness.py``'s ``collect=True``
trace) and the only torch consumer is the model.

``solve_rate = solved / SOLVABLE`` with the ``UNKNOWN`` and ``UNSOLVABLE`` counts
reported next to it, per ``docs/evolution/UNSOLVABLE.md`` sections 6 and the
design section 5.2.
"""

from __future__ import annotations

import os
import random
import time
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Optional, Sequence, Set, Tuple

import torch

from dynamic_env.engine import Action, DynamicEnv, State
from dynamic_env.spec import Spec, load_spec

from . import features as F
from .model import LLMAgentNet

# --------------------------------------------------------------------------
# Constants (static reward shaping identical to EvoConfig / harness.py)
# --------------------------------------------------------------------------

MAX_BUFFER_STEPS = 200000
STEP_COST = 0.05
INTERMEDIATE_REWARD = 0.30
GUARD_BONUS = 0.20
SUBGOAL_REWARD = 0.50
GOAL_REWARD = 1.00

#: The four shipped training specs and the two shipped held-out specs.
TRAIN_SPEC_IDS: Tuple[str, ...] = (
    "spec_minimal",
    "spec_multi_step",
    "spec_dynamic_axiom",
    "spec_dynamic_group",
)
HELDOUT_SPEC_IDS: Tuple[str, ...] = (
    "spec_multi_step_heldout",
    "spec_dynamic_group_deep",
)
#: Fresh held-out pool seed, distinct from the training seed and VAL_SEED.
EVAL_SEED = 424242

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.dirname(_HERE)
_SPEC_DIR = os.path.join(_REPO_ROOT, "data", "dynamic_env")
_HELDOUT_DIR = os.path.join(_SPEC_DIR, "heldout")


def spec_path(spec_id: str) -> str:
    """Path of a shipped spec; held-out ids live under ``heldout/``."""
    if spec_id in HELDOUT_SPEC_IDS:
        return os.path.join(_HELDOUT_DIR, spec_id + ".json")
    return os.path.join(_SPEC_DIR, spec_id + ".json")


def load_spec_by_id(spec_id: str) -> Spec:
    return load_spec(spec_path(spec_id))


# --------------------------------------------------------------------------
# Case + solvability filter
# --------------------------------------------------------------------------

@dataclass
class Case:
    """A spec plus an optional start state and an optional replay witness."""

    name: str
    spec: Spec
    start_state: Optional[State] = None
    witness: Optional[Tuple[str, ...]] = None
    kind: str = "train"
    max_steps: Optional[int] = None

    def form_key(self) -> str:
        start = self.start_state.identity() if self.start_state is not None else "reset"
        return self.spec.spec_id + "|" + start

    def step_budget(self) -> int:
        if self.max_steps is not None:
            return int(self.max_steps)
        return int(self.spec.max_steps)


def classify_case(case: Case, action_set: str = "current",
                  node_budget: int = 30000, time_budget: float = 3.0
                  ) -> Dict[str, Any]:
    """Call the decidable oracle; evaluation only, stdlib, no torch."""
    from evolution_trainer.solvability import classify

    return classify(case, action_set=action_set, node_budget=node_budget,
                    time_budget=time_budget)


def filter_pool(cases: Sequence[Case], action_set: str = "current",
                node_budget: int = 30000, time_budget: float = 3.0
                ) -> Dict[str, Any]:
    """Apply the design 5.2 rule to a pool.

    ``UNSOLVABLE`` -> dropped; ``UNKNOWN`` and ``SOLVABLE`` -> eligible. Returns
    the active pool, the dropped names, per-case verdicts and the three-way
    counts used to audit ``solve_rate``.
    """
    from evolution_trainer.solvability import SOLVABLE, UNKNOWN, UNSOLVABLE

    active: List[Case] = []
    excluded: List[str] = []
    verdicts: Dict[str, Any] = {}
    counts = {SOLVABLE: 0, UNKNOWN: 0, UNSOLVABLE: 0}
    for case in cases:
        result = classify_case(case, action_set=action_set,
                               node_budget=node_budget, time_budget=time_budget)
        verdict = result.get("verdict", UNKNOWN)
        verdicts[case.name] = result
        counts[verdict] = counts.get(verdict, 0) + 1
        if verdict == UNSOLVABLE:
            excluded.append(case.name)
        else:
            active.append(case)
    return {
        "active": active,
        "excluded": excluded,
        "verdicts": verdicts,
        "counts": counts,
    }


def solve_rate(solved: int, counts: Dict[str, int]) -> float:
    """``solve_rate = solved / SOLVABLE`` (UNSOLVABLE/UNKNOWN never inflate it)."""
    denominator = int(counts.get("SOLVABLE", 0))
    return float(solved) / float(denominator) if denominator else 0.0


# --------------------------------------------------------------------------
# Return-to-go (value target)
# --------------------------------------------------------------------------

def return_to_go(rewards: Sequence[float], gamma: float = 1.0) -> List[float]:
    """Normalized discounted return-to-go, min-max scaled into [0, 1]."""
    count = len(rewards)
    running = 0.0
    values = [0.0] * count
    for t in range(count - 1, -1, -1):
        running = float(rewards[t]) + gamma * running
        values[t] = running
    if not values:
        return values
    low, high = min(values), max(values)
    if high - low <= 1e-9:
        return [0.0] * count
    return [(value - low) / (high - low) for value in values]


def _subgoal_reached(flags: Sequence[int], target: Sequence[int],
                     mask: Sequence[int]) -> bool:
    return all(flags[i] == target[i] for i in range(len(target)) if mask[i])


# --------------------------------------------------------------------------
# Rollout
# --------------------------------------------------------------------------

def rollout_episode(net: LLMAgentNet, case: Case, rng: random.Random,
                    epsilon: float = 0.1, mask_illegal: bool = False,
                    allow_pop: bool = False, max_steps: Optional[int] = None,
                    subgoal_mask: Optional[Sequence[int]] = None,
                    round_index: int = 0, episode_index: int = 0,
                    scripted_keys: Optional[Sequence[str]] = None
                    ) -> Dict[str, Any]:
    """One episode of the current policy from ``case``'s start state.

    ``scripted_keys`` (optional) replays a fixed action-key word instead of the
    policy; it is used only for the optional one-time BFS-witness seed of the
    buffer (design section 5.1), never as a training signal.

    Returns ``{"solved", "steps", "records", ...}``. Each record is a
    self-contained per-step superset of the ``harness.py`` collect trace; the
    context needed by the attention head is rebuilt from the records at training
    time so no stale activations are stored.
    """
    started = time.perf_counter()
    spec = case.spec
    env = DynamicEnv(spec, allow_pop=allow_pop)
    state = case.start_state if case.start_state is not None else env.reset()
    obj_ids = sorted(spec.objectives)
    target = tuple(int(spec.objectives[oid]) for oid in obj_ids)
    if subgoal_mask is None:
        subgoal_mask = tuple(1 for _ in obj_ids)
    limit = int(max_steps if max_steps is not None else case.step_budget())

    credited: Set[int] = set()
    unlocked: Set[str] = set()
    seen: Set[str] = {state.identity()}
    history_states: List[torch.Tensor] = []
    history_actions: List[torch.Tensor] = []
    records: List[Dict[str, Any]] = []

    actions = env.legal_actions(state)
    prev_flags = env.objective_vector(state)
    set_legal = frozenset(a.objective_id for a in actions if a.kind == "set")
    guarded = frozenset(spec.guards)
    steps = 0
    solved = bool(env.goal_reached(state))

    while steps < limit and actions:
        sinp = F.state_inputs(env, state, subgoal_mask)
        if mask_illegal:
            from evolution_trainer.size_selection import masked_legal_actions

            masked = masked_legal_actions(env, state, seen, actions=actions)
            candidate_actions = list(masked) if masked else list(actions)
        else:
            candidate_actions = list(actions)

        ainp = F.action_inputs(env, state, candidate_actions,
                               sinp["node_ids"])
        if scripted_keys is not None:
            want = scripted_keys[steps] if steps < len(scripted_keys) else None
            chosen = next((i for i, candidate in enumerate(candidate_actions)
                           if candidate.key() == want), None)
            if chosen is None:
                break
            pooled = None
            chosen_latent = None
        else:
            with torch.no_grad():
                pooled, node_latent = net.encode_graph(
                    sinp["node_features"], sinp["type_idx"], sinp["adj"])
                action_latent = net.action_latents(
                    ainp["action_features"], ainp["action_ref_local"],
                    node_latent)
                tokens, ctx_mask = net.context_tokens(history_states,
                                                      history_actions, pooled)
                context_vec = net.attend(tokens, ctx_mask)
                logits = net.policy_logits(context_vec, action_latent)
            if epsilon > 0.0 and rng.random() < epsilon:
                chosen = rng.randrange(len(candidate_actions))
            else:
                chosen = max(range(len(candidate_actions)),
                             key=lambda i: (float(logits[i]), -i))
            chosen_latent = action_latent[chosen]
        action = candidate_actions[chosen]

        result = env.step(state, action)
        flags = env.objective_vector(result.state)
        reward = -STEP_COST
        for i in range(len(target)):
            if (subgoal_mask[i] and i not in credited
                    and prev_flags[i] != target[i] and flags[i] == target[i]):
                reward += INTERMEDIATE_REWARD
                credited.add(i)
        next_actions = env.legal_actions(result.state)
        next_set_legal = frozenset(
            a.objective_id for a in next_actions if a.kind == "set")
        new_unlocks = ((next_set_legal - set_legal) & guarded) - unlocked
        if new_unlocks:
            reward += GUARD_BONUS * len(new_unlocks)
            unlocked |= new_unlocks
        done_full = bool(result.done)
        done_sub = _subgoal_reached(flags, target, subgoal_mask)
        if done_full:
            reward += GOAL_REWARD
        elif done_sub:
            reward += SUBGOAL_REWARD

        ninp = F.state_inputs(env, result.state, subgoal_mask)
        records.append({
            "round": round_index,
            "spec_id": spec.spec_id,
            "case": case.name,
            "episode": episode_index,
            "t": steps,
            "node_features": sinp["node_features"],
            "type_idx": sinp["type_idx"],
            "adj": sinp["adj"],
            "action_features": ainp["action_features"],
            "action_ref_local": ainp["action_ref_local"],
            "action_keys": ainp["action_keys"],
            "chosen": int(chosen),
            "chosen_key": action.key(),
            "objective_vector": list(prev_flags),
            "target": list(target),
            "next_node_features": ninp["node_features"],
            "next_type_idx": ninp["type_idx"],
            "next_adj": ninp["adj"],
            "reward": round(float(reward), 8),
            "goal": bool(done_full),
            "return_to_go": 0.0,
        })

        if pooled is not None:
            history_states.append(pooled)
            history_actions.append(chosen_latent)
        prev_flags = flags
        state = result.state
        seen.add(state.identity())
        actions = next_actions
        set_legal = next_set_legal
        steps += 1
        solved = solved or done_full
        if done_full or done_sub:
            break

    solved = bool(env.goal_reached(state))
    targets = return_to_go([record["reward"] for record in records])
    for record, value in zip(records, targets):
        record["return_to_go"] = value
    return {
        "solved": solved,
        "steps": steps,
        "records": records,
        "case": case.name,
        "spec_id": spec.spec_id,
        "form_key": case.form_key(),
        "wall_s": round(time.perf_counter() - started, 6),
    }


# --------------------------------------------------------------------------
# Bounded FIFO replay buffer
# --------------------------------------------------------------------------

class FifoStepBuffer:
    """A FIFO buffer bounded by ``max_buffer_steps`` STEP records.

    Whole episodes are appended; the OLDEST episodes are evicted first until
    the total step count returns under the cap, so episode boundaries are never
    broken and attention can never cross an episode boundary.
    """

    def __init__(self, max_steps: int = MAX_BUFFER_STEPS) -> None:
        self.max_steps = int(max_steps)
        self.episodes: List[Dict[str, Any]] = []
        self._steps = 0

    def __len__(self) -> int:
        return self._steps

    @property
    def total_steps(self) -> int:
        return self._steps

    def add_episode(self, episode: Dict[str, Any]) -> None:
        records = episode.get("records", ())
        if not records:
            return
        self.episodes.append(episode)
        self._steps += len(records)
        self._evict()

    def _evict(self) -> None:
        while self.episodes and self._steps > self.max_steps:
            evicted = self.episodes.pop(0)
            self._steps -= len(evicted.get("records", ()))

    def covered_forms(self) -> Set[str]:
        return {episode.get("form_key", "") for episode in self.episodes}

    def iter_records(self) -> Iterable[Dict[str, Any]]:
        for episode in self.episodes:
            for record in episode.get("records", ()):
                yield record

    def stats(self) -> Dict[str, Any]:
        return {
            "episodes": len(self.episodes),
            "steps": self._steps,
            "max_steps": self.max_steps,
            "solved_episodes": sum(1 for e in self.episodes if e.get("solved")),
        }


# --------------------------------------------------------------------------
# Pools
# --------------------------------------------------------------------------

def training_cases(include_val64: bool = False,
                   include_heldout: bool = False) -> List[Case]:
    """The flywheel's rollout pool (default: the four shipped training specs).

    The two held-out specs and the ``validation_pool(64)`` are available behind
    explicit flags so a run can reproduce design section 5.1 literally, but the
    default keeps them out of training as design section 6.2 requires.
    """
    spec_ids = list(TRAIN_SPEC_IDS)
    if include_heldout:
        spec_ids += list(HELDOUT_SPEC_IDS)
    cases = [Case(name="train_" + spec_id, spec=load_spec_by_id(spec_id),
                  start_state=None, kind="train")
             for spec_id in spec_ids]
    if include_val64:
        from evolution_trainer.size_selection import validation_pool

        for val_case in validation_pool(64):
            cases.append(Case(name="val64_" + val_case.name, spec=val_case.spec,
                              start_state=val_case.start_state,
                              witness=tuple(val_case.witness), kind="val64"))
    return cases


def fresh_pool(size: int = 64, seed: int = EVAL_SEED,
               spec_ids: Optional[Sequence[str]] = None) -> List[Case]:
    """A FRESH held-out pool, disjoint from the canonical training forms.

    The pool builder of ``size_selection`` caches by size and ignores a second
    seed, so this rebuilds the stream with the requested seed.
    """
    from evolution_trainer.size_selection import (
        HELDOUT_SPEC_IDS as SS_HELDOUT,
        TRAIN_SPEC_IDS as SS_TRAIN,
        _candidate_stream,
        forbidden_forms,
    )

    ids = tuple(spec_ids) if spec_ids else tuple(SS_TRAIN) + tuple(SS_HELDOUT)
    forbidden: Set[str] = set()
    for forms in forbidden_forms().values():
        forbidden |= forms
    rng = random.Random(int(seed))
    seen: Set[str] = set()
    cases: List[Case] = []
    for val_case in _candidate_stream(rng, ids):
        key = val_case.spec_id + "|" + val_case.start_state.identity()
        if key in forbidden or key in seen:
            continue
        seen.add(key)
        cases.append(Case(name="fresh_" + val_case.name, spec=val_case.spec,
                          start_state=val_case.start_state,
                          witness=tuple(val_case.witness), kind="fresh"))
        if len(cases) >= int(size):
            break
    if len(cases) < int(size):
        raise RuntimeError("fresh pool too small: %d < %d"
                           % (len(cases), int(size)))
    return cases

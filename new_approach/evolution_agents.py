"""Stage 2: artificial reward/guidance scaffolding for the evolution branch.

A training-only wrapper emits a REWARD NODE after every step: ``GOOD`` when the
action matched the planner's intended step, ``BAD`` otherwise.  The agent is a
tabular Q-learner trained with ``use_demo=False`` (NO planner action-word
supervision): its ONLY learning signal is the guidance reward carried by that
node plus the terminal goal reward.  The reward node is removed at evaluation
time, so the learned policy is tested without the scaffolding.

The reward node semantics are exactly the operator's "some deterministic steps
will have a good reward, otherwise a bad reward".  Every state on the unique
optimal trajectory has an intended step; a state off that trajectory has none
and therefore earns BAD for whatever the agent does.

Pure standard library, seeded and bounded.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

from .env import BUILD_ACTIONS
from .evolution import EVO_ACTIONS
from .nodes import Group, Node, One, Zero, nat

# Reward node tag and the two guidance values it carries.
T_REWARD = 20
GOOD_VALUE = 0.5
BAD_VALUE = -0.5
GOAL_VALUE = 1.0
STEP_VALUE = -0.01


def reward_node(good: bool) -> Group:
    """The artificial reward NODE placed in the observation (Stage 2 only)."""
    return Group(nat(T_REWARD), (One() if good else Zero(),))


def read_reward(node: Group) -> float:
    """Read the scalar guidance value out of a reward node."""
    if not isinstance(node, Group) or node.arity() != 1:
        raise ValueError("not a reward node: %r" % (node,))
    val = node.items[0]
    if not isinstance(val, (Zero, One)):
        raise ValueError("bad reward payload: %r" % (val,))
    return GOOD_VALUE if isinstance(val, One) else BAD_VALUE


def _arity_map(actions) -> Dict[str, int]:
    return {a.name: a.arity for a in actions}


CORE_ORDER: Tuple[str, ...] = tuple(a.name for a in BUILD_ACTIONS)
CORE_ARITY: Dict[str, int] = _arity_map(BUILD_ACTIONS)
EVO_ORDER: Tuple[str, ...] = tuple(a.name for a in EVO_ACTIONS)
EVO_ARITY: Dict[str, int] = _arity_map(EVO_ACTIONS)


class Guidance:
    """The intended (deterministic) step for the scaffolded trajectory.

    For each case the unique optimal word is registered once; every prefix
    state maps to its next intended action in O(1).  A state that is NOT on the
    scaffolded trajectory has no intended step (``None``), so every action
    there earns a BAD reward.
    """

    def __init__(self, evo: bool = False) -> None:
        self.evo = evo
        self._onpath: Dict[str, Dict[str, str]] = {}
        self.hits = 0
        self.misses = 0

    def register(self, case_key: str, env, word: Sequence[str]) -> None:
        table = self._onpath.setdefault(case_key, {})
        state = env.reset()
        for action in word:
            table[state.canonical()] = action
            state = env.step(state, action)

    def intended(self, case_key: str, env, state) -> Optional[str]:
        table = self._onpath.get(case_key)
        if table is None:
            return None
        action = table.get(state.canonical())
        if action is None:
            self.misses += 1
        else:
            self.hits += 1
        return action


@dataclass
class TrainStats:
    episodes: int = 0
    successes: List[int] = field(default_factory=list)
    rewards: List[float] = field(default_factory=list)

    def curve(self, buckets: int = 12) -> List[str]:
        n = len(self.successes)
        if n == 0:
            return []
        step = max(1, n // buckets)
        out: List[str] = []
        for start in range(0, n, step):
            end = min(n, start + step)
            succ = sum(self.successes[start:end]) / (end - start)
            rew = sum(self.rewards[start:end]) / (end - start)
            out.append("ep %3d-%3d success=%.2f mean_reward=%+.3f"
                       % (start + 1, end, succ, rew))
        return out


@dataclass
class EvalResult:
    passed: int = 0
    total: int = 0
    per_name: Dict[str, bool] = field(default_factory=dict)
    walls: List[float] = field(default_factory=list)
    learned_states: int = 0


class GuidedQAgent:
    """Target-conditioned tabular Q-learning, deterministic given the seed."""

    def __init__(
        self,
        action_order: Sequence[str] = CORE_ORDER,
        arity: Optional[Dict[str, int]] = None,
        alpha: float = 0.5,
        gamma: float = 0.95,
        epsilon_start: float = 0.5,
        epsilon_end: float = 0.02,
        episodes: int = 60,
        seed: int = 20261009,
        step_reward: float = STEP_VALUE,
        goal_reward: float = GOAL_VALUE,
    ) -> None:
        self.action_order = tuple(action_order)
        self.arity = dict(arity) if arity is not None else CORE_ARITY
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon_start = epsilon_start
        self.epsilon_end = epsilon_end
        self.episodes = episodes
        self.seed = seed
        self.step_reward = step_reward
        self.goal_reward = goal_reward
        self.q: Dict[str, Dict[str, Dict[str, float]]] = {}
        self.rng = random.Random(seed)

    def valid(self, stack: Sequence[Node]) -> Tuple[str, ...]:
        return tuple(a for a in self.action_order
                     if self.arity.get(a, 0) <= len(stack))

    def _row(self, key: str, state) -> Dict[str, float]:
        return self.q.setdefault(key, {}).setdefault(state.canonical(), {})

    def values(self, key: str, state) -> Optional[Dict[str, float]]:
        return self.q.get(key, {}).get(state.canonical())

    def best_action(self, key: str, state) -> Optional[str]:
        row = self.values(key, state)
        if not row:
            return None
        valid = self.valid(state.stack)
        if not valid:
            return None
        best_a, best_v = None, float("-inf")
        for a in valid:
            v = row.get(a, 0.0)
            if v > best_v:
                best_v, best_a = v, a
        return best_a

    def _epsilon(self, episode: int) -> float:
        if self.episodes <= 1:
            return self.epsilon_end
        frac = episode / (self.episodes - 1)
        ratio = self.epsilon_end / self.epsilon_start
        return self.epsilon_start * (ratio ** frac)

    def epsilon_greedy(self, key: str, state, epsilon: float) -> str:
        valid = self.valid(state.stack)
        if self.rng.random() < epsilon:
            return self.rng.choice(valid)
        greedy = self.best_action(key, state)
        return greedy if greedy is not None else self.rng.choice(valid)

    def learn(self, key: str, state, action: str, reward: float, nxt,
              goal: bool) -> float:
        row = self._row(key, state)
        current = row.get(action, 0.0)
        bootstrap = 0.0 if goal else self._max_q(key, nxt)
        td = reward + self.gamma * bootstrap - current
        row[action] = current + self.alpha * td
        return td

    def _max_q(self, key: str, state) -> float:
        row = self.values(key, state)
        if not row:
            return 0.0
        valid = self.valid(state.stack)
        return max((row.get(a, 0.0) for a in valid), default=0.0)

    def digest(self) -> str:
        import hashlib
        h = hashlib.sha256()
        for key in sorted(self.q):
            h.update(key.encode())
            h.update(b"|")
            for state in sorted(self.q[key]):
                h.update(state.encode())
                h.update(b"|")
                row = self.q[key][state]
                for action in sorted(row):
                    h.update(("%s=%0.6f;" % (action, row[action])).encode())
        return h.hexdigest()

    def learned_states(self) -> int:
        return sum(len(states) for states in self.q.values())


def train_guided(
    agent: GuidedQAgent,
    cases: Sequence,
    episodes: int,
    guidance: Guidance,
) -> TrainStats:
    """Stage 2: reward/guidance nodes, NO planner action-word supervision."""
    stats = TrainStats()
    agent.rng = random.Random(agent.seed)
    for ep in range(episodes):
        epsilon = agent._epsilon(ep)
        success = 0
        total_reward = 0.0
        for case in cases:
            state = case.env.reset()
            for _ in range(case.max_steps):
                if case.env.goal_achieved(state):
                    success += 1
                    break
                action = agent.epsilon_greedy(case.key, state, epsilon)
                intended = guidance.intended(case.key, case.env, state)
                good = (action == intended)
                rnode = reward_node(good)          # artificial reward NODE
                nxt = case.env.step(state, action)
                goal = case.env.goal_achieved(nxt)
                reward = GOAL_VALUE if goal else read_reward(rnode)
                agent.learn(case.key, state, action, reward, nxt, goal)
                total_reward += reward
                state = nxt
                if goal:
                    success += 1
                    break
        stats.episodes += 1
        stats.successes.append(success)
        stats.rewards.append(total_reward)
    return stats


def evaluate(agent: GuidedQAgent, cases: Sequence,
             max_steps: Optional[int] = None) -> EvalResult:
    """Greedy evaluation with NO scaffolding."""
    import time
    result = EvalResult(total=len(cases))
    for case in cases:
        limit = max_steps if max_steps is not None else case.max_steps
        state = case.env.reset()
        t0 = time.perf_counter()
        for _ in range(limit):
            if case.env.goal_achieved(state):
                break
            action = agent.best_action(case.key, state)
            if action is None:
                break
            state = case.env.step(state, action)
        result.walls.append(time.perf_counter() - t0)
        passed = case.env.goal_achieved(state)
        result.passed += int(passed)
        result.per_name[case.name] = passed
    result.learned_states = agent.learned_states()
    return result

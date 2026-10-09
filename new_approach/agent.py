"""U2: deterministic agents for the minimal-node environment.

Two agents are implemented, both pure standard library and bounded:

``TabularAgent``
    A target-conditioned tabular Q-learning agent. The state value table is
    keyed by ``(goal_key, stack canonical form)`` so the same stack can map to
    different actions under different goals. Training is deterministic:
    a fixed seed drives the epsilon-greedy exploration and the table is
    populated in a fixed order. Two supervision modes:

    * ``use_demo=True``  - Monte-Carlo Q values are seeded from the planner's
      UNIQUE optimal trajectory (the deterministic planner output used as
      supervision, explicitly allowed by the dispatch), then refined by
      reward-driven Q-learning episodes.
    * ``use_demo=False`` - a pure reward-driven Q-learning control (no planner
      supervision at train time).

``CompositionalAgent``
    A structurally biased agent that LEARNS a local construction rule from the
    training trajectories: it records which action produced which node shape
    (kind, arity) and then builds any target by a post-order recursion that
    emits the learned action per node. It therefore generalizes to unseen
    targets built from shapes seen in training (the operator's "use the dynamic
    node to group basic nodes").

Every failure carries structured feedback ``state / expected / actual /
reason`` (see ``FailureFeedback`` in ``env.py``).
"""

from __future__ import annotations

import hashlib
import random
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

from .env import (
    ACTION_BY_NAME,
    BUILD_ACTIONS,
    FailureFeedback,
    MinimalEnv,
    State,
)
from .nodes import Change, Group, Node, One, Zero

ACTION_NAMES: Tuple[str, ...] = tuple(a.name for a in BUILD_ACTIONS)


def valid_actions(stack: Sequence[Node]) -> Tuple[str, ...]:
    """Actions whose arity is satisfiable by the current stack."""
    return tuple(a.name for a in BUILD_ACTIONS if a.arity <= len(stack))


@dataclass
class Step:
    action: str
    state: str
    reward: float
    goal: bool


@dataclass
class EpisodeResult:
    """The measured outcome of one agent episode."""

    passed: bool
    goal: str
    start: str
    trace: List[str] = field(default_factory=list)
    steps: List[Step] = field(default_factory=list)
    final_state: str = ""
    total_reward: float = 0.0
    reason: Optional[str] = None
    feedback: Optional[FailureFeedback] = None


def _node_shape(node: Node) -> Tuple[str, int]:
    kind = type(node).__name__
    arity = node.arity() if isinstance(node, Group) else 0
    return (kind, arity)


class TabularAgent:
    """Target-conditioned tabular Q-learning, deterministic given the seed."""

    def __init__(
        self,
        alpha: float = 0.5,
        gamma: float = 0.95,
        epsilon_start: float = 0.4,
        epsilon_end: float = 0.02,
        episodes: int = 60,
        seed: int = 20261009,
        step_reward: float = -0.01,
        goal_reward: float = 1.0,
        use_demo: bool = True,
    ) -> None:
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon_start = epsilon_start
        self.epsilon_end = epsilon_end
        self.episodes = episodes
        self.seed = seed
        self.step_reward = step_reward
        self.goal_reward = goal_reward
        self.use_demo = use_demo
        self.q: Dict[str, Dict[str, Dict[str, float]]] = {}
        self.rng = random.Random(seed)

    # ------------------------------------------------------------------
    # Q-table access
    # ------------------------------------------------------------------
    def _row(self, key: str, state: State) -> Dict[str, float]:
        return self.q.setdefault(key, {}).setdefault(state.canonical(), {})

    def known(self, key: str, state: State) -> bool:
        return state.canonical() in self.q.get(key, {})

    def values(self, key: str, state: State) -> Optional[Dict[str, float]]:
        return self.q.get(key, {}).get(state.canonical())

    def _max_q(self, key: str, state: State) -> float:
        row = self.values(key, state)
        if not row:
            return 0.0
        valid = valid_actions(state.stack)
        return max((row.get(a, 0.0) for a in valid), default=0.0)

    def best_action(self, key: str, state: State) -> Optional[str]:
        """Greedy action; ``None`` means the state was never learned."""
        row = self.values(key, state)
        if not row:
            return None
        valid = valid_actions(state.stack)
        best_value = max((row.get(a, float("-inf")) for a in valid),
                         default=float("-inf"))
        if best_value == float("-inf"):
            return None
        # Deterministic tie-break by the fixed action order.
        for a in valid:
            if row.get(a, float("-inf")) == best_value:
                return a
        return None

    def _epsilon(self, episode: int) -> float:
        if self.episodes <= 1:
            return self.epsilon_end
        frac = episode / (self.episodes - 1)
        ratio = self.epsilon_end / self.epsilon_start
        return self.epsilon_start * (ratio ** frac)

    def _epsilon_greedy(self, key: str, state: State, epsilon: float) -> str:
        valid = valid_actions(state.stack)
        if self.rng.random() < epsilon:
            return self.rng.choice(valid)
        greedy = self.best_action(key, state)
        return greedy if greedy is not None else self.rng.choice(valid)

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------
    def observe_trajectory(self, key: str, env: MinimalEnv,
                           word: Sequence[str]) -> None:
        """Seed Q with Monte-Carlo returns along a supervised optimal word."""
        states = [env.reset()]
        for action in word:
            states.append(env.step(states[-1], action))
        last = len(word) - 1
        returns = 0.0
        for idx in range(last, -1, -1):
            reward = self.goal_reward if idx == last else self.step_reward
            returns = reward + self.gamma * returns
            self._row(key, states[idx])[word[idx]] = returns

    def q_episode(self, key: str, env: MinimalEnv, max_steps: int,
                  epsilon: float) -> None:
        state = env.reset()
        for _ in range(max_steps):
            if env.goal_achieved(state):
                return
            action = self._epsilon_greedy(key, state, epsilon)
            nxt = env.step(state, action)
            goal = env.goal_achieved(nxt)
            reward = self.goal_reward if goal else self.step_reward
            row = self._row(key, state)
            current = row.get(action, 0.0)
            bootstrap = 0.0 if goal else self._max_q(key, nxt)
            row[action] = current + self.alpha * (
                reward + self.gamma * bootstrap - current
            )
            state = nxt

    def train(self, problems: Sequence[Tuple[str, MinimalEnv, Sequence[str],
                                             int]]) -> None:
        """Train over ``(key, env, optimal_word, max_steps)`` problems."""
        self.rng = random.Random(self.seed)
        if self.use_demo:
            for key, env, word, _ in problems:
                self.observe_trajectory(key, env, word)
        for episode in range(self.episodes):
            epsilon = self._epsilon(episode)
            for key, env, _, max_steps in problems:
                self.q_episode(key, env, max_steps, epsilon)

    # ------------------------------------------------------------------
    # Evaluation
    # ------------------------------------------------------------------
    def act_episode(self, key: str, env: MinimalEnv, max_steps: int,
                    goal_name: str) -> EpisodeResult:
        state = env.reset()
        result = EpisodeResult(passed=False, goal=goal_name,
                               start=state.canonical(),
                               trace=[state.canonical()])
        for _ in range(max_steps):
            if env.goal_achieved(state):
                result.passed = True
                result.final_state = state.canonical()
                return result
            action = self.best_action(key, state)
            if action is None:
                result.reason = ("agent has no learned action for this state "
                                 "(unseen state/goal)")
                result.final_state = state.canonical()
                result.feedback = FailureFeedback(
                    state=state.canonical(),
                    expected=goal_name,
                    actual="no Q entry",
                    reason="tabular agent did not generalise to this state",
                )
                return result
            nxt = env.step(state, action)
            goal = env.goal_achieved(nxt)
            reward = self.goal_reward if goal else self.step_reward
            result.total_reward += reward
            result.steps.append(Step(action=action, state=nxt.canonical(),
                                     reward=reward, goal=goal))
            result.trace.append(nxt.canonical())
            state = nxt
        result.final_state = state.canonical()
        result.reason = "step limit reached before the goal"
        result.feedback = FailureFeedback(
            state=state.canonical(),
            expected=goal_name,
            actual=state.canonical(),
            reason="agent did not reach the goal within the step bound",
        )
        return result

    # ------------------------------------------------------------------
    # Reproducibility
    # ------------------------------------------------------------------
    def digest(self) -> str:
        """Stable digest of the learned table (for determinism checks)."""
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

    # ------------------------------------------------------------------
    # Persistence (small JSON table; used to split train and validation)
    # ------------------------------------------------------------------
    def to_json(self) -> Dict[str, object]:
        return {
            "seed": self.seed,
            "alpha": self.alpha,
            "gamma": self.gamma,
            "episodes": self.episodes,
            "use_demo": self.use_demo,
            "q": self.q,
        }

    def save(self, path: str) -> None:
        import json

        with open(path, "w", encoding="utf-8") as fh:
            json.dump(self.to_json(), fh, sort_keys=True)

    @classmethod
    def load(cls, path: str) -> "TabularAgent":
        import json

        with open(path, "r", encoding="utf-8") as fh:
            data = json.load(fh)
        agent = cls(
            alpha=data.get("alpha", 0.5),
            gamma=data.get("gamma", 0.95),
            episodes=data.get("episodes", 60),
            seed=data.get("seed", 20261009),
            use_demo=data.get("use_demo", True),
        )
        agent.q = {
            key: {state: {a: float(v) for a, v in row.items()}
                  for state, row in states.items()}
            for key, states in data["q"].items()
        }
        return agent


class CompositionalAgent:
    """Structurally biased agent: learns a local build rule per node shape.

    From the supervised trajectories it records which action constructed which
    ``(kind, arity)`` shape, then builds ANY target by a post-order recursion.
    It generalizes to unseen Group targets whose shapes were seen in training.
    """

    def __init__(self) -> None:
        self.rule: Dict[Tuple[str, int], str] = {}
        self.conflicts: int = 0

    def observe_trajectory(self, env: MinimalEnv,
                           word: Sequence[str]) -> None:
        state = env.reset()
        for action in word:
            nxt = env.step(state, action)
            built = nxt.stack[-1]
            shape = _node_shape(built)
            existing = self.rule.get(shape)
            if existing is None:
                self.rule[shape] = action
            elif existing != action:
                self.conflicts += 1
            state = nxt

    def plan(self, node: Node) -> List[str]:
        """Post-order build using the learned local rule per node shape."""
        if isinstance(node, Zero) or isinstance(node, One):
            action = self.rule.get(_node_shape(node))
            if action is None:
                raise KeyError("no learned rule for %r" % (_node_shape(node),))
            return [action]
        if isinstance(node, Change):
            return self.plan(node.child) + [self.rule[("Change", 0)]]
        if isinstance(node, Group):
            names: List[str] = []
            names += self.plan(node.tag)
            for item in node.items:
                names += self.plan(item)
            action = self.rule.get(("Group", node.arity()))
            if action is None:
                raise KeyError("no learned rule for Group arity %d"
                               % (node.arity(),))
            return names + [action]
        raise TypeError(type(node))

    def shapes(self) -> Dict[str, str]:
        return {("%s/%d" % k): v for k, v in sorted(self.rule.items())}

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------
    def save(self, path: str) -> None:
        import json

        data = {
            "conflicts": self.conflicts,
            "rule": [["%s|%d" % k, v] for k, v in sorted(self.rule.items())],
        }
        with open(path, "w", encoding="utf-8") as fh:
            json.dump(data, fh, sort_keys=True)

    @classmethod
    def load(cls, path: str) -> "CompositionalAgent":
        import json

        with open(path, "r", encoding="utf-8") as fh:
            data = json.load(fh)
        agent = cls()
        agent.conflicts = int(data.get("conflicts", 0))
        for shape, action in data["rule"]:
            kind, arity = shape.split("|")
            agent.rule[(kind, int(arity))] = action
        return agent

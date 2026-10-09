"""Shared rollout / evaluation harness for the three approach arms.

Unit C compares three ways to train an agent on the dynamic-nodes environment:

* ``evolution`` - the Unit B fixed-weight evolutionary process (two-parent
  weight mixing + mutation);
* ``rl``       - pure policy gradient (REINFORCE with a baseline);
* ``mix``      - RL warm-start of the founder genomes, then fixed-weight
  evolution.

For the comparison to be FAIR, all three arms must see the SAME network
architecture, the SAME state/action features and the SAME reward signal. This
module owns that shared surface:

* :class:`Harness` builds the target-conditioned :class:`PolicyNet` and the
  :class:`ActionFeaturizer` exactly as :class:`EvolutionTrainer` does, and
  produces the input vector ``state_features(env, state, mask) +
  action_features(state, action)`` for every legal action;
* :meth:`Harness.rollout` replays the reward shaping of
  ``EvolutionTrainer.run_episode`` byte-for-byte: a per-step cost, an
  intermediate reward credited ONCE per objective, a guard bonus credited ONCE
  per unlocked guarded objective, a subgoal reward and a goal reward (the
  environment's own reward hook is the goal/step part). Subagent recursion is
  an evolution-internal training device and is disabled here, so every arm is
  scored on identical episodes.

Standard library only, like the rest of the dynamic-nodes work.
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

from dynamic_env.engine import Action, DynamicEnv, State
from dynamic_env.spec import Spec

from .features import ActionFeaturizer, objective_ids, state_dim, state_features
from .genome import PolicyNet


def subgoal_reached(flags: Sequence[int], target: Sequence[int],
                    mask: Sequence[int]) -> bool:
    """True when every masked objective already holds its target value."""
    return all(flags[i] == target[i] for i in range(len(target)) if mask[i])


@dataclass
class RolloutResult:
    """The outcome of one episode plus (optionally) the decision trace."""

    total_return: float
    steps: int
    solved: bool
    final_state: State
    trace: List[Dict[str, Any]] = field(default_factory=list)


class Harness:
    """Shared rollout and evaluation over one dynamic-nodes spec."""

    def __init__(self, spec: Spec, hidden: int = 12,
                 step_cost: float = 0.05,
                 intermediate_reward: float = 0.30,
                 guard_bonus: float = 0.20,
                 subgoal_reward: float = 0.50,
                 goal_reward: float = 1.00,
                 max_episode_steps: Optional[int] = None) -> None:
        self.spec = spec
        self.featurizer = ActionFeaturizer(spec)
        self.in_dim = state_dim(spec) + self.featurizer.dim
        self.net = PolicyNet(self.in_dim, hidden)
        self.obj_ids = objective_ids(spec)
        self.target = tuple(int(spec.objectives[oid]) for oid in self.obj_ids)
        self.root_mask = tuple(1 for _ in self.obj_ids)
        self.step_cost = step_cost
        self.intermediate_reward = intermediate_reward
        self.guard_bonus = guard_bonus
        self.subgoal_reward = subgoal_reward
        self.goal_reward = goal_reward
        self.max_episode_steps = max_episode_steps

    # -- network -----------------------------------------------------------
    def inputs(self, env: DynamicEnv, state: State,
               actions: Sequence[Action],
               mask: Sequence[int]) -> List[Tuple[float, ...]]:
        base = state_features(env, state, mask)
        return [base + self.featurizer.featurize(state, action)
                for action in actions]

    def logits(self, genome: Sequence[float], env: DynamicEnv, state: State,
               actions: Sequence[Action], mask: Sequence[int]) -> List[float]:
        return [self.net.forward(genome, x)
                for x in self.inputs(env, state, actions, mask)]

    @staticmethod
    def choose(logits: Sequence[float], rng: random.Random,
               explore: bool = True) -> int:
        """Argmax when ``explore`` is False, else a softmax draw (as Unit B)."""
        if not explore:
            return max(range(len(logits)), key=lambda i: (logits[i], -i))
        peak = max(logits)
        weights = [math.exp(value - peak) for value in logits]
        total = sum(weights)
        draw = rng.random() * total
        acc = 0.0
        for i, weight in enumerate(weights):
            acc += weight
            if draw <= acc:
                return i
        return len(logits) - 1

    # -- one episode -------------------------------------------------------
    def rollout(self, genome: Sequence[float],
                mask: Optional[Sequence[int]] = None,
                explore: bool = True,
                rng: Optional[random.Random] = None,
                collect: bool = False,
                max_steps: Optional[int] = None,
                scripted_keys: Optional[Sequence[str]] = None) -> RolloutResult:
        rng = rng or random.Random(0)
        mask = tuple(mask) if mask is not None else self.root_mask
        remaining = list(scripted_keys) if scripted_keys is not None else None
        env = DynamicEnv(self.spec)
        state = env.reset()
        limit = max_steps or self.max_episode_steps or self.spec.max_steps
        n_obj = len(self.obj_ids)
        total = 0.0
        steps = 0
        solved = False
        trace: List[Dict[str, Any]] = []

        prev_flags = env.objective_vector(state)
        if subgoal_reached(prev_flags, self.target, mask):
            return RolloutResult(0.0, 0, env.goal_reached(state), state, trace)

        actions = env.legal_actions(state)
        set_legal = frozenset(a.objective_id for a in actions if a.kind == "set")
        credited: set = set()
        unlocked_credited: set = set()
        guarded_ids = frozenset(self.spec.guards)

        while steps < limit and actions:
            xs = self.inputs(env, state, actions, mask)
            logits = [self.net.forward(genome, x) for x in xs]
            chosen: Optional[int] = None
            if remaining is not None:
                for key in list(remaining):
                    for i, candidate in enumerate(actions):
                        if candidate.key() == key:
                            chosen = i
                            remaining.remove(key)
                            break
                    if chosen is not None:
                        break
            if chosen is None:
                chosen = self.choose(logits, rng, explore=explore)
            action = actions[chosen]
            result = env.step(state, action)
            flags = env.objective_vector(result.state)
            reward = -self.step_cost
            for i in range(n_obj):
                if (mask[i] and i not in credited
                        and prev_flags[i] != self.target[i]
                        and flags[i] == self.target[i]):
                    reward += self.intermediate_reward
                    credited.add(i)
            next_actions = env.legal_actions(result.state)
            next_set_legal = frozenset(
                a.objective_id for a in next_actions if a.kind == "set")
            new_unlocks = ((next_set_legal - set_legal) & guarded_ids) - unlocked_credited
            if new_unlocks:
                reward += self.guard_bonus * len(new_unlocks)
                unlocked_credited |= new_unlocks
            done_full = result.done
            done_sub = subgoal_reached(flags, self.target, mask)
            if done_full:
                reward += self.goal_reward
            elif done_sub:
                reward += self.subgoal_reward
            total += reward
            steps += 1
            if collect:
                trace.append({"step": steps, "action": action.key(),
                              "state_vector": list(env.state_vector(result.state)),
                              "objective_vector": list(flags),
                              "reward": reward, "goal": bool(done_full),
                              "inputs": xs, "chosen": chosen})
            prev_flags = flags
            state = result.state
            actions = next_actions
            set_legal = next_set_legal
            if done_full or done_sub:
                solved = done_full
                break

        solved = solved or env.goal_reached(state)
        return RolloutResult(total, steps, solved, state, trace)

    # -- evaluation (identical for every arm) ------------------------------
    def evaluate(self, genome: Sequence[float], episodes: int = 30,
                 seed: int = 12345) -> Dict[str, Any]:
        """Deterministic greedy episode + ``episodes`` stochastic episodes."""
        greedy = self.rollout(genome, explore=False)
        rng = random.Random(seed)
        solved = 0
        returns: List[float] = []
        solved_steps: List[int] = []
        for _ in range(episodes):
            res = self.rollout(genome, explore=True, rng=rng)
            returns.append(res.total_return)
            if res.solved:
                solved += 1
                solved_steps.append(res.steps)
        return {
            "greedy_return": round(greedy.total_return, 6),
            "greedy_steps": greedy.steps,
            "greedy_solved": bool(greedy.solved),
            "eval_episodes": episodes,
            "eval_solved": solved,
            "eval_solve_rate": round(solved / float(max(1, episodes)), 6),
            "eval_mean_return": round(sum(returns) / max(1, len(returns)), 6),
            "eval_mean_steps_to_solve": (
                round(sum(solved_steps) / len(solved_steps), 4)
                if solved_steps else None),
        }

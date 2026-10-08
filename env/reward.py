import math
from abc import ABC
from env.core import (
    INode,
    IDefault,
    InheritableNode,
    Integer,
    Protocol,
    CountableTypeGroup,
    TmpInnerArg,
    IInstantiable)
from env.full_state import FullState

class IRewardEvaluator(INode, ABC):

    def evaluate(self, current_state: FullState, next_state: FullState) -> float:
        raise NotImplementedError

class DefaultRewardEvaluator(InheritableNode, IRewardEvaluator, IDefault, IInstantiable):

    idx_goal_reward = 1
    idx_step_penalty = 2

    # Step penalty is stored as milli-units (Integer nodes cannot hold floats);
    # 1000 == the default 1.0.
    STEP_PENALTY_SCALE = 1000

    @classmethod
    def protocol(cls) -> Protocol:
        return cls.default_protocol(CountableTypeGroup(
            Integer.as_type(),
            Integer.as_type(),
        ))

    @classmethod
    def create(cls, goal_reward: int = 10000, step_penalty: float = 1.0):
        return cls(
            Integer(int(goal_reward)),
            Integer(int(round(float(step_penalty) * cls.STEP_PENALTY_SCALE))),
        )

    @property
    def goal_reward(self) -> TmpInnerArg:
        return self.inner_arg(self.idx_goal_reward)

    @property
    def step_penalty(self) -> float:
        # Backward compatible with an evaluator built with only the goal
        # reward: the absent argument means the default 1.0.
        if len(self.args) < self.idx_step_penalty:
            return 1.0
        penalty = self.inner_arg(self.idx_step_penalty).apply().real(Integer)
        return penalty.as_int / self.STEP_PENALTY_SCALE

    def evaluate(self, current_state: FullState, next_state: FullState) -> float:
        step_penalty = self.step_penalty
        if next_state.goal_achieved():
            goal_reward = self.goal_reward.apply().cast(Integer).as_int
            return goal_reward - step_penalty  # Reached the objective

        if next_state.is_last_step_error():
            return -100 - step_penalty

        cost = next_state.final_cost().as_int
        reward = -math.log(cost + 1) - step_penalty

        return reward

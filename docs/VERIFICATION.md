# automath code verification (unit U2a)

Repo: `nexuslbs/automath` @ `main` `8b0507b4cf3bccd1ce71e888c073ba0e3fef9012`.
Method: read the code at that commit; every claim below carries `file:line` and a
verbatim quote (each quote <= 15 lines). The README is empty (0 bytes) and
`summary.md` (63 lines) is the de-facto prose doc, so the specification used here
is the code itself. No code was changed except this file.

---

## 1. What automath is

Automath is a goal-oriented symbolic/typed mathematics environment written in
pure Python. The whole object model is one recursive node type, `INode`, whose
concrete instances (`env/core.py:11`) form a typed tree; a `State` is a node that
groups a meta section, a tuple of scratch spaces, argument groups and definitions
(`env/state.py:435`), and a `FullState` is the RL-facing wrapper that adds the
immutable `MetaInfo` (which carries the goal, the action catalogue and options),
the current `HistoryNode` and the past `HistoryGroupNode`
(`env/full_state.py:431`). The `summary.md:4` opening states the intent:
`Automath is a Python-based environment for mathematical/symbolic computation with strong typing and state management.`

A *goal* is itself a node placed in the state: the user-supplied `IGoal` is
stored at `MetaInfo.goal` (`env/meta_env.py:1344`, `1434`) and is reduced to an
`IGoalAchieved` boolean inside the current state
(`env/full_state.py:463-472`). The agent never edits the set of nodes in place;
it submits an *action* (`RawAction` with an action index and up to three integer
arguments, `env/action.py:570-661`) and the action layer validates it, converts
it to a concrete `IBasicAction`, runs it against the `FullState` and returns a
new immutable `FullState` with the action appended to its history
(`env/action.py:319-552`). Learning therefore proceeds by searching sequences of
node transformations, and a math question is "solved" when the current state's
`goal_achieved()` flag is true.

---

## 2. STATE: how `FullState` is built, and the goal's place in it

`FullState` has exactly three children, declared as `idx_meta = 1`,
`idx_current = 2`, `idx_history = 3` (`env/full_state.py:440-450`):

```python
    idx_meta = 1
    idx_current = 2
    idx_history = 3

    @classmethod
    def protocol(cls) -> Protocol:
        return cls.default_protocol(CountableTypeGroup(
            MetaInfo.as_type(),
            HistoryNode.as_type(),
            HistoryGroupNode.as_type(),
        ))
```

`GoalEnv` builds the initial state with `FullState.with_node(meta)`
(`env/goal_env.py:54-57`), which delegates to `with_args(meta=node)`
(`env/full_state.py:452-454`):

```python
    @classmethod
    def with_node(cls, node: MetaInfo) -> typing.Self:
        return cls.with_args(meta=node)
```

`with_args` is where the GOAL enters the mutable state: it reads `meta.goal` and
constructs the current `HistoryNode` whose `State` already carries the goal's
boolean (`env/full_state.py:463-472`):

```python
        goal = meta.goal.apply().real(IGoal)
        options = meta.options.apply().real(MetaInfoOptions)
        max_steps_opt = options.max_steps.apply().real(Optional[IInt])
        max_steps = max_steps_opt.value.as_int if max_steps_opt.value is not None else None
        current = current if current is not None else HistoryNode.create_with_goal_and_options(
            goal=goal,
            remaining_steps=max_steps,
        )
        history = history if history is not None else HistoryGroupNode()
        return cls.new(meta, current, history)
```

`HistoryNode.create_with_goal_and_options` turns `IGoal` into `IGoalAchieved`
and stores it in the state meta (`env/full_state.py:323-334`), and
`State.create_with_goal` writes it into `StateMetaInfo` (`env/state.py:460-467`):

```python
    @classmethod
    def create_with_goal(cls, goal_achieved: IGoalAchieved) -> typing.Self:
        return cls(
            StateMetaInfo.with_goal_achieved(goal_achieved),
            ScratchGroup(),
            PartialArgsOuterGroup(),
            StateDefinitionGroup(),
        )
```

So the goal is observable in two places: the raw `IGoal` at
`(FullState.idx_meta, MetaInfo.idx_goal)` (read that way by the tests, e.g.
`test_suite/boolean_test.py:184-188`), and the boolean `IGoalAchieved` in
`current.state.meta_info.goal_achieved`, which is what terminates an episode.

`State` itself is a four-child node (`env/state.py:435-449`): `StateMetaInfo`,
`ScratchGroup` (scratch spaces), `PartialArgsOuterGroup` (argument groups) and
`StateDefinitionGroup` (function definitions).

`final_cost()` reads the cost that the last action stored in current meta
(`env/full_state.py:542-548`):

```python
    def final_cost(self) -> Integer:
        current = self.current.apply().real(HistoryNode)
        IsInstance.assert_type(current, HistoryNode)
        assert isinstance(current, HistoryNode)
        meta_data = current.meta_data.apply().real(MetaData)
        final_cost_opt = meta_data.final_cost.apply().real(Optional[Integer])
        return final_cost_opt.value_or_raise
```

`goal_achieved()` is simply the current `State`'s flag
(`env/full_state.py:550-552`):

```python
    def goal_achieved(self) -> bool:
        state = self.current_state.apply().real(State)
        return state.goal_achieved()
```

and `State.goal_achieved()` ANDs the (possibly grouped) `IGoalAchieved`
(`env/state.py:485-489`):

```python
    def goal_achieved(self) -> bool:
        meta_info = self.meta_info.apply().cast(StateMetaInfo)
        goal_achieved = meta_info.goal_achieved.apply().cast(IGoalAchieved)
        result = goal_achieved.as_bool is True
        return result
```

`is_last_step_error()` inspects the LAST history item's action data
(`env/full_state.py:641-649`):

```python
    def is_last_step_error(self) -> bool:
        history = self.history.apply().real(HistoryGroupNode).as_tuple
        if not history:
            return False
        action_data_opt = history[-1].action_data.apply().real(Optional[BaseActionData])
        action_data = action_data_opt.value
        if action_data is None:
            return False
        return action_data.is_error().as_bool
```

`final_cost()` counts a composite of action count, instruction count and four
memory sizes, weighted by per-step `CostMultiplier` (`env/meta_env.py:921-963`):

```python
        action_cost = actions * cost_multiplier_action
        instruction_cost = instructions * cost_multiplier_instruction
        processing_cost_value = action_cost + instruction_cost

        full_state_memory_cost = full_state_memory * cost_full_state_memory
        visible_state_memory_cost = visible_state_memory * cost_visible_state_memory
        main_state_memory_cost = main_state_memory * cost_main_state_memory
        run_memory_cost = run_memory * cost_run_memory
        memory_cost_value = (
            full_state_memory_cost
            + visible_state_memory_cost
            + main_state_memory_cost
            + run_memory_cost)

        final_cost = current_multiplier * processing_cost_value * memory_cost_value

        return Integer(final_cost)
```

The memory inputs are measured in `BaseAction.run_action_details`
(`env/action.py:504-521`): `len(new_full_state)` (full state), the `NodeData`
array length (visible state), `len(next_state)` (main state) and the action's own
run memory.

---

## 3. ACTIONS: interface, concrete set, validation and invalid handling

The minimal interfaces live in `env/meta_env.py:1150-1158`:

```python
class IActionOutput(INode, typing.Generic[S], ABC):

    def run_output(self, full_state: S) -> tuple[State, IActionOutputInfo]:
        raise NotImplementedError

class IAction(INode, typing.Generic[S], ABC):

    def run_action(self, full_state: S) -> S:
        raise NotImplementedError
```

`IBasicAction` adds the 3-integer encoding used by everything the agent emits
(`env/meta_env.py:1162-1180`), and `IRawAction` is the proposed, unresolved
action (`env/meta_env.py:1241-1244`):

```python
class IRawAction(IAction[S], typing.Generic[S], ABC):

    def to_action(self, full_state: S) -> IBasicAction[S]:
        raise NotImplementedError
```

`RawAction` is the concrete proposal container (`action index` + `arg1..arg3`),
and `to_action` resolves the index through the state's allowed-basic-actions
table (`env/action.py:602-612`):

```python
    def to_action(self, full_state: FullState) -> IBasicAction[FullState]:
        action_index = self.action_index.apply().real(MetaAllowedBasicActionsTypeIndex)
        arg1 = self.arg1.apply().real(Integer)
        arg2 = self.arg2.apply().real(Integer)
        arg3 = self.arg3.apply().real(Integer)

        action_type = action_index.find_in_outer_node(full_state).value_or_raise

        basic_action = action_type.type.from_raw(arg1.as_int, arg2.as_int, arg3.as_int)

        return basic_action
```

The curated concrete action set is `ESSENTIAL_ACTIONS`
(`env/node_types.py:31-54`):

```python
ESSENTIAL_ACTIONS = (
    action_impl.RestoreHistoryStateOutput,
    action_impl.VerifyGoal,
    action_impl.CreateDynamicGoal,
    action_impl.VerifyDynamicGoal,
    action_impl.DeleteDynamicGoalOutput,
    action_impl.ResetStateHiddenInfo,
    action_impl.DefineStateHiddenInfo,
    action_impl.CreateScratch,
    action_impl.DeleteScratchOutput,
    action_impl.ClearScratch,
    action_impl.DefineScratchFromDefault,
    action_impl.DefineScratchFromInt,
    action_impl.DefineScratchFromSingleArg,
    action_impl.DefineScratchFromIntIndex,
    action_impl.DefineScratchFromFunctionWithIntArg,
    action_impl.DefineScratchFromFunctionWithSingleArg,
    action_impl.DefineScratchFromFunctionWithArgs,
    action_impl.DefineScratchFromScratchNode,
    action_impl.UpdateScratchFromAnother,
    action_impl.CreateArgsGroup,
    action_impl.DeleteArgsGroupOutput,
    action_impl.DefineArgsGroup,
)
```

Beyond that tuple, `env/action_impl.py` also defines the generic meta actions
`DynamicAction` (`:114`), `GroupAction` (`:186`) and `RunScratch` (`:1384`). When
`allowed_actions` is not passed, `MetaInfo.with_defaults` builds the catalogue
from every `IAction`/`IInstantiable` subclass (`env/meta_env.py:1520-1529`).

Validation and application are staged in `BaseAction.inner_run`
(`env/action.py:319-395`). Each stage catches `InvalidNodeException` and wraps it
in a typed exception info object:

```python
        if isinstance(action, RawAction):
            raw_action = action

            try:
                raw_action.strict_validate()
                action_aux = raw_action.to_action(full_state)
                assert isinstance(action_aux, BaseAction)
                action = action_aux
            except InvalidNodeException as e:
                raise RawActionExceptionInfo(raw_action, e.info).as_exception() from e
```

```python
        try:
            action.strict_validate()
            meta = full_state.meta.apply().real(MetaInfo)
            allowed_actions = meta.allowed_actions.apply().real(GeneralTypeGroup[IAction])
            min_index = 1
            max_index = len(allowed_actions.as_tuple)
            HasArg(action.as_type(), allowed_actions).raise_on_false()
            action_type = allowed_actions.as_tuple.index(action.as_type()) + 1
            IsInsideRange.from_raw(
                value=action_type,
                min_value=min_index,
                max_value=max_index,
            ).raise_on_false()
        except InvalidNodeException as e:
            raise ActionTypeExceptionInfo(
                Optional.with_value(raw_action),
                action,
                e.info,
            ).as_exception() from e
```

The five error classes are `RawActionExceptionInfo`, `ActionTypeExceptionInfo`,
`ActionInputExceptionInfo`, `ActionOutputExceptionInfo` and
`BeforeActionExceptionInfo` (`env/action.py:140-305`).

An invalid action does NOT crash the environment. `run_action_details` catches
`InvalidActionException`, logs a debug symbol, keeps the CURRENT state unchanged
and records the error as action data (`env/action.py:442-449`):

```python
        except InvalidActionException as e:
            symbol = Symbol(
                node=e.info.as_node,
                node_types=full_state.node_types(),
            )
            env_logger.debug(str(symbol), exc_info=e)
            next_state = current.state.apply().real(State)
            action_data = e.to_action_data()
```

That action data is a `BaseActionData` subtype whose `exception` slot is
non-empty; `BaseActionData.is_error()` is exactly "the exception slot is not
empty" (`env/full_state.py:156-158`):

```python
    def is_error(self) -> IBoolean:
        exception_opt = self.exception.apply().real(Optional[IExceptionInfo])
        return Not(exception_opt.is_empty())
```

The typed error rows are `BeforeActionErrorActionData`,
`RawActionErrorActionData`, `ActionTypeErrorActionData`,
`ActionErrorActionData` and `ActionOutputErrorActionData`
(`env/full_state.py:183-296`). So a wrong action
is expressed IN the state (an error action-history entry with an
`IExceptionInfo`), and is visible via `is_last_step_error()`; only exceptions
that are not `InvalidNodeException` (e.g. a raw `AssertionError` from an
`assert`) still propagate as Python exceptions.

---

## 4. ENVIRONMENT + RL LOOP

`Environment` is the small RL shell (`env/environment.py:34-58`):

```python
    def reset(self) -> full_state_module.FullState:
        self._full_state = self._initial_state
        self._current_step = 0
        core.INode.clear_cache()
        return self._full_state

    def step(
        self,
        action: action_module.IAction[full_state_module.FullState],
    ) -> tuple[full_state_module.FullState, float, bool, bool]:
        reward_evaluator = self._reward_evaluator
        current_state = self._full_state
        next_state = action.run_action(current_state)
        reward = reward_evaluator.evaluate(
            current_state,
            next_state)
        self._current_step += 1
        terminated = next_state.goal_achieved()
        truncated = (
            (self._current_step >= self._max_steps and not terminated)
            if self._max_steps is not None
            else False
        )
        self._full_state = next_state
        return next_state, reward, terminated, truncated
```

The agent picks the action through the `BaseAgent` seam
(`env/base_agent.py:11-20`):

```python
    def select_action(self, state: FullState) -> RawAction:
        """Select an action based on the current state.

        Args:
            state: Current environment state

        Returns:
            Selected action to perform
        """
        raise NotImplementedError()
```

and `Trainer.train` is the loop that calls it and then `env.step`
(`agent/trainer.py:50-61`):

```python
            # Select and execute action
            raw_action = (
                static_actions[steps]
                if static_actions is not None
                else self.agent.select_action(state)
            )

            end_select = time.time()
            start_step = end_select

            next_state, reward, terminated, truncated = self.env.step(raw_action)
            done = terminated or truncated
```

Episodes end on goal (`terminated`) or max steps (`truncated`); an error itself
does not terminate the episode, it only yields the `-100` error reward on that
step. Reward is computed in `DefaultRewardEvaluator.evaluate`
(`env/reward.py:35-46`):

```python
    def evaluate(self, current_state: FullState, next_state: FullState) -> float:
        if next_state.goal_achieved():
            goal_reward = self.goal_reward.apply().cast(Integer).as_int
            return goal_reward  # Reached the objective

        if next_state.is_last_step_error():
            return -100

        cost = next_state.final_cost().as_int
        reward = -math.log(cost + 1)

        return reward
```

NOTE on `env/composite.py`: it is NOT an RL component. It defines one module-level
`Map` `core.FunctionExpr` (`env/composite.py:3-31`) used by the math/control-flow
node library. The RL loop lives entirely in `environment.py`, `reward.py`,
`goal_env.py`, `base_agent.py`, `agent/trainer.py`.

`env/core.py` supplies the node runtime that actions ride on: `BaseNode.run`
caches run results (`env/core.py:497-529`), and `INode.clear_cache` clears the
instance and run caches (`env/core.py:11-15`, `456-495`).

---

## 5. THE AGENT(S)

* `env/base_agent.py` (60 lines) is the abstract contract:
  `select_action` (`:11`), `train` (`:22`), `reset` (`:43`), `save` (`:46`),
  `load` (`:54`).
* `agent/demo_agent.py` (62 lines) replays a fixed `list[RawAction]`;
  `select_action` returns the next one or raises `IndexError`
  (`agent/demo_agent.py:37-43`); `train` is a no-op (`:45-56`).
* `agent/simple_agent.py` (896 lines) is the DEFAULT learner
  (`config/agent_settings.py:23` sets `AGENT_TYPE: str = "simple"`). It is a
  NumPy-only decomposed-Q-learning network (`SimpleNetwork`, `:36`), an
  experience buffer of capacity 500 (`:553`), epsilon-greedy
  `select_action` (`:575`), and `train` (`:674`) that stores a transition and
  trains every 3 steps (`:715-717`). `_train_q_network` (`:746`) samples a
  min(16) batch, computes `target_q_values = reward + gamma * max_next_q *
  (1 - terminated)` (`:818-822`) and updates weights via
  `network.q_decomposition_update` (`:825-829`). `save`/`load` use
  `np.savez`/`np.load` with the `W1,b1,...,W_arg3,b_arg3,steps_done,epsilon`
  keys (`agent/simple_agent.py:849-896`).
* `agent/smart_agent.py` (783 lines) is the torch DQN policy. The network is
  `DQN(NodeFeatureExtractor, ActionPredictor)` (`:261-326`); the agent builds a
  policy and target net on the chosen device, copies weights and uses
  `torch.optim.Adam` (`agent/smart_agent.py:452-474`):

```python
        self.policy_net = DQN(
            input_dim=input_dim,
            feature_dim=feature_dim,
            hidden_dim=hidden_dim,
            hidden_amount=hidden_amount,
            action_space_size=action_space_size,
            dropout_rate=dropout_rate,
        ).to(self.device)

        self.target_net = DQN(
            ...
        ).to(self.device)

        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()  # Target network is only used for evaluation

        # Create optimizer
        self.optimizer = torch.optim.Adam(self.policy_net.parameters(), lr=learning_rate)
```

  It uses an `ExperienceReplayBuffer` of capacity `replay_buffer_capacity`
  (`:329-383`, `:477`), only learns once the buffer holds `batch_size` samples
  (`:622`), pads variable node counts (`:648-671`), computes a primary
  `F.smooth_l1_loss` on joint Q-values plus `F.cross_entropy` on action logits
  and three `smooth_l1` arg losses, weighted `2.0/0.5` (`:693-710`), clips
  gradients to `[-1, 1]` and steps Adam (`:716-722`), and syncs the target net
  every `target_update_frequency` steps (`:727-729`).
* `agent/trainer.py` (182 lines) drives one episode and adds "inception"
  sub-training: for each transition it builds sub-goals
  (`CorrectActionValidator`, `ActionFromRawAction`, `ActionOutputFromAction`,
  `ActionOutputFromRawAction`, `NewStateFromActionOutput`, ...) and recursively
  creates a `GoalEnv`/`Trainer` (`agent/trainer.py:110-182`).
* `agent/train.py` (191 lines) is the high-level training routine: it walks the
  test suite's recorded `FullState` histories (`:44-64`), creates a `GoalEnv`
  seeded from each history state (`:142-147`) and trains. It saves the model
  every `save_interval` states and at the end (`:163-165`, `:187-189`).
* Root `train.py` (113 lines) reads `config/agent_settings.py`, constructs a
  `SimpleAgent` (`:61-72`), and only raises `NotImplementedError` for the
  `"smart"` branch (`:73-74`), with the `SmartAgent` import commented out at
  `:16`.

A trained SmartAgent checkpoint is a torch dict with `policy_net_state_dict`,
`target_net_state_dict`, `optimizer_state_dict`, `steps_done`, `epsilon`
(`agent/smart_agent.py:756-769`). A SimpleAgent checkpoint is an `.npz` written
by `np.savez` with the 14 weight/bias arrays plus `steps_done`/`epsilon`
(`agent/simple_agent.py:849-872`).

---

## 6. MATH PROBLEMS + SCORING

Two concrete problems and their goal nodes:

1. Arithmetic equality. `run_single_eq` defines the goal as a
   `HaveResultScratch` over `Eq(left, right)` (`test_suite/arithmetic_test.py:33-40`):

```python
def run_single_eq(left_expr: core.INode, right_expr: core.INode, result: bool):
    goal = node_types.HaveResultScratch.with_goal(core.Eq(left_expr, right_expr))
    env = GoalEnv(
        goal=goal,
        max_steps=3,
        allowed_actions=node_types.ESSENTIAL_ACTIONS,
    )
```

   and `test_binary_int_basic` gives the concrete operands
   `raw_expr=core.INumber.zero()`, `correct_expr=core.INumber.zero()`,
   `wrong_exprs=[core.INumber.one(), core.BinaryInt(...), core.Integer(0), ...]`
   (`test_suite/arithmetic_test.py:305-317`). The agent's job is to write the
   boolean into scratch 1 and call `VerifyGoal`, which makes the goal evaluate
   true or false depending on `Eq`.

2. Boolean comparison. `boolean_test.run` uses the same
   `HaveResultScratch` goal over an arbitrary runnable, e.g.
   `run(goal_expr=core.LessThan(core.Integer(1), core.Integer(2)), result=True)`
   (`test_suite/boolean_test.py:31-37`, `:336-339`).

The goal type `HaveResultScratch.evaluate` runs the goal expression and tests it
against the scratch content (`env/node_types.py:74-96`):

```python
class HaveResultScratch(Goal[IRunnable, StateScratchIndex], IInstantiable):

    @classmethod
    def goal_type(cls):
        return IRunnable

    @classmethod
    def eval_param_type(cls):
        return StateScratchIndex

    def evaluate(self, state: State, eval_param: StateScratchIndex):
        runnable = self.goal_inner_expr.apply().real(IRunnable)
        run_info = RunInfo.with_args(
            scope_data_group=ScopeDataGroup(),
            return_after_scope=Optional(),
        )
        eval_result = runnable.run(run_info.with_stats())
        _, goal_inner_expr = eval_result.as_tuple
        ...
        scratch = eval_param.find_in_node(state).value_or_raise
        content = scratch.value_or_raise
        return Eq(content, goal_inner_expr)
```

A question is decided SOLVED by the state flag, not by a numeric score. Inside
each test the final state is compared against an expected state that carries
`GoalAchieved.achieved()`; `run_single` (`test_suite/arithmetic_test.py:186-259`)
builds that expected state:

```python
    state_meta = state.StateMetaInfo.with_goal_achieved(state.GoalAchieved.create())
    env = GoalEnv(
        goal=goal,
        fn_initial_state=lambda meta: full_state.FullState.with_args(
            ...
        ),
        max_steps=1,
        allowed_actions=node_types.ESSENTIAL_ACTIONS,
    )
    ...
    state_meta = state_meta.with_new_args(
        goal_achieved=state.GoalAchieved.achieved(),
    )
    expected_state = state.State.from_raw(
        meta_info=state_meta,
        scratches=[correct_expr],
    )
    ...
    assert current_state == expected_state
```

At suite level the strongest assertion is `test_root._final_verification`, which
replays every recorded action and asserts the resulting state is identical
(`test_suite/test_root.py:50-60`):

```python
            if i < history_amount - 1:
                actual_next_fs, _ = full_state_case.at_history(index+1)
            else:
                actual_next_fs = full_state_case

            expected_next_state = next_fs.current_state.apply()
            actual_next_state = actual_next_fs.current_state.apply()
            ...
            assert expected_next_state == actual_next_state, f'{i_case}-{i}'
```

There is no aggregate numeric score in the suite; success is "all asserts hold
and `_final_verification` returns". The only scalar is the RL reward
(`+10000` on goal, `-100` on last-step error, `-log(final_cost()+1)` otherwise,
`env/reward.py:35-46`).

Commands (read from the code):
* `tests.py` defines `test()` (`tests.py:4`) but has NO `if __name__ ==
  "__main__"`, so `python tests.py` is NOT the entry point and prints nothing.
  The real entry point is `python -c 'import tests; tests.test()'`
  (`tests.py:4-12`).
* Fast subset: `python -c 'import tests_fast; tests_fast.test()'`
  (`tests_fast.py:5-14`; it sets `BaseNode.fast = True` before
  `test_root.test(fast=True)`).
* No-cache variant: `python -c 'import tests_no_cache; tests_no_cache.test()'`
  (`tests_no_cache.py:4-6`; it sets `BaseNode.cache_enabled = False`).
* Run from the repository root (`/opt/workspace/tmp/automath/repo`), where
  `test_suite` and `env` are importable packages.

---

## 7. STATE EXPLOSION / HISTORY BOUND

Every use of `max_history_state_size` and `max_steps`:

* Option definition and default: `MetaInfoOptions.idx_max_history_state_size = 1`,
  `idx_max_steps = 2` (`env/meta_env.py:182-183`); both are built with
  `Optional.with_int(...)` (`:294-295`), so the default is `None` (unbounded).
* `GoalEnv.__init__` forwards them into `_get_meta` -> `MetaInfo.with_defaults`
  (`env/goal_env.py:24-30`, `:44-52`; `env/meta_env.py:1516-1517`, `:1533-1536`).
* The ONLY truncation of history is in `BaseAction.run_action_details`
  (`env/action.py:488-489`):

```python
        if max_history_state_size is not None:
            history = history[-max_history_state_size.as_int:]
```

* `max_steps` is enforced per action by decrementing `remaining_steps` and
  failing the step when it hits 0 (`env/action.py:411-425`, `:459-462`); the
  episode-level cap is `Environment.step`'s `truncated`
  (`env/environment.py:52-56`), and `FullState.with_max_steps` rewrites
  `remaining_steps` (`env/full_state.py:605-639`).
* `config/agent_settings.py:17` declares `MAX_STATE_SIZE: int = 500000  # Max
  state cost before truncation`, but it is never read anywhere (grep finds only
  the definition); no cost-based truncation is wired.

The bound that holds today: `max_history_state_size` is `None` for every
`GoalEnv` constructed in this repo (tests pass `max_steps` and
`allowed_actions`, never `max_history_state_size`; `agent/trainer.py:171`
forwards `self.env.max_history_state_size`, which is `None`). Therefore history
grows WITHOUT truncation and the `history[-n:]` branch never fires. When a
caller does set it, older states are silently dropped from `history`, and
`at_history` can no longer reach them (`env/full_state.py:578-598`), so an
unbounded/truncated history CAN hide earlier states.

Independently, the *observation* can hide state: `StateMetaHiddenInfo`
(`env/state.py:192-276`) is applied by `DefineStateHiddenInfo`
(`env/action_impl.py:650-700`), and `NodeData.to_data_array_with_specs` drops
rows whose last column is 1 (`env/node_data.py:206-207`).

Caches (relevant to `tests_no_cache.py`):
* `BaseNode._instances` (hash-consing of nodes) and `BaseNode._cached_run`
  (memoised run results), cleared by `BaseNode.clear_actual_cache`
  (`env/core.py:460-461`, `:468-477`, `:493-495`, `:508-529`). The class flags
  are `cache_enabled = True` and `fast = False` (`env/core.py:458-459`).
* `NodeData._cache` and the per-instance `_my_cache`
  (`env/node_data.py:143-146`, `:173-177`, `:418-453`).
* `functools.cache` on `load_all_subclasses_sorted` (`env/env_utils.py:15-16`),
  on `goal_env._get_meta` (`env/goal_env.py:12`) and on
  `MetaInfo.with_defaults` (`env/meta_env.py:1509-1511`).
`tests_no_cache.py` disables the first of these by setting
`BaseNode.cache_enabled = False` before calling `tests.test()`
(`tests_no_cache.py:5-6`).

---

## 8. ERROR EXPRESSION (operator constraint 4307)

What the agent observes when an action fails is an ordinary successful step
whose returned `FullState` is unchanged except that the history gained a
`BaseActionData` row carrying the failure. The catch is
`env/action.py:442-449`:

```python
        except InvalidActionException as e:
            symbol = Symbol(
                node=e.info.as_node,
                node_types=full_state.node_types(),
            )
            env_logger.debug(str(symbol), exc_info=e)
            next_state = current.state.apply().real(State)
            action_data = e.to_action_data()
```

The failure CAUSE is encoded in the state, not left as an opaque Python
exception: `e.to_action_data()` returns the matching typed row
(`BaseActionData.to_action_data`, e.g. `RawActionExceptionInfo.to_action_data`,
`env/action.py:182-188`) whose `exception` slot holds the
`IExceptionInfo`; `BaseActionData.is_error()` reports non-empty
(`env/full_state.py:156-158`), and `FullState.is_last_step_error()` reads the
last row (`env/full_state.py:641-649`). `DefaultRewardEvaluator` turns that
into `-100` (`env/reward.py:40-41`). The exception is only logged at DEBUG through
`env_logger.debug` (`env/action.py:447`; the logger is configured in
`utils/env_logger.py:4-24`), and the environment does not raise. Caveat: only
`InvalidNodeException` is converted; a non-`InvalidNodeException` (for example a
bare `assert` failure inside an action) still propagates out of `step`.

---

## 9. TIMING HOOKS (operator constraint 4309)

Per-action wall time is already measured, at `time.time()` granularity, by the
`Trainer` loop (`agent/trainer.py:47-92`), splitting select / step / sub-train /
train:

```python
            start = time.time()
            start_select = start
            ...
            end_select = time.time()
            start_step = end_select

            next_state, reward, terminated, truncated = self.env.step(raw_action)
            done = terminated or truncated

            end_step = time.time()
            start_sub_train = end_step

            self.sub_train(raw_action)
```

The cleanest insertion point for a dedicated per-action timing probe is around
`self.env.step(raw_action)` (`agent/trainer.py:60`), or around
`action.run_action(current_state)` inside `Environment.step`
(`env/environment.py:46`). The agents also time their own work:
`SmartAgent.select_action` (`agent/smart_agent.py:517-533`) and `train`
(`:592-748`), `SimpleAgent.select_action` (`:589`, `:646`) and `train` (`:694`,
`:723-730`), and each test module via `test_utils.run_test`
(`test_suite/test_utils.py:12-19`).

Distorting factors for timing:
* The node instance/run caches (`env/core.py:460-461`, `:508-529`) make repeated
  identical nodes and runs cheaper; `tests_fast.py` flips `BaseNode.fast = True`
  (`tests_fast.py:7`), and `tests_no_cache.py` turns `BaseNode.cache_enabled` off
  (`tests_no_cache.py:5`).
* `NodeData._cache` (`env/node_data.py:143-146`) memoises state flattening, which
  is the most expensive per-step preprocessing.
* `Environment.reset` clears the core cache (`env/environment.py:37`) but does
  not clear `NodeData._cache`.
* All measurements use `time.time()`, not `time.perf_counter`, so they are
  wall-clock and subject to clock adjustments.

---

## 10. SMALL-LLM INTEGRATION SEAMS

The repo contains zero LLM/network code: the only third-party imports are
`numpy`, `sympy` and `torch` (see section 11). An OpenAI-compatible proposer can
be plugged in at these precise seams without touching the env:

1. **Agent action-selection interface**: `BaseAgent.select_action(state) ->
   RawAction` (`env/base_agent.py:11-20`), already the only hook `Trainer.train`
   calls (`agent/trainer.py:51-55`). A new `BaseAgent` implementation whose
   `select_action` asks an LLM for candidates is the primary seam; `Trainer`
   takes any `BaseAgent` (`agent/trainer.py:12-22`) and `train.py` already
   switches on `AGENT_TYPE` (`train.py:61-74`).
2. **The RawAction encoding**: `RawAction.with_raw_args(action_index, arg1,
   arg2, arg3)` (`env/action.py:633-646`) and `RawAction.from_basic_action`
   (`env/action.py:614-631`) are the exact conversion between "a concrete
   action node" and "the 4 integers the policy emits", so an LLM can propose
   either representation and be mapped to the other.
3. **The action catalogue the LLM must choose from**: `MetaInfo` exposes the
   allowed basic actions and their indices via
   `MetaAllowedBasicActionsTypeIndex.get_basic_action_index`
   (`env/full_state.py:754-779`), and `action_space_size()` is already computed
   for the policy (`env/full_state.py:600-603`, `env/environment.py:31-32`).
   `ESSENTIAL_ACTIONS` (`env/node_types.py:31-54`) is a ready-made short list to
   put in a prompt.

A proposer can therefore be added as a new `env/base_agent.py` subclass (or a
wrapper around an existing one) returning `RawAction`; no environment, reward
or state change is required.

---

## 11. DEPENDENCY TRUTH

Third-party imports found by grep over the `.py` files (plus stdlib
`typing/time/abc/functools/os/collections/math/logging/importlib/pkgutil`):

| package | requirements.txt | files that import it |
| --- | --- | --- |
| numpy | yes, `requirements.txt:1` | `agent/smart_agent.py:4`, `agent/simple_agent.py:3`, `env/node_data.py:3`, `test_suite/action_impl/action_01_state_meta.py:3` |
| sympy | yes, `requirements.txt:2` | `env/symbol.py:2-3` |
| torch | **NO** | `agent/smart_agent.py:6-8` (`import torch`, `from torch import nn`, `import torch.nn.functional as F`) |

`requirements.txt` is:

```text
numpy
sympy
scipy
pandas
matplotlib
pytest
```

UNDECLARED packages by name:
* `torch` - needed only by `agent/smart_agent.py:6-8`. This is why the default
  `AGENT_TYPE` is `"simple"` (`config/agent_settings.py:23`) and why root
  `train.py` comments out the SmartAgent import (`train.py:16`) and raises
  `NotImplementedError` for `agent_type == "smart"` (`train.py:73-74`); the torch
  code path cannot run from a clean `pip install -r requirements.txt`.

DECLARED but never imported by any `.py` file in the repo: `scipy`, `pandas`,
`matplotlib`, `pytest` (`requirements.txt:3-6`).

---

## 12. RUN INSTRUCTIONS

Working directory for all commands: the repository root
(`/opt/workspace/tmp/automath/repo`). Python 3 is required; `numpy` and `sympy`
are required for the suite (`env/node_data.py:3`, `env/symbol.py:2`); `torch` is
required only for `SmartAgent`.

Environment tests (real entry point - the `test()` functions are not guarded by
`__main__`):

```sh
# all tests (tests.py:4, test_root.test)
python -c 'import tests; tests.test()'

# fast subset (tests_fast.py:5, sets BaseNode.fast = True)
python -c 'import tests_fast; tests_fast.test()'

# cache-disabled variant (tests_no_cache.py:4, sets BaseNode.cache_enabled = False)
python -c 'import tests_no_cache; tests_no_cache.test()'
```

`python tests.py` (and likewise `python tests_fast.py` / `python
tests_no_cache.py`) exits silently without running anything, because none of
those files has an `if __name__ == "__main__":` block (`tests.py:1-12`,
`tests_fast.py:1-14`, `tests_no_cache.py:1-6`).

Training entry point:

```sh
# default: SimpleAgent (numpy), model saved to the path in config/agent_settings.py
python train.py

# to train the torch SmartAgent you must first install torch and un-comment
# SmartAgent in train.py:16 / :73-92 (the branch currently raises NotImplementedError)
```

Settings are read from `config/agent_settings.py` at import time (`train.py:8`,
`:38-59`), and the model directory is created from `MODEL_PATH`
(`train.py:35`; `MODEL_PATH` default `tmp/trained_model_simple_...pt` for the
simple agent, `config/agent_settings.py:25-29`).

---

## Summary of the 5 most important findings

1. The GOAL lives inside the state twice over: `MetaInfo.goal` (the raw `IGoal`)
   and `State.meta_info.goal_achieved` (the boolean built in
   `FullState.with_args`), so `GoalEnv` is correct that the initial state
   already carries the goal (`env/full_state.py:463-472`,
   `env/goal_env.py:54-57`).
2. `max_history_state_size` is dead in practice: it defaults to `None`, no
   caller in the repo sets it, so the only history truncation branch
   (`env/action.py:488-489`) never fires and history is unbounded; the only
   enforced bound is `max_steps`.
3. Invalid actions are not exceptions the agent sees: they are caught in
   `run_action_details`, the state is left unchanged, and a typed error row is
   appended to history, observable via `is_last_step_error()` and rewarded
   `-100` (`env/action.py:442-449`, `env/full_state.py:641-649`,
   `env/reward.py:40-41`).
4. `torch` is an undeclared dependency used only by `agent/smart_agent.py`;
   the default training path is the NumPy `SimpleAgent`, and root `train.py`
   disables the SmartAgent branch (`config/agent_settings.py:23`,
   `train.py:16`, `:73-74`, `agent/smart_agent.py:6-8`).
5. `python tests.py` does nothing; the real suite entry point is
   `python -c 'import tests; tests.test()'`, and correctness is asserted by
   exact state equality plus `test_root._final_verification`, not by a numeric
   score (`tests.py:4-12`, `test_suite/test_root.py:50-60`).

The implemented design that closes gaps 1-4 is documented in
[`docs/DESIGN.md`](DESIGN.md) (unit U3a).


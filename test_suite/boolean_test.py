from env import core, state, full_state, action, action_impl, meta_env, node_types
from env.goal_env import GoalEnv

def get_current_state(env: GoalEnv):
    return env.full_state.nested_arg(
        (full_state.FullState.idx_current, full_state.HistoryNode.idx_state)
    ).apply().cast(state.State)

def get_last_history_action(env: GoalEnv):
    history = env.full_state.history.apply().cast(full_state.HistoryGroupNode)
    last = history.as_tuple[-1]
    return last.action_data.apply().inner_arg(
        core.Optional.idx_value
    ).apply().cast(full_state.BaseActionData)

def get_from_int_type_index(node_type: type[core.INode], env: GoalEnv):
    selected_types = env.full_state.meta.apply().nested_arg((
        meta_env.MetaInfo.idx_from_int_group,
        meta_env.SubtypeOuterGroup.idx_subtypes,
    )).apply().cast(meta_env.GeneralTypeGroup)
    meta_idx = selected_types.as_tuple.index(node_type.as_type()) + 1
    return meta_idx

def get_basic_action_index(node_type: type[action.IBasicAction], env: GoalEnv):
    selected_types = env.full_state.meta.apply().real(
        meta_env.MetaInfo
    ).allowed_basic_actions.apply().cast(meta_env.GeneralTypeGroup)
    meta_idx = selected_types.as_tuple.index(node_type.as_type()) + 1
    return meta_idx

def build_env(goal_expr: core.IRunnable, max_steps: int = 3):
    """HaveResultScratch GoalEnv over ``goal_expr``.

    Shared by the boolean tests and ``scripts/run_case.py`` so the runner and
    the test build the case the same way.
    """
    goal = node_types.HaveResultScratch.with_goal(goal_expr)
    return GoalEnv(
        goal=goal,
        max_steps=max_steps,
        allowed_actions=node_types.ESSENTIAL_ACTIONS,
    )

def less_than_case() -> core.IRunnable:
    """The LessThan(Integer(1), Integer(2)) goal shared with scripts/run_case.py."""
    return core.LessThan(
        core.Integer(1),
        core.Integer(2),
    )

def run(goal_expr: core.IRunnable, result: bool):
    goal = node_types.HaveResultScratch.with_goal(goal_expr)
    env = build_env(goal_expr, max_steps=3)
    assert has_goal(env=env, goal=goal)

    current_state = get_current_state(env)

    state_meta = state.StateMetaInfo.with_goal_achieved(state.GoalAchieved.create())
    assert current_state == state.State.from_raw(
        meta_info=state_meta,
    )
    assert env.full_state.goal_achieved() is False

    # Run Action
    action_idx = get_basic_action_index(action_impl.CreateScratch, env)
    raw_action: action.RawAction = action.RawAction(
        full_state.MetaAllowedBasicActionsTypeIndex(action_idx),
        core.Integer(0),
        core.Integer(0),
        core.Integer(0),
    )
    full_action: action.BaseAction = action_impl.CreateScratch(core.Optional())
    output: meta_env.IActionOutput[full_state.FullState] = action_impl.CreateScratchOutput(
        state.StateScratchIndex(1),
        state.Scratch(),
    )
    env.step(raw_action)
    current_state = get_current_state(env)
    last_history_action = get_last_history_action(env)

    # Verify
    expected_history = full_state.SuccessActionData.from_args(
        raw_action=core.Optional(raw_action),
        action=core.Optional(full_action),
        output=core.Optional(output),
        exception=core.Optional(),
    )
    if last_history_action != expected_history:
        print('last_history_action:', env.symbol(last_history_action))
        print('expected_history:', env.symbol(expected_history))
    assert last_history_action == expected_history

    expected_state = state.State.from_raw(
        meta_info=state_meta,
        scratches=[None],
    )
    if current_state != expected_state:
        print('current_state:', env.symbol(current_state))
        print('expected_state:', env.symbol(expected_state))
    assert current_state == expected_state

    assert env.full_state.goal_achieved() is False

    # Run Action
    action_idx = get_basic_action_index(action_impl.DefineScratchFromInt, env)
    scratch_idx = 1
    meta_idx = get_from_int_type_index(state.IntBoolean, env)
    value = 1 if result else 0
    raw_action = action.RawAction(
        full_state.MetaAllowedBasicActionsTypeIndex(action_idx),
        core.Integer(scratch_idx),
        core.Integer(meta_idx),
        core.Integer(value),
    )
    full_action = action_impl.DefineScratchFromInt(
        state.StateScratchIndex(scratch_idx),
        full_state.MetaFromIntTypeIndex(meta_idx),
        core.Integer(value),
    )
    output = action_impl.DefineScratchOutput(
        state.StateScratchIndex(scratch_idx),
        state.Scratch(state.IntBoolean(value)),
    )
    env.step(raw_action)
    current_state = get_current_state(env)
    last_history_action = get_last_history_action(env)

    # Verify
    expected_history = full_state.SuccessActionData.from_args(
        raw_action=core.Optional(raw_action),
        action=core.Optional(full_action),
        output=core.Optional(output),
        exception=core.Optional(),
    )
    if last_history_action != expected_history:
        print('last_history_action:', env.symbol(last_history_action))
        print('expected_history:', env.symbol(expected_history))
    assert last_history_action == expected_history

    expected_state = state.State.from_raw(
        meta_info=state_meta,
        scratches=[state.IntBoolean(value)],
    )
    if current_state != expected_state:
        print('current_state:', env.symbol(current_state))
        print('expected_state:', env.symbol(expected_state))
    assert current_state == expected_state

    assert env.full_state.goal_achieved() is False

    # Run Action
    action_idx = get_basic_action_index(action_impl.VerifyGoal, env)
    meta_idx = get_from_int_type_index(state.StateScratchIndex, env)
    scratch_idx = 1
    raw_action = action.RawAction(
        full_state.MetaAllowedBasicActionsTypeIndex(action_idx),
        core.Integer(0),
        core.Integer(meta_idx),
        core.Integer(scratch_idx),
    )
    full_action = action_impl.VerifyGoal(
        core.Optional(),
        full_state.MetaFromIntTypeIndex(meta_idx),
        core.Integer(scratch_idx),
    )
    output = action_impl.VerifyGoalOutput(
        core.Optional(),
        state.StateScratchIndex(scratch_idx),
    )
    env.step(raw_action)
    current_state = get_current_state(env)
    last_history_action = get_last_history_action(env)

    # Verify
    expected_history = full_state.SuccessActionData.from_args(
        raw_action=core.Optional(raw_action),
        action=core.Optional(full_action),
        output=core.Optional(output),
        exception=core.Optional(),
    )
    if last_history_action != expected_history:
        print('last_history_action:', env.symbol(last_history_action))
        print('expected_history:', env.symbol(expected_history))
    assert last_history_action == expected_history

    state_meta = state_meta.with_new_args(
        goal_achieved=state.GoalAchieved.achieved(),
    )
    expected_state = state.State.from_raw(
        meta_info=state_meta,
        scratches=[state.IntBoolean(value)],
    )
    if current_state != expected_state:
        print('current_state:', env.symbol(current_state))
        print('expected_state:', env.symbol(expected_state))
    assert current_state == expected_state

    return env.full_state

def has_goal(env: GoalEnv, goal: meta_env.IGoal):
    selected_goal = env.full_state.nested_arg(
        (full_state.FullState.idx_meta, meta_env.MetaInfo.idx_goal)
    ).apply()
    return selected_goal == goal

def test_boolean() -> list[full_state.FullState]:
    final_states: list[full_state.FullState] = []

    final_states.append(run(goal_expr=core.IBoolean.false().as_node, result=False))
    final_states.append(run(goal_expr=core.IBoolean.true().as_node, result=True))

    final_states.append(run(goal_expr=core.Not(core.IBoolean.false()), result=True))
    final_states.append(run(goal_expr=core.Not(core.IBoolean.true()), result=False))

    final_states.append(run(goal_expr=core.IsEmpty(core.Optional()), result=True))
    final_states.append(run(goal_expr=core.IsEmpty(
        core.Optional(core.IBoolean.false())
    ), result=False))
    final_states.append(run(goal_expr=core.IsEmpty(
        core.Optional(core.IBoolean.true())
    ), result=False))
    final_states.append(run(goal_expr=core.IsEmpty(
        core.Optional(core.Optional())
    ), result=False))

    final_states.append(run(goal_expr=core.IsInstance(
        core.IBoolean.false(),
        core.IBoolean.as_type()
    ), result=True))
    final_states.append(run(goal_expr=core.IsInstance(
        core.IBoolean.false(),
        core.Optional.as_type()
    ), result=False))
    final_states.append(run(goal_expr=core.IsInstance(
        core.IntBoolean.true(),
        core.IBoolean.as_type()
    ), result=True))
    final_states.append(run(goal_expr=core.IsInstance(
        core.Optional(),
        core.IBoolean.as_type()
    ), result=False))
    final_states.append(run(goal_expr=core.IsInstance(
        core.Optional(core.IntBoolean.true()),
        core.Optional.as_type()
    ), result=True))
    final_states.append(run(goal_expr=core.IsInstance(
        core.Optional(),
        core.Optional.as_type()
    ), result=True))
    final_states.append(run(goal_expr=core.IsInstance(
        core.Eq(core.Optional(), core.Optional(core.IBoolean.false())),
        core.IBoolean.as_type()
    ), result=True))

    final_states.append(run(goal_expr=core.Eq(
        core.IBoolean.false(),
        core.IBoolean.false(),
    ), result=True))
    final_states.append(run(goal_expr=core.Eq(
        core.IBoolean.false(),
        core.IBoolean.true(),
    ), result=False))
    final_states.append(run(goal_expr=core.Eq(
        core.IBoolean.true(),
        core.IBoolean.false(),
    ), result=False))
    final_states.append(run(goal_expr=core.Eq(
        core.IBoolean.true(),
        core.IBoolean.true(),
    ), result=True))

    final_states.append(run(goal_expr=core.SameArgsAmount(
        core.IBoolean.true(),
        core.IBoolean.false(),
    ), result=True))
    final_states.append(run(goal_expr=core.SameArgsAmount(
        core.DefaultGroup(core.Integer(1), core.Integer(2)),
        core.DefaultGroup(core.Integer(3), core.Integer(4), core.Integer(5)),
    ), result=False))
    final_states.append(run(goal_expr=core.SameArgsAmount(
        core.DefaultGroup(core.Integer(1), core.Integer(2)),
        core.DefaultGroup(core.Integer(3), core.Integer(4)),
    ), result=True))

    final_states.append(run(goal_expr=core.And(
        core.IBoolean.false(),
        core.IBoolean.false(),
    ), result=False))
    final_states.append(run(goal_expr=core.And(
        core.IBoolean.false(),
        core.IBoolean.true(),
    ), result=False))
    final_states.append(run(goal_expr=core.And(
        core.IBoolean.true(),
        core.IBoolean.false(),
    ), result=False))
    final_states.append(run(goal_expr=core.And(
        core.IBoolean.true(),
        core.IBoolean.true(),
    ), result=True))
    final_states.append(run(goal_expr=core.And(
        core.IBoolean.true(),
        core.IBoolean.false(),
        core.IBoolean.true(),
    ), result=False))
    final_states.append(run(goal_expr=core.And(
        core.IBoolean.true(),
        core.IBoolean.true(),
        core.IBoolean.true(),
    ), result=True))

    final_states.append(run(goal_expr=core.Or(
        core.IBoolean.false(),
        core.IBoolean.false(),
    ), result=False))
    final_states.append(run(goal_expr=core.Or(
        core.IBoolean.false(),
        core.IBoolean.true(),
    ), result=True))
    final_states.append(run(goal_expr=core.Or(
        core.IBoolean.true(),
        core.IBoolean.false(),
    ), result=True))
    final_states.append(run(goal_expr=core.Or(
        core.IBoolean.true(),
        core.IBoolean.true(),
    ), result=True))
    final_states.append(run(goal_expr=core.Or(
        core.IBoolean.false(),
        core.IBoolean.false(),
        core.IBoolean.true(),
    ), result=True))
    final_states.append(run(goal_expr=core.Or(
        core.IBoolean.false(),
        core.IBoolean.false(),
        core.IBoolean.false(),
    ), result=False))

    final_states.append(run(goal_expr=core.GreaterThan(
        core.Integer(1),
        core.Integer(2),
    ), result=False))
    final_states.append(run(goal_expr=core.GreaterThan(
        core.Integer(2),
        core.Integer(1),
    ), result=True))
    final_states.append(run(goal_expr=core.GreaterThan(
        core.Integer(1),
        core.Integer(1),
    ), result=False))

    final_states.append(run(goal_expr=less_than_case(), result=True))
    final_states.append(run(goal_expr=core.LessThan(
        core.Integer(2),
        core.Integer(1),
    ), result=False))
    final_states.append(run(goal_expr=core.LessThan(
        core.Integer(1),
        core.Integer(1),
    ), result=False))

    final_states.append(run(goal_expr=core.GreaterOrEqual(
        core.Integer(1),
        core.Integer(2),
    ), result=False))
    final_states.append(run(goal_expr=core.GreaterOrEqual(
        core.Integer(2),
        core.Integer(1),
    ), result=True))
    final_states.append(run(goal_expr=core.GreaterOrEqual(
        core.Integer(1),
        core.Integer(1),
    ), result=True))

    final_states.append(run(goal_expr=core.LessOrEqual(
        core.Integer(1),
        core.Integer(2),
    ), result=True))
    final_states.append(run(goal_expr=core.LessOrEqual(
        core.Integer(2),
        core.Integer(1),
    ), result=False))
    final_states.append(run(goal_expr=core.LessOrEqual(
        core.Integer(1),
        core.Integer(1),
    ), result=True))

    final_states.append(run(goal_expr=core.IsInsideRange(
        core.Integer(1),
        core.Integer(2),
        core.Integer(3),
    ), result=False))
    final_states.append(run(goal_expr=core.IsInsideRange(
        core.Integer(2),
        core.Integer(1),
        core.Integer(3),
    ), result=True))
    final_states.append(run(goal_expr=core.IsInsideRange(
        core.Integer(4),
        core.Integer(4),
        core.Integer(4),
    ), result=True))
    final_states.append(run(goal_expr=core.IsInsideRange(
        core.NodeArgIndex(6),
        core.Integer(1),
        core.Integer(9),
    ), result=True))
    final_states.append(run(goal_expr=core.IsInsideRange(
        core.NodeArgIndex(3),
        core.Integer(7),
        core.Integer(9),
    ), result=False))

    return final_states

def test() -> list[full_state.FullState]:
    final_states: list[full_state.FullState] = []
    final_states += test_boolean()
    return final_states

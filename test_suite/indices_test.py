from env import core, state, full_state, action_impl, meta_env, node_types
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

def get_meta_group_type_index(meta_idx: int, node_type: type[core.INode], env: GoalEnv):
    selected_types = env.full_state.meta.apply().inner_arg(
        meta_idx
    ).apply().cast(meta_env.GeneralTypeGroup)
    meta_idx = selected_types.as_tuple.index(node_type.as_type()) + 1
    return meta_idx

def get_meta_subgroup_type_index(meta_idx: int, node_type: type[core.INode], env: GoalEnv):
    selected_types = env.full_state.meta.apply().nested_arg((
        meta_idx,
        meta_env.SubtypeOuterGroup.idx_subtypes,
    )).apply().cast(meta_env.GeneralTypeGroup)
    meta_idx = selected_types.as_tuple.index(node_type.as_type()) + 1
    return meta_idx

def get_type_details(node_type: type[core.INode], env: GoalEnv) -> meta_env.DetailedType:
    type_idx = env.full_state.node_types().index(node_type)
    details = env.full_state.meta.apply().nested_arg((
        meta_env.MetaInfo.idx_all_types_details,
    )).apply().cast(meta_env.DetailedTypeGroup)
    result = details.as_tuple[type_idx]
    assert isinstance(result, meta_env.DetailedType)
    return result

def run(
    env: GoalEnv,
    state_meta: state.StateMetaInfo,
    scratches: list[core.INode | None],
    args_groups: list[state.PartialArgsGroup],
    scratch_idx: int,
    node_type: type[core.INode],
    value: int,
    new_scratch: core.INode,
):
    # Run Action
    meta_idx = get_meta_subgroup_type_index(
        meta_env.MetaInfo.idx_full_state_int_index_group,
        node_type,
        env)
    raw_action = action_impl.DefineScratchFromIntIndex.from_raw(scratch_idx, meta_idx, value)
    full_action = action_impl.DefineScratchFromIntIndex(
        state.StateScratchIndex(scratch_idx),
        full_state.MetaFullStateIntIndexTypeIndex(meta_idx),
        core.Integer(value),
    )
    output = action_impl.DefineScratchOutput(
        state.StateScratchIndex(scratch_idx),
        state.Scratch(new_scratch),
    )
    env.step(raw_action)
    current_state = get_current_state(env)
    last_history_action = get_last_history_action(env)

    # Verify
    scratches = [
        new_scratch if i == scratch_idx-1 else s
        for i, s in enumerate(scratches)
    ]
    assert current_state == state.State.from_raw(
        meta_info=state_meta,
        scratches=scratches,
        args_groups=args_groups,
    )
    assert last_history_action == full_state.SuccessActionData.from_args(
        raw_action=core.Optional(),
        action=core.Optional(full_action),
        output=core.Optional(output),
        exception=core.Optional(),
    )
    assert env.full_state.goal_achieved() is False

    return scratches

def has_goal(env: GoalEnv, goal: meta_env.IGoal):
    selected_goal = env.full_state.nested_arg(
        (full_state.FullState.idx_meta, meta_env.MetaInfo.idx_goal)
    ).apply()
    return selected_goal == goal

def build_case(max_steps: int | None = None):
    """GoalEnv with the indices initial state, shared with scripts/run_case.py.

    Returns (env, goal, state_meta, scratches, args_groups); test_indices uses
    this same construction so the runner and the test cannot drift.
    """
    goal = node_types.HaveScratch.with_goal(core.Void())
    state_meta = state.StateMetaInfo.with_goal_expr(goal)
    scratches: list[core.INode | None] = [
        core.And(core.Void(), core.Void()),
        core.Or(
            core.LessThan(core.Param.from_int(1), core.Integer(0)),
            core.GreaterThan(core.Param.from_int(1), core.Integer(1))
        ),
        node_types.HaveScratch.with_goal(core.And(
            core.LessThan(core.Var.from_int(2), core.Var.from_int(1)),
            core.GreaterThan(core.Var.from_int(3), core.Var.from_int(1)),
            core.Or(
                core.Param.from_int(1),
                core.GreaterThan(core.Var.from_int(2), core.Var.from_int(1)),
                core.LessThan(core.Var.from_int(3), core.Var.from_int(1)),
            )
        )),
        None,
        action_impl.DeleteArgsGroupOutput.from_raw(3, 0, 0),
    ]
    args_groups = [
        state.PartialArgsGroup(
            core.Optional(core.TypeNode(core.Param)),
            core.Optional(core.TypeNode(core.Var)),
            core.Optional(state.State.create()),
        ),
        state.PartialArgsGroup.from_int(5),
        state.PartialArgsGroup.create(),
        state.PartialArgsGroup(
            core.Optional(state.PartialArgsGroup.from_int(2)),
        ),
    ]

    env = GoalEnv(
        goal=goal,
        fn_initial_state=lambda meta: full_state.FullState.with_args(
            meta=meta,
            current=full_state.HistoryNode.with_args(
                state=state.State.from_raw(
                    meta_info=state_meta,
                    scratches=scratches,
                    args_groups=args_groups,
                ),
                meta_data=full_state.MetaData.create(),
            )
        ),
        max_steps=max_steps,
    )
    return env, goal, state_meta, scratches, args_groups

def test_indices() -> list[full_state.FullState]:
    env, goal, state_meta, scratches, args_groups = build_case()

    assert has_goal(env=env, goal=goal)

    current_state = get_current_state(env)

    state_meta = state.StateMetaInfo.with_goal_achieved(state.GoalAchieved.create())
    assert current_state == state.State.from_raw(
        meta_info=state_meta,
        scratches=scratches,
        args_groups=args_groups,
    )
    assert env.full_state.goal_achieved() is False

    # CurrentStateArgsOuterGroupIndex
    scratches = run(
        env=env,
        state_meta=state_meta,
        scratches=scratches,
        args_groups=args_groups,
        scratch_idx=1,
        node_type=full_state.CurrentStateArgsOuterGroupIndex,
        value=2,
        new_scratch=args_groups[1],
    )
    scratches = run(
        env=env,
        state_meta=state_meta,
        scratches=scratches,
        args_groups=args_groups,
        scratch_idx=2,
        node_type=full_state.CurrentStateArgsOuterGroupIndex,
        value=1,
        new_scratch=args_groups[0],
    )
    scratches = run(
        env=env,
        state_meta=state_meta,
        scratches=scratches,
        args_groups=args_groups,
        scratch_idx=4,
        node_type=full_state.CurrentStateArgsOuterGroupIndex,
        value=3,
        new_scratch=args_groups[2],
    )

    # CurrentStateScratchIndex
    scratches = run(
        env=env,
        state_meta=state_meta,
        scratches=scratches,
        args_groups=args_groups,
        scratch_idx=1,
        node_type=full_state.CurrentStateScratchIndex,
        value=3,
        new_scratch=state.Scratch.with_value(scratches[2]),
    )
    scratches = run(
        env=env,
        state_meta=state_meta,
        scratches=scratches,
        args_groups=args_groups,
        scratch_idx=4,
        node_type=full_state.CurrentStateScratchIndex,
        value=5,
        new_scratch=state.Scratch.with_value(scratches[4]),
    )
    scratches = run(
        env=env,
        state_meta=state_meta,
        scratches=scratches,
        args_groups=args_groups,
        scratch_idx=2,
        node_type=full_state.CurrentStateScratchIndex,
        value=1,
        new_scratch=state.Scratch.with_value(scratches[0]),
    )

    # MetaAllActionsTypeIndex
    scratches = run(
        env=env,
        state_meta=state_meta,
        scratches=scratches,
        args_groups=args_groups,
        scratch_idx=1,
        node_type=full_state.MetaAllActionsTypeIndex,
        value=get_meta_subgroup_type_index(
            meta_env.MetaInfo.idx_all_actions,
            action_impl.CreateScratchOutput,
            env),
        new_scratch=core.TypeNode(action_impl.CreateScratchOutput),
    )
    scratches = run(
        env=env,
        state_meta=state_meta,
        scratches=scratches,
        args_groups=args_groups,
        scratch_idx=4,
        node_type=full_state.MetaAllActionsTypeIndex,
        value=get_meta_subgroup_type_index(
            meta_env.MetaInfo.idx_all_actions,
            action_impl.DefineScratchOutput,
            env),
        new_scratch=core.TypeNode(action_impl.DefineScratchOutput),
    )
    scratches = run(
        env=env,
        state_meta=state_meta,
        scratches=scratches,
        args_groups=args_groups,
        scratch_idx=2,
        node_type=full_state.MetaAllActionsTypeIndex,
        value=get_meta_subgroup_type_index(
            meta_env.MetaInfo.idx_all_actions,
            action_impl.VerifyGoal,
            env),
        new_scratch=core.TypeNode(action_impl.VerifyGoal),
    )

    # MetaAllTypesTypeIndex
    scratches = run(
        env=env,
        state_meta=state_meta,
        scratches=scratches,
        args_groups=args_groups,
        scratch_idx=1,
        node_type=full_state.MetaAllTypesTypeIndex,
        value=env.full_state.node_types().index(core.Param) + 1,
        new_scratch=core.TypeNode(core.Param),
    )
    scratches = run(
        env=env,
        state_meta=state_meta,
        scratches=scratches,
        args_groups=args_groups,
        scratch_idx=4,
        node_type=full_state.MetaAllTypesTypeIndex,
        value=env.full_state.node_types().index(core.FunctionCall) + 1,
        new_scratch=core.TypeNode(core.FunctionCall),
    )
    scratches = run(
        env=env,
        state_meta=state_meta,
        scratches=scratches,
        args_groups=args_groups,
        scratch_idx=2,
        node_type=full_state.MetaAllTypesTypeIndex,
        value=env.full_state.node_types().index(state.PartialArgsOuterGroup) + 1,
        new_scratch=core.TypeNode(state.PartialArgsOuterGroup),
    )

    # MetaAllowedActionsTypeIndex
    scratches = run(
        env=env,
        state_meta=state_meta,
        scratches=scratches,
        args_groups=args_groups,
        scratch_idx=1,
        node_type=full_state.MetaAllowedActionsTypeIndex,
        value=get_meta_group_type_index(
            meta_env.MetaInfo.idx_allowed_actions,
            action_impl.DefineScratchFromInt,
            env),
        new_scratch=core.TypeNode(action_impl.DefineScratchFromInt),
    )
    scratches = run(
        env=env,
        state_meta=state_meta,
        scratches=scratches,
        args_groups=args_groups,
        scratch_idx=4,
        node_type=full_state.MetaAllowedActionsTypeIndex,
        value=get_meta_group_type_index(
            meta_env.MetaInfo.idx_allowed_actions,
            action_impl.DeleteArgsGroupOutput,
            env),
        new_scratch=core.TypeNode(action_impl.DeleteArgsGroupOutput),
    )
    scratches = run(
        env=env,
        state_meta=state_meta,
        scratches=scratches,
        args_groups=args_groups,
        scratch_idx=2,
        node_type=full_state.MetaAllowedActionsTypeIndex,
        value=get_meta_group_type_index(
            meta_env.MetaInfo.idx_allowed_actions,
            action_impl.UpdateScratchFromAnother,
            env),
        new_scratch=core.TypeNode(action_impl.UpdateScratchFromAnother),
    )

    # MetaAllowedBasicActionsTypeIndex
    scratches = run(
        env=env,
        state_meta=state_meta,
        scratches=scratches,
        args_groups=args_groups,
        scratch_idx=1,
        node_type=full_state.MetaAllowedBasicActionsTypeIndex,
        value=get_meta_group_type_index(
            meta_env.MetaInfo.idx_allowed_basic_actions,
            action_impl.DefineScratchFromIntIndex,
            env),
        new_scratch=core.TypeNode(action_impl.DefineScratchFromIntIndex),
    )
    scratches = run(
        env=env,
        state_meta=state_meta,
        scratches=scratches,
        args_groups=args_groups,
        scratch_idx=4,
        node_type=full_state.MetaAllowedBasicActionsTypeIndex,
        value=get_meta_group_type_index(
            meta_env.MetaInfo.idx_allowed_basic_actions,
            action_impl.DefineScratchFromDefault,
            env),
        new_scratch=core.TypeNode(action_impl.DefineScratchFromDefault),
    )
    scratches = run(
        env=env,
        state_meta=state_meta,
        scratches=scratches,
        args_groups=args_groups,
        scratch_idx=4,
        node_type=full_state.MetaAllowedBasicActionsTypeIndex,
        value=get_meta_group_type_index(
            meta_env.MetaInfo.idx_allowed_basic_actions,
            action_impl.CreateArgsGroup,
            env),
        new_scratch=core.TypeNode(action_impl.CreateArgsGroup),
    )

    # MetaBasicActionsTypeIndex
    scratches = run(
        env=env,
        state_meta=state_meta,
        scratches=scratches,
        args_groups=args_groups,
        scratch_idx=4,
        node_type=full_state.MetaBasicActionsTypeIndex,
        value=get_meta_subgroup_type_index(
            meta_env.MetaInfo.idx_basic_actions,
            action_impl.DefineScratchFromIntIndex,
            env),
        new_scratch=core.TypeNode(action_impl.DefineScratchFromIntIndex),
    )
    scratches = run(
        env=env,
        state_meta=state_meta,
        scratches=scratches,
        args_groups=args_groups,
        scratch_idx=2,
        node_type=full_state.MetaBasicActionsTypeIndex,
        value=get_meta_subgroup_type_index(
            meta_env.MetaInfo.idx_basic_actions,
            action_impl.VerifyDynamicGoal,
            env),
        new_scratch=core.TypeNode(action_impl.VerifyDynamicGoal),
    )
    scratches = run(
        env=env,
        state_meta=state_meta,
        scratches=scratches,
        args_groups=args_groups,
        scratch_idx=1,
        node_type=full_state.MetaBasicActionsTypeIndex,
        value=get_meta_subgroup_type_index(
            meta_env.MetaInfo.idx_basic_actions,
            action_impl.DeleteArgsGroupOutput,
            env),
        new_scratch=core.TypeNode(action_impl.DeleteArgsGroupOutput),
    )

    # MetaBooleanTypeIndex
    scratches = run(
        env=env,
        state_meta=state_meta,
        scratches=scratches,
        args_groups=args_groups,
        scratch_idx=2,
        node_type=full_state.MetaBooleanTypeIndex,
        value=get_meta_subgroup_type_index(
            meta_env.MetaInfo.idx_boolean_group,
            core.And,
            env),
        new_scratch=core.TypeNode(core.And),
    )
    scratches = run(
        env=env,
        state_meta=state_meta,
        scratches=scratches,
        args_groups=args_groups,
        scratch_idx=4,
        node_type=full_state.MetaBooleanTypeIndex,
        value=get_meta_subgroup_type_index(
            meta_env.MetaInfo.idx_boolean_group,
            core.Not,
            env),
        new_scratch=core.TypeNode(core.Not),
    )
    scratches = run(
        env=env,
        state_meta=state_meta,
        scratches=scratches,
        args_groups=args_groups,
        scratch_idx=1,
        node_type=full_state.MetaBooleanTypeIndex,
        value=get_meta_subgroup_type_index(
            meta_env.MetaInfo.idx_boolean_group,
            state.GoalAchieved,
            env),
        new_scratch=core.TypeNode(state.GoalAchieved),
    )

    # MetaDefaultTypeIndex
    scratches = run(
        env=env,
        state_meta=state_meta,
        scratches=scratches,
        args_groups=args_groups,
        scratch_idx=2,
        node_type=full_state.MetaDefaultTypeIndex,
        value=get_meta_subgroup_type_index(
            meta_env.MetaInfo.idx_default_group,
            core.Protocol,
            env),
        new_scratch=core.TypeNode(core.Protocol),
    )
    scratches = run(
        env=env,
        state_meta=state_meta,
        scratches=scratches,
        args_groups=args_groups,
        scratch_idx=4,
        node_type=full_state.MetaDefaultTypeIndex,
        value=get_meta_subgroup_type_index(
            meta_env.MetaInfo.idx_default_group,
            core.Void,
            env),
        new_scratch=core.TypeNode(core.Void),
    )
    scratches = run(
        env=env,
        state_meta=state_meta,
        scratches=scratches,
        args_groups=args_groups,
        scratch_idx=1,
        node_type=full_state.MetaDefaultTypeIndex,
        value=get_meta_subgroup_type_index(
            meta_env.MetaInfo.idx_default_group,
            state.State,
            env),
        new_scratch=core.TypeNode(state.State),
    )

    # MetaFromIntTypeIndex
    scratches = run(
        env=env,
        state_meta=state_meta,
        scratches=scratches,
        args_groups=args_groups,
        scratch_idx=2,
        node_type=full_state.MetaFromIntTypeIndex,
        value=get_meta_subgroup_type_index(
            meta_env.MetaInfo.idx_from_int_group,
            core.Param,
            env),
        new_scratch=core.TypeNode(core.Param),
    )
    scratches = run(
        env=env,
        state_meta=state_meta,
        scratches=scratches,
        args_groups=args_groups,
        scratch_idx=4,
        node_type=full_state.MetaFromIntTypeIndex,
        value=get_meta_subgroup_type_index(
            meta_env.MetaInfo.idx_from_int_group,
            core.Integer,
            env),
        new_scratch=core.TypeNode(core.Integer),
    )
    scratches = run(
        env=env,
        state_meta=state_meta,
        scratches=scratches,
        args_groups=args_groups,
        scratch_idx=1,
        node_type=full_state.MetaFromIntTypeIndex,
        value=get_meta_subgroup_type_index(
            meta_env.MetaInfo.idx_from_int_group,
            core.CountableTypeGroup,
            env),
        new_scratch=core.TypeNode(core.CountableTypeGroup),
    )

    # MetaFullStateIndexTypeIndex
    scratches = run(
        env=env,
        state_meta=state_meta,
        scratches=scratches,
        args_groups=args_groups,
        scratch_idx=2,
        node_type=full_state.MetaFullStateIndexTypeIndex,
        value=get_meta_subgroup_type_index(
            meta_env.MetaInfo.idx_full_state_index_group,
            full_state.CurrentStateDefinitionIndex,
            env),
        new_scratch=core.TypeNode(full_state.CurrentStateDefinitionIndex),
    )
    scratches = run(
        env=env,
        state_meta=state_meta,
        scratches=scratches,
        args_groups=args_groups,
        scratch_idx=4,
        node_type=full_state.MetaFullStateIndexTypeIndex,
        value=get_meta_subgroup_type_index(
            meta_env.MetaInfo.idx_full_state_index_group,
            full_state.CurrentStateScratchIndex,
            env),
        new_scratch=core.TypeNode(full_state.CurrentStateScratchIndex),
    )
    scratches = run(
        env=env,
        state_meta=state_meta,
        scratches=scratches,
        args_groups=args_groups,
        scratch_idx=1,
        node_type=full_state.MetaFullStateIndexTypeIndex,
        value=get_meta_subgroup_type_index(
            meta_env.MetaInfo.idx_full_state_index_group,
            full_state.MetaAllActionsTypeIndex,
            env),
        new_scratch=core.TypeNode(full_state.MetaAllActionsTypeIndex),
    )

    # MetaFullStateIntIndexTypeIndex
    scratches = run(
        env=env,
        state_meta=state_meta,
        scratches=scratches,
        args_groups=args_groups,
        scratch_idx=2,
        node_type=full_state.MetaFullStateIntIndexTypeIndex,
        value=get_meta_subgroup_type_index(
            meta_env.MetaInfo.idx_full_state_int_index_group,
            full_state.CurrentStateArgsOuterGroupIndex,
            env),
        new_scratch=core.TypeNode(full_state.CurrentStateArgsOuterGroupIndex),
    )
    scratches = run(
        env=env,
        state_meta=state_meta,
        scratches=scratches,
        args_groups=args_groups,
        scratch_idx=4,
        node_type=full_state.MetaFullStateIntIndexTypeIndex,
        value=get_meta_subgroup_type_index(
            meta_env.MetaInfo.idx_full_state_int_index_group,
            full_state.MetaAllTypesTypeIndex,
            env),
        new_scratch=core.TypeNode(full_state.MetaAllTypesTypeIndex),
    )
    scratches = run(
        env=env,
        state_meta=state_meta,
        scratches=scratches,
        args_groups=args_groups,
        scratch_idx=1,
        node_type=full_state.MetaFullStateIntIndexTypeIndex,
        value=get_meta_subgroup_type_index(
            meta_env.MetaInfo.idx_full_state_int_index_group,
            full_state.MetaAllowedActionsTypeIndex,
            env),
        new_scratch=core.TypeNode(full_state.MetaAllowedActionsTypeIndex),
    )

    # MetaFunctionTypeIndex
    scratches = run(
        env=env,
        state_meta=state_meta,
        scratches=scratches,
        args_groups=args_groups,
        scratch_idx=4,
        node_type=full_state.MetaFunctionTypeIndex,
        value=get_meta_subgroup_type_index(
            meta_env.MetaInfo.idx_function_group,
            core.FunctionExpr,
            env),
        new_scratch=core.TypeNode(core.FunctionExpr),
    )
    scratches = run(
        env=env,
        state_meta=state_meta,
        scratches=scratches,
        args_groups=args_groups,
        scratch_idx=2,
        node_type=full_state.MetaFunctionTypeIndex,
        value=get_meta_subgroup_type_index(
            meta_env.MetaInfo.idx_function_group,
            core.FunctionWrapper,
            env),
        new_scratch=core.TypeNode(core.FunctionWrapper),
    )
    scratches = run(
        env=env,
        state_meta=state_meta,
        scratches=scratches,
        args_groups=args_groups,
        scratch_idx=1,
        node_type=full_state.MetaFunctionTypeIndex,
        value=get_meta_subgroup_type_index(
            meta_env.MetaInfo.idx_function_group,
            core.TypeNode,
            env),
        new_scratch=core.TypeNode(core.TypeNode),
    )

    # MetaGroupTypeIndex
    scratches = run(
        env=env,
        state_meta=state_meta,
        scratches=scratches,
        args_groups=args_groups,
        scratch_idx=4,
        node_type=full_state.MetaGroupTypeIndex,
        value=get_meta_subgroup_type_index(
            meta_env.MetaInfo.idx_group_outer_group,
            core.DefaultGroup,
            env),
        new_scratch=core.TypeNode(core.DefaultGroup),
    )
    scratches = run(
        env=env,
        state_meta=state_meta,
        scratches=scratches,
        args_groups=args_groups,
        scratch_idx=2,
        node_type=full_state.MetaGroupTypeIndex,
        value=get_meta_subgroup_type_index(
            meta_env.MetaInfo.idx_group_outer_group,
            meta_env.GeneralTypeGroup,
            env),
        new_scratch=core.TypeNode(meta_env.GeneralTypeGroup),
    )
    scratches = run(
        env=env,
        state_meta=state_meta,
        scratches=scratches,
        args_groups=args_groups,
        scratch_idx=1,
        node_type=full_state.MetaGroupTypeIndex,
        value=get_meta_subgroup_type_index(
            meta_env.MetaInfo.idx_group_outer_group,
            state.PartialArgsGroup,
            env),
        new_scratch=core.TypeNode(state.PartialArgsGroup),
    )

    # MetaIntTypeIndex
    scratches = run(
        env=env,
        state_meta=state_meta,
        scratches=scratches,
        args_groups=args_groups,
        scratch_idx=4,
        node_type=full_state.MetaIntTypeIndex,
        value=get_meta_subgroup_type_index(
            meta_env.MetaInfo.idx_int_group,
            core.Integer,
            env),
        new_scratch=core.TypeNode(core.Integer),
    )
    scratches = run(
        env=env,
        state_meta=state_meta,
        scratches=scratches,
        args_groups=args_groups,
        scratch_idx=2,
        node_type=full_state.MetaIntTypeIndex,
        value=get_meta_subgroup_type_index(
            meta_env.MetaInfo.idx_int_group,
            core.NodeArgIndex,
            env),
        new_scratch=core.TypeNode(core.NodeArgIndex),
    )
    scratches = run(
        env=env,
        state_meta=state_meta,
        scratches=scratches,
        args_groups=args_groups,
        scratch_idx=1,
        node_type=full_state.MetaIntTypeIndex,
        value=get_meta_subgroup_type_index(
            meta_env.MetaInfo.idx_int_group,
            full_state.MetaIntTypeIndex,
            env),
        new_scratch=core.TypeNode(full_state.MetaIntTypeIndex),
    )

    # MetaNodeIndexTypeIndex
    scratches = run(
        env=env,
        state_meta=state_meta,
        scratches=scratches,
        args_groups=args_groups,
        scratch_idx=4,
        node_type=full_state.MetaNodeIndexTypeIndex,
        value=get_meta_subgroup_type_index(
            meta_env.MetaInfo.idx_node_index_group,
            core.NodeArgIndex,
            env),
        new_scratch=core.TypeNode(core.NodeArgIndex),
    )
    scratches = run(
        env=env,
        state_meta=state_meta,
        scratches=scratches,
        args_groups=args_groups,
        scratch_idx=2,
        node_type=full_state.MetaNodeIndexTypeIndex,
        value=get_meta_subgroup_type_index(
            meta_env.MetaInfo.idx_node_index_group,
            core.NodeMainIndex,
            env),
        new_scratch=core.TypeNode(core.NodeMainIndex),
    )
    scratches = run(
        env=env,
        state_meta=state_meta,
        scratches=scratches,
        args_groups=args_groups,
        scratch_idx=1,
        node_type=full_state.MetaNodeIndexTypeIndex,
        value=get_meta_subgroup_type_index(
            meta_env.MetaInfo.idx_node_index_group,
            full_state.MetaGroupTypeIndex,
            env),
        new_scratch=core.TypeNode(full_state.MetaGroupTypeIndex),
    )

    # MetaSingleChildTypeIndex
    scratches = run(
        env=env,
        state_meta=state_meta,
        scratches=scratches,
        args_groups=args_groups,
        scratch_idx=4,
        node_type=full_state.MetaSingleChildTypeIndex,
        value=get_meta_subgroup_type_index(
            meta_env.MetaInfo.idx_single_child_group,
            state.Scratch,
            env),
        new_scratch=core.TypeNode(state.Scratch),
    )
    scratches = run(
        env=env,
        state_meta=state_meta,
        scratches=scratches,
        args_groups=args_groups,
        scratch_idx=2,
        node_type=full_state.MetaSingleChildTypeIndex,
        value=get_meta_subgroup_type_index(
            meta_env.MetaInfo.idx_single_child_group,
            full_state.FullState,
            env),
        new_scratch=core.TypeNode(full_state.FullState),
    )
    scratches = run(
        env=env,
        state_meta=state_meta,
        scratches=scratches,
        args_groups=args_groups,
        scratch_idx=1,
        node_type=full_state.MetaSingleChildTypeIndex,
        value=get_meta_subgroup_type_index(
            meta_env.MetaInfo.idx_single_child_group,
            core.ExceptionInfoWrapper,
            env),
        new_scratch=core.TypeNode(core.ExceptionInfoWrapper),
    )

    # MetaTypesDetailsTypeIndex
    scratches = run(
        env=env,
        state_meta=state_meta,
        scratches=scratches,
        args_groups=args_groups,
        scratch_idx=4,
        node_type=full_state.MetaTypesDetailsTypeIndex,
        value=env.full_state.node_types().index(core.Param) + 1,
        new_scratch=get_type_details(core.Param, env),
    )
    scratches = run(
        env=env,
        state_meta=state_meta,
        scratches=scratches,
        args_groups=args_groups,
        scratch_idx=2,
        node_type=full_state.MetaTypesDetailsTypeIndex,
        value=env.full_state.node_types().index(state.ScratchGroup) + 1,
        new_scratch=get_type_details(state.ScratchGroup, env),
    )
    scratches = run(
        env=env,
        state_meta=state_meta,
        scratches=scratches,
        args_groups=args_groups,
        scratch_idx=2,
        node_type=full_state.MetaTypesDetailsTypeIndex,
        value=env.full_state.node_types().index(core.BaseNode) + 1,
        new_scratch=get_type_details(core.BaseNode, env),
    )

    # Final Verification
    full_state_int_indices = env.full_state.nested_arg((
        full_state.FullState.idx_meta,
        meta_env.MetaInfo.idx_full_state_int_index_group,
        meta_env.SubtypeOuterGroup.idx_subtypes
    )).apply().cast(meta_env.GeneralTypeGroup).as_tuple
    set_indices_used = set()
    history = env.full_state.history.apply().cast(full_state.HistoryGroupNode)
    for h in history.as_tuple:
        action_data = h.action_data.apply().cast(
            full_state.Optional[full_state.BaseActionData]
        ).value_or_raise
        assert isinstance(action_data, full_state.BaseActionData)
        act = action_data.action.apply().cast(core.IOptional[meta_env.IAction]).value_or_raise
        if isinstance(act, action_impl.DefineScratchFromIntIndex):
            full_state_int_index = act.inner_arg(
                act.idx_type_index
            ).apply().cast(full_state.MetaFullStateIntIndexTypeIndex)
            index = full_state_int_indices[full_state_int_index.as_int - 1]
            if index not in set_indices_used:
                set_indices_used.add(index)
    ignore = set([
        core.TypeNode(full_state.CurrentStateDefinitionIndex),
    ])
    for index in full_state_int_indices:
        if index not in ignore:
            assert index in set_indices_used, f'Index {index.type} not used'

    return [env.full_state]

def test() -> list[full_state.FullState]:
    final_states: list[full_state.FullState] = []
    final_states += test_indices()
    return final_states

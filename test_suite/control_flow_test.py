# pylint: disable=too-many-lines
from env import core, full_state, state, meta_env, action_impl, node_types, composite
from env.goal_env import GoalEnv
from test_suite import test_utils

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

def get_meta_subgroup_type_index(meta_idx: int, node_type: type[core.INode], env: GoalEnv):
    selected_types = env.full_state.meta.apply().nested_arg((
        meta_idx,
        meta_env.SubtypeOuterGroup.idx_subtypes,
    )).apply().cast(meta_env.GeneralTypeGroup)
    meta_idx = selected_types.as_tuple.index(node_type.as_type()) + 1
    return meta_idx

def run(
    env: GoalEnv,
    state_meta: state.StateMetaInfo,
    scratches: list[core.INode | None],
    args_groups: list[state.PartialArgsGroup],
    scratch_idx: int,
    new_scratch: core.INode,
):
    # Run Action
    raw_action = action_impl.RunScratch.from_raw(scratch_idx, scratch_idx, 0)
    full_action = action_impl.RunScratch(
        state.StateScratchIndex(scratch_idx),
        state.StateScratchIndex(scratch_idx),
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

    expected_history = full_state.SuccessActionData.from_args(
        raw_action=core.Optional(),
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
        scratches=scratches,
        args_groups=args_groups,
    )
    if current_state != expected_state:
        print('current_state:', env.symbol(current_state))
        print('expected_state:', env.symbol(expected_state))
    assert current_state == expected_state

    assert env.full_state.goal_achieved() is False

    return scratches

def has_goal(env: GoalEnv, goal: meta_env.IGoal):
    selected_goal = env.full_state.nested_arg(
        (full_state.FullState.idx_meta, meta_env.MetaInfo.idx_goal)
    ).apply()
    return selected_goal == goal

def build_case(max_steps: int | None = None):
    """GoalEnv with the control-flow initial state, shared with scripts/run_case.py.

    Returns (env, goal, state_meta, scratches, args_groups); test_control_flow
    uses this same construction so the runner and the test cannot drift.
    """
    goal = node_types.HaveScratch.with_goal(core.Void())
    state_meta = state.StateMetaInfo.with_goal_expr(goal)

    if_scratches: list[core.INode | None] = [
        core.If(
            core.IBoolean.true(),
            core.Integer(1),
            core.IntGroup.from_ints([2, 3]),
        ),
        core.If(
            core.IBoolean.false(),
            core.Integer(1),
            core.IntGroup.from_ints([2, 3]),
        ),
    ]

    loop_scratches: list[core.INode | None] = [
        core.Loop.with_node(
            core.FunctionExpr(
                core.Protocol(
                    core.TypeAliasGroup(),
                    core.CountableTypeGroup(
                        core.CompositeType(
                            core.Optional.as_type(),
                            core.OptionalTypeGroup(
                                core.Integer.as_type(),
                            ),
                        ),
                    ),
                    core.CompositeType(
                        core.LoopGuard.as_type(),
                        core.CountableTypeGroup(
                            core.IntBoolean.as_type(),
                            core.UnionType(
                                core.Integer.as_type(),
                                core.CompositeType(
                                    core.DefaultGroup.as_type(),
                                    core.RestTypeGroup(
                                        core.UnionType(
                                            core.Integer.as_type(),
                                            core.CompositeType(
                                                core.Optional.as_type(),
                                                core.OptionalTypeGroup(
                                                    core.Integer.as_type(),
                                                ),
                                            ),
                                        ),
                                    ),
                                ),
                            ),
                        ),
                    ),
                ),
                core.If(
                    core.IsEmpty(core.Param.from_int(1)),
                    core.LoopGuard.with_args(
                        condition=core.IBoolean.true(),
                        result=core.Integer(9)
                    ),
                    core.LoopGuard.with_args(
                        condition=core.IBoolean.false(),
                        result=core.DefaultGroup(core.Integer(0), core.Param.from_int(1))
                    ),
                )
            ),
        ),

        core.Loop.with_node(
            core.FunctionExpr(
                core.Protocol(
                    core.TypeAliasGroup(
                        core.TypeAlias(
                            core.IOptional.as_type()
                        ),
                        core.TypeAlias(
                            core.IInt.as_type(),
                        ),
                        core.TypeAlias(
                            core.CompositeType(
                                core.TypeIndex(1),
                                core.OptionalTypeGroup(
                                    core.TypeIndex(2),
                                ),
                            ),
                        ),
                    ),
                    core.CountableTypeGroup(
                        core.TypeIndex(3),
                    ),
                    core.CompositeType(
                        core.LoopGuard.as_type(),
                        core.CountableTypeGroup(
                            core.IntBoolean.as_type(),
                            core.UnionType(
                                core.TypeIndex(2),
                                core.CompositeType(
                                    core.DefaultGroup.as_type(),
                                    core.RestTypeGroup(
                                        core.UnionType(
                                            core.TypeIndex(2),
                                            core.TypeIndex(3),
                                        ),
                                    ),
                                ),
                            ),
                        ),
                    ),
                ),
                core.If(
                    core.IsEmpty(core.Param.from_int(1)),
                    core.LoopGuard.with_args(
                        condition=core.IBoolean.true(),
                        result=core.Integer(9)
                    ),
                    core.LoopGuard.with_args(
                        condition=core.IBoolean.false(),
                        result=core.DefaultGroup(core.Integer(0), core.Param.from_int(1))
                    ),
                )
            ),
        ),
    ]

    default_fn_result = core.CompositeType(
        core.DefaultGroup.as_type(),
        core.CountableTypeGroup(
            core.CompositeType(
                core.DefaultGroup.as_type(),
                core.CountableTypeGroup(
                    core.IntBoolean.as_type(),
                    core.IntBoolean.as_type(),
                    core.IntBoolean.as_type(),
                ),
            ),
            core.CompositeType(
                core.DefaultGroup.as_type(),
                core.CountableTypeGroup(
                    core.IntBoolean.as_type(),
                    core.IntBoolean.as_type(),
                    core.IntBoolean.as_type(),
                ),
            ),
        ),
    )

    fn_scratches: list[core.INode | None] = [
        core.FunctionCall(
            core.FunctionExpr(
                core.Protocol(
                    core.TypeAliasGroup(),
                    core.RestTypeGroup(
                        core.CompositeType(
                            core.IntGroup.as_type(),
                            core.CountableTypeGroup(
                                core.Integer.as_type(),
                                core.Integer.as_type(),
                            ),
                        ),
                    ),
                    core.CompositeType(
                        core.DefaultGroup.as_type(),
                        core.RestTypeGroup(
                            core.CompositeType(
                                core.DefaultGroup.as_type(),
                                core.RestTypeGroup(
                                    core.IntBoolean.as_type(),
                                ),
                            ),
                        ),
                    ),
                ),
                core.FunctionCall(
                    core.FunctionWrapper(
                        core.Protocol(
                            core.TypeAliasGroup(),
                            core.RestTypeGroup(
                                core.CompositeType(
                                    core.IntGroup.as_type(),
                                    core.CountableTypeGroup(
                                        core.Integer.as_type(),
                                        core.Integer.as_type(),
                                    ),
                                ),
                            ),
                            core.CompositeType(
                                core.DefaultGroup.as_type(),
                                core.RestTypeGroup(
                                    core.CompositeType(
                                        core.DefaultGroup.as_type(),
                                        core.RestTypeGroup(
                                            core.IntBoolean.as_type(),
                                        ),
                                    ),
                                ),
                            ),
                        ),
                        core.FunctionCall(
                            core.FunctionExpr(
                                core.Protocol(
                                    core.TypeAliasGroup(),
                                    core.RestTypeGroup(
                                        core.CompositeType(
                                            core.IntGroup.as_type(),
                                            core.CountableTypeGroup(
                                                core.Integer.as_type(),
                                                core.Integer.as_type(),
                                            ),
                                        ),
                                    ),
                                    core.CompositeType(
                                        core.DefaultGroup.as_type(),
                                        core.RestTypeGroup(
                                            core.CompositeType(
                                                core.DefaultGroup.as_type(),
                                                core.RestTypeGroup(
                                                    core.IntBoolean.as_type(),
                                                ),
                                            ),
                                        ),
                                    ),
                                ),
                                core.DefaultGroup(
                                    core.DefaultGroup(
                                        core.FunctionCall(
                                            core.TypeNode(core.LessThan),
                                            core.Param.from_int(1),
                                        ),
                                        core.FunctionCall(
                                            core.TypeNode(core.LessThan),
                                            core.Param.from_int(2),
                                        ),
                                        core.FunctionCall(
                                            core.TypeNode(core.LessThan),
                                            core.Param.from_int(3),
                                        ),
                                    ),
                                    core.DefaultGroup(
                                        core.FunctionCall(
                                            core.TypeNode(core.GreaterThan),
                                            core.Param.from_int(1),
                                        ),
                                        core.FunctionCall(
                                            core.TypeNode(core.GreaterThan),
                                            core.Param.from_int(2),
                                        ),
                                        core.FunctionCall(
                                            core.TypeNode(core.GreaterThan),
                                            core.Param.from_int(3),
                                        ),
                                    ),
                                ),
                            ),
                            core.DefaultGroup(
                                core.IntGroup.from_ints([1, 2]),
                                core.Param(
                                    core.FarParentScope.create(),
                                    core.PlaceholderIndex(1),
                                ),
                                core.Param(
                                    core.NearParentScope.create(),
                                    core.PlaceholderIndex(1),
                                ),
                            )
                        ),
                    ),
                    core.DefaultGroup(
                        core.IntGroup.from_ints([2, 1]),
                    )
                ),
            ),
            core.DefaultGroup(
                core.IntGroup.from_ints([2, 2]),
            ),
        ),

        core.FunctionCall(
            core.FunctionWrapper(
                core.Protocol(
                    core.TypeAliasGroup(),
                    core.CountableTypeGroup(core.BaseInt.as_type()),
                    default_fn_result,
                ),
                core.FunctionCall(
                    core.FunctionWrapper(
                        core.Protocol(
                            core.TypeAliasGroup(),
                            core.CountableTypeGroup(core.Integer.as_type()),
                            default_fn_result,
                        ),
                        core.FunctionCall(
                            core.FunctionExpr(
                                core.Protocol(
                                    core.TypeAliasGroup(),
                                    core.CountableTypeGroup(
                                        core.CompositeType(
                                            core.IntGroup.as_type(),
                                            core.CountableTypeGroup(
                                                core.Integer.as_type(),
                                                core.Integer.as_type(),
                                            ),
                                        ),
                                        core.CompositeType(
                                            core.IntGroup.as_type(),
                                            core.CountableTypeGroup(
                                                core.Integer.as_type(),
                                                core.Integer.as_type(),
                                            ),
                                        ),
                                        core.CompositeType(
                                            core.IntGroup.as_type(),
                                            core.CountableTypeGroup(
                                                core.Integer.as_type(),
                                                core.Integer.as_type(),
                                            ),
                                        ),
                                    ),
                                    default_fn_result,
                                ),
                                core.DefaultGroup(
                                    core.DefaultGroup(
                                        core.FunctionCall(
                                            core.TypeNode(core.LessThan),
                                            core.Param.from_int(1),
                                        ),
                                        core.FunctionCall(
                                            core.TypeNode(core.LessThan),
                                            core.Param.from_int(2),
                                        ),
                                        core.FunctionCall(
                                            core.TypeNode(core.LessThan),
                                            core.Param.from_int(3),
                                        ),
                                    ),
                                    core.DefaultGroup(
                                        core.FunctionCall(
                                            core.TypeNode(core.GreaterThan),
                                            core.Param.from_int(1),
                                        ),
                                        core.FunctionCall(
                                            core.TypeNode(core.GreaterThan),
                                            core.Param.from_int(2),
                                        ),
                                        core.FunctionCall(
                                            core.TypeNode(core.GreaterThan),
                                            core.Param.from_int(3),
                                        ),
                                    ),
                                ),
                            ),
                            core.DefaultGroup(
                                core.IntGroup(
                                    core.Param(
                                        core.NearParentScope.create(),
                                        core.PlaceholderIndex(1),
                                    ),
                                    core.Param(
                                        core.FarParentScope.create(),
                                        core.PlaceholderIndex(1),
                                    ),
                                ),
                                core.IntGroup(
                                    core.Param(
                                        core.FarParentScope.create(),
                                        core.PlaceholderIndex(1),
                                    ),
                                    core.Param(
                                        core.FarParentScope.create(),
                                        core.PlaceholderIndex(1),
                                    ),
                                ),
                                core.IntGroup(
                                    core.Param(
                                        core.FarParentScope.create(),
                                        core.PlaceholderIndex(1),
                                    ),
                                    core.Param(
                                        core.NearParentScope.create(),
                                        core.PlaceholderIndex(1),
                                    ),
                                ),
                            )
                        ),
                    ),
                    core.DefaultGroup(
                        core.FunctionCall(
                            core.FunctionExpr(
                                core.Protocol(
                                    core.TypeAliasGroup(),
                                    core.CountableTypeGroup(
                                        core.Integer.as_type(),
                                    ),
                                    core.Integer.as_type(),
                                ),
                                core.Param.from_int(1),
                            ),
                            core.DefaultGroup(
                                core.Integer(1),
                            ),
                        ),
                    )
                ),
            ),
            core.FunctionCall(
                core.FunctionExpr(
                    core.Protocol(
                        core.TypeAliasGroup(),
                        core.RestTypeGroup(
                            core.Integer.as_type(),
                        ),
                        core.CompositeType(
                            core.DefaultGroup.as_type(),
                            core.RestTypeGroup(
                                core.Integer.as_type(),
                            ),
                        ),
                    ),
                    core.DefaultGroup(
                        core.Param.from_int(1),
                    ),
                ),
                core.DefaultGroup(
                    core.Integer(2),
                ),
            ),
        ),

        core.FunctionCall(
            core.FunctionExpr(
                core.Protocol(
                    core.TypeAliasGroup(
                        core.TypeAlias(
                            core.IInt.as_type()
                        ),
                    ),
                    core.CountableTypeGroup(
                        core.TypeIndex(1),
                        core.TypeIndex(1),
                        core.LazyTypeIndex(1),
                        core.TypeIndex(1),
                    ),
                    core.CompositeType(
                        core.DefaultGroup.as_type(),
                        core.CountableTypeGroup(
                            core.TypeIndex(1),
                            core.TypeIndex(1),
                            core.TypeIndex(1),
                            core.LazyTypeIndex(1),
                            core.TypeIndex(1),
                        ),
                    ),
                ),
                core.DefaultGroup(
                    core.Param.from_int(1),
                    core.Param.from_int(2),
                    core.Param.from_int(4),
                    core.Param.from_int(3),
                    core.Param.from_int(1),
                ),
            ),
            core.DefaultGroup(
                core.TypeEnforcer(
                    core.BaseInt.as_type(),
                    core.Integer(2),
                ),
                core.TypeEnforcer(
                    core.BaseInt.as_type(),
                    core.NodeArgIndex(5),
                ),
                core.NodeMainIndex(7),
                core.TypeEnforcer(
                    core.BaseInt.as_type(),
                    core.TypeIndex(9),
                ),
            ),
        ),
    ]

    assignment_scratches = [
        core.InstructionGroup(
            core.Assign(
                core.PlaceholderIndex(1),
                core.DefaultGroup(
                    core.IntGroup.from_ints([1, 2]),
                    core.IntGroup.from_ints([2, 2]),
                    core.IntGroup.from_ints([2, 1]),
                ),
            ),
            core.Return.with_node(
                state.Scratch(
                    core.Var.from_int(1),
                ),
            ),
        ),

        core.InstructionGroup(
            core.Assign(
                core.PlaceholderIndex(1),
                core.FunctionCall(
                    composite.Map,
                    core.DefaultGroup(
                        core.FunctionExpr(
                            core.Protocol(
                                core.TypeAliasGroup(),
                                core.CountableTypeGroup(
                                    core.CompositeType(
                                        core.IntGroup.as_type(),
                                        core.CountableTypeGroup(
                                            core.Integer.as_type(),
                                            core.Integer.as_type(),
                                        ),
                                    ),
                                ),
                                core.CompositeType(
                                    core.DefaultGroup.as_type(),
                                    core.CountableTypeGroup(
                                        core.IntBoolean.as_type(),
                                        core.IntBoolean.as_type(),
                                    ),
                                ),
                            ),
                            core.DefaultGroup(
                                core.FunctionCall(
                                    core.TypeNode(core.LessThan),
                                    core.Param.from_int(1),
                                ),
                                core.FunctionCall(
                                    core.TypeNode(core.GreaterThan),
                                    core.Param.from_int(1),
                                ),
                            ),
                        ),
                        core.DefaultGroup(
                            core.IntGroup.from_ints([1, 2]),
                        ),
                    ),
                ),
            ),
            core.Return.with_node(
                core.Var.from_int(1),
            ),
        ),

        core.InstructionGroup(
            core.Assign(
                core.PlaceholderIndex(1),
                core.FunctionCall(
                    composite.Map,
                    core.DefaultGroup(
                        core.FunctionExpr(
                            core.Protocol(
                                core.TypeAliasGroup(),
                                core.CountableTypeGroup(
                                    core.IntGroup.as_type(),
                                ),
                                core.DefaultGroup.as_type(),
                            ),
                            core.DefaultGroup(
                                core.FunctionCall(
                                    core.TypeNode(core.LessThan),
                                    core.Param.from_int(1),
                                ),
                                core.FunctionCall(
                                    core.TypeNode(core.GreaterThan),
                                    core.Param.from_int(1),
                                ),
                            ),
                        ),
                        core.DefaultGroup(
                            core.IntGroup.from_ints([1, 2]),
                            core.IntGroup.from_ints([2, 2]),
                            core.IntGroup.from_ints([2, 1]),
                        ),
                    ),
                ),
            ),
            core.Return.with_node(
                core.Var.from_int(1),
            ),
        ),

        core.InstructionGroup(
            core.Assign(
                core.PlaceholderIndex(1),
                core.FunctionWrapper(
                    core.Protocol(
                        core.TypeAliasGroup(),
                        core.CountableTypeGroup(
                            core.IFunction.as_type(),
                            core.DefaultGroup.as_type(),
                        ),
                        core.INode.as_type(),
                    ),
                    core.FunctionCall(
                        composite.Map,
                        core.DefaultGroup(
                            core.FunctionWrapper(
                                core.Protocol(
                                    core.TypeAliasGroup(),
                                    core.CountableTypeGroup(core.DefaultGroup.as_type()),
                                    core.INode.as_type(),
                                ),
                                core.FunctionCall(
                                    core.Param(
                                        core.NearParentScope.from_int(2),
                                        core.PlaceholderIndex(1),
                                    ),
                                    core.Param(
                                        core.NearParentScope.from_int(1),
                                        core.PlaceholderIndex(1),
                                    ),
                                ),
                            ),
                            core.Param(
                                core.NearParentScope.from_int(1),
                                core.PlaceholderIndex(2),
                            ),
                        ),
                    ),
                ),
            ),
            core.Assign(
                core.PlaceholderIndex(2),
                core.FunctionWrapper(
                    core.Protocol(
                        core.TypeAliasGroup(),
                        core.CountableTypeGroup(
                            core.DefaultGroup.as_type(),
                            core.IFunction.as_type(),
                        ),
                        core.INode.as_type(),
                    ),
                    core.FunctionCall(
                        composite.Map,
                        core.DefaultGroup(
                            core.FunctionWrapper(
                                core.Protocol(
                                    core.TypeAliasGroup(),
                                    core.CountableTypeGroup(core.DefaultGroup.as_type()),
                                    core.INode.as_type(),
                                ),
                                core.FunctionCall(
                                    core.Var(
                                        core.NearParentScope.from_int(3),
                                        core.PlaceholderIndex(1),
                                    ),
                                    core.DefaultGroup(
                                        core.Param(
                                            core.NearParentScope.from_int(1),
                                            core.PlaceholderIndex(1),
                                        ),
                                        core.Param(
                                            core.NearParentScope.from_int(2),
                                            core.PlaceholderIndex(2),
                                        ),
                                    ),
                                ),
                            ),
                            core.Param(
                                core.NearParentScope.from_int(1),
                                core.PlaceholderIndex(1),
                            ),
                        ),
                    ),
                ),
            ),
            core.Return.with_node(
                core.Var.from_int(2)
            ),
        ),

        core.InstructionGroup(
            core.Assign(
                core.PlaceholderIndex(1),
                core.DefaultGroup(
                    core.IntGroup.from_ints([1, 2]),
                    core.IntGroup.from_ints([2, 2]),
                    core.IntGroup.from_ints([2, 1]),
                ),
            ),
            core.Assign(
                core.PlaceholderIndex(2),
                core.DefaultGroup(
                    core.LessThan.as_type(),
                    core.GreaterThan.as_type(),
                ),
            ),
            core.Assign(
                core.PlaceholderIndex(3),
                core.FunctionWrapper(
                    core.Protocol(
                        core.TypeAliasGroup(),
                        core.CountableTypeGroup(
                            core.FunctionType(
                                core.CountableTypeGroup(
                                    core.IComparable.as_type(),
                                    core.IComparable.as_type(),
                                ),
                                core.IntBoolean.as_type(),
                            ),
                            core.CompositeType(
                                core.DefaultGroup.as_type(),
                                core.RestTypeGroup(
                                    core.CompositeType(
                                        core.IntGroup.as_type(),
                                        core.CountableTypeGroup(
                                            core.Integer.as_type(),
                                            core.Integer.as_type(),
                                        ),
                                    ),
                                ),
                            ),
                        ),
                        core.CompositeType(
                            core.DefaultGroup.as_type(),
                            core.RestTypeGroup(core.IntBoolean.as_type()),
                        ),
                    ),
                    core.FunctionCall(
                        composite.Map,
                        core.DefaultGroup(
                            core.FunctionWrapper(
                                core.Protocol(
                                    core.TypeAliasGroup(),
                                    core.CountableTypeGroup(
                                        core.IntGroup.as_type(),
                                    ),
                                    core.IntBoolean.as_type(),
                                ),
                                core.FunctionCall(
                                    core.Param(
                                        core.NearParentScope.from_int(2),
                                        core.PlaceholderIndex(1),
                                    ),
                                    core.Param(
                                        core.NearParentScope.from_int(1),
                                        core.PlaceholderIndex(1),
                                    ),
                                ),
                            ),
                            core.Param(
                                core.NearParentScope.from_int(1),
                                core.PlaceholderIndex(2),
                            ),
                        ),
                    ),
                ),
            ),
            core.Assign(
                core.PlaceholderIndex(4),
                core.FunctionWrapper(
                    core.Protocol(
                        core.TypeAliasGroup(),
                        core.CountableTypeGroup(
                            core.CompositeType(
                                core.DefaultGroup.as_type(),
                                core.RestTypeGroup(
                                    core.FunctionType(
                                        core.CountableTypeGroup(
                                            core.IComparable.as_type(),
                                            core.IComparable.as_type(),
                                        ),
                                        core.IntBoolean.as_type(),
                                    ),
                                ),
                            ),
                            core.CompositeType(
                                core.DefaultGroup.as_type(),
                                core.RestTypeGroup(
                                    core.CompositeType(
                                        core.IntGroup.as_type(),
                                        core.CountableTypeGroup(
                                            core.Integer.as_type(),
                                            core.Integer.as_type(),
                                        ),
                                    ),
                                ),
                            ),
                        ),
                        default_fn_result,
                    ),
                    core.FunctionCall(
                        composite.Map,
                        core.DefaultGroup(
                            core.FunctionWrapper(
                                core.Protocol(
                                    core.TypeAliasGroup(),
                                    core.CountableTypeGroup(
                                        core.Type(core.RunnableBoolean.as_type()),
                                    ),
                                    core.CompositeType(
                                        core.DefaultGroup.as_type(),
                                        core.RestTypeGroup(
                                            core.IBoolean.as_type(),
                                        ),
                                    ),
                                ),
                                core.FunctionCall(
                                    core.Var(
                                        core.NearParentScope.from_int(3),
                                        core.PlaceholderIndex(3),
                                    ),
                                    core.DefaultGroup(
                                        core.Param(
                                            core.NearParentScope.from_int(1),
                                            core.PlaceholderIndex(1),
                                        ),
                                        core.Param(
                                            core.NearParentScope.from_int(2),
                                            core.PlaceholderIndex(2),
                                        ),
                                    ),
                                ),
                            ),
                            core.Param(
                                core.NearParentScope.from_int(1),
                                core.PlaceholderIndex(1),
                            ),
                        ),
                    ),
                ),
            ),
            core.Assign(
                core.PlaceholderIndex(5),
                core.FunctionCall(
                    core.Var.from_int(4),
                    core.DefaultGroup(
                        core.Var.from_int(2),
                        core.Var.from_int(1),
                    ),
                ),
            ),
            core.Return.with_node(
                core.Var.from_int(5)
            ),
        ),

        core.InstructionGroup(
            core.Assign(
                core.PlaceholderIndex(1),
                core.FunctionExpr(
                    core.Protocol(
                        core.TypeAliasGroup(
                            core.TypeAlias(core.IInt.as_type()),
                            core.TypeAlias(core.IBoolean.as_type()),
                        ),
                        core.CountableTypeGroup(
                            core.FunctionType(
                                core.CountableTypeGroup(
                                    core.TypeIndex(1),
                                    core.LazyTypeIndex(1),
                                ),
                                core.TypeIndex(2),
                            ),
                            core.LazyTypeIndex(1),
                            core.LazyTypeIndex(1),
                        ),
                        core.LazyTypeIndex(2),
                    ),
                    core.FunctionCall(
                        core.Param.from_int(1),
                        core.DefaultGroup(
                            core.Param.from_int(2),
                            core.Param.from_int(3),
                        ),
                    ),
                ),
            ),
            core.Assign(
                core.PlaceholderIndex(2),
                core.FunctionCall(
                    core.Var.from_int(1),
                    core.DefaultGroup(
                        core.GreaterThan.as_type(),
                        core.Integer(1),
                        core.Integer(0),
                    ),
                ),
            ),
            core.Assign(
                core.PlaceholderIndex(3),
                core.FunctionCall(
                    core.Var.from_int(1),
                    core.DefaultGroup(
                        core.GreaterThan.as_type(),
                        core.Integer(0),
                        core.Integer(1),
                    ),
                ),
            ),
            core.Assign(
                core.PlaceholderIndex(4),
                core.FunctionCall(
                    core.Var.from_int(1),
                    core.DefaultGroup(
                        core.LessThan.as_type(),
                        core.Integer(0),
                        core.Integer(1),
                    ),
                ),
            ),
            core.Assign(
                core.PlaceholderIndex(5),
                core.FunctionCall(
                    core.Var.from_int(1),
                    core.DefaultGroup(
                        core.FunctionExpr(
                            core.Protocol(
                                core.TypeAliasGroup(),
                                core.CountableTypeGroup(
                                    core.Integer.as_type(),
                                    core.UnionType(
                                        core.BaseInt.as_type(),
                                        core.Integer.as_type(),
                                    ),
                                ),
                                core.IntBoolean.as_type(),
                            ),
                            core.LessThan(
                                core.Param.from_int(1),
                                core.Param.from_int(2),
                            ),
                        ),
                        core.Integer(1),
                        core.Integer(0),
                    ),
                ),
            ),
            core.Assign(
                core.PlaceholderIndex(6),
                core.FunctionCall(
                    core.Var.from_int(1),
                    core.DefaultGroup(
                        core.FunctionExpr(
                            core.Protocol(
                                core.TypeAliasGroup(),
                                core.CountableTypeGroup(
                                    core.IntersectionType(
                                        core.BaseInt.as_type(),
                                        core.Integer.as_type(),
                                    ),
                                    core.Integer.as_type(),
                                ),
                                core.IntBoolean.as_type(),
                            ),
                            core.GreaterThan(
                                core.Param.from_int(1),
                                core.Param.from_int(2),
                            ),
                        ),
                        core.Integer(1),
                        core.Integer(0),
                    ),
                ),
            ),
            core.Assign(
                core.PlaceholderIndex(7),
                core.FunctionCall(
                    core.Var.from_int(1),
                    core.DefaultGroup(
                        core.FunctionExpr(
                            core.Protocol(
                                core.TypeAliasGroup(),
                                core.CountableTypeGroup(
                                    core.IntersectionType(
                                        core.BaseInt.as_type(),
                                        core.Integer.as_type(),
                                    ),
                                    core.UnionType(
                                        core.BaseInt.as_type(),
                                        core.Integer.as_type(),
                                    ),
                                ),
                                core.IntBoolean.as_type(),
                            ),
                            core.GreaterThan(
                                core.Param.from_int(1),
                                core.Param.from_int(2),
                            ),
                        ),
                        core.Integer(0),
                        core.Integer(1),
                    ),
                ),
            ),
            core.Return.with_node(
                core.DefaultGroup(
                    core.Var.from_int(2),
                    core.Var.from_int(3),
                    core.Var.from_int(4),
                    core.Var.from_int(5),
                    core.Var.from_int(7),
                    core.Var.from_int(6),
                ),
            ),
        ),
    ]
    scratches = if_scratches + loop_scratches + fn_scratches + assignment_scratches

    args_groups = [state.PartialArgsGroup.create()]
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
    return (env, goal, state_meta, scratches, args_groups,
            if_scratches, loop_scratches, fn_scratches)

def test_control_flow() -> list[full_state.FullState]:
    (env, goal, state_meta, scratches, args_groups,
     if_scratches, loop_scratches, fn_scratches) = build_case()
    assert has_goal(env=env, goal=goal)

    def test_if():
        nonlocal scratches
        index = 0
        scratches = run(
            env=env,
            state_meta=state_meta,
            scratches=scratches,
            args_groups=args_groups,
            scratch_idx=index+1,
            new_scratch=core.Integer(1),
        )
        scratches = run(
            env=env,
            state_meta=state_meta,
            scratches=scratches,
            args_groups=args_groups,
            scratch_idx=index+2,
            new_scratch=core.IntGroup.from_ints([2, 3]),
        )
    test_utils.run_test('>>test_if', test_if)

    def test_loop():
        nonlocal scratches
        index = len(if_scratches)
        scratches = run(
            env=env,
            state_meta=state_meta,
            scratches=scratches,
            args_groups=args_groups,
            scratch_idx=index+1,
            new_scratch=core.Optional(
                core.DefaultGroup(core.Integer(0), core.Optional(core.Integer(9)))
            ),
        )
        scratches = run(
            env=env,
            state_meta=state_meta,
            scratches=scratches,
            args_groups=args_groups,
            scratch_idx=index+2,
            new_scratch=core.Optional(
                core.DefaultGroup(core.Integer(0), core.Optional(core.Integer(9)))
            ),
        )
    test_utils.run_test('>>test_loop', test_loop)

    default_result = core.DefaultGroup(
        core.DefaultGroup(
            core.IBoolean.true(),
            core.IBoolean.false(),
            core.IBoolean.false(),
        ),
        core.DefaultGroup(
            core.IBoolean.false(),
            core.IBoolean.false(),
            core.IBoolean.true(),
        ),
    )

    def test_fn():
        nonlocal scratches
        index = len(if_scratches + loop_scratches)
        scratches = run(
            env=env,
            state_meta=state_meta,
            scratches=scratches,
            args_groups=args_groups,
            scratch_idx=index+1,
            new_scratch=default_result,
        )
        scratches = run(
            env=env,
            state_meta=state_meta,
            scratches=scratches,
            args_groups=args_groups,
            scratch_idx=index+2,
            new_scratch=default_result,
        )
        scratches = run(
            env=env,
            state_meta=state_meta,
            scratches=scratches,
            args_groups=args_groups,
            scratch_idx=index+3,
            new_scratch=core.DefaultGroup(
                core.TypeEnforcer(
                    core.BaseInt.as_type(),
                    core.Integer(2),
                ),
                core.TypeEnforcer(
                    core.BaseInt.as_type(),
                    core.NodeArgIndex(5),
                ),
                core.TypeEnforcer(
                    core.BaseInt.as_type(),
                    core.TypeIndex(9),
                ),
                core.NodeMainIndex(7),
                core.TypeEnforcer(
                    core.BaseInt.as_type(),
                    core.Integer(2),
                ),
            ),
        )
    test_utils.run_test('>>test_fn', test_fn)

    def test_assignments():
        nonlocal scratches
        index = len(if_scratches + loop_scratches + fn_scratches)
        scratches = test_utils.run_test(
            name='>>>test_assignments-1',
            fn=lambda: run(
                env=env,
                state_meta=state_meta,
                scratches=scratches,
                args_groups=args_groups,
                scratch_idx=index+1,
                new_scratch=state.Scratch(core.DefaultGroup(
                    core.IntGroup.from_ints([1, 2]),
                    core.IntGroup.from_ints([2, 2]),
                    core.IntGroup.from_ints([2, 1]),
                )),
            ),
        )

        scratches = test_utils.run_test(
            name='>>>test_assignments-2',
            fn=lambda: run(
                env=env,
                state_meta=state_meta,
                scratches=scratches,
                args_groups=args_groups,
                scratch_idx=index+2,
                new_scratch=core.DefaultGroup(
                    core.DefaultGroup(
                        core.IBoolean.true(),
                        core.IBoolean.false(),
                    ),
                ),
            ),
        )

        scratches = test_utils.run_test(
            name='>>>test_assignments-3',
            fn=lambda: run(
                env=env,
                state_meta=state_meta,
                scratches=scratches,
                args_groups=args_groups,
                scratch_idx=index+3,
                new_scratch=core.DefaultGroup(
                    core.DefaultGroup(
                        core.IBoolean.true(),
                        core.IBoolean.false(),
                    ),
                    core.DefaultGroup(
                        core.IBoolean.false(),
                        core.IBoolean.false(),
                    ),
                    core.DefaultGroup(
                        core.IBoolean.false(),
                        core.IBoolean.true(),
                    ),
                ),
            ),
        )

        scratches = test_utils.run_test(
            name='>>>test_assignments-4',
            fn=lambda: run(
                env=env,
                state_meta=state_meta,
                scratches=scratches,
                args_groups=args_groups,
                scratch_idx=index+4,
                new_scratch=core.FunctionWrapper(
                    core.Protocol(
                        core.TypeAliasGroup(),
                        core.CountableTypeGroup(
                            core.DefaultGroup.as_type(),
                            core.IFunction.as_type(),
                        ),
                        core.INode.as_type(),
                    ),
                    core.FunctionCall(
                        composite.Map,
                        core.DefaultGroup(
                            core.FunctionWrapper(
                                core.Protocol(
                                    core.TypeAliasGroup(),
                                    core.CountableTypeGroup(core.DefaultGroup.as_type()),
                                    core.INode.as_type(),
                                ),
                                core.FunctionCall(
                                    core.FunctionWrapper(
                                        core.Protocol(
                                            core.TypeAliasGroup(),
                                            core.CountableTypeGroup(
                                                core.IFunction.as_type(),
                                                core.DefaultGroup.as_type(),
                                            ),
                                            core.INode.as_type(),
                                        ),
                                        core.FunctionCall(
                                            composite.Map,
                                            core.DefaultGroup(
                                                core.FunctionWrapper(
                                                    core.Protocol(
                                                        core.TypeAliasGroup(),
                                                        core.CountableTypeGroup(
                                                            core.DefaultGroup.as_type()
                                                        ),
                                                        core.INode.as_type(),
                                                    ),
                                                    core.FunctionCall(
                                                        core.Param(
                                                            core.NearParentScope.from_int(2),
                                                            core.PlaceholderIndex(1),
                                                        ),
                                                        core.Param(
                                                            core.NearParentScope.from_int(1),
                                                            core.PlaceholderIndex(1),
                                                        ),
                                                    ),
                                                ),
                                                core.Param(
                                                    core.NearParentScope.from_int(1),
                                                    core.PlaceholderIndex(2),
                                                ),
                                            ),
                                        ),
                                    ),
                                    core.DefaultGroup(
                                        core.Param(
                                            core.NearParentScope.from_int(1),
                                            core.PlaceholderIndex(1),
                                        ),
                                        core.Param(
                                            core.NearParentScope.from_int(2),
                                            core.PlaceholderIndex(2),
                                        ),
                                    ),
                                ),
                            ),
                            core.Param(
                                core.NearParentScope.from_int(1),
                                core.PlaceholderIndex(1),
                            ),
                        ),
                    ),
                ),
            ),
        )

        scratches = test_utils.run_test(
            name='>>>test_assignments-5',
            fn=lambda: run(
                env=env,
                state_meta=state_meta,
                scratches=scratches,
                args_groups=args_groups,
                scratch_idx=index+5,
                new_scratch=default_result,
            ),
        )

        scratches = test_utils.run_test(
            name='>>>test_assignments-6',
            fn=lambda: run(
                env=env,
                state_meta=state_meta,
                scratches=scratches,
                args_groups=args_groups,
                scratch_idx=index+6,
                new_scratch=core.DefaultGroup(
                    core.IBoolean.true(),
                    core.IBoolean.false(),
                    core.IBoolean.true(),
                    core.IBoolean.false(),
                    core.IBoolean.false(),
                    core.IBoolean.true(),
                ),
            ),
        )
    test_utils.run_test('>>test_assignments', test_assignments)

    return [env.full_state]

def test() -> list[full_state.FullState]:
    final_states: list[full_state.FullState] = []
    final_states += test_control_flow()
    return final_states

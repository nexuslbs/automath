import typing
from abc import ABC
from env.core import (
    INode,
    IsInstance,
    GreaterThan,
    LessOrEqual,
    IOpaqueScope,
    InheritableNode,
    IDefault,
    BaseGroup,
    IFromInt,
    IInt,
    Integer,
    INodeIndex,
    NodeMainBaseIndex,
    NodeArgBaseIndex,
    IFromSingleNode,
    IGroup,
    Not,
    IFunction,
    IBoolean,
    TypeNode,
    IOptional,
    Optional,
    Protocol,
    CountableTypeGroup,
    IExceptionInfo,
    IWrapper,
    CompositeType,
    OptionalTypeGroup,
    TmpInnerArg,
    TmpNestedArg,
    IInstantiable,
)
from env.state import (
    State,
    Scratch,
    PartialArgsGroup,
    IGoalAchieved,
    IGoal,
    StateDefinition,
)
from env.meta_env import (
    MetaInfo,
    IFullState,
    IFullStateIndex,
    FullStateIntBaseIndex,
    IAction,
    IBasicAction,
    IRawAction,
    IActionOutput,
    SubtypeOuterGroup,
    DetailedType,
    GeneralTypeGroup,
    MetaInfoOptions,
    MetaData,
    CostMultiplier,
    NewCostMultiplier,
)

T = typing.TypeVar('T', bound=INode)

###########################################################
####################### ACTION DATA #######################
###########################################################

class BaseActionData(InheritableNode, ABC):

    idx_raw_action = 1
    idx_action = 2
    idx_output = 3
    idx_exception = 4

    @classmethod
    def protocol(cls) -> Protocol:
        return cls.default_protocol(CountableTypeGroup(
            CompositeType(
                Optional.as_type(),
                OptionalTypeGroup(IRawAction.as_type()),
            ),
            CompositeType(
                Optional.as_type(),
                OptionalTypeGroup(IAction.as_type()),
            ),
            CompositeType(
                Optional.as_type(),
                OptionalTypeGroup(IActionOutput.as_type()),
            ),
            CompositeType(
                Optional.as_type(),
                OptionalTypeGroup(IExceptionInfo.as_type()),
            ),
        ))

    @property
    def raw_action(self) -> TmpInnerArg:
        return self.inner_arg(self.idx_raw_action)

    @property
    def action(self) -> TmpInnerArg:
        return self.inner_arg(self.idx_action)

    @property
    def output(self) -> TmpInnerArg:
        return self.inner_arg(self.idx_output)

    @property
    def exception(self) -> TmpInnerArg:
        return self.inner_arg(self.idx_exception)

    @classmethod
    def from_args(
        cls,
        raw_action: Optional[IRawAction],
        action: Optional[IAction],
        output: Optional[IActionOutput],
        exception: Optional[IExceptionInfo],
    ) -> typing.Self:
        return cls(raw_action, action, output, exception)

    def _strict_validate(self):
        alias_info = self._thin_strict_validate()

        raw_action = self.raw_action.apply().real(Optional[IRawAction]).value
        action = self.action.apply().real(Optional[IAction]).value
        output = self.output.apply().real(Optional[IActionOutput]).value
        exception = self.exception.apply().real(Optional[IExceptionInfo]).value

        if raw_action is not None:
            raw_action_typed = raw_action.real(IRawAction).as_node
            if action is not None:
                raw_action_typed.strict_validate()
            else:
                raw_action_typed.validate()

        if action is not None:
            action_typed = action.real(IAction).as_node
            if output is not None:
                action_typed.strict_validate()
            else:
                action_typed.validate()

        if output is not None:
            output_typed = output.real(IActionOutput).as_node
            if exception is not None:
                output_typed.strict_validate()
            else:
                output_typed.validate()

        if exception is not None:
            exception.real(IExceptionInfo).as_node.validate()

        return alias_info

    def is_error(self) -> IBoolean:
        exception_opt = self.exception.apply().real(Optional[IExceptionInfo])
        return Not(exception_opt.is_empty())

class SuccessActionData(BaseActionData, IInstantiable):

    @classmethod
    def protocol(cls) -> Protocol:
        return cls.default_protocol(CountableTypeGroup(
            CompositeType(
                Optional.as_type(),
                OptionalTypeGroup(IRawAction.as_type()),
            ),
            CompositeType(
                Optional.as_type(),
                CountableTypeGroup(IAction.as_type()),
            ),
            CompositeType(
                Optional.as_type(),
                CountableTypeGroup(IActionOutput.as_type()),
            ),
            CompositeType(
                Optional.as_type(),
                CountableTypeGroup(),
            ),
        ))

class BeforeActionErrorActionData(BaseActionData, IInstantiable):

    @classmethod
    def protocol(cls) -> Protocol:
        return cls.default_protocol(CountableTypeGroup(
            CompositeType(
                Optional.as_type(),
                CountableTypeGroup(),
            ),
            CompositeType(
                Optional.as_type(),
                CountableTypeGroup(),
            ),
            CompositeType(
                Optional.as_type(),
                CountableTypeGroup(),
            ),
            CompositeType(
                Optional.as_type(),
                CountableTypeGroup(IExceptionInfo.as_type()),
            ),
        ))

class RawActionErrorActionData(BaseActionData, IInstantiable):

    @classmethod
    def protocol(cls) -> Protocol:
        return cls.default_protocol(CountableTypeGroup(
            CompositeType(
                Optional.as_type(),
                CountableTypeGroup(IRawAction.as_type()),
            ),
            CompositeType(
                Optional.as_type(),
                CountableTypeGroup(),
            ),
            CompositeType(
                Optional.as_type(),
                CountableTypeGroup(),
            ),
            CompositeType(
                Optional.as_type(),
                CountableTypeGroup(IExceptionInfo.as_type()),
            ),
        ))

class ActionTypeErrorActionData(BaseActionData, IInstantiable):

    @classmethod
    def protocol(cls) -> Protocol:
        return cls.default_protocol(CountableTypeGroup(
            CompositeType(
                Optional.as_type(),
                OptionalTypeGroup(IRawAction.as_type()),
            ),
            CompositeType(
                Optional.as_type(),
                CountableTypeGroup(IAction.as_type()),
            ),
            CompositeType(
                Optional.as_type(),
                CountableTypeGroup(),
            ),
            CompositeType(
                Optional.as_type(),
                CountableTypeGroup(IExceptionInfo.as_type()),
            ),
        ))

class ActionErrorActionData(BaseActionData, IInstantiable):

    @classmethod
    def protocol(cls) -> Protocol:
        return cls.default_protocol(CountableTypeGroup(
            CompositeType(
                Optional.as_type(),
                OptionalTypeGroup(IRawAction.as_type()),
            ),
            CompositeType(
                Optional.as_type(),
                CountableTypeGroup(IAction.as_type()),
            ),
            CompositeType(
                Optional.as_type(),
                CountableTypeGroup(),
            ),
            CompositeType(
                Optional.as_type(),
                CountableTypeGroup(IExceptionInfo.as_type()),
            ),
        ))

class ActionOutputErrorActionData(BaseActionData, IInstantiable):

    @classmethod
    def protocol(cls) -> Protocol:
        return cls.default_protocol(CountableTypeGroup(
            CompositeType(
                Optional.as_type(),
                OptionalTypeGroup(IRawAction.as_type()),
            ),
            CompositeType(
                Optional.as_type(),
                CountableTypeGroup(IAction.as_type()),
            ),
            CompositeType(
                Optional.as_type(),
                CountableTypeGroup(IActionOutput.as_type()),
            ),
            CompositeType(
                Optional.as_type(),
                CountableTypeGroup(IExceptionInfo.as_type()),
            ),
        ))

###########################################################
################# FULL STATE DEFINITIONS ##################
###########################################################

class HistoryNode(InheritableNode, IDefault, IWrapper, IInstantiable):

    idx_state = 1
    idx_meta_data = 2
    idx_action_data = 3

    @classmethod
    def protocol(cls) -> Protocol:
        return cls.default_protocol(CountableTypeGroup(
            State.as_type(),
            MetaData.as_type(),
            CompositeType(
                Optional.as_type(),
                OptionalTypeGroup(BaseActionData.as_type()),
            ),
        ))

    @classmethod
    def create(cls):
        return cls(State.create(), Optional.create(), Optional.create())

    @classmethod
    def create_with_goal_and_options(
        cls,
        goal: IGoal,
        remaining_steps: int | None = None,
    ) -> typing.Self:
        goal_achieved = IGoalAchieved.from_goal_expr(goal)
        return cls(
            State.create_with_goal(goal_achieved),
            MetaData.with_args(remaining_steps=remaining_steps),
            Optional.create(),
        )

    @classmethod
    def with_args(
        cls,
        state: State,
        meta_data: MetaData,
        action_data: Optional[BaseActionData] | None = None,
    ):
        action_data = action_data if action_data is not None else Optional.create()
        return cls(state, meta_data, action_data)

    def with_new_args(
        self,
        state: State | None = None,
        meta_data: MetaData | None = None,
        action_data: Optional[BaseActionData] | None = None,
    ) -> typing.Self:
        state = (
            state
            if state is not None
            else self.state.apply().real(State))
        meta_data = (
            meta_data
            if meta_data is not None
            else self.meta_data.apply().real(MetaData))
        action_data = (
            action_data
            if action_data is not None
            else self.action_data.apply().real(Optional[BaseActionData]))
        return self.with_args(
            state=state,
            meta_data=meta_data,
            action_data=action_data)

    @property
    def state(self) -> TmpInnerArg:
        return self.inner_arg(self.idx_state)

    @property
    def meta_data(self) -> TmpInnerArg:
        return self.inner_arg(self.idx_meta_data)

    @property
    def action_data(self) -> TmpInnerArg:
        return self.inner_arg(self.idx_action_data)

    def minimal_meta(self) -> typing.Self:
        meta_data = self.meta_data.apply().real(MetaData)
        remaining_steps_value = meta_data.remaining_steps.apply().real(
            Optional[Integer]
        ).value
        remaining_steps = (
            remaining_steps_value.as_int
            if remaining_steps_value is not None
            else None)
        return self.with_new_args(
            meta_data=MetaData.with_args(
                remaining_steps=remaining_steps,
            )
        )

    def before_run_stats(self) -> typing.Self:
        meta_data = self.meta_data.apply().real(MetaData)
        remaining_steps_value = meta_data.remaining_steps.apply().real(
            Optional[Integer]
        ).value
        remaining_steps = (
            remaining_steps_value.as_int
            if remaining_steps_value is not None
            else None)
        new_cost_multiplier = meta_data.new_cost_multiplier.apply().real(
            Optional[NewCostMultiplier]
        ).value
        cost_multiplier = meta_data.cost_multiplier.apply().real(
            Optional[CostMultiplier]
        ).value
        return self.with_new_args(
            meta_data=MetaData.with_args(
                remaining_steps=remaining_steps,
                new_cost_multiplier=new_cost_multiplier,
                cost_multiplier=cost_multiplier,
                run_cost=None,
                final_cost=None,
            )
        )

class HistorySummaryActionData(BaseActionData, IInstantiable):
    """Marker payload of a bounded-history summary entry.

    A real, observable node type (not a bare int) stored in the action_data
    slot of a HistoryNode: it records how many entries the summary replaced
    and, when cheap, their summed final cost.
    """

    idx_dropped_count = 1
    idx_dropped_cost = 2

    @classmethod
    def protocol(cls) -> Protocol:
        return cls.default_protocol(CountableTypeGroup(
            Integer.as_type(),
            Integer.as_type(),
        ))

    @classmethod
    def create(cls, dropped_count: int = 0, dropped_cost: int = 0) -> typing.Self:
        return cls(Integer(int(dropped_count)), Integer(int(dropped_cost)))

    @property
    def dropped_count(self) -> TmpInnerArg:
        return self.inner_arg(self.idx_dropped_count)

    @property
    def dropped_cost(self) -> TmpInnerArg:
        return self.inner_arg(self.idx_dropped_cost)


def history_summary_data(node: HistoryNode) -> HistorySummaryActionData | None:
    action_data = node.action_data.apply().real(Optional[BaseActionData]).value
    if isinstance(action_data, HistorySummaryActionData):
        return action_data
    return None


def make_history_summary(dropped_count: int, dropped_cost: int = 0) -> HistoryNode:
    return HistoryNode.with_args(
        state=State.create(),
        meta_data=MetaData.create(),
        action_data=Optional(
            HistorySummaryActionData.create(dropped_count, dropped_cost)
        ),
    )


class HistoryGroupNode(BaseGroup[HistoryNode], IInstantiable):

    @classmethod
    def item_type(cls) -> TypeNode:
        return HistoryNode.as_type()

###########################################################
####################### FULL STATE ########################
###########################################################

class FullState(
    InheritableNode,
    IFullState,
    IOpaqueScope,
    IFromSingleNode[MetaInfo],
    IWrapper,
    IInstantiable,
):

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

    @classmethod
    def with_node(cls, node: MetaInfo) -> typing.Self:
        return cls.with_args(meta=node)

    @classmethod
    def with_args(
        cls,
        meta: MetaInfo,
        current: HistoryNode | None = None,
        history: HistoryGroupNode | None = None,
    ) -> typing.Self:
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

    def with_new_args(
        self,
        meta: MetaInfo | None = None,
        current: HistoryNode | None = None,
        history: HistoryGroupNode | None = None,
    ) -> typing.Self:
        meta = (
            meta
            if meta is not None
            else self.meta.apply().real(MetaInfo))
        current = (
            current
            if current is not None
            else self.current.apply().real(HistoryNode))
        history = (
            history
            if history is not None
            else self.history.apply().real(HistoryGroupNode))
        return self.with_args(
            meta=meta,
            current=current,
            history=history)

    @property
    def meta(self) -> TmpInnerArg:
        return self.inner_arg(self.idx_meta)

    @property
    def current(self) -> TmpInnerArg:
        return self.inner_arg(self.idx_current)

    @property
    def history(self) -> TmpInnerArg:
        return self.inner_arg(self.idx_history)

    @property
    def current_state(self) -> TmpNestedArg:
        return self.nested_arg(
            (self.idx_current, HistoryNode.idx_state)
        )

    @property
    def meta_info_options(self) -> MetaInfoOptions:
        meta = self.meta.apply().real(MetaInfo)
        options = meta.options.apply().real(MetaInfoOptions)
        return options

    @property
    def cost_multiplier(self) -> Optional[CostMultiplier]:
        current = self.current.apply().real(HistoryNode)
        IsInstance.assert_type(current, HistoryNode)
        assert isinstance(current, HistoryNode)
        meta_data = current.meta_data.apply().real(MetaData)
        cost_multiplier = meta_data.cost_multiplier.apply().real(Optional[CostMultiplier])
        return cost_multiplier

    @property
    def last_action_data(self) -> IOptional[BaseActionData]:
        history_group = self.history.apply().real(HistoryGroupNode)
        history = history_group.as_tuple
        if len(history) == 0:
            return Optional.create()
        last_item = history_group.as_tuple[-1]
        IsInstance.assert_type(last_item, HistoryNode)
        assert isinstance(last_item, HistoryNode)
        action_data_opt = last_item.action_data.apply().real(IOptional[BaseActionData])
        return action_data_opt

    def final_cost(self) -> Integer:
        current = self.current.apply().real(HistoryNode)
        IsInstance.assert_type(current, HistoryNode)
        assert isinstance(current, HistoryNode)
        meta_data = current.meta_data.apply().real(MetaData)
        final_cost_opt = meta_data.final_cost.apply().real(Optional[Integer])
        return final_cost_opt.value_or_raise

    def goal_achieved(self) -> bool:
        state = self.current_state.apply().real(State)
        return state.goal_achieved()

    def node_types(self) -> tuple[type[INode], ...]:
        meta = self.meta.apply().real(MetaInfo)
        all_types_wrappers = meta.all_types.apply().real(GeneralTypeGroup).as_tuple
        all_types = tuple(wrapper.type for wrapper in all_types_wrappers)
        return all_types

    def history_amount(self) -> int:
        return len(self.history.apply().real(HistoryGroupNode).as_tuple)

    def minimal_meta(self) -> typing.Self:
        current = self.current.apply().real(HistoryNode).minimal_meta()
        history_items = self.history.apply().real(HistoryGroupNode).as_tuple
        new_history = HistoryGroupNode.from_items(
            [item.minimal_meta() for item in history_items]
        )
        return self.with_new_args(
            current=current,
            history=new_history,
        )

    def before_run_stats(self) -> typing.Self:
        current = self.current.apply().real(HistoryNode).before_run_stats()
        return self.with_new_args(current=current)

    def at_history(self, index: int) -> tuple[typing.Self, IOptional[BaseActionData]]:
        history = self.history.apply().real(HistoryGroupNode).as_tuple
        GreaterThan.with_ints(index, 0).raise_on_false()
        assert index > 0
        LessOrEqual.with_ints(index, len(history)).raise_on_false()
        assert index <= len(history)
        item = history[index-1]
        current = HistoryNode.with_args(
            state=item.state.apply().real(State),
            meta_data=item.meta_data.apply().real(MetaData),
        )
        action_data_opt = item.action_data.apply().real(IOptional[BaseActionData])
        new_history_group = HistoryGroupNode.from_items(
            history[:index-1]
        )
        new_full_state = self.with_args(
            meta=self.meta.apply().real(MetaInfo),
            current=current,
            history=new_history_group,
        )
        return new_full_state, action_data_opt

    def action_space_size(self) -> int:
        meta = self.meta.apply().real(MetaInfo)
        allowed_basic_actions = meta.allowed_basic_actions.apply().real(GeneralTypeGroup)
        return len(allowed_basic_actions.as_tuple)

    def with_max_steps(
        self,
        max_steps: int | None = None,
    ) -> typing.Self:
        meta = self.meta.apply().real(MetaInfo)
        options = meta.options.apply().real(MetaInfoOptions)
        new_options = options.with_new_args(max_steps=max_steps)
        new_meta = meta.with_new_args(options=new_options)
        history_amount = self.history_amount()
        remaining_steps = (
            None
            if max_steps is None
            else max(max_steps - history_amount, 0)
        )
        current = self.current.apply().real(HistoryNode)
        meta_data = current.meta_data.apply().real(MetaData)
        meta_data = meta_data.with_new_args(remaining_steps=remaining_steps)
        new_current = current.with_new_args(meta_data=meta_data)
        history_items: list[HistoryNode] = []
        for i, item in enumerate(history_items):
            remaining_steps = (
                None
                if max_steps is None
                else max(max_steps - i, 0)
            )
            meta_data = item.meta_data.apply().real(MetaData)
            meta_data = meta_data.with_new_args(remaining_steps=remaining_steps)
            item = item.with_new_args(meta_data=meta_data)
            history_items[i] = item
        new_history = HistoryGroupNode.from_items(history_items)
        return self.with_new_args(
            meta=new_meta,
            current=new_current,
            history=new_history,
        )

    def is_last_step_error(self) -> bool:
        history = self.history.apply().real(HistoryGroupNode).as_tuple
        if not history:
            return False
        action_data_opt = history[-1].action_data.apply().real(Optional[BaseActionData])
        action_data = action_data_opt.value
        if action_data is None:
            return False
        return action_data.is_error().as_bool

    def dropped_history_count(self) -> int:
        """Total number of history entries replaced by summary nodes."""
        history = self.history.apply().real(HistoryGroupNode).as_tuple
        total = 0
        for item in history:
            summary = history_summary_data(item)
            if summary is not None:
                total += summary.dropped_count.apply().real(Integer).as_int
        return total

    def last_action_error(self) -> tuple[str, str, str, int] | None:
        """Structured view of the last action error, or None if the last action succeeded.

        Returns (action_type_name, error_class_name, message, action_index).
        """
        action_data_opt = self.last_action_data
        action_data = action_data_opt.value
        if action_data is None:
            return None
        if not action_data.is_error().as_bool:
            return None
        action_opt = action_data.action.apply().real(Optional[IAction])
        action = action_opt.value
        action_name = type(action).__name__ if action is not None else '<unknown>'
        exception_opt = action_data.exception.apply().real(Optional[IExceptionInfo])
        exception = exception_opt.value
        error_class = type(exception).__name__ if exception is not None else '<none>'
        message = ''
        if exception is not None:
            try:
                from env import symbol as symbol_module
                message = str(symbol_module.Symbol.default(exception))
            except Exception as e:  # pragma: no cover - defensive only
                message = f'<unrenderable {error_class}: {e}>'
            message = ' '.join(message.split())[:400]
        if not message:
            message = error_class
        return (
            action_name,
            error_class,
            message,
            self.history_amount(),
        )

    def render_observation(self, max_nodes: int = 40) -> str:
        """Bounded, compact text view of this state.

        Always contains: the goal, the current state (truncated to max_nodes
        lines), the final cost, the last action error (if any) and the number
        of dropped history entries.
        """
        max_nodes = max(1, int(max_nodes))
        meta = self.meta.apply().real(MetaInfo)
        goal = meta.goal.apply().real(IGoal)
        lines: list[str] = []
        lines.append(f'goal: {type(goal).__name__}')
        lines.append(f'goal_achieved: {self.goal_achieved()}')

        state = self.current_state.apply().real(State)
        state_text = ''
        try:
            from env import symbol as symbol_module
            state_text = str(symbol_module.Symbol.default(state))
        except Exception as e:  # pragma: no cover - defensive only
            state_text = f'<unrenderable {type(state).__name__}: {e}>'
        state_lines = [line for line in state_text.splitlines() if line.strip()]
        lines.append(f'current_state ({len(state_lines)} nodes):')
        for line in state_lines[:max_nodes]:
            lines.append('  ' + line.strip()[:200])

        try:
            cost = self.final_cost().as_int
        except Exception:
            cost = 0
        lines.append(f'cost: {cost}')

        error = self.last_action_error()
        if error is None:
            lines.append('last_error: none')
        else:
            action_name, error_class, message, action_index = error
            lines.append(
                f'last_error: action={action_name} class={error_class} '
                f'index={action_index} message={message}'
            )
        lines.append(f'dropped_history_count: {self.dropped_history_count()}')
        return '\n'.join(lines[:max_nodes + 8])

###########################################################
###################### MAIN INDICES #######################
###########################################################

class FullStateIndex(IFullStateIndex[FullState, T], typing.Generic[T], ABC):

    @classmethod
    def outer_type(cls):
        return FullState

class FullStateIntIndex(FullStateIntBaseIndex[FullState, T], typing.Generic[T], ABC):

    @classmethod
    def outer_type(cls):
        return FullState

class FullStateMainIndex(NodeMainBaseIndex, IFullStateIndex[FullState, INode], IInstantiable):

    @classmethod
    def outer_type(cls):
        return FullState

    def find_in_outer_node(self, node: FullState):
        return self.find_in_node(node)

    def replace_in_outer_target(self, target: FullState, new_node: INode):
        return self.replace_in_target(target, new_node)

class FullStateArgIndex(NodeArgBaseIndex, IFullStateIndex[FullState, INode], IInstantiable):

    @classmethod
    def outer_type(cls):
        return FullState

    @classmethod
    def item_type(cls) -> TypeNode:
        return INode.as_type()

    def find_in_outer_node(self, node: FullState):
        return self.find_in_node(node)

    def replace_in_outer_target(self, target: FullState, new_node: INode):
        return self.replace_in_target(target, new_node)

class FullStateGroupBaseIndex(FullStateIntIndex[T], ABC):

    @classmethod
    def group(cls, full_state: FullState) -> TmpNestedArg:
        raise NotImplementedError

    def find_in_outer_node(self, node: FullState):
        return self.find_arg(self.group(node).apply())

    def replace_in_outer_target(self, target: FullState, new_node: T):
        raise NotImplementedError

class FullStateReadonlyGroupBaseIndex(FullStateGroupBaseIndex[T], ABC):

    def replace_in_outer_target(self, target: FullState, new_node: T):
        return Optional.create()

class FullStateGroupTypeBaseIndex(FullStateReadonlyGroupBaseIndex[TypeNode[T]], ABC):

    @classmethod
    def item_type(cls) -> TypeNode:
        return TypeNode.as_type()

    @classmethod
    def inner_item_type(cls) -> type[T]:
        raise NotImplementedError

###########################################################
###################### META INDICES #######################
###########################################################

class MetaAllTypesTypeIndex(
    FullStateGroupTypeBaseIndex[INode],
    IInstantiable):

    @classmethod
    def inner_item_type(cls):
        return INode

    @classmethod
    def group(cls, full_state: FullState) -> TmpNestedArg:
        return full_state.nested_arg((
            FullState.idx_meta,
            MetaInfo.idx_all_types,
        ))

class MetaTypesDetailsTypeIndex(FullStateReadonlyGroupBaseIndex[DetailedType], IInstantiable):

    @classmethod
    def item_type(cls) -> TypeNode:
        return DetailedType.as_type()

    @classmethod
    def group(cls, full_state: FullState) -> TmpNestedArg:
        return full_state.nested_arg((
            FullState.idx_meta,
            MetaInfo.idx_all_types_details,
        ))

class MetaAllowedBasicActionsTypeIndex(
    FullStateGroupTypeBaseIndex[IBasicAction[FullState]],
    IInstantiable):

    @classmethod
    def inner_item_type(cls):
        return IBasicAction

    @classmethod
    def group(cls, full_state: FullState) -> TmpNestedArg:
        return full_state.nested_arg((
            FullState.idx_meta,
            MetaInfo.idx_allowed_basic_actions,
        ))

    @classmethod
    def get_basic_action_index(
        cls,
        node_type: type[IBasicAction],
        full_state: FullState,
    ) -> typing.Self:
        selected_types = full_state.meta.apply().real(
            MetaInfo
        ).allowed_basic_actions.apply().cast(GeneralTypeGroup)
        meta_idx = selected_types.as_tuple.index(node_type.as_type()) + 1
        return cls.from_int(meta_idx)

class MetaAllowedActionsTypeIndex(
    FullStateGroupTypeBaseIndex[IAction[FullState]],
    IInstantiable):

    @classmethod
    def inner_item_type(cls):
        return IAction

    @classmethod
    def group(cls, full_state: FullState) -> TmpNestedArg:
        return full_state.nested_arg((
            FullState.idx_meta,
            MetaInfo.idx_allowed_actions,
        ))

class MetaDefaultTypeIndex(FullStateGroupTypeBaseIndex[IDefault], IInstantiable):

    @classmethod
    def inner_item_type(cls):
        return IDefault

    @classmethod
    def group(cls, full_state: FullState) -> TmpNestedArg:
        return full_state.nested_arg((
            FullState.idx_meta,
            MetaInfo.idx_default_group,
            SubtypeOuterGroup.idx_subtypes,
        ))

class MetaFromIntTypeIndex(FullStateGroupTypeBaseIndex[IFromInt], IInstantiable):

    @classmethod
    def inner_item_type(cls):
        return IFromInt

    @classmethod
    def group(cls, full_state: FullState) -> TmpNestedArg:
        return full_state.nested_arg((
            FullState.idx_meta,
            MetaInfo.idx_from_int_group,
            SubtypeOuterGroup.idx_subtypes,
        ))

class MetaIntTypeIndex(FullStateGroupTypeBaseIndex[IInt], IInstantiable):

    @classmethod
    def inner_item_type(cls):
        return IInt

    @classmethod
    def group(cls, full_state: FullState) -> TmpNestedArg:
        return full_state.nested_arg((
            FullState.idx_meta,
            MetaInfo.idx_int_group,
            SubtypeOuterGroup.idx_subtypes,
        ))

class MetaNodeIndexTypeIndex(FullStateGroupTypeBaseIndex[INodeIndex], IInstantiable):

    @classmethod
    def inner_item_type(cls):
        return INodeIndex

    @classmethod
    def group(cls, full_state: FullState) -> TmpNestedArg:
        return full_state.nested_arg((
            FullState.idx_meta,
            MetaInfo.idx_node_index_group,
            SubtypeOuterGroup.idx_subtypes,
        ))

class MetaFullStateIndexTypeIndex(FullStateGroupTypeBaseIndex[FullStateIndex], IInstantiable):

    @classmethod
    def inner_item_type(cls):
        return FullStateIndex

    @classmethod
    def group(cls, full_state: FullState) -> TmpNestedArg:
        return full_state.nested_arg((
            FullState.idx_meta,
            MetaInfo.idx_full_state_index_group,
            SubtypeOuterGroup.idx_subtypes,
        ))

class MetaFullStateIntIndexTypeIndex(FullStateGroupTypeBaseIndex[FullStateIntIndex], IInstantiable):

    @classmethod
    def inner_item_type(cls):
        return FullStateIntIndex

    @classmethod
    def group(cls, full_state: FullState) -> TmpNestedArg:
        return full_state.nested_arg((
            FullState.idx_meta,
            MetaInfo.idx_full_state_int_index_group,
            SubtypeOuterGroup.idx_subtypes,
        ))

class MetaSingleChildTypeIndex(FullStateGroupTypeBaseIndex[IFromSingleNode], IInstantiable):

    @classmethod
    def inner_item_type(cls):
        return IFromSingleNode

    @classmethod
    def group(cls, full_state: FullState) -> TmpNestedArg:
        return full_state.nested_arg((
            FullState.idx_meta,
            MetaInfo.idx_single_child_group,
            SubtypeOuterGroup.idx_subtypes,
        ))

class MetaGroupTypeIndex(FullStateGroupTypeBaseIndex[IGroup], IInstantiable):

    @classmethod
    def inner_item_type(cls):
        return IGroup

    @classmethod
    def group(cls, full_state: FullState) -> TmpNestedArg:
        return full_state.nested_arg((
            FullState.idx_meta,
            MetaInfo.idx_group_outer_group,
            SubtypeOuterGroup.idx_subtypes,
        ))

class MetaFunctionTypeIndex(FullStateGroupTypeBaseIndex[IFunction], IInstantiable):

    @classmethod
    def inner_item_type(cls):
        return IFunction

    @classmethod
    def group(cls, full_state: FullState) -> TmpNestedArg:
        return full_state.nested_arg((
            FullState.idx_meta,
            MetaInfo.idx_function_group,
            SubtypeOuterGroup.idx_subtypes,
        ))

class MetaBooleanTypeIndex(FullStateGroupTypeBaseIndex[IBoolean], IInstantiable):

    @classmethod
    def inner_item_type(cls):
        return IBoolean

    @classmethod
    def group(cls, full_state: FullState) -> TmpNestedArg:
        return full_state.nested_arg((
            FullState.idx_meta,
            MetaInfo.idx_boolean_group,
            SubtypeOuterGroup.idx_subtypes,
        ))

O = typing.TypeVar('O', bound=IActionOutput)

class MetaAllActionsTypeIndex(
    FullStateGroupTypeBaseIndex[IAction[FullState]],
    IInstantiable):

    @classmethod
    def inner_item_type(cls):
        return IAction

    @classmethod
    def group(cls, full_state: FullState) -> TmpNestedArg:
        return full_state.nested_arg((
            FullState.idx_meta,
            MetaInfo.idx_all_actions,
            SubtypeOuterGroup.idx_subtypes,
        ))

class MetaBasicActionsTypeIndex(
    FullStateGroupTypeBaseIndex[IBasicAction[FullState]],
    IInstantiable):

    @classmethod
    def inner_item_type(cls):
        return IBasicAction

    @classmethod
    def group(cls, full_state: FullState) -> TmpNestedArg:
        return full_state.nested_arg((
            FullState.idx_meta,
            MetaInfo.idx_basic_actions,
            SubtypeOuterGroup.idx_subtypes,
        ))

###########################################################
################## CURRENT STATE INDICES ##################
###########################################################

class CurrentStateScratchIndex(FullStateReadonlyGroupBaseIndex[Scratch], IInstantiable):

    @classmethod
    def item_type(cls) -> TypeNode:
        return Scratch.as_type()

    @classmethod
    def group(cls, full_state: FullState) -> TmpNestedArg:
        return full_state.nested_arg((
            FullState.idx_current,
            HistoryNode.idx_state,
            State.idx_scratch_group,
        ))

class CurrentStateArgsOuterGroupIndex(
    FullStateReadonlyGroupBaseIndex[PartialArgsGroup],
    IInstantiable,
):

    @classmethod
    def item_type(cls) -> TypeNode:
        return PartialArgsGroup.as_type()

    @classmethod
    def group(cls, full_state: FullState) -> TmpNestedArg:
        return full_state.nested_arg((
            FullState.idx_current,
            HistoryNode.idx_state,
            State.idx_args_outer_group,
        ))

class CurrentStateDefinitionIndex(FullStateReadonlyGroupBaseIndex[StateDefinition], IInstantiable):

    @classmethod
    def item_type(cls) -> TypeNode:
        return StateDefinition.as_type()

    @classmethod
    def group(cls, full_state: FullState) -> TmpNestedArg:
        return full_state.nested_arg((
            FullState.idx_current,
            HistoryNode.idx_state,
            State.idx_definition_group,
        ))

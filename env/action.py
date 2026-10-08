from __future__ import annotations
import typing
from abc import ABC
from utils.env_logger import env_logger
from env.core import (
    BaseNode,
    InheritableNode,
    IExceptionInfo,
    Optional,
    IsInsideRange,
    InvalidNodeException,
    IOptional,
    Protocol,
    CountableTypeGroup,
    IInt,
    Integer,
    GreaterThan,
    IWrapper,
    CompositeType,
    OptionalTypeGroup,
    RunInfoStats,
    TmpInnerArg,
    IInstantiable,
    HasArg,
)
from env.state import State
from env.meta_env import (
    MetaInfo,
    IActionOutput,
    IAction,
    IBasicAction,
    IRawAction,
    GeneralTypeGroup,
    MetaInfoOptions,
    ActionInfo,
    IActionInfo,
    NewCostMultiplier,
    CostMultiplier,
    ActionFullInfo,
    RunProcessingCost,
    RunCost,
    FinalCost,
    RunMemoryCost,
)
from env.full_state import (
    FullState,
    MetaData,
    HistoryNode,
    HistoryGroupNode,
    history_summary_data,
    make_history_summary,
    BaseActionData,
    BeforeActionErrorActionData,
    RawActionErrorActionData,
    ActionTypeErrorActionData,
    ActionErrorActionData,
    ActionOutputErrorActionData,
    SuccessActionData,
    MetaAllowedBasicActionsTypeIndex,
)
from env.symbol import Symbol
from env.node_data import NodeData

###########################################################
########################## MAIN ###########################
###########################################################

O = typing.TypeVar('O', bound=IActionOutput)

class FullActionOutput(InheritableNode, IWrapper, IInstantiable):

    idx_raw_action = 1
    idx_action = 2
    idx_output = 3
    idx_new_state = 4

    @classmethod
    def protocol(cls) -> Protocol:
        return cls.default_protocol(CountableTypeGroup(
            CompositeType(
                Optional.as_type(),
                OptionalTypeGroup(IRawAction.as_type()),
            ),
            IAction.as_type(),
            IActionOutput.as_type(),
            State.as_type(),
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
    def new_state(self) -> TmpInnerArg:
        return self.inner_arg(self.idx_new_state)

    @classmethod
    def with_args(
        cls,
        raw_action: Optional[IRawAction],
        action: IAction,
        output: IActionOutput,
        new_state: State,
    ) -> FullActionOutput:
        return cls(
            raw_action,
            action,
            output,
            new_state,
        )

###########################################################
#################### ACTION EXCEPTION #####################
###########################################################

class IActionExceptionInfo(IExceptionInfo, ABC):

    def to_action_data(self) -> BaseActionData:
        raise NotImplementedError

    def as_exception(self):
        return InvalidActionException(self)

class InvalidActionException(InvalidNodeException):

    @property
    def info(self) -> IActionExceptionInfo:
        info = self.args[0]
        return typing.cast(IActionExceptionInfo, info)

    def to_action_data(self) -> BaseActionData:
        return self.info.to_action_data()

class BeforeActionExceptionInfo(InheritableNode, IActionExceptionInfo, IInstantiable):

    idx_exception = 1

    @classmethod
    def protocol(cls) -> Protocol:
        return cls.default_protocol(CountableTypeGroup(
            IExceptionInfo.as_type(),
        ))

    @property
    def exception(self) -> TmpInnerArg:
        return self.inner_arg(self.idx_exception)

    def to_action_data(self) -> BeforeActionErrorActionData:
        return BeforeActionErrorActionData.from_args(
            raw_action=Optional(),
            action=Optional(),
            output=Optional(),
            exception=Optional(self.exception.apply().real(IExceptionInfo)),
        )

class RawActionExceptionInfo(InheritableNode, IActionExceptionInfo, IInstantiable):

    idx_raw_action = 1
    idx_exception = 2

    @classmethod
    def protocol(cls) -> Protocol:
        return cls.default_protocol(CountableTypeGroup(
            RawAction.as_type(),
            IExceptionInfo.as_type(),
        ))

    @property
    def raw_action(self) -> TmpInnerArg:
        return self.inner_arg(self.idx_raw_action)

    @property
    def exception(self) -> TmpInnerArg:
        return self.inner_arg(self.idx_exception)

    def to_action_data(self) -> RawActionErrorActionData:
        return RawActionErrorActionData.from_args(
            raw_action=Optional(self.raw_action.apply()),
            action=Optional(),
            output=Optional(),
            exception=Optional(self.exception.apply().real(IExceptionInfo)),
        )

class ActionTypeExceptionInfo(InheritableNode, IActionExceptionInfo, IInstantiable):

    idx_raw_action = 1
    idx_action = 2
    idx_exception = 3

    @classmethod
    def protocol(cls) -> Protocol:
        return cls.default_protocol(CountableTypeGroup(
            CompositeType(
                Optional.as_type(),
                OptionalTypeGroup(IRawAction.as_type()),
            ),
            IAction.as_type(),
            IExceptionInfo.as_type(),
        ))

    @property
    def raw_action(self) -> TmpInnerArg:
        return self.inner_arg(self.idx_raw_action)

    @property
    def action(self) -> TmpInnerArg:
        return self.inner_arg(self.idx_action)

    @property
    def exception(self) -> TmpInnerArg:
        return self.inner_arg(self.idx_exception)

    def to_action_data(self) -> ActionTypeErrorActionData:
        return ActionTypeErrorActionData.from_args(
            raw_action=self.raw_action.apply().real(Optional[IRawAction]),
            action=Optional(self.action.apply()),
            output=Optional(),
            exception=Optional(self.exception.apply().real(IExceptionInfo)),
        )

class ActionInputExceptionInfo(InheritableNode, IActionExceptionInfo, IInstantiable):

    idx_raw_action = 1
    idx_action = 2
    idx_exception = 3

    @classmethod
    def protocol(cls) -> Protocol:
        return cls.default_protocol(CountableTypeGroup(
            CompositeType(
                Optional.as_type(),
                OptionalTypeGroup(IRawAction.as_type()),
            ),
            IAction.as_type(),
            IExceptionInfo.as_type(),
        ))

    @property
    def raw_action(self) -> TmpInnerArg:
        return self.inner_arg(self.idx_raw_action)

    @property
    def action(self) -> TmpInnerArg:
        return self.inner_arg(self.idx_action)

    @property
    def exception(self) -> TmpInnerArg:
        return self.inner_arg(self.idx_exception)

    def to_action_data(self) -> ActionErrorActionData:
        return ActionErrorActionData.from_args(
            raw_action=self.raw_action.apply().real(Optional[IRawAction]),
            action=Optional(self.action.apply()),
            output=Optional(),
            exception=Optional(self.exception.apply().real(IExceptionInfo)),
        )

class ActionOutputExceptionInfo(InheritableNode, IActionExceptionInfo, IInstantiable):

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
            IAction.as_type(),
            IActionOutput.as_type(),
            IExceptionInfo.as_type(),
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

    def to_action_data(self) -> ActionOutputErrorActionData:
        return ActionOutputErrorActionData.from_args(
            raw_action=self.raw_action.apply().real(Optional[IRawAction]),
            action=Optional(self.action.apply().real(BaseAction)),
            output=Optional(self.output.apply().real(IActionOutput)),
            exception=Optional(self.exception.apply().real(IExceptionInfo)),
        )

###########################################################
##################### IMPLEMENTATION ######################
###########################################################

def _history_entry_cost(item: HistoryNode) -> int:
    """Best-effort final cost recorded on a history entry (0 when absent)."""
    try:
        meta_data = item.meta_data.apply().real(MetaData)
        final_cost = meta_data.final_cost.apply().real(Optional[Integer]).value
        return final_cost.as_int if final_cost is not None else 0
    except Exception:
        return 0

def _apply_history_bound(history: list[HistoryNode], bound: int) -> list[HistoryNode]:
    """Bound FullState.history, replacing the dropped prefix by one summary node.

    Retained: the goal/initial entry, one HistorySummaryNode and the most
    recent (bound - 2) entries. The summary is cumulative: a previous summary
    at index 1 is folded into the new one, so dropped_count is the total
    number of entries ever dropped.
    """
    if bound <= 0:
        return []
    if len(history) <= bound:
        return history
    if bound == 1:
        return history[:1]

    prefix = history[:1]
    body = history[1:]
    dropped_count = 0
    dropped_cost = 0
    if body:
        previous = history_summary_data(body[0])
        if previous is not None:
            dropped_count += previous.dropped_count.apply().real(Integer).as_int
            dropped_cost += previous.dropped_cost.apply().real(Integer).as_int
            body = body[1:]

    keep_recent = max(bound - 2, 1)
    cut = max(len(body) - keep_recent, 0)
    dropped = body[:cut]
    kept = body[cut:]
    dropped_count += len(dropped)
    for item in dropped:
        dropped_cost += _history_entry_cost(item)

    summary = make_history_summary(dropped_count, dropped_cost)
    return prefix + [summary] + kept


class BaseAction(InheritableNode, IAction[FullState], typing.Generic[O], ABC):

    def as_action(self) -> typing.Self:
        return self

    def _run_action(self, full_state: FullState) -> tuple[O, IActionInfo]:
        raise NotImplementedError

    def inner_run(self, full_state: FullState) -> tuple[
        FullActionOutput,
        ActionFullInfo,
    ]:
        action: BaseAction = self
        raw_action: RawAction | None = None

        if isinstance(action, RawAction):
            raw_action = action

            try:
                raw_action.strict_validate()
                action_aux = raw_action.to_action(full_state)
                assert isinstance(action_aux, BaseAction)
                action = action_aux
            except InvalidNodeException as e:
                raise RawActionExceptionInfo(raw_action, e.info).as_exception() from e

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

        try:
            # pylint: disable=protected-access
            output, action_info = action._run_action(full_state)
            output.as_node.strict_validate()
            action_info = (
                action_info.normalize()
                if isinstance(action_info, ActionFullInfo)
                else action_info
            )
            assert isinstance(output, IActionOutput)
        except InvalidNodeException as e:
            raise ActionInputExceptionInfo(
                Optional.with_value(raw_action),
                action,
                e.info,
            ).as_exception() from e

        try:
            new_state, output_info = output.run_output(full_state)
            result = FullActionOutput.with_args(
                raw_action=Optional.with_value(raw_action),
                action=action,
                output=output,
                new_state=new_state,
            )
            result.strict_validate()
            output_info = (
                output_info.normalize()
                if isinstance(output_info, ActionFullInfo)
                else output_info
            )
            return result, ActionFullInfo(action_info, output_info)
        except InvalidNodeException as e:
            raise ActionOutputExceptionInfo(
                Optional.with_value(raw_action),
                action,
                output,
                e.info,
            ).as_exception() from e

    def run_action_details(self, full_state: FullState) -> tuple[
        FullState,
        BaseActionData,
        RunInfoStats,
    ]:
        meta = full_state.meta.apply().real(MetaInfo)
        options = meta.options.apply().real(MetaInfoOptions)
        max_history_state_size = options.max_history_state_size.apply().real(
            IOptional[IInt]
        ).value
        current = full_state.current.apply().real(HistoryNode)
        meta_data = current.meta_data.apply().real(MetaData)
        meta_info_options = full_state.meta_info_options
        last_cost_multiplier = full_state.cost_multiplier.value
        remaining_steps_opt = meta_data.remaining_steps.apply().real(Optional[Integer])
        remaining_steps = (
            (remaining_steps_opt.value.as_int)
            if remaining_steps_opt.value is not None
            else None)
        new_cost_multiplier: NewCostMultiplier | None = None
        processing_cost: RunProcessingCost | None = None
        run_memory_size = 0

        try:
            if remaining_steps is not None:
                try:
                    GreaterThan.with_ints(remaining_steps, 0).raise_on_false()
                except InvalidNodeException as e:
                    raise BeforeActionExceptionInfo(e.info).as_exception() from e
            full_output, action_full_info = self.inner_run(full_state)
            raw_action_opt = full_output.raw_action.apply().real(Optional[IRawAction])
            actual_action = full_output.action.apply().real(BaseAction)
            output = full_output.output.apply().real(IActionOutput)
            next_state = full_output.new_state.apply().real(State)
            action_full_info = action_full_info.normalize()
            new_cost_multiplier = action_full_info.get_new_cost_multiplier()
            processing_cost = action_full_info.get_processing_cost()
            run_memory_size = action_full_info.get_run_memory_size().as_int
            next_state.strict_validate()
            action_data: BaseActionData = SuccessActionData.from_args(
                raw_action=raw_action_opt,
                action=Optional(actual_action),
                output=Optional(output),
                exception=Optional.create(),
            )
        except InvalidActionException as e:
            symbol = Symbol(
                node=e.info.as_node,
                node_types=full_state.node_types(),
            )
            env_logger.debug(str(symbol), exc_info=e)
            next_state = current.state.apply().real(State)
            action_data = e.to_action_data()

        try:
            action_data.strict_validate()
        except Exception as e:
            print('action_data', Symbol(
                node=action_data,
                node_types=full_state.node_types(),
            ))
            raise e
        remaining_steps = (
            remaining_steps - 1
            if remaining_steps is not None
            else None)

        cost_multiplier = CostMultiplier.calculate(
            meta_info_options=meta_info_options,
            last_cost_multiplier=last_cost_multiplier,
            new_cost_multiplier=new_cost_multiplier,
        )

        current = current.with_new_args(
            action_data=Optional(action_data),
        )

        history = list(full_state.history.apply().real(HistoryGroupNode).as_tuple)
        history.append(current)

        meta_data = meta_data.with_new_args(
            remaining_steps,
            new_cost_multiplier=Optional.with_value(new_cost_multiplier),
            cost_multiplier=Optional.with_value(cost_multiplier),
        )

        current = HistoryNode.with_args(
            state=next_state,
            meta_data=meta_data,
        ).before_run_stats()

        if max_history_state_size is not None:
            history = _apply_history_bound(
                history,
                max_history_state_size.as_int,
            )

        new_full_state = FullState.with_args(
            meta=meta,
            current=current,
            history=HistoryGroupNode.from_items(history),
        )

        if processing_cost is not None:
            node_types = full_state.node_types()
            node_data = (
                NodeData(node=new_full_state, node_types=node_types)
                if not BaseNode.fast
                else None)

            full_state_memory_size = len(new_full_state)
            visible_state_memory_size = (
                len(node_data.to_data_array())
                if node_data is not None
                else full_state_memory_size
            )
            main_state_memory_size = len(next_state)

            memory_cost = RunMemoryCost.with_args(
                full_state_memory=full_state_memory_size,
                visible_state_memory=visible_state_memory_size,
                main_state_memory=main_state_memory_size,
                run_memory=run_memory_size,
            )
            run_cost = RunCost.with_args(
                processing_cost=processing_cost,
                memory_cost=memory_cost,
            )
            final_cost = FinalCost.with_args(
                cost_multiplier=cost_multiplier,
                run_cost=run_cost,
                options=meta_info_options,
            ).normalize()
            meta_data = meta_data.with_new_args(
                run_cost=Optional.with_value(run_cost),
                final_cost=final_cost.as_int,
            )
            current = current.with_new_args(
                meta_data=meta_data,
            )
            new_full_state = new_full_state.with_new_args(
                current=current,
            )

        instructions = (
            processing_cost.instructions.apply().real(Integer).as_int
            if processing_cost is not None
            else 1
        )
        stats = RunInfoStats.with_args(
            instructions=instructions,
            memory=run_memory_size,
        )

        return new_full_state, action_data, stats

    def run_action(self, full_state: FullState) -> FullState:
        full_state, _, __ = self.run_action_details(full_state)
        return full_state

class BasicAction(
    BaseAction[O],
    IBasicAction[FullState],
    typing.Generic[O],
    ABC,
):
    pass

class GeneralAction(
    BaseAction[typing.Self], # type: ignore[misc]
    IActionOutput[FullState],
    ABC,
):
    def _run_action(self, full_state: FullState) -> tuple[typing.Self, ActionInfo]:
        return self, ActionInfo.create()

class RawAction(BaseAction[IActionOutput], IRawAction[FullState], IInstantiable):

    idx_action_index = 1
    idx_arg1 = 2
    idx_arg2 = 3
    idx_arg3 = 4

    @classmethod
    def protocol(cls) -> Protocol:
        return cls.default_protocol(CountableTypeGroup(
            MetaAllowedBasicActionsTypeIndex.as_type(),
            Integer.as_type(),
            Integer.as_type(),
            Integer.as_type(),
        ))

    @property
    def action_index(self) -> TmpInnerArg:
        return self.inner_arg(self.idx_action_index)

    @property
    def arg1(self) -> TmpInnerArg:
        return self.inner_arg(self.idx_arg1)

    @property
    def arg2(self) -> TmpInnerArg:
        return self.inner_arg(self.idx_arg2)

    @property
    def arg3(self) -> TmpInnerArg:
        return self.inner_arg(self.idx_arg3)

    def to_action(self, full_state: FullState) -> IBasicAction[FullState]:
        action_index = self.action_index.apply().real(MetaAllowedBasicActionsTypeIndex)
        arg1 = self.arg1.apply().real(Integer)
        arg2 = self.arg2.apply().real(Integer)
        arg3 = self.arg3.apply().real(Integer)

        action_type = action_index.find_in_outer_node(full_state).value_or_raise

        basic_action = action_type.type.from_raw(arg1.as_int, arg2.as_int, arg3.as_int)

        return basic_action

    @classmethod
    def from_basic_action(
        cls,
        action: IBasicAction,
        full_state: FullState,
    ) -> typing.Self:
        args = action.to_raw_args().as_tuple
        assert len(args) == 3, f"Expected 3 arguments, got {len(args)} ({type(action)})"
        action_index = MetaAllowedBasicActionsTypeIndex.get_basic_action_index(
            node_type=type(action),
            full_state=full_state)
        assert action_index.as_int > 0, f"Invalid action type: {type(action)}"
        return cls.with_args(
            action_index=action_index,
            arg1=Integer(args[0].as_int),
            arg2=Integer(args[1].as_int),
            arg3=Integer(args[2].as_int),
        )

    @classmethod
    def with_raw_args(
        cls,
        action_index: int,
        arg1: int,
        arg2: int,
        arg3: int,
    ) -> typing.Self:
        return cls.with_args(
            action_index=MetaAllowedBasicActionsTypeIndex(action_index),
            arg1=Integer(arg1),
            arg2=Integer(arg2),
            arg3=Integer(arg3),
        )

    @classmethod
    def with_args(
        cls,
        action_index: MetaAllowedBasicActionsTypeIndex,
        arg1: Integer,
        arg2: Integer,
        arg3: Integer,
    ) -> typing.Self:
        return cls(
            action_index,
            arg1,
            arg2,
            arg3,
        )

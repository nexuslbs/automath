import typing
from env import (
    core,
    state,
    action,
    meta_env,
full_state,
)
from env.symbol import Symbol
from env.goal_env import GoalEnv
from env.node_types import HaveScratch
from test_suite import (
    test_utils,
    agent_test,
    basic_test,
    boolean_test,
    arithmetic_test,
    indices_test,
    control_flow_test,
)
from test_suite.action_impl import (
    action_00_action_meta,
    action_01_state_meta,
    action_02_manage_scratch,
    action_03_define_scratch,
    action_04_update_scratch,
    action_05_manage_args_group,
)
from utils.env_logger import env_logger

def _final_verification(final_states: list[full_state.FullState]):
    for i_case, full_state_case in enumerate(final_states):
        history_amount = full_state_case.history_amount()
        for i in range(history_amount):
            index = i+1
            current_fs, action_data_opt = full_state_case.at_history(index)
            action_data = action_data_opt.value
            assert current_fs is not None
            assert action_data is not None
            raw_action_opt = action_data.raw_action.apply().cast(
                core.IOptional[meta_env.IRawAction])
            action_opt = action_data.action.apply().cast(core.IOptional[meta_env.IAction])
            raw_action = raw_action_opt.value
            act = raw_action if raw_action is not None else action_opt.value
            assert act is not None
            assert isinstance(act, action.BaseAction)

            next_fs = act.run_action(current_fs)

            if i < history_amount - 1:
                actual_next_fs, _ = full_state_case.at_history(index+1)
            else:
                actual_next_fs = full_state_case

            expected_next_state = next_fs.current_state.apply()
            actual_next_state = actual_next_fs.current_state.apply()
            if expected_next_state != actual_next_state:
                print('actual_next_state:', Symbol.default(actual_next_state))
                print('expected_next_state:', Symbol.default(expected_next_state))
            assert expected_next_state == actual_next_state, f'{i_case}-{i}'
            last_action_data_opt = actual_next_fs.last_action_data
            if last_action_data_opt != action_data_opt:
                print('last_action_data_opt:', Symbol.default(last_action_data_opt.as_node))
                print('action_data_opt:', Symbol.default(action_data_opt.as_node))
            assert last_action_data_opt == action_data_opt, f'{i_case}-{i}'
            next_fs = next_fs.minimal_meta()
            actual_next_fs = actual_next_fs.minimal_meta()
            if next_fs != actual_next_fs:
                print('actual_next_fs:', Symbol.default(actual_next_fs))
                print('expected_next_fs:', Symbol.default(next_fs))
            assert next_fs == actual_next_fs, f'{i_case}-{i}'

            exception = action_data.exception.apply().cast(core.IOptional[core.IExceptionInfo])

            if exception.is_empty().as_bool:
                full_out, _ = act.inner_run(current_fs)
                output = full_out.output.apply().cast(action.IActionOutput)
                next_state = full_out.new_state.apply().cast(state.State)
                actual_output = action_data.output.apply().cast(
                    core.IOptional[action.IActionOutput]
                ).value
                assert next_state == expected_next_state
                assert output == actual_output

    return final_states

T = typing.TypeVar('T')

def _run_main_test(
    fn: typing.Callable[[], list[full_state.FullState]],
    fn_run_agent: typing.Callable[[list[full_state.FullState]], T],
    fn_additional_info_agent: typing.Callable[[T], str] | None = None,
):
    def fn_additional_info(final_states: list[full_state.FullState]):
        amount = test_utils.result_amount(final_states)
        action_amount = test_utils.result_action_amount(final_states)
        return f"Completed tests: {amount} ({action_amount} actions)"
    full_states = test_utils.run_module_test(fn, fn_additional_info)
    test_utils.run_test(
        f' -> agent ({fn.__module__})',
        lambda: fn_run_agent(full_states),
        fn_additional_info_agent,
    )
    return full_states

def _run_agent_tests(final_states: list[full_state.FullState]) -> tuple[int, int]:
    all_complete = 0
    for full_state_case in final_states:
        complete = agent_test.replay_actions_with_demo_agent(full_state_case)
        all_complete += complete
    return all_complete, len(final_states)

def _fn_additional_info_agent(values: tuple[int, int]):
    all_complete, amount = values
    suffix = ''
    if all_complete != amount:
        diff = amount - all_complete
        if diff == 1:
            suffix = f" ({diff} case without successful action to simulate)"
        else:
            suffix = f" ({diff} cases without successful actions to simulate)"
    return f"Agent completed: {all_complete}{suffix}"

def run_with_agent(
    fast: bool,
    fn_run_agent: typing.Callable[[list[full_state.FullState]], T],
    fn_additional_info_agent: typing.Callable[[T], str] | None = None,
) -> list[full_state.FullState]:
    final_states = test_utils.ModuleResults()

    def run(
        fn: typing.Callable[[], list[full_state.FullState]],
    ) -> list[full_state.FullState]:
        return _run_main_test(
            fn=fn,
            fn_run_agent=fn_run_agent,
            fn_additional_info_agent=fn_additional_info_agent,
        )


    final_states.fold(run(basic_test.test))
    test_utils.release_caches()

    final_states.fold(run(boolean_test.test))
    test_utils.release_caches()
    final_states.fold(run(arithmetic_test.test))
    test_utils.release_caches()
    final_states.fold(run(indices_test.test))
    test_utils.release_caches()

    final_states.fold(run(action_00_action_meta.test))
    test_utils.release_caches()
    final_states.fold(run(lambda: action_01_state_meta.test(fast)))
    test_utils.release_caches()
    final_states.fold(run(action_02_manage_scratch.test))
    test_utils.release_caches()
    final_states.fold(run(action_03_define_scratch.test))
    test_utils.release_caches()
    final_states.fold(run(action_04_update_scratch.test))
    test_utils.release_caches()
    final_states.fold(run(action_05_manage_args_group.test))
    test_utils.release_caches()

    final_states.fold(run(control_flow_test.test))
    test_utils.release_caches()

    return final_states

def _main_tests(fast: bool) -> list[full_state.FullState]:
    return run_with_agent(
        fast=fast,
        fn_run_agent=_run_agent_tests,
        fn_additional_info_agent=_fn_additional_info_agent,
    )

def initialize_cache() -> None:
    GoalEnv(HaveScratch.with_goal(core.Void()))

def all_tests(fast: bool) -> list[full_state.FullState]:
    try:
        test_utils.run_test('initialize_cache', initialize_cache)
        final_states = test_utils.run_test('main_tests', lambda: _main_tests(fast))
        test_utils.run_test('final_verification', lambda: _final_verification(final_states))
        return final_states
    except core.InvalidNodeException as e:
        symbol = Symbol.default(e.info.as_node)
        env_logger.debug(str(symbol), exc_info=e)
        raise e

def test(fast: bool = False) -> list[full_state.FullState]:
    return test_utils.run_test('all_tests', lambda: all_tests(fast))

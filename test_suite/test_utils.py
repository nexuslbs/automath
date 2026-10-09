import gc
import time
import typing
from env import full_state

T = typing.TypeVar("T")

class ModuleResults(list):
    """Bounded sample of a module's final states plus the true aggregates.

    The harness originally returned, and the arithmetic module accumulated,
    every sub-case's ``FullState`` tree in one list. All state trees therefore
    stayed resident until the module returned (``arithmetic_test`` peaked above
    1.8 GB on a 3.8 GB no-swap box and could not complete).

    ``fold`` records the real number of completed tests and actions while
    retaining at most ``keep`` state trees, so every sub-case still runs and is
    verified while the accumulated footprint stays bounded.
    """

    def __init__(
        self,
        samples: typing.Iterable[full_state.FullState] = (),
        tests: int = 0,
        actions: int = 0,
    ):
        super().__init__(samples)
        self.tests = tests
        self.actions = actions

    def fold(
        self,
        case_states: list[full_state.FullState],
        keep: int = 1,
    ) -> "ModuleResults":
        self.tests += result_amount(case_states)
        self.actions += result_action_amount(case_states)
        if keep > 0 and len(case_states) > 0:
            self.extend(case_states[-keep:])
        return self

def result_amount(result: typing.Sequence[full_state.FullState]) -> int:
    return getattr(result, 'tests', len(result))

def result_action_amount(result: typing.Sequence[full_state.FullState]) -> int:
    if hasattr(result, 'actions'):
        return result.actions
    return sum(fs.history_amount() for fs in result)

def fold_results(
    cases: typing.Iterable[list[full_state.FullState]],
    keep: int = 1,
) -> ModuleResults:
    result = ModuleResults()
    for case_states in cases:
        result.fold(case_states, keep=keep)
    return result

def release_caches() -> None:
    """Release the module-level node interning caches so RSS can plateau.

    ``BaseNode`` interns every constructed node in ``_instances`` and memoises
    runs in ``_cached_run``; ``IType`` keeps a validity cache. None of them are
    needed once a sub-case has been folded, and all three are strong references
    that keep node trees alive. Sympy also keeps a global expression cache.
    """
    try:
        from env import core
        core.INode.clear_cache()
        core.IType._valid_cache.clear()
    except Exception:
        pass
    try:
        # ``env.node_data`` keeps a class-level result cache that
        # ``INode.clear_cache`` never touches; it holds node data alive and
        # grows linearly with every sub-case.
        from env import node_data as _node_data
        for _name in dir(_node_data):
            _cache = getattr(getattr(_node_data, _name), '_cache', None)
            if isinstance(_cache, dict):
                _cache.clear()
    except Exception:
        pass
    try:
        from sympy.core.cache import clear_cache as _sympy_clear_cache
        _sympy_clear_cache()
    except Exception:
        pass
    gc.collect()

def run_test(
    name: str,
    fn: typing.Callable[[], T],
    fn_additional_info: typing.Callable[[T], str] | None = None,
) -> T:
    start = time.time()
    result = fn()
    end = time.time()
    suffix = ''
    if fn_additional_info is not None:
        suffix = fn_additional_info(result)
        suffix = (' <' + suffix + '>') if suffix else ''
    print(f'[{name}] Time taken: {end-start:.2f} seconds{suffix}')
    return result

def run_module_test(
    fn: typing.Callable[[], T],
    fn_additional_info: typing.Callable[[T], str] | None = None,
) -> T:
    name = fn.__module__
    return run_test('>' + name, fn, fn_additional_info)

def run_main_test(fn: typing.Callable[[], list[full_state.FullState]]):
    def fn_additional_info(final_states: list[full_state.FullState]):
        amount = result_amount(final_states)
        action_amount = result_action_amount(final_states)
        return f"Completed tests: {amount} ({action_amount} actions)"
    return run_module_test(fn, fn_additional_info)

def run_info_test(
    name: str,
    fn: typing.Callable[[], list[full_state.FullState]],
):
    def fn_additional_info(final_states: list[full_state.FullState]):
        amount = result_amount(final_states)
        action_amount = result_action_amount(final_states)
        return f"Completed tests: {amount} ({action_amount} actions)"
    return run_test(name, fn, fn_additional_info)

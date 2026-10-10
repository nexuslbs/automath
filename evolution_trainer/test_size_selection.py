"""pytest entry for the task-4344 selection tests (default-named mirror).

The library-runner contract lives in ``size_selection_tests.py`` (``check_*``
functions plus ``main``); this default-named module re-exports its ``test_*``
wrappers so that ``pytest evolution_trainer/`` (directory collection) finds
them too. ``pytest evolution_trainer/size_selection_tests.py`` works on its own
because pytest collects an explicitly passed file regardless of naming.
"""

from __future__ import annotations

from .size_selection_tests import (  # noqa: F401
    test_episode_seed_formula,
    test_mask_excludes_off_subtree,
    test_mean_rate_deterministic,
    test_pool_witnesses_replay,
    test_target_change_cases,
    test_target_pool_deterministic,
    test_target_variant_disjoint,
    test_val_batch_reproducible,
    test_val_pool_disjoint,
)

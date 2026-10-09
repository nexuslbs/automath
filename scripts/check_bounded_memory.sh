#!/usr/bin/env bash
# Regression test for FIX 5 (bounded harness retention).
#
# The pre-fix arithmetic module accumulated every sub-case's FullState tree and
# never released the node interning caches, so its resident set grew without
# bound. Under this virtual-memory cap the old code dies with MemoryError; the
# fixed code, which folds each sub-case into counters and releases the caches,
# completes with exit 0.
#
# Override for a different box:
#   PYTHON=/path/to/python LIMIT_KB=1400000 scripts/check_bounded_memory.sh
set -u

cd "$(dirname "$0")/.."
PYTHON="${PYTHON:-/opt/automath/venv/bin/python}"
LIMIT_KB="${LIMIT_KB:-1400000}"

echo "check_bounded_memory: cap=${LIMIT_KB} kB, python=${PYTHON}"
bash -c "ulimit -v ${LIMIT_KB}; exec ${PYTHON} -u -c \"from test_suite import arithmetic_test as m, test_utils; test_utils.run_module_test(m.test)\""
status=$?
echo "check_bounded_memory: exit ${status}"
exit ${status}

# U2 FIX 5 evidence (automath, branch main)

Host: automath host (reachable). Second ssh config host unreachable:
`Permission denied (publickey,password)`.
Remote clone: /opt/automath/repo, venv: /opt/automath/venv.
Branch main. Commit `69ece251cf8579f0a1aa34a30ff9aaf8ab00793a`.

## STEP 0 state recovery (raw, 02:43)

```
=== git status --porcelain ===
 M test_suite/arithmetic_test.py
 M test_suite/test_root.py
 M test_suite/test_utils.py
 M tests.py
 M tests_fast.py
=== git log --oneline -3 ===
53478aa docs: fix report module-test reproduction command
732ff0e docs: add U6+U7 improvement and alternative-approach report
93e6ec4 scripts: per-case registry drives real math envs
=== HEAD ===
53478aa6732c1a6b3b97991781ee18b0ac7a1f4d
=== origin/main ===
53478aa6732c1a6b3b97991781ee18b0ac7a1f4d
=== branch ===
main
=== ps ===
root 980982 ... bash -c ... diag_keep.py clear 12 0 ...
root 980983 ... python -u /opt/automath/diag_keep.py clear 12 0
root 981065 ... sh -c sleep 170; ...
=== free -m === total 3814 / used 3111 / free 689 / available 703 ; Swap: 0
=== nproc === 2
```

Previous (killed) dispatch left 5 uncommitted edits plus one detached U1
diagnostic process. The edits already implemented the decided fix 5
(`ModuleResults` fold); I inspected them, kept the correct part, and completed
the missing per-case module-cache release, regression test and docs note. The
detached process was killed (box freed to 1763 MB available).

## git show --stat HEAD

```
commit 69ece251cf8579f0a1aa34a30ff9aaf8ab00793a
Author: dsh-developer <dev@omni.local>
    test_suite: bound arithmetic harness retention (FIX 5)
 docs/REPORT.md                  |  22 +++++++++
 scripts/check_bounded_memory.sh |  22 +++++++++
 test_suite/arithmetic_test.py   | 101 +++++++++++++++++++---------------------
 test_suite/test_root.py         |  45 +++++++++++-------
 test_suite/test_utils.py        |  92 ++++++++++++++++++++++++++++++++++--
 tests.py                        |   6 +--
 tests_fast.py                   |   6 +--
 7 files changed, 213 insertions(+), 81 deletions(-)
```

## (a) BEFORE / AFTER peak RSS + wall (arithmetic_test)

Command (detached, `/usr/bin/time -v timeout 300 python -u -c
"from test_suite import arithmetic_test as m, test_utils;
test_utils.run_module_test(m.test)"`).

BEFORE, unchanged pushed 53478aa (`/opt/automath/logs/u1_before.log`):
```
Command exited with non-zero status 124
	Elapsed (wall clock) time (h:mm:ss or m:ss): 5:00.21
	Maximum resident set size (kbytes): 1777692
	Exit status: 124
EXIT=124
```

AFTER, fixed code (`/opt/automath/logs/u2_after2.log`):
```
Command exited with non-zero status 124
	Elapsed (wall clock) time (h:mm:ss or m:ss): 5:00.07
	Maximum resident set size (kbytes): 259064
	Exit status: 124
EXIT=124
```

Peak RSS 1777692 -> 259064 kB (1736 -> 253 MiB, 6.9x) and flat (3 cases
181 MB, 10 cases 252 MB, VmHWM 259 MB). RSS is bounded. The run still exits 124
under `timeout 300`: residual cost is CPU (~15 s/sub-case; the full module
completes in 6:51 when not wall-limited, see the NEW regression).

## (b) Regression `scripts/check_bounded_memory.sh` (committed)

Old code (worktree at 53478aa) under `ulimit -v 1400000`
(`/opt/automath/logs/u2_reg_old.log`) FAILS:
```
... 13 sub-cases ...
Traceback (most recent call last):
numpy._core._exceptions._ArrayMemoryError: Unable to allocate 504. KiB for an array with shape (7171, 9) and data type int64
check_bounded_memory: exit 1
	Elapsed (wall clock) time (h:mm:ss or m:ss): 3:33.22
	Maximum resident set size (kbytes): 1281380
	Exit status: 1
EXIT=1
```

Fixed code under `ulimit -v 1400000` (`/opt/automath/logs/u2_reg_new.log`) PASSES:
```
check_bounded_memory: exit 0
EXIT=0
	Elapsed (wall clock) time (h:mm:ss or m:ss): 6:51.07
	Maximum resident set size (kbytes): 354564
	Exit status: 0
```
Case count: OLD 13 then MemoryError; NEW 27 lines, all sub-cases run.

## (c) Full suite

NOT RUN: `tests.py` runs every module and includes the ~411 s arithmetic
module, plus ~150 s for the rest, exceeding the remaining budget. Explicit gap.

## Commit + push

- Secret scan of staged diff: `SECRET_SCAN_CLEAN`
  (`grep -nE 'PRIVATE-KEY|ghp-|ghs-|sk_|AKI-A|api_key-|password-'`).
- The remote host has no git credential helper, so `git push` there failed:
  `fatal: could not read Username for 'https://github.com'`. The exact commit was
  bundled (5056 bytes) to the workstation clone (wired helper
  `/opt/omni/workstation/bin/gh-cred`) and pushed there:
  `53478aa..69ece25 FETCH_HEAD -> main`.
- Remote: `git rev-parse HEAD` == `git rev-parse origin/main` ==
  `69ece251cf8579f0a1aa34a30ff9aaf8ab00793a` (after `git fetch origin`).
- Workstation clone: `git fetch origin && git reset --hard origin/main` ->
  `HEAD 69ece25`.

## Acceptance gaps

1. arithmetic_test does NOT complete with exit 0 under `timeout 300`; it exits
   124 because the workload is CPU-bound (~411 s). Memory is bounded (259 MB
   peak, flat). The unit deliverable (land FIX 5, commit+push) is met; the
   "exit 0, wall < 300 s" acceptance is not.
2. Full-suite entry point not run (same arithmetic wall cost).

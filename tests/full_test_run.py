#!/usr/bin/env python3
"""Full Test Suite — modular test runner for generate.py.

Discovers and runs all test suites from tests/suites/test_*.py.
Each suite registers its own tests via the Suite API.

Usage:
    python tests/full_test_run.py                          # Run all tests
    python tests/full_test_run.py --only test_sfx          # Run one suite
    python tests/full_test_run.py --only test_sfx,test_ps  # Run multiple
    python tests/full_test_run.py --list                   # List all tests
"""

import argparse
import importlib
import os
import subprocess
import sys
import time
from pathlib import Path


SCRIPT_DIR = Path(__file__).resolve().parent.parent
DEMOS = SCRIPT_DIR / "demos" / "full-test"

SUITES_DIR = Path(__file__).resolve().parent / "suites"


def discover_suites():
    """Auto-discover test suites from tests/suites/test_*.py.
    Files prefixed with 'exclude_' are ignored."""
    return sorted(
        p.stem for p in SUITES_DIR.glob("test_*.py")
        if not p.name.startswith("exclude_")
    )


class TestEntry:
    """A single registered test."""

    def __init__(self, name, cmd, output=None, prep=False):
        self.name = name
        self.cmd = cmd
        self.output = Path(output) if output else None
        self.prep = prep  # True = prep step, not counted in results


class Suite:
    """Collects tests registered by a suite module."""

    def __init__(self, suite_name):
        self.suite_name = suite_name
        self.tests: list[TestEntry] = []
        self.out_dir = DEMOS / suite_name
        self.prep_dir = self.out_dir / "prep"
        self._setup_fn = None
        self._cleanup_fn = None

    def add(self, name, cmd, output=None, prep=False):
        self.tests.append(TestEntry(name, cmd, output, prep))

    def on_setup(self, fn):
        """Register a setup function (called before tests run)."""
        self._setup_fn = fn

    def on_cleanup(self, fn):
        """Register a cleanup function (called after tests run)."""
        self._cleanup_fn = fn


class Results:
    """Track test results."""

    def __init__(self):
        self.entries = []  # (status, duration, name)
        self.passed = 0
        self.failed = 0
        self.skipped = 0


def run_cmd(cmd, cwd=None):
    """Run a command, return (success, duration)."""
    start = time.time()
    result = subprocess.run(cmd, cwd=cwd or SCRIPT_DIR)
    duration = int(time.time() - start)
    return result.returncode == 0, duration


def run_suite(suite: Suite, results: Results, counter: list, total: int, force: bool = False):
    """Execute all tests in a suite."""
    # Skip entire suite (incl. preps) if all real tests already have output
    real_tests = [t for t in suite.tests if not t.prep]
    if not force and real_tests and all(
        t.output and t.output.exists() and t.output.stat().st_size > 0
        for t in real_tests
    ):
        for t in real_tests:
            counter[0] += 1
            print()
            print()
            print("━" * 64)
            print(f"  [{counter[0]}/{total}] {t.name}")
            print("━" * 64)
            print()
            print(f"  SKIP (exists): {t.output.name}")
            results.entries.append(("SKIP", 0, t.name))
            results.skipped += 1
        return

    suite.out_dir.mkdir(parents=True, exist_ok=True)
    if any(t.prep for t in suite.tests):
        suite.prep_dir.mkdir(parents=True, exist_ok=True)

    # Setup
    if suite._setup_fn:
        suite._setup_fn()

    try:
        # Separate preps from real tests, run preps just-in-time before
        # the first real test that actually needs to execute.
        preps = [t for t in suite.tests if t.prep]
        real = [t for t in suite.tests if not t.prep]
        preps_done = False

        for test in real:
            counter[0] += 1
            idx = counter[0]

            print()
            print()
            print("━" * 64)
            print(f"  [{idx}/{total}] {test.name}")
            print("━" * 64)
            print()

            # Skip if output exists (--force disables)
            if not force and test.output and test.output.exists() and test.output.stat().st_size > 0:
                print(f"  SKIP (exists): {test.output.name}")
                results.entries.append(("SKIP", 0, test.name))
                results.skipped += 1
                continue

            # Run preps once before the first real test that needs execution
            if not preps_done:
                preps_done = True
                for prep in preps:
                    if not force and prep.output and prep.output.exists() and prep.output.stat().st_size > 0:
                        print(f"  CACHED: {prep.name}")
                        continue
                    print(f"  PREP: {prep.name}")
                    ok, dur = run_cmd(prep.cmd)
                    if not ok:
                        print(f"  PREP FAILED: {prep.name} ({dur}s)")
                        # Fail current + skip remaining
                        results.entries.append(("FAIL", dur, test.name))
                        results.failed += 1
                        for remaining in real[real.index(test) + 1:]:
                            counter[0] += 1
                            print(f"\n  [{counter[0]}/{total}] SKIP: {remaining.name} (prep failed)")
                            results.entries.append(("SKIP", 0, remaining.name))
                            results.skipped += 1
                        return

            ok, dur = run_cmd(test.cmd)

            if ok and test.output:
                if not test.output.exists() or test.output.stat().st_size == 0:
                    ok = False

            if ok:
                print(f"\n  >>> PASS ({dur}s)")
                results.entries.append(("PASS", dur, test.name))
                results.passed += 1
            else:
                print(f"\n  >>> FAIL ({dur}s)")
                results.entries.append(("FAIL", dur, test.name))
                results.failed += 1
    finally:
        if suite._cleanup_fn:
            suite._cleanup_fn()


def load_suites(only=None):
    """Import suite modules and collect registrations."""
    suites = []
    for name in discover_suites():
        if only and name not in only:
            continue
        mod = importlib.import_module(f"suites.{name}")
        suite = Suite(name)
        mod.register(suite)
        suites.append(suite)
    return suites


def count_real_tests(suites):
    """Count non-prep tests across all suites."""
    return sum(1 for s in suites for t in s.tests if not t.prep)


def main():
    parser = argparse.ArgumentParser(description="Full test suite for generate.py")
    parser.add_argument("--only", help="Comma-separated suite names to run")
    parser.add_argument("--force", action="store_true", help="Re-run tests even if output exists")
    parser.add_argument("--list", action="store_true", help="List all tests and exit")
    args = parser.parse_args()

    os.chdir(SCRIPT_DIR)
    sys.path.insert(0, str(Path(__file__).resolve().parent))

    only = set(args.only.split(",")) if args.only else None
    suites = load_suites(only)
    total = count_real_tests(suites)

    if args.list:
        print(f"\n  {total} tests in {len(suites)} suites:\n")
        idx = 0
        for suite in suites:
            for test in suite.tests:
                if test.prep:
                    print(f"      PREP  {test.name}")
                else:
                    idx += 1
                    print(f"  [{idx:2d}/{total}]  {test.name}")
            print()
        return

    print()
    print("═" * 64)
    print(f"  Full Test Suite — {total} tests in {len(suites)} suites")
    print("═" * 64)

    DEMOS.mkdir(parents=True, exist_ok=True)
    results = Results()
    counter = [0]

    for suite in suites:
        run_suite(suite, results, counter, total, force=args.force)

    # Summary
    print()
    print()
    print("━" * 64)
    print("  RESULTS")
    print("━" * 64)
    print()
    for status, dur, name in results.entries:
        print(f"  {status:<5s} {dur:>4d}s  {name}")
    print()
    print(f"  Total: {results.passed} passed, {results.failed} failed, {results.skipped} skipped")
    print()
    print("━" * 64)

    sys.exit(0 if results.failed == 0 else 1)


if __name__ == "__main__":
    main()

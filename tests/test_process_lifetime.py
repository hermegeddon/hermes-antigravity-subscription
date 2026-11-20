"""Regression tests for issue #50: orphaned agy workers and zombies.

Covers:
- ``_own_process_group`` arms PR_SET_PDEATHSIG on Linux so a dead parent
  takes its agy child with it.
- The interpreter-exit registry terminates still-running children even if
  close() paths were skipped, and never aborts on one stubborn child.
- ``_kill_process_tree`` / ``terminate_process`` reap killed children and
  clean the registry (no ``<defunct>`` entries left under a live worker).
"""

import contextlib
import ctypes
import os
import subprocess
import sys
import time
import unittest
from unittest import mock

import process
from process import (
    _kill_process_tree,
    _own_process_group,
    _terminate_tracked_processes,
    register_tracked_process,
    terminate_process,
    unregister_tracked_process,
)

_PLUGIN_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _is_windows_pid_alive(pid: int) -> bool:
    """Return True if a Windows process PID is still running."""
    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    kernel32.OpenProcess.argtypes = [ctypes.c_uint32, ctypes.c_int, ctypes.c_uint32]
    kernel32.OpenProcess.restype = ctypes.c_void_p
    kernel32.WaitForSingleObject.argtypes = [ctypes.c_void_p, ctypes.c_uint32]
    kernel32.WaitForSingleObject.restype = ctypes.c_uint32
    kernel32.GetExitCodeProcess.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_uint32)]
    kernel32.GetExitCodeProcess.restype = ctypes.c_int
    kernel32.CloseHandle.argtypes = [ctypes.c_void_p]
    kernel32.CloseHandle.restype = ctypes.c_int

    process_query_limited_info = 0x1000
    synchronize = 0x00100000
    still_active = 259
    handle = kernel32.OpenProcess(process_query_limited_info | synchronize, False, pid)
    if not handle:
        return False
    try:
        if kernel32.WaitForSingleObject(handle, 0) == 0:
            return False
        exit_code = ctypes.c_uint32()
        if kernel32.GetExitCodeProcess(handle, ctypes.byref(exit_code)):
            return exit_code.value == still_active
        return True
    finally:
        kernel32.CloseHandle(handle)


def _force_kill_windows_pid(pid: int) -> None:
    with contextlib.suppress(Exception):
        subprocess.run(
            ["taskkill", "/F", "/T", "/PID", str(pid)],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            check=False,
            **process._no_window(),
        )


def _fake_proc(pid=4242, poll_value=None, wait_exc=None):
    proc = mock.Mock()
    proc.pid = pid
    proc.poll.return_value = poll_value
    if wait_exc is not None:
        proc.wait.side_effect = wait_exc
    else:
        proc.wait.return_value = 0
    return proc


class _RegistryIsolation:
    """Save/restore the module-level tracked registry around each test."""

    def setUp(self):
        self._tracked = process._TRACKED_PROCS
        self._snapshot = set(self._tracked)
        self._tracked.clear()

    def tearDown(self):
        self._tracked.clear()
        self._tracked.update(self._snapshot)


class TestOwnProcessGroup(unittest.TestCase):
    @unittest.skipUnless(sys.platform == "linux", "PDEATHSIG is Linux-only")
    def test_linux_arms_pdeathsig_preexec(self):
        kwargs = _own_process_group()
        self.assertTrue(kwargs["start_new_session"])
        self.assertTrue(callable(kwargs["preexec_fn"]))

    @unittest.skipIf(sys.platform == "linux", "only the non-Linux path")
    def test_non_linux_has_no_preexec(self):
        kwargs = _own_process_group()
        self.assertNotIn("preexec_fn", kwargs)

    @unittest.skipUnless(sys.platform == "linux", "real fork test")
    def test_pdeathsig_kills_orphaned_grandchild(self):
        """Real reproduction of issue #50: parent dies hard, grandchild must die."""
        child_code = (
            "import subprocess, sys, os\n"
            "sys.path.insert(0, sys.argv[1])\n"
            "from process import _own_process_group\n"
            "gc = subprocess.Popen(['sleep', '300'], stdout=subprocess.DEVNULL,"
            " **_own_process_group())\n"
            "print(gc.pid); sys.stdout.flush()\n"
            "os._exit(0)\n"
        )
        parent = subprocess.Popen(
            [sys.executable, "-c", child_code, _PLUGIN_ROOT],
            stdout=subprocess.PIPE,
            text=True,
        )
        gcpid = int(parent.communicate()[0].strip())
        survived = None
        for _ in range(50):
            try:
                os.kill(gcpid, 0)
                survived = True
            except ProcessLookupError:
                survived = False
                break
            time.sleep(0.1)
        self.assertIs(survived, False, f"grandchild {gcpid} survived parent death")

    @unittest.skipUnless(os.name == "nt", "Windows Job Object")
    def test_job_object_kills_orphaned_grandchild_on_windows(self):
        """Real Windows parent dies via os._exit(0): Job Object must kill grandchild."""
        child_code = (
            "import ctypes, os, subprocess, sys\n"
            "sys.path.insert(0, sys.argv[1])\n"
            "import process\n"
            "orig_assign = process._assign_windows_job_object\n"
            "assigned = []\n"
            "def spy_assign(proc):\n"
            "    ok = orig_assign(proc)\n"
            "    assigned.append((ok, ctypes.get_last_error()))\n"
            "    return ok\n"
            "process._assign_windows_job_object = spy_assign\n"
            "gc = subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(60)'],"
            f" creationflags={process._CREATE_NO_WINDOW})\n"
            "process.register_tracked_process(gc)\n"
            "ok, err = assigned[0]\n"
            "k32 = ctypes.WinDLL('kernel32', use_last_error=True)\n"
            "k32.IsProcessInJob.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.POINTER(ctypes.c_int)]\n"
            "k32.IsProcessInJob.restype = ctypes.c_int\n"
            "k32.GetCurrentProcess.restype = ctypes.c_void_p\n"
            "in_job = ctypes.c_int()\n"
            "k32.IsProcessInJob(k32.GetCurrentProcess(), None, ctypes.byref(in_job))\n"
            "print(f'{gc.pid} {int(ok)} {err} {int(bool(in_job.value))}', flush=True)\n"
            "os._exit(0)\n"
        )
        parent = subprocess.Popen(
            [sys.executable, "-c", child_code, _PLUGIN_ROOT],
            stdout=subprocess.PIPE,
            text=True,
            **process._no_window(),
        )
        gcpid = None
        try:
            out, _ = parent.communicate(timeout=10)
            parts = out.strip().split()
            self.assertEqual(len(parts), 4, f"unexpected helper output: {out!r}")
            gcpid, assigned_ok, winerr, in_parent_job = (int(parts[0]), bool(int(parts[1])), int(parts[2]), bool(int(parts[3])))
            if not assigned_ok:
                if in_parent_job and winerr == 5:
                    self.skipTest("parent already inside a nested-job-restricted Windows Job Object")
                self.fail(f"_assign_windows_job_object returned False (winerr={winerr})")

            deadline = time.monotonic() + 5.0
            while time.monotonic() < deadline:
                if not _is_windows_pid_alive(gcpid):
                    break
                time.sleep(0.05)
            self.assertFalse(_is_windows_pid_alive(gcpid), f"grandchild {gcpid} survived parent death")
        finally:
            if parent.poll() is None:
                _force_kill_windows_pid(parent.pid)
                with contextlib.suppress(Exception):
                    parent.wait(timeout=2)
            if gcpid is not None and _is_windows_pid_alive(gcpid):
                _force_kill_windows_pid(gcpid)


class TestTrackedProcessRegistry(_RegistryIsolation, unittest.TestCase):
    def test_register_unregister_roundtrip(self):
        proc = _fake_proc()
        register_tracked_process(proc)
        self.assertIn(proc, self._tracked)
        unregister_tracked_process(proc)
        self.assertNotIn(proc, self._tracked)

    def test_unregister_is_idempotent(self):
        proc = _fake_proc()
        unregister_tracked_process(proc)  # never registered: must not raise
        register_tracked_process(proc)
        unregister_tracked_process(proc)
        unregister_tracked_process(proc)

    def test_exit_hook_terminates_only_live_children(self):
        live = _fake_proc(poll_value=None)
        dead = _fake_proc(poll_value=0)
        register_tracked_process(live)
        register_tracked_process(dead)
        with mock.patch.object(process, "terminate_process") as term:
            _terminate_tracked_processes()
        term.assert_called_once_with(live)
        self.assertEqual(self._tracked, set())

    def test_exit_hook_survives_terminate_failure(self):
        bad = _fake_proc(poll_value=None)
        good = _fake_proc(poll_value=None)
        register_tracked_process(bad)
        register_tracked_process(good)
        calls = []

        def fake_terminate(proc):
            calls.append(proc)
            if proc is bad:
                raise RuntimeError("taskkill exploded")

        with mock.patch.object(process, "terminate_process", side_effect=fake_terminate):
            _terminate_tracked_processes()
        # Set iteration order is arbitrary: both live children must be
        # attempted even though the first one blew up.
        self.assertEqual(sorted(calls, key=id), sorted([bad, good], key=id))


class TestKillReapsZombie(_RegistryIsolation, unittest.TestCase):
    @unittest.skipUnless(sys.platform == "linux", "killpg path is POSIX")
    def test_kill_process_tree_waits_and_unregisters(self):
        proc = _fake_proc(pid=999999)
        register_tracked_process(proc)
        with mock.patch("process.os.killpg") as killpg:
            _kill_process_tree(proc)
        killpg.assert_called_once()
        proc.wait.assert_called_once_with(timeout=2)
        self.assertNotIn(proc, self._tracked)

    @unittest.skipUnless(sys.platform == "linux", "killpg path is POSIX")
    def test_kill_process_tree_wait_failure_still_unregisters(self):
        proc = _fake_proc(pid=999999, wait_exc=subprocess.TimeoutExpired(cmd="agy", timeout=2))
        register_tracked_process(proc)
        with mock.patch("process.os.killpg"):
            _kill_process_tree(proc)
        self.assertNotIn(proc, self._tracked)

    @unittest.skipUnless(sys.platform == "linux", "killpg path is POSIX")
    def test_kill_process_tree_already_dead(self):
        proc = _fake_proc(poll_value=0)
        register_tracked_process(proc)
        _kill_process_tree(proc)
        proc.wait.assert_not_called()
        self.assertNotIn(proc, self._tracked)

    def test_terminate_process_unregisters_after_graceful_exit(self):
        proc = _fake_proc()
        register_tracked_process(proc)
        terminate_process(proc)
        self.assertNotIn(proc, self._tracked)

    def test_terminate_process_unregisters_after_fallback_kill(self):
        proc = _fake_proc(wait_exc=subprocess.TimeoutExpired(cmd="agy", timeout=2))
        register_tracked_process(proc)
        with mock.patch("process._kill_process_tree") as hard:
            terminate_process(proc)
        hard.assert_called_once_with(proc)
        self.assertNotIn(proc, self._tracked)

    @unittest.skipUnless(os.name == "nt", "Windows tree kill")
    def test_terminate_process_kills_windows_descendant_tree(self):
        parent_code = (
            "import subprocess, sys, time\n"
            "gc = subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(60)'],"
            f" creationflags={process._CREATE_NO_WINDOW})\n"
            "print(gc.pid, flush=True)\n"
            "time.sleep(60)\n"
        )
        parent = subprocess.Popen(
            [sys.executable, "-c", parent_code],
            stdout=subprocess.PIPE,
            text=True,
            **_own_process_group(),
        )
        gcpid = None
        try:
            gcpid = int(parent.stdout.readline().strip())
            self.assertTrue(_is_windows_pid_alive(gcpid))
            terminate_process(parent)

            deadline = time.monotonic() + 5.0
            while time.monotonic() < deadline:
                if not _is_windows_pid_alive(parent.pid) and not _is_windows_pid_alive(gcpid):
                    break
                time.sleep(0.05)

            self.assertFalse(_is_windows_pid_alive(parent.pid), f"parent {parent.pid} survived")
            self.assertFalse(_is_windows_pid_alive(gcpid), f"grandchild {gcpid} survived")
        finally:
            if parent.stdout is not None:
                with contextlib.suppress(Exception):
                    parent.stdout.close()
            if parent.poll() is None:
                _force_kill_windows_pid(parent.pid)
                with contextlib.suppress(Exception):
                    parent.wait(timeout=2)
            if gcpid is not None and _is_windows_pid_alive(gcpid):
                _force_kill_windows_pid(gcpid)


if __name__ == "__main__":
    unittest.main()

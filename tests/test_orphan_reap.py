"""Orphaned plugin-worker reaper: Linux /proc sweep and Windows owner-stamp sweep.

A Hermes process that dies without close() leaves its agy worker orphaned
(PPID=1 on Linux; detached tree or un-closed workspace on Windows), holding a
conversation open on the shared Google account. Pins that the sweep:

- Linux (/proc): kills a stream-json agy worker whose HOME sits inside a
  DELETED ``hermes_agy_*`` tempdir; never touches live or non-tempdir workers.
- Windows (``.hermes_owner.json``): kills a creation-timestamp-verified worker
  whose owner PID has exited (or been reused with a different creation time)
  and removes its ``hermes_agy_*`` tempdir; never touches live owners,
  unstamped dirs, or PID-reused unrelated processes.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import tempfile
import time
import unittest
from pathlib import Path
from unittest import mock

plugin_dir = Path(__file__).resolve().parent.parent
if str(plugin_dir) not in sys.path:
    sys.path.insert(0, str(plugin_dir))

from process import (  # noqa: E402
    _no_window,
    get_process_times_100ns,
    reap_orphaned_plugin_workers,
    record_isolated_worker_pid,
    setup_isolated_home,
)

_WORKER_ARGV = [
    "--input-format",
    "stream-json",
    "--disable-slash-commands",
]


class _FakeWorker:
    def __init__(self, home: str):
        env = dict(os.environ)
        env["HOME"] = home
        self.proc = subprocess.Popen(
            [sys.executable, "-c", "import time; time.sleep(30)", *_WORKER_ARGV],
            env=env,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )

    def alive(self) -> bool:
        return self.proc.poll() is None

    def wait_gone(self, seconds: float = 5.0) -> bool:
        deadline = time.time() + seconds
        while time.time() < deadline:
            if not self.alive():
                return True
            time.sleep(0.1)
        self.proc.kill()
        return False


@unittest.skipUnless(sys.platform.startswith("linux"), "orphan sweep is Linux-only (/proc)")
class OrphanReapTests(unittest.TestCase):
    def _mk_marker_home(self) -> str:
        base = tempfile.mkdtemp(prefix="hermes_agy_test_")
        return base

    def setUp(self):
        self._bases: list[str] = []

    def tearDown(self):
        for base in self._bases:
            shutil.rmtree(base, ignore_errors=True)

    def _home_under_marker(self) -> str:
        base = self._mk_marker_home()
        self._bases.append(base)
        home = os.path.join(base, "home")
        os.makedirs(home)  # like setup_isolated_home: the owner dir exists
        return home

    def test_orphan_with_deleted_tempdir_reaped(self):
        base = self._mk_marker_home()
        self._bases.append(base)
        home = os.path.join(base, "home")
        worker = _FakeWorker(home)
        self.assertTrue(worker.alive())
        shutil.rmtree(base, ignore_errors=True)  # owner gone
        reaped = reap_orphaned_plugin_workers()
        self.assertGreaterEqual(reaped, 1)
        self.assertTrue(worker.wait_gone())

    def test_worker_with_live_tempdir_untouched(self):
        home = self._home_under_marker()
        worker = _FakeWorker(home)
        try:
            reap_orphaned_plugin_workers()
            self.assertTrue(worker.alive())
        finally:
            worker.proc.kill()
            worker.proc.wait(timeout=5)

    def test_real_home_worker_untouched(self):
        home = tempfile.mkdtemp(prefix="plain_home_")
        self.addCleanup(shutil.rmtree, home, ignore_errors=True)
        worker = _FakeWorker(home)  # no hermes_agy_ marker in HOME
        try:
            reap_orphaned_plugin_workers()
            self.assertTrue(worker.alive())
        finally:
            worker.proc.kill()
            worker.proc.wait(timeout=5)


@unittest.skipUnless(os.name == "nt", "Windows owner-stamp orphan sweep")
class WindowsOrphanReapTests(unittest.TestCase):
    def setUp(self):
        self._sandbox = tempfile.TemporaryDirectory(prefix="hermes_win_reap_root_")
        self.addCleanup(self._sandbox.cleanup)
        self._pids_to_kill: list[int] = []
        self.addCleanup(self._cleanup_pids)

    def _cleanup_pids(self) -> None:
        for pid in self._pids_to_kill:
            if get_process_times_100ns(pid) is not None:
                subprocess.run(
                    ["taskkill", "/F", "/T", "/PID", str(pid)],
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                    check=False,
                    **_no_window(),
                )

    def test_windows_orphan_with_dead_owner_is_reaped_and_tempdir_removed(self):
        owner_script = (
            "import json, subprocess, sys, tempfile\n"
            "from pathlib import Path\n"
            "sys.path.insert(0, sys.argv[1])\n"
            "from process import _no_window, record_isolated_worker_pid, setup_isolated_home\n"
            "ws = Path(tempfile.mkdtemp(prefix='hermes_agy_orphan_', dir=sys.argv[2]))\n"
            "setup_isolated_home(ws)\n"
            "flags = getattr(subprocess, 'CREATE_NEW_PROCESS_GROUP', 0x00000200) | 0x08000000\n"
            "worker = subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(60)'], "
            "stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, creationflags=flags)\n"
            "record_isolated_worker_pid(ws, worker)\n"
            "print(json.dumps({'ws': str(ws), 'worker_pid': worker.pid}), flush=True)\n"
        )
        owner = subprocess.Popen(
            [sys.executable, "-c", owner_script, str(plugin_dir), self._sandbox.name],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            **_no_window(),
        )
        out, err = owner.communicate(timeout=10)
        self.assertEqual(owner.returncode, 0, f"owner helper failed: {err}")
        info = json.loads(out.strip().splitlines()[-1])
        ws = Path(info["ws"])
        worker_pid = int(info["worker_pid"])
        self._pids_to_kill.append(worker_pid)

        self.assertTrue(ws.is_dir())
        self.assertIsNotNone(get_process_times_100ns(worker_pid))

        with mock.patch("process.tempfile.gettempdir", return_value=self._sandbox.name):
            reaped = reap_orphaned_plugin_workers()

        self.assertEqual(reaped, 1)
        deadline = time.monotonic() + 5.0
        while time.monotonic() < deadline and get_process_times_100ns(worker_pid) is not None:
            time.sleep(0.05)
        self.assertIsNone(get_process_times_100ns(worker_pid))
        self.assertFalse(ws.exists())

    def test_windows_live_owner_and_unstamped_dir_are_untouched(self):
        live_ws = Path(tempfile.mkdtemp(prefix="hermes_agy_live_", dir=self._sandbox.name))
        setup_isolated_home(live_ws)
        worker = subprocess.Popen(
            [sys.executable, "-c", "import time; time.sleep(30)"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            **_no_window(),
        )
        self._pids_to_kill.append(worker.pid)
        record_isolated_worker_pid(live_ws, worker)

        unstamped_ws = Path(tempfile.mkdtemp(prefix="hermes_agy_unstamped_", dir=self._sandbox.name))

        with mock.patch("process.tempfile.gettempdir", return_value=self._sandbox.name):
            reaped = reap_orphaned_plugin_workers()

        self.assertEqual(reaped, 0)
        self.assertIsNone(worker.poll())
        self.assertTrue(live_ws.is_dir())
        self.assertTrue(unstamped_ws.is_dir())

    def test_windows_pid_reuse_mismatched_creation_time_does_not_kill_unrelated_pid(self):
        unrelated = subprocess.Popen(
            [sys.executable, "-c", "import time; time.sleep(30)"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            **_no_window(),
        )
        self._pids_to_kill.append(unrelated.pid)

        stale_ws = Path(tempfile.mkdtemp(prefix="hermes_agy_reused_", dir=self._sandbox.name))
        (stale_ws / ".hermes_owner.json").write_text(
            json.dumps(
                {
                    "owner_pid": unrelated.pid,
                    "owner_creation_100ns": 1,
                    "worker_pid": unrelated.pid,
                    "worker_creation_100ns": 1,
                }
            ),
            encoding="utf-8",
        )

        with mock.patch("process.tempfile.gettempdir", return_value=self._sandbox.name):
            reaped = reap_orphaned_plugin_workers()

        self.assertEqual(reaped, 0)
        self.assertIsNone(unrelated.poll())
        self.assertFalse(stale_ws.exists())


if __name__ == "__main__":
    unittest.main()
